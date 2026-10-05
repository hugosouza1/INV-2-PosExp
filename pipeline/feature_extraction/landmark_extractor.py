"""
landmark_extractor.py
======================
Etapa [3] do pipeline: roda o MediaPipe Holistic Landmarker frame a
frame e gera uma sequência de vetores de features por vídeo
(coordenadas normalizadas de pose corporal, mãos e rosto).

Usa a Tasks API do MediaPipe (`mediapipe.tasks.python.vision`), que é a
API disponível no mediapipe>=0.10 — a antiga `mp.solutions.holistic`
não existe mais nas versões atuais do pacote. Essa API precisa de um
modelo `.task` local; `ensure_model_downloaded` baixa o modelo oficial
do Google (holistic_landmarker, float16) na primeira execução e o
mantém em cache em `feature_extraction/models/`.

Contrato de entrada/saída:
    extract_landmarks(frames, config) -> np.ndarray[T, feature_dim]

Cada linha do array é o vetor de landmarks de um frame; componentes não
detectados (pose/mão/rosto ausente no frame) ficam preenchidos com
zeros nessa parte do vetor, preservando o comprimento T da sequência.
"""

from __future__ import annotations

import urllib.request
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

import numpy as np
import pandas as pd

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/holistic_landmarker/"
    "holistic_landmarker/float16/1/holistic_landmarker.task"
)
DEFAULT_MODEL_DIR = Path(__file__).resolve().parent / "models"
DEFAULT_MODEL_PATH = DEFAULT_MODEL_DIR / "holistic_landmarker.task"

# Contagens de landmarks do modelo Holistic Landmarker do MediaPipe.
NUM_POSE_LANDMARKS = 33
NUM_HAND_LANDMARKS = 21
NUM_FACE_LANDMARKS_FULL = 478

POSE_DIM = NUM_POSE_LANDMARKS * 4  # x, y, z, visibility
HAND_DIM = NUM_HAND_LANDMARKS * 3  # x, y, z
FACE_DIM_FULL = NUM_FACE_LANDMARKS_FULL * 3  # x, y, z

# Subconjunto semântico da malha facial (sobrancelhas + olhos + boca), a
# união dos índices de FaceLandmarksConnections.FACE_LANDMARKS_{LIPS,
# LEFT_EYE, RIGHT_EYE, LEFT_EYEBROW, RIGHT_EYEBROW} do próprio MediaPipe.
# Motivo pra não usar os 478 pontos da malha densa: em Libras a face só
# carrega informação linguística nos marcadores não-manuais (sobrancelha
# levantada em pergunta, boca acompanhando o sinal, olhos semicerrados
# etc.) — o resto da malha (bochecha, testa, contorno do rosto) é forma/
# identidade do rosto, não expressão, e pipelines de reconhecimento de
# expressão facial (ex: os 68 pontos clássicos do dlib) também usam só
# essas regiões. Manter os 478 pontos infla o vetor de features com
# ruído de alta dimensão (~85% do vetor total) sem ganho de sinal, o que
# é especialmente ruim treinando com poucos vídeos por classe.
FACE_SEMANTIC_LANDMARK_INDICES: tuple[int, ...] = (
    0, 7, 13, 14, 17, 33, 37, 39, 40, 46, 52, 53, 55, 61, 63, 65, 66, 70,
    78, 80, 81, 82, 84, 87, 88, 91, 95, 105, 107, 133, 144, 145, 146, 153,
    154, 155, 157, 158, 159, 160, 161, 163, 173, 178, 181, 185, 191, 246,
    249, 263, 267, 269, 270, 276, 282, 283, 285, 291, 293, 295, 296, 300,
    308, 310, 311, 312, 314, 317, 318, 321, 324, 334, 336, 362, 373, 374,
    375, 380, 381, 382, 384, 385, 386, 387, 388, 390, 398, 402, 405, 409,
    415, 466,
)
NUM_FACE_LANDMARKS_REDUCED = len(FACE_SEMANTIC_LANDMARK_INDICES)
FACE_DIM_REDUCED = NUM_FACE_LANDMARKS_REDUCED * 3  # x, y, z

# Índices dos ombros no modelo de pose do MediaPipe (iguais entre a API
# legada e a Tasks API — `vision.PoseLandmark.LEFT_SHOULDER`/`RIGHT_SHOULDER`).
# Usados como referência pra normalização corporal dos landmarks.
LEFT_SHOULDER_IDX = 11
RIGHT_SHOULDER_IDX = 12


@dataclass
class LandmarkExtractionConfig:
    model_path: str | Path = DEFAULT_MODEL_PATH
    include_face: bool = True
    # "reduced" (padrão): só sobrancelhas/olhos/boca (92 pontos, ver
    # FACE_SEMANTIC_LANDMARK_INDICES acima). "full": os 478 pontos da
    # malha densa do MediaPipe.
    face_mode: str = "reduced"
    min_detection_confidence: float = 0.5
    min_landmarks_confidence: float = 0.5
    # Centraliza todos os landmarks (pose/mãos/rosto) no ponto médio dos
    # ombros e escala pela largura dos ombros, antes de montar o vetor de
    # features. Sem isso, as coordenadas são absolutas em tela — o modelo
    # aprende a distância/posição do sinalizador em relação à câmera e o
    # porte físico dele, em vez do movimento do sinal, o que generaliza mal
    # pra sinalizadores novos (ver leave-one-signer-out em `model/train.py`).
    normalize_landmarks: bool = True

    @property
    def face_dim(self) -> int:
        if not self.include_face:
            return 0
        return FACE_DIM_REDUCED if self.face_mode == "reduced" else FACE_DIM_FULL

    @property
    def feature_dim(self) -> int:
        return POSE_DIM + 2 * HAND_DIM + self.face_dim


def ensure_model_downloaded(model_path: str | Path = DEFAULT_MODEL_PATH) -> Path:
    """Garante que o modelo `.task` do Holistic Landmarker está em disco,
    baixando-o do repositório oficial do MediaPipe se necessário."""
    model_path = Path(model_path)
    if not model_path.exists():
        model_path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(MODEL_URL, model_path)
    return model_path


def _flatten(landmarks: Sequence[Any] | None, num_points: int, include_visibility: bool) -> np.ndarray:
    """Achata uma lista de landmarks pra um vetor de tamanho fixo
    (num_points * (4 se include_visibility else 3)), preenchendo com
    zeros pontos ausentes (landmarks=None/vazio ou lista mais curta)."""
    per_point = 4 if include_visibility else 3
    vec = np.zeros(num_points * per_point, dtype=np.float32)
    for i, lm in enumerate(landmarks or []):
        if i >= num_points:
            break
        base = i * per_point
        vec[base] = lm.x or 0.0
        vec[base + 1] = lm.y or 0.0
        vec[base + 2] = lm.z or 0.0
        if include_visibility:
            vec[base + 3] = lm.visibility or 0.0
    return vec


def _flatten_by_indices(
    landmarks: Sequence[Any] | None, indices: Sequence[int], include_visibility: bool
) -> np.ndarray:
    """Como `_flatten`, mas em vez dos primeiros `num_points` pontos na
    ordem em que vêm, extrai só os índices específicos passados (usado
    pro subconjunto semântico de landmarks faciais)."""
    per_point = 4 if include_visibility else 3
    vec = np.zeros(len(indices) * per_point, dtype=np.float32)
    if not landmarks:
        return vec
    n_available = len(landmarks)
    for out_i, idx in enumerate(indices):
        if idx >= n_available:
            continue
        lm = landmarks[idx]
        base = out_i * per_point
        vec[base] = lm.x or 0.0
        vec[base + 1] = lm.y or 0.0
        vec[base + 2] = lm.z or 0.0
        if include_visibility:
            vec[base + 3] = lm.visibility or 0.0
    return vec


def _body_center_scale(pose_landmarks: Sequence[Any] | None) -> tuple[tuple[float, float, float], float]:
    """Centro (ponto médio dos ombros) e escala (largura dos ombros) pra
    normalizar os landmarks num referencial centrado no corpo do
    sinalizador. Sem ombros detectados, usa um referencial neutro
    (centro da imagem, escala 1) — os landmarks ficam como vieram."""
    if not pose_landmarks or len(pose_landmarks) <= RIGHT_SHOULDER_IDX:
        return (0.5, 0.5, 0.0), 1.0

    left = pose_landmarks[LEFT_SHOULDER_IDX]
    right = pose_landmarks[RIGHT_SHOULDER_IDX]
    center = ((left.x + right.x) / 2.0, (left.y + right.y) / 2.0, (left.z + right.z) / 2.0)
    scale = float(np.hypot(left.x - right.x, left.y - right.y))
    if scale < 1e-6:
        scale = 1.0
    return center, scale


def _normalize_points(
    landmarks: Sequence[Any] | None, center: tuple[float, float, float], scale: float
) -> Sequence[Any] | None:
    """Reexpressa cada landmark em `(x,y,z)` relativo a `center`, escalado
    por `scale`; preserva `visibility` quando presente. Lista vazia/None
    passa direto (`_flatten`/`_flatten_by_indices` já tratam isso como
    "nada detectado" e preenchem zeros)."""
    if not landmarks:
        return landmarks
    cx, cy, cz = center
    return [
        SimpleNamespace(
            x=(lm.x - cx) / scale,
            y=(lm.y - cy) / scale,
            z=(lm.z - cz) / scale,
            visibility=getattr(lm, "visibility", None),
        )
        for lm in landmarks
    ]


def result_to_feature_vector(result: Any, config: LandmarkExtractionConfig) -> np.ndarray:
    """Converte um `HolisticLandmarkerResult` (ou qualquer objeto com os
    mesmos atributos) num único vetor de features pro frame."""
    pose_landmarks = result.pose_landmarks
    left_hand_landmarks = result.left_hand_landmarks
    right_hand_landmarks = result.right_hand_landmarks
    face_landmarks = result.face_landmarks

    if config.normalize_landmarks:
        center, scale = _body_center_scale(pose_landmarks)
        pose_landmarks = _normalize_points(pose_landmarks, center, scale)
        left_hand_landmarks = _normalize_points(left_hand_landmarks, center, scale)
        right_hand_landmarks = _normalize_points(right_hand_landmarks, center, scale)
        face_landmarks = _normalize_points(face_landmarks, center, scale)

    parts = [
        _flatten(pose_landmarks, NUM_POSE_LANDMARKS, include_visibility=True),
        _flatten(left_hand_landmarks, NUM_HAND_LANDMARKS, include_visibility=False),
        _flatten(right_hand_landmarks, NUM_HAND_LANDMARKS, include_visibility=False),
    ]
    if config.include_face:
        if config.face_mode == "reduced":
            parts.append(
                _flatten_by_indices(face_landmarks, FACE_SEMANTIC_LANDMARK_INDICES, include_visibility=False)
            )
        else:
            parts.append(_flatten(face_landmarks, NUM_FACE_LANDMARKS_FULL, include_visibility=False))
    return np.concatenate(parts)


def extract_landmarks(
    frames: list[np.ndarray], config: LandmarkExtractionConfig | None = None
) -> np.ndarray:
    """Contrato público da etapa [3]: frames RGB -> sequência de landmarks."""
    config = config or LandmarkExtractionConfig()
    model_path = ensure_model_downloaded(config.model_path)

    import mediapipe as mp
    from mediapipe.tasks.python import vision
    from mediapipe.tasks.python.core.base_options import BaseOptions

    options = vision.HolisticLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.IMAGE,
        min_pose_detection_confidence=config.min_detection_confidence,
        min_face_detection_confidence=config.min_detection_confidence,
        min_pose_landmarks_confidence=config.min_landmarks_confidence,
        min_hand_landmarks_confidence=config.min_landmarks_confidence,
    )

    sequence = np.zeros((len(frames), config.feature_dim), dtype=np.float32)

    with vision.HolisticLandmarker.create_from_options(options) as landmarker:
        for i, frame in enumerate(frames):
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(frame))
            result = landmarker.detect(mp_image)
            sequence[i] = result_to_feature_vector(result, config)

    return sequence


def save_landmark_sequence(
    video_id: str,
    class_label: str,
    signer_id: str,
    sequence: np.ndarray,
    output_dir: str | Path,
) -> Path:
    """Persiste a sequência de landmarks de um vídeo em parquet, associada
    ao `video_id` e sem perder o rótulo (`class`) nem o sinalizador."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(sequence, columns=[f"feat_{i}" for i in range(sequence.shape[1])])
    df.insert(0, "frame_idx", np.arange(sequence.shape[0]))
    df.insert(0, "signer_id", signer_id)
    df.insert(0, "class", class_label)
    df.insert(0, "video_id", video_id)

    safe_name = Path(video_id).stem
    out_path = output_dir / f"{safe_name}.parquet"
    df.to_parquet(out_path, index=False)
    return out_path


def load_landmark_sequence(path: str | Path) -> tuple[np.ndarray, str, str, str]:
    """Lê de volta (sequence, video_id, class_label, signer_id) de um
    parquet gerado por `save_landmark_sequence`."""
    df = pd.read_parquet(path)
    df = df.sort_values("frame_idx")

    video_id = str(df["video_id"].iloc[0])
    class_label = str(df["class"].iloc[0])
    signer_id = str(df["signer_id"].iloc[0])

    feature_cols = [c for c in df.columns if c.startswith("feat_")]
    sequence = df[feature_cols].to_numpy(dtype=np.float32)
    return sequence, video_id, class_label, signer_id
