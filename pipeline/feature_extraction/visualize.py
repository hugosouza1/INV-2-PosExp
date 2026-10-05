"""
visualize.py
=============
Verificação visual de keypoints: roda o MediaPipe Holistic Landmarker
sobre um vídeo do dataset, desenha pose/mãos/malha facial por cima dos
frames e calcula as razões de aspecto (Aspect Ratios) do rosto —
EAR/MAR/BAR/NMAR/PUP/CAR/NAR/yaw — usadas para diagnosticar Expressões
Não-Manuais em Libras. Não faz parte do fluxo de inferência do
pipeline (etapas [2]-[5]); serve pra inspecionar se a extração de
keypoints está funcionando bem num vídeo real, cruzando com o rótulo
da classe do sinal (`dataset/minds-libras/annotations.csv`).

Adaptado de APL2-LIBRAS (rotulador.py / testes/teste.py +
testes/calculos_ar.py, ver `calculos_ar.py` deste pacote): o projeto
original roda ao vivo sobre webcam com a API legada
`mp.solutions.holistic`; essa API não existe mais na versão instalada
do mediapipe (só a Tasks API, `HolisticLandmarker`, está disponível),
então aqui a mesma lógica geométrica de diagnóstico roda sobre frames
de vídeo já gravado, usando `HolisticLandmarker` como fonte dos
landmarks.

Uso:
    python -m pipeline.feature_extraction.visualize \
        --video caminho.mp4 --output-dir resultados/
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from pipeline.feature_extraction import calculos_ar as ar
from pipeline.feature_extraction.landmark_extractor import (
    LandmarkExtractionConfig,
    ensure_model_downloaded,
)
from pipeline.preprocessing.video_preprocessor import PreprocessConfig, extract_frames

AR_FIELDS: tuple[str, ...] = (
    "ear_medio",
    "mar",
    "bar_altura",
    "bar_juntas",
    "nmar",
    "pup",
    "car",
    "nar",
    "yaw",
)


@dataclass
class KeypointVerificationResult:
    video_id: str
    class_label: Optional[str]
    n_frames: int
    pose_detection_rate: float
    left_hand_detection_rate: float
    right_hand_detection_rate: float
    face_detection_rate: float
    mean_ar: dict[str, float]
    output_video_path: Path
    output_csv_path: Path


def lookup_class_label(video_path: str | Path, annotations_csv: str | Path) -> Optional[str]:
    """Busca o rótulo (`class`) de um vídeo em annotations.csv a partir
    do nome do arquivo (coluna `video_id` ou `video_name`)."""
    import pandas as pd

    video_name = Path(video_path).name
    df = pd.read_csv(annotations_csv)
    match = df[(df["video_id"] == video_name) | (df["video_name"] == video_name)]
    if match.empty:
        return None
    return str(match.iloc[0]["class"])


def _draw_landmarks(canvas: np.ndarray, result, include_face: bool) -> None:
    from mediapipe.tasks.python import vision
    from mediapipe.tasks.python.vision.drawing_utils import DrawingSpec

    # Contorno facial fino (espessura/raio 1, verde-oliva), igual ao estilo
    # do rotulador.py original. Usa só os contornos (olhos, sobrancelhas,
    # boca, oval do rosto) em vez da tesselação completa (478 pontos): a
    # tesselação cheia, no tamanho de rosto típico de um vídeo do dataset,
    # vira uma mancha sólida em vez de malha legível — e contornos são
    # justamente a região que alimenta os Aspect Ratios calculados abaixo.
    face_spec = DrawingSpec(color=(10, 110, 80), thickness=1, circle_radius=1)
    hand_spec = DrawingSpec(color=(10, 110, 80), thickness=1, circle_radius=1)

    vision.drawing_utils.draw_landmarks(
        canvas, result.pose_landmarks, vision.PoseLandmarksConnections.POSE_LANDMARKS
    )
    vision.drawing_utils.draw_landmarks(
        canvas,
        result.left_hand_landmarks,
        vision.HandLandmarksConnections.HAND_CONNECTIONS,
        landmark_drawing_spec=hand_spec,
        connection_drawing_spec=hand_spec,
    )
    vision.drawing_utils.draw_landmarks(
        canvas,
        result.right_hand_landmarks,
        vision.HandLandmarksConnections.HAND_CONNECTIONS,
        landmark_drawing_spec=hand_spec,
        connection_drawing_spec=hand_spec,
    )
    if include_face:
        vision.drawing_utils.draw_landmarks(
            canvas,
            result.face_landmarks,
            vision.FaceLandmarksConnections.FACE_LANDMARKS_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=face_spec,
        )


def _overlay_text(canvas: np.ndarray, lines: list[str]) -> None:
    for i, line in enumerate(lines):
        y = 22 + i * 18
        cv2.putText(canvas, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(canvas, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)


def verify_keypoints(
    video_path: str | Path,
    output_dir: str | Path,
    preprocess_config: PreprocessConfig | None = None,
    landmark_config: LandmarkExtractionConfig | None = None,
    annotations_csv: str | Path | None = None,
) -> KeypointVerificationResult:
    """Roda a extração de keypoints sobre um vídeo e gera, em
    `output_dir`: um .mp4 anotado (landmarks + Aspect Ratios + rótulo
    desenhados) e um .csv por frame com flags de detecção e os
    Aspect Ratios faciais, pra diagnosticar a qualidade dos keypoints
    extraídos frente ao rótulo real do sinal."""
    preprocess_config = preprocess_config or PreprocessConfig(target_fps=12.0, target_size=(640, 640))
    landmark_config = landmark_config or LandmarkExtractionConfig()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    video_id = Path(video_path).name
    class_label = lookup_class_label(video_path, annotations_csv) if annotations_csv else None

    frames = extract_frames(video_path, preprocess_config)
    model_path = ensure_model_downloaded(landmark_config.model_path)

    import mediapipe as mp
    from mediapipe.tasks.python import vision
    from mediapipe.tasks.python.core.base_options import BaseOptions

    options = vision.HolisticLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.IMAGE,
        min_pose_detection_confidence=landmark_config.min_detection_confidence,
        min_face_detection_confidence=landmark_config.min_detection_confidence,
        min_pose_landmarks_confidence=landmark_config.min_landmarks_confidence,
        min_hand_landmarks_confidence=landmark_config.min_landmarks_confidence,
    )

    height, width = frames[0].shape[:2]
    safe_name = Path(video_id).stem
    output_video_path = output_dir / f"{safe_name}_keypoints.mp4"
    output_csv_path = output_dir / f"{safe_name}_keypoints.csv"

    writer = cv2.VideoWriter(
        str(output_video_path), cv2.VideoWriter_fourcc(*"mp4v"), preprocess_config.target_fps, (width, height)
    )

    pose_hits = left_hand_hits = right_hand_hits = face_hits = 0
    ar_sums = {field: 0.0 for field in AR_FIELDS}
    ar_hits = 0
    csv_rows: list[dict[str, object]] = []

    with vision.HolisticLandmarker.create_from_options(options) as landmarker:
        for frame_idx, frame in enumerate(frames):
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(frame))
            result = landmarker.detect(mp_image)

            has_pose = bool(result.pose_landmarks)
            has_left_hand = bool(result.left_hand_landmarks)
            has_right_hand = bool(result.right_hand_landmarks)
            has_face = bool(result.face_landmarks)
            pose_hits += has_pose
            left_hand_hits += has_left_hand
            right_hand_hits += has_right_hand
            face_hits += has_face

            row: dict[str, object] = {
                "frame_idx": frame_idx,
                "pose_detected": has_pose,
                "left_hand_detected": has_left_hand,
                "right_hand_detected": has_right_hand,
                "face_detected": has_face,
            }

            canvas = frame.copy()
            _draw_landmarks(canvas, result, landmark_config.include_face)

            overlay_lines = [f"video: {video_id}"]
            if class_label is not None:
                overlay_lines.append(f"classe (rotulo): {class_label}")

            if has_face:
                ratios = ar.calcular_todos_ratios(result.face_landmarks, width, height)
                for field in AR_FIELDS:
                    row[field] = ratios[field]
                    ar_sums[field] += ratios[field]
                ar_hits += 1
                overlay_lines.append(
                    f"EAR {ratios['ear_medio']:.2f} MAR {ratios['mar']:.2f} "
                    f"BARa {ratios['bar_altura']:.2f} BARj {ratios['bar_juntas']:.2f}"
                )
                overlay_lines.append(
                    f"NMAR {ratios['nmar']:.2f} PUP {ratios['pup']:.2f} "
                    f"CAR {ratios['car']:.2f} NAR {ratios['nar']:.2f} YAW {ratios['yaw']:.2f}"
                )
            else:
                for field in AR_FIELDS:
                    row[field] = None
                overlay_lines.append("rosto nao detectado")

            _overlay_text(canvas, overlay_lines)
            writer.write(cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
            csv_rows.append(row)

    writer.release()

    n_frames = len(frames)
    with open(output_csv_path, mode="w", newline="", encoding="utf-8") as f:
        fieldnames = ["frame_idx", "pose_detected", "left_hand_detected", "right_hand_detected", "face_detected", *AR_FIELDS]
        csv_writer = csv.DictWriter(f, fieldnames=fieldnames)
        csv_writer.writeheader()
        csv_writer.writerows(csv_rows)

    mean_ar = {field: (ar_sums[field] / ar_hits if ar_hits else 0.0) for field in AR_FIELDS}

    return KeypointVerificationResult(
        video_id=video_id,
        class_label=class_label,
        n_frames=n_frames,
        pose_detection_rate=pose_hits / n_frames,
        left_hand_detection_rate=left_hand_hits / n_frames,
        right_hand_detection_rate=right_hand_hits / n_frames,
        face_detection_rate=face_hits / n_frames,
        mean_ar=mean_ar,
        output_video_path=output_video_path,
        output_csv_path=output_csv_path,
    )


def render_landmarks_video(
    video_path: str | Path,
    output_path: str | Path,
    preprocess_config: PreprocessConfig | None = None,
    landmark_config: LandmarkExtractionConfig | None = None,
) -> Path:
    """Mantido por compatibilidade: só desenha os landmarks, sem
    calcular Aspect Ratios nem CSV de diagnóstico (ver `verify_keypoints`)."""
    result = verify_keypoints(
        video_path,
        output_dir=Path(output_path).parent,
        preprocess_config=preprocess_config,
        landmark_config=landmark_config,
    )
    result.output_video_path.replace(output_path)
    result.output_csv_path.unlink(missing_ok=True)
    return Path(output_path)


def main():
    parser = argparse.ArgumentParser(description="Verifica os keypoints extraídos de um vídeo.")
    parser.add_argument("--video", required=True, help="Vídeo de entrada")
    parser.add_argument("--output-dir", default="keypoint_verification", help="Diretório de saída")
    parser.add_argument(
        "--annotations",
        default=None,
        help="CSV de anotações do dataset (ex: dataset/minds-libras/annotations.csv), pra incluir o rótulo da classe",
    )
    args = parser.parse_args()

    result = verify_keypoints(args.video, args.output_dir, annotations_csv=args.annotations)
    print(f"Vídeo: {result.video_id} | classe (rótulo): {result.class_label}")
    print(f"Frames: {result.n_frames}")
    print(f"Detecção pose={result.pose_detection_rate:.0%} mao_esq={result.left_hand_detection_rate:.0%} "
          f"mao_dir={result.right_hand_detection_rate:.0%} rosto={result.face_detection_rate:.0%}")
    print(f"Vídeo anotado salvo em: {result.output_video_path}")
    print(f"CSV de diagnóstico salvo em: {result.output_csv_path}")


if __name__ == "__main__":
    main()
