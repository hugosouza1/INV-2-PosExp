"""
pipeline/feature_extraction/comparar_deteccao.py
==================================================
Compara configurações de detecção do MediaPipe Holistic num punhado de
vídeos de um sinalizador, SEM reextrair o dataset inteiro. Usa o MESMO
caminho do build_dataset_features.py (annotations.csv -> vídeo local ou
URL do Hugging Face -> extract_frames com PreprocessConfig padrão), então
o MediaPipe recebe exatamente os frames que recebe na extração real.

    A_image_0.5  -> configuração atual do landmark_extractor.py
    B_image_0.3  -> modo IMAGE, limiares de confiança mais baixos
    C_video_0.5  -> modo VIDEO (usa continuidade entre frames)
    D_video_0.3  -> modo VIDEO + limiares mais baixos

Imprime também o TAMANHO dos frames que chegam ao MediaPipe (se forem
pequenos, a mão fica com poucos pixels e a detecção sofre) e, com
--dump-frames, salva frames em que a config ATUAL (A) não achou mão
nenhuma, pra você abrir e conferir a olho.

Uso (PowerShell, na raiz do repo, tudo numa linha):
    python -m pipeline.feature_extraction.comparar_deteccao --dataset-dir dataset/minds-libras --signer Sinalizador10 --max-videos 6 --dump-frames 8
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

from pipeline.feature_extraction.build_dataset_features import DEFAULT_REPO_ID, resolve_video_source
from pipeline.feature_extraction.landmark_extractor import DEFAULT_MODEL_PATH, ensure_model_downloaded
from pipeline.preprocessing.video_preprocessor import PreprocessConfig, extract_frames

CONFIGS: dict[str, tuple[str, float]] = {
    "A_image_0.5": ("image", 0.5),
    "B_image_0.3": ("image", 0.3),
    "C_video_0.5": ("video", 0.5),
    "D_video_0.3": ("video", 0.3),
}


def detect_hands(frames, mode: str, conf: float, model_path: Path, fps: float):
    """Devolve duas listas de bool (mão esquerda / direita detectada, por frame).
    `frames` são RGB, como já saem do extract_frames."""
    options = vision.HolisticLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.VIDEO if mode == "video" else vision.RunningMode.IMAGE,
        min_pose_detection_confidence=conf,
        min_face_detection_confidence=conf,
        min_pose_landmarks_confidence=conf,
        min_hand_landmarks_confidence=conf,
    )
    left, right = [], []
    last_ts = -1
    with vision.HolisticLandmarker.create_from_options(options) as landmarker:
        for i, frame in enumerate(frames):
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(frame))
            if mode == "video":
                ts = max(int(i * 1000 / fps), last_ts + 1)  # timestamps estritamente crescentes
                last_ts = ts
                result = landmarker.detect_for_video(mp_image, ts)
            else:
                result = landmarker.detect(mp_image)
            left.append(bool(result.left_hand_landmarks))
            right.append(bool(result.right_hand_landmarks))
    return left, right


def main():
    parser = argparse.ArgumentParser(description="Compara configurações de detecção de mãos do MediaPipe.")
    parser.add_argument("--dataset-dir", required=True, help="Ex.: dataset/minds-libras (com annotations.csv)")
    parser.add_argument("--signer", required=True, help="Ex.: Sinalizador10")
    parser.add_argument("--repo", default=DEFAULT_REPO_ID)
    parser.add_argument("--max-videos", type=int, default=6, help="Quantos vídeos testar (espalhados entre as classes)")
    parser.add_argument("--dump-frames", type=int, default=0, help="Frames sem mão (config A) a salvar por vídeo")
    parser.add_argument("--dump-dir", default="frames_sem_mao")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    ann = pd.read_csv(dataset_dir / "annotations.csv")
    rows = ann[ann["user_id"].astype(str) == args.signer]
    if rows.empty:
        raise SystemExit(f"Sinalizador '{args.signer}' não encontrado. Disponíveis: {sorted(ann['user_id'].astype(str).unique())}")
    n = min(args.max_videos, len(rows))
    picks = rows.iloc[np.unique(np.linspace(0, len(rows) - 1, n).round().astype(int))]
    print(f"{len(rows)} vídeos de {args.signer} no CSV; testando {len(picks)} espalhados entre as classes.\n")

    pre_cfg = PreprocessConfig()
    fps = next((float(getattr(pre_cfg, k)) for k in ("target_fps", "fps") if getattr(pre_cfg, k, None)), 30.0)
    model_path = ensure_model_downloaded(DEFAULT_MODEL_PATH)
    dump_dir = Path(args.dump_dir)
    if args.dump_frames > 0:
        dump_dir.mkdir(parents=True, exist_ok=True)

    totals = {name: {"frames": 0, "left": 0, "right": 0, "any": 0} for name in CONFIGS}
    shapes = set()

    for vi, (_, row) in enumerate(picks.iterrows(), start=1):
        video_id = str(row["video_id"])
        filename = video_id if video_id.endswith(".mp4") else f"{video_id}.mp4"
        source = resolve_video_source(dataset_dir, filename, True, args.repo)
        try:
            frames = list(extract_frames(source, pre_cfg))
        except (FileNotFoundError, ValueError) as exc:
            print(f"[{vi}/{len(picks)}] {filename}: falha lendo ({exc}); pulando.")
            continue
        shapes.add(tuple(frames[0].shape))
        print(f"[{vi}/{len(picks)}] {filename}: {len(frames)} frames, formato {frames[0].shape}", flush=True)

        for name, (mode, conf) in CONFIGS.items():
            t0 = time.time()
            left, right = detect_hands(frames, mode, conf, model_path, fps)
            left_a, right_a = np.array(left), np.array(right)
            any_a = left_a | right_a
            t = totals[name]
            t["frames"] += len(frames)
            t["left"] += int(left_a.sum())
            t["right"] += int(right_a.sum())
            t["any"] += int(any_a.sum())
            print(f"    {name}: alguma mão em {100 * any_a.mean():.0f}% dos frames ({time.time() - t0:.0f}s)", flush=True)

            if name == "A_image_0.5" and args.dump_frames > 0:
                missing = np.flatnonzero(~any_a)
                if len(missing):
                    take = np.unique(np.linspace(0, len(missing) - 1, min(args.dump_frames, len(missing))).round().astype(int))
                    for k in missing[take]:
                        bgr = cv2.cvtColor(np.ascontiguousarray(frames[int(k)]), cv2.COLOR_RGB2BGR)
                        cv2.imwrite(str(dump_dir / f"{Path(filename).stem}_f{int(k):03d}.jpg"), bgr)

    print(f"\nFrames que chegam ao MediaPipe (altura, largura, canais): {sorted(shapes)}")
    print(f"\nResumo ({args.signer}) — % de frames com detecção:\n")
    print(f"{'config':<14}{'qualquer mão':>14}{'esquerda':>11}{'direita':>10}{'nenhuma':>10}")
    for name, t in totals.items():
        f = max(1, t["frames"])
        print(
            f"{name:<14}{100 * t['any'] / f:>13.1f}%{100 * t['left'] / f:>10.1f}%"
            f"{100 * t['right'] / f:>9.1f}%{100 * (f - t['any']) / f:>9.1f}%"
        )
    if args.dump_frames > 0:
        print(f"\nFrames sem mão (config A) salvos em: {dump_dir.resolve()}")


if __name__ == "__main__":
    main()