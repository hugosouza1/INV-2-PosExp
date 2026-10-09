"""
pipeline/feature_extraction/testar_resolucao.py
=================================================
Mede como a RESOLUÇÃO do frame de entrada afeta a detecção de mãos do
MediaPipe Holistic, nos mesmos vídeos de um sinalizador, sem reextrair
o dataset. Lê o vídeo na resolução original (local ou URL do Hugging
Face, como o --stream), amostra ~30 frames/s, reduz cada frame pra cada
tamanho de --sizes (INTER_AREA) e roda a config atual (modo IMAGE,
confiança 0.5) em todos.

O tamanho 224x224 aqui é um redimensionamento ESTICADO. Compare a linha
dele com o resultado do comparar_deteccao.py (A_image_0.5 = 31.1% no
Sinalizador10): se ficar parecido, o pipeline provavelmente também
estica; se ficar bem diferente, ele corta/preenche (letterbox) e vale
olhar o video_preprocessor.py.

Uso (PowerShell, na raiz do repo, tudo numa linha):
    python -m pipeline.feature_extraction.testar_resolucao --dataset-dir dataset/minds-libras --signer Sinalizador10 --max-videos 6
"""

from __future__ import annotations

import argparse
import time
from contextlib import ExitStack
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

from pipeline.feature_extraction.build_dataset_features import DEFAULT_REPO_ID, resolve_video_source
from pipeline.feature_extraction.landmark_extractor import DEFAULT_MODEL_PATH, ensure_model_downloaded


def make_landmarker(model_path: Path, conf: float):
    options = vision.HolisticLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.IMAGE,
        min_pose_detection_confidence=conf,
        min_face_detection_confidence=conf,
        min_pose_landmarks_confidence=conf,
        min_hand_landmarks_confidence=conf,
    )
    return vision.HolisticLandmarker.create_from_options(options)


def parse_sizes(text: str) -> list[tuple[int, int]]:
    sizes = []
    for item in text.split(","):
        w, h = item.lower().strip().split("x")
        sizes.append((int(w), int(h)))
    return sizes


def main():
    parser = argparse.ArgumentParser(description="Efeito da resolução na detecção de mãos do MediaPipe.")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--signer", required=True)
    parser.add_argument("--repo", default=DEFAULT_REPO_ID)
    parser.add_argument("--max-videos", type=int, default=6)
    parser.add_argument("--max-frames", type=int, default=150, help="Máx. de frames amostrados por vídeo")
    parser.add_argument("--sizes", default="224x224,640x360,960x540,1280x720", help="Lista LxA separada por vírgula")
    parser.add_argument("--conf", type=float, default=0.5)
    args = parser.parse_args()

    sizes = parse_sizes(args.sizes)
    dataset_dir = Path(args.dataset_dir)
    ann = pd.read_csv(dataset_dir / "annotations.csv")
    rows = ann[ann["user_id"].astype(str) == args.signer]
    if rows.empty:
        raise SystemExit(f"Sinalizador '{args.signer}' não encontrado. Disponíveis: {sorted(ann['user_id'].astype(str).unique())}")
    n = min(args.max_videos, len(rows))
    picks = rows.iloc[np.unique(np.linspace(0, len(rows) - 1, n).round().astype(int))]
    print(f"{len(rows)} vídeos de {args.signer} no CSV; testando {len(picks)}. Tamanhos: {sizes}\n")

    model_path = ensure_model_downloaded(DEFAULT_MODEL_PATH)
    tot = {s: {"frames": 0, "left": 0, "right": 0, "any": 0, "secs": 0.0} for s in sizes}

    for vi, (_, row) in enumerate(picks.iterrows(), start=1):
        video_id = str(row["video_id"])
        filename = video_id if video_id.endswith(".mp4") else f"{video_id}.mp4"
        source = resolve_video_source(dataset_dir, filename, True, args.repo)
        cap = cv2.VideoCapture(str(source))
        if not cap.isOpened():
            print(f"[{vi}/{len(picks)}] {filename}: não consegui abrir; pulando.")
            continue
        fps = float(row["fps"]) if "fps" in row and pd.notna(row["fps"]) else 30.0
        stride = max(1, round(fps / 30.0))
        print(f"[{vi}/{len(picks)}] {filename}: fps={fps:.0f}, amostrando 1 a cada {stride} frames", flush=True)

        kept = 0
        i = 0
        with ExitStack() as stack:
            landmarkers = {s: stack.enter_context(make_landmarker(model_path, args.conf)) for s in sizes}
            while kept < args.max_frames:
                ok, bgr = cap.read()
                if not ok:
                    break
                if i % stride == 0:
                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    for s, lm in landmarkers.items():
                        small = cv2.resize(rgb, s, interpolation=cv2.INTER_AREA)
                        t0 = time.time()
                        res = lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(small)))
                        t = tot[s]
                        t["secs"] += time.time() - t0
                        t["frames"] += 1
                        has_l, has_r = bool(res.left_hand_landmarks), bool(res.right_hand_landmarks)
                        t["left"] += has_l
                        t["right"] += has_r
                        t["any"] += has_l or has_r
                    kept += 1
                i += 1
        cap.release()
        print(f"    {kept} frames amostrados", flush=True)

    print(f"\nResumo ({args.signer}) — % de frames com detecção, por resolução de entrada:\n")
    print(f"{'resolução':<12}{'qualquer mão':>14}{'esquerda':>11}{'direita':>10}{'nenhuma':>10}{'s/frame':>10}")
    for s, t in tot.items():
        f = max(1, t["frames"])
        print(
            f"{s[0]}x{s[1]:<7}{100 * t['any'] / f:>13.1f}%{100 * t['left'] / f:>10.1f}%"
            f"{100 * t['right'] / f:>9.1f}%{100 * (f - t['any']) / f:>9.1f}%{t['secs'] / f:>10.3f}"
        )


if __name__ == "__main__":
    main()