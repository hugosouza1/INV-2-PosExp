"""
visualize.py
=============
Utilitário de depuração da etapa [3]: roda a extração de landmarks
sobre um vídeo e desenha pose/mãos/rosto por cima dos frames, salvando
um vídeo anotado. Não faz parte do fluxo de inferência do pipeline —
serve pra inspecionar visualmente se a extração está funcionando bem
num vídeo real.

Uso:
    python -m libras_pipeline.feature_extraction.visualize \
        --video caminho.mp4 --output anotado.mp4
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from pipeline.feature_extraction.landmark_extractor import (
    LandmarkExtractionConfig,
    ensure_model_downloaded,
)
from pipeline.preprocessing.video_preprocessor import PreprocessConfig, extract_frames


def render_landmarks_video(
    video_path: str | Path,
    output_path: str | Path,
    preprocess_config: PreprocessConfig | None = None,
    landmark_config: LandmarkExtractionConfig | None = None,
) -> Path:
    preprocess_config = preprocess_config or PreprocessConfig(target_fps=10.0, target_size=(480, 480))
    landmark_config = landmark_config or LandmarkExtractionConfig()

    frames = extract_frames(video_path, preprocess_config)
    model_path = ensure_model_downloaded(landmark_config.model_path)

    import mediapipe as mp
    from mediapipe.tasks.python import vision
    from mediapipe.tasks.python.core.base_options import BaseOptions

    options = vision.HolisticLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.IMAGE,
    )

    height, width = frames[0].shape[:2]
    output_path = Path(output_path)
    writer = cv2.VideoWriter(
        str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), preprocess_config.target_fps, (width, height)
    )

    with vision.HolisticLandmarker.create_from_options(options) as landmarker:
        for frame in frames:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(frame))
            result = landmarker.detect(mp_image)
            canvas = frame.copy()

            vision.drawing_utils.draw_landmarks(
                canvas, result.pose_landmarks, vision.PoseLandmarksConnections.POSE_LANDMARKS
            )
            vision.drawing_utils.draw_landmarks(
                canvas, result.left_hand_landmarks, vision.HandLandmarksConnections.HAND_CONNECTIONS
            )
            vision.drawing_utils.draw_landmarks(
                canvas, result.right_hand_landmarks, vision.HandLandmarksConnections.HAND_CONNECTIONS
            )
            if landmark_config.include_face:
                vision.drawing_utils.draw_landmarks(canvas, result.face_landmarks, None)

            writer.write(cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))

    writer.release()
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Desenha os landmarks extraídos por cima de um vídeo.")
    parser.add_argument("--video", required=True, help="Vídeo de entrada")
    parser.add_argument("--output", default="landmarks_preview.mp4", help="Vídeo anotado de saída")
    args = parser.parse_args()

    out = render_landmarks_video(args.video, args.output)
    print(f"Vídeo anotado salvo em: {out}")


if __name__ == "__main__":
    main()
