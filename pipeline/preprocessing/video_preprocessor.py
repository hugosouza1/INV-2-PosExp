"""
video_preprocessor.py
======================
Etapa [2] do pipeline: lê um vídeo bruto e devolve uma lista de frames
(RGB, uint8) padronizados em fps e resolução, com corte opcional dos
trechos parados no início/fim (baseado em intensidade de movimento).

Contrato de entrada/saída:
    extract_frames(video_path, config) -> list[np.ndarray[H, W, 3]]

Esse é o único ponto que a etapa [3] (extração de landmarks) precisa
conhecer deste módulo.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class PreprocessConfig:
    target_fps: float = 25.0
    target_size: tuple[int, int] = (224, 224)  # (largura, altura)
    trim_by_motion: bool = False
    motion_threshold: float = 2.0  # diferença média de intensidade entre frames consecutivos
    min_motion_frames: int = 1  # nº mínimo de frames mantidos mesmo sem detectar movimento


def read_video_frames(video_path: str | Path) -> tuple[list[np.ndarray], float]:
    """Lê todos os frames (RGB) de um vídeo e devolve (frames, fps_original)."""
    video_path = str(video_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Não foi possível abrir o vídeo: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frames: list[np.ndarray] = []
    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    finally:
        cap.release()

    if not frames:
        raise ValueError(f"Vídeo sem frames legíveis: {video_path}")

    return frames, fps


def resample_fps(
    frames: list[np.ndarray], original_fps: float, target_fps: float
) -> list[np.ndarray]:
    """Reamostra a sequência de frames pra um fps alvo (amostragem uniforme)."""
    if original_fps <= 0 or target_fps <= 0 or original_fps == target_fps:
        return frames

    duration_s = len(frames) / original_fps
    n_target = max(1, round(duration_s * target_fps))
    if n_target >= len(frames):
        return frames

    indices = np.linspace(0, len(frames) - 1, n_target)
    return [frames[int(round(i))] for i in indices]


def resize_frames(frames: list[np.ndarray], size: tuple[int, int]) -> list[np.ndarray]:
    """Redimensiona todos os frames pra uma resolução comum (largura, altura)."""
    return [cv2.resize(frame, size, interpolation=cv2.INTER_AREA) for frame in frames]


def trim_by_motion(
    frames: list[np.ndarray], threshold: float, min_frames: int
) -> list[np.ndarray]:
    """Recorta o início/fim parado, mantendo só do primeiro ao último frame
    em que a diferença pra o frame anterior ultrapassa `threshold`."""
    if len(frames) <= min_frames:
        return frames

    grays = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY).astype(np.float32) for f in frames]
    diffs = [0.0] + [
        float(np.abs(grays[i] - grays[i - 1]).mean()) for i in range(1, len(grays))
    ]

    moving = [i for i, d in enumerate(diffs) if d >= threshold]
    if not moving:
        return frames

    start, end = moving[0], moving[-1]
    trimmed = frames[start : end + 1]
    return trimmed if len(trimmed) >= min_frames else frames


def extract_frames(
    video_path: str | Path, config: PreprocessConfig | None = None
) -> list[np.ndarray]:
    """Contrato público da etapa [2]: vídeo bruto -> frames padronizados."""
    config = config or PreprocessConfig()

    frames, original_fps = read_video_frames(video_path)
    if config.trim_by_motion:
        frames = trim_by_motion(frames, config.motion_threshold, config.min_motion_frames)
    frames = resample_fps(frames, original_fps, config.target_fps)
    frames = resize_frames(frames, config.target_size)
    return frames
