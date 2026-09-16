"""
pipeline.py
============
Orquestra o pipeline ponta a ponta: recebe o caminho de um vídeo e
devolve a anotação estruturada do sinal reconhecido (etapa [5]).

    vídeo -> [2] preprocessing -> [3] feature_extraction -> [4] model -> [5] schema

Cada etapa só é usada pelo seu contrato público (funções/classes
importadas dos respectivos módulos) — o `pipeline.py` não conhece
detalhes internos de nenhuma delas.

Uso:
    python -m libras_pipeline.pipeline --video caminho.mp4 --model model.pt
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from pipeline.feature_extraction.landmark_extractor import (
    LandmarkExtractionConfig,
    extract_landmarks,
)
from pipeline.model.predict import SignPredictor
from pipeline.preprocessing.video_preprocessor import PreprocessConfig, extract_frames
from pipeline.schema.sign_annotation import SignAnnotation


@dataclass
class LibrasPipelineConfig:
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    landmarks: LandmarkExtractionConfig = field(default_factory=LandmarkExtractionConfig)


class LibrasPipeline:
    """Orquestra as etapas [2]-[5] pra um único vídeo."""

    def __init__(
        self,
        model_path: str | Path,
        config: Optional[LibrasPipelineConfig] = None,
        device: str = "cpu",
    ):
        self.config = config or LibrasPipelineConfig()
        self.predictor = SignPredictor(model_path, device=device)

    def run(self, video_path: str | Path, sinalizador_id: Optional[str] = None) -> SignAnnotation:
        frames = extract_frames(video_path, self.config.preprocess)
        sequence = extract_landmarks(frames, self.config.landmarks)
        label, confidence = self.predictor.predict(sequence)

        fps = self.config.preprocess.target_fps
        duration_ms = int(round(len(frames) / fps * 1000)) if fps else 0

        return SignAnnotation(
            sinal=label,
            confianca=confidence,
            inicio_ms=0,
            fim_ms=duration_ms,
            sinalizador_id=sinalizador_id,
        )


def main():
    parser = argparse.ArgumentParser(
        description="Roda o pipeline de reconhecimento de sinais em Libras num vídeo."
    )
    parser.add_argument("--video", required=True, help="Caminho do vídeo de entrada")
    parser.add_argument("--model", required=True, help="Caminho do checkpoint treinado (.pt)")
    parser.add_argument("--sinalizador-id", default=None)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    pipeline = LibrasPipeline(args.model, device=args.device)
    annotation = pipeline.run(args.video, sinalizador_id=args.sinalizador_id)
    print(annotation.to_json(indent=2))


if __name__ == "__main__":
    main()
