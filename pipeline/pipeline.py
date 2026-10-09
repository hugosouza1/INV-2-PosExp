"""
pipeline.py
============
Orquestra o pipeline ponta a ponta: lê os frames já tratados da memória
compartilhada (escritos pelo server.cpp) e devolve a anotação estruturada
do sinal reconhecido (etapa [5]).

    shm (frames tratados) -> [3] feature_extraction -> [4] model -> [5] schema

(ANTIGO: vídeo -> [2] preprocessing -> [3] -> [4] -> [5])

Cada etapa só é usada pelo seu contrato público (funções/classes
importadas dos respectivos módulos) — o `pipeline.py` não conhece
detalhes internos de nenhuma delas.

Uso:
    python -m libras_pipeline.pipeline --model model.pt
    # ANTIGO: python -m libras_pipeline.pipeline --video caminho.mp4 --model model.pt
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

# --- ANTIGO: leitura de vídeo por caminho + pré-processamento ---------------
# from pipeline.preprocessing.video_preprocessor import PreprocessConfig, extract_frames
# --- NOVO: leitura dos frames na memória compartilhada ----------------------
from pipeline.preprocessing.shm_reader import FrameReader, ShmConfig, coletar_frames
from pipeline.schema.sign_annotation import SignAnnotation


@dataclass
class LibrasPipelineConfig:
    # ANTIGO:
    # preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    shm: ShmConfig = field(default_factory=ShmConfig)
    landmarks: LandmarkExtractionConfig = field(default_factory=LandmarkExtractionConfig)


class LibrasPipeline:
    """Orquestra as etapas [2]-[5] pra um único sinal (janela de frames da shm)."""

    def __init__(
        self,
        model_path: str | Path,
        config: Optional[LibrasPipelineConfig] = None,
        device: str = "cpu",
    ):
        self.config = config or LibrasPipelineConfig()
        self.predictor = SignPredictor(model_path, device=device)
        self._reader: Optional[FrameReader] = None  # aberto sob demanda (o server pode subir depois)

    def _get_reader(self) -> FrameReader:
        if self._reader is None:
            self._reader = FrameReader(
                self.config.shm.path, ignore_current=self.config.shm.ignore_current
            )
        return self._reader

    def close(self):
        if self._reader is not None:
            self._reader.close()
            self._reader = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # --- ANTIGO: leitura de vídeo por caminho --------------------------------
    # def run(self, video_path: str | Path, sinalizador_id: Optional[str] = None) -> SignAnnotation:
    #     frames = extract_frames(video_path, self.config.preprocess)
    #     sequence = extract_landmarks(frames, self.config.landmarks)
    #     label, confidence = self.predictor.predict(sequence)
    #
    #     fps = self.config.preprocess.target_fps
    #     duration_ms = int(round(len(frames) / fps * 1000)) if fps else 0
    #
    #     return SignAnnotation(
    #         sinal=label,
    #         confianca=confidence,
    #         inicio_ms=0,
    #         fim_ms=duration_ms,
    #         sinalizador_id=sinalizador_id,
    #     )

    # --- NOVO: leitura dos frames na memória compartilhada -------------------
    def run(self, sinalizador_id: Optional[str] = None) -> SignAnnotation:
        reader = self._get_reader()
        # frames já tratados pelo server: sem resample/resize/trim aqui
        frames, stamps = coletar_frames(reader, self.config.shm)
        if not frames:
            raise TimeoutError(
                f"Nenhum frame novo em {self.config.shm.path} "
                f"em {self.config.shm.wait_first_s}s"
            )

        sequence = extract_landmarks(frames, self.config.landmarks)
        label, confidence = self.predictor.predict(sequence)

        # tempos reais vindos do server (ts_ms), relativos ao 1º frame da janela
        duration_ms = int(stamps[-1] - stamps[0])

        return SignAnnotation(
            sinal=label,
            confianca=confidence,
            inicio_ms=0,
            fim_ms=duration_ms,
            sinalizador_id=sinalizador_id,
        )


def main():
    parser = argparse.ArgumentParser(
        description="Roda o pipeline de reconhecimento de sinais em Libras a partir da memória compartilhada."
    )
    # ANTIGO:
    # parser.add_argument("--video", required=True, help="Caminho do vídeo de entrada")
    parser.add_argument("--model", required=True, help="Caminho do checkpoint treinado (.pt)")
    parser.add_argument("--shm-path", default="/dev/shm/frames_cam")
    parser.add_argument("--num-frames", type=int, default=32, help="Frames por janela/sinal")
    parser.add_argument("--sinalizador-id", default=None)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    config = LibrasPipelineConfig()
    config.shm.path = args.shm_path
    config.shm.num_frames = args.num_frames

    with LibrasPipeline(args.model, config=config, device=args.device) as pipeline:
        # ANTIGO: annotation = pipeline.run(args.video, sinalizador_id=args.sinalizador_id)
        annotation = pipeline.run(sinalizador_id=args.sinalizador_id)
        print(annotation.to_json(indent=2))


if __name__ == "__main__":
    main()