"""
build_dataset_features.py
===========================
Roda as etapas [2] e [3] (pré-processamento + extração de landmarks)
sobre todo o dataset baixado pela etapa [1] (`annotations.csv` +
`videos/`), gerando um `.parquet` por vídeo em `output_dir`, pronto pra
alimentar o treino do classificador (etapa [4]).

Só orquestra os contratos públicos de `preprocessing` e
`feature_extraction` — não depende de nenhum detalhe interno desses
módulos.

Uso:
    python -m libras_pipeline.feature_extraction.build_dataset_features \
        --dataset-dir dataset/minds-libras --output-dir dataset/features
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from pipeline.feature_extraction.landmark_extractor import (
    LandmarkExtractionConfig,
    extract_landmarks,
    save_landmark_sequence,
)
from pipeline.preprocessing.video_preprocessor import PreprocessConfig, extract_frames


def build_dataset_features(
    dataset_dir: str | Path,
    output_dir: str | Path,
    preprocess_config: PreprocessConfig | None = None,
    landmark_config: LandmarkExtractionConfig | None = None,
    overwrite: bool = False,
) -> list[Path]:
    dataset_dir = Path(dataset_dir)
    output_dir = Path(output_dir)
    annotations = pd.read_csv(dataset_dir / "annotations.csv")

    preprocess_config = preprocess_config or PreprocessConfig()
    landmark_config = landmark_config or LandmarkExtractionConfig()

    generated: list[Path] = []
    for _, row in annotations.iterrows():
        video_id = str(row["video_id"])
        class_label = str(row["class"])
        signer_id = str(row["user_id"])

        video_filename = video_id if video_id.endswith(".mp4") else f"{video_id}.mp4"
        video_path = dataset_dir / "videos" / video_filename
        if not video_path.exists():
            continue

        # Idempotente: se o job for interrompido (rede caiu, processo
        # morto etc.), rodar de novo só processa o que falta.
        existing_path = output_dir / f"{Path(video_id).stem}.parquet"
        if existing_path.exists() and not overwrite:
            generated.append(existing_path)
            continue

        frames = extract_frames(video_path, preprocess_config)
        sequence = extract_landmarks(frames, landmark_config)
        out_path = save_landmark_sequence(
            video_id=video_id,
            class_label=class_label,
            signer_id=signer_id,
            sequence=sequence,
            output_dir=output_dir,
        )
        generated.append(out_path)
        print(f"  [{len(generated)}/{len(annotations)}] features salvas: {out_path.name}")

    return generated


def main():
    parser = argparse.ArgumentParser(
        description="Extrai landmarks (pose/mãos/rosto) de todo o dataset MINDS-Libras."
    )
    parser.add_argument("--dataset-dir", required=True, help="Pasta do dataset (ex: dataset/minds-libras)")
    parser.add_argument("--output-dir", required=True, help="Pasta de saída dos .parquet")
    parser.add_argument(
        "--overwrite", action="store_true", help="Reprocessa vídeos que já têm .parquet salvo"
    )
    args = parser.parse_args()

    generated = build_dataset_features(
        Path(args.dataset_dir), Path(args.output_dir), overwrite=args.overwrite
    )
    print(f"\nConcluído. {len(generated)} sequências de features salvas em {args.output_dir}")


if __name__ == "__main__":
    main()
