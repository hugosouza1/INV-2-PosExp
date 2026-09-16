"""
download_minds_libras.py
=========================
Etapa [1] do pipeline (ingestão): baixa o dataset MINDS-Libras (vídeos
brutos + rótulos), IGNORANDO qualquer arquivo de landmarks/pose/
articulações que eventualmente exista no repositório. O objetivo é ter
só:

    dataset/
    ├── annotations.csv   -> rótulo (sinal/significado) de cada vídeo
    └── videos/
        ├── 01AcontecerSinalizador01-1.mp4
        ├── 01AcontecerSinalizador01-2.mp4
        └── ...

Uso:
    python -m libras_pipeline.ingestion.download_minds_libras \
        [--output ./dataset/minds-libras] [--repo ibmectech/minds-libras-raw]

Requisitos:
    pip install huggingface_hub
"""

import argparse
import shutil
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

# Extensões/arquivos que indicam dados JÁ PROCESSADOS (pose, landmarks,
# joints, profundidade etc.) — nunca baixamos isso, mesmo que apareça
# no repositório no futuro.
BLOCKED_KEYWORDS = [
    "landmark", "pose", "joint", "keypoint", "skeleton",
    "depth", "face_points", "holistic", "mediapipe",
]


def is_blocked(filename: str) -> bool:
    lower = filename.lower()
    return any(keyword in lower for keyword in BLOCKED_KEYWORDS)


def download_dataset(repo_id: str, output_dir: Path) -> None:
    api = HfApi()
    all_files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")

    # Mantém apenas: annotations.csv e tudo dentro de videos/*.mp4
    wanted_files = [
        f for f in all_files
        if not is_blocked(f)
        and (f == "annotations.csv" or (f.startswith("videos/") and f.endswith(".mp4")))
    ]

    skipped = [f for f in all_files if f not in wanted_files]

    print(f"Repositório: {repo_id}")
    print(f"Total de arquivos no repo: {len(all_files)}")
    print(f"Arquivos que serão baixados (vídeo + rótulo): {len(wanted_files)}")
    if skipped:
        print(f"Arquivos ignorados (landmarks/pose/outros): {len(skipped)}")
        for f in skipped[:10]:
            print(f"  - ignorado: {f}")
        if len(skipped) > 10:
            print(f"  ... e mais {len(skipped) - 10}")

    output_dir.mkdir(parents=True, exist_ok=True)

    for i, filename in enumerate(wanted_files, 1):
        local_path = hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename=filename,
        )
        dest_path = output_dir / filename
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(local_path, dest_path)
        if i % 50 == 0 or i == len(wanted_files):
            print(f"  [{i}/{len(wanted_files)}] baixado: {filename}")

    print(f"\nConcluído. Dataset salvo em: {output_dir.resolve()}")


def main():
    parser = argparse.ArgumentParser(description="Baixa o MINDS-Libras (só vídeo + rótulo).")
    parser.add_argument(
        "--repo",
        default="ibmectech/minds-libras-raw",
        help="Repositório do dataset no Hugging Face (padrão: ibmectech/minds-libras-raw)",
    )
    parser.add_argument(
        "--output",
        default="./dataset/minds-libras",
        help="Pasta de destino (padrão: ./dataset/minds-libras)",
    )
    args = parser.parse_args()

    download_dataset(repo_id=args.repo, output_dir=Path(args.output))


if __name__ == "__main__":
    main()
