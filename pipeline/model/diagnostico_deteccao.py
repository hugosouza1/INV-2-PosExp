"""
pipeline/model/diagnostico_deteccao.py
=======================================
Mede, por sinalizador, a % de frames em que o MediaPipe NÃO detectou
cada parte do corpo (o extrator preenche com zeros o que não detecta).
Se os sinalizadores com pior acurácia tiverem muito mais frames sem
mão detectada, o gargalo é a extração, não o modelo.

Layout das colunas (landmark_extractor.py, face_mode="reduced"):
    pose 0-131 | mão esquerda 132-194 | mão direita 195-257 | rosto 258-533

Uso (PowerShell, na raiz do repo):
    python -m pipeline.model.diagnostico_deteccao --features-dir dataset/features_v2
"""

from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np

from pipeline.model.dataset import load_all_samples

PARTS = {
    "pose": slice(0, 132),
    "mao_esq": slice(132, 195),
    "mao_dir": slice(195, 258),
    "rosto": slice(258, 534),
}


def main():
    parser = argparse.ArgumentParser(description="Taxa de detecção de landmarks por sinalizador.")
    parser.add_argument("--features-dir", required=True)
    args = parser.parse_args()

    samples = load_all_samples(args.features_dir)

    # signer -> parte -> [frames_ausentes, frames_total]
    counts = defaultdict(lambda: {name: [0, 0] for name in [*PARTS, "nenhuma_mao"]})
    for s in samples:
        seq = s.sequence
        absent = {name: ~np.any(seq[:, sl] != 0, axis=1) for name, sl in PARTS.items()}
        absent["nenhuma_mao"] = absent["mao_esq"] & absent["mao_dir"]
        for name, mask in absent.items():
            counts[s.signer_id][name][0] += int(mask.sum())
            counts[s.signer_id][name][1] += len(mask)

    print("% de frames SEM detecção, por sinalizador:\n")
    header = f"{'sinalizador':<16}" + "".join(f"{n:>13}" for n in [*PARTS, "nenhuma_mao"])
    print(header)
    for signer in sorted(counts):
        row = f"{signer:<16}"
        for name in [*PARTS, "nenhuma_mao"]:
            miss, total = counts[signer][name]
            row += f"{100 * miss / max(1, total):>12.1f}%"
        print(row)


if __name__ == "__main__":
    main()