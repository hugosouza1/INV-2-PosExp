"""
pipeline/model/verificar_modelo.py
====================================
Confere que um checkpoint carrega, aplica o frame_stride gravado nele e
classifica sequências reais. ATENÇÃO: o modelo final foi treinado em 100%
dos sinalizadores, então os acertos aqui NÃO medem generalização (ele já
viu esses vídeos) — serve só pra checar que o pipeline de inferência está
íntegro. A medida honesta de generalização continua sendo a do
leave-one-signer-out (65,4%).

Uso (PowerShell, numa linha):
    python -m pipeline.model.verificar_modelo --model model_v3_e100s2.pt --features-dir dataset/features_v2
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from pipeline.feature_extraction.landmark_extractor import load_landmark_sequence
from pipeline.model.predict import SignPredictor


def main():
    parser = argparse.ArgumentParser(description="Sanidade de um checkpoint treinado.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--features-dir", required=True)
    parser.add_argument("--n", type=int, default=10, help="Quantas amostras testar (espalhadas)")
    args = parser.parse_args()

    files = sorted(Path(args.features_dir).glob("*.parquet"))
    if not files:
        raise SystemExit(f"Nenhum .parquet em {args.features_dir}")
    picks = [files[i] for i in np.unique(np.linspace(0, len(files) - 1, min(args.n, len(files))).round().astype(int))]

    predictor = SignPredictor(args.model)
    print(f"Checkpoint: {args.model} | {len(predictor.label_classes)} classes | frame_stride={predictor.frame_stride}\n")
    hits, confs = 0, []
    for f in picks:
        sequence, video_id, label, _ = load_landmark_sequence(f)
        pred, conf = predictor.predict(sequence)
        hits += pred == label
        confs.append(conf)
        print(f"{video_id}: real={label:<12} previsto={pred:<12} confiança={conf:.2f}  {'OK' if pred == label else 'ERRO'}")
    print(f"\n{hits}/{len(picks)} corretos (amostras vistas no treino) | confiança média {np.mean(confs):.2f}")


if __name__ == "__main__":
    main()