"""
pipeline/model/confusion_matrix.py
====================================
Roda a mesma validação leave-one-signer-out de train.py e monta uma
matriz de confusão agregada (todos os folds), mostrando quais sinais o
modelo mais troca entre si quando testado em sinalizador nunca visto.

Tolerante a interrupções: o resultado de CADA fold é gravado em disco
(--cache-dir) assim que o fold termina. Se o computador reiniciar ou o
script for interrompido, basta rodar o MESMO comando de novo: folds já
concluídos (com os mesmos hiperparâmetros) são pulados.

Uso (PowerShell, na raiz do repo, tudo numa linha):
    python -m pipeline.model.confusion_matrix --features-dir dataset/features_v2 --epochs 80 --output confusion_matrix.png

Opções úteis:
    --augment        usa jitter temporal/espacial no treino (nunca no teste)
    --cache-dir      onde guardar os folds (padrão: confusion_cache)
    --fresh          ignora/apaga o cache e recomeça do zero
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # sem janela; só salva a imagem
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader

from pipeline.model.classifier import ClassifierConfig, SignLSTMClassifier
from pipeline.model.dataset import (
    SignSequenceDataset,
    build_label_mapping,
    collate_padded,
    leave_one_signer_out_splits,
    load_all_samples,
)


def _evaluate(model, loader, device) -> tuple[list[int], list[int]]:
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    with torch.no_grad():
        for sequences, lengths, labels in loader:
            sequences = sequences.to(device)
            preds = model(sequences, lengths).argmax(dim=-1).cpu()
            y_true.extend(labels.tolist())
            y_pred.extend(preds.tolist())
    return y_true, y_pred


def train_fold_and_predict(
    samples, label_to_idx, train_idx, test_idx, epochs, batch_size, lr, device,
    augment=False, eval_every=0,
) -> tuple[list[int], list[int]]:
    """Mesmo treino de train_one_fold, mas devolve (y_true, y_pred)."""
    train_ds = SignSequenceDataset(
        [samples[i] for i in train_idx], label_to_idx, augment=augment
    )
    test_ds = SignSequenceDataset([samples[i] for i in test_idx], label_to_idx)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_padded)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_padded)

    input_dim = samples[0].sequence.shape[1]
    config = ClassifierConfig(input_dim=input_dim, num_classes=len(label_to_idx))
    model = SignLSTMClassifier(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for sequences, lengths, labels in train_loader:
            sequences, labels = sequences.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(sequences, lengths), labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"    época {epoch + 1}/{epochs} loss={epoch_loss / len(train_loader):.4f}", flush=True)
        if eval_every and (epoch + 1) % eval_every == 0 and epoch + 1 < epochs:
            t, p = _evaluate(model, test_loader, device)
            acc = sum(a == b for a, b in zip(t, p)) / max(1, len(t))
            print(f"    >> época {epoch + 1}: acc no sinalizador de teste = {acc:.3f}", flush=True)
            model.train()

    return _evaluate(model, test_loader, device)


# Colunas do vetor de features (ver landmark_extractor.py, face_mode="reduced"):
# pose 0-131, mãos 132-257, rosto 258-533.
HANDS_POSE_COLS = slice(0, 258)


def apply_view(samples, frame_stride: int, features: str) -> None:
    """Subamostra frames (acelera a LSTM) e/ou fatia as features, in-place.
    Se treinar o modelo final com estas opções, a inferência (predict.py)
    precisa aplicar exatamente a mesma transformação."""
    for smp in samples:
        seq = smp.sequence
        if frame_stride > 1:
            seq = seq[::frame_stride]
        if features == "hands_pose":
            seq = seq[:, HANDS_POSE_COLS]
        smp.sequence = seq


# ---------------------------------------------------------------- cache

def _fold_path(cache_dir: Path, signer: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", str(signer))
    return cache_dir / f"fold_{safe}.json"


def _atomic_write_json(path: Path, data: dict) -> None:
    """Grava num .tmp e renomeia — se o PC cair no meio da escrita, o
    arquivo final nunca fica corrompido pela metade."""
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _load_fold(path: Path, params: dict) -> dict | None:
    """Devolve o fold salvo só se existir, estiver íntegro e tiver sido
    gerado com os mesmos hiperparâmetros da rodada atual."""
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None
    if data.get("params") != params:
        return None
    return data


# ----------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser(description="Matriz de confusão agregada (leave-one-signer-out).")
    parser.add_argument("--features-dir", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--augment", action="store_true", help="Augmentation só no treino")
    parser.add_argument("--frame-stride", type=int, default=1,
                        help="Usa 1 a cada N frames (2 = ~2x mais rápido)")
    parser.add_argument("--features", choices=["all", "hands_pose"], default="all",
                        help="hands_pose descarta o rosto (só pose + mãos)")
    parser.add_argument("--signers", nargs="+", default=None,
                        help="Roda só estes sinalizadores (ex.: Sinalizador01 Sinalizador12). Bom pra testes rápidos")
    parser.add_argument("--eval-every", type=int, default=0,
                        help="Imprime a acurácia de teste a cada N épocas (curva de generalização)")
    parser.add_argument("--cache-dir", default="confusion_cache")
    parser.add_argument("--fresh", action="store_true", help="Apaga o cache e recomeça do zero")
    parser.add_argument("--output", default="confusion_matrix.png")
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    if args.fresh and cache_dir.exists():
        shutil.rmtree(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    samples = load_all_samples(args.features_dir)
    apply_view(samples, args.frame_stride, args.features)
    label_to_idx = build_label_mapping(samples)
    idx_to_label = {i: label for label, i in label_to_idx.items()}
    classes = [idx_to_label[i] for i in range(len(idx_to_label))]

    params = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "augment": args.augment,
        "n_samples": len(samples),
        "n_classes": len(classes),
    }

    # Só entram no cache-key quando diferentes do padrão, pra não
    # invalidar resultados antigos (baseline de 20 épocas).
    if args.frame_stride != 1:
        params["frame_stride"] = args.frame_stride
    if args.features != "all":
        params["features"] = args.features

    splits = list(leave_one_signer_out_splits(samples))
    if args.signers:
        splits = [sp for sp in splits if sp[0] in args.signers]
        if not splits:
            raise SystemExit(f"Nenhum sinalizador encontrado entre: {args.signers}")
        print(f"Rodando só {len(splits)} fold(s): {[sp[0] for sp in splits]}")
    all_true: list[int] = []
    all_pred: list[int] = []

    for n, (signer, train_idx, test_idx) in enumerate(splits, start=1):
        path = _fold_path(cache_dir, signer)
        cached = _load_fold(path, params)
        if cached is not None:
            y_true, y_pred = cached["y_true"], cached["y_pred"]
            acc = sum(t == p for t, p in zip(y_true, y_pred)) / max(1, len(y_true))
            print(f"[{n}/{len(splits)}] {signer}: já concluído (cache) acc={acc:.3f} — pulando", flush=True)
        else:
            print(f"[{n}/{len(splits)}] {signer}: treinando...", flush=True)
            t0 = time.time()
            y_true, y_pred = train_fold_and_predict(
                samples, label_to_idx, train_idx, test_idx,
                args.epochs, args.batch_size, args.lr, args.device, args.augment, args.eval_every,
            )
            _atomic_write_json(path, {"signer": signer, "params": params, "y_true": y_true, "y_pred": y_pred})
            acc = sum(t == p for t, p in zip(y_true, y_pred)) / max(1, len(y_true))
            print(
                f"[{n}/{len(splits)}] {signer}: {len(y_true)} amostras, acc={acc:.3f}, "
                f"{(time.time() - t0) / 60:.1f} min — salvo em {path}",
                flush=True,
            )
        all_true.extend(y_true)
        all_pred.extend(y_pred)

    overall = sum(t == p for t, p in zip(all_true, all_pred)) / max(1, len(all_true))
    print(f"\nAcurácia agregada (todos os folds): {overall:.3f}")

    cm = confusion_matrix(all_true, all_pred, labels=list(range(len(classes))))
    side = max(8, len(classes) * 0.5)
    fig, ax = plt.subplots(figsize=(side, side))
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes).plot(
        ax=ax, xticks_rotation=90, cmap="Blues", colorbar=False
    )
    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    print(f"Matriz de confusão salva em {args.output}")

    pairs = [
        (cm[i][j], classes[i], classes[j])
        for i in range(len(classes))
        for j in range(len(classes))
        if i != j and cm[i][j] > 0
    ]
    pairs.sort(reverse=True)
    print("\nTop confusões (rótulo real -> previsto, contagem):")
    for count, real, previsto in pairs[:10]:
        print(f"  {real} -> {previsto}: {count}x")


if __name__ == "__main__":
    main()