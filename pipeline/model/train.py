"""
train.py
=========
Etapa [4] do pipeline (treino): treina o classificador de sinais sobre
as sequências de landmarks extraídas pela etapa [3], validando com
leave-one-signer-out (um fold por sinalizador, testado só nos vídeos
dele e treinado em todos os outros).

Uso:
    python -m libras_pipeline.model.train \
        --features-dir dataset/features --output model.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from pipeline.model.classifier import ClassifierConfig, SignLSTMClassifier, save_checkpoint
from pipeline.model.dataset import (
    SignSample,
    SignSequenceDataset,
    build_label_mapping,
    collate_padded,
    leave_one_signer_out_splits,
    load_all_samples,
)


def train_one_fold(
    samples: list[SignSample],
    label_to_idx: dict[str, int],
    train_idx: list[int],
    test_idx: list[int],
    epochs: int = 20,
    batch_size: int = 8,
    lr: float = 1e-3,
    device: str = "cpu",
) -> tuple[SignLSTMClassifier, float]:
    train_ds = SignSequenceDataset([samples[i] for i in train_idx], label_to_idx)
    test_ds = SignSequenceDataset([samples[i] for i in test_idx], label_to_idx)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_padded)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_padded)

    input_dim = samples[0].sequence.shape[1]
    config = ClassifierConfig(input_dim=input_dim, num_classes=len(label_to_idx))
    model = SignLSTMClassifier(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for _ in range(epochs):
        for sequences, lengths, labels in train_loader:
            sequences, labels = sequences.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(sequences, lengths)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for sequences, lengths, labels in test_loader:
            sequences, labels = sequences.to(device), labels.to(device)
            preds = model(sequences, lengths).argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.numel()
    accuracy = correct / total if total else 0.0
    return model, accuracy


def train_leave_one_signer_out(
    features_dir: str | Path,
    epochs: int = 20,
    batch_size: int = 8,
    lr: float = 1e-3,
    device: str = "cpu",
) -> tuple[SignLSTMClassifier, dict[str, int], dict[str, float]]:
    samples = load_all_samples(features_dir)
    label_to_idx = build_label_mapping(samples)

    fold_accuracies: dict[str, float] = {}
    best_model, best_accuracy = None, -1.0
    for signer, train_idx, test_idx in leave_one_signer_out_splits(samples):
        model, accuracy = train_one_fold(
            samples, label_to_idx, train_idx, test_idx, epochs, batch_size, lr, device
        )
        fold_accuracies[signer] = accuracy
        print(f"[leave-one-signer-out] sinalizador de teste={signer} acc={accuracy:.3f}")
        if accuracy > best_accuracy:
            best_model, best_accuracy = model, accuracy

    mean_accuracy = sum(fold_accuracies.values()) / len(fold_accuracies)
    print(f"Acurácia média (leave-one-signer-out): {mean_accuracy:.3f}")
    return best_model, label_to_idx, fold_accuracies


def main():
    parser = argparse.ArgumentParser(description="Treina o classificador de sinais em Libras.")
    parser.add_argument("--features-dir", required=True, help="Pasta com os .parquet gerados pela etapa [3]")
    parser.add_argument("--output", default="model.pt", help="Caminho de saída do checkpoint treinado")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    best_model, label_to_idx, _ = train_leave_one_signer_out(
        args.features_dir, args.epochs, args.batch_size, args.lr, args.device
    )
    idx_to_label = {i: label for label, i in label_to_idx.items()}
    label_classes = [idx_to_label[i] for i in range(len(idx_to_label))]
    save_checkpoint(best_model, label_classes, args.output)
    print(f"Modelo salvo em {args.output}")


if __name__ == "__main__":
    main()
