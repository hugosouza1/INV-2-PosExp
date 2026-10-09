"""
train.py
=========
Etapa [4] do pipeline (treino): treina o classificador de sinais sobre
as sequências de landmarks extraídas pela etapa [3].

Duas fases distintas, de propósito separadas:

1. Avaliação (leave-one-signer-out): mede generalização pra sinalizador
   nunca visto — um fold por sinalizador, testado só nele e treinado em
   todos os outros. Serve só pra medir, não gera o modelo de produção.
2. Modelo final: treinado em 100% dos sinalizadores (sem deixar nenhum
   de fora), usando os mesmos hiperparâmetros já validados na fase 1.
   É esse que é salvo como o checkpoint de produção.

(Antes, o checkpoint salvo era o de um único fold do LOSO — o que
"acertou mais" num sinalizador específico, mas que nunca tinha visto
esse mesmo sinalizador no treino. Não era o modelo mais bem treinado,
era o mais bem avaliado num recorte. Separado conforme alinhado.)

Uso (PowerShell, numa linha; só o modelo final, sem os 8 folds):
    python -m pipeline.model.train --features-dir dataset/features_v2 --output model_v3_e100s2.pt --epochs 100 --frame-stride 2 --skip-eval

--frame-stride N usa 1 a cada N frames (acelera ~N vezes sem perder
acurácia nos testes) e fica gravado no checkpoint: o predict.py aplica o
mesmo stride sozinho na inferência.
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


def apply_frame_stride(samples: list[SignSample], stride: int) -> None:
    """Subamostra os frames de cada sequência (1 a cada `stride`), in-place."""
    if stride > 1:
        for s in samples:
            s.sequence = s.sequence[::stride]


def train_one_fold(
    samples: list[SignSample],
    label_to_idx: dict,
    train_idx: list[int],
    test_idx: list[int],
    epochs: int = 20,
    batch_size: int = 8,
    lr: float = 1e-3,
    device: str = "cpu",
    augment: bool = False,
) -> tuple[SignLSTMClassifier, float]:
    train_ds = SignSequenceDataset([samples[i] for i in train_idx], label_to_idx, augment=augment)
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
            logits = model(sequences, lengths)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        if epoch == 0 or (epoch + 1) % 10 == 0:
            print(f"    época {epoch + 1}/{epochs} loss={epoch_loss / len(train_loader):.4f}", flush=True)

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
    frame_stride: int = 1,
    augment: bool = False,
) -> tuple[dict[str, int], dict[str, float]]:
    """Mede generalização pra sinalizador novo. NÃO devolve um modelo
    de produção — só os números de acurácia por fold (ver
    train_final_model para o modelo que de fato será salvo)."""
    samples = load_all_samples(features_dir)
    apply_frame_stride(samples, frame_stride)
    label_to_idx = build_label_mapping(samples)

    fold_accuracies: dict[str, float] = {}
    for signer, train_idx, test_idx in leave_one_signer_out_splits(samples):
        _, accuracy = train_one_fold(
            samples, label_to_idx, train_idx, test_idx, epochs, batch_size, lr, device, augment
        )
        fold_accuracies[signer] = accuracy
        print(f"[leave-one-signer-out] sinalizador de teste={signer} acc={accuracy:.3f}")

    mean_accuracy = sum(fold_accuracies.values()) / len(fold_accuracies)
    print(f"Acurácia média de generalização (leave-one-signer-out): {mean_accuracy:.3f}")
    return label_to_idx, fold_accuracies


def train_final_model(
    samples: list[SignSample],
    label_to_idx: dict,
    epochs: int = 20,
    batch_size: int = 8,
    lr: float = 1e-3,
    device: str = "cpu",
    augment: bool = False,
) -> SignLSTMClassifier:
    """Treina o modelo de produção usando TODOS os sinalizadores (sem
    leave-one-out). Rodar depois de já ter validado a generalização via
    train_leave_one_signer_out — aquele mede, esse é o que vai pra produção."""
    all_idx = list(range(len(samples)))
    # test_idx = train_idx só porque train_one_fold exige um test_loader;
    # a "acurácia" retornada aqui é em cima dos próprios dados de treino
    # e não tem significado de generalização — ignorada de propósito.
    model, _ = train_one_fold(
        samples, label_to_idx, all_idx, all_idx, epochs, batch_size, lr, device, augment
    )
    return model


def main():
    parser = argparse.ArgumentParser(description="Treina o classificador de sinais em Libras.")
    parser.add_argument("--features-dir", required=True, help="Pasta com os .parquet gerados pela etapa [3]")
    parser.add_argument("--output", default="model.pt", help="Caminho de saída do checkpoint treinado")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--frame-stride", type=int, default=1, help="Usa 1 a cada N frames (gravado no checkpoint)")
    parser.add_argument("--augment", action="store_true", help="Augmentation no treino (nos testes, sem ganho)")
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Pula a fase de avaliação leave-one-signer-out e treina só o modelo final (mais rápido, sem métrica de generalização nesta rodada)",
    )
    args = parser.parse_args()

    samples = load_all_samples(args.features_dir)
    apply_frame_stride(samples, args.frame_stride)
    label_to_idx = build_label_mapping(samples)

    if not args.skip_eval:
        train_leave_one_signer_out(
            args.features_dir, args.epochs, args.batch_size, args.lr, args.device,
            args.frame_stride, args.augment,
        )
        print()

    print("Treinando modelo final em 100% dos dados (produção)...")
    final_model = train_final_model(
        samples, label_to_idx, args.epochs, args.batch_size, args.lr, args.device, args.augment
    )

    idx_to_label = {i: label for label, i in label_to_idx.items()}
    label_classes = [idx_to_label[i] for i in range(len(idx_to_label))]
    save_checkpoint(
        final_model,
        label_classes,
        args.output,
        meta={
            "frame_stride": args.frame_stride,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "augment": args.augment,
            "n_samples": len(samples),
        },
    )
    print(f"Modelo salvo em {args.output}")


if __name__ == "__main__":
    main()