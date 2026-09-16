"""
dataset.py
===========
Carrega as sequências de landmarks persistidas pela etapa [3] (parquet)
pra treino/avaliação do classificador, incluindo o split
leave-one-signer-out usado pra medir generalização de verdade (e não
apenas memorização do sinalizador).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from torch.utils.data import Dataset

from pipeline.feature_extraction.landmark_extractor import load_landmark_sequence


@dataclass
class SignSample:
    sequence: np.ndarray  # (T, F)
    label: str
    signer_id: str
    video_id: str


def load_all_samples(features_dir: str | Path) -> list[SignSample]:
    features_dir = Path(features_dir)
    samples = []
    for path in sorted(features_dir.glob("*.parquet")):
        sequence, video_id, class_label, signer_id = load_landmark_sequence(path)
        samples.append(SignSample(sequence, class_label, signer_id, video_id))
    if not samples:
        raise ValueError(f"Nenhuma sequência de landmarks encontrada em {features_dir}")
    return samples


def build_label_mapping(samples: list[SignSample]) -> dict[str, int]:
    classes = sorted({s.label for s in samples})
    return {label: i for i, label in enumerate(classes)}


def leave_one_signer_out_splits(
    samples: list[SignSample],
) -> Iterator[tuple[str, list[int], list[int]]]:
    """Gera (sinalizador_de_teste, train_idx, test_idx) pra cada
    sinalizador presente no dataset."""
    signers = sorted({s.signer_id for s in samples})
    for signer in signers:
        train_idx = [i for i, s in enumerate(samples) if s.signer_id != signer]
        test_idx = [i for i, s in enumerate(samples) if s.signer_id == signer]
        yield signer, train_idx, test_idx


class SignSequenceDataset(Dataset):
    def __init__(self, samples: list[SignSample], label_to_idx: dict[str, int]):
        self.samples = samples
        self.label_to_idx = label_to_idx

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        sample = self.samples[idx]
        sequence = torch.from_numpy(sample.sequence.astype(np.float32))
        return sequence, self.label_to_idx[sample.label]


def collate_padded(
    batch: list[tuple[torch.Tensor, int]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sequences, labels = zip(*batch)
    lengths = torch.tensor([len(s) for s in sequences], dtype=torch.long)
    padded = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True)
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    return padded, lengths, labels_tensor
