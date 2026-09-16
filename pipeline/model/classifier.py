"""
classifier.py
==============
Arquitetura do classificador de sinais: uma LSTM (bidirecional por
padrão) sobre a sequência de landmarks, seguida de uma cabeça linear
que classifica a partir do último estado oculto. Inclui também
serialização (save/load) do checkpoint treinado.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn


@dataclass
class ClassifierConfig:
    input_dim: int
    num_classes: int
    hidden_dim: int = 128
    num_layers: int = 1
    dropout: float = 0.2
    bidirectional: bool = True


class SignLSTMClassifier(nn.Module):
    def __init__(self, config: ClassifierConfig):
        super().__init__()
        self.config = config
        self.lstm = nn.LSTM(
            input_size=config.input_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            batch_first=True,
            bidirectional=config.bidirectional,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
        )
        out_dim = config.hidden_dim * (2 if config.bidirectional else 1)
        self.classifier = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(out_dim, out_dim // 2),
            nn.ReLU(),
            nn.Linear(out_dim // 2, config.num_classes),
        )

    def forward(self, sequences: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        sequences: (B, T, F) já com padding.
        lengths: (B,) comprimento real de cada sequência (antes do padding).
        Retorna logits (B, num_classes).
        """
        packed = nn.utils.rnn.pack_padded_sequence(
            sequences, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (h_n, _) = self.lstm(packed)
        if self.config.bidirectional:
            last = torch.cat([h_n[-2], h_n[-1]], dim=-1)
        else:
            last = h_n[-1]
        return self.classifier(last)


def save_checkpoint(model: SignLSTMClassifier, label_classes: list[str], path: str | Path) -> None:
    torch.save(
        {
            "config": model.config.__dict__,
            "state_dict": model.state_dict(),
            "label_classes": label_classes,
        },
        path,
    )


def load_checkpoint(path: str | Path, map_location: str = "cpu") -> tuple[SignLSTMClassifier, list[str]]:
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    config = ClassifierConfig(**checkpoint["config"])
    model = SignLSTMClassifier(config)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint["label_classes"]
