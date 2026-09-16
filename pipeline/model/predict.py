"""
predict.py
===========
Etapa [4] do pipeline em produção (inferência): carrega um checkpoint
treinado e classifica uma sequência de landmarks já extraída pela
etapa [3].
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from pipeline.model.classifier import load_checkpoint


class SignPredictor:
    def __init__(self, model_path: str | Path, device: str = "cpu"):
        self.device = device
        self.model, self.label_classes = load_checkpoint(model_path, map_location=device)
        self.model.to(device)
        self.model.eval()

    def predict(self, sequence: np.ndarray) -> tuple[str, float]:
        """Recebe uma sequência (T, F) e devolve (rótulo previsto, confiança em [0, 1])."""
        tensor = torch.from_numpy(sequence.astype(np.float32)).unsqueeze(0).to(self.device)
        lengths = torch.tensor([tensor.shape[1]], dtype=torch.long)
        with torch.no_grad():
            logits = self.model(tensor, lengths)
            probs = F.softmax(logits, dim=-1).squeeze(0)
            confidence, idx = torch.max(probs, dim=-1)
        return self.label_classes[idx.item()], float(confidence.item())
