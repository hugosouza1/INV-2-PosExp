"""
pipeline/model/augmentation.py
================================
Augmentation leve para sequências de landmarks já normalizadas (ver
landmark_extractor.LandmarkExtractionConfig.normalize_landmarks=True):
jitter espacial (ruído gaussiano pequeno nas coordenadas) e jitter
temporal (simula sinalizar um pouco mais rápido ou devagar). Aplicar SÓ
no split de treino — nunca no de teste, senão a avaliação deixa de
refletir dado real.

(SignSequenceDataset em dataset.py já chama augment_sequence quando
augment=True.)
"""

from __future__ import annotations

import numpy as np


def spatial_jitter(sequence: np.ndarray, std: float = 0.01) -> np.ndarray:
    """Adiciona ruído gaussiano pequeno às coordenadas normalizadas.
    std=0.01 é seguro porque os landmarks já estão na escala da largura
    dos ombros — um std maior começa a distorcer a forma do sinal.

    O ruído só é aplicado onde há valor detectado (!= 0): zeros
    significam "mão/rosto/pose não detectado" no extrator, e adicionar
    ruído neles criaria landmarks falsos onde não havia nada."""
    noise = np.random.normal(0, std, size=sequence.shape).astype(sequence.dtype)
    noise *= sequence != 0
    return sequence + noise


def temporal_jitter(
    sequence: np.ndarray, scale_range: tuple[float, float] = (0.85, 1.15)
) -> np.ndarray:
    """Muda a velocidade do sinal: reamostra a sequência para um
    comprimento T' = T * escala (escala sorteada em scale_range) por
    interpolação linear. Devolve a sequência com o NOVO comprimento —
    escala > 1 = sinal mais lento (mais frames), < 1 = mais rápido. O
    modelo e o collate_padded aceitam comprimentos variáveis, então não
    é preciso voltar ao tamanho original (voltar anularia o efeito)."""
    t = sequence.shape[0]
    if t < 2:
        return sequence

    scale = np.random.uniform(*scale_range)
    t_new = max(2, int(round(t * scale)))

    pos = np.linspace(0, t - 1, t_new)
    i0 = np.floor(pos).astype(int)
    i1 = np.minimum(i0 + 1, t - 1)
    w = (pos - i0)[:, None].astype(sequence.dtype)
    out = (1.0 - w) * sequence[i0] + w * sequence[i1]
    return out.astype(sequence.dtype)


def augment_sequence(sequence: np.ndarray, p: float = 0.5) -> np.ndarray:
    """Aplica jitter temporal e espacial, cada um com probabilidade `p`,
    de forma independente. Usar só no split de treino."""
    if np.random.random() < p:
        sequence = temporal_jitter(sequence)
    if np.random.random() < p:
        sequence = spatial_jitter(sequence)
    return sequence