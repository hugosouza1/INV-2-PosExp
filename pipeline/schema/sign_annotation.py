"""
sign_annotation.py
====================
Etapa [5] do pipeline: contrato de saída estável entre esse pipeline e
a camada seguinte (montagem de frase em português, resposta de um
assistente, etc). Deve continuar igual mesmo se o modelo interno mudar.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Optional


@dataclass(frozen=True)
class SignAnnotation:
    sinal: str
    confianca: float
    inicio_ms: int
    fim_ms: int
    sinalizador_id: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, **kwargs) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, **kwargs)
