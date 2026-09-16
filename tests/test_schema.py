import json

import pytest

from pipeline.schema.sign_annotation import SignAnnotation


def test_to_dict_matches_fields():
    ann = SignAnnotation(sinal="Aluno", confianca=0.94, inicio_ms=320, fim_ms=1580, sinalizador_id="signer01")
    assert ann.to_dict() == {
        "sinal": "Aluno",
        "confianca": 0.94,
        "inicio_ms": 320,
        "fim_ms": 1580,
        "sinalizador_id": "signer01",
    }


def test_to_json_roundtrip():
    ann = SignAnnotation(sinal="Banco", confianca=0.5, inicio_ms=0, fim_ms=100)
    parsed = json.loads(ann.to_json())
    assert parsed["sinal"] == "Banco"
    assert parsed["sinalizador_id"] is None


def test_is_frozen():
    ann = SignAnnotation(sinal="X", confianca=1.0, inicio_ms=0, fim_ms=1)
    with pytest.raises(Exception):
        ann.sinal = "Y"
