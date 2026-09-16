import numpy as np

from pipeline import pipeline as pipeline_module
from pipeline.schema.sign_annotation import SignAnnotation


class _FakePredictor:
    def __init__(self, model_path, device="cpu"):
        self.model_path = model_path

    def predict(self, sequence):
        return "Aluno", 0.87


def test_pipeline_run_orchestrates_stages_and_builds_annotation(monkeypatch, tmp_path):
    fake_frames = [np.zeros((32, 32, 3), dtype=np.uint8)] * 25  # 25 frames @ 25fps -> 1000ms
    fake_sequence = np.zeros((25, 10), dtype=np.float32)

    monkeypatch.setattr(pipeline_module, "extract_frames", lambda path, config: fake_frames)
    monkeypatch.setattr(pipeline_module, "extract_landmarks", lambda frames, config: fake_sequence)
    monkeypatch.setattr(pipeline_module, "SignPredictor", _FakePredictor)

    pipe = pipeline_module.LibrasPipeline(model_path=tmp_path / "model.pt")
    annotation = pipe.run(tmp_path / "video.mp4", sinalizador_id="Sinalizador07")

    assert isinstance(annotation, SignAnnotation)
    assert annotation.sinal == "Aluno"
    assert annotation.confianca == 0.87
    assert annotation.inicio_ms == 0
    assert annotation.fim_ms == 1000
    assert annotation.sinalizador_id == "Sinalizador07"
