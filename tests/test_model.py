import numpy as np
import torch

from pipeline.model.classifier import (
    ClassifierConfig,
    SignLSTMClassifier,
    load_checkpoint,
    save_checkpoint,
)
from pipeline.model.dataset import SignSample, build_label_mapping, leave_one_signer_out_splits
from pipeline.model.predict import SignPredictor


def test_classifier_forward_shape():
    config = ClassifierConfig(input_dim=10, num_classes=4, hidden_dim=8)
    model = SignLSTMClassifier(config)
    sequences = torch.rand(3, 6, 10)
    lengths = torch.tensor([6, 4, 2])
    logits = model(sequences, lengths)
    assert logits.shape == (3, 4)


def test_build_label_mapping_is_sorted_and_unique():
    samples = [
        SignSample(np.zeros((1, 1)), "Banco", "S1", "v1"),
        SignSample(np.zeros((1, 1)), "Aluno", "S1", "v2"),
        SignSample(np.zeros((1, 1)), "Aluno", "S2", "v3"),
    ]
    assert build_label_mapping(samples) == {"Aluno": 0, "Banco": 1}


def test_leave_one_signer_out_splits_cover_every_signer_exactly_once_as_test():
    samples = [
        SignSample(np.zeros((1, 1)), "Aluno", "S1", "v1"),
        SignSample(np.zeros((1, 1)), "Aluno", "S2", "v2"),
        SignSample(np.zeros((1, 1)), "Banco", "S1", "v3"),
    ]
    folds = list(leave_one_signer_out_splits(samples))
    assert {signer for signer, _, _ in folds} == {"S1", "S2"}
    for signer, train_idx, test_idx in folds:
        assert all(samples[i].signer_id == signer for i in test_idx)
        assert all(samples[i].signer_id != signer for i in train_idx)
        assert set(train_idx) | set(test_idx) == set(range(len(samples)))


def test_checkpoint_roundtrip_and_predictor(tmp_path):
    config = ClassifierConfig(input_dim=5, num_classes=3, hidden_dim=4)
    model = SignLSTMClassifier(config)
    labels = ["Aluno", "Banco", "Cinco"]
    path = tmp_path / "model.pt"
    save_checkpoint(model, labels, path)

    loaded_model, loaded_labels = load_checkpoint(path)
    assert loaded_labels == labels
    assert isinstance(loaded_model, SignLSTMClassifier)

    predictor = SignPredictor(path)
    sequence = np.random.rand(7, 5).astype(np.float32)
    label, confidence = predictor.predict(sequence)
    assert label in labels
    assert 0.0 <= confidence <= 1.0
