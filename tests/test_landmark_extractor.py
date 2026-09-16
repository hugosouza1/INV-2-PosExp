from types import SimpleNamespace

import numpy as np

from pipeline.feature_extraction.landmark_extractor import (
    FACE_DIM_FULL,
    FACE_DIM_REDUCED,
    FACE_SEMANTIC_LANDMARK_INDICES,
    HAND_DIM,
    POSE_DIM,
    LandmarkExtractionConfig,
    _flatten,
    _flatten_by_indices,
    load_landmark_sequence,
    result_to_feature_vector,
    save_landmark_sequence,
)


def _lm(x, y, z, visibility=None):
    return SimpleNamespace(x=x, y=y, z=z, visibility=visibility)


def test_feature_dim_reduced_face_is_default_and_much_smaller_than_full():
    reduced = LandmarkExtractionConfig(include_face=True, face_mode="reduced").feature_dim
    full = LandmarkExtractionConfig(include_face=True, face_mode="full").feature_dim
    no_face = LandmarkExtractionConfig(include_face=False).feature_dim

    assert reduced == POSE_DIM + 2 * HAND_DIM + FACE_DIM_REDUCED
    assert full == POSE_DIM + 2 * HAND_DIM + FACE_DIM_FULL
    assert no_face == POSE_DIM + 2 * HAND_DIM
    assert reduced < full
    assert len(FACE_SEMANTIC_LANDMARK_INDICES) < 478


def test_flatten_by_indices_reads_only_requested_points():
    landmarks = [_lm(i / 100, i / 100, i / 100) for i in range(10)]
    vec = _flatten_by_indices(landmarks, indices=[2, 5], include_visibility=False)
    assert vec.shape == (6,)
    np.testing.assert_allclose(vec, [0.02, 0.02, 0.02, 0.05, 0.05, 0.05], rtol=1e-6)


def test_flatten_by_indices_empty_landmarks_returns_zeros():
    vec = _flatten_by_indices(None, indices=[0, 1, 2], include_visibility=False)
    assert vec.shape == (9,)
    assert np.all(vec == 0.0)


def test_flatten_pads_missing_points_with_zeros():
    vec = _flatten([_lm(0.1, 0.2, 0.3)], num_points=3, include_visibility=False)
    assert vec.shape == (9,)
    np.testing.assert_allclose(vec[:3], [0.1, 0.2, 0.3], rtol=1e-6)
    assert vec[3:].tolist() == [0.0] * 6


def test_flatten_empty_landmarks_returns_zeros():
    vec = _flatten(None, num_points=2, include_visibility=True)
    assert vec.shape == (8,)
    assert np.all(vec == 0.0)


def test_flatten_includes_visibility_when_requested():
    vec = _flatten([_lm(0.1, 0.2, 0.3, visibility=0.75)], num_points=1, include_visibility=True)
    np.testing.assert_allclose(vec, [0.1, 0.2, 0.3, 0.75], rtol=1e-6)


def test_result_to_feature_vector_concatenates_components():
    config = LandmarkExtractionConfig(include_face=False)
    result = SimpleNamespace(
        pose_landmarks=[_lm(0.1, 0.1, 0.1, 0.9)] * 33,
        left_hand_landmarks=[_lm(0.2, 0.2, 0.2)] * 21,
        right_hand_landmarks=[_lm(0.3, 0.3, 0.3)] * 21,
        face_landmarks=[],
    )
    vec = result_to_feature_vector(result, config)
    assert vec.shape == (config.feature_dim,)


def test_result_to_feature_vector_handles_missing_hand():
    config = LandmarkExtractionConfig(include_face=False)
    result = SimpleNamespace(
        pose_landmarks=[_lm(0.1, 0.1, 0.1, 0.9)] * 33,
        left_hand_landmarks=[],
        right_hand_landmarks=[_lm(0.3, 0.3, 0.3)] * 21,
        face_landmarks=[],
    )
    vec = result_to_feature_vector(result, config)
    assert vec.shape == (config.feature_dim,)
    left_hand_slice = vec[POSE_DIM : POSE_DIM + HAND_DIM]
    assert np.all(left_hand_slice == 0.0)


def test_save_and_load_landmark_sequence_roundtrip(tmp_path):
    sequence = np.random.rand(5, 12).astype(np.float32)
    path = save_landmark_sequence(
        video_id="01AcontecerSinalizador01-1.mp4",
        class_label="Acontecer",
        signer_id="Sinalizador01",
        sequence=sequence,
        output_dir=tmp_path,
    )
    loaded_sequence, video_id, class_label, signer_id = load_landmark_sequence(path)
    assert video_id == "01AcontecerSinalizador01-1.mp4"
    assert class_label == "Acontecer"
    assert signer_id == "Sinalizador01"
    np.testing.assert_allclose(loaded_sequence, sequence, rtol=1e-5)
