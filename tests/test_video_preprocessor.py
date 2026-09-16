import cv2
import numpy as np
import pytest

from pipeline.preprocessing.video_preprocessor import (
    PreprocessConfig,
    extract_frames,
    read_video_frames,
    resample_fps,
    resize_frames,
    trim_by_motion,
)


def _make_video(path, n_frames=30, fps=30, size=(64, 48)):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    for i in range(n_frames):
        frame = np.full((size[1], size[0], 3), i % 255, dtype=np.uint8)
        writer.write(frame)
    writer.release()


def test_resample_fps_changes_frame_count():
    frames = [np.zeros((4, 4, 3), dtype=np.uint8) for _ in range(30)]
    out = resample_fps(frames, original_fps=30, target_fps=10)
    assert len(out) == 10


def test_resample_fps_noop_when_same_fps():
    frames = [np.zeros((4, 4, 3), dtype=np.uint8) for _ in range(10)]
    out = resample_fps(frames, original_fps=25, target_fps=25)
    assert out is frames


def test_resize_frames_changes_shape():
    frames = [np.zeros((10, 10, 3), dtype=np.uint8) for _ in range(3)]
    out = resize_frames(frames, size=(20, 20))
    assert all(f.shape == (20, 20, 3) for f in out)


def test_trim_by_motion_keeps_only_moving_segment():
    still = np.zeros((10, 10, 3), dtype=np.uint8)
    moving = [np.full((10, 10, 3), i * 30, dtype=np.uint8) for i in range(5)]
    frames = [still, still] + moving + [still, still]
    trimmed = trim_by_motion(frames, threshold=5.0, min_frames=1)
    assert len(trimmed) <= len(frames)
    assert len(trimmed) >= len(moving)


def test_trim_by_motion_noop_when_all_still():
    frames = [np.zeros((10, 10, 3), dtype=np.uint8) for _ in range(5)]
    assert trim_by_motion(frames, threshold=5.0, min_frames=1) == frames


def test_extract_frames_from_real_video(tmp_path):
    video_path = tmp_path / "sample.mp4"
    _make_video(video_path, n_frames=30, fps=30, size=(64, 48))
    config = PreprocessConfig(target_fps=10, target_size=(32, 32))
    frames = extract_frames(video_path, config)
    assert len(frames) == 10
    assert frames[0].shape == (32, 32, 3)


def test_read_video_frames_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_video_frames(tmp_path / "nope.mp4")
