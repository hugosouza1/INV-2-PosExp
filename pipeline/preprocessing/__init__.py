# ANTIGO:
# from pipeline.preprocessing.video_preprocessor import PreprocessConfig, extract_frames
# __all__ = ["PreprocessConfig", "extract_frames"]

from pipeline.preprocessing.shm_reader import FrameReader, ShmConfig, coletar_frames

__all__ = ["FrameReader", "ShmConfig", "coletar_frames"]