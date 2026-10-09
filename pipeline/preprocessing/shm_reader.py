"""
shm_reader.py
=============
Substitui a leitura de vídeo por caminho (video_preprocessor.extract_frames)
na etapa [2]: lê os frames que o server.cpp já tratou e deixou na região de
memória compartilhada (/dev/shm/frames_cam).

Layout do bloco (precisa bater com ShmHeader do server.cpp):
    [0..15]   magic, width, height, channels  (4 x uint32)
    [16..23]  seq       (uint64, ímpar = escrita em andamento)
    [24..31]  frame_id  (uint64)
    [32..39]  ts_ms     (uint64)
    [64..]    pixels BGR uint8 (H x W x C)
"""

from __future__ import annotations

import mmap
import struct
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

HDR = 64
SEQ_OFF, FID_OFF, TS_OFF = 16, 24, 32
MAGIC = 0x46524D31  # "FRM1"


@dataclass
class ShmConfig:
    path: str = "/dev/shm/frames_cam"
    num_frames: int = 32          # frames coletados por sinal (janela)
    wait_first_s: float = 10.0    # espera máxima pelo 1º frame
    idle_timeout_s: float = 1.0   # sem frame novo por esse tempo => fim do sinal
    poll_s: float = 0.005         # intervalo de polling quando não há frame novo
    bgr_to_rgb: bool = True       # o server escreve BGR; o pipeline trabalha em RGB
    ignore_current: bool = True   # ignora o frame que já estava na shm ao abrir


class FrameReader:
    def __init__(self, path: str = "/dev/shm/frames_cam", ignore_current: bool = True):
        self.f = open(path, "rb")  # FileNotFoundError se o server ainda não criou a shm
        self.mm = mmap.mmap(self.f.fileno(), 0, access=mmap.ACCESS_READ)
        magic, self.w, self.h, self.c = struct.unpack_from("<IIII", self.mm, 0)
        assert magic == MAGIC, "bloco inválido"
        self.n = self.w * self.h * self.c
        # evita devolver um frame velho que ficou na shm de uma execução anterior
        self.last_id = self._u64(FID_OFF) if ignore_current else 0

    def _u64(self, off: int) -> int:
        return struct.unpack_from("<Q", self.mm, off)[0]

    def ler(self) -> Optional[tuple[np.ndarray, int]]:
        """Retorna (frame BGR HxWxC, ts_ms) se houver frame novo, senão None."""
        for _ in range(100):  # tentativas contra escrita concorrente
            s1 = self._u64(SEQ_OFF)
            if s1 & 1:
                continue
            fid = self._u64(FID_OFF)
            if fid == self.last_id:
                return None
            ts = self._u64(TS_OFF)
            img = np.frombuffer(self.mm, np.uint8, self.n, HDR).reshape(self.h, self.w, self.c).copy()
            if self._u64(SEQ_OFF) == s1:  # nada mudou durante a cópia
                self.last_id = fid
                return img, ts
        return None

    def close(self):
        self.mm.close()
        self.f.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def coletar_frames(
    reader: FrameReader, cfg: ShmConfig
) -> tuple[list[np.ndarray], list[int]]:
    """Coleta frames novos da shm até juntar `cfg.num_frames` ou o fluxo parar.

    Devolve (frames, timestamps_ms). Frames já saem tratados pelo server;
    aqui só se converte BGR -> RGB (se configurado).
    """
    frames: list[np.ndarray] = []
    stamps: list[int] = []
    t_ultimo = time.monotonic()

    while len(frames) < cfg.num_frames:
        res = reader.ler()
        agora = time.monotonic()
        if res is None:
            limite = cfg.idle_timeout_s if frames else cfg.wait_first_s
            if agora - t_ultimo > limite:
                break
            time.sleep(cfg.poll_s)
            continue

        frame, ts = res
        if cfg.bgr_to_rgb:
            frame = np.ascontiguousarray(frame[:, :, ::-1])
        frames.append(frame)
        stamps.append(ts)
        t_ultimo = agora

    return frames, stamps