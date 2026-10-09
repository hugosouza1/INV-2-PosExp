import mmap, struct, time
import numpy as np

HDR = 64
SEQ_OFF, FID_OFF, TS_OFF = 16, 24, 32

class FrameReader:
    def __init__(self, path="/dev/shm/frames_cam"):
        self.f = open(path, "rb")
        self.mm = mmap.mmap(self.f.fileno(), 0, access=mmap.ACCESS_READ)
        magic, self.w, self.h, self.c = struct.unpack_from("<IIII", self.mm, 0)
        assert magic == 0x46524D31, "bloco inválido"
        self.n = self.w * self.h * self.c
        self.last_id = 0

    def _u64(self, off):
        return struct.unpack_from("<Q", self.mm, off)[0]

    def ler(self):
        """Retorna (frame, ts_ms) se houver frame novo, senão None."""
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
        self.mm.close(); self.f.close()

if __name__ == "__main__":
    r = FrameReader()
    while True:
        res = r.ler()
        if res is None:
            time.sleep(0.005)
            continue
        frame, ts = res
        print(frame.shape, ts)