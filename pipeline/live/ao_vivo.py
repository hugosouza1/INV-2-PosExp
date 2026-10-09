"""ao_vivo.py: lê frames da shm, segmenta por pausa e roda o pipeline."""
import argparse, os, time
import cv2
import numpy as np

from pipeline.live.leitor import FrameReader
from pipeline.feature_extraction.landmark_extractor import extract_landmarks
from pipeline.model.predict import SignPredictor
from pipeline.schema.sign_annotation import SignAnnotation

SHM = "/dev/shm/frames_cam"
TARGET_FPS = 25.0        # o mesmo do PreprocessConfig (o treino usou isso)
SIZE = (224, 224)        # idem, resize esticado como no video_preprocessor


def abrir():
    while True:
        try:
            return FrameReader(SHM)
        except (FileNotFoundError, AssertionError):
            time.sleep(0.5)


def preparar(buf):
    """buf: [(bgr, ts_ms)] -> (frames RGB 224x224 a 25 fps, duração_ms).
    O servidor só grava 1 a cada 3 frames e só com rosto detectado, então os
    timestamps têm buracos. Reamostra por timestamp pra recuperar os 25 fps."""
    ts = np.array([t for _, t in buf], dtype=np.float64)
    dur = ts[-1] - ts[0]
    n = max(2, int(round(dur / 1000 * TARGET_FPS)))
    alvo = np.linspace(ts[0], ts[-1], n)
    idx = np.abs(ts[None, :] - alvo[:, None]).argmin(axis=1)

    cache = {}
    frames = []
    for i in idx:
        if i not in cache:
            rgb = cv2.cvtColor(buf[i][0], cv2.COLOR_BGR2RGB)
            cache[i] = cv2.resize(rgb, SIZE, interpolation=cv2.INTER_AREA)
        frames.append(cache[i])
    return frames, int(dur)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--gap", type=float, default=1.0, help="s sem frame novo = fim do sinal")
    ap.add_argument("--min-ms", type=int, default=500, help="ignora trechos menores que isso")
    ap.add_argument("--max-s", type=float, default=6.0, help="corta trechos mais longos que isso")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    predictor = SignPredictor(args.model, device=args.device)
    reader = abrir()
    inode = os.fstat(reader.f.fileno()).st_ino
    buf, last = [], time.time()
    print("Aguardando frames...", flush=True)

    while True:
        # o servidor recria o arquivo quando o cliente reconecta
        try:
            if os.stat(SHM).st_ino != inode:
                reader.close()
                reader = abrir()
                inode = os.fstat(reader.f.fileno()).st_ino
        except FileNotFoundError:
            pass

        res = reader.ler()
        if res is not None:
            buf.append(res)
            last = time.time()
            if (buf[-1][1] - buf[0][1]) / 1000 < args.max_s:
                continue

        pausou = buf and (time.time() - last) > args.gap
        estourou = buf and (buf[-1][1] - buf[0][1]) / 1000 >= args.max_s
        if pausou or estourou:
            dur_ms = buf[-1][1] - buf[0][1]
            if dur_ms >= args.min_ms and len(buf) >= 4:
                frames, dur_ms = preparar(buf)
                seq = extract_landmarks(frames)
                label, conf = predictor.predict(seq)
                print(SignAnnotation(
                    sinal=label, confianca=conf,
                    inicio_ms=int(buf[0][1]), fim_ms=int(buf[-1][1]),
                ).to_json(indent=2), flush=True)
            else:
                print(f"trecho curto demais ({len(buf)} frames), ignorado")
            buf = []
        else:
            time.sleep(0.005)


if __name__ == "__main__":
    main()