import os, time, ctypes
import numpy as np
import pandas as pd
from PIL import Image

IMAGE_PATHS = [
    "black-and-white-landscape-photographs-inspire-adventure-feature.jpg",
    "Alliance_I_master.jpg",
    "Wild-Horse-Wyoming-3284-Edit-Edit-BW.jpg",
]

LIB_PATH = "./libconv_cuda.so"   # CUDA .so that exports gpu_matrix_convolve
OUT_DIR = "outputs_cuda"
os.makedirs(OUT_DIR, exist_ok=True)

for p in IMAGE_PATHS:
    if not os.path.exists(p):
        raise FileNotFoundError(p)
if not os.path.exists(LIB_PATH):
    raise FileNotFoundError(LIB_PATH)

lib = ctypes.cdll.LoadLibrary(LIB_PATH)
lib.gpu_matrix_convolve.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),
    ctypes.c_int,  # M
    ctypes.c_int,  # N
]
lib.gpu_matrix_convolve.restype = None


def sobel_x_3x3():
    return np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ], dtype=np.float32)

def gaussian_blur_5x5():
    k = np.array([
        [1,  4,  6,  4, 1],
        [4, 16, 24, 16, 4],
        [6, 24, 36, 24, 6],
        [4, 16, 24, 16, 4],
        [1,  4,  6,  4, 1]
    ], dtype=np.float32)
    return k / k.sum()

def box_blur_7x7():
    k = np.ones((7, 7), dtype=np.float32)
    return k / k.sum()

FILTERS = [
    ("sobel_x_3x3", sobel_x_3x3()),
    ("gaussian_blur_5x5", gaussian_blur_5x5()),
    ("box_blur_7x7", box_blur_7x7()),
]


def load_gray_square(path):
    img = Image.open(path).convert("L")
    w, h = img.size
    m = min(w, h)
    left = (w - m) // 2
    top = (h - m) // 2
    img = img.crop((left, top, left + m, top + m))
    return np.asarray(img, dtype=np.float32), m


def run_conv_cuda(A, K):
    M = int(A.shape[0])
    N = int(K.shape[0])
    out = M - N + 1
    if out <= 0:
        return None, None

    A1 = np.ascontiguousarray(A.ravel(), dtype=np.float32)
    K1 = np.ascontiguousarray(K.ravel(), dtype=np.float32)
    C1 = np.zeros(out * out, dtype=np.float32)

    t0 = time.perf_counter()
    lib.gpu_matrix_convolve(A1, K1, C1, M, N)
    t1 = time.perf_counter()

    return C1.reshape(out, out), (t1 - t0) * 1000.0


def save_scaled(X, path):
    mn, mx = float(X.min()), float(X.max())
    if mx - mn < 1e-9:
        Y = np.zeros_like(X, dtype=np.uint8)
    else:
        Y = ((X - mn) / (mx - mn) * 255).astype(np.uint8)
    Image.fromarray(Y, mode="L").save(path)


rows = []

for img_path in IMAGE_PATHS:
    base = os.path.splitext(os.path.basename(img_path))[0]
    A, M = load_gray_square(img_path)

    for fname, K in FILTERS:
        N = int(K.shape[0])
        C, ms = run_conv_cuda(A, K)

        if C is None:
            rows.append({"image": base, "M": M, "filter": fname, "N": N, "time_ms": None, "note": "SKIPPED"})
            continue

        out_file = f"{base}_M{M}_{fname}_N{N}.png"
        save_scaled(C, os.path.join(OUT_DIR, out_file))

        rows.append({"image": base, "M": M, "filter": fname, "N": N, "time_ms": round(ms, 3), "note": ""})

df = pd.DataFrame(rows)[["image", "M", "filter", "N", "time_ms", "note"]]
print(df.to_string(index=False))
