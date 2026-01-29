import ctypes
import numpy as np
import time
import matplotlib.pyplot as plt

times = []
Ns = []
# Load shared library
lib = ctypes.cdll.LoadLibrary("./libconv_cuda.so")

# Define argument types
lib.gpu_matrix_convolve.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),
    ctypes.c_int,
    ctypes.c_int
]

A = np.random.rand(N, N).astype(np.float32)
B = np.random.rand(N, N).astype(np.float32)
C = np.zeros((N, N), dtype=np.float32)
N = 4
for M in range(512,3584,512):
    start = time.time()
    lib.gpu_matrix_convolve(A.ravel(), B.ravel(), C.ravel(), N, M)
    end = time.time()
    times.append(end-start)
    Ns.append(M)
    print(f"Python call to CUDA library completed in {end - start:.8f} seconds")

plt.plot(Ns, times, marker="o")
plt.xlabel("N")
plt.ylabel("Runtime (seconds)")
plt.title("Matrix CPU Runtime")
plt.grid(True)
plt.show()


