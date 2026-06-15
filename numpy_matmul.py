import numpy as np
import time

n = 500
A = np.random.rand(n, n)
B = np.random.rand(n, n)

start = time.perf_counter()
C_numpy = A @ B
elapsed = time.perf_counter() - start
print(f"NumPy:  {elapsed:.4f}s")


C_loop = np.zeros((n, n))
start = time.perf_counter()
for i in range(n):
    for j in range(n):
        for k in range(n):
            C_loop[i, j] += A[i, k] * B[k, j]
elapsed = time.perf_counter() - start
print(f"Loop:   {elapsed:.4f}s")

