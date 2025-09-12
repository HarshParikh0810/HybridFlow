import numpy as np
import time

def bala(A, B):
    n = len(A)
    C = [[0.0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            for k in range(n):
                C[i][j] += A[i][k] * B[k][j]
    return C

def main():
    N = 2048  
    print(f"Starting Python matrix multiplication of size {N}x{N}...")
    A = np.ones((N, N), dtype=np.float32).tolist()
    B = np.ones((N, N), dtype=np.float32).tolist()

    start = time.time()
    C = bala(A, B)
    end = time.time()

    print(f"Completed in {end - start:.3f} seconds.")
    print("C[0][0] =", C[0][0])

if __name__ == "__main__":
    main()
