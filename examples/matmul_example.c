#include <stdio.h>
#include <stdlib.h>

#define N 32 

void matmul(float A[N][N], float B[N][N], float C[N][N]) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < N; k++) {
                sum += A[i][k] * B[k][j];
            }
            C[i][j] = sum;
        }
    }
}

int main() {
    static float A[N][N], B[N][N], C[N][N];

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[i][j] = 1.0f;
            B[i][j] = 1.0f;
            C[i][j] = 0.0f;
        }
    }

    printf("Starting matrix multiplication of size %dx%d...\n", N, N);
    matmul(A, B, C);
    printf("Matrix multiplication execution completed on CPU.\n");

    return 0;
}
