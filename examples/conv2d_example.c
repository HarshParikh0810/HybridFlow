#include <stdio.h>

#define N 1024
#define K 1024

void conv2d(float input[N][N], float kernel[K][K], float output[N-K+1][N-K+1]) {
    for (int i = 0; i < N - K + 1; i++) {
        for (int j = 0; j < N - K + 1; j++) {
            float sum = 0.0f;
            for (int ki = 0; ki < K; ki++) {
                for (int kj = 0; kj < K; kj++) {
                    sum += input[i+ki][j+kj] * kernel[ki][kj];
                }
            }
            output[i][j] = sum;
        }
    }
}

int main() {
    static float input[N][N] = {0};
    static float kernel[K][K] = {1};
    static float output[N-K+1][N-K+1] = {0};

    conv2d(input, kernel, output);

    printf("CPU conv2d complete\n");
    return 0;
}
