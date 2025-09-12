#include <stdio.h>
#include <cuda_runtime.h>

#define N 32
#define K 3

__global__ void conv2d(float *input, float *kernel, float *output) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N - K + 1 && j < N - K + 1) {
        float sum = 0.0f;
        for (int ki = 0; ki < K; ki++) {
            for (int kj = 0; kj < K; kj++) {
                sum += input[(i+ki)*N + (j+kj)] * kernel[ki*K + kj];
            }
        }
        output[i*(N-K+1) + j] = sum;
    }
}

int main() {
    int out_size = (N-K+1)*(N-K+1);
    float *h_in = new float[N*N]();
    float *h_kernel = new float[K*K]();
    float *h_out = new float[out_size]();

    float *d_in, *d_kernel, *d_out;
    cudaMalloc(&d_in, N*N*sizeof(float));
    cudaMalloc(&d_kernel, K*K*sizeof(float));
    cudaMalloc(&d_out, out_size*sizeof(float));

    cudaMemcpy(d_in, h_in, N*N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, h_kernel, K*K*sizeof(float), cudaMemcpyHostToDevice);

    dim3 threads(16, 16);
    dim3 blocks((N-K+1+15)/16, (N-K+1+15)/16);
    conv2d<<<blocks, threads>>>(d_in, d_kernel, d_out);

    cudaMemcpy(h_out, d_out, out_size*sizeof(float), cudaMemcpyDeviceToHost);

    printf("GPU conv2d complete\n");

    cudaFree(d_in); cudaFree(d_kernel); cudaFree(d_out);
    delete[] h_in; delete[] h_kernel; delete[] h_out;

    return 0;
}
