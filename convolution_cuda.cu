#include <cuda_runtime.h>
#include <stdio.h>

#define TILE_WIDTH 16

__global__ void conv2dCuda(const float *A, const float *B, float *C, int M, int N) {
    int out = M - N + 1;

    int j = blockIdx.x * blockDim.x + threadIdx.x; // Col in output
    int i = blockIdx.y * blockDim.y + threadIdx.y; // Row in output

    if (i < out && j < out) {
        float sum = 0.0f;

        // Slide N×N filter over A at (i, j)
        for (int u = 0; u < N; u++) {
            int a_row = i + u;
            int a_base = a_row * M + j;
            int b_base = u * N;
            for (int v = 0; v < N; v++) {
                sum += A[a_base + v] * B[b_base + v];
            }
        }

        C[i * out + j] = sum;
    }
}


extern "C" void gpu_matrix_convolve(const float *h_A, const float *h_B, float *h_C, int M, int N) {
    int out = M - N + 1;
    if (out <= 0) {
        return;
    }

    size_t sizeA = (size_t)M * M * sizeof(float);
    size_t sizeB = (size_t)N * N * sizeof(float);
    size_t sizeC = (size_t)out * out * sizeof(float);

    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;

    cudaMalloc((void**)&d_A, sizeA);
    cudaMalloc((void**)&d_B, sizeB);
    cudaMalloc((void**)&d_C, sizeC);

    cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice);

    dim3 block(TILE_WIDTH, TILE_WIDTH);
    dim3 grid((out + TILE_WIDTH - 1) / TILE_WIDTH,
              (out + TILE_WIDTH - 1) / TILE_WIDTH);

    conv2dCuda<<<grid, block>>>(d_A, d_B, d_C, M, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}
