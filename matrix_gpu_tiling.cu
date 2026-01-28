#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define TILE_WIDTH 16

__global__ void matrixMultiplyTiled(float *A, float *B, float *C, int N) {
    __shared__ float ds_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float ds_B[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;

    float Pvalue = 0.0f;

    for (int m = 0; m < (N + TILE_WIDTH - 1) / TILE_WIDTH; ++m) {

        if (Row < N && (m * TILE_WIDTH + tx) < N)
            ds_A[ty][tx] = A[Row * N + m * TILE_WIDTH + tx];
        else
            ds_A[ty][tx] = 0.0f;

        if (Col < N && (m * TILE_WIDTH + ty) < N)
            ds_B[ty][tx] = B[(m * TILE_WIDTH + ty) * N + Col];
        else
            ds_B[ty][tx] = 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; ++k)
            Pvalue += ds_A[ty][k] * ds_B[k][tx];

        __syncthreads();
    }

    if (Row < N && Col < N)
        C[Row * N + Col] = Pvalue;
}


int main(int argc, char **argv) {
	int N = (argc > 1) ? atoi(argv[1]) : 1024;
	size_t size = N * N * sizeof(float);

	float *A = (float *)malloc(size);
	float *B = (float *)malloc(size);
	float *C = (float *)malloc(size);

	for (int i = 0; i < N * N; i++) {
		A[i] = rand() % 100/ 100.0f;
		B[i] = rand() % 100 / 100.0f;
	}

  float *dA, *dB, *dC;
  cudaMalloc(&dA, size);
  cudaMalloc(&dB, size);
  cudaMalloc(&dC, size);

  cudaMemcpy(dA, A, size, cudaMemcpyHostToDevice);
  cudaMemcpy(dB, B, size, cudaMemcpyHostToDevice);
  cudaMemcpy(dC, C, size, cudaMemcpyHostToDevice);

  dim3 block(16,16);
  dim3 grid((N+15)/16, (N+15)/16);

	clock_t start = clock();
	matrixMultiplyTiled<<<grid, block>>>(dA, dB, dC, N);
  cudaDeviceSynchronize();
  cudaMemcpy(C, dC, size, cudaMemcpyDeviceToHost);
	clock_t end = clock();

	double elapsed = (double)(end - start) / CLOCKS_PER_SEC;
	printf("CPU execution time (N=%d): %f seconds \n", N, elapsed);

  cudaFree(dA); cudaFree(dB); cudaFree(dC);
 	free(A); free(B); free(C);
	return 0;
}
