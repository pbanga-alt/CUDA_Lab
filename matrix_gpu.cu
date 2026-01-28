#include <stdio.h>
#include <stdlib.h>
#include <time.h>

__global__ void matrixMultiplyGPU (float *A, float *B, float *C, int N) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	if ( row < N && col < N) {
		float sum = 0.0f;
		for (int k = 0; k < N; k++) {
			sum += A[row * N + k] * B[k * N + col];
		}
		C[row * N + col] = sum;
	}
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
	matrixMultiplyGPU<<<grid, block>>>(dA, dB, dC, N);
  cudaDeviceSynchronize();
  cudaMemcpy(C, dC, size, cudaMemcpyHostToDevice);
	clock_t end = clock();

	double elapsed = (double)(end - start) / CLOCKS_PER_SEC;
	printf("CPU execution time (N=%d): %f seconds \n", N, elapsed);

  cudaFree(dA); cudaFree(dB); cudaFree(dC);
 	free(A); free(B); free(C);
	return 0;
}
