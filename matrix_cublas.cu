#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

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

  // Create cuBLAS handle
  cublasHandle_t handle;
  cublasCreate(&handle);

  float *dA, *dB, *dC;
  cudaMalloc(&dA, size);
  cudaMalloc(&dB, size);
  cudaMalloc(&dC, size);

  cudaMemcpy(dA, A, size, cudaMemcpyHostToDevice);
  cudaMemcpy(dB, B, size, cudaMemcpyHostToDevice);
  cudaMemcpy(dC, C, size, cudaMemcpyHostToDevice);

  //dim3 block(16,16);
  //dim3 grid((N+15)/16, (N+15)/16);
  float alpha = 0.0f;
  float beta = 0.0f;
	clock_t start = clock();
	cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, dB, N, dA, N, &beta, dC, N);
  cudaDeviceSynchronize();
  cudaMemcpy(C, dC, size, cudaMemcpyDeviceToHost);
	clock_t end = clock();

	double elapsed = (double)(end - start) / CLOCKS_PER_SEC;
	printf("CPU execution time (N=%d): %f seconds \n", N, elapsed);

  cudaFree(dA); cudaFree(dB); cudaFree(dC);
 	free(A); free(B); free(C);
	return 0;
}
