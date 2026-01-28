#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void printMatrix(const char *name, float *X, int rows, int cols) {
    printf("\n%s (%d x %d):\n", name, rows, cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%6.2f ", X[i * cols + j]);
        }
        printf("\n");
    }
}

void matrixConvolveCPU(float *A, float *B, float *C, int N, int M) {
    int out = M - N + 1;

    /* Print input matrices */
    printMatrix("Matrix A (Input)", A, M, M);
    printMatrix("Matrix B (Filter)", B, N, N);

    /* Convolution */
    for (int i = 0; i < out; i++) {
        for (int j = 0; j < out; j++) {
            float sum = 0.0f;

            for (int u = 0; u < N; u++) {
                for (int v = 0; v < N; v++) {
                    sum += A[(i + u) * M + (j + v)] * B[u * N + v];
                }
            }

            C[i * out + j] = sum;
        }
    }

    /* Print output matrix */
    printMatrix("Matrix C (Output)", C, out, out);
}

int main(int argc, char **argv) {
    int N = (argc > 2) ? atoi(argv[1]) : 4;   // filter size
    int M = (argc > 2) ? atoi(argv[2]) : 6;   // input size

    size_t size_n = N * N * sizeof(float);
    size_t size_m = M * M * sizeof(float);
    size_t size_out = (M - N + 1) * (M - N + 1) * sizeof(float);

    float *A = (float *)malloc(size_m);
    float *B = (float *)malloc(size_n);
    float *C = (float *)malloc(size_out);

    for (int i = 0; i < M * M; i++) {
        A[i] = rand() % 10;
    }
    for (int i = 0; i < N * N; i++) {
        B[i] = rand() % 10;
    }

    clock_t start = clock();
    matrixConvolveCPU(A, B, C, N, M);
    clock_t end = clock();

    double elapsed = (double)(end - start) / CLOCKS_PER_SEC;
    printf("\nCPU execution time: %f seconds\n", elapsed);

    free(A); free(B); free(C);
    return 0;
}
