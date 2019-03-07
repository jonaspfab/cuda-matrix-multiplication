#include <iostream>
#include <stdio.h>
#include <time.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include "NaiveMatrixMulti.cu"
#include "TilingMatrixMulti.cu"

using namespace std;

#define NAIVE_1D 0
#define NAIVE_2D 1
#define TILING 2
#define TILING_LOOP_UNROLLING 3
#define CU_BLAS 4

const char *strategyNames[5] = {
    "Naive 1D           ", 
    "Naive 2D           ", 
    "Tiling             ", 
    "Tiling+Unrolling   ", 
    "cuBLAS             "
};

/**
 * Multiplies matrix 'A' and 'B' and stores result in 'Y' using the cuBLAS
 * library
 *
 * Note that the matrix parameters must reference the device memory
 *
 * @param handle Created CuBlas handle
 */
void cuBlasMM(cublasHandle_t handle, int n, double *d_A, double *d_B, double *d_Y) {
    double alpha = 1.0;
	double beta = 0.0;

	cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, d_B, n, d_A, n, &beta, d_Y, n);
    cudaThreadSynchronize();
}

/**
 * Calculates the total error between the given result in 'Y' and the cuBLAS 
 * result
 *
 * Note that parameters 'd_A', 'd_B', and 'd_Y' must reference the device memory
 * while 'Y' has to reference the host memory
 *
 * @param handle Created CuBlas handle
 */
double calcError(cublasHandle_t handle, int n, double *d_A, double *d_B, double *d_Y, double *Y) {
    double *cublas_Y;
    int size = n * n * sizeof(double);
    cublas_Y = (double *)malloc(size);

    cuBlasMM(handle, n, d_A, d_B, d_Y);
    cudaMemcpy(cublas_Y, d_Y, size, cudaMemcpyDeviceToHost);

    double error = 0.0;

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            error += abs(Y[i * n + j] - cublas_Y[i * n + j]);
        }
    }

    return error;
}

/** Fills given matrix 'M' with random numbers */
void fillMatrix(int n, double *M) {
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++)
            M[j + i * n] = (double) ((int) rand() % 10);
	}
}

int main(int argc, char *argv[]) {
    // Create handle here so it won't be included in timing
    cublasHandle_t handle;
    cublasCreate(&handle);

    cout << "Strategy\t\tMatrix Order\tDim Blocks\tDim Grid\tMFlops\t\tError" << endl;

    for (int n = 32; n <= 1600; n += 256) {
        // Matrices stored on host memory
        double *A, *B, *Y;
        // Matrices stored on device memory
        double *d_A, *d_B, *d_Y;
        int size = n * n * sizeof(double);

	    A = (double *)malloc(size);
	    B = (double *)malloc(size);
	    Y = (double *)malloc(size);

        cudaMalloc((void **)&d_A, size);
        cudaMalloc((void **)&d_B, size);
        cudaMalloc((void **)&d_Y, size);

        // Fill matrices with random values
        fillMatrix(n, A);
        fillMatrix(n, B);

        cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

        // Loop over all 5 strategies
        for (int strategy = 0; strategy <= 4; strategy++) {
            cout << strategyNames[strategy] << "\t";
            cout << n << "\t\t";

            clock_t t = clock();

            switch (strategy) {
                case NAIVE_1D:
                    naiveMM(n, d_A, d_B, d_Y, false);
                    break;
                case NAIVE_2D:
                    naiveMM(n, d_A, d_B, d_Y, true);
                    break;
                case TILING:
                    tilingMM(n, d_A, d_B, d_Y, false);
                    break;
                case TILING_LOOP_UNROLLING:
                    tilingMM(n, d_A, d_B, d_Y, true);
                    break;
                case CU_BLAS:
                    cuBlasMM(handle, n, d_A, d_B, d_Y);
                    cout << "N/A\t\tN/A\t\t";
                    break;
                default:
                    cout << "\'" << strategy << "\' is not a valid strategy" << endl;
                    return -1;
            }

            t = clock() - t;
            cudaMemcpy(Y, d_Y, size, cudaMemcpyDeviceToHost);

            double nD = (double) n;
            cout << ((nD / 1000000.0) * nD * nD) / (t / ((double) CLOCKS_PER_SEC)) << "\t\t";
            cout << calcError(handle, n, d_A, d_B, d_Y, Y) << endl;        
        }

        free(A); free(B); free(Y);
        cudaFree(d_A); cudaFree(d_B); cudaFree(d_Y);
    }

    cublasDestroy(handle);

    return 0;
}
