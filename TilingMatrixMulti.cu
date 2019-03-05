#include <iostream>
#include <stdio.h>

#define TILE_SIZE 32

/**
 * CUDA kernel responsible for multiplying matrices 'A' and 'B', using a tiling
 * approach, and storing result in matrix 'Y'
 */
__global__ void tilingMMKernel(int n, double *A, double *B, double *Y) {
    int ii = blockIdx.x * TILE_SIZE;
    int jj = blockIdx.y * TILE_SIZE;

    int i = ii + threadIdx.x;
    int j = jj + threadIdx.y;

    double res = 0;
    __shared__ double s_A[TILE_SIZE][TILE_SIZE], s_B[TILE_SIZE][TILE_SIZE];

    for (int kk = 0; kk < n; kk += TILE_SIZE) {
        s_A[threadIdx.x][threadIdx.y] = A[i * n + kk + threadIdx.y];
        s_B[threadIdx.x][threadIdx.y] = B[(kk + threadIdx.x) * n + j];

        __syncthreads();
        for (int k = 0; k < TILE_SIZE; k++)
            res += s_A[threadIdx.x][k] * s_B[k][threadIdx.y];
        __syncthreads();
    }

    Y[i * n + j] = res;
}

/**
 * CUDA kernel responsible for multiplying matrices 'A' and 'B', using a tiling 
 * approach with unrolling, and storing result in matrix 'Y'
 *
 * Every thread is responsible for calculating 4 elements in the result matrix
 * 'Y'.
 */
__global__ void tilingUnrollingMMKernel(int n, double *A, double *B, double *Y) {
    int ii = blockIdx.x * TILE_SIZE;
    int jj = blockIdx.y * TILE_SIZE;

    int x = threadIdx.x * 2;
    int y = threadIdx.y * 2;

    int i = ii + x;
    int j = jj + y;

    double res00 = 0, res01 = 0, res10 = 0, res11 = 0;
    __shared__ double s_A[TILE_SIZE][TILE_SIZE], s_B[TILE_SIZE][TILE_SIZE];

    for (int kk = 0; kk < n; kk += TILE_SIZE) {
        s_A[x][y] = A[i * n + kk + y];
        s_A[x + 1][y] = A[(i + 1) * n + kk + y];
        s_A[x][y + 1] = A[i * n + kk + y + 1];
        s_A[x + 1][y + 1] = A[(i + 1) * n + kk + y + 1];
    
        s_B[x][y] = B[(kk + x) * n + j];
        s_B[x + 1][y] = B[(kk + x + 1) * n + j];
        s_B[x][y + 1] = B[(kk + x) * n + j + 1];
        s_B[x + 1][y + 1] = B[(kk + x + 1) * n + j + 1];

        __syncthreads();
        for (int k = 0; k < TILE_SIZE; k++) {
            res00 += s_A[x][k] * s_B[k][y];
            res01 += s_A[x][k] * s_B[k][y + 1];
            res10 += s_A[x + 1][k] * s_B[k][y];
            res11 += s_A[x + 1][k] * s_B[k][y + 1];
        }
        __syncthreads();
    }

    Y[i * n + j] = res00;
    Y[i * n + j + 1] = res01;
    Y[(i + 1) * n + j] = res10;
    Y[(i + 1) * n + j + 1] = res11;
}

/**
 * Multiplies matrices 'd_A' and 'd_B' using the tiling optimized approach and 
 * stores the result in matrix 'd_Y'
 * 
 * @param unroll Defines whether loop unrolling should be applied
 */
void tilingMM(int n, double *d_A, double *d_B, double *d_Y, bool unroll) {
    if (unroll) {
        dim3 dimBlock(TILE_SIZE / 2, TILE_SIZE / 2);
        dim3 dimGrid((n + TILE_SIZE - 1) / TILE_SIZE, (n + TILE_SIZE - 1) / TILE_SIZE);

        tilingUnrollingMMKernel<<<dimGrid, dimBlock>>>(n, d_A, d_B, d_Y);
    } else {
        dim3 dimBlock(TILE_SIZE, TILE_SIZE);
        dim3 dimGrid((n + TILE_SIZE - 1) / TILE_SIZE, (n + TILE_SIZE - 1) / TILE_SIZE);

        tilingMMKernel<<<dimGrid, dimBlock>>>(n, d_A, d_B, d_Y);
    }

    cudaThreadSynchronize();
}
