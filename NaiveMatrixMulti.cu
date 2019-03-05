#include <iostream>
#include <stdio.h>

using namespace std;

/**
 * CUDA kernel responsible for multiplying two matrices 'A' and 'B', using the 
 * naive approach, and storing result in matrix 'Y'
 * 
 * @param use2D Defines whether 2D blocks and grid are used. This only affects
 * the way the indices 'i' and 'j' are calculated.
 */
__global__ void naiveMMKernel(int n, double *A, double *B, double *Y, bool use2D) {
    int i, j;
    if (use2D) {
        i = threadIdx.y + (blockIdx.y * blockDim.y);
        j = threadIdx.x + (blockIdx.x * blockDim.x);
    } else {
        i = (threadIdx.x + (blockIdx.x * blockDim.x)) / n;
        j = (threadIdx.x + (blockIdx.x * blockDim.x)) - (i * n);
    }

    if (i >= n || j >= n) {
        return;
    }

    double res = 0;
    for (int k = 0; k < n; k++) {
        res += A[i * n + k] * B[k * n + j];
    }

    Y[i * n + j] = res;
}

/**
 * Multiplies matrices 'd_A' and 'd_B' using the naive approach and stores the 
 * result in matrix 'd_Y'
 * 
 * The input matrices have to reference the device memory
 * 
 * @param use2D Defines whether 2D blocks and grid should be used for the 
 * kernel configuration
 */
void naiveMM(int n, double *d_A, double *d_B, double *d_Y, bool use2D) {
    dim3 dimGrid, dimBlock;
    if (use2D) {
        // Total of 1024 threads
        dimBlock = dim3(32, 32);
        dimGrid = dim3((n + 31) / 32, (n + 31) / 32);
    } else {
        dimBlock = dim3(1024);
        dimGrid = dim3((n * n + 1023) / 1024);
    }

    naiveMMKernel<<<dimGrid, dimBlock>>>(n, d_A, d_B, d_Y, use2D);
    cudaThreadSynchronize();
}
