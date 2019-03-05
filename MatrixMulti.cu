#include <iostream>
#include <stdio.h>
#include <time.h>
#include "NaiveMatrixMulti.cu"
#include "TilingMatrixMulti.cu"

using namespace std;

#define NAIVE_1D 1
#define NAIVE_2D 2
#define TILING 3
#define TILING_LOOP_UNROLLING 4

/** Prints given matrix to console */
void printMatrix(int n, double *M) {
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++)
            cout << M[i * n + j] << "\t";

        cout << endl;
	}
}

/** Checks if matrix 'A' multiplied with matrix 'B' is 'Y' */
bool isResultCorrect(int n, double *A, double *B, double *Y) {
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
            double result = 0;
			for (int k = 0; k < n; k++)
				result += A[i * n + k] * B[k * n + j];

            if (abs(result - Y[i * n + j]) > 0.01)
                return false;
		}
	}

    return true;
}

/** Fills given matrix 'M' with random numbers */
void fillMatrix(int n, double *M) {
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++)
            M[j + i * n] = (double) ((int) rand() % 10);
	}
}

int main(int argc, char *argv[]) {
    if (argc < 3) {
        cout << "usage: " << argv[0] << " n strategy [-v]" << endl;
        cout << endl << "positional arguments:" << endl;
        cout << " n         Matrix size (Must be multiple of 32 for strategy 3 and 4)" << endl;
        cout << " strategy  Defines matrix multiplication strategy" << endl;
        cout << "            - 1 for naive matrix multi with 1D blocks and grid" << endl;
        cout << "            - 2 for naive matrix multi with 2D blocks and grid" << endl;
        cout << "            - 3 for tiling matrix multi" << endl;
        cout << "            - 4 for for tiling matrix multi with loop unrolling" << endl;
        cout << endl << "optional arguments:" << endl;
        cout << " -v        Validate matrix multiplication result" << endl;

        return -1;
    }

    // Parse command line arguments
	int n = atoi(argv[1]);
    int strategy = atoi(argv[2]);
    bool validateResult = argc > 3 ? !strcmp(argv[3], "-v") : false;

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
        default:
            cout << "\'" << strategy << "\' is not a valid strategy" << endl;
            return -1;
    }

    t = clock() - t;

    cudaMemcpy(Y, d_Y, size, cudaMemcpyDeviceToHost);

    cout << n << "\t";
    cout << abs(((float)(n * n * n) / 1000000.0) / (((float) t) / CLOCKS_PER_SEC)) << endl;
    if (validateResult)
        cout << "Result is " << (isResultCorrect(n, A, B, Y) ? "correct" : "incorrect") << endl;

    free(A); free(B); free(Y);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_Y);

    return 0;
}
