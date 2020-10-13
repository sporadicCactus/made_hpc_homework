#include <matutils.h>

void matmul_naive(double *A, double *B, double *C, int n) {
    zero_fill(C, n);
    for (int k = 0; k < n; k++) {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                C[n*i + j] += A[n*i + k] * B[n*k + j];
            }
        }
    }
}

void matvec_naive(double *A, double *b, double *c, int n) {
    for (int i = 0; i < n; i++) {
        c[i] = 0.0;
        for (int j = 0; j < n; j++) {
            c[i] += A[n*i + j]*b[j]; 
        }
    }
}
