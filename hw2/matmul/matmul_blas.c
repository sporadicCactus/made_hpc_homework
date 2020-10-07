#include <cblas.h>

void matmul_blas(double *A, double *B, double *C, int n) {
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1.0, A, n, B, n, 0.0, C, n);    
}
