#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <matutils.h>
#include <matmul_naive.h>
#include <matmul_strassen.h>

#define threshold 128

struct mat {
    double *ptr;
    int s_i;
    int s_j;
    int n_i;
    int n_j;
};

int min(int a, int b) {
    return (a < b) ? a : b;
}

int max(int a, int b) {
    return (a >= b) ? a : b;
}


void add(const struct mat a, const struct mat b, struct mat c) {
    assert(max(a.n_i, b.n_i) <= c.n_i);
    assert(max(a.n_j, b.n_j) <= c.n_j);

    struct mat more_rows = (a.n_i >= b.n_i) ? a : b;
    struct mat more_cols = (a.n_j >= b.n_j) ? a : b; 

    for (int i = 0; i < min(a.n_i, b.n_i); i++) {
        for (int j = 0; j < min(a.n_j, b.n_j); j++) {
            c.ptr[i*c.s_i + j*c.s_j] = a.ptr[i*a.s_i + j*a.s_j] + b.ptr[i*b.s_i + j*b.s_j];       
        } 
        for (int j = min(a.n_j, b.n_j); j < max(a.n_j, b.n_j); j++) {
            c.ptr[i*c.s_i + j*c.s_j] = more_cols.ptr[i*more_cols.s_i + j*more_cols.s_j];
        }
        for (int j = max(a.n_j, b.n_j); j < c.n_j; j++) {
            c.ptr[i*c.s_i + j*c.s_j] = 0.0;
        }
    } 
    for (int i = min(a.n_i, b.n_i); i < max(a.n_i, b.n_i); i++) {
        for (int j = 0; j < more_rows.n_j; j++) {
            c.ptr[i*c.s_i + j*c.s_j] = more_rows.ptr[i*more_rows.s_i + j*more_rows.s_j];
        }
        for (int j = more_rows.n_j; j < c.n_j; j++) {
            c.ptr[i*c.s_i + j*c.s_j] = 0.0;
        }
    }
    for (int i = max(a.n_i, b.n_i); i < c.n_i; i++) {
        for (int j = 0; j < c.n_j; j++) {
            c.ptr[i*c.s_i + j*c.s_j] = 0.0;
        }
    }
};

void weighted_add(const struct mat a, const struct mat b, struct mat c,
                  double wa, double wb) {
    assert(max(a.n_i, b.n_i) <= c.n_i);
    assert(max(a.n_j, b.n_j) <= c.n_j);

    struct mat more_rows = a.n_i > b.n_i ? a : b;
    double w_more_rows   = a.n_i > b.n_i ? wa : wb;
    struct mat more_cols = a.n_j > b.n_j ? a : b;
    double w_more_cols   = a.n_j > b.n_j ? wa : wb; 
    for (int i = 0; i < min(a.n_i, b.n_i); i++) {
        for (int j = 0; j < min(a.n_j, b.n_j); j++) {
            c.ptr[i*c.s_i + j*c.s_j] = wa*a.ptr[i*a.s_i + j*a.s_j] + wb*b.ptr[i*b.s_i + j*b.s_j];       
        } 
        for (int j = min(a.n_j, b.n_j); j < max(a.n_j, b.n_j); j++) {
            c.ptr[i*c.s_i + j*c.s_j] = w_more_cols*more_cols.ptr[i*more_cols.s_i + j*more_cols.s_j];
        }
        for (int j = max(a.n_j, b.n_j); j < c.n_j; j++) {
            c.ptr[i*c.s_i + j*c.s_j] = 0.0;
        }
    } 
    for (int i = min(a.n_i, b.n_i); i < max(a.n_i, b.n_i); i++) {
        for (int j = 0; j < more_rows.n_j; j++) {
            c.ptr[i*c.s_i + j*c.s_j] = w_more_rows*more_rows.ptr[i*more_rows.s_i + j*more_rows.s_j];
        }
        for (int j = more_rows.n_j; j < c.n_j; j++) {
            c.ptr[i*c.s_i + j*c.s_j] = 0.0;
        }
    }
    for (int i = max(a.n_i, b.n_i); i < c.n_i; i++) {
        for (int j = 0; j < c.n_j; j++) {
            c.ptr[i*c.s_i + j*c.s_j] = 0.0;
        }
    }
};

void add_inplace(const struct mat a, struct mat b) {
    int n_i = min(a.n_i, b.n_i);
    int n_j = min(a.n_j, b.n_j);
    for (int i = 0; i < n_i; i++) {
        for (int j = 0; j < n_j; j++) {
            b.ptr[i*b.s_i + j*b.s_j] += a.ptr[i*a.s_i + j*a.s_j];
        }
    }
}

void weighted_add_inplace(const struct mat a, struct mat b, double w) {
    int n_i = min(a.n_i, b.n_i);
    int n_j = min(a.n_j, b.n_j);
    for (int i = 0; i < n_i; i++) {
        for (int j = 0; j < n_j; j++) {
            b.ptr[i*b.s_i + j*b.s_j] += w*a.ptr[i*a.s_i + j*a.s_j];
        }
    }
}

void copy_pad(const struct mat a, struct mat b) {
    int n_i = min(a.n_i, b.n_i);
    int n_j = min(a.n_i, b.n_j);
    for (int i = 0; i < n_i; i++) {
        for (int j = 0; j < n_j; j++) {
            b.ptr[i*b.s_i + j*b.s_j] = a.ptr[i*a.s_i + j*a.s_j];
        }
        for (int j = n_j; j < b.n_j; j++) {
            b.ptr[i*b.s_i + j*b.s_j] = 0.0;
        } 
    }
    for (int i = n_i; i < b.n_i; i++) {
        for (int j = 0; j < b.n_j; j++) {
            b.ptr[i*b.s_i + j*b.s_j] = 0.0;
        }
    }
}

void matmul_strassen(double *A, double *B, double *C, int n) {
    if (n <= threshold) {
        matmul_naive(A, B, C, n);
        return; 
    }
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            C[i*n + j] = 0.0;
        }
    }

    int n_half = n - n/2; 

    struct mat a11 = {A, n, 1, n_half, n_half};
    struct mat a12 = {A + n_half, n, 1, n_half, n - n_half};
    struct mat a21 = {A + n*n_half, n, 1, n - n_half, n_half};
    struct mat a22 = {A + (n+1)*n_half, n, 1, n - n_half, n - n_half}; 

    struct mat b11 = {B, n, 1, n_half, n_half};
    struct mat b12 = {B + n_half, n, 1, n_half, n - n_half};
    struct mat b21 = {B + n*n_half, n, 1, n - n_half, n_half};
    struct mat b22 = {B + (n+1)*n_half, n, 1, n - n_half, n - n_half}; 

    struct mat c11 = {C, n, 1, n_half, n_half};
    struct mat c12 = {C + n_half, n, 1, n_half, n - n_half};
    struct mat c21 = {C + n*n_half, n, 1, n - n_half, n_half};
    struct mat c22 = {C + (n+1)*n_half, n, 1, n - n_half, n - n_half}; 

    double *T_1 = (double*)malloc(n_half*n_half*sizeof(double));
    double *T_2 = (double*)malloc(n_half*n_half*sizeof(double));
    double *T_3 = (double*)malloc(n_half*n_half*sizeof(double));

    struct mat t1 = {T_1, n_half, 1, n_half, n_half};
    struct mat t2 = {T_2, n_half, 1, n_half, n_half};
    struct mat t3 = {T_3, n_half, 1, n_half, n_half};

    //M1
    add(a11, a22, t1);
    add(b11, b22, t2);
    matmul_strassen(T_1, T_2, T_3, n_half);
    add_inplace(t3, c11);
    add_inplace(t3, c22);

    //M2
    add(a21, a22, t1); 
    copy_pad(b11, t2);
    matmul_strassen(T_1, T_2, T_3, n_half);
    add_inplace(t3, c21);
    weighted_add_inplace(t3, c22, -1.0); 

    //M3
    copy_pad(a11, t1);
    weighted_add(b12, b22, t2, 1.0, -1.0);
    matmul_strassen(T_1, T_2, T_3, n_half);
    add_inplace(t3, c12);
    add_inplace(t3, c22);

    //M4
    copy_pad(a22, t1);
    weighted_add(b21, b11, t2, 1.0, -1.0);
    matmul_strassen(T_1, T_2, T_3, n_half);
    add_inplace(t3, c11);
    add_inplace(t3, c21);

    //M5
    add(a11, a12, t1);
    copy_pad(b22, t2);
    matmul_strassen(T_1, T_2, T_3, n_half);
    weighted_add_inplace(t3, c11, -1.0);
    add_inplace(t3, c12);

    //M6
    weighted_add(a21, a11, t1, 1.0, -1.0);
    add(b11, b12, t2); 
    matmul_strassen(T_1, T_2, T_3, n_half);
    add_inplace(t3, c22);

    //M7
    weighted_add(a12, a22, t1, 1.0, -1.0);
    add(b21, b22, t2);
    matmul_strassen(T_1, T_2, T_3, n_half);
    add_inplace(t3, c11);

    free(T_1);
    free(T_2);
    free(T_3);
}
