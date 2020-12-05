#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#include <time.h>
#include <math.h>

int CHUNK = 256;

void gauss_seidel_single(float *A, float *b, float *x, int N, int max_it, float eps) {
    float *C = (float*)malloc(N*N*sizeof(float));
    float *d = (float*)malloc(N*sizeof(float));
    float *diff = (float*)malloc(N*sizeof(float));

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            C[i*N + j] = (i != j ? -A[i*N + j]/A[i*N + i] : 0.); 
        }
    }

    for (int i = 0; i < N; i++) {
        d[i] = b[i]/A[i*N + i];
    }

    for (int i = 0; i < N; i++) {
        x[i] = 0.;
    }

    for (int it = 0; it < max_it; it++) {
        for (int i = 0; i < N; i++) {
            float new_x = 0.;
            for (int j = 0; j < N; j++) {
                new_x += C[i*N + j] * x[j];
            }
            new_x += d[i];
            diff[i] = new_x - x[i];
            x[i] = new_x;
        }

        float err = 0.;
        for (int i = 0; i < N; i++) {
            err += fabs(diff[i]);
        }
        if (err < eps) break;
    }

    free(C);
    free(d);
}

void gauss_seidel_multi(float *A, float *b, float *x, int N, int max_it, float eps) {
    float *C = (float*)malloc(N*N*sizeof(float));
    float *C_T = (float*)malloc(N*N*sizeof(float));
    float *d = (float*)malloc(N*sizeof(float));
    float *x_t = (float*)malloc(N*sizeof(float));

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float c = (i != j ? -A[i*N + j]/A[i*N + i] : 0.); 
            C[i*N + j] = c;
            C_T[j*N + i] = c;
        }
    }

    for (int i = 0; i < N; i++) {
        d[i] = b[i]/A[i*N + i];
    }

    for (int i = 0; i < N; i++) {
        x[i] = 0.;
    }

    for (int it = 0; it < max_it; it++) {
        #pragma omp parallel for
        for (int i = 0; i < N; i++) {
            float acc = 0.;
            for (int j = i; j < N; j++) {
                acc += C[i*N + j] * x[j];
            }
            x_t[i] = acc + d[i];
        }
        
        float err = 0.;
        float new_x, old_x;
        for (int i = 0; i < N; i+=CHUNK) {
            int M = i + CHUNK > N ? N : i + CHUNK;
            for (int j = i; j < M; j++) {
                old_x = x[j];
                new_x = x_t[j];
                for (int k = j; k < M; k++) {
                    x_t[k] += C_T[j*N + k] * new_x;
                }
                err += fabs(new_x - old_x);
                x[j] = new_x;
            }

            #pragma omp parallel for
            for (int k = M; k < N; k++) {
                float acc = 0.;
                for (int j = i; j < M; j++) {
                    acc += C[k*N + j] * x[j]; 
                }
                x_t[k] += acc;
            }
        }

        if (err < eps) break;
    }

    free(C);
    free(d);
}

int N = 2000;
int max_it = 1000;
float eps = 1e-7;

int main() {

    float *A = (float*)malloc(N*N*sizeof(float));
    float *b = (float*)malloc(N*sizeof(float));
    float *x_single = (float*)malloc(N*sizeof(float));
    float *x_multi = (float*)malloc(N*sizeof(float));

    unsigned int state = 0;

    for (int i = 0; i < N; i++) {
        for (int j = i; j < N; j++) {
            float a = (i!=j ? -0.5 : 100.) + (float)rand_r(&state) / RAND_MAX;
            A[i*N + j] = a;
            A[j*N + i] = a;
        }
    }

    for (int i = 0; i < N; i++) {
        b[i] = (float)rand_r(&state) / RAND_MAX;
    }

    float start_t, end_t, err;
    start_t = omp_get_wtime();
    gauss_seidel_single(A, b, x_single, N, max_it, eps);
    end_t = omp_get_wtime();

    err = 0.;
    for (int i = 0; i < N; i++) {
        float err_i = 0.;
        for (int j = 0; j < N; j++) {
            err_i += A[i*N + j] * x_single[j];
        }
        err_i -= b[i];
        err += fabs(err_i);
    }

    printf("Single: %f seconds, error = %.5f\n", end_t - start_t, err);

    start_t = omp_get_wtime();
    gauss_seidel_multi(A, b, x_multi, N, max_it, eps);
    end_t = omp_get_wtime();

    err = 0.;
    for (int i = 0; i < N; i++) {
        float err_i = 0.;
        for (int j = 0; j < N; j++) {
            err_i += A[i*N + j] * x_multi[j];
        }
        err_i -= b[i];
        err += fabs(err_i);
    }

    printf("Multi: %f seconds, error = %.5f\n", end_t - start_t, err);

    return 0;
}
