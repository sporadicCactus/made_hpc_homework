#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>
#include <math.h>
#include <matutils.h>
#include <matmul_naive.h>
#include <matmul_blas.h>
#include <matmul_strassen.h>

void matmul_bench(int n, int n_trials) {
    double *A = (double*)malloc(n*n * sizeof(double));
    double *B = (double*)malloc(n*n * sizeof(double));
    double *C = (double*)malloc(n*n * sizeof(double));
    double time_mean;
    double time_var;

    printf("-------------------------\n");
    printf("n: %d, trials: %d\n", n, n_trials);

    time_mean = 0.0;
    time_var = 0.0;
    for (int i = 0; i < n_trials; i++) {
        random_fill(A, n);
        random_fill(B, n);
        double start_t = clock();
        matmul_blas(A,B,C,n);
        double end_t = clock();
        double time_diff = (end_t - start_t)/CLOCKS_PER_SEC;
        time_mean += time_diff;
        time_var += time_diff*time_diff;
    }
    time_mean = time_mean / n_trials;
    time_var = (time_var / n_trials - time_mean*time_mean)*n_trials/(n_trials - 1);
    printf("\nBlas matmul:\n");
    printf("Mean: %f seconds, deviation: %f seconds\n", time_mean, sqrt(time_var));

    time_mean = 0.0;
    time_var = 0.0;
    for (int i = 0; i < n_trials; i++) {
        random_fill(A, n);
        random_fill(B, n);
        double start_t = clock();
        matmul_naive(A,B,C,n);
        double end_t = clock();
        double time_diff = (end_t - start_t)/CLOCKS_PER_SEC;
        time_mean += time_diff;
        time_var += time_diff*time_diff;
    }
    time_mean = time_mean / n_trials;
    time_var = (time_var / n_trials - time_mean*time_mean)*n_trials/(n_trials - 1);
    printf("\nNaive matmul:\n");
    printf("Mean: %f seconds, deviation: %f seconds\n", time_mean, sqrt(time_var));

    time_mean = 0.0;
    time_var = 0.0;
    for (int i = 0; i < n_trials; i++) {
        random_fill(A, n);
        random_fill(B, n);
        double start_t = clock();
        matmul_strassen(A,B,C,n);
        double end_t = clock();
        double time_diff = (end_t - start_t)/CLOCKS_PER_SEC;
        time_mean += time_diff;
        time_var += time_diff*time_diff;
    }
    time_mean = time_mean / n_trials;
    time_var = (time_var / n_trials - time_mean*time_mean)*n_trials/(n_trials - 1);
    printf("\nStrassen matmul:\n");
    printf("Mean: %f seconds, deviation: %f seconds\n", time_mean, sqrt(time_var));

    printf("-------------------------\n");
}

void printmat(double *A, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            printf("%+4.2f ", A[n*i + j]);
        }
        printf("\n");
    }
    printf("\n");
}

int main() {
    int testsize = 333;

    double *A = (double*)malloc(testsize*testsize * sizeof(double));
    double *B = (double*)malloc(testsize*testsize * sizeof(double));
    double *C_naive = (double*)malloc(testsize*testsize * sizeof(double));
    double *C_blas = (double*)malloc(testsize*testsize * sizeof(double));
    double *C_strassen = (double*)malloc(testsize*testsize * sizeof(double));

    random_fill(A, testsize);
    random_fill(B, testsize);

    matmul_naive(A,B,C_naive,testsize);
    matmul_blas(A,B,C_blas,testsize);
    matmul_strassen(A,B,C_strassen,testsize);

    double acc = 0.0;
    for (int i = 0; i < testsize; i++) {
        for (int j = 0; j < testsize; j++) {
            //acc += abs(C_naive[testsize*i + j] - C_blas[testsize*i + j]); 
            double temp = (C_naive[testsize*i + j] - C_blas[testsize*i + j]);
            acc += temp*temp;
        }
    }
    acc = sqrt(acc) / testsize / testsize;

    if (acc < 1e-10) {
        printf("Naive matmul: Ok\n");
    } else {
        printf("Naive matmul: failed\n");
    }

    acc = 0.0;
    for (int i = 0; i < testsize; i++) {
        for (int j = 0; j < testsize; j++) {
            double temp = C_strassen[testsize*i + j] - C_blas[testsize*i + j];
            acc += temp*temp;
        }
    }
    acc = sqrt(acc) / testsize / testsize;
    if (acc < 1e-10) {
        printf("Strassen matmul: Ok\n");
    } else {
        printf("Strassen matmul: failed\n");
    }

    free(A);
    free(B);
    free(C_naive);
    free(C_blas);
    free(C_strassen);

    matmul_bench(512, 16);
    matmul_bench(1024, 8);
    matmul_bench(2048, 4);
}
