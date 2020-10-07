#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <matutils.h>

void zero_fill(double *M, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            M[N*i + j] = 0.0;
        }
    }
}

void random_fill(double *M, int N) {
    srand(clock());
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            M[N*i + j] = 2*((double)rand() / RAND_MAX) - 1.0;
        }
    }
}
