#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

int NUM_THROWS = 1e+8;

int main() {
    int threads_available = omp_get_max_threads();
    int throws_per_thread = NUM_THROWS / threads_available + (NUM_THROWS % threads_available ? 1 : 0);

    printf("threads available: %d\n", threads_available);
    printf("throws_per_thread: %d\n", throws_per_thread);

    srand(time(NULL));

    unsigned int states[threads_available];
    for (int i = 0; i < threads_available; i++) {
        states[i] = (unsigned int)rand();
    }

    unsigned int acc = 0;
    float start_t = omp_get_wtime();
    #pragma omp parallel reduction(+:acc) num_threads(threads_available)
    {
        int i = omp_get_thread_num();
        unsigned int state = states[i];
        unsigned int internal_acc = 0;
        for (int j = 0; j < throws_per_thread; j++) {
            float x = (float)rand_r(&state)/RAND_MAX;
            x = 2*x - 1;
            float y = (float)rand_r(&state)/RAND_MAX;
            y = 2*y - 1;

            float r2 = x*x + y*y;
            internal_acc += (r2 <= 1. ? 1 : 0);
        }
        acc += internal_acc;
    }
    float end_t = omp_get_wtime();

    float pi = 4*((float)acc) / (threads_available * throws_per_thread);

    float dev = (4*pi - pi*pi) / (threads_available * throws_per_thread);
    dev = sqrt(dev);

    printf("Done %d experiments in %4.2f seconds\n", threads_available*throws_per_thread, end_t-start_t);
    printf("pi = %f \u00b1 %f \n", pi, dev);

    return 0;
}
