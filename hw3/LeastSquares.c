#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#include <time.h>
#include <math.h>

#define M_PI 3.14159265358979323846


void generate_data(float *x_arr, float *y_arr, float a, float b, float noise_mag, int N) {
    float eps = 1e-20;
    srand(time(NULL));

    int n_threads = omp_get_max_threads();
    int chunk_size = (N / n_threads) + ((N % n_threads) > 0 ? 1 : 0); 

    unsigned int states[n_threads];
    for (int i = 0; i < n_threads; i++) {
        states[i] = (unsigned int)rand();
    }

    #pragma omp parallel num_threads(n_threads)
    {
        int i = omp_get_thread_num();
        unsigned int state = states[i];
        int start = i*chunk_size;
        int end = (start + chunk_size > N ? N : start + chunk_size);
        for (int j = start; j < end; j++) {
            float r1 = (float)rand_r(&state) / RAND_MAX;
            float r2 = (float)rand_r(&state) / RAND_MAX;

            float r = sqrt(log(1/(1 - r1 + eps)));

            float noise = noise_mag * r * cos(2*M_PI*r2);

            float x = 2*((float)rand_r(&state) / RAND_MAX) - 1;

            float y = a*x + b + noise; 

            x_arr[j] = x;
            y_arr[j] = y;
        }
    }
}

struct coefs {float a; float b;};

struct coefs least_squares(float *x_arr, float *y_arr, int N) {
    float x_mean = 0.;
    float y_mean = 0.;
    float x2_mean = 0.;
    float xy_mean = 0.;

    #pragma omp parallel for reduction(+:x_mean,y_mean,x2_mean,xy_mean)
    for (int i = 0; i < N; i++) {
        float x = x_arr[i];
        float y = y_arr[i];

        x_mean += x;
        y_mean += y;
        x2_mean += x*x;
        xy_mean += x*y;
    }

    x_mean /= N;
    y_mean /= N;
    x2_mean /= N;
    xy_mean /= N;

    float a = (xy_mean - x_mean*y_mean) / (x2_mean - x_mean*x_mean);
    float b = y_mean - a*x_mean;

    struct coefs c = {a, b};
    return c;
}

int main() {
    float a = 3.3;
    float b = -1.4;
    float noise_mag = 1.0;
    float start_t, end_t, gen_time, ls_time;

    int N = 1e+6;

    float *x_arr = (float*)malloc(N*sizeof(float));
    float *y_arr = (float*)malloc(N*sizeof(float));

    start_t = omp_get_wtime();
    generate_data(x_arr, y_arr, a, b, noise_mag, N);
    end_t = omp_get_wtime();
    gen_time = end_t - start_t;

    start_t = omp_get_wtime();
    struct coefs reg_coefs = least_squares(x_arr, y_arr, N);
    end_t = omp_get_wtime();
    ls_time = end_t - start_t;

    printf("Samples: %d, noise magnitude: %4.2f, generated in %6.5f seconds\n", N, noise_mag, gen_time);
    printf("Regression took %6.5f seconds\n", ls_time);
    printf("True coefficients: %4.2f, %4.2f\n", a, b);
    printf("LS   coefficients: %4.2f, %4.2f\n", reg_coefs.a, reg_coefs.b);

    return 0;
}
