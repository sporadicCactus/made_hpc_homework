#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

float dotprod(float * a, float * b, size_t N)
{
    int i, tid;
    float sum = 0.0;


    #pragma omp parallel for reduction(+:sum)
    for (i = 0; i < N; ++i)
    {
        tid = omp_get_thread_num();
        sum += a[i] * b[i];
        printf("tid = %d i = %d\n", tid, i);
    }

    return sum;
}

float dotprod_single(float *a, float *b, size_t N)
{
    int i, tid;
    float sum = 0.0;

    for (i = 0; i < N; ++i)
    {
        tid = omp_get_thread_num();
        sum += a[i] * b[i];
        printf("tid = %d i = %d\n", tid, i);
    }

    return sum;
}

int main (int argc, char *argv[])
{
    const size_t N = 100;
    int i;
    float sum, single_sum;
    float a[N], b[N];


    for (i = 0; i < N; ++i)
    {
        a[i] = b[i] = (double)i;
    }

    single_sum = dotprod_single(a, b, N);
    sum = dotprod(a, b, N);

    printf("Sum = %f, no parallel\n",single_sum);
    printf("Sum = %f\n",sum);

    return 0;
}
