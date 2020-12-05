#include <math.h>
#include <stdint.h>
#include <stdio.h>


// src_size x 3 -> 256 x N
__global__ void make_partial_histograms_BGR_kernel(uint8_t *src, int *dst, int src_size) {
    int pixel_idx = threadIdx.x + blockIdx.x*blockDim.x;
    int t_idx = threadIdx.x;

    __shared__ int s[256];

    for (int i = 0; t_idx + i*blockDim.x < 256; i++) {
        s[t_idx + i*blockDim.x] = 0;
    }

    __syncthreads();

    if (pixel_idx < src_size) {
        float Y_lin = 0.;
        float weights[3] = {0.0722, 0.7152, 0.2126};
        for (int i = 0; i < 3; i++) {
            float C = (float)src[3*pixel_idx + i]/255;
            float C_lin = C <= 0.04045 ? C/12.92 : pow((C+0.055)/1.055, 2.4);
            Y_lin += C_lin * weights[i];
        } 
        float Y = (Y_lin <= 0.0031308) ? Y_lin*12.92 : 1.055*pow(Y_lin, 1/2.4) - 0.055;
        int lum = round(Y*255);
        atomicAdd(s + lum, 1); 
    }
    __syncthreads();

    for (int i = 0; t_idx + i*blockDim.x < 256; i++) {
        dst[(t_idx + i*blockDim.x)*gridDim.x + blockIdx.x] = s[t_idx + i*blockDim.x];
    }
}


// 256 x N -> 256
__global__ void sum_partial_histograms_kernel(int *src, int *dst, int size) {
    int t_idx = threadIdx.x;

    extern __shared__ int s[];

    for (int i = 0; t_idx + i*blockDim.x < size; i++) {
        s[t_idx + i*blockDim.x] = src[blockIdx.x*size + t_idx + i*blockDim.x];
    }

    __syncthreads();

    while (size > 1) {
        int n_workers = size/2;
        n_workers += size%2 > 0 ? 1 : 0;
        for (int i = 0; t_idx + i*blockDim.x < n_workers; i++) {
            s[t_idx + i*blockDim.x] +=
                t_idx + i*blockDim.x + n_workers < size ?
                s[t_idx + i*blockDim.x + n_workers] :
                0;
        }
        size = n_workers;
        __syncthreads();
    }

    if (t_idx == 0) {
        dst[blockIdx.x] = s[0];
    }
}

__host__ void histogram(uint8_t *h_src, int *h_dst, int src_H, int src_W, int threads_per_block) {
    uint8_t *d_src;
    int *d_dst;
    int *d_partial_hist;
    cudaMalloc(&d_src, src_H*src_W*3);
    cudaMalloc(&d_dst, 256*sizeof(int));
    int partial_hist_size = (src_H*src_W)/threads_per_block;
    partial_hist_size += (src_H*src_W)%threads_per_block > 0 ? 1 : 0;
    cudaMalloc(&d_partial_hist, partial_hist_size*256*sizeof(int));

    cudaMemcpy(d_src, h_src, src_H*src_W*3, cudaMemcpyHostToDevice);

    make_partial_histograms_BGR_kernel<<<
        partial_hist_size,
        threads_per_block,
        256*sizeof(int)
    >>>(d_src, d_partial_hist, src_H*src_W);

    sum_partial_histograms_kernel<<<
        256,
        threads_per_block,
        partial_hist_size*sizeof(int)
    >>>(d_partial_hist, d_dst, partial_hist_size);

    cudaMemcpy(h_dst, d_dst, 256*sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_src);
    cudaFree(d_dst);
    cudaFree(d_partial_hist);

    printf("\n");
}
