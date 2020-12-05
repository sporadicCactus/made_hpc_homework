#include <math.h>
#include <stdint.h>
#include <stdio.h>


__global__ void apply_linear_filter_kernel(uint8_t *src, uint8_t *dst, float *fil, int src_H, int src_W, int src_C, int fil_H, int fil_W) {
    int h_idx = threadIdx.x + blockIdx.x*blockDim.x;
    int w_idx = threadIdx.y + blockIdx.y*blockDim.y;
    int c_idx = threadIdx.z + blockIdx.z*blockDim.z;

    if ((h_idx >= (src_H-fil_H+1))|(w_idx >= (src_W-fil_W+1))|(c_idx >= src_C)) return;

    extern __shared__ int s[];

    uint8_t *shared = (uint8_t*)s;

    int src_idx = h_idx*src_W*src_C + w_idx*src_C + c_idx;
    for (int i = 0; (h_idx + i*blockDim.x < src_H)&(threadIdx.x + i*blockDim.x < fil_H + blockDim.x - 1); i++) {
        for (int j = 0; (w_idx + j*blockDim.y < src_W)&(threadIdx.y + j*blockDim.y < fil_W + blockDim.y - 1); j++) {
            int shared_idx = (threadIdx.x + i*blockDim.x)*(blockDim.y + fil_W - 1)*blockDim.z + (threadIdx.y + j*blockDim.y)*blockDim.z + threadIdx.z;
            shared[shared_idx] = src[src_idx + i*blockDim.x*src_W*src_C + j*blockDim.y*src_C];
        }
    }

    float new_value = 0.0;
    float filter_sum = 0.0;
    for (int i = 0; i < fil_H; i++) {
        for (int j = 0; j < fil_W; j++) {
            int shared_idx = (threadIdx.x + i)*(blockDim.y + fil_W - 1)*blockDim.z + (threadIdx.y + j)*blockDim.z + threadIdx.z;
            new_value += (float)shared[shared_idx]*fil[i*fil_W + j];
            filter_sum += fil[i*fil_W + j];
        }
    }
    new_value = new_value / filter_sum;
    dst[h_idx*(src_W-fil_W+1)*(src_C) + w_idx*src_C + c_idx] = (uint8_t)round(new_value);
}


__host__ void apply_linear_filter(uint8_t *h_src, uint8_t *h_dst, float *h_fil, int src_H, int src_W, int src_C, int fil_H, int fil_W, int block_size) {
    uint8_t *d_src;
    uint8_t *d_dst;
    float *d_fil;
    cudaMalloc(&d_src, src_H*src_W*src_C);
    cudaMalloc(&d_dst, (src_H-fil_H+1)*(src_W-fil_W+1)*src_C);
    cudaMalloc(&d_fil, fil_H*fil_W*sizeof(float));
    cudaMemcpy(d_src, h_src, src_H*src_W*src_C, cudaMemcpyHostToDevice);
    cudaMemcpy(d_fil, h_fil, fil_H*fil_W*sizeof(float), cudaMemcpyHostToDevice);

    int x_blocks = (src_H-fil_H+1);
    x_blocks = x_blocks/block_size + (x_blocks%block_size ? 1 : 0);
    int y_blocks = (src_W-fil_W+1);
    y_blocks = y_blocks/block_size + (y_blocks%block_size ? 1 : 0);

    apply_linear_filter_kernel<<<
        dim3(x_blocks,y_blocks,1),
        dim3(block_size,block_size,src_C),
        sizeof(float)*(block_size+fil_H-1)*(block_size+fil_W-1)
    >>>(d_src, d_dst, d_fil, src_H, src_W, src_C, fil_H, fil_W);

    cudaMemcpy(h_dst, d_dst, (src_H-fil_H+1)*(src_W-fil_W+1)*src_C, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    cudaFree(d_src);
    cudaFree(d_dst);
    cudaFree(d_fil);
}
