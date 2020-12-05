#include <math.h>
#include <stdint.h>
#include <stdio.h>

__device__ uint8_t median_pixel(uint8_t *pixels, int stride_H, int stride_W, int size_H, int size_W) {
    int hist[256];
    for (int i = 0; i < 256; i++) {
            hist[i] = 0;
    }
    for (int i = 0; i < size_H; i++) {
        for (int j = 0; j < size_W; j++) {
            uint8_t pix = pixels[i*stride_H + j*stride_W];
            hist[pix] += 1;
        }
    }
    int lower_half_count = 0;
    int i;
    for (i = 0; i < 256; i++) {
        lower_half_count += hist[i];
        if (lower_half_count >= (size_H*size_W-1)/2) break;
    }
    return (uint8_t)i;
}



__global__ void apply_median_filter_kernel(uint8_t *src, uint8_t *dst, int src_H, int src_W, int src_C, int fil_H, int fil_W) {
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

    uint8_t *pixels = shared + threadIdx.x*(blockDim.y + fil_W - 1)*blockDim.z + threadIdx.y*blockDim.z + threadIdx.z;

    uint8_t pixel = median_pixel(pixels, (blockDim.y + fil_W - 1)*blockDim.z, blockDim.z, fil_H, fil_W);
    dst[h_idx*(src_W-fil_W+1)*(src_C) + w_idx*src_C + c_idx] = pixel;
}


__host__ void apply_median_filter(uint8_t *h_src, uint8_t *h_dst, int src_H, int src_W, int src_C, int fil_H, int fil_W, int block_size) {
    uint8_t *d_src;
    uint8_t *d_dst;
    cudaMalloc(&d_src, src_H*src_W*src_C);
    cudaMalloc(&d_dst, (src_H-fil_H+1)*(src_W-fil_W+1)*src_C);
    cudaMemcpy(d_src, h_src, src_H*src_W*src_C, cudaMemcpyHostToDevice);

    int x_blocks = (src_H-fil_H+1);
    x_blocks = x_blocks/block_size + (x_blocks%block_size ? 1 : 0);
    //x_blocks = 1;
    int y_blocks = (src_W-fil_W+1);
    y_blocks = y_blocks/block_size + (y_blocks%block_size ? 1 : 0);
    //y_blocks = 1;

    apply_median_filter_kernel<<<
        dim3(x_blocks,y_blocks,1),
        dim3(block_size,block_size,src_C),
        sizeof(float)*(block_size+fil_H-1)*(block_size+fil_W-1)
    >>>(d_src, d_dst, src_H, src_W, src_C, fil_H, fil_W);

    cudaMemcpy(h_dst, d_dst, (src_H-fil_H+1)*(src_W-fil_W+1)*src_C, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    cudaFree(d_src);
    cudaFree(d_dst);
}
