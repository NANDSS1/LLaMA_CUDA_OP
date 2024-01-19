
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdint.h>
#include <stdio.h>
#include <atomic>
#include <assert.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <bits/stdc++.h>

static __global__ void rms_norm_f32(const float * x, float * dst, const int ncols, const float eps) {
    const int row = blockIdx.x*blockDim.y + threadIdx.y;//计算行号，第几行就代表第几个sequence
    const int tid = threadIdx.x;//计算block内的id

    float tmp = 0.0f; // partial sum for thread in warp，定义寄存器

    for (int col = tid; col < ncols; col += WARP_SIZE) {
        const float xi = x[row*ncols + col];//从显存里面load数据到寄存器里面
        tmp += xi * xi;//x^2，每个thread计算hidden_dim/WARP_SIZE个x^2的和。
    }

    // sum up partial sums
#pragma unroll//循环展开
    for (int mask = 16; mask > 0; mask >>= 1) {
        tmp += __shfl_xor_sync(0xffffffff, tmp, mask, 32);//对warp内做一个reduce，xor这种蝶形reduece。
    }

    const float mean = tmp / ncols;
    const float scale = rsqrtf(mean + eps);//eps是防止分母为0

    for (int col = tid; col < ncols; col += WARP_SIZE) {
        dst[row*ncols + col] = scale * x[row*ncols + col];//逐元素相乘这个scale
    }
}

int main(){

}
