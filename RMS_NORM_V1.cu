#include <iostream>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>

// 定义一些常量
#define NROWS 1 // sequence数量
#define NCOLS 4096 // hidden_dim
#define WARP_SIZE 32 // warp 大小,一个seq用一个warp进行reduce，用warp_shfl即可。
#define EPS 1e-6 // 防止除零的小数
#define MAX 10

// 定义 RMSNorm 核函数
__global__ void rms_norm_f32(const float * x, float * dst, const int ncols, const float eps) {
    const int row = blockIdx.y*blockDim.y + threadIdx.y;//计算行号，第几行就代表第几个sequence，Idx是从0开始
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

// 定义 CPU 上的 RMSNorm 算子
void rms_norm_f32_cpu(const float * x, float * dst, const int nrows, const int ncols, const float eps) {
    for (int i = 0; i < nrows; i++) { // 遍历每一行
        float sum = 0.0f; // 计算平方和
        for (int j = 0; j < ncols; j++) {
            sum += x[i*ncols + j] * x[i*ncols + j];
        }
        float mean = sum / ncols; // 计算均方值
        float scale = 1.0f / sqrtf(mean + eps); // 计算归一化因子
        for (int j = 0; j < ncols; j++) {
            dst[i*ncols + j] = scale * x[i*ncols + j]; // 逐元素相乘
        }
    }
}

// 定义一个辅助函数，用于检查 CUDA 调用是否成功
void check_cuda_error(cudaError_t err, const char * msg) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << msg << ": " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

// 定义一个辅助函数，用于生成随机的浮点数数组
void random_float_array(float * arr, const int size) {
    srand(time(NULL)); // 设置随机数种子
    for (int i = 0; i < size; i++) {
        arr[i] = (float)rand() / MAX; // 生成 [0, 1] 之间的随机数
    }
}

// 定义一个辅助函数，用于计算两个浮点数数组的误差
void compute_error(const float * a, const float * b, const int size, float * max_err, float * avg_err) {
    *max_err = 0.0f; // 最大误差
    *avg_err = 0.0f; // 平均误差
    for (int i = 0; i < size; i++) {
        float err = fabs(a[i] - b[i]); // 计算每个元素的差的绝对值
        *max_err = fmax(*max_err, err); // 更新最大误差
        *avg_err += err; // 累加平均误差
    }
    *avg_err /= size; // 计算平均误差
}

// 主函数
int main() {
    // 生成测试数据
    float * x = new float[NROWS * NCOLS]; // 输入数组
    float * dst = new float[NROWS * NCOLS]; // 输出数组
    float * cpu_dst = new float[NROWS * NCOLS]; // CPU 上的输出数组
    random_float_array(x, NROWS * NCOLS); // 随机生成输入数组

    // 在 CPU 上实现 RMSNorm 算子
    rms_norm_f32_cpu(x, cpu_dst, NROWS, NCOLS, EPS);

    // 在 GPU 上调用 RMSNorm 核函数
    float * d_x, * d_dst; // 设备内存上的输入和输出数组
    check_cuda_error(cudaMalloc(&d_x, NROWS * NCOLS * sizeof(float)), "cudaMalloc d_x"); // 分配设备内存
    check_cuda_error(cudaMalloc(&d_dst, NROWS * NCOLS * sizeof(float)), "cudaMalloc d_dst"); // 分配设备内存
    check_cuda_error(cudaMemcpy(d_x, x, NROWS * NCOLS * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy x to d_x"); // 复制输入数组到设备内存
    dim3 grid(1,NROWS,1); // 定义网格维度
    dim3 block(WARP_SIZE,1,1); // 定义块维度
    rms_norm_f32<<<grid, block>>>(d_x, d_dst, NCOLS, EPS); // 启动核函数
    check_cuda_error(cudaGetLastError(), "rms_norm_f32 kernel launch"); // 检查核函数是否成功
    check_cuda_error(cudaDeviceSynchronize(), "cudaDeviceSynchronize"); // 同步设备
    check_cuda_error(cudaMemcpy(dst, d_dst, NROWS * NCOLS * sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy d_dst to dst"); // 复制
    /*cudaError_t cudaMemcpy(void *dst, const void *src, size_t count, cudaMemcpyKind kind),第一个目的地址，第二个源地址
    cuda api的返回值都是cudaError_t
    */

    float max_err = 0.0f;
    float avg_err = 0.0f;
    compute_error(dst,cpu_dst,NROWS * NCOLS,&max_err,&avg_err);
    std::cout<<"max error"<<max_err<<std::endl;
    std::cout<<"avg error"<<avg_err<<std::endl;
}