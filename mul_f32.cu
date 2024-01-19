#include <iostream>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>

// 定义一些常量
#define NROWS 100 // sequence数量
#define NCOLS 4096 // hidden_dim
#define WARP_SIZE 32 // warp 大小,一个seq用一个warp进行reduce，用warp_shfl即可。
#define EPS 1e-6 // 防止除零的小数
#define MAX 10000
#define CUDA_MUL_BLOCK_SIZE 256


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
static void mul_f32_cpu(const float* x,const float* y,float* dst,const int kx,const int ky){
    for(int i = 0;i < kx;i++){
        dst[i] = x[i]*y[i%ky];
    }
}

static __global__ void mul_f32(const float * x, const float * y, float * dst, const int kx, const int ky) {
    const int i = blockDim.x*blockIdx.x + threadIdx.x;

    if (i >= kx) {
        return;
    }
    dst[i] = x[i] * y[i%ky];//连续的，每次计算到1*hidden_dim个元素，y的索引就要重置，所以用i%ky索引y的元素。
}

static void mul_f32_cuda(const float * x, const float * y, float * dst, const int kx, const int ky, cudaStream_t stream) {
    const int num_blocks = (kx + CUDA_MUL_BLOCK_SIZE - 1) / CUDA_MUL_BLOCK_SIZE;
    mul_f32<<<num_blocks, CUDA_MUL_BLOCK_SIZE, 0, stream>>>(x, y, dst, kx, ky);
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

int main(){
    // 生成测试数据
    float* x = new float[NROWS * NCOLS]; // 输入数组
    float* gamma = new float[NCOLS];
    float* dst = new float[NROWS * NCOLS]; // 输出数组
    float* cpu_dst = new float[NROWS * NCOLS]; // CPU 上的输出数组
    random_float_array(x, NROWS * NCOLS); // 随机生成输入数组
    random_float_array(gamma,NCOLS);
    mul_f32_cpu(x,gamma,cpu_dst,NROWS*NCOLS,NCOLS);

    float * d_x, * d_dst,* d_gamma; // 设备内存上的输入和输出数组
    check_cuda_error(cudaMalloc(&d_x,sizeof(float)*NROWS * NCOLS),"cudaMalloc d_x");
    check_cuda_error(cudaMalloc(&d_dst,sizeof(float)*NROWS * NCOLS),"cudaMalloc d_dst");
    check_cuda_error(cudaMalloc(&d_gamma,sizeof(float) * NCOLS),"cudaMalloc d_gamma");
    check_cuda_error(cudaMemcpy(d_x, x, NROWS * NCOLS * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy x to d_x");
    check_cuda_error(cudaMemcpy(d_gamma, gamma, NCOLS * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy gamma to d_gamma");
    mul_f32_cuda(d_x,d_gamma,d_dst,NROWS * NCOLS,NCOLS,0);
    check_cuda_error(cudaMemcpy(dst,d_dst,NROWS*NCOLS*sizeof(float),cudaMemcpyDeviceToHost),"cudaMemcpy d_dst to dst");

    float max_err = 0.0f;
    float avg_err = 0.0f;
    compute_error(dst,cpu_dst,NROWS * NCOLS,&max_err,&avg_err);
    std::cout<<"max error"<<max_err<<std::endl;
    std::cout<<"avg error"<<avg_err<<std::endl;

}


    //mul_f32_cuda(src0_ddf_i, src1_ddf_i, dst_ddf_i, ne00*i01_diff, ne10*ne11, cudaStream_main);//输入input_tensor和权重tensor，还有output_tensor
    //
    //ne00*i01_diff代表，ne00代表tensor0在0维方向的维度(seq>=1)，i01_diff代表tensor0在1维的索引差(hidden_dim)。ne10代表tensor1在0维方向的维度(1)，ne11代表tensor1在1维方向的维度(hidden_dim)。




