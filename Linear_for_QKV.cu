#include <iostream>
#include <cublas_v2.h>
#define M 2
#define K 4096
#define N 4096
//seq hidden_dim hidden_dim
//MxK(input_tensor) KxN(weight) MxN(output_tensor) 

// 定义一个辅助函数，用于检查 CUDA 调用是否成功
void check_cuda_error(cudaError_t err, const char * msg){
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << msg << ": " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}


template<typename T>
void cpu_gemm(const T* A, const T* B, const T* C,T* dst, const T alpha,const T beta, int&& m, int&& k, int&& n){
    for(int i = 0; i < m; i++){
        //C[i][j]
        //这个i遍历，MxK的矩阵的每一行
        for(int j = 0; j < n; j++ ){
            //第一个j遍历，KxN矩阵的每一列
            dst[i*n+j] = 0;  //初始化dst[i*N+j]为0
            for(int kk = 0; kk < k; kk++){
                dst[i*n+j] += alpha*A[i*k+kk]*B[j+N*kk];//之前这里写错了 后面那个注释就是错的//dst[i*n+j] += alpha*A[i*k+kk]*B[j+k*kk];
            }
            dst[i*n+j] += beta*C[i*n+j]; //内部循环结束后添加beta*C[i*N+j]
        }
    }
}

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



int main() {
    cublasHandle_t handle;  // CUBLAS context
    float* h_A;  // Host arrays
    float* h_B;
    float* h_C;
    float* cpu_dst;
    float* d_A = 0;  // Device arrays
    float* d_B = 0;
    float* d_C = 0;
    float* dst;
    float low = 10.0;
    float high = 100.0;
    float alpha = low + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX)) *(high - low);
    float beta =  0;

    // Allocate and fill host memory
    h_A = new float[M*K];
    h_B = new float[K*N];
    h_C = new float[M*N];
    cpu_dst = new float[M*N];
    dst = new float[M*N];

    
    for(int i = 0; i < M*K; i++) {
        h_A[i] = low + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX)) *(high - low);
    }
    
    for(int i = 0; i < K*N; i++) {
        h_B[i] = low + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX)) *(high - low);
    }
    
    for(int i = 0; i < M*N; i++) {
        h_C[i] = 0;
    }

    cpu_gemm(h_A, h_B, h_C, cpu_dst, alpha, beta, M, K, N);

    // Initialize CUBLAS
    cublasCreate(&handle);
    
    // Allocate device memory
    check_cuda_error(cudaMalloc((void**)&d_A, M * K * sizeof(d_A[0])),"cudaMalloc d_A");
    check_cuda_error(cudaMalloc((void**)&d_B, K * N * sizeof(d_B[0])),"cudaMalloc d_B");
    check_cuda_error(cudaMalloc((void**)&d_C, M * N * sizeof(d_C[0])),"cudaMalloc d_C");
    // Copy data to the device
    check_cuda_error(cudaMemcpy(d_A,h_A,M * K * sizeof(d_A[0]),cudaMemcpyHostToDevice),"cudaMemcpy h_A 2 d_A");
    check_cuda_error(cudaMemcpy(d_B,h_B,K * N * sizeof(d_B[0]),cudaMemcpyHostToDevice),"cudaMemcpy h_B 2 d_B");
    check_cuda_error(cudaMemcpy(d_C,h_C,M * N * sizeof(d_C[0]),cudaMemcpyHostToDevice),"cudaMemcpy h_C 2 d_C");
    // Copy arrays to device

    // Perform operation with cublas
    //cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C, N);
    cublasSgemm(handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        N, M, K,
        &alpha,
        d_B,            // cuda device上的B矩阵，n * k * sizeof(float)
        N,              // B的leading_edge 是 n
        d_A,            // k * m * sizeof(float)
        K,
        &beta, d_C,
        N);

    check_cuda_error(cudaMemcpy(dst,d_C,M*N*sizeof(d_C[0]),cudaMemcpyDeviceToHost),"cudaMemcpy d_dst 2 dst");

    float max_err = 0.0f;
    float avg_err = 0.0f;
    compute_error(dst,cpu_dst,M * N,&max_err,&avg_err);
    std::cout<<"max error"<<max_err<<std::endl;
    std::cout<<"avg error"<<avg_err<<std::endl;

    // Destroy CUBLAS context
    cublasDestroy(handle);

    // Deallocate device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Deallocate host memory
    delete h_A;
    delete h_B;
    delete h_C;
    delete dst;
    delete cpu_dst;

    return 0;
}