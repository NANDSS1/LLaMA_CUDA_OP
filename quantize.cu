#include <iostream>
#include <cublas_v2.h>
#define QK8_1 32
#define BLOCK_DIM 32
#define HIDDEN_DIM 32
#define WARP_SIZE 32
#define SEQ_LEN 1

#ifdef GGML_CUDA_F16
typedef half dfloat; // dequantize float
typedef half2 dfloat2;
#else
typedef float dfloat; // dequantize float
typedef float2 dfloat2;
#endif //GGML_CUDA_F16

//用以计算量化
// 定义一个辅助函数，用于检查 CUDA 调用是否成功
void check_cuda_error(cudaError_t err, const char * msg){
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << msg << ": " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

typedef struct {
    dfloat2    ds;              // ds.x = delta, ds.y = sum ，这里的ds存了两个dfloat2 第一个half是scale，第二half是sum
    int8_t  qs[QK8_1];      // quants 量化的weight数据,每QK8_0个数共享一个scale
} block_q8_1;

static __global__ void quantize_q8_1(const dfloat * __restrict__ x,void * __restrict__ vy, const int kx, const int kx_padded){
    const int ix = blockDim.x*blockIdx.x + threadIdx.x; //0-4096
    if (ix >= kx_padded) {
        return;
    }
    //i_padded是每个线程的全局索引
    //kx_padded是填充之后的hidden_dim，填充的元素一律当做0
    //kx是未填充的hidden_dim，填充的原因是有可能未满足warp的倍数。
    const int iy = blockDim.y*blockIdx.y + threadIdx.y; //如果seq_num=1,那么iy就等于0,表示线程在y方向的全局id
    const int i_padded = iy*kx_padded + ix;//i_padded是线程的全局id //kx_padded是hidden_dim。
    block_q8_1 * y = (block_q8_1 *) vy;//vy是量化的结构体，空指针传进来做强转。
    const int ib = i_padded / QK8_1;//ib是全局warp索引，每一个warp共享一组scale为一组。//相同scale的weight为一组，这个是组的索引。0-31个weight索引是0,32-63索引是1。这里是全局的索引。表示当前线程处理的元素在哪个全局warp。
    const int iqs = i_padded % QK8_1; // quant index iqs计算的就是当前数据所在结构体内部的index，也就是warp内的元素索引。
    const dfloat xi = ix < kx ? x[iy*kx + ix] : 0.0f; //填充的全部赋值为0,kx是hidden_dim的维度大小。如果同时操作的sqe有多个，iy表明当前thread处于第几个seq。
    dfloat absMax = fabsf(xi); // 当前数据的绝对值
    dfloat sum = xi; 

    //一个block内部既做归约求和也做归约求最大值
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        absMax = fmaxf(absMax, __shfl_xor_sync(0xffffffff, absMax, mask, 32));
        sum += __shfl_xor_sync(0xffffffff, sum, mask, 32);
    }
    //套用均匀对称量化的量化公式
    //q = round(clip(r_i /scale,Q_{min},Q_{max}))
    //scale = scale = max(abs(weight)) / 127
    dfloat scale = absMax/127;
    const int8_t q = (absMax == 0.0f ? 0 : roundf(xi / scale));//转整形
    printf("%d-",q);
    //存储量化后的值
    y[ib].qs[iqs] = q;
    printf("%d-",y[ib].qs[iqs]);

    if(iqs > 0){
        return;
    }
    //只用iqs==0的线程将scale和sum写回
    y[ib].ds.x = scale;
    y[ib].ds.y = sum;
}

static __host__ void launch_kernel(const dfloat* d_x,void* d_vy,const int seqNum,const int hiddenDim,const int hiddenDim_padded){
    dim3 grid((hiddenDim-1/BLOCK_DIM)+BLOCK_DIM,seqNum,1);
    dim3 block(BLOCK_DIM,1,1);
    quantize_q8_1<<<grid,block>>>(d_x,d_vy,hiddenDim,hiddenDim_padded);
}

/*
static __host__ void quantize_q8_1_cpu(const dfloat* x,void* h_vy,const int seqNum,const int hiddenDim,const int hiddenDim_padded){
    dfloat scale = 0;
    dfloat sum = 0;
    dfloat absMax = 0; 
    block_q8_1 * y = (block_q8_1 *) h_vy;
    for(int i = 0;i <seqNum*hiddenDim;i++){
        if(i%WARP_SIZE == 0) {
            scale = absMax/127;
            y[i/WARP_SIZE].ds.x = scale;//i/WARP_SIZE是第几个warp
            y[i/WARP_SIZE].ds.y = sum;
            for(int j = 0;j < WARP_SIZE;j++){//j记录的warp内id
                const int8_t q = (absMax == 0.0f ? 0 : roundf(x[i+j] / scale));
                y[i/WARP_SIZE].qs[j] = q;
            }
            sum = 0;
            scale = 0;
            absMax = 0;
        }
        absMax = max(fabsf(x[i]),absMax);
        sum += x[i];
    }

}*/

static __host__ void quantize_q8_1_cpu(const dfloat* x,void* h_vy,const int seqNum,const int hiddenDim,const int hiddenDim_padded){
    printf("\n");
    dfloat scale = 0;
    dfloat sum = 0;
    dfloat absMax = 0; 
    block_q8_1 * y = (block_q8_1 *) h_vy;
    for(int i = 0; i < seqNum * hiddenDim; ++i) {
        absMax = max(fabsf(x[i]), absMax);
        sum += x[i];
    
        if ((i+1) % WARP_SIZE == 0 || i == seqNum * hiddenDim - 1) {
            // 如果已经处理了一个完整的warp，或者已经到了最后一个元素
            scale = absMax / 127;
            y[i / WARP_SIZE].ds.x = scale; // （i / WARP_SIZE）是warp的索引
            y[i / WARP_SIZE].ds.y = sum;
    
            for(int j = 0; j < WARP_SIZE; ++j) { // j 是warp内的索引
                if ((i / WARP_SIZE) * WARP_SIZE + j >= seqNum * hiddenDim) {
                    break; // 检查不要越过x的范围
                }
                const int8_t q = (absMax == 0.0f ? 0 : roundf(x[(i / WARP_SIZE) * WARP_SIZE + j] / scale));
                printf("%d- ",q);
                y[(i / WARP_SIZE)].qs[j] = q;
            }
    
            // 归零，准备处理下一个warp
            sum = 0.0f;
            scale = 0.0f;
            absMax = 0.0f;
        }
    }

}

void compute_error(block_q8_1 * a, block_q8_1 * b, const int size, float * max_err, float * avg_err) {
    *max_err = 0.0f; // 最大误差
    *avg_err = 0.0f; // 平均误差
    for (int i = 0; i < size; i++) {
        std::cout<<static_cast<int>(a[i/WARP_SIZE].qs[i%WARP_SIZE])<<" "<<static_cast<int>(b[i/WARP_SIZE].qs[i%WARP_SIZE])<<std::endl;
        float err = fabs(a[i/WARP_SIZE].qs[i%WARP_SIZE] - b[i/WARP_SIZE].qs[i%WARP_SIZE]); // 计算每个元素的差的绝对值
        *max_err = fmax(*max_err, err); // 更新最大误差
        *avg_err += err; // 累加平均误差
    }
    std::cout<< static_cast<int>(a[0].qs[0])<<" "<< static_cast<int>(b[0].qs[0]) <<std::endl;
    *avg_err /= size; // 计算平均误差
}


int main(){
    auto vy = new block_q8_1[(HIDDEN_DIM/WARP_SIZE)*SEQ_LEN];
    auto h_vy = new block_q8_1[(HIDDEN_DIM/WARP_SIZE)*SEQ_LEN];
    auto x = new dfloat[HIDDEN_DIM*SEQ_LEN];
    block_q8_1* d_vy;
    dfloat* d_x;
    //随机生成x
    dfloat low = 1.0;
    dfloat high = 1000.0;
    for(int i = 0; i < HIDDEN_DIM*SEQ_LEN; i++) {
        x[i] = low + static_cast <dfloat> (rand()) / (static_cast <dfloat> (RAND_MAX)) *(high - low);
    }
    check_cuda_error(cudaMalloc(&d_vy,sizeof(block_q8_1)*(HIDDEN_DIM/WARP_SIZE)*SEQ_LEN),"cudaMalloc,d_vy");
    check_cuda_error(cudaMalloc(&d_x,sizeof(dfloat)*HIDDEN_DIM*SEQ_LEN),"cudaMalloc,d_x");
    check_cuda_error(cudaMemcpy(d_x,x,sizeof(dfloat)*HIDDEN_DIM*SEQ_LEN,cudaMemcpyHostToDevice),"cudaMemcpy x host to device");
    launch_kernel(d_x,d_vy,SEQ_LEN,HIDDEN_DIM,HIDDEN_DIM);
    check_cuda_error(cudaMemcpy(vy,d_vy,sizeof(block_q8_1)*(HIDDEN_DIM/WARP_SIZE)*SEQ_LEN,cudaMemcpyDeviceToHost),"cudaMemcpy d_vy device to host");
    //for (int i = 0; i < HIDDEN_DIM*SEQ_LEN; i++) {
        //std::cout<<static_cast<int>(vy[i/WARP_SIZE].qs[i%WARP_SIZE])<<" "<<std::endl;
    //}

    quantize_q8_1_cpu(x,h_vy,SEQ_LEN,HIDDEN_DIM,HIDDEN_DIM);
    float max_err = 0.0f;
    float avg_err = 0.0f;
    compute_error(h_vy,vy,HIDDEN_DIM*SEQ_LEN,&max_err,&avg_err);
    std::cout<<"max error"<<max_err<<std::endl;
    std::cout<<"avg error"<<avg_err<<std::endl;
    cudaFree(d_x);
    cudaFree(d_vy);

    delete vy;
    delete h_vy;
    delete x;
}


