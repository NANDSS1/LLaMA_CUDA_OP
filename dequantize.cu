#include <iostream>
#define QK8_1 32
#define BLOCK_DIM 32//一个block在x方向分配的线程数
#define HIDDEN_DIM 4096
#define WARP_SIZE 32
#define SEQ_LEN 1
#define QSTURCT_T block_q8_0
template<typename T>
struct  block_q8_0{
    T    scale;              //量化的scale
    int8_t  qs[QK8_1];      // quants 量化后的weight数据,每QK8_0个数共享一个scale
};//记录量化后的权重的

// 定义一个辅助函数，用于检查 CUDA 调用是否成功
void check_cuda_error(cudaError_t err, const char * msg){
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << msg << ": " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}


template<typename T,typename BlockType>
static __global__ void dequantize(T * __restrict__ x,const void * __restrict__ vy,const int seqNum,const int hiddenDim,const int qSize){
    const int ix = blockDim.x*blockIdx.x + threadIdx.x; //0-4096，如果分的线程不够来，跨一个gridDim.x
    const int iy = blockDim.y*blockIdx.y + threadIdx.y; //记录第几行元素,y方向的id
    const BlockType * y = reinterpret_cast<const BlockType *>(vy);
    const int gid = iy*gridDim.x*blockDim.x + ix;//全局id,gridDim.x*blockDim.x = kx的话，刚好就是元素id
    if(gridDim.x*blockDim.x == hiddenDim) printf("gridDim.x*blockDim.x  == kx");
    int step = gridDim.x*blockDim.x*gridDim.y*blockDim.y;//分配总的线程数
    for(int i = gid;i <seqNum*hiddenDim; i+= step){
        x[i] = y[i / qSize].qs[i % qSize] * y[i / qSize].scale;//qSize是block_q8_0，qSize个元素共享一组scale
    }
}

template<typename T,typename BlockType,int blockDim>
static __host__ void launch_kernel(T* d_x,const void* d_vy,const int seqNum,const int hiddenDim,const int qSize){
    dim3 grid((hiddenDim-1/blockDim)+blockDim,seqNum,1);
    dim3 block(blockDim,1,1);
    //dequantize<float,BlockType><<<grid,block>>>(d_x,d_vy,seqNum,hiddenDim,qSize);
    dequantize<float,BlockType><<<grid,block>>>(d_x,d_vy,seqNum,hiddenDim,qSize);//反量化,自动推导参数
}

template<typename T,typename BlockType>
static __host__ void dequantize_cpu(T* cpu_x,const void* vy,const int seqNum,const int hiddenDim,const int qSize){
    const BlockType * y = reinterpret_cast<const BlockType *>(vy);
    const int sumNum = seqNum*hiddenDim;
    for(int i = 0;i<sumNum;i++){
        cpu_x[i] = y[i / qSize].qs[i % qSize] * y[i / qSize].scale;
    }
}

template<typename T>
static __host__ void compute_error(T* cpu_x,T* x,const int sumNum,float* max_err,float* avg_err){
    *max_err = 0.0f; // 最大误差
    *avg_err = 0.0f; // 平均误差
    for (int i = 0; i < sumNum; i++) {
        float err = fabs(cpu_x[i] - x[i]); // 计算每个元素的差的绝对值
        *max_err = fmax(*max_err, err); // 更新最大误差
        *avg_err += err; // 累加平均误差
    }
    *avg_err /= sumNum; // 计算平均误差

}
template<typename T,typename BlockType>
static __host__ void getRand(BlockType* vy,int structNum){
    float low = 1.0;
    float high = 1000.0;
    for(int i = 0;i < structNum;i++){
        vy[i].scale = low + static_cast <T> (rand()) / (static_cast <T> (RAND_MAX)) *(high - low);
        for(int j = 0;j < QK8_1;j++){
            vy[i].qs[j] = j;
        }
    }
}


int main(){
    const int blockDim = BLOCK_DIM;
    const int seqNum = SEQ_LEN;
    const int hiddenDim = HIDDEN_DIM;
    const int qSize = QK8_1;
    auto vy = new block_q8_0<float>[(hiddenDim/qSize)*seqNum];
    //对vy随机一下
    getRand<float,block_q8_0<float>>(vy,(hiddenDim/qSize)*seqNum);
    auto x = new float[hiddenDim*seqNum];
    auto cpu_x = new float[hiddenDim*seqNum];
    block_q8_0<float>* d_vy;
    float* d_x;
    check_cuda_error(cudaMalloc(&d_vy,sizeof(block_q8_0<float>)*(hiddenDim/qSize)*seqNum),"device malloc d_vy");
    check_cuda_error(cudaMalloc(&d_x,sizeof(float)*hiddenDim*seqNum),"device malloc d_x");
    check_cuda_error(cudaMemcpy(d_vy,vy,sizeof(block_q8_0<float>)*(hiddenDim/qSize)*seqNum,cudaMemcpyHostToDevice),"host copy data to device:vy->d_vy");
    launch_kernel<float,block_q8_0<float>,blockDim>(d_x,d_vy,seqNum,hiddenDim,qSize);
    check_cuda_error(cudaMemcpy(x,d_x,sizeof(float)*hiddenDim*seqNum,cudaMemcpyDeviceToHost),"device copy data to host:d_x -> x");\
    dequantize_cpu<float,block_q8_0<float>>(cpu_x,vy,seqNum,hiddenDim,qSize);
    float max_err = 0.0f;
    float avg_err = 0.0f;
    compute_error<float>(cpu_x,x,hiddenDim*seqNum,&max_err,&avg_err);
    std::cout<<"max error"<<max_err<<std::endl;
    std::cout<<"avg error"<<avg_err<<std::endl;
    for(int i = 0;i < hiddenDim*seqNum;i++){
        //std::cout<<x[i]<<std::endl;
    }


    cudaFree(d_vy);
    cudaFree(d_x);
    delete vy;
    delete x;
    delete cpu_x;


}