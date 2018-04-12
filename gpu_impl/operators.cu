#include "operators.cuh"

// M1 and M2 are vectors and o is the returned value from Kernel function
__device__ float* Kernel(cublasHandle_t handle, float* V1, float* V2, int N){
    float* V = (float*)malloc(N * sizeof(float));
    cudaMemcpy(V, V1, N * sizeof(float), cudaMemcpyDeviceToDevice);
    cublasSaxpy(handle, N, -1, V2, 1, V, 1);
    float* Out = (float*)malloc(1 * sizeof(float));
    cublasSnrm2(handle, N, V, 1, Out);
    *Out = 32*expf(-1/0.0006*(*Out));

    free(V);
    return *Out
}
// M1, M2 are square matrix with the same shape and N is the number of rows/columns, Out is the output square matrix of this kernel function with N rows 
__global__ void cov(cublasHandle_t handle, float* M1, float* M2, int N, float* Out){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for(int i=index; i< N; i+=stride){
        for(int j=0; j<N; j++){
            Kernel(handle, M1[i*N], M2[j*N], N, Out[index*N+j]);
        }
    }
}
// M is square matrix and N is the number of rows/columns
__device__ float inv(cublasHandle_t handle, float* M, int N){
    int *PivotArray = (int*)malloc(N*1*sizeof(int));
    int *infoArray = (int*)malloc(1*sizeof(int));
    float *Carray = (float*)malloc(N*N*sizeof(float));
    cublasSgetrfBatched(handle, N, &M, N, pivotArray, infoArray, 1);
    cublasSgetriBatched(handle, N, &M, N, pivotArray, Carray, N, infoArray, 1);

    free(PivotArray);
    free(infoArray);
    return Carray
}
