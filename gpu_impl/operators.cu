#include "operators.cuh"

// M1 and M2 are vectors and o is the returned value from Kernel function
__global__ void Kernel(cublasStatus_t stat, cublasHandle_t handle, float* V1, float* V2, int N, float* Out){
    float* V = (float*)malloc(N * sizeof(float));
    float alpha = -1.0f;
    memcpy(V, V1, N * sizeof(float));
    stat = cublasSaxpy(handle, N, &alpha, V2, 1, V, 1);
    if(stat!=CUBLAS_STATUS_SUCCESS){
        printf("Error message: error code %d for cublasSaxpy", stat);
    }
    stat = cublasSnrm2(handle, N, V, 1, Out);
    if(stat!=CUBLAS_STATUS_SUCCESS){
        printf("Error message: error code %d for cublasSnrm2", stat);
    }
    *Out = 32*expf(-1/0.0006*(*Out));
    free(V);
}
// M1, M2 are square matrix with the same shape and N is the number of rows/columns, Out is the output square matrix of this kernel function with N rows 
__global__ void cov(cublasStatus_t stat, cublasHandle_t handle, float* M1, float* M2, int N, float* Out){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for(int i=index; i< N; i+=stride){
        for(int j=0; j<N; j++){
            Kernel<<<1,1>>>(stat, handle, &M1[i*N], &M2[j*N], N, &Out[i*N+j]);
            //Out[i*N+j] = 2.0;
        }
    }
}
// M is square matrix and N is the number of rows/columns
__device__ float* inv(cublasStatus_t stat, cublasHandle_t handle, float* M, int N){
    int *pivotArray = (int*)malloc(N*1*sizeof(int));
    int *infoArray = (int*)malloc(1*sizeof(int));
    float *Carray = (float*)malloc(N*N*sizeof(float));
    stat = cublasSgetrfBatched(handle, N, &M, N, pivotArray, infoArray, 1);
    if(stat!=CUBLAS_STATUS_SUCCESS){
        printf("Error message: error code %d for cublasSgetrfBatched", stat);
    }
    stat = cublasSgetriBatched(handle, N, &M, N, pivotArray, &Carray, N, infoArray, 1);
    if(stat!=CUBLAS_STATUS_SUCCESS){
        printf("Error message: error code %d for cublasSgetriBatched", stat);
    }
    free(pivotArray);
    free(infoArray);
    return Carray;
}
