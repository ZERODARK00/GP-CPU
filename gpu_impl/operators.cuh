#include <string>
#include <cuda_runtime.h>
#include <cublas_v2.h>

// M1 and M2 are vectors and o is the returned value from Kernel function
__device__ float* Kernel(cublasHandle_t handle, float* V1, float* V2, int N);
// M1, M2 are square matrix with the same shape and N is the number of rows/columns, Out is the output square matrix of this kernel function with N rows 
__global__ void cov(cublasHandle_t handle, float* M1, float* M2, int N, float* Out);
// M is square matrix and N is the number of rows/columns
__device__ float inv(cublasHandle_t handle, float* M, int N);
