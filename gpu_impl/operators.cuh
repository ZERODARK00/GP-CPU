#include <string>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>

#define cudaCheckError() {                                          \
 cudaError_t e=cudaGetLastError();                                 \
 if(e!=cudaSuccess) {                                              \
   printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));           \
   exit(0); \
 }                                                                 \
}

static void HandleError( cudaError_t err, const char *file, int line){
    if(err != cudaSuccess){
        printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
        exit(EXIT_FAILURE);
    }
}

#define HANDLE_ERROR( err ) (HandleError(err, __FILE__, __LINE__))

// M1 and M2 are vectors and o is the returned value from Kernel function
__device__ float Kernel(float* V1, float* V2, int N);
// M1, M2 are square matrix with the same shape and N is the number of rows/columns, Out is the output square matrix of this kernel function with N rows 
__global__ void cov(float** M1, float** M2, int N, float** Out);
// M is square matrix and N is the number of rows/columns
__device__ float* inv(float* M, int N);
