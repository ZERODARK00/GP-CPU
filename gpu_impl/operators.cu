#include "operators.cuh"

// M1 and M2 are vectors and o is the returned value from Kernel function
__device__ float Kernel(float* V1, float* V2, int N){
    float* V = (float*)malloc(N * sizeof(float));
    float alpha = -1.0f;
    float out = 0.0f;
    
    cublasHandle_t handle;
    cublasStatus_t stat = cublasCreate(&handle);

    memcpy(V, V1, N * sizeof(float));
    stat = cublasSaxpy(handle, N, &alpha, V2, 1, V, 1);

    if(stat!=CUBLAS_STATUS_SUCCESS){
        printf("Error message: error code %d for cublasSaxpy", stat);
    }

    stat = cublasSnrm2(handle, N, V, 1, &out);

    if(stat!=CUBLAS_STATUS_SUCCESS){
        printf("Error message: error code %d for cublasSnrm2", stat);
    }

    out = 32*expf(-1*(out));

    // cleanup
    free(V);

    return out;
}

// M1, M2 are square matrix with the same shape and N is the number of rows/columns, Out is the output square matrix of this kernel function with N rows 
__global__ void cov(float** m1, float** m2, int N, float** out){
    // first, get the appropriate data block for this GPU block to work on
    float *M1 = m1[blockIdx.x];
    float *M2 = m2[blockIdx.x];
    float *Out = out[blockIdx.x];

    int index = threadIdx.x;
    int stride = blockDim.x;

    // compute the covariance in parallel
    for(int i=index; i< N; i+=stride){
        for(int j=0; j<N; j++){
            Out[i*N+j] = Kernel(&M1[i*N], &M2[j*N], N);
        }
    }
}

// M is square matrix and N is the number of rows/columns
__device__ float* inv(float* M, int N){
    float** dev_M = (float**)malloc(sizeof(float*));
	float** dev_out = (float**)malloc(sizeof(float*));
    int *pivotArray = (int*)malloc(N*1*sizeof(int));
    int *infoArray = (int*)malloc(1*sizeof(int));
    
	*dev_M = M;
    cublasHandle_t handle;
    cublasStatus_t stat = cublasCreate(&handle);

    stat = cublasSgetrfBatched(handle, N, dev_M, N, pivotArray, infoArray, 1);

    if(stat!=CUBLAS_STATUS_SUCCESS){
        printf("Error message: error code %d for cublasSgetrfBatched", stat);
    }

    stat = cublasSgetriBatched(handle, N, (const float**)dev_M, N, pivotArray, dev_out, N, infoArray, 1);

    if(stat!=CUBLAS_STATUS_SUCCESS){
        printf("Error message: error code %d for cublasSgetriBatched", stat);
    }

    // cleanup
    free(pivotArray);
    free(infoArray);

    return *dev_out;
}
