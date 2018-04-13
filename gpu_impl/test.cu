#include "operators.cuh"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int N=10;

void print_matrix(float* array, int n){
    for(int i=0; i<n; i++){
        for(int j=0; j<n; j++){
            printf("%.2f, ", array[j*n+i]);
        }
        printf("\n");
    }
}

int main(){

    srand((unsigned int)time(NULL));

    float *M1 = (float*)malloc(N*N*sizeof(float)); 
    float *M2 = (float*)malloc(N*N*sizeof(float));
    float *Out = (float*)malloc(N*N*sizeof(float));
    float a = 5.0;

    // initialize the matrix
    for(int i=0; i<N*N; i++){
        M1[i] = ((float)rand()/(float)(RAND_MAX)) * a;
        M2[i] = ((float)rand()/(float)(RAND_MAX)) * a;
    } 

    // create state variables of CUDA
    cudaError_t cudaStat;
    cublasStatus_t stat;
    cublasHandle_t handle;
    cublasCreate(&handle);

    // make and copy matrix onto GPU
    float *devM1, *devM2, *devOut;
    cudaMalloc((void**)&devM1, N*N*sizeof(float));
    cudaMalloc((void**)&devM2, N*N*sizeof(float));
    cudaMalloc((void**)&devOut, N*N*sizeof(float));
    cudaMemcpy(devM1, M1, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(devM2, M2, N * N * sizeof(float), cudaMemcpyHostToDevice);

    cov<<<1, N>>>(handle, devM1, devM2, N, devOut);
    cudaDeviceSynchronize();
    cudaMemcpy(Out, devOut, N * N * sizeof(float), cudaMemcpyDeviceToHost);

    printf("Here is matrix 1:\n");
    print_matrix(M1, N);
    printf("Here is matrix 2:\n");
    print_matrix(M2, N);
    printf("Here is the out:\n");
    print_matrix(Out, N);
    return 0;
}
