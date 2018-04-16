// System includes
#include <stdio.h>
#include <iostream>
#include <assert.h>

// CUDA runtime
#include <cuda_runtime.h>
#include "cublas.h"

#include <math.h>
#include <string.h>
#include "operators.hpp"
#include "operators.cuh"

#define NUM_SLAVES 5
#define CARD_SUPPORT_SET 20
#define NUM_FEATURES 9

// to compute local summary (running on GPU)
__global__ void slave_local(int N, float *S, float *D, float *yD, float *U, float *local_M, float *local_C) {
    __shared__ float *SD, *DD, *DS, *SS, *inv_DD_S;

    float **a = new float*[4];
    float **b = new float*[4];
    float **out = new float*[4];

    // Calculate for local summary
    // SD = covariance(S, D, Kernel);
    // DD = covariance(D, D, Kernel);
    // DS = covariance(D, S, Kernel);
    // SS = covariance(S, S, Kernel);

    a[0] = S; b[0] = D;
    a[1] = D; b[1] = D;
    a[2] = D; b[2] = S;
    a[3] = S; b[3] = S;


    // execute 4 covariance functions in parallel using 4 blocks
    cov<<<4,N>>>(a, b, N, out);

    // synchronice all device functions
    // cudaDeviceSynchronize();

    SD = out[0];
    DD = out[1];
    DS = out[2];
    SS = out[3];

    // calculate local summary
    cublasHandle_t handle;
    cublasStatus_t stat = cublasCreate(&handle);
    float alpha = 1.0f;
    float beta = 0.0f;
    int size = N*N;

    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N,N,N, &alpha, DS, N, inv(SS, N), N, &beta, inv_DD_S, N);
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N,N,N, &alpha, inv_DD_S, N, SD, N, &beta, inv_DD_S, N);

    alpha = -1.0;
    float *dd = malloc(sizeof(float)*size);
    memcpy(dd, DD, size);
    cublasSaxpy(handle, size, &alpha, inv_DD_S, 1, dd, 1);

    inv_DD_S = inv(dd, N);

    alpha = 1.0;

    // compute local mean
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N,N,N, &alpha, SD, N, inv_DD_S, N, &beta, local_M, N);
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N,1,N, &alpha, local_M, N, yD, N, &beta, local_M, N);

    // compute local cov
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N,N,N, &alpha, SD, N, inv_DD_S, N, &beta, local_C, N);
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N,N,N, &alpha, local_C, N, DS, N, &beta, local_C, N);
}

// to calculate for global summary (running on GPU)
__global__ void slave_global(int N, float *S, float *D, float *yD, float *U, float *local_C, float *global_C, float *global_M, float *pred_mean) {
    extern __shared__ float *SD, *DD, *DS, *SS, *inv_DD_S;

    // local copies
    float **a = new float*[5];
    float **b = new float*[5];
    float **out = new float*[5];

    // Calculate for global summary
    // mat UU = covariance(U, U, Kernel);
    // mat US = covariance(U, S, Kernel);
    // mat SU = covariance(S, U, Kernel);
    // mat UD = covariance(U, D, Kernel);
    // mat DU = covariance(D, U, Kernel);

    a[0] = U; b[0] = U;
    a[1] = U; b[1] = S;
    a[2] = S; b[2] = U;
    a[3] = U; b[3] = D;
    a[4] = D; b[4] = U;

    // execute 5 covariance functions in parallel using 5 blocks
    cov<<<5,N>>>(a, b, N, out);

    float *UU = out[0];
    float *US = out[1];
    float *SU = out[2];
    float *UD = out[3];
    float *DU = out[4];

    // calculate global summary
    float *local_US, *local_SU, *local_UU;
    cublasHandle_t handle;
    cublasStatus_t stat = cublasCreate(&handle);
    float alpha = 1.0;
    float beta = 0.0f;
    int size = N*N;

    // local_US
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N,N,N, &alpha, UD, N, inv_DD_S, N, &beta, local_US, N);
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N,N,N, &alpha, local_US, N, DS, N, &beta, local_US, N);

    // local_SU
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N,N,N, &alpha, SD, N, inv_DD_S, N, &beta, local_SU, N);
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N,N,N, &alpha, local_SU, N, DU, N, &beta, local_SU, N);

    // local_UU
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N,N,N, &alpha, UD, N, inv_DD_S, N, &beta, local_UU, N);
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N,N,N, &alpha, local_UU, N, DU, N, &beta, local_UU, N);

    // predictions stored in pred_mean
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N,N,N, &alpha, US, N, inv(global_C, N), N, &beta, pred_mean, N);
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N,N,N, &alpha, pred_mean, N, global_M, N, &beta, pred_mean, N);
}

// master runs on CPU
void master(mat S, int** pred, int* partition, mat train_data, mat train_target, mat test_data, mat test_target, int interval) {
    int	slaveCount;
    int samples = S.n_rows;

    float *global_M = new float[samples];
    float *global_C = new float[samples];

    float **train_data_arr = new float*[NUM_SLAVES];
    float **train_target_arr = new float*[NUM_SLAVES];
    float **test_data_arr = new float*[NUM_SLAVES];

    float **local_M_arr = new float*[NUM_SLAVES];
    float **local_C_arr = new float*[NUM_SLAVES];

    cudaStream_t *streams;
    int s = sizeof(float);

    // start NUM_SLAVES workers to calculate for local summary
    for (slaveCount = 0; slaveCount < NUM_SLAVES; slaveCount++) {
        // partitions
        train_data_arr[slaveCount] = matToArray(train_data.rows(slaveCount*interval, (slaveCount+1)*interval-1));
        train_target_arr[slaveCount] = matToArray(train_target.rows(slaveCount*interval, (slaveCount+1)*interval-1));
        test_data_arr[slaveCount] = matToArray(test_data.rows(slaveCount*interval, (slaveCount+1)*interval-1));

        // device copies
        float *d_support, *d_train_data, *d_train_target, *d_test_data, *local_M, *local_C;

        // Allocate space for device copies
        cudaMalloc((void **)&d_support, s);
        cudaMalloc((void **)&d_train_data, s);
        cudaMalloc((void **)&d_train_target, s);
        cudaMalloc((void **)&d_test_data, s);

        cudaMalloc((void **)&local_M, s);
        cudaMalloc((void **)&local_C, s);

        // Copy inputs to device
        cudaMemcpy(d_support, &S, s, cudaMemcpyHostToDevice);
        cudaMemcpy(d_train_data, &train_data_arr[slaveCount], s, cudaMemcpyHostToDevice);
        cudaMemcpy(d_train_target, &train_target_arr[slaveCount], s, cudaMemcpyHostToDevice);
        cudaMemcpy(d_test_data, &test_data_arr[slaveCount], s, cudaMemcpyHostToDevice);

        // create new stream for parallel grid execution
        cudaStreamCreate(&streams[slaveCount]);

        // launch one worker(slave) kernel per stream
        slave_local<<<1, 1, 0, streams[slaveCount]>>>(partition[slaveCount], d_support, d_train_data, d_train_target, d_test_data, local_M, local_C);

        // Copy result back to host
        cudaMemcpy(&local_M_arr[slaveCount], local_M, s, cudaMemcpyDeviceToHost);
        cudaMemcpy(&local_C_arr[slaveCount], local_C, s, cudaMemcpyDeviceToHost);

        // Cleanup
        cudaFree(d_support); cudaFree(d_train_data); cudaFree(d_train_target); cudaFree(d_test_data);
    }

    // synchronice all device functions
    cudaDeviceSynchronize();

    cublasHandle_t handle;
    cublasStatus_t stat = cublasCreate(&handle);
    float alpha = 1.0;

    // sum up local summary to get global summary
    for (slaveCount = 0; slaveCount < NUM_SLAVES; slaveCount++) {
        cublasSaxpy(handle, NUM_FEATURES * partition[slaveCount], &alpha, local_M_arr[slaveCount], 1, global_M, 1);
        cublasSaxpy(handle, NUM_FEATURES * partition[slaveCount], &alpha, local_C_arr[slaveCount], 1, global_C, 1);
    }

    // calculate for final prediction
    for (slaveCount = 0; slaveCount < NUM_SLAVES; slaveCount++) {
        // device copies
        float *d_support, *d_train_data, *d_train_target, *d_test_data, *local_C;
        float *d_global_M, *d_global_C;
        float *d_pred_M;

        // Allocate space for device copies
        cudaMalloc((void **)&d_support, s);
        cudaMalloc((void **)&d_train_data, s);
        cudaMalloc((void **)&d_train_target, s);
        cudaMalloc((void **)&d_test_data, s);
        cudaMalloc((void **)&local_C, s);

        cudaMalloc((void **)&d_global_M, s);
        cudaMalloc((void **)&d_global_C, s);
        cudaMalloc((void **)&d_pred_M, s);

        // Copy inputs to device
        cudaMemcpy(d_support, &S, s, cudaMemcpyHostToDevice);
        cudaMemcpy(d_train_data, &train_data_arr[slaveCount], s, cudaMemcpyHostToDevice);
        cudaMemcpy(d_train_target, &train_target_arr[slaveCount], s, cudaMemcpyHostToDevice);
        cudaMemcpy(d_test_data, &test_data_arr[slaveCount], s, cudaMemcpyHostToDevice);
        cudaMemcpy(local_C, &local_C_arr[slaveCount], s, cudaMemcpyHostToDevice);

        cudaMemcpy(d_global_M, &global_M, s, cudaMemcpyHostToDevice);
        cudaMemcpy(d_global_C, &global_C, s, cudaMemcpyHostToDevice);

        // launch one worker(slave) kernel per stream, reuse stream to access shared variables
        slave_global<<<1, 1, 0, streams[slaveCount]>>>(partition[slaveCount], d_support, d_train_data, d_train_target, d_test_data, local_C, d_global_M, d_global_C, d_pred_M);

        // Copy result back to host
        cudaMemcpy(&pred[slaveCount], d_pred_M, sizeof(float), cudaMemcpyDeviceToHost);

        // Cleanup
        cudaFree(d_support); cudaFree(d_train_data); cudaFree(d_train_target); cudaFree(d_test_data); cudaFree(local_C);
        cudaFree(d_global_M); cudaFree(d_global_C); cudaFree(d_pred_M);
    }

    // synchronice all device functions
    cudaDeviceSynchronize();

    // results are in pred (int** pred)
    cout<<"Done"<<endl;
}

// main runs on CPU
int main(int argc, char *argv[]){
    // load data from csv file
    std::string path = "data.csv";
    mat data = parseCsvFile(path, 1000);

    // normalise the dataset
    int rows = data.n_rows;
    int columns = data.n_cols;

    mat Max = max(data, 0);
    mat Min = min(data, 0);

    for(int i=0;i<rows;i++){
        // ignore the last target column
        for(int j=1;j<columns; j++){
            data(i,j) = (data(i,j)-Min(0, j))/Max(0, j);
        }
    }

    // split data into training and testing samples
    int all_samples = data.n_rows;
    mat train_data = data.rows(0, all_samples/2-1).cols(1, 8);
    mat train_target = data.rows(0, all_samples/2-1).col(0);
    mat test_data = data.rows(all_samples/2, all_samples-1).cols(1, 8);
    mat test_target = data.rows(all_samples/2, all_samples-1).col(0);

    int *pred = new int[all_samples-all_samples/2];

    // get the support data set and partitions of training data set
    mat support;
    int partitions[NUM_SLAVES+1];
    int intervals = all_samples/(2*NUM_SLAVES);
    for(int i=0;i<NUM_SLAVES;i++){
        partitions[i+1] = all_samples/(2*NUM_SLAVES);
        int idx = i*intervals;
        for(int j=0;j<CARD_SUPPORT_SET/NUM_SLAVES;j++){
            support.insert_rows(0, train_data.row(idx+j));
        }
    }

    // call master function (execute on CPU) to start slaves (working on GPU)
    master(support, &pred, partitions, train_data, train_target, test_data, test_target, intervals);

    // print out predictions in pred variable
    mat pred_M = zeros<mat>(all_samples-all_samples/2, 1);
    for(int i = 0; i < (all_samples-all_samples/2); i++){
        cout << pred[i] << "(" << test_target(i, 0) << ")" << "\t";
        if(i%10==0 && i!=0){
            cout<<endl;
        }
        pred_M(i, 0) = pred[i];
    }
    return(0);
}
