// System includes
#include <iostream>
#include <assert.h>

#include <math.h>
#include "operators.hpp"
#include "operators.cuh"
#include <time.h>

#define NUM_SLAVES 5
#define CARD_SUPPORT_SET 20
#define NUM_FEATURES 11
#define NUM_SAMPLES 20000

// to compute local summary (running on GPU)
__global__ void slave_local(int N, float *S, float *D, float *yD, float *local_M, float *local_C){
    float *SD, *DD, *DS, *SS;
    float *inv_DD_S = new float[N*N];

    float **a = new float*[4];
    float **b = new float*[4];
    float **out = new float*[4];

    // set A and B samples for parallel execution
    a[0] = S; b[0] = D;
    a[1] = D; b[1] = D;
    a[2] = D; b[2] = S;
    a[3] = S; b[3] = S;

    // execute 4 covariance functions in parallel using 4 blocks with N threads
    cov<<<4,N>>>(a, b, N, out);

    // get outputs of covariance functions
    SD = out[0];
    DD = out[1];
    DS = out[2];
    SS = out[3];

    // calculate local summary (using CuBLAS)
    cublasHandle_t handle;
    cublasStatus_t stat = cublasCreate(&handle);

    float alpha = 1.0f;
    float beta = 0.0f;
    int size = N*N;

    float *inv_SS = new float[N*N];
    float *DD_S = new float[N*N];

    inv<<<1,1>>>(SS, N, inv_SS);

    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N,N,N, &alpha, DS, N, inv_SS, N, &beta, DD_S, N);
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N,N,N, &alpha, DD_S, N, SD, N, &beta, inv_DD_S, N);

    alpha = -1.0;
    cublasSaxpy(handle, size, &alpha, inv_DD_S, 1, DD, 1);

    inv<<<1,1>>>(DD, N, inv_DD_S);

    alpha = 1.0;

    // compute local mean (using CuBLAS)
    float *local_M_temp = new float[N*N];
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N,N,N, &alpha, SD, N, inv_DD_S, N, &beta, local_M_temp, N);
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N,1,N, &alpha, local_M_temp, N, yD, N, &beta, local_M, N);

    // compute local covariance  (using CuBLAS)
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N,N,N, &alpha, local_M_temp, N, DS, N, &beta, local_C, N);

    // free memory in device
    free(a); free(b); free(out);
    free(inv_SS);
    free(DD_S);
    free(local_M_temp);
}

// to calculate for global summary (running on GPU)
__global__ void slave_global(int N, float *S, float *U, float *global_C, float *global_M, float *pred_mean) {
    float **a = new float*[1];
    float **b = new float*[1];
    float **out = new float*[1];

    // computation for UU, SU, UD, DU is skipped since we do not need them for prediction mean
    a[0] = U; b[0] = S;

    // execute 1 covariance function with N parallel threads
    cov<<<1,N>>>(a, b, N, out);

    // get the output
    float *US = out[0];

    // calculate for prediction mean (using CuBLAS)
    cublasHandle_t handle;
    cublasStatus_t stat = cublasCreate(&handle);
    float alpha = 1.0;
    float beta = 0.0f;
    
    float *inv_global_C = new float[N*N];
    inv<<<1,1>>>(global_C, N, inv_global_C);

    // predictions stored in pred_mean
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N,N,N, &alpha, US, N, inv_global_C, N, &beta, pred_mean, N);
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N,N,N, &alpha, pred_mean, N, global_M, N, &beta, pred_mean, N);
}

// master runs on CPU
void master(mat S, float* pred, int* partition, mat train_data, mat train_target, mat test_data, mat test_target, int interval) {
    int	slaveCount;
    
    float *S_set = matToArray(S);

    float *global_M = new float[interval];
    float *global_C = new float[interval*interval];

    float **train_data_arr = new float*[NUM_SLAVES];
    float **train_target_arr = new float*[NUM_SLAVES];
    float **test_data_arr = new float*[NUM_SLAVES];

    float **local_M_arr = new float*[NUM_SLAVES];
    float **local_C_arr = new float*[NUM_SLAVES];

    for(int i=0;i<NUM_SLAVES;i++){
        local_M_arr[i] = new float[interval];
        local_C_arr[i] = new float[interval*interval];
    }

    cudaStream_t streams[NUM_SLAVES];
    int s = sizeof(float);

    // device copies
    float *d_support, *d_train_data, *d_train_target, *d_test_data, *local_M, *local_C;

    // allocate space for device copies
    cudaMalloc((void **)&d_support, CARD_SUPPORT_SET*NUM_FEATURES*s);
    cudaMalloc((void **)&d_train_data, NUM_SLAVES*interval*NUM_FEATURES*s);
    cudaMalloc((void **)&d_train_target, NUM_SLAVES*interval*1*s);
    cudaMalloc((void **)&d_test_data, NUM_SLAVES*interval*NUM_FEATURES*s);

    cudaMalloc((void **)&local_M, NUM_SLAVES*interval*1*s);
    cudaMalloc((void **)&local_C, NUM_SLAVES*interval*interval*s);

    // copy common support set to device memory first
    cudaMemcpy(d_support, S_set, CARD_SUPPORT_SET*NUM_FEATURES*s, cudaMemcpyHostToDevice);
    
    for (slaveCount=0; slaveCount < NUM_SLAVES; slaveCount++){
        // split data for each slave
        train_data_arr[slaveCount] = matToArray(train_data.rows(slaveCount*interval, (slaveCount+1)*interval-1));
        train_target_arr[slaveCount] = matToArray(train_target.rows(slaveCount*interval, (slaveCount+1)*interval-1));
        test_data_arr[slaveCount] = matToArray(test_data.rows(slaveCount*interval, (slaveCount+1)*interval-1));

        // copy the data for train, target and test into device memory
        cudaMemcpy(&d_train_data[slaveCount*(interval*NUM_FEATURES)], train_data_arr[slaveCount], interval*NUM_FEATURES*s, cudaMemcpyHostToDevice);
        cudaMemcpy(&d_train_target[slaveCount*(interval*1)], train_target_arr[slaveCount], interval*1*s, cudaMemcpyHostToDevice);
        cudaMemcpy(&d_test_data[slaveCount*(interval*NUM_FEATURES)], test_data_arr[slaveCount], interval*NUM_FEATURES*s, cudaMemcpyHostToDevice);
    }

    // start NUM_SLAVES workers to calculate for local summary
    for (slaveCount = 0; slaveCount < NUM_SLAVES; slaveCount++) {
        // create new stream for parallel grid execution
        cudaStreamCreate(&streams[slaveCount]);

        // launch one worker(slave) kernel per stream
        slave_local<<<1, 1, 0, streams[slaveCount]>>>(partition[slaveCount], d_support, 
                &d_train_data[slaveCount*(interval*NUM_FEATURES)], 
                &d_train_target[slaveCount*(interval*1)], 
                &local_M[slaveCount*(interval*1)], 
                &local_C[slaveCount*(interval*interval)]);
    }

    // synchronice all streams
    for(int i=0; i<NUM_SLAVES; i++){
        cudaStreamSynchronize(streams[i]);
    }

    // Copy result back to host
    for (int slaveCount=0; slaveCount<NUM_SLAVES; slaveCount++){
        cudaMemcpy(local_M_arr[slaveCount], &local_M[slaveCount*(interval*1)], interval*1*s, cudaMemcpyDeviceToHost);
        cudaMemcpy(local_C_arr[slaveCount], &local_C[slaveCount*(interval*interval)], interval*interval*s, cudaMemcpyDeviceToHost);
    }

    // free device memory
    cudaFree(d_train_data);
    cudaFree(d_train_target);
    
    // sum up local summary to get global summary
    for (slaveCount = 0; slaveCount < NUM_SLAVES; slaveCount++) {
        for(int i=0; i<interval; i++){
            global_M[i] += local_M_arr[slaveCount][i];
            for(int j=0; j<interval; j++){
                global_C[i*interval+j] += local_C_arr[slaveCount][i*interval+j];
            }
        }
    }

    // initialize variables for global summary to be copied to device
    float *d_global_M, *d_global_C;
    float *d_pred_M;
    interval = (NUM_SAMPLES-NUM_SAMPLES/2)/NUM_SLAVES;

    // allocate space for global summaries on device
    cudaMalloc((void **)&d_global_M, interval*NUM_FEATURES*s);
    cudaMalloc((void **)&d_global_C, interval*interval*s);
    cudaMalloc((void **)&d_pred_M, NUM_SAMPLES/3*s);

    // copy global summaries to device
    cudaMemcpy(d_global_M, global_M, interval*NUM_FEATURES*s, cudaMemcpyHostToDevice);
    cudaMemcpy(d_global_C, global_C, interval*interval*s, cudaMemcpyHostToDevice);

    // calculate for final prediction
    for (slaveCount = 0; slaveCount < NUM_SLAVES; slaveCount++) {
        // launch one worker(slave) kernel per stream (reuse stream from previous)
        slave_global<<<1, 1, 0, streams[slaveCount]>>>(interval, d_support, 
                &d_test_data[slaveCount*interval*NUM_FEATURES], 
                d_global_M, d_global_C, &d_pred_M[slaveCount*(interval)]);
    }

    // synchronice all streams
    for(int i=0; i<NUM_SLAVES; i++){
        cudaStreamSynchronize(streams[i]);
    }

    // synchronice all device functions
    cudaDeviceSynchronize();

    // Copy prediciton result back to host
    for (int slaveCount=0; slaveCount<NUM_SLAVES; slaveCount++){
        cudaMemcpy(&pred[slaveCount], &d_pred_M[slaveCount*(interval)], interval*sizeof(float), cudaMemcpyDeviceToHost);
    }

    // Cleanup
    cudaFree(d_support);
    cudaFree(d_test_data); 
    cudaFree(local_C);
    cudaFree(d_global_M); 
    cudaFree(d_global_C); 
    cudaFree(d_pred_M);

    // results are in pred (float* pred)
}

// main runs on CPU
int main(void){
    clock_t start = clock();

    // load data from csv file
    std::string path = "../hdb.csv";
    mat data = parseCsvFile(path, NUM_SAMPLES);

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
    mat train_data = data.rows(0, all_samples/2-1).cols(1, 11);
    mat train_target = data.rows(0, all_samples/2-1).col(0);
    mat test_data = data.rows(all_samples/2, all_samples-1).cols(1, 11);
    mat test_target = data.rows(all_samples/2, all_samples-1).col(0);

    float *pred = new float[all_samples-all_samples/2];

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
    master(support, pred, partitions, train_data, train_target, test_data, test_target, intervals);

    clock_t end = clock();
    double time_spent= (double)(end-start) / CLOCKS_PER_SEC;
    printf("Total time for %d slaves to execute %d samples: %f\n", NUM_SLAVES, NUM_SAMPLES, time_spent); 

    // for printing out predictions in pred
    /*
    for(int i = 0; i < (all_samples-all_samples/2); i++){
        cout << pred[i] << "(" << test_target(i, 0) << ")" << "\t";
        if(i%10==0 && i!=0){
            cout<<endl;
        }
    }
    */

    return(0);
}
