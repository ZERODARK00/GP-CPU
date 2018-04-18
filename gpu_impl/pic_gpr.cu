// System includes
#include <iostream>
#include <assert.h>

#include <math.h>
#include "operators.hpp"
#include "operators.cuh"
#include <time.h>

#define NUM_SLAVES 5
#define CARD_SUPPORT_SET 20
#define NUM_FEATURES 8
#define NUM_SAMPLES 1000

// to compute local summary (running on GPU)
__global__ void slave_local(int N, float *S, float *D, float *yD, float *local_M, float *local_C) {
    //__shared__ float *SD, *DD, *DS, *SS, *inv_DD_S;
    float *SD, *DD, *DS, *SS;
    float *inv_DD_S = new float[N*N];

    float **a = new float*[4];
    float **b = new float*[4];
    float **out = new float*[4];

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
    float *dd = new float[size];
    memcpy(dd, DD, size*sizeof(float));
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
__global__ void slave_global(int N, float *S, float *U, float *global_C, float *global_M, float *pred_mean) {
    //extern __shared__ float *SD, *DD, *DS, *SS, *inv_DD_S;

    // local copies
    float **a = new float*[1];
    float **b = new float*[1];
    float **out = new float*[1];
    
    a[0] = U; b[0] = S;

    // execute 5 covariance functions in parallel using 5 blocks
    cov<<<1,N>>>(a, b, N, out);

    float *US = out[0];

    cublasHandle_t handle;
    cublasStatus_t stat = cublasCreate(&handle);
    float alpha = 1.0;
    float beta = 0.0f;
    
    // predictions stored in pred_mean
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N,N,N, &alpha, US, N, inv(global_C, N), N, &beta, pred_mean, N);
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

    float *d_support, *d_train_data, *d_train_target, *d_test_data, *local_M, *local_C;

    // Allocate space for device copies
    cudaMalloc((void **)&d_support, CARD_SUPPORT_SET*NUM_FEATURES*s);
    cudaMalloc((void **)&d_train_data, NUM_SLAVES*interval*NUM_FEATURES*s);
    cudaMalloc((void **)&d_train_target, NUM_SLAVES*interval*1*s);
    cudaMalloc((void **)&d_test_data, NUM_SLAVES*interval*NUM_FEATURES*s);

    cudaMalloc((void **)&local_M, NUM_SLAVES*interval*1*s);
    cudaMalloc((void **)&local_C, NUM_SLAVES*interval*interval*s);

    cudaMemcpy(d_support, S_set, CARD_SUPPORT_SET*NUM_FEATURES*s, cudaMemcpyHostToDevice);
    
    // start NUM_SLAVES workers to calculate for local summary
    for (slaveCount = 0; slaveCount < NUM_SLAVES; slaveCount++) {
        // create new stream for parallel grid execution
        cudaStreamCreate(&streams[slaveCount]);
        
        train_data_arr[slaveCount] = matToArray(train_data.rows(slaveCount*interval, (slaveCount+1)*interval-1));
        train_target_arr[slaveCount] = matToArray(train_target.rows(slaveCount*interval, (slaveCount+1)*interval-1));
        test_data_arr[slaveCount] = matToArray(test_data.rows(slaveCount*interval, (slaveCount+1)*interval-1));

        HANDLE_ERROR(cudaMemcpyAsync(&d_train_data[slaveCount*(interval*NUM_FEATURES)], train_data_arr[slaveCount], interval*NUM_FEATURES*s, cudaMemcpyHostToDevice, streams[slaveCount]));
        cudaMemcpyAsync(&d_train_target[slaveCount*(interval*1)], train_target_arr[slaveCount], interval*1*s, cudaMemcpyHostToDevice, streams[slaveCount]);
        cudaMemcpyAsync(&d_test_data[slaveCount*(interval*NUM_FEATURES)], test_data_arr[slaveCount], interval*NUM_FEATURES*s, cudaMemcpyHostToDevice, streams[slaveCount]);

        // launch one worker(slave) kernel per stream
        slave_local<<<1, 1, 0, streams[slaveCount]>>>(partition[slaveCount], d_support, 
                &d_train_data[slaveCount*(interval*NUM_FEATURES)], 
                &d_train_target[slaveCount*(interval*1)], 
                &local_M[slaveCount*(interval*1)], 
                &local_C[slaveCount*(interval*interval)]);

        // Copy result back to host
    }

    // synchronice all device functions
    for(int i=0; i<NUM_SLAVES; i++){
        cudaStreamSynchronize(streams[i]);
    }
    cudaDeviceSynchronize();
    for (int slaveCount=0; slaveCount<NUM_SLAVES; slaveCount++){
        cudaMemcpy(local_M_arr[slaveCount], &local_M[slaveCount*(interval*1)], interval*1*s, cudaMemcpyDeviceToHost);
        cudaMemcpy(local_C_arr[slaveCount], &local_C[slaveCount*(interval*interval)], interval*interval*s, cudaMemcpyDeviceToHost);
    }

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

    float *d_global_M, *d_global_C;
    float *d_pred_M;

    interval = (NUM_SAMPLES-NUM_SAMPLES/2)/NUM_SLAVES;
    cudaMalloc((void **)&d_global_M, interval*NUM_FEATURES*s);
    cudaMalloc((void **)&d_global_C, interval*interval*s);
    cudaMalloc((void **)&d_pred_M, NUM_SAMPLES/3*s);

    cudaMemcpy(d_global_M, global_M, interval*NUM_FEATURES*s, cudaMemcpyHostToDevice);
    cudaMemcpy(d_global_C, global_C, interval*interval*s, cudaMemcpyHostToDevice);
    // calculate for final prediction
    for (slaveCount = 0; slaveCount < NUM_SLAVES; slaveCount++) {

        // launch one worker(slave) kernel per stream, reuse stream to access shared variables
        slave_global<<<1, 1, 0, streams[slaveCount]>>>(interval, d_support, 
                &d_test_data[slaveCount*interval*NUM_FEATURES], 
                d_global_M, d_global_C, &d_pred_M[slaveCount*(interval)]);

    }

    // synchronice all device functions
    for(int i=0; i<NUM_SLAVES; i++){
        cudaStreamSynchronize(streams[i]);
    }
    cudaDeviceSynchronize();
    for (int slaveCount=0; slaveCount<NUM_SLAVES; slaveCount++){
        // Copy result back to host
        cudaMemcpy(&pred[slaveCount], &d_pred_M[slaveCount*(interval)], interval*sizeof(float), cudaMemcpyDeviceToHost);
    }
    // Cleanup
    cudaFree(d_support);
    cudaFree(d_test_data); 
    cudaFree(local_C);
    cudaFree(d_global_M); 
    cudaFree(d_global_C); 
    cudaFree(d_pred_M);
    // results are in pred (int** pred)
}

// main runs on CPU
int main(void){
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
    clock_t start = clock();
    master(support, pred, partitions, train_data, train_target, test_data, test_target, intervals);
    clock_t end = clock();
    double time_spent  = (double)(end-start) / CLOCKS_PER_SEC;
    printf("Total time for %d slaves to execute %d samples: %f\n", NUM_SLAVES, NUM_SAMPLES, time_spent); 

    // print out predictions in pred variable
    //mat pred_M = zeros<mat>(all_samples-all_samples/2, 1);
    /*
    for(int i = 0; i < (all_samples-all_samples/2); i++){
        cout << pred[i] << "(" << test_target(i, 0) << ")" << "\t";
        if(i%10==0 && i!=0){
            cout<<endl;
        }
       // pred_M(i, 0) = pred[i];
    }
    */
    return(0);
}
