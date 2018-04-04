// System includes
#include <stdio.h>
#include <iostream>
#include <assert.h>

// CUDA runtime
#include <cuda_runtime.h>

#include <math.h>
#include <string.h>
#include "operators.hpp"

#define NUM_SLAVES 5
#define CARD_SUPPORT_SET 20

// kernel function
float Kernel(mat M1, mat M2){
    // M1 and M2 are row vectors
    return(32*exp(-1/0.0006*norm(M1-M2, 2)));
}

// covariance function
__global__ void cov(mat *A, mat *B, mat *out, float (*Kernel)(mat A, mat B)){
    int A_samples = A[blockIdx.x].n_rows;
    int B_samples = B[blockIdx.x].n_rows;
    double noise = 0;
    out[blockIdx.x] = zeros<mat>(A_samples, B_samples);
    for(int i=0; i<A_samples; i++){
        for(int j=0; j<B_samples; j++){
            if(i==j){
                noise = 8.6;
            }else{
                noise = 0;
            }
            out[blockIdx.x](i, j) = Kernel(A[blockIdx.x].row(i), B[blockIdx.x].row(j)) + noise*noise;
        }
    }
}

// to compute local summary
__global__ void slave_local(mat* S, mat *D, mat *yD, mat *U, mat *local_M, mat *local_C, float (*Kernel)(mat M1, mat M2)) {
    int samples = S.n_rows;
    __shared__ mat SD, DD, DS, SS, inv_DD_S;

    mat global_M = zeros<mat>(samples, 1);
    mat global_C = zeros<mat>(samples, samples);

    mat *a, *b, *out;
    mat *d_a, *d_b, *d_out;

    int s = 4 * sizeof(mat);

    // Allocate space for device copies
    cudaMalloc((void **)&d_a, s);
    cudaMalloc((void **)&d_b, s);
    cudaMalloc((void **)&d_out, s);

    // Calculate for local summary
    // SD = covariance(S, D, Kernel);
    // DD = covariance(D, D, Kernel);
    // DS = covariance(D, S, Kernel);
    // SS = covariance(S, S, Kernel);

    a[0] = S; b[0] = D;
    a[1] = D; b[1] = D;
    a[2] = D; b[2] = S;
    a[3] = S; b[3] = S;

    // copy inputs to device
    cudaMemcpy(d_a, &a, s, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, &b, s, cudaMemcpyHostToDevice);
    cudaMemcpy(d_out, &out, s, cudaMemcpyHostToDevice);

    // execute 4 covariance functions in parallel using 4 blocks
    cov<<<4,1>>>(d_a, d_b, d_out, Kernel);

    // synchronice all device functions
    cudaDeviceSynchronize();

    // copy outputs to host
    cudaMemcpy(out, d_out, s, cudaMemcpyDeviceToHost);

    SD = out[0];
    DD = out[1];
    DS = out[2];
    SS = out[3];

    // calculate local summary
    inv_DD_S = inv(DD-DS*inv(SS)*SD);
    local_M = SD*inv_DD_S*yD;
    local_C = SD*inv_DD_S*DS;
}

// to calculate for global summary
__global__ void slave_global(mat *S, mat *D, mat *yD, mat *U, mat *local_C, mat *global_C, mat *global_M, double *d_pred_M, float (*Kernel)(mat M1, mat M2)) {
    extern __shared__ mat SD, DD, DS, SS, inv_DD_S;
    mat *a, *b, *out;
    mat *d_a, *d_b, *d_out;

    int s = 5 * sizeof(mat);

    // Allocate space for device copies
    cudaMalloc((void **)&d_a, s);
    cudaMalloc((void **)&d_b, s);
    cudaMalloc((void **)&d_out, s);

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

    // copy inputs to device
    cudaMemcpy(d_a, &a, s, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, &b, s, cudaMemcpyHostToDevice);
    cudaMemcpy(d_out, &out, s, cudaMemcpyHostToDevice);

    // execute 4 covariance functions in parallel using 4 blocks
    cov<<<5,1>>>(d_a, d_b, d_out, Kernel);

    // copy outputs to host
    cudaMemcpy(out, d_out, s, cudaMemcpyDeviceToHost);

    mat UU = out[0];
    mat US = out[1];
    mat SU = out[2];
    mat UD = out[3];
    mat DU = out[4];

    // calculate global summary
    mat local_US = UD*inv_DD_S*DS;
    mat local_SU = SD*inv_DD_S*DU;
    mat local_UU = UD*inv_DD_S*DU;
    mat Phi_US = US+US*pinv(SS)*local_C-local_US;
    mat pred_mean = (Phi_US*pinv(global_C)*global_M) + UD*inv_DD_S*yD;
    mat pred_covar = UU-(Phi_US*pinv(SS)*SU-US*pinv(SS)*local_SU-Phi_US*pinv(global_C)*trans(Phi_US))-local_UU;

    // compute predictions
    for(int i=0; i < pred_mean.n_elem;i++){
        d_pred_M[i] = pred_mean(i,0);
    }
}

void master(mat S, int** pred, int* partition, mat train_data, mat train_target, mat test_data, mat test_target, int interval, float (*Kernel)(mat M1, mat M2)) {
    int	slaveCount;
    int samples = S.n_rows;
    mat test_mean, test_covar;

    mat global_M = zeros<mat>(samples, 1);
    mat global_C = covariance(S, S, Kernel);

    mat *train_data_arr = new mat [NUM_SLAVES];
    mat *train_target_arr = new mat [NUM_SLAVES];
    mat *test_data_arr = new mat [NUM_SLAVES];

    mat *local_M_arr = new mat [NUM_SLAVES];
    mat *local_C_arr = new mat [NUM_SLAVES];

    cudaStream_t *streams;
    int s = sizeof(mat);

    // start NUM_SLAVES workers to calculate for local summary
    for (slaveCount = 0; slaveCount < NUM_SLAVES; slaveCount++) {
        // partitions
        train_data_arr[slaveCount] = train_data.rows(slaveCount*interval, (slaveCount+1)*interval-1);
        train_target_arr[slaveCount] = train_target.rows(slaveCount*interval, (slaveCount+1)*interval-1);
        test_data_arr[slaveCount] = test_data.rows(slaveCount*interval, (slaveCount+1)*interval-1);

        // device copies
        mat *d_support, *d_train_data, *d_train_target, *d_test_data, *local_M, *local_C;

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
        slave_local<<<1, 1, 0, streams[slaveCount]>>>(d_support, d_train_data, d_train_target, d_test_data, local_M, local_C, Kernel);

        // Copy result back to host
        cudaMemcpy(&local_M_arr[slaveCount], local_M, s, cudaMemcpyDeviceToHost);
        cudaMemcpy(&local_C_arr[slaveCount], local_C, s, cudaMemcpyDeviceToHost);

        // Cleanup
        cudaFree(d_support); cudaFree(d_train_data); cudaFree(d_train_target); cudaFree(d_test_data);
    }

    // synchronice all device functions
    cudaDeviceSynchronize();

    // sum up local summary to get global summary
    for (slaveCount = 0; slaveCount < NUM_SLAVES; slaveCount++) {
        global_M = global_M + local_M_arr[slaveCount];
        global_C = global_C + local_C_arr[slaveCount];
    }

    // calculate for final prediction
    for (slaveCount = 0; slaveCount < NUM_SLAVES; slaveCount++) {
        train_data_arr[slaveCount] = train_data.rows(slaveCount*interval, (slaveCount+1)*interval-1);
        train_target_arr[slaveCount] = train_target.rows(slaveCount*interval, (slaveCount+1)*interval-1);
        test_data_arr[slaveCount] = test_data.rows(slaveCount*interval, (slaveCount+1)*interval-1);

        // device copies
        mat *d_support, *d_train_data, *d_train_target, *d_test_data, *local_C;
        mat *d_global_M, *d_global_C;
        double *d_pred_M;

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
        slave_global<<<1, 1, 0, streams[slaveCount]>>>(d_support, d_train_data, d_train_target, d_test_data, local_C, d_global_M, d_global_C, d_pred_M, Kernel);

        // Copy result back to host
        cudaMemcpy(&pred[slaveCount], d_pred_M, sizeof(double), cudaMemcpyDeviceToHost);

        // Cleanup
        cudaFree(d_support); cudaFree(d_train_data); cudaFree(d_train_target); cudaFree(d_test_data); cudaFree(local_C);
        cudaFree(d_global_M); cudaFree(d_global_C); cudaFree(d_pred_M);
    }

    // synchronice all device functions
    cudaDeviceSynchronize();

    // results are in pred (int** pred)
    cout<<"Done"<<endl;
}

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
    master(support, &pred, partitions, train_data, train_target, test_data, test_target, intervals, Kernel);

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
