#include "mpi.h"
#include <math.h>
#include <string.h>
#include "operators.hpp"
#include <time.h>

#define TRAIN_TAG     1
#define DIE_TAG     0

#define MEAN_TAG 2
#define COVAR_TAG 3

#define NUM_SLAVES 7
#define CARD_SUPPORT_SET 20
#define NUM_SAMPLES 50000

float Kernel(mat M1, mat M2){
    // M1 and M2 are row vectors
    return(32*exp(-1/0.0006*norm(M1-M2, 2)));
}

void master(mat S, int* pred, int* partition, float (*Kernel)(mat M1, mat M2)){
	int	ntasks, rank, work;
    int samples = S.n_rows;
	mat test_mean, test_covar;

    mat global_M = zeros<mat>(samples, 1);
    mat global_C = covariance(S, S, Kernel);
    mat local_M = zeros<mat>(samples, 1);
    mat local_C = zeros<mat>(samples, samples);

    MPI_Status status;
	MPI_Comm_size(
        MPI_COMM_WORLD,   /* always use this */
        &ntasks);          /* #processes in application */

    /*
    * Tell all the slaves to start work.
    */
	for (rank = 1; rank < ntasks; ++rank) {
		work = 0/* get_next_work_request */;
		MPI_Send(&work,         /* message buffer */
            1,              /* one data item */
            MPI_INT,        /* data item is an integer */
            rank,           /* destination process rank */
            TRAIN_TAG,        /* user chosen message tag */
            MPI_COMM_WORLD);/* always use this */
	}

    /*
    * Receive results for local summary
    */
	for (rank = 1; rank < ntasks; ++rank) {
        // cout<<"Master: receive local summary."<<endl;

        double *rece_local_M = new double[samples];
        double *rece_local_C = new double[samples*samples];

		MPI_Recv(rece_local_M, samples, MPI_DOUBLE, rank, MEAN_TAG, MPI_COMM_WORLD, &status);
        MPI_Recv(rece_local_C, samples*samples, MPI_DOUBLE, rank, COVAR_TAG, MPI_COMM_WORLD, &status);

        for(int i=0;i<samples;i++){
            local_M(i, 0) = rece_local_M[i];
            for(int j=0;j<samples;j++){
                local_C(i,j) = rece_local_C[i*samples+j];
            }
        }

        // caculate for global summary
        global_M = global_M + local_M;
        global_C = global_C + local_C;
	}

    /*
    * Tell all the slaves about the global summary
    */
    for (rank = 1; rank < ntasks; ++rank) {
        double *send_global_M = new double[samples];
        double *send_global_C = new double[samples*samples];

        for(int i=0;i<samples;i++){
            send_global_M[i] = global_M(i, 0);
            for(int j=0;j<samples;j++){
                send_global_C[i*samples+j] = global_C(i,j);
            }
        }

		MPI_Send(send_global_M, samples, MPI_DOUBLE, rank, MEAN_TAG, MPI_COMM_WORLD);
        MPI_Send(send_global_C, samples*samples, MPI_DOUBLE, rank, COVAR_TAG, MPI_COMM_WORLD);
	}

    /*
    * Receive results for final prediction
    */
    int idx = 0;
	for (rank = 1; rank < ntasks; ++rank) {
        double *rece_pred_M = new double[partition[rank]];

		MPI_Recv(rece_pred_M, partition[rank], MPI_DOUBLE, rank, MEAN_TAG, MPI_COMM_WORLD, &status);

        for(int i=0;i<partition[rank];i++){
            pred[idx+i] = rece_pred_M[i];
        }

        idx += partition[rank];
	}

    /*
    * Tell all the slaves to exit.
    */
	for (rank = 1; rank < ntasks; ++rank) {
        work = 0;
		MPI_Send(&work, 0, MPI_INT, rank, DIE_TAG, MPI_COMM_WORLD);
	}
}

void slave(mat S, mat D, mat yD, mat U, float (*Kernel)(mat M1, mat M2)){
	int work;
    int samples = S.n_rows;
    
    mat SD, DD, DS, SS, inv_DD_S, local_M, local_C;
    mat global_M = zeros<mat>(samples, 1);
    mat global_C = zeros<mat>(samples, samples);

	MPI_Status status;

	while(true){
		MPI_Recv(&work, 1, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        /*
        * Check the tag of the received message.
        */
		if (status.MPI_TAG == DIE_TAG) {
			return;
		}
        else {
            // Calculate for local summary
            SD = covariance(S, D, Kernel);
            DD = covariance(D, D, Kernel);
            DS = covariance(D, S, Kernel);
            SS = covariance(S, S, Kernel);
            inv_DD_S = inv(DD-DS*inv(SS)*SD);
            local_M = SD*inv_DD_S*yD;
            local_C = SD*inv_DD_S*DS;

            double *send_local_M = new double[samples];
            double *send_local_C = new double[samples*samples];

            for(int i=0;i<samples;i++){
                send_local_M[i] = local_M(i, 0);
                for(int j=0;j<samples;j++){
                    send_local_C[i*samples+j] = local_C(i,j);
                }
            }

            MPI_Send(send_local_M, samples, MPI_DOUBLE, 0, MEAN_TAG, MPI_COMM_WORLD);
            MPI_Send(send_local_C, samples*samples, MPI_DOUBLE, 0, COVAR_TAG, MPI_COMM_WORLD);

            // Receive global summary
            double *rece_global_M = new double[samples];
            double *rece_global_C = new double[samples*samples];

            MPI_Recv(rece_global_M, samples, MPI_DOUBLE, 0, MEAN_TAG, MPI_COMM_WORLD, &status);
            MPI_Recv(rece_global_C, samples*samples, MPI_DOUBLE, 0, COVAR_TAG, MPI_COMM_WORLD, &status);

            for(int i=0;i<samples;i++){
                global_M(i, 0) = rece_global_M[i];
                for(int j=0;j<samples;j++){
                    global_C(i,j) = rece_global_C[i*samples+j];
                }
            }

            // Caculate for predictions
            mat UU = covariance(U, U, Kernel);
            mat US = covariance(U, S, Kernel);
            mat SU = covariance(S, U, Kernel);
            mat UD = covariance(U, D, Kernel);
            mat DU = covariance(D, U, Kernel);
            mat local_US = UD*inv_DD_S*DS;
            mat local_SU = SD*inv_DD_S*DU;
            mat local_UU = UD*inv_DD_S*DU;
            mat pred_mean = US*pinv(global_C)*global_M;
            mat pred_covar = UU - US*(pinv(SS)-pinv(global_C))*SU;

            int pred_samples = pred_mean.n_elem;

            double *send_pred_M = new double[pred_samples];
            for(int i=0; i<pred_mean.n_elem;i++){
                send_pred_M[i] = pred_mean(i,0);
            }

            MPI_Send(send_pred_M, pred_samples, MPI_DOUBLE, 0, MEAN_TAG, MPI_COMM_WORLD);
        }
	}
}

int main(int argc, char *argv[]){
    clock_t begin = clock();
    
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

	int myrank;
	MPI_Init(&argc, &argv);   /* initialize MPI */
	MPI_Comm_rank(
        MPI_COMM_WORLD,   /* always use this */
        &myrank);      /* process rank, 0 thru N-1 */

	if (myrank == 0) {
        mat pred_M = zeros<mat>(all_samples-all_samples/2, 1);
        master(support, pred, partitions, Kernel);
        for(int i=0;i<(all_samples-all_samples/2);i++){
            pred_M(i, 0) = pred[i];
        }
	} else {
        mat my_train_data = train_data.rows((myrank-1)*intervals, myrank*intervals-1);
        mat my_train_target = train_target.rows((myrank-1)*intervals, myrank*intervals-1);
        mat my_test_data = test_data.rows((myrank-1)*intervals, myrank*intervals-1);

		slave(support, my_train_data, my_train_target, my_test_data, Kernel);
	}
    
	MPI_Finalize();       /* cleanup MPI */

    clock_t end = clock();
    double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
    cout << "Total time for " << NUM_SAMPLES << " samples: " << time_spent << endl;
    
    return(0);
}
