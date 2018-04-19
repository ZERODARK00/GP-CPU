https://github.com/ZERODARK00/Parallel-GP

# For Parallel GP-CPU

The implementation of this parallel Guassian Process on CPU is based on this [paper](https://arxiv.org/abs/1305.5826).

## Requirements

- BLAS
- LAPACK
- MPI

## How to compile and run it

`mpic++ pic_gpr.cpp operators.cpp -o pic_gpr -lblas -llapack`

`mpirun -np 6 pic_gpr`


# For Parallel GP-GPU

The implementation for this parallel Gaussian Process on GPU can be found in gpu_impl.

## Requirements

- CUDA 8.0
- Compute capability >=3.5

## How to compile and run it

`nvcc -arch=compute_35 -rdc=true pic_gpr.cu operators.cu operators.cpp -o pic_gpr -lcublas_device -lcublas -lcudadevrt -std=c++11 -g -G`
