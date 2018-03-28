# For GP-CPU

The implementation of this parallel Guassian Process on CPU is based on this [paper](https://arxiv.org/abs/1305.5826).

## Requirements

- BLAS
- LAPACK
- MPI

## How to compile and run it

`mpic++ pic_gpr.cpp operators.cpp -o pic_gpr -lblas -llapack`

`mpirun -np 6 pic_gpr`
