# SofaCUDALinearSolver

A plugin for SOFA providing direct linear solver on GPU.
The implementation is based on CUDA and the cuSolver library.

Documentation: https://sofadefrost.github.io/SofaCUDALinearSolver/

## Dependencies

CUDA toolkit

## Build instructions

Follow the instructions from https://www.sofa-framework.org/community/doc/plugins/build-a-plugin-from-sources/ as any SOFA plugin.

## Description

Parallelization of direct linear solver is a hard problem.
The use case is for large linear systems.
In SOFA, it means a high number of degrees of freedom.
Depending on your hardware, there is a threshold on the number of DoFs: below this threshold it is faster to use an efficient CPU-based solver, and above this threshold it is more interesting to go with a GPU-based solver.

The solver provided in this plugin is based on the Cholesky factorization and has the following features, making it more efficient:

### Fill-in reduction

Fill-in is reduced using permutations. The followings methods are available:
  - "None: No permutation
  - "RCM": Symmetric Reverse Cuthill-McKee permutation
  - "AMD": Symmetric Approximate Minimum Degree Algorithm based on Quotient Graph
  - "METIS": nested dissection

### Symbolic Analysis

The matrix decomposition is divided in the symbolic analysis and the numerical factorization.
The symbolic analysis computes the shape of the decomposition factors.
It is entirely depending on the shape of the input matrix.
Meaning that if the input matrix does not change its shape, the symbolic analysis will be computed only once.
The symbolic analysis runs only if the input matrix changes.

### Data transfer

If the input matrix does not change, it is only necessary to copy the non-zero values from the CPU to the GPU.

## To do

- In some cases, the solving part (2 triangular systems) is faster on the CPU than on the GPU, but the numerical factorization is faster on the GPU. An hybrid approach (GPU and CPU) would help reach better performances.
- cuSolver provides an API for refactorization based on the LU decomposition. See if it can be interesting.
- Make the solver asynchronous
