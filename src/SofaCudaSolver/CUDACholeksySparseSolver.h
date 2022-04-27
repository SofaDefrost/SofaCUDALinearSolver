/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this program. If not, see <http://www.gnu.org/licenses/>.        *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#pragma once

#include <SofaCudaSolver/config.h>

#include <sofa/core/behavior/LinearSolver.h>
#include <sofa/component/linearsolver/iterative/MatrixLinearSolver.h>
#include <sofa/simulation/MechanicalVisitor.h>
#include <sofa/linearalgebra/SparseMatrix.h>
#include <sofa/linearalgebra/CompressedRowSparseMatrix.h>
#include <sofa/helper/map.h>
#include <cmath>

#include "cusolverSp.h"
#include "cusolverSp_LOWLEVEL_PREVIEW.h"
#include <cusparse.h>
#include <cuda_runtime.h>
#include <cuda.h>

#include <sofa/helper/OptionsGroup.h>

namespace sofa::component::linearsolver::direct
{

// Direct linear solver based on Sparse Cholesky factorization, implemented with the cuSOLVER library
template<class TMatrix, class TVector>
class CUDASparseCholeskySolver : public sofa::component::linearsolver::MatrixLinearSolver<TMatrix,TVector>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE2(CUDASparseCholeskySolver,TMatrix,TVector),SOFA_TEMPLATE2(sofa::component::linearsolver::MatrixLinearSolver,TMatrix,TVector));

    typedef TMatrix Matrix;
    typedef TVector Vector;
    typedef sofa::component::linearsolver::MatrixLinearSolver<TMatrix,TVector> Inherit;

    Data<sofa::helper::OptionsGroup> d_typePermutation;

    int rowsA;///< numbuer of rows
    int colsA;///< number of columns
    int nnz;///< number of non zero elements

    int singularity;
    double tol;

    // csr format
    int* host_RowPtr; 
    int* host_ColsInd; 
    double* host_values;

    int* device_RowPtr;
    int* device_ColsInd;
    double* device_values;

    int* host_perm;
    int* device_perm;
    int* host_map;

    double* host_b_permuted;
    double* host_x_permuted;

    cusolverSpHandle_t handle;
    cudaStream_t stream;
    cusparseHandle_t cusparseHandle;
    cusparseMatDescr_t descr;

    csrcholInfo_t device_info ;

    size_t size_internal;
    size_t size_work;
    size_t size_perm;

    double* device_x;
    double* device_b;

    void* buffer_gpu;
    void* buffer_cpu;
   
    bool notSameShape;

    int previous_n;
    int previous_nnz;

    sofa::type::vector<int> previous_ColsInd;
    sofa::type::vector<int> previous_RowPtr;

    CUDASparseCholeskySolver();
    ~CUDASparseCholeskySolver();
    void solve (Matrix& M, Vector& x, Vector& b) override;
    void invert(Matrix& M) override;
    
};
// compare the shape of 2 matrix given in csr format, return true if the don't have the same shape
bool compareMatrixShape(const int,const int *,const int *,const int,const int *,const int *) ;


#define checkCudaErrors(err) __checkCudaErrors(err, __FILE__, __LINE__)

inline void __checkCudaErrors(cudaError err, const char *file, const int line) {
  if (cudaSuccess != err) {
    fprintf(stderr, "%s(%i) : CUDA Runtime API error %d: %s.\n", file, line,
            (int)err, cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}

#define checksolver(status ) __checksolver(status, __FILE__, __LINE__)
inline void __checksolver( cusolverStatus_t status, const char *file, const int line){
    if(status != 0)
    {
        std::cout<<'\n'<< "Cuda Failure in" << file << " at line "<< line <<'\n'<< std::endl;
        exit(EXIT_FAILURE);
    }
}

#if  !defined(SOFA_PLUGIN_CUDASPARSECHOLESKYSOLVER_CPP)
extern template class SOFACUDASOLVER_API CUDASparseCholeskySolver< sofa::linearalgebra::CompressedRowSparseMatrix<SReal>, sofa::linearalgebra::FullVector<SReal> > ;
#endif

} // namespace sofa::component::linearsolver::direct
