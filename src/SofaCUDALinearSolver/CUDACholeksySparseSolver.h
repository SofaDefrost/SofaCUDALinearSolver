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

#include <SofaCUDALinearSolver/config.h>

#include <sofa/core/behavior/LinearSolver.h>
#include <sofa/component/linearsolver/iterative/MatrixLinearSolver.h>
#include <sofa/simulation/MechanicalVisitor.h>
#include <sofa/helper/OptionsGroup.h>
#include <cusolverSp.h>
#include <cusolverSp_LOWLEVEL_PREVIEW.h>

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
    typedef typename Matrix::Real Real;
    typedef sofa::component::linearsolver::MatrixLinearSolver<TMatrix,TVector> Inherit;

    void solve (Matrix& M, Vector& x, Vector& b) override;
    void invert(Matrix& M) override;
    void solveOnGPU(int n);

private:

    Data<sofa::helper::OptionsGroup> d_typePermutation;

    int rows;///< number of rows
    int cols;///< number of columns
    int nnz;///< number of non zero elements

    int singularity;

    // csr format
    int* host_RowPtr; 
    int* host_ColsInd;
    Real* host_values;

    int* host_RowPtr_permuted;
    int* host_ColsInd_permuted;
    Real* host_values_permuted;

    int* device_RowPtr;
    int* device_ColsInd;
    Real* device_values;

    int* host_perm;
    int* host_map;

    Real* host_b_permuted;
    Real* host_x_permuted;

    cusolverSpHandle_t handle;
    cudaStream_t stream;
    cusparseHandle_t cusparseHandle;
    cusparseMatDescr_t descr;

    csrcholInfo_t device_info ;

    size_t size_internal;
    size_t size_work;
    size_t size_perm;

    Real* device_x;
    Real* device_b;

    void* buffer_gpu;
    void* buffer_cpu;
   
    bool notSameShape;

    int previous_n;
    int previous_nnz;

    int reorder;

    sofa::type::vector<int> previous_ColsInd;
    sofa::type::vector<int> previous_RowPtr;

    CUDASparseCholeskySolver();
    ~CUDASparseCholeskySolver() override;
    void setWorkspace();
    void numericFactorization();

    sofa::linearalgebra::CompressedRowSparseMatrix<Real> m_filteredMatrix;
    
};
// compare the shape of 2 matrices given in csr format, return true if the matrices don't have the same shape
bool compareMatrixShape(int, const int *,const int *, int,const int *,const int *) ;

#if !defined(SOFA_PLUGIN_CUDASPARSECHOLESKYSOLVER_CPP)
    extern template class SOFACUDALINEARSOLVER_API CUDASparseCholeskySolver< CompressedRowSparseMatrix<float>,FullVector<float> > ;
    extern template class SOFACUDALINEARSOLVER_API CUDASparseCholeskySolver< CompressedRowSparseMatrix<sofa::type::Mat<3, 3, float> >,FullVector<float> > ;
    extern template class SOFACUDALINEARSOLVER_API CUDASparseCholeskySolver< CompressedRowSparseMatrix<double>,FullVector<double> > ;
    extern template class SOFACUDALINEARSOLVER_API CUDASparseCholeskySolver< CompressedRowSparseMatrix<sofa::type::Mat<3, 3, double> >,FullVector<double> > ;
#endif

} // namespace sofa::component::linearsolver::direct
