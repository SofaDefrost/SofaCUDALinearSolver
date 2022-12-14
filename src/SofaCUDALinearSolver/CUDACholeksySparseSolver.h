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
    void solve_impl(int n, Real* b, Real* x);

private:

    Data<sofa::helper::OptionsGroup> d_typePermutation;
    Data<sofa::helper::OptionsGroup> d_hardware;

    int rows;///< number of rows
    int cols;///< number of columns
    int nnz;///< number of non zero elements

    int singularity;

    // csr format
    int* host_RowPtr; 
    int* host_ColsInd;
    Real* host_values;

    // CRS format of the permuted matrix
    sofa::type::vector<int> host_rowPermuted;
    sofa::type::vector<int> host_colPermuted;
    sofa::type::vector<Real> host_valuePermuted;

    int* device_RowPtr;
    int* device_ColsInd;
    Real* device_values;

    sofa::type::vector<int> host_perm;
    sofa::type::vector<int> host_map;

    sofa::type::vector<Real> host_bPermuted;
    sofa::type::vector<Real> host_xPermuted;

    cusolverSpHandle_t handle;
    cudaStream_t stream;
    cusparseHandle_t cusparseHandle;
    cusparseMatDescr_t descr;

    csrcholInfo_t device_info ;
    csrcholInfoHost_t host_info;

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
    void createCholeskyInfo();
    void symbolicFactorization();

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
