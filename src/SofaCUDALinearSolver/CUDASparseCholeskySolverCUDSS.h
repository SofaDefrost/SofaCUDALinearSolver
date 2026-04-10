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

#include <sofa/linearalgebra/CompressedRowSparseMatrix.h>
#include <sofa/core/behavior/LinearSolver.h>
#include <sofa/component/linearsolver/iterative/MatrixLinearSolver.h>
#include <sofa/simulation/MechanicalVisitor.h>
#include <sofa/helper/OptionsGroup.h>

// cuDSS for GPU path (modern high-performance sparse direct solver)
#include <cudss.h>

// cusolverSp for CPU path (host functions)
#include <cusolverSp.h>
#include <cusolverSp_LOWLEVEL_PREVIEW.h>

namespace sofa::component::linearsolver::direct
{

// Direct linear solver based on Sparse Cholesky factorization, implemented with the cuDSS library (GPU) or cuSOLVER (CPU)
template<class TMatrix, class TVector>
class CUDASparseCholeskySolverCUDSS : public sofa::component::linearsolver::MatrixLinearSolver<TMatrix,TVector>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE2(CUDASparseCholeskySolverCUDSS,TMatrix,TVector),SOFA_TEMPLATE2(sofa::component::linearsolver::MatrixLinearSolver,TMatrix,TVector));

    typedef TMatrix Matrix;
    typedef TVector Vector;
    typedef typename Matrix::Real Real;
    typedef sofa::component::linearsolver::MatrixLinearSolver<TMatrix,TVector> Inherit;

    void solve (Matrix& M, Vector& x, Vector& b) override;
    void invert(Matrix& M) override;

private:

    Data<sofa::helper::OptionsGroup> d_typePermutation;
    Data<sofa::helper::OptionsGroup> d_hardware;

    int rows;   ///< number of rows
    int cols;   ///< number of columns
    int nnz;    ///< number of non zero elements

    // Host pointers to matrix data (points into m_filteredMatrix)
    int* host_RowPtr;
    int* host_ColsInd;
    Real* host_values;

    // Device memory for matrix (CSR format)
    int* device_RowPtr;
    int* device_ColsInd;
    Real* device_values;

    // Device memory for solution and RHS vectors
    Real* device_x;
    Real* device_b;

    // CUDA stream
    cudaStream_t stream;

    // ============ cuDSS (GPU path) ============
    cudssHandle_t cudssHandle;
    cudssConfig_t cudssConfig;
    cudssData_t cudssData;
    cudssMatrix_t cudssMatrixA;
    cudssMatrix_t cudssMatrixX;
    cudssMatrix_t cudssMatrixB;
    bool cudssInitialized;

    // ============ cusolverSp (CPU path) ============
    cusolverSpHandle_t cusolverHandle;
    cusparseMatDescr_t cusparseDescr;
    csrcholInfoHost_t host_info;
    void* buffer_cpu;
    size_t size_internal_cpu;
    size_t size_work_cpu;

    // CPU path permutation data
    int reorder;
    sofa::type::vector<int> host_perm;
    sofa::type::vector<int> host_map;
    sofa::type::vector<int> host_rowPermuted;
    sofa::type::vector<int> host_colPermuted;
    sofa::type::vector<Real> host_valuePermuted;
    sofa::type::vector<Real> host_bPermuted;
    sofa::type::vector<Real> host_xPermuted;
    size_t size_perm;
    void* buffer_perm;

    // ============ Common state ============
    bool notSameShape;
    int previous_n;
    int previous_nnz;

    sofa::type::vector<int> previous_ColsInd;
    sofa::type::vector<int> previous_RowPtr;

    CUDASparseCholeskySolverCUDSS();
    ~CUDASparseCholeskySolverCUDSS() override;

    // GPU path methods (cuDSS)
    void initCuDSS();
    void cleanupCuDSS();
    void invertGPU();
    void solveGPU(int n, Real* b_host, Real* x_host);

    // CPU path methods (cusolverSp)
    void initCuSolverHost();
    void cleanupCuSolverHost();
    void invertCPU();
    void solveCPU(int n, Real* b_host, Real* x_host);

    sofa::linearalgebra::CompressedRowSparseMatrix<Real> m_filteredMatrix;

};

// compare the shape of 2 matrices given in csr format, return true if the matrices don't have the same shape
bool compareMatrixShape(int, const int *,const int *, int,const int *,const int *) ;

#if !defined(SOFA_PLUGIN_CUDASPARSECHOLESKYSOLVERCUDSS_CPP)
    extern template class SOFACUDALINEARSOLVER_API CUDASparseCholeskySolverCUDSS< CompressedRowSparseMatrix<float>,FullVector<float> > ;
    extern template class SOFACUDALINEARSOLVER_API CUDASparseCholeskySolverCUDSS< CompressedRowSparseMatrix<sofa::type::Mat<3, 3, float> >,FullVector<float> > ;
    extern template class SOFACUDALINEARSOLVER_API CUDASparseCholeskySolverCUDSS< CompressedRowSparseMatrix<double>,FullVector<double> > ;
    extern template class SOFACUDALINEARSOLVER_API CUDASparseCholeskySolverCUDSS< CompressedRowSparseMatrix<sofa::type::Mat<3, 3, double> >,FullVector<double> > ;
#endif

} // namespace sofa::component::linearsolver::direct
