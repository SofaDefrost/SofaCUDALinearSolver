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

#include"cusolverSp.h"

namespace sofa::component::linearsolver::direct
{

// Direct linear solver based on Sparse Cholesky factorization, implemented with the cuSOLVER library
template<class TMatrix, class TVector>
class CUDASparseCholeskySolver : public sofa::component::linearsolver::MatrixLinearSolver<TMatrix,TVector>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE2(CUDASparseCholeskySolver,TMatrix,TVector),SOFA_TEMPLATE2(sofa::component::linearsolver::MatrixLinearSolver,TMatrix,TVector));

    Data<bool> f_verbose; ///< Dump system state at each iteration

    typedef TMatrix Matrix;
    typedef TVector Vector;
    typedef sofa::component::linearsolver::MatrixLinearSolver<TMatrix,TVector> Inherit;

    int rowsA;///< numbuer of rows
    int colsA;///< number of columns
    int nnz;///< number of non zero elements

    // csr format
    type::vector<int> host_RowPtr; 
    type::vector<int> host_ColsInd; 
    type::vector<double> host_values;

    type::vector<double> host_perm;///< fill-in reducing permutation

    cusolverSpHandle_t handle;
    cudaStream_t stream;
    cusparseMatDescr_t descr; 
   
    CUDASparseCholeskySolver();
    ~CUDASparseCholeskySolver();
    void solve (Matrix& M, Vector& x, Vector& b) override;
    void invert(Matrix& M) override;
    
};

#if  !defined(SOFA_PLUGIN_CUDASPARSECHOLESKYSOLVER_CPP)
extern template class SOFACUDASOLVER_API CUDASparseCholeskySolver< sofa::linearalgebra::CompressedRowSparseMatrix<SReal>, sofa::linearalgebra::FullVector<SReal> > ;
#endif

} // namespace sofa::component::linearsolver::direct