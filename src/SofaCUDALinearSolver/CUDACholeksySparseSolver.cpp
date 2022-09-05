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
#define SOFA_PLUGIN_CUDASPARSECHOLESKYSOLVER_CPP
#include <SofaCUDALinearSolver/CUDACholeksySparseSolver.inl>
#include <sofa/component/linearsolver/iterative/MatrixLinearSolver.inl>
#include <sofa/core/ObjectFactory.h>

namespace sofa::component::linearsolver::direct
{

using namespace sofa::linearalgebra;

int CUDASparseCholeskySolverClass = core::RegisterObject("Direct linear solver based on Sparse Cholesky factorization, implemented with the cuSOLVER library")
    .add< CUDASparseCholeskySolver< CompressedRowSparseMatrix<float>,FullVector<float> > >()
    .add< CUDASparseCholeskySolver< CompressedRowSparseMatrix<sofa::type::Mat<3, 3, float> >,FullVector<float> > >()
    .add< CUDASparseCholeskySolver< CompressedRowSparseMatrix<double>,FullVector<double> > >()
    .add< CUDASparseCholeskySolver< CompressedRowSparseMatrix<sofa::type::Mat<3, 3, double> >,FullVector<double> > >()
;

using OtherFloatingType = std::conditional_t<std::is_same_v<SReal, double>, float, double>;
template class SOFACUDALINEARSOLVER_API sofa::component::linearsolver::MatrixLinearSolver< CompressedRowSparseMatrix<OtherFloatingType>,FullVector<OtherFloatingType> > ;
template class SOFACUDALINEARSOLVER_API sofa::component::linearsolver::MatrixLinearSolver< CompressedRowSparseMatrix<sofa::type::Mat<3, 3, OtherFloatingType> >,FullVector<OtherFloatingType> > ;

template class SOFACUDALINEARSOLVER_API CUDASparseCholeskySolver< CompressedRowSparseMatrix<float>,FullVector<float> > ;
template class SOFACUDALINEARSOLVER_API CUDASparseCholeskySolver< CompressedRowSparseMatrix<sofa::type::Mat<3, 3, float> >,FullVector<float> > ;
template class SOFACUDALINEARSOLVER_API CUDASparseCholeskySolver< CompressedRowSparseMatrix<double>,FullVector<double> > ;
template class SOFACUDALINEARSOLVER_API CUDASparseCholeskySolver< CompressedRowSparseMatrix<sofa::type::Mat<3, 3, double> >,FullVector<double> > ;

} // namespace sofa::component::linearsolver::direct
