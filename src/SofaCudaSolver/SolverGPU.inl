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

#include "SolverGPU.h"
#include <sofa/helper/ScopedAdvancedTimer.h>


namespace sofa::component::linearsolver::direct
{

template<class TMatrix , class TVector>
SolverGPU<TMatrix,TVector>::SolverGPU()
    : f_verbose( initData(&f_verbose,false,"verbose","Dump system state at each iteration") )
    , d_typePermutation(initData(&d_typePermutation, "permutation", "Type of fill reducing permutation"))
    , d_typeSolver(initData(&d_typeSolver, "Solver", "Type of linear solver"))
    {
        sofa::helper::OptionsGroup d_typePermutationOptions(4,"None","RCM", "AMD", "METIS");
        d_typePermutationOptions.setSelectedItem(0); // default None
        d_typePermutation.setValue(d_typePermutationOptions);

        sofa::helper::OptionsGroup d_typeSolverOptions(3,"LU","Cholesky","QR");
        d_typeSolverOptions.setSelectedItem(1); // default Cholesky
        d_typeSolver.setValue(d_typeSolverOptions);

        cusolverSpCreate(&handle);
        cusparseCreate(&cusparseHandle);

        cudaStreamCreate(&stream);

        cusolverSpSetStream(handle, stream);
        cusparseSetStream(cusparseHandle, stream);

        cusparseCreateMatDescr(&descr);
        cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);

        cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);

        host_RowPtr = nullptr; 
        host_ColsInd = nullptr; 
        host_values = nullptr;
        device_RowPtr = nullptr;
        device_ColsInd = nullptr;
        device_values = nullptr;
        device_x = nullptr;
        device_b = nullptr;
        buffer_cpu = nullptr;

        singularity = 0;
        tol = 0.000001;

        firstStep = true;

    }

template<class TMatrix , class TVector>
SolverGPU<TMatrix,TVector>::~SolverGPU()
{   
    cusolverSpDestroy(handle);
}

template<class TMatrix , class TVector>
void SolverGPU<TMatrix,TVector>::solve(Matrix& M, Vector& x, Vector& b)
{
    int n = M.colBSize(); // avoidable, used to prevent compilation warning

    if(firstStep)
    {
        checkCudaErrors(cudaMalloc(&device_x, sizeof(double)*n));
        checkCudaErrors(cudaMalloc(&device_b, sizeof(double)*colsA));
    }

    checkCudaErrors(cudaMemcpyAsync(device_b, (double*)b.ptr(), sizeof(double)*colsA, cudaMemcpyHostToDevice, stream));
    checkCudaErrors(cudaMemcpyAsync(device_x, (double*)x.ptr(),sizeof(double)*colsA, cudaMemcpyHostToDevice, stream));
   
    cudaDeviceSynchronize();

    // compute and apply fill-in reducing permutation
    reorder = d_typePermutation.getValue().getSelectedId() ;
    solverType = d_typeSolver.getValue().getSelectedId();

    switch( solverType )
    {
        case 0:
        default:
        if(reorder) reorder = 1; // only RCM available for LU, see Nvidia documentation
        // LU on CPU only? path to GPU version not provided
            checksolver(cusolverSpDcsrlsvluHost(
                    handle, rowsA, nnz, descr, host_values, host_RowPtr,
                    host_ColsInd, b.ptr() , tol, reorder, x.ptr(),
                    &singularity));
            break;

        case 1://Cholesky
            checksolver(cusolverSpDcsrlsvchol(
                handle, rowsA, nnz, descr, device_values, device_RowPtr,
                device_ColsInd, device_b , tol, reorder, device_x,
                &singularity));
            checkCudaErrors(cudaDeviceSynchronize());

            checkCudaErrors( cudaMemcpy( (double*)x.ptr(), device_x,
             sizeof(double)*colsA, cudaMemcpyDeviceToHost));

            break;

        case 2://QR
            checksolver(cusolverSpDcsrlsvqr(
                handle, rowsA, nnz, descr, device_values, device_RowPtr,
                device_ColsInd, device_b , tol, reorder, device_x,
                &singularity));
            checkCudaErrors(cudaDeviceSynchronize());

            checkCudaErrors( cudaMemcpy( (double*)x.ptr(), device_x,
             sizeof(double)*colsA, cudaMemcpyDeviceToHost));

            break;
        
    }
    
    
}

template<class TMatrix , class TVector>
void SolverGPU<TMatrix,TVector>:: invert(Matrix& M)
{
   M.compress();
   rowsA = M.rowBSize();
   colsA = M.colBSize();
   previous_nnz = nnz;
   nnz = M.getColsValue().size(); // number of non zero coefficients
   // copy the matrix
   host_RowPtr = (int*) M.getRowBegin().data();
   host_ColsInd = (int*) M.getColsIndex().data();

    if(previous_nnz != nnz)
    {
            if(host_values) free(host_values);
            host_values = (double*)malloc(sizeof(double)*nnz);
    }
   for(int i=0; i<nnz ; i++) host_values[i] = (double) M.getColsValue()[i];

    //  allow memory on device
    if(firstStep) checkCudaErrors(cudaMalloc( &device_RowPtr, sizeof(int)*( rowsA +1) ));
    if(previous_nnz != nnz) 
    {   
        if(device_ColsInd) checkCudaErrors(cudaFree(device_ColsInd));
        checkCudaErrors(cudaMalloc( &device_ColsInd, sizeof(int)*nnz ));
        if(device_values) checkCudaErrors(cudaFree(device_values));
        checkCudaErrors(cudaMalloc( &device_values, sizeof(double)*nnz ));
    }

    // send data to the device
    checkCudaErrors( cudaMemcpyAsync( device_RowPtr, host_RowPtr, sizeof(int)*(rowsA+1), cudaMemcpyHostToDevice, stream) );
    checkCudaErrors( cudaMemcpyAsync( device_ColsInd, host_ColsInd, sizeof(int)*nnz, cudaMemcpyHostToDevice, stream ) );
    checkCudaErrors( cudaMemcpyAsync( device_values, host_values, sizeof(double)*nnz, cudaMemcpyHostToDevice, stream ) );

    cudaDeviceSynchronize();
  
}

}// namespace sofa::component::linearsolver::direct
