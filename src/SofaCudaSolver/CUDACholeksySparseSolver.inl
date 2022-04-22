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

#include "CUDACholeksySparseSolver.h"
#include <sofa/helper/ScopedAdvancedTimer.h>


namespace sofa::component::linearsolver::direct
{

template<class TMatrix , class TVector>
CUDASparseCholeskySolver<TMatrix,TVector>::CUDASparseCholeskySolver()
    : f_verbose( initData(&f_verbose,false,"verbose","Dump system state at each iteration") )
    {
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
        host_perm = nullptr;
        device_perm = nullptr;
        host_map = nullptr;
        host_RowPtrPermuted = nullptr;
        host_ColsIndPermuted = nullptr;
        host_valuesPermuted = nullptr;
        device_RowPtrPermuted = nullptr;
        device_ColsIndPermuted = nullptr;
        device_valuesPermuted = nullptr;
        device_x = nullptr;
        host_x_Permuted = nullptr;
        device_x_Permuted = nullptr;
        device_b = nullptr;
        device_b_Permuted = nullptr;
        host_b_Permuted = nullptr;
        buffer_cpu = nullptr;

        singularity = 0;
        tol = 0.000001;

    }

template<class TMatrix , class TVector>
CUDASparseCholeskySolver<TMatrix,TVector>::~CUDASparseCholeskySolver()
{}

template<class TMatrix , class TVector>
void CUDASparseCholeskySolver<TMatrix,TVector>::solve(Matrix& M, Vector& x, Vector& b)
{
    
    checkCudaErrors(cudaMalloc(&device_x, sizeof(double)*colsA));
    checkCudaErrors(cudaMalloc(&device_b, sizeof(double)*colsA));
    
    checkCudaErrors(cudaMemcpyAsync(device_b, (double*)b.ptr(), sizeof(double)*colsA, cudaMemcpyHostToDevice, stream));
    checkCudaErrors(cudaMemcpyAsync(device_x, (double*)x.ptr(),sizeof(double)*colsA, cudaMemcpyHostToDevice, stream));
   
    cudaDeviceSynchronize();

    int reorder = 0 ;
    
    checksolver(cusolverSpDcsrlsvchol(
        handle, rowsA, nnz, descr, device_values, device_RowPtr,
        device_ColsInd, device_b , tol, reorder, device_x,
        &singularity));
    
    checkCudaErrors(cudaDeviceSynchronize());

    checkCudaErrors( cudaMemcpy( (double*)x.ptr(), device_x, sizeof(double)*colsA, cudaMemcpyDeviceToHost));

    checkCudaErrors(cudaFree(device_x));
    checkCudaErrors(cudaFree(device_b));
    checkCudaErrors(cudaFree(device_RowPtr));
    checkCudaErrors(cudaFree(device_ColsInd));
    checkCudaErrors(cudaFree(device_values));
    checkCudaErrors(cudaFree(device_perm));

}

template<class TMatrix , class TVector>
void CUDASparseCholeskySolver<TMatrix,TVector>:: invert(Matrix& M)
{
   M.compress();
   rowsA = M.rowBSize();
   colsA = M.colBSize();
   nnz = M.getColsValue().size(); // number of non zero coefficients
   // copy the matrix
   host_RowPtr = (int*) M.getRowBegin().data();
   host_ColsInd = (int*) M.getColsIndex().data();

   if(host_values) free(host_values);
   host_values = (double*)malloc(sizeof(double)*nnz);
   for(int i=0; i<nnz ; i++) host_values[i] = (double) M.getColsValue()[i];

    //  allow memory on device
    checkCudaErrors(cudaMalloc( &device_RowPtr, sizeof(int)*( rowsA +1) ));
    checkCudaErrors(cudaMalloc( &device_ColsInd, sizeof(int)*nnz ));
    checkCudaErrors(cudaMalloc( &device_values, sizeof(double)*nnz ));

    checkCudaErrors(cudaMalloc(&device_perm, sizeof(double)*(colsA) ));

    // send data to the device
    checkCudaErrors( cudaMemcpyAsync( device_RowPtr, host_RowPtr, sizeof(int)*(rowsA+1), cudaMemcpyHostToDevice, stream) );
    checkCudaErrors( cudaMemcpyAsync( device_ColsInd, host_ColsInd, sizeof(int)*nnz, cudaMemcpyHostToDevice, stream ) );
    checkCudaErrors( cudaMemcpyAsync( device_values, host_values, sizeof(double)*nnz, cudaMemcpyHostToDevice, stream ) );

    cudaDeviceSynchronize();
 
    // compute fill reducing permutation
    
    // to-do : add the choice for the permutations

    //to do : apply permutation 
  
}

}// namespace sofa::component::linearsolver::direct
