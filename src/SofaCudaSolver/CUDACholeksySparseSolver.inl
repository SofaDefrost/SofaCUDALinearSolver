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
    
    cudaMalloc(&device_x, sizeof(double)*colsA);
    cudaMalloc(&device_x_Permuted, sizeof(double)*colsA);
    cudaMalloc(&device_b, sizeof(double)*colsA);
    

    cudaMemcpyAsync(device_RowPtr, host_RowPtr, sizeof(int)*(rowsA +1), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(device_ColsInd, host_ColsInd, sizeof(int)*nnz, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(device_values, host_values, sizeof(double)*nnz , cudaMemcpyHostToDevice, stream);

    cudaMemcpyAsync(device_RowPtrPermuted, host_RowPtrPermuted, sizeof(int)*(rowsA +1), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(device_ColsIndPermuted, host_ColsIndPermuted, sizeof(int)*nnz, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(device_valuesPermuted, host_valuesPermuted, sizeof(double)*nnz, cudaMemcpyHostToDevice, stream);

    cudaMemcpyAsync(device_b, (double*)b.ptr(), sizeof(double)*nnz, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(device_x, (double*)x.ptr(),sizeof(double)*nnz, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(device_perm, host_perm, sizeof(int)*colsA, cudaMemcpyHostToDevice, stream);

    if(host_b_Permuted) free(host_b_Permuted);
    host_b_Permuted = (double*)malloc(sizeof(double)*colsA);
    for(int i=0; i<colsA; i++) host_b_Permuted[i] = b.ptr()[ host_perm[i] ];

    cudaMemcpyAsync( device_b_Permuted, host_b_Permuted, sizeof(double)*colsA, cudaMemcpyHostToDevice, stream  );

    int reorder = 0 ;
       
    cusolverSpDcsrlsvchol(
        handle, rowsA, nnz, descr, device_valuesPermuted, device_RowPtrPermuted,
        device_ColsIndPermuted, device_b_Permuted , tol, reorder, device_x_Permuted,
        &singularity);
   
    cudaDeviceSynchronize();

    cudaFree(device_x);
    cudaFree(device_x_Permuted);
    cudaFree(device_b);
    cudaFree(device_b_Permuted);
    cudaFree(device_RowPtr);
    cudaFree(device_ColsInd);
    cudaFree(device_values);
    cudaFree(device_valuesPermuted);
    cudaFree(device_RowPtrPermuted);
    cudaFree(device_ColsIndPermuted);
    cudaFree(device_perm);
    
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
    cudaMalloc( &device_RowPtr, sizeof(int)*( rowsA +1) );
    cudaMalloc( &device_ColsInd, sizeof(int)*nnz );
    cudaMalloc( &device_values, sizeof(double)*nnz );

    cudaMalloc( &device_RowPtrPermuted, sizeof(int) *(rowsA +1));
    cudaMalloc( &device_ColsIndPermuted, sizeof(int)*nnz );
    cudaMalloc( &device_valuesPermuted, sizeof(double)*nnz );

    cudaMalloc(&device_x, sizeof(double)*(colsA) );
    cudaMalloc(&device_b, sizeof(double)*(colsA) );
    cudaMalloc(&device_b_Permuted, sizeof(double)*(colsA) );
    cudaMalloc(&device_perm, sizeof(double)*(colsA) );
 
    // compute fill reducing permutation
    if(host_perm) free(host_perm);
    host_perm = (int*)malloc(sizeof(int)*colsA);
    for(int i=0; i<colsA ; i++) host_perm[i] = i;
    // to-do : add the choice for the permutations
    cusolverSpXcsrsymrcmHost( handle, rowsA, nnz , descr, host_RowPtr, host_ColsInd, host_perm );

    if(host_RowPtrPermuted) free(host_RowPtrPermuted);
    host_RowPtrPermuted = (int*) malloc(sizeof(int)*(rowsA +1));

    if(host_ColsIndPermuted) free(host_ColsIndPermuted);
    host_ColsIndPermuted = (int*) malloc(sizeof(int)*nnz);

    memcpy(host_RowPtrPermuted, host_RowPtr, sizeof(int)*(rowsA+1));
    memcpy(host_ColsIndPermuted, host_ColsInd, sizeof(int)*(rowsA+1));

    // apply permutation
    size_t size_perm = 0;
    cusolverSpXcsrperm_bufferSizeHost(handle, rowsA, colsA, nnz, descr, host_RowPtrPermuted
                                    , host_ColsIndPermuted, host_perm, host_perm, &size_perm);

    if(host_map) free(host_map);
    host_map = (int*)malloc( sizeof(long unsigned int)*nnz );
    for(int j=0; j<nnz ; j++) host_map[j] = (long unsigned int)j ; // initialized to identity



/*
    if(buffer_cpu) free(buffer_cpu);
    buffer_cpu = (void*)malloc(sizeof(char)*size_perm);


    cusolverSpXcsrpermHost(handle, rowsA, colsA, nnz, descr, host_RowPtrPermuted, host_ColsIndPermuted,
                            host_perm, host_perm, host_map, buffer_cpu);
*/

    if(host_valuesPermuted) free(host_valuesPermuted);
    host_valuesPermuted = (double*) malloc(sizeof(double)*nnz);
    for(int i=0; i<nnz ; i++) host_valuesPermuted[i] = host_values[ host_map[i] ];

}

}// namespace sofa::component::linearsolver::direct
