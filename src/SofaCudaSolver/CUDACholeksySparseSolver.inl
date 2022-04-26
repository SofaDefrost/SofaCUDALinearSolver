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

#include <SofaCudaSolver/CUDACholeksySparseSolver.h>
#include <sofa/helper/ScopedAdvancedTimer.h>


namespace sofa::component::linearsolver::direct
{

template<class TMatrix , class TVector>
CUDASparseCholeskySolver<TMatrix,TVector>::CUDASparseCholeskySolver()
    : Inherit1()
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

        device_x = nullptr;
        device_b = nullptr;

        buffer_gpu = nullptr;
        device_info = nullptr;

        singularity = 0;
        tol = 0.000001;

        size_internal = 0;
        size_work = 0;

        notSameShape = true;

        nnz = 0;

        previous_n = 0 ;
        previous_nnz = 0;
        rowsA = 0;
        previous_ColsInd.clear() ;
        previous_RowPtr.clear() ;

    }

template<class TMatrix , class TVector>
CUDASparseCholeskySolver<TMatrix,TVector>::~CUDASparseCholeskySolver()
{
    checkCudaErrors(cudaFree(device_x));
    checkCudaErrors(cudaFree(device_b));
    checkCudaErrors(cudaFree(device_RowPtr));
    checkCudaErrors(cudaFree(device_ColsInd));
    checkCudaErrors(cudaFree(device_values));

    cusolverSpDestroy(handle);
    cusolverSpDestroyCsrcholInfo(device_info);
    cusparseDestroyMatDescr(descr);
    cudaStreamDestroy(stream);
}

template<class TMatrix , class TVector>
void CUDASparseCholeskySolver<TMatrix,TVector>::solve(Matrix& M, Vector& x, Vector& b)
{
    int n = M.colBSize()  ; // avoidable, used to prevent compilation warning
    
    checkCudaErrors(cudaMemcpyAsync(device_b, (double*)b.ptr(), sizeof(double)*n, cudaMemcpyHostToDevice, stream));
    checkCudaErrors(cudaMemcpyAsync(device_x, (double*)x.ptr(),sizeof(double)*n, cudaMemcpyHostToDevice, stream));
   
    cudaDeviceSynchronize();

    {
        sofa::helper::ScopedAdvancedTimer solveTimer("Solve");
        checksolver( cusolverSpDcsrcholSolve( handle, n, device_b, device_x, device_info, buffer_gpu ) );
    }

    checkCudaErrors(cudaDeviceSynchronize());

    checkCudaErrors( cudaMemcpyAsync( (double*)x.ptr(), device_x, sizeof(double)*n, cudaMemcpyDeviceToHost,stream));

    cudaDeviceSynchronize();

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
    host_values = (double*) M.getColsValue().data();

   
    notSameShape = compareMatrixShape(rowsA , host_ColsInd, host_RowPtr, previous_RowPtr.size()-1,  previous_ColsInd.data(), previous_RowPtr.data() );
    std::cout<< notSameShape << std::endl;
    //std::cout << rowsA << ' ' << previous_RowPtr.size() << std::endl;
    //std::cout<< previous_ColsInd[previous_ColsInd.size()-1] << ' ' << previous_RowPtr[previous_RowPtr.size()-1] << std::endl;
    //std::cout << host_RowPtr[rowsA+1] << ' ' << host_ColsInd[nnz-1] << std::endl;
    //std::cout << M.getColsIndex().size() << ' ' << rowsA << std::endl;

    // allocate memory on device
    if(previous_n != rowsA) 
    {
        checkCudaErrors(cudaMalloc( &device_RowPtr, sizeof(int)*( rowsA +1) ));

        checkCudaErrors(cudaMalloc(&device_x, sizeof(double)*colsA));
        checkCudaErrors(cudaMalloc(&device_b, sizeof(double)*colsA));
    }

    if(previous_nnz != nnz)
    {
        if(device_ColsInd) cudaFree(device_ColsInd);
        checkCudaErrors(cudaMalloc( &device_ColsInd, sizeof(int)*nnz ));
        if(device_values) cudaFree(device_values);
        checkCudaErrors(cudaMalloc( &device_values, sizeof(double)*nnz ));
    }
    // send data to the device
    checkCudaErrors( cudaMemcpyAsync( device_RowPtr, host_RowPtr, sizeof(int)*(rowsA+1), cudaMemcpyHostToDevice, stream) );
    checkCudaErrors( cudaMemcpyAsync( device_ColsInd, host_ColsInd, sizeof(int)*nnz, cudaMemcpyHostToDevice, stream ) );
    checkCudaErrors( cudaMemcpyAsync( device_values, host_values, sizeof(double)*nnz, cudaMemcpyHostToDevice, stream ) );

    cudaDeviceSynchronize();
    
    // factorize on device
    notSameShape = true; 
    if(notSameShape)
    {
        if(device_info) cusolverSpDestroyCsrcholInfo(device_info);
        checksolver(cusolverSpCreateCsrcholInfo(&device_info));
        {
            sofa::helper::ScopedAdvancedTimer symbolicTimer("Symbolic factorization");
            checksolver( cusolverSpXcsrcholAnalysis( handle, rowsA, nnz, descr, device_RowPtr, device_ColsInd, device_info ) ); // symbolic decomposition
        }
    }

    checksolver( cusolverSpDcsrcholBufferInfo( handle, rowsA, nnz, descr, device_values, device_RowPtr, device_ColsInd,
                                            device_info, &size_internal, &size_work ) ); //set workspace

     
    if(buffer_gpu) cudaFree(buffer_gpu);
    checkCudaErrors(cudaMalloc(&buffer_gpu, sizeof(char)*size_work));
    {
        sofa::helper::ScopedAdvancedTimer numericTimer("Numeric factorization");
        checksolver(cusolverSpDcsrcholFactor( handle, rowsA, nnz, descr, device_values, device_RowPtr, device_ColsInd,
                            device_info, buffer_gpu )); // numeric decomposition
    }

    //store the shape of the matrix
    previous_n = rowsA ;
    previous_nnz = nnz ;
    previous_RowPtr.resize( rowsA + 1 );
    previous_ColsInd.resize( nnz );
    previous_ColsInd.resize( nnz );
    for(int i=0;i<nnz;i++) previous_ColsInd[i] = host_ColsInd[i];
    for(int i=0; i<rowsA +1; i++) previous_RowPtr[i] = host_RowPtr[i];


    // compute fill reducing permutation
    
    // to-do : add the choice for the permutations

    //to do : apply permutation 

}

bool compareMatrixShape(const int s_M,const int * M_colind,const int * M_rowptr,const int s_P,const int * P_colind,const int * P_rowptr) {
    if (s_M != s_P) return true;
    //std::cout << 1.1 <<std::endl;
    if (M_rowptr[s_M] != P_rowptr[s_M] ) return true; 
    //std::cout << 1.2 <<std::endl;

    for (int i=0;i<s_P;i++) {
        if (M_rowptr[i]!=P_rowptr[i]) return true; 
    }
    //std::cout << 1.3 <<std::endl;

    for (int i=0;i<M_rowptr[s_M];i++) {
        if (M_colind[i]!=P_colind[i]) return true;
    }
    //std::cout << 1.3 <<std::endl;
    return false;
}



}// namespace sofa::component::linearsolver::direct
