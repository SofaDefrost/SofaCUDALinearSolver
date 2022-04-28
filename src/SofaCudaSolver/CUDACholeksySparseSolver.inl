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
    , d_typePermutation(initData(&d_typePermutation, "permutation", "Type of fill reducing permutation"))
    {
        sofa::helper::OptionsGroup d_typePermutationOptions(5,"None","RCM" ,"AMD", "METIS","test");
        d_typePermutationOptions.setSelectedItem(4); // default None
        d_typePermutation.setValue(d_typePermutationOptions);

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
        host_values_permuted = nullptr;

        host_b_permuted = nullptr;
        host_x_permuted = nullptr;

        device_RowPtr = nullptr;
        device_ColsInd = nullptr;
        device_values = nullptr;

        device_x = nullptr;
        device_b = nullptr;

        buffer_cpu = nullptr;
        buffer_gpu = nullptr;
        device_info = nullptr;

        host_perm = nullptr;
        device_perm = nullptr;
        host_map = nullptr;

        singularity = 0;
        tol = 0.000001;

        size_internal = 0;
        size_work = 0;
        size_perm = 0;

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

    //if(previous_n != n)
    if(notSameShape && d_typePermutation.getValue().getSelectedId() )
    {
        if(host_b_permuted) free(host_b_permuted);
        host_b_permuted = (double*)malloc(sizeof(double)*n);
        if(host_x_permuted) free(host_x_permuted);
        host_x_permuted = (double*)malloc(sizeof(double)*n);
    }

    if( d_typePermutation.getValue().getSelectedId() == 0 )
    {
        checkCudaErrors( cudaMemcpyAsync( device_b, b.ptr(), sizeof(double)*n, cudaMemcpyHostToDevice,stream));
    }
    else
    {
        for(int i=0;i<n;i++) host_b_permuted[i] = b[ host_perm[i] ];
        checkCudaErrors( cudaMemcpyAsync( device_b, host_b_permuted, sizeof(double)*n, cudaMemcpyHostToDevice,stream));
    }
    cudaDeviceSynchronize();

    {
        sofa::helper::ScopedAdvancedTimer solveTimer("Solve");
        checksolver( cusolverSpDcsrcholSolve( handle, n, device_b, device_x, device_info, buffer_gpu ) );
    }

    checkCudaErrors(cudaDeviceSynchronize());

    if( d_typePermutation.getValue().getSelectedId() == 0 )
    {
        checkCudaErrors( cudaMemcpyAsync( x.ptr(), device_x, sizeof(double)*n, cudaMemcpyDeviceToHost,stream));
         cudaDeviceSynchronize();
    }
    else
    {
        checkCudaErrors( cudaMemcpyAsync( host_x_permuted, device_x, sizeof(double)*n, cudaMemcpyDeviceToHost,stream));
        cudaDeviceSynchronize();

        for(int i=0;i<n;i++) x[host_perm[i]] = host_x_permuted[ i ];
    }
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
        if(host_values_permuted) free(host_values_permuted);
        host_values_permuted = (double*)malloc(sizeof(double)*nnz);
    }

    // compute fill reducing permutation
    if( d_typePermutation.getValue().getSelectedId() )
    {
        if(notSameShape)
        {
            if(host_perm) free(host_perm);
            host_perm =(int*) malloc(sizeof( int )*rowsA );

            switch( d_typePermutation.getValue().getSelectedId() )
            {
                default:
                case 0:// None, identity
                    //for(int i=0;i<rowsA;i++) host_perm[i] = i ;
                    break;

                case 1://RCM, Symmetric Reverse Cuthill-McKee permutation
                    checksolver( cusolverSpXcsrsymrcmHost(handle, rowsA, nnz, descr, host_RowPtr, host_ColsInd, host_perm) );
                    break;
                case 2://AMD, Symetric MinimumDegree Approximation
                    checksolver( cusolverSpXcsrsymamdHost(handle, rowsA, nnz, descr, host_RowPtr, host_ColsInd, host_perm) );
                    break;

                case 3://METIS, nested dissection
                    checksolver( cusolverSpXcsrmetisndHost(handle, rowsA, nnz, descr, host_RowPtr, host_ColsInd, nullptr, host_perm) );
                    break;

                case 4:// None, identity
                    for(int i=0;i<rowsA;i++) host_perm[i] = i ;
                    break;
            }
            checksolver( cusolverSpXcsrperm_bufferSizeHost(handle, rowsA, colsA, nnz, descr, host_RowPtr, host_ColsInd
                                                        , host_perm, host_perm, &size_perm) );

            if(buffer_cpu) free(buffer_cpu);
            buffer_cpu = (void*)malloc(sizeof(char)*size_perm) ;

            if(host_map) free(host_map);
            host_map = (int*)malloc(sizeof(int)*nnz);
            for(int i=0;i<nnz;i++) host_map[i] = i;
        }
        //apply the permutation
        checksolver( cusolverSpXcsrpermHost( handle, rowsA, colsA, nnz, descr, host_RowPtr, host_ColsInd
                                        , host_perm, host_perm, host_map, buffer_cpu ));
    }

    
    // send data to the device
    checkCudaErrors( cudaMemcpyAsync( device_RowPtr, host_RowPtr, sizeof(int)*(rowsA+1), cudaMemcpyHostToDevice, stream) );
    checkCudaErrors( cudaMemcpyAsync( device_ColsInd, host_ColsInd, sizeof(int)*nnz, cudaMemcpyHostToDevice, stream ) );
    if(d_typePermutation.getValue().getSelectedId())
    {
        for(int i=0;i<nnz;i++) host_values_permuted[i] = host_values[ host_map[i] ];
        checkCudaErrors( cudaMemcpyAsync( device_values, host_values_permuted, sizeof(double)*nnz, cudaMemcpyHostToDevice, stream ) );
    }
    else
    {
        checkCudaErrors( cudaMemcpyAsync( device_values, host_values, sizeof(double)*nnz, cudaMemcpyHostToDevice, stream ) );
    }

    cudaDeviceSynchronize();
    
    // factorize on device
    if(notSameShape)
    {
        if(device_info) cusolverSpDestroyCsrcholInfo(device_info);
        checksolver(cusolverSpCreateCsrcholInfo(&device_info));
        {
            sofa::helper::ScopedAdvancedTimer symbolicTimer("Symbolic factorization");
            checksolver( cusolverSpXcsrcholAnalysis( handle, rowsA, nnz, descr, device_RowPtr, device_ColsInd, device_info ) ); // symbolic decomposition
        }

        checksolver( cusolverSpDcsrcholBufferInfo( handle, rowsA, nnz, descr, device_values, device_RowPtr, device_ColsInd,
                                            device_info, &size_internal, &size_work ) ); //set workspace
    
     
        if(buffer_gpu) cudaFree(buffer_gpu);
        checkCudaErrors(cudaMalloc(&buffer_gpu, sizeof(char)*size_work));
    }
    
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

    // to-do : add the choice for the permutations

    //to do : apply permutation 

}

bool compareMatrixShape(const int s_M,const int * M_colind,const int * M_rowptr,const int s_P,const int * P_colind,const int * P_rowptr) {
    if (s_M != s_P) return true;
    if (M_rowptr[s_M] != P_rowptr[s_M] ) return true; 
    for (int i=0;i<s_P;i++) {
        if (M_rowptr[i]!=P_rowptr[i]) return true; 
    }
    for (int i=0;i<M_rowptr[s_M];i++) {
        if (M_colind[i]!=P_colind[i]) return true;
    }
    return false;
}



}// namespace sofa::component::linearsolver::direct
