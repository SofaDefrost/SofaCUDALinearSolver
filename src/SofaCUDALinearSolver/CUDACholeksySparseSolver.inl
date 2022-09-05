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

#include <SofaCUDALinearSolver/CUDACholeksySparseSolver.h>
#include <sofa/helper/ScopedAdvancedTimer.h>
#include <SofaCUDALinearSolver/utils.h>
#include <cusparse.h>


namespace sofa::component::linearsolver::direct
{

template<class TMatrix , class TVector>
CUDASparseCholeskySolver<TMatrix,TVector>::CUDASparseCholeskySolver()
    : Inherit1()
    , d_typePermutation(initData(&d_typePermutation, "permutation", "Type of fill-in reducing permutation"))
{
    sofa::helper::OptionsGroup d_typePermutationOptions(4,"None","RCM" ,"AMD", "METIS");
    d_typePermutationOptions.setSelectedItem(0); // default None
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

    host_RowPtr_permuted = nullptr;
    host_ColsInd_permuted = nullptr;
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
    host_map = nullptr;

    singularity = 0;

    size_internal = 0;
    size_work = 0;
    size_perm = 0;

    notSameShape = true;

    nnz = 0;
    previous_n = 0 ;
    previous_nnz = 0;
    rows = 0;
    reorder = 0;
    previous_ColsInd.clear() ;
    previous_RowPtr.clear() ;

}

template<class TMatrix , class TVector>
CUDASparseCholeskySolver<TMatrix,TVector>::~CUDASparseCholeskySolver()
{ 
    if(host_RowPtr_permuted) free(host_RowPtr_permuted);
    if(host_ColsInd_permuted) free(host_ColsInd_permuted);
    if(host_values_permuted) free(host_values_permuted);

    previous_RowPtr.clear();
    previous_ColsInd.clear();

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

template <class TMatrix, class TVector>
void CUDASparseCholeskySolver<TMatrix, TVector>::setWorkspace()
{
    if constexpr (std::is_same_v<Real, double>)
    {
        checksolver(cusolverSpDcsrcholBufferInfo( handle, rows, nnz, descr, device_values, device_RowPtr, device_ColsInd,
            device_info, &size_internal, &size_work ));
    }
    else
    {
        checksolver(cusolverSpScsrcholBufferInfo( handle, rows, nnz, descr, device_values, device_RowPtr, device_ColsInd,
            device_info, &size_internal, &size_work ));
    }
}

template <class TMatrix, class TVector>
void CUDASparseCholeskySolver<TMatrix, TVector>::numericFactorization()
{
    if constexpr (std::is_same_v<Real, double>)
    {
        checksolver(cusolverSpDcsrcholFactor( handle, rows, nnz, descr, device_values, device_RowPtr, device_ColsInd,
            device_info, buffer_gpu ));
    }
    else
    {
        checksolver(cusolverSpScsrcholFactor( handle, rows, nnz, descr, device_values, device_RowPtr, device_ColsInd,
            device_info, buffer_gpu ));
    }
}

template<class TMatrix , class TVector>
void CUDASparseCholeskySolver<TMatrix,TVector>:: invert(Matrix& M)
{
    {
        sofa::helper::ScopedAdvancedTimer copyTimer("copyMatrixData");
        m_filteredMatrix.copyNonZeros(M);
        m_filteredMatrix.compress();
    }

    reorder = d_typePermutation.getValue().getSelectedId() ;
    rows = m_filteredMatrix.rowSize();
    cols = m_filteredMatrix.colSize();
    nnz = m_filteredMatrix.getColsValue().size(); // number of non zero coefficients

    // copy the matrix
    host_RowPtr = (int*) m_filteredMatrix.getRowBegin().data();
    host_ColsInd = (int*) m_filteredMatrix.getColsIndex().data();
    host_values = (Real*) m_filteredMatrix.getColsValue().data();
 
    notSameShape = compareMatrixShape(rows , host_ColsInd, host_RowPtr, previous_RowPtr.size()-1,  previous_ColsInd.data(), previous_RowPtr.data() );

    // allocate memory
    if(previous_n < rows)
    {
        checkCudaErrors(cudaMalloc( &device_RowPtr, sizeof(int)*( rows +1) ));

        if(host_RowPtr_permuted) free(host_RowPtr_permuted);
        host_RowPtr_permuted = (int*)malloc(sizeof(int)*(rows+1));

        checkCudaErrors(cudaMalloc(&device_x, sizeof(Real)*cols));
        checkCudaErrors(cudaMalloc(&device_b, sizeof(Real)*cols));
    }

    if(previous_nnz < nnz)
    {
        if(device_ColsInd) cudaFree(device_ColsInd);
        checkCudaErrors(cudaMalloc( &device_ColsInd, sizeof(int)*nnz ));
        if(device_values) cudaFree(device_values);
        checkCudaErrors(cudaMalloc( &device_values, sizeof(Real)*nnz ));

        if(host_ColsInd_permuted) free(host_ColsInd_permuted);
        host_ColsInd_permuted = (int*)malloc(sizeof(int)*nnz);
        if(host_values_permuted) free(host_values_permuted);
        host_values_permuted = (Real*)malloc(sizeof(Real)*nnz);
    }

    // A = PAQ
    // compute fill reducing permutation
    if( (reorder != 0) && notSameShape)
    {
        if(host_perm) free(host_perm);
        host_perm =(int*) malloc(sizeof( int )*rows );

        switch( reorder )
        {
            default:
            case 0:// None, this case should not be visited 
                break;

            case 1://RCM, Symmetric Reverse Cuthill-McKee permutation
                checksolver( cusolverSpXcsrsymrcmHost(handle, rows, nnz, descr, host_RowPtr, host_ColsInd, host_perm) );
                break;

            case 2://AMD, Symetric Minimum Degree Approximation
                checksolver( cusolverSpXcsrsymamdHost(handle, rows, nnz, descr, host_RowPtr, host_ColsInd, host_perm) );
                break;

            case 3://METIS, nested dissection
                checksolver( cusolverSpXcsrmetisndHost(handle, rows, nnz, descr, host_RowPtr, host_ColsInd, nullptr, host_perm) );
                break;
        }
        checksolver( cusolverSpXcsrperm_bufferSizeHost(handle, rows, cols, nnz, descr, host_RowPtr, host_ColsInd
                                                    , host_perm, host_perm, &size_perm) );

        if(buffer_cpu) free(buffer_cpu);
        buffer_cpu = (void*)malloc(sizeof(char)*size_perm) ;

        if(host_map) free(host_map);
        host_map = (int*)malloc(sizeof(int)*nnz);
        for(int i=0;i<nnz;i++) host_map[i] = i;
        
        //apply the permutation
        for(int i=0;i<rows+1;i++) host_RowPtr_permuted[i] = host_RowPtr[i];
        for(int i=0;i<nnz;i++) host_ColsInd_permuted[i] = host_ColsInd[i];
        checksolver( cusolverSpXcsrpermHost( handle, rows, cols, nnz, descr, host_RowPtr_permuted, host_ColsInd_permuted
                                        , host_perm, host_perm, host_map, buffer_cpu ));
    }

    //store the shape of the matrix
    if(notSameShape)
    {
        previous_nnz = nnz ;
        previous_RowPtr.resize( rows + 1 );
        previous_ColsInd.resize( nnz );
        for(int i=0;i<nnz;i++) previous_ColsInd[i] = host_ColsInd[i];
        for(int i=0; i<rows +1; i++) previous_RowPtr[i] = host_RowPtr[i];
    }
    
    // send data to the device
    if (notSameShape)
    {
        const int* host_rowPtrToCopy = reorder != 0 ? host_RowPtr_permuted : host_RowPtr;
        const int* host_colPtrToCopy = reorder != 0 ? host_ColsInd_permuted : host_ColsInd;

        checkCudaErrors( cudaMemcpyAsync( device_RowPtr, host_rowPtrToCopy, sizeof(int)*(rows+1), cudaMemcpyHostToDevice, stream) );
        checkCudaErrors( cudaMemcpyAsync( device_ColsInd, host_colPtrToCopy, sizeof(int)*nnz, cudaMemcpyHostToDevice, stream ) );
    }

    if( reorder != 0)
    {
        for(int i=0;i<nnz;i++)
        {
            host_values_permuted[i] = host_values[ host_map[i] ];
        }
    }

    {
        const Real* hostValuesToCopy = reorder != 0 ? host_values_permuted : host_values;
        checkCudaErrors( cudaMemcpyAsync( device_values, hostValuesToCopy, sizeof(Real)*nnz, cudaMemcpyHostToDevice, stream ) );
    }
    
    // factorize on device LL^t = PAP^t 
    if(notSameShape)
    {
        if(device_info) cusolverSpDestroyCsrcholInfo(device_info);
        checksolver(cusolverSpCreateCsrcholInfo(&device_info));
        {
            sofa::helper::ScopedAdvancedTimer symbolicTimer("Symbolic factorization");
            checksolver( cusolverSpXcsrcholAnalysis( handle, rows, nnz, descr, device_RowPtr, device_ColsInd, device_info ) ); // symbolic decomposition
            cudaStreamSynchronize(stream);// for the timer
        }

        setWorkspace();
    
     
        if(buffer_gpu) cudaFree(buffer_gpu);
        checkCudaErrors(cudaMalloc(&buffer_gpu, sizeof(char)*size_work));
    }
    
    {
        sofa::helper::ScopedAdvancedTimer numericTimer("Numeric factorization");
        numericFactorization();
        cudaStreamSynchronize(stream);// for the timer
    }

    
}

template <class TMatrix, class TVector>
void CUDASparseCholeskySolver<TMatrix, TVector>::solveOnGPU(int n)
{
    if constexpr (std::is_same_v<Real, double>)
    {
        checksolver(cusolverSpDcsrcholSolve( handle, n, device_b, device_x, device_info, buffer_gpu ));
    }
    else
    {
        checksolver(cusolverSpScsrcholSolve( handle, n, device_b, device_x, device_info, buffer_gpu ));
    }
}

template<class TMatrix , class TVector>
void CUDASparseCholeskySolver<TMatrix,TVector>::solve(Matrix& M, Vector& x, Vector& b)
{
    int n = M.colSize()  ; // avoidable, used to prevent compilation warning

    if( previous_n < n && reorder !=0 )
    {
        if(host_b_permuted) free(host_b_permuted);
        host_b_permuted = (Real*)malloc(sizeof(Real)*n);

        if(host_x_permuted) free(host_x_permuted);
        host_x_permuted = (Real*)malloc(sizeof(Real)*n);
    }

    if( reorder != 0 )
    {
        sofa::helper::ScopedAdvancedTimer reorderRHSTimer("reorderRHS");
        for(int i = 0; i < n; ++i)
        {
            host_b_permuted[i] = b[ host_perm[i] ];
        }
    }
    Real* host_b = (reorder != 0) ? host_b_permuted : b.ptr();
    Real* host_x = (reorder != 0) ? host_x_permuted : x.ptr();

    {
        sofa::helper::ScopedAdvancedTimer copyRHSToDeviceTimer("copyRHSToDevice");
        checkCudaErrors(cudaMemcpyAsync( device_b, host_b, sizeof(Real)*n, cudaMemcpyHostToDevice,stream));
        checkCudaErrors(cudaStreamSynchronize(stream));
    }

    {
        // LL^t y = Pb
        sofa::helper::ScopedAdvancedTimer solveTimer("Solve");

        solveOnGPU(n);
        checkCudaErrors(cudaStreamSynchronize(stream));
    }

    {
        sofa::helper::ScopedAdvancedTimer copySolutionToHostTimer("copySolutionToHost");

        checkCudaErrors(cudaMemcpyAsync( host_x, device_x, sizeof(Real)*n, cudaMemcpyDeviceToHost,stream));
        cudaStreamSynchronize(stream);
    }

    if( reorder != 0 )
    {
        sofa::helper::ScopedAdvancedTimer reorderSolutionTimer("reorderSolution");
        for(int i = 0; i < n; ++i)
        {
            x[host_perm[i]] = host_x_permuted[ i ]; // Px = y
        }
    }


    previous_n = n ;
}

inline bool compareMatrixShape(const int s_M,const int * M_colind,const int * M_rowptr,const int s_P,const int * P_colind,const int * P_rowptr)
{
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