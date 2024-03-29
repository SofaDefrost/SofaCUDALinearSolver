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
    , d_hardware(initData(&d_hardware, "hardware", "On which hardware to solve the linear system: CPU or GPU"))
{
    sofa::helper::OptionsGroup typePermutationOptions{"None","RCM" ,"AMD", "METIS"};
    typePermutationOptions.setSelectedItem(0); // default None
    d_typePermutation.setValue(typePermutationOptions);

    sofa::helper::OptionsGroup hardwareOptions{"CPU", "GPU"};
    hardwareOptions.setSelectedItem(1);
    d_hardware.setValue(hardwareOptions);

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
    buffer_gpu = nullptr;
    device_info = nullptr;
    host_info = nullptr;

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
    previous_RowPtr.clear();
    previous_ColsInd.clear();

    checkCudaErrors(cudaFree(device_x));
    checkCudaErrors(cudaFree(device_b));
    checkCudaErrors(cudaFree(device_RowPtr));
    checkCudaErrors(cudaFree(device_ColsInd));
    checkCudaErrors(cudaFree(device_values));

    cusolverSpDestroy(handle);
    cusolverSpDestroyCsrcholInfo(device_info);
    cusolverSpDestroyCsrcholInfoHost(host_info);
    cusparseDestroyMatDescr(descr);
    cudaStreamDestroy(stream);
}

template <class TMatrix, class TVector>
void CUDASparseCholeskySolver<TMatrix, TVector>::setWorkspace()
{
    if (d_hardware.getValue().getSelectedId() == 0)
    {
        const int* hRow = reorder != 0 ? host_rowPermuted.data() : host_RowPtr;
        const int* hCol = reorder != 0 ? host_colPermuted.data() : host_ColsInd;
        const Real* hValues = reorder != 0 ? host_valuePermuted.data() : host_values;

        if constexpr (std::is_same_v<Real, double>)
        {
            checksolver(cusolverSpDcsrcholBufferInfoHost( handle, rows, nnz, descr, hValues, hRow, hCol,
                host_info, &size_internal, &size_work ));
        }
        else
        {
            checksolver(cusolverSpScsrcholBufferInfoHost( handle, rows, nnz, descr, device_values, hRow, hCol,
                host_info, &size_internal, &size_work ));
        }
    }
    else
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
}

template <class TMatrix, class TVector>
void CUDASparseCholeskySolver<TMatrix, TVector>::numericFactorization()
{
    if (d_hardware.getValue().getSelectedId() == 0)
    {
        const int* hRow = reorder != 0 ? host_rowPermuted.data() : host_RowPtr;
        const int* hCol = reorder != 0 ? host_colPermuted.data() : host_ColsInd;
        const Real* hValues = reorder != 0 ? host_valuePermuted.data() : host_values;

        if constexpr (std::is_same_v<Real, double>)
        {
            checksolver(cusolverSpDcsrcholFactorHost( handle, rows, nnz, descr, hValues, hRow, hCol,
                host_info, buffer_gpu ));
        }
        else
        {
            checksolver(cusolverSpScsrcholFactorHost( handle, rows, nnz, descr, hValues, hRow, hCol,
                host_info, buffer_gpu ));
        }
    }
    else
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
}

template <class TMatrix, class TVector>
void CUDASparseCholeskySolver<TMatrix, TVector>::createCholeskyInfo()
{
    cusolverSpDestroyCsrcholInfo(device_info);
    cusolverSpDestroyCsrcholInfoHost(host_info);

    if (d_hardware.getValue().getSelectedId() == 0)
    {
        checksolver(cusolverSpCreateCsrcholInfoHost(&host_info));
    }
    else
    {
        checksolver(cusolverSpCreateCsrcholInfo(&device_info));
    }
}

template <class TMatrix, class TVector>
void CUDASparseCholeskySolver<TMatrix, TVector>::symbolicFactorization()
{
    if (d_hardware.getValue().getSelectedId() == 0)
    {
        const int* hRow = reorder != 0 ? host_rowPermuted.data() : host_RowPtr;
        const int* hCol = reorder != 0 ? host_colPermuted.data() : host_ColsInd;
        checksolver(cusolverSpXcsrcholAnalysisHost( handle, rows, nnz, descr, hRow, hCol, host_info ));
    }
    else
    {
        checksolver(cusolverSpXcsrcholAnalysis( handle, rows, nnz, descr, device_RowPtr, device_ColsInd, device_info ));
        cudaStreamSynchronize(stream);// for the timer
    }
}

template<class TMatrix , class TVector>
void CUDASparseCholeskySolver<TMatrix,TVector>:: invert(Matrix& M)
{
    sofa::helper::ScopedAdvancedTimer invertTimer("invert");

    {
        sofa::helper::ScopedAdvancedTimer copyTimer("copyMatrixData");
        m_filteredMatrix.copyNonZeros(M);
        m_filteredMatrix.compress();
    }

    reorder = d_typePermutation.getValue().getSelectedId() ;
    rows = m_filteredMatrix.rowSize();
    cols = m_filteredMatrix.colSize();
    nnz = m_filteredMatrix.getColsValue().size(); // number of non zero coefficients

    host_RowPtr = (int*) m_filteredMatrix.getRowBegin().data();
    host_ColsInd = (int*) m_filteredMatrix.getColsIndex().data();
    host_values = (Real*) m_filteredMatrix.getColsValue().data();

    {
        sofa::helper::ScopedAdvancedTimer compareMatrixShapeTimer("compareMatrixShape");
        notSameShape = compareMatrixShape(rows , host_ColsInd, host_RowPtr, previous_RowPtr.size()-1,  previous_ColsInd.data(), previous_RowPtr.data() );
    }

    // allocate memory
    if(previous_n < rows)
    {
        host_rowPermuted.resize(rows + 1);

        if (d_hardware.getValue().getSelectedId() != 0)
        {
            checkCudaErrors(cudaMalloc( &device_RowPtr, sizeof(int)*( rows +1) ));

            checkCudaErrors(cudaMalloc(&device_x, sizeof(Real)*cols));
            checkCudaErrors(cudaMalloc(&device_b, sizeof(Real)*cols));
        }
    }

    if(previous_nnz < nnz)
    {
        if (d_hardware.getValue().getSelectedId() != 0)
        {
            if(device_ColsInd) cudaFree(device_ColsInd);
            checkCudaErrors(cudaMalloc( &device_ColsInd, sizeof(int)*nnz ));
            if(device_values) cudaFree(device_values);
            checkCudaErrors(cudaMalloc( &device_values, sizeof(Real)*nnz ));
        }
        host_colPermuted.resize(nnz);
        host_valuePermuted.resize(nnz);
    }

    // A = PAQ
    // compute fill reducing permutation
    if( reorder != 0 && notSameShape)
    {
        sofa::helper::ScopedAdvancedTimer permutationsTimer("Permutations");

        host_perm.resize(rows);

        switch( reorder )
        {
            default:
            case 0:// None, this case should not be visited 
                break;

            case 1://RCM, Symmetric Reverse Cuthill-McKee permutation
                checksolver( cusolverSpXcsrsymrcmHost(handle, rows, nnz, descr, host_RowPtr, host_ColsInd, host_perm.data()) );
                break;

            case 2://AMD, Symmetric Approximate Minimum Degree Algorithm based on Quotient Graph
                checksolver( cusolverSpXcsrsymamdHost(handle, rows, nnz, descr, host_RowPtr, host_ColsInd, host_perm.data()) );
                break;

            case 3://METIS, nested dissection
                checksolver( cusolverSpXcsrmetisndHost(handle, rows, nnz, descr, host_RowPtr, host_ColsInd, nullptr, host_perm.data()) );
                break;
        }
        checksolver( cusolverSpXcsrperm_bufferSizeHost(handle, rows, cols, nnz, descr, host_RowPtr, host_ColsInd
                                                    , host_perm.data(), host_perm.data(), &size_perm) );

        if(buffer_cpu) free(buffer_cpu);
        buffer_cpu = (void*)malloc(sizeof(char)*size_perm) ;

        host_map.resize(nnz);
        std::iota(host_map.begin(), host_map.end(), 0);

        //apply the permutation
        for(int i=0;i<rows+1;i++) host_rowPermuted[i] = host_RowPtr[i];
        for(int i=0;i<nnz;i++) host_colPermuted[i] = host_ColsInd[i];
        checksolver( cusolverSpXcsrpermHost( handle, rows, cols, nnz, descr, host_rowPermuted.data(), host_colPermuted.data()
                                        , host_perm.data(), host_perm.data(), host_map.data(), buffer_cpu ));
    }

    // send data to the device
    if (notSameShape && d_hardware.getValue().getSelectedId() != 0)
    {
        const int* host_rowPtrToCopy = reorder != 0 ? host_rowPermuted.data() : host_RowPtr;
        const int* host_colPtrToCopy = reorder != 0 ? host_colPermuted.data() : host_ColsInd;

        checkCudaErrors( cudaMemcpyAsync( device_RowPtr, host_rowPtrToCopy, sizeof(int)*(rows+1), cudaMemcpyHostToDevice, stream) );
        checkCudaErrors( cudaMemcpyAsync( device_ColsInd, host_colPtrToCopy, sizeof(int)*nnz, cudaMemcpyHostToDevice, stream ) );
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

    if( reorder != 0)
    {
        sofa::helper::ScopedAdvancedTimer reorderValuesTimer("ReorderValues");
        for(int i=0;i<nnz;i++)
        {
            host_valuePermuted[i] = host_values[ host_map[i] ];
        }
    }

    if (d_hardware.getValue().getSelectedId() != 0)
    {
        const Real* hostValuesToCopy = reorder != 0 ? host_valuePermuted.data() : host_values;
        checkCudaErrors( cudaMemcpyAsync( device_values, hostValuesToCopy, sizeof(Real)*nnz, cudaMemcpyHostToDevice, stream ) );
    }
    
    // factorize on device LL^t = PAP^t 
    if(notSameShape)
    {
        createCholeskyInfo();
        {
            sofa::helper::ScopedAdvancedTimer symbolicTimer("Symbolic factorization");
            symbolicFactorization();
        }

        setWorkspace();

        if(buffer_gpu) cudaFree(buffer_gpu);
        if (d_hardware.getValue().getSelectedId() == 0)
        {
            checkCudaErrors(cudaMallocHost(&buffer_gpu, sizeof(char)*size_work));
        }
        else
        {
            checkCudaErrors(cudaMalloc(&buffer_gpu, sizeof(char)*size_work));
        }
    }
    
    {
        sofa::helper::ScopedAdvancedTimer numericTimer("Numeric factorization");
        numericFactorization();
        cudaStreamSynchronize(stream);// for the timer
    }
}

template <class TMatrix, class TVector>
void CUDASparseCholeskySolver<TMatrix, TVector>::solve_impl(int n, Real* b, Real* x)
{
    if (d_hardware.getValue().getSelectedId() == 0)
    {
        Real* host_b = (reorder != 0) ? host_bPermuted.data() : b;
        Real* host_x = (reorder != 0) ? host_xPermuted.data() : x;

        if constexpr (std::is_same_v<Real, double>)
        {
            checksolver(cusolverSpDcsrcholSolveHost( handle, n, host_b, host_x, host_info, buffer_gpu ));
        }
        else
        {
            checksolver(cusolverSpScsrcholSolveHost( handle, n, host_b, host_x, host_info, buffer_gpu ));
        }
    }
    else
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
}

template<class TMatrix , class TVector>
void CUDASparseCholeskySolver<TMatrix,TVector>::solve(Matrix& M, Vector& x, Vector& b)
{
    sofa::helper::ScopedAdvancedTimer solveTimer("solve");
    int n = M.colSize()  ; // avoidable, used to prevent compilation warning

    if( previous_n < n && reorder !=0 )
    {
        host_bPermuted.resize(n);
        host_xPermuted.resize(n);
    }

    if( reorder != 0 )
    {
        sofa::helper::ScopedAdvancedTimer reorderRHSTimer("reorderRHS");
        for(int i = 0; i < n; ++i)
        {
            host_bPermuted[i] = b[ host_perm[i] ];
        }
    }
    Real* host_b = (reorder != 0) ? host_bPermuted.data() : b.ptr();
    Real* host_x = (reorder != 0) ? host_xPermuted.data() : x.ptr();

    if (d_hardware.getValue().getSelectedId() != 0)
    {
        sofa::helper::ScopedAdvancedTimer copyRHSToDeviceTimer("copyRHSToDevice");
        checkCudaErrors(cudaMemcpyAsync( device_b, host_b, sizeof(Real)*n, cudaMemcpyHostToDevice,stream));
        checkCudaErrors(cudaStreamSynchronize(stream));
    }

    {
        // LL^t y = Pb
        sofa::helper::ScopedAdvancedTimer solveTimer("Solve");

        solve_impl(n, x.ptr(), b.ptr());
        checkCudaErrors(cudaStreamSynchronize(stream));
    }

    if (d_hardware.getValue().getSelectedId() != 0)
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
            x[host_perm[i]] = host_xPermuted[ i ]; // Px = y
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
