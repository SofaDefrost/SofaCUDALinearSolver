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

#include <SofaCUDALinearSolver/CUDASparseCholeskySolverCUDSS.h>
#include <sofa/component/linearsolver/iterative/MatrixLinearSolver.inl>
#include <sofa/helper/ScopedAdvancedTimer.h>
#include <SofaCUDALinearSolver/utils.h>
#include <cusparse.h>
#include <cstring>
#include <numeric>

namespace sofa::component::linearsolver::direct
{

// Helper macro for cuDSS error checking
#define checkCuDSS(status) __checkCuDSS(status, __FILE__, __LINE__)

inline void __checkCuDSS(cudssStatus_t status, const char* file, int line)
{
    if (status != CUDSS_STATUS_SUCCESS)
    {
        const char* statusName = "UNKNOWN";
        switch (status)
        {
            case CUDSS_STATUS_SUCCESS: statusName = "SUCCESS"; break;
            case CUDSS_STATUS_NOT_INITIALIZED: statusName = "NOT_INITIALIZED"; break;
            case CUDSS_STATUS_ALLOC_FAILED: statusName = "ALLOC_FAILED"; break;
            case CUDSS_STATUS_INVALID_VALUE: statusName = "INVALID_VALUE"; break;
            case CUDSS_STATUS_NOT_SUPPORTED: statusName = "NOT_SUPPORTED"; break;
            case CUDSS_STATUS_EXECUTION_FAILED: statusName = "EXECUTION_FAILED"; break;
            case CUDSS_STATUS_INTERNAL_ERROR: statusName = "INTERNAL_ERROR"; break;
            default: break;
        }
        msg_error("SofaCUDALinearSolver") << "cuDSS error at " << file << ":" << line
                                          << " - Status: " << statusName;
        exit(EXIT_FAILURE);
    }
}

template<class TMatrix, class TVector>
CUDASparseCholeskySolverCUDSS<TMatrix,TVector>::CUDASparseCholeskySolverCUDSS()
    : Inherit1()
    , d_typePermutation(initData(&d_typePermutation, "permutation", "Type of fill-in reducing permutation (GPU: DEFAULT/AMD, CPU: None/RCM/AMD/METIS)"))
    , d_hardware(initData(&d_hardware, "hardware", "On which hardware to solve the linear system: CPU or GPU"))
{
    sofa::helper::OptionsGroup typePermutationOptions{{"None", "RCM", "AMD", "METIS"}};
    typePermutationOptions.setSelectedItem(0); // default None
    d_typePermutation.setValue(typePermutationOptions);

    sofa::helper::OptionsGroup hardwareOptions{{"CPU", "GPU"}};
    hardwareOptions.setSelectedItem(1); // default GPU
    d_hardware.setValue(hardwareOptions);

    // Create CUDA stream
    cudaStreamCreate(&stream);

    // Initialize pointers to nullptr
    host_RowPtr = nullptr;
    host_ColsInd = nullptr;
    host_values = nullptr;

    device_RowPtr = nullptr;
    device_ColsInd = nullptr;
    device_values = nullptr;
    device_x = nullptr;
    device_b = nullptr;

    // cuDSS state
    cudssHandle = nullptr;
    cudssConfig = nullptr;
    cudssData = nullptr;
    cudssMatrixA = nullptr;
    cudssMatrixX = nullptr;
    cudssMatrixB = nullptr;
    cudssInitialized = false;

    // cusolverSp state (CPU path)
    cusolverHandle = nullptr;
    cusparseDescr = nullptr;
    host_info = nullptr;
    buffer_cpu = nullptr;
    buffer_perm = nullptr;
    size_internal_cpu = 0;
    size_work_cpu = 0;
    size_perm = 0;
    reorder = 0;

    // Common state
    notSameShape = true;
    nnz = 0;
    rows = 0;
    cols = 0;
    previous_n = 0;
    previous_nnz = 0;

    previous_ColsInd.clear();
    previous_RowPtr.clear();
}

template<class TMatrix, class TVector>
CUDASparseCholeskySolverCUDSS<TMatrix,TVector>::~CUDASparseCholeskySolverCUDSS()
{
    cleanupCuDSS();
    cleanupCuSolverHost();

    if (device_x) cudaFree(device_x);
    if (device_b) cudaFree(device_b);
    if (device_RowPtr) cudaFree(device_RowPtr);
    if (device_ColsInd) cudaFree(device_ColsInd);
    if (device_values) cudaFree(device_values);

    cudaStreamDestroy(stream);
}

// ============================================================================
// cuDSS GPU Path
// ============================================================================

template<class TMatrix, class TVector>
void CUDASparseCholeskySolverCUDSS<TMatrix,TVector>::initCuDSS()
{
    if (cudssInitialized) return;

    checkCuDSS(cudssCreate(&cudssHandle));
    checkCuDSS(cudssSetStream(cudssHandle, stream));
    checkCuDSS(cudssConfigCreate(&cudssConfig));
    checkCuDSS(cudssDataCreate(cudssHandle, &cudssData));

    // Configure reordering algorithm based on user selection
    // cuDSS options: CUDSS_ALG_DEFAULT (METIS-based), CUDSS_ALG_3 (AMD)
    int permOption = d_typePermutation.getValue().getSelectedId();
    cudssAlgType_t reorderAlg = CUDSS_ALG_DEFAULT;

    if (permOption == 2) // AMD
    {
        reorderAlg = CUDSS_ALG_3;
    }
    // For None, RCM, METIS -> use DEFAULT (METIS-based nested dissection)

    checkCuDSS(cudssConfigSet(cudssConfig, CUDSS_CONFIG_REORDERING_ALG,
                              &reorderAlg, sizeof(reorderAlg)));

    cudssInitialized = true;
}

template<class TMatrix, class TVector>
void CUDASparseCholeskySolverCUDSS<TMatrix,TVector>::cleanupCuDSS()
{
    if (!cudssInitialized) return;

    if (cudssMatrixA) { cudssMatrixDestroy(cudssMatrixA); cudssMatrixA = nullptr; }
    if (cudssMatrixX) { cudssMatrixDestroy(cudssMatrixX); cudssMatrixX = nullptr; }
    if (cudssMatrixB) { cudssMatrixDestroy(cudssMatrixB); cudssMatrixB = nullptr; }
    if (cudssData) { cudssDataDestroy(cudssHandle, cudssData); cudssData = nullptr; }
    if (cudssConfig) { cudssConfigDestroy(cudssConfig); cudssConfig = nullptr; }
    if (cudssHandle) { cudssDestroy(cudssHandle); cudssHandle = nullptr; }

    cudssInitialized = false;
}

template<class TMatrix, class TVector>
void CUDASparseCholeskySolverCUDSS<TMatrix,TVector>::invertGPU()
{
    initCuDSS();

    // Allocate/reallocate device memory if needed
    if (previous_n < rows)
    {
        if (device_RowPtr) cudaFree(device_RowPtr);
        if (device_x) cudaFree(device_x);
        if (device_b) cudaFree(device_b);

        checkCudaErrors(cudaMalloc(&device_RowPtr, sizeof(int) * (rows + 1)));
        checkCudaErrors(cudaMalloc(&device_x, sizeof(Real) * cols));
        checkCudaErrors(cudaMalloc(&device_b, sizeof(Real) * cols));
    }

    if (previous_nnz < nnz)
    {
        if (device_ColsInd) cudaFree(device_ColsInd);
        if (device_values) cudaFree(device_values);

        checkCudaErrors(cudaMalloc(&device_ColsInd, sizeof(int) * nnz));
        checkCudaErrors(cudaMalloc(&device_values, sizeof(Real) * nnz));
    }

    // Copy matrix structure to device (only if shape changed)
    if (notSameShape)
    {
        checkCudaErrors(cudaMemcpyAsync(device_RowPtr, host_RowPtr,
                        sizeof(int) * (rows + 1), cudaMemcpyHostToDevice, stream));
        checkCudaErrors(cudaMemcpyAsync(device_ColsInd, host_ColsInd,
                        sizeof(int) * nnz, cudaMemcpyHostToDevice, stream));
    }

    // Always copy values (they change each time step)
    checkCudaErrors(cudaMemcpyAsync(device_values, host_values,
                    sizeof(Real) * nnz, cudaMemcpyHostToDevice, stream));

    // Recreate matrix wrappers if shape changed
    if (notSameShape)
    {
        // Destroy old matrix wrappers
        if (cudssMatrixA) { cudssMatrixDestroy(cudssMatrixA); cudssMatrixA = nullptr; }
        if (cudssMatrixX) { cudssMatrixDestroy(cudssMatrixX); cudssMatrixX = nullptr; }
        if (cudssMatrixB) { cudssMatrixDestroy(cudssMatrixB); cudssMatrixB = nullptr; }

        // Determine data type
        cudaDataType_t valueType = std::is_same_v<Real, double> ? CUDA_R_64F : CUDA_R_32F;

        // Create sparse matrix wrapper (SPD = Symmetric Positive Definite -> Cholesky)
        checkCuDSS(cudssMatrixCreateCsr(
            &cudssMatrixA,
            rows, cols, nnz,
            device_RowPtr,
            nullptr,  // rowEnd (optional)
            device_ColsInd,
            device_values,
            CUDA_R_32I,         // index type
            valueType,          // value type
            CUDSS_MTYPE_SPD,    // matrix type: symmetric positive definite
            CUDSS_MVIEW_FULL,   // full matrix view (we have full symmetric matrix)
            CUDSS_BASE_ZERO     // zero-based indexing
        ));

        // Create dense vector wrappers
        checkCuDSS(cudssMatrixCreateDn(&cudssMatrixX, rows, 1, rows, device_x,
                                       valueType, CUDSS_LAYOUT_COL_MAJOR));
        checkCuDSS(cudssMatrixCreateDn(&cudssMatrixB, rows, 1, rows, device_b,
                                       valueType, CUDSS_LAYOUT_COL_MAJOR));

        // Analysis phase (reordering + symbolic factorization)
        {
            sofa::helper::ScopedAdvancedTimer analysisTimer("cuDSS Analysis");
            checkCuDSS(cudssExecute(cudssHandle, CUDSS_PHASE_ANALYSIS, cudssConfig,
                                    cudssData, cudssMatrixA, cudssMatrixX, cudssMatrixB));
        }

        // Store shape for comparison
        previous_nnz = nnz;
        previous_RowPtr.resize(rows + 1);
        previous_ColsInd.resize(nnz);
        std::memcpy(previous_ColsInd.data(), host_ColsInd, sizeof(int) * nnz);
        std::memcpy(previous_RowPtr.data(), host_RowPtr, sizeof(int) * (rows + 1));
    }
    else
    {
        // Update values pointer in existing matrix wrapper
        checkCuDSS(cudssMatrixSetCsrPointers(cudssMatrixA, device_RowPtr, nullptr,
                                             device_ColsInd, device_values));
    }

    // Numeric factorization
    {
        sofa::helper::ScopedAdvancedTimer factorTimer("cuDSS Factorization");
        checkCuDSS(cudssExecute(cudssHandle, CUDSS_PHASE_FACTORIZATION, cudssConfig,
                                cudssData, cudssMatrixA, cudssMatrixX, cudssMatrixB));
        cudaStreamSynchronize(stream);
    }
}

template<class TMatrix, class TVector>
void CUDASparseCholeskySolverCUDSS<TMatrix,TVector>::solveGPU(int n, Real* b_host, Real* x_host)
{
    // Copy RHS to device
    {
        sofa::helper::ScopedAdvancedTimer copyTimer("copyRHSToDevice");
        checkCudaErrors(cudaMemcpyAsync(device_b, b_host, sizeof(Real) * n,
                        cudaMemcpyHostToDevice, stream));
    }

    // Solve
    {
        sofa::helper::ScopedAdvancedTimer solveTimer("cuDSS Solve");
        checkCuDSS(cudssExecute(cudssHandle, CUDSS_PHASE_SOLVE, cudssConfig,
                                cudssData, cudssMatrixA, cudssMatrixX, cudssMatrixB));
    }

    // Copy solution to host
    {
        sofa::helper::ScopedAdvancedTimer copyTimer("copySolutionToHost");
        checkCudaErrors(cudaMemcpyAsync(x_host, device_x, sizeof(Real) * n,
                        cudaMemcpyDeviceToHost, stream));
        cudaStreamSynchronize(stream);
    }
}

// ============================================================================
// cusolverSp CPU Path
// ============================================================================

template<class TMatrix, class TVector>
void CUDASparseCholeskySolverCUDSS<TMatrix,TVector>::initCuSolverHost()
{
    if (cusolverHandle) return;

    cusolverSpCreate(&cusolverHandle);
    cusparseCreateMatDescr(&cusparseDescr);
    cusparseSetMatType(cusparseDescr, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(cusparseDescr, CUSPARSE_INDEX_BASE_ZERO);
}

template<class TMatrix, class TVector>
void CUDASparseCholeskySolverCUDSS<TMatrix,TVector>::cleanupCuSolverHost()
{
    if (host_info) { cusolverSpDestroyCsrcholInfoHost(host_info); host_info = nullptr; }
    if (cusparseDescr) { cusparseDestroyMatDescr(cusparseDescr); cusparseDescr = nullptr; }
    if (cusolverHandle) { cusolverSpDestroy(cusolverHandle); cusolverHandle = nullptr; }
    if (buffer_cpu) { cudaFreeHost(buffer_cpu); buffer_cpu = nullptr; }
    if (buffer_perm) { free(buffer_perm); buffer_perm = nullptr; }
}

template<class TMatrix, class TVector>
void CUDASparseCholeskySolverCUDSS<TMatrix,TVector>::invertCPU()
{
    initCuSolverHost();

    reorder = d_typePermutation.getValue().getSelectedId();

    // Resize permutation buffers if needed
    if (previous_n < rows)
    {
        host_rowPermuted.resize(rows + 1);
        host_bPermuted.resize(rows);
        host_xPermuted.resize(rows);
    }

    if (previous_nnz < nnz)
    {
        host_colPermuted.resize(nnz);
        host_valuePermuted.resize(nnz);
    }

    // Compute fill-reducing permutation (only if shape changed)
    if (reorder != 0 && notSameShape)
    {
        sofa::helper::ScopedAdvancedTimer permTimer("Permutations");

        host_perm.resize(rows);

        switch (reorder)
        {
            case 1: // RCM
                checksolver(cusolverSpXcsrsymrcmHost(cusolverHandle, rows, nnz, cusparseDescr,
                            host_RowPtr, host_ColsInd, host_perm.data()));
                break;
            case 2: // AMD
                checksolver(cusolverSpXcsrsymamdHost(cusolverHandle, rows, nnz, cusparseDescr,
                            host_RowPtr, host_ColsInd, host_perm.data()));
                break;
            case 3: // METIS
                checksolver(cusolverSpXcsrmetisndHost(cusolverHandle, rows, nnz, cusparseDescr,
                            host_RowPtr, host_ColsInd, nullptr, host_perm.data()));
                break;
            default:
                break;
        }

        checksolver(cusolverSpXcsrperm_bufferSizeHost(cusolverHandle, rows, cols, nnz, cusparseDescr,
                    host_RowPtr, host_ColsInd, host_perm.data(), host_perm.data(), &size_perm));

        if (buffer_perm) free(buffer_perm);
        buffer_perm = malloc(size_perm);

        host_map.resize(nnz);
        std::iota(host_map.begin(), host_map.end(), 0);

        // Apply permutation to structure
        std::memcpy(host_rowPermuted.data(), host_RowPtr, sizeof(int) * (rows + 1));
        std::memcpy(host_colPermuted.data(), host_ColsInd, sizeof(int) * nnz);

        checksolver(cusolverSpXcsrpermHost(cusolverHandle, rows, cols, nnz, cusparseDescr,
                    host_rowPermuted.data(), host_colPermuted.data(), host_perm.data(),
                    host_perm.data(), host_map.data(), buffer_perm));
    }

    // Apply permutation to values
    if (reorder != 0)
    {
        sofa::helper::ScopedAdvancedTimer reorderTimer("ReorderValues");
        for (int i = 0; i < nnz; i++)
        {
            host_valuePermuted[i] = host_values[host_map[i]];
        }
    }

    const int* hRow = (reorder != 0) ? host_rowPermuted.data() : host_RowPtr;
    const int* hCol = (reorder != 0) ? host_colPermuted.data() : host_ColsInd;
    const Real* hValues = (reorder != 0) ? host_valuePermuted.data() : host_values;

    // Symbolic factorization (only if shape changed)
    if (notSameShape)
    {
        if (host_info) cusolverSpDestroyCsrcholInfoHost(host_info);
        checksolver(cusolverSpCreateCsrcholInfoHost(&host_info));

        {
            sofa::helper::ScopedAdvancedTimer symbolicTimer("Symbolic factorization");
            checksolver(cusolverSpXcsrcholAnalysisHost(cusolverHandle, rows, nnz,
                        cusparseDescr, hRow, hCol, host_info));
        }

        // Get buffer size
        if constexpr (std::is_same_v<Real, double>)
        {
            checksolver(cusolverSpDcsrcholBufferInfoHost(cusolverHandle, rows, nnz, cusparseDescr,
                        hValues, hRow, hCol, host_info, &size_internal_cpu, &size_work_cpu));
        }
        else
        {
            checksolver(cusolverSpScsrcholBufferInfoHost(cusolverHandle, rows, nnz, cusparseDescr,
                        hValues, hRow, hCol, host_info, &size_internal_cpu, &size_work_cpu));
        }

        if (buffer_cpu) cudaFreeHost(buffer_cpu);
        checkCudaErrors(cudaMallocHost(&buffer_cpu, size_work_cpu));

        // Store shape for comparison
        previous_nnz = nnz;
        previous_RowPtr.resize(rows + 1);
        previous_ColsInd.resize(nnz);
        std::memcpy(previous_ColsInd.data(), host_ColsInd, sizeof(int) * nnz);
        std::memcpy(previous_RowPtr.data(), host_RowPtr, sizeof(int) * (rows + 1));
    }

    // Numeric factorization
    {
        sofa::helper::ScopedAdvancedTimer numericTimer("Numeric factorization");
        if constexpr (std::is_same_v<Real, double>)
        {
            checksolver(cusolverSpDcsrcholFactorHost(cusolverHandle, rows, nnz, cusparseDescr,
                        hValues, hRow, hCol, host_info, buffer_cpu));
        }
        else
        {
            checksolver(cusolverSpScsrcholFactorHost(cusolverHandle, rows, nnz, cusparseDescr,
                        hValues, hRow, hCol, host_info, buffer_cpu));
        }
    }
}

template<class TMatrix, class TVector>
void CUDASparseCholeskySolverCUDSS<TMatrix,TVector>::solveCPU(int n, Real* b_host, Real* x_host)
{
    Real* solveB = b_host;
    Real* solveX = x_host;

    // Apply permutation to RHS
    if (reorder != 0)
    {
        sofa::helper::ScopedAdvancedTimer reorderTimer("reorderRHS");
        for (int i = 0; i < n; i++)
        {
            host_bPermuted[i] = b_host[host_perm[i]];
        }
        solveB = host_bPermuted.data();
        solveX = host_xPermuted.data();
    }

    // Solve
    {
        sofa::helper::ScopedAdvancedTimer solveTimer("Solve");
        if constexpr (std::is_same_v<Real, double>)
        {
            checksolver(cusolverSpDcsrcholSolveHost(cusolverHandle, n, solveB, solveX,
                        host_info, buffer_cpu));
        }
        else
        {
            checksolver(cusolverSpScsrcholSolveHost(cusolverHandle, n, solveB, solveX,
                        host_info, buffer_cpu));
        }
    }

    // Apply inverse permutation to solution
    if (reorder != 0)
    {
        sofa::helper::ScopedAdvancedTimer reorderTimer("reorderSolution");
        for (int i = 0; i < n; i++)
        {
            x_host[host_perm[i]] = host_xPermuted[i];
        }
    }
}

// ============================================================================
// Main interface methods
// ============================================================================

template<class TMatrix, class TVector>
void CUDASparseCholeskySolverCUDSS<TMatrix,TVector>::invert(Matrix& M)
{
    sofa::helper::ScopedAdvancedTimer invertTimer("invert");

    // Copy and compress matrix data
    {
        sofa::helper::ScopedAdvancedTimer copyTimer("copyMatrixData");
        m_filteredMatrix.copyNonZeros(M);
        m_filteredMatrix.compress();
    }

    rows = m_filteredMatrix.rowSize();
    cols = m_filteredMatrix.colSize();
    nnz = m_filteredMatrix.getColsValue().size();

    host_RowPtr = (int*)m_filteredMatrix.getRowBegin().data();
    host_ColsInd = (int*)m_filteredMatrix.getColsIndex().data();
    host_values = (Real*)m_filteredMatrix.getColsValue().data();

    // Check if matrix shape changed
    {
        sofa::helper::ScopedAdvancedTimer compareTimer("compareMatrixShape");
        notSameShape = compareMatrixShape(rows, host_ColsInd, host_RowPtr,
                       previous_RowPtr.size() - 1, previous_ColsInd.data(), previous_RowPtr.data());
    }

    // Dispatch to appropriate implementation
    if (d_hardware.getValue().getSelectedId() == 0)
    {
        invertCPU();
    }
    else
    {
        invertGPU();
    }

    previous_n = rows;
}

template<class TMatrix, class TVector>
void CUDASparseCholeskySolverCUDSS<TMatrix,TVector>::solve(Matrix& M, Vector& x, Vector& b)
{
    sofa::helper::ScopedAdvancedTimer solveTimer("solve");

    int n = M.colSize();

    if (d_hardware.getValue().getSelectedId() == 0)
    {
        solveCPU(n, b.ptr(), x.ptr());
    }
    else
    {
        solveGPU(n, b.ptr(), x.ptr());
    }
}

inline bool compareMatrixShape(const int s_M, const int* M_colind, const int* M_rowptr,
                                const int s_P, const int* P_colind, const int* P_rowptr)
{
    if (s_M != s_P) return true;
    if (s_P <= 0) return true;
    if (M_rowptr[s_M] != P_rowptr[s_M]) return true;
    if (std::memcmp(M_rowptr, P_rowptr, sizeof(int) * s_P) != 0) return true;
    if (std::memcmp(M_colind, P_colind, sizeof(int) * M_rowptr[s_M]) != 0) return true;
    return false;
}

} // namespace sofa::component::linearsolver::direct
