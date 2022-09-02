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

#include <cusolverSp.h>
#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>
#include <map>
#include <string>
#include <sofa/helper/logging/Messaging.h>

#define checkCudaErrors(err) __checkCudaErrors(err, __FILE__, __LINE__)
#define checksolver(status ) __checksolver(status, __FILE__, __LINE__)


inline void __checkCudaErrors(cudaError err, const char *file, const int line)
{
    if (cudaSuccess != err)
    {
        fprintf(stderr, "%s(%i) : CUDA Runtime API error %d: %s.\n", file, line,
                (int)err, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

inline void __checksolver( cusolverStatus_t status, const char *file, const int line)
{
    if(status != 0)
    {
        msg_error("SofaCUDALinearSolver") << "Cuda Failure in " << file << " at line "<< line;

        static const std::map<cusolverStatus_t, std::string> statusMap {
            { CUSOLVER_STATUS_SUCCESS, "success" },
            { CUSOLVER_STATUS_NOT_INITIALIZED, "not initialized" },
            { CUSOLVER_STATUS_ALLOC_FAILED, "alloc failed" },
            { CUSOLVER_STATUS_INVALID_VALUE, "invalid value" },
            { CUSOLVER_STATUS_ARCH_MISMATCH, "arch mismatch" },
            { CUSOLVER_STATUS_MAPPING_ERROR, "mapping error" },
            { CUSOLVER_STATUS_EXECUTION_FAILED, "execution failed" },
            { CUSOLVER_STATUS_INTERNAL_ERROR, "internal error" },
            { CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED, "matrix type not supported" },
            { CUSOLVER_STATUS_NOT_SUPPORTED, "not supported" },
            { CUSOLVER_STATUS_ZERO_PIVOT, "zero pivot" },
            { CUSOLVER_STATUS_INVALID_LICENSE, "invalid license" },
            { CUSOLVER_STATUS_IRS_PARAMS_NOT_INITIALIZED, "IRS params not initialized" },
            { CUSOLVER_STATUS_IRS_PARAMS_INVALID, "IRS params invalid" },
            { CUSOLVER_STATUS_IRS_PARAMS_INVALID_PREC, "IRS params invalid prec" },
            { CUSOLVER_STATUS_IRS_PARAMS_INVALID_REFINE, "IRS params invalid refine" },
            { CUSOLVER_STATUS_IRS_PARAMS_INVALID_MAXITER, "IRS params invalid maxiter" },
            { CUSOLVER_STATUS_IRS_INTERNAL_ERROR, "IRS internal error" },
            { CUSOLVER_STATUS_IRS_NOT_SUPPORTED, "IRS not supported" },
            { CUSOLVER_STATUS_IRS_OUT_OF_RANGE, "IRS out of range" },
            { CUSOLVER_STATUS_IRS_NRHS_NOT_SUPPORTED_FOR_REFINE_GMRES, "IRS NRHS not supported for refine GMRES" },
            { CUSOLVER_STATUS_IRS_INFOS_NOT_INITIALIZED, "IRS infos not initialized" },
            { CUSOLVER_STATUS_IRS_INFOS_NOT_DESTROYED, "IRS infos not destroyed" },
            { CUSOLVER_STATUS_IRS_MATRIX_SINGULAR, "IRS matrix singular" },
            { CUSOLVER_STATUS_INVALID_WORKSPACE, "invalid workspace" },
        };

        msg_error("SofaCUDALinearSolver") << "Status: " << status << " - " << statusMap.at(status);
        exit(EXIT_FAILURE);
    }
}

namespace SofaCUDALinearSolver
{
    inline bool cudaInit()
    {
        int device_count;
        checkCudaErrors(cudaGetDeviceCount(&device_count));

        if (device_count == 0)
        {
            msg_error("SofaCUDALinearSolver") << "No device supporting CUDA";
            return false;
        }

        return true;
    }
}
