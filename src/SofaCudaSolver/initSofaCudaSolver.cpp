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
#include <SofaCudaSolver/config.h>
#include <sofa/helper/system/PluginManager.h>
#include <sofa/helper/logging/Messaging.h>


namespace sofa
{

namespace component
{

extern "C" {

    SOFA_CUDASOLVER_API void initExternalModule();
    SOFA_CUDASOLVER_APIconst char* getModuleName();
    SOFA_CUDASOLVER_API const char* getModuleVersion();
    SOFA_CUDASOLVER_API const char* getModuleLicense();
    SOFA_CUDASOLVER_API const char* getModuleDescription();
}

const char* getModuleName()
{
    return "ISofaCudaSolverPlugin";
}

const char* getModuleVersion()
{
    return "0.1";
}

const char* getModuleLicense()
{
    return "Undefined";
}

const char* getModuleDescription()
{
    return "Linear solver with resolution on GPU";
}

}

}