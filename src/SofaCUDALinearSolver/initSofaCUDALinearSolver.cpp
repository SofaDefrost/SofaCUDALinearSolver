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
#include <SofaCUDALinearSolver/config.h>
#include <sofa/core/ObjectFactory.h>
#include <SofaCUDALinearSolver/utils.h>
using sofa::core::ObjectFactory;


extern "C" {

    SOFACUDALINEARSOLVER_API void initExternalModule();
    SOFACUDALINEARSOLVER_API const char* getModuleName();
    SOFACUDALINEARSOLVER_API const char* getModuleVersion();
    SOFACUDALINEARSOLVER_API const char* getModuleLicense();
    SOFACUDALINEARSOLVER_API const char* getModuleDescription();
    SOFACUDALINEARSOLVER_API const char* getModuleComponentList();
    SOFACUDALINEARSOLVER_API bool moduleIsInitialized();
}

bool isModuleInitialized = false;

void initExternalModule()
{
    static bool first = true;
    if (first)
    {
        isModuleInitialized = SofaCUDALinearSolver::cudaInit();
        first = false;
    }
}

const char* getModuleName()
{
    return sofa_tostring(SOFA_TARGET);
}

const char* getModuleLicense()
{
    return "";
}

const char* getModuleVersion()
{
    return sofa_tostring(SOFACUDALINEARSOLVER_VERSION);
}

const char* getModuleDescription()
{
    return "Linear solver with resolution on GPU";
}

const char* getModuleComponentList()
{
    /// string containing the names of the classes provided by the plugin
    static std::string classes = ObjectFactory::getInstance()->listClassesFromTarget(sofa_tostring(SOFA_TARGET));
    return classes.c_str();
}

bool moduleIsInitialized()
{
    return isModuleInitialized;
}
