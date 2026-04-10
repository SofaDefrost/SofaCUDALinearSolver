cmake_minimum_required(VERSION 3.24)
project(SofaCUDALinearSolver VERSION 1.0 LANGUAGES CUDA CXX)


# Find and load CMake configuration of packages containing this plugin's dependencies
## Mandatory dependencies
find_package(Sofa.Core REQUIRED)
sofa_find_package(Sofa.Component.LinearSolver.Iterative REQUIRED)
sofa_find_package(CUDAToolkit 12.0 REQUIRED)
sofa_find_package(cudss QUIET)

if(cudss_FOUND)
    message("cudss has been found, enabling CUDACholeksySparseSolverCUDSS.")
else()
    message("cudss has not been found, cudss-based Solver will not be compiled.")
endif()

# List all files
set(SOFACUDALINEARSOLVER_SRC_DIR src/${PROJECT_NAME})
set(HEADER_FILES
    ${SOFACUDALINEARSOLVER_SRC_DIR}/config.h.in
    ${SOFACUDALINEARSOLVER_SRC_DIR}/CUDACholeksySparseSolver.h
    ${SOFACUDALINEARSOLVER_SRC_DIR}/CUDACholeksySparseSolver.inl
    ${SOFACUDALINEARSOLVER_SRC_DIR}/utils.h
)
set(SOURCE_FILES
    ${SOFACUDALINEARSOLVER_SRC_DIR}/initSofaCUDALinearSolver.cpp
    ${SOFACUDALINEARSOLVER_SRC_DIR}/CUDACholeksySparseSolver.cpp
)
set(README_FILES
    README.md
)

if(cudss_FOUND)
    list(APPEND HEADER_FILES
        ${SOFACUDALINEARSOLVER_SRC_DIR}/CUDASparseCholeskySolverCUDSS.h
        ${SOFACUDALINEARSOLVER_SRC_DIR}/CUDASparseCholeskySolverCUDSS.inl
    )
    list(APPEND SOURCE_FILES
        ${SOFACUDALINEARSOLVER_SRC_DIR}/CUDASparseCholeskySolverCUDSS.cpp
    )
endif()

# Create the plugin library.
add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES} ${README_FILES})

# Link the plugin library to its dependency(ies).
target_link_libraries(${PROJECT_NAME} Sofa.Core)
target_link_libraries(${PROJECT_NAME} Sofa.Component.LinearSolver.Iterative)
target_link_libraries(${PROJECT_NAME} CUDA::cudart CUDA::cusparse CUDA::cusolver cudss)

if(cudss_FOUND)
    target_link_libraries(${PROJECT_NAME} cudss)
endif()

target_include_directories(${PROJECT_NAME} PUBLIC "/usr/include/cuda/")

# Create package Config, Version & Target files.
# Deploy the headers, resources, scenes & examples.
# Set the plugin 'relocatable' if built within SOFA.
# --> see SofaMacros.cmake
sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${PROJECT_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR ${PROJECT_NAME}
    RELOCATABLE "plugins"
    )

