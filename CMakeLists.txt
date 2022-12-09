cmake_minimum_required(VERSION 3.17)
project(SofaCUDALinearSolver VERSION 1.0 LANGUAGES CXX)


# Find and load CMake configuration of packages containing this plugin's dependencies
## Mandatory dependencies
find_package(Sofa.Core REQUIRED)
sofa_find_package(Sofa.Component.LinearSolver.Iterative REQUIRED)
find_package(CUDA REQUIRED)
find_package(CUDAToolkit REQUIRED)

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


cuda_include_directories("${SOFACUDALINEARSOLVER_SRC_DIR}")
cuda_include_directories("${CMAKE_CURRENT_SOURCE_DIR}")
# Create the plugin library.
cuda_add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES} ${README_FILES})

# Link the plugin library to its dependency(ies).
target_link_libraries(${PROJECT_NAME} Sofa.Core)
target_link_libraries(${PROJECT_NAME} Sofa.Component.LinearSolver.Iterative)
target_link_libraries(${PROJECT_NAME} CUDA::cudart CUDA::cusparse CUDA::cusolver)

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

