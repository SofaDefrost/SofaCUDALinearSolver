cmake_minimum_required(VERSION 3.17)
project(SofaCudaSolver VERSION 1.0 LANGUAGES CXX)


# Find and load CMake configuration of packages containing this plugin's dependencies
## Mandatory dependencies
find_package(SofaFramework REQUIRED)
sofa_find_package(Sofa.Component.LinearSolver.Iterative REQUIRED)
find_package(CUDA REQUIRED)
find_package(CUDAToolkit REQUIRED)

# List all files
set(SOFACUDASOLVER_SRC_DIR src/${PROJECT_NAME})
set(HEADER_FILES
    ${SOFACUDASOLVER_SRC_DIR}/config.h.in
    ${SOFACUDASOLVER_SRC_DIR}/CUDACholeksySparseSolver.h
    ${SOFACUDASOLVER_SRC_DIR}/CUDACholeksySparseSolver.inl
    ${SOFACUDASOLVER_SRC_DIR}/SolverGPU.h
    ${SOFACUDASOLVER_SRC_DIR}/SolverGPU.inl
    ${SOFACUDASOLVER_SRC_DIR}/utils.h
    )
set(SOURCE_FILES
    ${SOFACUDASOLVER_SRC_DIR}/initSofaCudaSolver.cpp
    ${SOFACUDASOLVER_SRC_DIR}/CUDACholeksySparseSolver.cpp
    ${SOFACUDASOLVER_SRC_DIR}/SolverGPU.cpp
    )
set(README_FILES
    README.md
    )
    

cuda_include_directories("${SOFACUDASOLVER_SRC_DIR}")
cuda_include_directories("${CMAKE_CURRENT_SOURCE_DIR}")
# Create the plugin library.
cuda_add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES} ${README_FILES})

# Link the plugin library to its dependency(ies).
target_link_libraries(${PROJECT_NAME} SofaFramework)
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

# Tests
# If SOFA_BUILD_TESTS exists and is OFF, then these tests will be auto-disabled
#cmake_dependent_option(SOFACUDASOLVER_BUILD_TESTS "Compile the automatic tests" ON "SOFA_BUILD_TESTS OR NOT DEFINED SOFA_BUILD_TESTS" OFF)

#if(SOFACUDASOLVER_BUILD_TESTS)
#    enable_testing()
#    add_subdirectory(SOFACUDASOLVER_test)
#endif()

