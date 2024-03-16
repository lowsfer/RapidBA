cmake_minimum_required(VERSION 3.10)
# cuda 11.2 causes register spill for many kernels
# set(CMAKE_CUDA_COMPILER "/usr/local/cuda-11.1/bin/nvcc")

project(RapidBA LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 75-real 86-real)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

#if(CMAKE_BUILD_TYPE MATCHES "Debug")
#    add_definitions(-D_DEBUG -D_GLIBCXX_DEBUG)
#endif(CMAKE_BUILD_TYPE MATCHES "Debug")

find_package( OpenCV REQUIRED )
include_directories(${OpenCV_INCLUDE_DIRS})

include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
link_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}/../lib64)

#set(CMAKE_CXX_FLAGS_DEBUG "-march=native -O0 -pipe -Wall -Werror -Wno-unused-local-typedefs -Wno-type-limits -Wno-unknown-pragmas -Wno-error=unused-local-typedefs -g")
#set(CMAKE_CXX_FLAGS_RELEASE "-march=native -O3 -pipe -Wall -Werror -Wno-unused-local-typedefs -Wno-type-limits -Wno-unknown-pragmas -Wno-error=unused-local-typedefs")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAG} -march=native -pipe -Wall -Werror -Wno-unused-local-typedefs -Wno-type-limits -Wno-unknown-pragmas -Wno-error=unused-local-typedefs -Wno-error=deprecated-declarations")

set(CUDA_SEPARABLE_COMPILATION OFF)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo -keep --expt-relaxed-constexpr --use_fast_math") #-res-usage
set(CUDA_PTXAS_FLAGS_RELEASE "${CUDA_PTXAS_FLAGS_RELEASE} -warn-lmem-usage -warn-spills -Werror") # -v
set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -Xptxas='${CUDA_PTXAS_FLAGS_RELEASE}'")
add_library(RapidBA
        RapidBA.cpp RapidBA.h
        derivative.h kernel.h cuda_hint.cuh
        computeHessian.cu
        csr.h
        kmat.h utils_kernel.h utils_general.h utils_host.h
        traits/Traits.h computeHessian.h computeSchur.cu computeSchur.h transpose.cuh
        GroupModel.cpp GroupModel.h containers.h GroupModelTypes.h GroupModelTypes.cpp
        blockSolvers/solveSchurPCG.cu blockSolvers/solveSchurPCG.h
        blockSolvers/SchurSolverPCG.cpp blockSolvers/SchurSolverPCG.h
        blockSolvers/SchurSolver.h blockSolvers/SchurSolver.cpp
        blockSolvers/HessianSolver.cpp blockSolvers/HessianSolver.h
        blockSolvers/HessianSolverExplicitSchurPCG.cpp blockSolvers/HessianSolverExplicitSchurPCG.h blockSolvers/solveHessian.h blockSolvers/solverHessian.cu
        BundleSolver.cpp BundleSolver.h BundleSolver.cu
        DiscreteModel.h  DiscreteModel.cpp
        traits/capture.h traits/base.h traits/intrinsic.h
        platform.h fwd.h ldg.cuh smemAccUtils.h
        ModelFilter.cpp ModelFilter.h UniversalModel.cpp UniversalModel.h)
target_compile_definitions(RapidBA PRIVATE RBA_SANITY_CHECK=0)

target_include_directories(RapidBA PRIVATE /opt/cub-1.8.0)

add_executable(tests
        test/DerivativeTest.cpp
        test/TestModel.h test/TestModel.cpp
        test/HessianTest.cpp
        test/UtilsTest.cpp
        test/SchurTest.cpp
        test/PerfTest.cpp
        test/PCGTest.cpp test/utils_test.cpp test/DeltaBTestAndDampTest.cpp
        test/ReduceTest.cpp test/BundleSolverTest.cpp)
target_compile_definitions(tests PRIVATE UNIT_TEST)
target_link_libraries(tests
        gtest gtest_main cudart RapidBA)

add_executable(demo main.cpp traits/intrinsic.h)
target_link_libraries(demo
        RapidBA cudart)
