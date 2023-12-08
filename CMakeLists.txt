cmake_minimum_required(VERSION 3.20)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CUDA_STANDARD_REQUIRED True)
project(FastKron 
  VERSION 1.0
  LANGUAGES CUDA CXX)
find_package(CUDA 12.0 REQUIRED)
find_package(OpenMP REQUIRED)
add_executable(kron tests/main.cu)

if(OpenMP_CXX_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${OpenMP_CXX_FLAGS}")
    target_link_libraries(kron PUBLIC OpenMP::OpenMP_CXX)
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CUDA_STANDARD_REQUIRED True)

set(SRC src)
set(DEVICE ${SRC}/device)

include(${DEVICE}/kron-kernels/kernels.cmake)
set(DEVICE_HEADER_FILES ${DEVICE}/kron-kernels/kernel_decl.inc ${DEVICE}/device_functions.cuh ${DEVICE}/kernel_info.h ${DEVICE}/kernel.cuh ${DEVICE}/otherkernels.cuh ${DEVICE}/params.h)
set(HEADER_FILES ${SRC}/env.h ${SRC}/autotuner.h ${SRC}/fastkron.h ${SRC}/handle.h ${SRC}/kernel_defs.cuh ${SRC}/kernel_invoker.h ${SRC}/kmmalgo.h ${SRC}/sliced_mul_shape.h ${SRC}/thread_pool.h ${SRC}/utils.h)
set(SOURCES ${SRC}/fastkron.cu ${SRC}/autotuner.cpp ${SRC}/distrib_handle.cu ${SRC}/env.cpp ${SRC}/handle.cu ${SRC}/kernel_invoker.cu ${SRC}/kmmalgo.cu ${SRC}/sliced_mul_shape.cpp)

cuda_add_library(FastKron SHARED
    ${SOURCES}
    ${HEADER_FILES}
    ${DEVICE_HEADER_FILES}
    ${CUDA_KERNELS}
    OPTIONS -std=c++17 -gencode arch=compute_80,code=sm_80
)

target_include_directories(FastKron PRIVATE ${SRC}/)

add_subdirectory(AnyOption)
add_dependencies(kron anyoption)
target_include_directories(kron PUBLIC AnyOption ${SRC}/)
#Fix NCCL
target_link_libraries(kron PUBLIC FastKron anyoption nccl)