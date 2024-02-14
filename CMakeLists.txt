cmake_minimum_required(VERSION 3.20)
set(CMAKE_DEBUG_POSTFIX d)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CUDA_STANDARD_REQUIRED True)
project(FastKron 
  VERSION 1.0
  LANGUAGES CUDA CXX)
find_package(CUDA 12.0 REQUIRED)
find_package(OpenMP REQUIRED)

if(OpenMP_CXX_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CUDA_STANDARD_REQUIRED True)

set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
set(CMAKE_CUDA_FLAGS_DEBUG "-G")

########NVCC FLAGS###############
if (DEFINED NVCC_GENCODE_FLAGS)
else()
set(NVCC_GENCODE_FLAGS "-gencode arch=compute_70,code=sm_70 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_90,code=sm_90")
endif()

set(NVCC_FLAGS "${NVCC_FLAGS} -std=c++17 ${NVCC_GENCODE_FLAGS}")
if (CMAKE_BUILD_TYPE MATCHES "Debug")
    set(NVCC_FLAGS "${NVCC_FLAGS} -G")
endif()
################################

set(SRC ${CMAKE_CURRENT_SOURCE_DIR}/src)
set(DEVICE ${SRC}/device)

##### FastKron ################
include(${DEVICE}/kron-kernels/kernels.cmake)
set(DEVICE_HEADER_FILES ${DEVICE}/kron-kernels/kernel_decl.inc
                        ${DEVICE}/utils.cuh ${DEVICE}/shared-loads.cuh ${DEVICE}/register-loads.cuh
                        ${DEVICE}/kernel_info.h ${DEVICE}/kernel.cuh ${DEVICE}/otherkernels.cuh
                        ${DEVICE}/params.h)
set(HEADER_FILES ${SRC}/env/env.h 
                 ${SRC}/autotuner/autotuner.h
                 ${SRC}/handle/handle.h ${SRC}/handle/op.h 
                 ${SRC}/kernel_db/kernel_defs.h ${SRC}/kernel_db/kernel_db.h 
                 ${SRC}/kmm/kmmalgo.h ${SRC}/kmm/matrix.h
                 ${SRC}/utils/thread_pool.h ${SRC}/utils/utils.h)

set(SOURCES ${SRC}/fastkron.cpp
            ${SRC}/env/env.cpp
            ${SRC}/autotuner/autotuner.cpp 
            ${SRC}/handle/handle.cpp ${SRC}/handle/distrib_handle.cu
            ${SRC}/kernel_db/kernel_db.cpp 
            ${SRC}/kmm/kmmalgo.cpp ${SRC}/kmm/matrix.cpp)

cuda_add_library(FastKron SHARED
    ${SOURCES}
    ${HEADER_FILES}
    ${DEVICE_HEADER_FILES}
    ${CUDA_KERNELS}
    OPTIONS ${NVCC_FLAGS}
)

#FixNCCL 
target_link_libraries(FastKron nccl)
target_include_directories(FastKron 
                           PRIVATE 
                              ${SRC}/
                              ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
                           PUBLIC
                              $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                              $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
set_target_properties(FastKron PROPERTIES PUBLIC_HEADER "include/fastkron.h")
install(TARGETS FastKron)

##########################

add_subdirectory(tests/)