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

#### Options ####
option(ENABLE_CUDA "Enable CUDA" OFF)
option(ENABLE_X86  "Enable X86" OFF)
option(ENABLE_ARM  "Enable ARM" OFF)
option(ENABLE_ROCM "Enable ROCM/HIP" OFF)

if(ENABLE_CUDA)
  add_definitions(-DENABLE_CUDA)
endif()
if(ENABLE_X86)
  add_definitions(-DENABLE_X86)
endif()
if(ENABLE_ARM)
  add_definitions(-DENABLE_ARM)
endif()
if(ENABLE_ROCM)
  add_definitions(-DENABLE_ROCM)
endif()
################

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
set(CUDA ${SRC}/kernels/cuda)
set(KERNELS ${SRC}/kernels)

##### FastKron ################
set(KERNEL_HEADER_FILES 
    ${KERNELS}/kernel_info.h
    ${KERNELS}/params.h
    #TODO Fix this
    ${SRC}/handle/distrib_handle.cu)

if(ENABLE_CUDA)
  include(${CUDA}/kron-kernels/kernels.cmake)
  set(CUDA_HEADER_FILES 
      ${CUDA}/kron-kernels/kernel_decl.inc
      ${CUDA}/utils.cuh 
      ${CUDA}/shared-loads.cuh
      ${CUDA}/register-loads.cuh
      ${CUDA}/kernel.cuh
      ${CUDA}/otherkernels.cuh
      ${SRC}/kernel_db/cuda_kernel_db.h
      ${SRC}/kernel_db/cuda_kernel_db.cpp
      ${CUDA_KERNELS})
else()
  set(CUDA_HEADER_FILES)
endif()

if (ENABLE_X86)
  set(X86_HEADER_FILES
      ${KERNELS}/cpu/x86/kron-kernels/kernels.cmake
      ${KERNELS}/cpu/x86/kron-kernels/kernel_decl.inc
      ${SRC}/kernel_db/cpu_kernel_db.cpp
      ${SRC}/kernel_db/cpu_kernel_db.h
      ${X86_KERNELS}
      )
else()
  set(X86_HEADER_FILES)
endif()

set(HEADER_FILES 
    ${SRC}/env/env.h 
    ${SRC}/autotuner/autotuner.h
    ${SRC}/handle/handle.h 
    ${SRC}/handle/op.h 
    ${SRC}/kernel_db/kernel_defs.h 
    ${SRC}/kmm/kmmalgo.h
    ${SRC}/kmm/matrix.h
    ${SRC}/utils/thread_pool.h
    ${SRC}/utils/utils.h)

set(SOURCES
    ${SRC}/fastkron.cpp
    ${SRC}/env/env.cpp
    ${SRC}/autotuner/autotuner.cpp 
    ${SRC}/handle/handle.cpp
    ${SRC}/kernel_db/kernel_db.cpp
    ${SRC}/kmm/kmmalgo.cpp
    ${SRC}/kmm/matrix.cpp)

cuda_add_library(FastKron 
  SHARED
    ${SOURCES}
    ${HEADER_FILES}
    ${KERNEL_HEADER_FILES}
    ${CUDA_HEADER_FILES}
    ${X86_HEADER_FILES}
  OPTIONS
    ${NVCC_FLAGS}
)

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

add_subdirectory(tests/)