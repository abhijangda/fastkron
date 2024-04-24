cmake_minimum_required(VERSION 3.25)
set(CMAKE_DEBUG_POSTFIX d)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_HIP_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CUDA_STANDARD_REQUIRED True)
set(CMAKE_HIP_STANDARD_REQUIRED True)

project(FastKron 
  VERSION 1.0
  LANGUAGES CXX)

find_package(OpenMP REQUIRED)

if(OpenMP_CXX_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_HIP_FLAGS "${CMAKE_CUDA_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

#### Options ####
option(ENABLE_CUDA "Enable CUDA" ON)
option(ENABLE_X86  "Enable X86" OFF)
option(ENABLE_ARM  "Enable ARM" OFF)
option(ENABLE_HIP "Enable ROCM/HIP" OFF)

if(ENABLE_CUDA)
  add_definitions(-DENABLE_CUDA)
  enable_language(CUDA)
  find_package(CUDA 11.0 REQUIRED)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED True)
  
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
  if (CMAKE_BUILD_TYPE MATCHES "Debug")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G")
  endif()
endif()

if(ENABLE_X86)
  add_definitions(-DENABLE_X86)
endif()

if(ENABLE_ARM)
  add_definitions(-DENABLE_ARM)
endif()

if(ENABLE_HIP)
  find_package(HIP 5.7 REQUIRED)
  add_definitions(-DENABLE_HIP -D__HIP_PLATFORM_AMD__)
  #TODO: Fix this
  set(HIP_INCLUDE_PATH /opt/rocm/include)
  set(CMAKE_HIP_STANDARD 17)
  set(CMAKE_HIP_STANDARD_REQUIRED True)
endif()
################

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# TODO set these as RELEASE flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2 -mfma -march=native -O3")

set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")

set(CMAKE_HIP_FLAGS "-O3 -Rpass-analysis=kernel-resource-usage --save-temps")

set(SRC ${CMAKE_CURRENT_SOURCE_DIR}/src)
set(CUDA ${SRC}/kernels/cuda)
set(KERNELS ${SRC}/kernels)

##### FastKron ################
set(KERNEL_HEADER_FILES 
    ${KERNELS}/kernel_info.h
    ${KERNELS}/params.h)

if(ENABLE_CUDA)
  if (EXISTS ${CUDA}/kron-kernels/kernels.cmake) 
    include(${CUDA}/kron-kernels/kernels.cmake)
  endif()
  if (EXISTS ${CUDA}/kron-kernels/kernel_decl.inc)
    set(CUDA_HEADER_FILES ${CUDA}/kron-kernels/kernel_decl.inc)
  else()
    set(CUDA_HEADER_FILES)
  endif()

  set(CUDA_HEADER_FILES 
      ${CUDA_HEADER_FILES}
      ${CUDA}/utils.cuh 
      ${CUDA}/shared-loads.cuh
      ${CUDA}/register-loads.cuh
      ${CUDA}/kernel.cuh
      ${CUDA}/otherkernels.cuh
      ${SRC}/kernel_db/cuda_kernel_db.h
      ${SRC}/kernel_db/cuda_kernel_db.cu
      ${CUDA_KERNELS})
else()
  set(CUDA_HEADER_FILES)
endif()

if(ENABLE_X86)
  include(${KERNELS}/cpu/x86/kron-kernels/kernels.cmake)
  set(X86_HEADER_FILES
      ${KERNELS}/cpu/x86/kron-kernels/kernel_decl.inc
      ${SRC}/kernel_db/cpu_kernel_db.cpp
      ${SRC}/kernel_db/cpu_kernel_db.h
      ${X86_KERNELS})
else()
  set(X86_HEADER_FILES)
endif()

if(ENABLE_HIP)
  include(${KERNELS}/hip/kron-kernels/kernels.cmake)
  set(HIP_HEADER_FILES 
      ${KERNELS}/hip/kron-kernels/kernel_decl.inc
      ${SRC}/kernel_db/hip_kernel_db.hip
      ${SRC}/kernel_db/hip_kernel_db.h
      ${HIP_KERNELS})
else()
  set(HIP_HEADER_FILES)
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

if (ENABLE_CUDA)
  set(SOURCES ${SOURCES} ${SRC}/handle/distrib_handle.cu)
else()
  set(SOURCES ${SOURCES} ${SRC}/handle/distrib_handle.cpp)
endif()

add_library(FastKron
  SHARED
    ${SOURCES}
    ${HEADER_FILES}
    ${KERNEL_HEADER_FILES}
    ${CUDA_HEADER_FILES}
    ${X86_HEADER_FILES}
    ${HIP_HEADER_FILES}
)

target_link_libraries(FastKron nccl)
target_include_directories(FastKron 
                           PRIVATE 
                              ${SRC}/
                           PUBLIC
                              $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                              $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
set_target_properties(FastKron PROPERTIES PUBLIC_HEADER "include/fastkron.h")
install(TARGETS FastKron)

add_subdirectory(tests/)
