cmake_minimum_required(VERSION 3.25)
set(CMAKE_DEBUG_POSTFIX d)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_HIP_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_HIP_STANDARD_REQUIRED True)

project(FastKron 
  VERSION 1.0
  LANGUAGES CXX)

find_package(OpenMP REQUIRED)
find_package(Python3 REQUIRED)

if(OpenMP_CXX_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_HIP_FLAGS "${CMAKE_CUDA_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "12.0")
    message (FATAL_ERROR "Require Clang 12.0 or later")
  endif()
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "11.0")
    message (FATAL_ERROR "Require GCC/G++ 11.0 or later")
  endif()
endif()

set(PKG_CONFIG_FILE libFastKron.pc)

#### Options ####
option(ENABLE_CUDA "Enable CUDA"                    ON)
option(ENABLE_X86  "Enable X86"                     ON)
option(ENABLE_ARM  "Enable ARM"                    OFF)
option(ENABLE_HIP  "Enable ROCM/HIP"               OFF)
option(FULL_TUNE   "Do Full Tuning"                OFF)
option(PYMODULE    "Install and Build as PyModule" OFF)

if(ENABLE_CUDA)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "70;80")
  endif()
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED True)
  add_definitions(-DENABLE_CUDA)
  enable_language(CUDA)
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
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=x86-64-v2 -O3")

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
  if (FULL_TUNE)
  else()
    set(FK_CUDA_ARCH "")
    foreach(SM_ARCH ${CMAKE_CUDA_ARCHITECTURES})
      if ((SM_ARCH STREQUAL "80") OR (SM_ARCH STRGREATER "80"))
        set(arch "ampere")
      elseif (SM_ARCH STREQUAL "70")
        set(arch "volta")
      elseif (SM_ARCH STRLESS 70)
        set(arch "maxwell")
      endif()
      set(FK_CUDA_ARCH ${arch} ${FK_CUDA_ARCH})
    endforeach()

    execute_process(COMMAND python3 ${SRC}/gen_tuner_kernels.py -backend cuda -archs ${FK_CUDA_ARCH} -same-factors 2 128,128 -same-factors 2 64,64 -same-factors 3 32,32 -same-factors 5 16,16 -same-factors 7 8,8 -same-factors 10 4,4 -same-factors 20 2,2 -opX N T -opF N T -types float double -match-configs-file ${SRC}/kernels/best-kernels/a100-kernels)
  endif()
  include(${CUDA}/kron-kernels/kernels.cmake OPTIONAL)
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
  if (FULL_TUNE)
  else()
    execute_process(COMMAND python3 ${SRC}/gen_tuner_kernels.py -backend x86 -archs sisd avx avx512 -same-factors 2 128,128 -same-factors 2 64,64 -same-factors 3 32,32 -same-factors 5 16,16 -same-factors 7 8,8 -same-factors 10 4,4 -same-factors 20 2,2 -opX N T -opF N T -types float double -match-configs-file ${SRC}/kernels/best-kernels/x86-avx-kernels)
  endif()
  include(${KERNELS}/cpu/x86/kron-kernels/kernels.cmake OPTIONAL)
  if (EXISTS ${KERNELS}/cpu/x86/kron-kernels/kernel_decl.inc) 
    set(X86_HEADER_FILES
        ${KERNELS}/cpu/x86/kron-kernels/kernel_decl.inc
        ${KERNELS}/cpu/vector-types.h
        ${KERNELS}/cpu/tensor.h)
  else()
    set(X86_HEADER_FILES)
  endif()

  set(X86_HEADER_FILES
      ${X86_HEADER_FILES}
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
    ${SRC}/utils/utils.h
    ${SRC}/utils/logger.h)

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

if (PYMODULE)
  ## PyBind ##
  add_subdirectory(pybind11)
  pybind11_add_module(FastKron
    SHARED
      ${SOURCES}
      ${SRC}/../pyfastkron/pywrapper.cpp
      ${HEADER_FILES}
      ${KERNEL_HEADER_FILES}
      ${CUDA_HEADER_FILES}
      ${X86_HEADER_FILES}
      ${HIP_HEADER_FILES})
  install(TARGETS FastKron DESTINATION ${SKBUILD_PROJECT_NAME})
else()
  add_library(FastKron
    SHARED
      ${SOURCES}
      ${HEADER_FILES}
      ${KERNEL_HEADER_FILES}
      ${CUDA_HEADER_FILES}
      ${X86_HEADER_FILES}
      ${HIP_HEADER_FILES}
  )
  add_subdirectory(tests/)
  ## PkgConfig ##

  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/libFastKron.pc.in ${PKG_CONFIG_FILE} @ONLY)
  add_custom_target(install_pkgconfig
                    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_BINARY_DIR}/${PKG_CONFIG_FILE} ${CMAKE_INSTALL_PREFIX}/lib/pkgconfig/)

  set_target_properties(FastKron PROPERTIES PUBLIC_HEADER "include/fastkron.h")
  install(TARGETS FastKron)
endif()

target_link_libraries(FastKron PRIVATE nccl)
target_include_directories(FastKron 
                           PRIVATE 
                              ${SRC}/
                           PUBLIC
                              $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                              $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)