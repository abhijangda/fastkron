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
add_executable(benchmark benchmarks/benchmark.cu)

if(OpenMP_CXX_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${OpenMP_CXX_FLAGS}")
    target_link_libraries(benchmark PUBLIC OpenMP::OpenMP_CXX)
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

set(SRC src)
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

#FixNCCL target_link_libraries(FastKron PUBLIC nccl)
target_include_directories(FastKron 
                           PRIVATE 
                              ${SRC}/
                              ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
                           PUBLIC
                              $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                              $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
set_target_properties(FastKron PROPERTIES PUBLIC_HEADER "include/fastkron.h")
##########################

##### kron ################
add_subdirectory(AnyOption)
add_dependencies(benchmark anyoption)
target_include_directories(benchmark PUBLIC AnyOption ${SRC}/ tests/)
target_link_libraries(benchmark PUBLIC FastKron anyoption nccl)
##########################

############################### Tests ###################################
find_package (Python3)
set(GEN_TUNER_KERNELS_PY ${CMAKE_CURRENT_SOURCE_DIR}/${SRC}/gen_tuner_kernels.py)

ADD_SUBDIRECTORY(googletest)
enable_testing()
include_directories(${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})

set(TESTS_INCLUDES ${SRC}/ tests)
set(TESTS_LIBS GTest::gtest_main FastKron nccl)

add_executable(single-gpu-no-fusion-tests tests/single-gpu-no-fusion-tests.cpp)
add_executable(single-gpu-fusion-tests tests/single-gpu-fusion-tests.cpp)
add_custom_target(gen-single-gpu-kernels
    COMMAND python ${GEN_TUNER_KERNELS_PY} -same-factors 20 2,2 -same-factors 10 4,4 -same-factors 8 8,8 -same-factors 6 16,16 -same-factors 5 32,32 -same-factors 4 64,64 -same-factors 3 128,128 -opX N -opF N -match-configs-file ${CMAKE_CURRENT_SOURCE_DIR}/tests/single-gpu-kernel-decls.in
)
target_include_directories(single-gpu-no-fusion-tests PUBLIC ${TESTS_INCLUDES})
target_link_libraries(single-gpu-no-fusion-tests ${TESTS_LIBS})

target_include_directories(single-gpu-fusion-tests PUBLIC ${TESTS_INCLUDES})
target_link_libraries(single-gpu-fusion-tests ${TESTS_LIBS})

add_executable(single-gpu-NT-tests tests/single-gpu-NT-tests.cpp)
add_custom_target(gen-single-gpu-NT-kernels
    COMMAND python ${GEN_TUNER_KERNELS_PY} -same-factors 8 8,8 -same-factors 6 16,16 -same-factors 5 32,32 -same-factors 4 64,64 -same-factors 3 128,128 -opX N -opF T -match-configs-file ${CMAKE_CURRENT_SOURCE_DIR}/tests/single-gpu-kernel-decls.in
)
target_include_directories(single-gpu-NT-tests PUBLIC ${TESTS_INCLUDES})
target_link_libraries(single-gpu-NT-tests ${TESTS_LIBS})

add_executable(single-gpu-tuner-tests tests/single-gpu-tuner-tests.cpp)
add_custom_target(gen-tuner-kernels
    COMMAND python ${GEN_TUNER_KERNELS_PY} -same-factors 4 16,16 -same-factors 3 64,64 -opX N -opF N
)
target_include_directories(single-gpu-tuner-tests PUBLIC ${TESTS_INCLUDES})
target_link_libraries(single-gpu-tuner-tests ${TESTS_LIBS})

add_executable(single-gpu-non-square-tuner-tests tests/single-gpu-non-square-tuner-tests.cpp)
add_custom_target(gen-non-square-tuner-test-kernels
    COMMAND python ${GEN_TUNER_KERNELS_PY} -same-factors 4 8,16 -same-factors 3 32,16 -same-factors 3 128,32 -opX N -opF N
)
target_include_directories(single-gpu-non-square-tuner-tests PUBLIC ${TESTS_INCLUDES})
target_link_libraries(single-gpu-non-square-tuner-tests ${TESTS_LIBS})

add_executable(single-gpu-distinct-shapes tests/single-gpu-distinct-shapes.cpp)
add_custom_target(gen-single-gpu-distinct-shapes
    COMMAND python ${GEN_TUNER_KERNELS_PY} -distinct-factors 3 8,16 16,8 8,32 -opX N -opF N
)
target_include_directories(single-gpu-distinct-shapes PUBLIC ${TESTS_INCLUDES})
target_link_libraries(single-gpu-distinct-shapes ${TESTS_LIBS})

add_executable(single-gpu-odd-shapes tests/single-gpu-odd-shapes.cpp)
add_custom_target(gen-single-gpu-odd-shapes
    COMMAND python ${GEN_TUNER_KERNELS_PY} -same-factors 2 31,16 -same-factors 2 16,31 -same-factors 4 31,31 -opX N -opF N
)
target_include_directories(single-gpu-odd-shapes PUBLIC ${TESTS_INCLUDES})
target_link_libraries(single-gpu-odd-shapes ${TESTS_LIBS})

add_executable(multi-gpu-no-fusion-tests tests/multi-gpu-no-fusion-tests.cpp)
add_custom_target(gen-multi-gpu-tests-kernel
    COMMAND python ${GEN_TUNER_KERNELS_PY} -same-factors 4 64,64 -same-factors 4 128,128 -dist-kernels -match-configs 128,64,64,64,4096,2,1,0,2,16 128,64,64,64,4096,2,1,1,2,16 128,128,128,128,8192,1,1,0,2,32 128,128,128,128,8192,1,1,1,2,32 -opX N -opF N
)
target_include_directories(multi-gpu-no-fusion-tests PUBLIC ${TESTS_INCLUDES})
target_link_libraries(multi-gpu-no-fusion-tests ${TESTS_LIBS})

add_executable(multi-gpu-tuner-tests tests/multi-gpu-tuner-tests.cpp)
add_custom_target(gen-multi-gpu-tuner-kernels
    COMMAND python ${GEN_TUNER_KERNELS_PY} -same-factors 5 16,16 -dist-kernels -opX N -opF N
)
target_include_directories(multi-gpu-tuner-tests PUBLIC ${TESTS_INCLUDES})
target_link_libraries(multi-gpu-tuner-tests ${TESTS_LIBS})

add_executable(multi-gpu-no-fusion-non-square-tests tests/multi-gpu-no-fusion-non-square-tests.cpp)
add_custom_target(gen-multi-gpu-no-fusion-non-square-tests-kernel
    COMMAND python ${GEN_TUNER_KERNELS_PY} -same-factors 5 8,32 -same-factors 4 64,16 -dist-kernels -opX N -opF N
)
target_include_directories(multi-gpu-no-fusion-non-square-tests PUBLIC ${TESTS_INCLUDES})
target_link_libraries(multi-gpu-no-fusion-non-square-tests ${TESTS_LIBS})

add_executable(multi-gpu-distinct-shapes tests/multi-gpu-distinct-shapes.cpp)
add_custom_target(gen-multi-gpu-distinct-shapes
    COMMAND python ${GEN_TUNER_KERNELS_PY} -distinct-factors 3 8,16 32,8 16,8 -match-configs 64,16,8,16,256 32,8,32,8,256 128,8,16,8,256 -dist-kernels -opX N -opF N
)
target_include_directories(multi-gpu-distinct-shapes PUBLIC ${TESTS_INCLUDES})
target_link_libraries(multi-gpu-distinct-shapes ${TESTS_LIBS})
###################################################################