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

############################### Tests ###################################
find_package (Python3)
set(GEN_TUNER_KERNELS_PY ${CMAKE_CURRENT_SOURCE_DIR}/${SRC}/gen_tuner_kernels.py)

ADD_SUBDIRECTORY(googletest)
enable_testing()
include_directories(${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})

set(TESTS_INCLUDES ${SRC}/ tests)
set(TESTS_LIBS GTest::gtest_main FastKron nccl)

add_executable(single-gpu-no-fusion-tests tests/single-gpu-no-fusion-tests.cu)
add_executable(single-gpu-fusion-tests tests/single-gpu-no-fusion-tests.cu)
add_custom_target(gen-single-gpu-kernels
    ALL
    COMMAND python ${GEN_TUNER_KERNELS_PY} -same-factors 20 2,2 -same-factors 10 4,4 -same-factors 8 8,8 -same-factors 6 16,16 -same-factors 5 32,32 -same-factors 4 64,64 -same-factors 3 128,128 -match-configs-file ${CMAKE_CURRENT_SOURCE_DIR}/tests/single-gpu-kernel-decls.in
)
target_include_directories(single-gpu-no-fusion-tests PUBLIC ${TESTS_INCLUDES})
target_link_libraries(single-gpu-no-fusion-tests ${TESTS_LIBS})

target_include_directories(single-gpu-fusion-tests PUBLIC ${TESTS_INCLUDES})
target_link_libraries(single-gpu-fusion-tests ${TESTS_LIBS})

add_executable(single-gpu-tuner-tests tests/single-gpu-tuner-tests.cu)
add_custom_target(gen-tuner-kernels
    ALL
    COMMAND python ${GEN_TUNER_KERNELS_PY} -same-factors 4 16,16 -same-factors 3 64,64
)
target_include_directories(single-gpu-tuner-tests PUBLIC ${TESTS_INCLUDES})
target_link_libraries(single-gpu-tuner-tests ${TESTS_LIBS})

add_executable(single-gpu-non-square-tuner-tests tests/single-gpu-non-square-tuner-tests.cu)
add_custom_target(gen-non-square-tuner-test-kernels
    ALL
    COMMAND python ${GEN_TUNER_KERNELS_PY} -same-factors 4 8,16 -same-factors 3 32,16 -same-factors 3 128,32
)
target_include_directories(single-gpu-non-square-tuner-tests PUBLIC ${TESTS_INCLUDES})
target_link_libraries(single-gpu-non-square-tuner-tests ${TESTS_LIBS})

add_executable(single-gpu-distinct-shapes tests/single-gpu-distinct-shapes.cu)
add_custom_target(gen-single-gpu-distinct-shapes
    ALL
    COMMAND python ${GEN_TUNER_KERNELS_PY} -distinct-factors 3 8,16 16,8 8,32
)
target_include_directories(single-gpu-distinct-shapes PUBLIC ${TESTS_INCLUDES})
target_link_libraries(single-gpu-distinct-shapes ${TESTS_LIBS})

add_executable(single-gpu-odd-shapes tests/single-gpu-odd-shapes.cu)
add_custom_target(gen-single-gpu-odd-shapes
    ALL
    COMMAND python ${GEN_TUNER_KERNELS_PY} -same-factors 2 31,16 -same-factors 2 16,31 -same-factors 4 31,31
)
target_include_directories(single-gpu-odd-shapes PUBLIC ${TESTS_INCLUDES})
target_link_libraries(single-gpu-odd-shapes ${TESTS_LIBS})

add_executable(multi-gpu-no-fusion-tests tests/multi-gpu-no-fusion-tests.cu)
add_custom_target(gen-multi-gpu-tests-kernel
    ALL
    COMMAND python ${GEN_TUNER_KERNELS_PY} -same-factors 4 64,64 -same-factors 4 128,128 -dist-kernels -match-configs 128,64,64,64,4096,2,1,0,2,16 128,64,64,64,4096,2,1,1,2,16 128,128,128,128,8192,1,1,0,2,32 128,128,128,128,8192,1,1,1,2,32
)
target_include_directories(multi-gpu-no-fusion-tests PUBLIC ${TESTS_INCLUDES})
target_link_libraries(multi-gpu-no-fusion-tests ${TESTS_LIBS})

add_executable(multi-gpu-tuner-tests tests/multi-gpu-tuner-tests.cu)
add_custom_target(gen-multi-gpu-tuner-kernels
    ALL
    COMMAND python ${GEN_TUNER_KERNELS_PY} -same-factors 5 16,16 -dist-kernels
)
target_include_directories(multi-gpu-tuner-tests PUBLIC ${TESTS_INCLUDES})
target_link_libraries(multi-gpu-tuner-tests ${TESTS_LIBS})

add_executable(multi-gpu-no-fusion-non-square-tests tests/multi-gpu-no-fusion-non-square-tests.cu)
add_custom_target(gen-multi-gpu-no-fusion-non-square-tests-kernel
    ALL
    COMMAND python ${GEN_TUNER_KERNELS_PY} -same-factors 5 8,32 -same-factors 4 64,16 -dist-kernels
)
target_include_directories(multi-gpu-no-fusion-non-square-tests PUBLIC ${TESTS_INCLUDES})
target_link_libraries(multi-gpu-no-fusion-non-square-tests ${TESTS_LIBS})

add_executable(multi-gpu-distinct-shapes tests/multi-gpu-distinct-shapes.cu)
add_custom_target(gen-multi-gpu-distinct-shapes
    ALL
    COMMAND python ${GEN_TUNER_KERNELS_PY} -distinct-factors 3 8,16 32,8 16,8 -match-configs 64,16,8,16,256 32,8,32,8,256 128,8,16,8,256 -dist-kernels
)
target_include_directories(multi-gpu-distinct-shapes PUBLIC ${TESTS_INCLUDES})
target_link_libraries(multi-gpu-distinct-shapes ${TESTS_LIBS})