set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CUDA_STANDARD_REQUIRED True)

set(HEADER_FILES env.h fastkron.h handle.h kernel_defs.cuh kernel_invoker.h kmmalgo.h sliced_mul_shape.h thread_pool.h utils.h)
set(SOURCES fastkron.cu distrib_handle.cu env.cpp handle.cu kernel_invoker.cu kmmalgo.cu sliced_mul_shape.cpp)
cuda_add_library(FastKron SHARED
    ${SOURCES}
    ${HEADER_FILES}
    OPTIONS -std=c++17 -gencode arch=compute_80,code=sm_80
)

target_include_directories(FastKron PRIVATE ./)