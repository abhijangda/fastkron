set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CUDA_STANDARD_REQUIRED True)

include(device/kron-kernels/CMakeLists.txt)
set(DEVICE_HEADER_FILES device/kron-kernels/kernel_decl.inc device/device_functions.cuh device/kernel_info.h device/kernel.cuh device/otherkernels.cuh device/params.h)
set(HEADER_FILES env.h autotuner.h fastkron.h handle.h kernel_defs.cuh kernel_invoker.h kmmalgo.h sliced_mul_shape.h thread_pool.h utils.h)
set(SOURCES fastkron.cu autotuner.cpp distrib_handle.cu env.cpp handle.cu kernel_invoker.cu kmmalgo.cu sliced_mul_shape.cpp)
cuda_add_library(FastKron SHARED
    ${SOURCES}
    ${HEADER_FILES}
    ${DEVICE_HEADER_FILES}
    ${CUDA_KERNELS}
    OPTIONS -std=c++17 -gencode arch=compute_80,code=sm_80
)

target_include_directories(FastKron PRIVATE ./)