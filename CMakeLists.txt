cmake_minimum_required(VERSION 3.20)
project(FastKron 
  VERSION 1.0
  LANGUAGES CUDA CXX)
find_package(CUDA 12.0 REQUIRED)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
add_executable(kron tests/main.cu)
cuda_add_library(FastKron SHARED
    src/fastkron.cu
    OPTIONS -gencode arch=compute_80,code=sm_80
)
target_link_libraries(FastKron ${CUDA_LIBRARIES})
target_link_libraries(kron PUBLIC FastKron)