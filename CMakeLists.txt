cmake_minimum_required(VERSION 3.20)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CUDA_STANDARD_REQUIRED True)
project(FastKron 
  VERSION 1.0
  LANGUAGES CUDA CXX)
find_package(CUDA 12.0 REQUIRED)
add_executable(kron tests/main.cu)
add_subdirectory(src)
target_link_libraries(FastKron ${CUDA_LIBRARIES})
target_link_libraries(kron PUBLIC FastKron)