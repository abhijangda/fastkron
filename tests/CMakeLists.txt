find_package (Python3)
set(GEN_TUNER_KERNELS_PY ${SRC}/gen_tuner_kernels.py)

ADD_SUBDIRECTORY(googletest EXCLUDE_FROM_ALL)
enable_testing()
include_directories(${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})
set(TEST_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src)

set(TESTS_INCLUDES ${CMAKE_CURRENT_SOURCE_DIR} ${SRC})
set(TESTS_LIBS GTest::gtest_main FastKron)
if (ENABLE_MULTI_GPU AND ENABLE_CUDA)
  set(TESTS_LIBS ${TESTS_LIBS} nccl)
endif()

add_subdirectory(benchmarks)

if (ENABLE_CUDA)
  add_subdirectory(cuda)
  add_custom_target(run-cuda-tests  
    COMMAND single-gpu-cuda-all 
    DEPENDS single-gpu-cuda-all)
endif()

if (ENABLE_X86)
  add_subdirectory(x86)
  add_custom_target(run-x86-tests
    COMMAND x86-cpu-all
    DEPENDS x86-cpu-all)
endif()