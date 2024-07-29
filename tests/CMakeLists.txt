find_package (Python3)
set(GEN_TUNER_KERNELS_PY ${SRC}/gen_tuner_kernels.py)

ADD_SUBDIRECTORY(googletest EXCLUDE_FROM_ALL)
enable_testing()
include_directories(${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})

set(TESTS_INCLUDES ${CMAKE_CURRENT_SOURCE_DIR} ${SRC})
set(TESTS_LIBS GTest::gtest_main FastKron)
if (ENABLE_MULTI_GPU AND ENABLE_CUDA)
  set(TESTS_LIBS ${TESTS_LIBS} nccl)
endif()

set(TEST_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src)

add_subdirectory(benchmarks)
if (ENABLE_CUDA)
    add_subdirectory(cuda)
endif()
if (ENABLE_X86)
    add_subdirectory(x86)
endif()
