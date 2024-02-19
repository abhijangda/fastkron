find_package (Python3)
set(GEN_TUNER_KERNELS_PY ${SRC}/gen_tuner_kernels.py)

ADD_SUBDIRECTORY(googletest EXCLUDE_FROM_ALL)
enable_testing()
include_directories(${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})

set(TESTS_INCLUDES ${SRC}/ tests)
set(TESTS_LIBS GTest::gtest_main FastKron nccl)

add_executable(single-gpu-no-fusion-tests single-gpu-no-fusion-tests.cpp)
add_executable(single-gpu-fusion-tests single-gpu-fusion-tests.cpp)
add_custom_target(gen-single-gpu-kernels
    COMMAND python ${GEN_TUNER_KERNELS_PY} -backend cuda -same-factors 20 2,2 -same-factors 10 4,4 -same-factors 8 8,8 -same-factors 6 16,16 -same-factors 5 32,32 -same-factors 4 64,64 -same-factors 3 128,128 -opX N -opF N -match-configs-file ${CMAKE_CURRENT_SOURCE_DIR}/single-gpu-kernel-decls.in
)
target_include_directories(single-gpu-no-fusion-tests PRIVATE ${TESTS_INCLUDES})
target_link_libraries(single-gpu-no-fusion-tests ${TESTS_LIBS})

target_include_directories(single-gpu-fusion-tests PRIVATE ${TESTS_INCLUDES})
target_link_libraries(single-gpu-fusion-tests ${TESTS_LIBS})

add_executable(single-gpu-TT-tests single-gpu-TT-tests.cpp)
add_custom_target(gen-single-gpu-TT-kernels
    COMMAND python ${GEN_TUNER_KERNELS_PY}  -backend cuda -same-factors 8 8,8 -same-factors 6 16,16 -same-factors 5 32,32 -same-factors 4 64,64 -same-factors 3 128,128 -opX T -opF T -match-configs-file ${CMAKE_CURRENT_SOURCE_DIR}/single-gpu-kernel-decls.in
)
target_include_directories(single-gpu-TT-tests PRIVATE ${TESTS_INCLUDES})
target_link_libraries(single-gpu-TT-tests ${TESTS_LIBS})

add_executable(single-gpu-tuner-tests single-gpu-tuner-tests.cpp)
add_custom_target(gen-tuner-kernels
    COMMAND python ${GEN_TUNER_KERNELS_PY} -backend cuda -same-factors 4 16,16 -same-factors 3 64,64 -opX N -opF N
)
target_include_directories(single-gpu-tuner-tests PRIVATE ${TESTS_INCLUDES})
target_link_libraries(single-gpu-tuner-tests ${TESTS_LIBS})

add_executable(single-gpu-non-square-tuner-tests single-gpu-non-square-tuner-tests.cpp)
add_custom_target(gen-non-square-tuner-test-kernels
    COMMAND python ${GEN_TUNER_KERNELS_PY} -backend cuda -same-factors 4 8,16 -same-factors 3 32,16 -same-factors 3 128,32 -opX N -opF N
)
target_include_directories(single-gpu-non-square-tuner-tests PRIVATE ${TESTS_INCLUDES})
target_link_libraries(single-gpu-non-square-tuner-tests ${TESTS_LIBS})

add_executable(single-gpu-non-square-TT-tests single-gpu-non-square-TT-tests.cpp)
add_custom_target(gen-single-gpu-non-square-TT-kernels
    COMMAND python ${GEN_TUNER_KERNELS_PY} -backend cuda -same-factors 4 8,16 -same-factors 3 32,16 -same-factors 3 128,32 -opX T -opF T
)
target_include_directories(single-gpu-non-square-TT-tests PRIVATE ${TESTS_INCLUDES})
target_link_libraries(single-gpu-non-square-TT-tests ${TESTS_LIBS})

add_executable(single-gpu-distinct-shapes single-gpu-distinct-shapes.cpp)
add_custom_target(gen-single-gpu-distinct-shapes
    COMMAND python ${GEN_TUNER_KERNELS_PY} -backend cuda -distinct-factors 3 8,16 16,8 8,32 -opX N -opF N
)
target_include_directories(single-gpu-distinct-shapes PRIVATE ${TESTS_INCLUDES})
target_link_libraries(single-gpu-distinct-shapes ${TESTS_LIBS})

add_executable(single-gpu-odd-shapes single-gpu-odd-shapes.cpp)
add_custom_target(gen-single-gpu-odd-shapes
    COMMAND python ${GEN_TUNER_KERNELS_PY} -backend cuda -same-factors 2 31,16 -same-factors 2 16,31 -same-factors 4 31,31 -opX N -opF N
)
target_include_directories(single-gpu-odd-shapes PRIVATE ${TESTS_INCLUDES})
target_link_libraries(single-gpu-odd-shapes ${TESTS_LIBS})

add_executable(multi-gpu-no-fusion-tests multi-gpu-no-fusion-tests.cpp)
add_custom_target(gen-multi-gpu-tests-kernel
    COMMAND python ${GEN_TUNER_KERNELS_PY} -backend cuda -same-factors 4 64,64 -same-factors 4 128,128 -dist-kernels -match-configs 128,64,64,64,4096,2,1,0,2,16 128,64,64,64,4096,2,1,1,2,16 128,128,128,128,8192,1,1,0,2,32 128,128,128,128,8192,1,1,1,2,32 -opX N -opF N
)
target_include_directories(multi-gpu-no-fusion-tests PRIVATE ${TESTS_INCLUDES})
target_link_libraries(multi-gpu-no-fusion-tests ${TESTS_LIBS})

add_executable(multi-gpu-tuner-tests multi-gpu-tuner-tests.cpp)
add_custom_target(gen-multi-gpu-tuner-kernels
    COMMAND python ${GEN_TUNER_KERNELS_PY} -backend cuda -same-factors 5 16,16 -dist-kernels -opX N -opF N
)
target_include_directories(multi-gpu-tuner-tests PRIVATE ${TESTS_INCLUDES})
target_link_libraries(multi-gpu-tuner-tests ${TESTS_LIBS})

add_executable(multi-gpu-no-fusion-non-square-tests multi-gpu-no-fusion-non-square-tests.cpp)
add_custom_target(gen-multi-gpu-no-fusion-non-square-tests-kernel
    COMMAND python ${GEN_TUNER_KERNELS_PY} -backend cuda -same-factors 5 8,32 -same-factors 4 64,16 -dist-kernels -opX N -opF N
)
target_include_directories(multi-gpu-no-fusion-non-square-tests PRIVATE ${TESTS_INCLUDES})
target_link_libraries(multi-gpu-no-fusion-non-square-tests ${TESTS_LIBS})

add_executable(multi-gpu-distinct-shapes multi-gpu-distinct-shapes.cpp)
add_custom_target(gen-multi-gpu-distinct-shapes
    COMMAND python ${GEN_TUNER_KERNELS_PY} -backend cuda -distinct-factors 3 8,16 32,8 16,8 -match-configs 64,16,8,16,256 32,8,32,8,256 128,8,16,8,256 -dist-kernels -opX N -opF N
)
target_include_directories(multi-gpu-distinct-shapes PRIVATE ${TESTS_INCLUDES})
target_link_libraries(multi-gpu-distinct-shapes ${TESTS_LIBS})

add_subdirectory(benchmarks)