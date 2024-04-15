# CUDA tests
set(CUDA_TEST_DEFINE -D TEST_BACKEND_CUDA)
set(CUDA_TESTS_INCLUDES ${TESTS_INCLUDES})

if (ENABLE_CUDA)
    set_source_files_properties(${TEST_SRC}/general-kernel-tests.cpp PROPERTIES LANGUAGE CUDA)
    set_source_files_properties(${TEST_SRC}/general-kernel-tests-TT.cpp PROPERTIES LANGUAGE CUDA)
    set_source_files_properties(${TEST_SRC}/multi-cuda-no-fusion-tests.cpp PROPERTIES LANGUAGE CUDA)
    set_source_files_properties(${TEST_SRC}/multi-cuda-tuner-tests.cpp PROPERTIES LANGUAGE CUDA)
    set_source_files_properties(${TEST_SRC}/multi-cuda-no-fusion-non-square-tests.cpp PROPERTIES LANGUAGE CUDA)
    set_source_files_properties(${TEST_SRC}/multi-cuda-distinct-shapes.cpp PROPERTIES LANGUAGE CUDA)
endif()

add_custom_target(gen-single-gpu-kernels 
  COMMAND python3 ${GEN_TUNER_KERNELS_PY} -backend cuda -archs ampere volta -same-factors 2 128,128 -same-factors 2 64,64 -same-factors 3 32,32 -same-factors 5 16,16 -same-factors 7 8,8 -same-factors 10 4,4 -same-factors 20 2,2 -opX N T -opF N T -types float double -match-configs-file ${SRC}/kernels/best-kernels/a100-kernels)

add_executable(single-gpu-cuda ${TEST_SRC}/general-kernel-tests.cpp)
target_include_directories(single-gpu-cuda PRIVATE ${TESTS_INCLUDES})
target_link_libraries(single-gpu-cuda ${TESTS_LIBS})
target_compile_definitions(single-gpu-cuda PRIVATE ${CUDA_TEST_DEFINE})

add_executable(single-gpu-cuda-TT ${TEST_SRC}/general-kernel-tests-TT.cpp)
target_include_directories(single-gpu-cuda-TT PRIVATE ${TESTS_INCLUDES})
target_link_libraries(single-gpu-cuda-TT ${TESTS_LIBS})
target_compile_definitions(single-gpu-cuda-TT PRIVATE ${CUDA_TEST_DEFINE})

add_executable(multi-cuda-no-fusion-tests ${TEST_SRC}/multi-cuda-no-fusion-tests.cpp)
add_custom_target(gen-multi-cuda-tests-kernel
    COMMAND python3 ${GEN_TUNER_KERNELS_PY} -backend cuda -archs ampere volta -same-factors 4 64,64 -same-factors 4 128,128 -types float -dist-kernels -opX N -opF N -match-configs cuda_ampere_128_f_64x64_32x64_1_1x4096_1x4x8 cuda_ampere_128_f_128x128_32x128_1_1x8192_1x4x16 cuda_volta_256_f_128x128_32x128_1_2x4096_1x2x16 cuda_volta_256_f_64x64_32x64_1_2x2048_2x1x8
)
target_include_directories(multi-cuda-no-fusion-tests PRIVATE ${TESTS_INCLUDES})
target_link_libraries(multi-cuda-no-fusion-tests ${TESTS_LIBS})
target_compile_definitions(multi-cuda-no-fusion-tests PRIVATE ${CUDA_TEST_DEFINE})

add_executable(multi-cuda-tuner-tests ${TEST_SRC}/multi-cuda-tuner-tests.cpp)
add_custom_target(gen-multi-cuda-tuner-kernels
    COMMAND python3 ${GEN_TUNER_KERNELS_PY} -backend cuda -archs ampere -types float -same-factors 5 16,16 -dist-kernels -opX N -opF N -num-kernels 50
)
target_include_directories(multi-cuda-tuner-tests PRIVATE ${TESTS_INCLUDES})
target_link_libraries(multi-cuda-tuner-tests ${TESTS_LIBS})
target_compile_definitions(multi-cuda-tuner-tests PRIVATE ${CUDA_TEST_DEFINE})

add_executable(multi-cuda-no-fusion-non-square-tests ${TEST_SRC}/multi-cuda-no-fusion-non-square-tests.cpp)
add_custom_target(gen-multi-cuda-no-fusion-non-square-tests-kernel
    COMMAND python3 ${GEN_TUNER_KERNELS_PY} -backend cuda -archs ampere -types float -same-factors 5 8,32 -same-factors 4 64,16 -dist-kernels -opX N -opF N -num-kernels 50
)
target_include_directories(multi-cuda-no-fusion-non-square-tests PRIVATE ${TESTS_INCLUDES})
target_link_libraries(multi-cuda-no-fusion-non-square-tests ${TESTS_LIBS})
target_compile_definitions(multi-cuda-no-fusion-non-square-tests PRIVATE ${CUDA_TEST_DEFINE})

add_executable(multi-cuda-distinct-shapes ${TEST_SRC}/multi-cuda-distinct-shapes.cpp)
add_custom_target(gen-multi-cuda-distinct-shapes
    COMMAND python3 ${GEN_TUNER_KERNELS_PY} -backend cuda -archs ampere -types float -distinct-factors 3 8,16 32,8 16,8 -dist-kernels -opX N -opF N -num-kernels 50
)
target_include_directories(multi-cuda-distinct-shapes PRIVATE ${TESTS_INCLUDES})
target_link_libraries(multi-cuda-distinct-shapes ${TESTS_LIBS})
target_compile_definitions(multi-cuda-distinct-shapes PRIVATE ${CUDA_TEST_DEFINE})