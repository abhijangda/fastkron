# CUDA tests
set(CUDA_TEST_DEFINE -D TEST_BACKEND_CUDA)
set(CUDA_TESTS_INCLUDES ${TESTS_INCLUDES})

set_source_files_properties(${TEST_SRC}/general-kernel-tests.cpp PROPERTIES LANGUAGE CUDA)
set_source_files_properties(${TEST_SRC}/general-kernel-tests-TT.cpp PROPERTIES LANGUAGE CUDA)
set_source_files_properties(${TEST_SRC}/multi-cuda-no-fusion-tests.cpp PROPERTIES LANGUAGE CUDA)
set_source_files_properties(${TEST_SRC}/multi-cuda-tuner-tests.cpp PROPERTIES LANGUAGE CUDA)
set_source_files_properties(${TEST_SRC}/multi-cuda-no-fusion-non-square-tests.cpp PROPERTIES LANGUAGE CUDA)
set_source_files_properties(${TEST_SRC}/multi-cuda-distinct-shapes.cpp PROPERTIES LANGUAGE CUDA)

add_custom_target(gen-single-gpu-kernels 
  COMMAND python ${GEN_TUNER_KERNELS_PY} -backend cuda -same-factors 2 128,128 -same-factors 2 64,64 -same-factors 3 32,32 -same-factors 5 16,16 -same-factors 7 8,8 -same-factors 10 4,4 -same-factors 20 2,2 -opX N T -opF N T -match-configs-file ${SRC}/kernels/best-kernels/a100-kernels)
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
    COMMAND python ${GEN_TUNER_KERNELS_PY} -backend cuda -same-factors 4 64,64 -same-factors 4 128,128 -dist-kernels -match-configs 128,64,64,64,4096,2,1,0,2,16 128,64,64,64,4096,2,1,1,2,16 128,128,128,128,8192,1,1,0,2,32 128,128,128,128,8192,1,1,1,2,32 -opX N -opF N
)
target_include_directories(multi-cuda-no-fusion-tests PRIVATE ${TESTS_INCLUDES})
target_link_libraries(multi-cuda-no-fusion-tests ${TESTS_LIBS})

add_executable(multi-cuda-tuner-tests ${TEST_SRC}/multi-cuda-tuner-tests.cpp)
add_custom_target(gen-multi-cuda-tuner-kernels
    COMMAND python ${GEN_TUNER_KERNELS_PY} -backend cuda -same-factors 5 16,16 -dist-kernels -opX N -opF N
)
target_include_directories(multi-cuda-tuner-tests PRIVATE ${TESTS_INCLUDES})
target_link_libraries(multi-cuda-tuner-tests ${TESTS_LIBS})

add_executable(multi-cuda-no-fusion-non-square-tests ${TEST_SRC}/multi-cuda-no-fusion-non-square-tests.cpp)
add_custom_target(gen-multi-cuda-no-fusion-non-square-tests-kernel
    COMMAND python ${GEN_TUNER_KERNELS_PY} -backend cuda -same-factors 5 8,32 -same-factors 4 64,16 -dist-kernels -opX N -opF N
)
target_include_directories(multi-cuda-no-fusion-non-square-tests PRIVATE ${TESTS_INCLUDES})
target_link_libraries(multi-cuda-no-fusion-non-square-tests ${TESTS_LIBS})

add_executable(multi-cuda-distinct-shapes ${TEST_SRC}/multi-cuda-distinct-shapes.cpp)
add_custom_target(gen-multi-cuda-distinct-shapes
    COMMAND python ${GEN_TUNER_KERNELS_PY} -backend cuda -distinct-factors 3 8,16 32,8 16,8 -match-configs 64,16,8,16,256 64,8,32,8,256 128,8,16,8,256 -dist-kernels -opX N -opF N
)
target_include_directories(multi-cuda-distinct-shapes PRIVATE ${TESTS_INCLUDES})
target_link_libraries(multi-cuda-distinct-shapes ${TESTS_LIBS})
