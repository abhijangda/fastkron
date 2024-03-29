# CUDA tests
set(CUDA_TEST_DEFINE -D TEST_BACKEND_CUDA)
set(CUDA_TESTS_INCLUDES ${TESTS_INCLUDES})

if (ENABLE_CUDA)
    set_source_files_properties(${TEST_SRC}/no-fusion-tests.cpp PROPERTIES LANGUAGE CUDA)
    set_source_files_properties(${TEST_SRC}/fusion-tests.cpp PROPERTIES LANGUAGE CUDA)
    set_source_files_properties(${TEST_SRC}/TT-tests.cpp PROPERTIES LANGUAGE CUDA)
    set_source_files_properties(${TEST_SRC}/tuner-tests.cpp PROPERTIES LANGUAGE CUDA)
    set_source_files_properties(${TEST_SRC}/non-square-tuner-tests.cpp PROPERTIES LANGUAGE CUDA)
    set_source_files_properties(${TEST_SRC}/non-square-TT-tests.cpp PROPERTIES LANGUAGE CUDA)
    set_source_files_properties(${TEST_SRC}/distinct-shapes.cpp PROPERTIES LANGUAGE CUDA)
    set_source_files_properties(${TEST_SRC}/odd-shapes.cpp PROPERTIES LANGUAGE CUDA)
    set_source_files_properties(${TEST_SRC}/no-fusion-tests.cpp PROPERTIES LANGUAGE CUDA)
    set_source_files_properties(${TEST_SRC}/multi-cuda-no-fusion-tests.cpp PROPERTIES LANGUAGE CUDA)
    set_source_files_properties(${TEST_SRC}/multi-cuda-tuner-tests.cpp PROPERTIES LANGUAGE CUDA)
    set_source_files_properties(${TEST_SRC}/multi-cuda-no-fusion-non-square-tests.cpp PROPERTIES LANGUAGE CUDA)
    set_source_files_properties(${TEST_SRC}/multi-cuda-distinct-shapes.cpp PROPERTIES LANGUAGE CUDA)
endif()

add_executable(single-cuda-no-fusion-tests ${TEST_SRC}/no-fusion-tests.cpp)
add_executable(single-cuda-fusion-tests    ${TEST_SRC}/fusion-tests.cpp)
add_custom_target(gen-single-cuda-kernels
    COMMAND python ${GEN_TUNER_KERNELS_PY} -backend cuda -same-factors 20 2,2 -same-factors 10 4,4 -same-factors 8 8,8 -same-factors 6 16,16 -same-factors 5 32,32 -same-factors 4 64,64 -same-factors 3 128,128 -opX N -opF N -match-configs-file ${CMAKE_CURRENT_SOURCE_DIR}/single-cuda-kernel-decls.in
)
target_include_directories(single-cuda-no-fusion-tests PRIVATE ${TESTS_INCLUDES})
target_link_libraries(single-cuda-no-fusion-tests ${TESTS_LIBS})
target_compile_definitions(single-cuda-no-fusion-tests PRIVATE ${CUDA_TEST_DEFINE})

target_include_directories(single-cuda-fusion-tests PRIVATE ${TESTS_INCLUDES})
target_link_libraries(single-cuda-fusion-tests ${TESTS_LIBS})
target_compile_definitions(single-cuda-fusion-tests PRIVATE ${CUDA_TEST_DEFINE})

add_executable(single-cuda-TT-tests ${TEST_SRC}/TT-tests.cpp)
add_custom_target(gen-single-cuda-TT-kernels
    COMMAND python ${GEN_TUNER_KERNELS_PY}  -backend cuda -same-factors 8 8,8 -same-factors 6 16,16 -same-factors 5 32,32 -same-factors 4 64,64 -same-factors 3 128,128 -opX T -opF T -match-configs-file ${CMAKE_CURRENT_SOURCE_DIR}/single-cuda-kernel-decls.in
)
target_include_directories(single-cuda-TT-tests PRIVATE ${TESTS_INCLUDES})
target_link_libraries(single-cuda-TT-tests ${TESTS_LIBS})
target_compile_definitions(single-cuda-TT-tests PRIVATE ${CUDA_TEST_DEFINE})

add_executable(single-cuda-tuner-tests ${TEST_SRC}/tuner-tests.cpp)
add_custom_target(gen-tuner-kernels
    COMMAND python ${GEN_TUNER_KERNELS_PY} -backend cuda -same-factors 4 16,16 -same-factors 3 64,64 -opX N -opF N
)
target_include_directories(single-cuda-tuner-tests PRIVATE ${TESTS_INCLUDES})
target_link_libraries(single-cuda-tuner-tests ${TESTS_LIBS})
target_compile_definitions(single-cuda-tuner-tests PRIVATE ${CUDA_TEST_DEFINE})

add_executable(single-cuda-non-square-tests ${TEST_SRC}/non-square-tuner-tests.cpp)
add_custom_target(gen-non-square-kernels
    COMMAND python ${GEN_TUNER_KERNELS_PY} -backend cuda -same-factors 4 8,16 -same-factors 3 32,16 -same-factors 3 128,32 -opX N -opF N
)
target_include_directories(single-cuda-non-square-tests PRIVATE ${TESTS_INCLUDES})
target_link_libraries(single-cuda-non-square-tests ${TESTS_LIBS})
target_compile_definitions(single-cuda-non-square-tests PRIVATE ${CUDA_TEST_DEFINE})

add_executable(single-cuda-non-square-TT-tests ${TEST_SRC}/non-square-TT-tests.cpp)
add_custom_target(gen-single-cuda-non-square-TT-kernels
    COMMAND python ${GEN_TUNER_KERNELS_PY} -backend cuda -same-factors 4 8,16 -same-factors 3 32,16 -same-factors 3 128,32 -opX T -opF T
)
target_include_directories(single-cuda-non-square-TT-tests PRIVATE ${TESTS_INCLUDES})
target_link_libraries(single-cuda-non-square-TT-tests ${TESTS_LIBS})
target_compile_definitions(single-cuda-non-square-TT-tests PRIVATE ${CUDA_TEST_DEFINE})

add_executable(single-cuda-distinct-shapes ${TEST_SRC}/distinct-shapes.cpp)
add_custom_target(gen-single-cuda-distinct-shapes
    COMMAND python ${GEN_TUNER_KERNELS_PY} -backend cuda -distinct-factors 3 8,16 16,8 8,32 -opX N -opF N
)
target_include_directories(single-cuda-distinct-shapes PRIVATE ${TESTS_INCLUDES})
target_link_libraries(single-cuda-distinct-shapes ${TESTS_LIBS})
target_compile_definitions(single-cuda-distinct-shapes PRIVATE ${CUDA_TEST_DEFINE})

add_executable(single-cuda-odd-shapes ${TEST_SRC}/odd-shapes.cpp)
add_custom_target(gen-single-cuda-odd-shapes
    COMMAND python ${GEN_TUNER_KERNELS_PY} -backend cuda -same-factors 2 31,16 -same-factors 2 16,31 -same-factors 4 31,31 -opX N -opF N
)
target_include_directories(single-cuda-odd-shapes PRIVATE ${TESTS_INCLUDES})
target_link_libraries(single-cuda-odd-shapes ${TESTS_LIBS})
target_compile_definitions(single-cuda-odd-shapes PRIVATE ${CUDA_TEST_DEFINE})

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

add_executable(general-kernel-tests ${TEST_SRC}/general-kernel-tests.cpp)
target_include_directories(general-kernel-tests PRIVATE ${TESTS_INCLUDES})
target_link_libraries(general-kernel-tests ${TESTS_LIBS})
target_compile_definitions(general-kernel-tests PRIVATE ${CUDA_TEST_DEFINE})

add_executable(general-kernel-tests-TT ${TEST_SRC}/general-kernel-tests-TT.cpp)
target_include_directories(general-kernel-tests-TT PRIVATE ${TESTS_INCLUDES})
target_link_libraries(general-kernel-tests-TT ${TESTS_LIBS})
target_compile_definitions(general-kernel-tests-TT PRIVATE ${CUDA_TEST_DEFINE})
