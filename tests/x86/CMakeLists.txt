# X86 tests
set(X86_TEST_DEFINE -D TEST_BACKEND_X86)
set(BACKEND x86)

add_executable(x86-no-fusion-tests ${TEST_SRC}/no-fusion-tests.cpp)
add_executable(x86-fusion-tests    ${TEST_SRC}/fusion-tests.cpp)
add_custom_target(gen-x86-kernels
    COMMAND python ${GEN_TUNER_KERNELS_PY} -backend ${BACKEND} -same-factors 20 2,2 -same-factors 10 4,4 -same-factors 8 8,8 -same-factors 6 16,16 -same-factors 5 32,32 -same-factors 4 64,64 -same-factors 3 128,128 -opX N -opF N
)
target_include_directories(x86-no-fusion-tests PRIVATE ${TESTS_INCLUDES})
target_link_libraries(x86-no-fusion-tests ${TESTS_LIBS})
target_compile_definitions(x86-no-fusion-tests PRIVATE ${X86_TEST_DEFINE})

target_include_directories(x86-fusion-tests PRIVATE ${TESTS_INCLUDES})
target_link_libraries(x86-fusion-tests ${TESTS_LIBS})
target_compile_definitions(x86-fusion-tests PRIVATE ${X86_TEST_DEFINE})

add_executable(x86-TT-tests ${TEST_SRC}/TT-tests.cpp)
add_custom_target(gen-x86-TT-kernels
    COMMAND python ${GEN_TUNER_KERNELS_PY} -backend ${BACKEND} -same-factors 8 8,8 -same-factors 6 16,16 -same-factors 5 32,32 -same-factors 4 64,64 -same-factors 3 128,128 -opX T -opF T
)
target_include_directories(x86-TT-tests PRIVATE ${TESTS_INCLUDES})
target_link_libraries(x86-TT-tests ${TESTS_LIBS})
target_compile_definitions(x86-TT-tests PRIVATE ${CUDA_TEST_DEFINE})

add_executable(x86-non-square-tests ${TEST_SRC}/non-square-tuner-tests.cpp)
add_custom_target(gen-x86-non-square-kernels
    COMMAND python ${GEN_TUNER_KERNELS_PY} -backend ${BACKEND} -same-factors 4 8,16 -same-factors 3 32,16 -same-factors 3 128,32 -opX N -opF N
)
target_include_directories(x86-non-square-tests PRIVATE ${TESTS_INCLUDES})
target_link_libraries(x86-non-square-tests ${TESTS_LIBS})
target_compile_definitions(x86-non-square-tests PRIVATE ${X86_TEST_DEFINE})

add_executable(x86-non-square-TT-tests ${TEST_SRC}/non-square-TT-tests.cpp)
add_custom_target(gen-x86-non-square-TT-kernels
    COMMAND python ${GEN_TUNER_KERNELS_PY} -backend ${BACKEND} -same-factors 4 8,16 -same-factors 3 32,16 -same-factors 3 128,32 -opX T -opF T
)
target_include_directories(x86-non-square-TT-tests PRIVATE ${TESTS_INCLUDES})
target_link_libraries(x86-non-square-TT-tests ${TESTS_LIBS})
target_compile_definitions(x86-non-square-TT-tests PRIVATE ${X86_TEST_DEFINE})

add_executable(x86-distinct-shapes ${TEST_SRC}/distinct-shapes.cpp)
add_custom_target(gen-x86-distinct-shapes
    COMMAND python ${GEN_TUNER_KERNELS_PY} -backend ${BACKEND} -distinct-factors 3 8,16 16,8 8,32 -opX N -opF N
)
target_include_directories(x86-distinct-shapes PRIVATE ${TESTS_INCLUDES})
target_link_libraries(x86-distinct-shapes ${TESTS_LIBS})
target_compile_definitions(x86-distinct-shapes PRIVATE ${X86_TEST_DEFINE})

add_executable(x86-odd-shapes ${TEST_SRC}/odd-shapes.cpp)
add_custom_target(gen-x86-odd-shapes
    COMMAND python ${GEN_TUNER_KERNELS_PY} -backend ${BACKEND} -same-factors 2 31,16 -same-factors 2 16,31 -same-factors 4 31,31 -opX N -opF N
)
target_include_directories(x86-odd-shapes PRIVATE ${TESTS_INCLUDES})
target_link_libraries(x86-odd-shapes ${TESTS_LIBS})
target_compile_definitions(x86-odd-shapes PRIVATE ${CUDA_TEST_DEFINE})