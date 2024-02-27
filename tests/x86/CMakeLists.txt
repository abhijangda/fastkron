# X86 tests
set(X86_TEST_DEFINE -D TEST_BACKEND_X86)

add_executable(x86-no-fusion-tests ${TEST_SRC}/no-fusion-tests.cpp)
add_executable(x86-fusion-tests    ${TEST_SRC}/fusion-tests.cpp)
add_custom_target(gen-x86-kernels
    COMMAND python ${GEN_TUNER_KERNELS_PY} -backend x86 -same-factors 20 2,2 -same-factors 10 4,4 -same-factors 8 8,8 -same-factors 6 16,16 -same-factors 5 32,32 -same-factors 4 64,64 -same-factors 3 128,128 -opX N -opF N
)
target_include_directories(x86-no-fusion-tests PRIVATE ${TESTS_INCLUDES})
target_link_libraries(x86-no-fusion-tests ${TESTS_LIBS})
target_compile_definitions(x86-no-fusion-tests PRIVATE ${X86_TEST_DEFINE})

target_include_directories(x86-fusion-tests PRIVATE ${TESTS_INCLUDES})
target_link_libraries(x86-fusion-tests ${TESTS_LIBS})
target_compile_definitions(x86-fusion-tests PRIVATE ${X86_TEST_DEFINE})