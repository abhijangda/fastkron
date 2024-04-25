# X86 tests
set(X86_TEST_DEFINE -D TEST_BACKEND_X86)

add_custom_target(gen-x86-kernels 
  COMMAND python3 ${GEN_TUNER_KERNELS_PY} -backend x86 -archs avx avx512 -same-factors 2 128,128 -same-factors 2 64,64 -same-factors 3 32,32 -same-factors 5 16,16 -same-factors 7 8,8 -same-factors 10 4,4 -same-factors 20 2,2 -opX N -opF N -types float -match-configs-file ${SRC}/kernels/best-kernels/x86-avx-kernels)

add_executable(x86-cpu-NN ${TEST_SRC}/general-kernel-tests.cpp)
target_include_directories(x86-cpu-NN PRIVATE ${TESTS_INCLUDES})
target_link_libraries(x86-cpu-NN ${TESTS_LIBS})
target_compile_definitions(x86-cpu-NN PRIVATE ${X86_TEST_DEFINE})

add_executable(x86-cpu-TT ${TEST_SRC}/general-kernel-tests-TT.cpp)
target_include_directories(x86-cpu-TT PRIVATE ${TESTS_INCLUDES})
target_link_libraries(x86-cpu-TT ${TESTS_LIBS})
target_compile_definitions(x86-cpu-TT PRIVATE ${X86_TEST_DEFINE})