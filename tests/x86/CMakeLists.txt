# X86 tests
set(X86_TEST_DEFINE -D TEST_BACKEND_X86)

add_executable(x86-cpu-NN ${TEST_SRC}/general-kernel-tests.cpp)
target_include_directories(x86-cpu-NN PRIVATE ${TESTS_INCLUDES})
target_link_libraries(x86-cpu-NN ${TESTS_LIBS})
target_compile_definitions(x86-cpu-NN PRIVATE ${X86_TEST_DEFINE})

add_executable(x86-cpu-TT ${TEST_SRC}/general-kernel-tests-TT.cpp)
target_include_directories(x86-cpu-TT PRIVATE ${TESTS_INCLUDES})
target_link_libraries(x86-cpu-TT ${TESTS_LIBS})
target_compile_definitions(x86-cpu-TT PRIVATE ${X86_TEST_DEFINE})