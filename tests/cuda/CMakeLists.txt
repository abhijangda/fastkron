# CUDA tests
set(CUDA_TEST_DEFINE -D TEST_BACKEND_CUDA)
set(CUDA_TESTS_INCLUDES ${TESTS_INCLUDES})

set_source_files_properties(${TEST_SRC}/general-kernel-tests.cpp PROPERTIES LANGUAGE CUDA)
set_source_files_properties(${TEST_SRC}/general-kernel-tests-TT.cpp PROPERTIES LANGUAGE CUDA)

add_executable(general-kernel-tests ${TEST_SRC}/general-kernel-tests.cpp)
target_include_directories(general-kernel-tests PRIVATE ${TESTS_INCLUDES})
target_link_libraries(general-kernel-tests ${TESTS_LIBS})
target_compile_definitions(general-kernel-tests PRIVATE ${CUDA_TEST_DEFINE})

add_executable(general-kernel-tests-TT ${TEST_SRC}/general-kernel-tests-TT.cpp)
target_include_directories(general-kernel-tests-TT PRIVATE ${TESTS_INCLUDES})
target_link_libraries(general-kernel-tests-TT ${TESTS_LIBS})
target_compile_definitions(general-kernel-tests-TT PRIVATE ${CUDA_TEST_DEFINE})
