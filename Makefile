ANYOPTION=-IAnyOption/ AnyOption/anyoption.cpp
TEST_INCLUDE_DIRS = -Igoogletest/googletest/include/  -Lgoogletest/build/lib/ -Isrc/ -Isrc/apps/ -Isrc/tests
GOOGLE_TEST_MAIN = googletest/googletest/src/gtest_main.cc
TEST_LFLAGS = -lgtest -lpthread

all: kron

libKron.so: src/kron.cu src/kron.h src/kernel.cuh src/kernel_decl.inc src/kernel_defs.cuh src/device_functions.cuh
	nvcc -Xcompiler=-fPIC,-shared,-fopenmp,-O3 $< -Isrc/ -o $@ -Xptxas=-v,-O3 -gencode arch=compute_70,code=sm_70

kron: tests/main.cu libKron.so tests/testBase.h
	nvcc $< -Xcompiler=-fopenmp,-O3,-Wall -Isrc/ $(ANYOPTION) -L. -lKron -o $@

single-gpu-no-fusion-tests: tests/single-gpu-no-fusion-tests.cu libKron.so tests/testBase.h
	nvcc $< $(TEST_INCLUDE_DIRS) $(TEST_LFLAGS) $(GOOGLE_TEST_MAIN) $(ARCH_CODE_FLAGS) -O3 -Xcompiler=-fopenmp,-O3,-Wall -L. -lKron -o $@

single-gpu-tests: tests/single-gpu-fusion-tests.cu libKron.so tests/testBase.h
	nvcc $< $(TEST_INCLUDE_DIRS) $(TEST_LFLAGS) $(GOOGLE_TEST_MAIN) $(ARCH_CODE_FLAGS) -O3 -Xcompiler=-fopenmp,-O3,-Wall -L. -lKron -o $@

run-tests: single-gpu-tests single-gpu-no-fusion-tests
	LD_LIBRARY_PATH=./: ./single-gpu-tests ; LD_LIBRARY_PATH=./: ./single-gpu-no-fusion-tests

clean:
	rm -rf kron libKron.so
