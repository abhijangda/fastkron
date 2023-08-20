ANYOPTION=-IAnyOption/ AnyOption/anyoption.cpp
TEST_INCLUDE_DIRS = -Igoogletest/googletest/include/  -Lgoogletest/build/lib/ -Isrc/ -Isrc/apps/ -Isrc/tests
GOOGLE_TEST_MAIN = googletest/googletest/src/gtest_main.cc
TEST_LFLAGS = -lgtest -lpthread

all: kron

libKron.so: src/kron.cu src/kron.h src/kron_device.cu src/kernel_decl.inc
	nvcc -Xcompiler=-fPIC,-shared,-fopenmp,-O3 $< -Isrc/ -o $@ -Xptxas=-v -gencode arch=compute_70,code=sm_70

kron: tests/main.cu libKron.so 
	nvcc $< -Xcompiler=-fopenmp,-O3,-Wall -Isrc/ $(ANYOPTION) -L. -lKron -o $@

single-gpu-tests: tests/single-gpu-tests.cu libKron.so tests/testBase.h
	nvcc $< $(TEST_INCLUDE_DIRS) $(TEST_LFLAGS) $(GOOGLE_TEST_MAIN) $(ARCH_CODE_FLAGS) -O3 -Xcompiler=-fopenmp,-O3,-Wall -L. -lKron -o $@

run-tests: single-gpu-tests
	LD_LIBRARY_PATH=./: ./single-gpu-tests

clean:
	rm -rf kron libKron.so
