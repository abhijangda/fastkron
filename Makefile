ANYOPTION=-IAnyOption/ AnyOption/anyoption.cpp
TEST_INCLUDE_DIRS = -Igoogletest/googletest/include/  -Lgoogletest/build/lib/ -Isrc/ -Isrc/apps/ -Isrc/tests
GOOGLE_TEST_MAIN = googletest/googletest/src/gtest_main.cc
TEST_LFLAGS = -lgtest -lpthread
NVCC=/usr/local/cuda/bin/nvcc

all: kron

libKron.so: src/kron.cu src/kron.h src/kernel.cuh src/kernel_decl.inc src/kernel_defs.cuh src/device_functions.cuh
	$(NVCC) -Xcompiler=-fPIC,-shared,-fopenmp,-O3 $< -Isrc/ -o $@ -Xptxas=-v -gencode arch=compute_70,code=sm_70

kron: tests/main.cu libKron.so tests/testBase.h
	$(NVCC) $< -Xcompiler=-fopenmp,-O3,-Wall -Isrc/ $(ANYOPTION) -L. -lKron -o $@

#Make tests
gen-kernels: src/gen_kernels.py
	python3 src/gen_kernels.py

single-gpu-no-fusion-tests: gen-kernels tests/single-gpu-no-fusion-tests.cu libKron.so tests/testBase.h
	$(NVCC) tests/$@.cu $(TEST_INCLUDE_DIRS) $(TEST_LFLAGS) $(GOOGLE_TEST_MAIN) $(ARCH_CODE_FLAGS) -O3 -Xcompiler=-fopenmp,-O3,-Wall -L. -lKron -o $@

run-single-gpu-no-fusion-tests: single-gpu-no-fusion-tests
	LD_LIBRARY_PATH=./: ./single-gpu-no-fusion-tests

single-gpu-fusion-tests: gen-kernels tests/single-gpu-fusion-tests.cu libKron.so tests/testBase.h 
	$(NVCC) tests/$@.cu $(TEST_INCLUDE_DIRS) $(TEST_LFLAGS) $(GOOGLE_TEST_MAIN) $(ARCH_CODE_FLAGS) -O3 -Xcompiler=-fopenmp,-O3,-Wall -L. -lKron -o $@

run-single-gpu-fusion-tests: single-gpu-fusion-tests
	LD_LIBRARY_PATH=./: ./single-gpu-fusion-tests

#Tests for the autotuner
gen-tuner-kernels: src/gen_tuner_kernels.py
	python3 src/gen_tuner_kernels.py -m 1 2 1 -k 65536 32768 16384 -n 4 3 2 -p 16 32 128 -q 16 32 128

single-gpu-tuner-tests: gen-tuner-kernels tests/single-gpu-tuner-tests.cu libKron.so tests/testBase.h
	$(NVCC) tests/$@.cu $(TEST_INCLUDE_DIRS) $(TEST_LFLAGS) $(GOOGLE_TEST_MAIN) $(ARCH_CODE_FLAGS) -O3 -Xcompiler=-fopenmp,-O3,-Wall -L. -lKron -o $@

run-single-gpu-tuner-tests: single-gpu-tuner-tests
	LD_LIBRARY_PATH=./: ./single-gpu-tuner-tests

#Run all tests
run-tests: single-gpu-tests single-gpu-no-fusion-tests run-tuner-test

clean:
	rm -rf kron libKron.so
