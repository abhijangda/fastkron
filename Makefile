ANYOPTION=-IAnyOption/ AnyOption/anyoption.cpp
TEST_INCLUDE_DIRS = -Igoogletest/googletest/include/  -Lgoogletest/build/lib/ -Isrc/ -Isrc/apps/ -Isrc/tests
GOOGLE_TEST_MAIN = googletest/googletest/src/gtest_main.cc
TEST_LFLAGS = -lgtest -lpthread
NVCC=/usr/local/cuda/bin/nvcc
ARCH_CODE_FLAGS=-gencode arch=compute_70,code=sm_70

all: kron

libKron.so: src/kron.cu src/kron.h src/kernel.cuh src/kernel_decl.inc src/kernel_defs.cuh src/device_functions.cuh
	$(NVCC) -Xcompiler=-fPIC,-shared,-fopenmp,-O3 $< -Isrc/ -o $@ -Xptxas=-v,-O3 $(ARCH_CODE_FLAGS) -lnccl

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
	python3 src/gen_tuner_kernels.py -same-factors 4 16 16 -same-factors 3 64 64 64

single-gpu-tuner-tests: gen-tuner-kernels tests/single-gpu-tuner-tests.cu libKron.so tests/testBase.h
	$(NVCC) tests/$@.cu $(TEST_INCLUDE_DIRS) $(TEST_LFLAGS) $(GOOGLE_TEST_MAIN) $(ARCH_CODE_FLAGS) -O3 -Xcompiler=-fopenmp,-O3,-Wall -L. -lKron -o $@

run-single-gpu-tuner-tests: single-gpu-tuner-tests
	LD_LIBRARY_PATH=./: ./single-gpu-tuner-tests

#Test for Non Square factors
gen-non-square-tuner-test-kernels: src/gen_tuner_kernels.py
	python3 src/gen_tuner_kernels.py -same-factors 4 8 16 -same-factors 5 8 16 -same-factors 3 32 16 -same-factors 3 32 64

single-gpu-non-square-tuner-tests: gen-non-square-tuner-test-kernels tests/single-gpu-non-square-tuner-tests.cu libKron.so tests/testBase.h
	$(NVCC) tests/$@.cu $(TEST_INCLUDE_DIRS) $(TEST_LFLAGS) $(GOOGLE_TEST_MAIN) $(ARCH_CODE_FLAGS) -O3 -Xcompiler=-fopenmp,-O3,-Wall -L. -lKron -o $@

run-single-gpu-non-square-tuner-tests: single-gpu-non-square-tuner-tests
	LD_LIBRARY_PATH=./: ./single-gpu-non-square-tuner-tests

#Multi GPU Tests
copy-multi-gpu-tests-kernel: tests/multi-gpu-tests-kernel_decl.inc
	cp tests/multi-gpu-tests-kernel_decl.inc src/kernel_decl.inc

multi-gpu-no-fusion-tests: copy-multi-gpu-tests-kernel libKron.so tests/testBase.h tests/multi-gpu-no-fusion-tests.cu
	$(NVCC) tests/$@.cu $(TEST_INCLUDE_DIRS) $(TEST_LFLAGS) $(GOOGLE_TEST_MAIN) $(ARCH_CODE_FLAGS) -O3 -Xcompiler=-fopenmp,-O3,-Wall -L. -lKron -o $@

run-multi-gpu-nccl-no-fusion-tests: multi-gpu-no-fusion-tests
	LD_LIBRARY_PATH=./: DIST_COMM=NCCL ./multi-gpu-no-fusion-tests

run-multi-gpu-p2p-no-fusion-tests: multi-gpu-no-fusion-tests
	LD_LIBRARY_PATH=./: DIST_COMM=P2P ./multi-gpu-no-fusion-tests


#Run all tests
run-all-single-gpu-tests: run-single-gpu-fusion-tests run-single-gpu-no-fusion-tests run-single-gpu-tuner-tests run-single-gpu-non-square-tuner-tests

run-all-multi-gpu-tests: run-multi-gpu-p2p-no-fusion-tests run-multi-gpu-nccl-no-fusion-tests

clean:
	rm -rf kron libKron.so
