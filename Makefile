ANYOPTION=-IAnyOption/ AnyOption/anyoption.cpp
TEST_INCLUDE_DIRS = -Igoogletest/googletest/include/  -Lgoogletest/build/lib/ -Isrc/ -Isrc/apps/ -Isrc/tests
GOOGLE_TEST_MAIN = googletest/googletest/src/gtest_main.cc
TEST_LFLAGS = -lgtest -lpthread
NVCC=/usr/local/cuda/bin/nvcc

all: kron

libKron.so: src/kron.cu src/kron.h src/kernel.cuh src/kernel_decl.inc src/kernel_defs.cuh src/device_functions.cuh
	$(NVCC) -Xcompiler=-fPIC,-shared,-fopenmp,-O3 $< -Isrc/ -o $@ -Xptxas=-v,-O3 -gencode arch=compute_70,code=sm_70

kron: tests/main.cu libKron.so tests/testBase.h
	$(NVCC) $< -Xcompiler=-fopenmp,-O3,-Wall -Isrc/ $(ANYOPTION) -L. -lKron -o $@

#Make tests
gen-kernels: src/gen_kernels.py
	python3 src/gen_kernels.py

single-gpu-no-fusion-tests: gen-kernels tests/single-gpu-no-fusion-tests.cu libKron.so tests/testBase.h
	$(NVCC) tests/$@.cu $(TEST_INCLUDE_DIRS) $(TEST_LFLAGS) $(GOOGLE_TEST_MAIN) $(ARCH_CODE_FLAGS) -O3 -Xcompiler=-fopenmp,-O3,-Wall -L. -lKron -o $@

run-single-gpu-no-fusion-tests: single-gpu-no-fusion-tests
	LD_LIBRARY_PATH=./: ./single-gpu-no-fusion-tests

single-gpu-tests: gen-kernels tests/single-gpu-fusion-tests.cu libKron.so tests/testBase.h 
	$(NVCC) tests/$@.cu $(TEST_INCLUDE_DIRS) $(TEST_LFLAGS) $(GOOGLE_TEST_MAIN) $(ARCH_CODE_FLAGS) -O3 -Xcompiler=-fopenmp,-O3,-Wall -L. -lKron -o $@

run-single-gpu-tests: single-gpu-tests
	LD_LIBRARY_PATH=./: ./single-gpu-tests

#Tests for the autotuner
gen-no-fusion-tuner-kernels: src/gen_tuner_kernels.py
	python3 src/gen_tuner_kernels.py -m 1 2 1 -k 65536 256 32768 -n 4 2 3 -p 16 16 32 -q 16 16 32

single-gpu-tuner-tests-no-fusion: gen-no-fusion-tuner-kernels tests/single-gpu-tuner-tests-no-fusion.cu libKron.so tests/testBase.h
	$(NVCC) tests/$@.cu $(TEST_INCLUDE_DIRS) $(TEST_LFLAGS) $(GOOGLE_TEST_MAIN) $(ARCH_CODE_FLAGS) -O3 -Xcompiler=-fopenmp,-O3,-Wall -L. -lKron -o $@

run-tuner-test-no-fusion: single-gpu-tuner-tests-no-fusion
	LD_LIBRARY_PATH=./: ./single-gpu-tuner-tests-no-fusion

#Run all tests
run-tests: single-gpu-tests single-gpu-no-fusion-tests run-tuner-test-no-fusion

clean:
	rm -rf kron libKron.so
