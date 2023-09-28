ANYOPTION=-IAnyOption/ AnyOption/anyoption.cpp
TEST_INCLUDE_DIRS = -Igoogletest/googletest/include/  -Lgoogletest/build/lib/ -Isrc/ -Isrc/apps/ -Isrc/tests
GOOGLE_TEST_MAIN = googletest/googletest/src/gtest_main.cc
TEST_LFLAGS = -lgtest -lpthread
GXX=g++

all: libKron.so

include src/device/Makefile

kron.o: src/kron.cu src/kernel_defs.cuh $(KRON_KERNELS)/kernel_decl.inc src/kron.h src/thread_pool.h
	$(NVCC) -Xcompiler=-fPIC,-fopenmp,-O3 $< -Isrc/ -I$(KRON_KERNELS) -c -o $@ -Xptxas=-v,-O3 $(ARCH_CODE_FLAGS)

libKron.so: device_kernels.o kron.o
	$(NVCC) -shared -lnccl -o $@ device_kernels.o kron.o

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
gen-multi-gpu-tests-kernel: src/gen_tuner_kernels.py
	python src/gen_tuner_kernels.py -same-factors 4 64 64 4 128 128 -dist-kernels -match-config 128,64,64,64,2,4096,2,16,1 128,128,128,128,1,8192,2,32,1

multi-gpu-no-fusion-tests: copy-multi-gpu-tests-kernel libKron.so tests/testBase.h tests/multi-gpu-no-fusion-tests.cu
	$(NVCC) tests/$@.cu $(TEST_INCLUDE_DIRS) $(TEST_LFLAGS) $(GOOGLE_TEST_MAIN) $(ARCH_CODE_FLAGS) -O3 -Xcompiler=-fopenmp,-O3,-Wall -L. -lKron -o $@

run-multi-gpu-nccl-no-fusion-tests: multi-gpu-no-fusion-tests
	LD_LIBRARY_PATH=./: DIST_COMM=NCCL ./multi-gpu-no-fusion-tests

run-multi-gpu-p2p-no-fusion-tests: multi-gpu-no-fusion-tests
	LD_LIBRARY_PATH=./: DIST_COMM=P2P ./multi-gpu-no-fusion-tests

gen-multi-gpu-tuner-kernels: src/gen_tuner_kernels.py
	python src/gen_tuner_kernels.py -same-factors 5 16 16 -dist-kernels 

multi-gpu-tuner-tests: gen-multi-gpu-tuner-kernels libKron.so tests/testBase.h tests/multi-gpu-tuner-tests.cu
	$(NVCC) tests/$@.cu $(TEST_INCLUDE_DIRS) $(TEST_LFLAGS) $(GOOGLE_TEST_MAIN) $(ARCH_CODE_FLAGS) -O3 -Xcompiler=-fopenmp,-O3,-Wall -L. -lKron -o $@

run-multi-gpu-tuner-tests: multi-gpu-tuner-tests
	LD_LIBRARY_PATH=./: DIST_COMM=P2P ./multi-gpu-tuner-tests

#Run all tests
run-all-single-gpu-tests: run-single-gpu-fusion-tests run-single-gpu-no-fusion-tests run-single-gpu-tuner-tests run-single-gpu-non-square-tuner-tests

run-all-multi-gpu-tests: run-multi-gpu-p2p-no-fusion-tests run-multi-gpu-nccl-no-fusion-tests

clean:
	rm -rf kron libKron.so
