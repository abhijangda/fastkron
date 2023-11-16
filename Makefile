ANYOPTION=-IAnyOption/ AnyOption/anyoption.cpp
TEST_INCLUDE_DIRS = -Igoogletest/googletest/include/  -Lgoogletest/build/lib/ -Isrc/ -Isrc/apps/ -Isrc/tests
GOOGLE_TEST_MAIN = googletest/googletest/src/gtest_main.cc
TEST_LFLAGS = -lgtest -lpthread
GXX=g++
GOOGLE_TEST = googletest
GOOGLE_TEST_BUILD = $(GOOGLE_TEST)/build

all: libKron.so

include src/device/Makefile

gtest:
	mkdir -p $(GOOGLE_TEST_BUILD) && cd $(GOOGLE_TEST_BUILD) && cmake .. && make -j

kron.o: src/kron.cu src/kernel_defs.cuh $(KRON_KERNELS)/kernel_decl.inc src/kron.h src/fastkron.h src/thread_pool.h src/device/params.h src/device/otherkernels.cuh
	$(NVCC) -std=c++17 -Xcompiler=-fPIC,-fopenmp $< -Isrc/ -I$(KRON_KERNELS) -c -o $@ -Xptxas=-v,-O3 $(ARCH_CODE_FLAGS) -g -O2

libKron.so: device_kernels.o kron.o
	$(NVCC) -shared -lnccl -o $@ device_kernels.o kron.o

kron: tests/main.cu libKron.so tests/testBase.h
	$(NVCC) $< -Xcompiler=-fopenmp,-O3,-Wall -Isrc/ $(ANYOPTION) -L. -lKron -o $@ -O3 -g

#Make tests
gen-single-gpu-kernels: src/gen_tuner_kernels.py tests/single-gpu-kernel-decls.in
	python3 src/gen_tuner_kernels.py -same-factors 20 2,2 -same-factors 10 4,4 -same-factors 8 8,8 -same-factors 6 16,16 -same-factors 5 32,32 -same-factors 4 64,64 -same-factors 3 128,128  -match-configs-file tests/single-gpu-kernel-decls.in
	
single-gpu-no-fusion-tests: libKron.so gtest tests/single-gpu-no-fusion-tests.cu tests/testBase.h
	$(NVCC) tests/$@.cu $(TEST_INCLUDE_DIRS) $(TEST_LFLAGS) $(GOOGLE_TEST_MAIN) $(ARCH_CODE_FLAGS) -O3 -Xcompiler=-fopenmp,-O3,-Wall -L. -lKron -o $@

run-single-gpu-no-fusion-tests: single-gpu-no-fusion-tests
	LD_LIBRARY_PATH=./: ./single-gpu-no-fusion-tests

single-gpu-fusion-tests: libKron.so tests/single-gpu-fusion-tests.cu libKron.so tests/testBase.h 
	$(NVCC) tests/$@.cu $(TEST_INCLUDE_DIRS) $(TEST_LFLAGS) $(GOOGLE_TEST_MAIN) $(ARCH_CODE_FLAGS) -O3 -Xcompiler=-fopenmp,-O3,-Wall -L. -lKron -o $@

run-single-gpu-fusion-tests: single-gpu-fusion-tests
	LD_LIBRARY_PATH=./: ./single-gpu-fusion-tests

#Tests for the autotuner
gen-tuner-kernels: src/gen_tuner_kernels.py
	python3 src/gen_tuner_kernels.py -same-factors 4 16,16 -same-factors 3 64,64

single-gpu-tuner-tests: tests/single-gpu-tuner-tests.cu libKron.so tests/testBase.h
	$(NVCC) tests/$@.cu $(TEST_INCLUDE_DIRS) $(TEST_LFLAGS) $(GOOGLE_TEST_MAIN) $(ARCH_CODE_FLAGS) -O3 -Xcompiler=-fopenmp,-O3,-Wall -L. -lKron -o $@

run-single-gpu-tuner-tests: single-gpu-tuner-tests
	LD_LIBRARY_PATH=./: ./single-gpu-tuner-tests

#Test for Non Square factors
gen-non-square-tuner-test-kernels: src/gen_tuner_kernels.py
	python3 src/gen_tuner_kernels.py -same-factors 4 8,16 -same-factors 3 32,16 -same-factors 3 128,32

single-gpu-non-square-tuner-tests: libKron.so tests/single-gpu-non-square-tuner-tests.cu tests/testBase.h
	$(NVCC) tests/$@.cu $(TEST_INCLUDE_DIRS) $(TEST_LFLAGS) $(GOOGLE_TEST_MAIN) $(ARCH_CODE_FLAGS) -O3 -Xcompiler=-fopenmp,-O3,-Wall -L. -lKron -o $@

run-single-gpu-non-square-tuner-tests: single-gpu-non-square-tuner-tests
	LD_LIBRARY_PATH=./: ./single-gpu-non-square-tuner-tests

#Test for Distinct Shapes Single GPU
gen-single-gpu-distinct-shapes: src/gen_tuner_kernels.py
	python3 src/gen_tuner_kernels.py -distinct-factors 3 8,16 16,8 8,32

single-gpu-distinct-shapes: libKron.so tests/single-gpu-distinct-shapes.cu tests/testBase.h
	$(NVCC) tests/$@.cu $(TEST_INCLUDE_DIRS) $(TEST_LFLAGS) $(GOOGLE_TEST_MAIN) $(ARCH_CODE_FLAGS) -O3 -Xcompiler=-fopenmp,-O3,-Wall -L. -lKron -o $@

run-single-gpu-distinct-shapes: single-gpu-distinct-shapes
	LD_LIBRARY_PATH=./: ./single-gpu-distinct-shapes

#Tests for Single GPU Odd Shapes
gen-single-gpu-odd-shapes: src/gen_tuner_kernels.py
	python3 src/gen_tuner_kernels.py -same-factors 2 31,16 -same-factors 2 16,31 -same-factors 4 31,31 

single-gpu-odd-shapes: libKron.so tests/single-gpu-odd-shapes.cu tests/testBase.h
	$(NVCC) tests/$@.cu $(TEST_INCLUDE_DIRS) $(TEST_LFLAGS) $(GOOGLE_TEST_MAIN) $(ARCH_CODE_FLAGS) -O3 -Xcompiler=-fopenmp,-O3,-Wall -L. -lKron -o $@

run-single-gpu-odd-shapes: single-gpu-odd-shapes
	LD_LIBRARY_PATH=./: ./single-gpu-odd-shapes

#Multi GPU Tests Square Factors 
gen-multi-gpu-tests-kernel: src/gen_tuner_kernels.py
	python src/gen_tuner_kernels.py -same-factors 4 64,64 -same-factors 4 128,128 -dist-kernels -match-configs 128,64,64,64,2,4096,2,16,1 128,128,128,128,1,8192,2,32,1

multi-gpu-no-fusion-tests: libKron.so tests/testBase.h tests/multi-gpu-no-fusion-tests.cu
	$(NVCC) tests/$@.cu $(TEST_INCLUDE_DIRS) $(TEST_LFLAGS) $(GOOGLE_TEST_MAIN) $(ARCH_CODE_FLAGS) -O3 -Xcompiler=-fopenmp,-O3,-Wall -L. -lKron -o $@

run-multi-gpu-nccl-no-fusion-tests: multi-gpu-no-fusion-tests
	LD_LIBRARY_PATH=./: DIST_COMM=NCCL ./multi-gpu-no-fusion-tests

run-multi-gpu-p2p-no-fusion-tests: multi-gpu-no-fusion-tests
	LD_LIBRARY_PATH=./: DIST_COMM=P2P ./multi-gpu-no-fusion-tests

#Multi GPU Tuner Tests
gen-multi-gpu-tuner-kernels: src/gen_tuner_kernels.py
	python src/gen_tuner_kernels.py -same-factors 5 16,16 -dist-kernels 

multi-gpu-tuner-tests: libKron.so tests/testBase.h tests/multi-gpu-tuner-tests.cu
	$(NVCC) tests/$@.cu $(TEST_INCLUDE_DIRS) $(TEST_LFLAGS) $(GOOGLE_TEST_MAIN) $(ARCH_CODE_FLAGS) -O3 -Xcompiler=-fopenmp,-O3,-Wall -L. -lKron -o $@

run-multi-gpu-tuner-tests: multi-gpu-tuner-tests
	LD_LIBRARY_PATH=./: ./multi-gpu-tuner-tests

#Multi GPU Tests Non-Square Factors 
gen-multi-gpu-no-fusion-non-square-tests-kernel: src/gen_tuner_kernels.py
	python src/gen_tuner_kernels.py -same-factors 5 8,32 -same-factors 4 64,16 -dist-kernels

multi-gpu-no-fusion-non-square-tests: libKron.so tests/testBase.h tests/multi-gpu-no-fusion-non-square-tests.cu
	$(NVCC) tests/$@.cu $(TEST_INCLUDE_DIRS) $(TEST_LFLAGS) $(GOOGLE_TEST_MAIN) $(ARCH_CODE_FLAGS) -O3 -Xcompiler=-fopenmp,-O3,-Wall -L. -lKron -o $@

run-p2p-multi-gpu-no-fusion-non-square-tests: multi-gpu-no-fusion-non-square-tests
	LD_LIBRARY_PATH=./: DIST_COMM=P2P ./$<

run-nccl-multi-gpu-no-fusion-non-square-tests: multi-gpu-no-fusion-non-square-tests
	LD_LIBRARY_PATH=./: DIST_COMM=NCCL ./$<

#Multi GPU different shapes
gen-multi-gpu-distinct-shapes: src/gen_tuner_kernels.py
	python3 src/gen_tuner_kernels.py -distinct-factors 3 8,16 32,8 16,8 -match-configs 64,16,8,16,1,256,1,8,1 32,8,32,8,1,256,1,2,1 128,8,16,8,1,256,1,1 -dist-kernels

multi-gpu-distinct-shapes: libKron.so tests/testBase.h tests/multi-gpu-distinct-shapes.cu
		$(NVCC) tests/$@.cu $(TEST_INCLUDE_DIRS) $(TEST_LFLAGS) $(GOOGLE_TEST_MAIN) $(ARCH_CODE_FLAGS) -O3 -Xcompiler=-fopenmp,-O3,-Wall -L. -lKron -o $@

run-p2p-multi-gpu-distinct-shapes: multi-gpu-distinct-shapes
	LD_LIBRARY_PATH=./: DIST_COMM=P2P ./multi-gpu-distinct-shapes

run-nccl-multi-gpu-distinct-shapes: multi-gpu-distinct-shapes
	LD_LIBRARY_PATH=./: DIST_COMM=NCCL ./multi-gpu-distinct-shapes

#Run all tests
#run-all-single-gpu-tests: run-single-gpu-fusion-tests run-single-gpu-no-fusion-tests run-single-gpu-tuner-tests run-single-gpu-non-square-tuner-tests

#run-all-multi-gpu-tests: run-multi-gpu-p2p-no-fusion-tests run-multi-gpu-nccl-no-fusion-tests

clean:
	rm -rf kron libKron.so
