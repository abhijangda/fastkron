all: kron

kron: src/kron.cu
	nvcc $< -I ../../include -I ../../tools/util/include -I ../common/ -o $@ -Xcompiler -fopenmp -O3

kron-eval: src/kron.cu
	nvcc $< -DEVAL -I ../../include -I ../../tools/util/include -I ../common/ -o $@ -Xcompiler -fopenmp -O3

kron-eval-debug: src/kron.cu
	nvcc $< -DEVAL -I ../../include -I ../../tools/util/include -I ../common/ -g -O0 -o $@ -Xcompiler -fopenmp

clean:
	rm -rf kron