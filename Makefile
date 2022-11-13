all: kron

kron: kron.cu
	nvcc kron.cu -I ../../include -I ../../tools/util/include -I ../common/ -o kron -Xcompiler -fopenmp -O3

kron-eval: kron.cu
	nvcc kron.cu -DEVAL -I ../../include -I ../../tools/util/include -I ../common/ -o kron -Xcompiler -fopenmp -O3

kron-eval-debug: kron.cu
	nvcc kron.cu -DEVAL -I ../../include -I ../../tools/util/include -I ../common/ -g -O0 -o kron -Xcompiler -fopenmp

clean:
	rm -rf kron