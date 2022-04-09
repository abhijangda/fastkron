all: kron

kron: kron.cu
	nvcc kron.cu -I ../../include -I ../../tools/util/include -I ../common/ -o kron -Xcompiler -fopenmp -O3
clean:
	rm -rf kron
