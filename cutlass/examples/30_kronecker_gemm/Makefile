all: kron

kron: kron.cu
	nvcc kron.cu -I ../../include -I ../../tools/util/include -I ../common/ -o kron
clean:
	rm -rf kron
