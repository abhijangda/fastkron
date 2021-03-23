
all:
	nvcc -lcublas src/sample_cublas.cu -o saxpy_kron
