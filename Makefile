matmul-kp: src/matmul-kp.cpp
	g++ src/matmul-kp.cpp -o $<

all:
	nvcc -lcublas src/sample_cublas.cu -o saxpy_kron
