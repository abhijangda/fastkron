cuda-example: cuda-example.cu
	nvcc $< -L ../build/ -lFastKron -I ../include -o $@ -g

x86-example: x86-example.cpp
	g++ $< -L ../build/ -lFastKron -I ../include -o $@ -g
