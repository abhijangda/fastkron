ANYOPTION=-IAnyOption/ AnyOption/anyoption.cpp

all: kron

libKron.so: src/kron.cu src/kron.h src/kron_device.cu src/kernel_decl.inc
	nvcc -Xcompiler=-fPIC,-shared,-fopenmp,-O3 $< -Isrc/ -o $@ -Xptxas=-v -gencode arch=compute_70,code=sm_70

kron: test/main.cu libKron.so 
	nvcc $< -Xcompiler=-fopenmp,-O3,-Wall -Isrc/ $(ANYOPTION) -L. -lKron -o $@

clean:
	rm -rf kron libKron.so
