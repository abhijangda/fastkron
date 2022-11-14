all: kron

libKron.so: src/kron.cu src/kron.h
	nvcc -Xcompiler=-fPIC,-shared,-fopenmp,-O3 $< -Isrc/ -o $@

kron: test/main.cu libKron.so
	nvcc $< -Xcompiler=-fopenmp,-O3 -Isrc/ -L. -lKron -o $@

clean:
	rm -rf kron