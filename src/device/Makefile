NVCC=/usr/local/cuda/bin/nvcc
STD_CPP=-std=c++17
ARCH_CODE_FLAGS=-gencode arch=compute_70,code=sm_70
DEVICE=src/device
KRON_KERNELS=$(DEVICE)/kron-kernels

-include $(KRON_KERNELS)/make_device_kernels

$(KRON_KERNELS)/%.o: $(KRON_KERNELS)/%.cu $(DEVICE)/device_functions.cuh $(DEVICE)/kernel.cuh src/handle.h src/device/params.h
	$(NVCC) -Xcompiler=-fPIC,-shared,-fopenmp,-O3 $(STD_CPP) $< -Isrc/ -c -o $@ -Xptxas=-O3,-v $(ARCH_CODE_FLAGS)

device_kernels.o: $(DEVICE_KERNELS)
	ld -relocatable $? -o $@