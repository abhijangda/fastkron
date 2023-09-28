NVCC=/usr/local/cuda/bin/nvcc
ARCH_CODE_FLAGS=-gencode arch=compute_70,code=sm_70
KRON_KERNELS=src/device/kron-kernels/

include $(KRON_KERNELS)make_device_kernels

device_kernels.o: $(DEVICE_KERNELS)
	ld -relocatable $< -o $@

$(KRON_KERNELS)/%.o: $(KRON_KERNELS)/%.cu #src/device_functions.cuh src/kernel.cuh
	$(NVCC) -Xcompiler=-fPIC,-shared,-fopenmp,-O3 $< -Isrc/ -c -o $@ -Xptxas=-v,-O3 $(ARCH_CODE_FLAGS) -lnccl