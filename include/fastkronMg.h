#include <stdint.h>
#include <cstddef>

#include <fastkron.h>

#pragma once

extern "C" {
//backends is a bitwise OR
fastKronError fastKronMgInitCUDA(fastKronHandle handle, void *streams, int gpus, int gpusInM = -1, int gpusInK = -1, int gpuLocalKrons = -1);

//TODO: modify such that the results are always written to the supplied result pointer 
fastKronError fastKronMgSGEMM(fastKronHandle handle, const uint32_t NumKronMats, void* x[], void* kronMats[], void* result[],
                              uint32_t M, uint32_t N, uint32_t K, uint32_t KronMatCols[], uint32_t KronMatRows[], 
                              void* temp1[], void* temp2[], void* stream);
fastKronError fastKronMgAllocX(fastKronHandle handle, void* dX[], void* hX, uint32_t M, uint32_t K);
fastKronError fastKronMgGatherY(fastKronHandle handle, void* dY[], void* hY, uint32_t M, uint32_t K, uint32_t NumKronMats, uint32_t KronMatCols[], uint32_t KronMatRows[]);
}