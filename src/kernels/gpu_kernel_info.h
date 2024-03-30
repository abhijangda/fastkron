#include "kernel_info.h"

struct GPUKernel : public KernelInfo {
  void* kernelFunc;

  uint NumThreads;
  uint AAlignment;
  uint KronAlignment;

  GPUKernel() {}
  GPUKernel(void* invokerFunc, Factor f, Factor tileF, Matrix tileX, 
             uint FusedFacs, bool DistributeToGPUs,
             uint RegK, uint RegQ, ElementType elemType, uint OptLevel,
             fastKronOp opX, fastKronOp opF,
             void*(*getKernelFunc)(), uint NumThreads,
             uint AAlignment, uint KronAlignment) :
             KernelInfo(invokerFunc, f, tileF, tileX, FusedFacs, DistributeToGPUs, 
             RegK, RegQ, elemType, OptLevel, opX, opF),
             NumThreads(NumThreads), kernelFunc(getKernelFunc()),
             AAlignment(AAlignment), KronAlignment(KronAlignment) {}

  bool isValid() {
    return true;
    const uint ValidThreads = ((tileX.n()/f.p())/RegK) * (tileF.q()/RegQ);
    if (NumThreads != ROUNDUP(ValidThreads, CUDA_WARP_SIZE)) {
      std::cout << "Invalid kernel config " << str() << std::endl; 
      return false;
    }
    return KernelInfo::isValid() && kernelFunc != nullptr;
  }

  virtual std::string str() const {
    std::stringstream info;
    info << NumThreads << "_" << KernelInfo::str();
    return info.str();
  }

  dim3 grid(KMMProblem problem) {
    Matrix tileX_ = getTileX(problem);
    Factor tileF_ = getTileF(problem);
    return dim3(DIVUP(problem.k(), tileX_.n()) * DIVUP(problem.f(0).q(), tileF_.q()),
                DIVUP(problem.m(), tileX_.m()),
                1);
  }

  dim3 block() {
    return dim3{NumThreads, 1, 1};
  }

  size_t sharedMemSize() {
    return totalTileSize();
  }

  size_t sharedMemSize(KMMProblem problem) {
    return totalTileSize(problem);
  }
};