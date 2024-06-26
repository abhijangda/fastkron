#include "kernel_info.h"

struct GPUKernel : public KernelInfo {
  void* kernelFunc;

  uint NumThreads;
  uint AAlignment;
  uint KronAlignment;

  GPUKernel() {}
  GPUKernel(void* invokerFunc, Factor f, Factor tileF, Matrix tileX, 
             uint FusedFacs, bool DistributeToGPUs,
             uint RegM, uint RegK, uint RegQ, FastKronType elemType, uint OptLevel,
             fastKronOp opX, fastKronOp opF,
             void*(*getKernelFunc)(), uint NumThreads,
             uint AAlignment, uint KronAlignment) :
             KernelInfo(invokerFunc, f, tileF, tileX, FusedFacs, DistributeToGPUs, 
             RegM, RegK, RegQ, elemType, OptLevel, opX, opF),
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
    info << runtimeStr() << "_" << archStr() << "_" << NumThreads << "_" << KernelInfo::str();
    return info.str();
  }

  dim3 grid(KMMProblem problem) {
    Matrix tileX_ = getTileX(problem);
    Factor tileF_ = getTileF(problem);
    if (opX == fastKronOp_N) {
      return dim3(DIVUP(problem.k(), tileX_.n()) * DIVUP(problem.f(0).q(), tileF_.q()),
                  DIVUP(problem.m(), tileX_.m()),
                  1);
    } else {
      //problem.k() can be very large, which can make grid.y more than the limit (65535).
      //Distribute grid.y to y and z
      uint32_t origGridy = DIVUP(problem.k(), tileX_.n()) * DIVUP(problem.f(0).q(), tileF_.q());
      dim3 grid = {0,0,0};
      if (origGridy <= 32768) {
        grid.y = origGridy;
        grid.z = 1;
      } else {
        grid.y = 32768;
        grid.z = DIVUP(origGridy, 32768);
      }
      return dim3(DIVUP(problem.m(), tileX_.m()),
                  grid.y, grid.z);
    }
  }

  virtual bool canCompute(KMMProblem problem, HardwareDetails* hardware, bool p2p, bool exactFuse = true) {
    if (KernelInfo::canCompute(problem, hardware, p2p, exactFuse)) {
      dim3 g = grid(problem);
      if (g.y >= 65536 || g.z >= 65536) {
        return false;
      }
      return true;
    }
    return false;
  }

  uint32_t numBlocks(KMMProblem problem) {
    dim3 g = grid(problem);
    return g.x*g.y*g.z;
  }

  dim3 block() {
    return dim3{NumThreads, 1, 1};
  }

  size_t sharedMemSize() {
    return totalTileSize();
  }

  size_t sharedMemSize(KMMProblem problem) {
    return MAX(totalTileSize(problem), totalTileSize());
  }
};