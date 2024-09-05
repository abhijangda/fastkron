#include "kmmkernel.h"

struct GPUKernel : public KMMKernel {
  void* kernelFunc;

  uint NumThreads;
  uint AAlignment;
  uint KronAlignment;

  GPUKernel() {}
  GPUKernel(void* invokerFunc, FastKronType elemType, Factor f, Factor tileF, Matrix tileX, 
             uint FusedFacs, bool DistributeToGPUs,
             uint RegM, uint RegK, uint RegQ, uint OptLevel,
             fastKronOp opX, fastKronOp opF,
             void*(*getKernelFunc)(), uint NumThreads,
             uint AAlignment, uint KronAlignment) :
             KMMKernel(invokerFunc, elemType, f, tileF, tileX, FusedFacs, DistributeToGPUs, 
             RegM, RegK, RegQ, OptLevel, opX, opF),
             NumThreads(NumThreads), kernelFunc(getKernelFunc()),
             AAlignment(AAlignment), KronAlignment(KronAlignment) {}

  bool isValid() {
    return true;
    const uint ValidThreads = ((tileX.n()/f.p())/getRegK()) * (tileF.q()/getRegQ());
    if (NumThreads != ROUNDUP(ValidThreads, CUDA_WARP_SIZE)) {
      std::cout << "Invalid kernel config " << str() << std::endl; 
      return false;
    }
    return KMMKernel::isValid();
  }

  virtual std::string str() const {
    std::stringstream info;
    info << backend() << "_" << arch() << "_" << NumThreads << "_" << KMMKernel::str();
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
    if (KMMKernel::canCompute(problem, hardware, p2p, exactFuse)) {
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
    return getMaxTotalTileSize();
  }

  size_t sharedMemSize(KMMProblem problem) {
    return MAX(getTotalTileSize(problem), getMaxTotalTileSize());
  }
};