#include "kernels/gpu_kmmkernel.h"

std::string GPUKMMKernel::str() const {
  std::stringstream info;
  info << backend() << "_" << arch() << "_" 
       << getNumThreads() << "_" << KMMKernel::str();
  return info.str();
}

dim3 GPUKMMKernel::grid(KMMProblem problem) const {
  Matrix tileX = getTileX(problem);
  Factor tileF = getTileF(problem);
  if (opX == fastKronOp_N) {
    return dim3(DIVUP(problem.k(), tileX.n()) * DIVUP(problem.f(0).q(), tileF.q()),
                DIVUP(problem.m(), tileX.m()),
                1);
  } else {
    //problem.k() can be very large, which can make grid.y more than the limit (65535).
    //Distribute grid.y to y and z.
    uint32_t origGridy = DIVUP(problem.k(), tileX.n()) *
                         DIVUP(problem.f(0).q(), tileF.q());
    dim3 grid = {0,0,0};
    if (origGridy <= 32768) {
      grid.y = origGridy;
      grid.z = 1;
    } else {
      grid.y = 32768;
      grid.z = DIVUP(origGridy, 32768);
    }
    return dim3(DIVUP(problem.m(), tileX.m()), grid.y, grid.z);
  }
}

bool GPUKMMKernel::canCompute(KMMProblem problem, const HardwareDetails* hw, bool p2p, 
                              KernelBatchType::Ty probBatchType,
                              bool exactFuse) {
  if (KMMKernel::canCompute(problem, hw, p2p, probBatchType, exactFuse)) {
    dim3 g = grid(problem);
    if (g.y >= 65536 || g.z >= 65536) {
      return false;
    }
    return true;
  }
  return false;
}