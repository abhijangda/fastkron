#include "kernels/gpu_kmmkernel.h"

std::string GPUKMMKernel::str() const {
  std::stringstream info;
  info << backend() << "_" << arch() << "_" 
       << getNumThreads() << "_" << KMMKernel::str();
  return info.str();
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