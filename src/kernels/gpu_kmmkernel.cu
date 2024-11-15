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
  return KMMKernel::canCompute(problem, hw, p2p, probBatchType, exactFuse);
}