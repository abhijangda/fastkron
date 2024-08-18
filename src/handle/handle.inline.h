#pragma once

inline bool FastKronHandle::hasBackend(fastKronBackend backend) {
  return (backends & backend);
}

inline KernelDatabase* FastKronHandle::getKernelDb(fastKronBackend backend) {
  switch (backend) {
    case fastKronBackend_X86:
      #ifdef ENABLE_X86
        return &x86Kernels;
      #endif
    case fastKronBackend_CUDA:
      #ifdef ENABLE_CUDA
        return &cudaKernels;
      #endif
    case fastKronBackend_HIP:
      #ifdef ENABLE_HIP
        return &hipKernels;
      #endif
    default:
      return nullptr;
  }
}

inline std::vector<KernelDatabase*> FastKronHandle::getAllKernelDbs() {
  std::vector<KernelDatabase*> out;
  if        (hasBackend(fastKronBackend_X86))  {
    out.push_back(getKernelDb(fastKronBackend_X86));
  } else if (hasBackend(fastKronBackend_CUDA)) {
    out.push_back(getKernelDb(fastKronBackend_CUDA));
  } else if (hasBackend(fastKronBackend_HIP))  {
    out.push_back(getKernelDb(fastKronBackend_HIP));
  }
  return out;
}

inline void FastKronHandle::setOptions(uint32_t options) {this->options = options;}

inline bool FastKronHandle::canTune() {
  return (options & fastKronOptionsTune) ==
          fastKronOptionsTune;
}

inline bool FastKronHandle::getUseFusion() {
  return (options & fastKronOptionsUseFusion) ==
          fastKronOptionsUseFusion;
}