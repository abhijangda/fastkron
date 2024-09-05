#include "kernels/kmmkernel.h"

#pragma once

struct CPUKernel : public KMMKernel {
  CPUKernel() {}
  CPUKernel(void* kernelInvoker, FastKronType elemType, Factor f, Factor tileF, Matrix tileX, 
            uint fusedFacs, bool P2PStore, 
            uint regM, uint regK, uint regQ, uint optLevel,
            fastKronOp opX, fastKronOp opF) : 
            KMMKernel (kernelInvoker, elemType, f, tileF, tileX, 
                        fusedFacs, P2PStore, regM, regK, regQ, optLevel, opX, opF) {}
};

struct X86Kernel : public CPUKernel {
  X86SIMD simd;
  X86Kernel() {}
  X86Kernel(X86SIMD simd, void* kernelInvoker, FastKronType elemType, Factor f, Factor tileF, Matrix tileX, 
            uint fusedFacs, bool P2PStore, 
            uint regM, uint regK, uint regQ, uint optLevel,
            fastKronOp opX, fastKronOp opF) :
            CPUKernel(kernelInvoker, elemType, f, tileF, tileX, fusedFacs, P2PStore, regM, regK, regQ, optLevel, opX, opF),
            simd(simd) {}
  
  virtual std::string backend() const {
    return "X86";
  }

  virtual std::string arch() const {
    return x86simdToStr(simd);
  }

  virtual std::string str() const {
    std::stringstream info;
    info << backend() << "_" << arch() << "_" << KMMKernel::str();
    return info.str();
  }

  virtual bool canCompute(KMMProblem problem, HardwareDetails* hardware, bool p2p, bool exactFuse = true) {
    if (CPUKernel::canCompute(problem, hardware, p2p, exactFuse)) {
      return simd <= ((X86ArchDetails*)hardware)->simd;
    }
    return false;
  }
};