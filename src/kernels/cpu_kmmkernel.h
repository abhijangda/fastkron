#include "kernels/kmmkernel.h"

#pragma once

/**
 * CPUKMMKernel - A subclass for KMMKernels running on CPU
 */
struct CPUKMMKernel : public KMMKernel {
public:
  CPUKMMKernel() {}
  CPUKMMKernel(void* kernelInvoker, FastKronType elemType,
               Factor f, Factor tileF, Matrix tileX, uint fusedFacs, bool P2PStore,
               uint regM, uint regK, uint regQ, uint optLevel,
               fastKronOp opX, fastKronOp opF, KernelBatchType::Ty kernelBatchType) : 
               KMMKernel(kernelInvoker, elemType, f, tileF, tileX,
                         fusedFacs, P2PStore, regM, regK, regQ,
                         optLevel, opX, opF, kernelBatchType) {}
};

/**
 * X86KMMKernel - A subclass for KMMKernels running on an X86 CPU
 *                This class contains a member to determine the SIMD architecture of the kernel.
 */
struct X86KMMKernel : public CPUKMMKernel {
  /**
   * @simd: The SIMD architecture of the kernel either AVX256, AVX512, or SISD.
   */
private:
  X86SIMD simd;

public:
  X86KMMKernel() {}
  X86KMMKernel(X86SIMD simd, void* kernelInvoker, FastKronType elemType,
            Factor f, Factor tileF, Matrix tileX, uint fusedFacs, bool P2PStore,
            uint regM, uint regK, uint regQ, uint optLevel,
            fastKronOp opX, fastKronOp opF, KernelBatchType::Ty kernelBatchType) :
            CPUKMMKernel(kernelInvoker, elemType, f, tileF, tileX, fusedFacs,
                         P2PStore, regM, regK, regQ, optLevel, opX, opF,
                         kernelBatchType),
            simd(simd) {}

  X86SIMD getSIMD() {return simd;}

  /**
   * canCompute - Overrides the method of KMMKernel and checks if simd of this kernel
   *              can run on the given hardware.
   */
  virtual bool canCompute(KMMProblem problem, const HardwareDetails* hw,
                          bool p2p, KernelBatchType::Ty probBatchType, bool exactFuse = true) {
    if (CPUKMMKernel::canCompute(problem, hw, p2p, probBatchType, exactFuse)) {
      //A CPU with higher SIMD width (say AVX512) always support a lower
      //SIMD width (say AVX256)
      return getSIMD() <= ((X86ArchDetails*)hw)->simd;
    }
    return false;
  }

  /**
   * backend - Overrides the method of KMMKernel and always return X86.
   */
  virtual std::string backend() const {return "X86";}

  /**
   * arch - Overrides the method of KMMKernel and return SIMD architecture.
   */
  virtual std::string arch()    const {return x86simdToStr(simd);}

  /**
   * str - Overrides the method of KMMKernel.
   */
  virtual std::string str() const {
    std::stringstream info;
    info << backend() << "_" << arch() << "_" << KMMKernel::str();
    return info.str();
  }
};