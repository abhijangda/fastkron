#include <filesystem>
#include <fstream>
#include <regex>
#include <set>
#include <immintrin.h>

#include "utils/logger.h"

#include "kernels/params.h"
#include "kernel_db/cpu_kernel_db.h"
#include "kernels/cpu_kmmkernel.h"

#ifdef ENABLE_X86
  //Defines ALL_X86_KERNELS array
  #include "kernels/cpu/x86/kron-kernels/kernel_decl.inc"
#endif

/**
 * @AllX86Kernels: An array of all X86 compiled kernels.
 */
X86KMMKernel AllX86Kernels[] = {
#ifdef ENABLE_X86
  ALL_X86_KERNELS
#endif
};

CPUKernelDatabase::CPUKernelDatabase() :
  KernelDatabase(), TileXs(), TileYs(), TileFs()
 {}

void CPUKernelDatabase::allocate_caches() {
  //Go through all loaded kernels and allocate cache for maximum size
  uint32_t maxTileX = 0;
  uint32_t maxTileF = 0;
  uint32_t maxTileY = 0;

  for (const auto & [_, kernels] : compiledKernels) {
    for (auto k : kernels) {
      maxTileX = std::max(k->getMaxTileX().numel(), maxTileX);
      maxTileF = std::max(k->getMaxTileF().numel(), maxTileF);
      maxTileY = std::max(k->getMaxTileY().numel(), maxTileY);
    }
  }
  TileXs.alloc(getMaxThreads(), maxTileX * sizeof(double));
  TileFs.alloc(getMaxThreads(), maxTileF * sizeof(double));
  TileYs.alloc(getMaxThreads(), maxTileY * sizeof(double));
}

fastKronError CPUKernelDatabase::procMemset(uint32_t, Matrix& m, float val) {
  memset<float>(m.data<float>(0), m.numel(), val);
  return fastKronSuccess;
}

fastKronError CPUKernelDatabase::procMalloc(uint32_t, size_t size, void*& ptr) {
  ptr = new char[size];
  return ptr != nullptr ? fastKronSuccess : fastKronInvalidArgument;
}

fastKronError CPUKernelDatabase::procFree(uint32_t, void* ptr) {
  if (ptr == NULL) return fastKronInvalidArgument;
  delete[] (char*)ptr;
  return fastKronSuccess;
}

template<typename KMMProblem, typename EpilogueParams>
fastKronError invoke(CPUKMMKernel& kernelInfo,
                     Matrix kernelTileX, Factor kernelTileF,
                     KMMProblem problem,
                     const uint fidx, CPUCaches& caches,
                     typename KMMProblem::Matrices intermediates,
                     DistributedParams distParams,
                     EpilogueParams epilogueParams,
                     KernelMode execMode) {
  KernelParams<KMMProblem> params (problem, &caches, kernelTileX, kernelTileF,
                                              fidx, execMode);
  FusedParams<KMMProblem> fusedParams (problem, intermediates, kernelInfo.getMaxTileX().n());
  //TODO: change this to kernel.invoke
  typedef void (*KronMatmulKernelTy)(KernelParams<KMMProblem>&, FusedParams<KMMProblem>&,
                                     DistributedParams&, EpilogueParams&);
  KronMatmulKernelTy(kernelInfo.kernelInvoker)(params, fusedParams, distParams, epilogueParams);
  return fastKronSuccess;
}

template<typename KMMProblem, typename EpilogueParams>
fastKronError CPUKernelDatabase::invokeKernel(KMMKernel* kernel, KMMProblem problem,
                                              const uint fidx,
                                              typename KMMProblem::Matrices intermediates,
                                              EpilogueParams epilogueParams,
                                              KernelMode execMode) {
  DistributedParams distParams;
  CPUKMMKernel& cpuKernel = dynamic_cast<CPUKMMKernel&>(*kernel);
  CPUCaches caches = {TileXs.ptr, TileFs.ptr, TileYs.ptr};
  Matrix kernelTileX = kernel->getTileX(problem);
  Factor kernelTileF = kernel->getTileF(problem);

  if (problem.n() > kernel->getFusedFacs()) {
    Logger(LogLevel::Debug) << "Kernel with " << kernel->getFusedFacs()      <<
                               " fused factors cannot compute problem with " <<
                               problem.n() << " factors" << std::endl;
    return fastKronInvalidArgument;
  }

  switch(kernel->getFusedFacs()) {
    case 1:
      return invoke(cpuKernel, kernelTileX, kernelTileF,
                    problem.template factorSlice<1>(), fidx,
                    caches, intermediates.template sliceOrEmpty<1>(0),
                    distParams, epilogueParams, execMode);
    case 2:
      return invoke(cpuKernel, kernelTileX, kernelTileF,
                    problem.template factorSlice<2>(), fidx,
                    caches, intermediates.template sliceOrEmpty<2>(0),
                    distParams, epilogueParams, execMode);
    case 3:
      return invoke(cpuKernel, kernelTileX, kernelTileF,
                    problem.template factorSlice<3>(), fidx,
                    caches, intermediates.template sliceOrEmpty<3>(0),
                    distParams, epilogueParams, execMode);
    case 4:
      return invoke(cpuKernel, kernelTileX, kernelTileF,
                    problem.template factorSlice<4>(), fidx,
                    caches, intermediates.template sliceOrEmpty<4>(0),
                    distParams, epilogueParams, execMode);
    case 5:
      return invoke(cpuKernel, kernelTileX, kernelTileF,
                    problem.template factorSlice<5>(), fidx,
                    caches, intermediates.template sliceOrEmpty<5>(0),
                    distParams, epilogueParams, execMode);
    case 6:
      return invoke(cpuKernel, kernelTileX, kernelTileF,
                    problem.template factorSlice<6>(), fidx,
                    caches, intermediates.template sliceOrEmpty<6>(0),
                    distParams, epilogueParams, execMode);
    default:
      Logger(LogLevel::Debug) << "Invalid number of fused kernels" << std::endl;
      return fastKronKernelNotFound;
  }
}

fastKronError CPUKernelDatabase::invokeKernel(KMMKernel* kernel, KMMProblem problem,
                                              const uint fidx,
                                              KMMProblem::Matrices intermediates,
                                              EpilogueParams epilogueParams,
                                              KernelMode execMode) {
if (kernel->getBatchType() == KernelBatchType::StridedBatched) {
    //Execute a single batch problem using a strided batched kernel
    KMMProblemStridedBatched stridedProblem(problem);
    EpilogueStridedBatchedParams stridedEpilogue(epilogueParams, problem.y());
    KMMProblemStridedBatched::Matrix stridedIntermediatesArr[intermediates.len()];
    for (uint32_t i = 0; i < intermediates.len(); i++) {
      stridedIntermediatesArr[i] = KMMProblemStridedBatched::Matrix(intermediates[i]);
    }
    return invokeKernel(kernel, stridedProblem, fidx,
                        KMMProblemStridedBatched::Matrices(stridedIntermediatesArr, intermediates.len()),
                        stridedEpilogue, execMode);
  }

  return invokeKernel<KMMProblem, EpilogueParams>(
    kernel, problem, fidx, intermediates, epilogueParams, execMode);
}

fastKronError CPUKernelDatabase::invokeKernel(KMMKernel* kernel, KMMProblemStridedBatched problem,
                                              const uint fidx,
                                              KMMProblemStridedBatched::Matrices intermediates,
                                              EpilogueStridedBatchedParams epilogueParams,
                                              KernelMode execMode) {
  return invokeKernel<KMMProblemStridedBatched, EpilogueStridedBatchedParams>(
            kernel, problem, fidx, intermediates, epilogueParams, execMode);
}

template<typename KMMProblemT, typename EpilogueParamsT>
fastKronError CPUKernelDatabase::timeKernel(KMMKernel* kernel, KMMProblemT problem,
                                            const uint fidx,
                                            DistributedParams /*distParams*/,
                                            EpilogueParamsT epilogueParams,
                                            KernelMode execMode,
                                            bool useP2PStore,
                                            int /*warmups*/, int runs,
                                            float& runtime) {
  runtime = std::numeric_limits<float>::max();
  //Avoid the SISD kernel when running on AVX/AVX512
  if ((*(dynamic_cast<const X86ArchDetails*>(hardware[0]))).simd != X86SIMD::SISD) {
    if (((X86KMMKernel*)kernel)->getSIMD() == X86SIMD::SISD) return fastKronSuccess;
  }
  std::vector<typename KMMProblemT::Matrix> vecIntermediates(10);
  typename KMMProblemT::Matrices fakeIntermediates(vecIntermediates.data(), vecIntermediates.size());

  fastKronError status;
  for (int sample = 0; sample < 10; sample++) {
    float avgtime = 0;
    for (int r = 0; r < runs; r++) {

      //Trashing the L3 Cache does not seem to have any effect
      uint32_t l3size = ((CPUArchDetails*)hardware[0])->totalL3Size();
      if ((problem.x().numel() + problem.y().numel()) * sizeof(float) <= l3size)
        parallelCopy(trash1, trash2, l3size);
      double startTime = getCurrTime();
      if (useP2PStore) {
        //FUTURE WORK
        // status = invokeP2PStoreKernel(kernel, problem, fidx,
        //                               distParams, epilogueParams, execMode);
      } else {
        status = invokeKernel(kernel, problem, fidx, fakeIntermediates,
                              epilogueParams, execMode);
      }
      double endTime = getCurrTime();
      avgtime += (float)((endTime - startTime)/1e3);
    }

    if (status != fastKronSuccess) {
      Logger(LogLevel::Info) << "Error in CPU autotuning "     <<
                                fastKronGetErrorString(status) <<
                                std::endl;
      return status;
    }

    runtime = std::min(avgtime/runs, runtime);
  }

  return status;
}

fastKronError CPUKernelDatabase::timeKernel(KMMKernel* kernel, KMMProblem problem,
                                            const uint fidx,
                                            DistributedParams distParams,
                                            EpilogueParams epilogueParams,
                                            KernelMode execMode,
                                            bool useP2PStore,
                                            int warmups, int runs,
                                            float& runtime) {
  return timeKernel<KMMProblem, EpilogueParams>(kernel, problem, fidx, distParams, epilogueParams,
                                                execMode, useP2PStore, warmups, runs, runtime);
}

fastKronError CPUKernelDatabase::timeKernel(KMMKernel* kernel, KMMProblemStridedBatched problem,
                                            const uint fidx,
                                            DistributedParams distParams,
                                            EpilogueStridedBatchedParams epilogueParams,
                                            KernelMode execMode,
                                            bool useP2PStore,
                                            int warmups, int runs,
                                            float& runtime) {
  return timeKernel<KMMProblemStridedBatched, EpilogueStridedBatchedParams>(
      kernel, problem, fidx, distParams, epilogueParams,
      execMode, useP2PStore, warmups, runs, runtime);
}

void cpuid(uint32_t in, uint32_t regs[4], uint32_t ecx = 0) {
#ifdef _WIN32
#else
  __asm__ (
    "cpuid" : "=a" (regs[0]), "=b" (regs[1]), "=c" (regs[2]), "=d" (regs[3]) :
    "a"(in), "c"(ecx)
  );
#endif
}

X86KernelDatabase::X86KernelDatabase() {
  unsigned cpuidregs[4];

  // Get vendor
  std::string cpuVendor = "";
  {
    char vendor[12];
    cpuid(0, cpuidregs);
    ((unsigned *)vendor)[0] = cpuidregs[1]; // EBX
    ((unsigned *)vendor)[1] = cpuidregs[3]; // EDX
    ((unsigned *)vendor)[2] = cpuidregs[2]; // ECX
    cpuVendor = std::string(vendor, 12);
  }

  //Get Physical Cores
  uint32_t cores = 0;
  if (cpuVendor == "GenuineIntel") {
    // Get DCP cache info
    cpuid(4, cpuidregs);
    cores = ((cpuidregs[0] >> 26) & 0x3f) + 1; // EAX[31:26] + 1
  } else if (cpuVendor == "AuthenticAMD") {
    // Get NC: Number of CPU cores - 1
    cpuid(0x80000008, cpuidregs);
    cores = ((unsigned)(cpuidregs[2] & 0xff)) + 1; // ECX[7:0] + 1
  }

  // Get processor brand string
  // This seems to be working for both Intel & AMD vendors
  std::string model = "";
  for(uint32_t i=0x80000002; i<0x80000005; ++i) {
      cpuid(i, cpuidregs);
      char name[16];
      ((unsigned*)name)[0] = cpuidregs[0];
      ((unsigned*)name)[1] = cpuidregs[1];
      ((unsigned*)name)[2] = cpuidregs[2];
      ((unsigned*)name)[3] = cpuidregs[3];

      model += std::string(name, 16);
  }

  //Get L1 cache size in KB
  uint32_t l1Size = 0;
  if (cpuVendor == "AuthenticAMD") {
    cpuid(0x80000005, cpuidregs);
    l1Size = (cpuidregs[2] >> 24) & 0xFF; //ECX[31:24]
  } else if (cpuVendor == "GenuineIntel") {
    cpuid(0x4, cpuidregs);
    uint32_t ways = (cpuidregs[1] >> 22); //EBX[31:22];
    uint32_t partitions = (cpuidregs[1] >> 12) & ((1 << 10) - 1); //EBX[21:12]
    uint32_t sets = cpuidregs[2]; //ECX[31:0]
    uint32_t linesize = cpuidregs[1] & ((1<<12)-1); //EBX[11:0]
    //This size in bytes
    l1Size = (ways + 1) * (partitions + 1) * (linesize + 1) * (sets + 1);
    l1Size = l1Size / 1024;
  } else {
    //TODO: error
    assert(false);
  }

  //Get L2 and L3 cache size in KB
  uint32_t l2Size = 0, l3Size = 0;
  {
    cpuid(0x80000006, cpuidregs);
    l2Size = (cpuidregs[2] >> 16) & 0xFFFF; //ECX[31:16]
  }
  if (cpuVendor == "AuthenticAMD") {
    cpuid(0x80000006, cpuidregs);
    l3Size = ((cpuidregs[3] >> 18) & 0x3FFF) * 512; //EDX[31:18]
  } else if (cpuVendor == "GenuineIntel") {
    memset(cpuidregs, 0, 4 * sizeof(uint32_t));
    cpuid(0x4, cpuidregs, 3);
    uint32_t ways = (cpuidregs[1] >> 22); //EBX[31:22];
    uint32_t partitions = (cpuidregs[1] >> 12) & ((1 << 10) - 1); //EBX[21:12]
    uint32_t sets = cpuidregs[2]; //ECX[31:0]
    uint32_t linesize = cpuidregs[1] & ((1<<12)-1); //EBX[11:0]
    //This size in bytes
    l3Size = (ways + 1) * (partitions + 1) * (linesize + 1) * (sets + 1);
    l3Size = l3Size / 1024;
  }

  //Get number of cpu sockets
  uint32_t sockets = 0;
  {
    //TODO: Only for linux
    std::set<std::string> socketset;
    std::filesystem::path syscpu = "/sys/devices/system/cpu/";

    if (!std::filesystem::exists(syscpu) or
        !std::filesystem::is_directory(syscpu)) {
      //TODO: What to do?
      std::cout << "Error " << syscpu << " not a directory" << std::endl;
    }

    for (const auto& cpudir : std::filesystem::directory_iterator(syscpu)) {
      std::regex cpuRegex((syscpu/"cpu").string() + "\\d+");

      if (std::filesystem::is_directory(cpudir) &&
          std::regex_search(cpudir.path().string(), cpuRegex)) {

        auto socket = cpudir.path() / "topology/physical_package_id";
        std::ifstream socketfile;
        socketfile.open(socket);
        if (socketfile.is_open()) {
          std::string socketstring;
          socketfile >> socketstring;
          socketset.insert(socketstring);
        } else {
          std::cout << "Error " << socket << " not present " << std::endl;
        }
      }
    }

    sockets = socketset.size();
  }

  __builtin_cpu_init();

  X86SIMD simd = SISD;
  if (__builtin_cpu_supports("fma")) {
    //Has AVX and AVX2
    if (__builtin_cpu_supports("avx2")) {
      simd = X86SIMD::AVX;
    }

    //Has AVX512?
    if (__builtin_cpu_supports("avx512f")) {
      simd = X86SIMD::AVX512;
    }
  }

  auto detail = new X86ArchDetails(cpuVendor, model, l1Size, l2Size, l3Size,
                                   sockets, cores, simd);
  hardware.push_back(detail);

  Logger(LogLevel::Info) << "Detected CPU " << std::endl
                         << (*detail) << std::endl;

  loadKernels<X86KMMKernel>(AllX86Kernels, sizeof(AllX86Kernels)/sizeof(X86KMMKernel));
  trash1 = new char[detail->totalL3Size()];
  trash2 = new char[detail->totalL3Size()];
  for (uint32_t i = 0; i < detail->totalL3Size(); i++) {
    trash1[i] = i % std::numeric_limits<char>::max();
  }

  allocate_caches();
}

template<typename KMMProblemT>
KMMKernel* X86KernelDatabase::findKernelAtOptLevel(KMMProblemT subProblem,
                                                   const std::vector<KMMKernel*>& kernelsForOptLevel) {
  if (kernelsForOptLevel.size() > 0) {
    //Find kernels that have either same P or same Q
    std::vector<KMMKernel*> kernelsWithSamePOrQ;
    std::copy_if(kernelsForOptLevel.begin(), kernelsForOptLevel.end(),
                 std::back_inserter(kernelsWithSamePOrQ),
                 [subProblem](auto& kernel){return kernel->getMaxFactor().p() == subProblem.f(0).p() or
                                            kernel->getMaxFactor().q() == subProblem.f(0).q();});
    std::vector<KMMKernel*> filteredKernels;
    if (kernelsWithSamePOrQ.size() > 0) {
      filteredKernels = kernelsWithSamePOrQ;
    } else {
      filteredKernels = kernelsForOptLevel;
    }
    filteredKernels = filterKernelsForKMM(subProblem, filteredKernels);

    X86SIMD simd = getX86CPUProperties().simd;
    std::vector<KMMKernel*> kernelsForArch;
    std::copy_if(filteredKernels.begin(), filteredKernels.end(),
                 std::back_inserter(kernelsForArch),
                 [simd, subProblem](auto& kernel){
                   return kernel->getFusedFacs() > 1 ||
                   //TODO: write conversion function kernel.asX86Kernel()
                    (((X86KMMKernel*)kernel)->getSIMD() == simd);
                 });
    if (kernelsForArch.size() == 0)
      kernelsForArch = filteredKernels;

    //sort kernels in descending order based on the number of threads a kernel invoke
    auto order = [subProblem, this](auto k1, auto k2) {
      return ((CPUKMMKernel*)k1)->getNumThreads(subProblem) >
             ((CPUKMMKernel*)k2)->getNumThreads(subProblem);
    };
    std::sort(kernelsForArch.begin(), kernelsForArch.end(), order);
    for (auto k : kernelsForArch) {
      if (((CPUKMMKernel*)k)->getNumThreads(subProblem) <= getMaxThreads()) {
        return k;
      }
    }

    //If no kernel is found then return the kernel with max reuse
    return kernelsForArch[kernelsForArch.size() - 1];
  }

  return nullptr;
}