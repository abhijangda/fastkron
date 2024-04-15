#include <filesystem>
#include <fstream>
#include <regex>
#include <set>
#include <immintrin.h>

#include "kernels/params.h"
#include "kernel_db/cpu_kernel_db.h"

#ifdef ENABLE_X86
  #include "kernels/cpu/x86/kron-kernels/kernel_decl.inc"
#endif

X86Kernel AllX86Kernels[] = {
#ifdef ENABLE_X86
  ALL_X86_KERNELS
#endif
};

CPUKernelDatabase::CPUKernelDatabase() : KernelDatabase() {}

template<uint FusedFacs>
fastKronError invoke(CPUKernel& kernelInfo, const uint kronIndex, 
                   KMMProblem problem,
                   DistributedParams distParams,
                   EpilogueParams epilogueParams,
                   KernelMode execMode) {
  //Create the grid and thread block
  KernelParams<FusedFacs> params (problem, kernelInfo.tileX, kernelInfo.tileF, kronIndex, execMode);
  FusedParams<FusedFacs> fusedParams (problem, kernelInfo.tileX.n());

  //Call kernel
  typedef void (*KronMatmulKernelTy)(KernelParams<FusedFacs>, FusedParams<FusedFacs>, 
                                     DistributedParams, EpilogueParams);
  KronMatmulKernelTy(kernelInfo.invokerFunc)(params, fusedParams, distParams, epilogueParams);
  return fastKronSuccess;
}

fastKronError CPUKernelDatabase::invokeKernel(KernelInfo* kernel, const uint kronIndex, 
                                            KMMProblem problem,
                                            EpilogueParams epilogueParams,
                                            KernelMode execMode) {
  DistributedParams distParams;
  CPUKernel& cpuKernel = dynamic_cast<CPUKernel&>(*kernel);

  switch(problem.n()) {
    case 1:
      return invoke<1>(cpuKernel, kronIndex, problem,
                       distParams, epilogueParams, execMode);
    case 2:
      return invoke<2>(cpuKernel, kronIndex, problem,
                       distParams, epilogueParams, execMode);
    case 3:
      return invoke<3>(cpuKernel, kronIndex, problem,
                       distParams, epilogueParams, execMode);
    case 4:
      return invoke<4>(cpuKernel, kronIndex, problem,
                       distParams, epilogueParams, execMode);
    case 5:
      return invoke<5>(cpuKernel, kronIndex, problem,
                       distParams, epilogueParams, execMode);
    case 6:
      return invoke<6>(cpuKernel, kronIndex, problem, 
                       distParams, epilogueParams, execMode);
    default:
      std::cout << "Invalid number of fused kernels" << std::endl;
      return fastKronKernelNotFound;
  }
}

fastKronError CPUKernelDatabase::procMalloc(uint32_t proc, size_t size, void*& ptr) {
  ptr = new float[size];
  return ptr != nullptr ? fastKronSuccess : fastKronInvalidArgument; 
}

fastKronError CPUKernelDatabase::procMemset(uint32_t proc, Matrix& m, float val) {
  memset<float>(m.data<float>(0), m.numel(), val);
  return fastKronSuccess;
}

fastKronError CPUKernelDatabase::procFree(uint32_t proc, void* ptr) {
  if (ptr == NULL) fastKronInvalidArgument;
  delete ptr;
  return fastKronSuccess;
}

//f_128x128_32x128_1_1x8192_1x16x4_NN_0_0
//f_128x128_128x128_1_1x8192_1x16x4_NN_0_0
//f_128x128_32x128_1_1x16384_1x16x4_NN_0_0
fastKronError CPUKernelDatabase::timeKernel(KernelInfo* kernel, const uint factorIdx, 
                                 KMMProblem problem, DistributedParams distParams, 
                                 EpilogueParams epilogueParams,
                                 KernelMode execMode, 
                                 bool distP2PStore,
                                 int warmups, int runs,
                                 float& runtime) {
  runtime = std::numeric_limits<float>::max();
  fastKronError status;
  for (int sample = 0; sample < 10; sample++) {
    float avgtime = 0;
    for (int r = 0; r < runs; r++) {
      //Trash L3 Cache
      uint32_t l3size = ((X86ArchDetails*)hardware[0])->totalL3Size();
      if ((problem.x().numel() + problem.y().numel()) * sizeof(float) <= l3size)
        parallelCopy(trash1, trash2, l3size);
      double startTime = getCurrTime();
      if (distP2PStore) {
        status = invokeP2PStoreKernel(kernel, factorIdx, problem,
                                      distParams, epilogueParams, execMode);
      } else {
        status = invokeKernel(kernel, factorIdx, problem,
                              epilogueParams, execMode);
      }
      double endTime = getCurrTime();
      avgtime += (float)((endTime - startTime)/1e3);
    }

    if (status != fastKronSuccess) {
      std::cout << "Error: " << status << std::endl;
      return status;
    }

    runtime = std::min(avgtime/runs, runtime);
  }
  
  return status;
}

void cpuid(uint32_t in, uint32_t regs[4]) {
#ifdef _WIN32
#else
  __asm__ (
    "cpuid" : "=a" (regs[0]), "=b" (regs[1]), "=c" (regs[2]), "=d" (regs[3]) :
    "a"(in), "c"(0)
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

  //Get L1 cache size in KB
  uint32_t l1Size = 0;
  {
    cpuid(0x80000005, cpuidregs);
    l1Size = (cpuidregs[2] >> 24) & 0xFF; //ECX[31:24]
  }

  //Get L2 and L3 cache size in KB
  uint32_t l2Size = 0, l3Size = 0;
  {
    cpuid(0x80000006, cpuidregs);
    l2Size = (cpuidregs[2] >> 16) & 0xFFFF; //ECX[31:16]
    l3Size = ((cpuidregs[3] >> 18) & 0x3FFF) * 512; //EDX[31:18]
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

  X86SIMD simd = NoSIMD;
  {
    //Has AVX?
    cpuid(1, cpuidregs);
    if ((cpuidregs[2] >> 28) & 0x1) {
      simd = X86SIMD::AVX;
    }

    //Has AVX2?
    cpuid(0x7, cpuidregs);
    if ((cpuidregs[2] >> 5) & 0x1) {
      // simd = X86SIMD::AVX2;
    }

    //Has AVX512?
    cpuid(0x7, cpuidregs);
    if ((cpuidregs[2] >> 16) & 0x1) {
      simd = X86SIMD::AVX512;
    }
  }

  auto detail = new X86ArchDetails(cpuVendor, l1Size, l2Size, l3Size, 
                                   sockets, cores, simd);
  hardware.push_back(detail);

  std::cout << "Detected CPU " << std::endl << (*detail) << std::endl;

  loadKernels<CPUKernel>(AllX86Kernels, sizeof(AllX86Kernels)/sizeof(X86Kernel));
  trash1 = new char[detail->totalL3Size()];
  trash2 = new char[detail->totalL3Size()];
  for (int i = 0; i < detail->totalL3Size(); i++) {
    trash1[i] = i % std::numeric_limits<char>::max();
  }
}