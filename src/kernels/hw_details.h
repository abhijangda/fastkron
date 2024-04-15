#include <string>
#include <iostream>

#pragma once

class HardwareDetails {
public:
  virtual ~HardwareDetails() {}
};

enum SMArch {
  SMArchNone,
  volta,
  ampere,
};

static std::string smArchToStr(SMArch arch) {
  switch (arch) {
    case SMArch::volta:
      return "volta";
    case SMArch::ampere:
      return "ampere";
  }

  return "";
}

static SMArch computeCapabilityToSMArch(uint major, uint minor) {
  uint32_t c = major * 10 + minor;
  if (c >= 80 && c < 90) {
    return SMArch::ampere;
  } else if (c >= 70 && c < 80) {
    return SMArch::volta;
  } else if (c >= 60 && c < 70) {
    assert(false);
  }
  return SMArch::SMArchNone;
}

class CUDAArchDetails : public HardwareDetails {
public:
  uint32_t numSMs;
  uint32_t maxBlocksPerSM;
  uint32_t maxThreadsPerBlock;
  uint32_t maxThreadsPerSM;
  uint32_t regsPerSM;
  uint32_t maxRegsPerThread;
  uint32_t sharedMemPerSM;
  uint32_t sharedMemPerBlock;
  std::string name;
  uint32_t computeMajor;
  uint32_t computeMinor;
  uint32_t warpSize;
  SMArch smArch;


  // CUDAArchDetail(uint32_t numSMs, uint32_t maxBlocksPerSM, uint32_t maxThreadsPerBlock,
  //                uint32_t maxThreadsPerSM, uint32_t regsPerSM, uint32_t sharedMemPerSM) :
  //                numSMs(numSMs), maxBlocksPerSM(maxBlocksPerSM), 
  //                maxThreadsPerBlock(maxThreadsPerBlock),
  //                maxThreadsPerSM(maxThreadsPerSM), 
  //                regsPerSM(regsPerSM), sharedMemPerSM(sharedMemPerSM) {}
  CUDAArchDetails(int dev);
  
  friend std::ostream& operator<<(std::ostream &out, const CUDAArchDetails& detail) {
    std::string indent = "    ";
    out << detail.name << std::endl <<
          indent << "Compute Capability      : " << (detail.computeMajor*10 + detail.computeMinor) << std::endl <<
          indent << "SMs                     : " << detail.numSMs       << std::endl <<
          indent << "Max Blocks per SM       : " << detail.maxBlocksPerSM << std::endl <<
          indent << "Max Threads per SM      : " << detail.maxThreadsPerSM << std::endl <<
          indent << "Registers Per SM        : " << detail.regsPerSM << std::endl <<
          indent << "Shared Memory per SM    : " << detail.sharedMemPerSM << std::endl<<
          indent << "Shared Memory Per Block : " << detail.sharedMemPerBlock << std::endl <<
          indent << "Warp Size               : " << detail.warpSize << std::endl
          ;
    return out;
  }

  virtual ~CUDAArchDetails() {}
};

enum X86SIMD {
  NoSIMD,
  AVX,
  AVX512
};

static std::string x86simdToStr(X86SIMD simd) {
  switch(simd) {
    case NoSIMD:
      return "NoSIMD";
    case AVX:
      return "AVX";
    case AVX512:
      return "AVX512";
  }
  return "";
}

class X86ArchDetails : public HardwareDetails {
public:
  std::string vendor;
  uint32_t l2Size;
  uint32_t l3Size;
  uint32_t sockets;
  uint32_t cores;
  X86SIMD simd;

  X86ArchDetails(std::string vendor, uint32_t l2Size, uint32_t l3Size, uint32_t sockets, uint32_t cores, X86SIMD simd) :
    vendor(vendor), l2Size(l2Size), l3Size(l3Size), sockets(sockets), cores(cores), simd(simd)
  {}

  friend std::ostream& operator<<(std::ostream& out, const X86ArchDetails& detail) {
    std::string indent = "    ";
    out << indent << "Vendor        : " << detail.vendor  << std::endl
        << indent << "L2 Cache Size : " << detail.l2Size  << std::endl
        << indent << "L3 Cache Size : " << detail.l3Size  << std::endl
        << indent << "Cores         : " << detail.cores   << std::endl
        << indent << "Sockets       : " << detail.sockets << std::endl
        << indent << "SIMD Type     : " << x86simdToStr(detail.simd) << std::endl;
    return out;
  }

  uint32_t totalL3Size() {return l3Size * sockets;}
};