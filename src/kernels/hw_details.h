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
  
  friend std::ostream& operator<<(std::ostream &out, const CUDAArchDetails &detail) {
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