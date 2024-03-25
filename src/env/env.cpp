#include <iostream>

#include <cstring>

#include "env.h"

namespace env {
  static char DIST_COMM[] = "DIST_COMM";
  static char FUSION[]    = "FUSION";
  static char TUNE[]      = "TUNE";

  bool boolFromIntEnv(char* env, bool defaultVal) {
    char* val = getenv(env);
    if (val == nullptr) return defaultVal;
    if (strcmp(val, "0") == 0) return false;
    if (strcmp(val, "1") == 0) return true;
    std::cout << "Invalid value " << env << "=" << val << std::endl;
    return defaultVal;
  }

  DistComm getDistComm() {
    char* val = getenv(DIST_COMM);
    if (val == nullptr) return DistComm::DistCommNone;
    if (strcmp(val, "P2P") == 0) return DistComm::P2P;
    if (strcmp(val, "NCCL") == 0) return DistComm::NCCL;
    std::cout << "Invalid value for DIST_COMM=" << val << std::endl;
    return DistComm::DistCommNone;
  }

  bool getFusion() {
    //Use fusion by default
    return boolFromIntEnv(FUSION, true);
  }

  bool getTune() {
    //DO tuning by default
    return boolFromIntEnv(TUNE, true);
  }
}