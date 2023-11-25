#pragma once

namespace env {
  static char DIST_COMM[] = "DIST_COMM";
  static char FUSION[]    = "FUSION";

  DistComm getDistComm() {
    char* val = getenv(DIST_COMM);
    if (val == nullptr) return DistComm::DistCommNone;
    if (strcmp(val, "P2P") == 0) return DistComm::P2P;
    if (strcmp(val, "NCCL") == 0) return DistComm::NCCL;
    std::cout << "Invalid value for DIST_COMM=" << val << std::endl;
    return DistComm::DistCommNone;
  }

  bool getFusion() {
    char* val = getenv(FUSION);
    //Use fusion by default
    if (val == nullptr) return true;
    if (strcmp(val, "0") == 0) return false;
    if (strcmp(val, "1") == 0) return true;
    std::cout << "Invalid value for FUSION=" << val << std::endl;
    return true;
  }
}