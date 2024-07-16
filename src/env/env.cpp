#include <iostream>

#include <cstring>

#include "env.h"

namespace env {
  #define ENV_FASTKRON(x) "FASTKRON_" x;

  static char DIST_COMM[] = "DIST_COMM";
  static char TUNE[]      = "TUNE";
  static char LOGLEVEL[]  = ENV_FASTKRON("LOG");

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
    if (strcmp(val, "P2P")  == 0) return DistComm::P2P;
    if (strcmp(val, "NCCL") == 0) return DistComm::NCCL;
    std::cout << "Invalid distributed communicator" << val << std::endl;
    return DistComm::DistCommNone;
  }

  LogLevel getLogLevel() {
    char *val = getenv(LOGLEVEL);
    if (val == nullptr) return LogLevel::Nothing;
    if (strcmp(val, "Info")  == 0) return LogLevel::Info;
    if (strcmp(val, "Debug") == 0) return LogLevel::Debug;
    std::cout << "Invalid log level '" << val << "'" << std::endl;
    return LogLevel::Nothing;
  }
}