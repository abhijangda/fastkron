#include <iostream>

#include <cstring>

#include "env/env.h"
#include "utils/logger.h"

namespace env {
  #define ENV_FASTKRON(x) "FASTKRON_" x;

  static char COMM[]      = ENV_FASTKRON("COMM");
  static char LOGLEVEL[]  = ENV_FASTKRON("LOG");

  char* strupr(char* str) {
    char *s = str;
    while (*s) {
      *s = toupper((unsigned char) *s);
      s++;
    }
    return s;
  }

  /**
   * intEnvToBool() - Convert an integer environment value to boolean.
   * @env: The environment variable.
   * @defaultBool: Default boolean value to return when env var is not defined.
   *
   * Return - True if value of env is 1, False if value of env is 0, and default 
   *          if env is not defined
   */
  bool intEnvToBool(char* env, bool defaultBool) {
    char* val = getenv(env);
    if (val        == nullptr) return defaultBool;
    if (strcmp(val, "0") == 0) return false;
    if (strcmp(val, "1") == 0) return true;
    Logger(LogLevel::Info) << "Invalid " << env << "=" << val << std::endl;
    return defaultBool;
  }

  /**
   * getDistComm() - Get DistComm value from environment value of COMM
   */
  DistComm getDistComm() {
    char* val = getenv(COMM);
    if (val           == nullptr) return DistComm::DistCommNone;
    strupr(val);
    if (strcmp(val, "P2P")  == 0) return DistComm::P2P;
    if (strcmp(val, "NCCL") == 0) return DistComm::NCCL;
    Logger(LogLevel::Info) << "Invalid " << COMM << "=" << val << std::endl;
    return DistComm::DistCommNone;
  }

  /**
   * getLogLevel() - Get LogLevel value from environment value of LOGLEVEL
   */
  LogLevel getLogLevel() {
    char *val = getenv(LOGLEVEL);
    if (val            == nullptr) return LogLevel::Nothing;
    strupr(val);
    if (strcmp(val, "INFO")  == 0) return LogLevel::Info;
    if (strcmp(val, "DEBUG") == 0) return LogLevel::Debug;
    Logger(LogLevel::Info) << "Invalid " << LOGLEVEL << "=" << val << std::endl;
    return LogLevel::Nothing;
  }
}

std::ostream& operator<<(std::ostream &out, DistComm comm) {
  switch (comm) {
    case DistComm::DistCommNone:
      out << "CommNone";
      break;
    case DistComm::P2P:
      out << "P2P";
      break;
    case DistComm::NCCL:
      out << "NCCL";
      break;
  }
  return out;
}