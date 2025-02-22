#pragma once

enum DistComm {
  DistCommNone = 0,
  P2P,
  NCCL,
};


std::ostream& operator<<(std::ostream &out, DistComm comm);

enum LogLevel {
  Nothing = 0,
  Info = 1,
  Debug = 2
};

namespace env {
  DistComm getDistComm();
  LogLevel getLogLevel();
  bool     getUseTune();
}