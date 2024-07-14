#pragma once

enum DistComm {
  DistCommNone = 0,
  P2P,
  NCCL,
};

enum LogLevel {
  Nothing = 0,
  Info = 1,
  Debug = 2
};

namespace env {
  DistComm getDistComm();
  LogLevel getLogLevel();
}