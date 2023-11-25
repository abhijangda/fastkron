#pragma once

enum DistComm {
  DistCommNone = 0,
  P2P,
  NCCL,
};

namespace env {
  DistComm getDistComm();
  bool getFusion();
}