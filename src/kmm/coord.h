#include "config.h"

// template<uint32_t Dim>
// class Coord {
//   uint32_t val[Dim];

// protected:
//   Coord(uint32_t... args) {
//     for (int i = 0; i < Dim; i++) {
//       this->val[i] = val[i];
//     }
//   }

//   uint32_t value(uint32_t i) const {return val[i];}
// };

class Coord2D {
  uint32_t val[2];

public:
  CUDA_DEVICE_HOST
  Coord2D(uint32_t i, uint32_t j) {
    val[0] = i;
    val[1] = j;
  }

  CUDA_DEVICE_HOST
  uint32_t i() const {return val[0];}
  CUDA_DEVICE_HOST
  uint32_t j() const {return val[1];}
};