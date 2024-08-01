#include <cassert>
#include <cstring>

#include "handle/handle.h"

fastKronError distributedKronMatmul(FastKronHandle&, const uint, void*[], void* [], void* [],
                                  uint , uint , uint , uint [], uint [], void** , void** ,
                                  void* ) {
  std::cout << "Not implemented" << std::endl;
  assert(false);
  return fastKronSuccess;
}

fastKronError FastKronHandle::allocDistributedX(void* [], void* , uint , uint ) {
  std::cout << "Not implemented" << std::endl;
  return fastKronSuccess;
}

fastKronError FastKronHandle::gatherDistributedY(void* [], void* , uint , uint , uint , uint [], uint []) {
  //TODO: Make FastKronError type
  std::cout << "Not implemented" << std::endl;
  return fastKronSuccess;
}

fastKronError FastKronHandle::distributedsgekmm(const uint NumKronMats, float* x[], float* kronMats[], float* result[],
  uint M, uint N, uint K, uint KronMatCols[], uint KronMatRows[], float** temp1, float** temp2,
  void* streams) {
    return distributedKronMatmul(*this, NumKronMats, (void**)x, (void**)kronMats, (void**)result, M, N, K, 
      KronMatCols, KronMatRows, (void**)temp1, (void**)temp2, streams);
}
