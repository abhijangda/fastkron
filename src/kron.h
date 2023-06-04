#ifndef __KRON_H__
#define __KRON_H__

struct FastKronHandle {
  const uint M_, N_, K_;
  const uint* KronMatCols_;
  const uint* KronMatRows_;
  const uint NumKronMats_;
  void* result_;

  FastKronHandle(uint M, uint N, uint K, uint* KronMatCols, uint* KronMatRows, uint NumKronMats) :
    M_(M), N_(N), K_(K), KronMatCols_(KronMatCols), KronMatRows_(KronMatRows), NumKronMats_(NumKronMats)
  {
    OutofCoreRows_ = 0;
    OutofCoreKrons_ = 0;
    temp_ = NULL;
    outOfCoreTemp1_ = NULL;
    outOfCoreTemp2_ = NULL;
  }
  
  void* temp_;
  
  uint OutofCoreRows_;
  uint OutofCoreKrons_;
  uint OutofCoreKronBatch_;
  void *outOfCoreTemp1_;
  void *outOfCoreTemp2_;

  void setOutOfCoreRowsCols(uint rows, uint cols, uint batch) {
    OutofCoreRows_ = rows;
    OutofCoreKrons_ = cols;
    OutofCoreKronBatch_ = batch;
  }

  template<typename T> void init(bool useUVA);
  void free();
};

cudaError_t kronSGEMM(FastKronHandle& handle, const uint NumKronMats, float* x, float* kronMats[], float** result,
                      uint M, uint N, uint K, uint KronMatCols[], uint KronMatRows[], cudaStream_t stream);

cudaError_t kronIGEMM(FastKronHandle& handle, const uint NumKronMats, int* x, int* kronMats[], int** result,
                      uint M, uint N, uint K, uint KronMatCols[], uint KronMatRows[], cudaStream_t stream);

cudaError_t kronDGEMM(FastKronHandle& handle, const uint NumKronMats, double* x, double* kronMats[], double** result,
  uint M, uint N, uint K, uint KronMatCols[], uint KronMatRows[], cudaStream_t stream);

cudaError_t kronSGEMMOutofCore(FastKronHandle& handle, const uint NumKronMats, float* x, float* kronMats[], float** result,
                               uint M, uint N, uint K, uint KronMatCols[], uint KronMatRows[], cudaStream_t stream);
cudaError_t kronSGEMMOutofCoreX(FastKronHandle& handle, const uint NumKronMats, float* x, float* kronMats[], float** result,
  uint M, uint N, uint K, uint KronMatCols[], uint KronMatRows[], cudaStream_t stream);
cudaError_t kronIGEMMOutofCoreX(FastKronHandle& handle, const uint NumKronMats, int* x, int* kronMats[], int** result,
  uint M, uint N, uint K, uint KronMatCols[], uint KronMatRows[], cudaStream_t stream);

#endif