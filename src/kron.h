#ifndef __KRON_H__
#define __KRON_H__

struct FastKronHandle {
  const uint M_, N_, K_;
  const uint* KronMatCols_;
  const uint* KronMatRows_;
  const uint NumKronMats_;
  void* result_;

  FastKronHandle(uint M, uint N, uint K, uint* KronMatCols, uint* KronMatRows, uint NumKronMats, void* result) :
    M_(M), N_(N), K_(K), KronMatCols_(KronMatCols), KronMatRows_(KronMatRows), NumKronMats_(NumKronMats), result_(result)
  {
    OutofCoreRows_ = 0;
    OutofCoreCols_ = 0;
    temp_ = NULL;
    outOfCoreTemp1_ = NULL;
    outOfCoreTemp2_ = NULL;
  }
  
  void* temp_;
  
  uint OutofCoreRows_;
  uint OutofCoreCols_;
  void *outOfCoreTemp1_;
  void *outOfCoreTemp2_;

  void setOutOfCoreRowsCols(uint rows, uint cols) {
    OutofCoreRows_ = rows;
    OutofCoreCols_ = cols;
  }

  template<typename T> void initTemps(bool useUVA);
  void free();
};

cudaError_t kronSGEMM(const uint NumKronMats, float* kronGemmResults[], float* x, float* kronMats[], float** result,
                      uint M, uint N, uint K, uint KronMatCols[], uint KronMatRows[], cudaStream_t stream);

cudaError_t kronIGEMM(const uint NumKronMats, int* kronGemmResults[], int* x, int* kronMats[], int** result,
                      uint M, uint N, uint K, uint KronMatCols[], uint KronMatRows[], cudaStream_t stream);

cudaError_t kronDGEMM(const uint NumKronMats, double* kronGemmResults[], double* x, double* kronMats[], double** result,
  uint M, uint N, uint K, uint KronMatCols[], uint KronMatRows[], cudaStream_t stream);

cudaError_t kronSGEMMOutofCore(const uint NumKronMats, float* kronGemmResults[], float* x, float* kronMats[], float** result,
                               uint M, uint N, uint K, uint KronMatCols[], uint KronMatRows[], cudaStream_t stream);
cudaError_t kronSGEMMOutofCoreX(const uint NumKronMats, float* kronGemmResults[], float* x, float* kronMats[], float** result,
  uint M, uint N, uint K, uint KronMatCols[], uint KronMatRows[], uint OnGPURows, uint MaxInnerKrons, cudaStream_t stream);

#endif