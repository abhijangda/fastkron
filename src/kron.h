#include <vector>
#include <nccl.h>
#include "thread_pool.h"

#ifndef __KRON_H__
#define __KRON_H__


#define NCCLCHECK(cmd) do {                         \
  ncclResult_t r = cmd;                             \
  if (r!= ncclSuccess) {                            \
    printf("Failed, NCCL error %s:%d '%s'\n",             \
        __FILE__,__LINE__,ncclGetErrorString(r));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

enum ElementType {
  Float,
  Double,
  Int,
  Long
};


//TODO: Change this to SlicedMatMulShape
//TODO: Add NumFusedKernels also as a parameter to KronmatmulShape for compiledKernels map
//TODO: Add array of NumFusedKernels of KronCols and KronRows
struct KronMatmulShape {
  uint KronCols;
  uint KronRows;
  uint ColsA;
  uint RowsA;
  uint NumFusedKerns;
  bool DistributeToGPUs;

  bool operator==(const KronMatmulShape& other) const {
    return KronCols == other.KronCols && KronRows == other.KronRows &&
    ColsA == other.ColsA && 
    NumFusedKerns == other.NumFusedKerns &&
    DistributeToGPUs == other.DistributeToGPUs;
  }

  bool sameKronSize(const KronMatmulShape& other) const {
    return KronCols == other.KronCols && KronRows == other.KronRows;
  }
  // bool operator>(const KronMatmulShape& other) const {
  //   return KronCols > other.KronCols && KronRows > other.KronRows && ColsA > other.ColsA;
  // }

  friend std::ostream& operator<<(std::ostream &out, const KronMatmulShape &shape) {
    out << shape.KronRows << "x" << shape.KronCols << "_" << shape.RowsA << "x" << shape.ColsA << "**" << shape.NumFusedKerns << "_" << shape.DistributeToGPUs;
    return out;
  }
};

struct KernelInfo {
  void* kernel;
  uint NumThreads;
  uint KronCols;
  uint KronRows;
  uint TileKronCols;
  uint TileRowsA;
  uint MaxColsA;
  uint CRegRows;
  uint CRegCols;
  uint NumFusedKerns;
  ElementType elemType;
  bool RowModTileIsZero;
  bool KEqVar;
  bool DistributeToGPUs;

  //TODO: Add SharedTileKronRows??
  KernelInfo() : kernel(nullptr) {}
  KernelInfo(void* kernel_, uint NumThreads_,  uint KronCols_, uint KronRows_, uint TileKronCols_,
             uint TileRowsA_, uint MaxColsA_, uint CRegRows_, uint CRegCols_, uint NumFusedKerns_,
             ElementType elemType_, bool RowModTileIsZero_, bool KEqVar_, bool DistributeToGPUs_) :
             kernel(kernel_), NumThreads(NumThreads_), KronCols(KronCols_), KronRows(KronRows_),
             TileKronCols(TileKronCols_), TileRowsA(TileRowsA_), MaxColsA(MaxColsA_), CRegRows(CRegRows_),
             CRegCols(CRegCols_), NumFusedKerns(NumFusedKerns_), elemType(elemType_), 
             RowModTileIsZero(RowModTileIsZero_), KEqVar(KEqVar_), DistributeToGPUs(DistributeToGPUs_) {}

  bool isValid() {return kernel != nullptr;}
  friend std::ostream& operator<<(std::ostream &out, const KernelInfo &shape) {
    out << shape.NumThreads << "_" << shape.KronCols << "x" << shape.KronRows <<
           "_" << shape.TileKronCols << "_" << 
           shape.TileRowsA << "x" << shape.MaxColsA << "_" <<
           shape.CRegRows << "x" << shape.CRegCols << "_" <<
           shape.NumFusedKerns << "_" << shape.RowModTileIsZero << "_" << 
           shape.KEqVar << "_" << shape.DistributeToGPUs;
      
    return out;
  }

  bool canCompute(KronMatmulShape shape) {
    return RowModTileIsZero == ((shape.RowsA % TileRowsA) == 0) &&
           this->NumFusedKerns == shape.NumFusedKerns &&
           this->DistributeToGPUs == shape.DistributeToGPUs &&
           MaxColsA <= shape.ColsA;
  //KEqVar == (shape.ColsA == MaxColsA) && 
  }

  bool isDistributedLike(KernelInfo& other) {
    return KEqVar == other.KEqVar && 
           RowModTileIsZero == other.RowModTileIsZero &&
           NumFusedKerns == other.NumFusedKerns &&
           MaxColsA == other.MaxColsA && DistributeToGPUs == true;
  }
};


struct TunedKernelFromStart {
  KernelInfo kernel;
  uint start, end;
  uint K;
  float time;

  TunedKernelFromStart(KernelInfo kernel_, uint start_, uint end_, uint K_, float time_):
    kernel(kernel_), start(start_), end(end_), K(K_), time(time_) {}
  TunedKernelFromStart() {}
  friend std::ostream& operator<<(std::ostream &out, const TunedKernelFromStart &k) {
    out << "[" << k.start << ", " << k.end << "] = " << k.K << " " << k.kernel << " runs for " << k.time << " ms";
    return out;
  }
};

typedef std::vector<TunedKernelFromStart> TunedKernelsSeries;
struct ThreadArgs;
enum DistComm {
  DistCommNone = 0,
  P2P,
  NCCL,
};

enum ProcType {
  ProcNone = 0,
  SingleThread,
  MultipleThread,
  MultipleProcess
};

struct FastKronHandle {
  const uint M_, N_, K_;
  const uint* KronMatCols_;
  const uint* KronMatRows_;
  const uint NumKronMats_;
  void* result_;

  FastKronHandle(uint M, uint N, uint K, uint* KronMatCols, uint* KronMatRows, uint NumKronMats) :
    M_(M), N_(N), K_(K), KronMatCols_(KronMatCols), KronMatRows_(KronMatRows), 
    NumKronMats_(NumKronMats), tunedKernelSeries()
  {
    gpuM_ = 0;
    gpuK_ = 0;
    temp_ = NULL;
    gpuTemp1_ = NULL;
    gpuTemp2_ = NULL;

    //Optimization Options
    useFusion_ = true;

    //Is Distributed
    isDistributed_ = false;
  }

  void* temp_;
  
  uint numGPUs_;
  uint gpuM_;
  uint gpuK_;
  uint gpusInM_;
  uint gpusInK_;
  uint perGPUKronBatch_;
  void **gpuTemp1_;
  void **gpuTemp2_;
  void **sendTemps_;
  void **recvTemps_;
  bool isDistributed_;
  DistComm distComm_;

  pthread_barrier_t* barriers_;
  thread_pool<ThreadArgs*>* threads_;

  template<typename T> void init();
  template<typename T> void initDistributed(int gpus, int gpusInM = -1, int gpusInK = -1, int gpuLocalKrons = -1);
  //TODO: these two functions should be a part of utils?
  template<typename T> cudaError_t allocDistributedX(T* dX[], T* hX);
  template<typename T> cudaError_t gatherDistributedY(T* dY[], T* hY);

  void free();

  //Options
  bool useFusion_;
  void setUseFusion(bool v) {useFusion_ = v;}
  bool getUseFusion()       {return useFusion_;}

  TunedKernelsSeries tunedKernelSeries;
  
  std::vector<ncclComm_t> ncclComms;

  void clean() {
    for (int i=0; i<ncclComms.size(); i++)
      ncclCommDestroy(ncclComms[i]);
  }
};

struct ThreadArgs {
  ThreadArgs() {}
  ThreadArgs(FastKronHandle* handle, uint NumKronMats, void* x, void** kronMats, void* result, 
            uint M, uint N, uint K, uint *KronMatCols, uint *KronMatRows, cudaStream_t* stream,
            uint gpuRow, uint gpuCol, uint gpusInM_, uint gpusInK_, pthread_barrier_t* barrier) : 
            handle(handle), NumKronMats(NumKronMats), x(x), kronMats(kronMats), result(result),
            M(M), N(N), K(K), KronMatCols(KronMatCols), KronMatRows(KronMatRows), stream(stream),
            gpuRow(gpuRow), gpuCol(gpuCol), gpusInM_(gpusInM_), gpusInK_(gpusInK_), barrier(barrier) {}

  FastKronHandle* handle;
  uint NumKronMats;
  void* x;
  void** kronMats;
  void* result;
  uint M;
  uint N;
  uint K;
  uint *KronMatCols;
  uint *KronMatRows;
  cudaStream_t* stream;
  uint gpuRow;
  uint gpuCol;
  uint gpusInM_;
  uint gpusInK_;
  pthread_barrier_t* barrier;

  struct ThreadResult {
    cudaError_t status;
    void* result;
  } threadResult;
};

//TODO: modify such that the results are always written to the supplied result pointer 
cudaError_t kronSGEMM(FastKronHandle& handle, const uint NumKronMats, float* x, float* kronMats[], float** result,
                      uint M, uint N, uint K, uint KronMatCols[], uint KronMatRows[], cudaStream_t stream);

cudaError_t kronIGEMM(FastKronHandle& handle, const uint NumKronMats, int* x, int* kronMats[], int** result,
                      uint M, uint N, uint K, uint KronMatCols[], uint KronMatRows[], cudaStream_t stream);

cudaError_t kronDGEMM(FastKronHandle& handle, const uint NumKronMats, double* x, double* kronMats[], double** result,
                      uint M, uint N, uint K, uint KronMatCols[], uint KronMatRows[], cudaStream_t stream);

cudaError_t kronSGEMMOutofCore(FastKronHandle& handle, const uint NumKronMats, float* x, float* kronMats[], float** result,
                               uint M, uint N, uint K, uint KronMatCols[], uint KronMatRows[], cudaStream_t stream);
cudaError_t kronSGEMMOutofCoreX(FastKronHandle& handle, const uint NumKronMats, float* x, float* kronMats[], float** result,
  uint M, uint N, uint K, uint KronMatCols[], uint KronMatRows[], cudaStream_t stream[]);
cudaError_t kronIGEMMOutofCoreX(FastKronHandle& handle, const uint NumKronMats, int* x, int* kronMats[], int** result,
  uint M, uint N, uint K, uint KronMatCols[], uint KronMatRows[], cudaStream_t stream[]);

cudaError_t kronDistributedSGEMM(FastKronHandle& handle, const uint NumKronMats, float* x[], float* kronMats[], float* result[],
                                 uint M, uint N, uint K, uint KronMatCols[], uint KronMatRows[], cudaStream_t stream[]);

cudaError_t kronSGEMMTune(FastKronHandle& handle, const uint NumKronMats, float* x, float* kronMats[], 
                          uint M, uint N, uint K, uint KronMatCols[], uint KronMatRows[],
                          cudaStream_t stream);
cudaError_t kronDGEMMTune(FastKronHandle& handle, const uint NumKronMats, double* x, double* kronMats[], 
                          uint M, uint N, uint K, uint KronMatCols[], uint KronMatRows[],
                          cudaStream_t stream);
cudaError_t kronIGEMMTune(FastKronHandle& handle, const uint NumKronMats, int* x, int* kronMats[],
                          uint M, uint N, uint K, uint KronMatCols[], uint KronMatRows[],
                          cudaStream_t stream);
#endif