#ifndef __KRON_H__
#define __KRON_H__

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
  uint NumFusedKernels;

  bool operator==(const KronMatmulShape& other) const {
    return KronCols == other.KronCols && KronRows == other.KronRows &&
    ColsA == other.ColsA && NumFusedKernels == other.NumFusedKernels;
  }

  bool sameKronSize(const KronMatmulShape& other) const {
    return KronCols == other.KronCols && KronRows == other.KronRows;
  }
  // bool operator>(const KronMatmulShape& other) const {
  //   return KronCols > other.KronCols && KronRows > other.KronRows && ColsA > other.ColsA;
  // }

  friend std::ostream& operator<<(std::ostream &out, const KronMatmulShape &shape) {
    out << shape.KronRows << "x" << shape.KronCols << "_" << shape.RowsA << "x" << shape.ColsA << "**" << shape.NumFusedKernels;
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

  //TODO: Add SharedTileKronRows??
  KernelInfo() : kernel(nullptr) {}
  KernelInfo(void* kernel_, uint NumThreads_,  uint KronCols_, uint KronRows_, uint TileKronCols_,
             uint TileRowsA_, uint MaxColsA_, uint CRegRows_, uint CRegCols_, uint NumFusedKerns_,
             ElementType elemType_, bool RowModTileIsZero_, bool KEqVar_) :
             kernel(kernel_), NumThreads(NumThreads_), KronCols(KronCols_), KronRows(KronRows_),
             TileKronCols(TileKronCols_), TileRowsA(TileRowsA_), MaxColsA(MaxColsA_), CRegRows(CRegRows_),
             CRegCols(CRegCols_), NumFusedKerns(NumFusedKerns_), elemType(elemType_), 
             RowModTileIsZero(RowModTileIsZero_), KEqVar(KEqVar_) {}

  bool isValid() {return kernel != nullptr;}
  friend std::ostream& operator<<(std::ostream &out, const KernelInfo &shape) {
    out << shape.TileRowsA << "x" << shape.MaxColsA << "_" 
       << shape.KronRows << "x" << shape.KronCols << "_" << shape.TileKronCols << "_"
       << shape.CRegRows << "x" << shape.CRegCols << "_"
       << shape.NumFusedKerns << "_" << shape.NumThreads << "_" << shape.KEqVar << "_" << shape.RowModTileIsZero;
      
    return out;
  }

  bool canCompute(KronMatmulShape shape, uint NumFusedKerns) {
    return KEqVar == (shape.ColsA == MaxColsA) && RowModTileIsZero == ((shape.RowsA % TileRowsA) == 0) && this->NumFusedKerns == NumFusedKerns;
  }
};

struct FastKronHandle {
  const uint M_, N_, K_;
  const uint* KronMatCols_;
  const uint* KronMatRows_;
  const uint NumKronMats_;
  void* result_;

  FastKronHandle(uint M, uint N, uint K, uint* KronMatCols, uint* KronMatRows, uint NumKronMats) :
    M_(M), N_(N), K_(K), KronMatCols_(KronMatCols), KronMatRows_(KronMatRows), 
    NumKronMats_(NumKronMats), tunedKernel()
  {
    OutofCoreRows_ = 0;
    OutofCoreKrons_ = 0;
    temp_ = NULL;
    outOfCoreTemp1_ = NULL;
    outOfCoreTemp2_ = NULL;

    //Optimization Options
    useFusion_ = true;
  }
  
  void* temp_;
  
  uint numGPUs_;
  uint OutofCoreRows_;
  uint OutofCoreKrons_;
  uint OutofCoreKronBatch_;
  void **outOfCoreTemp1_;
  void **outOfCoreTemp2_;

  void setOutOfCoreRowsCols(uint gpus, uint rows, uint cols, uint batch) {
    OutofCoreRows_ = rows;
    OutofCoreKrons_ = cols;
    OutofCoreKronBatch_ = batch;
    numGPUs_ = gpus;
  }

  template<typename T> void init(bool useUVA);
  void free();

  //Options
  bool useFusion_;
  void setUseFusion(bool v) {useFusion_ = v;}
  bool getUseFusion()       {return useFusion_;}

  KernelInfo tunedKernel;
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
  uint M, uint N, uint K, uint KronMatCols[], uint KronMatRows[], cudaStream_t stream[]);
cudaError_t kronIGEMMOutofCoreX(FastKronHandle& handle, const uint NumKronMats, int* x, int* kronMats[], int** result,
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