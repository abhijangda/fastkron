#pragma once

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
  uint AAlignment;
  uint KronAlignment;

  //TODO: Add SharedTileKronRows??
  KernelInfo() : kernel(nullptr) {}
  KernelInfo(void* kernel_, uint NumThreads_,  uint KronCols_, uint KronRows_, uint TileKronCols_,
             uint TileRowsA_, uint MaxColsA_, uint CRegRows_, uint CRegCols_, uint NumFusedKerns_,
             ElementType elemType_, bool RowModTileIsZero_, bool KEqVar_, bool DistributeToGPUs_,
             uint AAlignment_, uint KronAlignment_) :
             kernel(kernel_), NumThreads(NumThreads_), KronCols(KronCols_), KronRows(KronRows_),
             TileKronCols(TileKronCols_), TileRowsA(TileRowsA_), MaxColsA(MaxColsA_), CRegRows(CRegRows_),
             CRegCols(CRegCols_), NumFusedKerns(NumFusedKerns_), elemType(elemType_), 
             RowModTileIsZero(RowModTileIsZero_), KEqVar(KEqVar_), DistributeToGPUs(DistributeToGPUs_),
             AAlignment(AAlignment_), KronAlignment(KronAlignment_) {}

  bool isValid() {return kernel != nullptr;}
  friend std::ostream& operator<<(std::ostream &out, const KernelInfo &shape) {
    out << shape.NumThreads << "_" << shape.KronCols << "x" << shape.KronRows <<
           "_" << shape.TileKronCols << "_" << 
           shape.TileRowsA << "x" << shape.MaxColsA << "_" <<
           shape.CRegRows << "x" << shape.CRegCols << "_" <<
           shape.NumFusedKerns << "_" << shape.RowModTileIsZero << "_" << 
           shape.KEqVar << "_" << shape.DistributeToGPUs << "_" << 
           shape.AAlignment << "_" << shape.KronAlignment;
      
    return out;
  }

  bool canCompute(KronMatmulShape shape) {
    return RowModTileIsZero == ((shape.RowsA % TileRowsA) == 0) &&
           this->NumFusedKerns == shape.NumFusedKerns &&
           this->DistributeToGPUs == shape.DistributeToGPUs &&
           shape.ColsA % MaxColsA == 0;
  //KEqVar == (shape.ColsA == MaxColsA) && 
  }

  bool isDistributedLike(KernelInfo& other) {
    return KEqVar == other.KEqVar && 
           RowModTileIsZero == other.RowModTileIsZero &&
           NumFusedKerns == other.NumFusedKerns &&
           MaxColsA == other.MaxColsA && DistributeToGPUs == true;
  }
};