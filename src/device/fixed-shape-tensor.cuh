
// template<typename T, uint32_t Dims>
// class FixedShapeTensor {
//   using ElemT = T;
//   T* ptr;

// public:
//   CUDA_DEVICE_HOST
//   FixedShapeTensor(T* ptr): ptr(ptr) {}
//   CUDA_DEVICE_HOST
//   uint32_t dims()             {return Dims;}
//   CUDA_DEVICE_HOST
//   T& at(const uint32_t dim[Dims], const uint32_t size[Dims]) {
//     uint32_t id = dim[0];
//     #pragma unroll
//     for (uint32_t i = 1; i < dims(); i++) {
//       id += id * size[i] + dim[i];
//     }

//     return ptr[id];
//   }
//   CUDA_DEVICE_HOST
//   void set(const uint32_t dim[Dims], const uint32_t size[Dims], T val) {
//     at(dim, size) = val;
//   }
// };

template<typename T, uint32_t M, uint32_t N>
class FixedShapeTensor2D {
  T data[M][N];

public:
  CUDA_DEVICE_HOST
  FixedShapeTensor2D() {}

  CUDA_DEVICE_HOST
  uint32_t size(uint32_t dim) const {
    switch(dim) {
      case 0:
        return M;
      case 1:
        return N;
      default:
        return 0;
    }
  }

  CUDA_DEVICE_HOST
  T& at(uint32_t i, uint32_t j) {
    return data[i][j];
  }

  CUDA_DEVICE_HOST
  void set(uint32_t i, uint32_t j, T val) {
    data[i][j] = val;
  }
};

template<typename T, uint32_t M, uint32_t N, uint32_t K>
class FixedShapeTensor3D {
  T data[M][N][K];

public:
  CUDA_DEVICE_HOST
  FixedShapeTensor3D() {}

  CUDA_DEVICE_HOST
  uint32_t size(uint32_t dim) const {
    switch(dim) {
      case 0:
        return M;
      case 1:
        return N;
      case 2:
        return K;
      default:
        return 0;
    }
  }

  CUDA_DEVICE_HOST
  void zero() {
    #pragma unroll
    for (uint i = 0; i < size(0); i++) {
    #pragma unroll
    for (uint j = 0; j < size(1); j++) {
    #pragma unroll
    for (uint k = 0; k < size(2); k++) {
      set(i, j, k, (T)0);
    }}}
  }

  CUDA_DEVICE_HOST
  T& at(uint32_t i, uint32_t j, uint32_t k) {
    return data[i][j][k];
  }

  CUDA_DEVICE_HOST
  T& add(uint32_t i, uint32_t j, uint32_t k, T val) {
    set(i, j, k, at(i, j, k) + val);
  }

  CUDA_DEVICE_HOST
  void set(uint32_t i, uint32_t j, uint32_t k, T val) {
    data[i][j][k] = val;
  }
};

//Shared Memory Tensors
template<typename T, uint32_t TileP, uint32_t TileQ>
class DirectShared : public FixedShapeTensor2D<T, TileP, TileQ> {
  using Base = FixedShapeTensor2D<T, TileP, TileQ>;

public:
  CUDA_DEVICE_HOST
  DirectShared() {}

  CUDA_DEVICE_HOST
  //TODO: Make this Coord1D
  void store(uint32_t eIdx, uint32_t num, const T* elems) {
    #pragma unroll
    for (uint ve = 0; ve < num; ve++) {
      uint idx = eIdx + ve;
      Base::set(idx/Base::size(1), idx%Base::size(1), elems[ve]);
    }
  }

  CUDA_DEVICE_HOST
  //TODO: Make this Coord2D
  void store(uint32_t row, uint32_t col, uint32_t num, const T* elems) {
    #pragma unroll
    for (uint ve = 0; ve < num; ve++) {
      Base::set(row, col + ve, elems[ve]);
    }
  }
  
  CUDA_DEVICE_HOST
  T& at(uint32_t row, uint32_t col) {
    return Base::at(row, col);
  }

  CUDA_DEVICE_HOST
  uint32_t p() const {return Base::size(0);}
  CUDA_DEVICE_HOST
  uint32_t q() const {return Base::size(1);}
};

template<typename ElemT>
class ShiftShared : public Matrix {
public:
  CUDA_DEVICE_HOST
  ShiftShared(uint32_t rows, uint32_t cols, void* ptr) :
    Matrix(rows, cols, ptr) {}

  CUDA_DEVICE_HOST
  void store(uint32_t row, uint32_t startCol, uint32_t TileP, uint32_t RegK, 
             uint32_t numElems, ElemT* elems) {
    #pragma unroll
    for (uint i = 0; i < numElems; i++) {
      uint32_t shCol = startCol + i;
      uint32_t elem  = shCol%TileP;
      uint32_t slice = shCol/TileP;
      uint32_t shift = slice/RegK;

      set<ElemT>(row, slice*TileP + (shift + elem)%TileP, elems[i]);
    }
  }

  CUDA_DEVICE_HOST
  ElemT at(uint32_t row, uint32_t col) {
    return Matrix::at<ElemT>(row, col);
  }
};

//Register Tensors
template<typename T, uint32_t M, uint32_t K, uint32_t Q>
class YRegisters : public FixedShapeTensor3D<T, M, K, Q> {
  using Base = FixedShapeTensor3D<T, M, K, Q>;
public:
  //TODO: Make this Coord2D inside kernel outside of this struct
  uint32_t yK;
  uint32_t yQ;

public:
  CUDA_DEVICE_HOST
  YRegisters(uint32_t yK, uint32_t yQ) : yQ(yQ), yK(yK) {Base::zero();}
  
  CUDA_DEVICE_HOST
  uint32_t m()  {return M;}
  CUDA_DEVICE_HOST
  uint32_t k() {return K;}
  CUDA_DEVICE_HOST
  uint32_t q() {return Q;}
};

template<typename T, uint32_t M, uint32_t K, uint32_t P>
class XRegisters : public FixedShapeTensor3D<T, M, K, P>{
public:
  CUDA_DEVICE_HOST
  XRegisters() {}

  CUDA_DEVICE_HOST
  uint32_t m() {return M;}
  CUDA_DEVICE_HOST
  uint32_t k() {return K;}
  CUDA_DEVICE_HOST
  uint32_t p() {return P;}
};

template<typename T, uint32_t TileP, uint32_t CRegCols>
class FRegisters : public FixedShapeTensor2D<T, TileP, CRegCols>{
  public:
    CUDA_DEVICE_HOST
    FRegisters() {}
};