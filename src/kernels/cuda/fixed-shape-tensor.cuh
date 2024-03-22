#include "kmm/coord.h"

template<fastKronOp Layout, typename T, uint32_t M, uint32_t N>
class AbstractFixedShapeTensor2D {
public:
  CUDA_DEVICE_HOST
  AbstractFixedShapeTensor2D() {}

  CUDA_DEVICE_HOST
  static uint32_t numel() {return M*N;}

  CUDA_DEVICE_HOST
  uint32_t shape(uint32_t dim) const {
    if (Layout == fastKronOp_N) {
      switch(dim) {
        case  0: return M;
        case  1: return N;
        default: return 0;
      }
    } else if (Layout == fastKronOp_T) {
      switch(dim) {
        case  0: return N;
        case  1: return M;
        default: return 0;
      }
    }
  }

  CUDA_DEVICE_HOST
  uint32_t linearIdx(uint32_t i, uint32_t j) {
    if (Layout == fastKronOp_N)
      return i * shape(1) + j;
    else
      return j * shape(1) + i;
  }

  CUDA_DEVICE_HOST
  T& at(T data[], uint32_t i, uint32_t j) {
    return data[linearIdx(i, j)];
  }

  CUDA_DEVICE_HOST
  void set(T data[], uint32_t i, T val) {
    data[i] = val;
  }

  CUDA_DEVICE_HOST
  void set(T data[], uint32_t i, uint32_t j, T val) {
    data[linearIdx(i, j)] = val;
  }
};

template<fastKronOp Layout, typename T, uint32_t M, uint32_t N>
class FixedShapeTensor2D : public AbstractFixedShapeTensor2D<Layout, T, M, N> {
  using Base = AbstractFixedShapeTensor2D<Layout, T, M, N>;
  T data[M*N];

public:
  CUDA_DEVICE_HOST
  FixedShapeTensor2D() : Base() {}

  CUDA_DEVICE_HOST
  T& at(uint32_t i, uint32_t j) {
    return Base::at(data, i, j);
  }

  CUDA_DEVICE_HOST
  void set(uint32_t i, uint32_t j, T val) {
    Base::set(data, i, j, val);
  }
};

template<typename T, uint32_t M, uint32_t N, uint32_t K>
class AbstractFixedShapeTensor3D {
public:
  CUDA_DEVICE_HOST
  AbstractFixedShapeTensor3D() {}

  CUDA_DEVICE_HOST
  uint32_t numel() {return M*N*K;}

  CUDA_DEVICE_HOST
  uint32_t shape(uint32_t i) const {
    switch(i) {
      case  0: return M;
      case  1: return N;
      case  2: return K;
      default: return 0;
    }
  }

  CUDA_DEVICE_HOST
  void zero(T data[]) {
    #pragma unroll
    for (uint i = 0; i < numel(); i++) {
      data[i] = (T)0;
    }
  }

  CUDA_DEVICE_HOST
  T& at(T data[], uint32_t i, uint32_t j, uint32_t k) {
    return data[(i*shape(1)+j)*shape(2) + k];
  }

  CUDA_DEVICE_HOST
  void add(T data[], uint32_t i, uint32_t j, uint32_t k, T val) {
    set(data, i, j, k, at(data, i, j, k) + val);
  }

  CUDA_DEVICE_HOST
  void set(T data[], uint32_t i, uint32_t j, uint32_t k, T val) {
    data[(i*shape(1)+j)*shape(2) + k] = val;
  }
};

template<typename T, uint32_t M, uint32_t N, uint32_t K>
class FixedShapeTensor3D : public AbstractFixedShapeTensor3D<T, M, N, K> {
  T data[M*N*K];
  using Base = AbstractFixedShapeTensor3D<T, M, N, K>;

public:
  CUDA_DEVICE_HOST
  FixedShapeTensor3D() : Base() {}

  CUDA_DEVICE_HOST
  void zero() {Base::zero(data);}
  
  CUDA_DEVICE_HOST
  T& at(uint32_t i, uint32_t j, uint32_t k) {
    return Base::at(data, i, j, k);
  }

  CUDA_DEVICE_HOST
  void add(uint32_t i, uint32_t j, uint32_t k, T val) {
    Base::add(data, i, j, k, val);
  }

  CUDA_DEVICE_HOST
  void set(uint32_t i, uint32_t j, uint32_t k, T val) {
    Base::set(data, i, j, k, val);
  }
};

//Shared Memory Tensors
template<fastKronOp Layout, typename T, uint32_t TileP, uint32_t TileQ>
class DirectShared : public AbstractFixedShapeTensor2D<Layout, T, TileP, TileQ> {
  using Base = AbstractFixedShapeTensor2D<Layout, T, TileP, TileQ>;
  T* data;

public:
  CUDA_DEVICE
  DirectShared() : data(nullptr) {}

  CUDA_DEVICE_HOST
  DirectShared(T* data) : data(data) {}

  CUDA_DEVICE_HOST
  //TODO: Make this Coord1D
  void store(uint32_t eIdx, uint32_t num, const T* elems) {
    #pragma unroll
    for (uint ve = 0; ve < num; ve++) {
      uint idx = eIdx + ve;
      Base::set(data, idx, elems[ve]);
    }
  }

  CUDA_DEVICE_HOST
  //TODO: Make this Coord2D
  void store(uint32_t row, uint32_t col, uint32_t num, const T* elems) {
    #pragma unroll
    for (uint ve = 0; ve < num; ve++) {
      uint32_t idx = row * Base::shape(1) + col + ve;
      Base::set(data, idx, elems[ve]);
    }
  }
  
  CUDA_DEVICE_HOST
  T& at(uint32_t row, uint32_t col) {
    return Base::at(data, row, col);
  }

  CUDA_DEVICE_HOST
  uint32_t p() const {return TileP;}
  CUDA_DEVICE_HOST
  uint32_t q() const {return TileQ;}
};

template<fastKronOp Layout, typename T, uint32_t M, uint32_t N>
class ShiftShared : public AbstractFixedShapeTensor2D<Layout, T, M, N> {
  using Base = AbstractFixedShapeTensor2D<Layout, T, M, N>;
  T* data;

public:
  CUDA_DEVICE
  ShiftShared() : data(nullptr) {}
  CUDA_DEVICE_HOST
  ShiftShared(T* data) : data(data) {}

  CUDA_DEVICE_HOST
  void store(uint32_t row, uint32_t startCol, uint32_t TileP, uint32_t RegK, 
             uint32_t numElems, T* elems) {
    #pragma unroll
    for (uint i = 0; i < numElems; i++) {
      uint32_t shCol = startCol + i;
      uint32_t elem  = shCol%TileP;
      uint32_t slice = shCol/TileP;
      uint32_t shift = slice/RegK;

      Base::set(data, row, slice*TileP + (shift + elem)%TileP, elems[i]);
    }
  }

  CUDA_DEVICE_HOST
  T& at(uint32_t row, uint32_t col) {
    return Base::at(data, row, col);
  }

  CUDA_DEVICE_HOST
  uint32_t m() const {return M;}
  CUDA_DEVICE_HOST
  uint32_t n() const {return N;}
};

//Register Tensors
template<typename T, uint32_t M, uint32_t K, uint32_t Q>
class YRegisters : public FixedShapeTensor3D<T, M, K, Q> {
  using Base = FixedShapeTensor3D<T, M, K, Q>;

public:
  CUDA_DEVICE_HOST
  YRegisters() {Base::zero();}
  
  CUDA_DEVICE_HOST
  uint32_t m() const {return M;}
  CUDA_DEVICE_HOST
  uint32_t k() const {return K;}
  CUDA_DEVICE_HOST
  uint32_t q() const {return Q;}
};

template<typename T, uint32_t M, uint32_t K, uint32_t P>
class XRegisters : public FixedShapeTensor3D<T, M, K, P>{
public:
  CUDA_DEVICE_HOST
  XRegisters() {}

  CUDA_DEVICE_HOST
  uint32_t m() const {return M;}
  CUDA_DEVICE_HOST
  uint32_t k() const {return K;}
  CUDA_DEVICE_HOST
  uint32_t p() const {return P;}
};

template<typename T, uint32_t TileP, uint32_t CRegCols>
class FRegisters : public FixedShapeTensor2D<fastKronOp_N, T, TileP, CRegCols>{
  public:
    CUDA_DEVICE_HOST
    FRegisters() {}
};

class YElem : public Coord2D {
public:
  CUDA_DEVICE_HOST
  YElem(uint32_t yQ, uint32_t yK) : Coord2D(yQ, yK) {}

  CUDA_DEVICE_HOST
  uint32_t q() const {return i();}
  CUDA_DEVICE_HOST
  uint32_t k() const {return j();}
};