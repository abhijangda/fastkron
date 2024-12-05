#include "kmm/coord.h"

#define CUDA_DEVICE_ASSERT(x) ;
// assert(x)

template<fastKronOp Layout, typename T, uint32_t M, uint32_t N>
class AbstractFixedShapeTensor2D {
public:
  CUDA_DEVICE_HOST
  AbstractFixedShapeTensor2D() {}

  CUDA_DEVICE_HOST
  fastKronOp layout() const {return Layout;}

  CUDA_DEVICE_HOST
  static uint32_t numel() {return M*N;}

  CUDA_DEVICE_HOST
  static constexpr uint32_t shape(uint32_t dim) {
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

    return 0;
  }

  CUDA_DEVICE_HOST
  uint32_t linearIdx(uint32_t i, uint32_t j) const {
    if (Layout == fastKronOp_N)
      return i * shape(1) + j;
    else
      return j * shape(1) + i;
  }

  CUDA_DEVICE_HOST
  T& at(T data[], uint32_t i, uint32_t j) {
    // CUDA_DEVICE_ASSERT(linearIdx(i, j) < numel());
    return data[linearIdx(i, j)];
  }

  CUDA_DEVICE_HOST
  const T& at(const T data[], uint32_t i, uint32_t j) const {
    // CUDA_DEVICE_ASSERT(linearIdx(i, j) < numel());
    return data[linearIdx(i, j)];
  }

  CUDA_DEVICE_HOST
  void set(T data[], uint32_t i, T val) {
    data[i] = val;
  }

  CUDA_DEVICE_HOST
  void set(T data[], uint32_t i, uint32_t j, T val) {
    // CUDA_DEVICE_ASSERT(linearIdx(i, j) < numel());
    data[linearIdx(i, j)] = val;
  }
};

template<fastKronOp Layout, typename T, uint32_t M, uint32_t N>
class FixedShapeFactor : AbstractFixedShapeTensor2D<Layout, T, M, N> {
public:
  FixedShapeFactor() : AbstractFixedShapeTensor2D<Layout, T, M, N>() {}

  static constexpr fastKronOp Op() {return Layout;}
  static constexpr uint32_t P()  {return M;}
  static constexpr uint32_t Q()  {return N;}
};

template<fastKronOp Layout, typename T, uint32_t M_, uint32_t N_>
class FixedShapeMatrix : AbstractFixedShapeTensor2D<Layout, T, M_, N_> {
public:
  FixedShapeMatrix() : AbstractFixedShapeTensor2D<Layout, T, M_, N_>() {}

  static constexpr fastKronOp Op() {return Layout;}
  static constexpr uint32_t M()  {return M_;}
  static constexpr uint32_t N()  {return N_;}
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
  const T& at(uint32_t i, uint32_t j) const {
    return Base::at(data, i, j);
  }

  CUDA_DEVICE_HOST
  void set(uint32_t i, uint32_t j, T val) {
    Base::set(data, i, j, val);
  }

  template<typename F>
  CUDA_DEVICE_HOST
  void apply(F&& fn){
    #pragma unroll
    for (uint32_t m = 0; m < M; m++) {
    #pragma unroll
    for (uint32_t n = 0; n < N; n++) {
      fn(at(m, n), m, n);
    }}
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
      data[i] = 0;
    }
  }

  CUDA_DEVICE_HOST
  T& at(T data[], uint32_t i, uint32_t j, uint32_t k) {
    return data[(i*shape(1)+j)*shape(2) + k];
  }

  CUDA_DEVICE_HOST
  const T& at(const T data[], uint32_t i, uint32_t j, uint32_t k) const {
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

  CUDA_DEVICE_HOST
  T& at(T data[], uint32_t idx) {
    return data[idx];
  }
};

template<fastKronOp Layout, typename T, uint32_t M, uint32_t N, uint32_t K>
class FixedShapeTensor3D : public AbstractFixedShapeTensor3D<T, M, N, K> {
  T data[M*N*K];
  using Base = AbstractFixedShapeTensor3D<T, M, N, K>;

public:
  CUDA_DEVICE_HOST
  FixedShapeTensor3D() : Base() {}

  static constexpr fastKronOp layout() {return Layout;}
  CUDA_DEVICE_HOST
  void zero() {Base::zero(data);}

  CUDA_DEVICE_HOST
  T& at(uint32_t i, uint32_t j, uint32_t k) {
    return Base::at(data, i, j, k);
  }

  CUDA_DEVICE_HOST
  const T& at(uint32_t i, uint32_t j, uint32_t k) const {
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

  template<typename F>
  CUDA_DEVICE_HOST
  void apply(F&& fn){
    if (Layout == fastKronOp_N) {
      #pragma unroll
      for (uint32_t m = 0; m < M; m++) {
      #pragma unroll
      for (uint32_t k = 0; k < K; k++) {
      #pragma unroll
      for (uint32_t n = 0; n < N; n++) {
        fn(at(m, n, k), m, n, k);
      }}}
    } else {
      #pragma unroll
      for (uint32_t k = 0; k < K; k++) {
      #pragma unroll
      for (uint32_t n = 0; n < N; n++) {
      #pragma unroll
      for (uint32_t m = 0; m < M; m++) {
        fn(at(m, n, k), m, n, k);
      }}}
    }
  }
};

//Shared Memory Tensors
template<fastKronOp Layout, typename T, uint32_t TileP, uint32_t TileQ>
class DirectShared : public AbstractFixedShapeTensor2D<Layout, T, TileP, TileQ> {
  using Base = AbstractFixedShapeTensor2D<Layout, T, TileP, TileQ>;
  T* data;

public:
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
  void store(uint32_t row, uint32_t col, uint32_t num, const T* elems, fastKronOp elemLayout) {
    if (Layout == fastKronOp_N) {
      //CUDA MKM
      if (elemLayout == fastKronOp_N) {
        #pragma unroll
        for (uint ve = 0; ve < num; ve++) {
          uint32_t idx = row * Base::shape(1) + col + ve;
          Base::set(data, idx, elems[ve]);
        }
      } else {
        #pragma unroll
        for (uint ve = 0; ve < num; ve++) {
          uint32_t idx = (row + ve) * Base::shape(1) + col;
          Base::set(data, idx, elems[ve]);
        }
      }
    } else {
      //CUDA KMM
      if (elemLayout == fastKronOp_N) {
        #pragma unroll
        for (uint ve = 0; ve < num; ve++) {
          uint32_t idx = row * (q() + 1) + col + ve;
          Base::set(data, idx, elems[ve]);
        }
      } else {
        //Padding is probably not needed when using float4/double2
        #pragma unroll
        for (uint ve = 0; ve < num; ve++) {
          uint32_t idx = (row + ve) * (q() + 1) + col;
          Base::set(data, idx, elems[ve]);
        }
      }
    }
  }

  CUDA_DEVICE_HOST
  void store_row(uint32_t row, uint32_t num, const T* __restrict__  ptr) {
    memcpy(((Layout == fastKronOp_N) ? &at(row, 0) : &at(0, row)), ptr, num * sizeof(T));
    if (num < Base::shape(1)) {
      memset(((Layout == fastKronOp_N) ? &at(row, num) : &at(num, row)), 0, (Base::shape(1) - num)*sizeof(T));
    }
  }

  CUDA_DEVICE_HOST
  void zero_row(uint32_t row) {
    memset(((Layout == fastKronOp_N) ? &at(row, 0) : &at(0, row)), 0, Base::shape(1) * sizeof(T));
  }

  CUDA_DEVICE_HOST
  T& at(uint32_t row, uint32_t col) {
    return Base::at(data, row, col);
  }

  CUDA_DEVICE_HOST
  const T& at(uint32_t row, uint32_t col) const {
    return Base::at(data, row, col);
  }

  CUDA_DEVICE_HOST
  uint32_t p() const {return TileP;}
  CUDA_DEVICE_HOST
  uint32_t q() const {return TileQ;}
};

template<fastKronOp Layout, typename T, bool kXshSlicesSame,
         uint32_t kM, uint32_t kSlices, uint32_t kP>
class ShiftShared : public AbstractFixedShapeTensor2D<Layout, T, kM, kSlices * kP> {
  using Base = AbstractFixedShapeTensor2D<Layout, T, kM, kSlices * kP>;
  T* data;
  uint32_t ShTileK;

public:
  CUDA_DEVICE_HOST
  ShiftShared(T* data, uint32_t ShTileK) : data(data), ShTileK(ShTileK) {}

  CUDA_DEVICE_HOST
  void store(uint32_t startRow, uint32_t startCol, uint32_t RegK,
              uint32_t numElems, T* elems, fastKronOp elemOp) {
    #pragma unroll
    for (uint i = 0; i < numElems; i++) {
      uint32_t col = 0;
      uint32_t row = 0;
      if (Layout == fastKronOp_N) {
        if (elemOp == fastKronOp_N) {
          uint32_t shCol = startCol + i;
          uint32_t elem  = shCol%p();
          uint32_t slice = shCol/p();
          uint32_t shift = slice/RegK;
          //TODO: Do we need shift when TileK/RegK < 32? I do not think
          col = slice*p() + (shift + elem)%p();
          // CUDA_DEVICE_ASSERT(row * n() + col < numel());
          row = startRow;
        } else {
          uint32_t shCol = startCol;
          uint32_t elem  = shCol%p();
          uint32_t slice = shCol/p();
          uint32_t shift = slice/RegK;
          //TODO: Do we need shift when TileK/RegK < 32? I do not think
          col = slice*p() + (shift + elem)%p();
          // CUDA_DEVICE_ASSERT(row * n() + col < numel());
          row = startRow + i;
        }
      } else {
        if (elemOp == fastKronOp_T) {
          uint32_t shCol = startCol;
          uint32_t elem  = shCol%p();
          uint32_t slice = shCol/p();
          uint32_t shift = 0;//slice/RegK;
          col = shCol + i;
          //TODO: When shift is 0 use vector store
          row = (startRow + shift);
        } else {
          uint32_t shCol = startCol;
          uint32_t elem  = shCol%p();
          uint32_t slice = shCol/p();
          uint32_t shift = 0;//slice/RegK;
          col = shCol;
          //TODO: When shift is 0 use vector store
          row = (startRow + i + shift);// % kM; //TODO: Commenting out kM removes any shared mem store bank conflicts
        }
      }
      Base::set(data, row, col, elems[i]);
    }
  }

  CUDA_DEVICE_HOST
  void store(uint32_t row, uint32_t slice, uint32_t elem, uint32_t RegK,
             uint32_t numElems, T* elems) {
    //Only works for numElems == 1
    #pragma unroll
    for (uint i = 0; i < numElems; i++) {
      uint32_t shift = (Base::layout() == fastKronOp_N) ? slice/RegK : 0;
      uint32_t col = slice*p() + (shift + elem)%p();
      // CUDA_DEVICE_ASSERT(row * n() + col < numel());
      // printf("row %d col %d\n", row, col);
      Base::set(data, row, col, elems[i]);
    }
  }

  CUDA_DEVICE_HOST
  T& at(uint32_t row, uint32_t col) {
    // CUDA_DEVICE_ASSERT(row * n() + col < numel());
    return Base::at(data, row, col);
  }

  CUDA_DEVICE_HOST
  uint32_t numel() const {return m() * n();}

  CUDA_DEVICE_HOST
  uint32_t slices() const {return (kXshSlicesSame) ? kSlices : ShTileK/kP;}

  CUDA_DEVICE_HOST
  uint32_t m() const {return kM;}
  CUDA_DEVICE_HOST
  uint32_t n() const {return slices() * p();}
  CUDA_DEVICE_HOST
  uint32_t p() const {return kP;}
};

//Register Tensors
template<fastKronOp Layout, typename T, uint32_t M, uint32_t K, uint32_t Q,
         uint32_t MVectorLen = 1, uint32_t KVectorLen = 1>
class YRegisters : public FixedShapeTensor3D<Layout, T, M/MVectorLen, K/KVectorLen, Q> {
  using Base = FixedShapeTensor3D<Layout, T, M/MVectorLen, K/KVectorLen, Q>;

public:
  CUDA_DEVICE_HOST
  YRegisters() {Base::zero();}

  CUDA_DEVICE_HOST
  static constexpr uint32_t kvec() {return KVectorLen;}
  CUDA_DEVICE_HOST
  static constexpr uint32_t mvec() {return MVectorLen;}
  CUDA_DEVICE_HOST
  static constexpr uint32_t m() {return M/MVectorLen;}
  CUDA_DEVICE_HOST
  static constexpr uint32_t k() {return K/KVectorLen;}
  CUDA_DEVICE_HOST
  static constexpr uint32_t q() {return Q;}
};

template<fastKronOp Layout, typename T, uint32_t M, uint32_t K, uint32_t P>
class XRegisters : public FixedShapeTensor3D<Layout, T, M, K, P>{
public:
  CUDA_DEVICE_HOST
  XRegisters() {}

  CUDA_DEVICE_HOST
  constexpr uint32_t m() {return M;}
  CUDA_DEVICE_HOST
  constexpr uint32_t k() {return K;}
  CUDA_DEVICE_HOST
  constexpr uint32_t p() {return P;}
};

template<typename T, uint32_t TileP, uint32_t RegQ>
class FRegisters : public FixedShapeTensor2D<fastKronOp_N, T, TileP, RegQ>{
  public:
    CUDA_DEVICE_HOST
    FRegisters() {}

  CUDA_DEVICE_HOST
  constexpr uint32_t q() const {return RegQ;}
};

class YElem : public Coord3D {
public:
  CUDA_DEVICE_HOST
  YElem(uint32_t yM, uint32_t yQ, uint32_t yK) : Coord3D(yM, yQ, yK) {}

  CUDA_DEVICE_HOST
  uint32_t m() const {return Coord3D::i();}
  CUDA_DEVICE_HOST
  uint32_t q() const {return Coord3D::j();}
  CUDA_DEVICE_HOST
  uint32_t k() const {return Coord3D::k();}
};