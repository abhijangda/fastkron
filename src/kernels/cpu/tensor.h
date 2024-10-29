#include "kernels/cuda/fixed-shape-tensor.cuh"

#pragma once

//TODO: Think about this
template<typename T, bool isKMultipleOfTileK, bool isTileKSame,
         typename OptTileX>
class SliceCPU {
public:
  const Matrix parent;
  //TODO: Create Coord2D
  uint32_t startrow;
  uint32_t startcol;
  uint32_t tileRows_;
  uint32_t tileCols_;
  uint32_t rows;
  uint32_t cols;
  uint32_t P;
  T* ptr;

public:
  CUDA_DEVICE_HOST
  SliceCPU(uint32_t startrow, uint32_t startcol, uint32_t paramTileK, uint32_t P, Matrix parent) :
    parent(parent), startrow(startrow), startcol(startcol),
    tileRows_(OptTileX::M()),
    P(P),
    ptr(parent.data<T>(startrow, startcol, OptTileX::Op())) {
      tileCols_ = isTileKSame ? OptTileX::N() : paramTileK;
      rows = (tileRows_ == 1) ? 1 : MIN(tileRows_, parent.m() - startrow);
      cols = isKMultipleOfTileK ? tileCols() : MIN(tileCols(), parent.n() - startcol);
    }

  CUDA_DEVICE_HOST
  const T* data(uint32_t row, uint32_t slice, uint32_t elem) const {
    //TODO: get common parts out
    if (OptTileX::Op() == fastKronOp_N) {
      uint32_t idx = row * parent.n();
      idx += slice*P + elem;
      return &ptr[idx];
    } else if (OptTileX::Op() == fastKronOp_T) {
      uint32_t idx = slice*P + elem;
      idx = idx * parent.m() + row;
      return &ptr[idx];
    }

    return nullptr;
  }

  CUDA_DEVICE_HOST
  const T* data(uint32_t idx) const {
    return &ptr[idx];
  }

  CUDA_DEVICE_HOST
  uint32_t m() const {return rows;}
  CUDA_DEVICE_HOST
  uint32_t n() const {return cols;}
  CUDA_DEVICE_HOST
  uint32_t numel() const {return rows * cols;}
  CUDA_DEVICE_HOST
  uint32_t tileRows() const {return tileRows_;}
  CUDA_DEVICE_HOST
  uint32_t tileCols() const {return isTileKSame ? OptTileX::N() : tileCols_;}
};

template<typename T, fastKronOp Layout,
         typename OptTileX, typename OptF, typename OptTileF>
class TransposedDirectShared3D : public AbstractFixedShapeTensor2D<Layout, T, OptTileX::M(), OptTileX::N()> {
  using Base = AbstractFixedShapeTensor2D<Layout, T, OptTileX::M(), OptTileX::N()>;
  T* data;

public:
  CUDA_DEVICE_HOST
  TransposedDirectShared3D(T* data) : data(data) {}

  CUDA_DEVICE_HOST
  fastKronOp layout() {return Layout;}

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
  T& at(uint32_t row, uint32_t slice, uint32_t elem) {
    return Base::at(data, row, elem * slices() + slice);
  }

  CUDA_DEVICE_HOST
  const T& at(uint32_t row, uint32_t slice, uint32_t elem) const {
    return Base::at(data, row, elem * slices() + slice);
  }

  void zero(uint32_t startRow, uint32_t startSlice, uint32_t startElem, uint32_t endRow, uint32_t endSlice, uint32_t endElem) {
    for (uint32_t r = startRow; r < endRow; r++) {
      for (uint32_t e = startElem; e < endElem; e++) {
        for (uint32_t c = startSlice; c < endSlice; c++) {
          at(r, c, e) = 0.0f;
        }
      }
    }
  }

  CUDA_DEVICE_HOST
  uint32_t numel() const {return m() * n();}
  
  CUDA_DEVICE_HOST
  uint32_t slices() const {return OptTileX::N()/OptF::P();}

  CUDA_DEVICE_HOST
  uint32_t m() const {return OptTileX::M();}
  CUDA_DEVICE_HOST
  uint32_t n() const {return OptTileX::N();}
  CUDA_DEVICE_HOST
  uint32_t p() const {return OptTileF::P();}
};

template<typename T, fastKronOp OpY, typename OptTileX, typename OptTileF, typename OptF>
class YInterim : public AbstractFixedShapeTensor3D<T, OptTileX::M(), OptTileF::Q(), OptTileX::N()/OptF::P()> {
  using Base = AbstractFixedShapeTensor3D<T, OptTileX::M(), OptTileF::Q(), OptTileX::N()/OptF::P()>;
  T* data;

public:
  CUDA_DEVICE_HOST
  YInterim(T* data) : data(data) {}
  
  CUDA_DEVICE_HOST
  uint32_t m()      const {return OptTileX::M();}
  CUDA_DEVICE_HOST
  uint32_t slices() const {return OptTileX::N()/OptF::P();}
  CUDA_DEVICE_HOST
  uint32_t q()      const {return OptTileF::Q();}
  CUDA_DEVICE_HOST
  fastKronOp layout() const {return OpY;}

  CUDA_DEVICE_HOST
  T& at(const uint32_t m, const uint32_t q, const uint32_t slice) {
    if (OpY == fastKronOp_N)
      return Base::at(data, m, q, slice);
    else if (OpY == fastKronOp_T) {
      return Base::at(data, q * this->slices() * this->m() + slice * this->m() + m);
    }
  }
};