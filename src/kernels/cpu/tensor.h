#include "kernels/cuda/fixed-shape-tensor.cuh"

#pragma once

template<typename T, 
         typename OptTileX, typename OptF, typename OptTileF>
class TransposedDirectShared3D : public AbstractFixedShapeTensor2D<fastKronOp_N, T, OptTileX::M(), OptTileX::N()> {
  using Base = AbstractFixedShapeTensor2D<fastKronOp_N, T, OptTileX::M(), OptTileX::N()>;
  T* data;

public:
  CUDA_DEVICE_HOST
  TransposedDirectShared3D(T* data) : data(data) {}

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

template<typename T, typename OptTileX, typename OptTileF, typename OptF>
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
  T& at(const uint32_t m, const uint32_t q, const uint32_t slice) {
    return Base::at(data, m, q, slice);
  }
};