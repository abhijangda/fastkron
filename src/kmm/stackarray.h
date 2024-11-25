#include <initializer_list>

#include "config.h"

#pragma once

/**
 * StackArray - This class defines an array that is stored on the stack for a type and size.
 */
template<typename T, uint32_t MaxSize>
class StackArray {
public:
  /**
   * @array: The storage buffer of type @T with size @MaxSize.
   */
  T array[MaxSize];
  /**
   * @n: Length of elements filled in the array.
   */
  uint32_t n;

  StackArray() {
    for (uint32_t i = 0; i < MaxSize; i++) {
      array[i] = T();
    }
  }

public:
  StackArray(const T* ptrs, uint32_t n) : n(n) {
    if (ptrs) {
      for (uint32_t i = 0; i < n; i++) {
        array[i] = ptrs[i];
      }
    }

    for (uint32_t i = n; i < MaxSize; i++) {
      array[i] = T();
    }
  }

  StackArray(std::initializer_list<T> initList) : n(initList.size()) {
    int len = 0;
    for (auto elem : initList) {
      array[len++] = elem;
    }

    for (uint32_t i = n; i < MaxSize; i++) {
      array[i] = T();
    }
  }

  CUDA_DEVICE_HOST
  T& operator[](int index) {
    assert (index < n && index >= 0);
    return array[index];
  }

  CUDA_DEVICE_HOST
  T& operator[](uint32_t index) {
    assert (index < n);
    return array[index];
  }

  StackArray<T, MaxSize> sub(uint32_t start, uint32_t length) const {
    assert(length <= n);
    T ptrs[length];
    for (uint32_t i = 0; i < length; i++) {
      ptrs[i] = array[i + start];
    }

    return StackArray<T, MaxSize>(ptrs, length);
  }

  void push_front(const T& elem) {
    assert(n < MaxSize);
    for (int i = n; i >= 1; i--) {
      array[i] = array[i-1];
    }

    array[0] = elem;
    n++;
  }

  void push_back(const T& elem) {
    assert(n < MaxSize);
    array[n] = elem;
    n++;
  }

  CUDA_DEVICE_HOST
  uint32_t len() const {return n;}

  template<uint32_t SliceSize>
  StackArray<T, SliceSize> slice(uint32_t start) const {
    assert(SliceSize <= n);
    T ptrs[SliceSize];
    for (uint32_t i = 0; i < SliceSize; i++) {
      ptrs[i] = array[i + start];
    }

    return StackArray<T, SliceSize>(ptrs, SliceSize);
  }

  template<uint32_t SliceSize>
  StackArray<T, SliceSize> sliceOrEmpty(uint32_t start) const {
    T ptrs[SliceSize];
    for (uint32_t i = 0; i < SliceSize && i < n; i++) {
      ptrs[i] = array[i + start];
    }

    for (uint32_t i = n; i < SliceSize; i++) {
      ptrs[i] = T();
    }

    return StackArray<T, SliceSize>(ptrs, SliceSize);
  }

  StackArray<T, MaxSize> slice(uint32_t start, uint32_t size) const {
    assert(size <= n);
    T ptrs[size];
    for (uint32_t i = 0; i < size; i++) {
      ptrs[i] = array[i + start];
    }

    return StackArray<T, MaxSize>(ptrs, size);
  }

  StackArray(const StackArray& x) : StackArray (&x.array[0], x.len()) {}
};