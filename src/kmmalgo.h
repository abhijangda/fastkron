#include <functional>
#include <cassert>

#pragma once

struct KMMShape {
  const uint m;
  const uint n;
  const uint *qs;
  const uint *ps;

  KMMShape(uint m, uint n, uint *ps, uint *qs) : 
    m(m), n(n), ps(ps), qs(qs)
  {}
};

struct GeKMMPtrs {
  void * x;
  void ** fs;
  void * y;

  GeKMMPtrs() : x(nullptr), fs(nullptr), y(nullptr) {}

  GeKMMPtrs(void* x, void ** fs, void * y) : 
    x(x), fs(fs), y(y) {}
};

struct KMMProblem {
  KMMShape shape;
  GeKMMPtrs ptrs;

  const uint start;
  const uint end;
  uint k;
  uint l;

  KMMProblem(KMMShape shape, GeKMMPtrs ptrs, uint start, uint end, 
             const uint k, const uint l) : shape(shape), ptrs(ptrs), 
             start(start), end(end), k(k), l(l) {
    assert (start >= 0);
    assert (end <= shape.n);
    assert (start < end);
    assert (shape.n >= end - start);
  }
  
  KMMProblem(KMMShape shape, GeKMMPtrs ptrs) : shape(shape), ptrs(ptrs), start(0), end(shape.n) {
    k = 1;
    l = 1;
    for (int i = 0; i < shape.n; i++) {
      k *= shape.ps[i];
      l *= shape.qs[i];
    }
  }

  KMMProblem(KMMProblem problem, uint start, uint end, 
    const uint k, const uint l) : 
    KMMProblem(problem.shape, problem.ptrs, start, end, k, l) {}
};

cudaError_t executeGeKMM(const KMMProblem problem, void* temp1,
                         void* temp2, 
                         std::function<uint (const KMMProblem, void*, void*, cudaError_t&)> func);