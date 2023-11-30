#include <functional>
#include <cassert>
#include <stdio.h>

#pragma once

struct KMMShape {
  const uint m;
  const uint n;
  const uint *qs;
  const uint *ps;

  KMMShape(const uint m, const uint n, const uint *ps, const uint *qs) : 
    m(m), n(n), ps(ps), qs(qs)
  {}

  KMMShape rsub(int rstart, int subn, uint ps[], uint qs[]) const {
    assert (rstart >= 0);
    assert (subn <= n);
    assert (rstart - (subn - 1) >= 0);
    for (int i = 0; i < subn; i++) {
      ps[i]  = this->ps[rstart - i];
      qs[i]  = this->qs[rstart - i];
    }
    return KMMShape(m, subn, ps, qs);
  }

  KMMShape sub(int start, int subn, uint ps[], uint qs[]) const {
    assert (start >= 0);
    assert (subn <= n);
    assert (start + (subn - 1) <= n);
    for (int i = 0; i < subn; i++) {
      ps[i]  = this->ps[start + i];
      qs[i]  = this->qs[start + i];
    }
    return KMMShape(m, subn, ps, qs);
  }
};

struct GeKMMPtrs {
  void * x;
  void ** fs;
  void * y;

  GeKMMPtrs() : x(nullptr), fs(nullptr), y(nullptr) {}

  GeKMMPtrs(void* x, void ** fs, void * y) : 
    x(x), fs(fs), y(y) {}
  
  GeKMMPtrs swap(void* temp1, void* temp2) {
    void* x1 = y;
    void* y1;
    if (x1 == temp1) {        
      y1 = temp2;
    } else if (x1 == temp2) {
      y1 = temp1;
    }
    return GeKMMPtrs(x1, fs, y1);
  }

  GeKMMPtrs rsub(int rstart, int subn, void* fs[]) const {
    if (this->fs == nullptr) {
      return GeKMMPtrs(x, nullptr, y);
    }

    for (int i = 0; i < subn; i++) {
      fs[i] = this->fs[rstart - i];
    }
    return GeKMMPtrs(x, fs, y);
  }

  GeKMMPtrs sub(int start, int subn, void* fs[]) const {
    if (this->fs == nullptr) {
      return GeKMMPtrs(x, nullptr, y);
    }

    for (int i = 0; i < subn; i++) {
      fs[i] = this->fs[start + i];
    }
    return GeKMMPtrs(x, fs, y);
  }
};

struct KMMProblem {
  KMMShape shape;
  GeKMMPtrs ptrs;

  const int rstart;
  uint k;
  uint l;

  KMMProblem(KMMShape shape, GeKMMPtrs ptrs, int rstart, 
             const uint k, const uint l) : shape(shape), ptrs(ptrs), 
             rstart(rstart), k(k), l(l) {
    assert (rstart >= 0);
  }
  
  KMMProblem(KMMShape shape, GeKMMPtrs ptrs) : 
    KMMProblem(shape, ptrs, 0, 1, 1) {
    k = 1;
    l = 1;
    for (uint i = 0; i < shape.n; i++) {
      k *= shape.ps[i];
      l *= shape.qs[i];
    }
  }

  KMMProblem(KMMProblem problem, int rstart, 
    const uint k, const uint l) : 
    KMMProblem(problem.shape, problem.ptrs, rstart, k, l) {}
  
  KMMProblem rsub(GeKMMPtrs ptrs, uint ps[], uint qs[], void* fs[], 
                  int rstart, int num) const {
    uint subk = k, subl = l;
    for (int i = 0; i <= rstart - num; i++) {
      subl = (subl/shape.qs[i])*shape.ps[i];
    }
    for (int i = shape.n - 1; i > rstart; i--) {
      subk = (subk/shape.ps[i])*shape.qs[i];
    }

    return KMMProblem(shape.rsub(rstart, num, ps, qs),
                      ptrs.rsub(rstart, num, fs),
                      rstart, subk, subl);
  }
  KMMProblem sub(GeKMMPtrs ptrs, uint ps[], uint qs[], void* fs[], 
                 uint start, uint num) const {
    uint subk = k, subl = l;
    
    for (int i = 0; i < start; i++) {
      subl = (subl/shape.qs[i])*shape.ps[i];
    }
    for (int i = shape.n - 1; i >= start + num; i--) {
      subk = (subk/shape.ps[i])*shape.qs[i];
    }

    return KMMProblem(shape.sub(start, num, ps, qs),
                      ptrs.sub(start, num, fs),
                      start, subk, subl);
  }
};

cudaError_t executeGeKMM(const KMMProblem problem, void* temps[2],
                         void* result,
                         std::function<uint (const KMMProblem)> next,
                         std::function<cudaError_t (const KMMProblem, void*[2], void*)> func);
cudaError_t reverseExecuteGeKMM(const KMMProblem problem, void* temps[2],
                                void* result,
                                std::function<uint (const KMMProblem)> next,
                                std::function<cudaError_t (const KMMProblem, void*[2], void*)> func);