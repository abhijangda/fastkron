#include <functional>
#include <cassert>
#include <iostream>
#include <numeric>

#include <cuda_runtime.h>
#include <cuda.h>

#include "kmm/matrix.h"

#pragma once

struct KMMProblem {
public:
  static const int MaxN = 64;
  
  Matrix x;
  Matrix y;
  FactorArray fs;
  int n;

  KMMProblem(Matrix x, int n, FactorArray fs, Matrix y) :
    x(x), n(n), fs(fs), y(y) {}

public:
  KMMProblem(Matrix x, int n, Factor* fs, Matrix y) :
    x(x), n(n), fs(fs, n), y(y) {}

  KMMProblem(const uint m, const uint32_t n, const uint32_t *ps, const uint32_t *qs, 
             void* xptr, void* const* fsptr, void* yptr, const int k, 
             const int l) : x(m, k, xptr), y(m, l, yptr), n(n), fs(n, ps, qs, fsptr) {
  }
  
  KMMProblem(const uint m, const int n, const uint *ps, const uint *qs,
             void* x, void* const* fs, void* y) :
    KMMProblem(m, n, ps, qs, x, fs, y, 
               std::reduce(ps, ps+n, 1, std::multiplies<uint>()),
               std::reduce(qs, qs+n, 1, std::multiplies<uint>()))
    {}

  KMMProblem(const uint m, const int n, const uint *ps, const uint *qs) :
    KMMProblem(m, n, ps, qs, nullptr, nullptr, nullptr) {}

  // KMMProblem(KMMProblem prob, const int k, const int l) :
  //   KMMProblem(prob.x, prob.n, prob.ps, prob.qs, 
  //              prob.x, prob.fs, prob.y, k, l) {}

  // KMMProblem(KMMProblem problem, void* x, void** fs, void* y) :
  //   KMMProblem(problem.m, problem.n, problem.ps, problem.qs, x, fs, y) {}

  KMMProblem rsub(int rstart, int subn) const {    
    int subk = x.n(), subl = y.n();
    for (int i = 0; i <= rstart - subn; i++) {
      subl = (subl/fs[i].q())*fs[i].p();
    }
    for (int i = n - 1; i > rstart; i--) {
      subk = (subk/fs[i].p())*fs[i].q();
    }

    assert (rstart >= 0);
    assert (subn <= n);
    assert (rstart - (subn - 1) >= 0);

    return KMMProblem(Matrix(x.m(), subk, x.data), subn,
                      fs.sub(rstart - (subn - 1), subn),
                      Matrix(y.m(), subl, y.data));
  }

  KMMProblem sub(int start, int subn) const {
    int subk = x.n(), subl = y.n();
    
    for (int i = 0; i < start; i++) {
      subl = (subl/fs[i].q())*fs[i].p();
    }
    for (int i = n - 1; i >= start + subn; i--) {
      subk = (subk/fs[i].p())*fs[i].q();
    }

    assert (start >= 0);
    assert (subn <= n);
    assert (start + (subn - 1) <= n);
    //TODO: make Matrix::data a function
    return KMMProblem(Matrix(x.m(), subk, x.data), subn,
                      fs.sub(start, subn),
                      Matrix(y.m(), subl, y.data));
  }

  uint32_t* ps(uint32_t *array) {
    for (uint32_t i = 0; i < n; i++) {
      array[i] = fs[i].p();
    }
    return array;
  }

  uint32_t* qs(uint32_t *array) {
    for (uint32_t i = 0; i < n; i++) {
      array[i] = fs[i].q();
    }
    return array;
  }

  void swap(void* temp1, void* temp2) {
    void* x1 = y.ptr();
    void* y1;
    if (x1 == temp1) {        
      y1 = temp2;
    } else if (x1 == temp2) {
      y1 = temp1;
    }

    x.data = x1;
    y.data = y1;
  }
  
  uint32_t k() const {return x.n();}
  uint32_t l() const {return y.n();}
  uint32_t m() const {return x.m();}

  bool operator==(const KMMProblem& other) const {
    bool eq = x == other.x && n == other.n && y == other.y;
    if (eq) {
      for (int i = 0; i < n; i++) {
        eq = eq && fs[i] == other.fs[i];
      }
    }
    return eq;
  }

  bool sameFactorShapes() const {
    bool eq = true;
    for (int i = 1; i < n; i++) {
      eq = eq && fs[i-1].p() == fs[i].p() &&
                 fs[i-1].q() == fs[i].q();
    }

    return eq;
  }

  friend std::ostream& operator<<(std::ostream &out, const KMMProblem &problem) {
    out << problem.x.m() << "*(";
    if (problem.sameFactorShapes()) 
      out << problem.fs[0] << "^" << problem.n;
    else
      for (int i = 0; i < problem.n; i++) {
        out << problem.fs[i];
        if (i < problem.n - 1) out << "(x)";
      }
    out << ")";
    return out;
  }
};

template<>
struct std::hash<KMMProblem> {
  std::size_t operator()(const KMMProblem& k) const;
};

cudaError_t executeGeKMM(const KMMProblem problem, void* temps[2],
                         void* result,
                         std::function<uint (const KMMProblem)> next,
                         std::function<cudaError_t (const KMMProblem, int, void*[2], void*)> func);
cudaError_t reverseExecuteGeKMM(const KMMProblem problem, void* temps[2],
                                void* result,
                                std::function<uint (const KMMProblem)> next,
                                std::function<cudaError_t (const KMMProblem, int, void*[2], void*)> func);
bool checkDistributedKronSizes(const KMMProblem problem,
                               const uint LocalKrons, const uint gpusInK);