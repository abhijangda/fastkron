#include <functional>
#include <cassert>
#include <iostream>
#include <numeric>

#include <cuda_runtime.h>
#include <cuda.h>

#include "device/sliced_mul_shape.h"

#pragma once

struct KMMProblem {
  static const int MaxN = 64;
  
  Matrix x;
  Matrix y;
  StackArray<Matrix, MaxN> fs;
  int n;

  KMMProblem(Matrix x, int n, Matrix* fs, Matrix y) :
    x(x), n(n), fs(fs, n), y(y) {}

  KMMProblem(const uint m, const int n, const uint *ps, const uint *qs, 
             void* xptr, void* const* fsptr, void* yptr, const int k, 
             const int l) : x(m, k, xptr), y(m, l, yptr), n(n) {
    assert (n < MaxN);
    for (int i = 0; i < n; i++) {
      fs[i] = Matrix(ps[i], qs[i], 
                     (fsptr) ? fsptr[i] : nullptr);
    }

    for (int i = n; i < MaxN; i++) {
      this->fs[i] = Matrix();
    }
  }
  
  KMMProblem(const uint m, const int n, const uint *ps, const uint *qs,
             void* x, void* const* fs, void * y) :
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
    uint ps[n];
    uint qs[n];
    void* fs[n];
    
    int subk = k, subl = l;
    for (int i = 0; i <= rstart - subn; i++) {
      subl = (subl/this->qs[i])*this->ps[i];
    }
    for (int i = n - 1; i > rstart; i--) {
      subk = (subk/this->ps[i])*this->qs[i];
    }

    assert (rstart >= 0);
    assert (subn <= n);
    assert (rstart - (subn - 1) >= 0);
    for (int i = 0; i < subn; i++) {
      ps[i]  = this->ps[rstart - (subn - 1) + i];
      qs[i]  = this->qs[rstart - (subn - 1) + i];
    }

    for (int i = 0; i < subn; i++) {
      fs[i] = this->fs[rstart  - (subn - 1) + i];
    }

    return KMMProblem(m, subn, ps, qs,
                      x, fs, y, subk, subl);
  }

  KMMProblem sub(int start, int subn) const {
    uint ps[n];
    uint qs[n];
    void* fs[n];
    
    int subk = k, subl = l;
    
    for (int i = 0; i < start; i++) {
      subl = (subl/this->qs[i])*this->ps[i];
    }
    for (int i = n - 1; i >= start + subn; i--) {
      subk = (subk/this->ps[i])*this->qs[i];
    }

    assert (start >= 0);
    assert (subn <= n);
    assert (start + (subn - 1) <= n);
    for (int i = 0; i < subn; i++) {
      ps[i]  = this->ps[start + i];
      qs[i]  = this->qs[start + i];
    }

    for (int i = 0; i < subn; i++) {
      fs[i] = this->fs[start + i];
    }

    return KMMProblem(m, subn, ps, qs,
                      x, fs, y, subk, subl);
  }

  void swap(void* temp1, void* temp2) {
    void* x1 = y;
    void* y1;
    if (x1 == temp1) {        
      y1 = temp2;
    } else if (x1 == temp2) {
      y1 = temp1;
    }

    x = x1;
    y = y1;
  }
  
  bool operator==(const KMMProblem& other) const {
    bool eq = m == other.m && n == other.n;
    if (eq) {
      for (int i = 0; i < n; i++) {
        eq = eq && ps[i] == other.ps[i] && qs[i] == other.qs[i];
      }
    }
    return eq;
  }

  bool sameFactorShapes() const {
    bool eq = true;
    for (int i = 1; i < n; i++) {
      eq = eq && ps[i - 1] == ps[i] && 
                 qs[i - 1] == qs[i];      
    }

    return eq;
  }

  friend std::ostream& operator<<(std::ostream &out, const KMMProblem &problem) {
    out << problem.m << "*(";
    if (problem.sameFactorShapes()) 
      out << problem.ps[0] << "x" << problem.qs[0] << "^" << problem.n;
    else
      for (int i = 0; i < problem.n; i++) {
        out << problem.ps[i] << "x" << problem.qs[i];
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