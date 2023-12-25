#include <functional>
#include <cassert>
#include <iostream>
#include <numeric>

#include <cuda_runtime.h>
#include <cuda.h>

#include "kmm/matrix.h"
#include "config.h"

#pragma once

template<uint32_t MaxFactorsT>
struct KMMProblemT {
  static const uint32_t MaxFactors = MaxFactorsT;
public:
  using Factors = FactorArray<MaxFactors>;

private:
  Factors factors;
  Matrix in;
  Matrix out;

public:
  KMMProblemT(Matrix x, Factors fs, Matrix y) :
    in(x), factors(fs), out(y) {}

  KMMProblemT(Matrix x, int n, const Factor* fs, Matrix y) :
    in(x), factors(fs, n), out(y) {}

  KMMProblemT(const uint m, const uint32_t n, const uint32_t *ps, const uint32_t *qs, 
             void* xptr, void* const* fsptr, void* yptr, const int k, const int l) : 
             in(m, k, xptr), out(m, l, yptr), factors(n, ps, qs, fsptr) {}
  
  KMMProblemT(const uint m, const int n, const uint *ps, const uint *qs,
             void* x, void* const* fs, void* y) :
    KMMProblemT(m, n, ps, qs, x, fs, y,
               std::reduce(ps, ps+n, 1, std::multiplies<uint>()),
               std::reduce(qs, qs+n, 1, std::multiplies<uint>())) {}

  KMMProblemT(const uint m, const int n, const uint *ps, const uint *qs) :
    KMMProblemT(m, n, ps, qs, nullptr, nullptr, nullptr) {}

  KMMProblemT rsub(int rstart, int subn) const {    
    int subk = x().n(), subl = y().n();
    for (int i = 0; i <= rstart - subn; i++) {
      subl = (subl/factors[i].q())*factors[i].p();
    }
    for (int i = n() - 1; i > rstart; i--) {
      subk = (subk/factors[i].p())*factors[i].q();
    }

    assert (rstart >= 0);
    assert (subn <= n());
    assert (rstart - (subn - 1) >= 0);

    return KMMProblemT(Matrix(x().m(), subk, x().data()),
                      factors.sub(rstart - (subn - 1), subn),
                      Matrix(y().m(), subl, y().data()));
  }

  KMMProblemT sub(int start, int subn) const {
    int subk = x().n(), subl = y().n();
    
    for (int i = 0; i < start; i++) {
      subl = (subl/factors[i].q())*factors[i].p();
    }
    for (int i = n() - 1; i >= start + subn; i--) {
      subk = (subk/factors[i].p())*factors[i].q();
    }

    assert (start >= 0);
    assert (subn <= n());
    assert (start + (subn - 1) <= n());

    return KMMProblemT(Matrix(x().m(), subk, x().data()),
                      factors.sub(start, subn),
                      Matrix(y().m(), subl, y().data()));
  }

  template<uint32_t OtherMaxFactors>
  KMMProblemT(const KMMProblemT<OtherMaxFactors>& other) : 
    in(other.x()), out(other.y()), factors(other.fs(), other.n()) {

  }

  uint32_t* ps(uint32_t *array) {
    for (uint32_t i = 0; i < n(); i++) {
      array[i] = factors[i].p();
    }
    return array;
  }

  uint32_t* qs(uint32_t *array) {
    for (uint32_t i = 0; i < n(); i++) {
      array[i] = factors[i].q();
    }
    return array;
  }

  void swap(void* temp1, void* temp2) {
    void* x1 = y().data();
    void* y1;
    if (x1 == temp1) {        
      y1 = temp2;
    } else if (x1 == temp2) {
      y1 = temp1;
    }

    in = Matrix(x().m(), x().n(), x1);
    out = Matrix(y().m(), y().n(), y1);
  }

  CUDA_DEVICE_HOST
  const Matrix& x() const {return in;}
  CUDA_DEVICE_HOST
  const Matrix& y() const {return out;}  
  CUDA_DEVICE_HOST
  const Factor& f(int i) const {return factors[i];}
  CUDA_DEVICE_HOST
  const Factor* fs() const {return &factors.array[0];}
  
  CUDA_DEVICE_HOST
  uint32_t k() const {return x().n();}
  CUDA_DEVICE_HOST
  uint32_t l() const {return y().n();}
  CUDA_DEVICE_HOST
  uint32_t m() const {return x().m();}
  CUDA_DEVICE_HOST
  uint32_t n() const {return factors.len();}
  
  bool operator==(const KMMProblemT& other) const {
    bool eq = x() == other.x() && n() == other.n() && y() == other.y();
    if (eq) {
      for (int i = 0; i < n(); i++) {
        eq = eq && factors[i] == other.factors[i];
      }
    }
    return eq;
  }

  bool sameFactorShapes() const {
    bool eq = true;
    for (int i = 1; i < n(); i++) {
      eq = eq && factors[i-1].p() == factors[i].p() &&
                 factors[i-1].q() == factors[i].q();
    }

    return eq;
  }

  friend std::ostream& operator<<(std::ostream &out, const KMMProblemT &problem) {
    out << problem.x().m() << "*(";
    if (problem.sameFactorShapes()) 
      out << problem.factors[0] << "^" << problem.n();
    else
      for (int i = 0; i < problem.n(); i++) {
        out << problem.factors[i];
        if (i < problem.n() - 1) out << "(x)";
      }
    out << ")";
    return out;
  }
};

using KMMProblem = KMMProblemT<64>;

template<>
struct std::hash<KMMProblem> {
  std::size_t operator()(const KMMProblem& k) const;
};

cudaError_t executeGeKMM(const KMMProblem problem, void* temps[2],
                         uint32_t swaps,
                         std::function<uint (const KMMProblem)> next,
                         std::function<cudaError_t (const KMMProblem, int, void*[2], Matrix)> func);
cudaError_t reverseExecuteGeKMM(const KMMProblem problem, void* temps[2],
                                Matrix result,
                                std::function<uint (const KMMProblem)> next,
                                std::function<cudaError_t (const KMMProblem, int, void*[2], Matrix)> func);
bool checkDistributedKronSizes(const KMMProblem problem,
                               const uint LocalKrons, const uint gpusInK);