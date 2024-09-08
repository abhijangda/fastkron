#include <functional>
#include <cassert>
#include <iostream>
#include <numeric>
#include <initializer_list>

#include "kmm/matrix.h"
#include "config.h"
#include "fastkron.h"

#pragma once

/**
 * KMMProblemT represents a Kronecker Matrix Matrix Multiplication problem.
 * This class takes a maximum factors (MaxFactors) as a template and contains
 * X, and Z as matrices, element type, OpX, OpF, and an array of factors.
 */
template<uint32_t kMaxFactors>
class KMMProblemT {
  static const uint32_t MaxFactors = kMaxFactors;
public:
  using Factors = FactorArray<MaxFactors>;

private:
  /**
   * @eltype: Element data type
   * @in: Input matrix, i.e. X
   * @opIn: fastKron op on X
   * @opFactors: fastkron op on Factors
   * @out: Output matrix, i.e. Z
   * @factors: Array of all kronecker factors.
   */
  FastKronType eltype;
  Matrix in;
  fastKronOp opIn;
  fastKronOp opFactors;
  Matrix out;
  //On CUDA keep Factors at the end of class to get
  //best performance
  Factors factors;
  
public:
  KMMProblemT(FastKronType eltype, Matrix x, fastKronOp opX, Factors fs,
              fastKronOp opFs, Matrix y) :
              eltype(eltype), in(x), opIn(opX), opFactors(opFs), out(y),
              factors(fs) {}

  KMMProblemT(FastKronType eltype, Matrix x, fastKronOp opX, int n,
              const Factor* fs, fastKronOp opFs, Matrix y) :
              eltype(eltype), in(x), opIn(opX), opFactors(opFs), out(y),
              factors(fs, n) {}

  KMMProblemT(FastKronType eltype, Matrix x, fastKronOp opX, 
              std::initializer_list<Factor> fs, fastKronOp opFs, Matrix y) :
              KMMProblemT(eltype, x, opX, Factors(fs), opFs, y) {}

  KMMProblemT(FastKronType eltype, const uint m, const uint32_t n,
              const uint32_t *ps, const uint32_t *qs, void* xptr, 
              fastKronOp opX, void* const* fsptr, fastKronOp opFs, void* yptr,
              const int k, const int l) :
              eltype(eltype), in(m, k, xptr), opIn(opX), opFactors(opFs), 
              out(m, l, yptr), factors(n, ps, qs, fsptr) {}
  
  KMMProblemT(FastKronType eltype, const uint m, const int n,
              const uint *ps, const uint *qs,
              void* x, fastKronOp opX, void* const* fs, fastKronOp opFs, void* y) :
              KMMProblemT(eltype, m, n, ps, qs, x, opX, fs, opFs, y,
                          std::reduce(ps, ps+n, 1, std::multiplies<uint>()),
                          std::reduce(qs, qs+n, 1, std::multiplies<uint>())) {}

  KMMProblemT(FastKronType eltype, const uint m, const int n, 
              const uint *ps, const uint *qs, fastKronOp opX, fastKronOp opFs) :
              KMMProblemT(eltype, m, n, ps, qs, nullptr, 
                          opX, nullptr, opFs, nullptr) {}

  //TODO: Also initialize opIn and opFactors, and eltype?
  template<uint32_t OtherMaxFactors>
  KMMProblemT(const KMMProblemT<OtherMaxFactors>& other) : 
              eltype(other.type()), in(other.x()), opIn(other.opX()),
              opFactors(other.opFs()), out(other.y()), factors(other.fs(),
              other.n()) {}

  /**
   * Getters for members
   */
  CUDA_DEVICE_HOST
  const Matrix& x()      const {return in;}
  CUDA_DEVICE_HOST
  const Matrix& y()      const {return out;}  
  CUDA_DEVICE_HOST
  const Factor& f(int i) const {return factors[i];}
  CUDA_DEVICE_HOST
  const Factor* fs()     const {return &factors.array[0];}
  CUDA_DEVICE_HOST
  fastKronOp opFs()      const {return opFactors;}
  CUDA_DEVICE_HOST
  fastKronOp opX()       const {return opIn;}
  CUDA_DEVICE_HOST
  FastKronType type()    const {return eltype;}
  
  /**
   * Get values for several problem notations.
   * K is cols of X
   * L is cols of Y
   * m is rows of X and Y
   * n is number of factors
   */
  CUDA_DEVICE_HOST
  uint32_t k() const {return x().n();}
  CUDA_DEVICE_HOST
  uint32_t l() const {return y().n();}
  CUDA_DEVICE_HOST
  uint32_t m() const {return x().m();}
  CUDA_DEVICE_HOST
  uint32_t n() const {return factors.len();}

  /**
   * ps() / qs() - Return and write rows / cols of all factors to given array.
   */
  uint32_t* ps(uint32_t *array) const {
    if (!array) return nullptr;
    for (uint32_t i = 0; i < n(); i++) {
      array[i] = factors[i].p();
    }
    return array;
  }

  uint32_t* qs(uint32_t *array) const {
    if (!array) return nullptr;
    for (uint32_t i = 0; i < n(); i++) {
      array[i] = factors[i].q();
    }
    return array;
  }

  /**
   * flop() - Return number of floating point operations performed by the problem.
   */
  CUDA_DEVICE_HOST
  size_t flop() const {
    size_t ops = 0;
    long kk = k();
    for (int i = n() - 1; i >= 0; i--) {
      kk = (kk/f(i).p()) * f(i).q();
      ops += kk * f(i).p();
    }

    return 2 * ((long)x().m()) * ops;
  }

  void setOpX(fastKronOp op) {opIn = op;}

  /**
   * rsub() - Get a subproblem from an index i from end of given length.
   *          ....
   */
  KMMProblemT rsub(uint32_t rstart, uint32_t subn) const {
    assert (subn <= n());
    assert (rstart >= (subn - 1));

    uint32_t subk = x().n(), subl = y().n();
    if (rstart >= subn) {
      for (uint32_t i = 0; i <= rstart - subn; i++) {
        subl = (subl/factors[i].q())*factors[i].p();
      }
    }
    for (uint32_t i = n() - 1; i > rstart; i--) {
      subk = (subk/factors[i].p())*factors[i].q();
    }

    
    return KMMProblemT(type(), Matrix(x().m(), subk, x().data()), opX(),
                       factors.sub(rstart - (subn - 1), subn), opFs(),
                       Matrix(y().m(), subl, y().data()));
  }

  KMMProblemT sub(uint32_t start, uint32_t subn) const {
    uint32_t subk = x().n(), subl = y().n();

    assert(start <= n());
    assert(subn <= n());
    assert(start + (subn - 1) <= n());
    
    for (uint32_t i = 0; i < start; i++) {
      subl = (subl/factors[i].q())*factors[i].p();
    }
    for (uint32_t i = n() - 1; i >= start + subn; i--) {
      subk = (subk/factors[i].p())*factors[i].q();
    }

    return KMMProblemT(type(), Matrix(x().m(), subk, x().data()), opX(),
                       factors.sub(start, subn), opFs(),
                       Matrix(y().m(), subl, y().data()));
  }

  void swap(void* temp1, void* temp2) {
    void* x1 = y().data();
    void* y1 = nullptr;
    if (x1 == temp1) {        
      y1 = temp2;
    } else if (x1 == temp2) {
      y1 = temp1;
    }

    in = Matrix(x().m(), x().n(), x1);
    out = Matrix(y().m(), y().n(), y1);
  }

  bool operator==(const KMMProblemT& other) const {
    bool eq = x() == other.x() && opX() == other.opX()   &&
              n() == other.n() && opFs() == other.opFs() &&
              y() == other.y();
    if (eq) {
      for (uint32_t i = 0; i < n(); i++) {
        eq = eq && factors[i] == other.factors[i];
      }
    }
    return eq;
  }

  bool sameFactorShapes() const {
    bool eq = true;
    for (uint32_t i = 1; i < n(); i++) {
      eq = eq && factors[i-1].p() == factors[i].p() &&
                 factors[i-1].q() == factors[i].q();
    }

    return eq;
  }

  friend std::ostream& operator<<(std::ostream &out, const KMMProblemT &problem) {
    out << problem.x().m() << "x" << problem.k() << "*(";
    if (problem.sameFactorShapes()) 
      out << problem.factors[0] << "^" << problem.n();
    else
      for (uint32_t i = 0; i < problem.n(); i++) {
        out << problem.factors[i];
        if (i < problem.n() - 1) out << "(x)";
      }
    out << ")";
    out << "_" << problem.opX() << problem.opFs() << "_" << strOfFastKronType(problem.type());
    return out;
  }
};

/**
 * KMMProblem - defines a default KMMProblemT with MaxFactors=64
 */
using KMMProblem = KMMProblemT<64>;

template<>
struct std::hash<KMMProblem> {
  std::size_t operator()(const KMMProblem& k) const;
};

// template<>
// const Matrix& std::max(const Matrix& x, const Matrix& y) {
//   if (x.numel() < y.numel()) return x;
//   return y;
// }

struct KMMProblemComparator {
  bool operator()(const KMMProblem& a, const KMMProblem& b) const {
    return std::hash<KMMProblem>()(a) < std::hash<KMMProblem>()(b);
  }
};

fastKronError executeGeKMM(const KMMProblem problem, void* temps[2],
                         uint32_t swaps,
                         std::function<uint (const KMMProblem)> next,
                         std::function<fastKronError (const KMMProblem, int, void*[2], Matrix)> func);
fastKronError reverseExecuteGeKMM(const KMMProblem problem, void* temps[2],
                                Matrix result,
                                std::function<uint (const KMMProblem)> next,
                                std::function<fastKronError (const KMMProblem, int, void*[2], Matrix)> func);
bool checkDistributedKronSizes(const KMMProblem problem,
                               const uint LocalKrons, const uint gpusInK);