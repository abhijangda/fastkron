#include <functional>
#include <cassert>
#include <iostream>
#include <numeric>
#include <initializer_list>

#include "kmm/matrix.h"
#include "config.h"
#include "fastkron.h"

#pragma once

enum FastKronMMType {
  KMM,
  MKM
};

std::string strOfFastKronMMType(FastKronMMType mmtype);

/**
 * KMMProblemBase represents a Kronecker Matrix Matrix Multiplication problem.
 * This class takes a maximum factors (MaxFactors) as a template and contains
 * X, and Z as matrices, element type, OpX, OpF, and an array of factors.
 */
template<typename MatrixT, typename FactorT, uint32_t kMaxFactors>
class KMMProblemBase {
public:
  static const uint32_t MaxFactors = kMaxFactors;

  using Matrix = MatrixT;
  using Factor = FactorT;
  using Factors = FactorArrayBase<FactorT, MaxFactors>;

protected:
  /**
   * @eltype: Element data type
   * @in: Input matrix, i.e. X
   * @opIn: fastKron op on X
   * @opFactors: fastkron op on Factors
   * @out: Output matrix, i.e. Z
   * @factors: Array of all kronecker factors.
   */
  FastKronMMType kronType;
  FastKronType eltype;
  Matrix in;
  fastKronOp opIn;
  fastKronOp opFactors;
  Matrix out;
  //On CUDA keep Factors at the end of class to get
  //best performance
  Factors factors;
  
public:
  KMMProblemBase(FastKronMMType kronType, FastKronType eltype,
                 Matrix x, fastKronOp opX, Factors fs, fastKronOp opFs, Matrix y) :
              kronType(kronType), eltype(eltype), in(x), opIn(opX), opFactors(opFs), out(y),
              factors(fs) {}

  KMMProblemBase(FastKronMMType kronType, FastKronType eltype,
                 Matrix x, fastKronOp opX, 
                 int n, const Factor* fs, fastKronOp opFs, Matrix y) :
              kronType(kronType), eltype(eltype), in(x), opIn(opX), 
              opFactors(opFs), out(y), factors(fs, n) {}

  KMMProblemBase(FastKronMMType kronType, FastKronType eltype, 
                 Matrix x, fastKronOp opX, 
                 std::initializer_list<Factor> fs, fastKronOp opFs, Matrix y) :
              KMMProblemBase(kronType, eltype, x, opX, Factors(fs), opFs, y) {}

  // KMMProblemBase(FastKronType eltype, const uint m, const uint32_t n,
  //             const uint32_t *ps, const uint32_t *qs, void* xptr, 
  //             fastKronOp opX, void* const* fsptr, fastKronOp opFs, void* yptr,
  //             const int k, const int l) :
  //             eltype(eltype), in(m, k, xptr), opIn(opX), opFactors(opFs), 
  //             out(m, l, yptr), factors(n, ps, qs, fsptr) {}
  
  // KMMProblemBase(FastKronType eltype, const uint m, const int n,
  //             const uint *ps, const uint *qs,
  //             void* x, fastKronOp opX, void* const* fs, fastKronOp opFs, void* y) :
  //             KMMProblemBase(eltype, m, n, ps, qs, x, opX, fs, opFs, y,
  //                         std::reduce(ps, ps+n, 1, std::multiplies<uint>()),
  //                         std::reduce(qs, qs+n, 1, std::multiplies<uint>())) {}

  // KMMProblemBase(FastKronType eltype, const uint m, const int n, 
  //             const uint *ps, const uint *qs, fastKronOp opX, fastKronOp opFs) :
  //             KMMProblemBase(eltype, m, n, ps, qs, nullptr, 
  //                         opX, nullptr, opFs, nullptr) {}

  template<uint32_t OtherMaxFactors>
  KMMProblemBase(const KMMProblemBase<Matrix, Factor, OtherMaxFactors>& other) : 
              kronType(other.mmtype()), eltype(other.type()), in(other.x()), opIn(other.opX()),
              opFactors(other.opFs()), out(other.y()), factors(other.fs(),
              other.n()) {}

  static uint32_t getK(const uint32_t* ps, const uint32_t n) {
    return std::reduce(ps, ps+n, 1, std::multiplies<uint>());
  }

  static uint32_t getL(const uint32_t* qs, const uint32_t n) {
    return std::reduce(qs, qs+n, 1, std::multiplies<uint>());
  }

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
  CUDA_DEVICE_HOST
  FastKronMMType mmtype()  const {return kronType;}

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

  KMMProblemBase setFirstIterOutput(const Matrix y) const {
    return KMMProblemBase(mmtype(), type(), x(), opX(), n(), fs(), opFs(), y);
  }

  KMMProblemBase updateY(const Matrix y) const {
    return KMMProblemBase(mmtype(), type(), x(), opX(), n(), fs(), opFs(), y);
  }
  
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
   * rsub() - Get a subproblem from an index from end of given length.
   *          subproblem's k and l are updated.
   * @rstart: Index start from the end.
   * @subn: length of the subproblem from rstart.
   *
   */
  KMMProblemBase rsub(uint32_t rstart, uint32_t subn) const {
    assert (subn <= n());
    assert (rstart >= (subn - 1));

    uint32_t subk = x().n(), subl = y().n();
    if (true || mmtype() == FastKronMMType::MKM) {
      if (rstart >= subn) {
        for (uint32_t i = 0; i <= rstart - subn; i++) {
          subl = (subl/factors[i].q())*factors[i].p();
        }
      }
      for (uint32_t i = n() - 1; i > rstart; i--) {
        subk = (subk/factors[i].p())*factors[i].q();
      }
    } else {
      if (rstart >= subn) {
        for (uint32_t i = 0; i <= rstart - subn; i++) {
          subk = (subk/factors[i].p())*factors[i].q();
        }
      }
      for (uint32_t i = n() - 1; i > rstart; i--) {
        subl = (subl/factors[i].q())*factors[i].p();
      }
    }

    
    return KMMProblemBase(mmtype(), type(), x().sameRows(subk), opX(),
                          factors.sub(rstart - (subn - 1), subn), opFs(),
                          y().sameRows(subl));
  }

  /**
   * sub() - Get a subproblem from an index from start of given length.
   *          subproblem's k and l are updated.
   * @start: Index start from the start.
   * @subn: length of the subproblem from rstart.
   *
   */
  KMMProblemBase sub(uint32_t start, uint32_t subn) const {
    uint32_t subk = x().n(), subl = y().n();

    assert(start <= n());
    assert(subn <= n());
    assert(start + (subn - 1) <= n());
    
    if (true || mmtype() == FastKronMMType::MKM) {
      for (uint32_t i = 0; i < start; i++) {
        subl = (subl/factors[i].q())*factors[i].p();
      }
      for (uint32_t i = n() - 1; i >= start + subn; i--) {
        subk = (subk/factors[i].p())*factors[i].q();
      }
    } else if (mmtype() == FastKronMMType::KMM) {
      for (uint32_t i = n() - 1; i >= start + subn; i--) {
        subl = (subl/factors[i].q())*factors[i].p();
      }
      for (uint32_t i = 0; i < start; i++) {
        subk = (subk/factors[i].p())*factors[i].q();
      }
    }

    return KMMProblemBase(mmtype(), type(), x().sameRows(subk), opX(),
                          factors.sub(start, subn), opFs(),
                          y().sameRows(subl));
  }

  template<uint NumFactors>
  KMMProblemBase<MatrixT, FactorT, NumFactors> factorSlice() {
    return KMMProblemBase<MatrixT, FactorT, NumFactors>(*this);
  }

  void initMMIter(int i, bool isFirstIter, bool isLastIter) {}

  /**
   * swap() - Swap x and y pointers based on temporary pointers.
   */
  void swap(void* temp1, void* temp2) {
    void* x1 = y().data();
    void* y1 = nullptr;
    if (x1 == temp1) {        
      y1 = temp2;
    } else if (x1 == temp2) {
      y1 = temp1;
    }

    in = x().like(x1);
    out = y().like(y1);
  }

  bool operator==(const KMMProblemBase& other) const {
    bool eq = mmtype() == other.mmtype() &&
              x() == other.x() && opX() == other.opX()   &&
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

  friend std::ostream& operator<<(std::ostream &out, const KMMProblemBase &problem) {
    if (problem.mmtype() == FastKronMMType::MKM) {
      out << problem.x().m() << "x" << problem.k() << "*(";
      if (problem.sameFactorShapes()) 
        out << problem.factors[0] << "^" << problem.n();
      else
        for (uint32_t i = 0; i < problem.n(); i++) {
          out << problem.factors[i];
          if (i < problem.n() - 1) out << "(x)";
        }
      out << ")";
    } else if (problem.mmtype() == FastKronMMType::KMM) {
      out << "(";
      if (problem.sameFactorShapes()) 
        out << problem.factors[0] << "^" << problem.n();
      else
        for (uint32_t i = 0; i < problem.n(); i++) {
          out << problem.factors[i];
          if (i < problem.n() - 1) out << "(x)";
        }
      out << ")*" << problem.k() << "x" << problem.x().m();
    } 
    out << "_" << problem.opX() << problem.opFs() << "_" << strOfFastKronType(problem.type());
    return out;
  }
};

/**
 * KMMProblem - defines a default KMMProblemBase with MaxFactors=64 with normal Matrix and Factor
 */
template<uint32_t kMaxFactors>
using KMMProblemT = KMMProblemBase<Matrix, Factor, kMaxFactors>;
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

template<uint32_t kMaxFactors>
class KMMProblemStridedBatchedT : public KMMProblemBase<StridedBatchMatrix, StridedBatchFactor, kMaxFactors> {
private:
  using Base = KMMProblemBase<StridedBatchMatrix, StridedBatchFactor, kMaxFactors>;

public:
  using Factors = typename Base::Factors;
  using Matrix = typename Base::Matrix;
  using Factor = typename Base::Factor;

protected:
  int batches;
  KMMProblemStridedBatchedT(Base base, int batchCount) : Base(base), batches(batchCount) {}

public:
  KMMProblemStridedBatchedT(FastKronMMType kronType, FastKronType eltype, 
                            Matrix x, fastKronOp opX, Factors fs, fastKronOp opFs,
                            Matrix y, int batchCount) :
                            Base(kronType, eltype, x, opX, fs, opFs, y),
                            batches(batchCount) {}

  KMMProblemStridedBatchedT(FastKronMMType kronType, FastKronType eltype,
                            Matrix x, fastKronOp opX, 
                            int n, const Factor* fs, fastKronOp opFs, Matrix y,
                            int batchCount) :
                            Base(kronType, eltype, x, opX, n, fs, opFs, y),
                            batches(batchCount) {}

  KMMProblemStridedBatchedT(FastKronMMType kronType, FastKronType eltype,
                            Matrix x, fastKronOp opX, 
                            std::initializer_list<Factor> fs, fastKronOp opFs, Matrix y,
                            int batchCount) :
                            Base(kronType, eltype, x, opX, fs, opFs, y), batches(batchCount) {}
  
  template<uint32_t OtherMaxFactors>
  KMMProblemStridedBatchedT(const KMMProblemStridedBatchedT<OtherMaxFactors>& other) : 
              Base(other.mmtype(), other.type(), other.x(), other.opX(), 
                   other.n(), other.fs(), other.opFs(), other.y()),
              batches(other.batchCount()) {}

  KMMProblemStridedBatchedT rsub(uint32_t rstart, uint32_t subn) const {
    return KMMProblemStridedBatchedT(Base::rsub(rstart, subn), batches);
  }

  KMMProblemStridedBatchedT sub(uint32_t start, uint32_t subn) const {
    return KMMProblemStridedBatchedT(Base::sub(start, subn), batches);
  }
  
  KMMProblemStridedBatchedT setFirstIterOutput(const Matrix y) const {
    return KMMProblemStridedBatchedT(this->mmtype(), this->type(), this->x(), this->opX(), 
                                     this->n(), this->fs(), this->opFs(),
                                     y, batches);
  }

  KMMProblemStridedBatchedT updateY(const Matrix y) const {
    return KMMProblemStridedBatchedT(this->mmtype(), this->type(), this->x(), this->opX(), 
                                     this->n(), this->fs(), this->opFs(),
                                     y, batches);
  }

  template<typename T>
  KMMProblem batchProblem(uint b) const {
    typename Factor::Base baseFs[this->n()];

    for (uint32_t i = 0; i < this->n(); i++) {
      auto fi = this->f(i).template batch<T>(b);
      baseFs[i] = fi;
    }

    return KMMProblem(this->mmtype(), this->type(), this->x().template batch<T>(b), this->opX(),
                      this->n(), baseFs, this->opFs(), this->y().template batch<T>(b));
  }

  KMMProblem batchProblem(uint batch) const {
    switch (this->type()) {
      case FastKronFloat:
        return batchProblem<float>(batch);
      case FastKronDouble:
        return batchProblem<double>(batch);
      case FastKronInt:
        return batchProblem<int>(batch);
      default:
        std::cout << "Not Implemented for Type " << this->type() << std::endl;
        assert (false);
    }

    return batchProblem<float>(batch);
  }

  uint batchCount() const {return batches;}

  template<uint NumFactors>
  KMMProblemStridedBatchedT<NumFactors> factorSlice() {
    return KMMProblemStridedBatchedT<NumFactors>(*this);
  }

  void initMMIter(int i, bool isFirstIter, bool isLastIter) {
    if (isFirstIter ^ isLastIter || !isFirstIter && !isLastIter) {
      if (isFirstIter) {
        //Set output's batchstride as num elems of output matrix because output is a temp
        this->out = KMMProblemStridedBatchedT::Matrix(this->y().m(), this->y().n(), 
                                                      this->m() * this->l(), this->y().data());
      } else if (isLastIter) {
        //Set input's batchstride as num elems of input matrix because input is a temp
        this->in = KMMProblemStridedBatchedT::Matrix(this->x().m(), this->x().n(), 
                                                      this->m() * this->k(), this->x().data());
      } else if (!isFirstIter && !isLastIter) {
        //Do above things
        this->in = KMMProblemStridedBatchedT::Matrix(this->x().m(), this->x().n(), 
                                                      this->m() * this->k(), this->x().data());
        this->out = KMMProblemStridedBatchedT::Matrix(this->y().m(), this->y().n(), 
                                                      this->m() * this->l(), this->y().data());
      }
    }
  }

  void swap(void* temp1, void* temp2) {
    Base::swap(temp1, temp2);
    this->in = KMMProblemStridedBatchedT::Matrix(this->x().m(), this->x().n(), 
                                                 this->y().batchStride(), this->x().data());
  }

  CUDA_DEVICE_HOST
  size_t flop() const {
    return Base::flop() * batchCount();
  }

  friend std::ostream& operator<<(std::ostream &out, const KMMProblemStridedBatchedT &problem) {
    out << (const Base&)(problem);
    out << "_" << "stridedBatched" << "_" << problem.batchCount();
    return out;
  }
};

using KMMProblemStridedBatched = KMMProblemStridedBatchedT<64>;

template<>
struct std::hash<KMMProblemStridedBatched> {
  std::size_t operator()(const KMMProblemStridedBatched& k) const;
};

/**
 * executeGeMM() - Execute a function on the problem using the MKM/KMM algorithm. 
 *                  See paper for more details on the algorithm.
 * @problem: The problem to execute MKM/KMM algorithm.
 * @temps: Two Temporary buffers.
 * @swaps: Number of swaps of temporary buffers that will happen.
 * @next: The function to get next factor to compuate.
 * @func: The compute function for subproblem.
 *
 * Return - fastKronSuccess if succesfull otherwise the error.
 */

fastKronError executeGeMM(const KMMProblem problem, void* temps[2],
                           uint32_t swaps,
                           std::function<uint (const KMMProblem)> next,
                           std::function<fastKronError (const KMMProblem, int, void*[2], typename KMMProblem::Matrix)> func);

fastKronError executeGeMM(const KMMProblemStridedBatched problem, void* temps[2],
                           uint32_t swaps,
                           std::function<uint (const KMMProblemStridedBatched)> next,
                           std::function<fastKronError (const KMMProblemStridedBatched, int, void*[2], typename KMMProblemStridedBatched::Matrix)> func);

/**
 * reverseExecuteGeMM() - Execute a function on the problem using the reverse MKM/KMM algorithm
 *                         that starts from the first factor.
 * @problem: The problem to execute MKM/KMM algorithm.
 * @temps: Two Temporary buffers.
 * @swaps: Number of swaps of temporary buffers that will happen.
 * @next: The function to get next factor to compuate.
 * @func: The compute function for subproblem.
 *
 * Return - fastKronSuccess if succesfull otherwise the error.
 */
fastKronError reverseExecuteGeMM(const KMMProblem problem, void* temps[2],
                                typename KMMProblem::Matrix result,
                                std::function<uint (const KMMProblem)> next,
                                std::function<fastKronError (const KMMProblem, int, void*[2], typename KMMProblem::Matrix)> func);

fastKronError reverseExecuteGeMM(const KMMProblemStridedBatched problem, void* temps[2],
                                typename KMMProblemStridedBatched::Matrix result,
                                std::function<uint (const KMMProblemStridedBatched)> next,
                                std::function<fastKronError (const KMMProblemStridedBatched, int, void*[2], typename KMMProblemStridedBatched::Matrix)> func);

bool checkDistributedKronSizes(const KMMProblem problem,
                               const uint LocalKrons, const uint gpusInK);