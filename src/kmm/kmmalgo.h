#include <functional>
#include <cassert>
#include <iostream>

#include <cuda_runtime.h>
#include <cuda.h>

#pragma once

struct KMMProblem {
  int k;
  int l;

  uint m;
  int n;
  
  static const int MaxN = 64;
  
  uint qs[MaxN];
  uint ps[MaxN];
  
  void * x;
  void * fs[MaxN];
  void * y;

  KMMProblem(const uint m, const int n, const uint *ps, const uint *qs, 
             void* x, void* const* fs, void* y, const int k, 
             const int l) : m(m), n(n), x(x), y(y), k(k), l(l) {
    assert (n < MaxN);
    for (int i = 0; i < n; i++) {
      this->ps[i] = ps[i];
      this->qs[i] = qs[i];
      if (fs)
        this->fs[i] = fs[i];
      else
        this->fs[i] = nullptr;
    }

    for (int i = n; i < MaxN; i++) {
      this->ps[i] = this->qs[i] = 0;
      this->fs[i] = nullptr;
    }
  }
  
  KMMProblem(const uint m, const int n, const uint *ps, const uint *qs,
             void* x, void* const* fs, void * y) :
    KMMProblem(m, n, ps, qs, x, fs, y, 1, 1) {
    k = 1;
    l = 1;
    for (int i = 0; i < n; i++) {
      k *= ps[i];
      l *= qs[i];
    }
  }

  KMMProblem(const uint m, const int n, const uint *ps, const uint *qs) :
    KMMProblem(m, n, ps, qs, nullptr, nullptr, nullptr) {}

  KMMProblem(KMMProblem problem, const int k, const int l) :
    KMMProblem(problem.m, problem.n, problem.ps, problem.qs, 
               problem.x, problem.fs, problem.y, k, l) {}

  KMMProblem(KMMProblem problem, void* x, void** fs, void* y) :
    KMMProblem(problem.m, problem.n, problem.ps, problem.qs, x, fs, y) {}

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

  friend std::ostream& operator<<(std::ostream &out, const KMMProblem &problem) {
    out << problem.m << "* (";
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