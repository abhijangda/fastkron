#include <functional>
#include <cassert>
#include <stdio.h>

#pragma once

struct KMMProblem {
  //TODO: Remove rstart
  const int rstart;
  int k;
  int l;

  const uint m;
  const int n;
  const uint *qs;
  const uint *ps;
  
  void * x;
  void ** fs;
  void * y;

  KMMProblem(const uint m, const int n, const uint *ps, const uint *qs, 
             void* x, void ** fs, void* y, int rstart,
             const int k, const int l) : m(m), n(n), ps(ps), qs(qs),
             x(x), fs(fs), y(y), rstart(rstart), k(k), l(l) {
    assert (rstart >= 0);
  }
  
  KMMProblem(const uint m, const int n, const uint *ps, const uint *qs,
             void* x, void ** fs, void * y) :
    KMMProblem(m, n, ps, qs, x, fs, y, 0, 1, 1) {
    k = 1;
    l = 1;
    for (int i = 0; i < n; i++) {
      k *= ps[i];
      l *= qs[i];
    }
  }

  KMMProblem(const uint m, const int n, const uint *ps, const uint *qs) :
    KMMProblem(m, n, ps, qs, nullptr, nullptr, nullptr) {}

  KMMProblem(KMMProblem problem, int rstart,
    const int k, const int l) :
    KMMProblem(problem.m, problem.n, problem.ps, problem.qs, 
               problem.x, problem.fs, problem.y, rstart, k, l) {}
  
  KMMProblem rsub(uint ps[], uint qs[], void* fs[], 
                  int rstart, int subn) const {
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
      ps[i]  = this->ps[rstart - i];
      qs[i]  = this->qs[rstart - i];
    }

    if (this->fs && fs) {
      for (int i = 0; i < subn; i++) {
        fs[i] = this->fs[rstart - i];
      }
    }

    return KMMProblem(m, subn, ps, qs,
                      x, fs, y, rstart, subk, subl);
  }

  KMMProblem sub(uint ps[], uint qs[], void* fs[], 
                 int start, int subn) const {
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

    if (this->fs && fs) {
      for (int i = 0; i < subn; i++) {
        fs[i] = this->fs[start + i];
      }
    }

    return KMMProblem(m, subn, ps, qs,
                      x, fs, y, start, subk, subl);
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
};

cudaError_t executeGeKMM(const KMMProblem problem, void* temps[2],
                         void* result,
                         std::function<uint (const KMMProblem)> next,
                         std::function<cudaError_t (const KMMProblem, void*[2], void*)> func);
cudaError_t reverseExecuteGeKMM(const KMMProblem problem, void* temps[2],
                                void* result,
                                std::function<uint (const KMMProblem)> next,
                                std::function<cudaError_t (const KMMProblem, void*[2], void*)> func);