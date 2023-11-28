#include <stdio.h>

#include "kmmalgo.h"

cudaError_t executeGeKMM(const KMMProblem problem, void* temp1,
                         void* temp2,
                         std::function<uint (const KMMProblem)> next,
                         std::function<cudaError_t (const KMMProblem, void*, void*)> func) {
  uint k = problem.k;
  size_t l = k;
  int nextF = 1;
  GeKMMPtrs ptrs = problem.ptrs;
  cudaError_t err;

  for (int i = problem.shape.n - 1; i >= 0; i = i - nextF) {
    nextF = next(problem);
    for (int f = i; f > i - nextF; f--) {
      l = (l/problem.shape.ps[f])*problem.shape.qs[f];
    }
    uint qs[problem.shape.n];
    uint ps[problem.shape.n];
    void* fs[problem.shape.n];

    auto subProblem = problem.rsub(ptrs, ps, qs, fs, i, nextF, k, l);
    err = func(subProblem, temp1, temp2);
    if (err != cudaSuccess) break;
    k = l;
    ptrs = ptrs.swap(temp1, temp2);
  }

  return cudaSuccess;
}