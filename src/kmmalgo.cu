#include <stdio.h>

#include "kmmalgo.h"

cudaError_t executeGeKMM(const KMMProblem problem, void* temps[2],
                         void* result,
                         std::function<uint (const KMMProblem)> next,
                         std::function<cudaError_t (const KMMProblem, void*[2], void*)> func) {
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
    if(i < nextF) {
      ptrs = GeKMMPtrs(ptrs.x, ptrs.fs, result);
    }
    auto subProblem = problem.rsub(ptrs, ps, qs, fs, i, nextF, k, l);
    err = func(subProblem, temps, result);
    if (err != cudaSuccess) break;
    k = l;
    if (temps != nullptr)
      ptrs = ptrs.swap(temps[0], temps[1]);
  }

  return cudaSuccess;
}