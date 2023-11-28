#include <stdio.h>

#include "kmmalgo.h"

cudaError_t executeGeKMM(const KMMProblem problem, void* temp1,
                         void* temp2,
                         std::function<uint (const KMMProblem)> next,
                         std::function<uint (const KMMProblem, void*, void*)> func) {
  uint k = problem.k;
  size_t l = k;
  int nextF = 1;
  GeKMMPtrs ptrs = problem.ptrs;
  printf("ptrs-> f %p\n", ptrs.fs);
  for (int i = problem.shape.n - 1; i >= 0; i = i - nextF) {
    nextF = next(problem);
    for (int f = i; f >= i - nextF; f--) {
      l = (l/problem.shape.ps[f])*problem.shape.qs[f];
    }
    uint qs[problem.shape.n];
    uint ps[problem.shape.n];
    void* fs[problem.shape.n];

    //TODO: Do not need end only start
    auto subProblem = problem.rsub(ptrs, ps, qs, fs, i, nextF, k, l);
    func(subProblem, temp1, temp2);
    k = l;
    ptrs = ptrs.swap(temp1, temp2);
  }

  return cudaSuccess;
}