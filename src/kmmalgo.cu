#include <stdio.h>

#include "kmmalgo.h"

cudaError_t executeGeKMM(const KMMProblem problem, void* temp1,
                         void* temp2, 
                         std::function<uint (const KMMProblem, void*, void*)> func) {
  uint k = problem.k;
  size_t l = k;
  int nextF = 1;
  for (int i = problem.shape.n - 1; i >= 0; i = i - nextF) {
    l = (k/problem.shape.ps[i])*problem.shape.qs[i];
    KMMProblem subProblem(problem, i, i+1, k, l);
    nextF = func(subProblem, temp1, temp2);
    k = l;
  }

  return cudaSuccess;
}