#include "kmmalgo.h"

cudaError_t executeGeKMM(KMMProblem& problem, void* temp1,
                         void* temp2, 
                         std::function<uint (KMMProblem&, void*, void*, cudaError_t&)> func) {
  uint k = problem.k;
  size_t l = k;
  int nextF = 1;
  for (int i = problem.shape.n - 1; i >= 0; i = i - nextF) {
    l = (k/problem.shape.ps[i])*problem.shape.qs[i];
    
    KMMProblem subProblem(problem, i, i+1, k, l);

    cudaError_t e;
    nextF = func(subProblem, temp1, temp2, e);
    
    if (e != cudaSuccess) return e;
    k = l;
  }

  return cudaSuccess;
}