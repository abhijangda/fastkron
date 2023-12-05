#include <stdio.h>

#include "kmmalgo.h"

cudaError_t executeGeKMM(KMMProblem problem, void* temps[2],
                         void* result,
                         std::function<uint (const KMMProblem)> next,
                         std::function<cudaError_t (const KMMProblem, int rstart, void*[2], void*)> func) {
  uint k = problem.k;
  size_t l = k;
  int nextF = 1;
  cudaError_t err;

  for (int i = problem.n - 1; i >= 0; i = i - nextF) {
    nextF = next(problem);
    nextF = std::min(nextF, i+1);
    if (i < nextF) {
      problem.y = result;
    }
    auto subProblem = problem.rsub(i, nextF);
    err = func(subProblem, i, temps, result);
    if (err != cudaSuccess) break;
    k = l;
    if (temps != nullptr)
      problem.swap(temps[0], temps[1]);
  }

  return cudaSuccess;
}

cudaError_t reverseExecuteGeKMM(KMMProblem problem, void* temps[2],
                                void* result,
                                std::function<uint (const KMMProblem)> next,
                                std::function<cudaError_t (const KMMProblem, int start, void*[2], void*)> func) {
  int nextF = 1;
  cudaError_t err;
  
  for (int i = 0; i < problem.n; i = i + nextF) {
    nextF = next(problem);
    if (i < nextF) {
      problem.y = result;
    }
    auto subProblem = problem.sub(i, nextF);
    err = func(subProblem, i, temps, result);
    if (err != cudaSuccess) break;
    if (temps != nullptr)
      problem.swap(temps[0], temps[1]);
  }

  return cudaSuccess;
}