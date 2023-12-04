#include <stdio.h>

#include "kmmalgo.h"

cudaError_t executeGeKMM(KMMProblem problem, void* temps[2],
                         void* result,
                         std::function<uint (const KMMProblem)> next,
                         std::function<cudaError_t (const KMMProblem, void*[2], void*)> func) {
  uint k = problem.k;
  size_t l = k;
  int nextF = 1;
  cudaError_t err;

  for (int i = problem.n - 1; i >= 0; i = i - nextF) {
    nextF = next(problem);
    nextF = std::min(nextF, i+1);
    for (int f = i; f > i - nextF; f--) {
      l = (l/problem.ps[f])*problem.qs[f];
    }
    if (i < nextF) {
      problem.y = result;
    }
    uint qs[problem.n];
    uint ps[problem.n];
    void* fs[problem.n];
    auto subProblem = problem.rsub(ps, qs, fs, i, nextF);
    assert (subProblem.k == k);
    assert (subProblem.l == l);
    err = func(subProblem, temps, result);
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
                                std::function<cudaError_t (const KMMProblem, void*[2], void*)> func) {
  int nextF = 1;
  cudaError_t err;
  
  for (int i = 0; i < problem.n; i = i + nextF) {
    nextF = next(problem);
    if (i < nextF) {
      problem.y = result;
    }
    uint qs[problem.n];
    uint ps[problem.n];
    void* fs[problem.n];
    auto subProblem = problem.sub(ps, qs, fs, i, nextF);
    err = func(subProblem, temps, result);
    if (err != cudaSuccess) break;
    if (temps != nullptr)
      problem.swap(temps[0], temps[1]);
  }

  return cudaSuccess;
}