#include "kmmalgo.h"

std::size_t std::hash<KMMProblem>::operator()(const KMMProblem& problem) const {
  std::size_t h = hash<uint>()(problem.k()) ^ hash<uint>()(problem.n());
  for (int i = 0; i < problem.n(); i++) {
    h = h ^ problem.f(i).hash();
  }
  return h;
}

bool checkDistributedKronSizes(const uint NumKronMats, 
                               const uint M, const uint N, const uint K, 
                               const uint KronMatCols[], const uint KronMatRows[],
                               const uint LocalKrons, const uint gpusInK) {
  uint prevTempN = K;
  
  if (prevTempN % gpusInK != 0) return false;
    
  for (uint i = 0; i < NumKronMats; i += LocalKrons) {
    const uint kronMat = NumKronMats - i - 1;
    uint currTempN = prevTempN;
    for (int k = 0; k < std::min(LocalKrons, NumKronMats - i); k++) {
      currTempN = (currTempN/KronMatRows[kronMat - k])*KronMatCols[kronMat - k];
    }
  
    if (currTempN % gpusInK != 0) return false;
    prevTempN = currTempN;
  }

  return true;
}

bool checkDistributedKronSizes(const KMMProblem problem, const uint LocalN,
                               const uint gpusInK) {
  //Cannot do more than N local slicedmuls
  if (LocalN > problem.n()) return false;

  //If Row is divided among then local slicedmuls has to be less than N 
  if (gpusInK > 1 and LocalN >= problem.n()) return false;
  
  bool correct = true;

  executeGeKMM(problem, nullptr, 0,
    [](const KMMProblem kmm) {return 1;},
    [&correct, gpusInK](const KMMProblem kmm, int, void* t1, Matrix result) {
      correct = correct && (kmm.l() % gpusInK == 0);
      return cudaSuccess;
    });

  return correct;
}

//TODO: Change to backwardGeKMM
cudaError_t executeGeKMM(KMMProblem problem, void* tmps[2],
                         uint32_t swaps,
                         std::function<uint (const KMMProblem)> next,
                         std::function<cudaError_t (const KMMProblem, int rstart, void*[2], Matrix)> func) {
  int nextF = 1;

  void* firstIterOut;

  if (tmps != nullptr) {
    if (tmps[1] == nullptr) {
      if (swaps % 2 == 1) {
        tmps[1] = tmps[0];
        tmps[0] = problem.y().data();
      } else {
        tmps[0] = tmps[0];
        tmps[1] = problem.y().data();
      }
    }
    firstIterOut = tmps[0];
  } else {
    firstIterOut = problem.y().data();
  }

  Matrix result = problem.y();
  problem = KMMProblem(problem.x(), problem.n(), problem.fs(),
                       Matrix(problem.m(), problem.l(), firstIterOut));
  cudaError_t err;
  for (int i = problem.n() - 1; i >= 0; i = i - nextF) {
    nextF = next(problem);
    nextF = std::min(nextF, i+1);
    if (i < nextF) problem = KMMProblem(problem.x(), problem.n(), problem.fs(), result);
    err = func(problem.rsub(i, nextF), i, tmps, result);
    if (err != cudaSuccess) break;
    if (tmps != nullptr)
      problem.swap(tmps[0], tmps[1]);
  }

  return cudaSuccess;
}

//TODO: Change to forwardGeKMM
cudaError_t reverseExecuteGeKMM(KMMProblem problem, void* tmps[2],
                                Matrix result,
                                std::function<uint (const KMMProblem)> next,
                                std::function<cudaError_t (const KMMProblem, int start, void*[2], Matrix)> func) {
  int nextF = 1;
  cudaError_t err;
  for (int i = 0; i < problem.n(); i = i + nextF) {
    nextF = next(problem);
    if (i < nextF) problem = KMMProblem(problem.x(), problem.n(), problem.fs(), result);
    err = func(problem.rsub(i, nextF), i, tmps, result);
    if (err != cudaSuccess) break;
    if (tmps != nullptr)
      problem.swap(tmps[0], tmps[1]);
  }

  return cudaSuccess;
}