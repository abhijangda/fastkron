#include "kmmalgo.h"

std::size_t std::hash<KMMProblem>::operator()(const KMMProblem& shape) const {
  std::size_t h = hash<uint>()(shape.k()) ^ hash<uint>()(shape.n);
  for (int i = 0; i < shape.n; i++) {
    h = h ^ shape.fs[i].hash();
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
  bool correct = true;
  
  //Cannot do more than N local slicedmuls
  if (LocalN > problem.n) correct = false;

  //If Row is divided among then local slicedmuls has to be less than N 
  if (gpusInK > 1 and LocalN >= problem.n) correct = false;

  executeGeKMM(problem, nullptr, Matrix(),
    [](const KMMProblem kmm) {return 1;},
    [&correct, gpusInK](const KMMProblem kmm, int, void* t1, Matrix result) {
      correct = correct && (kmm.l() % gpusInK == 0);
      return cudaSuccess;
    });
  return correct;
}

//TODO: Change to backwardGeKMM
cudaError_t executeGeKMM(KMMProblem problem, void* temps[2],
                         Matrix result,
                         std::function<uint (const KMMProblem)> next,
                         std::function<cudaError_t (const KMMProblem, int rstart, void*[2], Matrix)> func) {
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
    if (temps != nullptr)
      problem.swap(temps[0], temps[1]);
  }

  return cudaSuccess;
}

//TODO: Change to forwardGeKMM
cudaError_t reverseExecuteGeKMM(KMMProblem problem, void* temps[2],
                                Matrix result,
                                std::function<uint (const KMMProblem)> next,
                                std::function<cudaError_t (const KMMProblem, int start, void*[2], Matrix)> func) {
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