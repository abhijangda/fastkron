#include "kmmalgo.h"

std::size_t std::hash<KMMProblem>::operator()(const KMMProblem& problem) const {
  std::size_t h = hash<uint>()(problem.m()) ^ hash<uint>()(problem.k()) ^ hash<uint>()(problem.n());
  for (uint32_t i = 0; i < problem.n(); i++) {
    h = h ^ problem.f(i).hash();
  }
  return h;
}

bool checkDistributedKronSizes(const uint32_t NumKronMats, 
                               const uint32_t /*M*/, const uint32_t /*N*/, const uint32_t K, 
                               const uint32_t KronMatCols[], const uint32_t KronMatRows[],
                               const uint32_t LocalKrons, const uint32_t gpusInK) {
  uint prevTempN = K;
  
  if (prevTempN % gpusInK != 0) return false;
    
  for (uint32_t i = 0; i < NumKronMats; i += LocalKrons) {
    const uint kronMat = NumKronMats - i - 1;
    uint currTempN = prevTempN;
    for (uint32_t k = 0; k < std::min(LocalKrons, NumKronMats - i); k++) {
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
    [](const KMMProblem /*kmm*/) {return 1;},
    [&correct, gpusInK](const KMMProblem kmm, int /*rstart*/, void* /*t1*/, Matrix /*result*/) {
      correct = correct && (kmm.l() % gpusInK == 0);
      return fastKronSuccess;
    });

  return correct;
}

//TODO: Change to backwardGeKMM
fastKronError executeGeKMM(KMMProblem problem, void* tmps[2], uint32_t swaps,
                         std::function<uint (const KMMProblem)> next,
                         std::function<fastKronError (const KMMProblem, int rstart, void*[2], Matrix)> func) {
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
  problem = KMMProblem(problem.type(), problem.x(), problem.opX(), problem.n(), problem.fs(),
                       problem.opFs(), Matrix(problem.m(), problem.l(), firstIterOut));
  fastKronError err;
  for (int i = problem.n() - 1; i >= 0; i = i - nextF) {
    nextF = next(problem);
    nextF = std::min(nextF, i+1);
    fastKronOp opX = problem.opX();
    //First iteration write output with op N
    if ((uint32_t)i < problem.n() - 1) {
      opX = fastKronOp_N;
    }
    if (i < nextF) problem = KMMProblem(problem.type(), problem.x(), opX, problem.n(), 
                                        problem.fs(), problem.opFs(), result);
    auto subProblem = problem.rsub(i, nextF);
    subProblem.setOpX(opX);
    err = func(subProblem, i, tmps, result);
    if (err != fastKronSuccess) break;
    if (tmps != nullptr)
      problem.swap(tmps[0], tmps[1]);
  }

  return fastKronSuccess;
}

//TODO: Change to forwardGeKMM
fastKronError reverseExecuteGeKMM(KMMProblem problem, void* tmps[2], Matrix result,
                                std::function<uint (const KMMProblem)> next,
                                std::function<fastKronError (const KMMProblem, int start, void*[2], Matrix)> func) {
  uint32_t nextF = 1;
  fastKronError err;
  for (uint32_t i = 0; i < problem.n(); i = i + nextF) {
    nextF = next(problem);
    if (i - (problem.n() - 1) < nextF) 
      problem = KMMProblem(problem.type(), problem.x(), problem.opX(), problem.n(), 
                           problem.fs(), problem.opFs(), result);
    err = func(problem.rsub(i, nextF), i, tmps, result);
    if (err != fastKronSuccess) break;
    if (tmps != nullptr)
      problem.swap(tmps[0], tmps[1]);
  }

  return fastKronSuccess;
}