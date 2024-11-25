#include "kmmalgo.h"

std::string strOfFastKronMMType(FastKronMMType mmtype) {
  if (mmtype == FastKronMMType::KMM) return "kmm";
  if (mmtype == FastKronMMType::MKM) return "mkm";
  return "";
}

std::size_t std::hash<KMMProblem>::operator()(const KMMProblem& problem) const {
  std::size_t h = hash<uint>()(problem.mmtype()) ^ hash<uint>()(problem.m()) ^
                  hash<uint>()(problem.k()) ^ hash<uint>()(problem.n());
  for (uint32_t i = 0; i < problem.n(); i++) {
    h = h ^ problem.f(i).hash();
  }
  return h;
}

std::size_t std::hash<KMMProblemStridedBatched>::operator()(const KMMProblemStridedBatched& problem) const {
  std::size_t h = std::hash<KMMProblem>()(problem.batchProblem(0)) ^ problem.batchCount();
  return h;
}

//TODO: Refactor below two functions for multi gpu kmm

bool checkDistributedKronSizes(const uint32_t NumKronMats, 
                               const uint32_t /*M*/,
                               const uint32_t /*N*/,
                               const uint32_t K, 
                               const uint32_t KronMatCols[],
                               const uint32_t KronMatRows[],
                               const uint32_t LocalKrons,
                               const uint32_t gpusInK) {
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

bool checkDistributedKronSizes(const KMMProblem problem, 
                               const uint LocalN,
                               const uint gpusInK) {
  //Cannot do more than N local slicedmuls
  if (LocalN > problem.n()) return false;

  //If Row is divided among then local slicedmuls has to be less than N 
  if (gpusInK > 1 and LocalN >= problem.n()) return false;
  
  bool correct = true;
  KMMProblem::Intermediates emptyIntermediates;
  executeGeMM(problem, emptyIntermediates, 0,
    [](const KMMProblem /*kmm*/) {return 1;},
    [&correct, gpusInK](const KMMProblem kmm, int /*rstart*/, Matrix /*result*/) {
      correct = correct && (kmm.l() % gpusInK == 0);
      return fastKronSuccess;
    });

  return correct;
}

/*MKM Algorithm functions*/
template<typename KMMProblemType>
fastKronError executeGeMM(KMMProblemType problem, typename KMMProblemType::Intermediates tmps, uint32_t swaps,
                           std::function<uint (const KMMProblemType)> next,
                           std::function<fastKronError (const KMMProblemType, int rstart,
                                                        typename KMMProblemType::Matrix)> func) {
  int nextF = 1;

  typename KMMProblemType::Matrix firstIterOut;
  typename KMMProblemType::Matrix result = problem.y();

  bool keepIntermediates = tmps.len() == (problem.n() - 1);

  if (keepIntermediates) {
  } else {
    if (tmps.len() > 0) {
      if (tmps[1].data() == nullptr) {
        if (swaps % 2 == 1) {
          tmps[1] = tmps[0];
          tmps[0] = problem.y();
        } else {
          tmps[0] = tmps[0];
          tmps[1] = problem.y();
        }
      }
      firstIterOut = result.like(tmps[0].data());
    } else {
      firstIterOut = result;
    }
  }

  problem = problem.setFirstIterOutput(firstIterOut);

  fastKronError err = fastKronSuccess;
  for (int i = problem.n() - 1; i >= 0; i = i - nextF) {
    nextF = next(problem);
    nextF = std::min(nextF, i+1);
    fastKronOp opX = problem.opX();
    //First iteration write output with op N
    if ((uint32_t)i < problem.n() - 1) {
      opX = fastKronOp_N;
    }
    auto subProblem = problem.rsub(i, nextF);
    if (i < nextF) {
      subProblem = subProblem.updateY(result);
    }
    else if (keepIntermediates) {
      subProblem = subProblem.updateY(tmps[i-1]);
    }

    if (keepIntermediates && i < problem.n() - 1) {
      subProblem = subProblem.updateX(tmps[(i + nextF)-1]);
    }

    subProblem.initMMIter(i, i == problem.n() - 1, i < nextF);
    subProblem.setOpX(opX);
    err = func(subProblem, i, result);
    if (err != fastKronSuccess) break;
    if (!keepIntermediates && tmps.len() > 0)
      problem.swap(tmps[0].data(), tmps[1].data());
  }

  return err;
}

fastKronError executeGeMM(const KMMProblem problem, KMMProblem::Intermediates temps,
                           uint32_t swaps,
                           std::function<uint (const KMMProblem)> next,
                           std::function<fastKronError (const KMMProblem, int,
                                         typename KMMProblem::Matrix)> func) {
  return executeGeMM<KMMProblem>(problem, temps, swaps, next, func);
}

fastKronError executeGeMM(const KMMProblemStridedBatched problem, KMMProblemStridedBatched::Intermediates temps,
                           uint32_t swaps,
                           std::function<uint (const KMMProblemStridedBatched)> next,
                           std::function<fastKronError (const KMMProblemStridedBatched, int, 
                                         typename KMMProblemStridedBatched::Matrix)> func) {
  return executeGeMM<KMMProblemStridedBatched>(problem, temps, swaps, next, func);
}

//TODO: Change to forwardGeKMM
template<typename KMMProblemType>
fastKronError reverseExecuteGeMM(KMMProblemType problem, void* tmps[2], typename KMMProblemType::Matrix result,
                                std::function<uint (const KMMProblemType)> next,
                                std::function<fastKronError (const KMMProblemType, int start, typename KMMProblemType::Matrix)> func) {
  uint32_t nextF = 1;
  fastKronError err = fastKronSuccess;
  for (uint32_t i = 0; i < problem.n(); i = i + nextF) {
    nextF = next(problem);
    if (i - (problem.n() - 1) < nextF) 
      problem = problem.updateY(result);
    err = func(problem.rsub(i, nextF), i, result);
    if (err != fastKronSuccess) break;
    if (tmps != nullptr)
      problem.swap(tmps[0], tmps[1]);
  }

  return err;
}

fastKronError reverseExecuteGeMM(const KMMProblem problem, void* temps[2],
                                typename KMMProblem::Matrix result,
                                std::function<uint (const KMMProblem)> next,
                                std::function<fastKronError (const KMMProblem, int, 
                                                             typename KMMProblem::Matrix)> func) {
  return reverseExecuteGeMM<KMMProblem>(problem, temps, result, next, func);
}

fastKronError reverseExecuteGeMM(const KMMProblemStridedBatched problem, void* temps[2],
                                typename KMMProblemStridedBatched::Matrix result,
                                std::function<uint (const KMMProblemStridedBatched)> next,
                                std::function<fastKronError (const KMMProblemStridedBatched, int, 
                                                             typename KMMProblemStridedBatched::Matrix)> func) {
  return reverseExecuteGeMM<KMMProblemStridedBatched>(problem, temps, result, next, func);
}