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
  KMMProblem::Matrices emptyIntermediates;
  executeGeMM(problem, emptyIntermediates,
    [](const KMMProblem /*kmm*/) {return 1;},
    [&correct, gpusInK](const KMMProblem kmm, int /*rstart*/, KMMProblem::Matrices /*results*/) {
      correct = correct && (kmm.l() % gpusInK == 0);
      return fastKronSuccess;
    });

  return correct;
}

fastKronError getIntermediates(bool keepIntermediates, const KMMProblem problem,
                               void* tmps[], uint32_t length,
                               std::function<uint (const KMMProblem)> next,
                               typename KMMProblem::Matrices& intermediates) {
  if (problem.n() == 1) {
    intermediates.push_back(problem.y());
    intermediates.push_back(problem.x());
    return fastKronSuccess;
  }

  if (tmps && not keepIntermediates) {
    if (tmps[1] == nullptr) {
      if (length % 2 == 1) {
        tmps[1] = tmps[0];
        tmps[0] = problem.y().data();
      } else {
        tmps[0] = tmps[0];
        tmps[1] = problem.y().data();
      }
    }
  }

  int t = 0;

  for (uint32_t i = 0; i < problem.n(); i++)
    intermediates.push_back(KMMProblem::Matrix());

  auto err = executeGeMM(problem, KMMProblem::Matrices({}), next, 
                        [&](const KMMProblem subProb, int32_t rstart, KMMProblem::Matrices) {
                          void* ptr;
                          if (keepIntermediates) {
                            ptr = tmps != nullptr ? tmps[rstart - 1] : nullptr;
                          } else {
                            ptr = tmps != nullptr ? tmps[t] : nullptr;
                            t = t ^ 1;
                          }
                          intermediates[rstart-subProb.n()+1] = subProb.y().like(ptr);
                          return fastKronSuccess;
                        });

  intermediates.push_back(problem.x());
  
  return err;
}

fastKronError getIntermediates(bool keepIntermediates, const KMMProblemStridedBatched problem,
                               void* tmps[], uint64_t* strideIntermediates, uint32_t length,
                               std::function<uint (const KMMProblemStridedBatched)> next,
                               typename KMMProblemStridedBatched::Matrices& intermediates) {
  if (problem.n() == 1) {
    intermediates.push_back(problem.y());
    intermediates.push_back(problem.x());
    return fastKronSuccess;
  }

  if (tmps && not keepIntermediates) {
    if (tmps[1] == nullptr) {
      if (length % 2 == 1) {
        tmps[1] = tmps[0];
        tmps[0] = problem.y().data();
      } else {
        tmps[0] = tmps[0];
        tmps[1] = problem.y().data();
      }
    }
  }

  int t = 0;

  for (uint32_t i = 0; i < problem.n(); i++)
    intermediates.push_back(typename KMMProblemStridedBatched::Matrix());

  auto err = executeGeMM(problem, KMMProblemStridedBatched::Matrices({}), next, 
                        [&](const KMMProblemStridedBatched subProb, int32_t rstart, KMMProblemStridedBatched::Matrices) {
                          void* ptr;
                          if (keepIntermediates) {
                            ptr = tmps != nullptr ? tmps[rstart - 1] : nullptr;
                          } else {
                            ptr = tmps != nullptr ? tmps[t] : nullptr;
                            t = t ^ 1;
                          }
                          uint64_t stride;
                          if (strideIntermediates) {
                            stride = strideIntermediates[rstart - 1];
                          } else {
                            stride = subProb.y().numel();
                          }
                          intermediates[rstart-subProb.n()+1] = subProb.y().like(ptr).diffBatchStride(stride);
                          return fastKronSuccess;
                        });
  //Since next can return anyvalue, we do not know if rstart is for the last kernel call.
  //So, stride for result of GeMM in intermediates can be wrong.
  //Fortunately, there is no need to set correct stride for last intermediate, i.e., result of GeMM, because the 
  //executeGeMM always write to problem.y() not intermediates[0].
  intermediates.push_back(problem.x());
  
  return err;
}


template<typename KMMProblemType>
fastKronError executeGeMM(KMMProblemType problem, typename KMMProblemType::Matrices tmps,
                          std::function<uint (const KMMProblemType)> next,
                          std::function<fastKronError (const KMMProblemType, int rstart,
                                                       typename KMMProblemType::Matrices)> func) {
  int nextF = 1;

  typename KMMProblemType::Matrix result = problem.y();

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

    if (tmps.len() > 0) {
      if (i < nextF) {} else {
        subProblem = subProblem.updateY(tmps[i-nextF+1]);
      }
      subProblem = subProblem.updateX(tmps[i+1]);
    }

    typename KMMProblemType::Matrices results({});
    if (tmps.len() > 0)
      results = tmps.slice(i - nextF + 1, nextF);

    subProblem.setOpX(opX);
    err = func(subProblem, i, results);
    if (err != fastKronSuccess) break;
  }

  return err;
}

fastKronError executeGeMM(const KMMProblem problem, typename KMMProblem::Matrices temps,
                          std::function<uint (const KMMProblem)> next,
                          std::function<fastKronError (const KMMProblem, int,
                                         typename KMMProblem::Matrices)> func) {
  return executeGeMM<KMMProblem>(problem, temps, next, func);
}

fastKronError executeGeMM(const KMMProblemStridedBatched problem, typename KMMProblemStridedBatched::Matrices temps,
                          std::function<uint (const KMMProblemStridedBatched)> next,
                          std::function<fastKronError (const KMMProblemStridedBatched, int, 
                                        typename KMMProblemStridedBatched::Matrices)> func) {
  return executeGeMM<KMMProblemStridedBatched>(problem, temps, next, func);
}

//TODO: Change to forwardGeKMM
template<typename KMMProblemType>
fastKronError reverseExecuteGeMM(KMMProblemType problem, void* tmps[2], typename KMMProblemType::Matrix result,
                                std::function<uint (const KMMProblemType)> next, 
                                //TODO: make function args consistent with executeGeMM 
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