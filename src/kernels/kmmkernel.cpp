#include "kernels/kmmkernel.h"

size_t KMMKernel::getMaxTotalTileSize() const {
  Matrix Xsh = Matrix(tileX.m(), (tileX.n()/f.p())*tileF.p());
  //TODO: make this tileF.size() + Xsh.size()
  return (tileF.numel() + Xsh.numel())*sizeOfFastKronType(elemType);
}

Matrix KMMKernel::getMaxTileY() const {
  return Matrix(tileX.m(), (tileX.n()/f.p()) * tileF.q());
}

Factor KMMKernel::getTileF(KMMProblem problem) const {
  Factor f_ = problem.f(0);
  return Factor(MIN(tileF.p(), f_.p()), MIN(tileF.q(), f_.q()));
}

Matrix KMMKernel::getTileX(KMMProblem problem) const {
  Factor f_ = problem.f(0);

  uint32_t kernelTileSlices = tileX.n()/f.p();
  uint32_t problemTileSlices = problem.x().n()/f_.p();

  uint32_t slices = 0;

  if (problemTileSlices >= kernelTileSlices) {
    slices = kernelTileSlices;
  } else {
    slices = MAX(1, MIN(tileX.n()/f_.p(), kernelTileSlices));
    slices = MIN(problemTileSlices, slices);
  }

  return Matrix(tileX.m(), slices * f_.p());
}

size_t KMMKernel::getTotalTileSize(KMMProblem problem) const {
  Matrix tileX_ = getTileX(problem);
  Factor f_ = problem.f(0);

  //Pad Xsh to TileP
  //Pad Fsh to TileP x TileQ
  Matrix Xsh = Matrix(tileX_.m(), 
                      (tileX_.n()/f_.p()) * tileF.p());
  return (tileF.numel() + Xsh.numel())*sizeOfFastKronType(elemType);
}

size_t KMMKernel::getNumThreads(KMMProblem problem) const {
  Matrix tileX_ = getTileX(problem);
  Factor tileF_ = getTileF(problem);

  return DIVUP(problem.k(), tileX_.n()) * 
          DIVUP(problem.f(0).q(), tileF_.q()) * 
          DIVUP(problem.m(), tileX_.m());
}

bool KMMKernel::isOptValid(KMMProblem problem, KernelOptimizations::Optimization opt) const {
  using Opts = KernelOptimizations::Optimization;
  switch (opt) {
    case Opts::None:
      return true;
    case Opts::XshSlicesSame:
      return getTileX(problem).n()/problem.f(0).p() == tileX.n()/f.p();
    case Opts::QMultipleOfTileQ:
      return problem.f(0).q() % tileF.q() == 0;
    case Opts::PMultipleOfTileP:
      return problem.f(0).p() % tileF.p() == 0;
    case Opts::KMultipleOfTileK:
      return problem.k() % getTileX(problem).n() == 0;
    case Opts::MMultipleOfTileM:
      return problem.m() % getTileX(problem).m() == 0;
    case Opts::QLeTileQ:
      return problem.f(0).q() <= f.q();
    case Opts::TileKSame:
      return getTileX(problem).n() == tileX.n();
    case Opts::FactorShapeSame:
      return f.p() == problem.f(0).p() && f.q() == problem.f(0).q();
    
    default:
      return false;
  }

  return false;
}

bool KMMKernel::canCompute(KMMProblem problem, const HardwareDetails*,
                           bool p2p, KernelBatchType::Ty probBatchType, 
                           bool exactFuse) {
  using Opts = KernelOptimizations::Optimization;

  bool ret = problem.mmtype() == mmType && problem.type() == elemType &&
              problem.opFs() == opF && problem.opX() == opX && 
              P2PStore == p2p && ((exactFuse && problem.n() == fusedFacs) || 
                                  (!exactFuse && problem.n() >= fusedFacs)) &&
              //tileX.n()/MaxF.p() > problem.f(0).p() && //Kernel's TileX is greater than P
              kernelBatchType == probBatchType;

  if (!ret) return false;

  bool followsAllOpts = true;
  uint lg = 0;
  for (Opts opt = Opts(lg); opt < Opts::NumOptimizations; opt = Opts(1 << lg), ++lg) {
    if ((KernelOptimizations::getOptimizations(optLevel) & opt) == opt) {
      followsAllOpts = followsAllOpts && isOptValid(problem, opt);
  }}

  return followsAllOpts;
}

static std::ostream& operator<<(std::ostream& os, const KernelBatchType::Ty& b) {
  switch (b) {
    case KernelBatchType::Normal:
      os << "cont";
      break;
    case KernelBatchType::StridedBatched:
      os << "strided";
      break;
    case KernelBatchType::Batch:
      os << "batched";
      break;
  }
  return os;
}

std::string KMMKernel::str() const {
  std::stringstream info;
  info << strOfFastKronMMType(mmType)
       << "_" << strOfFastKronType(elemType) 
       << "_" << f << "_" << tileF <<"_" << fusedFacs
       << "_" << tileX << "_" << regM << "x" << regK << "x" << regQ 
       << "_" << opX << opF << "_" << kernelBatchType << "_" << P2PStore << "_" << optLevel;
  return info.str();
}