#pragma once

template<typename SubClassKernel>
void KernelDatabase::loadKernels(SubClassKernel* kernels, uint32_t numKernels) {
  //Load kernels into compiledKernels map
  for (uint i = 0; i < numKernels; i++) {
    SubClassKernel& info = kernels[i];
    DbKey key {info.getMaxFactor(), info.getOpX(), info.getOpF(), KernelBatchType::Normal};
    auto iter = compiledKernels.find(key);
    if (iter == compiledKernels.end()) {
      compiledKernels.emplace(std::make_pair(key, std::vector<KMMKernel*>()));
    }
    compiledKernels.at(key).push_back(&info);
  }

  if (false && Logger(LogLevel::Debug).valid()) {
    //Print loaded kernels
    uint numKernelsLoaded = 0;
    Logger(LogLevel::Debug) << "Loading compiled kernels" << std::endl;
    for (auto iter : compiledKernels) {
      for (auto kernel : iter.second) {
        Logger(LogLevel::Debug) << kernel->str() << std::endl;
      }
      numKernelsLoaded += iter.second.size();
    }
    Logger(LogLevel::Debug) << "Number of kernels loaded: " << numKernelsLoaded << std::endl;
  }
}