#pragma once

template<KernelBatchType::Ty KernelBatch, typename ElemT, 
         typename KernelParams, typename EpilogueParams>
struct GetBatchedData {
  CUDA_DEVICE_HOST
  uint getBatchCount(const KernelParams& params);
  
  CUDA_DEVICE_HOST
  Matrix getXBatch(const KernelParams& params, int batch);
  
  CUDA_DEVICE_HOST
  Matrix getYBatch(const KernelParams& params, int batch);
  
  CUDA_DEVICE_HOST
  Factor getFBatch(const KernelParams& params, int fidx, int batch);
  
  CUDA_DEVICE_HOST
  Matrix getZBatch(const EpilogueParams& params, const Matrix& Y, int batch);
};

template<typename ElemT, typename KernelParams, typename EpilogueParams>
struct GetBatchedData<KernelBatchType::Normal, ElemT, KernelParams, EpilogueParams> {
  uint getBatchCount(const KernelParams& /*params*/) {return 1;}
  
  CUDA_DEVICE_HOST
  Matrix getXBatch(const KernelParams& params, int /*batch*/) {
    return params.problem.x();
  }
  
  CUDA_DEVICE_HOST
  Matrix getYBatch(const KernelParams& params, int /*batch*/) {
    return params.problem.y();
  }

  CUDA_DEVICE_HOST
  Factor getFBatch(const KernelParams& params, int fidx, int /*batch*/) {
    return params.problem.f(fidx);
  }

  CUDA_DEVICE_HOST
  Matrix getZBatch(const EpilogueParams& params, const Matrix& Y, int /*batch*/) {
    return Matrix(Y.m(), Y.n(), (void*)params.template z<ElemT>());
  }
};

template<typename ElemT, typename KernelParams, typename EpilogueParams>
struct GetBatchedData<KernelBatchType::StridedBatched, ElemT, KernelParams, EpilogueParams> {
  uint getBatchCount(const KernelParams& params) {return params.problem.batchCount();}

  CUDA_DEVICE_HOST
  Matrix getXBatch(const KernelParams& params, int batch) {
    return params.problem.x().template batch<ElemT>(batch);
  }
  
  CUDA_DEVICE_HOST
  Matrix getYBatch(const KernelParams& params, int batch) {
    return params.problem.y().template batch<ElemT>(batch);
  }

  CUDA_DEVICE_HOST
  Factor getFBatch(const KernelParams& params, int fidx, int batch) {
    return params.problem.f(fidx).template batch<ElemT>(batch);
  }

  CUDA_DEVICE_HOST
  Matrix getZBatch(const EpilogueParams& params, const Matrix& /*Y*/, int batch) {
    return params.getZ().template batch<ElemT>(batch);
  }
};