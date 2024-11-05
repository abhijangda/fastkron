#pragma once

/**
 * GetBatchedData - A template struct to obtain common parameters for 
 *                  single or strided batched problems. The struct is specialized 
 *                  for both batch types.
 */
template<KernelBatchType::Ty KernelBatch, typename ElemT, 
         typename KernelParams, typename EpilogueParams>
struct GetBatchedData {
  /**
   * getBatchCount - Get number of batches in the problem.
   */
  CUDA_DEVICE_HOST
  uint getBatchCount(const KernelParams& params) const;
  
  /**
   * getXBatch - Get input X for a batch.
   */
  CUDA_DEVICE_HOST
  Matrix getXBatch(const KernelParams& params, int batch) const;
  
  /**
   * getYBatch - Get output Y for a batch.
   */
  CUDA_DEVICE_HOST
  Matrix getYBatch(const KernelParams& params, int batch) const;
  
  /**
   * getFBatch - Get factor at an index for a batch.
   */
  CUDA_DEVICE_HOST
  Factor getFBatch(const KernelParams& params, int fidx, int batch) const;
  
  /**
   * getZBatch - Get input Z for a batch.
   */
  CUDA_DEVICE_HOST
  Matrix getZBatch(const EpilogueParams& params, const Matrix& Y, int batch) const;
};

/**
 * GetBatchedData specliazed for single batched problem.
 */
template<typename ElemT, typename KernelParams, typename EpilogueParams>
struct GetBatchedData<KernelBatchType::Normal, ElemT, KernelParams, EpilogueParams> {
  uint getBatchCount(const KernelParams& /*params*/) const {return 1;}
  
  CUDA_DEVICE_HOST
  Matrix getXBatch(const KernelParams& params, int /*batch*/) const {
    return params.problem.x();
  }
  
  CUDA_DEVICE_HOST
  Matrix getYBatch(const KernelParams& params, int /*batch*/) const {
    return params.problem.y();
  }

  CUDA_DEVICE_HOST
  Factor getFBatch(const KernelParams& params, int fidx, int /*batch*/) const {
    return params.problem.f(fidx);
  }

  CUDA_DEVICE_HOST
  Matrix getZBatch(const EpilogueParams& params, const Matrix& Y, int /*batch*/) const {
    return Matrix(Y.m(), Y.n(), (void*)params.template z<ElemT>());
  }
};

/**
 * GetBatchedData specliazed for strided batched problem.
 */
template<typename ElemT, typename KernelParams, typename EpilogueParams>
struct GetBatchedData<KernelBatchType::StridedBatched, ElemT, KernelParams, EpilogueParams> {
  uint getBatchCount(const KernelParams& params) const {return params.problem.batchCount();}

  CUDA_DEVICE_HOST
  Matrix getXBatch(const KernelParams& params, int batch) const {
    return params.problem.x().template batch<ElemT>(batch);
  }
  
  CUDA_DEVICE_HOST
  Matrix getYBatch(const KernelParams& params, int batch) const {
    return params.problem.y().template batch<ElemT>(batch);
  }

  CUDA_DEVICE_HOST
  Factor getFBatch(const KernelParams& params, int fidx, int batch) const {
    return params.problem.f(fidx).template batch<ElemT>(batch);
  }

  CUDA_DEVICE_HOST
  Matrix getZBatch(const EpilogueParams& params, const Matrix& /*Y*/, int batch) const {
    return params.getZ().template batch<ElemT>(batch);
  }
};