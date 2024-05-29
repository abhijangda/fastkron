#include <immintrin.h>
#include <cstring>
#include <cstdlib>
#include <omp.h>
#include "kernels/cuda/fixed-shape-tensor.cuh"

#pragma once

template<typename ElemT, typename VecT>
static inline void vectorLoad(const ElemT* ptr, VecT& data){assert(false);}

template<typename ElemT, typename VecT>
static inline void vectorStore(ElemT* ptr, const VecT& vec) {}

template<typename VecT>
static inline void vectorZero(VecT& data) {assert(false);}

template<typename VecT>
inline void vectorFMA(const VecT& a, const VecT& b, VecT& c) {assert(false);}

template<typename ElemT, typename VecT>
static inline void vectorGather(const ElemT* base, const uint32_t* gatherIdxs, VecT& data);

template<typename ElemT, typename VecT>
static inline void vectorBroadcast(const ElemT* ptr, VecT& data) {}

////////////////////////Single///////////////////////////////
struct SISDFloatWrapper {
  using VecT = float;
  float data;
};

struct SISDDoubleWrapper {
  using VecT = double;
  double data;
};

template<>
inline void vectorLoad(const float* ptr, float& data) {
  data = *ptr;
}

template<>
inline void vectorLoad(const double* ptr, double& data) {
  data = *ptr;
}

template<>
inline void vectorStore(float* ptr, const float& vec) {
  *ptr = vec;
}

template<>
inline void vectorStore(double* ptr, const double& vec) {
  *ptr = vec;
}

template<>
inline void vectorZero(float& data) {
  data = 0;
}

template<>
inline void vectorZero(double& data) {
  data = 0;
}

template<>
inline void vectorFMA(const float& a, const float& b, float& c) {
  c = a*b + c;
}

template<>
inline void vectorFMA(const double& a, const double& b, double& c) {
  c = a*b + c;
}

template<>
inline void vectorBroadcast(const float* ptr, float& data) {
  data = *ptr;
}

template<>
inline void vectorBroadcast(const double* ptr, double& data) {
  data = *ptr;
}

template<>
inline void vectorGather(const float* base, const uint32_t* gatherIdxs, float& data) {
  data = base[gatherIdxs[0]]; 
}

template<>
inline void vectorGather(const double* base, const uint32_t* gatherIdxs, double& data) {
  data = base[gatherIdxs[0]];
}
////////////////////////////////////////////////////////////

////////////////////////AVX-256/////////////////////////////
struct AVXFloatWrapper {
  using VecT = __m256;
  __m256 data;
};

struct AVXDoubleWrapper {
  using VecT = __m256d;
  __m256d data;
};

template<>
inline void vectorLoad(const float* ptr, __m256& data) {
  data = _mm256_loadu_ps(ptr);
}

template<>
inline void vectorLoad(const double* ptr, __m256d& data) {
  data = _mm256_loadu_pd(ptr);
}

template<>
inline void vectorStore(float* ptr, const __m256& vec) {
  _mm256_storeu_ps(ptr, vec);
}

template<>
inline void vectorStore(double* ptr, const __m256d& vec) {
  _mm256_storeu_pd(ptr, vec);
}

template<>
inline void vectorZero(__m256& data) {
  data = _mm256_setzero_ps();
}

template<>
inline void vectorZero(__m256d& data) {
  data = _mm256_setzero_pd();
}

template<>
inline void vectorFMA(const __m256& a, const __m256& b, __m256& c) {
  c = _mm256_fmadd_ps(a, b, c);
}

template<>
inline void vectorFMA(const __m256d& a, const __m256d& b, __m256d& c) {
  c = _mm256_fmadd_pd(a, b, c);
}

template<>
inline void vectorBroadcast(const float* ptr, __m256& data) {
  data = _mm256_broadcast_ss(ptr);
}

template<>
inline void vectorBroadcast(const double* ptr, __m256d& data) {
  data = _mm256_broadcast_sd(ptr);
}

template<>
inline void vectorGather(const float* base, const uint32_t* gatherIdxs, __m256& data) {
  __m256i vidx = _mm256_loadu_si256((__m256i*)gatherIdxs);
  data = _mm256_i32gather_ps(base, vidx, sizeof(float)); 
}

template<>
inline void vectorGather(const double* base, const uint32_t* gatherIdxs, __m256d& data) {
  __m128i vidx = _mm_loadu_si128((__m128i*)gatherIdxs);
  data = _mm256_i32gather_pd(base, vidx, sizeof(double));
}
//////////////////////////////////////////////////////////

////////////////////////AVX-512///////////////////////////
struct AVX512FloatWrapper {
  using VecT = __m512;
  __m512 data;
};

struct AVX512DoubleWrapper {
  using VecT = __m512d;
  __m512d data;
};

template<>
inline void vectorLoad(const float* ptr, __m512& data) {
  data = _mm512_loadu_ps(ptr);
}

template<>
inline void vectorLoad(const double* ptr, __m512d& data) {
  data = _mm512_loadu_pd(ptr);
}

template<>
inline void vectorStore(float* ptr, const __m512& vec) {
  _mm512_storeu_ps(ptr, vec);
}

template<>
inline void vectorStore(double* ptr, const __m512d& vec) {
  _mm512_storeu_pd(ptr, vec);
}

template<>
inline void vectorZero(__m512& data) {
  data = _mm512_setzero_ps();
}

template<>
inline void vectorZero(__m512d& data) {
  data = _mm512_setzero_pd();
}

template<>
inline void vectorFMA(const __m512& a, const __m512& b, __m512& c) {
  c = _mm512_fmadd_ps(a, b, c);
}

template<>
inline void vectorFMA(const __m512d& a, const __m512d& b, __m512d& c) {
  c = _mm512_fmadd_pd(a, b, c);
}

template<>
inline void vectorBroadcast(const float* ptr, __m512& data) {
  data = _mm512_set1_ps(*ptr);
}

template<>
inline void vectorBroadcast(const double* ptr, __m512d& data) {
  data = _mm512_set1_pd(*ptr);
}

template<>
inline void vectorGather(const float* base, const uint32_t* gatherIdxs, __m512& data) {
  __m512i vidx = _mm512_loadu_si512((__m512i*)gatherIdxs);
  data = _mm512_i32gather_ps(vidx, base, sizeof(float)); 
}

template<>
inline void vectorGather(const double* base, const uint32_t* gatherIdxs, __m512d& data) {
  __m256i vidx = _mm256_loadu_si256((__m256i*)gatherIdxs);
  data = _mm512_i32gather_pd(vidx, base, sizeof(double));
}
//////////////////////////////////////////////////////////

template<typename ElemT, typename VecT>
class X86Vector {
private:
  using VecWrapper = VecT;
  using UnderlyingVecT = typename VecT::VecT; 
  VecT vec;

public:
  static const uint32_t VectorLen = sizeof(VecT)/sizeof(ElemT);

  X86Vector(typename VecT::VecT v): vec{v} {}
  X86Vector() {}

  void load(const ElemT* ptr) {
    vectorLoad<ElemT, UnderlyingVecT>(ptr, vec.data);
  }

  void store(ElemT* ptr) const {
    vectorStore<ElemT, UnderlyingVecT>(ptr, vec.data);
  }
  
  void store(ElemT* ptr, uint32_t sz) const {
    if (sz == VectorLen)
      store(ptr);
    else if (VectorLen > 1) {
      ElemT elems[VectorLen];
      vectorStore<ElemT, UnderlyingVecT>(elems, vec.data);
      memcpy(ptr, elems, sz * sizeof(ElemT));
    }
  }

  void zero() {
    vectorZero<UnderlyingVecT>(vec.data);
  }

  void broadcast(const ElemT* ptr) {
    vectorBroadcast<ElemT, UnderlyingVecT>(ptr, vec.data);
  }

  void fmadd(const X86Vector<ElemT, VecT>& a, const X86Vector<ElemT, VecT>& b) {
    vectorFMA<UnderlyingVecT>(a.vec.data, b.vec.data, vec.data);
  }
  
  void gather(const ElemT* base, const uint32_t* gatherIdxs) {
    vectorGather(base, gatherIdxs, vec.data);
  }

  const typename VecT::VecT& data() {return vec.data;}
  void print() {
    ElemT elems[VectorLen];
    vectorStore<ElemT, UnderlyingVecT>(elems, vec.data);
    printf("%f\n", elems[0]);
  }
};

class SISDFloat : public X86Vector<float, SISDFloatWrapper> {
public:
  SISDFloat(SISDFloatWrapper::VecT v) : X86Vector<float, SISDFloatWrapper>(v) {}
  SISDFloat() {}
  static void transpose(SISDFloat rows[]) {
    rows[0] = rows[0];
  }
};

class SISDDouble : public X86Vector<double, SISDDoubleWrapper> {
public:
  SISDDouble(SISDDoubleWrapper::VecT v) : X86Vector<double, SISDDoubleWrapper>(v) {}
  SISDDouble() {}
  static void transpose(SISDDouble rows[]) {
    rows[0] = rows[0];
  }
};

class AVXFloat : public X86Vector<float, AVXFloatWrapper> {
public:
  AVXFloat(AVXFloatWrapper::VecT v) : X86Vector<float, AVXFloatWrapper>(v) {}
  AVXFloat() {}
  AVXFloat(float zero) {this->zero();}

  static void transpose(AVXFloat rows[]) {
    // https://stackoverflow.com/questions/25622745/transpose-an-8x8-float-using-avx-avx2
    __m256 __t0, __t1, __t2, __t3, __t4, __t5, __t6, __t7;
    __m256 __tt0, __tt1, __tt2, __tt3, __tt4, __tt5, __tt6, __tt7;
    __t0 = _mm256_unpacklo_ps(rows[0].data(), rows[1].data());
    __t1 = _mm256_unpackhi_ps(rows[0].data(), rows[1].data());
    __t2 = _mm256_unpacklo_ps(rows[2].data(), rows[3].data());
    __t3 = _mm256_unpackhi_ps(rows[2].data(), rows[3].data());
    __t4 = _mm256_unpacklo_ps(rows[4].data(), rows[5].data());
    __t5 = _mm256_unpackhi_ps(rows[4].data(), rows[5].data());
    __t6 = _mm256_unpacklo_ps(rows[6].data(), rows[7].data());
    __t7 = _mm256_unpackhi_ps(rows[6].data(), rows[7].data());
    __tt0 = _mm256_shuffle_ps(__t0,__t2,_MM_SHUFFLE(1,0,1,0));
    __tt1 = _mm256_shuffle_ps(__t0,__t2,_MM_SHUFFLE(3,2,3,2));
    __tt2 = _mm256_shuffle_ps(__t1,__t3,_MM_SHUFFLE(1,0,1,0));
    __tt3 = _mm256_shuffle_ps(__t1,__t3,_MM_SHUFFLE(3,2,3,2));
    __tt4 = _mm256_shuffle_ps(__t4,__t6,_MM_SHUFFLE(1,0,1,0));
    __tt5 = _mm256_shuffle_ps(__t4,__t6,_MM_SHUFFLE(3,2,3,2));
    __tt6 = _mm256_shuffle_ps(__t5,__t7,_MM_SHUFFLE(1,0,1,0));
    __tt7 = _mm256_shuffle_ps(__t5,__t7,_MM_SHUFFLE(3,2,3,2));
    rows[0] = AVXFloat(_mm256_permute2f128_ps(__tt0, __tt4, 0x20));
    rows[1] = AVXFloat(_mm256_permute2f128_ps(__tt1, __tt5, 0x20));
    rows[2] = AVXFloat(_mm256_permute2f128_ps(__tt2, __tt6, 0x20));
    rows[3] = AVXFloat(_mm256_permute2f128_ps(__tt3, __tt7, 0x20));
    rows[4] = AVXFloat(_mm256_permute2f128_ps(__tt0, __tt4, 0x31));
    rows[5] = AVXFloat(_mm256_permute2f128_ps(__tt1, __tt5, 0x31));
    rows[6] = AVXFloat(_mm256_permute2f128_ps(__tt2, __tt6, 0x31));
    rows[7] = AVXFloat(_mm256_permute2f128_ps(__tt3, __tt7, 0x31));
  }
};

class AVXDouble : public X86Vector<double, AVXDoubleWrapper> {
public:
  AVXDouble(AVXDoubleWrapper::VecT v) : X86Vector<double, AVXDoubleWrapper>(v) {}
  AVXDouble() {}
  AVXDouble(double zero) {this->zero();}

  static void transpose(AVXDouble rows[]) {
    // https://stackoverflow.com/questions/25622745/transpose-an-8x8-float-using-avx-avx2
    __m256d tmp3, tmp2, tmp1, tmp0;
    tmp0 = _mm256_shuffle_pd(rows[0].data(), rows[1].data(), 0x0);
    tmp2 = _mm256_shuffle_pd(rows[0].data(), rows[1].data(), 0xF);
    tmp1 = _mm256_shuffle_pd(rows[2].data(), rows[3].data(), 0x0);
    tmp3 = _mm256_shuffle_pd(rows[2].data(), rows[3].data(), 0xF);

    rows[0] = AVXDouble(_mm256_permute2f128_pd(tmp0, tmp1, 0x20));
    rows[1] = AVXDouble(_mm256_permute2f128_pd(tmp2, tmp3, 0x20));
    rows[2] = AVXDouble(_mm256_permute2f128_pd(tmp0, tmp1, 0x31));
    rows[3] = AVXDouble(_mm256_permute2f128_pd(tmp2, tmp3, 0x31));
  }
};

class AVX512Float : public X86Vector<float, AVX512FloatWrapper> {
public:
  AVX512Float(AVX512FloatWrapper::VecT v) : X86Vector<float, AVX512FloatWrapper>(v) {}
  AVX512Float() {}
  AVX512Float(float zero) {this->zero();}

  static void transpose(AVX512Float rows[]) {
    // https://gist.github.com/nihui/37d98b705a6a28911d77c502282b4748
    __m512 _tmp0 = _mm512_unpacklo_ps(rows[0].data(), rows[1].data());
    __m512 _tmp1 = _mm512_unpackhi_ps(rows[0].data(), rows[1].data());
    __m512 _tmp2 = _mm512_unpacklo_ps(rows[2].data(), rows[3].data());
    __m512 _tmp3 = _mm512_unpackhi_ps(rows[2].data(), rows[3].data());
    __m512 _tmp4 = _mm512_unpacklo_ps(rows[4].data(), rows[5].data());
    __m512 _tmp5 = _mm512_unpackhi_ps(rows[4].data(), rows[5].data());
    __m512 _tmp6 = _mm512_unpacklo_ps(rows[6].data(), rows[7].data());
    __m512 _tmp7 = _mm512_unpackhi_ps(rows[6].data(), rows[7].data());
    __m512 _tmp8 = _mm512_unpacklo_ps(rows[8].data(), rows[9].data());
    __m512 _tmp9 = _mm512_unpackhi_ps(rows[8].data(), rows[9].data());
    __m512 _tmpa = _mm512_unpacklo_ps(rows[10].data(), rows[11].data());
    __m512 _tmpb = _mm512_unpackhi_ps(rows[10].data(), rows[11].data());
    __m512 _tmpc = _mm512_unpacklo_ps(rows[12].data(), rows[13].data());
    __m512 _tmpd = _mm512_unpackhi_ps(rows[12].data(), rows[13].data());
    __m512 _tmpe = _mm512_unpacklo_ps(rows[14].data(), rows[15].data());
    __m512 _tmpf = _mm512_unpackhi_ps(rows[14].data(), rows[15].data());

    __m512 _tmpg = _mm512_shuffle_ps(_tmp0, _tmp2, _MM_SHUFFLE(1, 0, 1, 0));
    __m512 _tmph = _mm512_shuffle_ps(_tmp0, _tmp2, _MM_SHUFFLE(3, 2, 3, 2));
    __m512 _tmpi = _mm512_shuffle_ps(_tmp1, _tmp3, _MM_SHUFFLE(1, 0, 1, 0));
    __m512 _tmpj = _mm512_shuffle_ps(_tmp1, _tmp3, _MM_SHUFFLE(3, 2, 3, 2));
    __m512 _tmpk = _mm512_shuffle_ps(_tmp4, _tmp6, _MM_SHUFFLE(1, 0, 1, 0));
    __m512 _tmpl = _mm512_shuffle_ps(_tmp4, _tmp6, _MM_SHUFFLE(3, 2, 3, 2));
    __m512 _tmpm = _mm512_shuffle_ps(_tmp5, _tmp7, _MM_SHUFFLE(1, 0, 1, 0));
    __m512 _tmpn = _mm512_shuffle_ps(_tmp5, _tmp7, _MM_SHUFFLE(3, 2, 3, 2));
    __m512 _tmpo = _mm512_shuffle_ps(_tmp8, _tmpa, _MM_SHUFFLE(1, 0, 1, 0));
    __m512 _tmpp = _mm512_shuffle_ps(_tmp8, _tmpa, _MM_SHUFFLE(3, 2, 3, 2));
    __m512 _tmpq = _mm512_shuffle_ps(_tmp9, _tmpb, _MM_SHUFFLE(1, 0, 1, 0));
    __m512 _tmpr = _mm512_shuffle_ps(_tmp9, _tmpb, _MM_SHUFFLE(3, 2, 3, 2));
    __m512 _tmps = _mm512_shuffle_ps(_tmpc, _tmpe, _MM_SHUFFLE(1, 0, 1, 0));
    __m512 _tmpt = _mm512_shuffle_ps(_tmpc, _tmpe, _MM_SHUFFLE(3, 2, 3, 2));
    __m512 _tmpu = _mm512_shuffle_ps(_tmpd, _tmpf, _MM_SHUFFLE(1, 0, 1, 0));
    __m512 _tmpv = _mm512_shuffle_ps(_tmpd, _tmpf, _MM_SHUFFLE(3, 2, 3, 2));

    _tmp0 = _mm512_shuffle_f32x4(_tmpg, _tmpk, _MM_SHUFFLE(2, 0, 2, 0));
    _tmp1 = _mm512_shuffle_f32x4(_tmpo, _tmps, _MM_SHUFFLE(2, 0, 2, 0));
    _tmp2 = _mm512_shuffle_f32x4(_tmph, _tmpl, _MM_SHUFFLE(2, 0, 2, 0));
    _tmp3 = _mm512_shuffle_f32x4(_tmpp, _tmpt, _MM_SHUFFLE(2, 0, 2, 0));
    _tmp4 = _mm512_shuffle_f32x4(_tmpi, _tmpm, _MM_SHUFFLE(2, 0, 2, 0));
    _tmp5 = _mm512_shuffle_f32x4(_tmpq, _tmpu, _MM_SHUFFLE(2, 0, 2, 0));
    _tmp6 = _mm512_shuffle_f32x4(_tmpj, _tmpn, _MM_SHUFFLE(2, 0, 2, 0));
    _tmp7 = _mm512_shuffle_f32x4(_tmpr, _tmpv, _MM_SHUFFLE(2, 0, 2, 0));
    _tmp8 = _mm512_shuffle_f32x4(_tmpg, _tmpk, _MM_SHUFFLE(3, 1, 3, 1));
    _tmp9 = _mm512_shuffle_f32x4(_tmpo, _tmps, _MM_SHUFFLE(3, 1, 3, 1));
    _tmpa = _mm512_shuffle_f32x4(_tmph, _tmpl, _MM_SHUFFLE(3, 1, 3, 1));
    _tmpb = _mm512_shuffle_f32x4(_tmpp, _tmpt, _MM_SHUFFLE(3, 1, 3, 1));
    _tmpc = _mm512_shuffle_f32x4(_tmpi, _tmpm, _MM_SHUFFLE(3, 1, 3, 1));
    _tmpd = _mm512_shuffle_f32x4(_tmpq, _tmpu, _MM_SHUFFLE(3, 1, 3, 1));
    _tmpe = _mm512_shuffle_f32x4(_tmpj, _tmpn, _MM_SHUFFLE(3, 1, 3, 1));
    _tmpf = _mm512_shuffle_f32x4(_tmpr, _tmpv, _MM_SHUFFLE(3, 1, 3, 1));

    rows[0] = AVX512Float(_mm512_shuffle_f32x4(_tmp0, _tmp1, _MM_SHUFFLE(2, 0, 2, 0)));
    rows[1] = AVX512Float(_mm512_shuffle_f32x4(_tmp2, _tmp3, _MM_SHUFFLE(2, 0, 2, 0)));
    rows[2] = AVX512Float(_mm512_shuffle_f32x4(_tmp4, _tmp5, _MM_SHUFFLE(2, 0, 2, 0)));
    rows[3] = AVX512Float(_mm512_shuffle_f32x4(_tmp6, _tmp7, _MM_SHUFFLE(2, 0, 2, 0)));
    rows[4] = AVX512Float(_mm512_shuffle_f32x4(_tmp8, _tmp9, _MM_SHUFFLE(2, 0, 2, 0)));
    rows[5] = AVX512Float(_mm512_shuffle_f32x4(_tmpa, _tmpb, _MM_SHUFFLE(2, 0, 2, 0)));
    rows[6] = AVX512Float(_mm512_shuffle_f32x4(_tmpc, _tmpd, _MM_SHUFFLE(2, 0, 2, 0)));
    rows[7] = AVX512Float(_mm512_shuffle_f32x4(_tmpe, _tmpf, _MM_SHUFFLE(2, 0, 2, 0)));
    rows[8] = AVX512Float(_mm512_shuffle_f32x4(_tmp0, _tmp1, _MM_SHUFFLE(3, 1, 3, 1)));
    rows[9] = AVX512Float(_mm512_shuffle_f32x4(_tmp2, _tmp3, _MM_SHUFFLE(3, 1, 3, 1)));
    rows[10] = AVX512Float(_mm512_shuffle_f32x4(_tmp4, _tmp5, _MM_SHUFFLE(3, 1, 3, 1)));
    rows[11] = AVX512Float(_mm512_shuffle_f32x4(_tmp6, _tmp7, _MM_SHUFFLE(3, 1, 3, 1)));
    rows[12] = AVX512Float(_mm512_shuffle_f32x4(_tmp8, _tmp9, _MM_SHUFFLE(3, 1, 3, 1)));
    rows[13] = AVX512Float(_mm512_shuffle_f32x4(_tmpa, _tmpb, _MM_SHUFFLE(3, 1, 3, 1)));
    rows[14] = AVX512Float(_mm512_shuffle_f32x4(_tmpc, _tmpd, _MM_SHUFFLE(3, 1, 3, 1)));
    rows[15] = AVX512Float(_mm512_shuffle_f32x4(_tmpe, _tmpf, _MM_SHUFFLE(3, 1, 3, 1)));
  }
};

class AVX512Double : public X86Vector<double, AVX512DoubleWrapper> {
public:
  AVX512Double(AVX512DoubleWrapper::VecT v) : X86Vector<double, AVX512DoubleWrapper>(v) {}
  AVX512Double() {}
  AVX512Double(double zero) {}

  static void transpose(AVX512Double rows[]) {
    //https://github.com/romeric/Fastor/blob/master/Fastor/backend/transpose/transpose_kernels.h:_MM_TRANSPOSE8_PD
    __m512d __t0, __t1, __t2, __t3, __t4, __t5, __t6, __t7;
    __m512d __tt0, __tt1, __tt2, __tt3, __tt4, __tt5, __tt6, __tt7;

    constexpr int64_t idx1[8] = {0, 8 , 1 , 9 , 4 , 12, 5 , 13};
    constexpr int64_t idx2[8] = {2, 10, 3 , 11, 6 , 14, 7 , 15};
    constexpr int64_t idx3[8] = {0, 1 , 8 , 9 , 4 , 5 , 12, 13};
    constexpr int64_t idx4[8] = {2, 3 , 10, 11, 6 , 7 , 14, 15};
    constexpr int64_t idx5[8] = {4, 5 , 6 , 7 , 12, 13, 14, 15};

    __m512i vidx1 = _mm512_load_epi64(idx1);
    __m512i vidx2 = _mm512_load_epi64(idx2);
    __m512i vidx3 = _mm512_load_epi64(idx3);
    __m512i vidx4 = _mm512_load_epi64(idx4);
    __m512i vidx5 = _mm512_load_epi64(idx5);

    __t0 = _mm512_permutex2var_pd(rows[0].data(), vidx1, rows[1].data());
    __t1 = _mm512_permutex2var_pd(rows[0].data(), vidx2, rows[1].data());
    __t2 = _mm512_permutex2var_pd(rows[2].data(), vidx1, rows[3].data());
    __t3 = _mm512_permutex2var_pd(rows[2].data(), vidx2, rows[3].data());
    __t4 = _mm512_permutex2var_pd(rows[4].data(), vidx1, rows[5].data());
    __t5 = _mm512_permutex2var_pd(rows[4].data(), vidx2, rows[5].data());
    __t6 = _mm512_permutex2var_pd(rows[6].data(), vidx1, rows[7].data());
    __t7 = _mm512_permutex2var_pd(rows[6].data(), vidx2, rows[7].data());

    __tt0 = _mm512_permutex2var_pd(__t0, vidx3, __t2);
    __tt1 = _mm512_permutex2var_pd(__t0, vidx4, __t2);
    __tt2 = _mm512_permutex2var_pd(__t1, vidx3, __t3);
    __tt3 = _mm512_permutex2var_pd(__t1, vidx4, __t3);
    __tt4 = _mm512_permutex2var_pd(__t4, vidx3, __t6);
    __tt5 = _mm512_permutex2var_pd(__t4, vidx4, __t6);
    __tt6 = _mm512_permutex2var_pd(__t5, vidx3, __t7);
    __tt7 = _mm512_permutex2var_pd(__t5, vidx4, __t7);

    rows[0] = AVX512Double(_mm512_insertf64x4(__tt0,_mm512_castpd512_pd256(__tt4),0x1));
    rows[1] = AVX512Double(_mm512_insertf64x4(__tt1,_mm512_castpd512_pd256(__tt5),0x1));
    rows[2] = AVX512Double(_mm512_insertf64x4(__tt2,_mm512_castpd512_pd256(__tt6),0x1));
    rows[3] = AVX512Double(_mm512_insertf64x4(__tt3,_mm512_castpd512_pd256(__tt7),0x1));
    rows[4] = AVX512Double(_mm512_permutex2var_pd(__tt0, vidx5, __tt4));
    rows[5] = AVX512Double(_mm512_permutex2var_pd(__tt1, vidx5, __tt5));
    rows[6] = AVX512Double(_mm512_permutex2var_pd(__tt2, vidx5, __tt6));
    rows[7] = AVX512Double(_mm512_permutex2var_pd(__tt3, vidx5, __tt7));
  }
};

template<uint OptLevel, typename ElemT, fastKronOp OpF, typename DirectTileF>
void directCache(const Factor& F, DirectTileF& TileF, uint32_t tileP, uint32_t tileQ) {
  constexpr bool kQMultipleOfTileQ = KernelOptimizations::IsQMultipleOfTileQ(OptLevel);
  constexpr bool kPMultipleOfTileP = KernelOptimizations::IsPMultipleOfTileP(OptLevel);

  for (int row = 0; row < TileF.shape(0); row++) {
    if ((OpF == fastKronOp_N && (kPMultipleOfTileP || tileP + row < F.p())) ||
        (OpF == fastKronOp_T && (kQMultipleOfTileQ || tileQ + row < F.q()))) {
      uint32_t row_elems;
      ElemT* Fptr;
      if (OpF == fastKronOp_N) {
        row_elems = kQMultipleOfTileQ ? TileF.q() : MIN(TileF.q(), F.q() - tileQ);
        Fptr = F.data<ElemT>(tileP + row, tileQ, OpF);
      } else if (OpF == fastKronOp_T) {
        row_elems = kPMultipleOfTileP ? TileF.p() : MIN(TileF.p(), F.p() - tileP);
        Fptr = F.data<ElemT>(tileP, tileQ + row, OpF);
      }

      TileF.store_row(row, row_elems, Fptr);
    } else {
      TileF.zero_row(row);
    } 
  }
}

template<uint OptLevel, typename ElemT, fastKronOp OpX, typename X86VecT, uint FusedFacs, typename TileX_t, typename TrCache>
void transposeCache(const Matrix& X, const Factor& F, int fac, TileX_t& XTile, TrCache& Xcache, ElemT* tileBuff, uint32_t EffectiveTileK, uint32_t tileP) {
  const uint32_t VecTLen = X86VecT::VectorLen;
  const bool kPMultipleOfTileP = KernelOptimizations::IsPMultipleOfTileP(OptLevel);
  const bool kKMultipleOfTileK = KernelOptimizations::IsKMultipleOfTileK(OptLevel);
  const bool kTileKMultipleOfSlices = XTile.cols % VecTLen == 0;

  for (uint32_t m = 0; m < XTile.m(); m++) {
    for (uint32_t k = 0; k < XTile.cols; k += VecTLen * F.p()) {
      uint32_t p = 0;
      for (p = 0; p < Xcache.p(); p += VecTLen) {
        const bool UseAVXTrans = 
            VecTLen > 1 &&
            ((kKMultipleOfTileK && kTileKMultipleOfSlices)    || XTile.cols - k    >= VecTLen * F.p()) && 
            ((kPMultipleOfTileP && Xcache.p() % VecTLen == 0) || F.p() - tileP - p >= VecTLen) &&
            (Xcache.p() >= VecTLen);
        if (UseAVXTrans) {
          X86VecT slices[VecTLen];
          if (OpX == fastKronOp_N || (OpX == fastKronOp_T and fac != FusedFacs - 1)) {
            for (uint32_t slice = 0; slice < VecTLen; slice++) {
              const ElemT* ptr = (fac == FusedFacs - 1) ? 
                                 XTile.data(m, k/F.p() + slice, tileP + p) :
                                 &tileBuff[m * EffectiveTileK + k + slice*F.p() + tileP + p];
              slices[slice].load(ptr);
            }
            X86VecT::transpose(slices);
          } else if (OpX == fastKronOp_T and fac == FusedFacs - 1) {
            //Gather requires AVX2
            uint32_t gatherIdxs[VecTLen] = {0};
            for (uint pp = 0; pp < VecTLen; pp++) {
              const ElemT* ptr = XTile.data(m, k/F.p() + 0, tileP + p + pp);
              for (uint32_t slice = 0; slice < VecTLen; slice++) {
                gatherIdxs[slice] = slice * X.m() * F.p(); //TODO: Assumes TileM == 1
              }

              slices[pp].gather(ptr, gatherIdxs);
            }
          }

          for (uint32_t pp = 0; pp < VecTLen; pp++) {
            slices[pp].store(&Xcache.at(m, k/F.p(), p+pp));
          }
        } else {
          const uint32_t LeftSlices = (XTile.cols - k)/F.p();
          for (; p < MIN(Xcache.p(), F.p() - tileP); p++) {
            for (uint32_t slice = 0; slice < LeftSlices; slice++) {
              const ElemT* ptr = (fac == FusedFacs - 1) ? 
                                  XTile.data(m, k/F.p() + slice, tileP + p) :
                                  &tileBuff[m * EffectiveTileK + k + slice*F.p() + tileP + p];
              Xcache.at(m, k/F.p() + slice, p) = *ptr;
            }
          }

          Xcache.zero(m, k/F.p() + LeftSlices, p, 
                      m + 1, k/F.p() + VecTLen, Xcache.p());
        }
      }
    }
  }
}

template<typename XRegisters, typename FRegisters, typename YRegisters>
void mma(XRegisters& Xr, FRegisters& Fr, YRegisters& Yr) {
  
}

template<uint OptLevel, typename ElemT, typename X86VecT, uint MaxQ, uint MaxP, 
         uint TileP, uint TileQ, uint kTileK,
         uint TileM, uint FusedFacs, uint RegK, uint RegQ,
         fastKronOp OpF, typename SliceX, typename FCache_t, typename YTempBuffer>
__attribute__((always_inline)) static inline
void vectorMMAAndStore(uint32_t TileK, uint32_t tileM, uint32_t tileK, uint32_t tileP, uint32_t tileQ, uint32_t m, uint32_t q, uint32_t k, uint32_t fac, ElemT* TileX, FCache_t& FCache, uint32_t P, uint32_t Q, uint32_t K, SliceX& XTile, YTempBuffer& YCache, Matrix& Y, FusedParams<FusedFacs>& fusedParams) {
  constexpr bool kQMultipleOfTileQ = KernelOptimizations::IsQMultipleOfTileQ(OptLevel);
  constexpr bool kKMultipleOfTileK = KernelOptimizations::IsKMultipleOfTileK(OptLevel);

  const uint VectorLen = X86VecT::VectorLen;
  const uint32_t RegM = TileM;
  const uint32_t VecRegK = RegK/VectorLen;
  const uint32_t VecRegM = RegM; //(RegK < VectorLen) ? VectorLen/RegK : RegM;
  const uint32_t VecRegQ = RegQ;

  YRegisters<X86VecT, VecRegM, VecRegK, VecRegQ> YReg;

  if (tileP == 0) {
    YReg.zero();
  } else {
    for (uint32_t ym = 0; ym < YReg.m(); ym++) {
    for (uint32_t yq = 0; yq < YReg.q(); yq++) {
    for (uint32_t yk = 0; yk < YReg.k(); yk++) {
      YReg.at(ym, yk, yq).load(&YCache.at((m+ym), q + yq, k/TileP + yk * VectorLen));
    }}}
  }

  {
    for (uint32_t p = 0; p < TileP; p++) {
      XRegisters<X86VecT, VecRegM, VecRegK, 1> XReg;
      FRegisters<X86VecT, 1, VecRegQ> FReg;
      #pragma unroll
      for (uint32_t em = 0; em < XReg.m(); em++) {
        #pragma unroll
        for (uint32_t ek = 0; ek < XReg.k(); ek++) {
          XReg.at(em, ek, 0).load(&TileX[(m + em)*TileP*(kTileK/MaxP) + p * (kTileK/MaxP) + k/TileP + ek*VectorLen]);
      }}

      #pragma unroll
      for (uint32_t rq = 0; rq < FReg.shape(1); rq++) {
        // if (q == 0 && rq == 0) printf("p %d %f\n", p, TileF[p*TileQ + q + rq]);
        // if (OpF == fastKronOp_N)
          FReg.at(0, rq).broadcast(&FCache.at(p, q + rq));
        // else
        //   FReg.at(0, rq).broadcast(&FCache.at(q+rq, p));
      }

      #pragma unroll
      for (uint32_t rm = 0; rm < YReg.m(); rm++) {
      #pragma unroll
      for (uint32_t rk = 0; rk < YReg.k(); rk++) {
      #pragma unroll
      for (uint32_t rq = 0; rq < YReg.q(); rq++) {
        YReg.at(rm, rk, rq).fmadd(XReg.at(rm, rk, 0), FReg.at(0, rq));
      }}}
    }
  }

  if (TileP <= P && tileP < P - TileP) {
    for (uint32_t ym = 0; ym < VecRegM; ym++) {
    for (uint32_t yq = 0; yq < VecRegQ; yq++) {
    for (uint32_t yk = 0; yk < VecRegK; yk++) {
      YReg.at(ym, yk, yq).store(&YCache.at((m+ym), q + yq, k/TileP + yk * VectorLen));
    }}}
  } else {
    const uint32_t XTileSlices = TileK/P;
    const uint32_t XSlices     = K/P;

    for (uint32_t rm = 0; rm < VecRegM; rm++) {
    for (uint32_t rq = 0; rq < VecRegQ; rq++) {
    for (uint32_t rk = 0; rk < VecRegK; rk++) {
      const auto& reg = YReg.at(rm, rk, rq);
      if (fac > 0) {
        if (m + rm < XTile.m()) {
          reg.store(&YCache.at(m+rm, q + rq, k/TileP + rk * VectorLen));
        }
      } else {
        //TODO: Need to fix
        uint32_t memK;
        uint32_t slice = k/TileP + rk*VectorLen;
        if (!kKMultipleOfTileK && slice >= XTile.cols/P) continue;
        if (!kQMultipleOfTileQ && tileQ + q + rq >= Q) continue;

        if (FusedFacs > 1) {
          uint32_t xshCol = (rq + q) * XTileSlices + rk*VectorLen + k/TileP;
          //Scale shared mem slice idx to global mem idx
          uint32_t glSlice = (xshCol/XTileSlices)*XSlices;
          //Scale shared fused slice to global mem
          uint32_t sliceElem = ((xshCol%XTileSlices)/fusedParams.XShFusedSlices)*fusedParams.XglFusedSlices;
          //Elem idx in Fused Slice
          uint32_t elem = (tileK/TileK) * fusedParams.XShFusedSlices + xshCol%fusedParams.XShFusedSlices;
          memK = glSlice + sliceElem + elem; 
        } else {
          memK = (q + rq) * XSlices +
                  (tileK/TileK) * XTileSlices +
                  slice;
          if (TileQ < Q) {
            memK += tileQ * XSlices;
          }
        }

        if (m + rm < XTile.m()) {
          uint32_t slices = (kKMultipleOfTileK && kTileK % VectorLen == 0) ? 
                             VectorLen : (XTile.cols/P - slice);
          slices = MIN(VectorLen, slices);
          reg.store(Y.data<ElemT>(tileM + m + rm, memK, fastKronOp_N), slices);
        }
      }
    }}}
  }
}

template<typename ElemT, typename X86VecT, uint MaxQ, uint MaxP, uint TileP, 
         uint TileQ, uint kTileK, uint TileM, uint FusedFacs, 
         uint RegM, uint RegK, uint RegQ, uint OptLevel,
         fastKronOp OpX, fastKronOp OpF>
void threadWork(KernelParams<FusedFacs>& params,
               FusedParams<FusedFacs>& fusedParams, uint32_t tileM, uint32_t tileK, uint32_t tileQ, uint32_t TileK, uint32_t P, uint32_t Q, Matrix& X, Matrix& Y) {
  constexpr bool kXshSlicesSame    = KernelOptimizations::IsXshSlicesSame   (OptLevel);
  constexpr bool kKMultipleOfTileK = KernelOptimizations::IsKMultipleOfTileK(OptLevel);
  constexpr uint32_t kSlices = kTileK/MaxP;

  const uint32_t K = X.n();

  SliceCPU<ElemT, OpX> XTile(tileM, tileK, TileM, TileK,
                             kKMultipleOfTileK, P, X);

  const uint tid = omp_get_thread_num();
  YTempBuffer<ElemT, TileM, TileQ, kSlices> YCache((ElemT*)params.TileYs[tid]);

  for (int fac = FusedFacs - 1; fac >= 0; fac--) {
    TransposedDirectShared3D<fastKronOp_N, ElemT, TileM, TileP, kSlices> 
      TileX((ElemT*)params.TileXs[tid]);

    //Transpose X data and store to TileX to reduce TLB misses
    for (uint32_t tileP = 0; tileP < P; tileP += TileP) {
      const Factor F = Factor(P, Q, params.problem.f(fac).data());

      transposeCache<OptLevel, ElemT, OpX, X86VecT, FusedFacs>(X, F, fac, XTile, TileX, &YCache.at(0,0,0), TileK, tileP);

      DirectShared<OpF, ElemT, TileP, TileQ> TileF((ElemT*)params.TileFs[tid]);
      directCache<OptLevel, ElemT, OpF>(F, TileF, tileP, tileQ);
      
      for (uint32_t m = 0; m < XTile.m(); m += RegM) {
      for (uint32_t q = 0; q < TileQ; q += RegQ) {
      for (uint32_t k = 0; k < kSlices * TileP; k += RegK * TileP) {
        vectorMMAAndStore<OptLevel, ElemT, X86VecT, MaxQ, MaxP, TileP, TileQ, kTileK, TileM, FusedFacs, RegK, RegQ, OpF>
        (TileK, tileM, tileK, tileP, tileQ, m, q, k, fac, &TileX.at(0,0,0), TileF, P, Q, K, XTile, YCache, Y, fusedParams);
      }}}
    }
  }
}

template<typename ElemT, typename X86VecT, uint MaxQ, uint MaxP, uint TileP, 
         uint TileQ, uint kTileK, uint TileM, uint FusedFacs, 
         uint RegM, uint RegK, uint RegQ, uint OptLevel, 
         int XAlignment, int FAlignment,
         fastKronOp OpX, fastKronOp OpF>
void cpuKernel(KernelParams<FusedFacs>& params,
               FusedParams<FusedFacs>& fusedParams,
               DistributedParams& distParams,
               EpilogueParams& epilogueParams) {
  Matrix X = params.problem.x();
  Matrix Y = params.problem.y();
  Factor F = params.problem.f(0);

  const uint32_t K = X.n();

  static_assert(RegM == TileM, "x86 requires RegM == TileM");

  static_assert(kTileK % RegK == 0);
  static_assert(TileQ % RegQ == 0);
  static_assert(FusedFacs == 1 || 
                (FusedFacs > 1 && 
                 MaxP <= TileP && MaxQ <= TileQ && MaxP == MaxQ && 
                 OptLevel == KernelOptimizations::MaxOptLevel()));

  constexpr bool kFactorShapeSame = KernelOptimizations::IsFactorShapeSame(OptLevel);

  const uint Q = (kFactorShapeSame) ? MaxQ : F.q();
  const uint P = (kFactorShapeSame) ? MaxP : F.p();

  const uint XshSlices = getXshSlices<OptLevel, kTileK, MaxP>(params);
  const uint XSlices   = getXSlices  <OptLevel, MaxQ>(Y, params);
  const uint TileK     = getXTileK   <OptLevel, kTileK>(params);

  if (OpX == fastKronOp_N) {
    #pragma omp parallel for collapse(3)
    for (uint32_t tileM = 0; tileM < X.m(); tileM += TileM) {
    for (uint32_t tileK = 0; tileK < K    ; tileK += TileK) {
    for (uint32_t tileQ = 0; tileQ < Q    ; tileQ += TileQ) {
      threadWork<ElemT, X86VecT, MaxQ, MaxP, TileP, TileQ, kTileK, TileM, FusedFacs, RegM, RegK, RegQ, OptLevel, OpX, OpF> (
        params, fusedParams, tileM, tileK, tileQ, TileK, P, Q, X, Y
      );
    }}}
  } else if (OpX == fastKronOp_T) {
    #pragma omp parallel for collapse(3)
    for (uint32_t tileQ = 0; tileQ < Q    ; tileQ += TileQ) {
    for (uint32_t tileM = 0; tileM < X.m(); tileM += TileM) {
    for (uint32_t tileK = 0; tileK < K    ; tileK += TileK) {
      threadWork<ElemT, X86VecT, MaxQ, MaxP, TileP, TileQ, kTileK, TileM, FusedFacs, RegM, RegK, RegQ, OptLevel, OpX, OpF> (
        params, fusedParams, tileM, tileK, tileQ, TileK, P, Q, X, Y
      );
    }}}
  }
}