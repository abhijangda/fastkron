#include <immintrin.h>
#include <cstring>
#include <cstdlib>
#include <omp.h>

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

  void store(ElemT* ptr) {
    vectorStore<ElemT, UnderlyingVecT>(ptr, vec.data);
  }
  
  void store(ElemT* ptr, uint32_t sz) {
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

template<typename ElemT, typename X86VecT, uint MaxQ, uint MaxP, 
         uint TileP, uint TileQ, uint kTileK,
         uint TileM, uint FusedFacs, uint RegK, uint RegQ,
         fastKronOp OpF,
         bool kKMultipleOfTileK, bool kQMultipleOfTileQ, typename SliceX>
__attribute__((always_inline)) static inline
void vectorMMAAndStore(uint32_t TileK, uint32_t tileM, uint32_t tileK, uint32_t tileP, uint32_t tileQ, uint32_t m, uint32_t q, uint32_t k, uint32_t fac, ElemT* TileX, ElemT* TileF, uint32_t P, uint32_t Q, uint32_t K, SliceX& XTile, ElemT* tileBuff, Matrix& Y, FusedParams<FusedFacs>& fusedParams) {
  //TODO: Different vector lengths. AVX512, AVX256, AVX, SSE4.2, no vector based on underlying architecture
  const uint VectorLen = X86VecT::VectorLen;
  const uint32_t RegM = TileM;
  const uint32_t VecRegK = RegK/VectorLen;
  const uint32_t VecRegM = RegM; //(RegK < VectorLen) ? VectorLen/RegK : RegM;
  const uint32_t VecRegQ = RegQ;

  X86VecT yReg[VecRegM][VecRegQ][VecRegK];

  if (tileP == 0) {
    for (uint32_t ym = 0; ym < VecRegM; ym++) {
    for (uint32_t yq = 0; yq < VecRegQ; yq++) {
    for (uint32_t yk = 0; yk < VecRegK; yk++) {
      yReg[ym][yq][yk].zero();
    }}}
  } else {
    for (uint32_t ym = 0; ym < VecRegM; ym++) {
    for (uint32_t yk = 0; yk < VecRegK; yk++) {
    for (uint32_t yq = 0; yq < VecRegQ; yq++) {
      yReg[ym][yq][yk].load(&tileBuff[(m+ym)*TileQ*(kTileK/MaxP) + (q+yq)*(kTileK/MaxP) + k/TileP+yk*VectorLen]);
    }}}
  }

  if (false && VectorLen == 8 && VecRegM == 1 && VecRegK == 2 && VecRegQ == 4) {
    #if 0
    for (uint32_t p = 0; p < TileP; p += 2) {
      {
        ElemT* xptr = &TileX[(m)*TileP*(TileK/P) + p * (TileK/P) + k/TileP + 0];
        VectorType x0, x1;

        x0.load(xptr);
        x1.load(xptr + 1*VectorLen);

        VectorType f0, f1, f2, f3;
        if (OpF == fastKronOp_N) {
          ElemT* fptr = &TileF[p*TileQ + q];
          f0.broadcast(fptr);
          f1.broadcast(fptr + 1);
          f2.broadcast(fptr + 2);
          f3.broadcast(fptr + 3);
        } else {
          ElemT* fptr = &TileF[q*TileP + p];
          f0.broadcast(fptr);
          f1.broadcast(fptr + TileP);
          f2.broadcast(fptr + 2*TileP);
          f3.broadcast(fptr + 3*TileP);
        }

        _mm_prefetch(&TileX[(m)*TileP*(TileK/P) + (p + 1) * (TileK/P) + k/TileP + 0], _MM_HINT_T1);

        yReg[0][0][0].fmadd(x0, f0);
        yReg[0][1][0].fmadd(x0, f1);
        yReg[0][2][0].fmadd(x0, f2);
        yReg[0][3][0].fmadd(x0, f3);
        yReg[0][0][1].fmadd(x1, f0);
        yReg[0][1][1].fmadd(x1, f1);
        yReg[0][2][1].fmadd(x1, f2);
        yReg[0][3][1].fmadd(x1, f3);
      }
      {
        ElemT* xptr = &TileX[(m)*TileP*(TileK/P) + (p + 1) * (TileK/P) + k/TileP + 0];
        VectorType x0, x1;
        x0.load(xptr);
        x1.load(xptr + 1*VectorLen);

        VectorType f0, f1, f2, f3;
        if (OpF == fastKronOp_N) {
          ElemT* fptr = &TileF[(p+1)*TileQ + q];
          f0.broadcast(fptr);
          f1.broadcast(fptr + 1);
          f2.broadcast(fptr + 2);
          f3.broadcast(fptr + 3);
        } else {
          ElemT* fptr = &TileF[q*TileP + p+1];
          f0.broadcast(fptr);
          f1.broadcast(fptr + TileP);
          f2.broadcast(fptr + 2*TileP);
          f3.broadcast(fptr + 3*TileP);
        }

        if (p + 2 < TileP) {
          _mm_prefetch(&TileX[(m)*TileP*(TileK/P) + (p + 2) * (TileK/P) + k/TileP + 0], _MM_HINT_T1);
          if (OpF == fastKronOp_N) {
            ElemT* fptr = &TileF[(p+1)*TileQ + q];
            _mm_prefetch(fptr + 4, _MM_HINT_T1);
          }
        }

        yReg[0][0][0].fmadd(x0, f0);
        yReg[0][1][0].fmadd(x0, f1);
        yReg[0][2][0].fmadd(x0, f2);
        yReg[0][3][0].fmadd(x0, f3);
        yReg[0][0][1].fmadd(x1, f0);
        yReg[0][1][1].fmadd(x1, f1);
        yReg[0][2][1].fmadd(x1, f2);
        yReg[0][3][1].fmadd(x1, f3);

      }
    }
    #endif
  } else {
    for (uint32_t p = 0; p < TileP; p++) {
      X86VecT XReg[VecRegM][VecRegK];
      X86VecT FReg[VecRegQ];
      #pragma unroll
      for (uint32_t em = 0; em < VecRegM; em++) {
        #pragma unroll
        for (uint32_t ek = 0; ek < VecRegK; ek++) {
          XReg[em][ek].load(&TileX[(m + em)*TileP*(kTileK/MaxP) + p * (kTileK/MaxP) + k/TileP + ek*VectorLen]);
      }}

      #pragma unroll
      for (uint32_t rq = 0; rq < VecRegQ; rq++) {
        // if (q == 0 && rq == 0) printf("p %d %f\n", p, TileF[p*TileQ + q + rq]);
        if (OpF == fastKronOp_N)
          FReg[rq].broadcast(&TileF[p*TileQ + q + rq]);
        else
          FReg[rq].broadcast(&TileF[(q+rq)*TileP + p]);
      }

      #pragma unroll
      for (uint32_t rm = 0; rm < VecRegM; rm++) {
      #pragma unroll
      for (uint32_t rk = 0; rk < VecRegK; rk++) {
      #pragma unroll
      for (uint32_t rq = 0; rq < VecRegQ; rq++) {
        yReg[rm][rq][rk].fmadd(XReg[rm][rk], FReg[rq]);
      }}}
    }
  }

  if (TileP <= P && tileP < P - TileP) {
    for (uint32_t ym = 0; ym < VecRegM; ym++) {
    for (uint32_t yq = 0; yq < VecRegQ; yq++) {
    for (uint32_t yk = 0; yk < VecRegK; yk++) {
      uint32_t idx = (q+yq)*(kTileK/MaxP) + k/TileP+yk*VectorLen;
      yReg[ym][yq][yk].store(&tileBuff[(m+ym)*TileQ*(kTileK/MaxP) + idx]);
    }}}
  } else {
    const uint32_t XTileSlices = TileK/P;
    const uint32_t XSlices     = K/P;

    for (uint32_t rm = 0; rm < VecRegM; rm++) {
    for (uint32_t rq = 0; rq < VecRegQ; rq++) {
    for (uint32_t rk = 0; rk < VecRegK; rk++) {
      auto reg = yReg[rm][rq][rk];
      const uint32_t cacheK = (rq + q) * XTileSlices + rk*VectorLen + k/TileP;
      if (fac > 0) {
        if (m + rm < XTile.m()) {
          // ElemT b[8]; _mm256_storeu_ps(b, reg); printf("%f %f %f %f\n", b[0], b[1], b[2], b[3]);
          reg.store(&tileBuff[(m+rm)*TileK + cacheK]);
        }
      } else {
        //TODO: Need to fix
        uint32_t memK;
        uint32_t slice = k/TileP + rk*VectorLen;
        if (!kKMultipleOfTileK && slice >= XTile.cols/P) continue;
        if (!kQMultipleOfTileQ && tileQ + q + rq >= Q) continue;

        if (FusedFacs > 1) {
          uint32_t xshCol = cacheK;
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
         int XAlignment, int FAlignment,
         fastKronOp OpX, fastKronOp OpF>
void threadWork(KernelParams<FusedFacs>& params,
               FusedParams<FusedFacs>& fusedParams, uint32_t tileM, uint32_t tileK, uint32_t tileQ, uint32_t TileK, uint32_t P, uint32_t Q, Matrix& X, Matrix& Y) {
  constexpr bool kFactorShapeSame  = KernelOptimizations::IsFactorShapeSame (OptLevel);
  constexpr bool kXshSlicesSame    = KernelOptimizations::IsXshSlicesSame   (OptLevel);
  constexpr bool kQMultipleOfTileQ = KernelOptimizations::IsQMultipleOfTileQ(OptLevel);
  constexpr bool kPMultipleOfTileP = KernelOptimizations::IsPMultipleOfTileP(OptLevel);
  constexpr bool kKMultipleOfTileK = KernelOptimizations::IsKMultipleOfTileK(OptLevel);
  constexpr bool kQLeTileQ         = KernelOptimizations::IsQLeTileQ        (OptLevel);
  constexpr bool kTileKSame        = KernelOptimizations::IsTileKSame       (OptLevel);

  const uint32_t K = X.n();
  const uint32_t VectorLen = X86VecT::VectorLen;

  Slice<ElemT, OpX> XTile(tileM, tileK, 
                            (TileM == 1) ? 1 : MIN(TileM, X.m() - tileM), 
                            (kKMultipleOfTileK) ? TileK : MIN(TileK, K - tileK),
                            P, P, //TODO: setting this to P because XTile.data is not right for GPU backend
                            X);
  const uint tid = omp_get_thread_num();
  ElemT* tileBuff = (ElemT*)params.TileYs[tid];

  for (int fac = FusedFacs - 1; fac >= 0; fac--) {
    ElemT* TileX = (ElemT*)params.TileXs[tid];
    //Transpose X data and store to TileX to reduce TLB misses
    for (uint32_t tileP = 0; tileP < P; tileP += TileP) {
      for (uint32_t m = 0; m < XTile.m(); m++) {
        uint32_t NumSlices = VectorLen;
        const bool kTileKMultipleOfSlices = TileK % NumSlices == 0;
        for (uint32_t k = 0; k < TileK; k += NumSlices * P) {
          uint32_t p = 0;
          for (p = 0; p < TileP; p += VectorLen) {
            const bool ValidAVXTranspose =
                  ((kKMultipleOfTileK && kTileKMultipleOfSlices) || TileK - k >= NumSlices * P) && 
                  ((kPMultipleOfTileP && TileP % VectorLen == 0) || P - tileP - p >= VectorLen) &&
                  (TileP >= VectorLen);
            if (VectorLen > 1 && ValidAVXTranspose) {
              X86VecT slices[VectorLen];
              if (OpX == fastKronOp_N || (OpX == fastKronOp_T and fac != FusedFacs - 1)) {
                for (uint32_t sliceIdx = 0; sliceIdx < NumSlices; sliceIdx++) {
                  const ElemT* ptr = (fac == FusedFacs - 1) ? XTile.data(m, k + sliceIdx*P + tileP + p, 0) :
                                                              &tileBuff[m * TileK + k + sliceIdx*P + tileP + p];
                  slices[sliceIdx].load(ptr);
                }
                X86VecT::transpose(slices);
              } else if (OpX == fastKronOp_T and fac == FusedFacs - 1) {
                //TODO: Gather works with AVX2
                uint32_t gatherIdxs[VectorLen] = {0};
                for (uint pp = 0; pp < VectorLen; pp++) {
                  const ElemT* ptr = XTile.data(m, k + 0*P + tileP + p + pp, 0);
                  for (uint32_t sliceIdx = 0; sliceIdx < NumSlices; sliceIdx++) {
                    gatherIdxs[sliceIdx] = sliceIdx * X.m() * P; //TODO: Assumes TileM == 1
                  }

                  slices[pp].gather(ptr, gatherIdxs);
                }
              }

              for (uint32_t pp = 0; pp < VectorLen; pp++) {
                slices[pp].store(&TileX[m*TileP*(kTileK/MaxP) + (p + pp)*(kTileK/MaxP) + k/P]);
              }
            } else {
              uint32_t NumSlices1 = (TileK - k)/P;
              uint32_t remainingP = P - tileP - p;
              for (; p < MIN(TileP, P - tileP); p++) {
                for (uint32_t sliceIdx = 0; sliceIdx < NumSlices1; sliceIdx++) {
                  const ElemT* ptr = (fac == FusedFacs - 1) ? XTile.data(m, k + sliceIdx*P + tileP + p, 0) :
                                                              &tileBuff[m * TileK + k + sliceIdx*P + tileP + p];
                  TileX[m*TileP*(kTileK/MaxP) + p*(kTileK/MaxP) + k/P + sliceIdx] = *ptr;
                }
              }

              for (; p < TileP; p++) {
                for (uint32_t sliceIdx = NumSlices1; sliceIdx < NumSlices; sliceIdx++) {
                  TileX[m*TileP*(kTileK/MaxP) + p*(kTileK/MaxP) + k/P + sliceIdx] = 0.0f;
                }
              }
            }
          }
        }

        if (false && tileP == 0 && tid == 0) {
          printf("tileP %d k %d\n", tileP, tileK);
          for (int ii = 0; ii < TileP; ii++) {
            if (ii > 2) continue;
            printf("ii %d \n", ii);
            for (int jj = 0; jj < kTileK/MaxP; jj++) {
              printf("%d %.1f \n", jj, TileX[ii * (kTileK/MaxP) + jj]);
            }
            // printf("\n");
          }
        }
      }

      ElemT* TileF = (ElemT*)params.TileFs[tid]; //[TileP][TileQ];
      Factor F = params.problem.f(fac);
      if (OpF == fastKronOp_N) {
        for (int p = 0; p < TileP; p++) {
          if (kPMultipleOfTileP || tileP + p < P) {
            memcpy(&TileF[p*TileQ + 0], F.data<ElemT>(tileP + p, tileQ, OpF), 
                    (kQMultipleOfTileQ ? TileQ : MIN(TileQ, Q - tileQ)) * sizeof(ElemT));
            if (!kQMultipleOfTileQ && Q - tileQ < TileQ) {
              memset(&TileF[p*TileQ + Q - tileQ], 0, (TileQ - (Q - tileQ)) * sizeof(ElemT));
            }
          }
          else {
            memset(&TileF[p*TileQ + 0], 0, TileQ * sizeof(ElemT));
          }
        }
      } else if (OpF == fastKronOp_T) {
        //Access TileF in mma as transpose
        for (int q = 0; q < TileQ; q++) {
          if (tileQ + q < Q) {
            memcpy(&TileF[q*TileP + 0], F.data<ElemT>(tileP, tileQ + q, OpF), MIN(TileP, P - tileP) * sizeof(ElemT));
            if (P - tileP < TileP) {
              memset(&TileF[q*TileP + P - tileP], 0, (TileP - (P - tileP))*sizeof(ElemT));
            }
          } else {
            memset(&TileF[q*TileP], 0, TileP * sizeof(ElemT));
          }
        }
      }
      
      for (uint32_t m = 0; m < XTile.m(); m += RegM) {
      for (uint32_t q = 0; q < TileQ; q += RegQ) {
      for (uint32_t k = 0; k < kTileK/MaxP * TileP; k += RegK * TileP) {
        vectorMMAAndStore<ElemT, X86VecT, MaxQ, MaxP, TileP, TileQ, kTileK, TileM, FusedFacs, RegK, RegQ, OpF, kKMultipleOfTileK, kQMultipleOfTileQ>
        (TileK, tileM, tileK, tileP, tileQ, m, q, k, fac, TileX, TileF, P, Q, K, XTile, tileBuff, Y, fusedParams);
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

  const uint32_t YRegs = RegM * RegK * RegQ;
  const uint32_t XRegs = RegM * RegK;
  const uint32_t FRegs = RegQ;

  const uint32_t VectorLen = (XAlignment == 8 && RegK % 8 == 0) ? 8 : 1; //AVX256 length

  // static_assert(XAlignment < 8 or (XAlignment == 8 and RegK % VectorLen == 0));
  static_assert(kTileK % RegK == 0);
  static_assert(TileQ % RegQ == 0);
  static_assert(FusedFacs == 1 || 
                (FusedFacs > 1 && 
                 MaxP <= TileP && MaxQ <= TileQ && MaxP == MaxQ && 
                 OptLevel == KernelOptimizations::MaxOptLevel()));

  constexpr bool kFactorShapeSame  = KernelOptimizations::IsFactorShapeSame (OptLevel);
  constexpr bool kXshSlicesSame    = KernelOptimizations::IsXshSlicesSame   (OptLevel);
  constexpr bool kQMultipleOfTileQ = KernelOptimizations::IsQMultipleOfTileQ(OptLevel);
  constexpr bool kPMultipleOfTileP = KernelOptimizations::IsPMultipleOfTileP(OptLevel);
  constexpr bool kKMultipleOfTileK = KernelOptimizations::IsKMultipleOfTileK(OptLevel);
  constexpr bool kQLeTileQ         = KernelOptimizations::IsQLeTileQ        (OptLevel);
  constexpr bool kTileKSame        = KernelOptimizations::IsTileKSame       (OptLevel);

  const uint Q = (kFactorShapeSame) ? MaxQ : F.q();
  const uint P = (kFactorShapeSame) ? MaxP : F.p();

  const uint XshSlices = getXshSlices<kFactorShapeSame, kTileK, MaxP>(params);
  const uint XSlices   = getXSlices  <kFactorShapeSame, MaxQ>(Y, params);
  const uint TileK     = getXTileK   <kTileKSame, kTileK>(params);

  // uint threads = omp_get_max_threads();  
  //TODO: Allocate this in fastKron_initBackend
  // static ElemT** TileXs = params.TileXs;
  // static ElemT** TileYs = {nullptr};
  // static ElemT** TileFs = {nullptr};

  // if (TileXs[0] == nullptr) {
  //   for (int i = 0; i < 128; i++)  {
  //     TileXs[i] = (ElemT*)aligned_alloc(4096, TileM * kTileK * sizeof(ElemT));
  //     TileYs[i] = (ElemT*)aligned_alloc(4096, TileM * TileQ * (kTileK/MaxP) * sizeof(ElemT));
  //     TileFs[i] = (ElemT*)aligned_alloc(4096, TileP * TileQ * sizeof(ElemT));
  //   }
  // }

  if (OpX == fastKronOp_N) {
    #pragma omp parallel for collapse(3)
    for (uint32_t tileM = 0; tileM < X.m(); tileM += TileM) {
    for (uint32_t tileK = 0; tileK < K    ; tileK += TileK) {
    for (uint32_t tileQ = 0; tileQ < Q    ; tileQ += TileQ) {
      threadWork<ElemT, X86VecT, MaxQ, MaxP, TileP, TileQ, kTileK, TileM, FusedFacs, RegM, RegK, RegQ, OptLevel, XAlignment, FAlignment, OpX, OpF> (
        params, fusedParams, tileM, tileK, tileQ, TileK, P, Q, X, Y
      );
    }}}
  } else if (OpX == fastKronOp_T) {
    #pragma omp parallel for collapse(3)
    for (uint32_t tileQ = 0; tileQ < Q    ; tileQ += TileQ) {
    for (uint32_t tileM = 0; tileM < X.m(); tileM += TileM) {
    for (uint32_t tileK = 0; tileK < K    ; tileK += TileK) {
      threadWork<ElemT, X86VecT, MaxQ, MaxP, TileP, TileQ, kTileK, TileM, FusedFacs, RegM, RegK, RegQ, OptLevel, XAlignment, FAlignment, OpX, OpF> (
        params, fusedParams, tileM, tileK, tileQ, TileK, P, Q, X, Y
      );
    }}}
  }

  // free(TileXs);
}