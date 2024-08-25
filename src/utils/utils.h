#include <climits>

#include <sys/time.h>
#include <time.h>
#include <cstring>
#include <cstdint>

#pragma once

#define CUDA_LAST_ERROR do {                        \
  cudaError_t e = cudaGetLastError();               \
  if (e != cudaSuccess) {                           \
    printf("Failed: Cuda error %s:%d '%s'\n",       \
        __FILE__,__LINE__,cudaGetErrorString(e));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while (0)                                         \

#define CUDA_CHECK(cmd) do {                        \
  cudaError_t e = cmd;                              \
  if(e != cudaSuccess and                           \
     e != cudaErrorPeerAccessAlreadyEnabled) {      \
    printf("Failed: Cuda error %s:%d '%s'\n",       \
        __FILE__,__LINE__,cudaGetErrorString(e));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)                                          \

#define HIP_CHECK(cmd) do {                        \
  hipError_t e = cmd;                              \
  if(e != hipSuccess and                           \
     e != hipErrorPeerAccessAlreadyEnabled) {      \
    printf("Failed: HIP error %s:%d '%s'\n",       \
        __FILE__,__LINE__,hipGetErrorString(e));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)                                          \

#define NCCLCHECK(cmd) do {                         \
  ncclResult_t r = cmd;                             \
  if (r!= ncclSuccess) {                            \
    printf("Failed, NCCL error %s:%d '%s'\n",       \
        __FILE__,__LINE__,ncclGetErrorString(r));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

#define PTHREAD_BARRIER_CHECK(x) do {               \
  if (x != 0 &&                                     \
      x != PTHREAD_BARRIER_SERIAL_THREAD) {         \
    printf("Failed: pthread barrier error %s:%d\n", \
        __FILE__,__LINE__);                         \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while (0)                                         \

#define MIN(x,y)       (((x) < (y)) ? (x) : (y))
#define MAX(x,y)       (((x) > (y)) ? (x) : (y))
#define DIVUP(x,y)     (((x) + (y) - 1)/((y)))
#define ROUNDUP(x,y)   (DIVUP(x,y)*(y))
#define ROUNDDOWN(x,y) (x/y)*y

#define CUDA_WARP_SIZE 32U
#define NULL_CHECK(x) if ((x) == nullptr) return fastKronInvalidArgument;

static constexpr int log2(uint n) {return 31 - __builtin_clz(n);}
static constexpr int log2(int n) {return 31 - __builtin_clz(n);}

static inline double convertTimeValToDouble(struct timeval _time) {
  return ((double)_time.tv_sec)*1e6 + ((double)_time.tv_usec);
}

static inline struct timeval getTimeOfDay () {
  struct timeval _time;

  if (gettimeofday (&_time, NULL) == -1) {
    fprintf (stderr, "gettimeofday returned -1\n");
    perror ("");
    abort ();
  }

  return _time;
}

static inline double getCurrTime() {
  return convertTimeValToDouble(getTimeOfDay());
}

static inline int ilog2(uint x) {
  return sizeof(uint32_t) * CHAR_BIT - __builtin_clz(x) - 1;
}

static inline bool isPowerOf2(uint x) {
    return (x & (x - 1)) == 0;
}

static inline int ffs(uint x) {
  for (int i = 0; i < 32; i++) {
    if (((x >> i) & 1) == 1) return i;
  }
  return -1;
}

template<typename T>
static inline void memset(T* ptr, size_t nelem, T val) {
  for (uint32_t i = 0; i < nelem; i++)
    ptr[i] = val;
}

static inline void parallelCopy(char* trash1, char* trash2, uint32_t sz) {
  #pragma omp parallel for
  for (uint32_t i = 0; i < sz; i++) {
    trash1[i] = trash2[i];
  }
}
