#include <sys/time.h>
#include <time.h>

#pragma once

#define CUDA_LAST_ERROR do {                        \
  cudaError_t e = cudaGetLastError();               \
  if (e != cudaSuccess) {                           \
    printf("Failed: Cuda error %s:%d '%s'\n",       \
        __FILE__,__LINE__,cudaGetErrorString(e));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while (0)                                         \

#define NCCLCHECK(cmd) do {                         \
  ncclResult_t r = cmd;                             \
  if (r!= ncclSuccess) {                            \
    printf("Failed, NCCL error %s:%d '%s'\n",             \
        __FILE__,__LINE__,ncclGetErrorString(r));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

#define PTHREAD_BARRIER_CHECK(x) do {                        \
  if (x != 0 && x != PTHREAD_BARRIER_SERIAL_THREAD) {                           \
    printf("Failed: pthread barrier error %s:%d\n",       \
        __FILE__,__LINE__);   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while (0)                                         \

#define MIN(x,y) (((x) < (y)) ? (x) : (y))
#define MAX(x,y) (((x) > (y)) ? (x) : (y))
#define DIVUP(x,y) (((x) + (y) - 1)/((y)))
#define ROUNDUP(x,y) (DIVUP(x,y)*(y))
#define CUDA_WARP_SIZE 32

static constexpr int log2(uint n) {return 31 - __builtin_clz(n);}
static constexpr int log2(int n) {return 31 - __builtin_clz(n);}

static double convertTimeValToDouble(struct timeval _time) {
  return ((double)_time.tv_sec)*1e6 + ((double)_time.tv_usec);
}

static struct timeval getTimeOfDay () {
  struct timeval _time;

  if (gettimeofday (&_time, NULL) == -1) {
    fprintf (stderr, "gettimeofday returned -1\n");
    perror ("");
    abort ();
  }

  return _time;
}

static double getCurrTime() {
  return convertTimeValToDouble(getTimeOfDay());
}

static int ilog2(uint x)
{
  return sizeof(uint32_t) * CHAR_BIT - __builtin_clz(x) - 1;
}

static bool isPowerOf2(uint x)
{
    return (x & (x - 1)) == 0;
}