#include <sys/time.h>
#include <time.h>

#ifndef __UTILS_H__
#define __UTILS_H__

/**************************************************
                    Timing functions
**************************************************/
double convertTimeValToDouble(struct timeval _time) {
  return ((double)_time.tv_sec)*1e6 + ((double)_time.tv_usec);
}

struct timeval getTimeOfDay () {
  struct timeval _time;

  if (gettimeofday (&_time, NULL) == -1) {
    fprintf (stderr, "gettimeofday returned -1\n");
    perror ("");
    abort ();
  }

  return _time;
}

double getCurrTime() {
  return convertTimeValToDouble(getTimeOfDay());
}

int ilog2(uint x)
{
  return sizeof(uint32_t) * CHAR_BIT - __builtin_clz(x) - 1;
}

bool isPowerOf2(uint x)
{
    return (x & (x - 1)) == 0;
}
#endif