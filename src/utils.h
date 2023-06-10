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

#endif