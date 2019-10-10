#ifndef __TIMER_H__
#define __TIMER_H__

#include <sys/time.h>
#include <stdlib.h>

#ifdef __ve__

static unsigned long long inline
__cycle()
{
  void *vehva = (void *)0x1000;
  unsigned long long val;
  asm volatile ("lhm.l %0,0(%1)":"=r"(val):"r"(vehva));
  return val;
}

#else

#if defined(__x86_64__)

static inline unsigned long long __cycle()
{
  unsigned hi, lo;
  __asm__ __volatile__ ("rdtsc" : "=a"(lo), "=d"(hi));
  return ( (unsigned long long)lo)|( ((unsigned long long)hi)<<32 );
}

#elif defined(__i386__)

static inline unsigned long long __cycle()
{
  unsigned long long ret;
  __asm__ __volatile__("rdtsc" : "=A" (ret));
  return ret;
}

#else

#error "Must be define the function which get processor cycles, here."

#endif

static double inline
__clock()
{
  struct timeval tv;
  int rv;

  rv = gettimeofday(&tv, NULL);
  if (rv) {
    fprintf(stderr, "gettimeofday() returned with %d\n", rv);
    exit(1);
  }

  return tv.tv_sec + tv.tv_usec * 1.0e-6;
}

#endif

#endif
