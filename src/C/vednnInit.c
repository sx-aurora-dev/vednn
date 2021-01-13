#include "vednn-def.h" // "C" scratchpad funcs
#include <stdlib.h>
#include <stdio.h>

int __vednn_omp_num_threads = 0 ;

__attribute__((constructor))
void __vednn_init() {
  __vednn_omp_num_threads = 0;
  {
    const char *v = getenv("OMP_NUM_THREADS") ;
    const int   i = (v ==NULL ?  0 : atoi(v)) ;
    if (i > 0) __vednn_omp_num_threads = i ;
  }
  if (__vednn_omp_num_threads <= 0) {
    const char *v = getenv("VE_OMP_NUM_THREADS");
    const int   i = (v==NULL? 0: atoi(v));
    if (i > 0) __vednn_omp_num_threads = i ;
  }

#if defined(_OPENMP)
  if( __vednn_omp_num_threads <= 0 )
    __vednn_omp_num_threads = omp_get_max_threads();
#endif

  vednn_init_global_scratchpads();
}

__attribute__((destructor))
void __vednn_free() {
  vednn_free_global_scratchpads();
}
