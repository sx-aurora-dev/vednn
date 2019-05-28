#include <stdlib.h>
#include <stdio.h>

int __vednn_omp_num_threads = 0 ;

__attribute__((constructor))
void __vednn_init() {
  const char *v = getenv("OMP_NUM_THREADS") ;
  const int   i = (v ==NULL ?  0 : atoi(v)) ;

  if( i > 0 )
    __vednn_omp_num_threads = i ;
}
