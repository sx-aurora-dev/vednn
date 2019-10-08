#include "vednnActivationForward.h"
#include "vednn-def.h"
#include <stdio.h>

static inline vednnError_t
vednnActivationForward_wrapper(
    vednnActivationForward_t  pFunc,
    VEDNN_ACTIVATIONFWD_ARGS )
{
#ifndef VEDNN_USE_OPENMP
  return pFunc(VEDNN_ACTIVATIONFWD_ARGS_LIST);
#else
  if ( __vednn_omp_num_threads == 1 ) {
    return pFunc(VEDNN_ACTIVATIONFWD_ARGS_LIST);
  }
  else {
    vednnError_t rc = VEDNN_SUCCESS ;
#pragma omp parallel reduction(|:rc)
    {
      int64_t nthreads = omp_get_num_threads() ;
      int64_t threadid = omp_get_thread_num() ;

      int64_t eachNElement = nElements / nthreads ;
      int64_t remain       = nElements % nthreads ;

      int64_t elementBegin = eachNElement * threadid + ( threadid < remain ? threadid : remain ) ;
      int64_t myElement    = eachNElement + ( threadid < remain ? 1 : 0 ) ;

      if( myElement == 0 ) {
        rc |= VEDNN_SUCCESS ;
      }
      else {
        float* _pDataIn  = ((float *)pDataIn) + elementBegin ;
        float* _pDataOut = ((float *)pDataOut) + elementBegin ;

        rc |= pFunc((void*)_pDataIn, (void*) _pDataOut, myElement) ;
      }
    }
    return rc ;
  }
#endif
}

/* ----------------------------------------------------------------------- */
vednnError_t vednnActivationForward(
    const vednnActivationMode_t mode,
    VEDNN_ACTIVATIONFWD_ARGS )
{
#define OMPWRAP( IMPL ) WRAP_RET(vednnActivationForward_##IMPL, \
    vednnActivationForward_wrapper, VEDNN_ACTIVATIONFWD_ARGS_LIST )
  switch(mode) {
  case VEDNN_ACTIVATION_RELU :
    OMPWRAP(Relu);
  }
  fprintf(stderr, "VEDNN Error : vednnActivationForward : Invalid Parameter !!\n") ;
  return VEDNN_ERROR_INVALID_PARAM ;
#undef OMPWRAP
}
// vim: et sw=2 ts=2
