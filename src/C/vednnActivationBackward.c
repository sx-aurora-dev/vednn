#include "vednnActivationBackward.h"
#include "vednn-def.h"
#include <stdio.h>
#include <stdint.h>

  static inline vednnError_t
vednnActivationBackward_wrapper( vednnActivationBackward_t  pFunc,
    VEDNN_ACTIVATIONBKW_ARGS )
{
#ifndef VEDNN_USE_OPENMP
  return pFunc(VEDNN_ACTIVATIONBKW_ARGS_LIST);
#else
  if ( __vednn_omp_num_threads == 1 ) {
    return pFunc(VEDNN_ACTIVATIONBKW_ARGS_LIST);
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
        float* _pDataGradOut = ((float *)pDataGradOut) + elementBegin ;
        float* _pDataIn      = ((float *)pDataIn) + elementBegin ;
        float* _pDataGradIn  = ((float *)pDataGradIn) + elementBegin ;

        rc |= pFunc((void*)_pDataGradOut, (void*)_pDataIn, (void*) _pDataGradIn, myElement) ;
      }
    }
    return rc ;
  }
#endif // openmp
}

/* ------------------------- public API ---------------------------------- */
vednnError_t vednnActivationBackward(
    const vednnActivationMode_t mode,
    VEDNN_ACTIVATIONBKW_ARGS)
{
#define OMPWRAP( IMPL ) WRAP_RET(IMPL, \
    vednnConvolutionForward_wrapper, VEDNN_ACTIVATIONBKW_ARGS_LIST)
  switch(mode) {
    case VEDNN_ACTIVATION_RELU :
      OMPWRAP( vednnActivationBackward_Relu );
  }
  fprintf(stderr, "VEDNN Error : vednnActivationBackward : Invalid Parameter !!\n") ;
  return VEDNN_ERROR_INVALID_PARAM ;
#undef OMPWRAP
}
// vim: et sw=2 ts=2
