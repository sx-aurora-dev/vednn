
#include <stdio.h>
#include <stdint.h>

#include "vednnActivationForward.h"

#ifdef VEDNN_USE_OPENMP
#include <stdint.h>
#include <omp.h>
extern int __vednn_omp_num_threads ;
#endif

static inline vednnError_t
vednnActivationForward_wrapper(
    vednnActivationForward_t	pFunc,
    const void 			*pDataIn,
    void 			*pDataOut,
    const uint64_t		nElements
)
{
#ifdef VEDNN_USE_OPENMP
  if ( __vednn_omp_num_threads == 1 ) {
    return pFunc(pDataIn, pDataOut, nElements) ;
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
#else
  return pFunc(pDataIn, pDataOut, nElements) ;
#endif
}

/* ----------------------------------------------------------------------- */

vednnError_t vednnActivationForward(
    const vednnActivationMode_t		mode,
    const void 				*pDataIn,
    void 				*pDataOut,
    const uint64_t			nElements
)
{

  switch(mode) {

  case VEDNN_ACTIVATION_RELU :
    return vednnActivationForward_wrapper(
	vednnActivationForward_Relu,
	pDataIn, pDataOut, nElements ) ;

  default :
    fprintf(stderr, "VEDNN Error : vednnActivationForward : Invalid Parameter !!\n") ;
    return VEDNN_ERROR_INVALID_PARAM ;
  }

}

