#include <stdio.h>
#include <stdint.h>

#include "vednnActivationBackward.h"

#ifdef VEDNN_USE_OPENMP
#include <stdint.h>
#include <omp.h>
extern int __vednn_omp_num_threads ;
#endif

static inline vednnError_t
vednnActivationBackward_wrapper(
    vednnActivationBackward_t	pFunc,
    const void 			*pDataGradOut,
    const void 			*pDataIn,
    void 			*pDataGradIn,
    const uint64_t		nElements
)
{
#ifdef VEDNN_USE_OPENMP
  if ( __vednn_omp_num_threads == 1 ) {
    return pFunc(pDataGradOut, pDataIn, pDataGradIn, nElements) ;
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
#else
  return pFunc(pDataGradOut, pDataIn, pDataGradIn, nElements) ;
#endif
}

/* ----------------------------------------------------------------------- */

vednnError_t vednnActivationBackward(
    const vednnActivationMode_t		mode,
    const void 				*pDataGradOut,
    const void 				*pDataIn,
    void 				*pDataGradIn,
    const uint64_t			nElements
)
{

  switch(mode) {

  case VEDNN_ACTIVATION_RELU :
    return vednnActivationBackward_wrapper(
	vednnActivationBackward_Relu,
	pDataGradOut, pDataIn, pDataGradIn, nElements ) ;

  default :
    fprintf(stderr, "VEDNN Error : vednnActivationBackward : Invalid Parameter !!\n") ;
    return VEDNN_ERROR_INVALID_PARAM ;
  }

}
