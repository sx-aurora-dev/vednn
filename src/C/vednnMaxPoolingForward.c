#include <stdint.h>

#include "vednnMaxPoolingForward.h"

#ifdef VEDNN_USE_OPENMP
#include <stdint.h>
#include <omp.h>
extern int __vednn_omp_num_threads ;
#endif

static inline vednnError_t
vednnMaxPoolingForward_wrapper(
    vednnMaxPoolForward_t		pFunc,
    const vednnTensorParam_t 		*pParamIn,
    const void 				*pDataIn,
    const vednnTensorParam_t 		*pParamOut,
    void 				*pDataOut,
    const vednnPoolingParam_t		*pParamPool
)
{
#ifdef VEDNN_USE_OPENMP
  if ( __vednn_omp_num_threads == 1 ) {
    return pFunc(pParamIn,  pDataIn,
  	       pParamOut, pDataOut,
  	       pParamPool ) ;
  }
  else {
    vednnError_t rc = VEDNN_SUCCESS ;
#pragma omp parallel reduction(|:rc)
    {
      int64_t nthreads = omp_get_num_threads() ;
      int64_t threadid = omp_get_thread_num() ;

      int64_t allBatch = pParamIn->batch ;

      int64_t nBatch = allBatch / nthreads ;
      int64_t remain = allBatch % nthreads ;

      int64_t batchBegin = nBatch * threadid + ( threadid < remain ? threadid : remain ) ;
      int64_t myBatch = nBatch + ( threadid < remain ? 1 : 0 ) ;

      if( myBatch == 0 ) {
	rc |= VEDNN_SUCCESS ;
      }
      else {
	vednnTensorParam_t _pParamIn  = *pParamIn  ; _pParamIn.batch = myBatch ;
	vednnTensorParam_t _pParamOut = *pParamOut ; _pParamOut.batch = myBatch ;
	float* _pDataIn  = ((float *)pDataIn) + batchBegin * pParamIn->channel * pParamIn->height * pParamIn->width ;
	float* _pDataOut = ((float *)pDataOut) + batchBegin * pParamOut->channel * pParamOut->height * pParamOut->width ;

	rc |= pFunc(&_pParamIn, (void*)_pDataIn, &_pParamOut, (void*) _pDataOut, pParamPool) ;
      }
    }
    return rc ;
  }
#else
  return pFunc(pParamIn,  pDataIn,
	       pParamOut, pDataOut,
	       pParamPool ) ;
#endif
}

/* ----------------------------------------------------------------------- */

vednnError_t vednnMaxPoolingForward(
    const vednnTensorParam_t 		*pParamIn,
    const void 				*pDataIn,
    const vednnTensorParam_t 		*pParamOut,
    void 				*pDataOut,
    const vednnPoolingParam_t		*pParamPool
)
{

  // [todo] add variations
  if( pParamPool->padHeight == 0 && pParamPool->padWidth == 0
      && pParamPool->strideHeight == pParamPool->windowHeight
      && pParamPool->strideWidth == pParamPool->windowWidth
      && pParamOut->height*pParamPool->strideHeight <= pParamIn->height
      && pParamOut->width*pParamPool->strideWidth == pParamIn->width )
  {

    if( pParamOut->width <= 128   )
    {
      if( (pParamPool->windowWidth & 0x01) == 0
	  && (((uint64_t)pDataIn) & 0x07) == 0 )
      {
	return vednnMaxPoolingForward_wrapper(
	    vednnMaxPoolingForward_regular_ww2X_owU128_ialigned,
	    pParamIn,  pDataIn,
	    pParamOut, pDataOut,
	    pParamPool ) ;
      }
      else {
	return vednnMaxPoolingForward_wrapper(
	    vednnMaxPoolingForward_regular_owU128,
	    pParamIn,  pDataIn,
	    pParamOut, pDataOut,
	    pParamPool ) ;
      }
    }
    else
    {
      return vednnMaxPoolingForward_wrapper(
	  vednnMaxPoolingForward_regular,
	  pParamIn,  pDataIn,
	  pParamOut, pDataOut,
	  pParamPool ) ;
    }
  }
  else
  {
    return vednnMaxPoolingForward_wrapper(
	vednnMaxPoolingForward_default,
	pParamIn,  pDataIn,
	pParamOut, pDataOut,
	pParamPool ) ;
  }
}
