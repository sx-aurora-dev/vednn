
#include <stdint.h>
#include "vednnLinearForward.h"

#ifdef VEDNN_USE_OPENMP
#include <omp.h>
extern int __vednn_omp_num_threads ;
#endif

static inline vednnError_t
vednnLinearForward_wrapper(
    vednnLinearForward_t		pFunc,
    const uint64_t			inDim,
    const uint64_t			outDim,
    const uint64_t			nBatch,
    const void 				*pDataIn,
    const void 				*pDataWeight,
    void 				*pDataOut
)
{
#ifdef VEDNN_USE_OPENMP
  if ( __vednn_omp_num_threads == 1 ) {
    return pFunc(inDim, outDim, nBatch, pDataIn, pDataWeight, pDataOut ) ;
  }
  else {
    vednnError_t rc = VEDNN_SUCCESS ;
#pragma omp parallel reduction(|:rc)
    {
      uint64_t nthreads = omp_get_num_threads() ;
      uint64_t threadid = omp_get_thread_num() ;

      uint64_t nBatchEach = nBatch / nthreads ;
      uint64_t remain     = nBatch % nthreads ;

      uint64_t batchBegin = nBatchEach * threadid + ( threadid < remain ? threadid : remain ) ;
      uint64_t myBatch    = nBatchEach + ( threadid < remain ? 1 : 0 ) ;

      if( myBatch == 0 ) {
	rc |= VEDNN_SUCCESS ;
      }
      else {
	float* _pDataIn  = ((float *)pDataIn) + batchBegin * inDim ;
	float* _pDataOut = ((float *)pDataOut) + batchBegin * outDim ;

	rc |= pFunc(inDim, outDim, myBatch, _pDataIn, pDataWeight, _pDataOut ) ;
      }
    }
    return rc ;
  }
#else
  return pFunc(inDim, outDim, nBatch, pDataIn, pDataWeight, pDataOut ) ;
#endif
}

/* ----------------------------------------------------------------------- */
vednnError_t vednnLinearForward(
    const uint64_t			inDim,
    const uint64_t			outDim,
    const uint64_t			nBatch,
    const void 				*pDataIn,
    const void 				*pDataWeight,
    void 				*pDataOut
)
{
  // [todo] add variations

  if( outDim <= 32 )
  {
    return vednnLinearForward_wrapper(
	  vednnLinearForward_oU32,
	  inDim, outDim, nBatch,
	  pDataIn, pDataWeight, pDataOut ) ;
  }
  else
  {
    if( (outDim & 0x01) == 0 &&
	(((uint64_t)pDataWeight) & 0x07) == 0 && (((uint64_t)pDataOut) & 0x07) == 0 )
    {
	return vednnLinearForward_wrapper(
	    vednnLinearForward_o2X_woaligned,
	    inDim, outDim, nBatch,
	    pDataIn, pDataWeight, pDataOut ) ;
    }
    else {
	return vednnLinearForward_wrapper(
	    vednnLinearForward_default,
	    inDim, outDim, nBatch,
	    pDataIn, pDataWeight, pDataOut ) ;
    }
  }
}

