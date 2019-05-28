
#include <stdint.h>
#include "vednnLinearBackwardData.h"

#ifdef VEDNN_USE_OPENMP
#include <omp.h>
extern int __vednn_omp_num_threads ;
#endif


static inline vednnError_t
vednnLinearBackwardData_wrapper(
    vednnLinearBackwardData_t		pFunc,
    const uint64_t			inDim,
    const uint64_t			outDim,
    const uint64_t			nBatch,
    const void 				*pDataGradOut,
    const void 				*pDataWeight,
    void 				*pDataGradIn
)
{
#ifdef VEDNN_USE_OPENMP
  if ( __vednn_omp_num_threads == 1 ) {
    return pFunc(inDim, outDim, nBatch, pDataGradOut, pDataWeight, pDataGradIn ) ;
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
	float* _pDataGradOut = ((float *)pDataGradOut) + batchBegin * outDim ;
	float* _pDataGradIn  = ((float *)pDataGradIn) + batchBegin * inDim ;

	rc |= pFunc(inDim, outDim, myBatch, _pDataGradOut, pDataWeight, _pDataGradIn ) ;
      }
    }
    return rc ;
  }
#else
  return pFunc(inDim, outDim, nBatch, pDataGradOut, pDataWeight, pDataGradIn ) ;
#endif
}

/* ----------------------------------------------------------------------- */
vednnError_t vednnLinearBackwardData(
    const uint64_t			inDim,
    const uint64_t			outDim,
    const uint64_t			nBatch,
    const void 				*pDataGradOut,
    const void 				*pDataWeight,
    void 				*pDataGradIn
)
{
  // [todo] add variations
  if( outDim<=128 && inDim >=256 )
  {
    if( ((outDim&0x1))==0 && ((((uint64_t)pDataWeight)&0x7)==0) )
    {
      return vednnLinearBackwardData_wrapper(
	  vednnLinearBackwardData_o2XU128_waligned,
	  inDim, outDim, nBatch,
	  pDataGradOut, pDataWeight, pDataGradIn ) ;
    }
    else {
      return vednnLinearBackwardData_wrapper(
	  vednnLinearBackwardData_oU128,
	  inDim, outDim, nBatch,
	  pDataGradOut, pDataWeight, pDataGradIn ) ;
    }
  }
  else if( outDim <= 256 )
  {
    return vednnLinearBackwardData_wrapper(
	vednnLinearBackwardData_oU256,
	inDim, outDim, nBatch,
	pDataGradOut, pDataWeight, pDataGradIn ) ;
  }
  else if( ((outDim & 0x1) == 0)
	  && ((((uint64_t)pDataWeight)&0x7)==0)
	  && ((((uint64_t)pDataGradOut)&0x7)==0) )

  {
    return vednnLinearBackwardData_wrapper(
	vednnLinearBackwardData_o2X_woaligned,
	inDim, outDim, nBatch,
	pDataGradOut, pDataWeight, pDataGradIn ) ;
  }
  else
  {
    return vednnLinearBackwardData_wrapper(
	vednnLinearBackwardData_default,
	inDim, outDim, nBatch,
	pDataGradOut, pDataWeight, pDataGradIn ) ;
  }

}

