#include "vednnLinearBackwardData.h"
#include "vednn-def.h"

static inline vednnError_t
vednnLinearBackwardData_wrapper(
    vednnLinearBackwardData_t pFunc,
    VEDNN_LINEARBKD_ARGS
)
{
#ifndef VEDNN_USE_OPENMP
  return pFunc(VEDNN_LINEARBKD_ARGS_LIST);
#else
  if ( __vednn_omp_num_threads == 1 ) {
    return pFunc(VEDNN_LINEARBKD_ARGS_LIST);
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
#endif
}

/* ----------------------------------------------------------------------- */
vednnError_t vednnLinearBackwardData(
    const uint64_t      inDim,
    const uint64_t      outDim,
    const uint64_t      nBatch,
    const void         *pDataGradOut,
    const void         *pDataWeight,
    void         *pDataGradIn
)
{
#define OMPWRAP( IMPL ) WRAP_RET(vednnLinearBackwardData_##IMPL, \
    vednnLinearBackwardData_wrapper, VEDNN_LINEARBKD_ARGS_LIST)
  if( outDim<=128 && inDim >=256 )
  {
    if( ((outDim&0x1))==0 && ((((uint64_t)pDataWeight)&0x7)==0) )
      OMPWRAP(o2XU128_waligned);
    else
      OMPWRAP(oU128);
  }
  else if( outDim <= 256 )
    OMPWRAP(oU256);
  else if( ((outDim & 0x1) == 0)
      && ((((uint64_t)pDataWeight)&0x7)==0)
      && ((((uint64_t)pDataGradOut)&0x7)==0) )
    OMPWRAP(o2X_woaligned);
  else
    OMPWRAP(default);
#undef OMPWRAP
}
// vim: et sw=2 ts=2
