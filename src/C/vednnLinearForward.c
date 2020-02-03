#include "vednnLinearForward.h"
#include "vednn-def.h"

static inline vednnError_t
vednnLinearForward_wrapper(
    vednnLinearForward_t pFunc,
    VEDNN_LINEARFWD_ARGS )
{
#ifndef VEDNN_USE_OPENMP
  return pFunc(VEDNN_LINEARFWD_ARGS_LIST);
#else
  if ( __vednn_omp_num_threads == 1 ) {
    return pFunc(VEDNN_LINEARFWD_ARGS_LIST);
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
#endif
}

/* ----------------------------------------------------------------------- */
vednnError_t vednnLinearForward( VEDNN_LINEARFWD_ARGS )
{
#define OMPWRAP( IMPL ) WRAP_RET(vednnLinearForward_##IMPL, \
    vednnLinearForward_wrapper, VEDNN_LINEARFWD_ARGS_LIST)
  if( outDim <= 32 )
    OMPWRAP(oU32);
  else
  {
    if( (outDim & 0x01) == 0 &&
        (((uint64_t)pDataWeight) & 0x07) == 0 && (((uint64_t)pDataOut) & 0x07) == 0 )
      OMPWRAP(o2X_woaligned);
    else
      OMPWRAP(default);
  }
#undef OMPWRAP
}
// vim: et sw=2 ts=2
