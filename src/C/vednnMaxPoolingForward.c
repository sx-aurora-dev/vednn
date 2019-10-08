#include "vednnMaxPoolingForward.h"
#include "vednn-def.h"
#include <stdint.h>

static inline vednnError_t
vednnMaxPoolingForward_wrapper(
    vednnMaxPoolForward_t pFunc,
    VEDNN_MAXPOOLINGFWD_ARGS )
{
#ifndef VEDNN_USE_OPENMP
  return pFunc(VEDNN_MAXPOOLINGFWD_ARGS_LIST);
#else
  if ( __vednn_omp_num_threads == 1 ) {
    return pFunc(VEDNN_MAXPOOLINGFWD_ARGS_LIST);
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
#endif
}

/* ----------------------------------------------------------------------- */

vednnError_t vednnMaxPoolingForward( VEDNN_MAXPOOLINGFWD_ARGS )
{
#define OMPWRAP( IMPL ) WRAP_RET(vednnMaxPoolingForward_##IMPL, \
    vednnMaxPoolingForward_wrapper, VEDNN_MAXPOOLINGFWD_ARGS_LIST)
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
        OMPWRAP(regular_ww2X_owU128_ialigned);
      else
        OMPWRAP(regular_owU128);
    }
    else
      OMPWRAP(regular);
  } else
    OMPWRAP(default);
#undef OMPWRAP
}
// vim: et sw=2 ts=2
