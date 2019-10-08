#include "vednnMaxPoolingBackward.h"
#include "vednn-def.h"
#include <stdint.h>

static inline vednnError_t
vednnMaxPoolingBackward_wrapper(
    vednnMaxPoolBackward_t pFunc,
    VEDNN_MAXPOOLINGBKW_ARGS )
{
#ifndef VEDNN_USE_OPENMP
  return pFunc(VEDNN_MAXPOOLINGBKW_ARGS_LIST);
#else
  if ( __vednn_omp_num_threads == 1 ) {
    return pFunc(VEDNN_MAXPOOLINGBKW_ARGS_LIST);
  }
  else {
    vednnError_t rc = VEDNN_SUCCESS ;
#pragma omp parallel reduction(|:rc)
    {
      int64_t nthreads = omp_get_num_threads() ;
      int64_t threadid = omp_get_thread_num() ;

      int64_t allBatch = pParamGradOut->batch ;

      int64_t nBatch = allBatch / nthreads ;
      int64_t remain = allBatch % nthreads ;

      int64_t batchBegin = nBatch * threadid + ( threadid < remain ? threadid : remain ) ;
      int64_t myBatch = nBatch + ( threadid < remain ? 1 : 0 ) ;

      if( myBatch == 0 ) {
        rc |= VEDNN_SUCCESS ;
      }
      else {
        vednnTensorParam_t _pParamGradOut = *pParamGradOut ; _pParamGradOut.batch = myBatch ;
        vednnTensorParam_t _pParamOut = *pParamOut ; _pParamOut.batch = myBatch ;
        vednnTensorParam_t _pParamIn  = *pParamIn  ; _pParamIn.batch = myBatch ;
        vednnTensorParam_t _pParamGradIn  = *pParamGradIn  ; _pParamGradIn.batch = myBatch ;

        float* _pDataGradOut = ((float *)pDataGradOut) + batchBegin * pParamGradOut->channel * pParamGradOut->height * pParamGradOut->width ;
        float* _pDataOut = ((float *)pDataOut) + batchBegin * pParamOut->channel * pParamOut->height * pParamOut->width ;
        float* _pDataIn  = ((float *)pDataIn) + batchBegin * pParamIn->channel * pParamIn->height * pParamIn->width ;
        float* _pDataGradIn  = ((float *)pDataGradIn) + batchBegin * pParamGradIn->channel * pParamGradIn->height * pParamGradIn->width ;


        rc |= pFunc(&_pParamGradOut, (void*) _pDataGradOut,
            &_pParamOut,     (void*) _pDataOut,
            &_pParamIn,      (void*) _pDataIn,
            &_pParamGradIn,  (void*) _pDataGradIn,
            pParamPool ) ;
      }
    }
    return rc ;
  }
#endif
}

/* ----------------------------------------------------------------------- */
vednnError_t vednnMaxPoolingBackward(
    VEDNN_MAXPOOLINGBKW_ARGS )
{
#define OMPWRAP( IMPL ) WRAP_RET(vednnMaxPoolingBackward_##IMPL, \
    vednnMaxPoolingBackward_wrapper, VEDNN_MAXPOOLINGBKW_ARGS_LIST)
  if( pParamPool->padHeight == 0 && pParamPool->padWidth == 0
      && pParamPool->strideHeight == pParamPool->windowHeight
      && pParamPool->strideWidth == pParamPool->windowWidth
      && pParamOut->height*pParamPool->strideHeight <= pParamIn->height
      && pParamOut->width*pParamPool->strideWidth == pParamIn->width )
  {

    if( pParamOut->width <= 128 )
    {
      if( (pParamPool->windowWidth & 0x01) == 0
          && (((uint64_t)pDataIn) & 0x07) == 0
          && (((uint64_t)pDataGradIn) & 0x07) == 0 )
        OMPWRAP(regular_ww2X_owU128_ialigned);
      else
        OMPWRAP(regular_owU128);
    }
    else
      OMPWRAP(regular);
  }
  else
  OMPWRAP(default);
}
// vim: et sw=2 ts=2
