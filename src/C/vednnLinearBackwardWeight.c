#include "vednnLinearBackwardWeight.h"
#include "vednn-def.h"

static inline vednnError_t
vednnLinearBackwardWeight_wrapper(
    vednnLinearBackwardWeight_t pFunc,
    VEDNN_LINEARBKW_ARGS )
{
#ifndef VEDNN_USE_OPENMP
  return pFunc(VEDNN_LINEARBKW_ARGS_LIST);
#else
  if ( __vednn_omp_num_threads == 1 ) {
    return pFunc(VEDNN_LINEARBKW_ARGS_LIST, 0, inDim);
  }
  else {
    vednnError_t rc = VEDNN_SUCCESS ;
#pragma omp parallel reduction(|:rc)
    {
      int64_t nthreads = omp_get_num_threads() ;
      int64_t threadid = omp_get_thread_num() ;

      int64_t nInDim = inDim / nthreads ;
      int64_t remain = inDim % nthreads ;

      int64_t inDimBegin = nInDim * threadid + ( threadid < remain ? threadid : remain ) ;
      int64_t myInDim    = nInDim + ( threadid < remain ? 1 : 0 ) ;

      if( myInDim == 0 ) {
        rc |= VEDNN_SUCCESS ;
      }
      else  {
        rc |= pFunc(VEDNN_LINEARBKW_ARGS_LIST, inDimBegin, inDimBegin+myInDim) ;
      }
    }
    return rc ;
  }
#endif
}

/* ----------------------------------------------------------------------- */
vednnError_t vednnLinearBackwardWeight( VEDNN_LINEARBKW_ARGS )
{
#define OMPWRAP( IMPL ) WRAP_RET(vednnLinearBackwardWeight_##IMPL, \
    vednnLinearBackwardWeight_wrapper, VEDNN_LINEARBKW_ARGS_LIST )
  if( (outDim & 0x01) == 0 &&
      (((uint64_t)pDataGradWeight) & 0x07) == 0 && (((uint64_t)pDataGradOut) & 0x07) == 0 )
  {
    if( outDim <= 128 )
      OMPWRAP(o2XU128_woaligned);
    else
      OMPWRAP(o2X_woaligned);
  }
  else
    OMPWRAP(default);
#undef OMPWRAP
}
// vim: et sw=2 ts=2 ai




