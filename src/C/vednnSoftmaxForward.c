
#include <stdio.h>
#include <stdint.h>

#include <float.h>
#include <math.h>

#include "vednnSoftmaxForward.h"

#ifdef VEDNN_USE_OPENMP
#include <stdint.h>
#include <omp.h>
extern int __vednn_omp_num_threads ;
#endif

static inline vednnError_t
vednnSoftmaxForward_wrapper(
    vednnSoftmaxForward_t	pFunc,
    const void 			*pDataIn,
    void 			*pDataOut,
    const uint64_t		nBatch,
    const uint64_t		nClass
)
{
#ifdef VEDNN_USE_OPENMP
  if ( __vednn_omp_num_threads == 1 ) {
    return pFunc(pDataIn, pDataOut, nBatch, nClass) ;
  }
  else {
    vednnError_t rc = VEDNN_SUCCESS ;
#pragma omp parallel reduction(|:rc)
    {
      int64_t nthreads = omp_get_num_threads() ;
      int64_t threadid = omp_get_thread_num() ;

      int64_t eachNBatch = nBatch / nthreads ;
      int64_t remain     = nBatch % nthreads ;

      int64_t batchBegin = eachNBatch * threadid + ( threadid < remain ? threadid : remain ) ;
      int64_t myBatch    = eachNBatch + ( threadid < remain ? 1 : 0 ) ;

      if( myBatch == 0 ) {
	rc |= VEDNN_SUCCESS ;
      }
      else {
	float* _pDataIn  = ((float *)pDataIn)  + batchBegin * nClass;
	float* _pDataOut = ((float *)pDataOut) + batchBegin * nClass;

	rc |= pFunc((void*)_pDataIn, (void*) _pDataOut, myBatch, nClass) ;
      }
    }
    return rc ;
  }
#else
  return pFunc(pDataIn, pDataOut, nBatch, nClass) ;
#endif
}

/* ----------------------------------------------------------------------- */

vednnError_t vednnSoftmaxForward(
    const vednnSoftmaxMode_t		mode,
    const void 				*pDataIn,
    void 				*pDataOut,
    const uint64_t			nBatch,
    const uint64_t			nClass
)
{

  switch(mode) {
  case VEDNN_SOFTMAX_FAST :
    return vednnSoftmaxForward_wrapper(
	vednnSoftmaxForward_Fast,
	pDataIn, pDataOut, nBatch, nClass ) ;

  case VEDNN_SOFTMAX_ACCURATE :
    return vednnSoftmaxForward_wrapper(
	vednnSoftmaxForward_Accurate,
	pDataIn, pDataOut, nBatch, nClass ) ;

  case VEDNN_SOFTMAX_LOG :
    return vednnSoftmaxForward_wrapper(
	vednnSoftmaxForward_Log,
	pDataIn, pDataOut, nBatch, nClass ) ;

  default :
    fprintf(stderr, "VEDNN Error : vednnSoftmaxForward : Invalid Parameter !!\n") ;
    return VEDNN_ERROR_INVALID_PARAM ;
  }

}

static vednnError_t vednnSoftmaxForward_Fast (
    const void 			*pDataIn,
    void 			*pDataOut,
    const uint64_t		nBatch,
    const uint64_t		nClass
)
{

  const float *pIn  = (const float *) pDataIn ;
  float       *pOut = (float       *) pDataOut ;

  for(uint64_t b=0; b<nBatch; b++) {
    float sum = 0.f ;
    for(uint64_t i=0; i<nClass; i++) {
      sum += (pOut[i] = expf(pIn[i])) ;
    }

    float inv_sum = 1.f / sum ;
    for(uint64_t i=0; i<nClass; i++) {
      pOut[i] *= inv_sum ;
    }

    pIn  += nClass ;
    pOut += nClass ;
  }

  return VEDNN_SUCCESS ;
}

static vednnError_t vednnSoftmaxForward_Accurate (
    const void 			*pDataIn,
    void 			*pDataOut,
    const uint64_t		nBatch,
    const uint64_t		nClass
)
{

  const float *pIn  = (const float *) pDataIn ;
  float       *pOut = (float       *) pDataOut ;

  for(uint64_t b=0; b<nBatch; b++) {
    float max = -FLT_MAX ;
    for(uint64_t i=0; i<nClass; i++) {
      if( max < pIn[i] ) max = pIn[i] ;
    }

    float sum = 0.f ;
    for(uint64_t i=0; i<nClass; i++) {
      sum += (pOut[i] = expf(pIn[i]-max)) ;
    }

    float inv_sum = 1.f / sum ;
    for(uint64_t i=0; i<nClass; i++) {
      pOut[i] *= inv_sum ;
    }

    pIn  += nClass ;
    pOut += nClass ;
  }

  return VEDNN_SUCCESS ;
}


static vednnError_t vednnSoftmaxForward_Log (
    const void 			*pDataIn,
    void 			*pDataOut,
    const uint64_t		nBatch,
    const uint64_t		nClass
)
{

  const float *pIn  = (const float *) pDataIn ;
  float       *pOut = (float       *) pDataOut ;

  for(uint64_t b=0; b<nBatch; b++) {
    float max = -FLT_MAX ;
    for(uint64_t i=0; i<nClass; i++) {
      if( max < pIn[i] ) max = pIn[i] ;
    }

    float sum = 0.f ;
    for(uint64_t i=0; i<nClass; i++) {
      sum += (pOut[i] = (pIn[i]-max)) ;
    }

    float log_sum = logf(sum) ;
    for(uint64_t i=0; i<nClass; i++) {
      pOut[i] -= log_sum ;
    }

    pIn  += nClass ;
    pOut += nClass ;
  }

  return VEDNN_SUCCESS ;
}
