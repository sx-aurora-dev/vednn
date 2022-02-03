#include "vednnSoftmaxForward.h"
#include "vednn-def.h"
#include <stdio.h>
#include <float.h>
#include <math.h>

static VEDNN_SOFTMAXFWD_DECL(Fast);
static VEDNN_SOFTMAXFWD_DECL(Accurate);
static VEDNN_SOFTMAXFWD_DECL(Log);

static inline vednnError_t
vednnSoftmaxForward_wrapper(
    vednnSoftmaxForward_t pFunc,
    VEDNN_SOFTMAXFWD_ARGS )
{
#ifndef VEDNN_USE_OPENMP
  return pFunc(VEDNN_SOFTMAXFWD_ARGS_LIST);
#else
  if ( __vednn_omp_num_threads == 1 ) {
    return pFunc(VEDNN_SOFTMAXFWD_ARGS_LIST);
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
#endif
}

/* ----------------------------------------------------------------------- */

vednnError_t vednnSoftmaxForward(
    const vednnSoftmaxMode_t    mode,
    VEDNN_SOFTMAXFWD_ARGS)
{
#define OMPWRAP( IMPL ) WRAP_RET( vednnSoftmaxForward_##IMPL, \
    vednnSoftmaxForward_wrapper, VEDNN_SOFTMAXFWD_ARGS_LIST)
  switch(mode) {
    case VEDNN_SOFTMAX_FAST :
      OMPWRAP(Fast);
    case VEDNN_SOFTMAX_ACCURATE :
      OMPWRAP(Accurate);
    case VEDNN_SOFTMAX_LOG :
      OMPWRAP(Log);
  }
  fprintf(stderr, "VEDNN Error : vednnSoftmaxForward : Invalid Parameter !!\n") ;
  return VEDNN_ERROR_INVALID_PARAM ;
#undef OMPWRAP
}

static vednnError_t vednnSoftmaxForward_Fast ( VEDNN_SOFTMAXFWD_ARGS )
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

static vednnError_t vednnSoftmaxForward_Accurate ( VEDNN_SOFTMAXFWD_ARGS )
{

  const float *pIn  = (const float *) pDataIn ;
  float       *pOut = (float       *) pDataOut ;

  if( nClass <=128 && nBatch > nClass ) {
#pragma _NEC novector
    for(uint64_t b0=0; b0<nBatch; b0+=256) {
      const uint64_t blen = nBatch - b0 < 256 ? nBatch - b0 : 256 ;

      float max[256] ;
#pragma _NEC vreg(max)
      for(uint64_t b1=0; b1<blen; b1++) {
	max[b1] = -FLT_MAX ;
      }
      for(uint64_t i=0; i<nClass; i++) {
	for(uint64_t b1=0; b1<blen; b1++) {
	  if( max[b1] < pIn[(b0+b1)*nClass+i] ) max[b1] = pIn[(b0+b1)*nClass+i] ;
	}
      }

      float sum[256] ;
#pragma _NEC vreg(sum)
      for(uint64_t b1=0; b1<blen; b1++) {
	sum[b1] = 0.f ;
      }
      for(uint64_t i=0; i<nClass; i++) {
	for(uint64_t b1=0; b1<blen; b1++) {
	  sum[b1] += (pOut[(b0+b1)*nClass+i] = expf(pIn[(b0+b1)*nClass+i]-max[b1])) ;
	}
      }

      float inv_sum[256] ;
#pragma _NEC vreg(inv_sum)
      for(uint64_t b1=0; b1<blen; b1++) {
	inv_sum[b1] = 1.f / sum[b1] ;
      }
      for(uint64_t i=0; i<nClass; i++) {
	for(uint64_t b1=0; b1<blen; b1++) {
	  pOut[(b0+b1)*nClass+i] *= inv_sum[b1] ;
	}
      }
    }
  }
  else {
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
  }

  return VEDNN_SUCCESS ;
}


static vednnError_t vednnSoftmaxForward_Log ( VEDNN_SOFTMAXFWD_ARGS )
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
      sum += expf(pOut[i] = (pIn[i]-max)) ;
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
// vim: et sw=2 ts=2 ai
