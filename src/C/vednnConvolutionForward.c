
#include "vednnConvolutionForward.h"
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>

#ifdef VEDNN_USE_OPENMP
#include <omp.h>
extern int __vednn_omp_num_threads ;
#endif

static inline vednnError_t
vednnConvolutionForward_wrapper(
    vednnConvForward_t			pFunc,
    const vednnTensorParam_t 		*pParamIn,
    const void 				*pDataIn,
    const vednnFilterParam_t		*pParamKernel,
    const void 				*pDataKernel,
    const vednnBiasParam_t 		*pParamBias,
    const void 				*pDataBias,
    const vednnConvolutionParam_t	*pParamConv,
    const vednnTensorParam_t 		*pParamOut,
    void 				*pDataOut
)
{
#ifdef VEDNN_USE_OPENMP
  if ( __vednn_omp_num_threads == 1 ) {
    return pFunc(pParamIn, pDataIn, pParamKernel, pDataKernel, pParamBias, pDataBias, pParamConv, pParamOut, pDataOut) ;
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

	rc |= pFunc(&_pParamIn, (void*)_pDataIn, pParamKernel, pDataKernel,
	            pParamBias, pDataBias,
		    pParamConv, &_pParamOut, (void*) _pDataOut) ;
      }
    }
    return rc ;
  }
#else
  return pFunc(pParamIn, pDataIn, pParamKernel, pDataKernel,
               pParamBias, pDataBias,
               pParamConv, pParamOut, pDataOut) ;
#endif
}

/* ----------------------------------------------------------------------- */
static inline
vednnError_t vednnConvolutionForwardBody(
    const vednnTensorParam_t 		*pParamIn,
    const void 				*pDataIn,
    const vednnFilterParam_t		*pParamKernel,
    const void 				*pDataKernel,
    const vednnBiasParam_t 		*pParamBias,
    const void 				*pDataBias,
    const vednnTensorParam_t 		*pParamOut,
    void 				*pDataOut,
    const vednnConvolutionParam_t	*pParamConv,
    vednnConvolutionAlgorithm_t 	algo
)
{
  switch( pParamKernel->layout ) {
  case VEDNN_FILTER_LAYOUT_NCHW :
    break ;
  case VEDNN_FILTER_LAYOUT_HWCN :
    if( pParamConv->group > 1 ) {
      fprintf(stderr, "[VEDNN ERROR] VEDNN does not support grouped convolution with filter_hwcn\n") ;
      return VEDNN_ERROR_INVALID_PARAM ;
    }
    break ;
  default :
    fprintf(stderr, "[VEDNN ERROR] Unknown Filter Layout %d\n", pParamKernel->layout) ;
    return VEDNN_ERROR_INVALID_PARAM ;
  }

  if (algo == VEDNN_CONV_ALGORITHM_DIRECT)
  {
    // [todo] add variations
    if ( pParamOut->height * pParamOut->width <= 16  ) {
	return vednnConvolutionForward_wrapper(
	    vednnConvolutionForward_direct_vecC,
	    pParamIn, pDataIn, pParamKernel, pDataKernel,
	    pParamBias, pDataBias,
	    pParamConv, pParamOut, pDataOut);
    }
    else if (pParamConv->strideHeight == 1 && pParamConv->strideWidth == 1
	&& pParamConv->dilationHeight == 1 && pParamConv->dilationWidth == 1
	&& pParamIn->height == pParamOut->height
	&& pParamIn->width == pParamOut->width )
    {
      if (pParamKernel->width == 1 && pParamKernel->height == 1)
      {
	return vednnConvolutionForward_wrapper(
	    vednnConvolutionForward_direct_dil1_str1_pad0_ker1,
	    pParamIn, pDataIn, pParamKernel, pDataKernel,
	    pParamBias, pDataBias,
	    pParamConv, pParamOut, pDataOut);
      }
      else if (pParamKernel->height == 3 && pParamKernel->width == 3)
      {
	if (pParamIn->channel == pParamConv->group) // aka inputChannelGroup==1
	{
	  if (pParamOut->width <= 128)
	  {
	    return vednnConvolutionForward_wrapper(
		vednnConvolutionForward_direct_dil1_str1_padsame_ker3_c1_owU128,
		pParamIn, pDataIn, pParamKernel, pDataKernel,
		pParamBias, pDataBias,
		pParamConv, pParamOut, pDataOut );
	  }
	  else {
	    return vednnConvolutionForward_wrapper(
		vednnConvolutionForward_direct_dil1_str1_padsame_ker3_c1,
		pParamIn, pDataIn, pParamKernel, pDataKernel,
		pParamBias, pDataBias,
		pParamConv, pParamOut, pDataOut );
	  }
	}
	else
	{
	  return vednnConvolutionForward_wrapper(
	      vednnConvolutionForward_direct_dil1_str1_padsame_ker3,
	      pParamIn, pDataIn, pParamKernel, pDataKernel,
	      pParamBias, pDataBias,
	      pParamConv, pParamOut, pDataOut );
	}
      }
      else if (pParamKernel->height == 5 && pParamKernel->width == 5)
      {
	if( pParamOut->width <= 128 ) {
	  return vednnConvolutionForward_wrapper(
	      vednnConvolutionForward_direct_dil1_str1_padsame_ker5_owU128,
	      pParamIn, pDataIn, pParamKernel, pDataKernel,
	      pParamBias, pDataBias,
	      pParamConv, pParamOut, pDataOut );
	}
	else {
	  return vednnConvolutionForward_wrapper(
	      vednnConvolutionForward_direct_dil1_str1_padsame_ker5,
	      pParamIn, pDataIn, pParamKernel, pDataKernel,
	      pParamBias, pDataBias,
	      pParamConv, pParamOut, pDataOut );
	}
      }
      else if (pParamKernel->height == 2 && pParamKernel->width == 2)
      {
	return vednnConvolutionForward_wrapper(
	    vednnConvolutionForward_direct_dil1_str1_padsame_ker2,
	    pParamIn, pDataIn, pParamKernel, pDataKernel,
	    pParamBias, pDataBias,
	    pParamConv, pParamOut, pDataOut );
      }
      else
      {
	return vednnConvolutionForward_wrapper(
	    vednnConvolutionForward_direct_dil1_str1_padsame,
	    pParamIn, pDataIn, pParamKernel, pDataKernel,
	    pParamBias, pDataBias,
	    pParamConv, pParamOut, pDataOut );
      }
    }
    else if ( pParamConv->dilationHeight == 1 && pParamConv->dilationWidth == 1
	&& pParamConv->padHeight == 0  && pParamConv->padWidth == 0
	&& pParamOut->height == (pParamIn->height - pParamKernel->height) / pParamConv->strideHeight + 1
	&& pParamOut->width == (pParamIn->width - pParamKernel->width) / pParamConv->strideWidth + 1 )
    {
      if (pParamConv->strideHeight == 1 && pParamConv->strideWidth == 1 )
      {
	if ( pParamKernel->height == 3 && pParamKernel->width == 3
	    && (pParamIn->width <= 256)
	    && (pParamIn->width & 0x1) == 0  && (((uint64_t)pDataIn) & 0x7) == 0
	    && (pParamOut->width & 0x1) == 0 && (((uint64_t)pDataOut) & 0x7) == 0 )
	{
	  return vednnConvolutionForward_wrapper (
		  vednnConvolutionForward_direct_dil1_str1_pad0_ker3_iw2XU256_ow2X_ioaligned,
		  pParamIn, pDataIn, pParamKernel, pDataKernel,
		  pParamBias, pDataBias,
		  pParamConv, pParamOut, pDataOut ) ;
	}
	else if (pParamOut->width <= 128)
	{
	  return vednnConvolutionForward_wrapper (
	      vednnConvolutionForward_direct_dil1_str1_pad0_owU128,
	      pParamIn, pDataIn, pParamKernel, pDataKernel,
	      pParamBias, pDataBias,
	      pParamConv, pParamOut, pDataOut );
	}
	else
	{
	  return vednnConvolutionForward_wrapper(
	      vednnConvolutionForward_direct_dil1_str1_pad0,
	      pParamIn, pDataIn, pParamKernel, pDataKernel,
	      pParamBias, pDataBias,
	      pParamConv, pParamOut, pDataOut );
	}
      }
      else {
	if( pParamKernel->width == 1 && pParamKernel->height == 1 )
	{
	  if (pParamOut->width <= 128) {
	    return vednnConvolutionForward_wrapper(
		vednnConvolutionForward_direct_dil1_pad0_owU128_ker1,
		pParamIn, pDataIn, pParamKernel, pDataKernel,
		pParamBias, pDataBias,
		pParamConv, pParamOut, pDataOut );
	  }
	  else {
	    return vednnConvolutionForward_wrapper(
		vednnConvolutionForward_direct_dil1_pad0_ker1,
		pParamIn, pDataIn, pParamKernel, pDataKernel,
		pParamBias, pDataBias,
		pParamConv, pParamOut, pDataOut );
	  }
	}
	else {
	  if (pParamOut->width <= 128) {
	    return vednnConvolutionForward_wrapper(
		vednnConvolutionForward_direct_dil1_pad0_owU128,
		pParamIn, pDataIn, pParamKernel, pDataKernel,
		pParamBias, pDataBias,
		pParamConv, pParamOut, pDataOut );
	  }
	  else {
	    return vednnConvolutionForward_wrapper(
		vednnConvolutionForward_direct_dil1_pad0,
		pParamIn, pDataIn, pParamKernel, pDataKernel,
		pParamBias, pDataBias,
		pParamConv, pParamOut, pDataOut );
	  }
	}
      }
    }
    else {
      if (pParamOut->width <= 128)
      {
	return vednnConvolutionForward_wrapper (
	    vednnConvolutionForward_direct_owU128,
	    pParamIn, pDataIn, pParamKernel, pDataKernel,
	    pParamBias, pDataBias,
	    pParamConv, pParamOut, pDataOut );
      }
      else {
	return vednnConvolutionForward_wrapper(
	    vednnConvolutionForward_direct_default,
	    pParamIn, pDataIn, pParamKernel, pDataKernel,
	    pParamBias, pDataBias,
	    pParamConv, pParamOut, pDataOut );
      }
    }
  }
  else {
    return VEDNN_ERROR_INVALID_PARAM ;
  }
}

/* ----------------------------------------------------------------------- */
vednnError_t vednnConvolutionForwardAddBias(
    const vednnTensorParam_t 		*pParamIn,
    const void 				*pDataIn,
    const vednnFilterParam_t		*pParamKernel,
    const void 				*pDataKernel,
    const vednnBiasParam_t 		*pParamBias,
    const void 				*pDataBias,
    const vednnTensorParam_t 		*pParamOut,
    void 				*pDataOut,
    const vednnConvolutionParam_t	*pParamConv,
    vednnConvolutionAlgorithm_t 	algo
)
{
  return vednnConvolutionForwardBody(pParamIn, pDataIn,
            pParamKernel, pDataKernel, pParamBias, pDataBias,
	    pParamOut, pDataOut, pParamConv, algo );
}

vednnError_t vednnConvolutionForward(
    const vednnTensorParam_t 		*pParamIn,
    const void 				*pDataIn,
    const vednnFilterParam_t		*pParamKernel,
    const void 				*pDataKernel,
    const vednnTensorParam_t 		*pParamOut,
    void 				*pDataOut,
    const vednnConvolutionParam_t	*pParamConv,
    vednnConvolutionAlgorithm_t 	algo
)
{
  return vednnConvolutionForwardBody(pParamIn, pDataIn,
            pParamKernel, pDataKernel, NULL, NULL,
	    pParamOut, pDataOut, pParamConv, algo );
}
