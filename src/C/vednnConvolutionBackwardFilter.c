
#include "vednnConvolutionBackwardFilter.h"
#include <stdint.h>
#include <stdio.h>

#ifdef VEDNN_USE_OPENMP
#include <omp.h>
extern int __vednn_omp_num_threads ;
#endif

static inline vednnError_t
vednnConvolutionBackwardFilter_wrapper(
    vednnConvBackwardFilter_t		pFunc,
    const vednnTensorParam_t * restrict 	pParamIn,
    const void * restrict 			pDataIn,
    const vednnTensorParam_t * restrict 	pParamGradOut,
    const void * restrict 			pDataGradOut,
    const vednnConvolutionParam_t * restrict 	pParamConv,
    const vednnFilterParam_t * restrict 	pParamGradKernel,
    void * restrict 				pDataGradKernel
)
{
#ifdef VEDNN_USE_OPENMP
  if ( __vednn_omp_num_threads == 1 ) {
    int64_t gOutChannel = pParamGradOut->channel;
    int64_t group       = pParamConv->group;
    int64_t gOutChannelGroup = gOutChannel  / group;

    return pFunc(pParamIn, pDataIn, pParamGradOut, pDataGradOut,
  	  pParamConv, pParamGradKernel, pDataGradKernel,
	  0, gOutChannelGroup);
  }
  else {
    vednnError_t rc = VEDNN_SUCCESS ;
#pragma omp parallel reduction(|:rc)
    {
      int64_t nthreads = omp_get_num_threads() ;
      int64_t threadid = omp_get_thread_num() ;

      int64_t gOutChannel = pParamGradOut->channel;
      int64_t group       = pParamConv->group;
      int64_t gOutChannelGroup = gOutChannel  / group;

      int64_t nOChannlel = gOutChannelGroup / nthreads ;
      int64_t remain     = gOutChannelGroup % nthreads ;

      int64_t beginOChannel = nOChannlel * threadid + ( threadid < remain ? threadid : remain ) ;
      int64_t myOChannel    = nOChannlel + ( threadid < remain ? 1 : 0 ) ;

      if( myOChannel == 0 ) {
	rc |= VEDNN_SUCCESS ;
      }
      else  {

	rc |= pFunc(pParamIn, pDataIn, pParamGradOut, pDataGradOut,
		    pParamConv, pParamGradKernel, pDataGradKernel,
		    beginOChannel, myOChannel );
      }
    }
    return rc ;
  }
#else
  return pFunc(pParamIn, pDataIn, pParamGradOut, pDataGradOut,
	  pParamConv, pParamGradKernel, pDataGradKernel );
#endif
}

/* ----------------------------------------------------------------------- */
vednnError_t vednnConvolutionBackwardFilter(
    const vednnTensorParam_t 		*pParamIn,
    const void 				*pDataIn,
    const vednnTensorParam_t 		*pParamGradOut,
    const void 				*pDataGradOut,
    const vednnFilterParam_t		*pParamGradKernel,
    void 				*pDataGradKernel,
    const vednnConvolutionParam_t	*pParamConv,
    vednnConvolutionAlgorithm_t 	algo
)
{
  switch( pParamGradKernel->layout ) {
  case VEDNN_FILTER_LAYOUT_NCHW :
    break ;
  case VEDNN_FILTER_LAYOUT_HWCN :
#if 0
    if( pParamConv->group > 1 ) {
      fprintf(stderr, "[VEDNN ERROR] VEDNN does not support grouped convolution with filter_hwcn\n") ;
      return VEDNN_ERROR_INVALID_PARAM ;
    }
#else
    fprintf(stderr, "[VEDNN ERROR] Sorry. Now implementing ConvBackwardFilter(filter_hwcn)\n") ;
    return VEDNN_ERROR_INVALID_PARAM ;
#endif
    break ;
  default :
    fprintf(stderr, "[VEDNN ERROR] Unknown Filter Layout %d\n", pParamGradKernel->layout) ;
    return VEDNN_ERROR_INVALID_PARAM ;
  }

  if (algo == VEDNN_CONV_ALGORITHM_DIRECT)
  {
    // [todo] add variations
    if ( pParamGradOut->height * pParamGradOut->width <= 16 ||
	( pParamGradOut->height * pParamGradOut->width < 64
	  && pParamGradOut->height * pParamGradOut->width < pParamIn->channel)  ) {
      return vednnConvolutionBackwardFilter_wrapper(
	  vednnConvolutionBackwardFilter_direct_vecC,
	  pParamIn, pDataIn, pParamGradOut, pDataGradOut,
	  pParamConv, pParamGradKernel, pDataGradKernel );
    }
    else if (pParamConv->strideHeight == 1 && pParamConv->strideWidth == 1
	&& pParamConv->dilationHeight == 1 && pParamConv->dilationWidth == 1
	&& pParamIn->height == pParamGradOut->height
	&& pParamIn->width == pParamGradOut->width )
    {
      if (pParamGradKernel->height == 1 && pParamGradKernel->width == 1)
      {
	return vednnConvolutionBackwardFilter_wrapper(
	    vednnConvolutionBackwardFilter_direct_dil1_str1_padsame_ker1,
	    pParamIn, pDataIn, pParamGradOut, pDataGradOut,
	    pParamConv, pParamGradKernel, pDataGradKernel );
      }
      else if (pParamGradKernel->height == 3 && pParamGradKernel->width == 3)
      {
	if( pParamGradOut->width * pParamGradOut->height <= 256)
	{
	  return vednnConvolutionBackwardFilter_wrapper(
	      vednnConvolutionBackwardFilter_direct_dil1_str1_padsame_ker3_ohwU256,
	      pParamIn, pDataIn, pParamGradOut, pDataGradOut,
	      pParamConv, pParamGradKernel, pDataGradKernel );
	}
	else if( pParamGradOut->width <= 128)
	{
	  return vednnConvolutionBackwardFilter_wrapper(
	      vednnConvolutionBackwardFilter_direct_dil1_str1_padsame_ker3_owU128,
	      pParamIn, pDataIn, pParamGradOut, pDataGradOut,
	      pParamConv, pParamGradKernel, pDataGradKernel );
	}
	else
	{
	  return vednnConvolutionBackwardFilter_wrapper(
	      vednnConvolutionBackwardFilter_direct_dil1_str1_padsame_ker3,
	      pParamIn, pDataIn, pParamGradOut, pDataGradOut,
	      pParamConv, pParamGradKernel, pDataGradKernel );
	}
      }
      else if (pParamGradKernel->height == 5 && pParamGradKernel->width == 5)
      {
	if( pParamGradOut->width <= 128)
	{
	  return vednnConvolutionBackwardFilter_wrapper(
	      vednnConvolutionBackwardFilter_direct_dil1_str1_padsame_ker5_owU128,
	      pParamIn, pDataIn, pParamGradOut, pDataGradOut,
	      pParamConv, pParamGradKernel, pDataGradKernel );
	}
	else {
	  return vednnConvolutionBackwardFilter_wrapper(
	      vednnConvolutionBackwardFilter_direct_dil1_str1_padsame_ker5,
	      pParamIn, pDataIn, pParamGradOut, pDataGradOut,
	      pParamConv, pParamGradKernel, pDataGradKernel );
	}
      }
      else if (pParamGradKernel->height == 2 && pParamGradKernel->width == 2)
      {
	if( pParamGradOut->width <= 128 ) {
	  return vednnConvolutionBackwardFilter_wrapper(
	      vednnConvolutionBackwardFilter_direct_dil1_str1_padsame_ker2_owU128,
	      pParamIn, pDataIn, pParamGradOut, pDataGradOut,
	      pParamConv, pParamGradKernel, pDataGradKernel );
	}
	else {
	  return vednnConvolutionBackwardFilter_wrapper(
	      vednnConvolutionBackwardFilter_direct_dil1_str1_padsame_ker2,
	      pParamIn, pDataIn, pParamGradOut, pDataGradOut,
	      pParamConv, pParamGradKernel, pDataGradKernel );
	}
      }
      else
      {
	return vednnConvolutionBackwardFilter_wrapper(
	    vednnConvolutionBackwardFilter_direct_dil1_str1_padsame,
	    pParamIn, pDataIn, pParamGradOut, pDataGradOut,
	    pParamConv, pParamGradKernel, pDataGradKernel );
      }
    }
    else if (pParamConv->dilationHeight == 1 && pParamConv->dilationWidth == 1
	  && pParamConv->padHeight == 0 && pParamConv->padWidth == 0
	  && pParamGradOut->height == (pParamIn->height - pParamGradKernel->height) / pParamConv->strideHeight + 1
	  && pParamGradOut->width == (pParamIn->width - pParamGradKernel->width) / pParamConv->strideWidth + 1 )
    {
      if ( pParamGradKernel->height == 3 && pParamGradKernel->width == 3
	   && pParamConv->strideHeight == 1 && pParamConv->strideWidth == 1
	   && pParamIn->width <= 256
	   && (pParamIn->width & 0x01) == 0 && (((uint64_t)pDataIn) & 0x07) == 0
	   && (pParamGradOut->width & 0x01) == 0 && (((uint64_t)pDataGradOut) & 0x07) == 0 )
      {
	  return vednnConvolutionBackwardFilter_wrapper(
	      vednnConvolutionBackwardFilter_direct_dil1_str1_pad0_ker3_ow2X_iw2XU256_igoaligned,
	      pParamIn, pDataIn, pParamGradOut, pDataGradOut,
	      pParamConv, pParamGradKernel, pDataGradKernel );
      }
      else if (pParamGradOut->width <= 128 && pParamGradKernel->height == 3 && pParamGradKernel->width == 3 )
      {
	if( pParamConv->strideHeight == 1 && pParamConv->strideWidth == 1 ) {
	  return vednnConvolutionBackwardFilter_wrapper(
	      vednnConvolutionBackwardFilter_direct_dil1_str1_pad0_ker3_owU128,
	      pParamIn, pDataIn, pParamGradOut, pDataGradOut,
	      pParamConv, pParamGradKernel, pDataGradKernel );
	}
	else {
	  return vednnConvolutionBackwardFilter_wrapper(
	      vednnConvolutionBackwardFilter_direct_dil1_pad0_ker3_owU128,
	      pParamIn, pDataIn, pParamGradOut, pDataGradOut,
	      pParamConv, pParamGradKernel, pDataGradKernel );
	}
      }
      else if (pParamGradKernel->height == 1 && pParamGradKernel->width == 1) {
	if (pParamGradOut->height * pParamGradOut->width <= 64 ) {
	  return vednnConvolutionBackwardFilter_wrapper(
	      vednnConvolutionBackwardFilter_direct_dil1_pad0_ker1_ohwU64,
	      pParamIn, pDataIn, pParamGradOut, pDataGradOut,
	      pParamConv, pParamGradKernel, pDataGradKernel );
	}
	else if (pParamGradOut->height * pParamGradOut->width <= 128 ) {
	  return vednnConvolutionBackwardFilter_wrapper(
	      vednnConvolutionBackwardFilter_direct_dil1_pad0_ker1_ohwU128,
	      pParamIn, pDataIn, pParamGradOut, pDataGradOut,
	      pParamConv, pParamGradKernel, pDataGradKernel );
	}
	else if (pParamGradOut->width <= 32 ) {
	  return vednnConvolutionBackwardFilter_wrapper(
	      vednnConvolutionBackwardFilter_direct_dil1_pad0_ker1_owU32,
	      pParamIn, pDataIn, pParamGradOut, pDataGradOut,
	      pParamConv, pParamGradKernel, pDataGradKernel );
	}
	else {
	  return vednnConvolutionBackwardFilter_wrapper(
	      vednnConvolutionBackwardFilter_direct_dil1_pad0_ker1,
	      pParamIn, pDataIn, pParamGradOut, pDataGradOut,
	      pParamConv, pParamGradKernel, pDataGradKernel );
	}
      }
      else if (pParamGradOut->width <= 32 ){
	  return vednnConvolutionBackwardFilter_wrapper(
	      vednnConvolutionBackwardFilter_direct_dil1_pad0_owU32,
	      pParamIn, pDataIn, pParamGradOut, pDataGradOut,
	      pParamConv, pParamGradKernel, pDataGradKernel );
      }
      else {
	return vednnConvolutionBackwardFilter_wrapper(
	    vednnConvolutionBackwardFilter_direct_dil1_pad0,
	    pParamIn, pDataIn, pParamGradOut, pDataGradOut,
	    pParamConv, pParamGradKernel, pDataGradKernel );
      }
    }
    else {
      if (pParamGradOut->width <= 128)
      {
	return vednnConvolutionBackwardFilter_wrapper(
	    vednnConvolutionBackwardFilter_direct_owU128,
	    pParamIn, pDataIn, pParamGradOut, pDataGradOut,
	    pParamConv, pParamGradKernel, pDataGradKernel );
      }
      else {
	return vednnConvolutionBackwardFilter_wrapper(
	    vednnConvolutionBackwardFilter_direct_default,
	    pParamIn, pDataIn, pParamGradOut, pDataGradOut,
	    pParamConv, pParamGradKernel, pDataGradKernel );
      }
    }
  }
  else {
    return VEDNN_ERROR_INVALID_PARAM ;
  }
}

