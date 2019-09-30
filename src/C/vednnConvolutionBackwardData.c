
#include "vednnConvolutionBackwardData.h"
#include <stdint.h>
#include <stdio.h>

#ifdef VEDNN_USE_OPENMP
#include <omp.h>
extern int __vednn_omp_num_threads ;
#endif

static inline vednnError_t
vednnConvolutionBackwardData_wrapper(
    vednnConvBackwardData_t		pFunc,
    const vednnTensorParam_t  		*pParamGradOut,
    const void				*pDataGradOut,
    const vednnFilterParam_t		*pParamKernel,
    const void 				*pDataKernel,
    const vednnConvolutionParam_t	*pParamConv,
    const vednnTensorParam_t		*pParamGradIn,
    void 				*pDataGradIn
)
{
#ifdef VEDNN_USE_OPENMP
  if ( __vednn_omp_num_threads == 1 ) {
    return pFunc(pParamGradOut, pDataGradOut, pParamKernel, pDataKernel,
  	  pParamConv, pParamGradIn, pDataGradIn );
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
      else  {
	vednnTensorParam_t _pParamGradOut = *pParamGradOut; _pParamGradOut.batch = myBatch ;
	vednnTensorParam_t _pParamGradIn  = *pParamGradIn ; _pParamGradIn.batch = myBatch ;
	float* _pDataGradOut = ((float *)pDataGradOut) + batchBegin * pParamGradOut->channel * pParamGradOut->height * pParamGradOut->width ;
	float* _pDataGradIn  = ((float *)pDataGradIn) + batchBegin * pParamGradIn->channel * pParamGradIn->height * pParamGradIn->width ;

	rc |= pFunc(&_pParamGradOut, (void*)_pDataGradOut, pParamKernel, pDataKernel,
		    pParamConv, &_pParamGradIn, (void*) _pDataGradIn) ;
      }
    }
    return rc ;
  }
#else
  return pFunc(pParamGradOut, pDataGradOut, pParamKernel, pDataKernel,
	  pParamConv, pParamGradIn, pDataGradIn );
#endif
}

/* ----------------------------------------------------------------------- */
vednnError_t vednnConvolutionBackwardData(
    const vednnTensorParam_t 		*pParamGradOut,
    const void 				*pDataGradOut,
    const vednnFilterParam_t		*pParamKernel,
    const void 				*pDataKernel,
    const vednnTensorParam_t 		*pParamGradIn,
    void 				*pDataGradIn,
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

    if ( pParamGradIn->height * pParamGradIn->width <= 16 ||
	( pParamGradIn->height * pParamGradIn->width < 64
	  && pParamGradIn->height * pParamGradIn->width < pParamGradIn->channel ))
    {
	  return vednnConvolutionBackwardData_wrapper(
	      vednnConvolutionBackwardData_direct_vecC,
	      pParamGradOut, pDataGradOut, pParamKernel, pDataKernel,
	      pParamConv, pParamGradIn, pDataGradIn );
    }
    else if (pParamConv->strideHeight == 1 && pParamConv->strideWidth == 1
	&& pParamConv->dilationHeight == 1 && pParamConv->dilationWidth == 1 )
    {
      if( pParamGradIn->height == pParamGradOut->height
	  && pParamGradIn->width == pParamGradOut->width ) {
	if( pParamKernel->height == 5 && pParamKernel->width == 5) {
	  return vednnConvolutionBackwardData_wrapper(
	      vednnConvolutionBackwardData_direct_dil1_str1_padsame_ker5,
	      pParamGradOut, pDataGradOut, pParamKernel, pDataKernel,
	      pParamConv, pParamGradIn, pDataGradIn );
	}
	else if( pParamKernel->height == 3 && pParamKernel->width == 3) {
	  return vednnConvolutionBackwardData_wrapper(
	      vednnConvolutionBackwardData_direct_dil1_str1_padsame_ker3,
	      pParamGradOut, pDataGradOut, pParamKernel, pDataKernel,
	      pParamConv, pParamGradIn, pDataGradIn );
	}
	else if( pParamKernel->height == 2 && pParamKernel->width == 2) {
	  return vednnConvolutionBackwardData_wrapper(
	      vednnConvolutionBackwardData_direct_dil1_str1_padsame_ker2,
	      pParamGradOut, pDataGradOut, pParamKernel, pDataKernel,
	      pParamConv, pParamGradIn, pDataGradIn );
	}
	else if( pParamKernel->height == 1 && pParamKernel->width == 1) {
	  return vednnConvolutionBackwardData_wrapper(
	      vednnConvolutionBackwardData_direct_dil1_str1_padsame_ker1,
	      pParamGradOut, pDataGradOut, pParamKernel, pDataKernel,
	      pParamConv, pParamGradIn, pDataGradIn );
	}
	else {
	  return vednnConvolutionBackwardData_wrapper(
	      vednnConvolutionBackwardData_direct_dil1_str1_padsame,
	      pParamGradOut, pDataGradOut, pParamKernel, pDataKernel,
	      pParamConv, pParamGradIn, pDataGradIn );
	}
      }
      else if( pParamConv->padHeight == 0 && pParamConv->padWidth == 0
	    && pParamKernel->height == 3 && pParamKernel->width == 3
	    && (pParamGradIn->width & 0x01) == 0 && pParamGradIn->width <=256
	    && (pParamGradOut->width & 0x01) == 0
	    && (((uint64_t)pDataGradIn) & 0x07) == 0
	    && (((uint64_t)pDataGradOut) & 0x07) == 0 )
      {
	if( pParamGradIn->width <=32 ) {
	  return vednnConvolutionBackwardData_wrapper(
	      vednnConvolutionBackwardData_direct_dil1_str1_pad0_ker3_iw2XU32_ow2X_ioaligned,
	      pParamGradOut, pDataGradOut, pParamKernel, pDataKernel,
	      pParamConv, pParamGradIn, pDataGradIn );
	}
	else {
	  return vednnConvolutionBackwardData_wrapper(
	      vednnConvolutionBackwardData_direct_dil1_str1_pad0_ker3_iw2XU256_ow2X_ioaligned,
	      pParamGradOut, pDataGradOut, pParamKernel, pDataKernel,
	      pParamConv, pParamGradIn, pDataGradIn );
	}
      }
      else if (pParamGradIn->width <= 128)
      {
	if( pParamConv->padHeight == 0 && pParamConv->padWidth == 0
	    && pParamKernel->height == 3 && pParamKernel->width == 3 )
	{
	  return vednnConvolutionBackwardData_wrapper(
	      vednnConvolutionBackwardData_direct_dil1_str1_pad0_ker3_iwU128,
	      pParamGradOut, pDataGradOut, pParamKernel, pDataKernel,
	      pParamConv, pParamGradIn, pDataGradIn );
	}
	else {
	  return vednnConvolutionBackwardData_wrapper(
	      vednnConvolutionBackwardData_direct_dil1_str1_iwU128,
	      pParamGradOut, pDataGradOut, pParamKernel, pDataKernel,
	      pParamConv, pParamGradIn, pDataGradIn );
	}
      }
      else {
	return vednnConvolutionBackwardData_wrapper(
	    vednnConvolutionBackwardData_direct_dil1_str1,
	    pParamGradOut, pDataGradOut, pParamKernel, pDataKernel,
	    pParamConv, pParamGradIn, pDataGradIn );
      }
    }
    else
    {
      if( pParamConv->dilationHeight == 1 && pParamConv->dilationWidth == 1
	  && pParamConv->padHeight == 0 && pParamConv->padWidth == 0
	  && pParamKernel->height == 1 && pParamKernel->width == 1
	  && pParamGradOut->width <= 128 )
      {
	return vednnConvolutionBackwardData_wrapper(
	    vednnConvolutionBackwardData_direct_dil1_pad0_ker1_owU128,
	    pParamGradOut, pDataGradOut, pParamKernel, pDataKernel,
	    pParamConv, pParamGradIn, pDataGradIn );
      }
      if( pParamKernel->height == 5 && pParamKernel->width == 5
	  && pParamConv->dilationHeight == 1 && pParamConv->dilationWidth == 1
	  && pParamConv->strideHeight == 2 && pParamConv->strideWidth == 2
	  && pParamConv->padHeight == 2 && pParamConv->padWidth == 2 )
      {
	if (pParamGradIn->width <= 128) {
	  return vednnConvolutionBackwardData_wrapper(
	    vednnConvolutionBackwardData_direct_dil1_str2_pad2_ker5_iwU128,
	    pParamGradOut, pDataGradOut, pParamKernel, pDataKernel,
	    pParamConv, pParamGradIn, pDataGradIn );
	}
	else {
	  return vednnConvolutionBackwardData_wrapper(
	    vednnConvolutionBackwardData_direct_dil1_str2_pad2_ker5,
	    pParamGradOut, pDataGradOut, pParamKernel, pDataKernel,
	    pParamConv, pParamGradIn, pDataGradIn );
	}
      }
      // no else

      if (pParamGradIn->width <= 128)
      {
	if( pParamKernel->height == 3 && pParamKernel->width == 3 ) {
	  return vednnConvolutionBackwardData_wrapper(
	      vednnConvolutionBackwardData_direct_ker3_iwU128,
	      pParamGradOut, pDataGradOut, pParamKernel, pDataKernel,
	      pParamConv, pParamGradIn, pDataGradIn );
	}
	else if( pParamKernel->height == 5 && pParamKernel->width == 5 ) {
	  return vednnConvolutionBackwardData_wrapper(
	      vednnConvolutionBackwardData_direct_ker5_iwU128,
	      pParamGradOut, pDataGradOut, pParamKernel, pDataKernel,
	      pParamConv, pParamGradIn, pDataGradIn );
	}
	else {
	  return vednnConvolutionBackwardData_wrapper(
	      vednnConvolutionBackwardData_direct_iwU128,
	      pParamGradOut, pDataGradOut, pParamKernel, pDataKernel,
	      pParamConv, pParamGradIn, pDataGradIn );
	}
      }
      else {
	if( pParamKernel->height == 5 && pParamKernel->width == 5 ) {
	  return vednnConvolutionBackwardData_wrapper(
	    vednnConvolutionBackwardData_direct_ker5,
	    pParamGradOut, pDataGradOut, pParamKernel, pDataKernel,
	    pParamConv, pParamGradIn, pDataGradIn );
	}
	else {
	  return vednnConvolutionBackwardData_wrapper(
	    vednnConvolutionBackwardData_direct_default,
	    pParamGradOut, pDataGradOut, pParamKernel, pDataKernel,
	    pParamConv, pParamGradIn, pDataGradIn );
	}
      }
    }
  }
  else {
    return VEDNN_ERROR_INVALID_PARAM ;
  }
}

