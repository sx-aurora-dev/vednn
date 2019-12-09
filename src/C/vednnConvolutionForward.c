#include "vednnConvolutionForward.h"
#include "vednn-def.h"
#include <stdint.h>
#include <assert.h>
#include <stdio.h>

static inline vednnError_t
vednnConvolutionForward_wrapper(
    vednnConvForward_t			pFunc,
    VEDNN_CONVFWD_ARGS )
{
#ifndef VEDNN_USE_OPENMP
  return pFunc(VEDNN_CONVFWD_ARGS_LIST);
#else
  int64_t allBatch = pParamIn->batch; // check as in vednnx
  if (allBatch == 1 || __vednn_omp_num_threads == 1) {
    return pFunc(VEDNN_CONVFWD_ARGS_LIST);
  }else{
    vednnError_t rc = VEDNN_SUCCESS ;
#pragma omp parallel reduction(|:rc)
    {
      int64_t nthreads = omp_get_num_threads() ;
      int64_t threadid = omp_get_thread_num() ;

      int64_t nBatch = allBatch / nthreads ;
      int64_t remain = allBatch % nthreads ;

      int64_t batchBegin = nBatch * threadid + ( threadid < remain ? threadid : remain ) ;
      int64_t myBatch = nBatch + ( threadid < remain ? 1 : 0 ) ;

      if( myBatch == 0 ) {
        rc |= VEDNN_SUCCESS ;
      }else{
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

#define OMPWRAP( IMPL ) WRAP_RET(vednnConvolutionForward_direct_##IMPL, \
    vednnConvolutionForward_wrapper, VEDNN_CONVFWD_ARGS_LIST)
  if (algo == VEDNN_CONV_ALGORITHM_DIRECT)
  {
    if ( pParamOut->height * pParamOut->width <= 16  )
      OMPWRAP(vecC);
    else if (pParamConv->strideHeight == 1 && pParamConv->strideWidth == 1
        && pParamConv->dilationHeight == 1 && pParamConv->dilationWidth == 1
        && pParamIn->height == pParamOut->height
        && pParamIn->width == pParamOut->width )
    { // d1s1pS
      if (pParamKernel->width == 1 && pParamKernel->height == 1)
        OMPWRAP(dil1_str1_pad0_ker1);
      else if (pParamKernel->height == 3 && pParamKernel->width == 3)
      {
        if (pParamIn->channel == pParamConv->group) // aka inputChannelGroup==1
        {
          if (pParamOut->width <= 128)
            OMPWRAP(dil1_str1_padsame_ker3_c1_owU128);
          else
            OMPWRAP(dil1_str1_padsame_ker3_c1);
        }else
          OMPWRAP(dil1_str1_padsame_ker3);
      }else if (pParamKernel->height == 5 && pParamKernel->width == 5){
        if( pParamOut->width <= 128 )
          OMPWRAP(dil1_str1_padsame_ker5_owU128);
        else
          OMPWRAP(dil1_str1_padsame_ker5);
      }else if (pParamKernel->height == 2 && pParamKernel->width == 2)
        OMPWRAP(dil1_str1_padsame_ker2);
      else
        OMPWRAP(dil1_str1_padsame);
    }else if ( pParamConv->dilationHeight == 1 && pParamConv->dilationWidth == 1
        && pParamConv->padHeight == 0  && pParamConv->padWidth == 0
        && pParamOut->height == (pParamIn->height - pParamKernel->height) / pParamConv->strideHeight + 1
        && pParamOut->width == (pParamIn->width - pParamKernel->width) / pParamConv->strideWidth + 1 )
    { // d1p0 and oh expected value
      if (pParamConv->strideHeight == 1 && pParamConv->strideWidth == 1 )
      {
        if ( pParamKernel->height == 3 && pParamKernel->width == 3
            && (pParamIn->width <= 256)
            && (pParamIn->width & 0x1) == 0  && (((uint64_t)pDataIn) & 0x7) == 0
            && (pParamOut->width & 0x1) == 0 && (((uint64_t)pDataOut) & 0x7) == 0 )
          OMPWRAP(dil1_str1_pad0_ker3_iw2XU256_ow2X_ioaligned);
        else if ( pParamKernel->height == 4 && pParamKernel->width == 4  && (pParamIn->width <= 256) )
          OMPWRAP(dil1_str1_pad0_ker4_iwU256);
        else if (pParamOut->width <= 128)
          OMPWRAP(dil1_str1_pad0_owU128);
        else
          OMPWRAP(dil1_str1_pad0);
      } else if( pParamKernel->width == 1 && pParamKernel->height == 1 ){
        if (pParamOut->width <= 128)
          OMPWRAP(dil1_pad0_owU128_ker1);
        else
          OMPWRAP(dil1_pad0_ker1);
      }else{
        if (pParamOut->width <= 128)
          OMPWRAP(dil1_pad0_owU128);
        else
          OMPWRAP(dil1_pad0);
      }
    } else if (pParamConv->strideHeight == 2 && pParamConv->strideWidth == 2
        && pParamConv->dilationHeight == 1 && pParamConv->dilationWidth == 1
        && pParamConv->padHeight == 1 && pParamConv->padWidth == 1
        && pParamKernel->height == 3 && pParamKernel->width == 3
        && pParamOut->width <= 128 )
      OMPWRAP(dil1_str2_pad1_ker3_owU128); // N/A
    else if (pParamConv->strideHeight == 2 && pParamConv->strideWidth == 2
           && pParamConv->dilationHeight == 1 && pParamConv->dilationWidth == 1
           && pParamConv->padHeight == 1 && pParamConv->padWidth == 1
           && pParamKernel->height == 4 && pParamKernel->width == 4
           && pParamOut->width <= 128 )
      OMPWRAP(dil1_str2_pad1_ker4_owU128);
    else{
      if (pParamOut->width <= 128)
        OMPWRAP(owU128);
      else
        OMPWRAP(default);
    }
  }
  else{
    return VEDNN_ERROR_INVALID_PARAM ;
  }
}
#undef OMPWRAP

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
// vim: et sw=2 ts=2
