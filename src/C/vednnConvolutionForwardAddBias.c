
#include "vednnConvolutionForwardAddBias.h"

#ifdef VEDNN_USE_OPENMP
#include <stdint.h>
#include <omp.h>
extern int __vednn_omp_num_threads ;
#endif

static inline vednnError_t
vednnConvolutionForwardAddBias_wrapper(
    vednnConvForwardAddBias_t		pFunc,
    const vednnTensorParam_t  		*pParamIn,
    const void  			*pDataIn,
    const vednnFilterParam_t  		*pParamKernel,
    const void  			*pDataKernel,
    const vednnBiasParam_t  		*pParamBias,
    const void  			*pDataBias,
    const vednnConvolutionParam_t	*pParamConv,
    const vednnTensorParam_t		*pParamOut,
    void  				*pDataOut
)
{
#ifdef VEDNN_USE_OPENMP
  if ( __vednn_omp_num_threads == 1 ) {
    return pFunc(pParamIn, pDataIn, pParamKernel, pDataKernel,
  	       pParamBias, pDataBias, pParamConv, pParamOut, pDataOut);
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
      else  {
	vednnTensorParam_t _pParamIn  = *pParamIn  ; _pParamIn.batch = myBatch ;
	vednnTensorParam_t _pParamOut = *pParamOut ; _pParamOut.batch = myBatch ;
	float* _pDataIn  = ((float *)pDataIn) + batchBegin * pParamIn->channel * pParamIn->height * pParamIn->width ;
	float* _pDataOut = ((float *)pDataOut) + batchBegin * pParamOut->channel * pParamOut->height * pParamOut->width ;

	rc |= pFunc(&_pParamIn, (void*)_pDataIn, pParamKernel, pDataKernel,
		    pParamBias, pDataBias, pParamConv, &_pParamOut, (void*) _pDataOut) ;
      }
    }
    return rc ;
  }
#else
  return pFunc(pParamIn, pDataIn, pParamKernel, pDataKernel,
	       pParamBias, pDataBias, pParamConv, pParamOut, pDataOut);
#endif
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
  if (algo == VEDNN_CONV_ALGORITHM_DIRECT)
  {
    // [todo] add variations
    if (pParamConv->strideHeight == 1 && pParamConv->strideWidth == 1
	&& pParamConv->dilationHeight == 1 && pParamConv->dilationWidth == 1
	&& 2 * pParamConv->padHeight + 1 == pParamKernel->height
	&& 2 * pParamConv->padWidth + 1 == pParamKernel->width)
    {
      if (pParamKernel->width == 1 && pParamKernel->height == 1)
      {
	if (pParamKernel->inChannel % 1024 == 0)
	{
	  return vednnConvolutionForwardAddBias_wrapper(
	      vednnConvolutionForwardAddBias_direct_dil1_str1_pad0_ker1_c1024x,
	      pParamIn, pDataIn, pParamKernel, pDataKernel,
	      pParamBias, pDataBias, pParamConv, pParamOut, pDataOut);
	}
	else {
	  return vednnConvolutionForwardAddBias_wrapper(
	      vednnConvolutionForwardAddBias_direct_dil1_str1_pad0_ker1,
	      pParamIn, pDataIn, pParamKernel, pDataKernel, pParamBias,
	      pDataBias, pParamConv, pParamOut, pDataOut);
	}
      }
      else
      {
	if (pParamKernel->height == 3 && pParamKernel->width == 3)
	{
	  if (pParamKernel->inChannel == pParamConv->group)
	  {
	    if (pParamOut->width <= 128)
	    {
	      return vednnConvolutionForwardAddBias_wrapper(
		  vednnConvolutionForwardAddBias_direct_dil1_str1_padsame_ker3_c1_owU128,
		  pParamIn, pDataIn, pParamKernel, pDataKernel,
		  pParamBias, pDataBias, pParamConv, pParamOut, pDataOut );
	    }
	    else {
	      return vednnConvolutionForwardAddBias_wrapper(
		  vednnConvolutionForwardAddBias_direct_dil1_str1_padsame_ker3_c1,
		  pParamIn, pDataIn, pParamKernel, pDataKernel,
		  pParamBias, pDataBias, pParamConv, pParamOut, pDataOut );
	    }
	  }
	  else if (pParamKernel->inChannel % 1024 == 0)
	  {
	    return vednnConvolutionForwardAddBias_wrapper(
		vednnConvolutionForwardAddBias_direct_dil1_str1_padsame_ker3_c1024x,
		pParamIn, pDataIn, pParamKernel, pDataKernel,
		pParamBias, pDataBias, pParamConv, pParamOut, pDataOut );
	  }
	  else
	  {
	    return vednnConvolutionForwardAddBias_wrapper(
		vednnConvolutionForwardAddBias_direct_dil1_str1_padsame_ker3,
		pParamIn, pDataIn, pParamKernel, pDataKernel,
		pParamBias, pDataBias, pParamConv, pParamOut, pDataOut );
	  }
	}
	else
	{
	  return vednnConvolutionForwardAddBias_wrapper(
	      vednnConvolutionForwardAddBias_direct_dil1_str1_padsame,
	      pParamIn, pDataIn, pParamKernel, pDataKernel,
	      pParamBias, pDataBias, pParamConv, pParamOut, pDataOut );
	}
      }
    }

    else
    {
      return vednnConvolutionForwardAddBias_wrapper(
	  vednnConvolutionForwardAddBias_direct_default,
	  pParamIn, pDataIn, pParamKernel, pDataKernel,
	  pParamBias, pDataBias, pParamConv, pParamOut, pDataOut );
    }
  }
  else {
    return VEDNN_ERROR_INVALID_PARAM ;
  }
}

