#include "vednnConvolutionBackwardFilter.h"
#include "vednn-def.h"
#include <stdint.h>
#include <stdio.h>

  static inline vednnError_t
vednnConvolutionBackwardFilter_wrapper(
    vednnConvBackwardFilter_t pFunc,
    VEDNN_CONVBKF_ARGS )
{
#ifndef VEDNN_USE_OPENMP
  return pFunc( VEDNN_CONVBKF_ARGS_LIST );
#else // VEDNN_USE_OPENMP
#ifndef VEDNN_OMP_GROUP_PARALLEL
  if ( __vednn_omp_num_threads == 1 ) {
    int64_t gOutChannel = pParamGradOut->channel;
    int64_t group       = pParamConv->group;
    int64_t gOutChannelGroup = gOutChannel  / group;

    return pFunc(VEDNN_CONVBKF_ARGS_LIST, 0, gOutChannelGroup);
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

        rc |= pFunc(VEDNN_CONVBKF_ARGS_LIST, beginOChannel, myOChannel );
      }
    }
    return rc ;
  }
#else // VEDNN_OMP_GROUP_PARALLEL
  if ( __vednn_omp_num_threads == 1 ) {
    int64_t gOutChannel = pParamGradOut->channel;
    int64_t group       = pParamConv->group;
    int64_t gOutChannelGroup = gOutChannel  / group;

    return pFunc(VEDNN_CONVBKF_ARGS_LIST, 0, gOutChannelGroup, 0, group);
  }
  else {
    vednnError_t rc = VEDNN_SUCCESS ;
#pragma omp parallel reduction(|:rc)
    {
      int64_t nthreads = omp_get_num_threads() ;
      int64_t threadid = omp_get_thread_num() ;

      int64_t gOutChannel      = pParamGradOut->channel;
      int64_t group            = pParamConv->group;
      int64_t gOutChannelGroup = gOutChannel  / group;

      if( gOutChannelGroup >= group )
      {
	int64_t nOChannlel = gOutChannelGroup / nthreads ;
	int64_t remain     = gOutChannelGroup % nthreads ;

	int64_t beginOChannel = nOChannlel * threadid + ( threadid < remain ? threadid : remain ) ;
	int64_t myOChannel    = nOChannlel + ( threadid < remain ? 1 : 0 ) ;

	if( myOChannel == 0 ) {
	  rc |= VEDNN_SUCCESS ;
	}
	else  {
	  rc |= pFunc(VEDNN_CONVBKF_ARGS_LIST, beginOChannel, myOChannel, 0, group);
	}
      }
      else {
	int64_t nGroup = group / nthreads ;
	int64_t remain = group % nthreads ;

	int64_t beginGroup = nGroup * threadid + ( threadid < remain ? threadid : remain ) ;
	int64_t myGroup    = nGroup + ( threadid < remain ? 1 : 0 ) ;

	if( myGroup == 0 ) {
	  rc |= VEDNN_SUCCESS ;
	}
	else  {
	  rc |= pFunc(VEDNN_CONVBKF_ARGS_LIST, 0, gOutChannelGroup, beginGroup, myGroup);
	}
      }
    }
    return rc ;
  }
#endif // VEDNN_OMP_GROUP_PARALLEL
#endif // VEDNN_USE_OMP
}

/* ----------------------------------------------------------------------- */
vednnError_t vednnConvolutionBackwardFilter(
    const vednnTensorParam_t     *pParamIn,
    const void         *pDataIn,
    const vednnTensorParam_t     *pParamGradOut,
    const void         *pDataGradOut,
    const vednnFilterParam_t    *pParamGradKernel,
    void         *pDataGradKernel,
    const vednnConvolutionParam_t  *pParamConv,
    vednnConvolutionAlgorithm_t   algo
)
{
  switch( pParamGradKernel->layout ) {
    case VEDNN_FILTER_LAYOUT_NCHW :
      break ;
    case VEDNN_FILTER_LAYOUT_HWCN :
      if( pParamConv->group > 1 ) {
        fprintf(stderr, "[VEDNN ERROR] VEDNN does not support grouped convolution with filter_hwcn\n") ;
        return VEDNN_ERROR_INVALID_PARAM ;
      }
      break ;
    default :
      fprintf(stderr, "[VEDNN ERROR] Unknown Filter Layout %d\n", pParamGradKernel->layout) ;
      return VEDNN_ERROR_INVALID_PARAM ;
  }

  if (algo == VEDNN_CONV_ALGORITHM_DIRECT)
  {
#define OMPWRAP( IMPL ) WRAP_RET(vednnConvolutionBackwardFilter_direct_##IMPL, \
    vednnConvolutionBackwardFilter_wrapper, VEDNN_CONVBKF_ARGS_LIST )
#define DIL(N) (pParamConv->dilationHeight == (N) && pParamConv->dilationWidth == (N))
#define PAD(N) (pParamConv->padHeight == (N) && pParamConv->padWidth == (N))
#define STR(N) (pParamConv->strideHeight == (N) && pParamConv->strideWidth == (N))
#define KER(N) (pParamGradKernel->width == (N) && pParamGradKernel->height == (N))
#define IWU(N) (pParamIn->width <= (N))
#define OWU(N) (pParamGradOut->width <= (N))
#define OHWU(N) (pParamGradOut->width * pParamGradOut->height <= (N))
    if ( pParamGradOut->height * pParamGradOut->width <= 16 ||
        ( pParamGradOut->height * pParamGradOut->width < 64
          && pParamGradOut->height * pParamGradOut->width < pParamIn->channel / pParamConv->group )  ) {
      OMPWRAP(vecC);
    }else if (STR(1) && DIL(1)
        && pParamIn->height == pParamGradOut->height
        && pParamIn->width == pParamGradOut->width ) // d1s1pS
    {
      if (KER(3)) {
        if (OHWU(256))     OMPWRAP(dil1_str1_padsame_ker3_ohwU256);
        else if (OWU(128)) OMPWRAP(dil1_str1_padsame_ker3_owU128);
        else               OMPWRAP(dil1_str1_padsame_ker3);
      }else if (KER(1)) {  OMPWRAP(dil1_str1_padsame_ker1);
      }else if (KER(5)) {
        if (OWU(128))      OMPWRAP(dil1_str1_padsame_ker5_owU128);
        else               OMPWRAP(dil1_str1_padsame_ker5);
      }else if (KER(2)) {
        if (OWU(128))      OMPWRAP(dil1_str1_padsame_ker2_owU128);
        else               OMPWRAP(dil1_str1_padsame_ker2);
      }
      OMPWRAP(dil1_str1_padsame);
    }
    else if (DIL(1) && PAD(0)
        && pParamGradOut->height == (pParamIn->height - pParamGradKernel->height) / pParamConv->strideHeight + 1
        && pParamGradOut->width == (pParamIn->width - pParamGradKernel->width) / pParamConv->strideWidth + 1 )
    { // d1p0 and oh,ow correct for whatever stride
      if (KER(3) && STR(1) && IWU(256)
          && (pParamIn->width & 0x01) == 0 && (pParamGradOut->width & 0x01) == 0
          && (((uint64_t)pDataIn) & 0x07) == 0 && (((uint64_t)pDataGradOut) & 0x07) == 0 )
      {
        OMPWRAP(dil1_str1_pad0_ker3_ow2X_iw2XU256_igoaligned);
      }
      else if (KER(3) && OWU(128))
      {
        if (STR(1)) OMPWRAP(dil1_str1_pad0_ker3_owU128);
        else        OMPWRAP(dil1_pad0_ker3_owU128);
      }
      else if (KER(1))
      {
        if (OHWU(64))       OMPWRAP(dil1_pad0_ker1_ohwU64);
        else if (OHWU(128)) OMPWRAP(dil1_pad0_ker1_ohwU128);
        else if (OWU(32))   OMPWRAP(dil1_pad0_ker1_owU32);
        else                OMPWRAP(dil1_pad0_ker1);
      }
      else if (KER(4) && OWU(128) && STR(1)) OMPWRAP(dil1_str1_pad0_ker4_owU128);
      else if (OWU(32)) OMPWRAP(dil1_pad0_owU32);
      OMPWRAP(dil1_pad0);
    }
    else if(OWU(128))
    {
      if (KER(3)) {
        if (STR(2) && DIL(1) && PAD(1)) OMPWRAP(dil1_str2_pad1_ker3_owU128) ;
        else                            OMPWRAP(ker3_owU128) ;
      }else if (KER(4) && STR(2) && DIL(1) && PAD(1))
        OMPWRAP(dil1_str2_pad1_ker4_owU128) ;
      OMPWRAP(owU128);
    }
    OMPWRAP(default);
  }
  else {
    return VEDNN_ERROR_INVALID_PARAM ;
  }
#undef OHWU
#undef OWU
#undef IWU
#undef KER
#undef STR
#undef PAD
#undef DIL
#undef OMPWRAP
}
// vim: et sw=2 ts=2
