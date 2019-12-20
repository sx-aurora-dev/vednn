#include <stdint.h>
#include <stdlib.h>

#include "vednn.h"

#include "velintrin.h"
#define VLEN	(256)


template<filterLayout_t FLAYOUT, bool ADDBIAS, int NUMKERNEL, int C>
static inline void func(
    const float * __restrict__ pIn,
    const float * __restrict__ pKernel,
    const float * __restrict__ pBias,
    float * __restrict__ const pOut,
    const int64_t inChannel,
    const int64_t inWidth,
    const int64_t inHeight,
    const int64_t outChannel,
    const int64_t outWidth,
    const int64_t outHeight,
    const int64_t kernWidth,
    const int64_t kernHeight,
    const int64_t inChannelGroup,
    const int64_t outChannelGroup,
    const int64_t strideHeight,
    const int64_t strideWidth,
    const int64_t padHeight,
    const int64_t padWidth,
    const int64_t dilationHeight,
    const int64_t dilationWidth,
    const int64_t inGroupOffset,
    const int64_t outGroupOffset,
    const int64_t biasGroupOffset,
    const int64_t kernGroupOffset,
    const int64_t n,
    const int64_t k
)
{

  int64_t outIndex  = outGroupOffset + ((n * outChannel + k) * outHeight) * outWidth;

  const int64_t remain  = NUMKERNEL & 0x1 ;
  const int64_t nPacked = NUMKERNEL >> 1 ;

  float bias[NUMKERNEL] ;
#pragma clang loop unroll(full)
  for(int64_t kk=0; kk<NUMKERNEL; kk++) {
    bias[kk] = pBias[biasGroupOffset+k+kk] ;
  }


  const int64_t maxvl = VLEN ;
  const int64_t remvl = inChannelGroup-(C-1)*VLEN ;

  const float *pKerValue = ( FLAYOUT == VEDNN_FILTER_LAYOUT_NCHW ) ?
			    pKernel + kernGroupOffset + k * inChannelGroup  :
			    pKernel + kernGroupOffset + k ;

  const int64_t kernelDistance = ( FLAYOUT == VEDNN_FILTER_LAYOUT_NCHW ) ?
				 inChannelGroup * 1 * 1 :
				 1 ;

  const int64_t kernelStride = ( FLAYOUT == VEDNN_FILTER_LAYOUT_NCHW ) ?
			       1 * 1 :
			       outChannelGroup ;

  __vr vrk[C*NUMKERNEL] ;
#pragma clang loop unroll(full)
  for(int64_t kk=0; kk<NUMKERNEL; kk++) {
#pragma clang loop unroll(full)
    for(int64_t cc=0; cc<C-1; cc++) {
      vrk[kk*C+cc] = _vel_vldu_vssl(4*kernelStride, pKerValue+kk*kernelDistance+cc*kernelStride*maxvl, maxvl) ;
    }
    {
      int64_t cc=C-1;
      vrk[kk*C+cc] = _vel_vldu_vssl(4*kernelStride, pKerValue+kk*kernelDistance+cc*kernelStride*maxvl, remvl) ;
    }
  }

  __vr vrkp[C*nPacked] ;
#pragma clang loop unroll(full)
  for(int64_t kk=0; kk<nPacked; kk++) {
#pragma clang loop unroll(full)
    for(int64_t cc=0; cc<C-1; cc++) {
      vrkp[kk*C+cc] = _vel_vshf_vvvsl(vrk[(2*kk+remain)*C+cc], vrk[(2*kk+remain+1)*C+cc], VE_VSHUFFLE_YUZU, maxvl) ;
    }
    {
      int64_t cc=C-1;
      vrkp[kk*C+cc] = _vel_vshf_vvvsl(vrk[(2*kk+remain)*C+cc], vrk[(2*kk+remain+1)*C+cc], VE_VSHUFFLE_YUZU, remvl) ;
    }
  }

  for(int64_t y=0; y<outHeight; y++) {
    for(int64_t x=0; x<outWidth; x++) {
      int64_t inputIndex = inGroupOffset + (n * inChannel * inHeight + y*strideHeight) * inWidth + x * strideWidth;

      __vr vri[C] ;
#pragma clang loop unroll(full)
      for(int64_t cc=0; cc<C-1; cc++) {
        vri[cc] = _vel_vldu_vssl(4*inHeight*inWidth, &pIn[inputIndex+cc*maxvl*inHeight*inWidth], maxvl) ;
      }
      {
        int64_t cc=C-1;
        vri[cc] = _vel_vldu_vssl(4*inHeight*inWidth, &pIn[inputIndex+cc*maxvl*inHeight*inWidth], remvl) ;
      }

      __vr vrip[C] ;
#pragma clang loop unroll(full)
      for(int64_t cc=0; cc<C-1; cc++) {
        vrip[cc] = _vel_vshf_vvvsl(vri[cc], vri[cc], VE_VSHUFFLE_YUZU, maxvl) ;
      }
      {
        int64_t cc=C-1;
        vrip[cc] = _vel_vshf_vvvsl(vri[cc], vri[cc], VE_VSHUFFLE_YUZU, remvl) ;
      }

      __vr vrsum0 = _vel_vbrds_vsl(0.f, VLEN) ;
      __vr vrsum[nPacked] ;
#pragma clang loop unroll(full)
      for(int64_t kk=0; kk<nPacked; kk++) {
	vrsum[kk] = _vel_vbrdl_vsl(0UL, VLEN) ;
      }

#pragma clang loop unroll(full)
      for(int64_t cc=0; cc<C-1; cc++) {
	if( remain ) {
	  vrsum0 = _vel_vfmads_vvvvvl(vrsum0, vri[cc], vrk[0*C+cc], vrsum0, maxvl) ;
	}
#pragma clang loop unroll(full)
	for(int64_t kk=0; kk<nPacked; kk++) {
	  vrsum[kk] = _vel_pvfmad_vvvvvl(vrsum[kk] , vrip[cc], vrkp[kk*C+cc], vrsum[kk], maxvl) ;
	}
      }
      {
	int64_t cc=C-1;
	if( remain ) {
	  vrsum0 = _vel_vfmads_vvvvvl(vrsum0, vri[cc], vrk[0*C+cc], vrsum0, remvl) ;
	}
#pragma clang loop unroll(full)
	for(int64_t kk=0; kk<nPacked; kk++) {
	  vrsum[kk] = _vel_pvfmad_vvvvvl(vrsum[kk] , vrip[cc], vrkp[kk*C+cc], vrsum[kk], remvl) ;
	}
      }

      // store
      if( remain ) {
    	vrsum0 = _vel_vfsums_vvl(vrsum0, VLEN) ;
    	if(ADDBIAS) vrsum0 = _vel_vfadds_vsvl(bias[0], vrsum0, 1) ;
    	_vel_vstu_vssl(vrsum0, 4, &pOut[outIndex+0*outHeight*outWidth+y*outWidth+x], 1) ;
      }

#pragma clang loop unroll(full)
      for(int64_t kk=0; kk<nPacked; kk++) {
    	__vr vrsumU = _vel_vfsums_vvl(vrsum[kk], maxvl) ;
    	if(ADDBIAS) vrsumU = _vel_vfadds_vsvl(bias[2*kk+remain], vrsumU, 1) ;
    	_vel_vstu_vssl(vrsumU, 4, &pOut[outIndex+(2*kk+remain)*outHeight*outWidth+y*outWidth+x], 1) ;
    	__vr vrsumL = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsum[kk],32, maxvl), maxvl) ;
    	if(ADDBIAS) vrsumL = _vel_vfadds_vsvl(bias[2*kk+remain+1], vrsumL, 1) ;
    	_vel_vstu_vssl(vrsumL, 4, &pOut[outIndex+(2*kk+remain+1)*outHeight*outWidth+y*outWidth+x], 1) ;
      }
    }
  }
}

template<filterLayout_t FLAYOUT, bool ADDBIAS, int NUMKERNEL>
static inline void selector(
    const float * __restrict__ pIn,
    const float * __restrict__ pKernel,
    const float * __restrict__ pBias,
    float * __restrict__ const pOut,
    const int64_t inChannel,
    const int64_t inWidth,
    const int64_t inHeight,
    const int64_t outChannel,
    const int64_t outWidth,
    const int64_t outHeight,
    const int64_t kernWidth,
    const int64_t kernHeight,
    const int64_t inChannelGroup,
    const int64_t outChannelGroup,
    const int64_t strideHeight,
    const int64_t strideWidth,
    const int64_t padHeight,
    const int64_t padWidth,
    const int64_t dilationHeight,
    const int64_t dilationWidth,
    const int64_t inGroupOffset,
    const int64_t outGroupOffset,
    const int64_t biasGroupOffset,
    const int64_t kernGroupOffset,
    const int64_t n,
    const int64_t k
)
{
  if( inChannelGroup <= 512 ) {
    if( inChannelGroup <= 256 ) {
      func<FLAYOUT,ADDBIAS,NUMKERNEL,1>(
	 pIn, pKernel, pBias, pOut,
	 inChannel, inWidth, inHeight,
	 outChannel, outWidth, outHeight,
	 kernWidth, kernHeight,
	 inChannelGroup, outChannelGroup,
	 strideHeight, strideWidth,
	 padHeight, padWidth,
	 dilationHeight, dilationWidth,
	 inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	 n, k ) ;
    }
    else {
      func<FLAYOUT,ADDBIAS,NUMKERNEL,2>(
	 pIn, pKernel, pBias, pOut,
	 inChannel, inWidth, inHeight,
	 outChannel, outWidth, outHeight,
	 kernWidth, kernHeight,
	 inChannelGroup, outChannelGroup,
	 strideHeight, strideWidth,
	 padHeight, padWidth,
	 dilationHeight, dilationWidth,
	 inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	 n, k ) ;
    }
  }
  else {
    if( inChannelGroup <= 768 ) {
      func<FLAYOUT,ADDBIAS,NUMKERNEL,3>(
	 pIn, pKernel, pBias, pOut,
	 inChannel, inWidth, inHeight,
	 outChannel, outWidth, outHeight,
	 kernWidth, kernHeight,
	 inChannelGroup, outChannelGroup,
	 strideHeight, strideWidth,
	 padHeight, padWidth,
	 dilationHeight, dilationWidth,
	 inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	 n, k ) ;
    }
    else {
      func<FLAYOUT,ADDBIAS,NUMKERNEL,4>(
	 pIn, pKernel, pBias, pOut,
	 inChannel, inWidth, inHeight,
	 outChannel, outWidth, outHeight,
	 kernWidth, kernHeight,
	 inChannelGroup, outChannelGroup,
	 strideHeight, strideWidth,
	 padHeight, padWidth,
	 dilationHeight, dilationWidth,
	 inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	 n, k ) ;
    }
  }


}

template<filterLayout_t FLAYOUT, bool ADDBIAS>
static inline void convloop(
    const float * __restrict__ pIn,
    const float * __restrict__ pKernel,
    const float * __restrict__ pBias,
    float * __restrict__ const pOut,
    const int64_t batch,
    const int64_t group,
    const int64_t inChannel,
    const int64_t inWidth,
    const int64_t inHeight,
    const int64_t outChannel,
    const int64_t outWidth,
    const int64_t outHeight,
    const int64_t kernWidth,
    const int64_t kernHeight,
    const int64_t inChannelGroup,
    const int64_t outChannelGroup,
    const int64_t strideHeight,
    const int64_t strideWidth,
    const int64_t padHeight,
    const int64_t padWidth,
    const int64_t dilationHeight,
    const int64_t dilationWidth
)
{
  for (int64_t n = 0; n < batch; n++) {
    for (int64_t g = 0; g < group; g++) {
      const int64_t inGroupOffset   = g * inChannelGroup * inHeight * inWidth;
      const int64_t outGroupOffset  = g * outChannelGroup * outHeight * outWidth;
      const int64_t biasGroupOffset = g * outChannelGroup;
      const int64_t kernGroupOffset = g * outChannelGroup * inChannelGroup * kernHeight * kernWidth;

      const int64_t remain = outChannelGroup & 0xf ;

      int k = 0 ;
      switch( remain ) {
      case 1 :
	selector<FLAYOUT,ADDBIAS,1>(
	   pIn, pKernel, pBias, pOut,
	   inChannel, inWidth, inHeight,
	   outChannel, outWidth, outHeight,
	   kernWidth, kernHeight,
	   inChannelGroup, outChannelGroup,
	   strideHeight, strideWidth,
	   padHeight, padWidth,
	   dilationHeight, dilationWidth,
	   inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	   n, k ) ;
	k+=1 ;
	break ;
      case 2 :
	selector<FLAYOUT,ADDBIAS,2>(
	   pIn, pKernel, pBias, pOut,
	   inChannel, inWidth, inHeight,
	   outChannel, outWidth, outHeight,
	   kernWidth, kernHeight,
	   inChannelGroup, outChannelGroup,
	   strideHeight, strideWidth,
	   padHeight, padWidth,
	   dilationHeight, dilationWidth,
	   inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	   n, k ) ;
	k+=2 ;
	break ;
      case 3 :
	selector<FLAYOUT,ADDBIAS,3>(
	   pIn, pKernel, pBias, pOut,
	   inChannel, inWidth, inHeight,
	   outChannel, outWidth, outHeight,
	   kernWidth, kernHeight,
	   inChannelGroup, outChannelGroup,
	   strideHeight, strideWidth,
	   padHeight, padWidth,
	   dilationHeight, dilationWidth,
	   inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	   n, k ) ;
	k+=3 ;
	break ;
      case 4 :
	selector<FLAYOUT,ADDBIAS,4>(
	   pIn, pKernel, pBias, pOut,
	   inChannel, inWidth, inHeight,
	   outChannel, outWidth, outHeight,
	   kernWidth, kernHeight,
	   inChannelGroup, outChannelGroup,
	   strideHeight, strideWidth,
	   padHeight, padWidth,
	   dilationHeight, dilationWidth,
	   inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	   n, k ) ;
	k+=4 ;
	break ;
      case 5 :
	selector<FLAYOUT,ADDBIAS,5>(
	   pIn, pKernel, pBias, pOut,
	   inChannel, inWidth, inHeight,
	   outChannel, outWidth, outHeight,
	   kernWidth, kernHeight,
	   inChannelGroup, outChannelGroup,
	   strideHeight, strideWidth,
	   padHeight, padWidth,
	   dilationHeight, dilationWidth,
	   inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	   n, k ) ;
	k+=5 ;
	break ;
      case 6 :
	selector<FLAYOUT,ADDBIAS,6>(
	   pIn, pKernel, pBias, pOut,
	   inChannel, inWidth, inHeight,
	   outChannel, outWidth, outHeight,
	   kernWidth, kernHeight,
	   inChannelGroup, outChannelGroup,
	   strideHeight, strideWidth,
	   padHeight, padWidth,
	   dilationHeight, dilationWidth,
	   inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	   n, k ) ;
	k+=6 ;
	break ;
      case 7 :
	selector<FLAYOUT,ADDBIAS,7>(
	   pIn, pKernel, pBias, pOut,
	   inChannel, inWidth, inHeight,
	   outChannel, outWidth, outHeight,
	   kernWidth, kernHeight,
	   inChannelGroup, outChannelGroup,
	   strideHeight, strideWidth,
	   padHeight, padWidth,
	   dilationHeight, dilationWidth,
	   inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	   n, k ) ;
	k+=7 ;
	break ;
      case 8 :
	selector<FLAYOUT,ADDBIAS,8>(
	   pIn, pKernel, pBias, pOut,
	   inChannel, inWidth, inHeight,
	   outChannel, outWidth, outHeight,
	   kernWidth, kernHeight,
	   inChannelGroup, outChannelGroup,
	   strideHeight, strideWidth,
	   padHeight, padWidth,
	   dilationHeight, dilationWidth,
	   inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	   n, k ) ;
	k+=8 ;
	break ;
      case 9 :
	selector<FLAYOUT,ADDBIAS,9>(
	   pIn, pKernel, pBias, pOut,
	   inChannel, inWidth, inHeight,
	   outChannel, outWidth, outHeight,
	   kernWidth, kernHeight,
	   inChannelGroup, outChannelGroup,
	   strideHeight, strideWidth,
	   padHeight, padWidth,
	   dilationHeight, dilationWidth,
	   inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	   n, k ) ;
	k+=9 ;
	break ;
      case 10 :
	selector<FLAYOUT,ADDBIAS,10>(pIn, pKernel, pBias, pOut,
	   inChannel, inWidth, inHeight,
	   outChannel, outWidth, outHeight,
	   kernWidth, kernHeight,
	   inChannelGroup, outChannelGroup,
	   strideHeight, strideWidth,
	   padHeight, padWidth,
	   dilationHeight, dilationWidth,
	   inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	   n, k ) ;
	k+=10 ;
	break ;
      case 11 :
	selector<FLAYOUT,ADDBIAS,11>(
	   pIn, pKernel, pBias, pOut,
	   inChannel, inWidth, inHeight,
	   outChannel, outWidth, outHeight,
	   kernWidth, kernHeight,
	   inChannelGroup, outChannelGroup,
	   strideHeight, strideWidth,
	   padHeight, padWidth,
	   dilationHeight, dilationWidth,
	   inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	   n, k ) ;;
	k+=11 ;
	break ;
      case 12 :
	selector<FLAYOUT,ADDBIAS,12>(
	   pIn, pKernel, pBias, pOut,
	   inChannel, inWidth, inHeight,
	   outChannel, outWidth, outHeight,
	   kernWidth, kernHeight,
	   inChannelGroup, outChannelGroup,
	   strideHeight, strideWidth,
	   padHeight, padWidth,
	   dilationHeight, dilationWidth,
	   inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	   n, k ) ;
	k+=12 ;
	break ;
      case 13 :
	selector<FLAYOUT,ADDBIAS,13>(
	   pIn, pKernel, pBias, pOut,
	   inChannel, inWidth, inHeight,
	   outChannel, outWidth, outHeight,
	   kernWidth, kernHeight,
	   inChannelGroup, outChannelGroup,
	   strideHeight, strideWidth,
	   padHeight, padWidth,
	   dilationHeight, dilationWidth,
	   inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	   n, k ) ;
	k+=13 ;
	break ;
      case 14 :
	selector<FLAYOUT,ADDBIAS,14>(
	   pIn, pKernel, pBias, pOut,
	   inChannel, inWidth, inHeight,
	   outChannel, outWidth, outHeight,
	   kernWidth, kernHeight,
	   inChannelGroup, outChannelGroup,
	   strideHeight, strideWidth,
	   padHeight, padWidth,
	   dilationHeight, dilationWidth,
	   inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	   n, k ) ;
	k+=14 ;
	break ;
      case 15 :
	selector<FLAYOUT,ADDBIAS,15>(
	   pIn, pKernel, pBias, pOut,
	   inChannel, inWidth, inHeight,
	   outChannel, outWidth, outHeight,
	   kernWidth, kernHeight,
	   inChannelGroup, outChannelGroup,
	   strideHeight, strideWidth,
	   padHeight, padWidth,
	   dilationHeight, dilationWidth,
	   inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	   n, k ) ;
	k+=15 ;
	break ;
      default :
	break ;
      }
      for (; k < outChannelGroup; k+=16) {
	selector<FLAYOUT,ADDBIAS,16>(
	   pIn, pKernel, pBias, pOut,
	   inChannel, inWidth, inHeight,
	   outChannel, outWidth, outHeight,
	   kernWidth, kernHeight,
	   inChannelGroup, outChannelGroup,
	   strideHeight, strideWidth,
	   padHeight, padWidth,
	   dilationHeight, dilationWidth,
	   inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	   n, k ) ;
      } // outChannel
    } // group
  } // batch
}

extern "C" vednnError_t
vednnConvolutionForward_direct_vecC_dil1_pad0_ker1_cU1024(
    const vednnTensorParam_t *  	pParamIn,
    const void *  			pDataIn,
    const vednnFilterParam_t *  	pParamKernel,
    const void *  			pDataKernel,
    const vednnBiasParam_t * 		pParamBias,
    const void * 			pDataBias,
    const vednnConvolutionParam_t *  	pParamConv,
    const vednnTensorParam_t *  	pParamOut,
    void *  				pDataOut
)
{
  const int64_t batch      = pParamIn->batch;
  const int64_t inChannel  = pParamIn->channel;
  const int64_t inWidth    = pParamIn->width;
  const int64_t inHeight   = pParamIn->height;
  const int64_t outChannel = pParamOut->channel;
  const int64_t outWidth   = pParamOut->width;
  const int64_t outHeight  = pParamOut->height;
  const int64_t kernWidth  = pParamKernel->width;		// 1
  const int64_t kernHeight = pParamKernel->height;		// 1

  const int64_t filter_layout = pParamKernel->layout ;

  const int64_t group          = pParamConv->group;
  const int64_t strideWidth    = pParamConv->strideWidth;;
  const int64_t strideHeight   = pParamConv->strideHeight;
  const int64_t padWidth       = pParamConv->padWidth;		// 0
  const int64_t padHeight      = pParamConv->padHeight;		// 0
  const int64_t dilationWidth  = pParamConv->dilationWidth;	// 1
  const int64_t dilationHeight = pParamConv->dilationHeight;	// 1

  const int64_t inChannelGroup  = inChannel  / group;   // equal to pDataKernel->inChannel
  const int64_t outChannelGroup = outChannel / group;   // equal to pDataKernel->outChannel

  const float * pIn     = (const float *) pDataIn;
  const float * pKernel = (const float *) pDataKernel;
  const float * pBias   = (const float *) pDataBias;
  float * const pOut    = (float * const) pDataOut;

  if( filter_layout == VEDNN_FILTER_LAYOUT_NCHW) {
    if( pDataBias == NULL ) {
      convloop<VEDNN_FILTER_LAYOUT_NCHW, false>(pIn, pKernel, pBias, pOut,
		   batch, group,
		   inChannel, inWidth, inHeight,
		   outChannel, outWidth, outHeight,
		   kernWidth, kernHeight,
		   inChannelGroup, outChannelGroup,
		   strideHeight, strideWidth,
		   padHeight, padWidth,
		   dilationHeight, dilationWidth ) ;
    }
    else {
      convloop<VEDNN_FILTER_LAYOUT_NCHW, true>(pIn, pKernel, pBias, pOut,
		   batch, group,
		   inChannel, inWidth, inHeight,
		   outChannel, outWidth, outHeight,
		   kernWidth, kernHeight,
		   inChannelGroup, outChannelGroup,
		   strideHeight, strideWidth,
		   padHeight, padWidth,
		   dilationHeight, dilationWidth ) ;
    }
  }
  else {
    if( pDataBias == NULL ) {
      convloop<VEDNN_FILTER_LAYOUT_HWCN, false>(pIn, pKernel, pBias, pOut,
		   batch, group,
		   inChannel, inWidth, inHeight,
		   outChannel, outWidth, outHeight,
		   kernWidth, kernHeight,
		   inChannelGroup, outChannelGroup,
		   strideHeight, strideWidth,
		   padHeight, padWidth,
		   dilationHeight, dilationWidth ) ;
    }
    else {
      convloop<VEDNN_FILTER_LAYOUT_HWCN, true>(pIn, pKernel, pBias, pOut,
		   batch, group,
		   inChannel, inWidth, inHeight,
		   outChannel, outWidth, outHeight,
		   kernWidth, kernHeight,
		   inChannelGroup, outChannelGroup,
		   strideHeight, strideWidth,
		   padHeight, padWidth,
		   dilationHeight, dilationWidth ) ;
    }
  }


  return VEDNN_SUCCESS;
}
