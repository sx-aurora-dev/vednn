#include <stdint.h>
#include <stdlib.h>

#include "vednn.h"

#include "velintrin.h"
#define VLEN	(256)

template<filterLayout_t FLAYOUT, int NUMKERNEL, int R, int S>
static inline void func(
    const float * __restrict__ pIn,
    const float * __restrict__ pGOut,
    float * __restrict__ const pGKernel,
    const int64_t gOutChannel,
    const int64_t gOutWidth,
    const int64_t gOutHeight,
    const int64_t inChannel,
    const int64_t inWidth,
    const int64_t inHeight,
    const int64_t gKernWidth,
    const int64_t gKernHeight,
    const int64_t strideWidth,
    const int64_t strideHeight,
    const int64_t inChannelGroup,
    const int64_t gOutChannelGroup,
    const int64_t inGroupOffset,
    const int64_t gOutGroupOffset,
    const int64_t gKernGroupOffset,
    const int64_t batch,
    const int64_t k,
    const int64_t r,
    const int64_t s
)
{
  const int64_t remain  = NUMKERNEL & 0x1 ;
  const int64_t nPacked = NUMKERNEL >> 1 ;

  for (int64_t c=0; c<inChannelGroup; c++) {

    __vr vrsum0[R*S] ;
    __vr vrsum[nPacked*R*S] ;
#pragma clang loop unroll(full)
    for(int64_t rs=0; rs<R*S; rs++) {
      vrsum0[rs] = _vel_vbrds_vsl(0.f, VLEN) ;
#pragma clang loop unroll(full)
      for(int64_t kk=0; kk<nPacked; kk++) {
	vrsum[kk*R*S+rs] = _vel_pvbrd_vsl(0UL, VLEN) ;
      }
    }

    for (int64_t n=0; n<batch; n++) {
      for (int64_t y = 0; y < gOutHeight ; y ++ ) {
	for (int64_t x = 0; x < gOutWidth ; x += VLEN ) {
	  const int64_t vl = gOutWidth - x < VLEN ? gOutWidth - x  : VLEN ;

	  const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;

	  const int64_t outIndex = gOutGroupOffset + ((n * gOutChannel + k) * gOutHeight + y) * gOutWidth + x  ;


	  __vr vrin[R*S] ;
#pragma clang loop unroll(full)
	  for(int64_t rr=0; rr<R; rr++) {
#pragma clang loop unroll(full)
	    for(int64_t ss=0; ss<S; ss++) {
	      vrin[rr*S+ss] = _vel_vldu_vssl(4*strideWidth,&pInChannel[(y*strideHeight+r+rr)*inWidth+x*strideWidth+s+ss], vl) ;
	    }
	  }

	  if( nPacked > 0 ) {
#pragma clang loop unroll(full)
	    for(int64_t rr=0; rr<R; rr++) {
#pragma clang loop unroll(full)
	      for(int64_t ss=0; ss<S; ss++) {
		vrin[rr*S+ss] = _vel_vshf_vvvsl(vrin[rr*S+ss], vrin[rr*S+ss], VE_VSHUFFLE_YUZU, vl) ;
	      }
	    }
	  }

	  __vr vrgout[NUMKERNEL] ;
#pragma clang loop unroll(full)
	  for(int64_t kk=0; kk<NUMKERNEL; kk++) {
	    vrgout[kk] = _vel_vldu_vssl(4, pGOut+outIndex+kk*gOutHeight*gOutWidth, vl) ;
	  }

	  __vr vrgoutp[NUMKERNEL]  ;
#pragma clang loop unroll(full)
	  for(int64_t kk=0; kk<nPacked; kk++) {
	    vrgoutp[kk] = _vel_vshf_vvvsl(vrgout[2*kk+remain], vrgout[2*kk+remain+1], VE_VSHUFFLE_YUZU, vl) ;
	  }

#pragma clang loop unroll(full)
	  for(int64_t rs=0; rs<R*S; rs++) {
	    if( remain ) {
	      vrsum0[rs]  = _vel_vfmads_vvvvvl(vrsum0[rs], vrin[rs], vrgout[0], vrsum0[rs], vl) ;
	    }
#pragma clang loop unroll(full)
	    for(int64_t kk=0; kk<nPacked; kk++) {
	      vrsum[kk*R*S+rs] = _vel_pvfmad_vvvvvl(vrsum[kk*R*S+rs], vrin[rs], vrgoutp[kk], vrsum[kk*R*S+rs], vl) ;
	    }
	  }
	} // gOutWidth
      } // gOutHeight
    } // batch

#define FILTER_OFFSET(k,c,r,s) ( gKernGroupOffset + filter_index<FLAYOUT>(k,c,r,s, inChannelGroup, gOutChannelGroup, gKernHeight, gKernWidth) )

#pragma clang loop unroll(full)
    for(int64_t rr=0; rr<R; rr++) {
#pragma clang loop unroll(full)
      for(int64_t ss=0; ss<S; ss++) {
	int64_t rs = rr*S+ss ;
	if( remain ) {
	  vrsum0[rs] = _vel_vfsums_vvl(vrsum0[rs], VLEN) ;
	  _vel_vstu_vssl(vrsum0[rs], 4, pGKernel+FILTER_OFFSET(k+0,c,r+rr,s+ss), 1) ;
	}

#pragma clang loop unroll(full)
	for(int64_t kk=0; kk<nPacked; kk++) {
	  __vr vrsumU = _vel_vfsums_vvl(vrsum[kk*R*S+rs], VLEN) ;
	  _vel_vstu_vssl(vrsumU, 4, pGKernel + FILTER_OFFSET(k+2*kk+remain,  c,r+rr,s+ss), 1) ;
	  __vr vrsumL = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsum[kk*R*S+rs],32, VLEN), VLEN);
	  _vel_vstu_vssl(vrsumL, 4, pGKernel + FILTER_OFFSET(k+2*kk+remain+1,c,r+rr,s+ss), 1) ;
	}
      }
    }

#undef FILTER_OFFSET
  } // inChannel
}

template<filterLayout_t FLAYOUT, int NUMKERNEL>
static inline void RSLoop(
    const float * __restrict__ pIn,
    const float * __restrict__ pGOut,
    float * __restrict__ const pGKernel,
    const int64_t gOutChannel,
    const int64_t gOutWidth,
    const int64_t gOutHeight,
    const int64_t inChannel,
    const int64_t inWidth,
    const int64_t inHeight,
    const int64_t gKernWidth,
    const int64_t gKernHeight,
    const int64_t strideWidth,
    const int64_t strideHeight,
    const int64_t inChannelGroup,
    const int64_t gOutChannelGroup,
    const int64_t inGroupOffset,
    const int64_t gOutGroupOffset,
    const int64_t gKernGroupOffset,
    const int64_t batch,
    const int64_t k )
{
  int64_t r = 0;

  switch(gKernHeight % 3) {
  case 1 :
    {
      int64_t s = 0;
      switch( gKernWidth % 3 ) {
      case 1 :
	func<FLAYOUT, NUMKERNEL, 1, 1>(pIn, pGOut, pGKernel,
	   gOutChannel, gOutWidth, gOutHeight,
	   inChannel, inWidth, inHeight,
	   gKernWidth, gKernHeight,
	   strideWidth, strideHeight,
	   inChannelGroup, gOutChannelGroup,
	   inGroupOffset, gOutGroupOffset, gKernGroupOffset,
	   batch, k, r, s) ;
	s+=1 ;
	break ;
      case 2 :
	func<FLAYOUT, NUMKERNEL, 1, 2>(pIn, pGOut, pGKernel,
	   gOutChannel, gOutWidth, gOutHeight,
	   inChannel, inWidth, inHeight,
	   gKernWidth, gKernHeight,
	   strideWidth, strideHeight,
	   inChannelGroup, gOutChannelGroup,
	   inGroupOffset, gOutGroupOffset, gKernGroupOffset,
	   batch, k, r, s) ;
	s+=2 ;
	break ;
      default : ;
      }
      for (; s<gKernWidth; s+=3) {
	func<FLAYOUT, NUMKERNEL, 1, 3>(pIn, pGOut, pGKernel,
	   gOutChannel, gOutWidth, gOutHeight,
	   inChannel, inWidth, inHeight,
	   gKernWidth, gKernHeight,
	   strideWidth, strideHeight,
	   inChannelGroup, gOutChannelGroup,
	   inGroupOffset, gOutGroupOffset, gKernGroupOffset,
	   batch, k, r, s) ;
      }
      r+=1 ;
    }
    break ;
  case 2 :
    {
      int64_t s = 0;
      switch( gKernWidth % 3 ) {
      case 1 :
	func<FLAYOUT, NUMKERNEL, 2, 1>(pIn, pGOut, pGKernel,
	   gOutChannel, gOutWidth, gOutHeight,
	   inChannel, inWidth, inHeight,
	   gKernWidth, gKernHeight,
	   strideWidth, strideHeight,
	   inChannelGroup, gOutChannelGroup,
	   inGroupOffset, gOutGroupOffset, gKernGroupOffset,
	   batch, k, r, s) ;
	s+=1 ;
	break ;
      case 2 :
	func<FLAYOUT, NUMKERNEL, 2, 2>(pIn, pGOut, pGKernel,
	   gOutChannel, gOutWidth, gOutHeight,
	   inChannel, inWidth, inHeight,
	   gKernWidth, gKernHeight,
	   strideWidth, strideHeight,
	   inChannelGroup, gOutChannelGroup,
	   inGroupOffset, gOutGroupOffset, gKernGroupOffset,
	   batch, k, r, s) ;
	s+=2 ;
	break ;
      default : ;
      }
      for (; s<gKernWidth; s+=3) {
	func<FLAYOUT, NUMKERNEL, 2, 3>(pIn, pGOut, pGKernel,
	   gOutChannel, gOutWidth, gOutHeight,
	   inChannel, inWidth, inHeight,
	   gKernWidth, gKernHeight,
	   strideWidth, strideHeight,
	   inChannelGroup, gOutChannelGroup,
	   inGroupOffset, gOutGroupOffset, gKernGroupOffset,
	   batch, k, r, s) ;
      }
      r+=2 ;
    }
    break ;
  default :
    break ;
  }
  for(; r<gKernHeight; r+=3) {
    int64_t s = 0;
    switch( gKernWidth % 3 ) {
    case 1 :
      func<FLAYOUT, NUMKERNEL, 3, 1>(pIn, pGOut, pGKernel,
	 gOutChannel, gOutWidth, gOutHeight,
	 inChannel, inWidth, inHeight,
	 gKernWidth, gKernHeight,
	 strideWidth, strideHeight,
	 inChannelGroup, gOutChannelGroup,
	 inGroupOffset, gOutGroupOffset, gKernGroupOffset,
	 batch, k, r, s) ;
      s+=1 ;
      break ;
    case 2 :
      func<FLAYOUT, NUMKERNEL, 3, 2>(pIn, pGOut, pGKernel,
	 gOutChannel, gOutWidth, gOutHeight,
	 inChannel, inWidth, inHeight,
	 gKernWidth, gKernHeight,
	 strideWidth, strideHeight,
	 inChannelGroup, gOutChannelGroup,
	 inGroupOffset, gOutGroupOffset, gKernGroupOffset,
	 batch, k, r, s) ;
      s+=2 ;
      break ;
    default : ;
    }
    for (; s<gKernWidth; s+=3) {
      func<FLAYOUT, NUMKERNEL, 3, 3>(pIn, pGOut, pGKernel,
	 gOutChannel, gOutWidth, gOutHeight,
	 inChannel, inWidth, inHeight,
	 gKernWidth, gKernHeight,
	 strideWidth, strideHeight,
	 inChannelGroup, gOutChannelGroup,
	 inGroupOffset, gOutGroupOffset, gKernGroupOffset,
	 batch, k, r, s) ;
    }
  }
}

template<filterLayout_t FLAYOUT>
static inline void convloop(
    const float * __restrict__ pIn,
    const float * __restrict__ pGOut,
    float * __restrict__ const pGKernel,
    const int64_t batch,
    const int64_t group,
    const int64_t inChannel,
    const int64_t inWidth,
    const int64_t inHeight,
    const int64_t gOutChannel,
    const int64_t gOutWidth,
    const int64_t gOutHeight,
    const int64_t gKernWidth,
    const int64_t gKernHeight,
    const int64_t inChannelGroup,
    const int64_t gOutChannelGroup,
    const int64_t beginOChannel,
    const int64_t nOChannel,
    const int64_t strideWidth,
    const int64_t strideHeight,
    const int64_t padWidth,
    const int64_t padHeight,
    const int64_t dilationWidth,
    const int64_t dilationHeight
)
{
  for (int64_t g = 0; g < group; g++) {
    int64_t inGroupOffset    = g * inChannelGroup  * inHeight  * inWidth;
    int64_t gOutGroupOffset  = g * gOutChannelGroup * gOutHeight * gOutWidth;
    int64_t gKernGroupOffset = g * gOutChannelGroup * inChannelGroup * gKernHeight * gKernWidth;

    const int64_t remain = nOChannel & 0x7 ;


    int64_t k=0;
    switch(remain) {
    case 1:
      RSLoop<FLAYOUT, 1>(pIn, pGOut, pGKernel,
	 gOutChannel, gOutWidth, gOutHeight,
	 inChannel, inWidth, inHeight,
	 gKernWidth, gKernHeight,
	 strideWidth, strideHeight,
	 inChannelGroup, gOutChannelGroup,
	 inGroupOffset, gOutGroupOffset, gKernGroupOffset,
	 batch, beginOChannel + k ) ;
      k+=1 ;
      break ;
    case 2:
      RSLoop<FLAYOUT, 2>(pIn, pGOut, pGKernel,
	 gOutChannel, gOutWidth, gOutHeight,
	 inChannel, inWidth, inHeight,
	 gKernWidth, gKernHeight,
	 strideWidth, strideHeight,
	 inChannelGroup, gOutChannelGroup,
	 inGroupOffset, gOutGroupOffset, gKernGroupOffset,
	 batch, beginOChannel + k ) ;
      k+=2 ;
      break ;
    case 3:
      RSLoop<FLAYOUT, 3>(pIn, pGOut, pGKernel,
	 gOutChannel, gOutWidth, gOutHeight,
	 inChannel, inWidth, inHeight,
	 gKernWidth, gKernHeight,
	 strideWidth, strideHeight,
	 inChannelGroup, gOutChannelGroup,
	 inGroupOffset, gOutGroupOffset, gKernGroupOffset,
	 batch, beginOChannel + k ) ;
      k+=3 ;
      break ;
    case 4:
      RSLoop<FLAYOUT, 4>(pIn, pGOut, pGKernel,
	 gOutChannel, gOutWidth, gOutHeight,
	 inChannel, inWidth, inHeight,
	 gKernWidth, gKernHeight,
	 strideWidth, strideHeight,
	 inChannelGroup, gOutChannelGroup,
	 inGroupOffset, gOutGroupOffset, gKernGroupOffset,
	 batch, beginOChannel + k ) ;
      k+=4 ;
      break ;
    case 5:
      RSLoop<FLAYOUT, 5>(pIn, pGOut, pGKernel,
	 gOutChannel, gOutWidth, gOutHeight,
	 inChannel, inWidth, inHeight,
	 gKernWidth, gKernHeight,
	 strideWidth, strideHeight,
	 inChannelGroup, gOutChannelGroup,
	 inGroupOffset, gOutGroupOffset, gKernGroupOffset,
	 batch, beginOChannel + k ) ;
      k+=5 ;
      break ;
    case 6:
      RSLoop<FLAYOUT, 6>(pIn, pGOut, pGKernel,
	 gOutChannel, gOutWidth, gOutHeight,
	 inChannel, inWidth, inHeight,
	 gKernWidth, gKernHeight,
	 strideWidth, strideHeight,
	 inChannelGroup, gOutChannelGroup,
	 inGroupOffset, gOutGroupOffset, gKernGroupOffset,
	 batch, beginOChannel + k ) ;
      k+=6 ;
      break ;
    case 7:
      RSLoop<FLAYOUT, 7>(pIn, pGOut, pGKernel,
	 gOutChannel, gOutWidth, gOutHeight,
	 inChannel, inWidth, inHeight,
	 gKernWidth, gKernHeight,
	 strideWidth, strideHeight,
	 inChannelGroup, gOutChannelGroup,
	 inGroupOffset, gOutGroupOffset, gKernGroupOffset,
	 batch, beginOChannel + k ) ;
      k+=7 ;
      break ;
    default :
      break ;
    }
    for (; k<nOChannel; ) {
      RSLoop<FLAYOUT, 8>(pIn, pGOut, pGKernel,
	 gOutChannel, gOutWidth, gOutHeight,
	 inChannel, inWidth, inHeight,
	 gKernWidth, gKernHeight,
	 strideWidth, strideHeight,
	 inChannelGroup, gOutChannelGroup,
	 inGroupOffset, gOutGroupOffset, gKernGroupOffset,
	 batch, beginOChannel + k ) ;
      k+=8 ;
    } // gOutChannel
  } // group
}


extern "C" vednnError_t
vednnConvolutionBackwardFilter_direct_dil1_pad0(
    const vednnTensorParam_t * restrict 	pParamIn,
    const void * restrict 			pDataIn,
    const vednnTensorParam_t * restrict 	pParamGradOut,
    const void * restrict 			pDataGradOut,
    const vednnConvolutionParam_t * restrict 	pParamConv,
    const vednnFilterParam_t * restrict 	pParamGradKernel,
    void * restrict 				pDataGradKernel
#ifdef VEDNN_USE_OPENMP
    ,
    const int64_t				beginOChannel,
    const int64_t				nOChannel
#endif
)
{
  const int64_t inChannel   = pParamIn->channel;
  const int64_t inWidth     = pParamIn->width;
  const int64_t inHeight    = pParamIn->height;
  const int64_t batch       = pParamGradOut->batch;
  const int64_t gOutChannel = pParamGradOut->channel;
  const int64_t gOutWidth   = pParamGradOut->width;
  const int64_t gOutHeight  = pParamGradOut->height;
  const int64_t gKernWidth  = pParamGradKernel->width;
  const int64_t gKernHeight = pParamGradKernel->height;

  const int64_t filter_layout = pParamGradKernel->layout ;

  const int64_t group          = pParamConv->group;
  const int64_t strideWidth    = pParamConv->strideWidth;;
  const int64_t strideHeight   = pParamConv->strideHeight;
  const int64_t padWidth       = pParamConv->padWidth;
  const int64_t padHeight      = pParamConv->padHeight;
  const int64_t dilationWidth  = pParamConv->dilationWidth;
  const int64_t dilationHeight = pParamConv->dilationHeight;

  const int64_t inChannelGroup   =  inChannel   / group;
  const int64_t gOutChannelGroup = gOutChannel  / group;

  const float * pIn      = (const float *) pDataIn;
  const float * pGOut    = (const float *) pDataGradOut;
  float * const pGKernel = (float * const) pDataGradKernel;

#ifndef VEDNN_USE_OPENMP
  const int64_t beginOChannel = 0 ;
  const int64_t nOChannel     = gOutChannelGroup ;
#endif

  if( filter_layout == VEDNN_FILTER_LAYOUT_NCHW) {
    convloop<VEDNN_FILTER_LAYOUT_NCHW>(pIn, pGOut, pGKernel,
	       batch, group,
	       inChannel, inWidth, inHeight,
	       gOutChannel, gOutWidth, gOutHeight,
	       gKernWidth, gKernHeight,
	       inChannelGroup, gOutChannelGroup,
	       beginOChannel, nOChannel,
	       strideWidth, strideHeight,
	       padWidth, padHeight,
	       dilationWidth, dilationHeight) ;
  }
  else {
    convloop<VEDNN_FILTER_LAYOUT_HWCN>(pIn, pGOut, pGKernel,
	       batch, group,
	       inChannel, inWidth, inHeight,
	       gOutChannel, gOutWidth, gOutHeight,
	       gKernWidth, gKernHeight,
	       inChannelGroup, gOutChannelGroup,
	       beginOChannel, nOChannel,
	       strideWidth, strideHeight,
	       padWidth, padHeight,
	       dilationWidth, dilationHeight) ;
  }

  return VEDNN_SUCCESS;
}
