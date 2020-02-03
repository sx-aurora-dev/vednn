#include <stdint.h>
#include <stdlib.h>

#include "vednn.h"

#include "velintrin.h"
#define VLEN	(256)


template<filterLayout_t FLAYOUT, int NUMCHANNEL>
static inline void func(
    const float * __restrict__ pGOut,
    const float * __restrict__ pKernel,
    float * __restrict__ const pGIn,
    const int64_t gOutChannel,
    const int64_t gOutWidth,
    const int64_t gOutHeight,
    const int64_t gInChannel,
    const int64_t gInWidth,
    const int64_t gInHeight,
    const int64_t gInChannelGroup,
    const int64_t gOutChannelGroup,
    const int64_t gInGroupOffset,
    const int64_t gOutGroupOffset,
    const int64_t kernGroupOffset,
    const int64_t n,
    const int64_t c
)
{
  const int64_t remain  = NUMCHANNEL & 0x1 ;
  const int64_t nPacked = NUMCHANNEL >> 1 ;

  int64_t gInIndex = gInGroupOffset + ((n * gInChannel + c) * gInHeight ) * gInWidth ;

  for (int64_t gip = 0; gip < gInHeight * gInWidth ; gip+=VLEN) {
    const int64_t vl = gInHeight * gInWidth - gip < VLEN ? gInHeight * gInWidth - gip : VLEN ;

    __vr vrsum0  = _vel_vbrds_vsl(0.f, vl) ;
    __vr vrsum[nPacked] ;
#pragma clang loop unroll(full)
    for(int64_t cc=0; cc<nPacked; cc++) {
	vrsum[cc] = _vel_pvbrd_vsl(0UL, vl) ;
    }

    for (int64_t k=0; k<gOutChannelGroup; k++) {

      const float *pGOutChannel = pGOut + gOutGroupOffset + ((n * gOutChannel + k) * gOutHeight * gOutWidth ) ;

      /* memory access errors might be caused */
      __vr vrgout = _vel_vldu_vssl(4,pGOutChannel+gip, vl) ;

#define FILTER_OFFSET(k,c,r,s) ( kernGroupOffset + filter_index<FLAYOUT>(k,c,r,s, gInChannelGroup, gOutChannelGroup, 1, 1) )

#define VFADD(VRGOUT,K,R,S) {									\
	__vr vrgoutP = _vel_vshf_vvvsl(VRGOUT, VRGOUT, VE_VSHUFFLE_YUZU, vl) ;			\
	if( remain ) {										\
	  const float    kerValue0  = pKernel[FILTER_OFFSET(K,c+ 0,R,S)] ;			\
	  vrsum0 = _vel_vfmads_vvsvl(vrsum0, kerValue0, VRGOUT, vl) ;				\
	}											\
	_Pragma("clang loop unroll(full)")							\
	for(int64_t cc=0; cc<nPacked; cc++) {							\
	  const uint64_t kerValue = _vel_pack_f32p(pKernel + FILTER_OFFSET(K,c+2*cc+remain,  R,S),	\
						   pKernel + FILTER_OFFSET(K,c+2*cc+remain+1,R,S)) ;	\
	  vrsum[cc] = _vel_pvfmad_vvsvl(vrsum[cc], kerValue, vrgoutP, vl) ;			\
	}											\
      }

      VFADD(vrgout, k+0, 0, 0) ;

#undef VFADD
#undef FILTER_OFFSET
    } // gInChannel

    if(remain) {
	_vel_vstu_vssl(vrsum0, 4, pGIn+gInIndex + 0 * gInHeight * gInWidth, vl) ;
    }
#pragma clang loop unroll(full)
    for(int64_t cc=0; cc<nPacked; cc++) {
      _vel_vstu_vssl(vrsum[cc], 4, pGIn+gInIndex + (2*cc+remain)   * gInHeight * gInWidth, vl) ;
      _vel_vstl_vssl(vrsum[cc], 4, pGIn+gInIndex + (2*cc+remain+1) * gInHeight * gInWidth, vl) ;
    }

    gInIndex += vl ;

  } // gOutPixels
}


template<int NUMCHANNEL>
static inline void func_even_filter_nchwaligned(
    const float * __restrict__ pGOut,
    const float * __restrict__ pKernel,
    float * __restrict__ const pGIn,
    const int64_t gOutChannel,
    const int64_t gOutWidth,
    const int64_t gOutHeight,
    const int64_t gInChannel,
    const int64_t gInWidth,
    const int64_t gInHeight,
    const int64_t gInChannelGroup,
    const int64_t gOutChannelGroup,
    const int64_t gInGroupOffset,
    const int64_t gOutGroupOffset,
    const int64_t kernGroupOffset,
    const int64_t n,
    const int64_t c
)
{
  const int64_t nPacked = NUMCHANNEL >> 1 ;

  int64_t gInIndex = gInGroupOffset + ((n * gInChannel + c) * gInHeight ) * gInWidth ;

  for (int64_t gip = 0; gip < gInHeight * gInWidth ; gip+=VLEN) {
    const int64_t vl = gInHeight * gInWidth - gip < VLEN ? gInHeight * gInWidth - gip : VLEN ;

    __vr vrsum[nPacked] ;
#pragma clang loop unroll(full)
    for(int64_t cc=0; cc<nPacked; cc++) {
	vrsum[cc] = _vel_pvbrd_vsl(0UL, vl) ;
    }

    for (int64_t k=0 ; k<gOutChannelGroup; k++) {
      const float *pGOutChannel = pGOut + gOutGroupOffset + ((n * gOutChannel + k) * gOutHeight * gOutWidth ) ;

      /* memory access errors might be caused */
      __vr vrgout_k0 = _vel_vldu_vssl(4,pGOutChannel+gip, vl) ;

#define FILTER_OFFSET(k,c,r,s) ( kernGroupOffset + filter_index<VEDNN_FILTER_LAYOUT_NCHW>(k,c,r,s, gInChannelGroup, gOutChannelGroup, 1, 1) )

#define VFADD(VRGOUT,K,R,S) {									\
        const uint64_t* kerValue_u64 = (uint64_t*)(pKernel + FILTER_OFFSET(K,c+ 0,R,S)) ;	\
	__vr vrgoutP = _vel_vshf_vvvsl(VRGOUT, VRGOUT, VE_VSHUFFLE_YUZU, vl) ;			\
	_Pragma("clang loop unroll(full)")							\
	for(int64_t cc=0; cc<nPacked; cc++) {							\
	  vrsum[cc] = _vel_pvfmad_vvsvl(vrsum[cc], kerValue_u64[cc], vrgoutP, vl) ;		\
	}											\
      }
      VFADD(vrgout_k0, k+0, 0, 0) ;

#undef VFADD
#undef FILTER_OFFSET
    } // gInChannel

#pragma clang loop unroll(full)
    for(int64_t cc=0; cc<nPacked; cc++) {
      _vel_vstl_vssl(vrsum[cc], 4, pGIn+gInIndex + (2*cc)   * gInHeight * gInWidth, vl) ;
      _vel_vstu_vssl(vrsum[cc], 4, pGIn+gInIndex + (2*cc+1) * gInHeight * gInWidth, vl) ;
    }

    gInIndex += vl ;

  } // gOutPixels
}

template<filterLayout_t FLAYOUT>
static inline void convloop(
    const float * __restrict__ pGOut,
    const float * __restrict__ pKernel,
    float * __restrict__ const pGIn,
    const int64_t batch,
    const int64_t group,
    const int64_t gOutChannel,
    const int64_t gOutWidth,
    const int64_t gOutHeight,
    const int64_t gInChannel,
    const int64_t gInWidth,
    const int64_t gInHeight,
    const int64_t gInChannelGroup,
    const int64_t gOutChannelGroup
)
{

  for (int64_t n=0; n<batch; n++) {
    for (int64_t g = 0; g < group; g++) {

      int64_t gInGroupOffset  = g * gInChannelGroup * gInHeight * gInWidth;
      int64_t gOutGroupOffset = g * gOutChannelGroup * gOutHeight * gOutWidth;
      int64_t kernGroupOffset = g * gOutChannelGroup * gInChannelGroup * 1 * 1;

      const int64_t remain = gInChannelGroup & 0xf ;

      int64_t c=0;
      switch(remain) {
      case 1:
	func<FLAYOUT, 1>(pGOut, pKernel, pGIn,
	   gOutChannel, gOutWidth, gOutHeight,
	   gInChannel, gInWidth, gInHeight,
	   gInChannelGroup, gOutChannelGroup,
	   gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	   n, c ) ;
	c+=1 ;
	break ;
      case 2:
	if( FLAYOUT == VEDNN_FILTER_LAYOUT_NCHW
	    && (((uint64_t)pKernel) & 0x07) == 0 && (gOutChannelGroup & 0x01) == 0 ) {
	  func_even_filter_nchwaligned<2>(pGOut, pKernel, pGIn,
	     gOutChannel, gOutWidth, gOutHeight,
	     gInChannel, gInWidth, gInHeight,
	     gInChannelGroup, gOutChannelGroup,
	     gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	     n, c) ;
	}
	else {
	  func<FLAYOUT, 2>(pGOut, pKernel, pGIn,
	     gOutChannel, gOutWidth, gOutHeight,
	     gInChannel, gInWidth, gInHeight,
	     gInChannelGroup, gOutChannelGroup,
	     gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	     n, c) ;
	}
	c+=2 ;
	break ;
      case 3:
	func<FLAYOUT, 3>(pGOut, pKernel, pGIn,
	   gOutChannel, gOutWidth, gOutHeight,
	   gInChannel, gInWidth, gInHeight,
	   gInChannelGroup, gOutChannelGroup,
	   gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	   n, c ) ;
	c+=3 ;
	break ;
      case 4:
	if( FLAYOUT == VEDNN_FILTER_LAYOUT_NCHW
	    && (((uint64_t)pKernel) & 0x07) == 0 && (gOutChannelGroup & 0x01) == 0 ) {
	  func_even_filter_nchwaligned<4>(pGOut, pKernel, pGIn,
	     gOutChannel, gOutWidth, gOutHeight,
	     gInChannel, gInWidth, gInHeight,
	     gInChannelGroup, gOutChannelGroup,
	     gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	     n, c) ;
	}
	else {
	  func<FLAYOUT, 4>(pGOut, pKernel, pGIn,
	     gOutChannel, gOutWidth, gOutHeight,
	     gInChannel, gInWidth, gInHeight,
	     gInChannelGroup, gOutChannelGroup,
	     gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	     n, c) ;
	}
	c+=4 ;
	break ;
      case 5:
	func<FLAYOUT, 5>(pGOut, pKernel, pGIn,
	   gOutChannel, gOutWidth, gOutHeight,
	   gInChannel, gInWidth, gInHeight,
	   gInChannelGroup, gOutChannelGroup,
	   gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	   n, c ) ;
	c+=5 ;
	break ;
      case 6:
	if( FLAYOUT == VEDNN_FILTER_LAYOUT_NCHW
	    && (((uint64_t)pKernel) & 0x07) == 0 && (gOutChannelGroup & 0x01) == 0 ) {
	  func_even_filter_nchwaligned<6>(pGOut, pKernel, pGIn,
	     gOutChannel, gOutWidth, gOutHeight,
	     gInChannel, gInWidth, gInHeight,
	     gInChannelGroup, gOutChannelGroup,
	     gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	     n, c) ;
	}
	else {
	  func<FLAYOUT, 6>(pGOut, pKernel, pGIn,
	     gOutChannel, gOutWidth, gOutHeight,
	     gInChannel, gInWidth, gInHeight,
	     gInChannelGroup, gOutChannelGroup,
	     gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	     n, c) ;
	}
	c+=6 ;
	break ;
      case 7:
	func<FLAYOUT, 7>(pGOut, pKernel, pGIn,
	   gOutChannel, gOutWidth, gOutHeight,
	   gInChannel, gInWidth, gInHeight,
	   gInChannelGroup, gOutChannelGroup,
	   gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	   n, c ) ;
	c+=7 ;
	break ;
      case 8:
	if( FLAYOUT == VEDNN_FILTER_LAYOUT_NCHW
	    && (((uint64_t)pKernel) & 0x07) == 0 && (gOutChannelGroup & 0x01) == 0 ) {
	  func_even_filter_nchwaligned<8>(pGOut, pKernel, pGIn,
	     gOutChannel, gOutWidth, gOutHeight,
	     gInChannel, gInWidth, gInHeight,
	     gInChannelGroup, gOutChannelGroup,
	     gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	     n, c) ;
	}
	else {
	  func<FLAYOUT, 8>(pGOut, pKernel, pGIn,
	     gOutChannel, gOutWidth, gOutHeight,
	     gInChannel, gInWidth, gInHeight,
	     gInChannelGroup, gOutChannelGroup,
	     gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	     n, c) ;
	}
	c+=8 ;
	break ;
      case 9:
	func<FLAYOUT, 9>(pGOut, pKernel, pGIn,
	   gOutChannel, gOutWidth, gOutHeight,
	   gInChannel, gInWidth, gInHeight,
	   gInChannelGroup, gOutChannelGroup,
	   gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	   n, c ) ;
	c+=9 ;
	break ;
      case 10:
	if( FLAYOUT == VEDNN_FILTER_LAYOUT_NCHW
	    && (((uint64_t)pKernel) & 0x07) == 0 && (gOutChannelGroup & 0x01) == 0 ) {
	  func_even_filter_nchwaligned<10>(pGOut, pKernel, pGIn,
	     gOutChannel, gOutWidth, gOutHeight,
	     gInChannel, gInWidth, gInHeight,
	     gInChannelGroup, gOutChannelGroup,
	     gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	     n, c) ;
	}
	else {
	  func<FLAYOUT, 10>(pGOut, pKernel, pGIn,
	     gOutChannel, gOutWidth, gOutHeight,
	     gInChannel, gInWidth, gInHeight,
	     gInChannelGroup, gOutChannelGroup,
	     gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	     n, c) ;
	}
	c+=10 ;
	break ;
      case 11:
	func<FLAYOUT, 11>(pGOut, pKernel, pGIn,
	   gOutChannel, gOutWidth, gOutHeight,
	   gInChannel, gInWidth, gInHeight,
	   gInChannelGroup, gOutChannelGroup,
	   gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	   n, c ) ;
	c+=11 ;
	break ;
      case 12:
	if( FLAYOUT == VEDNN_FILTER_LAYOUT_NCHW
	    && (((uint64_t)pKernel) & 0x07) == 0 && (gOutChannelGroup & 0x01) == 0 ) {
	  func_even_filter_nchwaligned<12>(pGOut, pKernel, pGIn,
	     gOutChannel, gOutWidth, gOutHeight,
	     gInChannel, gInWidth, gInHeight,
	     gInChannelGroup, gOutChannelGroup,
	     gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	     n, c) ;
	}
	else {
	  func<FLAYOUT, 12>(pGOut, pKernel, pGIn,
	     gOutChannel, gOutWidth, gOutHeight,
	     gInChannel, gInWidth, gInHeight,
	     gInChannelGroup, gOutChannelGroup,
	     gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	     n, c) ;
	}
	c+=12 ;
	break ;
      case 13:
	func<FLAYOUT, 13>(pGOut, pKernel, pGIn,
	   gOutChannel, gOutWidth, gOutHeight,
	   gInChannel, gInWidth, gInHeight,
	   gInChannelGroup, gOutChannelGroup,
	   gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	   n, c ) ;
	c+=13 ;
	break ;
      case 14:
	if( FLAYOUT == VEDNN_FILTER_LAYOUT_NCHW
	    && (((uint64_t)pKernel) & 0x07) == 0 && (gOutChannelGroup & 0x01) == 0 ) {
	  func_even_filter_nchwaligned<14>(pGOut, pKernel, pGIn,
	     gOutChannel, gOutWidth, gOutHeight,
	     gInChannel, gInWidth, gInHeight,
	     gInChannelGroup, gOutChannelGroup,
	     gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	     n, c) ;
	}
	else {
	  func<FLAYOUT, 14>(pGOut, pKernel, pGIn,
	     gOutChannel, gOutWidth, gOutHeight,
	     gInChannel, gInWidth, gInHeight,
	     gInChannelGroup, gOutChannelGroup,
	     gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	     n, c) ;
	}
	c+=14 ;
	break ;
      case 15:
	func<FLAYOUT, 15>(pGOut, pKernel, pGIn,
	   gOutChannel, gOutWidth, gOutHeight,
	   gInChannel, gInWidth, gInHeight,
	   gInChannelGroup, gOutChannelGroup,
	   gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	   n, c ) ;
	c+=15 ;
	break ;
      default :
	break ;
      }
      for (; c<gInChannelGroup; ) {
	if( FLAYOUT == VEDNN_FILTER_LAYOUT_NCHW
	    && (((uint64_t)pKernel) & 0x07) == 0 && (gOutChannelGroup & 0x01) == 0 ) {
	  func_even_filter_nchwaligned<16>(pGOut, pKernel, pGIn,
	     gOutChannel, gOutWidth, gOutHeight,
	     gInChannel, gInWidth, gInHeight,
	     gInChannelGroup, gOutChannelGroup,
	     gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	     n, c) ;
	}
	else {
	  func<FLAYOUT, 16>(pGOut, pKernel, pGIn,
	     gOutChannel, gOutWidth, gOutHeight,
	     gInChannel, gInWidth, gInHeight,
	     gInChannelGroup, gOutChannelGroup,
	     gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	     n, c) ;
	}
	c+= 16 ;
      } // gInChannel
    } // group
  } // batch
}

extern "C"
vednnError_t
vednnConvolutionBackwardData_direct_dil1_str1_padsame_ker1(
    const vednnTensorParam_t * 		pParamGradOut,
    const void *			pDataGradOut,
    const vednnFilterParam_t *	 	pParamKernel,
    const void * 			pDataKernel,
    const vednnConvolutionParam_t * 	pParamConv,
    const vednnTensorParam_t * 		pParamGradIn,
    void * 				pDataGradIn
)
{
  const int64_t batch       = pParamGradOut->batch;
  const int64_t gOutChannel = pParamGradOut->channel;
  const int64_t gOutWidth   = pParamGradOut->width;
  const int64_t gOutHeight  = pParamGradOut->height;
  const int64_t gInChannel  = pParamGradIn->channel;
  const int64_t gInWidth    = pParamGradIn->width;
  const int64_t gInHeight   = pParamGradIn->height;
//  const int64_t kernWidth   = pParamKernel->width;		// 1
//  const int64_t kernHeight  = pParamKernel->height;		// 1

  const int64_t filter_layout = pParamKernel->layout ;

  const int64_t group          = pParamConv->group;
//  const int64_t strideWidth    = pParamConv->strideWidth;	// 1
//  const int64_t strideHeight   = pParamConv->strideHeight;	// 1
//  const int64_t padWidth       = pParamConv->padWidth;	// 1
//  const int64_t padHeight      = pParamConv->padHeight;	// 1
//  const int64_t dilationWidth  = pParamConv->dilationWidth;	// 1
//  const int64_t dilationHeight = pParamConv->dilationHeight;	// 1

  const int64_t gOutChannelGroup = gOutChannel  / group;
  const int64_t gInChannelGroup  = gInChannel / group;

  const float *  pGOut   = (const float *) pDataGradOut;
  const float *  pKernel = (const float *) pDataKernel;
  float *  const pGIn    = (float * const) pDataGradIn;


  if( filter_layout == VEDNN_FILTER_LAYOUT_NCHW) {
    convloop<VEDNN_FILTER_LAYOUT_NCHW>(pGOut, pKernel, pGIn,
	       batch, group,
	       gOutChannel, gOutWidth, gOutHeight,
	       gInChannel, gInWidth, gInHeight,
	       gInChannelGroup, gOutChannelGroup ) ;
  }
  else {
    convloop<VEDNN_FILTER_LAYOUT_HWCN>(pGOut, pKernel, pGIn,
	       batch, group,
	       gOutChannel, gOutWidth, gOutHeight,
	       gInChannel, gInWidth, gInHeight,
	       gInChannelGroup, gOutChannelGroup ) ;
  }

  return VEDNN_SUCCESS;
}
