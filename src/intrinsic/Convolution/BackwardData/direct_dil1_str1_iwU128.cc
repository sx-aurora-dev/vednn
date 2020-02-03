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
    const int64_t kernWidth,
    const int64_t kernHeight,
    const int64_t padWidth,
    const int64_t padHeight,
    const int64_t dilationWidth,
    const int64_t dilationHeight,
    const int64_t gInChannelGroup,
    const int64_t gOutChannelGroup,
    const int64_t gInGroupOffset,
    const int64_t gOutGroupOffset,
    const int64_t kernGroupOffset,
    const int64_t n,
    const int64_t c,
    const int64_t nH,
    const __vr    vrh,
    const __vr    vrw
)
{
  const int64_t remain  = NUMCHANNEL & 0x1 ;
  const int64_t nPacked = NUMCHANNEL >> 1 ;

  int64_t gInIndex = gInGroupOffset + ((n * gInChannel + c) * gInHeight ) * gInWidth  ;

  for (int64_t h=0; h<gInHeight; h+=nH) {
    const int64_t vl = gInWidth * (gInHeight - h < nH ? gInHeight - h : nH) ;
    const int64_t gip = h * gInWidth ;

    __vr vrsum0  = _vel_vbrds_vsl(0.f, vl) ;
    __vr vrsum[nPacked] ;
#pragma clang loop unroll(full)
    for(int64_t cc=0; cc<nPacked; cc++) {
	vrsum[cc] = _vel_pvbrd_vsl(0UL, vl) ;
    }

    for (int64_t r=0; r<kernHeight; r++) {
      __vr vry = _vel_vaddsl_vsvl(padHeight-r+h, vrh, vl) ;

      __vm256 vmy1 =  _vel_vfmklge_mvl(vry, vl) ;
      __vm256 vmy2 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutHeight,vry, vl), vl) ;

      __vm256 vmy = _vel_andm_mmm(vmy1, vmy2) ;

      for (int64_t s=0; s<kernWidth; s++) {
	__vr vrx = _vel_vaddsl_vsvl(padWidth-s, vrw, vl) ;

	__vm256 vmx1 =  _vel_vfmklge_mvl(vrx, vl) ;
	__vm256 vmx2 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutWidth,vrx, vl), vl) ;

	__vm256 vmx = _vel_andm_mmm(vmx1, vmx2) ;

	__vm256 vmall = _vel_andm_mmm(vmy,vmx) ;

	for (int64_t k=0; k<gOutChannelGroup; k++) {
	  int64_t gOutIndex    = gOutGroupOffset + ((n * gOutChannel + k) * gOutHeight) * gOutWidth ;
	  __vr vrgout_ptr = _vel_vsfa_vvssl(_vel_vaddsl_vvvl(_vel_vmulsl_vsvl(gOutWidth, vry, vl), vrx, vl),
					2,
					(unsigned long)(pGOut+gOutIndex), vl) ;

	  __vr vrgout = _vel_vgtu_vvssml(vrgout_ptr, 0, 0, vmall, vl) ;

#define FILTER_OFFSET(k,c,r,s) ( kernGroupOffset + filter_index<FLAYOUT>(k,c,r,s, gInChannelGroup, gOutChannelGroup, kernHeight, kernWidth) )

#define VFADD(VRGOUT,K,R,S) {								\
	    VRGOUT = _vel_vmrg_vsvml(0.f, VRGOUT, vmall, vl) ;				\
	    __vr vrgoutP = _vel_vshf_vvvsl(VRGOUT, VRGOUT, VE_VSHUFFLE_YUZU, vl) ;	\
	    if( remain ) {								\
              const float    kerValue0  = pKernel[FILTER_OFFSET(K,c+ 0,R,S)] ;		\
	      vrsum0 = _vel_vfmads_vvsvl(vrsum0, kerValue0, VRGOUT, vl) ;		\
	    }										\
	    _Pragma("clang loop unroll(full)")						\
	    for(int64_t cc=0; cc<nPacked; cc++) {					\
	      const uint64_t kerValue = _vel_pack_f32p(pKernel + FILTER_OFFSET(K,c+2*cc+remain,R,S),		\
						       pKernel + FILTER_OFFSET(K,c+2*cc+remain+1,R,S)) ;	\
	      vrsum[cc] = _vel_pvfmad_vvsvl(vrsum[cc], kerValue, vrgoutP, vl) ;		\
	    }										\
	  }

	  VFADD(vrgout, k, r, s) ;
#undef VFADD
#undef FILTER_OFFSET
	} // gOutChannel

      } // kernWidth
    } // kernHeight

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
    const int64_t kernWidth,
    const int64_t kernHeight,
    const int64_t gInChannelGroup,
    const int64_t gOutChannelGroup,
    const int64_t padWidth,
    const int64_t padHeight,
    const int64_t dilationWidth,
    const int64_t dilationHeight
)
{

  const int64_t nH = VLEN / gInWidth ;

  __vr vrseq = _vel_vseq_vl(nH*gInWidth) ;
  __vr vrh  = _vel_vdivsl_vvsl(vrseq, gInWidth, nH*gInWidth) ;
  __vr vrw  = _vel_vsubsl_vvvl(vrseq, _vel_vmulul_vsvl(gInWidth,vrh, nH*gInWidth), nH*gInWidth) ;

  for (int64_t n=0; n<batch; n++) {
    for (int64_t g = 0; g < group; g++) {

      int64_t gInGroupOffset  = g * gInChannelGroup * gInHeight * gInWidth;
      int64_t gOutGroupOffset = g * gOutChannelGroup * gOutHeight * gOutWidth;
      int64_t kernGroupOffset = g * gOutChannelGroup * gInChannelGroup * kernHeight * kernWidth;

      const int64_t remain = gInChannelGroup & 0xf ;

      int64_t c=0;
      switch(remain) {
      case 1:
	func<FLAYOUT, 1>(pGOut, pKernel, pGIn,
	   gOutChannel, gOutWidth, gOutHeight,
	   gInChannel, gInWidth, gInHeight,
	   kernWidth, kernHeight,
	   padWidth, padHeight,
	   dilationWidth, dilationHeight,
	   gInChannelGroup, gOutChannelGroup,
	   gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	   n, c, nH, vrh, vrw) ;
	c+=1 ;
	break ;
      case 2:
	func<FLAYOUT, 2>(pGOut, pKernel, pGIn,
	   gOutChannel, gOutWidth, gOutHeight,
	   gInChannel, gInWidth, gInHeight,
	   kernWidth, kernHeight,
	   padWidth, padHeight,
	   dilationWidth, dilationHeight,
	   gInChannelGroup, gOutChannelGroup,
	   gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	   n, c, nH, vrh, vrw) ;
	c+=2 ;
	break ;
      case 3:
	func<FLAYOUT, 3>(pGOut, pKernel, pGIn,
	   gOutChannel, gOutWidth, gOutHeight,
	   gInChannel, gInWidth, gInHeight,
	   kernWidth, kernHeight,
	   padWidth, padHeight,
	   dilationWidth, dilationHeight,
	   gInChannelGroup, gOutChannelGroup,
	   gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	   n, c, nH, vrh, vrw) ;
	c+=3 ;
	break ;
      case 4:
	func<FLAYOUT, 4>(pGOut, pKernel, pGIn,
	   gOutChannel, gOutWidth, gOutHeight,
	   gInChannel, gInWidth, gInHeight,
	   kernWidth, kernHeight,
	   padWidth, padHeight,
	   dilationWidth, dilationHeight,
	   gInChannelGroup, gOutChannelGroup,
	   gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	   n, c, nH, vrh, vrw) ;
	c+=4 ;
	break ;
      case 5:
	func<FLAYOUT, 5>(pGOut, pKernel, pGIn,
	   gOutChannel, gOutWidth, gOutHeight,
	   gInChannel, gInWidth, gInHeight,
	   kernWidth, kernHeight,
	   padWidth, padHeight,
	   dilationWidth, dilationHeight,
	   gInChannelGroup, gOutChannelGroup,
	   gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	   n, c, nH, vrh, vrw) ;
	c+=5 ;
	break ;
      case 6:
	func<FLAYOUT, 6>(pGOut, pKernel, pGIn,
	   gOutChannel, gOutWidth, gOutHeight,
	   gInChannel, gInWidth, gInHeight,
	   kernWidth, kernHeight,
	   padWidth, padHeight,
	   dilationWidth, dilationHeight,
	   gInChannelGroup, gOutChannelGroup,
	   gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	   n, c, nH, vrh, vrw) ;
	c+=6 ;
	break ;
      case 7:
	func<FLAYOUT, 7>(pGOut, pKernel, pGIn,
	   gOutChannel, gOutWidth, gOutHeight,
	   gInChannel, gInWidth, gInHeight,
	   kernWidth, kernHeight,
	   padWidth, padHeight,
	   dilationWidth, dilationHeight,
	   gInChannelGroup, gOutChannelGroup,
	   gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	   n, c, nH, vrh, vrw) ;
	c+=7 ;
	break ;
      case 8:
	func<FLAYOUT, 8>(pGOut, pKernel, pGIn,
	   gOutChannel, gOutWidth, gOutHeight,
	   gInChannel, gInWidth, gInHeight,
	   kernWidth, kernHeight,
	   padWidth, padHeight,
	   dilationWidth, dilationHeight,
	   gInChannelGroup, gOutChannelGroup,
	   gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	   n, c, nH, vrh, vrw) ;
	c+=8 ;
	break ;
      case 9:
	func<FLAYOUT, 9>(pGOut, pKernel, pGIn,
	   gOutChannel, gOutWidth, gOutHeight,
	   gInChannel, gInWidth, gInHeight,
	   kernWidth, kernHeight,
	   padWidth, padHeight,
	   dilationWidth, dilationHeight,
	   gInChannelGroup, gOutChannelGroup,
	   gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	   n, c, nH, vrh, vrw) ;
	c+=9 ;
	break ;
      case 10:
	func<FLAYOUT, 10>(pGOut, pKernel, pGIn,
	   gOutChannel, gOutWidth, gOutHeight,
	   gInChannel, gInWidth, gInHeight,
	   kernWidth, kernHeight,
	   padWidth, padHeight,
	   dilationWidth, dilationHeight,
	   gInChannelGroup, gOutChannelGroup,
	   gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	   n, c, nH, vrh, vrw) ;
	c+=10 ;
	break ;
      case 11:
	func<FLAYOUT, 11>(pGOut, pKernel, pGIn,
	   gOutChannel, gOutWidth, gOutHeight,
	   gInChannel, gInWidth, gInHeight,
	   kernWidth, kernHeight,
	   padWidth, padHeight,
	   dilationWidth, dilationHeight,
	   gInChannelGroup, gOutChannelGroup,
	   gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	   n, c, nH, vrh, vrw) ;
	c+=11 ;
	break ;
      case 12:
	func<FLAYOUT, 12>(pGOut, pKernel, pGIn,
	   gOutChannel, gOutWidth, gOutHeight,
	   gInChannel, gInWidth, gInHeight,
	   kernWidth, kernHeight,
	   padWidth, padHeight,
	   dilationWidth, dilationHeight,
	   gInChannelGroup, gOutChannelGroup,
	   gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	   n, c, nH, vrh, vrw) ;
	c+=12 ;
	break ;
      case 13:
	func<FLAYOUT, 13>(pGOut, pKernel, pGIn,
	   gOutChannel, gOutWidth, gOutHeight,
	   gInChannel, gInWidth, gInHeight,
	   kernWidth, kernHeight,
	   padWidth, padHeight,
	   dilationWidth, dilationHeight,
	   gInChannelGroup, gOutChannelGroup,
	   gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	   n, c, nH, vrh, vrw) ;
	c+=13 ;
	break ;
      case 14:
	func<FLAYOUT, 14>(pGOut, pKernel, pGIn,
	   gOutChannel, gOutWidth, gOutHeight,
	   gInChannel, gInWidth, gInHeight,
	   kernWidth, kernHeight,
	   padWidth, padHeight,
	   dilationWidth, dilationHeight,
	   gInChannelGroup, gOutChannelGroup,
	   gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	   n, c, nH, vrh, vrw) ;
	c+=14 ;
	break ;
      case 15:
	func<FLAYOUT, 15>(pGOut, pKernel, pGIn,
	   gOutChannel, gOutWidth, gOutHeight,
	   gInChannel, gInWidth, gInHeight,
	   kernWidth, kernHeight,
	   padWidth, padHeight,
	   dilationWidth, dilationHeight,
	   gInChannelGroup, gOutChannelGroup,
	   gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	   n, c, nH, vrh, vrw) ;
	c+=15 ;
	break ;
      default :
	break ;
      }
      for (; c<gInChannelGroup; ) {
	func<FLAYOUT, 16>(pGOut, pKernel, pGIn,
	   gOutChannel, gOutWidth, gOutHeight,
	   gInChannel, gInWidth, gInHeight,
	   kernWidth, kernHeight,
	   padWidth, padHeight,
	   dilationWidth, dilationHeight,
	   gInChannelGroup, gOutChannelGroup,
	   gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	   n, c, nH, vrh, vrw) ;
	c+= 16 ;
      } // gInChannel
    } // group
  } // batch
}

extern "C"
vednnError_t
vednnConvolutionBackwardData_direct_dil1_str1_iwU128(
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
  const int64_t kernWidth   = pParamKernel->width;
  const int64_t kernHeight  = pParamKernel->height;

  const int64_t filter_layout = pParamKernel->layout ;

  const int64_t group          = pParamConv->group;
//  const int64_t strideWidth    = pParamConv->strideWidth;
//  const int64_t strideHeight   = pParamConv->strideHeight;
  const int64_t padWidth       = pParamConv->padWidth;
  const int64_t padHeight      = pParamConv->padHeight;
  const int64_t dilationWidth  = pParamConv->dilationWidth;
  const int64_t dilationHeight = pParamConv->dilationHeight;

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
	       kernWidth, kernHeight,
	       gInChannelGroup, gOutChannelGroup,
	       padWidth, padHeight,
	       dilationWidth, dilationHeight) ;
  }
  else {
    convloop<VEDNN_FILTER_LAYOUT_HWCN>(pGOut, pKernel, pGIn,
	       batch, group,
	       gOutChannel, gOutWidth, gOutHeight,
	       gInChannel, gInWidth, gInHeight,
	       kernWidth, kernHeight,
	       gInChannelGroup, gOutChannelGroup,
	       padWidth, padHeight,
	       dilationWidth, dilationHeight) ;
  }

  return VEDNN_SUCCESS;
}
