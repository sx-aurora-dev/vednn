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

  int64_t gInIndex = gInGroupOffset + ((n * gInChannel + c) * gInHeight) * gInWidth  ;

  for (int64_t h = 0; h < gInHeight ; h++ ) {
    for (int64_t w = 0; w < gInWidth ; w += VLEN ) {
      const int64_t vl = gInWidth - w < VLEN ? gInWidth - w  : VLEN ;

      const int64_t vlhalf = (vl >> 1) + (vl & 0x1) ;

      const int64_t xmin_s0 = (w+2) / 2 ;
      const int64_t xmin_s2 = (w+0) / 2 ;
      const int64_t xmin_s4 = w >= 2 ? (w-2) / 2 : 0 ;

      __vr vrw   = _vel_vaddsl_vsvl(w, _vel_vseq_vl(vl), vl) ;

      __vr vrj_s0 = _vel_vaddsl_vsvl(2, vrw, vl) ;
      __vr vrj_s1 = _vel_vaddsl_vsvl(1, vrw, vl) ;
      __vr vrj_s2 = vrw ;
      __vr vrj_s3 = _vel_vaddsl_vsvl(-1, vrw, vl) ;
      __vr vrj_s4 = _vel_vaddsl_vsvl(-2, vrw, vl) ;

      __vr vrx_s0 = _vel_vdivsl_vvsl(vrj_s0, 2, vl) ;
      __vr vrx_s12 = _vel_vdivsl_vvsl(vrj_s1, 2, vl) ;
      __vr vrx_s34 = _vel_vdivsl_vvsl(vrj_s3, 2, vl) ;

      __vm256 vmx0_s0 =  _vel_vfmkleq_mvl(_vel_vcmpsl_vvvl(vrj_s0, _vel_vmulsl_vsvl(2, vrx_s0, vl), vl), vl) ;
      __vm256 vmx1_s0 =  _vel_vfmklge_mvl(vrx_s0, vl) ;
      __vm256 vmx2_s0 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutWidth,vrx_s0, vl), vl) ;
      __vm256 vmx_s0 = _vel_andm_mmm(_vel_andm_mmm(vmx0_s0, vmx1_s0), vmx2_s0) ;

      __vm256 vmx0_s1 =  _vel_vfmkleq_mvl(_vel_vcmpsl_vvvl(vrj_s1, _vel_vmulsl_vsvl(2, vrx_s12, vl), vl), vl) ;
      __vm256 vmx0_s2 =  _vel_vfmkleq_mvl(_vel_vcmpsl_vvvl(vrj_s2, _vel_vmulsl_vsvl(2, vrx_s12, vl), vl), vl) ;
      __vm256 vmx1_s12 =  _vel_vfmklge_mvl(vrx_s12, vl) ;
      __vm256 vmx2_s12 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutWidth,vrx_s12, vl), vl) ;
      __vm256 vmx_s12 = _vel_andm_mmm(vmx1_s12, vmx2_s12) ;
      __vm256 vmx_s1 = _vel_andm_mmm(vmx0_s1, vmx_s12) ;
      __vm256 vmx_s2 = _vel_andm_mmm(vmx0_s2, vmx_s12) ;

      __vm256 vmx0_s3 =  _vel_vfmkleq_mvl(_vel_vcmpsl_vvvl(vrj_s3, _vel_vmulsl_vsvl(2, vrx_s34, vl), vl), vl) ;
      __vm256 vmx0_s4 =  _vel_vfmkleq_mvl(_vel_vcmpsl_vvvl(vrj_s4, _vel_vmulsl_vsvl(2, vrx_s34, vl), vl), vl) ;
      __vm256 vmx1_s34 =  _vel_vfmklge_mvl(vrx_s34, vl) ;
      __vm256 vmx2_s34 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutWidth,vrx_s34, vl), vl) ;
      __vm256 vmx_s34 = _vel_andm_mmm(vmx1_s34, vmx2_s34) ;
      __vm256 vmx_s3 = _vel_andm_mmm(vmx0_s3, vmx_s34) ;
      __vm256 vmx_s4 = _vel_andm_mmm(vmx0_s4, vmx_s34) ;

      __vr vrsum0_s0  = _vel_vbrds_vsl(0.f, vlhalf) ;
      __vr vrsum0_s1  = _vel_vbrds_vsl(0.f, vlhalf) ;
      __vr vrsum0_s2  = _vel_vbrds_vsl(0.f, vlhalf) ;
      __vr vrsum0_s3  = _vel_vbrds_vsl(0.f, vlhalf) ;
      __vr vrsum0_s4  = _vel_vbrds_vsl(0.f, vlhalf) ;
      __vr vrsum_s0[nPacked] ;
      __vr vrsum_s1[nPacked] ;
      __vr vrsum_s2[nPacked] ;
      __vr vrsum_s3[nPacked] ;
      __vr vrsum_s4[nPacked] ;
#pragma clang loop unroll(full)
      for(int64_t cc=0; cc<nPacked; cc++) {
	vrsum_s0[cc] = _vel_pvbrd_vsl(0UL, vlhalf) ;
	vrsum_s1[cc] = _vel_pvbrd_vsl(0UL, vlhalf) ;
	vrsum_s2[cc] = _vel_pvbrd_vsl(0UL, vlhalf) ;
	vrsum_s3[cc] = _vel_pvbrd_vsl(0UL, vlhalf) ;
	vrsum_s4[cc] = _vel_pvbrd_vsl(0UL, vlhalf) ;
      }

      for (int64_t r=0; r<kernHeight; r++) {
	int64_t i = h - r + 2 ;
	int64_t y = i/2;
	if ( y*2 != i || y < 0 || gOutHeight <= y)  continue ;

	for (int64_t k=0; k<gOutChannelGroup; k++) {
	  int64_t gOutIndex    = gOutGroupOffset + ((n * gOutChannel + k) * gOutHeight) * gOutWidth ;

	  __vr vrgout_s01 = _vel_vldu_vssl(4, pGOut+gOutIndex+gOutWidth*y + xmin_s0 , vlhalf) ;
	  __vr vrgout_s23 = _vel_vldu_vssl(4, pGOut+gOutIndex+gOutWidth*y + xmin_s2 , vlhalf) ;
	  __vr vrgout_s4  = _vel_vldu_vssl(4, pGOut+gOutIndex+gOutWidth*y + xmin_s4 , vlhalf) ;

#define FILTER_OFFSET(k,c,r,s) ( kernGroupOffset + filter_index<FLAYOUT>(k,c,r,s, gInChannelGroup, gOutChannelGroup, kernHeight, kernWidth) )

#define VFADD(VRGOUT, VRSUM, VRSUM0, K,R,S) {						\
	    __vr vrgoutP = _vel_vshf_vvvsl(VRGOUT, VRGOUT, VE_VSHUFFLE_YUZU, vlhalf) ;	\
	    if ( remain ) {								\
	      const float    kerValue0  = pKernel[FILTER_OFFSET(K,c+ 0,R,S)] ;		\
	      VRSUM0 = _vel_vfmads_vvsvl(VRSUM0, kerValue0, VRGOUT, vlhalf) ;		\
	    }										\
	    _Pragma("clang loop unroll(full)")						\
	    for(int64_t cc=0; cc<nPacked; cc++) {					\
	      const uint64_t kerValue = _vel_pack_f32p(pKernel + FILTER_OFFSET(K,c+2*cc+remain,  R,S),		\
						       pKernel + FILTER_OFFSET(K,c+2*cc+remain+1,R,S)) ;	\
              VRSUM[cc] = _vel_pvfmad_vvsvl(VRSUM[cc], kerValue, vrgoutP, vlhalf) ;		\
	    }										\
	  }

	  VFADD(vrgout_s01, vrsum_s0, vrsum0_s0, k+0, r, 0) ;
	  VFADD(vrgout_s01, vrsum_s1, vrsum0_s1, k+0, r, 1) ;
	  VFADD(vrgout_s23, vrsum_s2, vrsum0_s2, k+0, r, 2) ;
	  VFADD(vrgout_s23, vrsum_s3, vrsum0_s3, k+0, r, 3) ;
	  VFADD(vrgout_s4,  vrsum_s4, vrsum0_s4, k+0, r, 4) ;

#undef VFADD
#undef FILTER_OFFSET
	}

      } // kernHeight


      {
	vrsum0_s0 = _vel_vex_vvmvl(vrsum0_s0, vmx_s0, _vel_pvbrd_vsl(0UL, vl), vl) ;
	vrsum0_s1 = _vel_vex_vvmvl(vrsum0_s1, vmx_s1, _vel_pvbrd_vsl(0UL, vl), vl) ;
	vrsum0_s2 = _vel_vex_vvmvl(vrsum0_s2, vmx_s2, _vel_pvbrd_vsl(0UL, vl), vl) ;
	vrsum0_s3 = _vel_vex_vvmvl(vrsum0_s3, vmx_s3, _vel_pvbrd_vsl(0UL, vl), vl) ;
	vrsum0_s4 = _vel_vex_vvmvl(vrsum0_s4, vmx_s4, _vel_pvbrd_vsl(0UL, vl), vl) ;
	__vr vrsum0 = _vel_vfadds_vvvl(_vel_vfadds_vvvl(vrsum0_s0, _vel_pvfadd_vvvl(vrsum0_s1, vrsum0_s2, vl), vl),
				       _vel_vfadds_vvvl(vrsum0_s3, vrsum0_s4, vl), vl) ;
  	_vel_vstu_vssl(vrsum0, 4, pGIn+gInIndex + 0 * gInHeight * gInWidth, vl) ;
      }

#pragma clang loop unroll(full)
      for(int64_t cc=0; cc<nPacked; cc++) {
	__vr _vrsum_s0 = _vel_vex_vvmvl(vrsum_s0[cc], vmx_s0, _vel_pvbrd_vsl(0UL, vl), vl) ;
	__vr _vrsum_s1 = _vel_vex_vvmvl(vrsum_s1[cc], vmx_s1, _vel_pvbrd_vsl(0UL, vl), vl) ;
	__vr _vrsum_s2 = _vel_vex_vvmvl(vrsum_s2[cc], vmx_s2, _vel_pvbrd_vsl(0UL, vl), vl) ;
	__vr _vrsum_s3 = _vel_vex_vvmvl(vrsum_s3[cc], vmx_s3, _vel_pvbrd_vsl(0UL, vl), vl) ;
	__vr _vrsum_s4 = _vel_vex_vvmvl(vrsum_s4[cc], vmx_s4, _vel_pvbrd_vsl(0UL, vl), vl) ;
	__vr vrsum = _vel_pvfadd_vvvl(_vel_pvfadd_vvvl(_vrsum_s0, _vel_pvfadd_vvvl(_vrsum_s1, _vrsum_s2, vl), vl),
				      _vel_pvfadd_vvvl(_vrsum_s3, _vrsum_s4, vl), vl) ;
  	_vel_vstu_vssl(vrsum, 4, pGIn+gInIndex + (2*cc+remain  ) * gInHeight * gInWidth, vl) ;
  	_vel_vstl_vssl(vrsum, 4, pGIn+gInIndex + (2*cc+remain+1) * gInHeight * gInWidth, vl) ;
      }

      gInIndex += vl ;
    } // gInWidth
  } // gInHeight
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
    const int64_t gOutChannelGroup
)
{
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
	   gInChannelGroup, gOutChannelGroup,
	   gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	   n, c ) ;
	c+=1 ;
	break ;
      case 2:
	func<FLAYOUT, 2>(pGOut, pKernel, pGIn,
	   gOutChannel, gOutWidth, gOutHeight,
	   gInChannel, gInWidth, gInHeight,
	   kernWidth, kernHeight,
	   gInChannelGroup, gOutChannelGroup,
	   gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	   n, c ) ;
	c+=2 ;
	break ;
      case 3:
	func<FLAYOUT, 3>(pGOut, pKernel, pGIn,
	   gOutChannel, gOutWidth, gOutHeight,
	   gInChannel, gInWidth, gInHeight,
	   kernWidth, kernHeight,
	   gInChannelGroup, gOutChannelGroup,
	   gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	   n, c ) ;
	c+=3 ;
	break ;
      case 4:
	func<FLAYOUT, 4>(pGOut, pKernel, pGIn,
	   gOutChannel, gOutWidth, gOutHeight,
	   gInChannel, gInWidth, gInHeight,
	   kernWidth, kernHeight,
	   gInChannelGroup, gOutChannelGroup,
	   gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	   n, c ) ;
	c+=4 ;
	break ;
      case 5:
	func<FLAYOUT, 5>(pGOut, pKernel, pGIn,
	   gOutChannel, gOutWidth, gOutHeight,
	   gInChannel, gInWidth, gInHeight,
	   kernWidth, kernHeight,
	   gInChannelGroup, gOutChannelGroup,
	   gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	   n, c ) ;
	c+=5 ;
	break ;
      case 6:
	func<FLAYOUT, 6>(pGOut, pKernel, pGIn,
	   gOutChannel, gOutWidth, gOutHeight,
	   gInChannel, gInWidth, gInHeight,
	   kernWidth, kernHeight,
	   gInChannelGroup, gOutChannelGroup,
	   gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	   n, c ) ;
	c+=6 ;
	break ;
      case 7:
	func<FLAYOUT, 7>(pGOut, pKernel, pGIn,
	   gOutChannel, gOutWidth, gOutHeight,
	   gInChannel, gInWidth, gInHeight,
	   kernWidth, kernHeight,
	   gInChannelGroup, gOutChannelGroup,
	   gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	   n, c ) ;
	c+=7 ;
	break ;
      case 8:
	func<FLAYOUT, 8>(pGOut, pKernel, pGIn,
	   gOutChannel, gOutWidth, gOutHeight,
	   gInChannel, gInWidth, gInHeight,
	   kernWidth, kernHeight,
	   gInChannelGroup, gOutChannelGroup,
	   gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	   n, c ) ;
	c+=8 ;
	break ;
      case 9:
	func<FLAYOUT, 9>(pGOut, pKernel, pGIn,
	   gOutChannel, gOutWidth, gOutHeight,
	   gInChannel, gInWidth, gInHeight,
	   kernWidth, kernHeight,
	   gInChannelGroup, gOutChannelGroup,
	   gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	   n, c ) ;
	c+=9 ;
	break ;
      case 10:
	func<FLAYOUT, 10>(pGOut, pKernel, pGIn,
	   gOutChannel, gOutWidth, gOutHeight,
	   gInChannel, gInWidth, gInHeight,
	   kernWidth, kernHeight,
	   gInChannelGroup, gOutChannelGroup,
	   gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	   n, c ) ;
	c+=10 ;
	break ;
      case 11:
	func<FLAYOUT, 11>(pGOut, pKernel, pGIn,
	   gOutChannel, gOutWidth, gOutHeight,
	   gInChannel, gInWidth, gInHeight,
	   kernWidth, kernHeight,
	   gInChannelGroup, gOutChannelGroup,
	   gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	   n, c ) ;
	c+=11 ;
	break ;
      case 12:
	func<FLAYOUT, 12>(pGOut, pKernel, pGIn,
	   gOutChannel, gOutWidth, gOutHeight,
	   gInChannel, gInWidth, gInHeight,
	   kernWidth, kernHeight,
	   gInChannelGroup, gOutChannelGroup,
	   gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	   n, c ) ;
	c+=12 ;
	break ;
      case 13:
	func<FLAYOUT, 13>(pGOut, pKernel, pGIn,
	   gOutChannel, gOutWidth, gOutHeight,
	   gInChannel, gInWidth, gInHeight,
	   kernWidth, kernHeight,
	   gInChannelGroup, gOutChannelGroup,
	   gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	   n, c ) ;
	c+=13 ;
	break ;
      case 14:
	func<FLAYOUT, 14>(pGOut, pKernel, pGIn,
	   gOutChannel, gOutWidth, gOutHeight,
	   gInChannel, gInWidth, gInHeight,
	   kernWidth, kernHeight,
	   gInChannelGroup, gOutChannelGroup,
	   gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	   n, c ) ;
	c+=14 ;
	break ;
      case 15:
	func<FLAYOUT, 15>(pGOut, pKernel, pGIn,
	   gOutChannel, gOutWidth, gOutHeight,
	   gInChannel, gInWidth, gInHeight,
	   kernWidth, kernHeight,
	   gInChannelGroup, gOutChannelGroup,
	   gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	   n, c ) ;
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
	   gInChannelGroup, gOutChannelGroup,
	   gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	   n, c) ;
	c+= 16 ;
      } // gInChannel
    } // group
  } // batch
}

extern "C"
vednnError_t
vednnConvolutionBackwardData_direct_dil1_str2_pad2_ker5(
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
//  const int64_t strideWidth    = pParamConv->strideWidth;		// 2
//  const int64_t strideHeight   = pParamConv->strideHeight;		// 2
//  const int64_t padWidth       = pParamConv->padWidth;		// 2
//  const int64_t padHeight      = pParamConv->padHeight;		// 2
//  const int64_t dilationWidth  = pParamConv->dilationWidth;		// 1
//  const int64_t dilationHeight = pParamConv->dilationHeight;		// 1

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
	       gInChannelGroup, gOutChannelGroup ) ;
  }
  else {
    convloop<VEDNN_FILTER_LAYOUT_HWCN>(pGOut, pKernel, pGIn,
	       batch, group,
	       gOutChannel, gOutWidth, gOutHeight,
	       gInChannel, gInWidth, gInHeight,
	       kernWidth, kernHeight,
	       gInChannelGroup, gOutChannelGroup ) ;
  }

  return VEDNN_SUCCESS;
}
