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
    const int64_t c,
    const int64_t nH,
    const __vr    vrw,
    const __vr    vry
)
{
  const int64_t remain  = NUMCHANNEL & 0x1 ;
  const int64_t nPacked = NUMCHANNEL >> 1 ;

  int64_t gInIndex = gInGroupOffset + ((n * gInChannel + c) * gInHeight ) * gInWidth  ;

  __vr vrx_s0 = _vel_vaddsl_vsvl(0-0, vrw, nH*gInWidth) ;
  __vr vrx_s1 = _vel_vaddsl_vsvl(0-1, vrw, nH*gInWidth) ;
  __vr vrx_s2 = _vel_vaddsl_vsvl(0-2, vrw, nH*gInWidth) ;
  __vr vrx_s3 = _vel_vaddsl_vsvl(0-3, vrw, nH*gInWidth) ;

  __vm256 vmx_s0 = _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutWidth,vrx_s0, nH*gInWidth), nH*gInWidth) ;

  __vm256 vmx1_s1 = _vel_vfmklge_mvl(vrx_s1, nH*gInWidth) ;
  __vm256 vmx2_s1 = _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutWidth,vrx_s1, nH*gInWidth), nH*gInWidth) ;
  __vm256 vmx_s1  = _vel_andm_mmm(vmx1_s1, vmx2_s1) ;

  __vm256 vmx1_s2 = _vel_vfmklge_mvl(vrx_s2, nH*gInWidth) ;
  __vm256 vmx2_s2 = _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutWidth,vrx_s2, nH*gInWidth), nH*gInWidth) ;
  __vm256 vmx_s2  = _vel_andm_mmm(vmx1_s2, vmx2_s2) ;

  __vm256 vmx_s3 = _vel_vfmklge_mvl(vrx_s3, nH*gInWidth) ;

  for (int64_t h=0; h<gInHeight; h+=nH) {
    const int64_t vl  = gInWidth * (gInHeight - h < nH ? gInHeight - h : nH) ;
    const int64_t vl2 = gOutWidth * (gInHeight - h < nH ? gInHeight - h : nH) ;

    const int64_t gip = h * gInWidth ;

    __vr vrsum0_s0  = _vel_vbrds_vsl(0.f, vl) ;
    __vr vrsum0_s1  = _vel_vbrds_vsl(0.f, vl) ;
    __vr vrsum0_s2  = _vel_vbrds_vsl(0.f, vl) ;
    __vr vrsum0_s3  = _vel_vbrds_vsl(0.f, vl) ;
    __vr vrsum_s0[nPacked] ;
    __vr vrsum_s1[nPacked] ;
    __vr vrsum_s2[nPacked] ;
    __vr vrsum_s3[nPacked] ;
#pragma clang loop unroll(full)
    for(int64_t cc=0; cc<nPacked; cc++) {
      vrsum_s0[cc] = _vel_pvbrd_vsl(0UL, vl) ;
      vrsum_s1[cc] = _vel_pvbrd_vsl(0UL, vl) ;
      vrsum_s2[cc] = _vel_pvbrd_vsl(0UL, vl) ;
      vrsum_s3[cc] = _vel_pvbrd_vsl(0UL, vl) ;
    }

    __vr vry_r0 = _vel_vaddsl_vsvl(h-0, vry, vl2) ;
    __vr vry_r1 = _vel_vaddsl_vsvl(h-1, vry, vl2) ;
    __vr vry_r2 = _vel_vaddsl_vsvl(h-2, vry, vl2) ;
    __vr vry_r3 = _vel_vaddsl_vsvl(h-3, vry, vl2) ;

    __vm256 vmy1_r0 = _vel_vfmklge_mvl(vry_r0, vl) ;
    __vm256 vmy2_r0 = _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutHeight,vry_r0, vl), vl) ;
    __vm256 vmy_r0 = _vel_andm_mmm(vmy1_r0, vmy2_r0) ;

    __vm256 vmy1_r1 = _vel_vfmklge_mvl(vry_r1, vl) ;
    __vm256 vmy2_r1 = _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutHeight,vry_r1, vl), vl) ;
    __vm256 vmy_r1  = _vel_andm_mmm(vmy1_r1, vmy2_r1) ;

    __vm256 vmy1_r2 = _vel_vfmklge_mvl(vry_r2, vl) ;
    __vm256 vmy2_r2 = _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutHeight,vry_r2, vl), vl) ;
    __vm256 vmy_r2  = _vel_andm_mmm(vmy1_r2, vmy2_r2) ;

    __vm256 vmy1_r3 = _vel_vfmklge_mvl(vry_r3, vl) ;
    __vm256 vmy2_r3 = _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutHeight,vry_r3, vl), vl) ;
    __vm256 vmy_r3  = _vel_andm_mmm(vmy1_r3, vmy2_r3) ;

    int64_t k=0;
    for (; k<gOutChannelGroup; k+=1) {
      int64_t gOutIndex    = gOutGroupOffset + ((n * gOutChannel + k) * gOutHeight) * gOutWidth ;

      __vr vrgout_r0 = _vel_vldu_vssl(4, pGOut+gOutIndex+(h-0)*gOutWidth, vl2) ;
      __vr vrgout_r1 = _vel_vldu_vssl(4, pGOut+gOutIndex+(h-1)*gOutWidth, vl2) ;
      __vr vrgout_r2 = _vel_vldu_vssl(4, pGOut+gOutIndex+(h-2)*gOutWidth, vl2) ;
      __vr vrgout_r3 = _vel_vldu_vssl(4, pGOut+gOutIndex+(h-3)*gOutWidth, vl2) ;


#define FILTER_OFFSET(k,c,r,s) ( kernGroupOffset + filter_index<FLAYOUT>(k,c,r,s, gInChannelGroup, gOutChannelGroup, 4, 4) )

#define VFADD(VRGOUT, VRSUM, VRSUM0, K,R,S) {								\
	__vr vrgoutP = _vel_vshf_vvvsl(VRGOUT, VRGOUT, VE_VSHUFFLE_YUZU, vl) ;				\
	if( remain ) {											\
	  const float    kerValue0  = pKernel[FILTER_OFFSET(K,c+ 0,R,S)] ;				\
	  VRSUM0 = _vel_vfmads_vvsvl(VRSUM0, kerValue0, VRGOUT, vl2) ;					\
	}												\
	_Pragma("clang loop unroll(full)")								\
	for(int64_t cc=0; cc<nPacked; cc++) {								\
	  const uint64_t kerValue = _vel_pack_f32p(pKernel + FILTER_OFFSET(K,c+2*cc+remain,  R,S),	\
						   pKernel + FILTER_OFFSET(K,c+2*cc+remain+1,R,S)) ;	\
	  VRSUM[cc] = _vel_pvfmad_vvsvl(VRSUM[cc], kerValue, vrgoutP, vl2) ;				\
	}												\
      }

      vrgout_r0 = _vel_vmrg_vsvml(0.f, vrgout_r0, vmy_r0, vl) ;
      VFADD(vrgout_r0, vrsum_s0, vrsum0_s0, k+0, 0, 0) ;
      VFADD(vrgout_r0, vrsum_s1, vrsum0_s1, k+0, 0, 1) ;
      VFADD(vrgout_r0, vrsum_s2, vrsum0_s2, k+0, 0, 2) ;
      VFADD(vrgout_r0, vrsum_s3, vrsum0_s3, k+0, 0, 3) ;

      vrgout_r1 = _vel_vmrg_vsvml(0.f, vrgout_r1, vmy_r1, vl) ;
      VFADD(vrgout_r1, vrsum_s0, vrsum0_s0, k+0, 1, 0) ;
      VFADD(vrgout_r1, vrsum_s1, vrsum0_s1, k+0, 1, 1) ;
      VFADD(vrgout_r1, vrsum_s2, vrsum0_s2, k+0, 1, 2) ;
      VFADD(vrgout_r1, vrsum_s3, vrsum0_s3, k+0, 1, 3) ;

      vrgout_r2 = _vel_vmrg_vsvml(0.f, vrgout_r2, vmy_r2, vl) ;
      VFADD(vrgout_r2, vrsum_s0, vrsum0_s0, k+0, 2, 0) ;
      VFADD(vrgout_r2, vrsum_s1, vrsum0_s1, k+0, 2, 1) ;
      VFADD(vrgout_r2, vrsum_s2, vrsum0_s2, k+0, 2, 2) ;
      VFADD(vrgout_r2, vrsum_s3, vrsum0_s3, k+0, 2, 3) ;

      vrgout_r3 = _vel_vmrg_vsvml(0.f, vrgout_r3, vmy_r3, vl) ;
      VFADD(vrgout_r3, vrsum_s0, vrsum0_s0, k+0, 3, 0) ;
      VFADD(vrgout_r3, vrsum_s1, vrsum0_s1, k+0, 3, 1) ;
      VFADD(vrgout_r3, vrsum_s2, vrsum0_s2, k+0, 3, 2) ;
      VFADD(vrgout_r3, vrsum_s3, vrsum0_s3, k+0, 3, 3) ;

#undef VFADD
#undef FILTER_OFFSET
    }

    if(remain) {
      vrsum0_s0 = _vel_vex_vvmvl(vrsum0_s0, vmx_s0, _vel_vbrds_vsl(0.f, vl), vl) ;
      vrsum0_s1 = _vel_vex_vvmvl(vrsum0_s1, vmx_s1, _vel_vbrds_vsl(0.f, vl), vl) ;
      vrsum0_s2 = _vel_vex_vvmvl(vrsum0_s2, vmx_s2, _vel_vbrds_vsl(0.f, vl), vl) ;
      vrsum0_s3 = _vel_vex_vvmvl(vrsum0_s3, vmx_s3, _vel_vbrds_vsl(0.f, vl), vl) ;
      __vr vrsum0 = _vel_vfadds_vvvl(_vel_vfadds_vvvl(vrsum0_s0, vrsum0_s1, vl),
				     _vel_vfadds_vvvl(vrsum0_s2, vrsum0_s3, vl), vl) ;
      _vel_vstu_vssl(vrsum0, 4, pGIn+gInIndex + 0 * gInHeight * gInWidth, vl) ;
    }
#pragma clang loop unroll(full)
    for(int64_t cc=0; cc<nPacked; cc++) {
      __vr _vrsum_s0 = _vel_vex_vvmvl(vrsum_s0[cc], vmx_s0, _vel_pvbrd_vsl(0UL, vl), vl) ;
      __vr _vrsum_s1 = _vel_vex_vvmvl(vrsum_s1[cc], vmx_s1, _vel_pvbrd_vsl(0UL, vl), vl) ;
      __vr _vrsum_s2 = _vel_vex_vvmvl(vrsum_s2[cc], vmx_s2, _vel_pvbrd_vsl(0UL, vl), vl) ;
      __vr _vrsum_s3 = _vel_vex_vvmvl(vrsum_s3[cc], vmx_s3, _vel_pvbrd_vsl(0UL, vl), vl) ;
      __vr vrsum = _vel_pvfadd_vvvl(_vel_pvfadd_vvvl(_vrsum_s0, _vrsum_s1, vl),
				    _vel_pvfadd_vvvl(_vrsum_s2, _vrsum_s3, vl), vl) ;
      _vel_vstu_vssl(vrsum, 4, pGIn+gInIndex + (2*cc+remain)   * gInHeight * gInWidth, vl) ;
      _vel_vstl_vssl(vrsum, 4, pGIn+gInIndex + (2*cc+remain+1) * gInHeight * gInWidth, vl) ;
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
    const int64_t padWidth,		// 0
    const int64_t padHeight,		// 0
    const int64_t dilationWidth,	// 1
    const int64_t dilationHeight	// 1
)
{

  const int64_t nH = VLEN / gInWidth ;

  __vr vrseq = _vel_vseq_vl(nH*gInWidth) ;
  __vr vrh  = _vel_vdivsl_vvsl(vrseq, gInWidth, nH*gInWidth) ;
  __vr vrw  = _vel_vsubsl_vvvl(vrseq, _vel_vmulul_vsvl(gInWidth,vrh, nH*gInWidth), nH*gInWidth) ;

  __vr vry  = _vel_vdivsl_vvsl(vrseq, gOutWidth, nH*gOutWidth) ;

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
	   n, c, nH, vrw, vry) ;
	c+=1 ;
	break ;
      case 2:
	func<FLAYOUT, 2>(pGOut, pKernel, pGIn,
	   gOutChannel, gOutWidth, gOutHeight,
	   gInChannel, gInWidth, gInHeight,
	   kernWidth, kernHeight,
	   gInChannelGroup, gOutChannelGroup,
	   gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	   n, c, nH, vrw, vry) ;
	c+=2 ;
	break ;
      case 3:
	func<FLAYOUT, 3>(pGOut, pKernel, pGIn,
	   gOutChannel, gOutWidth, gOutHeight,
	   gInChannel, gInWidth, gInHeight,
	   kernWidth, kernHeight,
	   gInChannelGroup, gOutChannelGroup,
	   gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	   n, c, nH, vrw, vry) ;
	c+=3 ;
	break ;
      case 4:
	func<FLAYOUT, 4>(pGOut, pKernel, pGIn,
	   gOutChannel, gOutWidth, gOutHeight,
	   gInChannel, gInWidth, gInHeight,
	   kernWidth, kernHeight,
	   gInChannelGroup, gOutChannelGroup,
	   gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	   n, c, nH, vrw, vry) ;
	c+=4 ;
	break ;
      case 5:
	func<FLAYOUT, 5>(pGOut, pKernel, pGIn,
	   gOutChannel, gOutWidth, gOutHeight,
	   gInChannel, gInWidth, gInHeight,
	   kernWidth, kernHeight,
	   gInChannelGroup, gOutChannelGroup,
	   gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	   n, c, nH, vrw, vry) ;
	c+=5 ;
	break ;
      case 6:
	func<FLAYOUT, 6>(pGOut, pKernel, pGIn,
	   gOutChannel, gOutWidth, gOutHeight,
	   gInChannel, gInWidth, gInHeight,
	   kernWidth, kernHeight,
	   gInChannelGroup, gOutChannelGroup,
	   gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	   n, c, nH, vrw, vry) ;
	c+=6 ;
	break ;
      case 7:
	func<FLAYOUT, 7>(pGOut, pKernel, pGIn,
	   gOutChannel, gOutWidth, gOutHeight,
	   gInChannel, gInWidth, gInHeight,
	   kernWidth, kernHeight,
	   gInChannelGroup, gOutChannelGroup,
	   gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	   n, c, nH, vrw, vry) ;
	c+=7 ;
	break ;
      case 8:
	func<FLAYOUT, 8>(pGOut, pKernel, pGIn,
	   gOutChannel, gOutWidth, gOutHeight,
	   gInChannel, gInWidth, gInHeight,
	   kernWidth, kernHeight,
	   gInChannelGroup, gOutChannelGroup,
	   gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	   n, c, nH, vrw, vry) ;
	c+=8 ;
	break ;
      case 9:
	func<FLAYOUT, 9>(pGOut, pKernel, pGIn,
	   gOutChannel, gOutWidth, gOutHeight,
	   gInChannel, gInWidth, gInHeight,
	   kernWidth, kernHeight,
	   gInChannelGroup, gOutChannelGroup,
	   gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	   n, c, nH, vrw, vry) ;
	c+=9 ;
	break ;
      case 10:
	func<FLAYOUT, 10>(pGOut, pKernel, pGIn,
	   gOutChannel, gOutWidth, gOutHeight,
	   gInChannel, gInWidth, gInHeight,
	   kernWidth, kernHeight,
	   gInChannelGroup, gOutChannelGroup,
	   gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	   n, c, nH, vrw, vry) ;
	c+=10 ;
	break ;
      case 11:
	func<FLAYOUT, 11>(pGOut, pKernel, pGIn,
	   gOutChannel, gOutWidth, gOutHeight,
	   gInChannel, gInWidth, gInHeight,
	   kernWidth, kernHeight,
	   gInChannelGroup, gOutChannelGroup,
	   gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	   n, c, nH, vrw, vry) ;
	c+=11 ;
	break ;
      case 12:
	func<FLAYOUT, 12>(pGOut, pKernel, pGIn,
	   gOutChannel, gOutWidth, gOutHeight,
	   gInChannel, gInWidth, gInHeight,
	   kernWidth, kernHeight,
	   gInChannelGroup, gOutChannelGroup,
	   gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	   n, c, nH, vrw, vry) ;
	c+=12 ;
	break ;
      case 13:
	func<FLAYOUT, 13>(pGOut, pKernel, pGIn,
	   gOutChannel, gOutWidth, gOutHeight,
	   gInChannel, gInWidth, gInHeight,
	   kernWidth, kernHeight,
	   gInChannelGroup, gOutChannelGroup,
	   gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	   n, c, nH, vrw, vry) ;
	c+=13 ;
	break ;
      case 14:
	func<FLAYOUT, 14>(pGOut, pKernel, pGIn,
	   gOutChannel, gOutWidth, gOutHeight,
	   gInChannel, gInWidth, gInHeight,
	   kernWidth, kernHeight,
	   gInChannelGroup, gOutChannelGroup,
	   gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	   n, c, nH, vrw, vry) ;
	c+=14 ;
	break ;
      case 15:
	func<FLAYOUT, 15>(pGOut, pKernel, pGIn,
	   gOutChannel, gOutWidth, gOutHeight,
	   gInChannel, gInWidth, gInHeight,
	   kernWidth, kernHeight,
	   gInChannelGroup, gOutChannelGroup,
	   gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	   n, c, nH, vrw, vry) ;
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
	   n, c, nH, vrw, vry) ;
	c+= 16 ;
      } // gInChannel
    } // group
  } // batch
}

extern "C"
vednnError_t
vednnConvolutionBackwardData_direct_dil1_str1_pad0_ker4_iwU128(
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
