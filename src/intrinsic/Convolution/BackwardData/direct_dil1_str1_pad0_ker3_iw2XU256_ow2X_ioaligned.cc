#include <stdint.h>
#include <stdlib.h>

#include "vednn.h"
#include "vednn_util.h"

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
    const int64_t nH
)
{
  int64_t gInIndex = gInGroupOffset + ((n * gInChannel + c) * gInHeight ) * gInWidth  ;

  const int64_t gInWidthHalf = gInWidth >> 1 ;

  __vr vrseq = _vel_vseq_vl(nH*gInWidthHalf) ;
  __vr vrh  = _vel_vdivsl_vvsl(vrseq, gInWidthHalf, nH*gInWidthHalf) ;
  __vr vrw  = _vel_vmulsl_vsvl(2, _vel_vsubsl_vvvl(vrseq, _vel_vmulul_vsvl(gInWidthHalf,vrh, nH*gInWidthHalf), nH*gInWidthHalf), nH*gInWidthHalf) ;
  __vr vrhw = _vel_vaddsl_vvvl(_vel_vmulsl_vsvl(gOutWidth, vrh, nH*gInWidthHalf), vrw, nH*gInWidthHalf) ;

  __vr vrx_s2 = _vel_vaddsl_vsvl(-2, vrw, nH*gInWidthHalf) ;
  __vm256 vmx_s2 =  _vel_vfmklge_mvl(vrx_s2, nH*gInWidthHalf) ;

  __vr vrx_s0 = vrw ;
  __vm256 vmx_s0 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutWidth,vrx_s0, nH*gInWidthHalf), nH*gInWidthHalf) ;

  for (int64_t h=0; h<gInHeight; h+=nH) {
    const int64_t vl = gInWidthHalf * (gInHeight - h < nH ? gInHeight - h : nH) ;

    __vr vrsum0 = _vel_pvbrd_vsl(0UL, vl) ;
    __vr vrsum1 = _vel_pvbrd_vsl(0UL, vl) ;
    __vr vrsum2 = _vel_pvbrd_vsl(0UL, vl) ;
    __vr vrsum3 = _vel_pvbrd_vsl(0UL, vl) ;
    __vr vrsum4 = _vel_pvbrd_vsl(0UL, vl) ;
    __vr vrsum5 = _vel_pvbrd_vsl(0UL, vl) ;
    __vr vrsum6 = _vel_pvbrd_vsl(0UL, vl) ;
    __vr vrsum7 = _vel_pvbrd_vsl(0UL, vl) ;
    __vr vrsum8 = _vel_pvbrd_vsl(0UL, vl) ;
    __vr vrsum9 = _vel_pvbrd_vsl(0UL, vl) ;
    __vr vrsumA = _vel_pvbrd_vsl(0UL, vl) ;
    __vr vrsumB = _vel_pvbrd_vsl(0UL, vl) ;
    __vr vrsumC = _vel_pvbrd_vsl(0UL, vl) ;
    __vr vrsumD = _vel_pvbrd_vsl(0UL, vl) ;
    __vr vrsumE = _vel_pvbrd_vsl(0UL, vl) ;
    __vr vrsumF = _vel_pvbrd_vsl(0UL, vl) ;

    __vr vry_r2 = _vel_vaddsl_vsvl(h-2, vrh, vl) ;
    __vm256 vmy1_r2 =  _vel_vfmklge_mvl(vry_r2, vl) ;
    __vm256 vmy_r2 = vmy1_r2 ;

    __vm256 vmall_r2s2 = _vel_andm_mmm(vmy_r2,vmx_s2) ;
    __vm256 vmall_r2s0 = _vel_andm_mmm(vmy_r2,vmx_s0) ;

    __vr vry_r1 = _vel_vaddsl_vsvl(h-1, vrh, vl) ;
    __vm256 vmy1_r1 =  _vel_vfmklge_mvl(vry_r1, vl) ;
    __vm256 vmy2_r1 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutHeight,vry_r1, vl), vl) ;
    __vm256 vmy_r1 = _vel_andm_mmm(vmy1_r1, vmy2_r1) ;

    __vm256 vmall_r1s2 = _vel_andm_mmm(vmy_r1,vmx_s2) ;
    __vm256 vmall_r1s0 = _vel_andm_mmm(vmy_r1,vmx_s0) ;

    __vr vry_r0= _vel_vaddsl_vsvl(h, vrh, vl) ;
    __vm256 vmy2_r0 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutHeight,vry_r0, vl), vl) ;
    __vm256 vmy_r0 = vmy2_r0 ;

    __vm256 vmall_r0s2 = _vel_andm_mmm(vmy_r0,vmx_s2) ;
    __vm256 vmall_r0s0 = _vel_andm_mmm(vmy_r0,vmx_s0) ;

    for (int64_t k=0; k<gOutChannelGroup; k++) {
      int64_t gOutIndex    = gOutGroupOffset + ((n * gOutChannel + k) * gOutHeight) * gOutWidth ;

      __vr vrgout_ptr_r2s2 = _vel_vsfa_vvssl(vrhw, 2, (unsigned long)(pGOut+gOutIndex+(h-2)*gOutHeight-2), vl) ;
      __vr vrgout_ptr_r1s2 = _vel_vsfa_vvssl(vrhw, 2, (unsigned long)(pGOut+gOutIndex+(h-1)*gOutHeight-2), vl) ;
      __vr vrgout_ptr_r0s2 = _vel_vsfa_vvssl(vrhw, 2, (unsigned long)(pGOut+gOutIndex+(h-0)*gOutHeight-2), vl) ;

      __vr vrgout_r2s2 = _vel_vgt_vvssml(vrgout_ptr_r2s2, 0, 0, vmall_r2s2, vl) ;
      __vr vrgout_r1s2 = _vel_vgt_vvssml(vrgout_ptr_r1s2, 0, 0, vmall_r1s2, vl) ;
      __vr vrgout_r0s2 = _vel_vgt_vvssml(vrgout_ptr_r0s2, 0, 0, vmall_r0s2, vl) ;

      __vr vrgout_r2s0 = _vel_vmv_vsvl(1, vrgout_r2s2, vl) ;
      __vr vrgout_r1s0 = _vel_vmv_vsvl(1, vrgout_r1s2, vl) ;
      __vr vrgout_r0s0 = _vel_vmv_vsvl(1, vrgout_r0s2, vl) ;

#define FILTER_OFFSET(k,c,r,s)  ( kernGroupOffset + filter_index<FLAYOUT>(k,c,r,s, gInChannelGroup, gOutChannelGroup, kernHeight, kernWidth) )

#define VFADD(VRGOUT, K, R, S)													\
      {																	\
	if(NUMCHANNEL>= 1) vrsum0 = _vel_pvfmad_vvsvl(vrsum0, _vel_pack_f32a(pKernel + FILTER_OFFSET(K,c+ 0,R,S)), VRGOUT, vl) ;	\
	if(NUMCHANNEL>= 2) vrsum1 = _vel_pvfmad_vvsvl(vrsum1, _vel_pack_f32a(pKernel + FILTER_OFFSET(K,c+ 1,R,S)), VRGOUT, vl) ;	\
	if(NUMCHANNEL>= 3) vrsum2 = _vel_pvfmad_vvsvl(vrsum2, _vel_pack_f32a(pKernel + FILTER_OFFSET(K,c+ 2,R,S)), VRGOUT, vl) ;	\
	if(NUMCHANNEL>= 4) vrsum3 = _vel_pvfmad_vvsvl(vrsum3, _vel_pack_f32a(pKernel + FILTER_OFFSET(K,c+ 3,R,S)), VRGOUT, vl) ;	\
	if(NUMCHANNEL>= 5) vrsum4 = _vel_pvfmad_vvsvl(vrsum4, _vel_pack_f32a(pKernel + FILTER_OFFSET(K,c+ 4,R,S)), VRGOUT, vl) ;	\
	if(NUMCHANNEL>= 6) vrsum5 = _vel_pvfmad_vvsvl(vrsum5, _vel_pack_f32a(pKernel + FILTER_OFFSET(K,c+ 5,R,S)), VRGOUT, vl) ;	\
	if(NUMCHANNEL>= 7) vrsum6 = _vel_pvfmad_vvsvl(vrsum6, _vel_pack_f32a(pKernel + FILTER_OFFSET(K,c+ 6,R,S)), VRGOUT, vl) ;	\
	if(NUMCHANNEL>= 8) vrsum7 = _vel_pvfmad_vvsvl(vrsum7, _vel_pack_f32a(pKernel + FILTER_OFFSET(K,c+ 7,R,S)), VRGOUT, vl) ;	\
	if(NUMCHANNEL>= 9) vrsum8 = _vel_pvfmad_vvsvl(vrsum8, _vel_pack_f32a(pKernel + FILTER_OFFSET(K,c+ 8,R,S)), VRGOUT, vl) ;	\
	if(NUMCHANNEL>=10) vrsum9 = _vel_pvfmad_vvsvl(vrsum9, _vel_pack_f32a(pKernel + FILTER_OFFSET(K,c+ 9,R,S)), VRGOUT, vl) ;	\
	if(NUMCHANNEL>=11) vrsumA = _vel_pvfmad_vvsvl(vrsumA, _vel_pack_f32a(pKernel + FILTER_OFFSET(K,c+10,R,S)), VRGOUT, vl) ;	\
	if(NUMCHANNEL>=12) vrsumB = _vel_pvfmad_vvsvl(vrsumB, _vel_pack_f32a(pKernel + FILTER_OFFSET(K,c+11,R,S)), VRGOUT, vl) ;	\
	if(NUMCHANNEL>=13) vrsumC = _vel_pvfmad_vvsvl(vrsumC, _vel_pack_f32a(pKernel + FILTER_OFFSET(K,c+12,R,S)), VRGOUT, vl) ;	\
	if(NUMCHANNEL>=14) vrsumD = _vel_pvfmad_vvsvl(vrsumD, _vel_pack_f32a(pKernel + FILTER_OFFSET(K,c+13,R,S)), VRGOUT, vl) ;	\
	if(NUMCHANNEL>=15) vrsumE = _vel_pvfmad_vvsvl(vrsumE, _vel_pack_f32a(pKernel + FILTER_OFFSET(K,c+14,R,S)), VRGOUT, vl) ;	\
	if(NUMCHANNEL>=16) vrsumF = _vel_pvfmad_vvsvl(vrsumF, _vel_pack_f32a(pKernel + FILTER_OFFSET(K,c+15,R,S)), VRGOUT, vl) ;	\
      }

      vrgout_r2s2 = _vel_vmrg_vsvml(0UL, vrgout_r2s2, vmall_r2s2, vl) ;
      vrgout_r2s0 = _vel_vmrg_vsvml(0UL, vrgout_r2s0, vmall_r2s0, vl) ;
      __vr vrgout_r2s1 = _vel_vshf_vvvsl(vrgout_r2s2, vrgout_r2s0, VE_VSHUFFLE_ZLYU, vl) ;
      VFADD(vrgout_r2s2, k, 2, 2) ;
      VFADD(vrgout_r2s1, k, 2, 1) ;
      VFADD(vrgout_r2s0, k, 2, 0) ;

      vrgout_r1s2 = _vel_vmrg_vsvml(0UL, vrgout_r1s2, vmall_r1s2, vl) ;
      vrgout_r1s0 = _vel_vmrg_vsvml(0UL, vrgout_r1s0, vmall_r1s0, vl) ;
      __vr vrgout_r1s1 = _vel_vshf_vvvsl(vrgout_r1s2, vrgout_r1s0, VE_VSHUFFLE_ZLYU, vl) ;
      VFADD(vrgout_r1s2, k, 1, 2) ;
      VFADD(vrgout_r1s1, k, 1, 1) ;
      VFADD(vrgout_r1s0, k, 1, 0) ;

      vrgout_r0s2 = _vel_vmrg_vsvml(0UL, vrgout_r0s2, vmall_r0s2, vl) ;
      vrgout_r0s0 = _vel_vmrg_vsvml(0UL, vrgout_r0s0, vmall_r0s0, vl) ;
      __vr vrgout_r0s1 = _vel_vshf_vvvsl(vrgout_r0s2, vrgout_r0s0, VE_VSHUFFLE_ZLYU, vl) ;
      VFADD(vrgout_r0s2, k, 0, 2) ;
      VFADD(vrgout_r0s1, k, 0, 1) ;
      VFADD(vrgout_r0s0, k, 0, 0) ;

#undef VFADD
#undef FILTER_OFFSET
    } // gOutChannel

    if(NUMCHANNEL>= 1) _vel_vst_vssl(vrsum0, 8, pGIn+gInIndex + 0 * gInHeight * gInWidth, vl) ;
    if(NUMCHANNEL>= 2) _vel_vst_vssl(vrsum1, 8, pGIn+gInIndex + 1 * gInHeight * gInWidth, vl) ;
    if(NUMCHANNEL>= 3) _vel_vst_vssl(vrsum2, 8, pGIn+gInIndex + 2 * gInHeight * gInWidth, vl) ;
    if(NUMCHANNEL>= 4) _vel_vst_vssl(vrsum3, 8, pGIn+gInIndex + 3 * gInHeight * gInWidth, vl) ;
    if(NUMCHANNEL>= 5) _vel_vst_vssl(vrsum4, 8, pGIn+gInIndex + 4 * gInHeight * gInWidth, vl) ;
    if(NUMCHANNEL>= 6) _vel_vst_vssl(vrsum5, 8, pGIn+gInIndex + 5 * gInHeight * gInWidth, vl) ;
    if(NUMCHANNEL>= 7) _vel_vst_vssl(vrsum6, 8, pGIn+gInIndex + 6 * gInHeight * gInWidth, vl) ;
    if(NUMCHANNEL>= 8) _vel_vst_vssl(vrsum7, 8, pGIn+gInIndex + 7 * gInHeight * gInWidth, vl) ;
    if(NUMCHANNEL>= 9) _vel_vst_vssl(vrsum8, 8, pGIn+gInIndex + 8 * gInHeight * gInWidth, vl) ;
    if(NUMCHANNEL>=10) _vel_vst_vssl(vrsum9, 8, pGIn+gInIndex + 9 * gInHeight * gInWidth, vl) ;
    if(NUMCHANNEL>=11) _vel_vst_vssl(vrsumA, 8, pGIn+gInIndex +10 * gInHeight * gInWidth, vl) ;
    if(NUMCHANNEL>=12) _vel_vst_vssl(vrsumB, 8, pGIn+gInIndex +11 * gInHeight * gInWidth, vl) ;
    if(NUMCHANNEL>=13) _vel_vst_vssl(vrsumC, 8, pGIn+gInIndex +12 * gInHeight * gInWidth, vl) ;
    if(NUMCHANNEL>=14) _vel_vst_vssl(vrsumD, 8, pGIn+gInIndex +13 * gInHeight * gInWidth, vl) ;
    if(NUMCHANNEL>=15) _vel_vst_vssl(vrsumE, 8, pGIn+gInIndex +14 * gInHeight * gInWidth, vl) ;
    if(NUMCHANNEL>=15) _vel_vst_vssl(vrsumE, 8, pGIn+gInIndex +14 * gInHeight * gInWidth, vl) ;
    if(NUMCHANNEL>=16) _vel_vst_vssl(vrsumF, 8, pGIn+gInIndex +15 * gInHeight * gInWidth, vl) ;

    gInIndex += 2*vl ;
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
    const int64_t gOutChannelGroup
)
{
  const int64_t gInWidthHalf = gInWidth >> 1 ;
  const int64_t nH = VLEN / gInWidthHalf ;

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
	   n, c, nH ) ;
	c+=1 ;
	break ;
      case 2:
	func<FLAYOUT, 2>(pGOut, pKernel, pGIn,
	   gOutChannel, gOutWidth, gOutHeight,
	   gInChannel, gInWidth, gInHeight,
	   kernWidth, kernHeight,
	   gInChannelGroup, gOutChannelGroup,
	   gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	   n, c, nH ) ;
	c+=2 ;
	break ;
      case 3:
	func<FLAYOUT, 3>(pGOut, pKernel, pGIn,
	   gOutChannel, gOutWidth, gOutHeight,
	   gInChannel, gInWidth, gInHeight,
	   kernWidth, kernHeight,
	   gInChannelGroup, gOutChannelGroup,
	   gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	   n, c, nH ) ;
	c+=3 ;
	break ;
      case 4:
	func<FLAYOUT, 4>(pGOut, pKernel, pGIn,
	   gOutChannel, gOutWidth, gOutHeight,
	   gInChannel, gInWidth, gInHeight,
	   kernWidth, kernHeight,
	   gInChannelGroup, gOutChannelGroup,
	   gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	   n, c, nH ) ;
	c+=4 ;
	break ;
      case 5:
	func<FLAYOUT, 5>(pGOut, pKernel, pGIn,
	   gOutChannel, gOutWidth, gOutHeight,
	   gInChannel, gInWidth, gInHeight,
	   kernWidth, kernHeight,
	   gInChannelGroup, gOutChannelGroup,
	   gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	   n, c, nH ) ;
	c+=5 ;
	break ;
      case 6:
	func<FLAYOUT, 6>(pGOut, pKernel, pGIn,
	   gOutChannel, gOutWidth, gOutHeight,
	   gInChannel, gInWidth, gInHeight,
	   kernWidth, kernHeight,
	   gInChannelGroup, gOutChannelGroup,
	   gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	   n, c, nH ) ;
	c+=6 ;
	break ;
      case 7:
	func<FLAYOUT, 7>(pGOut, pKernel, pGIn,
	   gOutChannel, gOutWidth, gOutHeight,
	   gInChannel, gInWidth, gInHeight,
	   kernWidth, kernHeight,
	   gInChannelGroup, gOutChannelGroup,
	   gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	   n, c, nH ) ;
	c+=7 ;
	break ;
      case 8:
	func<FLAYOUT, 8>(pGOut, pKernel, pGIn,
	   gOutChannel, gOutWidth, gOutHeight,
	   gInChannel, gInWidth, gInHeight,
	   kernWidth, kernHeight,
	   gInChannelGroup, gOutChannelGroup,
	   gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	   n, c, nH ) ;
	c+=8 ;
	break ;
      case 9:
	func<FLAYOUT, 9>(pGOut, pKernel, pGIn,
	   gOutChannel, gOutWidth, gOutHeight,
	   gInChannel, gInWidth, gInHeight,
	   kernWidth, kernHeight,
	   gInChannelGroup, gOutChannelGroup,
	   gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	   n, c, nH ) ;
	c+=9 ;
	break ;
      case 10:
	func<FLAYOUT, 10>(pGOut, pKernel, pGIn,
	   gOutChannel, gOutWidth, gOutHeight,
	   gInChannel, gInWidth, gInHeight,
	   kernWidth, kernHeight,
	   gInChannelGroup, gOutChannelGroup,
	   gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	   n, c, nH ) ;
	c+=10 ;
	break ;
      case 11:
	func<FLAYOUT, 11>(pGOut, pKernel, pGIn,
	   gOutChannel, gOutWidth, gOutHeight,
	   gInChannel, gInWidth, gInHeight,
	   kernWidth, kernHeight,
	   gInChannelGroup, gOutChannelGroup,
	   gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	   n, c, nH ) ;
	c+=11 ;
	break ;
      case 12:
	func<FLAYOUT, 12>(pGOut, pKernel, pGIn,
	   gOutChannel, gOutWidth, gOutHeight,
	   gInChannel, gInWidth, gInHeight,
	   kernWidth, kernHeight,
	   gInChannelGroup, gOutChannelGroup,
	   gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	   n, c, nH ) ;
	c+=12 ;
	break ;
      case 13:
	func<FLAYOUT, 13>(pGOut, pKernel, pGIn,
	   gOutChannel, gOutWidth, gOutHeight,
	   gInChannel, gInWidth, gInHeight,
	   kernWidth, kernHeight,
	   gInChannelGroup, gOutChannelGroup,
	   gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	   n, c, nH ) ;
	c+=13 ;
	break ;
      case 14:
	func<FLAYOUT, 14>(pGOut, pKernel, pGIn,
	   gOutChannel, gOutWidth, gOutHeight,
	   gInChannel, gInWidth, gInHeight,
	   kernWidth, kernHeight,
	   gInChannelGroup, gOutChannelGroup,
	   gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	   n, c, nH ) ;
	c+=14 ;
	break ;
      case 15:
	func<FLAYOUT, 15>(pGOut, pKernel, pGIn,
	   gOutChannel, gOutWidth, gOutHeight,
	   gInChannel, gInWidth, gInHeight,
	   kernWidth, kernHeight,
	   gInChannelGroup, gOutChannelGroup,
	   gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	   n, c, nH ) ;
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
	   n, c, nH ) ;
	c+= 16 ;
      } // gInChannel
    } // group
  } // batch
}

extern "C"
vednnError_t
vednnConvolutionBackwardData_direct_dil1_str1_pad0_ker3_iw2XU256_ow2X_ioaligned(
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
//  const int64_t strideWidth    = pParamConv->strideWidth;	// 1
//  const int64_t strideHeight   = pParamConv->strideHeight;	// 1
//  const int64_t padWidth       = pParamConv->padWidth;	// 0
//  const int64_t padHeight      = pParamConv->padHeight;	// 0
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
