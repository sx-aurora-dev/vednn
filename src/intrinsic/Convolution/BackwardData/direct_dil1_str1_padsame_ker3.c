#include <stdint.h>
#include <stdlib.h>

#include "vednn.h"

#include "veintrin.h"
#define VLEN	(256)

vednnError_t
vednnConvolutionBackwardData_direct_dil1_str1_padsame_ker3 (
    const vednnTensorParam_t * restrict 	pParamGradOut,
    const void * restrict 			pDataGradOut,
    const vednnFilterParam_t * restrict 	pParamKernel,
    const void * restrict 			pDataKernel,
    const vednnConvolutionParam_t * restrict 	pParamConv,
    const vednnTensorParam_t * restrict 	pParamGradIn,
    void * restrict 				pDataGradIn
)
{
  const int64_t batch       = pParamGradOut->batch;
  const int64_t gOutChannel = pParamGradOut->channel;
  const int64_t gOutWidth   = pParamGradOut->width;
  const int64_t gOutHeight  = pParamGradOut->height;
  const int64_t gInChannel  = pParamGradIn->channel;
  const int64_t gInWidth    = pParamGradIn->width;
  const int64_t gInHeight   = pParamGradIn->height;
  const int64_t kernWidth   = pParamKernel->width;		/* must be 2*padWidth  + 1 */
  const int64_t kernHeight  = pParamKernel->height;		/* must be 2*padHeight + 1 */

  const int64_t group          = pParamConv->group;
//  const int64_t strideWidth    = pParamConv->strideWidth; 	/* must be 1 */
//  const int64_t strideHeight   = pParamConv->strideHeight;	/* must be 1 */
  const int64_t padWidth       = pParamConv->padWidth;
  const int64_t padHeight      = pParamConv->padHeight;

//  const int64_t dilationWidth  = pParamConv->dilationWidth;	/* must be 1 */
//  const int64_t dilationHeight = pParamConv->dilationHeight;	/* must be 1 */

  const int64_t gOutChannelGroup = gOutChannel / group;
  const int64_t gInChannelGroup  = gInChannel  / group;

  const float * restrict pGOut   = pDataGradOut;
  const float * restrict pKernel = pDataKernel;
  float * restrict const pGIn    = pDataGradIn;

  const int gInPixels= gInHeight*gInWidth ;

  /* intrinsic version 1 */
  {

    for (int64_t n=0; n<batch; n++) {
      for (int64_t g = 0; g < group; g++) {

	int64_t gInGroupOffset  = g * gInChannelGroup * gInHeight * gInWidth;
	int64_t gOutGroupOffset = g * gOutChannelGroup  * gOutHeight  * gOutWidth;
	int64_t kernGroupOffset = g * gOutChannelGroup  * gInChannelGroup * kernHeight * kernWidth;

	int k=0;
	if ( (gInChannelGroup & 0x01) == 1 ) {
	  for (int64_t gip = 0; gip < gInPixels; gip+=VLEN) {
	    const int64_t vl = gInPixels - gip < VLEN ? gInPixels - gip : VLEN ;

	    const int64_t gInIndex = gInGroupOffset + ((n * gInChannel + k) * gInHeight ) * gInWidth + gip ;

	    _ve_lvl(vl) ;

	    __vr vrseq = _ve_vseq_v() ;			// hw
	    __vr vridx = _ve_vaddsl_vsv(gip, vrseq) ;	// op + hw

	    __vr vrsum = _ve_vbrdu_vs_f32(0.f) ;

	    __vr vrh   = _ve_vdivsl_vvs(vridx, gInWidth) ;
	    __vr vrix   = _ve_vsubsl_vvv(vridx, _ve_vmulul_vsv(gInWidth,vrh)) ;

	    __vr vry_r0 = _ve_vaddsl_vsv(1, vrh) ;
	    __vr vry_r2 = _ve_vaddsl_vsv(-1, vrh) ;

	    __vr vrx_s0 = _ve_vaddsl_vsv(1, vrix) ;
	    __vr vrx_s2 = _ve_vaddsl_vsv(-1, vrix) ;

	    __vm256 vmy_r0 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(gOutHeight,vry_r0)) ;
	    __vm256 vmy_r2 = _ve_vfmkl_mcv(VECC_GE, vry_r2) ;

	    __vm256 vmx_s0 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(gOutWidth,vrx_s0)) ;
	    __vm256 vmx_s2 = _ve_vfmkl_mcv(VECC_GE, vrx_s2) ;

	    __vm256 vmall_r0s0 = _ve_andm_mmm(vmy_r0,vmx_s0) ;
	    __vm256 vmall_r0s1 = vmy_r0 ;
	    __vm256 vmall_r0s2 = _ve_andm_mmm(vmy_r0,vmx_s2) ;

	    __vm256 vmall_r1s0 = vmx_s0 ;
	    __vm256 vmall_r1s2 = vmx_s2 ;

	    __vm256 vmall_s0_r2 = _ve_andm_mmm(vmy_r2,vmx_s0) ;
	    __vm256 vmall_s1_r2 = vmy_r2 ;
	    __vm256 vmall_s2_r2 = _ve_andm_mmm(vmy_r2,vmx_s2) ;

	    for (int64_t c=0; c<gOutChannelGroup; c++) {
	      const float *pGOutChannel = pGOut + gOutGroupOffset + ((n * gOutChannel + c) * gOutHeight * gOutWidth ) ;
	      const float *pKerValue = pKernel + kernGroupOffset + ((c * gInChannelGroup + k) * kernHeight) * kernWidth ;

	      /* memory access errors mihgt be caused */
	      __vr vrgout_r0s0 = _ve_vldu_vss(4,&pGOutChannel[gip+(padHeight-0)*gOutWidth+(padWidth-0)]) ;
	      __vr vrgout_r0s1 = _ve_vldu_vss(4,&pGOutChannel[gip+(padHeight-0)*gOutWidth+(padWidth-1)]) ;
	      __vr vrgout_r0s2 = _ve_vldu_vss(4,&pGOutChannel[gip+(padHeight-0)*gOutWidth+(padWidth-2)]) ;
	      __vr vrgout_r1s0 = _ve_vldu_vss(4,&pGOutChannel[gip+(padHeight-1)*gOutWidth+(padWidth-0)]) ;
	      __vr vrgout_r1s1 = _ve_vldu_vss(4,&pGOutChannel[gip+(padHeight-1)*gOutWidth+(padWidth-1)]) ;
	      __vr vrgout_r1s2 = _ve_vldu_vss(4,&pGOutChannel[gip+(padHeight-1)*gOutWidth+(padWidth-2)]) ;
	      __vr vrgout_r2s0 = _ve_vldu_vss(4,&pGOutChannel[gip+(padHeight-2)*gOutWidth+(padWidth-0)]) ;
	      __vr vrgout_r2s1 = _ve_vldu_vss(4,&pGOutChannel[gip+(padHeight-2)*gOutWidth+(padWidth-1)]) ;
	      __vr vrgout_r2s2 = _ve_vldu_vss(4,&pGOutChannel[gip+(padHeight-2)*gOutWidth+(padWidth-2)]) ;

	      vrgout_r0s0 = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), vrgout_r0s0, vmall_r0s0) ;
	      vrgout_r0s1 = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), vrgout_r0s1, vmall_r0s1) ;
	      vrgout_r0s2 = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), vrgout_r0s2, vmall_r0s2) ;

	      vrsum = _ve_vfmads_vvsv(vrsum, *pKerValue, vrgout_r0s0) ; pKerValue++ ;
	      vrsum = _ve_vfmads_vvsv(vrsum, *pKerValue, vrgout_r0s1) ; pKerValue++ ;
	      vrsum = _ve_vfmads_vvsv(vrsum, *pKerValue, vrgout_r0s2) ; pKerValue++ ;


	      vrgout_r1s0 = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), vrgout_r1s0, vmall_r1s0) ;
	      vrgout_r1s2 = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), vrgout_r1s2, vmall_r1s2) ;

	      vrsum = _ve_vfmads_vvsv(vrsum, *pKerValue, vrgout_r1s0) ; pKerValue++ ;
	      vrsum = _ve_vfmads_vvsv(vrsum, *pKerValue, vrgout_r1s1) ; pKerValue++ ;
	      vrsum = _ve_vfmads_vvsv(vrsum, *pKerValue, vrgout_r1s2) ; pKerValue++ ;


	      vrgout_r2s0 = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), vrgout_r2s0, vmall_s0_r2) ;
	      vrgout_r2s1 = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), vrgout_r2s1, vmall_s1_r2) ;
	      vrgout_r2s2 = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), vrgout_r2s2, vmall_s2_r2) ;

	      vrsum = _ve_vfmads_vvsv(vrsum, *pKerValue, vrgout_r2s0) ; pKerValue++ ;
	      vrsum = _ve_vfmads_vvsv(vrsum, *pKerValue, vrgout_r2s1) ; pKerValue++ ;
	      vrsum = _ve_vfmads_vvsv(vrsum, *pKerValue, vrgout_r2s2) ; pKerValue++ ;

	    } // gInChannel

	    _ve_vstu_vss(vrsum, 4, pGIn+gInIndex) ;

	  } // gInPixels

	  k+=1 ;
	}
	if ( ((gInChannelGroup >> 1) & 0x01) == 1 ) {
	  for (int64_t gip = 0; gip < gInPixels; gip+=VLEN) {
	    const int64_t vl = gInPixels - gip < VLEN ? gInPixels - gip : VLEN ;

	    const int64_t gInIndex = gInGroupOffset + ((n * gInChannel + k) * gInHeight ) * gInWidth + gip ;

	    _ve_lvl(vl) ;

	    __vr vrseq = _ve_vseq_v() ;			// hw
	    __vr vridx = _ve_vaddsl_vsv(gip, vrseq) ;	// op + hw

	    __vr vrsum01 = _ve_pvbrd_vs_i64(0UL) ;

	    __vr vrh   = _ve_vdivsl_vvs(vridx, gInWidth) ;
	    __vr vrix   = _ve_vsubsl_vvv(vridx, _ve_vmulul_vsv(gInWidth,vrh)) ;

	    __vr vry_r0 = _ve_vaddsl_vsv(1, vrh) ;
	    __vr vry_r2 = _ve_vaddsl_vsv(-1, vrh) ;

	    __vr vrx_s0 = _ve_vaddsl_vsv(1, vrix) ;
	    __vr vrx_s2 = _ve_vaddsl_vsv(-1, vrix) ;

	    __vm256 vmy_r0 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(gOutHeight,vry_r0)) ;
	    __vm256 vmy_r2 = _ve_vfmkl_mcv(VECC_GE, vry_r2) ;

	    __vm256 vmx_s0 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(gOutWidth,vrx_s0)) ;
	    __vm256 vmx_s2 = _ve_vfmkl_mcv(VECC_GE, vrx_s2) ;

	    __vm256 vmall_r0s0 = _ve_andm_mmm(vmy_r0,vmx_s0) ;
	    __vm256 vmall_r0s1 = vmy_r0 ;
	    __vm256 vmall_r0s2 = _ve_andm_mmm(vmy_r0,vmx_s2) ;

	    __vm256 vmall_r1s0 = vmx_s0 ;
	    __vm256 vmall_r1s2 = vmx_s2 ;

	    __vm256 vmall_s0_r2 = _ve_andm_mmm(vmy_r2,vmx_s0) ;
	    __vm256 vmall_s1_r2 = vmy_r2 ;
	    __vm256 vmall_s2_r2 = _ve_andm_mmm(vmy_r2,vmx_s2) ;

	    for (int64_t c=0; c<gOutChannelGroup; c++) {
	      const float *pGOutChannel = pGOut + gOutGroupOffset + ((n * gOutChannel + c) * gOutHeight * gOutWidth ) ;
	      const float *pKerValue = pKernel + kernGroupOffset + ((c * gInChannelGroup + k) * kernHeight) * kernWidth ;

	      /* memory access errors mihgt be caused */
	      __vr vrgout_r0s0 = _ve_vldu_vss(4,&pGOutChannel[gip+(padHeight-0)*gOutWidth+(padWidth-0)]) ;
	      __vr vrgout_r0s1 = _ve_vldu_vss(4,&pGOutChannel[gip+(padHeight-0)*gOutWidth+(padWidth-1)]) ;
	      __vr vrgout_r0s2 = _ve_vldu_vss(4,&pGOutChannel[gip+(padHeight-0)*gOutWidth+(padWidth-2)]) ;
	      vrgout_r0s0 = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), vrgout_r0s0, vmall_r0s0) ;
	      vrgout_r0s1 = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), vrgout_r0s1, vmall_r0s1) ;
	      vrgout_r0s2 = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), vrgout_r0s2, vmall_r0s2) ;
	      __vr vrgoutP_r0s0 = _ve_vshf_vvvs(vrgout_r0s0, vrgout_r0s0, VE_VSHUFFLE_YUZU) ;
	      __vr vrgoutP_r0s1 = _ve_vshf_vvvs(vrgout_r0s1, vrgout_r0s1, VE_VSHUFFLE_YUZU) ;
	      __vr vrgoutP_r0s2 = _ve_vshf_vvvs(vrgout_r0s2, vrgout_r0s2, VE_VSHUFFLE_YUZU) ;

#define FILTER_IC2(VROUT)								\
{											\
  const uint64_t kerValue01 = _ve_pack_f32p(pKerValue,					\
					    pKerValue+    kernHeight * kernWidth) ;	\
  vrsum01 = _ve_pvfmad_vvsv(vrsum01, kerValue01, VROUT) ;				\
}

	      FILTER_IC2(vrgoutP_r0s0) ; pKerValue++ ;
	      FILTER_IC2(vrgoutP_r0s1) ; pKerValue++ ;
	      FILTER_IC2(vrgoutP_r0s2) ; pKerValue++ ;

	      __vr vrgout_r1s0 = _ve_vldu_vss(4,&pGOutChannel[gip+(padHeight-1)*gOutWidth+(padWidth-0)]) ;
	      __vr vrgout_r1s1 = _ve_vldu_vss(4,&pGOutChannel[gip+(padHeight-1)*gOutWidth+(padWidth-1)]) ;
	      __vr vrgout_r1s2 = _ve_vldu_vss(4,&pGOutChannel[gip+(padHeight-1)*gOutWidth+(padWidth-2)]) ;
	      vrgout_r1s0 = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), vrgout_r1s0, vmall_r1s0) ;
	      vrgout_r1s2 = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), vrgout_r1s2, vmall_r1s2) ;
	      __vr vrgoutP_r1s0 = _ve_vshf_vvvs(vrgout_r1s0, vrgout_r1s0, VE_VSHUFFLE_YUZU) ;
	      __vr vrgoutP_r1s1 = _ve_vshf_vvvs(vrgout_r1s1, vrgout_r1s1, VE_VSHUFFLE_YUZU) ;
	      __vr vrgoutP_r1s2 = _ve_vshf_vvvs(vrgout_r1s2, vrgout_r1s2, VE_VSHUFFLE_YUZU) ;

	      FILTER_IC2(vrgoutP_r1s0) ; pKerValue++ ;
	      FILTER_IC2(vrgoutP_r1s1) ; pKerValue++ ;
	      FILTER_IC2(vrgoutP_r1s2) ; pKerValue++ ;


	      __vr vrgout_r2s0 = _ve_vldu_vss(4,&pGOutChannel[gip+(padHeight-2)*gOutWidth+(padWidth-0)]) ;
	      __vr vrgout_r2s1 = _ve_vldu_vss(4,&pGOutChannel[gip+(padHeight-2)*gOutWidth+(padWidth-1)]) ;
	      __vr vrgout_r2s2 = _ve_vldu_vss(4,&pGOutChannel[gip+(padHeight-2)*gOutWidth+(padWidth-2)]) ;
	      vrgout_r2s0 = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), vrgout_r2s0, vmall_s0_r2) ;
	      vrgout_r2s1 = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), vrgout_r2s1, vmall_s1_r2) ;
	      vrgout_r2s2 = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), vrgout_r2s2, vmall_s2_r2) ;
	      __vr vrgoutP_r2s0 = _ve_vshf_vvvs(vrgout_r2s0, vrgout_r2s0, VE_VSHUFFLE_YUZU) ;
	      __vr vrgoutP_r2s1 = _ve_vshf_vvvs(vrgout_r2s1, vrgout_r2s1, VE_VSHUFFLE_YUZU) ;
	      __vr vrgoutP_r2s2 = _ve_vshf_vvvs(vrgout_r2s2, vrgout_r2s2, VE_VSHUFFLE_YUZU) ;

	      FILTER_IC2(vrgoutP_r2s0) ; pKerValue++ ;
	      FILTER_IC2(vrgoutP_r2s1) ; pKerValue++ ;
	      FILTER_IC2(vrgoutP_r2s2) ; pKerValue++ ;
#undef FILTER_IC2

	    } // gInChannel

	    _ve_vstu_vss(vrsum01, 4, pGIn+gInIndex) ;
	    _ve_vstl_vss(vrsum01, 4, pGIn+gInIndex+  gInPixels) ;

	  } // gInPixels

	  k+=2 ;
	}
	if ( ((gInChannelGroup >> 2) & 0x01) == 1 ) {
	  for (int64_t gip = 0; gip < gInPixels; gip+=VLEN) {
	    const int64_t vl = gInPixels - gip < VLEN ? gInPixels - gip : VLEN ;

	    const int64_t gInIndex = gInGroupOffset + ((n * gInChannel + k) * gInHeight ) * gInWidth + gip ;

	    _ve_lvl(vl) ;

	    __vr vrseq = _ve_vseq_v() ;			// hw
	    __vr vridx = _ve_vaddsl_vsv(gip, vrseq) ;	// op + hw

	    __vr vrsum01 = _ve_pvbrd_vs_i64(0UL) ;
	    __vr vrsum23 = _ve_pvbrd_vs_i64(0UL) ;

	    __vr vrh   = _ve_vdivsl_vvs(vridx, gInWidth) ;
	    __vr vrix   = _ve_vsubsl_vvv(vridx, _ve_vmulul_vsv(gInWidth,vrh)) ;

	    __vr vry_r0 = _ve_vaddsl_vsv(1, vrh) ;
	    __vr vry_r2 = _ve_vaddsl_vsv(-1, vrh) ;

	    __vr vrx_s0 = _ve_vaddsl_vsv(1, vrix) ;
	    __vr vrx_s2 = _ve_vaddsl_vsv(-1, vrix) ;

	    __vm256 vmy_r0 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(gOutHeight,vry_r0)) ;
	    __vm256 vmy_r2 = _ve_vfmkl_mcv(VECC_GE, vry_r2) ;

	    __vm256 vmx_s0 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(gOutWidth,vrx_s0)) ;
	    __vm256 vmx_s2 = _ve_vfmkl_mcv(VECC_GE, vrx_s2) ;

	    __vm256 vmall_r0s0 = _ve_andm_mmm(vmy_r0,vmx_s0) ;
	    __vm256 vmall_r0s1 = vmy_r0 ;
	    __vm256 vmall_r0s2 = _ve_andm_mmm(vmy_r0,vmx_s2) ;

	    __vm256 vmall_r1s0 = vmx_s0 ;
	    __vm256 vmall_r1s2 = vmx_s2 ;

	    __vm256 vmall_s0_r2 = _ve_andm_mmm(vmy_r2,vmx_s0) ;
	    __vm256 vmall_s1_r2 = vmy_r2 ;
	    __vm256 vmall_s2_r2 = _ve_andm_mmm(vmy_r2,vmx_s2) ;

	    for (int64_t c=0; c<gOutChannelGroup; c++) {
	      const float *pGOutChannel = pGOut + gOutGroupOffset + ((n * gOutChannel + c) * gOutHeight * gOutWidth ) ;
	      const float *pKerValue = pKernel + kernGroupOffset + ((c * gInChannelGroup + k) * kernHeight) * kernWidth ;

	      /* memory access errors mihgt be caused */
	      __vr vrgout_r0s0 = _ve_vldu_vss(4,&pGOutChannel[gip+(padHeight-0)*gOutWidth+(padWidth-0)]) ;
	      __vr vrgout_r0s1 = _ve_vldu_vss(4,&pGOutChannel[gip+(padHeight-0)*gOutWidth+(padWidth-1)]) ;
	      __vr vrgout_r0s2 = _ve_vldu_vss(4,&pGOutChannel[gip+(padHeight-0)*gOutWidth+(padWidth-2)]) ;
	      vrgout_r0s0 = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), vrgout_r0s0, vmall_r0s0) ;
	      vrgout_r0s1 = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), vrgout_r0s1, vmall_r0s1) ;
	      vrgout_r0s2 = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), vrgout_r0s2, vmall_r0s2) ;
	      __vr vrgoutP_r0s0 = _ve_vshf_vvvs(vrgout_r0s0, vrgout_r0s0, VE_VSHUFFLE_YUZU) ;
	      __vr vrgoutP_r0s1 = _ve_vshf_vvvs(vrgout_r0s1, vrgout_r0s1, VE_VSHUFFLE_YUZU) ;
	      __vr vrgoutP_r0s2 = _ve_vshf_vvvs(vrgout_r0s2, vrgout_r0s2, VE_VSHUFFLE_YUZU) ;

#define FILTER_IC4(VROUT)								\
{											\
  const uint64_t kerValue01 = _ve_pack_f32p(pKerValue,					\
					    pKerValue+    kernHeight * kernWidth) ;	\
  const uint64_t kerValue23 = _ve_pack_f32p(pKerValue+ 2* kernHeight * kernWidth,	\
					    pKerValue+ 3* kernHeight * kernWidth) ;	\
  vrsum01 = _ve_pvfmad_vvsv(vrsum01, kerValue01, VROUT) ;				\
  vrsum23 = _ve_pvfmad_vvsv(vrsum23, kerValue23, VROUT) ;				\
}

	      FILTER_IC4(vrgoutP_r0s0) ; pKerValue++ ;
	      FILTER_IC4(vrgoutP_r0s1) ; pKerValue++ ;
	      FILTER_IC4(vrgoutP_r0s2) ; pKerValue++ ;

	      __vr vrgout_r1s0 = _ve_vldu_vss(4,&pGOutChannel[gip+(padHeight-1)*gOutWidth+(padWidth-0)]) ;
	      __vr vrgout_r1s1 = _ve_vldu_vss(4,&pGOutChannel[gip+(padHeight-1)*gOutWidth+(padWidth-1)]) ;
	      __vr vrgout_r1s2 = _ve_vldu_vss(4,&pGOutChannel[gip+(padHeight-1)*gOutWidth+(padWidth-2)]) ;
	      vrgout_r1s0 = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), vrgout_r1s0, vmall_r1s0) ;
	      vrgout_r1s2 = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), vrgout_r1s2, vmall_r1s2) ;
	      __vr vrgoutP_r1s0 = _ve_vshf_vvvs(vrgout_r1s0, vrgout_r1s0, VE_VSHUFFLE_YUZU) ;
	      __vr vrgoutP_r1s1 = _ve_vshf_vvvs(vrgout_r1s1, vrgout_r1s1, VE_VSHUFFLE_YUZU) ;
	      __vr vrgoutP_r1s2 = _ve_vshf_vvvs(vrgout_r1s2, vrgout_r1s2, VE_VSHUFFLE_YUZU) ;

	      FILTER_IC4(vrgoutP_r1s0) ; pKerValue++ ;
	      FILTER_IC4(vrgoutP_r1s1) ; pKerValue++ ;
	      FILTER_IC4(vrgoutP_r1s2) ; pKerValue++ ;


	      __vr vrgout_r2s0 = _ve_vldu_vss(4,&pGOutChannel[gip+(padHeight-2)*gOutWidth+(padWidth-0)]) ;
	      __vr vrgout_r2s1 = _ve_vldu_vss(4,&pGOutChannel[gip+(padHeight-2)*gOutWidth+(padWidth-1)]) ;
	      __vr vrgout_r2s2 = _ve_vldu_vss(4,&pGOutChannel[gip+(padHeight-2)*gOutWidth+(padWidth-2)]) ;
	      vrgout_r2s0 = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), vrgout_r2s0, vmall_s0_r2) ;
	      vrgout_r2s1 = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), vrgout_r2s1, vmall_s1_r2) ;
	      vrgout_r2s2 = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), vrgout_r2s2, vmall_s2_r2) ;
	      __vr vrgoutP_r2s0 = _ve_vshf_vvvs(vrgout_r2s0, vrgout_r2s0, VE_VSHUFFLE_YUZU) ;
	      __vr vrgoutP_r2s1 = _ve_vshf_vvvs(vrgout_r2s1, vrgout_r2s1, VE_VSHUFFLE_YUZU) ;
	      __vr vrgoutP_r2s2 = _ve_vshf_vvvs(vrgout_r2s2, vrgout_r2s2, VE_VSHUFFLE_YUZU) ;

	      FILTER_IC4(vrgoutP_r2s0) ; pKerValue++ ;
	      FILTER_IC4(vrgoutP_r2s1) ; pKerValue++ ;
	      FILTER_IC4(vrgoutP_r2s2) ; pKerValue++ ;
#undef FILTER_IC4

	    } // gInChannel

	    _ve_vstu_vss(vrsum01, 4, pGIn+gInIndex) ;
	    _ve_vstl_vss(vrsum01, 4, pGIn+gInIndex+  gInPixels) ;
	    _ve_vstu_vss(vrsum23, 4, pGIn+gInIndex+2*gInPixels) ;
	    _ve_vstl_vss(vrsum23, 4, pGIn+gInIndex+3*gInPixels) ;

	  } // gInPixels

	  k+=4 ;
	}
	if ( ((gInChannelGroup >> 3) & 0x01) == 1 ) {
	  for (int64_t gip = 0; gip < gInPixels; gip+=VLEN) {
	    const int64_t vl = gInPixels - gip < VLEN ? gInPixels - gip : VLEN ;

	    const int64_t gInIndex = gInGroupOffset + ((n * gInChannel + k) * gInHeight ) * gInWidth + gip ;

	    _ve_lvl(vl) ;

	    __vr vrseq = _ve_vseq_v() ;			// hw
	    __vr vridx = _ve_vaddsl_vsv(gip, vrseq) ;	// op + hw

	    __vr vrsum01 = _ve_pvbrd_vs_i64(0UL) ;
	    __vr vrsum23 = _ve_pvbrd_vs_i64(0UL) ;
	    __vr vrsum45 = _ve_pvbrd_vs_i64(0UL) ;
	    __vr vrsum67 = _ve_pvbrd_vs_i64(0UL) ;

	    __vr vrh   = _ve_vdivsl_vvs(vridx, gInWidth) ;
	    __vr vrix   = _ve_vsubsl_vvv(vridx, _ve_vmulul_vsv(gInWidth,vrh)) ;

	    __vr vry_r0 = _ve_vaddsl_vsv(1, vrh) ;
	    __vr vry_r2 = _ve_vaddsl_vsv(-1, vrh) ;

	    __vr vrx_s0 = _ve_vaddsl_vsv(1, vrix) ;
	    __vr vrx_s2 = _ve_vaddsl_vsv(-1, vrix) ;

	    __vm256 vmy_r0 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(gOutHeight,vry_r0)) ;
	    __vm256 vmy_r2 = _ve_vfmkl_mcv(VECC_GE, vry_r2) ;

	    __vm256 vmx_s0 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(gOutWidth,vrx_s0)) ;
	    __vm256 vmx_s2 = _ve_vfmkl_mcv(VECC_GE, vrx_s2) ;

	    __vm256 vmall_r0s0 = _ve_andm_mmm(vmy_r0,vmx_s0) ;
	    __vm256 vmall_r0s1 = vmy_r0 ;
	    __vm256 vmall_r0s2 = _ve_andm_mmm(vmy_r0,vmx_s2) ;

	    __vm256 vmall_r1s0 = vmx_s0 ;
	    __vm256 vmall_r1s2 = vmx_s2 ;

	    __vm256 vmall_s0_r2 = _ve_andm_mmm(vmy_r2,vmx_s0) ;
	    __vm256 vmall_s1_r2 = vmy_r2 ;
	    __vm256 vmall_s2_r2 = _ve_andm_mmm(vmy_r2,vmx_s2) ;

	    for (int64_t c=0; c<gOutChannelGroup; c++) {
	      const float *pGOutChannel = pGOut + gOutGroupOffset + ((n * gOutChannel + c) * gOutHeight * gOutWidth ) ;
	      const float *pKerValue = pKernel + kernGroupOffset + ((c * gInChannelGroup + k) * kernHeight) * kernWidth ;

	      /* memory access errors mihgt be caused */
	      __vr vrgout_r0s0 = _ve_vldu_vss(4,&pGOutChannel[gip+(padHeight-0)*gOutWidth+(padWidth-0)]) ;
	      __vr vrgout_r0s1 = _ve_vldu_vss(4,&pGOutChannel[gip+(padHeight-0)*gOutWidth+(padWidth-1)]) ;
	      __vr vrgout_r0s2 = _ve_vldu_vss(4,&pGOutChannel[gip+(padHeight-0)*gOutWidth+(padWidth-2)]) ;
	      vrgout_r0s0 = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), vrgout_r0s0, vmall_r0s0) ;
	      vrgout_r0s1 = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), vrgout_r0s1, vmall_r0s1) ;
	      vrgout_r0s2 = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), vrgout_r0s2, vmall_r0s2) ;
	      __vr vrgoutP_r0s0 = _ve_vshf_vvvs(vrgout_r0s0, vrgout_r0s0, VE_VSHUFFLE_YUZU) ;
	      __vr vrgoutP_r0s1 = _ve_vshf_vvvs(vrgout_r0s1, vrgout_r0s1, VE_VSHUFFLE_YUZU) ;
	      __vr vrgoutP_r0s2 = _ve_vshf_vvvs(vrgout_r0s2, vrgout_r0s2, VE_VSHUFFLE_YUZU) ;

#define FILTER_IC8(VROUT)								\
{											\
  const uint64_t kerValue01 = _ve_pack_f32p(pKerValue,					\
					    pKerValue+    kernHeight * kernWidth) ;	\
  const uint64_t kerValue23 = _ve_pack_f32p(pKerValue+ 2* kernHeight * kernWidth,	\
					    pKerValue+ 3* kernHeight * kernWidth) ;	\
  vrsum01 = _ve_pvfmad_vvsv(vrsum01, kerValue01, VROUT) ;				\
  vrsum23 = _ve_pvfmad_vvsv(vrsum23, kerValue23, VROUT) ;				\
  const uint64_t kerValue45 = _ve_pack_f32p(pKerValue+ 4* kernHeight * kernWidth,	\
					    pKerValue+ 5* kernHeight * kernWidth) ;	\
  const uint64_t kerValue67 = _ve_pack_f32p(pKerValue+ 6* kernHeight * kernWidth,	\
					    pKerValue+ 7* kernHeight * kernWidth) ;	\
  vrsum45 = _ve_pvfmad_vvsv(vrsum45, kerValue45, VROUT) ;				\
  vrsum67 = _ve_pvfmad_vvsv(vrsum67, kerValue67, VROUT) ;				\
}

	      FILTER_IC8(vrgoutP_r0s0) ; pKerValue++ ;
	      FILTER_IC8(vrgoutP_r0s1) ; pKerValue++ ;
	      FILTER_IC8(vrgoutP_r0s2) ; pKerValue++ ;

	      __vr vrgout_r1s0 = _ve_vldu_vss(4,&pGOutChannel[gip+(padHeight-1)*gOutWidth+(padWidth-0)]) ;
	      __vr vrgout_r1s1 = _ve_vldu_vss(4,&pGOutChannel[gip+(padHeight-1)*gOutWidth+(padWidth-1)]) ;
	      __vr vrgout_r1s2 = _ve_vldu_vss(4,&pGOutChannel[gip+(padHeight-1)*gOutWidth+(padWidth-2)]) ;
	      vrgout_r1s0 = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), vrgout_r1s0, vmall_r1s0) ;
	      vrgout_r1s2 = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), vrgout_r1s2, vmall_r1s2) ;
	      __vr vrgoutP_r1s0 = _ve_vshf_vvvs(vrgout_r1s0, vrgout_r1s0, VE_VSHUFFLE_YUZU) ;
	      __vr vrgoutP_r1s1 = _ve_vshf_vvvs(vrgout_r1s1, vrgout_r1s1, VE_VSHUFFLE_YUZU) ;
	      __vr vrgoutP_r1s2 = _ve_vshf_vvvs(vrgout_r1s2, vrgout_r1s2, VE_VSHUFFLE_YUZU) ;

	      FILTER_IC8(vrgoutP_r1s0) ; pKerValue++ ;
	      FILTER_IC8(vrgoutP_r1s1) ; pKerValue++ ;
	      FILTER_IC8(vrgoutP_r1s2) ; pKerValue++ ;


	      __vr vrgout_r2s0 = _ve_vldu_vss(4,&pGOutChannel[gip+(padHeight-2)*gOutWidth+(padWidth-0)]) ;
	      __vr vrgout_r2s1 = _ve_vldu_vss(4,&pGOutChannel[gip+(padHeight-2)*gOutWidth+(padWidth-1)]) ;
	      __vr vrgout_r2s2 = _ve_vldu_vss(4,&pGOutChannel[gip+(padHeight-2)*gOutWidth+(padWidth-2)]) ;
	      vrgout_r2s0 = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), vrgout_r2s0, vmall_s0_r2) ;
	      vrgout_r2s1 = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), vrgout_r2s1, vmall_s1_r2) ;
	      vrgout_r2s2 = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), vrgout_r2s2, vmall_s2_r2) ;
	      __vr vrgoutP_r2s0 = _ve_vshf_vvvs(vrgout_r2s0, vrgout_r2s0, VE_VSHUFFLE_YUZU) ;
	      __vr vrgoutP_r2s1 = _ve_vshf_vvvs(vrgout_r2s1, vrgout_r2s1, VE_VSHUFFLE_YUZU) ;
	      __vr vrgoutP_r2s2 = _ve_vshf_vvvs(vrgout_r2s2, vrgout_r2s2, VE_VSHUFFLE_YUZU) ;

	      FILTER_IC8(vrgoutP_r2s0) ; pKerValue++ ;
	      FILTER_IC8(vrgoutP_r2s1) ; pKerValue++ ;
	      FILTER_IC8(vrgoutP_r2s2) ; pKerValue++ ;
#undef FILTER_IC8

	    } // gInChannel

	    _ve_vstu_vss(vrsum01, 4, pGIn+gInIndex) ;
	    _ve_vstl_vss(vrsum01, 4, pGIn+gInIndex+  gInPixels) ;
	    _ve_vstu_vss(vrsum23, 4, pGIn+gInIndex+2*gInPixels) ;
	    _ve_vstl_vss(vrsum23, 4, pGIn+gInIndex+3*gInPixels) ;
	    _ve_vstu_vss(vrsum45, 4, pGIn+gInIndex+4*gInPixels) ;
	    _ve_vstl_vss(vrsum45, 4, pGIn+gInIndex+5*gInPixels) ;
	    _ve_vstu_vss(vrsum67, 4, pGIn+gInIndex+6*gInPixels) ;
	    _ve_vstl_vss(vrsum67, 4, pGIn+gInIndex+7*gInPixels) ;

	  } // gInPixels

	  k+=8 ;
	}
	for (; k<gInChannelGroup; k+=16) {
	  for (int64_t gip = 0; gip < gInPixels; gip+=VLEN) {
	    const int64_t vl = gInPixels - gip < VLEN ? gInPixels - gip : VLEN ;

	    const int64_t gInIndex = gInGroupOffset + ((n * gInChannel + k) * gInHeight ) * gInWidth + gip ;

	    _ve_lvl(vl) ;

	    __vr vrseq = _ve_vseq_v() ;			// hw
	    __vr vridx = _ve_vaddsl_vsv(gip, vrseq) ;	// op + hw

	    __vr vrsum01 = _ve_pvbrd_vs_i64(0UL) ;
	    __vr vrsum23 = _ve_pvbrd_vs_i64(0UL) ;
	    __vr vrsum45 = _ve_pvbrd_vs_i64(0UL) ;
	    __vr vrsum67 = _ve_pvbrd_vs_i64(0UL) ;
	    __vr vrsum89 = _ve_pvbrd_vs_i64(0UL) ;
	    __vr vrsumAB = _ve_pvbrd_vs_i64(0UL) ;
	    __vr vrsumCD = _ve_pvbrd_vs_i64(0UL) ;
	    __vr vrsumEF = _ve_pvbrd_vs_i64(0UL) ;

	    __vr vrh   = _ve_vdivsl_vvs(vridx, gInWidth) ;
	    __vr vrix   = _ve_vsubsl_vvv(vridx, _ve_vmulul_vsv(gInWidth,vrh)) ;

	    __vr vry_r0 = _ve_vaddsl_vsv(1, vrh) ;
	    __vr vry_r2 = _ve_vaddsl_vsv(-1, vrh) ;

	    __vr vrx_s0 = _ve_vaddsl_vsv(1, vrix) ;
	    __vr vrx_s2 = _ve_vaddsl_vsv(-1, vrix) ;

	    __vm256 vmy_r0 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(gOutHeight,vry_r0)) ;
	    __vm256 vmy_r2 = _ve_vfmkl_mcv(VECC_GE, vry_r2) ;

	    __vm256 vmx_s0 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(gOutWidth,vrx_s0)) ;
	    __vm256 vmx_s2 = _ve_vfmkl_mcv(VECC_GE, vrx_s2) ;

	    __vm256 vmall_r0s0 = _ve_andm_mmm(vmy_r0,vmx_s0) ;
	    __vm256 vmall_r0s1 = vmy_r0 ;
	    __vm256 vmall_r0s2 = _ve_andm_mmm(vmy_r0,vmx_s2) ;

	    __vm256 vmall_r1s0 = vmx_s0 ;
	    __vm256 vmall_r1s2 = vmx_s2 ;

	    __vm256 vmall_s0_r2 = _ve_andm_mmm(vmy_r2,vmx_s0) ;
	    __vm256 vmall_s1_r2 = vmy_r2 ;
	    __vm256 vmall_s2_r2 = _ve_andm_mmm(vmy_r2,vmx_s2) ;

	    for (int64_t c=0; c<gOutChannelGroup; c++) {
	      const float *pGOutChannel = pGOut + gOutGroupOffset + ((n * gOutChannel + c) * gOutHeight * gOutWidth ) ;
	      const float *pKerValue = pKernel + kernGroupOffset + ((c * gInChannelGroup + k) * kernHeight) * kernWidth ;

	      /* memory access errors mihgt be caused */
	      __vr vrgout_r0s0 = _ve_vldu_vss(4,&pGOutChannel[gip+(padHeight-0)*gOutWidth+(padWidth-0)]) ;
	      __vr vrgout_r0s1 = _ve_vldu_vss(4,&pGOutChannel[gip+(padHeight-0)*gOutWidth+(padWidth-1)]) ;
	      __vr vrgout_r0s2 = _ve_vldu_vss(4,&pGOutChannel[gip+(padHeight-0)*gOutWidth+(padWidth-2)]) ;
	      vrgout_r0s0 = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), vrgout_r0s0, vmall_r0s0) ;
	      vrgout_r0s1 = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), vrgout_r0s1, vmall_r0s1) ;
	      vrgout_r0s2 = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), vrgout_r0s2, vmall_r0s2) ;
	      __vr vrgoutP_r0s0 = _ve_vshf_vvvs(vrgout_r0s0, vrgout_r0s0, VE_VSHUFFLE_YUZU) ;
	      __vr vrgoutP_r0s1 = _ve_vshf_vvvs(vrgout_r0s1, vrgout_r0s1, VE_VSHUFFLE_YUZU) ;
	      __vr vrgoutP_r0s2 = _ve_vshf_vvvs(vrgout_r0s2, vrgout_r0s2, VE_VSHUFFLE_YUZU) ;

#define FILTER_IC16(VROUT)								\
{											\
  const uint64_t kerValue01 = _ve_pack_f32p(pKerValue,					\
					    pKerValue+    kernHeight * kernWidth) ;	\
  const uint64_t kerValue23 = _ve_pack_f32p(pKerValue+ 2* kernHeight * kernWidth,	\
					    pKerValue+ 3* kernHeight * kernWidth) ;	\
  vrsum01 = _ve_pvfmad_vvsv(vrsum01, kerValue01, VROUT) ;				\
  vrsum23 = _ve_pvfmad_vvsv(vrsum23, kerValue23, VROUT) ;				\
  const uint64_t kerValue45 = _ve_pack_f32p(pKerValue+ 4* kernHeight * kernWidth,	\
				pKerValue+ 5* kernHeight * kernWidth) ;			\
  const uint64_t kerValue67 = _ve_pack_f32p(pKerValue+ 6* kernHeight * kernWidth,	\
					    pKerValue+ 7* kernHeight * kernWidth) ;	\
  vrsum45 = _ve_pvfmad_vvsv(vrsum45, kerValue45, VROUT) ;				\
  vrsum67 = _ve_pvfmad_vvsv(vrsum67, kerValue67, VROUT) ;				\
  const uint64_t kerValue89 = _ve_pack_f32p(pKerValue+ 8* kernHeight * kernWidth,	\
					    pKerValue+ 9* kernHeight * kernWidth) ;	\
  const uint64_t kerValueAB = _ve_pack_f32p(pKerValue+10* kernHeight * kernWidth,	\
					    pKerValue+11* kernHeight * kernWidth) ;	\
  vrsum89 = _ve_pvfmad_vvsv(vrsum89, kerValue89, VROUT) ;				\
  vrsumAB = _ve_pvfmad_vvsv(vrsumAB, kerValueAB, VROUT) ;				\
  const uint64_t kerValueCD = _ve_pack_f32p(pKerValue+12* kernHeight * kernWidth,	\
					    pKerValue+13* kernHeight * kernWidth) ;	\
  const uint64_t kerValueEF = _ve_pack_f32p(pKerValue+14* kernHeight * kernWidth,	\
					    pKerValue+15* kernHeight * kernWidth) ;	\
  vrsumCD = _ve_pvfmad_vvsv(vrsumCD, kerValueCD, VROUT) ;				\
  vrsumEF = _ve_pvfmad_vvsv(vrsumEF, kerValueEF, VROUT) ;				\
}

	      FILTER_IC16(vrgoutP_r0s0) ; pKerValue++ ;
	      FILTER_IC16(vrgoutP_r0s1) ; pKerValue++ ;
	      FILTER_IC16(vrgoutP_r0s2) ; pKerValue++ ;

	      __vr vrgout_r1s0 = _ve_vldu_vss(4,&pGOutChannel[gip+(padHeight-1)*gOutWidth+(padWidth-0)]) ;
	      __vr vrgout_r1s1 = _ve_vldu_vss(4,&pGOutChannel[gip+(padHeight-1)*gOutWidth+(padWidth-1)]) ;
	      __vr vrgout_r1s2 = _ve_vldu_vss(4,&pGOutChannel[gip+(padHeight-1)*gOutWidth+(padWidth-2)]) ;
	      vrgout_r1s0 = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), vrgout_r1s0, vmall_r1s0) ;
	      vrgout_r1s2 = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), vrgout_r1s2, vmall_r1s2) ;
	      __vr vrgoutP_r1s0 = _ve_vshf_vvvs(vrgout_r1s0, vrgout_r1s0, VE_VSHUFFLE_YUZU) ;
	      __vr vrgoutP_r1s1 = _ve_vshf_vvvs(vrgout_r1s1, vrgout_r1s1, VE_VSHUFFLE_YUZU) ;
	      __vr vrgoutP_r1s2 = _ve_vshf_vvvs(vrgout_r1s2, vrgout_r1s2, VE_VSHUFFLE_YUZU) ;

	      FILTER_IC16(vrgoutP_r1s0) ; pKerValue++ ;
	      FILTER_IC16(vrgoutP_r1s1) ; pKerValue++ ;
	      FILTER_IC16(vrgoutP_r1s2) ; pKerValue++ ;


	      __vr vrgout_r2s0 = _ve_vldu_vss(4,&pGOutChannel[gip+(padHeight-2)*gOutWidth+(padWidth-0)]) ;
	      __vr vrgout_r2s1 = _ve_vldu_vss(4,&pGOutChannel[gip+(padHeight-2)*gOutWidth+(padWidth-1)]) ;
	      __vr vrgout_r2s2 = _ve_vldu_vss(4,&pGOutChannel[gip+(padHeight-2)*gOutWidth+(padWidth-2)]) ;
	      vrgout_r2s0 = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), vrgout_r2s0, vmall_s0_r2) ;
	      vrgout_r2s1 = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), vrgout_r2s1, vmall_s1_r2) ;
	      vrgout_r2s2 = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), vrgout_r2s2, vmall_s2_r2) ;
	      __vr vrgoutP_r2s0 = _ve_vshf_vvvs(vrgout_r2s0, vrgout_r2s0, VE_VSHUFFLE_YUZU) ;
	      __vr vrgoutP_r2s1 = _ve_vshf_vvvs(vrgout_r2s1, vrgout_r2s1, VE_VSHUFFLE_YUZU) ;
	      __vr vrgoutP_r2s2 = _ve_vshf_vvvs(vrgout_r2s2, vrgout_r2s2, VE_VSHUFFLE_YUZU) ;

	      FILTER_IC16(vrgoutP_r2s0) ; pKerValue++ ;
	      FILTER_IC16(vrgoutP_r2s1) ; pKerValue++ ;
	      FILTER_IC16(vrgoutP_r2s2) ; pKerValue++ ;
#undef FILTER_IC16

	    } // gInChannel

	    _ve_vstu_vss(vrsum01, 4, pGIn+gInIndex) ;
	    _ve_vstl_vss(vrsum01, 4, pGIn+gInIndex+  gInPixels) ;
	    _ve_vstu_vss(vrsum23, 4, pGIn+gInIndex+2*gInPixels) ;
	    _ve_vstl_vss(vrsum23, 4, pGIn+gInIndex+3*gInPixels) ;
	    _ve_vstu_vss(vrsum45, 4, pGIn+gInIndex+4*gInPixels) ;
	    _ve_vstl_vss(vrsum45, 4, pGIn+gInIndex+5*gInPixels) ;
	    _ve_vstu_vss(vrsum67, 4, pGIn+gInIndex+6*gInPixels) ;
	    _ve_vstl_vss(vrsum67, 4, pGIn+gInIndex+7*gInPixels) ;
	    _ve_vstu_vss(vrsum89, 4, pGIn+gInIndex+8*gInPixels) ;
	    _ve_vstl_vss(vrsum89, 4, pGIn+gInIndex+9*gInPixels) ;
	    _ve_vstu_vss(vrsumAB, 4, pGIn+gInIndex+10*gInPixels) ;
	    _ve_vstl_vss(vrsumAB, 4, pGIn+gInIndex+11*gInPixels) ;
	    _ve_vstu_vss(vrsumCD, 4, pGIn+gInIndex+12*gInPixels) ;
	    _ve_vstl_vss(vrsumCD, 4, pGIn+gInIndex+13*gInPixels) ;
	    _ve_vstu_vss(vrsumEF, 4, pGIn+gInIndex+14*gInPixels) ;
	    _ve_vstl_vss(vrsumEF, 4, pGIn+gInIndex+15*gInPixels) ;
	  } // gInPixels

	} // gOutChannel

      } // group
    } // batch
  }


  return VEDNN_SUCCESS;
}
