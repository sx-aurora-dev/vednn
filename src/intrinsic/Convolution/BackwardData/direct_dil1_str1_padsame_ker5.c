#include <stdint.h>
#include <stdlib.h>

#include "vednn.h"

#include "veintrin.h"
#define VLEN	(256)

vednnError_t
vednnConvolutionBackwardData_direct_dil1_str1_padsame_ker5 (
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

	    __vr vrsum = _ve_vbrdu_vs_f32(0.0f) ;

	    __vr vrh   = _ve_vdivsl_vvs(vridx, gInWidth) ;
	    __vr vrix  = _ve_vsubsl_vvv(vridx, _ve_vmulul_vsv(gInWidth,vrh)) ;

	    __vr vrx_s0 = _ve_vaddsl_vsv(2, vrix) ;
	    __vr vrx_s1 = _ve_vaddsl_vsv(1, vrix) ;

	    __vm256 vmx1_s3 = _ve_vfmkl_mcv(VECC_GE, _ve_vaddsl_vsv(-1, vrix)) ;
	    __vm256 vmx1_s4 = _ve_vfmkl_mcv(VECC_GE, _ve_vaddsl_vsv(-2, vrix)) ;

	    __vm256 vmx2_s0 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(gOutWidth-2,vrix)) ;
	    __vm256 vmx2_s1 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(gOutWidth-1,vrix)) ;

	    __vm256 vmx_s0  = vmx2_s0 ;
	    __vm256 vmx_s1  = vmx2_s1 ;
	    __vm256 vmx_s3  = vmx1_s3 ;
	    __vm256 vmx_s4  = vmx1_s4 ;

	    for (int64_t r=0; r<kernHeight; r++) {
	      __vr vry = _ve_vaddsl_vsv(padHeight-r, vrh) ;

	      __vm256 vmy1 = _ve_vfmkl_mcv(VECC_GE, vry) ;
	      __vm256 vmy2 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(gOutHeight,vry)) ;
	      __vm256 vmy  = _ve_andm_mmm(vmy1, vmy2) ;

	      __vm256 vmall_s0 = _ve_andm_mmm(vmy,vmx_s0) ;
	      __vm256 vmall_s1 = _ve_andm_mmm(vmy,vmx_s1) ;
	      __vm256 vmall_s2 = vmy ;
	      __vm256 vmall_s3 = _ve_andm_mmm(vmy,vmx_s3) ;
	      __vm256 vmall_s4 = _ve_andm_mmm(vmy,vmx_s4) ;

	      for (int64_t c=0; c<gOutChannelGroup; c++) {

		const float *pGOutChannel = pGOut + gOutGroupOffset + ((n * gOutChannel + c) * gOutHeight * gOutWidth ) ;

		/* memory access errors mihgt be caused */
		__vr vrgout_s0 = _ve_vldu_vss(4,&pGOutChannel[gip+(padHeight-r)*gOutWidth+(padWidth-0)]) ;
		__vr vrgout_s1 = _ve_vldu_vss(4,&pGOutChannel[gip+(padHeight-r)*gOutWidth+(padWidth-1)]) ;
		__vr vrgout_s2 = _ve_vldu_vss(4,&pGOutChannel[gip+(padHeight-r)*gOutWidth+(padWidth-2)]) ;
		__vr vrgout_s3 = _ve_vldu_vss(4,&pGOutChannel[gip+(padHeight-r)*gOutWidth+(padWidth-3)]) ;
		__vr vrgout_s4 = _ve_vldu_vss(4,&pGOutChannel[gip+(padHeight-r)*gOutWidth+(padWidth-4)]) ;

		const float *pKerValue = pKernel + kernGroupOffset + ((c * gInChannelGroup + k) * kernHeight + r) * kernWidth ;
#define VFMAD1(PKERVALUE, VRGOUT, VMR) {			\
  VRGOUT = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), VRGOUT, VMR) ;	\
  vrsum = _ve_vfmads_vvsv(vrsum, *(PKERVALUE), VRGOUT) ;	\
}
		VFMAD1(pKerValue,vrgout_s0,vmall_s0) ; pKerValue++ ;
		VFMAD1(pKerValue,vrgout_s1,vmall_s1) ; pKerValue++ ;
		VFMAD1(pKerValue,vrgout_s2,vmall_s2) ; pKerValue++ ;
		VFMAD1(pKerValue,vrgout_s3,vmall_s3) ; pKerValue++ ;
		VFMAD1(pKerValue,vrgout_s4,vmall_s4) ; pKerValue++ ;
#undef PVFMAD1
	      } // gInChannel
	    } // kernHeight

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
	    __vr vrix  = _ve_vsubsl_vvv(vridx, _ve_vmulul_vsv(gInWidth,vrh)) ;

	    __vr vrx_s0 = _ve_vaddsl_vsv(2, vrix) ;
	    __vr vrx_s1 = _ve_vaddsl_vsv(1, vrix) ;

	    __vm256 vmx1_s3 = _ve_vfmkl_mcv(VECC_GE, _ve_vaddsl_vsv(-1, vrix)) ;
	    __vm256 vmx1_s4 = _ve_vfmkl_mcv(VECC_GE, _ve_vaddsl_vsv(-2, vrix)) ;

	    __vm256 vmx2_s0 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(gOutWidth-2,vrix)) ;
	    __vm256 vmx2_s1 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(gOutWidth-1,vrix)) ;

	    __vm256 vmx_s0  = vmx2_s0 ;
	    __vm256 vmx_s1  = vmx2_s1 ;
	    __vm256 vmx_s3  = vmx1_s3 ;
	    __vm256 vmx_s4  = vmx1_s4 ;

	    for (int64_t r=0; r<kernHeight; r++) {
	      __vr vry = _ve_vaddsl_vsv(padHeight-r, vrh) ;

	      __vm256 vmy1 = _ve_vfmkl_mcv(VECC_GE, vry) ;
	      __vm256 vmy2 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(gOutHeight,vry)) ;
	      __vm256 vmy  = _ve_andm_mmm(vmy1, vmy2) ;

	      __vm256 vmall_s0 = _ve_andm_mmm(vmy,vmx_s0) ;
	      __vm256 vmall_s1 = _ve_andm_mmm(vmy,vmx_s1) ;
	      __vm256 vmall_s2 = vmy ;
	      __vm256 vmall_s3 = _ve_andm_mmm(vmy,vmx_s3) ;
	      __vm256 vmall_s4 = _ve_andm_mmm(vmy,vmx_s4) ;

	      for (int64_t c=0; c<gOutChannelGroup; c++) {

		const float *pGOutChannel = pGOut + gOutGroupOffset + ((n * gOutChannel + c) * gOutHeight * gOutWidth ) ;

		/* memory access errors mihgt be caused */
		__vr vrgout_s0 = _ve_vldu_vss(4,&pGOutChannel[gip+(padHeight-r)*gOutWidth+(padWidth-0)]) ;
		__vr vrgout_s1 = _ve_vldu_vss(4,&pGOutChannel[gip+(padHeight-r)*gOutWidth+(padWidth-1)]) ;
		__vr vrgout_s2 = _ve_vldu_vss(4,&pGOutChannel[gip+(padHeight-r)*gOutWidth+(padWidth-2)]) ;
		__vr vrgout_s3 = _ve_vldu_vss(4,&pGOutChannel[gip+(padHeight-r)*gOutWidth+(padWidth-3)]) ;
		__vr vrgout_s4 = _ve_vldu_vss(4,&pGOutChannel[gip+(padHeight-r)*gOutWidth+(padWidth-4)]) ;

		const float *pKerValue = pKernel + kernGroupOffset + ((c * gInChannelGroup + k) * kernHeight + r) * kernWidth ;
#define PVFMAD2(PKERVALUE, VRGOUT, VMR) {						\
  VRGOUT = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), VRGOUT, VMR) ;				\
  __vr vrgoutP = _ve_vshf_vvvs(VRGOUT, VRGOUT, VE_VSHUFFLE_YUZU) ;			\
  const uint64_t kerValue01 = _ve_pack_f32p(PKERVALUE,					\
		                            PKERVALUE+    kernHeight * kernWidth) ;	\
  vrsum01 = _ve_pvfmad_vvsv(vrsum01, kerValue01, vrgoutP) ;	\
}
		PVFMAD2(pKerValue,vrgout_s0,vmall_s0) ; pKerValue++ ;
		PVFMAD2(pKerValue,vrgout_s1,vmall_s1) ; pKerValue++ ;
		PVFMAD2(pKerValue,vrgout_s2,vmall_s2) ; pKerValue++ ;
		PVFMAD2(pKerValue,vrgout_s3,vmall_s3) ; pKerValue++ ;
		PVFMAD2(pKerValue,vrgout_s4,vmall_s4) ; pKerValue++ ;
#undef PVFMAD2
	      } // gInChannel
	    } // kernHeight

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
	    __vr vrix  = _ve_vsubsl_vvv(vridx, _ve_vmulul_vsv(gInWidth,vrh)) ;

	    __vr vrx_s0 = _ve_vaddsl_vsv(2, vrix) ;
	    __vr vrx_s1 = _ve_vaddsl_vsv(1, vrix) ;

	    __vm256 vmx1_s3 = _ve_vfmkl_mcv(VECC_GE, _ve_vaddsl_vsv(-1, vrix)) ;
	    __vm256 vmx1_s4 = _ve_vfmkl_mcv(VECC_GE, _ve_vaddsl_vsv(-2, vrix)) ;

	    __vm256 vmx2_s0 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(gOutWidth-2,vrix)) ;
	    __vm256 vmx2_s1 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(gOutWidth-1,vrix)) ;

	    __vm256 vmx_s0  = vmx2_s0 ;
	    __vm256 vmx_s1  = vmx2_s1 ;
	    __vm256 vmx_s3  = vmx1_s3 ;
	    __vm256 vmx_s4  = vmx1_s4 ;

	    for (int64_t r=0; r<kernHeight; r++) {
	      __vr vry = _ve_vaddsl_vsv(padHeight-r, vrh) ;

	      __vm256 vmy1 = _ve_vfmkl_mcv(VECC_GE, vry) ;
	      __vm256 vmy2 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(gOutHeight,vry)) ;
	      __vm256 vmy  = _ve_andm_mmm(vmy1, vmy2) ;

	      __vm256 vmall_s0 = _ve_andm_mmm(vmy,vmx_s0) ;
	      __vm256 vmall_s1 = _ve_andm_mmm(vmy,vmx_s1) ;
	      __vm256 vmall_s2 = vmy ;
	      __vm256 vmall_s3 = _ve_andm_mmm(vmy,vmx_s3) ;
	      __vm256 vmall_s4 = _ve_andm_mmm(vmy,vmx_s4) ;

	      for (int64_t c=0; c<gOutChannelGroup; c++) {

		const float *pGOutChannel = pGOut + gOutGroupOffset + ((n * gOutChannel + c) * gOutHeight * gOutWidth ) ;

		/* memory access errors mihgt be caused */
		__vr vrgout_s0 = _ve_vldu_vss(4,&pGOutChannel[gip+(padHeight-r)*gOutWidth+(padWidth-0)]) ;
		__vr vrgout_s1 = _ve_vldu_vss(4,&pGOutChannel[gip+(padHeight-r)*gOutWidth+(padWidth-1)]) ;
		__vr vrgout_s2 = _ve_vldu_vss(4,&pGOutChannel[gip+(padHeight-r)*gOutWidth+(padWidth-2)]) ;
		__vr vrgout_s3 = _ve_vldu_vss(4,&pGOutChannel[gip+(padHeight-r)*gOutWidth+(padWidth-3)]) ;
		__vr vrgout_s4 = _ve_vldu_vss(4,&pGOutChannel[gip+(padHeight-r)*gOutWidth+(padWidth-4)]) ;

		const float *pKerValue = pKernel + kernGroupOffset + ((c * gInChannelGroup + k) * kernHeight + r) * kernWidth ;
#define PVFMAD4(PKERVALUE, VRGOUT, VMR) {						\
  VRGOUT = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), VRGOUT, VMR) ;				\
  __vr vrgoutP = _ve_vshf_vvvs(VRGOUT, VRGOUT, VE_VSHUFFLE_YUZU) ;			\
  const uint64_t kerValue01 = _ve_pack_f32p(PKERVALUE,					\
		                            PKERVALUE+    kernHeight * kernWidth) ;	\
  const uint64_t kerValue23 = _ve_pack_f32p(PKERVALUE+ 2* kernHeight * kernWidth,	\
                                            PKERVALUE+ 3* kernHeight * kernWidth) ;	\
  vrsum01 = _ve_pvfmad_vvsv(vrsum01, kerValue01, vrgoutP) ;	\
  vrsum23 = _ve_pvfmad_vvsv(vrsum23, kerValue23, vrgoutP) ;	\
}
		PVFMAD4(pKerValue,vrgout_s0,vmall_s0) ; pKerValue++ ;
		PVFMAD4(pKerValue,vrgout_s1,vmall_s1) ; pKerValue++ ;
		PVFMAD4(pKerValue,vrgout_s2,vmall_s2) ; pKerValue++ ;
		PVFMAD4(pKerValue,vrgout_s3,vmall_s3) ; pKerValue++ ;
		PVFMAD4(pKerValue,vrgout_s4,vmall_s4) ; pKerValue++ ;
#undef PVFMAD4
	      } // gInChannel
	    } // kernHeight

	    _ve_vstu_vss(vrsum01, 4, pGIn+gInIndex) ;
	    _ve_vstl_vss(vrsum01, 4, pGIn+gInIndex+  gInPixels) ;
	    _ve_vstu_vss(vrsum23, 4, pGIn+gInIndex+2*gInPixels) ;
	    _ve_vstl_vss(vrsum23, 4, pGIn+gInIndex+3*gInPixels) ;

	  } // gInPixels

	  k+=4 ;
	}
	for (; k<gInChannelGroup; k+=8) {
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
	    __vr vrix  = _ve_vsubsl_vvv(vridx, _ve_vmulul_vsv(gInWidth,vrh)) ;

	    __vr vrx_s0 = _ve_vaddsl_vsv(2, vrix) ;
	    __vr vrx_s1 = _ve_vaddsl_vsv(1, vrix) ;

	    __vm256 vmx1_s3 = _ve_vfmkl_mcv(VECC_GE, _ve_vaddsl_vsv(-1, vrix)) ;
	    __vm256 vmx1_s4 = _ve_vfmkl_mcv(VECC_GE, _ve_vaddsl_vsv(-2, vrix)) ;

	    __vm256 vmx2_s0 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(gOutWidth-2,vrix)) ;
	    __vm256 vmx2_s1 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(gOutWidth-1,vrix)) ;

	    __vm256 vmx_s0  = vmx2_s0 ;
	    __vm256 vmx_s1  = vmx2_s1 ;
	    __vm256 vmx_s3  = vmx1_s3 ;
	    __vm256 vmx_s4  = vmx1_s4 ;

	    for (int64_t r=0; r<kernHeight; r++) {
	      __vr vry = _ve_vaddsl_vsv(padHeight-r, vrh) ;

	      __vm256 vmy1 = _ve_vfmkl_mcv(VECC_GE, vry) ;
	      __vm256 vmy2 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(gOutHeight,vry)) ;
	      __vm256 vmy  = _ve_andm_mmm(vmy1, vmy2) ;

	      __vm256 vmall_s0 = _ve_andm_mmm(vmy,vmx_s0) ;
	      __vm256 vmall_s1 = _ve_andm_mmm(vmy,vmx_s1) ;
	      __vm256 vmall_s2 = vmy ;
	      __vm256 vmall_s3 = _ve_andm_mmm(vmy,vmx_s3) ;
	      __vm256 vmall_s4 = _ve_andm_mmm(vmy,vmx_s4) ;

	      for (int64_t c=0; c<gOutChannelGroup; c++) {

		const float *pGOutChannel = pGOut + gOutGroupOffset + ((n * gOutChannel + c) * gOutHeight * gOutWidth ) ;

		/* memory access errors mihgt be caused */
		__vr vrgout_s0 = _ve_vldu_vss(4,&pGOutChannel[gip+(padHeight-r)*gOutWidth+(padWidth-0)]) ;
		__vr vrgout_s1 = _ve_vldu_vss(4,&pGOutChannel[gip+(padHeight-r)*gOutWidth+(padWidth-1)]) ;
		__vr vrgout_s2 = _ve_vldu_vss(4,&pGOutChannel[gip+(padHeight-r)*gOutWidth+(padWidth-2)]) ;
		__vr vrgout_s3 = _ve_vldu_vss(4,&pGOutChannel[gip+(padHeight-r)*gOutWidth+(padWidth-3)]) ;
		__vr vrgout_s4 = _ve_vldu_vss(4,&pGOutChannel[gip+(padHeight-r)*gOutWidth+(padWidth-4)]) ;

		const float *pKerValue = pKernel + kernGroupOffset + ((c * gInChannelGroup + k) * kernHeight + r) * kernWidth ;
#define PVFMAD8(PKERVALUE, VRGOUT, VMR) {						\
  VRGOUT = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), VRGOUT, VMR) ;				\
  __vr vrgoutP = _ve_vshf_vvvs(VRGOUT, VRGOUT, VE_VSHUFFLE_YUZU) ;			\
  const uint64_t kerValue01 = _ve_pack_f32p(PKERVALUE,					\
		                            PKERVALUE+    kernHeight * kernWidth) ;	\
  const uint64_t kerValue23 = _ve_pack_f32p(PKERVALUE+ 2* kernHeight * kernWidth,	\
                                            PKERVALUE+ 3* kernHeight * kernWidth) ;	\
  const uint64_t kerValue45 = _ve_pack_f32p(PKERVALUE+ 4* kernHeight * kernWidth,	\
					    PKERVALUE+ 5* kernHeight * kernWidth) ;	\
  const uint64_t kerValue67 = _ve_pack_f32p(PKERVALUE+ 6* kernHeight * kernWidth,	\
					    PKERVALUE+ 7* kernHeight * kernWidth) ;	\
  vrsum01 = _ve_pvfmad_vvsv(vrsum01, kerValue01, vrgoutP) ;	\
  vrsum23 = _ve_pvfmad_vvsv(vrsum23, kerValue23, vrgoutP) ;	\
  vrsum45 = _ve_pvfmad_vvsv(vrsum45, kerValue45, vrgoutP) ;	\
  vrsum67 = _ve_pvfmad_vvsv(vrsum67, kerValue67, vrgoutP) ;	\
}
		PVFMAD8(pKerValue,vrgout_s0,vmall_s0) ; pKerValue++ ;
		PVFMAD8(pKerValue,vrgout_s1,vmall_s1) ; pKerValue++ ;
		PVFMAD8(pKerValue,vrgout_s2,vmall_s2) ; pKerValue++ ;
		PVFMAD8(pKerValue,vrgout_s3,vmall_s3) ; pKerValue++ ;
		PVFMAD8(pKerValue,vrgout_s4,vmall_s4) ; pKerValue++ ;
#undef PVFMAD8
	      } // gInChannel
	    } // kernHeight

	    _ve_vstu_vss(vrsum01, 4, pGIn+gInIndex) ;
	    _ve_vstl_vss(vrsum01, 4, pGIn+gInIndex+  gInPixels) ;
	    _ve_vstu_vss(vrsum23, 4, pGIn+gInIndex+2*gInPixels) ;
	    _ve_vstl_vss(vrsum23, 4, pGIn+gInIndex+3*gInPixels) ;
	    _ve_vstu_vss(vrsum45, 4, pGIn+gInIndex+4*gInPixels) ;
	    _ve_vstl_vss(vrsum45, 4, pGIn+gInIndex+5*gInPixels) ;
	    _ve_vstu_vss(vrsum67, 4, pGIn+gInIndex+6*gInPixels) ;
	    _ve_vstl_vss(vrsum67, 4, pGIn+gInIndex+7*gInPixels) ;

	  } // gInPixels

	} // gOutChannel
      } // group
    } // batch
  }

  return VEDNN_SUCCESS;
}
