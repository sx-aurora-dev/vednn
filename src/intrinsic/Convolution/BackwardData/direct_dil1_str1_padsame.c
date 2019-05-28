#include <stdint.h>
#include <stdlib.h>

#include "vednn.h"

#include "veintrin.h"
#define VLEN	(256)

vednnError_t
vednnConvolutionBackwardData_direct_dil1_str1_padsame (
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
	    __vr vrw   = _ve_vsubsl_vvv(vridx, _ve_vmulul_vsv(gInWidth,vrh)) ;

	    for (int64_t r=0; r<kernHeight; r++) {
	      __vr vry = _ve_vaddsl_vsv(padHeight-r, vrh) ;

	      __vm256 vmy1 = _ve_vfmkl_mcv(VECC_GE, vry) ;
	      __vm256 vmy2 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(gOutHeight,vry)) ;

	      __vm256 vmy = _ve_andm_mmm(vmy1, vmy2) ;

	      for (int64_t s=0; s<kernWidth; s++) {
		__vr vrx = _ve_vaddsl_vsv(padWidth-s, vrw) ;

		__vm256 vmx1 = _ve_vfmkl_mcv(VECC_GE, vrx) ;
		__vm256 vmx2 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(gOutWidth,vrx)) ;

		__vm256 vmx = _ve_andm_mmm(vmx1, vmx2) ;

		__vm256 vmall = _ve_andm_mmm(vmy,vmx) ;

		for (int64_t c=0; c<gOutChannelGroup; c++) {

		  const float *pGOutChannel = pGOut + gOutGroupOffset + ((n * gOutChannel + c) * gOutHeight * gOutWidth ) ;

		  /* memory access errors mihgt be caused */
		  __vr vrgout = _ve_vldu_vss(4,&pGOutChannel[gip+(padHeight-r)*gOutWidth+(padWidth-s)]) ;
		  vrgout = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), vrgout, vmall) ;

		  const float *pKerValue = pKernel + kernGroupOffset + ((c * gInChannelGroup + k) * kernHeight + r) * kernWidth + s;

		  vrsum = _ve_vfmads_vvsv(vrsum, *pKerValue, vrgout) ;

		} // gOutChannel

	      } // kernWidth
	    } // kernHeight

	    _ve_vstu_vss(vrsum, 4, pGIn+gInIndex) ;

	  } // gGInPixels

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
	    __vr vrw   = _ve_vsubsl_vvv(vridx, _ve_vmulul_vsv(gInWidth,vrh)) ;

	    for (int64_t r=0; r<kernHeight; r++) {
	      __vr vry = _ve_vaddsl_vsv(padHeight-r, vrh) ;

	      __vm256 vmy1 = _ve_vfmkl_mcv(VECC_GE, vry) ;
	      __vm256 vmy2 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(gOutHeight,vry)) ;
	      __vm256 vmy  = _ve_andm_mmm(vmy1, vmy2) ;

	      for (int64_t s=0; s<kernWidth; s++) {
		__vr vrx = _ve_vaddsl_vsv(padWidth-s, vrw) ;

		__vm256 vmx1 = _ve_vfmkl_mcv(VECC_GE, vrx) ;
		__vm256 vmx2 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(gOutWidth,vrx)) ;
		__vm256 vmx  = _ve_andm_mmm(vmx1, vmx2) ;

		__vm256 vmall = _ve_andm_mmm(vmy,vmx) ;

		for (int64_t c=0; c<gOutChannelGroup; c++) {

		  const float *pGOutChannel = pGOut + gOutGroupOffset + ((n * gOutChannel + c) * gOutHeight * gOutWidth ) ;

		  /* memory access errors mihgt be caused */
		  __vr vrgout = _ve_vldu_vss(4,&pGOutChannel[gip+(padHeight-r)*gOutWidth+(padWidth-s)]) ;
		  vrgout = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), vrgout, vmall) ;

		  __vr vrgtouP = _ve_vshf_vvvs(vrgout, vrgout, VE_VSHUFFLE_YUZU) ;

		  const float *pKerValue = pKernel + kernGroupOffset + ((c * gInChannelGroup + k) * kernHeight + r) * kernWidth + s;

		  const uint64_t kerValue01 = _ve_pack_f32p(pKerValue,
							    pKerValue+   kernHeight * kernWidth) ;

		  vrsum01 = _ve_pvfmad_vvsv(vrsum01, kerValue01, vrgtouP) ;

		} // gOutChannel

	      } // kernWidth
	    } // kernHeight

	    _ve_vstu_vss(vrsum01, 4, pGIn+gInIndex) ;
	    _ve_vstl_vss(vrsum01, 4, pGIn+gInIndex+gInPixels) ;

	  } // gInPixels

	  k+=2 ;
	}
	if ( ((gInChannelGroup >> 2) & 0x01) == 1 ) {
	  for (int64_t gip = 0; gip < gInPixels; gip+=VLEN) {
	    const int64_t vl = gInPixels - gip < VLEN ? gInPixels - gip : VLEN ;

	    const int64_t ginIndex = gInGroupOffset + ((n * gInChannel + k) * gInHeight ) * gInWidth + gip ;

	    _ve_lvl(vl) ;

	    __vr vrseq = _ve_vseq_v() ;			// hw
	    __vr vridx = _ve_vaddsl_vsv(gip, vrseq) ;	// op + hw

	    __vr vrsum01 = _ve_pvbrd_vs_i64(0UL) ;
	    __vr vrsum23 = _ve_pvbrd_vs_i64(0UL) ;

	    __vr vrh   = _ve_vdivsl_vvs(vridx, gInWidth) ;
	    __vr vrw   = _ve_vsubsl_vvv(vridx, _ve_vmulul_vsv(gInWidth,vrh)) ;

	    for (int64_t r=0; r<kernHeight; r++) {
	      __vr vry = _ve_vaddsl_vsv(padHeight-r, vrh) ;

	      __vm256 vmy1 = _ve_vfmkl_mcv(VECC_GE, vry) ;
	      __vm256 vmy2 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(gOutHeight,vry)) ;
	      __vm256 vmy  = _ve_andm_mmm(vmy1, vmy2) ;

	      for (int64_t s=0; s<kernWidth; s++) {
		__vr vrx = _ve_vaddsl_vsv(padWidth-s, vrw) ;

		__vm256 vmx1 = _ve_vfmkl_mcv(VECC_GE, vrx) ;
		__vm256 vmx2 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(gOutWidth,vrx)) ;
		__vm256 vmx  = _ve_andm_mmm(vmx1, vmx2) ;

		__vm256 vmall = _ve_andm_mmm(vmy,vmx) ;

		for (int64_t c=0; c<gOutChannelGroup; c++) {

		  const float *pGOutChannel = pGOut + gOutGroupOffset + ((n * gOutChannel + c) * gOutHeight * gOutWidth ) ;

		  /* memory access errors mihgt be caused */
		  __vr vrgout = _ve_vldu_vss(4,&pGOutChannel[gip+(padHeight-r)*gOutWidth+(padWidth-s)]) ;
		  vrgout = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), vrgout, vmall) ;

		  __vr vrgoutP = _ve_vshf_vvvs(vrgout, vrgout, VE_VSHUFFLE_YUZU) ;

		  const float *pKerValue = pKernel + kernGroupOffset + ((c * gInChannelGroup + k) * kernHeight + r) * kernWidth + s;

		  const uint64_t kerValue01 = _ve_pack_f32p(pKerValue,
							    pKerValue+    kernHeight * kernWidth) ;
		  const uint64_t kerValue23 = _ve_pack_f32p(pKerValue+ 2* kernHeight * kernWidth,
							    pKerValue+ 3* kernHeight * kernWidth) ;

		  vrsum01 = _ve_pvfmad_vvsv(vrsum01, kerValue01, vrgoutP) ;
		  vrsum23 = _ve_pvfmad_vvsv(vrsum23, kerValue23, vrgoutP) ;

		} // gOutChannel

	      } // kernWidth
	    } // kernHeight

	    _ve_vstu_vss(vrsum01, 4, pGIn+ginIndex) ;
	    _ve_vstl_vss(vrsum01, 4, pGIn+ginIndex+  gInPixels) ;
	    _ve_vstu_vss(vrsum23, 4, pGIn+ginIndex+2*gInPixels) ;
	    _ve_vstl_vss(vrsum23, 4, pGIn+ginIndex+3*gInPixels) ;

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
	    __vr vrix   = _ve_vsubsl_vvv(vridx, _ve_vmulul_vsv(gInWidth,vrh)) ;

	    for (int64_t r=0; r<kernHeight; r++) {
	      __vr vry = _ve_vaddsl_vsv(padHeight-r, vrh) ;

	      __vm256 vmy1 = _ve_vfmkl_mcv(VECC_GE, vry) ;
	      __vm256 vmy2 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(gOutHeight,vry)) ;
	      __vm256 vmy  = _ve_andm_mmm(vmy1, vmy2) ;

	      for (int64_t s=0; s<kernWidth; s++) {
		__vr vrx = _ve_vaddsl_vsv(padWidth-s, vrix) ;

		__vm256 vmx1 = _ve_vfmkl_mcv(VECC_GE, vrx) ;
		__vm256 vmx2 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(gOutWidth,vrx)) ;
		__vm256 vmx  = _ve_andm_mmm(vmx1, vmx2) ;

		__vm256 vmall = _ve_andm_mmm(vmy,vmx) ;

		for (int64_t c=0; c<gOutChannelGroup; c++) {

		  const float *pGOutChannel = pGOut + gOutGroupOffset + ((n * gOutChannel + c) * gOutHeight * gOutWidth ) ;

		  /* memory access errors mihgt be caused */
		  __vr vrgout = _ve_vldu_vss(4,&pGOutChannel[gip+(padHeight-r)*gOutWidth+(padWidth-s)]) ;
		  vrgout = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), vrgout, vmall) ;

		  __vr vrgoutP = _ve_vshf_vvvs(vrgout, vrgout, VE_VSHUFFLE_YUZU) ;

		  const float *pKerValue = pKernel + kernGroupOffset + ((c * gInChannelGroup + k) * kernHeight + r) * kernWidth + s;

		  const uint64_t kerValue01 = _ve_pack_f32p(pKerValue,
							    pKerValue+    kernHeight * kernWidth) ;
		  const uint64_t kerValue23 = _ve_pack_f32p(pKerValue+ 2* kernHeight * kernWidth,
							    pKerValue+ 3* kernHeight * kernWidth) ;
		  const uint64_t kerValue45 = _ve_pack_f32p(pKerValue+ 4* kernHeight * kernWidth,
							    pKerValue+ 5* kernHeight * kernWidth) ;
		  const uint64_t kerValue67 = _ve_pack_f32p(pKerValue+ 6* kernHeight * kernWidth,
							    pKerValue+ 7* kernHeight * kernWidth) ;

		  vrsum01 = _ve_pvfmad_vvsv(vrsum01, kerValue01, vrgoutP) ;
		  vrsum23 = _ve_pvfmad_vvsv(vrsum23, kerValue23, vrgoutP) ;
		  vrsum45 = _ve_pvfmad_vvsv(vrsum45, kerValue45, vrgoutP) ;
		  vrsum67 = _ve_pvfmad_vvsv(vrsum67, kerValue67, vrgoutP) ;

		} // gInChannel

	      } // kernWidth
	    } // kernHeight

	    _ve_vstu_vss(vrsum01, 4, pGIn+gInIndex) ;
	    _ve_vstl_vss(vrsum01, 4, pGIn+gInIndex+  gInPixels) ;
	    _ve_vstu_vss(vrsum23, 4, pGIn+gInIndex+2*gInPixels) ;
	    _ve_vstl_vss(vrsum23, 4, pGIn+gInIndex+3*gInPixels) ;
	    _ve_vstu_vss(vrsum45, 4, pGIn+gInIndex+4*gInPixels) ;
	    _ve_vstl_vss(vrsum45, 4, pGIn+gInIndex+5*gInPixels) ;
	    _ve_vstu_vss(vrsum67, 4, pGIn+gInIndex+6*gInPixels) ;
	    _ve_vstl_vss(vrsum67, 4, pGIn+gInIndex+7*gInPixels) ;

	  } // gOutPixels

	} // gOutChannel
      } // group
    } // batch
  }


  return VEDNN_SUCCESS;
}
