#include <stdint.h>
#include <stdlib.h>

#include "vednn.h"

#include "veintrin.h"
#define VLEN	(256)

vednnError_t
vednnConvolutionForward_direct_dil1_str1_padsame_ker2(
    const vednnTensorParam_t * restrict 	pParamIn,
    const void * restrict 			pDataIn,
    const vednnFilterParam_t * restrict 	pParamKernel,
    const void * restrict 			pDataKernel,
    const vednnConvolutionParam_t * restrict 	pParamConv,
    const vednnTensorParam_t * restrict 	pParamOut,
    void * restrict 				pDataOut
)
{
  const int64_t batch      = pParamIn->batch;
  const int64_t inChannel  = pParamIn->channel;
  const int64_t inWidth    = pParamIn->width;
  const int64_t inHeight   = pParamIn->height;
  const int64_t outChannel = pParamOut->channel;
  const int64_t outWidth   = pParamOut->width;		/* must be equal to inWidth */
  const int64_t outHeight  = pParamOut->height;		/* must be equal to inHeight */
  const int64_t kernWidth  = pParamKernel->width;	/* must be 2 */
  const int64_t kernHeight = pParamKernel->height;	/* must be 2 */

  const int64_t group          = pParamConv->group;
//  const int64_t strideWidth    = pParamConv->strideWidth;	/* must be 1 */
//  const int64_t strideHeight   = pParamConv->strideHeight;	/* must be 1 */
  const int64_t padWidth       = pParamConv->padWidth;		/* must be 0 */
  const int64_t padHeight      = pParamConv->padHeight;		/* must be 0 */
//  const int64_t dilationWidth  = pParamConv->dilationWidth;	/* must be 1 */
//  const int64_t dilationHeight = pParamConv->dilationHeight;	/* must be 1 */

  const int64_t inChannelGroup  = inChannel  / group;   // equal to pDataKernel->inChannel
  const int64_t outChannelGroup = outChannel / group;   // equal to pDataKernel->outChannel

  const float * restrict pIn     = pDataIn;
  const float * restrict pKernel = pDataKernel;
  float * restrict const pOut    = pDataOut;

  const int oPixels= outHeight*outWidth ;


  {
    for (int64_t n = 0; n < batch; n++) {
      for (int64_t g = 0; g < group; g++) {
	const int64_t inGroupOffset   = g * inChannelGroup * inHeight * inWidth;
	const int64_t outGroupOffset  = g * outChannelGroup * outHeight * outWidth;
	const int64_t kernGroupOffset = g * outChannelGroup * inChannelGroup * kernHeight * kernWidth;

	int k = 0 ;
	if ( (outChannelGroup & 0x01) == 1 ) {
	  int64_t outIndex = outGroupOffset + (n * outChannel + k) * oPixels ;

	  for (int64_t op = 0; op < oPixels; op+=VLEN) {
	    const int64_t vl = oPixels - op < VLEN ? oPixels - op : VLEN ;

	    _ve_lvl(vl) ;

	    __vr vrseq = _ve_vseq_v() ;			// xy
	    __vr vridx = _ve_vaddsl_vsv(op, vrseq) ;	// op + xy

	    __vr vrsum = _ve_vbrdu_vs_f32(0.0f) ;
	    __vr vry   = _ve_vdivsl_vvs(vridx, outWidth) ;
	    __vr vrx   = _ve_vsubsl_vvv(vridx, _ve_vmulul_vsv(outWidth,vry)) ;

	    __vm256 vm_r1 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(inHeight-1,vry)) ;	// condition(y+1 < inHeight)
	    __vm256 vm_s1 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(inWidth-1,vrx)) ;	// condition(x+1 < inWidth)

	    __vm256 vm_r1s0 = vm_r1 ;
	    __vm256 vm_r0s1 = vm_s1 ;
	    __vm256 vm_r1s1 = _ve_andm_mmm(vm_r1,vm_s1) ;

	    for (int64_t c = 0; c < inChannelGroup; c++) {

	      const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;
	      const float *pKerValue  = pKernel + kernGroupOffset + ((k * inChannelGroup + c) * kernHeight ) * kernWidth ;

	      /* memory access errors mihgt be caused */
	      __vr vrin_r0s0 = _ve_vldu_vss(4, pInChannel+op) ;
	      __vr vrin_r0s1 = _ve_vldu_vss(4, pInChannel+op+1) ;
	      __vr vrin_r1s0 = _ve_vldu_vss(4, pInChannel+op+inWidth) ;
	      __vr vrin_r1s1 = _ve_vldu_vss(4, pInChannel+op+inWidth+1) ;

	      vrin_r0s1 = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), vrin_r0s1, vm_r0s1) ;
	      vrin_r1s0 = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), vrin_r1s0, vm_r1s0) ;
	      vrin_r1s1 = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), vrin_r1s1, vm_r1s1) ;


	      vrsum = _ve_vfmads_vvsv(vrsum, pKerValue[0], vrin_r0s0) ;
	      vrsum = _ve_vfmads_vvsv(vrsum, pKerValue[1], vrin_r0s1) ;
	      vrsum = _ve_vfmads_vvsv(vrsum, pKerValue[2], vrin_r1s0) ;
	      vrsum = _ve_vfmads_vvsv(vrsum, pKerValue[3], vrin_r1s1) ;
	    } // inChannel

	    _ve_vstu_vss(vrsum, 4, pOut+outIndex) ;

	    outIndex += vl ;
	  } // outPixels
	  k++ ;
	}
	if ( ((outChannelGroup >> 1) & 0x01) == 1 ) {
	  int64_t outIndex0 = outGroupOffset + (n * outChannel + k  ) * oPixels ;
	  int64_t outIndex1 = outGroupOffset + (n * outChannel + k+1) * oPixels ;

	  for (int64_t op = 0; op < oPixels; op+=VLEN) {
	    const int64_t vl = oPixels - op < VLEN ? oPixels - op : VLEN ;

	    _ve_lvl(vl) ;

	    __vr vrseq = _ve_vseq_v() ;			// xy
	    __vr vridx = _ve_vaddsl_vsv(op, vrseq) ;	// op + xy

	    __vr vrsum01 = _ve_pvbrd_vs_i64(0UL) ;

	    __vr vry   = _ve_vdivsl_vvs(vridx, outWidth) ;
	    __vr vrx   = _ve_vsubsl_vvv(vridx, _ve_vmulul_vsv(outWidth,vry)) ;

	    __vm256 vm_r1 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(inHeight-1,vry)) ;	// condition(y+1 < inHeight)
	    __vm256 vm_s1 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(inWidth-1,vrx)) ;	// condition(x+1 < inWidth)

	    __vm256 vm_r1s0 = vm_r1 ;
	    __vm256 vm_r0s1 = vm_s1 ;
	    __vm256 vm_r1s1 = _ve_andm_mmm(vm_r1,vm_s1) ;

	    for (int64_t c = 0; c < inChannelGroup; c++) {

	      const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;
	      const float *pKerValue  = pKernel + kernGroupOffset + ((k * inChannelGroup + c) * kernHeight ) * kernWidth ;

	      /* memory access errors mihgt be caused */
	      __vr vrin_r0s0 = _ve_vldu_vss(4, pInChannel+op) ;
	      __vr vrin_r0s1 = _ve_vldu_vss(4, pInChannel+op+1) ;
	      __vr vrin_r1s0 = _ve_vldu_vss(4, pInChannel+op+inWidth) ;
	      __vr vrin_r1s1 = _ve_vldu_vss(4, pInChannel+op+inWidth+1) ;

	      vrin_r0s1 = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), vrin_r0s1, vm_r0s1) ;
	      vrin_r1s0 = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), vrin_r1s0, vm_r1s0) ;
	      vrin_r1s1 = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), vrin_r1s1, vm_r1s1) ;

	      __vr vrinP_r0s0 = _ve_vshf_vvvs(vrin_r0s0, vrin_r0s0, VE_VSHUFFLE_YUZU) ;
	      __vr vrinP_r0s1 = _ve_vshf_vvvs(vrin_r0s1, vrin_r0s1, VE_VSHUFFLE_YUZU) ;
	      __vr vrinP_r1s0 = _ve_vshf_vvvs(vrin_r1s0, vrin_r1s0, VE_VSHUFFLE_YUZU) ;
	      __vr vrinP_r1s1 = _ve_vshf_vvvs(vrin_r1s1, vrin_r1s1, VE_VSHUFFLE_YUZU) ;

	      const uint64_t kerValue01_r0s0 = _ve_pack_f32p(pKerValue,
							     pKerValue+      inChannelGroup * kernHeight * kernWidth) ;
	      vrsum01 = _ve_pvfmad_vvsv(vrsum01, kerValue01_r0s0, vrinP_r0s0) ;


	      const uint64_t kerValue01_r0s1 = _ve_pack_f32p(pKerValue                                               +1,
							     pKerValue+      inChannelGroup * kernHeight * kernWidth +1) ;
	      vrsum01 = _ve_pvfmad_vvsv(vrsum01, kerValue01_r0s1, vrinP_r0s1) ;


	      const uint64_t kerValue01_r1s0 = _ve_pack_f32p(pKerValue                                               +2,
							     pKerValue+      inChannelGroup * kernHeight * kernWidth +2) ;
	      vrsum01 = _ve_pvfmad_vvsv(vrsum01, kerValue01_r1s0, vrinP_r1s0) ;


	      const uint64_t kerValue01_r1s1 = _ve_pack_f32p(pKerValue                                               +3,
							     pKerValue+      inChannelGroup * kernHeight * kernWidth +3) ;
	      vrsum01 = _ve_pvfmad_vvsv(vrsum01, kerValue01_r1s1, vrinP_r1s1) ;
	    } // inChannel

	    _ve_vstu_vss(vrsum01, 4, pOut+outIndex0) ;
	    _ve_vstl_vss(vrsum01, 4, pOut+outIndex1) ;

	    outIndex0 += vl ;
	    outIndex1 += vl ;
	  } // outPixels

	  k+=2 ;
	}
	for ( ; k < outChannelGroup; ) {
//	if ( ((outChannelGroup >> 2) & 0x01) == 1 ) {
	  int64_t outIndex0 = outGroupOffset + (n * outChannel + k  ) * oPixels ;
	  int64_t outIndex1 = outGroupOffset + (n * outChannel + k+1) * oPixels ;
	  int64_t outIndex2 = outGroupOffset + (n * outChannel + k+2) * oPixels ;
	  int64_t outIndex3 = outGroupOffset + (n * outChannel + k+3) * oPixels ;

	  for (int64_t op = 0; op < oPixels; op+=VLEN) {
	    const int64_t vl = oPixels - op < VLEN ? oPixels - op : VLEN ;

	    _ve_lvl(vl) ;

	    __vr vrseq = _ve_vseq_v() ;			// xy
	    __vr vridx = _ve_vaddsl_vsv(op, vrseq) ;	// op + xy

	    __vr vrsum01 = _ve_pvbrd_vs_i64(0UL) ;
	    __vr vrsum23 = _ve_pvbrd_vs_i64(0UL) ;

	    __vr vry   = _ve_vdivsl_vvs(vridx, outWidth) ;
	    __vr vrx   = _ve_vsubsl_vvv(vridx, _ve_vmulul_vsv(outWidth,vry)) ;

	    __vm256 vm_r1 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(inHeight-1,vry)) ;	// condition(y+1 < inHeight)
	    __vm256 vm_s1 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(inWidth-1,vrx)) ;	// condition(x+1 < inWidth)

	    __vm256 vm_r1s0 = vm_r1 ;
	    __vm256 vm_r0s1 = vm_s1 ;
	    __vm256 vm_r1s1 = _ve_andm_mmm(vm_r1,vm_s1) ;

	    for (int64_t c = 0; c < inChannelGroup; c++) {

	      const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;
	      const float *pKerValue  = pKernel + kernGroupOffset + ((k * inChannelGroup + c) * kernHeight ) * kernWidth ;

	      /* memory access errors mihgt be caused */
	      __vr vrin_r0s0 = _ve_vldu_vss(4, pInChannel+op) ;
	      __vr vrin_r0s1 = _ve_vldu_vss(4, pInChannel+op+1) ;
	      __vr vrin_r1s0 = _ve_vldu_vss(4, pInChannel+op+inWidth) ;
	      __vr vrin_r1s1 = _ve_vldu_vss(4, pInChannel+op+inWidth+1) ;

	      vrin_r0s1 = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), vrin_r0s1, vm_r0s1) ;
	      vrin_r1s0 = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), vrin_r1s0, vm_r1s0) ;
	      vrin_r1s1 = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), vrin_r1s1, vm_r1s1) ;

	      __vr vrinP_r0s0 = _ve_vshf_vvvs(vrin_r0s0, vrin_r0s0, VE_VSHUFFLE_YUZU) ;
	      __vr vrinP_r0s1 = _ve_vshf_vvvs(vrin_r0s1, vrin_r0s1, VE_VSHUFFLE_YUZU) ;
	      __vr vrinP_r1s0 = _ve_vshf_vvvs(vrin_r1s0, vrin_r1s0, VE_VSHUFFLE_YUZU) ;
	      __vr vrinP_r1s1 = _ve_vshf_vvvs(vrin_r1s1, vrin_r1s1, VE_VSHUFFLE_YUZU) ;

	      const uint64_t kerValue01_r0s0 = _ve_pack_f32p(pKerValue,
							     pKerValue+      inChannelGroup * kernHeight * kernWidth) ;
	      const uint64_t kerValue23_r0s0 = _ve_pack_f32p(pKerValue + 2 * inChannelGroup * kernHeight * kernWidth,
							     pKerValue + 3 * inChannelGroup * kernHeight * kernWidth) ;
	      vrsum01 = _ve_pvfmad_vvsv(vrsum01, kerValue01_r0s0, vrinP_r0s0) ;
	      vrsum23 = _ve_pvfmad_vvsv(vrsum23, kerValue23_r0s0, vrinP_r0s0) ;


	      const uint64_t kerValue01_r0s1 = _ve_pack_f32p(pKerValue                                               +1,
							     pKerValue+      inChannelGroup * kernHeight * kernWidth +1) ;
	      const uint64_t kerValue23_r0s1 = _ve_pack_f32p(pKerValue + 2 * inChannelGroup * kernHeight * kernWidth +1,
							     pKerValue + 3 * inChannelGroup * kernHeight * kernWidth +1) ;
	      vrsum01 = _ve_pvfmad_vvsv(vrsum01, kerValue01_r0s1, vrinP_r0s1) ;
	      vrsum23 = _ve_pvfmad_vvsv(vrsum23, kerValue23_r0s1, vrinP_r0s1) ;


	      const uint64_t kerValue01_r1s0 = _ve_pack_f32p(pKerValue                                               +2,
							     pKerValue+      inChannelGroup * kernHeight * kernWidth +2) ;
	      const uint64_t kerValue23_r1s0 = _ve_pack_f32p(pKerValue + 2 * inChannelGroup * kernHeight * kernWidth +2,
							     pKerValue + 3 * inChannelGroup * kernHeight * kernWidth +2) ;
	      vrsum01 = _ve_pvfmad_vvsv(vrsum01, kerValue01_r1s0, vrinP_r1s0) ;
	      vrsum23 = _ve_pvfmad_vvsv(vrsum23, kerValue23_r1s0, vrinP_r1s0) ;


	      const uint64_t kerValue01_r1s1 = _ve_pack_f32p(pKerValue                                               +3,
							     pKerValue+      inChannelGroup * kernHeight * kernWidth +3) ;
	      const uint64_t kerValue23_r1s1 = _ve_pack_f32p(pKerValue + 2 * inChannelGroup * kernHeight * kernWidth +3,
							     pKerValue + 3 * inChannelGroup * kernHeight * kernWidth +3) ;
	      vrsum01 = _ve_pvfmad_vvsv(vrsum01, kerValue01_r1s1, vrinP_r1s1) ;
	      vrsum23 = _ve_pvfmad_vvsv(vrsum23, kerValue23_r1s1, vrinP_r1s1) ;
	    } // inChannel

	    _ve_vstu_vss(vrsum01, 4, pOut+outIndex0) ;
	    _ve_vstl_vss(vrsum01, 4, pOut+outIndex1) ;
	    _ve_vstu_vss(vrsum23, 4, pOut+outIndex2) ;
	    _ve_vstl_vss(vrsum23, 4, pOut+outIndex3) ;

	    outIndex0 += vl ;
	    outIndex1 += vl ;
	    outIndex2 += vl ;
	    outIndex3 += vl ;
	  } // outPixels

	  k+=4 ;
	}
#if 0
	for ( ; k < outChannelGroup; k+=8) {
	  int64_t outIndex0 = outGroupOffset + (n * outChannel + k  ) * oPixels ;
	  int64_t outIndex1 = outGroupOffset + (n * outChannel + k+1) * oPixels ;
	  int64_t outIndex2 = outGroupOffset + (n * outChannel + k+2) * oPixels ;
	  int64_t outIndex3 = outGroupOffset + (n * outChannel + k+3) * oPixels ;
	  int64_t outIndex4 = outGroupOffset + (n * outChannel + k+4) * oPixels ;
	  int64_t outIndex5 = outGroupOffset + (n * outChannel + k+5) * oPixels ;
	  int64_t outIndex6 = outGroupOffset + (n * outChannel + k+6) * oPixels ;
	  int64_t outIndex7 = outGroupOffset + (n * outChannel + k+7) * oPixels ;

	  for (int64_t op = 0; op < oPixels; op+=VLEN) {
	    const int64_t vl = oPixels - op < VLEN ? oPixels - op : VLEN ;

	    _ve_lvl(vl) ;

	    __vr vrseq = _ve_vseq_v() ;			// xy
	    __vr vridx = _ve_vaddsl_vsv(op, vrseq) ;	// op + xy

	    __vr vrsum01 = _ve_pvbrd_vs_i64(0UL) ;
	    __vr vrsum23 = _ve_pvbrd_vs_i64(0UL) ;
	    __vr vrsum45 = _ve_pvbrd_vs_i64(0UL) ;
	    __vr vrsum67 = _ve_pvbrd_vs_i64(0UL) ;

	    __vr vry   = _ve_vdivsl_vvs(vridx, outWidth) ;
	    __vr vrx   = _ve_vsubsl_vvv(vridx, _ve_vmulul_vsv(outWidth,vry)) ;

	    __vm256 vm_r1 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(inHeight-1,vry)) ;	// condition(y+1 < inHeight)
	    __vm256 vm_s1 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(inWidth-1,vrx)) ;	// condition(x+1 < inWidth)

	    __vm256 vm_r1s0 = vm_r1 ;
	    __vm256 vm_r0s1 = vm_s1 ;
	    __vm256 vm_r1s1 = _ve_andm_mmm(vm_r1,vm_s1) ;

	    for (int64_t c = 0; c < inChannelGroup; c++) {

	      const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;
	      const float *pKerValue  = pKernel + kernGroupOffset + ((k * inChannelGroup + c) * kernHeight ) * kernWidth ;

	      /* memory access errors mihgt be caused */
	      __vr vrin_r0s0 = _ve_vldu_vss(4, pInChannel+op) ;
	      __vr vrin_r0s1 = _ve_vldu_vss(4, pInChannel+op+1) ;
	      __vr vrin_r1s0 = _ve_vldu_vss(4, pInChannel+op+inWidth) ;
	      __vr vrin_r1s1 = _ve_vldu_vss(4, pInChannel+op+inWidth+1) ;

	      vrin_r0s1 = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), vrin_r0s1, vm_r0s1) ;
	      vrin_r1s0 = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), vrin_r1s0, vm_r1s0) ;
	      vrin_r1s1 = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), vrin_r1s1, vm_r1s1) ;

	      __vr vrinP_r0s0 = _ve_vshf_vvvs(vrin_r0s0, vrin_r0s0, VE_VSHUFFLE_YUZU) ;
	      __vr vrinP_r0s1 = _ve_vshf_vvvs(vrin_r0s1, vrin_r0s1, VE_VSHUFFLE_YUZU) ;
	      __vr vrinP_r1s0 = _ve_vshf_vvvs(vrin_r1s0, vrin_r1s0, VE_VSHUFFLE_YUZU) ;
	      __vr vrinP_r1s1 = _ve_vshf_vvvs(vrin_r1s1, vrin_r1s1, VE_VSHUFFLE_YUZU) ;

	      const uint64_t kerValue01_r0s0 = _ve_pack_f32p(pKerValue,
							     pKerValue+      inChannelGroup * kernHeight * kernWidth) ;
	      const uint64_t kerValue23_r0s0 = _ve_pack_f32p(pKerValue + 2 * inChannelGroup * kernHeight * kernWidth,
							     pKerValue + 3 * inChannelGroup * kernHeight * kernWidth) ;
	      const uint64_t kerValue45_r0s0 = _ve_pack_f32p(pKerValue + 4 * inChannelGroup * kernHeight * kernWidth,
							     pKerValue + 5 * inChannelGroup * kernHeight * kernWidth) ;
	      const uint64_t kerValue67_r0s0 = _ve_pack_f32p(pKerValue + 6 * inChannelGroup * kernHeight * kernWidth,
							     pKerValue + 7 * inChannelGroup * kernHeight * kernWidth) ;
	      vrsum01 = _ve_pvfmad_vvsv(vrsum01, kerValue01_r0s0, vrinP_r0s0) ;
	      vrsum23 = _ve_pvfmad_vvsv(vrsum23, kerValue23_r0s0, vrinP_r0s0) ;
	      vrsum45 = _ve_pvfmad_vvsv(vrsum45, kerValue45_r0s0, vrinP_r0s0) ;
	      vrsum67 = _ve_pvfmad_vvsv(vrsum67, kerValue67_r0s0, vrinP_r0s0) ;


	      const uint64_t kerValue01_r0s1 = _ve_pack_f32p(pKerValue                                               +1,
							     pKerValue+      inChannelGroup * kernHeight * kernWidth +1) ;
	      const uint64_t kerValue23_r0s1 = _ve_pack_f32p(pKerValue + 2 * inChannelGroup * kernHeight * kernWidth +1,
							     pKerValue + 3 * inChannelGroup * kernHeight * kernWidth +1) ;
	      const uint64_t kerValue45_r0s1 = _ve_pack_f32p(pKerValue + 4 * inChannelGroup * kernHeight * kernWidth +1,
							     pKerValue + 5 * inChannelGroup * kernHeight * kernWidth +1) ;
	      const uint64_t kerValue67_r0s1 = _ve_pack_f32p(pKerValue + 6 * inChannelGroup * kernHeight * kernWidth +1,
							     pKerValue + 7 * inChannelGroup * kernHeight * kernWidth +1) ;
	      vrsum01 = _ve_pvfmad_vvsv(vrsum01, kerValue01_r0s1, vrinP_r0s1) ;
	      vrsum23 = _ve_pvfmad_vvsv(vrsum23, kerValue23_r0s1, vrinP_r0s1) ;
	      vrsum45 = _ve_pvfmad_vvsv(vrsum45, kerValue45_r0s1, vrinP_r0s1) ;
	      vrsum67 = _ve_pvfmad_vvsv(vrsum67, kerValue67_r0s1, vrinP_r0s1) ;


	      const uint64_t kerValue01_r1s0 = _ve_pack_f32p(pKerValue                                               +2,
							     pKerValue+      inChannelGroup * kernHeight * kernWidth +2) ;
	      const uint64_t kerValue23_r1s0 = _ve_pack_f32p(pKerValue + 2 * inChannelGroup * kernHeight * kernWidth +2,
							     pKerValue + 3 * inChannelGroup * kernHeight * kernWidth +2) ;
	      const uint64_t kerValue45_r1s0 = _ve_pack_f32p(pKerValue + 4 * inChannelGroup * kernHeight * kernWidth +2,
							     pKerValue + 5 * inChannelGroup * kernHeight * kernWidth +2) ;
	      const uint64_t kerValue67_r1s0 = _ve_pack_f32p(pKerValue + 6 * inChannelGroup * kernHeight * kernWidth +2,
							     pKerValue + 7 * inChannelGroup * kernHeight * kernWidth +2) ;
	      vrsum01 = _ve_pvfmad_vvsv(vrsum01, kerValue01_r1s0, vrinP_r1s0) ;
	      vrsum23 = _ve_pvfmad_vvsv(vrsum23, kerValue23_r1s0, vrinP_r1s0) ;
	      vrsum45 = _ve_pvfmad_vvsv(vrsum45, kerValue45_r1s0, vrinP_r1s0) ;
	      vrsum67 = _ve_pvfmad_vvsv(vrsum67, kerValue67_r1s0, vrinP_r1s0) ;


	      const uint64_t kerValue01_r1s1 = _ve_pack_f32p(pKerValue                                               +3,
							     pKerValue+      inChannelGroup * kernHeight * kernWidth +3) ;
	      const uint64_t kerValue23_r1s1 = _ve_pack_f32p(pKerValue + 2 * inChannelGroup * kernHeight * kernWidth +3,
							     pKerValue + 3 * inChannelGroup * kernHeight * kernWidth +3) ;
	      const uint64_t kerValue45_r1s1 = _ve_pack_f32p(pKerValue + 4 * inChannelGroup * kernHeight * kernWidth +3,
							     pKerValue + 5 * inChannelGroup * kernHeight * kernWidth +3) ;
	      const uint64_t kerValue67_r1s1 = _ve_pack_f32p(pKerValue + 6 * inChannelGroup * kernHeight * kernWidth +3,
							     pKerValue + 7 * inChannelGroup * kernHeight * kernWidth +3) ;
	      vrsum01 = _ve_pvfmad_vvsv(vrsum01, kerValue01_r1s1, vrinP_r1s1) ;
	      vrsum23 = _ve_pvfmad_vvsv(vrsum23, kerValue23_r1s1, vrinP_r1s1) ;
	      vrsum45 = _ve_pvfmad_vvsv(vrsum45, kerValue45_r1s1, vrinP_r1s1) ;
	      vrsum67 = _ve_pvfmad_vvsv(vrsum67, kerValue67_r1s1, vrinP_r1s1) ;
	    } // inChannel

	    _ve_vstu_vss(vrsum01, 4, pOut+outIndex0) ;
	    _ve_vstl_vss(vrsum01, 4, pOut+outIndex1) ;
	    _ve_vstu_vss(vrsum23, 4, pOut+outIndex2) ;
	    _ve_vstl_vss(vrsum23, 4, pOut+outIndex3) ;
	    _ve_vstu_vss(vrsum45, 4, pOut+outIndex4) ;
	    _ve_vstl_vss(vrsum45, 4, pOut+outIndex5) ;
	    _ve_vstu_vss(vrsum67, 4, pOut+outIndex6) ;
	    _ve_vstl_vss(vrsum67, 4, pOut+outIndex7) ;

	    outIndex0 += vl ;
	    outIndex1 += vl ;
	    outIndex2 += vl ;
	    outIndex3 += vl ;
	    outIndex4 += vl ;
	    outIndex5 += vl ;
	    outIndex6 += vl ;
	    outIndex7 += vl ;
	  } // outPixels
	} // outChannel
#endif
      } // group
    } // batch
  }

  return VEDNN_SUCCESS;
}
