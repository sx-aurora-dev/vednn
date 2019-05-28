#include <stdint.h>
#include <stdlib.h>

#include "vednn.h"

#include "veintrin.h"
#define VLEN	(256)


vednnError_t
vednnConvolutionForward_direct_dil1_str1_padsame_ker3(
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
  const int64_t outWidth   = pParamOut->width;
  const int64_t outHeight  = pParamOut->height;
  const int64_t kernWidth  = pParamKernel->width;		/* must be 3 */
  const int64_t kernHeight = pParamKernel->height;		/* must be 3 */

  const int64_t group          = pParamConv->group;
//  const int64_t strideWidth    = pParamConv->strideWidth;	/* must be 1 */
//  const int64_t strideHeight   = pParamConv->strideHeight;	/* must be 1 */
  const int64_t padWidth       = pParamConv->padWidth;
  const int64_t padHeight      = pParamConv->padHeight;
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

	    __vr vrsum = _ve_vbrdu_vs_f32(0.0f) ;

	    __vr vrseq = _ve_vseq_v() ;			// xy
	    __vr vridx = _ve_vaddsl_vsv(op, vrseq) ;	// op + xy

	    __vr vry   = _ve_vdivsl_vvs(vridx, outWidth) ;
	    __vr vrx   = _ve_vsubsl_vvv(vridx, _ve_vmulul_vsv(outWidth,vry)) ;

	    __vr vrh_r0 = _ve_vaddsl_vsv( -padHeight, vry) ;
	    __vr vrh_r2 = _ve_vaddsl_vsv(2-padHeight, vry) ;

	    __vr vrw_s0 = _ve_vaddsl_vsv( -padWidth, vrx) ;
	    __vr vrw_s2 = _ve_vaddsl_vsv(2-padWidth, vrx) ;

	    __vm256 vm01_r0 = _ve_vfmkl_mcv(VECC_GE, vrh_r0) ;
	    __vm256 vm01_r2 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(inHeight,vrh_r2)) ;

	    __vm256 vm23_s0  = _ve_vfmkl_mcv(VECC_GE, vrw_s0) ;
	    __vm256 vm23_s2  = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(inWidth,vrw_s2)) ;


	    __vm256 vmall_r0s0 = _ve_andm_mmm(vm01_r0,vm23_s0) ;
	    __vm256 vmall_r0s1 = vm01_r0 ;
	    __vm256 vmall_r0s2 = _ve_andm_mmm(vm01_r0, vm23_s2) ;

	    __vm256 vmall_r1s0 = vm23_s0 ;
	    __vm256 vmall_r1s2 = vm23_s2 ;

	    __vm256 vmall_r2s0 = _ve_andm_mmm(vm01_r2,vm23_s0) ;
	    __vm256 vmall_r2s1 = vm01_r2 ;
	    __vm256 vmall_r2s2 = _ve_andm_mmm(vm01_r2, vm23_s2) ;

	    for (int64_t c = 0; c < inChannelGroup; c++) {
	      const float *pKerValue = pKernel + kernGroupOffset + ((k * inChannelGroup + c) * kernHeight) * kernWidth ;

	      const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;

	      /* memory access errors mihgt be caused */
	      __vr vrin_r0s0 = _ve_vldu_vss(4,&pInChannel[op-inWidth-1]) ;
	      __vr vrin_r0s1 = _ve_vldu_vss(4,&pInChannel[op-inWidth  ]) ;
	      __vr vrin_r0s2 = _ve_vldu_vss(4,&pInChannel[op-inWidth+1]) ;
	      __vr vrin_r1s0 = _ve_vldu_vss(4,&pInChannel[op+       -1]) ;
	      __vr vrin_r1s1 = _ve_vldu_vss(4,&pInChannel[op          ]) ;
	      __vr vrin_r1s2 = _ve_vldu_vss(4,&pInChannel[op+       +1]) ;
	      __vr vrin_r2s0 = _ve_vldu_vss(4,&pInChannel[op+inWidth-1]) ;
	      __vr vrin_r2s1 = _ve_vldu_vss(4,&pInChannel[op+inWidth  ]) ;
	      __vr vrin_r2s2 = _ve_vldu_vss(4,&pInChannel[op+inWidth+1]) ;

	      __vr vrzerof = _ve_vbrdu_vs_f32(0.0f) ;

	      vrin_r0s0 = _ve_vmrg_vvvm(vrzerof, vrin_r0s0, vmall_r0s0) ;
	      vrin_r0s1 = _ve_vmrg_vvvm(vrzerof, vrin_r0s1, vmall_r0s1) ;
	      vrin_r0s2 = _ve_vmrg_vvvm(vrzerof, vrin_r0s2, vmall_r0s2) ;

	      vrin_r1s0 = _ve_vmrg_vvvm(vrzerof, vrin_r1s0, vmall_r1s0) ;
	      vrin_r1s2 = _ve_vmrg_vvvm(vrzerof, vrin_r1s2, vmall_r1s2) ;

	      vrin_r2s0 = _ve_vmrg_vvvm(vrzerof, vrin_r2s0, vmall_r2s0) ;
	      vrin_r2s1 = _ve_vmrg_vvvm(vrzerof, vrin_r2s1, vmall_r2s1) ;
	      vrin_r2s2 = _ve_vmrg_vvvm(vrzerof, vrin_r2s2, vmall_r2s2) ;

#define FILTER_OC1(VRIN,VRSUM)									\
{												\
  VRSUM = _ve_vfmads_vvsv(VRSUM, *pKerValue, VRIN) ;						\
}
		FILTER_OC1(vrin_r0s0, vrsum) ; pKerValue++ ;
		FILTER_OC1(vrin_r0s1, vrsum) ; pKerValue++ ;
		FILTER_OC1(vrin_r0s2, vrsum) ; pKerValue++ ;
		FILTER_OC1(vrin_r1s0, vrsum) ; pKerValue++ ;
		FILTER_OC1(vrin_r1s1, vrsum) ; pKerValue++ ;
		FILTER_OC1(vrin_r1s2, vrsum) ; pKerValue++ ;
		FILTER_OC1(vrin_r2s0, vrsum) ; pKerValue++ ;
		FILTER_OC1(vrin_r2s1, vrsum) ; pKerValue++ ;
		FILTER_OC1(vrin_r2s2, vrsum) ; pKerValue++ ;
#undef FILTER_OC1

	    } // inChannel

	    _ve_vstu_vss(vrsum, 4, pOut+outIndex) ;

	    outIndex += vl ;
	  } // outPixels
	  k++ ;
	}
	if ( ((outChannelGroup >> 1) & 0x01) == 1 ) {
	  int64_t outIndex = outGroupOffset + (n * outChannel + k  ) * oPixels ;

	  for (int64_t op = 0; op < oPixels; op+=VLEN) {
	    const int64_t vl = oPixels - op < VLEN ? oPixels - op : VLEN ;

	    _ve_lvl(vl) ;

	    __vr vrsum01 = _ve_pvbrd_vs_i64(0UL) ;

	    __vr vrseq = _ve_vseq_v() ;			// xy
	    __vr vridx = _ve_vaddsl_vsv(op, vrseq) ;	// op + xy

	    __vr vry   = _ve_vdivsl_vvs(vridx, outWidth) ;
	    __vr vrx   = _ve_vsubsl_vvv(vridx, _ve_vmulul_vsv(outWidth,vry)) ;

	    __vr vrh_r0 = _ve_vaddsl_vsv( -padHeight, vry) ;
	    __vr vrh_r2 = _ve_vaddsl_vsv(2-padHeight, vry) ;

	    __vr vrw_s0 = _ve_vaddsl_vsv( -padWidth, vrx) ;
	    __vr vrw_s2 = _ve_vaddsl_vsv(2-padWidth, vrx) ;

	    __vm256 vm01_r0 = _ve_vfmkl_mcv(VECC_GE, vrh_r0) ;
	    __vm256 vm01_r2 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(inHeight,vrh_r2)) ;

	    __vm256 vm23_s0  = _ve_vfmkl_mcv(VECC_GE, vrw_s0) ;
	    __vm256 vm23_s2  = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(inWidth,vrw_s2)) ;


	    __vm256 vmall_r0s0 = _ve_andm_mmm(vm01_r0,vm23_s0) ;
	    __vm256 vmall_r0s1 = vm01_r0 ;
	    __vm256 vmall_r0s2 = _ve_andm_mmm(vm01_r0, vm23_s2) ;

	    __vm256 vmall_r1s0 = vm23_s0 ;
	    __vm256 vmall_r1s2 = vm23_s2 ;

	    __vm256 vmall_r2s0 = _ve_andm_mmm(vm01_r2,vm23_s0) ;
	    __vm256 vmall_r2s1 = vm01_r2 ;
	    __vm256 vmall_r2s2 = _ve_andm_mmm(vm01_r2, vm23_s2) ;

	    for (int64_t c = 0; c < inChannelGroup; c++) {
	      const float *pKerValue = pKernel + kernGroupOffset + ((k * inChannelGroup + c) * kernHeight) * kernWidth ;

	      const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;

	      /* memory access errors mihgt be caused */
	      __vr vrin_r0s0 = _ve_vldu_vss(4,&pInChannel[op-inWidth-1]) ;
	      __vr vrin_r0s1 = _ve_vldu_vss(4,&pInChannel[op-inWidth  ]) ;
	      __vr vrin_r0s2 = _ve_vldu_vss(4,&pInChannel[op-inWidth+1]) ;
	      __vr vrin_r1s0 = _ve_vldu_vss(4,&pInChannel[op+       -1]) ;
	      __vr vrin_r1s1 = _ve_vldu_vss(4,&pInChannel[op          ]) ;
	      __vr vrin_r1s2 = _ve_vldu_vss(4,&pInChannel[op+       +1]) ;
	      __vr vrin_r2s0 = _ve_vldu_vss(4,&pInChannel[op+inWidth-1]) ;
	      __vr vrin_r2s1 = _ve_vldu_vss(4,&pInChannel[op+inWidth  ]) ;
	      __vr vrin_r2s2 = _ve_vldu_vss(4,&pInChannel[op+inWidth+1]) ;

	      __vr vrzerof = _ve_vbrdu_vs_f32(0.0f) ;

	      vrin_r0s0 = _ve_vmrg_vvvm(vrzerof, vrin_r0s0, vmall_r0s0) ;
	      vrin_r0s1 = _ve_vmrg_vvvm(vrzerof, vrin_r0s1, vmall_r0s1) ;
	      vrin_r0s2 = _ve_vmrg_vvvm(vrzerof, vrin_r0s2, vmall_r0s2) ;
	      __vr vrinP_r0s0 = _ve_vshf_vvvs(vrin_r0s0, vrin_r0s0, VE_VSHUFFLE_YUZU) ;
	      __vr vrinP_r0s1 = _ve_vshf_vvvs(vrin_r0s1, vrin_r0s1, VE_VSHUFFLE_YUZU) ;
	      __vr vrinP_r0s2 = _ve_vshf_vvvs(vrin_r0s2, vrin_r0s2, VE_VSHUFFLE_YUZU) ;

	      vrin_r1s0 = _ve_vmrg_vvvm(vrzerof, vrin_r1s0, vmall_r1s0) ;
	      vrin_r1s2 = _ve_vmrg_vvvm(vrzerof, vrin_r1s2, vmall_r1s2) ;
	      __vr vrinP_r1s0 = _ve_vshf_vvvs(vrin_r1s0, vrin_r1s0, VE_VSHUFFLE_YUZU) ;
	      __vr vrinP_r1s1 = _ve_vshf_vvvs(vrin_r1s1, vrin_r1s1, VE_VSHUFFLE_YUZU) ;
	      __vr vrinP_r1s2 = _ve_vshf_vvvs(vrin_r1s2, vrin_r1s2, VE_VSHUFFLE_YUZU) ;

	      vrin_r2s0 = _ve_vmrg_vvvm(vrzerof, vrin_r2s0, vmall_r2s0) ;
	      vrin_r2s1 = _ve_vmrg_vvvm(vrzerof, vrin_r2s1, vmall_r2s1) ;
	      vrin_r2s2 = _ve_vmrg_vvvm(vrzerof, vrin_r2s2, vmall_r2s2) ;
	      __vr vrinP_r2s0 = _ve_vshf_vvvs(vrin_r2s0, vrin_r2s0, VE_VSHUFFLE_YUZU) ;
	      __vr vrinP_r2s1 = _ve_vshf_vvvs(vrin_r2s1, vrin_r2s1, VE_VSHUFFLE_YUZU) ;
	      __vr vrinP_r2s2 = _ve_vshf_vvvs(vrin_r2s2, vrin_r2s2, VE_VSHUFFLE_YUZU) ;

#define FILTER_OC2(VRIN,VRSUM)									\
{												\
  const uint64_t kerValue = _ve_pack_f32p(pKerValue,						\
					  pKerValue + inChannelGroup * kernHeight * kernWidth) ;\
  VRSUM = _ve_pvfmad_vvsv(VRSUM, kerValue, VRIN) ;						\
}
		FILTER_OC2(vrinP_r0s0, vrsum01) ; pKerValue++ ;
		FILTER_OC2(vrinP_r0s1, vrsum01) ; pKerValue++ ;
		FILTER_OC2(vrinP_r0s2, vrsum01) ; pKerValue++ ;
		FILTER_OC2(vrinP_r1s0, vrsum01) ; pKerValue++ ;
		FILTER_OC2(vrinP_r1s1, vrsum01) ; pKerValue++ ;
		FILTER_OC2(vrinP_r1s2, vrsum01) ; pKerValue++ ;
		FILTER_OC2(vrinP_r2s0, vrsum01) ; pKerValue++ ;
		FILTER_OC2(vrinP_r2s1, vrsum01) ; pKerValue++ ;
		FILTER_OC2(vrinP_r2s2, vrsum01) ; pKerValue++ ;
#undef FILTER_OC2

	    } // inChannel

	    _ve_vstu_vss(vrsum01, 4, pOut+outIndex) ;
	    _ve_vstl_vss(vrsum01, 4, pOut+outIndex+   oPixels) ;

	    outIndex += vl ;
	  } // outPixels

	  k+=2 ;
	}
	if ( ((outChannelGroup >> 2) & 0x01) == 1 ) {
	  int64_t outIndex = outGroupOffset + (n * outChannel + k  ) * oPixels ;

	  for (int64_t op = 0; op < oPixels; op+=VLEN) {
	    const int64_t vl = oPixels - op < VLEN ? oPixels - op : VLEN ;

	    _ve_lvl(vl) ;

	    __vr vrsum01 = _ve_pvbrd_vs_i64(0UL) ;
	    __vr vrsum23 = _ve_pvbrd_vs_i64(0UL) ;

	    __vr vrseq = _ve_vseq_v() ;			// xy
	    __vr vridx = _ve_vaddsl_vsv(op, vrseq) ;	// op + xy

	    __vr vry   = _ve_vdivsl_vvs(vridx, outWidth) ;
	    __vr vrx   = _ve_vsubsl_vvv(vridx, _ve_vmulul_vsv(outWidth,vry)) ;

	    __vr vrh_r0 = _ve_vaddsl_vsv( -padHeight, vry) ;
	    __vr vrh_r2 = _ve_vaddsl_vsv(2-padHeight, vry) ;

	    __vr vrw_s0 = _ve_vaddsl_vsv( -padWidth, vrx) ;
	    __vr vrw_s2 = _ve_vaddsl_vsv(2-padWidth, vrx) ;

	    __vm256 vm01_r0 = _ve_vfmkl_mcv(VECC_GE, vrh_r0) ;
	    __vm256 vm01_r2 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(inHeight,vrh_r2)) ;

	    __vm256 vm23_s0  = _ve_vfmkl_mcv(VECC_GE, vrw_s0) ;
	    __vm256 vm23_s2  = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(inWidth,vrw_s2)) ;


	    __vm256 vmall_r0s0 = _ve_andm_mmm(vm01_r0,vm23_s0) ;
	    __vm256 vmall_r0s1 = vm01_r0 ;
	    __vm256 vmall_r0s2 = _ve_andm_mmm(vm01_r0, vm23_s2) ;

	    __vm256 vmall_r1s0 = vm23_s0 ;
	    __vm256 vmall_r1s2 = vm23_s2 ;

	    __vm256 vmall_r2s0 = _ve_andm_mmm(vm01_r2,vm23_s0) ;
	    __vm256 vmall_r2s1 = vm01_r2 ;
	    __vm256 vmall_r2s2 = _ve_andm_mmm(vm01_r2, vm23_s2) ;

	    for (int64_t c = 0; c < inChannelGroup; c++) {
	      const float *pKerValue = pKernel + kernGroupOffset + ((k * inChannelGroup + c) * kernHeight) * kernWidth ;

	      const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;

	      /* memory access errors mihgt be caused */
	      __vr vrin_r0s0 = _ve_vldu_vss(4,&pInChannel[op-inWidth-1]) ;
	      __vr vrin_r0s1 = _ve_vldu_vss(4,&pInChannel[op-inWidth  ]) ;
	      __vr vrin_r0s2 = _ve_vldu_vss(4,&pInChannel[op-inWidth+1]) ;
	      __vr vrin_r1s0 = _ve_vldu_vss(4,&pInChannel[op+       -1]) ;
	      __vr vrin_r1s1 = _ve_vldu_vss(4,&pInChannel[op          ]) ;
	      __vr vrin_r1s2 = _ve_vldu_vss(4,&pInChannel[op+       +1]) ;
	      __vr vrin_r2s0 = _ve_vldu_vss(4,&pInChannel[op+inWidth-1]) ;
	      __vr vrin_r2s1 = _ve_vldu_vss(4,&pInChannel[op+inWidth  ]) ;
	      __vr vrin_r2s2 = _ve_vldu_vss(4,&pInChannel[op+inWidth+1]) ;

	      __vr vrzerof = _ve_vbrdu_vs_f32(0.0f) ;

	      vrin_r0s0 = _ve_vmrg_vvvm(vrzerof, vrin_r0s0, vmall_r0s0) ;
	      vrin_r0s1 = _ve_vmrg_vvvm(vrzerof, vrin_r0s1, vmall_r0s1) ;
	      vrin_r0s2 = _ve_vmrg_vvvm(vrzerof, vrin_r0s2, vmall_r0s2) ;
	      __vr vrinP_r0s0 = _ve_vshf_vvvs(vrin_r0s0, vrin_r0s0, VE_VSHUFFLE_YUZU) ;
	      __vr vrinP_r0s1 = _ve_vshf_vvvs(vrin_r0s1, vrin_r0s1, VE_VSHUFFLE_YUZU) ;
	      __vr vrinP_r0s2 = _ve_vshf_vvvs(vrin_r0s2, vrin_r0s2, VE_VSHUFFLE_YUZU) ;

	      vrin_r1s0 = _ve_vmrg_vvvm(vrzerof, vrin_r1s0, vmall_r1s0) ;
	      vrin_r1s2 = _ve_vmrg_vvvm(vrzerof, vrin_r1s2, vmall_r1s2) ;
	      __vr vrinP_r1s0 = _ve_vshf_vvvs(vrin_r1s0, vrin_r1s0, VE_VSHUFFLE_YUZU) ;
	      __vr vrinP_r1s1 = _ve_vshf_vvvs(vrin_r1s1, vrin_r1s1, VE_VSHUFFLE_YUZU) ;
	      __vr vrinP_r1s2 = _ve_vshf_vvvs(vrin_r1s2, vrin_r1s2, VE_VSHUFFLE_YUZU) ;

	      vrin_r2s0 = _ve_vmrg_vvvm(vrzerof, vrin_r2s0, vmall_r2s0) ;
	      vrin_r2s1 = _ve_vmrg_vvvm(vrzerof, vrin_r2s1, vmall_r2s1) ;
	      vrin_r2s2 = _ve_vmrg_vvvm(vrzerof, vrin_r2s2, vmall_r2s2) ;
	      __vr vrinP_r2s0 = _ve_vshf_vvvs(vrin_r2s0, vrin_r2s0, VE_VSHUFFLE_YUZU) ;
	      __vr vrinP_r2s1 = _ve_vshf_vvvs(vrin_r2s1, vrin_r2s1, VE_VSHUFFLE_YUZU) ;
	      __vr vrinP_r2s2 = _ve_vshf_vvvs(vrin_r2s2, vrin_r2s2, VE_VSHUFFLE_YUZU) ;

#define FILTER_OC4(VRIN,VRSUM0,VRSUM1)									\
{													\
  const uint64_t kerValue0 = _ve_pack_f32p(pKerValue,							\
					   pKerValue +     inChannelGroup * kernHeight * kernWidth) ;	\
  const uint64_t kerValue1 = _ve_pack_f32p(pKerValue + 2 * inChannelGroup * kernHeight * kernWidth,	\
					   pKerValue + 3 * inChannelGroup * kernHeight * kernWidth) ;	\
  VRSUM0 = _ve_pvfmad_vvsv(VRSUM0, kerValue0, VRIN) ;								\
  VRSUM1 = _ve_pvfmad_vvsv(VRSUM1, kerValue1, VRIN) ;								\
}
		FILTER_OC4(vrinP_r0s0, vrsum01,vrsum23) ; pKerValue++ ;
		FILTER_OC4(vrinP_r0s1, vrsum01,vrsum23) ; pKerValue++ ;
		FILTER_OC4(vrinP_r0s2, vrsum01,vrsum23) ; pKerValue++ ;
		FILTER_OC4(vrinP_r1s0, vrsum01,vrsum23) ; pKerValue++ ;
		FILTER_OC4(vrinP_r1s1, vrsum01,vrsum23) ; pKerValue++ ;
		FILTER_OC4(vrinP_r1s2, vrsum01,vrsum23) ; pKerValue++ ;
		FILTER_OC4(vrinP_r2s0, vrsum01,vrsum23) ; pKerValue++ ;
		FILTER_OC4(vrinP_r2s1, vrsum01,vrsum23) ; pKerValue++ ;
		FILTER_OC4(vrinP_r2s2, vrsum01,vrsum23) ; pKerValue++ ;
#undef FILTER_OC4

	    } // inChannel

	    _ve_vstu_vss(vrsum01, 4, pOut+outIndex) ;
	    _ve_vstl_vss(vrsum01, 4, pOut+outIndex+   oPixels) ;
	    _ve_vstu_vss(vrsum23, 4, pOut+outIndex+ 2*oPixels) ;
	    _ve_vstl_vss(vrsum23, 4, pOut+outIndex+ 3*oPixels) ;

	    outIndex += vl ;
	  } // outPixels

	  k+=4 ;
	}
	if ( ((outChannelGroup >> 3) & 0x01) == 1 ) {
	  int64_t outIndex = outGroupOffset + (n * outChannel + k  ) * oPixels ;

	  for (int64_t op = 0; op < oPixels; op+=VLEN) {
	    const int64_t vl = oPixels - op < VLEN ? oPixels - op : VLEN ;

	    _ve_lvl(vl) ;

	    __vr vrsum01 = _ve_pvbrd_vs_i64(0UL) ;
	    __vr vrsum23 = _ve_pvbrd_vs_i64(0UL) ;
	    __vr vrsum45 = _ve_pvbrd_vs_i64(0UL) ;
	    __vr vrsum67 = _ve_pvbrd_vs_i64(0UL) ;

	    __vr vrseq = _ve_vseq_v() ;			// xy
	    __vr vridx = _ve_vaddsl_vsv(op, vrseq) ;	// op + xy

	    __vr vry   = _ve_vdivsl_vvs(vridx, outWidth) ;
	    __vr vrx   = _ve_vsubsl_vvv(vridx, _ve_vmulul_vsv(outWidth,vry)) ;

	    __vr vrh_r0 = _ve_vaddsl_vsv( -padHeight, vry) ;
	    __vr vrh_r2 = _ve_vaddsl_vsv(2-padHeight, vry) ;

	    __vr vrw_s0 = _ve_vaddsl_vsv( -padWidth, vrx) ;
	    __vr vrw_s2 = _ve_vaddsl_vsv(2-padWidth, vrx) ;

	    __vm256 vm01_r0 = _ve_vfmkl_mcv(VECC_GE, vrh_r0) ;
	    __vm256 vm01_r2 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(inHeight,vrh_r2)) ;

	    __vm256 vm23_s0  = _ve_vfmkl_mcv(VECC_GE, vrw_s0) ;
	    __vm256 vm23_s2  = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(inWidth,vrw_s2)) ;


	    __vm256 vmall_r0s0 = _ve_andm_mmm(vm01_r0,vm23_s0) ;
	    __vm256 vmall_r0s1 = vm01_r0 ;
	    __vm256 vmall_r0s2 = _ve_andm_mmm(vm01_r0, vm23_s2) ;

	    __vm256 vmall_r1s0 = vm23_s0 ;
	    __vm256 vmall_r1s2 = vm23_s2 ;

	    __vm256 vmall_r2s0 = _ve_andm_mmm(vm01_r2,vm23_s0) ;
	    __vm256 vmall_r2s1 = vm01_r2 ;
	    __vm256 vmall_r2s2 = _ve_andm_mmm(vm01_r2, vm23_s2) ;

	    for (int64_t c = 0; c < inChannelGroup; c++) {
	      const float *pKerValue = pKernel + kernGroupOffset + ((k * inChannelGroup + c) * kernHeight) * kernWidth ;

	      const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;

	      /* memory access errors mihgt be caused */
	      __vr vrin_r0s0 = _ve_vldu_vss(4,&pInChannel[op-inWidth-1]) ;
	      __vr vrin_r0s1 = _ve_vldu_vss(4,&pInChannel[op-inWidth  ]) ;
	      __vr vrin_r0s2 = _ve_vldu_vss(4,&pInChannel[op-inWidth+1]) ;
	      __vr vrin_r1s0 = _ve_vldu_vss(4,&pInChannel[op+       -1]) ;
	      __vr vrin_r1s1 = _ve_vldu_vss(4,&pInChannel[op          ]) ;
	      __vr vrin_r1s2 = _ve_vldu_vss(4,&pInChannel[op+       +1]) ;
	      __vr vrin_r2s0 = _ve_vldu_vss(4,&pInChannel[op+inWidth-1]) ;
	      __vr vrin_r2s1 = _ve_vldu_vss(4,&pInChannel[op+inWidth  ]) ;
	      __vr vrin_r2s2 = _ve_vldu_vss(4,&pInChannel[op+inWidth+1]) ;

	      __vr vrzerof = _ve_vbrdu_vs_f32(0.0f) ;

	      vrin_r0s0 = _ve_vmrg_vvvm(vrzerof, vrin_r0s0, vmall_r0s0) ;
	      vrin_r0s1 = _ve_vmrg_vvvm(vrzerof, vrin_r0s1, vmall_r0s1) ;
	      vrin_r0s2 = _ve_vmrg_vvvm(vrzerof, vrin_r0s2, vmall_r0s2) ;
	      __vr vrinP_r0s0 = _ve_vshf_vvvs(vrin_r0s0, vrin_r0s0, VE_VSHUFFLE_YUZU) ;
	      __vr vrinP_r0s1 = _ve_vshf_vvvs(vrin_r0s1, vrin_r0s1, VE_VSHUFFLE_YUZU) ;
	      __vr vrinP_r0s2 = _ve_vshf_vvvs(vrin_r0s2, vrin_r0s2, VE_VSHUFFLE_YUZU) ;

	      vrin_r1s0 = _ve_vmrg_vvvm(vrzerof, vrin_r1s0, vmall_r1s0) ;
	      vrin_r1s2 = _ve_vmrg_vvvm(vrzerof, vrin_r1s2, vmall_r1s2) ;
	      __vr vrinP_r1s0 = _ve_vshf_vvvs(vrin_r1s0, vrin_r1s0, VE_VSHUFFLE_YUZU) ;
	      __vr vrinP_r1s1 = _ve_vshf_vvvs(vrin_r1s1, vrin_r1s1, VE_VSHUFFLE_YUZU) ;
	      __vr vrinP_r1s2 = _ve_vshf_vvvs(vrin_r1s2, vrin_r1s2, VE_VSHUFFLE_YUZU) ;

	      vrin_r2s0 = _ve_vmrg_vvvm(vrzerof, vrin_r2s0, vmall_r2s0) ;
	      vrin_r2s1 = _ve_vmrg_vvvm(vrzerof, vrin_r2s1, vmall_r2s1) ;
	      vrin_r2s2 = _ve_vmrg_vvvm(vrzerof, vrin_r2s2, vmall_r2s2) ;
	      __vr vrinP_r2s0 = _ve_vshf_vvvs(vrin_r2s0, vrin_r2s0, VE_VSHUFFLE_YUZU) ;
	      __vr vrinP_r2s1 = _ve_vshf_vvvs(vrin_r2s1, vrin_r2s1, VE_VSHUFFLE_YUZU) ;
	      __vr vrinP_r2s2 = _ve_vshf_vvvs(vrin_r2s2, vrin_r2s2, VE_VSHUFFLE_YUZU) ;

#define FILTER_OC4(VRIN,VRSUM0,VRSUM1,N)									\
{														\
  const uint64_t kerValue0 = _ve_pack_f32p(pKerValue +  (N)    * inChannelGroup * kernHeight * kernWidth,	\
					   pKerValue + ((N)+1) * inChannelGroup * kernHeight * kernWidth) ;	\
  const uint64_t kerValue1 = _ve_pack_f32p(pKerValue + ((N)+2) * inChannelGroup * kernHeight * kernWidth,	\
					   pKerValue + ((N)+3) * inChannelGroup * kernHeight * kernWidth) ;	\
  VRSUM0 = _ve_pvfmad_vvsv(VRSUM0, kerValue0, VRIN) ;								\
  VRSUM1 = _ve_pvfmad_vvsv(VRSUM1, kerValue1, VRIN) ;								\
}
#define FILTER_OC8(VRIN)		\
{					\
  FILTER_OC4(VRIN,vrsum01,vrsum23,0) ;	\
  FILTER_OC4(VRIN,vrsum45,vrsum67,4) ;	\
}
		FILTER_OC8(vrinP_r0s0) ; pKerValue++ ;
		FILTER_OC8(vrinP_r0s1) ; pKerValue++ ;
		FILTER_OC8(vrinP_r0s2) ; pKerValue++ ;
		FILTER_OC8(vrinP_r1s0) ; pKerValue++ ;
		FILTER_OC8(vrinP_r1s1) ; pKerValue++ ;
		FILTER_OC8(vrinP_r1s2) ; pKerValue++ ;
		FILTER_OC8(vrinP_r2s0) ; pKerValue++ ;
		FILTER_OC8(vrinP_r2s1) ; pKerValue++ ;
		FILTER_OC8(vrinP_r2s2) ; pKerValue++ ;
#undef FILTER_OC4
#undef FILTER_OC8
	    } // inChannel

	    _ve_vstu_vss(vrsum01, 4, pOut+outIndex) ;
	    _ve_vstl_vss(vrsum01, 4, pOut+outIndex+   oPixels) ;
	    _ve_vstu_vss(vrsum23, 4, pOut+outIndex+ 2*oPixels) ;
	    _ve_vstl_vss(vrsum23, 4, pOut+outIndex+ 3*oPixels) ;
	    _ve_vstu_vss(vrsum45, 4, pOut+outIndex+ 4*oPixels) ;
	    _ve_vstl_vss(vrsum45, 4, pOut+outIndex+ 5*oPixels) ;
	    _ve_vstu_vss(vrsum67, 4, pOut+outIndex+ 6*oPixels) ;
	    _ve_vstl_vss(vrsum67, 4, pOut+outIndex+ 7*oPixels) ;

	    outIndex += vl ;
	  } // outPixels
	  k+=8 ;
	}
	for ( ; k < outChannelGroup; k+=16) {
	  int64_t outIndex = outGroupOffset + (n * outChannel + k  ) * oPixels ;

	  for (int64_t op = 0; op < oPixels; op+=VLEN) {
	    const int64_t vl = oPixels - op < VLEN ? oPixels - op : VLEN ;

	    _ve_lvl(vl) ;

	    __vr vrsum01 = _ve_pvbrd_vs_i64(0UL) ;
	    __vr vrsum23 = _ve_pvbrd_vs_i64(0UL) ;
	    __vr vrsum45 = _ve_pvbrd_vs_i64(0UL) ;
	    __vr vrsum67 = _ve_pvbrd_vs_i64(0UL) ;
	    __vr vrsum89 = _ve_pvbrd_vs_i64(0UL) ;
	    __vr vrsumAB = _ve_pvbrd_vs_i64(0UL) ;
	    __vr vrsumCD = _ve_pvbrd_vs_i64(0UL) ;
	    __vr vrsumEF = _ve_pvbrd_vs_i64(0UL) ;

	    __vr vrseq = _ve_vseq_v() ;			// xy
	    __vr vridx = _ve_vaddsl_vsv(op, vrseq) ;	// op + xy

	    __vr vry   = _ve_vdivsl_vvs(vridx, outWidth) ;
	    __vr vrx   = _ve_vsubsl_vvv(vridx, _ve_vmulul_vsv(outWidth,vry)) ;

	    __vr vrh_r0 = _ve_vaddsl_vsv( -padHeight, vry) ;
	    __vr vrh_r2 = _ve_vaddsl_vsv(2-padHeight, vry) ;

	    __vr vrw_s0 = _ve_vaddsl_vsv( -padWidth, vrx) ;
	    __vr vrw_s2 = _ve_vaddsl_vsv(2-padWidth, vrx) ;

	    __vm256 vm01_r0 = _ve_vfmkl_mcv(VECC_GE, vrh_r0) ;
	    __vm256 vm01_r2 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(inHeight,vrh_r2)) ;

	    __vm256 vm23_s0  = _ve_vfmkl_mcv(VECC_GE, vrw_s0) ;
	    __vm256 vm23_s2  = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(inWidth,vrw_s2)) ;


	    __vm256 vmall_r0s0 = _ve_andm_mmm(vm01_r0,vm23_s0) ;
	    __vm256 vmall_r0s1 = vm01_r0 ;
	    __vm256 vmall_r0s2 = _ve_andm_mmm(vm01_r0, vm23_s2) ;

	    __vm256 vmall_r1s0 = vm23_s0 ;
	    __vm256 vmall_r1s2 = vm23_s2 ;

	    __vm256 vmall_r2s0 = _ve_andm_mmm(vm01_r2,vm23_s0) ;
	    __vm256 vmall_r2s1 = vm01_r2 ;
	    __vm256 vmall_r2s2 = _ve_andm_mmm(vm01_r2, vm23_s2) ;

	    for (int64_t c = 0; c < inChannelGroup; c++) {
	      const float *pKerValue = pKernel + kernGroupOffset + ((k * inChannelGroup + c) * kernHeight) * kernWidth ;

	      const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;

	      /* memory access errors mihgt be caused */
	      __vr vrin_r0s0 = _ve_vldu_vss(4,&pInChannel[op-inWidth-1]) ;
	      __vr vrin_r0s1 = _ve_vldu_vss(4,&pInChannel[op-inWidth  ]) ;
	      __vr vrin_r0s2 = _ve_vldu_vss(4,&pInChannel[op-inWidth+1]) ;
	      __vr vrin_r1s0 = _ve_vldu_vss(4,&pInChannel[op+       -1]) ;
	      __vr vrin_r1s1 = _ve_vldu_vss(4,&pInChannel[op          ]) ;
	      __vr vrin_r1s2 = _ve_vldu_vss(4,&pInChannel[op+       +1]) ;
	      __vr vrin_r2s0 = _ve_vldu_vss(4,&pInChannel[op+inWidth-1]) ;
	      __vr vrin_r2s1 = _ve_vldu_vss(4,&pInChannel[op+inWidth  ]) ;
	      __vr vrin_r2s2 = _ve_vldu_vss(4,&pInChannel[op+inWidth+1]) ;

	      __vr vrzerof = _ve_vbrdu_vs_f32(0.0f) ;

	      vrin_r0s0 = _ve_vmrg_vvvm(vrzerof, vrin_r0s0, vmall_r0s0) ;
	      vrin_r0s1 = _ve_vmrg_vvvm(vrzerof, vrin_r0s1, vmall_r0s1) ;
	      vrin_r0s2 = _ve_vmrg_vvvm(vrzerof, vrin_r0s2, vmall_r0s2) ;
	      __vr vrinP_r0s0 = _ve_vshf_vvvs(vrin_r0s0, vrin_r0s0, VE_VSHUFFLE_YUZU) ;
	      __vr vrinP_r0s1 = _ve_vshf_vvvs(vrin_r0s1, vrin_r0s1, VE_VSHUFFLE_YUZU) ;
	      __vr vrinP_r0s2 = _ve_vshf_vvvs(vrin_r0s2, vrin_r0s2, VE_VSHUFFLE_YUZU) ;

	      vrin_r1s0 = _ve_vmrg_vvvm(vrzerof, vrin_r1s0, vmall_r1s0) ;
	      vrin_r1s2 = _ve_vmrg_vvvm(vrzerof, vrin_r1s2, vmall_r1s2) ;
	      __vr vrinP_r1s0 = _ve_vshf_vvvs(vrin_r1s0, vrin_r1s0, VE_VSHUFFLE_YUZU) ;
	      __vr vrinP_r1s1 = _ve_vshf_vvvs(vrin_r1s1, vrin_r1s1, VE_VSHUFFLE_YUZU) ;
	      __vr vrinP_r1s2 = _ve_vshf_vvvs(vrin_r1s2, vrin_r1s2, VE_VSHUFFLE_YUZU) ;

	      vrin_r2s0 = _ve_vmrg_vvvm(vrzerof, vrin_r2s0, vmall_r2s0) ;
	      vrin_r2s1 = _ve_vmrg_vvvm(vrzerof, vrin_r2s1, vmall_r2s1) ;
	      vrin_r2s2 = _ve_vmrg_vvvm(vrzerof, vrin_r2s2, vmall_r2s2) ;
	      __vr vrinP_r2s0 = _ve_vshf_vvvs(vrin_r2s0, vrin_r2s0, VE_VSHUFFLE_YUZU) ;
	      __vr vrinP_r2s1 = _ve_vshf_vvvs(vrin_r2s1, vrin_r2s1, VE_VSHUFFLE_YUZU) ;
	      __vr vrinP_r2s2 = _ve_vshf_vvvs(vrin_r2s2, vrin_r2s2, VE_VSHUFFLE_YUZU) ;

#define FILTER_OC4(VRIN,VRSUM0,VRSUM1,N)									\
{														\
  const uint64_t kerValue0 = _ve_pack_f32p(pKerValue +  (N)    * inChannelGroup * kernHeight * kernWidth,	\
					   pKerValue + ((N)+1) * inChannelGroup * kernHeight * kernWidth) ;	\
  const uint64_t kerValue1 = _ve_pack_f32p(pKerValue + ((N)+2) * inChannelGroup * kernHeight * kernWidth,	\
					   pKerValue + ((N)+3) * inChannelGroup * kernHeight * kernWidth) ;	\
  VRSUM0 = _ve_pvfmad_vvsv(VRSUM0, kerValue0, VRIN) ;								\
  VRSUM1 = _ve_pvfmad_vvsv(VRSUM1, kerValue1, VRIN) ;								\
}
#define FILTER_OC16(VRIN)		\
{					\
  FILTER_OC4(VRIN,vrsum01,vrsum23,0) ;	\
  FILTER_OC4(VRIN,vrsum45,vrsum67,4) ;	\
  FILTER_OC4(VRIN,vrsum89,vrsumAB,8) ;	\
  FILTER_OC4(VRIN,vrsumCD,vrsumEF,12) ;	\
}
		FILTER_OC16(vrinP_r0s0) ; pKerValue++ ;
		FILTER_OC16(vrinP_r0s1) ; pKerValue++ ;
		FILTER_OC16(vrinP_r0s2) ; pKerValue++ ;
		FILTER_OC16(vrinP_r1s0) ; pKerValue++ ;
		FILTER_OC16(vrinP_r1s1) ; pKerValue++ ;
		FILTER_OC16(vrinP_r1s2) ; pKerValue++ ;
		FILTER_OC16(vrinP_r2s0) ; pKerValue++ ;
		FILTER_OC16(vrinP_r2s1) ; pKerValue++ ;
		FILTER_OC16(vrinP_r2s2) ; pKerValue++ ;
#undef FILTER_OC4
#undef FILTER_OC16
	    } // inChannel

	    _ve_vstu_vss(vrsum01, 4, pOut+outIndex) ;
	    _ve_vstl_vss(vrsum01, 4, pOut+outIndex+   oPixels) ;
	    _ve_vstu_vss(vrsum23, 4, pOut+outIndex+ 2*oPixels) ;
	    _ve_vstl_vss(vrsum23, 4, pOut+outIndex+ 3*oPixels) ;
	    _ve_vstu_vss(vrsum45, 4, pOut+outIndex+ 4*oPixels) ;
	    _ve_vstl_vss(vrsum45, 4, pOut+outIndex+ 5*oPixels) ;
	    _ve_vstu_vss(vrsum67, 4, pOut+outIndex+ 6*oPixels) ;
	    _ve_vstl_vss(vrsum67, 4, pOut+outIndex+ 7*oPixels) ;
	    _ve_vstu_vss(vrsum89, 4, pOut+outIndex+ 8*oPixels) ;
	    _ve_vstl_vss(vrsum89, 4, pOut+outIndex+ 9*oPixels) ;
	    _ve_vstu_vss(vrsumAB, 4, pOut+outIndex+10*oPixels) ;
	    _ve_vstl_vss(vrsumAB, 4, pOut+outIndex+11*oPixels) ;
	    _ve_vstu_vss(vrsumCD, 4, pOut+outIndex+12*oPixels) ;
	    _ve_vstl_vss(vrsumCD, 4, pOut+outIndex+13*oPixels) ;
	    _ve_vstu_vss(vrsumEF, 4, pOut+outIndex+14*oPixels) ;
	    _ve_vstl_vss(vrsumEF, 4, pOut+outIndex+15*oPixels) ;

	    outIndex += vl ;
	  } // outPixels
	} // outChannel
      } // group
    } // batch
  }

  return VEDNN_SUCCESS;
}

