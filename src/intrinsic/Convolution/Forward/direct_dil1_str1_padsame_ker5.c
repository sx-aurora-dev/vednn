#include <stdint.h>
#include <stdlib.h>

#include "vednn.h"

#include "veintrin.h"
#define VLEN	(256)

vednnError_t
vednnConvolutionForward_direct_dil1_str1_padsame_ker5(
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
  const int64_t kernWidth  = pParamKernel->width;		/* must be 2*padWidth  + 1 */
  const int64_t kernHeight = pParamKernel->height;		/* must be 2*padHeight + 1 */

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

	    __vr vrseq = _ve_vseq_v() ;			// xy
	    __vr vridx = _ve_vaddsl_vsv(op, vrseq) ;	// op + xy

	    __vr vrsum = _ve_vbrdu_vs_f32(0.0f) ;

	    __vr vry   = _ve_vdivsl_vvs(vridx, outWidth) ;
	    __vr vrx   = _ve_vsubsl_vvv(vridx, _ve_vmulul_vsv(outWidth,vry)) ;

	    __vm256 vmw0_s0 = _ve_vfmkl_mcv(VECC_GE, _ve_vaddsl_vsv(-2,  vrx)) ;		// condition(0 <= w)
	    __vm256 vmw0_s1 = _ve_vfmkl_mcv(VECC_GE, _ve_vaddsl_vsv(-1,  vrx)) ;		// condition(0 <= w)

	    __vm256 vmw1_s3 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(inWidth-1,vrx)) ;	// condition(w < inWidth)
	    __vm256 vmw1_s4 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(inWidth-2,vrx)) ;	// condition(w < inWidth)

	    __vm256 vmw_s0  = vmw0_s0 ;
	    __vm256 vmw_s1  = vmw0_s1 ;
	    __vm256 vmw_s3  = vmw1_s3 ;
	    __vm256 vmw_s4  = vmw1_s4 ;

	    for (int64_t r = 0; r < kernHeight; r++) {
	      __vr vrh = _ve_vaddsl_vsv(r-padHeight, vry) ;

	      __vm256 vmh0 = _ve_vfmkl_mcv(VECC_GE, vrh) ;				// condition(0 <= h)
	      __vm256 vmh1 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(inHeight,vrh)) ;	// condition(h < inHeight)
	      __vm256 vmh  = _ve_andm_mmm(vmh0, vmh1) ;

	      __vm256 vmall_s0 = _ve_andm_mmm(vmh,vmw_s0) ;
	      __vm256 vmall_s1 = _ve_andm_mmm(vmh,vmw_s1) ;
	      __vm256 vmall_s2 = vmh ;
	      __vm256 vmall_s3 = _ve_andm_mmm(vmh,vmw_s3) ;
	      __vm256 vmall_s4 = _ve_andm_mmm(vmh,vmw_s4) ;

	      for (int64_t c = 0; c < inChannelGroup; c++) {
		const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;

		/* memory access errors mihgt be caused */
		__vr vrin_s0 = _ve_vldu_vss(4,&pInChannel[op+(r-padHeight)*inWidth-2]) ;
		__vr vrin_s1 = _ve_vldu_vss(4,&pInChannel[op+(r-padHeight)*inWidth-1]) ;
		__vr vrin_s2 = _ve_vldu_vss(4,&pInChannel[op+(r-padHeight)*inWidth  ]) ;
		__vr vrin_s3 = _ve_vldu_vss(4,&pInChannel[op+(r-padHeight)*inWidth+1]) ;
		__vr vrin_s4 = _ve_vldu_vss(4,&pInChannel[op+(r-padHeight)*inWidth+2]) ;

		const float *pKerValue = pKernel + kernGroupOffset + ((k * inChannelGroup + c) * kernHeight + r) * kernWidth ;

#define VFMAD1(VRIN, VMR, PKERVALUE) {				\
  VRIN = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), VRIN, VMR) ;	\
  vrsum = _ve_vfmads_vvsv(vrsum, *(PKERVALUE), VRIN) ;		\
}
		VFMAD1(vrin_s0, vmall_s0, pKerValue) ; pKerValue++ ;
		VFMAD1(vrin_s1, vmall_s1, pKerValue) ; pKerValue++ ;
		VFMAD1(vrin_s2, vmall_s2, pKerValue) ; pKerValue++ ;
		VFMAD1(vrin_s3, vmall_s3, pKerValue) ; pKerValue++ ;
		VFMAD1(vrin_s4, vmall_s4, pKerValue) ; pKerValue++ ;
#undef VFMAD1

	      } // inChannel
	    } // kernHeight

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

	    __vm256 vmw0_s0 = _ve_vfmkl_mcv(VECC_GE, _ve_vaddsl_vsv(-2,  vrx)) ;		// condition(0 <= w)
	    __vm256 vmw0_s1 = _ve_vfmkl_mcv(VECC_GE, _ve_vaddsl_vsv(-1,  vrx)) ;		// condition(0 <= w)

	    __vm256 vmw1_s3 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(inWidth-1,vrx)) ;	// condition(w < inWidth)
	    __vm256 vmw1_s4 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(inWidth-2,vrx)) ;	// condition(w < inWidth)

	    __vm256 vmw_s0  = vmw0_s0 ;
	    __vm256 vmw_s1  = vmw0_s1 ;
	    __vm256 vmw_s3  = vmw1_s3 ;
	    __vm256 vmw_s4  = vmw1_s4 ;

	    for (int64_t r = 0; r < kernHeight; r++) {
	      __vr vrh = _ve_vaddsl_vsv(r-padHeight, vry) ;

	      __vm256 vmh0 = _ve_vfmkl_mcv(VECC_GE, vrh) ;				// condition(0 <= h)
	      __vm256 vmh1 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(inHeight,vrh)) ;	// condition(h < inHeight)
	      __vm256 vmh  = _ve_andm_mmm(vmh0, vmh1) ;

	      __vm256 vmall_s0 = _ve_andm_mmm(vmh,vmw_s0) ;
	      __vm256 vmall_s1 = _ve_andm_mmm(vmh,vmw_s1) ;
	      __vm256 vmall_s2 = vmh ;
	      __vm256 vmall_s3 = _ve_andm_mmm(vmh,vmw_s3) ;
	      __vm256 vmall_s4 = _ve_andm_mmm(vmh,vmw_s4) ;

	      for (int64_t c = 0; c < inChannelGroup; c++) {
		const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;

		/* memory access errors mihgt be caused */
		__vr vrin_s0 = _ve_vldu_vss(4,&pInChannel[op+(r-padHeight)*inWidth-2]) ;
		__vr vrin_s1 = _ve_vldu_vss(4,&pInChannel[op+(r-padHeight)*inWidth-1]) ;
		__vr vrin_s2 = _ve_vldu_vss(4,&pInChannel[op+(r-padHeight)*inWidth  ]) ;
		__vr vrin_s3 = _ve_vldu_vss(4,&pInChannel[op+(r-padHeight)*inWidth+1]) ;
		__vr vrin_s4 = _ve_vldu_vss(4,&pInChannel[op+(r-padHeight)*inWidth+2]) ;

		const float *pKerValue = pKernel + kernGroupOffset + ((k * inChannelGroup + c) * kernHeight + r) * kernWidth ;

#define PVFMAD2(VRIN, VMR, PKERVALUE) {				\
  VRIN = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), VRIN, VMR) ;	\
  __vr vrinP = _ve_vshf_vvvs(VRIN, VRIN, VE_VSHUFFLE_YUZU) ;	\
  const uint64_t kerValue01 = _ve_pack_f32p(PKERVALUE,		\
					    PKERVALUE +     inChannelGroup * kernHeight * kernWidth) ;	\
  vrsum01 = _ve_pvfmad_vvsv(vrsum01, kerValue01, vrinP) ;	\
}
		PVFMAD2(vrin_s0, vmall_s0, pKerValue) ; pKerValue++ ;
		PVFMAD2(vrin_s1, vmall_s1, pKerValue) ; pKerValue++ ;
		PVFMAD2(vrin_s2, vmall_s2, pKerValue) ; pKerValue++ ;
		PVFMAD2(vrin_s3, vmall_s3, pKerValue) ; pKerValue++ ;
		PVFMAD2(vrin_s4, vmall_s4, pKerValue) ; pKerValue++ ;
#undef PVFMAD2

	      } // inChannel
	    } // kernHeight

	    _ve_vstu_vss(vrsum01, 4, pOut+outIndex0) ;
	    _ve_vstl_vss(vrsum01, 4, pOut+outIndex1) ;

	    outIndex0 += vl ;
	    outIndex1 += vl ;
	  } // outPixels

	  k+=2 ;
	}
	if ( ((outChannelGroup >> 2) & 0x01) == 1 ) {
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

	    __vm256 vmw0_s0 = _ve_vfmkl_mcv(VECC_GE, _ve_vaddsl_vsv(-2,  vrx)) ;		// condition(0 <= w)
	    __vm256 vmw0_s1 = _ve_vfmkl_mcv(VECC_GE, _ve_vaddsl_vsv(-1,  vrx)) ;		// condition(0 <= w)

	    __vm256 vmw1_s3 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(inWidth-1,vrx)) ;	// condition(w < inWidth)
	    __vm256 vmw1_s4 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(inWidth-2,vrx)) ;	// condition(w < inWidth)

	    __vm256 vmw_s0  = vmw0_s0 ;
	    __vm256 vmw_s1  = vmw0_s1 ;
	    __vm256 vmw_s3  = vmw1_s3 ;
	    __vm256 vmw_s4  = vmw1_s4 ;

	    for (int64_t r = 0; r < kernHeight; r++) {
	      __vr vrh = _ve_vaddsl_vsv(r-padHeight, vry) ;

	      __vm256 vmh0 = _ve_vfmkl_mcv(VECC_GE, vrh) ;				// condition(0 <= h)
	      __vm256 vmh1 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(inHeight,vrh)) ;	// condition(h < inHeight)
	      __vm256 vmh  = _ve_andm_mmm(vmh0, vmh1) ;

	      __vm256 vmall_s0 = _ve_andm_mmm(vmh,vmw_s0) ;
	      __vm256 vmall_s1 = _ve_andm_mmm(vmh,vmw_s1) ;
	      __vm256 vmall_s2 = vmh ;
	      __vm256 vmall_s3 = _ve_andm_mmm(vmh,vmw_s3) ;
	      __vm256 vmall_s4 = _ve_andm_mmm(vmh,vmw_s4) ;

	      for (int64_t c = 0; c < inChannelGroup; c++) {
		const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;

		/* memory access errors mihgt be caused */
		__vr vrin_s0 = _ve_vldu_vss(4,&pInChannel[op+(r-padHeight)*inWidth-2]) ;
		__vr vrin_s1 = _ve_vldu_vss(4,&pInChannel[op+(r-padHeight)*inWidth-1]) ;
		__vr vrin_s2 = _ve_vldu_vss(4,&pInChannel[op+(r-padHeight)*inWidth  ]) ;
		__vr vrin_s3 = _ve_vldu_vss(4,&pInChannel[op+(r-padHeight)*inWidth+1]) ;
		__vr vrin_s4 = _ve_vldu_vss(4,&pInChannel[op+(r-padHeight)*inWidth+2]) ;

		const float *pKerValue = pKernel + kernGroupOffset + ((k * inChannelGroup + c) * kernHeight + r) * kernWidth ;

#define PVFMAD4(VRIN, VMR, PKERVALUE) {				\
  VRIN = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), VRIN, VMR) ;	\
  __vr vrinP = _ve_vshf_vvvs(VRIN, VRIN, VE_VSHUFFLE_YUZU) ;	\
  const uint64_t kerValue01 = _ve_pack_f32p(PKERVALUE,		\
					    PKERVALUE +     inChannelGroup * kernHeight * kernWidth) ;	\
  const uint64_t kerValue23 = _ve_pack_f32p(PKERVALUE + 2 * inChannelGroup * kernHeight * kernWidth,	\
					    PKERVALUE + 3 * inChannelGroup * kernHeight * kernWidth) ;	\
  vrsum01 = _ve_pvfmad_vvsv(vrsum01, kerValue01, vrinP) ;	\
  vrsum23 = _ve_pvfmad_vvsv(vrsum23, kerValue23, vrinP) ;	\
}
		PVFMAD4(vrin_s0, vmall_s0, pKerValue) ; pKerValue++ ;
		PVFMAD4(vrin_s1, vmall_s1, pKerValue) ; pKerValue++ ;
		PVFMAD4(vrin_s2, vmall_s2, pKerValue) ; pKerValue++ ;
		PVFMAD4(vrin_s3, vmall_s3, pKerValue) ; pKerValue++ ;
		PVFMAD4(vrin_s4, vmall_s4, pKerValue) ; pKerValue++ ;
#undef PVFMAD4

	      } // inChannel
	    } // kernHeight

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

	    __vm256 vmw0_s0 = _ve_vfmkl_mcv(VECC_GE, _ve_vaddsl_vsv(-2,  vrx)) ;		// condition(0 <= w)
	    __vm256 vmw0_s1 = _ve_vfmkl_mcv(VECC_GE, _ve_vaddsl_vsv(-1,  vrx)) ;		// condition(0 <= w)

	    __vm256 vmw1_s3 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(inWidth-1,vrx)) ;	// condition(w < inWidth)
	    __vm256 vmw1_s4 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(inWidth-2,vrx)) ;	// condition(w < inWidth)

	    __vm256 vmw_s0  = vmw0_s0 ;
	    __vm256 vmw_s1  = vmw0_s1 ;
	    __vm256 vmw_s3  = vmw1_s3 ;
	    __vm256 vmw_s4  = vmw1_s4 ;

	    for (int64_t r = 0; r < kernHeight; r++) {
	      __vr vrh = _ve_vaddsl_vsv(r-padHeight, vry) ;

	      __vm256 vmh0 = _ve_vfmkl_mcv(VECC_GE, vrh) ;				// condition(0 <= h)
	      __vm256 vmh1 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(inHeight,vrh)) ;	// condition(h < inHeight)
	      __vm256 vmh  = _ve_andm_mmm(vmh0, vmh1) ;

	      __vm256 vmall_s0 = _ve_andm_mmm(vmh,vmw_s0) ;
	      __vm256 vmall_s1 = _ve_andm_mmm(vmh,vmw_s1) ;
	      __vm256 vmall_s2 = vmh ;
	      __vm256 vmall_s3 = _ve_andm_mmm(vmh,vmw_s3) ;
	      __vm256 vmall_s4 = _ve_andm_mmm(vmh,vmw_s4) ;

	      for (int64_t c = 0; c < inChannelGroup; c++) {
		const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;

		/* memory access errors mihgt be caused */
		__vr vrin_s0 = _ve_vldu_vss(4,&pInChannel[op+(r-padHeight)*inWidth-2]) ;
		__vr vrin_s1 = _ve_vldu_vss(4,&pInChannel[op+(r-padHeight)*inWidth-1]) ;
		__vr vrin_s2 = _ve_vldu_vss(4,&pInChannel[op+(r-padHeight)*inWidth  ]) ;
		__vr vrin_s3 = _ve_vldu_vss(4,&pInChannel[op+(r-padHeight)*inWidth+1]) ;
		__vr vrin_s4 = _ve_vldu_vss(4,&pInChannel[op+(r-padHeight)*inWidth+2]) ;

		const float *pKerValue = pKernel + kernGroupOffset + ((k * inChannelGroup + c) * kernHeight + r) * kernWidth ;

#define PVFMAD8(VRIN, VMR, PKERVALUE) {				\
  VRIN = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), VRIN, VMR) ;	\
  __vr vrinP = _ve_vshf_vvvs(VRIN, VRIN, VE_VSHUFFLE_YUZU) ;	\
  const uint64_t kerValue01 = _ve_pack_f32p(PKERVALUE,		\
					    PKERVALUE +     inChannelGroup * kernHeight * kernWidth) ;	\
  const uint64_t kerValue23 = _ve_pack_f32p(PKERVALUE + 2 * inChannelGroup * kernHeight * kernWidth,	\
					    PKERVALUE + 3 * inChannelGroup * kernHeight * kernWidth) ;	\
  const uint64_t kerValue45 = _ve_pack_f32p(PKERVALUE + 4 * inChannelGroup * kernHeight * kernWidth,	\
					    PKERVALUE + 5 * inChannelGroup * kernHeight * kernWidth) ;	\
  const uint64_t kerValue67 = _ve_pack_f32p(PKERVALUE + 6 * inChannelGroup * kernHeight * kernWidth,	\
					    PKERVALUE + 7 * inChannelGroup * kernHeight * kernWidth) ;	\
  vrsum01 = _ve_pvfmad_vvsv(vrsum01, kerValue01, vrinP) ;	\
  vrsum23 = _ve_pvfmad_vvsv(vrsum23, kerValue23, vrinP) ;	\
  vrsum45 = _ve_pvfmad_vvsv(vrsum45, kerValue45, vrinP) ;	\
  vrsum67 = _ve_pvfmad_vvsv(vrsum67, kerValue67, vrinP) ;	\
}
		PVFMAD8(vrin_s0, vmall_s0, pKerValue) ; pKerValue++ ;
		PVFMAD8(vrin_s1, vmall_s1, pKerValue) ; pKerValue++ ;
		PVFMAD8(vrin_s2, vmall_s2, pKerValue) ; pKerValue++ ;
		PVFMAD8(vrin_s3, vmall_s3, pKerValue) ; pKerValue++ ;
		PVFMAD8(vrin_s4, vmall_s4, pKerValue) ; pKerValue++ ;
#undef PVFMAD8
	      } // inChannel
	    } // kernHeight

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
      } // group
    } // batch
  }

  return VEDNN_SUCCESS;
}

