#include <stdint.h>
#include <stdlib.h>

#include "vednn.h"

#include "veintrin.h"
#define VLEN	(256)

static inline void c1(
    const float * restrict pGOut,
    const float * restrict pKernel,
    float * restrict const pGIn,
    const int64_t gOutChannel,
    const int64_t gOutWidth,
    const int64_t gOutHeight,
    const int64_t gInChannel,
    const int64_t gInWidth,
    const int64_t gInHeight,
    const int64_t kernWidth,
    const int64_t kernHeight,
    const int64_t strideWidth,
    const int64_t strideHeight,
    const int64_t padWidth,
    const int64_t padHeight,
    const int64_t dilationWidth,
    const int64_t dilationHeight,
    const int64_t gInChannelGroup,
    const int64_t gOutChannelGroup,
    const int64_t gInGroupOffset,
    const int64_t gOutGroupOffset,
    const int64_t kernGroupOffset,
    const int64_t gInPixels,
    const int64_t n,
    const int64_t c
)
{
  for (int64_t h = 0; h < gInHeight ; h++ ) {
    for (int64_t w = 0; w < gInWidth ; w += VLEN ) {
      const int64_t vl = gInWidth - w < VLEN ? gInWidth - w  : VLEN ;

      const int64_t gInIndex = gInGroupOffset + ((n * gInChannel + c) * gInHeight + h) * gInWidth + w ;

      _ve_lvl(vl) ;

      __vr vrsum = _ve_vbrdu_vs_f32(0.f) ;

      __vr vrw   = _ve_vaddsl_vsv(w, _ve_vseq_v()) ;

      for (int64_t r=0; r<kernHeight; r++) {
	int64_t i = h - r * dilationHeight + padHeight ;
	int64_t y = i/strideHeight;
	if ( y*strideHeight != i || y < 0 || gOutHeight <= y)  continue ;

	for (int64_t s=0; s<kernWidth; s++) {
	  __vr vrj = _ve_vaddsl_vsv(padWidth-s*dilationWidth, vrw) ;
	  __vr vrx = _ve_vdivsl_vvs(vrj, strideWidth) ;

	  __vm256 vmx0 = _ve_vfmkl_mcv(VECC_IEQ, _ve_vcmpsl_vvv(vrj, _ve_vmulsl_vsv(strideWidth, vrx))) ;
	  __vm256 vmx1 = _ve_vfmkl_mcv(VECC_GE, vrx) ;
	  __vm256 vmx2 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(gOutWidth,vrx)) ;

	  __vm256 vmx = _ve_andm_mmm(_ve_andm_mmm(vmx0, vmx1), vmx2) ;

	  for (int64_t k=0; k<gOutChannelGroup; k++) {
	    int64_t gOutIndex    = gOutGroupOffset + ((n * gOutChannel + k) * gOutHeight) * gOutWidth ;
	    __vr vrgout_ptr = _ve_vsfa_vvss(_ve_vaddsl_vsv(gOutWidth*y, vrx),
					  2,
					  (unsigned long)(pGOut+gOutIndex)) ;

	    __vr vrgout = _ve_vgtu_vvm(vrgout_ptr, vmx) ;
	    vrgout = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), vrgout, vmx) ;


	    const float *pKerValue = pKernel + kernGroupOffset + ((k * gInChannelGroup + c) * kernHeight + r) * kernWidth + s;

	    vrsum = _ve_vfmads_vvsv(vrsum, *pKerValue, vrgout) ;
	  } // gOutChannel

	} // kernWidth
      } // kernHeight

      _ve_vstu_vss(vrsum, 4, pGIn+gInIndex) ;

    } // gInWidth
  } // gInHeight
}


static inline void c2(
    const float * restrict pGOut,
    const float * restrict pKernel,
    float * restrict const pGIn,
    const int64_t gOutChannel,
    const int64_t gOutWidth,
    const int64_t gOutHeight,
    const int64_t gInChannel,
    const int64_t gInWidth,
    const int64_t gInHeight,
    const int64_t kernWidth,
    const int64_t kernHeight,
    const int64_t strideWidth,
    const int64_t strideHeight,
    const int64_t padWidth,
    const int64_t padHeight,
    const int64_t dilationWidth,
    const int64_t dilationHeight,
    const int64_t gInChannelGroup,
    const int64_t gOutChannelGroup,
    const int64_t gInGroupOffset,
    const int64_t gOutGroupOffset,
    const int64_t kernGroupOffset,
    const int64_t gInPixels,
    const int64_t n,
    const int64_t c
)
{
  for (int64_t h = 0; h < gInHeight ; h++ ) {
    for (int64_t w = 0; w < gInWidth ; w += VLEN ) {
      const int64_t vl = gInWidth - w < VLEN ? gInWidth - w  : VLEN ;

      const int64_t gInIndex = gInGroupOffset + ((n * gInChannel + c) * gInHeight + h) * gInWidth + w ;

      _ve_lvl(vl) ;

      __vr vrsum01 = _ve_vbrd_vs_i64(0UL) ;

      __vr vrw   = _ve_vaddsl_vsv(w, _ve_vseq_v()) ;

      for (int64_t r=0; r<kernHeight; r++) {
	int64_t i = h - r * dilationHeight + padHeight ;
	int64_t y = i/strideHeight;
	if ( y*strideHeight != i || y < 0 || gOutHeight <= y)  continue ;

	for (int64_t s=0; s<kernWidth; s++) {
	  __vr vrj = _ve_vaddsl_vsv(padWidth-s*dilationWidth, vrw) ;
	  __vr vrx = _ve_vdivsl_vvs(vrj, strideWidth) ;

	  __vm256 vmx0 = _ve_vfmkl_mcv(VECC_IEQ, _ve_vcmpsl_vvv(vrj, _ve_vmulsl_vsv(strideWidth, vrx))) ;
	  __vm256 vmx1 = _ve_vfmkl_mcv(VECC_GE, vrx) ;
	  __vm256 vmx2 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(gOutWidth,vrx)) ;

	  __vm256 vmx = _ve_andm_mmm(_ve_andm_mmm(vmx0, vmx1), vmx2) ;

	  for (int64_t k=0; k<gOutChannelGroup; k++) {
	    int64_t gOutIndex    = gOutGroupOffset + ((n * gOutChannel + k) * gOutHeight) * gOutWidth ;
	    __vr vrgout_ptr = _ve_vsfa_vvss(_ve_vaddsl_vsv(gOutWidth*y, vrx),
					  2,
					  (unsigned long)(pGOut+gOutIndex)) ;

	    __vr vrgout = _ve_vgtu_vvm(vrgout_ptr, vmx) ;
	    vrgout = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), vrgout, vmx) ;

	    __vr vrgoutP = _ve_vshf_vvvs(vrgout, vrgout, VE_VSHUFFLE_YUZU) ;

	    const float *pKerValue = pKernel + kernGroupOffset + ((k * gInChannelGroup + c) * kernHeight + r) * kernWidth + s;

	    const uint64_t kerValue01 = _ve_pack_f32p(pKerValue,
						      pKerValue +     kernHeight * kernWidth ) ;

	    vrsum01 = _ve_pvfmad_vvsv(vrsum01, kerValue01, vrgoutP) ;
	  } // gOutChannel

	} // kernWidth
      } // kernHeight

      _ve_vstu_vss(vrsum01, 4, pGIn+gInIndex) ;
      _ve_vstl_vss(vrsum01, 4, pGIn+gInIndex+  gInPixels) ;

    } // gInWidth
  } // gInHeight
}


static inline void c4(
    const float * restrict pGOut,
    const float * restrict pKernel,
    float * restrict const pGIn,
    const int64_t gOutChannel,
    const int64_t gOutWidth,
    const int64_t gOutHeight,
    const int64_t gInChannel,
    const int64_t gInWidth,
    const int64_t gInHeight,
    const int64_t kernWidth,
    const int64_t kernHeight,
    const int64_t strideWidth,
    const int64_t strideHeight,
    const int64_t padWidth,
    const int64_t padHeight,
    const int64_t dilationWidth,
    const int64_t dilationHeight,
    const int64_t gInChannelGroup,
    const int64_t gOutChannelGroup,
    const int64_t gInGroupOffset,
    const int64_t gOutGroupOffset,
    const int64_t kernGroupOffset,
    const int64_t gInPixels,
    const int64_t n,
    const int64_t c
)
{
  for (int64_t h = 0; h < gInHeight ; h++ ) {
    for (int64_t w = 0; w < gInWidth ; w += VLEN ) {
      const int64_t vl = gInWidth - w < VLEN ? gInWidth - w  : VLEN ;

      const int64_t gInIndex = gInGroupOffset + ((n * gInChannel + c) * gInHeight + h) * gInWidth + w ;

      _ve_lvl(vl) ;

      __vr vrsum01 = _ve_vbrd_vs_i64(0UL) ;
      __vr vrsum23 = _ve_vbrd_vs_i64(0UL) ;

      __vr vrw   = _ve_vaddsl_vsv(w, _ve_vseq_v()) ;

      for (int64_t r=0; r<kernHeight; r++) {
	int64_t i = h - r * dilationHeight + padHeight ;
	int64_t y = i/strideHeight;
	if ( y*strideHeight != i || y < 0 || gOutHeight <= y)  continue ;

	for (int64_t s=0; s<kernWidth; s++) {
	  __vr vrj = _ve_vaddsl_vsv(padWidth-s*dilationWidth, vrw) ;
	  __vr vrx = _ve_vdivsl_vvs(vrj, strideWidth) ;

	  __vm256 vmx0 = _ve_vfmkl_mcv(VECC_IEQ, _ve_vcmpsl_vvv(vrj, _ve_vmulsl_vsv(strideWidth, vrx))) ;
	  __vm256 vmx1 = _ve_vfmkl_mcv(VECC_GE, vrx) ;
	  __vm256 vmx2 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(gOutWidth,vrx)) ;

	  __vm256 vmx = _ve_andm_mmm(_ve_andm_mmm(vmx0, vmx1), vmx2) ;

	  for (int64_t k=0; k<gOutChannelGroup; k++) {
	    int64_t gOutIndex    = gOutGroupOffset + ((n * gOutChannel + k) * gOutHeight) * gOutWidth ;
	    __vr vrgout_ptr = _ve_vsfa_vvss(_ve_vaddsl_vsv(gOutWidth*y, vrx),
					  2,
					  (unsigned long)(pGOut+gOutIndex)) ;

	    __vr vrgout = _ve_vgtu_vvm(vrgout_ptr, vmx) ;
	    vrgout = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), vrgout, vmx) ;

	    __vr vrgoutP = _ve_vshf_vvvs(vrgout, vrgout, VE_VSHUFFLE_YUZU) ;

	    const float *pKerValue = pKernel + kernGroupOffset + ((k * gInChannelGroup + c) * kernHeight + r) * kernWidth + s;

	    const uint64_t kerValue01 = _ve_pack_f32p(pKerValue,
						      pKerValue +     kernHeight * kernWidth ) ;
	    const uint64_t kerValue23 = _ve_pack_f32p(pKerValue + 2 * kernHeight * kernWidth,
						      pKerValue + 3 * kernHeight * kernWidth ) ;

	    vrsum01 = _ve_pvfmad_vvsv(vrsum01, kerValue01, vrgoutP) ;
	    vrsum23 = _ve_pvfmad_vvsv(vrsum23, kerValue23, vrgoutP) ;
	  } // gOutChannel

	} // kernWidth
      } // kernHeight

      _ve_vstu_vss(vrsum01, 4, pGIn+gInIndex) ;
      _ve_vstl_vss(vrsum01, 4, pGIn+gInIndex+  gInPixels) ;
      _ve_vstu_vss(vrsum23, 4, pGIn+gInIndex+2*gInPixels) ;
      _ve_vstl_vss(vrsum23, 4, pGIn+gInIndex+3*gInPixels) ;

    } // gInWidth
  } // gInHeight
}


static inline void c8(
    const float * restrict pGOut,
    const float * restrict pKernel,
    float * restrict const pGIn,
    const int64_t gOutChannel,
    const int64_t gOutWidth,
    const int64_t gOutHeight,
    const int64_t gInChannel,
    const int64_t gInWidth,
    const int64_t gInHeight,
    const int64_t kernWidth,
    const int64_t kernHeight,
    const int64_t strideWidth,
    const int64_t strideHeight,
    const int64_t padWidth,
    const int64_t padHeight,
    const int64_t dilationWidth,
    const int64_t dilationHeight,
    const int64_t gInChannelGroup,
    const int64_t gOutChannelGroup,
    const int64_t gInGroupOffset,
    const int64_t gOutGroupOffset,
    const int64_t kernGroupOffset,
    const int64_t gInPixels,
    const int64_t n,
    const int64_t c
)
{
  for (int64_t h = 0; h < gInHeight ; h++ ) {
    for (int64_t w = 0; w < gInWidth ; w += VLEN ) {
      const int64_t vl = gInWidth - w < VLEN ? gInWidth - w  : VLEN ;

      const int64_t gInIndex = gInGroupOffset + ((n * gInChannel + c) * gInHeight + h) * gInWidth + w ;

      _ve_lvl(vl) ;

      __vr vrsum01 = _ve_vbrd_vs_i64(0UL) ;
      __vr vrsum23 = _ve_vbrd_vs_i64(0UL) ;
      __vr vrsum45 = _ve_vbrd_vs_i64(0UL) ;
      __vr vrsum67 = _ve_vbrd_vs_i64(0UL) ;

      __vr vrw   = _ve_vaddsl_vsv(w, _ve_vseq_v()) ;

      for (int64_t r=0; r<kernHeight; r++) {
	int64_t i = h - r * dilationHeight + padHeight ;
	int64_t y = i/strideHeight;
	if ( y*strideHeight != i || y < 0 || gOutHeight <= y)  continue ;

	for (int64_t s=0; s<kernWidth; s++) {
	  __vr vrj = _ve_vaddsl_vsv(padWidth-s*dilationWidth, vrw) ;
	  __vr vrx = _ve_vdivsl_vvs(vrj, strideWidth) ;

	  __vm256 vmx0 = _ve_vfmkl_mcv(VECC_IEQ, _ve_vcmpsl_vvv(vrj, _ve_vmulsl_vsv(strideWidth, vrx))) ;
	  __vm256 vmx1 = _ve_vfmkl_mcv(VECC_GE, vrx) ;
	  __vm256 vmx2 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(gOutWidth,vrx)) ;

	  __vm256 vmx = _ve_andm_mmm(_ve_andm_mmm(vmx0, vmx1), vmx2) ;

	  for (int64_t k=0; k<gOutChannelGroup; k++) {
	    int64_t gOutIndex    = gOutGroupOffset + ((n * gOutChannel + k) * gOutHeight) * gOutWidth ;
	    __vr vrgout_ptr = _ve_vsfa_vvss(_ve_vaddsl_vsv(gOutWidth*y, vrx),
					  2,
					  (unsigned long)(pGOut+gOutIndex)) ;

	    __vr vrgout = _ve_vgtu_vvm(vrgout_ptr, vmx) ;
	    vrgout = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), vrgout, vmx) ;

	    __vr vrgoutP = _ve_vshf_vvvs(vrgout, vrgout, VE_VSHUFFLE_YUZU) ;

	    const float *pKerValue = pKernel + kernGroupOffset + ((k * gInChannelGroup + c) * kernHeight + r) * kernWidth + s;

	    const uint64_t kerValue01 = _ve_pack_f32p(pKerValue,
						      pKerValue +     kernHeight * kernWidth ) ;
	    const uint64_t kerValue23 = _ve_pack_f32p(pKerValue + 2 * kernHeight * kernWidth,
						      pKerValue + 3 * kernHeight * kernWidth ) ;
	    const uint64_t kerValue45 = _ve_pack_f32p(pKerValue + 4 * kernHeight * kernWidth,
						      pKerValue + 5 * kernHeight * kernWidth ) ;
	    const uint64_t kerValue67 = _ve_pack_f32p(pKerValue + 6 * kernHeight * kernWidth,
						      pKerValue + 7 * kernHeight * kernWidth ) ;

	    vrsum01 = _ve_pvfmad_vvsv(vrsum01, kerValue01, vrgoutP) ;
	    vrsum23 = _ve_pvfmad_vvsv(vrsum23, kerValue23, vrgoutP) ;
	    vrsum45 = _ve_pvfmad_vvsv(vrsum45, kerValue45, vrgoutP) ;
	    vrsum67 = _ve_pvfmad_vvsv(vrsum67, kerValue67, vrgoutP) ;
	  } // gOutChannel

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

    } // gInWidth
  } // gInHeight
}


static inline void c16(
    const float * restrict pGOut,
    const float * restrict pKernel,
    float * restrict const pGIn,
    const int64_t gOutChannel,
    const int64_t gOutWidth,
    const int64_t gOutHeight,
    const int64_t gInChannel,
    const int64_t gInWidth,
    const int64_t gInHeight,
    const int64_t kernWidth,
    const int64_t kernHeight,
    const int64_t strideWidth,
    const int64_t strideHeight,
    const int64_t padWidth,
    const int64_t padHeight,
    const int64_t dilationWidth,
    const int64_t dilationHeight,
    const int64_t gInChannelGroup,
    const int64_t gOutChannelGroup,
    const int64_t gInGroupOffset,
    const int64_t gOutGroupOffset,
    const int64_t kernGroupOffset,
    const int64_t gInPixels,
    const int64_t n,
    const int64_t c
)
{
  for (int64_t h = 0; h < gInHeight ; h++ ) {
    for (int64_t w = 0; w < gInWidth ; w += VLEN ) {
      const int64_t vl = gInWidth - w < VLEN ? gInWidth - w  : VLEN ;

      const int64_t gInIndex = gInGroupOffset + ((n * gInChannel + c) * gInHeight + h) * gInWidth + w ;

      _ve_lvl(vl) ;

      __vr vrsum01 = _ve_vbrd_vs_i64(0UL) ;
      __vr vrsum23 = _ve_vbrd_vs_i64(0UL) ;
      __vr vrsum45 = _ve_vbrd_vs_i64(0UL) ;
      __vr vrsum67 = _ve_vbrd_vs_i64(0UL) ;
      __vr vrsum89 = _ve_vbrd_vs_i64(0UL) ;
      __vr vrsumAB = _ve_vbrd_vs_i64(0UL) ;
      __vr vrsumCD = _ve_vbrd_vs_i64(0UL) ;
      __vr vrsumEF = _ve_vbrd_vs_i64(0UL) ;

      __vr vrw   = _ve_vaddsl_vsv(w, _ve_vseq_v()) ;

      for (int64_t r=0; r<kernHeight; r++) {
	int64_t i = h - r * dilationHeight + padHeight ;
	int64_t y = i/strideHeight;
	if ( y*strideHeight != i || y < 0 || gOutHeight <= y)  continue ;

	for (int64_t s=0; s<kernWidth; s++) {
	  __vr vrj = _ve_vaddsl_vsv(padWidth-s*dilationWidth, vrw) ;
	  __vr vrx = _ve_vdivsl_vvs(vrj, strideWidth) ;

	  __vm256 vmx0 = _ve_vfmkl_mcv(VECC_IEQ, _ve_vcmpsl_vvv(vrj, _ve_vmulsl_vsv(strideWidth, vrx))) ;
	  __vm256 vmx1 = _ve_vfmkl_mcv(VECC_GE, vrx) ;
	  __vm256 vmx2 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(gOutWidth,vrx)) ;

	  __vm256 vmx = _ve_andm_mmm(_ve_andm_mmm(vmx0, vmx1), vmx2) ;

	  for (int64_t k=0; k<gOutChannelGroup; k++) {
	    int64_t gOutIndex    = gOutGroupOffset + ((n * gOutChannel + k) * gOutHeight) * gOutWidth ;
	    __vr vrgout_ptr = _ve_vsfa_vvss(_ve_vaddsl_vsv(gOutWidth*y, vrx),
					  2,
					  (unsigned long)(pGOut+gOutIndex)) ;

	    __vr vrgout = _ve_vgtu_vvm(vrgout_ptr, vmx) ;
	    vrgout = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), vrgout, vmx) ;

	    __vr vrgoutP = _ve_vshf_vvvs(vrgout, vrgout, VE_VSHUFFLE_YUZU) ;

	    const float *pKerValue = pKernel + kernGroupOffset + ((k * gInChannelGroup + c) * kernHeight + r) * kernWidth + s;

	    const uint64_t kerValue01 = _ve_pack_f32p(pKerValue,
						      pKerValue +     kernHeight * kernWidth ) ;
	    const uint64_t kerValue23 = _ve_pack_f32p(pKerValue + 2 * kernHeight * kernWidth,
						      pKerValue + 3 * kernHeight * kernWidth ) ;
	    const uint64_t kerValue45 = _ve_pack_f32p(pKerValue + 4 * kernHeight * kernWidth,
						      pKerValue + 5 * kernHeight * kernWidth ) ;
	    const uint64_t kerValue67 = _ve_pack_f32p(pKerValue + 6 * kernHeight * kernWidth,
						      pKerValue + 7 * kernHeight * kernWidth ) ;
	    const uint64_t kerValue89 = _ve_pack_f32p(pKerValue + 8 * kernHeight * kernWidth,
						      pKerValue + 9 * kernHeight * kernWidth ) ;
	    const uint64_t kerValueAB = _ve_pack_f32p(pKerValue +10 * kernHeight * kernWidth,
						      pKerValue +11 * kernHeight * kernWidth ) ;
	    const uint64_t kerValueCD = _ve_pack_f32p(pKerValue +12 * kernHeight * kernWidth,
						      pKerValue +13 * kernHeight * kernWidth ) ;
	    const uint64_t kerValueEF = _ve_pack_f32p(pKerValue +14 * kernHeight * kernWidth,
						      pKerValue +15 * kernHeight * kernWidth ) ;

	    vrsum01 = _ve_pvfmad_vvsv(vrsum01, kerValue01, vrgoutP) ;
	    vrsum23 = _ve_pvfmad_vvsv(vrsum23, kerValue23, vrgoutP) ;
	    vrsum45 = _ve_pvfmad_vvsv(vrsum45, kerValue45, vrgoutP) ;
	    vrsum67 = _ve_pvfmad_vvsv(vrsum67, kerValue67, vrgoutP) ;
	    vrsum89 = _ve_pvfmad_vvsv(vrsum89, kerValue89, vrgoutP) ;
	    vrsumAB = _ve_pvfmad_vvsv(vrsumAB, kerValueAB, vrgoutP) ;
	    vrsumCD = _ve_pvfmad_vvsv(vrsumCD, kerValueCD, vrgoutP) ;
	    vrsumEF = _ve_pvfmad_vvsv(vrsumEF, kerValueEF, vrgoutP) ;
	  } // gOutChannel

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
      _ve_vstu_vss(vrsum89, 4, pGIn+gInIndex+8*gInPixels) ;
      _ve_vstl_vss(vrsum89, 4, pGIn+gInIndex+9*gInPixels) ;
      _ve_vstu_vss(vrsumAB, 4, pGIn+gInIndex+10*gInPixels) ;
      _ve_vstl_vss(vrsumAB, 4, pGIn+gInIndex+11*gInPixels) ;
      _ve_vstu_vss(vrsumCD, 4, pGIn+gInIndex+12*gInPixels) ;
      _ve_vstl_vss(vrsumCD, 4, pGIn+gInIndex+13*gInPixels) ;
      _ve_vstu_vss(vrsumEF, 4, pGIn+gInIndex+14*gInPixels) ;
      _ve_vstl_vss(vrsumEF, 4, pGIn+gInIndex+15*gInPixels) ;

    } // gInWidth
  } // gInHeight
}


vednnError_t
vednnConvolutionBackwardData_direct_default(
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
  const int64_t kernWidth   = pParamKernel->width;
  const int64_t kernHeight  = pParamKernel->height;

  const int64_t group          = pParamConv->group;
  const int64_t strideWidth    = pParamConv->strideWidth;;
  const int64_t strideHeight   = pParamConv->strideHeight;
  const int64_t padWidth       = pParamConv->padWidth;
  const int64_t padHeight      = pParamConv->padHeight;
  const int64_t dilationWidth  = pParamConv->dilationWidth;
  const int64_t dilationHeight = pParamConv->dilationHeight;

  const int64_t gOutChannelGroup = gOutChannel  / group;
  const int64_t gInChannelGroup  = gInChannel / group;

  const float * restrict pGOut   = pDataGradOut;
  const float * restrict pKernel = pDataKernel;
  float * restrict const pGIn   = pDataGradIn;

  const int64_t gInPixels= gInHeight*gInWidth ;

  /* intrinsic version 1 */
  {

    for (int64_t n=0; n<batch; n++) {
      for (int64_t g = 0; g < group; g++) {

	int64_t gInGroupOffset  = g * gInChannelGroup * gInHeight * gInWidth;
	int64_t gOutGroupOffset = g * gOutChannelGroup * gOutHeight * gOutWidth;
	int64_t kernGroupOffset = g * gOutChannelGroup * gInChannelGroup * kernHeight * kernWidth;

	int64_t c=0;
	if( (gInChannelGroup & 0x01 ) == 1 ) {

	  c1(pGOut, pKernel, pGIn,
	     gOutChannel, gOutWidth, gOutHeight,
             gInChannel, gInWidth, gInHeight,
             kernWidth, kernHeight,
             strideWidth, strideHeight,
	     padWidth, padHeight,
	     dilationWidth, dilationHeight,
	     gInChannelGroup, gOutChannelGroup,
	     gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	     gInPixels, n, c ) ;

	  c+=1 ;
	}
	if( ((gInChannelGroup>>1) & 0x01 ) == 1 ) {

	  c2(pGOut, pKernel, pGIn,
	     gOutChannel, gOutWidth, gOutHeight,
             gInChannel, gInWidth, gInHeight,
             kernWidth, kernHeight,
             strideWidth, strideHeight,
	     padWidth, padHeight,
	     dilationWidth, dilationHeight,
	     gInChannelGroup, gOutChannelGroup,
	     gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	     gInPixels, n, c ) ;

	  c+=2 ;
	}
	if( ((gInChannelGroup>>2) & 0x01 ) == 1 ) {

	  c4(pGOut, pKernel, pGIn,
	     gOutChannel, gOutWidth, gOutHeight,
             gInChannel, gInWidth, gInHeight,
             kernWidth, kernHeight,
             strideWidth, strideHeight,
	     padWidth, padHeight,
	     dilationWidth, dilationHeight,
	     gInChannelGroup, gOutChannelGroup,
	     gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	     gInPixels, n, c ) ;

	  c+=4 ;
	}
	if( ((gInChannelGroup>>3) & 0x01 ) == 1 ) {

	  c8(pGOut, pKernel, pGIn,
	     gOutChannel, gOutWidth, gOutHeight,
             gInChannel, gInWidth, gInHeight,
             kernWidth, kernHeight,
             strideWidth, strideHeight,
	     padWidth, padHeight,
	     dilationWidth, dilationHeight,
	     gInChannelGroup, gOutChannelGroup,
	     gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	     gInPixels, n, c ) ;

	  c+=8 ;
	}
	for (; c<gInChannelGroup; ) {
	  c16(pGOut, pKernel, pGIn,
	     gOutChannel, gOutWidth, gOutHeight,
             gInChannel, gInWidth, gInHeight,
             kernWidth, kernHeight,
             strideWidth, strideHeight,
	     padWidth, padHeight,
	     dilationWidth, dilationHeight,
	     gInChannelGroup, gOutChannelGroup,
	     gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	     gInPixels, n, c) ;

	  c+= 16 ;

	} // gInChannel
      } // group
    } // batch
  }


  return VEDNN_SUCCESS;
}

#if 0	// reference version
vednnError_t
vednnConvolutionBackwardData_direct_default2(
    const vednnTensorParam_t * restrict 	pParamGradOut,
    const void * restrict 			pDataGradOut,
    const vednnFilterParam_t * restrict 	pParamKernel,
    const void * restrict 			pDataKernel,
    const vednnConvolutionParam_t * restrict 	pParamConv,
    const vednnTensorParam_t * restrict 	pParamGradIn,
    void * restrict 				pDataGradIn
)
{
  const int64_t batch      = pParamGradOut->batch;
  const int64_t gOutChannel= pParamGradOut->channel;
  const int64_t gOutWidth  = pParamGradOut->width;
  const int64_t gOutHeight = pParamGradOut->height;
  const int64_t gInChannel = pParamGradIn->channel;
  const int64_t gInWidth   = pParamGradIn->width;
  const int64_t gInHeight  = pParamGradIn->height;
  const int64_t kernWidth  = pParamKernel->width;
  const int64_t kernHeight = pParamKernel->height;

  const int64_t group          = pParamConv->group;
  const int64_t strideWidth    = pParamConv->strideWidth;;
  const int64_t strideHeight   = pParamConv->strideHeight;
  const int64_t padWidth       = pParamConv->padWidth;
  const int64_t padHeight      = pParamConv->padHeight;
  const int64_t dilationWidth  = pParamConv->dilationWidth;
  const int64_t dilationHeight = pParamConv->dilationHeight;

  const int64_t gOutChannelGroup = gOutChannel  / group;
  const int64_t gInChannelGroup  = gInChannel / group;

  const float * restrict pGOut   = pDataGradOut;
  const float * restrict pKernel = pDataKernel;
  float * restrict const pGIn   = pDataGradIn;

  const int oPixels= gOutHeight*gOutWidth ;
#if 0
  /* version 1 : base version */
  {

    for (int64_t n=0; n<batch; n++) {
      for (int64_t g = 0; g < group; g++) {
	int64_t gInGroupOffset  = g * gInChannelGroup * gInHeight * gInWidth;
	int64_t gOutGroupOffset = g * gOutChannelGroup * gOutHeight  * gOutWidth;
	int64_t kernGroupOffset = g * gOutChannelGroup * gInChannelGroup * kernHeight * kernWidth;


	for (int64_t c=0; c<gInChannelGroup; c++) {
	  for (int64_t h=0; h<gInHeight; h++) {
	    for (int64_t w=0; w<gInWidth; w++) {
	      int64_t gInIndex = gInGroupOffset + ((n * gInChannel + c) * gInHeight + h) * gInWidth + w;

	      float sum = 0.0f ;

	      for (int64_t r=0; r<kernHeight; r++) {
		int64_t i = h - r * dilationHeight + padHeight ;
		int64_t y = i/strideHeight;
		if ( y*strideHeight != i || y < 0 || gOutHeight <= y)  continue ;

		for (int64_t s=0; s<kernWidth; s++) {
		  int64_t j = w - s * dilationWidth  + padWidth ;
		  int64_t x = j/strideWidth ;
		  if (x*strideWidth !=j || x < 0 || gOutWidth <= x) continue;

		  for (int64_t k=0; k<gOutChannelGroup; k++) {
		    int64_t gOutIndex   = gOutGroupOffset + ((n * gOutChannel + k) * gOutHeight + y) * gOutWidth + x;
		    int64_t kernelIndex = kernGroupOffset + ((k * gInChannelGroup + c) * kernHeight + r) * kernWidth + s;
		    sum += (pGOut[gOutIndex] * pKernel[kernelIndex]);
		  } // gOutChannel

		} // kernWidth
	      } // kernHeight

	      pGIn[gInIndex] = sum ;

	    } // gInWidth
	  } // gInHeight
	} // gInChannel
      } // group
    } // batch
  }
#else
  /* version 0 : generated from forward propagation */
  {
    float * restrict pIn     = (float * restrict) pDataGradIn ;
    float * restrict pKernel = (float * restrict) pDataKernel;
    float * restrict pOut    = (float * restrict) pDataGradOut;

    const int64_t outChannel = gOutChannel ;
    const int64_t inChannel  = gInChannel ;

    const int64_t inChannelGroup  = gInChannelGroup ;
    const int64_t outChannelGroup = gOutChannelGroup ;

    const int64_t inHeight = gInHeight ;
    const int64_t inWidth  = gInWidth ;

    const int64_t outHeight = gOutHeight ;
    const int64_t outWidth  = gOutWidth ;

    for(int64_t i=0; i<inChannel*inHeight*inWidth; i++) pIn[i] = 0.0f ;

    for (int64_t n=0; n<batch; n++) {
      for (int64_t g = 0; g < group; g++) {
	int64_t inGroupOffset   = g * inChannelGroup  * inHeight  * inWidth;
	int64_t outGroupOffset  = g * outChannelGroup * outHeight * outWidth;
	int64_t kernGroupOffset = g * outChannelGroup * inChannelGroup * kernHeight * kernWidth;

	for (int64_t k=0; k<outChannelGroup; k++) {
	  for (int64_t p=0; p<outHeight; p++) {
	    int64_t i = p * strideHeight - padHeight;
	    for (int64_t q=0; q<outWidth; q++) {
	      int64_t j = q * strideWidth - padWidth;
	      int64_t outIndex  = outGroupOffset + ((n * outChannel + k) * outHeight + p) * outWidth + q;
	      for (int64_t c=0; c<inChannelGroup; c++) {
		for (int64_t h=0; h<kernHeight; h++) {
		  for (int64_t w=0; w<kernWidth; w++) {
		    int64_t y = i + h * dilationHeight;
		    int64_t x = j + w * dilationWidth;
		    if (y < 0 || inHeight <= y) {
		      continue;
		    }
		    if (x < 0 || inWidth <= x) {
		      continue;
		    }
		    int64_t inputIndex  = inGroupOffset + ((n * inChannel + c) * inHeight + y) * inWidth + x;
		    int64_t kernelIndex = kernGroupOffset + ((k * inChannelGroup + c) * kernHeight + h) * kernWidth + w;
		    pIn[inputIndex] += (pOut[outIndex] * pKernel[kernelIndex]);
		  } // kernWidth
		} // kernHeight
	      } // inChannel
	    } // outWidth
	  } // outHeight
	} // outChannel
      } // group
    } // batch
  }
#endif

  return VEDNN_SUCCESS;
}
#endif
