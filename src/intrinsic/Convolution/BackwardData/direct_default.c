#include <stdint.h>
#include <stdlib.h>

#include "vednn.h"

#include "velintrin.h"
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

      __vr vrsum = _vel_vbrds_vsl(0.f, vl) ;

      __vr vrw   = _vel_vaddsl_vsvl(w, _vel_vseq_vl(vl), vl) ;

      for (int64_t r=0; r<kernHeight; r++) {
	int64_t i = h - r * dilationHeight + padHeight ;
	int64_t y = i/strideHeight;
	if ( y*strideHeight != i || y < 0 || gOutHeight <= y)  continue ;

	for (int64_t s=0; s<kernWidth; s++) {
	  __vr vrj = _vel_vaddsl_vsvl(padWidth-s*dilationWidth, vrw, vl) ;
	  __vr vrx = _vel_vdivsl_vvsl(vrj, strideWidth, vl) ;

	  __vm256 vmx0 =  _vel_vfmkleq_mvl(_vel_vcmpsl_vvvl(vrj, _vel_vmulsl_vsvl(strideWidth, vrx, vl), vl), vl) ;
	  __vm256 vmx1 =  _vel_vfmklge_mvl(vrx, vl) ;
	  __vm256 vmx2 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutWidth,vrx, vl), vl) ;

	  __vm256 vmx = _vel_andm_mmm(_vel_andm_mmm(vmx0, vmx1), vmx2) ;

	  for (int64_t k=0; k<gOutChannelGroup; k++) {
	    int64_t gOutIndex    = gOutGroupOffset + ((n * gOutChannel + k) * gOutHeight) * gOutWidth ;
	    __vr vrgout_ptr = _vel_vsfa_vvssl(_vel_vaddsl_vsvl(gOutWidth*y, vrx, vl),
					      2,
					      (unsigned long)(pGOut+gOutIndex), vl) ;

	    __vr vrgout = _vel_vgtu_vvssml(vrgout_ptr, 0, 0, vmx, vl) ;
	    vrgout = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrgout, vmx, vl) ;


	    const float *pKerValue = pKernel + kernGroupOffset + ((k * gInChannelGroup + c) * kernHeight + r) * kernWidth + s;

	    vrsum = _vel_vfmads_vvsvl(vrsum, *pKerValue, vrgout, vl) ;
	  } // gOutChannel

	} // kernWidth
      } // kernHeight

      _vel_vstu_vssl(vrsum, 4, pGIn+gInIndex, vl) ;

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

      __vr vrsum01 = _vel_vbrdl_vsl(0UL, vl) ;

      __vr vrw   = _vel_vaddsl_vsvl(w, _vel_vseq_vl(vl), vl) ;

      for (int64_t r=0; r<kernHeight; r++) {
	int64_t i = h - r * dilationHeight + padHeight ;
	int64_t y = i/strideHeight;
	if ( y*strideHeight != i || y < 0 || gOutHeight <= y)  continue ;

	for (int64_t s=0; s<kernWidth; s++) {
	  __vr vrj = _vel_vaddsl_vsvl(padWidth-s*dilationWidth, vrw, vl) ;
	  __vr vrx = _vel_vdivsl_vvsl(vrj, strideWidth, vl) ;

	  __vm256 vmx0 =  _vel_vfmkleq_mvl(_vel_vcmpsl_vvvl(vrj, _vel_vmulsl_vsvl(strideWidth, vrx, vl), vl), vl) ;
	  __vm256 vmx1 =  _vel_vfmklge_mvl(vrx, vl) ;
	  __vm256 vmx2 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutWidth,vrx, vl), vl) ;

	  __vm256 vmx = _vel_andm_mmm(_vel_andm_mmm(vmx0, vmx1), vmx2) ;

	  for (int64_t k=0; k<gOutChannelGroup; k++) {
	    int64_t gOutIndex    = gOutGroupOffset + ((n * gOutChannel + k) * gOutHeight) * gOutWidth ;
	    __vr vrgout_ptr = _vel_vsfa_vvssl(_vel_vaddsl_vsvl(gOutWidth*y, vrx, vl),
					      2,
					      (unsigned long)(pGOut+gOutIndex), vl) ;

	    __vr vrgout = _vel_vgtu_vvssml(vrgout_ptr, 0, 0, vmx, vl) ;
	    vrgout = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrgout, vmx, vl) ;

	    __vr vrgoutP = _vel_vshf_vvvsl(vrgout, vrgout, VE_VSHUFFLE_YUZU, vl) ;

	    const float *pKerValue = pKernel + kernGroupOffset + ((k * gInChannelGroup + c) * kernHeight + r) * kernWidth + s;

	    const uint64_t kerValue01 = _vel_pack_f32p(pKerValue,
						       pKerValue +     kernHeight * kernWidth ) ;

	    vrsum01 = _vel_pvfmad_vvsvl(vrsum01, kerValue01, vrgoutP, vl) ;
	  } // gOutChannel

	} // kernWidth
      } // kernHeight

      _vel_vstu_vssl(vrsum01, 4, pGIn+gInIndex, vl) ;
      _vel_vstl_vssl(vrsum01, 4, pGIn+gInIndex+  gInPixels, vl) ;

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

      __vr vrsum01 = _vel_vbrdl_vsl(0UL, vl) ;
      __vr vrsum23 = _vel_vbrdl_vsl(0UL, vl) ;

      __vr vrw   = _vel_vaddsl_vsvl(w, _vel_vseq_vl(vl), vl) ;

      for (int64_t r=0; r<kernHeight; r++) {
	int64_t i = h - r * dilationHeight + padHeight ;
	int64_t y = i/strideHeight;
	if ( y*strideHeight != i || y < 0 || gOutHeight <= y)  continue ;

	for (int64_t s=0; s<kernWidth; s++) {
	  __vr vrj = _vel_vaddsl_vsvl(padWidth-s*dilationWidth, vrw, vl) ;
	  __vr vrx = _vel_vdivsl_vvsl(vrj, strideWidth, vl) ;

	  __vm256 vmx0 =  _vel_vfmkleq_mvl(_vel_vcmpsl_vvvl(vrj, _vel_vmulsl_vsvl(strideWidth, vrx, vl), vl), vl) ;
	  __vm256 vmx1 =  _vel_vfmklge_mvl(vrx, vl) ;
	  __vm256 vmx2 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutWidth,vrx, vl), vl) ;

	  __vm256 vmx = _vel_andm_mmm(_vel_andm_mmm(vmx0, vmx1), vmx2) ;

	  for (int64_t k=0; k<gOutChannelGroup; k++) {
	    int64_t gOutIndex    = gOutGroupOffset + ((n * gOutChannel + k) * gOutHeight) * gOutWidth ;
	    __vr vrgout_ptr = _vel_vsfa_vvssl(_vel_vaddsl_vsvl(gOutWidth*y, vrx, vl),
					      2,
					      (unsigned long)(pGOut+gOutIndex), vl) ;

	    __vr vrgout = _vel_vgtu_vvssml(vrgout_ptr, 0, 0, vmx, vl) ;
	    vrgout = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrgout, vmx, vl) ;

	    __vr vrgoutP = _vel_vshf_vvvsl(vrgout, vrgout, VE_VSHUFFLE_YUZU, vl) ;

	    const float *pKerValue = pKernel + kernGroupOffset + ((k * gInChannelGroup + c) * kernHeight + r) * kernWidth + s;

	    const uint64_t kerValue01 = _vel_pack_f32p(pKerValue,
						       pKerValue +     kernHeight * kernWidth ) ;
	    const uint64_t kerValue23 = _vel_pack_f32p(pKerValue + 2 * kernHeight * kernWidth,
						       pKerValue + 3 * kernHeight * kernWidth ) ;

	    vrsum01 = _vel_pvfmad_vvsvl(vrsum01, kerValue01, vrgoutP, vl) ;
	    vrsum23 = _vel_pvfmad_vvsvl(vrsum23, kerValue23, vrgoutP, vl) ;
	  } // gOutChannel

	} // kernWidth
      } // kernHeight

      _vel_vstu_vssl(vrsum01, 4, pGIn+gInIndex, vl) ;
      _vel_vstl_vssl(vrsum01, 4, pGIn+gInIndex+  gInPixels, vl) ;
      _vel_vstu_vssl(vrsum23, 4, pGIn+gInIndex+2*gInPixels, vl) ;
      _vel_vstl_vssl(vrsum23, 4, pGIn+gInIndex+3*gInPixels, vl) ;

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

      __vr vrsum01 = _vel_vbrdl_vsl(0UL, vl) ;
      __vr vrsum23 = _vel_vbrdl_vsl(0UL, vl) ;
      __vr vrsum45 = _vel_vbrdl_vsl(0UL, vl) ;
      __vr vrsum67 = _vel_vbrdl_vsl(0UL, vl) ;

      __vr vrw   = _vel_vaddsl_vsvl(w, _vel_vseq_vl(vl), vl) ;

      for (int64_t r=0; r<kernHeight; r++) {
	int64_t i = h - r * dilationHeight + padHeight ;
	int64_t y = i/strideHeight;
	if ( y*strideHeight != i || y < 0 || gOutHeight <= y)  continue ;

	for (int64_t s=0; s<kernWidth; s++) {
	  __vr vrj = _vel_vaddsl_vsvl(padWidth-s*dilationWidth, vrw, vl) ;
	  __vr vrx = _vel_vdivsl_vvsl(vrj, strideWidth, vl) ;

	  __vm256 vmx0 =  _vel_vfmkleq_mvl(_vel_vcmpsl_vvvl(vrj, _vel_vmulsl_vsvl(strideWidth, vrx, vl), vl), vl) ;
	  __vm256 vmx1 =  _vel_vfmklge_mvl(vrx, vl) ;
	  __vm256 vmx2 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutWidth,vrx, vl), vl) ;

	  __vm256 vmx = _vel_andm_mmm(_vel_andm_mmm(vmx0, vmx1), vmx2) ;

	  for (int64_t k=0; k<gOutChannelGroup; k++) {
	    int64_t gOutIndex    = gOutGroupOffset + ((n * gOutChannel + k) * gOutHeight) * gOutWidth ;
	    __vr vrgout_ptr = _vel_vsfa_vvssl(_vel_vaddsl_vsvl(gOutWidth*y, vrx, vl),
					      2,
					      (unsigned long)(pGOut+gOutIndex), vl) ;

	    __vr vrgout = _vel_vgtu_vvssml(vrgout_ptr, 0, 0, vmx, vl) ;
	    vrgout = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrgout, vmx, vl) ;

	    __vr vrgoutP = _vel_vshf_vvvsl(vrgout, vrgout, VE_VSHUFFLE_YUZU, vl) ;

	    const float *pKerValue = pKernel + kernGroupOffset + ((k * gInChannelGroup + c) * kernHeight + r) * kernWidth + s;

	    const uint64_t kerValue01 = _vel_pack_f32p(pKerValue,
						       pKerValue +     kernHeight * kernWidth ) ;
	    const uint64_t kerValue23 = _vel_pack_f32p(pKerValue + 2 * kernHeight * kernWidth,
						       pKerValue + 3 * kernHeight * kernWidth ) ;
	    const uint64_t kerValue45 = _vel_pack_f32p(pKerValue + 4 * kernHeight * kernWidth,
						       pKerValue + 5 * kernHeight * kernWidth ) ;
	    const uint64_t kerValue67 = _vel_pack_f32p(pKerValue + 6 * kernHeight * kernWidth,
						       pKerValue + 7 * kernHeight * kernWidth ) ;

	    vrsum01 = _vel_pvfmad_vvsvl(vrsum01, kerValue01, vrgoutP, vl) ;
	    vrsum23 = _vel_pvfmad_vvsvl(vrsum23, kerValue23, vrgoutP, vl) ;
	    vrsum45 = _vel_pvfmad_vvsvl(vrsum45, kerValue45, vrgoutP, vl) ;
	    vrsum67 = _vel_pvfmad_vvsvl(vrsum67, kerValue67, vrgoutP, vl) ;
	  } // gOutChannel

	} // kernWidth
      } // kernHeight

      _vel_vstu_vssl(vrsum01, 4, pGIn+gInIndex, vl) ;
      _vel_vstl_vssl(vrsum01, 4, pGIn+gInIndex+  gInPixels, vl) ;
      _vel_vstu_vssl(vrsum23, 4, pGIn+gInIndex+2*gInPixels, vl) ;
      _vel_vstl_vssl(vrsum23, 4, pGIn+gInIndex+3*gInPixels, vl) ;
      _vel_vstu_vssl(vrsum45, 4, pGIn+gInIndex+4*gInPixels, vl) ;
      _vel_vstl_vssl(vrsum45, 4, pGIn+gInIndex+5*gInPixels, vl) ;
      _vel_vstu_vssl(vrsum67, 4, pGIn+gInIndex+6*gInPixels, vl) ;
      _vel_vstl_vssl(vrsum67, 4, pGIn+gInIndex+7*gInPixels, vl) ;

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

      __vr vrsum01 = _vel_vbrdl_vsl(0UL, vl) ;
      __vr vrsum23 = _vel_vbrdl_vsl(0UL, vl) ;
      __vr vrsum45 = _vel_vbrdl_vsl(0UL, vl) ;
      __vr vrsum67 = _vel_vbrdl_vsl(0UL, vl) ;
      __vr vrsum89 = _vel_vbrdl_vsl(0UL, vl) ;
      __vr vrsumAB = _vel_vbrdl_vsl(0UL, vl) ;
      __vr vrsumCD = _vel_vbrdl_vsl(0UL, vl) ;
      __vr vrsumEF = _vel_vbrdl_vsl(0UL, vl) ;

      __vr vrw   = _vel_vaddsl_vsvl(w, _vel_vseq_vl(vl), vl) ;

      for (int64_t r=0; r<kernHeight; r++) {
	int64_t i = h - r * dilationHeight + padHeight ;
	int64_t y = i/strideHeight;
	if ( y*strideHeight != i || y < 0 || gOutHeight <= y)  continue ;

	for (int64_t s=0; s<kernWidth; s++) {
	  __vr vrj = _vel_vaddsl_vsvl(padWidth-s*dilationWidth, vrw, vl) ;
	  __vr vrx = _vel_vdivsl_vvsl(vrj, strideWidth, vl) ;

	  __vm256 vmx0 =  _vel_vfmkleq_mvl(_vel_vcmpsl_vvvl(vrj, _vel_vmulsl_vsvl(strideWidth, vrx, vl), vl), vl) ;
	  __vm256 vmx1 =  _vel_vfmklge_mvl(vrx, vl) ;
	  __vm256 vmx2 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutWidth,vrx, vl), vl) ;

	  __vm256 vmx = _vel_andm_mmm(_vel_andm_mmm(vmx0, vmx1), vmx2) ;

	  for (int64_t k=0; k<gOutChannelGroup; k++) {
	    int64_t gOutIndex    = gOutGroupOffset + ((n * gOutChannel + k) * gOutHeight) * gOutWidth ;
	    __vr vrgout_ptr = _vel_vsfa_vvssl(_vel_vaddsl_vsvl(gOutWidth*y, vrx, vl),
					      2,
					      (unsigned long)(pGOut+gOutIndex), vl) ;

	    __vr vrgout = _vel_vgtu_vvssml(vrgout_ptr, 0, 0, vmx, vl) ;
	    vrgout = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrgout, vmx, vl) ;

	    __vr vrgoutP = _vel_vshf_vvvsl(vrgout, vrgout, VE_VSHUFFLE_YUZU, vl) ;

	    const float *pKerValue = pKernel + kernGroupOffset + ((k * gInChannelGroup + c) * kernHeight + r) * kernWidth + s;

	    const uint64_t kerValue01 = _vel_pack_f32p(pKerValue,
						       pKerValue +     kernHeight * kernWidth ) ;
	    const uint64_t kerValue23 = _vel_pack_f32p(pKerValue + 2 * kernHeight * kernWidth,
						       pKerValue + 3 * kernHeight * kernWidth ) ;
	    const uint64_t kerValue45 = _vel_pack_f32p(pKerValue + 4 * kernHeight * kernWidth,
						       pKerValue + 5 * kernHeight * kernWidth ) ;
	    const uint64_t kerValue67 = _vel_pack_f32p(pKerValue + 6 * kernHeight * kernWidth,
						       pKerValue + 7 * kernHeight * kernWidth ) ;
	    const uint64_t kerValue89 = _vel_pack_f32p(pKerValue + 8 * kernHeight * kernWidth,
						       pKerValue + 9 * kernHeight * kernWidth ) ;
	    const uint64_t kerValueAB = _vel_pack_f32p(pKerValue +10 * kernHeight * kernWidth,
						       pKerValue +11 * kernHeight * kernWidth ) ;
	    const uint64_t kerValueCD = _vel_pack_f32p(pKerValue +12 * kernHeight * kernWidth,
						       pKerValue +13 * kernHeight * kernWidth ) ;
	    const uint64_t kerValueEF = _vel_pack_f32p(pKerValue +14 * kernHeight * kernWidth,
						       pKerValue +15 * kernHeight * kernWidth ) ;

	    vrsum01 = _vel_pvfmad_vvsvl(vrsum01, kerValue01, vrgoutP, vl) ;
	    vrsum23 = _vel_pvfmad_vvsvl(vrsum23, kerValue23, vrgoutP, vl) ;
	    vrsum45 = _vel_pvfmad_vvsvl(vrsum45, kerValue45, vrgoutP, vl) ;
	    vrsum67 = _vel_pvfmad_vvsvl(vrsum67, kerValue67, vrgoutP, vl) ;
	    vrsum89 = _vel_pvfmad_vvsvl(vrsum89, kerValue89, vrgoutP, vl) ;
	    vrsumAB = _vel_pvfmad_vvsvl(vrsumAB, kerValueAB, vrgoutP, vl) ;
	    vrsumCD = _vel_pvfmad_vvsvl(vrsumCD, kerValueCD, vrgoutP, vl) ;
	    vrsumEF = _vel_pvfmad_vvsvl(vrsumEF, kerValueEF, vrgoutP, vl) ;
	  } // gOutChannel

	} // kernWidth
      } // kernHeight

      _vel_vstu_vssl(vrsum01, 4, pGIn+gInIndex, vl) ;
      _vel_vstl_vssl(vrsum01, 4, pGIn+gInIndex+  gInPixels, vl) ;
      _vel_vstu_vssl(vrsum23, 4, pGIn+gInIndex+2*gInPixels, vl) ;
      _vel_vstl_vssl(vrsum23, 4, pGIn+gInIndex+3*gInPixels, vl) ;
      _vel_vstu_vssl(vrsum45, 4, pGIn+gInIndex+4*gInPixels, vl) ;
      _vel_vstl_vssl(vrsum45, 4, pGIn+gInIndex+5*gInPixels, vl) ;
      _vel_vstu_vssl(vrsum67, 4, pGIn+gInIndex+6*gInPixels, vl) ;
      _vel_vstl_vssl(vrsum67, 4, pGIn+gInIndex+7*gInPixels, vl) ;
      _vel_vstu_vssl(vrsum89, 4, pGIn+gInIndex+8*gInPixels, vl) ;
      _vel_vstl_vssl(vrsum89, 4, pGIn+gInIndex+9*gInPixels, vl) ;
      _vel_vstu_vssl(vrsumAB, 4, pGIn+gInIndex+10*gInPixels, vl) ;
      _vel_vstl_vssl(vrsumAB, 4, pGIn+gInIndex+11*gInPixels, vl) ;
      _vel_vstu_vssl(vrsumCD, 4, pGIn+gInIndex+12*gInPixels, vl) ;
      _vel_vstl_vssl(vrsumCD, 4, pGIn+gInIndex+13*gInPixels, vl) ;
      _vel_vstu_vssl(vrsumEF, 4, pGIn+gInIndex+14*gInPixels, vl) ;
      _vel_vstl_vssl(vrsumEF, 4, pGIn+gInIndex+15*gInPixels, vl) ;

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
