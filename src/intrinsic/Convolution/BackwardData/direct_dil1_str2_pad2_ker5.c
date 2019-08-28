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

      const int64_t xmin_s0 = (w+2) / 2 ;
      const int64_t xmin_s2 = (w+0) / 2 ;
      const int64_t xmin_s4 = w >= 2 ? (w-2) / 2 : 0 ;

      __vr vrw   = _vel_vaddsl_vsvl(w, _vel_vseq_vl(vl), vl) ;

      __vr vrj_s0 = _vel_vaddsl_vsvl(2, vrw, vl) ;
      __vr vrj_s1 = _vel_vaddsl_vsvl(1, vrw, vl) ;
      __vr vrj_s2 = vrw ;
      __vr vrj_s3 = _vel_vaddsl_vsvl(-1, vrw, vl) ;
      __vr vrj_s4 = _vel_vaddsl_vsvl(-2, vrw, vl) ;

      __vr vrx_s0 = _vel_vdivsl_vvsl(vrj_s0, 2, vl) ;
      __vr vrx_s12 = _vel_vdivsl_vvsl(vrj_s1, 2, vl) ;
      __vr vrx_s34 = _vel_vdivsl_vvsl(vrj_s3, 2, vl) ;

      __vm256 vmx0_s0 =  _vel_vfmkleq_mvl(_vel_vcmpsl_vvvl(vrj_s0, _vel_vmulsl_vsvl(2, vrx_s0, vl), vl), vl) ;
      __vm256 vmx1_s0 =  _vel_vfmklge_mvl(vrx_s0, vl) ;
      __vm256 vmx2_s0 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutWidth,vrx_s0, vl), vl) ;
      __vm256 vmx_s0 = _vel_andm_mmm(_vel_andm_mmm(vmx0_s0, vmx1_s0), vmx2_s0) ;

      __vm256 vmx0_s1 =  _vel_vfmkleq_mvl(_vel_vcmpsl_vvvl(vrj_s1, _vel_vmulsl_vsvl(2, vrx_s12, vl), vl), vl) ;
      __vm256 vmx0_s2 =  _vel_vfmkleq_mvl(_vel_vcmpsl_vvvl(vrj_s2, _vel_vmulsl_vsvl(2, vrx_s12, vl), vl), vl) ;
      __vm256 vmx1_s12 =  _vel_vfmklge_mvl(vrx_s12, vl) ;
      __vm256 vmx2_s12 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutWidth,vrx_s12, vl), vl) ;
      __vm256 vmx_s12 = _vel_andm_mmm(vmx1_s12, vmx2_s12) ;
      __vm256 vmx_s1 = _vel_andm_mmm(vmx0_s1, vmx_s12) ;
      __vm256 vmx_s2 = _vel_andm_mmm(vmx0_s2, vmx_s12) ;

      __vm256 vmx0_s3 =  _vel_vfmkleq_mvl(_vel_vcmpsl_vvvl(vrj_s3, _vel_vmulsl_vsvl(2, vrx_s34, vl), vl), vl) ;
      __vm256 vmx0_s4 =  _vel_vfmkleq_mvl(_vel_vcmpsl_vvvl(vrj_s4, _vel_vmulsl_vsvl(2, vrx_s34, vl), vl), vl) ;
      __vm256 vmx1_s34 =  _vel_vfmklge_mvl(vrx_s34, vl) ;
      __vm256 vmx2_s34 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutWidth,vrx_s34, vl), vl) ;
      __vm256 vmx_s34 = _vel_andm_mmm(vmx1_s34, vmx2_s34) ;
      __vm256 vmx_s3 = _vel_andm_mmm(vmx0_s3, vmx_s34) ;
      __vm256 vmx_s4 = _vel_andm_mmm(vmx0_s4, vmx_s34) ;

      __vr vrsum_s0 = _vel_vbrds_vsl(0.f, vl/2) ;
      __vr vrsum_s1 = _vel_vbrds_vsl(0.f, vl/2) ;
      __vr vrsum_s2 = _vel_vbrds_vsl(0.f, vl/2) ;
      __vr vrsum_s3 = _vel_vbrds_vsl(0.f, vl/2) ;
      __vr vrsum_s4 = _vel_vbrds_vsl(0.f, vl/2) ;

      for (int64_t r=0; r<kernHeight; r++) {
	int64_t i = h - r + 2 ;
	int64_t y = i/2;
	if ( y*2 != i || y < 0 || gOutHeight <= y)  continue ;

	for (int64_t k=0; k<gOutChannelGroup; k++) {
	  int64_t gOutIndex    = gOutGroupOffset + ((n * gOutChannel + k) * gOutHeight) * gOutWidth ;
	  const float *pKerValue = pKernel + kernGroupOffset + ((k * gInChannelGroup + c) * kernHeight) * kernWidth;

	  __vr vrgout_s01 = _vel_vldu_vssl(4, pGOut+gOutIndex+gOutWidth*y + xmin_s0 , vl/2) ;
	  __vr vrgout_s23 = _vel_vldu_vssl(4, pGOut+gOutIndex+gOutWidth*y + xmin_s2 , vl/2) ;
	  __vr vrgout_s4  = _vel_vldu_vssl(4, pGOut+gOutIndex+gOutWidth*y + xmin_s4 , vl/2) ;

#define VFMAD_C1(VRGOUT, K, R, S)	{										\
	    const float kerValue = pKerValue[(((K)*gInChannelGroup + 0) * kernHeight +(R)) * kernWidth + (S) ] ;	\
	    vrsum_s##S = _vel_vfmads_vvsvl(vrsum_s##S, kerValue, VRGOUT, vl/2) ;					\
}

	  VFMAD_C1(vrgout_s01, 0, r, 0) ;
	  VFMAD_C1(vrgout_s01, 0, r, 1) ;
	  VFMAD_C1(vrgout_s23, 0, r, 2) ;
	  VFMAD_C1(vrgout_s23, 0, r, 3) ;
	  VFMAD_C1(vrgout_s4,  0, r, 4) ;

#undef VFMAD_C1
	}

      } // kernHeight

      vrsum_s0 = _vel_vex_vvmvl(vrsum_s0, vmx_s0, _vel_vbrds_vsl(0.f, vl), vl) ;
      vrsum_s1 = _vel_vex_vvmvl(vrsum_s1, vmx_s1, _vel_vbrds_vsl(0.f, vl), vl) ;
      vrsum_s2 = _vel_vex_vvmvl(vrsum_s2, vmx_s2, _vel_vbrds_vsl(0.f, vl), vl) ;
      vrsum_s3 = _vel_vex_vvmvl(vrsum_s3, vmx_s3, _vel_vbrds_vsl(0.f, vl), vl) ;
      vrsum_s4 = _vel_vex_vvmvl(vrsum_s4, vmx_s4, _vel_vbrds_vsl(0.f, vl), vl) ;
      __vr vrsum = _vel_vfadds_vvvl(_vel_vfadds_vvvl(vrsum_s0, _vel_vfadds_vvvl(vrsum_s1, vrsum_s2, vl), vl),
                                    _vel_vfadds_vvvl(vrsum_s3, vrsum_s4, vl), vl) ;
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

      const int64_t xmin_s0 = (w+2) / 2 ;
      const int64_t xmin_s2 = (w+0) / 2 ;
      const int64_t xmin_s4 = w >= 2 ? (w-2) / 2 : 0 ;

      __vr vrw   = _vel_vaddsl_vsvl(w, _vel_vseq_vl(vl), vl) ;

      __vr vrj_s0 = _vel_vaddsl_vsvl(2, vrw, vl) ;
      __vr vrj_s1 = _vel_vaddsl_vsvl(1, vrw, vl) ;
      __vr vrj_s2 = vrw ;
      __vr vrj_s3 = _vel_vaddsl_vsvl(-1, vrw, vl) ;
      __vr vrj_s4 = _vel_vaddsl_vsvl(-2, vrw, vl) ;

      __vr vrx_s0 = _vel_vdivsl_vvsl(vrj_s0, 2, vl) ;
      __vr vrx_s12 = _vel_vdivsl_vvsl(vrj_s1, 2, vl) ;
      __vr vrx_s34 = _vel_vdivsl_vvsl(vrj_s3, 2, vl) ;

      __vm256 vmx0_s0 =  _vel_vfmkleq_mvl(_vel_vcmpsl_vvvl(vrj_s0, _vel_vmulsl_vsvl(2, vrx_s0, vl), vl), vl) ;
      __vm256 vmx1_s0 =  _vel_vfmklge_mvl(vrx_s0, vl) ;
      __vm256 vmx2_s0 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutWidth,vrx_s0, vl), vl) ;
      __vm256 vmx_s0 = _vel_andm_mmm(_vel_andm_mmm(vmx0_s0, vmx1_s0), vmx2_s0) ;

      __vm256 vmx0_s1 =  _vel_vfmkleq_mvl(_vel_vcmpsl_vvvl(vrj_s1, _vel_vmulsl_vsvl(2, vrx_s12, vl), vl), vl) ;
      __vm256 vmx0_s2 =  _vel_vfmkleq_mvl(_vel_vcmpsl_vvvl(vrj_s2, _vel_vmulsl_vsvl(2, vrx_s12, vl), vl), vl) ;
      __vm256 vmx1_s12 =  _vel_vfmklge_mvl(vrx_s12, vl) ;
      __vm256 vmx2_s12 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutWidth,vrx_s12, vl), vl) ;
      __vm256 vmx_s12 = _vel_andm_mmm(vmx1_s12, vmx2_s12) ;
      __vm256 vmx_s1 = _vel_andm_mmm(vmx0_s1, vmx_s12) ;
      __vm256 vmx_s2 = _vel_andm_mmm(vmx0_s2, vmx_s12) ;

      __vm256 vmx0_s3 =  _vel_vfmkleq_mvl(_vel_vcmpsl_vvvl(vrj_s3, _vel_vmulsl_vsvl(2, vrx_s34, vl), vl), vl) ;
      __vm256 vmx0_s4 =  _vel_vfmkleq_mvl(_vel_vcmpsl_vvvl(vrj_s4, _vel_vmulsl_vsvl(2, vrx_s34, vl), vl), vl) ;
      __vm256 vmx1_s34 =  _vel_vfmklge_mvl(vrx_s34, vl) ;
      __vm256 vmx2_s34 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutWidth,vrx_s34, vl), vl) ;
      __vm256 vmx_s34 = _vel_andm_mmm(vmx1_s34, vmx2_s34) ;
      __vm256 vmx_s3 = _vel_andm_mmm(vmx0_s3, vmx_s34) ;
      __vm256 vmx_s4 = _vel_andm_mmm(vmx0_s4, vmx_s34) ;

      __vr vrsum01_s0 = _vel_vbrdl_vsl(0UL, vl/2) ;
      __vr vrsum01_s1 = _vel_vbrdl_vsl(0UL, vl/2) ;
      __vr vrsum01_s2 = _vel_vbrdl_vsl(0UL, vl/2) ;
      __vr vrsum01_s3 = _vel_vbrdl_vsl(0UL, vl/2) ;
      __vr vrsum01_s4 = _vel_vbrdl_vsl(0UL, vl/2) ;

      for (int64_t r=0; r<kernHeight; r++) {
	int64_t i = h - r + 2 ;
	int64_t y = i/2;
	if ( y*2 != i || y < 0 || gOutHeight <= y)  continue ;

	for (int64_t k=0; k<gOutChannelGroup; k++) {
	  int64_t gOutIndex    = gOutGroupOffset + ((n * gOutChannel + k) * gOutHeight) * gOutWidth ;
	  const float *pKerValue = pKernel + kernGroupOffset + ((k * gInChannelGroup + c) * kernHeight) * kernWidth;

	  __vr vrgout_s01 = _vel_vldu_vssl(4, pGOut+gOutIndex+gOutWidth*y + xmin_s0 , vl/2) ;
	  __vr vrgout_s23 = _vel_vldu_vssl(4, pGOut+gOutIndex+gOutWidth*y + xmin_s2 , vl/2) ;
	  __vr vrgout_s4  = _vel_vldu_vssl(4, pGOut+gOutIndex+gOutWidth*y + xmin_s4 , vl/2) ;

#define VFMAD_C2(VRGOUT, K, R, S)	{												\
	    const uint64_t kerValue01 = _vel_pack_f32p(pKerValue + (((K)*gInChannelGroup + 0) * kernHeight +(R)) * kernWidth + (S),	\
						       pKerValue + (((K)*gInChannelGroup + 1) * kernHeight +(R)) * kernWidth + (S)) ;	\
	    __vr vrgoutP = _vel_vshf_vvvsl(VRGOUT, VRGOUT, VE_VSHUFFLE_YUZU, vl/2) ;		\
	    vrsum01_s##S = _vel_pvfmad_vvsvl(vrsum01_s##S, kerValue01, vrgoutP, vl/2) ;		\
}

	  VFMAD_C2(vrgout_s01, 0, r, 0) ;
	  VFMAD_C2(vrgout_s01, 0, r, 1) ;
	  VFMAD_C2(vrgout_s23, 0, r, 2) ;
	  VFMAD_C2(vrgout_s23, 0, r, 3) ;
	  VFMAD_C2(vrgout_s4,  0, r, 4) ;

#undef VFMAD_C2
	}

      } // kernHeight

      vrsum01_s0 = _vel_vex_vvmvl(vrsum01_s0, vmx_s0, _vel_vbrdl_vsl(0UL, vl), vl) ;
      vrsum01_s1 = _vel_vex_vvmvl(vrsum01_s1, vmx_s1, _vel_vbrdl_vsl(0UL, vl), vl) ;
      vrsum01_s2 = _vel_vex_vvmvl(vrsum01_s2, vmx_s2, _vel_vbrdl_vsl(0UL, vl), vl) ;
      vrsum01_s3 = _vel_vex_vvmvl(vrsum01_s3, vmx_s3, _vel_vbrdl_vsl(0UL, vl), vl) ;
      vrsum01_s4 = _vel_vex_vvmvl(vrsum01_s4, vmx_s4, _vel_vbrdl_vsl(0UL, vl), vl) ;
      __vr vrsum01 = _vel_pvfadd_vvvl(_vel_pvfadd_vvvl(vrsum01_s0, _vel_pvfadd_vvvl(vrsum01_s1, vrsum01_s2, vl), vl),
                                      _vel_pvfadd_vvvl(vrsum01_s3, vrsum01_s4, vl), vl) ;
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

      const int64_t xmin_s0 = (w+2) / 2 ;
      const int64_t xmin_s2 = (w+0) / 2 ;
      const int64_t xmin_s4 = w >= 2 ? (w-2) / 2 : 0 ;

      __vr vrw   = _vel_vaddsl_vsvl(w, _vel_vseq_vl(vl), vl) ;

      __vr vrj_s0 = _vel_vaddsl_vsvl(2, vrw, vl) ;
      __vr vrj_s1 = _vel_vaddsl_vsvl(1, vrw, vl) ;
      __vr vrj_s2 = vrw ;
      __vr vrj_s3 = _vel_vaddsl_vsvl(-1, vrw, vl) ;
      __vr vrj_s4 = _vel_vaddsl_vsvl(-2, vrw, vl) ;

      __vr vrx_s0 = _vel_vdivsl_vvsl(vrj_s0, 2, vl) ;
      __vr vrx_s12 = _vel_vdivsl_vvsl(vrj_s1, 2, vl) ;
      __vr vrx_s34 = _vel_vdivsl_vvsl(vrj_s3, 2, vl) ;

      __vm256 vmx0_s0 =  _vel_vfmkleq_mvl(_vel_vcmpsl_vvvl(vrj_s0, _vel_vmulsl_vsvl(2, vrx_s0, vl), vl), vl) ;
      __vm256 vmx1_s0 =  _vel_vfmklge_mvl(vrx_s0, vl) ;
      __vm256 vmx2_s0 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutWidth,vrx_s0, vl), vl) ;
      __vm256 vmx_s0 = _vel_andm_mmm(_vel_andm_mmm(vmx0_s0, vmx1_s0), vmx2_s0) ;

      __vm256 vmx0_s1 =  _vel_vfmkleq_mvl(_vel_vcmpsl_vvvl(vrj_s1, _vel_vmulsl_vsvl(2, vrx_s12, vl), vl), vl) ;
      __vm256 vmx0_s2 =  _vel_vfmkleq_mvl(_vel_vcmpsl_vvvl(vrj_s2, _vel_vmulsl_vsvl(2, vrx_s12, vl), vl), vl) ;
      __vm256 vmx1_s12 =  _vel_vfmklge_mvl(vrx_s12, vl) ;
      __vm256 vmx2_s12 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutWidth,vrx_s12, vl), vl) ;
      __vm256 vmx_s12 = _vel_andm_mmm(vmx1_s12, vmx2_s12) ;
      __vm256 vmx_s1 = _vel_andm_mmm(vmx0_s1, vmx_s12) ;
      __vm256 vmx_s2 = _vel_andm_mmm(vmx0_s2, vmx_s12) ;

      __vm256 vmx0_s3 =  _vel_vfmkleq_mvl(_vel_vcmpsl_vvvl(vrj_s3, _vel_vmulsl_vsvl(2, vrx_s34, vl), vl), vl) ;
      __vm256 vmx0_s4 =  _vel_vfmkleq_mvl(_vel_vcmpsl_vvvl(vrj_s4, _vel_vmulsl_vsvl(2, vrx_s34, vl), vl), vl) ;
      __vm256 vmx1_s34 =  _vel_vfmklge_mvl(vrx_s34, vl) ;
      __vm256 vmx2_s34 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutWidth,vrx_s34, vl), vl) ;
      __vm256 vmx_s34 = _vel_andm_mmm(vmx1_s34, vmx2_s34) ;
      __vm256 vmx_s3 = _vel_andm_mmm(vmx0_s3, vmx_s34) ;
      __vm256 vmx_s4 = _vel_andm_mmm(vmx0_s4, vmx_s34) ;

      __vr vrsum01_s0 = _vel_vbrdl_vsl(0UL, vl/2) ;
      __vr vrsum23_s0 = _vel_vbrdl_vsl(0UL, vl/2) ;

      __vr vrsum01_s1 = _vel_vbrdl_vsl(0UL, vl/2) ;
      __vr vrsum23_s1 = _vel_vbrdl_vsl(0UL, vl/2) ;

      __vr vrsum01_s2 = _vel_vbrdl_vsl(0UL, vl/2) ;
      __vr vrsum23_s2 = _vel_vbrdl_vsl(0UL, vl/2) ;

      __vr vrsum01_s3 = _vel_vbrdl_vsl(0UL, vl/2) ;
      __vr vrsum23_s3 = _vel_vbrdl_vsl(0UL, vl/2) ;

      __vr vrsum01_s4 = _vel_vbrdl_vsl(0UL, vl/2) ;
      __vr vrsum23_s4 = _vel_vbrdl_vsl(0UL, vl/2) ;

      for (int64_t r=0; r<kernHeight; r++) {
	int64_t i = h - r + 2 ;
	int64_t y = i/2;
	if ( y*2 != i || y < 0 || gOutHeight <= y)  continue ;

	for (int64_t k=0; k<gOutChannelGroup; k++) {
	  int64_t gOutIndex    = gOutGroupOffset + ((n * gOutChannel + k) * gOutHeight) * gOutWidth ;
	  const float *pKerValue = pKernel + kernGroupOffset + ((k * gInChannelGroup + c) * kernHeight) * kernWidth;

	  __vr vrgout_s01 = _vel_vldu_vssl(4, pGOut+gOutIndex+gOutWidth*y + xmin_s0 , vl/2) ;
	  __vr vrgout_s23 = _vel_vldu_vssl(4, pGOut+gOutIndex+gOutWidth*y + xmin_s2 , vl/2) ;
	  __vr vrgout_s4  = _vel_vldu_vssl(4, pGOut+gOutIndex+gOutWidth*y + xmin_s4 , vl/2) ;

#define VFMAD_C4(VRGOUT, K, R, S)	{												\
	    const uint64_t kerValue01 = _vel_pack_f32p(pKerValue + (((K)*gInChannelGroup + 0) * kernHeight +(R)) * kernWidth + (S),	\
						       pKerValue + (((K)*gInChannelGroup + 1) * kernHeight +(R)) * kernWidth + (S)) ;	\
	    const uint64_t kerValue23 = _vel_pack_f32p(pKerValue + (((K)*gInChannelGroup + 2) * kernHeight +(R)) * kernWidth + (S),	\
						       pKerValue + (((K)*gInChannelGroup + 3) * kernHeight +(R)) * kernWidth + (S)) ;	\
	    __vr vrgoutP = _vel_vshf_vvvsl(VRGOUT, VRGOUT, VE_VSHUFFLE_YUZU, vl/2) ;		\
	    vrsum01_s##S = _vel_pvfmad_vvsvl(vrsum01_s##S, kerValue01, vrgoutP, vl/2) ;		\
	    vrsum23_s##S = _vel_pvfmad_vvsvl(vrsum23_s##S, kerValue23, vrgoutP, vl/2) ;		\
}

	  VFMAD_C4(vrgout_s01, 0, r, 0) ;
	  VFMAD_C4(vrgout_s01, 0, r, 1) ;
	  VFMAD_C4(vrgout_s23, 0, r, 2) ;
	  VFMAD_C4(vrgout_s23, 0, r, 3) ;
	  VFMAD_C4(vrgout_s4,  0, r, 4) ;

#undef VFMAD_C4
	}

      } // kernHeight

      vrsum01_s0 = _vel_vex_vvmvl(vrsum01_s0, vmx_s0, _vel_vbrdl_vsl(0UL, vl), vl) ;
      vrsum01_s1 = _vel_vex_vvmvl(vrsum01_s1, vmx_s1, _vel_vbrdl_vsl(0UL, vl), vl) ;
      vrsum01_s2 = _vel_vex_vvmvl(vrsum01_s2, vmx_s2, _vel_vbrdl_vsl(0UL, vl), vl) ;
      vrsum01_s3 = _vel_vex_vvmvl(vrsum01_s3, vmx_s3, _vel_vbrdl_vsl(0UL, vl), vl) ;
      vrsum01_s4 = _vel_vex_vvmvl(vrsum01_s4, vmx_s4, _vel_vbrdl_vsl(0UL, vl), vl) ;
      __vr vrsum01 = _vel_pvfadd_vvvl(_vel_pvfadd_vvvl(vrsum01_s0, _vel_pvfadd_vvvl(vrsum01_s1, vrsum01_s2, vl), vl),
                                      _vel_pvfadd_vvvl(vrsum01_s3, vrsum01_s4, vl), vl) ;
      _vel_vstu_vssl(vrsum01, 4, pGIn+gInIndex, vl) ;
      _vel_vstl_vssl(vrsum01, 4, pGIn+gInIndex+  gInPixels, vl) ;

      vrsum23_s0 = _vel_vex_vvmvl(vrsum23_s0, vmx_s0, _vel_vbrdl_vsl(0UL, vl), vl) ;
      vrsum23_s1 = _vel_vex_vvmvl(vrsum23_s1, vmx_s1, _vel_vbrdl_vsl(0UL, vl), vl) ;
      vrsum23_s2 = _vel_vex_vvmvl(vrsum23_s2, vmx_s2, _vel_vbrdl_vsl(0UL, vl), vl) ;
      vrsum23_s3 = _vel_vex_vvmvl(vrsum23_s3, vmx_s3, _vel_vbrdl_vsl(0UL, vl), vl) ;
      vrsum23_s4 = _vel_vex_vvmvl(vrsum23_s4, vmx_s4, _vel_vbrdl_vsl(0UL, vl), vl) ;
      __vr vrsum23 = _vel_pvfadd_vvvl(_vel_pvfadd_vvvl(vrsum23_s0, _vel_pvfadd_vvvl(vrsum23_s1, vrsum23_s2, vl), vl),
                                      _vel_pvfadd_vvvl(vrsum23_s3, vrsum23_s4, vl), vl) ;
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

      const int64_t xmin_s0 = (w+2) / 2 ;
      const int64_t xmin_s2 = (w+0) / 2 ;
      const int64_t xmin_s4 = w >= 2 ? (w-2) / 2 : 0 ;

      __vr vrw   = _vel_vaddsl_vsvl(w, _vel_vseq_vl(vl), vl) ;

      __vr vrj_s0 = _vel_vaddsl_vsvl(2, vrw, vl) ;
      __vr vrj_s1 = _vel_vaddsl_vsvl(1, vrw, vl) ;
      __vr vrj_s2 = vrw ;
      __vr vrj_s3 = _vel_vaddsl_vsvl(-1, vrw, vl) ;
      __vr vrj_s4 = _vel_vaddsl_vsvl(-2, vrw, vl) ;

      __vr vrx_s0 = _vel_vdivsl_vvsl(vrj_s0, 2, vl) ;
      __vr vrx_s12 = _vel_vdivsl_vvsl(vrj_s1, 2, vl) ;
      __vr vrx_s34 = _vel_vdivsl_vvsl(vrj_s3, 2, vl) ;

      __vm256 vmx0_s0 =  _vel_vfmkleq_mvl(_vel_vcmpsl_vvvl(vrj_s0, _vel_vmulsl_vsvl(2, vrx_s0, vl), vl), vl) ;
      __vm256 vmx1_s0 =  _vel_vfmklge_mvl(vrx_s0, vl) ;
      __vm256 vmx2_s0 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutWidth,vrx_s0, vl), vl) ;
      __vm256 vmx_s0 = _vel_andm_mmm(_vel_andm_mmm(vmx0_s0, vmx1_s0), vmx2_s0) ;

      __vm256 vmx0_s1 =  _vel_vfmkleq_mvl(_vel_vcmpsl_vvvl(vrj_s1, _vel_vmulsl_vsvl(2, vrx_s12, vl), vl), vl) ;
      __vm256 vmx0_s2 =  _vel_vfmkleq_mvl(_vel_vcmpsl_vvvl(vrj_s2, _vel_vmulsl_vsvl(2, vrx_s12, vl), vl), vl) ;
      __vm256 vmx1_s12 =  _vel_vfmklge_mvl(vrx_s12, vl) ;
      __vm256 vmx2_s12 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutWidth,vrx_s12, vl), vl) ;
      __vm256 vmx_s12 = _vel_andm_mmm(vmx1_s12, vmx2_s12) ;
      __vm256 vmx_s1 = _vel_andm_mmm(vmx0_s1, vmx_s12) ;
      __vm256 vmx_s2 = _vel_andm_mmm(vmx0_s2, vmx_s12) ;

      __vm256 vmx0_s3 =  _vel_vfmkleq_mvl(_vel_vcmpsl_vvvl(vrj_s3, _vel_vmulsl_vsvl(2, vrx_s34, vl), vl), vl) ;
      __vm256 vmx0_s4 =  _vel_vfmkleq_mvl(_vel_vcmpsl_vvvl(vrj_s4, _vel_vmulsl_vsvl(2, vrx_s34, vl), vl), vl) ;
      __vm256 vmx1_s34 =  _vel_vfmklge_mvl(vrx_s34, vl) ;
      __vm256 vmx2_s34 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutWidth,vrx_s34, vl), vl) ;
      __vm256 vmx_s34 = _vel_andm_mmm(vmx1_s34, vmx2_s34) ;
      __vm256 vmx_s3 = _vel_andm_mmm(vmx0_s3, vmx_s34) ;
      __vm256 vmx_s4 = _vel_andm_mmm(vmx0_s4, vmx_s34) ;

      __vr vrsum01_s0 = _vel_vbrdl_vsl(0UL, vl/2) ;
      __vr vrsum23_s0 = _vel_vbrdl_vsl(0UL, vl/2) ;
      __vr vrsum45_s0 = _vel_vbrdl_vsl(0UL, vl/2) ;
      __vr vrsum67_s0 = _vel_vbrdl_vsl(0UL, vl/2) ;

      __vr vrsum01_s1 = _vel_vbrdl_vsl(0UL, vl/2) ;
      __vr vrsum23_s1 = _vel_vbrdl_vsl(0UL, vl/2) ;
      __vr vrsum45_s1 = _vel_vbrdl_vsl(0UL, vl/2) ;
      __vr vrsum67_s1 = _vel_vbrdl_vsl(0UL, vl/2) ;

      __vr vrsum01_s2 = _vel_vbrdl_vsl(0UL, vl/2) ;
      __vr vrsum23_s2 = _vel_vbrdl_vsl(0UL, vl/2) ;
      __vr vrsum45_s2 = _vel_vbrdl_vsl(0UL, vl/2) ;
      __vr vrsum67_s2 = _vel_vbrdl_vsl(0UL, vl/2) ;

      __vr vrsum01_s3 = _vel_vbrdl_vsl(0UL, vl/2) ;
      __vr vrsum23_s3 = _vel_vbrdl_vsl(0UL, vl/2) ;
      __vr vrsum45_s3 = _vel_vbrdl_vsl(0UL, vl/2) ;
      __vr vrsum67_s3 = _vel_vbrdl_vsl(0UL, vl/2) ;

      __vr vrsum01_s4 = _vel_vbrdl_vsl(0UL, vl/2) ;
      __vr vrsum23_s4 = _vel_vbrdl_vsl(0UL, vl/2) ;
      __vr vrsum45_s4 = _vel_vbrdl_vsl(0UL, vl/2) ;
      __vr vrsum67_s4 = _vel_vbrdl_vsl(0UL, vl/2) ;

      for (int64_t r=0; r<kernHeight; r++) {
	int64_t i = h - r + 2 ;
	int64_t y = i/2;
	if ( y*2 != i || y < 0 || gOutHeight <= y)  continue ;

	for (int64_t k=0; k<gOutChannelGroup; k++) {
	  int64_t gOutIndex    = gOutGroupOffset + ((n * gOutChannel + k) * gOutHeight) * gOutWidth ;
	  const float *pKerValue = pKernel + kernGroupOffset + ((k * gInChannelGroup + c) * kernHeight) * kernWidth;

	  __vr vrgout_s01 = _vel_vldu_vssl(4, pGOut+gOutIndex+gOutWidth*y + xmin_s0 , vl/2) ;
	  __vr vrgout_s23 = _vel_vldu_vssl(4, pGOut+gOutIndex+gOutWidth*y + xmin_s2 , vl/2) ;
	  __vr vrgout_s4  = _vel_vldu_vssl(4, pGOut+gOutIndex+gOutWidth*y + xmin_s4 , vl/2) ;

#define VFMAD_C8(VRGOUT, K, R, S)	{												\
	    const uint64_t kerValue01 = _vel_pack_f32p(pKerValue + (((K)*gInChannelGroup + 0) * kernHeight +(R)) * kernWidth + (S),	\
						       pKerValue + (((K)*gInChannelGroup + 1) * kernHeight +(R)) * kernWidth + (S)) ;	\
	    const uint64_t kerValue23 = _vel_pack_f32p(pKerValue + (((K)*gInChannelGroup + 2) * kernHeight +(R)) * kernWidth + (S),	\
						       pKerValue + (((K)*gInChannelGroup + 3) * kernHeight +(R)) * kernWidth + (S)) ;	\
	    const uint64_t kerValue45 = _vel_pack_f32p(pKerValue + (((K)*gInChannelGroup + 4) * kernHeight +(R)) * kernWidth + (S),	\
						       pKerValue + (((K)*gInChannelGroup + 5) * kernHeight +(R)) * kernWidth + (S)) ;	\
	    const uint64_t kerValue67 = _vel_pack_f32p(pKerValue + (((K)*gInChannelGroup + 6) * kernHeight +(R)) * kernWidth + (S),	\
						       pKerValue + (((K)*gInChannelGroup + 7) * kernHeight +(R)) * kernWidth + (S)) ;	\
	    __vr vrgoutP = _vel_vshf_vvvsl(VRGOUT, VRGOUT, VE_VSHUFFLE_YUZU, vl/2) ;		\
	    vrsum01_s##S = _vel_pvfmad_vvsvl(vrsum01_s##S, kerValue01, vrgoutP, vl/2) ;		\
	    vrsum23_s##S = _vel_pvfmad_vvsvl(vrsum23_s##S, kerValue23, vrgoutP, vl/2) ;		\
	    vrsum45_s##S = _vel_pvfmad_vvsvl(vrsum45_s##S, kerValue45, vrgoutP, vl/2) ;		\
	    vrsum67_s##S = _vel_pvfmad_vvsvl(vrsum67_s##S, kerValue67, vrgoutP, vl/2) ;		\
}

	  VFMAD_C8(vrgout_s01, 0, r, 0) ;
	  VFMAD_C8(vrgout_s01, 0, r, 1) ;
	  VFMAD_C8(vrgout_s23, 0, r, 2) ;
	  VFMAD_C8(vrgout_s23, 0, r, 3) ;
	  VFMAD_C8(vrgout_s4,  0, r, 4) ;

#undef VFMAD_C8
	}

      } // kernHeight

       ;
      vrsum01_s0 = _vel_vex_vvmvl(vrsum01_s0, vmx_s0, _vel_vbrdl_vsl(0UL, vl), vl) ;
      vrsum01_s1 = _vel_vex_vvmvl(vrsum01_s1, vmx_s1, _vel_vbrdl_vsl(0UL, vl), vl) ;
      vrsum01_s2 = _vel_vex_vvmvl(vrsum01_s2, vmx_s2, _vel_vbrdl_vsl(0UL, vl), vl) ;
      vrsum01_s3 = _vel_vex_vvmvl(vrsum01_s3, vmx_s3, _vel_vbrdl_vsl(0UL, vl), vl) ;
      vrsum01_s4 = _vel_vex_vvmvl(vrsum01_s4, vmx_s4, _vel_vbrdl_vsl(0UL, vl), vl) ;
      __vr vrsum01 = _vel_pvfadd_vvvl(_vel_pvfadd_vvvl(vrsum01_s0, _vel_pvfadd_vvvl(vrsum01_s1, vrsum01_s2, vl), vl),
                                      _vel_pvfadd_vvvl(vrsum01_s3, vrsum01_s4, vl), vl) ;
      _vel_vstu_vssl(vrsum01, 4, pGIn+gInIndex, vl) ;
      _vel_vstl_vssl(vrsum01, 4, pGIn+gInIndex+  gInPixels, vl) ;

      vrsum23_s0 = _vel_vex_vvmvl(vrsum23_s0, vmx_s0, _vel_vbrdl_vsl(0UL, vl), vl) ;
      vrsum23_s1 = _vel_vex_vvmvl(vrsum23_s1, vmx_s1, _vel_vbrdl_vsl(0UL, vl), vl) ;
      vrsum23_s2 = _vel_vex_vvmvl(vrsum23_s2, vmx_s2, _vel_vbrdl_vsl(0UL, vl), vl) ;
      vrsum23_s3 = _vel_vex_vvmvl(vrsum23_s3, vmx_s3, _vel_vbrdl_vsl(0UL, vl), vl) ;
      vrsum23_s4 = _vel_vex_vvmvl(vrsum23_s4, vmx_s4, _vel_vbrdl_vsl(0UL, vl), vl) ;
      __vr vrsum23 = _vel_pvfadd_vvvl(_vel_pvfadd_vvvl(vrsum23_s0, _vel_pvfadd_vvvl(vrsum23_s1, vrsum23_s2, vl), vl),
                                      _vel_pvfadd_vvvl(vrsum23_s3, vrsum23_s4, vl), vl) ;
      _vel_vstu_vssl(vrsum23, 4, pGIn+gInIndex+2*gInPixels, vl) ;
      _vel_vstl_vssl(vrsum23, 4, pGIn+gInIndex+3*gInPixels, vl) ;

      vrsum45_s0 = _vel_vex_vvmvl(vrsum45_s0, vmx_s0, _vel_vbrdl_vsl(0UL, vl), vl) ;
      vrsum45_s1 = _vel_vex_vvmvl(vrsum45_s1, vmx_s1, _vel_vbrdl_vsl(0UL, vl), vl) ;
      vrsum45_s2 = _vel_vex_vvmvl(vrsum45_s2, vmx_s2, _vel_vbrdl_vsl(0UL, vl), vl) ;
      vrsum45_s3 = _vel_vex_vvmvl(vrsum45_s3, vmx_s3, _vel_vbrdl_vsl(0UL, vl), vl) ;
      vrsum45_s4 = _vel_vex_vvmvl(vrsum45_s4, vmx_s4, _vel_vbrdl_vsl(0UL, vl), vl) ;
      __vr vrsum45 = _vel_pvfadd_vvvl(_vel_pvfadd_vvvl(vrsum45_s0, _vel_pvfadd_vvvl(vrsum45_s1, vrsum45_s2, vl), vl),
                                      _vel_pvfadd_vvvl(vrsum45_s3, vrsum45_s4, vl), vl) ;
      _vel_vstu_vssl(vrsum45, 4, pGIn+gInIndex+4*gInPixels, vl) ;
      _vel_vstl_vssl(vrsum45, 4, pGIn+gInIndex+5*gInPixels, vl) ;

      vrsum67_s0 = _vel_vex_vvmvl(vrsum67_s0, vmx_s0, _vel_vbrdl_vsl(0UL, vl), vl) ;
      vrsum67_s1 = _vel_vex_vvmvl(vrsum67_s1, vmx_s1, _vel_vbrdl_vsl(0UL, vl), vl) ;
      vrsum67_s2 = _vel_vex_vvmvl(vrsum67_s2, vmx_s2, _vel_vbrdl_vsl(0UL, vl), vl) ;
      vrsum67_s3 = _vel_vex_vvmvl(vrsum67_s3, vmx_s3, _vel_vbrdl_vsl(0UL, vl), vl) ;
      vrsum67_s4 = _vel_vex_vvmvl(vrsum67_s4, vmx_s4, _vel_vbrdl_vsl(0UL, vl), vl) ;
      __vr vrsum67 = _vel_pvfadd_vvvl(_vel_pvfadd_vvvl(vrsum67_s0, _vel_pvfadd_vvvl(vrsum67_s1, vrsum67_s2, vl), vl),
                                      _vel_pvfadd_vvvl(vrsum67_s3, vrsum67_s4, vl), vl) ;
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

      const int64_t xmin_s0 = (w+2) / 2 ;
      const int64_t xmin_s2 = (w+0) / 2 ;
      const int64_t xmin_s4 = w >= 2 ? (w-2) / 2 : 0 ;

      __vr vrw   = _vel_vaddsl_vsvl(w, _vel_vseq_vl(vl), vl) ;

      __vr vrj_s0 = _vel_vaddsl_vsvl(2, vrw, vl) ;
      __vr vrj_s1 = _vel_vaddsl_vsvl(1, vrw, vl) ;
      __vr vrj_s2 = vrw ;
      __vr vrj_s3 = _vel_vaddsl_vsvl(-1, vrw, vl) ;
      __vr vrj_s4 = _vel_vaddsl_vsvl(-2, vrw, vl) ;

      __vr vrx_s0 = _vel_vdivsl_vvsl(vrj_s0, 2, vl) ;
      __vr vrx_s12 = _vel_vdivsl_vvsl(vrj_s1, 2, vl) ;
      __vr vrx_s34 = _vel_vdivsl_vvsl(vrj_s3, 2, vl) ;

      __vm256 vmx0_s0 =  _vel_vfmkleq_mvl(_vel_vcmpsl_vvvl(vrj_s0, _vel_vmulsl_vsvl(2, vrx_s0, vl), vl), vl) ;
      __vm256 vmx1_s0 =  _vel_vfmklge_mvl(vrx_s0, vl) ;
      __vm256 vmx2_s0 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutWidth,vrx_s0, vl), vl) ;
      __vm256 vmx_s0 = _vel_andm_mmm(_vel_andm_mmm(vmx0_s0, vmx1_s0), vmx2_s0) ;

      __vm256 vmx0_s1 =  _vel_vfmkleq_mvl(_vel_vcmpsl_vvvl(vrj_s1, _vel_vmulsl_vsvl(2, vrx_s12, vl), vl), vl) ;
      __vm256 vmx0_s2 =  _vel_vfmkleq_mvl(_vel_vcmpsl_vvvl(vrj_s2, _vel_vmulsl_vsvl(2, vrx_s12, vl), vl), vl) ;
      __vm256 vmx1_s12 =  _vel_vfmklge_mvl(vrx_s12, vl) ;
      __vm256 vmx2_s12 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutWidth,vrx_s12, vl), vl) ;
      __vm256 vmx_s12 = _vel_andm_mmm(vmx1_s12, vmx2_s12) ;
      __vm256 vmx_s1 = _vel_andm_mmm(vmx0_s1, vmx_s12) ;
      __vm256 vmx_s2 = _vel_andm_mmm(vmx0_s2, vmx_s12) ;

      __vm256 vmx0_s3 =  _vel_vfmkleq_mvl(_vel_vcmpsl_vvvl(vrj_s3, _vel_vmulsl_vsvl(2, vrx_s34, vl), vl), vl) ;
      __vm256 vmx0_s4 =  _vel_vfmkleq_mvl(_vel_vcmpsl_vvvl(vrj_s4, _vel_vmulsl_vsvl(2, vrx_s34, vl), vl), vl) ;
      __vm256 vmx1_s34 =  _vel_vfmklge_mvl(vrx_s34, vl) ;
      __vm256 vmx2_s34 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutWidth,vrx_s34, vl), vl) ;
      __vm256 vmx_s34 = _vel_andm_mmm(vmx1_s34, vmx2_s34) ;
      __vm256 vmx_s3 = _vel_andm_mmm(vmx0_s3, vmx_s34) ;
      __vm256 vmx_s4 = _vel_andm_mmm(vmx0_s4, vmx_s34) ;

      __vr vrsum01_s0 = _vel_vbrdl_vsl(0UL, vl/2) ;
      __vr vrsum23_s0 = _vel_vbrdl_vsl(0UL, vl/2) ;
      __vr vrsum45_s0 = _vel_vbrdl_vsl(0UL, vl/2) ;
      __vr vrsum67_s0 = _vel_vbrdl_vsl(0UL, vl/2) ;
      __vr vrsum89_s0 = _vel_vbrdl_vsl(0UL, vl/2) ;
      __vr vrsumAB_s0 = _vel_vbrdl_vsl(0UL, vl/2) ;
      __vr vrsumCD_s0 = _vel_vbrdl_vsl(0UL, vl/2) ;
      __vr vrsumEF_s0 = _vel_vbrdl_vsl(0UL, vl/2) ;

      __vr vrsum01_s1 = _vel_vbrdl_vsl(0UL, vl/2) ;
      __vr vrsum23_s1 = _vel_vbrdl_vsl(0UL, vl/2) ;
      __vr vrsum45_s1 = _vel_vbrdl_vsl(0UL, vl/2) ;
      __vr vrsum67_s1 = _vel_vbrdl_vsl(0UL, vl/2) ;
      __vr vrsum89_s1 = _vel_vbrdl_vsl(0UL, vl/2) ;
      __vr vrsumAB_s1 = _vel_vbrdl_vsl(0UL, vl/2) ;
      __vr vrsumCD_s1 = _vel_vbrdl_vsl(0UL, vl/2) ;
      __vr vrsumEF_s1 = _vel_vbrdl_vsl(0UL, vl/2) ;

      __vr vrsum01_s2 = _vel_vbrdl_vsl(0UL, vl/2) ;
      __vr vrsum23_s2 = _vel_vbrdl_vsl(0UL, vl/2) ;
      __vr vrsum45_s2 = _vel_vbrdl_vsl(0UL, vl/2) ;
      __vr vrsum67_s2 = _vel_vbrdl_vsl(0UL, vl/2) ;
      __vr vrsum89_s2 = _vel_vbrdl_vsl(0UL, vl/2) ;
      __vr vrsumAB_s2 = _vel_vbrdl_vsl(0UL, vl/2) ;
      __vr vrsumCD_s2 = _vel_vbrdl_vsl(0UL, vl/2) ;
      __vr vrsumEF_s2 = _vel_vbrdl_vsl(0UL, vl/2) ;

      __vr vrsum01_s3 = _vel_vbrdl_vsl(0UL, vl/2) ;
      __vr vrsum23_s3 = _vel_vbrdl_vsl(0UL, vl/2) ;
      __vr vrsum45_s3 = _vel_vbrdl_vsl(0UL, vl/2) ;
      __vr vrsum67_s3 = _vel_vbrdl_vsl(0UL, vl/2) ;
      __vr vrsum89_s3 = _vel_vbrdl_vsl(0UL, vl/2) ;
      __vr vrsumAB_s3 = _vel_vbrdl_vsl(0UL, vl/2) ;
      __vr vrsumCD_s3 = _vel_vbrdl_vsl(0UL, vl/2) ;
      __vr vrsumEF_s3 = _vel_vbrdl_vsl(0UL, vl/2) ;

      __vr vrsum01_s4 = _vel_vbrdl_vsl(0UL, vl/2) ;
      __vr vrsum23_s4 = _vel_vbrdl_vsl(0UL, vl/2) ;
      __vr vrsum45_s4 = _vel_vbrdl_vsl(0UL, vl/2) ;
      __vr vrsum67_s4 = _vel_vbrdl_vsl(0UL, vl/2) ;
      __vr vrsum89_s4 = _vel_vbrdl_vsl(0UL, vl/2) ;
      __vr vrsumAB_s4 = _vel_vbrdl_vsl(0UL, vl/2) ;
      __vr vrsumCD_s4 = _vel_vbrdl_vsl(0UL, vl/2) ;
      __vr vrsumEF_s4 = _vel_vbrdl_vsl(0UL, vl/2) ;

      for (int64_t r=0; r<kernHeight; r++) {
	int64_t i = h - r + 2 ;
	int64_t y = i/2;
	if ( y*2 != i || y < 0 || gOutHeight <= y)  continue ;

	for (int64_t k=0; k<gOutChannelGroup; k++) {
	  int64_t gOutIndex    = gOutGroupOffset + ((n * gOutChannel + k) * gOutHeight) * gOutWidth ;
	  const float *pKerValue = pKernel + kernGroupOffset + ((k * gInChannelGroup + c) * kernHeight) * kernWidth;

	  __vr vrgout_s01 = _vel_vldu_vssl(4, pGOut+gOutIndex+gOutWidth*y + xmin_s0 , vl/2) ;
	  __vr vrgout_s23 = _vel_vldu_vssl(4, pGOut+gOutIndex+gOutWidth*y + xmin_s2 , vl/2) ;
	  __vr vrgout_s4  = _vel_vldu_vssl(4, pGOut+gOutIndex+gOutWidth*y + xmin_s4 , vl/2) ;

#define VFMAD_C16(VRGOUT, K, R, S)	{												\
	    const uint64_t kerValue01 = _vel_pack_f32p(pKerValue + (((K)*gInChannelGroup + 0) * kernHeight +(R)) * kernWidth + (S),	\
						       pKerValue + (((K)*gInChannelGroup + 1) * kernHeight +(R)) * kernWidth + (S)) ;	\
	    const uint64_t kerValue23 = _vel_pack_f32p(pKerValue + (((K)*gInChannelGroup + 2) * kernHeight +(R)) * kernWidth + (S),	\
						       pKerValue + (((K)*gInChannelGroup + 3) * kernHeight +(R)) * kernWidth + (S)) ;	\
	    const uint64_t kerValue45 = _vel_pack_f32p(pKerValue + (((K)*gInChannelGroup + 4) * kernHeight +(R)) * kernWidth + (S),	\
						       pKerValue + (((K)*gInChannelGroup + 5) * kernHeight +(R)) * kernWidth + (S)) ;	\
	    const uint64_t kerValue67 = _vel_pack_f32p(pKerValue + (((K)*gInChannelGroup + 6) * kernHeight +(R)) * kernWidth + (S),	\
						       pKerValue + (((K)*gInChannelGroup + 7) * kernHeight +(R)) * kernWidth + (S)) ;	\
	    const uint64_t kerValue89 = _vel_pack_f32p(pKerValue + (((K)*gInChannelGroup + 8) * kernHeight +(R)) * kernWidth + (S),	\
						       pKerValue + (((K)*gInChannelGroup + 9) * kernHeight +(R)) * kernWidth + (S)) ;	\
	    const uint64_t kerValueAB = _vel_pack_f32p(pKerValue + (((K)*gInChannelGroup +10) * kernHeight +(R)) * kernWidth + (S),	\
						       pKerValue + (((K)*gInChannelGroup +11) * kernHeight +(R)) * kernWidth + (S)) ;	\
	    const uint64_t kerValueCD = _vel_pack_f32p(pKerValue + (((K)*gInChannelGroup +12) * kernHeight +(R)) * kernWidth + (S),	\
						       pKerValue + (((K)*gInChannelGroup +13) * kernHeight +(R)) * kernWidth + (S)) ;	\
	    const uint64_t kerValueEF = _vel_pack_f32p(pKerValue + (((K)*gInChannelGroup +14) * kernHeight +(R)) * kernWidth + (S),	\
						       pKerValue + (((K)*gInChannelGroup +15) * kernHeight +(R)) * kernWidth + (S)) ;	\
	    __vr vrgoutP = _vel_vshf_vvvsl(VRGOUT, VRGOUT, VE_VSHUFFLE_YUZU, vl/2) ;		\
	    vrsum01_s##S = _vel_pvfmad_vvsvl(vrsum01_s##S, kerValue01, vrgoutP, vl/2) ;		\
	    vrsum23_s##S = _vel_pvfmad_vvsvl(vrsum23_s##S, kerValue23, vrgoutP, vl/2) ;		\
	    vrsum45_s##S = _vel_pvfmad_vvsvl(vrsum45_s##S, kerValue45, vrgoutP, vl/2) ;		\
	    vrsum67_s##S = _vel_pvfmad_vvsvl(vrsum67_s##S, kerValue67, vrgoutP, vl/2) ;		\
	    vrsum89_s##S = _vel_pvfmad_vvsvl(vrsum89_s##S, kerValue89, vrgoutP, vl/2) ;		\
	    vrsumAB_s##S = _vel_pvfmad_vvsvl(vrsumAB_s##S, kerValueAB, vrgoutP, vl/2) ;		\
	    vrsumCD_s##S = _vel_pvfmad_vvsvl(vrsumCD_s##S, kerValueCD, vrgoutP, vl/2) ;		\
	    vrsumEF_s##S = _vel_pvfmad_vvsvl(vrsumEF_s##S, kerValueEF, vrgoutP, vl/2) ;		\
}

	  VFMAD_C16(vrgout_s01, 0, r, 0) ;
	  VFMAD_C16(vrgout_s01, 0, r, 1) ;
	  VFMAD_C16(vrgout_s23, 0, r, 2) ;
	  VFMAD_C16(vrgout_s23, 0, r, 3) ;
	  VFMAD_C16(vrgout_s4,  0, r, 4) ;

#undef VFMAD_C16
	}

      } // kernHeight

       ;
      vrsum01_s0 = _vel_vex_vvmvl(vrsum01_s0, vmx_s0, _vel_vbrdl_vsl(0UL, vl), vl) ;
      vrsum01_s1 = _vel_vex_vvmvl(vrsum01_s1, vmx_s1, _vel_vbrdl_vsl(0UL, vl), vl) ;
      vrsum01_s2 = _vel_vex_vvmvl(vrsum01_s2, vmx_s2, _vel_vbrdl_vsl(0UL, vl), vl) ;
      vrsum01_s3 = _vel_vex_vvmvl(vrsum01_s3, vmx_s3, _vel_vbrdl_vsl(0UL, vl), vl) ;
      vrsum01_s4 = _vel_vex_vvmvl(vrsum01_s4, vmx_s4, _vel_vbrdl_vsl(0UL, vl), vl) ;
      __vr vrsum01 = _vel_pvfadd_vvvl(_vel_pvfadd_vvvl(vrsum01_s0, _vel_pvfadd_vvvl(vrsum01_s1, vrsum01_s2, vl), vl),
                                      _vel_pvfadd_vvvl(vrsum01_s3, vrsum01_s4, vl), vl) ;
      _vel_vstu_vssl(vrsum01, 4, pGIn+gInIndex, vl) ;
      _vel_vstl_vssl(vrsum01, 4, pGIn+gInIndex+  gInPixels, vl) ;

      vrsum23_s0 = _vel_vex_vvmvl(vrsum23_s0, vmx_s0, _vel_vbrdl_vsl(0UL, vl), vl) ;
      vrsum23_s1 = _vel_vex_vvmvl(vrsum23_s1, vmx_s1, _vel_vbrdl_vsl(0UL, vl), vl) ;
      vrsum23_s2 = _vel_vex_vvmvl(vrsum23_s2, vmx_s2, _vel_vbrdl_vsl(0UL, vl), vl) ;
      vrsum23_s3 = _vel_vex_vvmvl(vrsum23_s3, vmx_s3, _vel_vbrdl_vsl(0UL, vl), vl) ;
      vrsum23_s4 = _vel_vex_vvmvl(vrsum23_s4, vmx_s4, _vel_vbrdl_vsl(0UL, vl), vl) ;
      __vr vrsum23 = _vel_pvfadd_vvvl(_vel_pvfadd_vvvl(vrsum23_s0, _vel_pvfadd_vvvl(vrsum23_s1, vrsum23_s2, vl), vl),
                                      _vel_pvfadd_vvvl(vrsum23_s3, vrsum23_s4, vl), vl) ;
      _vel_vstu_vssl(vrsum23, 4, pGIn+gInIndex+2*gInPixels, vl) ;
      _vel_vstl_vssl(vrsum23, 4, pGIn+gInIndex+3*gInPixels, vl) ;

      vrsum45_s0 = _vel_vex_vvmvl(vrsum45_s0, vmx_s0, _vel_vbrdl_vsl(0UL, vl), vl) ;
      vrsum45_s1 = _vel_vex_vvmvl(vrsum45_s1, vmx_s1, _vel_vbrdl_vsl(0UL, vl), vl) ;
      vrsum45_s2 = _vel_vex_vvmvl(vrsum45_s2, vmx_s2, _vel_vbrdl_vsl(0UL, vl), vl) ;
      vrsum45_s3 = _vel_vex_vvmvl(vrsum45_s3, vmx_s3, _vel_vbrdl_vsl(0UL, vl), vl) ;
      vrsum45_s4 = _vel_vex_vvmvl(vrsum45_s4, vmx_s4, _vel_vbrdl_vsl(0UL, vl), vl) ;
      __vr vrsum45 = _vel_pvfadd_vvvl(_vel_pvfadd_vvvl(vrsum45_s0, _vel_pvfadd_vvvl(vrsum45_s1, vrsum45_s2, vl), vl),
                                      _vel_pvfadd_vvvl(vrsum45_s3, vrsum45_s4, vl), vl) ;
      _vel_vstu_vssl(vrsum45, 4, pGIn+gInIndex+4*gInPixels, vl) ;
      _vel_vstl_vssl(vrsum45, 4, pGIn+gInIndex+5*gInPixels, vl) ;

      vrsum67_s0 = _vel_vex_vvmvl(vrsum67_s0, vmx_s0, _vel_vbrdl_vsl(0UL, vl), vl) ;
      vrsum67_s1 = _vel_vex_vvmvl(vrsum67_s1, vmx_s1, _vel_vbrdl_vsl(0UL, vl), vl) ;
      vrsum67_s2 = _vel_vex_vvmvl(vrsum67_s2, vmx_s2, _vel_vbrdl_vsl(0UL, vl), vl) ;
      vrsum67_s3 = _vel_vex_vvmvl(vrsum67_s3, vmx_s3, _vel_vbrdl_vsl(0UL, vl), vl) ;
      vrsum67_s4 = _vel_vex_vvmvl(vrsum67_s4, vmx_s4, _vel_vbrdl_vsl(0UL, vl), vl) ;
      __vr vrsum67 = _vel_pvfadd_vvvl(_vel_pvfadd_vvvl(vrsum67_s0, _vel_pvfadd_vvvl(vrsum67_s1, vrsum67_s2, vl), vl),
                                      _vel_pvfadd_vvvl(vrsum67_s3, vrsum67_s4, vl), vl) ;
      _vel_vstu_vssl(vrsum67, 4, pGIn+gInIndex+6*gInPixels, vl) ;
      _vel_vstl_vssl(vrsum67, 4, pGIn+gInIndex+7*gInPixels, vl) ;

      vrsum89_s0 = _vel_vex_vvmvl(vrsum89_s0, vmx_s0, _vel_vbrdl_vsl(0UL, vl), vl) ;
      vrsum89_s1 = _vel_vex_vvmvl(vrsum89_s1, vmx_s1, _vel_vbrdl_vsl(0UL, vl), vl) ;
      vrsum89_s2 = _vel_vex_vvmvl(vrsum89_s2, vmx_s2, _vel_vbrdl_vsl(0UL, vl), vl) ;
      vrsum89_s3 = _vel_vex_vvmvl(vrsum89_s3, vmx_s3, _vel_vbrdl_vsl(0UL, vl), vl) ;
      vrsum89_s4 = _vel_vex_vvmvl(vrsum89_s4, vmx_s4, _vel_vbrdl_vsl(0UL, vl), vl) ;
      __vr vrsum89 = _vel_pvfadd_vvvl(_vel_pvfadd_vvvl(vrsum89_s0, _vel_pvfadd_vvvl(vrsum89_s1, vrsum89_s2, vl), vl),
                                      _vel_pvfadd_vvvl(vrsum89_s3, vrsum89_s4, vl), vl) ;
      _vel_vstu_vssl(vrsum89, 4, pGIn+gInIndex+8*gInPixels, vl) ;
      _vel_vstl_vssl(vrsum89, 4, pGIn+gInIndex+9*gInPixels, vl) ;

      vrsumAB_s0 = _vel_vex_vvmvl(vrsumAB_s0, vmx_s0, _vel_vbrdl_vsl(0UL, vl), vl) ;
      vrsumAB_s1 = _vel_vex_vvmvl(vrsumAB_s1, vmx_s1, _vel_vbrdl_vsl(0UL, vl), vl) ;
      vrsumAB_s2 = _vel_vex_vvmvl(vrsumAB_s2, vmx_s2, _vel_vbrdl_vsl(0UL, vl), vl) ;
      vrsumAB_s3 = _vel_vex_vvmvl(vrsumAB_s3, vmx_s3, _vel_vbrdl_vsl(0UL, vl), vl) ;
      vrsumAB_s4 = _vel_vex_vvmvl(vrsumAB_s4, vmx_s4, _vel_vbrdl_vsl(0UL, vl), vl) ;
      __vr vrsumAB = _vel_pvfadd_vvvl(_vel_pvfadd_vvvl(vrsumAB_s0, _vel_pvfadd_vvvl(vrsumAB_s1, vrsumAB_s2, vl), vl),
                                      _vel_pvfadd_vvvl(vrsumAB_s3, vrsumAB_s4, vl), vl) ;
      _vel_vstu_vssl(vrsumAB, 4, pGIn+gInIndex+10*gInPixels, vl) ;
      _vel_vstl_vssl(vrsumAB, 4, pGIn+gInIndex+11*gInPixels, vl) ;

      vrsumCD_s0 = _vel_vex_vvmvl(vrsumCD_s0, vmx_s0, _vel_vbrdl_vsl(0UL, vl), vl) ;
      vrsumCD_s1 = _vel_vex_vvmvl(vrsumCD_s1, vmx_s1, _vel_vbrdl_vsl(0UL, vl), vl) ;
      vrsumCD_s2 = _vel_vex_vvmvl(vrsumCD_s2, vmx_s2, _vel_vbrdl_vsl(0UL, vl), vl) ;
      vrsumCD_s3 = _vel_vex_vvmvl(vrsumCD_s3, vmx_s3, _vel_vbrdl_vsl(0UL, vl), vl) ;
      vrsumCD_s4 = _vel_vex_vvmvl(vrsumCD_s4, vmx_s4, _vel_vbrdl_vsl(0UL, vl), vl) ;
      __vr vrsumCD = _vel_pvfadd_vvvl(_vel_pvfadd_vvvl(vrsumCD_s0, _vel_pvfadd_vvvl(vrsumCD_s1, vrsumCD_s2, vl), vl),
                                      _vel_pvfadd_vvvl(vrsumCD_s3, vrsumCD_s4, vl), vl) ;
      _vel_vstu_vssl(vrsumCD, 4, pGIn+gInIndex+12*gInPixels, vl) ;
      _vel_vstl_vssl(vrsumCD, 4, pGIn+gInIndex+13*gInPixels, vl) ;

      vrsumEF_s0 = _vel_vex_vvmvl(vrsumEF_s0, vmx_s0, _vel_vbrdl_vsl(0UL, vl), vl) ;
      vrsumEF_s1 = _vel_vex_vvmvl(vrsumEF_s1, vmx_s1, _vel_vbrdl_vsl(0UL, vl), vl) ;
      vrsumEF_s2 = _vel_vex_vvmvl(vrsumEF_s2, vmx_s2, _vel_vbrdl_vsl(0UL, vl), vl) ;
      vrsumEF_s3 = _vel_vex_vvmvl(vrsumEF_s3, vmx_s3, _vel_vbrdl_vsl(0UL, vl), vl) ;
      vrsumEF_s4 = _vel_vex_vvmvl(vrsumEF_s4, vmx_s4, _vel_vbrdl_vsl(0UL, vl), vl) ;
      __vr vrsumEF = _vel_pvfadd_vvvl(_vel_pvfadd_vvvl(vrsumEF_s0, _vel_pvfadd_vvvl(vrsumEF_s1, vrsumEF_s2, vl), vl),
                                      _vel_pvfadd_vvvl(vrsumEF_s3, vrsumEF_s4, vl), vl) ;
      _vel_vstu_vssl(vrsumEF, 4, pGIn+gInIndex+14*gInPixels, vl) ;
      _vel_vstl_vssl(vrsumEF, 4, pGIn+gInIndex+15*gInPixels, vl) ;

    } // gInWidth
  } // gInHeight
}


vednnError_t
vednnConvolutionBackwardData_direct_dil1_str2_pad2_ker5(
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
//  const int64_t strideWidth    = pParamConv->strideWidth;;	// must be 2
//  const int64_t strideHeight   = pParamConv->strideHeight;	// must be 2
//  const int64_t padWidth       = pParamConv->padWidth;	// must be 2
//  const int64_t padHeight      = pParamConv->padHeight;	// must be 2
//  const int64_t dilationWidth  = pParamConv->dilationWidth;	// must be 1
//  const int64_t dilationHeight = pParamConv->dilationHeight;	// must be 1

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
