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
    const int64_t gInChannelGroup,
    const int64_t gOutChannelGroup,
    const int64_t gInGroupOffset,
    const int64_t gOutGroupOffset,
    const int64_t kernGroupOffset,
    const int64_t gInPixels,
    const int64_t n,
    const int64_t c,
    const int64_t nH,
    const __vr vrh,
    const __vr vrw
)
{

  for (int64_t h=0; h<gInHeight; h+=nH) {
    const int64_t vl = gInWidth * (gInHeight - h < nH ? gInHeight - h : nH) ;
    const int64_t gip = h * gInWidth ;

    const int64_t gInIndex = gInGroupOffset + ((n * gInChannel + c) * gInHeight ) * gInWidth + gip ;

    _ve_lvl(vl) ;

    __vr vrsum_s0 = _ve_vbrdu_vs_f32(0.f) ;
    __vr vrsum_s1 = _ve_vbrdu_vs_f32(0.f) ;
    __vr vrsum_s2 = _ve_vbrdu_vs_f32(0.f) ;
    __vr vrsum_s3 = _ve_vbrdu_vs_f32(0.f) ;
    __vr vrsum_s4 = _ve_vbrdu_vs_f32(0.f) ;

    __vr vrj_s0 = _ve_vaddsl_vsv( 2, vrw) ;
    __vr vrj_s1 = _ve_vaddsl_vsv( 1, vrw) ;
    __vr vrj_s2 = _ve_vaddsl_vsv( 0, vrw) ;
    __vr vrj_s3 = _ve_vaddsl_vsv(-1, vrw) ;
    __vr vrj_s4 = _ve_vaddsl_vsv(-2, vrw) ;

    __vr vrx_s0 = _ve_vdivsl_vvs(vrj_s0, 2) ;
    __vr vrx_s1 = _ve_vdivsl_vvs(vrj_s1, 2) ;
    __vr vrx_s2 = _ve_vdivsl_vvs(vrj_s2, 2) ;
    __vr vrx_s3 = _ve_vdivsl_vvs(vrj_s3, 2) ;
    __vr vrx_s4 = _ve_vdivsl_vvs(vrj_s4, 2) ;

    __vm256 vmx0_s0 = _ve_vfmkl_mcv(VECC_IEQ, _ve_vcmpsl_vvv(vrj_s0, _ve_vmulsl_vsv(2, vrx_s0))) ;
    __vm256 vmx1_s0 = _ve_vfmkl_mcv(VECC_GE, vrx_s0) ;
    __vm256 vmx2_s0 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(gOutWidth,vrx_s0)) ;
    __vm256 vmx_s0 = _ve_andm_mmm(_ve_andm_mmm(vmx0_s0, vmx1_s0), vmx2_s0) ;

    __vm256 vmx0_s1 = _ve_vfmkl_mcv(VECC_IEQ, _ve_vcmpsl_vvv(vrj_s1, _ve_vmulsl_vsv(2, vrx_s1))) ;
    __vm256 vmx1_s1 = _ve_vfmkl_mcv(VECC_GE, vrx_s1) ;
    __vm256 vmx2_s1 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(gOutWidth,vrx_s1)) ;
    __vm256 vmx_s1 = _ve_andm_mmm(_ve_andm_mmm(vmx0_s1, vmx1_s1), vmx2_s1) ;

    __vm256 vmx0_s2 = _ve_vfmkl_mcv(VECC_IEQ, _ve_vcmpsl_vvv(vrj_s2, _ve_vmulsl_vsv(2, vrx_s2))) ;
    __vm256 vmx1_s2 = _ve_vfmkl_mcv(VECC_GE, vrx_s2) ;
    __vm256 vmx2_s2 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(gOutWidth,vrx_s2)) ;
    __vm256 vmx_s2 = _ve_andm_mmm(_ve_andm_mmm(vmx0_s2, vmx1_s2), vmx2_s2) ;

    __vm256 vmx0_s3 = _ve_vfmkl_mcv(VECC_IEQ, _ve_vcmpsl_vvv(vrj_s3, _ve_vmulsl_vsv(2, vrx_s3))) ;
    __vm256 vmx1_s3 = _ve_vfmkl_mcv(VECC_GE, vrx_s3) ;
    __vm256 vmx2_s3 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(gOutWidth,vrx_s3)) ;
    __vm256 vmx_s3 = _ve_andm_mmm(_ve_andm_mmm(vmx0_s3, vmx1_s3), vmx2_s3) ;

    __vm256 vmx0_s4 = _ve_vfmkl_mcv(VECC_IEQ, _ve_vcmpsl_vvv(vrj_s4, _ve_vmulsl_vsv(2, vrx_s4))) ;
    __vm256 vmx1_s4 = _ve_vfmkl_mcv(VECC_GE, vrx_s4) ;
    __vm256 vmx2_s4 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(gOutWidth,vrx_s4)) ;
    __vm256 vmx_s4 = _ve_andm_mmm(_ve_andm_mmm(vmx0_s4, vmx1_s4), vmx2_s4) ;

    for (int64_t r=0; r<kernHeight; r++) {
      __vr vri = _ve_vaddsl_vsv(2-r+h, vrh) ;
      __vr vry = _ve_vdivsl_vvs(vri, 2) ;

      __vm256 vmy0 = _ve_vfmkl_mcv(VECC_IEQ, _ve_vcmpsl_vvv(vri, _ve_vmulsl_vsv(2, vry))) ;
      __vm256 vmy1 = _ve_vfmkl_mcv(VECC_GE, vry) ;
      __vm256 vmy2 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(gOutHeight,vry)) ;
      __vm256 vmy = _ve_andm_mmm(_ve_andm_mmm(vmy0, vmy1), vmy2) ;

      int64_t k=0;
      for (; k<gOutChannelGroup; k++) {
	int64_t gOutIndex    = gOutGroupOffset + ((n * gOutChannel + k) * gOutHeight) * gOutWidth ;
	const float *pKerValue = pKernel + kernGroupOffset + (((k  ) * gInChannelGroup + c) * kernHeight) * kernWidth ;

#define VFADD_C1(VRGOUT, K, R, S)  {												\
	  const float kerValue = pKerValue[(((K)*gInChannelGroup + 0) * kernHeight +(R)) * kernWidth + (S) ] ;	\
	  vrsum_s##S = _ve_vfmads_vvsv(vrsum_s##S, kerValue, VRGOUT) ;	\
	}

	__vr vrgout_ptr_k0_s2 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry), vrx_s2),
					   2,
					   (unsigned long)(pGOut+gOutIndex)) ;
	__vr vrgout_k0_s2 = _ve_vgtu_vvm(vrgout_ptr_k0_s2, vmy) ;
	vrgout_k0_s2 = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), vrgout_k0_s2, vmy) ;

	VFADD_C1(vrgout_k0_s2, 0, r, 0)
	VFADD_C1(vrgout_k0_s2, 0, r, 1)
	VFADD_C1(vrgout_k0_s2, 0, r, 2)
	VFADD_C1(vrgout_k0_s2, 0, r, 3)
	VFADD_C1(vrgout_k0_s2, 0, r, 4)
#undef VFADD_C2
      }
    } // kernHeight

    vrsum_s0 = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.f), _ve_vmv_vsv( 2, vrsum_s0), vmx_s0) ;
    vrsum_s1 = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.f), _ve_vmv_vsv( 1, vrsum_s1), vmx_s1) ;
    vrsum_s2 = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.f),                 vrsum_s2, vmx_s2) ;
    vrsum_s3 = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.f), _ve_vmv_vsv(-1, vrsum_s3), vmx_s3) ;
    vrsum_s4 = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.f), _ve_vmv_vsv(-2, vrsum_s4), vmx_s4) ;
    __vr vrsum = _ve_vfadds_vvv(_ve_vfadds_vvv(vrsum_s0, _ve_vfadds_vvv(vrsum_s1, vrsum_s2)),
                                _ve_vfadds_vvv(vrsum_s3, vrsum_s4)) ;
    _ve_vstu_vss(vrsum, 4, pGIn+gInIndex) ;
  } // gOutPixels
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
    const int64_t c,
    const int64_t nH,
    const __vr vrh,
    const __vr vrw
)
{

  for (int64_t h=0; h<gInHeight; h+=nH) {
    const int64_t vl = gInWidth * (gInHeight - h < nH ? gInHeight - h : nH) ;
    const int64_t gip = h * gInWidth ;

    const int64_t gInIndex = gInGroupOffset + ((n * gInChannel + c) * gInHeight ) * gInWidth + gip ;

    _ve_lvl(vl) ;

    __vr vrsum01_s0 = _ve_vbrd_vs_i64(0UL) ;
    __vr vrsum01_s1 = _ve_vbrd_vs_i64(0UL) ;
    __vr vrsum01_s2 = _ve_vbrd_vs_i64(0UL) ;
    __vr vrsum01_s3 = _ve_vbrd_vs_i64(0UL) ;
    __vr vrsum01_s4 = _ve_vbrd_vs_i64(0UL) ;

    __vr vrj_s0 = _ve_vaddsl_vsv( 2, vrw) ;
    __vr vrj_s1 = _ve_vaddsl_vsv( 1, vrw) ;
    __vr vrj_s2 = _ve_vaddsl_vsv( 0, vrw) ;
    __vr vrj_s3 = _ve_vaddsl_vsv(-1, vrw) ;
    __vr vrj_s4 = _ve_vaddsl_vsv(-2, vrw) ;

    __vr vrx_s0 = _ve_vdivsl_vvs(vrj_s0, 2) ;
    __vr vrx_s1 = _ve_vdivsl_vvs(vrj_s1, 2) ;
    __vr vrx_s2 = _ve_vdivsl_vvs(vrj_s2, 2) ;
    __vr vrx_s3 = _ve_vdivsl_vvs(vrj_s3, 2) ;
    __vr vrx_s4 = _ve_vdivsl_vvs(vrj_s4, 2) ;

    __vm256 vmx0_s0 = _ve_vfmkl_mcv(VECC_IEQ, _ve_vcmpsl_vvv(vrj_s0, _ve_vmulsl_vsv(2, vrx_s0))) ;
    __vm256 vmx1_s0 = _ve_vfmkl_mcv(VECC_GE, vrx_s0) ;
    __vm256 vmx2_s0 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(gOutWidth,vrx_s0)) ;
    __vm256 vmx_s0 = _ve_andm_mmm(_ve_andm_mmm(vmx0_s0, vmx1_s0), vmx2_s0) ;

    __vm256 vmx0_s1 = _ve_vfmkl_mcv(VECC_IEQ, _ve_vcmpsl_vvv(vrj_s1, _ve_vmulsl_vsv(2, vrx_s1))) ;
    __vm256 vmx1_s1 = _ve_vfmkl_mcv(VECC_GE, vrx_s1) ;
    __vm256 vmx2_s1 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(gOutWidth,vrx_s1)) ;
    __vm256 vmx_s1 = _ve_andm_mmm(_ve_andm_mmm(vmx0_s1, vmx1_s1), vmx2_s1) ;

    __vm256 vmx0_s2 = _ve_vfmkl_mcv(VECC_IEQ, _ve_vcmpsl_vvv(vrj_s2, _ve_vmulsl_vsv(2, vrx_s2))) ;
    __vm256 vmx1_s2 = _ve_vfmkl_mcv(VECC_GE, vrx_s2) ;
    __vm256 vmx2_s2 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(gOutWidth,vrx_s2)) ;
    __vm256 vmx_s2 = _ve_andm_mmm(_ve_andm_mmm(vmx0_s2, vmx1_s2), vmx2_s2) ;

    __vm256 vmx0_s3 = _ve_vfmkl_mcv(VECC_IEQ, _ve_vcmpsl_vvv(vrj_s3, _ve_vmulsl_vsv(2, vrx_s3))) ;
    __vm256 vmx1_s3 = _ve_vfmkl_mcv(VECC_GE, vrx_s3) ;
    __vm256 vmx2_s3 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(gOutWidth,vrx_s3)) ;
    __vm256 vmx_s3 = _ve_andm_mmm(_ve_andm_mmm(vmx0_s3, vmx1_s3), vmx2_s3) ;

    __vm256 vmx0_s4 = _ve_vfmkl_mcv(VECC_IEQ, _ve_vcmpsl_vvv(vrj_s4, _ve_vmulsl_vsv(2, vrx_s4))) ;
    __vm256 vmx1_s4 = _ve_vfmkl_mcv(VECC_GE, vrx_s4) ;
    __vm256 vmx2_s4 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(gOutWidth,vrx_s4)) ;
    __vm256 vmx_s4 = _ve_andm_mmm(_ve_andm_mmm(vmx0_s4, vmx1_s4), vmx2_s4) ;

    for (int64_t r=0; r<kernHeight; r++) {
      __vr vri = _ve_vaddsl_vsv(2-r+h, vrh) ;
      __vr vry = _ve_vdivsl_vvs(vri, 2) ;

      __vm256 vmy0 = _ve_vfmkl_mcv(VECC_IEQ, _ve_vcmpsl_vvv(vri, _ve_vmulsl_vsv(2, vry))) ;
      __vm256 vmy1 = _ve_vfmkl_mcv(VECC_GE, vry) ;
      __vm256 vmy2 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(gOutHeight,vry)) ;
      __vm256 vmy = _ve_andm_mmm(_ve_andm_mmm(vmy0, vmy1), vmy2) ;

      int64_t k=0;
      for (; k<gOutChannelGroup; k++) {
	int64_t gOutIndex    = gOutGroupOffset + ((n * gOutChannel + k) * gOutHeight) * gOutWidth ;
	const float *pKerValue = pKernel + kernGroupOffset + (((k  ) * gInChannelGroup + c) * kernHeight) * kernWidth ;

#define VFADD_C2(VRGOUT, K, R, S)  {												\
	  const uint64_t kerValue01 = _ve_pack_f32p(pKerValue + (((K)*gInChannelGroup + 0) * kernHeight +(R)) * kernWidth + (S),	\
						    pKerValue + (((K)*gInChannelGroup + 1) * kernHeight +(R)) * kernWidth + (S)) ;	\
	  __vr vrgoutP = _ve_vshf_vvvs(VRGOUT, VRGOUT, VE_VSHUFFLE_YUZU) ;	\
	  vrsum01_s##S = _ve_pvfmad_vvsv(vrsum01_s##S, kerValue01, vrgoutP) ;	\
	}

	__vr vrgout_ptr_k0_s2 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry), vrx_s2),
					   2,
					   (unsigned long)(pGOut+gOutIndex)) ;
	__vr vrgout_k0_s2 = _ve_vgtu_vvm(vrgout_ptr_k0_s2, vmy) ;
	vrgout_k0_s2 = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), vrgout_k0_s2, vmy) ;

	VFADD_C2(vrgout_k0_s2, 0, r, 0)
	VFADD_C2(vrgout_k0_s2, 0, r, 1)
	VFADD_C2(vrgout_k0_s2, 0, r, 2)
	VFADD_C2(vrgout_k0_s2, 0, r, 3)
	VFADD_C2(vrgout_k0_s2, 0, r, 4)
#undef VFADD_C2
      }
    } // kernHeight

    vrsum01_s0 = _ve_vmrg_vvvm(_ve_vbrd_vs_i64(0UL), _ve_vmv_vsv( 2, vrsum01_s0), vmx_s0) ;
    vrsum01_s1 = _ve_vmrg_vvvm(_ve_vbrd_vs_i64(0UL), _ve_vmv_vsv( 1, vrsum01_s1), vmx_s1) ;
    vrsum01_s2 = _ve_vmrg_vvvm(_ve_vbrd_vs_i64(0UL),                 vrsum01_s2, vmx_s2) ;
    vrsum01_s3 = _ve_vmrg_vvvm(_ve_vbrd_vs_i64(0UL), _ve_vmv_vsv(-1, vrsum01_s3), vmx_s3) ;
    vrsum01_s4 = _ve_vmrg_vvvm(_ve_vbrd_vs_i64(0UL), _ve_vmv_vsv(-2, vrsum01_s4), vmx_s4) ;
    __vr vrsum01 = _ve_pvfadd_vvv(_ve_pvfadd_vvv(vrsum01_s0, _ve_pvfadd_vvv(vrsum01_s1, vrsum01_s2)),
                                  _ve_pvfadd_vvv(vrsum01_s3, vrsum01_s4)) ;
    _ve_vstu_vss(vrsum01, 4, pGIn+gInIndex) ;
    _ve_vstl_vss(vrsum01, 4, pGIn+gInIndex+  gInPixels) ;
  } // gOutPixels
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
    const int64_t c,
    const int64_t nH,
    const __vr vrh,
    const __vr vrw
)
{

  for (int64_t h=0; h<gInHeight; h+=nH) {
    const int64_t vl = gInWidth * (gInHeight - h < nH ? gInHeight - h : nH) ;
    const int64_t gip = h * gInWidth ;

    const int64_t gInIndex = gInGroupOffset + ((n * gInChannel + c) * gInHeight ) * gInWidth + gip ;

    _ve_lvl(vl) ;

    __vr vrsum01_s0 = _ve_vbrd_vs_i64(0UL) ;
    __vr vrsum23_s0 = _ve_vbrd_vs_i64(0UL) ;

    __vr vrsum01_s1 = _ve_vbrd_vs_i64(0UL) ;
    __vr vrsum23_s1 = _ve_vbrd_vs_i64(0UL) ;

    __vr vrsum01_s2 = _ve_vbrd_vs_i64(0UL) ;
    __vr vrsum23_s2 = _ve_vbrd_vs_i64(0UL) ;

    __vr vrsum01_s3 = _ve_vbrd_vs_i64(0UL) ;
    __vr vrsum23_s3 = _ve_vbrd_vs_i64(0UL) ;

    __vr vrsum01_s4 = _ve_vbrd_vs_i64(0UL) ;
    __vr vrsum23_s4 = _ve_vbrd_vs_i64(0UL) ;

    __vr vrj_s0 = _ve_vaddsl_vsv( 2, vrw) ;
    __vr vrj_s1 = _ve_vaddsl_vsv( 1, vrw) ;
    __vr vrj_s2 = _ve_vaddsl_vsv( 0, vrw) ;
    __vr vrj_s3 = _ve_vaddsl_vsv(-1, vrw) ;
    __vr vrj_s4 = _ve_vaddsl_vsv(-2, vrw) ;

    __vr vrx_s0 = _ve_vdivsl_vvs(vrj_s0, 2) ;
    __vr vrx_s1 = _ve_vdivsl_vvs(vrj_s1, 2) ;
    __vr vrx_s2 = _ve_vdivsl_vvs(vrj_s2, 2) ;
    __vr vrx_s3 = _ve_vdivsl_vvs(vrj_s3, 2) ;
    __vr vrx_s4 = _ve_vdivsl_vvs(vrj_s4, 2) ;

    __vm256 vmx0_s0 = _ve_vfmkl_mcv(VECC_IEQ, _ve_vcmpsl_vvv(vrj_s0, _ve_vmulsl_vsv(2, vrx_s0))) ;
    __vm256 vmx1_s0 = _ve_vfmkl_mcv(VECC_GE, vrx_s0) ;
    __vm256 vmx2_s0 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(gOutWidth,vrx_s0)) ;
    __vm256 vmx_s0 = _ve_andm_mmm(_ve_andm_mmm(vmx0_s0, vmx1_s0), vmx2_s0) ;

    __vm256 vmx0_s1 = _ve_vfmkl_mcv(VECC_IEQ, _ve_vcmpsl_vvv(vrj_s1, _ve_vmulsl_vsv(2, vrx_s1))) ;
    __vm256 vmx1_s1 = _ve_vfmkl_mcv(VECC_GE, vrx_s1) ;
    __vm256 vmx2_s1 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(gOutWidth,vrx_s1)) ;
    __vm256 vmx_s1 = _ve_andm_mmm(_ve_andm_mmm(vmx0_s1, vmx1_s1), vmx2_s1) ;

    __vm256 vmx0_s2 = _ve_vfmkl_mcv(VECC_IEQ, _ve_vcmpsl_vvv(vrj_s2, _ve_vmulsl_vsv(2, vrx_s2))) ;
    __vm256 vmx1_s2 = _ve_vfmkl_mcv(VECC_GE, vrx_s2) ;
    __vm256 vmx2_s2 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(gOutWidth,vrx_s2)) ;
    __vm256 vmx_s2 = _ve_andm_mmm(_ve_andm_mmm(vmx0_s2, vmx1_s2), vmx2_s2) ;

    __vm256 vmx0_s3 = _ve_vfmkl_mcv(VECC_IEQ, _ve_vcmpsl_vvv(vrj_s3, _ve_vmulsl_vsv(2, vrx_s3))) ;
    __vm256 vmx1_s3 = _ve_vfmkl_mcv(VECC_GE, vrx_s3) ;
    __vm256 vmx2_s3 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(gOutWidth,vrx_s3)) ;
    __vm256 vmx_s3 = _ve_andm_mmm(_ve_andm_mmm(vmx0_s3, vmx1_s3), vmx2_s3) ;

    __vm256 vmx0_s4 = _ve_vfmkl_mcv(VECC_IEQ, _ve_vcmpsl_vvv(vrj_s4, _ve_vmulsl_vsv(2, vrx_s4))) ;
    __vm256 vmx1_s4 = _ve_vfmkl_mcv(VECC_GE, vrx_s4) ;
    __vm256 vmx2_s4 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(gOutWidth,vrx_s4)) ;
    __vm256 vmx_s4 = _ve_andm_mmm(_ve_andm_mmm(vmx0_s4, vmx1_s4), vmx2_s4) ;

    for (int64_t r=0; r<kernHeight; r++) {
      __vr vri = _ve_vaddsl_vsv(2-r+h, vrh) ;
      __vr vry = _ve_vdivsl_vvs(vri, 2) ;

      __vm256 vmy0 = _ve_vfmkl_mcv(VECC_IEQ, _ve_vcmpsl_vvv(vri, _ve_vmulsl_vsv(2, vry))) ;
      __vm256 vmy1 = _ve_vfmkl_mcv(VECC_GE, vry) ;
      __vm256 vmy2 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(gOutHeight,vry)) ;
      __vm256 vmy = _ve_andm_mmm(_ve_andm_mmm(vmy0, vmy1), vmy2) ;

      int64_t k=0;
      for (; k<gOutChannelGroup; k++) {
	int64_t gOutIndex    = gOutGroupOffset + ((n * gOutChannel + k) * gOutHeight) * gOutWidth ;
	const float *pKerValue = pKernel + kernGroupOffset + (((k  ) * gInChannelGroup + c) * kernHeight) * kernWidth ;

#define VFADD_C4(VRGOUT, K, R, S)  {												\
	  const uint64_t kerValue01 = _ve_pack_f32p(pKerValue + (((K)*gInChannelGroup + 0) * kernHeight +(R)) * kernWidth + (S),	\
						    pKerValue + (((K)*gInChannelGroup + 1) * kernHeight +(R)) * kernWidth + (S)) ;	\
	  const uint64_t kerValue23 = _ve_pack_f32p(pKerValue + (((K)*gInChannelGroup + 2) * kernHeight +(R)) * kernWidth + (S),	\
						    pKerValue + (((K)*gInChannelGroup + 3) * kernHeight +(R)) * kernWidth + (S)) ;	\
	  __vr vrgoutP = _ve_vshf_vvvs(VRGOUT, VRGOUT, VE_VSHUFFLE_YUZU) ;	\
	  vrsum01_s##S = _ve_pvfmad_vvsv(vrsum01_s##S, kerValue01, vrgoutP) ;	\
	  vrsum23_s##S = _ve_pvfmad_vvsv(vrsum23_s##S, kerValue23, vrgoutP) ;	\
	}

	__vr vrgout_ptr_k0_s2 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry), vrx_s2),
					   2,
					   (unsigned long)(pGOut+gOutIndex)) ;
	__vr vrgout_k0_s2 = _ve_vgtu_vvm(vrgout_ptr_k0_s2, vmy) ;
	vrgout_k0_s2 = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), vrgout_k0_s2, vmy) ;

	VFADD_C4(vrgout_k0_s2, 0, r, 0)
	VFADD_C4(vrgout_k0_s2, 0, r, 1)
	VFADD_C4(vrgout_k0_s2, 0, r, 2)
	VFADD_C4(vrgout_k0_s2, 0, r, 3)
	VFADD_C4(vrgout_k0_s2, 0, r, 4)
#undef VFADD_C4
      }
    } // kernHeight

    vrsum01_s0 = _ve_vmrg_vvvm(_ve_vbrd_vs_i64(0UL), _ve_vmv_vsv( 2, vrsum01_s0), vmx_s0) ;
    vrsum01_s1 = _ve_vmrg_vvvm(_ve_vbrd_vs_i64(0UL), _ve_vmv_vsv( 1, vrsum01_s1), vmx_s1) ;
    vrsum01_s2 = _ve_vmrg_vvvm(_ve_vbrd_vs_i64(0UL),                 vrsum01_s2, vmx_s2) ;
    vrsum01_s3 = _ve_vmrg_vvvm(_ve_vbrd_vs_i64(0UL), _ve_vmv_vsv(-1, vrsum01_s3), vmx_s3) ;
    vrsum01_s4 = _ve_vmrg_vvvm(_ve_vbrd_vs_i64(0UL), _ve_vmv_vsv(-2, vrsum01_s4), vmx_s4) ;
    __vr vrsum01 = _ve_pvfadd_vvv(_ve_pvfadd_vvv(vrsum01_s0, _ve_pvfadd_vvv(vrsum01_s1, vrsum01_s2)),
                                  _ve_pvfadd_vvv(vrsum01_s3, vrsum01_s4)) ;
    _ve_vstu_vss(vrsum01, 4, pGIn+gInIndex) ;
    _ve_vstl_vss(vrsum01, 4, pGIn+gInIndex+  gInPixels) ;

    vrsum23_s0 = _ve_vmrg_vvvm(_ve_vbrd_vs_i64(0UL), _ve_vmv_vsv( 2, vrsum23_s0), vmx_s0) ;
    vrsum23_s1 = _ve_vmrg_vvvm(_ve_vbrd_vs_i64(0UL), _ve_vmv_vsv( 1, vrsum23_s1), vmx_s1) ;
    vrsum23_s2 = _ve_vmrg_vvvm(_ve_vbrd_vs_i64(0UL),                 vrsum23_s2, vmx_s2) ;
    vrsum23_s3 = _ve_vmrg_vvvm(_ve_vbrd_vs_i64(0UL), _ve_vmv_vsv(-1, vrsum23_s3), vmx_s3) ;
    vrsum23_s4 = _ve_vmrg_vvvm(_ve_vbrd_vs_i64(0UL), _ve_vmv_vsv(-2, vrsum23_s4), vmx_s4) ;
    __vr vrsum23 = _ve_pvfadd_vvv(_ve_pvfadd_vvv(vrsum23_s0, _ve_pvfadd_vvv(vrsum23_s1, vrsum23_s2)),
                                  _ve_pvfadd_vvv(vrsum23_s3, vrsum23_s4)) ;
    _ve_vstu_vss(vrsum23, 4, pGIn+gInIndex+2*gInPixels) ;
    _ve_vstl_vss(vrsum23, 4, pGIn+gInIndex+3*gInPixels) ;
  } // gOutPixels
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
    const int64_t c,
    const int64_t nH,
    const __vr vrh,
    const __vr vrw
)
{

  for (int64_t h=0; h<gInHeight; h+=nH) {
    const int64_t vl = gInWidth * (gInHeight - h < nH ? gInHeight - h : nH) ;
    const int64_t gip = h * gInWidth ;

    const int64_t gInIndex = gInGroupOffset + ((n * gInChannel + c) * gInHeight ) * gInWidth + gip ;

    _ve_lvl(vl) ;

    __vr vrsum01_s0 = _ve_vbrd_vs_i64(0UL) ;
    __vr vrsum23_s0 = _ve_vbrd_vs_i64(0UL) ;
    __vr vrsum45_s0 = _ve_vbrd_vs_i64(0UL) ;
    __vr vrsum67_s0 = _ve_vbrd_vs_i64(0UL) ;

    __vr vrsum01_s1 = _ve_vbrd_vs_i64(0UL) ;
    __vr vrsum23_s1 = _ve_vbrd_vs_i64(0UL) ;
    __vr vrsum45_s1 = _ve_vbrd_vs_i64(0UL) ;
    __vr vrsum67_s1 = _ve_vbrd_vs_i64(0UL) ;

    __vr vrsum01_s2 = _ve_vbrd_vs_i64(0UL) ;
    __vr vrsum23_s2 = _ve_vbrd_vs_i64(0UL) ;
    __vr vrsum45_s2 = _ve_vbrd_vs_i64(0UL) ;
    __vr vrsum67_s2 = _ve_vbrd_vs_i64(0UL) ;

    __vr vrsum01_s3 = _ve_vbrd_vs_i64(0UL) ;
    __vr vrsum23_s3 = _ve_vbrd_vs_i64(0UL) ;
    __vr vrsum45_s3 = _ve_vbrd_vs_i64(0UL) ;
    __vr vrsum67_s3 = _ve_vbrd_vs_i64(0UL) ;

    __vr vrsum01_s4 = _ve_vbrd_vs_i64(0UL) ;
    __vr vrsum23_s4 = _ve_vbrd_vs_i64(0UL) ;
    __vr vrsum45_s4 = _ve_vbrd_vs_i64(0UL) ;
    __vr vrsum67_s4 = _ve_vbrd_vs_i64(0UL) ;

    __vr vrj_s0 = _ve_vaddsl_vsv( 2, vrw) ;
    __vr vrj_s1 = _ve_vaddsl_vsv( 1, vrw) ;
    __vr vrj_s2 = _ve_vaddsl_vsv( 0, vrw) ;
    __vr vrj_s3 = _ve_vaddsl_vsv(-1, vrw) ;
    __vr vrj_s4 = _ve_vaddsl_vsv(-2, vrw) ;

    __vr vrx_s0 = _ve_vdivsl_vvs(vrj_s0, 2) ;
    __vr vrx_s1 = _ve_vdivsl_vvs(vrj_s1, 2) ;
    __vr vrx_s2 = _ve_vdivsl_vvs(vrj_s2, 2) ;
    __vr vrx_s3 = _ve_vdivsl_vvs(vrj_s3, 2) ;
    __vr vrx_s4 = _ve_vdivsl_vvs(vrj_s4, 2) ;

    __vm256 vmx0_s0 = _ve_vfmkl_mcv(VECC_IEQ, _ve_vcmpsl_vvv(vrj_s0, _ve_vmulsl_vsv(2, vrx_s0))) ;
    __vm256 vmx1_s0 = _ve_vfmkl_mcv(VECC_GE, vrx_s0) ;
    __vm256 vmx2_s0 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(gOutWidth,vrx_s0)) ;
    __vm256 vmx_s0 = _ve_andm_mmm(_ve_andm_mmm(vmx0_s0, vmx1_s0), vmx2_s0) ;

    __vm256 vmx0_s1 = _ve_vfmkl_mcv(VECC_IEQ, _ve_vcmpsl_vvv(vrj_s1, _ve_vmulsl_vsv(2, vrx_s1))) ;
    __vm256 vmx1_s1 = _ve_vfmkl_mcv(VECC_GE, vrx_s1) ;
    __vm256 vmx2_s1 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(gOutWidth,vrx_s1)) ;
    __vm256 vmx_s1 = _ve_andm_mmm(_ve_andm_mmm(vmx0_s1, vmx1_s1), vmx2_s1) ;

    __vm256 vmx0_s2 = _ve_vfmkl_mcv(VECC_IEQ, _ve_vcmpsl_vvv(vrj_s2, _ve_vmulsl_vsv(2, vrx_s2))) ;
    __vm256 vmx1_s2 = _ve_vfmkl_mcv(VECC_GE, vrx_s2) ;
    __vm256 vmx2_s2 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(gOutWidth,vrx_s2)) ;
    __vm256 vmx_s2 = _ve_andm_mmm(_ve_andm_mmm(vmx0_s2, vmx1_s2), vmx2_s2) ;

    __vm256 vmx0_s3 = _ve_vfmkl_mcv(VECC_IEQ, _ve_vcmpsl_vvv(vrj_s3, _ve_vmulsl_vsv(2, vrx_s3))) ;
    __vm256 vmx1_s3 = _ve_vfmkl_mcv(VECC_GE, vrx_s3) ;
    __vm256 vmx2_s3 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(gOutWidth,vrx_s3)) ;
    __vm256 vmx_s3 = _ve_andm_mmm(_ve_andm_mmm(vmx0_s3, vmx1_s3), vmx2_s3) ;

    __vm256 vmx0_s4 = _ve_vfmkl_mcv(VECC_IEQ, _ve_vcmpsl_vvv(vrj_s4, _ve_vmulsl_vsv(2, vrx_s4))) ;
    __vm256 vmx1_s4 = _ve_vfmkl_mcv(VECC_GE, vrx_s4) ;
    __vm256 vmx2_s4 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(gOutWidth,vrx_s4)) ;
    __vm256 vmx_s4 = _ve_andm_mmm(_ve_andm_mmm(vmx0_s4, vmx1_s4), vmx2_s4) ;

    for (int64_t r=0; r<kernHeight; r++) {
      __vr vri = _ve_vaddsl_vsv(2-r+h, vrh) ;
      __vr vry = _ve_vdivsl_vvs(vri, 2) ;

      __vm256 vmy0 = _ve_vfmkl_mcv(VECC_IEQ, _ve_vcmpsl_vvv(vri, _ve_vmulsl_vsv(2, vry))) ;
      __vm256 vmy1 = _ve_vfmkl_mcv(VECC_GE, vry) ;
      __vm256 vmy2 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(gOutHeight,vry)) ;
      __vm256 vmy = _ve_andm_mmm(_ve_andm_mmm(vmy0, vmy1), vmy2) ;

      int64_t k=0;
      for (; k<gOutChannelGroup; k++) {
	int64_t gOutIndex    = gOutGroupOffset + ((n * gOutChannel + k) * gOutHeight) * gOutWidth ;
	const float *pKerValue = pKernel + kernGroupOffset + (((k  ) * gInChannelGroup + c) * kernHeight) * kernWidth ;

#define VFADD_C8(VRGOUT, K, R, S)  {												\
	  const uint64_t kerValue01 = _ve_pack_f32p(pKerValue + (((K)*gInChannelGroup + 0) * kernHeight +(R)) * kernWidth + (S),	\
						    pKerValue + (((K)*gInChannelGroup + 1) * kernHeight +(R)) * kernWidth + (S)) ;	\
	  const uint64_t kerValue23 = _ve_pack_f32p(pKerValue + (((K)*gInChannelGroup + 2) * kernHeight +(R)) * kernWidth + (S),	\
						    pKerValue + (((K)*gInChannelGroup + 3) * kernHeight +(R)) * kernWidth + (S)) ;	\
	  const uint64_t kerValue45 = _ve_pack_f32p(pKerValue + (((K)*gInChannelGroup + 4) * kernHeight +(R)) * kernWidth + (S),	\
						    pKerValue + (((K)*gInChannelGroup + 5) * kernHeight +(R)) * kernWidth + (S)) ;	\
	  const uint64_t kerValue67 = _ve_pack_f32p(pKerValue + (((K)*gInChannelGroup + 6) * kernHeight +(R)) * kernWidth + (S),	\
						    pKerValue + (((K)*gInChannelGroup + 7) * kernHeight +(R)) * kernWidth + (S)) ;	\
	  __vr vrgoutP = _ve_vshf_vvvs(VRGOUT, VRGOUT, VE_VSHUFFLE_YUZU) ;	\
	  vrsum01_s##S = _ve_pvfmad_vvsv(vrsum01_s##S, kerValue01, vrgoutP) ;	\
	  vrsum23_s##S = _ve_pvfmad_vvsv(vrsum23_s##S, kerValue23, vrgoutP) ;	\
	  vrsum45_s##S = _ve_pvfmad_vvsv(vrsum45_s##S, kerValue45, vrgoutP) ;	\
	  vrsum67_s##S = _ve_pvfmad_vvsv(vrsum67_s##S, kerValue67, vrgoutP) ;	\
	}

	__vr vrgout_ptr_k0_s2 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry), vrx_s2),
					   2,
					   (unsigned long)(pGOut+gOutIndex)) ;
	__vr vrgout_k0_s2 = _ve_vgtu_vvm(vrgout_ptr_k0_s2, vmy) ;
	vrgout_k0_s2 = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), vrgout_k0_s2, vmy) ;

	VFADD_C8(vrgout_k0_s2, 0, r, 0)
	VFADD_C8(vrgout_k0_s2, 0, r, 1)
	VFADD_C8(vrgout_k0_s2, 0, r, 2)
	VFADD_C8(vrgout_k0_s2, 0, r, 3)
	VFADD_C8(vrgout_k0_s2, 0, r, 4)
#undef VFADD_C8
      }
    } // kernHeight

    vrsum01_s0 = _ve_vmrg_vvvm(_ve_vbrd_vs_i64(0UL), _ve_vmv_vsv( 2, vrsum01_s0), vmx_s0) ;
    vrsum01_s1 = _ve_vmrg_vvvm(_ve_vbrd_vs_i64(0UL), _ve_vmv_vsv( 1, vrsum01_s1), vmx_s1) ;
    vrsum01_s2 = _ve_vmrg_vvvm(_ve_vbrd_vs_i64(0UL),                 vrsum01_s2, vmx_s2) ;
    vrsum01_s3 = _ve_vmrg_vvvm(_ve_vbrd_vs_i64(0UL), _ve_vmv_vsv(-1, vrsum01_s3), vmx_s3) ;
    vrsum01_s4 = _ve_vmrg_vvvm(_ve_vbrd_vs_i64(0UL), _ve_vmv_vsv(-2, vrsum01_s4), vmx_s4) ;
    __vr vrsum01 = _ve_pvfadd_vvv(_ve_pvfadd_vvv(vrsum01_s0, _ve_pvfadd_vvv(vrsum01_s1, vrsum01_s2)),
                                  _ve_pvfadd_vvv(vrsum01_s3, vrsum01_s4)) ;
    _ve_vstu_vss(vrsum01, 4, pGIn+gInIndex) ;
    _ve_vstl_vss(vrsum01, 4, pGIn+gInIndex+  gInPixels) ;

    vrsum23_s0 = _ve_vmrg_vvvm(_ve_vbrd_vs_i64(0UL), _ve_vmv_vsv( 2, vrsum23_s0), vmx_s0) ;
    vrsum23_s1 = _ve_vmrg_vvvm(_ve_vbrd_vs_i64(0UL), _ve_vmv_vsv( 1, vrsum23_s1), vmx_s1) ;
    vrsum23_s2 = _ve_vmrg_vvvm(_ve_vbrd_vs_i64(0UL),                 vrsum23_s2, vmx_s2) ;
    vrsum23_s3 = _ve_vmrg_vvvm(_ve_vbrd_vs_i64(0UL), _ve_vmv_vsv(-1, vrsum23_s3), vmx_s3) ;
    vrsum23_s4 = _ve_vmrg_vvvm(_ve_vbrd_vs_i64(0UL), _ve_vmv_vsv(-2, vrsum23_s4), vmx_s4) ;
    __vr vrsum23 = _ve_pvfadd_vvv(_ve_pvfadd_vvv(vrsum23_s0, _ve_pvfadd_vvv(vrsum23_s1, vrsum23_s2)),
                                  _ve_pvfadd_vvv(vrsum23_s3, vrsum23_s4)) ;
    _ve_vstu_vss(vrsum23, 4, pGIn+gInIndex+2*gInPixels) ;
    _ve_vstl_vss(vrsum23, 4, pGIn+gInIndex+3*gInPixels) ;

    vrsum45_s0 = _ve_vmrg_vvvm(_ve_vbrd_vs_i64(0UL), _ve_vmv_vsv( 2, vrsum45_s0), vmx_s0) ;
    vrsum45_s1 = _ve_vmrg_vvvm(_ve_vbrd_vs_i64(0UL), _ve_vmv_vsv( 1, vrsum45_s1), vmx_s1) ;
    vrsum45_s2 = _ve_vmrg_vvvm(_ve_vbrd_vs_i64(0UL),                 vrsum45_s2, vmx_s2) ;
    vrsum45_s3 = _ve_vmrg_vvvm(_ve_vbrd_vs_i64(0UL), _ve_vmv_vsv(-1, vrsum45_s3), vmx_s3) ;
    vrsum45_s4 = _ve_vmrg_vvvm(_ve_vbrd_vs_i64(0UL), _ve_vmv_vsv(-2, vrsum45_s4), vmx_s4) ;
    __vr vrsum45 = _ve_pvfadd_vvv(_ve_pvfadd_vvv(vrsum45_s0, _ve_pvfadd_vvv(vrsum45_s1, vrsum45_s2)),
                                  _ve_pvfadd_vvv(vrsum45_s3, vrsum45_s4)) ;
    _ve_vstu_vss(vrsum45, 4, pGIn+gInIndex+4*gInPixels) ;
    _ve_vstl_vss(vrsum45, 4, pGIn+gInIndex+5*gInPixels) ;

    vrsum67_s0 = _ve_vmrg_vvvm(_ve_vbrd_vs_i64(0UL), _ve_vmv_vsv( 2, vrsum67_s0), vmx_s0) ;
    vrsum67_s1 = _ve_vmrg_vvvm(_ve_vbrd_vs_i64(0UL), _ve_vmv_vsv( 1, vrsum67_s1), vmx_s1) ;
    vrsum67_s2 = _ve_vmrg_vvvm(_ve_vbrd_vs_i64(0UL),                 vrsum67_s2, vmx_s2) ;
    vrsum67_s3 = _ve_vmrg_vvvm(_ve_vbrd_vs_i64(0UL), _ve_vmv_vsv(-1, vrsum67_s3), vmx_s3) ;
    vrsum67_s4 = _ve_vmrg_vvvm(_ve_vbrd_vs_i64(0UL), _ve_vmv_vsv(-2, vrsum67_s4), vmx_s4) ;
    __vr vrsum67 = _ve_pvfadd_vvv(_ve_pvfadd_vvv(vrsum67_s0, _ve_pvfadd_vvv(vrsum67_s1, vrsum67_s2)),
                                  _ve_pvfadd_vvv(vrsum67_s3, vrsum67_s4)) ;
    _ve_vstu_vss(vrsum67, 4, pGIn+gInIndex+6*gInPixels) ;
    _ve_vstl_vss(vrsum67, 4, pGIn+gInIndex+7*gInPixels) ;

  } // gOutPixels
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
    const int64_t c,
    const int64_t nH,
    const __vr vrh,
    const __vr vrw
)
{

  for (int64_t h=0; h<gInHeight; h+=nH) {
    const int64_t vl = gInWidth * (gInHeight - h < nH ? gInHeight - h : nH) ;
    const int64_t gip = h * gInWidth ;

    const int64_t gInIndex = gInGroupOffset + ((n * gInChannel + c) * gInHeight ) * gInWidth + gip ;

    _ve_lvl(vl) ;

    __vr vrsum01_s0 = _ve_vbrd_vs_i64(0UL) ;
    __vr vrsum23_s0 = _ve_vbrd_vs_i64(0UL) ;
    __vr vrsum45_s0 = _ve_vbrd_vs_i64(0UL) ;
    __vr vrsum67_s0 = _ve_vbrd_vs_i64(0UL) ;
    __vr vrsum89_s0 = _ve_vbrd_vs_i64(0UL) ;
    __vr vrsumAB_s0 = _ve_vbrd_vs_i64(0UL) ;
    __vr vrsumCD_s0 = _ve_vbrd_vs_i64(0UL) ;
    __vr vrsumEF_s0 = _ve_vbrd_vs_i64(0UL) ;

    __vr vrsum01_s1 = _ve_vbrd_vs_i64(0UL) ;
    __vr vrsum23_s1 = _ve_vbrd_vs_i64(0UL) ;
    __vr vrsum45_s1 = _ve_vbrd_vs_i64(0UL) ;
    __vr vrsum67_s1 = _ve_vbrd_vs_i64(0UL) ;
    __vr vrsum89_s1 = _ve_vbrd_vs_i64(0UL) ;
    __vr vrsumAB_s1 = _ve_vbrd_vs_i64(0UL) ;
    __vr vrsumCD_s1 = _ve_vbrd_vs_i64(0UL) ;
    __vr vrsumEF_s1 = _ve_vbrd_vs_i64(0UL) ;

    __vr vrsum01_s2 = _ve_vbrd_vs_i64(0UL) ;
    __vr vrsum23_s2 = _ve_vbrd_vs_i64(0UL) ;
    __vr vrsum45_s2 = _ve_vbrd_vs_i64(0UL) ;
    __vr vrsum67_s2 = _ve_vbrd_vs_i64(0UL) ;
    __vr vrsum89_s2 = _ve_vbrd_vs_i64(0UL) ;
    __vr vrsumAB_s2 = _ve_vbrd_vs_i64(0UL) ;
    __vr vrsumCD_s2 = _ve_vbrd_vs_i64(0UL) ;
    __vr vrsumEF_s2 = _ve_vbrd_vs_i64(0UL) ;

    __vr vrsum01_s3 = _ve_vbrd_vs_i64(0UL) ;
    __vr vrsum23_s3 = _ve_vbrd_vs_i64(0UL) ;
    __vr vrsum45_s3 = _ve_vbrd_vs_i64(0UL) ;
    __vr vrsum67_s3 = _ve_vbrd_vs_i64(0UL) ;
    __vr vrsum89_s3 = _ve_vbrd_vs_i64(0UL) ;
    __vr vrsumAB_s3 = _ve_vbrd_vs_i64(0UL) ;
    __vr vrsumCD_s3 = _ve_vbrd_vs_i64(0UL) ;
    __vr vrsumEF_s3 = _ve_vbrd_vs_i64(0UL) ;

    __vr vrsum01_s4 = _ve_vbrd_vs_i64(0UL) ;
    __vr vrsum23_s4 = _ve_vbrd_vs_i64(0UL) ;
    __vr vrsum45_s4 = _ve_vbrd_vs_i64(0UL) ;
    __vr vrsum67_s4 = _ve_vbrd_vs_i64(0UL) ;
    __vr vrsum89_s4 = _ve_vbrd_vs_i64(0UL) ;
    __vr vrsumAB_s4 = _ve_vbrd_vs_i64(0UL) ;
    __vr vrsumCD_s4 = _ve_vbrd_vs_i64(0UL) ;
    __vr vrsumEF_s4 = _ve_vbrd_vs_i64(0UL) ;

    __vr vrj_s0 = _ve_vaddsl_vsv( 2, vrw) ;
    __vr vrj_s1 = _ve_vaddsl_vsv( 1, vrw) ;
    __vr vrj_s2 = _ve_vaddsl_vsv( 0, vrw) ;
    __vr vrj_s3 = _ve_vaddsl_vsv(-1, vrw) ;
    __vr vrj_s4 = _ve_vaddsl_vsv(-2, vrw) ;

    __vr vrx_s0 = _ve_vdivsl_vvs(vrj_s0, 2) ;
    __vr vrx_s1 = _ve_vdivsl_vvs(vrj_s1, 2) ;
    __vr vrx_s2 = _ve_vdivsl_vvs(vrj_s2, 2) ;
    __vr vrx_s3 = _ve_vdivsl_vvs(vrj_s3, 2) ;
    __vr vrx_s4 = _ve_vdivsl_vvs(vrj_s4, 2) ;

    __vm256 vmx0_s0 = _ve_vfmkl_mcv(VECC_IEQ, _ve_vcmpsl_vvv(vrj_s0, _ve_vmulsl_vsv(2, vrx_s0))) ;
    __vm256 vmx1_s0 = _ve_vfmkl_mcv(VECC_GE, vrx_s0) ;
    __vm256 vmx2_s0 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(gOutWidth,vrx_s0)) ;
    __vm256 vmx_s0 = _ve_andm_mmm(_ve_andm_mmm(vmx0_s0, vmx1_s0), vmx2_s0) ;

    __vm256 vmx0_s1 = _ve_vfmkl_mcv(VECC_IEQ, _ve_vcmpsl_vvv(vrj_s1, _ve_vmulsl_vsv(2, vrx_s1))) ;
    __vm256 vmx1_s1 = _ve_vfmkl_mcv(VECC_GE, vrx_s1) ;
    __vm256 vmx2_s1 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(gOutWidth,vrx_s1)) ;
    __vm256 vmx_s1 = _ve_andm_mmm(_ve_andm_mmm(vmx0_s1, vmx1_s1), vmx2_s1) ;

    __vm256 vmx0_s2 = _ve_vfmkl_mcv(VECC_IEQ, _ve_vcmpsl_vvv(vrj_s2, _ve_vmulsl_vsv(2, vrx_s2))) ;
    __vm256 vmx1_s2 = _ve_vfmkl_mcv(VECC_GE, vrx_s2) ;
    __vm256 vmx2_s2 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(gOutWidth,vrx_s2)) ;
    __vm256 vmx_s2 = _ve_andm_mmm(_ve_andm_mmm(vmx0_s2, vmx1_s2), vmx2_s2) ;

    __vm256 vmx0_s3 = _ve_vfmkl_mcv(VECC_IEQ, _ve_vcmpsl_vvv(vrj_s3, _ve_vmulsl_vsv(2, vrx_s3))) ;
    __vm256 vmx1_s3 = _ve_vfmkl_mcv(VECC_GE, vrx_s3) ;
    __vm256 vmx2_s3 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(gOutWidth,vrx_s3)) ;
    __vm256 vmx_s3 = _ve_andm_mmm(_ve_andm_mmm(vmx0_s3, vmx1_s3), vmx2_s3) ;

    __vm256 vmx0_s4 = _ve_vfmkl_mcv(VECC_IEQ, _ve_vcmpsl_vvv(vrj_s4, _ve_vmulsl_vsv(2, vrx_s4))) ;
    __vm256 vmx1_s4 = _ve_vfmkl_mcv(VECC_GE, vrx_s4) ;
    __vm256 vmx2_s4 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(gOutWidth,vrx_s4)) ;
    __vm256 vmx_s4 = _ve_andm_mmm(_ve_andm_mmm(vmx0_s4, vmx1_s4), vmx2_s4) ;

    for (int64_t r=0; r<kernHeight; r++) {
      __vr vri = _ve_vaddsl_vsv(2-r+h, vrh) ;
      __vr vry = _ve_vdivsl_vvs(vri, 2) ;

      __vm256 vmy0 = _ve_vfmkl_mcv(VECC_IEQ, _ve_vcmpsl_vvv(vri, _ve_vmulsl_vsv(2, vry))) ;
      __vm256 vmy1 = _ve_vfmkl_mcv(VECC_GE, vry) ;
      __vm256 vmy2 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(gOutHeight,vry)) ;
      __vm256 vmy = _ve_andm_mmm(_ve_andm_mmm(vmy0, vmy1), vmy2) ;

      int64_t k=0;
      for (; k<gOutChannelGroup; k++) {
	int64_t gOutIndex    = gOutGroupOffset + ((n * gOutChannel + k) * gOutHeight) * gOutWidth ;
	const float *pKerValue = pKernel + kernGroupOffset + (((k  ) * gInChannelGroup + c) * kernHeight) * kernWidth ;

#define VFADD_C16(VRGOUT, K, R, S)  {												\
	  const uint64_t kerValue01 = _ve_pack_f32p(pKerValue + (((K)*gInChannelGroup + 0) * kernHeight +(R)) * kernWidth + (S),	\
						    pKerValue + (((K)*gInChannelGroup + 1) * kernHeight +(R)) * kernWidth + (S)) ;	\
	  const uint64_t kerValue23 = _ve_pack_f32p(pKerValue + (((K)*gInChannelGroup + 2) * kernHeight +(R)) * kernWidth + (S),	\
						    pKerValue + (((K)*gInChannelGroup + 3) * kernHeight +(R)) * kernWidth + (S)) ;	\
	  const uint64_t kerValue45 = _ve_pack_f32p(pKerValue + (((K)*gInChannelGroup + 4) * kernHeight +(R)) * kernWidth + (S),	\
						    pKerValue + (((K)*gInChannelGroup + 5) * kernHeight +(R)) * kernWidth + (S)) ;	\
	  const uint64_t kerValue67 = _ve_pack_f32p(pKerValue + (((K)*gInChannelGroup + 6) * kernHeight +(R)) * kernWidth + (S),	\
						    pKerValue + (((K)*gInChannelGroup + 7) * kernHeight +(R)) * kernWidth + (S)) ;	\
	  const uint64_t kerValue89 = _ve_pack_f32p(pKerValue + (((K)*gInChannelGroup + 8) * kernHeight +(R)) * kernWidth + (S),	\
						    pKerValue + (((K)*gInChannelGroup + 9) * kernHeight +(R)) * kernWidth + (S)) ;	\
	  const uint64_t kerValueAB = _ve_pack_f32p(pKerValue + (((K)*gInChannelGroup +10) * kernHeight +(R)) * kernWidth + (S),	\
						    pKerValue + (((K)*gInChannelGroup +11) * kernHeight +(R)) * kernWidth + (S)) ;	\
	  const uint64_t kerValueCD = _ve_pack_f32p(pKerValue + (((K)*gInChannelGroup +12) * kernHeight +(R)) * kernWidth + (S),	\
						    pKerValue + (((K)*gInChannelGroup +13) * kernHeight +(R)) * kernWidth + (S)) ;	\
	  const uint64_t kerValueEF = _ve_pack_f32p(pKerValue + (((K)*gInChannelGroup +14) * kernHeight +(R)) * kernWidth + (S),	\
						    pKerValue + (((K)*gInChannelGroup +15) * kernHeight +(R)) * kernWidth + (S)) ;	\
	  __vr vrgoutP = _ve_vshf_vvvs(VRGOUT, VRGOUT, VE_VSHUFFLE_YUZU) ;	\
	  vrsum01_s##S = _ve_pvfmad_vvsv(vrsum01_s##S, kerValue01, vrgoutP) ;	\
	  vrsum23_s##S = _ve_pvfmad_vvsv(vrsum23_s##S, kerValue23, vrgoutP) ;	\
	  vrsum45_s##S = _ve_pvfmad_vvsv(vrsum45_s##S, kerValue45, vrgoutP) ;	\
	  vrsum67_s##S = _ve_pvfmad_vvsv(vrsum67_s##S, kerValue67, vrgoutP) ;	\
	  vrsum89_s##S = _ve_pvfmad_vvsv(vrsum89_s##S, kerValue89, vrgoutP) ;	\
	  vrsumAB_s##S = _ve_pvfmad_vvsv(vrsumAB_s##S, kerValueAB, vrgoutP) ;	\
	  vrsumCD_s##S = _ve_pvfmad_vvsv(vrsumCD_s##S, kerValueCD, vrgoutP) ;	\
	  vrsumEF_s##S = _ve_pvfmad_vvsv(vrsumEF_s##S, kerValueEF, vrgoutP) ;	\
	}

	__vr vrgout_ptr_k0_s2 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry), vrx_s2),
					   2,
					   (unsigned long)(pGOut+gOutIndex)) ;
	__vr vrgout_k0_s2 = _ve_vgtu_vvm(vrgout_ptr_k0_s2, vmy) ;
	vrgout_k0_s2 = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), vrgout_k0_s2, vmy) ;

	VFADD_C16(vrgout_k0_s2, 0, r, 0)
	VFADD_C16(vrgout_k0_s2, 0, r, 1)
	VFADD_C16(vrgout_k0_s2, 0, r, 2)
	VFADD_C16(vrgout_k0_s2, 0, r, 3)
	VFADD_C16(vrgout_k0_s2, 0, r, 4)
#undef VFADD_C16
      }
    } // kernHeight

    vrsum01_s0 = _ve_vmrg_vvvm(_ve_vbrd_vs_i64(0UL), _ve_vmv_vsv( 2, vrsum01_s0), vmx_s0) ;
    vrsum01_s1 = _ve_vmrg_vvvm(_ve_vbrd_vs_i64(0UL), _ve_vmv_vsv( 1, vrsum01_s1), vmx_s1) ;
    vrsum01_s2 = _ve_vmrg_vvvm(_ve_vbrd_vs_i64(0UL),                 vrsum01_s2, vmx_s2) ;
    vrsum01_s3 = _ve_vmrg_vvvm(_ve_vbrd_vs_i64(0UL), _ve_vmv_vsv(-1, vrsum01_s3), vmx_s3) ;
    vrsum01_s4 = _ve_vmrg_vvvm(_ve_vbrd_vs_i64(0UL), _ve_vmv_vsv(-2, vrsum01_s4), vmx_s4) ;
    __vr vrsum01 = _ve_pvfadd_vvv(_ve_pvfadd_vvv(vrsum01_s0, _ve_pvfadd_vvv(vrsum01_s1, vrsum01_s2)),
                                  _ve_pvfadd_vvv(vrsum01_s3, vrsum01_s4)) ;
    _ve_vstu_vss(vrsum01, 4, pGIn+gInIndex) ;
    _ve_vstl_vss(vrsum01, 4, pGIn+gInIndex+  gInPixels) ;

    vrsum23_s0 = _ve_vmrg_vvvm(_ve_vbrd_vs_i64(0UL), _ve_vmv_vsv( 2, vrsum23_s0), vmx_s0) ;
    vrsum23_s1 = _ve_vmrg_vvvm(_ve_vbrd_vs_i64(0UL), _ve_vmv_vsv( 1, vrsum23_s1), vmx_s1) ;
    vrsum23_s2 = _ve_vmrg_vvvm(_ve_vbrd_vs_i64(0UL),                 vrsum23_s2, vmx_s2) ;
    vrsum23_s3 = _ve_vmrg_vvvm(_ve_vbrd_vs_i64(0UL), _ve_vmv_vsv(-1, vrsum23_s3), vmx_s3) ;
    vrsum23_s4 = _ve_vmrg_vvvm(_ve_vbrd_vs_i64(0UL), _ve_vmv_vsv(-2, vrsum23_s4), vmx_s4) ;
    __vr vrsum23 = _ve_pvfadd_vvv(_ve_pvfadd_vvv(vrsum23_s0, _ve_pvfadd_vvv(vrsum23_s1, vrsum23_s2)),
                                  _ve_pvfadd_vvv(vrsum23_s3, vrsum23_s4)) ;
    _ve_vstu_vss(vrsum23, 4, pGIn+gInIndex+2*gInPixels) ;
    _ve_vstl_vss(vrsum23, 4, pGIn+gInIndex+3*gInPixels) ;

    vrsum45_s0 = _ve_vmrg_vvvm(_ve_vbrd_vs_i64(0UL), _ve_vmv_vsv( 2, vrsum45_s0), vmx_s0) ;
    vrsum45_s1 = _ve_vmrg_vvvm(_ve_vbrd_vs_i64(0UL), _ve_vmv_vsv( 1, vrsum45_s1), vmx_s1) ;
    vrsum45_s2 = _ve_vmrg_vvvm(_ve_vbrd_vs_i64(0UL),                 vrsum45_s2, vmx_s2) ;
    vrsum45_s3 = _ve_vmrg_vvvm(_ve_vbrd_vs_i64(0UL), _ve_vmv_vsv(-1, vrsum45_s3), vmx_s3) ;
    vrsum45_s4 = _ve_vmrg_vvvm(_ve_vbrd_vs_i64(0UL), _ve_vmv_vsv(-2, vrsum45_s4), vmx_s4) ;
    __vr vrsum45 = _ve_pvfadd_vvv(_ve_pvfadd_vvv(vrsum45_s0, _ve_pvfadd_vvv(vrsum45_s1, vrsum45_s2)),
                                  _ve_pvfadd_vvv(vrsum45_s3, vrsum45_s4)) ;
    _ve_vstu_vss(vrsum45, 4, pGIn+gInIndex+4*gInPixels) ;
    _ve_vstl_vss(vrsum45, 4, pGIn+gInIndex+5*gInPixels) ;

    vrsum67_s0 = _ve_vmrg_vvvm(_ve_vbrd_vs_i64(0UL), _ve_vmv_vsv( 2, vrsum67_s0), vmx_s0) ;
    vrsum67_s1 = _ve_vmrg_vvvm(_ve_vbrd_vs_i64(0UL), _ve_vmv_vsv( 1, vrsum67_s1), vmx_s1) ;
    vrsum67_s2 = _ve_vmrg_vvvm(_ve_vbrd_vs_i64(0UL),                 vrsum67_s2, vmx_s2) ;
    vrsum67_s3 = _ve_vmrg_vvvm(_ve_vbrd_vs_i64(0UL), _ve_vmv_vsv(-1, vrsum67_s3), vmx_s3) ;
    vrsum67_s4 = _ve_vmrg_vvvm(_ve_vbrd_vs_i64(0UL), _ve_vmv_vsv(-2, vrsum67_s4), vmx_s4) ;
    __vr vrsum67 = _ve_pvfadd_vvv(_ve_pvfadd_vvv(vrsum67_s0, _ve_pvfadd_vvv(vrsum67_s1, vrsum67_s2)),
                                  _ve_pvfadd_vvv(vrsum67_s3, vrsum67_s4)) ;
    _ve_vstu_vss(vrsum67, 4, pGIn+gInIndex+6*gInPixels) ;
    _ve_vstl_vss(vrsum67, 4, pGIn+gInIndex+7*gInPixels) ;

    vrsum89_s0 = _ve_vmrg_vvvm(_ve_vbrd_vs_i64(0UL), _ve_vmv_vsv( 2, vrsum89_s0), vmx_s0) ;
    vrsum89_s1 = _ve_vmrg_vvvm(_ve_vbrd_vs_i64(0UL), _ve_vmv_vsv( 1, vrsum89_s1), vmx_s1) ;
    vrsum89_s2 = _ve_vmrg_vvvm(_ve_vbrd_vs_i64(0UL),                 vrsum89_s2, vmx_s2) ;
    vrsum89_s3 = _ve_vmrg_vvvm(_ve_vbrd_vs_i64(0UL), _ve_vmv_vsv(-1, vrsum89_s3), vmx_s3) ;
    vrsum89_s4 = _ve_vmrg_vvvm(_ve_vbrd_vs_i64(0UL), _ve_vmv_vsv(-2, vrsum89_s4), vmx_s4) ;
    __vr vrsum89 = _ve_pvfadd_vvv(_ve_pvfadd_vvv(vrsum89_s0, _ve_pvfadd_vvv(vrsum89_s1, vrsum89_s2)),
                                  _ve_pvfadd_vvv(vrsum89_s3, vrsum89_s4)) ;
    _ve_vstu_vss(vrsum89, 4, pGIn+gInIndex+8*gInPixels) ;
    _ve_vstl_vss(vrsum89, 4, pGIn+gInIndex+9*gInPixels) ;

    vrsumAB_s0 = _ve_vmrg_vvvm(_ve_vbrd_vs_i64(0UL), _ve_vmv_vsv( 2, vrsumAB_s0), vmx_s0) ;
    vrsumAB_s1 = _ve_vmrg_vvvm(_ve_vbrd_vs_i64(0UL), _ve_vmv_vsv( 1, vrsumAB_s1), vmx_s1) ;
    vrsumAB_s2 = _ve_vmrg_vvvm(_ve_vbrd_vs_i64(0UL),                 vrsumAB_s2, vmx_s2) ;
    vrsumAB_s3 = _ve_vmrg_vvvm(_ve_vbrd_vs_i64(0UL), _ve_vmv_vsv(-1, vrsumAB_s3), vmx_s3) ;
    vrsumAB_s4 = _ve_vmrg_vvvm(_ve_vbrd_vs_i64(0UL), _ve_vmv_vsv(-2, vrsumAB_s4), vmx_s4) ;
    __vr vrsumAB = _ve_pvfadd_vvv(_ve_pvfadd_vvv(vrsumAB_s0, _ve_pvfadd_vvv(vrsumAB_s1, vrsumAB_s2)),
                                  _ve_pvfadd_vvv(vrsumAB_s3, vrsumAB_s4)) ;
    _ve_vstu_vss(vrsumAB, 4, pGIn+gInIndex+10*gInPixels) ;
    _ve_vstl_vss(vrsumAB, 4, pGIn+gInIndex+11*gInPixels) ;

    vrsumCD_s0 = _ve_vmrg_vvvm(_ve_vbrd_vs_i64(0UL), _ve_vmv_vsv( 2, vrsumCD_s0), vmx_s0) ;
    vrsumCD_s1 = _ve_vmrg_vvvm(_ve_vbrd_vs_i64(0UL), _ve_vmv_vsv( 1, vrsumCD_s1), vmx_s1) ;
    vrsumCD_s2 = _ve_vmrg_vvvm(_ve_vbrd_vs_i64(0UL),                 vrsumCD_s2, vmx_s2) ;
    vrsumCD_s3 = _ve_vmrg_vvvm(_ve_vbrd_vs_i64(0UL), _ve_vmv_vsv(-1, vrsumCD_s3), vmx_s3) ;
    vrsumCD_s4 = _ve_vmrg_vvvm(_ve_vbrd_vs_i64(0UL), _ve_vmv_vsv(-2, vrsumCD_s4), vmx_s4) ;
    __vr vrsumCD = _ve_pvfadd_vvv(_ve_pvfadd_vvv(vrsumCD_s0, _ve_pvfadd_vvv(vrsumCD_s1, vrsumCD_s2)),
                                  _ve_pvfadd_vvv(vrsumCD_s3, vrsumCD_s4)) ;
    _ve_vstu_vss(vrsumCD, 4, pGIn+gInIndex+12*gInPixels) ;
    _ve_vstl_vss(vrsumCD, 4, pGIn+gInIndex+13*gInPixels) ;

    vrsumEF_s0 = _ve_vmrg_vvvm(_ve_vbrd_vs_i64(0UL), _ve_vmv_vsv( 2, vrsumEF_s0), vmx_s0) ;
    vrsumEF_s1 = _ve_vmrg_vvvm(_ve_vbrd_vs_i64(0UL), _ve_vmv_vsv( 1, vrsumEF_s1), vmx_s1) ;
    vrsumEF_s2 = _ve_vmrg_vvvm(_ve_vbrd_vs_i64(0UL),                 vrsumEF_s2, vmx_s2) ;
    vrsumEF_s3 = _ve_vmrg_vvvm(_ve_vbrd_vs_i64(0UL), _ve_vmv_vsv(-1, vrsumEF_s3), vmx_s3) ;
    vrsumEF_s4 = _ve_vmrg_vvvm(_ve_vbrd_vs_i64(0UL), _ve_vmv_vsv(-2, vrsumEF_s4), vmx_s4) ;
    __vr vrsumEF = _ve_pvfadd_vvv(_ve_pvfadd_vvv(vrsumEF_s0, _ve_pvfadd_vvv(vrsumEF_s1, vrsumEF_s2)),
                                  _ve_pvfadd_vvv(vrsumEF_s3, vrsumEF_s4)) ;
    _ve_vstu_vss(vrsumEF, 4, pGIn+gInIndex+14*gInPixels) ;
    _ve_vstl_vss(vrsumEF, 4, pGIn+gInIndex+15*gInPixels) ;
  } // gOutPixels
}


vednnError_t
vednnConvolutionBackwardData_direct_dil1_str2_pad2_ker5_iwU128(
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

  const int gInPixels= gInHeight*gInWidth ;

  /* intrinsic version 1 */
  {
    const int64_t nH = VLEN / gInWidth ;

    _ve_lvl(nH*gInWidth) ;
    __vr vrseq = _ve_vseq_v() ;
    __vr vrh  = _ve_vdivsl_vvs(vrseq, gInWidth) ;
    __vr vrw  = _ve_vsubsl_vvv(vrseq, _ve_vmulul_vsv(gInWidth,vrh)) ;

    for (int64_t n=0; n<batch; n++) {
      for (int64_t g = 0; g < group; g++) {

	int64_t gInGroupOffset  = g * gInChannelGroup * gInHeight * gInWidth;
	int64_t gOutGroupOffset = g * gOutChannelGroup * gOutHeight * gOutWidth;
	int64_t kernGroupOffset = g * gOutChannelGroup * gInChannelGroup * kernHeight * kernWidth;

	int64_t k=0;
	if( (gInChannelGroup & 0x01 ) == 1 ) {

	  c1(pGOut, pKernel, pGIn,
	     gOutChannel, gOutWidth, gOutHeight,
             gInChannel, gInWidth, gInHeight,
             kernWidth, kernHeight,
	     gInChannelGroup, gOutChannelGroup,
	     gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	     gInPixels, n, k,
	     nH, vrh, vrw ) ;

	  k++ ;
	}
	if( ((gInChannelGroup>>1) & 0x01 ) == 1 ) {

	  c2(pGOut, pKernel, pGIn,
	     gOutChannel, gOutWidth, gOutHeight,
             gInChannel, gInWidth, gInHeight,
             kernWidth, kernHeight,
	     gInChannelGroup, gOutChannelGroup,
	     gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	     gInPixels, n, k,
	     nH, vrh, vrw ) ;

	  k+=2 ;
	}
	if( ((gInChannelGroup>>2) & 0x01 ) == 1 ) {

	  c4(pGOut, pKernel, pGIn,
	     gOutChannel, gOutWidth, gOutHeight,
             gInChannel, gInWidth, gInHeight,
             kernWidth, kernHeight,
	     gInChannelGroup, gOutChannelGroup,
	     gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	     gInPixels, n, k,
	     nH, vrh, vrw ) ;

	  k+=4 ;
	}
	if( ((gInChannelGroup>>3) & 0x01 ) == 1 ) {

	  c8(pGOut, pKernel, pGIn,
	     gOutChannel, gOutWidth, gOutHeight,
             gInChannel, gInWidth, gInHeight,
             kernWidth, kernHeight,
	     gInChannelGroup, gOutChannelGroup,
	     gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	     gInPixels, n, k,
	     nH, vrh, vrw ) ;

	  k+=8 ;
	}
	for (; k<gInChannelGroup; k+=16) {
	  c16(pGOut, pKernel, pGIn,
	     gOutChannel, gOutWidth, gOutHeight,
             gInChannel, gInWidth, gInHeight,
             kernWidth, kernHeight,
	     gInChannelGroup, gOutChannelGroup,
	     gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	     gInPixels, n, k,
	     nH, vrh, vrw ) ;

	} // gInChannel
      } // group
    } // batch
  }


  return VEDNN_SUCCESS;
}
