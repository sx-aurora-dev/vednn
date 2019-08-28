#include <stdint.h>
#include <stdlib.h>

#include "vednn.h"

#include "velintrin.h"
#define VLEN	(256)

// force no-inline to avoid vr-spill in the most-inner loop
static __attribute__((noinline))  void c1(
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
    const int64_t nH
)
{
  __vr vrseq = _vel_vseq_vl(nH*gInWidth) ;
  __vr vrh  = _vel_vdivsl_vvsl(vrseq, gInWidth, nH*gInWidth) ;
  __vr vrw  = _vel_vsubsl_vvvl(vrseq, _vel_vmulul_vsvl(gInWidth,vrh, nH*gInWidth), nH*gInWidth) ;

  for (int64_t h=0; h<gInHeight; h+=nH) {
    const int64_t vl = gInWidth * (gInHeight - h < nH ? gInHeight - h : nH) ;
    const int64_t gip = h * gInWidth ;

    const int64_t gInIndex = gInGroupOffset + ((n * gInChannel + c) * gInHeight ) * gInWidth + gip ;

    __vr vrsum_s0 = _vel_vbrds_vsl(0.f, vl) ;
    __vr vrsum_s1 = _vel_vbrds_vsl(0.f, vl) ;
    __vr vrsum_s2 = _vel_vbrds_vsl(0.f, vl) ;
    __vr vrsum_s3 = _vel_vbrds_vsl(0.f, vl) ;
    __vr vrsum_s4 = _vel_vbrds_vsl(0.f, vl) ;

    __vr vri_r0 = _vel_vaddsl_vsvl(2-0+h, vrh, vl) ;
    __vr vri_r1 = _vel_vaddsl_vsvl(2-1+h, vrh, vl) ;
    __vr vri_r2 = _vel_vaddsl_vsvl(2-2+h, vrh, vl) ;
    __vr vri_r3 = _vel_vaddsl_vsvl(2-3+h, vrh, vl) ;
    __vr vri_r4 = _vel_vaddsl_vsvl(2-4+h, vrh, vl) ;

    __vr vry_r0 = _vel_vdivsl_vvsl(vri_r0, 2, vl) ;
    __vr vry_r1 = _vel_vdivsl_vvsl(vri_r1, 2, vl) ;
    __vr vry_r2 = _vel_vdivsl_vvsl(vri_r2, 2, vl) ;
    __vr vry_r3 = _vel_vdivsl_vvsl(vri_r3, 2, vl) ;
    __vr vry_r4 = _vel_vdivsl_vvsl(vri_r4, 2, vl) ;

    __vm256 vmy0_r0 =  _vel_vfmkleq_mvl(_vel_vcmpsl_vvvl(vri_r0, _vel_vmulsl_vsvl(2, vry_r0, vl), vl), vl) ;
    __vm256 vmy1_r0 =  _vel_vfmklge_mvl(vry_r0, vl) ;
    __vm256 vmy2_r0 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutHeight,vry_r0, vl), vl) ;
    __vm256 vmy_r0 = _vel_andm_mmm(_vel_andm_mmm(vmy0_r0, vmy1_r0), vmy2_r0) ;

    __vm256 vmy0_r1 =  _vel_vfmkleq_mvl(_vel_vcmpsl_vvvl(vri_r1, _vel_vmulsl_vsvl(2, vry_r1, vl), vl), vl) ;
    __vm256 vmy1_r1 =  _vel_vfmklge_mvl(vry_r1, vl) ;
    __vm256 vmy2_r1 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutHeight,vry_r1, vl), vl) ;
    __vm256 vmy_r1 = _vel_andm_mmm(_vel_andm_mmm(vmy0_r1, vmy1_r1), vmy2_r1) ;

    __vm256 vmy0_r2 =  _vel_vfmkleq_mvl(_vel_vcmpsl_vvvl(vri_r2, _vel_vmulsl_vsvl(2, vry_r2, vl), vl), vl) ;
    __vm256 vmy1_r2 =  _vel_vfmklge_mvl(vry_r2, vl) ;
    __vm256 vmy2_r2 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutHeight,vry_r2, vl), vl) ;
    __vm256 vmy_r2 = _vel_andm_mmm(_vel_andm_mmm(vmy0_r2, vmy1_r2), vmy2_r2) ;

    __vm256 vmy0_r3 =  _vel_vfmkleq_mvl(_vel_vcmpsl_vvvl(vri_r3, _vel_vmulsl_vsvl(2, vry_r3, vl), vl), vl) ;
    __vm256 vmy1_r3 =  _vel_vfmklge_mvl(vry_r3, vl) ;
    __vm256 vmy2_r3 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutHeight,vry_r3, vl), vl) ;
    __vm256 vmy_r3 = _vel_andm_mmm(_vel_andm_mmm(vmy0_r3, vmy1_r3), vmy2_r3) ;

    __vm256 vmy0_r4 =  _vel_vfmkleq_mvl(_vel_vcmpsl_vvvl(vri_r4, _vel_vmulsl_vsvl(2, vry_r4, vl), vl), vl) ;
    __vm256 vmy1_r4 =  _vel_vfmklge_mvl(vry_r4, vl) ;
    __vm256 vmy2_r4 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutHeight,vry_r4, vl), vl) ;
    __vm256 vmy_r4 = _vel_andm_mmm(_vel_andm_mmm(vmy0_r4, vmy1_r4), vmy2_r4) ;

    __vr vrj_s0 = _vel_vaddsl_vsvl( 2, vrw, vl) ;
    __vr vrj_s1 = _vel_vaddsl_vsvl( 1, vrw, vl) ;
    __vr vrj_s2 = _vel_vaddsl_vsvl( 0, vrw, vl) ;
    __vr vrj_s3 = _vel_vaddsl_vsvl(-1, vrw, vl) ;
    __vr vrj_s4 = _vel_vaddsl_vsvl(-2, vrw, vl) ;

    __vr vrx_s0 = _vel_vdivsl_vvsl(vrj_s0, 2, vl) ;
    __vr vrx_s1 = _vel_vdivsl_vvsl(vrj_s1, 2, vl) ;
    __vr vrx_s2 = _vel_vdivsl_vvsl(vrj_s2, 2, vl) ;
    __vr vrx_s3 = _vel_vdivsl_vvsl(vrj_s3, 2, vl) ;
    __vr vrx_s4 = _vel_vdivsl_vvsl(vrj_s4, 2, vl) ;

    __vm256 vmx0_s0 =  _vel_vfmkleq_mvl(_vel_vcmpsl_vvvl(vrj_s0, _vel_vmulsl_vsvl(2, vrx_s0, vl), vl), vl) ;
    __vm256 vmx1_s0 =  _vel_vfmklge_mvl(vrx_s0, vl) ;
    __vm256 vmx2_s0 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutWidth,vrx_s0, vl), vl) ;
    __vm256 vmx_s0 = _vel_andm_mmm(_vel_andm_mmm(vmx0_s0, vmx1_s0), vmx2_s0) ;

    __vm256 vmx0_s1 =  _vel_vfmkleq_mvl(_vel_vcmpsl_vvvl(vrj_s1, _vel_vmulsl_vsvl(2, vrx_s1, vl), vl), vl) ;
    __vm256 vmx1_s1 =  _vel_vfmklge_mvl(vrx_s1, vl) ;
    __vm256 vmx2_s1 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutWidth,vrx_s1, vl), vl) ;
    __vm256 vmx_s1 = _vel_andm_mmm(_vel_andm_mmm(vmx0_s1, vmx1_s1), vmx2_s1) ;

    __vm256 vmx0_s2 =  _vel_vfmkleq_mvl(_vel_vcmpsl_vvvl(vrj_s2, _vel_vmulsl_vsvl(2, vrx_s2, vl), vl), vl) ;
    __vm256 vmx1_s2 =  _vel_vfmklge_mvl(vrx_s2, vl) ;
    __vm256 vmx2_s2 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutWidth,vrx_s2, vl), vl) ;
    __vm256 vmx_s2 = _vel_andm_mmm(_vel_andm_mmm(vmx0_s2, vmx1_s2), vmx2_s2) ;

    __vm256 vmx0_s3 =  _vel_vfmkleq_mvl(_vel_vcmpsl_vvvl(vrj_s3, _vel_vmulsl_vsvl(2, vrx_s3, vl), vl), vl) ;
    __vm256 vmx1_s3 =  _vel_vfmklge_mvl(vrx_s3, vl) ;
    __vm256 vmx2_s3 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutWidth,vrx_s3, vl), vl) ;
    __vm256 vmx_s3 = _vel_andm_mmm(_vel_andm_mmm(vmx0_s3, vmx1_s3), vmx2_s3) ;

    __vm256 vmx0_s4 =  _vel_vfmkleq_mvl(_vel_vcmpsl_vvvl(vrj_s4, _vel_vmulsl_vsvl(2, vrx_s4, vl), vl), vl) ;
    __vm256 vmx1_s4 =  _vel_vfmklge_mvl(vrx_s4, vl) ;
    __vm256 vmx2_s4 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutWidth,vrx_s4, vl), vl) ;
    __vm256 vmx_s4 = _vel_andm_mmm(_vel_andm_mmm(vmx0_s4, vmx1_s4), vmx2_s4) ;

    for (int64_t k=0; k<gOutChannelGroup; k++) {
      int64_t gOutIndex    = gOutGroupOffset + ((n * gOutChannel + k) * gOutHeight) * gOutWidth ;
      const float *pKerValue = pKernel + kernGroupOffset + (((k  ) * gInChannelGroup + c) * kernHeight) * kernWidth ;

#define VFADD_C1(VRGOUT, K, R, S)  {										\
	const float kerValue = pKerValue[(((K)*gInChannelGroup + 0) * kernHeight +(R)) * kernWidth + (S) ] ;	\
	vrsum_s##S = _vel_vfmads_vvsvl(vrsum_s##S, kerValue, VRGOUT, vl) ;					\
      }

      __vr vrgout_ptr_k0_r0s2 = _vel_vsfa_vvssl(_vel_vaddsl_vvvl(_vel_vmulsl_vsvl(gOutWidth, vry_r0, vl), vrx_s2, vl),
					 2,
					 (unsigned long)(pGOut+gOutIndex), vl) ;
      __vr vrgout_k0_r0s2 = _vel_vgtu_vvssml(vrgout_ptr_k0_r0s2, 0, 0, vmy_r0, vl) ;
      __vr vrgout_ptr_k0_r1s2 = _vel_vsfa_vvssl(_vel_vaddsl_vvvl(_vel_vmulsl_vsvl(gOutWidth, vry_r1, vl), vrx_s2, vl),
					 2,
					 (unsigned long)(pGOut+gOutIndex), vl) ;
      __vr vrgout_k0_r1s2 = _vel_vgtu_vvssml(vrgout_ptr_k0_r1s2, 0, 0, vmy_r1, vl) ;
      __vr vrgout_ptr_k0_r2s2 = _vel_vsfa_vvssl(_vel_vaddsl_vvvl(_vel_vmulsl_vsvl(gOutWidth, vry_r2, vl), vrx_s2, vl),
					 2,
					 (unsigned long)(pGOut+gOutIndex), vl) ;
      __vr vrgout_k0_r2s2 = _vel_vgtu_vvssml(vrgout_ptr_k0_r2s2, 0, 0, vmy_r2, vl) ;
      __vr vrgout_ptr_k0_r3s2 = _vel_vsfa_vvssl(_vel_vaddsl_vvvl(_vel_vmulsl_vsvl(gOutWidth, vry_r3, vl), vrx_s2, vl),
					 2,
					 (unsigned long)(pGOut+gOutIndex), vl) ;
      __vr vrgout_k0_r3s2 = _vel_vgtu_vvssml(vrgout_ptr_k0_r3s2, 0, 0, vmy_r3, vl) ;
      __vr vrgout_ptr_k0_r4s2 = _vel_vsfa_vvssl(_vel_vaddsl_vvvl(_vel_vmulsl_vsvl(gOutWidth, vry_r4, vl), vrx_s2, vl),
					 2,
					 (unsigned long)(pGOut+gOutIndex), vl) ;
      __vr vrgout_k0_r4s2 = _vel_vgtu_vvssml(vrgout_ptr_k0_r4s2, 0, 0, vmy_r4, vl) ;

      vrgout_k0_r0s2 = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrgout_k0_r0s2, vmy_r0, vl) ;
      VFADD_C1(vrgout_k0_r0s2, 0, 0, 0)
      VFADD_C1(vrgout_k0_r0s2, 0, 0, 1)
      VFADD_C1(vrgout_k0_r0s2, 0, 0, 2)
      VFADD_C1(vrgout_k0_r0s2, 0, 0, 3)
      VFADD_C1(vrgout_k0_r0s2, 0, 0, 4)

      vrgout_k0_r1s2 = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrgout_k0_r1s2, vmy_r1, vl) ;
      VFADD_C1(vrgout_k0_r1s2, 0, 1, 0)
      VFADD_C1(vrgout_k0_r1s2, 0, 1, 1)
      VFADD_C1(vrgout_k0_r1s2, 0, 1, 2)
      VFADD_C1(vrgout_k0_r1s2, 0, 1, 3)
      VFADD_C1(vrgout_k0_r1s2, 0, 1, 4)

      vrgout_k0_r2s2 = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrgout_k0_r2s2, vmy_r2, vl) ;
      VFADD_C1(vrgout_k0_r2s2, 0, 2, 0)
      VFADD_C1(vrgout_k0_r2s2, 0, 2, 1)
      VFADD_C1(vrgout_k0_r2s2, 0, 2, 2)
      VFADD_C1(vrgout_k0_r2s2, 0, 2, 3)
      VFADD_C1(vrgout_k0_r2s2, 0, 2, 4)

      vrgout_k0_r3s2 = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrgout_k0_r3s2, vmy_r3, vl) ;
      VFADD_C1(vrgout_k0_r3s2, 0, 3, 0)
      VFADD_C1(vrgout_k0_r3s2, 0, 3, 1)
      VFADD_C1(vrgout_k0_r3s2, 0, 3, 2)
      VFADD_C1(vrgout_k0_r3s2, 0, 3, 3)
      VFADD_C1(vrgout_k0_r3s2, 0, 3, 4)

      vrgout_k0_r4s2 = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrgout_k0_r4s2, vmy_r4, vl) ;
      VFADD_C1(vrgout_k0_r4s2, 0, 4, 0)
      VFADD_C1(vrgout_k0_r4s2, 0, 4, 1)
      VFADD_C1(vrgout_k0_r4s2, 0, 4, 2)
      VFADD_C1(vrgout_k0_r4s2, 0, 4, 3)
      VFADD_C1(vrgout_k0_r4s2, 0, 4, 4)

#undef VFADD_C1
    }

    vrsum_s0 = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.f, vl), _vel_vmv_vsvl( 2, vrsum_s0, vl), vmx_s0, vl) ;
    vrsum_s1 = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.f, vl), _vel_vmv_vsvl( 1, vrsum_s1, vl), vmx_s1, vl) ;
    vrsum_s2 = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.f, vl),                 vrsum_s2, vmx_s2, vl) ;
    vrsum_s3 = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.f, vl), _vel_vmv_vsvl(-1, vrsum_s3, vl), vmx_s3, vl) ;
    vrsum_s4 = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.f, vl), _vel_vmv_vsvl(-2, vrsum_s4, vl), vmx_s4, vl) ;
    __vr vrsum = _vel_vfadds_vvvl(_vel_vfadds_vvvl(vrsum_s0, _vel_vfadds_vvvl(vrsum_s1, vrsum_s2, vl), vl),
                                  _vel_vfadds_vvvl(vrsum_s3, vrsum_s4, vl), vl) ;
    _vel_vstu_vssl(vrsum, 4, pGIn+gInIndex, vl) ;

  } // gOutPixels
}


// force no-inline to avoid vr-spill in the most-inner loop
static __attribute__((noinline))  void c2(
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
    const int64_t nH
)
{
  __vr vrseq = _vel_vseq_vl(nH*gInWidth) ;
  __vr vrh  = _vel_vdivsl_vvsl(vrseq, gInWidth, nH*gInWidth) ;
  __vr vrw  = _vel_vsubsl_vvvl(vrseq, _vel_vmulul_vsvl(gInWidth,vrh, nH*gInWidth), nH*gInWidth) ;


  for (int64_t h=0; h<gInHeight; h+=nH) {
    const int64_t vl = gInWidth * (gInHeight - h < nH ? gInHeight - h : nH) ;
    const int64_t gip = h * gInWidth ;

    const int64_t gInIndex = gInGroupOffset + ((n * gInChannel + c) * gInHeight ) * gInWidth + gip ;

    __vr vrsum01_s0 = _vel_vbrdl_vsl(0UL, vl) ;
    __vr vrsum01_s1 = _vel_vbrdl_vsl(0UL, vl) ;
    __vr vrsum01_s2 = _vel_vbrdl_vsl(0UL, vl) ;
    __vr vrsum01_s3 = _vel_vbrdl_vsl(0UL, vl) ;
    __vr vrsum01_s4 = _vel_vbrdl_vsl(0UL, vl) ;

    __vr vri_r0 = _vel_vaddsl_vsvl(2-0+h, vrh, vl) ;
    __vr vri_r1 = _vel_vaddsl_vsvl(2-1+h, vrh, vl) ;
    __vr vri_r2 = _vel_vaddsl_vsvl(2-2+h, vrh, vl) ;
    __vr vri_r3 = _vel_vaddsl_vsvl(2-3+h, vrh, vl) ;
    __vr vri_r4 = _vel_vaddsl_vsvl(2-4+h, vrh, vl) ;

    __vr vry_r0 = _vel_vdivsl_vvsl(vri_r0, 2, vl) ;
    __vr vry_r1 = _vel_vdivsl_vvsl(vri_r1, 2, vl) ;
    __vr vry_r2 = _vel_vdivsl_vvsl(vri_r2, 2, vl) ;
    __vr vry_r3 = _vel_vdivsl_vvsl(vri_r3, 2, vl) ;
    __vr vry_r4 = _vel_vdivsl_vvsl(vri_r4, 2, vl) ;

    __vm256 vmy0_r0 =  _vel_vfmkleq_mvl(_vel_vcmpsl_vvvl(vri_r0, _vel_vmulsl_vsvl(2, vry_r0, vl), vl), vl) ;
    __vm256 vmy1_r0 =  _vel_vfmklge_mvl(vry_r0, vl) ;
    __vm256 vmy2_r0 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutHeight,vry_r0, vl), vl) ;
    __vm256 vmy_r0 = _vel_andm_mmm(_vel_andm_mmm(vmy0_r0, vmy1_r0), vmy2_r0) ;

    __vm256 vmy0_r1 =  _vel_vfmkleq_mvl(_vel_vcmpsl_vvvl(vri_r1, _vel_vmulsl_vsvl(2, vry_r1, vl), vl), vl) ;
    __vm256 vmy1_r1 =  _vel_vfmklge_mvl(vry_r1, vl) ;
    __vm256 vmy2_r1 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutHeight,vry_r1, vl), vl) ;
    __vm256 vmy_r1 = _vel_andm_mmm(_vel_andm_mmm(vmy0_r1, vmy1_r1), vmy2_r1) ;

    __vm256 vmy0_r2 =  _vel_vfmkleq_mvl(_vel_vcmpsl_vvvl(vri_r2, _vel_vmulsl_vsvl(2, vry_r2, vl), vl), vl) ;
    __vm256 vmy1_r2 =  _vel_vfmklge_mvl(vry_r2, vl) ;
    __vm256 vmy2_r2 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutHeight,vry_r2, vl), vl) ;
    __vm256 vmy_r2 = _vel_andm_mmm(_vel_andm_mmm(vmy0_r2, vmy1_r2), vmy2_r2) ;

    __vm256 vmy0_r3 =  _vel_vfmkleq_mvl(_vel_vcmpsl_vvvl(vri_r3, _vel_vmulsl_vsvl(2, vry_r3, vl), vl), vl) ;
    __vm256 vmy1_r3 =  _vel_vfmklge_mvl(vry_r3, vl) ;
    __vm256 vmy2_r3 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutHeight,vry_r3, vl), vl) ;
    __vm256 vmy_r3 = _vel_andm_mmm(_vel_andm_mmm(vmy0_r3, vmy1_r3), vmy2_r3) ;

    __vm256 vmy0_r4 =  _vel_vfmkleq_mvl(_vel_vcmpsl_vvvl(vri_r4, _vel_vmulsl_vsvl(2, vry_r4, vl), vl), vl) ;
    __vm256 vmy1_r4 =  _vel_vfmklge_mvl(vry_r4, vl) ;
    __vm256 vmy2_r4 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutHeight,vry_r4, vl), vl) ;
    __vm256 vmy_r4 = _vel_andm_mmm(_vel_andm_mmm(vmy0_r4, vmy1_r4), vmy2_r4) ;

    __vr vrj_s0 = _vel_vaddsl_vsvl( 2, vrw, vl) ;
    __vr vrj_s1 = _vel_vaddsl_vsvl( 1, vrw, vl) ;
    __vr vrj_s2 = _vel_vaddsl_vsvl( 0, vrw, vl) ;
    __vr vrj_s3 = _vel_vaddsl_vsvl(-1, vrw, vl) ;
    __vr vrj_s4 = _vel_vaddsl_vsvl(-2, vrw, vl) ;

    __vr vrx_s0 = _vel_vdivsl_vvsl(vrj_s0, 2, vl) ;
    __vr vrx_s1 = _vel_vdivsl_vvsl(vrj_s1, 2, vl) ;
    __vr vrx_s2 = _vel_vdivsl_vvsl(vrj_s2, 2, vl) ;
    __vr vrx_s3 = _vel_vdivsl_vvsl(vrj_s3, 2, vl) ;
    __vr vrx_s4 = _vel_vdivsl_vvsl(vrj_s4, 2, vl) ;

    __vm256 vmx0_s0 =  _vel_vfmkleq_mvl(_vel_vcmpsl_vvvl(vrj_s0, _vel_vmulsl_vsvl(2, vrx_s0, vl), vl), vl) ;
    __vm256 vmx1_s0 =  _vel_vfmklge_mvl(vrx_s0, vl) ;
    __vm256 vmx2_s0 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutWidth,vrx_s0, vl), vl) ;
    __vm256 vmx_s0 = _vel_andm_mmm(_vel_andm_mmm(vmx0_s0, vmx1_s0), vmx2_s0) ;

    __vm256 vmx0_s1 =  _vel_vfmkleq_mvl(_vel_vcmpsl_vvvl(vrj_s1, _vel_vmulsl_vsvl(2, vrx_s1, vl), vl), vl) ;
    __vm256 vmx1_s1 =  _vel_vfmklge_mvl(vrx_s1, vl) ;
    __vm256 vmx2_s1 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutWidth,vrx_s1, vl), vl) ;
    __vm256 vmx_s1 = _vel_andm_mmm(_vel_andm_mmm(vmx0_s1, vmx1_s1), vmx2_s1) ;

    __vm256 vmx0_s2 =  _vel_vfmkleq_mvl(_vel_vcmpsl_vvvl(vrj_s2, _vel_vmulsl_vsvl(2, vrx_s2, vl), vl), vl) ;
    __vm256 vmx1_s2 =  _vel_vfmklge_mvl(vrx_s2, vl) ;
    __vm256 vmx2_s2 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutWidth,vrx_s2, vl), vl) ;
    __vm256 vmx_s2 = _vel_andm_mmm(_vel_andm_mmm(vmx0_s2, vmx1_s2), vmx2_s2) ;

    __vm256 vmx0_s3 =  _vel_vfmkleq_mvl(_vel_vcmpsl_vvvl(vrj_s3, _vel_vmulsl_vsvl(2, vrx_s3, vl), vl), vl) ;
    __vm256 vmx1_s3 =  _vel_vfmklge_mvl(vrx_s3, vl) ;
    __vm256 vmx2_s3 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutWidth,vrx_s3, vl), vl) ;
    __vm256 vmx_s3 = _vel_andm_mmm(_vel_andm_mmm(vmx0_s3, vmx1_s3), vmx2_s3) ;

    __vm256 vmx0_s4 =  _vel_vfmkleq_mvl(_vel_vcmpsl_vvvl(vrj_s4, _vel_vmulsl_vsvl(2, vrx_s4, vl), vl), vl) ;
    __vm256 vmx1_s4 =  _vel_vfmklge_mvl(vrx_s4, vl) ;
    __vm256 vmx2_s4 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutWidth,vrx_s4, vl), vl) ;
    __vm256 vmx_s4 = _vel_andm_mmm(_vel_andm_mmm(vmx0_s4, vmx1_s4), vmx2_s4) ;

    for (int64_t k=0; k<gOutChannelGroup; k++) {
      int64_t gOutIndex    = gOutGroupOffset + ((n * gOutChannel + k) * gOutHeight) * gOutWidth ;
      const float *pKerValue = pKernel + kernGroupOffset + (((k  ) * gInChannelGroup + c) * kernHeight) * kernWidth ;

#define VFADD_C2(VRGOUT, K, R, S)  {													\
	const uint64_t kerValue01 = _vel_pack_f32p(pKerValue + (((K)*gInChannelGroup + 0) * kernHeight +(R)) * kernWidth + (S),		\
						  pKerValue + (((K)*gInChannelGroup + 1) * kernHeight +(R)) * kernWidth + (S)) ;	\
	__vr vrgoutP = _vel_vshf_vvvsl(VRGOUT, VRGOUT, VE_VSHUFFLE_YUZU, vl) ;		\
	vrsum01_s##S = _vel_pvfmad_vvsvl(vrsum01_s##S, kerValue01, vrgoutP, vl) ;	\
      }

      __vr vrgout_ptr_k0_r0s2 = _vel_vsfa_vvssl(_vel_vaddsl_vvvl(_vel_vmulsl_vsvl(gOutWidth, vry_r0, vl), vrx_s2, vl),
					 2,
					 (unsigned long)(pGOut+gOutIndex), vl) ;
      __vr vrgout_k0_r0s2 = _vel_vgtu_vvssml(vrgout_ptr_k0_r0s2, 0, 0, vmy_r0, vl) ;
      __vr vrgout_ptr_k0_r1s2 = _vel_vsfa_vvssl(_vel_vaddsl_vvvl(_vel_vmulsl_vsvl(gOutWidth, vry_r1, vl), vrx_s2, vl),
					 2,
					 (unsigned long)(pGOut+gOutIndex), vl) ;
      __vr vrgout_k0_r1s2 = _vel_vgtu_vvssml(vrgout_ptr_k0_r1s2, 0, 0, vmy_r1, vl) ;
      __vr vrgout_ptr_k0_r2s2 = _vel_vsfa_vvssl(_vel_vaddsl_vvvl(_vel_vmulsl_vsvl(gOutWidth, vry_r2, vl), vrx_s2, vl),
					 2,
					 (unsigned long)(pGOut+gOutIndex), vl) ;
      __vr vrgout_k0_r2s2 = _vel_vgtu_vvssml(vrgout_ptr_k0_r2s2, 0, 0, vmy_r2, vl) ;
      __vr vrgout_ptr_k0_r3s2 = _vel_vsfa_vvssl(_vel_vaddsl_vvvl(_vel_vmulsl_vsvl(gOutWidth, vry_r3, vl), vrx_s2, vl),
					 2,
					 (unsigned long)(pGOut+gOutIndex), vl) ;
      __vr vrgout_k0_r3s2 = _vel_vgtu_vvssml(vrgout_ptr_k0_r3s2, 0, 0, vmy_r3, vl) ;
      __vr vrgout_ptr_k0_r4s2 = _vel_vsfa_vvssl(_vel_vaddsl_vvvl(_vel_vmulsl_vsvl(gOutWidth, vry_r4, vl), vrx_s2, vl),
					 2,
					 (unsigned long)(pGOut+gOutIndex), vl) ;
      __vr vrgout_k0_r4s2 = _vel_vgtu_vvssml(vrgout_ptr_k0_r4s2, 0, 0, vmy_r4, vl) ;

      vrgout_k0_r0s2 = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrgout_k0_r0s2, vmy_r0, vl) ;
      VFADD_C2(vrgout_k0_r0s2, 0, 0, 0)
      VFADD_C2(vrgout_k0_r0s2, 0, 0, 1)
      VFADD_C2(vrgout_k0_r0s2, 0, 0, 2)
      VFADD_C2(vrgout_k0_r0s2, 0, 0, 3)
      VFADD_C2(vrgout_k0_r0s2, 0, 0, 4)

      vrgout_k0_r1s2 = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrgout_k0_r1s2, vmy_r1, vl) ;
      VFADD_C2(vrgout_k0_r1s2, 0, 1, 0)
      VFADD_C2(vrgout_k0_r1s2, 0, 1, 1)
      VFADD_C2(vrgout_k0_r1s2, 0, 1, 2)
      VFADD_C2(vrgout_k0_r1s2, 0, 1, 3)
      VFADD_C2(vrgout_k0_r1s2, 0, 1, 4)

      vrgout_k0_r2s2 = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrgout_k0_r2s2, vmy_r2, vl) ;
      VFADD_C2(vrgout_k0_r2s2, 0, 2, 0)
      VFADD_C2(vrgout_k0_r2s2, 0, 2, 1)
      VFADD_C2(vrgout_k0_r2s2, 0, 2, 2)
      VFADD_C2(vrgout_k0_r2s2, 0, 2, 3)
      VFADD_C2(vrgout_k0_r2s2, 0, 2, 4)

      vrgout_k0_r3s2 = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrgout_k0_r3s2, vmy_r3, vl) ;
      VFADD_C2(vrgout_k0_r3s2, 0, 3, 0)
      VFADD_C2(vrgout_k0_r3s2, 0, 3, 1)
      VFADD_C2(vrgout_k0_r3s2, 0, 3, 2)
      VFADD_C2(vrgout_k0_r3s2, 0, 3, 3)
      VFADD_C2(vrgout_k0_r3s2, 0, 3, 4)

      vrgout_k0_r4s2 = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrgout_k0_r4s2, vmy_r4, vl) ;
      VFADD_C2(vrgout_k0_r4s2, 0, 4, 0)
      VFADD_C2(vrgout_k0_r4s2, 0, 4, 1)
      VFADD_C2(vrgout_k0_r4s2, 0, 4, 2)
      VFADD_C2(vrgout_k0_r4s2, 0, 4, 3)
      VFADD_C2(vrgout_k0_r4s2, 0, 4, 4)

#undef VFADD_C2
    }

    vrsum01_s0 = _vel_vmrg_vvvml(_vel_vbrdl_vsl(0UL, vl), _vel_vmv_vsvl( 2, vrsum01_s0, vl), vmx_s0, vl) ;
    vrsum01_s1 = _vel_vmrg_vvvml(_vel_vbrdl_vsl(0UL, vl), _vel_vmv_vsvl( 1, vrsum01_s1, vl), vmx_s1, vl) ;
    vrsum01_s2 = _vel_vmrg_vvvml(_vel_vbrdl_vsl(0UL, vl),                 vrsum01_s2, vmx_s2, vl) ;
    vrsum01_s3 = _vel_vmrg_vvvml(_vel_vbrdl_vsl(0UL, vl), _vel_vmv_vsvl(-1, vrsum01_s3, vl), vmx_s3, vl) ;
    vrsum01_s4 = _vel_vmrg_vvvml(_vel_vbrdl_vsl(0UL, vl), _vel_vmv_vsvl(-2, vrsum01_s4, vl), vmx_s4, vl) ;
    __vr vrsum01 = _vel_pvfadd_vvvl(_vel_pvfadd_vvvl(vrsum01_s0, _vel_pvfadd_vvvl(vrsum01_s1, vrsum01_s2, vl), vl),
                                    _vel_pvfadd_vvvl(vrsum01_s3, vrsum01_s4, vl), vl) ;
    _vel_vstu_vssl(vrsum01, 4, pGIn+gInIndex, vl) ;
    _vel_vstl_vssl(vrsum01, 4, pGIn+gInIndex+  gInPixels, vl) ;

  } // gOutPixels
}


// force no-inline to avoid vr-spill in the most-inner loop
static __attribute__((noinline))  void c4(
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
    const int64_t nH
)
{
  __vr vrseq = _vel_vseq_vl(nH*gInWidth) ;
  __vr vrh  = _vel_vdivsl_vvsl(vrseq, gInWidth, nH*gInWidth) ;
  __vr vrw  = _vel_vsubsl_vvvl(vrseq, _vel_vmulul_vsvl(gInWidth,vrh, nH*gInWidth), nH*gInWidth) ;

  for (int64_t h=0; h<gInHeight; h+=nH) {
    const int64_t vl = gInWidth * (gInHeight - h < nH ? gInHeight - h : nH) ;
    const int64_t gip = h * gInWidth ;

    const int64_t gInIndex = gInGroupOffset + ((n * gInChannel + c) * gInHeight ) * gInWidth + gip ;

     ;

    __vr vrsum01_s0 = _vel_vbrdl_vsl(0UL, vl) ;
    __vr vrsum23_s0 = _vel_vbrdl_vsl(0UL, vl) ;

    __vr vrsum01_s1 = _vel_vbrdl_vsl(0UL, vl) ;
    __vr vrsum23_s1 = _vel_vbrdl_vsl(0UL, vl) ;

    __vr vrsum01_s2 = _vel_vbrdl_vsl(0UL, vl) ;
    __vr vrsum23_s2 = _vel_vbrdl_vsl(0UL, vl) ;

    __vr vrsum01_s3 = _vel_vbrdl_vsl(0UL, vl) ;
    __vr vrsum23_s3 = _vel_vbrdl_vsl(0UL, vl) ;

    __vr vrsum01_s4 = _vel_vbrdl_vsl(0UL, vl) ;
    __vr vrsum23_s4 = _vel_vbrdl_vsl(0UL, vl) ;

    __vr vri_r0 = _vel_vaddsl_vsvl(2-0+h, vrh, vl) ;
    __vr vri_r1 = _vel_vaddsl_vsvl(2-1+h, vrh, vl) ;
    __vr vri_r2 = _vel_vaddsl_vsvl(2-2+h, vrh, vl) ;
    __vr vri_r3 = _vel_vaddsl_vsvl(2-3+h, vrh, vl) ;
    __vr vri_r4 = _vel_vaddsl_vsvl(2-4+h, vrh, vl) ;

    __vr vry_r0 = _vel_vdivsl_vvsl(vri_r0, 2, vl) ;
    __vr vry_r1 = _vel_vdivsl_vvsl(vri_r1, 2, vl) ;
    __vr vry_r2 = _vel_vdivsl_vvsl(vri_r2, 2, vl) ;
    __vr vry_r3 = _vel_vdivsl_vvsl(vri_r3, 2, vl) ;
    __vr vry_r4 = _vel_vdivsl_vvsl(vri_r4, 2, vl) ;

    __vm256 vmy0_r0 =  _vel_vfmkleq_mvl(_vel_vcmpsl_vvvl(vri_r0, _vel_vmulsl_vsvl(2, vry_r0, vl), vl), vl) ;
    __vm256 vmy1_r0 =  _vel_vfmklge_mvl(vry_r0, vl) ;
    __vm256 vmy2_r0 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutHeight,vry_r0, vl), vl) ;
    __vm256 vmy_r0 = _vel_andm_mmm(_vel_andm_mmm(vmy0_r0, vmy1_r0), vmy2_r0) ;

    __vm256 vmy0_r1 =  _vel_vfmkleq_mvl(_vel_vcmpsl_vvvl(vri_r1, _vel_vmulsl_vsvl(2, vry_r1, vl), vl), vl) ;
    __vm256 vmy1_r1 =  _vel_vfmklge_mvl(vry_r1, vl) ;
    __vm256 vmy2_r1 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutHeight,vry_r1, vl), vl) ;
    __vm256 vmy_r1 = _vel_andm_mmm(_vel_andm_mmm(vmy0_r1, vmy1_r1), vmy2_r1) ;

    __vm256 vmy0_r2 =  _vel_vfmkleq_mvl(_vel_vcmpsl_vvvl(vri_r2, _vel_vmulsl_vsvl(2, vry_r2, vl), vl), vl) ;
    __vm256 vmy1_r2 =  _vel_vfmklge_mvl(vry_r2, vl) ;
    __vm256 vmy2_r2 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutHeight,vry_r2, vl), vl) ;
    __vm256 vmy_r2 = _vel_andm_mmm(_vel_andm_mmm(vmy0_r2, vmy1_r2), vmy2_r2) ;

    __vm256 vmy0_r3 =  _vel_vfmkleq_mvl(_vel_vcmpsl_vvvl(vri_r3, _vel_vmulsl_vsvl(2, vry_r3, vl), vl), vl) ;
    __vm256 vmy1_r3 =  _vel_vfmklge_mvl(vry_r3, vl) ;
    __vm256 vmy2_r3 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutHeight,vry_r3, vl), vl) ;
    __vm256 vmy_r3 = _vel_andm_mmm(_vel_andm_mmm(vmy0_r3, vmy1_r3), vmy2_r3) ;

    __vm256 vmy0_r4 =  _vel_vfmkleq_mvl(_vel_vcmpsl_vvvl(vri_r4, _vel_vmulsl_vsvl(2, vry_r4, vl), vl), vl) ;
    __vm256 vmy1_r4 =  _vel_vfmklge_mvl(vry_r4, vl) ;
    __vm256 vmy2_r4 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutHeight,vry_r4, vl), vl) ;
    __vm256 vmy_r4 = _vel_andm_mmm(_vel_andm_mmm(vmy0_r4, vmy1_r4), vmy2_r4) ;

    __vr vrj_s0 = _vel_vaddsl_vsvl( 2, vrw, vl) ;
    __vr vrj_s1 = _vel_vaddsl_vsvl( 1, vrw, vl) ;
    __vr vrj_s2 = _vel_vaddsl_vsvl( 0, vrw, vl) ;
    __vr vrj_s3 = _vel_vaddsl_vsvl(-1, vrw, vl) ;
    __vr vrj_s4 = _vel_vaddsl_vsvl(-2, vrw, vl) ;

    __vr vrx_s0 = _vel_vdivsl_vvsl(vrj_s0, 2, vl) ;
    __vr vrx_s1 = _vel_vdivsl_vvsl(vrj_s1, 2, vl) ;
    __vr vrx_s2 = _vel_vdivsl_vvsl(vrj_s2, 2, vl) ;
    __vr vrx_s3 = _vel_vdivsl_vvsl(vrj_s3, 2, vl) ;
    __vr vrx_s4 = _vel_vdivsl_vvsl(vrj_s4, 2, vl) ;

    __vm256 vmx0_s0 =  _vel_vfmkleq_mvl(_vel_vcmpsl_vvvl(vrj_s0, _vel_vmulsl_vsvl(2, vrx_s0, vl), vl), vl) ;
    __vm256 vmx1_s0 =  _vel_vfmklge_mvl(vrx_s0, vl) ;
    __vm256 vmx2_s0 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutWidth,vrx_s0, vl), vl) ;
    __vm256 vmx_s0 = _vel_andm_mmm(_vel_andm_mmm(vmx0_s0, vmx1_s0), vmx2_s0) ;

    __vm256 vmx0_s1 =  _vel_vfmkleq_mvl(_vel_vcmpsl_vvvl(vrj_s1, _vel_vmulsl_vsvl(2, vrx_s1, vl), vl), vl) ;
    __vm256 vmx1_s1 =  _vel_vfmklge_mvl(vrx_s1, vl) ;
    __vm256 vmx2_s1 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutWidth,vrx_s1, vl), vl) ;
    __vm256 vmx_s1 = _vel_andm_mmm(_vel_andm_mmm(vmx0_s1, vmx1_s1), vmx2_s1) ;

    __vm256 vmx0_s2 =  _vel_vfmkleq_mvl(_vel_vcmpsl_vvvl(vrj_s2, _vel_vmulsl_vsvl(2, vrx_s2, vl), vl), vl) ;
    __vm256 vmx1_s2 =  _vel_vfmklge_mvl(vrx_s2, vl) ;
    __vm256 vmx2_s2 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutWidth,vrx_s2, vl), vl) ;
    __vm256 vmx_s2 = _vel_andm_mmm(_vel_andm_mmm(vmx0_s2, vmx1_s2), vmx2_s2) ;

    __vm256 vmx0_s3 =  _vel_vfmkleq_mvl(_vel_vcmpsl_vvvl(vrj_s3, _vel_vmulsl_vsvl(2, vrx_s3, vl), vl), vl) ;
    __vm256 vmx1_s3 =  _vel_vfmklge_mvl(vrx_s3, vl) ;
    __vm256 vmx2_s3 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutWidth,vrx_s3, vl), vl) ;
    __vm256 vmx_s3 = _vel_andm_mmm(_vel_andm_mmm(vmx0_s3, vmx1_s3), vmx2_s3) ;

    __vm256 vmx0_s4 =  _vel_vfmkleq_mvl(_vel_vcmpsl_vvvl(vrj_s4, _vel_vmulsl_vsvl(2, vrx_s4, vl), vl), vl) ;
    __vm256 vmx1_s4 =  _vel_vfmklge_mvl(vrx_s4, vl) ;
    __vm256 vmx2_s4 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutWidth,vrx_s4, vl), vl) ;
    __vm256 vmx_s4 = _vel_andm_mmm(_vel_andm_mmm(vmx0_s4, vmx1_s4), vmx2_s4) ;

    for (int64_t k=0; k<gOutChannelGroup; k++) {
      int64_t gOutIndex    = gOutGroupOffset + ((n * gOutChannel + k) * gOutHeight) * gOutWidth ;
      const float *pKerValue = pKernel + kernGroupOffset + (((k  ) * gInChannelGroup + c) * kernHeight) * kernWidth ;

#define VFADD_C4(VRGOUT, K, R, S)  {													\
	const uint64_t kerValue01 = _vel_pack_f32p(pKerValue + (((K)*gInChannelGroup + 0) * kernHeight +(R)) * kernWidth + (S),		\
						  pKerValue + (((K)*gInChannelGroup + 1) * kernHeight +(R)) * kernWidth + (S)) ;	\
	const uint64_t kerValue23 = _vel_pack_f32p(pKerValue + (((K)*gInChannelGroup + 2) * kernHeight +(R)) * kernWidth + (S),		\
						  pKerValue + (((K)*gInChannelGroup + 3) * kernHeight +(R)) * kernWidth + (S)) ;	\
	__vr vrgoutP = _vel_vshf_vvvsl(VRGOUT, VRGOUT, VE_VSHUFFLE_YUZU, vl) ;		\
	vrsum01_s##S = _vel_pvfmad_vvsvl(vrsum01_s##S, kerValue01, vrgoutP, vl) ;	\
	vrsum23_s##S = _vel_pvfmad_vvsvl(vrsum23_s##S, kerValue23, vrgoutP, vl) ;	\
      }

      __vr vrgout_ptr_k0_r0s2 = _vel_vsfa_vvssl(_vel_vaddsl_vvvl(_vel_vmulsl_vsvl(gOutWidth, vry_r0, vl), vrx_s2, vl),
					 2,
					 (unsigned long)(pGOut+gOutIndex), vl) ;
      __vr vrgout_k0_r0s2 = _vel_vgtu_vvssml(vrgout_ptr_k0_r0s2, 0, 0, vmy_r0, vl) ;
      __vr vrgout_ptr_k0_r1s2 = _vel_vsfa_vvssl(_vel_vaddsl_vvvl(_vel_vmulsl_vsvl(gOutWidth, vry_r1, vl), vrx_s2, vl),
					 2,
					 (unsigned long)(pGOut+gOutIndex), vl) ;
      __vr vrgout_k0_r1s2 = _vel_vgtu_vvssml(vrgout_ptr_k0_r1s2, 0, 0, vmy_r1, vl) ;
      __vr vrgout_ptr_k0_r2s2 = _vel_vsfa_vvssl(_vel_vaddsl_vvvl(_vel_vmulsl_vsvl(gOutWidth, vry_r2, vl), vrx_s2, vl),
					 2,
					 (unsigned long)(pGOut+gOutIndex), vl) ;
      __vr vrgout_k0_r2s2 = _vel_vgtu_vvssml(vrgout_ptr_k0_r2s2, 0, 0, vmy_r2, vl) ;
      __vr vrgout_ptr_k0_r3s2 = _vel_vsfa_vvssl(_vel_vaddsl_vvvl(_vel_vmulsl_vsvl(gOutWidth, vry_r3, vl), vrx_s2, vl),
					 2,
					 (unsigned long)(pGOut+gOutIndex), vl) ;
      __vr vrgout_k0_r3s2 = _vel_vgtu_vvssml(vrgout_ptr_k0_r3s2, 0, 0, vmy_r3, vl) ;
      __vr vrgout_ptr_k0_r4s2 = _vel_vsfa_vvssl(_vel_vaddsl_vvvl(_vel_vmulsl_vsvl(gOutWidth, vry_r4, vl), vrx_s2, vl),
					 2,
					 (unsigned long)(pGOut+gOutIndex), vl) ;
      __vr vrgout_k0_r4s2 = _vel_vgtu_vvssml(vrgout_ptr_k0_r4s2, 0, 0, vmy_r4, vl) ;

      vrgout_k0_r0s2 = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrgout_k0_r0s2, vmy_r0, vl) ;
      VFADD_C4(vrgout_k0_r0s2, 0, 0, 0)
      VFADD_C4(vrgout_k0_r0s2, 0, 0, 1)
      VFADD_C4(vrgout_k0_r0s2, 0, 0, 2)
      VFADD_C4(vrgout_k0_r0s2, 0, 0, 3)
      VFADD_C4(vrgout_k0_r0s2, 0, 0, 4)

      vrgout_k0_r1s2 = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrgout_k0_r1s2, vmy_r1, vl) ;
      VFADD_C4(vrgout_k0_r1s2, 0, 1, 0)
      VFADD_C4(vrgout_k0_r1s2, 0, 1, 1)
      VFADD_C4(vrgout_k0_r1s2, 0, 1, 2)
      VFADD_C4(vrgout_k0_r1s2, 0, 1, 3)
      VFADD_C4(vrgout_k0_r1s2, 0, 1, 4)

      vrgout_k0_r2s2 = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrgout_k0_r2s2, vmy_r2, vl) ;
      VFADD_C4(vrgout_k0_r2s2, 0, 2, 0)
      VFADD_C4(vrgout_k0_r2s2, 0, 2, 1)
      VFADD_C4(vrgout_k0_r2s2, 0, 2, 2)
      VFADD_C4(vrgout_k0_r2s2, 0, 2, 3)
      VFADD_C4(vrgout_k0_r2s2, 0, 2, 4)

      vrgout_k0_r3s2 = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrgout_k0_r3s2, vmy_r3, vl) ;
      VFADD_C4(vrgout_k0_r3s2, 0, 3, 0)
      VFADD_C4(vrgout_k0_r3s2, 0, 3, 1)
      VFADD_C4(vrgout_k0_r3s2, 0, 3, 2)
      VFADD_C4(vrgout_k0_r3s2, 0, 3, 3)
      VFADD_C4(vrgout_k0_r3s2, 0, 3, 4)

      vrgout_k0_r4s2 = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrgout_k0_r4s2, vmy_r4, vl) ;
      VFADD_C4(vrgout_k0_r4s2, 0, 4, 0)
      VFADD_C4(vrgout_k0_r4s2, 0, 4, 1)
      VFADD_C4(vrgout_k0_r4s2, 0, 4, 2)
      VFADD_C4(vrgout_k0_r4s2, 0, 4, 3)
      VFADD_C4(vrgout_k0_r4s2, 0, 4, 4)

#undef VFADD_C4
    }

    vrsum01_s0 = _vel_vmrg_vvvml(_vel_vbrdl_vsl(0UL, vl), _vel_vmv_vsvl( 2, vrsum01_s0, vl), vmx_s0, vl) ;
    vrsum01_s1 = _vel_vmrg_vvvml(_vel_vbrdl_vsl(0UL, vl), _vel_vmv_vsvl( 1, vrsum01_s1, vl), vmx_s1, vl) ;
    vrsum01_s2 = _vel_vmrg_vvvml(_vel_vbrdl_vsl(0UL, vl),                 vrsum01_s2, vmx_s2, vl) ;
    vrsum01_s3 = _vel_vmrg_vvvml(_vel_vbrdl_vsl(0UL, vl), _vel_vmv_vsvl(-1, vrsum01_s3, vl), vmx_s3, vl) ;
    vrsum01_s4 = _vel_vmrg_vvvml(_vel_vbrdl_vsl(0UL, vl), _vel_vmv_vsvl(-2, vrsum01_s4, vl), vmx_s4, vl) ;
    __vr vrsum01 = _vel_pvfadd_vvvl(_vel_pvfadd_vvvl(vrsum01_s0, _vel_pvfadd_vvvl(vrsum01_s1, vrsum01_s2, vl), vl),
                                    _vel_pvfadd_vvvl(vrsum01_s3, vrsum01_s4, vl), vl) ;
    _vel_vstu_vssl(vrsum01, 4, pGIn+gInIndex, vl) ;
    _vel_vstl_vssl(vrsum01, 4, pGIn+gInIndex+  gInPixels, vl) ;

    vrsum23_s0 = _vel_vmrg_vvvml(_vel_vbrdl_vsl(0UL, vl), _vel_vmv_vsvl( 2, vrsum23_s0, vl), vmx_s0, vl) ;
    vrsum23_s1 = _vel_vmrg_vvvml(_vel_vbrdl_vsl(0UL, vl), _vel_vmv_vsvl( 1, vrsum23_s1, vl), vmx_s1, vl) ;
    vrsum23_s2 = _vel_vmrg_vvvml(_vel_vbrdl_vsl(0UL, vl),                 vrsum23_s2, vmx_s2, vl) ;
    vrsum23_s3 = _vel_vmrg_vvvml(_vel_vbrdl_vsl(0UL, vl), _vel_vmv_vsvl(-1, vrsum23_s3, vl), vmx_s3, vl) ;
    vrsum23_s4 = _vel_vmrg_vvvml(_vel_vbrdl_vsl(0UL, vl), _vel_vmv_vsvl(-2, vrsum23_s4, vl), vmx_s4, vl) ;
    __vr vrsum23 = _vel_pvfadd_vvvl(_vel_pvfadd_vvvl(vrsum23_s0, _vel_pvfadd_vvvl(vrsum23_s1, vrsum23_s2, vl), vl),
                                    _vel_pvfadd_vvvl(vrsum23_s3, vrsum23_s4, vl), vl) ;
    _vel_vstu_vssl(vrsum23, 4, pGIn+gInIndex+2*gInPixels, vl) ;
    _vel_vstl_vssl(vrsum23, 4, pGIn+gInIndex+3*gInPixels, vl) ;


  } // gOutPixels
}

// force no-inline to avoid vr-spill in the most-inner loop
static __attribute__((noinline))  void c8(
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
    const int64_t nH
)
{
  __vr vrseq = _vel_vseq_vl(nH*gInWidth) ;
  __vr vrh  = _vel_vdivsl_vvsl(vrseq, gInWidth, nH*gInWidth) ;
  __vr vrw  = _vel_vsubsl_vvvl(vrseq, _vel_vmulul_vsvl(gInWidth,vrh, nH*gInWidth), nH*gInWidth) ;

  for (int64_t h=0; h<gInHeight; h+=nH) {
    const int64_t vl = gInWidth * (gInHeight - h < nH ? gInHeight - h : nH) ;
    const int64_t gip = h * gInWidth ;

    const int64_t gInIndex = gInGroupOffset + ((n * gInChannel + c) * gInHeight ) * gInWidth + gip ;

    __vr vrsum01_s0 = _vel_vbrdl_vsl(0UL, vl) ;
    __vr vrsum23_s0 = _vel_vbrdl_vsl(0UL, vl) ;
    __vr vrsum45_s0 = _vel_vbrdl_vsl(0UL, vl) ;
    __vr vrsum67_s0 = _vel_vbrdl_vsl(0UL, vl) ;

    __vr vrsum01_s1 = _vel_vbrdl_vsl(0UL, vl) ;
    __vr vrsum23_s1 = _vel_vbrdl_vsl(0UL, vl) ;
    __vr vrsum45_s1 = _vel_vbrdl_vsl(0UL, vl) ;
    __vr vrsum67_s1 = _vel_vbrdl_vsl(0UL, vl) ;

    __vr vrsum01_s2 = _vel_vbrdl_vsl(0UL, vl) ;
    __vr vrsum23_s2 = _vel_vbrdl_vsl(0UL, vl) ;
    __vr vrsum45_s2 = _vel_vbrdl_vsl(0UL, vl) ;
    __vr vrsum67_s2 = _vel_vbrdl_vsl(0UL, vl) ;

    __vr vrsum01_s3 = _vel_vbrdl_vsl(0UL, vl) ;
    __vr vrsum23_s3 = _vel_vbrdl_vsl(0UL, vl) ;
    __vr vrsum45_s3 = _vel_vbrdl_vsl(0UL, vl) ;
    __vr vrsum67_s3 = _vel_vbrdl_vsl(0UL, vl) ;

    __vr vrsum01_s4 = _vel_vbrdl_vsl(0UL, vl) ;
    __vr vrsum23_s4 = _vel_vbrdl_vsl(0UL, vl) ;
    __vr vrsum45_s4 = _vel_vbrdl_vsl(0UL, vl) ;
    __vr vrsum67_s4 = _vel_vbrdl_vsl(0UL, vl) ;

    __vr vri_r0 = _vel_vaddsl_vsvl(2-0+h, vrh, vl) ;
    __vr vri_r1 = _vel_vaddsl_vsvl(2-1+h, vrh, vl) ;
    __vr vri_r2 = _vel_vaddsl_vsvl(2-2+h, vrh, vl) ;
    __vr vri_r3 = _vel_vaddsl_vsvl(2-3+h, vrh, vl) ;
    __vr vri_r4 = _vel_vaddsl_vsvl(2-4+h, vrh, vl) ;

    __vr vry_r0 = _vel_vdivsl_vvsl(vri_r0, 2, vl) ;
    __vr vry_r1 = _vel_vdivsl_vvsl(vri_r1, 2, vl) ;
    __vr vry_r2 = _vel_vdivsl_vvsl(vri_r2, 2, vl) ;
    __vr vry_r3 = _vel_vdivsl_vvsl(vri_r3, 2, vl) ;
    __vr vry_r4 = _vel_vdivsl_vvsl(vri_r4, 2, vl) ;

    __vm256 vmy0_r0 =  _vel_vfmkleq_mvl(_vel_vcmpsl_vvvl(vri_r0, _vel_vmulsl_vsvl(2, vry_r0, vl), vl), vl) ;
    __vm256 vmy1_r0 =  _vel_vfmklge_mvl(vry_r0, vl) ;
    __vm256 vmy2_r0 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutHeight,vry_r0, vl), vl) ;
    __vm256 vmy_r0 = _vel_andm_mmm(_vel_andm_mmm(vmy0_r0, vmy1_r0), vmy2_r0) ;

    __vm256 vmy0_r1 =  _vel_vfmkleq_mvl(_vel_vcmpsl_vvvl(vri_r1, _vel_vmulsl_vsvl(2, vry_r1, vl), vl), vl) ;
    __vm256 vmy1_r1 =  _vel_vfmklge_mvl(vry_r1, vl) ;
    __vm256 vmy2_r1 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutHeight,vry_r1, vl), vl) ;
    __vm256 vmy_r1 = _vel_andm_mmm(_vel_andm_mmm(vmy0_r1, vmy1_r1), vmy2_r1) ;

    __vm256 vmy0_r2 =  _vel_vfmkleq_mvl(_vel_vcmpsl_vvvl(vri_r2, _vel_vmulsl_vsvl(2, vry_r2, vl), vl), vl) ;
    __vm256 vmy1_r2 =  _vel_vfmklge_mvl(vry_r2, vl) ;
    __vm256 vmy2_r2 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutHeight,vry_r2, vl), vl) ;
    __vm256 vmy_r2 = _vel_andm_mmm(_vel_andm_mmm(vmy0_r2, vmy1_r2), vmy2_r2) ;

    __vm256 vmy0_r3 =  _vel_vfmkleq_mvl(_vel_vcmpsl_vvvl(vri_r3, _vel_vmulsl_vsvl(2, vry_r3, vl), vl), vl) ;
    __vm256 vmy1_r3 =  _vel_vfmklge_mvl(vry_r3, vl) ;
    __vm256 vmy2_r3 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutHeight,vry_r3, vl), vl) ;
    __vm256 vmy_r3 = _vel_andm_mmm(_vel_andm_mmm(vmy0_r3, vmy1_r3), vmy2_r3) ;

    __vm256 vmy0_r4 =  _vel_vfmkleq_mvl(_vel_vcmpsl_vvvl(vri_r4, _vel_vmulsl_vsvl(2, vry_r4, vl), vl), vl) ;
    __vm256 vmy1_r4 =  _vel_vfmklge_mvl(vry_r4, vl) ;
    __vm256 vmy2_r4 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutHeight,vry_r4, vl), vl) ;
    __vm256 vmy_r4 = _vel_andm_mmm(_vel_andm_mmm(vmy0_r4, vmy1_r4), vmy2_r4) ;

    __vr vrj_s0 = _vel_vaddsl_vsvl( 2, vrw, vl) ;
    __vr vrj_s1 = _vel_vaddsl_vsvl( 1, vrw, vl) ;
    __vr vrj_s2 = _vel_vaddsl_vsvl( 0, vrw, vl) ;
    __vr vrj_s3 = _vel_vaddsl_vsvl(-1, vrw, vl) ;
    __vr vrj_s4 = _vel_vaddsl_vsvl(-2, vrw, vl) ;

    __vr vrx_s0 = _vel_vdivsl_vvsl(vrj_s0, 2, vl) ;
    __vr vrx_s1 = _vel_vdivsl_vvsl(vrj_s1, 2, vl) ;
    __vr vrx_s2 = _vel_vdivsl_vvsl(vrj_s2, 2, vl) ;
    __vr vrx_s3 = _vel_vdivsl_vvsl(vrj_s3, 2, vl) ;
    __vr vrx_s4 = _vel_vdivsl_vvsl(vrj_s4, 2, vl) ;

    __vm256 vmx0_s0 =  _vel_vfmkleq_mvl(_vel_vcmpsl_vvvl(vrj_s0, _vel_vmulsl_vsvl(2, vrx_s0, vl), vl), vl) ;
    __vm256 vmx1_s0 =  _vel_vfmklge_mvl(vrx_s0, vl) ;
    __vm256 vmx2_s0 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutWidth,vrx_s0, vl), vl) ;
    __vm256 vmx_s0 = _vel_andm_mmm(_vel_andm_mmm(vmx0_s0, vmx1_s0), vmx2_s0) ;

    __vm256 vmx0_s1 =  _vel_vfmkleq_mvl(_vel_vcmpsl_vvvl(vrj_s1, _vel_vmulsl_vsvl(2, vrx_s1, vl), vl), vl) ;
    __vm256 vmx1_s1 =  _vel_vfmklge_mvl(vrx_s1, vl) ;
    __vm256 vmx2_s1 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutWidth,vrx_s1, vl), vl) ;
    __vm256 vmx_s1 = _vel_andm_mmm(_vel_andm_mmm(vmx0_s1, vmx1_s1), vmx2_s1) ;

    __vm256 vmx0_s2 =  _vel_vfmkleq_mvl(_vel_vcmpsl_vvvl(vrj_s2, _vel_vmulsl_vsvl(2, vrx_s2, vl), vl), vl) ;
    __vm256 vmx1_s2 =  _vel_vfmklge_mvl(vrx_s2, vl) ;
    __vm256 vmx2_s2 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutWidth,vrx_s2, vl), vl) ;
    __vm256 vmx_s2 = _vel_andm_mmm(_vel_andm_mmm(vmx0_s2, vmx1_s2), vmx2_s2) ;

    __vm256 vmx0_s3 =  _vel_vfmkleq_mvl(_vel_vcmpsl_vvvl(vrj_s3, _vel_vmulsl_vsvl(2, vrx_s3, vl), vl), vl) ;
    __vm256 vmx1_s3 =  _vel_vfmklge_mvl(vrx_s3, vl) ;
    __vm256 vmx2_s3 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutWidth,vrx_s3, vl), vl) ;
    __vm256 vmx_s3 = _vel_andm_mmm(_vel_andm_mmm(vmx0_s3, vmx1_s3), vmx2_s3) ;

    __vm256 vmx0_s4 =  _vel_vfmkleq_mvl(_vel_vcmpsl_vvvl(vrj_s4, _vel_vmulsl_vsvl(2, vrx_s4, vl), vl), vl) ;
    __vm256 vmx1_s4 =  _vel_vfmklge_mvl(vrx_s4, vl) ;
    __vm256 vmx2_s4 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutWidth,vrx_s4, vl), vl) ;
    __vm256 vmx_s4 = _vel_andm_mmm(_vel_andm_mmm(vmx0_s4, vmx1_s4), vmx2_s4) ;

    for (int64_t k=0; k<gOutChannelGroup; k++) {
      int64_t gOutIndex    = gOutGroupOffset + ((n * gOutChannel + k) * gOutHeight) * gOutWidth ;
      const float *pKerValue = pKernel + kernGroupOffset + (((k  ) * gInChannelGroup + c) * kernHeight) * kernWidth ;

#define VFADD_C8(VRGOUT, K, R, S)  {													\
	const uint64_t kerValue01 = _vel_pack_f32p(pKerValue + (((K)*gInChannelGroup + 0) * kernHeight +(R)) * kernWidth + (S),		\
						   pKerValue + (((K)*gInChannelGroup + 1) * kernHeight +(R)) * kernWidth + (S)) ;	\
	const uint64_t kerValue23 = _vel_pack_f32p(pKerValue + (((K)*gInChannelGroup + 2) * kernHeight +(R)) * kernWidth + (S),		\
						   pKerValue + (((K)*gInChannelGroup + 3) * kernHeight +(R)) * kernWidth + (S)) ;	\
	const uint64_t kerValue45 = _vel_pack_f32p(pKerValue + (((K)*gInChannelGroup + 4) * kernHeight +(R)) * kernWidth + (S),		\
						   pKerValue + (((K)*gInChannelGroup + 5) * kernHeight +(R)) * kernWidth + (S)) ;	\
	const uint64_t kerValue67 = _vel_pack_f32p(pKerValue + (((K)*gInChannelGroup + 6) * kernHeight +(R)) * kernWidth + (S),		\
						   pKerValue + (((K)*gInChannelGroup + 7) * kernHeight +(R)) * kernWidth + (S)) ;	\
	__vr vrgoutP = _vel_vshf_vvvsl(VRGOUT, VRGOUT, VE_VSHUFFLE_YUZU, vl) ;	\
	vrsum01_s##S = _vel_pvfmad_vvsvl(vrsum01_s##S, kerValue01, vrgoutP, vl) ;	\
	vrsum23_s##S = _vel_pvfmad_vvsvl(vrsum23_s##S, kerValue23, vrgoutP, vl) ;	\
	vrsum45_s##S = _vel_pvfmad_vvsvl(vrsum45_s##S, kerValue45, vrgoutP, vl) ;	\
	vrsum67_s##S = _vel_pvfmad_vvsvl(vrsum67_s##S, kerValue67, vrgoutP, vl) ;	\
      }

      __vr vrgout_ptr_k0_r0s2 = _vel_vsfa_vvssl(_vel_vaddsl_vvvl(_vel_vmulsl_vsvl(gOutWidth, vry_r0, vl), vrx_s2, vl),
					 2,
					 (unsigned long)(pGOut+gOutIndex), vl) ;
      __vr vrgout_k0_r0s2 = _vel_vgtu_vvssml(vrgout_ptr_k0_r0s2, 0, 0, vmy_r0, vl) ;
      __vr vrgout_ptr_k0_r1s2 = _vel_vsfa_vvssl(_vel_vaddsl_vvvl(_vel_vmulsl_vsvl(gOutWidth, vry_r1, vl), vrx_s2, vl),
					 2,
					 (unsigned long)(pGOut+gOutIndex), vl) ;
      __vr vrgout_k0_r1s2 = _vel_vgtu_vvssml(vrgout_ptr_k0_r1s2, 0, 0, vmy_r1, vl) ;
      __vr vrgout_ptr_k0_r2s2 = _vel_vsfa_vvssl(_vel_vaddsl_vvvl(_vel_vmulsl_vsvl(gOutWidth, vry_r2, vl), vrx_s2, vl),
					 2,
					 (unsigned long)(pGOut+gOutIndex), vl) ;
      __vr vrgout_k0_r2s2 = _vel_vgtu_vvssml(vrgout_ptr_k0_r2s2, 0, 0, vmy_r2, vl) ;
      __vr vrgout_ptr_k0_r3s2 = _vel_vsfa_vvssl(_vel_vaddsl_vvvl(_vel_vmulsl_vsvl(gOutWidth, vry_r3, vl), vrx_s2, vl),
					 2,
					 (unsigned long)(pGOut+gOutIndex), vl) ;
      __vr vrgout_k0_r3s2 = _vel_vgtu_vvssml(vrgout_ptr_k0_r3s2, 0, 0, vmy_r3, vl) ;
      __vr vrgout_ptr_k0_r4s2 = _vel_vsfa_vvssl(_vel_vaddsl_vvvl(_vel_vmulsl_vsvl(gOutWidth, vry_r4, vl), vrx_s2, vl),
					 2,
					 (unsigned long)(pGOut+gOutIndex), vl) ;
      __vr vrgout_k0_r4s2 = _vel_vgtu_vvssml(vrgout_ptr_k0_r4s2, 0, 0, vmy_r4, vl) ;

      vrgout_k0_r0s2 = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrgout_k0_r0s2, vmy_r0, vl) ;
      VFADD_C8(vrgout_k0_r0s2, 0, 0, 0)
      VFADD_C8(vrgout_k0_r0s2, 0, 0, 1)
      VFADD_C8(vrgout_k0_r0s2, 0, 0, 2)
      VFADD_C8(vrgout_k0_r0s2, 0, 0, 3)
      VFADD_C8(vrgout_k0_r0s2, 0, 0, 4)

      vrgout_k0_r1s2 = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrgout_k0_r1s2, vmy_r1, vl) ;
      VFADD_C8(vrgout_k0_r1s2, 0, 1, 0)
      VFADD_C8(vrgout_k0_r1s2, 0, 1, 1)
      VFADD_C8(vrgout_k0_r1s2, 0, 1, 2)
      VFADD_C8(vrgout_k0_r1s2, 0, 1, 3)
      VFADD_C8(vrgout_k0_r1s2, 0, 1, 4)

      vrgout_k0_r2s2 = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrgout_k0_r2s2, vmy_r2, vl) ;
      VFADD_C8(vrgout_k0_r2s2, 0, 2, 0)
      VFADD_C8(vrgout_k0_r2s2, 0, 2, 1)
      VFADD_C8(vrgout_k0_r2s2, 0, 2, 2)
      VFADD_C8(vrgout_k0_r2s2, 0, 2, 3)
      VFADD_C8(vrgout_k0_r2s2, 0, 2, 4)

      vrgout_k0_r3s2 = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrgout_k0_r3s2, vmy_r3, vl) ;
      VFADD_C8(vrgout_k0_r3s2, 0, 3, 0)
      VFADD_C8(vrgout_k0_r3s2, 0, 3, 1)
      VFADD_C8(vrgout_k0_r3s2, 0, 3, 2)
      VFADD_C8(vrgout_k0_r3s2, 0, 3, 3)
      VFADD_C8(vrgout_k0_r3s2, 0, 3, 4)

      vrgout_k0_r4s2 = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrgout_k0_r4s2, vmy_r4, vl) ;
      VFADD_C8(vrgout_k0_r4s2, 0, 4, 0)
      VFADD_C8(vrgout_k0_r4s2, 0, 4, 1)
      VFADD_C8(vrgout_k0_r4s2, 0, 4, 2)
      VFADD_C8(vrgout_k0_r4s2, 0, 4, 3)
      VFADD_C8(vrgout_k0_r4s2, 0, 4, 4)

#undef VFADD_C8
    }

    vrsum01_s0 = _vel_vmrg_vvvml(_vel_vbrdl_vsl(0UL, vl), _vel_vmv_vsvl( 2, vrsum01_s0, vl), vmx_s0, vl) ;
    vrsum01_s1 = _vel_vmrg_vvvml(_vel_vbrdl_vsl(0UL, vl), _vel_vmv_vsvl( 1, vrsum01_s1, vl), vmx_s1, vl) ;
    vrsum01_s2 = _vel_vmrg_vvvml(_vel_vbrdl_vsl(0UL, vl),                 vrsum01_s2, vmx_s2, vl) ;
    vrsum01_s3 = _vel_vmrg_vvvml(_vel_vbrdl_vsl(0UL, vl), _vel_vmv_vsvl(-1, vrsum01_s3, vl), vmx_s3, vl) ;
    vrsum01_s4 = _vel_vmrg_vvvml(_vel_vbrdl_vsl(0UL, vl), _vel_vmv_vsvl(-2, vrsum01_s4, vl), vmx_s4, vl) ;
    __vr vrsum01 = _vel_pvfadd_vvvl(_vel_pvfadd_vvvl(vrsum01_s0, _vel_pvfadd_vvvl(vrsum01_s1, vrsum01_s2, vl), vl),
                                    _vel_pvfadd_vvvl(vrsum01_s3, vrsum01_s4, vl), vl) ;
    _vel_vstu_vssl(vrsum01, 4, pGIn+gInIndex, vl) ;
    _vel_vstl_vssl(vrsum01, 4, pGIn+gInIndex+  gInPixels, vl) ;

    vrsum23_s0 = _vel_vmrg_vvvml(_vel_vbrdl_vsl(0UL, vl), _vel_vmv_vsvl( 2, vrsum23_s0, vl), vmx_s0, vl) ;
    vrsum23_s1 = _vel_vmrg_vvvml(_vel_vbrdl_vsl(0UL, vl), _vel_vmv_vsvl( 1, vrsum23_s1, vl), vmx_s1, vl) ;
    vrsum23_s2 = _vel_vmrg_vvvml(_vel_vbrdl_vsl(0UL, vl),                 vrsum23_s2, vmx_s2, vl) ;
    vrsum23_s3 = _vel_vmrg_vvvml(_vel_vbrdl_vsl(0UL, vl), _vel_vmv_vsvl(-1, vrsum23_s3, vl), vmx_s3, vl) ;
    vrsum23_s4 = _vel_vmrg_vvvml(_vel_vbrdl_vsl(0UL, vl), _vel_vmv_vsvl(-2, vrsum23_s4, vl), vmx_s4, vl) ;
    __vr vrsum23 = _vel_pvfadd_vvvl(_vel_pvfadd_vvvl(vrsum23_s0, _vel_pvfadd_vvvl(vrsum23_s1, vrsum23_s2, vl), vl),
                                    _vel_pvfadd_vvvl(vrsum23_s3, vrsum23_s4, vl), vl) ;
    _vel_vstu_vssl(vrsum23, 4, pGIn+gInIndex+2*gInPixels, vl) ;
    _vel_vstl_vssl(vrsum23, 4, pGIn+gInIndex+3*gInPixels, vl) ;

    vrsum45_s0 = _vel_vmrg_vvvml(_vel_vbrdl_vsl(0UL, vl), _vel_vmv_vsvl( 2, vrsum45_s0, vl), vmx_s0, vl) ;
    vrsum45_s1 = _vel_vmrg_vvvml(_vel_vbrdl_vsl(0UL, vl), _vel_vmv_vsvl( 1, vrsum45_s1, vl), vmx_s1, vl) ;
    vrsum45_s2 = _vel_vmrg_vvvml(_vel_vbrdl_vsl(0UL, vl),                 vrsum45_s2, vmx_s2, vl) ;
    vrsum45_s3 = _vel_vmrg_vvvml(_vel_vbrdl_vsl(0UL, vl), _vel_vmv_vsvl(-1, vrsum45_s3, vl), vmx_s3, vl) ;
    vrsum45_s4 = _vel_vmrg_vvvml(_vel_vbrdl_vsl(0UL, vl), _vel_vmv_vsvl(-2, vrsum45_s4, vl), vmx_s4, vl) ;
    __vr vrsum45 = _vel_pvfadd_vvvl(_vel_pvfadd_vvvl(vrsum45_s0, _vel_pvfadd_vvvl(vrsum45_s1, vrsum45_s2, vl), vl),
                                    _vel_pvfadd_vvvl(vrsum45_s3, vrsum45_s4, vl), vl) ;
    _vel_vstu_vssl(vrsum45, 4, pGIn+gInIndex+4*gInPixels, vl) ;
    _vel_vstl_vssl(vrsum45, 4, pGIn+gInIndex+5*gInPixels, vl) ;

    vrsum67_s0 = _vel_vmrg_vvvml(_vel_vbrdl_vsl(0UL, vl), _vel_vmv_vsvl( 2, vrsum67_s0, vl), vmx_s0, vl) ;
    vrsum67_s1 = _vel_vmrg_vvvml(_vel_vbrdl_vsl(0UL, vl), _vel_vmv_vsvl( 1, vrsum67_s1, vl), vmx_s1, vl) ;
    vrsum67_s2 = _vel_vmrg_vvvml(_vel_vbrdl_vsl(0UL, vl),                 vrsum67_s2, vmx_s2, vl) ;
    vrsum67_s3 = _vel_vmrg_vvvml(_vel_vbrdl_vsl(0UL, vl), _vel_vmv_vsvl(-1, vrsum67_s3, vl), vmx_s3, vl) ;
    vrsum67_s4 = _vel_vmrg_vvvml(_vel_vbrdl_vsl(0UL, vl), _vel_vmv_vsvl(-2, vrsum67_s4, vl), vmx_s4, vl) ;
    __vr vrsum67 = _vel_pvfadd_vvvl(_vel_pvfadd_vvvl(vrsum67_s0, _vel_pvfadd_vvvl(vrsum67_s1, vrsum67_s2, vl), vl),
                                    _vel_pvfadd_vvvl(vrsum67_s3, vrsum67_s4, vl), vl) ;
    _vel_vstu_vssl(vrsum67, 4, pGIn+gInIndex+6*gInPixels, vl) ;
    _vel_vstl_vssl(vrsum67, 4, pGIn+gInIndex+7*gInPixels, vl) ;

  } // gOutPixels
}

// force no-inline to avoid vr-spill in the most-inner loop
static __attribute__((noinline))  void c16(
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
    const int64_t nH
)
{
  __vr vrseq = _vel_vseq_vl(nH*gInWidth) ;
  __vr vrh  = _vel_vdivsl_vvsl(vrseq, gInWidth, nH*gInWidth) ;
  __vr vrw  = _vel_vsubsl_vvvl(vrseq, _vel_vmulul_vsvl(gInWidth,vrh, nH*gInWidth), nH*gInWidth) ;

  for (int64_t h=0; h<gInHeight; h+=nH) {
    const int64_t vl = gInWidth * (gInHeight - h < nH ? gInHeight - h : nH) ;
    const int64_t gip = h * gInWidth ;

    const int64_t gInIndex = gInGroupOffset + ((n * gInChannel + c) * gInHeight ) * gInWidth + gip ;

    __vr vrsum01_s0 = _vel_vbrdl_vsl(0UL, vl) ;
    __vr vrsum23_s0 = _vel_vbrdl_vsl(0UL, vl) ;
    __vr vrsum45_s0 = _vel_vbrdl_vsl(0UL, vl) ;
    __vr vrsum67_s0 = _vel_vbrdl_vsl(0UL, vl) ;
    __vr vrsum89_s0 = _vel_vbrdl_vsl(0UL, vl) ;
    __vr vrsumAB_s0 = _vel_vbrdl_vsl(0UL, vl) ;
    __vr vrsumCD_s0 = _vel_vbrdl_vsl(0UL, vl) ;
    __vr vrsumEF_s0 = _vel_vbrdl_vsl(0UL, vl) ;

    __vr vrsum01_s1 = _vel_vbrdl_vsl(0UL, vl) ;
    __vr vrsum23_s1 = _vel_vbrdl_vsl(0UL, vl) ;
    __vr vrsum45_s1 = _vel_vbrdl_vsl(0UL, vl) ;
    __vr vrsum67_s1 = _vel_vbrdl_vsl(0UL, vl) ;
    __vr vrsum89_s1 = _vel_vbrdl_vsl(0UL, vl) ;
    __vr vrsumAB_s1 = _vel_vbrdl_vsl(0UL, vl) ;
    __vr vrsumCD_s1 = _vel_vbrdl_vsl(0UL, vl) ;
    __vr vrsumEF_s1 = _vel_vbrdl_vsl(0UL, vl) ;

    __vr vrsum01_s2 = _vel_vbrdl_vsl(0UL, vl) ;
    __vr vrsum23_s2 = _vel_vbrdl_vsl(0UL, vl) ;
    __vr vrsum45_s2 = _vel_vbrdl_vsl(0UL, vl) ;
    __vr vrsum67_s2 = _vel_vbrdl_vsl(0UL, vl) ;
    __vr vrsum89_s2 = _vel_vbrdl_vsl(0UL, vl) ;
    __vr vrsumAB_s2 = _vel_vbrdl_vsl(0UL, vl) ;
    __vr vrsumCD_s2 = _vel_vbrdl_vsl(0UL, vl) ;
    __vr vrsumEF_s2 = _vel_vbrdl_vsl(0UL, vl) ;

    __vr vrsum01_s3 = _vel_vbrdl_vsl(0UL, vl) ;
    __vr vrsum23_s3 = _vel_vbrdl_vsl(0UL, vl) ;
    __vr vrsum45_s3 = _vel_vbrdl_vsl(0UL, vl) ;
    __vr vrsum67_s3 = _vel_vbrdl_vsl(0UL, vl) ;
    __vr vrsum89_s3 = _vel_vbrdl_vsl(0UL, vl) ;
    __vr vrsumAB_s3 = _vel_vbrdl_vsl(0UL, vl) ;
    __vr vrsumCD_s3 = _vel_vbrdl_vsl(0UL, vl) ;
    __vr vrsumEF_s3 = _vel_vbrdl_vsl(0UL, vl) ;

    __vr vrsum01_s4 = _vel_vbrdl_vsl(0UL, vl) ;
    __vr vrsum23_s4 = _vel_vbrdl_vsl(0UL, vl) ;
    __vr vrsum45_s4 = _vel_vbrdl_vsl(0UL, vl) ;
    __vr vrsum67_s4 = _vel_vbrdl_vsl(0UL, vl) ;
    __vr vrsum89_s4 = _vel_vbrdl_vsl(0UL, vl) ;
    __vr vrsumAB_s4 = _vel_vbrdl_vsl(0UL, vl) ;
    __vr vrsumCD_s4 = _vel_vbrdl_vsl(0UL, vl) ;
    __vr vrsumEF_s4 = _vel_vbrdl_vsl(0UL, vl) ;

    __vr vri_r0 = _vel_vaddsl_vsvl(2-0+h, vrh, vl) ;
    __vr vri_r1 = _vel_vaddsl_vsvl(2-1+h, vrh, vl) ;
    __vr vri_r2 = _vel_vaddsl_vsvl(2-2+h, vrh, vl) ;
    __vr vri_r3 = _vel_vaddsl_vsvl(2-3+h, vrh, vl) ;
    __vr vri_r4 = _vel_vaddsl_vsvl(2-4+h, vrh, vl) ;

    __vr vry_r0 = _vel_vdivsl_vvsl(vri_r0, 2, vl) ;
    __vr vry_r1 = _vel_vdivsl_vvsl(vri_r1, 2, vl) ;
    __vr vry_r2 = _vel_vdivsl_vvsl(vri_r2, 2, vl) ;
    __vr vry_r3 = _vel_vdivsl_vvsl(vri_r3, 2, vl) ;
    __vr vry_r4 = _vel_vdivsl_vvsl(vri_r4, 2, vl) ;

    __vm256 vmy0_r0 =  _vel_vfmkleq_mvl(_vel_vcmpsl_vvvl(vri_r0, _vel_vmulsl_vsvl(2, vry_r0, vl), vl), vl) ;
    __vm256 vmy1_r0 =  _vel_vfmklge_mvl(vry_r0, vl) ;
    __vm256 vmy2_r0 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutHeight,vry_r0, vl), vl) ;
    __vm256 vmy_r0 = _vel_andm_mmm(_vel_andm_mmm(vmy0_r0, vmy1_r0), vmy2_r0) ;

    __vm256 vmy0_r1 =  _vel_vfmkleq_mvl(_vel_vcmpsl_vvvl(vri_r1, _vel_vmulsl_vsvl(2, vry_r1, vl), vl), vl) ;
    __vm256 vmy1_r1 =  _vel_vfmklge_mvl(vry_r1, vl) ;
    __vm256 vmy2_r1 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutHeight,vry_r1, vl), vl) ;
    __vm256 vmy_r1 = _vel_andm_mmm(_vel_andm_mmm(vmy0_r1, vmy1_r1), vmy2_r1) ;

    __vm256 vmy0_r2 =  _vel_vfmkleq_mvl(_vel_vcmpsl_vvvl(vri_r2, _vel_vmulsl_vsvl(2, vry_r2, vl), vl), vl) ;
    __vm256 vmy1_r2 =  _vel_vfmklge_mvl(vry_r2, vl) ;
    __vm256 vmy2_r2 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutHeight,vry_r2, vl), vl) ;
    __vm256 vmy_r2 = _vel_andm_mmm(_vel_andm_mmm(vmy0_r2, vmy1_r2), vmy2_r2) ;

    __vm256 vmy0_r3 =  _vel_vfmkleq_mvl(_vel_vcmpsl_vvvl(vri_r3, _vel_vmulsl_vsvl(2, vry_r3, vl), vl), vl) ;
    __vm256 vmy1_r3 =  _vel_vfmklge_mvl(vry_r3, vl) ;
    __vm256 vmy2_r3 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutHeight,vry_r3, vl), vl) ;
    __vm256 vmy_r3 = _vel_andm_mmm(_vel_andm_mmm(vmy0_r3, vmy1_r3), vmy2_r3) ;

    __vm256 vmy0_r4 =  _vel_vfmkleq_mvl(_vel_vcmpsl_vvvl(vri_r4, _vel_vmulsl_vsvl(2, vry_r4, vl), vl), vl) ;
    __vm256 vmy1_r4 =  _vel_vfmklge_mvl(vry_r4, vl) ;
    __vm256 vmy2_r4 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutHeight,vry_r4, vl), vl) ;
    __vm256 vmy_r4 = _vel_andm_mmm(_vel_andm_mmm(vmy0_r4, vmy1_r4), vmy2_r4) ;

    __vr vrj_s0 = _vel_vaddsl_vsvl( 2, vrw, vl) ;
    __vr vrj_s1 = _vel_vaddsl_vsvl( 1, vrw, vl) ;
    __vr vrj_s2 = _vel_vaddsl_vsvl( 0, vrw, vl) ;
    __vr vrj_s3 = _vel_vaddsl_vsvl(-1, vrw, vl) ;
    __vr vrj_s4 = _vel_vaddsl_vsvl(-2, vrw, vl) ;

    __vr vrx_s0 = _vel_vdivsl_vvsl(vrj_s0, 2, vl) ;
    __vr vrx_s1 = _vel_vdivsl_vvsl(vrj_s1, 2, vl) ;
    __vr vrx_s2 = _vel_vdivsl_vvsl(vrj_s2, 2, vl) ;
    __vr vrx_s3 = _vel_vdivsl_vvsl(vrj_s3, 2, vl) ;
    __vr vrx_s4 = _vel_vdivsl_vvsl(vrj_s4, 2, vl) ;

    __vm256 vmx0_s0 =  _vel_vfmkleq_mvl(_vel_vcmpsl_vvvl(vrj_s0, _vel_vmulsl_vsvl(2, vrx_s0, vl), vl), vl) ;
    __vm256 vmx1_s0 =  _vel_vfmklge_mvl(vrx_s0, vl) ;
    __vm256 vmx2_s0 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutWidth,vrx_s0, vl), vl) ;
    __vm256 vmx_s0 = _vel_andm_mmm(_vel_andm_mmm(vmx0_s0, vmx1_s0), vmx2_s0) ;

    __vm256 vmx0_s1 =  _vel_vfmkleq_mvl(_vel_vcmpsl_vvvl(vrj_s1, _vel_vmulsl_vsvl(2, vrx_s1, vl), vl), vl) ;
    __vm256 vmx1_s1 =  _vel_vfmklge_mvl(vrx_s1, vl) ;
    __vm256 vmx2_s1 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutWidth,vrx_s1, vl), vl) ;
    __vm256 vmx_s1 = _vel_andm_mmm(_vel_andm_mmm(vmx0_s1, vmx1_s1), vmx2_s1) ;

    __vm256 vmx0_s2 =  _vel_vfmkleq_mvl(_vel_vcmpsl_vvvl(vrj_s2, _vel_vmulsl_vsvl(2, vrx_s2, vl), vl), vl) ;
    __vm256 vmx1_s2 =  _vel_vfmklge_mvl(vrx_s2, vl) ;
    __vm256 vmx2_s2 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutWidth,vrx_s2, vl), vl) ;
    __vm256 vmx_s2 = _vel_andm_mmm(_vel_andm_mmm(vmx0_s2, vmx1_s2), vmx2_s2) ;

    __vm256 vmx0_s3 =  _vel_vfmkleq_mvl(_vel_vcmpsl_vvvl(vrj_s3, _vel_vmulsl_vsvl(2, vrx_s3, vl), vl), vl) ;
    __vm256 vmx1_s3 =  _vel_vfmklge_mvl(vrx_s3, vl) ;
    __vm256 vmx2_s3 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutWidth,vrx_s3, vl), vl) ;
    __vm256 vmx_s3 = _vel_andm_mmm(_vel_andm_mmm(vmx0_s3, vmx1_s3), vmx2_s3) ;

    __vm256 vmx0_s4 =  _vel_vfmkleq_mvl(_vel_vcmpsl_vvvl(vrj_s4, _vel_vmulsl_vsvl(2, vrx_s4, vl), vl), vl) ;
    __vm256 vmx1_s4 =  _vel_vfmklge_mvl(vrx_s4, vl) ;
    __vm256 vmx2_s4 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutWidth,vrx_s4, vl), vl) ;
    __vm256 vmx_s4 = _vel_andm_mmm(_vel_andm_mmm(vmx0_s4, vmx1_s4), vmx2_s4) ;

    for (int64_t k=0; k<gOutChannelGroup; k++) {
      int64_t gOutIndex    = gOutGroupOffset + ((n * gOutChannel + k) * gOutHeight) * gOutWidth ;
      const float *pKerValue = pKernel + kernGroupOffset + (((k  ) * gInChannelGroup + c) * kernHeight) * kernWidth ;

#define VFADD_C16(VRGOUT, K, R, S)  {													\
	const uint64_t kerValue01 = _vel_pack_f32p(pKerValue + (((K)*gInChannelGroup + 0) * kernHeight +(R)) * kernWidth + (S),		\
						   pKerValue + (((K)*gInChannelGroup + 1) * kernHeight +(R)) * kernWidth + (S)) ;	\
	const uint64_t kerValue23 = _vel_pack_f32p(pKerValue + (((K)*gInChannelGroup + 2) * kernHeight +(R)) * kernWidth + (S),		\
						   pKerValue + (((K)*gInChannelGroup + 3) * kernHeight +(R)) * kernWidth + (S)) ;	\
	const uint64_t kerValue45 = _vel_pack_f32p(pKerValue + (((K)*gInChannelGroup + 4) * kernHeight +(R)) * kernWidth + (S),		\
						   pKerValue + (((K)*gInChannelGroup + 5) * kernHeight +(R)) * kernWidth + (S)) ;	\
	const uint64_t kerValue67 = _vel_pack_f32p(pKerValue + (((K)*gInChannelGroup + 6) * kernHeight +(R)) * kernWidth + (S),		\
						   pKerValue + (((K)*gInChannelGroup + 7) * kernHeight +(R)) * kernWidth + (S)) ;	\
	const uint64_t kerValue89 = _vel_pack_f32p(pKerValue + (((K)*gInChannelGroup + 8) * kernHeight +(R)) * kernWidth + (S),		\
						   pKerValue + (((K)*gInChannelGroup + 9) * kernHeight +(R)) * kernWidth + (S)) ;	\
	const uint64_t kerValueAB = _vel_pack_f32p(pKerValue + (((K)*gInChannelGroup +10) * kernHeight +(R)) * kernWidth + (S),		\
						   pKerValue + (((K)*gInChannelGroup +11) * kernHeight +(R)) * kernWidth + (S)) ;	\
	const uint64_t kerValueCD = _vel_pack_f32p(pKerValue + (((K)*gInChannelGroup +12) * kernHeight +(R)) * kernWidth + (S),		\
						   pKerValue + (((K)*gInChannelGroup +13) * kernHeight +(R)) * kernWidth + (S)) ;	\
	const uint64_t kerValueEF = _vel_pack_f32p(pKerValue + (((K)*gInChannelGroup +14) * kernHeight +(R)) * kernWidth + (S),		\
						   pKerValue + (((K)*gInChannelGroup +15) * kernHeight +(R)) * kernWidth + (S)) ;	\
	__vr vrgoutP = _vel_vshf_vvvsl(VRGOUT, VRGOUT, VE_VSHUFFLE_YUZU, vl) ;	\
	vrsum01_s##S = _vel_pvfmad_vvsvl(vrsum01_s##S, kerValue01, vrgoutP, vl) ;	\
	vrsum23_s##S = _vel_pvfmad_vvsvl(vrsum23_s##S, kerValue23, vrgoutP, vl) ;	\
	vrsum45_s##S = _vel_pvfmad_vvsvl(vrsum45_s##S, kerValue45, vrgoutP, vl) ;	\
	vrsum67_s##S = _vel_pvfmad_vvsvl(vrsum67_s##S, kerValue67, vrgoutP, vl) ;	\
	vrsum89_s##S = _vel_pvfmad_vvsvl(vrsum89_s##S, kerValue89, vrgoutP, vl) ;	\
	vrsumAB_s##S = _vel_pvfmad_vvsvl(vrsumAB_s##S, kerValueAB, vrgoutP, vl) ;	\
	vrsumCD_s##S = _vel_pvfmad_vvsvl(vrsumCD_s##S, kerValueCD, vrgoutP, vl) ;	\
	vrsumEF_s##S = _vel_pvfmad_vvsvl(vrsumEF_s##S, kerValueEF, vrgoutP, vl) ;	\
      }

      __vr vrgout_ptr_k0_r0s2 = _vel_vsfa_vvssl(_vel_vaddsl_vvvl(_vel_vmulsl_vsvl(gOutWidth, vry_r0, vl), vrx_s2, vl),
					 2,
					 (unsigned long)(pGOut+gOutIndex), vl) ;
      __vr vrgout_k0_r0s2 = _vel_vgtu_vvssml(vrgout_ptr_k0_r0s2, 0, 0, vmy_r0, vl) ;
      __vr vrgout_ptr_k0_r1s2 = _vel_vsfa_vvssl(_vel_vaddsl_vvvl(_vel_vmulsl_vsvl(gOutWidth, vry_r1, vl), vrx_s2, vl),
					 2,
					 (unsigned long)(pGOut+gOutIndex), vl) ;
      __vr vrgout_k0_r1s2 = _vel_vgtu_vvssml(vrgout_ptr_k0_r1s2, 0, 0, vmy_r1, vl) ;
      __vr vrgout_ptr_k0_r2s2 = _vel_vsfa_vvssl(_vel_vaddsl_vvvl(_vel_vmulsl_vsvl(gOutWidth, vry_r2, vl), vrx_s2, vl),
					 2,
					 (unsigned long)(pGOut+gOutIndex), vl) ;
      __vr vrgout_k0_r2s2 = _vel_vgtu_vvssml(vrgout_ptr_k0_r2s2, 0, 0, vmy_r2, vl) ;
      __vr vrgout_ptr_k0_r3s2 = _vel_vsfa_vvssl(_vel_vaddsl_vvvl(_vel_vmulsl_vsvl(gOutWidth, vry_r3, vl), vrx_s2, vl),
					 2,
					 (unsigned long)(pGOut+gOutIndex), vl) ;
      __vr vrgout_k0_r3s2 = _vel_vgtu_vvssml(vrgout_ptr_k0_r3s2, 0, 0, vmy_r3, vl) ;
      __vr vrgout_ptr_k0_r4s2 = _vel_vsfa_vvssl(_vel_vaddsl_vvvl(_vel_vmulsl_vsvl(gOutWidth, vry_r4, vl), vrx_s2, vl),
					 2,
					 (unsigned long)(pGOut+gOutIndex), vl) ;
      __vr vrgout_k0_r4s2 = _vel_vgtu_vvssml(vrgout_ptr_k0_r4s2, 0, 0, vmy_r4, vl) ;

      vrgout_k0_r0s2 = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrgout_k0_r0s2, vmy_r0, vl) ;
      VFADD_C16(vrgout_k0_r0s2, 0, 0, 0)
      VFADD_C16(vrgout_k0_r0s2, 0, 0, 1)
      VFADD_C16(vrgout_k0_r0s2, 0, 0, 2)
      VFADD_C16(vrgout_k0_r0s2, 0, 0, 3)
      VFADD_C16(vrgout_k0_r0s2, 0, 0, 4)

      vrgout_k0_r1s2 = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrgout_k0_r1s2, vmy_r1, vl) ;
      VFADD_C16(vrgout_k0_r1s2, 0, 1, 0)
      VFADD_C16(vrgout_k0_r1s2, 0, 1, 1)
      VFADD_C16(vrgout_k0_r1s2, 0, 1, 2)
      VFADD_C16(vrgout_k0_r1s2, 0, 1, 3)
      VFADD_C16(vrgout_k0_r1s2, 0, 1, 4)

      vrgout_k0_r2s2 = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrgout_k0_r2s2, vmy_r2, vl) ;
      VFADD_C16(vrgout_k0_r2s2, 0, 2, 0)
      VFADD_C16(vrgout_k0_r2s2, 0, 2, 1)
      VFADD_C16(vrgout_k0_r2s2, 0, 2, 2)
      VFADD_C16(vrgout_k0_r2s2, 0, 2, 3)
      VFADD_C16(vrgout_k0_r2s2, 0, 2, 4)

      vrgout_k0_r3s2 = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrgout_k0_r3s2, vmy_r3, vl) ;
      VFADD_C16(vrgout_k0_r3s2, 0, 3, 0)
      VFADD_C16(vrgout_k0_r3s2, 0, 3, 1)
      VFADD_C16(vrgout_k0_r3s2, 0, 3, 2)
      VFADD_C16(vrgout_k0_r3s2, 0, 3, 3)
      VFADD_C16(vrgout_k0_r3s2, 0, 3, 4)

      vrgout_k0_r4s2 = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrgout_k0_r4s2, vmy_r4, vl) ;
      VFADD_C16(vrgout_k0_r4s2, 0, 4, 0)
      VFADD_C16(vrgout_k0_r4s2, 0, 4, 1)
      VFADD_C16(vrgout_k0_r4s2, 0, 4, 2)
      VFADD_C16(vrgout_k0_r4s2, 0, 4, 3)
      VFADD_C16(vrgout_k0_r4s2, 0, 4, 4)

#undef VFADD_C16
    }

    vrsum01_s0 = _vel_vmrg_vvvml(_vel_vbrdl_vsl(0UL, vl), _vel_vmv_vsvl( 2, vrsum01_s0, vl), vmx_s0, vl) ;
    vrsum01_s1 = _vel_vmrg_vvvml(_vel_vbrdl_vsl(0UL, vl), _vel_vmv_vsvl( 1, vrsum01_s1, vl), vmx_s1, vl) ;
    vrsum01_s2 = _vel_vmrg_vvvml(_vel_vbrdl_vsl(0UL, vl),                 vrsum01_s2, vmx_s2, vl) ;
    vrsum01_s3 = _vel_vmrg_vvvml(_vel_vbrdl_vsl(0UL, vl), _vel_vmv_vsvl(-1, vrsum01_s3, vl), vmx_s3, vl) ;
    vrsum01_s4 = _vel_vmrg_vvvml(_vel_vbrdl_vsl(0UL, vl), _vel_vmv_vsvl(-2, vrsum01_s4, vl), vmx_s4, vl) ;
    __vr vrsum01 = _vel_pvfadd_vvvl(_vel_pvfadd_vvvl(vrsum01_s0, _vel_pvfadd_vvvl(vrsum01_s1, vrsum01_s2, vl), vl),
                                    _vel_pvfadd_vvvl(vrsum01_s3, vrsum01_s4, vl), vl) ;
    _vel_vstu_vssl(vrsum01, 4, pGIn+gInIndex, vl) ;
    _vel_vstl_vssl(vrsum01, 4, pGIn+gInIndex+  gInPixels, vl) ;

    vrsum23_s0 = _vel_vmrg_vvvml(_vel_vbrdl_vsl(0UL, vl), _vel_vmv_vsvl( 2, vrsum23_s0, vl), vmx_s0, vl) ;
    vrsum23_s1 = _vel_vmrg_vvvml(_vel_vbrdl_vsl(0UL, vl), _vel_vmv_vsvl( 1, vrsum23_s1, vl), vmx_s1, vl) ;
    vrsum23_s2 = _vel_vmrg_vvvml(_vel_vbrdl_vsl(0UL, vl),                 vrsum23_s2, vmx_s2, vl) ;
    vrsum23_s3 = _vel_vmrg_vvvml(_vel_vbrdl_vsl(0UL, vl), _vel_vmv_vsvl(-1, vrsum23_s3, vl), vmx_s3, vl) ;
    vrsum23_s4 = _vel_vmrg_vvvml(_vel_vbrdl_vsl(0UL, vl), _vel_vmv_vsvl(-2, vrsum23_s4, vl), vmx_s4, vl) ;
    __vr vrsum23 = _vel_pvfadd_vvvl(_vel_pvfadd_vvvl(vrsum23_s0, _vel_pvfadd_vvvl(vrsum23_s1, vrsum23_s2, vl), vl),
                                    _vel_pvfadd_vvvl(vrsum23_s3, vrsum23_s4, vl), vl) ;
    _vel_vstu_vssl(vrsum23, 4, pGIn+gInIndex+2*gInPixels, vl) ;
    _vel_vstl_vssl(vrsum23, 4, pGIn+gInIndex+3*gInPixels, vl) ;

    vrsum45_s0 = _vel_vmrg_vvvml(_vel_vbrdl_vsl(0UL, vl), _vel_vmv_vsvl( 2, vrsum45_s0, vl), vmx_s0, vl) ;
    vrsum45_s1 = _vel_vmrg_vvvml(_vel_vbrdl_vsl(0UL, vl), _vel_vmv_vsvl( 1, vrsum45_s1, vl), vmx_s1, vl) ;
    vrsum45_s2 = _vel_vmrg_vvvml(_vel_vbrdl_vsl(0UL, vl),                 vrsum45_s2, vmx_s2, vl) ;
    vrsum45_s3 = _vel_vmrg_vvvml(_vel_vbrdl_vsl(0UL, vl), _vel_vmv_vsvl(-1, vrsum45_s3, vl), vmx_s3, vl) ;
    vrsum45_s4 = _vel_vmrg_vvvml(_vel_vbrdl_vsl(0UL, vl), _vel_vmv_vsvl(-2, vrsum45_s4, vl), vmx_s4, vl) ;
    __vr vrsum45 = _vel_pvfadd_vvvl(_vel_pvfadd_vvvl(vrsum45_s0, _vel_pvfadd_vvvl(vrsum45_s1, vrsum45_s2, vl), vl),
                                    _vel_pvfadd_vvvl(vrsum45_s3, vrsum45_s4, vl), vl) ;
    _vel_vstu_vssl(vrsum45, 4, pGIn+gInIndex+4*gInPixels, vl) ;
    _vel_vstl_vssl(vrsum45, 4, pGIn+gInIndex+5*gInPixels, vl) ;

    vrsum67_s0 = _vel_vmrg_vvvml(_vel_vbrdl_vsl(0UL, vl), _vel_vmv_vsvl( 2, vrsum67_s0, vl), vmx_s0, vl) ;
    vrsum67_s1 = _vel_vmrg_vvvml(_vel_vbrdl_vsl(0UL, vl), _vel_vmv_vsvl( 1, vrsum67_s1, vl), vmx_s1, vl) ;
    vrsum67_s2 = _vel_vmrg_vvvml(_vel_vbrdl_vsl(0UL, vl),                 vrsum67_s2, vmx_s2, vl) ;
    vrsum67_s3 = _vel_vmrg_vvvml(_vel_vbrdl_vsl(0UL, vl), _vel_vmv_vsvl(-1, vrsum67_s3, vl), vmx_s3, vl) ;
    vrsum67_s4 = _vel_vmrg_vvvml(_vel_vbrdl_vsl(0UL, vl), _vel_vmv_vsvl(-2, vrsum67_s4, vl), vmx_s4, vl) ;
    __vr vrsum67 = _vel_pvfadd_vvvl(_vel_pvfadd_vvvl(vrsum67_s0, _vel_pvfadd_vvvl(vrsum67_s1, vrsum67_s2, vl), vl),
                                    _vel_pvfadd_vvvl(vrsum67_s3, vrsum67_s4, vl), vl) ;
    _vel_vstu_vssl(vrsum67, 4, pGIn+gInIndex+6*gInPixels, vl) ;
    _vel_vstl_vssl(vrsum67, 4, pGIn+gInIndex+7*gInPixels, vl) ;

    vrsum89_s0 = _vel_vmrg_vvvml(_vel_vbrdl_vsl(0UL, vl), _vel_vmv_vsvl( 2, vrsum89_s0, vl), vmx_s0, vl) ;
    vrsum89_s1 = _vel_vmrg_vvvml(_vel_vbrdl_vsl(0UL, vl), _vel_vmv_vsvl( 1, vrsum89_s1, vl), vmx_s1, vl) ;
    vrsum89_s2 = _vel_vmrg_vvvml(_vel_vbrdl_vsl(0UL, vl),                 vrsum89_s2, vmx_s2, vl) ;
    vrsum89_s3 = _vel_vmrg_vvvml(_vel_vbrdl_vsl(0UL, vl), _vel_vmv_vsvl(-1, vrsum89_s3, vl), vmx_s3, vl) ;
    vrsum89_s4 = _vel_vmrg_vvvml(_vel_vbrdl_vsl(0UL, vl), _vel_vmv_vsvl(-2, vrsum89_s4, vl), vmx_s4, vl) ;
    __vr vrsum89 = _vel_pvfadd_vvvl(_vel_pvfadd_vvvl(vrsum89_s0, _vel_pvfadd_vvvl(vrsum89_s1, vrsum89_s2, vl), vl),
                                    _vel_pvfadd_vvvl(vrsum89_s3, vrsum89_s4, vl), vl) ;
    _vel_vstu_vssl(vrsum89, 4, pGIn+gInIndex+8*gInPixels, vl) ;
    _vel_vstl_vssl(vrsum89, 4, pGIn+gInIndex+9*gInPixels, vl) ;

    vrsumAB_s0 = _vel_vmrg_vvvml(_vel_vbrdl_vsl(0UL, vl), _vel_vmv_vsvl( 2, vrsumAB_s0, vl), vmx_s0, vl) ;
    vrsumAB_s1 = _vel_vmrg_vvvml(_vel_vbrdl_vsl(0UL, vl), _vel_vmv_vsvl( 1, vrsumAB_s1, vl), vmx_s1, vl) ;
    vrsumAB_s2 = _vel_vmrg_vvvml(_vel_vbrdl_vsl(0UL, vl),                 vrsumAB_s2, vmx_s2, vl) ;
    vrsumAB_s3 = _vel_vmrg_vvvml(_vel_vbrdl_vsl(0UL, vl), _vel_vmv_vsvl(-1, vrsumAB_s3, vl), vmx_s3, vl) ;
    vrsumAB_s4 = _vel_vmrg_vvvml(_vel_vbrdl_vsl(0UL, vl), _vel_vmv_vsvl(-2, vrsumAB_s4, vl), vmx_s4, vl) ;
    __vr vrsumAB = _vel_pvfadd_vvvl(_vel_pvfadd_vvvl(vrsumAB_s0, _vel_pvfadd_vvvl(vrsumAB_s1, vrsumAB_s2, vl), vl),
                                    _vel_pvfadd_vvvl(vrsumAB_s3, vrsumAB_s4, vl), vl) ;
    _vel_vstu_vssl(vrsumAB, 4, pGIn+gInIndex+10*gInPixels, vl) ;
    _vel_vstl_vssl(vrsumAB, 4, pGIn+gInIndex+11*gInPixels, vl) ;

    vrsumCD_s0 = _vel_vmrg_vvvml(_vel_vbrdl_vsl(0UL, vl), _vel_vmv_vsvl( 2, vrsumCD_s0, vl), vmx_s0, vl) ;
    vrsumCD_s1 = _vel_vmrg_vvvml(_vel_vbrdl_vsl(0UL, vl), _vel_vmv_vsvl( 1, vrsumCD_s1, vl), vmx_s1, vl) ;
    vrsumCD_s2 = _vel_vmrg_vvvml(_vel_vbrdl_vsl(0UL, vl),                 vrsumCD_s2, vmx_s2, vl) ;
    vrsumCD_s3 = _vel_vmrg_vvvml(_vel_vbrdl_vsl(0UL, vl), _vel_vmv_vsvl(-1, vrsumCD_s3, vl), vmx_s3, vl) ;
    vrsumCD_s4 = _vel_vmrg_vvvml(_vel_vbrdl_vsl(0UL, vl), _vel_vmv_vsvl(-2, vrsumCD_s4, vl), vmx_s4, vl) ;
    __vr vrsumCD = _vel_pvfadd_vvvl(_vel_pvfadd_vvvl(vrsumCD_s0, _vel_pvfadd_vvvl(vrsumCD_s1, vrsumCD_s2, vl), vl),
                                    _vel_pvfadd_vvvl(vrsumCD_s3, vrsumCD_s4, vl), vl) ;
    _vel_vstu_vssl(vrsumCD, 4, pGIn+gInIndex+12*gInPixels, vl) ;
    _vel_vstl_vssl(vrsumCD, 4, pGIn+gInIndex+13*gInPixels, vl) ;

    vrsumEF_s0 = _vel_vmrg_vvvml(_vel_vbrdl_vsl(0UL, vl), _vel_vmv_vsvl( 2, vrsumEF_s0, vl), vmx_s0, vl) ;
    vrsumEF_s1 = _vel_vmrg_vvvml(_vel_vbrdl_vsl(0UL, vl), _vel_vmv_vsvl( 1, vrsumEF_s1, vl), vmx_s1, vl) ;
    vrsumEF_s2 = _vel_vmrg_vvvml(_vel_vbrdl_vsl(0UL, vl),                 vrsumEF_s2, vmx_s2, vl) ;
    vrsumEF_s3 = _vel_vmrg_vvvml(_vel_vbrdl_vsl(0UL, vl), _vel_vmv_vsvl(-1, vrsumEF_s3, vl), vmx_s3, vl) ;
    vrsumEF_s4 = _vel_vmrg_vvvml(_vel_vbrdl_vsl(0UL, vl), _vel_vmv_vsvl(-2, vrsumEF_s4, vl), vmx_s4, vl) ;
    __vr vrsumEF = _vel_pvfadd_vvvl(_vel_pvfadd_vvvl(vrsumEF_s0, _vel_pvfadd_vvvl(vrsumEF_s1, vrsumEF_s2, vl), vl),
                                    _vel_pvfadd_vvvl(vrsumEF_s3, vrsumEF_s4, vl), vl) ;
    _vel_vstu_vssl(vrsumEF, 4, pGIn+gInIndex+14*gInPixels, vl) ;
    _vel_vstl_vssl(vrsumEF, 4, pGIn+gInIndex+15*gInPixels, vl) ;
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
	     nH ) ;

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
	     nH ) ;

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
	     nH ) ;

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
	     nH ) ;

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
	     nH ) ;

	} // gInChannel
      } // group
    } // batch
  }


  return VEDNN_SUCCESS;
}
