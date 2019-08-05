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
    __vr vrsum = _ve_vbrdu_vs_f32(0.f) ;

    __vr vri_r0 = _ve_vaddsl_vsv(padHeight-0*dilationHeight+h, vrh) ;
    __vr vri_r1 = _ve_vaddsl_vsv(padHeight-1*dilationHeight+h, vrh) ;
    __vr vri_r2 = _ve_vaddsl_vsv(padHeight-2*dilationHeight+h, vrh) ;

    __vr vry_r0 = _ve_vdivsl_vvs(vri_r0, strideHeight) ;
    __vr vry_r1 = _ve_vdivsl_vvs(vri_r1, strideHeight) ;
    __vr vry_r2 = _ve_vdivsl_vvs(vri_r2, strideHeight) ;

    __vr vrj_s0 = _ve_vaddsl_vsv(padWidth-0*dilationWidth, vrw) ;
    __vr vrj_s1 = _ve_vaddsl_vsv(padWidth-1*dilationWidth, vrw) ;
    __vr vrj_s2 = _ve_vaddsl_vsv(padWidth-2*dilationWidth, vrw) ;

    __vr vrx_s0 = _ve_vdivsl_vvs(vrj_s0, strideWidth) ;
    __vr vrx_s1 = _ve_vdivsl_vvs(vrj_s1, strideWidth) ;
    __vr vrx_s2 = _ve_vdivsl_vvs(vrj_s2, strideWidth) ;

    __vm256 vmy0_r0 = _ve_vfmkl_mcv(VECC_IEQ, _ve_vcmpsl_vvv(vri_r0, _ve_vmulsl_vsv(strideHeight, vry_r0))) ;
    __vm256 vmy1_r0 = _ve_vfmkl_mcv(VECC_GE, vry_r0) ;
    __vm256 vmy2_r0 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(gOutHeight,vry_r0)) ;
    __vm256 vmy_r0 = _ve_andm_mmm(_ve_andm_mmm(vmy0_r0, vmy1_r0), vmy2_r0) ;

    __vm256 vmy0_r1 = _ve_vfmkl_mcv(VECC_IEQ, _ve_vcmpsl_vvv(vri_r1, _ve_vmulsl_vsv(strideHeight, vry_r1))) ;
    __vm256 vmy1_r1 = _ve_vfmkl_mcv(VECC_GE, vry_r1) ;
    __vm256 vmy2_r1 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(gOutHeight,vry_r1)) ;
    __vm256 vmy_r1 = _ve_andm_mmm(_ve_andm_mmm(vmy0_r1, vmy1_r1), vmy2_r1) ;

    __vm256 vmy0_r2 = _ve_vfmkl_mcv(VECC_IEQ, _ve_vcmpsl_vvv(vri_r2, _ve_vmulsl_vsv(strideHeight, vry_r2))) ;
    __vm256 vmy1_r2 = _ve_vfmkl_mcv(VECC_GE, vry_r2) ;
    __vm256 vmy2_r2 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(gOutHeight,vry_r2)) ;
    __vm256 vmy_r2 = _ve_andm_mmm(_ve_andm_mmm(vmy0_r2, vmy1_r2), vmy2_r2) ;

    __vm256 vmx0_s0 = _ve_vfmkl_mcv(VECC_IEQ, _ve_vcmpsl_vvv(vrj_s0, _ve_vmulsl_vsv(strideWidth, vrx_s0))) ;
    __vm256 vmx1_s0 = _ve_vfmkl_mcv(VECC_GE, vrx_s0) ;
    __vm256 vmx2_s0 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(gOutWidth,vrx_s0)) ;
    __vm256 vmx_s0 = _ve_andm_mmm(_ve_andm_mmm(vmx0_s0, vmx1_s0), vmx2_s0) ;

    __vm256 vmx0_s1 = _ve_vfmkl_mcv(VECC_IEQ, _ve_vcmpsl_vvv(vrj_s1, _ve_vmulsl_vsv(strideWidth, vrx_s1))) ;
    __vm256 vmx1_s1 = _ve_vfmkl_mcv(VECC_GE, vrx_s1) ;
    __vm256 vmx2_s1 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(gOutWidth,vrx_s1)) ;
    __vm256 vmx_s1 = _ve_andm_mmm(_ve_andm_mmm(vmx0_s1, vmx1_s1), vmx2_s1) ;

    __vm256 vmx0_s2 = _ve_vfmkl_mcv(VECC_IEQ, _ve_vcmpsl_vvv(vrj_s2, _ve_vmulsl_vsv(strideWidth, vrx_s2))) ;
    __vm256 vmx1_s2 = _ve_vfmkl_mcv(VECC_GE, vrx_s2) ;
    __vm256 vmx2_s2 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(gOutWidth,vrx_s2)) ;
    __vm256 vmx_s2 = _ve_andm_mmm(_ve_andm_mmm(vmx0_s2, vmx1_s2), vmx2_s2) ;

    __vm256 vmall_r0s0 = _ve_andm_mmm(vmy_r0,vmx_s0) ;
    __vm256 vmall_r0s1 = _ve_andm_mmm(vmy_r0,vmx_s1) ;
    __vm256 vmall_r0s2 = _ve_andm_mmm(vmy_r0,vmx_s2) ;

    __vm256 vmall_r1s0 = _ve_andm_mmm(vmy_r1,vmx_s0) ;
    __vm256 vmall_r1s1 = _ve_andm_mmm(vmy_r1,vmx_s1) ;
    __vm256 vmall_r1s2 = _ve_andm_mmm(vmy_r1,vmx_s2) ;

    __vm256 vmall_r2s0 = _ve_andm_mmm(vmy_r2,vmx_s0) ;
    __vm256 vmall_r2s1 = _ve_andm_mmm(vmy_r2,vmx_s1) ;
    __vm256 vmall_r2s2 = _ve_andm_mmm(vmy_r2,vmx_s2) ;

    int64_t k=0;
    if( (gOutChannelGroup & 0x01 ) == 1 ) {
      int64_t gOutIndex    = gOutGroupOffset + ((n * gOutChannel + k) * gOutHeight) * gOutWidth ;
      const float *pKerValue = pKernel + kernGroupOffset + (((k  ) * gInChannelGroup + c) * kernHeight ) * kernWidth ;

#define VFADD_C1(VRGOUT, VM, K, R, S)  {									\
	const float kerValue = pKerValue[(((K)*gInChannelGroup + 0) * kernHeight +(R)) * kernWidth + (S) ] ;	\
	VRGOUT = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), VRGOUT, VM) ;						\
	vrsum = _ve_vfmads_vvsv(vrsum, kerValue, VRGOUT) ;							\
      }

      __vr vrgout_ptr_k0_r0_s0 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry_r0), vrx_s0),
					 2,
					 (unsigned long)(pGOut+gOutIndex)) ;
      __vr vrgout_k0_r0_s0 = _ve_vgtu_vvm(vrgout_ptr_k0_r0_s0, vmall_r0s0) ;
      __vr vrgout_ptr_k0_r0_s1 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry_r0), vrx_s1),
					 2,
					 (unsigned long)(pGOut+gOutIndex)) ;
      __vr vrgout_k0_r0_s1 = _ve_vgtu_vvm(vrgout_ptr_k0_r0_s1, vmall_r0s1) ;
      __vr vrgout_ptr_k0_r0_s2 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry_r0), vrx_s2),
					 2,
					 (unsigned long)(pGOut+gOutIndex)) ;
      __vr vrgout_k0_r0_s2 = _ve_vgtu_vvm(vrgout_ptr_k0_r0_s2, vmall_r0s2) ;

      VFADD_C1(vrgout_k0_r0_s0, vmall_r0s0, 0, 0, 0)
      VFADD_C1(vrgout_k0_r0_s1, vmall_r0s1, 0, 0, 1)
      VFADD_C1(vrgout_k0_r0_s2, vmall_r0s2, 0, 0, 2)

      __vr vrgout_ptr_k0_r1_s0 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry_r1), vrx_s0),
					 2,
					 (unsigned long)(pGOut+gOutIndex)) ;
      __vr vrgout_k0_r1_s0 = _ve_vgtu_vvm(vrgout_ptr_k0_r1_s0, vmall_r1s0) ;
      __vr vrgout_ptr_k0_r1_s1 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry_r1), vrx_s1),
					 2,
					 (unsigned long)(pGOut+gOutIndex)) ;
      __vr vrgout_k0_r1_s1 = _ve_vgtu_vvm(vrgout_ptr_k0_r1_s1, vmall_r1s1) ;
      __vr vrgout_ptr_k0_r1_s2 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry_r1), vrx_s2),
					 2,
					 (unsigned long)(pGOut+gOutIndex)) ;
      __vr vrgout_k0_r1_s2 = _ve_vgtu_vvm(vrgout_ptr_k0_r1_s2, vmall_r1s2) ;

      VFADD_C1(vrgout_k0_r1_s0, vmall_r1s0, 0, 1, 0)
      VFADD_C1(vrgout_k0_r1_s1, vmall_r1s1, 0, 1, 1)
      VFADD_C1(vrgout_k0_r1_s2, vmall_r1s2, 0, 1, 2)

      __vr vrgout_ptr_k0_r2_s0 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry_r2), vrx_s0),
					 2,
					 (unsigned long)(pGOut+gOutIndex)) ;
      __vr vrgout_k0_r2_s0 = _ve_vgtu_vvm(vrgout_ptr_k0_r2_s0, vmall_r2s0) ;
      __vr vrgout_ptr_k0_r2_s1 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry_r2), vrx_s1),
					 2,
					 (unsigned long)(pGOut+gOutIndex)) ;
      __vr vrgout_k0_r2_s1 = _ve_vgtu_vvm(vrgout_ptr_k0_r2_s1, vmall_r2s1) ;
      __vr vrgout_ptr_k0_r2_s2 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry_r2), vrx_s2),
					 2,
					 (unsigned long)(pGOut+gOutIndex)) ;
      __vr vrgout_k0_r2_s2 = _ve_vgtu_vvm(vrgout_ptr_k0_r2_s2, vmall_r2s2) ;

      VFADD_C1(vrgout_k0_r2_s0, vmall_r2s0, 0, 2, 0)
      VFADD_C1(vrgout_k0_r2_s1, vmall_r2s1, 0, 2, 1)
      VFADD_C1(vrgout_k0_r2_s2, vmall_r2s2, 0, 2, 2)

      k+=1 ;
    }
    if( ((gOutChannelGroup >> 1) & 0x01 ) == 1 ) {
      int64_t gOutIndex    = gOutGroupOffset + ((n * gOutChannel + k) * gOutHeight) * gOutWidth ;
      const float *pKerValue = pKernel + kernGroupOffset + (((k  ) * gInChannelGroup + c) * kernHeight) * kernWidth ;

      __vr vrgout_ptr_k0_r0_s0 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry_r0), vrx_s0),
					 2,
					 (unsigned long)(pGOut+gOutIndex)) ;
      __vr vrgout_k0_r0_s0 = _ve_vgtu_vvm(vrgout_ptr_k0_r0_s0, vmall_r0s0) ;
      __vr vrgout_ptr_k1_r0_s0 = _ve_vaddsl_vsv(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_r0_s0) ;
      __vr vrgout_k1_r0_s0 = _ve_vgtu_vvm(vrgout_ptr_k1_r0_s0, vmall_r0s0) ;

      VFADD_C1(vrgout_k0_r0_s0, vmall_r0s0, 0, 0, 0)
      VFADD_C1(vrgout_k1_r0_s0, vmall_r0s0, 1, 0, 0)

      __vr vrgout_ptr_k0_r0_s1 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry_r0), vrx_s1),
					 2,
					 (unsigned long)(pGOut+gOutIndex)) ;
      __vr vrgout_k0_r0_s1 = _ve_vgtu_vvm(vrgout_ptr_k0_r0_s1, vmall_r0s1) ;
      __vr vrgout_ptr_k1_r0_s1 = _ve_vaddsl_vsv(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_r0_s1) ;
      __vr vrgout_k1_r0_s1 = _ve_vgtu_vvm(vrgout_ptr_k1_r0_s1, vmall_r0s1) ;

      VFADD_C1(vrgout_k0_r0_s1, vmall_r0s1, 0, 0, 1)
      VFADD_C1(vrgout_k1_r0_s1, vmall_r0s1, 1, 0, 1)

      __vr vrgout_ptr_k0_r0_s2 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry_r0), vrx_s2),
					 2,
					 (unsigned long)(pGOut+gOutIndex)) ;
      __vr vrgout_k0_r0_s2 = _ve_vgtu_vvm(vrgout_ptr_k0_r0_s2, vmall_r0s2) ;
      __vr vrgout_ptr_k1_r0_s2 = _ve_vaddsl_vsv(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_r0_s2) ;
      __vr vrgout_k1_r0_s2 = _ve_vgtu_vvm(vrgout_ptr_k1_r0_s2, vmall_r0s2) ;

      VFADD_C1(vrgout_k0_r0_s2, vmall_r0s2, 0, 0, 2)
      VFADD_C1(vrgout_k1_r0_s2, vmall_r0s2, 1, 0, 2)

      __vr vrgout_ptr_k0_r1_s0 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry_r1), vrx_s0),
					 2,
					 (unsigned long)(pGOut+gOutIndex)) ;
      __vr vrgout_k0_r1_s0 = _ve_vgtu_vvm(vrgout_ptr_k0_r1_s0, vmall_r1s0) ;
      __vr vrgout_ptr_k1_r1_s0 = _ve_vaddsl_vsv(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_r1_s0) ;
      __vr vrgout_k1_r1_s0 = _ve_vgtu_vvm(vrgout_ptr_k1_r1_s0, vmall_r1s0) ;

      VFADD_C1(vrgout_k0_r1_s0, vmall_r1s0, 0, 1, 0)
      VFADD_C1(vrgout_k1_r1_s0, vmall_r1s0, 1, 1, 0)

      __vr vrgout_ptr_k0_r1_s1 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry_r1), vrx_s1),
					 2,
					 (unsigned long)(pGOut+gOutIndex)) ;
      __vr vrgout_k0_r1_s1 = _ve_vgtu_vvm(vrgout_ptr_k0_r1_s1, vmall_r1s1) ;
      __vr vrgout_ptr_k1_r1_s1 = _ve_vaddsl_vsv(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_r1_s1) ;
      __vr vrgout_k1_r1_s1 = _ve_vgtu_vvm(vrgout_ptr_k1_r1_s1, vmall_r1s1) ;

      VFADD_C1(vrgout_k0_r1_s1, vmall_r1s1, 0, 1, 1)
      VFADD_C1(vrgout_k1_r1_s1, vmall_r1s1, 1, 1, 1)

      __vr vrgout_ptr_k0_r1_s2 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry_r1), vrx_s2),
					 2,
					 (unsigned long)(pGOut+gOutIndex)) ;
      __vr vrgout_k0_r1_s2 = _ve_vgtu_vvm(vrgout_ptr_k0_r1_s2, vmall_r1s2) ;
      __vr vrgout_ptr_k1_r1_s2 = _ve_vaddsl_vsv(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_r1_s2) ;
      __vr vrgout_k1_r1_s2 = _ve_vgtu_vvm(vrgout_ptr_k1_r1_s2, vmall_r1s2) ;

      VFADD_C1(vrgout_k0_r1_s2, vmall_r1s2, 0, 1, 2)
      VFADD_C1(vrgout_k1_r1_s2, vmall_r1s2, 1, 1, 2)

      __vr vrgout_ptr_k0_r2_s0 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry_r2), vrx_s0),
					 2,
					 (unsigned long)(pGOut+gOutIndex)) ;
      __vr vrgout_k0_r2_s0 = _ve_vgtu_vvm(vrgout_ptr_k0_r2_s0, vmall_r2s0) ;
      __vr vrgout_ptr_k1_r2_s0 = _ve_vaddsl_vsv(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_r2_s0) ;
      __vr vrgout_k1_r2_s0 = _ve_vgtu_vvm(vrgout_ptr_k1_r2_s0, vmall_r2s0) ;

      VFADD_C1(vrgout_k0_r2_s0, vmall_r2s0, 0, 2, 0)
      VFADD_C1(vrgout_k1_r2_s0, vmall_r2s0, 1, 2, 0)

      __vr vrgout_ptr_k0_r2_s1 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry_r2), vrx_s1),
					 2,
					 (unsigned long)(pGOut+gOutIndex)) ;
      __vr vrgout_k0_r2_s1 = _ve_vgtu_vvm(vrgout_ptr_k0_r2_s1, vmall_r2s1) ;
      __vr vrgout_ptr_k1_r2_s1 = _ve_vaddsl_vsv(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_r2_s1) ;
      __vr vrgout_k1_r2_s1 = _ve_vgtu_vvm(vrgout_ptr_k1_r2_s1, vmall_r2s1) ;

      VFADD_C1(vrgout_k0_r2_s1, vmall_r2s1, 0, 2, 1)
      VFADD_C1(vrgout_k1_r2_s1, vmall_r2s1, 1, 2, 1)

      __vr vrgout_ptr_k0_r2_s2 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry_r2), vrx_s2),
					 2,
					 (unsigned long)(pGOut+gOutIndex)) ;
      __vr vrgout_k0_r2_s2 = _ve_vgtu_vvm(vrgout_ptr_k0_r2_s2, vmall_r2s2) ;
      __vr vrgout_ptr_k1_r2_s2 = _ve_vaddsl_vsv(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_r2_s2) ;
      __vr vrgout_k1_r2_s2 = _ve_vgtu_vvm(vrgout_ptr_k1_r2_s2, vmall_r2s2) ;

      VFADD_C1(vrgout_k0_r2_s2, vmall_r2s2, 0, 2, 2)
      VFADD_C1(vrgout_k1_r2_s2, vmall_r2s2, 1, 2, 2)

      k+=2 ;
    }
    if( ((gOutChannelGroup >> 2) & 0x01 ) == 1 ) {
      int64_t gOutIndex    = gOutGroupOffset + ((n * gOutChannel + k) * gOutHeight) * gOutWidth ;
      const float *pKerValue = pKernel + kernGroupOffset + (((k  ) * gInChannelGroup + c) * kernHeight) * kernWidth ;

      __vr vrgout_ptr_k0_r0_s0 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry_r0), vrx_s0),
					 2,
					 (unsigned long)(pGOut+gOutIndex)) ;
      __vr vrgout_k0_r0_s0 = _ve_vgtu_vvm(vrgout_ptr_k0_r0_s0, vmall_r0s0) ;
      __vr vrgout_ptr_k1_r0_s0 = _ve_vaddsl_vsv(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_r0_s0) ;
      __vr vrgout_k1_r0_s0 = _ve_vgtu_vvm(vrgout_ptr_k1_r0_s0, vmall_r0s0) ;
      __vr vrgout_ptr_r0_k2_s0 = _ve_vaddsl_vsv(4*2*gOutHeight*gOutWidth, vrgout_ptr_k0_r0_s0) ;
      __vr vrgout_k2_r0_s0 = _ve_vgtu_vvm(vrgout_ptr_r0_k2_s0, vmall_r0s0) ;
      __vr vrgout_ptr_k3_r0_s0 = _ve_vaddsl_vsv(4*3*gOutHeight*gOutWidth, vrgout_ptr_k0_r0_s0) ;
      __vr vrgout_k3_r0_s0 = _ve_vgtu_vvm(vrgout_ptr_k3_r0_s0, vmall_r0s0) ;

      VFADD_C1(vrgout_k0_r0_s0, vmall_r0s0, 0, 0, 0)
      VFADD_C1(vrgout_k1_r0_s0, vmall_r0s0, 1, 0, 0)
      VFADD_C1(vrgout_k2_r0_s0, vmall_r0s0, 2, 0, 0)
      VFADD_C1(vrgout_k3_r0_s0, vmall_r0s0, 3, 0, 0)

      __vr vrgout_ptr_k0_r0_s1 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry_r0), vrx_s1),
					 2,
					 (unsigned long)(pGOut+gOutIndex)) ;
      __vr vrgout_k0_r0_s1 = _ve_vgtu_vvm(vrgout_ptr_k0_r0_s1, vmall_r0s1) ;
      __vr vrgout_ptr_k1_r0_s1 = _ve_vaddsl_vsv(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_r0_s1) ;
      __vr vrgout_k1_r0_s1 = _ve_vgtu_vvm(vrgout_ptr_k1_r0_s1, vmall_r0s1) ;
      __vr vrgout_ptr_k2_r0_s1 = _ve_vaddsl_vsv(4*2*gOutHeight*gOutWidth, vrgout_ptr_k0_r0_s1) ;
      __vr vrgout_k2_r0_s1 = _ve_vgtu_vvm(vrgout_ptr_k2_r0_s1, vmall_r0s1) ;
      __vr vrgout_ptr_k3_r0_s1 = _ve_vaddsl_vsv(4*3*gOutHeight*gOutWidth, vrgout_ptr_k0_r0_s1) ;
      __vr vrgout_k3_r0_s1 = _ve_vgtu_vvm(vrgout_ptr_k3_r0_s1, vmall_r0s1) ;

      VFADD_C1(vrgout_k0_r0_s1, vmall_r0s1, 0, 0, 1)
      VFADD_C1(vrgout_k1_r0_s1, vmall_r0s1, 1, 0, 1)
      VFADD_C1(vrgout_k2_r0_s1, vmall_r0s1, 2, 0, 1)
      VFADD_C1(vrgout_k3_r0_s1, vmall_r0s1, 3, 0, 1)

      __vr vrgout_ptr_k0_r0_s2 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry_r0), vrx_s2),
					 2,
					 (unsigned long)(pGOut+gOutIndex)) ;
      __vr vrgout_k0_r0_s2 = _ve_vgtu_vvm(vrgout_ptr_k0_r0_s2, vmall_r0s2) ;
      __vr vrgout_ptr_k1_r0_s2 = _ve_vaddsl_vsv(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_r0_s2) ;
      __vr vrgout_k1_r0_s2 = _ve_vgtu_vvm(vrgout_ptr_k1_r0_s2, vmall_r0s2) ;
      __vr vrgout_ptr_k2_r0_s2 = _ve_vaddsl_vsv(4*2*gOutHeight*gOutWidth, vrgout_ptr_k0_r0_s2) ;
      __vr vrgout_k2_r0_s2 = _ve_vgtu_vvm(vrgout_ptr_k2_r0_s2, vmall_r0s2) ;
      __vr vrgout_ptr_k3_r0_s2 = _ve_vaddsl_vsv(4*3*gOutHeight*gOutWidth, vrgout_ptr_k0_r0_s2) ;
      __vr vrgout_k3_r0_s2 = _ve_vgtu_vvm(vrgout_ptr_k3_r0_s2, vmall_r0s2) ;

      VFADD_C1(vrgout_k0_r0_s2, vmall_r0s2, 0, 0, 2)
      VFADD_C1(vrgout_k1_r0_s2, vmall_r0s2, 1, 0, 2)
      VFADD_C1(vrgout_k2_r0_s2, vmall_r0s2, 2, 0, 2)
      VFADD_C1(vrgout_k3_r0_s2, vmall_r0s2, 3, 0, 2)

      __vr vrgout_ptr_k0_r1_s0 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry_r1), vrx_s0),
					 2,
					 (unsigned long)(pGOut+gOutIndex)) ;
      __vr vrgout_k0_r1_s0 = _ve_vgtu_vvm(vrgout_ptr_k0_r1_s0, vmall_r1s0) ;
      __vr vrgout_ptr_k1_r1_s0 = _ve_vaddsl_vsv(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_r1_s0) ;
      __vr vrgout_k1_r1_s0 = _ve_vgtu_vvm(vrgout_ptr_k1_r1_s0, vmall_r1s0) ;
      __vr vrgout_ptr_k2_r1_s0 = _ve_vaddsl_vsv(4*2*gOutHeight*gOutWidth, vrgout_ptr_k0_r1_s0) ;
      __vr vrgout_k2_r1_s0 = _ve_vgtu_vvm(vrgout_ptr_k2_r1_s0, vmall_r1s0) ;
      __vr vrgout_ptr_k3_r1_s0 = _ve_vaddsl_vsv(4*3*gOutHeight*gOutWidth, vrgout_ptr_k0_r1_s0) ;
      __vr vrgout_k3_r1_s0 = _ve_vgtu_vvm(vrgout_ptr_k3_r1_s0, vmall_r1s0) ;

      VFADD_C1(vrgout_k0_r1_s0, vmall_r1s0, 0, 1, 0)
      VFADD_C1(vrgout_k1_r1_s0, vmall_r1s0, 1, 1, 0)
      VFADD_C1(vrgout_k2_r1_s0, vmall_r1s0, 2, 1, 0)
      VFADD_C1(vrgout_k3_r1_s0, vmall_r1s0, 3, 1, 0)

      __vr vrgout_ptr_k0_r1_s1 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry_r1), vrx_s1),
					 2,
					 (unsigned long)(pGOut+gOutIndex)) ;
      __vr vrgout_k0_r1_s1 = _ve_vgtu_vvm(vrgout_ptr_k0_r1_s1, vmall_r1s1) ;
      __vr vrgout_ptr_k1_r1_s1 = _ve_vaddsl_vsv(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_r1_s1) ;
      __vr vrgout_k1_r1_s1 = _ve_vgtu_vvm(vrgout_ptr_k1_r1_s1, vmall_r1s1) ;
      __vr vrgout_ptr_k2_r1_s1 = _ve_vaddsl_vsv(4*2*gOutHeight*gOutWidth, vrgout_ptr_k0_r1_s1) ;
      __vr vrgout_k2_r1_s1 = _ve_vgtu_vvm(vrgout_ptr_k2_r1_s1, vmall_r1s1) ;
      __vr vrgout_ptr_k3_r1_s1 = _ve_vaddsl_vsv(4*3*gOutHeight*gOutWidth, vrgout_ptr_k0_r1_s1) ;
      __vr vrgout_k3_r1_s1 = _ve_vgtu_vvm(vrgout_ptr_k3_r1_s1, vmall_r1s1) ;

      VFADD_C1(vrgout_k0_r1_s1, vmall_r1s1, 0, 1, 1)
      VFADD_C1(vrgout_k1_r1_s1, vmall_r1s1, 1, 1, 1)
      VFADD_C1(vrgout_k2_r1_s1, vmall_r1s1, 2, 1, 1)
      VFADD_C1(vrgout_k3_r1_s1, vmall_r1s1, 3, 1, 1)

      __vr vrgout_ptr_k0_r1_s2 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry_r1), vrx_s2),
					 2,
					 (unsigned long)(pGOut+gOutIndex)) ;
      __vr vrgout_k0_r1_s2 = _ve_vgtu_vvm(vrgout_ptr_k0_r1_s2, vmall_r1s2) ;
      __vr vrgout_ptr_k1_r1_s2 = _ve_vaddsl_vsv(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_r1_s2) ;
      __vr vrgout_k1_r1_s2 = _ve_vgtu_vvm(vrgout_ptr_k1_r1_s2, vmall_r1s2) ;
      __vr vrgout_ptr_k2_r1_s2 = _ve_vaddsl_vsv(4*2*gOutHeight*gOutWidth, vrgout_ptr_k0_r1_s2) ;
      __vr vrgout_k2_r1_s2 = _ve_vgtu_vvm(vrgout_ptr_k2_r1_s2, vmall_r1s2) ;
      __vr vrgout_ptr_k3_r1_s2 = _ve_vaddsl_vsv(4*3*gOutHeight*gOutWidth, vrgout_ptr_k0_r1_s2) ;
      __vr vrgout_k3_r1_s2 = _ve_vgtu_vvm(vrgout_ptr_k3_r1_s2, vmall_r1s2) ;

      VFADD_C1(vrgout_k0_r1_s2, vmall_r1s2, 0, 1, 2)
      VFADD_C1(vrgout_k1_r1_s2, vmall_r1s2, 1, 1, 2)
      VFADD_C1(vrgout_k2_r1_s2, vmall_r1s2, 2, 1, 2)
      VFADD_C1(vrgout_k3_r1_s2, vmall_r1s2, 3, 1, 2)

      __vr vrgout_ptr_k0_r2_s0 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry_r2), vrx_s0),
					 2,
					 (unsigned long)(pGOut+gOutIndex)) ;
      __vr vrgout_k0_r2_s0 = _ve_vgtu_vvm(vrgout_ptr_k0_r2_s0, vmall_r2s0) ;
      __vr vrgout_ptr_k1_r2_s0 = _ve_vaddsl_vsv(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_r2_s0) ;
      __vr vrgout_k1_r2_s0 = _ve_vgtu_vvm(vrgout_ptr_k1_r2_s0, vmall_r2s0) ;
      __vr vrgout_ptr_k2_r2_s0 = _ve_vaddsl_vsv(4*2*gOutHeight*gOutWidth, vrgout_ptr_k0_r2_s0) ;
      __vr vrgout_k2_r2_s0 = _ve_vgtu_vvm(vrgout_ptr_k2_r2_s0, vmall_r2s0) ;
      __vr vrgout_ptr_k3_r2_s0 = _ve_vaddsl_vsv(4*3*gOutHeight*gOutWidth, vrgout_ptr_k0_r2_s0) ;
      __vr vrgout_k3_r2_s0 = _ve_vgtu_vvm(vrgout_ptr_k3_r2_s0, vmall_r2s0) ;

      VFADD_C1(vrgout_k0_r2_s0, vmall_r2s0, 0, 2, 0)
      VFADD_C1(vrgout_k1_r2_s0, vmall_r2s0, 1, 2, 0)
      VFADD_C1(vrgout_k2_r2_s0, vmall_r2s0, 2, 2, 0)
      VFADD_C1(vrgout_k3_r2_s0, vmall_r2s0, 3, 2, 0)

      __vr vrgout_ptr_k0_r2_s1 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry_r2), vrx_s1),
					 2,
					 (unsigned long)(pGOut+gOutIndex)) ;
      __vr vrgout_k0_r2_s1 = _ve_vgtu_vvm(vrgout_ptr_k0_r2_s1, vmall_r2s1) ;
      __vr vrgout_ptr_k1_r2_s1 = _ve_vaddsl_vsv(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_r2_s1) ;
      __vr vrgout_k1_r2_s1 = _ve_vgtu_vvm(vrgout_ptr_k1_r2_s1, vmall_r2s1) ;
      __vr vrgout_ptr_k2_r2_s1 = _ve_vaddsl_vsv(4*2*gOutHeight*gOutWidth, vrgout_ptr_k0_r2_s1) ;
      __vr vrgout_k2_r2_s1 = _ve_vgtu_vvm(vrgout_ptr_k2_r2_s1, vmall_r2s1) ;
      __vr vrgout_ptr_k3_r2_s1 = _ve_vaddsl_vsv(4*3*gOutHeight*gOutWidth, vrgout_ptr_k0_r2_s1) ;
      __vr vrgout_k3_r2_s1 = _ve_vgtu_vvm(vrgout_ptr_k3_r2_s1, vmall_r2s1) ;

      VFADD_C1(vrgout_k0_r2_s1, vmall_r2s1, 0, 2, 1)
      VFADD_C1(vrgout_k1_r2_s1, vmall_r2s1, 1, 2, 1)
      VFADD_C1(vrgout_k2_r2_s1, vmall_r2s1, 2, 2, 1)
      VFADD_C1(vrgout_k3_r2_s1, vmall_r2s1, 3, 2, 1)

      __vr vrgout_ptr_k0_r2_s2 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry_r2), vrx_s2),
					 2,
					 (unsigned long)(pGOut+gOutIndex)) ;
      __vr vrgout_k0_r2_s2 = _ve_vgtu_vvm(vrgout_ptr_k0_r2_s2, vmall_r2s2) ;
      __vr vrgout_ptr_k1_r2_s2 = _ve_vaddsl_vsv(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_r2_s2) ;
      __vr vrgout_k1_r2_s2 = _ve_vgtu_vvm(vrgout_ptr_k1_r2_s2, vmall_r2s2) ;
      __vr vrgout_ptr_k2_r2_s2 = _ve_vaddsl_vsv(4*2*gOutHeight*gOutWidth, vrgout_ptr_k0_r2_s2) ;
      __vr vrgout_k2_r2_s2 = _ve_vgtu_vvm(vrgout_ptr_k2_r2_s2, vmall_r2s2) ;
      __vr vrgout_ptr_k3_r2_s2 = _ve_vaddsl_vsv(4*3*gOutHeight*gOutWidth, vrgout_ptr_k0_r2_s2) ;
      __vr vrgout_k3_r2_s2 = _ve_vgtu_vvm(vrgout_ptr_k3_r2_s2, vmall_r2s2) ;

      VFADD_C1(vrgout_k0_r2_s2, vmall_r2s2, 0, 2, 2)
      VFADD_C1(vrgout_k1_r2_s2, vmall_r2s2, 1, 2, 2)
      VFADD_C1(vrgout_k2_r2_s2, vmall_r2s2, 2, 2, 2)
      VFADD_C1(vrgout_k3_r2_s2, vmall_r2s2, 3, 2, 2)

      k+=4 ;
    }
    for (; k<gOutChannelGroup; k+=8) {
      int64_t gOutIndex    = gOutGroupOffset + ((n * gOutChannel + k) * gOutHeight) * gOutWidth ;
      const float *pKerValue = pKernel + kernGroupOffset + (((k  ) * gInChannelGroup + c) * kernHeight) * kernWidth ;

      __vr vrgout_ptr_k0_r0_s0 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry_r0), vrx_s0),
					 2,
					 (unsigned long)(pGOut+gOutIndex)) ;
      __vr vrgout_k0_r0_s0 = _ve_vgtu_vvm(vrgout_ptr_k0_r0_s0, vmall_r0s0) ;
      __vr vrgout_ptr_k1_r0_s0 = _ve_vaddsl_vsv(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_r0_s0) ;
      __vr vrgout_k1_r0_s0 = _ve_vgtu_vvm(vrgout_ptr_k1_r0_s0, vmall_r0s0) ;
      __vr vrgout_ptr_r0_k2_s0 = _ve_vaddsl_vsv(4*2*gOutHeight*gOutWidth, vrgout_ptr_k0_r0_s0) ;
      __vr vrgout_k2_r0_s0 = _ve_vgtu_vvm(vrgout_ptr_r0_k2_s0, vmall_r0s0) ;
      __vr vrgout_ptr_k3_r0_s0 = _ve_vaddsl_vsv(4*3*gOutHeight*gOutWidth, vrgout_ptr_k0_r0_s0) ;
      __vr vrgout_k3_r0_s0 = _ve_vgtu_vvm(vrgout_ptr_k3_r0_s0, vmall_r0s0) ;
      __vr vrgout_ptr_k4_r0_s0 = _ve_vaddsl_vsv(4*4*gOutHeight*gOutWidth, vrgout_ptr_k0_r0_s0) ;
      __vr vrgout_k4_r0_s0 = _ve_vgtu_vvm(vrgout_ptr_k4_r0_s0, vmall_r0s0) ;
      __vr vrgout_ptr_k5_r0_s0 = _ve_vaddsl_vsv(4*5*gOutHeight*gOutWidth, vrgout_ptr_k0_r0_s0) ;
      __vr vrgout_k5_r0_s0 = _ve_vgtu_vvm(vrgout_ptr_k5_r0_s0, vmall_r0s0) ;
      __vr vrgout_ptr_k6_r0_s0 = _ve_vaddsl_vsv(4*6*gOutHeight*gOutWidth, vrgout_ptr_k0_r0_s0) ;
      __vr vrgout_k6_r0_s0 = _ve_vgtu_vvm(vrgout_ptr_k6_r0_s0, vmall_r0s0) ;
      __vr vrgout_ptr_k7_r0_s0 = _ve_vaddsl_vsv(4*7*gOutHeight*gOutWidth, vrgout_ptr_k0_r0_s0) ;
      __vr vrgout_k7_r0_s0 = _ve_vgtu_vvm(vrgout_ptr_k7_r0_s0, vmall_r0s0) ;

      VFADD_C1(vrgout_k0_r0_s0, vmall_r0s0, 0, 0, 0)
      VFADD_C1(vrgout_k1_r0_s0, vmall_r0s0, 1, 0, 0)
      VFADD_C1(vrgout_k2_r0_s0, vmall_r0s0, 2, 0, 0)
      VFADD_C1(vrgout_k3_r0_s0, vmall_r0s0, 3, 0, 0)
      VFADD_C1(vrgout_k4_r0_s0, vmall_r0s0, 4, 0, 0)
      VFADD_C1(vrgout_k5_r0_s0, vmall_r0s0, 5, 0, 0)
      VFADD_C1(vrgout_k6_r0_s0, vmall_r0s0, 6, 0, 0)
      VFADD_C1(vrgout_k7_r0_s0, vmall_r0s0, 7, 0, 0)

      __vr vrgout_ptr_k0_r0_s1 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry_r0), vrx_s1),
					 2,
					 (unsigned long)(pGOut+gOutIndex)) ;
      __vr vrgout_k0_r0_s1 = _ve_vgtu_vvm(vrgout_ptr_k0_r0_s1, vmall_r0s1) ;
      __vr vrgout_ptr_k1_r0_s1 = _ve_vaddsl_vsv(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_r0_s1) ;
      __vr vrgout_k1_r0_s1 = _ve_vgtu_vvm(vrgout_ptr_k1_r0_s1, vmall_r0s1) ;
      __vr vrgout_ptr_k2_r0_s1 = _ve_vaddsl_vsv(4*2*gOutHeight*gOutWidth, vrgout_ptr_k0_r0_s1) ;
      __vr vrgout_k2_r0_s1 = _ve_vgtu_vvm(vrgout_ptr_k2_r0_s1, vmall_r0s1) ;
      __vr vrgout_ptr_k3_r0_s1 = _ve_vaddsl_vsv(4*3*gOutHeight*gOutWidth, vrgout_ptr_k0_r0_s1) ;
      __vr vrgout_k3_r0_s1 = _ve_vgtu_vvm(vrgout_ptr_k3_r0_s1, vmall_r0s1) ;
      __vr vrgout_ptr_k4_r0_s1 = _ve_vaddsl_vsv(4*4*gOutHeight*gOutWidth, vrgout_ptr_k0_r0_s1) ;
      __vr vrgout_k4_r0_s1 = _ve_vgtu_vvm(vrgout_ptr_k4_r0_s1, vmall_r0s1) ;
      __vr vrgout_ptr_k5_r0_s1 = _ve_vaddsl_vsv(4*5*gOutHeight*gOutWidth, vrgout_ptr_k0_r0_s1) ;
      __vr vrgout_k5_r0_s1 = _ve_vgtu_vvm(vrgout_ptr_k5_r0_s1, vmall_r0s1) ;
      __vr vrgout_ptr_k6_r0_s1 = _ve_vaddsl_vsv(4*6*gOutHeight*gOutWidth, vrgout_ptr_k0_r0_s1) ;
      __vr vrgout_k6_r0_s1 = _ve_vgtu_vvm(vrgout_ptr_k6_r0_s1, vmall_r0s1) ;
      __vr vrgout_ptr_k7_r0_s1 = _ve_vaddsl_vsv(4*7*gOutHeight*gOutWidth, vrgout_ptr_k0_r0_s1) ;
      __vr vrgout_k7_r0_s1 = _ve_vgtu_vvm(vrgout_ptr_k7_r0_s1, vmall_r0s1) ;

      VFADD_C1(vrgout_k0_r0_s1, vmall_r0s1, 0, 0, 1)
      VFADD_C1(vrgout_k1_r0_s1, vmall_r0s1, 1, 0, 1)
      VFADD_C1(vrgout_k2_r0_s1, vmall_r0s1, 2, 0, 1)
      VFADD_C1(vrgout_k3_r0_s1, vmall_r0s1, 3, 0, 1)
      VFADD_C1(vrgout_k4_r0_s1, vmall_r0s1, 4, 0, 1)
      VFADD_C1(vrgout_k5_r0_s1, vmall_r0s1, 5, 0, 1)
      VFADD_C1(vrgout_k6_r0_s1, vmall_r0s1, 6, 0, 1)
      VFADD_C1(vrgout_k7_r0_s1, vmall_r0s1, 7, 0, 1)

      __vr vrgout_ptr_k0_r0_s2 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry_r0), vrx_s2),
					 2,
					 (unsigned long)(pGOut+gOutIndex)) ;
      __vr vrgout_k0_r0_s2 = _ve_vgtu_vvm(vrgout_ptr_k0_r0_s2, vmall_r0s2) ;
      __vr vrgout_ptr_k1_r0_s2 = _ve_vaddsl_vsv(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_r0_s2) ;
      __vr vrgout_k1_r0_s2 = _ve_vgtu_vvm(vrgout_ptr_k1_r0_s2, vmall_r0s2) ;
      __vr vrgout_ptr_k2_r0_s2 = _ve_vaddsl_vsv(4*2*gOutHeight*gOutWidth, vrgout_ptr_k0_r0_s2) ;
      __vr vrgout_k2_r0_s2 = _ve_vgtu_vvm(vrgout_ptr_k2_r0_s2, vmall_r0s2) ;
      __vr vrgout_ptr_k3_r0_s2 = _ve_vaddsl_vsv(4*3*gOutHeight*gOutWidth, vrgout_ptr_k0_r0_s2) ;
      __vr vrgout_k3_r0_s2 = _ve_vgtu_vvm(vrgout_ptr_k3_r0_s2, vmall_r0s2) ;
      __vr vrgout_ptr_k4_r0_s2 = _ve_vaddsl_vsv(4*4*gOutHeight*gOutWidth, vrgout_ptr_k0_r0_s2) ;
      __vr vrgout_k4_r0_s2 = _ve_vgtu_vvm(vrgout_ptr_k4_r0_s2, vmall_r0s2) ;
      __vr vrgout_ptr_k5_r0_s2 = _ve_vaddsl_vsv(4*5*gOutHeight*gOutWidth, vrgout_ptr_k0_r0_s2) ;
      __vr vrgout_k5_r0_s2 = _ve_vgtu_vvm(vrgout_ptr_k5_r0_s2, vmall_r0s2) ;
      __vr vrgout_ptr_k6_r0_s2 = _ve_vaddsl_vsv(4*6*gOutHeight*gOutWidth, vrgout_ptr_k0_r0_s2) ;
      __vr vrgout_k6_r0_s2 = _ve_vgtu_vvm(vrgout_ptr_k6_r0_s2, vmall_r0s2) ;
      __vr vrgout_ptr_k7_r0_s2 = _ve_vaddsl_vsv(4*7*gOutHeight*gOutWidth, vrgout_ptr_k0_r0_s2) ;
      __vr vrgout_k7_r0_s2 = _ve_vgtu_vvm(vrgout_ptr_k7_r0_s2, vmall_r0s2) ;

      VFADD_C1(vrgout_k0_r0_s2, vmall_r0s2, 0, 0, 2)
      VFADD_C1(vrgout_k1_r0_s2, vmall_r0s2, 1, 0, 2)
      VFADD_C1(vrgout_k2_r0_s2, vmall_r0s2, 2, 0, 2)
      VFADD_C1(vrgout_k3_r0_s2, vmall_r0s2, 3, 0, 2)
      VFADD_C1(vrgout_k4_r0_s2, vmall_r0s2, 4, 0, 2)
      VFADD_C1(vrgout_k5_r0_s2, vmall_r0s2, 5, 0, 2)
      VFADD_C1(vrgout_k6_r0_s2, vmall_r0s2, 6, 0, 2)
      VFADD_C1(vrgout_k7_r0_s2, vmall_r0s2, 7, 0, 2)

      __vr vrgout_ptr_k0_r1_s0 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry_r1), vrx_s0),
					 2,
					 (unsigned long)(pGOut+gOutIndex)) ;
      __vr vrgout_k0_r1_s0 = _ve_vgtu_vvm(vrgout_ptr_k0_r1_s0, vmall_r1s0) ;
      __vr vrgout_ptr_k1_r1_s0 = _ve_vaddsl_vsv(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_r1_s0) ;
      __vr vrgout_k1_r1_s0 = _ve_vgtu_vvm(vrgout_ptr_k1_r1_s0, vmall_r1s0) ;
      __vr vrgout_ptr_k2_r1_s0 = _ve_vaddsl_vsv(4*2*gOutHeight*gOutWidth, vrgout_ptr_k0_r1_s0) ;
      __vr vrgout_k2_r1_s0 = _ve_vgtu_vvm(vrgout_ptr_k2_r1_s0, vmall_r1s0) ;
      __vr vrgout_ptr_k3_r1_s0 = _ve_vaddsl_vsv(4*3*gOutHeight*gOutWidth, vrgout_ptr_k0_r1_s0) ;
      __vr vrgout_k3_r1_s0 = _ve_vgtu_vvm(vrgout_ptr_k3_r1_s0, vmall_r1s0) ;
      __vr vrgout_ptr_k4_r1_s0 = _ve_vaddsl_vsv(4*4*gOutHeight*gOutWidth, vrgout_ptr_k0_r1_s0) ;
      __vr vrgout_k4_r1_s0 = _ve_vgtu_vvm(vrgout_ptr_k4_r1_s0, vmall_r1s0) ;
      __vr vrgout_ptr_k5_r1_s0 = _ve_vaddsl_vsv(4*5*gOutHeight*gOutWidth, vrgout_ptr_k0_r1_s0) ;
      __vr vrgout_k5_r1_s0 = _ve_vgtu_vvm(vrgout_ptr_k5_r1_s0, vmall_r1s0) ;
      __vr vrgout_ptr_k6_r1_s0 = _ve_vaddsl_vsv(4*6*gOutHeight*gOutWidth, vrgout_ptr_k0_r1_s0) ;
      __vr vrgout_k6_r1_s0 = _ve_vgtu_vvm(vrgout_ptr_k6_r1_s0, vmall_r1s0) ;
      __vr vrgout_ptr_k7_r1_s0 = _ve_vaddsl_vsv(4*7*gOutHeight*gOutWidth, vrgout_ptr_k0_r1_s0) ;
      __vr vrgout_k7_r1_s0 = _ve_vgtu_vvm(vrgout_ptr_k7_r1_s0, vmall_r1s0) ;

      VFADD_C1(vrgout_k0_r1_s0, vmall_r1s0, 0, 1, 0)
      VFADD_C1(vrgout_k1_r1_s0, vmall_r1s0, 1, 1, 0)
      VFADD_C1(vrgout_k2_r1_s0, vmall_r1s0, 2, 1, 0)
      VFADD_C1(vrgout_k3_r1_s0, vmall_r1s0, 3, 1, 0)
      VFADD_C1(vrgout_k4_r1_s0, vmall_r1s0, 4, 1, 0)
      VFADD_C1(vrgout_k5_r1_s0, vmall_r1s0, 5, 1, 0)
      VFADD_C1(vrgout_k6_r1_s0, vmall_r1s0, 6, 1, 0)
      VFADD_C1(vrgout_k7_r1_s0, vmall_r1s0, 7, 1, 0)

      __vr vrgout_ptr_k0_r1_s1 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry_r1), vrx_s1),
					 2,
					 (unsigned long)(pGOut+gOutIndex)) ;
      __vr vrgout_k0_r1_s1 = _ve_vgtu_vvm(vrgout_ptr_k0_r1_s1, vmall_r1s1) ;
      __vr vrgout_ptr_k1_r1_s1 = _ve_vaddsl_vsv(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_r1_s1) ;
      __vr vrgout_k1_r1_s1 = _ve_vgtu_vvm(vrgout_ptr_k1_r1_s1, vmall_r1s1) ;
      __vr vrgout_ptr_k2_r1_s1 = _ve_vaddsl_vsv(4*2*gOutHeight*gOutWidth, vrgout_ptr_k0_r1_s1) ;
      __vr vrgout_k2_r1_s1 = _ve_vgtu_vvm(vrgout_ptr_k2_r1_s1, vmall_r1s1) ;
      __vr vrgout_ptr_k3_r1_s1 = _ve_vaddsl_vsv(4*3*gOutHeight*gOutWidth, vrgout_ptr_k0_r1_s1) ;
      __vr vrgout_k3_r1_s1 = _ve_vgtu_vvm(vrgout_ptr_k3_r1_s1, vmall_r1s1) ;
      __vr vrgout_ptr_k4_r1_s1 = _ve_vaddsl_vsv(4*4*gOutHeight*gOutWidth, vrgout_ptr_k0_r1_s1) ;
      __vr vrgout_k4_r1_s1 = _ve_vgtu_vvm(vrgout_ptr_k4_r1_s1, vmall_r1s1) ;
      __vr vrgout_ptr_k5_r1_s1 = _ve_vaddsl_vsv(4*5*gOutHeight*gOutWidth, vrgout_ptr_k0_r1_s1) ;
      __vr vrgout_k5_r1_s1 = _ve_vgtu_vvm(vrgout_ptr_k5_r1_s1, vmall_r1s1) ;
      __vr vrgout_ptr_k6_r1_s1 = _ve_vaddsl_vsv(4*6*gOutHeight*gOutWidth, vrgout_ptr_k0_r1_s1) ;
      __vr vrgout_k6_r1_s1 = _ve_vgtu_vvm(vrgout_ptr_k6_r1_s1, vmall_r1s1) ;
      __vr vrgout_ptr_k7_r1_s1 = _ve_vaddsl_vsv(4*7*gOutHeight*gOutWidth, vrgout_ptr_k0_r1_s1) ;
      __vr vrgout_k7_r1_s1 = _ve_vgtu_vvm(vrgout_ptr_k7_r1_s1, vmall_r1s1) ;

      VFADD_C1(vrgout_k0_r1_s1, vmall_r1s1, 0, 1, 1)
      VFADD_C1(vrgout_k1_r1_s1, vmall_r1s1, 1, 1, 1)
      VFADD_C1(vrgout_k2_r1_s1, vmall_r1s1, 2, 1, 1)
      VFADD_C1(vrgout_k3_r1_s1, vmall_r1s1, 3, 1, 1)
      VFADD_C1(vrgout_k4_r1_s1, vmall_r1s1, 4, 1, 1)
      VFADD_C1(vrgout_k5_r1_s1, vmall_r1s1, 5, 1, 1)
      VFADD_C1(vrgout_k6_r1_s1, vmall_r1s1, 6, 1, 1)
      VFADD_C1(vrgout_k7_r1_s1, vmall_r1s1, 7, 1, 1)

      __vr vrgout_ptr_k0_r1_s2 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry_r1), vrx_s2),
					 2,
					 (unsigned long)(pGOut+gOutIndex)) ;
      __vr vrgout_k0_r1_s2 = _ve_vgtu_vvm(vrgout_ptr_k0_r1_s2, vmall_r1s2) ;
      __vr vrgout_ptr_k1_r1_s2 = _ve_vaddsl_vsv(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_r1_s2) ;
      __vr vrgout_k1_r1_s2 = _ve_vgtu_vvm(vrgout_ptr_k1_r1_s2, vmall_r1s2) ;
      __vr vrgout_ptr_k2_r1_s2 = _ve_vaddsl_vsv(4*2*gOutHeight*gOutWidth, vrgout_ptr_k0_r1_s2) ;
      __vr vrgout_k2_r1_s2 = _ve_vgtu_vvm(vrgout_ptr_k2_r1_s2, vmall_r1s2) ;
      __vr vrgout_ptr_k3_r1_s2 = _ve_vaddsl_vsv(4*3*gOutHeight*gOutWidth, vrgout_ptr_k0_r1_s2) ;
      __vr vrgout_k3_r1_s2 = _ve_vgtu_vvm(vrgout_ptr_k3_r1_s2, vmall_r1s2) ;
      __vr vrgout_ptr_k4_r1_s2 = _ve_vaddsl_vsv(4*4*gOutHeight*gOutWidth, vrgout_ptr_k0_r1_s2) ;
      __vr vrgout_k4_r1_s2 = _ve_vgtu_vvm(vrgout_ptr_k4_r1_s2, vmall_r1s2) ;
      __vr vrgout_ptr_k5_r1_s2 = _ve_vaddsl_vsv(4*5*gOutHeight*gOutWidth, vrgout_ptr_k0_r1_s2) ;
      __vr vrgout_k5_r1_s2 = _ve_vgtu_vvm(vrgout_ptr_k5_r1_s2, vmall_r1s2) ;
      __vr vrgout_ptr_k6_r1_s2 = _ve_vaddsl_vsv(4*6*gOutHeight*gOutWidth, vrgout_ptr_k0_r1_s2) ;
      __vr vrgout_k6_r1_s2 = _ve_vgtu_vvm(vrgout_ptr_k6_r1_s2, vmall_r1s2) ;
      __vr vrgout_ptr_k7_r1_s2 = _ve_vaddsl_vsv(4*7*gOutHeight*gOutWidth, vrgout_ptr_k0_r1_s2) ;
      __vr vrgout_k7_r1_s2 = _ve_vgtu_vvm(vrgout_ptr_k7_r1_s2, vmall_r1s2) ;

      VFADD_C1(vrgout_k0_r1_s2, vmall_r1s2, 0, 1, 2)
      VFADD_C1(vrgout_k1_r1_s2, vmall_r1s2, 1, 1, 2)
      VFADD_C1(vrgout_k2_r1_s2, vmall_r1s2, 2, 1, 2)
      VFADD_C1(vrgout_k3_r1_s2, vmall_r1s2, 3, 1, 2)
      VFADD_C1(vrgout_k4_r1_s2, vmall_r1s2, 4, 1, 2)
      VFADD_C1(vrgout_k5_r1_s2, vmall_r1s2, 5, 1, 2)
      VFADD_C1(vrgout_k6_r1_s2, vmall_r1s2, 6, 1, 2)
      VFADD_C1(vrgout_k7_r1_s2, vmall_r1s2, 7, 1, 2)

      __vr vrgout_ptr_k0_r2_s0 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry_r2), vrx_s0),
					 2,
					 (unsigned long)(pGOut+gOutIndex)) ;
      __vr vrgout_k0_r2_s0 = _ve_vgtu_vvm(vrgout_ptr_k0_r2_s0, vmall_r2s0) ;
      __vr vrgout_ptr_k1_r2_s0 = _ve_vaddsl_vsv(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_r2_s0) ;
      __vr vrgout_k1_r2_s0 = _ve_vgtu_vvm(vrgout_ptr_k1_r2_s0, vmall_r2s0) ;
      __vr vrgout_ptr_k2_r2_s0 = _ve_vaddsl_vsv(4*2*gOutHeight*gOutWidth, vrgout_ptr_k0_r2_s0) ;
      __vr vrgout_k2_r2_s0 = _ve_vgtu_vvm(vrgout_ptr_k2_r2_s0, vmall_r2s0) ;
      __vr vrgout_ptr_k3_r2_s0 = _ve_vaddsl_vsv(4*3*gOutHeight*gOutWidth, vrgout_ptr_k0_r2_s0) ;
      __vr vrgout_k3_r2_s0 = _ve_vgtu_vvm(vrgout_ptr_k3_r2_s0, vmall_r2s0) ;
      __vr vrgout_ptr_k4_r2_s0 = _ve_vaddsl_vsv(4*4*gOutHeight*gOutWidth, vrgout_ptr_k0_r2_s0) ;
      __vr vrgout_k4_r2_s0 = _ve_vgtu_vvm(vrgout_ptr_k4_r2_s0, vmall_r2s0) ;
      __vr vrgout_ptr_k5_r2_s0 = _ve_vaddsl_vsv(4*5*gOutHeight*gOutWidth, vrgout_ptr_k0_r2_s0) ;
      __vr vrgout_k5_r2_s0 = _ve_vgtu_vvm(vrgout_ptr_k5_r2_s0, vmall_r2s0) ;
      __vr vrgout_ptr_k6_r2_s0 = _ve_vaddsl_vsv(4*6*gOutHeight*gOutWidth, vrgout_ptr_k0_r2_s0) ;
      __vr vrgout_k6_r2_s0 = _ve_vgtu_vvm(vrgout_ptr_k6_r2_s0, vmall_r2s0) ;
      __vr vrgout_ptr_k7_r2_s0 = _ve_vaddsl_vsv(4*7*gOutHeight*gOutWidth, vrgout_ptr_k0_r2_s0) ;
      __vr vrgout_k7_r2_s0 = _ve_vgtu_vvm(vrgout_ptr_k7_r2_s0, vmall_r2s0) ;

      VFADD_C1(vrgout_k0_r2_s0, vmall_r2s0, 0, 2, 0)
      VFADD_C1(vrgout_k1_r2_s0, vmall_r2s0, 1, 2, 0)
      VFADD_C1(vrgout_k2_r2_s0, vmall_r2s0, 2, 2, 0)
      VFADD_C1(vrgout_k3_r2_s0, vmall_r2s0, 3, 2, 0)
      VFADD_C1(vrgout_k4_r2_s0, vmall_r2s0, 4, 2, 0)
      VFADD_C1(vrgout_k5_r2_s0, vmall_r2s0, 5, 2, 0)
      VFADD_C1(vrgout_k6_r2_s0, vmall_r2s0, 6, 2, 0)
      VFADD_C1(vrgout_k7_r2_s0, vmall_r2s0, 7, 2, 0)

      __vr vrgout_ptr_k0_r2_s1 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry_r2), vrx_s1),
					 2,
					 (unsigned long)(pGOut+gOutIndex)) ;
      __vr vrgout_k0_r2_s1 = _ve_vgtu_vvm(vrgout_ptr_k0_r2_s1, vmall_r2s1) ;
      __vr vrgout_ptr_k1_r2_s1 = _ve_vaddsl_vsv(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_r2_s1) ;
      __vr vrgout_k1_r2_s1 = _ve_vgtu_vvm(vrgout_ptr_k1_r2_s1, vmall_r2s1) ;
      __vr vrgout_ptr_k2_r2_s1 = _ve_vaddsl_vsv(4*2*gOutHeight*gOutWidth, vrgout_ptr_k0_r2_s1) ;
      __vr vrgout_k2_r2_s1 = _ve_vgtu_vvm(vrgout_ptr_k2_r2_s1, vmall_r2s1) ;
      __vr vrgout_ptr_k3_r2_s1 = _ve_vaddsl_vsv(4*3*gOutHeight*gOutWidth, vrgout_ptr_k0_r2_s1) ;
      __vr vrgout_k3_r2_s1 = _ve_vgtu_vvm(vrgout_ptr_k3_r2_s1, vmall_r2s1) ;
      __vr vrgout_ptr_k4_r2_s1 = _ve_vaddsl_vsv(4*4*gOutHeight*gOutWidth, vrgout_ptr_k0_r2_s1) ;
      __vr vrgout_k4_r2_s1 = _ve_vgtu_vvm(vrgout_ptr_k4_r2_s1, vmall_r2s1) ;
      __vr vrgout_ptr_k5_r2_s1 = _ve_vaddsl_vsv(4*5*gOutHeight*gOutWidth, vrgout_ptr_k0_r2_s1) ;
      __vr vrgout_k5_r2_s1 = _ve_vgtu_vvm(vrgout_ptr_k5_r2_s1, vmall_r2s1) ;
      __vr vrgout_ptr_k6_r2_s1 = _ve_vaddsl_vsv(4*6*gOutHeight*gOutWidth, vrgout_ptr_k0_r2_s1) ;
      __vr vrgout_k6_r2_s1 = _ve_vgtu_vvm(vrgout_ptr_k6_r2_s1, vmall_r2s1) ;
      __vr vrgout_ptr_k7_r2_s1 = _ve_vaddsl_vsv(4*7*gOutHeight*gOutWidth, vrgout_ptr_k0_r2_s1) ;
      __vr vrgout_k7_r2_s1 = _ve_vgtu_vvm(vrgout_ptr_k7_r2_s1, vmall_r2s1) ;

      VFADD_C1(vrgout_k0_r2_s1, vmall_r2s1, 0, 2, 1)
      VFADD_C1(vrgout_k1_r2_s1, vmall_r2s1, 1, 2, 1)
      VFADD_C1(vrgout_k2_r2_s1, vmall_r2s1, 2, 2, 1)
      VFADD_C1(vrgout_k3_r2_s1, vmall_r2s1, 3, 2, 1)
      VFADD_C1(vrgout_k4_r2_s1, vmall_r2s1, 4, 2, 1)
      VFADD_C1(vrgout_k5_r2_s1, vmall_r2s1, 5, 2, 1)
      VFADD_C1(vrgout_k6_r2_s1, vmall_r2s1, 6, 2, 1)
      VFADD_C1(vrgout_k7_r2_s1, vmall_r2s1, 7, 2, 1)

      __vr vrgout_ptr_k0_r2_s2 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry_r2), vrx_s2),
					 2,
					 (unsigned long)(pGOut+gOutIndex)) ;
      __vr vrgout_k0_r2_s2 = _ve_vgtu_vvm(vrgout_ptr_k0_r2_s2, vmall_r2s2) ;
      __vr vrgout_ptr_k1_r2_s2 = _ve_vaddsl_vsv(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_r2_s2) ;
      __vr vrgout_k1_r2_s2 = _ve_vgtu_vvm(vrgout_ptr_k1_r2_s2, vmall_r2s2) ;
      __vr vrgout_ptr_k2_r2_s2 = _ve_vaddsl_vsv(4*2*gOutHeight*gOutWidth, vrgout_ptr_k0_r2_s2) ;
      __vr vrgout_k2_r2_s2 = _ve_vgtu_vvm(vrgout_ptr_k2_r2_s2, vmall_r2s2) ;
      __vr vrgout_ptr_k3_r2_s2 = _ve_vaddsl_vsv(4*3*gOutHeight*gOutWidth, vrgout_ptr_k0_r2_s2) ;
      __vr vrgout_k3_r2_s2 = _ve_vgtu_vvm(vrgout_ptr_k3_r2_s2, vmall_r2s2) ;
      __vr vrgout_ptr_k4_r2_s2 = _ve_vaddsl_vsv(4*4*gOutHeight*gOutWidth, vrgout_ptr_k0_r2_s2) ;
      __vr vrgout_k4_r2_s2 = _ve_vgtu_vvm(vrgout_ptr_k4_r2_s2, vmall_r2s2) ;
      __vr vrgout_ptr_k5_r2_s2 = _ve_vaddsl_vsv(4*5*gOutHeight*gOutWidth, vrgout_ptr_k0_r2_s2) ;
      __vr vrgout_k5_r2_s2 = _ve_vgtu_vvm(vrgout_ptr_k5_r2_s2, vmall_r2s2) ;
      __vr vrgout_ptr_k6_r2_s2 = _ve_vaddsl_vsv(4*6*gOutHeight*gOutWidth, vrgout_ptr_k0_r2_s2) ;
      __vr vrgout_k6_r2_s2 = _ve_vgtu_vvm(vrgout_ptr_k6_r2_s2, vmall_r2s2) ;
      __vr vrgout_ptr_k7_r2_s2 = _ve_vaddsl_vsv(4*7*gOutHeight*gOutWidth, vrgout_ptr_k0_r2_s2) ;
      __vr vrgout_k7_r2_s2 = _ve_vgtu_vvm(vrgout_ptr_k7_r2_s2, vmall_r2s2) ;

      VFADD_C1(vrgout_k0_r2_s2, vmall_r2s2, 0, 2, 2)
      VFADD_C1(vrgout_k1_r2_s2, vmall_r2s2, 1, 2, 2)
      VFADD_C1(vrgout_k2_r2_s2, vmall_r2s2, 2, 2, 2)
      VFADD_C1(vrgout_k3_r2_s2, vmall_r2s2, 3, 2, 2)
      VFADD_C1(vrgout_k4_r2_s2, vmall_r2s2, 4, 2, 2)
      VFADD_C1(vrgout_k5_r2_s2, vmall_r2s2, 5, 2, 2)
      VFADD_C1(vrgout_k6_r2_s2, vmall_r2s2, 6, 2, 2)
      VFADD_C1(vrgout_k7_r2_s2, vmall_r2s2, 7, 2, 2)
#undef VFADD_C2

    } // gOutChannel

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
    __vr vrsum01 = _ve_pvbrd_vs_i64(0UL) ;

    __vr vri_r0 = _ve_vaddsl_vsv(padHeight-0*dilationHeight+h, vrh) ;
    __vr vri_r1 = _ve_vaddsl_vsv(padHeight-1*dilationHeight+h, vrh) ;
    __vr vri_r2 = _ve_vaddsl_vsv(padHeight-2*dilationHeight+h, vrh) ;

    __vr vry_r0 = _ve_vdivsl_vvs(vri_r0, strideHeight) ;
    __vr vry_r1 = _ve_vdivsl_vvs(vri_r1, strideHeight) ;
    __vr vry_r2 = _ve_vdivsl_vvs(vri_r2, strideHeight) ;

    __vr vrj_s0 = _ve_vaddsl_vsv(padWidth-0*dilationWidth, vrw) ;
    __vr vrj_s1 = _ve_vaddsl_vsv(padWidth-1*dilationWidth, vrw) ;
    __vr vrj_s2 = _ve_vaddsl_vsv(padWidth-2*dilationWidth, vrw) ;

    __vr vrx_s0 = _ve_vdivsl_vvs(vrj_s0, strideWidth) ;
    __vr vrx_s1 = _ve_vdivsl_vvs(vrj_s1, strideWidth) ;
    __vr vrx_s2 = _ve_vdivsl_vvs(vrj_s2, strideWidth) ;


    __vm256 vmy0_r0 = _ve_vfmkl_mcv(VECC_IEQ, _ve_vcmpsl_vvv(vri_r0, _ve_vmulsl_vsv(strideHeight, vry_r0))) ;
    __vm256 vmy1_r0 = _ve_vfmkl_mcv(VECC_GE, vry_r0) ;
    __vm256 vmy2_r0 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(gOutHeight,vry_r0)) ;
    __vm256 vmy_r0 = _ve_andm_mmm(_ve_andm_mmm(vmy0_r0, vmy1_r0), vmy2_r0) ;

    __vm256 vmy0_r1 = _ve_vfmkl_mcv(VECC_IEQ, _ve_vcmpsl_vvv(vri_r1, _ve_vmulsl_vsv(strideHeight, vry_r1))) ;
    __vm256 vmy1_r1 = _ve_vfmkl_mcv(VECC_GE, vry_r1) ;
    __vm256 vmy2_r1 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(gOutHeight,vry_r1)) ;
    __vm256 vmy_r1 = _ve_andm_mmm(_ve_andm_mmm(vmy0_r1, vmy1_r1), vmy2_r1) ;

    __vm256 vmy0_r2 = _ve_vfmkl_mcv(VECC_IEQ, _ve_vcmpsl_vvv(vri_r2, _ve_vmulsl_vsv(strideHeight, vry_r2))) ;
    __vm256 vmy1_r2 = _ve_vfmkl_mcv(VECC_GE, vry_r2) ;
    __vm256 vmy2_r2 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(gOutHeight,vry_r2)) ;
    __vm256 vmy_r2 = _ve_andm_mmm(_ve_andm_mmm(vmy0_r2, vmy1_r2), vmy2_r2) ;

    __vm256 vmx0_s0 = _ve_vfmkl_mcv(VECC_IEQ, _ve_vcmpsl_vvv(vrj_s0, _ve_vmulsl_vsv(strideWidth, vrx_s0))) ;
    __vm256 vmx1_s0 = _ve_vfmkl_mcv(VECC_GE, vrx_s0) ;
    __vm256 vmx2_s0 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(gOutWidth,vrx_s0)) ;
    __vm256 vmx_s0 = _ve_andm_mmm(_ve_andm_mmm(vmx0_s0, vmx1_s0), vmx2_s0) ;

    __vm256 vmx0_s1 = _ve_vfmkl_mcv(VECC_IEQ, _ve_vcmpsl_vvv(vrj_s1, _ve_vmulsl_vsv(strideWidth, vrx_s1))) ;
    __vm256 vmx1_s1 = _ve_vfmkl_mcv(VECC_GE, vrx_s1) ;
    __vm256 vmx2_s1 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(gOutWidth,vrx_s1)) ;
    __vm256 vmx_s1 = _ve_andm_mmm(_ve_andm_mmm(vmx0_s1, vmx1_s1), vmx2_s1) ;

    __vm256 vmx0_s2 = _ve_vfmkl_mcv(VECC_IEQ, _ve_vcmpsl_vvv(vrj_s2, _ve_vmulsl_vsv(strideWidth, vrx_s2))) ;
    __vm256 vmx1_s2 = _ve_vfmkl_mcv(VECC_GE, vrx_s2) ;
    __vm256 vmx2_s2 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(gOutWidth,vrx_s2)) ;
    __vm256 vmx_s2 = _ve_andm_mmm(_ve_andm_mmm(vmx0_s2, vmx1_s2), vmx2_s2) ;

    __vm256 vmall_r0s0 = _ve_andm_mmm(vmy_r0,vmx_s0) ;
    __vm256 vmall_r0s1 = _ve_andm_mmm(vmy_r0,vmx_s1) ;
    __vm256 vmall_r0s2 = _ve_andm_mmm(vmy_r0,vmx_s2) ;

    __vm256 vmall_r1s0 = _ve_andm_mmm(vmy_r1,vmx_s0) ;
    __vm256 vmall_r1s1 = _ve_andm_mmm(vmy_r1,vmx_s1) ;
    __vm256 vmall_r1s2 = _ve_andm_mmm(vmy_r1,vmx_s2) ;

    __vm256 vmall_r2s0 = _ve_andm_mmm(vmy_r2,vmx_s0) ;
    __vm256 vmall_r2s1 = _ve_andm_mmm(vmy_r2,vmx_s1) ;
    __vm256 vmall_r2s2 = _ve_andm_mmm(vmy_r2,vmx_s2) ;

    int64_t k=0;
    if( (gOutChannelGroup & 0x01 ) == 1 ) {
      int64_t gOutIndex    = gOutGroupOffset + ((n * gOutChannel + k) * gOutHeight) * gOutWidth ;
      const float *pKerValue = pKernel + kernGroupOffset + (((k  ) * gInChannelGroup + c) * kernHeight ) * kernWidth ;

#define VFADD_C2(VRGOUT, VM, K, R, S)  {												\
	const uint64_t kerValue01 = _ve_pack_f32p(pKerValue + (((K)*gInChannelGroup + 0) * kernHeight +(R)) * kernWidth + (S),		\
						  pKerValue + (((K)*gInChannelGroup + 1) * kernHeight +(R)) * kernWidth + (S)) ;	\
	VRGOUT = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), VRGOUT, VM) ;				\
	__vr vrgoutP = _ve_vshf_vvvs(VRGOUT, VRGOUT, VE_VSHUFFLE_YUZU) ;			\
	vrsum01 = _ve_pvfmad_vvsv(vrsum01, kerValue01, vrgoutP) ;				\
      }

      __vr vrgout_ptr_k0_r0_s0 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry_r0), vrx_s0),
					 2,
					 (unsigned long)(pGOut+gOutIndex)) ;
      __vr vrgout_k0_r0_s0 = _ve_vgtu_vvm(vrgout_ptr_k0_r0_s0, vmall_r0s0) ;
      __vr vrgout_ptr_k0_r0_s1 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry_r0), vrx_s1),
					 2,
					 (unsigned long)(pGOut+gOutIndex)) ;
      __vr vrgout_k0_r0_s1 = _ve_vgtu_vvm(vrgout_ptr_k0_r0_s1, vmall_r0s1) ;
      __vr vrgout_ptr_k0_r0_s2 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry_r0), vrx_s2),
					 2,
					 (unsigned long)(pGOut+gOutIndex)) ;
      __vr vrgout_k0_r0_s2 = _ve_vgtu_vvm(vrgout_ptr_k0_r0_s2, vmall_r0s2) ;

      VFADD_C2(vrgout_k0_r0_s0, vmall_r0s0, 0, 0, 0)
      VFADD_C2(vrgout_k0_r0_s1, vmall_r0s1, 0, 0, 1)
      VFADD_C2(vrgout_k0_r0_s2, vmall_r0s2, 0, 0, 2)

      __vr vrgout_ptr_k0_r1_s0 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry_r1), vrx_s0),
					 2,
					 (unsigned long)(pGOut+gOutIndex)) ;
      __vr vrgout_k0_r1_s0 = _ve_vgtu_vvm(vrgout_ptr_k0_r1_s0, vmall_r1s0) ;
      __vr vrgout_ptr_k0_r1_s1 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry_r1), vrx_s1),
					 2,
					 (unsigned long)(pGOut+gOutIndex)) ;
      __vr vrgout_k0_r1_s1 = _ve_vgtu_vvm(vrgout_ptr_k0_r1_s1, vmall_r1s1) ;
      __vr vrgout_ptr_k0_r1_s2 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry_r1), vrx_s2),
					 2,
					 (unsigned long)(pGOut+gOutIndex)) ;
      __vr vrgout_k0_r1_s2 = _ve_vgtu_vvm(vrgout_ptr_k0_r1_s2, vmall_r1s2) ;

      VFADD_C2(vrgout_k0_r1_s0, vmall_r1s0, 0, 1, 0)
      VFADD_C2(vrgout_k0_r1_s1, vmall_r1s1, 0, 1, 1)
      VFADD_C2(vrgout_k0_r1_s2, vmall_r1s2, 0, 1, 2)

      __vr vrgout_ptr_k0_r2_s0 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry_r2), vrx_s0),
					 2,
					 (unsigned long)(pGOut+gOutIndex)) ;
      __vr vrgout_k0_r2_s0 = _ve_vgtu_vvm(vrgout_ptr_k0_r2_s0, vmall_r2s0) ;
      __vr vrgout_ptr_k0_r2_s1 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry_r2), vrx_s1),
					 2,
					 (unsigned long)(pGOut+gOutIndex)) ;
      __vr vrgout_k0_r2_s1 = _ve_vgtu_vvm(vrgout_ptr_k0_r2_s1, vmall_r2s1) ;
      __vr vrgout_ptr_k0_r2_s2 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry_r2), vrx_s2),
					 2,
					 (unsigned long)(pGOut+gOutIndex)) ;
      __vr vrgout_k0_r2_s2 = _ve_vgtu_vvm(vrgout_ptr_k0_r2_s2, vmall_r2s2) ;

      VFADD_C2(vrgout_k0_r2_s0, vmall_r2s0, 0, 2, 0)
      VFADD_C2(vrgout_k0_r2_s1, vmall_r2s1, 0, 2, 1)
      VFADD_C2(vrgout_k0_r2_s2, vmall_r2s2, 0, 2, 2)

      k+=1 ;
    }
    if( ((gOutChannelGroup >> 1) & 0x01 ) == 1 ) {
      int64_t gOutIndex    = gOutGroupOffset + ((n * gOutChannel + k) * gOutHeight) * gOutWidth ;
      const float *pKerValue = pKernel + kernGroupOffset + (((k  ) * gInChannelGroup + c) * kernHeight) * kernWidth ;

      __vr vrgout_ptr_k0_r0_s0 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry_r0), vrx_s0),
					 2,
					 (unsigned long)(pGOut+gOutIndex)) ;
      __vr vrgout_k0_r0_s0 = _ve_vgtu_vvm(vrgout_ptr_k0_r0_s0, vmall_r0s0) ;
      __vr vrgout_ptr_k1_r0_s0 = _ve_vaddsl_vsv(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_r0_s0) ;
      __vr vrgout_k1_r0_s0 = _ve_vgtu_vvm(vrgout_ptr_k1_r0_s0, vmall_r0s0) ;

      VFADD_C2(vrgout_k0_r0_s0, vmall_r0s0, 0, 0, 0)
      VFADD_C2(vrgout_k1_r0_s0, vmall_r0s0, 1, 0, 0)

      __vr vrgout_ptr_k0_r0_s1 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry_r0), vrx_s1),
					 2,
					 (unsigned long)(pGOut+gOutIndex)) ;
      __vr vrgout_k0_r0_s1 = _ve_vgtu_vvm(vrgout_ptr_k0_r0_s1, vmall_r0s1) ;
      __vr vrgout_ptr_k1_r0_s1 = _ve_vaddsl_vsv(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_r0_s1) ;
      __vr vrgout_k1_r0_s1 = _ve_vgtu_vvm(vrgout_ptr_k1_r0_s1, vmall_r0s1) ;

      VFADD_C2(vrgout_k0_r0_s1, vmall_r0s1, 0, 0, 1)
      VFADD_C2(vrgout_k1_r0_s1, vmall_r0s1, 1, 0, 1)

      __vr vrgout_ptr_k0_r0_s2 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry_r0), vrx_s2),
					 2,
					 (unsigned long)(pGOut+gOutIndex)) ;
      __vr vrgout_k0_r0_s2 = _ve_vgtu_vvm(vrgout_ptr_k0_r0_s2, vmall_r0s2) ;
      __vr vrgout_ptr_k1_r0_s2 = _ve_vaddsl_vsv(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_r0_s2) ;
      __vr vrgout_k1_r0_s2 = _ve_vgtu_vvm(vrgout_ptr_k1_r0_s2, vmall_r0s2) ;

      VFADD_C2(vrgout_k0_r0_s2, vmall_r0s2, 0, 0, 2)
      VFADD_C2(vrgout_k1_r0_s2, vmall_r0s2, 1, 0, 2)

      __vr vrgout_ptr_k0_r1_s0 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry_r1), vrx_s0),
					 2,
					 (unsigned long)(pGOut+gOutIndex)) ;
      __vr vrgout_k0_r1_s0 = _ve_vgtu_vvm(vrgout_ptr_k0_r1_s0, vmall_r1s0) ;
      __vr vrgout_ptr_k1_r1_s0 = _ve_vaddsl_vsv(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_r1_s0) ;
      __vr vrgout_k1_r1_s0 = _ve_vgtu_vvm(vrgout_ptr_k1_r1_s0, vmall_r1s0) ;

      VFADD_C2(vrgout_k0_r1_s0, vmall_r1s0, 0, 1, 0)
      VFADD_C2(vrgout_k1_r1_s0, vmall_r1s0, 1, 1, 0)

      __vr vrgout_ptr_k0_r1_s1 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry_r1), vrx_s1),
					 2,
					 (unsigned long)(pGOut+gOutIndex)) ;
      __vr vrgout_k0_r1_s1 = _ve_vgtu_vvm(vrgout_ptr_k0_r1_s1, vmall_r1s1) ;
      __vr vrgout_ptr_k1_r1_s1 = _ve_vaddsl_vsv(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_r1_s1) ;
      __vr vrgout_k1_r1_s1 = _ve_vgtu_vvm(vrgout_ptr_k1_r1_s1, vmall_r1s1) ;

      VFADD_C2(vrgout_k0_r1_s1, vmall_r1s1, 0, 1, 1)
      VFADD_C2(vrgout_k1_r1_s1, vmall_r1s1, 1, 1, 1)

      __vr vrgout_ptr_k0_r1_s2 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry_r1), vrx_s2),
					 2,
					 (unsigned long)(pGOut+gOutIndex)) ;
      __vr vrgout_k0_r1_s2 = _ve_vgtu_vvm(vrgout_ptr_k0_r1_s2, vmall_r1s2) ;
      __vr vrgout_ptr_k1_r1_s2 = _ve_vaddsl_vsv(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_r1_s2) ;
      __vr vrgout_k1_r1_s2 = _ve_vgtu_vvm(vrgout_ptr_k1_r1_s2, vmall_r1s2) ;

      VFADD_C2(vrgout_k0_r1_s2, vmall_r1s2, 0, 1, 2)
      VFADD_C2(vrgout_k1_r1_s2, vmall_r1s2, 1, 1, 2)

      __vr vrgout_ptr_k0_r2_s0 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry_r2), vrx_s0),
					 2,
					 (unsigned long)(pGOut+gOutIndex)) ;
      __vr vrgout_k0_r2_s0 = _ve_vgtu_vvm(vrgout_ptr_k0_r2_s0, vmall_r2s0) ;
      __vr vrgout_ptr_k1_r2_s0 = _ve_vaddsl_vsv(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_r2_s0) ;
      __vr vrgout_k1_r2_s0 = _ve_vgtu_vvm(vrgout_ptr_k1_r2_s0, vmall_r2s0) ;

      VFADD_C2(vrgout_k0_r2_s0, vmall_r2s0, 0, 2, 0)
      VFADD_C2(vrgout_k1_r2_s0, vmall_r2s0, 1, 2, 0)

      __vr vrgout_ptr_k0_r2_s1 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry_r2), vrx_s1),
					 2,
					 (unsigned long)(pGOut+gOutIndex)) ;
      __vr vrgout_k0_r2_s1 = _ve_vgtu_vvm(vrgout_ptr_k0_r2_s1, vmall_r2s1) ;
      __vr vrgout_ptr_k1_r2_s1 = _ve_vaddsl_vsv(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_r2_s1) ;
      __vr vrgout_k1_r2_s1 = _ve_vgtu_vvm(vrgout_ptr_k1_r2_s1, vmall_r2s1) ;

      VFADD_C2(vrgout_k0_r2_s1, vmall_r2s1, 0, 2, 1)
      VFADD_C2(vrgout_k1_r2_s1, vmall_r2s1, 1, 2, 1)

      __vr vrgout_ptr_k0_r2_s2 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry_r2), vrx_s2),
					 2,
					 (unsigned long)(pGOut+gOutIndex)) ;
      __vr vrgout_k0_r2_s2 = _ve_vgtu_vvm(vrgout_ptr_k0_r2_s2, vmall_r2s2) ;
      __vr vrgout_ptr_k1_r2_s2 = _ve_vaddsl_vsv(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_r2_s2) ;
      __vr vrgout_k1_r2_s2 = _ve_vgtu_vvm(vrgout_ptr_k1_r2_s2, vmall_r2s2) ;

      VFADD_C2(vrgout_k0_r2_s2, vmall_r2s2, 0, 2, 2)
      VFADD_C2(vrgout_k1_r2_s2, vmall_r2s2, 1, 2, 2)

      k+=2 ;
    }
    if( ((gOutChannelGroup >> 2) & 0x01 ) == 1 ) {
      int64_t gOutIndex    = gOutGroupOffset + ((n * gOutChannel + k) * gOutHeight) * gOutWidth ;
      const float *pKerValue = pKernel + kernGroupOffset + (((k  ) * gInChannelGroup + c) * kernHeight) * kernWidth ;

      __vr vrgout_ptr_k0_r0_s0 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry_r0), vrx_s0),
					 2,
					 (unsigned long)(pGOut+gOutIndex)) ;
      __vr vrgout_k0_r0_s0 = _ve_vgtu_vvm(vrgout_ptr_k0_r0_s0, vmall_r0s0) ;
      __vr vrgout_ptr_k1_r0_s0 = _ve_vaddsl_vsv(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_r0_s0) ;
      __vr vrgout_k1_r0_s0 = _ve_vgtu_vvm(vrgout_ptr_k1_r0_s0, vmall_r0s0) ;
      __vr vrgout_ptr_r0_k2_s0 = _ve_vaddsl_vsv(4*2*gOutHeight*gOutWidth, vrgout_ptr_k0_r0_s0) ;
      __vr vrgout_k2_r0_s0 = _ve_vgtu_vvm(vrgout_ptr_r0_k2_s0, vmall_r0s0) ;
      __vr vrgout_ptr_k3_r0_s0 = _ve_vaddsl_vsv(4*3*gOutHeight*gOutWidth, vrgout_ptr_k0_r0_s0) ;
      __vr vrgout_k3_r0_s0 = _ve_vgtu_vvm(vrgout_ptr_k3_r0_s0, vmall_r0s0) ;

      VFADD_C2(vrgout_k0_r0_s0, vmall_r0s0, 0, 0, 0)
      VFADD_C2(vrgout_k1_r0_s0, vmall_r0s0, 1, 0, 0)
      VFADD_C2(vrgout_k2_r0_s0, vmall_r0s0, 2, 0, 0)
      VFADD_C2(vrgout_k3_r0_s0, vmall_r0s0, 3, 0, 0)

      __vr vrgout_ptr_k0_r0_s1 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry_r0), vrx_s1),
					 2,
					 (unsigned long)(pGOut+gOutIndex)) ;
      __vr vrgout_k0_r0_s1 = _ve_vgtu_vvm(vrgout_ptr_k0_r0_s1, vmall_r0s1) ;
      __vr vrgout_ptr_k1_r0_s1 = _ve_vaddsl_vsv(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_r0_s1) ;
      __vr vrgout_k1_r0_s1 = _ve_vgtu_vvm(vrgout_ptr_k1_r0_s1, vmall_r0s1) ;
      __vr vrgout_ptr_k2_r0_s1 = _ve_vaddsl_vsv(4*2*gOutHeight*gOutWidth, vrgout_ptr_k0_r0_s1) ;
      __vr vrgout_k2_r0_s1 = _ve_vgtu_vvm(vrgout_ptr_k2_r0_s1, vmall_r0s1) ;
      __vr vrgout_ptr_k3_r0_s1 = _ve_vaddsl_vsv(4*3*gOutHeight*gOutWidth, vrgout_ptr_k0_r0_s1) ;
      __vr vrgout_k3_r0_s1 = _ve_vgtu_vvm(vrgout_ptr_k3_r0_s1, vmall_r0s1) ;

      VFADD_C2(vrgout_k0_r0_s1, vmall_r0s1, 0, 0, 1)
      VFADD_C2(vrgout_k1_r0_s1, vmall_r0s1, 1, 0, 1)
      VFADD_C2(vrgout_k2_r0_s1, vmall_r0s1, 2, 0, 1)
      VFADD_C2(vrgout_k3_r0_s1, vmall_r0s1, 3, 0, 1)

      __vr vrgout_ptr_k0_r0_s2 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry_r0), vrx_s2),
					 2,
					 (unsigned long)(pGOut+gOutIndex)) ;
      __vr vrgout_k0_r0_s2 = _ve_vgtu_vvm(vrgout_ptr_k0_r0_s2, vmall_r0s2) ;
      __vr vrgout_ptr_k1_r0_s2 = _ve_vaddsl_vsv(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_r0_s2) ;
      __vr vrgout_k1_r0_s2 = _ve_vgtu_vvm(vrgout_ptr_k1_r0_s2, vmall_r0s2) ;
      __vr vrgout_ptr_k2_r0_s2 = _ve_vaddsl_vsv(4*2*gOutHeight*gOutWidth, vrgout_ptr_k0_r0_s2) ;
      __vr vrgout_k2_r0_s2 = _ve_vgtu_vvm(vrgout_ptr_k2_r0_s2, vmall_r0s2) ;
      __vr vrgout_ptr_k3_r0_s2 = _ve_vaddsl_vsv(4*3*gOutHeight*gOutWidth, vrgout_ptr_k0_r0_s2) ;
      __vr vrgout_k3_r0_s2 = _ve_vgtu_vvm(vrgout_ptr_k3_r0_s2, vmall_r0s2) ;

      VFADD_C2(vrgout_k0_r0_s2, vmall_r0s2, 0, 0, 2)
      VFADD_C2(vrgout_k1_r0_s2, vmall_r0s2, 1, 0, 2)
      VFADD_C2(vrgout_k2_r0_s2, vmall_r0s2, 2, 0, 2)
      VFADD_C2(vrgout_k3_r0_s2, vmall_r0s2, 3, 0, 2)

      __vr vrgout_ptr_k0_r1_s0 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry_r1), vrx_s0),
					 2,
					 (unsigned long)(pGOut+gOutIndex)) ;
      __vr vrgout_k0_r1_s0 = _ve_vgtu_vvm(vrgout_ptr_k0_r1_s0, vmall_r1s0) ;
      __vr vrgout_ptr_k1_r1_s0 = _ve_vaddsl_vsv(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_r1_s0) ;
      __vr vrgout_k1_r1_s0 = _ve_vgtu_vvm(vrgout_ptr_k1_r1_s0, vmall_r1s0) ;
      __vr vrgout_ptr_k2_r1_s0 = _ve_vaddsl_vsv(4*2*gOutHeight*gOutWidth, vrgout_ptr_k0_r1_s0) ;
      __vr vrgout_k2_r1_s0 = _ve_vgtu_vvm(vrgout_ptr_k2_r1_s0, vmall_r1s0) ;
      __vr vrgout_ptr_k3_r1_s0 = _ve_vaddsl_vsv(4*3*gOutHeight*gOutWidth, vrgout_ptr_k0_r1_s0) ;
      __vr vrgout_k3_r1_s0 = _ve_vgtu_vvm(vrgout_ptr_k3_r1_s0, vmall_r1s0) ;

      VFADD_C2(vrgout_k0_r1_s0, vmall_r1s0, 0, 1, 0)
      VFADD_C2(vrgout_k1_r1_s0, vmall_r1s0, 1, 1, 0)
      VFADD_C2(vrgout_k2_r1_s0, vmall_r1s0, 2, 1, 0)
      VFADD_C2(vrgout_k3_r1_s0, vmall_r1s0, 3, 1, 0)

      __vr vrgout_ptr_k0_r1_s1 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry_r1), vrx_s1),
					 2,
					 (unsigned long)(pGOut+gOutIndex)) ;
      __vr vrgout_k0_r1_s1 = _ve_vgtu_vvm(vrgout_ptr_k0_r1_s1, vmall_r1s1) ;
      __vr vrgout_ptr_k1_r1_s1 = _ve_vaddsl_vsv(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_r1_s1) ;
      __vr vrgout_k1_r1_s1 = _ve_vgtu_vvm(vrgout_ptr_k1_r1_s1, vmall_r1s1) ;
      __vr vrgout_ptr_k2_r1_s1 = _ve_vaddsl_vsv(4*2*gOutHeight*gOutWidth, vrgout_ptr_k0_r1_s1) ;
      __vr vrgout_k2_r1_s1 = _ve_vgtu_vvm(vrgout_ptr_k2_r1_s1, vmall_r1s1) ;
      __vr vrgout_ptr_k3_r1_s1 = _ve_vaddsl_vsv(4*3*gOutHeight*gOutWidth, vrgout_ptr_k0_r1_s1) ;
      __vr vrgout_k3_r1_s1 = _ve_vgtu_vvm(vrgout_ptr_k3_r1_s1, vmall_r1s1) ;

      VFADD_C2(vrgout_k0_r1_s1, vmall_r1s1, 0, 1, 1)
      VFADD_C2(vrgout_k1_r1_s1, vmall_r1s1, 1, 1, 1)
      VFADD_C2(vrgout_k2_r1_s1, vmall_r1s1, 2, 1, 1)
      VFADD_C2(vrgout_k3_r1_s1, vmall_r1s1, 3, 1, 1)

      __vr vrgout_ptr_k0_r1_s2 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry_r1), vrx_s2),
					 2,
					 (unsigned long)(pGOut+gOutIndex)) ;
      __vr vrgout_k0_r1_s2 = _ve_vgtu_vvm(vrgout_ptr_k0_r1_s2, vmall_r1s2) ;
      __vr vrgout_ptr_k1_r1_s2 = _ve_vaddsl_vsv(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_r1_s2) ;
      __vr vrgout_k1_r1_s2 = _ve_vgtu_vvm(vrgout_ptr_k1_r1_s2, vmall_r1s2) ;
      __vr vrgout_ptr_k2_r1_s2 = _ve_vaddsl_vsv(4*2*gOutHeight*gOutWidth, vrgout_ptr_k0_r1_s2) ;
      __vr vrgout_k2_r1_s2 = _ve_vgtu_vvm(vrgout_ptr_k2_r1_s2, vmall_r1s2) ;
      __vr vrgout_ptr_k3_r1_s2 = _ve_vaddsl_vsv(4*3*gOutHeight*gOutWidth, vrgout_ptr_k0_r1_s2) ;
      __vr vrgout_k3_r1_s2 = _ve_vgtu_vvm(vrgout_ptr_k3_r1_s2, vmall_r1s2) ;

      VFADD_C2(vrgout_k0_r1_s2, vmall_r1s2, 0, 1, 2)
      VFADD_C2(vrgout_k1_r1_s2, vmall_r1s2, 1, 1, 2)
      VFADD_C2(vrgout_k2_r1_s2, vmall_r1s2, 2, 1, 2)
      VFADD_C2(vrgout_k3_r1_s2, vmall_r1s2, 3, 1, 2)

      __vr vrgout_ptr_k0_r2_s0 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry_r2), vrx_s0),
					 2,
					 (unsigned long)(pGOut+gOutIndex)) ;
      __vr vrgout_k0_r2_s0 = _ve_vgtu_vvm(vrgout_ptr_k0_r2_s0, vmall_r2s0) ;
      __vr vrgout_ptr_k1_r2_s0 = _ve_vaddsl_vsv(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_r2_s0) ;
      __vr vrgout_k1_r2_s0 = _ve_vgtu_vvm(vrgout_ptr_k1_r2_s0, vmall_r2s0) ;
      __vr vrgout_ptr_k2_r2_s0 = _ve_vaddsl_vsv(4*2*gOutHeight*gOutWidth, vrgout_ptr_k0_r2_s0) ;
      __vr vrgout_k2_r2_s0 = _ve_vgtu_vvm(vrgout_ptr_k2_r2_s0, vmall_r2s0) ;
      __vr vrgout_ptr_k3_r2_s0 = _ve_vaddsl_vsv(4*3*gOutHeight*gOutWidth, vrgout_ptr_k0_r2_s0) ;
      __vr vrgout_k3_r2_s0 = _ve_vgtu_vvm(vrgout_ptr_k3_r2_s0, vmall_r2s0) ;

      VFADD_C2(vrgout_k0_r2_s0, vmall_r2s0, 0, 2, 0)
      VFADD_C2(vrgout_k1_r2_s0, vmall_r2s0, 1, 2, 0)
      VFADD_C2(vrgout_k2_r2_s0, vmall_r2s0, 2, 2, 0)
      VFADD_C2(vrgout_k3_r2_s0, vmall_r2s0, 3, 2, 0)

      __vr vrgout_ptr_k0_r2_s1 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry_r2), vrx_s1),
					 2,
					 (unsigned long)(pGOut+gOutIndex)) ;
      __vr vrgout_k0_r2_s1 = _ve_vgtu_vvm(vrgout_ptr_k0_r2_s1, vmall_r2s1) ;
      __vr vrgout_ptr_k1_r2_s1 = _ve_vaddsl_vsv(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_r2_s1) ;
      __vr vrgout_k1_r2_s1 = _ve_vgtu_vvm(vrgout_ptr_k1_r2_s1, vmall_r2s1) ;
      __vr vrgout_ptr_k2_r2_s1 = _ve_vaddsl_vsv(4*2*gOutHeight*gOutWidth, vrgout_ptr_k0_r2_s1) ;
      __vr vrgout_k2_r2_s1 = _ve_vgtu_vvm(vrgout_ptr_k2_r2_s1, vmall_r2s1) ;
      __vr vrgout_ptr_k3_r2_s1 = _ve_vaddsl_vsv(4*3*gOutHeight*gOutWidth, vrgout_ptr_k0_r2_s1) ;
      __vr vrgout_k3_r2_s1 = _ve_vgtu_vvm(vrgout_ptr_k3_r2_s1, vmall_r2s1) ;

      VFADD_C2(vrgout_k0_r2_s1, vmall_r2s1, 0, 2, 1)
      VFADD_C2(vrgout_k1_r2_s1, vmall_r2s1, 1, 2, 1)
      VFADD_C2(vrgout_k2_r2_s1, vmall_r2s1, 2, 2, 1)
      VFADD_C2(vrgout_k3_r2_s1, vmall_r2s1, 3, 2, 1)

      __vr vrgout_ptr_k0_r2_s2 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry_r2), vrx_s2),
					 2,
					 (unsigned long)(pGOut+gOutIndex)) ;
      __vr vrgout_k0_r2_s2 = _ve_vgtu_vvm(vrgout_ptr_k0_r2_s2, vmall_r2s2) ;
      __vr vrgout_ptr_k1_r2_s2 = _ve_vaddsl_vsv(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_r2_s2) ;
      __vr vrgout_k1_r2_s2 = _ve_vgtu_vvm(vrgout_ptr_k1_r2_s2, vmall_r2s2) ;
      __vr vrgout_ptr_k2_r2_s2 = _ve_vaddsl_vsv(4*2*gOutHeight*gOutWidth, vrgout_ptr_k0_r2_s2) ;
      __vr vrgout_k2_r2_s2 = _ve_vgtu_vvm(vrgout_ptr_k2_r2_s2, vmall_r2s2) ;
      __vr vrgout_ptr_k3_r2_s2 = _ve_vaddsl_vsv(4*3*gOutHeight*gOutWidth, vrgout_ptr_k0_r2_s2) ;
      __vr vrgout_k3_r2_s2 = _ve_vgtu_vvm(vrgout_ptr_k3_r2_s2, vmall_r2s2) ;

      VFADD_C2(vrgout_k0_r2_s2, vmall_r2s2, 0, 2, 2)
      VFADD_C2(vrgout_k1_r2_s2, vmall_r2s2, 1, 2, 2)
      VFADD_C2(vrgout_k2_r2_s2, vmall_r2s2, 2, 2, 2)
      VFADD_C2(vrgout_k3_r2_s2, vmall_r2s2, 3, 2, 2)

      k+=4 ;
    }
    for (; k<gOutChannelGroup; k+=8) {
      int64_t gOutIndex    = gOutGroupOffset + ((n * gOutChannel + k) * gOutHeight) * gOutWidth ;
      const float *pKerValue = pKernel + kernGroupOffset + (((k  ) * gInChannelGroup + c) * kernHeight) * kernWidth ;

      __vr vrgout_ptr_k0_r0_s0 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry_r0), vrx_s0),
					 2,
					 (unsigned long)(pGOut+gOutIndex)) ;
      __vr vrgout_k0_r0_s0 = _ve_vgtu_vvm(vrgout_ptr_k0_r0_s0, vmall_r0s0) ;
      __vr vrgout_ptr_k1_r0_s0 = _ve_vaddsl_vsv(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_r0_s0) ;
      __vr vrgout_k1_r0_s0 = _ve_vgtu_vvm(vrgout_ptr_k1_r0_s0, vmall_r0s0) ;
      __vr vrgout_ptr_r0_k2_s0 = _ve_vaddsl_vsv(4*2*gOutHeight*gOutWidth, vrgout_ptr_k0_r0_s0) ;
      __vr vrgout_k2_r0_s0 = _ve_vgtu_vvm(vrgout_ptr_r0_k2_s0, vmall_r0s0) ;
      __vr vrgout_ptr_k3_r0_s0 = _ve_vaddsl_vsv(4*3*gOutHeight*gOutWidth, vrgout_ptr_k0_r0_s0) ;
      __vr vrgout_k3_r0_s0 = _ve_vgtu_vvm(vrgout_ptr_k3_r0_s0, vmall_r0s0) ;
      __vr vrgout_ptr_k4_r0_s0 = _ve_vaddsl_vsv(4*4*gOutHeight*gOutWidth, vrgout_ptr_k0_r0_s0) ;
      __vr vrgout_k4_r0_s0 = _ve_vgtu_vvm(vrgout_ptr_k4_r0_s0, vmall_r0s0) ;
      __vr vrgout_ptr_k5_r0_s0 = _ve_vaddsl_vsv(4*5*gOutHeight*gOutWidth, vrgout_ptr_k0_r0_s0) ;
      __vr vrgout_k5_r0_s0 = _ve_vgtu_vvm(vrgout_ptr_k5_r0_s0, vmall_r0s0) ;
      __vr vrgout_ptr_k6_r0_s0 = _ve_vaddsl_vsv(4*6*gOutHeight*gOutWidth, vrgout_ptr_k0_r0_s0) ;
      __vr vrgout_k6_r0_s0 = _ve_vgtu_vvm(vrgout_ptr_k6_r0_s0, vmall_r0s0) ;
      __vr vrgout_ptr_k7_r0_s0 = _ve_vaddsl_vsv(4*7*gOutHeight*gOutWidth, vrgout_ptr_k0_r0_s0) ;
      __vr vrgout_k7_r0_s0 = _ve_vgtu_vvm(vrgout_ptr_k7_r0_s0, vmall_r0s0) ;

      VFADD_C2(vrgout_k0_r0_s0, vmall_r0s0, 0, 0, 0)
      VFADD_C2(vrgout_k1_r0_s0, vmall_r0s0, 1, 0, 0)
      VFADD_C2(vrgout_k2_r0_s0, vmall_r0s0, 2, 0, 0)
      VFADD_C2(vrgout_k3_r0_s0, vmall_r0s0, 3, 0, 0)
      VFADD_C2(vrgout_k4_r0_s0, vmall_r0s0, 4, 0, 0)
      VFADD_C2(vrgout_k5_r0_s0, vmall_r0s0, 5, 0, 0)
      VFADD_C2(vrgout_k6_r0_s0, vmall_r0s0, 6, 0, 0)
      VFADD_C2(vrgout_k7_r0_s0, vmall_r0s0, 7, 0, 0)

      __vr vrgout_ptr_k0_r0_s1 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry_r0), vrx_s1),
					 2,
					 (unsigned long)(pGOut+gOutIndex)) ;
      __vr vrgout_k0_r0_s1 = _ve_vgtu_vvm(vrgout_ptr_k0_r0_s1, vmall_r0s1) ;
      __vr vrgout_ptr_k1_r0_s1 = _ve_vaddsl_vsv(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_r0_s1) ;
      __vr vrgout_k1_r0_s1 = _ve_vgtu_vvm(vrgout_ptr_k1_r0_s1, vmall_r0s1) ;
      __vr vrgout_ptr_k2_r0_s1 = _ve_vaddsl_vsv(4*2*gOutHeight*gOutWidth, vrgout_ptr_k0_r0_s1) ;
      __vr vrgout_k2_r0_s1 = _ve_vgtu_vvm(vrgout_ptr_k2_r0_s1, vmall_r0s1) ;
      __vr vrgout_ptr_k3_r0_s1 = _ve_vaddsl_vsv(4*3*gOutHeight*gOutWidth, vrgout_ptr_k0_r0_s1) ;
      __vr vrgout_k3_r0_s1 = _ve_vgtu_vvm(vrgout_ptr_k3_r0_s1, vmall_r0s1) ;
      __vr vrgout_ptr_k4_r0_s1 = _ve_vaddsl_vsv(4*4*gOutHeight*gOutWidth, vrgout_ptr_k0_r0_s1) ;
      __vr vrgout_k4_r0_s1 = _ve_vgtu_vvm(vrgout_ptr_k4_r0_s1, vmall_r0s1) ;
      __vr vrgout_ptr_k5_r0_s1 = _ve_vaddsl_vsv(4*5*gOutHeight*gOutWidth, vrgout_ptr_k0_r0_s1) ;
      __vr vrgout_k5_r0_s1 = _ve_vgtu_vvm(vrgout_ptr_k5_r0_s1, vmall_r0s1) ;
      __vr vrgout_ptr_k6_r0_s1 = _ve_vaddsl_vsv(4*6*gOutHeight*gOutWidth, vrgout_ptr_k0_r0_s1) ;
      __vr vrgout_k6_r0_s1 = _ve_vgtu_vvm(vrgout_ptr_k6_r0_s1, vmall_r0s1) ;
      __vr vrgout_ptr_k7_r0_s1 = _ve_vaddsl_vsv(4*7*gOutHeight*gOutWidth, vrgout_ptr_k0_r0_s1) ;
      __vr vrgout_k7_r0_s1 = _ve_vgtu_vvm(vrgout_ptr_k7_r0_s1, vmall_r0s1) ;

      VFADD_C2(vrgout_k0_r0_s1, vmall_r0s1, 0, 0, 1)
      VFADD_C2(vrgout_k1_r0_s1, vmall_r0s1, 1, 0, 1)
      VFADD_C2(vrgout_k2_r0_s1, vmall_r0s1, 2, 0, 1)
      VFADD_C2(vrgout_k3_r0_s1, vmall_r0s1, 3, 0, 1)
      VFADD_C2(vrgout_k4_r0_s1, vmall_r0s1, 4, 0, 1)
      VFADD_C2(vrgout_k5_r0_s1, vmall_r0s1, 5, 0, 1)
      VFADD_C2(vrgout_k6_r0_s1, vmall_r0s1, 6, 0, 1)
      VFADD_C2(vrgout_k7_r0_s1, vmall_r0s1, 7, 0, 1)

      __vr vrgout_ptr_k0_r0_s2 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry_r0), vrx_s2),
					 2,
					 (unsigned long)(pGOut+gOutIndex)) ;
      __vr vrgout_k0_r0_s2 = _ve_vgtu_vvm(vrgout_ptr_k0_r0_s2, vmall_r0s2) ;
      __vr vrgout_ptr_k1_r0_s2 = _ve_vaddsl_vsv(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_r0_s2) ;
      __vr vrgout_k1_r0_s2 = _ve_vgtu_vvm(vrgout_ptr_k1_r0_s2, vmall_r0s2) ;
      __vr vrgout_ptr_k2_r0_s2 = _ve_vaddsl_vsv(4*2*gOutHeight*gOutWidth, vrgout_ptr_k0_r0_s2) ;
      __vr vrgout_k2_r0_s2 = _ve_vgtu_vvm(vrgout_ptr_k2_r0_s2, vmall_r0s2) ;
      __vr vrgout_ptr_k3_r0_s2 = _ve_vaddsl_vsv(4*3*gOutHeight*gOutWidth, vrgout_ptr_k0_r0_s2) ;
      __vr vrgout_k3_r0_s2 = _ve_vgtu_vvm(vrgout_ptr_k3_r0_s2, vmall_r0s2) ;
      __vr vrgout_ptr_k4_r0_s2 = _ve_vaddsl_vsv(4*4*gOutHeight*gOutWidth, vrgout_ptr_k0_r0_s2) ;
      __vr vrgout_k4_r0_s2 = _ve_vgtu_vvm(vrgout_ptr_k4_r0_s2, vmall_r0s2) ;
      __vr vrgout_ptr_k5_r0_s2 = _ve_vaddsl_vsv(4*5*gOutHeight*gOutWidth, vrgout_ptr_k0_r0_s2) ;
      __vr vrgout_k5_r0_s2 = _ve_vgtu_vvm(vrgout_ptr_k5_r0_s2, vmall_r0s2) ;
      __vr vrgout_ptr_k6_r0_s2 = _ve_vaddsl_vsv(4*6*gOutHeight*gOutWidth, vrgout_ptr_k0_r0_s2) ;
      __vr vrgout_k6_r0_s2 = _ve_vgtu_vvm(vrgout_ptr_k6_r0_s2, vmall_r0s2) ;
      __vr vrgout_ptr_k7_r0_s2 = _ve_vaddsl_vsv(4*7*gOutHeight*gOutWidth, vrgout_ptr_k0_r0_s2) ;
      __vr vrgout_k7_r0_s2 = _ve_vgtu_vvm(vrgout_ptr_k7_r0_s2, vmall_r0s2) ;

      VFADD_C2(vrgout_k0_r0_s2, vmall_r0s2, 0, 0, 2)
      VFADD_C2(vrgout_k1_r0_s2, vmall_r0s2, 1, 0, 2)
      VFADD_C2(vrgout_k2_r0_s2, vmall_r0s2, 2, 0, 2)
      VFADD_C2(vrgout_k3_r0_s2, vmall_r0s2, 3, 0, 2)
      VFADD_C2(vrgout_k4_r0_s2, vmall_r0s2, 4, 0, 2)
      VFADD_C2(vrgout_k5_r0_s2, vmall_r0s2, 5, 0, 2)
      VFADD_C2(vrgout_k6_r0_s2, vmall_r0s2, 6, 0, 2)
      VFADD_C2(vrgout_k7_r0_s2, vmall_r0s2, 7, 0, 2)

      __vr vrgout_ptr_k0_r1_s0 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry_r1), vrx_s0),
					 2,
					 (unsigned long)(pGOut+gOutIndex)) ;
      __vr vrgout_k0_r1_s0 = _ve_vgtu_vvm(vrgout_ptr_k0_r1_s0, vmall_r1s0) ;
      __vr vrgout_ptr_k1_r1_s0 = _ve_vaddsl_vsv(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_r1_s0) ;
      __vr vrgout_k1_r1_s0 = _ve_vgtu_vvm(vrgout_ptr_k1_r1_s0, vmall_r1s0) ;
      __vr vrgout_ptr_k2_r1_s0 = _ve_vaddsl_vsv(4*2*gOutHeight*gOutWidth, vrgout_ptr_k0_r1_s0) ;
      __vr vrgout_k2_r1_s0 = _ve_vgtu_vvm(vrgout_ptr_k2_r1_s0, vmall_r1s0) ;
      __vr vrgout_ptr_k3_r1_s0 = _ve_vaddsl_vsv(4*3*gOutHeight*gOutWidth, vrgout_ptr_k0_r1_s0) ;
      __vr vrgout_k3_r1_s0 = _ve_vgtu_vvm(vrgout_ptr_k3_r1_s0, vmall_r1s0) ;
      __vr vrgout_ptr_k4_r1_s0 = _ve_vaddsl_vsv(4*4*gOutHeight*gOutWidth, vrgout_ptr_k0_r1_s0) ;
      __vr vrgout_k4_r1_s0 = _ve_vgtu_vvm(vrgout_ptr_k4_r1_s0, vmall_r1s0) ;
      __vr vrgout_ptr_k5_r1_s0 = _ve_vaddsl_vsv(4*5*gOutHeight*gOutWidth, vrgout_ptr_k0_r1_s0) ;
      __vr vrgout_k5_r1_s0 = _ve_vgtu_vvm(vrgout_ptr_k5_r1_s0, vmall_r1s0) ;
      __vr vrgout_ptr_k6_r1_s0 = _ve_vaddsl_vsv(4*6*gOutHeight*gOutWidth, vrgout_ptr_k0_r1_s0) ;
      __vr vrgout_k6_r1_s0 = _ve_vgtu_vvm(vrgout_ptr_k6_r1_s0, vmall_r1s0) ;
      __vr vrgout_ptr_k7_r1_s0 = _ve_vaddsl_vsv(4*7*gOutHeight*gOutWidth, vrgout_ptr_k0_r1_s0) ;
      __vr vrgout_k7_r1_s0 = _ve_vgtu_vvm(vrgout_ptr_k7_r1_s0, vmall_r1s0) ;

      VFADD_C2(vrgout_k0_r1_s0, vmall_r1s0, 0, 1, 0)
      VFADD_C2(vrgout_k1_r1_s0, vmall_r1s0, 1, 1, 0)
      VFADD_C2(vrgout_k2_r1_s0, vmall_r1s0, 2, 1, 0)
      VFADD_C2(vrgout_k3_r1_s0, vmall_r1s0, 3, 1, 0)
      VFADD_C2(vrgout_k4_r1_s0, vmall_r1s0, 4, 1, 0)
      VFADD_C2(vrgout_k5_r1_s0, vmall_r1s0, 5, 1, 0)
      VFADD_C2(vrgout_k6_r1_s0, vmall_r1s0, 6, 1, 0)
      VFADD_C2(vrgout_k7_r1_s0, vmall_r1s0, 7, 1, 0)

      __vr vrgout_ptr_k0_r1_s1 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry_r1), vrx_s1),
					 2,
					 (unsigned long)(pGOut+gOutIndex)) ;
      __vr vrgout_k0_r1_s1 = _ve_vgtu_vvm(vrgout_ptr_k0_r1_s1, vmall_r1s1) ;
      __vr vrgout_ptr_k1_r1_s1 = _ve_vaddsl_vsv(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_r1_s1) ;
      __vr vrgout_k1_r1_s1 = _ve_vgtu_vvm(vrgout_ptr_k1_r1_s1, vmall_r1s1) ;
      __vr vrgout_ptr_k2_r1_s1 = _ve_vaddsl_vsv(4*2*gOutHeight*gOutWidth, vrgout_ptr_k0_r1_s1) ;
      __vr vrgout_k2_r1_s1 = _ve_vgtu_vvm(vrgout_ptr_k2_r1_s1, vmall_r1s1) ;
      __vr vrgout_ptr_k3_r1_s1 = _ve_vaddsl_vsv(4*3*gOutHeight*gOutWidth, vrgout_ptr_k0_r1_s1) ;
      __vr vrgout_k3_r1_s1 = _ve_vgtu_vvm(vrgout_ptr_k3_r1_s1, vmall_r1s1) ;
      __vr vrgout_ptr_k4_r1_s1 = _ve_vaddsl_vsv(4*4*gOutHeight*gOutWidth, vrgout_ptr_k0_r1_s1) ;
      __vr vrgout_k4_r1_s1 = _ve_vgtu_vvm(vrgout_ptr_k4_r1_s1, vmall_r1s1) ;
      __vr vrgout_ptr_k5_r1_s1 = _ve_vaddsl_vsv(4*5*gOutHeight*gOutWidth, vrgout_ptr_k0_r1_s1) ;
      __vr vrgout_k5_r1_s1 = _ve_vgtu_vvm(vrgout_ptr_k5_r1_s1, vmall_r1s1) ;
      __vr vrgout_ptr_k6_r1_s1 = _ve_vaddsl_vsv(4*6*gOutHeight*gOutWidth, vrgout_ptr_k0_r1_s1) ;
      __vr vrgout_k6_r1_s1 = _ve_vgtu_vvm(vrgout_ptr_k6_r1_s1, vmall_r1s1) ;
      __vr vrgout_ptr_k7_r1_s1 = _ve_vaddsl_vsv(4*7*gOutHeight*gOutWidth, vrgout_ptr_k0_r1_s1) ;
      __vr vrgout_k7_r1_s1 = _ve_vgtu_vvm(vrgout_ptr_k7_r1_s1, vmall_r1s1) ;

      VFADD_C2(vrgout_k0_r1_s1, vmall_r1s1, 0, 1, 1)
      VFADD_C2(vrgout_k1_r1_s1, vmall_r1s1, 1, 1, 1)
      VFADD_C2(vrgout_k2_r1_s1, vmall_r1s1, 2, 1, 1)
      VFADD_C2(vrgout_k3_r1_s1, vmall_r1s1, 3, 1, 1)
      VFADD_C2(vrgout_k4_r1_s1, vmall_r1s1, 4, 1, 1)
      VFADD_C2(vrgout_k5_r1_s1, vmall_r1s1, 5, 1, 1)
      VFADD_C2(vrgout_k6_r1_s1, vmall_r1s1, 6, 1, 1)
      VFADD_C2(vrgout_k7_r1_s1, vmall_r1s1, 7, 1, 1)

      __vr vrgout_ptr_k0_r1_s2 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry_r1), vrx_s2),
					 2,
					 (unsigned long)(pGOut+gOutIndex)) ;
      __vr vrgout_k0_r1_s2 = _ve_vgtu_vvm(vrgout_ptr_k0_r1_s2, vmall_r1s2) ;
      __vr vrgout_ptr_k1_r1_s2 = _ve_vaddsl_vsv(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_r1_s2) ;
      __vr vrgout_k1_r1_s2 = _ve_vgtu_vvm(vrgout_ptr_k1_r1_s2, vmall_r1s2) ;
      __vr vrgout_ptr_k2_r1_s2 = _ve_vaddsl_vsv(4*2*gOutHeight*gOutWidth, vrgout_ptr_k0_r1_s2) ;
      __vr vrgout_k2_r1_s2 = _ve_vgtu_vvm(vrgout_ptr_k2_r1_s2, vmall_r1s2) ;
      __vr vrgout_ptr_k3_r1_s2 = _ve_vaddsl_vsv(4*3*gOutHeight*gOutWidth, vrgout_ptr_k0_r1_s2) ;
      __vr vrgout_k3_r1_s2 = _ve_vgtu_vvm(vrgout_ptr_k3_r1_s2, vmall_r1s2) ;
      __vr vrgout_ptr_k4_r1_s2 = _ve_vaddsl_vsv(4*4*gOutHeight*gOutWidth, vrgout_ptr_k0_r1_s2) ;
      __vr vrgout_k4_r1_s2 = _ve_vgtu_vvm(vrgout_ptr_k4_r1_s2, vmall_r1s2) ;
      __vr vrgout_ptr_k5_r1_s2 = _ve_vaddsl_vsv(4*5*gOutHeight*gOutWidth, vrgout_ptr_k0_r1_s2) ;
      __vr vrgout_k5_r1_s2 = _ve_vgtu_vvm(vrgout_ptr_k5_r1_s2, vmall_r1s2) ;
      __vr vrgout_ptr_k6_r1_s2 = _ve_vaddsl_vsv(4*6*gOutHeight*gOutWidth, vrgout_ptr_k0_r1_s2) ;
      __vr vrgout_k6_r1_s2 = _ve_vgtu_vvm(vrgout_ptr_k6_r1_s2, vmall_r1s2) ;
      __vr vrgout_ptr_k7_r1_s2 = _ve_vaddsl_vsv(4*7*gOutHeight*gOutWidth, vrgout_ptr_k0_r1_s2) ;
      __vr vrgout_k7_r1_s2 = _ve_vgtu_vvm(vrgout_ptr_k7_r1_s2, vmall_r1s2) ;

      VFADD_C2(vrgout_k0_r1_s2, vmall_r1s2, 0, 1, 2)
      VFADD_C2(vrgout_k1_r1_s2, vmall_r1s2, 1, 1, 2)
      VFADD_C2(vrgout_k2_r1_s2, vmall_r1s2, 2, 1, 2)
      VFADD_C2(vrgout_k3_r1_s2, vmall_r1s2, 3, 1, 2)
      VFADD_C2(vrgout_k4_r1_s2, vmall_r1s2, 4, 1, 2)
      VFADD_C2(vrgout_k5_r1_s2, vmall_r1s2, 5, 1, 2)
      VFADD_C2(vrgout_k6_r1_s2, vmall_r1s2, 6, 1, 2)
      VFADD_C2(vrgout_k7_r1_s2, vmall_r1s2, 7, 1, 2)

      __vr vrgout_ptr_k0_r2_s0 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry_r2), vrx_s0),
					 2,
					 (unsigned long)(pGOut+gOutIndex)) ;
      __vr vrgout_k0_r2_s0 = _ve_vgtu_vvm(vrgout_ptr_k0_r2_s0, vmall_r2s0) ;
      __vr vrgout_ptr_k1_r2_s0 = _ve_vaddsl_vsv(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_r2_s0) ;
      __vr vrgout_k1_r2_s0 = _ve_vgtu_vvm(vrgout_ptr_k1_r2_s0, vmall_r2s0) ;
      __vr vrgout_ptr_k2_r2_s0 = _ve_vaddsl_vsv(4*2*gOutHeight*gOutWidth, vrgout_ptr_k0_r2_s0) ;
      __vr vrgout_k2_r2_s0 = _ve_vgtu_vvm(vrgout_ptr_k2_r2_s0, vmall_r2s0) ;
      __vr vrgout_ptr_k3_r2_s0 = _ve_vaddsl_vsv(4*3*gOutHeight*gOutWidth, vrgout_ptr_k0_r2_s0) ;
      __vr vrgout_k3_r2_s0 = _ve_vgtu_vvm(vrgout_ptr_k3_r2_s0, vmall_r2s0) ;
      __vr vrgout_ptr_k4_r2_s0 = _ve_vaddsl_vsv(4*4*gOutHeight*gOutWidth, vrgout_ptr_k0_r2_s0) ;
      __vr vrgout_k4_r2_s0 = _ve_vgtu_vvm(vrgout_ptr_k4_r2_s0, vmall_r2s0) ;
      __vr vrgout_ptr_k5_r2_s0 = _ve_vaddsl_vsv(4*5*gOutHeight*gOutWidth, vrgout_ptr_k0_r2_s0) ;
      __vr vrgout_k5_r2_s0 = _ve_vgtu_vvm(vrgout_ptr_k5_r2_s0, vmall_r2s0) ;
      __vr vrgout_ptr_k6_r2_s0 = _ve_vaddsl_vsv(4*6*gOutHeight*gOutWidth, vrgout_ptr_k0_r2_s0) ;
      __vr vrgout_k6_r2_s0 = _ve_vgtu_vvm(vrgout_ptr_k6_r2_s0, vmall_r2s0) ;
      __vr vrgout_ptr_k7_r2_s0 = _ve_vaddsl_vsv(4*7*gOutHeight*gOutWidth, vrgout_ptr_k0_r2_s0) ;
      __vr vrgout_k7_r2_s0 = _ve_vgtu_vvm(vrgout_ptr_k7_r2_s0, vmall_r2s0) ;

      VFADD_C2(vrgout_k0_r2_s0, vmall_r2s0, 0, 2, 0)
      VFADD_C2(vrgout_k1_r2_s0, vmall_r2s0, 1, 2, 0)
      VFADD_C2(vrgout_k2_r2_s0, vmall_r2s0, 2, 2, 0)
      VFADD_C2(vrgout_k3_r2_s0, vmall_r2s0, 3, 2, 0)
      VFADD_C2(vrgout_k4_r2_s0, vmall_r2s0, 4, 2, 0)
      VFADD_C2(vrgout_k5_r2_s0, vmall_r2s0, 5, 2, 0)
      VFADD_C2(vrgout_k6_r2_s0, vmall_r2s0, 6, 2, 0)
      VFADD_C2(vrgout_k7_r2_s0, vmall_r2s0, 7, 2, 0)

      __vr vrgout_ptr_k0_r2_s1 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry_r2), vrx_s1),
					 2,
					 (unsigned long)(pGOut+gOutIndex)) ;
      __vr vrgout_k0_r2_s1 = _ve_vgtu_vvm(vrgout_ptr_k0_r2_s1, vmall_r2s1) ;
      __vr vrgout_ptr_k1_r2_s1 = _ve_vaddsl_vsv(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_r2_s1) ;
      __vr vrgout_k1_r2_s1 = _ve_vgtu_vvm(vrgout_ptr_k1_r2_s1, vmall_r2s1) ;
      __vr vrgout_ptr_k2_r2_s1 = _ve_vaddsl_vsv(4*2*gOutHeight*gOutWidth, vrgout_ptr_k0_r2_s1) ;
      __vr vrgout_k2_r2_s1 = _ve_vgtu_vvm(vrgout_ptr_k2_r2_s1, vmall_r2s1) ;
      __vr vrgout_ptr_k3_r2_s1 = _ve_vaddsl_vsv(4*3*gOutHeight*gOutWidth, vrgout_ptr_k0_r2_s1) ;
      __vr vrgout_k3_r2_s1 = _ve_vgtu_vvm(vrgout_ptr_k3_r2_s1, vmall_r2s1) ;
      __vr vrgout_ptr_k4_r2_s1 = _ve_vaddsl_vsv(4*4*gOutHeight*gOutWidth, vrgout_ptr_k0_r2_s1) ;
      __vr vrgout_k4_r2_s1 = _ve_vgtu_vvm(vrgout_ptr_k4_r2_s1, vmall_r2s1) ;
      __vr vrgout_ptr_k5_r2_s1 = _ve_vaddsl_vsv(4*5*gOutHeight*gOutWidth, vrgout_ptr_k0_r2_s1) ;
      __vr vrgout_k5_r2_s1 = _ve_vgtu_vvm(vrgout_ptr_k5_r2_s1, vmall_r2s1) ;
      __vr vrgout_ptr_k6_r2_s1 = _ve_vaddsl_vsv(4*6*gOutHeight*gOutWidth, vrgout_ptr_k0_r2_s1) ;
      __vr vrgout_k6_r2_s1 = _ve_vgtu_vvm(vrgout_ptr_k6_r2_s1, vmall_r2s1) ;
      __vr vrgout_ptr_k7_r2_s1 = _ve_vaddsl_vsv(4*7*gOutHeight*gOutWidth, vrgout_ptr_k0_r2_s1) ;
      __vr vrgout_k7_r2_s1 = _ve_vgtu_vvm(vrgout_ptr_k7_r2_s1, vmall_r2s1) ;

      VFADD_C2(vrgout_k0_r2_s1, vmall_r2s1, 0, 2, 1)
      VFADD_C2(vrgout_k1_r2_s1, vmall_r2s1, 1, 2, 1)
      VFADD_C2(vrgout_k2_r2_s1, vmall_r2s1, 2, 2, 1)
      VFADD_C2(vrgout_k3_r2_s1, vmall_r2s1, 3, 2, 1)
      VFADD_C2(vrgout_k4_r2_s1, vmall_r2s1, 4, 2, 1)
      VFADD_C2(vrgout_k5_r2_s1, vmall_r2s1, 5, 2, 1)
      VFADD_C2(vrgout_k6_r2_s1, vmall_r2s1, 6, 2, 1)
      VFADD_C2(vrgout_k7_r2_s1, vmall_r2s1, 7, 2, 1)

      __vr vrgout_ptr_k0_r2_s2 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry_r2), vrx_s2),
					 2,
					 (unsigned long)(pGOut+gOutIndex)) ;
      __vr vrgout_k0_r2_s2 = _ve_vgtu_vvm(vrgout_ptr_k0_r2_s2, vmall_r2s2) ;
      __vr vrgout_ptr_k1_r2_s2 = _ve_vaddsl_vsv(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_r2_s2) ;
      __vr vrgout_k1_r2_s2 = _ve_vgtu_vvm(vrgout_ptr_k1_r2_s2, vmall_r2s2) ;
      __vr vrgout_ptr_k2_r2_s2 = _ve_vaddsl_vsv(4*2*gOutHeight*gOutWidth, vrgout_ptr_k0_r2_s2) ;
      __vr vrgout_k2_r2_s2 = _ve_vgtu_vvm(vrgout_ptr_k2_r2_s2, vmall_r2s2) ;
      __vr vrgout_ptr_k3_r2_s2 = _ve_vaddsl_vsv(4*3*gOutHeight*gOutWidth, vrgout_ptr_k0_r2_s2) ;
      __vr vrgout_k3_r2_s2 = _ve_vgtu_vvm(vrgout_ptr_k3_r2_s2, vmall_r2s2) ;
      __vr vrgout_ptr_k4_r2_s2 = _ve_vaddsl_vsv(4*4*gOutHeight*gOutWidth, vrgout_ptr_k0_r2_s2) ;
      __vr vrgout_k4_r2_s2 = _ve_vgtu_vvm(vrgout_ptr_k4_r2_s2, vmall_r2s2) ;
      __vr vrgout_ptr_k5_r2_s2 = _ve_vaddsl_vsv(4*5*gOutHeight*gOutWidth, vrgout_ptr_k0_r2_s2) ;
      __vr vrgout_k5_r2_s2 = _ve_vgtu_vvm(vrgout_ptr_k5_r2_s2, vmall_r2s2) ;
      __vr vrgout_ptr_k6_r2_s2 = _ve_vaddsl_vsv(4*6*gOutHeight*gOutWidth, vrgout_ptr_k0_r2_s2) ;
      __vr vrgout_k6_r2_s2 = _ve_vgtu_vvm(vrgout_ptr_k6_r2_s2, vmall_r2s2) ;
      __vr vrgout_ptr_k7_r2_s2 = _ve_vaddsl_vsv(4*7*gOutHeight*gOutWidth, vrgout_ptr_k0_r2_s2) ;
      __vr vrgout_k7_r2_s2 = _ve_vgtu_vvm(vrgout_ptr_k7_r2_s2, vmall_r2s2) ;

      VFADD_C2(vrgout_k0_r2_s2, vmall_r2s2, 0, 2, 2)
      VFADD_C2(vrgout_k1_r2_s2, vmall_r2s2, 1, 2, 2)
      VFADD_C2(vrgout_k2_r2_s2, vmall_r2s2, 2, 2, 2)
      VFADD_C2(vrgout_k3_r2_s2, vmall_r2s2, 3, 2, 2)
      VFADD_C2(vrgout_k4_r2_s2, vmall_r2s2, 4, 2, 2)
      VFADD_C2(vrgout_k5_r2_s2, vmall_r2s2, 5, 2, 2)
      VFADD_C2(vrgout_k6_r2_s2, vmall_r2s2, 6, 2, 2)
      VFADD_C2(vrgout_k7_r2_s2, vmall_r2s2, 7, 2, 2)
#undef VFADD_C2

    } // gOutChannel

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
    __vr vrsum01 = _ve_pvbrd_vs_i64(0UL) ;
    __vr vrsum23 = _ve_pvbrd_vs_i64(0UL) ;

    __vr vri_r0 = _ve_vaddsl_vsv(padHeight-0*dilationHeight+h, vrh) ;
    __vr vri_r1 = _ve_vaddsl_vsv(padHeight-1*dilationHeight+h, vrh) ;
    __vr vri_r2 = _ve_vaddsl_vsv(padHeight-2*dilationHeight+h, vrh) ;

    __vr vry_r0 = _ve_vdivsl_vvs(vri_r0, strideHeight) ;
    __vr vry_r1 = _ve_vdivsl_vvs(vri_r1, strideHeight) ;
    __vr vry_r2 = _ve_vdivsl_vvs(vri_r2, strideHeight) ;

    __vr vrj_s0 = _ve_vaddsl_vsv(padWidth-0*dilationWidth, vrw) ;
    __vr vrj_s1 = _ve_vaddsl_vsv(padWidth-1*dilationWidth, vrw) ;
    __vr vrj_s2 = _ve_vaddsl_vsv(padWidth-2*dilationWidth, vrw) ;

    __vr vrx_s0 = _ve_vdivsl_vvs(vrj_s0, strideWidth) ;
    __vr vrx_s1 = _ve_vdivsl_vvs(vrj_s1, strideWidth) ;
    __vr vrx_s2 = _ve_vdivsl_vvs(vrj_s2, strideWidth) ;


    __vm256 vmy0_r0 = _ve_vfmkl_mcv(VECC_IEQ, _ve_vcmpsl_vvv(vri_r0, _ve_vmulsl_vsv(strideHeight, vry_r0))) ;
    __vm256 vmy1_r0 = _ve_vfmkl_mcv(VECC_GE, vry_r0) ;
    __vm256 vmy2_r0 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(gOutHeight,vry_r0)) ;
    __vm256 vmy_r0 = _ve_andm_mmm(_ve_andm_mmm(vmy0_r0, vmy1_r0), vmy2_r0) ;

    __vm256 vmy0_r1 = _ve_vfmkl_mcv(VECC_IEQ, _ve_vcmpsl_vvv(vri_r1, _ve_vmulsl_vsv(strideHeight, vry_r1))) ;
    __vm256 vmy1_r1 = _ve_vfmkl_mcv(VECC_GE, vry_r1) ;
    __vm256 vmy2_r1 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(gOutHeight,vry_r1)) ;
    __vm256 vmy_r1 = _ve_andm_mmm(_ve_andm_mmm(vmy0_r1, vmy1_r1), vmy2_r1) ;

    __vm256 vmy0_r2 = _ve_vfmkl_mcv(VECC_IEQ, _ve_vcmpsl_vvv(vri_r2, _ve_vmulsl_vsv(strideHeight, vry_r2))) ;
    __vm256 vmy1_r2 = _ve_vfmkl_mcv(VECC_GE, vry_r2) ;
    __vm256 vmy2_r2 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(gOutHeight,vry_r2)) ;
    __vm256 vmy_r2 = _ve_andm_mmm(_ve_andm_mmm(vmy0_r2, vmy1_r2), vmy2_r2) ;

    __vm256 vmx0_s0 = _ve_vfmkl_mcv(VECC_IEQ, _ve_vcmpsl_vvv(vrj_s0, _ve_vmulsl_vsv(strideWidth, vrx_s0))) ;
    __vm256 vmx1_s0 = _ve_vfmkl_mcv(VECC_GE, vrx_s0) ;
    __vm256 vmx2_s0 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(gOutWidth,vrx_s0)) ;
    __vm256 vmx_s0 = _ve_andm_mmm(_ve_andm_mmm(vmx0_s0, vmx1_s0), vmx2_s0) ;

    __vm256 vmx0_s1 = _ve_vfmkl_mcv(VECC_IEQ, _ve_vcmpsl_vvv(vrj_s1, _ve_vmulsl_vsv(strideWidth, vrx_s1))) ;
    __vm256 vmx1_s1 = _ve_vfmkl_mcv(VECC_GE, vrx_s1) ;
    __vm256 vmx2_s1 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(gOutWidth,vrx_s1)) ;
    __vm256 vmx_s1 = _ve_andm_mmm(_ve_andm_mmm(vmx0_s1, vmx1_s1), vmx2_s1) ;

    __vm256 vmx0_s2 = _ve_vfmkl_mcv(VECC_IEQ, _ve_vcmpsl_vvv(vrj_s2, _ve_vmulsl_vsv(strideWidth, vrx_s2))) ;
    __vm256 vmx1_s2 = _ve_vfmkl_mcv(VECC_GE, vrx_s2) ;
    __vm256 vmx2_s2 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(gOutWidth,vrx_s2)) ;
    __vm256 vmx_s2 = _ve_andm_mmm(_ve_andm_mmm(vmx0_s2, vmx1_s2), vmx2_s2) ;

    __vm256 vmall_r0s0 = _ve_andm_mmm(vmy_r0,vmx_s0) ;
    __vm256 vmall_r0s1 = _ve_andm_mmm(vmy_r0,vmx_s1) ;
    __vm256 vmall_r0s2 = _ve_andm_mmm(vmy_r0,vmx_s2) ;

    __vm256 vmall_r1s0 = _ve_andm_mmm(vmy_r1,vmx_s0) ;
    __vm256 vmall_r1s1 = _ve_andm_mmm(vmy_r1,vmx_s1) ;
    __vm256 vmall_r1s2 = _ve_andm_mmm(vmy_r1,vmx_s2) ;

    __vm256 vmall_r2s0 = _ve_andm_mmm(vmy_r2,vmx_s0) ;
    __vm256 vmall_r2s1 = _ve_andm_mmm(vmy_r2,vmx_s1) ;
    __vm256 vmall_r2s2 = _ve_andm_mmm(vmy_r2,vmx_s2) ;

    int64_t k=0;
    if( (gOutChannelGroup & 0x01 ) == 1 ) {
      int64_t gOutIndex    = gOutGroupOffset + ((n * gOutChannel + k) * gOutHeight) * gOutWidth ;
      const float *pKerValue = pKernel + kernGroupOffset + (((k  ) * gInChannelGroup + c) * kernHeight ) * kernWidth ;

#define VFADD_C4(VRGOUT, VM, K, R, S)  {												\
	const uint64_t kerValue01 = _ve_pack_f32p(pKerValue + (((K)*gInChannelGroup + 0) * kernHeight +(R)) * kernWidth + (S),		\
						  pKerValue + (((K)*gInChannelGroup + 1) * kernHeight +(R)) * kernWidth + (S)) ;	\
	const uint64_t kerValue23 = _ve_pack_f32p(pKerValue + (((K)*gInChannelGroup + 2) * kernHeight +(R)) * kernWidth + (S),		\
						  pKerValue + (((K)*gInChannelGroup + 3) * kernHeight +(R)) * kernWidth + (S)) ;	\
	VRGOUT = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), VRGOUT, VM) ;			\
	__vr vrgoutP = _ve_vshf_vvvs(VRGOUT, VRGOUT, VE_VSHUFFLE_YUZU) ;			\
	vrsum01 = _ve_pvfmad_vvsv(vrsum01, kerValue01, vrgoutP) ;				\
	vrsum23 = _ve_pvfmad_vvsv(vrsum23, kerValue23, vrgoutP) ;				\
      }

      __vr vrgout_ptr_k0_r0_s0 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry_r0), vrx_s0),
					 2,
					 (unsigned long)(pGOut+gOutIndex)) ;
      __vr vrgout_k0_r0_s0 = _ve_vgtu_vvm(vrgout_ptr_k0_r0_s0, vmall_r0s0) ;
      __vr vrgout_ptr_k0_r0_s1 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry_r0), vrx_s1),
					 2,
					 (unsigned long)(pGOut+gOutIndex)) ;
      __vr vrgout_k0_r0_s1 = _ve_vgtu_vvm(vrgout_ptr_k0_r0_s1, vmall_r0s1) ;
      __vr vrgout_ptr_k0_r0_s2 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry_r0), vrx_s2),
					 2,
					 (unsigned long)(pGOut+gOutIndex)) ;
      __vr vrgout_k0_r0_s2 = _ve_vgtu_vvm(vrgout_ptr_k0_r0_s2, vmall_r0s2) ;

      VFADD_C4(vrgout_k0_r0_s0, vmall_r0s0, 0, 0, 0)
      VFADD_C4(vrgout_k0_r0_s1, vmall_r0s1, 0, 0, 1)
      VFADD_C4(vrgout_k0_r0_s2, vmall_r0s2, 0, 0, 2)

      __vr vrgout_ptr_k0_r1_s0 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry_r1), vrx_s0),
					 2,
					 (unsigned long)(pGOut+gOutIndex)) ;
      __vr vrgout_k0_r1_s0 = _ve_vgtu_vvm(vrgout_ptr_k0_r1_s0, vmall_r1s0) ;
      __vr vrgout_ptr_k0_r1_s1 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry_r1), vrx_s1),
					 2,
					 (unsigned long)(pGOut+gOutIndex)) ;
      __vr vrgout_k0_r1_s1 = _ve_vgtu_vvm(vrgout_ptr_k0_r1_s1, vmall_r1s1) ;
      __vr vrgout_ptr_k0_r1_s2 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry_r1), vrx_s2),
					 2,
					 (unsigned long)(pGOut+gOutIndex)) ;
      __vr vrgout_k0_r1_s2 = _ve_vgtu_vvm(vrgout_ptr_k0_r1_s2, vmall_r1s2) ;

      VFADD_C4(vrgout_k0_r1_s0, vmall_r1s0, 0, 1, 0)
      VFADD_C4(vrgout_k0_r1_s1, vmall_r1s1, 0, 1, 1)
      VFADD_C4(vrgout_k0_r1_s2, vmall_r1s2, 0, 1, 2)

      __vr vrgout_ptr_k0_r2_s0 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry_r2), vrx_s0),
					 2,
					 (unsigned long)(pGOut+gOutIndex)) ;
      __vr vrgout_k0_r2_s0 = _ve_vgtu_vvm(vrgout_ptr_k0_r2_s0, vmall_r2s0) ;
      __vr vrgout_ptr_k0_r2_s1 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry_r2), vrx_s1),
					 2,
					 (unsigned long)(pGOut+gOutIndex)) ;
      __vr vrgout_k0_r2_s1 = _ve_vgtu_vvm(vrgout_ptr_k0_r2_s1, vmall_r2s1) ;
      __vr vrgout_ptr_k0_r2_s2 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry_r2), vrx_s2),
					 2,
					 (unsigned long)(pGOut+gOutIndex)) ;
      __vr vrgout_k0_r2_s2 = _ve_vgtu_vvm(vrgout_ptr_k0_r2_s2, vmall_r2s2) ;

      VFADD_C4(vrgout_k0_r2_s0, vmall_r2s0, 0, 2, 0)
      VFADD_C4(vrgout_k0_r2_s1, vmall_r2s1, 0, 2, 1)
      VFADD_C4(vrgout_k0_r2_s2, vmall_r2s2, 0, 2, 2)

      k+=1 ;
    }
    if( ((gOutChannelGroup >> 1) & 0x01 ) == 1 ) {
      int64_t gOutIndex    = gOutGroupOffset + ((n * gOutChannel + k) * gOutHeight) * gOutWidth ;
      const float *pKerValue = pKernel + kernGroupOffset + (((k  ) * gInChannelGroup + c) * kernHeight) * kernWidth ;

      __vr vrgout_ptr_k0_r0_s0 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry_r0), vrx_s0),
					 2,
					 (unsigned long)(pGOut+gOutIndex)) ;
      __vr vrgout_k0_r0_s0 = _ve_vgtu_vvm(vrgout_ptr_k0_r0_s0, vmall_r0s0) ;
      __vr vrgout_ptr_k1_r0_s0 = _ve_vaddsl_vsv(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_r0_s0) ;
      __vr vrgout_k1_r0_s0 = _ve_vgtu_vvm(vrgout_ptr_k1_r0_s0, vmall_r0s0) ;

      VFADD_C4(vrgout_k0_r0_s0, vmall_r0s0, 0, 0, 0)
      VFADD_C4(vrgout_k1_r0_s0, vmall_r0s0, 1, 0, 0)

      __vr vrgout_ptr_k0_r0_s1 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry_r0), vrx_s1),
					 2,
					 (unsigned long)(pGOut+gOutIndex)) ;
      __vr vrgout_k0_r0_s1 = _ve_vgtu_vvm(vrgout_ptr_k0_r0_s1, vmall_r0s1) ;
      __vr vrgout_ptr_k1_r0_s1 = _ve_vaddsl_vsv(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_r0_s1) ;
      __vr vrgout_k1_r0_s1 = _ve_vgtu_vvm(vrgout_ptr_k1_r0_s1, vmall_r0s1) ;

      VFADD_C4(vrgout_k0_r0_s1, vmall_r0s1, 0, 0, 1)
      VFADD_C4(vrgout_k1_r0_s1, vmall_r0s1, 1, 0, 1)

      __vr vrgout_ptr_k0_r0_s2 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry_r0), vrx_s2),
					 2,
					 (unsigned long)(pGOut+gOutIndex)) ;
      __vr vrgout_k0_r0_s2 = _ve_vgtu_vvm(vrgout_ptr_k0_r0_s2, vmall_r0s2) ;
      __vr vrgout_ptr_k1_r0_s2 = _ve_vaddsl_vsv(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_r0_s2) ;
      __vr vrgout_k1_r0_s2 = _ve_vgtu_vvm(vrgout_ptr_k1_r0_s2, vmall_r0s2) ;

      VFADD_C4(vrgout_k0_r0_s2, vmall_r0s2, 0, 0, 2)
      VFADD_C4(vrgout_k1_r0_s2, vmall_r0s2, 1, 0, 2)

      __vr vrgout_ptr_k0_r1_s0 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry_r1), vrx_s0),
					 2,
					 (unsigned long)(pGOut+gOutIndex)) ;
      __vr vrgout_k0_r1_s0 = _ve_vgtu_vvm(vrgout_ptr_k0_r1_s0, vmall_r1s0) ;
      __vr vrgout_ptr_k1_r1_s0 = _ve_vaddsl_vsv(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_r1_s0) ;
      __vr vrgout_k1_r1_s0 = _ve_vgtu_vvm(vrgout_ptr_k1_r1_s0, vmall_r1s0) ;

      VFADD_C4(vrgout_k0_r1_s0, vmall_r1s0, 0, 1, 0)
      VFADD_C4(vrgout_k1_r1_s0, vmall_r1s0, 1, 1, 0)

      __vr vrgout_ptr_k0_r1_s1 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry_r1), vrx_s1),
					 2,
					 (unsigned long)(pGOut+gOutIndex)) ;
      __vr vrgout_k0_r1_s1 = _ve_vgtu_vvm(vrgout_ptr_k0_r1_s1, vmall_r1s1) ;
      __vr vrgout_ptr_k1_r1_s1 = _ve_vaddsl_vsv(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_r1_s1) ;
      __vr vrgout_k1_r1_s1 = _ve_vgtu_vvm(vrgout_ptr_k1_r1_s1, vmall_r1s1) ;

      VFADD_C4(vrgout_k0_r1_s1, vmall_r1s1, 0, 1, 1)
      VFADD_C4(vrgout_k1_r1_s1, vmall_r1s1, 1, 1, 1)

      __vr vrgout_ptr_k0_r1_s2 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry_r1), vrx_s2),
					 2,
					 (unsigned long)(pGOut+gOutIndex)) ;
      __vr vrgout_k0_r1_s2 = _ve_vgtu_vvm(vrgout_ptr_k0_r1_s2, vmall_r1s2) ;
      __vr vrgout_ptr_k1_r1_s2 = _ve_vaddsl_vsv(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_r1_s2) ;
      __vr vrgout_k1_r1_s2 = _ve_vgtu_vvm(vrgout_ptr_k1_r1_s2, vmall_r1s2) ;

      VFADD_C4(vrgout_k0_r1_s2, vmall_r1s2, 0, 1, 2)
      VFADD_C4(vrgout_k1_r1_s2, vmall_r1s2, 1, 1, 2)

      __vr vrgout_ptr_k0_r2_s0 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry_r2), vrx_s0),
					 2,
					 (unsigned long)(pGOut+gOutIndex)) ;
      __vr vrgout_k0_r2_s0 = _ve_vgtu_vvm(vrgout_ptr_k0_r2_s0, vmall_r2s0) ;
      __vr vrgout_ptr_k1_r2_s0 = _ve_vaddsl_vsv(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_r2_s0) ;
      __vr vrgout_k1_r2_s0 = _ve_vgtu_vvm(vrgout_ptr_k1_r2_s0, vmall_r2s0) ;

      VFADD_C4(vrgout_k0_r2_s0, vmall_r2s0, 0, 2, 0)
      VFADD_C4(vrgout_k1_r2_s0, vmall_r2s0, 1, 2, 0)

      __vr vrgout_ptr_k0_r2_s1 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry_r2), vrx_s1),
					 2,
					 (unsigned long)(pGOut+gOutIndex)) ;
      __vr vrgout_k0_r2_s1 = _ve_vgtu_vvm(vrgout_ptr_k0_r2_s1, vmall_r2s1) ;
      __vr vrgout_ptr_k1_r2_s1 = _ve_vaddsl_vsv(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_r2_s1) ;
      __vr vrgout_k1_r2_s1 = _ve_vgtu_vvm(vrgout_ptr_k1_r2_s1, vmall_r2s1) ;

      VFADD_C4(vrgout_k0_r2_s1, vmall_r2s1, 0, 2, 1)
      VFADD_C4(vrgout_k1_r2_s1, vmall_r2s1, 1, 2, 1)

      __vr vrgout_ptr_k0_r2_s2 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry_r2), vrx_s2),
					 2,
					 (unsigned long)(pGOut+gOutIndex)) ;
      __vr vrgout_k0_r2_s2 = _ve_vgtu_vvm(vrgout_ptr_k0_r2_s2, vmall_r2s2) ;
      __vr vrgout_ptr_k1_r2_s2 = _ve_vaddsl_vsv(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_r2_s2) ;
      __vr vrgout_k1_r2_s2 = _ve_vgtu_vvm(vrgout_ptr_k1_r2_s2, vmall_r2s2) ;

      VFADD_C4(vrgout_k0_r2_s2, vmall_r2s2, 0, 2, 2)
      VFADD_C4(vrgout_k1_r2_s2, vmall_r2s2, 1, 2, 2)

      k+=2 ;
    }
    if( ((gOutChannelGroup >> 2) & 0x01 ) == 1 ) {
      int64_t gOutIndex    = gOutGroupOffset + ((n * gOutChannel + k) * gOutHeight) * gOutWidth ;
      const float *pKerValue = pKernel + kernGroupOffset + (((k  ) * gInChannelGroup + c) * kernHeight) * kernWidth ;

      __vr vrgout_ptr_k0_r0_s0 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry_r0), vrx_s0),
					 2,
					 (unsigned long)(pGOut+gOutIndex)) ;
      __vr vrgout_k0_r0_s0 = _ve_vgtu_vvm(vrgout_ptr_k0_r0_s0, vmall_r0s0) ;
      __vr vrgout_ptr_k1_r0_s0 = _ve_vaddsl_vsv(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_r0_s0) ;
      __vr vrgout_k1_r0_s0 = _ve_vgtu_vvm(vrgout_ptr_k1_r0_s0, vmall_r0s0) ;
      __vr vrgout_ptr_r0_k2_s0 = _ve_vaddsl_vsv(4*2*gOutHeight*gOutWidth, vrgout_ptr_k0_r0_s0) ;
      __vr vrgout_k2_r0_s0 = _ve_vgtu_vvm(vrgout_ptr_r0_k2_s0, vmall_r0s0) ;
      __vr vrgout_ptr_k3_r0_s0 = _ve_vaddsl_vsv(4*3*gOutHeight*gOutWidth, vrgout_ptr_k0_r0_s0) ;
      __vr vrgout_k3_r0_s0 = _ve_vgtu_vvm(vrgout_ptr_k3_r0_s0, vmall_r0s0) ;

      VFADD_C4(vrgout_k0_r0_s0, vmall_r0s0, 0, 0, 0)
      VFADD_C4(vrgout_k1_r0_s0, vmall_r0s0, 1, 0, 0)
      VFADD_C4(vrgout_k2_r0_s0, vmall_r0s0, 2, 0, 0)
      VFADD_C4(vrgout_k3_r0_s0, vmall_r0s0, 3, 0, 0)

      __vr vrgout_ptr_k0_r0_s1 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry_r0), vrx_s1),
					 2,
					 (unsigned long)(pGOut+gOutIndex)) ;
      __vr vrgout_k0_r0_s1 = _ve_vgtu_vvm(vrgout_ptr_k0_r0_s1, vmall_r0s1) ;
      __vr vrgout_ptr_k1_r0_s1 = _ve_vaddsl_vsv(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_r0_s1) ;
      __vr vrgout_k1_r0_s1 = _ve_vgtu_vvm(vrgout_ptr_k1_r0_s1, vmall_r0s1) ;
      __vr vrgout_ptr_k2_r0_s1 = _ve_vaddsl_vsv(4*2*gOutHeight*gOutWidth, vrgout_ptr_k0_r0_s1) ;
      __vr vrgout_k2_r0_s1 = _ve_vgtu_vvm(vrgout_ptr_k2_r0_s1, vmall_r0s1) ;
      __vr vrgout_ptr_k3_r0_s1 = _ve_vaddsl_vsv(4*3*gOutHeight*gOutWidth, vrgout_ptr_k0_r0_s1) ;
      __vr vrgout_k3_r0_s1 = _ve_vgtu_vvm(vrgout_ptr_k3_r0_s1, vmall_r0s1) ;

      VFADD_C4(vrgout_k0_r0_s1, vmall_r0s1, 0, 0, 1)
      VFADD_C4(vrgout_k1_r0_s1, vmall_r0s1, 1, 0, 1)
      VFADD_C4(vrgout_k2_r0_s1, vmall_r0s1, 2, 0, 1)
      VFADD_C4(vrgout_k3_r0_s1, vmall_r0s1, 3, 0, 1)

      __vr vrgout_ptr_k0_r0_s2 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry_r0), vrx_s2),
					 2,
					 (unsigned long)(pGOut+gOutIndex)) ;
      __vr vrgout_k0_r0_s2 = _ve_vgtu_vvm(vrgout_ptr_k0_r0_s2, vmall_r0s2) ;
      __vr vrgout_ptr_k1_r0_s2 = _ve_vaddsl_vsv(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_r0_s2) ;
      __vr vrgout_k1_r0_s2 = _ve_vgtu_vvm(vrgout_ptr_k1_r0_s2, vmall_r0s2) ;
      __vr vrgout_ptr_k2_r0_s2 = _ve_vaddsl_vsv(4*2*gOutHeight*gOutWidth, vrgout_ptr_k0_r0_s2) ;
      __vr vrgout_k2_r0_s2 = _ve_vgtu_vvm(vrgout_ptr_k2_r0_s2, vmall_r0s2) ;
      __vr vrgout_ptr_k3_r0_s2 = _ve_vaddsl_vsv(4*3*gOutHeight*gOutWidth, vrgout_ptr_k0_r0_s2) ;
      __vr vrgout_k3_r0_s2 = _ve_vgtu_vvm(vrgout_ptr_k3_r0_s2, vmall_r0s2) ;

      VFADD_C4(vrgout_k0_r0_s2, vmall_r0s2, 0, 0, 2)
      VFADD_C4(vrgout_k1_r0_s2, vmall_r0s2, 1, 0, 2)
      VFADD_C4(vrgout_k2_r0_s2, vmall_r0s2, 2, 0, 2)
      VFADD_C4(vrgout_k3_r0_s2, vmall_r0s2, 3, 0, 2)

      __vr vrgout_ptr_k0_r1_s0 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry_r1), vrx_s0),
					 2,
					 (unsigned long)(pGOut+gOutIndex)) ;
      __vr vrgout_k0_r1_s0 = _ve_vgtu_vvm(vrgout_ptr_k0_r1_s0, vmall_r1s0) ;
      __vr vrgout_ptr_k1_r1_s0 = _ve_vaddsl_vsv(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_r1_s0) ;
      __vr vrgout_k1_r1_s0 = _ve_vgtu_vvm(vrgout_ptr_k1_r1_s0, vmall_r1s0) ;
      __vr vrgout_ptr_k2_r1_s0 = _ve_vaddsl_vsv(4*2*gOutHeight*gOutWidth, vrgout_ptr_k0_r1_s0) ;
      __vr vrgout_k2_r1_s0 = _ve_vgtu_vvm(vrgout_ptr_k2_r1_s0, vmall_r1s0) ;
      __vr vrgout_ptr_k3_r1_s0 = _ve_vaddsl_vsv(4*3*gOutHeight*gOutWidth, vrgout_ptr_k0_r1_s0) ;
      __vr vrgout_k3_r1_s0 = _ve_vgtu_vvm(vrgout_ptr_k3_r1_s0, vmall_r1s0) ;

      VFADD_C4(vrgout_k0_r1_s0, vmall_r1s0, 0, 1, 0)
      VFADD_C4(vrgout_k1_r1_s0, vmall_r1s0, 1, 1, 0)
      VFADD_C4(vrgout_k2_r1_s0, vmall_r1s0, 2, 1, 0)
      VFADD_C4(vrgout_k3_r1_s0, vmall_r1s0, 3, 1, 0)

      __vr vrgout_ptr_k0_r1_s1 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry_r1), vrx_s1),
					 2,
					 (unsigned long)(pGOut+gOutIndex)) ;
      __vr vrgout_k0_r1_s1 = _ve_vgtu_vvm(vrgout_ptr_k0_r1_s1, vmall_r1s1) ;
      __vr vrgout_ptr_k1_r1_s1 = _ve_vaddsl_vsv(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_r1_s1) ;
      __vr vrgout_k1_r1_s1 = _ve_vgtu_vvm(vrgout_ptr_k1_r1_s1, vmall_r1s1) ;
      __vr vrgout_ptr_k2_r1_s1 = _ve_vaddsl_vsv(4*2*gOutHeight*gOutWidth, vrgout_ptr_k0_r1_s1) ;
      __vr vrgout_k2_r1_s1 = _ve_vgtu_vvm(vrgout_ptr_k2_r1_s1, vmall_r1s1) ;
      __vr vrgout_ptr_k3_r1_s1 = _ve_vaddsl_vsv(4*3*gOutHeight*gOutWidth, vrgout_ptr_k0_r1_s1) ;
      __vr vrgout_k3_r1_s1 = _ve_vgtu_vvm(vrgout_ptr_k3_r1_s1, vmall_r1s1) ;

      VFADD_C4(vrgout_k0_r1_s1, vmall_r1s1, 0, 1, 1)
      VFADD_C4(vrgout_k1_r1_s1, vmall_r1s1, 1, 1, 1)
      VFADD_C4(vrgout_k2_r1_s1, vmall_r1s1, 2, 1, 1)
      VFADD_C4(vrgout_k3_r1_s1, vmall_r1s1, 3, 1, 1)

      __vr vrgout_ptr_k0_r1_s2 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry_r1), vrx_s2),
					 2,
					 (unsigned long)(pGOut+gOutIndex)) ;
      __vr vrgout_k0_r1_s2 = _ve_vgtu_vvm(vrgout_ptr_k0_r1_s2, vmall_r1s2) ;
      __vr vrgout_ptr_k1_r1_s2 = _ve_vaddsl_vsv(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_r1_s2) ;
      __vr vrgout_k1_r1_s2 = _ve_vgtu_vvm(vrgout_ptr_k1_r1_s2, vmall_r1s2) ;
      __vr vrgout_ptr_k2_r1_s2 = _ve_vaddsl_vsv(4*2*gOutHeight*gOutWidth, vrgout_ptr_k0_r1_s2) ;
      __vr vrgout_k2_r1_s2 = _ve_vgtu_vvm(vrgout_ptr_k2_r1_s2, vmall_r1s2) ;
      __vr vrgout_ptr_k3_r1_s2 = _ve_vaddsl_vsv(4*3*gOutHeight*gOutWidth, vrgout_ptr_k0_r1_s2) ;
      __vr vrgout_k3_r1_s2 = _ve_vgtu_vvm(vrgout_ptr_k3_r1_s2, vmall_r1s2) ;

      VFADD_C4(vrgout_k0_r1_s2, vmall_r1s2, 0, 1, 2)
      VFADD_C4(vrgout_k1_r1_s2, vmall_r1s2, 1, 1, 2)
      VFADD_C4(vrgout_k2_r1_s2, vmall_r1s2, 2, 1, 2)
      VFADD_C4(vrgout_k3_r1_s2, vmall_r1s2, 3, 1, 2)

      __vr vrgout_ptr_k0_r2_s0 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry_r2), vrx_s0),
					 2,
					 (unsigned long)(pGOut+gOutIndex)) ;
      __vr vrgout_k0_r2_s0 = _ve_vgtu_vvm(vrgout_ptr_k0_r2_s0, vmall_r2s0) ;
      __vr vrgout_ptr_k1_r2_s0 = _ve_vaddsl_vsv(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_r2_s0) ;
      __vr vrgout_k1_r2_s0 = _ve_vgtu_vvm(vrgout_ptr_k1_r2_s0, vmall_r2s0) ;
      __vr vrgout_ptr_k2_r2_s0 = _ve_vaddsl_vsv(4*2*gOutHeight*gOutWidth, vrgout_ptr_k0_r2_s0) ;
      __vr vrgout_k2_r2_s0 = _ve_vgtu_vvm(vrgout_ptr_k2_r2_s0, vmall_r2s0) ;
      __vr vrgout_ptr_k3_r2_s0 = _ve_vaddsl_vsv(4*3*gOutHeight*gOutWidth, vrgout_ptr_k0_r2_s0) ;
      __vr vrgout_k3_r2_s0 = _ve_vgtu_vvm(vrgout_ptr_k3_r2_s0, vmall_r2s0) ;

      VFADD_C4(vrgout_k0_r2_s0, vmall_r2s0, 0, 2, 0)
      VFADD_C4(vrgout_k1_r2_s0, vmall_r2s0, 1, 2, 0)
      VFADD_C4(vrgout_k2_r2_s0, vmall_r2s0, 2, 2, 0)
      VFADD_C4(vrgout_k3_r2_s0, vmall_r2s0, 3, 2, 0)

      __vr vrgout_ptr_k0_r2_s1 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry_r2), vrx_s1),
					 2,
					 (unsigned long)(pGOut+gOutIndex)) ;
      __vr vrgout_k0_r2_s1 = _ve_vgtu_vvm(vrgout_ptr_k0_r2_s1, vmall_r2s1) ;
      __vr vrgout_ptr_k1_r2_s1 = _ve_vaddsl_vsv(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_r2_s1) ;
      __vr vrgout_k1_r2_s1 = _ve_vgtu_vvm(vrgout_ptr_k1_r2_s1, vmall_r2s1) ;
      __vr vrgout_ptr_k2_r2_s1 = _ve_vaddsl_vsv(4*2*gOutHeight*gOutWidth, vrgout_ptr_k0_r2_s1) ;
      __vr vrgout_k2_r2_s1 = _ve_vgtu_vvm(vrgout_ptr_k2_r2_s1, vmall_r2s1) ;
      __vr vrgout_ptr_k3_r2_s1 = _ve_vaddsl_vsv(4*3*gOutHeight*gOutWidth, vrgout_ptr_k0_r2_s1) ;
      __vr vrgout_k3_r2_s1 = _ve_vgtu_vvm(vrgout_ptr_k3_r2_s1, vmall_r2s1) ;

      VFADD_C4(vrgout_k0_r2_s1, vmall_r2s1, 0, 2, 1)
      VFADD_C4(vrgout_k1_r2_s1, vmall_r2s1, 1, 2, 1)
      VFADD_C4(vrgout_k2_r2_s1, vmall_r2s1, 2, 2, 1)
      VFADD_C4(vrgout_k3_r2_s1, vmall_r2s1, 3, 2, 1)

      __vr vrgout_ptr_k0_r2_s2 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry_r2), vrx_s2),
					 2,
					 (unsigned long)(pGOut+gOutIndex)) ;
      __vr vrgout_k0_r2_s2 = _ve_vgtu_vvm(vrgout_ptr_k0_r2_s2, vmall_r2s2) ;
      __vr vrgout_ptr_k1_r2_s2 = _ve_vaddsl_vsv(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_r2_s2) ;
      __vr vrgout_k1_r2_s2 = _ve_vgtu_vvm(vrgout_ptr_k1_r2_s2, vmall_r2s2) ;
      __vr vrgout_ptr_k2_r2_s2 = _ve_vaddsl_vsv(4*2*gOutHeight*gOutWidth, vrgout_ptr_k0_r2_s2) ;
      __vr vrgout_k2_r2_s2 = _ve_vgtu_vvm(vrgout_ptr_k2_r2_s2, vmall_r2s2) ;
      __vr vrgout_ptr_k3_r2_s2 = _ve_vaddsl_vsv(4*3*gOutHeight*gOutWidth, vrgout_ptr_k0_r2_s2) ;
      __vr vrgout_k3_r2_s2 = _ve_vgtu_vvm(vrgout_ptr_k3_r2_s2, vmall_r2s2) ;

      VFADD_C4(vrgout_k0_r2_s2, vmall_r2s2, 0, 2, 2)
      VFADD_C4(vrgout_k1_r2_s2, vmall_r2s2, 1, 2, 2)
      VFADD_C4(vrgout_k2_r2_s2, vmall_r2s2, 2, 2, 2)
      VFADD_C4(vrgout_k3_r2_s2, vmall_r2s2, 3, 2, 2)

      k+=4 ;
    }
    for (; k<gOutChannelGroup; k+=8) {
      int64_t gOutIndex    = gOutGroupOffset + ((n * gOutChannel + k) * gOutHeight) * gOutWidth ;
      const float *pKerValue = pKernel + kernGroupOffset + (((k  ) * gInChannelGroup + c) * kernHeight) * kernWidth ;

      __vr vrgout_ptr_k0_r0_s0 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry_r0), vrx_s0),
					 2,
					 (unsigned long)(pGOut+gOutIndex)) ;
      __vr vrgout_k0_r0_s0 = _ve_vgtu_vvm(vrgout_ptr_k0_r0_s0, vmall_r0s0) ;
      __vr vrgout_ptr_k1_r0_s0 = _ve_vaddsl_vsv(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_r0_s0) ;
      __vr vrgout_k1_r0_s0 = _ve_vgtu_vvm(vrgout_ptr_k1_r0_s0, vmall_r0s0) ;
      __vr vrgout_ptr_r0_k2_s0 = _ve_vaddsl_vsv(4*2*gOutHeight*gOutWidth, vrgout_ptr_k0_r0_s0) ;
      __vr vrgout_k2_r0_s0 = _ve_vgtu_vvm(vrgout_ptr_r0_k2_s0, vmall_r0s0) ;
      __vr vrgout_ptr_k3_r0_s0 = _ve_vaddsl_vsv(4*3*gOutHeight*gOutWidth, vrgout_ptr_k0_r0_s0) ;
      __vr vrgout_k3_r0_s0 = _ve_vgtu_vvm(vrgout_ptr_k3_r0_s0, vmall_r0s0) ;
      __vr vrgout_ptr_k4_r0_s0 = _ve_vaddsl_vsv(4*4*gOutHeight*gOutWidth, vrgout_ptr_k0_r0_s0) ;
      __vr vrgout_k4_r0_s0 = _ve_vgtu_vvm(vrgout_ptr_k4_r0_s0, vmall_r0s0) ;
      __vr vrgout_ptr_k5_r0_s0 = _ve_vaddsl_vsv(4*5*gOutHeight*gOutWidth, vrgout_ptr_k0_r0_s0) ;
      __vr vrgout_k5_r0_s0 = _ve_vgtu_vvm(vrgout_ptr_k5_r0_s0, vmall_r0s0) ;
      __vr vrgout_ptr_k6_r0_s0 = _ve_vaddsl_vsv(4*6*gOutHeight*gOutWidth, vrgout_ptr_k0_r0_s0) ;
      __vr vrgout_k6_r0_s0 = _ve_vgtu_vvm(vrgout_ptr_k6_r0_s0, vmall_r0s0) ;
      __vr vrgout_ptr_k7_r0_s0 = _ve_vaddsl_vsv(4*7*gOutHeight*gOutWidth, vrgout_ptr_k0_r0_s0) ;
      __vr vrgout_k7_r0_s0 = _ve_vgtu_vvm(vrgout_ptr_k7_r0_s0, vmall_r0s0) ;

      VFADD_C4(vrgout_k0_r0_s0, vmall_r0s0, 0, 0, 0)
      VFADD_C4(vrgout_k1_r0_s0, vmall_r0s0, 1, 0, 0)
      VFADD_C4(vrgout_k2_r0_s0, vmall_r0s0, 2, 0, 0)
      VFADD_C4(vrgout_k3_r0_s0, vmall_r0s0, 3, 0, 0)
      VFADD_C4(vrgout_k4_r0_s0, vmall_r0s0, 4, 0, 0)
      VFADD_C4(vrgout_k5_r0_s0, vmall_r0s0, 5, 0, 0)
      VFADD_C4(vrgout_k6_r0_s0, vmall_r0s0, 6, 0, 0)
      VFADD_C4(vrgout_k7_r0_s0, vmall_r0s0, 7, 0, 0)

      __vr vrgout_ptr_k0_r0_s1 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry_r0), vrx_s1),
					 2,
					 (unsigned long)(pGOut+gOutIndex)) ;
      __vr vrgout_k0_r0_s1 = _ve_vgtu_vvm(vrgout_ptr_k0_r0_s1, vmall_r0s1) ;
      __vr vrgout_ptr_k1_r0_s1 = _ve_vaddsl_vsv(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_r0_s1) ;
      __vr vrgout_k1_r0_s1 = _ve_vgtu_vvm(vrgout_ptr_k1_r0_s1, vmall_r0s1) ;
      __vr vrgout_ptr_k2_r0_s1 = _ve_vaddsl_vsv(4*2*gOutHeight*gOutWidth, vrgout_ptr_k0_r0_s1) ;
      __vr vrgout_k2_r0_s1 = _ve_vgtu_vvm(vrgout_ptr_k2_r0_s1, vmall_r0s1) ;
      __vr vrgout_ptr_k3_r0_s1 = _ve_vaddsl_vsv(4*3*gOutHeight*gOutWidth, vrgout_ptr_k0_r0_s1) ;
      __vr vrgout_k3_r0_s1 = _ve_vgtu_vvm(vrgout_ptr_k3_r0_s1, vmall_r0s1) ;
      __vr vrgout_ptr_k4_r0_s1 = _ve_vaddsl_vsv(4*4*gOutHeight*gOutWidth, vrgout_ptr_k0_r0_s1) ;
      __vr vrgout_k4_r0_s1 = _ve_vgtu_vvm(vrgout_ptr_k4_r0_s1, vmall_r0s1) ;
      __vr vrgout_ptr_k5_r0_s1 = _ve_vaddsl_vsv(4*5*gOutHeight*gOutWidth, vrgout_ptr_k0_r0_s1) ;
      __vr vrgout_k5_r0_s1 = _ve_vgtu_vvm(vrgout_ptr_k5_r0_s1, vmall_r0s1) ;
      __vr vrgout_ptr_k6_r0_s1 = _ve_vaddsl_vsv(4*6*gOutHeight*gOutWidth, vrgout_ptr_k0_r0_s1) ;
      __vr vrgout_k6_r0_s1 = _ve_vgtu_vvm(vrgout_ptr_k6_r0_s1, vmall_r0s1) ;
      __vr vrgout_ptr_k7_r0_s1 = _ve_vaddsl_vsv(4*7*gOutHeight*gOutWidth, vrgout_ptr_k0_r0_s1) ;
      __vr vrgout_k7_r0_s1 = _ve_vgtu_vvm(vrgout_ptr_k7_r0_s1, vmall_r0s1) ;

      VFADD_C4(vrgout_k0_r0_s1, vmall_r0s1, 0, 0, 1)
      VFADD_C4(vrgout_k1_r0_s1, vmall_r0s1, 1, 0, 1)
      VFADD_C4(vrgout_k2_r0_s1, vmall_r0s1, 2, 0, 1)
      VFADD_C4(vrgout_k3_r0_s1, vmall_r0s1, 3, 0, 1)
      VFADD_C4(vrgout_k4_r0_s1, vmall_r0s1, 4, 0, 1)
      VFADD_C4(vrgout_k5_r0_s1, vmall_r0s1, 5, 0, 1)
      VFADD_C4(vrgout_k6_r0_s1, vmall_r0s1, 6, 0, 1)
      VFADD_C4(vrgout_k7_r0_s1, vmall_r0s1, 7, 0, 1)

      __vr vrgout_ptr_k0_r0_s2 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry_r0), vrx_s2),
					 2,
					 (unsigned long)(pGOut+gOutIndex)) ;
      __vr vrgout_k0_r0_s2 = _ve_vgtu_vvm(vrgout_ptr_k0_r0_s2, vmall_r0s2) ;
      __vr vrgout_ptr_k1_r0_s2 = _ve_vaddsl_vsv(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_r0_s2) ;
      __vr vrgout_k1_r0_s2 = _ve_vgtu_vvm(vrgout_ptr_k1_r0_s2, vmall_r0s2) ;
      __vr vrgout_ptr_k2_r0_s2 = _ve_vaddsl_vsv(4*2*gOutHeight*gOutWidth, vrgout_ptr_k0_r0_s2) ;
      __vr vrgout_k2_r0_s2 = _ve_vgtu_vvm(vrgout_ptr_k2_r0_s2, vmall_r0s2) ;
      __vr vrgout_ptr_k3_r0_s2 = _ve_vaddsl_vsv(4*3*gOutHeight*gOutWidth, vrgout_ptr_k0_r0_s2) ;
      __vr vrgout_k3_r0_s2 = _ve_vgtu_vvm(vrgout_ptr_k3_r0_s2, vmall_r0s2) ;
      __vr vrgout_ptr_k4_r0_s2 = _ve_vaddsl_vsv(4*4*gOutHeight*gOutWidth, vrgout_ptr_k0_r0_s2) ;
      __vr vrgout_k4_r0_s2 = _ve_vgtu_vvm(vrgout_ptr_k4_r0_s2, vmall_r0s2) ;
      __vr vrgout_ptr_k5_r0_s2 = _ve_vaddsl_vsv(4*5*gOutHeight*gOutWidth, vrgout_ptr_k0_r0_s2) ;
      __vr vrgout_k5_r0_s2 = _ve_vgtu_vvm(vrgout_ptr_k5_r0_s2, vmall_r0s2) ;
      __vr vrgout_ptr_k6_r0_s2 = _ve_vaddsl_vsv(4*6*gOutHeight*gOutWidth, vrgout_ptr_k0_r0_s2) ;
      __vr vrgout_k6_r0_s2 = _ve_vgtu_vvm(vrgout_ptr_k6_r0_s2, vmall_r0s2) ;
      __vr vrgout_ptr_k7_r0_s2 = _ve_vaddsl_vsv(4*7*gOutHeight*gOutWidth, vrgout_ptr_k0_r0_s2) ;
      __vr vrgout_k7_r0_s2 = _ve_vgtu_vvm(vrgout_ptr_k7_r0_s2, vmall_r0s2) ;

      VFADD_C4(vrgout_k0_r0_s2, vmall_r0s2, 0, 0, 2)
      VFADD_C4(vrgout_k1_r0_s2, vmall_r0s2, 1, 0, 2)
      VFADD_C4(vrgout_k2_r0_s2, vmall_r0s2, 2, 0, 2)
      VFADD_C4(vrgout_k3_r0_s2, vmall_r0s2, 3, 0, 2)
      VFADD_C4(vrgout_k4_r0_s2, vmall_r0s2, 4, 0, 2)
      VFADD_C4(vrgout_k5_r0_s2, vmall_r0s2, 5, 0, 2)
      VFADD_C4(vrgout_k6_r0_s2, vmall_r0s2, 6, 0, 2)
      VFADD_C4(vrgout_k7_r0_s2, vmall_r0s2, 7, 0, 2)

      __vr vrgout_ptr_k0_r1_s0 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry_r1), vrx_s0),
					 2,
					 (unsigned long)(pGOut+gOutIndex)) ;
      __vr vrgout_k0_r1_s0 = _ve_vgtu_vvm(vrgout_ptr_k0_r1_s0, vmall_r1s0) ;
      __vr vrgout_ptr_k1_r1_s0 = _ve_vaddsl_vsv(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_r1_s0) ;
      __vr vrgout_k1_r1_s0 = _ve_vgtu_vvm(vrgout_ptr_k1_r1_s0, vmall_r1s0) ;
      __vr vrgout_ptr_k2_r1_s0 = _ve_vaddsl_vsv(4*2*gOutHeight*gOutWidth, vrgout_ptr_k0_r1_s0) ;
      __vr vrgout_k2_r1_s0 = _ve_vgtu_vvm(vrgout_ptr_k2_r1_s0, vmall_r1s0) ;
      __vr vrgout_ptr_k3_r1_s0 = _ve_vaddsl_vsv(4*3*gOutHeight*gOutWidth, vrgout_ptr_k0_r1_s0) ;
      __vr vrgout_k3_r1_s0 = _ve_vgtu_vvm(vrgout_ptr_k3_r1_s0, vmall_r1s0) ;
      __vr vrgout_ptr_k4_r1_s0 = _ve_vaddsl_vsv(4*4*gOutHeight*gOutWidth, vrgout_ptr_k0_r1_s0) ;
      __vr vrgout_k4_r1_s0 = _ve_vgtu_vvm(vrgout_ptr_k4_r1_s0, vmall_r1s0) ;
      __vr vrgout_ptr_k5_r1_s0 = _ve_vaddsl_vsv(4*5*gOutHeight*gOutWidth, vrgout_ptr_k0_r1_s0) ;
      __vr vrgout_k5_r1_s0 = _ve_vgtu_vvm(vrgout_ptr_k5_r1_s0, vmall_r1s0) ;
      __vr vrgout_ptr_k6_r1_s0 = _ve_vaddsl_vsv(4*6*gOutHeight*gOutWidth, vrgout_ptr_k0_r1_s0) ;
      __vr vrgout_k6_r1_s0 = _ve_vgtu_vvm(vrgout_ptr_k6_r1_s0, vmall_r1s0) ;
      __vr vrgout_ptr_k7_r1_s0 = _ve_vaddsl_vsv(4*7*gOutHeight*gOutWidth, vrgout_ptr_k0_r1_s0) ;
      __vr vrgout_k7_r1_s0 = _ve_vgtu_vvm(vrgout_ptr_k7_r1_s0, vmall_r1s0) ;

      VFADD_C4(vrgout_k0_r1_s0, vmall_r1s0, 0, 1, 0)
      VFADD_C4(vrgout_k1_r1_s0, vmall_r1s0, 1, 1, 0)
      VFADD_C4(vrgout_k2_r1_s0, vmall_r1s0, 2, 1, 0)
      VFADD_C4(vrgout_k3_r1_s0, vmall_r1s0, 3, 1, 0)
      VFADD_C4(vrgout_k4_r1_s0, vmall_r1s0, 4, 1, 0)
      VFADD_C4(vrgout_k5_r1_s0, vmall_r1s0, 5, 1, 0)
      VFADD_C4(vrgout_k6_r1_s0, vmall_r1s0, 6, 1, 0)
      VFADD_C4(vrgout_k7_r1_s0, vmall_r1s0, 7, 1, 0)

      __vr vrgout_ptr_k0_r1_s1 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry_r1), vrx_s1),
					 2,
					 (unsigned long)(pGOut+gOutIndex)) ;
      __vr vrgout_k0_r1_s1 = _ve_vgtu_vvm(vrgout_ptr_k0_r1_s1, vmall_r1s1) ;
      __vr vrgout_ptr_k1_r1_s1 = _ve_vaddsl_vsv(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_r1_s1) ;
      __vr vrgout_k1_r1_s1 = _ve_vgtu_vvm(vrgout_ptr_k1_r1_s1, vmall_r1s1) ;
      __vr vrgout_ptr_k2_r1_s1 = _ve_vaddsl_vsv(4*2*gOutHeight*gOutWidth, vrgout_ptr_k0_r1_s1) ;
      __vr vrgout_k2_r1_s1 = _ve_vgtu_vvm(vrgout_ptr_k2_r1_s1, vmall_r1s1) ;
      __vr vrgout_ptr_k3_r1_s1 = _ve_vaddsl_vsv(4*3*gOutHeight*gOutWidth, vrgout_ptr_k0_r1_s1) ;
      __vr vrgout_k3_r1_s1 = _ve_vgtu_vvm(vrgout_ptr_k3_r1_s1, vmall_r1s1) ;
      __vr vrgout_ptr_k4_r1_s1 = _ve_vaddsl_vsv(4*4*gOutHeight*gOutWidth, vrgout_ptr_k0_r1_s1) ;
      __vr vrgout_k4_r1_s1 = _ve_vgtu_vvm(vrgout_ptr_k4_r1_s1, vmall_r1s1) ;
      __vr vrgout_ptr_k5_r1_s1 = _ve_vaddsl_vsv(4*5*gOutHeight*gOutWidth, vrgout_ptr_k0_r1_s1) ;
      __vr vrgout_k5_r1_s1 = _ve_vgtu_vvm(vrgout_ptr_k5_r1_s1, vmall_r1s1) ;
      __vr vrgout_ptr_k6_r1_s1 = _ve_vaddsl_vsv(4*6*gOutHeight*gOutWidth, vrgout_ptr_k0_r1_s1) ;
      __vr vrgout_k6_r1_s1 = _ve_vgtu_vvm(vrgout_ptr_k6_r1_s1, vmall_r1s1) ;
      __vr vrgout_ptr_k7_r1_s1 = _ve_vaddsl_vsv(4*7*gOutHeight*gOutWidth, vrgout_ptr_k0_r1_s1) ;
      __vr vrgout_k7_r1_s1 = _ve_vgtu_vvm(vrgout_ptr_k7_r1_s1, vmall_r1s1) ;

      VFADD_C4(vrgout_k0_r1_s1, vmall_r1s1, 0, 1, 1)
      VFADD_C4(vrgout_k1_r1_s1, vmall_r1s1, 1, 1, 1)
      VFADD_C4(vrgout_k2_r1_s1, vmall_r1s1, 2, 1, 1)
      VFADD_C4(vrgout_k3_r1_s1, vmall_r1s1, 3, 1, 1)
      VFADD_C4(vrgout_k4_r1_s1, vmall_r1s1, 4, 1, 1)
      VFADD_C4(vrgout_k5_r1_s1, vmall_r1s1, 5, 1, 1)
      VFADD_C4(vrgout_k6_r1_s1, vmall_r1s1, 6, 1, 1)
      VFADD_C4(vrgout_k7_r1_s1, vmall_r1s1, 7, 1, 1)

      __vr vrgout_ptr_k0_r1_s2 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry_r1), vrx_s2),
					 2,
					 (unsigned long)(pGOut+gOutIndex)) ;
      __vr vrgout_k0_r1_s2 = _ve_vgtu_vvm(vrgout_ptr_k0_r1_s2, vmall_r1s2) ;
      __vr vrgout_ptr_k1_r1_s2 = _ve_vaddsl_vsv(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_r1_s2) ;
      __vr vrgout_k1_r1_s2 = _ve_vgtu_vvm(vrgout_ptr_k1_r1_s2, vmall_r1s2) ;
      __vr vrgout_ptr_k2_r1_s2 = _ve_vaddsl_vsv(4*2*gOutHeight*gOutWidth, vrgout_ptr_k0_r1_s2) ;
      __vr vrgout_k2_r1_s2 = _ve_vgtu_vvm(vrgout_ptr_k2_r1_s2, vmall_r1s2) ;
      __vr vrgout_ptr_k3_r1_s2 = _ve_vaddsl_vsv(4*3*gOutHeight*gOutWidth, vrgout_ptr_k0_r1_s2) ;
      __vr vrgout_k3_r1_s2 = _ve_vgtu_vvm(vrgout_ptr_k3_r1_s2, vmall_r1s2) ;
      __vr vrgout_ptr_k4_r1_s2 = _ve_vaddsl_vsv(4*4*gOutHeight*gOutWidth, vrgout_ptr_k0_r1_s2) ;
      __vr vrgout_k4_r1_s2 = _ve_vgtu_vvm(vrgout_ptr_k4_r1_s2, vmall_r1s2) ;
      __vr vrgout_ptr_k5_r1_s2 = _ve_vaddsl_vsv(4*5*gOutHeight*gOutWidth, vrgout_ptr_k0_r1_s2) ;
      __vr vrgout_k5_r1_s2 = _ve_vgtu_vvm(vrgout_ptr_k5_r1_s2, vmall_r1s2) ;
      __vr vrgout_ptr_k6_r1_s2 = _ve_vaddsl_vsv(4*6*gOutHeight*gOutWidth, vrgout_ptr_k0_r1_s2) ;
      __vr vrgout_k6_r1_s2 = _ve_vgtu_vvm(vrgout_ptr_k6_r1_s2, vmall_r1s2) ;
      __vr vrgout_ptr_k7_r1_s2 = _ve_vaddsl_vsv(4*7*gOutHeight*gOutWidth, vrgout_ptr_k0_r1_s2) ;
      __vr vrgout_k7_r1_s2 = _ve_vgtu_vvm(vrgout_ptr_k7_r1_s2, vmall_r1s2) ;

      VFADD_C4(vrgout_k0_r1_s2, vmall_r1s2, 0, 1, 2)
      VFADD_C4(vrgout_k1_r1_s2, vmall_r1s2, 1, 1, 2)
      VFADD_C4(vrgout_k2_r1_s2, vmall_r1s2, 2, 1, 2)
      VFADD_C4(vrgout_k3_r1_s2, vmall_r1s2, 3, 1, 2)
      VFADD_C4(vrgout_k4_r1_s2, vmall_r1s2, 4, 1, 2)
      VFADD_C4(vrgout_k5_r1_s2, vmall_r1s2, 5, 1, 2)
      VFADD_C4(vrgout_k6_r1_s2, vmall_r1s2, 6, 1, 2)
      VFADD_C4(vrgout_k7_r1_s2, vmall_r1s2, 7, 1, 2)

      __vr vrgout_ptr_k0_r2_s0 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry_r2), vrx_s0),
					 2,
					 (unsigned long)(pGOut+gOutIndex)) ;
      __vr vrgout_k0_r2_s0 = _ve_vgtu_vvm(vrgout_ptr_k0_r2_s0, vmall_r2s0) ;
      __vr vrgout_ptr_k1_r2_s0 = _ve_vaddsl_vsv(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_r2_s0) ;
      __vr vrgout_k1_r2_s0 = _ve_vgtu_vvm(vrgout_ptr_k1_r2_s0, vmall_r2s0) ;
      __vr vrgout_ptr_k2_r2_s0 = _ve_vaddsl_vsv(4*2*gOutHeight*gOutWidth, vrgout_ptr_k0_r2_s0) ;
      __vr vrgout_k2_r2_s0 = _ve_vgtu_vvm(vrgout_ptr_k2_r2_s0, vmall_r2s0) ;
      __vr vrgout_ptr_k3_r2_s0 = _ve_vaddsl_vsv(4*3*gOutHeight*gOutWidth, vrgout_ptr_k0_r2_s0) ;
      __vr vrgout_k3_r2_s0 = _ve_vgtu_vvm(vrgout_ptr_k3_r2_s0, vmall_r2s0) ;
      __vr vrgout_ptr_k4_r2_s0 = _ve_vaddsl_vsv(4*4*gOutHeight*gOutWidth, vrgout_ptr_k0_r2_s0) ;
      __vr vrgout_k4_r2_s0 = _ve_vgtu_vvm(vrgout_ptr_k4_r2_s0, vmall_r2s0) ;
      __vr vrgout_ptr_k5_r2_s0 = _ve_vaddsl_vsv(4*5*gOutHeight*gOutWidth, vrgout_ptr_k0_r2_s0) ;
      __vr vrgout_k5_r2_s0 = _ve_vgtu_vvm(vrgout_ptr_k5_r2_s0, vmall_r2s0) ;
      __vr vrgout_ptr_k6_r2_s0 = _ve_vaddsl_vsv(4*6*gOutHeight*gOutWidth, vrgout_ptr_k0_r2_s0) ;
      __vr vrgout_k6_r2_s0 = _ve_vgtu_vvm(vrgout_ptr_k6_r2_s0, vmall_r2s0) ;
      __vr vrgout_ptr_k7_r2_s0 = _ve_vaddsl_vsv(4*7*gOutHeight*gOutWidth, vrgout_ptr_k0_r2_s0) ;
      __vr vrgout_k7_r2_s0 = _ve_vgtu_vvm(vrgout_ptr_k7_r2_s0, vmall_r2s0) ;

      VFADD_C4(vrgout_k0_r2_s0, vmall_r2s0, 0, 2, 0)
      VFADD_C4(vrgout_k1_r2_s0, vmall_r2s0, 1, 2, 0)
      VFADD_C4(vrgout_k2_r2_s0, vmall_r2s0, 2, 2, 0)
      VFADD_C4(vrgout_k3_r2_s0, vmall_r2s0, 3, 2, 0)
      VFADD_C4(vrgout_k4_r2_s0, vmall_r2s0, 4, 2, 0)
      VFADD_C4(vrgout_k5_r2_s0, vmall_r2s0, 5, 2, 0)
      VFADD_C4(vrgout_k6_r2_s0, vmall_r2s0, 6, 2, 0)
      VFADD_C4(vrgout_k7_r2_s0, vmall_r2s0, 7, 2, 0)

      __vr vrgout_ptr_k0_r2_s1 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry_r2), vrx_s1),
					 2,
					 (unsigned long)(pGOut+gOutIndex)) ;
      __vr vrgout_k0_r2_s1 = _ve_vgtu_vvm(vrgout_ptr_k0_r2_s1, vmall_r2s1) ;
      __vr vrgout_ptr_k1_r2_s1 = _ve_vaddsl_vsv(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_r2_s1) ;
      __vr vrgout_k1_r2_s1 = _ve_vgtu_vvm(vrgout_ptr_k1_r2_s1, vmall_r2s1) ;
      __vr vrgout_ptr_k2_r2_s1 = _ve_vaddsl_vsv(4*2*gOutHeight*gOutWidth, vrgout_ptr_k0_r2_s1) ;
      __vr vrgout_k2_r2_s1 = _ve_vgtu_vvm(vrgout_ptr_k2_r2_s1, vmall_r2s1) ;
      __vr vrgout_ptr_k3_r2_s1 = _ve_vaddsl_vsv(4*3*gOutHeight*gOutWidth, vrgout_ptr_k0_r2_s1) ;
      __vr vrgout_k3_r2_s1 = _ve_vgtu_vvm(vrgout_ptr_k3_r2_s1, vmall_r2s1) ;
      __vr vrgout_ptr_k4_r2_s1 = _ve_vaddsl_vsv(4*4*gOutHeight*gOutWidth, vrgout_ptr_k0_r2_s1) ;
      __vr vrgout_k4_r2_s1 = _ve_vgtu_vvm(vrgout_ptr_k4_r2_s1, vmall_r2s1) ;
      __vr vrgout_ptr_k5_r2_s1 = _ve_vaddsl_vsv(4*5*gOutHeight*gOutWidth, vrgout_ptr_k0_r2_s1) ;
      __vr vrgout_k5_r2_s1 = _ve_vgtu_vvm(vrgout_ptr_k5_r2_s1, vmall_r2s1) ;
      __vr vrgout_ptr_k6_r2_s1 = _ve_vaddsl_vsv(4*6*gOutHeight*gOutWidth, vrgout_ptr_k0_r2_s1) ;
      __vr vrgout_k6_r2_s1 = _ve_vgtu_vvm(vrgout_ptr_k6_r2_s1, vmall_r2s1) ;
      __vr vrgout_ptr_k7_r2_s1 = _ve_vaddsl_vsv(4*7*gOutHeight*gOutWidth, vrgout_ptr_k0_r2_s1) ;
      __vr vrgout_k7_r2_s1 = _ve_vgtu_vvm(vrgout_ptr_k7_r2_s1, vmall_r2s1) ;

      VFADD_C4(vrgout_k0_r2_s1, vmall_r2s1, 0, 2, 1)
      VFADD_C4(vrgout_k1_r2_s1, vmall_r2s1, 1, 2, 1)
      VFADD_C4(vrgout_k2_r2_s1, vmall_r2s1, 2, 2, 1)
      VFADD_C4(vrgout_k3_r2_s1, vmall_r2s1, 3, 2, 1)
      VFADD_C4(vrgout_k4_r2_s1, vmall_r2s1, 4, 2, 1)
      VFADD_C4(vrgout_k5_r2_s1, vmall_r2s1, 5, 2, 1)
      VFADD_C4(vrgout_k6_r2_s1, vmall_r2s1, 6, 2, 1)
      VFADD_C4(vrgout_k7_r2_s1, vmall_r2s1, 7, 2, 1)

      __vr vrgout_ptr_k0_r2_s2 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry_r2), vrx_s2),
					 2,
					 (unsigned long)(pGOut+gOutIndex)) ;
      __vr vrgout_k0_r2_s2 = _ve_vgtu_vvm(vrgout_ptr_k0_r2_s2, vmall_r2s2) ;
      __vr vrgout_ptr_k1_r2_s2 = _ve_vaddsl_vsv(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_r2_s2) ;
      __vr vrgout_k1_r2_s2 = _ve_vgtu_vvm(vrgout_ptr_k1_r2_s2, vmall_r2s2) ;
      __vr vrgout_ptr_k2_r2_s2 = _ve_vaddsl_vsv(4*2*gOutHeight*gOutWidth, vrgout_ptr_k0_r2_s2) ;
      __vr vrgout_k2_r2_s2 = _ve_vgtu_vvm(vrgout_ptr_k2_r2_s2, vmall_r2s2) ;
      __vr vrgout_ptr_k3_r2_s2 = _ve_vaddsl_vsv(4*3*gOutHeight*gOutWidth, vrgout_ptr_k0_r2_s2) ;
      __vr vrgout_k3_r2_s2 = _ve_vgtu_vvm(vrgout_ptr_k3_r2_s2, vmall_r2s2) ;
      __vr vrgout_ptr_k4_r2_s2 = _ve_vaddsl_vsv(4*4*gOutHeight*gOutWidth, vrgout_ptr_k0_r2_s2) ;
      __vr vrgout_k4_r2_s2 = _ve_vgtu_vvm(vrgout_ptr_k4_r2_s2, vmall_r2s2) ;
      __vr vrgout_ptr_k5_r2_s2 = _ve_vaddsl_vsv(4*5*gOutHeight*gOutWidth, vrgout_ptr_k0_r2_s2) ;
      __vr vrgout_k5_r2_s2 = _ve_vgtu_vvm(vrgout_ptr_k5_r2_s2, vmall_r2s2) ;
      __vr vrgout_ptr_k6_r2_s2 = _ve_vaddsl_vsv(4*6*gOutHeight*gOutWidth, vrgout_ptr_k0_r2_s2) ;
      __vr vrgout_k6_r2_s2 = _ve_vgtu_vvm(vrgout_ptr_k6_r2_s2, vmall_r2s2) ;
      __vr vrgout_ptr_k7_r2_s2 = _ve_vaddsl_vsv(4*7*gOutHeight*gOutWidth, vrgout_ptr_k0_r2_s2) ;
      __vr vrgout_k7_r2_s2 = _ve_vgtu_vvm(vrgout_ptr_k7_r2_s2, vmall_r2s2) ;

      VFADD_C4(vrgout_k0_r2_s2, vmall_r2s2, 0, 2, 2)
      VFADD_C4(vrgout_k1_r2_s2, vmall_r2s2, 1, 2, 2)
      VFADD_C4(vrgout_k2_r2_s2, vmall_r2s2, 2, 2, 2)
      VFADD_C4(vrgout_k3_r2_s2, vmall_r2s2, 3, 2, 2)
      VFADD_C4(vrgout_k4_r2_s2, vmall_r2s2, 4, 2, 2)
      VFADD_C4(vrgout_k5_r2_s2, vmall_r2s2, 5, 2, 2)
      VFADD_C4(vrgout_k6_r2_s2, vmall_r2s2, 6, 2, 2)
      VFADD_C4(vrgout_k7_r2_s2, vmall_r2s2, 7, 2, 2)
#undef VFADD_C4

    } // gOutChannel

    _ve_vstu_vss(vrsum01, 4, pGIn+gInIndex) ;
    _ve_vstl_vss(vrsum01, 4, pGIn+gInIndex+  gInPixels) ;
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
    __vr vrsum01 = _ve_pvbrd_vs_i64(0UL) ;
    __vr vrsum23 = _ve_pvbrd_vs_i64(0UL) ;
    __vr vrsum45 = _ve_pvbrd_vs_i64(0UL) ;
    __vr vrsum67 = _ve_pvbrd_vs_i64(0UL) ;

    __vr vri_r0 = _ve_vaddsl_vsv(padHeight-0*dilationHeight+h, vrh) ;
    __vr vri_r1 = _ve_vaddsl_vsv(padHeight-1*dilationHeight+h, vrh) ;
    __vr vri_r2 = _ve_vaddsl_vsv(padHeight-2*dilationHeight+h, vrh) ;

    __vr vry_r0 = _ve_vdivsl_vvs(vri_r0, strideHeight) ;
    __vr vry_r1 = _ve_vdivsl_vvs(vri_r1, strideHeight) ;
    __vr vry_r2 = _ve_vdivsl_vvs(vri_r2, strideHeight) ;

    __vr vrj_s0 = _ve_vaddsl_vsv(padWidth-0*dilationWidth, vrw) ;
    __vr vrj_s1 = _ve_vaddsl_vsv(padWidth-1*dilationWidth, vrw) ;
    __vr vrj_s2 = _ve_vaddsl_vsv(padWidth-2*dilationWidth, vrw) ;

    __vr vrx_s0 = _ve_vdivsl_vvs(vrj_s0, strideWidth) ;
    __vr vrx_s1 = _ve_vdivsl_vvs(vrj_s1, strideWidth) ;
    __vr vrx_s2 = _ve_vdivsl_vvs(vrj_s2, strideWidth) ;


    __vm256 vmy0_r0 = _ve_vfmkl_mcv(VECC_IEQ, _ve_vcmpsl_vvv(vri_r0, _ve_vmulsl_vsv(strideHeight, vry_r0))) ;
    __vm256 vmy1_r0 = _ve_vfmkl_mcv(VECC_GE, vry_r0) ;
    __vm256 vmy2_r0 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(gOutHeight,vry_r0)) ;
    __vm256 vmy_r0 = _ve_andm_mmm(_ve_andm_mmm(vmy0_r0, vmy1_r0), vmy2_r0) ;

    __vm256 vmy0_r1 = _ve_vfmkl_mcv(VECC_IEQ, _ve_vcmpsl_vvv(vri_r1, _ve_vmulsl_vsv(strideHeight, vry_r1))) ;
    __vm256 vmy1_r1 = _ve_vfmkl_mcv(VECC_GE, vry_r1) ;
    __vm256 vmy2_r1 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(gOutHeight,vry_r1)) ;
    __vm256 vmy_r1 = _ve_andm_mmm(_ve_andm_mmm(vmy0_r1, vmy1_r1), vmy2_r1) ;

    __vm256 vmy0_r2 = _ve_vfmkl_mcv(VECC_IEQ, _ve_vcmpsl_vvv(vri_r2, _ve_vmulsl_vsv(strideHeight, vry_r2))) ;
    __vm256 vmy1_r2 = _ve_vfmkl_mcv(VECC_GE, vry_r2) ;
    __vm256 vmy2_r2 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(gOutHeight,vry_r2)) ;
    __vm256 vmy_r2 = _ve_andm_mmm(_ve_andm_mmm(vmy0_r2, vmy1_r2), vmy2_r2) ;

    __vm256 vmx0_s0 = _ve_vfmkl_mcv(VECC_IEQ, _ve_vcmpsl_vvv(vrj_s0, _ve_vmulsl_vsv(strideWidth, vrx_s0))) ;
    __vm256 vmx1_s0 = _ve_vfmkl_mcv(VECC_GE, vrx_s0) ;
    __vm256 vmx2_s0 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(gOutWidth,vrx_s0)) ;
    __vm256 vmx_s0 = _ve_andm_mmm(_ve_andm_mmm(vmx0_s0, vmx1_s0), vmx2_s0) ;

    __vm256 vmx0_s1 = _ve_vfmkl_mcv(VECC_IEQ, _ve_vcmpsl_vvv(vrj_s1, _ve_vmulsl_vsv(strideWidth, vrx_s1))) ;
    __vm256 vmx1_s1 = _ve_vfmkl_mcv(VECC_GE, vrx_s1) ;
    __vm256 vmx2_s1 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(gOutWidth,vrx_s1)) ;
    __vm256 vmx_s1 = _ve_andm_mmm(_ve_andm_mmm(vmx0_s1, vmx1_s1), vmx2_s1) ;

    __vm256 vmx0_s2 = _ve_vfmkl_mcv(VECC_IEQ, _ve_vcmpsl_vvv(vrj_s2, _ve_vmulsl_vsv(strideWidth, vrx_s2))) ;
    __vm256 vmx1_s2 = _ve_vfmkl_mcv(VECC_GE, vrx_s2) ;
    __vm256 vmx2_s2 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(gOutWidth,vrx_s2)) ;
    __vm256 vmx_s2 = _ve_andm_mmm(_ve_andm_mmm(vmx0_s2, vmx1_s2), vmx2_s2) ;

    __vm256 vmall_r0s0 = _ve_andm_mmm(vmy_r0,vmx_s0) ;
    __vm256 vmall_r0s1 = _ve_andm_mmm(vmy_r0,vmx_s1) ;
    __vm256 vmall_r0s2 = _ve_andm_mmm(vmy_r0,vmx_s2) ;

    __vm256 vmall_r1s0 = _ve_andm_mmm(vmy_r1,vmx_s0) ;
    __vm256 vmall_r1s1 = _ve_andm_mmm(vmy_r1,vmx_s1) ;
    __vm256 vmall_r1s2 = _ve_andm_mmm(vmy_r1,vmx_s2) ;

    __vm256 vmall_r2s0 = _ve_andm_mmm(vmy_r2,vmx_s0) ;
    __vm256 vmall_r2s1 = _ve_andm_mmm(vmy_r2,vmx_s1) ;
    __vm256 vmall_r2s2 = _ve_andm_mmm(vmy_r2,vmx_s2) ;

    int64_t k=0;
    if( (gOutChannelGroup & 0x01 ) == 1 ) {
      int64_t gOutIndex    = gOutGroupOffset + ((n * gOutChannel + k) * gOutHeight) * gOutWidth ;
      const float *pKerValue = pKernel + kernGroupOffset + (((k  ) * gInChannelGroup + c) * kernHeight ) * kernWidth ;

#define VFADD_C8(VRGOUT, VM, K, R, S)  {												\
	const uint64_t kerValue01 = _ve_pack_f32p(pKerValue + (((K)*gInChannelGroup + 0) * kernHeight +(R)) * kernWidth + (S),		\
						  pKerValue + (((K)*gInChannelGroup + 1) * kernHeight +(R)) * kernWidth + (S)) ;	\
	const uint64_t kerValue23 = _ve_pack_f32p(pKerValue + (((K)*gInChannelGroup + 2) * kernHeight +(R)) * kernWidth + (S),		\
						  pKerValue + (((K)*gInChannelGroup + 3) * kernHeight +(R)) * kernWidth + (S)) ;	\
	const uint64_t kerValue45 = _ve_pack_f32p(pKerValue + (((K)*gInChannelGroup + 4) * kernHeight +(R)) * kernWidth + (S),		\
						  pKerValue + (((K)*gInChannelGroup + 5) * kernHeight +(R)) * kernWidth + (S)) ;	\
	const uint64_t kerValue67 = _ve_pack_f32p(pKerValue + (((K)*gInChannelGroup + 6) * kernHeight +(R)) * kernWidth + (S),		\
						  pKerValue + (((K)*gInChannelGroup + 7) * kernHeight +(R)) * kernWidth + (S)) ;	\
	VRGOUT = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), VRGOUT, VM) ;			\
	__vr vrgoutP = _ve_vshf_vvvs(VRGOUT, VRGOUT, VE_VSHUFFLE_YUZU) ;			\
	vrsum01 = _ve_pvfmad_vvsv(vrsum01, kerValue01, vrgoutP) ;				\
	vrsum23 = _ve_pvfmad_vvsv(vrsum23, kerValue23, vrgoutP) ;				\
	vrsum45 = _ve_pvfmad_vvsv(vrsum45, kerValue45, vrgoutP) ;				\
	vrsum67 = _ve_pvfmad_vvsv(vrsum67, kerValue67, vrgoutP) ;				\
      }

      __vr vrgout_ptr_k0_r0_s0 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry_r0), vrx_s0),
					 2,
					 (unsigned long)(pGOut+gOutIndex)) ;
      __vr vrgout_k0_r0_s0 = _ve_vgtu_vvm(vrgout_ptr_k0_r0_s0, vmall_r0s0) ;
      __vr vrgout_ptr_k0_r0_s1 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry_r0), vrx_s1),
					 2,
					 (unsigned long)(pGOut+gOutIndex)) ;
      __vr vrgout_k0_r0_s1 = _ve_vgtu_vvm(vrgout_ptr_k0_r0_s1, vmall_r0s1) ;
      __vr vrgout_ptr_k0_r0_s2 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry_r0), vrx_s2),
					 2,
					 (unsigned long)(pGOut+gOutIndex)) ;
      __vr vrgout_k0_r0_s2 = _ve_vgtu_vvm(vrgout_ptr_k0_r0_s2, vmall_r0s2) ;

      VFADD_C8(vrgout_k0_r0_s0, vmall_r0s0, 0, 0, 0)
      VFADD_C8(vrgout_k0_r0_s1, vmall_r0s1, 0, 0, 1)
      VFADD_C8(vrgout_k0_r0_s2, vmall_r0s2, 0, 0, 2)

      __vr vrgout_ptr_k0_r1_s0 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry_r1), vrx_s0),
					 2,
					 (unsigned long)(pGOut+gOutIndex)) ;
      __vr vrgout_k0_r1_s0 = _ve_vgtu_vvm(vrgout_ptr_k0_r1_s0, vmall_r1s0) ;
      __vr vrgout_ptr_k0_r1_s1 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry_r1), vrx_s1),
					 2,
					 (unsigned long)(pGOut+gOutIndex)) ;
      __vr vrgout_k0_r1_s1 = _ve_vgtu_vvm(vrgout_ptr_k0_r1_s1, vmall_r1s1) ;
      __vr vrgout_ptr_k0_r1_s2 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry_r1), vrx_s2),
					 2,
					 (unsigned long)(pGOut+gOutIndex)) ;
      __vr vrgout_k0_r1_s2 = _ve_vgtu_vvm(vrgout_ptr_k0_r1_s2, vmall_r1s2) ;

      VFADD_C8(vrgout_k0_r1_s0, vmall_r1s0, 0, 1, 0)
      VFADD_C8(vrgout_k0_r1_s1, vmall_r1s1, 0, 1, 1)
      VFADD_C8(vrgout_k0_r1_s2, vmall_r1s2, 0, 1, 2)

      __vr vrgout_ptr_k0_r2_s0 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry_r2), vrx_s0),
					 2,
					 (unsigned long)(pGOut+gOutIndex)) ;
      __vr vrgout_k0_r2_s0 = _ve_vgtu_vvm(vrgout_ptr_k0_r2_s0, vmall_r2s0) ;
      __vr vrgout_ptr_k0_r2_s1 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry_r2), vrx_s1),
					 2,
					 (unsigned long)(pGOut+gOutIndex)) ;
      __vr vrgout_k0_r2_s1 = _ve_vgtu_vvm(vrgout_ptr_k0_r2_s1, vmall_r2s1) ;
      __vr vrgout_ptr_k0_r2_s2 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry_r2), vrx_s2),
					 2,
					 (unsigned long)(pGOut+gOutIndex)) ;
      __vr vrgout_k0_r2_s2 = _ve_vgtu_vvm(vrgout_ptr_k0_r2_s2, vmall_r2s2) ;

      VFADD_C8(vrgout_k0_r2_s0, vmall_r2s0, 0, 2, 0)
      VFADD_C8(vrgout_k0_r2_s1, vmall_r2s1, 0, 2, 1)
      VFADD_C8(vrgout_k0_r2_s2, vmall_r2s2, 0, 2, 2)

      k+=1 ;
    }
    if( ((gOutChannelGroup >> 1) & 0x01 ) == 1 ) {
      int64_t gOutIndex    = gOutGroupOffset + ((n * gOutChannel + k) * gOutHeight) * gOutWidth ;
      const float *pKerValue = pKernel + kernGroupOffset + (((k  ) * gInChannelGroup + c) * kernHeight) * kernWidth ;

      __vr vrgout_ptr_k0_r0_s0 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry_r0), vrx_s0),
					 2,
					 (unsigned long)(pGOut+gOutIndex)) ;
      __vr vrgout_k0_r0_s0 = _ve_vgtu_vvm(vrgout_ptr_k0_r0_s0, vmall_r0s0) ;
      __vr vrgout_ptr_k1_r0_s0 = _ve_vaddsl_vsv(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_r0_s0) ;
      __vr vrgout_k1_r0_s0 = _ve_vgtu_vvm(vrgout_ptr_k1_r0_s0, vmall_r0s0) ;

      VFADD_C8(vrgout_k0_r0_s0, vmall_r0s0, 0, 0, 0)
      VFADD_C8(vrgout_k1_r0_s0, vmall_r0s0, 1, 0, 0)

      __vr vrgout_ptr_k0_r0_s1 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry_r0), vrx_s1),
					 2,
					 (unsigned long)(pGOut+gOutIndex)) ;
      __vr vrgout_k0_r0_s1 = _ve_vgtu_vvm(vrgout_ptr_k0_r0_s1, vmall_r0s1) ;
      __vr vrgout_ptr_k1_r0_s1 = _ve_vaddsl_vsv(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_r0_s1) ;
      __vr vrgout_k1_r0_s1 = _ve_vgtu_vvm(vrgout_ptr_k1_r0_s1, vmall_r0s1) ;

      VFADD_C8(vrgout_k0_r0_s1, vmall_r0s1, 0, 0, 1)
      VFADD_C8(vrgout_k1_r0_s1, vmall_r0s1, 1, 0, 1)

      __vr vrgout_ptr_k0_r0_s2 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry_r0), vrx_s2),
					 2,
					 (unsigned long)(pGOut+gOutIndex)) ;
      __vr vrgout_k0_r0_s2 = _ve_vgtu_vvm(vrgout_ptr_k0_r0_s2, vmall_r0s2) ;
      __vr vrgout_ptr_k1_r0_s2 = _ve_vaddsl_vsv(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_r0_s2) ;
      __vr vrgout_k1_r0_s2 = _ve_vgtu_vvm(vrgout_ptr_k1_r0_s2, vmall_r0s2) ;

      VFADD_C8(vrgout_k0_r0_s2, vmall_r0s2, 0, 0, 2)
      VFADD_C8(vrgout_k1_r0_s2, vmall_r0s2, 1, 0, 2)

      __vr vrgout_ptr_k0_r1_s0 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry_r1), vrx_s0),
					 2,
					 (unsigned long)(pGOut+gOutIndex)) ;
      __vr vrgout_k0_r1_s0 = _ve_vgtu_vvm(vrgout_ptr_k0_r1_s0, vmall_r1s0) ;
      __vr vrgout_ptr_k1_r1_s0 = _ve_vaddsl_vsv(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_r1_s0) ;
      __vr vrgout_k1_r1_s0 = _ve_vgtu_vvm(vrgout_ptr_k1_r1_s0, vmall_r1s0) ;

      VFADD_C8(vrgout_k0_r1_s0, vmall_r1s0, 0, 1, 0)
      VFADD_C8(vrgout_k1_r1_s0, vmall_r1s0, 1, 1, 0)

      __vr vrgout_ptr_k0_r1_s1 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry_r1), vrx_s1),
					 2,
					 (unsigned long)(pGOut+gOutIndex)) ;
      __vr vrgout_k0_r1_s1 = _ve_vgtu_vvm(vrgout_ptr_k0_r1_s1, vmall_r1s1) ;
      __vr vrgout_ptr_k1_r1_s1 = _ve_vaddsl_vsv(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_r1_s1) ;
      __vr vrgout_k1_r1_s1 = _ve_vgtu_vvm(vrgout_ptr_k1_r1_s1, vmall_r1s1) ;

      VFADD_C8(vrgout_k0_r1_s1, vmall_r1s1, 0, 1, 1)
      VFADD_C8(vrgout_k1_r1_s1, vmall_r1s1, 1, 1, 1)

      __vr vrgout_ptr_k0_r1_s2 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry_r1), vrx_s2),
					 2,
					 (unsigned long)(pGOut+gOutIndex)) ;
      __vr vrgout_k0_r1_s2 = _ve_vgtu_vvm(vrgout_ptr_k0_r1_s2, vmall_r1s2) ;
      __vr vrgout_ptr_k1_r1_s2 = _ve_vaddsl_vsv(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_r1_s2) ;
      __vr vrgout_k1_r1_s2 = _ve_vgtu_vvm(vrgout_ptr_k1_r1_s2, vmall_r1s2) ;

      VFADD_C8(vrgout_k0_r1_s2, vmall_r1s2, 0, 1, 2)
      VFADD_C8(vrgout_k1_r1_s2, vmall_r1s2, 1, 1, 2)

      __vr vrgout_ptr_k0_r2_s0 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry_r2), vrx_s0),
					 2,
					 (unsigned long)(pGOut+gOutIndex)) ;
      __vr vrgout_k0_r2_s0 = _ve_vgtu_vvm(vrgout_ptr_k0_r2_s0, vmall_r2s0) ;
      __vr vrgout_ptr_k1_r2_s0 = _ve_vaddsl_vsv(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_r2_s0) ;
      __vr vrgout_k1_r2_s0 = _ve_vgtu_vvm(vrgout_ptr_k1_r2_s0, vmall_r2s0) ;

      VFADD_C8(vrgout_k0_r2_s0, vmall_r2s0, 0, 2, 0)
      VFADD_C8(vrgout_k1_r2_s0, vmall_r2s0, 1, 2, 0)

      __vr vrgout_ptr_k0_r2_s1 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry_r2), vrx_s1),
					 2,
					 (unsigned long)(pGOut+gOutIndex)) ;
      __vr vrgout_k0_r2_s1 = _ve_vgtu_vvm(vrgout_ptr_k0_r2_s1, vmall_r2s1) ;
      __vr vrgout_ptr_k1_r2_s1 = _ve_vaddsl_vsv(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_r2_s1) ;
      __vr vrgout_k1_r2_s1 = _ve_vgtu_vvm(vrgout_ptr_k1_r2_s1, vmall_r2s1) ;

      VFADD_C8(vrgout_k0_r2_s1, vmall_r2s1, 0, 2, 1)
      VFADD_C8(vrgout_k1_r2_s1, vmall_r2s1, 1, 2, 1)

      __vr vrgout_ptr_k0_r2_s2 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry_r2), vrx_s2),
					 2,
					 (unsigned long)(pGOut+gOutIndex)) ;
      __vr vrgout_k0_r2_s2 = _ve_vgtu_vvm(vrgout_ptr_k0_r2_s2, vmall_r2s2) ;
      __vr vrgout_ptr_k1_r2_s2 = _ve_vaddsl_vsv(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_r2_s2) ;
      __vr vrgout_k1_r2_s2 = _ve_vgtu_vvm(vrgout_ptr_k1_r2_s2, vmall_r2s2) ;

      VFADD_C8(vrgout_k0_r2_s2, vmall_r2s2, 0, 2, 2)
      VFADD_C8(vrgout_k1_r2_s2, vmall_r2s2, 1, 2, 2)

      k+=2 ;
    }
    if( ((gOutChannelGroup >> 2) & 0x01 ) == 1 ) {
      int64_t gOutIndex    = gOutGroupOffset + ((n * gOutChannel + k) * gOutHeight) * gOutWidth ;
      const float *pKerValue = pKernel + kernGroupOffset + (((k  ) * gInChannelGroup + c) * kernHeight) * kernWidth ;

      __vr vrgout_ptr_k0_r0_s0 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry_r0), vrx_s0),
					 2,
					 (unsigned long)(pGOut+gOutIndex)) ;
      __vr vrgout_k0_r0_s0 = _ve_vgtu_vvm(vrgout_ptr_k0_r0_s0, vmall_r0s0) ;
      __vr vrgout_ptr_k1_r0_s0 = _ve_vaddsl_vsv(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_r0_s0) ;
      __vr vrgout_k1_r0_s0 = _ve_vgtu_vvm(vrgout_ptr_k1_r0_s0, vmall_r0s0) ;
      __vr vrgout_ptr_r0_k2_s0 = _ve_vaddsl_vsv(4*2*gOutHeight*gOutWidth, vrgout_ptr_k0_r0_s0) ;
      __vr vrgout_k2_r0_s0 = _ve_vgtu_vvm(vrgout_ptr_r0_k2_s0, vmall_r0s0) ;
      __vr vrgout_ptr_k3_r0_s0 = _ve_vaddsl_vsv(4*3*gOutHeight*gOutWidth, vrgout_ptr_k0_r0_s0) ;
      __vr vrgout_k3_r0_s0 = _ve_vgtu_vvm(vrgout_ptr_k3_r0_s0, vmall_r0s0) ;

      VFADD_C8(vrgout_k0_r0_s0, vmall_r0s0, 0, 0, 0)
      VFADD_C8(vrgout_k1_r0_s0, vmall_r0s0, 1, 0, 0)
      VFADD_C8(vrgout_k2_r0_s0, vmall_r0s0, 2, 0, 0)
      VFADD_C8(vrgout_k3_r0_s0, vmall_r0s0, 3, 0, 0)

      __vr vrgout_ptr_k0_r0_s1 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry_r0), vrx_s1),
					 2,
					 (unsigned long)(pGOut+gOutIndex)) ;
      __vr vrgout_k0_r0_s1 = _ve_vgtu_vvm(vrgout_ptr_k0_r0_s1, vmall_r0s1) ;
      __vr vrgout_ptr_k1_r0_s1 = _ve_vaddsl_vsv(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_r0_s1) ;
      __vr vrgout_k1_r0_s1 = _ve_vgtu_vvm(vrgout_ptr_k1_r0_s1, vmall_r0s1) ;
      __vr vrgout_ptr_k2_r0_s1 = _ve_vaddsl_vsv(4*2*gOutHeight*gOutWidth, vrgout_ptr_k0_r0_s1) ;
      __vr vrgout_k2_r0_s1 = _ve_vgtu_vvm(vrgout_ptr_k2_r0_s1, vmall_r0s1) ;
      __vr vrgout_ptr_k3_r0_s1 = _ve_vaddsl_vsv(4*3*gOutHeight*gOutWidth, vrgout_ptr_k0_r0_s1) ;
      __vr vrgout_k3_r0_s1 = _ve_vgtu_vvm(vrgout_ptr_k3_r0_s1, vmall_r0s1) ;

      VFADD_C8(vrgout_k0_r0_s1, vmall_r0s1, 0, 0, 1)
      VFADD_C8(vrgout_k1_r0_s1, vmall_r0s1, 1, 0, 1)
      VFADD_C8(vrgout_k2_r0_s1, vmall_r0s1, 2, 0, 1)
      VFADD_C8(vrgout_k3_r0_s1, vmall_r0s1, 3, 0, 1)

      __vr vrgout_ptr_k0_r0_s2 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry_r0), vrx_s2),
					 2,
					 (unsigned long)(pGOut+gOutIndex)) ;
      __vr vrgout_k0_r0_s2 = _ve_vgtu_vvm(vrgout_ptr_k0_r0_s2, vmall_r0s2) ;
      __vr vrgout_ptr_k1_r0_s2 = _ve_vaddsl_vsv(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_r0_s2) ;
      __vr vrgout_k1_r0_s2 = _ve_vgtu_vvm(vrgout_ptr_k1_r0_s2, vmall_r0s2) ;
      __vr vrgout_ptr_k2_r0_s2 = _ve_vaddsl_vsv(4*2*gOutHeight*gOutWidth, vrgout_ptr_k0_r0_s2) ;
      __vr vrgout_k2_r0_s2 = _ve_vgtu_vvm(vrgout_ptr_k2_r0_s2, vmall_r0s2) ;
      __vr vrgout_ptr_k3_r0_s2 = _ve_vaddsl_vsv(4*3*gOutHeight*gOutWidth, vrgout_ptr_k0_r0_s2) ;
      __vr vrgout_k3_r0_s2 = _ve_vgtu_vvm(vrgout_ptr_k3_r0_s2, vmall_r0s2) ;

      VFADD_C8(vrgout_k0_r0_s2, vmall_r0s2, 0, 0, 2)
      VFADD_C8(vrgout_k1_r0_s2, vmall_r0s2, 1, 0, 2)
      VFADD_C8(vrgout_k2_r0_s2, vmall_r0s2, 2, 0, 2)
      VFADD_C8(vrgout_k3_r0_s2, vmall_r0s2, 3, 0, 2)

      __vr vrgout_ptr_k0_r1_s0 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry_r1), vrx_s0),
					 2,
					 (unsigned long)(pGOut+gOutIndex)) ;
      __vr vrgout_k0_r1_s0 = _ve_vgtu_vvm(vrgout_ptr_k0_r1_s0, vmall_r1s0) ;
      __vr vrgout_ptr_k1_r1_s0 = _ve_vaddsl_vsv(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_r1_s0) ;
      __vr vrgout_k1_r1_s0 = _ve_vgtu_vvm(vrgout_ptr_k1_r1_s0, vmall_r1s0) ;
      __vr vrgout_ptr_k2_r1_s0 = _ve_vaddsl_vsv(4*2*gOutHeight*gOutWidth, vrgout_ptr_k0_r1_s0) ;
      __vr vrgout_k2_r1_s0 = _ve_vgtu_vvm(vrgout_ptr_k2_r1_s0, vmall_r1s0) ;
      __vr vrgout_ptr_k3_r1_s0 = _ve_vaddsl_vsv(4*3*gOutHeight*gOutWidth, vrgout_ptr_k0_r1_s0) ;
      __vr vrgout_k3_r1_s0 = _ve_vgtu_vvm(vrgout_ptr_k3_r1_s0, vmall_r1s0) ;

      VFADD_C8(vrgout_k0_r1_s0, vmall_r1s0, 0, 1, 0)
      VFADD_C8(vrgout_k1_r1_s0, vmall_r1s0, 1, 1, 0)
      VFADD_C8(vrgout_k2_r1_s0, vmall_r1s0, 2, 1, 0)
      VFADD_C8(vrgout_k3_r1_s0, vmall_r1s0, 3, 1, 0)

      __vr vrgout_ptr_k0_r1_s1 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry_r1), vrx_s1),
					 2,
					 (unsigned long)(pGOut+gOutIndex)) ;
      __vr vrgout_k0_r1_s1 = _ve_vgtu_vvm(vrgout_ptr_k0_r1_s1, vmall_r1s1) ;
      __vr vrgout_ptr_k1_r1_s1 = _ve_vaddsl_vsv(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_r1_s1) ;
      __vr vrgout_k1_r1_s1 = _ve_vgtu_vvm(vrgout_ptr_k1_r1_s1, vmall_r1s1) ;
      __vr vrgout_ptr_k2_r1_s1 = _ve_vaddsl_vsv(4*2*gOutHeight*gOutWidth, vrgout_ptr_k0_r1_s1) ;
      __vr vrgout_k2_r1_s1 = _ve_vgtu_vvm(vrgout_ptr_k2_r1_s1, vmall_r1s1) ;
      __vr vrgout_ptr_k3_r1_s1 = _ve_vaddsl_vsv(4*3*gOutHeight*gOutWidth, vrgout_ptr_k0_r1_s1) ;
      __vr vrgout_k3_r1_s1 = _ve_vgtu_vvm(vrgout_ptr_k3_r1_s1, vmall_r1s1) ;

      VFADD_C8(vrgout_k0_r1_s1, vmall_r1s1, 0, 1, 1)
      VFADD_C8(vrgout_k1_r1_s1, vmall_r1s1, 1, 1, 1)
      VFADD_C8(vrgout_k2_r1_s1, vmall_r1s1, 2, 1, 1)
      VFADD_C8(vrgout_k3_r1_s1, vmall_r1s1, 3, 1, 1)

      __vr vrgout_ptr_k0_r1_s2 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry_r1), vrx_s2),
					 2,
					 (unsigned long)(pGOut+gOutIndex)) ;
      __vr vrgout_k0_r1_s2 = _ve_vgtu_vvm(vrgout_ptr_k0_r1_s2, vmall_r1s2) ;
      __vr vrgout_ptr_k1_r1_s2 = _ve_vaddsl_vsv(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_r1_s2) ;
      __vr vrgout_k1_r1_s2 = _ve_vgtu_vvm(vrgout_ptr_k1_r1_s2, vmall_r1s2) ;
      __vr vrgout_ptr_k2_r1_s2 = _ve_vaddsl_vsv(4*2*gOutHeight*gOutWidth, vrgout_ptr_k0_r1_s2) ;
      __vr vrgout_k2_r1_s2 = _ve_vgtu_vvm(vrgout_ptr_k2_r1_s2, vmall_r1s2) ;
      __vr vrgout_ptr_k3_r1_s2 = _ve_vaddsl_vsv(4*3*gOutHeight*gOutWidth, vrgout_ptr_k0_r1_s2) ;
      __vr vrgout_k3_r1_s2 = _ve_vgtu_vvm(vrgout_ptr_k3_r1_s2, vmall_r1s2) ;

      VFADD_C8(vrgout_k0_r1_s2, vmall_r1s2, 0, 1, 2)
      VFADD_C8(vrgout_k1_r1_s2, vmall_r1s2, 1, 1, 2)
      VFADD_C8(vrgout_k2_r1_s2, vmall_r1s2, 2, 1, 2)
      VFADD_C8(vrgout_k3_r1_s2, vmall_r1s2, 3, 1, 2)

      __vr vrgout_ptr_k0_r2_s0 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry_r2), vrx_s0),
					 2,
					 (unsigned long)(pGOut+gOutIndex)) ;
      __vr vrgout_k0_r2_s0 = _ve_vgtu_vvm(vrgout_ptr_k0_r2_s0, vmall_r2s0) ;
      __vr vrgout_ptr_k1_r2_s0 = _ve_vaddsl_vsv(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_r2_s0) ;
      __vr vrgout_k1_r2_s0 = _ve_vgtu_vvm(vrgout_ptr_k1_r2_s0, vmall_r2s0) ;
      __vr vrgout_ptr_k2_r2_s0 = _ve_vaddsl_vsv(4*2*gOutHeight*gOutWidth, vrgout_ptr_k0_r2_s0) ;
      __vr vrgout_k2_r2_s0 = _ve_vgtu_vvm(vrgout_ptr_k2_r2_s0, vmall_r2s0) ;
      __vr vrgout_ptr_k3_r2_s0 = _ve_vaddsl_vsv(4*3*gOutHeight*gOutWidth, vrgout_ptr_k0_r2_s0) ;
      __vr vrgout_k3_r2_s0 = _ve_vgtu_vvm(vrgout_ptr_k3_r2_s0, vmall_r2s0) ;

      VFADD_C8(vrgout_k0_r2_s0, vmall_r2s0, 0, 2, 0)
      VFADD_C8(vrgout_k1_r2_s0, vmall_r2s0, 1, 2, 0)
      VFADD_C8(vrgout_k2_r2_s0, vmall_r2s0, 2, 2, 0)
      VFADD_C8(vrgout_k3_r2_s0, vmall_r2s0, 3, 2, 0)

      __vr vrgout_ptr_k0_r2_s1 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry_r2), vrx_s1),
					 2,
					 (unsigned long)(pGOut+gOutIndex)) ;
      __vr vrgout_k0_r2_s1 = _ve_vgtu_vvm(vrgout_ptr_k0_r2_s1, vmall_r2s1) ;
      __vr vrgout_ptr_k1_r2_s1 = _ve_vaddsl_vsv(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_r2_s1) ;
      __vr vrgout_k1_r2_s1 = _ve_vgtu_vvm(vrgout_ptr_k1_r2_s1, vmall_r2s1) ;
      __vr vrgout_ptr_k2_r2_s1 = _ve_vaddsl_vsv(4*2*gOutHeight*gOutWidth, vrgout_ptr_k0_r2_s1) ;
      __vr vrgout_k2_r2_s1 = _ve_vgtu_vvm(vrgout_ptr_k2_r2_s1, vmall_r2s1) ;
      __vr vrgout_ptr_k3_r2_s1 = _ve_vaddsl_vsv(4*3*gOutHeight*gOutWidth, vrgout_ptr_k0_r2_s1) ;
      __vr vrgout_k3_r2_s1 = _ve_vgtu_vvm(vrgout_ptr_k3_r2_s1, vmall_r2s1) ;

      VFADD_C8(vrgout_k0_r2_s1, vmall_r2s1, 0, 2, 1)
      VFADD_C8(vrgout_k1_r2_s1, vmall_r2s1, 1, 2, 1)
      VFADD_C8(vrgout_k2_r2_s1, vmall_r2s1, 2, 2, 1)
      VFADD_C8(vrgout_k3_r2_s1, vmall_r2s1, 3, 2, 1)

      __vr vrgout_ptr_k0_r2_s2 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry_r2), vrx_s2),
					 2,
					 (unsigned long)(pGOut+gOutIndex)) ;
      __vr vrgout_k0_r2_s2 = _ve_vgtu_vvm(vrgout_ptr_k0_r2_s2, vmall_r2s2) ;
      __vr vrgout_ptr_k1_r2_s2 = _ve_vaddsl_vsv(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_r2_s2) ;
      __vr vrgout_k1_r2_s2 = _ve_vgtu_vvm(vrgout_ptr_k1_r2_s2, vmall_r2s2) ;
      __vr vrgout_ptr_k2_r2_s2 = _ve_vaddsl_vsv(4*2*gOutHeight*gOutWidth, vrgout_ptr_k0_r2_s2) ;
      __vr vrgout_k2_r2_s2 = _ve_vgtu_vvm(vrgout_ptr_k2_r2_s2, vmall_r2s2) ;
      __vr vrgout_ptr_k3_r2_s2 = _ve_vaddsl_vsv(4*3*gOutHeight*gOutWidth, vrgout_ptr_k0_r2_s2) ;
      __vr vrgout_k3_r2_s2 = _ve_vgtu_vvm(vrgout_ptr_k3_r2_s2, vmall_r2s2) ;

      VFADD_C8(vrgout_k0_r2_s2, vmall_r2s2, 0, 2, 2)
      VFADD_C8(vrgout_k1_r2_s2, vmall_r2s2, 1, 2, 2)
      VFADD_C8(vrgout_k2_r2_s2, vmall_r2s2, 2, 2, 2)
      VFADD_C8(vrgout_k3_r2_s2, vmall_r2s2, 3, 2, 2)

      k+=4 ;
    }
    for (; k<gOutChannelGroup; k+=8) {
      int64_t gOutIndex    = gOutGroupOffset + ((n * gOutChannel + k) * gOutHeight) * gOutWidth ;
      const float *pKerValue = pKernel + kernGroupOffset + (((k  ) * gInChannelGroup + c) * kernHeight) * kernWidth ;

      __vr vrgout_ptr_k0_r0_s0 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry_r0), vrx_s0),
					 2,
					 (unsigned long)(pGOut+gOutIndex)) ;
      __vr vrgout_k0_r0_s0 = _ve_vgtu_vvm(vrgout_ptr_k0_r0_s0, vmall_r0s0) ;
      __vr vrgout_ptr_k1_r0_s0 = _ve_vaddsl_vsv(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_r0_s0) ;
      __vr vrgout_k1_r0_s0 = _ve_vgtu_vvm(vrgout_ptr_k1_r0_s0, vmall_r0s0) ;
      __vr vrgout_ptr_r0_k2_s0 = _ve_vaddsl_vsv(4*2*gOutHeight*gOutWidth, vrgout_ptr_k0_r0_s0) ;
      __vr vrgout_k2_r0_s0 = _ve_vgtu_vvm(vrgout_ptr_r0_k2_s0, vmall_r0s0) ;
      __vr vrgout_ptr_k3_r0_s0 = _ve_vaddsl_vsv(4*3*gOutHeight*gOutWidth, vrgout_ptr_k0_r0_s0) ;
      __vr vrgout_k3_r0_s0 = _ve_vgtu_vvm(vrgout_ptr_k3_r0_s0, vmall_r0s0) ;
      __vr vrgout_ptr_k4_r0_s0 = _ve_vaddsl_vsv(4*4*gOutHeight*gOutWidth, vrgout_ptr_k0_r0_s0) ;
      __vr vrgout_k4_r0_s0 = _ve_vgtu_vvm(vrgout_ptr_k4_r0_s0, vmall_r0s0) ;
      __vr vrgout_ptr_k5_r0_s0 = _ve_vaddsl_vsv(4*5*gOutHeight*gOutWidth, vrgout_ptr_k0_r0_s0) ;
      __vr vrgout_k5_r0_s0 = _ve_vgtu_vvm(vrgout_ptr_k5_r0_s0, vmall_r0s0) ;
      __vr vrgout_ptr_k6_r0_s0 = _ve_vaddsl_vsv(4*6*gOutHeight*gOutWidth, vrgout_ptr_k0_r0_s0) ;
      __vr vrgout_k6_r0_s0 = _ve_vgtu_vvm(vrgout_ptr_k6_r0_s0, vmall_r0s0) ;
      __vr vrgout_ptr_k7_r0_s0 = _ve_vaddsl_vsv(4*7*gOutHeight*gOutWidth, vrgout_ptr_k0_r0_s0) ;
      __vr vrgout_k7_r0_s0 = _ve_vgtu_vvm(vrgout_ptr_k7_r0_s0, vmall_r0s0) ;

      VFADD_C8(vrgout_k0_r0_s0, vmall_r0s0, 0, 0, 0)
      VFADD_C8(vrgout_k1_r0_s0, vmall_r0s0, 1, 0, 0)
      VFADD_C8(vrgout_k2_r0_s0, vmall_r0s0, 2, 0, 0)
      VFADD_C8(vrgout_k3_r0_s0, vmall_r0s0, 3, 0, 0)
      VFADD_C8(vrgout_k4_r0_s0, vmall_r0s0, 4, 0, 0)
      VFADD_C8(vrgout_k5_r0_s0, vmall_r0s0, 5, 0, 0)
      VFADD_C8(vrgout_k6_r0_s0, vmall_r0s0, 6, 0, 0)
      VFADD_C8(vrgout_k7_r0_s0, vmall_r0s0, 7, 0, 0)

      __vr vrgout_ptr_k0_r0_s1 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry_r0), vrx_s1),
					 2,
					 (unsigned long)(pGOut+gOutIndex)) ;
      __vr vrgout_k0_r0_s1 = _ve_vgtu_vvm(vrgout_ptr_k0_r0_s1, vmall_r0s1) ;
      __vr vrgout_ptr_k1_r0_s1 = _ve_vaddsl_vsv(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_r0_s1) ;
      __vr vrgout_k1_r0_s1 = _ve_vgtu_vvm(vrgout_ptr_k1_r0_s1, vmall_r0s1) ;
      __vr vrgout_ptr_k2_r0_s1 = _ve_vaddsl_vsv(4*2*gOutHeight*gOutWidth, vrgout_ptr_k0_r0_s1) ;
      __vr vrgout_k2_r0_s1 = _ve_vgtu_vvm(vrgout_ptr_k2_r0_s1, vmall_r0s1) ;
      __vr vrgout_ptr_k3_r0_s1 = _ve_vaddsl_vsv(4*3*gOutHeight*gOutWidth, vrgout_ptr_k0_r0_s1) ;
      __vr vrgout_k3_r0_s1 = _ve_vgtu_vvm(vrgout_ptr_k3_r0_s1, vmall_r0s1) ;
      __vr vrgout_ptr_k4_r0_s1 = _ve_vaddsl_vsv(4*4*gOutHeight*gOutWidth, vrgout_ptr_k0_r0_s1) ;
      __vr vrgout_k4_r0_s1 = _ve_vgtu_vvm(vrgout_ptr_k4_r0_s1, vmall_r0s1) ;
      __vr vrgout_ptr_k5_r0_s1 = _ve_vaddsl_vsv(4*5*gOutHeight*gOutWidth, vrgout_ptr_k0_r0_s1) ;
      __vr vrgout_k5_r0_s1 = _ve_vgtu_vvm(vrgout_ptr_k5_r0_s1, vmall_r0s1) ;
      __vr vrgout_ptr_k6_r0_s1 = _ve_vaddsl_vsv(4*6*gOutHeight*gOutWidth, vrgout_ptr_k0_r0_s1) ;
      __vr vrgout_k6_r0_s1 = _ve_vgtu_vvm(vrgout_ptr_k6_r0_s1, vmall_r0s1) ;
      __vr vrgout_ptr_k7_r0_s1 = _ve_vaddsl_vsv(4*7*gOutHeight*gOutWidth, vrgout_ptr_k0_r0_s1) ;
      __vr vrgout_k7_r0_s1 = _ve_vgtu_vvm(vrgout_ptr_k7_r0_s1, vmall_r0s1) ;

      VFADD_C8(vrgout_k0_r0_s1, vmall_r0s1, 0, 0, 1)
      VFADD_C8(vrgout_k1_r0_s1, vmall_r0s1, 1, 0, 1)
      VFADD_C8(vrgout_k2_r0_s1, vmall_r0s1, 2, 0, 1)
      VFADD_C8(vrgout_k3_r0_s1, vmall_r0s1, 3, 0, 1)
      VFADD_C8(vrgout_k4_r0_s1, vmall_r0s1, 4, 0, 1)
      VFADD_C8(vrgout_k5_r0_s1, vmall_r0s1, 5, 0, 1)
      VFADD_C8(vrgout_k6_r0_s1, vmall_r0s1, 6, 0, 1)
      VFADD_C8(vrgout_k7_r0_s1, vmall_r0s1, 7, 0, 1)

      __vr vrgout_ptr_k0_r0_s2 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry_r0), vrx_s2),
					 2,
					 (unsigned long)(pGOut+gOutIndex)) ;
      __vr vrgout_k0_r0_s2 = _ve_vgtu_vvm(vrgout_ptr_k0_r0_s2, vmall_r0s2) ;
      __vr vrgout_ptr_k1_r0_s2 = _ve_vaddsl_vsv(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_r0_s2) ;
      __vr vrgout_k1_r0_s2 = _ve_vgtu_vvm(vrgout_ptr_k1_r0_s2, vmall_r0s2) ;
      __vr vrgout_ptr_k2_r0_s2 = _ve_vaddsl_vsv(4*2*gOutHeight*gOutWidth, vrgout_ptr_k0_r0_s2) ;
      __vr vrgout_k2_r0_s2 = _ve_vgtu_vvm(vrgout_ptr_k2_r0_s2, vmall_r0s2) ;
      __vr vrgout_ptr_k3_r0_s2 = _ve_vaddsl_vsv(4*3*gOutHeight*gOutWidth, vrgout_ptr_k0_r0_s2) ;
      __vr vrgout_k3_r0_s2 = _ve_vgtu_vvm(vrgout_ptr_k3_r0_s2, vmall_r0s2) ;
      __vr vrgout_ptr_k4_r0_s2 = _ve_vaddsl_vsv(4*4*gOutHeight*gOutWidth, vrgout_ptr_k0_r0_s2) ;
      __vr vrgout_k4_r0_s2 = _ve_vgtu_vvm(vrgout_ptr_k4_r0_s2, vmall_r0s2) ;
      __vr vrgout_ptr_k5_r0_s2 = _ve_vaddsl_vsv(4*5*gOutHeight*gOutWidth, vrgout_ptr_k0_r0_s2) ;
      __vr vrgout_k5_r0_s2 = _ve_vgtu_vvm(vrgout_ptr_k5_r0_s2, vmall_r0s2) ;
      __vr vrgout_ptr_k6_r0_s2 = _ve_vaddsl_vsv(4*6*gOutHeight*gOutWidth, vrgout_ptr_k0_r0_s2) ;
      __vr vrgout_k6_r0_s2 = _ve_vgtu_vvm(vrgout_ptr_k6_r0_s2, vmall_r0s2) ;
      __vr vrgout_ptr_k7_r0_s2 = _ve_vaddsl_vsv(4*7*gOutHeight*gOutWidth, vrgout_ptr_k0_r0_s2) ;
      __vr vrgout_k7_r0_s2 = _ve_vgtu_vvm(vrgout_ptr_k7_r0_s2, vmall_r0s2) ;

      VFADD_C8(vrgout_k0_r0_s2, vmall_r0s2, 0, 0, 2)
      VFADD_C8(vrgout_k1_r0_s2, vmall_r0s2, 1, 0, 2)
      VFADD_C8(vrgout_k2_r0_s2, vmall_r0s2, 2, 0, 2)
      VFADD_C8(vrgout_k3_r0_s2, vmall_r0s2, 3, 0, 2)
      VFADD_C8(vrgout_k4_r0_s2, vmall_r0s2, 4, 0, 2)
      VFADD_C8(vrgout_k5_r0_s2, vmall_r0s2, 5, 0, 2)
      VFADD_C8(vrgout_k6_r0_s2, vmall_r0s2, 6, 0, 2)
      VFADD_C8(vrgout_k7_r0_s2, vmall_r0s2, 7, 0, 2)

      __vr vrgout_ptr_k0_r1_s0 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry_r1), vrx_s0),
					 2,
					 (unsigned long)(pGOut+gOutIndex)) ;
      __vr vrgout_k0_r1_s0 = _ve_vgtu_vvm(vrgout_ptr_k0_r1_s0, vmall_r1s0) ;
      __vr vrgout_ptr_k1_r1_s0 = _ve_vaddsl_vsv(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_r1_s0) ;
      __vr vrgout_k1_r1_s0 = _ve_vgtu_vvm(vrgout_ptr_k1_r1_s0, vmall_r1s0) ;
      __vr vrgout_ptr_k2_r1_s0 = _ve_vaddsl_vsv(4*2*gOutHeight*gOutWidth, vrgout_ptr_k0_r1_s0) ;
      __vr vrgout_k2_r1_s0 = _ve_vgtu_vvm(vrgout_ptr_k2_r1_s0, vmall_r1s0) ;
      __vr vrgout_ptr_k3_r1_s0 = _ve_vaddsl_vsv(4*3*gOutHeight*gOutWidth, vrgout_ptr_k0_r1_s0) ;
      __vr vrgout_k3_r1_s0 = _ve_vgtu_vvm(vrgout_ptr_k3_r1_s0, vmall_r1s0) ;
      __vr vrgout_ptr_k4_r1_s0 = _ve_vaddsl_vsv(4*4*gOutHeight*gOutWidth, vrgout_ptr_k0_r1_s0) ;
      __vr vrgout_k4_r1_s0 = _ve_vgtu_vvm(vrgout_ptr_k4_r1_s0, vmall_r1s0) ;
      __vr vrgout_ptr_k5_r1_s0 = _ve_vaddsl_vsv(4*5*gOutHeight*gOutWidth, vrgout_ptr_k0_r1_s0) ;
      __vr vrgout_k5_r1_s0 = _ve_vgtu_vvm(vrgout_ptr_k5_r1_s0, vmall_r1s0) ;
      __vr vrgout_ptr_k6_r1_s0 = _ve_vaddsl_vsv(4*6*gOutHeight*gOutWidth, vrgout_ptr_k0_r1_s0) ;
      __vr vrgout_k6_r1_s0 = _ve_vgtu_vvm(vrgout_ptr_k6_r1_s0, vmall_r1s0) ;
      __vr vrgout_ptr_k7_r1_s0 = _ve_vaddsl_vsv(4*7*gOutHeight*gOutWidth, vrgout_ptr_k0_r1_s0) ;
      __vr vrgout_k7_r1_s0 = _ve_vgtu_vvm(vrgout_ptr_k7_r1_s0, vmall_r1s0) ;

      VFADD_C8(vrgout_k0_r1_s0, vmall_r1s0, 0, 1, 0)
      VFADD_C8(vrgout_k1_r1_s0, vmall_r1s0, 1, 1, 0)
      VFADD_C8(vrgout_k2_r1_s0, vmall_r1s0, 2, 1, 0)
      VFADD_C8(vrgout_k3_r1_s0, vmall_r1s0, 3, 1, 0)
      VFADD_C8(vrgout_k4_r1_s0, vmall_r1s0, 4, 1, 0)
      VFADD_C8(vrgout_k5_r1_s0, vmall_r1s0, 5, 1, 0)
      VFADD_C8(vrgout_k6_r1_s0, vmall_r1s0, 6, 1, 0)
      VFADD_C8(vrgout_k7_r1_s0, vmall_r1s0, 7, 1, 0)

      __vr vrgout_ptr_k0_r1_s1 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry_r1), vrx_s1),
					 2,
					 (unsigned long)(pGOut+gOutIndex)) ;
      __vr vrgout_k0_r1_s1 = _ve_vgtu_vvm(vrgout_ptr_k0_r1_s1, vmall_r1s1) ;
      __vr vrgout_ptr_k1_r1_s1 = _ve_vaddsl_vsv(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_r1_s1) ;
      __vr vrgout_k1_r1_s1 = _ve_vgtu_vvm(vrgout_ptr_k1_r1_s1, vmall_r1s1) ;
      __vr vrgout_ptr_k2_r1_s1 = _ve_vaddsl_vsv(4*2*gOutHeight*gOutWidth, vrgout_ptr_k0_r1_s1) ;
      __vr vrgout_k2_r1_s1 = _ve_vgtu_vvm(vrgout_ptr_k2_r1_s1, vmall_r1s1) ;
      __vr vrgout_ptr_k3_r1_s1 = _ve_vaddsl_vsv(4*3*gOutHeight*gOutWidth, vrgout_ptr_k0_r1_s1) ;
      __vr vrgout_k3_r1_s1 = _ve_vgtu_vvm(vrgout_ptr_k3_r1_s1, vmall_r1s1) ;
      __vr vrgout_ptr_k4_r1_s1 = _ve_vaddsl_vsv(4*4*gOutHeight*gOutWidth, vrgout_ptr_k0_r1_s1) ;
      __vr vrgout_k4_r1_s1 = _ve_vgtu_vvm(vrgout_ptr_k4_r1_s1, vmall_r1s1) ;
      __vr vrgout_ptr_k5_r1_s1 = _ve_vaddsl_vsv(4*5*gOutHeight*gOutWidth, vrgout_ptr_k0_r1_s1) ;
      __vr vrgout_k5_r1_s1 = _ve_vgtu_vvm(vrgout_ptr_k5_r1_s1, vmall_r1s1) ;
      __vr vrgout_ptr_k6_r1_s1 = _ve_vaddsl_vsv(4*6*gOutHeight*gOutWidth, vrgout_ptr_k0_r1_s1) ;
      __vr vrgout_k6_r1_s1 = _ve_vgtu_vvm(vrgout_ptr_k6_r1_s1, vmall_r1s1) ;
      __vr vrgout_ptr_k7_r1_s1 = _ve_vaddsl_vsv(4*7*gOutHeight*gOutWidth, vrgout_ptr_k0_r1_s1) ;
      __vr vrgout_k7_r1_s1 = _ve_vgtu_vvm(vrgout_ptr_k7_r1_s1, vmall_r1s1) ;

      VFADD_C8(vrgout_k0_r1_s1, vmall_r1s1, 0, 1, 1)
      VFADD_C8(vrgout_k1_r1_s1, vmall_r1s1, 1, 1, 1)
      VFADD_C8(vrgout_k2_r1_s1, vmall_r1s1, 2, 1, 1)
      VFADD_C8(vrgout_k3_r1_s1, vmall_r1s1, 3, 1, 1)
      VFADD_C8(vrgout_k4_r1_s1, vmall_r1s1, 4, 1, 1)
      VFADD_C8(vrgout_k5_r1_s1, vmall_r1s1, 5, 1, 1)
      VFADD_C8(vrgout_k6_r1_s1, vmall_r1s1, 6, 1, 1)
      VFADD_C8(vrgout_k7_r1_s1, vmall_r1s1, 7, 1, 1)

      __vr vrgout_ptr_k0_r1_s2 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry_r1), vrx_s2),
					 2,
					 (unsigned long)(pGOut+gOutIndex)) ;
      __vr vrgout_k0_r1_s2 = _ve_vgtu_vvm(vrgout_ptr_k0_r1_s2, vmall_r1s2) ;
      __vr vrgout_ptr_k1_r1_s2 = _ve_vaddsl_vsv(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_r1_s2) ;
      __vr vrgout_k1_r1_s2 = _ve_vgtu_vvm(vrgout_ptr_k1_r1_s2, vmall_r1s2) ;
      __vr vrgout_ptr_k2_r1_s2 = _ve_vaddsl_vsv(4*2*gOutHeight*gOutWidth, vrgout_ptr_k0_r1_s2) ;
      __vr vrgout_k2_r1_s2 = _ve_vgtu_vvm(vrgout_ptr_k2_r1_s2, vmall_r1s2) ;
      __vr vrgout_ptr_k3_r1_s2 = _ve_vaddsl_vsv(4*3*gOutHeight*gOutWidth, vrgout_ptr_k0_r1_s2) ;
      __vr vrgout_k3_r1_s2 = _ve_vgtu_vvm(vrgout_ptr_k3_r1_s2, vmall_r1s2) ;
      __vr vrgout_ptr_k4_r1_s2 = _ve_vaddsl_vsv(4*4*gOutHeight*gOutWidth, vrgout_ptr_k0_r1_s2) ;
      __vr vrgout_k4_r1_s2 = _ve_vgtu_vvm(vrgout_ptr_k4_r1_s2, vmall_r1s2) ;
      __vr vrgout_ptr_k5_r1_s2 = _ve_vaddsl_vsv(4*5*gOutHeight*gOutWidth, vrgout_ptr_k0_r1_s2) ;
      __vr vrgout_k5_r1_s2 = _ve_vgtu_vvm(vrgout_ptr_k5_r1_s2, vmall_r1s2) ;
      __vr vrgout_ptr_k6_r1_s2 = _ve_vaddsl_vsv(4*6*gOutHeight*gOutWidth, vrgout_ptr_k0_r1_s2) ;
      __vr vrgout_k6_r1_s2 = _ve_vgtu_vvm(vrgout_ptr_k6_r1_s2, vmall_r1s2) ;
      __vr vrgout_ptr_k7_r1_s2 = _ve_vaddsl_vsv(4*7*gOutHeight*gOutWidth, vrgout_ptr_k0_r1_s2) ;
      __vr vrgout_k7_r1_s2 = _ve_vgtu_vvm(vrgout_ptr_k7_r1_s2, vmall_r1s2) ;

      VFADD_C8(vrgout_k0_r1_s2, vmall_r1s2, 0, 1, 2)
      VFADD_C8(vrgout_k1_r1_s2, vmall_r1s2, 1, 1, 2)
      VFADD_C8(vrgout_k2_r1_s2, vmall_r1s2, 2, 1, 2)
      VFADD_C8(vrgout_k3_r1_s2, vmall_r1s2, 3, 1, 2)
      VFADD_C8(vrgout_k4_r1_s2, vmall_r1s2, 4, 1, 2)
      VFADD_C8(vrgout_k5_r1_s2, vmall_r1s2, 5, 1, 2)
      VFADD_C8(vrgout_k6_r1_s2, vmall_r1s2, 6, 1, 2)
      VFADD_C8(vrgout_k7_r1_s2, vmall_r1s2, 7, 1, 2)

      __vr vrgout_ptr_k0_r2_s0 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry_r2), vrx_s0),
					 2,
					 (unsigned long)(pGOut+gOutIndex)) ;
      __vr vrgout_k0_r2_s0 = _ve_vgtu_vvm(vrgout_ptr_k0_r2_s0, vmall_r2s0) ;
      __vr vrgout_ptr_k1_r2_s0 = _ve_vaddsl_vsv(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_r2_s0) ;
      __vr vrgout_k1_r2_s0 = _ve_vgtu_vvm(vrgout_ptr_k1_r2_s0, vmall_r2s0) ;
      __vr vrgout_ptr_k2_r2_s0 = _ve_vaddsl_vsv(4*2*gOutHeight*gOutWidth, vrgout_ptr_k0_r2_s0) ;
      __vr vrgout_k2_r2_s0 = _ve_vgtu_vvm(vrgout_ptr_k2_r2_s0, vmall_r2s0) ;
      __vr vrgout_ptr_k3_r2_s0 = _ve_vaddsl_vsv(4*3*gOutHeight*gOutWidth, vrgout_ptr_k0_r2_s0) ;
      __vr vrgout_k3_r2_s0 = _ve_vgtu_vvm(vrgout_ptr_k3_r2_s0, vmall_r2s0) ;
      __vr vrgout_ptr_k4_r2_s0 = _ve_vaddsl_vsv(4*4*gOutHeight*gOutWidth, vrgout_ptr_k0_r2_s0) ;
      __vr vrgout_k4_r2_s0 = _ve_vgtu_vvm(vrgout_ptr_k4_r2_s0, vmall_r2s0) ;
      __vr vrgout_ptr_k5_r2_s0 = _ve_vaddsl_vsv(4*5*gOutHeight*gOutWidth, vrgout_ptr_k0_r2_s0) ;
      __vr vrgout_k5_r2_s0 = _ve_vgtu_vvm(vrgout_ptr_k5_r2_s0, vmall_r2s0) ;
      __vr vrgout_ptr_k6_r2_s0 = _ve_vaddsl_vsv(4*6*gOutHeight*gOutWidth, vrgout_ptr_k0_r2_s0) ;
      __vr vrgout_k6_r2_s0 = _ve_vgtu_vvm(vrgout_ptr_k6_r2_s0, vmall_r2s0) ;
      __vr vrgout_ptr_k7_r2_s0 = _ve_vaddsl_vsv(4*7*gOutHeight*gOutWidth, vrgout_ptr_k0_r2_s0) ;
      __vr vrgout_k7_r2_s0 = _ve_vgtu_vvm(vrgout_ptr_k7_r2_s0, vmall_r2s0) ;

      VFADD_C8(vrgout_k0_r2_s0, vmall_r2s0, 0, 2, 0)
      VFADD_C8(vrgout_k1_r2_s0, vmall_r2s0, 1, 2, 0)
      VFADD_C8(vrgout_k2_r2_s0, vmall_r2s0, 2, 2, 0)
      VFADD_C8(vrgout_k3_r2_s0, vmall_r2s0, 3, 2, 0)
      VFADD_C8(vrgout_k4_r2_s0, vmall_r2s0, 4, 2, 0)
      VFADD_C8(vrgout_k5_r2_s0, vmall_r2s0, 5, 2, 0)
      VFADD_C8(vrgout_k6_r2_s0, vmall_r2s0, 6, 2, 0)
      VFADD_C8(vrgout_k7_r2_s0, vmall_r2s0, 7, 2, 0)

      __vr vrgout_ptr_k0_r2_s1 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry_r2), vrx_s1),
					 2,
					 (unsigned long)(pGOut+gOutIndex)) ;
      __vr vrgout_k0_r2_s1 = _ve_vgtu_vvm(vrgout_ptr_k0_r2_s1, vmall_r2s1) ;
      __vr vrgout_ptr_k1_r2_s1 = _ve_vaddsl_vsv(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_r2_s1) ;
      __vr vrgout_k1_r2_s1 = _ve_vgtu_vvm(vrgout_ptr_k1_r2_s1, vmall_r2s1) ;
      __vr vrgout_ptr_k2_r2_s1 = _ve_vaddsl_vsv(4*2*gOutHeight*gOutWidth, vrgout_ptr_k0_r2_s1) ;
      __vr vrgout_k2_r2_s1 = _ve_vgtu_vvm(vrgout_ptr_k2_r2_s1, vmall_r2s1) ;
      __vr vrgout_ptr_k3_r2_s1 = _ve_vaddsl_vsv(4*3*gOutHeight*gOutWidth, vrgout_ptr_k0_r2_s1) ;
      __vr vrgout_k3_r2_s1 = _ve_vgtu_vvm(vrgout_ptr_k3_r2_s1, vmall_r2s1) ;
      __vr vrgout_ptr_k4_r2_s1 = _ve_vaddsl_vsv(4*4*gOutHeight*gOutWidth, vrgout_ptr_k0_r2_s1) ;
      __vr vrgout_k4_r2_s1 = _ve_vgtu_vvm(vrgout_ptr_k4_r2_s1, vmall_r2s1) ;
      __vr vrgout_ptr_k5_r2_s1 = _ve_vaddsl_vsv(4*5*gOutHeight*gOutWidth, vrgout_ptr_k0_r2_s1) ;
      __vr vrgout_k5_r2_s1 = _ve_vgtu_vvm(vrgout_ptr_k5_r2_s1, vmall_r2s1) ;
      __vr vrgout_ptr_k6_r2_s1 = _ve_vaddsl_vsv(4*6*gOutHeight*gOutWidth, vrgout_ptr_k0_r2_s1) ;
      __vr vrgout_k6_r2_s1 = _ve_vgtu_vvm(vrgout_ptr_k6_r2_s1, vmall_r2s1) ;
      __vr vrgout_ptr_k7_r2_s1 = _ve_vaddsl_vsv(4*7*gOutHeight*gOutWidth, vrgout_ptr_k0_r2_s1) ;
      __vr vrgout_k7_r2_s1 = _ve_vgtu_vvm(vrgout_ptr_k7_r2_s1, vmall_r2s1) ;

      VFADD_C8(vrgout_k0_r2_s1, vmall_r2s1, 0, 2, 1)
      VFADD_C8(vrgout_k1_r2_s1, vmall_r2s1, 1, 2, 1)
      VFADD_C8(vrgout_k2_r2_s1, vmall_r2s1, 2, 2, 1)
      VFADD_C8(vrgout_k3_r2_s1, vmall_r2s1, 3, 2, 1)
      VFADD_C8(vrgout_k4_r2_s1, vmall_r2s1, 4, 2, 1)
      VFADD_C8(vrgout_k5_r2_s1, vmall_r2s1, 5, 2, 1)
      VFADD_C8(vrgout_k6_r2_s1, vmall_r2s1, 6, 2, 1)
      VFADD_C8(vrgout_k7_r2_s1, vmall_r2s1, 7, 2, 1)

      __vr vrgout_ptr_k0_r2_s2 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry_r2), vrx_s2),
					 2,
					 (unsigned long)(pGOut+gOutIndex)) ;
      __vr vrgout_k0_r2_s2 = _ve_vgtu_vvm(vrgout_ptr_k0_r2_s2, vmall_r2s2) ;
      __vr vrgout_ptr_k1_r2_s2 = _ve_vaddsl_vsv(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_r2_s2) ;
      __vr vrgout_k1_r2_s2 = _ve_vgtu_vvm(vrgout_ptr_k1_r2_s2, vmall_r2s2) ;
      __vr vrgout_ptr_k2_r2_s2 = _ve_vaddsl_vsv(4*2*gOutHeight*gOutWidth, vrgout_ptr_k0_r2_s2) ;
      __vr vrgout_k2_r2_s2 = _ve_vgtu_vvm(vrgout_ptr_k2_r2_s2, vmall_r2s2) ;
      __vr vrgout_ptr_k3_r2_s2 = _ve_vaddsl_vsv(4*3*gOutHeight*gOutWidth, vrgout_ptr_k0_r2_s2) ;
      __vr vrgout_k3_r2_s2 = _ve_vgtu_vvm(vrgout_ptr_k3_r2_s2, vmall_r2s2) ;
      __vr vrgout_ptr_k4_r2_s2 = _ve_vaddsl_vsv(4*4*gOutHeight*gOutWidth, vrgout_ptr_k0_r2_s2) ;
      __vr vrgout_k4_r2_s2 = _ve_vgtu_vvm(vrgout_ptr_k4_r2_s2, vmall_r2s2) ;
      __vr vrgout_ptr_k5_r2_s2 = _ve_vaddsl_vsv(4*5*gOutHeight*gOutWidth, vrgout_ptr_k0_r2_s2) ;
      __vr vrgout_k5_r2_s2 = _ve_vgtu_vvm(vrgout_ptr_k5_r2_s2, vmall_r2s2) ;
      __vr vrgout_ptr_k6_r2_s2 = _ve_vaddsl_vsv(4*6*gOutHeight*gOutWidth, vrgout_ptr_k0_r2_s2) ;
      __vr vrgout_k6_r2_s2 = _ve_vgtu_vvm(vrgout_ptr_k6_r2_s2, vmall_r2s2) ;
      __vr vrgout_ptr_k7_r2_s2 = _ve_vaddsl_vsv(4*7*gOutHeight*gOutWidth, vrgout_ptr_k0_r2_s2) ;
      __vr vrgout_k7_r2_s2 = _ve_vgtu_vvm(vrgout_ptr_k7_r2_s2, vmall_r2s2) ;

      VFADD_C8(vrgout_k0_r2_s2, vmall_r2s2, 0, 2, 2)
      VFADD_C8(vrgout_k1_r2_s2, vmall_r2s2, 1, 2, 2)
      VFADD_C8(vrgout_k2_r2_s2, vmall_r2s2, 2, 2, 2)
      VFADD_C8(vrgout_k3_r2_s2, vmall_r2s2, 3, 2, 2)
      VFADD_C8(vrgout_k4_r2_s2, vmall_r2s2, 4, 2, 2)
      VFADD_C8(vrgout_k5_r2_s2, vmall_r2s2, 5, 2, 2)
      VFADD_C8(vrgout_k6_r2_s2, vmall_r2s2, 6, 2, 2)
      VFADD_C8(vrgout_k7_r2_s2, vmall_r2s2, 7, 2, 2)
#undef VFADD_C8

    } // gOutChannel

    _ve_vstu_vss(vrsum01, 4, pGIn+gInIndex) ;
    _ve_vstl_vss(vrsum01, 4, pGIn+gInIndex+  gInPixels) ;
    _ve_vstu_vss(vrsum23, 4, pGIn+gInIndex+2*gInPixels) ;
    _ve_vstl_vss(vrsum23, 4, pGIn+gInIndex+3*gInPixels) ;
    _ve_vstu_vss(vrsum45, 4, pGIn+gInIndex+4*gInPixels) ;
    _ve_vstl_vss(vrsum45, 4, pGIn+gInIndex+5*gInPixels) ;
    _ve_vstu_vss(vrsum67, 4, pGIn+gInIndex+6*gInPixels) ;
    _ve_vstl_vss(vrsum67, 4, pGIn+gInIndex+7*gInPixels) ;
  } // gOutPixels
}

vednnError_t
vednnConvolutionBackwardData_direct_ker3_iwU128(
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
             strideWidth, strideHeight,
	     padWidth, padHeight,
	     dilationWidth, dilationHeight,
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
             strideWidth, strideHeight,
	     padWidth, padHeight,
	     dilationWidth, dilationHeight,
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
             strideWidth, strideHeight,
	     padWidth, padHeight,
	     dilationWidth, dilationHeight,
	     gInChannelGroup, gOutChannelGroup,
	     gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	     gInPixels, n, k,
	     nH, vrh, vrw ) ;

	  k+=4 ;
	}
	for (; k<gInChannelGroup; k+=8) {
	  c8(pGOut, pKernel, pGIn,
	     gOutChannel, gOutWidth, gOutHeight,
             gInChannel, gInWidth, gInHeight,
             kernWidth, kernHeight,
             strideWidth, strideHeight,
	     padWidth, padHeight,
	     dilationWidth, dilationHeight,
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
