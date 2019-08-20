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

    __vr vrj_s0 = _ve_vaddsl_vsv(padWidth-0*dilationWidth, vrw) ;
    __vr vrj_s1 = _ve_vaddsl_vsv(padWidth-1*dilationWidth, vrw) ;
    __vr vrj_s2 = _ve_vaddsl_vsv(padWidth-2*dilationWidth, vrw) ;
    __vr vrj_s3 = _ve_vaddsl_vsv(padWidth-3*dilationWidth, vrw) ;
    __vr vrj_s4 = _ve_vaddsl_vsv(padWidth-4*dilationWidth, vrw) ;

    __vr vrx_s0 = _ve_vdivsl_vvs(vrj_s0, strideWidth) ;
    __vr vrx_s1 = _ve_vdivsl_vvs(vrj_s1, strideWidth) ;
    __vr vrx_s2 = _ve_vdivsl_vvs(vrj_s2, strideWidth) ;
    __vr vrx_s3 = _ve_vdivsl_vvs(vrj_s3, strideWidth) ;
    __vr vrx_s4 = _ve_vdivsl_vvs(vrj_s4, strideWidth) ;

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

    __vm256 vmx0_s3 = _ve_vfmkl_mcv(VECC_IEQ, _ve_vcmpsl_vvv(vrj_s3, _ve_vmulsl_vsv(strideWidth, vrx_s3))) ;
    __vm256 vmx1_s3 = _ve_vfmkl_mcv(VECC_GE, vrx_s3) ;
    __vm256 vmx2_s3 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(gOutWidth,vrx_s3)) ;
    __vm256 vmx_s3 = _ve_andm_mmm(_ve_andm_mmm(vmx0_s3, vmx1_s3), vmx2_s3) ;

    __vm256 vmx0_s4 = _ve_vfmkl_mcv(VECC_IEQ, _ve_vcmpsl_vvv(vrj_s4, _ve_vmulsl_vsv(strideWidth, vrx_s4))) ;
    __vm256 vmx1_s4 = _ve_vfmkl_mcv(VECC_GE, vrx_s4) ;
    __vm256 vmx2_s4 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(gOutWidth,vrx_s4)) ;
    __vm256 vmx_s4 = _ve_andm_mmm(_ve_andm_mmm(vmx0_s4, vmx1_s4), vmx2_s4) ;

    for (int64_t r=0; r<kernHeight; r++) {
      __vr vri = _ve_vaddsl_vsv(padHeight-r*dilationHeight+h, vrh) ;
      __vr vry = _ve_vdivsl_vvs(vri, strideHeight) ;

      __vm256 vmy0 = _ve_vfmkl_mcv(VECC_IEQ, _ve_vcmpsl_vvv(vri, _ve_vmulsl_vsv(strideHeight, vry))) ;
      __vm256 vmy1 = _ve_vfmkl_mcv(VECC_GE, vry) ;
      __vm256 vmy2 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(gOutHeight,vry)) ;

      __vm256 vmy = _ve_andm_mmm(_ve_andm_mmm(vmy0, vmy1), vmy2) ;


      __vm256 vmall_s0 = _ve_andm_mmm(vmy,vmx_s0) ;
      __vm256 vmall_s1 = _ve_andm_mmm(vmy,vmx_s1) ;
      __vm256 vmall_s2 = _ve_andm_mmm(vmy,vmx_s2) ;
      __vm256 vmall_s3 = _ve_andm_mmm(vmy,vmx_s3) ;
      __vm256 vmall_s4 = _ve_andm_mmm(vmy,vmx_s4) ;

      int64_t k=0;
      if( (gOutChannelGroup & 0x01 ) == 1 ) {
	int64_t gOutIndex    = gOutGroupOffset + ((n * gOutChannel + k) * gOutHeight) * gOutWidth ;
	const float *pKerValue = pKernel + kernGroupOffset + (((k  ) * gInChannelGroup + c) * kernHeight) * kernWidth ;


#define VFADD_C1(VRGOUT, VM, K, R, S)  {												\
	  const uint64_t kerValue = pKerValue[(((K)*gInChannelGroup + 0) * kernHeight +(R)) * kernWidth + (S) ] ;	\
	  VRGOUT = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), VRGOUT, VM) ;			\
	  vrsum = _ve_pvfmad_vvsv(vrsum, kerValue, VRGOUT) ;				\
	}

	__vr vrgout_ptr_k0_s0 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry), vrx_s0),
					   2,
					   (unsigned long)(pGOut+gOutIndex)) ;
	__vr vrgout_k0_s0 = _ve_vgtu_vvm(vrgout_ptr_k0_s0, vmall_s0) ;
	__vr vrgout_ptr_k1_s0 = _ve_vaddsl_vsv(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_s0) ;
	__vr vrgout_k1_s0 = _ve_vgtu_vvm(vrgout_ptr_k1_s0, vmall_s0) ;

	VFADD_C1(vrgout_k0_s0, vmall_s0, 0, r, 0)

	__vr vrgout_ptr_k0_s1 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry), vrx_s1),
					   2,
					   (unsigned long)(pGOut+gOutIndex)) ;
	__vr vrgout_k0_s1 = _ve_vgtu_vvm(vrgout_ptr_k0_s1, vmall_s1) ;

	VFADD_C1(vrgout_k0_s1, vmall_s1, 0, r, 1)


	__vr vrgout_ptr_k0_s2 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry), vrx_s2),
					   2,
					   (unsigned long)(pGOut+gOutIndex)) ;
	__vr vrgout_k0_s2 = _ve_vgtu_vvm(vrgout_ptr_k0_s2, vmall_s2) ;

	VFADD_C1(vrgout_k0_s2, vmall_s2, 0, r, 2)

	__vr vrgout_ptr_k0_s3 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry), vrx_s3),
					   2,
					   (unsigned long)(pGOut+gOutIndex)) ;
	__vr vrgout_k0_s3 = _ve_vgtu_vvm(vrgout_ptr_k0_s3, vmall_s3) ;
	__vr vrgout_ptr_k1_s3 = _ve_vaddsl_vsv(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_s3) ;

	VFADD_C1(vrgout_k0_s3, vmall_s3, 0, r, 3)

	__vr vrgout_ptr_k0_s4 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry), vrx_s4),
					   2,
					   (unsigned long)(pGOut+gOutIndex)) ;
	__vr vrgout_k0_s4 = _ve_vgtu_vvm(vrgout_ptr_k0_s4, vmall_s4) ;

	VFADD_C1(vrgout_k0_s4, vmall_s4, 0, r, 4)

	k+=1 ;
      }
      if( ((gOutChannelGroup >> 1) & 0x01 ) == 1 ) {
	int64_t gOutIndex    = gOutGroupOffset + ((n * gOutChannel + k) * gOutHeight) * gOutWidth ;
	const float *pKerValue = pKernel + kernGroupOffset + (((k  ) * gInChannelGroup + c) * kernHeight) * kernWidth ;

	__vr vrgout_ptr_k0_s0 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry), vrx_s0),
					   2,
					   (unsigned long)(pGOut+gOutIndex)) ;
	__vr vrgout_k0_s0 = _ve_vgtu_vvm(vrgout_ptr_k0_s0, vmall_s0) ;
	__vr vrgout_ptr_k1_s0 = _ve_vaddsl_vsv(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_s0) ;
	__vr vrgout_k1_s0 = _ve_vgtu_vvm(vrgout_ptr_k1_s0, vmall_s0) ;

	VFADD_C1(vrgout_k0_s0, vmall_s0, 0, r, 0)
	VFADD_C1(vrgout_k1_s0, vmall_s0, 1, r, 0)

	__vr vrgout_ptr_k0_s1 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry), vrx_s1),
					   2,
					   (unsigned long)(pGOut+gOutIndex)) ;
	__vr vrgout_k0_s1 = _ve_vgtu_vvm(vrgout_ptr_k0_s1, vmall_s1) ;
	__vr vrgout_ptr_k1_s1 = _ve_vaddsl_vsv(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_s1) ;
	__vr vrgout_k1_s1 = _ve_vgtu_vvm(vrgout_ptr_k1_s1, vmall_s1) ;

	VFADD_C1(vrgout_k0_s1, vmall_s1, 0, r, 1)
	VFADD_C1(vrgout_k1_s1, vmall_s1, 1, r, 1)


	__vr vrgout_ptr_k0_s2 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry), vrx_s2),
					   2,
					   (unsigned long)(pGOut+gOutIndex)) ;
	__vr vrgout_k0_s2 = _ve_vgtu_vvm(vrgout_ptr_k0_s2, vmall_s2) ;
	__vr vrgout_ptr_k1_s2 = _ve_vaddsl_vsv(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_s2) ;
	__vr vrgout_k1_s2 = _ve_vgtu_vvm(vrgout_ptr_k1_s2, vmall_s2) ;

	VFADD_C1(vrgout_k0_s2, vmall_s2, 0, r, 2)
	VFADD_C1(vrgout_k1_s2, vmall_s2, 1, r, 2)

	__vr vrgout_ptr_k0_s3 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry), vrx_s3),
					   2,
					   (unsigned long)(pGOut+gOutIndex)) ;
	__vr vrgout_k0_s3 = _ve_vgtu_vvm(vrgout_ptr_k0_s3, vmall_s3) ;
	__vr vrgout_ptr_k1_s3 = _ve_vaddsl_vsv(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_s3) ;
	__vr vrgout_k1_s3 = _ve_vgtu_vvm(vrgout_ptr_k1_s3, vmall_s3) ;
	__vr vrgout_ptr_k2_s3 = _ve_vaddsl_vsv(4*2*gOutHeight*gOutWidth, vrgout_ptr_k0_s3) ;

	VFADD_C1(vrgout_k0_s3, vmall_s3, 0, r, 3)
	VFADD_C1(vrgout_k1_s3, vmall_s3, 1, r, 3)

	__vr vrgout_ptr_k0_s4 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry), vrx_s4),
					   2,
					   (unsigned long)(pGOut+gOutIndex)) ;
	__vr vrgout_k0_s4 = _ve_vgtu_vvm(vrgout_ptr_k0_s4, vmall_s4) ;
	__vr vrgout_ptr_k1_s4 = _ve_vaddsl_vsv(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_s4) ;
	__vr vrgout_k1_s4 = _ve_vgtu_vvm(vrgout_ptr_k1_s4, vmall_s4) ;

	VFADD_C1(vrgout_k0_s4, vmall_s4, 0, r, 4)
	VFADD_C1(vrgout_k1_s4, vmall_s4, 1, r, 4)

	k+=2 ;
      }
      if( ((gOutChannelGroup >> 2) & 0x01 ) == 1 ) {
	int64_t gOutIndex    = gOutGroupOffset + ((n * gOutChannel + k) * gOutHeight) * gOutWidth ;
	const float *pKerValue = pKernel + kernGroupOffset + (((k  ) * gInChannelGroup + c) * kernHeight) * kernWidth ;

	__vr vrgout_ptr_k0_s0 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry), vrx_s0),
					   2,
					   (unsigned long)(pGOut+gOutIndex)) ;
	__vr vrgout_k0_s0 = _ve_vgtu_vvm(vrgout_ptr_k0_s0, vmall_s0) ;
	__vr vrgout_ptr_k1_s0 = _ve_vaddsl_vsv(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_s0) ;
	__vr vrgout_k1_s0 = _ve_vgtu_vvm(vrgout_ptr_k1_s0, vmall_s0) ;
	__vr vrgout_ptr_k2_s0 = _ve_vaddsl_vsv(4*2*gOutHeight*gOutWidth, vrgout_ptr_k0_s0) ;
	__vr vrgout_k2_s0 = _ve_vgtu_vvm(vrgout_ptr_k2_s0, vmall_s0) ;
	__vr vrgout_ptr_k3_s0 = _ve_vaddsl_vsv(4*3*gOutHeight*gOutWidth, vrgout_ptr_k0_s0) ;
	__vr vrgout_k3_s0 = _ve_vgtu_vvm(vrgout_ptr_k3_s0, vmall_s0) ;

	VFADD_C1(vrgout_k0_s0, vmall_s0, 0, r, 0)
	VFADD_C1(vrgout_k1_s0, vmall_s0, 1, r, 0)
	VFADD_C1(vrgout_k2_s0, vmall_s0, 2, r, 0)
	VFADD_C1(vrgout_k3_s0, vmall_s0, 3, r, 0)

	__vr vrgout_ptr_k0_s1 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry), vrx_s1),
					   2,
					   (unsigned long)(pGOut+gOutIndex)) ;
	__vr vrgout_k0_s1 = _ve_vgtu_vvm(vrgout_ptr_k0_s1, vmall_s1) ;
	__vr vrgout_ptr_k1_s1 = _ve_vaddsl_vsv(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_s1) ;
	__vr vrgout_k1_s1 = _ve_vgtu_vvm(vrgout_ptr_k1_s1, vmall_s1) ;
	__vr vrgout_ptr_k2_s1 = _ve_vaddsl_vsv(4*2*gOutHeight*gOutWidth, vrgout_ptr_k0_s1) ;
	__vr vrgout_k2_s1 = _ve_vgtu_vvm(vrgout_ptr_k2_s1, vmall_s1) ;
	__vr vrgout_ptr_k3_s1 = _ve_vaddsl_vsv(4*3*gOutHeight*gOutWidth, vrgout_ptr_k0_s1) ;
	__vr vrgout_k3_s1 = _ve_vgtu_vvm(vrgout_ptr_k3_s1, vmall_s1) ;

	VFADD_C1(vrgout_k0_s1, vmall_s1, 0, r, 1)
	VFADD_C1(vrgout_k1_s1, vmall_s1, 1, r, 1)
	VFADD_C1(vrgout_k2_s1, vmall_s1, 2, r, 1)
	VFADD_C1(vrgout_k3_s1, vmall_s1, 3, r, 1)


	__vr vrgout_ptr_k0_s2 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry), vrx_s2),
					   2,
					   (unsigned long)(pGOut+gOutIndex)) ;
	__vr vrgout_k0_s2 = _ve_vgtu_vvm(vrgout_ptr_k0_s2, vmall_s2) ;
	__vr vrgout_ptr_k1_s2 = _ve_vaddsl_vsv(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_s2) ;
	__vr vrgout_k1_s2 = _ve_vgtu_vvm(vrgout_ptr_k1_s2, vmall_s2) ;
	__vr vrgout_ptr_k2_s2 = _ve_vaddsl_vsv(4*2*gOutHeight*gOutWidth, vrgout_ptr_k0_s2) ;
	__vr vrgout_k2_s2 = _ve_vgtu_vvm(vrgout_ptr_k2_s2, vmall_s2) ;
	__vr vrgout_ptr_k3_s2 = _ve_vaddsl_vsv(4*3*gOutHeight*gOutWidth, vrgout_ptr_k0_s2) ;
	__vr vrgout_k3_s2 = _ve_vgtu_vvm(vrgout_ptr_k3_s2, vmall_s2) ;

	VFADD_C1(vrgout_k0_s2, vmall_s2, 0, r, 2)
	VFADD_C1(vrgout_k1_s2, vmall_s2, 1, r, 2)
	VFADD_C1(vrgout_k2_s2, vmall_s2, 2, r, 2)
	VFADD_C1(vrgout_k3_s2, vmall_s2, 3, r, 2)

	__vr vrgout_ptr_k0_s3 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry), vrx_s3),
					   2,
					   (unsigned long)(pGOut+gOutIndex)) ;
	__vr vrgout_k0_s3 = _ve_vgtu_vvm(vrgout_ptr_k0_s3, vmall_s3) ;
	__vr vrgout_ptr_k1_s3 = _ve_vaddsl_vsv(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_s3) ;
	__vr vrgout_k1_s3 = _ve_vgtu_vvm(vrgout_ptr_k1_s3, vmall_s3) ;
	__vr vrgout_ptr_k2_s3 = _ve_vaddsl_vsv(4*2*gOutHeight*gOutWidth, vrgout_ptr_k0_s3) ;
	__vr vrgout_k2_s3 = _ve_vgtu_vvm(vrgout_ptr_k2_s3, vmall_s3) ;
	__vr vrgout_ptr_k3_s3 = _ve_vaddsl_vsv(4*3*gOutHeight*gOutWidth, vrgout_ptr_k0_s3) ;
	__vr vrgout_k3_s3 = _ve_vgtu_vvm(vrgout_ptr_k3_s3, vmall_s3) ;
	__vr vrgout_ptr_k4_s3 = _ve_vaddsl_vsv(4*4*gOutHeight*gOutWidth, vrgout_ptr_k0_s3) ;

	VFADD_C1(vrgout_k0_s3, vmall_s3, 0, r, 3)
	VFADD_C1(vrgout_k1_s3, vmall_s3, 1, r, 3)
	VFADD_C1(vrgout_k2_s3, vmall_s3, 2, r, 3)
	VFADD_C1(vrgout_k3_s3, vmall_s3, 3, r, 3)

	__vr vrgout_ptr_k0_s4 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry), vrx_s4),
					   2,
					   (unsigned long)(pGOut+gOutIndex)) ;
	__vr vrgout_k0_s4 = _ve_vgtu_vvm(vrgout_ptr_k0_s4, vmall_s4) ;
	__vr vrgout_ptr_k1_s4 = _ve_vaddsl_vsv(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_s4) ;
	__vr vrgout_k1_s4 = _ve_vgtu_vvm(vrgout_ptr_k1_s4, vmall_s4) ;
	__vr vrgout_ptr_k2_s4 = _ve_vaddsl_vsv(4*2*gOutHeight*gOutWidth, vrgout_ptr_k0_s4) ;
	__vr vrgout_k2_s4 = _ve_vgtu_vvm(vrgout_ptr_k2_s4, vmall_s4) ;
	__vr vrgout_ptr_k3_s4 = _ve_vaddsl_vsv(4*3*gOutHeight*gOutWidth, vrgout_ptr_k0_s4) ;
	__vr vrgout_k3_s4 = _ve_vgtu_vvm(vrgout_ptr_k3_s4, vmall_s4) ;

	VFADD_C1(vrgout_k0_s4, vmall_s4, 0, r, 4)
	VFADD_C1(vrgout_k1_s4, vmall_s4, 1, r, 4)
	VFADD_C1(vrgout_k2_s4, vmall_s4, 2, r, 4)
	VFADD_C1(vrgout_k3_s4, vmall_s4, 3, r, 4)

	k+=4 ;
      }
      for (; k<gOutChannelGroup; k+=8) {
	int64_t gOutIndex    = gOutGroupOffset + ((n * gOutChannel + k) * gOutHeight) * gOutWidth ;
	const float *pKerValue = pKernel + kernGroupOffset + (((k  ) * gInChannelGroup + c) * kernHeight) * kernWidth ;

	__vr vrgout_ptr_k0_s0 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry), vrx_s0),
					   2,
					   (unsigned long)(pGOut+gOutIndex)) ;
	__vr vrgout_k0_s0 = _ve_vgtu_vvm(vrgout_ptr_k0_s0, vmall_s0) ;
	__vr vrgout_ptr_k1_s0 = _ve_vaddsl_vsv(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_s0) ;
	__vr vrgout_k1_s0 = _ve_vgtu_vvm(vrgout_ptr_k1_s0, vmall_s0) ;
	__vr vrgout_ptr_k2_s0 = _ve_vaddsl_vsv(4*2*gOutHeight*gOutWidth, vrgout_ptr_k0_s0) ;
	__vr vrgout_k2_s0 = _ve_vgtu_vvm(vrgout_ptr_k2_s0, vmall_s0) ;
	__vr vrgout_ptr_k3_s0 = _ve_vaddsl_vsv(4*3*gOutHeight*gOutWidth, vrgout_ptr_k0_s0) ;
	__vr vrgout_k3_s0 = _ve_vgtu_vvm(vrgout_ptr_k3_s0, vmall_s0) ;
	__vr vrgout_ptr_k4_s0 = _ve_vaddsl_vsv(4*4*gOutHeight*gOutWidth, vrgout_ptr_k0_s0) ;
	__vr vrgout_k4_s0 = _ve_vgtu_vvm(vrgout_ptr_k4_s0, vmall_s0) ;
	__vr vrgout_ptr_k5_s0 = _ve_vaddsl_vsv(4*5*gOutHeight*gOutWidth, vrgout_ptr_k0_s0) ;
	__vr vrgout_k5_s0 = _ve_vgtu_vvm(vrgout_ptr_k5_s0, vmall_s0) ;
	__vr vrgout_ptr_k6_s0 = _ve_vaddsl_vsv(4*6*gOutHeight*gOutWidth, vrgout_ptr_k0_s0) ;
	__vr vrgout_k6_s0 = _ve_vgtu_vvm(vrgout_ptr_k6_s0, vmall_s0) ;
	__vr vrgout_ptr_k7_s0 = _ve_vaddsl_vsv(4*7*gOutHeight*gOutWidth, vrgout_ptr_k0_s0) ;
	__vr vrgout_k7_s0 = _ve_vgtu_vvm(vrgout_ptr_k7_s0, vmall_s0) ;

	VFADD_C1(vrgout_k0_s0, vmall_s0, 0, r, 0)
	VFADD_C1(vrgout_k1_s0, vmall_s0, 1, r, 0)
	VFADD_C1(vrgout_k2_s0, vmall_s0, 2, r, 0)
	VFADD_C1(vrgout_k3_s0, vmall_s0, 3, r, 0)
	VFADD_C1(vrgout_k4_s0, vmall_s0, 4, r, 0)
	VFADD_C1(vrgout_k5_s0, vmall_s0, 5, r, 0)
	VFADD_C1(vrgout_k6_s0, vmall_s0, 6, r, 0)
	VFADD_C1(vrgout_k7_s0, vmall_s0, 7, r, 0)

	__vr vrgout_ptr_k0_s1 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry), vrx_s1),
					   2,
					   (unsigned long)(pGOut+gOutIndex)) ;
	__vr vrgout_k0_s1 = _ve_vgtu_vvm(vrgout_ptr_k0_s1, vmall_s1) ;
	__vr vrgout_ptr_k1_s1 = _ve_vaddsl_vsv(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_s1) ;
	__vr vrgout_k1_s1 = _ve_vgtu_vvm(vrgout_ptr_k1_s1, vmall_s1) ;
	__vr vrgout_ptr_k2_s1 = _ve_vaddsl_vsv(4*2*gOutHeight*gOutWidth, vrgout_ptr_k0_s1) ;
	__vr vrgout_k2_s1 = _ve_vgtu_vvm(vrgout_ptr_k2_s1, vmall_s1) ;
	__vr vrgout_ptr_k3_s1 = _ve_vaddsl_vsv(4*3*gOutHeight*gOutWidth, vrgout_ptr_k0_s1) ;
	__vr vrgout_k3_s1 = _ve_vgtu_vvm(vrgout_ptr_k3_s1, vmall_s1) ;
	__vr vrgout_ptr_k4_s1 = _ve_vaddsl_vsv(4*4*gOutHeight*gOutWidth, vrgout_ptr_k0_s1) ;
	__vr vrgout_k4_s1 = _ve_vgtu_vvm(vrgout_ptr_k4_s1, vmall_s1) ;
	__vr vrgout_ptr_k5_s1 = _ve_vaddsl_vsv(4*5*gOutHeight*gOutWidth, vrgout_ptr_k0_s1) ;
	__vr vrgout_k5_s1 = _ve_vgtu_vvm(vrgout_ptr_k5_s1, vmall_s1) ;
	__vr vrgout_ptr_k6_s1 = _ve_vaddsl_vsv(4*6*gOutHeight*gOutWidth, vrgout_ptr_k0_s1) ;
	__vr vrgout_k6_s1 = _ve_vgtu_vvm(vrgout_ptr_k6_s1, vmall_s1) ;
	__vr vrgout_ptr_k7_s1 = _ve_vaddsl_vsv(4*7*gOutHeight*gOutWidth, vrgout_ptr_k0_s1) ;
	__vr vrgout_k7_s1 = _ve_vgtu_vvm(vrgout_ptr_k7_s1, vmall_s1) ;

	VFADD_C1(vrgout_k0_s1, vmall_s1, 0, r, 1)
	VFADD_C1(vrgout_k1_s1, vmall_s1, 1, r, 1)
	VFADD_C1(vrgout_k2_s1, vmall_s1, 2, r, 1)
	VFADD_C1(vrgout_k3_s1, vmall_s1, 3, r, 1)
	VFADD_C1(vrgout_k4_s1, vmall_s1, 4, r, 1)
	VFADD_C1(vrgout_k5_s1, vmall_s1, 5, r, 1)
	VFADD_C1(vrgout_k6_s1, vmall_s1, 6, r, 1)
	VFADD_C1(vrgout_k7_s1, vmall_s1, 7, r, 1)


	__vr vrgout_ptr_k0_s2 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry), vrx_s2),
					   2,
					   (unsigned long)(pGOut+gOutIndex)) ;
	__vr vrgout_k0_s2 = _ve_vgtu_vvm(vrgout_ptr_k0_s2, vmall_s2) ;
	__vr vrgout_ptr_k1_s2 = _ve_vaddsl_vsv(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_s2) ;
	__vr vrgout_k1_s2 = _ve_vgtu_vvm(vrgout_ptr_k1_s2, vmall_s2) ;
	__vr vrgout_ptr_k2_s2 = _ve_vaddsl_vsv(4*2*gOutHeight*gOutWidth, vrgout_ptr_k0_s2) ;
	__vr vrgout_k2_s2 = _ve_vgtu_vvm(vrgout_ptr_k2_s2, vmall_s2) ;
	__vr vrgout_ptr_k3_s2 = _ve_vaddsl_vsv(4*3*gOutHeight*gOutWidth, vrgout_ptr_k0_s2) ;
	__vr vrgout_k3_s2 = _ve_vgtu_vvm(vrgout_ptr_k3_s2, vmall_s2) ;
	__vr vrgout_ptr_k4_s2 = _ve_vaddsl_vsv(4*4*gOutHeight*gOutWidth, vrgout_ptr_k0_s2) ;
	__vr vrgout_k4_s2 = _ve_vgtu_vvm(vrgout_ptr_k4_s2, vmall_s2) ;
	__vr vrgout_ptr_k5_s2 = _ve_vaddsl_vsv(4*5*gOutHeight*gOutWidth, vrgout_ptr_k0_s2) ;
	__vr vrgout_k5_s2 = _ve_vgtu_vvm(vrgout_ptr_k5_s2, vmall_s2) ;
	__vr vrgout_ptr_k6_s2 = _ve_vaddsl_vsv(4*6*gOutHeight*gOutWidth, vrgout_ptr_k0_s2) ;
	__vr vrgout_k6_s2 = _ve_vgtu_vvm(vrgout_ptr_k6_s2, vmall_s2) ;
	__vr vrgout_ptr_k7_s2 = _ve_vaddsl_vsv(4*7*gOutHeight*gOutWidth, vrgout_ptr_k0_s2) ;
	__vr vrgout_k7_s2 = _ve_vgtu_vvm(vrgout_ptr_k7_s2, vmall_s2) ;

	VFADD_C1(vrgout_k0_s2, vmall_s2, 0, r, 2)
	VFADD_C1(vrgout_k1_s2, vmall_s2, 1, r, 2)
	VFADD_C1(vrgout_k2_s2, vmall_s2, 2, r, 2)
	VFADD_C1(vrgout_k3_s2, vmall_s2, 3, r, 2)
	VFADD_C1(vrgout_k4_s2, vmall_s2, 4, r, 2)
	VFADD_C1(vrgout_k5_s2, vmall_s2, 5, r, 2)
	VFADD_C1(vrgout_k6_s2, vmall_s2, 6, r, 2)
	VFADD_C1(vrgout_k7_s2, vmall_s2, 7, r, 2)


	__vr vrgout_ptr_k0_s3 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry), vrx_s3),
					   2,
					   (unsigned long)(pGOut+gOutIndex)) ;
	__vr vrgout_k0_s3 = _ve_vgtu_vvm(vrgout_ptr_k0_s3, vmall_s3) ;
	__vr vrgout_ptr_k1_s3 = _ve_vaddsl_vsv(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_s3) ;
	__vr vrgout_k1_s3 = _ve_vgtu_vvm(vrgout_ptr_k1_s3, vmall_s3) ;
	__vr vrgout_ptr_k2_s3 = _ve_vaddsl_vsv(4*2*gOutHeight*gOutWidth, vrgout_ptr_k0_s3) ;
	__vr vrgout_k2_s3 = _ve_vgtu_vvm(vrgout_ptr_k2_s3, vmall_s3) ;
	__vr vrgout_ptr_k3_s3 = _ve_vaddsl_vsv(4*3*gOutHeight*gOutWidth, vrgout_ptr_k0_s3) ;
	__vr vrgout_k3_s3 = _ve_vgtu_vvm(vrgout_ptr_k3_s3, vmall_s3) ;
	__vr vrgout_ptr_k4_s3 = _ve_vaddsl_vsv(4*4*gOutHeight*gOutWidth, vrgout_ptr_k0_s3) ;
	__vr vrgout_k4_s3 = _ve_vgtu_vvm(vrgout_ptr_k4_s3, vmall_s3) ;
	__vr vrgout_ptr_k5_s3 = _ve_vaddsl_vsv(4*5*gOutHeight*gOutWidth, vrgout_ptr_k0_s3) ;
	__vr vrgout_k5_s3 = _ve_vgtu_vvm(vrgout_ptr_k5_s3, vmall_s3) ;
	__vr vrgout_ptr_k6_s3 = _ve_vaddsl_vsv(4*6*gOutHeight*gOutWidth, vrgout_ptr_k0_s3) ;
	__vr vrgout_k6_s3 = _ve_vgtu_vvm(vrgout_ptr_k6_s3, vmall_s3) ;
	__vr vrgout_ptr_k7_s3 = _ve_vaddsl_vsv(4*7*gOutHeight*gOutWidth, vrgout_ptr_k0_s3) ;
	__vr vrgout_k7_s3 = _ve_vgtu_vvm(vrgout_ptr_k7_s3, vmall_s3) ;

	VFADD_C1(vrgout_k0_s3, vmall_s3, 0, r, 3)
	VFADD_C1(vrgout_k1_s3, vmall_s3, 1, r, 3)
	VFADD_C1(vrgout_k2_s3, vmall_s3, 2, r, 3)
	VFADD_C1(vrgout_k3_s3, vmall_s3, 3, r, 3)
	VFADD_C1(vrgout_k4_s3, vmall_s3, 4, r, 3)
	VFADD_C1(vrgout_k5_s3, vmall_s3, 5, r, 3)
	VFADD_C1(vrgout_k6_s3, vmall_s3, 6, r, 3)
	VFADD_C1(vrgout_k7_s3, vmall_s3, 7, r, 3)


	__vr vrgout_ptr_k0_s4 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry), vrx_s4),
					   2,
					   (unsigned long)(pGOut+gOutIndex)) ;
	__vr vrgout_k0_s4 = _ve_vgtu_vvm(vrgout_ptr_k0_s4, vmall_s4) ;
	__vr vrgout_ptr_k1_s4 = _ve_vaddsl_vsv(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_s4) ;
	__vr vrgout_k1_s4 = _ve_vgtu_vvm(vrgout_ptr_k1_s4, vmall_s4) ;
	__vr vrgout_ptr_k2_s4 = _ve_vaddsl_vsv(4*2*gOutHeight*gOutWidth, vrgout_ptr_k0_s4) ;
	__vr vrgout_k2_s4 = _ve_vgtu_vvm(vrgout_ptr_k2_s4, vmall_s4) ;
	__vr vrgout_ptr_k3_s4 = _ve_vaddsl_vsv(4*3*gOutHeight*gOutWidth, vrgout_ptr_k0_s4) ;
	__vr vrgout_k3_s4 = _ve_vgtu_vvm(vrgout_ptr_k3_s4, vmall_s4) ;
	__vr vrgout_ptr_k4_s4 = _ve_vaddsl_vsv(4*4*gOutHeight*gOutWidth, vrgout_ptr_k0_s4) ;
	__vr vrgout_k4_s4 = _ve_vgtu_vvm(vrgout_ptr_k4_s4, vmall_s4) ;
	__vr vrgout_ptr_k5_s4 = _ve_vaddsl_vsv(4*5*gOutHeight*gOutWidth, vrgout_ptr_k0_s4) ;
	__vr vrgout_k5_s4 = _ve_vgtu_vvm(vrgout_ptr_k5_s4, vmall_s4) ;
	__vr vrgout_ptr_k6_s4 = _ve_vaddsl_vsv(4*6*gOutHeight*gOutWidth, vrgout_ptr_k0_s4) ;
	__vr vrgout_k6_s4 = _ve_vgtu_vvm(vrgout_ptr_k6_s4, vmall_s4) ;
	__vr vrgout_ptr_k7_s4 = _ve_vaddsl_vsv(4*7*gOutHeight*gOutWidth, vrgout_ptr_k0_s4) ;
	__vr vrgout_k7_s4 = _ve_vgtu_vvm(vrgout_ptr_k7_s4, vmall_s4) ;

	VFADD_C1(vrgout_k0_s4, vmall_s4, 0, r, 4)
	VFADD_C1(vrgout_k1_s4, vmall_s4, 1, r, 4)
	VFADD_C1(vrgout_k2_s4, vmall_s4, 2, r, 4)
	VFADD_C1(vrgout_k3_s4, vmall_s4, 3, r, 4)
	VFADD_C1(vrgout_k4_s4, vmall_s4, 4, r, 4)
	VFADD_C1(vrgout_k5_s4, vmall_s4, 5, r, 4)
	VFADD_C1(vrgout_k6_s4, vmall_s4, 6, r, 4)
	VFADD_C1(vrgout_k7_s4, vmall_s4, 7, r, 4)

#undef VFADD_C1
      } // gOutChannel
    } // kernHeight

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

    __vr vrj_s0 = _ve_vaddsl_vsv(padWidth-0*dilationWidth, vrw) ;
    __vr vrj_s1 = _ve_vaddsl_vsv(padWidth-1*dilationWidth, vrw) ;
    __vr vrj_s2 = _ve_vaddsl_vsv(padWidth-2*dilationWidth, vrw) ;
    __vr vrj_s3 = _ve_vaddsl_vsv(padWidth-3*dilationWidth, vrw) ;
    __vr vrj_s4 = _ve_vaddsl_vsv(padWidth-4*dilationWidth, vrw) ;

    __vr vrx_s0 = _ve_vdivsl_vvs(vrj_s0, strideWidth) ;
    __vr vrx_s1 = _ve_vdivsl_vvs(vrj_s1, strideWidth) ;
    __vr vrx_s2 = _ve_vdivsl_vvs(vrj_s2, strideWidth) ;
    __vr vrx_s3 = _ve_vdivsl_vvs(vrj_s3, strideWidth) ;
    __vr vrx_s4 = _ve_vdivsl_vvs(vrj_s4, strideWidth) ;

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

    __vm256 vmx0_s3 = _ve_vfmkl_mcv(VECC_IEQ, _ve_vcmpsl_vvv(vrj_s3, _ve_vmulsl_vsv(strideWidth, vrx_s3))) ;
    __vm256 vmx1_s3 = _ve_vfmkl_mcv(VECC_GE, vrx_s3) ;
    __vm256 vmx2_s3 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(gOutWidth,vrx_s3)) ;
    __vm256 vmx_s3 = _ve_andm_mmm(_ve_andm_mmm(vmx0_s3, vmx1_s3), vmx2_s3) ;

    __vm256 vmx0_s4 = _ve_vfmkl_mcv(VECC_IEQ, _ve_vcmpsl_vvv(vrj_s4, _ve_vmulsl_vsv(strideWidth, vrx_s4))) ;
    __vm256 vmx1_s4 = _ve_vfmkl_mcv(VECC_GE, vrx_s4) ;
    __vm256 vmx2_s4 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(gOutWidth,vrx_s4)) ;
    __vm256 vmx_s4 = _ve_andm_mmm(_ve_andm_mmm(vmx0_s4, vmx1_s4), vmx2_s4) ;

    for (int64_t r=0; r<kernHeight; r++) {
      __vr vri = _ve_vaddsl_vsv(padHeight-r*dilationHeight+h, vrh) ;
      __vr vry = _ve_vdivsl_vvs(vri, strideHeight) ;

      __vm256 vmy0 = _ve_vfmkl_mcv(VECC_IEQ, _ve_vcmpsl_vvv(vri, _ve_vmulsl_vsv(strideHeight, vry))) ;
      __vm256 vmy1 = _ve_vfmkl_mcv(VECC_GE, vry) ;
      __vm256 vmy2 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(gOutHeight,vry)) ;

      __vm256 vmy = _ve_andm_mmm(_ve_andm_mmm(vmy0, vmy1), vmy2) ;


      __vm256 vmall_s0 = _ve_andm_mmm(vmy,vmx_s0) ;
      __vm256 vmall_s1 = _ve_andm_mmm(vmy,vmx_s1) ;
      __vm256 vmall_s2 = _ve_andm_mmm(vmy,vmx_s2) ;
      __vm256 vmall_s3 = _ve_andm_mmm(vmy,vmx_s3) ;
      __vm256 vmall_s4 = _ve_andm_mmm(vmy,vmx_s4) ;

      int64_t k=0;
      if( (gOutChannelGroup & 0x01 ) == 1 ) {
	int64_t gOutIndex    = gOutGroupOffset + ((n * gOutChannel + k) * gOutHeight) * gOutWidth ;
	const float *pKerValue = pKernel + kernGroupOffset + (((k  ) * gInChannelGroup + c) * kernHeight) * kernWidth ;


#define VFADD_C2(VRGOUT, VM, K, R, S)  {												\
	  const uint64_t kerValue01 = _ve_pack_f32p(pKerValue + (((K)*gInChannelGroup + 0) * kernHeight +(R)) * kernWidth + (S),	\
						    pKerValue + (((K)*gInChannelGroup + 1) * kernHeight +(R)) * kernWidth + (S)) ;	\
	  VRGOUT = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), VRGOUT, VM) ;				\
	  __vr vrgoutP = _ve_vshf_vvvs(VRGOUT, VRGOUT, VE_VSHUFFLE_YUZU) ;			\
	  vrsum01 = _ve_pvfmad_vvsv(vrsum01, kerValue01, vrgoutP) ;				\
	}

	__vr vrgout_ptr_k0_s0 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry), vrx_s0),
					   2,
					   (unsigned long)(pGOut+gOutIndex)) ;
	__vr vrgout_k0_s0 = _ve_vgtu_vvm(vrgout_ptr_k0_s0, vmall_s0) ;
	__vr vrgout_ptr_k1_s0 = _ve_vaddsl_vsv(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_s0) ;
	__vr vrgout_k1_s0 = _ve_vgtu_vvm(vrgout_ptr_k1_s0, vmall_s0) ;

	VFADD_C2(vrgout_k0_s0, vmall_s0, 0, r, 0)

	__vr vrgout_ptr_k0_s1 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry), vrx_s1),
					   2,
					   (unsigned long)(pGOut+gOutIndex)) ;
	__vr vrgout_k0_s1 = _ve_vgtu_vvm(vrgout_ptr_k0_s1, vmall_s1) ;

	VFADD_C2(vrgout_k0_s1, vmall_s1, 0, r, 1)


	__vr vrgout_ptr_k0_s2 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry), vrx_s2),
					   2,
					   (unsigned long)(pGOut+gOutIndex)) ;
	__vr vrgout_k0_s2 = _ve_vgtu_vvm(vrgout_ptr_k0_s2, vmall_s2) ;

	VFADD_C2(vrgout_k0_s2, vmall_s2, 0, r, 2)

	__vr vrgout_ptr_k0_s3 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry), vrx_s3),
					   2,
					   (unsigned long)(pGOut+gOutIndex)) ;
	__vr vrgout_k0_s3 = _ve_vgtu_vvm(vrgout_ptr_k0_s3, vmall_s3) ;
	__vr vrgout_ptr_k1_s3 = _ve_vaddsl_vsv(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_s3) ;

	VFADD_C2(vrgout_k0_s3, vmall_s3, 0, r, 3)

	__vr vrgout_ptr_k0_s4 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry), vrx_s4),
					   2,
					   (unsigned long)(pGOut+gOutIndex)) ;
	__vr vrgout_k0_s4 = _ve_vgtu_vvm(vrgout_ptr_k0_s4, vmall_s4) ;

	VFADD_C2(vrgout_k0_s4, vmall_s4, 0, r, 4)

	k+=1 ;
      }
      if( ((gOutChannelGroup >> 1) & 0x01 ) == 1 ) {
	int64_t gOutIndex    = gOutGroupOffset + ((n * gOutChannel + k) * gOutHeight) * gOutWidth ;
	const float *pKerValue = pKernel + kernGroupOffset + (((k  ) * gInChannelGroup + c) * kernHeight) * kernWidth ;

	__vr vrgout_ptr_k0_s0 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry), vrx_s0),
					   2,
					   (unsigned long)(pGOut+gOutIndex)) ;
	__vr vrgout_k0_s0 = _ve_vgtu_vvm(vrgout_ptr_k0_s0, vmall_s0) ;
	__vr vrgout_ptr_k1_s0 = _ve_vaddsl_vsv(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_s0) ;
	__vr vrgout_k1_s0 = _ve_vgtu_vvm(vrgout_ptr_k1_s0, vmall_s0) ;

	VFADD_C2(vrgout_k0_s0, vmall_s0, 0, r, 0)
	VFADD_C2(vrgout_k1_s0, vmall_s0, 1, r, 0)

	__vr vrgout_ptr_k0_s1 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry), vrx_s1),
					   2,
					   (unsigned long)(pGOut+gOutIndex)) ;
	__vr vrgout_k0_s1 = _ve_vgtu_vvm(vrgout_ptr_k0_s1, vmall_s1) ;
	__vr vrgout_ptr_k1_s1 = _ve_vaddsl_vsv(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_s1) ;
	__vr vrgout_k1_s1 = _ve_vgtu_vvm(vrgout_ptr_k1_s1, vmall_s1) ;

	VFADD_C2(vrgout_k0_s1, vmall_s1, 0, r, 1)
	VFADD_C2(vrgout_k1_s1, vmall_s1, 1, r, 1)


	__vr vrgout_ptr_k0_s2 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry), vrx_s2),
					   2,
					   (unsigned long)(pGOut+gOutIndex)) ;
	__vr vrgout_k0_s2 = _ve_vgtu_vvm(vrgout_ptr_k0_s2, vmall_s2) ;
	__vr vrgout_ptr_k1_s2 = _ve_vaddsl_vsv(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_s2) ;
	__vr vrgout_k1_s2 = _ve_vgtu_vvm(vrgout_ptr_k1_s2, vmall_s2) ;

	VFADD_C2(vrgout_k0_s2, vmall_s2, 0, r, 2)
	VFADD_C2(vrgout_k1_s2, vmall_s2, 1, r, 2)

	__vr vrgout_ptr_k0_s3 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry), vrx_s3),
					   2,
					   (unsigned long)(pGOut+gOutIndex)) ;
	__vr vrgout_k0_s3 = _ve_vgtu_vvm(vrgout_ptr_k0_s3, vmall_s3) ;
	__vr vrgout_ptr_k1_s3 = _ve_vaddsl_vsv(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_s3) ;
	__vr vrgout_k1_s3 = _ve_vgtu_vvm(vrgout_ptr_k1_s3, vmall_s3) ;
	__vr vrgout_ptr_k2_s3 = _ve_vaddsl_vsv(4*2*gOutHeight*gOutWidth, vrgout_ptr_k0_s3) ;

	VFADD_C2(vrgout_k0_s3, vmall_s3, 0, r, 3)
	VFADD_C2(vrgout_k1_s3, vmall_s3, 1, r, 3)

	__vr vrgout_ptr_k0_s4 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry), vrx_s4),
					   2,
					   (unsigned long)(pGOut+gOutIndex)) ;
	__vr vrgout_k0_s4 = _ve_vgtu_vvm(vrgout_ptr_k0_s4, vmall_s4) ;
	__vr vrgout_ptr_k1_s4 = _ve_vaddsl_vsv(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_s4) ;
	__vr vrgout_k1_s4 = _ve_vgtu_vvm(vrgout_ptr_k1_s4, vmall_s4) ;

	VFADD_C2(vrgout_k0_s4, vmall_s4, 0, r, 4)
	VFADD_C2(vrgout_k1_s4, vmall_s4, 1, r, 4)

	k+=2 ;
      }
      if( ((gOutChannelGroup >> 2) & 0x01 ) == 1 ) {
	int64_t gOutIndex    = gOutGroupOffset + ((n * gOutChannel + k) * gOutHeight) * gOutWidth ;
	const float *pKerValue = pKernel + kernGroupOffset + (((k  ) * gInChannelGroup + c) * kernHeight) * kernWidth ;

	__vr vrgout_ptr_k0_s0 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry), vrx_s0),
					   2,
					   (unsigned long)(pGOut+gOutIndex)) ;
	__vr vrgout_k0_s0 = _ve_vgtu_vvm(vrgout_ptr_k0_s0, vmall_s0) ;
	__vr vrgout_ptr_k1_s0 = _ve_vaddsl_vsv(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_s0) ;
	__vr vrgout_k1_s0 = _ve_vgtu_vvm(vrgout_ptr_k1_s0, vmall_s0) ;
	__vr vrgout_ptr_k2_s0 = _ve_vaddsl_vsv(4*2*gOutHeight*gOutWidth, vrgout_ptr_k0_s0) ;
	__vr vrgout_k2_s0 = _ve_vgtu_vvm(vrgout_ptr_k2_s0, vmall_s0) ;
	__vr vrgout_ptr_k3_s0 = _ve_vaddsl_vsv(4*3*gOutHeight*gOutWidth, vrgout_ptr_k0_s0) ;
	__vr vrgout_k3_s0 = _ve_vgtu_vvm(vrgout_ptr_k3_s0, vmall_s0) ;

	VFADD_C2(vrgout_k0_s0, vmall_s0, 0, r, 0)
	VFADD_C2(vrgout_k1_s0, vmall_s0, 1, r, 0)
	VFADD_C2(vrgout_k2_s0, vmall_s0, 2, r, 0)
	VFADD_C2(vrgout_k3_s0, vmall_s0, 3, r, 0)

	__vr vrgout_ptr_k0_s1 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry), vrx_s1),
					   2,
					   (unsigned long)(pGOut+gOutIndex)) ;
	__vr vrgout_k0_s1 = _ve_vgtu_vvm(vrgout_ptr_k0_s1, vmall_s1) ;
	__vr vrgout_ptr_k1_s1 = _ve_vaddsl_vsv(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_s1) ;
	__vr vrgout_k1_s1 = _ve_vgtu_vvm(vrgout_ptr_k1_s1, vmall_s1) ;
	__vr vrgout_ptr_k2_s1 = _ve_vaddsl_vsv(4*2*gOutHeight*gOutWidth, vrgout_ptr_k0_s1) ;
	__vr vrgout_k2_s1 = _ve_vgtu_vvm(vrgout_ptr_k2_s1, vmall_s1) ;
	__vr vrgout_ptr_k3_s1 = _ve_vaddsl_vsv(4*3*gOutHeight*gOutWidth, vrgout_ptr_k0_s1) ;
	__vr vrgout_k3_s1 = _ve_vgtu_vvm(vrgout_ptr_k3_s1, vmall_s1) ;

	VFADD_C2(vrgout_k0_s1, vmall_s1, 0, r, 1)
	VFADD_C2(vrgout_k1_s1, vmall_s1, 1, r, 1)
	VFADD_C2(vrgout_k2_s1, vmall_s1, 2, r, 1)
	VFADD_C2(vrgout_k3_s1, vmall_s1, 3, r, 1)


	__vr vrgout_ptr_k0_s2 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry), vrx_s2),
					   2,
					   (unsigned long)(pGOut+gOutIndex)) ;
	__vr vrgout_k0_s2 = _ve_vgtu_vvm(vrgout_ptr_k0_s2, vmall_s2) ;
	__vr vrgout_ptr_k1_s2 = _ve_vaddsl_vsv(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_s2) ;
	__vr vrgout_k1_s2 = _ve_vgtu_vvm(vrgout_ptr_k1_s2, vmall_s2) ;
	__vr vrgout_ptr_k2_s2 = _ve_vaddsl_vsv(4*2*gOutHeight*gOutWidth, vrgout_ptr_k0_s2) ;
	__vr vrgout_k2_s2 = _ve_vgtu_vvm(vrgout_ptr_k2_s2, vmall_s2) ;
	__vr vrgout_ptr_k3_s2 = _ve_vaddsl_vsv(4*3*gOutHeight*gOutWidth, vrgout_ptr_k0_s2) ;
	__vr vrgout_k3_s2 = _ve_vgtu_vvm(vrgout_ptr_k3_s2, vmall_s2) ;

	VFADD_C2(vrgout_k0_s2, vmall_s2, 0, r, 2)
	VFADD_C2(vrgout_k1_s2, vmall_s2, 1, r, 2)
	VFADD_C2(vrgout_k2_s2, vmall_s2, 2, r, 2)
	VFADD_C2(vrgout_k3_s2, vmall_s2, 3, r, 2)

	__vr vrgout_ptr_k0_s3 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry), vrx_s3),
					   2,
					   (unsigned long)(pGOut+gOutIndex)) ;
	__vr vrgout_k0_s3 = _ve_vgtu_vvm(vrgout_ptr_k0_s3, vmall_s3) ;
	__vr vrgout_ptr_k1_s3 = _ve_vaddsl_vsv(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_s3) ;
	__vr vrgout_k1_s3 = _ve_vgtu_vvm(vrgout_ptr_k1_s3, vmall_s3) ;
	__vr vrgout_ptr_k2_s3 = _ve_vaddsl_vsv(4*2*gOutHeight*gOutWidth, vrgout_ptr_k0_s3) ;
	__vr vrgout_k2_s3 = _ve_vgtu_vvm(vrgout_ptr_k2_s3, vmall_s3) ;
	__vr vrgout_ptr_k3_s3 = _ve_vaddsl_vsv(4*3*gOutHeight*gOutWidth, vrgout_ptr_k0_s3) ;
	__vr vrgout_k3_s3 = _ve_vgtu_vvm(vrgout_ptr_k3_s3, vmall_s3) ;
	__vr vrgout_ptr_k4_s3 = _ve_vaddsl_vsv(4*4*gOutHeight*gOutWidth, vrgout_ptr_k0_s3) ;

	VFADD_C2(vrgout_k0_s3, vmall_s3, 0, r, 3)
	VFADD_C2(vrgout_k1_s3, vmall_s3, 1, r, 3)
	VFADD_C2(vrgout_k2_s3, vmall_s3, 2, r, 3)
	VFADD_C2(vrgout_k3_s3, vmall_s3, 3, r, 3)

	__vr vrgout_ptr_k0_s4 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry), vrx_s4),
					   2,
					   (unsigned long)(pGOut+gOutIndex)) ;
	__vr vrgout_k0_s4 = _ve_vgtu_vvm(vrgout_ptr_k0_s4, vmall_s4) ;
	__vr vrgout_ptr_k1_s4 = _ve_vaddsl_vsv(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_s4) ;
	__vr vrgout_k1_s4 = _ve_vgtu_vvm(vrgout_ptr_k1_s4, vmall_s4) ;
	__vr vrgout_ptr_k2_s4 = _ve_vaddsl_vsv(4*2*gOutHeight*gOutWidth, vrgout_ptr_k0_s4) ;
	__vr vrgout_k2_s4 = _ve_vgtu_vvm(vrgout_ptr_k2_s4, vmall_s4) ;
	__vr vrgout_ptr_k3_s4 = _ve_vaddsl_vsv(4*3*gOutHeight*gOutWidth, vrgout_ptr_k0_s4) ;
	__vr vrgout_k3_s4 = _ve_vgtu_vvm(vrgout_ptr_k3_s4, vmall_s4) ;

	VFADD_C2(vrgout_k0_s4, vmall_s4, 0, r, 4)
	VFADD_C2(vrgout_k1_s4, vmall_s4, 1, r, 4)
	VFADD_C2(vrgout_k2_s4, vmall_s4, 2, r, 4)
	VFADD_C2(vrgout_k3_s4, vmall_s4, 3, r, 4)

	k+=4 ;
      }
      for (; k<gOutChannelGroup; k+=8) {
	int64_t gOutIndex    = gOutGroupOffset + ((n * gOutChannel + k) * gOutHeight) * gOutWidth ;
	const float *pKerValue = pKernel + kernGroupOffset + (((k  ) * gInChannelGroup + c) * kernHeight) * kernWidth ;

	__vr vrgout_ptr_k0_s0 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry), vrx_s0),
					   2,
					   (unsigned long)(pGOut+gOutIndex)) ;
	__vr vrgout_k0_s0 = _ve_vgtu_vvm(vrgout_ptr_k0_s0, vmall_s0) ;
	__vr vrgout_ptr_k1_s0 = _ve_vaddsl_vsv(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_s0) ;
	__vr vrgout_k1_s0 = _ve_vgtu_vvm(vrgout_ptr_k1_s0, vmall_s0) ;
	__vr vrgout_ptr_k2_s0 = _ve_vaddsl_vsv(4*2*gOutHeight*gOutWidth, vrgout_ptr_k0_s0) ;
	__vr vrgout_k2_s0 = _ve_vgtu_vvm(vrgout_ptr_k2_s0, vmall_s0) ;
	__vr vrgout_ptr_k3_s0 = _ve_vaddsl_vsv(4*3*gOutHeight*gOutWidth, vrgout_ptr_k0_s0) ;
	__vr vrgout_k3_s0 = _ve_vgtu_vvm(vrgout_ptr_k3_s0, vmall_s0) ;
	__vr vrgout_ptr_k4_s0 = _ve_vaddsl_vsv(4*4*gOutHeight*gOutWidth, vrgout_ptr_k0_s0) ;
	__vr vrgout_k4_s0 = _ve_vgtu_vvm(vrgout_ptr_k4_s0, vmall_s0) ;
	__vr vrgout_ptr_k5_s0 = _ve_vaddsl_vsv(4*5*gOutHeight*gOutWidth, vrgout_ptr_k0_s0) ;
	__vr vrgout_k5_s0 = _ve_vgtu_vvm(vrgout_ptr_k5_s0, vmall_s0) ;
	__vr vrgout_ptr_k6_s0 = _ve_vaddsl_vsv(4*6*gOutHeight*gOutWidth, vrgout_ptr_k0_s0) ;
	__vr vrgout_k6_s0 = _ve_vgtu_vvm(vrgout_ptr_k6_s0, vmall_s0) ;
	__vr vrgout_ptr_k7_s0 = _ve_vaddsl_vsv(4*7*gOutHeight*gOutWidth, vrgout_ptr_k0_s0) ;
	__vr vrgout_k7_s0 = _ve_vgtu_vvm(vrgout_ptr_k7_s0, vmall_s0) ;

	VFADD_C2(vrgout_k0_s0, vmall_s0, 0, r, 0)
	VFADD_C2(vrgout_k1_s0, vmall_s0, 1, r, 0)
	VFADD_C2(vrgout_k2_s0, vmall_s0, 2, r, 0)
	VFADD_C2(vrgout_k3_s0, vmall_s0, 3, r, 0)
	VFADD_C2(vrgout_k4_s0, vmall_s0, 4, r, 0)
	VFADD_C2(vrgout_k5_s0, vmall_s0, 5, r, 0)
	VFADD_C2(vrgout_k6_s0, vmall_s0, 6, r, 0)
	VFADD_C2(vrgout_k7_s0, vmall_s0, 7, r, 0)

	__vr vrgout_ptr_k0_s1 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry), vrx_s1),
					   2,
					   (unsigned long)(pGOut+gOutIndex)) ;
	__vr vrgout_k0_s1 = _ve_vgtu_vvm(vrgout_ptr_k0_s1, vmall_s1) ;
	__vr vrgout_ptr_k1_s1 = _ve_vaddsl_vsv(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_s1) ;
	__vr vrgout_k1_s1 = _ve_vgtu_vvm(vrgout_ptr_k1_s1, vmall_s1) ;
	__vr vrgout_ptr_k2_s1 = _ve_vaddsl_vsv(4*2*gOutHeight*gOutWidth, vrgout_ptr_k0_s1) ;
	__vr vrgout_k2_s1 = _ve_vgtu_vvm(vrgout_ptr_k2_s1, vmall_s1) ;
	__vr vrgout_ptr_k3_s1 = _ve_vaddsl_vsv(4*3*gOutHeight*gOutWidth, vrgout_ptr_k0_s1) ;
	__vr vrgout_k3_s1 = _ve_vgtu_vvm(vrgout_ptr_k3_s1, vmall_s1) ;
	__vr vrgout_ptr_k4_s1 = _ve_vaddsl_vsv(4*4*gOutHeight*gOutWidth, vrgout_ptr_k0_s1) ;
	__vr vrgout_k4_s1 = _ve_vgtu_vvm(vrgout_ptr_k4_s1, vmall_s1) ;
	__vr vrgout_ptr_k5_s1 = _ve_vaddsl_vsv(4*5*gOutHeight*gOutWidth, vrgout_ptr_k0_s1) ;
	__vr vrgout_k5_s1 = _ve_vgtu_vvm(vrgout_ptr_k5_s1, vmall_s1) ;
	__vr vrgout_ptr_k6_s1 = _ve_vaddsl_vsv(4*6*gOutHeight*gOutWidth, vrgout_ptr_k0_s1) ;
	__vr vrgout_k6_s1 = _ve_vgtu_vvm(vrgout_ptr_k6_s1, vmall_s1) ;
	__vr vrgout_ptr_k7_s1 = _ve_vaddsl_vsv(4*7*gOutHeight*gOutWidth, vrgout_ptr_k0_s1) ;
	__vr vrgout_k7_s1 = _ve_vgtu_vvm(vrgout_ptr_k7_s1, vmall_s1) ;

	VFADD_C2(vrgout_k0_s1, vmall_s1, 0, r, 1)
	VFADD_C2(vrgout_k1_s1, vmall_s1, 1, r, 1)
	VFADD_C2(vrgout_k2_s1, vmall_s1, 2, r, 1)
	VFADD_C2(vrgout_k3_s1, vmall_s1, 3, r, 1)
	VFADD_C2(vrgout_k4_s1, vmall_s1, 4, r, 1)
	VFADD_C2(vrgout_k5_s1, vmall_s1, 5, r, 1)
	VFADD_C2(vrgout_k6_s1, vmall_s1, 6, r, 1)
	VFADD_C2(vrgout_k7_s1, vmall_s1, 7, r, 1)


	__vr vrgout_ptr_k0_s2 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry), vrx_s2),
					   2,
					   (unsigned long)(pGOut+gOutIndex)) ;
	__vr vrgout_k0_s2 = _ve_vgtu_vvm(vrgout_ptr_k0_s2, vmall_s2) ;
	__vr vrgout_ptr_k1_s2 = _ve_vaddsl_vsv(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_s2) ;
	__vr vrgout_k1_s2 = _ve_vgtu_vvm(vrgout_ptr_k1_s2, vmall_s2) ;
	__vr vrgout_ptr_k2_s2 = _ve_vaddsl_vsv(4*2*gOutHeight*gOutWidth, vrgout_ptr_k0_s2) ;
	__vr vrgout_k2_s2 = _ve_vgtu_vvm(vrgout_ptr_k2_s2, vmall_s2) ;
	__vr vrgout_ptr_k3_s2 = _ve_vaddsl_vsv(4*3*gOutHeight*gOutWidth, vrgout_ptr_k0_s2) ;
	__vr vrgout_k3_s2 = _ve_vgtu_vvm(vrgout_ptr_k3_s2, vmall_s2) ;
	__vr vrgout_ptr_k4_s2 = _ve_vaddsl_vsv(4*4*gOutHeight*gOutWidth, vrgout_ptr_k0_s2) ;
	__vr vrgout_k4_s2 = _ve_vgtu_vvm(vrgout_ptr_k4_s2, vmall_s2) ;
	__vr vrgout_ptr_k5_s2 = _ve_vaddsl_vsv(4*5*gOutHeight*gOutWidth, vrgout_ptr_k0_s2) ;
	__vr vrgout_k5_s2 = _ve_vgtu_vvm(vrgout_ptr_k5_s2, vmall_s2) ;
	__vr vrgout_ptr_k6_s2 = _ve_vaddsl_vsv(4*6*gOutHeight*gOutWidth, vrgout_ptr_k0_s2) ;
	__vr vrgout_k6_s2 = _ve_vgtu_vvm(vrgout_ptr_k6_s2, vmall_s2) ;
	__vr vrgout_ptr_k7_s2 = _ve_vaddsl_vsv(4*7*gOutHeight*gOutWidth, vrgout_ptr_k0_s2) ;
	__vr vrgout_k7_s2 = _ve_vgtu_vvm(vrgout_ptr_k7_s2, vmall_s2) ;

	VFADD_C2(vrgout_k0_s2, vmall_s2, 0, r, 2)
	VFADD_C2(vrgout_k1_s2, vmall_s2, 1, r, 2)
	VFADD_C2(vrgout_k2_s2, vmall_s2, 2, r, 2)
	VFADD_C2(vrgout_k3_s2, vmall_s2, 3, r, 2)
	VFADD_C2(vrgout_k4_s2, vmall_s2, 4, r, 2)
	VFADD_C2(vrgout_k5_s2, vmall_s2, 5, r, 2)
	VFADD_C2(vrgout_k6_s2, vmall_s2, 6, r, 2)
	VFADD_C2(vrgout_k7_s2, vmall_s2, 7, r, 2)


	__vr vrgout_ptr_k0_s3 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry), vrx_s3),
					   2,
					   (unsigned long)(pGOut+gOutIndex)) ;
	__vr vrgout_k0_s3 = _ve_vgtu_vvm(vrgout_ptr_k0_s3, vmall_s3) ;
	__vr vrgout_ptr_k1_s3 = _ve_vaddsl_vsv(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_s3) ;
	__vr vrgout_k1_s3 = _ve_vgtu_vvm(vrgout_ptr_k1_s3, vmall_s3) ;
	__vr vrgout_ptr_k2_s3 = _ve_vaddsl_vsv(4*2*gOutHeight*gOutWidth, vrgout_ptr_k0_s3) ;
	__vr vrgout_k2_s3 = _ve_vgtu_vvm(vrgout_ptr_k2_s3, vmall_s3) ;
	__vr vrgout_ptr_k3_s3 = _ve_vaddsl_vsv(4*3*gOutHeight*gOutWidth, vrgout_ptr_k0_s3) ;
	__vr vrgout_k3_s3 = _ve_vgtu_vvm(vrgout_ptr_k3_s3, vmall_s3) ;
	__vr vrgout_ptr_k4_s3 = _ve_vaddsl_vsv(4*4*gOutHeight*gOutWidth, vrgout_ptr_k0_s3) ;
	__vr vrgout_k4_s3 = _ve_vgtu_vvm(vrgout_ptr_k4_s3, vmall_s3) ;
	__vr vrgout_ptr_k5_s3 = _ve_vaddsl_vsv(4*5*gOutHeight*gOutWidth, vrgout_ptr_k0_s3) ;
	__vr vrgout_k5_s3 = _ve_vgtu_vvm(vrgout_ptr_k5_s3, vmall_s3) ;
	__vr vrgout_ptr_k6_s3 = _ve_vaddsl_vsv(4*6*gOutHeight*gOutWidth, vrgout_ptr_k0_s3) ;
	__vr vrgout_k6_s3 = _ve_vgtu_vvm(vrgout_ptr_k6_s3, vmall_s3) ;
	__vr vrgout_ptr_k7_s3 = _ve_vaddsl_vsv(4*7*gOutHeight*gOutWidth, vrgout_ptr_k0_s3) ;
	__vr vrgout_k7_s3 = _ve_vgtu_vvm(vrgout_ptr_k7_s3, vmall_s3) ;

	VFADD_C2(vrgout_k0_s3, vmall_s3, 0, r, 3)
	VFADD_C2(vrgout_k1_s3, vmall_s3, 1, r, 3)
	VFADD_C2(vrgout_k2_s3, vmall_s3, 2, r, 3)
	VFADD_C2(vrgout_k3_s3, vmall_s3, 3, r, 3)
	VFADD_C2(vrgout_k4_s3, vmall_s3, 4, r, 3)
	VFADD_C2(vrgout_k5_s3, vmall_s3, 5, r, 3)
	VFADD_C2(vrgout_k6_s3, vmall_s3, 6, r, 3)
	VFADD_C2(vrgout_k7_s3, vmall_s3, 7, r, 3)


	__vr vrgout_ptr_k0_s4 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry), vrx_s4),
					   2,
					   (unsigned long)(pGOut+gOutIndex)) ;
	__vr vrgout_k0_s4 = _ve_vgtu_vvm(vrgout_ptr_k0_s4, vmall_s4) ;
	__vr vrgout_ptr_k1_s4 = _ve_vaddsl_vsv(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_s4) ;
	__vr vrgout_k1_s4 = _ve_vgtu_vvm(vrgout_ptr_k1_s4, vmall_s4) ;
	__vr vrgout_ptr_k2_s4 = _ve_vaddsl_vsv(4*2*gOutHeight*gOutWidth, vrgout_ptr_k0_s4) ;
	__vr vrgout_k2_s4 = _ve_vgtu_vvm(vrgout_ptr_k2_s4, vmall_s4) ;
	__vr vrgout_ptr_k3_s4 = _ve_vaddsl_vsv(4*3*gOutHeight*gOutWidth, vrgout_ptr_k0_s4) ;
	__vr vrgout_k3_s4 = _ve_vgtu_vvm(vrgout_ptr_k3_s4, vmall_s4) ;
	__vr vrgout_ptr_k4_s4 = _ve_vaddsl_vsv(4*4*gOutHeight*gOutWidth, vrgout_ptr_k0_s4) ;
	__vr vrgout_k4_s4 = _ve_vgtu_vvm(vrgout_ptr_k4_s4, vmall_s4) ;
	__vr vrgout_ptr_k5_s4 = _ve_vaddsl_vsv(4*5*gOutHeight*gOutWidth, vrgout_ptr_k0_s4) ;
	__vr vrgout_k5_s4 = _ve_vgtu_vvm(vrgout_ptr_k5_s4, vmall_s4) ;
	__vr vrgout_ptr_k6_s4 = _ve_vaddsl_vsv(4*6*gOutHeight*gOutWidth, vrgout_ptr_k0_s4) ;
	__vr vrgout_k6_s4 = _ve_vgtu_vvm(vrgout_ptr_k6_s4, vmall_s4) ;
	__vr vrgout_ptr_k7_s4 = _ve_vaddsl_vsv(4*7*gOutHeight*gOutWidth, vrgout_ptr_k0_s4) ;
	__vr vrgout_k7_s4 = _ve_vgtu_vvm(vrgout_ptr_k7_s4, vmall_s4) ;

	VFADD_C2(vrgout_k0_s4, vmall_s4, 0, r, 4)
	VFADD_C2(vrgout_k1_s4, vmall_s4, 1, r, 4)
	VFADD_C2(vrgout_k2_s4, vmall_s4, 2, r, 4)
	VFADD_C2(vrgout_k3_s4, vmall_s4, 3, r, 4)
	VFADD_C2(vrgout_k4_s4, vmall_s4, 4, r, 4)
	VFADD_C2(vrgout_k5_s4, vmall_s4, 5, r, 4)
	VFADD_C2(vrgout_k6_s4, vmall_s4, 6, r, 4)
	VFADD_C2(vrgout_k7_s4, vmall_s4, 7, r, 4)

#undef VFADD_C2
      } // gOutChannel
    } // kernHeight

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

    __vr vrj_s0 = _ve_vaddsl_vsv(padWidth-0*dilationWidth, vrw) ;
    __vr vrj_s1 = _ve_vaddsl_vsv(padWidth-1*dilationWidth, vrw) ;
    __vr vrj_s2 = _ve_vaddsl_vsv(padWidth-2*dilationWidth, vrw) ;
    __vr vrj_s3 = _ve_vaddsl_vsv(padWidth-3*dilationWidth, vrw) ;
    __vr vrj_s4 = _ve_vaddsl_vsv(padWidth-4*dilationWidth, vrw) ;

    __vr vrx_s0 = _ve_vdivsl_vvs(vrj_s0, strideWidth) ;
    __vr vrx_s1 = _ve_vdivsl_vvs(vrj_s1, strideWidth) ;
    __vr vrx_s2 = _ve_vdivsl_vvs(vrj_s2, strideWidth) ;
    __vr vrx_s3 = _ve_vdivsl_vvs(vrj_s3, strideWidth) ;
    __vr vrx_s4 = _ve_vdivsl_vvs(vrj_s4, strideWidth) ;

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

    __vm256 vmx0_s3 = _ve_vfmkl_mcv(VECC_IEQ, _ve_vcmpsl_vvv(vrj_s3, _ve_vmulsl_vsv(strideWidth, vrx_s3))) ;
    __vm256 vmx1_s3 = _ve_vfmkl_mcv(VECC_GE, vrx_s3) ;
    __vm256 vmx2_s3 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(gOutWidth,vrx_s3)) ;
    __vm256 vmx_s3 = _ve_andm_mmm(_ve_andm_mmm(vmx0_s3, vmx1_s3), vmx2_s3) ;

    __vm256 vmx0_s4 = _ve_vfmkl_mcv(VECC_IEQ, _ve_vcmpsl_vvv(vrj_s4, _ve_vmulsl_vsv(strideWidth, vrx_s4))) ;
    __vm256 vmx1_s4 = _ve_vfmkl_mcv(VECC_GE, vrx_s4) ;
    __vm256 vmx2_s4 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(gOutWidth,vrx_s4)) ;
    __vm256 vmx_s4 = _ve_andm_mmm(_ve_andm_mmm(vmx0_s4, vmx1_s4), vmx2_s4) ;

    for (int64_t r=0; r<kernHeight; r++) {
      __vr vri = _ve_vaddsl_vsv(padHeight-r*dilationHeight+h, vrh) ;
      __vr vry = _ve_vdivsl_vvs(vri, strideHeight) ;

      __vm256 vmy0 = _ve_vfmkl_mcv(VECC_IEQ, _ve_vcmpsl_vvv(vri, _ve_vmulsl_vsv(strideHeight, vry))) ;
      __vm256 vmy1 = _ve_vfmkl_mcv(VECC_GE, vry) ;
      __vm256 vmy2 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(gOutHeight,vry)) ;

      __vm256 vmy = _ve_andm_mmm(_ve_andm_mmm(vmy0, vmy1), vmy2) ;


      __vm256 vmall_s0 = _ve_andm_mmm(vmy,vmx_s0) ;
      __vm256 vmall_s1 = _ve_andm_mmm(vmy,vmx_s1) ;
      __vm256 vmall_s2 = _ve_andm_mmm(vmy,vmx_s2) ;
      __vm256 vmall_s3 = _ve_andm_mmm(vmy,vmx_s3) ;
      __vm256 vmall_s4 = _ve_andm_mmm(vmy,vmx_s4) ;

      int64_t k=0;
      if( (gOutChannelGroup & 0x01 ) == 1 ) {
	int64_t gOutIndex    = gOutGroupOffset + ((n * gOutChannel + k) * gOutHeight) * gOutWidth ;
	const float *pKerValue = pKernel + kernGroupOffset + (((k  ) * gInChannelGroup + c) * kernHeight) * kernWidth ;


#define VFADD_C4(VRGOUT, VM, K, R, S)  {												\
	  const uint64_t kerValue01 = _ve_pack_f32p(pKerValue + (((K)*gInChannelGroup + 0) * kernHeight +(R)) * kernWidth + (S),	\
						    pKerValue + (((K)*gInChannelGroup + 1) * kernHeight +(R)) * kernWidth + (S)) ;	\
	  const uint64_t kerValue23 = _ve_pack_f32p(pKerValue + (((K)*gInChannelGroup + 2) * kernHeight +(R)) * kernWidth + (S),	\
						    pKerValue + (((K)*gInChannelGroup + 3) * kernHeight +(R)) * kernWidth + (S)) ;	\
	  VRGOUT = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), VRGOUT, VM) ;				\
	  __vr vrgoutP = _ve_vshf_vvvs(VRGOUT, VRGOUT, VE_VSHUFFLE_YUZU) ;			\
	  vrsum01 = _ve_pvfmad_vvsv(vrsum01, kerValue01, vrgoutP) ;				\
	  vrsum23 = _ve_pvfmad_vvsv(vrsum23, kerValue23, vrgoutP) ;				\
	}

	__vr vrgout_ptr_k0_s0 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry), vrx_s0),
					   2,
					   (unsigned long)(pGOut+gOutIndex)) ;
	__vr vrgout_k0_s0 = _ve_vgtu_vvm(vrgout_ptr_k0_s0, vmall_s0) ;
	__vr vrgout_ptr_k1_s0 = _ve_vaddsl_vsv(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_s0) ;
	__vr vrgout_k1_s0 = _ve_vgtu_vvm(vrgout_ptr_k1_s0, vmall_s0) ;

	VFADD_C4(vrgout_k0_s0, vmall_s0, 0, r, 0)

	__vr vrgout_ptr_k0_s1 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry), vrx_s1),
					   2,
					   (unsigned long)(pGOut+gOutIndex)) ;
	__vr vrgout_k0_s1 = _ve_vgtu_vvm(vrgout_ptr_k0_s1, vmall_s1) ;

	VFADD_C4(vrgout_k0_s1, vmall_s1, 0, r, 1)


	__vr vrgout_ptr_k0_s2 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry), vrx_s2),
					   2,
					   (unsigned long)(pGOut+gOutIndex)) ;
	__vr vrgout_k0_s2 = _ve_vgtu_vvm(vrgout_ptr_k0_s2, vmall_s2) ;

	VFADD_C4(vrgout_k0_s2, vmall_s2, 0, r, 2)

	__vr vrgout_ptr_k0_s3 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry), vrx_s3),
					   2,
					   (unsigned long)(pGOut+gOutIndex)) ;
	__vr vrgout_k0_s3 = _ve_vgtu_vvm(vrgout_ptr_k0_s3, vmall_s3) ;
	__vr vrgout_ptr_k1_s3 = _ve_vaddsl_vsv(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_s3) ;

	VFADD_C4(vrgout_k0_s3, vmall_s3, 0, r, 3)

	__vr vrgout_ptr_k0_s4 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry), vrx_s4),
					   2,
					   (unsigned long)(pGOut+gOutIndex)) ;
	__vr vrgout_k0_s4 = _ve_vgtu_vvm(vrgout_ptr_k0_s4, vmall_s4) ;

	VFADD_C4(vrgout_k0_s4, vmall_s4, 0, r, 4)

	k+=1 ;
      }
      if( ((gOutChannelGroup >> 1) & 0x01 ) == 1 ) {
	int64_t gOutIndex    = gOutGroupOffset + ((n * gOutChannel + k) * gOutHeight) * gOutWidth ;
	const float *pKerValue = pKernel + kernGroupOffset + (((k  ) * gInChannelGroup + c) * kernHeight) * kernWidth ;

	__vr vrgout_ptr_k0_s0 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry), vrx_s0),
					   2,
					   (unsigned long)(pGOut+gOutIndex)) ;
	__vr vrgout_k0_s0 = _ve_vgtu_vvm(vrgout_ptr_k0_s0, vmall_s0) ;
	__vr vrgout_ptr_k1_s0 = _ve_vaddsl_vsv(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_s0) ;
	__vr vrgout_k1_s0 = _ve_vgtu_vvm(vrgout_ptr_k1_s0, vmall_s0) ;

	VFADD_C4(vrgout_k0_s0, vmall_s0, 0, r, 0)
	VFADD_C4(vrgout_k1_s0, vmall_s0, 1, r, 0)

	__vr vrgout_ptr_k0_s1 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry), vrx_s1),
					   2,
					   (unsigned long)(pGOut+gOutIndex)) ;
	__vr vrgout_k0_s1 = _ve_vgtu_vvm(vrgout_ptr_k0_s1, vmall_s1) ;
	__vr vrgout_ptr_k1_s1 = _ve_vaddsl_vsv(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_s1) ;
	__vr vrgout_k1_s1 = _ve_vgtu_vvm(vrgout_ptr_k1_s1, vmall_s1) ;

	VFADD_C4(vrgout_k0_s1, vmall_s1, 0, r, 1)
	VFADD_C4(vrgout_k1_s1, vmall_s1, 1, r, 1)


	__vr vrgout_ptr_k0_s2 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry), vrx_s2),
					   2,
					   (unsigned long)(pGOut+gOutIndex)) ;
	__vr vrgout_k0_s2 = _ve_vgtu_vvm(vrgout_ptr_k0_s2, vmall_s2) ;
	__vr vrgout_ptr_k1_s2 = _ve_vaddsl_vsv(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_s2) ;
	__vr vrgout_k1_s2 = _ve_vgtu_vvm(vrgout_ptr_k1_s2, vmall_s2) ;

	VFADD_C4(vrgout_k0_s2, vmall_s2, 0, r, 2)
	VFADD_C4(vrgout_k1_s2, vmall_s2, 1, r, 2)

	__vr vrgout_ptr_k0_s3 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry), vrx_s3),
					   2,
					   (unsigned long)(pGOut+gOutIndex)) ;
	__vr vrgout_k0_s3 = _ve_vgtu_vvm(vrgout_ptr_k0_s3, vmall_s3) ;
	__vr vrgout_ptr_k1_s3 = _ve_vaddsl_vsv(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_s3) ;
	__vr vrgout_k1_s3 = _ve_vgtu_vvm(vrgout_ptr_k1_s3, vmall_s3) ;
	__vr vrgout_ptr_k2_s3 = _ve_vaddsl_vsv(4*2*gOutHeight*gOutWidth, vrgout_ptr_k0_s3) ;

	VFADD_C4(vrgout_k0_s3, vmall_s3, 0, r, 3)
	VFADD_C4(vrgout_k1_s3, vmall_s3, 1, r, 3)

	__vr vrgout_ptr_k0_s4 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry), vrx_s4),
					   2,
					   (unsigned long)(pGOut+gOutIndex)) ;
	__vr vrgout_k0_s4 = _ve_vgtu_vvm(vrgout_ptr_k0_s4, vmall_s4) ;
	__vr vrgout_ptr_k1_s4 = _ve_vaddsl_vsv(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_s4) ;
	__vr vrgout_k1_s4 = _ve_vgtu_vvm(vrgout_ptr_k1_s4, vmall_s4) ;

	VFADD_C4(vrgout_k0_s4, vmall_s4, 0, r, 4)
	VFADD_C4(vrgout_k1_s4, vmall_s4, 1, r, 4)

	k+=2 ;
      }
      if( ((gOutChannelGroup >> 2) & 0x01 ) == 1 ) {
	int64_t gOutIndex    = gOutGroupOffset + ((n * gOutChannel + k) * gOutHeight) * gOutWidth ;
	const float *pKerValue = pKernel + kernGroupOffset + (((k  ) * gInChannelGroup + c) * kernHeight) * kernWidth ;

	__vr vrgout_ptr_k0_s0 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry), vrx_s0),
					   2,
					   (unsigned long)(pGOut+gOutIndex)) ;
	__vr vrgout_k0_s0 = _ve_vgtu_vvm(vrgout_ptr_k0_s0, vmall_s0) ;
	__vr vrgout_ptr_k1_s0 = _ve_vaddsl_vsv(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_s0) ;
	__vr vrgout_k1_s0 = _ve_vgtu_vvm(vrgout_ptr_k1_s0, vmall_s0) ;
	__vr vrgout_ptr_k2_s0 = _ve_vaddsl_vsv(4*2*gOutHeight*gOutWidth, vrgout_ptr_k0_s0) ;
	__vr vrgout_k2_s0 = _ve_vgtu_vvm(vrgout_ptr_k2_s0, vmall_s0) ;
	__vr vrgout_ptr_k3_s0 = _ve_vaddsl_vsv(4*3*gOutHeight*gOutWidth, vrgout_ptr_k0_s0) ;
	__vr vrgout_k3_s0 = _ve_vgtu_vvm(vrgout_ptr_k3_s0, vmall_s0) ;

	VFADD_C4(vrgout_k0_s0, vmall_s0, 0, r, 0)
	VFADD_C4(vrgout_k1_s0, vmall_s0, 1, r, 0)
	VFADD_C4(vrgout_k2_s0, vmall_s0, 2, r, 0)
	VFADD_C4(vrgout_k3_s0, vmall_s0, 3, r, 0)

	__vr vrgout_ptr_k0_s1 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry), vrx_s1),
					   2,
					   (unsigned long)(pGOut+gOutIndex)) ;
	__vr vrgout_k0_s1 = _ve_vgtu_vvm(vrgout_ptr_k0_s1, vmall_s1) ;
	__vr vrgout_ptr_k1_s1 = _ve_vaddsl_vsv(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_s1) ;
	__vr vrgout_k1_s1 = _ve_vgtu_vvm(vrgout_ptr_k1_s1, vmall_s1) ;
	__vr vrgout_ptr_k2_s1 = _ve_vaddsl_vsv(4*2*gOutHeight*gOutWidth, vrgout_ptr_k0_s1) ;
	__vr vrgout_k2_s1 = _ve_vgtu_vvm(vrgout_ptr_k2_s1, vmall_s1) ;
	__vr vrgout_ptr_k3_s1 = _ve_vaddsl_vsv(4*3*gOutHeight*gOutWidth, vrgout_ptr_k0_s1) ;
	__vr vrgout_k3_s1 = _ve_vgtu_vvm(vrgout_ptr_k3_s1, vmall_s1) ;

	VFADD_C4(vrgout_k0_s1, vmall_s1, 0, r, 1)
	VFADD_C4(vrgout_k1_s1, vmall_s1, 1, r, 1)
	VFADD_C4(vrgout_k2_s1, vmall_s1, 2, r, 1)
	VFADD_C4(vrgout_k3_s1, vmall_s1, 3, r, 1)


	__vr vrgout_ptr_k0_s2 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry), vrx_s2),
					   2,
					   (unsigned long)(pGOut+gOutIndex)) ;
	__vr vrgout_k0_s2 = _ve_vgtu_vvm(vrgout_ptr_k0_s2, vmall_s2) ;
	__vr vrgout_ptr_k1_s2 = _ve_vaddsl_vsv(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_s2) ;
	__vr vrgout_k1_s2 = _ve_vgtu_vvm(vrgout_ptr_k1_s2, vmall_s2) ;
	__vr vrgout_ptr_k2_s2 = _ve_vaddsl_vsv(4*2*gOutHeight*gOutWidth, vrgout_ptr_k0_s2) ;
	__vr vrgout_k2_s2 = _ve_vgtu_vvm(vrgout_ptr_k2_s2, vmall_s2) ;
	__vr vrgout_ptr_k3_s2 = _ve_vaddsl_vsv(4*3*gOutHeight*gOutWidth, vrgout_ptr_k0_s2) ;
	__vr vrgout_k3_s2 = _ve_vgtu_vvm(vrgout_ptr_k3_s2, vmall_s2) ;

	VFADD_C4(vrgout_k0_s2, vmall_s2, 0, r, 2)
	VFADD_C4(vrgout_k1_s2, vmall_s2, 1, r, 2)
	VFADD_C4(vrgout_k2_s2, vmall_s2, 2, r, 2)
	VFADD_C4(vrgout_k3_s2, vmall_s2, 3, r, 2)

	__vr vrgout_ptr_k0_s3 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry), vrx_s3),
					   2,
					   (unsigned long)(pGOut+gOutIndex)) ;
	__vr vrgout_k0_s3 = _ve_vgtu_vvm(vrgout_ptr_k0_s3, vmall_s3) ;
	__vr vrgout_ptr_k1_s3 = _ve_vaddsl_vsv(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_s3) ;
	__vr vrgout_k1_s3 = _ve_vgtu_vvm(vrgout_ptr_k1_s3, vmall_s3) ;
	__vr vrgout_ptr_k2_s3 = _ve_vaddsl_vsv(4*2*gOutHeight*gOutWidth, vrgout_ptr_k0_s3) ;
	__vr vrgout_k2_s3 = _ve_vgtu_vvm(vrgout_ptr_k2_s3, vmall_s3) ;
	__vr vrgout_ptr_k3_s3 = _ve_vaddsl_vsv(4*3*gOutHeight*gOutWidth, vrgout_ptr_k0_s3) ;
	__vr vrgout_k3_s3 = _ve_vgtu_vvm(vrgout_ptr_k3_s3, vmall_s3) ;
	__vr vrgout_ptr_k4_s3 = _ve_vaddsl_vsv(4*4*gOutHeight*gOutWidth, vrgout_ptr_k0_s3) ;

	VFADD_C4(vrgout_k0_s3, vmall_s3, 0, r, 3)
	VFADD_C4(vrgout_k1_s3, vmall_s3, 1, r, 3)
	VFADD_C4(vrgout_k2_s3, vmall_s3, 2, r, 3)
	VFADD_C4(vrgout_k3_s3, vmall_s3, 3, r, 3)

	__vr vrgout_ptr_k0_s4 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry), vrx_s4),
					   2,
					   (unsigned long)(pGOut+gOutIndex)) ;
	__vr vrgout_k0_s4 = _ve_vgtu_vvm(vrgout_ptr_k0_s4, vmall_s4) ;
	__vr vrgout_ptr_k1_s4 = _ve_vaddsl_vsv(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_s4) ;
	__vr vrgout_k1_s4 = _ve_vgtu_vvm(vrgout_ptr_k1_s4, vmall_s4) ;
	__vr vrgout_ptr_k2_s4 = _ve_vaddsl_vsv(4*2*gOutHeight*gOutWidth, vrgout_ptr_k0_s4) ;
	__vr vrgout_k2_s4 = _ve_vgtu_vvm(vrgout_ptr_k2_s4, vmall_s4) ;
	__vr vrgout_ptr_k3_s4 = _ve_vaddsl_vsv(4*3*gOutHeight*gOutWidth, vrgout_ptr_k0_s4) ;
	__vr vrgout_k3_s4 = _ve_vgtu_vvm(vrgout_ptr_k3_s4, vmall_s4) ;

	VFADD_C4(vrgout_k0_s4, vmall_s4, 0, r, 4)
	VFADD_C4(vrgout_k1_s4, vmall_s4, 1, r, 4)
	VFADD_C4(vrgout_k2_s4, vmall_s4, 2, r, 4)
	VFADD_C4(vrgout_k3_s4, vmall_s4, 3, r, 4)

	k+=4 ;
      }
      for (; k<gOutChannelGroup; k+=8) {
	int64_t gOutIndex    = gOutGroupOffset + ((n * gOutChannel + k) * gOutHeight) * gOutWidth ;
	const float *pKerValue = pKernel + kernGroupOffset + (((k  ) * gInChannelGroup + c) * kernHeight) * kernWidth ;

	__vr vrgout_ptr_k0_s0 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry), vrx_s0),
					   2,
					   (unsigned long)(pGOut+gOutIndex)) ;
	__vr vrgout_k0_s0 = _ve_vgtu_vvm(vrgout_ptr_k0_s0, vmall_s0) ;
	__vr vrgout_ptr_k1_s0 = _ve_vaddsl_vsv(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_s0) ;
	__vr vrgout_k1_s0 = _ve_vgtu_vvm(vrgout_ptr_k1_s0, vmall_s0) ;
	__vr vrgout_ptr_k2_s0 = _ve_vaddsl_vsv(4*2*gOutHeight*gOutWidth, vrgout_ptr_k0_s0) ;
	__vr vrgout_k2_s0 = _ve_vgtu_vvm(vrgout_ptr_k2_s0, vmall_s0) ;
	__vr vrgout_ptr_k3_s0 = _ve_vaddsl_vsv(4*3*gOutHeight*gOutWidth, vrgout_ptr_k0_s0) ;
	__vr vrgout_k3_s0 = _ve_vgtu_vvm(vrgout_ptr_k3_s0, vmall_s0) ;
	__vr vrgout_ptr_k4_s0 = _ve_vaddsl_vsv(4*4*gOutHeight*gOutWidth, vrgout_ptr_k0_s0) ;
	__vr vrgout_k4_s0 = _ve_vgtu_vvm(vrgout_ptr_k4_s0, vmall_s0) ;
	__vr vrgout_ptr_k5_s0 = _ve_vaddsl_vsv(4*5*gOutHeight*gOutWidth, vrgout_ptr_k0_s0) ;
	__vr vrgout_k5_s0 = _ve_vgtu_vvm(vrgout_ptr_k5_s0, vmall_s0) ;
	__vr vrgout_ptr_k6_s0 = _ve_vaddsl_vsv(4*6*gOutHeight*gOutWidth, vrgout_ptr_k0_s0) ;
	__vr vrgout_k6_s0 = _ve_vgtu_vvm(vrgout_ptr_k6_s0, vmall_s0) ;
	__vr vrgout_ptr_k7_s0 = _ve_vaddsl_vsv(4*7*gOutHeight*gOutWidth, vrgout_ptr_k0_s0) ;
	__vr vrgout_k7_s0 = _ve_vgtu_vvm(vrgout_ptr_k7_s0, vmall_s0) ;

	VFADD_C4(vrgout_k0_s0, vmall_s0, 0, r, 0)
	VFADD_C4(vrgout_k1_s0, vmall_s0, 1, r, 0)
	VFADD_C4(vrgout_k2_s0, vmall_s0, 2, r, 0)
	VFADD_C4(vrgout_k3_s0, vmall_s0, 3, r, 0)
	VFADD_C4(vrgout_k4_s0, vmall_s0, 4, r, 0)
	VFADD_C4(vrgout_k5_s0, vmall_s0, 5, r, 0)
	VFADD_C4(vrgout_k6_s0, vmall_s0, 6, r, 0)
	VFADD_C4(vrgout_k7_s0, vmall_s0, 7, r, 0)

	__vr vrgout_ptr_k0_s1 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry), vrx_s1),
					   2,
					   (unsigned long)(pGOut+gOutIndex)) ;
	__vr vrgout_k0_s1 = _ve_vgtu_vvm(vrgout_ptr_k0_s1, vmall_s1) ;
	__vr vrgout_ptr_k1_s1 = _ve_vaddsl_vsv(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_s1) ;
	__vr vrgout_k1_s1 = _ve_vgtu_vvm(vrgout_ptr_k1_s1, vmall_s1) ;
	__vr vrgout_ptr_k2_s1 = _ve_vaddsl_vsv(4*2*gOutHeight*gOutWidth, vrgout_ptr_k0_s1) ;
	__vr vrgout_k2_s1 = _ve_vgtu_vvm(vrgout_ptr_k2_s1, vmall_s1) ;
	__vr vrgout_ptr_k3_s1 = _ve_vaddsl_vsv(4*3*gOutHeight*gOutWidth, vrgout_ptr_k0_s1) ;
	__vr vrgout_k3_s1 = _ve_vgtu_vvm(vrgout_ptr_k3_s1, vmall_s1) ;
	__vr vrgout_ptr_k4_s1 = _ve_vaddsl_vsv(4*4*gOutHeight*gOutWidth, vrgout_ptr_k0_s1) ;
	__vr vrgout_k4_s1 = _ve_vgtu_vvm(vrgout_ptr_k4_s1, vmall_s1) ;
	__vr vrgout_ptr_k5_s1 = _ve_vaddsl_vsv(4*5*gOutHeight*gOutWidth, vrgout_ptr_k0_s1) ;
	__vr vrgout_k5_s1 = _ve_vgtu_vvm(vrgout_ptr_k5_s1, vmall_s1) ;
	__vr vrgout_ptr_k6_s1 = _ve_vaddsl_vsv(4*6*gOutHeight*gOutWidth, vrgout_ptr_k0_s1) ;
	__vr vrgout_k6_s1 = _ve_vgtu_vvm(vrgout_ptr_k6_s1, vmall_s1) ;
	__vr vrgout_ptr_k7_s1 = _ve_vaddsl_vsv(4*7*gOutHeight*gOutWidth, vrgout_ptr_k0_s1) ;
	__vr vrgout_k7_s1 = _ve_vgtu_vvm(vrgout_ptr_k7_s1, vmall_s1) ;

	VFADD_C4(vrgout_k0_s1, vmall_s1, 0, r, 1)
	VFADD_C4(vrgout_k1_s1, vmall_s1, 1, r, 1)
	VFADD_C4(vrgout_k2_s1, vmall_s1, 2, r, 1)
	VFADD_C4(vrgout_k3_s1, vmall_s1, 3, r, 1)
	VFADD_C4(vrgout_k4_s1, vmall_s1, 4, r, 1)
	VFADD_C4(vrgout_k5_s1, vmall_s1, 5, r, 1)
	VFADD_C4(vrgout_k6_s1, vmall_s1, 6, r, 1)
	VFADD_C4(vrgout_k7_s1, vmall_s1, 7, r, 1)


	__vr vrgout_ptr_k0_s2 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry), vrx_s2),
					   2,
					   (unsigned long)(pGOut+gOutIndex)) ;
	__vr vrgout_k0_s2 = _ve_vgtu_vvm(vrgout_ptr_k0_s2, vmall_s2) ;
	__vr vrgout_ptr_k1_s2 = _ve_vaddsl_vsv(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_s2) ;
	__vr vrgout_k1_s2 = _ve_vgtu_vvm(vrgout_ptr_k1_s2, vmall_s2) ;
	__vr vrgout_ptr_k2_s2 = _ve_vaddsl_vsv(4*2*gOutHeight*gOutWidth, vrgout_ptr_k0_s2) ;
	__vr vrgout_k2_s2 = _ve_vgtu_vvm(vrgout_ptr_k2_s2, vmall_s2) ;
	__vr vrgout_ptr_k3_s2 = _ve_vaddsl_vsv(4*3*gOutHeight*gOutWidth, vrgout_ptr_k0_s2) ;
	__vr vrgout_k3_s2 = _ve_vgtu_vvm(vrgout_ptr_k3_s2, vmall_s2) ;
	__vr vrgout_ptr_k4_s2 = _ve_vaddsl_vsv(4*4*gOutHeight*gOutWidth, vrgout_ptr_k0_s2) ;
	__vr vrgout_k4_s2 = _ve_vgtu_vvm(vrgout_ptr_k4_s2, vmall_s2) ;
	__vr vrgout_ptr_k5_s2 = _ve_vaddsl_vsv(4*5*gOutHeight*gOutWidth, vrgout_ptr_k0_s2) ;
	__vr vrgout_k5_s2 = _ve_vgtu_vvm(vrgout_ptr_k5_s2, vmall_s2) ;
	__vr vrgout_ptr_k6_s2 = _ve_vaddsl_vsv(4*6*gOutHeight*gOutWidth, vrgout_ptr_k0_s2) ;
	__vr vrgout_k6_s2 = _ve_vgtu_vvm(vrgout_ptr_k6_s2, vmall_s2) ;
	__vr vrgout_ptr_k7_s2 = _ve_vaddsl_vsv(4*7*gOutHeight*gOutWidth, vrgout_ptr_k0_s2) ;
	__vr vrgout_k7_s2 = _ve_vgtu_vvm(vrgout_ptr_k7_s2, vmall_s2) ;

	VFADD_C4(vrgout_k0_s2, vmall_s2, 0, r, 2)
	VFADD_C4(vrgout_k1_s2, vmall_s2, 1, r, 2)
	VFADD_C4(vrgout_k2_s2, vmall_s2, 2, r, 2)
	VFADD_C4(vrgout_k3_s2, vmall_s2, 3, r, 2)
	VFADD_C4(vrgout_k4_s2, vmall_s2, 4, r, 2)
	VFADD_C4(vrgout_k5_s2, vmall_s2, 5, r, 2)
	VFADD_C4(vrgout_k6_s2, vmall_s2, 6, r, 2)
	VFADD_C4(vrgout_k7_s2, vmall_s2, 7, r, 2)


	__vr vrgout_ptr_k0_s3 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry), vrx_s3),
					   2,
					   (unsigned long)(pGOut+gOutIndex)) ;
	__vr vrgout_k0_s3 = _ve_vgtu_vvm(vrgout_ptr_k0_s3, vmall_s3) ;
	__vr vrgout_ptr_k1_s3 = _ve_vaddsl_vsv(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_s3) ;
	__vr vrgout_k1_s3 = _ve_vgtu_vvm(vrgout_ptr_k1_s3, vmall_s3) ;
	__vr vrgout_ptr_k2_s3 = _ve_vaddsl_vsv(4*2*gOutHeight*gOutWidth, vrgout_ptr_k0_s3) ;
	__vr vrgout_k2_s3 = _ve_vgtu_vvm(vrgout_ptr_k2_s3, vmall_s3) ;
	__vr vrgout_ptr_k3_s3 = _ve_vaddsl_vsv(4*3*gOutHeight*gOutWidth, vrgout_ptr_k0_s3) ;
	__vr vrgout_k3_s3 = _ve_vgtu_vvm(vrgout_ptr_k3_s3, vmall_s3) ;
	__vr vrgout_ptr_k4_s3 = _ve_vaddsl_vsv(4*4*gOutHeight*gOutWidth, vrgout_ptr_k0_s3) ;
	__vr vrgout_k4_s3 = _ve_vgtu_vvm(vrgout_ptr_k4_s3, vmall_s3) ;
	__vr vrgout_ptr_k5_s3 = _ve_vaddsl_vsv(4*5*gOutHeight*gOutWidth, vrgout_ptr_k0_s3) ;
	__vr vrgout_k5_s3 = _ve_vgtu_vvm(vrgout_ptr_k5_s3, vmall_s3) ;
	__vr vrgout_ptr_k6_s3 = _ve_vaddsl_vsv(4*6*gOutHeight*gOutWidth, vrgout_ptr_k0_s3) ;
	__vr vrgout_k6_s3 = _ve_vgtu_vvm(vrgout_ptr_k6_s3, vmall_s3) ;
	__vr vrgout_ptr_k7_s3 = _ve_vaddsl_vsv(4*7*gOutHeight*gOutWidth, vrgout_ptr_k0_s3) ;
	__vr vrgout_k7_s3 = _ve_vgtu_vvm(vrgout_ptr_k7_s3, vmall_s3) ;

	VFADD_C4(vrgout_k0_s3, vmall_s3, 0, r, 3)
	VFADD_C4(vrgout_k1_s3, vmall_s3, 1, r, 3)
	VFADD_C4(vrgout_k2_s3, vmall_s3, 2, r, 3)
	VFADD_C4(vrgout_k3_s3, vmall_s3, 3, r, 3)
	VFADD_C4(vrgout_k4_s3, vmall_s3, 4, r, 3)
	VFADD_C4(vrgout_k5_s3, vmall_s3, 5, r, 3)
	VFADD_C4(vrgout_k6_s3, vmall_s3, 6, r, 3)
	VFADD_C4(vrgout_k7_s3, vmall_s3, 7, r, 3)


	__vr vrgout_ptr_k0_s4 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry), vrx_s4),
					   2,
					   (unsigned long)(pGOut+gOutIndex)) ;
	__vr vrgout_k0_s4 = _ve_vgtu_vvm(vrgout_ptr_k0_s4, vmall_s4) ;
	__vr vrgout_ptr_k1_s4 = _ve_vaddsl_vsv(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_s4) ;
	__vr vrgout_k1_s4 = _ve_vgtu_vvm(vrgout_ptr_k1_s4, vmall_s4) ;
	__vr vrgout_ptr_k2_s4 = _ve_vaddsl_vsv(4*2*gOutHeight*gOutWidth, vrgout_ptr_k0_s4) ;
	__vr vrgout_k2_s4 = _ve_vgtu_vvm(vrgout_ptr_k2_s4, vmall_s4) ;
	__vr vrgout_ptr_k3_s4 = _ve_vaddsl_vsv(4*3*gOutHeight*gOutWidth, vrgout_ptr_k0_s4) ;
	__vr vrgout_k3_s4 = _ve_vgtu_vvm(vrgout_ptr_k3_s4, vmall_s4) ;
	__vr vrgout_ptr_k4_s4 = _ve_vaddsl_vsv(4*4*gOutHeight*gOutWidth, vrgout_ptr_k0_s4) ;
	__vr vrgout_k4_s4 = _ve_vgtu_vvm(vrgout_ptr_k4_s4, vmall_s4) ;
	__vr vrgout_ptr_k5_s4 = _ve_vaddsl_vsv(4*5*gOutHeight*gOutWidth, vrgout_ptr_k0_s4) ;
	__vr vrgout_k5_s4 = _ve_vgtu_vvm(vrgout_ptr_k5_s4, vmall_s4) ;
	__vr vrgout_ptr_k6_s4 = _ve_vaddsl_vsv(4*6*gOutHeight*gOutWidth, vrgout_ptr_k0_s4) ;
	__vr vrgout_k6_s4 = _ve_vgtu_vvm(vrgout_ptr_k6_s4, vmall_s4) ;
	__vr vrgout_ptr_k7_s4 = _ve_vaddsl_vsv(4*7*gOutHeight*gOutWidth, vrgout_ptr_k0_s4) ;
	__vr vrgout_k7_s4 = _ve_vgtu_vvm(vrgout_ptr_k7_s4, vmall_s4) ;

	VFADD_C4(vrgout_k0_s4, vmall_s4, 0, r, 4)
	VFADD_C4(vrgout_k1_s4, vmall_s4, 1, r, 4)
	VFADD_C4(vrgout_k2_s4, vmall_s4, 2, r, 4)
	VFADD_C4(vrgout_k3_s4, vmall_s4, 3, r, 4)
	VFADD_C4(vrgout_k4_s4, vmall_s4, 4, r, 4)
	VFADD_C4(vrgout_k5_s4, vmall_s4, 5, r, 4)
	VFADD_C4(vrgout_k6_s4, vmall_s4, 6, r, 4)
	VFADD_C4(vrgout_k7_s4, vmall_s4, 7, r, 4)

#undef VFADD_C4
      } // gOutChannel
    } // kernHeight

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

    __vr vrj_s0 = _ve_vaddsl_vsv(padWidth-0*dilationWidth, vrw) ;
    __vr vrj_s1 = _ve_vaddsl_vsv(padWidth-1*dilationWidth, vrw) ;
    __vr vrj_s2 = _ve_vaddsl_vsv(padWidth-2*dilationWidth, vrw) ;
    __vr vrj_s3 = _ve_vaddsl_vsv(padWidth-3*dilationWidth, vrw) ;
    __vr vrj_s4 = _ve_vaddsl_vsv(padWidth-4*dilationWidth, vrw) ;

    __vr vrx_s0 = _ve_vdivsl_vvs(vrj_s0, strideWidth) ;
    __vr vrx_s1 = _ve_vdivsl_vvs(vrj_s1, strideWidth) ;
    __vr vrx_s2 = _ve_vdivsl_vvs(vrj_s2, strideWidth) ;
    __vr vrx_s3 = _ve_vdivsl_vvs(vrj_s3, strideWidth) ;
    __vr vrx_s4 = _ve_vdivsl_vvs(vrj_s4, strideWidth) ;

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

    __vm256 vmx0_s3 = _ve_vfmkl_mcv(VECC_IEQ, _ve_vcmpsl_vvv(vrj_s3, _ve_vmulsl_vsv(strideWidth, vrx_s3))) ;
    __vm256 vmx1_s3 = _ve_vfmkl_mcv(VECC_GE, vrx_s3) ;
    __vm256 vmx2_s3 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(gOutWidth,vrx_s3)) ;
    __vm256 vmx_s3 = _ve_andm_mmm(_ve_andm_mmm(vmx0_s3, vmx1_s3), vmx2_s3) ;

    __vm256 vmx0_s4 = _ve_vfmkl_mcv(VECC_IEQ, _ve_vcmpsl_vvv(vrj_s4, _ve_vmulsl_vsv(strideWidth, vrx_s4))) ;
    __vm256 vmx1_s4 = _ve_vfmkl_mcv(VECC_GE, vrx_s4) ;
    __vm256 vmx2_s4 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(gOutWidth,vrx_s4)) ;
    __vm256 vmx_s4 = _ve_andm_mmm(_ve_andm_mmm(vmx0_s4, vmx1_s4), vmx2_s4) ;

    for (int64_t r=0; r<kernHeight; r++) {
      __vr vri = _ve_vaddsl_vsv(padHeight-r*dilationHeight+h, vrh) ;
      __vr vry = _ve_vdivsl_vvs(vri, strideHeight) ;

      __vm256 vmy0 = _ve_vfmkl_mcv(VECC_IEQ, _ve_vcmpsl_vvv(vri, _ve_vmulsl_vsv(strideHeight, vry))) ;
      __vm256 vmy1 = _ve_vfmkl_mcv(VECC_GE, vry) ;
      __vm256 vmy2 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(gOutHeight,vry)) ;

      __vm256 vmy = _ve_andm_mmm(_ve_andm_mmm(vmy0, vmy1), vmy2) ;


      __vm256 vmall_s0 = _ve_andm_mmm(vmy,vmx_s0) ;
      __vm256 vmall_s1 = _ve_andm_mmm(vmy,vmx_s1) ;
      __vm256 vmall_s2 = _ve_andm_mmm(vmy,vmx_s2) ;
      __vm256 vmall_s3 = _ve_andm_mmm(vmy,vmx_s3) ;
      __vm256 vmall_s4 = _ve_andm_mmm(vmy,vmx_s4) ;

      int64_t k=0;
      if( (gOutChannelGroup & 0x01 ) == 1 ) {
	int64_t gOutIndex    = gOutGroupOffset + ((n * gOutChannel + k) * gOutHeight) * gOutWidth ;
	const float *pKerValue = pKernel + kernGroupOffset + (((k  ) * gInChannelGroup + c) * kernHeight) * kernWidth ;


#define VFADD_C8(VRGOUT, VM, K, R, S)  {												\
	  const uint64_t kerValue01 = _ve_pack_f32p(pKerValue + (((K)*gInChannelGroup + 0) * kernHeight +(R)) * kernWidth + (S),	\
						    pKerValue + (((K)*gInChannelGroup + 1) * kernHeight +(R)) * kernWidth + (S)) ;	\
	  const uint64_t kerValue23 = _ve_pack_f32p(pKerValue + (((K)*gInChannelGroup + 2) * kernHeight +(R)) * kernWidth + (S),	\
						    pKerValue + (((K)*gInChannelGroup + 3) * kernHeight +(R)) * kernWidth + (S)) ;	\
	  const uint64_t kerValue45 = _ve_pack_f32p(pKerValue + (((K)*gInChannelGroup + 4) * kernHeight +(R)) * kernWidth + (S),	\
						    pKerValue + (((K)*gInChannelGroup + 5) * kernHeight +(R)) * kernWidth + (S)) ;	\
	  const uint64_t kerValue67 = _ve_pack_f32p(pKerValue + (((K)*gInChannelGroup + 6) * kernHeight +(R)) * kernWidth + (S),	\
						    pKerValue + (((K)*gInChannelGroup + 7) * kernHeight +(R)) * kernWidth + (S)) ;	\
	  VRGOUT = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), VRGOUT, VM) ;				\
	  __vr vrgoutP = _ve_vshf_vvvs(VRGOUT, VRGOUT, VE_VSHUFFLE_YUZU) ;			\
	  vrsum01 = _ve_pvfmad_vvsv(vrsum01, kerValue01, vrgoutP) ;				\
	  vrsum23 = _ve_pvfmad_vvsv(vrsum23, kerValue23, vrgoutP) ;				\
	  vrsum45 = _ve_pvfmad_vvsv(vrsum45, kerValue45, vrgoutP) ;				\
	  vrsum67 = _ve_pvfmad_vvsv(vrsum67, kerValue67, vrgoutP) ;				\
	}

	__vr vrgout_ptr_k0_s0 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry), vrx_s0),
					   2,
					   (unsigned long)(pGOut+gOutIndex)) ;
	__vr vrgout_k0_s0 = _ve_vgtu_vvm(vrgout_ptr_k0_s0, vmall_s0) ;
	__vr vrgout_ptr_k1_s0 = _ve_vaddsl_vsv(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_s0) ;
	__vr vrgout_k1_s0 = _ve_vgtu_vvm(vrgout_ptr_k1_s0, vmall_s0) ;

	VFADD_C8(vrgout_k0_s0, vmall_s0, 0, r, 0)

	__vr vrgout_ptr_k0_s1 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry), vrx_s1),
					   2,
					   (unsigned long)(pGOut+gOutIndex)) ;
	__vr vrgout_k0_s1 = _ve_vgtu_vvm(vrgout_ptr_k0_s1, vmall_s1) ;

	VFADD_C8(vrgout_k0_s1, vmall_s1, 0, r, 1)


	__vr vrgout_ptr_k0_s2 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry), vrx_s2),
					   2,
					   (unsigned long)(pGOut+gOutIndex)) ;
	__vr vrgout_k0_s2 = _ve_vgtu_vvm(vrgout_ptr_k0_s2, vmall_s2) ;

	VFADD_C8(vrgout_k0_s2, vmall_s2, 0, r, 2)

	__vr vrgout_ptr_k0_s3 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry), vrx_s3),
					   2,
					   (unsigned long)(pGOut+gOutIndex)) ;
	__vr vrgout_k0_s3 = _ve_vgtu_vvm(vrgout_ptr_k0_s3, vmall_s3) ;
	__vr vrgout_ptr_k1_s3 = _ve_vaddsl_vsv(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_s3) ;

	VFADD_C8(vrgout_k0_s3, vmall_s3, 0, r, 3)

	__vr vrgout_ptr_k0_s4 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry), vrx_s4),
					   2,
					   (unsigned long)(pGOut+gOutIndex)) ;
	__vr vrgout_k0_s4 = _ve_vgtu_vvm(vrgout_ptr_k0_s4, vmall_s4) ;

	VFADD_C8(vrgout_k0_s4, vmall_s4, 0, r, 4)

	k+=1 ;
      }
      if( ((gOutChannelGroup >> 1) & 0x01 ) == 1 ) {
	int64_t gOutIndex    = gOutGroupOffset + ((n * gOutChannel + k) * gOutHeight) * gOutWidth ;
	const float *pKerValue = pKernel + kernGroupOffset + (((k  ) * gInChannelGroup + c) * kernHeight) * kernWidth ;

	__vr vrgout_ptr_k0_s0 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry), vrx_s0),
					   2,
					   (unsigned long)(pGOut+gOutIndex)) ;
	__vr vrgout_k0_s0 = _ve_vgtu_vvm(vrgout_ptr_k0_s0, vmall_s0) ;
	__vr vrgout_ptr_k1_s0 = _ve_vaddsl_vsv(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_s0) ;
	__vr vrgout_k1_s0 = _ve_vgtu_vvm(vrgout_ptr_k1_s0, vmall_s0) ;

	VFADD_C8(vrgout_k0_s0, vmall_s0, 0, r, 0)
	VFADD_C8(vrgout_k1_s0, vmall_s0, 1, r, 0)

	__vr vrgout_ptr_k0_s1 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry), vrx_s1),
					   2,
					   (unsigned long)(pGOut+gOutIndex)) ;
	__vr vrgout_k0_s1 = _ve_vgtu_vvm(vrgout_ptr_k0_s1, vmall_s1) ;
	__vr vrgout_ptr_k1_s1 = _ve_vaddsl_vsv(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_s1) ;
	__vr vrgout_k1_s1 = _ve_vgtu_vvm(vrgout_ptr_k1_s1, vmall_s1) ;

	VFADD_C8(vrgout_k0_s1, vmall_s1, 0, r, 1)
	VFADD_C8(vrgout_k1_s1, vmall_s1, 1, r, 1)


	__vr vrgout_ptr_k0_s2 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry), vrx_s2),
					   2,
					   (unsigned long)(pGOut+gOutIndex)) ;
	__vr vrgout_k0_s2 = _ve_vgtu_vvm(vrgout_ptr_k0_s2, vmall_s2) ;
	__vr vrgout_ptr_k1_s2 = _ve_vaddsl_vsv(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_s2) ;
	__vr vrgout_k1_s2 = _ve_vgtu_vvm(vrgout_ptr_k1_s2, vmall_s2) ;

	VFADD_C8(vrgout_k0_s2, vmall_s2, 0, r, 2)
	VFADD_C8(vrgout_k1_s2, vmall_s2, 1, r, 2)

	__vr vrgout_ptr_k0_s3 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry), vrx_s3),
					   2,
					   (unsigned long)(pGOut+gOutIndex)) ;
	__vr vrgout_k0_s3 = _ve_vgtu_vvm(vrgout_ptr_k0_s3, vmall_s3) ;
	__vr vrgout_ptr_k1_s3 = _ve_vaddsl_vsv(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_s3) ;
	__vr vrgout_k1_s3 = _ve_vgtu_vvm(vrgout_ptr_k1_s3, vmall_s3) ;
	__vr vrgout_ptr_k2_s3 = _ve_vaddsl_vsv(4*2*gOutHeight*gOutWidth, vrgout_ptr_k0_s3) ;

	VFADD_C8(vrgout_k0_s3, vmall_s3, 0, r, 3)
	VFADD_C8(vrgout_k1_s3, vmall_s3, 1, r, 3)

	__vr vrgout_ptr_k0_s4 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry), vrx_s4),
					   2,
					   (unsigned long)(pGOut+gOutIndex)) ;
	__vr vrgout_k0_s4 = _ve_vgtu_vvm(vrgout_ptr_k0_s4, vmall_s4) ;
	__vr vrgout_ptr_k1_s4 = _ve_vaddsl_vsv(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_s4) ;
	__vr vrgout_k1_s4 = _ve_vgtu_vvm(vrgout_ptr_k1_s4, vmall_s4) ;

	VFADD_C8(vrgout_k0_s4, vmall_s4, 0, r, 4)
	VFADD_C8(vrgout_k1_s4, vmall_s4, 1, r, 4)

	k+=2 ;
      }
      if( ((gOutChannelGroup >> 2) & 0x01 ) == 1 ) {
	int64_t gOutIndex    = gOutGroupOffset + ((n * gOutChannel + k) * gOutHeight) * gOutWidth ;
	const float *pKerValue = pKernel + kernGroupOffset + (((k  ) * gInChannelGroup + c) * kernHeight) * kernWidth ;

	__vr vrgout_ptr_k0_s0 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry), vrx_s0),
					   2,
					   (unsigned long)(pGOut+gOutIndex)) ;
	__vr vrgout_k0_s0 = _ve_vgtu_vvm(vrgout_ptr_k0_s0, vmall_s0) ;
	__vr vrgout_ptr_k1_s0 = _ve_vaddsl_vsv(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_s0) ;
	__vr vrgout_k1_s0 = _ve_vgtu_vvm(vrgout_ptr_k1_s0, vmall_s0) ;
	__vr vrgout_ptr_k2_s0 = _ve_vaddsl_vsv(4*2*gOutHeight*gOutWidth, vrgout_ptr_k0_s0) ;
	__vr vrgout_k2_s0 = _ve_vgtu_vvm(vrgout_ptr_k2_s0, vmall_s0) ;
	__vr vrgout_ptr_k3_s0 = _ve_vaddsl_vsv(4*3*gOutHeight*gOutWidth, vrgout_ptr_k0_s0) ;
	__vr vrgout_k3_s0 = _ve_vgtu_vvm(vrgout_ptr_k3_s0, vmall_s0) ;

	VFADD_C8(vrgout_k0_s0, vmall_s0, 0, r, 0)
	VFADD_C8(vrgout_k1_s0, vmall_s0, 1, r, 0)
	VFADD_C8(vrgout_k2_s0, vmall_s0, 2, r, 0)
	VFADD_C8(vrgout_k3_s0, vmall_s0, 3, r, 0)

	__vr vrgout_ptr_k0_s1 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry), vrx_s1),
					   2,
					   (unsigned long)(pGOut+gOutIndex)) ;
	__vr vrgout_k0_s1 = _ve_vgtu_vvm(vrgout_ptr_k0_s1, vmall_s1) ;
	__vr vrgout_ptr_k1_s1 = _ve_vaddsl_vsv(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_s1) ;
	__vr vrgout_k1_s1 = _ve_vgtu_vvm(vrgout_ptr_k1_s1, vmall_s1) ;
	__vr vrgout_ptr_k2_s1 = _ve_vaddsl_vsv(4*2*gOutHeight*gOutWidth, vrgout_ptr_k0_s1) ;
	__vr vrgout_k2_s1 = _ve_vgtu_vvm(vrgout_ptr_k2_s1, vmall_s1) ;
	__vr vrgout_ptr_k3_s1 = _ve_vaddsl_vsv(4*3*gOutHeight*gOutWidth, vrgout_ptr_k0_s1) ;
	__vr vrgout_k3_s1 = _ve_vgtu_vvm(vrgout_ptr_k3_s1, vmall_s1) ;

	VFADD_C8(vrgout_k0_s1, vmall_s1, 0, r, 1)
	VFADD_C8(vrgout_k1_s1, vmall_s1, 1, r, 1)
	VFADD_C8(vrgout_k2_s1, vmall_s1, 2, r, 1)
	VFADD_C8(vrgout_k3_s1, vmall_s1, 3, r, 1)


	__vr vrgout_ptr_k0_s2 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry), vrx_s2),
					   2,
					   (unsigned long)(pGOut+gOutIndex)) ;
	__vr vrgout_k0_s2 = _ve_vgtu_vvm(vrgout_ptr_k0_s2, vmall_s2) ;
	__vr vrgout_ptr_k1_s2 = _ve_vaddsl_vsv(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_s2) ;
	__vr vrgout_k1_s2 = _ve_vgtu_vvm(vrgout_ptr_k1_s2, vmall_s2) ;
	__vr vrgout_ptr_k2_s2 = _ve_vaddsl_vsv(4*2*gOutHeight*gOutWidth, vrgout_ptr_k0_s2) ;
	__vr vrgout_k2_s2 = _ve_vgtu_vvm(vrgout_ptr_k2_s2, vmall_s2) ;
	__vr vrgout_ptr_k3_s2 = _ve_vaddsl_vsv(4*3*gOutHeight*gOutWidth, vrgout_ptr_k0_s2) ;
	__vr vrgout_k3_s2 = _ve_vgtu_vvm(vrgout_ptr_k3_s2, vmall_s2) ;

	VFADD_C8(vrgout_k0_s2, vmall_s2, 0, r, 2)
	VFADD_C8(vrgout_k1_s2, vmall_s2, 1, r, 2)
	VFADD_C8(vrgout_k2_s2, vmall_s2, 2, r, 2)
	VFADD_C8(vrgout_k3_s2, vmall_s2, 3, r, 2)

	__vr vrgout_ptr_k0_s3 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry), vrx_s3),
					   2,
					   (unsigned long)(pGOut+gOutIndex)) ;
	__vr vrgout_k0_s3 = _ve_vgtu_vvm(vrgout_ptr_k0_s3, vmall_s3) ;
	__vr vrgout_ptr_k1_s3 = _ve_vaddsl_vsv(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_s3) ;
	__vr vrgout_k1_s3 = _ve_vgtu_vvm(vrgout_ptr_k1_s3, vmall_s3) ;
	__vr vrgout_ptr_k2_s3 = _ve_vaddsl_vsv(4*2*gOutHeight*gOutWidth, vrgout_ptr_k0_s3) ;
	__vr vrgout_k2_s3 = _ve_vgtu_vvm(vrgout_ptr_k2_s3, vmall_s3) ;
	__vr vrgout_ptr_k3_s3 = _ve_vaddsl_vsv(4*3*gOutHeight*gOutWidth, vrgout_ptr_k0_s3) ;
	__vr vrgout_k3_s3 = _ve_vgtu_vvm(vrgout_ptr_k3_s3, vmall_s3) ;
	__vr vrgout_ptr_k4_s3 = _ve_vaddsl_vsv(4*4*gOutHeight*gOutWidth, vrgout_ptr_k0_s3) ;

	VFADD_C8(vrgout_k0_s3, vmall_s3, 0, r, 3)
	VFADD_C8(vrgout_k1_s3, vmall_s3, 1, r, 3)
	VFADD_C8(vrgout_k2_s3, vmall_s3, 2, r, 3)
	VFADD_C8(vrgout_k3_s3, vmall_s3, 3, r, 3)

	__vr vrgout_ptr_k0_s4 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry), vrx_s4),
					   2,
					   (unsigned long)(pGOut+gOutIndex)) ;
	__vr vrgout_k0_s4 = _ve_vgtu_vvm(vrgout_ptr_k0_s4, vmall_s4) ;
	__vr vrgout_ptr_k1_s4 = _ve_vaddsl_vsv(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_s4) ;
	__vr vrgout_k1_s4 = _ve_vgtu_vvm(vrgout_ptr_k1_s4, vmall_s4) ;
	__vr vrgout_ptr_k2_s4 = _ve_vaddsl_vsv(4*2*gOutHeight*gOutWidth, vrgout_ptr_k0_s4) ;
	__vr vrgout_k2_s4 = _ve_vgtu_vvm(vrgout_ptr_k2_s4, vmall_s4) ;
	__vr vrgout_ptr_k3_s4 = _ve_vaddsl_vsv(4*3*gOutHeight*gOutWidth, vrgout_ptr_k0_s4) ;
	__vr vrgout_k3_s4 = _ve_vgtu_vvm(vrgout_ptr_k3_s4, vmall_s4) ;

	VFADD_C8(vrgout_k0_s4, vmall_s4, 0, r, 4)
	VFADD_C8(vrgout_k1_s4, vmall_s4, 1, r, 4)
	VFADD_C8(vrgout_k2_s4, vmall_s4, 2, r, 4)
	VFADD_C8(vrgout_k3_s4, vmall_s4, 3, r, 4)

	k+=4 ;
      }
      for (; k<gOutChannelGroup; k+=8) {
	int64_t gOutIndex    = gOutGroupOffset + ((n * gOutChannel + k) * gOutHeight) * gOutWidth ;
	const float *pKerValue = pKernel + kernGroupOffset + (((k  ) * gInChannelGroup + c) * kernHeight) * kernWidth ;

	__vr vrgout_ptr_k0_s0 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry), vrx_s0),
					   2,
					   (unsigned long)(pGOut+gOutIndex)) ;
	__vr vrgout_k0_s0 = _ve_vgtu_vvm(vrgout_ptr_k0_s0, vmall_s0) ;
	__vr vrgout_ptr_k1_s0 = _ve_vaddsl_vsv(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_s0) ;
	__vr vrgout_k1_s0 = _ve_vgtu_vvm(vrgout_ptr_k1_s0, vmall_s0) ;
	__vr vrgout_ptr_k2_s0 = _ve_vaddsl_vsv(4*2*gOutHeight*gOutWidth, vrgout_ptr_k0_s0) ;
	__vr vrgout_k2_s0 = _ve_vgtu_vvm(vrgout_ptr_k2_s0, vmall_s0) ;
	__vr vrgout_ptr_k3_s0 = _ve_vaddsl_vsv(4*3*gOutHeight*gOutWidth, vrgout_ptr_k0_s0) ;
	__vr vrgout_k3_s0 = _ve_vgtu_vvm(vrgout_ptr_k3_s0, vmall_s0) ;
	__vr vrgout_ptr_k4_s0 = _ve_vaddsl_vsv(4*4*gOutHeight*gOutWidth, vrgout_ptr_k0_s0) ;
	__vr vrgout_k4_s0 = _ve_vgtu_vvm(vrgout_ptr_k4_s0, vmall_s0) ;
	__vr vrgout_ptr_k5_s0 = _ve_vaddsl_vsv(4*5*gOutHeight*gOutWidth, vrgout_ptr_k0_s0) ;
	__vr vrgout_k5_s0 = _ve_vgtu_vvm(vrgout_ptr_k5_s0, vmall_s0) ;
	__vr vrgout_ptr_k6_s0 = _ve_vaddsl_vsv(4*6*gOutHeight*gOutWidth, vrgout_ptr_k0_s0) ;
	__vr vrgout_k6_s0 = _ve_vgtu_vvm(vrgout_ptr_k6_s0, vmall_s0) ;
	__vr vrgout_ptr_k7_s0 = _ve_vaddsl_vsv(4*7*gOutHeight*gOutWidth, vrgout_ptr_k0_s0) ;
	__vr vrgout_k7_s0 = _ve_vgtu_vvm(vrgout_ptr_k7_s0, vmall_s0) ;

	VFADD_C8(vrgout_k0_s0, vmall_s0, 0, r, 0)
	VFADD_C8(vrgout_k1_s0, vmall_s0, 1, r, 0)
	VFADD_C8(vrgout_k2_s0, vmall_s0, 2, r, 0)
	VFADD_C8(vrgout_k3_s0, vmall_s0, 3, r, 0)
	VFADD_C8(vrgout_k4_s0, vmall_s0, 4, r, 0)
	VFADD_C8(vrgout_k5_s0, vmall_s0, 5, r, 0)
	VFADD_C8(vrgout_k6_s0, vmall_s0, 6, r, 0)
	VFADD_C8(vrgout_k7_s0, vmall_s0, 7, r, 0)

	__vr vrgout_ptr_k0_s1 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry), vrx_s1),
					   2,
					   (unsigned long)(pGOut+gOutIndex)) ;
	__vr vrgout_k0_s1 = _ve_vgtu_vvm(vrgout_ptr_k0_s1, vmall_s1) ;
	__vr vrgout_ptr_k1_s1 = _ve_vaddsl_vsv(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_s1) ;
	__vr vrgout_k1_s1 = _ve_vgtu_vvm(vrgout_ptr_k1_s1, vmall_s1) ;
	__vr vrgout_ptr_k2_s1 = _ve_vaddsl_vsv(4*2*gOutHeight*gOutWidth, vrgout_ptr_k0_s1) ;
	__vr vrgout_k2_s1 = _ve_vgtu_vvm(vrgout_ptr_k2_s1, vmall_s1) ;
	__vr vrgout_ptr_k3_s1 = _ve_vaddsl_vsv(4*3*gOutHeight*gOutWidth, vrgout_ptr_k0_s1) ;
	__vr vrgout_k3_s1 = _ve_vgtu_vvm(vrgout_ptr_k3_s1, vmall_s1) ;
	__vr vrgout_ptr_k4_s1 = _ve_vaddsl_vsv(4*4*gOutHeight*gOutWidth, vrgout_ptr_k0_s1) ;
	__vr vrgout_k4_s1 = _ve_vgtu_vvm(vrgout_ptr_k4_s1, vmall_s1) ;
	__vr vrgout_ptr_k5_s1 = _ve_vaddsl_vsv(4*5*gOutHeight*gOutWidth, vrgout_ptr_k0_s1) ;
	__vr vrgout_k5_s1 = _ve_vgtu_vvm(vrgout_ptr_k5_s1, vmall_s1) ;
	__vr vrgout_ptr_k6_s1 = _ve_vaddsl_vsv(4*6*gOutHeight*gOutWidth, vrgout_ptr_k0_s1) ;
	__vr vrgout_k6_s1 = _ve_vgtu_vvm(vrgout_ptr_k6_s1, vmall_s1) ;
	__vr vrgout_ptr_k7_s1 = _ve_vaddsl_vsv(4*7*gOutHeight*gOutWidth, vrgout_ptr_k0_s1) ;
	__vr vrgout_k7_s1 = _ve_vgtu_vvm(vrgout_ptr_k7_s1, vmall_s1) ;

	VFADD_C8(vrgout_k0_s1, vmall_s1, 0, r, 1)
	VFADD_C8(vrgout_k1_s1, vmall_s1, 1, r, 1)
	VFADD_C8(vrgout_k2_s1, vmall_s1, 2, r, 1)
	VFADD_C8(vrgout_k3_s1, vmall_s1, 3, r, 1)
	VFADD_C8(vrgout_k4_s1, vmall_s1, 4, r, 1)
	VFADD_C8(vrgout_k5_s1, vmall_s1, 5, r, 1)
	VFADD_C8(vrgout_k6_s1, vmall_s1, 6, r, 1)
	VFADD_C8(vrgout_k7_s1, vmall_s1, 7, r, 1)


	__vr vrgout_ptr_k0_s2 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry), vrx_s2),
					   2,
					   (unsigned long)(pGOut+gOutIndex)) ;
	__vr vrgout_k0_s2 = _ve_vgtu_vvm(vrgout_ptr_k0_s2, vmall_s2) ;
	__vr vrgout_ptr_k1_s2 = _ve_vaddsl_vsv(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_s2) ;
	__vr vrgout_k1_s2 = _ve_vgtu_vvm(vrgout_ptr_k1_s2, vmall_s2) ;
	__vr vrgout_ptr_k2_s2 = _ve_vaddsl_vsv(4*2*gOutHeight*gOutWidth, vrgout_ptr_k0_s2) ;
	__vr vrgout_k2_s2 = _ve_vgtu_vvm(vrgout_ptr_k2_s2, vmall_s2) ;
	__vr vrgout_ptr_k3_s2 = _ve_vaddsl_vsv(4*3*gOutHeight*gOutWidth, vrgout_ptr_k0_s2) ;
	__vr vrgout_k3_s2 = _ve_vgtu_vvm(vrgout_ptr_k3_s2, vmall_s2) ;
	__vr vrgout_ptr_k4_s2 = _ve_vaddsl_vsv(4*4*gOutHeight*gOutWidth, vrgout_ptr_k0_s2) ;
	__vr vrgout_k4_s2 = _ve_vgtu_vvm(vrgout_ptr_k4_s2, vmall_s2) ;
	__vr vrgout_ptr_k5_s2 = _ve_vaddsl_vsv(4*5*gOutHeight*gOutWidth, vrgout_ptr_k0_s2) ;
	__vr vrgout_k5_s2 = _ve_vgtu_vvm(vrgout_ptr_k5_s2, vmall_s2) ;
	__vr vrgout_ptr_k6_s2 = _ve_vaddsl_vsv(4*6*gOutHeight*gOutWidth, vrgout_ptr_k0_s2) ;
	__vr vrgout_k6_s2 = _ve_vgtu_vvm(vrgout_ptr_k6_s2, vmall_s2) ;
	__vr vrgout_ptr_k7_s2 = _ve_vaddsl_vsv(4*7*gOutHeight*gOutWidth, vrgout_ptr_k0_s2) ;
	__vr vrgout_k7_s2 = _ve_vgtu_vvm(vrgout_ptr_k7_s2, vmall_s2) ;

	VFADD_C8(vrgout_k0_s2, vmall_s2, 0, r, 2)
	VFADD_C8(vrgout_k1_s2, vmall_s2, 1, r, 2)
	VFADD_C8(vrgout_k2_s2, vmall_s2, 2, r, 2)
	VFADD_C8(vrgout_k3_s2, vmall_s2, 3, r, 2)
	VFADD_C8(vrgout_k4_s2, vmall_s2, 4, r, 2)
	VFADD_C8(vrgout_k5_s2, vmall_s2, 5, r, 2)
	VFADD_C8(vrgout_k6_s2, vmall_s2, 6, r, 2)
	VFADD_C8(vrgout_k7_s2, vmall_s2, 7, r, 2)


	__vr vrgout_ptr_k0_s3 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry), vrx_s3),
					   2,
					   (unsigned long)(pGOut+gOutIndex)) ;
	__vr vrgout_k0_s3 = _ve_vgtu_vvm(vrgout_ptr_k0_s3, vmall_s3) ;
	__vr vrgout_ptr_k1_s3 = _ve_vaddsl_vsv(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_s3) ;
	__vr vrgout_k1_s3 = _ve_vgtu_vvm(vrgout_ptr_k1_s3, vmall_s3) ;
	__vr vrgout_ptr_k2_s3 = _ve_vaddsl_vsv(4*2*gOutHeight*gOutWidth, vrgout_ptr_k0_s3) ;
	__vr vrgout_k2_s3 = _ve_vgtu_vvm(vrgout_ptr_k2_s3, vmall_s3) ;
	__vr vrgout_ptr_k3_s3 = _ve_vaddsl_vsv(4*3*gOutHeight*gOutWidth, vrgout_ptr_k0_s3) ;
	__vr vrgout_k3_s3 = _ve_vgtu_vvm(vrgout_ptr_k3_s3, vmall_s3) ;
	__vr vrgout_ptr_k4_s3 = _ve_vaddsl_vsv(4*4*gOutHeight*gOutWidth, vrgout_ptr_k0_s3) ;
	__vr vrgout_k4_s3 = _ve_vgtu_vvm(vrgout_ptr_k4_s3, vmall_s3) ;
	__vr vrgout_ptr_k5_s3 = _ve_vaddsl_vsv(4*5*gOutHeight*gOutWidth, vrgout_ptr_k0_s3) ;
	__vr vrgout_k5_s3 = _ve_vgtu_vvm(vrgout_ptr_k5_s3, vmall_s3) ;
	__vr vrgout_ptr_k6_s3 = _ve_vaddsl_vsv(4*6*gOutHeight*gOutWidth, vrgout_ptr_k0_s3) ;
	__vr vrgout_k6_s3 = _ve_vgtu_vvm(vrgout_ptr_k6_s3, vmall_s3) ;
	__vr vrgout_ptr_k7_s3 = _ve_vaddsl_vsv(4*7*gOutHeight*gOutWidth, vrgout_ptr_k0_s3) ;
	__vr vrgout_k7_s3 = _ve_vgtu_vvm(vrgout_ptr_k7_s3, vmall_s3) ;

	VFADD_C8(vrgout_k0_s3, vmall_s3, 0, r, 3)
	VFADD_C8(vrgout_k1_s3, vmall_s3, 1, r, 3)
	VFADD_C8(vrgout_k2_s3, vmall_s3, 2, r, 3)
	VFADD_C8(vrgout_k3_s3, vmall_s3, 3, r, 3)
	VFADD_C8(vrgout_k4_s3, vmall_s3, 4, r, 3)
	VFADD_C8(vrgout_k5_s3, vmall_s3, 5, r, 3)
	VFADD_C8(vrgout_k6_s3, vmall_s3, 6, r, 3)
	VFADD_C8(vrgout_k7_s3, vmall_s3, 7, r, 3)


	__vr vrgout_ptr_k0_s4 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry), vrx_s4),
					   2,
					   (unsigned long)(pGOut+gOutIndex)) ;
	__vr vrgout_k0_s4 = _ve_vgtu_vvm(vrgout_ptr_k0_s4, vmall_s4) ;
	__vr vrgout_ptr_k1_s4 = _ve_vaddsl_vsv(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_s4) ;
	__vr vrgout_k1_s4 = _ve_vgtu_vvm(vrgout_ptr_k1_s4, vmall_s4) ;
	__vr vrgout_ptr_k2_s4 = _ve_vaddsl_vsv(4*2*gOutHeight*gOutWidth, vrgout_ptr_k0_s4) ;
	__vr vrgout_k2_s4 = _ve_vgtu_vvm(vrgout_ptr_k2_s4, vmall_s4) ;
	__vr vrgout_ptr_k3_s4 = _ve_vaddsl_vsv(4*3*gOutHeight*gOutWidth, vrgout_ptr_k0_s4) ;
	__vr vrgout_k3_s4 = _ve_vgtu_vvm(vrgout_ptr_k3_s4, vmall_s4) ;
	__vr vrgout_ptr_k4_s4 = _ve_vaddsl_vsv(4*4*gOutHeight*gOutWidth, vrgout_ptr_k0_s4) ;
	__vr vrgout_k4_s4 = _ve_vgtu_vvm(vrgout_ptr_k4_s4, vmall_s4) ;
	__vr vrgout_ptr_k5_s4 = _ve_vaddsl_vsv(4*5*gOutHeight*gOutWidth, vrgout_ptr_k0_s4) ;
	__vr vrgout_k5_s4 = _ve_vgtu_vvm(vrgout_ptr_k5_s4, vmall_s4) ;
	__vr vrgout_ptr_k6_s4 = _ve_vaddsl_vsv(4*6*gOutHeight*gOutWidth, vrgout_ptr_k0_s4) ;
	__vr vrgout_k6_s4 = _ve_vgtu_vvm(vrgout_ptr_k6_s4, vmall_s4) ;
	__vr vrgout_ptr_k7_s4 = _ve_vaddsl_vsv(4*7*gOutHeight*gOutWidth, vrgout_ptr_k0_s4) ;
	__vr vrgout_k7_s4 = _ve_vgtu_vvm(vrgout_ptr_k7_s4, vmall_s4) ;

	VFADD_C8(vrgout_k0_s4, vmall_s4, 0, r, 4)
	VFADD_C8(vrgout_k1_s4, vmall_s4, 1, r, 4)
	VFADD_C8(vrgout_k2_s4, vmall_s4, 2, r, 4)
	VFADD_C8(vrgout_k3_s4, vmall_s4, 3, r, 4)
	VFADD_C8(vrgout_k4_s4, vmall_s4, 4, r, 4)
	VFADD_C8(vrgout_k5_s4, vmall_s4, 5, r, 4)
	VFADD_C8(vrgout_k6_s4, vmall_s4, 6, r, 4)
	VFADD_C8(vrgout_k7_s4, vmall_s4, 7, r, 4)

#undef VFADD_C8
      } // gOutChannel
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
    __vr vrsum89 = _ve_pvbrd_vs_i64(0UL) ;
    __vr vrsumAB = _ve_pvbrd_vs_i64(0UL) ;
    __vr vrsumCD = _ve_pvbrd_vs_i64(0UL) ;
    __vr vrsumEF = _ve_pvbrd_vs_i64(0UL) ;

    __vr vrj_s0 = _ve_vaddsl_vsv(padWidth-0*dilationWidth, vrw) ;
    __vr vrj_s1 = _ve_vaddsl_vsv(padWidth-1*dilationWidth, vrw) ;
    __vr vrj_s2 = _ve_vaddsl_vsv(padWidth-2*dilationWidth, vrw) ;
    __vr vrj_s3 = _ve_vaddsl_vsv(padWidth-3*dilationWidth, vrw) ;
    __vr vrj_s4 = _ve_vaddsl_vsv(padWidth-4*dilationWidth, vrw) ;

    __vr vrx_s0 = _ve_vdivsl_vvs(vrj_s0, strideWidth) ;
    __vr vrx_s1 = _ve_vdivsl_vvs(vrj_s1, strideWidth) ;
    __vr vrx_s2 = _ve_vdivsl_vvs(vrj_s2, strideWidth) ;
    __vr vrx_s3 = _ve_vdivsl_vvs(vrj_s3, strideWidth) ;
    __vr vrx_s4 = _ve_vdivsl_vvs(vrj_s4, strideWidth) ;

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

    __vm256 vmx0_s3 = _ve_vfmkl_mcv(VECC_IEQ, _ve_vcmpsl_vvv(vrj_s3, _ve_vmulsl_vsv(strideWidth, vrx_s3))) ;
    __vm256 vmx1_s3 = _ve_vfmkl_mcv(VECC_GE, vrx_s3) ;
    __vm256 vmx2_s3 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(gOutWidth,vrx_s3)) ;
    __vm256 vmx_s3 = _ve_andm_mmm(_ve_andm_mmm(vmx0_s3, vmx1_s3), vmx2_s3) ;

    __vm256 vmx0_s4 = _ve_vfmkl_mcv(VECC_IEQ, _ve_vcmpsl_vvv(vrj_s4, _ve_vmulsl_vsv(strideWidth, vrx_s4))) ;
    __vm256 vmx1_s4 = _ve_vfmkl_mcv(VECC_GE, vrx_s4) ;
    __vm256 vmx2_s4 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(gOutWidth,vrx_s4)) ;
    __vm256 vmx_s4 = _ve_andm_mmm(_ve_andm_mmm(vmx0_s4, vmx1_s4), vmx2_s4) ;

    for (int64_t r=0; r<kernHeight; r++) {
      __vr vri = _ve_vaddsl_vsv(padHeight-r*dilationHeight+h, vrh) ;
      __vr vry = _ve_vdivsl_vvs(vri, strideHeight) ;

      __vm256 vmy0 = _ve_vfmkl_mcv(VECC_IEQ, _ve_vcmpsl_vvv(vri, _ve_vmulsl_vsv(strideHeight, vry))) ;
      __vm256 vmy1 = _ve_vfmkl_mcv(VECC_GE, vry) ;
      __vm256 vmy2 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(gOutHeight,vry)) ;

      __vm256 vmy = _ve_andm_mmm(_ve_andm_mmm(vmy0, vmy1), vmy2) ;


      __vm256 vmall_s0 = _ve_andm_mmm(vmy,vmx_s0) ;
      __vm256 vmall_s1 = _ve_andm_mmm(vmy,vmx_s1) ;
      __vm256 vmall_s2 = _ve_andm_mmm(vmy,vmx_s2) ;
      __vm256 vmall_s3 = _ve_andm_mmm(vmy,vmx_s3) ;
      __vm256 vmall_s4 = _ve_andm_mmm(vmy,vmx_s4) ;

      int64_t k=0;
      if( (gOutChannelGroup & 0x01 ) == 1 ) {
	int64_t gOutIndex    = gOutGroupOffset + ((n * gOutChannel + k) * gOutHeight) * gOutWidth ;
	const float *pKerValue = pKernel + kernGroupOffset + (((k  ) * gInChannelGroup + c) * kernHeight) * kernWidth ;


#define VFADD_C16(VRGOUT, VM, K, R, S)  {												\
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
	  VRGOUT = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), VRGOUT, VM) ;				\
	  __vr vrgoutP = _ve_vshf_vvvs(VRGOUT, VRGOUT, VE_VSHUFFLE_YUZU) ;			\
	  vrsum01 = _ve_pvfmad_vvsv(vrsum01, kerValue01, vrgoutP) ;				\
	  vrsum23 = _ve_pvfmad_vvsv(vrsum23, kerValue23, vrgoutP) ;				\
	  vrsum45 = _ve_pvfmad_vvsv(vrsum45, kerValue45, vrgoutP) ;				\
	  vrsum67 = _ve_pvfmad_vvsv(vrsum67, kerValue67, vrgoutP) ;				\
	  vrsum89 = _ve_pvfmad_vvsv(vrsum89, kerValue89, vrgoutP) ;				\
	  vrsumAB = _ve_pvfmad_vvsv(vrsumAB, kerValueAB, vrgoutP) ;				\
	  vrsumCD = _ve_pvfmad_vvsv(vrsumCD, kerValueCD, vrgoutP) ;				\
	  vrsumEF = _ve_pvfmad_vvsv(vrsumEF, kerValueEF, vrgoutP) ;				\
	}

	__vr vrgout_ptr_k0_s0 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry), vrx_s0),
					   2,
					   (unsigned long)(pGOut+gOutIndex)) ;
	__vr vrgout_k0_s0 = _ve_vgtu_vvm(vrgout_ptr_k0_s0, vmall_s0) ;
	__vr vrgout_ptr_k1_s0 = _ve_vaddsl_vsv(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_s0) ;
	__vr vrgout_k1_s0 = _ve_vgtu_vvm(vrgout_ptr_k1_s0, vmall_s0) ;

	VFADD_C16(vrgout_k0_s0, vmall_s0, 0, r, 0)

	__vr vrgout_ptr_k0_s1 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry), vrx_s1),
					   2,
					   (unsigned long)(pGOut+gOutIndex)) ;
	__vr vrgout_k0_s1 = _ve_vgtu_vvm(vrgout_ptr_k0_s1, vmall_s1) ;

	VFADD_C16(vrgout_k0_s1, vmall_s1, 0, r, 1)


	__vr vrgout_ptr_k0_s2 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry), vrx_s2),
					   2,
					   (unsigned long)(pGOut+gOutIndex)) ;
	__vr vrgout_k0_s2 = _ve_vgtu_vvm(vrgout_ptr_k0_s2, vmall_s2) ;

	VFADD_C16(vrgout_k0_s2, vmall_s2, 0, r, 2)

	__vr vrgout_ptr_k0_s3 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry), vrx_s3),
					   2,
					   (unsigned long)(pGOut+gOutIndex)) ;
	__vr vrgout_k0_s3 = _ve_vgtu_vvm(vrgout_ptr_k0_s3, vmall_s3) ;
	__vr vrgout_ptr_k1_s3 = _ve_vaddsl_vsv(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_s3) ;

	VFADD_C16(vrgout_k0_s3, vmall_s3, 0, r, 3)

	__vr vrgout_ptr_k0_s4 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry), vrx_s4),
					   2,
					   (unsigned long)(pGOut+gOutIndex)) ;
	__vr vrgout_k0_s4 = _ve_vgtu_vvm(vrgout_ptr_k0_s4, vmall_s4) ;

	VFADD_C16(vrgout_k0_s4, vmall_s4, 0, r, 4)

	k+=1 ;
      }
      if( ((gOutChannelGroup >> 1) & 0x01 ) == 1 ) {
	int64_t gOutIndex    = gOutGroupOffset + ((n * gOutChannel + k) * gOutHeight) * gOutWidth ;
	const float *pKerValue = pKernel + kernGroupOffset + (((k  ) * gInChannelGroup + c) * kernHeight) * kernWidth ;

	__vr vrgout_ptr_k0_s0 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry), vrx_s0),
					   2,
					   (unsigned long)(pGOut+gOutIndex)) ;
	__vr vrgout_k0_s0 = _ve_vgtu_vvm(vrgout_ptr_k0_s0, vmall_s0) ;
	__vr vrgout_ptr_k1_s0 = _ve_vaddsl_vsv(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_s0) ;
	__vr vrgout_k1_s0 = _ve_vgtu_vvm(vrgout_ptr_k1_s0, vmall_s0) ;

	VFADD_C16(vrgout_k0_s0, vmall_s0, 0, r, 0)
	VFADD_C16(vrgout_k1_s0, vmall_s0, 1, r, 0)

	__vr vrgout_ptr_k0_s1 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry), vrx_s1),
					   2,
					   (unsigned long)(pGOut+gOutIndex)) ;
	__vr vrgout_k0_s1 = _ve_vgtu_vvm(vrgout_ptr_k0_s1, vmall_s1) ;
	__vr vrgout_ptr_k1_s1 = _ve_vaddsl_vsv(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_s1) ;
	__vr vrgout_k1_s1 = _ve_vgtu_vvm(vrgout_ptr_k1_s1, vmall_s1) ;

	VFADD_C16(vrgout_k0_s1, vmall_s1, 0, r, 1)
	VFADD_C16(vrgout_k1_s1, vmall_s1, 1, r, 1)


	__vr vrgout_ptr_k0_s2 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry), vrx_s2),
					   2,
					   (unsigned long)(pGOut+gOutIndex)) ;
	__vr vrgout_k0_s2 = _ve_vgtu_vvm(vrgout_ptr_k0_s2, vmall_s2) ;
	__vr vrgout_ptr_k1_s2 = _ve_vaddsl_vsv(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_s2) ;
	__vr vrgout_k1_s2 = _ve_vgtu_vvm(vrgout_ptr_k1_s2, vmall_s2) ;

	VFADD_C16(vrgout_k0_s2, vmall_s2, 0, r, 2)
	VFADD_C16(vrgout_k1_s2, vmall_s2, 1, r, 2)

	__vr vrgout_ptr_k0_s3 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry), vrx_s3),
					   2,
					   (unsigned long)(pGOut+gOutIndex)) ;
	__vr vrgout_k0_s3 = _ve_vgtu_vvm(vrgout_ptr_k0_s3, vmall_s3) ;
	__vr vrgout_ptr_k1_s3 = _ve_vaddsl_vsv(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_s3) ;
	__vr vrgout_k1_s3 = _ve_vgtu_vvm(vrgout_ptr_k1_s3, vmall_s3) ;
	__vr vrgout_ptr_k2_s3 = _ve_vaddsl_vsv(4*2*gOutHeight*gOutWidth, vrgout_ptr_k0_s3) ;

	VFADD_C16(vrgout_k0_s3, vmall_s3, 0, r, 3)
	VFADD_C16(vrgout_k1_s3, vmall_s3, 1, r, 3)

	__vr vrgout_ptr_k0_s4 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry), vrx_s4),
					   2,
					   (unsigned long)(pGOut+gOutIndex)) ;
	__vr vrgout_k0_s4 = _ve_vgtu_vvm(vrgout_ptr_k0_s4, vmall_s4) ;
	__vr vrgout_ptr_k1_s4 = _ve_vaddsl_vsv(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_s4) ;
	__vr vrgout_k1_s4 = _ve_vgtu_vvm(vrgout_ptr_k1_s4, vmall_s4) ;

	VFADD_C16(vrgout_k0_s4, vmall_s4, 0, r, 4)
	VFADD_C16(vrgout_k1_s4, vmall_s4, 1, r, 4)

	k+=2 ;
      }
      if( ((gOutChannelGroup >> 2) & 0x01 ) == 1 ) {
	int64_t gOutIndex    = gOutGroupOffset + ((n * gOutChannel + k) * gOutHeight) * gOutWidth ;
	const float *pKerValue = pKernel + kernGroupOffset + (((k  ) * gInChannelGroup + c) * kernHeight) * kernWidth ;

	__vr vrgout_ptr_k0_s0 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry), vrx_s0),
					   2,
					   (unsigned long)(pGOut+gOutIndex)) ;
	__vr vrgout_k0_s0 = _ve_vgtu_vvm(vrgout_ptr_k0_s0, vmall_s0) ;
	__vr vrgout_ptr_k1_s0 = _ve_vaddsl_vsv(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_s0) ;
	__vr vrgout_k1_s0 = _ve_vgtu_vvm(vrgout_ptr_k1_s0, vmall_s0) ;
	__vr vrgout_ptr_k2_s0 = _ve_vaddsl_vsv(4*2*gOutHeight*gOutWidth, vrgout_ptr_k0_s0) ;
	__vr vrgout_k2_s0 = _ve_vgtu_vvm(vrgout_ptr_k2_s0, vmall_s0) ;
	__vr vrgout_ptr_k3_s0 = _ve_vaddsl_vsv(4*3*gOutHeight*gOutWidth, vrgout_ptr_k0_s0) ;
	__vr vrgout_k3_s0 = _ve_vgtu_vvm(vrgout_ptr_k3_s0, vmall_s0) ;

	VFADD_C16(vrgout_k0_s0, vmall_s0, 0, r, 0)
	VFADD_C16(vrgout_k1_s0, vmall_s0, 1, r, 0)
	VFADD_C16(vrgout_k2_s0, vmall_s0, 2, r, 0)
	VFADD_C16(vrgout_k3_s0, vmall_s0, 3, r, 0)

	__vr vrgout_ptr_k0_s1 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry), vrx_s1),
					   2,
					   (unsigned long)(pGOut+gOutIndex)) ;
	__vr vrgout_k0_s1 = _ve_vgtu_vvm(vrgout_ptr_k0_s1, vmall_s1) ;
	__vr vrgout_ptr_k1_s1 = _ve_vaddsl_vsv(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_s1) ;
	__vr vrgout_k1_s1 = _ve_vgtu_vvm(vrgout_ptr_k1_s1, vmall_s1) ;
	__vr vrgout_ptr_k2_s1 = _ve_vaddsl_vsv(4*2*gOutHeight*gOutWidth, vrgout_ptr_k0_s1) ;
	__vr vrgout_k2_s1 = _ve_vgtu_vvm(vrgout_ptr_k2_s1, vmall_s1) ;
	__vr vrgout_ptr_k3_s1 = _ve_vaddsl_vsv(4*3*gOutHeight*gOutWidth, vrgout_ptr_k0_s1) ;
	__vr vrgout_k3_s1 = _ve_vgtu_vvm(vrgout_ptr_k3_s1, vmall_s1) ;

	VFADD_C16(vrgout_k0_s1, vmall_s1, 0, r, 1)
	VFADD_C16(vrgout_k1_s1, vmall_s1, 1, r, 1)
	VFADD_C16(vrgout_k2_s1, vmall_s1, 2, r, 1)
	VFADD_C16(vrgout_k3_s1, vmall_s1, 3, r, 1)


	__vr vrgout_ptr_k0_s2 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry), vrx_s2),
					   2,
					   (unsigned long)(pGOut+gOutIndex)) ;
	__vr vrgout_k0_s2 = _ve_vgtu_vvm(vrgout_ptr_k0_s2, vmall_s2) ;
	__vr vrgout_ptr_k1_s2 = _ve_vaddsl_vsv(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_s2) ;
	__vr vrgout_k1_s2 = _ve_vgtu_vvm(vrgout_ptr_k1_s2, vmall_s2) ;
	__vr vrgout_ptr_k2_s2 = _ve_vaddsl_vsv(4*2*gOutHeight*gOutWidth, vrgout_ptr_k0_s2) ;
	__vr vrgout_k2_s2 = _ve_vgtu_vvm(vrgout_ptr_k2_s2, vmall_s2) ;
	__vr vrgout_ptr_k3_s2 = _ve_vaddsl_vsv(4*3*gOutHeight*gOutWidth, vrgout_ptr_k0_s2) ;
	__vr vrgout_k3_s2 = _ve_vgtu_vvm(vrgout_ptr_k3_s2, vmall_s2) ;

	VFADD_C16(vrgout_k0_s2, vmall_s2, 0, r, 2)
	VFADD_C16(vrgout_k1_s2, vmall_s2, 1, r, 2)
	VFADD_C16(vrgout_k2_s2, vmall_s2, 2, r, 2)
	VFADD_C16(vrgout_k3_s2, vmall_s2, 3, r, 2)

	__vr vrgout_ptr_k0_s3 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry), vrx_s3),
					   2,
					   (unsigned long)(pGOut+gOutIndex)) ;
	__vr vrgout_k0_s3 = _ve_vgtu_vvm(vrgout_ptr_k0_s3, vmall_s3) ;
	__vr vrgout_ptr_k1_s3 = _ve_vaddsl_vsv(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_s3) ;
	__vr vrgout_k1_s3 = _ve_vgtu_vvm(vrgout_ptr_k1_s3, vmall_s3) ;
	__vr vrgout_ptr_k2_s3 = _ve_vaddsl_vsv(4*2*gOutHeight*gOutWidth, vrgout_ptr_k0_s3) ;
	__vr vrgout_k2_s3 = _ve_vgtu_vvm(vrgout_ptr_k2_s3, vmall_s3) ;
	__vr vrgout_ptr_k3_s3 = _ve_vaddsl_vsv(4*3*gOutHeight*gOutWidth, vrgout_ptr_k0_s3) ;
	__vr vrgout_k3_s3 = _ve_vgtu_vvm(vrgout_ptr_k3_s3, vmall_s3) ;
	__vr vrgout_ptr_k4_s3 = _ve_vaddsl_vsv(4*4*gOutHeight*gOutWidth, vrgout_ptr_k0_s3) ;

	VFADD_C16(vrgout_k0_s3, vmall_s3, 0, r, 3)
	VFADD_C16(vrgout_k1_s3, vmall_s3, 1, r, 3)
	VFADD_C16(vrgout_k2_s3, vmall_s3, 2, r, 3)
	VFADD_C16(vrgout_k3_s3, vmall_s3, 3, r, 3)

	__vr vrgout_ptr_k0_s4 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry), vrx_s4),
					   2,
					   (unsigned long)(pGOut+gOutIndex)) ;
	__vr vrgout_k0_s4 = _ve_vgtu_vvm(vrgout_ptr_k0_s4, vmall_s4) ;
	__vr vrgout_ptr_k1_s4 = _ve_vaddsl_vsv(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_s4) ;
	__vr vrgout_k1_s4 = _ve_vgtu_vvm(vrgout_ptr_k1_s4, vmall_s4) ;
	__vr vrgout_ptr_k2_s4 = _ve_vaddsl_vsv(4*2*gOutHeight*gOutWidth, vrgout_ptr_k0_s4) ;
	__vr vrgout_k2_s4 = _ve_vgtu_vvm(vrgout_ptr_k2_s4, vmall_s4) ;
	__vr vrgout_ptr_k3_s4 = _ve_vaddsl_vsv(4*3*gOutHeight*gOutWidth, vrgout_ptr_k0_s4) ;
	__vr vrgout_k3_s4 = _ve_vgtu_vvm(vrgout_ptr_k3_s4, vmall_s4) ;

	VFADD_C16(vrgout_k0_s4, vmall_s4, 0, r, 4)
	VFADD_C16(vrgout_k1_s4, vmall_s4, 1, r, 4)
	VFADD_C16(vrgout_k2_s4, vmall_s4, 2, r, 4)
	VFADD_C16(vrgout_k3_s4, vmall_s4, 3, r, 4)

	k+=4 ;
      }
      for (; k<gOutChannelGroup; k+=8) {
	int64_t gOutIndex    = gOutGroupOffset + ((n * gOutChannel + k) * gOutHeight) * gOutWidth ;
	const float *pKerValue = pKernel + kernGroupOffset + (((k  ) * gInChannelGroup + c) * kernHeight) * kernWidth ;

	__vr vrgout_ptr_k0_s0 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry), vrx_s0),
					   2,
					   (unsigned long)(pGOut+gOutIndex)) ;
	__vr vrgout_k0_s0 = _ve_vgtu_vvm(vrgout_ptr_k0_s0, vmall_s0) ;
	__vr vrgout_ptr_k1_s0 = _ve_vaddsl_vsv(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_s0) ;
	__vr vrgout_k1_s0 = _ve_vgtu_vvm(vrgout_ptr_k1_s0, vmall_s0) ;
	__vr vrgout_ptr_k2_s0 = _ve_vaddsl_vsv(4*2*gOutHeight*gOutWidth, vrgout_ptr_k0_s0) ;
	__vr vrgout_k2_s0 = _ve_vgtu_vvm(vrgout_ptr_k2_s0, vmall_s0) ;
	__vr vrgout_ptr_k3_s0 = _ve_vaddsl_vsv(4*3*gOutHeight*gOutWidth, vrgout_ptr_k0_s0) ;
	__vr vrgout_k3_s0 = _ve_vgtu_vvm(vrgout_ptr_k3_s0, vmall_s0) ;
	__vr vrgout_ptr_k4_s0 = _ve_vaddsl_vsv(4*4*gOutHeight*gOutWidth, vrgout_ptr_k0_s0) ;
	__vr vrgout_k4_s0 = _ve_vgtu_vvm(vrgout_ptr_k4_s0, vmall_s0) ;
	__vr vrgout_ptr_k5_s0 = _ve_vaddsl_vsv(4*5*gOutHeight*gOutWidth, vrgout_ptr_k0_s0) ;
	__vr vrgout_k5_s0 = _ve_vgtu_vvm(vrgout_ptr_k5_s0, vmall_s0) ;
	__vr vrgout_ptr_k6_s0 = _ve_vaddsl_vsv(4*6*gOutHeight*gOutWidth, vrgout_ptr_k0_s0) ;
	__vr vrgout_k6_s0 = _ve_vgtu_vvm(vrgout_ptr_k6_s0, vmall_s0) ;
	__vr vrgout_ptr_k7_s0 = _ve_vaddsl_vsv(4*7*gOutHeight*gOutWidth, vrgout_ptr_k0_s0) ;
	__vr vrgout_k7_s0 = _ve_vgtu_vvm(vrgout_ptr_k7_s0, vmall_s0) ;

	VFADD_C16(vrgout_k0_s0, vmall_s0, 0, r, 0)
	VFADD_C16(vrgout_k1_s0, vmall_s0, 1, r, 0)
	VFADD_C16(vrgout_k2_s0, vmall_s0, 2, r, 0)
	VFADD_C16(vrgout_k3_s0, vmall_s0, 3, r, 0)
	VFADD_C16(vrgout_k4_s0, vmall_s0, 4, r, 0)
	VFADD_C16(vrgout_k5_s0, vmall_s0, 5, r, 0)
	VFADD_C16(vrgout_k6_s0, vmall_s0, 6, r, 0)
	VFADD_C16(vrgout_k7_s0, vmall_s0, 7, r, 0)

	__vr vrgout_ptr_k0_s1 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry), vrx_s1),
					   2,
					   (unsigned long)(pGOut+gOutIndex)) ;
	__vr vrgout_k0_s1 = _ve_vgtu_vvm(vrgout_ptr_k0_s1, vmall_s1) ;
	__vr vrgout_ptr_k1_s1 = _ve_vaddsl_vsv(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_s1) ;
	__vr vrgout_k1_s1 = _ve_vgtu_vvm(vrgout_ptr_k1_s1, vmall_s1) ;
	__vr vrgout_ptr_k2_s1 = _ve_vaddsl_vsv(4*2*gOutHeight*gOutWidth, vrgout_ptr_k0_s1) ;
	__vr vrgout_k2_s1 = _ve_vgtu_vvm(vrgout_ptr_k2_s1, vmall_s1) ;
	__vr vrgout_ptr_k3_s1 = _ve_vaddsl_vsv(4*3*gOutHeight*gOutWidth, vrgout_ptr_k0_s1) ;
	__vr vrgout_k3_s1 = _ve_vgtu_vvm(vrgout_ptr_k3_s1, vmall_s1) ;
	__vr vrgout_ptr_k4_s1 = _ve_vaddsl_vsv(4*4*gOutHeight*gOutWidth, vrgout_ptr_k0_s1) ;
	__vr vrgout_k4_s1 = _ve_vgtu_vvm(vrgout_ptr_k4_s1, vmall_s1) ;
	__vr vrgout_ptr_k5_s1 = _ve_vaddsl_vsv(4*5*gOutHeight*gOutWidth, vrgout_ptr_k0_s1) ;
	__vr vrgout_k5_s1 = _ve_vgtu_vvm(vrgout_ptr_k5_s1, vmall_s1) ;
	__vr vrgout_ptr_k6_s1 = _ve_vaddsl_vsv(4*6*gOutHeight*gOutWidth, vrgout_ptr_k0_s1) ;
	__vr vrgout_k6_s1 = _ve_vgtu_vvm(vrgout_ptr_k6_s1, vmall_s1) ;
	__vr vrgout_ptr_k7_s1 = _ve_vaddsl_vsv(4*7*gOutHeight*gOutWidth, vrgout_ptr_k0_s1) ;
	__vr vrgout_k7_s1 = _ve_vgtu_vvm(vrgout_ptr_k7_s1, vmall_s1) ;

	VFADD_C16(vrgout_k0_s1, vmall_s1, 0, r, 1)
	VFADD_C16(vrgout_k1_s1, vmall_s1, 1, r, 1)
	VFADD_C16(vrgout_k2_s1, vmall_s1, 2, r, 1)
	VFADD_C16(vrgout_k3_s1, vmall_s1, 3, r, 1)
	VFADD_C16(vrgout_k4_s1, vmall_s1, 4, r, 1)
	VFADD_C16(vrgout_k5_s1, vmall_s1, 5, r, 1)
	VFADD_C16(vrgout_k6_s1, vmall_s1, 6, r, 1)
	VFADD_C16(vrgout_k7_s1, vmall_s1, 7, r, 1)


	__vr vrgout_ptr_k0_s2 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry), vrx_s2),
					   2,
					   (unsigned long)(pGOut+gOutIndex)) ;
	__vr vrgout_k0_s2 = _ve_vgtu_vvm(vrgout_ptr_k0_s2, vmall_s2) ;
	__vr vrgout_ptr_k1_s2 = _ve_vaddsl_vsv(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_s2) ;
	__vr vrgout_k1_s2 = _ve_vgtu_vvm(vrgout_ptr_k1_s2, vmall_s2) ;
	__vr vrgout_ptr_k2_s2 = _ve_vaddsl_vsv(4*2*gOutHeight*gOutWidth, vrgout_ptr_k0_s2) ;
	__vr vrgout_k2_s2 = _ve_vgtu_vvm(vrgout_ptr_k2_s2, vmall_s2) ;
	__vr vrgout_ptr_k3_s2 = _ve_vaddsl_vsv(4*3*gOutHeight*gOutWidth, vrgout_ptr_k0_s2) ;
	__vr vrgout_k3_s2 = _ve_vgtu_vvm(vrgout_ptr_k3_s2, vmall_s2) ;
	__vr vrgout_ptr_k4_s2 = _ve_vaddsl_vsv(4*4*gOutHeight*gOutWidth, vrgout_ptr_k0_s2) ;
	__vr vrgout_k4_s2 = _ve_vgtu_vvm(vrgout_ptr_k4_s2, vmall_s2) ;
	__vr vrgout_ptr_k5_s2 = _ve_vaddsl_vsv(4*5*gOutHeight*gOutWidth, vrgout_ptr_k0_s2) ;
	__vr vrgout_k5_s2 = _ve_vgtu_vvm(vrgout_ptr_k5_s2, vmall_s2) ;
	__vr vrgout_ptr_k6_s2 = _ve_vaddsl_vsv(4*6*gOutHeight*gOutWidth, vrgout_ptr_k0_s2) ;
	__vr vrgout_k6_s2 = _ve_vgtu_vvm(vrgout_ptr_k6_s2, vmall_s2) ;
	__vr vrgout_ptr_k7_s2 = _ve_vaddsl_vsv(4*7*gOutHeight*gOutWidth, vrgout_ptr_k0_s2) ;
	__vr vrgout_k7_s2 = _ve_vgtu_vvm(vrgout_ptr_k7_s2, vmall_s2) ;

	VFADD_C16(vrgout_k0_s2, vmall_s2, 0, r, 2)
	VFADD_C16(vrgout_k1_s2, vmall_s2, 1, r, 2)
	VFADD_C16(vrgout_k2_s2, vmall_s2, 2, r, 2)
	VFADD_C16(vrgout_k3_s2, vmall_s2, 3, r, 2)
	VFADD_C16(vrgout_k4_s2, vmall_s2, 4, r, 2)
	VFADD_C16(vrgout_k5_s2, vmall_s2, 5, r, 2)
	VFADD_C16(vrgout_k6_s2, vmall_s2, 6, r, 2)
	VFADD_C16(vrgout_k7_s2, vmall_s2, 7, r, 2)


	__vr vrgout_ptr_k0_s3 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry), vrx_s3),
					   2,
					   (unsigned long)(pGOut+gOutIndex)) ;
	__vr vrgout_k0_s3 = _ve_vgtu_vvm(vrgout_ptr_k0_s3, vmall_s3) ;
	__vr vrgout_ptr_k1_s3 = _ve_vaddsl_vsv(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_s3) ;
	__vr vrgout_k1_s3 = _ve_vgtu_vvm(vrgout_ptr_k1_s3, vmall_s3) ;
	__vr vrgout_ptr_k2_s3 = _ve_vaddsl_vsv(4*2*gOutHeight*gOutWidth, vrgout_ptr_k0_s3) ;
	__vr vrgout_k2_s3 = _ve_vgtu_vvm(vrgout_ptr_k2_s3, vmall_s3) ;
	__vr vrgout_ptr_k3_s3 = _ve_vaddsl_vsv(4*3*gOutHeight*gOutWidth, vrgout_ptr_k0_s3) ;
	__vr vrgout_k3_s3 = _ve_vgtu_vvm(vrgout_ptr_k3_s3, vmall_s3) ;
	__vr vrgout_ptr_k4_s3 = _ve_vaddsl_vsv(4*4*gOutHeight*gOutWidth, vrgout_ptr_k0_s3) ;
	__vr vrgout_k4_s3 = _ve_vgtu_vvm(vrgout_ptr_k4_s3, vmall_s3) ;
	__vr vrgout_ptr_k5_s3 = _ve_vaddsl_vsv(4*5*gOutHeight*gOutWidth, vrgout_ptr_k0_s3) ;
	__vr vrgout_k5_s3 = _ve_vgtu_vvm(vrgout_ptr_k5_s3, vmall_s3) ;
	__vr vrgout_ptr_k6_s3 = _ve_vaddsl_vsv(4*6*gOutHeight*gOutWidth, vrgout_ptr_k0_s3) ;
	__vr vrgout_k6_s3 = _ve_vgtu_vvm(vrgout_ptr_k6_s3, vmall_s3) ;
	__vr vrgout_ptr_k7_s3 = _ve_vaddsl_vsv(4*7*gOutHeight*gOutWidth, vrgout_ptr_k0_s3) ;
	__vr vrgout_k7_s3 = _ve_vgtu_vvm(vrgout_ptr_k7_s3, vmall_s3) ;

	VFADD_C16(vrgout_k0_s3, vmall_s3, 0, r, 3)
	VFADD_C16(vrgout_k1_s3, vmall_s3, 1, r, 3)
	VFADD_C16(vrgout_k2_s3, vmall_s3, 2, r, 3)
	VFADD_C16(vrgout_k3_s3, vmall_s3, 3, r, 3)
	VFADD_C16(vrgout_k4_s3, vmall_s3, 4, r, 3)
	VFADD_C16(vrgout_k5_s3, vmall_s3, 5, r, 3)
	VFADD_C16(vrgout_k6_s3, vmall_s3, 6, r, 3)
	VFADD_C16(vrgout_k7_s3, vmall_s3, 7, r, 3)


	__vr vrgout_ptr_k0_s4 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry), vrx_s4),
					   2,
					   (unsigned long)(pGOut+gOutIndex)) ;
	__vr vrgout_k0_s4 = _ve_vgtu_vvm(vrgout_ptr_k0_s4, vmall_s4) ;
	__vr vrgout_ptr_k1_s4 = _ve_vaddsl_vsv(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_s4) ;
	__vr vrgout_k1_s4 = _ve_vgtu_vvm(vrgout_ptr_k1_s4, vmall_s4) ;
	__vr vrgout_ptr_k2_s4 = _ve_vaddsl_vsv(4*2*gOutHeight*gOutWidth, vrgout_ptr_k0_s4) ;
	__vr vrgout_k2_s4 = _ve_vgtu_vvm(vrgout_ptr_k2_s4, vmall_s4) ;
	__vr vrgout_ptr_k3_s4 = _ve_vaddsl_vsv(4*3*gOutHeight*gOutWidth, vrgout_ptr_k0_s4) ;
	__vr vrgout_k3_s4 = _ve_vgtu_vvm(vrgout_ptr_k3_s4, vmall_s4) ;
	__vr vrgout_ptr_k4_s4 = _ve_vaddsl_vsv(4*4*gOutHeight*gOutWidth, vrgout_ptr_k0_s4) ;
	__vr vrgout_k4_s4 = _ve_vgtu_vvm(vrgout_ptr_k4_s4, vmall_s4) ;
	__vr vrgout_ptr_k5_s4 = _ve_vaddsl_vsv(4*5*gOutHeight*gOutWidth, vrgout_ptr_k0_s4) ;
	__vr vrgout_k5_s4 = _ve_vgtu_vvm(vrgout_ptr_k5_s4, vmall_s4) ;
	__vr vrgout_ptr_k6_s4 = _ve_vaddsl_vsv(4*6*gOutHeight*gOutWidth, vrgout_ptr_k0_s4) ;
	__vr vrgout_k6_s4 = _ve_vgtu_vvm(vrgout_ptr_k6_s4, vmall_s4) ;
	__vr vrgout_ptr_k7_s4 = _ve_vaddsl_vsv(4*7*gOutHeight*gOutWidth, vrgout_ptr_k0_s4) ;
	__vr vrgout_k7_s4 = _ve_vgtu_vvm(vrgout_ptr_k7_s4, vmall_s4) ;

	VFADD_C16(vrgout_k0_s4, vmall_s4, 0, r, 4)
	VFADD_C16(vrgout_k1_s4, vmall_s4, 1, r, 4)
	VFADD_C16(vrgout_k2_s4, vmall_s4, 2, r, 4)
	VFADD_C16(vrgout_k3_s4, vmall_s4, 3, r, 4)
	VFADD_C16(vrgout_k4_s4, vmall_s4, 4, r, 4)
	VFADD_C16(vrgout_k5_s4, vmall_s4, 5, r, 4)
	VFADD_C16(vrgout_k6_s4, vmall_s4, 6, r, 4)
	VFADD_C16(vrgout_k7_s4, vmall_s4, 7, r, 4)

      } // gOutChannel
    } // kernHeight

#undef VFADD_C16

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
  } // gOutPixels
}


vednnError_t
vednnConvolutionBackwardData_direct_ker5_iwU128(
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
	     gInPixels, n, k,
	     nH, vrh, vrw ) ;

	  k+=8 ;
	}
	for (; k<gInChannelGroup; k+=16) {
	  c16(pGOut, pKernel, pGIn,
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
