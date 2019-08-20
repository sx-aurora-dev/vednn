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

    for (int64_t r=0; r<kernHeight; r++) {
      __vr vri = _ve_vaddsl_vsv(padHeight-r*dilationHeight+h, vrh) ;
      __vr vry = _ve_vdivsl_vvs(vri, strideHeight) ;

      __vm256 vmy0 = _ve_vfmkl_mcv(VECC_IEQ, _ve_vcmpsl_vvv(vri, _ve_vmulsl_vsv(strideHeight, vry))) ;
      __vm256 vmy1 = _ve_vfmkl_mcv(VECC_GE, vry) ;
      __vm256 vmy2 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(gOutHeight,vry)) ;

      __vm256 vmy = _ve_andm_mmm(_ve_andm_mmm(vmy0, vmy1), vmy2) ;

      for (int64_t s=0; s<kernWidth; s++) {
	__vr vrj = _ve_vaddsl_vsv(padWidth-s*dilationWidth, vrw) ;
	__vr vrx = _ve_vdivsl_vvs(vrj, strideWidth) ;

	__vm256 vmx0 = _ve_vfmkl_mcv(VECC_IEQ, _ve_vcmpsl_vvv(vrj, _ve_vmulsl_vsv(strideWidth, vrx))) ;
	__vm256 vmx1 = _ve_vfmkl_mcv(VECC_GE, vrx) ;
	__vm256 vmx2 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(gOutWidth,vrx)) ;

	__vm256 vmx = _ve_andm_mmm(_ve_andm_mmm(vmx0, vmx1), vmx2) ;

	__vm256 vmall = _ve_andm_mmm(vmy,vmx) ;

	int64_t k=0;
	if( (gOutChannelGroup & 0x01 ) == 1 ) {
	  int64_t gOutIndex    = gOutGroupOffset + ((n * gOutChannel + k) * gOutHeight) * gOutWidth ;
	  const float *pKerValue_k0 = pKernel + kernGroupOffset + (((k  ) * gInChannelGroup + c) * kernHeight + r) * kernWidth + s;

	  __vr vrgout_ptr_k0 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry), vrx),
					     2,
					     (unsigned long)(pGOut+gOutIndex)) ;
	  __vr vrgout_k0 = _ve_vgtu_vvm(vrgout_ptr_k0, vmall) ;

#define VFADD_C1(KTOKEN)												\
	  const uint64_t kerValue01_##KTOKEN = _ve_pack_f32p(pKerValue_##KTOKEN,					\
						             pKerValue_##KTOKEN + 1 * kernHeight * kernWidth ) ;	\
	  vrgout_##KTOKEN = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), vrgout_##KTOKEN, vmall) ;				\
	  vrsum = _ve_vfmads_vvsv(vrsum, pKerValue_##KTOKEN[0], vrgout_##KTOKEN) ;

	  VFADD_C1(k0)

	  k+=1 ;
	}
	if( ((gOutChannelGroup >> 1) & 0x01 ) == 1 ) {
	  int64_t gOutIndex    = gOutGroupOffset + ((n * gOutChannel + k) * gOutHeight) * gOutWidth ;
	  const float *pKerValue_k0 = pKernel + kernGroupOffset + (((k  ) * gInChannelGroup + c) * kernHeight + r) * kernWidth + s;
	  const float *pKerValue_k1 = pKernel + kernGroupOffset + (((k+1) * gInChannelGroup + c) * kernHeight + r) * kernWidth + s;

	  __vr vrgout_ptr_k0 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry), vrx),
					     2,
					     (unsigned long)(pGOut+gOutIndex)) ;
	  __vr vrgout_k0 = _ve_vgtu_vvm(vrgout_ptr_k0, vmall) ;
	  __vr vrgout_ptr_k1 = _ve_vaddsl_vsv(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0) ;
	  __vr vrgout_k1 = _ve_vgtu_vvm(vrgout_ptr_k1, vmall) ;


	  VFADD_C1(k0)
	  VFADD_C1(k1)

	  k+=2 ;
	}
	if( ((gOutChannelGroup >> 2) & 0x01 ) == 1 ) {
	  int64_t gOutIndex    = gOutGroupOffset + ((n * gOutChannel + k) * gOutHeight) * gOutWidth ;
	  const float *pKerValue_k0 = pKernel + kernGroupOffset + (((k  ) * gInChannelGroup + c) * kernHeight + r) * kernWidth + s;
	  const float *pKerValue_k1 = pKernel + kernGroupOffset + (((k+1) * gInChannelGroup + c) * kernHeight + r) * kernWidth + s;
	  const float *pKerValue_k2 = pKernel + kernGroupOffset + (((k+2) * gInChannelGroup + c) * kernHeight + r) * kernWidth + s;
	  const float *pKerValue_k3 = pKernel + kernGroupOffset + (((k+3) * gInChannelGroup + c) * kernHeight + r) * kernWidth + s;

	  __vr vrgout_ptr_k0 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry), vrx),
					     2,
					     (unsigned long)(pGOut+gOutIndex)) ;
	  __vr vrgout_k0 = _ve_vgtu_vvm(vrgout_ptr_k0, vmall) ;
	  __vr vrgout_ptr_k1 = _ve_vaddsl_vsv(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0) ;
	  __vr vrgout_k1 = _ve_vgtu_vvm(vrgout_ptr_k1, vmall) ;
	  __vr vrgout_ptr_k2 = _ve_vaddsl_vsv(4*2*gOutHeight*gOutWidth, vrgout_ptr_k0) ;
	  __vr vrgout_k2 = _ve_vgtu_vvm(vrgout_ptr_k2, vmall) ;
	  __vr vrgout_ptr_k3 = _ve_vaddsl_vsv(4*3*gOutHeight*gOutWidth, vrgout_ptr_k0) ;
	  __vr vrgout_k3 = _ve_vgtu_vvm(vrgout_ptr_k3, vmall) ;

	  VFADD_C1(k0)
	  VFADD_C1(k1)
	  VFADD_C1(k2)
	  VFADD_C1(k3)

	  k+=4 ;
	}
	for (; k<gOutChannelGroup; k+=8) {
	  int64_t gOutIndex    = gOutGroupOffset + ((n * gOutChannel + k) * gOutHeight) * gOutWidth ;
	  const float *pKerValue_k0 = pKernel + kernGroupOffset + (((k  ) * gInChannelGroup + c) * kernHeight + r) * kernWidth + s;
	  const float *pKerValue_k1 = pKernel + kernGroupOffset + (((k+1) * gInChannelGroup + c) * kernHeight + r) * kernWidth + s;
	  const float *pKerValue_k2 = pKernel + kernGroupOffset + (((k+2) * gInChannelGroup + c) * kernHeight + r) * kernWidth + s;
	  const float *pKerValue_k3 = pKernel + kernGroupOffset + (((k+3) * gInChannelGroup + c) * kernHeight + r) * kernWidth + s;
	  const float *pKerValue_k4 = pKernel + kernGroupOffset + (((k+4) * gInChannelGroup + c) * kernHeight + r) * kernWidth + s;
	  const float *pKerValue_k5 = pKernel + kernGroupOffset + (((k+5) * gInChannelGroup + c) * kernHeight + r) * kernWidth + s;
	  const float *pKerValue_k6 = pKernel + kernGroupOffset + (((k+6) * gInChannelGroup + c) * kernHeight + r) * kernWidth + s;
	  const float *pKerValue_k7 = pKernel + kernGroupOffset + (((k+7) * gInChannelGroup + c) * kernHeight + r) * kernWidth + s;

	  __vr vrgout_ptr_k0 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry), vrx),
					     2,
					     (unsigned long)(pGOut+gOutIndex)) ;
	  __vr vrgout_k0 = _ve_vgtu_vvm(vrgout_ptr_k0, vmall) ;
	  __vr vrgout_ptr_k1 = _ve_vaddsl_vsv(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0) ;
	  __vr vrgout_k1 = _ve_vgtu_vvm(vrgout_ptr_k1, vmall) ;
	  __vr vrgout_ptr_k2 = _ve_vaddsl_vsv(4*2*gOutHeight*gOutWidth, vrgout_ptr_k0) ;
	  __vr vrgout_k2 = _ve_vgtu_vvm(vrgout_ptr_k2, vmall) ;
	  __vr vrgout_ptr_k3 = _ve_vaddsl_vsv(4*3*gOutHeight*gOutWidth, vrgout_ptr_k0) ;
	  __vr vrgout_k3 = _ve_vgtu_vvm(vrgout_ptr_k3, vmall) ;
	  __vr vrgout_ptr_k4 = _ve_vaddsl_vsv(4*4*gOutHeight*gOutWidth, vrgout_ptr_k0) ;
	  __vr vrgout_k4 = _ve_vgtu_vvm(vrgout_ptr_k4, vmall) ;
	  __vr vrgout_ptr_k5 = _ve_vaddsl_vsv(4*5*gOutHeight*gOutWidth, vrgout_ptr_k0) ;
	  __vr vrgout_k5 = _ve_vgtu_vvm(vrgout_ptr_k5, vmall) ;
	  __vr vrgout_ptr_k6 = _ve_vaddsl_vsv(4*6*gOutHeight*gOutWidth, vrgout_ptr_k0) ;
	  __vr vrgout_k6 = _ve_vgtu_vvm(vrgout_ptr_k6, vmall) ;
	  __vr vrgout_ptr_k7 = _ve_vaddsl_vsv(4*7*gOutHeight*gOutWidth, vrgout_ptr_k0) ;
	  __vr vrgout_k7 = _ve_vgtu_vvm(vrgout_ptr_k7, vmall) ;

	  VFADD_C1(k0)
	  VFADD_C1(k1)
	  VFADD_C1(k2)
	  VFADD_C1(k3)
	  VFADD_C1(k4)
	  VFADD_C1(k5)
	  VFADD_C1(k6)
	  VFADD_C1(k7)
#undef VFADD_C1

	} // gOutChannel
      } // kernWidth
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

    for (int64_t r=0; r<kernHeight; r++) {
      __vr vri = _ve_vaddsl_vsv(padHeight-r*dilationHeight+h, vrh) ;
      __vr vry = _ve_vdivsl_vvs(vri, strideHeight) ;

      __vm256 vmy0 = _ve_vfmkl_mcv(VECC_IEQ, _ve_vcmpsl_vvv(vri, _ve_vmulsl_vsv(strideHeight, vry))) ;
      __vm256 vmy1 = _ve_vfmkl_mcv(VECC_GE, vry) ;
      __vm256 vmy2 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(gOutHeight,vry)) ;

      __vm256 vmy = _ve_andm_mmm(_ve_andm_mmm(vmy0, vmy1), vmy2) ;

      for (int64_t s=0; s<kernWidth; s++) {
	__vr vrj = _ve_vaddsl_vsv(padWidth-s*dilationWidth, vrw) ;
	__vr vrx = _ve_vdivsl_vvs(vrj, strideWidth) ;

	__vm256 vmx0 = _ve_vfmkl_mcv(VECC_IEQ, _ve_vcmpsl_vvv(vrj, _ve_vmulsl_vsv(strideWidth, vrx))) ;
	__vm256 vmx1 = _ve_vfmkl_mcv(VECC_GE, vrx) ;
	__vm256 vmx2 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(gOutWidth,vrx)) ;

	__vm256 vmx = _ve_andm_mmm(_ve_andm_mmm(vmx0, vmx1), vmx2) ;

	__vm256 vmall = _ve_andm_mmm(vmy,vmx) ;

	int64_t k=0;
	if( (gOutChannelGroup & 0x01 ) == 1 ) {
	  int64_t gOutIndex    = gOutGroupOffset + ((n * gOutChannel + k) * gOutHeight) * gOutWidth ;
	  const float *pKerValue_k0 = pKernel + kernGroupOffset + (((k  ) * gInChannelGroup + c) * kernHeight + r) * kernWidth + s;

	  __vr vrgout_ptr_k0 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry), vrx),
					     2,
					     (unsigned long)(pGOut+gOutIndex)) ;
	  __vr vrgout_k0 = _ve_vgtu_vvm(vrgout_ptr_k0, vmall) ;

#define VFADD_C2(KTOKEN)												\
	  const uint64_t kerValue01_##KTOKEN = _ve_pack_f32p(pKerValue_##KTOKEN,					\
						             pKerValue_##KTOKEN + 1 * kernHeight * kernWidth ) ;	\
	  vrgout_##KTOKEN = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), vrgout_##KTOKEN, vmall) ;			\
	  __vr vrgoutP_##KTOKEN = _ve_vshf_vvvs(vrgout_##KTOKEN, vrgout_##KTOKEN, VE_VSHUFFLE_YUZU) ;		\
	  vrsum01 = _ve_pvfmad_vvsv(vrsum01, kerValue01_##KTOKEN, vrgoutP_##KTOKEN) ;

	  VFADD_C2(k0)

	  k+=1 ;
	}
	if( ((gOutChannelGroup >> 1) & 0x01 ) == 1 ) {
	  int64_t gOutIndex    = gOutGroupOffset + ((n * gOutChannel + k) * gOutHeight) * gOutWidth ;
	  const float *pKerValue_k0 = pKernel + kernGroupOffset + (((k  ) * gInChannelGroup + c) * kernHeight + r) * kernWidth + s;
	  const float *pKerValue_k1 = pKernel + kernGroupOffset + (((k+1) * gInChannelGroup + c) * kernHeight + r) * kernWidth + s;

	  __vr vrgout_ptr_k0 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry), vrx),
					     2,
					     (unsigned long)(pGOut+gOutIndex)) ;
	  __vr vrgout_k0 = _ve_vgtu_vvm(vrgout_ptr_k0, vmall) ;
	  __vr vrgout_ptr_k1 = _ve_vaddsl_vsv(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0) ;
	  __vr vrgout_k1 = _ve_vgtu_vvm(vrgout_ptr_k1, vmall) ;


	  VFADD_C2(k0)
	  VFADD_C2(k1)

	  k+=2 ;
	}
	if( ((gOutChannelGroup >> 2) & 0x01 ) == 1 ) {
	  int64_t gOutIndex    = gOutGroupOffset + ((n * gOutChannel + k) * gOutHeight) * gOutWidth ;
	  const float *pKerValue_k0 = pKernel + kernGroupOffset + (((k  ) * gInChannelGroup + c) * kernHeight + r) * kernWidth + s;
	  const float *pKerValue_k1 = pKernel + kernGroupOffset + (((k+1) * gInChannelGroup + c) * kernHeight + r) * kernWidth + s;
	  const float *pKerValue_k2 = pKernel + kernGroupOffset + (((k+2) * gInChannelGroup + c) * kernHeight + r) * kernWidth + s;
	  const float *pKerValue_k3 = pKernel + kernGroupOffset + (((k+3) * gInChannelGroup + c) * kernHeight + r) * kernWidth + s;

	  __vr vrgout_ptr_k0 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry), vrx),
					     2,
					     (unsigned long)(pGOut+gOutIndex)) ;
	  __vr vrgout_k0 = _ve_vgtu_vvm(vrgout_ptr_k0, vmall) ;
	  __vr vrgout_ptr_k1 = _ve_vaddsl_vsv(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0) ;
	  __vr vrgout_k1 = _ve_vgtu_vvm(vrgout_ptr_k1, vmall) ;
	  __vr vrgout_ptr_k2 = _ve_vaddsl_vsv(4*2*gOutHeight*gOutWidth, vrgout_ptr_k0) ;
	  __vr vrgout_k2 = _ve_vgtu_vvm(vrgout_ptr_k2, vmall) ;
	  __vr vrgout_ptr_k3 = _ve_vaddsl_vsv(4*3*gOutHeight*gOutWidth, vrgout_ptr_k0) ;
	  __vr vrgout_k3 = _ve_vgtu_vvm(vrgout_ptr_k3, vmall) ;

	  VFADD_C2(k0)
	  VFADD_C2(k1)
	  VFADD_C2(k2)
	  VFADD_C2(k3)

	  k+=4 ;
	}
	for (; k<gOutChannelGroup; k+=8) {
	  int64_t gOutIndex    = gOutGroupOffset + ((n * gOutChannel + k) * gOutHeight) * gOutWidth ;
	  const float *pKerValue_k0 = pKernel + kernGroupOffset + (((k  ) * gInChannelGroup + c) * kernHeight + r) * kernWidth + s;
	  const float *pKerValue_k1 = pKernel + kernGroupOffset + (((k+1) * gInChannelGroup + c) * kernHeight + r) * kernWidth + s;
	  const float *pKerValue_k2 = pKernel + kernGroupOffset + (((k+2) * gInChannelGroup + c) * kernHeight + r) * kernWidth + s;
	  const float *pKerValue_k3 = pKernel + kernGroupOffset + (((k+3) * gInChannelGroup + c) * kernHeight + r) * kernWidth + s;
	  const float *pKerValue_k4 = pKernel + kernGroupOffset + (((k+4) * gInChannelGroup + c) * kernHeight + r) * kernWidth + s;
	  const float *pKerValue_k5 = pKernel + kernGroupOffset + (((k+5) * gInChannelGroup + c) * kernHeight + r) * kernWidth + s;
	  const float *pKerValue_k6 = pKernel + kernGroupOffset + (((k+6) * gInChannelGroup + c) * kernHeight + r) * kernWidth + s;
	  const float *pKerValue_k7 = pKernel + kernGroupOffset + (((k+7) * gInChannelGroup + c) * kernHeight + r) * kernWidth + s;

	  __vr vrgout_ptr_k0 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry), vrx),
					     2,
					     (unsigned long)(pGOut+gOutIndex)) ;
	  __vr vrgout_k0 = _ve_vgtu_vvm(vrgout_ptr_k0, vmall) ;
	  __vr vrgout_ptr_k1 = _ve_vaddsl_vsv(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0) ;
	  __vr vrgout_k1 = _ve_vgtu_vvm(vrgout_ptr_k1, vmall) ;
	  __vr vrgout_ptr_k2 = _ve_vaddsl_vsv(4*2*gOutHeight*gOutWidth, vrgout_ptr_k0) ;
	  __vr vrgout_k2 = _ve_vgtu_vvm(vrgout_ptr_k2, vmall) ;
	  __vr vrgout_ptr_k3 = _ve_vaddsl_vsv(4*3*gOutHeight*gOutWidth, vrgout_ptr_k0) ;
	  __vr vrgout_k3 = _ve_vgtu_vvm(vrgout_ptr_k3, vmall) ;
	  __vr vrgout_ptr_k4 = _ve_vaddsl_vsv(4*4*gOutHeight*gOutWidth, vrgout_ptr_k0) ;
	  __vr vrgout_k4 = _ve_vgtu_vvm(vrgout_ptr_k4, vmall) ;
	  __vr vrgout_ptr_k5 = _ve_vaddsl_vsv(4*5*gOutHeight*gOutWidth, vrgout_ptr_k0) ;
	  __vr vrgout_k5 = _ve_vgtu_vvm(vrgout_ptr_k5, vmall) ;
	  __vr vrgout_ptr_k6 = _ve_vaddsl_vsv(4*6*gOutHeight*gOutWidth, vrgout_ptr_k0) ;
	  __vr vrgout_k6 = _ve_vgtu_vvm(vrgout_ptr_k6, vmall) ;
	  __vr vrgout_ptr_k7 = _ve_vaddsl_vsv(4*7*gOutHeight*gOutWidth, vrgout_ptr_k0) ;
	  __vr vrgout_k7 = _ve_vgtu_vvm(vrgout_ptr_k7, vmall) ;

	  VFADD_C2(k0)
	  VFADD_C2(k1)
	  VFADD_C2(k2)
	  VFADD_C2(k3)
	  VFADD_C2(k4)
	  VFADD_C2(k5)
	  VFADD_C2(k6)
	  VFADD_C2(k7)
#undef VFADD_C2

	} // gOutChannel
      } // kernWidth
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

    for (int64_t r=0; r<kernHeight; r++) {
      __vr vri = _ve_vaddsl_vsv(padHeight-r*dilationHeight+h, vrh) ;
      __vr vry = _ve_vdivsl_vvs(vri, strideHeight) ;

      __vm256 vmy0 = _ve_vfmkl_mcv(VECC_IEQ, _ve_vcmpsl_vvv(vri, _ve_vmulsl_vsv(strideHeight, vry))) ;
      __vm256 vmy1 = _ve_vfmkl_mcv(VECC_GE, vry) ;
      __vm256 vmy2 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(gOutHeight,vry)) ;

      __vm256 vmy = _ve_andm_mmm(_ve_andm_mmm(vmy0, vmy1), vmy2) ;

      for (int64_t s=0; s<kernWidth; s++) {
	__vr vrj = _ve_vaddsl_vsv(padWidth-s*dilationWidth, vrw) ;
	__vr vrx = _ve_vdivsl_vvs(vrj, strideWidth) ;

	__vm256 vmx0 = _ve_vfmkl_mcv(VECC_IEQ, _ve_vcmpsl_vvv(vrj, _ve_vmulsl_vsv(strideWidth, vrx))) ;
	__vm256 vmx1 = _ve_vfmkl_mcv(VECC_GE, vrx) ;
	__vm256 vmx2 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(gOutWidth,vrx)) ;

	__vm256 vmx = _ve_andm_mmm(_ve_andm_mmm(vmx0, vmx1), vmx2) ;

	__vm256 vmall = _ve_andm_mmm(vmy,vmx) ;

	int64_t k=0;
	if( (gOutChannelGroup & 0x01 ) == 1 ) {
	  int64_t gOutIndex    = gOutGroupOffset + ((n * gOutChannel + k) * gOutHeight) * gOutWidth ;
	  const float *pKerValue_k0 = pKernel + kernGroupOffset + (((k  ) * gInChannelGroup + c) * kernHeight + r) * kernWidth + s;

	  __vr vrgout_ptr_k0 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry), vrx),
					     2,
					     (unsigned long)(pGOut+gOutIndex)) ;
	  __vr vrgout_k0 = _ve_vgtu_vvm(vrgout_ptr_k0, vmall) ;

#define VFADD_C4(KTOKEN)												\
	  const uint64_t kerValue01_##KTOKEN = _ve_pack_f32p(pKerValue_##KTOKEN,					\
						             pKerValue_##KTOKEN + 1 * kernHeight * kernWidth ) ;	\
	  const uint64_t kerValue23_##KTOKEN = _ve_pack_f32p(pKerValue_##KTOKEN + 2 * kernHeight * kernWidth,		\
						             pKerValue_##KTOKEN + 3 * kernHeight * kernWidth ) ;	\
	  vrgout_##KTOKEN = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), vrgout_##KTOKEN, vmall) ;			\
	  __vr vrgoutP_##KTOKEN = _ve_vshf_vvvs(vrgout_##KTOKEN, vrgout_##KTOKEN, VE_VSHUFFLE_YUZU) ;		\
	  vrsum01 = _ve_pvfmad_vvsv(vrsum01, kerValue01_##KTOKEN, vrgoutP_##KTOKEN) ;				\
	  vrsum23 = _ve_pvfmad_vvsv(vrsum23, kerValue23_##KTOKEN, vrgoutP_##KTOKEN) ;

	  VFADD_C4(k0)

	  k+=1 ;
	}
	if( ((gOutChannelGroup >> 1) & 0x01 ) == 1 ) {
	  int64_t gOutIndex    = gOutGroupOffset + ((n * gOutChannel + k) * gOutHeight) * gOutWidth ;
	  const float *pKerValue_k0 = pKernel + kernGroupOffset + (((k  ) * gInChannelGroup + c) * kernHeight + r) * kernWidth + s;
	  const float *pKerValue_k1 = pKernel + kernGroupOffset + (((k+1) * gInChannelGroup + c) * kernHeight + r) * kernWidth + s;

	  __vr vrgout_ptr_k0 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry), vrx),
					     2,
					     (unsigned long)(pGOut+gOutIndex)) ;
	  __vr vrgout_k0 = _ve_vgtu_vvm(vrgout_ptr_k0, vmall) ;
	  __vr vrgout_ptr_k1 = _ve_vaddsl_vsv(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0) ;
	  __vr vrgout_k1 = _ve_vgtu_vvm(vrgout_ptr_k1, vmall) ;


	  VFADD_C4(k0)
	  VFADD_C4(k1)

	  k+=2 ;
	}
	if( ((gOutChannelGroup >> 2) & 0x01 ) == 1 ) {
	  int64_t gOutIndex    = gOutGroupOffset + ((n * gOutChannel + k) * gOutHeight) * gOutWidth ;
	  const float *pKerValue_k0 = pKernel + kernGroupOffset + (((k  ) * gInChannelGroup + c) * kernHeight + r) * kernWidth + s;
	  const float *pKerValue_k1 = pKernel + kernGroupOffset + (((k+1) * gInChannelGroup + c) * kernHeight + r) * kernWidth + s;
	  const float *pKerValue_k2 = pKernel + kernGroupOffset + (((k+2) * gInChannelGroup + c) * kernHeight + r) * kernWidth + s;
	  const float *pKerValue_k3 = pKernel + kernGroupOffset + (((k+3) * gInChannelGroup + c) * kernHeight + r) * kernWidth + s;

	  __vr vrgout_ptr_k0 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry), vrx),
					     2,
					     (unsigned long)(pGOut+gOutIndex)) ;
	  __vr vrgout_k0 = _ve_vgtu_vvm(vrgout_ptr_k0, vmall) ;
	  __vr vrgout_ptr_k1 = _ve_vaddsl_vsv(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0) ;
	  __vr vrgout_k1 = _ve_vgtu_vvm(vrgout_ptr_k1, vmall) ;
	  __vr vrgout_ptr_k2 = _ve_vaddsl_vsv(4*2*gOutHeight*gOutWidth, vrgout_ptr_k0) ;
	  __vr vrgout_k2 = _ve_vgtu_vvm(vrgout_ptr_k2, vmall) ;
	  __vr vrgout_ptr_k3 = _ve_vaddsl_vsv(4*3*gOutHeight*gOutWidth, vrgout_ptr_k0) ;
	  __vr vrgout_k3 = _ve_vgtu_vvm(vrgout_ptr_k3, vmall) ;

	  VFADD_C4(k0)
	  VFADD_C4(k1)
	  VFADD_C4(k2)
	  VFADD_C4(k3)

	  k+=4 ;
	}
	for (; k<gOutChannelGroup; k+=8) {
	  int64_t gOutIndex    = gOutGroupOffset + ((n * gOutChannel + k) * gOutHeight) * gOutWidth ;
	  const float *pKerValue_k0 = pKernel + kernGroupOffset + (((k  ) * gInChannelGroup + c) * kernHeight + r) * kernWidth + s;
	  const float *pKerValue_k1 = pKernel + kernGroupOffset + (((k+1) * gInChannelGroup + c) * kernHeight + r) * kernWidth + s;
	  const float *pKerValue_k2 = pKernel + kernGroupOffset + (((k+2) * gInChannelGroup + c) * kernHeight + r) * kernWidth + s;
	  const float *pKerValue_k3 = pKernel + kernGroupOffset + (((k+3) * gInChannelGroup + c) * kernHeight + r) * kernWidth + s;
	  const float *pKerValue_k4 = pKernel + kernGroupOffset + (((k+4) * gInChannelGroup + c) * kernHeight + r) * kernWidth + s;
	  const float *pKerValue_k5 = pKernel + kernGroupOffset + (((k+5) * gInChannelGroup + c) * kernHeight + r) * kernWidth + s;
	  const float *pKerValue_k6 = pKernel + kernGroupOffset + (((k+6) * gInChannelGroup + c) * kernHeight + r) * kernWidth + s;
	  const float *pKerValue_k7 = pKernel + kernGroupOffset + (((k+7) * gInChannelGroup + c) * kernHeight + r) * kernWidth + s;

	  __vr vrgout_ptr_k0 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry), vrx),
					     2,
					     (unsigned long)(pGOut+gOutIndex)) ;
	  __vr vrgout_k0 = _ve_vgtu_vvm(vrgout_ptr_k0, vmall) ;
	  __vr vrgout_ptr_k1 = _ve_vaddsl_vsv(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0) ;
	  __vr vrgout_k1 = _ve_vgtu_vvm(vrgout_ptr_k1, vmall) ;
	  __vr vrgout_ptr_k2 = _ve_vaddsl_vsv(4*2*gOutHeight*gOutWidth, vrgout_ptr_k0) ;
	  __vr vrgout_k2 = _ve_vgtu_vvm(vrgout_ptr_k2, vmall) ;
	  __vr vrgout_ptr_k3 = _ve_vaddsl_vsv(4*3*gOutHeight*gOutWidth, vrgout_ptr_k0) ;
	  __vr vrgout_k3 = _ve_vgtu_vvm(vrgout_ptr_k3, vmall) ;
	  __vr vrgout_ptr_k4 = _ve_vaddsl_vsv(4*4*gOutHeight*gOutWidth, vrgout_ptr_k0) ;
	  __vr vrgout_k4 = _ve_vgtu_vvm(vrgout_ptr_k4, vmall) ;
	  __vr vrgout_ptr_k5 = _ve_vaddsl_vsv(4*5*gOutHeight*gOutWidth, vrgout_ptr_k0) ;
	  __vr vrgout_k5 = _ve_vgtu_vvm(vrgout_ptr_k5, vmall) ;
	  __vr vrgout_ptr_k6 = _ve_vaddsl_vsv(4*6*gOutHeight*gOutWidth, vrgout_ptr_k0) ;
	  __vr vrgout_k6 = _ve_vgtu_vvm(vrgout_ptr_k6, vmall) ;
	  __vr vrgout_ptr_k7 = _ve_vaddsl_vsv(4*7*gOutHeight*gOutWidth, vrgout_ptr_k0) ;
	  __vr vrgout_k7 = _ve_vgtu_vvm(vrgout_ptr_k7, vmall) ;

	  VFADD_C4(k0)
	  VFADD_C4(k1)
	  VFADD_C4(k2)
	  VFADD_C4(k3)
	  VFADD_C4(k4)
	  VFADD_C4(k5)
	  VFADD_C4(k6)
	  VFADD_C4(k7)
#undef VFADD_C4

	} // gOutChannel
      } // kernWidth
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

    for (int64_t r=0; r<kernHeight; r++) {
      __vr vri = _ve_vaddsl_vsv(padHeight-r*dilationHeight+h, vrh) ;
      __vr vry = _ve_vdivsl_vvs(vri, strideHeight) ;

      __vm256 vmy0 = _ve_vfmkl_mcv(VECC_IEQ, _ve_vcmpsl_vvv(vri, _ve_vmulsl_vsv(strideHeight, vry))) ;
      __vm256 vmy1 = _ve_vfmkl_mcv(VECC_GE, vry) ;
      __vm256 vmy2 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(gOutHeight,vry)) ;

      __vm256 vmy = _ve_andm_mmm(_ve_andm_mmm(vmy0, vmy1), vmy2) ;

      for (int64_t s=0; s<kernWidth; s++) {
	__vr vrj = _ve_vaddsl_vsv(padWidth-s*dilationWidth, vrw) ;
	__vr vrx = _ve_vdivsl_vvs(vrj, strideWidth) ;

	__vm256 vmx0 = _ve_vfmkl_mcv(VECC_IEQ, _ve_vcmpsl_vvv(vrj, _ve_vmulsl_vsv(strideWidth, vrx))) ;
	__vm256 vmx1 = _ve_vfmkl_mcv(VECC_GE, vrx) ;
	__vm256 vmx2 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(gOutWidth,vrx)) ;

	__vm256 vmx = _ve_andm_mmm(_ve_andm_mmm(vmx0, vmx1), vmx2) ;

	__vm256 vmall = _ve_andm_mmm(vmy,vmx) ;

	int64_t k=0;
	if( (gOutChannelGroup & 0x01 ) == 1 ) {
	  int64_t gOutIndex    = gOutGroupOffset + ((n * gOutChannel + k) * gOutHeight) * gOutWidth ;
	  const float *pKerValue_k0 = pKernel + kernGroupOffset + (((k  ) * gInChannelGroup + c) * kernHeight + r) * kernWidth + s;

	  __vr vrgout_ptr_k0 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry), vrx),
					     2,
					     (unsigned long)(pGOut+gOutIndex)) ;
	  __vr vrgout_k0 = _ve_vgtu_vvm(vrgout_ptr_k0, vmall) ;

#define VFADD_C8(KTOKEN)												\
	  const uint64_t kerValue01_##KTOKEN = _ve_pack_f32p(pKerValue_##KTOKEN,					\
						             pKerValue_##KTOKEN + 1 * kernHeight * kernWidth ) ;	\
	  const uint64_t kerValue23_##KTOKEN = _ve_pack_f32p(pKerValue_##KTOKEN + 2 * kernHeight * kernWidth,		\
						             pKerValue_##KTOKEN + 3 * kernHeight * kernWidth ) ;	\
	  const uint64_t kerValue45_##KTOKEN = _ve_pack_f32p(pKerValue_##KTOKEN + 4 * kernHeight * kernWidth,		\
						             pKerValue_##KTOKEN + 5 * kernHeight * kernWidth ) ;	\
	  const uint64_t kerValue67_##KTOKEN = _ve_pack_f32p(pKerValue_##KTOKEN + 6 * kernHeight * kernWidth,		\
						             pKerValue_##KTOKEN + 7 * kernHeight * kernWidth ) ;	\
	  vrgout_##KTOKEN = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), vrgout_##KTOKEN, vmall) ;			\
	  __vr vrgoutP_##KTOKEN = _ve_vshf_vvvs(vrgout_##KTOKEN, vrgout_##KTOKEN, VE_VSHUFFLE_YUZU) ;		\
	  vrsum01 = _ve_pvfmad_vvsv(vrsum01, kerValue01_##KTOKEN, vrgoutP_##KTOKEN) ;				\
	  vrsum23 = _ve_pvfmad_vvsv(vrsum23, kerValue23_##KTOKEN, vrgoutP_##KTOKEN) ;				\
	  vrsum45 = _ve_pvfmad_vvsv(vrsum45, kerValue45_##KTOKEN, vrgoutP_##KTOKEN) ;				\
	  vrsum67 = _ve_pvfmad_vvsv(vrsum67, kerValue67_##KTOKEN, vrgoutP_##KTOKEN) ;

	  VFADD_C8(k0)

	  k+=1 ;
	}
	if( ((gOutChannelGroup >> 1) & 0x01 ) == 1 ) {
	  int64_t gOutIndex    = gOutGroupOffset + ((n * gOutChannel + k) * gOutHeight) * gOutWidth ;
	  const float *pKerValue_k0 = pKernel + kernGroupOffset + (((k  ) * gInChannelGroup + c) * kernHeight + r) * kernWidth + s;
	  const float *pKerValue_k1 = pKernel + kernGroupOffset + (((k+1) * gInChannelGroup + c) * kernHeight + r) * kernWidth + s;

	  __vr vrgout_ptr_k0 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry), vrx),
					     2,
					     (unsigned long)(pGOut+gOutIndex)) ;
	  __vr vrgout_k0 = _ve_vgtu_vvm(vrgout_ptr_k0, vmall) ;
	  __vr vrgout_ptr_k1 = _ve_vaddsl_vsv(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0) ;
	  __vr vrgout_k1 = _ve_vgtu_vvm(vrgout_ptr_k1, vmall) ;


	  VFADD_C8(k0)
	  VFADD_C8(k1)

	  k+=2 ;
	}
	if( ((gOutChannelGroup >> 2) & 0x01 ) == 1 ) {
	  int64_t gOutIndex    = gOutGroupOffset + ((n * gOutChannel + k) * gOutHeight) * gOutWidth ;
	  const float *pKerValue_k0 = pKernel + kernGroupOffset + (((k  ) * gInChannelGroup + c) * kernHeight + r) * kernWidth + s;
	  const float *pKerValue_k1 = pKernel + kernGroupOffset + (((k+1) * gInChannelGroup + c) * kernHeight + r) * kernWidth + s;
	  const float *pKerValue_k2 = pKernel + kernGroupOffset + (((k+2) * gInChannelGroup + c) * kernHeight + r) * kernWidth + s;
	  const float *pKerValue_k3 = pKernel + kernGroupOffset + (((k+3) * gInChannelGroup + c) * kernHeight + r) * kernWidth + s;

	  __vr vrgout_ptr_k0 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry), vrx),
					     2,
					     (unsigned long)(pGOut+gOutIndex)) ;
	  __vr vrgout_k0 = _ve_vgtu_vvm(vrgout_ptr_k0, vmall) ;
	  __vr vrgout_ptr_k1 = _ve_vaddsl_vsv(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0) ;
	  __vr vrgout_k1 = _ve_vgtu_vvm(vrgout_ptr_k1, vmall) ;
	  __vr vrgout_ptr_k2 = _ve_vaddsl_vsv(4*2*gOutHeight*gOutWidth, vrgout_ptr_k0) ;
	  __vr vrgout_k2 = _ve_vgtu_vvm(vrgout_ptr_k2, vmall) ;
	  __vr vrgout_ptr_k3 = _ve_vaddsl_vsv(4*3*gOutHeight*gOutWidth, vrgout_ptr_k0) ;
	  __vr vrgout_k3 = _ve_vgtu_vvm(vrgout_ptr_k3, vmall) ;

	  VFADD_C8(k0)
	  VFADD_C8(k1)
	  VFADD_C8(k2)
	  VFADD_C8(k3)

	  k+=4 ;
	}
	for (; k<gOutChannelGroup; k+=8) {
	  int64_t gOutIndex    = gOutGroupOffset + ((n * gOutChannel + k) * gOutHeight) * gOutWidth ;
	  const float *pKerValue_k0 = pKernel + kernGroupOffset + (((k  ) * gInChannelGroup + c) * kernHeight + r) * kernWidth + s;
	  const float *pKerValue_k1 = pKernel + kernGroupOffset + (((k+1) * gInChannelGroup + c) * kernHeight + r) * kernWidth + s;
	  const float *pKerValue_k2 = pKernel + kernGroupOffset + (((k+2) * gInChannelGroup + c) * kernHeight + r) * kernWidth + s;
	  const float *pKerValue_k3 = pKernel + kernGroupOffset + (((k+3) * gInChannelGroup + c) * kernHeight + r) * kernWidth + s;
	  const float *pKerValue_k4 = pKernel + kernGroupOffset + (((k+4) * gInChannelGroup + c) * kernHeight + r) * kernWidth + s;
	  const float *pKerValue_k5 = pKernel + kernGroupOffset + (((k+5) * gInChannelGroup + c) * kernHeight + r) * kernWidth + s;
	  const float *pKerValue_k6 = pKernel + kernGroupOffset + (((k+6) * gInChannelGroup + c) * kernHeight + r) * kernWidth + s;
	  const float *pKerValue_k7 = pKernel + kernGroupOffset + (((k+7) * gInChannelGroup + c) * kernHeight + r) * kernWidth + s;

	  __vr vrgout_ptr_k0 = _ve_vsfa_vvss(_ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vry), vrx),
					     2,
					     (unsigned long)(pGOut+gOutIndex)) ;
	  __vr vrgout_k0 = _ve_vgtu_vvm(vrgout_ptr_k0, vmall) ;
	  __vr vrgout_ptr_k1 = _ve_vaddsl_vsv(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0) ;
	  __vr vrgout_k1 = _ve_vgtu_vvm(vrgout_ptr_k1, vmall) ;
	  __vr vrgout_ptr_k2 = _ve_vaddsl_vsv(4*2*gOutHeight*gOutWidth, vrgout_ptr_k0) ;
	  __vr vrgout_k2 = _ve_vgtu_vvm(vrgout_ptr_k2, vmall) ;
	  __vr vrgout_ptr_k3 = _ve_vaddsl_vsv(4*3*gOutHeight*gOutWidth, vrgout_ptr_k0) ;
	  __vr vrgout_k3 = _ve_vgtu_vvm(vrgout_ptr_k3, vmall) ;
	  __vr vrgout_ptr_k4 = _ve_vaddsl_vsv(4*4*gOutHeight*gOutWidth, vrgout_ptr_k0) ;
	  __vr vrgout_k4 = _ve_vgtu_vvm(vrgout_ptr_k4, vmall) ;
	  __vr vrgout_ptr_k5 = _ve_vaddsl_vsv(4*5*gOutHeight*gOutWidth, vrgout_ptr_k0) ;
	  __vr vrgout_k5 = _ve_vgtu_vvm(vrgout_ptr_k5, vmall) ;
	  __vr vrgout_ptr_k6 = _ve_vaddsl_vsv(4*6*gOutHeight*gOutWidth, vrgout_ptr_k0) ;
	  __vr vrgout_k6 = _ve_vgtu_vvm(vrgout_ptr_k6, vmall) ;
	  __vr vrgout_ptr_k7 = _ve_vaddsl_vsv(4*7*gOutHeight*gOutWidth, vrgout_ptr_k0) ;
	  __vr vrgout_k7 = _ve_vgtu_vvm(vrgout_ptr_k7, vmall) ;

	  VFADD_C8(k0)
	  VFADD_C8(k1)
	  VFADD_C8(k2)
	  VFADD_C8(k3)
	  VFADD_C8(k4)
	  VFADD_C8(k5)
	  VFADD_C8(k6)
	  VFADD_C8(k7)
#undef VFADD_C8

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
  } // gOutPixels
}

vednnError_t
vednnConvolutionBackwardData_direct_iwU128(
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
