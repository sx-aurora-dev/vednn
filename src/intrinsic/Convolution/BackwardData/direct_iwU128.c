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

    __vr vrsum = _vel_vbrds_vsl(0.f, vl) ;

    for (int64_t r=0; r<kernHeight; r++) {
      __vr vri = _vel_vaddsl_vsvl(padHeight-r*dilationHeight+h, vrh, vl) ;
      __vr vry = _vel_vdivsl_vvsl(vri, strideHeight, vl) ;

      __vm256 vmy0 =  _vel_vfmkleq_mvl(_vel_vcmpsl_vvvl(vri, _vel_vmulsl_vsvl(strideHeight, vry, vl), vl), vl) ;
      __vm256 vmy1 =  _vel_vfmklge_mvl(vry, vl) ;
      __vm256 vmy2 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutHeight,vry, vl), vl) ;

      __vm256 vmy = _vel_andm_mmm(_vel_andm_mmm(vmy0, vmy1), vmy2) ;

      for (int64_t s=0; s<kernWidth; s++) {
	__vr vrj = _vel_vaddsl_vsvl(padWidth-s*dilationWidth, vrw, vl) ;
	__vr vrx = _vel_vdivsl_vvsl(vrj, strideWidth, vl) ;

	__vm256 vmx0 =  _vel_vfmkleq_mvl(_vel_vcmpsl_vvvl(vrj, _vel_vmulsl_vsvl(strideWidth, vrx, vl), vl), vl) ;
	__vm256 vmx1 =  _vel_vfmklge_mvl(vrx, vl) ;
	__vm256 vmx2 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutWidth,vrx, vl), vl) ;

	__vm256 vmx = _vel_andm_mmm(_vel_andm_mmm(vmx0, vmx1), vmx2) ;

	__vm256 vmall = _vel_andm_mmm(vmy,vmx) ;

	int64_t k=0;
	if( (gOutChannelGroup & 0x01 ) == 1 ) {
	  int64_t gOutIndex    = gOutGroupOffset + ((n * gOutChannel + k) * gOutHeight) * gOutWidth ;
	  const float *pKerValue_k0 = pKernel + kernGroupOffset + (((k  ) * gInChannelGroup + c) * kernHeight + r) * kernWidth + s;

	  __vr vrgout_ptr_k0 = _vel_vsfa_vvssl(_vel_vaddsl_vvvl(_vel_vmulsl_vsvl(gOutWidth, vry, vl), vrx, vl),
					     2,
					     (unsigned long)(pGOut+gOutIndex), vl) ;
	  __vr vrgout_k0 = _vel_vgtu_vvssml(vrgout_ptr_k0, 0, 0, vmall, vl) ;

#define VFADD_C1(KTOKEN)										\
	  vrgout_##KTOKEN = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrgout_##KTOKEN, vmall, vl) ;	\
	  vrsum = _vel_vfmads_vvsvl(vrsum, pKerValue_##KTOKEN[0], vrgout_##KTOKEN, vl) ;

	  VFADD_C1(k0)

	  k+=1 ;
	}
	if( ((gOutChannelGroup >> 1) & 0x01 ) == 1 ) {
	  int64_t gOutIndex    = gOutGroupOffset + ((n * gOutChannel + k) * gOutHeight) * gOutWidth ;
	  const float *pKerValue_k0 = pKernel + kernGroupOffset + (((k  ) * gInChannelGroup + c) * kernHeight + r) * kernWidth + s;
	  const float *pKerValue_k1 = pKernel + kernGroupOffset + (((k+1) * gInChannelGroup + c) * kernHeight + r) * kernWidth + s;

	  __vr vrgout_ptr_k0 = _vel_vsfa_vvssl(_vel_vaddsl_vvvl(_vel_vmulsl_vsvl(gOutWidth, vry, vl), vrx, vl),
					     2,
					     (unsigned long)(pGOut+gOutIndex), vl) ;
	  __vr vrgout_k0 = _vel_vgtu_vvssml(vrgout_ptr_k0, 0, 0, vmall, vl) ;
	  __vr vrgout_ptr_k1 = _vel_vaddsl_vsvl(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0, vl) ;
	  __vr vrgout_k1 = _vel_vgtu_vvssml(vrgout_ptr_k1, 0, 0, vmall, vl) ;


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

	  __vr vrgout_ptr_k0 = _vel_vsfa_vvssl(_vel_vaddsl_vvvl(_vel_vmulsl_vsvl(gOutWidth, vry, vl), vrx, vl),
					     2,
					     (unsigned long)(pGOut+gOutIndex), vl) ;
	  __vr vrgout_k0 = _vel_vgtu_vvssml(vrgout_ptr_k0, 0, 0, vmall, vl) ;
	  __vr vrgout_ptr_k1 = _vel_vaddsl_vsvl(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0, vl) ;
	  __vr vrgout_k1 = _vel_vgtu_vvssml(vrgout_ptr_k1, 0, 0, vmall, vl) ;
	  __vr vrgout_ptr_k2 = _vel_vaddsl_vsvl(4*2*gOutHeight*gOutWidth, vrgout_ptr_k0, vl) ;
	  __vr vrgout_k2 = _vel_vgtu_vvssml(vrgout_ptr_k2, 0, 0, vmall, vl) ;
	  __vr vrgout_ptr_k3 = _vel_vaddsl_vsvl(4*3*gOutHeight*gOutWidth, vrgout_ptr_k0, vl) ;
	  __vr vrgout_k3 = _vel_vgtu_vvssml(vrgout_ptr_k3, 0, 0, vmall, vl) ;

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

	  __vr vrgout_ptr_k0 = _vel_vsfa_vvssl(_vel_vaddsl_vvvl(_vel_vmulsl_vsvl(gOutWidth, vry, vl), vrx, vl),
					     2,
					     (unsigned long)(pGOut+gOutIndex), vl) ;
	  __vr vrgout_k0 = _vel_vgtu_vvssml(vrgout_ptr_k0, 0, 0, vmall, vl) ;
	  __vr vrgout_ptr_k1 = _vel_vaddsl_vsvl(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0, vl) ;
	  __vr vrgout_k1 = _vel_vgtu_vvssml(vrgout_ptr_k1, 0, 0, vmall, vl) ;
	  __vr vrgout_ptr_k2 = _vel_vaddsl_vsvl(4*2*gOutHeight*gOutWidth, vrgout_ptr_k0, vl) ;
	  __vr vrgout_k2 = _vel_vgtu_vvssml(vrgout_ptr_k2, 0, 0, vmall, vl) ;
	  __vr vrgout_ptr_k3 = _vel_vaddsl_vsvl(4*3*gOutHeight*gOutWidth, vrgout_ptr_k0, vl) ;
	  __vr vrgout_k3 = _vel_vgtu_vvssml(vrgout_ptr_k3, 0, 0, vmall, vl) ;
	  __vr vrgout_ptr_k4 = _vel_vaddsl_vsvl(4*4*gOutHeight*gOutWidth, vrgout_ptr_k0, vl) ;
	  __vr vrgout_k4 = _vel_vgtu_vvssml(vrgout_ptr_k4, 0, 0, vmall, vl) ;
	  __vr vrgout_ptr_k5 = _vel_vaddsl_vsvl(4*5*gOutHeight*gOutWidth, vrgout_ptr_k0, vl) ;
	  __vr vrgout_k5 = _vel_vgtu_vvssml(vrgout_ptr_k5, 0, 0, vmall, vl) ;
	  __vr vrgout_ptr_k6 = _vel_vaddsl_vsvl(4*6*gOutHeight*gOutWidth, vrgout_ptr_k0, vl) ;
	  __vr vrgout_k6 = _vel_vgtu_vvssml(vrgout_ptr_k6, 0, 0, vmall, vl) ;
	  __vr vrgout_ptr_k7 = _vel_vaddsl_vsvl(4*7*gOutHeight*gOutWidth, vrgout_ptr_k0, vl) ;
	  __vr vrgout_k7 = _vel_vgtu_vvssml(vrgout_ptr_k7, 0, 0, vmall, vl) ;

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

    _vel_vstu_vssl(vrsum, 4, pGIn+gInIndex, vl) ;
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

    __vr vrsum01 = _vel_pvbrd_vsl(0UL, vl) ;

    for (int64_t r=0; r<kernHeight; r++) {
      __vr vri = _vel_vaddsl_vsvl(padHeight-r*dilationHeight+h, vrh, vl) ;
      __vr vry = _vel_vdivsl_vvsl(vri, strideHeight, vl) ;

      __vm256 vmy0 =  _vel_vfmkleq_mvl(_vel_vcmpsl_vvvl(vri, _vel_vmulsl_vsvl(strideHeight, vry, vl), vl), vl) ;
      __vm256 vmy1 =  _vel_vfmklge_mvl(vry, vl) ;
      __vm256 vmy2 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutHeight,vry, vl), vl) ;

      __vm256 vmy = _vel_andm_mmm(_vel_andm_mmm(vmy0, vmy1), vmy2) ;

      for (int64_t s=0; s<kernWidth; s++) {
	__vr vrj = _vel_vaddsl_vsvl(padWidth-s*dilationWidth, vrw, vl) ;
	__vr vrx = _vel_vdivsl_vvsl(vrj, strideWidth, vl) ;

	__vm256 vmx0 =  _vel_vfmkleq_mvl(_vel_vcmpsl_vvvl(vrj, _vel_vmulsl_vsvl(strideWidth, vrx, vl), vl), vl) ;
	__vm256 vmx1 =  _vel_vfmklge_mvl(vrx, vl) ;
	__vm256 vmx2 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutWidth,vrx, vl), vl) ;

	__vm256 vmx = _vel_andm_mmm(_vel_andm_mmm(vmx0, vmx1), vmx2) ;

	__vm256 vmall = _vel_andm_mmm(vmy,vmx) ;

	int64_t k=0;
	if( (gOutChannelGroup & 0x01 ) == 1 ) {
	  int64_t gOutIndex    = gOutGroupOffset + ((n * gOutChannel + k) * gOutHeight) * gOutWidth ;
	  const float *pKerValue_k0 = pKernel + kernGroupOffset + (((k  ) * gInChannelGroup + c) * kernHeight + r) * kernWidth + s;

	  __vr vrgout_ptr_k0 = _vel_vsfa_vvssl(_vel_vaddsl_vvvl(_vel_vmulsl_vsvl(gOutWidth, vry, vl), vrx, vl),
					     2,
					     (unsigned long)(pGOut+gOutIndex), vl) ;
	  __vr vrgout_k0 = _vel_vgtu_vvssml(vrgout_ptr_k0, 0, 0, vmall, vl) ;

#define VFADD_C2(KTOKEN)												\
	  const uint64_t kerValue01_##KTOKEN = _vel_pack_f32p(pKerValue_##KTOKEN,					\
						              pKerValue_##KTOKEN + 1 * kernHeight * kernWidth ) ;	\
	  vrgout_##KTOKEN = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrgout_##KTOKEN, vmall, vl) ;			\
	  __vr vrgoutP_##KTOKEN = _vel_vshf_vvvsl(vrgout_##KTOKEN, vrgout_##KTOKEN, VE_VSHUFFLE_YUZU, vl) ;		\
	  vrsum01 = _vel_pvfmad_vvsvl(vrsum01, kerValue01_##KTOKEN, vrgoutP_##KTOKEN, vl) ;

	  VFADD_C2(k0)

	  k+=1 ;
	}
	if( ((gOutChannelGroup >> 1) & 0x01 ) == 1 ) {
	  int64_t gOutIndex    = gOutGroupOffset + ((n * gOutChannel + k) * gOutHeight) * gOutWidth ;
	  const float *pKerValue_k0 = pKernel + kernGroupOffset + (((k  ) * gInChannelGroup + c) * kernHeight + r) * kernWidth + s;
	  const float *pKerValue_k1 = pKernel + kernGroupOffset + (((k+1) * gInChannelGroup + c) * kernHeight + r) * kernWidth + s;

	  __vr vrgout_ptr_k0 = _vel_vsfa_vvssl(_vel_vaddsl_vvvl(_vel_vmulsl_vsvl(gOutWidth, vry, vl), vrx, vl),
					     2,
					     (unsigned long)(pGOut+gOutIndex), vl) ;
	  __vr vrgout_k0 = _vel_vgtu_vvssml(vrgout_ptr_k0, 0, 0, vmall, vl) ;
	  __vr vrgout_ptr_k1 = _vel_vaddsl_vsvl(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0, vl) ;
	  __vr vrgout_k1 = _vel_vgtu_vvssml(vrgout_ptr_k1, 0, 0, vmall, vl) ;


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

	  __vr vrgout_ptr_k0 = _vel_vsfa_vvssl(_vel_vaddsl_vvvl(_vel_vmulsl_vsvl(gOutWidth, vry, vl), vrx, vl),
					     2,
					     (unsigned long)(pGOut+gOutIndex), vl) ;
	  __vr vrgout_k0 = _vel_vgtu_vvssml(vrgout_ptr_k0, 0, 0, vmall, vl) ;
	  __vr vrgout_ptr_k1 = _vel_vaddsl_vsvl(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0, vl) ;
	  __vr vrgout_k1 = _vel_vgtu_vvssml(vrgout_ptr_k1, 0, 0, vmall, vl) ;
	  __vr vrgout_ptr_k2 = _vel_vaddsl_vsvl(4*2*gOutHeight*gOutWidth, vrgout_ptr_k0, vl) ;
	  __vr vrgout_k2 = _vel_vgtu_vvssml(vrgout_ptr_k2, 0, 0, vmall, vl) ;
	  __vr vrgout_ptr_k3 = _vel_vaddsl_vsvl(4*3*gOutHeight*gOutWidth, vrgout_ptr_k0, vl) ;
	  __vr vrgout_k3 = _vel_vgtu_vvssml(vrgout_ptr_k3, 0, 0, vmall, vl) ;

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

	  __vr vrgout_ptr_k0 = _vel_vsfa_vvssl(_vel_vaddsl_vvvl(_vel_vmulsl_vsvl(gOutWidth, vry, vl), vrx, vl),
					     2,
					     (unsigned long)(pGOut+gOutIndex), vl) ;
	  __vr vrgout_k0 = _vel_vgtu_vvssml(vrgout_ptr_k0, 0, 0, vmall, vl) ;
	  __vr vrgout_ptr_k1 = _vel_vaddsl_vsvl(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0, vl) ;
	  __vr vrgout_k1 = _vel_vgtu_vvssml(vrgout_ptr_k1, 0, 0, vmall, vl) ;
	  __vr vrgout_ptr_k2 = _vel_vaddsl_vsvl(4*2*gOutHeight*gOutWidth, vrgout_ptr_k0, vl) ;
	  __vr vrgout_k2 = _vel_vgtu_vvssml(vrgout_ptr_k2, 0, 0, vmall, vl) ;
	  __vr vrgout_ptr_k3 = _vel_vaddsl_vsvl(4*3*gOutHeight*gOutWidth, vrgout_ptr_k0, vl) ;
	  __vr vrgout_k3 = _vel_vgtu_vvssml(vrgout_ptr_k3, 0, 0, vmall, vl) ;
	  __vr vrgout_ptr_k4 = _vel_vaddsl_vsvl(4*4*gOutHeight*gOutWidth, vrgout_ptr_k0, vl) ;
	  __vr vrgout_k4 = _vel_vgtu_vvssml(vrgout_ptr_k4, 0, 0, vmall, vl) ;
	  __vr vrgout_ptr_k5 = _vel_vaddsl_vsvl(4*5*gOutHeight*gOutWidth, vrgout_ptr_k0, vl) ;
	  __vr vrgout_k5 = _vel_vgtu_vvssml(vrgout_ptr_k5, 0, 0, vmall, vl) ;
	  __vr vrgout_ptr_k6 = _vel_vaddsl_vsvl(4*6*gOutHeight*gOutWidth, vrgout_ptr_k0, vl) ;
	  __vr vrgout_k6 = _vel_vgtu_vvssml(vrgout_ptr_k6, 0, 0, vmall, vl) ;
	  __vr vrgout_ptr_k7 = _vel_vaddsl_vsvl(4*7*gOutHeight*gOutWidth, vrgout_ptr_k0, vl) ;
	  __vr vrgout_k7 = _vel_vgtu_vvssml(vrgout_ptr_k7, 0, 0, vmall, vl) ;

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

    _vel_vstu_vssl(vrsum01, 4, pGIn+gInIndex, vl) ;
    _vel_vstl_vssl(vrsum01, 4, pGIn+gInIndex+  gInPixels, vl) ;
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

    __vr vrsum01 = _vel_pvbrd_vsl(0UL, vl) ;
    __vr vrsum23 = _vel_pvbrd_vsl(0UL, vl) ;

    for (int64_t r=0; r<kernHeight; r++) {
      __vr vri = _vel_vaddsl_vsvl(padHeight-r*dilationHeight+h, vrh, vl) ;
      __vr vry = _vel_vdivsl_vvsl(vri, strideHeight, vl) ;

      __vm256 vmy0 =  _vel_vfmkleq_mvl(_vel_vcmpsl_vvvl(vri, _vel_vmulsl_vsvl(strideHeight, vry, vl), vl), vl) ;
      __vm256 vmy1 =  _vel_vfmklge_mvl(vry, vl) ;
      __vm256 vmy2 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutHeight,vry, vl), vl) ;

      __vm256 vmy = _vel_andm_mmm(_vel_andm_mmm(vmy0, vmy1), vmy2) ;

      for (int64_t s=0; s<kernWidth; s++) {
	__vr vrj = _vel_vaddsl_vsvl(padWidth-s*dilationWidth, vrw, vl) ;
	__vr vrx = _vel_vdivsl_vvsl(vrj, strideWidth, vl) ;

	__vm256 vmx0 =  _vel_vfmkleq_mvl(_vel_vcmpsl_vvvl(vrj, _vel_vmulsl_vsvl(strideWidth, vrx, vl), vl), vl) ;
	__vm256 vmx1 =  _vel_vfmklge_mvl(vrx, vl) ;
	__vm256 vmx2 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutWidth,vrx, vl), vl) ;

	__vm256 vmx = _vel_andm_mmm(_vel_andm_mmm(vmx0, vmx1), vmx2) ;

	__vm256 vmall = _vel_andm_mmm(vmy,vmx) ;

	int64_t k=0;
	if( (gOutChannelGroup & 0x01 ) == 1 ) {
	  int64_t gOutIndex    = gOutGroupOffset + ((n * gOutChannel + k) * gOutHeight) * gOutWidth ;
	  const float *pKerValue_k0 = pKernel + kernGroupOffset + (((k  ) * gInChannelGroup + c) * kernHeight + r) * kernWidth + s;

	  __vr vrgout_ptr_k0 = _vel_vsfa_vvssl(_vel_vaddsl_vvvl(_vel_vmulsl_vsvl(gOutWidth, vry, vl), vrx, vl),
					     2,
					     (unsigned long)(pGOut+gOutIndex), vl) ;
	  __vr vrgout_k0 = _vel_vgtu_vvssml(vrgout_ptr_k0, 0, 0, vmall, vl) ;

#define VFADD_C4(KTOKEN)												\
	  const uint64_t kerValue01_##KTOKEN = _vel_pack_f32p(pKerValue_##KTOKEN,					\
						              pKerValue_##KTOKEN + 1 * kernHeight * kernWidth ) ;	\
	  const uint64_t kerValue23_##KTOKEN = _vel_pack_f32p(pKerValue_##KTOKEN + 2 * kernHeight * kernWidth,		\
						              pKerValue_##KTOKEN + 3 * kernHeight * kernWidth ) ;	\
	  vrgout_##KTOKEN = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrgout_##KTOKEN, vmall, vl) ;			\
	  __vr vrgoutP_##KTOKEN = _vel_vshf_vvvsl(vrgout_##KTOKEN, vrgout_##KTOKEN, VE_VSHUFFLE_YUZU, vl) ;		\
	  vrsum01 = _vel_pvfmad_vvsvl(vrsum01, kerValue01_##KTOKEN, vrgoutP_##KTOKEN, vl) ;				\
	  vrsum23 = _vel_pvfmad_vvsvl(vrsum23, kerValue23_##KTOKEN, vrgoutP_##KTOKEN, vl) ;

	  VFADD_C4(k0)

	  k+=1 ;
	}
	if( ((gOutChannelGroup >> 1) & 0x01 ) == 1 ) {
	  int64_t gOutIndex    = gOutGroupOffset + ((n * gOutChannel + k) * gOutHeight) * gOutWidth ;
	  const float *pKerValue_k0 = pKernel + kernGroupOffset + (((k  ) * gInChannelGroup + c) * kernHeight + r) * kernWidth + s;
	  const float *pKerValue_k1 = pKernel + kernGroupOffset + (((k+1) * gInChannelGroup + c) * kernHeight + r) * kernWidth + s;

	  __vr vrgout_ptr_k0 = _vel_vsfa_vvssl(_vel_vaddsl_vvvl(_vel_vmulsl_vsvl(gOutWidth, vry, vl), vrx, vl),
					     2,
					     (unsigned long)(pGOut+gOutIndex), vl) ;
	  __vr vrgout_k0 = _vel_vgtu_vvssml(vrgout_ptr_k0, 0, 0, vmall, vl) ;
	  __vr vrgout_ptr_k1 = _vel_vaddsl_vsvl(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0, vl) ;
	  __vr vrgout_k1 = _vel_vgtu_vvssml(vrgout_ptr_k1, 0, 0, vmall, vl) ;


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

	  __vr vrgout_ptr_k0 = _vel_vsfa_vvssl(_vel_vaddsl_vvvl(_vel_vmulsl_vsvl(gOutWidth, vry, vl), vrx, vl),
					     2,
					     (unsigned long)(pGOut+gOutIndex), vl) ;
	  __vr vrgout_k0 = _vel_vgtu_vvssml(vrgout_ptr_k0, 0, 0, vmall, vl) ;
	  __vr vrgout_ptr_k1 = _vel_vaddsl_vsvl(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0, vl) ;
	  __vr vrgout_k1 = _vel_vgtu_vvssml(vrgout_ptr_k1, 0, 0, vmall, vl) ;
	  __vr vrgout_ptr_k2 = _vel_vaddsl_vsvl(4*2*gOutHeight*gOutWidth, vrgout_ptr_k0, vl) ;
	  __vr vrgout_k2 = _vel_vgtu_vvssml(vrgout_ptr_k2, 0, 0, vmall, vl) ;
	  __vr vrgout_ptr_k3 = _vel_vaddsl_vsvl(4*3*gOutHeight*gOutWidth, vrgout_ptr_k0, vl) ;
	  __vr vrgout_k3 = _vel_vgtu_vvssml(vrgout_ptr_k3, 0, 0, vmall, vl) ;

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

	  __vr vrgout_ptr_k0 = _vel_vsfa_vvssl(_vel_vaddsl_vvvl(_vel_vmulsl_vsvl(gOutWidth, vry, vl), vrx, vl),
					     2,
					     (unsigned long)(pGOut+gOutIndex), vl) ;
	  __vr vrgout_k0 = _vel_vgtu_vvssml(vrgout_ptr_k0, 0, 0, vmall, vl) ;
	  __vr vrgout_ptr_k1 = _vel_vaddsl_vsvl(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0, vl) ;
	  __vr vrgout_k1 = _vel_vgtu_vvssml(vrgout_ptr_k1, 0, 0, vmall, vl) ;
	  __vr vrgout_ptr_k2 = _vel_vaddsl_vsvl(4*2*gOutHeight*gOutWidth, vrgout_ptr_k0, vl) ;
	  __vr vrgout_k2 = _vel_vgtu_vvssml(vrgout_ptr_k2, 0, 0, vmall, vl) ;
	  __vr vrgout_ptr_k3 = _vel_vaddsl_vsvl(4*3*gOutHeight*gOutWidth, vrgout_ptr_k0, vl) ;
	  __vr vrgout_k3 = _vel_vgtu_vvssml(vrgout_ptr_k3, 0, 0, vmall, vl) ;
	  __vr vrgout_ptr_k4 = _vel_vaddsl_vsvl(4*4*gOutHeight*gOutWidth, vrgout_ptr_k0, vl) ;
	  __vr vrgout_k4 = _vel_vgtu_vvssml(vrgout_ptr_k4, 0, 0, vmall, vl) ;
	  __vr vrgout_ptr_k5 = _vel_vaddsl_vsvl(4*5*gOutHeight*gOutWidth, vrgout_ptr_k0, vl) ;
	  __vr vrgout_k5 = _vel_vgtu_vvssml(vrgout_ptr_k5, 0, 0, vmall, vl) ;
	  __vr vrgout_ptr_k6 = _vel_vaddsl_vsvl(4*6*gOutHeight*gOutWidth, vrgout_ptr_k0, vl) ;
	  __vr vrgout_k6 = _vel_vgtu_vvssml(vrgout_ptr_k6, 0, 0, vmall, vl) ;
	  __vr vrgout_ptr_k7 = _vel_vaddsl_vsvl(4*7*gOutHeight*gOutWidth, vrgout_ptr_k0, vl) ;
	  __vr vrgout_k7 = _vel_vgtu_vvssml(vrgout_ptr_k7, 0, 0, vmall, vl) ;

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

    _vel_vstu_vssl(vrsum01, 4, pGIn+gInIndex, vl) ;
    _vel_vstl_vssl(vrsum01, 4, pGIn+gInIndex+  gInPixels, vl) ;
    _vel_vstu_vssl(vrsum23, 4, pGIn+gInIndex+2*gInPixels, vl) ;
    _vel_vstl_vssl(vrsum23, 4, pGIn+gInIndex+3*gInPixels, vl) ;
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

    __vr vrsum01 = _vel_pvbrd_vsl(0UL, vl) ;
    __vr vrsum23 = _vel_pvbrd_vsl(0UL, vl) ;
    __vr vrsum45 = _vel_pvbrd_vsl(0UL, vl) ;
    __vr vrsum67 = _vel_pvbrd_vsl(0UL, vl) ;

    for (int64_t r=0; r<kernHeight; r++) {
      __vr vri = _vel_vaddsl_vsvl(padHeight-r*dilationHeight+h, vrh, vl) ;
      __vr vry = _vel_vdivsl_vvsl(vri, strideHeight, vl) ;

      __vm256 vmy0 =  _vel_vfmkleq_mvl(_vel_vcmpsl_vvvl(vri, _vel_vmulsl_vsvl(strideHeight, vry, vl), vl), vl) ;
      __vm256 vmy1 =  _vel_vfmklge_mvl(vry, vl) ;
      __vm256 vmy2 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutHeight,vry, vl), vl) ;

      __vm256 vmy = _vel_andm_mmm(_vel_andm_mmm(vmy0, vmy1), vmy2) ;

      for (int64_t s=0; s<kernWidth; s++) {
	__vr vrj = _vel_vaddsl_vsvl(padWidth-s*dilationWidth, vrw, vl) ;
	__vr vrx = _vel_vdivsl_vvsl(vrj, strideWidth, vl) ;

	__vm256 vmx0 =  _vel_vfmkleq_mvl(_vel_vcmpsl_vvvl(vrj, _vel_vmulsl_vsvl(strideWidth, vrx, vl), vl), vl) ;
	__vm256 vmx1 =  _vel_vfmklge_mvl(vrx, vl) ;
	__vm256 vmx2 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutWidth,vrx, vl), vl) ;

	__vm256 vmx = _vel_andm_mmm(_vel_andm_mmm(vmx0, vmx1), vmx2) ;

	__vm256 vmall = _vel_andm_mmm(vmy,vmx) ;

	int64_t k=0;
	if( (gOutChannelGroup & 0x01 ) == 1 ) {
	  int64_t gOutIndex    = gOutGroupOffset + ((n * gOutChannel + k) * gOutHeight) * gOutWidth ;
	  const float *pKerValue_k0 = pKernel + kernGroupOffset + (((k  ) * gInChannelGroup + c) * kernHeight + r) * kernWidth + s;

	  __vr vrgout_ptr_k0 = _vel_vsfa_vvssl(_vel_vaddsl_vvvl(_vel_vmulsl_vsvl(gOutWidth, vry, vl), vrx, vl),
					     2,
					     (unsigned long)(pGOut+gOutIndex), vl) ;
	  __vr vrgout_k0 = _vel_vgtu_vvssml(vrgout_ptr_k0, 0, 0, vmall, vl) ;

#define VFADD_C8(KTOKEN)												\
	  const uint64_t kerValue01_##KTOKEN = _vel_pack_f32p(pKerValue_##KTOKEN,					\
						              pKerValue_##KTOKEN + 1 * kernHeight * kernWidth ) ;	\
	  const uint64_t kerValue23_##KTOKEN = _vel_pack_f32p(pKerValue_##KTOKEN + 2 * kernHeight * kernWidth,		\
						              pKerValue_##KTOKEN + 3 * kernHeight * kernWidth ) ;	\
	  const uint64_t kerValue45_##KTOKEN = _vel_pack_f32p(pKerValue_##KTOKEN + 4 * kernHeight * kernWidth,		\
						              pKerValue_##KTOKEN + 5 * kernHeight * kernWidth ) ;	\
	  const uint64_t kerValue67_##KTOKEN = _vel_pack_f32p(pKerValue_##KTOKEN + 6 * kernHeight * kernWidth,		\
						              pKerValue_##KTOKEN + 7 * kernHeight * kernWidth ) ;	\
	  vrgout_##KTOKEN = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrgout_##KTOKEN, vmall, vl) ;			\
	  __vr vrgoutP_##KTOKEN = _vel_vshf_vvvsl(vrgout_##KTOKEN, vrgout_##KTOKEN, VE_VSHUFFLE_YUZU, vl) ;		\
	  vrsum01 = _vel_pvfmad_vvsvl(vrsum01, kerValue01_##KTOKEN, vrgoutP_##KTOKEN, vl) ;				\
	  vrsum23 = _vel_pvfmad_vvsvl(vrsum23, kerValue23_##KTOKEN, vrgoutP_##KTOKEN, vl) ;				\
	  vrsum45 = _vel_pvfmad_vvsvl(vrsum45, kerValue45_##KTOKEN, vrgoutP_##KTOKEN, vl) ;				\
	  vrsum67 = _vel_pvfmad_vvsvl(vrsum67, kerValue67_##KTOKEN, vrgoutP_##KTOKEN, vl) ;

	  VFADD_C8(k0)

	  k+=1 ;
	}
	if( ((gOutChannelGroup >> 1) & 0x01 ) == 1 ) {
	  int64_t gOutIndex    = gOutGroupOffset + ((n * gOutChannel + k) * gOutHeight) * gOutWidth ;
	  const float *pKerValue_k0 = pKernel + kernGroupOffset + (((k  ) * gInChannelGroup + c) * kernHeight + r) * kernWidth + s;
	  const float *pKerValue_k1 = pKernel + kernGroupOffset + (((k+1) * gInChannelGroup + c) * kernHeight + r) * kernWidth + s;

	  __vr vrgout_ptr_k0 = _vel_vsfa_vvssl(_vel_vaddsl_vvvl(_vel_vmulsl_vsvl(gOutWidth, vry, vl), vrx, vl),
					     2,
					     (unsigned long)(pGOut+gOutIndex), vl) ;
	  __vr vrgout_k0 = _vel_vgtu_vvssml(vrgout_ptr_k0, 0, 0, vmall, vl) ;
	  __vr vrgout_ptr_k1 = _vel_vaddsl_vsvl(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0, vl) ;
	  __vr vrgout_k1 = _vel_vgtu_vvssml(vrgout_ptr_k1, 0, 0, vmall, vl) ;


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

	  __vr vrgout_ptr_k0 = _vel_vsfa_vvssl(_vel_vaddsl_vvvl(_vel_vmulsl_vsvl(gOutWidth, vry, vl), vrx, vl),
					     2,
					     (unsigned long)(pGOut+gOutIndex), vl) ;
	  __vr vrgout_k0 = _vel_vgtu_vvssml(vrgout_ptr_k0, 0, 0, vmall, vl) ;
	  __vr vrgout_ptr_k1 = _vel_vaddsl_vsvl(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0, vl) ;
	  __vr vrgout_k1 = _vel_vgtu_vvssml(vrgout_ptr_k1, 0, 0, vmall, vl) ;
	  __vr vrgout_ptr_k2 = _vel_vaddsl_vsvl(4*2*gOutHeight*gOutWidth, vrgout_ptr_k0, vl) ;
	  __vr vrgout_k2 = _vel_vgtu_vvssml(vrgout_ptr_k2, 0, 0, vmall, vl) ;
	  __vr vrgout_ptr_k3 = _vel_vaddsl_vsvl(4*3*gOutHeight*gOutWidth, vrgout_ptr_k0, vl) ;
	  __vr vrgout_k3 = _vel_vgtu_vvssml(vrgout_ptr_k3, 0, 0, vmall, vl) ;

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

	  __vr vrgout_ptr_k0 = _vel_vsfa_vvssl(_vel_vaddsl_vvvl(_vel_vmulsl_vsvl(gOutWidth, vry, vl), vrx, vl),
					     2,
					     (unsigned long)(pGOut+gOutIndex), vl) ;
	  __vr vrgout_k0 = _vel_vgtu_vvssml(vrgout_ptr_k0, 0, 0, vmall, vl) ;
	  __vr vrgout_ptr_k1 = _vel_vaddsl_vsvl(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0, vl) ;
	  __vr vrgout_k1 = _vel_vgtu_vvssml(vrgout_ptr_k1, 0, 0, vmall, vl) ;
	  __vr vrgout_ptr_k2 = _vel_vaddsl_vsvl(4*2*gOutHeight*gOutWidth, vrgout_ptr_k0, vl) ;
	  __vr vrgout_k2 = _vel_vgtu_vvssml(vrgout_ptr_k2, 0, 0, vmall, vl) ;
	  __vr vrgout_ptr_k3 = _vel_vaddsl_vsvl(4*3*gOutHeight*gOutWidth, vrgout_ptr_k0, vl) ;
	  __vr vrgout_k3 = _vel_vgtu_vvssml(vrgout_ptr_k3, 0, 0, vmall, vl) ;
	  __vr vrgout_ptr_k4 = _vel_vaddsl_vsvl(4*4*gOutHeight*gOutWidth, vrgout_ptr_k0, vl) ;
	  __vr vrgout_k4 = _vel_vgtu_vvssml(vrgout_ptr_k4, 0, 0, vmall, vl) ;
	  __vr vrgout_ptr_k5 = _vel_vaddsl_vsvl(4*5*gOutHeight*gOutWidth, vrgout_ptr_k0, vl) ;
	  __vr vrgout_k5 = _vel_vgtu_vvssml(vrgout_ptr_k5, 0, 0, vmall, vl) ;
	  __vr vrgout_ptr_k6 = _vel_vaddsl_vsvl(4*6*gOutHeight*gOutWidth, vrgout_ptr_k0, vl) ;
	  __vr vrgout_k6 = _vel_vgtu_vvssml(vrgout_ptr_k6, 0, 0, vmall, vl) ;
	  __vr vrgout_ptr_k7 = _vel_vaddsl_vsvl(4*7*gOutHeight*gOutWidth, vrgout_ptr_k0, vl) ;
	  __vr vrgout_k7 = _vel_vgtu_vvssml(vrgout_ptr_k7, 0, 0, vmall, vl) ;

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

    _vel_vstu_vssl(vrsum01, 4, pGIn+gInIndex, vl) ;
    _vel_vstl_vssl(vrsum01, 4, pGIn+gInIndex+  gInPixels, vl) ;
    _vel_vstu_vssl(vrsum23, 4, pGIn+gInIndex+2*gInPixels, vl) ;
    _vel_vstl_vssl(vrsum23, 4, pGIn+gInIndex+3*gInPixels, vl) ;
    _vel_vstu_vssl(vrsum45, 4, pGIn+gInIndex+4*gInPixels, vl) ;
    _vel_vstl_vssl(vrsum45, 4, pGIn+gInIndex+5*gInPixels, vl) ;
    _vel_vstu_vssl(vrsum67, 4, pGIn+gInIndex+6*gInPixels, vl) ;
    _vel_vstl_vssl(vrsum67, 4, pGIn+gInIndex+7*gInPixels, vl) ;
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

     ;
    __vr vrseq = _vel_vseq_vl(nH*gInWidth) ;
    __vr vrh  = _vel_vdivsl_vvsl(vrseq, gInWidth, nH*gInWidth) ;
    __vr vrw  = _vel_vsubsl_vvvl(vrseq, _vel_vmulul_vsvl(gInWidth,vrh, nH*gInWidth), nH*gInWidth) ;

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
