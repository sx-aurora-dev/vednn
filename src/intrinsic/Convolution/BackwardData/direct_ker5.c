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

      __vr vrj_s0 = _vel_vaddsl_vsvl(padWidth-0*dilationWidth, vrw, vl) ;
      __vr vrj_s1 = _vel_vaddsl_vsvl(padWidth-1*dilationWidth, vrw, vl) ;
      __vr vrj_s2 = _vel_vaddsl_vsvl(padWidth-2*dilationWidth, vrw, vl) ;
      __vr vrj_s3 = _vel_vaddsl_vsvl(padWidth-3*dilationWidth, vrw, vl) ;
      __vr vrj_s4 = _vel_vaddsl_vsvl(padWidth-4*dilationWidth, vrw, vl) ;

      __vr vrx_s0 = _vel_vdivsl_vvsl(vrj_s0, strideWidth, vl) ;
      __vr vrx_s1 = _vel_vdivsl_vvsl(vrj_s1, strideWidth, vl) ;
      __vr vrx_s2 = _vel_vdivsl_vvsl(vrj_s2, strideWidth, vl) ;
      __vr vrx_s3 = _vel_vdivsl_vvsl(vrj_s3, strideWidth, vl) ;
      __vr vrx_s4 = _vel_vdivsl_vvsl(vrj_s4, strideWidth, vl) ;

      __vm256 vmx0_s0 =  _vel_vfmkleq_mvl(_vel_vcmpsl_vvvl(vrj_s0, _vel_vmulsl_vsvl(strideWidth, vrx_s0, vl), vl), vl) ;
      __vm256 vmx1_s0 =  _vel_vfmklge_mvl(vrx_s0, vl) ;
      __vm256 vmx2_s0 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutWidth,vrx_s0, vl), vl) ;
      __vm256 vmx_s0 = _vel_andm_mmm(_vel_andm_mmm(vmx0_s0, vmx1_s0), vmx2_s0) ;

      __vm256 vmx0_s1 =  _vel_vfmkleq_mvl(_vel_vcmpsl_vvvl(vrj_s1, _vel_vmulsl_vsvl(strideWidth, vrx_s1, vl), vl), vl) ;
      __vm256 vmx1_s1 =  _vel_vfmklge_mvl(vrx_s1, vl) ;
      __vm256 vmx2_s1 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutWidth,vrx_s1, vl), vl) ;
      __vm256 vmx_s1 = _vel_andm_mmm(_vel_andm_mmm(vmx0_s1, vmx1_s1), vmx2_s1) ;

      __vm256 vmx0_s2 =  _vel_vfmkleq_mvl(_vel_vcmpsl_vvvl(vrj_s2, _vel_vmulsl_vsvl(strideWidth, vrx_s2, vl), vl), vl) ;
      __vm256 vmx1_s2 =  _vel_vfmklge_mvl(vrx_s2, vl) ;
      __vm256 vmx2_s2 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutWidth,vrx_s2, vl), vl) ;
      __vm256 vmx_s2 = _vel_andm_mmm(_vel_andm_mmm(vmx0_s2, vmx1_s2), vmx2_s2) ;

      __vm256 vmx0_s3 =  _vel_vfmkleq_mvl(_vel_vcmpsl_vvvl(vrj_s3, _vel_vmulsl_vsvl(strideWidth, vrx_s3, vl), vl), vl) ;
      __vm256 vmx1_s3 =  _vel_vfmklge_mvl(vrx_s3, vl) ;
      __vm256 vmx2_s3 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutWidth,vrx_s3, vl), vl) ;
      __vm256 vmx_s3 = _vel_andm_mmm(_vel_andm_mmm(vmx0_s3, vmx1_s3), vmx2_s3) ;

      __vm256 vmx0_s4 =  _vel_vfmkleq_mvl(_vel_vcmpsl_vvvl(vrj_s4, _vel_vmulsl_vsvl(strideWidth, vrx_s4, vl), vl), vl) ;
      __vm256 vmx1_s4 =  _vel_vfmklge_mvl(vrx_s4, vl) ;
      __vm256 vmx2_s4 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutWidth,vrx_s4, vl), vl) ;
      __vm256 vmx_s4 = _vel_andm_mmm(_vel_andm_mmm(vmx0_s4, vmx1_s4), vmx2_s4) ;

      for (int64_t r=0; r<kernHeight; r++) {
	int64_t i = h - r * dilationHeight + padHeight ;
	int64_t y = i/strideHeight;
	if ( y*strideHeight != i || y < 0 || gOutHeight <= y)  continue ;

	int64_t k=0;
	if( (gOutChannelGroup & 0x01)  == 1 ) {
	  int64_t gOutIndex    = gOutGroupOffset + ((n * gOutChannel + k) * gOutHeight) * gOutWidth ;
	  const float *pKerValue = pKernel + kernGroupOffset + ((k * gInChannelGroup + c) * kernHeight) * kernWidth;

	  __vr vrgout_ptr_k0_s0 = _vel_vsfa_vvssl(_vel_vaddsl_vsvl(gOutWidth*y, vrx_s0, vl),
					     2,
					     (unsigned long)(pGOut+gOutIndex), vl) ;
	  __vr vrgout_k0_s0 = _vel_vgtu_vvssml(vrgout_ptr_k0_s0, 0, 0, vmx_s0, vl) ;
	  __vr vrgout_ptr_k0_s1 = _vel_vsfa_vvssl(_vel_vaddsl_vsvl(gOutWidth*y, vrx_s1, vl),
					     2,
					     (unsigned long)(pGOut+gOutIndex), vl) ;
	  __vr vrgout_k0_s1 = _vel_vgtu_vvssml(vrgout_ptr_k0_s1, 0, 0, vmx_s1, vl) ;
	  __vr vrgout_ptr_k0_s2 = _vel_vsfa_vvssl(_vel_vaddsl_vsvl(gOutWidth*y, vrx_s2, vl),
					     2,
					     (unsigned long)(pGOut+gOutIndex), vl) ;
	  __vr vrgout_k0_s2 = _vel_vgtu_vvssml(vrgout_ptr_k0_s2, 0, 0, vmx_s2, vl) ;
	  __vr vrgout_ptr_k0_s3 = _vel_vsfa_vvssl(_vel_vaddsl_vsvl(gOutWidth*y, vrx_s3, vl),
					     2,
					     (unsigned long)(pGOut+gOutIndex), vl) ;
	  __vr vrgout_k0_s3 = _vel_vgtu_vvssml(vrgout_ptr_k0_s3, 0, 0, vmx_s3, vl) ;
	  __vr vrgout_ptr_k0_s4 = _vel_vsfa_vvssl(_vel_vaddsl_vsvl(gOutWidth*y, vrx_s4, vl),
					     2,
					     (unsigned long)(pGOut+gOutIndex), vl) ;
	  __vr vrgout_k0_s4 = _vel_vgtu_vvssml(vrgout_ptr_k0_s4, 0, 0, vmx_s4, vl) ;

#define VFMAD_C1(VRGOUT, VM, K, R, S)	{										\
	    const float kerValue = pKerValue[(((K)*gInChannelGroup + 0) * kernHeight +(R)) * kernWidth + (S) ] ;	\
	    VRGOUT = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), VRGOUT, VM, vl) ;					\
	    vrsum = _vel_vfmads_vvsvl(vrsum, kerValue, VRGOUT, vl) ;							\
}
	  VFMAD_C1(vrgout_k0_s0, vmx_s0, 0, r, 0) ;
	  VFMAD_C1(vrgout_k0_s1, vmx_s1, 0, r, 1) ;
	  VFMAD_C1(vrgout_k0_s2, vmx_s2, 0, r, 2) ;
	  VFMAD_C1(vrgout_k0_s3, vmx_s3, 0, r, 3) ;
	  VFMAD_C1(vrgout_k0_s4, vmx_s4, 0, r, 4) ;

	  k+=1 ;
	}
	if( ( (gOutChannelGroup >> 1) & 0x01)  == 1 ) {
	  int64_t gOutIndex    = gOutGroupOffset + ((n * gOutChannel + k) * gOutHeight) * gOutWidth ;
	  const float *pKerValue = pKernel + kernGroupOffset + ((k * gInChannelGroup + c) * kernHeight) * kernWidth;

	  __vr vrgout_ptr_k0_s0 = _vel_vsfa_vvssl(_vel_vaddsl_vsvl(gOutWidth*y, vrx_s0, vl),
					     2,
					     (unsigned long)(pGOut+gOutIndex), vl) ;
	  __vr vrgout_k0_s0 = _vel_vgtu_vvssml(vrgout_ptr_k0_s0, 0, 0, vmx_s0, vl) ;
	  __vr vrgout_ptr_k0_s1 = _vel_vsfa_vvssl(_vel_vaddsl_vsvl(gOutWidth*y, vrx_s1, vl),
					     2,
					     (unsigned long)(pGOut+gOutIndex), vl) ;
	  __vr vrgout_k0_s1 = _vel_vgtu_vvssml(vrgout_ptr_k0_s1, 0, 0, vmx_s1, vl) ;
	  __vr vrgout_ptr_k0_s2 = _vel_vsfa_vvssl(_vel_vaddsl_vsvl(gOutWidth*y, vrx_s2, vl),
					     2,
					     (unsigned long)(pGOut+gOutIndex), vl) ;
	  __vr vrgout_k0_s2 = _vel_vgtu_vvssml(vrgout_ptr_k0_s2, 0, 0, vmx_s2, vl) ;
	  __vr vrgout_ptr_k0_s3 = _vel_vsfa_vvssl(_vel_vaddsl_vsvl(gOutWidth*y, vrx_s3, vl),
					     2,
					     (unsigned long)(pGOut+gOutIndex), vl) ;
	  __vr vrgout_k0_s3 = _vel_vgtu_vvssml(vrgout_ptr_k0_s3, 0, 0, vmx_s3, vl) ;
	  __vr vrgout_ptr_k0_s4 = _vel_vsfa_vvssl(_vel_vaddsl_vsvl(gOutWidth*y, vrx_s4, vl),
					     2,
					     (unsigned long)(pGOut+gOutIndex), vl) ;
	  __vr vrgout_k0_s4 = _vel_vgtu_vvssml(vrgout_ptr_k0_s4, 0, 0, vmx_s4, vl) ;
	  VFMAD_C1(vrgout_k0_s0, vmx_s0, 0, r, 0) ;
	  VFMAD_C1(vrgout_k0_s1, vmx_s1, 0, r, 1) ;
	  VFMAD_C1(vrgout_k0_s2, vmx_s2, 0, r, 2) ;
	  VFMAD_C1(vrgout_k0_s3, vmx_s3, 0, r, 3) ;
	  VFMAD_C1(vrgout_k0_s4, vmx_s4, 0, r, 4) ;

	  __vr vrgout_ptr_k1_s0 = _vel_vaddsl_vsvl(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_s0, vl) ;
	  __vr vrgout_k1_s0 = _vel_vgtu_vvssml(vrgout_ptr_k1_s0, 0, 0, vmx_s0, vl) ;
	  __vr vrgout_ptr_k1_s1 = _vel_vaddsl_vsvl(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_s1, vl) ;
	  __vr vrgout_k1_s1 = _vel_vgtu_vvssml(vrgout_ptr_k1_s1, 0, 0, vmx_s1, vl) ;
	  __vr vrgout_ptr_k1_s2 = _vel_vaddsl_vsvl(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_s2, vl) ;
	  __vr vrgout_k1_s2 = _vel_vgtu_vvssml(vrgout_ptr_k1_s2, 0, 0, vmx_s2, vl) ;
	  __vr vrgout_ptr_k1_s3 = _vel_vaddsl_vsvl(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_s3, vl) ;
	  __vr vrgout_k1_s3 = _vel_vgtu_vvssml(vrgout_ptr_k1_s3, 0, 0, vmx_s3, vl) ;
	  __vr vrgout_ptr_k1_s4 = _vel_vaddsl_vsvl(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_s4, vl) ;
	  __vr vrgout_k1_s4 = _vel_vgtu_vvssml(vrgout_ptr_k1_s4, 0, 0, vmx_s4, vl) ;
	  VFMAD_C1(vrgout_k1_s0, vmx_s0, 1, r, 0) ;
	  VFMAD_C1(vrgout_k1_s1, vmx_s1, 1, r, 1) ;
	  VFMAD_C1(vrgout_k1_s2, vmx_s2, 1, r, 2) ;
	  VFMAD_C1(vrgout_k1_s3, vmx_s3, 1, r, 3) ;
	  VFMAD_C1(vrgout_k1_s4, vmx_s4, 1, r, 4) ;

	  k+=2 ;
	}
	if( ( (gOutChannelGroup >> 2) & 0x01)  == 1 ) {
	  int64_t gOutIndex    = gOutGroupOffset + ((n * gOutChannel + k) * gOutHeight) * gOutWidth ;
	  const float *pKerValue = pKernel + kernGroupOffset + ((k * gInChannelGroup + c) * kernHeight) * kernWidth;

	  __vr vrgout_ptr_k0_s0 = _vel_vsfa_vvssl(_vel_vaddsl_vsvl(gOutWidth*y, vrx_s0, vl),
					     2,
					     (unsigned long)(pGOut+gOutIndex), vl) ;
	  __vr vrgout_k0_s0 = _vel_vgtu_vvssml(vrgout_ptr_k0_s0, 0, 0, vmx_s0, vl) ;
	  __vr vrgout_ptr_k0_s1 = _vel_vsfa_vvssl(_vel_vaddsl_vsvl(gOutWidth*y, vrx_s1, vl),
					     2,
					     (unsigned long)(pGOut+gOutIndex), vl) ;
	  __vr vrgout_k0_s1 = _vel_vgtu_vvssml(vrgout_ptr_k0_s1, 0, 0, vmx_s1, vl) ;
	  __vr vrgout_ptr_k0_s2 = _vel_vsfa_vvssl(_vel_vaddsl_vsvl(gOutWidth*y, vrx_s2, vl),
					     2,
					     (unsigned long)(pGOut+gOutIndex), vl) ;
	  __vr vrgout_k0_s2 = _vel_vgtu_vvssml(vrgout_ptr_k0_s2, 0, 0, vmx_s2, vl) ;
	  __vr vrgout_ptr_k0_s3 = _vel_vsfa_vvssl(_vel_vaddsl_vsvl(gOutWidth*y, vrx_s3, vl),
					     2,
					     (unsigned long)(pGOut+gOutIndex), vl) ;
	  __vr vrgout_k0_s3 = _vel_vgtu_vvssml(vrgout_ptr_k0_s3, 0, 0, vmx_s3, vl) ;
	  __vr vrgout_ptr_k0_s4 = _vel_vsfa_vvssl(_vel_vaddsl_vsvl(gOutWidth*y, vrx_s4, vl),
					     2,
					     (unsigned long)(pGOut+gOutIndex), vl) ;
	  __vr vrgout_k0_s4 = _vel_vgtu_vvssml(vrgout_ptr_k0_s4, 0, 0, vmx_s4, vl) ;
	  VFMAD_C1(vrgout_k0_s0, vmx_s0, 0, r, 0) ;
	  VFMAD_C1(vrgout_k0_s1, vmx_s1, 0, r, 1) ;
	  VFMAD_C1(vrgout_k0_s2, vmx_s2, 0, r, 2) ;
	  VFMAD_C1(vrgout_k0_s3, vmx_s3, 0, r, 3) ;
	  VFMAD_C1(vrgout_k0_s4, vmx_s4, 0, r, 4) ;

	  __vr vrgout_ptr_k1_s0 = _vel_vaddsl_vsvl(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_s0, vl) ;
	  __vr vrgout_k1_s0 = _vel_vgtu_vvssml(vrgout_ptr_k1_s0, 0, 0, vmx_s0, vl) ;
	  __vr vrgout_ptr_k1_s1 = _vel_vaddsl_vsvl(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_s1, vl) ;
	  __vr vrgout_k1_s1 = _vel_vgtu_vvssml(vrgout_ptr_k1_s1, 0, 0, vmx_s1, vl) ;
	  __vr vrgout_ptr_k1_s2 = _vel_vaddsl_vsvl(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_s2, vl) ;
	  __vr vrgout_k1_s2 = _vel_vgtu_vvssml(vrgout_ptr_k1_s2, 0, 0, vmx_s2, vl) ;
	  __vr vrgout_ptr_k1_s3 = _vel_vaddsl_vsvl(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_s3, vl) ;
	  __vr vrgout_k1_s3 = _vel_vgtu_vvssml(vrgout_ptr_k1_s3, 0, 0, vmx_s3, vl) ;
	  __vr vrgout_ptr_k1_s4 = _vel_vaddsl_vsvl(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_s4, vl) ;
	  __vr vrgout_k1_s4 = _vel_vgtu_vvssml(vrgout_ptr_k1_s4, 0, 0, vmx_s4, vl) ;
	  VFMAD_C1(vrgout_k1_s0, vmx_s0, 1, r, 0) ;
	  VFMAD_C1(vrgout_k1_s1, vmx_s1, 1, r, 1) ;
	  VFMAD_C1(vrgout_k1_s2, vmx_s2, 1, r, 2) ;
	  VFMAD_C1(vrgout_k1_s3, vmx_s3, 1, r, 3) ;
	  VFMAD_C1(vrgout_k1_s4, vmx_s4, 1, r, 4) ;

	  __vr vrgout_ptr_k2_s0 = _vel_vaddsl_vsvl(4*2*gOutHeight*gOutWidth, vrgout_ptr_k0_s0, vl) ;
	  __vr vrgout_k2_s0 = _vel_vgtu_vvssml(vrgout_ptr_k2_s0, 0, 0, vmx_s0, vl) ;
	  __vr vrgout_ptr_k2_s1 = _vel_vaddsl_vsvl(4*2*gOutHeight*gOutWidth, vrgout_ptr_k0_s1, vl) ;
	  __vr vrgout_k2_s1 = _vel_vgtu_vvssml(vrgout_ptr_k2_s1, 0, 0, vmx_s1, vl) ;
	  __vr vrgout_ptr_k2_s2 = _vel_vaddsl_vsvl(4*2*gOutHeight*gOutWidth, vrgout_ptr_k0_s2, vl) ;
	  __vr vrgout_k2_s2 = _vel_vgtu_vvssml(vrgout_ptr_k2_s2, 0, 0, vmx_s2, vl) ;
	  __vr vrgout_ptr_k2_s3 = _vel_vaddsl_vsvl(4*2*gOutHeight*gOutWidth, vrgout_ptr_k0_s3, vl) ;
	  __vr vrgout_k2_s3 = _vel_vgtu_vvssml(vrgout_ptr_k2_s3, 0, 0, vmx_s3, vl) ;
	  __vr vrgout_ptr_k2_s4 = _vel_vaddsl_vsvl(4*2*gOutHeight*gOutWidth, vrgout_ptr_k0_s4, vl) ;
	  __vr vrgout_k2_s4 = _vel_vgtu_vvssml(vrgout_ptr_k2_s4, 0, 0, vmx_s4, vl) ;
	  VFMAD_C1(vrgout_k2_s0, vmx_s0, 2, r, 0) ;
	  VFMAD_C1(vrgout_k2_s1, vmx_s1, 2, r, 1) ;
	  VFMAD_C1(vrgout_k2_s2, vmx_s2, 2, r, 2) ;
	  VFMAD_C1(vrgout_k2_s3, vmx_s3, 2, r, 3) ;
	  VFMAD_C1(vrgout_k2_s4, vmx_s4, 2, r, 4) ;

	  __vr vrgout_ptr_k3_s0 = _vel_vaddsl_vsvl(4*3*gOutHeight*gOutWidth, vrgout_ptr_k0_s0, vl) ;
	  __vr vrgout_k3_s0 = _vel_vgtu_vvssml(vrgout_ptr_k3_s0, 0, 0, vmx_s0, vl) ;
	  __vr vrgout_ptr_k3_s1 = _vel_vaddsl_vsvl(4*3*gOutHeight*gOutWidth, vrgout_ptr_k0_s1, vl) ;
	  __vr vrgout_k3_s1 = _vel_vgtu_vvssml(vrgout_ptr_k3_s1, 0, 0, vmx_s1, vl) ;
	  __vr vrgout_ptr_k3_s2 = _vel_vaddsl_vsvl(4*3*gOutHeight*gOutWidth, vrgout_ptr_k0_s2, vl) ;
	  __vr vrgout_k3_s2 = _vel_vgtu_vvssml(vrgout_ptr_k3_s2, 0, 0, vmx_s2, vl) ;
	  __vr vrgout_ptr_k3_s3 = _vel_vaddsl_vsvl(4*3*gOutHeight*gOutWidth, vrgout_ptr_k0_s3, vl) ;
	  __vr vrgout_k3_s3 = _vel_vgtu_vvssml(vrgout_ptr_k3_s3, 0, 0, vmx_s3, vl) ;
	  __vr vrgout_ptr_k3_s4 = _vel_vaddsl_vsvl(4*3*gOutHeight*gOutWidth, vrgout_ptr_k0_s4, vl) ;
	  __vr vrgout_k3_s4 = _vel_vgtu_vvssml(vrgout_ptr_k3_s4, 0, 0, vmx_s4, vl) ;
	  VFMAD_C1(vrgout_k3_s0, vmx_s0, 3, r, 0) ;
	  VFMAD_C1(vrgout_k3_s1, vmx_s1, 3, r, 1) ;
	  VFMAD_C1(vrgout_k3_s2, vmx_s2, 3, r, 2) ;
	  VFMAD_C1(vrgout_k3_s3, vmx_s3, 3, r, 3) ;
	  VFMAD_C1(vrgout_k3_s4, vmx_s4, 3, r, 4) ;

	  k+=4 ;
	}
	for ( ; k<gOutChannelGroup; k+=8 ) {
	  int64_t gOutIndex    = gOutGroupOffset + ((n * gOutChannel + k) * gOutHeight) * gOutWidth ;
	  const float *pKerValue = pKernel + kernGroupOffset + ((k * gInChannelGroup + c) * kernHeight) * kernWidth;

	  __vr vrgout_ptr_k0_s0 = _vel_vsfa_vvssl(_vel_vaddsl_vsvl(gOutWidth*y, vrx_s0, vl),
					     2,
					     (unsigned long)(pGOut+gOutIndex), vl) ;
	  __vr vrgout_k0_s0 = _vel_vgtu_vvssml(vrgout_ptr_k0_s0, 0, 0, vmx_s0, vl) ;
	  __vr vrgout_ptr_k0_s1 = _vel_vsfa_vvssl(_vel_vaddsl_vsvl(gOutWidth*y, vrx_s1, vl),
					     2,
					     (unsigned long)(pGOut+gOutIndex), vl) ;
	  __vr vrgout_k0_s1 = _vel_vgtu_vvssml(vrgout_ptr_k0_s1, 0, 0, vmx_s1, vl) ;
	  __vr vrgout_ptr_k0_s2 = _vel_vsfa_vvssl(_vel_vaddsl_vsvl(gOutWidth*y, vrx_s2, vl),
					     2,
					     (unsigned long)(pGOut+gOutIndex), vl) ;
	  __vr vrgout_k0_s2 = _vel_vgtu_vvssml(vrgout_ptr_k0_s2, 0, 0, vmx_s2, vl) ;
	  __vr vrgout_ptr_k0_s3 = _vel_vsfa_vvssl(_vel_vaddsl_vsvl(gOutWidth*y, vrx_s3, vl),
					     2,
					     (unsigned long)(pGOut+gOutIndex), vl) ;
	  __vr vrgout_k0_s3 = _vel_vgtu_vvssml(vrgout_ptr_k0_s3, 0, 0, vmx_s3, vl) ;
	  __vr vrgout_ptr_k0_s4 = _vel_vsfa_vvssl(_vel_vaddsl_vsvl(gOutWidth*y, vrx_s4, vl),
					     2,
					     (unsigned long)(pGOut+gOutIndex), vl) ;
	  __vr vrgout_k0_s4 = _vel_vgtu_vvssml(vrgout_ptr_k0_s4, 0, 0, vmx_s4, vl) ;
	  VFMAD_C1(vrgout_k0_s0, vmx_s0, 0, r, 0) ;
	  VFMAD_C1(vrgout_k0_s1, vmx_s1, 0, r, 1) ;
	  VFMAD_C1(vrgout_k0_s2, vmx_s2, 0, r, 2) ;
	  VFMAD_C1(vrgout_k0_s3, vmx_s3, 0, r, 3) ;
	  VFMAD_C1(vrgout_k0_s4, vmx_s4, 0, r, 4) ;

	  __vr vrgout_ptr_k1_s0 = _vel_vaddsl_vsvl(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_s0, vl) ;
	  __vr vrgout_k1_s0 = _vel_vgtu_vvssml(vrgout_ptr_k1_s0, 0, 0, vmx_s0, vl) ;
	  __vr vrgout_ptr_k1_s1 = _vel_vaddsl_vsvl(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_s1, vl) ;
	  __vr vrgout_k1_s1 = _vel_vgtu_vvssml(vrgout_ptr_k1_s1, 0, 0, vmx_s1, vl) ;
	  __vr vrgout_ptr_k1_s2 = _vel_vaddsl_vsvl(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_s2, vl) ;
	  __vr vrgout_k1_s2 = _vel_vgtu_vvssml(vrgout_ptr_k1_s2, 0, 0, vmx_s2, vl) ;
	  __vr vrgout_ptr_k1_s3 = _vel_vaddsl_vsvl(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_s3, vl) ;
	  __vr vrgout_k1_s3 = _vel_vgtu_vvssml(vrgout_ptr_k1_s3, 0, 0, vmx_s3, vl) ;
	  __vr vrgout_ptr_k1_s4 = _vel_vaddsl_vsvl(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_s4, vl) ;
	  __vr vrgout_k1_s4 = _vel_vgtu_vvssml(vrgout_ptr_k1_s4, 0, 0, vmx_s4, vl) ;
	  VFMAD_C1(vrgout_k1_s0, vmx_s0, 1, r, 0) ;
	  VFMAD_C1(vrgout_k1_s1, vmx_s1, 1, r, 1) ;
	  VFMAD_C1(vrgout_k1_s2, vmx_s2, 1, r, 2) ;
	  VFMAD_C1(vrgout_k1_s3, vmx_s3, 1, r, 3) ;
	  VFMAD_C1(vrgout_k1_s4, vmx_s4, 1, r, 4) ;

	  __vr vrgout_ptr_k2_s0 = _vel_vaddsl_vsvl(4*2*gOutHeight*gOutWidth, vrgout_ptr_k0_s0, vl) ;
	  __vr vrgout_k2_s0 = _vel_vgtu_vvssml(vrgout_ptr_k2_s0, 0, 0, vmx_s0, vl) ;
	  __vr vrgout_ptr_k2_s1 = _vel_vaddsl_vsvl(4*2*gOutHeight*gOutWidth, vrgout_ptr_k0_s1, vl) ;
	  __vr vrgout_k2_s1 = _vel_vgtu_vvssml(vrgout_ptr_k2_s1, 0, 0, vmx_s1, vl) ;
	  __vr vrgout_ptr_k2_s2 = _vel_vaddsl_vsvl(4*2*gOutHeight*gOutWidth, vrgout_ptr_k0_s2, vl) ;
	  __vr vrgout_k2_s2 = _vel_vgtu_vvssml(vrgout_ptr_k2_s2, 0, 0, vmx_s2, vl) ;
	  __vr vrgout_ptr_k2_s3 = _vel_vaddsl_vsvl(4*2*gOutHeight*gOutWidth, vrgout_ptr_k0_s3, vl) ;
	  __vr vrgout_k2_s3 = _vel_vgtu_vvssml(vrgout_ptr_k2_s3, 0, 0, vmx_s3, vl) ;
	  __vr vrgout_ptr_k2_s4 = _vel_vaddsl_vsvl(4*2*gOutHeight*gOutWidth, vrgout_ptr_k0_s4, vl) ;
	  __vr vrgout_k2_s4 = _vel_vgtu_vvssml(vrgout_ptr_k2_s4, 0, 0, vmx_s4, vl) ;
	  VFMAD_C1(vrgout_k2_s0, vmx_s0, 2, r, 0) ;
	  VFMAD_C1(vrgout_k2_s1, vmx_s1, 2, r, 1) ;
	  VFMAD_C1(vrgout_k2_s2, vmx_s2, 2, r, 2) ;
	  VFMAD_C1(vrgout_k2_s3, vmx_s3, 2, r, 3) ;
	  VFMAD_C1(vrgout_k2_s4, vmx_s4, 2, r, 4) ;

	  __vr vrgout_ptr_k3_s0 = _vel_vaddsl_vsvl(4*3*gOutHeight*gOutWidth, vrgout_ptr_k0_s0, vl) ;
	  __vr vrgout_k3_s0 = _vel_vgtu_vvssml(vrgout_ptr_k3_s0, 0, 0, vmx_s0, vl) ;
	  __vr vrgout_ptr_k3_s1 = _vel_vaddsl_vsvl(4*3*gOutHeight*gOutWidth, vrgout_ptr_k0_s1, vl) ;
	  __vr vrgout_k3_s1 = _vel_vgtu_vvssml(vrgout_ptr_k3_s1, 0, 0, vmx_s1, vl) ;
	  __vr vrgout_ptr_k3_s2 = _vel_vaddsl_vsvl(4*3*gOutHeight*gOutWidth, vrgout_ptr_k0_s2, vl) ;
	  __vr vrgout_k3_s2 = _vel_vgtu_vvssml(vrgout_ptr_k3_s2, 0, 0, vmx_s2, vl) ;
	  __vr vrgout_ptr_k3_s3 = _vel_vaddsl_vsvl(4*3*gOutHeight*gOutWidth, vrgout_ptr_k0_s3, vl) ;
	  __vr vrgout_k3_s3 = _vel_vgtu_vvssml(vrgout_ptr_k3_s3, 0, 0, vmx_s3, vl) ;
	  __vr vrgout_ptr_k3_s4 = _vel_vaddsl_vsvl(4*3*gOutHeight*gOutWidth, vrgout_ptr_k0_s4, vl) ;
	  __vr vrgout_k3_s4 = _vel_vgtu_vvssml(vrgout_ptr_k3_s4, 0, 0, vmx_s4, vl) ;
	  VFMAD_C1(vrgout_k3_s0, vmx_s0, 3, r, 0) ;
	  VFMAD_C1(vrgout_k3_s1, vmx_s1, 3, r, 1) ;
	  VFMAD_C1(vrgout_k3_s2, vmx_s2, 3, r, 2) ;
	  VFMAD_C1(vrgout_k3_s3, vmx_s3, 3, r, 3) ;
	  VFMAD_C1(vrgout_k3_s4, vmx_s4, 3, r, 4) ;

	  __vr vrgout_ptr_k4_s0 = _vel_vaddsl_vsvl(4*4*gOutHeight*gOutWidth, vrgout_ptr_k0_s0, vl) ;
	  __vr vrgout_k4_s0 = _vel_vgtu_vvssml(vrgout_ptr_k4_s0, 0, 0, vmx_s0, vl) ;
	  __vr vrgout_ptr_k4_s1 = _vel_vaddsl_vsvl(4*4*gOutHeight*gOutWidth, vrgout_ptr_k0_s1, vl) ;
	  __vr vrgout_k4_s1 = _vel_vgtu_vvssml(vrgout_ptr_k4_s1, 0, 0, vmx_s1, vl) ;
	  __vr vrgout_ptr_k4_s2 = _vel_vaddsl_vsvl(4*4*gOutHeight*gOutWidth, vrgout_ptr_k0_s2, vl) ;
	  __vr vrgout_k4_s2 = _vel_vgtu_vvssml(vrgout_ptr_k4_s2, 0, 0, vmx_s2, vl) ;
	  __vr vrgout_ptr_k4_s3 = _vel_vaddsl_vsvl(4*4*gOutHeight*gOutWidth, vrgout_ptr_k0_s3, vl) ;
	  __vr vrgout_k4_s3 = _vel_vgtu_vvssml(vrgout_ptr_k4_s3, 0, 0, vmx_s3, vl) ;
	  __vr vrgout_ptr_k4_s4 = _vel_vaddsl_vsvl(4*4*gOutHeight*gOutWidth, vrgout_ptr_k0_s4, vl) ;
	  __vr vrgout_k4_s4 = _vel_vgtu_vvssml(vrgout_ptr_k4_s4, 0, 0, vmx_s4, vl) ;
	  VFMAD_C1(vrgout_k4_s0, vmx_s0, 4, r, 0) ;
	  VFMAD_C1(vrgout_k4_s1, vmx_s1, 4, r, 1) ;
	  VFMAD_C1(vrgout_k4_s2, vmx_s2, 4, r, 2) ;
	  VFMAD_C1(vrgout_k4_s3, vmx_s3, 4, r, 3) ;
	  VFMAD_C1(vrgout_k4_s4, vmx_s4, 4, r, 4) ;

	  __vr vrgout_ptr_k5_s0 = _vel_vaddsl_vsvl(4*5*gOutHeight*gOutWidth, vrgout_ptr_k0_s0, vl) ;
	  __vr vrgout_k5_s0 = _vel_vgtu_vvssml(vrgout_ptr_k5_s0, 0, 0, vmx_s0, vl) ;
	  __vr vrgout_ptr_k5_s1 = _vel_vaddsl_vsvl(4*5*gOutHeight*gOutWidth, vrgout_ptr_k0_s1, vl) ;
	  __vr vrgout_k5_s1 = _vel_vgtu_vvssml(vrgout_ptr_k5_s1, 0, 0, vmx_s1, vl) ;
	  __vr vrgout_ptr_k5_s2 = _vel_vaddsl_vsvl(4*5*gOutHeight*gOutWidth, vrgout_ptr_k0_s2, vl) ;
	  __vr vrgout_k5_s2 = _vel_vgtu_vvssml(vrgout_ptr_k5_s2, 0, 0, vmx_s2, vl) ;
	  __vr vrgout_ptr_k5_s3 = _vel_vaddsl_vsvl(4*5*gOutHeight*gOutWidth, vrgout_ptr_k0_s3, vl) ;
	  __vr vrgout_k5_s3 = _vel_vgtu_vvssml(vrgout_ptr_k5_s3, 0, 0, vmx_s3, vl) ;
	  __vr vrgout_ptr_k5_s4 = _vel_vaddsl_vsvl(4*5*gOutHeight*gOutWidth, vrgout_ptr_k0_s4, vl) ;
	  __vr vrgout_k5_s4 = _vel_vgtu_vvssml(vrgout_ptr_k5_s4, 0, 0, vmx_s4, vl) ;
	  VFMAD_C1(vrgout_k5_s0, vmx_s0, 5, r, 0) ;
	  VFMAD_C1(vrgout_k5_s1, vmx_s1, 5, r, 1) ;
	  VFMAD_C1(vrgout_k5_s2, vmx_s2, 5, r, 2) ;
	  VFMAD_C1(vrgout_k5_s3, vmx_s3, 5, r, 3) ;
	  VFMAD_C1(vrgout_k5_s4, vmx_s4, 5, r, 4) ;

	  __vr vrgout_ptr_k6_s0 = _vel_vaddsl_vsvl(4*6*gOutHeight*gOutWidth, vrgout_ptr_k0_s0, vl) ;
	  __vr vrgout_k6_s0 = _vel_vgtu_vvssml(vrgout_ptr_k6_s0, 0, 0, vmx_s0, vl) ;
	  __vr vrgout_ptr_k6_s1 = _vel_vaddsl_vsvl(4*6*gOutHeight*gOutWidth, vrgout_ptr_k0_s1, vl) ;
	  __vr vrgout_k6_s1 = _vel_vgtu_vvssml(vrgout_ptr_k6_s1, 0, 0, vmx_s1, vl) ;
	  __vr vrgout_ptr_k6_s2 = _vel_vaddsl_vsvl(4*6*gOutHeight*gOutWidth, vrgout_ptr_k0_s2, vl) ;
	  __vr vrgout_k6_s2 = _vel_vgtu_vvssml(vrgout_ptr_k6_s2, 0, 0, vmx_s2, vl) ;
	  __vr vrgout_ptr_k6_s3 = _vel_vaddsl_vsvl(4*6*gOutHeight*gOutWidth, vrgout_ptr_k0_s3, vl) ;
	  __vr vrgout_k6_s3 = _vel_vgtu_vvssml(vrgout_ptr_k6_s3, 0, 0, vmx_s3, vl) ;
	  __vr vrgout_ptr_k6_s4 = _vel_vaddsl_vsvl(4*6*gOutHeight*gOutWidth, vrgout_ptr_k0_s4, vl) ;
	  __vr vrgout_k6_s4 = _vel_vgtu_vvssml(vrgout_ptr_k6_s4, 0, 0, vmx_s4, vl) ;
	  VFMAD_C1(vrgout_k6_s0, vmx_s0, 6, r, 0) ;
	  VFMAD_C1(vrgout_k6_s1, vmx_s1, 6, r, 1) ;
	  VFMAD_C1(vrgout_k6_s2, vmx_s2, 6, r, 2) ;
	  VFMAD_C1(vrgout_k6_s3, vmx_s3, 6, r, 3) ;
	  VFMAD_C1(vrgout_k6_s4, vmx_s4, 6, r, 4) ;

	  __vr vrgout_ptr_k7_s0 = _vel_vaddsl_vsvl(4*7*gOutHeight*gOutWidth, vrgout_ptr_k0_s0, vl) ;
	  __vr vrgout_k7_s0 = _vel_vgtu_vvssml(vrgout_ptr_k7_s0, 0, 0, vmx_s0, vl) ;
	  __vr vrgout_ptr_k7_s1 = _vel_vaddsl_vsvl(4*7*gOutHeight*gOutWidth, vrgout_ptr_k0_s1, vl) ;
	  __vr vrgout_k7_s1 = _vel_vgtu_vvssml(vrgout_ptr_k7_s1, 0, 0, vmx_s1, vl) ;
	  __vr vrgout_ptr_k7_s2 = _vel_vaddsl_vsvl(4*7*gOutHeight*gOutWidth, vrgout_ptr_k0_s2, vl) ;
	  __vr vrgout_k7_s2 = _vel_vgtu_vvssml(vrgout_ptr_k7_s2, 0, 0, vmx_s2, vl) ;
	  __vr vrgout_ptr_k7_s3 = _vel_vaddsl_vsvl(4*7*gOutHeight*gOutWidth, vrgout_ptr_k0_s3, vl) ;
	  __vr vrgout_k7_s3 = _vel_vgtu_vvssml(vrgout_ptr_k7_s3, 0, 0, vmx_s3, vl) ;
	  __vr vrgout_ptr_k7_s4 = _vel_vaddsl_vsvl(4*7*gOutHeight*gOutWidth, vrgout_ptr_k0_s4, vl) ;
	  __vr vrgout_k7_s4 = _vel_vgtu_vvssml(vrgout_ptr_k7_s4, 0, 0, vmx_s4, vl) ;
	  VFMAD_C1(vrgout_k7_s0, vmx_s0, 7, r, 0) ;
	  VFMAD_C1(vrgout_k7_s1, vmx_s1, 7, r, 1) ;
	  VFMAD_C1(vrgout_k7_s2, vmx_s2, 7, r, 2) ;
	  VFMAD_C1(vrgout_k7_s3, vmx_s3, 7, r, 3) ;
	  VFMAD_C1(vrgout_k7_s4, vmx_s4, 7, r, 4) ;

#undef VFMAD_C1
	} // gOutChannel

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

      __vr vrj_s0 = _vel_vaddsl_vsvl(padWidth-0*dilationWidth, vrw, vl) ;
      __vr vrj_s1 = _vel_vaddsl_vsvl(padWidth-1*dilationWidth, vrw, vl) ;
      __vr vrj_s2 = _vel_vaddsl_vsvl(padWidth-2*dilationWidth, vrw, vl) ;
      __vr vrj_s3 = _vel_vaddsl_vsvl(padWidth-3*dilationWidth, vrw, vl) ;
      __vr vrj_s4 = _vel_vaddsl_vsvl(padWidth-4*dilationWidth, vrw, vl) ;

      __vr vrx_s0 = _vel_vdivsl_vvsl(vrj_s0, strideWidth, vl) ;
      __vr vrx_s1 = _vel_vdivsl_vvsl(vrj_s1, strideWidth, vl) ;
      __vr vrx_s2 = _vel_vdivsl_vvsl(vrj_s2, strideWidth, vl) ;
      __vr vrx_s3 = _vel_vdivsl_vvsl(vrj_s3, strideWidth, vl) ;
      __vr vrx_s4 = _vel_vdivsl_vvsl(vrj_s4, strideWidth, vl) ;

      __vm256 vmx0_s0 =  _vel_vfmkleq_mvl(_vel_vcmpsl_vvvl(vrj_s0, _vel_vmulsl_vsvl(strideWidth, vrx_s0, vl), vl), vl) ;
      __vm256 vmx1_s0 =  _vel_vfmklge_mvl(vrx_s0, vl) ;
      __vm256 vmx2_s0 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutWidth,vrx_s0, vl), vl) ;
      __vm256 vmx_s0 = _vel_andm_mmm(_vel_andm_mmm(vmx0_s0, vmx1_s0), vmx2_s0) ;

      __vm256 vmx0_s1 =  _vel_vfmkleq_mvl(_vel_vcmpsl_vvvl(vrj_s1, _vel_vmulsl_vsvl(strideWidth, vrx_s1, vl), vl), vl) ;
      __vm256 vmx1_s1 =  _vel_vfmklge_mvl(vrx_s1, vl) ;
      __vm256 vmx2_s1 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutWidth,vrx_s1, vl), vl) ;
      __vm256 vmx_s1 = _vel_andm_mmm(_vel_andm_mmm(vmx0_s1, vmx1_s1), vmx2_s1) ;

      __vm256 vmx0_s2 =  _vel_vfmkleq_mvl(_vel_vcmpsl_vvvl(vrj_s2, _vel_vmulsl_vsvl(strideWidth, vrx_s2, vl), vl), vl) ;
      __vm256 vmx1_s2 =  _vel_vfmklge_mvl(vrx_s2, vl) ;
      __vm256 vmx2_s2 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutWidth,vrx_s2, vl), vl) ;
      __vm256 vmx_s2 = _vel_andm_mmm(_vel_andm_mmm(vmx0_s2, vmx1_s2), vmx2_s2) ;

      __vm256 vmx0_s3 =  _vel_vfmkleq_mvl(_vel_vcmpsl_vvvl(vrj_s3, _vel_vmulsl_vsvl(strideWidth, vrx_s3, vl), vl), vl) ;
      __vm256 vmx1_s3 =  _vel_vfmklge_mvl(vrx_s3, vl) ;
      __vm256 vmx2_s3 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutWidth,vrx_s3, vl), vl) ;
      __vm256 vmx_s3 = _vel_andm_mmm(_vel_andm_mmm(vmx0_s3, vmx1_s3), vmx2_s3) ;

      __vm256 vmx0_s4 =  _vel_vfmkleq_mvl(_vel_vcmpsl_vvvl(vrj_s4, _vel_vmulsl_vsvl(strideWidth, vrx_s4, vl), vl), vl) ;
      __vm256 vmx1_s4 =  _vel_vfmklge_mvl(vrx_s4, vl) ;
      __vm256 vmx2_s4 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutWidth,vrx_s4, vl), vl) ;
      __vm256 vmx_s4 = _vel_andm_mmm(_vel_andm_mmm(vmx0_s4, vmx1_s4), vmx2_s4) ;

      for (int64_t r=0; r<kernHeight; r++) {
	int64_t i = h - r * dilationHeight + padHeight ;
	int64_t y = i/strideHeight;
	if ( y*strideHeight != i || y < 0 || gOutHeight <= y)  continue ;

	int64_t k=0;
	if( (gOutChannelGroup & 0x01)  == 1 ) {
	  int64_t gOutIndex    = gOutGroupOffset + ((n * gOutChannel + k) * gOutHeight) * gOutWidth ;
	  const float *pKerValue = pKernel + kernGroupOffset + ((k * gInChannelGroup + c) * kernHeight) * kernWidth;

	  __vr vrgout_ptr_k0_s0 = _vel_vsfa_vvssl(_vel_vaddsl_vsvl(gOutWidth*y, vrx_s0, vl),
					     2,
					     (unsigned long)(pGOut+gOutIndex), vl) ;
	  __vr vrgout_k0_s0 = _vel_vgtu_vvssml(vrgout_ptr_k0_s0, 0, 0, vmx_s0, vl) ;
	  __vr vrgout_ptr_k0_s1 = _vel_vsfa_vvssl(_vel_vaddsl_vsvl(gOutWidth*y, vrx_s1, vl),
					     2,
					     (unsigned long)(pGOut+gOutIndex), vl) ;
	  __vr vrgout_k0_s1 = _vel_vgtu_vvssml(vrgout_ptr_k0_s1, 0, 0, vmx_s1, vl) ;
	  __vr vrgout_ptr_k0_s2 = _vel_vsfa_vvssl(_vel_vaddsl_vsvl(gOutWidth*y, vrx_s2, vl),
					     2,
					     (unsigned long)(pGOut+gOutIndex), vl) ;
	  __vr vrgout_k0_s2 = _vel_vgtu_vvssml(vrgout_ptr_k0_s2, 0, 0, vmx_s2, vl) ;
	  __vr vrgout_ptr_k0_s3 = _vel_vsfa_vvssl(_vel_vaddsl_vsvl(gOutWidth*y, vrx_s3, vl),
					     2,
					     (unsigned long)(pGOut+gOutIndex), vl) ;
	  __vr vrgout_k0_s3 = _vel_vgtu_vvssml(vrgout_ptr_k0_s3, 0, 0, vmx_s3, vl) ;
	  __vr vrgout_ptr_k0_s4 = _vel_vsfa_vvssl(_vel_vaddsl_vsvl(gOutWidth*y, vrx_s4, vl),
					     2,
					     (unsigned long)(pGOut+gOutIndex), vl) ;
	  __vr vrgout_k0_s4 = _vel_vgtu_vvssml(vrgout_ptr_k0_s4, 0, 0, vmx_s4, vl) ;

#define VFMAD_C2(VRGOUT, VM, K, R, S)	{												\
	    const uint64_t kerValue01 = _vel_pack_f32p(pKerValue + (((K)*gInChannelGroup + 0) * kernHeight +(R)) * kernWidth + (S),	\
						       pKerValue + (((K)*gInChannelGroup + 1) * kernHeight +(R)) * kernWidth + (S)) ;	\
	    VRGOUT = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), VRGOUT, VM, vl) ;	\
	    __vr vrgoutP = _vel_vshf_vvvsl(VRGOUT, VRGOUT, VE_VSHUFFLE_YUZU, vl) ;	\
	    vrsum01 = _vel_pvfmad_vvsvl(vrsum01, kerValue01, vrgoutP, vl) ;		\
}
	  VFMAD_C2(vrgout_k0_s0, vmx_s0, 0, r, 0) ;
	  VFMAD_C2(vrgout_k0_s1, vmx_s1, 0, r, 1) ;
	  VFMAD_C2(vrgout_k0_s2, vmx_s2, 0, r, 2) ;
	  VFMAD_C2(vrgout_k0_s3, vmx_s3, 0, r, 3) ;
	  VFMAD_C2(vrgout_k0_s4, vmx_s4, 0, r, 4) ;

	  k+=1 ;
	}
	if( ( (gOutChannelGroup >> 1) & 0x01)  == 1 ) {
	  int64_t gOutIndex    = gOutGroupOffset + ((n * gOutChannel + k) * gOutHeight) * gOutWidth ;
	  const float *pKerValue = pKernel + kernGroupOffset + ((k * gInChannelGroup + c) * kernHeight) * kernWidth;

	  __vr vrgout_ptr_k0_s0 = _vel_vsfa_vvssl(_vel_vaddsl_vsvl(gOutWidth*y, vrx_s0, vl),
					     2,
					     (unsigned long)(pGOut+gOutIndex), vl) ;
	  __vr vrgout_k0_s0 = _vel_vgtu_vvssml(vrgout_ptr_k0_s0, 0, 0, vmx_s0, vl) ;
	  __vr vrgout_ptr_k0_s1 = _vel_vsfa_vvssl(_vel_vaddsl_vsvl(gOutWidth*y, vrx_s1, vl),
					     2,
					     (unsigned long)(pGOut+gOutIndex), vl) ;
	  __vr vrgout_k0_s1 = _vel_vgtu_vvssml(vrgout_ptr_k0_s1, 0, 0, vmx_s1, vl) ;
	  __vr vrgout_ptr_k0_s2 = _vel_vsfa_vvssl(_vel_vaddsl_vsvl(gOutWidth*y, vrx_s2, vl),
					     2,
					     (unsigned long)(pGOut+gOutIndex), vl) ;
	  __vr vrgout_k0_s2 = _vel_vgtu_vvssml(vrgout_ptr_k0_s2, 0, 0, vmx_s2, vl) ;
	  __vr vrgout_ptr_k0_s3 = _vel_vsfa_vvssl(_vel_vaddsl_vsvl(gOutWidth*y, vrx_s3, vl),
					     2,
					     (unsigned long)(pGOut+gOutIndex), vl) ;
	  __vr vrgout_k0_s3 = _vel_vgtu_vvssml(vrgout_ptr_k0_s3, 0, 0, vmx_s3, vl) ;
	  __vr vrgout_ptr_k0_s4 = _vel_vsfa_vvssl(_vel_vaddsl_vsvl(gOutWidth*y, vrx_s4, vl),
					     2,
					     (unsigned long)(pGOut+gOutIndex), vl) ;
	  __vr vrgout_k0_s4 = _vel_vgtu_vvssml(vrgout_ptr_k0_s4, 0, 0, vmx_s4, vl) ;
	  VFMAD_C2(vrgout_k0_s0, vmx_s0, 0, r, 0) ;
	  VFMAD_C2(vrgout_k0_s1, vmx_s1, 0, r, 1) ;
	  VFMAD_C2(vrgout_k0_s2, vmx_s2, 0, r, 2) ;
	  VFMAD_C2(vrgout_k0_s3, vmx_s3, 0, r, 3) ;
	  VFMAD_C2(vrgout_k0_s4, vmx_s4, 0, r, 4) ;

	  __vr vrgout_ptr_k1_s0 = _vel_vaddsl_vsvl(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_s0, vl) ;
	  __vr vrgout_k1_s0 = _vel_vgtu_vvssml(vrgout_ptr_k1_s0, 0, 0, vmx_s0, vl) ;
	  __vr vrgout_ptr_k1_s1 = _vel_vaddsl_vsvl(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_s1, vl) ;
	  __vr vrgout_k1_s1 = _vel_vgtu_vvssml(vrgout_ptr_k1_s1, 0, 0, vmx_s1, vl) ;
	  __vr vrgout_ptr_k1_s2 = _vel_vaddsl_vsvl(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_s2, vl) ;
	  __vr vrgout_k1_s2 = _vel_vgtu_vvssml(vrgout_ptr_k1_s2, 0, 0, vmx_s2, vl) ;
	  __vr vrgout_ptr_k1_s3 = _vel_vaddsl_vsvl(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_s3, vl) ;
	  __vr vrgout_k1_s3 = _vel_vgtu_vvssml(vrgout_ptr_k1_s3, 0, 0, vmx_s3, vl) ;
	  __vr vrgout_ptr_k1_s4 = _vel_vaddsl_vsvl(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_s4, vl) ;
	  __vr vrgout_k1_s4 = _vel_vgtu_vvssml(vrgout_ptr_k1_s4, 0, 0, vmx_s4, vl) ;
	  VFMAD_C2(vrgout_k1_s0, vmx_s0, 1, r, 0) ;
	  VFMAD_C2(vrgout_k1_s1, vmx_s1, 1, r, 1) ;
	  VFMAD_C2(vrgout_k1_s2, vmx_s2, 1, r, 2) ;
	  VFMAD_C2(vrgout_k1_s3, vmx_s3, 1, r, 3) ;
	  VFMAD_C2(vrgout_k1_s4, vmx_s4, 1, r, 4) ;

	  k+=2 ;
	}
	if( ( (gOutChannelGroup >> 2) & 0x01)  == 1 ) {
	  int64_t gOutIndex    = gOutGroupOffset + ((n * gOutChannel + k) * gOutHeight) * gOutWidth ;
	  const float *pKerValue = pKernel + kernGroupOffset + ((k * gInChannelGroup + c) * kernHeight) * kernWidth;

	  __vr vrgout_ptr_k0_s0 = _vel_vsfa_vvssl(_vel_vaddsl_vsvl(gOutWidth*y, vrx_s0, vl),
					     2,
					     (unsigned long)(pGOut+gOutIndex), vl) ;
	  __vr vrgout_k0_s0 = _vel_vgtu_vvssml(vrgout_ptr_k0_s0, 0, 0, vmx_s0, vl) ;
	  __vr vrgout_ptr_k0_s1 = _vel_vsfa_vvssl(_vel_vaddsl_vsvl(gOutWidth*y, vrx_s1, vl),
					     2,
					     (unsigned long)(pGOut+gOutIndex), vl) ;
	  __vr vrgout_k0_s1 = _vel_vgtu_vvssml(vrgout_ptr_k0_s1, 0, 0, vmx_s1, vl) ;
	  __vr vrgout_ptr_k0_s2 = _vel_vsfa_vvssl(_vel_vaddsl_vsvl(gOutWidth*y, vrx_s2, vl),
					     2,
					     (unsigned long)(pGOut+gOutIndex), vl) ;
	  __vr vrgout_k0_s2 = _vel_vgtu_vvssml(vrgout_ptr_k0_s2, 0, 0, vmx_s2, vl) ;
	  __vr vrgout_ptr_k0_s3 = _vel_vsfa_vvssl(_vel_vaddsl_vsvl(gOutWidth*y, vrx_s3, vl),
					     2,
					     (unsigned long)(pGOut+gOutIndex), vl) ;
	  __vr vrgout_k0_s3 = _vel_vgtu_vvssml(vrgout_ptr_k0_s3, 0, 0, vmx_s3, vl) ;
	  __vr vrgout_ptr_k0_s4 = _vel_vsfa_vvssl(_vel_vaddsl_vsvl(gOutWidth*y, vrx_s4, vl),
					     2,
					     (unsigned long)(pGOut+gOutIndex), vl) ;
	  __vr vrgout_k0_s4 = _vel_vgtu_vvssml(vrgout_ptr_k0_s4, 0, 0, vmx_s4, vl) ;
	  VFMAD_C2(vrgout_k0_s0, vmx_s0, 0, r, 0) ;
	  VFMAD_C2(vrgout_k0_s1, vmx_s1, 0, r, 1) ;
	  VFMAD_C2(vrgout_k0_s2, vmx_s2, 0, r, 2) ;
	  VFMAD_C2(vrgout_k0_s3, vmx_s3, 0, r, 3) ;
	  VFMAD_C2(vrgout_k0_s4, vmx_s4, 0, r, 4) ;

	  __vr vrgout_ptr_k1_s0 = _vel_vaddsl_vsvl(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_s0, vl) ;
	  __vr vrgout_k1_s0 = _vel_vgtu_vvssml(vrgout_ptr_k1_s0, 0, 0, vmx_s0, vl) ;
	  __vr vrgout_ptr_k1_s1 = _vel_vaddsl_vsvl(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_s1, vl) ;
	  __vr vrgout_k1_s1 = _vel_vgtu_vvssml(vrgout_ptr_k1_s1, 0, 0, vmx_s1, vl) ;
	  __vr vrgout_ptr_k1_s2 = _vel_vaddsl_vsvl(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_s2, vl) ;
	  __vr vrgout_k1_s2 = _vel_vgtu_vvssml(vrgout_ptr_k1_s2, 0, 0, vmx_s2, vl) ;
	  __vr vrgout_ptr_k1_s3 = _vel_vaddsl_vsvl(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_s3, vl) ;
	  __vr vrgout_k1_s3 = _vel_vgtu_vvssml(vrgout_ptr_k1_s3, 0, 0, vmx_s3, vl) ;
	  __vr vrgout_ptr_k1_s4 = _vel_vaddsl_vsvl(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_s4, vl) ;
	  __vr vrgout_k1_s4 = _vel_vgtu_vvssml(vrgout_ptr_k1_s4, 0, 0, vmx_s4, vl) ;
	  VFMAD_C2(vrgout_k1_s0, vmx_s0, 1, r, 0) ;
	  VFMAD_C2(vrgout_k1_s1, vmx_s1, 1, r, 1) ;
	  VFMAD_C2(vrgout_k1_s2, vmx_s2, 1, r, 2) ;
	  VFMAD_C2(vrgout_k1_s3, vmx_s3, 1, r, 3) ;
	  VFMAD_C2(vrgout_k1_s4, vmx_s4, 1, r, 4) ;

	  __vr vrgout_ptr_k2_s0 = _vel_vaddsl_vsvl(4*2*gOutHeight*gOutWidth, vrgout_ptr_k0_s0, vl) ;
	  __vr vrgout_k2_s0 = _vel_vgtu_vvssml(vrgout_ptr_k2_s0, 0, 0, vmx_s0, vl) ;
	  __vr vrgout_ptr_k2_s1 = _vel_vaddsl_vsvl(4*2*gOutHeight*gOutWidth, vrgout_ptr_k0_s1, vl) ;
	  __vr vrgout_k2_s1 = _vel_vgtu_vvssml(vrgout_ptr_k2_s1, 0, 0, vmx_s1, vl) ;
	  __vr vrgout_ptr_k2_s2 = _vel_vaddsl_vsvl(4*2*gOutHeight*gOutWidth, vrgout_ptr_k0_s2, vl) ;
	  __vr vrgout_k2_s2 = _vel_vgtu_vvssml(vrgout_ptr_k2_s2, 0, 0, vmx_s2, vl) ;
	  __vr vrgout_ptr_k2_s3 = _vel_vaddsl_vsvl(4*2*gOutHeight*gOutWidth, vrgout_ptr_k0_s3, vl) ;
	  __vr vrgout_k2_s3 = _vel_vgtu_vvssml(vrgout_ptr_k2_s3, 0, 0, vmx_s3, vl) ;
	  __vr vrgout_ptr_k2_s4 = _vel_vaddsl_vsvl(4*2*gOutHeight*gOutWidth, vrgout_ptr_k0_s4, vl) ;
	  __vr vrgout_k2_s4 = _vel_vgtu_vvssml(vrgout_ptr_k2_s4, 0, 0, vmx_s4, vl) ;
	  VFMAD_C2(vrgout_k2_s0, vmx_s0, 2, r, 0) ;
	  VFMAD_C2(vrgout_k2_s1, vmx_s1, 2, r, 1) ;
	  VFMAD_C2(vrgout_k2_s2, vmx_s2, 2, r, 2) ;
	  VFMAD_C2(vrgout_k2_s3, vmx_s3, 2, r, 3) ;
	  VFMAD_C2(vrgout_k2_s4, vmx_s4, 2, r, 4) ;

	  __vr vrgout_ptr_k3_s0 = _vel_vaddsl_vsvl(4*3*gOutHeight*gOutWidth, vrgout_ptr_k0_s0, vl) ;
	  __vr vrgout_k3_s0 = _vel_vgtu_vvssml(vrgout_ptr_k3_s0, 0, 0, vmx_s0, vl) ;
	  __vr vrgout_ptr_k3_s1 = _vel_vaddsl_vsvl(4*3*gOutHeight*gOutWidth, vrgout_ptr_k0_s1, vl) ;
	  __vr vrgout_k3_s1 = _vel_vgtu_vvssml(vrgout_ptr_k3_s1, 0, 0, vmx_s1, vl) ;
	  __vr vrgout_ptr_k3_s2 = _vel_vaddsl_vsvl(4*3*gOutHeight*gOutWidth, vrgout_ptr_k0_s2, vl) ;
	  __vr vrgout_k3_s2 = _vel_vgtu_vvssml(vrgout_ptr_k3_s2, 0, 0, vmx_s2, vl) ;
	  __vr vrgout_ptr_k3_s3 = _vel_vaddsl_vsvl(4*3*gOutHeight*gOutWidth, vrgout_ptr_k0_s3, vl) ;
	  __vr vrgout_k3_s3 = _vel_vgtu_vvssml(vrgout_ptr_k3_s3, 0, 0, vmx_s3, vl) ;
	  __vr vrgout_ptr_k3_s4 = _vel_vaddsl_vsvl(4*3*gOutHeight*gOutWidth, vrgout_ptr_k0_s4, vl) ;
	  __vr vrgout_k3_s4 = _vel_vgtu_vvssml(vrgout_ptr_k3_s4, 0, 0, vmx_s4, vl) ;
	  VFMAD_C2(vrgout_k3_s0, vmx_s0, 3, r, 0) ;
	  VFMAD_C2(vrgout_k3_s1, vmx_s1, 3, r, 1) ;
	  VFMAD_C2(vrgout_k3_s2, vmx_s2, 3, r, 2) ;
	  VFMAD_C2(vrgout_k3_s3, vmx_s3, 3, r, 3) ;
	  VFMAD_C2(vrgout_k3_s4, vmx_s4, 3, r, 4) ;

	  k+=4 ;
	}
	for ( ; k<gOutChannelGroup; k+=8 ) {
	  int64_t gOutIndex    = gOutGroupOffset + ((n * gOutChannel + k) * gOutHeight) * gOutWidth ;
	  const float *pKerValue = pKernel + kernGroupOffset + ((k * gInChannelGroup + c) * kernHeight) * kernWidth;

	  __vr vrgout_ptr_k0_s0 = _vel_vsfa_vvssl(_vel_vaddsl_vsvl(gOutWidth*y, vrx_s0, vl),
					     2,
					     (unsigned long)(pGOut+gOutIndex), vl) ;
	  __vr vrgout_k0_s0 = _vel_vgtu_vvssml(vrgout_ptr_k0_s0, 0, 0, vmx_s0, vl) ;
	  __vr vrgout_ptr_k0_s1 = _vel_vsfa_vvssl(_vel_vaddsl_vsvl(gOutWidth*y, vrx_s1, vl),
					     2,
					     (unsigned long)(pGOut+gOutIndex), vl) ;
	  __vr vrgout_k0_s1 = _vel_vgtu_vvssml(vrgout_ptr_k0_s1, 0, 0, vmx_s1, vl) ;
	  __vr vrgout_ptr_k0_s2 = _vel_vsfa_vvssl(_vel_vaddsl_vsvl(gOutWidth*y, vrx_s2, vl),
					     2,
					     (unsigned long)(pGOut+gOutIndex), vl) ;
	  __vr vrgout_k0_s2 = _vel_vgtu_vvssml(vrgout_ptr_k0_s2, 0, 0, vmx_s2, vl) ;
	  __vr vrgout_ptr_k0_s3 = _vel_vsfa_vvssl(_vel_vaddsl_vsvl(gOutWidth*y, vrx_s3, vl),
					     2,
					     (unsigned long)(pGOut+gOutIndex), vl) ;
	  __vr vrgout_k0_s3 = _vel_vgtu_vvssml(vrgout_ptr_k0_s3, 0, 0, vmx_s3, vl) ;
	  __vr vrgout_ptr_k0_s4 = _vel_vsfa_vvssl(_vel_vaddsl_vsvl(gOutWidth*y, vrx_s4, vl),
					     2,
					     (unsigned long)(pGOut+gOutIndex), vl) ;
	  __vr vrgout_k0_s4 = _vel_vgtu_vvssml(vrgout_ptr_k0_s4, 0, 0, vmx_s4, vl) ;
	  VFMAD_C2(vrgout_k0_s0, vmx_s0, 0, r, 0) ;
	  VFMAD_C2(vrgout_k0_s1, vmx_s1, 0, r, 1) ;
	  VFMAD_C2(vrgout_k0_s2, vmx_s2, 0, r, 2) ;
	  VFMAD_C2(vrgout_k0_s3, vmx_s3, 0, r, 3) ;
	  VFMAD_C2(vrgout_k0_s4, vmx_s4, 0, r, 4) ;

	  __vr vrgout_ptr_k1_s0 = _vel_vaddsl_vsvl(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_s0, vl) ;
	  __vr vrgout_k1_s0 = _vel_vgtu_vvssml(vrgout_ptr_k1_s0, 0, 0, vmx_s0, vl) ;
	  __vr vrgout_ptr_k1_s1 = _vel_vaddsl_vsvl(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_s1, vl) ;
	  __vr vrgout_k1_s1 = _vel_vgtu_vvssml(vrgout_ptr_k1_s1, 0, 0, vmx_s1, vl) ;
	  __vr vrgout_ptr_k1_s2 = _vel_vaddsl_vsvl(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_s2, vl) ;
	  __vr vrgout_k1_s2 = _vel_vgtu_vvssml(vrgout_ptr_k1_s2, 0, 0, vmx_s2, vl) ;
	  __vr vrgout_ptr_k1_s3 = _vel_vaddsl_vsvl(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_s3, vl) ;
	  __vr vrgout_k1_s3 = _vel_vgtu_vvssml(vrgout_ptr_k1_s3, 0, 0, vmx_s3, vl) ;
	  __vr vrgout_ptr_k1_s4 = _vel_vaddsl_vsvl(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_s4, vl) ;
	  __vr vrgout_k1_s4 = _vel_vgtu_vvssml(vrgout_ptr_k1_s4, 0, 0, vmx_s4, vl) ;
	  VFMAD_C2(vrgout_k1_s0, vmx_s0, 1, r, 0) ;
	  VFMAD_C2(vrgout_k1_s1, vmx_s1, 1, r, 1) ;
	  VFMAD_C2(vrgout_k1_s2, vmx_s2, 1, r, 2) ;
	  VFMAD_C2(vrgout_k1_s3, vmx_s3, 1, r, 3) ;
	  VFMAD_C2(vrgout_k1_s4, vmx_s4, 1, r, 4) ;

	  __vr vrgout_ptr_k2_s0 = _vel_vaddsl_vsvl(4*2*gOutHeight*gOutWidth, vrgout_ptr_k0_s0, vl) ;
	  __vr vrgout_k2_s0 = _vel_vgtu_vvssml(vrgout_ptr_k2_s0, 0, 0, vmx_s0, vl) ;
	  __vr vrgout_ptr_k2_s1 = _vel_vaddsl_vsvl(4*2*gOutHeight*gOutWidth, vrgout_ptr_k0_s1, vl) ;
	  __vr vrgout_k2_s1 = _vel_vgtu_vvssml(vrgout_ptr_k2_s1, 0, 0, vmx_s1, vl) ;
	  __vr vrgout_ptr_k2_s2 = _vel_vaddsl_vsvl(4*2*gOutHeight*gOutWidth, vrgout_ptr_k0_s2, vl) ;
	  __vr vrgout_k2_s2 = _vel_vgtu_vvssml(vrgout_ptr_k2_s2, 0, 0, vmx_s2, vl) ;
	  __vr vrgout_ptr_k2_s3 = _vel_vaddsl_vsvl(4*2*gOutHeight*gOutWidth, vrgout_ptr_k0_s3, vl) ;
	  __vr vrgout_k2_s3 = _vel_vgtu_vvssml(vrgout_ptr_k2_s3, 0, 0, vmx_s3, vl) ;
	  __vr vrgout_ptr_k2_s4 = _vel_vaddsl_vsvl(4*2*gOutHeight*gOutWidth, vrgout_ptr_k0_s4, vl) ;
	  __vr vrgout_k2_s4 = _vel_vgtu_vvssml(vrgout_ptr_k2_s4, 0, 0, vmx_s4, vl) ;
	  VFMAD_C2(vrgout_k2_s0, vmx_s0, 2, r, 0) ;
	  VFMAD_C2(vrgout_k2_s1, vmx_s1, 2, r, 1) ;
	  VFMAD_C2(vrgout_k2_s2, vmx_s2, 2, r, 2) ;
	  VFMAD_C2(vrgout_k2_s3, vmx_s3, 2, r, 3) ;
	  VFMAD_C2(vrgout_k2_s4, vmx_s4, 2, r, 4) ;

	  __vr vrgout_ptr_k3_s0 = _vel_vaddsl_vsvl(4*3*gOutHeight*gOutWidth, vrgout_ptr_k0_s0, vl) ;
	  __vr vrgout_k3_s0 = _vel_vgtu_vvssml(vrgout_ptr_k3_s0, 0, 0, vmx_s0, vl) ;
	  __vr vrgout_ptr_k3_s1 = _vel_vaddsl_vsvl(4*3*gOutHeight*gOutWidth, vrgout_ptr_k0_s1, vl) ;
	  __vr vrgout_k3_s1 = _vel_vgtu_vvssml(vrgout_ptr_k3_s1, 0, 0, vmx_s1, vl) ;
	  __vr vrgout_ptr_k3_s2 = _vel_vaddsl_vsvl(4*3*gOutHeight*gOutWidth, vrgout_ptr_k0_s2, vl) ;
	  __vr vrgout_k3_s2 = _vel_vgtu_vvssml(vrgout_ptr_k3_s2, 0, 0, vmx_s2, vl) ;
	  __vr vrgout_ptr_k3_s3 = _vel_vaddsl_vsvl(4*3*gOutHeight*gOutWidth, vrgout_ptr_k0_s3, vl) ;
	  __vr vrgout_k3_s3 = _vel_vgtu_vvssml(vrgout_ptr_k3_s3, 0, 0, vmx_s3, vl) ;
	  __vr vrgout_ptr_k3_s4 = _vel_vaddsl_vsvl(4*3*gOutHeight*gOutWidth, vrgout_ptr_k0_s4, vl) ;
	  __vr vrgout_k3_s4 = _vel_vgtu_vvssml(vrgout_ptr_k3_s4, 0, 0, vmx_s4, vl) ;
	  VFMAD_C2(vrgout_k3_s0, vmx_s0, 3, r, 0) ;
	  VFMAD_C2(vrgout_k3_s1, vmx_s1, 3, r, 1) ;
	  VFMAD_C2(vrgout_k3_s2, vmx_s2, 3, r, 2) ;
	  VFMAD_C2(vrgout_k3_s3, vmx_s3, 3, r, 3) ;
	  VFMAD_C2(vrgout_k3_s4, vmx_s4, 3, r, 4) ;

	  __vr vrgout_ptr_k4_s0 = _vel_vaddsl_vsvl(4*4*gOutHeight*gOutWidth, vrgout_ptr_k0_s0, vl) ;
	  __vr vrgout_k4_s0 = _vel_vgtu_vvssml(vrgout_ptr_k4_s0, 0, 0, vmx_s0, vl) ;
	  __vr vrgout_ptr_k4_s1 = _vel_vaddsl_vsvl(4*4*gOutHeight*gOutWidth, vrgout_ptr_k0_s1, vl) ;
	  __vr vrgout_k4_s1 = _vel_vgtu_vvssml(vrgout_ptr_k4_s1, 0, 0, vmx_s1, vl) ;
	  __vr vrgout_ptr_k4_s2 = _vel_vaddsl_vsvl(4*4*gOutHeight*gOutWidth, vrgout_ptr_k0_s2, vl) ;
	  __vr vrgout_k4_s2 = _vel_vgtu_vvssml(vrgout_ptr_k4_s2, 0, 0, vmx_s2, vl) ;
	  __vr vrgout_ptr_k4_s3 = _vel_vaddsl_vsvl(4*4*gOutHeight*gOutWidth, vrgout_ptr_k0_s3, vl) ;
	  __vr vrgout_k4_s3 = _vel_vgtu_vvssml(vrgout_ptr_k4_s3, 0, 0, vmx_s3, vl) ;
	  __vr vrgout_ptr_k4_s4 = _vel_vaddsl_vsvl(4*4*gOutHeight*gOutWidth, vrgout_ptr_k0_s4, vl) ;
	  __vr vrgout_k4_s4 = _vel_vgtu_vvssml(vrgout_ptr_k4_s4, 0, 0, vmx_s4, vl) ;
	  VFMAD_C2(vrgout_k4_s0, vmx_s0, 4, r, 0) ;
	  VFMAD_C2(vrgout_k4_s1, vmx_s1, 4, r, 1) ;
	  VFMAD_C2(vrgout_k4_s2, vmx_s2, 4, r, 2) ;
	  VFMAD_C2(vrgout_k4_s3, vmx_s3, 4, r, 3) ;
	  VFMAD_C2(vrgout_k4_s4, vmx_s4, 4, r, 4) ;

	  __vr vrgout_ptr_k5_s0 = _vel_vaddsl_vsvl(4*5*gOutHeight*gOutWidth, vrgout_ptr_k0_s0, vl) ;
	  __vr vrgout_k5_s0 = _vel_vgtu_vvssml(vrgout_ptr_k5_s0, 0, 0, vmx_s0, vl) ;
	  __vr vrgout_ptr_k5_s1 = _vel_vaddsl_vsvl(4*5*gOutHeight*gOutWidth, vrgout_ptr_k0_s1, vl) ;
	  __vr vrgout_k5_s1 = _vel_vgtu_vvssml(vrgout_ptr_k5_s1, 0, 0, vmx_s1, vl) ;
	  __vr vrgout_ptr_k5_s2 = _vel_vaddsl_vsvl(4*5*gOutHeight*gOutWidth, vrgout_ptr_k0_s2, vl) ;
	  __vr vrgout_k5_s2 = _vel_vgtu_vvssml(vrgout_ptr_k5_s2, 0, 0, vmx_s2, vl) ;
	  __vr vrgout_ptr_k5_s3 = _vel_vaddsl_vsvl(4*5*gOutHeight*gOutWidth, vrgout_ptr_k0_s3, vl) ;
	  __vr vrgout_k5_s3 = _vel_vgtu_vvssml(vrgout_ptr_k5_s3, 0, 0, vmx_s3, vl) ;
	  __vr vrgout_ptr_k5_s4 = _vel_vaddsl_vsvl(4*5*gOutHeight*gOutWidth, vrgout_ptr_k0_s4, vl) ;
	  __vr vrgout_k5_s4 = _vel_vgtu_vvssml(vrgout_ptr_k5_s4, 0, 0, vmx_s4, vl) ;
	  VFMAD_C2(vrgout_k5_s0, vmx_s0, 5, r, 0) ;
	  VFMAD_C2(vrgout_k5_s1, vmx_s1, 5, r, 1) ;
	  VFMAD_C2(vrgout_k5_s2, vmx_s2, 5, r, 2) ;
	  VFMAD_C2(vrgout_k5_s3, vmx_s3, 5, r, 3) ;
	  VFMAD_C2(vrgout_k5_s4, vmx_s4, 5, r, 4) ;

	  __vr vrgout_ptr_k6_s0 = _vel_vaddsl_vsvl(4*6*gOutHeight*gOutWidth, vrgout_ptr_k0_s0, vl) ;
	  __vr vrgout_k6_s0 = _vel_vgtu_vvssml(vrgout_ptr_k6_s0, 0, 0, vmx_s0, vl) ;
	  __vr vrgout_ptr_k6_s1 = _vel_vaddsl_vsvl(4*6*gOutHeight*gOutWidth, vrgout_ptr_k0_s1, vl) ;
	  __vr vrgout_k6_s1 = _vel_vgtu_vvssml(vrgout_ptr_k6_s1, 0, 0, vmx_s1, vl) ;
	  __vr vrgout_ptr_k6_s2 = _vel_vaddsl_vsvl(4*6*gOutHeight*gOutWidth, vrgout_ptr_k0_s2, vl) ;
	  __vr vrgout_k6_s2 = _vel_vgtu_vvssml(vrgout_ptr_k6_s2, 0, 0, vmx_s2, vl) ;
	  __vr vrgout_ptr_k6_s3 = _vel_vaddsl_vsvl(4*6*gOutHeight*gOutWidth, vrgout_ptr_k0_s3, vl) ;
	  __vr vrgout_k6_s3 = _vel_vgtu_vvssml(vrgout_ptr_k6_s3, 0, 0, vmx_s3, vl) ;
	  __vr vrgout_ptr_k6_s4 = _vel_vaddsl_vsvl(4*6*gOutHeight*gOutWidth, vrgout_ptr_k0_s4, vl) ;
	  __vr vrgout_k6_s4 = _vel_vgtu_vvssml(vrgout_ptr_k6_s4, 0, 0, vmx_s4, vl) ;
	  VFMAD_C2(vrgout_k6_s0, vmx_s0, 6, r, 0) ;
	  VFMAD_C2(vrgout_k6_s1, vmx_s1, 6, r, 1) ;
	  VFMAD_C2(vrgout_k6_s2, vmx_s2, 6, r, 2) ;
	  VFMAD_C2(vrgout_k6_s3, vmx_s3, 6, r, 3) ;
	  VFMAD_C2(vrgout_k6_s4, vmx_s4, 6, r, 4) ;

	  __vr vrgout_ptr_k7_s0 = _vel_vaddsl_vsvl(4*7*gOutHeight*gOutWidth, vrgout_ptr_k0_s0, vl) ;
	  __vr vrgout_k7_s0 = _vel_vgtu_vvssml(vrgout_ptr_k7_s0, 0, 0, vmx_s0, vl) ;
	  __vr vrgout_ptr_k7_s1 = _vel_vaddsl_vsvl(4*7*gOutHeight*gOutWidth, vrgout_ptr_k0_s1, vl) ;
	  __vr vrgout_k7_s1 = _vel_vgtu_vvssml(vrgout_ptr_k7_s1, 0, 0, vmx_s1, vl) ;
	  __vr vrgout_ptr_k7_s2 = _vel_vaddsl_vsvl(4*7*gOutHeight*gOutWidth, vrgout_ptr_k0_s2, vl) ;
	  __vr vrgout_k7_s2 = _vel_vgtu_vvssml(vrgout_ptr_k7_s2, 0, 0, vmx_s2, vl) ;
	  __vr vrgout_ptr_k7_s3 = _vel_vaddsl_vsvl(4*7*gOutHeight*gOutWidth, vrgout_ptr_k0_s3, vl) ;
	  __vr vrgout_k7_s3 = _vel_vgtu_vvssml(vrgout_ptr_k7_s3, 0, 0, vmx_s3, vl) ;
	  __vr vrgout_ptr_k7_s4 = _vel_vaddsl_vsvl(4*7*gOutHeight*gOutWidth, vrgout_ptr_k0_s4, vl) ;
	  __vr vrgout_k7_s4 = _vel_vgtu_vvssml(vrgout_ptr_k7_s4, 0, 0, vmx_s4, vl) ;
	  VFMAD_C2(vrgout_k7_s0, vmx_s0, 7, r, 0) ;
	  VFMAD_C2(vrgout_k7_s1, vmx_s1, 7, r, 1) ;
	  VFMAD_C2(vrgout_k7_s2, vmx_s2, 7, r, 2) ;
	  VFMAD_C2(vrgout_k7_s3, vmx_s3, 7, r, 3) ;
	  VFMAD_C2(vrgout_k7_s4, vmx_s4, 7, r, 4) ;

#undef VFMAD_C2
	} // gOutChannel

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

      __vr vrj_s0 = _vel_vaddsl_vsvl(padWidth-0*dilationWidth, vrw, vl) ;
      __vr vrj_s1 = _vel_vaddsl_vsvl(padWidth-1*dilationWidth, vrw, vl) ;
      __vr vrj_s2 = _vel_vaddsl_vsvl(padWidth-2*dilationWidth, vrw, vl) ;
      __vr vrj_s3 = _vel_vaddsl_vsvl(padWidth-3*dilationWidth, vrw, vl) ;
      __vr vrj_s4 = _vel_vaddsl_vsvl(padWidth-4*dilationWidth, vrw, vl) ;

      __vr vrx_s0 = _vel_vdivsl_vvsl(vrj_s0, strideWidth, vl) ;
      __vr vrx_s1 = _vel_vdivsl_vvsl(vrj_s1, strideWidth, vl) ;
      __vr vrx_s2 = _vel_vdivsl_vvsl(vrj_s2, strideWidth, vl) ;
      __vr vrx_s3 = _vel_vdivsl_vvsl(vrj_s3, strideWidth, vl) ;
      __vr vrx_s4 = _vel_vdivsl_vvsl(vrj_s4, strideWidth, vl) ;

      __vm256 vmx0_s0 =  _vel_vfmkleq_mvl(_vel_vcmpsl_vvvl(vrj_s0, _vel_vmulsl_vsvl(strideWidth, vrx_s0, vl), vl), vl) ;
      __vm256 vmx1_s0 =  _vel_vfmklge_mvl(vrx_s0, vl) ;
      __vm256 vmx2_s0 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutWidth,vrx_s0, vl), vl) ;
      __vm256 vmx_s0 = _vel_andm_mmm(_vel_andm_mmm(vmx0_s0, vmx1_s0), vmx2_s0) ;

      __vm256 vmx0_s1 =  _vel_vfmkleq_mvl(_vel_vcmpsl_vvvl(vrj_s1, _vel_vmulsl_vsvl(strideWidth, vrx_s1, vl), vl), vl) ;
      __vm256 vmx1_s1 =  _vel_vfmklge_mvl(vrx_s1, vl) ;
      __vm256 vmx2_s1 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutWidth,vrx_s1, vl), vl) ;
      __vm256 vmx_s1 = _vel_andm_mmm(_vel_andm_mmm(vmx0_s1, vmx1_s1), vmx2_s1) ;

      __vm256 vmx0_s2 =  _vel_vfmkleq_mvl(_vel_vcmpsl_vvvl(vrj_s2, _vel_vmulsl_vsvl(strideWidth, vrx_s2, vl), vl), vl) ;
      __vm256 vmx1_s2 =  _vel_vfmklge_mvl(vrx_s2, vl) ;
      __vm256 vmx2_s2 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutWidth,vrx_s2, vl), vl) ;
      __vm256 vmx_s2 = _vel_andm_mmm(_vel_andm_mmm(vmx0_s2, vmx1_s2), vmx2_s2) ;

      __vm256 vmx0_s3 =  _vel_vfmkleq_mvl(_vel_vcmpsl_vvvl(vrj_s3, _vel_vmulsl_vsvl(strideWidth, vrx_s3, vl), vl), vl) ;
      __vm256 vmx1_s3 =  _vel_vfmklge_mvl(vrx_s3, vl) ;
      __vm256 vmx2_s3 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutWidth,vrx_s3, vl), vl) ;
      __vm256 vmx_s3 = _vel_andm_mmm(_vel_andm_mmm(vmx0_s3, vmx1_s3), vmx2_s3) ;

      __vm256 vmx0_s4 =  _vel_vfmkleq_mvl(_vel_vcmpsl_vvvl(vrj_s4, _vel_vmulsl_vsvl(strideWidth, vrx_s4, vl), vl), vl) ;
      __vm256 vmx1_s4 =  _vel_vfmklge_mvl(vrx_s4, vl) ;
      __vm256 vmx2_s4 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutWidth,vrx_s4, vl), vl) ;
      __vm256 vmx_s4 = _vel_andm_mmm(_vel_andm_mmm(vmx0_s4, vmx1_s4), vmx2_s4) ;

      for (int64_t r=0; r<kernHeight; r++) {
	int64_t i = h - r * dilationHeight + padHeight ;
	int64_t y = i/strideHeight;
	if ( y*strideHeight != i || y < 0 || gOutHeight <= y)  continue ;

	int64_t k=0;
	if( (gOutChannelGroup & 0x01)  == 1 ) {
	  int64_t gOutIndex    = gOutGroupOffset + ((n * gOutChannel + k) * gOutHeight) * gOutWidth ;
	  const float *pKerValue = pKernel + kernGroupOffset + ((k * gInChannelGroup + c) * kernHeight) * kernWidth;

	  __vr vrgout_ptr_k0_s0 = _vel_vsfa_vvssl(_vel_vaddsl_vsvl(gOutWidth*y, vrx_s0, vl),
					     2,
					     (unsigned long)(pGOut+gOutIndex), vl) ;
	  __vr vrgout_k0_s0 = _vel_vgtu_vvssml(vrgout_ptr_k0_s0, 0, 0, vmx_s0, vl) ;
	  __vr vrgout_ptr_k0_s1 = _vel_vsfa_vvssl(_vel_vaddsl_vsvl(gOutWidth*y, vrx_s1, vl),
					     2,
					     (unsigned long)(pGOut+gOutIndex), vl) ;
	  __vr vrgout_k0_s1 = _vel_vgtu_vvssml(vrgout_ptr_k0_s1, 0, 0, vmx_s1, vl) ;
	  __vr vrgout_ptr_k0_s2 = _vel_vsfa_vvssl(_vel_vaddsl_vsvl(gOutWidth*y, vrx_s2, vl),
					     2,
					     (unsigned long)(pGOut+gOutIndex), vl) ;
	  __vr vrgout_k0_s2 = _vel_vgtu_vvssml(vrgout_ptr_k0_s2, 0, 0, vmx_s2, vl) ;
	  __vr vrgout_ptr_k0_s3 = _vel_vsfa_vvssl(_vel_vaddsl_vsvl(gOutWidth*y, vrx_s3, vl),
					     2,
					     (unsigned long)(pGOut+gOutIndex), vl) ;
	  __vr vrgout_k0_s3 = _vel_vgtu_vvssml(vrgout_ptr_k0_s3, 0, 0, vmx_s3, vl) ;
	  __vr vrgout_ptr_k0_s4 = _vel_vsfa_vvssl(_vel_vaddsl_vsvl(gOutWidth*y, vrx_s4, vl),
					     2,
					     (unsigned long)(pGOut+gOutIndex), vl) ;
	  __vr vrgout_k0_s4 = _vel_vgtu_vvssml(vrgout_ptr_k0_s4, 0, 0, vmx_s4, vl) ;

#define VFMAD_C4(VRGOUT, VM, K, R, S)	{												\
	    const uint64_t kerValue01 = _vel_pack_f32p(pKerValue + (((K)*gInChannelGroup + 0) * kernHeight +(R)) * kernWidth + (S),	\
						       pKerValue + (((K)*gInChannelGroup + 1) * kernHeight +(R)) * kernWidth + (S)) ;	\
	    const uint64_t kerValue23 = _vel_pack_f32p(pKerValue + (((K)*gInChannelGroup + 2) * kernHeight +(R)) * kernWidth + (S),	\
						       pKerValue + (((K)*gInChannelGroup + 3) * kernHeight +(R)) * kernWidth + (S)) ;	\
	    VRGOUT = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), VRGOUT, VM, vl) ;	\
	    __vr vrgoutP = _vel_vshf_vvvsl(VRGOUT, VRGOUT, VE_VSHUFFLE_YUZU, vl) ;	\
	    vrsum01 = _vel_pvfmad_vvsvl(vrsum01, kerValue01, vrgoutP, vl) ;		\
	    vrsum23 = _vel_pvfmad_vvsvl(vrsum23, kerValue23, vrgoutP, vl) ;		\
}
	  VFMAD_C4(vrgout_k0_s0, vmx_s0, 0, r, 0) ;
	  VFMAD_C4(vrgout_k0_s1, vmx_s1, 0, r, 1) ;
	  VFMAD_C4(vrgout_k0_s2, vmx_s2, 0, r, 2) ;
	  VFMAD_C4(vrgout_k0_s3, vmx_s3, 0, r, 3) ;
	  VFMAD_C4(vrgout_k0_s4, vmx_s4, 0, r, 4) ;

	  k+=1 ;
	}
	if( ( (gOutChannelGroup >> 1) & 0x01)  == 1 ) {
	  int64_t gOutIndex    = gOutGroupOffset + ((n * gOutChannel + k) * gOutHeight) * gOutWidth ;
	  const float *pKerValue = pKernel + kernGroupOffset + ((k * gInChannelGroup + c) * kernHeight) * kernWidth;

	  __vr vrgout_ptr_k0_s0 = _vel_vsfa_vvssl(_vel_vaddsl_vsvl(gOutWidth*y, vrx_s0, vl),
					     2,
					     (unsigned long)(pGOut+gOutIndex), vl) ;
	  __vr vrgout_k0_s0 = _vel_vgtu_vvssml(vrgout_ptr_k0_s0, 0, 0, vmx_s0, vl) ;
	  __vr vrgout_ptr_k0_s1 = _vel_vsfa_vvssl(_vel_vaddsl_vsvl(gOutWidth*y, vrx_s1, vl),
					     2,
					     (unsigned long)(pGOut+gOutIndex), vl) ;
	  __vr vrgout_k0_s1 = _vel_vgtu_vvssml(vrgout_ptr_k0_s1, 0, 0, vmx_s1, vl) ;
	  __vr vrgout_ptr_k0_s2 = _vel_vsfa_vvssl(_vel_vaddsl_vsvl(gOutWidth*y, vrx_s2, vl),
					     2,
					     (unsigned long)(pGOut+gOutIndex), vl) ;
	  __vr vrgout_k0_s2 = _vel_vgtu_vvssml(vrgout_ptr_k0_s2, 0, 0, vmx_s2, vl) ;
	  __vr vrgout_ptr_k0_s3 = _vel_vsfa_vvssl(_vel_vaddsl_vsvl(gOutWidth*y, vrx_s3, vl),
					     2,
					     (unsigned long)(pGOut+gOutIndex), vl) ;
	  __vr vrgout_k0_s3 = _vel_vgtu_vvssml(vrgout_ptr_k0_s3, 0, 0, vmx_s3, vl) ;
	  __vr vrgout_ptr_k0_s4 = _vel_vsfa_vvssl(_vel_vaddsl_vsvl(gOutWidth*y, vrx_s4, vl),
					     2,
					     (unsigned long)(pGOut+gOutIndex), vl) ;
	  __vr vrgout_k0_s4 = _vel_vgtu_vvssml(vrgout_ptr_k0_s4, 0, 0, vmx_s4, vl) ;
	  VFMAD_C4(vrgout_k0_s0, vmx_s0, 0, r, 0) ;
	  VFMAD_C4(vrgout_k0_s1, vmx_s1, 0, r, 1) ;
	  VFMAD_C4(vrgout_k0_s2, vmx_s2, 0, r, 2) ;
	  VFMAD_C4(vrgout_k0_s3, vmx_s3, 0, r, 3) ;
	  VFMAD_C4(vrgout_k0_s4, vmx_s4, 0, r, 4) ;

	  __vr vrgout_ptr_k1_s0 = _vel_vaddsl_vsvl(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_s0, vl) ;
	  __vr vrgout_k1_s0 = _vel_vgtu_vvssml(vrgout_ptr_k1_s0, 0, 0, vmx_s0, vl) ;
	  __vr vrgout_ptr_k1_s1 = _vel_vaddsl_vsvl(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_s1, vl) ;
	  __vr vrgout_k1_s1 = _vel_vgtu_vvssml(vrgout_ptr_k1_s1, 0, 0, vmx_s1, vl) ;
	  __vr vrgout_ptr_k1_s2 = _vel_vaddsl_vsvl(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_s2, vl) ;
	  __vr vrgout_k1_s2 = _vel_vgtu_vvssml(vrgout_ptr_k1_s2, 0, 0, vmx_s2, vl) ;
	  __vr vrgout_ptr_k1_s3 = _vel_vaddsl_vsvl(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_s3, vl) ;
	  __vr vrgout_k1_s3 = _vel_vgtu_vvssml(vrgout_ptr_k1_s3, 0, 0, vmx_s3, vl) ;
	  __vr vrgout_ptr_k1_s4 = _vel_vaddsl_vsvl(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_s4, vl) ;
	  __vr vrgout_k1_s4 = _vel_vgtu_vvssml(vrgout_ptr_k1_s4, 0, 0, vmx_s4, vl) ;
	  VFMAD_C4(vrgout_k1_s0, vmx_s0, 1, r, 0) ;
	  VFMAD_C4(vrgout_k1_s1, vmx_s1, 1, r, 1) ;
	  VFMAD_C4(vrgout_k1_s2, vmx_s2, 1, r, 2) ;
	  VFMAD_C4(vrgout_k1_s3, vmx_s3, 1, r, 3) ;
	  VFMAD_C4(vrgout_k1_s4, vmx_s4, 1, r, 4) ;

	  k+=2 ;
	}
	if( ( (gOutChannelGroup >> 2) & 0x01)  == 1 ) {
	  int64_t gOutIndex    = gOutGroupOffset + ((n * gOutChannel + k) * gOutHeight) * gOutWidth ;
	  const float *pKerValue = pKernel + kernGroupOffset + ((k * gInChannelGroup + c) * kernHeight) * kernWidth;

	  __vr vrgout_ptr_k0_s0 = _vel_vsfa_vvssl(_vel_vaddsl_vsvl(gOutWidth*y, vrx_s0, vl),
					     2,
					     (unsigned long)(pGOut+gOutIndex), vl) ;
	  __vr vrgout_k0_s0 = _vel_vgtu_vvssml(vrgout_ptr_k0_s0, 0, 0, vmx_s0, vl) ;
	  __vr vrgout_ptr_k0_s1 = _vel_vsfa_vvssl(_vel_vaddsl_vsvl(gOutWidth*y, vrx_s1, vl),
					     2,
					     (unsigned long)(pGOut+gOutIndex), vl) ;
	  __vr vrgout_k0_s1 = _vel_vgtu_vvssml(vrgout_ptr_k0_s1, 0, 0, vmx_s1, vl) ;
	  __vr vrgout_ptr_k0_s2 = _vel_vsfa_vvssl(_vel_vaddsl_vsvl(gOutWidth*y, vrx_s2, vl),
					     2,
					     (unsigned long)(pGOut+gOutIndex), vl) ;
	  __vr vrgout_k0_s2 = _vel_vgtu_vvssml(vrgout_ptr_k0_s2, 0, 0, vmx_s2, vl) ;
	  __vr vrgout_ptr_k0_s3 = _vel_vsfa_vvssl(_vel_vaddsl_vsvl(gOutWidth*y, vrx_s3, vl),
					     2,
					     (unsigned long)(pGOut+gOutIndex), vl) ;
	  __vr vrgout_k0_s3 = _vel_vgtu_vvssml(vrgout_ptr_k0_s3, 0, 0, vmx_s3, vl) ;
	  __vr vrgout_ptr_k0_s4 = _vel_vsfa_vvssl(_vel_vaddsl_vsvl(gOutWidth*y, vrx_s4, vl),
					     2,
					     (unsigned long)(pGOut+gOutIndex), vl) ;
	  __vr vrgout_k0_s4 = _vel_vgtu_vvssml(vrgout_ptr_k0_s4, 0, 0, vmx_s4, vl) ;
	  VFMAD_C4(vrgout_k0_s0, vmx_s0, 0, r, 0) ;
	  VFMAD_C4(vrgout_k0_s1, vmx_s1, 0, r, 1) ;
	  VFMAD_C4(vrgout_k0_s2, vmx_s2, 0, r, 2) ;
	  VFMAD_C4(vrgout_k0_s3, vmx_s3, 0, r, 3) ;
	  VFMAD_C4(vrgout_k0_s4, vmx_s4, 0, r, 4) ;

	  __vr vrgout_ptr_k1_s0 = _vel_vaddsl_vsvl(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_s0, vl) ;
	  __vr vrgout_k1_s0 = _vel_vgtu_vvssml(vrgout_ptr_k1_s0, 0, 0, vmx_s0, vl) ;
	  __vr vrgout_ptr_k1_s1 = _vel_vaddsl_vsvl(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_s1, vl) ;
	  __vr vrgout_k1_s1 = _vel_vgtu_vvssml(vrgout_ptr_k1_s1, 0, 0, vmx_s1, vl) ;
	  __vr vrgout_ptr_k1_s2 = _vel_vaddsl_vsvl(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_s2, vl) ;
	  __vr vrgout_k1_s2 = _vel_vgtu_vvssml(vrgout_ptr_k1_s2, 0, 0, vmx_s2, vl) ;
	  __vr vrgout_ptr_k1_s3 = _vel_vaddsl_vsvl(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_s3, vl) ;
	  __vr vrgout_k1_s3 = _vel_vgtu_vvssml(vrgout_ptr_k1_s3, 0, 0, vmx_s3, vl) ;
	  __vr vrgout_ptr_k1_s4 = _vel_vaddsl_vsvl(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_s4, vl) ;
	  __vr vrgout_k1_s4 = _vel_vgtu_vvssml(vrgout_ptr_k1_s4, 0, 0, vmx_s4, vl) ;
	  VFMAD_C4(vrgout_k1_s0, vmx_s0, 1, r, 0) ;
	  VFMAD_C4(vrgout_k1_s1, vmx_s1, 1, r, 1) ;
	  VFMAD_C4(vrgout_k1_s2, vmx_s2, 1, r, 2) ;
	  VFMAD_C4(vrgout_k1_s3, vmx_s3, 1, r, 3) ;
	  VFMAD_C4(vrgout_k1_s4, vmx_s4, 1, r, 4) ;

	  __vr vrgout_ptr_k2_s0 = _vel_vaddsl_vsvl(4*2*gOutHeight*gOutWidth, vrgout_ptr_k0_s0, vl) ;
	  __vr vrgout_k2_s0 = _vel_vgtu_vvssml(vrgout_ptr_k2_s0, 0, 0, vmx_s0, vl) ;
	  __vr vrgout_ptr_k2_s1 = _vel_vaddsl_vsvl(4*2*gOutHeight*gOutWidth, vrgout_ptr_k0_s1, vl) ;
	  __vr vrgout_k2_s1 = _vel_vgtu_vvssml(vrgout_ptr_k2_s1, 0, 0, vmx_s1, vl) ;
	  __vr vrgout_ptr_k2_s2 = _vel_vaddsl_vsvl(4*2*gOutHeight*gOutWidth, vrgout_ptr_k0_s2, vl) ;
	  __vr vrgout_k2_s2 = _vel_vgtu_vvssml(vrgout_ptr_k2_s2, 0, 0, vmx_s2, vl) ;
	  __vr vrgout_ptr_k2_s3 = _vel_vaddsl_vsvl(4*2*gOutHeight*gOutWidth, vrgout_ptr_k0_s3, vl) ;
	  __vr vrgout_k2_s3 = _vel_vgtu_vvssml(vrgout_ptr_k2_s3, 0, 0, vmx_s3, vl) ;
	  __vr vrgout_ptr_k2_s4 = _vel_vaddsl_vsvl(4*2*gOutHeight*gOutWidth, vrgout_ptr_k0_s4, vl) ;
	  __vr vrgout_k2_s4 = _vel_vgtu_vvssml(vrgout_ptr_k2_s4, 0, 0, vmx_s4, vl) ;
	  VFMAD_C4(vrgout_k2_s0, vmx_s0, 2, r, 0) ;
	  VFMAD_C4(vrgout_k2_s1, vmx_s1, 2, r, 1) ;
	  VFMAD_C4(vrgout_k2_s2, vmx_s2, 2, r, 2) ;
	  VFMAD_C4(vrgout_k2_s3, vmx_s3, 2, r, 3) ;
	  VFMAD_C4(vrgout_k2_s4, vmx_s4, 2, r, 4) ;

	  __vr vrgout_ptr_k3_s0 = _vel_vaddsl_vsvl(4*3*gOutHeight*gOutWidth, vrgout_ptr_k0_s0, vl) ;
	  __vr vrgout_k3_s0 = _vel_vgtu_vvssml(vrgout_ptr_k3_s0, 0, 0, vmx_s0, vl) ;
	  __vr vrgout_ptr_k3_s1 = _vel_vaddsl_vsvl(4*3*gOutHeight*gOutWidth, vrgout_ptr_k0_s1, vl) ;
	  __vr vrgout_k3_s1 = _vel_vgtu_vvssml(vrgout_ptr_k3_s1, 0, 0, vmx_s1, vl) ;
	  __vr vrgout_ptr_k3_s2 = _vel_vaddsl_vsvl(4*3*gOutHeight*gOutWidth, vrgout_ptr_k0_s2, vl) ;
	  __vr vrgout_k3_s2 = _vel_vgtu_vvssml(vrgout_ptr_k3_s2, 0, 0, vmx_s2, vl) ;
	  __vr vrgout_ptr_k3_s3 = _vel_vaddsl_vsvl(4*3*gOutHeight*gOutWidth, vrgout_ptr_k0_s3, vl) ;
	  __vr vrgout_k3_s3 = _vel_vgtu_vvssml(vrgout_ptr_k3_s3, 0, 0, vmx_s3, vl) ;
	  __vr vrgout_ptr_k3_s4 = _vel_vaddsl_vsvl(4*3*gOutHeight*gOutWidth, vrgout_ptr_k0_s4, vl) ;
	  __vr vrgout_k3_s4 = _vel_vgtu_vvssml(vrgout_ptr_k3_s4, 0, 0, vmx_s4, vl) ;
	  VFMAD_C4(vrgout_k3_s0, vmx_s0, 3, r, 0) ;
	  VFMAD_C4(vrgout_k3_s1, vmx_s1, 3, r, 1) ;
	  VFMAD_C4(vrgout_k3_s2, vmx_s2, 3, r, 2) ;
	  VFMAD_C4(vrgout_k3_s3, vmx_s3, 3, r, 3) ;
	  VFMAD_C4(vrgout_k3_s4, vmx_s4, 3, r, 4) ;

	  k+=4 ;
	}
	for ( ; k<gOutChannelGroup; k+=8 ) {
	  int64_t gOutIndex    = gOutGroupOffset + ((n * gOutChannel + k) * gOutHeight) * gOutWidth ;
	  const float *pKerValue = pKernel + kernGroupOffset + ((k * gInChannelGroup + c) * kernHeight) * kernWidth;

	  __vr vrgout_ptr_k0_s0 = _vel_vsfa_vvssl(_vel_vaddsl_vsvl(gOutWidth*y, vrx_s0, vl),
					     2,
					     (unsigned long)(pGOut+gOutIndex), vl) ;
	  __vr vrgout_k0_s0 = _vel_vgtu_vvssml(vrgout_ptr_k0_s0, 0, 0, vmx_s0, vl) ;
	  __vr vrgout_ptr_k0_s1 = _vel_vsfa_vvssl(_vel_vaddsl_vsvl(gOutWidth*y, vrx_s1, vl),
					     2,
					     (unsigned long)(pGOut+gOutIndex), vl) ;
	  __vr vrgout_k0_s1 = _vel_vgtu_vvssml(vrgout_ptr_k0_s1, 0, 0, vmx_s1, vl) ;
	  __vr vrgout_ptr_k0_s2 = _vel_vsfa_vvssl(_vel_vaddsl_vsvl(gOutWidth*y, vrx_s2, vl),
					     2,
					     (unsigned long)(pGOut+gOutIndex), vl) ;
	  __vr vrgout_k0_s2 = _vel_vgtu_vvssml(vrgout_ptr_k0_s2, 0, 0, vmx_s2, vl) ;
	  __vr vrgout_ptr_k0_s3 = _vel_vsfa_vvssl(_vel_vaddsl_vsvl(gOutWidth*y, vrx_s3, vl),
					     2,
					     (unsigned long)(pGOut+gOutIndex), vl) ;
	  __vr vrgout_k0_s3 = _vel_vgtu_vvssml(vrgout_ptr_k0_s3, 0, 0, vmx_s3, vl) ;
	  __vr vrgout_ptr_k0_s4 = _vel_vsfa_vvssl(_vel_vaddsl_vsvl(gOutWidth*y, vrx_s4, vl),
					     2,
					     (unsigned long)(pGOut+gOutIndex), vl) ;
	  __vr vrgout_k0_s4 = _vel_vgtu_vvssml(vrgout_ptr_k0_s4, 0, 0, vmx_s4, vl) ;
	  VFMAD_C4(vrgout_k0_s0, vmx_s0, 0, r, 0) ;
	  VFMAD_C4(vrgout_k0_s1, vmx_s1, 0, r, 1) ;
	  VFMAD_C4(vrgout_k0_s2, vmx_s2, 0, r, 2) ;
	  VFMAD_C4(vrgout_k0_s3, vmx_s3, 0, r, 3) ;
	  VFMAD_C4(vrgout_k0_s4, vmx_s4, 0, r, 4) ;

	  __vr vrgout_ptr_k1_s0 = _vel_vaddsl_vsvl(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_s0, vl) ;
	  __vr vrgout_k1_s0 = _vel_vgtu_vvssml(vrgout_ptr_k1_s0, 0, 0, vmx_s0, vl) ;
	  __vr vrgout_ptr_k1_s1 = _vel_vaddsl_vsvl(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_s1, vl) ;
	  __vr vrgout_k1_s1 = _vel_vgtu_vvssml(vrgout_ptr_k1_s1, 0, 0, vmx_s1, vl) ;
	  __vr vrgout_ptr_k1_s2 = _vel_vaddsl_vsvl(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_s2, vl) ;
	  __vr vrgout_k1_s2 = _vel_vgtu_vvssml(vrgout_ptr_k1_s2, 0, 0, vmx_s2, vl) ;
	  __vr vrgout_ptr_k1_s3 = _vel_vaddsl_vsvl(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_s3, vl) ;
	  __vr vrgout_k1_s3 = _vel_vgtu_vvssml(vrgout_ptr_k1_s3, 0, 0, vmx_s3, vl) ;
	  __vr vrgout_ptr_k1_s4 = _vel_vaddsl_vsvl(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_s4, vl) ;
	  __vr vrgout_k1_s4 = _vel_vgtu_vvssml(vrgout_ptr_k1_s4, 0, 0, vmx_s4, vl) ;
	  VFMAD_C4(vrgout_k1_s0, vmx_s0, 1, r, 0) ;
	  VFMAD_C4(vrgout_k1_s1, vmx_s1, 1, r, 1) ;
	  VFMAD_C4(vrgout_k1_s2, vmx_s2, 1, r, 2) ;
	  VFMAD_C4(vrgout_k1_s3, vmx_s3, 1, r, 3) ;
	  VFMAD_C4(vrgout_k1_s4, vmx_s4, 1, r, 4) ;

	  __vr vrgout_ptr_k2_s0 = _vel_vaddsl_vsvl(4*2*gOutHeight*gOutWidth, vrgout_ptr_k0_s0, vl) ;
	  __vr vrgout_k2_s0 = _vel_vgtu_vvssml(vrgout_ptr_k2_s0, 0, 0, vmx_s0, vl) ;
	  __vr vrgout_ptr_k2_s1 = _vel_vaddsl_vsvl(4*2*gOutHeight*gOutWidth, vrgout_ptr_k0_s1, vl) ;
	  __vr vrgout_k2_s1 = _vel_vgtu_vvssml(vrgout_ptr_k2_s1, 0, 0, vmx_s1, vl) ;
	  __vr vrgout_ptr_k2_s2 = _vel_vaddsl_vsvl(4*2*gOutHeight*gOutWidth, vrgout_ptr_k0_s2, vl) ;
	  __vr vrgout_k2_s2 = _vel_vgtu_vvssml(vrgout_ptr_k2_s2, 0, 0, vmx_s2, vl) ;
	  __vr vrgout_ptr_k2_s3 = _vel_vaddsl_vsvl(4*2*gOutHeight*gOutWidth, vrgout_ptr_k0_s3, vl) ;
	  __vr vrgout_k2_s3 = _vel_vgtu_vvssml(vrgout_ptr_k2_s3, 0, 0, vmx_s3, vl) ;
	  __vr vrgout_ptr_k2_s4 = _vel_vaddsl_vsvl(4*2*gOutHeight*gOutWidth, vrgout_ptr_k0_s4, vl) ;
	  __vr vrgout_k2_s4 = _vel_vgtu_vvssml(vrgout_ptr_k2_s4, 0, 0, vmx_s4, vl) ;
	  VFMAD_C4(vrgout_k2_s0, vmx_s0, 2, r, 0) ;
	  VFMAD_C4(vrgout_k2_s1, vmx_s1, 2, r, 1) ;
	  VFMAD_C4(vrgout_k2_s2, vmx_s2, 2, r, 2) ;
	  VFMAD_C4(vrgout_k2_s3, vmx_s3, 2, r, 3) ;
	  VFMAD_C4(vrgout_k2_s4, vmx_s4, 2, r, 4) ;

	  __vr vrgout_ptr_k3_s0 = _vel_vaddsl_vsvl(4*3*gOutHeight*gOutWidth, vrgout_ptr_k0_s0, vl) ;
	  __vr vrgout_k3_s0 = _vel_vgtu_vvssml(vrgout_ptr_k3_s0, 0, 0, vmx_s0, vl) ;
	  __vr vrgout_ptr_k3_s1 = _vel_vaddsl_vsvl(4*3*gOutHeight*gOutWidth, vrgout_ptr_k0_s1, vl) ;
	  __vr vrgout_k3_s1 = _vel_vgtu_vvssml(vrgout_ptr_k3_s1, 0, 0, vmx_s1, vl) ;
	  __vr vrgout_ptr_k3_s2 = _vel_vaddsl_vsvl(4*3*gOutHeight*gOutWidth, vrgout_ptr_k0_s2, vl) ;
	  __vr vrgout_k3_s2 = _vel_vgtu_vvssml(vrgout_ptr_k3_s2, 0, 0, vmx_s2, vl) ;
	  __vr vrgout_ptr_k3_s3 = _vel_vaddsl_vsvl(4*3*gOutHeight*gOutWidth, vrgout_ptr_k0_s3, vl) ;
	  __vr vrgout_k3_s3 = _vel_vgtu_vvssml(vrgout_ptr_k3_s3, 0, 0, vmx_s3, vl) ;
	  __vr vrgout_ptr_k3_s4 = _vel_vaddsl_vsvl(4*3*gOutHeight*gOutWidth, vrgout_ptr_k0_s4, vl) ;
	  __vr vrgout_k3_s4 = _vel_vgtu_vvssml(vrgout_ptr_k3_s4, 0, 0, vmx_s4, vl) ;
	  VFMAD_C4(vrgout_k3_s0, vmx_s0, 3, r, 0) ;
	  VFMAD_C4(vrgout_k3_s1, vmx_s1, 3, r, 1) ;
	  VFMAD_C4(vrgout_k3_s2, vmx_s2, 3, r, 2) ;
	  VFMAD_C4(vrgout_k3_s3, vmx_s3, 3, r, 3) ;
	  VFMAD_C4(vrgout_k3_s4, vmx_s4, 3, r, 4) ;

	  __vr vrgout_ptr_k4_s0 = _vel_vaddsl_vsvl(4*4*gOutHeight*gOutWidth, vrgout_ptr_k0_s0, vl) ;
	  __vr vrgout_k4_s0 = _vel_vgtu_vvssml(vrgout_ptr_k4_s0, 0, 0, vmx_s0, vl) ;
	  __vr vrgout_ptr_k4_s1 = _vel_vaddsl_vsvl(4*4*gOutHeight*gOutWidth, vrgout_ptr_k0_s1, vl) ;
	  __vr vrgout_k4_s1 = _vel_vgtu_vvssml(vrgout_ptr_k4_s1, 0, 0, vmx_s1, vl) ;
	  __vr vrgout_ptr_k4_s2 = _vel_vaddsl_vsvl(4*4*gOutHeight*gOutWidth, vrgout_ptr_k0_s2, vl) ;
	  __vr vrgout_k4_s2 = _vel_vgtu_vvssml(vrgout_ptr_k4_s2, 0, 0, vmx_s2, vl) ;
	  __vr vrgout_ptr_k4_s3 = _vel_vaddsl_vsvl(4*4*gOutHeight*gOutWidth, vrgout_ptr_k0_s3, vl) ;
	  __vr vrgout_k4_s3 = _vel_vgtu_vvssml(vrgout_ptr_k4_s3, 0, 0, vmx_s3, vl) ;
	  __vr vrgout_ptr_k4_s4 = _vel_vaddsl_vsvl(4*4*gOutHeight*gOutWidth, vrgout_ptr_k0_s4, vl) ;
	  __vr vrgout_k4_s4 = _vel_vgtu_vvssml(vrgout_ptr_k4_s4, 0, 0, vmx_s4, vl) ;
	  VFMAD_C4(vrgout_k4_s0, vmx_s0, 4, r, 0) ;
	  VFMAD_C4(vrgout_k4_s1, vmx_s1, 4, r, 1) ;
	  VFMAD_C4(vrgout_k4_s2, vmx_s2, 4, r, 2) ;
	  VFMAD_C4(vrgout_k4_s3, vmx_s3, 4, r, 3) ;
	  VFMAD_C4(vrgout_k4_s4, vmx_s4, 4, r, 4) ;

	  __vr vrgout_ptr_k5_s0 = _vel_vaddsl_vsvl(4*5*gOutHeight*gOutWidth, vrgout_ptr_k0_s0, vl) ;
	  __vr vrgout_k5_s0 = _vel_vgtu_vvssml(vrgout_ptr_k5_s0, 0, 0, vmx_s0, vl) ;
	  __vr vrgout_ptr_k5_s1 = _vel_vaddsl_vsvl(4*5*gOutHeight*gOutWidth, vrgout_ptr_k0_s1, vl) ;
	  __vr vrgout_k5_s1 = _vel_vgtu_vvssml(vrgout_ptr_k5_s1, 0, 0, vmx_s1, vl) ;
	  __vr vrgout_ptr_k5_s2 = _vel_vaddsl_vsvl(4*5*gOutHeight*gOutWidth, vrgout_ptr_k0_s2, vl) ;
	  __vr vrgout_k5_s2 = _vel_vgtu_vvssml(vrgout_ptr_k5_s2, 0, 0, vmx_s2, vl) ;
	  __vr vrgout_ptr_k5_s3 = _vel_vaddsl_vsvl(4*5*gOutHeight*gOutWidth, vrgout_ptr_k0_s3, vl) ;
	  __vr vrgout_k5_s3 = _vel_vgtu_vvssml(vrgout_ptr_k5_s3, 0, 0, vmx_s3, vl) ;
	  __vr vrgout_ptr_k5_s4 = _vel_vaddsl_vsvl(4*5*gOutHeight*gOutWidth, vrgout_ptr_k0_s4, vl) ;
	  __vr vrgout_k5_s4 = _vel_vgtu_vvssml(vrgout_ptr_k5_s4, 0, 0, vmx_s4, vl) ;
	  VFMAD_C4(vrgout_k5_s0, vmx_s0, 5, r, 0) ;
	  VFMAD_C4(vrgout_k5_s1, vmx_s1, 5, r, 1) ;
	  VFMAD_C4(vrgout_k5_s2, vmx_s2, 5, r, 2) ;
	  VFMAD_C4(vrgout_k5_s3, vmx_s3, 5, r, 3) ;
	  VFMAD_C4(vrgout_k5_s4, vmx_s4, 5, r, 4) ;

	  __vr vrgout_ptr_k6_s0 = _vel_vaddsl_vsvl(4*6*gOutHeight*gOutWidth, vrgout_ptr_k0_s0, vl) ;
	  __vr vrgout_k6_s0 = _vel_vgtu_vvssml(vrgout_ptr_k6_s0, 0, 0, vmx_s0, vl) ;
	  __vr vrgout_ptr_k6_s1 = _vel_vaddsl_vsvl(4*6*gOutHeight*gOutWidth, vrgout_ptr_k0_s1, vl) ;
	  __vr vrgout_k6_s1 = _vel_vgtu_vvssml(vrgout_ptr_k6_s1, 0, 0, vmx_s1, vl) ;
	  __vr vrgout_ptr_k6_s2 = _vel_vaddsl_vsvl(4*6*gOutHeight*gOutWidth, vrgout_ptr_k0_s2, vl) ;
	  __vr vrgout_k6_s2 = _vel_vgtu_vvssml(vrgout_ptr_k6_s2, 0, 0, vmx_s2, vl) ;
	  __vr vrgout_ptr_k6_s3 = _vel_vaddsl_vsvl(4*6*gOutHeight*gOutWidth, vrgout_ptr_k0_s3, vl) ;
	  __vr vrgout_k6_s3 = _vel_vgtu_vvssml(vrgout_ptr_k6_s3, 0, 0, vmx_s3, vl) ;
	  __vr vrgout_ptr_k6_s4 = _vel_vaddsl_vsvl(4*6*gOutHeight*gOutWidth, vrgout_ptr_k0_s4, vl) ;
	  __vr vrgout_k6_s4 = _vel_vgtu_vvssml(vrgout_ptr_k6_s4, 0, 0, vmx_s4, vl) ;
	  VFMAD_C4(vrgout_k6_s0, vmx_s0, 6, r, 0) ;
	  VFMAD_C4(vrgout_k6_s1, vmx_s1, 6, r, 1) ;
	  VFMAD_C4(vrgout_k6_s2, vmx_s2, 6, r, 2) ;
	  VFMAD_C4(vrgout_k6_s3, vmx_s3, 6, r, 3) ;
	  VFMAD_C4(vrgout_k6_s4, vmx_s4, 6, r, 4) ;

	  __vr vrgout_ptr_k7_s0 = _vel_vaddsl_vsvl(4*7*gOutHeight*gOutWidth, vrgout_ptr_k0_s0, vl) ;
	  __vr vrgout_k7_s0 = _vel_vgtu_vvssml(vrgout_ptr_k7_s0, 0, 0, vmx_s0, vl) ;
	  __vr vrgout_ptr_k7_s1 = _vel_vaddsl_vsvl(4*7*gOutHeight*gOutWidth, vrgout_ptr_k0_s1, vl) ;
	  __vr vrgout_k7_s1 = _vel_vgtu_vvssml(vrgout_ptr_k7_s1, 0, 0, vmx_s1, vl) ;
	  __vr vrgout_ptr_k7_s2 = _vel_vaddsl_vsvl(4*7*gOutHeight*gOutWidth, vrgout_ptr_k0_s2, vl) ;
	  __vr vrgout_k7_s2 = _vel_vgtu_vvssml(vrgout_ptr_k7_s2, 0, 0, vmx_s2, vl) ;
	  __vr vrgout_ptr_k7_s3 = _vel_vaddsl_vsvl(4*7*gOutHeight*gOutWidth, vrgout_ptr_k0_s3, vl) ;
	  __vr vrgout_k7_s3 = _vel_vgtu_vvssml(vrgout_ptr_k7_s3, 0, 0, vmx_s3, vl) ;
	  __vr vrgout_ptr_k7_s4 = _vel_vaddsl_vsvl(4*7*gOutHeight*gOutWidth, vrgout_ptr_k0_s4, vl) ;
	  __vr vrgout_k7_s4 = _vel_vgtu_vvssml(vrgout_ptr_k7_s4, 0, 0, vmx_s4, vl) ;
	  VFMAD_C4(vrgout_k7_s0, vmx_s0, 7, r, 0) ;
	  VFMAD_C4(vrgout_k7_s1, vmx_s1, 7, r, 1) ;
	  VFMAD_C4(vrgout_k7_s2, vmx_s2, 7, r, 2) ;
	  VFMAD_C4(vrgout_k7_s3, vmx_s3, 7, r, 3) ;
	  VFMAD_C4(vrgout_k7_s4, vmx_s4, 7, r, 4) ;

#undef VFMAD_C4
	} // gOutChannel

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

      __vr vrj_s0 = _vel_vaddsl_vsvl(padWidth-0*dilationWidth, vrw, vl) ;
      __vr vrj_s1 = _vel_vaddsl_vsvl(padWidth-1*dilationWidth, vrw, vl) ;
      __vr vrj_s2 = _vel_vaddsl_vsvl(padWidth-2*dilationWidth, vrw, vl) ;
      __vr vrj_s3 = _vel_vaddsl_vsvl(padWidth-3*dilationWidth, vrw, vl) ;
      __vr vrj_s4 = _vel_vaddsl_vsvl(padWidth-4*dilationWidth, vrw, vl) ;

      __vr vrx_s0 = _vel_vdivsl_vvsl(vrj_s0, strideWidth, vl) ;
      __vr vrx_s1 = _vel_vdivsl_vvsl(vrj_s1, strideWidth, vl) ;
      __vr vrx_s2 = _vel_vdivsl_vvsl(vrj_s2, strideWidth, vl) ;
      __vr vrx_s3 = _vel_vdivsl_vvsl(vrj_s3, strideWidth, vl) ;
      __vr vrx_s4 = _vel_vdivsl_vvsl(vrj_s4, strideWidth, vl) ;

      __vm256 vmx0_s0 =  _vel_vfmkleq_mvl(_vel_vcmpsl_vvvl(vrj_s0, _vel_vmulsl_vsvl(strideWidth, vrx_s0, vl), vl), vl) ;
      __vm256 vmx1_s0 =  _vel_vfmklge_mvl(vrx_s0, vl) ;
      __vm256 vmx2_s0 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutWidth,vrx_s0, vl), vl) ;
      __vm256 vmx_s0 = _vel_andm_mmm(_vel_andm_mmm(vmx0_s0, vmx1_s0), vmx2_s0) ;

      __vm256 vmx0_s1 =  _vel_vfmkleq_mvl(_vel_vcmpsl_vvvl(vrj_s1, _vel_vmulsl_vsvl(strideWidth, vrx_s1, vl), vl), vl) ;
      __vm256 vmx1_s1 =  _vel_vfmklge_mvl(vrx_s1, vl) ;
      __vm256 vmx2_s1 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutWidth,vrx_s1, vl), vl) ;
      __vm256 vmx_s1 = _vel_andm_mmm(_vel_andm_mmm(vmx0_s1, vmx1_s1), vmx2_s1) ;

      __vm256 vmx0_s2 =  _vel_vfmkleq_mvl(_vel_vcmpsl_vvvl(vrj_s2, _vel_vmulsl_vsvl(strideWidth, vrx_s2, vl), vl), vl) ;
      __vm256 vmx1_s2 =  _vel_vfmklge_mvl(vrx_s2, vl) ;
      __vm256 vmx2_s2 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutWidth,vrx_s2, vl), vl) ;
      __vm256 vmx_s2 = _vel_andm_mmm(_vel_andm_mmm(vmx0_s2, vmx1_s2), vmx2_s2) ;

      __vm256 vmx0_s3 =  _vel_vfmkleq_mvl(_vel_vcmpsl_vvvl(vrj_s3, _vel_vmulsl_vsvl(strideWidth, vrx_s3, vl), vl), vl) ;
      __vm256 vmx1_s3 =  _vel_vfmklge_mvl(vrx_s3, vl) ;
      __vm256 vmx2_s3 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutWidth,vrx_s3, vl), vl) ;
      __vm256 vmx_s3 = _vel_andm_mmm(_vel_andm_mmm(vmx0_s3, vmx1_s3), vmx2_s3) ;

      __vm256 vmx0_s4 =  _vel_vfmkleq_mvl(_vel_vcmpsl_vvvl(vrj_s4, _vel_vmulsl_vsvl(strideWidth, vrx_s4, vl), vl), vl) ;
      __vm256 vmx1_s4 =  _vel_vfmklge_mvl(vrx_s4, vl) ;
      __vm256 vmx2_s4 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutWidth,vrx_s4, vl), vl) ;
      __vm256 vmx_s4 = _vel_andm_mmm(_vel_andm_mmm(vmx0_s4, vmx1_s4), vmx2_s4) ;

      for (int64_t r=0; r<kernHeight; r++) {
	int64_t i = h - r * dilationHeight + padHeight ;
	int64_t y = i/strideHeight;
	if ( y*strideHeight != i || y < 0 || gOutHeight <= y)  continue ;

	int64_t k=0;
	if( (gOutChannelGroup & 0x01)  == 1 ) {
	  int64_t gOutIndex    = gOutGroupOffset + ((n * gOutChannel + k) * gOutHeight) * gOutWidth ;
	  const float *pKerValue = pKernel + kernGroupOffset + ((k * gInChannelGroup + c) * kernHeight) * kernWidth;

	  __vr vrgout_ptr_k0_s0 = _vel_vsfa_vvssl(_vel_vaddsl_vsvl(gOutWidth*y, vrx_s0, vl),
					     2,
					     (unsigned long)(pGOut+gOutIndex), vl) ;
	  __vr vrgout_k0_s0 = _vel_vgtu_vvssml(vrgout_ptr_k0_s0, 0, 0, vmx_s0, vl) ;
	  __vr vrgout_ptr_k0_s1 = _vel_vsfa_vvssl(_vel_vaddsl_vsvl(gOutWidth*y, vrx_s1, vl),
					     2,
					     (unsigned long)(pGOut+gOutIndex), vl) ;
	  __vr vrgout_k0_s1 = _vel_vgtu_vvssml(vrgout_ptr_k0_s1, 0, 0, vmx_s1, vl) ;
	  __vr vrgout_ptr_k0_s2 = _vel_vsfa_vvssl(_vel_vaddsl_vsvl(gOutWidth*y, vrx_s2, vl),
					     2,
					     (unsigned long)(pGOut+gOutIndex), vl) ;
	  __vr vrgout_k0_s2 = _vel_vgtu_vvssml(vrgout_ptr_k0_s2, 0, 0, vmx_s2, vl) ;
	  __vr vrgout_ptr_k0_s3 = _vel_vsfa_vvssl(_vel_vaddsl_vsvl(gOutWidth*y, vrx_s3, vl),
					     2,
					     (unsigned long)(pGOut+gOutIndex), vl) ;
	  __vr vrgout_k0_s3 = _vel_vgtu_vvssml(vrgout_ptr_k0_s3, 0, 0, vmx_s3, vl) ;
	  __vr vrgout_ptr_k0_s4 = _vel_vsfa_vvssl(_vel_vaddsl_vsvl(gOutWidth*y, vrx_s4, vl),
					     2,
					     (unsigned long)(pGOut+gOutIndex), vl) ;
	  __vr vrgout_k0_s4 = _vel_vgtu_vvssml(vrgout_ptr_k0_s4, 0, 0, vmx_s4, vl) ;

#define VFMAD_C8(VRGOUT, VM, K, R, S)	{												\
	    const uint64_t kerValue01 = _vel_pack_f32p(pKerValue + (((K)*gInChannelGroup + 0) * kernHeight +(R)) * kernWidth + (S),	\
						       pKerValue + (((K)*gInChannelGroup + 1) * kernHeight +(R)) * kernWidth + (S)) ;	\
	    const uint64_t kerValue23 = _vel_pack_f32p(pKerValue + (((K)*gInChannelGroup + 2) * kernHeight +(R)) * kernWidth + (S),	\
						       pKerValue + (((K)*gInChannelGroup + 3) * kernHeight +(R)) * kernWidth + (S)) ;	\
	    const uint64_t kerValue45 = _vel_pack_f32p(pKerValue + (((K)*gInChannelGroup + 4) * kernHeight +(R)) * kernWidth + (S),	\
						       pKerValue + (((K)*gInChannelGroup + 5) * kernHeight +(R)) * kernWidth + (S)) ;	\
	    const uint64_t kerValue67 = _vel_pack_f32p(pKerValue + (((K)*gInChannelGroup + 6) * kernHeight +(R)) * kernWidth + (S),	\
						       pKerValue + (((K)*gInChannelGroup + 7) * kernHeight +(R)) * kernWidth + (S)) ;	\
	    VRGOUT = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), VRGOUT, VM, vl) ;	\
	    __vr vrgoutP = _vel_vshf_vvvsl(VRGOUT, VRGOUT, VE_VSHUFFLE_YUZU, vl) ;	\
	    vrsum01 = _vel_pvfmad_vvsvl(vrsum01, kerValue01, vrgoutP, vl) ;		\
	    vrsum23 = _vel_pvfmad_vvsvl(vrsum23, kerValue23, vrgoutP, vl) ;		\
	    vrsum45 = _vel_pvfmad_vvsvl(vrsum45, kerValue45, vrgoutP, vl) ;		\
	    vrsum67 = _vel_pvfmad_vvsvl(vrsum67, kerValue67, vrgoutP, vl) ;		\
}
	  VFMAD_C8(vrgout_k0_s0, vmx_s0, 0, r, 0) ;
	  VFMAD_C8(vrgout_k0_s1, vmx_s1, 0, r, 1) ;
	  VFMAD_C8(vrgout_k0_s2, vmx_s2, 0, r, 2) ;
	  VFMAD_C8(vrgout_k0_s3, vmx_s3, 0, r, 3) ;
	  VFMAD_C8(vrgout_k0_s4, vmx_s4, 0, r, 4) ;

	  k+=1 ;
	}
	if( ( (gOutChannelGroup >> 1) & 0x01)  == 1 ) {
	  int64_t gOutIndex    = gOutGroupOffset + ((n * gOutChannel + k) * gOutHeight) * gOutWidth ;
	  const float *pKerValue = pKernel + kernGroupOffset + ((k * gInChannelGroup + c) * kernHeight) * kernWidth;

	  __vr vrgout_ptr_k0_s0 = _vel_vsfa_vvssl(_vel_vaddsl_vsvl(gOutWidth*y, vrx_s0, vl),
					     2,
					     (unsigned long)(pGOut+gOutIndex), vl) ;
	  __vr vrgout_k0_s0 = _vel_vgtu_vvssml(vrgout_ptr_k0_s0, 0, 0, vmx_s0, vl) ;
	  __vr vrgout_ptr_k0_s1 = _vel_vsfa_vvssl(_vel_vaddsl_vsvl(gOutWidth*y, vrx_s1, vl),
					     2,
					     (unsigned long)(pGOut+gOutIndex), vl) ;
	  __vr vrgout_k0_s1 = _vel_vgtu_vvssml(vrgout_ptr_k0_s1, 0, 0, vmx_s1, vl) ;
	  __vr vrgout_ptr_k0_s2 = _vel_vsfa_vvssl(_vel_vaddsl_vsvl(gOutWidth*y, vrx_s2, vl),
					     2,
					     (unsigned long)(pGOut+gOutIndex), vl) ;
	  __vr vrgout_k0_s2 = _vel_vgtu_vvssml(vrgout_ptr_k0_s2, 0, 0, vmx_s2, vl) ;
	  __vr vrgout_ptr_k0_s3 = _vel_vsfa_vvssl(_vel_vaddsl_vsvl(gOutWidth*y, vrx_s3, vl),
					     2,
					     (unsigned long)(pGOut+gOutIndex), vl) ;
	  __vr vrgout_k0_s3 = _vel_vgtu_vvssml(vrgout_ptr_k0_s3, 0, 0, vmx_s3, vl) ;
	  __vr vrgout_ptr_k0_s4 = _vel_vsfa_vvssl(_vel_vaddsl_vsvl(gOutWidth*y, vrx_s4, vl),
					     2,
					     (unsigned long)(pGOut+gOutIndex), vl) ;
	  __vr vrgout_k0_s4 = _vel_vgtu_vvssml(vrgout_ptr_k0_s4, 0, 0, vmx_s4, vl) ;
	  VFMAD_C8(vrgout_k0_s0, vmx_s0, 0, r, 0) ;
	  VFMAD_C8(vrgout_k0_s1, vmx_s1, 0, r, 1) ;
	  VFMAD_C8(vrgout_k0_s2, vmx_s2, 0, r, 2) ;
	  VFMAD_C8(vrgout_k0_s3, vmx_s3, 0, r, 3) ;
	  VFMAD_C8(vrgout_k0_s4, vmx_s4, 0, r, 4) ;

	  __vr vrgout_ptr_k1_s0 = _vel_vaddsl_vsvl(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_s0, vl) ;
	  __vr vrgout_k1_s0 = _vel_vgtu_vvssml(vrgout_ptr_k1_s0, 0, 0, vmx_s0, vl) ;
	  __vr vrgout_ptr_k1_s1 = _vel_vaddsl_vsvl(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_s1, vl) ;
	  __vr vrgout_k1_s1 = _vel_vgtu_vvssml(vrgout_ptr_k1_s1, 0, 0, vmx_s1, vl) ;
	  __vr vrgout_ptr_k1_s2 = _vel_vaddsl_vsvl(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_s2, vl) ;
	  __vr vrgout_k1_s2 = _vel_vgtu_vvssml(vrgout_ptr_k1_s2, 0, 0, vmx_s2, vl) ;
	  __vr vrgout_ptr_k1_s3 = _vel_vaddsl_vsvl(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_s3, vl) ;
	  __vr vrgout_k1_s3 = _vel_vgtu_vvssml(vrgout_ptr_k1_s3, 0, 0, vmx_s3, vl) ;
	  __vr vrgout_ptr_k1_s4 = _vel_vaddsl_vsvl(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_s4, vl) ;
	  __vr vrgout_k1_s4 = _vel_vgtu_vvssml(vrgout_ptr_k1_s4, 0, 0, vmx_s4, vl) ;
	  VFMAD_C8(vrgout_k1_s0, vmx_s0, 1, r, 0) ;
	  VFMAD_C8(vrgout_k1_s1, vmx_s1, 1, r, 1) ;
	  VFMAD_C8(vrgout_k1_s2, vmx_s2, 1, r, 2) ;
	  VFMAD_C8(vrgout_k1_s3, vmx_s3, 1, r, 3) ;
	  VFMAD_C8(vrgout_k1_s4, vmx_s4, 1, r, 4) ;

	  k+=2 ;
	}
	if( ( (gOutChannelGroup >> 2) & 0x01)  == 1 ) {
	  int64_t gOutIndex    = gOutGroupOffset + ((n * gOutChannel + k) * gOutHeight) * gOutWidth ;
	  const float *pKerValue = pKernel + kernGroupOffset + ((k * gInChannelGroup + c) * kernHeight) * kernWidth;

	  __vr vrgout_ptr_k0_s0 = _vel_vsfa_vvssl(_vel_vaddsl_vsvl(gOutWidth*y, vrx_s0, vl),
					     2,
					     (unsigned long)(pGOut+gOutIndex), vl) ;
	  __vr vrgout_k0_s0 = _vel_vgtu_vvssml(vrgout_ptr_k0_s0, 0, 0, vmx_s0, vl) ;
	  __vr vrgout_ptr_k0_s1 = _vel_vsfa_vvssl(_vel_vaddsl_vsvl(gOutWidth*y, vrx_s1, vl),
					     2,
					     (unsigned long)(pGOut+gOutIndex), vl) ;
	  __vr vrgout_k0_s1 = _vel_vgtu_vvssml(vrgout_ptr_k0_s1, 0, 0, vmx_s1, vl) ;
	  __vr vrgout_ptr_k0_s2 = _vel_vsfa_vvssl(_vel_vaddsl_vsvl(gOutWidth*y, vrx_s2, vl),
					     2,
					     (unsigned long)(pGOut+gOutIndex), vl) ;
	  __vr vrgout_k0_s2 = _vel_vgtu_vvssml(vrgout_ptr_k0_s2, 0, 0, vmx_s2, vl) ;
	  __vr vrgout_ptr_k0_s3 = _vel_vsfa_vvssl(_vel_vaddsl_vsvl(gOutWidth*y, vrx_s3, vl),
					     2,
					     (unsigned long)(pGOut+gOutIndex), vl) ;
	  __vr vrgout_k0_s3 = _vel_vgtu_vvssml(vrgout_ptr_k0_s3, 0, 0, vmx_s3, vl) ;
	  __vr vrgout_ptr_k0_s4 = _vel_vsfa_vvssl(_vel_vaddsl_vsvl(gOutWidth*y, vrx_s4, vl),
					     2,
					     (unsigned long)(pGOut+gOutIndex), vl) ;
	  __vr vrgout_k0_s4 = _vel_vgtu_vvssml(vrgout_ptr_k0_s4, 0, 0, vmx_s4, vl) ;
	  VFMAD_C8(vrgout_k0_s0, vmx_s0, 0, r, 0) ;
	  VFMAD_C8(vrgout_k0_s1, vmx_s1, 0, r, 1) ;
	  VFMAD_C8(vrgout_k0_s2, vmx_s2, 0, r, 2) ;
	  VFMAD_C8(vrgout_k0_s3, vmx_s3, 0, r, 3) ;
	  VFMAD_C8(vrgout_k0_s4, vmx_s4, 0, r, 4) ;

	  __vr vrgout_ptr_k1_s0 = _vel_vaddsl_vsvl(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_s0, vl) ;
	  __vr vrgout_k1_s0 = _vel_vgtu_vvssml(vrgout_ptr_k1_s0, 0, 0, vmx_s0, vl) ;
	  __vr vrgout_ptr_k1_s1 = _vel_vaddsl_vsvl(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_s1, vl) ;
	  __vr vrgout_k1_s1 = _vel_vgtu_vvssml(vrgout_ptr_k1_s1, 0, 0, vmx_s1, vl) ;
	  __vr vrgout_ptr_k1_s2 = _vel_vaddsl_vsvl(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_s2, vl) ;
	  __vr vrgout_k1_s2 = _vel_vgtu_vvssml(vrgout_ptr_k1_s2, 0, 0, vmx_s2, vl) ;
	  __vr vrgout_ptr_k1_s3 = _vel_vaddsl_vsvl(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_s3, vl) ;
	  __vr vrgout_k1_s3 = _vel_vgtu_vvssml(vrgout_ptr_k1_s3, 0, 0, vmx_s3, vl) ;
	  __vr vrgout_ptr_k1_s4 = _vel_vaddsl_vsvl(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_s4, vl) ;
	  __vr vrgout_k1_s4 = _vel_vgtu_vvssml(vrgout_ptr_k1_s4, 0, 0, vmx_s4, vl) ;
	  VFMAD_C8(vrgout_k1_s0, vmx_s0, 1, r, 0) ;
	  VFMAD_C8(vrgout_k1_s1, vmx_s1, 1, r, 1) ;
	  VFMAD_C8(vrgout_k1_s2, vmx_s2, 1, r, 2) ;
	  VFMAD_C8(vrgout_k1_s3, vmx_s3, 1, r, 3) ;
	  VFMAD_C8(vrgout_k1_s4, vmx_s4, 1, r, 4) ;

	  __vr vrgout_ptr_k2_s0 = _vel_vaddsl_vsvl(4*2*gOutHeight*gOutWidth, vrgout_ptr_k0_s0, vl) ;
	  __vr vrgout_k2_s0 = _vel_vgtu_vvssml(vrgout_ptr_k2_s0, 0, 0, vmx_s0, vl) ;
	  __vr vrgout_ptr_k2_s1 = _vel_vaddsl_vsvl(4*2*gOutHeight*gOutWidth, vrgout_ptr_k0_s1, vl) ;
	  __vr vrgout_k2_s1 = _vel_vgtu_vvssml(vrgout_ptr_k2_s1, 0, 0, vmx_s1, vl) ;
	  __vr vrgout_ptr_k2_s2 = _vel_vaddsl_vsvl(4*2*gOutHeight*gOutWidth, vrgout_ptr_k0_s2, vl) ;
	  __vr vrgout_k2_s2 = _vel_vgtu_vvssml(vrgout_ptr_k2_s2, 0, 0, vmx_s2, vl) ;
	  __vr vrgout_ptr_k2_s3 = _vel_vaddsl_vsvl(4*2*gOutHeight*gOutWidth, vrgout_ptr_k0_s3, vl) ;
	  __vr vrgout_k2_s3 = _vel_vgtu_vvssml(vrgout_ptr_k2_s3, 0, 0, vmx_s3, vl) ;
	  __vr vrgout_ptr_k2_s4 = _vel_vaddsl_vsvl(4*2*gOutHeight*gOutWidth, vrgout_ptr_k0_s4, vl) ;
	  __vr vrgout_k2_s4 = _vel_vgtu_vvssml(vrgout_ptr_k2_s4, 0, 0, vmx_s4, vl) ;
	  VFMAD_C8(vrgout_k2_s0, vmx_s0, 2, r, 0) ;
	  VFMAD_C8(vrgout_k2_s1, vmx_s1, 2, r, 1) ;
	  VFMAD_C8(vrgout_k2_s2, vmx_s2, 2, r, 2) ;
	  VFMAD_C8(vrgout_k2_s3, vmx_s3, 2, r, 3) ;
	  VFMAD_C8(vrgout_k2_s4, vmx_s4, 2, r, 4) ;

	  __vr vrgout_ptr_k3_s0 = _vel_vaddsl_vsvl(4*3*gOutHeight*gOutWidth, vrgout_ptr_k0_s0, vl) ;
	  __vr vrgout_k3_s0 = _vel_vgtu_vvssml(vrgout_ptr_k3_s0, 0, 0, vmx_s0, vl) ;
	  __vr vrgout_ptr_k3_s1 = _vel_vaddsl_vsvl(4*3*gOutHeight*gOutWidth, vrgout_ptr_k0_s1, vl) ;
	  __vr vrgout_k3_s1 = _vel_vgtu_vvssml(vrgout_ptr_k3_s1, 0, 0, vmx_s1, vl) ;
	  __vr vrgout_ptr_k3_s2 = _vel_vaddsl_vsvl(4*3*gOutHeight*gOutWidth, vrgout_ptr_k0_s2, vl) ;
	  __vr vrgout_k3_s2 = _vel_vgtu_vvssml(vrgout_ptr_k3_s2, 0, 0, vmx_s2, vl) ;
	  __vr vrgout_ptr_k3_s3 = _vel_vaddsl_vsvl(4*3*gOutHeight*gOutWidth, vrgout_ptr_k0_s3, vl) ;
	  __vr vrgout_k3_s3 = _vel_vgtu_vvssml(vrgout_ptr_k3_s3, 0, 0, vmx_s3, vl) ;
	  __vr vrgout_ptr_k3_s4 = _vel_vaddsl_vsvl(4*3*gOutHeight*gOutWidth, vrgout_ptr_k0_s4, vl) ;
	  __vr vrgout_k3_s4 = _vel_vgtu_vvssml(vrgout_ptr_k3_s4, 0, 0, vmx_s4, vl) ;
	  VFMAD_C8(vrgout_k3_s0, vmx_s0, 3, r, 0) ;
	  VFMAD_C8(vrgout_k3_s1, vmx_s1, 3, r, 1) ;
	  VFMAD_C8(vrgout_k3_s2, vmx_s2, 3, r, 2) ;
	  VFMAD_C8(vrgout_k3_s3, vmx_s3, 3, r, 3) ;
	  VFMAD_C8(vrgout_k3_s4, vmx_s4, 3, r, 4) ;

	  k+=4 ;
	}
	for ( ; k<gOutChannelGroup; k+=8 ) {
	  int64_t gOutIndex    = gOutGroupOffset + ((n * gOutChannel + k) * gOutHeight) * gOutWidth ;
	  const float *pKerValue = pKernel + kernGroupOffset + ((k * gInChannelGroup + c) * kernHeight) * kernWidth;

	  __vr vrgout_ptr_k0_s0 = _vel_vsfa_vvssl(_vel_vaddsl_vsvl(gOutWidth*y, vrx_s0, vl),
					     2,
					     (unsigned long)(pGOut+gOutIndex), vl) ;
	  __vr vrgout_k0_s0 = _vel_vgtu_vvssml(vrgout_ptr_k0_s0, 0, 0, vmx_s0, vl) ;
	  __vr vrgout_ptr_k0_s1 = _vel_vsfa_vvssl(_vel_vaddsl_vsvl(gOutWidth*y, vrx_s1, vl),
					     2,
					     (unsigned long)(pGOut+gOutIndex), vl) ;
	  __vr vrgout_k0_s1 = _vel_vgtu_vvssml(vrgout_ptr_k0_s1, 0, 0, vmx_s1, vl) ;
	  __vr vrgout_ptr_k0_s2 = _vel_vsfa_vvssl(_vel_vaddsl_vsvl(gOutWidth*y, vrx_s2, vl),
					     2,
					     (unsigned long)(pGOut+gOutIndex), vl) ;
	  __vr vrgout_k0_s2 = _vel_vgtu_vvssml(vrgout_ptr_k0_s2, 0, 0, vmx_s2, vl) ;
	  __vr vrgout_ptr_k0_s3 = _vel_vsfa_vvssl(_vel_vaddsl_vsvl(gOutWidth*y, vrx_s3, vl),
					     2,
					     (unsigned long)(pGOut+gOutIndex), vl) ;
	  __vr vrgout_k0_s3 = _vel_vgtu_vvssml(vrgout_ptr_k0_s3, 0, 0, vmx_s3, vl) ;
	  __vr vrgout_ptr_k0_s4 = _vel_vsfa_vvssl(_vel_vaddsl_vsvl(gOutWidth*y, vrx_s4, vl),
					     2,
					     (unsigned long)(pGOut+gOutIndex), vl) ;
	  __vr vrgout_k0_s4 = _vel_vgtu_vvssml(vrgout_ptr_k0_s4, 0, 0, vmx_s4, vl) ;
	  VFMAD_C8(vrgout_k0_s0, vmx_s0, 0, r, 0) ;
	  VFMAD_C8(vrgout_k0_s1, vmx_s1, 0, r, 1) ;
	  VFMAD_C8(vrgout_k0_s2, vmx_s2, 0, r, 2) ;
	  VFMAD_C8(vrgout_k0_s3, vmx_s3, 0, r, 3) ;
	  VFMAD_C8(vrgout_k0_s4, vmx_s4, 0, r, 4) ;

	  __vr vrgout_ptr_k1_s0 = _vel_vaddsl_vsvl(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_s0, vl) ;
	  __vr vrgout_k1_s0 = _vel_vgtu_vvssml(vrgout_ptr_k1_s0, 0, 0, vmx_s0, vl) ;
	  __vr vrgout_ptr_k1_s1 = _vel_vaddsl_vsvl(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_s1, vl) ;
	  __vr vrgout_k1_s1 = _vel_vgtu_vvssml(vrgout_ptr_k1_s1, 0, 0, vmx_s1, vl) ;
	  __vr vrgout_ptr_k1_s2 = _vel_vaddsl_vsvl(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_s2, vl) ;
	  __vr vrgout_k1_s2 = _vel_vgtu_vvssml(vrgout_ptr_k1_s2, 0, 0, vmx_s2, vl) ;
	  __vr vrgout_ptr_k1_s3 = _vel_vaddsl_vsvl(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_s3, vl) ;
	  __vr vrgout_k1_s3 = _vel_vgtu_vvssml(vrgout_ptr_k1_s3, 0, 0, vmx_s3, vl) ;
	  __vr vrgout_ptr_k1_s4 = _vel_vaddsl_vsvl(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_s4, vl) ;
	  __vr vrgout_k1_s4 = _vel_vgtu_vvssml(vrgout_ptr_k1_s4, 0, 0, vmx_s4, vl) ;
	  VFMAD_C8(vrgout_k1_s0, vmx_s0, 1, r, 0) ;
	  VFMAD_C8(vrgout_k1_s1, vmx_s1, 1, r, 1) ;
	  VFMAD_C8(vrgout_k1_s2, vmx_s2, 1, r, 2) ;
	  VFMAD_C8(vrgout_k1_s3, vmx_s3, 1, r, 3) ;
	  VFMAD_C8(vrgout_k1_s4, vmx_s4, 1, r, 4) ;

	  __vr vrgout_ptr_k2_s0 = _vel_vaddsl_vsvl(4*2*gOutHeight*gOutWidth, vrgout_ptr_k0_s0, vl) ;
	  __vr vrgout_k2_s0 = _vel_vgtu_vvssml(vrgout_ptr_k2_s0, 0, 0, vmx_s0, vl) ;
	  __vr vrgout_ptr_k2_s1 = _vel_vaddsl_vsvl(4*2*gOutHeight*gOutWidth, vrgout_ptr_k0_s1, vl) ;
	  __vr vrgout_k2_s1 = _vel_vgtu_vvssml(vrgout_ptr_k2_s1, 0, 0, vmx_s1, vl) ;
	  __vr vrgout_ptr_k2_s2 = _vel_vaddsl_vsvl(4*2*gOutHeight*gOutWidth, vrgout_ptr_k0_s2, vl) ;
	  __vr vrgout_k2_s2 = _vel_vgtu_vvssml(vrgout_ptr_k2_s2, 0, 0, vmx_s2, vl) ;
	  __vr vrgout_ptr_k2_s3 = _vel_vaddsl_vsvl(4*2*gOutHeight*gOutWidth, vrgout_ptr_k0_s3, vl) ;
	  __vr vrgout_k2_s3 = _vel_vgtu_vvssml(vrgout_ptr_k2_s3, 0, 0, vmx_s3, vl) ;
	  __vr vrgout_ptr_k2_s4 = _vel_vaddsl_vsvl(4*2*gOutHeight*gOutWidth, vrgout_ptr_k0_s4, vl) ;
	  __vr vrgout_k2_s4 = _vel_vgtu_vvssml(vrgout_ptr_k2_s4, 0, 0, vmx_s4, vl) ;
	  VFMAD_C8(vrgout_k2_s0, vmx_s0, 2, r, 0) ;
	  VFMAD_C8(vrgout_k2_s1, vmx_s1, 2, r, 1) ;
	  VFMAD_C8(vrgout_k2_s2, vmx_s2, 2, r, 2) ;
	  VFMAD_C8(vrgout_k2_s3, vmx_s3, 2, r, 3) ;
	  VFMAD_C8(vrgout_k2_s4, vmx_s4, 2, r, 4) ;

	  __vr vrgout_ptr_k3_s0 = _vel_vaddsl_vsvl(4*3*gOutHeight*gOutWidth, vrgout_ptr_k0_s0, vl) ;
	  __vr vrgout_k3_s0 = _vel_vgtu_vvssml(vrgout_ptr_k3_s0, 0, 0, vmx_s0, vl) ;
	  __vr vrgout_ptr_k3_s1 = _vel_vaddsl_vsvl(4*3*gOutHeight*gOutWidth, vrgout_ptr_k0_s1, vl) ;
	  __vr vrgout_k3_s1 = _vel_vgtu_vvssml(vrgout_ptr_k3_s1, 0, 0, vmx_s1, vl) ;
	  __vr vrgout_ptr_k3_s2 = _vel_vaddsl_vsvl(4*3*gOutHeight*gOutWidth, vrgout_ptr_k0_s2, vl) ;
	  __vr vrgout_k3_s2 = _vel_vgtu_vvssml(vrgout_ptr_k3_s2, 0, 0, vmx_s2, vl) ;
	  __vr vrgout_ptr_k3_s3 = _vel_vaddsl_vsvl(4*3*gOutHeight*gOutWidth, vrgout_ptr_k0_s3, vl) ;
	  __vr vrgout_k3_s3 = _vel_vgtu_vvssml(vrgout_ptr_k3_s3, 0, 0, vmx_s3, vl) ;
	  __vr vrgout_ptr_k3_s4 = _vel_vaddsl_vsvl(4*3*gOutHeight*gOutWidth, vrgout_ptr_k0_s4, vl) ;
	  __vr vrgout_k3_s4 = _vel_vgtu_vvssml(vrgout_ptr_k3_s4, 0, 0, vmx_s4, vl) ;
	  VFMAD_C8(vrgout_k3_s0, vmx_s0, 3, r, 0) ;
	  VFMAD_C8(vrgout_k3_s1, vmx_s1, 3, r, 1) ;
	  VFMAD_C8(vrgout_k3_s2, vmx_s2, 3, r, 2) ;
	  VFMAD_C8(vrgout_k3_s3, vmx_s3, 3, r, 3) ;
	  VFMAD_C8(vrgout_k3_s4, vmx_s4, 3, r, 4) ;

	  __vr vrgout_ptr_k4_s0 = _vel_vaddsl_vsvl(4*4*gOutHeight*gOutWidth, vrgout_ptr_k0_s0, vl) ;
	  __vr vrgout_k4_s0 = _vel_vgtu_vvssml(vrgout_ptr_k4_s0, 0, 0, vmx_s0, vl) ;
	  __vr vrgout_ptr_k4_s1 = _vel_vaddsl_vsvl(4*4*gOutHeight*gOutWidth, vrgout_ptr_k0_s1, vl) ;
	  __vr vrgout_k4_s1 = _vel_vgtu_vvssml(vrgout_ptr_k4_s1, 0, 0, vmx_s1, vl) ;
	  __vr vrgout_ptr_k4_s2 = _vel_vaddsl_vsvl(4*4*gOutHeight*gOutWidth, vrgout_ptr_k0_s2, vl) ;
	  __vr vrgout_k4_s2 = _vel_vgtu_vvssml(vrgout_ptr_k4_s2, 0, 0, vmx_s2, vl) ;
	  __vr vrgout_ptr_k4_s3 = _vel_vaddsl_vsvl(4*4*gOutHeight*gOutWidth, vrgout_ptr_k0_s3, vl) ;
	  __vr vrgout_k4_s3 = _vel_vgtu_vvssml(vrgout_ptr_k4_s3, 0, 0, vmx_s3, vl) ;
	  __vr vrgout_ptr_k4_s4 = _vel_vaddsl_vsvl(4*4*gOutHeight*gOutWidth, vrgout_ptr_k0_s4, vl) ;
	  __vr vrgout_k4_s4 = _vel_vgtu_vvssml(vrgout_ptr_k4_s4, 0, 0, vmx_s4, vl) ;
	  VFMAD_C8(vrgout_k4_s0, vmx_s0, 4, r, 0) ;
	  VFMAD_C8(vrgout_k4_s1, vmx_s1, 4, r, 1) ;
	  VFMAD_C8(vrgout_k4_s2, vmx_s2, 4, r, 2) ;
	  VFMAD_C8(vrgout_k4_s3, vmx_s3, 4, r, 3) ;
	  VFMAD_C8(vrgout_k4_s4, vmx_s4, 4, r, 4) ;

	  __vr vrgout_ptr_k5_s0 = _vel_vaddsl_vsvl(4*5*gOutHeight*gOutWidth, vrgout_ptr_k0_s0, vl) ;
	  __vr vrgout_k5_s0 = _vel_vgtu_vvssml(vrgout_ptr_k5_s0, 0, 0, vmx_s0, vl) ;
	  __vr vrgout_ptr_k5_s1 = _vel_vaddsl_vsvl(4*5*gOutHeight*gOutWidth, vrgout_ptr_k0_s1, vl) ;
	  __vr vrgout_k5_s1 = _vel_vgtu_vvssml(vrgout_ptr_k5_s1, 0, 0, vmx_s1, vl) ;
	  __vr vrgout_ptr_k5_s2 = _vel_vaddsl_vsvl(4*5*gOutHeight*gOutWidth, vrgout_ptr_k0_s2, vl) ;
	  __vr vrgout_k5_s2 = _vel_vgtu_vvssml(vrgout_ptr_k5_s2, 0, 0, vmx_s2, vl) ;
	  __vr vrgout_ptr_k5_s3 = _vel_vaddsl_vsvl(4*5*gOutHeight*gOutWidth, vrgout_ptr_k0_s3, vl) ;
	  __vr vrgout_k5_s3 = _vel_vgtu_vvssml(vrgout_ptr_k5_s3, 0, 0, vmx_s3, vl) ;
	  __vr vrgout_ptr_k5_s4 = _vel_vaddsl_vsvl(4*5*gOutHeight*gOutWidth, vrgout_ptr_k0_s4, vl) ;
	  __vr vrgout_k5_s4 = _vel_vgtu_vvssml(vrgout_ptr_k5_s4, 0, 0, vmx_s4, vl) ;
	  VFMAD_C8(vrgout_k5_s0, vmx_s0, 5, r, 0) ;
	  VFMAD_C8(vrgout_k5_s1, vmx_s1, 5, r, 1) ;
	  VFMAD_C8(vrgout_k5_s2, vmx_s2, 5, r, 2) ;
	  VFMAD_C8(vrgout_k5_s3, vmx_s3, 5, r, 3) ;
	  VFMAD_C8(vrgout_k5_s4, vmx_s4, 5, r, 4) ;

	  __vr vrgout_ptr_k6_s0 = _vel_vaddsl_vsvl(4*6*gOutHeight*gOutWidth, vrgout_ptr_k0_s0, vl) ;
	  __vr vrgout_k6_s0 = _vel_vgtu_vvssml(vrgout_ptr_k6_s0, 0, 0, vmx_s0, vl) ;
	  __vr vrgout_ptr_k6_s1 = _vel_vaddsl_vsvl(4*6*gOutHeight*gOutWidth, vrgout_ptr_k0_s1, vl) ;
	  __vr vrgout_k6_s1 = _vel_vgtu_vvssml(vrgout_ptr_k6_s1, 0, 0, vmx_s1, vl) ;
	  __vr vrgout_ptr_k6_s2 = _vel_vaddsl_vsvl(4*6*gOutHeight*gOutWidth, vrgout_ptr_k0_s2, vl) ;
	  __vr vrgout_k6_s2 = _vel_vgtu_vvssml(vrgout_ptr_k6_s2, 0, 0, vmx_s2, vl) ;
	  __vr vrgout_ptr_k6_s3 = _vel_vaddsl_vsvl(4*6*gOutHeight*gOutWidth, vrgout_ptr_k0_s3, vl) ;
	  __vr vrgout_k6_s3 = _vel_vgtu_vvssml(vrgout_ptr_k6_s3, 0, 0, vmx_s3, vl) ;
	  __vr vrgout_ptr_k6_s4 = _vel_vaddsl_vsvl(4*6*gOutHeight*gOutWidth, vrgout_ptr_k0_s4, vl) ;
	  __vr vrgout_k6_s4 = _vel_vgtu_vvssml(vrgout_ptr_k6_s4, 0, 0, vmx_s4, vl) ;
	  VFMAD_C8(vrgout_k6_s0, vmx_s0, 6, r, 0) ;
	  VFMAD_C8(vrgout_k6_s1, vmx_s1, 6, r, 1) ;
	  VFMAD_C8(vrgout_k6_s2, vmx_s2, 6, r, 2) ;
	  VFMAD_C8(vrgout_k6_s3, vmx_s3, 6, r, 3) ;
	  VFMAD_C8(vrgout_k6_s4, vmx_s4, 6, r, 4) ;

	  __vr vrgout_ptr_k7_s0 = _vel_vaddsl_vsvl(4*7*gOutHeight*gOutWidth, vrgout_ptr_k0_s0, vl) ;
	  __vr vrgout_k7_s0 = _vel_vgtu_vvssml(vrgout_ptr_k7_s0, 0, 0, vmx_s0, vl) ;
	  __vr vrgout_ptr_k7_s1 = _vel_vaddsl_vsvl(4*7*gOutHeight*gOutWidth, vrgout_ptr_k0_s1, vl) ;
	  __vr vrgout_k7_s1 = _vel_vgtu_vvssml(vrgout_ptr_k7_s1, 0, 0, vmx_s1, vl) ;
	  __vr vrgout_ptr_k7_s2 = _vel_vaddsl_vsvl(4*7*gOutHeight*gOutWidth, vrgout_ptr_k0_s2, vl) ;
	  __vr vrgout_k7_s2 = _vel_vgtu_vvssml(vrgout_ptr_k7_s2, 0, 0, vmx_s2, vl) ;
	  __vr vrgout_ptr_k7_s3 = _vel_vaddsl_vsvl(4*7*gOutHeight*gOutWidth, vrgout_ptr_k0_s3, vl) ;
	  __vr vrgout_k7_s3 = _vel_vgtu_vvssml(vrgout_ptr_k7_s3, 0, 0, vmx_s3, vl) ;
	  __vr vrgout_ptr_k7_s4 = _vel_vaddsl_vsvl(4*7*gOutHeight*gOutWidth, vrgout_ptr_k0_s4, vl) ;
	  __vr vrgout_k7_s4 = _vel_vgtu_vvssml(vrgout_ptr_k7_s4, 0, 0, vmx_s4, vl) ;
	  VFMAD_C8(vrgout_k7_s0, vmx_s0, 7, r, 0) ;
	  VFMAD_C8(vrgout_k7_s1, vmx_s1, 7, r, 1) ;
	  VFMAD_C8(vrgout_k7_s2, vmx_s2, 7, r, 2) ;
	  VFMAD_C8(vrgout_k7_s3, vmx_s3, 7, r, 3) ;
	  VFMAD_C8(vrgout_k7_s4, vmx_s4, 7, r, 4) ;

#undef VFMAD_C8
	} // gOutChannel

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

      __vr vrj_s0 = _vel_vaddsl_vsvl(padWidth-0*dilationWidth, vrw, vl) ;
      __vr vrj_s1 = _vel_vaddsl_vsvl(padWidth-1*dilationWidth, vrw, vl) ;
      __vr vrj_s2 = _vel_vaddsl_vsvl(padWidth-2*dilationWidth, vrw, vl) ;
      __vr vrj_s3 = _vel_vaddsl_vsvl(padWidth-3*dilationWidth, vrw, vl) ;
      __vr vrj_s4 = _vel_vaddsl_vsvl(padWidth-4*dilationWidth, vrw, vl) ;

      __vr vrx_s0 = _vel_vdivsl_vvsl(vrj_s0, strideWidth, vl) ;
      __vr vrx_s1 = _vel_vdivsl_vvsl(vrj_s1, strideWidth, vl) ;
      __vr vrx_s2 = _vel_vdivsl_vvsl(vrj_s2, strideWidth, vl) ;
      __vr vrx_s3 = _vel_vdivsl_vvsl(vrj_s3, strideWidth, vl) ;
      __vr vrx_s4 = _vel_vdivsl_vvsl(vrj_s4, strideWidth, vl) ;

      __vm256 vmx0_s0 =  _vel_vfmkleq_mvl(_vel_vcmpsl_vvvl(vrj_s0, _vel_vmulsl_vsvl(strideWidth, vrx_s0, vl), vl), vl) ;
      __vm256 vmx1_s0 =  _vel_vfmklge_mvl(vrx_s0, vl) ;
      __vm256 vmx2_s0 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutWidth,vrx_s0, vl), vl) ;
      __vm256 vmx_s0 = _vel_andm_mmm(_vel_andm_mmm(vmx0_s0, vmx1_s0), vmx2_s0) ;

      __vm256 vmx0_s1 =  _vel_vfmkleq_mvl(_vel_vcmpsl_vvvl(vrj_s1, _vel_vmulsl_vsvl(strideWidth, vrx_s1, vl), vl), vl) ;
      __vm256 vmx1_s1 =  _vel_vfmklge_mvl(vrx_s1, vl) ;
      __vm256 vmx2_s1 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutWidth,vrx_s1, vl), vl) ;
      __vm256 vmx_s1 = _vel_andm_mmm(_vel_andm_mmm(vmx0_s1, vmx1_s1), vmx2_s1) ;

      __vm256 vmx0_s2 =  _vel_vfmkleq_mvl(_vel_vcmpsl_vvvl(vrj_s2, _vel_vmulsl_vsvl(strideWidth, vrx_s2, vl), vl), vl) ;
      __vm256 vmx1_s2 =  _vel_vfmklge_mvl(vrx_s2, vl) ;
      __vm256 vmx2_s2 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutWidth,vrx_s2, vl), vl) ;
      __vm256 vmx_s2 = _vel_andm_mmm(_vel_andm_mmm(vmx0_s2, vmx1_s2), vmx2_s2) ;

      __vm256 vmx0_s3 =  _vel_vfmkleq_mvl(_vel_vcmpsl_vvvl(vrj_s3, _vel_vmulsl_vsvl(strideWidth, vrx_s3, vl), vl), vl) ;
      __vm256 vmx1_s3 =  _vel_vfmklge_mvl(vrx_s3, vl) ;
      __vm256 vmx2_s3 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutWidth,vrx_s3, vl), vl) ;
      __vm256 vmx_s3 = _vel_andm_mmm(_vel_andm_mmm(vmx0_s3, vmx1_s3), vmx2_s3) ;

      __vm256 vmx0_s4 =  _vel_vfmkleq_mvl(_vel_vcmpsl_vvvl(vrj_s4, _vel_vmulsl_vsvl(strideWidth, vrx_s4, vl), vl), vl) ;
      __vm256 vmx1_s4 =  _vel_vfmklge_mvl(vrx_s4, vl) ;
      __vm256 vmx2_s4 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutWidth,vrx_s4, vl), vl) ;
      __vm256 vmx_s4 = _vel_andm_mmm(_vel_andm_mmm(vmx0_s4, vmx1_s4), vmx2_s4) ;

      for (int64_t r=0; r<kernHeight; r++) {
	int64_t i = h - r * dilationHeight + padHeight ;
	int64_t y = i/strideHeight;
	if ( y*strideHeight != i || y < 0 || gOutHeight <= y)  continue ;

	int64_t k=0;
	if( (gOutChannelGroup & 0x01)  == 1 ) {
	  int64_t gOutIndex    = gOutGroupOffset + ((n * gOutChannel + k) * gOutHeight) * gOutWidth ;
	  const float *pKerValue = pKernel + kernGroupOffset + ((k * gInChannelGroup + c) * kernHeight) * kernWidth;

	  __vr vrgout_ptr_k0_s0 = _vel_vsfa_vvssl(_vel_vaddsl_vsvl(gOutWidth*y, vrx_s0, vl),
					     2,
					     (unsigned long)(pGOut+gOutIndex), vl) ;
	  __vr vrgout_k0_s0 = _vel_vgtu_vvssml(vrgout_ptr_k0_s0, 0, 0, vmx_s0, vl) ;
	  __vr vrgout_ptr_k0_s1 = _vel_vsfa_vvssl(_vel_vaddsl_vsvl(gOutWidth*y, vrx_s1, vl),
					     2,
					     (unsigned long)(pGOut+gOutIndex), vl) ;
	  __vr vrgout_k0_s1 = _vel_vgtu_vvssml(vrgout_ptr_k0_s1, 0, 0, vmx_s1, vl) ;
	  __vr vrgout_ptr_k0_s2 = _vel_vsfa_vvssl(_vel_vaddsl_vsvl(gOutWidth*y, vrx_s2, vl),
					     2,
					     (unsigned long)(pGOut+gOutIndex), vl) ;
	  __vr vrgout_k0_s2 = _vel_vgtu_vvssml(vrgout_ptr_k0_s2, 0, 0, vmx_s2, vl) ;
	  __vr vrgout_ptr_k0_s3 = _vel_vsfa_vvssl(_vel_vaddsl_vsvl(gOutWidth*y, vrx_s3, vl),
					     2,
					     (unsigned long)(pGOut+gOutIndex), vl) ;
	  __vr vrgout_k0_s3 = _vel_vgtu_vvssml(vrgout_ptr_k0_s3, 0, 0, vmx_s3, vl) ;
	  __vr vrgout_ptr_k0_s4 = _vel_vsfa_vvssl(_vel_vaddsl_vsvl(gOutWidth*y, vrx_s4, vl),
					     2,
					     (unsigned long)(pGOut+gOutIndex), vl) ;
	  __vr vrgout_k0_s4 = _vel_vgtu_vvssml(vrgout_ptr_k0_s4, 0, 0, vmx_s4, vl) ;

#define VFMAD_C16(VRGOUT, VM, K, R, S)	{												\
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
	    VRGOUT = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), VRGOUT, VM, vl) ;	\
	    __vr vrgoutP = _vel_vshf_vvvsl(VRGOUT, VRGOUT, VE_VSHUFFLE_YUZU, vl) ;	\
	    vrsum01 = _vel_pvfmad_vvsvl(vrsum01, kerValue01, vrgoutP, vl) ;		\
	    vrsum23 = _vel_pvfmad_vvsvl(vrsum23, kerValue23, vrgoutP, vl) ;		\
	    vrsum45 = _vel_pvfmad_vvsvl(vrsum45, kerValue45, vrgoutP, vl) ;		\
	    vrsum67 = _vel_pvfmad_vvsvl(vrsum67, kerValue67, vrgoutP, vl) ;		\
	    vrsum89 = _vel_pvfmad_vvsvl(vrsum89, kerValue89, vrgoutP, vl) ;		\
	    vrsumAB = _vel_pvfmad_vvsvl(vrsumAB, kerValueAB, vrgoutP, vl) ;		\
	    vrsumCD = _vel_pvfmad_vvsvl(vrsumCD, kerValueCD, vrgoutP, vl) ;		\
	    vrsumEF = _vel_pvfmad_vvsvl(vrsumEF, kerValueEF, vrgoutP, vl) ;		\
}
	  VFMAD_C16(vrgout_k0_s0, vmx_s0, 0, r, 0) ;
	  VFMAD_C16(vrgout_k0_s1, vmx_s1, 0, r, 1) ;
	  VFMAD_C16(vrgout_k0_s2, vmx_s2, 0, r, 2) ;
	  VFMAD_C16(vrgout_k0_s3, vmx_s3, 0, r, 3) ;
	  VFMAD_C16(vrgout_k0_s4, vmx_s4, 0, r, 4) ;

	  k+=1 ;
	}
	if( ( (gOutChannelGroup >> 1) & 0x01)  == 1 ) {
	  int64_t gOutIndex    = gOutGroupOffset + ((n * gOutChannel + k) * gOutHeight) * gOutWidth ;
	  const float *pKerValue = pKernel + kernGroupOffset + ((k * gInChannelGroup + c) * kernHeight) * kernWidth;

	  __vr vrgout_ptr_k0_s0 = _vel_vsfa_vvssl(_vel_vaddsl_vsvl(gOutWidth*y, vrx_s0, vl),
					     2,
					     (unsigned long)(pGOut+gOutIndex), vl) ;
	  __vr vrgout_k0_s0 = _vel_vgtu_vvssml(vrgout_ptr_k0_s0, 0, 0, vmx_s0, vl) ;
	  __vr vrgout_ptr_k0_s1 = _vel_vsfa_vvssl(_vel_vaddsl_vsvl(gOutWidth*y, vrx_s1, vl),
					     2,
					     (unsigned long)(pGOut+gOutIndex), vl) ;
	  __vr vrgout_k0_s1 = _vel_vgtu_vvssml(vrgout_ptr_k0_s1, 0, 0, vmx_s1, vl) ;
	  __vr vrgout_ptr_k0_s2 = _vel_vsfa_vvssl(_vel_vaddsl_vsvl(gOutWidth*y, vrx_s2, vl),
					     2,
					     (unsigned long)(pGOut+gOutIndex), vl) ;
	  __vr vrgout_k0_s2 = _vel_vgtu_vvssml(vrgout_ptr_k0_s2, 0, 0, vmx_s2, vl) ;
	  __vr vrgout_ptr_k0_s3 = _vel_vsfa_vvssl(_vel_vaddsl_vsvl(gOutWidth*y, vrx_s3, vl),
					     2,
					     (unsigned long)(pGOut+gOutIndex), vl) ;
	  __vr vrgout_k0_s3 = _vel_vgtu_vvssml(vrgout_ptr_k0_s3, 0, 0, vmx_s3, vl) ;
	  __vr vrgout_ptr_k0_s4 = _vel_vsfa_vvssl(_vel_vaddsl_vsvl(gOutWidth*y, vrx_s4, vl),
					     2,
					     (unsigned long)(pGOut+gOutIndex), vl) ;
	  __vr vrgout_k0_s4 = _vel_vgtu_vvssml(vrgout_ptr_k0_s4, 0, 0, vmx_s4, vl) ;
	  VFMAD_C16(vrgout_k0_s0, vmx_s0, 0, r, 0) ;
	  VFMAD_C16(vrgout_k0_s1, vmx_s1, 0, r, 1) ;
	  VFMAD_C16(vrgout_k0_s2, vmx_s2, 0, r, 2) ;
	  VFMAD_C16(vrgout_k0_s3, vmx_s3, 0, r, 3) ;
	  VFMAD_C16(vrgout_k0_s4, vmx_s4, 0, r, 4) ;

	  __vr vrgout_ptr_k1_s0 = _vel_vaddsl_vsvl(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_s0, vl) ;
	  __vr vrgout_k1_s0 = _vel_vgtu_vvssml(vrgout_ptr_k1_s0, 0, 0, vmx_s0, vl) ;
	  __vr vrgout_ptr_k1_s1 = _vel_vaddsl_vsvl(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_s1, vl) ;
	  __vr vrgout_k1_s1 = _vel_vgtu_vvssml(vrgout_ptr_k1_s1, 0, 0, vmx_s1, vl) ;
	  __vr vrgout_ptr_k1_s2 = _vel_vaddsl_vsvl(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_s2, vl) ;
	  __vr vrgout_k1_s2 = _vel_vgtu_vvssml(vrgout_ptr_k1_s2, 0, 0, vmx_s2, vl) ;
	  __vr vrgout_ptr_k1_s3 = _vel_vaddsl_vsvl(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_s3, vl) ;
	  __vr vrgout_k1_s3 = _vel_vgtu_vvssml(vrgout_ptr_k1_s3, 0, 0, vmx_s3, vl) ;
	  __vr vrgout_ptr_k1_s4 = _vel_vaddsl_vsvl(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_s4, vl) ;
	  __vr vrgout_k1_s4 = _vel_vgtu_vvssml(vrgout_ptr_k1_s4, 0, 0, vmx_s4, vl) ;
	  VFMAD_C16(vrgout_k1_s0, vmx_s0, 1, r, 0) ;
	  VFMAD_C16(vrgout_k1_s1, vmx_s1, 1, r, 1) ;
	  VFMAD_C16(vrgout_k1_s2, vmx_s2, 1, r, 2) ;
	  VFMAD_C16(vrgout_k1_s3, vmx_s3, 1, r, 3) ;
	  VFMAD_C16(vrgout_k1_s4, vmx_s4, 1, r, 4) ;

	  k+=2 ;
	}
	if( ( (gOutChannelGroup >> 2) & 0x01)  == 1 ) {
	  int64_t gOutIndex    = gOutGroupOffset + ((n * gOutChannel + k) * gOutHeight) * gOutWidth ;
	  const float *pKerValue = pKernel + kernGroupOffset + ((k * gInChannelGroup + c) * kernHeight) * kernWidth;

	  __vr vrgout_ptr_k0_s0 = _vel_vsfa_vvssl(_vel_vaddsl_vsvl(gOutWidth*y, vrx_s0, vl),
					     2,
					     (unsigned long)(pGOut+gOutIndex), vl) ;
	  __vr vrgout_k0_s0 = _vel_vgtu_vvssml(vrgout_ptr_k0_s0, 0, 0, vmx_s0, vl) ;
	  __vr vrgout_ptr_k0_s1 = _vel_vsfa_vvssl(_vel_vaddsl_vsvl(gOutWidth*y, vrx_s1, vl),
					     2,
					     (unsigned long)(pGOut+gOutIndex), vl) ;
	  __vr vrgout_k0_s1 = _vel_vgtu_vvssml(vrgout_ptr_k0_s1, 0, 0, vmx_s1, vl) ;
	  __vr vrgout_ptr_k0_s2 = _vel_vsfa_vvssl(_vel_vaddsl_vsvl(gOutWidth*y, vrx_s2, vl),
					     2,
					     (unsigned long)(pGOut+gOutIndex), vl) ;
	  __vr vrgout_k0_s2 = _vel_vgtu_vvssml(vrgout_ptr_k0_s2, 0, 0, vmx_s2, vl) ;
	  __vr vrgout_ptr_k0_s3 = _vel_vsfa_vvssl(_vel_vaddsl_vsvl(gOutWidth*y, vrx_s3, vl),
					     2,
					     (unsigned long)(pGOut+gOutIndex), vl) ;
	  __vr vrgout_k0_s3 = _vel_vgtu_vvssml(vrgout_ptr_k0_s3, 0, 0, vmx_s3, vl) ;
	  __vr vrgout_ptr_k0_s4 = _vel_vsfa_vvssl(_vel_vaddsl_vsvl(gOutWidth*y, vrx_s4, vl),
					     2,
					     (unsigned long)(pGOut+gOutIndex), vl) ;
	  __vr vrgout_k0_s4 = _vel_vgtu_vvssml(vrgout_ptr_k0_s4, 0, 0, vmx_s4, vl) ;
	  VFMAD_C16(vrgout_k0_s0, vmx_s0, 0, r, 0) ;
	  VFMAD_C16(vrgout_k0_s1, vmx_s1, 0, r, 1) ;
	  VFMAD_C16(vrgout_k0_s2, vmx_s2, 0, r, 2) ;
	  VFMAD_C16(vrgout_k0_s3, vmx_s3, 0, r, 3) ;
	  VFMAD_C16(vrgout_k0_s4, vmx_s4, 0, r, 4) ;

	  __vr vrgout_ptr_k1_s0 = _vel_vaddsl_vsvl(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_s0, vl) ;
	  __vr vrgout_k1_s0 = _vel_vgtu_vvssml(vrgout_ptr_k1_s0, 0, 0, vmx_s0, vl) ;
	  __vr vrgout_ptr_k1_s1 = _vel_vaddsl_vsvl(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_s1, vl) ;
	  __vr vrgout_k1_s1 = _vel_vgtu_vvssml(vrgout_ptr_k1_s1, 0, 0, vmx_s1, vl) ;
	  __vr vrgout_ptr_k1_s2 = _vel_vaddsl_vsvl(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_s2, vl) ;
	  __vr vrgout_k1_s2 = _vel_vgtu_vvssml(vrgout_ptr_k1_s2, 0, 0, vmx_s2, vl) ;
	  __vr vrgout_ptr_k1_s3 = _vel_vaddsl_vsvl(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_s3, vl) ;
	  __vr vrgout_k1_s3 = _vel_vgtu_vvssml(vrgout_ptr_k1_s3, 0, 0, vmx_s3, vl) ;
	  __vr vrgout_ptr_k1_s4 = _vel_vaddsl_vsvl(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_s4, vl) ;
	  __vr vrgout_k1_s4 = _vel_vgtu_vvssml(vrgout_ptr_k1_s4, 0, 0, vmx_s4, vl) ;
	  VFMAD_C16(vrgout_k1_s0, vmx_s0, 1, r, 0) ;
	  VFMAD_C16(vrgout_k1_s1, vmx_s1, 1, r, 1) ;
	  VFMAD_C16(vrgout_k1_s2, vmx_s2, 1, r, 2) ;
	  VFMAD_C16(vrgout_k1_s3, vmx_s3, 1, r, 3) ;
	  VFMAD_C16(vrgout_k1_s4, vmx_s4, 1, r, 4) ;

	  __vr vrgout_ptr_k2_s0 = _vel_vaddsl_vsvl(4*2*gOutHeight*gOutWidth, vrgout_ptr_k0_s0, vl) ;
	  __vr vrgout_k2_s0 = _vel_vgtu_vvssml(vrgout_ptr_k2_s0, 0, 0, vmx_s0, vl) ;
	  __vr vrgout_ptr_k2_s1 = _vel_vaddsl_vsvl(4*2*gOutHeight*gOutWidth, vrgout_ptr_k0_s1, vl) ;
	  __vr vrgout_k2_s1 = _vel_vgtu_vvssml(vrgout_ptr_k2_s1, 0, 0, vmx_s1, vl) ;
	  __vr vrgout_ptr_k2_s2 = _vel_vaddsl_vsvl(4*2*gOutHeight*gOutWidth, vrgout_ptr_k0_s2, vl) ;
	  __vr vrgout_k2_s2 = _vel_vgtu_vvssml(vrgout_ptr_k2_s2, 0, 0, vmx_s2, vl) ;
	  __vr vrgout_ptr_k2_s3 = _vel_vaddsl_vsvl(4*2*gOutHeight*gOutWidth, vrgout_ptr_k0_s3, vl) ;
	  __vr vrgout_k2_s3 = _vel_vgtu_vvssml(vrgout_ptr_k2_s3, 0, 0, vmx_s3, vl) ;
	  __vr vrgout_ptr_k2_s4 = _vel_vaddsl_vsvl(4*2*gOutHeight*gOutWidth, vrgout_ptr_k0_s4, vl) ;
	  __vr vrgout_k2_s4 = _vel_vgtu_vvssml(vrgout_ptr_k2_s4, 0, 0, vmx_s4, vl) ;
	  VFMAD_C16(vrgout_k2_s0, vmx_s0, 2, r, 0) ;
	  VFMAD_C16(vrgout_k2_s1, vmx_s1, 2, r, 1) ;
	  VFMAD_C16(vrgout_k2_s2, vmx_s2, 2, r, 2) ;
	  VFMAD_C16(vrgout_k2_s3, vmx_s3, 2, r, 3) ;
	  VFMAD_C16(vrgout_k2_s4, vmx_s4, 2, r, 4) ;

	  __vr vrgout_ptr_k3_s0 = _vel_vaddsl_vsvl(4*3*gOutHeight*gOutWidth, vrgout_ptr_k0_s0, vl) ;
	  __vr vrgout_k3_s0 = _vel_vgtu_vvssml(vrgout_ptr_k3_s0, 0, 0, vmx_s0, vl) ;
	  __vr vrgout_ptr_k3_s1 = _vel_vaddsl_vsvl(4*3*gOutHeight*gOutWidth, vrgout_ptr_k0_s1, vl) ;
	  __vr vrgout_k3_s1 = _vel_vgtu_vvssml(vrgout_ptr_k3_s1, 0, 0, vmx_s1, vl) ;
	  __vr vrgout_ptr_k3_s2 = _vel_vaddsl_vsvl(4*3*gOutHeight*gOutWidth, vrgout_ptr_k0_s2, vl) ;
	  __vr vrgout_k3_s2 = _vel_vgtu_vvssml(vrgout_ptr_k3_s2, 0, 0, vmx_s2, vl) ;
	  __vr vrgout_ptr_k3_s3 = _vel_vaddsl_vsvl(4*3*gOutHeight*gOutWidth, vrgout_ptr_k0_s3, vl) ;
	  __vr vrgout_k3_s3 = _vel_vgtu_vvssml(vrgout_ptr_k3_s3, 0, 0, vmx_s3, vl) ;
	  __vr vrgout_ptr_k3_s4 = _vel_vaddsl_vsvl(4*3*gOutHeight*gOutWidth, vrgout_ptr_k0_s4, vl) ;
	  __vr vrgout_k3_s4 = _vel_vgtu_vvssml(vrgout_ptr_k3_s4, 0, 0, vmx_s4, vl) ;
	  VFMAD_C16(vrgout_k3_s0, vmx_s0, 3, r, 0) ;
	  VFMAD_C16(vrgout_k3_s1, vmx_s1, 3, r, 1) ;
	  VFMAD_C16(vrgout_k3_s2, vmx_s2, 3, r, 2) ;
	  VFMAD_C16(vrgout_k3_s3, vmx_s3, 3, r, 3) ;
	  VFMAD_C16(vrgout_k3_s4, vmx_s4, 3, r, 4) ;

	  k+=4 ;
	}
	for ( ; k<gOutChannelGroup; k+=8 ) {
	  int64_t gOutIndex    = gOutGroupOffset + ((n * gOutChannel + k) * gOutHeight) * gOutWidth ;
	  const float *pKerValue = pKernel + kernGroupOffset + ((k * gInChannelGroup + c) * kernHeight) * kernWidth;

	  __vr vrgout_ptr_k0_s0 = _vel_vsfa_vvssl(_vel_vaddsl_vsvl(gOutWidth*y, vrx_s0, vl),
					     2,
					     (unsigned long)(pGOut+gOutIndex), vl) ;
	  __vr vrgout_k0_s0 = _vel_vgtu_vvssml(vrgout_ptr_k0_s0, 0, 0, vmx_s0, vl) ;
	  __vr vrgout_ptr_k0_s1 = _vel_vsfa_vvssl(_vel_vaddsl_vsvl(gOutWidth*y, vrx_s1, vl),
					     2,
					     (unsigned long)(pGOut+gOutIndex), vl) ;
	  __vr vrgout_k0_s1 = _vel_vgtu_vvssml(vrgout_ptr_k0_s1, 0, 0, vmx_s1, vl) ;
	  __vr vrgout_ptr_k0_s2 = _vel_vsfa_vvssl(_vel_vaddsl_vsvl(gOutWidth*y, vrx_s2, vl),
					     2,
					     (unsigned long)(pGOut+gOutIndex), vl) ;
	  __vr vrgout_k0_s2 = _vel_vgtu_vvssml(vrgout_ptr_k0_s2, 0, 0, vmx_s2, vl) ;
	  __vr vrgout_ptr_k0_s3 = _vel_vsfa_vvssl(_vel_vaddsl_vsvl(gOutWidth*y, vrx_s3, vl),
					     2,
					     (unsigned long)(pGOut+gOutIndex), vl) ;
	  __vr vrgout_k0_s3 = _vel_vgtu_vvssml(vrgout_ptr_k0_s3, 0, 0, vmx_s3, vl) ;
	  __vr vrgout_ptr_k0_s4 = _vel_vsfa_vvssl(_vel_vaddsl_vsvl(gOutWidth*y, vrx_s4, vl),
					     2,
					     (unsigned long)(pGOut+gOutIndex), vl) ;
	  __vr vrgout_k0_s4 = _vel_vgtu_vvssml(vrgout_ptr_k0_s4, 0, 0, vmx_s4, vl) ;
	  VFMAD_C16(vrgout_k0_s0, vmx_s0, 0, r, 0) ;
	  VFMAD_C16(vrgout_k0_s1, vmx_s1, 0, r, 1) ;
	  VFMAD_C16(vrgout_k0_s2, vmx_s2, 0, r, 2) ;
	  VFMAD_C16(vrgout_k0_s3, vmx_s3, 0, r, 3) ;
	  VFMAD_C16(vrgout_k0_s4, vmx_s4, 0, r, 4) ;

	  __vr vrgout_ptr_k1_s0 = _vel_vaddsl_vsvl(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_s0, vl) ;
	  __vr vrgout_k1_s0 = _vel_vgtu_vvssml(vrgout_ptr_k1_s0, 0, 0, vmx_s0, vl) ;
	  __vr vrgout_ptr_k1_s1 = _vel_vaddsl_vsvl(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_s1, vl) ;
	  __vr vrgout_k1_s1 = _vel_vgtu_vvssml(vrgout_ptr_k1_s1, 0, 0, vmx_s1, vl) ;
	  __vr vrgout_ptr_k1_s2 = _vel_vaddsl_vsvl(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_s2, vl) ;
	  __vr vrgout_k1_s2 = _vel_vgtu_vvssml(vrgout_ptr_k1_s2, 0, 0, vmx_s2, vl) ;
	  __vr vrgout_ptr_k1_s3 = _vel_vaddsl_vsvl(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_s3, vl) ;
	  __vr vrgout_k1_s3 = _vel_vgtu_vvssml(vrgout_ptr_k1_s3, 0, 0, vmx_s3, vl) ;
	  __vr vrgout_ptr_k1_s4 = _vel_vaddsl_vsvl(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_s4, vl) ;
	  __vr vrgout_k1_s4 = _vel_vgtu_vvssml(vrgout_ptr_k1_s4, 0, 0, vmx_s4, vl) ;
	  VFMAD_C16(vrgout_k1_s0, vmx_s0, 1, r, 0) ;
	  VFMAD_C16(vrgout_k1_s1, vmx_s1, 1, r, 1) ;
	  VFMAD_C16(vrgout_k1_s2, vmx_s2, 1, r, 2) ;
	  VFMAD_C16(vrgout_k1_s3, vmx_s3, 1, r, 3) ;
	  VFMAD_C16(vrgout_k1_s4, vmx_s4, 1, r, 4) ;

	  __vr vrgout_ptr_k2_s0 = _vel_vaddsl_vsvl(4*2*gOutHeight*gOutWidth, vrgout_ptr_k0_s0, vl) ;
	  __vr vrgout_k2_s0 = _vel_vgtu_vvssml(vrgout_ptr_k2_s0, 0, 0, vmx_s0, vl) ;
	  __vr vrgout_ptr_k2_s1 = _vel_vaddsl_vsvl(4*2*gOutHeight*gOutWidth, vrgout_ptr_k0_s1, vl) ;
	  __vr vrgout_k2_s1 = _vel_vgtu_vvssml(vrgout_ptr_k2_s1, 0, 0, vmx_s1, vl) ;
	  __vr vrgout_ptr_k2_s2 = _vel_vaddsl_vsvl(4*2*gOutHeight*gOutWidth, vrgout_ptr_k0_s2, vl) ;
	  __vr vrgout_k2_s2 = _vel_vgtu_vvssml(vrgout_ptr_k2_s2, 0, 0, vmx_s2, vl) ;
	  __vr vrgout_ptr_k2_s3 = _vel_vaddsl_vsvl(4*2*gOutHeight*gOutWidth, vrgout_ptr_k0_s3, vl) ;
	  __vr vrgout_k2_s3 = _vel_vgtu_vvssml(vrgout_ptr_k2_s3, 0, 0, vmx_s3, vl) ;
	  __vr vrgout_ptr_k2_s4 = _vel_vaddsl_vsvl(4*2*gOutHeight*gOutWidth, vrgout_ptr_k0_s4, vl) ;
	  __vr vrgout_k2_s4 = _vel_vgtu_vvssml(vrgout_ptr_k2_s4, 0, 0, vmx_s4, vl) ;
	  VFMAD_C16(vrgout_k2_s0, vmx_s0, 2, r, 0) ;
	  VFMAD_C16(vrgout_k2_s1, vmx_s1, 2, r, 1) ;
	  VFMAD_C16(vrgout_k2_s2, vmx_s2, 2, r, 2) ;
	  VFMAD_C16(vrgout_k2_s3, vmx_s3, 2, r, 3) ;
	  VFMAD_C16(vrgout_k2_s4, vmx_s4, 2, r, 4) ;

	  __vr vrgout_ptr_k3_s0 = _vel_vaddsl_vsvl(4*3*gOutHeight*gOutWidth, vrgout_ptr_k0_s0, vl) ;
	  __vr vrgout_k3_s0 = _vel_vgtu_vvssml(vrgout_ptr_k3_s0, 0, 0, vmx_s0, vl) ;
	  __vr vrgout_ptr_k3_s1 = _vel_vaddsl_vsvl(4*3*gOutHeight*gOutWidth, vrgout_ptr_k0_s1, vl) ;
	  __vr vrgout_k3_s1 = _vel_vgtu_vvssml(vrgout_ptr_k3_s1, 0, 0, vmx_s1, vl) ;
	  __vr vrgout_ptr_k3_s2 = _vel_vaddsl_vsvl(4*3*gOutHeight*gOutWidth, vrgout_ptr_k0_s2, vl) ;
	  __vr vrgout_k3_s2 = _vel_vgtu_vvssml(vrgout_ptr_k3_s2, 0, 0, vmx_s2, vl) ;
	  __vr vrgout_ptr_k3_s3 = _vel_vaddsl_vsvl(4*3*gOutHeight*gOutWidth, vrgout_ptr_k0_s3, vl) ;
	  __vr vrgout_k3_s3 = _vel_vgtu_vvssml(vrgout_ptr_k3_s3, 0, 0, vmx_s3, vl) ;
	  __vr vrgout_ptr_k3_s4 = _vel_vaddsl_vsvl(4*3*gOutHeight*gOutWidth, vrgout_ptr_k0_s4, vl) ;
	  __vr vrgout_k3_s4 = _vel_vgtu_vvssml(vrgout_ptr_k3_s4, 0, 0, vmx_s4, vl) ;
	  VFMAD_C16(vrgout_k3_s0, vmx_s0, 3, r, 0) ;
	  VFMAD_C16(vrgout_k3_s1, vmx_s1, 3, r, 1) ;
	  VFMAD_C16(vrgout_k3_s2, vmx_s2, 3, r, 2) ;
	  VFMAD_C16(vrgout_k3_s3, vmx_s3, 3, r, 3) ;
	  VFMAD_C16(vrgout_k3_s4, vmx_s4, 3, r, 4) ;

	  __vr vrgout_ptr_k4_s0 = _vel_vaddsl_vsvl(4*4*gOutHeight*gOutWidth, vrgout_ptr_k0_s0, vl) ;
	  __vr vrgout_k4_s0 = _vel_vgtu_vvssml(vrgout_ptr_k4_s0, 0, 0, vmx_s0, vl) ;
	  __vr vrgout_ptr_k4_s1 = _vel_vaddsl_vsvl(4*4*gOutHeight*gOutWidth, vrgout_ptr_k0_s1, vl) ;
	  __vr vrgout_k4_s1 = _vel_vgtu_vvssml(vrgout_ptr_k4_s1, 0, 0, vmx_s1, vl) ;
	  __vr vrgout_ptr_k4_s2 = _vel_vaddsl_vsvl(4*4*gOutHeight*gOutWidth, vrgout_ptr_k0_s2, vl) ;
	  __vr vrgout_k4_s2 = _vel_vgtu_vvssml(vrgout_ptr_k4_s2, 0, 0, vmx_s2, vl) ;
	  __vr vrgout_ptr_k4_s3 = _vel_vaddsl_vsvl(4*4*gOutHeight*gOutWidth, vrgout_ptr_k0_s3, vl) ;
	  __vr vrgout_k4_s3 = _vel_vgtu_vvssml(vrgout_ptr_k4_s3, 0, 0, vmx_s3, vl) ;
	  __vr vrgout_ptr_k4_s4 = _vel_vaddsl_vsvl(4*4*gOutHeight*gOutWidth, vrgout_ptr_k0_s4, vl) ;
	  __vr vrgout_k4_s4 = _vel_vgtu_vvssml(vrgout_ptr_k4_s4, 0, 0, vmx_s4, vl) ;
	  VFMAD_C16(vrgout_k4_s0, vmx_s0, 4, r, 0) ;
	  VFMAD_C16(vrgout_k4_s1, vmx_s1, 4, r, 1) ;
	  VFMAD_C16(vrgout_k4_s2, vmx_s2, 4, r, 2) ;
	  VFMAD_C16(vrgout_k4_s3, vmx_s3, 4, r, 3) ;
	  VFMAD_C16(vrgout_k4_s4, vmx_s4, 4, r, 4) ;

	  __vr vrgout_ptr_k5_s0 = _vel_vaddsl_vsvl(4*5*gOutHeight*gOutWidth, vrgout_ptr_k0_s0, vl) ;
	  __vr vrgout_k5_s0 = _vel_vgtu_vvssml(vrgout_ptr_k5_s0, 0, 0, vmx_s0, vl) ;
	  __vr vrgout_ptr_k5_s1 = _vel_vaddsl_vsvl(4*5*gOutHeight*gOutWidth, vrgout_ptr_k0_s1, vl) ;
	  __vr vrgout_k5_s1 = _vel_vgtu_vvssml(vrgout_ptr_k5_s1, 0, 0, vmx_s1, vl) ;
	  __vr vrgout_ptr_k5_s2 = _vel_vaddsl_vsvl(4*5*gOutHeight*gOutWidth, vrgout_ptr_k0_s2, vl) ;
	  __vr vrgout_k5_s2 = _vel_vgtu_vvssml(vrgout_ptr_k5_s2, 0, 0, vmx_s2, vl) ;
	  __vr vrgout_ptr_k5_s3 = _vel_vaddsl_vsvl(4*5*gOutHeight*gOutWidth, vrgout_ptr_k0_s3, vl) ;
	  __vr vrgout_k5_s3 = _vel_vgtu_vvssml(vrgout_ptr_k5_s3, 0, 0, vmx_s3, vl) ;
	  __vr vrgout_ptr_k5_s4 = _vel_vaddsl_vsvl(4*5*gOutHeight*gOutWidth, vrgout_ptr_k0_s4, vl) ;
	  __vr vrgout_k5_s4 = _vel_vgtu_vvssml(vrgout_ptr_k5_s4, 0, 0, vmx_s4, vl) ;
	  VFMAD_C16(vrgout_k5_s0, vmx_s0, 5, r, 0) ;
	  VFMAD_C16(vrgout_k5_s1, vmx_s1, 5, r, 1) ;
	  VFMAD_C16(vrgout_k5_s2, vmx_s2, 5, r, 2) ;
	  VFMAD_C16(vrgout_k5_s3, vmx_s3, 5, r, 3) ;
	  VFMAD_C16(vrgout_k5_s4, vmx_s4, 5, r, 4) ;

	  __vr vrgout_ptr_k6_s0 = _vel_vaddsl_vsvl(4*6*gOutHeight*gOutWidth, vrgout_ptr_k0_s0, vl) ;
	  __vr vrgout_k6_s0 = _vel_vgtu_vvssml(vrgout_ptr_k6_s0, 0, 0, vmx_s0, vl) ;
	  __vr vrgout_ptr_k6_s1 = _vel_vaddsl_vsvl(4*6*gOutHeight*gOutWidth, vrgout_ptr_k0_s1, vl) ;
	  __vr vrgout_k6_s1 = _vel_vgtu_vvssml(vrgout_ptr_k6_s1, 0, 0, vmx_s1, vl) ;
	  __vr vrgout_ptr_k6_s2 = _vel_vaddsl_vsvl(4*6*gOutHeight*gOutWidth, vrgout_ptr_k0_s2, vl) ;
	  __vr vrgout_k6_s2 = _vel_vgtu_vvssml(vrgout_ptr_k6_s2, 0, 0, vmx_s2, vl) ;
	  __vr vrgout_ptr_k6_s3 = _vel_vaddsl_vsvl(4*6*gOutHeight*gOutWidth, vrgout_ptr_k0_s3, vl) ;
	  __vr vrgout_k6_s3 = _vel_vgtu_vvssml(vrgout_ptr_k6_s3, 0, 0, vmx_s3, vl) ;
	  __vr vrgout_ptr_k6_s4 = _vel_vaddsl_vsvl(4*6*gOutHeight*gOutWidth, vrgout_ptr_k0_s4, vl) ;
	  __vr vrgout_k6_s4 = _vel_vgtu_vvssml(vrgout_ptr_k6_s4, 0, 0, vmx_s4, vl) ;
	  VFMAD_C16(vrgout_k6_s0, vmx_s0, 6, r, 0) ;
	  VFMAD_C16(vrgout_k6_s1, vmx_s1, 6, r, 1) ;
	  VFMAD_C16(vrgout_k6_s2, vmx_s2, 6, r, 2) ;
	  VFMAD_C16(vrgout_k6_s3, vmx_s3, 6, r, 3) ;
	  VFMAD_C16(vrgout_k6_s4, vmx_s4, 6, r, 4) ;

	  __vr vrgout_ptr_k7_s0 = _vel_vaddsl_vsvl(4*7*gOutHeight*gOutWidth, vrgout_ptr_k0_s0, vl) ;
	  __vr vrgout_k7_s0 = _vel_vgtu_vvssml(vrgout_ptr_k7_s0, 0, 0, vmx_s0, vl) ;
	  __vr vrgout_ptr_k7_s1 = _vel_vaddsl_vsvl(4*7*gOutHeight*gOutWidth, vrgout_ptr_k0_s1, vl) ;
	  __vr vrgout_k7_s1 = _vel_vgtu_vvssml(vrgout_ptr_k7_s1, 0, 0, vmx_s1, vl) ;
	  __vr vrgout_ptr_k7_s2 = _vel_vaddsl_vsvl(4*7*gOutHeight*gOutWidth, vrgout_ptr_k0_s2, vl) ;
	  __vr vrgout_k7_s2 = _vel_vgtu_vvssml(vrgout_ptr_k7_s2, 0, 0, vmx_s2, vl) ;
	  __vr vrgout_ptr_k7_s3 = _vel_vaddsl_vsvl(4*7*gOutHeight*gOutWidth, vrgout_ptr_k0_s3, vl) ;
	  __vr vrgout_k7_s3 = _vel_vgtu_vvssml(vrgout_ptr_k7_s3, 0, 0, vmx_s3, vl) ;
	  __vr vrgout_ptr_k7_s4 = _vel_vaddsl_vsvl(4*7*gOutHeight*gOutWidth, vrgout_ptr_k0_s4, vl) ;
	  __vr vrgout_k7_s4 = _vel_vgtu_vvssml(vrgout_ptr_k7_s4, 0, 0, vmx_s4, vl) ;
	  VFMAD_C16(vrgout_k7_s0, vmx_s0, 7, r, 0) ;
	  VFMAD_C16(vrgout_k7_s1, vmx_s1, 7, r, 1) ;
	  VFMAD_C16(vrgout_k7_s2, vmx_s2, 7, r, 2) ;
	  VFMAD_C16(vrgout_k7_s3, vmx_s3, 7, r, 3) ;
	  VFMAD_C16(vrgout_k7_s4, vmx_s4, 7, r, 4) ;

#undef VFMAD_C16
	} // gOutChannel

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
vednnConvolutionBackwardData_direct_ker5(
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
