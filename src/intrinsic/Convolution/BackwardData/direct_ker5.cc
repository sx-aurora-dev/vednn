#include <stdint.h>
#include <stdlib.h>

#include "vednn.h"

#include "velintrin.h"
#define VLEN	(256)

template<filterLayout_t FLAYOUT, int NUMCHANNEL>
static inline void func(
    const float * __restrict__ pGOut,
    const float * __restrict__ pKernel,
    float * __restrict__ const pGIn,
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
    const int64_t n,
    const int64_t c
)
{
  const int64_t remain  = NUMCHANNEL & 0x1 ;
  const int64_t nPacked = NUMCHANNEL >> 1 ;

  int64_t gInIndex = gInGroupOffset + ((n * gInChannel + c) * gInHeight ) * gInWidth  ;

  for (int64_t h = 0; h < gInHeight ; h++ ) {
    for (int64_t w = 0; w < gInWidth ; w += VLEN ) {
      const int64_t vl = gInWidth - w < VLEN ? gInWidth - w  : VLEN ;

      __vr vrsum0  = _vel_vbrds_vsl(0.f, vl) ;
      __vr vrsum[nPacked] ;
#pragma clang loop unroll(full)
      for(int64_t cc=0; cc<nPacked; cc++) {
	vrsum[cc] = _vel_pvbrd_vsl(0UL, vl) ;
      }

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

#define FILTER_OFFSET(k,c,r,s) ( kernGroupOffset + filter_index<FLAYOUT>(k,c,r,s, gInChannelGroup, gOutChannelGroup, kernHeight, kernWidth) )
#define FILTER_DISTANCE_BY_C()   ( FLAYOUT == VEDNN_FILTER_LAYOUT_NCHW ? kernHeight * kernWidth : gOutChannelGroup ) ;
#define VFMAD(VRGOUT, VM, K, R, S)	{						\
            const int64_t filter_offset   = FILTER_OFFSET(K,c+ 0,R,S) ;			\
            const int64_t filter_distance = FILTER_DISTANCE_BY_C() ;			\
	    VRGOUT = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), VRGOUT, VM, vl) ;	\
	    __vr vrgoutP = _vel_vshf_vvvsl(VRGOUT, VRGOUT, VE_VSHUFFLE_YUZU, vl) ;	\
            if( remain ) {								\
              const float    kerValue0  = pKernel[filter_offset] ;			\
	      vrsum0 = _vel_vfmads_vvsvl(vrsum0, kerValue0, VRGOUT, vl) ;		\
            }										\
	    _Pragma("clang loop unroll(full)")						\
	    for(int64_t cc=0; cc<nPacked; cc++) {					\
	      const uint64_t kerValue = _vel_pack_f32p(pKernel + filter_offset + (2*cc+remain)   * filter_distance,	\
						       pKernel + filter_offset + (2*cc+remain+1) * filter_distance) ;	\
	      vrsum[cc] = _vel_pvfmad_vvsvl(vrsum[cc], kerValue, vrgoutP, vl) ;		\
	    }										\
	  }

	  VFMAD(vrgout_k0_s0, vmx_s0, k+0, r, 0) ;
	  VFMAD(vrgout_k0_s1, vmx_s1, k+0, r, 1) ;
	  VFMAD(vrgout_k0_s2, vmx_s2, k+0, r, 2) ;
	  VFMAD(vrgout_k0_s3, vmx_s3, k+0, r, 3) ;
	  VFMAD(vrgout_k0_s4, vmx_s4, k+0, r, 4) ;

	  k+=1 ;
	}
	if( ( (gOutChannelGroup >> 1) & 0x01)  == 1 ) {
	  int64_t gOutIndex    = gOutGroupOffset + ((n * gOutChannel + k) * gOutHeight) * gOutWidth ;

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
	  VFMAD(vrgout_k0_s0, vmx_s0, k+0, r, 0) ;
	  VFMAD(vrgout_k0_s1, vmx_s1, k+0, r, 1) ;
	  VFMAD(vrgout_k0_s2, vmx_s2, k+0, r, 2) ;
	  VFMAD(vrgout_k0_s3, vmx_s3, k+0, r, 3) ;
	  VFMAD(vrgout_k0_s4, vmx_s4, k+0, r, 4) ;

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
	  VFMAD(vrgout_k1_s0, vmx_s0, k+1, r, 0) ;
	  VFMAD(vrgout_k1_s1, vmx_s1, k+1, r, 1) ;
	  VFMAD(vrgout_k1_s2, vmx_s2, k+1, r, 2) ;
	  VFMAD(vrgout_k1_s3, vmx_s3, k+1, r, 3) ;
	  VFMAD(vrgout_k1_s4, vmx_s4, k+1, r, 4) ;

	  k+=2 ;
	}
	if( ( (gOutChannelGroup >> 2) & 0x01)  == 1 ) {
	  int64_t gOutIndex    = gOutGroupOffset + ((n * gOutChannel + k) * gOutHeight) * gOutWidth ;

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
	  VFMAD(vrgout_k0_s0, vmx_s0, k+0, r, 0) ;
	  VFMAD(vrgout_k0_s1, vmx_s1, k+0, r, 1) ;
	  VFMAD(vrgout_k0_s2, vmx_s2, k+0, r, 2) ;
	  VFMAD(vrgout_k0_s3, vmx_s3, k+0, r, 3) ;
	  VFMAD(vrgout_k0_s4, vmx_s4, k+0, r, 4) ;

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
	  VFMAD(vrgout_k1_s0, vmx_s0, k+1, r, 0) ;
	  VFMAD(vrgout_k1_s1, vmx_s1, k+1, r, 1) ;
	  VFMAD(vrgout_k1_s2, vmx_s2, k+1, r, 2) ;
	  VFMAD(vrgout_k1_s3, vmx_s3, k+1, r, 3) ;
	  VFMAD(vrgout_k1_s4, vmx_s4, k+1, r, 4) ;

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
	  VFMAD(vrgout_k2_s0, vmx_s0, k+2, r, 0) ;
	  VFMAD(vrgout_k2_s1, vmx_s1, k+2, r, 1) ;
	  VFMAD(vrgout_k2_s2, vmx_s2, k+2, r, 2) ;
	  VFMAD(vrgout_k2_s3, vmx_s3, k+2, r, 3) ;
	  VFMAD(vrgout_k2_s4, vmx_s4, k+2, r, 4) ;

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
	  VFMAD(vrgout_k3_s0, vmx_s0, k+3, r, 0) ;
	  VFMAD(vrgout_k3_s1, vmx_s1, k+3, r, 1) ;
	  VFMAD(vrgout_k3_s2, vmx_s2, k+3, r, 2) ;
	  VFMAD(vrgout_k3_s3, vmx_s3, k+3, r, 3) ;
	  VFMAD(vrgout_k3_s4, vmx_s4, k+3, r, 4) ;

	  k+=4 ;
	}
	for ( ; k<gOutChannelGroup; k+=8 ) {
	  int64_t gOutIndex    = gOutGroupOffset + ((n * gOutChannel + k) * gOutHeight) * gOutWidth ;

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
	  VFMAD(vrgout_k0_s0, vmx_s0, k+0, r, 0) ;
	  VFMAD(vrgout_k0_s1, vmx_s1, k+0, r, 1) ;
	  VFMAD(vrgout_k0_s2, vmx_s2, k+0, r, 2) ;
	  VFMAD(vrgout_k0_s3, vmx_s3, k+0, r, 3) ;
	  VFMAD(vrgout_k0_s4, vmx_s4, k+0, r, 4) ;

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
	  VFMAD(vrgout_k1_s0, vmx_s0, k+1, r, 0) ;
	  VFMAD(vrgout_k1_s1, vmx_s1, k+1, r, 1) ;
	  VFMAD(vrgout_k1_s2, vmx_s2, k+1, r, 2) ;
	  VFMAD(vrgout_k1_s3, vmx_s3, k+1, r, 3) ;
	  VFMAD(vrgout_k1_s4, vmx_s4, k+1, r, 4) ;

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
	  VFMAD(vrgout_k2_s0, vmx_s0, k+2, r, 0) ;
	  VFMAD(vrgout_k2_s1, vmx_s1, k+2, r, 1) ;
	  VFMAD(vrgout_k2_s2, vmx_s2, k+2, r, 2) ;
	  VFMAD(vrgout_k2_s3, vmx_s3, k+2, r, 3) ;
	  VFMAD(vrgout_k2_s4, vmx_s4, k+2, r, 4) ;

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
	  VFMAD(vrgout_k3_s0, vmx_s0, k+3, r, 0) ;
	  VFMAD(vrgout_k3_s1, vmx_s1, k+3, r, 1) ;
	  VFMAD(vrgout_k3_s2, vmx_s2, k+3, r, 2) ;
	  VFMAD(vrgout_k3_s3, vmx_s3, k+3, r, 3) ;
	  VFMAD(vrgout_k3_s4, vmx_s4, k+3, r, 4) ;

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
	  VFMAD(vrgout_k4_s0, vmx_s0, k+4, r, 0) ;
	  VFMAD(vrgout_k4_s1, vmx_s1, k+4, r, 1) ;
	  VFMAD(vrgout_k4_s2, vmx_s2, k+4, r, 2) ;
	  VFMAD(vrgout_k4_s3, vmx_s3, k+4, r, 3) ;
	  VFMAD(vrgout_k4_s4, vmx_s4, k+4, r, 4) ;

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
	  VFMAD(vrgout_k5_s0, vmx_s0, k+5, r, 0) ;
	  VFMAD(vrgout_k5_s1, vmx_s1, k+5, r, 1) ;
	  VFMAD(vrgout_k5_s2, vmx_s2, k+5, r, 2) ;
	  VFMAD(vrgout_k5_s3, vmx_s3, k+5, r, 3) ;
	  VFMAD(vrgout_k5_s4, vmx_s4, k+5, r, 4) ;

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
	  VFMAD(vrgout_k6_s0, vmx_s0, k+6, r, 0) ;
	  VFMAD(vrgout_k6_s1, vmx_s1, k+6, r, 1) ;
	  VFMAD(vrgout_k6_s2, vmx_s2, k+6, r, 2) ;
	  VFMAD(vrgout_k6_s3, vmx_s3, k+6, r, 3) ;
	  VFMAD(vrgout_k6_s4, vmx_s4, k+6, r, 4) ;

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
	  VFMAD(vrgout_k7_s0, vmx_s0, k+7, r, 0) ;
	  VFMAD(vrgout_k7_s1, vmx_s1, k+7, r, 1) ;
	  VFMAD(vrgout_k7_s2, vmx_s2, k+7, r, 2) ;
	  VFMAD(vrgout_k7_s3, vmx_s3, k+7, r, 3) ;
	  VFMAD(vrgout_k7_s4, vmx_s4, k+7, r, 4) ;

#undef FILTER_OFFSET
#undef VFMAD
	} // gOutChannel

      } // kernHeight

      if(remain) {
	_vel_vstu_vssl(vrsum0, 4, pGIn+gInIndex + 0 * gInHeight * gInWidth, vl) ;
      }
#pragma clang loop unroll(full)
      for(int64_t cc=0; cc<nPacked; cc++) {
	_vel_vstu_vssl(vrsum[cc], 4, pGIn+gInIndex + (2*cc+remain)   * gInHeight * gInWidth, vl) ;
	_vel_vstl_vssl(vrsum[cc], 4, pGIn+gInIndex + (2*cc+remain+1) * gInHeight * gInWidth, vl) ;
      }

      gInIndex += vl ;

    } // gInWidth
  } // gInHeight
}

template<filterLayout_t FLAYOUT>
static inline void convloop(
    const float * __restrict__ pGOut,
    const float * __restrict__ pKernel,
    float * __restrict__ const pGIn,
    const int64_t batch,
    const int64_t group,
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
    const int64_t strideWidth,
    const int64_t strideHeight,
    const int64_t padWidth,
    const int64_t padHeight,
    const int64_t dilationWidth,
    const int64_t dilationHeight
)
{
  for (int64_t n=0; n<batch; n++) {
    for (int64_t g = 0; g < group; g++) {

      int64_t gInGroupOffset  = g * gInChannelGroup * gInHeight * gInWidth;
      int64_t gOutGroupOffset = g * gOutChannelGroup * gOutHeight * gOutWidth;
      int64_t kernGroupOffset = g * gOutChannelGroup * gInChannelGroup * kernHeight * kernWidth;

      const int64_t remain = gInChannelGroup & 0xf ;

      int64_t c=0;
      switch(remain) {
      case 1:
	func<FLAYOUT, 1>(pGOut, pKernel, pGIn,
	   gOutChannel, gOutWidth, gOutHeight,
	   gInChannel, gInWidth, gInHeight,
	   kernWidth, kernHeight,
	   strideWidth, strideHeight,
	   padWidth, padHeight,
	   dilationWidth, dilationHeight,
	   gInChannelGroup, gOutChannelGroup,
	   gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	   n, c ) ;
	c+=1 ;
	break ;
      case 2:
	func<FLAYOUT, 2>(pGOut, pKernel, pGIn,
	   gOutChannel, gOutWidth, gOutHeight,
	   gInChannel, gInWidth, gInHeight,
	   kernWidth, kernHeight,
	   strideWidth, strideHeight,
	   padWidth, padHeight,
	   dilationWidth, dilationHeight,
	   gInChannelGroup, gOutChannelGroup,
	   gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	   n, c ) ;
	c+=2 ;
	break ;
      case 3:
	func<FLAYOUT, 3>(pGOut, pKernel, pGIn,
	   gOutChannel, gOutWidth, gOutHeight,
	   gInChannel, gInWidth, gInHeight,
	   kernWidth, kernHeight,
	   strideWidth, strideHeight,
	   padWidth, padHeight,
	   dilationWidth, dilationHeight,
	   gInChannelGroup, gOutChannelGroup,
	   gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	   n, c ) ;
	c+=3 ;
	break ;
      case 4:
	func<FLAYOUT, 4>(pGOut, pKernel, pGIn,
	   gOutChannel, gOutWidth, gOutHeight,
	   gInChannel, gInWidth, gInHeight,
	   kernWidth, kernHeight,
	   strideWidth, strideHeight,
	   padWidth, padHeight,
	   dilationWidth, dilationHeight,
	   gInChannelGroup, gOutChannelGroup,
	   gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	   n, c ) ;
	c+=4 ;
	break ;
      case 5:
	func<FLAYOUT, 5>(pGOut, pKernel, pGIn,
	   gOutChannel, gOutWidth, gOutHeight,
	   gInChannel, gInWidth, gInHeight,
	   kernWidth, kernHeight,
	   strideWidth, strideHeight,
	   padWidth, padHeight,
	   dilationWidth, dilationHeight,
	   gInChannelGroup, gOutChannelGroup,
	   gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	   n, c ) ;
	c+=5 ;
	break ;
      case 6:
	func<FLAYOUT, 6>(pGOut, pKernel, pGIn,
	   gOutChannel, gOutWidth, gOutHeight,
	   gInChannel, gInWidth, gInHeight,
	   kernWidth, kernHeight,
	   strideWidth, strideHeight,
	   padWidth, padHeight,
	   dilationWidth, dilationHeight,
	   gInChannelGroup, gOutChannelGroup,
	   gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	   n, c ) ;
	c+=6 ;
	break ;
      case 7:
	func<FLAYOUT, 7>(pGOut, pKernel, pGIn,
	   gOutChannel, gOutWidth, gOutHeight,
	   gInChannel, gInWidth, gInHeight,
	   kernWidth, kernHeight,
	   strideWidth, strideHeight,
	   padWidth, padHeight,
	   dilationWidth, dilationHeight,
	   gInChannelGroup, gOutChannelGroup,
	   gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	   n, c ) ;
	c+=7 ;
	break ;
      case 8:
	func<FLAYOUT, 8>(pGOut, pKernel, pGIn,
	   gOutChannel, gOutWidth, gOutHeight,
	   gInChannel, gInWidth, gInHeight,
	   kernWidth, kernHeight,
	   strideWidth, strideHeight,
	   padWidth, padHeight,
	   dilationWidth, dilationHeight,
	   gInChannelGroup, gOutChannelGroup,
	   gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	   n, c ) ;
	c+=8 ;
	break ;
      case 9:
	func<FLAYOUT, 9>(pGOut, pKernel, pGIn,
	   gOutChannel, gOutWidth, gOutHeight,
	   gInChannel, gInWidth, gInHeight,
	   kernWidth, kernHeight,
	   strideWidth, strideHeight,
	   padWidth, padHeight,
	   dilationWidth, dilationHeight,
	   gInChannelGroup, gOutChannelGroup,
	   gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	   n, c ) ;
	c+=9 ;
	break ;
      case 10:
	func<FLAYOUT, 10>(pGOut, pKernel, pGIn,
	   gOutChannel, gOutWidth, gOutHeight,
	   gInChannel, gInWidth, gInHeight,
	   kernWidth, kernHeight,
	   strideWidth, strideHeight,
	   padWidth, padHeight,
	   dilationWidth, dilationHeight,
	   gInChannelGroup, gOutChannelGroup,
	   gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	   n, c ) ;
	c+=10 ;
	break ;
      case 11:
	func<FLAYOUT, 11>(pGOut, pKernel, pGIn,
	   gOutChannel, gOutWidth, gOutHeight,
	   gInChannel, gInWidth, gInHeight,
	   kernWidth, kernHeight,
	   strideWidth, strideHeight,
	   padWidth, padHeight,
	   dilationWidth, dilationHeight,
	   gInChannelGroup, gOutChannelGroup,
	   gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	   n, c ) ;
	c+=11 ;
	break ;
      case 12:
	func<FLAYOUT, 12>(pGOut, pKernel, pGIn,
	   gOutChannel, gOutWidth, gOutHeight,
	   gInChannel, gInWidth, gInHeight,
	   kernWidth, kernHeight,
	   strideWidth, strideHeight,
	   padWidth, padHeight,
	   dilationWidth, dilationHeight,
	   gInChannelGroup, gOutChannelGroup,
	   gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	   n, c ) ;
	c+=12 ;
	break ;
      case 13:
	func<FLAYOUT, 13>(pGOut, pKernel, pGIn,
	   gOutChannel, gOutWidth, gOutHeight,
	   gInChannel, gInWidth, gInHeight,
	   kernWidth, kernHeight,
	   strideWidth, strideHeight,
	   padWidth, padHeight,
	   dilationWidth, dilationHeight,
	   gInChannelGroup, gOutChannelGroup,
	   gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	   n, c ) ;
	c+=13 ;
	break ;
      case 14:
	func<FLAYOUT, 14>(pGOut, pKernel, pGIn,
	   gOutChannel, gOutWidth, gOutHeight,
	   gInChannel, gInWidth, gInHeight,
	   kernWidth, kernHeight,
	   strideWidth, strideHeight,
	   padWidth, padHeight,
	   dilationWidth, dilationHeight,
	   gInChannelGroup, gOutChannelGroup,
	   gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	   n, c ) ;
	c+=14 ;
	break ;
      case 15:
	func<FLAYOUT, 15>(pGOut, pKernel, pGIn,
	   gOutChannel, gOutWidth, gOutHeight,
	   gInChannel, gInWidth, gInHeight,
	   kernWidth, kernHeight,
	   strideWidth, strideHeight,
	   padWidth, padHeight,
	   dilationWidth, dilationHeight,
	   gInChannelGroup, gOutChannelGroup,
	   gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	   n, c ) ;
	c+=15 ;
	break ;
      default :
	break ;
      }
      for (; c<gInChannelGroup; ) {
	func<FLAYOUT, 16>(pGOut, pKernel, pGIn,
	   gOutChannel, gOutWidth, gOutHeight,
	   gInChannel, gInWidth, gInHeight,
	   kernWidth, kernHeight,
	   strideWidth, strideHeight,
	   padWidth, padHeight,
	   dilationWidth, dilationHeight,
	   gInChannelGroup, gOutChannelGroup,
	   gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	   n, c) ;
	c+= 16 ;
      } // gInChannel
    } // group
  } // batch
}

extern "C"
vednnError_t
vednnConvolutionBackwardData_direct_ker5(
    const vednnTensorParam_t * 		pParamGradOut,
    const void *			pDataGradOut,
    const vednnFilterParam_t *	 	pParamKernel,
    const void * 			pDataKernel,
    const vednnConvolutionParam_t * 	pParamConv,
    const vednnTensorParam_t * 		pParamGradIn,
    void * 				pDataGradIn
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

  const int64_t filter_layout = pParamKernel->layout ;

  const int64_t group          = pParamConv->group;
  const int64_t strideWidth    = pParamConv->strideWidth;;
  const int64_t strideHeight   = pParamConv->strideHeight;
  const int64_t padWidth       = pParamConv->padWidth;
  const int64_t padHeight      = pParamConv->padHeight;
  const int64_t dilationWidth  = pParamConv->dilationWidth;
  const int64_t dilationHeight = pParamConv->dilationHeight;

  const int64_t gOutChannelGroup = gOutChannel  / group;
  const int64_t gInChannelGroup  = gInChannel / group;

  const float *  pGOut   = (const float *) pDataGradOut;
  const float *  pKernel = (const float *) pDataKernel;
  float *  const pGIn    = (float * const) pDataGradIn;


  if( filter_layout == VEDNN_FILTER_LAYOUT_NCHW) {
    convloop<VEDNN_FILTER_LAYOUT_NCHW>(pGOut, pKernel, pGIn,
	       batch, group,
	       gOutChannel, gOutWidth, gOutHeight,
	       gInChannel, gInWidth, gInHeight,
	       kernWidth, kernHeight,
	       gInChannelGroup, gOutChannelGroup,
	       strideWidth, strideHeight,
	       padWidth, padHeight,
	       dilationWidth, dilationHeight) ;
  }
  else {
    convloop<VEDNN_FILTER_LAYOUT_HWCN>(pGOut, pKernel, pGIn,
	       batch, group,
	       gOutChannel, gOutWidth, gOutHeight,
	       gInChannel, gInWidth, gInHeight,
	       kernWidth, kernHeight,
	       gInChannelGroup, gOutChannelGroup,
	       strideWidth, strideHeight,
	       padWidth, padHeight,
	       dilationWidth, dilationHeight) ;
  }

  return VEDNN_SUCCESS;
}
