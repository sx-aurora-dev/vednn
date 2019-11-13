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
    const int64_t padWidth,
    const int64_t padHeight,
    const int64_t gInChannelGroup,
    const int64_t gOutChannelGroup,
    const int64_t gInGroupOffset,
    const int64_t gOutGroupOffset,
    const int64_t kernGroupOffset,
    const int64_t n,
    const int64_t c,
    const int64_t nH
)
{
  const int64_t remain  = NUMCHANNEL & 0x1 ;
  const int64_t nPacked = NUMCHANNEL >> 1 ;

  const int64_t maxvl = nH*gInWidth ;

  int64_t gInIndex = gInGroupOffset + ((n * gInChannel + c) * gInHeight ) * gInWidth  ;

  __vr vrseq = _vel_vseq_vl(maxvl) ;
  __vr vrh  = _vel_vdivsl_vvsl(vrseq, gInWidth, maxvl) ;
  __vr vrw  = _vel_vsubsl_vvvl(vrseq, _vel_vmulul_vsvl(gInWidth,vrh, maxvl), maxvl) ;

  __vr vrj_s0 = _vel_vaddsl_vsvl(padWidth-0, vrw, maxvl) ;
  __vr vrj_s1 = _vel_vaddsl_vsvl(padWidth-1, vrw, maxvl) ;
  __vr vrj_s2 = _vel_vaddsl_vsvl(padWidth-2, vrw, maxvl) ;

  __vr vrx_s0 = _vel_vdivsl_vvsl(vrj_s0, 2, maxvl) ;
  __vr vrx_s1 = _vel_vdivsl_vvsl(vrj_s1, 2, maxvl) ;
  __vr vrx_s2 = _vel_vdivsl_vvsl(vrj_s2, 2, maxvl) ;

  __vm256 vmx0_s0 =  _vel_vfmkleq_mvl(_vel_vcmpsl_vvvl(vrj_s0, _vel_vmulsl_vsvl(2, vrx_s0, maxvl), maxvl), maxvl) ;
  __vm256 vmx1_s0 =  _vel_vfmklge_mvl(vrx_s0, maxvl) ;
  __vm256 vmx2_s0 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutWidth,vrx_s0, maxvl), maxvl) ;
  __vm256 vmx_s0 = _vel_andm_mmm(_vel_andm_mmm(vmx0_s0, vmx1_s0), vmx2_s0) ;

  __vm256 vmx0_s1 =  _vel_vfmkleq_mvl(_vel_vcmpsl_vvvl(vrj_s1, _vel_vmulsl_vsvl(2, vrx_s1, maxvl), maxvl), maxvl) ;
  __vm256 vmx1_s1 =  _vel_vfmklge_mvl(vrx_s1, maxvl) ;
  __vm256 vmx2_s1 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutWidth,vrx_s1, maxvl), maxvl) ;
  __vm256 vmx_s1 = _vel_andm_mmm(_vel_andm_mmm(vmx0_s1, vmx1_s1), vmx2_s1) ;

  __vm256 vmx0_s2 =  _vel_vfmkleq_mvl(_vel_vcmpsl_vvvl(vrj_s2, _vel_vmulsl_vsvl(2, vrx_s2, maxvl), maxvl), maxvl) ;
  __vm256 vmx1_s2 =  _vel_vfmklge_mvl(vrx_s2, maxvl) ;
  __vm256 vmx2_s2 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutWidth,vrx_s2, maxvl), maxvl) ;
  __vm256 vmx_s2 = _vel_andm_mmm(_vel_andm_mmm(vmx0_s2, vmx1_s2), vmx2_s2) ;

  __vr vrx_s01 = _vel_vmrg_vvvml(vrx_s0, vrx_s1, vmx_s1, maxvl) ;

  for (int64_t h=0; h<gInHeight; h+=nH) {
    const int64_t vl = gInWidth * (gInHeight - h < nH ? gInHeight - h : nH) ;

    __vr vrsum0  = _vel_vbrds_vsl(0.f, vl) ;
    __vr vrsum[nPacked] ;
#pragma clang loop unroll(full)
    for(int64_t cc=0; cc<nPacked; cc++) {
      vrsum[cc] = _vel_pvbrd_vsl(0UL, vl) ;
    }

    __vr vri_r0 = _vel_vaddsl_vsvl(padHeight-0+h, vrh, vl) ;
    __vr vri_r1 = _vel_vaddsl_vsvl(padHeight-1+h, vrh, vl) ;
    __vr vri_r2 = _vel_vaddsl_vsvl(padHeight-2+h, vrh, vl) ;

    __vr vry_r0 = _vel_vdivsl_vvsl(vri_r0, 2, vl) ;
    __vr vry_r1 = _vel_vdivsl_vvsl(vri_r1, 2, vl) ;
    __vr vry_r2 = _vel_vdivsl_vvsl(vri_r2, 2, vl) ;

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

    __vm256 vmxy_r0s0  = _vel_andm_mmm(vmy_r0,vmx_s0) ;
    __vm256 vmxy_r0s1  = _vel_andm_mmm(vmy_r0,vmx_s1) ;
    __vm256 vmxy_r0s2  = _vel_andm_mmm(vmy_r0,vmx_s2) ;

    __vm256 vmxy_r1s0 = _vel_andm_mmm(vmy_r1,vmx_s0) ;
    __vm256 vmxy_r1s1 = _vel_andm_mmm(vmy_r1,vmx_s1) ;
    __vm256 vmxy_r1s2 = _vel_andm_mmm(vmy_r1,vmx_s2) ;

    __vm256 vmxy_r2s0 = _vel_andm_mmm(vmy_r2,vmx_s0) ;
    __vm256 vmxy_r2s1 = _vel_andm_mmm(vmy_r2,vmx_s1) ;
    __vm256 vmxy_r2s2 = _vel_andm_mmm(vmy_r2,vmx_s2) ;

    __vm256 vmxy_r0s01 = _vel_orm_mmm(vmxy_r0s0, vmxy_r0s1) ;
    __vm256 vmxy_r1s01 = _vel_orm_mmm(vmxy_r1s0, vmxy_r1s1) ;
    __vm256 vmxy_r2s01 = _vel_orm_mmm(vmxy_r2s0, vmxy_r2s1) ;

    int64_t k=0;
    if( (gOutChannelGroup & 0x01 ) == 1 ) {
      int64_t gOutIndex    = gOutGroupOffset + ((n * gOutChannel + k) * gOutHeight) * gOutWidth ;

#define FILTER_OFFSET(k,c,r,s) ( kernGroupOffset + filter_index<FLAYOUT>(k,c,r,s, gInChannelGroup, gOutChannelGroup, kernHeight, kernWidth) )
#define FILTER_DISTANCE_BY_C()   ( FLAYOUT == VEDNN_FILTER_LAYOUT_NCHW ? kernHeight * kernWidth : gOutChannelGroup ) ;
#define VFADD(VRGOUT, VM, K, R, S)	{							\
	const int64_t filter_offset   = FILTER_OFFSET(K,c+ 0,R,S) ;				\
	const int64_t filter_distance = FILTER_DISTANCE_BY_C() ;				\
	__vr vrgout  = _vel_vmrg_vsvml(0.f, VRGOUT, VM, vl) ;						\
	__vr vrgoutP = _vel_vshf_vvvsl(vrgout, vrgout, VE_VSHUFFLE_YUZU, vl) ;			\
	if( remain ) {										\
	  const float kerValue  = pKernel[filter_offset] ;					\
	  vrsum0 = _vel_vfmads_vvsvl(vrsum0, kerValue, vrgout, vl) ;				\
	}											\
	_Pragma("clang loop unroll(full)")							\
	for(int64_t cc=0; cc<nPacked; cc++) {							\
	  const uint64_t kerValue = _vel_pack_f32p(pKernel + filter_offset + (2*cc+remain)   * filter_distance,		\
						   pKernel + filter_offset + (2*cc+remain+1) * filter_distance) ;	\
	  vrsum[cc] = _vel_pvfmad_vvsvl(vrsum[cc], kerValue, vrgoutP, vl) ;			\
	}											\
      }


      __vr vrgout_ptr_k0_r0_s01 = _vel_vsfa_vvssl(_vel_vaddsl_vvvl(_vel_vmulsl_vsvl(gOutWidth, vry_r0, vl), vrx_s01, vl),
					 2,
					 (unsigned long)(pGOut+gOutIndex), vl) ;
      __vr vrgout_k0_r0_s01 = _vel_vgtu_vvssml(vrgout_ptr_k0_r0_s01, 0, 0, vmxy_r0s01, vl) ;
      __vr vrgout_ptr_k0_r0_s2 = _vel_vsfa_vvssl(_vel_vaddsl_vvvl(_vel_vmulsl_vsvl(gOutWidth, vry_r0, vl), vrx_s2, vl),
					 2,
					 (unsigned long)(pGOut+gOutIndex), vl) ;
      __vr vrgout_k0_r0_s2 = _vel_vgtu_vvssml(vrgout_ptr_k0_r0_s2, 0, 0, vmxy_r0s2, vl) ;

      VFADD(vrgout_k0_r0_s01, vmxy_r0s0, k+0, 0, 0) ;
      VFADD(vrgout_k0_r0_s01, vmxy_r0s1, k+0, 0, 1) ;
      VFADD(vrgout_k0_r0_s2,  vmxy_r0s2, k+0, 0, 2) ;


      __vr vrgout_ptr_k0_r1_s01 = _vel_vsfa_vvssl(_vel_vaddsl_vvvl(_vel_vmulsl_vsvl(gOutWidth, vry_r1, vl), vrx_s01, vl),
					 2,
					 (unsigned long)(pGOut+gOutIndex), vl) ;
      __vr vrgout_k0_r1_s01 = _vel_vgtu_vvssml(vrgout_ptr_k0_r1_s01, 0, 0, vmxy_r1s01, vl) ;
      __vr vrgout_ptr_k0_r1_s2 = _vel_vsfa_vvssl(_vel_vaddsl_vvvl(_vel_vmulsl_vsvl(gOutWidth, vry_r1, vl), vrx_s2, vl),
					 2,
					 (unsigned long)(pGOut+gOutIndex), vl) ;
      __vr vrgout_k0_r1_s2 = _vel_vgtu_vvssml(vrgout_ptr_k0_r1_s2, 0, 0, vmxy_r1s2, vl) ;

      VFADD(vrgout_k0_r1_s01, vmxy_r1s0, k+0, 1, 0) ;
      VFADD(vrgout_k0_r1_s01, vmxy_r1s1, k+0, 1, 1) ;
      VFADD(vrgout_k0_r1_s2,  vmxy_r1s2, k+0, 1, 2) ;

      __vr vrgout_ptr_k0_r2_s01 = _vel_vsfa_vvssl(_vel_vaddsl_vvvl(_vel_vmulsl_vsvl(gOutWidth, vry_r2, vl), vrx_s01, vl),
					 2,
					 (unsigned long)(pGOut+gOutIndex), vl) ;
      __vr vrgout_k0_r2_s01 = _vel_vgtu_vvssml(vrgout_ptr_k0_r2_s01, 0, 0, vmxy_r2s01, vl) ;
      __vr vrgout_ptr_k0_r2_s2 = _vel_vsfa_vvssl(_vel_vaddsl_vvvl(_vel_vmulsl_vsvl(gOutWidth, vry_r2, vl), vrx_s2, vl),
					 2,
					 (unsigned long)(pGOut+gOutIndex), vl) ;
      __vr vrgout_k0_r2_s2 = _vel_vgtu_vvssml(vrgout_ptr_k0_r2_s2, 0, 0, vmxy_r2s2, vl) ;

      VFADD(vrgout_k0_r2_s01, vmxy_r2s0, k+0, 2, 0) ;
      VFADD(vrgout_k0_r2_s01, vmxy_r2s1, k+0, 2, 1) ;
      VFADD(vrgout_k0_r2_s2,  vmxy_r2s2, k+0, 2, 2) ;

      k+=1 ;
    }
    if( ((gOutChannelGroup >> 1) & 0x01 ) == 1 ) {
      int64_t gOutIndex    = gOutGroupOffset + ((n * gOutChannel + k) * gOutHeight) * gOutWidth ;

      __vr vrgout_ptr_k0_r0_s01 = _vel_vsfa_vvssl(_vel_vaddsl_vvvl(_vel_vmulsl_vsvl(gOutWidth, vry_r0, vl), vrx_s01, vl),
					 2,
					 (unsigned long)(pGOut+gOutIndex), vl) ;
      __vr vrgout_k0_r0_s01 = _vel_vgtu_vvssml(vrgout_ptr_k0_r0_s01, 0, 0, vmxy_r0s01, vl) ;
      __vr vrgout_ptr_k1_r0_s01 = _vel_vaddsl_vsvl(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_r0_s01, vl) ;
      __vr vrgout_k1_r0_s01 = _vel_vgtu_vvssml(vrgout_ptr_k1_r0_s01, 0, 0, vmxy_r0s01, vl) ;

      VFADD(vrgout_k0_r0_s01, vmxy_r0s0, k+0, 0, 0) ;
      VFADD(vrgout_k1_r0_s01, vmxy_r0s0, k+1, 0, 0) ;

      VFADD(vrgout_k0_r0_s01, vmxy_r0s1, k+0, 0, 1) ;
      VFADD(vrgout_k1_r0_s01, vmxy_r0s1, k+1, 0, 1) ;

      __vr vrgout_ptr_k0_r0_s2 = _vel_vsfa_vvssl(_vel_vaddsl_vvvl(_vel_vmulsl_vsvl(gOutWidth, vry_r0, vl), vrx_s2, vl),
					 2,
					 (unsigned long)(pGOut+gOutIndex), vl) ;
      __vr vrgout_k0_r0_s2 = _vel_vgtu_vvssml(vrgout_ptr_k0_r0_s2, 0, 0, vmxy_r0s2, vl) ;
      __vr vrgout_ptr_k1_r0_s2 = _vel_vaddsl_vsvl(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_r0_s2, vl) ;
      __vr vrgout_k1_r0_s2 = _vel_vgtu_vvssml(vrgout_ptr_k1_r0_s2, 0, 0, vmxy_r0s2, vl) ;

      VFADD(vrgout_k0_r0_s2, vmxy_r0s2, k+0, 0, 2) ;
      VFADD(vrgout_k1_r0_s2, vmxy_r0s2, k+1, 0, 2) ;

      __vr vrgout_ptr_k0_r1_s01 = _vel_vsfa_vvssl(_vel_vaddsl_vvvl(_vel_vmulsl_vsvl(gOutWidth, vry_r1, vl), vrx_s01, vl),
					 2,
					 (unsigned long)(pGOut+gOutIndex), vl) ;
      __vr vrgout_k0_r1_s01 = _vel_vgtu_vvssml(vrgout_ptr_k0_r1_s01, 0, 0, vmxy_r1s01, vl) ;
      __vr vrgout_ptr_k1_r1_s01 = _vel_vaddsl_vsvl(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_r1_s01, vl) ;
      __vr vrgout_k1_r1_s01 = _vel_vgtu_vvssml(vrgout_ptr_k1_r1_s01, 0, 0, vmxy_r1s01, vl) ;

      VFADD(vrgout_k0_r1_s01, vmxy_r1s0, k+0, 1, 0) ;
      VFADD(vrgout_k1_r1_s01, vmxy_r1s0, k+1, 1, 0) ;

      VFADD(vrgout_k0_r1_s01, vmxy_r1s1, k+0, 1, 1) ;
      VFADD(vrgout_k1_r1_s01, vmxy_r1s1, k+1, 1, 1) ;

      __vr vrgout_ptr_k0_r1_s2 = _vel_vsfa_vvssl(_vel_vaddsl_vvvl(_vel_vmulsl_vsvl(gOutWidth, vry_r1, vl), vrx_s2, vl),
					 2,
					 (unsigned long)(pGOut+gOutIndex), vl) ;
      __vr vrgout_k0_r1_s2 = _vel_vgtu_vvssml(vrgout_ptr_k0_r1_s2, 0, 0, vmxy_r1s2, vl) ;
      __vr vrgout_ptr_k1_r1_s2 = _vel_vaddsl_vsvl(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_r1_s2, vl) ;
      __vr vrgout_k1_r1_s2 = _vel_vgtu_vvssml(vrgout_ptr_k1_r1_s2, 0, 0, vmxy_r1s2, vl) ;

      VFADD(vrgout_k0_r1_s2, vmxy_r1s2, k+0, 1, 2) ;
      VFADD(vrgout_k1_r1_s2, vmxy_r1s2, k+1, 1, 2) ;

      __vr vrgout_ptr_k0_r2_s01 = _vel_vsfa_vvssl(_vel_vaddsl_vvvl(_vel_vmulsl_vsvl(gOutWidth, vry_r2, vl), vrx_s01, vl),
					 2,
					 (unsigned long)(pGOut+gOutIndex), vl) ;
      __vr vrgout_k0_r2_s01 = _vel_vgtu_vvssml(vrgout_ptr_k0_r2_s01, 0, 0, vmxy_r2s01, vl) ;
      __vr vrgout_ptr_k1_r2_s01 = _vel_vaddsl_vsvl(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_r2_s01, vl) ;
      __vr vrgout_k1_r2_s01 = _vel_vgtu_vvssml(vrgout_ptr_k1_r2_s01, 0, 0, vmxy_r2s01, vl) ;

      VFADD(vrgout_k0_r2_s01, vmxy_r2s0, k+0, 2, 0) ;
      VFADD(vrgout_k1_r2_s01, vmxy_r2s0, k+1, 2, 0) ;

      VFADD(vrgout_k0_r2_s01, vmxy_r2s1, k+0, 2, 1) ;
      VFADD(vrgout_k1_r2_s01, vmxy_r2s1, k+1, 2, 1) ;

      __vr vrgout_ptr_k0_r2_s2 = _vel_vsfa_vvssl(_vel_vaddsl_vvvl(_vel_vmulsl_vsvl(gOutWidth, vry_r2, vl), vrx_s2, vl),
					 2,
					 (unsigned long)(pGOut+gOutIndex), vl) ;
      __vr vrgout_k0_r2_s2 = _vel_vgtu_vvssml(vrgout_ptr_k0_r2_s2, 0, 0, vmxy_r2s2, vl) ;
      __vr vrgout_ptr_k1_r2_s2 = _vel_vaddsl_vsvl(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_r2_s2, vl) ;
      __vr vrgout_k1_r2_s2 = _vel_vgtu_vvssml(vrgout_ptr_k1_r2_s2, 0, 0, vmxy_r2s2, vl) ;

      VFADD(vrgout_k0_r2_s2, vmxy_r2s2, k+0, 2, 2) ;
      VFADD(vrgout_k1_r2_s2, vmxy_r2s2, k+1, 2, 2) ;

      k+=2 ;
    }
    for (; k<gOutChannelGroup; k+=4) {
      int64_t gOutIndex    = gOutGroupOffset + ((n * gOutChannel + k) * gOutHeight) * gOutWidth ;

      __vr vrgout_ptr_k0_r0_s01 = _vel_vsfa_vvssl(_vel_vaddsl_vvvl(_vel_vmulsl_vsvl(gOutWidth, vry_r0, vl), vrx_s01, vl),
					 2,
					 (unsigned long)(pGOut+gOutIndex), vl) ;
      __vr vrgout_k0_r0_s01 = _vel_vgtu_vvssml(vrgout_ptr_k0_r0_s01, 0, 0, vmxy_r0s01, vl) ;
      __vr vrgout_ptr_k1_r0_s01 = _vel_vaddsl_vsvl(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_r0_s01, vl) ;
      __vr vrgout_k1_r0_s01 = _vel_vgtu_vvssml(vrgout_ptr_k1_r0_s01, 0, 0, vmxy_r0s01, vl) ;
      __vr vrgout_ptr_r0_k2_s01 = _vel_vaddsl_vsvl(4*2*gOutHeight*gOutWidth, vrgout_ptr_k0_r0_s01, vl) ;
      __vr vrgout_k2_r0_s01 = _vel_vgtu_vvssml(vrgout_ptr_r0_k2_s01, 0, 0, vmxy_r0s01, vl) ;
      __vr vrgout_ptr_k3_r0_s01 = _vel_vaddsl_vsvl(4*3*gOutHeight*gOutWidth, vrgout_ptr_k0_r0_s01, vl) ;
      __vr vrgout_k3_r0_s01 = _vel_vgtu_vvssml(vrgout_ptr_k3_r0_s01, 0, 0, vmxy_r0s01, vl) ;

      VFADD(vrgout_k0_r0_s01, vmxy_r0s0, k+0, 0, 0) ;
      VFADD(vrgout_k1_r0_s01, vmxy_r0s0, k+1, 0, 0) ;
      VFADD(vrgout_k2_r0_s01, vmxy_r0s0, k+2, 0, 0) ;
      VFADD(vrgout_k3_r0_s01, vmxy_r0s0, k+3, 0, 0) ;

      VFADD(vrgout_k0_r0_s01, vmxy_r0s1, k+0, 0, 1) ;
      VFADD(vrgout_k1_r0_s01, vmxy_r0s1, k+1, 0, 1) ;
      VFADD(vrgout_k2_r0_s01, vmxy_r0s1, k+2, 0, 1) ;
      VFADD(vrgout_k3_r0_s01, vmxy_r0s1, k+3, 0, 1) ;

      __vr vrgout_ptr_k0_r0_s2 = _vel_vsfa_vvssl(_vel_vaddsl_vvvl(_vel_vmulsl_vsvl(gOutWidth, vry_r0, vl), vrx_s2, vl),
					 2,
					 (unsigned long)(pGOut+gOutIndex), vl) ;
      __vr vrgout_k0_r0_s2 = _vel_vgtu_vvssml(vrgout_ptr_k0_r0_s2, 0, 0, vmxy_r0s2, vl) ;
      __vr vrgout_ptr_k1_r0_s2 = _vel_vaddsl_vsvl(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_r0_s2, vl) ;
      __vr vrgout_k1_r0_s2 = _vel_vgtu_vvssml(vrgout_ptr_k1_r0_s2, 0, 0, vmxy_r0s2, vl) ;
      __vr vrgout_ptr_k2_r0_s2 = _vel_vaddsl_vsvl(4*2*gOutHeight*gOutWidth, vrgout_ptr_k0_r0_s2, vl) ;
      __vr vrgout_k2_r0_s2 = _vel_vgtu_vvssml(vrgout_ptr_k2_r0_s2, 0, 0, vmxy_r0s2, vl) ;
      __vr vrgout_ptr_k3_r0_s2 = _vel_vaddsl_vsvl(4*3*gOutHeight*gOutWidth, vrgout_ptr_k0_r0_s2, vl) ;
      __vr vrgout_k3_r0_s2 = _vel_vgtu_vvssml(vrgout_ptr_k3_r0_s2, 0, 0, vmxy_r0s2, vl) ;

      VFADD(vrgout_k0_r0_s2, vmxy_r0s2, k+0, 0, 2) ;
      VFADD(vrgout_k1_r0_s2, vmxy_r0s2, k+1, 0, 2) ;
      VFADD(vrgout_k2_r0_s2, vmxy_r0s2, k+2, 0, 2) ;
      VFADD(vrgout_k3_r0_s2, vmxy_r0s2, k+3, 0, 2) ;

      __vr vrgout_ptr_k0_r1_s01 = _vel_vsfa_vvssl(_vel_vaddsl_vvvl(_vel_vmulsl_vsvl(gOutWidth, vry_r1, vl), vrx_s01, vl),
					 2,
					 (unsigned long)(pGOut+gOutIndex), vl) ;
      __vr vrgout_k0_r1_s01 = _vel_vgtu_vvssml(vrgout_ptr_k0_r1_s01, 0, 0, vmxy_r1s01, vl) ;
      __vr vrgout_ptr_k1_r1_s01 = _vel_vaddsl_vsvl(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_r1_s01, vl) ;
      __vr vrgout_k1_r1_s01 = _vel_vgtu_vvssml(vrgout_ptr_k1_r1_s01, 0, 0, vmxy_r1s01, vl) ;
      __vr vrgout_ptr_k2_r1_s01 = _vel_vaddsl_vsvl(4*2*gOutHeight*gOutWidth, vrgout_ptr_k0_r1_s01, vl) ;
      __vr vrgout_k2_r1_s01 = _vel_vgtu_vvssml(vrgout_ptr_k2_r1_s01, 0, 0, vmxy_r1s01, vl) ;
      __vr vrgout_ptr_k3_r1_s01 = _vel_vaddsl_vsvl(4*3*gOutHeight*gOutWidth, vrgout_ptr_k0_r1_s01, vl) ;
      __vr vrgout_k3_r1_s01 = _vel_vgtu_vvssml(vrgout_ptr_k3_r1_s01, 0, 0, vmxy_r1s01, vl) ;

      VFADD(vrgout_k0_r1_s01, vmxy_r1s0, k+0, 1, 0) ;
      VFADD(vrgout_k1_r1_s01, vmxy_r1s0, k+1, 1, 0) ;
      VFADD(vrgout_k2_r1_s01, vmxy_r1s0, k+2, 1, 0) ;
      VFADD(vrgout_k3_r1_s01, vmxy_r1s0, k+3, 1, 0) ;

      VFADD(vrgout_k0_r1_s01, vmxy_r1s1, k+0, 1, 1) ;
      VFADD(vrgout_k1_r1_s01, vmxy_r1s1, k+1, 1, 1) ;
      VFADD(vrgout_k2_r1_s01, vmxy_r1s1, k+2, 1, 1) ;
      VFADD(vrgout_k3_r1_s01, vmxy_r1s1, k+3, 1, 1) ;

      __vr vrgout_ptr_k0_r1_s2 = _vel_vsfa_vvssl(_vel_vaddsl_vvvl(_vel_vmulsl_vsvl(gOutWidth, vry_r1, vl), vrx_s2, vl),
					 2,
					 (unsigned long)(pGOut+gOutIndex), vl) ;
      __vr vrgout_k0_r1_s2 = _vel_vgtu_vvssml(vrgout_ptr_k0_r1_s2, 0, 0, vmxy_r1s2, vl) ;
      __vr vrgout_ptr_k1_r1_s2 = _vel_vaddsl_vsvl(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_r1_s2, vl) ;
      __vr vrgout_k1_r1_s2 = _vel_vgtu_vvssml(vrgout_ptr_k1_r1_s2, 0, 0, vmxy_r1s2, vl) ;
      __vr vrgout_ptr_k2_r1_s2 = _vel_vaddsl_vsvl(4*2*gOutHeight*gOutWidth, vrgout_ptr_k0_r1_s2, vl) ;
      __vr vrgout_k2_r1_s2 = _vel_vgtu_vvssml(vrgout_ptr_k2_r1_s2, 0, 0, vmxy_r1s2, vl) ;
      __vr vrgout_ptr_k3_r1_s2 = _vel_vaddsl_vsvl(4*3*gOutHeight*gOutWidth, vrgout_ptr_k0_r1_s2, vl) ;
      __vr vrgout_k3_r1_s2 = _vel_vgtu_vvssml(vrgout_ptr_k3_r1_s2, 0, 0, vmxy_r1s2, vl) ;

      VFADD(vrgout_k0_r1_s2, vmxy_r1s2, k+0, 1, 2) ;
      VFADD(vrgout_k1_r1_s2, vmxy_r1s2, k+1, 1, 2) ;
      VFADD(vrgout_k2_r1_s2, vmxy_r1s2, k+2, 1, 2) ;
      VFADD(vrgout_k3_r1_s2, vmxy_r1s2, k+3, 1, 2) ;

      __vr vrgout_ptr_k0_r2_s01 = _vel_vsfa_vvssl(_vel_vaddsl_vvvl(_vel_vmulsl_vsvl(gOutWidth, vry_r2, vl), vrx_s01, vl),
					 2,
					 (unsigned long)(pGOut+gOutIndex), vl) ;
      __vr vrgout_k0_r2_s01 = _vel_vgtu_vvssml(vrgout_ptr_k0_r2_s01, 0, 0, vmxy_r2s01, vl) ;
      __vr vrgout_ptr_k1_r2_s01 = _vel_vaddsl_vsvl(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_r2_s01, vl) ;
      __vr vrgout_k1_r2_s01 = _vel_vgtu_vvssml(vrgout_ptr_k1_r2_s01, 0, 0, vmxy_r2s01, vl) ;
      __vr vrgout_ptr_k2_r2_s01 = _vel_vaddsl_vsvl(4*2*gOutHeight*gOutWidth, vrgout_ptr_k0_r2_s01, vl) ;
      __vr vrgout_k2_r2_s01 = _vel_vgtu_vvssml(vrgout_ptr_k2_r2_s01, 0, 0, vmxy_r2s01, vl) ;
      __vr vrgout_ptr_k3_r2_s01 = _vel_vaddsl_vsvl(4*3*gOutHeight*gOutWidth, vrgout_ptr_k0_r2_s01, vl) ;
      __vr vrgout_k3_r2_s01 = _vel_vgtu_vvssml(vrgout_ptr_k3_r2_s01, 0, 0, vmxy_r2s01, vl) ;

      VFADD(vrgout_k0_r2_s01, vmxy_r2s0, k+0, 2, 0) ;
      VFADD(vrgout_k1_r2_s01, vmxy_r2s0, k+1, 2, 0) ;
      VFADD(vrgout_k2_r2_s01, vmxy_r2s0, k+2, 2, 0) ;
      VFADD(vrgout_k3_r2_s01, vmxy_r2s0, k+3, 2, 0) ;

      VFADD(vrgout_k0_r2_s01, vmxy_r2s1, k+0, 2, 1) ;
      VFADD(vrgout_k1_r2_s01, vmxy_r2s1, k+1, 2, 1) ;
      VFADD(vrgout_k2_r2_s01, vmxy_r2s1, k+2, 2, 1) ;
      VFADD(vrgout_k3_r2_s01, vmxy_r2s1, k+3, 2, 1) ;

      __vr vrgout_ptr_k0_r2_s2 = _vel_vsfa_vvssl(_vel_vaddsl_vvvl(_vel_vmulsl_vsvl(gOutWidth, vry_r2, vl), vrx_s2, vl),
					 2,
					 (unsigned long)(pGOut+gOutIndex), vl) ;
      __vr vrgout_k0_r2_s2 = _vel_vgtu_vvssml(vrgout_ptr_k0_r2_s2, 0, 0, vmxy_r2s2, vl) ;
      __vr vrgout_ptr_k1_r2_s2 = _vel_vaddsl_vsvl(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_r2_s2, vl) ;
      __vr vrgout_k1_r2_s2 = _vel_vgtu_vvssml(vrgout_ptr_k1_r2_s2, 0, 0, vmxy_r2s2, vl) ;
      __vr vrgout_ptr_k2_r2_s2 = _vel_vaddsl_vsvl(4*2*gOutHeight*gOutWidth, vrgout_ptr_k0_r2_s2, vl) ;
      __vr vrgout_k2_r2_s2 = _vel_vgtu_vvssml(vrgout_ptr_k2_r2_s2, 0, 0, vmxy_r2s2, vl) ;
      __vr vrgout_ptr_k3_r2_s2 = _vel_vaddsl_vsvl(4*3*gOutHeight*gOutWidth, vrgout_ptr_k0_r2_s2, vl) ;
      __vr vrgout_k3_r2_s2 = _vel_vgtu_vvssml(vrgout_ptr_k3_r2_s2, 0, 0, vmxy_r2s2, vl) ;

      VFADD(vrgout_k0_r2_s2, vmxy_r2s2, k+0, 2, 2) ;
      VFADD(vrgout_k1_r2_s2, vmxy_r2s2, k+1, 2, 2) ;
      VFADD(vrgout_k2_r2_s2, vmxy_r2s2, k+2, 2, 2) ;
      VFADD(vrgout_k3_r2_s2, vmxy_r2s2, k+3, 2, 2) ;

#undef VFADD
#undef FILTER_OFFSET
    } // gOutChannel

    if( remain ) {
      _vel_vstu_vssl(vrsum0, 4, pGIn+gInIndex + 0 * gInHeight * gInWidth, vl) ;
    }
#pragma clang loop unroll(full)
    for(int64_t cc=0; cc<nPacked; cc++) {
	_vel_vstu_vssl(vrsum[cc], 4, pGIn+gInIndex + (2*cc+remain  ) * gInHeight * gInWidth, vl) ;
	_vel_vstl_vssl(vrsum[cc], 4, pGIn+gInIndex + (2*cc+remain+1) * gInHeight * gInWidth, vl) ;
    }

    gInIndex += vl ;
  } // gOutPixels
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
    const int64_t kernWidth,		// 3
    const int64_t kernHeight,		// 3
    const int64_t gInChannelGroup,
    const int64_t gOutChannelGroup,
    const int64_t strideWidth,		// 2
    const int64_t strideHeight,		// 2
    const int64_t padWidth,
    const int64_t padHeight,
    const int64_t dilationWidth,	// 1
    const int64_t dilationHeight	// 1
)
{

  const int64_t nH = VLEN / gInWidth ;

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
	   padWidth, padHeight,
	   gInChannelGroup, gOutChannelGroup,
	   gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	   n, c, nH) ;
	c+=1 ;
	break ;
      case 2:
	func<FLAYOUT, 2>(pGOut, pKernel, pGIn,
	   gOutChannel, gOutWidth, gOutHeight,
	   gInChannel, gInWidth, gInHeight,
	   kernWidth, kernHeight,
	   padWidth, padHeight,
	   gInChannelGroup, gOutChannelGroup,
	   gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	   n, c, nH) ;
	c+=2 ;
	break ;
      case 3:
	func<FLAYOUT, 3>(pGOut, pKernel, pGIn,
	   gOutChannel, gOutWidth, gOutHeight,
	   gInChannel, gInWidth, gInHeight,
	   kernWidth, kernHeight,
	   padWidth, padHeight,
	   gInChannelGroup, gOutChannelGroup,
	   gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	   n, c, nH) ;
	c+=3 ;
	break ;
      case 4:
	func<FLAYOUT, 4>(pGOut, pKernel, pGIn,
	   gOutChannel, gOutWidth, gOutHeight,
	   gInChannel, gInWidth, gInHeight,
	   kernWidth, kernHeight,
	   padWidth, padHeight,
	   gInChannelGroup, gOutChannelGroup,
	   gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	   n, c, nH) ;
	c+=4 ;
	break ;
      case 5:
	func<FLAYOUT, 5>(pGOut, pKernel, pGIn,
	   gOutChannel, gOutWidth, gOutHeight,
	   gInChannel, gInWidth, gInHeight,
	   kernWidth, kernHeight,
	   padWidth, padHeight,
	   gInChannelGroup, gOutChannelGroup,
	   gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	   n, c, nH) ;
	c+=5 ;
	break ;
      case 6:
	func<FLAYOUT, 6>(pGOut, pKernel, pGIn,
	   gOutChannel, gOutWidth, gOutHeight,
	   gInChannel, gInWidth, gInHeight,
	   kernWidth, kernHeight,
	   padWidth, padHeight,
	   gInChannelGroup, gOutChannelGroup,
	   gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	   n, c, nH) ;
	c+=6 ;
	break ;
      case 7:
	func<FLAYOUT, 7>(pGOut, pKernel, pGIn,
	   gOutChannel, gOutWidth, gOutHeight,
	   gInChannel, gInWidth, gInHeight,
	   kernWidth, kernHeight,
	   padWidth, padHeight,
	   gInChannelGroup, gOutChannelGroup,
	   gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	   n, c, nH) ;
	c+=7 ;
	break ;
      case 8:
	func<FLAYOUT, 8>(pGOut, pKernel, pGIn,
	   gOutChannel, gOutWidth, gOutHeight,
	   gInChannel, gInWidth, gInHeight,
	   kernWidth, kernHeight,
	   padWidth, padHeight,
	   gInChannelGroup, gOutChannelGroup,
	   gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	   n, c, nH) ;
	c+=8 ;
	break ;
      case 9:
	func<FLAYOUT, 9>(pGOut, pKernel, pGIn,
	   gOutChannel, gOutWidth, gOutHeight,
	   gInChannel, gInWidth, gInHeight,
	   kernWidth, kernHeight,
	   padWidth, padHeight,
	   gInChannelGroup, gOutChannelGroup,
	   gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	   n, c, nH) ;
	c+=9 ;
	break ;
      case 10:
	func<FLAYOUT, 10>(pGOut, pKernel, pGIn,
	   gOutChannel, gOutWidth, gOutHeight,
	   gInChannel, gInWidth, gInHeight,
	   kernWidth, kernHeight,
	   padWidth, padHeight,
	   gInChannelGroup, gOutChannelGroup,
	   gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	   n, c, nH) ;
	c+=10 ;
	break ;
      case 11:
	func<FLAYOUT, 11>(pGOut, pKernel, pGIn,
	   gOutChannel, gOutWidth, gOutHeight,
	   gInChannel, gInWidth, gInHeight,
	   kernWidth, kernHeight,
	   padWidth, padHeight,
	   gInChannelGroup, gOutChannelGroup,
	   gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	   n, c, nH) ;
	c+=11 ;
	break ;
      case 12:
	func<FLAYOUT, 12>(pGOut, pKernel, pGIn,
	   gOutChannel, gOutWidth, gOutHeight,
	   gInChannel, gInWidth, gInHeight,
	   kernWidth, kernHeight,
	   padWidth, padHeight,
	   gInChannelGroup, gOutChannelGroup,
	   gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	   n, c, nH) ;
	c+=12 ;
	break ;
      case 13:
	func<FLAYOUT, 13>(pGOut, pKernel, pGIn,
	   gOutChannel, gOutWidth, gOutHeight,
	   gInChannel, gInWidth, gInHeight,
	   kernWidth, kernHeight,
	   padWidth, padHeight,
	   gInChannelGroup, gOutChannelGroup,
	   gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	   n, c, nH) ;
	c+=13 ;
	break ;
      case 14:
	func<FLAYOUT, 14>(pGOut, pKernel, pGIn,
	   gOutChannel, gOutWidth, gOutHeight,
	   gInChannel, gInWidth, gInHeight,
	   kernWidth, kernHeight,
	   padWidth, padHeight,
	   gInChannelGroup, gOutChannelGroup,
	   gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	   n, c, nH) ;
	c+=14 ;
	break ;
      case 15:
	func<FLAYOUT, 15>(pGOut, pKernel, pGIn,
	   gOutChannel, gOutWidth, gOutHeight,
	   gInChannel, gInWidth, gInHeight,
	   kernWidth, kernHeight,
	   padWidth, padHeight,
	   gInChannelGroup, gOutChannelGroup,
	   gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	   n, c, nH) ;
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
	   padWidth, padHeight,
	   gInChannelGroup, gOutChannelGroup,
	   gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	   n, c, nH) ;
	c+= 16 ;
      } // gInChannel
    } // group
  } // batch
}

extern "C"
vednnError_t
vednnConvolutionBackwardData_direct_dil1_str2_ker3_iwU256 (
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
