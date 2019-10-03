#include <stdint.h>
#include <stdlib.h>

#include "vednn.h"
#include "vednn_util.h"

#include "velintrin.h"
#define VLEN	(256)

template<filterLayout_t FLAYOUT, int NUMCHANNEL>
static __attribute__((noinline)) void func_odd(
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
    const int64_t c,
    const int64_t nH
)
{
  int64_t gInIndex = gInGroupOffset + ((n * gInChannel + c) * gInHeight ) * gInWidth  ;

  __vr vrseq = _vel_vseq_vl(nH*gInWidth) ;
  __vr vrh  = _vel_vdivsl_vvsl(vrseq, gInWidth, nH*gInWidth) ;
  __vr vrw  = _vel_vsubsl_vvvl(vrseq, _vel_vmulul_vsvl(gInWidth,vrh, nH*gInWidth), nH*gInWidth) ;

  for (int64_t h=0; h<gInHeight; h+=nH) {
    const int64_t vl = gInWidth * (gInHeight - h < nH ? gInHeight - h : nH) ;
    const int64_t gip = h * gInWidth ;

    __vr vrsum0_s12  = _vel_vbrds_vsl(0.f, vl) ;
    __vr vrsum12_s12 = _vel_pvbrd_vsl(0UL, vl) ;
    __vr vrsum34_s12 = _vel_pvbrd_vsl(0UL, vl) ;
    __vr vrsum56_s12 = _vel_pvbrd_vsl(0UL, vl) ;
    __vr vrsum78_s12 = _vel_pvbrd_vsl(0UL, vl) ;
    __vr vrsum9A_s12 = _vel_pvbrd_vsl(0UL, vl) ;
    __vr vrsumBC_s12 = _vel_pvbrd_vsl(0UL, vl) ;
    __vr vrsumDE_s12 = _vel_pvbrd_vsl(0UL, vl) ;

    __vr vrsum0_s0  = _vel_vbrds_vsl(0.f, vl) ;
    __vr vrsum12_s0 = _vel_pvbrd_vsl(0UL, vl) ;
    __vr vrsum34_s0 = _vel_pvbrd_vsl(0UL, vl) ;
    __vr vrsum56_s0 = _vel_pvbrd_vsl(0UL, vl) ;
    __vr vrsum78_s0 = _vel_pvbrd_vsl(0UL, vl) ;
    __vr vrsum9A_s0 = _vel_pvbrd_vsl(0UL, vl) ;
    __vr vrsumBC_s0 = _vel_pvbrd_vsl(0UL, vl) ;
    __vr vrsumDE_s0 = _vel_pvbrd_vsl(0UL, vl) ;

    __vr vri_r0 = _vel_vaddsl_vsvl(padHeight-0*dilationHeight+h, vrh, vl) ;
    __vr vri_r1 = _vel_vaddsl_vsvl(padHeight-1*dilationHeight+h, vrh, vl) ;
    __vr vri_r2 = _vel_vaddsl_vsvl(padHeight-2*dilationHeight+h, vrh, vl) ;

    __vr vry_r0 = _vel_vdivsl_vvsl(vri_r0, strideHeight, vl) ;
    __vr vry_r1 = _vel_vdivsl_vvsl(vri_r1, strideHeight, vl) ;
    __vr vry_r2 = _vel_vdivsl_vvsl(vri_r2, strideHeight, vl) ;

    __vr vrj_s0 = _vel_vaddsl_vsvl(padWidth-0*dilationWidth, vrw, vl) ;
    __vr vrj_s1 = _vel_vaddsl_vsvl(padWidth-1*dilationWidth, vrw, vl) ;
    __vr vrj_s2 = _vel_vaddsl_vsvl(padWidth-2*dilationWidth, vrw, vl) ;

    __vr vrx_s0 = _vel_vdivsl_vvsl(vrj_s0, strideWidth, vl) ;
    __vr vrx_s1 = _vel_vdivsl_vvsl(vrj_s1, strideWidth, vl) ;
    __vr vrx_s2 = _vel_vdivsl_vvsl(vrj_s2, strideWidth, vl) ;


    __vm256 vmy0_r0 =  _vel_vfmkleq_mvl(_vel_vcmpsl_vvvl(vri_r0, _vel_vmulsl_vsvl(strideHeight, vry_r0, vl), vl), vl) ;
    __vm256 vmy1_r0 =  _vel_vfmklge_mvl(vry_r0, vl) ;
    __vm256 vmy2_r0 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutHeight,vry_r0, vl), vl) ;
    __vm256 vmy_r0 = _vel_andm_mmm(_vel_andm_mmm(vmy0_r0, vmy1_r0), vmy2_r0) ;

    __vm256 vmy0_r1 =  _vel_vfmkleq_mvl(_vel_vcmpsl_vvvl(vri_r1, _vel_vmulsl_vsvl(strideHeight, vry_r1, vl), vl), vl) ;
    __vm256 vmy1_r1 =  _vel_vfmklge_mvl(vry_r1, vl) ;
    __vm256 vmy2_r1 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutHeight,vry_r1, vl), vl) ;
    __vm256 vmy_r1 = _vel_andm_mmm(_vel_andm_mmm(vmy0_r1, vmy1_r1), vmy2_r1) ;

    __vm256 vmy0_r2 =  _vel_vfmkleq_mvl(_vel_vcmpsl_vvvl(vri_r2, _vel_vmulsl_vsvl(strideHeight, vry_r2, vl), vl), vl) ;
    __vm256 vmy1_r2 =  _vel_vfmklge_mvl(vry_r2, vl) ;
    __vm256 vmy2_r2 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutHeight,vry_r2, vl), vl) ;
    __vm256 vmy_r2 = _vel_andm_mmm(_vel_andm_mmm(vmy0_r2, vmy1_r2), vmy2_r2) ;

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

    __vm256 vmall_r0s0 = _vel_andm_mmm(vmy_r0,vmx_s0) ;
    __vm256 vmall_r0s1 = _vel_andm_mmm(vmy_r0,vmx_s1) ;
    __vm256 vmall_r0s2 = _vel_andm_mmm(vmy_r0,vmx_s2) ;
    __vm256 vmall_r0   = _vel_orm_mmm(vmall_r0s1, vmall_r0s2) ;

    __vm256 vmall_r1s0 = _vel_andm_mmm(vmy_r1,vmx_s0) ;
    __vm256 vmall_r1s1 = _vel_andm_mmm(vmy_r1,vmx_s1) ;
    __vm256 vmall_r1s2 = _vel_andm_mmm(vmy_r1,vmx_s2) ;
    __vm256 vmall_r1   = _vel_orm_mmm(vmall_r1s1, vmall_r1s2) ;

    __vm256 vmall_r2s0 = _vel_andm_mmm(vmy_r2,vmx_s0) ;
    __vm256 vmall_r2s1 = _vel_andm_mmm(vmy_r2,vmx_s1) ;
    __vm256 vmall_r2s2 = _vel_andm_mmm(vmy_r2,vmx_s2) ;
    __vm256 vmall_r2   = _vel_orm_mmm(vmall_r2s1, vmall_r2s2) ;

    int64_t k=0;
    if( (gOutChannelGroup & 0x01 ) == 1 ) {
      int64_t gOutIndex    = gOutGroupOffset + ((n * gOutChannel + k) * gOutHeight) * gOutWidth ;

#define FILTER_OFFSET(k,c,r,s) ( kernGroupOffset + filter_index<FLAYOUT>(k,c,r,s, gInChannelGroup, gOutChannelGroup, kernHeight, kernWidth) )
#define FILTER_DISTANCE_BY_C()   ( FLAYOUT == VEDNN_FILTER_LAYOUT_NCHW ? kernHeight * kernWidth : gOutChannelGroup ) ;
#define VFADD(VRGOUT, VM, K, R, S, STOKEN)	{							\
	const int64_t filter_offset   = FILTER_OFFSET(K,c+ 0,R,S) ;					\
	const int64_t filter_distance = FILTER_DISTANCE_BY_C() ;					\
	const float    kerValue0  =  pKernel[filter_offset + 0 * filter_distance] ;			\
	const uint64_t kerValue12 = _vel_pack_f32p(pKernel + filter_offset + 1 * filter_distance,	\
						   pKernel + filter_offset + 2 * filter_distance) ;	\
	const uint64_t kerValue34 = _vel_pack_f32p(pKernel + filter_offset + 3 * filter_distance,	\
						   pKernel + filter_offset + 4 * filter_distance) ;	\
	const uint64_t kerValue56 = _vel_pack_f32p(pKernel + filter_offset + 5 * filter_distance,	\
						   pKernel + filter_offset + 6 * filter_distance) ;	\
	const uint64_t kerValue78 = _vel_pack_f32p(pKernel + filter_offset + 7 * filter_distance,	\
						   pKernel + filter_offset + 8 * filter_distance) ;	\
	const uint64_t kerValue9A = _vel_pack_f32p(pKernel + filter_offset + 9 * filter_distance,	\
						   pKernel + filter_offset +10 * filter_distance) ;	\
	const uint64_t kerValueBC = _vel_pack_f32p(pKernel + filter_offset +11 * filter_distance,	\
						   pKernel + filter_offset +12 * filter_distance) ;	\
	const uint64_t kerValueDE = _vel_pack_f32p(pKernel + filter_offset +13 * filter_distance,	\
						   pKernel + filter_offset +14 * filter_distance) ;	\
	__vr vrgout  = _vel_vmrg_vsvml(0.f, VRGOUT, VM, vl) ;						\
	__vr vrgoutP = _vel_vshf_vvvsl(vrgout, vrgout, VE_VSHUFFLE_YUZU, vl) ;				\
	vrsum0_##STOKEN = _vel_vfmads_vvsvl(vrsum0_##STOKEN, kerValue0, vrgout, vl) ;			\
	if(NUMCHANNEL>= 3) vrsum12_##STOKEN = _vel_pvfmad_vvsvl(vrsum12_##STOKEN, kerValue12, vrgoutP, vl) ;	\
	if(NUMCHANNEL>= 5) vrsum34_##STOKEN = _vel_pvfmad_vvsvl(vrsum34_##STOKEN, kerValue34, vrgoutP, vl) ;	\
	if(NUMCHANNEL>= 7) vrsum56_##STOKEN = _vel_pvfmad_vvsvl(vrsum56_##STOKEN, kerValue56, vrgoutP, vl) ;	\
	if(NUMCHANNEL>= 9) vrsum78_##STOKEN = _vel_pvfmad_vvsvl(vrsum78_##STOKEN, kerValue78, vrgoutP, vl) ;	\
	if(NUMCHANNEL>=11) vrsum9A_##STOKEN = _vel_pvfmad_vvsvl(vrsum9A_##STOKEN, kerValue9A, vrgoutP, vl) ;	\
	if(NUMCHANNEL>=13) vrsumBC_##STOKEN = _vel_pvfmad_vvsvl(vrsumBC_##STOKEN, kerValueBC, vrgoutP, vl) ;	\
	if(NUMCHANNEL>=15) vrsumDE_##STOKEN = _vel_pvfmad_vvsvl(vrsumDE_##STOKEN, kerValueDE, vrgoutP, vl) ;	\
      }

      __vr vrgout_ptr_k0_r0_s1 = _vel_vsfa_vvssl(_vel_vaddsl_vvvl(_vel_vmulsl_vsvl(gOutWidth, vry_r0, vl), vrx_s1, vl),
					 2,
					 (unsigned long)(pGOut+gOutIndex), vl) ;
      __vr vrgout_k0_r0_s1 = _vel_vgtu_vvssml(vrgout_ptr_k0_r0_s1, 0, 0, vmall_r0, vl) ;

      VFADD(vrgout_k0_r0_s1, vmall_r0s2, k+0, 0, 2, s12) ;

      VFADD(vrgout_k0_r0_s1, vmall_r0s1, k+0, 0, 1, s12) ;

      VFADD(vrgout_k0_r0_s1, vmall_r0s1, k+0, 0, 0, s0) ;

      __vr vrgout_ptr_k0_r1_s1 = _vel_vsfa_vvssl(_vel_vaddsl_vvvl(_vel_vmulsl_vsvl(gOutWidth, vry_r1, vl), vrx_s1, vl),
					 2,
					 (unsigned long)(pGOut+gOutIndex), vl) ;
      __vr vrgout_k0_r1_s1 = _vel_vgtu_vvssml(vrgout_ptr_k0_r1_s1, 0, 0, vmall_r1, vl) ;

      VFADD(vrgout_k0_r1_s1, vmall_r1s2, k+0, 1, 2, s12) ;

      VFADD(vrgout_k0_r1_s1, vmall_r1s1, k+0, 1, 1, s12) ;

      VFADD(vrgout_k0_r1_s1, vmall_r1s1, k+0, 1, 0, s0) ;

      __vr vrgout_ptr_k0_r2_s1 = _vel_vsfa_vvssl(_vel_vaddsl_vvvl(_vel_vmulsl_vsvl(gOutWidth, vry_r2, vl), vrx_s1, vl),
					 2,
					 (unsigned long)(pGOut+gOutIndex), vl) ;
      __vr vrgout_k0_r2_s1 = _vel_vgtu_vvssml(vrgout_ptr_k0_r2_s1, 0, 0, vmall_r2 , vl) ;

      VFADD(vrgout_k0_r2_s1, vmall_r2s2, k+0, 2, 2, s12) ;

      VFADD(vrgout_k0_r2_s1, vmall_r2s1, k+0, 2, 1, s12) ;

      VFADD(vrgout_k0_r2_s1, vmall_r2s1, k+0, 2, 0, s0) ;

      k+=1 ;
    }
    if( ((gOutChannelGroup >> 1) & 0x01 ) == 1 ) {
      int64_t gOutIndex    = gOutGroupOffset + ((n * gOutChannel + k) * gOutHeight) * gOutWidth ;

      __vr vrgout_ptr_k0_r0_s1 = _vel_vsfa_vvssl(_vel_vaddsl_vvvl(_vel_vmulsl_vsvl(gOutWidth, vry_r0, vl), vrx_s1, vl),
					 2,
					 (unsigned long)(pGOut+gOutIndex), vl) ;
      __vr vrgout_k0_r0_s1 = _vel_vgtu_vvssml(vrgout_ptr_k0_r0_s1, 0, 0, vmall_r0, vl) ;
      __vr vrgout_ptr_k1_r0_s1 = _vel_vaddsl_vsvl(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_r0_s1, vl) ;
      __vr vrgout_k1_r0_s1 = _vel_vgtu_vvssml(vrgout_ptr_k1_r0_s1, 0, 0, vmall_r0, vl) ;

      VFADD(vrgout_k0_r0_s1, vmall_r0s2, k+0, 0, 2, s12) ;
      VFADD(vrgout_k1_r0_s1, vmall_r0s2, k+1, 0, 2, s12) ;

      VFADD(vrgout_k0_r0_s1, vmall_r0s1, k+0, 0, 1, s12) ;
      VFADD(vrgout_k1_r0_s1, vmall_r0s1, k+1, 0, 1, s12) ;

      VFADD(vrgout_k0_r0_s1, vmall_r0s1, k+0, 0, 0, s0) ;
      VFADD(vrgout_k1_r0_s1, vmall_r0s1, k+1, 0, 0, s0) ;

      __vr vrgout_ptr_k0_r1_s1 = _vel_vsfa_vvssl(_vel_vaddsl_vvvl(_vel_vmulsl_vsvl(gOutWidth, vry_r1, vl), vrx_s1, vl),
					 2,
					 (unsigned long)(pGOut+gOutIndex), vl) ;
      __vr vrgout_k0_r1_s1 = _vel_vgtu_vvssml(vrgout_ptr_k0_r1_s1, 0, 0, vmall_r1, vl) ;
      __vr vrgout_ptr_k1_r1_s1 = _vel_vaddsl_vsvl(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_r1_s1, vl) ;
      __vr vrgout_k1_r1_s1 = _vel_vgtu_vvssml(vrgout_ptr_k1_r1_s1, 0, 0, vmall_r1, vl) ;

      VFADD(vrgout_k0_r1_s1, vmall_r1s2, k+0, 1, 2, s12) ;
      VFADD(vrgout_k1_r1_s1, vmall_r1s2, k+1, 1, 2, s12) ;

      VFADD(vrgout_k0_r1_s1, vmall_r1s1, k+0, 1, 1, s12) ;
      VFADD(vrgout_k1_r1_s1, vmall_r1s1, k+1, 1, 1, s12) ;

      VFADD(vrgout_k0_r1_s1, vmall_r1s1, k+0, 1, 0, s0) ;
      VFADD(vrgout_k1_r1_s1, vmall_r1s1, k+1, 1, 0, s0) ;

      __vr vrgout_ptr_k0_r2_s1 = _vel_vsfa_vvssl(_vel_vaddsl_vvvl(_vel_vmulsl_vsvl(gOutWidth, vry_r2, vl), vrx_s1, vl),
					 2,
					 (unsigned long)(pGOut+gOutIndex), vl) ;
      __vr vrgout_k0_r2_s1 = _vel_vgtu_vvssml(vrgout_ptr_k0_r2_s1, 0, 0, vmall_r2 , vl) ;
      __vr vrgout_ptr_k1_r2_s1 = _vel_vaddsl_vsvl(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_r2_s1, vl) ;
      __vr vrgout_k1_r2_s1 = _vel_vgtu_vvssml(vrgout_ptr_k1_r2_s1, 0, 0, vmall_r2 , vl) ;

      VFADD(vrgout_k0_r2_s1, vmall_r2s2, k+0, 2, 2, s12) ;
      VFADD(vrgout_k1_r2_s1, vmall_r2s2, k+1, 2, 2, s12) ;

      VFADD(vrgout_k0_r2_s1, vmall_r2s1, k+0, 2, 1, s12) ;
      VFADD(vrgout_k1_r2_s1, vmall_r2s1, k+1, 2, 1, s12) ;

      VFADD(vrgout_k0_r2_s1, vmall_r2s1, k+0, 2, 0, s0) ;
      VFADD(vrgout_k1_r2_s1, vmall_r2s1, k+1, 2, 0, s0) ;

      k+=2 ;
    }
#if 0
    if( ((gOutChannelGroup >> 2) & 0x01 ) == 1 ) {
      int64_t gOutIndex    = gOutGroupOffset + ((n * gOutChannel + k) * gOutHeight) * gOutWidth ;

      __vr vrgout_ptr_k0_r0_s1 = _vel_vsfa_vvssl(_vel_vaddsl_vvvl(_vel_vmulsl_vsvl(gOutWidth, vry_r0, vl), vrx_s1, vl),
					 2,
					 (unsigned long)(pGOut+gOutIndex), vl) ;
      __vr vrgout_k0_r0_s1 = _vel_vgtu_vvssml(vrgout_ptr_k0_r0_s1, 0, 0, vmall_r0, vl) ;
      __vr vrgout_ptr_k1_r0_s1 = _vel_vaddsl_vsvl(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_r0_s1, vl) ;
      __vr vrgout_k1_r0_s1 = _vel_vgtu_vvssml(vrgout_ptr_k1_r0_s1, 0, 0, vmall_r0, vl) ;
      __vr vrgout_ptr_k2_r0_s1 = _vel_vaddsl_vsvl(4*2*gOutHeight*gOutWidth, vrgout_ptr_k0_r0_s1, vl) ;
      __vr vrgout_k2_r0_s1 = _vel_vgtu_vvssml(vrgout_ptr_k2_r0_s1, 0, 0, vmall_r0, vl) ;
      __vr vrgout_ptr_k3_r0_s1 = _vel_vaddsl_vsvl(4*3*gOutHeight*gOutWidth, vrgout_ptr_k0_r0_s1, vl) ;
      __vr vrgout_k3_r0_s1 = _vel_vgtu_vvssml(vrgout_ptr_k3_r0_s1, 0, 0, vmall_r0, vl) ;

      VFADD(vrgout_k0_r0_s1, vmall_r0s2, k+0, 0, 2, s12) ;
      VFADD(vrgout_k1_r0_s1, vmall_r0s2, k+1, 0, 2, s12) ;
      VFADD(vrgout_k2_r0_s1, vmall_r0s2, k+2, 0, 2, s12) ;
      VFADD(vrgout_k3_r0_s1, vmall_r0s2, k+3, 0, 2, s12) ;

      VFADD(vrgout_k0_r0_s1, vmall_r0s1, k+0, 0, 1, s12) ;
      VFADD(vrgout_k1_r0_s1, vmall_r0s1, k+1, 0, 1, s12) ;
      VFADD(vrgout_k2_r0_s1, vmall_r0s1, k+2, 0, 1, s12) ;
      VFADD(vrgout_k3_r0_s1, vmall_r0s1, k+3, 0, 1, s12) ;

      VFADD(vrgout_k0_r0_s1, vmall_r0s1, k+0, 0, 0, s0) ;
      VFADD(vrgout_k1_r0_s1, vmall_r0s1, k+1, 0, 0, s0) ;
      VFADD(vrgout_k2_r0_s1, vmall_r0s1, k+2, 0, 0, s0) ;
      VFADD(vrgout_k3_r0_s1, vmall_r0s1, k+3, 0, 0, s0) ;

      __vr vrgout_ptr_k0_r1_s1 = _vel_vsfa_vvssl(_vel_vaddsl_vvvl(_vel_vmulsl_vsvl(gOutWidth, vry_r1, vl), vrx_s1, vl),
					 2,
					 (unsigned long)(pGOut+gOutIndex), vl) ;
      __vr vrgout_k0_r1_s1 = _vel_vgtu_vvssml(vrgout_ptr_k0_r1_s1, 0, 0, vmall_r1, vl) ;
      __vr vrgout_ptr_k1_r1_s1 = _vel_vaddsl_vsvl(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_r1_s1, vl) ;
      __vr vrgout_k1_r1_s1 = _vel_vgtu_vvssml(vrgout_ptr_k1_r1_s1, 0, 0, vmall_r1, vl) ;
      __vr vrgout_ptr_k2_r1_s1 = _vel_vaddsl_vsvl(4*2*gOutHeight*gOutWidth, vrgout_ptr_k0_r1_s1, vl) ;
      __vr vrgout_k2_r1_s1 = _vel_vgtu_vvssml(vrgout_ptr_k2_r1_s1, 0, 0, vmall_r1, vl) ;
      __vr vrgout_ptr_k3_r1_s1 = _vel_vaddsl_vsvl(4*3*gOutHeight*gOutWidth, vrgout_ptr_k0_r1_s1, vl) ;
      __vr vrgout_k3_r1_s1 = _vel_vgtu_vvssml(vrgout_ptr_k3_r1_s1, 0, 0, vmall_r1, vl) ;

      VFADD(vrgout_k0_r1_s1, vmall_r1s2, k+0, 1, 2, s12) ;
      VFADD(vrgout_k1_r1_s1, vmall_r1s2, k+1, 1, 2, s12) ;
      VFADD(vrgout_k2_r1_s1, vmall_r1s2, k+2, 1, 2, s12) ;
      VFADD(vrgout_k3_r1_s1, vmall_r1s2, k+3, 1, 2, s12) ;

      VFADD(vrgout_k0_r1_s1, vmall_r1s1, k+0, 1, 1, s12) ;
      VFADD(vrgout_k1_r1_s1, vmall_r1s1, k+1, 1, 1, s12) ;
      VFADD(vrgout_k2_r1_s1, vmall_r1s1, k+2, 1, 1, s12) ;
      VFADD(vrgout_k3_r1_s1, vmall_r1s1, k+3, 1, 1, s12) ;

      VFADD(vrgout_k0_r1_s1, vmall_r1s1, k+0, 1, 0, s0) ;
      VFADD(vrgout_k1_r1_s1, vmall_r1s1, k+1, 1, 0, s0) ;
      VFADD(vrgout_k2_r1_s1, vmall_r1s1, k+2, 1, 0, s0) ;
      VFADD(vrgout_k3_r1_s1, vmall_r1s1, k+3, 1, 0, s0) ;


      __vr vrgout_ptr_k0_r2_s1 = _vel_vsfa_vvssl(_vel_vaddsl_vvvl(_vel_vmulsl_vsvl(gOutWidth, vry_r2, vl), vrx_s1, vl),
					 2,
					 (unsigned long)(pGOut+gOutIndex), vl) ;
      __vr vrgout_k0_r2_s1 = _vel_vgtu_vvssml(vrgout_ptr_k0_r2_s1, 0, 0, vmall_r2 , vl) ;
      __vr vrgout_ptr_k1_r2_s1 = _vel_vaddsl_vsvl(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_r2_s1, vl) ;
      __vr vrgout_k1_r2_s1 = _vel_vgtu_vvssml(vrgout_ptr_k1_r2_s1, 0, 0, vmall_r2 , vl) ;
      __vr vrgout_ptr_k2_r2_s1 = _vel_vaddsl_vsvl(4*2*gOutHeight*gOutWidth, vrgout_ptr_k0_r2_s1, vl) ;
      __vr vrgout_k2_r2_s1 = _vel_vgtu_vvssml(vrgout_ptr_k2_r2_s1, 0, 0, vmall_r2 , vl) ;
      __vr vrgout_ptr_k3_r2_s1 = _vel_vaddsl_vsvl(4*3*gOutHeight*gOutWidth, vrgout_ptr_k0_r2_s1, vl) ;
      __vr vrgout_k3_r2_s1 = _vel_vgtu_vvssml(vrgout_ptr_k3_r2_s1, 0, 0, vmall_r2 , vl) ;

      VFADD(vrgout_k0_r2_s1, vmall_r2s2, k+0, 2, 2, s12) ;
      VFADD(vrgout_k1_r2_s1, vmall_r2s2, k+1, 2, 2, s12) ;
      VFADD(vrgout_k2_r2_s1, vmall_r2s2, k+2, 2, 2, s12) ;
      VFADD(vrgout_k3_r2_s1, vmall_r2s2, k+3, 2, 2, s12) ;

      VFADD(vrgout_k0_r2_s1, vmall_r2s1, k+0, 2, 1, s12) ;
      VFADD(vrgout_k1_r2_s1, vmall_r2s1, k+1, 2, 1, s12) ;
      VFADD(vrgout_k2_r2_s1, vmall_r2s1, k+2, 2, 1, s12) ;
      VFADD(vrgout_k3_r2_s1, vmall_r2s1, k+3, 2, 1, s12) ;

      VFADD(vrgout_k0_r2_s1, vmall_r2s1, k+0, 2, 0, s0) ;
      VFADD(vrgout_k1_r2_s1, vmall_r2s1, k+1, 2, 0, s0) ;
      VFADD(vrgout_k2_r2_s1, vmall_r2s1, k+2, 2, 0, s0) ;
      VFADD(vrgout_k3_r2_s1, vmall_r2s1, k+3, 2, 0, s0) ;

      k+=4 ;
    }
    for (; k<gOutChannelGroup; k+=8) {
      int64_t gOutIndex    = gOutGroupOffset + ((n * gOutChannel + k) * gOutHeight) * gOutWidth ;

      __vr vrgout_ptr_k0_r0_s1 = _vel_vsfa_vvssl(_vel_vaddsl_vvvl(_vel_vmulsl_vsvl(gOutWidth, vry_r0, vl), vrx_s1, vl),
					 2,
					 (unsigned long)(pGOut+gOutIndex), vl) ;
      __vr vrgout_k0_r0_s1 = _vel_vgtu_vvssml(vrgout_ptr_k0_r0_s1, 0, 0, vmall_r0, vl) ;
      __vr vrgout_ptr_k1_r0_s1 = _vel_vaddsl_vsvl(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_r0_s1, vl) ;
      __vr vrgout_k1_r0_s1 = _vel_vgtu_vvssml(vrgout_ptr_k1_r0_s1, 0, 0, vmall_r0, vl) ;
      __vr vrgout_ptr_k2_r0_s1 = _vel_vaddsl_vsvl(4*2*gOutHeight*gOutWidth, vrgout_ptr_k0_r0_s1, vl) ;
      __vr vrgout_k2_r0_s1 = _vel_vgtu_vvssml(vrgout_ptr_k2_r0_s1, 0, 0, vmall_r0, vl) ;
      __vr vrgout_ptr_k3_r0_s1 = _vel_vaddsl_vsvl(4*3*gOutHeight*gOutWidth, vrgout_ptr_k0_r0_s1, vl) ;
      __vr vrgout_k3_r0_s1 = _vel_vgtu_vvssml(vrgout_ptr_k3_r0_s1, 0, 0, vmall_r0, vl) ;
      __vr vrgout_ptr_k4_r0_s1 = _vel_vaddsl_vsvl(4*4*gOutHeight*gOutWidth, vrgout_ptr_k0_r0_s1, vl) ;
      __vr vrgout_k4_r0_s1 = _vel_vgtu_vvssml(vrgout_ptr_k4_r0_s1, 0, 0, vmall_r0, vl) ;
      __vr vrgout_ptr_k5_r0_s1 = _vel_vaddsl_vsvl(4*5*gOutHeight*gOutWidth, vrgout_ptr_k0_r0_s1, vl) ;
      __vr vrgout_k5_r0_s1 = _vel_vgtu_vvssml(vrgout_ptr_k5_r0_s1, 0, 0, vmall_r0, vl) ;
      __vr vrgout_ptr_k6_r0_s1 = _vel_vaddsl_vsvl(4*6*gOutHeight*gOutWidth, vrgout_ptr_k0_r0_s1, vl) ;
      __vr vrgout_k6_r0_s1 = _vel_vgtu_vvssml(vrgout_ptr_k6_r0_s1, 0, 0, vmall_r0, vl) ;
      __vr vrgout_ptr_k7_r0_s1 = _vel_vaddsl_vsvl(4*7*gOutHeight*gOutWidth, vrgout_ptr_k0_r0_s1, vl) ;
      __vr vrgout_k7_r0_s1 = _vel_vgtu_vvssml(vrgout_ptr_k7_r0_s1, 0, 0, vmall_r0, vl) ;

      VFADD(vrgout_k0_r0_s1, vmall_r0s2, k+0, 0, 2, s12) ;
      VFADD(vrgout_k1_r0_s1, vmall_r0s2, k+1, 0, 2, s12) ;
      VFADD(vrgout_k2_r0_s1, vmall_r0s2, k+2, 0, 2, s12) ;
      VFADD(vrgout_k3_r0_s1, vmall_r0s2, k+3, 0, 2, s12) ;
      VFADD(vrgout_k4_r0_s1, vmall_r0s2, k+4, 0, 2, s12) ;
      VFADD(vrgout_k5_r0_s1, vmall_r0s2, k+5, 0, 2, s12) ;
      VFADD(vrgout_k6_r0_s1, vmall_r0s2, k+6, 0, 2, s12) ;
      VFADD(vrgout_k7_r0_s1, vmall_r0s2, k+7, 0, 2, s12) ;

      VFADD(vrgout_k0_r0_s1, vmall_r0s1, k+0, 0, 1, s12) ;
      VFADD(vrgout_k1_r0_s1, vmall_r0s1, k+1, 0, 1, s12) ;
      VFADD(vrgout_k2_r0_s1, vmall_r0s1, k+2, 0, 1, s12) ;
      VFADD(vrgout_k3_r0_s1, vmall_r0s1, k+3, 0, 1, s12) ;
      VFADD(vrgout_k4_r0_s1, vmall_r0s1, k+4, 0, 1, s12) ;
      VFADD(vrgout_k5_r0_s1, vmall_r0s1, k+5, 0, 1, s12) ;
      VFADD(vrgout_k6_r0_s1, vmall_r0s1, k+6, 0, 1, s12) ;
      VFADD(vrgout_k7_r0_s1, vmall_r0s1, k+7, 0, 1, s12) ;

      VFADD(vrgout_k0_r0_s1, vmall_r0s1, k+0, 0, 0, s0) ;
      VFADD(vrgout_k1_r0_s1, vmall_r0s1, k+1, 0, 0, s0) ;
      VFADD(vrgout_k2_r0_s1, vmall_r0s1, k+2, 0, 0, s0) ;
      VFADD(vrgout_k3_r0_s1, vmall_r0s1, k+3, 0, 0, s0) ;
      VFADD(vrgout_k4_r0_s1, vmall_r0s1, k+4, 0, 0, s0) ;
      VFADD(vrgout_k5_r0_s1, vmall_r0s1, k+5, 0, 0, s0) ;
      VFADD(vrgout_k6_r0_s1, vmall_r0s1, k+6, 0, 0, s0) ;
      VFADD(vrgout_k7_r0_s1, vmall_r0s1, k+7, 0, 0, s0) ;


      __vr vrgout_ptr_k0_r1_s1 = _vel_vsfa_vvssl(_vel_vaddsl_vvvl(_vel_vmulsl_vsvl(gOutWidth, vry_r1, vl), vrx_s1, vl),
					 2,
					 (unsigned long)(pGOut+gOutIndex), vl) ;
      __vr vrgout_k0_r1_s1 = _vel_vgtu_vvssml(vrgout_ptr_k0_r1_s1, 0, 0, vmall_r1, vl) ;
      __vr vrgout_ptr_k1_r1_s1 = _vel_vaddsl_vsvl(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_r1_s1, vl) ;
      __vr vrgout_k1_r1_s1 = _vel_vgtu_vvssml(vrgout_ptr_k1_r1_s1, 0, 0, vmall_r1, vl) ;
      __vr vrgout_ptr_k2_r1_s1 = _vel_vaddsl_vsvl(4*2*gOutHeight*gOutWidth, vrgout_ptr_k0_r1_s1, vl) ;
      __vr vrgout_k2_r1_s1 = _vel_vgtu_vvssml(vrgout_ptr_k2_r1_s1, 0, 0, vmall_r1, vl) ;
      __vr vrgout_ptr_k3_r1_s1 = _vel_vaddsl_vsvl(4*3*gOutHeight*gOutWidth, vrgout_ptr_k0_r1_s1, vl) ;
      __vr vrgout_k3_r1_s1 = _vel_vgtu_vvssml(vrgout_ptr_k3_r1_s1, 0, 0, vmall_r1, vl) ;
      __vr vrgout_ptr_k4_r1_s1 = _vel_vaddsl_vsvl(4*4*gOutHeight*gOutWidth, vrgout_ptr_k0_r1_s1, vl) ;
      __vr vrgout_k4_r1_s1 = _vel_vgtu_vvssml(vrgout_ptr_k4_r1_s1, 0, 0, vmall_r1, vl) ;
      __vr vrgout_ptr_k5_r1_s1 = _vel_vaddsl_vsvl(4*5*gOutHeight*gOutWidth, vrgout_ptr_k0_r1_s1, vl) ;
      __vr vrgout_k5_r1_s1 = _vel_vgtu_vvssml(vrgout_ptr_k5_r1_s1, 0, 0, vmall_r1, vl) ;
      __vr vrgout_ptr_k6_r1_s1 = _vel_vaddsl_vsvl(4*6*gOutHeight*gOutWidth, vrgout_ptr_k0_r1_s1, vl) ;
      __vr vrgout_k6_r1_s1 = _vel_vgtu_vvssml(vrgout_ptr_k6_r1_s1, 0, 0, vmall_r1, vl) ;
      __vr vrgout_ptr_k7_r1_s1 = _vel_vaddsl_vsvl(4*7*gOutHeight*gOutWidth, vrgout_ptr_k0_r1_s1, vl) ;
      __vr vrgout_k7_r1_s1 = _vel_vgtu_vvssml(vrgout_ptr_k7_r1_s1, 0, 0, vmall_r1, vl) ;

      VFADD(vrgout_k0_r1_s1, vmall_r1s2, k+0, 1, 2, s12) ;
      VFADD(vrgout_k1_r1_s1, vmall_r1s2, k+1, 1, 2, s12) ;
      VFADD(vrgout_k2_r1_s1, vmall_r1s2, k+2, 1, 2, s12) ;
      VFADD(vrgout_k3_r1_s1, vmall_r1s2, k+3, 1, 2, s12) ;
      VFADD(vrgout_k4_r1_s1, vmall_r1s2, k+4, 1, 2, s12) ;
      VFADD(vrgout_k5_r1_s1, vmall_r1s2, k+5, 1, 2, s12) ;
      VFADD(vrgout_k6_r1_s1, vmall_r1s2, k+6, 1, 2, s12) ;
      VFADD(vrgout_k7_r1_s1, vmall_r1s2, k+7, 1, 2, s12) ;

      VFADD(vrgout_k0_r1_s1, vmall_r1s1, k+0, 1, 1, s12) ;
      VFADD(vrgout_k1_r1_s1, vmall_r1s1, k+1, 1, 1, s12) ;
      VFADD(vrgout_k2_r1_s1, vmall_r1s1, k+2, 1, 1, s12) ;
      VFADD(vrgout_k3_r1_s1, vmall_r1s1, k+3, 1, 1, s12) ;
      VFADD(vrgout_k4_r1_s1, vmall_r1s1, k+4, 1, 1, s12) ;
      VFADD(vrgout_k5_r1_s1, vmall_r1s1, k+5, 1, 1, s12) ;
      VFADD(vrgout_k6_r1_s1, vmall_r1s1, k+6, 1, 1, s12) ;
      VFADD(vrgout_k7_r1_s1, vmall_r1s1, k+7, 1, 1, s12) ;

      VFADD(vrgout_k0_r1_s1, vmall_r1s1, k+0, 1, 0, s0) ;
      VFADD(vrgout_k1_r1_s1, vmall_r1s1, k+1, 1, 0, s0) ;
      VFADD(vrgout_k2_r1_s1, vmall_r1s1, k+2, 1, 0, s0) ;
      VFADD(vrgout_k3_r1_s1, vmall_r1s1, k+3, 1, 0, s0) ;
      VFADD(vrgout_k4_r1_s1, vmall_r1s1, k+4, 1, 0, s0) ;
      VFADD(vrgout_k5_r1_s1, vmall_r1s1, k+5, 1, 0, s0) ;
      VFADD(vrgout_k6_r1_s1, vmall_r1s1, k+6, 1, 0, s0) ;
      VFADD(vrgout_k7_r1_s1, vmall_r1s1, k+7, 1, 0, s0) ;


      __vr vrgout_ptr_k0_r2_s1 = _vel_vsfa_vvssl(_vel_vaddsl_vvvl(_vel_vmulsl_vsvl(gOutWidth, vry_r2, vl), vrx_s1, vl),
					 2,
					 (unsigned long)(pGOut+gOutIndex), vl) ;
      __vr vrgout_k0_r2_s1 = _vel_vgtu_vvssml(vrgout_ptr_k0_r2_s1, 0, 0, vmall_r2 , vl) ;
      __vr vrgout_ptr_k1_r2_s1 = _vel_vaddsl_vsvl(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_r2_s1, vl) ;
      __vr vrgout_k1_r2_s1 = _vel_vgtu_vvssml(vrgout_ptr_k1_r2_s1, 0, 0, vmall_r2 , vl) ;
      __vr vrgout_ptr_k2_r2_s1 = _vel_vaddsl_vsvl(4*2*gOutHeight*gOutWidth, vrgout_ptr_k0_r2_s1, vl) ;
      __vr vrgout_k2_r2_s1 = _vel_vgtu_vvssml(vrgout_ptr_k2_r2_s1, 0, 0, vmall_r2 , vl) ;
      __vr vrgout_ptr_k3_r2_s1 = _vel_vaddsl_vsvl(4*3*gOutHeight*gOutWidth, vrgout_ptr_k0_r2_s1, vl) ;
      __vr vrgout_k3_r2_s1 = _vel_vgtu_vvssml(vrgout_ptr_k3_r2_s1, 0, 0, vmall_r2 , vl) ;
      __vr vrgout_ptr_k4_r2_s1 = _vel_vaddsl_vsvl(4*4*gOutHeight*gOutWidth, vrgout_ptr_k0_r2_s1, vl) ;
      __vr vrgout_k4_r2_s1 = _vel_vgtu_vvssml(vrgout_ptr_k4_r2_s1, 0, 0, vmall_r2 , vl) ;
      __vr vrgout_ptr_k5_r2_s1 = _vel_vaddsl_vsvl(4*5*gOutHeight*gOutWidth, vrgout_ptr_k0_r2_s1, vl) ;
      __vr vrgout_k5_r2_s1 = _vel_vgtu_vvssml(vrgout_ptr_k5_r2_s1, 0, 0, vmall_r2 , vl) ;
      __vr vrgout_ptr_k6_r2_s1 = _vel_vaddsl_vsvl(4*6*gOutHeight*gOutWidth, vrgout_ptr_k0_r2_s1, vl) ;
      __vr vrgout_k6_r2_s1 = _vel_vgtu_vvssml(vrgout_ptr_k6_r2_s1, 0, 0, vmall_r2 , vl) ;
      __vr vrgout_ptr_k7_r2_s1 = _vel_vaddsl_vsvl(4*7*gOutHeight*gOutWidth, vrgout_ptr_k0_r2_s1, vl) ;
      __vr vrgout_k7_r2_s1 = _vel_vgtu_vvssml(vrgout_ptr_k7_r2_s1, 0, 0, vmall_r2 , vl) ;

      VFADD(vrgout_k0_r2_s1, vmall_r2s2, k+0, 2, 2, s12) ;
      VFADD(vrgout_k1_r2_s1, vmall_r2s2, k+1, 2, 2, s12) ;
      VFADD(vrgout_k2_r2_s1, vmall_r2s2, k+2, 2, 2, s12) ;
      VFADD(vrgout_k3_r2_s1, vmall_r2s2, k+3, 2, 2, s12) ;
      VFADD(vrgout_k4_r2_s1, vmall_r2s2, k+4, 2, 2, s12) ;
      VFADD(vrgout_k5_r2_s1, vmall_r2s2, k+5, 2, 2, s12) ;
      VFADD(vrgout_k6_r2_s1, vmall_r2s2, k+6, 2, 2, s12) ;
      VFADD(vrgout_k7_r2_s1, vmall_r2s2, k+7, 2, 2, s12) ;

      VFADD(vrgout_k0_r2_s1, vmall_r2s1, k+0, 2, 1, s12) ;
      VFADD(vrgout_k1_r2_s1, vmall_r2s1, k+1, 2, 1, s12) ;
      VFADD(vrgout_k2_r2_s1, vmall_r2s1, k+2, 2, 1, s12) ;
      VFADD(vrgout_k3_r2_s1, vmall_r2s1, k+3, 2, 1, s12) ;
      VFADD(vrgout_k4_r2_s1, vmall_r2s1, k+4, 2, 1, s12) ;
      VFADD(vrgout_k5_r2_s1, vmall_r2s1, k+5, 2, 1, s12) ;
      VFADD(vrgout_k6_r2_s1, vmall_r2s1, k+6, 2, 1, s12) ;
      VFADD(vrgout_k7_r2_s1, vmall_r2s1, k+7, 2, 1, s12) ;

      VFADD(vrgout_k0_r2_s1, vmall_r2s1, k+0, 2, 0, s0) ;
      VFADD(vrgout_k1_r2_s1, vmall_r2s1, k+1, 2, 0, s0) ;
      VFADD(vrgout_k2_r2_s1, vmall_r2s1, k+2, 2, 0, s0) ;
      VFADD(vrgout_k3_r2_s1, vmall_r2s1, k+3, 2, 0, s0) ;
      VFADD(vrgout_k4_r2_s1, vmall_r2s1, k+4, 2, 0, s0) ;
      VFADD(vrgout_k5_r2_s1, vmall_r2s1, k+5, 2, 0, s0) ;
      VFADD(vrgout_k6_r2_s1, vmall_r2s1, k+6, 2, 0, s0) ;
      VFADD(vrgout_k7_r2_s1, vmall_r2s1, k+7, 2, 0, s0) ;

    } // gOutChannel
#else
    for (; k<gOutChannelGroup; ) {
      int64_t gOutIndex    = gOutGroupOffset + ((n * gOutChannel + k) * gOutHeight) * gOutWidth ;

      __vr vrgout_ptr_k0_r0_s1 = _vel_vsfa_vvssl(_vel_vaddsl_vvvl(_vel_vmulsl_vsvl(gOutWidth, vry_r0, vl), vrx_s1, vl),
					 2,
					 (unsigned long)(pGOut+gOutIndex), vl) ;
      __vr vrgout_k0_r0_s1 = _vel_vgtu_vvssml(vrgout_ptr_k0_r0_s1, 0, 0, vmall_r0, vl) ;
      __vr vrgout_ptr_k1_r0_s1 = _vel_vaddsl_vsvl(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_r0_s1, vl) ;
      __vr vrgout_k1_r0_s1 = _vel_vgtu_vvssml(vrgout_ptr_k1_r0_s1, 0, 0, vmall_r0, vl) ;
      __vr vrgout_ptr_k2_r0_s1 = _vel_vaddsl_vsvl(4*2*gOutHeight*gOutWidth, vrgout_ptr_k0_r0_s1, vl) ;
      __vr vrgout_k2_r0_s1 = _vel_vgtu_vvssml(vrgout_ptr_k2_r0_s1, 0, 0, vmall_r0, vl) ;
      __vr vrgout_ptr_k3_r0_s1 = _vel_vaddsl_vsvl(4*3*gOutHeight*gOutWidth, vrgout_ptr_k0_r0_s1, vl) ;
      __vr vrgout_k3_r0_s1 = _vel_vgtu_vvssml(vrgout_ptr_k3_r0_s1, 0, 0, vmall_r0, vl) ;

      VFADD(vrgout_k0_r0_s1, vmall_r0s2, k+0, 0, 2, s12) ;
      VFADD(vrgout_k1_r0_s1, vmall_r0s2, k+1, 0, 2, s12) ;
      VFADD(vrgout_k2_r0_s1, vmall_r0s2, k+2, 0, 2, s12) ;
      VFADD(vrgout_k3_r0_s1, vmall_r0s2, k+3, 0, 2, s12) ;

      VFADD(vrgout_k0_r0_s1, vmall_r0s1, k+0, 0, 1, s12) ;
      VFADD(vrgout_k1_r0_s1, vmall_r0s1, k+1, 0, 1, s12) ;
      VFADD(vrgout_k2_r0_s1, vmall_r0s1, k+2, 0, 1, s12) ;
      VFADD(vrgout_k3_r0_s1, vmall_r0s1, k+3, 0, 1, s12) ;

      VFADD(vrgout_k0_r0_s1, vmall_r0s1, k+0, 0, 0, s0) ;
      VFADD(vrgout_k1_r0_s1, vmall_r0s1, k+1, 0, 0, s0) ;
      VFADD(vrgout_k2_r0_s1, vmall_r0s1, k+2, 0, 0, s0) ;
      VFADD(vrgout_k3_r0_s1, vmall_r0s1, k+3, 0, 0, s0) ;

      __vr vrgout_ptr_k0_r1_s1 = _vel_vsfa_vvssl(_vel_vaddsl_vvvl(_vel_vmulsl_vsvl(gOutWidth, vry_r1, vl), vrx_s1, vl),
					 2,
					 (unsigned long)(pGOut+gOutIndex), vl) ;
      __vr vrgout_k0_r1_s1 = _vel_vgtu_vvssml(vrgout_ptr_k0_r1_s1, 0, 0, vmall_r1, vl) ;
      __vr vrgout_ptr_k1_r1_s1 = _vel_vaddsl_vsvl(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_r1_s1, vl) ;
      __vr vrgout_k1_r1_s1 = _vel_vgtu_vvssml(vrgout_ptr_k1_r1_s1, 0, 0, vmall_r1, vl) ;
      __vr vrgout_ptr_k2_r1_s1 = _vel_vaddsl_vsvl(4*2*gOutHeight*gOutWidth, vrgout_ptr_k0_r1_s1, vl) ;
      __vr vrgout_k2_r1_s1 = _vel_vgtu_vvssml(vrgout_ptr_k2_r1_s1, 0, 0, vmall_r1, vl) ;
      __vr vrgout_ptr_k3_r1_s1 = _vel_vaddsl_vsvl(4*3*gOutHeight*gOutWidth, vrgout_ptr_k0_r1_s1, vl) ;
      __vr vrgout_k3_r1_s1 = _vel_vgtu_vvssml(vrgout_ptr_k3_r1_s1, 0, 0, vmall_r1, vl) ;

      VFADD(vrgout_k0_r1_s1, vmall_r1s2, k+0, 1, 2, s12) ;
      VFADD(vrgout_k1_r1_s1, vmall_r1s2, k+1, 1, 2, s12) ;
      VFADD(vrgout_k2_r1_s1, vmall_r1s2, k+2, 1, 2, s12) ;
      VFADD(vrgout_k3_r1_s1, vmall_r1s2, k+3, 1, 2, s12) ;

      VFADD(vrgout_k0_r1_s1, vmall_r1s1, k+0, 1, 1, s12) ;
      VFADD(vrgout_k1_r1_s1, vmall_r1s1, k+1, 1, 1, s12) ;
      VFADD(vrgout_k2_r1_s1, vmall_r1s1, k+2, 1, 1, s12) ;
      VFADD(vrgout_k3_r1_s1, vmall_r1s1, k+3, 1, 1, s12) ;

      VFADD(vrgout_k0_r1_s1, vmall_r1s1, k+0, 1, 0, s0) ;
      VFADD(vrgout_k1_r1_s1, vmall_r1s1, k+1, 1, 0, s0) ;
      VFADD(vrgout_k2_r1_s1, vmall_r1s1, k+2, 1, 0, s0) ;
      VFADD(vrgout_k3_r1_s1, vmall_r1s1, k+3, 1, 0, s0) ;


      __vr vrgout_ptr_k0_r2_s1 = _vel_vsfa_vvssl(_vel_vaddsl_vvvl(_vel_vmulsl_vsvl(gOutWidth, vry_r2, vl), vrx_s1, vl),
					 2,
					 (unsigned long)(pGOut+gOutIndex), vl) ;
      __vr vrgout_k0_r2_s1 = _vel_vgtu_vvssml(vrgout_ptr_k0_r2_s1, 0, 0, vmall_r2 , vl) ;
      __vr vrgout_ptr_k1_r2_s1 = _vel_vaddsl_vsvl(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_r2_s1, vl) ;
      __vr vrgout_k1_r2_s1 = _vel_vgtu_vvssml(vrgout_ptr_k1_r2_s1, 0, 0, vmall_r2 , vl) ;
      __vr vrgout_ptr_k2_r2_s1 = _vel_vaddsl_vsvl(4*2*gOutHeight*gOutWidth, vrgout_ptr_k0_r2_s1, vl) ;
      __vr vrgout_k2_r2_s1 = _vel_vgtu_vvssml(vrgout_ptr_k2_r2_s1, 0, 0, vmall_r2 , vl) ;
      __vr vrgout_ptr_k3_r2_s1 = _vel_vaddsl_vsvl(4*3*gOutHeight*gOutWidth, vrgout_ptr_k0_r2_s1, vl) ;
      __vr vrgout_k3_r2_s1 = _vel_vgtu_vvssml(vrgout_ptr_k3_r2_s1, 0, 0, vmall_r2 , vl) ;

      VFADD(vrgout_k0_r2_s1, vmall_r2s2, k+0, 2, 2, s12) ;
      VFADD(vrgout_k1_r2_s1, vmall_r2s2, k+1, 2, 2, s12) ;
      VFADD(vrgout_k2_r2_s1, vmall_r2s2, k+2, 2, 2, s12) ;
      VFADD(vrgout_k3_r2_s1, vmall_r2s2, k+3, 2, 2, s12) ;

      VFADD(vrgout_k0_r2_s1, vmall_r2s1, k+0, 2, 1, s12) ;
      VFADD(vrgout_k1_r2_s1, vmall_r2s1, k+1, 2, 1, s12) ;
      VFADD(vrgout_k2_r2_s1, vmall_r2s1, k+2, 2, 1, s12) ;
      VFADD(vrgout_k3_r2_s1, vmall_r2s1, k+3, 2, 1, s12) ;

      VFADD(vrgout_k0_r2_s1, vmall_r2s1, k+0, 2, 0, s0) ;
      VFADD(vrgout_k1_r2_s1, vmall_r2s1, k+1, 2, 0, s0) ;
      VFADD(vrgout_k2_r2_s1, vmall_r2s1, k+2, 2, 0, s0) ;
      VFADD(vrgout_k3_r2_s1, vmall_r2s1, k+3, 2, 0, s0) ;

      k+=4 ;
    }
#endif
#undef VFADD
#undef FILTER_OFFSET


    {
      __vr vrsum0 = _vel_vfadds_vvvl(_vel_vmrg_vsvml(0.f, _vel_vmv_vsvl(1, vrsum0_s0, vl), vmx_s0, vl), vrsum0_s12, vl) ;
      _vel_vstu_vssl(vrsum0, 4, pGIn+gInIndex + 0 * gInHeight * gInWidth, vl) ;
    }
    if(NUMCHANNEL>= 3) {
      __vr vrsum12 = _vel_pvfadd_vvvl(_vel_vmrg_vsvml(0UL, _vel_vmv_vsvl(1, vrsum12_s0, vl), vmx_s0, vl), vrsum12_s12, vl) ;
      _vel_vstu_vssl(vrsum12, 4, pGIn+gInIndex + 1 * gInHeight * gInWidth, vl) ;
      _vel_vstl_vssl(vrsum12, 4, pGIn+gInIndex + 2 * gInHeight * gInWidth, vl) ;
    }
    if(NUMCHANNEL>= 5) {
      __vr vrsum34 = _vel_pvfadd_vvvl(_vel_vmrg_vsvml(0UL, _vel_vmv_vsvl(1, vrsum34_s0, vl), vmx_s0, vl), vrsum34_s12, vl) ;
      _vel_vstu_vssl(vrsum34, 4, pGIn+gInIndex + 3 * gInHeight * gInWidth, vl) ;
      _vel_vstl_vssl(vrsum34, 4, pGIn+gInIndex + 4 * gInHeight * gInWidth, vl) ;
    }
    if(NUMCHANNEL>= 7) {
      __vr vrsum56 = _vel_pvfadd_vvvl(_vel_vmrg_vsvml(0UL, _vel_vmv_vsvl(1, vrsum56_s0, vl), vmx_s0, vl), vrsum56_s12, vl) ;
      _vel_vstu_vssl(vrsum56, 4, pGIn+gInIndex + 5 * gInHeight * gInWidth, vl) ;
      _vel_vstl_vssl(vrsum56, 4, pGIn+gInIndex + 6 * gInHeight * gInWidth, vl) ;
    }
    if(NUMCHANNEL>= 9) {
      __vr vrsum78 = _vel_pvfadd_vvvl(_vel_vmrg_vsvml(0UL, _vel_vmv_vsvl(1, vrsum78_s0, vl), vmx_s0, vl), vrsum78_s12, vl) ;
      _vel_vstu_vssl(vrsum78, 4, pGIn+gInIndex + 7 * gInHeight * gInWidth, vl) ;
      _vel_vstl_vssl(vrsum78, 4, pGIn+gInIndex + 8 * gInHeight * gInWidth, vl) ;
    }
    if(NUMCHANNEL>=11) {
      __vr vrsum9A = _vel_pvfadd_vvvl(_vel_vmrg_vsvml(0UL, _vel_vmv_vsvl(1, vrsum9A_s0, vl), vmx_s0, vl), vrsum9A_s12, vl) ;
      _vel_vstu_vssl(vrsum9A, 4, pGIn+gInIndex + 9 * gInHeight * gInWidth, vl) ;
      _vel_vstl_vssl(vrsum9A, 4, pGIn+gInIndex +10 * gInHeight * gInWidth, vl) ;
    }
    if(NUMCHANNEL>=13) {
      __vr vrsumBC = _vel_pvfadd_vvvl(_vel_vmrg_vsvml(0UL, _vel_vmv_vsvl(1, vrsumBC_s0, vl), vmx_s0, vl), vrsumBC_s12, vl) ;
      _vel_vstu_vssl(vrsumBC, 4, pGIn+gInIndex +11 * gInHeight * gInWidth, vl) ;
      _vel_vstl_vssl(vrsumBC, 4, pGIn+gInIndex +12 * gInHeight * gInWidth, vl) ;
    }
    if(NUMCHANNEL>=15) {
      __vr vrsumDE = _vel_pvfadd_vvvl(_vel_vmrg_vsvml(0UL, _vel_vmv_vsvl(1, vrsumDE_s0, vl), vmx_s0, vl), vrsumDE_s12, vl) ;
      _vel_vstu_vssl(vrsumDE, 4, pGIn+gInIndex +13 * gInHeight * gInWidth, vl) ;
      _vel_vstl_vssl(vrsumDE, 4, pGIn+gInIndex +14 * gInHeight * gInWidth, vl) ;
    }

    gInIndex += vl ;
  } // gOutPixels
}


template<filterLayout_t FLAYOUT, int NUMCHANNEL>
static __attribute__((noinline)) void func_even(
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
    const int64_t c,
    const int64_t nH
)
{
  int64_t gInIndex = gInGroupOffset + ((n * gInChannel + c) * gInHeight ) * gInWidth  ;

  __vr vrseq = _vel_vseq_vl(nH*gInWidth) ;
  __vr vrh  = _vel_vdivsl_vvsl(vrseq, gInWidth, nH*gInWidth) ;
  __vr vrw  = _vel_vsubsl_vvvl(vrseq, _vel_vmulul_vsvl(gInWidth,vrh, nH*gInWidth), nH*gInWidth) ;

  for (int64_t h=0; h<gInHeight; h+=nH) {
    const int64_t vl = gInWidth * (gInHeight - h < nH ? gInHeight - h : nH) ;
    const int64_t gip = h * gInWidth ;

    __vr vrsum01_s12 = _vel_pvbrd_vsl(0UL, vl) ;
    __vr vrsum23_s12 = _vel_pvbrd_vsl(0UL, vl) ;
    __vr vrsum45_s12 = _vel_pvbrd_vsl(0UL, vl) ;
    __vr vrsum67_s12 = _vel_pvbrd_vsl(0UL, vl) ;
    __vr vrsum89_s12 = _vel_pvbrd_vsl(0UL, vl) ;
    __vr vrsumAB_s12 = _vel_pvbrd_vsl(0UL, vl) ;
    __vr vrsumCD_s12 = _vel_pvbrd_vsl(0UL, vl) ;
    __vr vrsumEF_s12 = _vel_pvbrd_vsl(0UL, vl) ;

    __vr vrsum01_s0 = _vel_pvbrd_vsl(0UL, vl) ;
    __vr vrsum23_s0 = _vel_pvbrd_vsl(0UL, vl) ;
    __vr vrsum45_s0 = _vel_pvbrd_vsl(0UL, vl) ;
    __vr vrsum67_s0 = _vel_pvbrd_vsl(0UL, vl) ;
    __vr vrsum89_s0 = _vel_pvbrd_vsl(0UL, vl) ;
    __vr vrsumAB_s0 = _vel_pvbrd_vsl(0UL, vl) ;
    __vr vrsumCD_s0 = _vel_pvbrd_vsl(0UL, vl) ;
    __vr vrsumEF_s0 = _vel_pvbrd_vsl(0UL, vl) ;

    __vr vri_r0 = _vel_vaddsl_vsvl(padHeight-0*dilationHeight+h, vrh, vl) ;
    __vr vri_r1 = _vel_vaddsl_vsvl(padHeight-1*dilationHeight+h, vrh, vl) ;
    __vr vri_r2 = _vel_vaddsl_vsvl(padHeight-2*dilationHeight+h, vrh, vl) ;

    __vr vry_r0 = _vel_vdivsl_vvsl(vri_r0, strideHeight, vl) ;
    __vr vry_r1 = _vel_vdivsl_vvsl(vri_r1, strideHeight, vl) ;
    __vr vry_r2 = _vel_vdivsl_vvsl(vri_r2, strideHeight, vl) ;

    __vr vrj_s0 = _vel_vaddsl_vsvl(padWidth-0*dilationWidth, vrw, vl) ;
    __vr vrj_s1 = _vel_vaddsl_vsvl(padWidth-1*dilationWidth, vrw, vl) ;
    __vr vrj_s2 = _vel_vaddsl_vsvl(padWidth-2*dilationWidth, vrw, vl) ;

    __vr vrx_s0 = _vel_vdivsl_vvsl(vrj_s0, strideWidth, vl) ;
    __vr vrx_s1 = _vel_vdivsl_vvsl(vrj_s1, strideWidth, vl) ;
    __vr vrx_s2 = _vel_vdivsl_vvsl(vrj_s2, strideWidth, vl) ;


    __vm256 vmy0_r0 =  _vel_vfmkleq_mvl(_vel_vcmpsl_vvvl(vri_r0, _vel_vmulsl_vsvl(strideHeight, vry_r0, vl), vl), vl) ;
    __vm256 vmy1_r0 =  _vel_vfmklge_mvl(vry_r0, vl) ;
    __vm256 vmy2_r0 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutHeight,vry_r0, vl), vl) ;
    __vm256 vmy_r0 = _vel_andm_mmm(_vel_andm_mmm(vmy0_r0, vmy1_r0), vmy2_r0) ;

    __vm256 vmy0_r1 =  _vel_vfmkleq_mvl(_vel_vcmpsl_vvvl(vri_r1, _vel_vmulsl_vsvl(strideHeight, vry_r1, vl), vl), vl) ;
    __vm256 vmy1_r1 =  _vel_vfmklge_mvl(vry_r1, vl) ;
    __vm256 vmy2_r1 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutHeight,vry_r1, vl), vl) ;
    __vm256 vmy_r1 = _vel_andm_mmm(_vel_andm_mmm(vmy0_r1, vmy1_r1), vmy2_r1) ;

    __vm256 vmy0_r2 =  _vel_vfmkleq_mvl(_vel_vcmpsl_vvvl(vri_r2, _vel_vmulsl_vsvl(strideHeight, vry_r2, vl), vl), vl) ;
    __vm256 vmy1_r2 =  _vel_vfmklge_mvl(vry_r2, vl) ;
    __vm256 vmy2_r2 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutHeight,vry_r2, vl), vl) ;
    __vm256 vmy_r2 = _vel_andm_mmm(_vel_andm_mmm(vmy0_r2, vmy1_r2), vmy2_r2) ;

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

    __vm256 vmall_r0s0 = _vel_andm_mmm(vmy_r0,vmx_s0) ;
    __vm256 vmall_r0s1 = _vel_andm_mmm(vmy_r0,vmx_s1) ;
    __vm256 vmall_r0s2 = _vel_andm_mmm(vmy_r0,vmx_s2) ;
    __vm256 vmall_r0   = _vel_orm_mmm(vmall_r0s1, vmall_r0s2) ;

    __vm256 vmall_r1s0 = _vel_andm_mmm(vmy_r1,vmx_s0) ;
    __vm256 vmall_r1s1 = _vel_andm_mmm(vmy_r1,vmx_s1) ;
    __vm256 vmall_r1s2 = _vel_andm_mmm(vmy_r1,vmx_s2) ;
    __vm256 vmall_r1   = _vel_orm_mmm(vmall_r1s1, vmall_r1s2) ;

    __vm256 vmall_r2s0 = _vel_andm_mmm(vmy_r2,vmx_s0) ;
    __vm256 vmall_r2s1 = _vel_andm_mmm(vmy_r2,vmx_s1) ;
    __vm256 vmall_r2s2 = _vel_andm_mmm(vmy_r2,vmx_s2) ;
    __vm256 vmall_r2   = _vel_orm_mmm(vmall_r2s1, vmall_r2s2) ;

    int64_t k=0;
    if( (gOutChannelGroup & 0x01 ) == 1 ) {
      int64_t gOutIndex    = gOutGroupOffset + ((n * gOutChannel + k) * gOutHeight) * gOutWidth ;

#define FILTER_OFFSET(k,c,r,s) ( kernGroupOffset + filter_index<FLAYOUT>(k,c,r,s, gInChannelGroup, gOutChannelGroup, kernHeight, kernWidth) )
#define FILTER_DISTANCE_BY_C()   ( FLAYOUT == VEDNN_FILTER_LAYOUT_NCHW ? kernHeight * kernWidth : gOutChannelGroup ) ;
#define VFADD(VRGOUT, VM, K, R, S, STOKEN)	{							\
	const int64_t filter_offset   = FILTER_OFFSET(K,c+ 0,R,S) ;					\
	const int64_t filter_distance = FILTER_DISTANCE_BY_C() ;					\
	const uint64_t kerValue01 = _vel_pack_f32p(pKernel + filter_offset + 0 * filter_distance,	\
						   pKernel + filter_offset + 1 * filter_distance) ;	\
	const uint64_t kerValue23 = _vel_pack_f32p(pKernel + filter_offset + 2 * filter_distance,	\
						   pKernel + filter_offset + 3 * filter_distance) ;	\
	const uint64_t kerValue45 = _vel_pack_f32p(pKernel + filter_offset + 4 * filter_distance,	\
						   pKernel + filter_offset + 5 * filter_distance) ;	\
	const uint64_t kerValue67 = _vel_pack_f32p(pKernel + filter_offset + 6 * filter_distance,	\
						   pKernel + filter_offset + 7 * filter_distance) ;	\
	const uint64_t kerValue89 = _vel_pack_f32p(pKernel + filter_offset + 8 * filter_distance,	\
						   pKernel + filter_offset + 9 * filter_distance) ;	\
	const uint64_t kerValueAB = _vel_pack_f32p(pKernel + filter_offset +10 * filter_distance,	\
						   pKernel + filter_offset +11 * filter_distance) ;	\
	const uint64_t kerValueCD = _vel_pack_f32p(pKernel + filter_offset +12 * filter_distance,	\
						   pKernel + filter_offset +13 * filter_distance) ;	\
	const uint64_t kerValueEF = _vel_pack_f32p(pKernel + filter_offset +14 * filter_distance,	\
						   pKernel + filter_offset +15 * filter_distance) ;	\
	__vr vrgout  = _vel_vmrg_vsvml(0.f, VRGOUT, VM, vl) ;					\
	__vr vrgoutP = _vel_vshf_vvvsl(vrgout, vrgout, VE_VSHUFFLE_YUZU, vl) ;			\
	if(NUMCHANNEL>= 2) vrsum01_##STOKEN = _vel_pvfmad_vvsvl(vrsum01_##STOKEN, kerValue01, vrgoutP, vl) ;	\
	if(NUMCHANNEL>= 4) vrsum23_##STOKEN = _vel_pvfmad_vvsvl(vrsum23_##STOKEN, kerValue23, vrgoutP, vl) ;	\
	if(NUMCHANNEL>= 6) vrsum45_##STOKEN = _vel_pvfmad_vvsvl(vrsum45_##STOKEN, kerValue45, vrgoutP, vl) ;	\
	if(NUMCHANNEL>= 8) vrsum67_##STOKEN = _vel_pvfmad_vvsvl(vrsum67_##STOKEN, kerValue67, vrgoutP, vl) ;	\
	if(NUMCHANNEL>=10) vrsum89_##STOKEN = _vel_pvfmad_vvsvl(vrsum89_##STOKEN, kerValue89, vrgoutP, vl) ;	\
	if(NUMCHANNEL>=12) vrsumAB_##STOKEN = _vel_pvfmad_vvsvl(vrsumAB_##STOKEN, kerValueAB, vrgoutP, vl) ;	\
	if(NUMCHANNEL>=14) vrsumCD_##STOKEN = _vel_pvfmad_vvsvl(vrsumCD_##STOKEN, kerValueCD, vrgoutP, vl) ;	\
	if(NUMCHANNEL>=16) vrsumEF_##STOKEN = _vel_pvfmad_vvsvl(vrsumEF_##STOKEN, kerValueEF, vrgoutP, vl) ;	\
      }

      __vr vrgout_ptr_k0_r0_s1 = _vel_vsfa_vvssl(_vel_vaddsl_vvvl(_vel_vmulsl_vsvl(gOutWidth, vry_r0, vl), vrx_s1, vl),
					 2,
					 (unsigned long)(pGOut+gOutIndex), vl) ;
      __vr vrgout_k0_r0_s1 = _vel_vgtu_vvssml(vrgout_ptr_k0_r0_s1, 0, 0, vmall_r0, vl) ;

      VFADD(vrgout_k0_r0_s1, vmall_r0s2, k+0, 0, 2, s12) ;

      VFADD(vrgout_k0_r0_s1, vmall_r0s1, k+0, 0, 1, s12) ;

      VFADD(vrgout_k0_r0_s1, vmall_r0s1, k+0, 0, 0, s0) ;

      __vr vrgout_ptr_k0_r1_s1 = _vel_vsfa_vvssl(_vel_vaddsl_vvvl(_vel_vmulsl_vsvl(gOutWidth, vry_r1, vl), vrx_s1, vl),
					 2,
					 (unsigned long)(pGOut+gOutIndex), vl) ;
      __vr vrgout_k0_r1_s1 = _vel_vgtu_vvssml(vrgout_ptr_k0_r1_s1, 0, 0, vmall_r1, vl) ;

      VFADD(vrgout_k0_r1_s1, vmall_r1s2, k+0, 1, 2, s12) ;

      VFADD(vrgout_k0_r1_s1, vmall_r1s1, k+0, 1, 1, s12) ;

      VFADD(vrgout_k0_r1_s1, vmall_r1s1, k+0, 1, 0, s0) ;

      __vr vrgout_ptr_k0_r2_s1 = _vel_vsfa_vvssl(_vel_vaddsl_vvvl(_vel_vmulsl_vsvl(gOutWidth, vry_r2, vl), vrx_s1, vl),
					 2,
					 (unsigned long)(pGOut+gOutIndex), vl) ;
      __vr vrgout_k0_r2_s1 = _vel_vgtu_vvssml(vrgout_ptr_k0_r2_s1, 0, 0, vmall_r2 , vl) ;

      VFADD(vrgout_k0_r2_s1, vmall_r2s2, k+0, 2, 2, s12) ;

      VFADD(vrgout_k0_r2_s1, vmall_r2s1, k+0, 2, 1, s12) ;

      VFADD(vrgout_k0_r2_s1, vmall_r2s1, k+0, 2, 0, s0) ;

      k+=1 ;
    }
    if( ((gOutChannelGroup >> 1) & 0x01 ) == 1 ) {
      int64_t gOutIndex    = gOutGroupOffset + ((n * gOutChannel + k) * gOutHeight) * gOutWidth ;

      __vr vrgout_ptr_k0_r0_s1 = _vel_vsfa_vvssl(_vel_vaddsl_vvvl(_vel_vmulsl_vsvl(gOutWidth, vry_r0, vl), vrx_s1, vl),
					 2,
					 (unsigned long)(pGOut+gOutIndex), vl) ;
      __vr vrgout_k0_r0_s1 = _vel_vgtu_vvssml(vrgout_ptr_k0_r0_s1, 0, 0, vmall_r0, vl) ;
      __vr vrgout_ptr_k1_r0_s1 = _vel_vaddsl_vsvl(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_r0_s1, vl) ;
      __vr vrgout_k1_r0_s1 = _vel_vgtu_vvssml(vrgout_ptr_k1_r0_s1, 0, 0, vmall_r0, vl) ;

      VFADD(vrgout_k0_r0_s1, vmall_r0s2, k+0, 0, 2, s12) ;
      VFADD(vrgout_k1_r0_s1, vmall_r0s2, k+1, 0, 2, s12) ;

      VFADD(vrgout_k0_r0_s1, vmall_r0s1, k+0, 0, 1, s12) ;
      VFADD(vrgout_k1_r0_s1, vmall_r0s1, k+1, 0, 1, s12) ;

      VFADD(vrgout_k0_r0_s1, vmall_r0s1, k+0, 0, 0, s0) ;
      VFADD(vrgout_k1_r0_s1, vmall_r0s1, k+1, 0, 0, s0) ;

      __vr vrgout_ptr_k0_r1_s1 = _vel_vsfa_vvssl(_vel_vaddsl_vvvl(_vel_vmulsl_vsvl(gOutWidth, vry_r1, vl), vrx_s1, vl),
					 2,
					 (unsigned long)(pGOut+gOutIndex), vl) ;
      __vr vrgout_k0_r1_s1 = _vel_vgtu_vvssml(vrgout_ptr_k0_r1_s1, 0, 0, vmall_r1, vl) ;
      __vr vrgout_ptr_k1_r1_s1 = _vel_vaddsl_vsvl(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_r1_s1, vl) ;
      __vr vrgout_k1_r1_s1 = _vel_vgtu_vvssml(vrgout_ptr_k1_r1_s1, 0, 0, vmall_r1, vl) ;

      VFADD(vrgout_k0_r1_s1, vmall_r1s2, k+0, 1, 2, s12) ;
      VFADD(vrgout_k1_r1_s1, vmall_r1s2, k+1, 1, 2, s12) ;

      VFADD(vrgout_k0_r1_s1, vmall_r1s1, k+0, 1, 1, s12) ;
      VFADD(vrgout_k1_r1_s1, vmall_r1s1, k+1, 1, 1, s12) ;

      VFADD(vrgout_k0_r1_s1, vmall_r1s1, k+0, 1, 0, s0) ;
      VFADD(vrgout_k1_r1_s1, vmall_r1s1, k+1, 1, 0, s0) ;

      __vr vrgout_ptr_k0_r2_s1 = _vel_vsfa_vvssl(_vel_vaddsl_vvvl(_vel_vmulsl_vsvl(gOutWidth, vry_r2, vl), vrx_s1, vl),
					 2,
					 (unsigned long)(pGOut+gOutIndex), vl) ;
      __vr vrgout_k0_r2_s1 = _vel_vgtu_vvssml(vrgout_ptr_k0_r2_s1, 0, 0, vmall_r2 , vl) ;
      __vr vrgout_ptr_k1_r2_s1 = _vel_vaddsl_vsvl(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_r2_s1, vl) ;
      __vr vrgout_k1_r2_s1 = _vel_vgtu_vvssml(vrgout_ptr_k1_r2_s1, 0, 0, vmall_r2 , vl) ;

      VFADD(vrgout_k0_r2_s1, vmall_r2s2, k+0, 2, 2, s12) ;
      VFADD(vrgout_k1_r2_s1, vmall_r2s2, k+1, 2, 2, s12) ;

      VFADD(vrgout_k0_r2_s1, vmall_r2s1, k+0, 2, 1, s12) ;
      VFADD(vrgout_k1_r2_s1, vmall_r2s1, k+1, 2, 1, s12) ;

      VFADD(vrgout_k0_r2_s1, vmall_r2s1, k+0, 2, 0, s0) ;
      VFADD(vrgout_k1_r2_s1, vmall_r2s1, k+1, 2, 0, s0) ;

      k+=2 ;
    }
    if( ((gOutChannelGroup >> 2) & 0x01 ) == 1 ) {
      int64_t gOutIndex    = gOutGroupOffset + ((n * gOutChannel + k) * gOutHeight) * gOutWidth ;

      __vr vrgout_ptr_k0_r0_s1 = _vel_vsfa_vvssl(_vel_vaddsl_vvvl(_vel_vmulsl_vsvl(gOutWidth, vry_r0, vl), vrx_s1, vl),
					 2,
					 (unsigned long)(pGOut+gOutIndex), vl) ;
      __vr vrgout_k0_r0_s1 = _vel_vgtu_vvssml(vrgout_ptr_k0_r0_s1, 0, 0, vmall_r0, vl) ;
      __vr vrgout_ptr_k1_r0_s1 = _vel_vaddsl_vsvl(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_r0_s1, vl) ;
      __vr vrgout_k1_r0_s1 = _vel_vgtu_vvssml(vrgout_ptr_k1_r0_s1, 0, 0, vmall_r0, vl) ;
      __vr vrgout_ptr_k2_r0_s1 = _vel_vaddsl_vsvl(4*2*gOutHeight*gOutWidth, vrgout_ptr_k0_r0_s1, vl) ;
      __vr vrgout_k2_r0_s1 = _vel_vgtu_vvssml(vrgout_ptr_k2_r0_s1, 0, 0, vmall_r0, vl) ;
      __vr vrgout_ptr_k3_r0_s1 = _vel_vaddsl_vsvl(4*3*gOutHeight*gOutWidth, vrgout_ptr_k0_r0_s1, vl) ;
      __vr vrgout_k3_r0_s1 = _vel_vgtu_vvssml(vrgout_ptr_k3_r0_s1, 0, 0, vmall_r0, vl) ;

      VFADD(vrgout_k0_r0_s1, vmall_r0s2, k+0, 0, 2, s12) ;
      VFADD(vrgout_k1_r0_s1, vmall_r0s2, k+1, 0, 2, s12) ;
      VFADD(vrgout_k2_r0_s1, vmall_r0s2, k+2, 0, 2, s12) ;
      VFADD(vrgout_k3_r0_s1, vmall_r0s2, k+3, 0, 2, s12) ;

      VFADD(vrgout_k0_r0_s1, vmall_r0s1, k+0, 0, 1, s12) ;
      VFADD(vrgout_k1_r0_s1, vmall_r0s1, k+1, 0, 1, s12) ;
      VFADD(vrgout_k2_r0_s1, vmall_r0s1, k+2, 0, 1, s12) ;
      VFADD(vrgout_k3_r0_s1, vmall_r0s1, k+3, 0, 1, s12) ;

      VFADD(vrgout_k0_r0_s1, vmall_r0s1, k+0, 0, 0, s0) ;
      VFADD(vrgout_k1_r0_s1, vmall_r0s1, k+1, 0, 0, s0) ;
      VFADD(vrgout_k2_r0_s1, vmall_r0s1, k+2, 0, 0, s0) ;
      VFADD(vrgout_k3_r0_s1, vmall_r0s1, k+3, 0, 0, s0) ;

      __vr vrgout_ptr_k0_r1_s1 = _vel_vsfa_vvssl(_vel_vaddsl_vvvl(_vel_vmulsl_vsvl(gOutWidth, vry_r1, vl), vrx_s1, vl),
					 2,
					 (unsigned long)(pGOut+gOutIndex), vl) ;
      __vr vrgout_k0_r1_s1 = _vel_vgtu_vvssml(vrgout_ptr_k0_r1_s1, 0, 0, vmall_r1, vl) ;
      __vr vrgout_ptr_k1_r1_s1 = _vel_vaddsl_vsvl(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_r1_s1, vl) ;
      __vr vrgout_k1_r1_s1 = _vel_vgtu_vvssml(vrgout_ptr_k1_r1_s1, 0, 0, vmall_r1, vl) ;
      __vr vrgout_ptr_k2_r1_s1 = _vel_vaddsl_vsvl(4*2*gOutHeight*gOutWidth, vrgout_ptr_k0_r1_s1, vl) ;
      __vr vrgout_k2_r1_s1 = _vel_vgtu_vvssml(vrgout_ptr_k2_r1_s1, 0, 0, vmall_r1, vl) ;
      __vr vrgout_ptr_k3_r1_s1 = _vel_vaddsl_vsvl(4*3*gOutHeight*gOutWidth, vrgout_ptr_k0_r1_s1, vl) ;
      __vr vrgout_k3_r1_s1 = _vel_vgtu_vvssml(vrgout_ptr_k3_r1_s1, 0, 0, vmall_r1, vl) ;

      VFADD(vrgout_k0_r1_s1, vmall_r1s2, k+0, 1, 2, s12) ;
      VFADD(vrgout_k1_r1_s1, vmall_r1s2, k+1, 1, 2, s12) ;
      VFADD(vrgout_k2_r1_s1, vmall_r1s2, k+2, 1, 2, s12) ;
      VFADD(vrgout_k3_r1_s1, vmall_r1s2, k+3, 1, 2, s12) ;

      VFADD(vrgout_k0_r1_s1, vmall_r1s1, k+0, 1, 1, s12) ;
      VFADD(vrgout_k1_r1_s1, vmall_r1s1, k+1, 1, 1, s12) ;
      VFADD(vrgout_k2_r1_s1, vmall_r1s1, k+2, 1, 1, s12) ;
      VFADD(vrgout_k3_r1_s1, vmall_r1s1, k+3, 1, 1, s12) ;

      VFADD(vrgout_k0_r1_s1, vmall_r1s1, k+0, 1, 0, s0) ;
      VFADD(vrgout_k1_r1_s1, vmall_r1s1, k+1, 1, 0, s0) ;
      VFADD(vrgout_k2_r1_s1, vmall_r1s1, k+2, 1, 0, s0) ;
      VFADD(vrgout_k3_r1_s1, vmall_r1s1, k+3, 1, 0, s0) ;


      __vr vrgout_ptr_k0_r2_s1 = _vel_vsfa_vvssl(_vel_vaddsl_vvvl(_vel_vmulsl_vsvl(gOutWidth, vry_r2, vl), vrx_s1, vl),
					 2,
					 (unsigned long)(pGOut+gOutIndex), vl) ;
      __vr vrgout_k0_r2_s1 = _vel_vgtu_vvssml(vrgout_ptr_k0_r2_s1, 0, 0, vmall_r2 , vl) ;
      __vr vrgout_ptr_k1_r2_s1 = _vel_vaddsl_vsvl(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_r2_s1, vl) ;
      __vr vrgout_k1_r2_s1 = _vel_vgtu_vvssml(vrgout_ptr_k1_r2_s1, 0, 0, vmall_r2 , vl) ;
      __vr vrgout_ptr_k2_r2_s1 = _vel_vaddsl_vsvl(4*2*gOutHeight*gOutWidth, vrgout_ptr_k0_r2_s1, vl) ;
      __vr vrgout_k2_r2_s1 = _vel_vgtu_vvssml(vrgout_ptr_k2_r2_s1, 0, 0, vmall_r2 , vl) ;
      __vr vrgout_ptr_k3_r2_s1 = _vel_vaddsl_vsvl(4*3*gOutHeight*gOutWidth, vrgout_ptr_k0_r2_s1, vl) ;
      __vr vrgout_k3_r2_s1 = _vel_vgtu_vvssml(vrgout_ptr_k3_r2_s1, 0, 0, vmall_r2 , vl) ;

      VFADD(vrgout_k0_r2_s1, vmall_r2s2, k+0, 2, 2, s12) ;
      VFADD(vrgout_k1_r2_s1, vmall_r2s2, k+1, 2, 2, s12) ;
      VFADD(vrgout_k2_r2_s1, vmall_r2s2, k+2, 2, 2, s12) ;
      VFADD(vrgout_k3_r2_s1, vmall_r2s2, k+3, 2, 2, s12) ;

      VFADD(vrgout_k0_r2_s1, vmall_r2s1, k+0, 2, 1, s12) ;
      VFADD(vrgout_k1_r2_s1, vmall_r2s1, k+1, 2, 1, s12) ;
      VFADD(vrgout_k2_r2_s1, vmall_r2s1, k+2, 2, 1, s12) ;
      VFADD(vrgout_k3_r2_s1, vmall_r2s1, k+3, 2, 1, s12) ;

      VFADD(vrgout_k0_r2_s1, vmall_r2s1, k+0, 2, 0, s0) ;
      VFADD(vrgout_k1_r2_s1, vmall_r2s1, k+1, 2, 0, s0) ;
      VFADD(vrgout_k2_r2_s1, vmall_r2s1, k+2, 2, 0, s0) ;
      VFADD(vrgout_k3_r2_s1, vmall_r2s1, k+3, 2, 0, s0) ;

      k+=4 ;
    }
    for (; k<gOutChannelGroup; k+=8) {
      int64_t gOutIndex    = gOutGroupOffset + ((n * gOutChannel + k) * gOutHeight) * gOutWidth ;

      __vr vrgout_ptr_k0_r0_s1 = _vel_vsfa_vvssl(_vel_vaddsl_vvvl(_vel_vmulsl_vsvl(gOutWidth, vry_r0, vl), vrx_s1, vl),
					 2,
					 (unsigned long)(pGOut+gOutIndex), vl) ;
      __vr vrgout_k0_r0_s1 = _vel_vgtu_vvssml(vrgout_ptr_k0_r0_s1, 0, 0, vmall_r0, vl) ;
      __vr vrgout_ptr_k1_r0_s1 = _vel_vaddsl_vsvl(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_r0_s1, vl) ;
      __vr vrgout_k1_r0_s1 = _vel_vgtu_vvssml(vrgout_ptr_k1_r0_s1, 0, 0, vmall_r0, vl) ;
      __vr vrgout_ptr_k2_r0_s1 = _vel_vaddsl_vsvl(4*2*gOutHeight*gOutWidth, vrgout_ptr_k0_r0_s1, vl) ;
      __vr vrgout_k2_r0_s1 = _vel_vgtu_vvssml(vrgout_ptr_k2_r0_s1, 0, 0, vmall_r0, vl) ;
      __vr vrgout_ptr_k3_r0_s1 = _vel_vaddsl_vsvl(4*3*gOutHeight*gOutWidth, vrgout_ptr_k0_r0_s1, vl) ;
      __vr vrgout_k3_r0_s1 = _vel_vgtu_vvssml(vrgout_ptr_k3_r0_s1, 0, 0, vmall_r0, vl) ;
      __vr vrgout_ptr_k4_r0_s1 = _vel_vaddsl_vsvl(4*4*gOutHeight*gOutWidth, vrgout_ptr_k0_r0_s1, vl) ;
      __vr vrgout_k4_r0_s1 = _vel_vgtu_vvssml(vrgout_ptr_k4_r0_s1, 0, 0, vmall_r0, vl) ;
      __vr vrgout_ptr_k5_r0_s1 = _vel_vaddsl_vsvl(4*5*gOutHeight*gOutWidth, vrgout_ptr_k0_r0_s1, vl) ;
      __vr vrgout_k5_r0_s1 = _vel_vgtu_vvssml(vrgout_ptr_k5_r0_s1, 0, 0, vmall_r0, vl) ;
      __vr vrgout_ptr_k6_r0_s1 = _vel_vaddsl_vsvl(4*6*gOutHeight*gOutWidth, vrgout_ptr_k0_r0_s1, vl) ;
      __vr vrgout_k6_r0_s1 = _vel_vgtu_vvssml(vrgout_ptr_k6_r0_s1, 0, 0, vmall_r0, vl) ;
      __vr vrgout_ptr_k7_r0_s1 = _vel_vaddsl_vsvl(4*7*gOutHeight*gOutWidth, vrgout_ptr_k0_r0_s1, vl) ;
      __vr vrgout_k7_r0_s1 = _vel_vgtu_vvssml(vrgout_ptr_k7_r0_s1, 0, 0, vmall_r0, vl) ;

      VFADD(vrgout_k0_r0_s1, vmall_r0s2, k+0, 0, 2, s12) ;
      VFADD(vrgout_k1_r0_s1, vmall_r0s2, k+1, 0, 2, s12) ;
      VFADD(vrgout_k2_r0_s1, vmall_r0s2, k+2, 0, 2, s12) ;
      VFADD(vrgout_k3_r0_s1, vmall_r0s2, k+3, 0, 2, s12) ;
      VFADD(vrgout_k4_r0_s1, vmall_r0s2, k+4, 0, 2, s12) ;
      VFADD(vrgout_k5_r0_s1, vmall_r0s2, k+5, 0, 2, s12) ;
      VFADD(vrgout_k6_r0_s1, vmall_r0s2, k+6, 0, 2, s12) ;
      VFADD(vrgout_k7_r0_s1, vmall_r0s2, k+7, 0, 2, s12) ;

      VFADD(vrgout_k0_r0_s1, vmall_r0s1, k+0, 0, 1, s12) ;
      VFADD(vrgout_k1_r0_s1, vmall_r0s1, k+1, 0, 1, s12) ;
      VFADD(vrgout_k2_r0_s1, vmall_r0s1, k+2, 0, 1, s12) ;
      VFADD(vrgout_k3_r0_s1, vmall_r0s1, k+3, 0, 1, s12) ;
      VFADD(vrgout_k4_r0_s1, vmall_r0s1, k+4, 0, 1, s12) ;
      VFADD(vrgout_k5_r0_s1, vmall_r0s1, k+5, 0, 1, s12) ;
      VFADD(vrgout_k6_r0_s1, vmall_r0s1, k+6, 0, 1, s12) ;
      VFADD(vrgout_k7_r0_s1, vmall_r0s1, k+7, 0, 1, s12) ;

      VFADD(vrgout_k0_r0_s1, vmall_r0s1, k+0, 0, 0, s0) ;
      VFADD(vrgout_k1_r0_s1, vmall_r0s1, k+1, 0, 0, s0) ;
      VFADD(vrgout_k2_r0_s1, vmall_r0s1, k+2, 0, 0, s0) ;
      VFADD(vrgout_k3_r0_s1, vmall_r0s1, k+3, 0, 0, s0) ;
      VFADD(vrgout_k4_r0_s1, vmall_r0s1, k+4, 0, 0, s0) ;
      VFADD(vrgout_k5_r0_s1, vmall_r0s1, k+5, 0, 0, s0) ;
      VFADD(vrgout_k6_r0_s1, vmall_r0s1, k+6, 0, 0, s0) ;
      VFADD(vrgout_k7_r0_s1, vmall_r0s1, k+7, 0, 0, s0) ;


      __vr vrgout_ptr_k0_r1_s1 = _vel_vsfa_vvssl(_vel_vaddsl_vvvl(_vel_vmulsl_vsvl(gOutWidth, vry_r1, vl), vrx_s1, vl),
					 2,
					 (unsigned long)(pGOut+gOutIndex), vl) ;
      __vr vrgout_k0_r1_s1 = _vel_vgtu_vvssml(vrgout_ptr_k0_r1_s1, 0, 0, vmall_r1, vl) ;
      __vr vrgout_ptr_k1_r1_s1 = _vel_vaddsl_vsvl(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_r1_s1, vl) ;
      __vr vrgout_k1_r1_s1 = _vel_vgtu_vvssml(vrgout_ptr_k1_r1_s1, 0, 0, vmall_r1, vl) ;
      __vr vrgout_ptr_k2_r1_s1 = _vel_vaddsl_vsvl(4*2*gOutHeight*gOutWidth, vrgout_ptr_k0_r1_s1, vl) ;
      __vr vrgout_k2_r1_s1 = _vel_vgtu_vvssml(vrgout_ptr_k2_r1_s1, 0, 0, vmall_r1, vl) ;
      __vr vrgout_ptr_k3_r1_s1 = _vel_vaddsl_vsvl(4*3*gOutHeight*gOutWidth, vrgout_ptr_k0_r1_s1, vl) ;
      __vr vrgout_k3_r1_s1 = _vel_vgtu_vvssml(vrgout_ptr_k3_r1_s1, 0, 0, vmall_r1, vl) ;
      __vr vrgout_ptr_k4_r1_s1 = _vel_vaddsl_vsvl(4*4*gOutHeight*gOutWidth, vrgout_ptr_k0_r1_s1, vl) ;
      __vr vrgout_k4_r1_s1 = _vel_vgtu_vvssml(vrgout_ptr_k4_r1_s1, 0, 0, vmall_r1, vl) ;
      __vr vrgout_ptr_k5_r1_s1 = _vel_vaddsl_vsvl(4*5*gOutHeight*gOutWidth, vrgout_ptr_k0_r1_s1, vl) ;
      __vr vrgout_k5_r1_s1 = _vel_vgtu_vvssml(vrgout_ptr_k5_r1_s1, 0, 0, vmall_r1, vl) ;
      __vr vrgout_ptr_k6_r1_s1 = _vel_vaddsl_vsvl(4*6*gOutHeight*gOutWidth, vrgout_ptr_k0_r1_s1, vl) ;
      __vr vrgout_k6_r1_s1 = _vel_vgtu_vvssml(vrgout_ptr_k6_r1_s1, 0, 0, vmall_r1, vl) ;
      __vr vrgout_ptr_k7_r1_s1 = _vel_vaddsl_vsvl(4*7*gOutHeight*gOutWidth, vrgout_ptr_k0_r1_s1, vl) ;
      __vr vrgout_k7_r1_s1 = _vel_vgtu_vvssml(vrgout_ptr_k7_r1_s1, 0, 0, vmall_r1, vl) ;

      VFADD(vrgout_k0_r1_s1, vmall_r1s2, k+0, 1, 2, s12) ;
      VFADD(vrgout_k1_r1_s1, vmall_r1s2, k+1, 1, 2, s12) ;
      VFADD(vrgout_k2_r1_s1, vmall_r1s2, k+2, 1, 2, s12) ;
      VFADD(vrgout_k3_r1_s1, vmall_r1s2, k+3, 1, 2, s12) ;
      VFADD(vrgout_k4_r1_s1, vmall_r1s2, k+4, 1, 2, s12) ;
      VFADD(vrgout_k5_r1_s1, vmall_r1s2, k+5, 1, 2, s12) ;
      VFADD(vrgout_k6_r1_s1, vmall_r1s2, k+6, 1, 2, s12) ;
      VFADD(vrgout_k7_r1_s1, vmall_r1s2, k+7, 1, 2, s12) ;

      VFADD(vrgout_k0_r1_s1, vmall_r1s1, k+0, 1, 1, s12) ;
      VFADD(vrgout_k1_r1_s1, vmall_r1s1, k+1, 1, 1, s12) ;
      VFADD(vrgout_k2_r1_s1, vmall_r1s1, k+2, 1, 1, s12) ;
      VFADD(vrgout_k3_r1_s1, vmall_r1s1, k+3, 1, 1, s12) ;
      VFADD(vrgout_k4_r1_s1, vmall_r1s1, k+4, 1, 1, s12) ;
      VFADD(vrgout_k5_r1_s1, vmall_r1s1, k+5, 1, 1, s12) ;
      VFADD(vrgout_k6_r1_s1, vmall_r1s1, k+6, 1, 1, s12) ;
      VFADD(vrgout_k7_r1_s1, vmall_r1s1, k+7, 1, 1, s12) ;

      VFADD(vrgout_k0_r1_s1, vmall_r1s1, k+0, 1, 0, s0) ;
      VFADD(vrgout_k1_r1_s1, vmall_r1s1, k+1, 1, 0, s0) ;
      VFADD(vrgout_k2_r1_s1, vmall_r1s1, k+2, 1, 0, s0) ;
      VFADD(vrgout_k3_r1_s1, vmall_r1s1, k+3, 1, 0, s0) ;
      VFADD(vrgout_k4_r1_s1, vmall_r1s1, k+4, 1, 0, s0) ;
      VFADD(vrgout_k5_r1_s1, vmall_r1s1, k+5, 1, 0, s0) ;
      VFADD(vrgout_k6_r1_s1, vmall_r1s1, k+6, 1, 0, s0) ;
      VFADD(vrgout_k7_r1_s1, vmall_r1s1, k+7, 1, 0, s0) ;


      __vr vrgout_ptr_k0_r2_s1 = _vel_vsfa_vvssl(_vel_vaddsl_vvvl(_vel_vmulsl_vsvl(gOutWidth, vry_r2, vl), vrx_s1, vl),
					 2,
					 (unsigned long)(pGOut+gOutIndex), vl) ;
      __vr vrgout_k0_r2_s1 = _vel_vgtu_vvssml(vrgout_ptr_k0_r2_s1, 0, 0, vmall_r2 , vl) ;
      __vr vrgout_ptr_k1_r2_s1 = _vel_vaddsl_vsvl(4*1*gOutHeight*gOutWidth, vrgout_ptr_k0_r2_s1, vl) ;
      __vr vrgout_k1_r2_s1 = _vel_vgtu_vvssml(vrgout_ptr_k1_r2_s1, 0, 0, vmall_r2 , vl) ;
      __vr vrgout_ptr_k2_r2_s1 = _vel_vaddsl_vsvl(4*2*gOutHeight*gOutWidth, vrgout_ptr_k0_r2_s1, vl) ;
      __vr vrgout_k2_r2_s1 = _vel_vgtu_vvssml(vrgout_ptr_k2_r2_s1, 0, 0, vmall_r2 , vl) ;
      __vr vrgout_ptr_k3_r2_s1 = _vel_vaddsl_vsvl(4*3*gOutHeight*gOutWidth, vrgout_ptr_k0_r2_s1, vl) ;
      __vr vrgout_k3_r2_s1 = _vel_vgtu_vvssml(vrgout_ptr_k3_r2_s1, 0, 0, vmall_r2 , vl) ;
      __vr vrgout_ptr_k4_r2_s1 = _vel_vaddsl_vsvl(4*4*gOutHeight*gOutWidth, vrgout_ptr_k0_r2_s1, vl) ;
      __vr vrgout_k4_r2_s1 = _vel_vgtu_vvssml(vrgout_ptr_k4_r2_s1, 0, 0, vmall_r2 , vl) ;
      __vr vrgout_ptr_k5_r2_s1 = _vel_vaddsl_vsvl(4*5*gOutHeight*gOutWidth, vrgout_ptr_k0_r2_s1, vl) ;
      __vr vrgout_k5_r2_s1 = _vel_vgtu_vvssml(vrgout_ptr_k5_r2_s1, 0, 0, vmall_r2 , vl) ;
      __vr vrgout_ptr_k6_r2_s1 = _vel_vaddsl_vsvl(4*6*gOutHeight*gOutWidth, vrgout_ptr_k0_r2_s1, vl) ;
      __vr vrgout_k6_r2_s1 = _vel_vgtu_vvssml(vrgout_ptr_k6_r2_s1, 0, 0, vmall_r2 , vl) ;
      __vr vrgout_ptr_k7_r2_s1 = _vel_vaddsl_vsvl(4*7*gOutHeight*gOutWidth, vrgout_ptr_k0_r2_s1, vl) ;
      __vr vrgout_k7_r2_s1 = _vel_vgtu_vvssml(vrgout_ptr_k7_r2_s1, 0, 0, vmall_r2 , vl) ;

      VFADD(vrgout_k0_r2_s1, vmall_r2s2, k+0, 2, 2, s12) ;
      VFADD(vrgout_k1_r2_s1, vmall_r2s2, k+1, 2, 2, s12) ;
      VFADD(vrgout_k2_r2_s1, vmall_r2s2, k+2, 2, 2, s12) ;
      VFADD(vrgout_k3_r2_s1, vmall_r2s2, k+3, 2, 2, s12) ;
      VFADD(vrgout_k4_r2_s1, vmall_r2s2, k+4, 2, 2, s12) ;
      VFADD(vrgout_k5_r2_s1, vmall_r2s2, k+5, 2, 2, s12) ;
      VFADD(vrgout_k6_r2_s1, vmall_r2s2, k+6, 2, 2, s12) ;
      VFADD(vrgout_k7_r2_s1, vmall_r2s2, k+7, 2, 2, s12) ;

      VFADD(vrgout_k0_r2_s1, vmall_r2s1, k+0, 2, 1, s12) ;
      VFADD(vrgout_k1_r2_s1, vmall_r2s1, k+1, 2, 1, s12) ;
      VFADD(vrgout_k2_r2_s1, vmall_r2s1, k+2, 2, 1, s12) ;
      VFADD(vrgout_k3_r2_s1, vmall_r2s1, k+3, 2, 1, s12) ;
      VFADD(vrgout_k4_r2_s1, vmall_r2s1, k+4, 2, 1, s12) ;
      VFADD(vrgout_k5_r2_s1, vmall_r2s1, k+5, 2, 1, s12) ;
      VFADD(vrgout_k6_r2_s1, vmall_r2s1, k+6, 2, 1, s12) ;
      VFADD(vrgout_k7_r2_s1, vmall_r2s1, k+7, 2, 1, s12) ;

      VFADD(vrgout_k0_r2_s1, vmall_r2s1, k+0, 2, 0, s0) ;
      VFADD(vrgout_k1_r2_s1, vmall_r2s1, k+1, 2, 0, s0) ;
      VFADD(vrgout_k2_r2_s1, vmall_r2s1, k+2, 2, 0, s0) ;
      VFADD(vrgout_k3_r2_s1, vmall_r2s1, k+3, 2, 0, s0) ;
      VFADD(vrgout_k4_r2_s1, vmall_r2s1, k+4, 2, 0, s0) ;
      VFADD(vrgout_k5_r2_s1, vmall_r2s1, k+5, 2, 0, s0) ;
      VFADD(vrgout_k6_r2_s1, vmall_r2s1, k+6, 2, 0, s0) ;
      VFADD(vrgout_k7_r2_s1, vmall_r2s1, k+7, 2, 0, s0) ;

#undef VFADD
#undef FILTER_OFFSET
    } // gOutChannel

    if(NUMCHANNEL>= 2) {
      __vr vrsum01 = _vel_pvfadd_vvvl(_vel_vmrg_vsvml(0UL, _vel_vmv_vsvl(1, vrsum01_s0, vl), vmx_s0, vl), vrsum01_s12, vl) ;
      _vel_vstu_vssl(vrsum01, 4, pGIn+gInIndex + 0 * gInHeight * gInWidth, vl) ;
      _vel_vstl_vssl(vrsum01, 4, pGIn+gInIndex + 1 * gInHeight * gInWidth, vl) ;
    }
    if(NUMCHANNEL>= 4) {
      __vr vrsum23 = _vel_pvfadd_vvvl(_vel_vmrg_vsvml(0UL, _vel_vmv_vsvl(1, vrsum23_s0, vl), vmx_s0, vl), vrsum23_s12, vl) ;
      _vel_vstu_vssl(vrsum23, 4, pGIn+gInIndex + 2 * gInHeight * gInWidth, vl) ;
      _vel_vstl_vssl(vrsum23, 4, pGIn+gInIndex + 3 * gInHeight * gInWidth, vl) ;
    }
    if(NUMCHANNEL>= 6) {
      __vr vrsum45 = _vel_pvfadd_vvvl(_vel_vmrg_vsvml(0UL, _vel_vmv_vsvl(1, vrsum45_s0, vl), vmx_s0, vl), vrsum45_s12, vl) ;
      _vel_vstu_vssl(vrsum45, 4, pGIn+gInIndex + 4 * gInHeight * gInWidth, vl) ;
      _vel_vstl_vssl(vrsum45, 4, pGIn+gInIndex + 5 * gInHeight * gInWidth, vl) ;
    }
    if(NUMCHANNEL>= 8) {
      __vr vrsum67 = _vel_pvfadd_vvvl(_vel_vmrg_vsvml(0UL, _vel_vmv_vsvl(1, vrsum67_s0, vl), vmx_s0, vl), vrsum67_s12, vl) ;
      _vel_vstu_vssl(vrsum67, 4, pGIn+gInIndex + 6 * gInHeight * gInWidth, vl) ;
      _vel_vstl_vssl(vrsum67, 4, pGIn+gInIndex + 7 * gInHeight * gInWidth, vl) ;
    }
    if(NUMCHANNEL>=10) {
      __vr vrsum89 = _vel_pvfadd_vvvl(_vel_vmrg_vsvml(0UL, _vel_vmv_vsvl(1, vrsum89_s0, vl), vmx_s0, vl), vrsum89_s12, vl) ;
      _vel_vstu_vssl(vrsum89, 4, pGIn+gInIndex + 8 * gInHeight * gInWidth, vl) ;
      _vel_vstl_vssl(vrsum89, 4, pGIn+gInIndex + 9 * gInHeight * gInWidth, vl) ;
    }
    if(NUMCHANNEL>=12) {
      __vr vrsumAB = _vel_pvfadd_vvvl(_vel_vmrg_vsvml(0UL, _vel_vmv_vsvl(1, vrsumAB_s0, vl), vmx_s0, vl), vrsumAB_s12, vl) ;
      _vel_vstu_vssl(vrsumAB, 4, pGIn+gInIndex +10 * gInHeight * gInWidth, vl) ;
      _vel_vstl_vssl(vrsumAB, 4, pGIn+gInIndex +11 * gInHeight * gInWidth, vl) ;
    }
    if(NUMCHANNEL>=14) {
      __vr vrsumCD = _vel_pvfadd_vvvl(_vel_vmrg_vsvml(0UL, _vel_vmv_vsvl(1, vrsumCD_s0, vl), vmx_s0, vl), vrsumCD_s12, vl) ;
      _vel_vstu_vssl(vrsumCD, 4, pGIn+gInIndex +12 * gInHeight * gInWidth, vl) ;
      _vel_vstl_vssl(vrsumCD, 4, pGIn+gInIndex +13 * gInHeight * gInWidth, vl) ;
    }
    if(NUMCHANNEL>=16) {
      __vr vrsumEF = _vel_pvfadd_vvvl(_vel_vmrg_vsvml(0UL, _vel_vmv_vsvl(1, vrsumEF_s0, vl), vmx_s0, vl), vrsumEF_s12, vl) ;
      _vel_vstu_vssl(vrsumEF, 4, pGIn+gInIndex +14 * gInHeight * gInWidth, vl) ;
      _vel_vstl_vssl(vrsumEF, 4, pGIn+gInIndex +15 * gInHeight * gInWidth, vl) ;
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
	func_odd<FLAYOUT, 1>(pGOut, pKernel, pGIn,
	   gOutChannel, gOutWidth, gOutHeight,
	   gInChannel, gInWidth, gInHeight,
	   kernWidth, kernHeight,
	   strideWidth, strideHeight,
	   padWidth, padHeight,
	   dilationWidth, dilationHeight,
	   gInChannelGroup, gOutChannelGroup,
	   gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	   n, c, nH) ;
	c+=1 ;
	break ;
      case 2:
	func_even<FLAYOUT, 2>(pGOut, pKernel, pGIn,
	   gOutChannel, gOutWidth, gOutHeight,
	   gInChannel, gInWidth, gInHeight,
	   kernWidth, kernHeight,
	   strideWidth, strideHeight,
	   padWidth, padHeight,
	   dilationWidth, dilationHeight,
	   gInChannelGroup, gOutChannelGroup,
	   gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	   n, c, nH) ;
	c+=2 ;
	break ;
      case 3:
	func_odd<FLAYOUT, 3>(pGOut, pKernel, pGIn,
	   gOutChannel, gOutWidth, gOutHeight,
	   gInChannel, gInWidth, gInHeight,
	   kernWidth, kernHeight,
	   strideWidth, strideHeight,
	   padWidth, padHeight,
	   dilationWidth, dilationHeight,
	   gInChannelGroup, gOutChannelGroup,
	   gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	   n, c, nH) ;
	c+=3 ;
	break ;
      case 4:
	func_even<FLAYOUT, 4>(pGOut, pKernel, pGIn,
	   gOutChannel, gOutWidth, gOutHeight,
	   gInChannel, gInWidth, gInHeight,
	   kernWidth, kernHeight,
	   strideWidth, strideHeight,
	   padWidth, padHeight,
	   dilationWidth, dilationHeight,
	   gInChannelGroup, gOutChannelGroup,
	   gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	   n, c, nH) ;
	c+=4 ;
	break ;
      case 5:
	func_odd<FLAYOUT, 5>(pGOut, pKernel, pGIn,
	   gOutChannel, gOutWidth, gOutHeight,
	   gInChannel, gInWidth, gInHeight,
	   kernWidth, kernHeight,
	   strideWidth, strideHeight,
	   padWidth, padHeight,
	   dilationWidth, dilationHeight,
	   gInChannelGroup, gOutChannelGroup,
	   gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	   n, c, nH) ;
	c+=5 ;
	break ;
      case 6:
	func_even<FLAYOUT, 6>(pGOut, pKernel, pGIn,
	   gOutChannel, gOutWidth, gOutHeight,
	   gInChannel, gInWidth, gInHeight,
	   kernWidth, kernHeight,
	   strideWidth, strideHeight,
	   padWidth, padHeight,
	   dilationWidth, dilationHeight,
	   gInChannelGroup, gOutChannelGroup,
	   gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	   n, c, nH) ;
	c+=6 ;
	break ;
      case 7:
	func_odd<FLAYOUT, 7>(pGOut, pKernel, pGIn,
	   gOutChannel, gOutWidth, gOutHeight,
	   gInChannel, gInWidth, gInHeight,
	   kernWidth, kernHeight,
	   strideWidth, strideHeight,
	   padWidth, padHeight,
	   dilationWidth, dilationHeight,
	   gInChannelGroup, gOutChannelGroup,
	   gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	   n, c, nH) ;
	c+=7 ;
	break ;
      case 8:
	func_even<FLAYOUT, 8>(pGOut, pKernel, pGIn,
	   gOutChannel, gOutWidth, gOutHeight,
	   gInChannel, gInWidth, gInHeight,
	   kernWidth, kernHeight,
	   strideWidth, strideHeight,
	   padWidth, padHeight,
	   dilationWidth, dilationHeight,
	   gInChannelGroup, gOutChannelGroup,
	   gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	   n, c, nH) ;
	c+=8 ;
	break ;
      case 9:
	func_odd<FLAYOUT, 9>(pGOut, pKernel, pGIn,
	   gOutChannel, gOutWidth, gOutHeight,
	   gInChannel, gInWidth, gInHeight,
	   kernWidth, kernHeight,
	   strideWidth, strideHeight,
	   padWidth, padHeight,
	   dilationWidth, dilationHeight,
	   gInChannelGroup, gOutChannelGroup,
	   gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	   n, c, nH) ;
	c+=9 ;
	break ;
      case 10:
	func_even<FLAYOUT, 10>(pGOut, pKernel, pGIn,
	   gOutChannel, gOutWidth, gOutHeight,
	   gInChannel, gInWidth, gInHeight,
	   kernWidth, kernHeight,
	   strideWidth, strideHeight,
	   padWidth, padHeight,
	   dilationWidth, dilationHeight,
	   gInChannelGroup, gOutChannelGroup,
	   gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	   n, c, nH) ;
	c+=10 ;
	break ;
      case 11:
	func_odd<FLAYOUT, 11>(pGOut, pKernel, pGIn,
	   gOutChannel, gOutWidth, gOutHeight,
	   gInChannel, gInWidth, gInHeight,
	   kernWidth, kernHeight,
	   strideWidth, strideHeight,
	   padWidth, padHeight,
	   dilationWidth, dilationHeight,
	   gInChannelGroup, gOutChannelGroup,
	   gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	   n, c, nH) ;
	c+=11 ;
	break ;
      case 12:
	func_even<FLAYOUT, 12>(pGOut, pKernel, pGIn,
	   gOutChannel, gOutWidth, gOutHeight,
	   gInChannel, gInWidth, gInHeight,
	   kernWidth, kernHeight,
	   strideWidth, strideHeight,
	   padWidth, padHeight,
	   dilationWidth, dilationHeight,
	   gInChannelGroup, gOutChannelGroup,
	   gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	   n, c, nH) ;
	c+=12 ;
	break ;
      case 13:
	func_odd<FLAYOUT, 13>(pGOut, pKernel, pGIn,
	   gOutChannel, gOutWidth, gOutHeight,
	   gInChannel, gInWidth, gInHeight,
	   kernWidth, kernHeight,
	   strideWidth, strideHeight,
	   padWidth, padHeight,
	   dilationWidth, dilationHeight,
	   gInChannelGroup, gOutChannelGroup,
	   gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	   n, c, nH) ;
	c+=13 ;
	break ;
      case 14:
	func_even<FLAYOUT, 14>(pGOut, pKernel, pGIn,
	   gOutChannel, gOutWidth, gOutHeight,
	   gInChannel, gInWidth, gInHeight,
	   kernWidth, kernHeight,
	   strideWidth, strideHeight,
	   padWidth, padHeight,
	   dilationWidth, dilationHeight,
	   gInChannelGroup, gOutChannelGroup,
	   gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	   n, c, nH) ;
	c+=14 ;
	break ;
      case 15:
	func_odd<FLAYOUT, 15>(pGOut, pKernel, pGIn,
	   gOutChannel, gOutWidth, gOutHeight,
	   gInChannel, gInWidth, gInHeight,
	   kernWidth, kernHeight,
	   strideWidth, strideHeight,
	   padWidth, padHeight,
	   dilationWidth, dilationHeight,
	   gInChannelGroup, gOutChannelGroup,
	   gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	   n, c, nH) ;
	c+=15 ;
	break ;
      default :
	break ;
      }
      for (; c<gInChannelGroup; ) {
	func_even<FLAYOUT, 16>(pGOut, pKernel, pGIn,
	   gOutChannel, gOutWidth, gOutHeight,
	   gInChannel, gInWidth, gInHeight,
	   kernWidth, kernHeight,
	   strideWidth, strideHeight,
	   padWidth, padHeight,
	   dilationWidth, dilationHeight,
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
vednnConvolutionBackwardData_direct_dil1_str2_pad1_ker3_iwU128(
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
