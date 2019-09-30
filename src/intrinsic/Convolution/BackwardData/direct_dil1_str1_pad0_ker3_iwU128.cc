#include <stdint.h>
#include <stdlib.h>

#include "vednn.h"
#include "vednn_util.h"

#include "velintrin.h"
#define VLEN	(256)

template<filterLayout_t FLAYOUT, int NUMCHANNEL>
static inline void func_odd(
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
  __vr vrhw = _vel_vaddsl_vvvl(_vel_vmulsl_vsvl(gOutWidth, vrh, nH*gInWidth), vrw, nH*gInWidth) ;

  __vr vrx_s2 = _vel_vaddsl_vsvl(-2, vrw, nH*gInWidth) ;
  __vm256 vmx1_s2 =  _vel_vfmklge_mvl(vrx_s2, nH*gInWidth) ;
  __vm256 vmx_s2 = vmx1_s2 ;

  __vr vrx_s1 = _vel_vaddsl_vsvl(-1, vrw, nH*gInWidth) ;
  __vm256 vmx1_s1 =  _vel_vfmklge_mvl(vrx_s1, nH*gInWidth) ;
  __vm256 vmx2_s1 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutWidth,vrx_s1, nH*gInWidth), nH*gInWidth) ;
  __vm256 vmx_s1 = _vel_andm_mmm(vmx1_s1, vmx2_s1) ;

  __vr vrx_s0 = vrw ;
  __vm256 vmx2_s0 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutWidth,vrx_s0, nH*gInWidth), nH*gInWidth) ;
  __vm256 vmx_s0 = vmx2_s0 ;

  for (int64_t h=0; h<gInHeight; h+=nH) {
    const int64_t vl = gInWidth * (gInHeight - h < nH ? gInHeight - h : nH) ;

    __vr vrsum0  = _vel_vbrds_vsl(0.f, vl) ;
    __vr vrsum12 = _vel_pvbrd_vsl(0UL, vl) ;
    __vr vrsum34 = _vel_pvbrd_vsl(0UL, vl) ;
    __vr vrsum56 = _vel_pvbrd_vsl(0UL, vl) ;
    __vr vrsum78 = _vel_pvbrd_vsl(0UL, vl) ;
    __vr vrsum9A = _vel_pvbrd_vsl(0UL, vl) ;
    __vr vrsumBC = _vel_pvbrd_vsl(0UL, vl) ;
    __vr vrsumDE = _vel_pvbrd_vsl(0UL, vl) ;

    __vr vry_r2 = _vel_vaddsl_vsvl(h-2, vrh, vl) ;
    __vm256 vmy1_r2 =  _vel_vfmklge_mvl(vry_r2, vl) ;
    __vm256 vmy_r2 = vmy1_r2 ;

    __vm256 vmall_r2s2 = _vel_andm_mmm(vmy_r2,vmx_s2) ;
    __vm256 vmall_r2s1 = _vel_andm_mmm(vmy_r2,vmx_s1) ;
    __vm256 vmall_r2s0 = _vel_andm_mmm(vmy_r2,vmx_s0) ;

    __vr vry_r1 = _vel_vaddsl_vsvl(h-1, vrh, vl) ;
    __vm256 vmy1_r1 =  _vel_vfmklge_mvl(vry_r1, vl) ;
    __vm256 vmy2_r1 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutHeight,vry_r1, vl), vl) ;
    __vm256 vmy_r1 = _vel_andm_mmm(vmy1_r1, vmy2_r1) ;

    __vm256 vmall_r1s2 = _vel_andm_mmm(vmy_r1,vmx_s2) ;
    __vm256 vmall_r1s1 = _vel_andm_mmm(vmy_r1,vmx_s1) ;
    __vm256 vmall_r1s0 = _vel_andm_mmm(vmy_r1,vmx_s0) ;

    __vr vry_r0= _vel_vaddsl_vsvl(h, vrh, vl) ;
    __vm256 vmy2_r0 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutHeight,vry_r0, vl), vl) ;
    __vm256 vmy_r0 = vmy2_r0 ;

    __vm256 vmall_r0s2 = _vel_andm_mmm(vmy_r0,vmx_s2) ;
    __vm256 vmall_r0s1 = _vel_andm_mmm(vmy_r0,vmx_s1) ;
    __vm256 vmall_r0s0 = _vel_andm_mmm(vmy_r0,vmx_s0) ;

    for (int64_t k=0; k<gOutChannelGroup; k++) {
      int64_t gOutIndex    = gOutGroupOffset + ((n * gOutChannel + k) * gOutHeight) * gOutWidth ;

      __vr vrgout_ptr_r2s2 = _vel_vsfa_vvssl(vrhw, 2, (unsigned long)(pGOut+gOutIndex+(h-2)*gOutWidth-2), vl) ;
      __vr vrgout_ptr_r1s2 = _vel_vsfa_vvssl(vrhw, 2, (unsigned long)(pGOut+gOutIndex+(h-1)*gOutWidth-2), vl) ;
      __vr vrgout_ptr_r0s2 = _vel_vsfa_vvssl(vrhw, 2, (unsigned long)(pGOut+gOutIndex+(h-0)*gOutWidth-2), vl) ;

      __vr vrgout_r2s2 = _vel_vgtu_vvssml(vrgout_ptr_r2s2, 0, 0, vmall_r2s2, vl) ;
      __vr vrgout_r1s2 = _vel_vgtu_vvssml(vrgout_ptr_r1s2, 0, 0, vmall_r1s2, vl) ;
      __vr vrgout_r0s2 = _vel_vgtu_vvssml(vrgout_ptr_r0s2, 0, 0, vmall_r0s2, vl) ;

      __vr vrgout_r2s1 = _vel_vmv_vsvl(1, vrgout_r2s2, vl) ;
      __vr vrgout_r2s0 = _vel_vmv_vsvl(2, vrgout_r2s2, vl) ;

      __vr vrgout_r1s1 = _vel_vmv_vsvl(1, vrgout_r1s2, vl) ;
      __vr vrgout_r1s0 = _vel_vmv_vsvl(2, vrgout_r1s2, vl) ;

      __vr vrgout_r0s1 = _vel_vmv_vsvl(1, vrgout_r0s2, vl) ;
      __vr vrgout_r0s0 = _vel_vmv_vsvl(2, vrgout_r0s2, vl) ;

#define FILTER_OFFSET(k,c,r,s) ( kernGroupOffset + filter_index<FLAYOUT>(k,c,r,s, gInChannelGroup, gOutChannelGroup, kernHeight, kernWidth) )

#define VFADD(VRGOUT,VM,K,R,S) {								\
	const float    kerValue0  = pKernel[FILTER_OFFSET(K,c+ 0,R,S)] ;			\
	const uint64_t kerValue12 = _vel_pack_f32p(pKernel + FILTER_OFFSET(K,c+ 1,R,S),		\
						   pKernel + FILTER_OFFSET(K,c+ 2,R,S)) ;	\
	const uint64_t kerValue34 = _vel_pack_f32p(pKernel + FILTER_OFFSET(K,c+ 3,R,S),		\
						   pKernel + FILTER_OFFSET(K,c+ 4,R,S)) ;	\
	const uint64_t kerValue56 = _vel_pack_f32p(pKernel + FILTER_OFFSET(K,c+ 5,R,S),		\
						   pKernel + FILTER_OFFSET(K,c+ 6,R,S)) ;	\
	const uint64_t kerValue78 = _vel_pack_f32p(pKernel + FILTER_OFFSET(K,c+ 7,R,S),		\
						   pKernel + FILTER_OFFSET(K,c+ 8,R,S)) ;	\
	const uint64_t kerValue9A = _vel_pack_f32p(pKernel + FILTER_OFFSET(K,c+ 9,R,S),		\
						   pKernel + FILTER_OFFSET(K,c+10,R,S)) ;	\
	const uint64_t kerValueBC = _vel_pack_f32p(pKernel + FILTER_OFFSET(K,c+11,R,S),		\
						   pKernel + FILTER_OFFSET(K,c+12,R,S)) ;	\
	const uint64_t kerValueDE = _vel_pack_f32p(pKernel + FILTER_OFFSET(K,c+13,R,S),		\
						   pKernel + FILTER_OFFSET(K,c+14,R,S)) ;	\
	VRGOUT = _vel_vmrg_vsvml(0.f, VRGOUT, VM, vl) ;						\
	__vr vrgoutP = _vel_vshf_vvvsl(VRGOUT, VRGOUT, VE_VSHUFFLE_YUZU, vl) ;			\
	vrsum0 = _vel_vfmads_vvsvl(vrsum0, kerValue0, VRGOUT, vl) ;				\
	if(NUMCHANNEL>= 3) vrsum12 = _vel_pvfmad_vvsvl(vrsum12, kerValue12, vrgoutP, vl) ;	\
	if(NUMCHANNEL>= 5) vrsum34 = _vel_pvfmad_vvsvl(vrsum34, kerValue34, vrgoutP, vl) ;	\
	if(NUMCHANNEL>= 7) vrsum56 = _vel_pvfmad_vvsvl(vrsum56, kerValue56, vrgoutP, vl) ;	\
	if(NUMCHANNEL>= 9) vrsum78 = _vel_pvfmad_vvsvl(vrsum78, kerValue78, vrgoutP, vl) ;	\
	if(NUMCHANNEL>=11) vrsum9A = _vel_pvfmad_vvsvl(vrsum9A, kerValue9A, vrgoutP, vl) ;	\
	if(NUMCHANNEL>=13) vrsumBC = _vel_pvfmad_vvsvl(vrsumBC, kerValueBC, vrgoutP, vl) ;	\
	if(NUMCHANNEL>=15) vrsumDE = _vel_pvfmad_vvsvl(vrsumDE, kerValueDE, vrgoutP, vl) ;	\
      }

      VFADD(vrgout_r2s2, vmall_r2s2, k, 2, 2) ;
      VFADD(vrgout_r2s1, vmall_r2s1, k, 2, 1) ;
      VFADD(vrgout_r2s0, vmall_r2s0, k, 2, 0) ;

      VFADD(vrgout_r1s2, vmall_r1s2, k, 1, 2) ;
      VFADD(vrgout_r1s1, vmall_r1s1, k, 1, 1) ;
      VFADD(vrgout_r1s0, vmall_r1s0, k, 1, 0) ;

      VFADD(vrgout_r0s2, vmall_r0s2, k, 0, 2) ;
      VFADD(vrgout_r0s1, vmall_r0s1, k, 0, 1) ;
      VFADD(vrgout_r0s0, vmall_r0s0, k, 0, 0) ;

#undef VFADD
#undef FILTER_OFFSET
    } // gOutChannel

    _vel_vstu_vssl(vrsum0, 4, pGIn+gInIndex + 0 * gInHeight * gInWidth, vl) ;
    if(NUMCHANNEL>= 3) {
	_vel_vstu_vssl(vrsum12, 4, pGIn+gInIndex + 1 * gInHeight * gInWidth, vl) ;
	_vel_vstl_vssl(vrsum12, 4, pGIn+gInIndex + 2 * gInHeight * gInWidth, vl) ;
    }
    if(NUMCHANNEL>= 5) {
	_vel_vstu_vssl(vrsum34, 4, pGIn+gInIndex + 3 * gInHeight * gInWidth, vl) ;
	_vel_vstl_vssl(vrsum34, 4, pGIn+gInIndex + 4 * gInHeight * gInWidth, vl) ;
    }
    if(NUMCHANNEL>= 7) {
	_vel_vstu_vssl(vrsum56, 4, pGIn+gInIndex + 5 * gInHeight * gInWidth, vl) ;
	_vel_vstl_vssl(vrsum56, 4, pGIn+gInIndex + 6 * gInHeight * gInWidth, vl) ;
    }
    if(NUMCHANNEL>= 9) {
	_vel_vstu_vssl(vrsum78, 4, pGIn+gInIndex + 7 * gInHeight * gInWidth, vl) ;
	_vel_vstl_vssl(vrsum78, 4, pGIn+gInIndex + 8 * gInHeight * gInWidth, vl) ;
    }
    if(NUMCHANNEL>=11) {
	_vel_vstu_vssl(vrsum9A, 4, pGIn+gInIndex + 9 * gInHeight * gInWidth, vl) ;
	_vel_vstl_vssl(vrsum9A, 4, pGIn+gInIndex +10 * gInHeight * gInWidth, vl) ;
    }
    if(NUMCHANNEL>=13) {
	_vel_vstu_vssl(vrsumBC, 4, pGIn+gInIndex +11 * gInHeight * gInWidth, vl) ;
	_vel_vstl_vssl(vrsumBC, 4, pGIn+gInIndex +12 * gInHeight * gInWidth, vl) ;
    }
    if(NUMCHANNEL>=15) {
	_vel_vstu_vssl(vrsumDE, 4, pGIn+gInIndex +13 * gInHeight * gInWidth, vl) ;
	_vel_vstl_vssl(vrsumDE, 4, pGIn+gInIndex +14 * gInHeight * gInWidth, vl) ;
    }

    gInIndex += vl ;
  } // gOutPixels
}


template<filterLayout_t FLAYOUT, int NUMCHANNEL>
static inline void func_even(
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
  __vr vrhw = _vel_vaddsl_vvvl(_vel_vmulsl_vsvl(gOutWidth, vrh, nH*gInWidth), vrw, nH*gInWidth) ;

  __vr vrx_s2 = _vel_vaddsl_vsvl(-2, vrw, nH*gInWidth) ;
  __vm256 vmx1_s2 =  _vel_vfmklge_mvl(vrx_s2, nH*gInWidth) ;
  __vm256 vmx_s2 = vmx1_s2 ;

  __vr vrx_s1 = _vel_vaddsl_vsvl(-1, vrw, nH*gInWidth) ;
  __vm256 vmx1_s1 =  _vel_vfmklge_mvl(vrx_s1, nH*gInWidth) ;
  __vm256 vmx2_s1 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutWidth,vrx_s1, nH*gInWidth), nH*gInWidth) ;
  __vm256 vmx_s1 = _vel_andm_mmm(vmx1_s1, vmx2_s1) ;

  __vr vrx_s0 = vrw ;
  __vm256 vmx2_s0 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutWidth,vrx_s0, nH*gInWidth), nH*gInWidth) ;
  __vm256 vmx_s0 = vmx2_s0 ;

  for (int64_t h=0; h<gInHeight; h+=nH) {
    const int64_t vl = gInWidth * (gInHeight - h < nH ? gInHeight - h : nH) ;

    __vr vrsum01 = _vel_pvbrd_vsl(0UL, vl) ;
    __vr vrsum23 = _vel_pvbrd_vsl(0UL, vl) ;
    __vr vrsum45 = _vel_pvbrd_vsl(0UL, vl) ;
    __vr vrsum67 = _vel_pvbrd_vsl(0UL, vl) ;
    __vr vrsum89 = _vel_pvbrd_vsl(0UL, vl) ;
    __vr vrsumAB = _vel_pvbrd_vsl(0UL, vl) ;
    __vr vrsumCD = _vel_pvbrd_vsl(0UL, vl) ;
    __vr vrsumEF = _vel_pvbrd_vsl(0UL, vl) ;

    __vr vry_r2 = _vel_vaddsl_vsvl(h-2, vrh, vl) ;
    __vm256 vmy1_r2 =  _vel_vfmklge_mvl(vry_r2, vl) ;
    __vm256 vmy_r2 = vmy1_r2 ;

    __vm256 vmall_r2s2 = _vel_andm_mmm(vmy_r2,vmx_s2) ;
    __vm256 vmall_r2s1 = _vel_andm_mmm(vmy_r2,vmx_s1) ;
    __vm256 vmall_r2s0 = _vel_andm_mmm(vmy_r2,vmx_s0) ;

    __vr vry_r1 = _vel_vaddsl_vsvl(h-1, vrh, vl) ;
    __vm256 vmy1_r1 =  _vel_vfmklge_mvl(vry_r1, vl) ;
    __vm256 vmy2_r1 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutHeight,vry_r1, vl), vl) ;
    __vm256 vmy_r1 = _vel_andm_mmm(vmy1_r1, vmy2_r1) ;

    __vm256 vmall_r1s2 = _vel_andm_mmm(vmy_r1,vmx_s2) ;
    __vm256 vmall_r1s1 = _vel_andm_mmm(vmy_r1,vmx_s1) ;
    __vm256 vmall_r1s0 = _vel_andm_mmm(vmy_r1,vmx_s0) ;

    __vr vry_r0= _vel_vaddsl_vsvl(h, vrh, vl) ;
    __vm256 vmy2_r0 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutHeight,vry_r0, vl), vl) ;
    __vm256 vmy_r0 = vmy2_r0 ;

    __vm256 vmall_r0s2 = _vel_andm_mmm(vmy_r0,vmx_s2) ;
    __vm256 vmall_r0s1 = _vel_andm_mmm(vmy_r0,vmx_s1) ;
    __vm256 vmall_r0s0 = _vel_andm_mmm(vmy_r0,vmx_s0) ;

    for (int64_t k=0; k<gOutChannelGroup; k++) {
      int64_t gOutIndex    = gOutGroupOffset + ((n * gOutChannel + k) * gOutHeight) * gOutWidth ;

       __vr vrgout_ptr_r2s2 = _vel_vsfa_vvssl(vrhw, 2, (unsigned long)(pGOut+gOutIndex+(h-2)*gOutWidth-2), vl) ;
      __vr vrgout_ptr_r1s2 = _vel_vsfa_vvssl(vrhw, 2, (unsigned long)(pGOut+gOutIndex+(h-1)*gOutWidth-2), vl) ;
      __vr vrgout_ptr_r0s2 = _vel_vsfa_vvssl(vrhw, 2, (unsigned long)(pGOut+gOutIndex+(h-0)*gOutWidth-2), vl) ;

      __vr vrgout_r2s2 = _vel_vgtu_vvssml(vrgout_ptr_r2s2, 0, 0, vmall_r2s2, vl) ;
      __vr vrgout_r1s2 = _vel_vgtu_vvssml(vrgout_ptr_r1s2, 0, 0, vmall_r1s2, vl) ;
      __vr vrgout_r0s2 = _vel_vgtu_vvssml(vrgout_ptr_r0s2, 0, 0, vmall_r0s2, vl) ;

      __vr vrgout_r2s1 = _vel_vmv_vsvl(1, vrgout_r2s2, vl) ;
      __vr vrgout_r2s0 = _vel_vmv_vsvl(2, vrgout_r2s2, vl) ;

      __vr vrgout_r1s1 = _vel_vmv_vsvl(1, vrgout_r1s2, vl) ;
      __vr vrgout_r1s0 = _vel_vmv_vsvl(2, vrgout_r1s2, vl) ;

      __vr vrgout_r0s1 = _vel_vmv_vsvl(1, vrgout_r0s2, vl) ;
      __vr vrgout_r0s0 = _vel_vmv_vsvl(2, vrgout_r0s2, vl) ;

#define FILTER_OFFSET(k,c,r,s) ( kernGroupOffset + filter_index<FLAYOUT>(k,c,r,s, gInChannelGroup, gOutChannelGroup, kernHeight, kernWidth) )

#define VFADD(VRGOUT,VM,K,R,S) {								\
	const uint64_t kerValue01 = _vel_pack_f32p(pKernel + FILTER_OFFSET(K,c+ 0,R,S),		\
						   pKernel + FILTER_OFFSET(K,c+ 1,R,S)) ;	\
	const uint64_t kerValue23 = _vel_pack_f32p(pKernel + FILTER_OFFSET(K,c+ 2,R,S),		\
						   pKernel + FILTER_OFFSET(K,c+ 3,R,S)) ;	\
	const uint64_t kerValue45 = _vel_pack_f32p(pKernel + FILTER_OFFSET(K,c+ 4,R,S),		\
						   pKernel + FILTER_OFFSET(K,c+ 5,R,S)) ;	\
	const uint64_t kerValue67 = _vel_pack_f32p(pKernel + FILTER_OFFSET(K,c+ 6,R,S),		\
						   pKernel + FILTER_OFFSET(K,c+ 7,R,S)) ;	\
	const uint64_t kerValue89 = _vel_pack_f32p(pKernel + FILTER_OFFSET(K,c+ 8,R,S),		\
						   pKernel + FILTER_OFFSET(K,c+ 9,R,S)) ;	\
	const uint64_t kerValueAB = _vel_pack_f32p(pKernel + FILTER_OFFSET(K,c+10,R,S),		\
						   pKernel + FILTER_OFFSET(K,c+11,R,S)) ;	\
	const uint64_t kerValueCD = _vel_pack_f32p(pKernel + FILTER_OFFSET(K,c+12,R,S),		\
						   pKernel + FILTER_OFFSET(K,c+13,R,S)) ;	\
	const uint64_t kerValueEF = _vel_pack_f32p(pKernel + FILTER_OFFSET(K,c+14,R,S),		\
						   pKernel + FILTER_OFFSET(K,c+15,R,S)) ;	\
	VRGOUT = _vel_vmrg_vsvml(0.f, VRGOUT, VM, vl) ;						\
	__vr vrgoutP = _vel_vshf_vvvsl(VRGOUT, VRGOUT, VE_VSHUFFLE_YUZU, vl) ;			\
	if(NUMCHANNEL>= 2) vrsum01 = _vel_pvfmad_vvsvl(vrsum01, kerValue01, vrgoutP, vl) ;	\
	if(NUMCHANNEL>= 4) vrsum23 = _vel_pvfmad_vvsvl(vrsum23, kerValue23, vrgoutP, vl) ;	\
	if(NUMCHANNEL>= 6) vrsum45 = _vel_pvfmad_vvsvl(vrsum45, kerValue45, vrgoutP, vl) ;	\
	if(NUMCHANNEL>= 8) vrsum67 = _vel_pvfmad_vvsvl(vrsum67, kerValue67, vrgoutP, vl) ;	\
	if(NUMCHANNEL>=10) vrsum89 = _vel_pvfmad_vvsvl(vrsum89, kerValue89, vrgoutP, vl) ;	\
	if(NUMCHANNEL>=12) vrsumAB = _vel_pvfmad_vvsvl(vrsumAB, kerValueAB, vrgoutP, vl) ;	\
	if(NUMCHANNEL>=14) vrsumCD = _vel_pvfmad_vvsvl(vrsumCD, kerValueCD, vrgoutP, vl) ;	\
	if(NUMCHANNEL>=16) vrsumEF = _vel_pvfmad_vvsvl(vrsumEF, kerValueEF, vrgoutP, vl) ;	\
      }

      VFADD(vrgout_r2s2, vmall_r2s2, k, 2, 2) ;
      VFADD(vrgout_r2s1, vmall_r2s1, k, 2, 1) ;
      VFADD(vrgout_r2s0, vmall_r2s0, k, 2, 0) ;

      VFADD(vrgout_r1s2, vmall_r1s2, k, 1, 2) ;
      VFADD(vrgout_r1s1, vmall_r1s1, k, 1, 1) ;
      VFADD(vrgout_r1s0, vmall_r1s0, k, 1, 0) ;

      VFADD(vrgout_r0s2, vmall_r0s2, k, 0, 2) ;
      VFADD(vrgout_r0s1, vmall_r0s1, k, 0, 1) ;
      VFADD(vrgout_r0s0, vmall_r0s0, k, 0, 0) ;

#undef VFADD
#undef FILTER_OFFSET
    } // gOutChannel

    if(NUMCHANNEL>= 2) {
	_vel_vstu_vssl(vrsum01, 4, pGIn+gInIndex + 0 * gInHeight * gInWidth, vl) ;
	_vel_vstl_vssl(vrsum01, 4, pGIn+gInIndex + 1 * gInHeight * gInWidth, vl) ;
    }
    if(NUMCHANNEL>= 4) {
	_vel_vstu_vssl(vrsum23, 4, pGIn+gInIndex + 2 * gInHeight * gInWidth, vl) ;
	_vel_vstl_vssl(vrsum23, 4, pGIn+gInIndex + 3 * gInHeight * gInWidth, vl) ;
    }
    if(NUMCHANNEL>= 6) {
	_vel_vstu_vssl(vrsum45, 4, pGIn+gInIndex + 4 * gInHeight * gInWidth, vl) ;
	_vel_vstl_vssl(vrsum45, 4, pGIn+gInIndex + 5 * gInHeight * gInWidth, vl) ;
    }
    if(NUMCHANNEL>= 8) {
	_vel_vstu_vssl(vrsum67, 4, pGIn+gInIndex + 6 * gInHeight * gInWidth, vl) ;
	_vel_vstl_vssl(vrsum67, 4, pGIn+gInIndex + 7 * gInHeight * gInWidth, vl) ;
    }
    if(NUMCHANNEL>=10) {
	_vel_vstu_vssl(vrsum89, 4, pGIn+gInIndex + 8 * gInHeight * gInWidth, vl) ;
	_vel_vstl_vssl(vrsum89, 4, pGIn+gInIndex + 9 * gInHeight * gInWidth, vl) ;
    }
    if(NUMCHANNEL>=12) {
	_vel_vstu_vssl(vrsumAB, 4, pGIn+gInIndex +10 * gInHeight * gInWidth, vl) ;
	_vel_vstl_vssl(vrsumAB, 4, pGIn+gInIndex +11 * gInHeight * gInWidth, vl) ;
    }
    if(NUMCHANNEL>=14) {
	_vel_vstu_vssl(vrsumCD, 4, pGIn+gInIndex +12 * gInHeight * gInWidth, vl) ;
	_vel_vstl_vssl(vrsumCD, 4, pGIn+gInIndex +13 * gInHeight * gInWidth, vl) ;
    }
    if(NUMCHANNEL>=16) {
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
    const int64_t gOutChannelGroup
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
	   gInChannelGroup, gOutChannelGroup,
	   gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	   n, c, nH ) ;
	c+=1 ;
	break ;
      case 2:
	func_even<FLAYOUT, 2>(pGOut, pKernel, pGIn,
	   gOutChannel, gOutWidth, gOutHeight,
	   gInChannel, gInWidth, gInHeight,
	   kernWidth, kernHeight,
	   gInChannelGroup, gOutChannelGroup,
	   gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	   n, c, nH ) ;
	c+=2 ;
	break ;
      case 3:
	func_odd<FLAYOUT, 3>(pGOut, pKernel, pGIn,
	   gOutChannel, gOutWidth, gOutHeight,
	   gInChannel, gInWidth, gInHeight,
	   kernWidth, kernHeight,
	   gInChannelGroup, gOutChannelGroup,
	   gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	   n, c, nH ) ;
	c+=3 ;
	break ;
      case 4:
	func_even<FLAYOUT, 4>(pGOut, pKernel, pGIn,
	   gOutChannel, gOutWidth, gOutHeight,
	   gInChannel, gInWidth, gInHeight,
	   kernWidth, kernHeight,
	   gInChannelGroup, gOutChannelGroup,
	   gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	   n, c, nH ) ;
	c+=4 ;
	break ;
      case 5:
	func_odd<FLAYOUT, 5>(pGOut, pKernel, pGIn,
	   gOutChannel, gOutWidth, gOutHeight,
	   gInChannel, gInWidth, gInHeight,
	   kernWidth, kernHeight,
	   gInChannelGroup, gOutChannelGroup,
	   gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	   n, c, nH ) ;
	c+=5 ;
	break ;
      case 6:
	func_even<FLAYOUT, 6>(pGOut, pKernel, pGIn,
	   gOutChannel, gOutWidth, gOutHeight,
	   gInChannel, gInWidth, gInHeight,
	   kernWidth, kernHeight,
	   gInChannelGroup, gOutChannelGroup,
	   gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	   n, c, nH ) ;
	c+=6 ;
	break ;
      case 7:
	func_odd<FLAYOUT, 7>(pGOut, pKernel, pGIn,
	   gOutChannel, gOutWidth, gOutHeight,
	   gInChannel, gInWidth, gInHeight,
	   kernWidth, kernHeight,
	   gInChannelGroup, gOutChannelGroup,
	   gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	   n, c, nH ) ;
	c+=7 ;
	break ;
      case 8:
	func_even<FLAYOUT, 8>(pGOut, pKernel, pGIn,
	   gOutChannel, gOutWidth, gOutHeight,
	   gInChannel, gInWidth, gInHeight,
	   kernWidth, kernHeight,
	   gInChannelGroup, gOutChannelGroup,
	   gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	   n, c, nH ) ;
	c+=8 ;
	break ;
      case 9:
	func_odd<FLAYOUT, 9>(pGOut, pKernel, pGIn,
	   gOutChannel, gOutWidth, gOutHeight,
	   gInChannel, gInWidth, gInHeight,
	   kernWidth, kernHeight,
	   gInChannelGroup, gOutChannelGroup,
	   gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	   n, c, nH ) ;
	c+=9 ;
	break ;
      case 10:
	func_even<FLAYOUT, 10>(pGOut, pKernel, pGIn,
	   gOutChannel, gOutWidth, gOutHeight,
	   gInChannel, gInWidth, gInHeight,
	   kernWidth, kernHeight,
	   gInChannelGroup, gOutChannelGroup,
	   gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	   n, c, nH ) ;
	c+=10 ;
	break ;
      case 11:
	func_odd<FLAYOUT, 11>(pGOut, pKernel, pGIn,
	   gOutChannel, gOutWidth, gOutHeight,
	   gInChannel, gInWidth, gInHeight,
	   kernWidth, kernHeight,
	   gInChannelGroup, gOutChannelGroup,
	   gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	   n, c, nH ) ;
	c+=11 ;
	break ;
      case 12:
	func_even<FLAYOUT, 12>(pGOut, pKernel, pGIn,
	   gOutChannel, gOutWidth, gOutHeight,
	   gInChannel, gInWidth, gInHeight,
	   kernWidth, kernHeight,
	   gInChannelGroup, gOutChannelGroup,
	   gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	   n, c, nH ) ;
	c+=12 ;
	break ;
      case 13:
	func_odd<FLAYOUT, 13>(pGOut, pKernel, pGIn,
	   gOutChannel, gOutWidth, gOutHeight,
	   gInChannel, gInWidth, gInHeight,
	   kernWidth, kernHeight,
	   gInChannelGroup, gOutChannelGroup,
	   gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	   n, c, nH ) ;
	c+=13 ;
	break ;
      case 14:
	func_even<FLAYOUT, 14>(pGOut, pKernel, pGIn,
	   gOutChannel, gOutWidth, gOutHeight,
	   gInChannel, gInWidth, gInHeight,
	   kernWidth, kernHeight,
	   gInChannelGroup, gOutChannelGroup,
	   gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	   n, c, nH ) ;
	c+=14 ;
	break ;
      case 15:
	func_odd<FLAYOUT, 15>(pGOut, pKernel, pGIn,
	   gOutChannel, gOutWidth, gOutHeight,
	   gInChannel, gInWidth, gInHeight,
	   kernWidth, kernHeight,
	   gInChannelGroup, gOutChannelGroup,
	   gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	   n, c, nH ) ;
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
	   gInChannelGroup, gOutChannelGroup,
	   gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	   n, c, nH ) ;
	c+= 16 ;
      } // gInChannel
    } // group
  } // batch
}

extern "C"
vednnError_t
vednnConvolutionBackwardData_direct_dil1_str1_pad0_ker3_iwU128(
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
//  const int64_t strideWidth    = pParamConv->strideWidth;	// 1
//  const int64_t strideHeight   = pParamConv->strideHeight;	// 1
//  const int64_t padWidth       = pParamConv->padWidth;	// 0
//  const int64_t padHeight      = pParamConv->padHeight;	// 0
//  const int64_t dilationWidth  = pParamConv->dilationWidth;	// 1
//  const int64_t dilationHeight = pParamConv->dilationHeight;	// 1

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
	       gInChannelGroup, gOutChannelGroup ) ;
  }
  else {
    convloop<VEDNN_FILTER_LAYOUT_HWCN>(pGOut, pKernel, pGIn,
	       batch, group,
	       gOutChannel, gOutWidth, gOutHeight,
	       gInChannel, gInWidth, gInHeight,
	       kernWidth, kernHeight,
	       gInChannelGroup, gOutChannelGroup ) ;
  }

  return VEDNN_SUCCESS;
}
