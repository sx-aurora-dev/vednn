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
    const int64_t c
)
{
  int64_t gInIndex = gInGroupOffset + ((n * gInChannel + c) * gInHeight) * gInWidth  ;

  for (int64_t h = 0; h < gInHeight ; h++ ) {
    for (int64_t w = 0; w < gInWidth ; w += VLEN ) {
      const int64_t vl = gInWidth - w < VLEN ? gInWidth - w  : VLEN ;

      const int64_t xmin_s0 = (w+2) / 2 ;
      const int64_t xmin_s2 = (w+0) / 2 ;
      const int64_t xmin_s4 = w >= 2 ? (w-2) / 2 : 0 ;

      __vr vrw   = _vel_vaddsl_vsvl(w, _vel_vseq_vl(vl), vl) ;

      __vr vrj_s0 = _vel_vaddsl_vsvl(2, vrw, vl) ;
      __vr vrj_s1 = _vel_vaddsl_vsvl(1, vrw, vl) ;
      __vr vrj_s2 = vrw ;
      __vr vrj_s3 = _vel_vaddsl_vsvl(-1, vrw, vl) ;
      __vr vrj_s4 = _vel_vaddsl_vsvl(-2, vrw, vl) ;

      __vr vrx_s0 = _vel_vdivsl_vvsl(vrj_s0, 2, vl) ;
      __vr vrx_s12 = _vel_vdivsl_vvsl(vrj_s1, 2, vl) ;
      __vr vrx_s34 = _vel_vdivsl_vvsl(vrj_s3, 2, vl) ;

      __vm256 vmx0_s0 =  _vel_vfmkleq_mvl(_vel_vcmpsl_vvvl(vrj_s0, _vel_vmulsl_vsvl(2, vrx_s0, vl), vl), vl) ;
      __vm256 vmx1_s0 =  _vel_vfmklge_mvl(vrx_s0, vl) ;
      __vm256 vmx2_s0 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutWidth,vrx_s0, vl), vl) ;
      __vm256 vmx_s0 = _vel_andm_mmm(_vel_andm_mmm(vmx0_s0, vmx1_s0), vmx2_s0) ;

      __vm256 vmx0_s1 =  _vel_vfmkleq_mvl(_vel_vcmpsl_vvvl(vrj_s1, _vel_vmulsl_vsvl(2, vrx_s12, vl), vl), vl) ;
      __vm256 vmx0_s2 =  _vel_vfmkleq_mvl(_vel_vcmpsl_vvvl(vrj_s2, _vel_vmulsl_vsvl(2, vrx_s12, vl), vl), vl) ;
      __vm256 vmx1_s12 =  _vel_vfmklge_mvl(vrx_s12, vl) ;
      __vm256 vmx2_s12 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutWidth,vrx_s12, vl), vl) ;
      __vm256 vmx_s12 = _vel_andm_mmm(vmx1_s12, vmx2_s12) ;
      __vm256 vmx_s1 = _vel_andm_mmm(vmx0_s1, vmx_s12) ;
      __vm256 vmx_s2 = _vel_andm_mmm(vmx0_s2, vmx_s12) ;

      __vm256 vmx0_s3 =  _vel_vfmkleq_mvl(_vel_vcmpsl_vvvl(vrj_s3, _vel_vmulsl_vsvl(2, vrx_s34, vl), vl), vl) ;
      __vm256 vmx0_s4 =  _vel_vfmkleq_mvl(_vel_vcmpsl_vvvl(vrj_s4, _vel_vmulsl_vsvl(2, vrx_s34, vl), vl), vl) ;
      __vm256 vmx1_s34 =  _vel_vfmklge_mvl(vrx_s34, vl) ;
      __vm256 vmx2_s34 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutWidth,vrx_s34, vl), vl) ;
      __vm256 vmx_s34 = _vel_andm_mmm(vmx1_s34, vmx2_s34) ;
      __vm256 vmx_s3 = _vel_andm_mmm(vmx0_s3, vmx_s34) ;
      __vm256 vmx_s4 = _vel_andm_mmm(vmx0_s4, vmx_s34) ;

      __vr vrsum0_s0  = _vel_vbrds_vsl(0.f, vl/2) ;
      __vr vrsum12_s0 = _vel_pvbrd_vsl(0UL, vl/2) ;
      __vr vrsum34_s0 = _vel_pvbrd_vsl(0UL, vl/2) ;
      __vr vrsum56_s0 = _vel_pvbrd_vsl(0UL, vl/2) ;
      __vr vrsum78_s0 = _vel_pvbrd_vsl(0UL, vl/2) ;
      __vr vrsum9A_s0 = _vel_pvbrd_vsl(0UL, vl/2) ;
      __vr vrsumBC_s0 = _vel_pvbrd_vsl(0UL, vl/2) ;
      __vr vrsumDE_s0 = _vel_pvbrd_vsl(0UL, vl/2) ;

      __vr vrsum0_s1  = _vel_vbrds_vsl(0.f, vl/2) ;
      __vr vrsum12_s1 = _vel_pvbrd_vsl(0UL, vl/2) ;
      __vr vrsum34_s1 = _vel_pvbrd_vsl(0UL, vl/2) ;
      __vr vrsum56_s1 = _vel_pvbrd_vsl(0UL, vl/2) ;
      __vr vrsum78_s1 = _vel_pvbrd_vsl(0UL, vl/2) ;
      __vr vrsum9A_s1 = _vel_pvbrd_vsl(0UL, vl/2) ;
      __vr vrsumBC_s1 = _vel_pvbrd_vsl(0UL, vl/2) ;
      __vr vrsumDE_s1 = _vel_pvbrd_vsl(0UL, vl/2) ;

      __vr vrsum0_s2  = _vel_vbrds_vsl(0.f, vl/2) ;
      __vr vrsum12_s2 = _vel_pvbrd_vsl(0UL, vl/2) ;
      __vr vrsum34_s2 = _vel_pvbrd_vsl(0UL, vl/2) ;
      __vr vrsum56_s2 = _vel_pvbrd_vsl(0UL, vl/2) ;
      __vr vrsum78_s2 = _vel_pvbrd_vsl(0UL, vl/2) ;
      __vr vrsum9A_s2 = _vel_pvbrd_vsl(0UL, vl/2) ;
      __vr vrsumBC_s2 = _vel_pvbrd_vsl(0UL, vl/2) ;
      __vr vrsumDE_s2 = _vel_pvbrd_vsl(0UL, vl/2) ;

      __vr vrsum0_s3  = _vel_vbrds_vsl(0.f, vl/2) ;
      __vr vrsum12_s3 = _vel_pvbrd_vsl(0UL, vl/2) ;
      __vr vrsum34_s3 = _vel_pvbrd_vsl(0UL, vl/2) ;
      __vr vrsum56_s3 = _vel_pvbrd_vsl(0UL, vl/2) ;
      __vr vrsum78_s3 = _vel_pvbrd_vsl(0UL, vl/2) ;
      __vr vrsum9A_s3 = _vel_pvbrd_vsl(0UL, vl/2) ;
      __vr vrsumBC_s3 = _vel_pvbrd_vsl(0UL, vl/2) ;
      __vr vrsumDE_s3 = _vel_pvbrd_vsl(0UL, vl/2) ;

      __vr vrsum0_s4  = _vel_vbrds_vsl(0.f, vl/2) ;
      __vr vrsum12_s4 = _vel_pvbrd_vsl(0UL, vl/2) ;
      __vr vrsum34_s4 = _vel_pvbrd_vsl(0UL, vl/2) ;
      __vr vrsum56_s4 = _vel_pvbrd_vsl(0UL, vl/2) ;
      __vr vrsum78_s4 = _vel_pvbrd_vsl(0UL, vl/2) ;
      __vr vrsum9A_s4 = _vel_pvbrd_vsl(0UL, vl/2) ;
      __vr vrsumBC_s4 = _vel_pvbrd_vsl(0UL, vl/2) ;
      __vr vrsumDE_s4 = _vel_pvbrd_vsl(0UL, vl/2) ;

      for (int64_t r=0; r<kernHeight; r++) {
	int64_t i = h - r + 2 ;
	int64_t y = i/2;
	if ( y*2 != i || y < 0 || gOutHeight <= y)  continue ;

	for (int64_t k=0; k<gOutChannelGroup; k++) {
	  int64_t gOutIndex    = gOutGroupOffset + ((n * gOutChannel + k) * gOutHeight) * gOutWidth ;

	  __vr vrgout_s01 = _vel_vldu_vssl(4, pGOut+gOutIndex+gOutWidth*y + xmin_s0 , vl/2) ;
	  __vr vrgout_s23 = _vel_vldu_vssl(4, pGOut+gOutIndex+gOutWidth*y + xmin_s2 , vl/2) ;
	  __vr vrgout_s4  = _vel_vldu_vssl(4, pGOut+gOutIndex+gOutWidth*y + xmin_s4 , vl/2) ;

#define FILTER_OFFSET(k,c,r,s) ( kernGroupOffset + filter_index<FLAYOUT>(k,c,r,s, gInChannelGroup, gOutChannelGroup, kernHeight, kernWidth) )

#define VFADD(VRGOUT,K,R,S) {									\
	    const float    kerValue0  = pKernel[FILTER_OFFSET(K,c+ 0,R,S)] ;			\
	    const uint64_t kerValue12 = _vel_pack_f32p(pKernel + FILTER_OFFSET(K,c+ 1,R,S),	\
						       pKernel + FILTER_OFFSET(K,c+ 2,R,S)) ;	\
	    const uint64_t kerValue34 = _vel_pack_f32p(pKernel + FILTER_OFFSET(K,c+ 3,R,S),	\
						       pKernel + FILTER_OFFSET(K,c+ 4,R,S)) ;	\
	    const uint64_t kerValue56 = _vel_pack_f32p(pKernel + FILTER_OFFSET(K,c+ 5,R,S),	\
						       pKernel + FILTER_OFFSET(K,c+ 6,R,S)) ;	\
	    const uint64_t kerValue78 = _vel_pack_f32p(pKernel + FILTER_OFFSET(K,c+ 7,R,S),	\
						       pKernel + FILTER_OFFSET(K,c+ 8,R,S)) ;	\
	    const uint64_t kerValue9A = _vel_pack_f32p(pKernel + FILTER_OFFSET(K,c+ 9,R,S),	\
						       pKernel + FILTER_OFFSET(K,c+10,R,S)) ;	\
	    const uint64_t kerValueBC = _vel_pack_f32p(pKernel + FILTER_OFFSET(K,c+11,R,S),	\
						       pKernel + FILTER_OFFSET(K,c+12,R,S)) ;	\
	    const uint64_t kerValueDE = _vel_pack_f32p(pKernel + FILTER_OFFSET(K,c+13,R,S),	\
						       pKernel + FILTER_OFFSET(K,c+14,R,S)) ;	\
	    __vr vrgoutP = _vel_vshf_vvvsl(VRGOUT, VRGOUT, VE_VSHUFFLE_YUZU, vl/2) ;		\
	    vrsum0_s##S = _vel_vfmads_vvsvl(vrsum0_s##S, kerValue0, VRGOUT, vl) ;				\
	    if(NUMCHANNEL>= 3) vrsum12_s##S = _vel_pvfmad_vvsvl(vrsum12_s##S, kerValue12, vrgoutP, vl) ;	\
	    if(NUMCHANNEL>= 5) vrsum34_s##S = _vel_pvfmad_vvsvl(vrsum34_s##S, kerValue34, vrgoutP, vl) ;	\
	    if(NUMCHANNEL>= 7) vrsum56_s##S = _vel_pvfmad_vvsvl(vrsum56_s##S, kerValue56, vrgoutP, vl) ;	\
	    if(NUMCHANNEL>= 9) vrsum78_s##S = _vel_pvfmad_vvsvl(vrsum78_s##S, kerValue78, vrgoutP, vl) ;	\
	    if(NUMCHANNEL>=11) vrsum9A_s##S = _vel_pvfmad_vvsvl(vrsum9A_s##S, kerValue9A, vrgoutP, vl) ;	\
	    if(NUMCHANNEL>=13) vrsumBC_s##S = _vel_pvfmad_vvsvl(vrsumBC_s##S, kerValueBC, vrgoutP, vl) ;	\
	    if(NUMCHANNEL>=15) vrsumDE_s##S = _vel_pvfmad_vvsvl(vrsumDE_s##S, kerValueDE, vrgoutP, vl) ;	\
	  }

	  VFADD(vrgout_s01, k+0, r, 0) ;
	  VFADD(vrgout_s01, k+0, r, 1) ;
	  VFADD(vrgout_s23, k+0, r, 2) ;
	  VFADD(vrgout_s23, k+0, r, 3) ;
	  VFADD(vrgout_s4,  k+0, r, 4) ;

#undef VFADD
#undef FILTER_OFFSET
	}

      } // kernHeight


      {
	vrsum0_s0 = _vel_vex_vvmvl(vrsum0_s0, vmx_s0, _vel_pvbrd_vsl(0UL, vl), vl) ;
	vrsum0_s1 = _vel_vex_vvmvl(vrsum0_s1, vmx_s1, _vel_pvbrd_vsl(0UL, vl), vl) ;
	vrsum0_s2 = _vel_vex_vvmvl(vrsum0_s2, vmx_s2, _vel_pvbrd_vsl(0UL, vl), vl) ;
	vrsum0_s3 = _vel_vex_vvmvl(vrsum0_s3, vmx_s3, _vel_pvbrd_vsl(0UL, vl), vl) ;
	vrsum0_s4 = _vel_vex_vvmvl(vrsum0_s4, vmx_s4, _vel_pvbrd_vsl(0UL, vl), vl) ;
	__vr vrsum0 = _vel_vfadds_vvvl(_vel_vfadds_vvvl(vrsum0_s0, _vel_pvfadd_vvvl(vrsum0_s1, vrsum0_s2, vl), vl),
				       _vel_vfadds_vvvl(vrsum0_s3, vrsum0_s4, vl), vl) ;
  	_vel_vstu_vssl(vrsum0, 4, pGIn+gInIndex + 0 * gInHeight * gInWidth, vl) ;
      }

      if(NUMCHANNEL>= 3) {
	vrsum12_s0 = _vel_vex_vvmvl(vrsum12_s0, vmx_s0, _vel_pvbrd_vsl(0UL, vl), vl) ;
	vrsum12_s1 = _vel_vex_vvmvl(vrsum12_s1, vmx_s1, _vel_pvbrd_vsl(0UL, vl), vl) ;
	vrsum12_s2 = _vel_vex_vvmvl(vrsum12_s2, vmx_s2, _vel_pvbrd_vsl(0UL, vl), vl) ;
	vrsum12_s3 = _vel_vex_vvmvl(vrsum12_s3, vmx_s3, _vel_pvbrd_vsl(0UL, vl), vl) ;
	vrsum12_s4 = _vel_vex_vvmvl(vrsum12_s4, vmx_s4, _vel_pvbrd_vsl(0UL, vl), vl) ;
	__vr vrsum12 = _vel_pvfadd_vvvl(_vel_pvfadd_vvvl(vrsum12_s0, _vel_pvfadd_vvvl(vrsum12_s1, vrsum12_s2, vl), vl),
					_vel_pvfadd_vvvl(vrsum12_s3, vrsum12_s4, vl), vl) ;
  	_vel_vstu_vssl(vrsum12, 4, pGIn+gInIndex + 1 * gInHeight * gInWidth, vl) ;
  	_vel_vstl_vssl(vrsum12, 4, pGIn+gInIndex + 2 * gInHeight * gInWidth, vl) ;
      }
      if(NUMCHANNEL>= 5) {
	vrsum34_s0 = _vel_vex_vvmvl(vrsum34_s0, vmx_s0, _vel_pvbrd_vsl(0UL, vl), vl) ;
	vrsum34_s1 = _vel_vex_vvmvl(vrsum34_s1, vmx_s1, _vel_pvbrd_vsl(0UL, vl), vl) ;
	vrsum34_s2 = _vel_vex_vvmvl(vrsum34_s2, vmx_s2, _vel_pvbrd_vsl(0UL, vl), vl) ;
	vrsum34_s3 = _vel_vex_vvmvl(vrsum34_s3, vmx_s3, _vel_pvbrd_vsl(0UL, vl), vl) ;
	vrsum34_s4 = _vel_vex_vvmvl(vrsum34_s4, vmx_s4, _vel_pvbrd_vsl(0UL, vl), vl) ;
	__vr vrsum34 = _vel_pvfadd_vvvl(_vel_pvfadd_vvvl(vrsum34_s0, _vel_pvfadd_vvvl(vrsum34_s1, vrsum34_s2, vl), vl),
					_vel_pvfadd_vvvl(vrsum34_s3, vrsum34_s4, vl), vl) ;
  	_vel_vstu_vssl(vrsum34, 4, pGIn+gInIndex + 3 * gInHeight * gInWidth, vl) ;
  	_vel_vstl_vssl(vrsum34, 4, pGIn+gInIndex + 4 * gInHeight * gInWidth, vl) ;
      }
      if(NUMCHANNEL>= 7) {
	vrsum56_s0 = _vel_vex_vvmvl(vrsum56_s0, vmx_s0, _vel_pvbrd_vsl(0UL, vl), vl) ;
	vrsum56_s1 = _vel_vex_vvmvl(vrsum56_s1, vmx_s1, _vel_pvbrd_vsl(0UL, vl), vl) ;
	vrsum56_s2 = _vel_vex_vvmvl(vrsum56_s2, vmx_s2, _vel_pvbrd_vsl(0UL, vl), vl) ;
	vrsum56_s3 = _vel_vex_vvmvl(vrsum56_s3, vmx_s3, _vel_pvbrd_vsl(0UL, vl), vl) ;
	vrsum56_s4 = _vel_vex_vvmvl(vrsum56_s4, vmx_s4, _vel_pvbrd_vsl(0UL, vl), vl) ;
	__vr vrsum56 = _vel_pvfadd_vvvl(_vel_pvfadd_vvvl(vrsum56_s0, _vel_pvfadd_vvvl(vrsum56_s1, vrsum56_s2, vl), vl),
					_vel_pvfadd_vvvl(vrsum56_s3, vrsum56_s4, vl), vl) ;
  	_vel_vstu_vssl(vrsum56, 4, pGIn+gInIndex + 5 * gInHeight * gInWidth, vl) ;
  	_vel_vstl_vssl(vrsum56, 4, pGIn+gInIndex + 6 * gInHeight * gInWidth, vl) ;
      }
      if(NUMCHANNEL>= 9) {
	vrsum78_s0 = _vel_vex_vvmvl(vrsum78_s0, vmx_s0, _vel_pvbrd_vsl(0UL, vl), vl) ;
	vrsum78_s1 = _vel_vex_vvmvl(vrsum78_s1, vmx_s1, _vel_pvbrd_vsl(0UL, vl), vl) ;
	vrsum78_s2 = _vel_vex_vvmvl(vrsum78_s2, vmx_s2, _vel_pvbrd_vsl(0UL, vl), vl) ;
	vrsum78_s3 = _vel_vex_vvmvl(vrsum78_s3, vmx_s3, _vel_pvbrd_vsl(0UL, vl), vl) ;
	vrsum78_s4 = _vel_vex_vvmvl(vrsum78_s4, vmx_s4, _vel_pvbrd_vsl(0UL, vl), vl) ;
	__vr vrsum78 = _vel_pvfadd_vvvl(_vel_pvfadd_vvvl(vrsum78_s0, _vel_pvfadd_vvvl(vrsum78_s1, vrsum78_s2, vl), vl),
					_vel_pvfadd_vvvl(vrsum78_s3, vrsum78_s4, vl), vl) ;
  	_vel_vstu_vssl(vrsum78, 4, pGIn+gInIndex + 7 * gInHeight * gInWidth, vl) ;
  	_vel_vstl_vssl(vrsum78, 4, pGIn+gInIndex + 8 * gInHeight * gInWidth, vl) ;
      }
      if(NUMCHANNEL>=11) {
	vrsum9A_s0 = _vel_vex_vvmvl(vrsum9A_s0, vmx_s0, _vel_pvbrd_vsl(0UL, vl), vl) ;
	vrsum9A_s1 = _vel_vex_vvmvl(vrsum9A_s1, vmx_s1, _vel_pvbrd_vsl(0UL, vl), vl) ;
	vrsum9A_s2 = _vel_vex_vvmvl(vrsum9A_s2, vmx_s2, _vel_pvbrd_vsl(0UL, vl), vl) ;
	vrsum9A_s3 = _vel_vex_vvmvl(vrsum9A_s3, vmx_s3, _vel_pvbrd_vsl(0UL, vl), vl) ;
	vrsum9A_s4 = _vel_vex_vvmvl(vrsum9A_s4, vmx_s4, _vel_pvbrd_vsl(0UL, vl), vl) ;
	__vr vrsum9A = _vel_pvfadd_vvvl(_vel_pvfadd_vvvl(vrsum9A_s0, _vel_pvfadd_vvvl(vrsum9A_s1, vrsum9A_s2, vl), vl),
					_vel_pvfadd_vvvl(vrsum9A_s3, vrsum9A_s4, vl), vl) ;
  	_vel_vstu_vssl(vrsum9A, 4, pGIn+gInIndex + 9 * gInHeight * gInWidth, vl) ;
  	_vel_vstl_vssl(vrsum9A, 4, pGIn+gInIndex +10 * gInHeight * gInWidth, vl) ;
      }
      if(NUMCHANNEL>=13) {
	vrsumBC_s0 = _vel_vex_vvmvl(vrsumBC_s0, vmx_s0, _vel_pvbrd_vsl(0UL, vl), vl) ;
	vrsumBC_s1 = _vel_vex_vvmvl(vrsumBC_s1, vmx_s1, _vel_pvbrd_vsl(0UL, vl), vl) ;
	vrsumBC_s2 = _vel_vex_vvmvl(vrsumBC_s2, vmx_s2, _vel_pvbrd_vsl(0UL, vl), vl) ;
	vrsumBC_s3 = _vel_vex_vvmvl(vrsumBC_s3, vmx_s3, _vel_pvbrd_vsl(0UL, vl), vl) ;
	vrsumBC_s4 = _vel_vex_vvmvl(vrsumBC_s4, vmx_s4, _vel_pvbrd_vsl(0UL, vl), vl) ;
	__vr vrsumBC = _vel_pvfadd_vvvl(_vel_pvfadd_vvvl(vrsumBC_s0, _vel_pvfadd_vvvl(vrsumBC_s1, vrsumBC_s2, vl), vl),
					_vel_pvfadd_vvvl(vrsumBC_s3, vrsumBC_s4, vl), vl) ;
  	_vel_vstu_vssl(vrsumBC, 4, pGIn+gInIndex +11 * gInHeight * gInWidth, vl) ;
  	_vel_vstl_vssl(vrsumBC, 4, pGIn+gInIndex +12 * gInHeight * gInWidth, vl) ;
      }
      if(NUMCHANNEL>=15) {
	vrsumDE_s0 = _vel_vex_vvmvl(vrsumDE_s0, vmx_s0, _vel_pvbrd_vsl(0UL, vl), vl) ;
	vrsumDE_s1 = _vel_vex_vvmvl(vrsumDE_s1, vmx_s1, _vel_pvbrd_vsl(0UL, vl), vl) ;
	vrsumDE_s2 = _vel_vex_vvmvl(vrsumDE_s2, vmx_s2, _vel_pvbrd_vsl(0UL, vl), vl) ;
	vrsumDE_s3 = _vel_vex_vvmvl(vrsumDE_s3, vmx_s3, _vel_pvbrd_vsl(0UL, vl), vl) ;
	vrsumDE_s4 = _vel_vex_vvmvl(vrsumDE_s4, vmx_s4, _vel_pvbrd_vsl(0UL, vl), vl) ;
	__vr vrsumDE = _vel_pvfadd_vvvl(_vel_pvfadd_vvvl(vrsumDE_s0, _vel_pvfadd_vvvl(vrsumDE_s1, vrsumDE_s2, vl), vl),
					_vel_pvfadd_vvvl(vrsumDE_s3, vrsumDE_s4, vl), vl) ;
  	_vel_vstu_vssl(vrsumDE, 4, pGIn+gInIndex +13 * gInHeight * gInWidth, vl) ;
  	_vel_vstl_vssl(vrsumDE, 4, pGIn+gInIndex +14 * gInHeight * gInWidth, vl) ;
      }

      gInIndex += vl ;
    } // gInWidth
  } // gInHeight
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
    const int64_t c
)
{
  int64_t gInIndex = gInGroupOffset + ((n * gInChannel + c) * gInHeight) * gInWidth  ;

  for (int64_t h = 0; h < gInHeight ; h++ ) {
    for (int64_t w = 0; w < gInWidth ; w += VLEN ) {
      const int64_t vl = gInWidth - w < VLEN ? gInWidth - w  : VLEN ;

      const int64_t xmin_s0 = (w+2) / 2 ;
      const int64_t xmin_s2 = (w+0) / 2 ;
      const int64_t xmin_s4 = w >= 2 ? (w-2) / 2 : 0 ;

      __vr vrw   = _vel_vaddsl_vsvl(w, _vel_vseq_vl(vl), vl) ;

      __vr vrj_s0 = _vel_vaddsl_vsvl(2, vrw, vl) ;
      __vr vrj_s1 = _vel_vaddsl_vsvl(1, vrw, vl) ;
      __vr vrj_s2 = vrw ;
      __vr vrj_s3 = _vel_vaddsl_vsvl(-1, vrw, vl) ;
      __vr vrj_s4 = _vel_vaddsl_vsvl(-2, vrw, vl) ;

      __vr vrx_s0 = _vel_vdivsl_vvsl(vrj_s0, 2, vl) ;
      __vr vrx_s12 = _vel_vdivsl_vvsl(vrj_s1, 2, vl) ;
      __vr vrx_s34 = _vel_vdivsl_vvsl(vrj_s3, 2, vl) ;

      __vm256 vmx0_s0 =  _vel_vfmkleq_mvl(_vel_vcmpsl_vvvl(vrj_s0, _vel_vmulsl_vsvl(2, vrx_s0, vl), vl), vl) ;
      __vm256 vmx1_s0 =  _vel_vfmklge_mvl(vrx_s0, vl) ;
      __vm256 vmx2_s0 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutWidth,vrx_s0, vl), vl) ;
      __vm256 vmx_s0 = _vel_andm_mmm(_vel_andm_mmm(vmx0_s0, vmx1_s0), vmx2_s0) ;

      __vm256 vmx0_s1 =  _vel_vfmkleq_mvl(_vel_vcmpsl_vvvl(vrj_s1, _vel_vmulsl_vsvl(2, vrx_s12, vl), vl), vl) ;
      __vm256 vmx0_s2 =  _vel_vfmkleq_mvl(_vel_vcmpsl_vvvl(vrj_s2, _vel_vmulsl_vsvl(2, vrx_s12, vl), vl), vl) ;
      __vm256 vmx1_s12 =  _vel_vfmklge_mvl(vrx_s12, vl) ;
      __vm256 vmx2_s12 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutWidth,vrx_s12, vl), vl) ;
      __vm256 vmx_s12 = _vel_andm_mmm(vmx1_s12, vmx2_s12) ;
      __vm256 vmx_s1 = _vel_andm_mmm(vmx0_s1, vmx_s12) ;
      __vm256 vmx_s2 = _vel_andm_mmm(vmx0_s2, vmx_s12) ;

      __vm256 vmx0_s3 =  _vel_vfmkleq_mvl(_vel_vcmpsl_vvvl(vrj_s3, _vel_vmulsl_vsvl(2, vrx_s34, vl), vl), vl) ;
      __vm256 vmx0_s4 =  _vel_vfmkleq_mvl(_vel_vcmpsl_vvvl(vrj_s4, _vel_vmulsl_vsvl(2, vrx_s34, vl), vl), vl) ;
      __vm256 vmx1_s34 =  _vel_vfmklge_mvl(vrx_s34, vl) ;
      __vm256 vmx2_s34 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutWidth,vrx_s34, vl), vl) ;
      __vm256 vmx_s34 = _vel_andm_mmm(vmx1_s34, vmx2_s34) ;
      __vm256 vmx_s3 = _vel_andm_mmm(vmx0_s3, vmx_s34) ;
      __vm256 vmx_s4 = _vel_andm_mmm(vmx0_s4, vmx_s34) ;

      __vr vrsum01_s0 = _vel_pvbrd_vsl(0UL, vl/2) ;
      __vr vrsum23_s0 = _vel_pvbrd_vsl(0UL, vl/2) ;
      __vr vrsum45_s0 = _vel_pvbrd_vsl(0UL, vl/2) ;
      __vr vrsum67_s0 = _vel_pvbrd_vsl(0UL, vl/2) ;
      __vr vrsum89_s0 = _vel_pvbrd_vsl(0UL, vl/2) ;
      __vr vrsumAB_s0 = _vel_pvbrd_vsl(0UL, vl/2) ;
      __vr vrsumCD_s0 = _vel_pvbrd_vsl(0UL, vl/2) ;
      __vr vrsumEF_s0 = _vel_pvbrd_vsl(0UL, vl/2) ;

      __vr vrsum01_s1 = _vel_pvbrd_vsl(0UL, vl/2) ;
      __vr vrsum23_s1 = _vel_pvbrd_vsl(0UL, vl/2) ;
      __vr vrsum45_s1 = _vel_pvbrd_vsl(0UL, vl/2) ;
      __vr vrsum67_s1 = _vel_pvbrd_vsl(0UL, vl/2) ;
      __vr vrsum89_s1 = _vel_pvbrd_vsl(0UL, vl/2) ;
      __vr vrsumAB_s1 = _vel_pvbrd_vsl(0UL, vl/2) ;
      __vr vrsumCD_s1 = _vel_pvbrd_vsl(0UL, vl/2) ;
      __vr vrsumEF_s1 = _vel_pvbrd_vsl(0UL, vl/2) ;

      __vr vrsum01_s2 = _vel_pvbrd_vsl(0UL, vl/2) ;
      __vr vrsum23_s2 = _vel_pvbrd_vsl(0UL, vl/2) ;
      __vr vrsum45_s2 = _vel_pvbrd_vsl(0UL, vl/2) ;
      __vr vrsum67_s2 = _vel_pvbrd_vsl(0UL, vl/2) ;
      __vr vrsum89_s2 = _vel_pvbrd_vsl(0UL, vl/2) ;
      __vr vrsumAB_s2 = _vel_pvbrd_vsl(0UL, vl/2) ;
      __vr vrsumCD_s2 = _vel_pvbrd_vsl(0UL, vl/2) ;
      __vr vrsumEF_s2 = _vel_pvbrd_vsl(0UL, vl/2) ;

      __vr vrsum01_s3 = _vel_pvbrd_vsl(0UL, vl/2) ;
      __vr vrsum23_s3 = _vel_pvbrd_vsl(0UL, vl/2) ;
      __vr vrsum45_s3 = _vel_pvbrd_vsl(0UL, vl/2) ;
      __vr vrsum67_s3 = _vel_pvbrd_vsl(0UL, vl/2) ;
      __vr vrsum89_s3 = _vel_pvbrd_vsl(0UL, vl/2) ;
      __vr vrsumAB_s3 = _vel_pvbrd_vsl(0UL, vl/2) ;
      __vr vrsumCD_s3 = _vel_pvbrd_vsl(0UL, vl/2) ;
      __vr vrsumEF_s3 = _vel_pvbrd_vsl(0UL, vl/2) ;

      __vr vrsum01_s4 = _vel_pvbrd_vsl(0UL, vl/2) ;
      __vr vrsum23_s4 = _vel_pvbrd_vsl(0UL, vl/2) ;
      __vr vrsum45_s4 = _vel_pvbrd_vsl(0UL, vl/2) ;
      __vr vrsum67_s4 = _vel_pvbrd_vsl(0UL, vl/2) ;
      __vr vrsum89_s4 = _vel_pvbrd_vsl(0UL, vl/2) ;
      __vr vrsumAB_s4 = _vel_pvbrd_vsl(0UL, vl/2) ;
      __vr vrsumCD_s4 = _vel_pvbrd_vsl(0UL, vl/2) ;
      __vr vrsumEF_s4 = _vel_pvbrd_vsl(0UL, vl/2) ;

      for (int64_t r=0; r<kernHeight; r++) {
	int64_t i = h - r + 2 ;
	int64_t y = i/2;
	if ( y*2 != i || y < 0 || gOutHeight <= y)  continue ;

	for (int64_t k=0; k<gOutChannelGroup; k++) {
	  int64_t gOutIndex    = gOutGroupOffset + ((n * gOutChannel + k) * gOutHeight) * gOutWidth ;
	  const float *pKerValue = pKernel + kernGroupOffset + ((k * gInChannelGroup + c) * kernHeight) * kernWidth;

	  __vr vrgout_s01 = _vel_vldu_vssl(4, pGOut+gOutIndex+gOutWidth*y + xmin_s0 , vl/2) ;
	  __vr vrgout_s23 = _vel_vldu_vssl(4, pGOut+gOutIndex+gOutWidth*y + xmin_s2 , vl/2) ;
	  __vr vrgout_s4  = _vel_vldu_vssl(4, pGOut+gOutIndex+gOutWidth*y + xmin_s4 , vl/2) ;

#define FILTER_OFFSET(k,c,r,s) ( kernGroupOffset + filter_index<FLAYOUT>(k,c,r,s, gInChannelGroup, gOutChannelGroup, kernHeight, kernWidth) )

#define VFADD(VRGOUT,K,R,S) {									\
	    const uint64_t kerValue01 = _vel_pack_f32p(pKernel + FILTER_OFFSET(K,c+ 0,R,S),	\
						       pKernel + FILTER_OFFSET(K,c+ 1,R,S)) ;	\
	    const uint64_t kerValue23 = _vel_pack_f32p(pKernel + FILTER_OFFSET(K,c+ 2,R,S),	\
						       pKernel + FILTER_OFFSET(K,c+ 3,R,S)) ;	\
	    const uint64_t kerValue45 = _vel_pack_f32p(pKernel + FILTER_OFFSET(K,c+ 4,R,S),	\
						       pKernel + FILTER_OFFSET(K,c+ 5,R,S)) ;	\
	    const uint64_t kerValue67 = _vel_pack_f32p(pKernel + FILTER_OFFSET(K,c+ 6,R,S),	\
						       pKernel + FILTER_OFFSET(K,c+ 7,R,S)) ;	\
	    const uint64_t kerValue89 = _vel_pack_f32p(pKernel + FILTER_OFFSET(K,c+ 8,R,S),	\
						       pKernel + FILTER_OFFSET(K,c+ 9,R,S)) ;	\
	    const uint64_t kerValueAB = _vel_pack_f32p(pKernel + FILTER_OFFSET(K,c+10,R,S),	\
						       pKernel + FILTER_OFFSET(K,c+11,R,S)) ;	\
	    const uint64_t kerValueCD = _vel_pack_f32p(pKernel + FILTER_OFFSET(K,c+12,R,S),	\
						       pKernel + FILTER_OFFSET(K,c+13,R,S)) ;	\
	    const uint64_t kerValueEF = _vel_pack_f32p(pKernel + FILTER_OFFSET(K,c+14,R,S),	\
						       pKernel + FILTER_OFFSET(K,c+15,R,S)) ;	\
	    __vr vrgoutP = _vel_vshf_vvvsl(VRGOUT, VRGOUT, VE_VSHUFFLE_YUZU, vl/2) ;		\
	    if(NUMCHANNEL>= 2) vrsum01_s##S = _vel_pvfmad_vvsvl(vrsum01_s##S, kerValue01, vrgoutP, vl/2) ;	\
	    if(NUMCHANNEL>= 4) vrsum23_s##S = _vel_pvfmad_vvsvl(vrsum23_s##S, kerValue23, vrgoutP, vl/2) ;	\
	    if(NUMCHANNEL>= 6) vrsum45_s##S = _vel_pvfmad_vvsvl(vrsum45_s##S, kerValue45, vrgoutP, vl/2) ;	\
	    if(NUMCHANNEL>= 8) vrsum67_s##S = _vel_pvfmad_vvsvl(vrsum67_s##S, kerValue67, vrgoutP, vl/2) ;	\
	    if(NUMCHANNEL>=10) vrsum89_s##S = _vel_pvfmad_vvsvl(vrsum89_s##S, kerValue89, vrgoutP, vl/2) ;	\
	    if(NUMCHANNEL>=12) vrsumAB_s##S = _vel_pvfmad_vvsvl(vrsumAB_s##S, kerValueAB, vrgoutP, vl/2) ;	\
	    if(NUMCHANNEL>=14) vrsumCD_s##S = _vel_pvfmad_vvsvl(vrsumCD_s##S, kerValueCD, vrgoutP, vl/2) ;	\
	    if(NUMCHANNEL>=16) vrsumEF_s##S = _vel_pvfmad_vvsvl(vrsumEF_s##S, kerValueEF, vrgoutP, vl/2) ;	\
	  }

	  VFADD(vrgout_s01, k+0, r, 0) ;
	  VFADD(vrgout_s01, k+0, r, 1) ;
	  VFADD(vrgout_s23, k+0, r, 2) ;
	  VFADD(vrgout_s23, k+0, r, 3) ;
	  VFADD(vrgout_s4,  k+0, r, 4) ;

#undef VFADD
#undef FILTER_OFFSET
	}

      } // kernHeight

      if(NUMCHANNEL>= 2) {
	vrsum01_s0 = _vel_vex_vvmvl(vrsum01_s0, vmx_s0, _vel_pvbrd_vsl(0UL, vl), vl) ;
	vrsum01_s1 = _vel_vex_vvmvl(vrsum01_s1, vmx_s1, _vel_pvbrd_vsl(0UL, vl), vl) ;
	vrsum01_s2 = _vel_vex_vvmvl(vrsum01_s2, vmx_s2, _vel_pvbrd_vsl(0UL, vl), vl) ;
	vrsum01_s3 = _vel_vex_vvmvl(vrsum01_s3, vmx_s3, _vel_pvbrd_vsl(0UL, vl), vl) ;
	vrsum01_s4 = _vel_vex_vvmvl(vrsum01_s4, vmx_s4, _vel_pvbrd_vsl(0UL, vl), vl) ;
	__vr vrsum01 = _vel_pvfadd_vvvl(_vel_pvfadd_vvvl(vrsum01_s0, _vel_pvfadd_vvvl(vrsum01_s1, vrsum01_s2, vl), vl),
					_vel_pvfadd_vvvl(vrsum01_s3, vrsum01_s4, vl), vl) ;
  	_vel_vstu_vssl(vrsum01, 4, pGIn+gInIndex + 0 * gInHeight * gInWidth, vl) ;
  	_vel_vstl_vssl(vrsum01, 4, pGIn+gInIndex + 1 * gInHeight * gInWidth, vl) ;
      }
      if(NUMCHANNEL>= 4) {
	vrsum23_s0 = _vel_vex_vvmvl(vrsum23_s0, vmx_s0, _vel_pvbrd_vsl(0UL, vl), vl) ;
	vrsum23_s1 = _vel_vex_vvmvl(vrsum23_s1, vmx_s1, _vel_pvbrd_vsl(0UL, vl), vl) ;
	vrsum23_s2 = _vel_vex_vvmvl(vrsum23_s2, vmx_s2, _vel_pvbrd_vsl(0UL, vl), vl) ;
	vrsum23_s3 = _vel_vex_vvmvl(vrsum23_s3, vmx_s3, _vel_pvbrd_vsl(0UL, vl), vl) ;
	vrsum23_s4 = _vel_vex_vvmvl(vrsum23_s4, vmx_s4, _vel_pvbrd_vsl(0UL, vl), vl) ;
	__vr vrsum23 = _vel_pvfadd_vvvl(_vel_pvfadd_vvvl(vrsum23_s0, _vel_pvfadd_vvvl(vrsum23_s1, vrsum23_s2, vl), vl),
					_vel_pvfadd_vvvl(vrsum23_s3, vrsum23_s4, vl), vl) ;
  	_vel_vstu_vssl(vrsum23, 4, pGIn+gInIndex + 2 * gInHeight * gInWidth, vl) ;
  	_vel_vstl_vssl(vrsum23, 4, pGIn+gInIndex + 3 * gInHeight * gInWidth, vl) ;
      }
      if(NUMCHANNEL>= 6) {
	vrsum45_s0 = _vel_vex_vvmvl(vrsum45_s0, vmx_s0, _vel_pvbrd_vsl(0UL, vl), vl) ;
	vrsum45_s1 = _vel_vex_vvmvl(vrsum45_s1, vmx_s1, _vel_pvbrd_vsl(0UL, vl), vl) ;
	vrsum45_s2 = _vel_vex_vvmvl(vrsum45_s2, vmx_s2, _vel_pvbrd_vsl(0UL, vl), vl) ;
	vrsum45_s3 = _vel_vex_vvmvl(vrsum45_s3, vmx_s3, _vel_pvbrd_vsl(0UL, vl), vl) ;
	vrsum45_s4 = _vel_vex_vvmvl(vrsum45_s4, vmx_s4, _vel_pvbrd_vsl(0UL, vl), vl) ;
	__vr vrsum45 = _vel_pvfadd_vvvl(_vel_pvfadd_vvvl(vrsum45_s0, _vel_pvfadd_vvvl(vrsum45_s1, vrsum45_s2, vl), vl),
					_vel_pvfadd_vvvl(vrsum45_s3, vrsum45_s4, vl), vl) ;
  	_vel_vstu_vssl(vrsum45, 4, pGIn+gInIndex + 4 * gInHeight * gInWidth, vl) ;
  	_vel_vstl_vssl(vrsum45, 4, pGIn+gInIndex + 5 * gInHeight * gInWidth, vl) ;
      }
      if(NUMCHANNEL>= 8) {
	vrsum67_s0 = _vel_vex_vvmvl(vrsum67_s0, vmx_s0, _vel_pvbrd_vsl(0UL, vl), vl) ;
	vrsum67_s1 = _vel_vex_vvmvl(vrsum67_s1, vmx_s1, _vel_pvbrd_vsl(0UL, vl), vl) ;
	vrsum67_s2 = _vel_vex_vvmvl(vrsum67_s2, vmx_s2, _vel_pvbrd_vsl(0UL, vl), vl) ;
	vrsum67_s3 = _vel_vex_vvmvl(vrsum67_s3, vmx_s3, _vel_pvbrd_vsl(0UL, vl), vl) ;
	vrsum67_s4 = _vel_vex_vvmvl(vrsum67_s4, vmx_s4, _vel_pvbrd_vsl(0UL, vl), vl) ;
	__vr vrsum67 = _vel_pvfadd_vvvl(_vel_pvfadd_vvvl(vrsum67_s0, _vel_pvfadd_vvvl(vrsum67_s1, vrsum67_s2, vl), vl),
					_vel_pvfadd_vvvl(vrsum67_s3, vrsum67_s4, vl), vl) ;
  	_vel_vstu_vssl(vrsum67, 4, pGIn+gInIndex + 6 * gInHeight * gInWidth, vl) ;
  	_vel_vstl_vssl(vrsum67, 4, pGIn+gInIndex + 7 * gInHeight * gInWidth, vl) ;
      }
      if(NUMCHANNEL>=10) {
	vrsum89_s0 = _vel_vex_vvmvl(vrsum89_s0, vmx_s0, _vel_pvbrd_vsl(0UL, vl), vl) ;
	vrsum89_s1 = _vel_vex_vvmvl(vrsum89_s1, vmx_s1, _vel_pvbrd_vsl(0UL, vl), vl) ;
	vrsum89_s2 = _vel_vex_vvmvl(vrsum89_s2, vmx_s2, _vel_pvbrd_vsl(0UL, vl), vl) ;
	vrsum89_s3 = _vel_vex_vvmvl(vrsum89_s3, vmx_s3, _vel_pvbrd_vsl(0UL, vl), vl) ;
	vrsum89_s4 = _vel_vex_vvmvl(vrsum89_s4, vmx_s4, _vel_pvbrd_vsl(0UL, vl), vl) ;
	__vr vrsum89 = _vel_pvfadd_vvvl(_vel_pvfadd_vvvl(vrsum89_s0, _vel_pvfadd_vvvl(vrsum89_s1, vrsum89_s2, vl), vl),
					_vel_pvfadd_vvvl(vrsum89_s3, vrsum89_s4, vl), vl) ;
  	_vel_vstu_vssl(vrsum89, 4, pGIn+gInIndex + 8 * gInHeight * gInWidth, vl) ;
  	_vel_vstl_vssl(vrsum89, 4, pGIn+gInIndex + 9 * gInHeight * gInWidth, vl) ;
      }
      if(NUMCHANNEL>=12) {
	vrsumAB_s0 = _vel_vex_vvmvl(vrsumAB_s0, vmx_s0, _vel_pvbrd_vsl(0UL, vl), vl) ;
	vrsumAB_s1 = _vel_vex_vvmvl(vrsumAB_s1, vmx_s1, _vel_pvbrd_vsl(0UL, vl), vl) ;
	vrsumAB_s2 = _vel_vex_vvmvl(vrsumAB_s2, vmx_s2, _vel_pvbrd_vsl(0UL, vl), vl) ;
	vrsumAB_s3 = _vel_vex_vvmvl(vrsumAB_s3, vmx_s3, _vel_pvbrd_vsl(0UL, vl), vl) ;
	vrsumAB_s4 = _vel_vex_vvmvl(vrsumAB_s4, vmx_s4, _vel_pvbrd_vsl(0UL, vl), vl) ;
	__vr vrsumAB = _vel_pvfadd_vvvl(_vel_pvfadd_vvvl(vrsumAB_s0, _vel_pvfadd_vvvl(vrsumAB_s1, vrsumAB_s2, vl), vl),
					_vel_pvfadd_vvvl(vrsumAB_s3, vrsumAB_s4, vl), vl) ;
  	_vel_vstu_vssl(vrsumAB, 4, pGIn+gInIndex +10 * gInHeight * gInWidth, vl) ;
  	_vel_vstl_vssl(vrsumAB, 4, pGIn+gInIndex +11 * gInHeight * gInWidth, vl) ;
      }
      if(NUMCHANNEL>=14) {
	vrsumCD_s0 = _vel_vex_vvmvl(vrsumCD_s0, vmx_s0, _vel_pvbrd_vsl(0UL, vl), vl) ;
	vrsumCD_s1 = _vel_vex_vvmvl(vrsumCD_s1, vmx_s1, _vel_pvbrd_vsl(0UL, vl), vl) ;
	vrsumCD_s2 = _vel_vex_vvmvl(vrsumCD_s2, vmx_s2, _vel_pvbrd_vsl(0UL, vl), vl) ;
	vrsumCD_s3 = _vel_vex_vvmvl(vrsumCD_s3, vmx_s3, _vel_pvbrd_vsl(0UL, vl), vl) ;
	vrsumCD_s4 = _vel_vex_vvmvl(vrsumCD_s4, vmx_s4, _vel_pvbrd_vsl(0UL, vl), vl) ;
	__vr vrsumCD = _vel_pvfadd_vvvl(_vel_pvfadd_vvvl(vrsumCD_s0, _vel_pvfadd_vvvl(vrsumCD_s1, vrsumCD_s2, vl), vl),
					_vel_pvfadd_vvvl(vrsumCD_s3, vrsumCD_s4, vl), vl) ;
  	_vel_vstu_vssl(vrsumCD, 4, pGIn+gInIndex +12 * gInHeight * gInWidth, vl) ;
  	_vel_vstl_vssl(vrsumCD, 4, pGIn+gInIndex +13 * gInHeight * gInWidth, vl) ;
      }
      if(NUMCHANNEL>=16) {
	vrsumEF_s0 = _vel_vex_vvmvl(vrsumEF_s0, vmx_s0, _vel_pvbrd_vsl(0UL, vl), vl) ;
	vrsumEF_s1 = _vel_vex_vvmvl(vrsumEF_s1, vmx_s1, _vel_pvbrd_vsl(0UL, vl), vl) ;
	vrsumEF_s2 = _vel_vex_vvmvl(vrsumEF_s2, vmx_s2, _vel_pvbrd_vsl(0UL, vl), vl) ;
	vrsumEF_s3 = _vel_vex_vvmvl(vrsumEF_s3, vmx_s3, _vel_pvbrd_vsl(0UL, vl), vl) ;
	vrsumEF_s4 = _vel_vex_vvmvl(vrsumEF_s4, vmx_s4, _vel_pvbrd_vsl(0UL, vl), vl) ;
	__vr vrsumEF = _vel_pvfadd_vvvl(_vel_pvfadd_vvvl(vrsumEF_s0, _vel_pvfadd_vvvl(vrsumEF_s1, vrsumEF_s2, vl), vl),
					_vel_pvfadd_vvvl(vrsumEF_s3, vrsumEF_s4, vl), vl) ;
  	_vel_vstu_vssl(vrsumEF, 4, pGIn+gInIndex +14 * gInHeight * gInWidth, vl) ;
  	_vel_vstl_vssl(vrsumEF, 4, pGIn+gInIndex +15 * gInHeight * gInWidth, vl) ;
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
    const int64_t gOutChannelGroup
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
	func_odd<FLAYOUT, 1>(pGOut, pKernel, pGIn,
	   gOutChannel, gOutWidth, gOutHeight,
	   gInChannel, gInWidth, gInHeight,
	   kernWidth, kernHeight,
	   gInChannelGroup, gOutChannelGroup,
	   gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	   n, c ) ;
	c+=1 ;
	break ;
      case 2:
	func_even<FLAYOUT, 2>(pGOut, pKernel, pGIn,
	   gOutChannel, gOutWidth, gOutHeight,
	   gInChannel, gInWidth, gInHeight,
	   kernWidth, kernHeight,
	   gInChannelGroup, gOutChannelGroup,
	   gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	   n, c ) ;
	c+=2 ;
	break ;
      case 3:
	func_odd<FLAYOUT, 3>(pGOut, pKernel, pGIn,
	   gOutChannel, gOutWidth, gOutHeight,
	   gInChannel, gInWidth, gInHeight,
	   kernWidth, kernHeight,
	   gInChannelGroup, gOutChannelGroup,
	   gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	   n, c ) ;
	c+=3 ;
	break ;
      case 4:
	func_even<FLAYOUT, 4>(pGOut, pKernel, pGIn,
	   gOutChannel, gOutWidth, gOutHeight,
	   gInChannel, gInWidth, gInHeight,
	   kernWidth, kernHeight,
	   gInChannelGroup, gOutChannelGroup,
	   gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	   n, c ) ;
	c+=4 ;
	break ;
      case 5:
	func_odd<FLAYOUT, 5>(pGOut, pKernel, pGIn,
	   gOutChannel, gOutWidth, gOutHeight,
	   gInChannel, gInWidth, gInHeight,
	   kernWidth, kernHeight,
	   gInChannelGroup, gOutChannelGroup,
	   gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	   n, c ) ;
	c+=5 ;
	break ;
      case 6:
	func_even<FLAYOUT, 6>(pGOut, pKernel, pGIn,
	   gOutChannel, gOutWidth, gOutHeight,
	   gInChannel, gInWidth, gInHeight,
	   kernWidth, kernHeight,
	   gInChannelGroup, gOutChannelGroup,
	   gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	   n, c ) ;
	c+=6 ;
	break ;
      case 7:
	func_odd<FLAYOUT, 7>(pGOut, pKernel, pGIn,
	   gOutChannel, gOutWidth, gOutHeight,
	   gInChannel, gInWidth, gInHeight,
	   kernWidth, kernHeight,
	   gInChannelGroup, gOutChannelGroup,
	   gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	   n, c ) ;
	c+=7 ;
	break ;
      case 8:
	func_even<FLAYOUT, 8>(pGOut, pKernel, pGIn,
	   gOutChannel, gOutWidth, gOutHeight,
	   gInChannel, gInWidth, gInHeight,
	   kernWidth, kernHeight,
	   gInChannelGroup, gOutChannelGroup,
	   gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	   n, c ) ;
	c+=8 ;
	break ;
      case 9:
	func_odd<FLAYOUT, 9>(pGOut, pKernel, pGIn,
	   gOutChannel, gOutWidth, gOutHeight,
	   gInChannel, gInWidth, gInHeight,
	   kernWidth, kernHeight,
	   gInChannelGroup, gOutChannelGroup,
	   gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	   n, c ) ;
	c+=9 ;
	break ;
      case 10:
	func_even<FLAYOUT, 10>(pGOut, pKernel, pGIn,
	   gOutChannel, gOutWidth, gOutHeight,
	   gInChannel, gInWidth, gInHeight,
	   kernWidth, kernHeight,
	   gInChannelGroup, gOutChannelGroup,
	   gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	   n, c ) ;
	c+=10 ;
	break ;
      case 11:
	func_odd<FLAYOUT, 11>(pGOut, pKernel, pGIn,
	   gOutChannel, gOutWidth, gOutHeight,
	   gInChannel, gInWidth, gInHeight,
	   kernWidth, kernHeight,
	   gInChannelGroup, gOutChannelGroup,
	   gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	   n, c ) ;
	c+=11 ;
	break ;
      case 12:
	func_even<FLAYOUT, 12>(pGOut, pKernel, pGIn,
	   gOutChannel, gOutWidth, gOutHeight,
	   gInChannel, gInWidth, gInHeight,
	   kernWidth, kernHeight,
	   gInChannelGroup, gOutChannelGroup,
	   gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	   n, c ) ;
	c+=12 ;
	break ;
      case 13:
	func_odd<FLAYOUT, 13>(pGOut, pKernel, pGIn,
	   gOutChannel, gOutWidth, gOutHeight,
	   gInChannel, gInWidth, gInHeight,
	   kernWidth, kernHeight,
	   gInChannelGroup, gOutChannelGroup,
	   gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	   n, c ) ;
	c+=13 ;
	break ;
      case 14:
	func_even<FLAYOUT, 14>(pGOut, pKernel, pGIn,
	   gOutChannel, gOutWidth, gOutHeight,
	   gInChannel, gInWidth, gInHeight,
	   kernWidth, kernHeight,
	   gInChannelGroup, gOutChannelGroup,
	   gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	   n, c ) ;
	c+=14 ;
	break ;
      case 15:
	func_odd<FLAYOUT, 15>(pGOut, pKernel, pGIn,
	   gOutChannel, gOutWidth, gOutHeight,
	   gInChannel, gInWidth, gInHeight,
	   kernWidth, kernHeight,
	   gInChannelGroup, gOutChannelGroup,
	   gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	   n, c ) ;
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
	   n, c) ;
	c+= 16 ;
      } // gInChannel
    } // group
  } // batch
}

extern "C"
vednnError_t
vednnConvolutionBackwardData_direct_dil1_str2_pad2_ker5(
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
//  const int64_t strideWidth    = pParamConv->strideWidth;		// 2
//  const int64_t strideHeight   = pParamConv->strideHeight;		// 2
//  const int64_t padWidth       = pParamConv->padWidth;		// 2
//  const int64_t padHeight      = pParamConv->padHeight;		// 2
//  const int64_t dilationWidth  = pParamConv->dilationWidth;		// 1
//  const int64_t dilationHeight = pParamConv->dilationHeight;		// 1

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
