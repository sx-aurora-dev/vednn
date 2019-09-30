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
  int64_t gInIndex = gInGroupOffset + ((n * gInChannel + c) * gInHeight ) * gInWidth  ;

  for (int64_t h = 0; h < gInHeight ; h++ ) {
    for (int64_t w = 0; w < gInWidth ; w += VLEN ) {
      const int64_t vl = gInWidth - w < VLEN ? gInWidth - w  : VLEN ;

      __vr vrsum0  = _vel_vbrds_vsl(0.f, vl) ;
      __vr vrsum12 = _vel_pvbrd_vsl(0UL, vl) ;
      __vr vrsum34 = _vel_pvbrd_vsl(0UL, vl) ;
      __vr vrsum56 = _vel_pvbrd_vsl(0UL, vl) ;
      __vr vrsum78 = _vel_pvbrd_vsl(0UL, vl) ;
      __vr vrsum9A = _vel_pvbrd_vsl(0UL, vl) ;
      __vr vrsumBC = _vel_pvbrd_vsl(0UL, vl) ;
      __vr vrsumDE = _vel_pvbrd_vsl(0UL, vl) ;

      __vr vrw   = _vel_vaddsl_vsvl(w, _vel_vseq_vl(vl), vl) ;

      for (int64_t r=0; r<kernHeight; r++) {
	int64_t i = h - r * dilationHeight + padHeight ;
	int64_t y = i/strideHeight;
	if ( y*strideHeight != i || y < 0 || gOutHeight <= y)  continue ;

	for (int64_t s=0; s<kernWidth; s++) {
	  __vr vrj = _vel_vaddsl_vsvl(padWidth-s*dilationWidth, vrw, vl) ;
	  __vr vrx = _vel_vdivsl_vvsl(vrj, strideWidth, vl) ;

	  __vm256 vmx0 =  _vel_vfmkleq_mvl(_vel_vcmpsl_vvvl(vrj, _vel_vmulsl_vsvl(strideWidth, vrx, vl), vl), vl) ;
	  __vm256 vmx1 =  _vel_vfmklge_mvl(vrx, vl) ;
	  __vm256 vmx2 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutWidth,vrx, vl), vl) ;

	  __vm256 vmx = _vel_andm_mmm(_vel_andm_mmm(vmx0, vmx1), vmx2) ;

	  for (int64_t k=0; k<gOutChannelGroup; k++) {
	    int64_t gOutIndex    = gOutGroupOffset + ((n * gOutChannel + k) * gOutHeight) * gOutWidth ;
	    __vr vrgout_ptr = _vel_vsfa_vvssl(_vel_vaddsl_vsvl(gOutWidth*y, vrx, vl),
					      2,
					      (unsigned long)(pGOut+gOutIndex), vl) ;

	    __vr vrgout = _vel_vgtu_vvssml(vrgout_ptr, 0, 0, vmx, vl) ;
	    vrgout = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrgout, vmx, vl) ;

	    __vr vrgoutP = _vel_vshf_vvvsl(vrgout, vrgout, VE_VSHUFFLE_YUZU, vl) ;

#define FILTER_OFFSET(k,c,r,s) ( kernGroupOffset + filter_index<FLAYOUT>(k,c,r,s, gInChannelGroup, gOutChannelGroup, kernHeight, kernWidth) )

	    const float    kerValue0  = pKernel[FILTER_OFFSET(k,c+ 0,r,s)] ;
	    const uint64_t kerValue12 = _vel_pack_f32p(pKernel + FILTER_OFFSET(k,c+ 1,r,s),
						       pKernel + FILTER_OFFSET(k,c+ 2,r,s)) ;
	    const uint64_t kerValue34 = _vel_pack_f32p(pKernel + FILTER_OFFSET(k,c+ 3,r,s),
						       pKernel + FILTER_OFFSET(k,c+ 4,r,s)) ;
	    const uint64_t kerValue56 = _vel_pack_f32p(pKernel + FILTER_OFFSET(k,c+ 5,r,s),
						       pKernel + FILTER_OFFSET(k,c+ 6,r,s)) ;
	    const uint64_t kerValue78 = _vel_pack_f32p(pKernel + FILTER_OFFSET(k,c+ 7,r,s),
						       pKernel + FILTER_OFFSET(k,c+ 8,r,s)) ;
	    const uint64_t kerValue9A = _vel_pack_f32p(pKernel + FILTER_OFFSET(k,c+ 9,r,s),
						       pKernel + FILTER_OFFSET(k,c+10,r,s)) ;
	    const uint64_t kerValueBC = _vel_pack_f32p(pKernel + FILTER_OFFSET(k,c+11,r,s),
						       pKernel + FILTER_OFFSET(k,c+12,r,s)) ;
	    const uint64_t kerValueDE = _vel_pack_f32p(pKernel + FILTER_OFFSET(k,c+13,r,s),
						       pKernel + FILTER_OFFSET(k,c+14,r,s)) ;

	    vrsum0 = _vel_vfmads_vvsvl(vrsum0, kerValue0, vrgout, vl) ;
	    if(NUMCHANNEL>= 3) vrsum12 = _vel_pvfmad_vvsvl(vrsum12, kerValue12, vrgoutP, vl) ;
	    if(NUMCHANNEL>= 5) vrsum34 = _vel_pvfmad_vvsvl(vrsum34, kerValue34, vrgoutP, vl) ;
	    if(NUMCHANNEL>= 7) vrsum56 = _vel_pvfmad_vvsvl(vrsum56, kerValue56, vrgoutP, vl) ;
	    if(NUMCHANNEL>= 9) vrsum78 = _vel_pvfmad_vvsvl(vrsum78, kerValue78, vrgoutP, vl) ;
	    if(NUMCHANNEL>=11) vrsum9A = _vel_pvfmad_vvsvl(vrsum9A, kerValue9A, vrgoutP, vl) ;
	    if(NUMCHANNEL>=13) vrsumBC = _vel_pvfmad_vvsvl(vrsumBC, kerValueBC, vrgoutP, vl) ;
	    if(NUMCHANNEL>=15) vrsumDE = _vel_pvfmad_vvsvl(vrsumDE, kerValueDE, vrgoutP, vl) ;

#undef FILTER_OFFSET
	  } // gOutChannel

	} // kernWidth
      } // kernHeight

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
  int64_t gInIndex = gInGroupOffset + ((n * gInChannel + c) * gInHeight ) * gInWidth  ;

  for (int64_t h = 0; h < gInHeight ; h++ ) {
    for (int64_t w = 0; w < gInWidth ; w += VLEN ) {
      const int64_t vl = gInWidth - w < VLEN ? gInWidth - w  : VLEN ;

      __vr vrsum01 = _vel_pvbrd_vsl(0UL, vl) ;
      __vr vrsum23 = _vel_pvbrd_vsl(0UL, vl) ;
      __vr vrsum45 = _vel_pvbrd_vsl(0UL, vl) ;
      __vr vrsum67 = _vel_pvbrd_vsl(0UL, vl) ;
      __vr vrsum89 = _vel_pvbrd_vsl(0UL, vl) ;
      __vr vrsumAB = _vel_pvbrd_vsl(0UL, vl) ;
      __vr vrsumCD = _vel_pvbrd_vsl(0UL, vl) ;
      __vr vrsumEF = _vel_pvbrd_vsl(0UL, vl) ;

      __vr vrw   = _vel_vaddsl_vsvl(w, _vel_vseq_vl(vl), vl) ;

      for (int64_t r=0; r<kernHeight; r++) {
	int64_t i = h - r * dilationHeight + padHeight ;
	int64_t y = i/strideHeight;
	if ( y*strideHeight != i || y < 0 || gOutHeight <= y)  continue ;

	for (int64_t s=0; s<kernWidth; s++) {
	  __vr vrj = _vel_vaddsl_vsvl(padWidth-s*dilationWidth, vrw, vl) ;
	  __vr vrx = _vel_vdivsl_vvsl(vrj, strideWidth, vl) ;

	  __vm256 vmx0 =  _vel_vfmkleq_mvl(_vel_vcmpsl_vvvl(vrj, _vel_vmulsl_vsvl(strideWidth, vrx, vl), vl), vl) ;
	  __vm256 vmx1 =  _vel_vfmklge_mvl(vrx, vl) ;
	  __vm256 vmx2 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutWidth,vrx, vl), vl) ;

	  __vm256 vmx = _vel_andm_mmm(_vel_andm_mmm(vmx0, vmx1), vmx2) ;

	  for (int64_t k=0; k<gOutChannelGroup; k++) {
	    int64_t gOutIndex    = gOutGroupOffset + ((n * gOutChannel + k) * gOutHeight) * gOutWidth ;
	    __vr vrgout_ptr = _vel_vsfa_vvssl(_vel_vaddsl_vsvl(gOutWidth*y, vrx, vl),
					      2,
					      (unsigned long)(pGOut+gOutIndex), vl) ;

	    __vr vrgout = _vel_vgtu_vvssml(vrgout_ptr, 0, 0, vmx, vl) ;
	    vrgout = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrgout, vmx, vl) ;

	    __vr vrgoutP = _vel_vshf_vvvsl(vrgout, vrgout, VE_VSHUFFLE_YUZU, vl) ;

#define FILTER_OFFSET(k,c,r,s) ( kernGroupOffset + filter_index<FLAYOUT>(k,c,r,s, gInChannelGroup, gOutChannelGroup, kernHeight, kernWidth) )

	    const uint64_t kerValue01 = _vel_pack_f32p(pKernel + FILTER_OFFSET(k,c+ 0,r,s),
						       pKernel + FILTER_OFFSET(k,c+ 1,r,s)) ;
	    const uint64_t kerValue23 = _vel_pack_f32p(pKernel + FILTER_OFFSET(k,c+ 2,r,s),
						       pKernel + FILTER_OFFSET(k,c+ 3,r,s)) ;
	    const uint64_t kerValue45 = _vel_pack_f32p(pKernel + FILTER_OFFSET(k,c+ 4,r,s),
						       pKernel + FILTER_OFFSET(k,c+ 5,r,s)) ;
	    const uint64_t kerValue67 = _vel_pack_f32p(pKernel + FILTER_OFFSET(k,c+ 6,r,s),
						       pKernel + FILTER_OFFSET(k,c+ 7,r,s)) ;
	    const uint64_t kerValue89 = _vel_pack_f32p(pKernel + FILTER_OFFSET(k,c+ 8,r,s),
						       pKernel + FILTER_OFFSET(k,c+ 9,r,s)) ;
	    const uint64_t kerValueAB = _vel_pack_f32p(pKernel + FILTER_OFFSET(k,c+10,r,s),
						       pKernel + FILTER_OFFSET(k,c+11,r,s)) ;
	    const uint64_t kerValueCD = _vel_pack_f32p(pKernel + FILTER_OFFSET(k,c+12,r,s),
						       pKernel + FILTER_OFFSET(k,c+13,r,s)) ;
	    const uint64_t kerValueEF = _vel_pack_f32p(pKernel + FILTER_OFFSET(k,c+14,r,s),
						       pKernel + FILTER_OFFSET(k,c+15,r,s)) ;

	    if(NUMCHANNEL>= 2) vrsum01 = _vel_pvfmad_vvsvl(vrsum01, kerValue01, vrgoutP, vl) ;
	    if(NUMCHANNEL>= 4) vrsum23 = _vel_pvfmad_vvsvl(vrsum23, kerValue23, vrgoutP, vl) ;
	    if(NUMCHANNEL>= 6) vrsum45 = _vel_pvfmad_vvsvl(vrsum45, kerValue45, vrgoutP, vl) ;
	    if(NUMCHANNEL>= 8) vrsum67 = _vel_pvfmad_vvsvl(vrsum67, kerValue67, vrgoutP, vl) ;
	    if(NUMCHANNEL>=10) vrsum89 = _vel_pvfmad_vvsvl(vrsum89, kerValue89, vrgoutP, vl) ;
	    if(NUMCHANNEL>=12) vrsumAB = _vel_pvfmad_vvsvl(vrsumAB, kerValueAB, vrgoutP, vl) ;
	    if(NUMCHANNEL>=14) vrsumCD = _vel_pvfmad_vvsvl(vrsumCD, kerValueCD, vrgoutP, vl) ;
	    if(NUMCHANNEL>=16) vrsumEF = _vel_pvfmad_vvsvl(vrsumEF, kerValueEF, vrgoutP, vl) ;

#undef FILTER_OFFSET
	  } // gOutChannel

	} // kernWidth
      } // kernHeight

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
	func_odd<FLAYOUT, 1>(pGOut, pKernel, pGIn,
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
	func_even<FLAYOUT, 2>(pGOut, pKernel, pGIn,
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
	func_odd<FLAYOUT, 3>(pGOut, pKernel, pGIn,
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
	func_even<FLAYOUT, 4>(pGOut, pKernel, pGIn,
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
	func_odd<FLAYOUT, 5>(pGOut, pKernel, pGIn,
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
	func_even<FLAYOUT, 6>(pGOut, pKernel, pGIn,
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
	func_odd<FLAYOUT, 7>(pGOut, pKernel, pGIn,
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
	func_even<FLAYOUT, 8>(pGOut, pKernel, pGIn,
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
	func_odd<FLAYOUT, 9>(pGOut, pKernel, pGIn,
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
	func_even<FLAYOUT, 10>(pGOut, pKernel, pGIn,
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
	func_odd<FLAYOUT, 11>(pGOut, pKernel, pGIn,
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
	func_even<FLAYOUT, 12>(pGOut, pKernel, pGIn,
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
	func_odd<FLAYOUT, 13>(pGOut, pKernel, pGIn,
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
	func_even<FLAYOUT, 14>(pGOut, pKernel, pGIn,
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
	func_odd<FLAYOUT, 15>(pGOut, pKernel, pGIn,
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
	func_even<FLAYOUT, 16>(pGOut, pKernel, pGIn,
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
vednnConvolutionBackwardData_direct_default(
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

#if 0	// reference version
vednnError_t
vednnConvolutionBackwardData_direct_default2(
    const vednnTensorParam_t * restrict 	pParamGradOut,
    const void * restrict 			pDataGradOut,
    const vednnFilterParam_t * restrict 	pParamKernel,
    const void * restrict 			pDataKernel,
    const vednnConvolutionParam_t * restrict 	pParamConv,
    const vednnTensorParam_t * restrict 	pParamGradIn,
    void * restrict 				pDataGradIn
)
{
  const int64_t batch      = pParamGradOut->batch;
  const int64_t gOutChannel= pParamGradOut->channel;
  const int64_t gOutWidth  = pParamGradOut->width;
  const int64_t gOutHeight = pParamGradOut->height;
  const int64_t gInChannel = pParamGradIn->channel;
  const int64_t gInWidth   = pParamGradIn->width;
  const int64_t gInHeight  = pParamGradIn->height;
  const int64_t kernWidth  = pParamKernel->width;
  const int64_t kernHeight = pParamKernel->height;

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

  const int oPixels= gOutHeight*gOutWidth ;
#if 0
  /* version 1 : base version */
  {

    for (int64_t n=0; n<batch; n++) {
      for (int64_t g = 0; g < group; g++) {
	int64_t gInGroupOffset  = g * gInChannelGroup * gInHeight * gInWidth;
	int64_t gOutGroupOffset = g * gOutChannelGroup * gOutHeight  * gOutWidth;
	int64_t kernGroupOffset = g * gOutChannelGroup * gInChannelGroup * kernHeight * kernWidth;


	for (int64_t c=0; c<gInChannelGroup; c++) {
	  for (int64_t h=0; h<gInHeight; h++) {
	    for (int64_t w=0; w<gInWidth; w++) {
	      int64_t gInIndex = gInGroupOffset + ((n * gInChannel + c) * gInHeight + h) * gInWidth + w;

	      float sum = 0.0f ;

	      for (int64_t r=0; r<kernHeight; r++) {
		int64_t i = h - r * dilationHeight + padHeight ;
		int64_t y = i/strideHeight;
		if ( y*strideHeight != i || y < 0 || gOutHeight <= y)  continue ;

		for (int64_t s=0; s<kernWidth; s++) {
		  int64_t j = w - s * dilationWidth  + padWidth ;
		  int64_t x = j/strideWidth ;
		  if (x*strideWidth !=j || x < 0 || gOutWidth <= x) continue;

		  for (int64_t k=0; k<gOutChannelGroup; k++) {
		    int64_t gOutIndex   = gOutGroupOffset + ((n * gOutChannel + k) * gOutHeight + y) * gOutWidth + x;
		    int64_t kernelIndex = kernGroupOffset + ((k * gInChannelGroup + c) * kernHeight + r) * kernWidth + s;
		    sum += (pGOut[gOutIndex] * pKernel[kernelIndex]);
		  } // gOutChannel

		} // kernWidth
	      } // kernHeight

	      pGIn[gInIndex] = sum ;

	    } // gInWidth
	  } // gInHeight
	} // gInChannel
      } // group
    } // batch
  }
#else
  /* version 0 : generated from forward propagation */
  {
    float * restrict pIn     = (float * restrict) pDataGradIn ;
    float * restrict pKernel = (float * restrict) pDataKernel;
    float * restrict pOut    = (float * restrict) pDataGradOut;

    const int64_t outChannel = gOutChannel ;
    const int64_t inChannel  = gInChannel ;

    const int64_t inChannelGroup  = gInChannelGroup ;
    const int64_t outChannelGroup = gOutChannelGroup ;

    const int64_t inHeight = gInHeight ;
    const int64_t inWidth  = gInWidth ;

    const int64_t outHeight = gOutHeight ;
    const int64_t outWidth  = gOutWidth ;

    for(int64_t i=0; i<inChannel*inHeight*inWidth; i++) pIn[i] = 0.0f ;

    for (int64_t n=0; n<batch; n++) {
      for (int64_t g = 0; g < group; g++) {
	int64_t inGroupOffset   = g * inChannelGroup  * inHeight  * inWidth;
	int64_t outGroupOffset  = g * outChannelGroup * outHeight * outWidth;
	int64_t kernGroupOffset = g * outChannelGroup * inChannelGroup * kernHeight * kernWidth;

	for (int64_t k=0; k<outChannelGroup; k++) {
	  for (int64_t p=0; p<outHeight; p++) {
	    int64_t i = p * strideHeight - padHeight;
	    for (int64_t q=0; q<outWidth; q++) {
	      int64_t j = q * strideWidth - padWidth;
	      int64_t outIndex  = outGroupOffset + ((n * outChannel + k) * outHeight + p) * outWidth + q;
	      for (int64_t c=0; c<inChannelGroup; c++) {
		for (int64_t h=0; h<kernHeight; h++) {
		  for (int64_t w=0; w<kernWidth; w++) {
		    int64_t y = i + h * dilationHeight;
		    int64_t x = j + w * dilationWidth;
		    if (y < 0 || inHeight <= y) {
		      continue;
		    }
		    if (x < 0 || inWidth <= x) {
		      continue;
		    }
		    int64_t inputIndex  = inGroupOffset + ((n * inChannel + c) * inHeight + y) * inWidth + x;
		    int64_t kernelIndex = kernGroupOffset + ((k * inChannelGroup + c) * kernHeight + h) * kernWidth + w;
		    pIn[inputIndex] += (pOut[outIndex] * pKernel[kernelIndex]);
		  } // kernWidth
		} // kernHeight
	      } // inChannel
	    } // outWidth
	  } // outHeight
	} // outChannel
      } // group
    } // batch
  }
#endif

  return VEDNN_SUCCESS;
}
#endif