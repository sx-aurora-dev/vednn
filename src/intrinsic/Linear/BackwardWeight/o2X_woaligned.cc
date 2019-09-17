#include <stdint.h>
#include <float.h>

#include <stdio.h>

#include "vednn.h"

#include "velintrin.h"
#define VLEN	(256)

template <int BATCH, bool UPDATE>
static inline void func(
  const uint64_t	inDim,
  const uint64_t	outDim,
  const uint64_t	nInDim,
  const float * 	pIn,
  const float * 	pGOut,
  float * 		pGWeight
)
{
  for(int64_t o=0; o<outDim; o+=2*VLEN) {
    const int64_t vl = (outDim-o < 2*VLEN ? outDim - o : 2*VLEN) >> 1 ;

    __vr vrgout_b0 = _vel_vld_vssl(8, pGOut+0*outDim+o, vl) ;
    __vr vrgout_b1 = _vel_vld_vssl(8, pGOut+1*outDim+o, vl) ;
    __vr vrgout_b2 = _vel_vld_vssl(8, pGOut+2*outDim+o, vl) ;
    __vr vrgout_b3 = _vel_vld_vssl(8, pGOut+3*outDim+o, vl) ;
    __vr vrgout_b4 = _vel_vld_vssl(8, pGOut+4*outDim+o, vl) ;
    __vr vrgout_b5 = _vel_vld_vssl(8, pGOut+5*outDim+o, vl) ;
    __vr vrgout_b6 = _vel_vld_vssl(8, pGOut+6*outDim+o, vl) ;
    __vr vrgout_b7 = _vel_vld_vssl(8, pGOut+7*outDim+o, vl) ;

    int64_t i=0;
    if(nInDim & 0x1) {
      __vr vrgw ;

      if(UPDATE) {
	vrgw = _vel_vld_vssl(8, pGWeight+(i+0)*outDim+o, vl) ;

	const uint64_t in_b0 = _vel_pack_f32a(pIn+0*inDim+i) ;
	vrgw = _vel_pvfmad_vvsvl(vrgw, in_b0, vrgout_b0, vl) ;
      }
      else {
	const uint64_t in_b0 = _vel_pack_f32a(pIn+0*inDim+i) ;
	vrgw = _vel_pvfmul_vsvl(in_b0, vrgout_b0, vl) ;
      }

      if(BATCH>=2)  {
	const uint64_t in_b1 = _vel_pack_f32a(pIn+1*inDim+i) ;
	vrgw = _vel_pvfmad_vvsvl(vrgw, in_b1, vrgout_b1, vl) ;
      }
      if(BATCH>=3) {
	const uint64_t in_b2 = _vel_pack_f32a(pIn+2*inDim+i) ;
	vrgw = _vel_pvfmad_vvsvl(vrgw, in_b2, vrgout_b2, vl) ;
      }
      if(BATCH>=4) {
	const uint64_t in_b3 = _vel_pack_f32a(pIn+3*inDim+i) ;
	vrgw = _vel_pvfmad_vvsvl(vrgw, in_b3, vrgout_b3, vl) ;
      }
      if(BATCH>=5) {
	const uint64_t in_b4 = _vel_pack_f32a(pIn+4*inDim+i) ;
	vrgw = _vel_pvfmad_vvsvl(vrgw, in_b4, vrgout_b4, vl) ;
      }
      if(BATCH>=6) {
	const uint64_t in_b5 = _vel_pack_f32a(pIn+5*inDim+i) ;
	vrgw = _vel_pvfmad_vvsvl(vrgw, in_b5, vrgout_b5, vl) ;
      }
      if(BATCH>=7) {
	const uint64_t in_b6 = _vel_pack_f32a(pIn+6*inDim+i) ;
	vrgw = _vel_pvfmad_vvsvl(vrgw, in_b6, vrgout_b6, vl) ;
      }
      if(BATCH>=8) {
	const uint64_t in_b7 = _vel_pack_f32a(pIn+7*inDim+i) ;
	vrgw = _vel_pvfmad_vvsvl(vrgw, in_b7, vrgout_b7, vl) ;
      }

      _vel_vst_vssl(vrgw, 8, pGWeight+i*outDim+o, vl) ;

      i+=1;
    }
    if((nInDim>>1) & 0x1) {
      __vr vrgw_i0 ;
      __vr vrgw_i1 ;

      if(UPDATE) {
	vrgw_i0 = _vel_vld_vssl(8, pGWeight+(i+0)*outDim+o, vl) ;
	vrgw_i1 = _vel_vld_vssl(8, pGWeight+(i+1)*outDim+o, vl) ;

	const uint64_t in_i0_b0 = _vel_pack_f32a(pIn+0*inDim+i+0) ;
	const uint64_t in_i1_b0 = _vel_pack_f32a(pIn+0*inDim+i+1) ;
	vrgw_i0 = _vel_pvfmad_vvsvl(vrgw_i0, in_i0_b0, vrgout_b0, vl) ;
	vrgw_i1 = _vel_pvfmad_vvsvl(vrgw_i1, in_i1_b0, vrgout_b0, vl) ;
      }
      else {
	const uint64_t in_i0_b0 = _vel_pack_f32a(pIn+0*inDim+i+0) ;
	const uint64_t in_i1_b0 = _vel_pack_f32a(pIn+0*inDim+i+1) ;
	vrgw_i0 = _vel_pvfmul_vsvl(in_i0_b0, vrgout_b0, vl) ;
	vrgw_i1 = _vel_pvfmul_vsvl(in_i1_b0, vrgout_b0, vl) ;
      }

      if(BATCH>=2)  {
	const uint64_t in_i0_b1 = _vel_pack_f32a(pIn+1*inDim+i+0) ;
	const uint64_t in_i1_b1 = _vel_pack_f32a(pIn+1*inDim+i+1) ;
	vrgw_i0 = _vel_pvfmad_vvsvl(vrgw_i0, in_i0_b1, vrgout_b1, vl) ;
	vrgw_i1 = _vel_pvfmad_vvsvl(vrgw_i1, in_i1_b1, vrgout_b1, vl) ;
      }
      if(BATCH>=3) {
	const uint64_t in_i0_b2 = _vel_pack_f32a(pIn+2*inDim+i+0) ;
	const uint64_t in_i1_b2 = _vel_pack_f32a(pIn+2*inDim+i+1) ;
	vrgw_i0 = _vel_pvfmad_vvsvl(vrgw_i0, in_i0_b2, vrgout_b2, vl) ;
	vrgw_i1 = _vel_pvfmad_vvsvl(vrgw_i1, in_i1_b2, vrgout_b2, vl) ;
      }
      if(BATCH>=4) {
	const uint64_t in_i0_b3 = _vel_pack_f32a(pIn+3*inDim+i+0) ;
	const uint64_t in_i1_b3 = _vel_pack_f32a(pIn+3*inDim+i+1) ;
	vrgw_i0 = _vel_pvfmad_vvsvl(vrgw_i0, in_i0_b3, vrgout_b3, vl) ;
	vrgw_i1 = _vel_pvfmad_vvsvl(vrgw_i1, in_i1_b3, vrgout_b3, vl) ;
      }
      if(BATCH>=5) {
	const uint64_t in_i0_b4 = _vel_pack_f32a(pIn+4*inDim+i+0) ;
	const uint64_t in_i1_b4 = _vel_pack_f32a(pIn+4*inDim+i+1) ;
	vrgw_i0 = _vel_pvfmad_vvsvl(vrgw_i0, in_i0_b4, vrgout_b4, vl) ;
	vrgw_i1 = _vel_pvfmad_vvsvl(vrgw_i1, in_i1_b4, vrgout_b4, vl) ;
      }
      if(BATCH>=6) {
	const uint64_t in_i0_b5 = _vel_pack_f32a(pIn+5*inDim+i+0) ;
	const uint64_t in_i1_b5 = _vel_pack_f32a(pIn+5*inDim+i+1) ;
	vrgw_i0 = _vel_pvfmad_vvsvl(vrgw_i0, in_i0_b5, vrgout_b5, vl) ;
	vrgw_i1 = _vel_pvfmad_vvsvl(vrgw_i1, in_i1_b5, vrgout_b5, vl) ;
      }
      if(BATCH>=7) {
	const uint64_t in_i0_b6 = _vel_pack_f32a(pIn+6*inDim+i+0) ;
	const uint64_t in_i1_b6 = _vel_pack_f32a(pIn+6*inDim+i+1) ;
	vrgw_i0 = _vel_pvfmad_vvsvl(vrgw_i0, in_i0_b6, vrgout_b6, vl) ;
	vrgw_i1 = _vel_pvfmad_vvsvl(vrgw_i1, in_i1_b6, vrgout_b6, vl) ;
      }
      if(BATCH>=8) {
	const uint64_t in_i0_b7 = _vel_pack_f32a(pIn+7*inDim+i+0) ;
	const uint64_t in_i1_b7 = _vel_pack_f32a(pIn+7*inDim+i+1) ;
	vrgw_i0 = _vel_pvfmad_vvsvl(vrgw_i0, in_i0_b7, vrgout_b7, vl) ;
	vrgw_i1 = _vel_pvfmad_vvsvl(vrgw_i1, in_i1_b7, vrgout_b7, vl) ;
      }

      _vel_vst_vssl(vrgw_i0, 8, pGWeight+(i+0)*outDim+o, vl) ;
      _vel_vst_vssl(vrgw_i1, 8, pGWeight+(i+1)*outDim+o, vl) ;

      i+=2 ;
    }
    if((nInDim>>2) & 0x1) {
      __vr vrgw_i0 ;
      __vr vrgw_i1 ;
      __vr vrgw_i2 ;
      __vr vrgw_i3 ;


      if(UPDATE) {
	vrgw_i0 = _vel_vld_vssl(8, pGWeight+(i+0)*outDim+o, vl) ;
	vrgw_i1 = _vel_vld_vssl(8, pGWeight+(i+1)*outDim+o, vl) ;
	vrgw_i2 = _vel_vld_vssl(8, pGWeight+(i+2)*outDim+o, vl) ;
	vrgw_i3 = _vel_vld_vssl(8, pGWeight+(i+3)*outDim+o, vl) ;

	const uint64_t in_i0_b0 = _vel_pack_f32a(pIn+0*inDim+i+0) ;
	const uint64_t in_i1_b0 = _vel_pack_f32a(pIn+0*inDim+i+1) ;
	const uint64_t in_i2_b0 = _vel_pack_f32a(pIn+0*inDim+i+2) ;
	const uint64_t in_i3_b0 = _vel_pack_f32a(pIn+0*inDim+i+3) ;
	vrgw_i0 = _vel_pvfmad_vvsvl(vrgw_i0, in_i0_b0, vrgout_b0, vl) ;
	vrgw_i1 = _vel_pvfmad_vvsvl(vrgw_i1, in_i1_b0, vrgout_b0, vl) ;
	vrgw_i2 = _vel_pvfmad_vvsvl(vrgw_i2, in_i2_b0, vrgout_b0, vl) ;
	vrgw_i3 = _vel_pvfmad_vvsvl(vrgw_i3, in_i3_b0, vrgout_b0, vl) ;
      }
      else {
	const uint64_t in_i0_b0 = _vel_pack_f32a(pIn+0*inDim+i+0) ;
	const uint64_t in_i1_b0 = _vel_pack_f32a(pIn+0*inDim+i+1) ;
	const uint64_t in_i2_b0 = _vel_pack_f32a(pIn+0*inDim+i+2) ;
	const uint64_t in_i3_b0 = _vel_pack_f32a(pIn+0*inDim+i+3) ;
	vrgw_i0 = _vel_pvfmul_vsvl(in_i0_b0, vrgout_b0, vl) ;
	vrgw_i1 = _vel_pvfmul_vsvl(in_i1_b0, vrgout_b0, vl) ;
	vrgw_i2 = _vel_pvfmul_vsvl(in_i2_b0, vrgout_b0, vl) ;
	vrgw_i3 = _vel_pvfmul_vsvl(in_i3_b0, vrgout_b0, vl) ;
      }

      if(BATCH>=2)  {
	const uint64_t in_i0_b1 = _vel_pack_f32a(pIn+1*inDim+i+0) ;
	const uint64_t in_i1_b1 = _vel_pack_f32a(pIn+1*inDim+i+1) ;
	const uint64_t in_i2_b1 = _vel_pack_f32a(pIn+1*inDim+i+2) ;
	const uint64_t in_i3_b1 = _vel_pack_f32a(pIn+1*inDim+i+3) ;
	vrgw_i0 = _vel_pvfmad_vvsvl(vrgw_i0, in_i0_b1, vrgout_b1, vl) ;
	vrgw_i1 = _vel_pvfmad_vvsvl(vrgw_i1, in_i1_b1, vrgout_b1, vl) ;
	vrgw_i2 = _vel_pvfmad_vvsvl(vrgw_i2, in_i2_b1, vrgout_b1, vl) ;
	vrgw_i3 = _vel_pvfmad_vvsvl(vrgw_i3, in_i3_b1, vrgout_b1, vl) ;
      }
      if(BATCH>=3) {
	const uint64_t in_i0_b2 = _vel_pack_f32a(pIn+2*inDim+i+0) ;
	const uint64_t in_i1_b2 = _vel_pack_f32a(pIn+2*inDim+i+1) ;
	const uint64_t in_i2_b2 = _vel_pack_f32a(pIn+2*inDim+i+2) ;
	const uint64_t in_i3_b2 = _vel_pack_f32a(pIn+2*inDim+i+3) ;
	vrgw_i0 = _vel_pvfmad_vvsvl(vrgw_i0, in_i0_b2, vrgout_b2, vl) ;
	vrgw_i1 = _vel_pvfmad_vvsvl(vrgw_i1, in_i1_b2, vrgout_b2, vl) ;
	vrgw_i2 = _vel_pvfmad_vvsvl(vrgw_i2, in_i2_b2, vrgout_b2, vl) ;
	vrgw_i3 = _vel_pvfmad_vvsvl(vrgw_i3, in_i3_b2, vrgout_b2, vl) ;
      }
      if(BATCH>=4) {
	const uint64_t in_i0_b3 = _vel_pack_f32a(pIn+3*inDim+i+0) ;
	const uint64_t in_i1_b3 = _vel_pack_f32a(pIn+3*inDim+i+1) ;
	const uint64_t in_i2_b3 = _vel_pack_f32a(pIn+3*inDim+i+2) ;
	const uint64_t in_i3_b3 = _vel_pack_f32a(pIn+3*inDim+i+3) ;
	vrgw_i0 = _vel_pvfmad_vvsvl(vrgw_i0, in_i0_b3, vrgout_b3, vl) ;
	vrgw_i1 = _vel_pvfmad_vvsvl(vrgw_i1, in_i1_b3, vrgout_b3, vl) ;
	vrgw_i2 = _vel_pvfmad_vvsvl(vrgw_i2, in_i2_b3, vrgout_b3, vl) ;
	vrgw_i3 = _vel_pvfmad_vvsvl(vrgw_i3, in_i3_b3, vrgout_b3, vl) ;
      }
      if(BATCH>=5) {
	const uint64_t in_i0_b4 = _vel_pack_f32a(pIn+4*inDim+i+0) ;
	const uint64_t in_i1_b4 = _vel_pack_f32a(pIn+4*inDim+i+1) ;
	const uint64_t in_i2_b4 = _vel_pack_f32a(pIn+4*inDim+i+2) ;
	const uint64_t in_i3_b4 = _vel_pack_f32a(pIn+4*inDim+i+3) ;
	vrgw_i0 = _vel_pvfmad_vvsvl(vrgw_i0, in_i0_b4, vrgout_b4, vl) ;
	vrgw_i1 = _vel_pvfmad_vvsvl(vrgw_i1, in_i1_b4, vrgout_b4, vl) ;
	vrgw_i2 = _vel_pvfmad_vvsvl(vrgw_i2, in_i2_b4, vrgout_b4, vl) ;
	vrgw_i3 = _vel_pvfmad_vvsvl(vrgw_i3, in_i3_b4, vrgout_b4, vl) ;
      }
      if(BATCH>=6) {
	const uint64_t in_i0_b5 = _vel_pack_f32a(pIn+5*inDim+i+0) ;
	const uint64_t in_i1_b5 = _vel_pack_f32a(pIn+5*inDim+i+1) ;
	const uint64_t in_i2_b5 = _vel_pack_f32a(pIn+5*inDim+i+2) ;
	const uint64_t in_i3_b5 = _vel_pack_f32a(pIn+5*inDim+i+3) ;
	vrgw_i0 = _vel_pvfmad_vvsvl(vrgw_i0, in_i0_b5, vrgout_b5, vl) ;
	vrgw_i1 = _vel_pvfmad_vvsvl(vrgw_i1, in_i1_b5, vrgout_b5, vl) ;
	vrgw_i2 = _vel_pvfmad_vvsvl(vrgw_i2, in_i2_b5, vrgout_b5, vl) ;
	vrgw_i3 = _vel_pvfmad_vvsvl(vrgw_i3, in_i3_b5, vrgout_b5, vl) ;
      }
      if(BATCH>=7) {
	const uint64_t in_i0_b6 = _vel_pack_f32a(pIn+6*inDim+i+0) ;
	const uint64_t in_i1_b6 = _vel_pack_f32a(pIn+6*inDim+i+1) ;
	const uint64_t in_i2_b6 = _vel_pack_f32a(pIn+6*inDim+i+2) ;
	const uint64_t in_i3_b6 = _vel_pack_f32a(pIn+6*inDim+i+3) ;
	vrgw_i0 = _vel_pvfmad_vvsvl(vrgw_i0, in_i0_b6, vrgout_b6, vl) ;
	vrgw_i1 = _vel_pvfmad_vvsvl(vrgw_i1, in_i1_b6, vrgout_b6, vl) ;
	vrgw_i2 = _vel_pvfmad_vvsvl(vrgw_i2, in_i2_b6, vrgout_b6, vl) ;
	vrgw_i3 = _vel_pvfmad_vvsvl(vrgw_i3, in_i3_b6, vrgout_b6, vl) ;
      }
      if(BATCH>=8) {
	const uint64_t in_i0_b7 = _vel_pack_f32a(pIn+7*inDim+i+0) ;
	const uint64_t in_i1_b7 = _vel_pack_f32a(pIn+7*inDim+i+1) ;
	const uint64_t in_i2_b7 = _vel_pack_f32a(pIn+7*inDim+i+2) ;
	const uint64_t in_i3_b7 = _vel_pack_f32a(pIn+7*inDim+i+3) ;
	vrgw_i0 = _vel_pvfmad_vvsvl(vrgw_i0, in_i0_b7, vrgout_b7, vl) ;
	vrgw_i1 = _vel_pvfmad_vvsvl(vrgw_i1, in_i1_b7, vrgout_b7, vl) ;
	vrgw_i2 = _vel_pvfmad_vvsvl(vrgw_i2, in_i2_b7, vrgout_b7, vl) ;
	vrgw_i3 = _vel_pvfmad_vvsvl(vrgw_i3, in_i3_b7, vrgout_b7, vl) ;
      }

      _vel_vst_vssl(vrgw_i0, 8, pGWeight+(i+0)*outDim+o, vl) ;
      _vel_vst_vssl(vrgw_i1, 8, pGWeight+(i+1)*outDim+o, vl) ;
      _vel_vst_vssl(vrgw_i2, 8, pGWeight+(i+2)*outDim+o, vl) ;
      _vel_vst_vssl(vrgw_i3, 8, pGWeight+(i+3)*outDim+o, vl) ;

      i+=4 ;
    }
    for(; i<nInDim; i+=8) {
      __vr vrgw_i0 ;
      __vr vrgw_i1 ;
      __vr vrgw_i2 ;
      __vr vrgw_i3 ;
      __vr vrgw_i4 ;
      __vr vrgw_i5 ;
      __vr vrgw_i6 ;
      __vr vrgw_i7 ;

      if(UPDATE) {
	vrgw_i0 = _vel_vld_vssl(8, pGWeight+(i+0)*outDim+o, vl) ;
	vrgw_i1 = _vel_vld_vssl(8, pGWeight+(i+1)*outDim+o, vl) ;
	vrgw_i2 = _vel_vld_vssl(8, pGWeight+(i+2)*outDim+o, vl) ;
	vrgw_i3 = _vel_vld_vssl(8, pGWeight+(i+3)*outDim+o, vl) ;
	vrgw_i4 = _vel_vld_vssl(8, pGWeight+(i+4)*outDim+o, vl) ;
	vrgw_i5 = _vel_vld_vssl(8, pGWeight+(i+5)*outDim+o, vl) ;
	vrgw_i6 = _vel_vld_vssl(8, pGWeight+(i+6)*outDim+o, vl) ;
	vrgw_i7 = _vel_vld_vssl(8, pGWeight+(i+7)*outDim+o, vl) ;

	const uint64_t in_i0_b0 = _vel_pack_f32a(pIn+0*inDim+i+0) ;
	const uint64_t in_i1_b0 = _vel_pack_f32a(pIn+0*inDim+i+1) ;
	const uint64_t in_i2_b0 = _vel_pack_f32a(pIn+0*inDim+i+2) ;
	const uint64_t in_i3_b0 = _vel_pack_f32a(pIn+0*inDim+i+3) ;
	const uint64_t in_i4_b0 = _vel_pack_f32a(pIn+0*inDim+i+4) ;
	const uint64_t in_i5_b0 = _vel_pack_f32a(pIn+0*inDim+i+5) ;
	const uint64_t in_i6_b0 = _vel_pack_f32a(pIn+0*inDim+i+6) ;
	const uint64_t in_i7_b0 = _vel_pack_f32a(pIn+0*inDim+i+7) ;
	vrgw_i0 = _vel_pvfmad_vvsvl(vrgw_i0, in_i0_b0, vrgout_b0, vl) ;
	vrgw_i1 = _vel_pvfmad_vvsvl(vrgw_i1, in_i1_b0, vrgout_b0, vl) ;
	vrgw_i2 = _vel_pvfmad_vvsvl(vrgw_i2, in_i2_b0, vrgout_b0, vl) ;
	vrgw_i3 = _vel_pvfmad_vvsvl(vrgw_i3, in_i3_b0, vrgout_b0, vl) ;
	vrgw_i4 = _vel_pvfmad_vvsvl(vrgw_i4, in_i4_b0, vrgout_b0, vl) ;
	vrgw_i5 = _vel_pvfmad_vvsvl(vrgw_i5, in_i5_b0, vrgout_b0, vl) ;
	vrgw_i6 = _vel_pvfmad_vvsvl(vrgw_i6, in_i6_b0, vrgout_b0, vl) ;
	vrgw_i7 = _vel_pvfmad_vvsvl(vrgw_i7, in_i7_b0, vrgout_b0, vl) ;
      }
      else {
	const uint64_t in_i0_b0 = _vel_pack_f32a(pIn+0*inDim+i+0) ;
	const uint64_t in_i1_b0 = _vel_pack_f32a(pIn+0*inDim+i+1) ;
	const uint64_t in_i2_b0 = _vel_pack_f32a(pIn+0*inDim+i+2) ;
	const uint64_t in_i3_b0 = _vel_pack_f32a(pIn+0*inDim+i+3) ;
	const uint64_t in_i4_b0 = _vel_pack_f32a(pIn+0*inDim+i+4) ;
	const uint64_t in_i5_b0 = _vel_pack_f32a(pIn+0*inDim+i+5) ;
	const uint64_t in_i6_b0 = _vel_pack_f32a(pIn+0*inDim+i+6) ;
	const uint64_t in_i7_b0 = _vel_pack_f32a(pIn+0*inDim+i+7) ;
	vrgw_i0 = _vel_pvfmul_vsvl(in_i0_b0, vrgout_b0, vl) ;
	vrgw_i1 = _vel_pvfmul_vsvl(in_i1_b0, vrgout_b0, vl) ;
	vrgw_i2 = _vel_pvfmul_vsvl(in_i2_b0, vrgout_b0, vl) ;
	vrgw_i3 = _vel_pvfmul_vsvl(in_i3_b0, vrgout_b0, vl) ;
	vrgw_i4 = _vel_pvfmul_vsvl(in_i4_b0, vrgout_b0, vl) ;
	vrgw_i5 = _vel_pvfmul_vsvl(in_i5_b0, vrgout_b0, vl) ;
	vrgw_i6 = _vel_pvfmul_vsvl(in_i6_b0, vrgout_b0, vl) ;
	vrgw_i7 = _vel_pvfmul_vsvl(in_i7_b0, vrgout_b0, vl) ;
      }

      if(BATCH>=2)  {
	const uint64_t in_i0_b1 = _vel_pack_f32a(pIn+1*inDim+i+0) ;
	const uint64_t in_i1_b1 = _vel_pack_f32a(pIn+1*inDim+i+1) ;
	const uint64_t in_i2_b1 = _vel_pack_f32a(pIn+1*inDim+i+2) ;
	const uint64_t in_i3_b1 = _vel_pack_f32a(pIn+1*inDim+i+3) ;
	const uint64_t in_i4_b1 = _vel_pack_f32a(pIn+1*inDim+i+4) ;
	const uint64_t in_i5_b1 = _vel_pack_f32a(pIn+1*inDim+i+5) ;
	const uint64_t in_i6_b1 = _vel_pack_f32a(pIn+1*inDim+i+6) ;
	const uint64_t in_i7_b1 = _vel_pack_f32a(pIn+1*inDim+i+7) ;
	vrgw_i0 = _vel_pvfmad_vvsvl(vrgw_i0, in_i0_b1, vrgout_b1, vl) ;
	vrgw_i1 = _vel_pvfmad_vvsvl(vrgw_i1, in_i1_b1, vrgout_b1, vl) ;
	vrgw_i2 = _vel_pvfmad_vvsvl(vrgw_i2, in_i2_b1, vrgout_b1, vl) ;
	vrgw_i3 = _vel_pvfmad_vvsvl(vrgw_i3, in_i3_b1, vrgout_b1, vl) ;
	vrgw_i4 = _vel_pvfmad_vvsvl(vrgw_i4, in_i4_b1, vrgout_b1, vl) ;
	vrgw_i5 = _vel_pvfmad_vvsvl(vrgw_i5, in_i5_b1, vrgout_b1, vl) ;
	vrgw_i6 = _vel_pvfmad_vvsvl(vrgw_i6, in_i6_b1, vrgout_b1, vl) ;
	vrgw_i7 = _vel_pvfmad_vvsvl(vrgw_i7, in_i7_b1, vrgout_b1, vl) ;
      }
      if(BATCH>=3) {
	const uint64_t in_i0_b2 = _vel_pack_f32a(pIn+2*inDim+i+0) ;
	const uint64_t in_i1_b2 = _vel_pack_f32a(pIn+2*inDim+i+1) ;
	const uint64_t in_i2_b2 = _vel_pack_f32a(pIn+2*inDim+i+2) ;
	const uint64_t in_i3_b2 = _vel_pack_f32a(pIn+2*inDim+i+3) ;
	const uint64_t in_i4_b2 = _vel_pack_f32a(pIn+2*inDim+i+4) ;
	const uint64_t in_i5_b2 = _vel_pack_f32a(pIn+2*inDim+i+5) ;
	const uint64_t in_i6_b2 = _vel_pack_f32a(pIn+2*inDim+i+6) ;
	const uint64_t in_i7_b2 = _vel_pack_f32a(pIn+2*inDim+i+7) ;
	vrgw_i0 = _vel_pvfmad_vvsvl(vrgw_i0, in_i0_b2, vrgout_b2, vl) ;
	vrgw_i1 = _vel_pvfmad_vvsvl(vrgw_i1, in_i1_b2, vrgout_b2, vl) ;
	vrgw_i2 = _vel_pvfmad_vvsvl(vrgw_i2, in_i2_b2, vrgout_b2, vl) ;
	vrgw_i3 = _vel_pvfmad_vvsvl(vrgw_i3, in_i3_b2, vrgout_b2, vl) ;
	vrgw_i4 = _vel_pvfmad_vvsvl(vrgw_i4, in_i4_b2, vrgout_b2, vl) ;
	vrgw_i5 = _vel_pvfmad_vvsvl(vrgw_i5, in_i5_b2, vrgout_b2, vl) ;
	vrgw_i6 = _vel_pvfmad_vvsvl(vrgw_i6, in_i6_b2, vrgout_b2, vl) ;
	vrgw_i7 = _vel_pvfmad_vvsvl(vrgw_i7, in_i7_b2, vrgout_b2, vl) ;
      }
      if(BATCH>=4) {
	const uint64_t in_i0_b3 = _vel_pack_f32a(pIn+3*inDim+i+0) ;
	const uint64_t in_i1_b3 = _vel_pack_f32a(pIn+3*inDim+i+1) ;
	const uint64_t in_i2_b3 = _vel_pack_f32a(pIn+3*inDim+i+2) ;
	const uint64_t in_i3_b3 = _vel_pack_f32a(pIn+3*inDim+i+3) ;
	const uint64_t in_i4_b3 = _vel_pack_f32a(pIn+3*inDim+i+4) ;
	const uint64_t in_i5_b3 = _vel_pack_f32a(pIn+3*inDim+i+5) ;
	const uint64_t in_i6_b3 = _vel_pack_f32a(pIn+3*inDim+i+6) ;
	const uint64_t in_i7_b3 = _vel_pack_f32a(pIn+3*inDim+i+7) ;
	vrgw_i0 = _vel_pvfmad_vvsvl(vrgw_i0, in_i0_b3, vrgout_b3, vl) ;
	vrgw_i1 = _vel_pvfmad_vvsvl(vrgw_i1, in_i1_b3, vrgout_b3, vl) ;
	vrgw_i2 = _vel_pvfmad_vvsvl(vrgw_i2, in_i2_b3, vrgout_b3, vl) ;
	vrgw_i3 = _vel_pvfmad_vvsvl(vrgw_i3, in_i3_b3, vrgout_b3, vl) ;
	vrgw_i4 = _vel_pvfmad_vvsvl(vrgw_i4, in_i4_b3, vrgout_b3, vl) ;
	vrgw_i5 = _vel_pvfmad_vvsvl(vrgw_i5, in_i5_b3, vrgout_b3, vl) ;
	vrgw_i6 = _vel_pvfmad_vvsvl(vrgw_i6, in_i6_b3, vrgout_b3, vl) ;
	vrgw_i7 = _vel_pvfmad_vvsvl(vrgw_i7, in_i7_b3, vrgout_b3, vl) ;
      }
      if(BATCH>=5) {
	const uint64_t in_i0_b4 = _vel_pack_f32a(pIn+4*inDim+i+0) ;
	const uint64_t in_i1_b4 = _vel_pack_f32a(pIn+4*inDim+i+1) ;
	const uint64_t in_i2_b4 = _vel_pack_f32a(pIn+4*inDim+i+2) ;
	const uint64_t in_i3_b4 = _vel_pack_f32a(pIn+4*inDim+i+3) ;
	const uint64_t in_i4_b4 = _vel_pack_f32a(pIn+4*inDim+i+4) ;
	const uint64_t in_i5_b4 = _vel_pack_f32a(pIn+4*inDim+i+5) ;
	const uint64_t in_i6_b4 = _vel_pack_f32a(pIn+4*inDim+i+6) ;
	const uint64_t in_i7_b4 = _vel_pack_f32a(pIn+4*inDim+i+7) ;
	vrgw_i0 = _vel_pvfmad_vvsvl(vrgw_i0, in_i0_b4, vrgout_b4, vl) ;
	vrgw_i1 = _vel_pvfmad_vvsvl(vrgw_i1, in_i1_b4, vrgout_b4, vl) ;
	vrgw_i2 = _vel_pvfmad_vvsvl(vrgw_i2, in_i2_b4, vrgout_b4, vl) ;
	vrgw_i3 = _vel_pvfmad_vvsvl(vrgw_i3, in_i3_b4, vrgout_b4, vl) ;
	vrgw_i4 = _vel_pvfmad_vvsvl(vrgw_i4, in_i4_b4, vrgout_b4, vl) ;
	vrgw_i5 = _vel_pvfmad_vvsvl(vrgw_i5, in_i5_b4, vrgout_b4, vl) ;
	vrgw_i6 = _vel_pvfmad_vvsvl(vrgw_i6, in_i6_b4, vrgout_b4, vl) ;
	vrgw_i7 = _vel_pvfmad_vvsvl(vrgw_i7, in_i7_b4, vrgout_b4, vl) ;
      }
      if(BATCH>=6) {
	const uint64_t in_i0_b5 = _vel_pack_f32a(pIn+5*inDim+i+0) ;
	const uint64_t in_i1_b5 = _vel_pack_f32a(pIn+5*inDim+i+1) ;
	const uint64_t in_i2_b5 = _vel_pack_f32a(pIn+5*inDim+i+2) ;
	const uint64_t in_i3_b5 = _vel_pack_f32a(pIn+5*inDim+i+3) ;
	const uint64_t in_i4_b5 = _vel_pack_f32a(pIn+5*inDim+i+4) ;
	const uint64_t in_i5_b5 = _vel_pack_f32a(pIn+5*inDim+i+5) ;
	const uint64_t in_i6_b5 = _vel_pack_f32a(pIn+5*inDim+i+6) ;
	const uint64_t in_i7_b5 = _vel_pack_f32a(pIn+5*inDim+i+7) ;
	vrgw_i0 = _vel_pvfmad_vvsvl(vrgw_i0, in_i0_b5, vrgout_b5, vl) ;
	vrgw_i1 = _vel_pvfmad_vvsvl(vrgw_i1, in_i1_b5, vrgout_b5, vl) ;
	vrgw_i2 = _vel_pvfmad_vvsvl(vrgw_i2, in_i2_b5, vrgout_b5, vl) ;
	vrgw_i3 = _vel_pvfmad_vvsvl(vrgw_i3, in_i3_b5, vrgout_b5, vl) ;
	vrgw_i4 = _vel_pvfmad_vvsvl(vrgw_i4, in_i4_b5, vrgout_b5, vl) ;
	vrgw_i5 = _vel_pvfmad_vvsvl(vrgw_i5, in_i5_b5, vrgout_b5, vl) ;
	vrgw_i6 = _vel_pvfmad_vvsvl(vrgw_i6, in_i6_b5, vrgout_b5, vl) ;
	vrgw_i7 = _vel_pvfmad_vvsvl(vrgw_i7, in_i7_b5, vrgout_b5, vl) ;
      }
      if(BATCH>=7) {
	const uint64_t in_i0_b6 = _vel_pack_f32a(pIn+6*inDim+i+0) ;
	const uint64_t in_i1_b6 = _vel_pack_f32a(pIn+6*inDim+i+1) ;
	const uint64_t in_i2_b6 = _vel_pack_f32a(pIn+6*inDim+i+2) ;
	const uint64_t in_i3_b6 = _vel_pack_f32a(pIn+6*inDim+i+3) ;
	const uint64_t in_i4_b6 = _vel_pack_f32a(pIn+6*inDim+i+4) ;
	const uint64_t in_i5_b6 = _vel_pack_f32a(pIn+6*inDim+i+5) ;
	const uint64_t in_i6_b6 = _vel_pack_f32a(pIn+6*inDim+i+6) ;
	const uint64_t in_i7_b6 = _vel_pack_f32a(pIn+6*inDim+i+7) ;
	vrgw_i0 = _vel_pvfmad_vvsvl(vrgw_i0, in_i0_b6, vrgout_b6, vl) ;
	vrgw_i1 = _vel_pvfmad_vvsvl(vrgw_i1, in_i1_b6, vrgout_b6, vl) ;
	vrgw_i2 = _vel_pvfmad_vvsvl(vrgw_i2, in_i2_b6, vrgout_b6, vl) ;
	vrgw_i3 = _vel_pvfmad_vvsvl(vrgw_i3, in_i3_b6, vrgout_b6, vl) ;
	vrgw_i4 = _vel_pvfmad_vvsvl(vrgw_i4, in_i4_b6, vrgout_b6, vl) ;
	vrgw_i5 = _vel_pvfmad_vvsvl(vrgw_i5, in_i5_b6, vrgout_b6, vl) ;
	vrgw_i6 = _vel_pvfmad_vvsvl(vrgw_i6, in_i6_b6, vrgout_b6, vl) ;
	vrgw_i7 = _vel_pvfmad_vvsvl(vrgw_i7, in_i7_b6, vrgout_b6, vl) ;
      }
      if(BATCH>=8) {
	const uint64_t in_i0_b7 = _vel_pack_f32a(pIn+7*inDim+i+0) ;
	const uint64_t in_i1_b7 = _vel_pack_f32a(pIn+7*inDim+i+1) ;
	const uint64_t in_i2_b7 = _vel_pack_f32a(pIn+7*inDim+i+2) ;
	const uint64_t in_i3_b7 = _vel_pack_f32a(pIn+7*inDim+i+3) ;
	const uint64_t in_i4_b7 = _vel_pack_f32a(pIn+7*inDim+i+4) ;
	const uint64_t in_i5_b7 = _vel_pack_f32a(pIn+7*inDim+i+5) ;
	const uint64_t in_i6_b7 = _vel_pack_f32a(pIn+7*inDim+i+6) ;
	const uint64_t in_i7_b7 = _vel_pack_f32a(pIn+7*inDim+i+7) ;
	vrgw_i0 = _vel_pvfmad_vvsvl(vrgw_i0, in_i0_b7, vrgout_b7, vl) ;
	vrgw_i1 = _vel_pvfmad_vvsvl(vrgw_i1, in_i1_b7, vrgout_b7, vl) ;
	vrgw_i2 = _vel_pvfmad_vvsvl(vrgw_i2, in_i2_b7, vrgout_b7, vl) ;
	vrgw_i3 = _vel_pvfmad_vvsvl(vrgw_i3, in_i3_b7, vrgout_b7, vl) ;
	vrgw_i4 = _vel_pvfmad_vvsvl(vrgw_i4, in_i4_b7, vrgout_b7, vl) ;
	vrgw_i5 = _vel_pvfmad_vvsvl(vrgw_i5, in_i5_b7, vrgout_b7, vl) ;
	vrgw_i6 = _vel_pvfmad_vvsvl(vrgw_i6, in_i6_b7, vrgout_b7, vl) ;
	vrgw_i7 = _vel_pvfmad_vvsvl(vrgw_i7, in_i7_b7, vrgout_b7, vl) ;
      }

      _vel_vst_vssl(vrgw_i0, 8, pGWeight+(i+0)*outDim+o, vl) ;
      _vel_vst_vssl(vrgw_i1, 8, pGWeight+(i+1)*outDim+o, vl) ;
      _vel_vst_vssl(vrgw_i2, 8, pGWeight+(i+2)*outDim+o, vl) ;
      _vel_vst_vssl(vrgw_i3, 8, pGWeight+(i+3)*outDim+o, vl) ;
      _vel_vst_vssl(vrgw_i4, 8, pGWeight+(i+4)*outDim+o, vl) ;
      _vel_vst_vssl(vrgw_i5, 8, pGWeight+(i+5)*outDim+o, vl) ;
      _vel_vst_vssl(vrgw_i6, 8, pGWeight+(i+6)*outDim+o, vl) ;
      _vel_vst_vssl(vrgw_i7, 8, pGWeight+(i+7)*outDim+o, vl) ;
    }
  }
}


extern "C"
vednnError_t vednnLinearBackwardWeight_o2X_woaligned(
    const uint64_t			inDim,
    const uint64_t			outDim,
    const uint64_t			nBatch,
    const void * 			pDataIn,
    const void * 			pDataGradOut,
    void * 				pDataGradWeight
#ifdef VEDNN_USE_OPENMP
    ,
    const uint64_t			inDimBegin,
    const uint64_t			inDimEnd
#endif
)
{
  const float * __restrict__ pIn       = (const float * __restrict__) pDataIn;
  const float * __restrict__ pGOut     = (const float * __restrict__) pDataGradOut;
  float * __restrict__ const pGWeight  = (float * __restrict__ const) pDataGradWeight;

#ifndef VEDNN_USE_OPENMP
    const uint64_t inDimBegin = 0 ;
    const uint64_t inDimEnd   = inDim ;
#endif

  int64_t n=0;
  int64_t batchRemain = nBatch % 8 ;

  switch( batchRemain ) {
  case 1 :
    func<1,false>(inDim, outDim, inDimEnd-inDimBegin,
	          pIn+inDimBegin+n*inDim, pGOut+n*outDim, pGWeight+inDimBegin*outDim) ;
    n+=1 ;
    break ;
  case 2 :
   func<2,false>(inDim, outDim, inDimEnd-inDimBegin,
	         pIn+inDimBegin+n*inDim, pGOut+n*outDim, pGWeight+inDimBegin*outDim) ;
    n+=2 ;
    break ;
  case 3 :
    func<3,false>(inDim, outDim, inDimEnd-inDimBegin,
	          pIn+inDimBegin+n*inDim, pGOut+n*outDim, pGWeight+inDimBegin*outDim) ;
    n+=3 ;
    break ;
  case 4 :
    func<4,false>(inDim, outDim, inDimEnd-inDimBegin,
	          pIn+inDimBegin+n*inDim, pGOut+n*outDim, pGWeight+inDimBegin*outDim) ;
    n+=4 ;
    break ;
  case 5 :
    func<5,false>(inDim, outDim, inDimEnd-inDimBegin,
	          pIn+inDimBegin+n*inDim, pGOut+n*outDim, pGWeight+inDimBegin*outDim) ;
    n+=5 ;
    break ;
  case 6 :
    func<6,false>(inDim, outDim, inDimEnd-inDimBegin,
	          pIn+inDimBegin+n*inDim, pGOut+n*outDim, pGWeight+inDimBegin*outDim) ;
    n+=6 ;
    break ;
  case 7 :
    func<7,false>(inDim, outDim, inDimEnd-inDimBegin,
	          pIn+inDimBegin+n*inDim, pGOut+n*outDim, pGWeight+inDimBegin*outDim) ;
    n+=7 ;
    break ;
  default :
    if( nBatch >= 8 ) {
      func<8,false>(inDim, outDim, inDimEnd-inDimBegin,
		    pIn+inDimBegin+n*inDim, pGOut+n*outDim, pGWeight+inDimBegin*outDim) ;
      n+=8 ;
    }
    break ;
  }

  for(; n<nBatch; n+=8) {
    func<8,true>(inDim, outDim, inDimEnd-inDimBegin,
		 pIn+inDimBegin+n*inDim, pGOut+n*outDim, pGWeight+inDimBegin*outDim) ;
  }

  return VEDNN_SUCCESS ;
}
