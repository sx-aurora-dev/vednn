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
  for(int64_t o=0; o<outDim; o+=VLEN) {
    const int64_t vl = outDim-o < VLEN ? outDim-o : VLEN ;

    __vr vrgout_b0 = _vel_vldu_vssl(4, pGOut+0*outDim+o, vl) ;
    __vr vrgout_b1 = _vel_vldu_vssl(4, pGOut+1*outDim+o, vl) ;
    __vr vrgout_b2 = _vel_vldu_vssl(4, pGOut+2*outDim+o, vl) ;
    __vr vrgout_b3 = _vel_vldu_vssl(4, pGOut+3*outDim+o, vl) ;
    __vr vrgout_b4 = _vel_vldu_vssl(4, pGOut+4*outDim+o, vl) ;
    __vr vrgout_b5 = _vel_vldu_vssl(4, pGOut+5*outDim+o, vl) ;
    __vr vrgout_b6 = _vel_vldu_vssl(4, pGOut+6*outDim+o, vl) ;
    __vr vrgout_b7 = _vel_vldu_vssl(4, pGOut+7*outDim+o, vl) ;

    __vr vrgoutP_b0 = _vel_vshf_vvvsl(vrgout_b0, vrgout_b0, VE_VSHUFFLE_YUZU, vl) ;
    __vr vrgoutP_b1 = _vel_vshf_vvvsl(vrgout_b1, vrgout_b1, VE_VSHUFFLE_YUZU, vl) ;
    __vr vrgoutP_b2 = _vel_vshf_vvvsl(vrgout_b2, vrgout_b2, VE_VSHUFFLE_YUZU, vl) ;
    __vr vrgoutP_b3 = _vel_vshf_vvvsl(vrgout_b3, vrgout_b3, VE_VSHUFFLE_YUZU, vl) ;
    __vr vrgoutP_b4 = _vel_vshf_vvvsl(vrgout_b4, vrgout_b4, VE_VSHUFFLE_YUZU, vl) ;
    __vr vrgoutP_b5 = _vel_vshf_vvvsl(vrgout_b5, vrgout_b5, VE_VSHUFFLE_YUZU, vl) ;
    __vr vrgoutP_b6 = _vel_vshf_vvvsl(vrgout_b6, vrgout_b6, VE_VSHUFFLE_YUZU, vl) ;
    __vr vrgoutP_b7 = _vel_vshf_vvvsl(vrgout_b7, vrgout_b7, VE_VSHUFFLE_YUZU, vl) ;

    int64_t i=0;
    if(nInDim & 0x1) {
      __vr vrgw ;

      if(UPDATE) {
	vrgw = _vel_vldu_vssl(4, pGWeight+(i+0)*outDim+o, vl) ;

	const float in_b0 = pIn[0*inDim+i] ;
	vrgw = _vel_vfmads_vvsvl(vrgw, in_b0, vrgout_b0, vl) ;
      }
      else {
	const float in_b0 = pIn[0*inDim+i] ;
	vrgw = _vel_vfmuls_vsvl(in_b0, vrgout_b0, vl) ;
      }

      if(BATCH>=2)  {
	const float in_b1 = pIn[1*inDim+i] ;
	vrgw = _vel_vfmads_vvsvl(vrgw, in_b1, vrgout_b1, vl) ;
      }
      if(BATCH>=3) {
	const float in_b2 = pIn[2*inDim+i] ;
	vrgw = _vel_vfmads_vvsvl(vrgw, in_b2, vrgout_b2, vl) ;
      }
      if(BATCH>=4) {
	const float in_b3 = pIn[3*inDim+i] ;
	vrgw = _vel_vfmads_vvsvl(vrgw, in_b3, vrgout_b3, vl) ;
      }
      if(BATCH>=5) {
	const float in_b4 = pIn[4*inDim+i] ;
	vrgw = _vel_vfmads_vvsvl(vrgw, in_b4, vrgout_b4, vl) ;
      }
      if(BATCH>=6) {
	const float in_b5 = pIn[5*inDim+i] ;
	vrgw = _vel_vfmads_vvsvl(vrgw, in_b5, vrgout_b5, vl) ;
      }
      if(BATCH>=7) {
	const float in_b6 = pIn[6*inDim+i] ;
	vrgw = _vel_vfmads_vvsvl(vrgw, in_b6, vrgout_b6, vl) ;
      }
      if(BATCH>=8) {
	const float in_b7 = pIn[7*inDim+i] ;
	vrgw = _vel_vfmads_vvsvl(vrgw, in_b7, vrgout_b7, vl) ;
      }

      _vel_vstu_vssl(vrgw, 4, pGWeight+i*outDim+o, vl) ;

      i+=1;
    }
    if((nInDim>>1) & 0x1) {
      __vr vrgw_i01 ;

      if(UPDATE) {
	__vr vrgw_i0 = _vel_vldu_vssl(4, pGWeight+(i+0)*outDim+o, vl) ;
	__vr vrgw_i1 = _vel_vldu_vssl(4, pGWeight+(i+1)*outDim+o, vl) ;

	vrgw_i01 = _vel_vshf_vvvsl(vrgw_i0, vrgw_i1, VE_VSHUFFLE_YUZU, vl) ;

	const uint64_t in_i01_b0 = _vel_pack_f32p(pIn+0*inDim+i+0, pIn+0*inDim+i+1) ;
	vrgw_i01 = _vel_pvfmad_vvsvl(vrgw_i01, in_i01_b0, vrgoutP_b0, vl) ;
      }
      else {
	const uint64_t in_i01_b0 = _vel_pack_f32p(pIn+0*inDim+i+0, pIn+0*inDim+i+1) ;
	vrgw_i01 = _vel_pvfmul_vsvl(in_i01_b0, vrgoutP_b0, vl) ;
      }

      if(BATCH>=2) {
	const uint64_t in_i01_b1 = _vel_pack_f32p(pIn+1*inDim+i+0, pIn+1*inDim+i+1) ;
	vrgw_i01 = _vel_pvfmad_vvsvl(vrgw_i01, in_i01_b1, vrgoutP_b1, vl) ;
      }
      if(BATCH>=3) {
	const uint64_t in_i01_b2 = _vel_pack_f32p(pIn+2*inDim+i+0, pIn+2*inDim+i+1) ;
	vrgw_i01 = _vel_pvfmad_vvsvl(vrgw_i01, in_i01_b2, vrgoutP_b2, vl) ;
      }
      if(BATCH>=4) {
	const uint64_t in_i01_b3 = _vel_pack_f32p(pIn+3*inDim+i+0, pIn+3*inDim+i+1) ;
	vrgw_i01 = _vel_pvfmad_vvsvl(vrgw_i01, in_i01_b3, vrgoutP_b3, vl) ;
      }
      if(BATCH>=5) {
	const uint64_t in_i01_b4 = _vel_pack_f32p(pIn+4*inDim+i+0, pIn+4*inDim+i+1) ;
	vrgw_i01 = _vel_pvfmad_vvsvl(vrgw_i01, in_i01_b4, vrgoutP_b4, vl) ;
      }
      if(BATCH>=6) {
	const uint64_t in_i01_b5 = _vel_pack_f32p(pIn+5*inDim+i+0, pIn+5*inDim+i+1) ;
	vrgw_i01 = _vel_pvfmad_vvsvl(vrgw_i01, in_i01_b5, vrgoutP_b5, vl) ;
      }
      if(BATCH>=7) {
	const uint64_t in_i01_b6 = _vel_pack_f32p(pIn+6*inDim+i+0, pIn+6*inDim+i+1) ;
	vrgw_i01 = _vel_pvfmad_vvsvl(vrgw_i01, in_i01_b6, vrgoutP_b6, vl) ;
      }
      if(BATCH>=8) {
	const uint64_t in_i01_b7 = _vel_pack_f32p(pIn+7*inDim+i+0, pIn+7*inDim+i+1) ;
	vrgw_i01 = _vel_pvfmad_vvsvl(vrgw_i01, in_i01_b7, vrgoutP_b7, vl) ;
      }

      _vel_vstu_vssl(vrgw_i01, 4, pGWeight+(i+0)*outDim+o, vl) ;
      _vel_vstl_vssl(vrgw_i01, 4, pGWeight+(i+1)*outDim+o, vl) ;

      i+=2 ;
    }
    if((nInDim>>2) & 0x1) {
      __vr vrgw_i01 ;
      __vr vrgw_i23 ;

      if(UPDATE) {
	__vr vrgw_i0 = _vel_vldu_vssl(4, pGWeight+(i+0)*outDim+o, vl) ;
	__vr vrgw_i1 = _vel_vldu_vssl(4, pGWeight+(i+1)*outDim+o, vl) ;
	__vr vrgw_i2 = _vel_vldu_vssl(4, pGWeight+(i+2)*outDim+o, vl) ;
	__vr vrgw_i3 = _vel_vldu_vssl(4, pGWeight+(i+3)*outDim+o, vl) ;

	vrgw_i01 = _vel_vshf_vvvsl(vrgw_i0, vrgw_i1, VE_VSHUFFLE_YUZU, vl) ;
	vrgw_i23 = _vel_vshf_vvvsl(vrgw_i2, vrgw_i3, VE_VSHUFFLE_YUZU, vl) ;

	const uint64_t in_i01_b0 = _vel_pack_f32p(pIn+0*inDim+i+0, pIn+0*inDim+i+1) ;
	const uint64_t in_i23_b0 = _vel_pack_f32p(pIn+0*inDim+i+2, pIn+0*inDim+i+3) ;
	vrgw_i01 = _vel_pvfmad_vvsvl(vrgw_i01, in_i01_b0, vrgoutP_b0, vl) ;
	vrgw_i23 = _vel_pvfmad_vvsvl(vrgw_i23, in_i23_b0, vrgoutP_b0, vl) ;
      }
      else {
	const uint64_t in_i01_b0 = _vel_pack_f32p(pIn+0*inDim+i+0, pIn+0*inDim+i+1) ;
	const uint64_t in_i23_b0 = _vel_pack_f32p(pIn+0*inDim+i+2, pIn+0*inDim+i+3) ;
	vrgw_i01 = _vel_pvfmul_vsvl(in_i01_b0, vrgoutP_b0, vl) ;
	vrgw_i23 = _vel_pvfmul_vsvl(in_i23_b0, vrgoutP_b0, vl) ;
      }

      if(BATCH>=2) {
	const uint64_t in_i01_b1 = _vel_pack_f32p(pIn+1*inDim+i+0, pIn+1*inDim+i+1) ;
	const uint64_t in_i23_b1 = _vel_pack_f32p(pIn+1*inDim+i+2, pIn+1*inDim+i+3) ;
	vrgw_i01 = _vel_pvfmad_vvsvl(vrgw_i01, in_i01_b1, vrgoutP_b1, vl) ;
	vrgw_i23 = _vel_pvfmad_vvsvl(vrgw_i23, in_i23_b1, vrgoutP_b1, vl) ;
      }
      if(BATCH>=3) {
	const uint64_t in_i01_b2 = _vel_pack_f32p(pIn+2*inDim+i+0, pIn+2*inDim+i+1) ;
	const uint64_t in_i23_b2 = _vel_pack_f32p(pIn+2*inDim+i+2, pIn+2*inDim+i+3) ;
	vrgw_i01 = _vel_pvfmad_vvsvl(vrgw_i01, in_i01_b2, vrgoutP_b2, vl) ;
	vrgw_i23 = _vel_pvfmad_vvsvl(vrgw_i23, in_i23_b2, vrgoutP_b2, vl) ;
      }
      if(BATCH>=4) {
	const uint64_t in_i01_b3 = _vel_pack_f32p(pIn+3*inDim+i+0, pIn+3*inDim+i+1) ;
	const uint64_t in_i23_b3 = _vel_pack_f32p(pIn+3*inDim+i+2, pIn+3*inDim+i+3) ;
	vrgw_i01 = _vel_pvfmad_vvsvl(vrgw_i01, in_i01_b3, vrgoutP_b3, vl) ;
	vrgw_i23 = _vel_pvfmad_vvsvl(vrgw_i23, in_i23_b3, vrgoutP_b3, vl) ;
      }
      if(BATCH>=5) {
	const uint64_t in_i01_b4 = _vel_pack_f32p(pIn+4*inDim+i+0, pIn+4*inDim+i+1) ;
	const uint64_t in_i23_b4 = _vel_pack_f32p(pIn+4*inDim+i+2, pIn+4*inDim+i+3) ;
	vrgw_i01 = _vel_pvfmad_vvsvl(vrgw_i01, in_i01_b4, vrgoutP_b4, vl) ;
	vrgw_i23 = _vel_pvfmad_vvsvl(vrgw_i23, in_i23_b4, vrgoutP_b4, vl) ;
      }
      if(BATCH>=6) {
	const uint64_t in_i01_b5 = _vel_pack_f32p(pIn+5*inDim+i+0, pIn+5*inDim+i+1) ;
	const uint64_t in_i23_b5 = _vel_pack_f32p(pIn+5*inDim+i+2, pIn+5*inDim+i+3) ;
	vrgw_i01 = _vel_pvfmad_vvsvl(vrgw_i01, in_i01_b5, vrgoutP_b5, vl) ;
	vrgw_i23 = _vel_pvfmad_vvsvl(vrgw_i23, in_i23_b5, vrgoutP_b5, vl) ;
      }
      if(BATCH>=7) {
	const uint64_t in_i01_b6 = _vel_pack_f32p(pIn+6*inDim+i+0, pIn+6*inDim+i+1) ;
	const uint64_t in_i23_b6 = _vel_pack_f32p(pIn+6*inDim+i+2, pIn+6*inDim+i+3) ;
	vrgw_i01 = _vel_pvfmad_vvsvl(vrgw_i01, in_i01_b6, vrgoutP_b6, vl) ;
	vrgw_i23 = _vel_pvfmad_vvsvl(vrgw_i23, in_i23_b6, vrgoutP_b6, vl) ;
      }
      if(BATCH>=8) {
	const uint64_t in_i01_b7 = _vel_pack_f32p(pIn+7*inDim+i+0, pIn+7*inDim+i+1) ;
	const uint64_t in_i23_b7 = _vel_pack_f32p(pIn+7*inDim+i+2, pIn+7*inDim+i+3) ;
	vrgw_i01 = _vel_pvfmad_vvsvl(vrgw_i01, in_i01_b7, vrgoutP_b7, vl) ;
	vrgw_i23 = _vel_pvfmad_vvsvl(vrgw_i23, in_i23_b7, vrgoutP_b7, vl) ;
      }

      _vel_vstu_vssl(vrgw_i01, 4, pGWeight+(i+0)*outDim+o, vl) ;
      _vel_vstl_vssl(vrgw_i01, 4, pGWeight+(i+1)*outDim+o, vl) ;
      _vel_vstu_vssl(vrgw_i23, 4, pGWeight+(i+2)*outDim+o, vl) ;
      _vel_vstl_vssl(vrgw_i23, 4, pGWeight+(i+3)*outDim+o, vl) ;

      i+=4 ;
    }
    for(; i<nInDim; i+=8) {
      __vr vrgw_i01 ;
      __vr vrgw_i23 ;
      __vr vrgw_i45 ;
      __vr vrgw_i67 ;

      if(UPDATE) {
	__vr vrgw_i0 = _vel_vldu_vssl(4, pGWeight+(i+0)*outDim+o, vl) ;
	__vr vrgw_i1 = _vel_vldu_vssl(4, pGWeight+(i+1)*outDim+o, vl) ;
	__vr vrgw_i2 = _vel_vldu_vssl(4, pGWeight+(i+2)*outDim+o, vl) ;
	__vr vrgw_i3 = _vel_vldu_vssl(4, pGWeight+(i+3)*outDim+o, vl) ;
	__vr vrgw_i4 = _vel_vldu_vssl(4, pGWeight+(i+4)*outDim+o, vl) ;
	__vr vrgw_i5 = _vel_vldu_vssl(4, pGWeight+(i+5)*outDim+o, vl) ;
	__vr vrgw_i6 = _vel_vldu_vssl(4, pGWeight+(i+6)*outDim+o, vl) ;
	__vr vrgw_i7 = _vel_vldu_vssl(4, pGWeight+(i+7)*outDim+o, vl) ;

	vrgw_i01 = _vel_vshf_vvvsl(vrgw_i0, vrgw_i1, VE_VSHUFFLE_YUZU, vl) ;
	vrgw_i23 = _vel_vshf_vvvsl(vrgw_i2, vrgw_i3, VE_VSHUFFLE_YUZU, vl) ;
	vrgw_i45 = _vel_vshf_vvvsl(vrgw_i4, vrgw_i5, VE_VSHUFFLE_YUZU, vl) ;
	vrgw_i67 = _vel_vshf_vvvsl(vrgw_i6, vrgw_i7, VE_VSHUFFLE_YUZU, vl) ;

	const uint64_t in_i01_b0 = _vel_pack_f32p(pIn+0*inDim+i+0, pIn+0*inDim+i+1) ;
	const uint64_t in_i23_b0 = _vel_pack_f32p(pIn+0*inDim+i+2, pIn+0*inDim+i+3) ;
	const uint64_t in_i45_b0 = _vel_pack_f32p(pIn+0*inDim+i+4, pIn+0*inDim+i+5) ;
	const uint64_t in_i67_b0 = _vel_pack_f32p(pIn+0*inDim+i+6, pIn+0*inDim+i+7) ;
	vrgw_i01 = _vel_pvfmad_vvsvl(vrgw_i01, in_i01_b0, vrgoutP_b0, vl) ;
	vrgw_i23 = _vel_pvfmad_vvsvl(vrgw_i23, in_i23_b0, vrgoutP_b0, vl) ;
	vrgw_i45 = _vel_pvfmad_vvsvl(vrgw_i45, in_i45_b0, vrgoutP_b0, vl) ;
	vrgw_i67 = _vel_pvfmad_vvsvl(vrgw_i67, in_i67_b0, vrgoutP_b0, vl) ;
      }
      else {
	const uint64_t in_i01_b0 = _vel_pack_f32p(pIn+0*inDim+i+0, pIn+0*inDim+i+1) ;
	const uint64_t in_i23_b0 = _vel_pack_f32p(pIn+0*inDim+i+2, pIn+0*inDim+i+3) ;
	const uint64_t in_i45_b0 = _vel_pack_f32p(pIn+0*inDim+i+4, pIn+0*inDim+i+5) ;
	const uint64_t in_i67_b0 = _vel_pack_f32p(pIn+0*inDim+i+6, pIn+0*inDim+i+7) ;
	vrgw_i01 = _vel_pvfmul_vsvl(in_i01_b0, vrgoutP_b0, vl) ;
	vrgw_i23 = _vel_pvfmul_vsvl(in_i23_b0, vrgoutP_b0, vl) ;
	vrgw_i45 = _vel_pvfmul_vsvl(in_i45_b0, vrgoutP_b0, vl) ;
	vrgw_i67 = _vel_pvfmul_vsvl(in_i67_b0, vrgoutP_b0, vl) ;
      }

      if(BATCH>=2) {
	const uint64_t in_i01_b1 = _vel_pack_f32p(pIn+1*inDim+i+0, pIn+1*inDim+i+1) ;
	const uint64_t in_i23_b1 = _vel_pack_f32p(pIn+1*inDim+i+2, pIn+1*inDim+i+3) ;
	const uint64_t in_i45_b1 = _vel_pack_f32p(pIn+1*inDim+i+4, pIn+1*inDim+i+5) ;
	const uint64_t in_i67_b1 = _vel_pack_f32p(pIn+1*inDim+i+6, pIn+1*inDim+i+7) ;
	vrgw_i01 = _vel_pvfmad_vvsvl(vrgw_i01, in_i01_b1, vrgoutP_b1, vl) ;
	vrgw_i23 = _vel_pvfmad_vvsvl(vrgw_i23, in_i23_b1, vrgoutP_b1, vl) ;
	vrgw_i45 = _vel_pvfmad_vvsvl(vrgw_i45, in_i45_b1, vrgoutP_b1, vl) ;
	vrgw_i67 = _vel_pvfmad_vvsvl(vrgw_i67, in_i67_b1, vrgoutP_b1, vl) ;
      }
      if(BATCH>=3) {
	const uint64_t in_i01_b2 = _vel_pack_f32p(pIn+2*inDim+i+0, pIn+2*inDim+i+1) ;
	const uint64_t in_i23_b2 = _vel_pack_f32p(pIn+2*inDim+i+2, pIn+2*inDim+i+3) ;
	const uint64_t in_i45_b2 = _vel_pack_f32p(pIn+2*inDim+i+4, pIn+2*inDim+i+5) ;
	const uint64_t in_i67_b2 = _vel_pack_f32p(pIn+2*inDim+i+6, pIn+2*inDim+i+7) ;
	vrgw_i01 = _vel_pvfmad_vvsvl(vrgw_i01, in_i01_b2, vrgoutP_b2, vl) ;
	vrgw_i23 = _vel_pvfmad_vvsvl(vrgw_i23, in_i23_b2, vrgoutP_b2, vl) ;
	vrgw_i45 = _vel_pvfmad_vvsvl(vrgw_i45, in_i45_b2, vrgoutP_b2, vl) ;
	vrgw_i67 = _vel_pvfmad_vvsvl(vrgw_i67, in_i67_b2, vrgoutP_b2, vl) ;
      }
      if(BATCH>=4) {
	const uint64_t in_i01_b3 = _vel_pack_f32p(pIn+3*inDim+i+0, pIn+3*inDim+i+1) ;
	const uint64_t in_i23_b3 = _vel_pack_f32p(pIn+3*inDim+i+2, pIn+3*inDim+i+3) ;
	const uint64_t in_i45_b3 = _vel_pack_f32p(pIn+3*inDim+i+4, pIn+3*inDim+i+5) ;
	const uint64_t in_i67_b3 = _vel_pack_f32p(pIn+3*inDim+i+6, pIn+3*inDim+i+7) ;
	vrgw_i01 = _vel_pvfmad_vvsvl(vrgw_i01, in_i01_b3, vrgoutP_b3, vl) ;
	vrgw_i23 = _vel_pvfmad_vvsvl(vrgw_i23, in_i23_b3, vrgoutP_b3, vl) ;
	vrgw_i45 = _vel_pvfmad_vvsvl(vrgw_i45, in_i45_b3, vrgoutP_b3, vl) ;
	vrgw_i67 = _vel_pvfmad_vvsvl(vrgw_i67, in_i67_b3, vrgoutP_b3, vl) ;
      }
      if(BATCH>=5) {
	const uint64_t in_i01_b4 = _vel_pack_f32p(pIn+4*inDim+i+0, pIn+4*inDim+i+1) ;
	const uint64_t in_i23_b4 = _vel_pack_f32p(pIn+4*inDim+i+2, pIn+4*inDim+i+3) ;
	const uint64_t in_i45_b4 = _vel_pack_f32p(pIn+4*inDim+i+4, pIn+4*inDim+i+5) ;
	const uint64_t in_i67_b4 = _vel_pack_f32p(pIn+4*inDim+i+6, pIn+4*inDim+i+7) ;
	vrgw_i01 = _vel_pvfmad_vvsvl(vrgw_i01, in_i01_b4, vrgoutP_b4, vl) ;
	vrgw_i23 = _vel_pvfmad_vvsvl(vrgw_i23, in_i23_b4, vrgoutP_b4, vl) ;
	vrgw_i45 = _vel_pvfmad_vvsvl(vrgw_i45, in_i45_b4, vrgoutP_b4, vl) ;
	vrgw_i67 = _vel_pvfmad_vvsvl(vrgw_i67, in_i67_b4, vrgoutP_b4, vl) ;
      }
      if(BATCH>=6) {
	const uint64_t in_i01_b5 = _vel_pack_f32p(pIn+5*inDim+i+0, pIn+5*inDim+i+1) ;
	const uint64_t in_i23_b5 = _vel_pack_f32p(pIn+5*inDim+i+2, pIn+5*inDim+i+3) ;
	const uint64_t in_i45_b5 = _vel_pack_f32p(pIn+5*inDim+i+4, pIn+5*inDim+i+5) ;
	const uint64_t in_i67_b5 = _vel_pack_f32p(pIn+5*inDim+i+6, pIn+5*inDim+i+7) ;
	vrgw_i01 = _vel_pvfmad_vvsvl(vrgw_i01, in_i01_b5, vrgoutP_b5, vl) ;
	vrgw_i23 = _vel_pvfmad_vvsvl(vrgw_i23, in_i23_b5, vrgoutP_b5, vl) ;
	vrgw_i45 = _vel_pvfmad_vvsvl(vrgw_i45, in_i45_b5, vrgoutP_b5, vl) ;
	vrgw_i67 = _vel_pvfmad_vvsvl(vrgw_i67, in_i67_b5, vrgoutP_b5, vl) ;
      }
      if(BATCH>=7) {
	const uint64_t in_i01_b6 = _vel_pack_f32p(pIn+6*inDim+i+0, pIn+6*inDim+i+1) ;
	const uint64_t in_i23_b6 = _vel_pack_f32p(pIn+6*inDim+i+2, pIn+6*inDim+i+3) ;
	const uint64_t in_i45_b6 = _vel_pack_f32p(pIn+6*inDim+i+4, pIn+6*inDim+i+5) ;
	const uint64_t in_i67_b6 = _vel_pack_f32p(pIn+6*inDim+i+6, pIn+6*inDim+i+7) ;
	vrgw_i01 = _vel_pvfmad_vvsvl(vrgw_i01, in_i01_b6, vrgoutP_b6, vl) ;
	vrgw_i23 = _vel_pvfmad_vvsvl(vrgw_i23, in_i23_b6, vrgoutP_b6, vl) ;
	vrgw_i45 = _vel_pvfmad_vvsvl(vrgw_i45, in_i45_b6, vrgoutP_b6, vl) ;
	vrgw_i67 = _vel_pvfmad_vvsvl(vrgw_i67, in_i67_b6, vrgoutP_b6, vl) ;
      }
      if(BATCH>=8) {
	const uint64_t in_i01_b7 = _vel_pack_f32p(pIn+7*inDim+i+0, pIn+7*inDim+i+1) ;
	const uint64_t in_i23_b7 = _vel_pack_f32p(pIn+7*inDim+i+2, pIn+7*inDim+i+3) ;
	const uint64_t in_i45_b7 = _vel_pack_f32p(pIn+7*inDim+i+4, pIn+7*inDim+i+5) ;
	const uint64_t in_i67_b7 = _vel_pack_f32p(pIn+7*inDim+i+6, pIn+7*inDim+i+7) ;
	vrgw_i01 = _vel_pvfmad_vvsvl(vrgw_i01, in_i01_b7, vrgoutP_b7, vl) ;
	vrgw_i23 = _vel_pvfmad_vvsvl(vrgw_i23, in_i23_b7, vrgoutP_b7, vl) ;
	vrgw_i45 = _vel_pvfmad_vvsvl(vrgw_i45, in_i45_b7, vrgoutP_b7, vl) ;
	vrgw_i67 = _vel_pvfmad_vvsvl(vrgw_i67, in_i67_b7, vrgoutP_b7, vl) ;
      }

      _vel_vstu_vssl(vrgw_i01, 4, pGWeight+(i+0)*outDim+o, vl) ;
      _vel_vstl_vssl(vrgw_i01, 4, pGWeight+(i+1)*outDim+o, vl) ;
      _vel_vstu_vssl(vrgw_i23, 4, pGWeight+(i+2)*outDim+o, vl) ;
      _vel_vstl_vssl(vrgw_i23, 4, pGWeight+(i+3)*outDim+o, vl) ;
      _vel_vstu_vssl(vrgw_i45, 4, pGWeight+(i+4)*outDim+o, vl) ;
      _vel_vstl_vssl(vrgw_i45, 4, pGWeight+(i+5)*outDim+o, vl) ;
      _vel_vstu_vssl(vrgw_i67, 4, pGWeight+(i+6)*outDim+o, vl) ;
      _vel_vstl_vssl(vrgw_i67, 4, pGWeight+(i+7)*outDim+o, vl) ;
    }
  }
}


extern "C"
vednnError_t vednnLinearBackwardWeight_default(
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


#if 0 // reference code
vednnError_t vednnLinearBackwardWeight_default(
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

  for(int64_t i=inDimBegin; i<inDimEnd; i++) {
    for(int64_t o=0; o<outDim; o++) {
      pGWeight[i*outDim+o] = 0.f ;
    }
  }

  for(int64_t i=inDimBegin; i<inDimEnd; i++) {
    for(int64_t b=0; b<nBatch; b++) {
      const float in = pIn[b*inDim+i] ;
      for(int64_t o=0; o<outDim; o++) {
	pGWeight[i*outDim+o] += in * pGOut[b*outDim+o] ;
      }
    }
  }

  return VEDNN_SUCCESS ;
}
#endif
