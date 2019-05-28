#include <stdint.h>
#include <float.h>

#include <stdio.h>

#include "vednn.h"

#include "veintrin.h"
#define VLEN	(256)

static inline void b1(
  const uint64_t	inDim,
  const uint64_t	outDim,
  const float * 	pIn,
  const float * 	pWeight,
  float * 		pOut
)
{
  for(int64_t o=0; o<outDim; o+=VLEN) {
    const int64_t vl = outDim-o < VLEN ? outDim - o : VLEN ;
    _ve_lvl(vl) ;
    __vr vrsum_b0 = _ve_vbrd_vs_i64(0UL) ;

    int64_t i=0;
    if((inDim & 0x01)==1) {
      __vr vrw = _ve_vldu_vss(4, pWeight+i*outDim+o) ;
      vrsum_b0 = _ve_vfmads_vvsv(vrsum_b0, pIn[0*inDim+i], vrw) ;
      i+=1 ;
    }
    if(((inDim>>1) & 0x01)==1) {
      __vr vrw_i0 = _ve_vldu_vss(4, pWeight+(i  )*outDim+o) ;
      __vr vrw_i1 = _ve_vldu_vss(4, pWeight+(i+1)*outDim+o) ;

      __vr vrw_i01 = _ve_vshf_vvvs(vrw_i0, vrw_i1, VE_VSHUFFLE_YUZU) ;

      const uint64_t in_i01_b0 = _ve_pack_f32p(pIn+0*inDim+i+0, pIn+0*inDim+i+1) ;
      vrsum_b0 = _ve_pvfmad_vvsv(vrsum_b0, in_i01_b0, vrw_i01) ;
      i+=2 ;
    }
    for(; i<inDim; i+=4 ) {
      __vr vrw_i0 = _ve_vldu_vss(4, pWeight+(i  )*outDim+o) ;
      __vr vrw_i1 = _ve_vldu_vss(4, pWeight+(i+1)*outDim+o) ;
      __vr vrw_i2 = _ve_vldu_vss(4, pWeight+(i+2)*outDim+o) ;
      __vr vrw_i3 = _ve_vldu_vss(4, pWeight+(i+3)*outDim+o) ;

      __vr vrw_i01 = _ve_vshf_vvvs(vrw_i0, vrw_i1, VE_VSHUFFLE_YUZU) ;
      __vr vrw_i23 = _ve_vshf_vvvs(vrw_i2, vrw_i3, VE_VSHUFFLE_YUZU) ;

      const uint64_t in_i01_b0 = _ve_pack_f32p(pIn+0*inDim+i+0, pIn+0*inDim+i+1) ;
      vrsum_b0 = _ve_pvfmad_vvsv(vrsum_b0, in_i01_b0, vrw_i01) ;
      const uint64_t in_i23_b0 = _ve_pack_f32p(pIn+0*inDim+i+2, pIn+0*inDim+i+3) ;
      vrsum_b0 = _ve_pvfmad_vvsv(vrsum_b0, in_i23_b0, vrw_i23) ;
    }

    _ve_vstu_vss(_ve_vfadds_vvv(vrsum_b0, _ve_vsll_vvs(vrsum_b0,32)), 4, pOut+0*outDim+o) ;
  }
}

static inline void b2(
  const uint64_t	inDim,
  const uint64_t	outDim,
  const float * 	pIn,
  const float * 	pWeight,
  float * 		pOut
)
{
  for(int64_t o=0; o<outDim; o+=VLEN) {
    const int64_t vl = outDim-o < VLEN ? outDim - o : VLEN ;
    _ve_lvl(vl) ;
    __vr vrsum_b0 = _ve_vbrd_vs_i64(0UL) ;
    __vr vrsum_b1 = _ve_vbrd_vs_i64(0UL) ;

    int64_t i=0;
    if((inDim & 0x01)==1) {
      __vr vrw = _ve_vldu_vss(4, pWeight+i*outDim+o) ;
      vrsum_b0 = _ve_vfmads_vvsv(vrsum_b0, pIn[0*inDim+i], vrw) ;
      vrsum_b1 = _ve_vfmads_vvsv(vrsum_b1, pIn[1*inDim+i], vrw) ;

      i+=1 ;
    }
    if(((inDim>>1) & 0x01)==1) {
      __vr vrw_i0 = _ve_vldu_vss(4, pWeight+(i  )*outDim+o) ;
      __vr vrw_i1 = _ve_vldu_vss(4, pWeight+(i+1)*outDim+o) ;

      __vr vrw_i01 = _ve_vshf_vvvs(vrw_i0, vrw_i1, VE_VSHUFFLE_YUZU) ;

      const uint64_t in_i01_b0 = _ve_pack_f32p(pIn+0*inDim+i+0, pIn+0*inDim+i+1) ;
      const uint64_t in_i01_b1 = _ve_pack_f32p(pIn+1*inDim+i+0, pIn+1*inDim+i+1) ;

      vrsum_b0 = _ve_pvfmad_vvsv(vrsum_b0, in_i01_b0, vrw_i01) ;
      vrsum_b1 = _ve_pvfmad_vvsv(vrsum_b1, in_i01_b1, vrw_i01) ;

      i+=2 ;
    }
    for(; i<inDim; i+=4 ) {
      __vr vrw_i0 = _ve_vldu_vss(4, pWeight+(i  )*outDim+o) ;
      __vr vrw_i1 = _ve_vldu_vss(4, pWeight+(i+1)*outDim+o) ;
      __vr vrw_i2 = _ve_vldu_vss(4, pWeight+(i+2)*outDim+o) ;
      __vr vrw_i3 = _ve_vldu_vss(4, pWeight+(i+3)*outDim+o) ;

      __vr vrw_i01 = _ve_vshf_vvvs(vrw_i0, vrw_i1, VE_VSHUFFLE_YUZU) ;
      __vr vrw_i23 = _ve_vshf_vvvs(vrw_i2, vrw_i3, VE_VSHUFFLE_YUZU) ;

      const uint64_t in_i01_b0 = _ve_pack_f32p(pIn+0*inDim+i+0, pIn+0*inDim+i+1) ;
      const uint64_t in_i01_b1 = _ve_pack_f32p(pIn+1*inDim+i+0, pIn+1*inDim+i+1) ;

      vrsum_b0 = _ve_pvfmad_vvsv(vrsum_b0, in_i01_b0, vrw_i01) ;
      vrsum_b1 = _ve_pvfmad_vvsv(vrsum_b1, in_i01_b1, vrw_i01) ;

      const uint64_t in_i23_b0 = _ve_pack_f32p(pIn+0*inDim+i+2, pIn+0*inDim+i+3) ;
      const uint64_t in_i23_b1 = _ve_pack_f32p(pIn+1*inDim+i+2, pIn+1*inDim+i+3) ;

      vrsum_b0 = _ve_pvfmad_vvsv(vrsum_b0, in_i23_b0, vrw_i23) ;
      vrsum_b1 = _ve_pvfmad_vvsv(vrsum_b1, in_i23_b1, vrw_i23) ;
    }

    _ve_vstu_vss(_ve_vfadds_vvv(vrsum_b0, _ve_vsll_vvs(vrsum_b0,32)), 4, pOut+0*outDim+o) ;
    _ve_vstu_vss(_ve_vfadds_vvv(vrsum_b1, _ve_vsll_vvs(vrsum_b1,32)), 4, pOut+1*outDim+o) ;
  }
}

static inline void b3(
  const uint64_t	inDim,
  const uint64_t	outDim,
  const float * 	pIn,
  const float * 	pWeight,
  float * 		pOut
)
{
  for(int64_t o=0; o<outDim; o+=VLEN) {
    const int64_t vl = outDim-o < VLEN ? outDim - o : VLEN ;
    _ve_lvl(vl) ;
    __vr vrsum_b0 = _ve_vbrd_vs_i64(0UL) ;
    __vr vrsum_b1 = _ve_vbrd_vs_i64(0UL) ;
    __vr vrsum_b2 = _ve_vbrd_vs_i64(0UL) ;

    int64_t i=0;
    if((inDim & 0x01)==1) {
      __vr vrw = _ve_vldu_vss(4, pWeight+i*outDim+o) ;
      vrsum_b0 = _ve_vfmads_vvsv(vrsum_b0, pIn[0*inDim+i], vrw) ;
      vrsum_b1 = _ve_vfmads_vvsv(vrsum_b1, pIn[1*inDim+i], vrw) ;
      vrsum_b2 = _ve_vfmads_vvsv(vrsum_b2, pIn[2*inDim+i], vrw) ;

      i+=1 ;
    }
    if(((inDim>>1) & 0x01)==1) {
      __vr vrw_i0 = _ve_vldu_vss(4, pWeight+(i  )*outDim+o) ;
      __vr vrw_i1 = _ve_vldu_vss(4, pWeight+(i+1)*outDim+o) ;

      __vr vrw_i01 = _ve_vshf_vvvs(vrw_i0, vrw_i1, VE_VSHUFFLE_YUZU) ;

      const uint64_t in_i01_b0 = _ve_pack_f32p(pIn+0*inDim+i+0, pIn+0*inDim+i+1) ;
      const uint64_t in_i01_b1 = _ve_pack_f32p(pIn+1*inDim+i+0, pIn+1*inDim+i+1) ;
      const uint64_t in_i01_b2 = _ve_pack_f32p(pIn+2*inDim+i+0, pIn+2*inDim+i+1) ;

      vrsum_b0 = _ve_pvfmad_vvsv(vrsum_b0, in_i01_b0, vrw_i01) ;
      vrsum_b1 = _ve_pvfmad_vvsv(vrsum_b1, in_i01_b1, vrw_i01) ;
      vrsum_b2 = _ve_pvfmad_vvsv(vrsum_b2, in_i01_b2, vrw_i01) ;

      i+=2 ;
    }
    for(; i<inDim; i+=4 ) {
      __vr vrw_i0 = _ve_vldu_vss(4, pWeight+(i  )*outDim+o) ;
      __vr vrw_i1 = _ve_vldu_vss(4, pWeight+(i+1)*outDim+o) ;
      __vr vrw_i2 = _ve_vldu_vss(4, pWeight+(i+2)*outDim+o) ;
      __vr vrw_i3 = _ve_vldu_vss(4, pWeight+(i+3)*outDim+o) ;

      __vr vrw_i01 = _ve_vshf_vvvs(vrw_i0, vrw_i1, VE_VSHUFFLE_YUZU) ;
      __vr vrw_i23 = _ve_vshf_vvvs(vrw_i2, vrw_i3, VE_VSHUFFLE_YUZU) ;

      const uint64_t in_i01_b0 = _ve_pack_f32p(pIn+0*inDim+i+0, pIn+0*inDim+i+1) ;
      const uint64_t in_i01_b1 = _ve_pack_f32p(pIn+1*inDim+i+0, pIn+1*inDim+i+1) ;
      const uint64_t in_i01_b2 = _ve_pack_f32p(pIn+2*inDim+i+0, pIn+2*inDim+i+1) ;

      vrsum_b0 = _ve_pvfmad_vvsv(vrsum_b0, in_i01_b0, vrw_i01) ;
      vrsum_b1 = _ve_pvfmad_vvsv(vrsum_b1, in_i01_b1, vrw_i01) ;
      vrsum_b2 = _ve_pvfmad_vvsv(vrsum_b2, in_i01_b2, vrw_i01) ;

      const uint64_t in_i23_b0 = _ve_pack_f32p(pIn+0*inDim+i+2, pIn+0*inDim+i+3) ;
      const uint64_t in_i23_b1 = _ve_pack_f32p(pIn+1*inDim+i+2, pIn+1*inDim+i+3) ;
      const uint64_t in_i23_b2 = _ve_pack_f32p(pIn+2*inDim+i+2, pIn+2*inDim+i+3) ;

      vrsum_b0 = _ve_pvfmad_vvsv(vrsum_b0, in_i23_b0, vrw_i23) ;
      vrsum_b1 = _ve_pvfmad_vvsv(vrsum_b1, in_i23_b1, vrw_i23) ;
      vrsum_b2 = _ve_pvfmad_vvsv(vrsum_b2, in_i23_b2, vrw_i23) ;
    }

    _ve_vstu_vss(_ve_vfadds_vvv(vrsum_b0, _ve_vsll_vvs(vrsum_b0,32)), 4, pOut+0*outDim+o) ;
    _ve_vstu_vss(_ve_vfadds_vvv(vrsum_b1, _ve_vsll_vvs(vrsum_b1,32)), 4, pOut+1*outDim+o) ;
    _ve_vstu_vss(_ve_vfadds_vvv(vrsum_b2, _ve_vsll_vvs(vrsum_b2,32)), 4, pOut+2*outDim+o) ;
  }
}

static inline void b4(
  const uint64_t	inDim,
  const uint64_t	outDim,
  const float * 	pIn,
  const float * 	pWeight,
  float * 		pOut
)
{
  for(int64_t o=0; o<outDim; o+=VLEN) {
    const int64_t vl = outDim-o < VLEN ? outDim - o : VLEN ;
    _ve_lvl(vl) ;
    __vr vrsum_b0 = _ve_vbrd_vs_i64(0UL) ;
    __vr vrsum_b1 = _ve_vbrd_vs_i64(0UL) ;
    __vr vrsum_b2 = _ve_vbrd_vs_i64(0UL) ;
    __vr vrsum_b3 = _ve_vbrd_vs_i64(0UL) ;

    int64_t i=0;
    if((inDim & 0x01)==1) {
      __vr vrw = _ve_vldu_vss(4, pWeight+i*outDim+o) ;
      vrsum_b0 = _ve_vfmads_vvsv(vrsum_b0, pIn[0*inDim+i], vrw) ;
      vrsum_b1 = _ve_vfmads_vvsv(vrsum_b1, pIn[1*inDim+i], vrw) ;
      vrsum_b2 = _ve_vfmads_vvsv(vrsum_b2, pIn[2*inDim+i], vrw) ;
      vrsum_b3 = _ve_vfmads_vvsv(vrsum_b3, pIn[3*inDim+i], vrw) ;

      i+=1 ;
    }
    if(((inDim>>1) & 0x01)==1) {
      __vr vrw_i0 = _ve_vldu_vss(4, pWeight+(i  )*outDim+o) ;
      __vr vrw_i1 = _ve_vldu_vss(4, pWeight+(i+1)*outDim+o) ;

      __vr vrw_i01 = _ve_vshf_vvvs(vrw_i0, vrw_i1, VE_VSHUFFLE_YUZU) ;

      const uint64_t in_i01_b0 = _ve_pack_f32p(pIn+0*inDim+i+0, pIn+0*inDim+i+1) ;
      const uint64_t in_i01_b1 = _ve_pack_f32p(pIn+1*inDim+i+0, pIn+1*inDim+i+1) ;
      const uint64_t in_i01_b2 = _ve_pack_f32p(pIn+2*inDim+i+0, pIn+2*inDim+i+1) ;
      const uint64_t in_i01_b3 = _ve_pack_f32p(pIn+3*inDim+i+0, pIn+3*inDim+i+1) ;

      vrsum_b0 = _ve_pvfmad_vvsv(vrsum_b0, in_i01_b0, vrw_i01) ;
      vrsum_b1 = _ve_pvfmad_vvsv(vrsum_b1, in_i01_b1, vrw_i01) ;
      vrsum_b2 = _ve_pvfmad_vvsv(vrsum_b2, in_i01_b2, vrw_i01) ;
      vrsum_b3 = _ve_pvfmad_vvsv(vrsum_b3, in_i01_b3, vrw_i01) ;

      i+=2 ;
    }
    for(; i<inDim; i+=4 ) {
      __vr vrw_i0 = _ve_vldu_vss(4, pWeight+(i  )*outDim+o) ;
      __vr vrw_i1 = _ve_vldu_vss(4, pWeight+(i+1)*outDim+o) ;
      __vr vrw_i2 = _ve_vldu_vss(4, pWeight+(i+2)*outDim+o) ;
      __vr vrw_i3 = _ve_vldu_vss(4, pWeight+(i+3)*outDim+o) ;

      __vr vrw_i01 = _ve_vshf_vvvs(vrw_i0, vrw_i1, VE_VSHUFFLE_YUZU) ;
      __vr vrw_i23 = _ve_vshf_vvvs(vrw_i2, vrw_i3, VE_VSHUFFLE_YUZU) ;

      const uint64_t in_i01_b0 = _ve_pack_f32p(pIn+0*inDim+i+0, pIn+0*inDim+i+1) ;
      const uint64_t in_i01_b1 = _ve_pack_f32p(pIn+1*inDim+i+0, pIn+1*inDim+i+1) ;
      const uint64_t in_i01_b2 = _ve_pack_f32p(pIn+2*inDim+i+0, pIn+2*inDim+i+1) ;
      const uint64_t in_i01_b3 = _ve_pack_f32p(pIn+3*inDim+i+0, pIn+3*inDim+i+1) ;

      vrsum_b0 = _ve_pvfmad_vvsv(vrsum_b0, in_i01_b0, vrw_i01) ;
      vrsum_b1 = _ve_pvfmad_vvsv(vrsum_b1, in_i01_b1, vrw_i01) ;
      vrsum_b2 = _ve_pvfmad_vvsv(vrsum_b2, in_i01_b2, vrw_i01) ;
      vrsum_b3 = _ve_pvfmad_vvsv(vrsum_b3, in_i01_b3, vrw_i01) ;

      const uint64_t in_i23_b0 = _ve_pack_f32p(pIn+0*inDim+i+2, pIn+0*inDim+i+3) ;
      const uint64_t in_i23_b1 = _ve_pack_f32p(pIn+1*inDim+i+2, pIn+1*inDim+i+3) ;
      const uint64_t in_i23_b2 = _ve_pack_f32p(pIn+2*inDim+i+2, pIn+2*inDim+i+3) ;
      const uint64_t in_i23_b3 = _ve_pack_f32p(pIn+3*inDim+i+2, pIn+3*inDim+i+3) ;

      vrsum_b0 = _ve_pvfmad_vvsv(vrsum_b0, in_i23_b0, vrw_i23) ;
      vrsum_b1 = _ve_pvfmad_vvsv(vrsum_b1, in_i23_b1, vrw_i23) ;
      vrsum_b2 = _ve_pvfmad_vvsv(vrsum_b2, in_i23_b2, vrw_i23) ;
      vrsum_b3 = _ve_pvfmad_vvsv(vrsum_b3, in_i23_b3, vrw_i23) ;
    }

    _ve_vstu_vss(_ve_vfadds_vvv(vrsum_b0, _ve_vsll_vvs(vrsum_b0,32)), 4, pOut+0*outDim+o) ;
    _ve_vstu_vss(_ve_vfadds_vvv(vrsum_b1, _ve_vsll_vvs(vrsum_b1,32)), 4, pOut+1*outDim+o) ;
    _ve_vstu_vss(_ve_vfadds_vvv(vrsum_b2, _ve_vsll_vvs(vrsum_b2,32)), 4, pOut+2*outDim+o) ;
    _ve_vstu_vss(_ve_vfadds_vvv(vrsum_b3, _ve_vsll_vvs(vrsum_b3,32)), 4, pOut+3*outDim+o) ;
  }
}

static inline void b5(
  const uint64_t	inDim,
  const uint64_t	outDim,
  const float * 	pIn,
  const float * 	pWeight,
  float * 		pOut
)
{
  for(int64_t o=0; o<outDim; o+=VLEN) {
    const int64_t vl = outDim-o < VLEN ? outDim - o : VLEN ;
    _ve_lvl(vl) ;
    __vr vrsum_b0 = _ve_vbrd_vs_i64(0UL) ;
    __vr vrsum_b1 = _ve_vbrd_vs_i64(0UL) ;
    __vr vrsum_b2 = _ve_vbrd_vs_i64(0UL) ;
    __vr vrsum_b3 = _ve_vbrd_vs_i64(0UL) ;
    __vr vrsum_b4 = _ve_vbrd_vs_i64(0UL) ;

    int64_t i=0;
    if((inDim & 0x01)==1) {
      __vr vrw = _ve_vldu_vss(4, pWeight+i*outDim+o) ;
      vrsum_b0 = _ve_vfmads_vvsv(vrsum_b0, pIn[0*inDim+i], vrw) ;
      vrsum_b1 = _ve_vfmads_vvsv(vrsum_b1, pIn[1*inDim+i], vrw) ;
      vrsum_b2 = _ve_vfmads_vvsv(vrsum_b2, pIn[2*inDim+i], vrw) ;
      vrsum_b3 = _ve_vfmads_vvsv(vrsum_b3, pIn[3*inDim+i], vrw) ;
      vrsum_b4 = _ve_vfmads_vvsv(vrsum_b4, pIn[4*inDim+i], vrw) ;

      i+=1 ;
    }
    if(((inDim>>1) & 0x01)==1) {
      __vr vrw_i0 = _ve_vldu_vss(4, pWeight+(i  )*outDim+o) ;
      __vr vrw_i1 = _ve_vldu_vss(4, pWeight+(i+1)*outDim+o) ;

      __vr vrw_i01 = _ve_vshf_vvvs(vrw_i0, vrw_i1, VE_VSHUFFLE_YUZU) ;

      const uint64_t in_i01_b0 = _ve_pack_f32p(pIn+0*inDim+i+0, pIn+0*inDim+i+1) ;
      const uint64_t in_i01_b1 = _ve_pack_f32p(pIn+1*inDim+i+0, pIn+1*inDim+i+1) ;
      const uint64_t in_i01_b2 = _ve_pack_f32p(pIn+2*inDim+i+0, pIn+2*inDim+i+1) ;
      const uint64_t in_i01_b3 = _ve_pack_f32p(pIn+3*inDim+i+0, pIn+3*inDim+i+1) ;
      const uint64_t in_i01_b4 = _ve_pack_f32p(pIn+4*inDim+i+0, pIn+4*inDim+i+1) ;

      vrsum_b0 = _ve_pvfmad_vvsv(vrsum_b0, in_i01_b0, vrw_i01) ;
      vrsum_b1 = _ve_pvfmad_vvsv(vrsum_b1, in_i01_b1, vrw_i01) ;
      vrsum_b2 = _ve_pvfmad_vvsv(vrsum_b2, in_i01_b2, vrw_i01) ;
      vrsum_b3 = _ve_pvfmad_vvsv(vrsum_b3, in_i01_b3, vrw_i01) ;
      vrsum_b4 = _ve_pvfmad_vvsv(vrsum_b4, in_i01_b4, vrw_i01) ;

      i+=2 ;
    }
    for(; i<inDim; i+=4 ) {
      __vr vrw_i0 = _ve_vldu_vss(4, pWeight+(i  )*outDim+o) ;
      __vr vrw_i1 = _ve_vldu_vss(4, pWeight+(i+1)*outDim+o) ;
      __vr vrw_i2 = _ve_vldu_vss(4, pWeight+(i+2)*outDim+o) ;
      __vr vrw_i3 = _ve_vldu_vss(4, pWeight+(i+3)*outDim+o) ;

      __vr vrw_i01 = _ve_vshf_vvvs(vrw_i0, vrw_i1, VE_VSHUFFLE_YUZU) ;
      __vr vrw_i23 = _ve_vshf_vvvs(vrw_i2, vrw_i3, VE_VSHUFFLE_YUZU) ;

      const uint64_t in_i01_b0 = _ve_pack_f32p(pIn+0*inDim+i+0, pIn+0*inDim+i+1) ;
      const uint64_t in_i01_b1 = _ve_pack_f32p(pIn+1*inDim+i+0, pIn+1*inDim+i+1) ;
      const uint64_t in_i01_b2 = _ve_pack_f32p(pIn+2*inDim+i+0, pIn+2*inDim+i+1) ;
      const uint64_t in_i01_b3 = _ve_pack_f32p(pIn+3*inDim+i+0, pIn+3*inDim+i+1) ;
      const uint64_t in_i01_b4 = _ve_pack_f32p(pIn+4*inDim+i+0, pIn+4*inDim+i+1) ;

      vrsum_b0 = _ve_pvfmad_vvsv(vrsum_b0, in_i01_b0, vrw_i01) ;
      vrsum_b1 = _ve_pvfmad_vvsv(vrsum_b1, in_i01_b1, vrw_i01) ;
      vrsum_b2 = _ve_pvfmad_vvsv(vrsum_b2, in_i01_b2, vrw_i01) ;
      vrsum_b3 = _ve_pvfmad_vvsv(vrsum_b3, in_i01_b3, vrw_i01) ;
      vrsum_b4 = _ve_pvfmad_vvsv(vrsum_b4, in_i01_b4, vrw_i01) ;

      const uint64_t in_i23_b0 = _ve_pack_f32p(pIn+0*inDim+i+2, pIn+0*inDim+i+3) ;
      const uint64_t in_i23_b1 = _ve_pack_f32p(pIn+1*inDim+i+2, pIn+1*inDim+i+3) ;
      const uint64_t in_i23_b2 = _ve_pack_f32p(pIn+2*inDim+i+2, pIn+2*inDim+i+3) ;
      const uint64_t in_i23_b3 = _ve_pack_f32p(pIn+3*inDim+i+2, pIn+3*inDim+i+3) ;
      const uint64_t in_i23_b4 = _ve_pack_f32p(pIn+4*inDim+i+2, pIn+4*inDim+i+3) ;

      vrsum_b0 = _ve_pvfmad_vvsv(vrsum_b0, in_i23_b0, vrw_i23) ;
      vrsum_b1 = _ve_pvfmad_vvsv(vrsum_b1, in_i23_b1, vrw_i23) ;
      vrsum_b2 = _ve_pvfmad_vvsv(vrsum_b2, in_i23_b2, vrw_i23) ;
      vrsum_b3 = _ve_pvfmad_vvsv(vrsum_b3, in_i23_b3, vrw_i23) ;
      vrsum_b4 = _ve_pvfmad_vvsv(vrsum_b4, in_i23_b4, vrw_i23) ;
    }

    _ve_vstu_vss(_ve_vfadds_vvv(vrsum_b0, _ve_vsll_vvs(vrsum_b0,32)), 4, pOut+0*outDim+o) ;
    _ve_vstu_vss(_ve_vfadds_vvv(vrsum_b1, _ve_vsll_vvs(vrsum_b1,32)), 4, pOut+1*outDim+o) ;
    _ve_vstu_vss(_ve_vfadds_vvv(vrsum_b2, _ve_vsll_vvs(vrsum_b2,32)), 4, pOut+2*outDim+o) ;
    _ve_vstu_vss(_ve_vfadds_vvv(vrsum_b3, _ve_vsll_vvs(vrsum_b3,32)), 4, pOut+3*outDim+o) ;
    _ve_vstu_vss(_ve_vfadds_vvv(vrsum_b4, _ve_vsll_vvs(vrsum_b4,32)), 4, pOut+4*outDim+o) ;
  }
}


static inline void b6(
  const uint64_t	inDim,
  const uint64_t	outDim,
  const float * 	pIn,
  const float * 	pWeight,
  float * 		pOut
)
{
  for(int64_t o=0; o<outDim; o+=VLEN) {
    const int64_t vl = outDim-o < VLEN ? outDim - o : VLEN ;
    _ve_lvl(vl) ;
    __vr vrsum_b0 = _ve_vbrd_vs_i64(0UL) ;
    __vr vrsum_b1 = _ve_vbrd_vs_i64(0UL) ;
    __vr vrsum_b2 = _ve_vbrd_vs_i64(0UL) ;
    __vr vrsum_b3 = _ve_vbrd_vs_i64(0UL) ;
    __vr vrsum_b4 = _ve_vbrd_vs_i64(0UL) ;
    __vr vrsum_b5 = _ve_vbrd_vs_i64(0UL) ;

    int64_t i=0;
    if((inDim & 0x01)==1) {
      __vr vrw = _ve_vldu_vss(4, pWeight+i*outDim+o) ;
      vrsum_b0 = _ve_vfmads_vvsv(vrsum_b0, pIn[0*inDim+i], vrw) ;
      vrsum_b1 = _ve_vfmads_vvsv(vrsum_b1, pIn[1*inDim+i], vrw) ;
      vrsum_b2 = _ve_vfmads_vvsv(vrsum_b2, pIn[2*inDim+i], vrw) ;
      vrsum_b3 = _ve_vfmads_vvsv(vrsum_b3, pIn[3*inDim+i], vrw) ;
      vrsum_b4 = _ve_vfmads_vvsv(vrsum_b4, pIn[4*inDim+i], vrw) ;
      vrsum_b5 = _ve_vfmads_vvsv(vrsum_b5, pIn[5*inDim+i], vrw) ;

      i+=1 ;
    }
    if(((inDim>>1) & 0x01)==1) {
      __vr vrw_i0 = _ve_vldu_vss(4, pWeight+(i  )*outDim+o) ;
      __vr vrw_i1 = _ve_vldu_vss(4, pWeight+(i+1)*outDim+o) ;

      __vr vrw_i01 = _ve_vshf_vvvs(vrw_i0, vrw_i1, VE_VSHUFFLE_YUZU) ;

      const uint64_t in_i01_b0 = _ve_pack_f32p(pIn+0*inDim+i+0, pIn+0*inDim+i+1) ;
      const uint64_t in_i01_b1 = _ve_pack_f32p(pIn+1*inDim+i+0, pIn+1*inDim+i+1) ;
      const uint64_t in_i01_b2 = _ve_pack_f32p(pIn+2*inDim+i+0, pIn+2*inDim+i+1) ;
      const uint64_t in_i01_b3 = _ve_pack_f32p(pIn+3*inDim+i+0, pIn+3*inDim+i+1) ;
      const uint64_t in_i01_b4 = _ve_pack_f32p(pIn+4*inDim+i+0, pIn+4*inDim+i+1) ;
      const uint64_t in_i01_b5 = _ve_pack_f32p(pIn+5*inDim+i+0, pIn+5*inDim+i+1) ;

      vrsum_b0 = _ve_pvfmad_vvsv(vrsum_b0, in_i01_b0, vrw_i01) ;
      vrsum_b1 = _ve_pvfmad_vvsv(vrsum_b1, in_i01_b1, vrw_i01) ;
      vrsum_b2 = _ve_pvfmad_vvsv(vrsum_b2, in_i01_b2, vrw_i01) ;
      vrsum_b3 = _ve_pvfmad_vvsv(vrsum_b3, in_i01_b3, vrw_i01) ;
      vrsum_b4 = _ve_pvfmad_vvsv(vrsum_b4, in_i01_b4, vrw_i01) ;
      vrsum_b5 = _ve_pvfmad_vvsv(vrsum_b5, in_i01_b5, vrw_i01) ;

      i+=2 ;
    }
    for(; i<inDim; i+=4 ) {
      __vr vrw_i0 = _ve_vldu_vss(4, pWeight+(i  )*outDim+o) ;
      __vr vrw_i1 = _ve_vldu_vss(4, pWeight+(i+1)*outDim+o) ;
      __vr vrw_i2 = _ve_vldu_vss(4, pWeight+(i+2)*outDim+o) ;
      __vr vrw_i3 = _ve_vldu_vss(4, pWeight+(i+3)*outDim+o) ;

      __vr vrw_i01 = _ve_vshf_vvvs(vrw_i0, vrw_i1, VE_VSHUFFLE_YUZU) ;
      __vr vrw_i23 = _ve_vshf_vvvs(vrw_i2, vrw_i3, VE_VSHUFFLE_YUZU) ;

      const uint64_t in_i01_b0 = _ve_pack_f32p(pIn+0*inDim+i+0, pIn+0*inDim+i+1) ;
      const uint64_t in_i01_b1 = _ve_pack_f32p(pIn+1*inDim+i+0, pIn+1*inDim+i+1) ;
      const uint64_t in_i01_b2 = _ve_pack_f32p(pIn+2*inDim+i+0, pIn+2*inDim+i+1) ;
      const uint64_t in_i01_b3 = _ve_pack_f32p(pIn+3*inDim+i+0, pIn+3*inDim+i+1) ;
      const uint64_t in_i01_b4 = _ve_pack_f32p(pIn+4*inDim+i+0, pIn+4*inDim+i+1) ;
      const uint64_t in_i01_b5 = _ve_pack_f32p(pIn+5*inDim+i+0, pIn+5*inDim+i+1) ;

      vrsum_b0 = _ve_pvfmad_vvsv(vrsum_b0, in_i01_b0, vrw_i01) ;
      vrsum_b1 = _ve_pvfmad_vvsv(vrsum_b1, in_i01_b1, vrw_i01) ;
      vrsum_b2 = _ve_pvfmad_vvsv(vrsum_b2, in_i01_b2, vrw_i01) ;
      vrsum_b3 = _ve_pvfmad_vvsv(vrsum_b3, in_i01_b3, vrw_i01) ;
      vrsum_b4 = _ve_pvfmad_vvsv(vrsum_b4, in_i01_b4, vrw_i01) ;
      vrsum_b5 = _ve_pvfmad_vvsv(vrsum_b5, in_i01_b5, vrw_i01) ;

      const uint64_t in_i23_b0 = _ve_pack_f32p(pIn+0*inDim+i+2, pIn+0*inDim+i+3) ;
      const uint64_t in_i23_b1 = _ve_pack_f32p(pIn+1*inDim+i+2, pIn+1*inDim+i+3) ;
      const uint64_t in_i23_b2 = _ve_pack_f32p(pIn+2*inDim+i+2, pIn+2*inDim+i+3) ;
      const uint64_t in_i23_b3 = _ve_pack_f32p(pIn+3*inDim+i+2, pIn+3*inDim+i+3) ;
      const uint64_t in_i23_b4 = _ve_pack_f32p(pIn+4*inDim+i+2, pIn+4*inDim+i+3) ;
      const uint64_t in_i23_b5 = _ve_pack_f32p(pIn+5*inDim+i+2, pIn+5*inDim+i+3) ;

      vrsum_b0 = _ve_pvfmad_vvsv(vrsum_b0, in_i23_b0, vrw_i23) ;
      vrsum_b1 = _ve_pvfmad_vvsv(vrsum_b1, in_i23_b1, vrw_i23) ;
      vrsum_b2 = _ve_pvfmad_vvsv(vrsum_b2, in_i23_b2, vrw_i23) ;
      vrsum_b3 = _ve_pvfmad_vvsv(vrsum_b3, in_i23_b3, vrw_i23) ;
      vrsum_b4 = _ve_pvfmad_vvsv(vrsum_b4, in_i23_b4, vrw_i23) ;
      vrsum_b5 = _ve_pvfmad_vvsv(vrsum_b5, in_i23_b5, vrw_i23) ;
    }

    _ve_vstu_vss(_ve_vfadds_vvv(vrsum_b0, _ve_vsll_vvs(vrsum_b0,32)), 4, pOut+0*outDim+o) ;
    _ve_vstu_vss(_ve_vfadds_vvv(vrsum_b1, _ve_vsll_vvs(vrsum_b1,32)), 4, pOut+1*outDim+o) ;
    _ve_vstu_vss(_ve_vfadds_vvv(vrsum_b2, _ve_vsll_vvs(vrsum_b2,32)), 4, pOut+2*outDim+o) ;
    _ve_vstu_vss(_ve_vfadds_vvv(vrsum_b3, _ve_vsll_vvs(vrsum_b3,32)), 4, pOut+3*outDim+o) ;
    _ve_vstu_vss(_ve_vfadds_vvv(vrsum_b4, _ve_vsll_vvs(vrsum_b4,32)), 4, pOut+4*outDim+o) ;
    _ve_vstu_vss(_ve_vfadds_vvv(vrsum_b5, _ve_vsll_vvs(vrsum_b5,32)), 4, pOut+5*outDim+o) ;
  }
}


vednnError_t vednnLinearForward_default(
    const uint64_t			inDim,
    const uint64_t			outDim,
    const uint64_t			nBatch,
    const void * restrict		pDataIn,
    const void * restrict		pDataWeight,
    void * restrict			pDataOut
)
{

  const float * restrict pIn     = (const float * restrict) pDataIn;
  const float * restrict pWeight = (const float * restrict) pDataWeight;
  float * restrict const pOut    = (float * restrict const) pDataOut;

  int64_t n=0;
  int64_t batchRemain = nBatch % 6 ;

  switch( batchRemain ) {
  case 1 :
    b1(inDim, outDim,
       pIn+n*inDim, pWeight, pOut+n*outDim) ;
    n+=1 ;
    break ;
  case 2 :
    b2(inDim, outDim,
       pIn+n*inDim, pWeight, pOut+n*outDim) ;
    n+=2 ;
    break ;
  case 3 :
    b3(inDim, outDim,
       pIn+n*inDim, pWeight, pOut+n*outDim) ;
    n+=3 ;
    break ;
  case 4 :
    b4(inDim, outDim,
       pIn+n*inDim, pWeight, pOut+n*outDim) ;
    n+=4 ;
    break ;
  case 5 :
    b5(inDim, outDim,
       pIn+n*inDim, pWeight, pOut+n*outDim) ;
    n+=5 ;
    break ;
  default :
    break ;
  }
  for(; n<nBatch; n+=6) {
    b6(inDim, outDim,
       pIn+n*inDim, pWeight, pOut+n*outDim) ;
  }

  return VEDNN_SUCCESS ;
}

#if 0 // reference code
vednnError_t vednnLinearForward_default(
    const uint64_t			inDim,
    const uint64_t			outDim,
    const uint64_t			nBatch,
    const void * restrict		pDataIn,
    const void * restrict		pDataWeight,
    void * restrict			pDataOut
)
{
  const float * restrict pIn     = pDataIn;
  const float * restrict pWeight = pDataWeight;
  float * restrict const pOut    = pDataOut;

  for(int64_t n=0; n<nBatch; n++) {
    for(int64_t o=0; o<outDim; o++) {
      float sum = 0.f ;
      for(int64_t i=0; i<inDim; i++ ) {
	sum += pWeight[i*outDim+o] * pIn[n*inDim+i] ;
      }
      pOut[n*outDim+o] = sum ;
    }
  }

  return VEDNN_SUCCESS ;
}
#endif


