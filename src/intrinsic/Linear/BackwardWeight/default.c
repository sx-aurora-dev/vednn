#include <stdint.h>
#include <float.h>

#include <stdio.h>

#include "vednn.h"

#include "veintrin.h"
#define VLEN	(256)


static inline void b1(
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
    _ve_lvl(vl) ;

    __vr vrgout_b0 = _ve_vldu_vss(4, pGOut+0*outDim+o) ;

    int64_t i=0;
    if(nInDim & 0x1) {
      const float in_b0 = pIn[0*inDim+i] ;
      __vr vrgw = _ve_vfmuls_vsv(in_b0, vrgout_b0) ;
      _ve_vstu_vss(vrgw, 4, pGWeight+i*outDim+o) ;

      i+=1;
    }
    if((nInDim>>1) & 0x1) {

      const float in_b0_i0 = pIn[0*inDim+i+0] ;
      const float in_b0_i1 = pIn[0*inDim+i+1] ;
      __vr vrgw_i0 = _ve_vfmuls_vsv(in_b0_i0, vrgout_b0) ;
      __vr vrgw_i1 = _ve_vfmuls_vsv(in_b0_i1, vrgout_b0) ;

      _ve_vstu_vss(vrgw_i0, 4, pGWeight+(i+0)*outDim+o) ;
      _ve_vstu_vss(vrgw_i1, 4, pGWeight+(i+1)*outDim+o) ;

      i+=2 ;
    }
    if((nInDim>>2) & 0x1) {

      const float in_b0_i0 = pIn[0*inDim+i+0] ;
      const float in_b0_i1 = pIn[0*inDim+i+1] ;
      const float in_b0_i2 = pIn[0*inDim+i+2] ;
      const float in_b0_i3 = pIn[0*inDim+i+3] ;
      __vr vrgw_i0 = _ve_vfmuls_vsv(in_b0_i0, vrgout_b0) ;
      __vr vrgw_i1 = _ve_vfmuls_vsv(in_b0_i1, vrgout_b0) ;
      __vr vrgw_i2 = _ve_vfmuls_vsv(in_b0_i2, vrgout_b0) ;
      __vr vrgw_i3 = _ve_vfmuls_vsv(in_b0_i3, vrgout_b0) ;

      _ve_vstu_vss(vrgw_i0, 4, pGWeight+(i+0)*outDim+o) ;
      _ve_vstu_vss(vrgw_i1, 4, pGWeight+(i+1)*outDim+o) ;
      _ve_vstu_vss(vrgw_i2, 4, pGWeight+(i+2)*outDim+o) ;
      _ve_vstu_vss(vrgw_i3, 4, pGWeight+(i+3)*outDim+o) ;

      i+=4 ;
    }
    for(; i<nInDim; i+=8) {

      const float in_b0_i0 = pIn[0*inDim+i+0] ;
      const float in_b0_i1 = pIn[0*inDim+i+1] ;
      const float in_b0_i2 = pIn[0*inDim+i+2] ;
      const float in_b0_i3 = pIn[0*inDim+i+3] ;
      const float in_b0_i4 = pIn[0*inDim+i+4] ;
      const float in_b0_i5 = pIn[0*inDim+i+5] ;
      const float in_b0_i6 = pIn[0*inDim+i+6] ;
      const float in_b0_i7 = pIn[0*inDim+i+7] ;
      __vr vrgw_i0 = _ve_vfmuls_vsv(in_b0_i0, vrgout_b0) ;
      __vr vrgw_i1 = _ve_vfmuls_vsv(in_b0_i1, vrgout_b0) ;
      __vr vrgw_i2 = _ve_vfmuls_vsv(in_b0_i2, vrgout_b0) ;
      __vr vrgw_i3 = _ve_vfmuls_vsv(in_b0_i3, vrgout_b0) ;
      __vr vrgw_i4 = _ve_vfmuls_vsv(in_b0_i4, vrgout_b0) ;
      __vr vrgw_i5 = _ve_vfmuls_vsv(in_b0_i5, vrgout_b0) ;
      __vr vrgw_i6 = _ve_vfmuls_vsv(in_b0_i6, vrgout_b0) ;
      __vr vrgw_i7 = _ve_vfmuls_vsv(in_b0_i7, vrgout_b0) ;

      _ve_vstu_vss(vrgw_i0, 4, pGWeight+(i+0)*outDim+o) ;
      _ve_vstu_vss(vrgw_i1, 4, pGWeight+(i+1)*outDim+o) ;
      _ve_vstu_vss(vrgw_i2, 4, pGWeight+(i+2)*outDim+o) ;
      _ve_vstu_vss(vrgw_i3, 4, pGWeight+(i+3)*outDim+o) ;
      _ve_vstu_vss(vrgw_i4, 4, pGWeight+(i+4)*outDim+o) ;
      _ve_vstu_vss(vrgw_i5, 4, pGWeight+(i+5)*outDim+o) ;
      _ve_vstu_vss(vrgw_i6, 4, pGWeight+(i+6)*outDim+o) ;
      _ve_vstu_vss(vrgw_i7, 4, pGWeight+(i+7)*outDim+o) ;

    }
  }
}

static inline void b2(
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
    _ve_lvl(vl) ;

    __vr vrgout_b0 = _ve_vldu_vss(4, pGOut+0*outDim+o) ;
    __vr vrgout_b1 = _ve_vldu_vss(4, pGOut+1*outDim+o) ;

    int64_t i=0;
    if(nInDim & 0x1) {
      const float in_b0 = pIn[0*inDim+i] ;
      __vr vrgw = _ve_vfmuls_vsv(in_b0, vrgout_b0) ;

      const float in_b1 = pIn[1*inDim+i] ;
      vrgw = _ve_vfmads_vvsv(vrgw, in_b1, vrgout_b1) ;

      _ve_vstu_vss(vrgw, 4, pGWeight+i*outDim+o) ;

      i+=1;
    }
    if((nInDim>>1) & 0x1) {

      const float in_b0_i0 = pIn[0*inDim+i+0] ;
      const float in_b0_i1 = pIn[0*inDim+i+1] ;
      __vr vrgw_i0 = _ve_vfmuls_vsv(in_b0_i0, vrgout_b0) ;
      __vr vrgw_i1 = _ve_vfmuls_vsv(in_b0_i1, vrgout_b0) ;

      const float in_b1_i0 = pIn[1*inDim+i+0] ;
      const float in_b1_i1 = pIn[1*inDim+i+1] ;
      vrgw_i0 = _ve_vfmads_vvsv(vrgw_i0, in_b1_i0, vrgout_b1) ;
      vrgw_i1 = _ve_vfmads_vvsv(vrgw_i1, in_b1_i1, vrgout_b1) ;

      _ve_vstu_vss(vrgw_i0, 4, pGWeight+(i+0)*outDim+o) ;
      _ve_vstu_vss(vrgw_i1, 4, pGWeight+(i+1)*outDim+o) ;

      i+=2 ;
    }
    if((nInDim>>2) & 0x1) {

      const float in_b0_i0 = pIn[0*inDim+i+0] ;
      const float in_b0_i1 = pIn[0*inDim+i+1] ;
      const float in_b0_i2 = pIn[0*inDim+i+2] ;
      const float in_b0_i3 = pIn[0*inDim+i+3] ;
      __vr vrgw_i0 = _ve_vfmuls_vsv(in_b0_i0, vrgout_b0) ;
      __vr vrgw_i1 = _ve_vfmuls_vsv(in_b0_i1, vrgout_b0) ;
      __vr vrgw_i2 = _ve_vfmuls_vsv(in_b0_i2, vrgout_b0) ;
      __vr vrgw_i3 = _ve_vfmuls_vsv(in_b0_i3, vrgout_b0) ;

      const float in_b1_i0 = pIn[1*inDim+i+0] ;
      const float in_b1_i1 = pIn[1*inDim+i+1] ;
      const float in_b1_i2 = pIn[1*inDim+i+2] ;
      const float in_b1_i3 = pIn[1*inDim+i+3] ;
      vrgw_i0 = _ve_vfmads_vvsv(vrgw_i0, in_b1_i0, vrgout_b1) ;
      vrgw_i1 = _ve_vfmads_vvsv(vrgw_i1, in_b1_i1, vrgout_b1) ;
      vrgw_i2 = _ve_vfmads_vvsv(vrgw_i2, in_b1_i2, vrgout_b1) ;
      vrgw_i3 = _ve_vfmads_vvsv(vrgw_i3, in_b1_i3, vrgout_b1) ;

      _ve_vstu_vss(vrgw_i0, 4, pGWeight+(i+0)*outDim+o) ;
      _ve_vstu_vss(vrgw_i1, 4, pGWeight+(i+1)*outDim+o) ;
      _ve_vstu_vss(vrgw_i2, 4, pGWeight+(i+2)*outDim+o) ;
      _ve_vstu_vss(vrgw_i3, 4, pGWeight+(i+3)*outDim+o) ;

      i+=4 ;
    }
    for(; i<nInDim; i+=8) {

      const float in_b0_i0 = pIn[0*inDim+i+0] ;
      const float in_b0_i1 = pIn[0*inDim+i+1] ;
      const float in_b0_i2 = pIn[0*inDim+i+2] ;
      const float in_b0_i3 = pIn[0*inDim+i+3] ;
      const float in_b0_i4 = pIn[0*inDim+i+4] ;
      const float in_b0_i5 = pIn[0*inDim+i+5] ;
      const float in_b0_i6 = pIn[0*inDim+i+6] ;
      const float in_b0_i7 = pIn[0*inDim+i+7] ;
      __vr vrgw_i0 = _ve_vfmuls_vsv(in_b0_i0, vrgout_b0) ;
      __vr vrgw_i1 = _ve_vfmuls_vsv(in_b0_i1, vrgout_b0) ;
      __vr vrgw_i2 = _ve_vfmuls_vsv(in_b0_i2, vrgout_b0) ;
      __vr vrgw_i3 = _ve_vfmuls_vsv(in_b0_i3, vrgout_b0) ;
      __vr vrgw_i4 = _ve_vfmuls_vsv(in_b0_i4, vrgout_b0) ;
      __vr vrgw_i5 = _ve_vfmuls_vsv(in_b0_i5, vrgout_b0) ;
      __vr vrgw_i6 = _ve_vfmuls_vsv(in_b0_i6, vrgout_b0) ;
      __vr vrgw_i7 = _ve_vfmuls_vsv(in_b0_i7, vrgout_b0) ;

      const float in_b1_i0 = pIn[1*inDim+i+0] ;
      const float in_b1_i1 = pIn[1*inDim+i+1] ;
      const float in_b1_i2 = pIn[1*inDim+i+2] ;
      const float in_b1_i3 = pIn[1*inDim+i+3] ;
      const float in_b1_i4 = pIn[1*inDim+i+4] ;
      const float in_b1_i5 = pIn[1*inDim+i+5] ;
      const float in_b1_i6 = pIn[1*inDim+i+6] ;
      const float in_b1_i7 = pIn[1*inDim+i+7] ;
      vrgw_i0 = _ve_vfmads_vvsv(vrgw_i0, in_b1_i0, vrgout_b1) ;
      vrgw_i1 = _ve_vfmads_vvsv(vrgw_i1, in_b1_i1, vrgout_b1) ;
      vrgw_i2 = _ve_vfmads_vvsv(vrgw_i2, in_b1_i2, vrgout_b1) ;
      vrgw_i3 = _ve_vfmads_vvsv(vrgw_i3, in_b1_i3, vrgout_b1) ;
      vrgw_i4 = _ve_vfmads_vvsv(vrgw_i4, in_b1_i4, vrgout_b1) ;
      vrgw_i5 = _ve_vfmads_vvsv(vrgw_i5, in_b1_i5, vrgout_b1) ;
      vrgw_i6 = _ve_vfmads_vvsv(vrgw_i6, in_b1_i6, vrgout_b1) ;
      vrgw_i7 = _ve_vfmads_vvsv(vrgw_i7, in_b1_i7, vrgout_b1) ;

      _ve_vstu_vss(vrgw_i0, 4, pGWeight+(i+0)*outDim+o) ;
      _ve_vstu_vss(vrgw_i1, 4, pGWeight+(i+1)*outDim+o) ;
      _ve_vstu_vss(vrgw_i2, 4, pGWeight+(i+2)*outDim+o) ;
      _ve_vstu_vss(vrgw_i3, 4, pGWeight+(i+3)*outDim+o) ;
      _ve_vstu_vss(vrgw_i4, 4, pGWeight+(i+4)*outDim+o) ;
      _ve_vstu_vss(vrgw_i5, 4, pGWeight+(i+5)*outDim+o) ;
      _ve_vstu_vss(vrgw_i6, 4, pGWeight+(i+6)*outDim+o) ;
      _ve_vstu_vss(vrgw_i7, 4, pGWeight+(i+7)*outDim+o) ;
    }
  }
}

static inline void b3(
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
    _ve_lvl(vl) ;

    __vr vrgout_b0 = _ve_vldu_vss(4, pGOut+0*outDim+o) ;
    __vr vrgout_b1 = _ve_vldu_vss(4, pGOut+1*outDim+o) ;
    __vr vrgout_b2 = _ve_vldu_vss(4, pGOut+2*outDim+o) ;

    int64_t i=0;
    if(nInDim & 0x1) {
      const float in_b0 = pIn[0*inDim+i] ;
      __vr vrgw = _ve_vfmuls_vsv(in_b0, vrgout_b0) ;

      const float in_b1 = pIn[1*inDim+i] ;
      vrgw = _ve_vfmads_vvsv(vrgw, in_b1, vrgout_b1) ;

      const float in_b2 = pIn[2*inDim+i] ;
      vrgw = _ve_vfmads_vvsv(vrgw, in_b2, vrgout_b2) ;

      _ve_vstu_vss(vrgw, 4, pGWeight+i*outDim+o) ;

      i+=1;
    }
    if((nInDim>>1) & 0x1) {

      const float in_b0_i0 = pIn[0*inDim+i+0] ;
      const float in_b0_i1 = pIn[0*inDim+i+1] ;
      __vr vrgw_i0 = _ve_vfmuls_vsv(in_b0_i0, vrgout_b0) ;
      __vr vrgw_i1 = _ve_vfmuls_vsv(in_b0_i1, vrgout_b0) ;

      const float in_b1_i0 = pIn[1*inDim+i+0] ;
      const float in_b1_i1 = pIn[1*inDim+i+1] ;
      vrgw_i0 = _ve_vfmads_vvsv(vrgw_i0, in_b1_i0, vrgout_b1) ;
      vrgw_i1 = _ve_vfmads_vvsv(vrgw_i1, in_b1_i1, vrgout_b1) ;

      const float in_b2_i0 = pIn[2*inDim+i+0] ;
      const float in_b2_i1 = pIn[2*inDim+i+1] ;
      vrgw_i0 = _ve_vfmads_vvsv(vrgw_i0, in_b2_i0, vrgout_b2) ;
      vrgw_i1 = _ve_vfmads_vvsv(vrgw_i1, in_b2_i1, vrgout_b2) ;

      _ve_vstu_vss(vrgw_i0, 4, pGWeight+(i+0)*outDim+o) ;
      _ve_vstu_vss(vrgw_i1, 4, pGWeight+(i+1)*outDim+o) ;

      i+=2 ;
    }
    if((nInDim>>2) & 0x1) {

      const float in_b0_i0 = pIn[0*inDim+i+0] ;
      const float in_b0_i1 = pIn[0*inDim+i+1] ;
      const float in_b0_i2 = pIn[0*inDim+i+2] ;
      const float in_b0_i3 = pIn[0*inDim+i+3] ;
      __vr vrgw_i0 = _ve_vfmuls_vsv(in_b0_i0, vrgout_b0) ;
      __vr vrgw_i1 = _ve_vfmuls_vsv(in_b0_i1, vrgout_b0) ;
      __vr vrgw_i2 = _ve_vfmuls_vsv(in_b0_i2, vrgout_b0) ;
      __vr vrgw_i3 = _ve_vfmuls_vsv(in_b0_i3, vrgout_b0) ;

      const float in_b1_i0 = pIn[1*inDim+i+0] ;
      const float in_b1_i1 = pIn[1*inDim+i+1] ;
      const float in_b1_i2 = pIn[1*inDim+i+2] ;
      const float in_b1_i3 = pIn[1*inDim+i+3] ;
      vrgw_i0 = _ve_vfmads_vvsv(vrgw_i0, in_b1_i0, vrgout_b1) ;
      vrgw_i1 = _ve_vfmads_vvsv(vrgw_i1, in_b1_i1, vrgout_b1) ;
      vrgw_i2 = _ve_vfmads_vvsv(vrgw_i2, in_b1_i2, vrgout_b1) ;
      vrgw_i3 = _ve_vfmads_vvsv(vrgw_i3, in_b1_i3, vrgout_b1) ;

      const float in_b2_i0 = pIn[2*inDim+i+0] ;
      const float in_b2_i1 = pIn[2*inDim+i+1] ;
      const float in_b2_i2 = pIn[2*inDim+i+2] ;
      const float in_b2_i3 = pIn[2*inDim+i+3] ;
      vrgw_i0 = _ve_vfmads_vvsv(vrgw_i0, in_b2_i0, vrgout_b2) ;
      vrgw_i1 = _ve_vfmads_vvsv(vrgw_i1, in_b2_i1, vrgout_b2) ;
      vrgw_i2 = _ve_vfmads_vvsv(vrgw_i2, in_b2_i2, vrgout_b2) ;
      vrgw_i3 = _ve_vfmads_vvsv(vrgw_i3, in_b2_i3, vrgout_b2) ;

      _ve_vstu_vss(vrgw_i0, 4, pGWeight+(i+0)*outDim+o) ;
      _ve_vstu_vss(vrgw_i1, 4, pGWeight+(i+1)*outDim+o) ;
      _ve_vstu_vss(vrgw_i2, 4, pGWeight+(i+2)*outDim+o) ;
      _ve_vstu_vss(vrgw_i3, 4, pGWeight+(i+3)*outDim+o) ;

      i+=4 ;
    }
    for(; i<nInDim; i+=8) {

      const float in_b0_i0 = pIn[0*inDim+i+0] ;
      const float in_b0_i1 = pIn[0*inDim+i+1] ;
      const float in_b0_i2 = pIn[0*inDim+i+2] ;
      const float in_b0_i3 = pIn[0*inDim+i+3] ;
      const float in_b0_i4 = pIn[0*inDim+i+4] ;
      const float in_b0_i5 = pIn[0*inDim+i+5] ;
      const float in_b0_i6 = pIn[0*inDim+i+6] ;
      const float in_b0_i7 = pIn[0*inDim+i+7] ;
      __vr vrgw_i0 = _ve_vfmuls_vsv(in_b0_i0, vrgout_b0) ;
      __vr vrgw_i1 = _ve_vfmuls_vsv(in_b0_i1, vrgout_b0) ;
      __vr vrgw_i2 = _ve_vfmuls_vsv(in_b0_i2, vrgout_b0) ;
      __vr vrgw_i3 = _ve_vfmuls_vsv(in_b0_i3, vrgout_b0) ;
      __vr vrgw_i4 = _ve_vfmuls_vsv(in_b0_i4, vrgout_b0) ;
      __vr vrgw_i5 = _ve_vfmuls_vsv(in_b0_i5, vrgout_b0) ;
      __vr vrgw_i6 = _ve_vfmuls_vsv(in_b0_i6, vrgout_b0) ;
      __vr vrgw_i7 = _ve_vfmuls_vsv(in_b0_i7, vrgout_b0) ;

      const float in_b1_i0 = pIn[1*inDim+i+0] ;
      const float in_b1_i1 = pIn[1*inDim+i+1] ;
      const float in_b1_i2 = pIn[1*inDim+i+2] ;
      const float in_b1_i3 = pIn[1*inDim+i+3] ;
      const float in_b1_i4 = pIn[1*inDim+i+4] ;
      const float in_b1_i5 = pIn[1*inDim+i+5] ;
      const float in_b1_i6 = pIn[1*inDim+i+6] ;
      const float in_b1_i7 = pIn[1*inDim+i+7] ;
      vrgw_i0 = _ve_vfmads_vvsv(vrgw_i0, in_b1_i0, vrgout_b1) ;
      vrgw_i1 = _ve_vfmads_vvsv(vrgw_i1, in_b1_i1, vrgout_b1) ;
      vrgw_i2 = _ve_vfmads_vvsv(vrgw_i2, in_b1_i2, vrgout_b1) ;
      vrgw_i3 = _ve_vfmads_vvsv(vrgw_i3, in_b1_i3, vrgout_b1) ;
      vrgw_i4 = _ve_vfmads_vvsv(vrgw_i4, in_b1_i4, vrgout_b1) ;
      vrgw_i5 = _ve_vfmads_vvsv(vrgw_i5, in_b1_i5, vrgout_b1) ;
      vrgw_i6 = _ve_vfmads_vvsv(vrgw_i6, in_b1_i6, vrgout_b1) ;
      vrgw_i7 = _ve_vfmads_vvsv(vrgw_i7, in_b1_i7, vrgout_b1) ;

      const float in_b2_i0 = pIn[2*inDim+i+0] ;
      const float in_b2_i1 = pIn[2*inDim+i+1] ;
      const float in_b2_i2 = pIn[2*inDim+i+2] ;
      const float in_b2_i3 = pIn[2*inDim+i+3] ;
      const float in_b2_i4 = pIn[2*inDim+i+4] ;
      const float in_b2_i5 = pIn[2*inDim+i+5] ;
      const float in_b2_i6 = pIn[2*inDim+i+6] ;
      const float in_b2_i7 = pIn[2*inDim+i+7] ;
      vrgw_i0 = _ve_vfmads_vvsv(vrgw_i0, in_b2_i0, vrgout_b2) ;
      vrgw_i1 = _ve_vfmads_vvsv(vrgw_i1, in_b2_i1, vrgout_b2) ;
      vrgw_i2 = _ve_vfmads_vvsv(vrgw_i2, in_b2_i2, vrgout_b2) ;
      vrgw_i3 = _ve_vfmads_vvsv(vrgw_i3, in_b2_i3, vrgout_b2) ;
      vrgw_i4 = _ve_vfmads_vvsv(vrgw_i4, in_b2_i4, vrgout_b2) ;
      vrgw_i5 = _ve_vfmads_vvsv(vrgw_i5, in_b2_i5, vrgout_b2) ;
      vrgw_i6 = _ve_vfmads_vvsv(vrgw_i6, in_b2_i6, vrgout_b2) ;
      vrgw_i7 = _ve_vfmads_vvsv(vrgw_i7, in_b2_i7, vrgout_b2) ;

      _ve_vstu_vss(vrgw_i0, 4, pGWeight+(i+0)*outDim+o) ;
      _ve_vstu_vss(vrgw_i1, 4, pGWeight+(i+1)*outDim+o) ;
      _ve_vstu_vss(vrgw_i2, 4, pGWeight+(i+2)*outDim+o) ;
      _ve_vstu_vss(vrgw_i3, 4, pGWeight+(i+3)*outDim+o) ;
      _ve_vstu_vss(vrgw_i4, 4, pGWeight+(i+4)*outDim+o) ;
      _ve_vstu_vss(vrgw_i5, 4, pGWeight+(i+5)*outDim+o) ;
      _ve_vstu_vss(vrgw_i6, 4, pGWeight+(i+6)*outDim+o) ;
      _ve_vstu_vss(vrgw_i7, 4, pGWeight+(i+7)*outDim+o) ;
    }
  }
}

static inline void b4(
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
    _ve_lvl(vl) ;

    __vr vrgout_b0 = _ve_vldu_vss(4, pGOut+0*outDim+o) ;
    __vr vrgout_b1 = _ve_vldu_vss(4, pGOut+1*outDim+o) ;
    __vr vrgout_b2 = _ve_vldu_vss(4, pGOut+2*outDim+o) ;
    __vr vrgout_b3 = _ve_vldu_vss(4, pGOut+3*outDim+o) ;

    int64_t i=0;
    if(nInDim & 0x1) {
      const float in_b0 = pIn[0*inDim+i] ;
      __vr vrgw = _ve_vfmuls_vsv(in_b0, vrgout_b0) ;

      const float in_b1 = pIn[1*inDim+i] ;
      vrgw = _ve_vfmads_vvsv(vrgw, in_b1, vrgout_b1) ;

      const float in_b2 = pIn[2*inDim+i] ;
      vrgw = _ve_vfmads_vvsv(vrgw, in_b2, vrgout_b2) ;

      const float in_b3 = pIn[3*inDim+i] ;
      vrgw = _ve_vfmads_vvsv(vrgw, in_b3, vrgout_b3) ;

      _ve_vstu_vss(vrgw, 4, pGWeight+i*outDim+o) ;

      i+=1;
    }
    if((nInDim>>1) & 0x1) {

      const float in_b0_i0 = pIn[0*inDim+i+0] ;
      const float in_b0_i1 = pIn[0*inDim+i+1] ;
      __vr vrgw_i0 = _ve_vfmuls_vsv(in_b0_i0, vrgout_b0) ;
      __vr vrgw_i1 = _ve_vfmuls_vsv(in_b0_i1, vrgout_b0) ;

      const float in_b1_i0 = pIn[1*inDim+i+0] ;
      const float in_b1_i1 = pIn[1*inDim+i+1] ;
      vrgw_i0 = _ve_vfmads_vvsv(vrgw_i0, in_b1_i0, vrgout_b1) ;
      vrgw_i1 = _ve_vfmads_vvsv(vrgw_i1, in_b1_i1, vrgout_b1) ;

      const float in_b2_i0 = pIn[2*inDim+i+0] ;
      const float in_b2_i1 = pIn[2*inDim+i+1] ;
      vrgw_i0 = _ve_vfmads_vvsv(vrgw_i0, in_b2_i0, vrgout_b2) ;
      vrgw_i1 = _ve_vfmads_vvsv(vrgw_i1, in_b2_i1, vrgout_b2) ;

      const float in_b3_i0 = pIn[3*inDim+i+0] ;
      const float in_b3_i1 = pIn[3*inDim+i+1] ;
      vrgw_i0 = _ve_vfmads_vvsv(vrgw_i0, in_b3_i0, vrgout_b3) ;
      vrgw_i1 = _ve_vfmads_vvsv(vrgw_i1, in_b3_i1, vrgout_b3) ;

      _ve_vstu_vss(vrgw_i0, 4, pGWeight+(i+0)*outDim+o) ;
      _ve_vstu_vss(vrgw_i1, 4, pGWeight+(i+1)*outDim+o) ;

      i+=2 ;
    }
    if((nInDim>>2) & 0x1) {

      const float in_b0_i0 = pIn[0*inDim+i+0] ;
      const float in_b0_i1 = pIn[0*inDim+i+1] ;
      const float in_b0_i2 = pIn[0*inDim+i+2] ;
      const float in_b0_i3 = pIn[0*inDim+i+3] ;
      __vr vrgw_i0 = _ve_vfmuls_vsv(in_b0_i0, vrgout_b0) ;
      __vr vrgw_i1 = _ve_vfmuls_vsv(in_b0_i1, vrgout_b0) ;
      __vr vrgw_i2 = _ve_vfmuls_vsv(in_b0_i2, vrgout_b0) ;
      __vr vrgw_i3 = _ve_vfmuls_vsv(in_b0_i3, vrgout_b0) ;

      const float in_b1_i0 = pIn[1*inDim+i+0] ;
      const float in_b1_i1 = pIn[1*inDim+i+1] ;
      const float in_b1_i2 = pIn[1*inDim+i+2] ;
      const float in_b1_i3 = pIn[1*inDim+i+3] ;
      vrgw_i0 = _ve_vfmads_vvsv(vrgw_i0, in_b1_i0, vrgout_b1) ;
      vrgw_i1 = _ve_vfmads_vvsv(vrgw_i1, in_b1_i1, vrgout_b1) ;
      vrgw_i2 = _ve_vfmads_vvsv(vrgw_i2, in_b1_i2, vrgout_b1) ;
      vrgw_i3 = _ve_vfmads_vvsv(vrgw_i3, in_b1_i3, vrgout_b1) ;

      const float in_b2_i0 = pIn[2*inDim+i+0] ;
      const float in_b2_i1 = pIn[2*inDim+i+1] ;
      const float in_b2_i2 = pIn[2*inDim+i+2] ;
      const float in_b2_i3 = pIn[2*inDim+i+3] ;
      vrgw_i0 = _ve_vfmads_vvsv(vrgw_i0, in_b2_i0, vrgout_b2) ;
      vrgw_i1 = _ve_vfmads_vvsv(vrgw_i1, in_b2_i1, vrgout_b2) ;
      vrgw_i2 = _ve_vfmads_vvsv(vrgw_i2, in_b2_i2, vrgout_b2) ;
      vrgw_i3 = _ve_vfmads_vvsv(vrgw_i3, in_b2_i3, vrgout_b2) ;

      const float in_b3_i0 = pIn[3*inDim+i+0] ;
      const float in_b3_i1 = pIn[3*inDim+i+1] ;
      const float in_b3_i2 = pIn[3*inDim+i+2] ;
      const float in_b3_i3 = pIn[3*inDim+i+3] ;
      vrgw_i0 = _ve_vfmads_vvsv(vrgw_i0, in_b3_i0, vrgout_b3) ;
      vrgw_i1 = _ve_vfmads_vvsv(vrgw_i1, in_b3_i1, vrgout_b3) ;
      vrgw_i2 = _ve_vfmads_vvsv(vrgw_i2, in_b3_i2, vrgout_b3) ;
      vrgw_i3 = _ve_vfmads_vvsv(vrgw_i3, in_b3_i3, vrgout_b3) ;

      _ve_vstu_vss(vrgw_i0, 4, pGWeight+(i+0)*outDim+o) ;
      _ve_vstu_vss(vrgw_i1, 4, pGWeight+(i+1)*outDim+o) ;
      _ve_vstu_vss(vrgw_i2, 4, pGWeight+(i+2)*outDim+o) ;
      _ve_vstu_vss(vrgw_i3, 4, pGWeight+(i+3)*outDim+o) ;

      i+=4 ;
    }
    for(; i<nInDim; i+=8) {

      const float in_b0_i0 = pIn[0*inDim+i+0] ;
      const float in_b0_i1 = pIn[0*inDim+i+1] ;
      const float in_b0_i2 = pIn[0*inDim+i+2] ;
      const float in_b0_i3 = pIn[0*inDim+i+3] ;
      const float in_b0_i4 = pIn[0*inDim+i+4] ;
      const float in_b0_i5 = pIn[0*inDim+i+5] ;
      const float in_b0_i6 = pIn[0*inDim+i+6] ;
      const float in_b0_i7 = pIn[0*inDim+i+7] ;
      __vr vrgw_i0 = _ve_vfmuls_vsv(in_b0_i0, vrgout_b0) ;
      __vr vrgw_i1 = _ve_vfmuls_vsv(in_b0_i1, vrgout_b0) ;
      __vr vrgw_i2 = _ve_vfmuls_vsv(in_b0_i2, vrgout_b0) ;
      __vr vrgw_i3 = _ve_vfmuls_vsv(in_b0_i3, vrgout_b0) ;
      __vr vrgw_i4 = _ve_vfmuls_vsv(in_b0_i4, vrgout_b0) ;
      __vr vrgw_i5 = _ve_vfmuls_vsv(in_b0_i5, vrgout_b0) ;
      __vr vrgw_i6 = _ve_vfmuls_vsv(in_b0_i6, vrgout_b0) ;
      __vr vrgw_i7 = _ve_vfmuls_vsv(in_b0_i7, vrgout_b0) ;

      const float in_b1_i0 = pIn[1*inDim+i+0] ;
      const float in_b1_i1 = pIn[1*inDim+i+1] ;
      const float in_b1_i2 = pIn[1*inDim+i+2] ;
      const float in_b1_i3 = pIn[1*inDim+i+3] ;
      const float in_b1_i4 = pIn[1*inDim+i+4] ;
      const float in_b1_i5 = pIn[1*inDim+i+5] ;
      const float in_b1_i6 = pIn[1*inDim+i+6] ;
      const float in_b1_i7 = pIn[1*inDim+i+7] ;
      vrgw_i0 = _ve_vfmads_vvsv(vrgw_i0, in_b1_i0, vrgout_b1) ;
      vrgw_i1 = _ve_vfmads_vvsv(vrgw_i1, in_b1_i1, vrgout_b1) ;
      vrgw_i2 = _ve_vfmads_vvsv(vrgw_i2, in_b1_i2, vrgout_b1) ;
      vrgw_i3 = _ve_vfmads_vvsv(vrgw_i3, in_b1_i3, vrgout_b1) ;
      vrgw_i4 = _ve_vfmads_vvsv(vrgw_i4, in_b1_i4, vrgout_b1) ;
      vrgw_i5 = _ve_vfmads_vvsv(vrgw_i5, in_b1_i5, vrgout_b1) ;
      vrgw_i6 = _ve_vfmads_vvsv(vrgw_i6, in_b1_i6, vrgout_b1) ;
      vrgw_i7 = _ve_vfmads_vvsv(vrgw_i7, in_b1_i7, vrgout_b1) ;

      const float in_b2_i0 = pIn[2*inDim+i+0] ;
      const float in_b2_i1 = pIn[2*inDim+i+1] ;
      const float in_b2_i2 = pIn[2*inDim+i+2] ;
      const float in_b2_i3 = pIn[2*inDim+i+3] ;
      const float in_b2_i4 = pIn[2*inDim+i+4] ;
      const float in_b2_i5 = pIn[2*inDim+i+5] ;
      const float in_b2_i6 = pIn[2*inDim+i+6] ;
      const float in_b2_i7 = pIn[2*inDim+i+7] ;
      vrgw_i0 = _ve_vfmads_vvsv(vrgw_i0, in_b2_i0, vrgout_b2) ;
      vrgw_i1 = _ve_vfmads_vvsv(vrgw_i1, in_b2_i1, vrgout_b2) ;
      vrgw_i2 = _ve_vfmads_vvsv(vrgw_i2, in_b2_i2, vrgout_b2) ;
      vrgw_i3 = _ve_vfmads_vvsv(vrgw_i3, in_b2_i3, vrgout_b2) ;
      vrgw_i4 = _ve_vfmads_vvsv(vrgw_i4, in_b2_i4, vrgout_b2) ;
      vrgw_i5 = _ve_vfmads_vvsv(vrgw_i5, in_b2_i5, vrgout_b2) ;
      vrgw_i6 = _ve_vfmads_vvsv(vrgw_i6, in_b2_i6, vrgout_b2) ;
      vrgw_i7 = _ve_vfmads_vvsv(vrgw_i7, in_b2_i7, vrgout_b2) ;

      const float in_b3_i0 = pIn[3*inDim+i+0] ;
      const float in_b3_i1 = pIn[3*inDim+i+1] ;
      const float in_b3_i2 = pIn[3*inDim+i+2] ;
      const float in_b3_i3 = pIn[3*inDim+i+3] ;
      const float in_b3_i4 = pIn[3*inDim+i+4] ;
      const float in_b3_i5 = pIn[3*inDim+i+5] ;
      const float in_b3_i6 = pIn[3*inDim+i+6] ;
      const float in_b3_i7 = pIn[3*inDim+i+7] ;
      vrgw_i0 = _ve_vfmads_vvsv(vrgw_i0, in_b3_i0, vrgout_b3) ;
      vrgw_i1 = _ve_vfmads_vvsv(vrgw_i1, in_b3_i1, vrgout_b3) ;
      vrgw_i2 = _ve_vfmads_vvsv(vrgw_i2, in_b3_i2, vrgout_b3) ;
      vrgw_i3 = _ve_vfmads_vvsv(vrgw_i3, in_b3_i3, vrgout_b3) ;
      vrgw_i4 = _ve_vfmads_vvsv(vrgw_i4, in_b3_i4, vrgout_b3) ;
      vrgw_i5 = _ve_vfmads_vvsv(vrgw_i5, in_b3_i5, vrgout_b3) ;
      vrgw_i6 = _ve_vfmads_vvsv(vrgw_i6, in_b3_i6, vrgout_b3) ;
      vrgw_i7 = _ve_vfmads_vvsv(vrgw_i7, in_b3_i7, vrgout_b3) ;

      _ve_vstu_vss(vrgw_i0, 4, pGWeight+(i+0)*outDim+o) ;
      _ve_vstu_vss(vrgw_i1, 4, pGWeight+(i+1)*outDim+o) ;
      _ve_vstu_vss(vrgw_i2, 4, pGWeight+(i+2)*outDim+o) ;
      _ve_vstu_vss(vrgw_i3, 4, pGWeight+(i+3)*outDim+o) ;
      _ve_vstu_vss(vrgw_i4, 4, pGWeight+(i+4)*outDim+o) ;
      _ve_vstu_vss(vrgw_i5, 4, pGWeight+(i+5)*outDim+o) ;
      _ve_vstu_vss(vrgw_i6, 4, pGWeight+(i+6)*outDim+o) ;
      _ve_vstu_vss(vrgw_i7, 4, pGWeight+(i+7)*outDim+o) ;
    }
  }
}

static inline void b4_update(
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
    _ve_lvl(vl) ;

    __vr vrgout_b0 = _ve_vldu_vss(4, pGOut+0*outDim+o) ;
    __vr vrgout_b1 = _ve_vldu_vss(4, pGOut+1*outDim+o) ;
    __vr vrgout_b2 = _ve_vldu_vss(4, pGOut+2*outDim+o) ;
    __vr vrgout_b3 = _ve_vldu_vss(4, pGOut+3*outDim+o) ;

    int64_t i=0;
    if(nInDim & 0x1) {
      __vr vrgw = _ve_vldu_vss(4, pGWeight+i*outDim+o) ;

      const float in_b0 = pIn[0*inDim+i] ;
      vrgw = _ve_vfmads_vvsv(vrgw, in_b0, vrgout_b0) ;

      const float in_b1 = pIn[1*inDim+i] ;
      vrgw = _ve_vfmads_vvsv(vrgw, in_b1, vrgout_b1) ;

      const float in_b2 = pIn[2*inDim+i] ;
      vrgw = _ve_vfmads_vvsv(vrgw, in_b2, vrgout_b2) ;

      const float in_b3 = pIn[3*inDim+i] ;
      vrgw = _ve_vfmads_vvsv(vrgw, in_b3, vrgout_b3) ;

      _ve_vstu_vss(vrgw, 4, pGWeight+i*outDim+o) ;

      i+=1;
    }
    if((nInDim>>1) & 0x1) {
      __vr vrgw_i0 = _ve_vldu_vss(4, pGWeight+(i+0)*outDim+o) ;
      __vr vrgw_i1 = _ve_vldu_vss(4, pGWeight+(i+1)*outDim+o) ;

      const float in_b0_i0 = pIn[0*inDim+i+0] ;
      const float in_b0_i1 = pIn[0*inDim+i+1] ;
      vrgw_i0 = _ve_vfmads_vvsv(vrgw_i0, in_b0_i0, vrgout_b0) ;
      vrgw_i1 = _ve_vfmads_vvsv(vrgw_i1, in_b0_i1, vrgout_b0) ;

      const float in_b1_i0 = pIn[1*inDim+i+0] ;
      const float in_b1_i1 = pIn[1*inDim+i+1] ;
      vrgw_i0 = _ve_vfmads_vvsv(vrgw_i0, in_b1_i0, vrgout_b1) ;
      vrgw_i1 = _ve_vfmads_vvsv(vrgw_i1, in_b1_i1, vrgout_b1) ;

      const float in_b2_i0 = pIn[2*inDim+i+0] ;
      const float in_b2_i1 = pIn[2*inDim+i+1] ;
      vrgw_i0 = _ve_vfmads_vvsv(vrgw_i0, in_b2_i0, vrgout_b2) ;
      vrgw_i1 = _ve_vfmads_vvsv(vrgw_i1, in_b2_i1, vrgout_b2) ;

      const float in_b3_i0 = pIn[3*inDim+i+0] ;
      const float in_b3_i1 = pIn[3*inDim+i+1] ;
      vrgw_i0 = _ve_vfmads_vvsv(vrgw_i0, in_b3_i0, vrgout_b3) ;
      vrgw_i1 = _ve_vfmads_vvsv(vrgw_i1, in_b3_i1, vrgout_b3) ;

      _ve_vstu_vss(vrgw_i0, 4, pGWeight+(i+0)*outDim+o) ;
      _ve_vstu_vss(vrgw_i1, 4, pGWeight+(i+1)*outDim+o) ;

      i+=2 ;
    }
    if((nInDim>>2) & 0x1) {
      __vr vrgw_i0 = _ve_vldu_vss(4, pGWeight+(i+0)*outDim+o) ;
      __vr vrgw_i1 = _ve_vldu_vss(4, pGWeight+(i+1)*outDim+o) ;
      __vr vrgw_i2 = _ve_vldu_vss(4, pGWeight+(i+2)*outDim+o) ;
      __vr vrgw_i3 = _ve_vldu_vss(4, pGWeight+(i+3)*outDim+o) ;

      const float in_b0_i0 = pIn[0*inDim+i+0] ;
      const float in_b0_i1 = pIn[0*inDim+i+1] ;
      const float in_b0_i2 = pIn[0*inDim+i+2] ;
      const float in_b0_i3 = pIn[0*inDim+i+3] ;
      vrgw_i0 = _ve_vfmads_vvsv(vrgw_i0, in_b0_i0, vrgout_b0) ;
      vrgw_i1 = _ve_vfmads_vvsv(vrgw_i1, in_b0_i1, vrgout_b0) ;
      vrgw_i2 = _ve_vfmads_vvsv(vrgw_i2, in_b0_i2, vrgout_b0) ;
      vrgw_i3 = _ve_vfmads_vvsv(vrgw_i3, in_b0_i3, vrgout_b0) ;

      const float in_b1_i0 = pIn[1*inDim+i+0] ;
      const float in_b1_i1 = pIn[1*inDim+i+1] ;
      const float in_b1_i2 = pIn[1*inDim+i+2] ;
      const float in_b1_i3 = pIn[1*inDim+i+3] ;
      vrgw_i0 = _ve_vfmads_vvsv(vrgw_i0, in_b1_i0, vrgout_b1) ;
      vrgw_i1 = _ve_vfmads_vvsv(vrgw_i1, in_b1_i1, vrgout_b1) ;
      vrgw_i2 = _ve_vfmads_vvsv(vrgw_i2, in_b1_i2, vrgout_b1) ;
      vrgw_i3 = _ve_vfmads_vvsv(vrgw_i3, in_b1_i3, vrgout_b1) ;

      const float in_b2_i0 = pIn[2*inDim+i+0] ;
      const float in_b2_i1 = pIn[2*inDim+i+1] ;
      const float in_b2_i2 = pIn[2*inDim+i+2] ;
      const float in_b2_i3 = pIn[2*inDim+i+3] ;
      vrgw_i0 = _ve_vfmads_vvsv(vrgw_i0, in_b2_i0, vrgout_b2) ;
      vrgw_i1 = _ve_vfmads_vvsv(vrgw_i1, in_b2_i1, vrgout_b2) ;
      vrgw_i2 = _ve_vfmads_vvsv(vrgw_i2, in_b2_i2, vrgout_b2) ;
      vrgw_i3 = _ve_vfmads_vvsv(vrgw_i3, in_b2_i3, vrgout_b2) ;

      const float in_b3_i0 = pIn[3*inDim+i+0] ;
      const float in_b3_i1 = pIn[3*inDim+i+1] ;
      const float in_b3_i2 = pIn[3*inDim+i+2] ;
      const float in_b3_i3 = pIn[3*inDim+i+3] ;
      vrgw_i0 = _ve_vfmads_vvsv(vrgw_i0, in_b3_i0, vrgout_b3) ;
      vrgw_i1 = _ve_vfmads_vvsv(vrgw_i1, in_b3_i1, vrgout_b3) ;
      vrgw_i2 = _ve_vfmads_vvsv(vrgw_i2, in_b3_i2, vrgout_b3) ;
      vrgw_i3 = _ve_vfmads_vvsv(vrgw_i3, in_b3_i3, vrgout_b3) ;

      _ve_vstu_vss(vrgw_i0, 4, pGWeight+(i+0)*outDim+o) ;
      _ve_vstu_vss(vrgw_i1, 4, pGWeight+(i+1)*outDim+o) ;
      _ve_vstu_vss(vrgw_i2, 4, pGWeight+(i+2)*outDim+o) ;
      _ve_vstu_vss(vrgw_i3, 4, pGWeight+(i+3)*outDim+o) ;

      i+=4 ;
    }
    for(; i<nInDim; i+=8) {
      __vr vrgw_i0 = _ve_vldu_vss(4, pGWeight+(i+0)*outDim+o) ;
      __vr vrgw_i1 = _ve_vldu_vss(4, pGWeight+(i+1)*outDim+o) ;
      __vr vrgw_i2 = _ve_vldu_vss(4, pGWeight+(i+2)*outDim+o) ;
      __vr vrgw_i3 = _ve_vldu_vss(4, pGWeight+(i+3)*outDim+o) ;
      __vr vrgw_i4 = _ve_vldu_vss(4, pGWeight+(i+4)*outDim+o) ;
      __vr vrgw_i5 = _ve_vldu_vss(4, pGWeight+(i+5)*outDim+o) ;
      __vr vrgw_i6 = _ve_vldu_vss(4, pGWeight+(i+6)*outDim+o) ;
      __vr vrgw_i7 = _ve_vldu_vss(4, pGWeight+(i+7)*outDim+o) ;

      const float in_b0_i0 = pIn[0*inDim+i+0] ;
      const float in_b0_i1 = pIn[0*inDim+i+1] ;
      const float in_b0_i2 = pIn[0*inDim+i+2] ;
      const float in_b0_i3 = pIn[0*inDim+i+3] ;
      const float in_b0_i4 = pIn[0*inDim+i+4] ;
      const float in_b0_i5 = pIn[0*inDim+i+5] ;
      const float in_b0_i6 = pIn[0*inDim+i+6] ;
      const float in_b0_i7 = pIn[0*inDim+i+7] ;
      vrgw_i0 = _ve_vfmads_vvsv(vrgw_i0, in_b0_i0, vrgout_b0) ;
      vrgw_i1 = _ve_vfmads_vvsv(vrgw_i1, in_b0_i1, vrgout_b0) ;
      vrgw_i2 = _ve_vfmads_vvsv(vrgw_i2, in_b0_i2, vrgout_b0) ;
      vrgw_i3 = _ve_vfmads_vvsv(vrgw_i3, in_b0_i3, vrgout_b0) ;
      vrgw_i4 = _ve_vfmads_vvsv(vrgw_i4, in_b0_i4, vrgout_b0) ;
      vrgw_i5 = _ve_vfmads_vvsv(vrgw_i5, in_b0_i5, vrgout_b0) ;
      vrgw_i6 = _ve_vfmads_vvsv(vrgw_i6, in_b0_i6, vrgout_b0) ;
      vrgw_i7 = _ve_vfmads_vvsv(vrgw_i7, in_b0_i7, vrgout_b0) ;

      const float in_b1_i0 = pIn[1*inDim+i+0] ;
      const float in_b1_i1 = pIn[1*inDim+i+1] ;
      const float in_b1_i2 = pIn[1*inDim+i+2] ;
      const float in_b1_i3 = pIn[1*inDim+i+3] ;
      const float in_b1_i4 = pIn[1*inDim+i+4] ;
      const float in_b1_i5 = pIn[1*inDim+i+5] ;
      const float in_b1_i6 = pIn[1*inDim+i+6] ;
      const float in_b1_i7 = pIn[1*inDim+i+7] ;
      vrgw_i0 = _ve_vfmads_vvsv(vrgw_i0, in_b1_i0, vrgout_b1) ;
      vrgw_i1 = _ve_vfmads_vvsv(vrgw_i1, in_b1_i1, vrgout_b1) ;
      vrgw_i2 = _ve_vfmads_vvsv(vrgw_i2, in_b1_i2, vrgout_b1) ;
      vrgw_i3 = _ve_vfmads_vvsv(vrgw_i3, in_b1_i3, vrgout_b1) ;
      vrgw_i4 = _ve_vfmads_vvsv(vrgw_i4, in_b1_i4, vrgout_b1) ;
      vrgw_i5 = _ve_vfmads_vvsv(vrgw_i5, in_b1_i5, vrgout_b1) ;
      vrgw_i6 = _ve_vfmads_vvsv(vrgw_i6, in_b1_i6, vrgout_b1) ;
      vrgw_i7 = _ve_vfmads_vvsv(vrgw_i7, in_b1_i7, vrgout_b1) ;

      const float in_b2_i0 = pIn[2*inDim+i+0] ;
      const float in_b2_i1 = pIn[2*inDim+i+1] ;
      const float in_b2_i2 = pIn[2*inDim+i+2] ;
      const float in_b2_i3 = pIn[2*inDim+i+3] ;
      const float in_b2_i4 = pIn[2*inDim+i+4] ;
      const float in_b2_i5 = pIn[2*inDim+i+5] ;
      const float in_b2_i6 = pIn[2*inDim+i+6] ;
      const float in_b2_i7 = pIn[2*inDim+i+7] ;
      vrgw_i0 = _ve_vfmads_vvsv(vrgw_i0, in_b2_i0, vrgout_b2) ;
      vrgw_i1 = _ve_vfmads_vvsv(vrgw_i1, in_b2_i1, vrgout_b2) ;
      vrgw_i2 = _ve_vfmads_vvsv(vrgw_i2, in_b2_i2, vrgout_b2) ;
      vrgw_i3 = _ve_vfmads_vvsv(vrgw_i3, in_b2_i3, vrgout_b2) ;
      vrgw_i4 = _ve_vfmads_vvsv(vrgw_i4, in_b2_i4, vrgout_b2) ;
      vrgw_i5 = _ve_vfmads_vvsv(vrgw_i5, in_b2_i5, vrgout_b2) ;
      vrgw_i6 = _ve_vfmads_vvsv(vrgw_i6, in_b2_i6, vrgout_b2) ;
      vrgw_i7 = _ve_vfmads_vvsv(vrgw_i7, in_b2_i7, vrgout_b2) ;

      const float in_b3_i0 = pIn[3*inDim+i+0] ;
      const float in_b3_i1 = pIn[3*inDim+i+1] ;
      const float in_b3_i2 = pIn[3*inDim+i+2] ;
      const float in_b3_i3 = pIn[3*inDim+i+3] ;
      const float in_b3_i4 = pIn[3*inDim+i+4] ;
      const float in_b3_i5 = pIn[3*inDim+i+5] ;
      const float in_b3_i6 = pIn[3*inDim+i+6] ;
      const float in_b3_i7 = pIn[3*inDim+i+7] ;
      vrgw_i0 = _ve_vfmads_vvsv(vrgw_i0, in_b3_i0, vrgout_b3) ;
      vrgw_i1 = _ve_vfmads_vvsv(vrgw_i1, in_b3_i1, vrgout_b3) ;
      vrgw_i2 = _ve_vfmads_vvsv(vrgw_i2, in_b3_i2, vrgout_b3) ;
      vrgw_i3 = _ve_vfmads_vvsv(vrgw_i3, in_b3_i3, vrgout_b3) ;
      vrgw_i4 = _ve_vfmads_vvsv(vrgw_i4, in_b3_i4, vrgout_b3) ;
      vrgw_i5 = _ve_vfmads_vvsv(vrgw_i5, in_b3_i5, vrgout_b3) ;
      vrgw_i6 = _ve_vfmads_vvsv(vrgw_i6, in_b3_i6, vrgout_b3) ;
      vrgw_i7 = _ve_vfmads_vvsv(vrgw_i7, in_b3_i7, vrgout_b3) ;

      _ve_vstu_vss(vrgw_i0, 4, pGWeight+(i+0)*outDim+o) ;
      _ve_vstu_vss(vrgw_i1, 4, pGWeight+(i+1)*outDim+o) ;
      _ve_vstu_vss(vrgw_i2, 4, pGWeight+(i+2)*outDim+o) ;
      _ve_vstu_vss(vrgw_i3, 4, pGWeight+(i+3)*outDim+o) ;
      _ve_vstu_vss(vrgw_i4, 4, pGWeight+(i+4)*outDim+o) ;
      _ve_vstu_vss(vrgw_i5, 4, pGWeight+(i+5)*outDim+o) ;
      _ve_vstu_vss(vrgw_i6, 4, pGWeight+(i+6)*outDim+o) ;
      _ve_vstu_vss(vrgw_i7, 4, pGWeight+(i+7)*outDim+o) ;
    }
  }
}


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
  const float * restrict pIn       = pDataIn;
  const float * restrict pGOut     = pDataGradOut;
  float * restrict const pGWeight  = pDataGradWeight;

#ifndef VEDNN_USE_OPENMP
    const uint64_t inDimBegin = 0 ;
    const uint64_t inDimEnd   = inDim ;
#endif

  int64_t n=0;
  int64_t batchRemain = nBatch % 4 ;

  switch( batchRemain ) {
  case 1 :
    b1(inDim, outDim, inDimEnd-inDimBegin,
       pIn+inDimBegin+n*inDim, pGOut+n*outDim, pGWeight+inDimBegin*outDim) ;
    n+=1 ;
    break ;
  case 2 :
    b2(inDim, outDim, inDimEnd-inDimBegin,
       pIn+inDimBegin+n*inDim, pGOut+n*outDim, pGWeight+inDimBegin*outDim) ;
    n+=2 ;
    break ;
  case 3 :
    b3(inDim, outDim, inDimEnd-inDimBegin,
       pIn+inDimBegin+n*inDim, pGOut+n*outDim, pGWeight+inDimBegin*outDim) ;
    n+=3 ;
    break ;
  default :
    if( nBatch >= 4 ) {
      b4(inDim, outDim, inDimEnd-inDimBegin,
         pIn+inDimBegin+n*inDim, pGOut+n*outDim, pGWeight+inDimBegin*outDim) ;
      n+=4 ;
    }
    break ;
  }

  for(; n<nBatch; n+=4) {
    b4_update(inDim, outDim, inDimEnd-inDimBegin,
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
  const float * restrict pIn       = pDataIn;
  const float * restrict pGOut     = pDataGradOut;
  float * restrict const pGWeight  = pDataGradWeight;

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
