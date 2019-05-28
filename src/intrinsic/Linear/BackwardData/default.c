#include <stdint.h>
#include <float.h>

#include <stdio.h>

#include "vednn.h"

#include "veintrin.h"
#define VLEN	(256)


static inline void b1(
  const uint64_t	inDim,
  const uint64_t	outDim,
  const float * 	pGOut,
  const float * 	pWeight,
  float * 		pGIn
)
{
  int64_t i=0 ;
  switch(inDim&0x3) {
  case 1:
    {
      _ve_lvl(VLEN) ;
      __vr vrsum_b0 = _ve_vbrdu_vs_f32(0.f) ;

      for(int64_t o=0; o<outDim; o+=VLEN) {
	const int64_t vl = outDim-o < VLEN ? outDim - o : VLEN ;
	_ve_lvl(vl) ;
	__vr vrw    = _ve_vldu_vss(4, pWeight+i*outDim+o) ;
	__vr vro_b0 = _ve_vldu_vss(4, pGOut+0*outDim+o) ;

	vrsum_b0 = _ve_vfmads_vvvv(vrsum_b0, vrw, vro_b0) ;
      }

      _ve_lvl(VLEN) ;
      vrsum_b0 = _ve_vfsums_vv(vrsum_b0);

      _ve_lvl(1) ;
      _ve_vstu_vss(vrsum_b0, 4, pGIn+0*inDim+i) ;
    }
    i+=1;
    break ;
  case 2 :
    {
      _ve_lvl(VLEN) ;
      __vr vrsum_b0_i0 = _ve_vbrdu_vs_f32(0.f) ;
      __vr vrsum_b0_i1 = _ve_vbrdu_vs_f32(0.f) ;

      for(int64_t o=0; o<outDim; o+=VLEN) {
	const int64_t vl = outDim-o < VLEN ? outDim - o : VLEN ;
	_ve_lvl(vl) ;
	__vr vrw_i0 = _ve_vldu_vss(4, pWeight+(i  )*outDim+o) ;
	__vr vrw_i1 = _ve_vldu_vss(4, pWeight+(i+1)*outDim+o) ;

	__vr vro_b0 = _ve_vldu_vss(4, pGOut+0*outDim+o) ;

	vrsum_b0_i0 = _ve_vfmads_vvvv(vrsum_b0_i0, vrw_i0, vro_b0) ;
	vrsum_b0_i1 = _ve_vfmads_vvvv(vrsum_b0_i1, vrw_i1, vro_b0) ;
      }

      _ve_lvl(VLEN) ;
      vrsum_b0_i0 = _ve_vfsums_vv(vrsum_b0_i0);
      vrsum_b0_i1 = _ve_vfsums_vv(vrsum_b0_i1);

      _ve_lvl(1) ;
      _ve_vstu_vss(vrsum_b0_i0, 4, pGIn+0*inDim+i) ;
      _ve_vstu_vss(vrsum_b0_i1, 4, pGIn+0*inDim+i+1) ;
    }

    i+=2 ;
    break ;
  case 3:
    {
      _ve_lvl(VLEN) ;
      __vr vrsum_b0_i0 = _ve_vbrdu_vs_f32(0.f) ;
      __vr vrsum_b0_i1 = _ve_vbrdu_vs_f32(0.f) ;
      __vr vrsum_b0_i2 = _ve_vbrdu_vs_f32(0.f) ;

      for(int64_t o=0; o<outDim; o+=VLEN) {
	const int64_t vl = outDim-o < VLEN ? outDim - o : VLEN ;
	_ve_lvl(vl) ;
	__vr vrw_i0 = _ve_vldu_vss(4, pWeight+(i  )*outDim+o) ;
	__vr vrw_i1 = _ve_vldu_vss(4, pWeight+(i+1)*outDim+o) ;
	__vr vrw_i2 = _ve_vldu_vss(4, pWeight+(i+2)*outDim+o) ;

	__vr vro_b0 = _ve_vldu_vss(4, pGOut+0*outDim+o) ;

	vrsum_b0_i0 = _ve_vfmads_vvvv(vrsum_b0_i0, vrw_i0, vro_b0) ;
	vrsum_b0_i1 = _ve_vfmads_vvvv(vrsum_b0_i1, vrw_i1, vro_b0) ;
	vrsum_b0_i2 = _ve_vfmads_vvvv(vrsum_b0_i2, vrw_i2, vro_b0) ;
      }

      _ve_lvl(VLEN) ;
      vrsum_b0_i0 = _ve_vfsums_vv(vrsum_b0_i0);
      vrsum_b0_i1 = _ve_vfsums_vv(vrsum_b0_i1);
      vrsum_b0_i2 = _ve_vfsums_vv(vrsum_b0_i2);

      _ve_lvl(1) ;
      _ve_vstu_vss(vrsum_b0_i0, 4, pGIn+0*inDim+i) ;
      _ve_vstu_vss(vrsum_b0_i1, 4, pGIn+0*inDim+i+1) ;
      _ve_vstu_vss(vrsum_b0_i2, 4, pGIn+0*inDim+i+2) ;

    }
    i+=3 ;
    break ;
  default : break ;
  }
  for(; i<inDim; i+=4)
  {
    _ve_lvl(VLEN) ;
    __vr vrsum_b0_i0 = _ve_vbrdu_vs_f32(0.f) ;
    __vr vrsum_b0_i1 = _ve_vbrdu_vs_f32(0.f) ;
    __vr vrsum_b0_i2 = _ve_vbrdu_vs_f32(0.f) ;
    __vr vrsum_b0_i3 = _ve_vbrdu_vs_f32(0.f) ;

    for(int64_t o=0; o<outDim; o+=VLEN) {
      const int64_t vl = outDim-o < VLEN ? outDim - o : VLEN ;
      _ve_lvl(vl) ;
      __vr vrw_i0 = _ve_vldu_vss(4, pWeight+(i  )*outDim+o) ;
      __vr vrw_i1 = _ve_vldu_vss(4, pWeight+(i+1)*outDim+o) ;
      __vr vrw_i2 = _ve_vldu_vss(4, pWeight+(i+2)*outDim+o) ;
      __vr vrw_i3 = _ve_vldu_vss(4, pWeight+(i+3)*outDim+o) ;

      __vr vro_b0 = _ve_vldu_vss(4, pGOut+0*outDim+o) ;

      vrsum_b0_i0 = _ve_vfmads_vvvv(vrsum_b0_i0, vrw_i0, vro_b0) ;
      vrsum_b0_i1 = _ve_vfmads_vvvv(vrsum_b0_i1, vrw_i1, vro_b0) ;
      vrsum_b0_i2 = _ve_vfmads_vvvv(vrsum_b0_i2, vrw_i2, vro_b0) ;
      vrsum_b0_i3 = _ve_vfmads_vvvv(vrsum_b0_i3, vrw_i3, vro_b0) ;
    }

    _ve_lvl(VLEN) ;
    vrsum_b0_i0 = _ve_vfsums_vv(vrsum_b0_i0);
    vrsum_b0_i1 = _ve_vfsums_vv(vrsum_b0_i1);
    vrsum_b0_i2 = _ve_vfsums_vv(vrsum_b0_i2);
    vrsum_b0_i3 = _ve_vfsums_vv(vrsum_b0_i3);

    _ve_lvl(1) ;
    _ve_vstu_vss(vrsum_b0_i0, 4, pGIn+0*inDim+i) ;
    _ve_vstu_vss(vrsum_b0_i1, 4, pGIn+0*inDim+i+1) ;
    _ve_vstu_vss(vrsum_b0_i2, 4, pGIn+0*inDim+i+2) ;
    _ve_vstu_vss(vrsum_b0_i3, 4, pGIn+0*inDim+i+3) ;
  }
}

static inline void b2(
  const uint64_t	inDim,
  const uint64_t	outDim,
  const float * 	pGOut,
  const float * 	pWeight,
  float * 		pGIn
)
{
  int64_t i=0 ;
  switch(inDim&0x3) {
  case 1:
    {
      _ve_lvl(VLEN) ;
      __vr vrsum_b0 = _ve_vbrdu_vs_f32(0.f) ;
      __vr vrsum_b1 = _ve_vbrdu_vs_f32(0.f) ;

      for(int64_t o=0; o<outDim; o+=VLEN) {
	const int64_t vl = outDim-o < VLEN ? outDim - o : VLEN ;
	_ve_lvl(vl) ;
	__vr vrw    = _ve_vldu_vss(4, pWeight+i*outDim+o) ;
	__vr vro_b0 = _ve_vldu_vss(4, pGOut+0*outDim+o) ;
	__vr vro_b1 = _ve_vldu_vss(4, pGOut+1*outDim+o) ;

	vrsum_b0 = _ve_vfmads_vvvv(vrsum_b0, vrw, vro_b0) ;
	vrsum_b1 = _ve_vfmads_vvvv(vrsum_b1, vrw, vro_b1) ;
      }

      _ve_lvl(VLEN) ;
      vrsum_b0 = _ve_vfsums_vv(vrsum_b0);
      vrsum_b1 = _ve_vfsums_vv(vrsum_b1);

      _ve_lvl(1) ;
      _ve_vstu_vss(vrsum_b0, 4, pGIn+0*inDim+i) ;
      _ve_vstu_vss(vrsum_b1, 4, pGIn+1*inDim+i) ;
    }
    i+=1;
    break ;
  case 2 :
    {
      _ve_lvl(VLEN) ;
      __vr vrsum_b0_i0 = _ve_vbrdu_vs_f32(0.f) ;
      __vr vrsum_b1_i0 = _ve_vbrdu_vs_f32(0.f) ;
      __vr vrsum_b0_i1 = _ve_vbrdu_vs_f32(0.f) ;
      __vr vrsum_b1_i1 = _ve_vbrdu_vs_f32(0.f) ;

      for(int64_t o=0; o<outDim; o+=VLEN) {
	const int64_t vl = outDim-o < VLEN ? outDim - o : VLEN ;
	_ve_lvl(vl) ;
	__vr vrw_i0 = _ve_vldu_vss(4, pWeight+(i  )*outDim+o) ;
	__vr vrw_i1 = _ve_vldu_vss(4, pWeight+(i+1)*outDim+o) ;

	__vr vro_b0 = _ve_vldu_vss(4, pGOut+0*outDim+o) ;
	__vr vro_b1 = _ve_vldu_vss(4, pGOut+1*outDim+o) ;

	vrsum_b0_i0 = _ve_vfmads_vvvv(vrsum_b0_i0, vrw_i0, vro_b0) ;
	vrsum_b1_i0 = _ve_vfmads_vvvv(vrsum_b1_i0, vrw_i0, vro_b1) ;

	vrsum_b0_i1 = _ve_vfmads_vvvv(vrsum_b0_i1, vrw_i1, vro_b0) ;
	vrsum_b1_i1 = _ve_vfmads_vvvv(vrsum_b1_i1, vrw_i1, vro_b1) ;
      }

      _ve_lvl(VLEN) ;
      vrsum_b0_i0 = _ve_vfsums_vv(vrsum_b0_i0);
      vrsum_b1_i0 = _ve_vfsums_vv(vrsum_b1_i0);
      vrsum_b0_i1 = _ve_vfsums_vv(vrsum_b0_i1);
      vrsum_b1_i1 = _ve_vfsums_vv(vrsum_b1_i1);

      _ve_lvl(1) ;
      _ve_vstu_vss(vrsum_b0_i0, 4, pGIn+0*inDim+i) ;
      _ve_vstu_vss(vrsum_b1_i0, 4, pGIn+1*inDim+i) ;
      _ve_vstu_vss(vrsum_b0_i1, 4, pGIn+0*inDim+i+1) ;
      _ve_vstu_vss(vrsum_b1_i1, 4, pGIn+1*inDim+i+1) ;
    }

    i+=2 ;
    break ;
  case 3:
    {
      _ve_lvl(VLEN) ;
      __vr vrsum_b0_i0 = _ve_vbrdu_vs_f32(0.f) ;
      __vr vrsum_b1_i0 = _ve_vbrdu_vs_f32(0.f) ;
      __vr vrsum_b0_i1 = _ve_vbrdu_vs_f32(0.f) ;
      __vr vrsum_b1_i1 = _ve_vbrdu_vs_f32(0.f) ;
      __vr vrsum_b0_i2 = _ve_vbrdu_vs_f32(0.f) ;
      __vr vrsum_b1_i2 = _ve_vbrdu_vs_f32(0.f) ;

      for(int64_t o=0; o<outDim; o+=VLEN) {
	const int64_t vl = outDim-o < VLEN ? outDim - o : VLEN ;
	_ve_lvl(vl) ;
	__vr vrw_i0 = _ve_vldu_vss(4, pWeight+(i  )*outDim+o) ;
	__vr vrw_i1 = _ve_vldu_vss(4, pWeight+(i+1)*outDim+o) ;
	__vr vrw_i2 = _ve_vldu_vss(4, pWeight+(i+2)*outDim+o) ;

	__vr vro_b0 = _ve_vldu_vss(4, pGOut+0*outDim+o) ;
	__vr vro_b1 = _ve_vldu_vss(4, pGOut+1*outDim+o) ;

	vrsum_b0_i0 = _ve_vfmads_vvvv(vrsum_b0_i0, vrw_i0, vro_b0) ;
	vrsum_b1_i0 = _ve_vfmads_vvvv(vrsum_b1_i0, vrw_i0, vro_b1) ;

	vrsum_b0_i1 = _ve_vfmads_vvvv(vrsum_b0_i1, vrw_i1, vro_b0) ;
	vrsum_b1_i1 = _ve_vfmads_vvvv(vrsum_b1_i1, vrw_i1, vro_b1) ;

	vrsum_b0_i2 = _ve_vfmads_vvvv(vrsum_b0_i2, vrw_i2, vro_b0) ;
	vrsum_b1_i2 = _ve_vfmads_vvvv(vrsum_b1_i2, vrw_i2, vro_b1) ;
      }

      _ve_lvl(VLEN) ;
      vrsum_b0_i0 = _ve_vfsums_vv(vrsum_b0_i0);
      vrsum_b1_i0 = _ve_vfsums_vv(vrsum_b1_i0);
      vrsum_b0_i1 = _ve_vfsums_vv(vrsum_b0_i1);
      vrsum_b1_i1 = _ve_vfsums_vv(vrsum_b1_i1);
      vrsum_b0_i2 = _ve_vfsums_vv(vrsum_b0_i2);
      vrsum_b1_i2 = _ve_vfsums_vv(vrsum_b1_i2);

      _ve_lvl(1) ;
      _ve_vstu_vss(vrsum_b0_i0, 4, pGIn+0*inDim+i) ;
      _ve_vstu_vss(vrsum_b1_i0, 4, pGIn+1*inDim+i) ;
      _ve_vstu_vss(vrsum_b0_i1, 4, pGIn+0*inDim+i+1) ;
      _ve_vstu_vss(vrsum_b1_i1, 4, pGIn+1*inDim+i+1) ;
      _ve_vstu_vss(vrsum_b0_i2, 4, pGIn+0*inDim+i+2) ;
      _ve_vstu_vss(vrsum_b1_i2, 4, pGIn+1*inDim+i+2) ;

    }
    i+=3 ;
    break ;
  default : break ;
  }
  for(; i<inDim; i+=4)
  {
    _ve_lvl(VLEN) ;
    __vr vrsum_b0_i0 = _ve_vbrdu_vs_f32(0.f) ;
    __vr vrsum_b1_i0 = _ve_vbrdu_vs_f32(0.f) ;
    __vr vrsum_b0_i1 = _ve_vbrdu_vs_f32(0.f) ;
    __vr vrsum_b1_i1 = _ve_vbrdu_vs_f32(0.f) ;
    __vr vrsum_b0_i2 = _ve_vbrdu_vs_f32(0.f) ;
    __vr vrsum_b1_i2 = _ve_vbrdu_vs_f32(0.f) ;
    __vr vrsum_b0_i3 = _ve_vbrdu_vs_f32(0.f) ;
    __vr vrsum_b1_i3 = _ve_vbrdu_vs_f32(0.f) ;

    for(int64_t o=0; o<outDim; o+=VLEN) {
      const int64_t vl = outDim-o < VLEN ? outDim - o : VLEN ;
      _ve_lvl(vl) ;
      __vr vrw_i0 = _ve_vldu_vss(4, pWeight+(i  )*outDim+o) ;
      __vr vrw_i1 = _ve_vldu_vss(4, pWeight+(i+1)*outDim+o) ;
      __vr vrw_i2 = _ve_vldu_vss(4, pWeight+(i+2)*outDim+o) ;
      __vr vrw_i3 = _ve_vldu_vss(4, pWeight+(i+3)*outDim+o) ;

      __vr vro_b0 = _ve_vldu_vss(4, pGOut+0*outDim+o) ;
      __vr vro_b1 = _ve_vldu_vss(4, pGOut+1*outDim+o) ;

      vrsum_b0_i0 = _ve_vfmads_vvvv(vrsum_b0_i0, vrw_i0, vro_b0) ;
      vrsum_b1_i0 = _ve_vfmads_vvvv(vrsum_b1_i0, vrw_i0, vro_b1) ;

      vrsum_b0_i1 = _ve_vfmads_vvvv(vrsum_b0_i1, vrw_i1, vro_b0) ;
      vrsum_b1_i1 = _ve_vfmads_vvvv(vrsum_b1_i1, vrw_i1, vro_b1) ;

      vrsum_b0_i2 = _ve_vfmads_vvvv(vrsum_b0_i2, vrw_i2, vro_b0) ;
      vrsum_b1_i2 = _ve_vfmads_vvvv(vrsum_b1_i2, vrw_i2, vro_b1) ;

      vrsum_b0_i3 = _ve_vfmads_vvvv(vrsum_b0_i3, vrw_i3, vro_b0) ;
      vrsum_b1_i3 = _ve_vfmads_vvvv(vrsum_b1_i3, vrw_i3, vro_b1) ;
    }

    _ve_lvl(VLEN) ;
    vrsum_b0_i0 = _ve_vfsums_vv(vrsum_b0_i0);
    vrsum_b1_i0 = _ve_vfsums_vv(vrsum_b1_i0);
    vrsum_b0_i1 = _ve_vfsums_vv(vrsum_b0_i1);
    vrsum_b1_i1 = _ve_vfsums_vv(vrsum_b1_i1);
    vrsum_b0_i2 = _ve_vfsums_vv(vrsum_b0_i2);
    vrsum_b1_i2 = _ve_vfsums_vv(vrsum_b1_i2);
    vrsum_b0_i3 = _ve_vfsums_vv(vrsum_b0_i3);
    vrsum_b1_i3 = _ve_vfsums_vv(vrsum_b1_i3);

    _ve_lvl(1) ;
    _ve_vstu_vss(vrsum_b0_i0, 4, pGIn+0*inDim+i) ;
    _ve_vstu_vss(vrsum_b1_i0, 4, pGIn+1*inDim+i) ;
    _ve_vstu_vss(vrsum_b0_i1, 4, pGIn+0*inDim+i+1) ;
    _ve_vstu_vss(vrsum_b1_i1, 4, pGIn+1*inDim+i+1) ;
    _ve_vstu_vss(vrsum_b0_i2, 4, pGIn+0*inDim+i+2) ;
    _ve_vstu_vss(vrsum_b1_i2, 4, pGIn+1*inDim+i+2) ;
    _ve_vstu_vss(vrsum_b0_i3, 4, pGIn+0*inDim+i+3) ;
    _ve_vstu_vss(vrsum_b1_i3, 4, pGIn+1*inDim+i+3) ;
  }
}


static inline void b3(
  const uint64_t	inDim,
  const uint64_t	outDim,
  const float * 	pGOut,
  const float * 	pWeight,
  float * 		pGIn
)
{
  int64_t i=0 ;
  switch(inDim&0x3) {
  case 1:
    {
      _ve_lvl(VLEN) ;
      __vr vrsum_b0 = _ve_vbrdu_vs_f32(0.f) ;
      __vr vrsum_b1 = _ve_vbrdu_vs_f32(0.f) ;
      __vr vrsum_b2 = _ve_vbrdu_vs_f32(0.f) ;

      for(int64_t o=0; o<outDim; o+=VLEN) {
	const int64_t vl = outDim-o < VLEN ? outDim - o : VLEN ;
	_ve_lvl(vl) ;
	__vr vrw    = _ve_vldu_vss(4, pWeight+i*outDim+o) ;
	__vr vro_b0 = _ve_vldu_vss(4, pGOut+0*outDim+o) ;
	__vr vro_b1 = _ve_vldu_vss(4, pGOut+1*outDim+o) ;
	__vr vro_b2 = _ve_vldu_vss(4, pGOut+2*outDim+o) ;

	vrsum_b0 = _ve_vfmads_vvvv(vrsum_b0, vrw, vro_b0) ;
	vrsum_b1 = _ve_vfmads_vvvv(vrsum_b1, vrw, vro_b1) ;
	vrsum_b2 = _ve_vfmads_vvvv(vrsum_b2, vrw, vro_b2) ;
      }

      _ve_lvl(VLEN) ;
      vrsum_b0 = _ve_vfsums_vv(vrsum_b0);
      vrsum_b1 = _ve_vfsums_vv(vrsum_b1);
      vrsum_b2 = _ve_vfsums_vv(vrsum_b2);

      _ve_lvl(1) ;
      _ve_vstu_vss(vrsum_b0, 4, pGIn+0*inDim+i) ;
      _ve_vstu_vss(vrsum_b1, 4, pGIn+1*inDim+i) ;
      _ve_vstu_vss(vrsum_b2, 4, pGIn+2*inDim+i) ;
    }
    i+=1;
    break ;
  case 2 :
    {
      _ve_lvl(VLEN) ;
      __vr vrsum_b0_i0 = _ve_vbrdu_vs_f32(0.f) ;
      __vr vrsum_b1_i0 = _ve_vbrdu_vs_f32(0.f) ;
      __vr vrsum_b2_i0 = _ve_vbrdu_vs_f32(0.f) ;
      __vr vrsum_b0_i1 = _ve_vbrdu_vs_f32(0.f) ;
      __vr vrsum_b1_i1 = _ve_vbrdu_vs_f32(0.f) ;
      __vr vrsum_b2_i1 = _ve_vbrdu_vs_f32(0.f) ;

      for(int64_t o=0; o<outDim; o+=VLEN) {
	const int64_t vl = outDim-o < VLEN ? outDim - o : VLEN ;
	_ve_lvl(vl) ;
	__vr vrw_i0 = _ve_vldu_vss(4, pWeight+(i  )*outDim+o) ;
	__vr vrw_i1 = _ve_vldu_vss(4, pWeight+(i+1)*outDim+o) ;

	__vr vro_b0 = _ve_vldu_vss(4, pGOut+0*outDim+o) ;
	__vr vro_b1 = _ve_vldu_vss(4, pGOut+1*outDim+o) ;
	__vr vro_b2 = _ve_vldu_vss(4, pGOut+2*outDim+o) ;

	vrsum_b0_i0 = _ve_vfmads_vvvv(vrsum_b0_i0, vrw_i0, vro_b0) ;
	vrsum_b1_i0 = _ve_vfmads_vvvv(vrsum_b1_i0, vrw_i0, vro_b1) ;
	vrsum_b2_i0 = _ve_vfmads_vvvv(vrsum_b2_i0, vrw_i0, vro_b2) ;

	vrsum_b0_i1 = _ve_vfmads_vvvv(vrsum_b0_i1, vrw_i1, vro_b0) ;
	vrsum_b1_i1 = _ve_vfmads_vvvv(vrsum_b1_i1, vrw_i1, vro_b1) ;
	vrsum_b2_i1 = _ve_vfmads_vvvv(vrsum_b2_i1, vrw_i1, vro_b2) ;
      }

      _ve_lvl(VLEN) ;
      vrsum_b0_i0 = _ve_vfsums_vv(vrsum_b0_i0);
      vrsum_b1_i0 = _ve_vfsums_vv(vrsum_b1_i0);
      vrsum_b2_i0 = _ve_vfsums_vv(vrsum_b2_i0);
      vrsum_b0_i1 = _ve_vfsums_vv(vrsum_b0_i1);
      vrsum_b1_i1 = _ve_vfsums_vv(vrsum_b1_i1);
      vrsum_b2_i1 = _ve_vfsums_vv(vrsum_b2_i1);

      _ve_lvl(1) ;
      _ve_vstu_vss(vrsum_b0_i0, 4, pGIn+0*inDim+i) ;
      _ve_vstu_vss(vrsum_b1_i0, 4, pGIn+1*inDim+i) ;
      _ve_vstu_vss(vrsum_b2_i0, 4, pGIn+2*inDim+i) ;
      _ve_vstu_vss(vrsum_b0_i1, 4, pGIn+0*inDim+i+1) ;
      _ve_vstu_vss(vrsum_b1_i1, 4, pGIn+1*inDim+i+1) ;
      _ve_vstu_vss(vrsum_b2_i1, 4, pGIn+2*inDim+i+1) ;
    }

    i+=2 ;
    break ;
  case 3:
    {
      _ve_lvl(VLEN) ;
      __vr vrsum_b0_i0 = _ve_vbrdu_vs_f32(0.f) ;
      __vr vrsum_b1_i0 = _ve_vbrdu_vs_f32(0.f) ;
      __vr vrsum_b2_i0 = _ve_vbrdu_vs_f32(0.f) ;
      __vr vrsum_b0_i1 = _ve_vbrdu_vs_f32(0.f) ;
      __vr vrsum_b1_i1 = _ve_vbrdu_vs_f32(0.f) ;
      __vr vrsum_b2_i1 = _ve_vbrdu_vs_f32(0.f) ;
      __vr vrsum_b0_i2 = _ve_vbrdu_vs_f32(0.f) ;
      __vr vrsum_b1_i2 = _ve_vbrdu_vs_f32(0.f) ;
      __vr vrsum_b2_i2 = _ve_vbrdu_vs_f32(0.f) ;

      for(int64_t o=0; o<outDim; o+=VLEN) {
	const int64_t vl = outDim-o < VLEN ? outDim - o : VLEN ;
	_ve_lvl(vl) ;
	__vr vrw_i0 = _ve_vldu_vss(4, pWeight+(i  )*outDim+o) ;
	__vr vrw_i1 = _ve_vldu_vss(4, pWeight+(i+1)*outDim+o) ;
	__vr vrw_i2 = _ve_vldu_vss(4, pWeight+(i+2)*outDim+o) ;

	__vr vro_b0 = _ve_vldu_vss(4, pGOut+0*outDim+o) ;
	__vr vro_b1 = _ve_vldu_vss(4, pGOut+1*outDim+o) ;
	__vr vro_b2 = _ve_vldu_vss(4, pGOut+2*outDim+o) ;

	vrsum_b0_i0 = _ve_vfmads_vvvv(vrsum_b0_i0, vrw_i0, vro_b0) ;
	vrsum_b1_i0 = _ve_vfmads_vvvv(vrsum_b1_i0, vrw_i0, vro_b1) ;
	vrsum_b2_i0 = _ve_vfmads_vvvv(vrsum_b2_i0, vrw_i0, vro_b2) ;

	vrsum_b0_i1 = _ve_vfmads_vvvv(vrsum_b0_i1, vrw_i1, vro_b0) ;
	vrsum_b1_i1 = _ve_vfmads_vvvv(vrsum_b1_i1, vrw_i1, vro_b1) ;
	vrsum_b2_i1 = _ve_vfmads_vvvv(vrsum_b2_i1, vrw_i1, vro_b2) ;

	vrsum_b0_i2 = _ve_vfmads_vvvv(vrsum_b0_i2, vrw_i2, vro_b0) ;
	vrsum_b1_i2 = _ve_vfmads_vvvv(vrsum_b1_i2, vrw_i2, vro_b1) ;
	vrsum_b2_i2 = _ve_vfmads_vvvv(vrsum_b2_i2, vrw_i2, vro_b2) ;
      }

      _ve_lvl(VLEN) ;
      vrsum_b0_i0 = _ve_vfsums_vv(vrsum_b0_i0);
      vrsum_b1_i0 = _ve_vfsums_vv(vrsum_b1_i0);
      vrsum_b2_i0 = _ve_vfsums_vv(vrsum_b2_i0);
      vrsum_b0_i1 = _ve_vfsums_vv(vrsum_b0_i1);
      vrsum_b1_i1 = _ve_vfsums_vv(vrsum_b1_i1);
      vrsum_b2_i1 = _ve_vfsums_vv(vrsum_b2_i1);
      vrsum_b0_i2 = _ve_vfsums_vv(vrsum_b0_i2);
      vrsum_b1_i2 = _ve_vfsums_vv(vrsum_b1_i2);
      vrsum_b2_i2 = _ve_vfsums_vv(vrsum_b2_i2);

      _ve_lvl(1) ;
      _ve_vstu_vss(vrsum_b0_i0, 4, pGIn+0*inDim+i) ;
      _ve_vstu_vss(vrsum_b1_i0, 4, pGIn+1*inDim+i) ;
      _ve_vstu_vss(vrsum_b2_i0, 4, pGIn+2*inDim+i) ;
      _ve_vstu_vss(vrsum_b0_i1, 4, pGIn+0*inDim+i+1) ;
      _ve_vstu_vss(vrsum_b1_i1, 4, pGIn+1*inDim+i+1) ;
      _ve_vstu_vss(vrsum_b2_i1, 4, pGIn+2*inDim+i+1) ;
      _ve_vstu_vss(vrsum_b0_i2, 4, pGIn+0*inDim+i+2) ;
      _ve_vstu_vss(vrsum_b1_i2, 4, pGIn+1*inDim+i+2) ;
      _ve_vstu_vss(vrsum_b2_i2, 4, pGIn+2*inDim+i+2) ;

    }
    i+=3 ;
    break ;
  default : break ;
  }
  for(; i<inDim; i+=4)
  {
    _ve_lvl(VLEN) ;
    __vr vrsum_b0_i0 = _ve_vbrdu_vs_f32(0.f) ;
    __vr vrsum_b1_i0 = _ve_vbrdu_vs_f32(0.f) ;
    __vr vrsum_b2_i0 = _ve_vbrdu_vs_f32(0.f) ;
    __vr vrsum_b0_i1 = _ve_vbrdu_vs_f32(0.f) ;
    __vr vrsum_b1_i1 = _ve_vbrdu_vs_f32(0.f) ;
    __vr vrsum_b2_i1 = _ve_vbrdu_vs_f32(0.f) ;
    __vr vrsum_b0_i2 = _ve_vbrdu_vs_f32(0.f) ;
    __vr vrsum_b1_i2 = _ve_vbrdu_vs_f32(0.f) ;
    __vr vrsum_b2_i2 = _ve_vbrdu_vs_f32(0.f) ;
    __vr vrsum_b0_i3 = _ve_vbrdu_vs_f32(0.f) ;
    __vr vrsum_b1_i3 = _ve_vbrdu_vs_f32(0.f) ;
    __vr vrsum_b2_i3 = _ve_vbrdu_vs_f32(0.f) ;

    for(int64_t o=0; o<outDim; o+=VLEN) {
      const int64_t vl = outDim-o < VLEN ? outDim - o : VLEN ;
      _ve_lvl(vl) ;
      __vr vrw_i0 = _ve_vldu_vss(4, pWeight+(i  )*outDim+o) ;
      __vr vrw_i1 = _ve_vldu_vss(4, pWeight+(i+1)*outDim+o) ;
      __vr vrw_i2 = _ve_vldu_vss(4, pWeight+(i+2)*outDim+o) ;
      __vr vrw_i3 = _ve_vldu_vss(4, pWeight+(i+3)*outDim+o) ;

      __vr vro_b0 = _ve_vldu_vss(4, pGOut+0*outDim+o) ;
      __vr vro_b1 = _ve_vldu_vss(4, pGOut+1*outDim+o) ;
      __vr vro_b2 = _ve_vldu_vss(4, pGOut+2*outDim+o) ;

      vrsum_b0_i0 = _ve_vfmads_vvvv(vrsum_b0_i0, vrw_i0, vro_b0) ;
      vrsum_b1_i0 = _ve_vfmads_vvvv(vrsum_b1_i0, vrw_i0, vro_b1) ;
      vrsum_b2_i0 = _ve_vfmads_vvvv(vrsum_b2_i0, vrw_i0, vro_b2) ;

      vrsum_b0_i1 = _ve_vfmads_vvvv(vrsum_b0_i1, vrw_i1, vro_b0) ;
      vrsum_b1_i1 = _ve_vfmads_vvvv(vrsum_b1_i1, vrw_i1, vro_b1) ;
      vrsum_b2_i1 = _ve_vfmads_vvvv(vrsum_b2_i1, vrw_i1, vro_b2) ;

      vrsum_b0_i2 = _ve_vfmads_vvvv(vrsum_b0_i2, vrw_i2, vro_b0) ;
      vrsum_b1_i2 = _ve_vfmads_vvvv(vrsum_b1_i2, vrw_i2, vro_b1) ;
      vrsum_b2_i2 = _ve_vfmads_vvvv(vrsum_b2_i2, vrw_i2, vro_b2) ;

      vrsum_b0_i3 = _ve_vfmads_vvvv(vrsum_b0_i3, vrw_i3, vro_b0) ;
      vrsum_b1_i3 = _ve_vfmads_vvvv(vrsum_b1_i3, vrw_i3, vro_b1) ;
      vrsum_b2_i3 = _ve_vfmads_vvvv(vrsum_b2_i3, vrw_i3, vro_b2) ;
    }

    _ve_lvl(VLEN) ;
    vrsum_b0_i0 = _ve_vfsums_vv(vrsum_b0_i0);
    vrsum_b1_i0 = _ve_vfsums_vv(vrsum_b1_i0);
    vrsum_b2_i0 = _ve_vfsums_vv(vrsum_b2_i0);
    vrsum_b0_i1 = _ve_vfsums_vv(vrsum_b0_i1);
    vrsum_b1_i1 = _ve_vfsums_vv(vrsum_b1_i1);
    vrsum_b2_i1 = _ve_vfsums_vv(vrsum_b2_i1);
    vrsum_b0_i2 = _ve_vfsums_vv(vrsum_b0_i2);
    vrsum_b1_i2 = _ve_vfsums_vv(vrsum_b1_i2);
    vrsum_b2_i2 = _ve_vfsums_vv(vrsum_b2_i2);
    vrsum_b0_i3 = _ve_vfsums_vv(vrsum_b0_i3);
    vrsum_b1_i3 = _ve_vfsums_vv(vrsum_b1_i3);
    vrsum_b2_i3 = _ve_vfsums_vv(vrsum_b2_i3);

    _ve_lvl(1) ;
    _ve_vstu_vss(vrsum_b0_i0, 4, pGIn+0*inDim+i) ;
    _ve_vstu_vss(vrsum_b1_i0, 4, pGIn+1*inDim+i) ;
    _ve_vstu_vss(vrsum_b2_i0, 4, pGIn+2*inDim+i) ;
    _ve_vstu_vss(vrsum_b0_i1, 4, pGIn+0*inDim+i+1) ;
    _ve_vstu_vss(vrsum_b1_i1, 4, pGIn+1*inDim+i+1) ;
    _ve_vstu_vss(vrsum_b2_i1, 4, pGIn+2*inDim+i+1) ;
    _ve_vstu_vss(vrsum_b0_i2, 4, pGIn+0*inDim+i+2) ;
    _ve_vstu_vss(vrsum_b1_i2, 4, pGIn+1*inDim+i+2) ;
    _ve_vstu_vss(vrsum_b2_i2, 4, pGIn+2*inDim+i+2) ;
    _ve_vstu_vss(vrsum_b0_i3, 4, pGIn+0*inDim+i+3) ;
    _ve_vstu_vss(vrsum_b1_i3, 4, pGIn+1*inDim+i+3) ;
    _ve_vstu_vss(vrsum_b2_i3, 4, pGIn+2*inDim+i+3) ;
  }
}


static inline void b4(
  const uint64_t	inDim,
  const uint64_t	outDim,
  const float * 	pGOut,
  const float * 	pWeight,
  float * 		pGIn
)
{
  int64_t i=0 ;
  switch(inDim&0x3) {
  case 1:
    {
      _ve_lvl(VLEN) ;
      __vr vrsum_b0 = _ve_vbrdu_vs_f32(0.f) ;
      __vr vrsum_b1 = _ve_vbrdu_vs_f32(0.f) ;
      __vr vrsum_b2 = _ve_vbrdu_vs_f32(0.f) ;
      __vr vrsum_b3 = _ve_vbrdu_vs_f32(0.f) ;

      for(int64_t o=0; o<outDim; o+=VLEN) {
	const int64_t vl = outDim-o < VLEN ? outDim - o : VLEN ;
	_ve_lvl(vl) ;
	__vr vrw    = _ve_vldu_vss(4, pWeight+i*outDim+o) ;
	__vr vro_b0 = _ve_vldu_vss(4, pGOut+0*outDim+o) ;
	__vr vro_b1 = _ve_vldu_vss(4, pGOut+1*outDim+o) ;
	__vr vro_b2 = _ve_vldu_vss(4, pGOut+2*outDim+o) ;
	__vr vro_b3 = _ve_vldu_vss(4, pGOut+3*outDim+o) ;

	vrsum_b0 = _ve_vfmads_vvvv(vrsum_b0, vrw, vro_b0) ;
	vrsum_b1 = _ve_vfmads_vvvv(vrsum_b1, vrw, vro_b1) ;
	vrsum_b2 = _ve_vfmads_vvvv(vrsum_b2, vrw, vro_b2) ;
	vrsum_b3 = _ve_vfmads_vvvv(vrsum_b3, vrw, vro_b3) ;
      }

      _ve_lvl(VLEN) ;
      vrsum_b0 = _ve_vfsums_vv(vrsum_b0);
      vrsum_b1 = _ve_vfsums_vv(vrsum_b1);
      vrsum_b2 = _ve_vfsums_vv(vrsum_b2);
      vrsum_b3 = _ve_vfsums_vv(vrsum_b3);

      _ve_lvl(1) ;
      _ve_vstu_vss(vrsum_b0, 4, pGIn+0*inDim+i) ;
      _ve_vstu_vss(vrsum_b1, 4, pGIn+1*inDim+i) ;
      _ve_vstu_vss(vrsum_b2, 4, pGIn+2*inDim+i) ;
      _ve_vstu_vss(vrsum_b3, 4, pGIn+3*inDim+i) ;
    }
    i+=1;
    break ;
  case 2 :
    {
      _ve_lvl(VLEN) ;
      __vr vrsum_b0_i0 = _ve_vbrdu_vs_f32(0.f) ;
      __vr vrsum_b1_i0 = _ve_vbrdu_vs_f32(0.f) ;
      __vr vrsum_b2_i0 = _ve_vbrdu_vs_f32(0.f) ;
      __vr vrsum_b3_i0 = _ve_vbrdu_vs_f32(0.f) ;
      __vr vrsum_b0_i1 = _ve_vbrdu_vs_f32(0.f) ;
      __vr vrsum_b1_i1 = _ve_vbrdu_vs_f32(0.f) ;
      __vr vrsum_b2_i1 = _ve_vbrdu_vs_f32(0.f) ;
      __vr vrsum_b3_i1 = _ve_vbrdu_vs_f32(0.f) ;

      for(int64_t o=0; o<outDim; o+=VLEN) {
	const int64_t vl = outDim-o < VLEN ? outDim - o : VLEN ;
	_ve_lvl(vl) ;
	__vr vrw_i0 = _ve_vldu_vss(4, pWeight+(i  )*outDim+o) ;
	__vr vrw_i1 = _ve_vldu_vss(4, pWeight+(i+1)*outDim+o) ;

	__vr vro_b0 = _ve_vldu_vss(4, pGOut+0*outDim+o) ;
	__vr vro_b1 = _ve_vldu_vss(4, pGOut+1*outDim+o) ;
	__vr vro_b2 = _ve_vldu_vss(4, pGOut+2*outDim+o) ;
	__vr vro_b3 = _ve_vldu_vss(4, pGOut+3*outDim+o) ;

	vrsum_b0_i0 = _ve_vfmads_vvvv(vrsum_b0_i0, vrw_i0, vro_b0) ;
	vrsum_b1_i0 = _ve_vfmads_vvvv(vrsum_b1_i0, vrw_i0, vro_b1) ;
	vrsum_b2_i0 = _ve_vfmads_vvvv(vrsum_b2_i0, vrw_i0, vro_b2) ;
	vrsum_b3_i0 = _ve_vfmads_vvvv(vrsum_b3_i0, vrw_i0, vro_b3) ;

	vrsum_b0_i1 = _ve_vfmads_vvvv(vrsum_b0_i1, vrw_i1, vro_b0) ;
	vrsum_b1_i1 = _ve_vfmads_vvvv(vrsum_b1_i1, vrw_i1, vro_b1) ;
	vrsum_b2_i1 = _ve_vfmads_vvvv(vrsum_b2_i1, vrw_i1, vro_b2) ;
	vrsum_b3_i1 = _ve_vfmads_vvvv(vrsum_b3_i1, vrw_i1, vro_b3) ;
      }

      _ve_lvl(VLEN) ;
      vrsum_b0_i0 = _ve_vfsums_vv(vrsum_b0_i0);
      vrsum_b1_i0 = _ve_vfsums_vv(vrsum_b1_i0);
      vrsum_b2_i0 = _ve_vfsums_vv(vrsum_b2_i0);
      vrsum_b3_i0 = _ve_vfsums_vv(vrsum_b3_i0);
      vrsum_b0_i1 = _ve_vfsums_vv(vrsum_b0_i1);
      vrsum_b1_i1 = _ve_vfsums_vv(vrsum_b1_i1);
      vrsum_b2_i1 = _ve_vfsums_vv(vrsum_b2_i1);
      vrsum_b3_i1 = _ve_vfsums_vv(vrsum_b3_i1);


      _ve_lvl(1) ;
      _ve_vstu_vss(vrsum_b0_i0, 4, pGIn+0*inDim+i) ;
      _ve_vstu_vss(vrsum_b1_i0, 4, pGIn+1*inDim+i) ;
      _ve_vstu_vss(vrsum_b2_i0, 4, pGIn+2*inDim+i) ;
      _ve_vstu_vss(vrsum_b3_i0, 4, pGIn+3*inDim+i) ;
      _ve_vstu_vss(vrsum_b0_i1, 4, pGIn+0*inDim+i+1) ;
      _ve_vstu_vss(vrsum_b1_i1, 4, pGIn+1*inDim+i+1) ;
      _ve_vstu_vss(vrsum_b2_i1, 4, pGIn+2*inDim+i+1) ;
      _ve_vstu_vss(vrsum_b3_i1, 4, pGIn+3*inDim+i+1) ;
    }

    i+=2 ;
    break ;
  case 3:
    {
      _ve_lvl(VLEN) ;
      __vr vrsum_b0_i0 = _ve_vbrdu_vs_f32(0.f) ;
      __vr vrsum_b1_i0 = _ve_vbrdu_vs_f32(0.f) ;
      __vr vrsum_b2_i0 = _ve_vbrdu_vs_f32(0.f) ;
      __vr vrsum_b3_i0 = _ve_vbrdu_vs_f32(0.f) ;
      __vr vrsum_b0_i1 = _ve_vbrdu_vs_f32(0.f) ;
      __vr vrsum_b1_i1 = _ve_vbrdu_vs_f32(0.f) ;
      __vr vrsum_b2_i1 = _ve_vbrdu_vs_f32(0.f) ;
      __vr vrsum_b3_i1 = _ve_vbrdu_vs_f32(0.f) ;
      __vr vrsum_b0_i2 = _ve_vbrdu_vs_f32(0.f) ;
      __vr vrsum_b1_i2 = _ve_vbrdu_vs_f32(0.f) ;
      __vr vrsum_b2_i2 = _ve_vbrdu_vs_f32(0.f) ;
      __vr vrsum_b3_i2 = _ve_vbrdu_vs_f32(0.f) ;

      for(int64_t o=0; o<outDim; o+=VLEN) {
	const int64_t vl = outDim-o < VLEN ? outDim - o : VLEN ;
	_ve_lvl(vl) ;
	__vr vrw_i0 = _ve_vldu_vss(4, pWeight+(i  )*outDim+o) ;
	__vr vrw_i1 = _ve_vldu_vss(4, pWeight+(i+1)*outDim+o) ;
	__vr vrw_i2 = _ve_vldu_vss(4, pWeight+(i+2)*outDim+o) ;

	__vr vro_b0 = _ve_vldu_vss(4, pGOut+0*outDim+o) ;
	__vr vro_b1 = _ve_vldu_vss(4, pGOut+1*outDim+o) ;
	__vr vro_b2 = _ve_vldu_vss(4, pGOut+2*outDim+o) ;
	__vr vro_b3 = _ve_vldu_vss(4, pGOut+3*outDim+o) ;

	vrsum_b0_i0 = _ve_vfmads_vvvv(vrsum_b0_i0, vrw_i0, vro_b0) ;
	vrsum_b1_i0 = _ve_vfmads_vvvv(vrsum_b1_i0, vrw_i0, vro_b1) ;
	vrsum_b2_i0 = _ve_vfmads_vvvv(vrsum_b2_i0, vrw_i0, vro_b2) ;
	vrsum_b3_i0 = _ve_vfmads_vvvv(vrsum_b3_i0, vrw_i0, vro_b3) ;

	vrsum_b0_i1 = _ve_vfmads_vvvv(vrsum_b0_i1, vrw_i1, vro_b0) ;
	vrsum_b1_i1 = _ve_vfmads_vvvv(vrsum_b1_i1, vrw_i1, vro_b1) ;
	vrsum_b2_i1 = _ve_vfmads_vvvv(vrsum_b2_i1, vrw_i1, vro_b2) ;
	vrsum_b3_i1 = _ve_vfmads_vvvv(vrsum_b3_i1, vrw_i1, vro_b3) ;

	vrsum_b0_i2 = _ve_vfmads_vvvv(vrsum_b0_i2, vrw_i2, vro_b0) ;
	vrsum_b1_i2 = _ve_vfmads_vvvv(vrsum_b1_i2, vrw_i2, vro_b1) ;
	vrsum_b2_i2 = _ve_vfmads_vvvv(vrsum_b2_i2, vrw_i2, vro_b2) ;
	vrsum_b3_i2 = _ve_vfmads_vvvv(vrsum_b3_i2, vrw_i2, vro_b3) ;
      }

      _ve_lvl(VLEN) ;
      vrsum_b0_i0 = _ve_vfsums_vv(vrsum_b0_i0);
      vrsum_b1_i0 = _ve_vfsums_vv(vrsum_b1_i0);
      vrsum_b2_i0 = _ve_vfsums_vv(vrsum_b2_i0);
      vrsum_b3_i0 = _ve_vfsums_vv(vrsum_b3_i0);
      vrsum_b0_i1 = _ve_vfsums_vv(vrsum_b0_i1);
      vrsum_b1_i1 = _ve_vfsums_vv(vrsum_b1_i1);
      vrsum_b2_i1 = _ve_vfsums_vv(vrsum_b2_i1);
      vrsum_b3_i1 = _ve_vfsums_vv(vrsum_b3_i1);
      vrsum_b0_i2 = _ve_vfsums_vv(vrsum_b0_i2);
      vrsum_b1_i2 = _ve_vfsums_vv(vrsum_b1_i2);
      vrsum_b2_i2 = _ve_vfsums_vv(vrsum_b2_i2);
      vrsum_b3_i2 = _ve_vfsums_vv(vrsum_b3_i2);

      _ve_lvl(1) ;
      _ve_vstu_vss(vrsum_b0_i0, 4, pGIn+0*inDim+i) ;
      _ve_vstu_vss(vrsum_b1_i0, 4, pGIn+1*inDim+i) ;
      _ve_vstu_vss(vrsum_b2_i0, 4, pGIn+2*inDim+i) ;
      _ve_vstu_vss(vrsum_b3_i0, 4, pGIn+3*inDim+i) ;
      _ve_vstu_vss(vrsum_b0_i1, 4, pGIn+0*inDim+i+1) ;
      _ve_vstu_vss(vrsum_b1_i1, 4, pGIn+1*inDim+i+1) ;
      _ve_vstu_vss(vrsum_b2_i1, 4, pGIn+2*inDim+i+1) ;
      _ve_vstu_vss(vrsum_b3_i1, 4, pGIn+3*inDim+i+1) ;
      _ve_vstu_vss(vrsum_b0_i2, 4, pGIn+0*inDim+i+2) ;
      _ve_vstu_vss(vrsum_b1_i2, 4, pGIn+1*inDim+i+2) ;
      _ve_vstu_vss(vrsum_b2_i2, 4, pGIn+2*inDim+i+2) ;
      _ve_vstu_vss(vrsum_b3_i2, 4, pGIn+3*inDim+i+2) ;

    }
    i+=3 ;
    break ;
  default : break ;
  }
  for(; i<inDim; i+=4)
  {
    _ve_lvl(VLEN) ;
    __vr vrsum_b0_i0 = _ve_vbrdu_vs_f32(0.f) ;
    __vr vrsum_b1_i0 = _ve_vbrdu_vs_f32(0.f) ;
    __vr vrsum_b2_i0 = _ve_vbrdu_vs_f32(0.f) ;
    __vr vrsum_b3_i0 = _ve_vbrdu_vs_f32(0.f) ;
    __vr vrsum_b0_i1 = _ve_vbrdu_vs_f32(0.f) ;
    __vr vrsum_b1_i1 = _ve_vbrdu_vs_f32(0.f) ;
    __vr vrsum_b2_i1 = _ve_vbrdu_vs_f32(0.f) ;
    __vr vrsum_b3_i1 = _ve_vbrdu_vs_f32(0.f) ;
    __vr vrsum_b0_i2 = _ve_vbrdu_vs_f32(0.f) ;
    __vr vrsum_b1_i2 = _ve_vbrdu_vs_f32(0.f) ;
    __vr vrsum_b2_i2 = _ve_vbrdu_vs_f32(0.f) ;
    __vr vrsum_b3_i2 = _ve_vbrdu_vs_f32(0.f) ;
    __vr vrsum_b0_i3 = _ve_vbrdu_vs_f32(0.f) ;
    __vr vrsum_b1_i3 = _ve_vbrdu_vs_f32(0.f) ;
    __vr vrsum_b2_i3 = _ve_vbrdu_vs_f32(0.f) ;
    __vr vrsum_b3_i3 = _ve_vbrdu_vs_f32(0.f) ;

    for(int64_t o=0; o<outDim; o+=VLEN) {
      const int64_t vl = outDim-o < VLEN ? outDim - o : VLEN ;
      _ve_lvl(vl) ;
      __vr vrw_i0 = _ve_vldu_vss(4, pWeight+(i  )*outDim+o) ;
      __vr vrw_i1 = _ve_vldu_vss(4, pWeight+(i+1)*outDim+o) ;
      __vr vrw_i2 = _ve_vldu_vss(4, pWeight+(i+2)*outDim+o) ;
      __vr vrw_i3 = _ve_vldu_vss(4, pWeight+(i+3)*outDim+o) ;

      __vr vro_b0 = _ve_vldu_vss(4, pGOut+0*outDim+o) ;
      __vr vro_b1 = _ve_vldu_vss(4, pGOut+1*outDim+o) ;
      __vr vro_b2 = _ve_vldu_vss(4, pGOut+2*outDim+o) ;
      __vr vro_b3 = _ve_vldu_vss(4, pGOut+3*outDim+o) ;

      vrsum_b0_i0 = _ve_vfmads_vvvv(vrsum_b0_i0, vrw_i0, vro_b0) ;
      vrsum_b1_i0 = _ve_vfmads_vvvv(vrsum_b1_i0, vrw_i0, vro_b1) ;
      vrsum_b2_i0 = _ve_vfmads_vvvv(vrsum_b2_i0, vrw_i0, vro_b2) ;
      vrsum_b3_i0 = _ve_vfmads_vvvv(vrsum_b3_i0, vrw_i0, vro_b3) ;

      vrsum_b0_i1 = _ve_vfmads_vvvv(vrsum_b0_i1, vrw_i1, vro_b0) ;
      vrsum_b1_i1 = _ve_vfmads_vvvv(vrsum_b1_i1, vrw_i1, vro_b1) ;
      vrsum_b2_i1 = _ve_vfmads_vvvv(vrsum_b2_i1, vrw_i1, vro_b2) ;
      vrsum_b3_i1 = _ve_vfmads_vvvv(vrsum_b3_i1, vrw_i1, vro_b3) ;

      vrsum_b0_i2 = _ve_vfmads_vvvv(vrsum_b0_i2, vrw_i2, vro_b0) ;
      vrsum_b1_i2 = _ve_vfmads_vvvv(vrsum_b1_i2, vrw_i2, vro_b1) ;
      vrsum_b2_i2 = _ve_vfmads_vvvv(vrsum_b2_i2, vrw_i2, vro_b2) ;
      vrsum_b3_i2 = _ve_vfmads_vvvv(vrsum_b3_i2, vrw_i2, vro_b3) ;

      vrsum_b0_i3 = _ve_vfmads_vvvv(vrsum_b0_i3, vrw_i3, vro_b0) ;
      vrsum_b1_i3 = _ve_vfmads_vvvv(vrsum_b1_i3, vrw_i3, vro_b1) ;
      vrsum_b2_i3 = _ve_vfmads_vvvv(vrsum_b2_i3, vrw_i3, vro_b2) ;
      vrsum_b3_i3 = _ve_vfmads_vvvv(vrsum_b3_i3, vrw_i3, vro_b3) ;
    }

    _ve_lvl(VLEN) ;
    vrsum_b0_i0 = _ve_vfsums_vv(vrsum_b0_i0);
    vrsum_b1_i0 = _ve_vfsums_vv(vrsum_b1_i0);
    vrsum_b2_i0 = _ve_vfsums_vv(vrsum_b2_i0);
    vrsum_b3_i0 = _ve_vfsums_vv(vrsum_b3_i0);
    vrsum_b0_i1 = _ve_vfsums_vv(vrsum_b0_i1);
    vrsum_b1_i1 = _ve_vfsums_vv(vrsum_b1_i1);
    vrsum_b2_i1 = _ve_vfsums_vv(vrsum_b2_i1);
    vrsum_b3_i1 = _ve_vfsums_vv(vrsum_b3_i1);
    vrsum_b0_i2 = _ve_vfsums_vv(vrsum_b0_i2);
    vrsum_b1_i2 = _ve_vfsums_vv(vrsum_b1_i2);
    vrsum_b2_i2 = _ve_vfsums_vv(vrsum_b2_i2);
    vrsum_b3_i2 = _ve_vfsums_vv(vrsum_b3_i2);
    vrsum_b0_i3 = _ve_vfsums_vv(vrsum_b0_i3);
    vrsum_b1_i3 = _ve_vfsums_vv(vrsum_b1_i3);
    vrsum_b2_i3 = _ve_vfsums_vv(vrsum_b2_i3);
    vrsum_b3_i3 = _ve_vfsums_vv(vrsum_b3_i3);

    _ve_lvl(1) ;
    _ve_vstu_vss(vrsum_b0_i0, 4, pGIn+0*inDim+i) ;
    _ve_vstu_vss(vrsum_b1_i0, 4, pGIn+1*inDim+i) ;
    _ve_vstu_vss(vrsum_b2_i0, 4, pGIn+2*inDim+i) ;
    _ve_vstu_vss(vrsum_b3_i0, 4, pGIn+3*inDim+i) ;
    _ve_vstu_vss(vrsum_b0_i1, 4, pGIn+0*inDim+i+1) ;
    _ve_vstu_vss(vrsum_b1_i1, 4, pGIn+1*inDim+i+1) ;
    _ve_vstu_vss(vrsum_b2_i1, 4, pGIn+2*inDim+i+1) ;
    _ve_vstu_vss(vrsum_b3_i1, 4, pGIn+3*inDim+i+1) ;
    _ve_vstu_vss(vrsum_b0_i2, 4, pGIn+0*inDim+i+2) ;
    _ve_vstu_vss(vrsum_b1_i2, 4, pGIn+1*inDim+i+2) ;
    _ve_vstu_vss(vrsum_b2_i2, 4, pGIn+2*inDim+i+2) ;
    _ve_vstu_vss(vrsum_b3_i2, 4, pGIn+3*inDim+i+2) ;
    _ve_vstu_vss(vrsum_b0_i3, 4, pGIn+0*inDim+i+3) ;
    _ve_vstu_vss(vrsum_b1_i3, 4, pGIn+1*inDim+i+3) ;
    _ve_vstu_vss(vrsum_b2_i3, 4, pGIn+2*inDim+i+3) ;
    _ve_vstu_vss(vrsum_b3_i3, 4, pGIn+3*inDim+i+3) ;
  }
}

vednnError_t vednnLinearBackwardData_default(
    const uint64_t			inDim,
    const uint64_t			outDim,
    const uint64_t			nBatch,
    const void * restrict		pDataGradOut,
    const void * restrict		pDataWeight,
    void * restrict			pDataGradIn
)
{
  const float * restrict pGOut   = (const float * restrict) pDataGradOut;
  const float * restrict pWeight = (const float * restrict) pDataWeight;
  float * restrict const pGIn    = (float * restrict const) pDataGradIn;

  int64_t n=0;
  int64_t batchRemain = nBatch % 4 ;

  switch( batchRemain ) {
  case 1:
    b1(inDim, outDim,
       pGOut+n*outDim, pWeight, pGIn+n*inDim ) ;
    n+=1 ;
    break ;
  case 2:
    b2(inDim, outDim,
       pGOut+n*outDim, pWeight, pGIn+n*inDim ) ;
    n+=2 ;
    break ;
  case 3:
    b3(inDim, outDim,
       pGOut+n*outDim, pWeight, pGIn+n*inDim ) ;
    n+=3;
    break ;
  default : break ;
  }
  for(; n<nBatch; n+=4) {
    b4(inDim, outDim,
       pGOut+n*outDim, pWeight, pGIn+n*inDim ) ;
  }

  return VEDNN_SUCCESS ;
}


#if 0 // reference code
vednnError_t vednnLinearBackwardData_default(
    const uint64_t			inDim,
    const uint64_t			outDim,
    const uint64_t			nBatch,
    const void * restrict		pDataGradOut,
    const void * restrict		pDataWeight,
    void * restrict			pDataGradIn
)
{
  const float * restrict pGOut   = pDataGradOut;
  const float * restrict pWeight = pDataWeight;
  float * restrict const pGIn    = pDataGradIn;

  for(int64_t n=0; n<nBatch; n++) {
    for(int64_t i=0; i<inDim; i++ ) {
      float sum = 0.f ;
      for(int64_t o=0; o<outDim; o++) {
	sum += pWeight[i*outDim+o] * pGOut[n*outDim+o] ;
      }
      pGIn[n*inDim+i] = sum ;
    }
  }

  return VEDNN_SUCCESS ;
}
#endif


