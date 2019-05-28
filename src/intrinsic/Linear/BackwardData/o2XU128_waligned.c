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
  for(int i=0; i<inDim; i+=VLEN) {
    const int64_t vl = inDim - i < VLEN ? inDim - i : VLEN ;
    _ve_lvl(vl) ;

    __vr vrsum_b0 = _ve_vbrd_vs_i64(0UL) ;

    for(int64_t o=0; o<outDim; o+=2)
    {
      __vr vrw = _ve_vld_vss(4*outDim, pWeight+i*outDim+o) ;

      const uint64_t go0 = _ve_pack_f32p(pGOut+0*outDim+o+1, pGOut+0*outDim+o) ;

      vrsum_b0 = _ve_pvfmad_vvsv(vrsum_b0, go0, vrw) ;
    }

    _ve_vstu_vss(_ve_vfadds_vvv(vrsum_b0, _ve_vsll_vvs(vrsum_b0,32)), 4, pGIn+0*inDim+i) ;
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
  for(int i=0; i<inDim; i+=VLEN) {
    const int64_t vl = inDim - i < VLEN ? inDim - i : VLEN ;
    _ve_lvl(vl) ;

    __vr vrsum_b0 = _ve_vbrd_vs_i64(0UL) ;
    __vr vrsum_b1 = _ve_vbrd_vs_i64(0UL) ;

    for(int64_t o=0; o<outDim; o+=2)
    {
      __vr vrw = _ve_vld_vss(4*outDim, pWeight+i*outDim+o) ;

      const uint64_t go0 = _ve_pack_f32p(pGOut+0*outDim+o+1, pGOut+0*outDim+o) ;
      const uint64_t go1 = _ve_pack_f32p(pGOut+1*outDim+o+1, pGOut+1*outDim+o) ;

      vrsum_b0 = _ve_pvfmad_vvsv(vrsum_b0, go0, vrw) ;
      vrsum_b1 = _ve_pvfmad_vvsv(vrsum_b1, go1, vrw) ;
    }

    _ve_vstu_vss(_ve_vfadds_vvv(vrsum_b0, _ve_vsll_vvs(vrsum_b0,32)), 4, pGIn+0*inDim+i) ;
    _ve_vstu_vss(_ve_vfadds_vvv(vrsum_b1, _ve_vsll_vvs(vrsum_b1,32)), 4, pGIn+1*inDim+i) ;
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
  for(int i=0; i<inDim; i+=VLEN) {
    const int64_t vl = inDim - i < VLEN ? inDim - i : VLEN ;
    _ve_lvl(vl) ;

    __vr vrsum_b0 = _ve_vbrd_vs_i64(0UL) ;
    __vr vrsum_b1 = _ve_vbrd_vs_i64(0UL) ;
    __vr vrsum_b2 = _ve_vbrd_vs_i64(0UL) ;

    for(int64_t o=0; o<outDim; o+=2)
    {
      __vr vrw = _ve_vld_vss(4*outDim, pWeight+i*outDim+o) ;

      const uint64_t go0 = _ve_pack_f32p(pGOut+0*outDim+o+1, pGOut+0*outDim+o) ;
      const uint64_t go1 = _ve_pack_f32p(pGOut+1*outDim+o+1, pGOut+1*outDim+o) ;
      const uint64_t go2 = _ve_pack_f32p(pGOut+2*outDim+o+1, pGOut+2*outDim+o) ;

      vrsum_b0 = _ve_pvfmad_vvsv(vrsum_b0, go0, vrw) ;
      vrsum_b1 = _ve_pvfmad_vvsv(vrsum_b1, go1, vrw) ;
      vrsum_b2 = _ve_pvfmad_vvsv(vrsum_b2, go2, vrw) ;
    }

    _ve_vstu_vss(_ve_vfadds_vvv(vrsum_b0, _ve_vsll_vvs(vrsum_b0,32)), 4, pGIn+0*inDim+i) ;
    _ve_vstu_vss(_ve_vfadds_vvv(vrsum_b1, _ve_vsll_vvs(vrsum_b1,32)), 4, pGIn+1*inDim+i) ;
    _ve_vstu_vss(_ve_vfadds_vvv(vrsum_b2, _ve_vsll_vvs(vrsum_b2,32)), 4, pGIn+2*inDim+i) ;
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
  for(int i=0; i<inDim; i+=VLEN) {
    const int64_t vl = inDim - i < VLEN ? inDim - i : VLEN ;
    _ve_lvl(vl) ;

    __vr vrsum_b0 = _ve_vbrd_vs_i64(0UL) ;
    __vr vrsum_b1 = _ve_vbrd_vs_i64(0UL) ;
    __vr vrsum_b2 = _ve_vbrd_vs_i64(0UL) ;
    __vr vrsum_b3 = _ve_vbrd_vs_i64(0UL) ;

    for(int64_t o=0; o<outDim; o+=2)
    {
      __vr vrw = _ve_vld_vss(4*outDim, pWeight+i*outDim+o) ;

      const uint64_t go0 = _ve_pack_f32p(pGOut+0*outDim+o+1, pGOut+0*outDim+o) ;
      const uint64_t go1 = _ve_pack_f32p(pGOut+1*outDim+o+1, pGOut+1*outDim+o) ;
      const uint64_t go2 = _ve_pack_f32p(pGOut+2*outDim+o+1, pGOut+2*outDim+o) ;
      const uint64_t go3 = _ve_pack_f32p(pGOut+3*outDim+o+1, pGOut+3*outDim+o) ;

      vrsum_b0 = _ve_pvfmad_vvsv(vrsum_b0, go0, vrw) ;
      vrsum_b1 = _ve_pvfmad_vvsv(vrsum_b1, go1, vrw) ;
      vrsum_b2 = _ve_pvfmad_vvsv(vrsum_b2, go2, vrw) ;
      vrsum_b3 = _ve_pvfmad_vvsv(vrsum_b3, go3, vrw) ;
    }

    _ve_vstu_vss(_ve_vfadds_vvv(vrsum_b0, _ve_vsll_vvs(vrsum_b0,32)), 4, pGIn+0*inDim+i) ;
    _ve_vstu_vss(_ve_vfadds_vvv(vrsum_b1, _ve_vsll_vvs(vrsum_b1,32)), 4, pGIn+1*inDim+i) ;
    _ve_vstu_vss(_ve_vfadds_vvv(vrsum_b2, _ve_vsll_vvs(vrsum_b2,32)), 4, pGIn+2*inDim+i) ;
    _ve_vstu_vss(_ve_vfadds_vvv(vrsum_b3, _ve_vsll_vvs(vrsum_b3,32)), 4, pGIn+3*inDim+i) ;
  }
}

static inline void b5(
  const uint64_t	inDim,
  const uint64_t	outDim,
  const float * 	pGOut,
  const float * 	pWeight,
  float * 		pGIn
)
{
  for(int i=0; i<inDim; i+=VLEN) {
    const int64_t vl = inDim - i < VLEN ? inDim - i : VLEN ;
    _ve_lvl(vl) ;

    __vr vrsum_b0 = _ve_vbrd_vs_i64(0UL) ;
    __vr vrsum_b1 = _ve_vbrd_vs_i64(0UL) ;
    __vr vrsum_b2 = _ve_vbrd_vs_i64(0UL) ;
    __vr vrsum_b3 = _ve_vbrd_vs_i64(0UL) ;
    __vr vrsum_b4 = _ve_vbrd_vs_i64(0UL) ;

    for(int64_t o=0; o<outDim; o+=2)
    {
      __vr vrw = _ve_vld_vss(4*outDim, pWeight+i*outDim+o) ;

      const uint64_t go0 = _ve_pack_f32p(pGOut+0*outDim+o+1, pGOut+0*outDim+o) ;
      const uint64_t go1 = _ve_pack_f32p(pGOut+1*outDim+o+1, pGOut+1*outDim+o) ;
      const uint64_t go2 = _ve_pack_f32p(pGOut+2*outDim+o+1, pGOut+2*outDim+o) ;
      const uint64_t go3 = _ve_pack_f32p(pGOut+3*outDim+o+1, pGOut+3*outDim+o) ;
      const uint64_t go4 = _ve_pack_f32p(pGOut+4*outDim+o+1, pGOut+4*outDim+o) ;

      vrsum_b0 = _ve_pvfmad_vvsv(vrsum_b0, go0, vrw) ;
      vrsum_b1 = _ve_pvfmad_vvsv(vrsum_b1, go1, vrw) ;
      vrsum_b2 = _ve_pvfmad_vvsv(vrsum_b2, go2, vrw) ;
      vrsum_b3 = _ve_pvfmad_vvsv(vrsum_b3, go3, vrw) ;
      vrsum_b4 = _ve_pvfmad_vvsv(vrsum_b4, go4, vrw) ;
    }

    _ve_vstu_vss(_ve_vfadds_vvv(vrsum_b0, _ve_vsll_vvs(vrsum_b0,32)), 4, pGIn+0*inDim+i) ;
    _ve_vstu_vss(_ve_vfadds_vvv(vrsum_b1, _ve_vsll_vvs(vrsum_b1,32)), 4, pGIn+1*inDim+i) ;
    _ve_vstu_vss(_ve_vfadds_vvv(vrsum_b2, _ve_vsll_vvs(vrsum_b2,32)), 4, pGIn+2*inDim+i) ;
    _ve_vstu_vss(_ve_vfadds_vvv(vrsum_b3, _ve_vsll_vvs(vrsum_b3,32)), 4, pGIn+3*inDim+i) ;
    _ve_vstu_vss(_ve_vfadds_vvv(vrsum_b4, _ve_vsll_vvs(vrsum_b4,32)), 4, pGIn+4*inDim+i) ;
  }
}

static inline void b6(
  const uint64_t	inDim,
  const uint64_t	outDim,
  const float * 	pGOut,
  const float * 	pWeight,
  float * 		pGIn
)
{
  for(int i=0; i<inDim; i+=VLEN) {
    const int64_t vl = inDim - i < VLEN ? inDim - i : VLEN ;
    _ve_lvl(vl) ;

    __vr vrsum_b0 = _ve_vbrd_vs_i64(0UL) ;
    __vr vrsum_b1 = _ve_vbrd_vs_i64(0UL) ;
    __vr vrsum_b2 = _ve_vbrd_vs_i64(0UL) ;
    __vr vrsum_b3 = _ve_vbrd_vs_i64(0UL) ;
    __vr vrsum_b4 = _ve_vbrd_vs_i64(0UL) ;
    __vr vrsum_b5 = _ve_vbrd_vs_i64(0UL) ;

    for(int64_t o=0; o<outDim; o+=2)
    {
      __vr vrw = _ve_vld_vss(4*outDim, pWeight+i*outDim+o) ;

      const uint64_t go0 = _ve_pack_f32p(pGOut+0*outDim+o+1, pGOut+0*outDim+o) ;
      const uint64_t go1 = _ve_pack_f32p(pGOut+1*outDim+o+1, pGOut+1*outDim+o) ;
      const uint64_t go2 = _ve_pack_f32p(pGOut+2*outDim+o+1, pGOut+2*outDim+o) ;
      const uint64_t go3 = _ve_pack_f32p(pGOut+3*outDim+o+1, pGOut+3*outDim+o) ;
      const uint64_t go4 = _ve_pack_f32p(pGOut+4*outDim+o+1, pGOut+4*outDim+o) ;
      const uint64_t go5 = _ve_pack_f32p(pGOut+5*outDim+o+1, pGOut+5*outDim+o) ;

      vrsum_b0 = _ve_pvfmad_vvsv(vrsum_b0, go0, vrw) ;
      vrsum_b1 = _ve_pvfmad_vvsv(vrsum_b1, go1, vrw) ;
      vrsum_b2 = _ve_pvfmad_vvsv(vrsum_b2, go2, vrw) ;
      vrsum_b3 = _ve_pvfmad_vvsv(vrsum_b3, go3, vrw) ;
      vrsum_b4 = _ve_pvfmad_vvsv(vrsum_b4, go4, vrw) ;
      vrsum_b5 = _ve_pvfmad_vvsv(vrsum_b5, go5, vrw) ;
    }

    _ve_vstu_vss(_ve_vfadds_vvv(vrsum_b0, _ve_vsll_vvs(vrsum_b0,32)), 4, pGIn+0*inDim+i) ;
    _ve_vstu_vss(_ve_vfadds_vvv(vrsum_b1, _ve_vsll_vvs(vrsum_b1,32)), 4, pGIn+1*inDim+i) ;
    _ve_vstu_vss(_ve_vfadds_vvv(vrsum_b2, _ve_vsll_vvs(vrsum_b2,32)), 4, pGIn+2*inDim+i) ;
    _ve_vstu_vss(_ve_vfadds_vvv(vrsum_b3, _ve_vsll_vvs(vrsum_b3,32)), 4, pGIn+3*inDim+i) ;
    _ve_vstu_vss(_ve_vfadds_vvv(vrsum_b4, _ve_vsll_vvs(vrsum_b4,32)), 4, pGIn+4*inDim+i) ;
    _ve_vstu_vss(_ve_vfadds_vvv(vrsum_b5, _ve_vsll_vvs(vrsum_b5,32)), 4, pGIn+5*inDim+i) ;
  }
}

static inline void b7(
  const uint64_t	inDim,
  const uint64_t	outDim,
  const float * 	pGOut,
  const float * 	pWeight,
  float * 		pGIn
)
{
  for(int i=0; i<inDim; i+=VLEN) {
    const int64_t vl = inDim - i < VLEN ? inDim - i : VLEN ;
    _ve_lvl(vl) ;

    __vr vrsum_b0 = _ve_vbrd_vs_i64(0UL) ;
    __vr vrsum_b1 = _ve_vbrd_vs_i64(0UL) ;
    __vr vrsum_b2 = _ve_vbrd_vs_i64(0UL) ;
    __vr vrsum_b3 = _ve_vbrd_vs_i64(0UL) ;
    __vr vrsum_b4 = _ve_vbrd_vs_i64(0UL) ;
    __vr vrsum_b5 = _ve_vbrd_vs_i64(0UL) ;
    __vr vrsum_b6 = _ve_vbrd_vs_i64(0UL) ;

    for(int64_t o=0; o<outDim; o+=2)
    {
      __vr vrw = _ve_vld_vss(4*outDim, pWeight+i*outDim+o) ;

      const uint64_t go0 = _ve_pack_f32p(pGOut+0*outDim+o+1, pGOut+0*outDim+o) ;
      const uint64_t go1 = _ve_pack_f32p(pGOut+1*outDim+o+1, pGOut+1*outDim+o) ;
      const uint64_t go2 = _ve_pack_f32p(pGOut+2*outDim+o+1, pGOut+2*outDim+o) ;
      const uint64_t go3 = _ve_pack_f32p(pGOut+3*outDim+o+1, pGOut+3*outDim+o) ;
      const uint64_t go4 = _ve_pack_f32p(pGOut+4*outDim+o+1, pGOut+4*outDim+o) ;
      const uint64_t go5 = _ve_pack_f32p(pGOut+5*outDim+o+1, pGOut+5*outDim+o) ;
      const uint64_t go6 = _ve_pack_f32p(pGOut+6*outDim+o+1, pGOut+6*outDim+o) ;

      vrsum_b0 = _ve_pvfmad_vvsv(vrsum_b0, go0, vrw) ;
      vrsum_b1 = _ve_pvfmad_vvsv(vrsum_b1, go1, vrw) ;
      vrsum_b2 = _ve_pvfmad_vvsv(vrsum_b2, go2, vrw) ;
      vrsum_b3 = _ve_pvfmad_vvsv(vrsum_b3, go3, vrw) ;
      vrsum_b4 = _ve_pvfmad_vvsv(vrsum_b4, go4, vrw) ;
      vrsum_b5 = _ve_pvfmad_vvsv(vrsum_b5, go5, vrw) ;
      vrsum_b6 = _ve_pvfmad_vvsv(vrsum_b6, go6, vrw) ;
    }

    _ve_vstu_vss(_ve_vfadds_vvv(vrsum_b0, _ve_vsll_vvs(vrsum_b0,32)), 4, pGIn+0*inDim+i) ;
    _ve_vstu_vss(_ve_vfadds_vvv(vrsum_b1, _ve_vsll_vvs(vrsum_b1,32)), 4, pGIn+1*inDim+i) ;
    _ve_vstu_vss(_ve_vfadds_vvv(vrsum_b2, _ve_vsll_vvs(vrsum_b2,32)), 4, pGIn+2*inDim+i) ;
    _ve_vstu_vss(_ve_vfadds_vvv(vrsum_b3, _ve_vsll_vvs(vrsum_b3,32)), 4, pGIn+3*inDim+i) ;
    _ve_vstu_vss(_ve_vfadds_vvv(vrsum_b4, _ve_vsll_vvs(vrsum_b4,32)), 4, pGIn+4*inDim+i) ;
    _ve_vstu_vss(_ve_vfadds_vvv(vrsum_b5, _ve_vsll_vvs(vrsum_b5,32)), 4, pGIn+5*inDim+i) ;
    _ve_vstu_vss(_ve_vfadds_vvv(vrsum_b6, _ve_vsll_vvs(vrsum_b6,32)), 4, pGIn+6*inDim+i) ;
  }
}

static inline void b8(
  const uint64_t	inDim,
  const uint64_t	outDim,
  const float * 	pGOut,
  const float * 	pWeight,
  float * 		pGIn
)
{
  for(int i=0; i<inDim; i+=VLEN) {
    const int64_t vl = inDim - i < VLEN ? inDim - i : VLEN ;
    _ve_lvl(vl) ;

    __vr vrsum_b0 = _ve_vbrd_vs_i64(0UL) ;
    __vr vrsum_b1 = _ve_vbrd_vs_i64(0UL) ;
    __vr vrsum_b2 = _ve_vbrd_vs_i64(0UL) ;
    __vr vrsum_b3 = _ve_vbrd_vs_i64(0UL) ;
    __vr vrsum_b4 = _ve_vbrd_vs_i64(0UL) ;
    __vr vrsum_b5 = _ve_vbrd_vs_i64(0UL) ;
    __vr vrsum_b6 = _ve_vbrd_vs_i64(0UL) ;
    __vr vrsum_b7 = _ve_vbrd_vs_i64(0UL) ;

    for(int64_t o=0; o<outDim; o+=2)
    {
      __vr vrw = _ve_vld_vss(4*outDim, pWeight+i*outDim+o) ;

      const uint64_t go0 = _ve_pack_f32p(pGOut+0*outDim+o+1, pGOut+0*outDim+o) ;
      const uint64_t go1 = _ve_pack_f32p(pGOut+1*outDim+o+1, pGOut+1*outDim+o) ;
      const uint64_t go2 = _ve_pack_f32p(pGOut+2*outDim+o+1, pGOut+2*outDim+o) ;
      const uint64_t go3 = _ve_pack_f32p(pGOut+3*outDim+o+1, pGOut+3*outDim+o) ;
      const uint64_t go4 = _ve_pack_f32p(pGOut+4*outDim+o+1, pGOut+4*outDim+o) ;
      const uint64_t go5 = _ve_pack_f32p(pGOut+5*outDim+o+1, pGOut+5*outDim+o) ;
      const uint64_t go6 = _ve_pack_f32p(pGOut+6*outDim+o+1, pGOut+6*outDim+o) ;
      const uint64_t go7 = _ve_pack_f32p(pGOut+7*outDim+o+1, pGOut+7*outDim+o) ;

      vrsum_b0 = _ve_pvfmad_vvsv(vrsum_b0, go0, vrw) ;
      vrsum_b1 = _ve_pvfmad_vvsv(vrsum_b1, go1, vrw) ;
      vrsum_b2 = _ve_pvfmad_vvsv(vrsum_b2, go2, vrw) ;
      vrsum_b3 = _ve_pvfmad_vvsv(vrsum_b3, go3, vrw) ;
      vrsum_b4 = _ve_pvfmad_vvsv(vrsum_b4, go4, vrw) ;
      vrsum_b5 = _ve_pvfmad_vvsv(vrsum_b5, go5, vrw) ;
      vrsum_b6 = _ve_pvfmad_vvsv(vrsum_b6, go6, vrw) ;
      vrsum_b7 = _ve_pvfmad_vvsv(vrsum_b7, go7, vrw) ;
    }

    _ve_vstu_vss(_ve_vfadds_vvv(vrsum_b0, _ve_vsll_vvs(vrsum_b0,32)), 4, pGIn+0*inDim+i) ;
    _ve_vstu_vss(_ve_vfadds_vvv(vrsum_b1, _ve_vsll_vvs(vrsum_b1,32)), 4, pGIn+1*inDim+i) ;
    _ve_vstu_vss(_ve_vfadds_vvv(vrsum_b2, _ve_vsll_vvs(vrsum_b2,32)), 4, pGIn+2*inDim+i) ;
    _ve_vstu_vss(_ve_vfadds_vvv(vrsum_b3, _ve_vsll_vvs(vrsum_b3,32)), 4, pGIn+3*inDim+i) ;
    _ve_vstu_vss(_ve_vfadds_vvv(vrsum_b4, _ve_vsll_vvs(vrsum_b4,32)), 4, pGIn+4*inDim+i) ;
    _ve_vstu_vss(_ve_vfadds_vvv(vrsum_b5, _ve_vsll_vvs(vrsum_b5,32)), 4, pGIn+5*inDim+i) ;
    _ve_vstu_vss(_ve_vfadds_vvv(vrsum_b6, _ve_vsll_vvs(vrsum_b6,32)), 4, pGIn+6*inDim+i) ;
    _ve_vstu_vss(_ve_vfadds_vvv(vrsum_b7, _ve_vsll_vvs(vrsum_b7,32)), 4, pGIn+7*inDim+i) ;
  }
}

static inline void b16(
  const uint64_t	inDim,
  const uint64_t	outDim,
  const float * 	pGOut,
  const float * 	pWeight,
  float * 		pGIn
)
{
  for(int i=0; i<inDim; i+=VLEN) {
    const int64_t vl = inDim - i < VLEN ? inDim - i : VLEN ;
    _ve_lvl(vl) ;

    __vr vrsum_b0 = _ve_vbrd_vs_i64(0UL) ;
    __vr vrsum_b1 = _ve_vbrd_vs_i64(0UL) ;
    __vr vrsum_b2 = _ve_vbrd_vs_i64(0UL) ;
    __vr vrsum_b3 = _ve_vbrd_vs_i64(0UL) ;
    __vr vrsum_b4 = _ve_vbrd_vs_i64(0UL) ;
    __vr vrsum_b5 = _ve_vbrd_vs_i64(0UL) ;
    __vr vrsum_b6 = _ve_vbrd_vs_i64(0UL) ;
    __vr vrsum_b7 = _ve_vbrd_vs_i64(0UL) ;
    __vr vrsum_b8 = _ve_vbrd_vs_i64(0UL) ;
    __vr vrsum_b9 = _ve_vbrd_vs_i64(0UL) ;
    __vr vrsum_bA = _ve_vbrd_vs_i64(0UL) ;
    __vr vrsum_bB = _ve_vbrd_vs_i64(0UL) ;
    __vr vrsum_bC = _ve_vbrd_vs_i64(0UL) ;
    __vr vrsum_bD = _ve_vbrd_vs_i64(0UL) ;
    __vr vrsum_bE = _ve_vbrd_vs_i64(0UL) ;
    __vr vrsum_bF = _ve_vbrd_vs_i64(0UL) ;


    for(int64_t o=0; o<outDim; o+=2)
    {
      __vr vrw = _ve_vld_vss(4*outDim, pWeight+i*outDim+o) ;

      const uint64_t go0 = _ve_pack_f32p(pGOut+0*outDim+o+1, pGOut+0*outDim+o) ;
      const uint64_t go1 = _ve_pack_f32p(pGOut+1*outDim+o+1, pGOut+1*outDim+o) ;
      const uint64_t go2 = _ve_pack_f32p(pGOut+2*outDim+o+1, pGOut+2*outDim+o) ;
      const uint64_t go3 = _ve_pack_f32p(pGOut+3*outDim+o+1, pGOut+3*outDim+o) ;
      const uint64_t go4 = _ve_pack_f32p(pGOut+4*outDim+o+1, pGOut+4*outDim+o) ;
      const uint64_t go5 = _ve_pack_f32p(pGOut+5*outDim+o+1, pGOut+5*outDim+o) ;
      const uint64_t go6 = _ve_pack_f32p(pGOut+6*outDim+o+1, pGOut+6*outDim+o) ;
      const uint64_t go7 = _ve_pack_f32p(pGOut+7*outDim+o+1, pGOut+7*outDim+o) ;

      vrsum_b0 = _ve_pvfmad_vvsv(vrsum_b0, go0, vrw) ;
      vrsum_b1 = _ve_pvfmad_vvsv(vrsum_b1, go1, vrw) ;
      vrsum_b2 = _ve_pvfmad_vvsv(vrsum_b2, go2, vrw) ;
      vrsum_b3 = _ve_pvfmad_vvsv(vrsum_b3, go3, vrw) ;
      vrsum_b4 = _ve_pvfmad_vvsv(vrsum_b4, go4, vrw) ;
      vrsum_b5 = _ve_pvfmad_vvsv(vrsum_b5, go5, vrw) ;
      vrsum_b6 = _ve_pvfmad_vvsv(vrsum_b6, go6, vrw) ;
      vrsum_b7 = _ve_pvfmad_vvsv(vrsum_b7, go7, vrw) ;

      const uint64_t go8 = _ve_pack_f32p(pGOut+8*outDim+o+1, pGOut+8*outDim+o) ;
      const uint64_t go9 = _ve_pack_f32p(pGOut+9*outDim+o+1, pGOut+9*outDim+o) ;
      const uint64_t goA = _ve_pack_f32p(pGOut+10*outDim+o+1, pGOut+10*outDim+o) ;
      const uint64_t goB = _ve_pack_f32p(pGOut+11*outDim+o+1, pGOut+11*outDim+o) ;
      const uint64_t goC = _ve_pack_f32p(pGOut+12*outDim+o+1, pGOut+12*outDim+o) ;
      const uint64_t goD = _ve_pack_f32p(pGOut+13*outDim+o+1, pGOut+13*outDim+o) ;
      const uint64_t goE = _ve_pack_f32p(pGOut+14*outDim+o+1, pGOut+14*outDim+o) ;
      const uint64_t goF = _ve_pack_f32p(pGOut+15*outDim+o+1, pGOut+15*outDim+o) ;

      vrsum_b8 = _ve_pvfmad_vvsv(vrsum_b8, go8, vrw) ;
      vrsum_b9 = _ve_pvfmad_vvsv(vrsum_b9, go9, vrw) ;
      vrsum_bA = _ve_pvfmad_vvsv(vrsum_bA, goA, vrw) ;
      vrsum_bB = _ve_pvfmad_vvsv(vrsum_bB, goB, vrw) ;
      vrsum_bC = _ve_pvfmad_vvsv(vrsum_bC, goC, vrw) ;
      vrsum_bD = _ve_pvfmad_vvsv(vrsum_bD, goD, vrw) ;
      vrsum_bE = _ve_pvfmad_vvsv(vrsum_bE, goE, vrw) ;
      vrsum_bF = _ve_pvfmad_vvsv(vrsum_bF, goF, vrw) ;
    }

    _ve_vstu_vss(_ve_vfadds_vvv(vrsum_b0, _ve_vsll_vvs(vrsum_b0,32)), 4, pGIn+0*inDim+i) ;
    _ve_vstu_vss(_ve_vfadds_vvv(vrsum_b1, _ve_vsll_vvs(vrsum_b1,32)), 4, pGIn+1*inDim+i) ;
    _ve_vstu_vss(_ve_vfadds_vvv(vrsum_b2, _ve_vsll_vvs(vrsum_b2,32)), 4, pGIn+2*inDim+i) ;
    _ve_vstu_vss(_ve_vfadds_vvv(vrsum_b3, _ve_vsll_vvs(vrsum_b3,32)), 4, pGIn+3*inDim+i) ;
    _ve_vstu_vss(_ve_vfadds_vvv(vrsum_b4, _ve_vsll_vvs(vrsum_b4,32)), 4, pGIn+4*inDim+i) ;
    _ve_vstu_vss(_ve_vfadds_vvv(vrsum_b5, _ve_vsll_vvs(vrsum_b5,32)), 4, pGIn+5*inDim+i) ;
    _ve_vstu_vss(_ve_vfadds_vvv(vrsum_b6, _ve_vsll_vvs(vrsum_b6,32)), 4, pGIn+6*inDim+i) ;
    _ve_vstu_vss(_ve_vfadds_vvv(vrsum_b7, _ve_vsll_vvs(vrsum_b7,32)), 4, pGIn+7*inDim+i) ;
    _ve_vstu_vss(_ve_vfadds_vvv(vrsum_b8, _ve_vsll_vvs(vrsum_b8,32)), 4, pGIn+8*inDim+i) ;
    _ve_vstu_vss(_ve_vfadds_vvv(vrsum_b9, _ve_vsll_vvs(vrsum_b9,32)), 4, pGIn+9*inDim+i) ;
    _ve_vstu_vss(_ve_vfadds_vvv(vrsum_bA, _ve_vsll_vvs(vrsum_bA,32)), 4, pGIn+10*inDim+i) ;
    _ve_vstu_vss(_ve_vfadds_vvv(vrsum_bB, _ve_vsll_vvs(vrsum_bB,32)), 4, pGIn+11*inDim+i) ;
    _ve_vstu_vss(_ve_vfadds_vvv(vrsum_bC, _ve_vsll_vvs(vrsum_bC,32)), 4, pGIn+12*inDim+i) ;
    _ve_vstu_vss(_ve_vfadds_vvv(vrsum_bD, _ve_vsll_vvs(vrsum_bD,32)), 4, pGIn+13*inDim+i) ;
    _ve_vstu_vss(_ve_vfadds_vvv(vrsum_bE, _ve_vsll_vvs(vrsum_bE,32)), 4, pGIn+14*inDim+i) ;
    _ve_vstu_vss(_ve_vfadds_vvv(vrsum_bF, _ve_vsll_vvs(vrsum_bF,32)), 4, pGIn+15*inDim+i) ;
  }
}

vednnError_t vednnLinearBackwardData_o2XU128_waligned(
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
  int64_t batchRemain = nBatch & 0x7 ;

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
  case 4:
    b4(inDim, outDim,
       pGOut+n*outDim, pWeight, pGIn+n*inDim ) ;
    n+=4;
    break ;
  case 5:
    b5(inDim, outDim,
       pGOut+n*outDim, pWeight, pGIn+n*inDim ) ;
    n+=5;
    break ;
  case 6:
    b6(inDim, outDim,
       pGOut+n*outDim, pWeight, pGIn+n*inDim ) ;
    n+=6;
    break ;
  case 7:
    b7(inDim, outDim,
       pGOut+n*outDim, pWeight, pGIn+n*inDim ) ;
    n+=7;
    break ;
  default : break ;
  }
  if((nBatch>>3) & 0x1) {
    b8(inDim, outDim,
       pGOut+n*outDim, pWeight, pGIn+n*inDim ) ;
    n+=8 ;
  }
  for(; n<nBatch; n+=16) {
    b16(inDim, outDim,
        pGOut+n*outDim, pWeight, pGIn+n*inDim ) ;
  }

  return VEDNN_SUCCESS ;
}
