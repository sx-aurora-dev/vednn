#include <stdint.h>
#include <float.h>

#include <stdio.h>

#include "vednn.h"

#include "velintrin.h"
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
    __vr vrsum_b0 = _vel_vbrdl_vsl(0UL, vl) ;

    for(int64_t o=0; o<outDim; o+=2)
    {
      __vr vrw = _vel_vld_vssl(4*outDim, pWeight+i*outDim+o, vl) ;

      const uint64_t go0 = _vel_pack_f32p(pGOut+0*outDim+o+1, pGOut+0*outDim+o) ;

      vrsum_b0 = _vel_pvfmad_vvsvl(vrsum_b0, go0, vrw, vl) ;
    }

    vrsum_b0 = _vel_vfadds_vvvl(vrsum_b0, _vel_vsll_vvsl(vrsum_b0,32,vl),vl) ;
    _vel_vstu_vssl(vrsum_b0, 4, pGIn+0*inDim+i, vl) ;
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
    __vr vrsum_b0 = _vel_vbrdl_vsl(0UL, vl) ;
    __vr vrsum_b1 = _vel_vbrdl_vsl(0UL, vl) ;

    for(int64_t o=0; o<outDim; o+=2)
    {
      __vr vrw = _vel_vld_vssl(4*outDim, pWeight+i*outDim+o, vl) ;

      const uint64_t go0 = _vel_pack_f32p(pGOut+0*outDim+o+1, pGOut+0*outDim+o) ;
      const uint64_t go1 = _vel_pack_f32p(pGOut+1*outDim+o+1, pGOut+1*outDim+o) ;

      vrsum_b0 = _vel_pvfmad_vvsvl(vrsum_b0, go0, vrw, vl) ;
      vrsum_b1 = _vel_pvfmad_vvsvl(vrsum_b1, go1, vrw, vl) ;
    }

    vrsum_b0 = _vel_vfadds_vvvl(vrsum_b0, _vel_vsll_vvsl(vrsum_b0,32,vl),vl) ;
    _vel_vstu_vssl(vrsum_b0, 4, pGIn+0*inDim+i, vl) ;
    vrsum_b1 = _vel_vfadds_vvvl(vrsum_b1, _vel_vsll_vvsl(vrsum_b1,32,vl),vl) ;
    _vel_vstu_vssl(vrsum_b1, 4, pGIn+1*inDim+i, vl) ;
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
    __vr vrsum_b0 = _vel_vbrdl_vsl(0UL, vl) ;
    __vr vrsum_b1 = _vel_vbrdl_vsl(0UL, vl) ;
    __vr vrsum_b2 = _vel_vbrdl_vsl(0UL, vl) ;

    for(int64_t o=0; o<outDim; o+=2)
    {
      __vr vrw = _vel_vld_vssl(4*outDim, pWeight+i*outDim+o, vl) ;

      const uint64_t go0 = _vel_pack_f32p(pGOut+0*outDim+o+1, pGOut+0*outDim+o) ;
      const uint64_t go1 = _vel_pack_f32p(pGOut+1*outDim+o+1, pGOut+1*outDim+o) ;
      const uint64_t go2 = _vel_pack_f32p(pGOut+2*outDim+o+1, pGOut+2*outDim+o) ;

      vrsum_b0 = _vel_pvfmad_vvsvl(vrsum_b0, go0, vrw, vl) ;
      vrsum_b1 = _vel_pvfmad_vvsvl(vrsum_b1, go1, vrw, vl) ;
      vrsum_b2 = _vel_pvfmad_vvsvl(vrsum_b2, go2, vrw, vl) ;
    }

    vrsum_b0 = _vel_vfadds_vvvl(vrsum_b0, _vel_vsll_vvsl(vrsum_b0,32,vl),vl) ;
    _vel_vstu_vssl(vrsum_b0, 4, pGIn+0*inDim+i, vl) ;
    vrsum_b1 = _vel_vfadds_vvvl(vrsum_b1, _vel_vsll_vvsl(vrsum_b1,32,vl),vl) ;
    _vel_vstu_vssl(vrsum_b1, 4, pGIn+1*inDim+i, vl) ;
    vrsum_b2 = _vel_vfadds_vvvl(vrsum_b2, _vel_vsll_vvsl(vrsum_b2,32,vl),vl) ;
    _vel_vstu_vssl(vrsum_b2, 4, pGIn+2*inDim+i, vl) ;
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
    __vr vrsum_b0 = _vel_vbrdl_vsl(0UL, vl) ;
    __vr vrsum_b1 = _vel_vbrdl_vsl(0UL, vl) ;
    __vr vrsum_b2 = _vel_vbrdl_vsl(0UL, vl) ;
    __vr vrsum_b3 = _vel_vbrdl_vsl(0UL, vl) ;

    for(int64_t o=0; o<outDim; o+=2)
    {
      __vr vrw = _vel_vld_vssl(4*outDim, pWeight+i*outDim+o, vl) ;

      const uint64_t go0 = _vel_pack_f32p(pGOut+0*outDim+o+1, pGOut+0*outDim+o) ;
      const uint64_t go1 = _vel_pack_f32p(pGOut+1*outDim+o+1, pGOut+1*outDim+o) ;
      const uint64_t go2 = _vel_pack_f32p(pGOut+2*outDim+o+1, pGOut+2*outDim+o) ;
      const uint64_t go3 = _vel_pack_f32p(pGOut+3*outDim+o+1, pGOut+3*outDim+o) ;

      vrsum_b0 = _vel_pvfmad_vvsvl(vrsum_b0, go0, vrw, vl) ;
      vrsum_b1 = _vel_pvfmad_vvsvl(vrsum_b1, go1, vrw, vl) ;
      vrsum_b2 = _vel_pvfmad_vvsvl(vrsum_b2, go2, vrw, vl) ;
      vrsum_b3 = _vel_pvfmad_vvsvl(vrsum_b3, go3, vrw, vl) ;
    }

    vrsum_b0 = _vel_vfadds_vvvl(vrsum_b0, _vel_vsll_vvsl(vrsum_b0,32,vl),vl) ;
    _vel_vstu_vssl(vrsum_b0, 4, pGIn+0*inDim+i, vl) ;
    vrsum_b1 = _vel_vfadds_vvvl(vrsum_b1, _vel_vsll_vvsl(vrsum_b1,32,vl),vl) ;
    _vel_vstu_vssl(vrsum_b1, 4, pGIn+1*inDim+i, vl) ;
    vrsum_b2 = _vel_vfadds_vvvl(vrsum_b2, _vel_vsll_vvsl(vrsum_b2,32,vl),vl) ;
    _vel_vstu_vssl(vrsum_b2, 4, pGIn+2*inDim+i, vl) ;
    vrsum_b3 = _vel_vfadds_vvvl(vrsum_b3, _vel_vsll_vvsl(vrsum_b3,32,vl),vl) ;
    _vel_vstu_vssl(vrsum_b3, 4, pGIn+3*inDim+i, vl) ;
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
    __vr vrsum_b0 = _vel_vbrdl_vsl(0UL, vl) ;
    __vr vrsum_b1 = _vel_vbrdl_vsl(0UL, vl) ;
    __vr vrsum_b2 = _vel_vbrdl_vsl(0UL, vl) ;
    __vr vrsum_b3 = _vel_vbrdl_vsl(0UL, vl) ;
    __vr vrsum_b4 = _vel_vbrdl_vsl(0UL, vl) ;

    for(int64_t o=0; o<outDim; o+=2)
    {
      __vr vrw = _vel_vld_vssl(4*outDim, pWeight+i*outDim+o, vl) ;

      const uint64_t go0 = _vel_pack_f32p(pGOut+0*outDim+o+1, pGOut+0*outDim+o) ;
      const uint64_t go1 = _vel_pack_f32p(pGOut+1*outDim+o+1, pGOut+1*outDim+o) ;
      const uint64_t go2 = _vel_pack_f32p(pGOut+2*outDim+o+1, pGOut+2*outDim+o) ;
      const uint64_t go3 = _vel_pack_f32p(pGOut+3*outDim+o+1, pGOut+3*outDim+o) ;
      const uint64_t go4 = _vel_pack_f32p(pGOut+4*outDim+o+1, pGOut+4*outDim+o) ;

      vrsum_b0 = _vel_pvfmad_vvsvl(vrsum_b0, go0, vrw, vl) ;
      vrsum_b1 = _vel_pvfmad_vvsvl(vrsum_b1, go1, vrw, vl) ;
      vrsum_b2 = _vel_pvfmad_vvsvl(vrsum_b2, go2, vrw, vl) ;
      vrsum_b3 = _vel_pvfmad_vvsvl(vrsum_b3, go3, vrw, vl) ;
      vrsum_b4 = _vel_pvfmad_vvsvl(vrsum_b4, go4, vrw, vl) ;
    }

    vrsum_b0 = _vel_vfadds_vvvl(vrsum_b0, _vel_vsll_vvsl(vrsum_b0,32,vl),vl) ;
    _vel_vstu_vssl(vrsum_b0, 4, pGIn+0*inDim+i, vl) ;
    vrsum_b1 = _vel_vfadds_vvvl(vrsum_b1, _vel_vsll_vvsl(vrsum_b1,32,vl),vl) ;
    _vel_vstu_vssl(vrsum_b1, 4, pGIn+1*inDim+i, vl) ;
    vrsum_b2 = _vel_vfadds_vvvl(vrsum_b2, _vel_vsll_vvsl(vrsum_b2,32,vl),vl) ;
    _vel_vstu_vssl(vrsum_b2, 4, pGIn+2*inDim+i, vl) ;
    vrsum_b3 = _vel_vfadds_vvvl(vrsum_b3, _vel_vsll_vvsl(vrsum_b3,32,vl),vl) ;
    _vel_vstu_vssl(vrsum_b3, 4, pGIn+3*inDim+i, vl) ;
    vrsum_b4 = _vel_vfadds_vvvl(vrsum_b4, _vel_vsll_vvsl(vrsum_b4,32,vl),vl) ;
    _vel_vstu_vssl(vrsum_b4, 4, pGIn+4*inDim+i, vl) ;
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
    __vr vrsum_b0 = _vel_vbrdl_vsl(0UL, vl) ;
    __vr vrsum_b1 = _vel_vbrdl_vsl(0UL, vl) ;
    __vr vrsum_b2 = _vel_vbrdl_vsl(0UL, vl) ;
    __vr vrsum_b3 = _vel_vbrdl_vsl(0UL, vl) ;
    __vr vrsum_b4 = _vel_vbrdl_vsl(0UL, vl) ;
    __vr vrsum_b5 = _vel_vbrdl_vsl(0UL, vl) ;

    for(int64_t o=0; o<outDim; o+=2)
    {
      __vr vrw = _vel_vld_vssl(4*outDim, pWeight+i*outDim+o, vl) ;

      const uint64_t go0 = _vel_pack_f32p(pGOut+0*outDim+o+1, pGOut+0*outDim+o) ;
      const uint64_t go1 = _vel_pack_f32p(pGOut+1*outDim+o+1, pGOut+1*outDim+o) ;
      const uint64_t go2 = _vel_pack_f32p(pGOut+2*outDim+o+1, pGOut+2*outDim+o) ;
      const uint64_t go3 = _vel_pack_f32p(pGOut+3*outDim+o+1, pGOut+3*outDim+o) ;
      const uint64_t go4 = _vel_pack_f32p(pGOut+4*outDim+o+1, pGOut+4*outDim+o) ;
      const uint64_t go5 = _vel_pack_f32p(pGOut+5*outDim+o+1, pGOut+5*outDim+o) ;

      vrsum_b0 = _vel_pvfmad_vvsvl(vrsum_b0, go0, vrw, vl) ;
      vrsum_b1 = _vel_pvfmad_vvsvl(vrsum_b1, go1, vrw, vl) ;
      vrsum_b2 = _vel_pvfmad_vvsvl(vrsum_b2, go2, vrw, vl) ;
      vrsum_b3 = _vel_pvfmad_vvsvl(vrsum_b3, go3, vrw, vl) ;
      vrsum_b4 = _vel_pvfmad_vvsvl(vrsum_b4, go4, vrw, vl) ;
      vrsum_b5 = _vel_pvfmad_vvsvl(vrsum_b5, go5, vrw, vl) ;
    }

    vrsum_b0 = _vel_vfadds_vvvl(vrsum_b0, _vel_vsll_vvsl(vrsum_b0,32,vl),vl) ;
    _vel_vstu_vssl(vrsum_b0, 4, pGIn+0*inDim+i, vl) ;
    vrsum_b1 = _vel_vfadds_vvvl(vrsum_b1, _vel_vsll_vvsl(vrsum_b1,32,vl),vl) ;
    _vel_vstu_vssl(vrsum_b1, 4, pGIn+1*inDim+i, vl) ;
    vrsum_b2 = _vel_vfadds_vvvl(vrsum_b2, _vel_vsll_vvsl(vrsum_b2,32,vl),vl) ;
    _vel_vstu_vssl(vrsum_b2, 4, pGIn+2*inDim+i, vl) ;
    vrsum_b3 = _vel_vfadds_vvvl(vrsum_b3, _vel_vsll_vvsl(vrsum_b3,32,vl),vl) ;
    _vel_vstu_vssl(vrsum_b3, 4, pGIn+3*inDim+i, vl) ;
    vrsum_b4 = _vel_vfadds_vvvl(vrsum_b4, _vel_vsll_vvsl(vrsum_b4,32,vl),vl) ;
    _vel_vstu_vssl(vrsum_b4, 4, pGIn+4*inDim+i, vl) ;
    vrsum_b5 = _vel_vfadds_vvvl(vrsum_b5, _vel_vsll_vvsl(vrsum_b5,32,vl),vl) ;
    _vel_vstu_vssl(vrsum_b5, 4, pGIn+5*inDim+i, vl) ;
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
    __vr vrsum_b0 = _vel_vbrdl_vsl(0UL, vl) ;
    __vr vrsum_b1 = _vel_vbrdl_vsl(0UL, vl) ;
    __vr vrsum_b2 = _vel_vbrdl_vsl(0UL, vl) ;
    __vr vrsum_b3 = _vel_vbrdl_vsl(0UL, vl) ;
    __vr vrsum_b4 = _vel_vbrdl_vsl(0UL, vl) ;
    __vr vrsum_b5 = _vel_vbrdl_vsl(0UL, vl) ;
    __vr vrsum_b6 = _vel_vbrdl_vsl(0UL, vl) ;

    for(int64_t o=0; o<outDim; o+=2)
    {
      __vr vrw = _vel_vld_vssl(4*outDim, pWeight+i*outDim+o, vl) ;

      const uint64_t go0 = _vel_pack_f32p(pGOut+0*outDim+o+1, pGOut+0*outDim+o) ;
      const uint64_t go1 = _vel_pack_f32p(pGOut+1*outDim+o+1, pGOut+1*outDim+o) ;
      const uint64_t go2 = _vel_pack_f32p(pGOut+2*outDim+o+1, pGOut+2*outDim+o) ;
      const uint64_t go3 = _vel_pack_f32p(pGOut+3*outDim+o+1, pGOut+3*outDim+o) ;
      const uint64_t go4 = _vel_pack_f32p(pGOut+4*outDim+o+1, pGOut+4*outDim+o) ;
      const uint64_t go5 = _vel_pack_f32p(pGOut+5*outDim+o+1, pGOut+5*outDim+o) ;
      const uint64_t go6 = _vel_pack_f32p(pGOut+6*outDim+o+1, pGOut+6*outDim+o) ;

      vrsum_b0 = _vel_pvfmad_vvsvl(vrsum_b0, go0, vrw, vl) ;
      vrsum_b1 = _vel_pvfmad_vvsvl(vrsum_b1, go1, vrw, vl) ;
      vrsum_b2 = _vel_pvfmad_vvsvl(vrsum_b2, go2, vrw, vl) ;
      vrsum_b3 = _vel_pvfmad_vvsvl(vrsum_b3, go3, vrw, vl) ;
      vrsum_b4 = _vel_pvfmad_vvsvl(vrsum_b4, go4, vrw, vl) ;
      vrsum_b5 = _vel_pvfmad_vvsvl(vrsum_b5, go5, vrw, vl) ;
      vrsum_b6 = _vel_pvfmad_vvsvl(vrsum_b6, go6, vrw, vl) ;
    }

    vrsum_b0 = _vel_vfadds_vvvl(vrsum_b0, _vel_vsll_vvsl(vrsum_b0,32,vl),vl) ;
    _vel_vstu_vssl(vrsum_b0, 4, pGIn+0*inDim+i, vl) ;
    vrsum_b1 = _vel_vfadds_vvvl(vrsum_b1, _vel_vsll_vvsl(vrsum_b1,32,vl),vl) ;
    _vel_vstu_vssl(vrsum_b1, 4, pGIn+1*inDim+i, vl) ;
    vrsum_b2 = _vel_vfadds_vvvl(vrsum_b2, _vel_vsll_vvsl(vrsum_b2,32,vl),vl) ;
    _vel_vstu_vssl(vrsum_b2, 4, pGIn+2*inDim+i, vl) ;
    vrsum_b3 = _vel_vfadds_vvvl(vrsum_b3, _vel_vsll_vvsl(vrsum_b3,32,vl),vl) ;
    _vel_vstu_vssl(vrsum_b3, 4, pGIn+3*inDim+i, vl) ;
    vrsum_b4 = _vel_vfadds_vvvl(vrsum_b4, _vel_vsll_vvsl(vrsum_b4,32,vl),vl) ;
    _vel_vstu_vssl(vrsum_b4, 4, pGIn+4*inDim+i, vl) ;
    vrsum_b5 = _vel_vfadds_vvvl(vrsum_b5, _vel_vsll_vvsl(vrsum_b5,32,vl),vl) ;
    _vel_vstu_vssl(vrsum_b5, 4, pGIn+5*inDim+i, vl) ;
    vrsum_b6 = _vel_vfadds_vvvl(vrsum_b6, _vel_vsll_vvsl(vrsum_b6,32,vl),vl) ;
    _vel_vstu_vssl(vrsum_b6, 4, pGIn+6*inDim+i, vl) ;
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
    __vr vrsum_b0 = _vel_vbrdl_vsl(0UL, vl) ;
    __vr vrsum_b1 = _vel_vbrdl_vsl(0UL, vl) ;
    __vr vrsum_b2 = _vel_vbrdl_vsl(0UL, vl) ;
    __vr vrsum_b3 = _vel_vbrdl_vsl(0UL, vl) ;
    __vr vrsum_b4 = _vel_vbrdl_vsl(0UL, vl) ;
    __vr vrsum_b5 = _vel_vbrdl_vsl(0UL, vl) ;
    __vr vrsum_b6 = _vel_vbrdl_vsl(0UL, vl) ;
    __vr vrsum_b7 = _vel_vbrdl_vsl(0UL, vl) ;

    for(int64_t o=0; o<outDim; o+=2)
    {
      __vr vrw = _vel_vld_vssl(4*outDim, pWeight+i*outDim+o, vl) ;

      const uint64_t go0 = _vel_pack_f32p(pGOut+0*outDim+o+1, pGOut+0*outDim+o) ;
      const uint64_t go1 = _vel_pack_f32p(pGOut+1*outDim+o+1, pGOut+1*outDim+o) ;
      const uint64_t go2 = _vel_pack_f32p(pGOut+2*outDim+o+1, pGOut+2*outDim+o) ;
      const uint64_t go3 = _vel_pack_f32p(pGOut+3*outDim+o+1, pGOut+3*outDim+o) ;
      const uint64_t go4 = _vel_pack_f32p(pGOut+4*outDim+o+1, pGOut+4*outDim+o) ;
      const uint64_t go5 = _vel_pack_f32p(pGOut+5*outDim+o+1, pGOut+5*outDim+o) ;
      const uint64_t go6 = _vel_pack_f32p(pGOut+6*outDim+o+1, pGOut+6*outDim+o) ;
      const uint64_t go7 = _vel_pack_f32p(pGOut+7*outDim+o+1, pGOut+7*outDim+o) ;

      vrsum_b0 = _vel_pvfmad_vvsvl(vrsum_b0, go0, vrw, vl) ;
      vrsum_b1 = _vel_pvfmad_vvsvl(vrsum_b1, go1, vrw, vl) ;
      vrsum_b2 = _vel_pvfmad_vvsvl(vrsum_b2, go2, vrw, vl) ;
      vrsum_b3 = _vel_pvfmad_vvsvl(vrsum_b3, go3, vrw, vl) ;
      vrsum_b4 = _vel_pvfmad_vvsvl(vrsum_b4, go4, vrw, vl) ;
      vrsum_b5 = _vel_pvfmad_vvsvl(vrsum_b5, go5, vrw, vl) ;
      vrsum_b6 = _vel_pvfmad_vvsvl(vrsum_b6, go6, vrw, vl) ;
      vrsum_b7 = _vel_pvfmad_vvsvl(vrsum_b7, go7, vrw, vl) ;
    }

    vrsum_b0 = _vel_vfadds_vvvl(vrsum_b0, _vel_vsll_vvsl(vrsum_b0,32,vl),vl) ;
    _vel_vstu_vssl(vrsum_b0, 4, pGIn+0*inDim+i, vl) ;
    vrsum_b1 = _vel_vfadds_vvvl(vrsum_b1, _vel_vsll_vvsl(vrsum_b1,32,vl),vl) ;
    _vel_vstu_vssl(vrsum_b1, 4, pGIn+1*inDim+i, vl) ;
    vrsum_b2 = _vel_vfadds_vvvl(vrsum_b2, _vel_vsll_vvsl(vrsum_b2,32,vl),vl) ;
    _vel_vstu_vssl(vrsum_b2, 4, pGIn+2*inDim+i, vl) ;
    vrsum_b3 = _vel_vfadds_vvvl(vrsum_b3, _vel_vsll_vvsl(vrsum_b3,32,vl),vl) ;
    _vel_vstu_vssl(vrsum_b3, 4, pGIn+3*inDim+i, vl) ;
    vrsum_b4 = _vel_vfadds_vvvl(vrsum_b4, _vel_vsll_vvsl(vrsum_b4,32,vl),vl) ;
    _vel_vstu_vssl(vrsum_b4, 4, pGIn+4*inDim+i, vl) ;
    vrsum_b5 = _vel_vfadds_vvvl(vrsum_b5, _vel_vsll_vvsl(vrsum_b5,32,vl),vl) ;
    _vel_vstu_vssl(vrsum_b5, 4, pGIn+5*inDim+i, vl) ;
    vrsum_b6 = _vel_vfadds_vvvl(vrsum_b6, _vel_vsll_vvsl(vrsum_b6,32,vl),vl) ;
    _vel_vstu_vssl(vrsum_b6, 4, pGIn+6*inDim+i, vl) ;
    vrsum_b7 = _vel_vfadds_vvvl(vrsum_b7, _vel_vsll_vvsl(vrsum_b7,32,vl),vl) ;
    _vel_vstu_vssl(vrsum_b7, 4, pGIn+7*inDim+i, vl) ;
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
    __vr vrsum_b0 = _vel_vbrdl_vsl(0UL, vl) ;
    __vr vrsum_b1 = _vel_vbrdl_vsl(0UL, vl) ;
    __vr vrsum_b2 = _vel_vbrdl_vsl(0UL, vl) ;
    __vr vrsum_b3 = _vel_vbrdl_vsl(0UL, vl) ;
    __vr vrsum_b4 = _vel_vbrdl_vsl(0UL, vl) ;
    __vr vrsum_b5 = _vel_vbrdl_vsl(0UL, vl) ;
    __vr vrsum_b6 = _vel_vbrdl_vsl(0UL, vl) ;
    __vr vrsum_b7 = _vel_vbrdl_vsl(0UL, vl) ;
    __vr vrsum_b8 = _vel_vbrdl_vsl(0UL, vl) ;
    __vr vrsum_b9 = _vel_vbrdl_vsl(0UL, vl) ;
    __vr vrsum_bA = _vel_vbrdl_vsl(0UL, vl) ;
    __vr vrsum_bB = _vel_vbrdl_vsl(0UL, vl) ;
    __vr vrsum_bC = _vel_vbrdl_vsl(0UL, vl) ;
    __vr vrsum_bD = _vel_vbrdl_vsl(0UL, vl) ;
    __vr vrsum_bE = _vel_vbrdl_vsl(0UL, vl) ;
    __vr vrsum_bF = _vel_vbrdl_vsl(0UL, vl) ;


    for(int64_t o=0; o<outDim; o+=2)
    {
      __vr vrw = _vel_vld_vssl(4*outDim, pWeight+i*outDim+o, vl) ;

      const uint64_t go0 = _vel_pack_f32p(pGOut+0*outDim+o+1, pGOut+0*outDim+o) ;
      const uint64_t go1 = _vel_pack_f32p(pGOut+1*outDim+o+1, pGOut+1*outDim+o) ;
      const uint64_t go2 = _vel_pack_f32p(pGOut+2*outDim+o+1, pGOut+2*outDim+o) ;
      const uint64_t go3 = _vel_pack_f32p(pGOut+3*outDim+o+1, pGOut+3*outDim+o) ;
      const uint64_t go4 = _vel_pack_f32p(pGOut+4*outDim+o+1, pGOut+4*outDim+o) ;
      const uint64_t go5 = _vel_pack_f32p(pGOut+5*outDim+o+1, pGOut+5*outDim+o) ;
      const uint64_t go6 = _vel_pack_f32p(pGOut+6*outDim+o+1, pGOut+6*outDim+o) ;
      const uint64_t go7 = _vel_pack_f32p(pGOut+7*outDim+o+1, pGOut+7*outDim+o) ;

      vrsum_b0 = _vel_pvfmad_vvsvl(vrsum_b0, go0, vrw, vl) ;
      vrsum_b1 = _vel_pvfmad_vvsvl(vrsum_b1, go1, vrw, vl) ;
      vrsum_b2 = _vel_pvfmad_vvsvl(vrsum_b2, go2, vrw, vl) ;
      vrsum_b3 = _vel_pvfmad_vvsvl(vrsum_b3, go3, vrw, vl) ;
      vrsum_b4 = _vel_pvfmad_vvsvl(vrsum_b4, go4, vrw, vl) ;
      vrsum_b5 = _vel_pvfmad_vvsvl(vrsum_b5, go5, vrw, vl) ;
      vrsum_b6 = _vel_pvfmad_vvsvl(vrsum_b6, go6, vrw, vl) ;
      vrsum_b7 = _vel_pvfmad_vvsvl(vrsum_b7, go7, vrw, vl) ;

      const uint64_t go8 = _vel_pack_f32p(pGOut+8*outDim+o+1, pGOut+8*outDim+o) ;
      const uint64_t go9 = _vel_pack_f32p(pGOut+9*outDim+o+1, pGOut+9*outDim+o) ;
      const uint64_t goA = _vel_pack_f32p(pGOut+10*outDim+o+1, pGOut+10*outDim+o) ;
      const uint64_t goB = _vel_pack_f32p(pGOut+11*outDim+o+1, pGOut+11*outDim+o) ;
      const uint64_t goC = _vel_pack_f32p(pGOut+12*outDim+o+1, pGOut+12*outDim+o) ;
      const uint64_t goD = _vel_pack_f32p(pGOut+13*outDim+o+1, pGOut+13*outDim+o) ;
      const uint64_t goE = _vel_pack_f32p(pGOut+14*outDim+o+1, pGOut+14*outDim+o) ;
      const uint64_t goF = _vel_pack_f32p(pGOut+15*outDim+o+1, pGOut+15*outDim+o) ;

      vrsum_b8 = _vel_pvfmad_vvsvl(vrsum_b8, go8, vrw, vl) ;
      vrsum_b9 = _vel_pvfmad_vvsvl(vrsum_b9, go9, vrw, vl) ;
      vrsum_bA = _vel_pvfmad_vvsvl(vrsum_bA, goA, vrw, vl) ;
      vrsum_bB = _vel_pvfmad_vvsvl(vrsum_bB, goB, vrw, vl) ;
      vrsum_bC = _vel_pvfmad_vvsvl(vrsum_bC, goC, vrw, vl) ;
      vrsum_bD = _vel_pvfmad_vvsvl(vrsum_bD, goD, vrw, vl) ;
      vrsum_bE = _vel_pvfmad_vvsvl(vrsum_bE, goE, vrw, vl) ;
      vrsum_bF = _vel_pvfmad_vvsvl(vrsum_bF, goF, vrw, vl) ;
    }

    vrsum_b0 = _vel_vfadds_vvvl(vrsum_b0, _vel_vsll_vvsl(vrsum_b0,32,vl),vl) ;
    _vel_vstu_vssl(vrsum_b0, 4, pGIn+0*inDim+i, vl) ;
    vrsum_b1 = _vel_vfadds_vvvl(vrsum_b1, _vel_vsll_vvsl(vrsum_b1,32,vl),vl) ;
    _vel_vstu_vssl(vrsum_b1, 4, pGIn+1*inDim+i, vl) ;
    vrsum_b2 = _vel_vfadds_vvvl(vrsum_b2, _vel_vsll_vvsl(vrsum_b2,32,vl),vl) ;
    _vel_vstu_vssl(vrsum_b2, 4, pGIn+2*inDim+i, vl) ;
    vrsum_b3 = _vel_vfadds_vvvl(vrsum_b3, _vel_vsll_vvsl(vrsum_b3,32,vl),vl) ;
    _vel_vstu_vssl(vrsum_b3, 4, pGIn+3*inDim+i, vl) ;
    vrsum_b4 = _vel_vfadds_vvvl(vrsum_b4, _vel_vsll_vvsl(vrsum_b4,32,vl),vl) ;
    _vel_vstu_vssl(vrsum_b4, 4, pGIn+4*inDim+i, vl) ;
    vrsum_b5 = _vel_vfadds_vvvl(vrsum_b5, _vel_vsll_vvsl(vrsum_b5,32,vl),vl) ;
    _vel_vstu_vssl(vrsum_b5, 4, pGIn+5*inDim+i, vl) ;
    vrsum_b6 = _vel_vfadds_vvvl(vrsum_b6, _vel_vsll_vvsl(vrsum_b6,32,vl),vl) ;
    _vel_vstu_vssl(vrsum_b6, 4, pGIn+6*inDim+i, vl) ;
    vrsum_b7 = _vel_vfadds_vvvl(vrsum_b7, _vel_vsll_vvsl(vrsum_b7,32,vl),vl) ;
    _vel_vstu_vssl(vrsum_b7, 4, pGIn+7*inDim+i, vl) ;
    vrsum_b8 = _vel_vfadds_vvvl(vrsum_b8, _vel_vsll_vvsl(vrsum_b8,32,vl),vl) ;
    _vel_vstu_vssl(vrsum_b8, 4, pGIn+8*inDim+i, vl) ;
    vrsum_b9 = _vel_vfadds_vvvl(vrsum_b9, _vel_vsll_vvsl(vrsum_b9,32,vl),vl) ;
    _vel_vstu_vssl(vrsum_b9, 4, pGIn+9*inDim+i, vl) ;
    vrsum_bA = _vel_vfadds_vvvl(vrsum_bA, _vel_vsll_vvsl(vrsum_bA,32,vl),vl) ;
    _vel_vstu_vssl(vrsum_bA, 4, pGIn+10*inDim+i, vl) ;
    vrsum_bB = _vel_vfadds_vvvl(vrsum_bB, _vel_vsll_vvsl(vrsum_bB,32,vl),vl) ;
    _vel_vstu_vssl(vrsum_bB, 4, pGIn+11*inDim+i, vl) ;
    vrsum_bC = _vel_vfadds_vvvl(vrsum_bC, _vel_vsll_vvsl(vrsum_bC,32,vl),vl) ;
    _vel_vstu_vssl(vrsum_bC, 4, pGIn+12*inDim+i, vl) ;
    vrsum_bD = _vel_vfadds_vvvl(vrsum_bD, _vel_vsll_vvsl(vrsum_bD,32,vl),vl) ;
    _vel_vstu_vssl(vrsum_bD, 4, pGIn+13*inDim+i, vl) ;
    vrsum_bE = _vel_vfadds_vvvl(vrsum_bE, _vel_vsll_vvsl(vrsum_bE,32,vl),vl) ;
    _vel_vstu_vssl(vrsum_bE, 4, pGIn+14*inDim+i, vl) ;
    vrsum_bF = _vel_vfadds_vvvl(vrsum_bF, _vel_vsll_vvsl(vrsum_bF,32,vl),vl) ;
    _vel_vstu_vssl(vrsum_bF, 4, pGIn+15*inDim+i, vl) ;

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
