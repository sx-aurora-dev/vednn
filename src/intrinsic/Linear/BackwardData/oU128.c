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

    __vr vrsum_b0 = _ve_vbrdu_vs_f32(0.f) ;

    for(int64_t o=0; o<outDim; o++)
    {
      __vr vrw = _ve_vldu_vss(4*outDim, pWeight+i*outDim+o) ;

      vrsum_b0 = _ve_vfmads_vvsv(vrsum_b0, pGOut[0*outDim+o], vrw ) ;
    }

    _ve_vstu_vss(vrsum_b0, 4, pGIn+0*inDim+i) ;
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

    __vr vrsum_b0 = _ve_vbrdu_vs_f32(0.f) ;
    __vr vrsum_b1 = _ve_vbrdu_vs_f32(0.f) ;

    for(int64_t o=0; o<outDim; o++)
    {
      __vr vrw = _ve_vldu_vss(4*outDim, pWeight+i*outDim+o) ;

      vrsum_b0 = _ve_vfmads_vvsv(vrsum_b0, pGOut[0*outDim+o], vrw ) ;
      vrsum_b1 = _ve_vfmads_vvsv(vrsum_b1, pGOut[1*outDim+o], vrw ) ;
    }

    _ve_vstu_vss(vrsum_b0, 4, pGIn+0*inDim+i) ;
    _ve_vstu_vss(vrsum_b1, 4, pGIn+1*inDim+i) ;
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

    __vr vrsum_b0 = _ve_vbrdu_vs_f32(0.f) ;
    __vr vrsum_b1 = _ve_vbrdu_vs_f32(0.f) ;
    __vr vrsum_b2 = _ve_vbrdu_vs_f32(0.f) ;

    for(int64_t o=0; o<outDim; o++)
    {
      __vr vrw = _ve_vldu_vss(4*outDim, pWeight+i*outDim+o) ;

      vrsum_b0 = _ve_vfmads_vvsv(vrsum_b0, pGOut[0*outDim+o], vrw ) ;
      vrsum_b1 = _ve_vfmads_vvsv(vrsum_b1, pGOut[1*outDim+o], vrw ) ;
      vrsum_b2 = _ve_vfmads_vvsv(vrsum_b2, pGOut[2*outDim+o], vrw ) ;
    }

    _ve_vstu_vss(vrsum_b0, 4, pGIn+0*inDim+i) ;
    _ve_vstu_vss(vrsum_b1, 4, pGIn+1*inDim+i) ;
    _ve_vstu_vss(vrsum_b2, 4, pGIn+2*inDim+i) ;
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

    __vr vrsum_b01 = _ve_vbrd_vs_i64(0UL) ;
    __vr vrsum_b23 = _ve_vbrd_vs_i64(0UL) ;

    for(int64_t o=0; o<outDim; o++)
    {
      __vr vrw = _ve_vldu_vss(4*outDim, pWeight+i*outDim+o) ;
      __vr vrwP = _ve_vshf_vvvs(vrw, vrw, VE_VSHUFFLE_YUZU) ;

      const uint64_t go01 = _ve_pack_f32p(pGOut+0*outDim+o, pGOut+1*outDim+o) ;
      const uint64_t go23 = _ve_pack_f32p(pGOut+2*outDim+o, pGOut+3*outDim+o) ;

      vrsum_b01 = _ve_pvfmad_vvsv(vrsum_b01, go01, vrwP ) ;
      vrsum_b23 = _ve_pvfmad_vvsv(vrsum_b23, go23, vrwP ) ;
    }

    _ve_vstu_vss(vrsum_b01, 4, pGIn+0*inDim+i) ;
    _ve_vstl_vss(vrsum_b01, 4, pGIn+1*inDim+i) ;
    _ve_vstu_vss(vrsum_b23, 4, pGIn+2*inDim+i) ;
    _ve_vstl_vss(vrsum_b23, 4, pGIn+3*inDim+i) ;
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

    __vr vrsum_b01 = _ve_vbrd_vs_i64(0UL) ;
    __vr vrsum_b23 = _ve_vbrd_vs_i64(0UL) ;
    __vr vrsum_b4  = _ve_vbrdu_vs_f32(0.f) ;

    for(int64_t o=0; o<outDim; o++)
    {
      __vr vrw = _ve_vldu_vss(4*outDim, pWeight+i*outDim+o) ;
      __vr vrwP = _ve_vshf_vvvs(vrw, vrw, VE_VSHUFFLE_YUZU) ;

      const uint64_t go01 = _ve_pack_f32p(pGOut+0*outDim+o, pGOut+1*outDim+o) ;
      const uint64_t go23 = _ve_pack_f32p(pGOut+2*outDim+o, pGOut+3*outDim+o) ;

      vrsum_b01 = _ve_pvfmad_vvsv(vrsum_b01, go01, vrwP ) ;
      vrsum_b23 = _ve_pvfmad_vvsv(vrsum_b23, go23, vrwP ) ;
      vrsum_b4 = _ve_vfmads_vvsv(vrsum_b4, pGOut[4*outDim+o], vrw ) ;
    }

    _ve_vstu_vss(vrsum_b01, 4, pGIn+0*inDim+i) ;
    _ve_vstl_vss(vrsum_b01, 4, pGIn+1*inDim+i) ;
    _ve_vstu_vss(vrsum_b23, 4, pGIn+2*inDim+i) ;
    _ve_vstl_vss(vrsum_b23, 4, pGIn+3*inDim+i) ;
    _ve_vstu_vss(vrsum_b4,  4, pGIn+4*inDim+i) ;
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

    __vr vrsum_b01 = _ve_vbrd_vs_i64(0UL) ;
    __vr vrsum_b23 = _ve_vbrd_vs_i64(0UL) ;
    __vr vrsum_b45 = _ve_vbrd_vs_i64(0UL) ;

    for(int64_t o=0; o<outDim; o++)
    {
      __vr vrw = _ve_vldu_vss(4*outDim, pWeight+i*outDim+o) ;
      __vr vrwP = _ve_vshf_vvvs(vrw, vrw, VE_VSHUFFLE_YUZU) ;

      const uint64_t go01 = _ve_pack_f32p(pGOut+0*outDim+o, pGOut+1*outDim+o) ;
      const uint64_t go23 = _ve_pack_f32p(pGOut+2*outDim+o, pGOut+3*outDim+o) ;
      const uint64_t go45 = _ve_pack_f32p(pGOut+4*outDim+o, pGOut+5*outDim+o) ;

      vrsum_b01 = _ve_pvfmad_vvsv(vrsum_b01, go01, vrwP ) ;
      vrsum_b23 = _ve_pvfmad_vvsv(vrsum_b23, go23, vrwP ) ;
      vrsum_b45 = _ve_pvfmad_vvsv(vrsum_b45, go45, vrwP ) ;
    }

    _ve_vstu_vss(vrsum_b01, 4, pGIn+0*inDim+i) ;
    _ve_vstl_vss(vrsum_b01, 4, pGIn+1*inDim+i) ;
    _ve_vstu_vss(vrsum_b23, 4, pGIn+2*inDim+i) ;
    _ve_vstl_vss(vrsum_b23, 4, pGIn+3*inDim+i) ;
    _ve_vstu_vss(vrsum_b45, 4, pGIn+4*inDim+i) ;
    _ve_vstl_vss(vrsum_b45, 4, pGIn+5*inDim+i) ;

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

    __vr vrsum_b01 = _ve_vbrd_vs_i64(0UL) ;
    __vr vrsum_b23 = _ve_vbrd_vs_i64(0UL) ;
    __vr vrsum_b45 = _ve_vbrd_vs_i64(0UL) ;
    __vr vrsum_b6  = _ve_vbrdu_vs_f32(0.f) ;

    for(int64_t o=0; o<outDim; o++)
    {
      __vr vrw = _ve_vldu_vss(4*outDim, pWeight+i*outDim+o) ;
      __vr vrwP = _ve_vshf_vvvs(vrw, vrw, VE_VSHUFFLE_YUZU) ;

      const uint64_t go01 = _ve_pack_f32p(pGOut+0*outDim+o, pGOut+1*outDim+o) ;
      const uint64_t go23 = _ve_pack_f32p(pGOut+2*outDim+o, pGOut+3*outDim+o) ;
      const uint64_t go45 = _ve_pack_f32p(pGOut+4*outDim+o, pGOut+5*outDim+o) ;

      vrsum_b01 = _ve_pvfmad_vvsv(vrsum_b01, go01, vrwP ) ;
      vrsum_b23 = _ve_pvfmad_vvsv(vrsum_b23, go23, vrwP ) ;
      vrsum_b45 = _ve_pvfmad_vvsv(vrsum_b45, go45, vrwP ) ;
      vrsum_b6  = _ve_vfmads_vvsv(vrsum_b6, pGOut[6*outDim+o], vrw ) ;
    }

    _ve_vstu_vss(vrsum_b01, 4, pGIn+0*inDim+i) ;
    _ve_vstl_vss(vrsum_b01, 4, pGIn+1*inDim+i) ;
    _ve_vstu_vss(vrsum_b23, 4, pGIn+2*inDim+i) ;
    _ve_vstl_vss(vrsum_b23, 4, pGIn+3*inDim+i) ;
    _ve_vstu_vss(vrsum_b45, 4, pGIn+4*inDim+i) ;
    _ve_vstl_vss(vrsum_b45, 4, pGIn+5*inDim+i) ;
    _ve_vstu_vss(vrsum_b6,  4, pGIn+6*inDim+i) ;
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

    __vr vrsum_b01 = _ve_vbrd_vs_i64(0UL) ;
    __vr vrsum_b23 = _ve_vbrd_vs_i64(0UL) ;
    __vr vrsum_b45 = _ve_vbrd_vs_i64(0UL) ;
    __vr vrsum_b67 = _ve_vbrd_vs_i64(0UL) ;

    for(int64_t o=0; o<outDim; o++)
    {
      __vr vrw = _ve_vldu_vss(4*outDim, pWeight+i*outDim+o) ;
      __vr vrwP = _ve_vshf_vvvs(vrw, vrw, VE_VSHUFFLE_YUZU) ;

      const uint64_t go01 = _ve_pack_f32p(pGOut+0*outDim+o, pGOut+1*outDim+o) ;
      const uint64_t go23 = _ve_pack_f32p(pGOut+2*outDim+o, pGOut+3*outDim+o) ;
      const uint64_t go45 = _ve_pack_f32p(pGOut+4*outDim+o, pGOut+5*outDim+o) ;
      const uint64_t go67 = _ve_pack_f32p(pGOut+6*outDim+o, pGOut+7*outDim+o) ;

      vrsum_b01 = _ve_pvfmad_vvsv(vrsum_b01, go01, vrwP ) ;
      vrsum_b23 = _ve_pvfmad_vvsv(vrsum_b23, go23, vrwP ) ;
      vrsum_b45 = _ve_pvfmad_vvsv(vrsum_b45, go45, vrwP ) ;
      vrsum_b67 = _ve_pvfmad_vvsv(vrsum_b67, go67, vrwP ) ;
    }

    _ve_vstu_vss(vrsum_b01, 4, pGIn+0*inDim+i) ;
    _ve_vstl_vss(vrsum_b01, 4, pGIn+1*inDim+i) ;
    _ve_vstu_vss(vrsum_b23, 4, pGIn+2*inDim+i) ;
    _ve_vstl_vss(vrsum_b23, 4, pGIn+3*inDim+i) ;
    _ve_vstu_vss(vrsum_b45, 4, pGIn+4*inDim+i) ;
    _ve_vstl_vss(vrsum_b45, 4, pGIn+5*inDim+i) ;
    _ve_vstu_vss(vrsum_b67, 4, pGIn+6*inDim+i) ;
    _ve_vstl_vss(vrsum_b67, 4, pGIn+7*inDim+i) ;
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

    __vr vrsum_b01 = _ve_vbrd_vs_i64(0UL) ;
    __vr vrsum_b23 = _ve_vbrd_vs_i64(0UL) ;
    __vr vrsum_b45 = _ve_vbrd_vs_i64(0UL) ;
    __vr vrsum_b67 = _ve_vbrd_vs_i64(0UL) ;
    __vr vrsum_b89 = _ve_vbrd_vs_i64(0UL) ;
    __vr vrsum_bAB = _ve_vbrd_vs_i64(0UL) ;
    __vr vrsum_bCD = _ve_vbrd_vs_i64(0UL) ;
    __vr vrsum_bEF = _ve_vbrd_vs_i64(0UL) ;


    for(int64_t o=0; o<outDim; o++)
    {
      __vr vrw = _ve_vldu_vss(4*outDim, pWeight+i*outDim+o) ;
      __vr vrwP = _ve_vshf_vvvs(vrw, vrw, VE_VSHUFFLE_YUZU) ;

      const uint64_t go01 = _ve_pack_f32p(pGOut+0*outDim+o, pGOut+1*outDim+o) ;
      const uint64_t go23 = _ve_pack_f32p(pGOut+2*outDim+o, pGOut+3*outDim+o) ;
      const uint64_t go45 = _ve_pack_f32p(pGOut+4*outDim+o, pGOut+5*outDim+o) ;
      const uint64_t go67 = _ve_pack_f32p(pGOut+6*outDim+o, pGOut+7*outDim+o) ;
      const uint64_t go89 = _ve_pack_f32p(pGOut+8*outDim+o, pGOut+9*outDim+o) ;
      const uint64_t goAB = _ve_pack_f32p(pGOut+10*outDim+o, pGOut+11*outDim+o) ;
      const uint64_t goCD = _ve_pack_f32p(pGOut+12*outDim+o, pGOut+13*outDim+o) ;
      const uint64_t goEF = _ve_pack_f32p(pGOut+14*outDim+o, pGOut+15*outDim+o) ;

      vrsum_b01 = _ve_pvfmad_vvsv(vrsum_b01, go01, vrwP ) ;
      vrsum_b23 = _ve_pvfmad_vvsv(vrsum_b23, go23, vrwP ) ;
      vrsum_b45 = _ve_pvfmad_vvsv(vrsum_b45, go45, vrwP ) ;
      vrsum_b67 = _ve_pvfmad_vvsv(vrsum_b67, go67, vrwP ) ;
      vrsum_b89 = _ve_pvfmad_vvsv(vrsum_b89, go89, vrwP ) ;
      vrsum_bAB = _ve_pvfmad_vvsv(vrsum_bAB, goAB, vrwP ) ;
      vrsum_bCD = _ve_pvfmad_vvsv(vrsum_bCD, goCD, vrwP ) ;
      vrsum_bEF = _ve_pvfmad_vvsv(vrsum_bEF, goEF, vrwP ) ;
    }

    _ve_vstu_vss(vrsum_b01, 4, pGIn+0*inDim+i) ;
    _ve_vstl_vss(vrsum_b01, 4, pGIn+1*inDim+i) ;
    _ve_vstu_vss(vrsum_b23, 4, pGIn+2*inDim+i) ;
    _ve_vstl_vss(vrsum_b23, 4, pGIn+3*inDim+i) ;
    _ve_vstu_vss(vrsum_b45, 4, pGIn+4*inDim+i) ;
    _ve_vstl_vss(vrsum_b45, 4, pGIn+5*inDim+i) ;
    _ve_vstu_vss(vrsum_b67, 4, pGIn+6*inDim+i) ;
    _ve_vstl_vss(vrsum_b67, 4, pGIn+7*inDim+i) ;
    _ve_vstu_vss(vrsum_b89, 4, pGIn+8*inDim+i) ;
    _ve_vstl_vss(vrsum_b89, 4, pGIn+9*inDim+i) ;
    _ve_vstu_vss(vrsum_bAB, 4, pGIn+10*inDim+i) ;
    _ve_vstl_vss(vrsum_bAB, 4, pGIn+11*inDim+i) ;
    _ve_vstu_vss(vrsum_bCD, 4, pGIn+12*inDim+i) ;
    _ve_vstl_vss(vrsum_bCD, 4, pGIn+13*inDim+i) ;
    _ve_vstu_vss(vrsum_bEF, 4, pGIn+14*inDim+i) ;
    _ve_vstl_vss(vrsum_bEF, 4, pGIn+15*inDim+i) ;
  }
}

vednnError_t vednnLinearBackwardData_oU128(
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
