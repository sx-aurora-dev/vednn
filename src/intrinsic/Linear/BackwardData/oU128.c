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
    __vr vrsum_b0 = _vel_vbrds_vsl(0.f, vl) ;

    for(int64_t o=0; o<outDim; o++)
    {
      __vr vrw = _vel_vldu_vssl(4*outDim, pWeight+i*outDim+o, vl) ;

      vrsum_b0 = _vel_vfmads_vvsvl(vrsum_b0, pGOut[0*outDim+o], vrw, vl) ;
    }

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
    __vr vrsum_b0 = _vel_vbrds_vsl(0.f, vl) ;
    __vr vrsum_b1 = _vel_vbrds_vsl(0.f, vl) ;

    for(int64_t o=0; o<outDim; o++)
    {
      __vr vrw = _vel_vldu_vssl(4*outDim, pWeight+i*outDim+o, vl) ;

      vrsum_b0 = _vel_vfmads_vvsvl(vrsum_b0, pGOut[0*outDim+o], vrw, vl) ;
      vrsum_b1 = _vel_vfmads_vvsvl(vrsum_b1, pGOut[1*outDim+o], vrw, vl) ;
    }

    _vel_vstu_vssl(vrsum_b0, 4, pGIn+0*inDim+i, vl) ;
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
    __vr vrsum_b0 = _vel_vbrds_vsl(0.f, vl) ;
    __vr vrsum_b1 = _vel_vbrds_vsl(0.f, vl) ;
    __vr vrsum_b2 = _vel_vbrds_vsl(0.f, vl) ;

    for(int64_t o=0; o<outDim; o++)
    {
      __vr vrw = _vel_vldu_vssl(4*outDim, pWeight+i*outDim+o, vl) ;

      vrsum_b0 = _vel_vfmads_vvsvl(vrsum_b0, pGOut[0*outDim+o], vrw, vl) ;
      vrsum_b1 = _vel_vfmads_vvsvl(vrsum_b1, pGOut[1*outDim+o], vrw, vl) ;
      vrsum_b2 = _vel_vfmads_vvsvl(vrsum_b2, pGOut[2*outDim+o], vrw, vl) ;
    }

    _vel_vstu_vssl(vrsum_b0, 4, pGIn+0*inDim+i, vl) ;
    _vel_vstu_vssl(vrsum_b1, 4, pGIn+1*inDim+i, vl) ;
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
    __vr vrsum_b01 = _vel_vbrdl_vsl(0UL, vl) ;
    __vr vrsum_b23 = _vel_vbrdl_vsl(0UL, vl) ;

    for(int64_t o=0; o<outDim; o++)
    {
      __vr vrw = _vel_vldu_vssl(4*outDim, pWeight+i*outDim+o, vl) ;
      __vr vrwP = _vel_vshf_vvvsl(vrw, vrw, VE_VSHUFFLE_YUZU, vl) ;

      const uint64_t go01 = _vel_pack_f32p(pGOut+0*outDim+o, pGOut+1*outDim+o) ;
      const uint64_t go23 = _vel_pack_f32p(pGOut+2*outDim+o, pGOut+3*outDim+o) ;

      vrsum_b01 = _vel_pvfmad_vvsvl(vrsum_b01, go01, vrwP, vl) ;
      vrsum_b23 = _vel_pvfmad_vvsvl(vrsum_b23, go23, vrwP, vl) ;
    }

    _vel_vstu_vssl(vrsum_b01, 4, pGIn+0*inDim+i, vl) ;
    _vel_vstl_vssl(vrsum_b01, 4, pGIn+1*inDim+i, vl) ;
    _vel_vstu_vssl(vrsum_b23, 4, pGIn+2*inDim+i, vl) ;
    _vel_vstl_vssl(vrsum_b23, 4, pGIn+3*inDim+i, vl) ;
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
    __vr vrsum_b01 = _vel_vbrdl_vsl(0UL, vl) ;
    __vr vrsum_b23 = _vel_vbrdl_vsl(0UL, vl) ;
    __vr vrsum_b4  = _vel_vbrds_vsl(0.f, vl) ;

    for(int64_t o=0; o<outDim; o++)
    {
      __vr vrw = _vel_vldu_vssl(4*outDim, pWeight+i*outDim+o, vl) ;
      __vr vrwP = _vel_vshf_vvvsl(vrw, vrw, VE_VSHUFFLE_YUZU, vl) ;

      const uint64_t go01 = _vel_pack_f32p(pGOut+0*outDim+o, pGOut+1*outDim+o) ;
      const uint64_t go23 = _vel_pack_f32p(pGOut+2*outDim+o, pGOut+3*outDim+o) ;

      vrsum_b01 = _vel_pvfmad_vvsvl(vrsum_b01, go01, vrwP, vl) ;
      vrsum_b23 = _vel_pvfmad_vvsvl(vrsum_b23, go23, vrwP, vl) ;
      vrsum_b4 = _vel_vfmads_vvsvl(vrsum_b4, pGOut[4*outDim+o], vrw, vl) ;
    }

    _vel_vstu_vssl(vrsum_b01, 4, pGIn+0*inDim+i, vl) ;
    _vel_vstl_vssl(vrsum_b01, 4, pGIn+1*inDim+i, vl) ;
    _vel_vstu_vssl(vrsum_b23, 4, pGIn+2*inDim+i, vl) ;
    _vel_vstl_vssl(vrsum_b23, 4, pGIn+3*inDim+i, vl) ;
    _vel_vstu_vssl(vrsum_b4,  4, pGIn+4*inDim+i, vl) ;
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
    __vr vrsum_b01 = _vel_vbrdl_vsl(0UL, vl) ;
    __vr vrsum_b23 = _vel_vbrdl_vsl(0UL, vl) ;
    __vr vrsum_b45 = _vel_vbrdl_vsl(0UL, vl) ;

    for(int64_t o=0; o<outDim; o++)
    {
      __vr vrw = _vel_vldu_vssl(4*outDim, pWeight+i*outDim+o, vl) ;
      __vr vrwP = _vel_vshf_vvvsl(vrw, vrw, VE_VSHUFFLE_YUZU, vl) ;

      const uint64_t go01 = _vel_pack_f32p(pGOut+0*outDim+o, pGOut+1*outDim+o) ;
      const uint64_t go23 = _vel_pack_f32p(pGOut+2*outDim+o, pGOut+3*outDim+o) ;
      const uint64_t go45 = _vel_pack_f32p(pGOut+4*outDim+o, pGOut+5*outDim+o) ;

      vrsum_b01 = _vel_pvfmad_vvsvl(vrsum_b01, go01, vrwP, vl) ;
      vrsum_b23 = _vel_pvfmad_vvsvl(vrsum_b23, go23, vrwP, vl) ;
      vrsum_b45 = _vel_pvfmad_vvsvl(vrsum_b45, go45, vrwP, vl) ;
    }

    _vel_vstu_vssl(vrsum_b01, 4, pGIn+0*inDim+i, vl) ;
    _vel_vstl_vssl(vrsum_b01, 4, pGIn+1*inDim+i, vl) ;
    _vel_vstu_vssl(vrsum_b23, 4, pGIn+2*inDim+i, vl) ;
    _vel_vstl_vssl(vrsum_b23, 4, pGIn+3*inDim+i, vl) ;
    _vel_vstu_vssl(vrsum_b45, 4, pGIn+4*inDim+i, vl) ;
    _vel_vstl_vssl(vrsum_b45, 4, pGIn+5*inDim+i, vl) ;

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
    __vr vrsum_b01 = _vel_vbrdl_vsl(0UL, vl) ;
    __vr vrsum_b23 = _vel_vbrdl_vsl(0UL, vl) ;
    __vr vrsum_b45 = _vel_vbrdl_vsl(0UL, vl) ;
    __vr vrsum_b6  = _vel_vbrds_vsl(0.f, vl) ;

    for(int64_t o=0; o<outDim; o++)
    {
      __vr vrw = _vel_vldu_vssl(4*outDim, pWeight+i*outDim+o, vl) ;
      __vr vrwP = _vel_vshf_vvvsl(vrw, vrw, VE_VSHUFFLE_YUZU, vl) ;

      const uint64_t go01 = _vel_pack_f32p(pGOut+0*outDim+o, pGOut+1*outDim+o) ;
      const uint64_t go23 = _vel_pack_f32p(pGOut+2*outDim+o, pGOut+3*outDim+o) ;
      const uint64_t go45 = _vel_pack_f32p(pGOut+4*outDim+o, pGOut+5*outDim+o) ;

      vrsum_b01 = _vel_pvfmad_vvsvl(vrsum_b01, go01, vrwP, vl) ;
      vrsum_b23 = _vel_pvfmad_vvsvl(vrsum_b23, go23, vrwP, vl) ;
      vrsum_b45 = _vel_pvfmad_vvsvl(vrsum_b45, go45, vrwP, vl) ;
      vrsum_b6  = _vel_vfmads_vvsvl(vrsum_b6, pGOut[6*outDim+o], vrw, vl) ;
    }

    _vel_vstu_vssl(vrsum_b01, 4, pGIn+0*inDim+i, vl) ;
    _vel_vstl_vssl(vrsum_b01, 4, pGIn+1*inDim+i, vl) ;
    _vel_vstu_vssl(vrsum_b23, 4, pGIn+2*inDim+i, vl) ;
    _vel_vstl_vssl(vrsum_b23, 4, pGIn+3*inDim+i, vl) ;
    _vel_vstu_vssl(vrsum_b45, 4, pGIn+4*inDim+i, vl) ;
    _vel_vstl_vssl(vrsum_b45, 4, pGIn+5*inDim+i, vl) ;
    _vel_vstu_vssl(vrsum_b6,  4, pGIn+6*inDim+i, vl) ;
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
    __vr vrsum_b01 = _vel_vbrdl_vsl(0UL, vl) ;
    __vr vrsum_b23 = _vel_vbrdl_vsl(0UL, vl) ;
    __vr vrsum_b45 = _vel_vbrdl_vsl(0UL, vl) ;
    __vr vrsum_b67 = _vel_vbrdl_vsl(0UL, vl) ;

    for(int64_t o=0; o<outDim; o++)
    {
      __vr vrw = _vel_vldu_vssl(4*outDim, pWeight+i*outDim+o, vl) ;
      __vr vrwP = _vel_vshf_vvvsl(vrw, vrw, VE_VSHUFFLE_YUZU, vl) ;

      const uint64_t go01 = _vel_pack_f32p(pGOut+0*outDim+o, pGOut+1*outDim+o) ;
      const uint64_t go23 = _vel_pack_f32p(pGOut+2*outDim+o, pGOut+3*outDim+o) ;
      const uint64_t go45 = _vel_pack_f32p(pGOut+4*outDim+o, pGOut+5*outDim+o) ;
      const uint64_t go67 = _vel_pack_f32p(pGOut+6*outDim+o, pGOut+7*outDim+o) ;

      vrsum_b01 = _vel_pvfmad_vvsvl(vrsum_b01, go01, vrwP, vl) ;
      vrsum_b23 = _vel_pvfmad_vvsvl(vrsum_b23, go23, vrwP, vl) ;
      vrsum_b45 = _vel_pvfmad_vvsvl(vrsum_b45, go45, vrwP, vl) ;
      vrsum_b67 = _vel_pvfmad_vvsvl(vrsum_b67, go67, vrwP, vl) ;
    }

    _vel_vstu_vssl(vrsum_b01, 4, pGIn+0*inDim+i, vl) ;
    _vel_vstl_vssl(vrsum_b01, 4, pGIn+1*inDim+i, vl) ;
    _vel_vstu_vssl(vrsum_b23, 4, pGIn+2*inDim+i, vl) ;
    _vel_vstl_vssl(vrsum_b23, 4, pGIn+3*inDim+i, vl) ;
    _vel_vstu_vssl(vrsum_b45, 4, pGIn+4*inDim+i, vl) ;
    _vel_vstl_vssl(vrsum_b45, 4, pGIn+5*inDim+i, vl) ;
    _vel_vstu_vssl(vrsum_b67, 4, pGIn+6*inDim+i, vl) ;
    _vel_vstl_vssl(vrsum_b67, 4, pGIn+7*inDim+i, vl) ;
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
    __vr vrsum_b01 = _vel_vbrdl_vsl(0UL, vl) ;
    __vr vrsum_b23 = _vel_vbrdl_vsl(0UL, vl) ;
    __vr vrsum_b45 = _vel_vbrdl_vsl(0UL, vl) ;
    __vr vrsum_b67 = _vel_vbrdl_vsl(0UL, vl) ;
    __vr vrsum_b89 = _vel_vbrdl_vsl(0UL, vl) ;
    __vr vrsum_bAB = _vel_vbrdl_vsl(0UL, vl) ;
    __vr vrsum_bCD = _vel_vbrdl_vsl(0UL, vl) ;
    __vr vrsum_bEF = _vel_vbrdl_vsl(0UL, vl) ;


    for(int64_t o=0; o<outDim; o++)
    {
      __vr vrw = _vel_vldu_vssl(4*outDim, pWeight+i*outDim+o, vl) ;
      __vr vrwP = _vel_vshf_vvvsl(vrw, vrw, VE_VSHUFFLE_YUZU, vl) ;

      const uint64_t go01 = _vel_pack_f32p(pGOut+0*outDim+o, pGOut+1*outDim+o) ;
      const uint64_t go23 = _vel_pack_f32p(pGOut+2*outDim+o, pGOut+3*outDim+o) ;
      const uint64_t go45 = _vel_pack_f32p(pGOut+4*outDim+o, pGOut+5*outDim+o) ;
      const uint64_t go67 = _vel_pack_f32p(pGOut+6*outDim+o, pGOut+7*outDim+o) ;
      const uint64_t go89 = _vel_pack_f32p(pGOut+8*outDim+o, pGOut+9*outDim+o) ;
      const uint64_t goAB = _vel_pack_f32p(pGOut+10*outDim+o, pGOut+11*outDim+o) ;
      const uint64_t goCD = _vel_pack_f32p(pGOut+12*outDim+o, pGOut+13*outDim+o) ;
      const uint64_t goEF = _vel_pack_f32p(pGOut+14*outDim+o, pGOut+15*outDim+o) ;

      vrsum_b01 = _vel_pvfmad_vvsvl(vrsum_b01, go01, vrwP, vl) ;
      vrsum_b23 = _vel_pvfmad_vvsvl(vrsum_b23, go23, vrwP, vl) ;
      vrsum_b45 = _vel_pvfmad_vvsvl(vrsum_b45, go45, vrwP, vl) ;
      vrsum_b67 = _vel_pvfmad_vvsvl(vrsum_b67, go67, vrwP, vl) ;
      vrsum_b89 = _vel_pvfmad_vvsvl(vrsum_b89, go89, vrwP, vl) ;
      vrsum_bAB = _vel_pvfmad_vvsvl(vrsum_bAB, goAB, vrwP, vl) ;
      vrsum_bCD = _vel_pvfmad_vvsvl(vrsum_bCD, goCD, vrwP, vl) ;
      vrsum_bEF = _vel_pvfmad_vvsvl(vrsum_bEF, goEF, vrwP, vl) ;
    }

    _vel_vstu_vssl(vrsum_b01, 4, pGIn+0*inDim+i, vl) ;
    _vel_vstl_vssl(vrsum_b01, 4, pGIn+1*inDim+i, vl) ;
    _vel_vstu_vssl(vrsum_b23, 4, pGIn+2*inDim+i, vl) ;
    _vel_vstl_vssl(vrsum_b23, 4, pGIn+3*inDim+i, vl) ;
    _vel_vstu_vssl(vrsum_b45, 4, pGIn+4*inDim+i, vl) ;
    _vel_vstl_vssl(vrsum_b45, 4, pGIn+5*inDim+i, vl) ;
    _vel_vstu_vssl(vrsum_b67, 4, pGIn+6*inDim+i, vl) ;
    _vel_vstl_vssl(vrsum_b67, 4, pGIn+7*inDim+i, vl) ;
    _vel_vstu_vssl(vrsum_b89, 4, pGIn+8*inDim+i, vl) ;
    _vel_vstl_vssl(vrsum_b89, 4, pGIn+9*inDim+i, vl) ;
    _vel_vstu_vssl(vrsum_bAB, 4, pGIn+10*inDim+i, vl) ;
    _vel_vstl_vssl(vrsum_bAB, 4, pGIn+11*inDim+i, vl) ;
    _vel_vstu_vssl(vrsum_bCD, 4, pGIn+12*inDim+i, vl) ;
    _vel_vstl_vssl(vrsum_bCD, 4, pGIn+13*inDim+i, vl) ;
    _vel_vstu_vssl(vrsum_bEF, 4, pGIn+14*inDim+i, vl) ;
    _vel_vstl_vssl(vrsum_bEF, 4, pGIn+15*inDim+i, vl) ;
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
