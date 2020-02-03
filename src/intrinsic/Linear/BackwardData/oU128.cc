#include <stdint.h>
#include <float.h>

#include <stdio.h>

#include "vednn.h"

#include "velintrin.h"
#define VLEN	(256)

template <int BATCH>
static inline void func_batch_odd(
  const uint64_t	inDim,
  const uint64_t	outDim,
  const float * 	pGOut,
  const float * 	pWeight,
  float * 		pGIn
)
{
  for(int i=0; i<inDim; i+=VLEN) {
    const int64_t vl = inDim - i < VLEN ? inDim - i : VLEN ;
    __vr vrsum_b0  = _vel_vbrds_vsl(0.f, vl) ;
    __vr vrsum_b12 = _vel_vbrdl_vsl(0UL, vl) ;
    __vr vrsum_b34 = _vel_vbrdl_vsl(0UL, vl) ;
    __vr vrsum_b56 = _vel_vbrdl_vsl(0UL, vl) ;
    __vr vrsum_b78 = _vel_vbrdl_vsl(0UL, vl) ;
    __vr vrsum_b9A = _vel_vbrdl_vsl(0UL, vl) ;
    __vr vrsum_bBC = _vel_vbrdl_vsl(0UL, vl) ;
    __vr vrsum_bDE = _vel_vbrdl_vsl(0UL, vl) ;

    for(int64_t o=0; o<outDim; o++)
    {
      __vr vrw = _vel_vldu_vssl(4*outDim, pWeight+i*outDim+o, vl) ;
      __vr vrwP = _vel_vshf_vvvsl(vrw, vrw, VE_VSHUFFLE_YUZU, vl) ;

      const float go0 = pGOut[0*outDim+o] ;
      vrsum_b0 = _vel_vfmads_vvsvl(vrsum_b0, go0, vrw, vl) ;

      if(BATCH>= 3) {
	const uint64_t go12 = _vel_pack_f32p(pGOut+1*outDim+o, pGOut+2*outDim+o) ;
	vrsum_b12 = _vel_pvfmad_vvsvl(vrsum_b12, go12, vrwP, vl) ;
      }
      if(BATCH>= 5) {
	const uint64_t go34 = _vel_pack_f32p(pGOut+3*outDim+o, pGOut+4*outDim+o) ;
	vrsum_b34 = _vel_pvfmad_vvsvl(vrsum_b34, go34, vrwP, vl) ;
      }
      if(BATCH>= 7) {
	const uint64_t go56 = _vel_pack_f32p(pGOut+5*outDim+o, pGOut+6*outDim+o) ;
	vrsum_b56 = _vel_pvfmad_vvsvl(vrsum_b56, go56, vrwP, vl) ;
      }
      if(BATCH>= 9) {
	const uint64_t go78 = _vel_pack_f32p(pGOut+7*outDim+o, pGOut+8*outDim+o) ;
	vrsum_b78 = _vel_pvfmad_vvsvl(vrsum_b78, go78, vrwP, vl) ;
      }
      if(BATCH>=11) {
	const uint64_t go9A = _vel_pack_f32p(pGOut+9*outDim+o, pGOut+10*outDim+o) ;
	vrsum_b9A = _vel_pvfmad_vvsvl(vrsum_b9A, go9A, vrwP, vl) ;
      }
      if(BATCH>=13) {
	const uint64_t goBC = _vel_pack_f32p(pGOut+11*outDim+o, pGOut+12*outDim+o) ;
	vrsum_bBC = _vel_pvfmad_vvsvl(vrsum_bBC, goBC, vrwP, vl) ;
      }
      if(BATCH>=15) {
	const uint64_t goDE = _vel_pack_f32p(pGOut+13*outDim+o, pGOut+14*outDim+o) ;
	vrsum_bDE = _vel_pvfmad_vvsvl(vrsum_bDE, goDE, vrwP, vl) ;
      }
    }

    _vel_vstu_vssl(vrsum_b0, 4, pGIn+0*inDim+i, vl) ;

    if(BATCH>= 3) {
      _vel_vstu_vssl(vrsum_b12, 4, pGIn+1*inDim+i, vl) ;
      _vel_vstl_vssl(vrsum_b12, 4, pGIn+2*inDim+i, vl) ;
    }
    if(BATCH>= 5) {
      _vel_vstu_vssl(vrsum_b34, 4, pGIn+3*inDim+i, vl) ;
      _vel_vstl_vssl(vrsum_b34, 4, pGIn+4*inDim+i, vl) ;
    }
    if(BATCH>= 7) {
      _vel_vstu_vssl(vrsum_b56, 4, pGIn+5*inDim+i, vl) ;
      _vel_vstl_vssl(vrsum_b56, 4, pGIn+6*inDim+i, vl) ;
    }
    if(BATCH>= 9) {
      _vel_vstu_vssl(vrsum_b78, 4, pGIn+7*inDim+i, vl) ;
      _vel_vstl_vssl(vrsum_b78, 4, pGIn+8*inDim+i, vl) ;
    }
    if(BATCH>=11) {
      _vel_vstu_vssl(vrsum_b9A, 4, pGIn+9*inDim+i, vl) ;
      _vel_vstl_vssl(vrsum_b9A, 4, pGIn+10*inDim+i, vl) ;
    }
    if(BATCH>=13) {
      _vel_vstu_vssl(vrsum_bBC, 4, pGIn+11*inDim+i, vl) ;
      _vel_vstl_vssl(vrsum_bBC, 4, pGIn+12*inDim+i, vl) ;
    }
    if(BATCH>=15) {
      _vel_vstu_vssl(vrsum_bDE, 4, pGIn+13*inDim+i, vl) ;
      _vel_vstl_vssl(vrsum_bDE, 4, pGIn+14*inDim+i, vl) ;
    }
  }
}

template <int BATCH>
static inline void func_batch_even(
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

      if(BATCH>= 2) {
	const uint64_t go01 = _vel_pack_f32p(pGOut+0*outDim+o, pGOut+1*outDim+o) ;
	vrsum_b01 = _vel_pvfmad_vvsvl(vrsum_b01, go01, vrwP, vl) ;
      }
      if(BATCH>= 4) {
	const uint64_t go23 = _vel_pack_f32p(pGOut+2*outDim+o, pGOut+3*outDim+o) ;
	vrsum_b23 = _vel_pvfmad_vvsvl(vrsum_b23, go23, vrwP, vl) ;
      }
      if(BATCH>= 6) {
	const uint64_t go45 = _vel_pack_f32p(pGOut+4*outDim+o, pGOut+5*outDim+o) ;
	vrsum_b45 = _vel_pvfmad_vvsvl(vrsum_b45, go45, vrwP, vl) ;
      }
      if(BATCH>= 8) {
	const uint64_t go67 = _vel_pack_f32p(pGOut+6*outDim+o, pGOut+7*outDim+o) ;
	vrsum_b67 = _vel_pvfmad_vvsvl(vrsum_b67, go67, vrwP, vl) ;
      }
      if(BATCH>=10) {
	const uint64_t go89 = _vel_pack_f32p(pGOut+8*outDim+o, pGOut+9*outDim+o) ;
	vrsum_b89 = _vel_pvfmad_vvsvl(vrsum_b89, go89, vrwP, vl) ;
      }
      if(BATCH>=12) {
	const uint64_t goAB = _vel_pack_f32p(pGOut+10*outDim+o, pGOut+11*outDim+o) ;
	vrsum_bAB = _vel_pvfmad_vvsvl(vrsum_bAB, goAB, vrwP, vl) ;
      }
      if(BATCH>=14) {
	const uint64_t goCD = _vel_pack_f32p(pGOut+12*outDim+o, pGOut+13*outDim+o) ;
	vrsum_bCD = _vel_pvfmad_vvsvl(vrsum_bCD, goCD, vrwP, vl) ;
      }
      if(BATCH>=16) {
	const uint64_t goEF = _vel_pack_f32p(pGOut+14*outDim+o, pGOut+15*outDim+o) ;
	vrsum_bEF = _vel_pvfmad_vvsvl(vrsum_bEF, goEF, vrwP, vl) ;
      }
    }

    if(BATCH>= 2) {
      _vel_vstu_vssl(vrsum_b01, 4, pGIn+0*inDim+i, vl) ;
      _vel_vstl_vssl(vrsum_b01, 4, pGIn+1*inDim+i, vl) ;
    }
    if(BATCH>= 4) {
      _vel_vstu_vssl(vrsum_b23, 4, pGIn+2*inDim+i, vl) ;
      _vel_vstl_vssl(vrsum_b23, 4, pGIn+3*inDim+i, vl) ;
    }
    if(BATCH>= 6) {
      _vel_vstu_vssl(vrsum_b45, 4, pGIn+4*inDim+i, vl) ;
      _vel_vstl_vssl(vrsum_b45, 4, pGIn+5*inDim+i, vl) ;
    }
    if(BATCH>= 8) {
      _vel_vstu_vssl(vrsum_b67, 4, pGIn+6*inDim+i, vl) ;
      _vel_vstl_vssl(vrsum_b67, 4, pGIn+7*inDim+i, vl) ;
    }
    if(BATCH>=10) {
      _vel_vstu_vssl(vrsum_b89, 4, pGIn+8*inDim+i, vl) ;
      _vel_vstl_vssl(vrsum_b89, 4, pGIn+9*inDim+i, vl) ;
    }
    if(BATCH>=12) {
      _vel_vstu_vssl(vrsum_bAB, 4, pGIn+10*inDim+i, vl) ;
      _vel_vstl_vssl(vrsum_bAB, 4, pGIn+11*inDim+i, vl) ;
    }
    if(BATCH>=14) {
      _vel_vstu_vssl(vrsum_bCD, 4, pGIn+12*inDim+i, vl) ;
      _vel_vstl_vssl(vrsum_bCD, 4, pGIn+13*inDim+i, vl) ;
    }
    if(BATCH>=16) {
      _vel_vstu_vssl(vrsum_bEF, 4, pGIn+14*inDim+i, vl) ;
      _vel_vstl_vssl(vrsum_bEF, 4, pGIn+15*inDim+i, vl) ;
    }
  }
}

extern "C"
vednnError_t vednnLinearBackwardData_oU128(
    const uint64_t			inDim,
    const uint64_t			outDim,
    const uint64_t			nBatch,
    const void * __restrict__		pDataGradOut,
    const void * __restrict__		pDataWeight,
    void * __restrict__			pDataGradIn
)
{
  const float * __restrict__ pGOut   = (const float * __restrict__) pDataGradOut;
  const float * __restrict__ pWeight = (const float * __restrict__) pDataWeight;
  float * __restrict__ const pGIn    = (float * __restrict__ const) pDataGradIn;

  int64_t n=0;
  int64_t batchRemain = nBatch & 0xF ;

  switch( batchRemain ) {
  case 1:
    func_batch_odd<1>(inDim, outDim, pGOut+n*outDim, pWeight, pGIn+n*inDim ) ;
    n+=7;
    break ;
  case 2:
    func_batch_even<2>(inDim, outDim, pGOut+n*outDim, pWeight, pGIn+n*inDim ) ;
    n+=2 ;
    break ;
  case 3:
    func_batch_odd<3>(inDim, outDim, pGOut+n*outDim, pWeight, pGIn+n*inDim ) ;
    n+=3;
    break ;
  case 4:
    func_batch_even<4>(inDim, outDim, pGOut+n*outDim, pWeight, pGIn+n*inDim ) ;
    n+=4 ;
    break ;
  case 5:
    func_batch_odd<5>(inDim, outDim, pGOut+n*outDim, pWeight, pGIn+n*inDim ) ;
    n+=5;
    break ;
  case 6:
    func_batch_even<6>(inDim, outDim, pGOut+n*outDim, pWeight, pGIn+n*inDim ) ;
    n+=6 ;
    break ;
  case 7:
    func_batch_odd<7>(inDim, outDim, pGOut+n*outDim, pWeight, pGIn+n*inDim ) ;
    n+=7;
    break ;
  case 8:
    func_batch_even<8>(inDim, outDim, pGOut+n*outDim, pWeight, pGIn+n*inDim ) ;
    n+=8 ;
    break ;
  case 9:
    func_batch_odd<9>(inDim, outDim, pGOut+n*outDim, pWeight, pGIn+n*inDim ) ;
    n+=9;
    break ;
  case 10:
    func_batch_even<10>(inDim, outDim, pGOut+n*outDim, pWeight, pGIn+n*inDim ) ;
    n+=10 ;
    break ;
  case 11:
    func_batch_odd<11>(inDim, outDim, pGOut+n*outDim, pWeight, pGIn+n*inDim ) ;
    n+=11;
    break ;
  case 12:
    func_batch_even<12>(inDim, outDim, pGOut+n*outDim, pWeight, pGIn+n*inDim ) ;
    n+=12 ;
    break ;
  case 13:
    func_batch_odd<13>(inDim, outDim, pGOut+n*outDim, pWeight, pGIn+n*inDim ) ;
    n+=13;
    break ;
  case 14:
    func_batch_even<14>(inDim, outDim, pGOut+n*outDim, pWeight, pGIn+n*inDim ) ;
    n+=14 ;
    break ;
  case 15:
    func_batch_odd<15>(inDim, outDim, pGOut+n*outDim, pWeight, pGIn+n*inDim ) ;
    n+=15;
    break ;
  default : break ;
  }
  for(; n<nBatch; n+=16) {
    func_batch_even<16>(inDim, outDim, pGOut+n*outDim, pWeight, pGIn+n*inDim ) ;
  }

  return VEDNN_SUCCESS ;
}
