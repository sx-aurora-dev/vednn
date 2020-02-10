#include <stdint.h>
#include <float.h>

#include <stdio.h>

#include "vednn.h"

#include "velintrin.h"
//#define VLEN	(192)

template <int BATCH>
static inline void func(
  const uint64_t	inDim,
  const uint64_t	outDim,
  const float * 	pIn,
  const float * 	pWeight,
  float * 		pOut,
  const uint64_t 	mvl
)
{
  for(int64_t o=0; o<outDim; o+=2*mvl) {
    const int64_t vl = (outDim-o < 2*mvl ? outDim - o : 2*mvl) >> 1 ;


    __vr vrsum[BATCH] ;
#pragma clang loop unroll(full)
    for(int64_t b=0; b<BATCH; b++) {
      vrsum[b] = _vel_pvbrd_vsl(0UL, vl) ;
    }

    int64_t i=0;
    if((inDim & 0x01)==1) {
	__vr vrw_i0 = _vel_vld_vssl(8, pWeight+(i  )*outDim+o, vl) ;
#pragma clang loop unroll(full)
	for(int64_t b=0; b<BATCH; b++) {
	  const uint64_t in = _vel_pack_f32a(pIn+b*inDim+i+0) ;
	  vrsum[b] = _vel_pvfmad_vvsvl(vrsum[b], in, vrw_i0, vl) ;
	}

	i+=1 ;
    }
    if(((inDim>>1) & 0x01)==1) {
	__vr vrw_i0 = _vel_vld_vssl(8, pWeight+(i  )*outDim+o, vl) ;
#pragma clang loop unroll(full)
	for(int64_t b=0; b<BATCH; b++) {
	  const uint64_t in = _vel_pack_f32a(pIn+b*inDim+i+0) ;
	  vrsum[b] = _vel_pvfmad_vvsvl(vrsum[b], in, vrw_i0, vl) ;
	}

	__vr vrw_i1 = _vel_vld_vssl(8, pWeight+(i+1)*outDim+o, vl) ;
#pragma clang loop unroll(full)
	for(int64_t b=0; b<BATCH; b++) {
	  const uint64_t in = _vel_pack_f32a(pIn+b*inDim+i+1) ;
	  vrsum[b] = _vel_pvfmad_vvsvl(vrsum[b], in, vrw_i1, vl) ;
	}
	i+=2 ;
    }
    if(((inDim>>2) & 0x01)==1) {
	__vr vrw_i0 = _vel_vld_vssl(8, pWeight+(i  )*outDim+o, vl) ;
#pragma clang loop unroll(full)
	for(int64_t b=0; b<BATCH; b++) {
	  const uint64_t in = _vel_pack_f32a(pIn+b*inDim+i+0) ;
	  vrsum[b] = _vel_pvfmad_vvsvl(vrsum[b], in, vrw_i0, vl) ;
	}

	__vr vrw_i1 = _vel_vld_vssl(8, pWeight+(i+1)*outDim+o, vl) ;
#pragma clang loop unroll(full)
	for(int64_t b=0; b<BATCH; b++) {
	  const uint64_t in = _vel_pack_f32a(pIn+b*inDim+i+1) ;
	  vrsum[b] = _vel_pvfmad_vvsvl(vrsum[b], in, vrw_i1, vl) ;
	}

	__vr vrw_i2 = _vel_vld_vssl(8, pWeight+(i+2)*outDim+o, vl) ;
#pragma clang loop unroll(full)
	for(int64_t b=0; b<BATCH; b++) {
	  const uint64_t in = _vel_pack_f32a(pIn+b*inDim+i+2) ;
	  vrsum[b] = _vel_pvfmad_vvsvl(vrsum[b], in, vrw_i2, vl) ;
	}

	__vr vrw_i3 = _vel_vld_vssl(8, pWeight+(i+3)*outDim+o, vl) ;
#pragma clang loop unroll(full)
	for(int64_t b=0; b<BATCH; b++) {
	  const uint64_t in = _vel_pack_f32a(pIn+b*inDim+i+3) ;
	  vrsum[b] = _vel_pvfmad_vvsvl(vrsum[b], in, vrw_i3, vl) ;
	}

      i+=4 ;
    }
    for(; i<inDim; i+=8 ) {
	__vr vrw_i0 = _vel_vld_vssl(8, pWeight+(i  )*outDim+o, vl) ;
#pragma clang loop unroll(full)
	for(int64_t b=0; b<BATCH; b++) {
	  const uint64_t in = _vel_pack_f32a(pIn+b*inDim+i+0) ;
	  vrsum[b] = _vel_pvfmad_vvsvl(vrsum[b], in, vrw_i0, vl) ;
	}

	__vr vrw_i1 = _vel_vld_vssl(8, pWeight+(i+1)*outDim+o, vl) ;
#pragma clang loop unroll(full)
	for(int64_t b=0; b<BATCH; b++) {
	  const uint64_t in = _vel_pack_f32a(pIn+b*inDim+i+1) ;
	  vrsum[b] = _vel_pvfmad_vvsvl(vrsum[b], in, vrw_i1, vl) ;
	}

	__vr vrw_i2 = _vel_vld_vssl(8, pWeight+(i+2)*outDim+o, vl) ;
#pragma clang loop unroll(full)
	for(int64_t b=0; b<BATCH; b++) {
	  const uint64_t in = _vel_pack_f32a(pIn+b*inDim+i+2) ;
	  vrsum[b] = _vel_pvfmad_vvsvl(vrsum[b], in, vrw_i2, vl) ;
	}

	__vr vrw_i3 = _vel_vld_vssl(8, pWeight+(i+3)*outDim+o, vl) ;
#pragma clang loop unroll(full)
	for(int64_t b=0; b<BATCH; b++) {
	  const uint64_t in = _vel_pack_f32a(pIn+b*inDim+i+3) ;
	  vrsum[b] = _vel_pvfmad_vvsvl(vrsum[b], in, vrw_i3, vl) ;
	}

	__vr vrw_i4 = _vel_vld_vssl(8, pWeight+(i+4)*outDim+o, vl) ;
#pragma clang loop unroll(full)
	for(int64_t b=0; b<BATCH; b++) {
	  const uint64_t in = _vel_pack_f32a(pIn+b*inDim+i+4) ;
	  vrsum[b] = _vel_pvfmad_vvsvl(vrsum[b], in, vrw_i4, vl) ;
	}

	__vr vrw_i5 = _vel_vld_vssl(8, pWeight+(i+5)*outDim+o, vl) ;
#pragma clang loop unroll(full)
	for(int64_t b=0; b<BATCH; b++) {
	  const uint64_t in = _vel_pack_f32a(pIn+b*inDim+i+5) ;
	  vrsum[b] = _vel_pvfmad_vvsvl(vrsum[b], in, vrw_i5, vl) ;
	}

	__vr vrw_i6 = _vel_vld_vssl(8, pWeight+(i+6)*outDim+o, vl) ;
#pragma clang loop unroll(full)
	for(int64_t b=0; b<BATCH; b++) {
	  const uint64_t in = _vel_pack_f32a(pIn+b*inDim+i+6) ;
	  vrsum[b] = _vel_pvfmad_vvsvl(vrsum[b], in, vrw_i6, vl) ;
	}

	__vr vrw_i7 = _vel_vld_vssl(8, pWeight+(i+7)*outDim+o, vl) ;
#pragma clang loop unroll(full)
	for(int64_t b=0; b<BATCH; b++) {
	  const uint64_t in = _vel_pack_f32a(pIn+b*inDim+i+7) ;
	  vrsum[b] = _vel_pvfmad_vvsvl(vrsum[b], in, vrw_i7, vl) ;
	}
    }

#pragma clang loop unroll(full)
    for(int64_t b=0; b<BATCH; b++) {
      _vel_vst_vssl(vrsum[b], 8, pOut+b*outDim+o, vl) ;
    }
  }
}
extern "C"
vednnError_t vednnLinearForward_o2X_woaligned(
    const uint64_t			inDim,
    const uint64_t			outDim,
    const uint64_t			nBatch,
    const void * __restrict__		pDataIn,
    const void * __restrict__		pDataWeight,
    void * __restrict__			pDataOut
)
{

  const float * __restrict__ pIn     = (const float * __restrict__) pDataIn;
  const float * __restrict__ pWeight = (const float * __restrict__) pDataWeight;
  float * __restrict__ const pOut    = (float * __restrict__ const) pDataOut;

  int64_t n=0;
  int64_t batchRemain = nBatch & 0x07 ;

  int64_t mvl ;
  if( outDim % (256*2) == 0 )
    mvl = 256 ;
  else if ( outDim % (192*2) == 0 )
    mvl = 192 ;
  else if( outDim % (256*2) < outDim % (192*2) )
    mvl = 192 ;
  else
    mvl = 256 ;

  switch( batchRemain ) {
  case 1 :
    func<1>(inDim, outDim, pIn+n*inDim, pWeight, pOut+n*outDim, mvl) ;
    n+=1 ;
    break ;
  case 2 :
    func<2>(inDim, outDim, pIn+n*inDim, pWeight, pOut+n*outDim, mvl) ;
    n+=2 ;
    break ;
  case 3 :
    func<3>(inDim, outDim, pIn+n*inDim, pWeight, pOut+n*outDim, mvl) ;
    n+=3 ;
    break ;
  case 4 :
    func<4>(inDim, outDim, pIn+n*inDim, pWeight, pOut+n*outDim, mvl) ;
    n+=4 ;
    break ;
  case 5 :
    func<5>(inDim, outDim, pIn+n*inDim, pWeight, pOut+n*outDim, mvl) ;
    n+=5 ;
    break ;
  case 6 :
    func<6>(inDim, outDim, pIn+n*inDim, pWeight, pOut+n*outDim, mvl) ;
    n+=6 ;
    break ;
  case 7 :
    func<7>(inDim, outDim, pIn+n*inDim, pWeight, pOut+n*outDim, mvl) ;
    n+=7 ;
    break ;
  default :
    break ;
  }
  for(; n<nBatch; n+=8) {
    func<8>(inDim, outDim, pIn+n*inDim, pWeight, pOut+n*outDim, mvl) ;
  }

  return VEDNN_SUCCESS ;
}
