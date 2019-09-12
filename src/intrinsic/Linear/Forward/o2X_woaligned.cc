#include <stdint.h>
#include <float.h>

#include <stdio.h>

#include "vednn.h"

#include "velintrin.h"
#define VLEN	(256)

template <int BATCH>
static inline void func(
  const uint64_t	inDim,
  const uint64_t	outDim,
  const float * 	pIn,
  const float * 	pWeight,
  float * 		pOut
)
{
  for(int64_t o=0; o<outDim; o+=2*VLEN) {
    const int64_t vl = (outDim-o < 2*VLEN ? outDim - o : 2*VLEN) >> 1 ;

    __vr vrsum_b0 = _vel_pvbrd_vsl(0UL, vl) ;
    __vr vrsum_b1 = _vel_pvbrd_vsl(0UL, vl) ;
    __vr vrsum_b2 = _vel_pvbrd_vsl(0UL, vl) ;
    __vr vrsum_b3 = _vel_pvbrd_vsl(0UL, vl) ;

    int64_t i=0;
    if((inDim & 0x01)==1) {
	__vr vrw_i0 = _vel_vld_vssl(8, pWeight+(i  )*outDim+o, vl) ;
	const uint64_t i_b0_0 = _vel_pack_f32a(pIn+0*inDim+i) ;
	const uint64_t i_b1_0 = _vel_pack_f32a(pIn+1*inDim+i) ;
	const uint64_t i_b2_0 = _vel_pack_f32a(pIn+2*inDim+i) ;
	const uint64_t i_b3_0 = _vel_pack_f32a(pIn+3*inDim+i) ;
	if(BATCH >=1) vrsum_b0 = _vel_pvfmad_vvsvl(vrsum_b0, i_b0_0, vrw_i0, vl) ;
	if(BATCH >=2) vrsum_b1 = _vel_pvfmad_vvsvl(vrsum_b1, i_b1_0, vrw_i0, vl) ;
	if(BATCH >=3) vrsum_b2 = _vel_pvfmad_vvsvl(vrsum_b2, i_b2_0, vrw_i0, vl) ;
	if(BATCH >=4) vrsum_b3 = _vel_pvfmad_vvsvl(vrsum_b3, i_b3_0, vrw_i0, vl) ;

	i+=1 ;
    }
    if(((inDim>>1) & 0x01)==1) {
	__vr vrw_i0 = _vel_vld_vssl(8, pWeight+(i  )*outDim+o, vl) ;
	const uint64_t i_b0_0 = _vel_pack_f32a(pIn+0*inDim+i) ;
	const uint64_t i_b1_0 = _vel_pack_f32a(pIn+1*inDim+i) ;
	const uint64_t i_b2_0 = _vel_pack_f32a(pIn+2*inDim+i) ;
	const uint64_t i_b3_0 = _vel_pack_f32a(pIn+3*inDim+i) ;
	if(BATCH >=1) vrsum_b0 = _vel_pvfmad_vvsvl(vrsum_b0, i_b0_0, vrw_i0, vl) ;
	if(BATCH >=2) vrsum_b1 = _vel_pvfmad_vvsvl(vrsum_b1, i_b1_0, vrw_i0, vl) ;
	if(BATCH >=3) vrsum_b2 = _vel_pvfmad_vvsvl(vrsum_b2, i_b2_0, vrw_i0, vl) ;
	if(BATCH >=4) vrsum_b3 = _vel_pvfmad_vvsvl(vrsum_b3, i_b3_0, vrw_i0, vl) ;

	__vr vrw_i1 = _vel_vld_vssl(8, pWeight+(i+1)*outDim+o, vl) ;
	const uint64_t i_b0_1 = _vel_pack_f32a(pIn+0*inDim+i+1) ;
	const uint64_t i_b1_1 = _vel_pack_f32a(pIn+1*inDim+i+1) ;
	const uint64_t i_b2_1 = _vel_pack_f32a(pIn+2*inDim+i+1) ;
	const uint64_t i_b3_1 = _vel_pack_f32a(pIn+3*inDim+i+1) ;
	if(BATCH >=1) vrsum_b0 = _vel_pvfmad_vvsvl(vrsum_b0, i_b0_1, vrw_i1, vl) ;
	if(BATCH >=2) vrsum_b1 = _vel_pvfmad_vvsvl(vrsum_b1, i_b1_1, vrw_i1, vl) ;
	if(BATCH >=3) vrsum_b2 = _vel_pvfmad_vvsvl(vrsum_b2, i_b2_1, vrw_i1, vl) ;
	if(BATCH >=4) vrsum_b3 = _vel_pvfmad_vvsvl(vrsum_b3, i_b3_1, vrw_i1, vl) ;

	i+=2 ;
    }
    if(((inDim>>2) & 0x01)==1) {
	__vr vrw_i0 = _vel_vld_vssl(8, pWeight+(i  )*outDim+o, vl) ;
	const uint64_t i_b0_0 = _vel_pack_f32a(pIn+0*inDim+i) ;
	const uint64_t i_b1_0 = _vel_pack_f32a(pIn+1*inDim+i) ;
	const uint64_t i_b2_0 = _vel_pack_f32a(pIn+2*inDim+i) ;
	const uint64_t i_b3_0 = _vel_pack_f32a(pIn+3*inDim+i) ;
	if(BATCH >=1) vrsum_b0 = _vel_pvfmad_vvsvl(vrsum_b0, i_b0_0, vrw_i0, vl) ;
	if(BATCH >=2) vrsum_b1 = _vel_pvfmad_vvsvl(vrsum_b1, i_b1_0, vrw_i0, vl) ;
	if(BATCH >=3) vrsum_b2 = _vel_pvfmad_vvsvl(vrsum_b2, i_b2_0, vrw_i0, vl) ;
	if(BATCH >=4) vrsum_b3 = _vel_pvfmad_vvsvl(vrsum_b3, i_b3_0, vrw_i0, vl) ;

	__vr vrw_i1 = _vel_vld_vssl(8, pWeight+(i+1)*outDim+o, vl) ;
	const uint64_t i_b0_1 = _vel_pack_f32a(pIn+0*inDim+i+1) ;
	const uint64_t i_b1_1 = _vel_pack_f32a(pIn+1*inDim+i+1) ;
	const uint64_t i_b2_1 = _vel_pack_f32a(pIn+2*inDim+i+1) ;
	const uint64_t i_b3_1 = _vel_pack_f32a(pIn+3*inDim+i+1) ;
	if(BATCH >=1) vrsum_b0 = _vel_pvfmad_vvsvl(vrsum_b0, i_b0_1, vrw_i1, vl) ;
	if(BATCH >=2) vrsum_b1 = _vel_pvfmad_vvsvl(vrsum_b1, i_b1_1, vrw_i1, vl) ;
	if(BATCH >=3) vrsum_b2 = _vel_pvfmad_vvsvl(vrsum_b2, i_b2_1, vrw_i1, vl) ;
	if(BATCH >=4) vrsum_b3 = _vel_pvfmad_vvsvl(vrsum_b3, i_b3_1, vrw_i1, vl) ;

	__vr vrw_i2 = _vel_vld_vssl(8, pWeight+(i+2)*outDim+o, vl) ;
	const uint64_t i_b0_2 = _vel_pack_f32a(pIn+0*inDim+i+2) ;
	const uint64_t i_b1_2 = _vel_pack_f32a(pIn+1*inDim+i+2) ;
	const uint64_t i_b2_2 = _vel_pack_f32a(pIn+2*inDim+i+2) ;
	const uint64_t i_b3_2 = _vel_pack_f32a(pIn+3*inDim+i+2) ;
	if(BATCH >=1) vrsum_b0 = _vel_pvfmad_vvsvl(vrsum_b0, i_b0_2, vrw_i2, vl) ;
	if(BATCH >=2) vrsum_b1 = _vel_pvfmad_vvsvl(vrsum_b1, i_b1_2, vrw_i2, vl) ;
	if(BATCH >=3) vrsum_b2 = _vel_pvfmad_vvsvl(vrsum_b2, i_b2_2, vrw_i2, vl) ;
	if(BATCH >=4) vrsum_b3 = _vel_pvfmad_vvsvl(vrsum_b3, i_b3_2, vrw_i2, vl) ;

	__vr vrw_i3 = _vel_vld_vssl(8, pWeight+(i+3)*outDim+o, vl) ;
	const uint64_t i_b0_3 = _vel_pack_f32a(pIn+0*inDim+i+3) ;
	const uint64_t i_b1_3 = _vel_pack_f32a(pIn+1*inDim+i+3) ;
	const uint64_t i_b2_3 = _vel_pack_f32a(pIn+2*inDim+i+3) ;
	const uint64_t i_b3_3 = _vel_pack_f32a(pIn+3*inDim+i+3) ;
	if(BATCH >=1) vrsum_b0 = _vel_pvfmad_vvsvl(vrsum_b0, i_b0_3, vrw_i3, vl) ;
	if(BATCH >=2) vrsum_b1 = _vel_pvfmad_vvsvl(vrsum_b1, i_b1_3, vrw_i3, vl) ;
	if(BATCH >=3) vrsum_b2 = _vel_pvfmad_vvsvl(vrsum_b2, i_b2_3, vrw_i3, vl) ;
	if(BATCH >=4) vrsum_b3 = _vel_pvfmad_vvsvl(vrsum_b3, i_b3_3, vrw_i3, vl) ;

      i+=4 ;
    }
    for(; i<inDim; i+=8 ) {
	__vr vrw_i0 = _vel_vld_vssl(8, pWeight+(i  )*outDim+o, vl) ;
	const uint64_t i_b0_0 = _vel_pack_f32a(pIn+0*inDim+i) ;
	const uint64_t i_b1_0 = _vel_pack_f32a(pIn+1*inDim+i) ;
	const uint64_t i_b2_0 = _vel_pack_f32a(pIn+2*inDim+i) ;
	const uint64_t i_b3_0 = _vel_pack_f32a(pIn+3*inDim+i) ;
	if(BATCH >=1) vrsum_b0 = _vel_pvfmad_vvsvl(vrsum_b0, i_b0_0, vrw_i0, vl) ;
	if(BATCH >=2) vrsum_b1 = _vel_pvfmad_vvsvl(vrsum_b1, i_b1_0, vrw_i0, vl) ;
	if(BATCH >=3) vrsum_b2 = _vel_pvfmad_vvsvl(vrsum_b2, i_b2_0, vrw_i0, vl) ;
	if(BATCH >=4) vrsum_b3 = _vel_pvfmad_vvsvl(vrsum_b3, i_b3_0, vrw_i0, vl) ;

	__vr vrw_i1 = _vel_vld_vssl(8, pWeight+(i+1)*outDim+o, vl) ;
	const uint64_t i_b0_1 = _vel_pack_f32a(pIn+0*inDim+i+1) ;
	const uint64_t i_b1_1 = _vel_pack_f32a(pIn+1*inDim+i+1) ;
	const uint64_t i_b2_1 = _vel_pack_f32a(pIn+2*inDim+i+1) ;
	const uint64_t i_b3_1 = _vel_pack_f32a(pIn+3*inDim+i+1) ;
	if(BATCH >=1) vrsum_b0 = _vel_pvfmad_vvsvl(vrsum_b0, i_b0_1, vrw_i1, vl) ;
	if(BATCH >=2) vrsum_b1 = _vel_pvfmad_vvsvl(vrsum_b1, i_b1_1, vrw_i1, vl) ;
	if(BATCH >=3) vrsum_b2 = _vel_pvfmad_vvsvl(vrsum_b2, i_b2_1, vrw_i1, vl) ;
	if(BATCH >=4) vrsum_b3 = _vel_pvfmad_vvsvl(vrsum_b3, i_b3_1, vrw_i1, vl) ;

	__vr vrw_i2 = _vel_vld_vssl(8, pWeight+(i+2)*outDim+o, vl) ;
	const uint64_t i_b0_2 = _vel_pack_f32a(pIn+0*inDim+i+2) ;
	const uint64_t i_b1_2 = _vel_pack_f32a(pIn+1*inDim+i+2) ;
	const uint64_t i_b2_2 = _vel_pack_f32a(pIn+2*inDim+i+2) ;
	const uint64_t i_b3_2 = _vel_pack_f32a(pIn+3*inDim+i+2) ;
	if(BATCH >=1) vrsum_b0 = _vel_pvfmad_vvsvl(vrsum_b0, i_b0_2, vrw_i2, vl) ;
	if(BATCH >=2) vrsum_b1 = _vel_pvfmad_vvsvl(vrsum_b1, i_b1_2, vrw_i2, vl) ;
	if(BATCH >=3) vrsum_b2 = _vel_pvfmad_vvsvl(vrsum_b2, i_b2_2, vrw_i2, vl) ;
	if(BATCH >=4) vrsum_b3 = _vel_pvfmad_vvsvl(vrsum_b3, i_b3_2, vrw_i2, vl) ;

	__vr vrw_i3 = _vel_vld_vssl(8, pWeight+(i+3)*outDim+o, vl) ;
	const uint64_t i_b0_3 = _vel_pack_f32a(pIn+0*inDim+i+3) ;
	const uint64_t i_b1_3 = _vel_pack_f32a(pIn+1*inDim+i+3) ;
	const uint64_t i_b2_3 = _vel_pack_f32a(pIn+2*inDim+i+3) ;
	const uint64_t i_b3_3 = _vel_pack_f32a(pIn+3*inDim+i+3) ;
	if(BATCH >=1) vrsum_b0 = _vel_pvfmad_vvsvl(vrsum_b0, i_b0_3, vrw_i3, vl) ;
	if(BATCH >=2) vrsum_b1 = _vel_pvfmad_vvsvl(vrsum_b1, i_b1_3, vrw_i3, vl) ;
	if(BATCH >=3) vrsum_b2 = _vel_pvfmad_vvsvl(vrsum_b2, i_b2_3, vrw_i3, vl) ;
	if(BATCH >=4) vrsum_b3 = _vel_pvfmad_vvsvl(vrsum_b3, i_b3_3, vrw_i3, vl) ;

	__vr vrw_i4 = _vel_vld_vssl(8, pWeight+(i+4)*outDim+o, vl) ;
	const uint64_t i_b0_4 = _vel_pack_f32a(pIn+0*inDim+i+4) ;
	const uint64_t i_b1_4 = _vel_pack_f32a(pIn+1*inDim+i+4) ;
	const uint64_t i_b2_4 = _vel_pack_f32a(pIn+2*inDim+i+4) ;
	const uint64_t i_b3_4 = _vel_pack_f32a(pIn+3*inDim+i+4) ;
	if(BATCH >=1) vrsum_b0 = _vel_pvfmad_vvsvl(vrsum_b0, i_b0_4, vrw_i4, vl) ;
	if(BATCH >=2) vrsum_b1 = _vel_pvfmad_vvsvl(vrsum_b1, i_b1_4, vrw_i4, vl) ;
	if(BATCH >=3) vrsum_b2 = _vel_pvfmad_vvsvl(vrsum_b2, i_b2_4, vrw_i4, vl) ;
	if(BATCH >=4) vrsum_b3 = _vel_pvfmad_vvsvl(vrsum_b3, i_b3_4, vrw_i4, vl) ;

	__vr vrw_i5 = _vel_vld_vssl(8, pWeight+(i+5)*outDim+o, vl) ;
	const uint64_t i_b0_5 = _vel_pack_f32a(pIn+0*inDim+i+5) ;
	const uint64_t i_b1_5 = _vel_pack_f32a(pIn+1*inDim+i+5) ;
	const uint64_t i_b2_5 = _vel_pack_f32a(pIn+2*inDim+i+5) ;
	const uint64_t i_b3_5 = _vel_pack_f32a(pIn+3*inDim+i+5) ;
	if(BATCH >=1) vrsum_b0 = _vel_pvfmad_vvsvl(vrsum_b0, i_b0_5, vrw_i5, vl) ;
	if(BATCH >=2) vrsum_b1 = _vel_pvfmad_vvsvl(vrsum_b1, i_b1_5, vrw_i5, vl) ;
	if(BATCH >=3) vrsum_b2 = _vel_pvfmad_vvsvl(vrsum_b2, i_b2_5, vrw_i5, vl) ;
	if(BATCH >=4) vrsum_b3 = _vel_pvfmad_vvsvl(vrsum_b3, i_b3_5, vrw_i5, vl) ;

	__vr vrw_i6 = _vel_vld_vssl(8, pWeight+(i+6)*outDim+o, vl) ;
	const uint64_t i_b0_6 = _vel_pack_f32a(pIn+0*inDim+i+6) ;
	const uint64_t i_b1_6 = _vel_pack_f32a(pIn+1*inDim+i+6) ;
	const uint64_t i_b2_6 = _vel_pack_f32a(pIn+2*inDim+i+6) ;
	const uint64_t i_b3_6 = _vel_pack_f32a(pIn+3*inDim+i+6) ;
	if(BATCH >=1) vrsum_b0 = _vel_pvfmad_vvsvl(vrsum_b0, i_b0_6, vrw_i6, vl) ;
	if(BATCH >=2) vrsum_b1 = _vel_pvfmad_vvsvl(vrsum_b1, i_b1_6, vrw_i6, vl) ;
	if(BATCH >=3) vrsum_b2 = _vel_pvfmad_vvsvl(vrsum_b2, i_b2_6, vrw_i6, vl) ;
	if(BATCH >=4) vrsum_b3 = _vel_pvfmad_vvsvl(vrsum_b3, i_b3_6, vrw_i6, vl) ;

	__vr vrw_i7 = _vel_vld_vssl(8, pWeight+(i+7)*outDim+o, vl) ;
	const uint64_t i_b0_7 = _vel_pack_f32a(pIn+0*inDim+i+7) ;
	const uint64_t i_b1_7 = _vel_pack_f32a(pIn+1*inDim+i+7) ;
	const uint64_t i_b2_7 = _vel_pack_f32a(pIn+2*inDim+i+7) ;
	const uint64_t i_b3_7 = _vel_pack_f32a(pIn+3*inDim+i+7) ;
	if(BATCH >=1) vrsum_b0 = _vel_pvfmad_vvsvl(vrsum_b0, i_b0_7, vrw_i7, vl) ;
	if(BATCH >=2) vrsum_b1 = _vel_pvfmad_vvsvl(vrsum_b1, i_b1_7, vrw_i7, vl) ;
	if(BATCH >=3) vrsum_b2 = _vel_pvfmad_vvsvl(vrsum_b2, i_b2_7, vrw_i7, vl) ;
	if(BATCH >=4) vrsum_b3 = _vel_pvfmad_vvsvl(vrsum_b3, i_b3_7, vrw_i7, vl) ;
    }

    if(BATCH >=1) _vel_vst_vssl(vrsum_b0, 8, pOut+0*outDim+o, vl) ;
    if(BATCH >=2) _vel_vst_vssl(vrsum_b1, 8, pOut+1*outDim+o, vl) ;
    if(BATCH >=3) _vel_vst_vssl(vrsum_b2, 8, pOut+2*outDim+o, vl) ;
    if(BATCH >=4) _vel_vst_vssl(vrsum_b3, 8, pOut+3*outDim+o, vl) ;
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
  int64_t batchRemain = nBatch & 0x03 ;

  switch( batchRemain ) {
  case 1 :
    func<1>(inDim, outDim, pIn+n*inDim, pWeight, pOut+n*outDim) ;
    n+=1 ;
    break ;
  case 2 :
    func<2>(inDim, outDim, pIn+n*inDim, pWeight, pOut+n*outDim) ;
    n+=2 ;
    break ;
  case 3 :
    func<3>(inDim, outDim, pIn+n*inDim, pWeight, pOut+n*outDim) ;
    n+=3 ;
    break ;
  default :
    break ;
  }
  for(; n<nBatch; n+=4) {
    func<4>(inDim, outDim, pIn+n*inDim, pWeight, pOut+n*outDim) ;
  }

  return VEDNN_SUCCESS ;
}
