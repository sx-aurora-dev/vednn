#include <stdint.h>
#include <float.h>

#include <stdio.h>

#include "vednn.h"

#include "velintrin.h"
#define VLEN	(256)

vednnError_t vednnLinearForward_o2X_woaligned(
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
  int64_t batchRemain = nBatch & 0x03 ;

  switch( batchRemain ) {
  case 1 :
    for(int64_t o=0; o<outDim; o+=2*VLEN) {
      const int64_t vl = (outDim-o < 2*VLEN ? outDim - o : 2*VLEN) >> 1 ;

      __vr vrsum_b0 = _vel_pvbrd_vsl(0UL, vl) ;

      int64_t i=0;
      if((inDim & 0x01)==1) {
	__vr vrw = _vel_vld_vssl(8, pWeight+i*outDim+o, vl) ;

	const uint64_t i_b0_0 = _vel_pack_f32a(pIn+(n  )*inDim+i) ;
	vrsum_b0 = _vel_pvfmad_vvsvl(vrsum_b0, i_b0_0, vrw, vl) ;
	i+=1 ;
      }
      if(((inDim>>1) & 0x01)==1) {
	__vr vrw_i0 = _vel_vld_vssl(8, pWeight+(i  )*outDim+o, vl) ;
	__vr vrw_i1 = _vel_vld_vssl(8, pWeight+(i+1)*outDim+o, vl) ;

	const uint64_t i_b0_0 = _vel_pack_f32a(pIn+(n  )*inDim+i) ;
	const uint64_t i_b0_1 = _vel_pack_f32a(pIn+(n  )*inDim+i+1) ;

	vrsum_b0 = _vel_pvfmad_vvsvl(vrsum_b0, i_b0_0, vrw_i0, vl) ;
	vrsum_b0 = _vel_pvfmad_vvsvl(vrsum_b0, i_b0_1, vrw_i1, vl) ;

	i+=2 ;
      }
      for(; i<inDim; i+=4 ) {
	__vr vrw_i0 = _vel_vld_vssl(8, pWeight+(i  )*outDim+o, vl) ;
	__vr vrw_i1 = _vel_vld_vssl(8, pWeight+(i+1)*outDim+o, vl) ;
	__vr vrw_i2 = _vel_vld_vssl(8, pWeight+(i+2)*outDim+o, vl) ;
	__vr vrw_i3 = _vel_vld_vssl(8, pWeight+(i+3)*outDim+o, vl) ;

	const uint64_t i_b0_0 = _vel_pack_f32a(pIn+(n  )*inDim+i) ;
	const uint64_t i_b0_1 = _vel_pack_f32a(pIn+(n  )*inDim+i+1) ;
	const uint64_t i_b0_2 = _vel_pack_f32a(pIn+(n  )*inDim+i+2) ;
	const uint64_t i_b0_3 = _vel_pack_f32a(pIn+(n  )*inDim+i+3) ;

	vrsum_b0 = _vel_pvfmad_vvsvl(vrsum_b0, i_b0_0, vrw_i0, vl) ;
	vrsum_b0 = _vel_pvfmad_vvsvl(vrsum_b0, i_b0_1, vrw_i1, vl) ;
	vrsum_b0 = _vel_pvfmad_vvsvl(vrsum_b0, i_b0_2, vrw_i2, vl) ;
	vrsum_b0 = _vel_pvfmad_vvsvl(vrsum_b0, i_b0_3, vrw_i3, vl) ;
      }

      _vel_vst_vssl(vrsum_b0, 8, pOut+(n  )*outDim+o, vl) ;
    }

    n+=1 ;
    break ;
  case 2 :
    for(int64_t o=0; o<outDim; o+=2*VLEN) {
      const int64_t vl = (outDim-o < 2*VLEN ? outDim - o : 2*VLEN) >> 1 ;

      __vr vrsum_b0 = _vel_pvbrd_vsl(0UL, vl) ;
      __vr vrsum_b1 = _vel_pvbrd_vsl(0UL, vl) ;

      int64_t i=0;
      if((inDim & 0x01)==1) {
	__vr vrw = _vel_vld_vssl(8, pWeight+i*outDim+o, vl) ;

	const uint64_t i_b0_0 = _vel_pack_f32a(pIn+(n  )*inDim+i) ;
	const uint64_t i_b1_0 = _vel_pack_f32a(pIn+(n+1)*inDim+i) ;

	vrsum_b0 = _vel_pvfmad_vvsvl(vrsum_b0, i_b0_0, vrw, vl) ;
	vrsum_b1 = _vel_pvfmad_vvsvl(vrsum_b1, i_b1_0, vrw, vl) ;

	i+=1 ;
      }
      if(((inDim>>1) & 0x01)==1) {
	__vr vrw_i0 = _vel_vld_vssl(8, pWeight+(i  )*outDim+o, vl) ;
	__vr vrw_i1 = _vel_vld_vssl(8, pWeight+(i+1)*outDim+o, vl) ;

	const uint64_t i_b0_0 = _vel_pack_f32a(pIn+(n  )*inDim+i) ;
	const uint64_t i_b0_1 = _vel_pack_f32a(pIn+(n  )*inDim+i+1) ;

	const uint64_t i_b1_0 = _vel_pack_f32a(pIn+(n+1)*inDim+i) ;
	const uint64_t i_b1_1 = _vel_pack_f32a(pIn+(n+1)*inDim+i+1) ;

	vrsum_b0 = _vel_pvfmad_vvsvl(vrsum_b0, i_b0_0, vrw_i0, vl) ;
	vrsum_b1 = _vel_pvfmad_vvsvl(vrsum_b1, i_b1_0, vrw_i0, vl) ;

	vrsum_b0 = _vel_pvfmad_vvsvl(vrsum_b0, i_b0_1, vrw_i1, vl) ;
	vrsum_b1 = _vel_pvfmad_vvsvl(vrsum_b1, i_b1_1, vrw_i1, vl) ;

	i+=2 ;
      }
      for(; i<inDim; i+=4 ) {
	__vr vrw_i0 = _vel_vld_vssl(8, pWeight+(i  )*outDim+o, vl) ;
	__vr vrw_i1 = _vel_vld_vssl(8, pWeight+(i+1)*outDim+o, vl) ;
	__vr vrw_i2 = _vel_vld_vssl(8, pWeight+(i+2)*outDim+o, vl) ;
	__vr vrw_i3 = _vel_vld_vssl(8, pWeight+(i+3)*outDim+o, vl) ;

	const uint64_t i_b0_0 = _vel_pack_f32a(pIn+(n  )*inDim+i) ;
	const uint64_t i_b0_1 = _vel_pack_f32a(pIn+(n  )*inDim+i+1) ;
	const uint64_t i_b0_2 = _vel_pack_f32a(pIn+(n  )*inDim+i+2) ;
	const uint64_t i_b0_3 = _vel_pack_f32a(pIn+(n  )*inDim+i+3) ;

	const uint64_t i_b1_0 = _vel_pack_f32a(pIn+(n+1)*inDim+i) ;
	const uint64_t i_b1_1 = _vel_pack_f32a(pIn+(n+1)*inDim+i+1) ;
	const uint64_t i_b1_2 = _vel_pack_f32a(pIn+(n+1)*inDim+i+2) ;
	const uint64_t i_b1_3 = _vel_pack_f32a(pIn+(n+1)*inDim+i+3) ;

	vrsum_b0 = _vel_pvfmad_vvsvl(vrsum_b0, i_b0_0, vrw_i0, vl) ;
	vrsum_b1 = _vel_pvfmad_vvsvl(vrsum_b1, i_b1_0, vrw_i0, vl) ;

	vrsum_b0 = _vel_pvfmad_vvsvl(vrsum_b0, i_b0_1, vrw_i1, vl) ;
	vrsum_b1 = _vel_pvfmad_vvsvl(vrsum_b1, i_b1_1, vrw_i1, vl) ;

	vrsum_b0 = _vel_pvfmad_vvsvl(vrsum_b0, i_b0_2, vrw_i2, vl) ;
	vrsum_b1 = _vel_pvfmad_vvsvl(vrsum_b1, i_b1_2, vrw_i2, vl) ;

	vrsum_b0 = _vel_pvfmad_vvsvl(vrsum_b0, i_b0_3, vrw_i3, vl) ;
	vrsum_b1 = _vel_pvfmad_vvsvl(vrsum_b1, i_b1_3, vrw_i3, vl) ;
      }

      _vel_vst_vssl(vrsum_b0, 8, pOut+(n  )*outDim+o, vl) ;
      _vel_vst_vssl(vrsum_b1, 8, pOut+(n+1)*outDim+o, vl) ;
    }
    n+=2 ;
    break ;
  case 3 :
    for(int64_t o=0; o<outDim; o+=2*VLEN) {
      const int64_t vl = (outDim-o < 2*VLEN ? outDim - o : 2*VLEN) >> 1 ;

      __vr vrsum_b0 = _vel_pvbrd_vsl(0UL, vl) ;
      __vr vrsum_b1 = _vel_pvbrd_vsl(0UL, vl) ;
      __vr vrsum_b2 = _vel_pvbrd_vsl(0UL, vl) ;

      int64_t i=0;
      if((inDim & 0x01)==1) {
	__vr vrw = _vel_vld_vssl(8, pWeight+i*outDim+o, vl) ;

	const uint64_t i_b0_0 = _vel_pack_f32a(pIn+(n  )*inDim+i) ;
	const uint64_t i_b1_0 = _vel_pack_f32a(pIn+(n+1)*inDim+i) ;
	const uint64_t i_b2_0 = _vel_pack_f32a(pIn+(n+2)*inDim+i) ;

	vrsum_b0 = _vel_pvfmad_vvsvl(vrsum_b0, i_b0_0, vrw, vl) ;
	vrsum_b1 = _vel_pvfmad_vvsvl(vrsum_b1, i_b1_0, vrw, vl) ;
	vrsum_b2 = _vel_pvfmad_vvsvl(vrsum_b2, i_b2_0, vrw, vl) ;

	i+=1 ;
      }
      if(((inDim>>1) & 0x01)==1) {
	__vr vrw_i0 = _vel_vld_vssl(8, pWeight+(i  )*outDim+o, vl) ;
	__vr vrw_i1 = _vel_vld_vssl(8, pWeight+(i+1)*outDim+o, vl) ;

	const uint64_t i_b0_0 = _vel_pack_f32a(pIn+(n  )*inDim+i) ;
	const uint64_t i_b0_1 = _vel_pack_f32a(pIn+(n  )*inDim+i+1) ;

	const uint64_t i_b1_0 = _vel_pack_f32a(pIn+(n+1)*inDim+i) ;
	const uint64_t i_b1_1 = _vel_pack_f32a(pIn+(n+1)*inDim+i+1) ;

	const uint64_t i_b2_0 = _vel_pack_f32a(pIn+(n+2)*inDim+i) ;
	const uint64_t i_b2_1 = _vel_pack_f32a(pIn+(n+2)*inDim+i+1) ;

	vrsum_b0 = _vel_pvfmad_vvsvl(vrsum_b0, i_b0_0, vrw_i0, vl) ;
	vrsum_b1 = _vel_pvfmad_vvsvl(vrsum_b1, i_b1_0, vrw_i0, vl) ;
	vrsum_b2 = _vel_pvfmad_vvsvl(vrsum_b2, i_b2_0, vrw_i0, vl) ;

	vrsum_b0 = _vel_pvfmad_vvsvl(vrsum_b0, i_b0_1, vrw_i1, vl) ;
	vrsum_b1 = _vel_pvfmad_vvsvl(vrsum_b1, i_b1_1, vrw_i1, vl) ;
	vrsum_b2 = _vel_pvfmad_vvsvl(vrsum_b2, i_b2_1, vrw_i1, vl) ;

	i+=2 ;
      }
      for(; i<inDim; i+=4 ) {
	__vr vrw_i0 = _vel_vld_vssl(8, pWeight+(i  )*outDim+o, vl) ;
	__vr vrw_i1 = _vel_vld_vssl(8, pWeight+(i+1)*outDim+o, vl) ;
	__vr vrw_i2 = _vel_vld_vssl(8, pWeight+(i+2)*outDim+o, vl) ;
	__vr vrw_i3 = _vel_vld_vssl(8, pWeight+(i+3)*outDim+o, vl) ;

	const uint64_t i_b0_0 = _vel_pack_f32a(pIn+(n  )*inDim+i) ;
	const uint64_t i_b0_1 = _vel_pack_f32a(pIn+(n  )*inDim+i+1) ;
	const uint64_t i_b0_2 = _vel_pack_f32a(pIn+(n  )*inDim+i+2) ;
	const uint64_t i_b0_3 = _vel_pack_f32a(pIn+(n  )*inDim+i+3) ;

	const uint64_t i_b1_0 = _vel_pack_f32a(pIn+(n+1)*inDim+i) ;
	const uint64_t i_b1_1 = _vel_pack_f32a(pIn+(n+1)*inDim+i+1) ;
	const uint64_t i_b1_2 = _vel_pack_f32a(pIn+(n+1)*inDim+i+2) ;
	const uint64_t i_b1_3 = _vel_pack_f32a(pIn+(n+1)*inDim+i+3) ;

	const uint64_t i_b2_0 = _vel_pack_f32a(pIn+(n+2)*inDim+i) ;
	const uint64_t i_b2_1 = _vel_pack_f32a(pIn+(n+2)*inDim+i+1) ;
	const uint64_t i_b2_2 = _vel_pack_f32a(pIn+(n+2)*inDim+i+2) ;
	const uint64_t i_b2_3 = _vel_pack_f32a(pIn+(n+2)*inDim+i+3) ;

	vrsum_b0 = _vel_pvfmad_vvsvl(vrsum_b0, i_b0_0, vrw_i0, vl) ;
	vrsum_b1 = _vel_pvfmad_vvsvl(vrsum_b1, i_b1_0, vrw_i0, vl) ;
	vrsum_b2 = _vel_pvfmad_vvsvl(vrsum_b2, i_b2_0, vrw_i0, vl) ;

	vrsum_b0 = _vel_pvfmad_vvsvl(vrsum_b0, i_b0_1, vrw_i1, vl) ;
	vrsum_b1 = _vel_pvfmad_vvsvl(vrsum_b1, i_b1_1, vrw_i1, vl) ;
	vrsum_b2 = _vel_pvfmad_vvsvl(vrsum_b2, i_b2_1, vrw_i1, vl) ;

	vrsum_b0 = _vel_pvfmad_vvsvl(vrsum_b0, i_b0_2, vrw_i2, vl) ;
	vrsum_b1 = _vel_pvfmad_vvsvl(vrsum_b1, i_b1_2, vrw_i2, vl) ;
	vrsum_b2 = _vel_pvfmad_vvsvl(vrsum_b2, i_b2_2, vrw_i2, vl) ;

	vrsum_b0 = _vel_pvfmad_vvsvl(vrsum_b0, i_b0_3, vrw_i3, vl) ;
	vrsum_b1 = _vel_pvfmad_vvsvl(vrsum_b1, i_b1_3, vrw_i3, vl) ;
	vrsum_b2 = _vel_pvfmad_vvsvl(vrsum_b2, i_b2_3, vrw_i3, vl) ;
      }

      _vel_vst_vssl(vrsum_b0, 8, pOut+(n  )*outDim+o, vl) ;
      _vel_vst_vssl(vrsum_b1, 8, pOut+(n+1)*outDim+o, vl) ;
      _vel_vst_vssl(vrsum_b2, 8, pOut+(n+2)*outDim+o, vl) ;
    }
    n+=3 ;
    break ;
  default :
    break ;
  }
  for(; n<nBatch; n+=4) {
    for(int64_t o=0; o<outDim; o+=2*VLEN) {
      const int64_t vl = (outDim-o < 2*VLEN ? outDim - o : 2*VLEN) >> 1 ;

      __vr vrsum_b0 = _vel_pvbrd_vsl(0UL, vl) ;
      __vr vrsum_b1 = _vel_pvbrd_vsl(0UL, vl) ;
      __vr vrsum_b2 = _vel_pvbrd_vsl(0UL, vl) ;
      __vr vrsum_b3 = _vel_pvbrd_vsl(0UL, vl) ;

      int64_t i=0;
      if((inDim & 0x01)==1) {
	__vr vrw = _vel_vld_vssl(8, pWeight+i*outDim+o, vl) ;

	const uint64_t i_b0_0 = _vel_pack_f32a(pIn+(n  )*inDim+i) ;
	const uint64_t i_b1_0 = _vel_pack_f32a(pIn+(n+1)*inDim+i) ;
	const uint64_t i_b2_0 = _vel_pack_f32a(pIn+(n+2)*inDim+i) ;
	const uint64_t i_b3_0 = _vel_pack_f32a(pIn+(n+3)*inDim+i) ;

	vrsum_b0 = _vel_pvfmad_vvsvl(vrsum_b0, i_b0_0, vrw, vl) ;
	vrsum_b1 = _vel_pvfmad_vvsvl(vrsum_b1, i_b1_0, vrw, vl) ;
	vrsum_b2 = _vel_pvfmad_vvsvl(vrsum_b2, i_b2_0, vrw, vl) ;
	vrsum_b3 = _vel_pvfmad_vvsvl(vrsum_b3, i_b3_0, vrw, vl) ;

	i+=1 ;
      }
      if(((inDim>>1) & 0x01)==1) {
	__vr vrw_i0 = _vel_vld_vssl(8, pWeight+(i  )*outDim+o, vl) ;
	__vr vrw_i1 = _vel_vld_vssl(8, pWeight+(i+1)*outDim+o, vl) ;

	const uint64_t i_b0_0 = _vel_pack_f32a(pIn+(n  )*inDim+i) ;
	const uint64_t i_b0_1 = _vel_pack_f32a(pIn+(n  )*inDim+i+1) ;

	const uint64_t i_b1_0 = _vel_pack_f32a(pIn+(n+1)*inDim+i) ;
	const uint64_t i_b1_1 = _vel_pack_f32a(pIn+(n+1)*inDim+i+1) ;

	const uint64_t i_b2_0 = _vel_pack_f32a(pIn+(n+2)*inDim+i) ;
	const uint64_t i_b2_1 = _vel_pack_f32a(pIn+(n+2)*inDim+i+1) ;

	const uint64_t i_b3_0 = _vel_pack_f32a(pIn+(n+3)*inDim+i) ;
	const uint64_t i_b3_1 = _vel_pack_f32a(pIn+(n+3)*inDim+i+1) ;

	vrsum_b0 = _vel_pvfmad_vvsvl(vrsum_b0, i_b0_0, vrw_i0, vl) ;
	vrsum_b1 = _vel_pvfmad_vvsvl(vrsum_b1, i_b1_0, vrw_i0, vl) ;
	vrsum_b2 = _vel_pvfmad_vvsvl(vrsum_b2, i_b2_0, vrw_i0, vl) ;
	vrsum_b3 = _vel_pvfmad_vvsvl(vrsum_b3, i_b3_0, vrw_i0, vl) ;

	vrsum_b0 = _vel_pvfmad_vvsvl(vrsum_b0, i_b0_1, vrw_i1, vl) ;
	vrsum_b1 = _vel_pvfmad_vvsvl(vrsum_b1, i_b1_1, vrw_i1, vl) ;
	vrsum_b2 = _vel_pvfmad_vvsvl(vrsum_b2, i_b2_1, vrw_i1, vl) ;
	vrsum_b3 = _vel_pvfmad_vvsvl(vrsum_b3, i_b3_1, vrw_i1, vl) ;

	i+=2 ;
      }
      for(; i<inDim; i+=4 ) {
	__vr vrw_i0 = _vel_vld_vssl(8, pWeight+(i  )*outDim+o, vl) ;
	__vr vrw_i1 = _vel_vld_vssl(8, pWeight+(i+1)*outDim+o, vl) ;
	__vr vrw_i2 = _vel_vld_vssl(8, pWeight+(i+2)*outDim+o, vl) ;
	__vr vrw_i3 = _vel_vld_vssl(8, pWeight+(i+3)*outDim+o, vl) ;

	const uint64_t i_b0_0 = _vel_pack_f32a(pIn+(n  )*inDim+i) ;
	const uint64_t i_b0_1 = _vel_pack_f32a(pIn+(n  )*inDim+i+1) ;
	const uint64_t i_b0_2 = _vel_pack_f32a(pIn+(n  )*inDim+i+2) ;
	const uint64_t i_b0_3 = _vel_pack_f32a(pIn+(n  )*inDim+i+3) ;

	const uint64_t i_b1_0 = _vel_pack_f32a(pIn+(n+1)*inDim+i) ;
	const uint64_t i_b1_1 = _vel_pack_f32a(pIn+(n+1)*inDim+i+1) ;
	const uint64_t i_b1_2 = _vel_pack_f32a(pIn+(n+1)*inDim+i+2) ;
	const uint64_t i_b1_3 = _vel_pack_f32a(pIn+(n+1)*inDim+i+3) ;

	const uint64_t i_b2_0 = _vel_pack_f32a(pIn+(n+2)*inDim+i) ;
	const uint64_t i_b2_1 = _vel_pack_f32a(pIn+(n+2)*inDim+i+1) ;
	const uint64_t i_b2_2 = _vel_pack_f32a(pIn+(n+2)*inDim+i+2) ;
	const uint64_t i_b2_3 = _vel_pack_f32a(pIn+(n+2)*inDim+i+3) ;

	const uint64_t i_b3_0 = _vel_pack_f32a(pIn+(n+3)*inDim+i) ;
	const uint64_t i_b3_1 = _vel_pack_f32a(pIn+(n+3)*inDim+i+1) ;
	const uint64_t i_b3_2 = _vel_pack_f32a(pIn+(n+3)*inDim+i+2) ;
	const uint64_t i_b3_3 = _vel_pack_f32a(pIn+(n+3)*inDim+i+3) ;

	vrsum_b0 = _vel_pvfmad_vvsvl(vrsum_b0, i_b0_0, vrw_i0, vl) ;
	vrsum_b1 = _vel_pvfmad_vvsvl(vrsum_b1, i_b1_0, vrw_i0, vl) ;
	vrsum_b2 = _vel_pvfmad_vvsvl(vrsum_b2, i_b2_0, vrw_i0, vl) ;
	vrsum_b3 = _vel_pvfmad_vvsvl(vrsum_b3, i_b3_0, vrw_i0, vl) ;

	vrsum_b0 = _vel_pvfmad_vvsvl(vrsum_b0, i_b0_1, vrw_i1, vl) ;
	vrsum_b1 = _vel_pvfmad_vvsvl(vrsum_b1, i_b1_1, vrw_i1, vl) ;
	vrsum_b2 = _vel_pvfmad_vvsvl(vrsum_b2, i_b2_1, vrw_i1, vl) ;
	vrsum_b3 = _vel_pvfmad_vvsvl(vrsum_b3, i_b3_1, vrw_i1, vl) ;

	vrsum_b0 = _vel_pvfmad_vvsvl(vrsum_b0, i_b0_2, vrw_i2, vl) ;
	vrsum_b1 = _vel_pvfmad_vvsvl(vrsum_b1, i_b1_2, vrw_i2, vl) ;
	vrsum_b2 = _vel_pvfmad_vvsvl(vrsum_b2, i_b2_2, vrw_i2, vl) ;
	vrsum_b3 = _vel_pvfmad_vvsvl(vrsum_b3, i_b3_2, vrw_i2, vl) ;

	vrsum_b0 = _vel_pvfmad_vvsvl(vrsum_b0, i_b0_3, vrw_i3, vl) ;
	vrsum_b1 = _vel_pvfmad_vvsvl(vrsum_b1, i_b1_3, vrw_i3, vl) ;
	vrsum_b2 = _vel_pvfmad_vvsvl(vrsum_b2, i_b2_3, vrw_i3, vl) ;
	vrsum_b3 = _vel_pvfmad_vvsvl(vrsum_b3, i_b3_3, vrw_i3, vl) ;
      }

      _vel_vst_vssl(vrsum_b0, 8, pOut+(n  )*outDim+o, vl) ;
      _vel_vst_vssl(vrsum_b1, 8, pOut+(n+1)*outDim+o, vl) ;
      _vel_vst_vssl(vrsum_b2, 8, pOut+(n+2)*outDim+o, vl) ;
      _vel_vst_vssl(vrsum_b3, 8, pOut+(n+3)*outDim+o, vl) ;
    }
  }

  return VEDNN_SUCCESS ;
}
