#include <stdint.h>
#include <float.h>

#include <stdio.h>

#include "vednn.h"

#include "veintrin.h"
#define VLEN	(256)


vednnError_t vednnLinearForward_oU32(
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

  const int64_t maxvl = inDim < VLEN ? inDim : VLEN ;

  switch( batchRemain ) {
  case 1:
    for(int64_t o=0; o<outDim; o++) {

      _ve_lvl(maxvl) ;
      __vr vrsum_b0 = _ve_vbrdu_vs_f32(0.0f) ;

      for(int64_t i=0; i<inDim; i+=VLEN ) {
	const int64_t vl = inDim-i < VLEN ? inDim-i : VLEN ;
	_ve_lvl(vl) ;

	__vr vrw = _ve_vldu_vss(outDim*4, pWeight+i*outDim+o) ;

	__vr vri_b0 = _ve_vldu_vss(4, pIn+(n+0)*inDim+i) ;

	vrsum_b0 = _ve_vfmads_vvvv(vrsum_b0, vrw, vri_b0) ;
      }
      _ve_lvl(maxvl) ;
      vrsum_b0 = _ve_vfsums_vv(vrsum_b0) ;

      _ve_lvl(1) ;
      _ve_vstu_vss(vrsum_b0, 4, pOut+(n+0)*outDim+o) ;
    }
    n+=1 ;
    break ;
  case 2:
    for(int64_t o=0; o<outDim; o++) {

      _ve_lvl(maxvl) ;
      __vr vrsum_b0 = _ve_vbrdu_vs_f32(0.0f) ;
      __vr vrsum_b1 = _ve_vbrdu_vs_f32(0.0f) ;

      for(int64_t i=0; i<inDim; i+=VLEN ) {
	const int64_t vl = inDim-i < VLEN ? inDim-i : VLEN ;
	_ve_lvl(vl) ;

	__vr vrw = _ve_vldu_vss(outDim*4, pWeight+i*outDim+o) ;

	__vr vri_b0 = _ve_vldu_vss(4, pIn+(n+0)*inDim+i) ;
	__vr vri_b1 = _ve_vldu_vss(4, pIn+(n+1)*inDim+i) ;

	vrsum_b0 = _ve_vfmads_vvvv(vrsum_b0, vrw, vri_b0) ;
	vrsum_b1 = _ve_vfmads_vvvv(vrsum_b1, vrw, vri_b1) ;
      }
      _ve_lvl(maxvl) ;
      vrsum_b0 = _ve_vfsums_vv(vrsum_b0) ;
      vrsum_b1 = _ve_vfsums_vv(vrsum_b1) ;

      _ve_lvl(1) ;
      _ve_vstu_vss(vrsum_b0, 4, pOut+(n+0)*outDim+o) ;
      _ve_vstu_vss(vrsum_b1, 4, pOut+(n+1)*outDim+o) ;
    }
    n+=2 ;
    break ;
  case 3:
    for(int64_t o=0; o<outDim; o++) {

      _ve_lvl(maxvl) ;
      __vr vrsum_b0 = _ve_vbrdu_vs_f32(0.0f) ;
      __vr vrsum_b1 = _ve_vbrdu_vs_f32(0.0f) ;
      __vr vrsum_b2 = _ve_vbrdu_vs_f32(0.0f) ;

      for(int64_t i=0; i<inDim; i+=VLEN ) {
	const int64_t vl = inDim-i < VLEN ? inDim-i : VLEN ;
	_ve_lvl(vl) ;

	__vr vrw = _ve_vldu_vss(outDim*4, pWeight+i*outDim+o) ;

	__vr vri_b0 = _ve_vldu_vss(4, pIn+(n+0)*inDim+i) ;
	__vr vri_b1 = _ve_vldu_vss(4, pIn+(n+1)*inDim+i) ;
	__vr vri_b2 = _ve_vldu_vss(4, pIn+(n+2)*inDim+i) ;

	vrsum_b0 = _ve_vfmads_vvvv(vrsum_b0, vrw, vri_b0) ;
	vrsum_b1 = _ve_vfmads_vvvv(vrsum_b1, vrw, vri_b1) ;
	vrsum_b2 = _ve_vfmads_vvvv(vrsum_b2, vrw, vri_b2) ;
      }
      _ve_lvl(maxvl) ;
      vrsum_b0 = _ve_vfsums_vv(vrsum_b0) ;
      vrsum_b1 = _ve_vfsums_vv(vrsum_b1) ;
      vrsum_b2 = _ve_vfsums_vv(vrsum_b2) ;

      _ve_lvl(1) ;
      _ve_vstu_vss(vrsum_b0, 4, pOut+(n+0)*outDim+o) ;
      _ve_vstu_vss(vrsum_b1, 4, pOut+(n+1)*outDim+o) ;
      _ve_vstu_vss(vrsum_b2, 4, pOut+(n+2)*outDim+o) ;
    }
    n+=3 ;
    break ;
  default :
    break ;
  }

  for(; n<nBatch; n+=4) {
    for(int64_t o=0; o<outDim; o++) {

      _ve_lvl(maxvl) ;
      __vr vrsum_b0 = _ve_vbrdu_vs_f32(0.0f) ;
      __vr vrsum_b1 = _ve_vbrdu_vs_f32(0.0f) ;
      __vr vrsum_b2 = _ve_vbrdu_vs_f32(0.0f) ;
      __vr vrsum_b3 = _ve_vbrdu_vs_f32(0.0f) ;

      for(int64_t i=0; i<inDim; i+=VLEN ) {
	const int64_t vl = inDim-i < VLEN ? inDim-i : VLEN ;
	_ve_lvl(vl) ;

	__vr vrw = _ve_vldu_vss(outDim*4, pWeight+i*outDim+o) ;

	__vr vri_b0 = _ve_vldu_vss(4, pIn+(n+0)*inDim+i) ;
	__vr vri_b1 = _ve_vldu_vss(4, pIn+(n+1)*inDim+i) ;
	__vr vri_b2 = _ve_vldu_vss(4, pIn+(n+2)*inDim+i) ;
	__vr vri_b3 = _ve_vldu_vss(4, pIn+(n+3)*inDim+i) ;

	vrsum_b0 = _ve_vfmads_vvvv(vrsum_b0, vrw, vri_b0) ;
	vrsum_b1 = _ve_vfmads_vvvv(vrsum_b1, vrw, vri_b1) ;
	vrsum_b2 = _ve_vfmads_vvvv(vrsum_b2, vrw, vri_b2) ;
	vrsum_b3 = _ve_vfmads_vvvv(vrsum_b3, vrw, vri_b3) ;
      }
      _ve_lvl(maxvl) ;
      vrsum_b0 = _ve_vfsums_vv(vrsum_b0) ;
      vrsum_b1 = _ve_vfsums_vv(vrsum_b1) ;
      vrsum_b2 = _ve_vfsums_vv(vrsum_b2) ;
      vrsum_b3 = _ve_vfsums_vv(vrsum_b3) ;

      _ve_lvl(1) ;
      _ve_vstu_vss(vrsum_b0, 4, pOut+(n+0)*outDim+o) ;
      _ve_vstu_vss(vrsum_b1, 4, pOut+(n+1)*outDim+o) ;
      _ve_vstu_vss(vrsum_b2, 4, pOut+(n+2)*outDim+o) ;
      _ve_vstu_vss(vrsum_b3, 4, pOut+(n+3)*outDim+o) ;
    }
  }

  return VEDNN_SUCCESS ;
}

#if 0 // reference code
vednnError_t vednnLinearForward_default(
    const uint64_t			inDim,
    const uint64_t			outDim,
    const uint64_t			nBatch,
    const void * restrict		pDataIn,
    const void * restrict		pDataWeight,
    void * restrict			pDataOut
)
{
  const float * restrict pIn     = pDataIn;
  const float * restrict pWeight = pDataWeight;
  float * restrict const pOut    = pDataOut;

  for(int64_t n=0; n<nBatch; n++) {
    for(int64_t o=0; o<outDim; o++) {
      float sum = 0.f ;
      for(int64_t i=0; i<inDim; i++ ) {
	sum += pWeight[i*outDim+o] * pIn[n*inDim+i] ;
      }
      pOut[n*outDim+o] = sum ;
    }
  }

  return VEDNN_SUCCESS ;
}
#endif


