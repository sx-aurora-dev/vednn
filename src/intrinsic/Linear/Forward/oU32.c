#include <stdint.h>
#include <float.h>

#include <stdio.h>

#include "vednn.h"

#include "velintrin.h"
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

      __vr vrsum_b0 = _vel_vbrds_vsl(0.0f, maxvl) ;

      for(int64_t i=0; i<inDim; i+=VLEN ) {
	const int64_t vl = inDim-i < VLEN ? inDim-i : VLEN ;

	__vr vrw = _vel_vldu_vssl(outDim*4, pWeight+i*outDim+o, vl) ;

	__vr vri_b0 = _vel_vldu_vssl(4, pIn+(n+0)*inDim+i, vl) ;

	vrsum_b0 = _vel_vfmads_vvvvvl(vrsum_b0, vrw, vri_b0, vrsum_b0, vl) ;
      }
      vrsum_b0 = _vel_vfsums_vvl(vrsum_b0, maxvl) ;

      _vel_vstu_vssl(vrsum_b0, 4, pOut+(n+0)*outDim+o, 1) ;
    }
    n+=1 ;
    break ;

  case 2:
    for(int64_t o=0; o<outDim; o++) {

      __vr vrsum_b0 = _vel_vbrds_vsl(0.0f, maxvl) ;
      __vr vrsum_b1 = _vel_vbrds_vsl(0.0f, maxvl) ;

      for(int64_t i=0; i<inDim; i+=VLEN ) {
	const int64_t vl = inDim-i < VLEN ? inDim-i : VLEN ;

	__vr vrw = _vel_vldu_vssl(outDim*4, pWeight+i*outDim+o, vl) ;

	__vr vri_b0 = _vel_vldu_vssl(4, pIn+(n+0)*inDim+i, vl) ;
	__vr vri_b1 = _vel_vldu_vssl(4, pIn+(n+1)*inDim+i, vl) ;

	vrsum_b0 = _vel_vfmads_vvvvvl(vrsum_b0, vrw, vri_b0, vrsum_b0, vl) ;
	vrsum_b1 = _vel_vfmads_vvvvvl(vrsum_b1, vrw, vri_b1, vrsum_b1, vl) ;
      }
      vrsum_b0 = _vel_vfsums_vvl(vrsum_b0, maxvl) ;
      vrsum_b1 = _vel_vfsums_vvl(vrsum_b1, maxvl) ;

      _vel_vstu_vssl(vrsum_b0, 4, pOut+(n+0)*outDim+o, 1) ;
      _vel_vstu_vssl(vrsum_b1, 4, pOut+(n+1)*outDim+o, 1) ;
    }
    n+=2 ;
    break ;
  case 3:
    for(int64_t o=0; o<outDim; o++) {

      __vr vrsum_b0 = _vel_vbrds_vsl(0.0f, maxvl) ;
      __vr vrsum_b1 = _vel_vbrds_vsl(0.0f, maxvl) ;
      __vr vrsum_b2 = _vel_vbrds_vsl(0.0f, maxvl) ;

      for(int64_t i=0; i<inDim; i+=VLEN ) {
	const int64_t vl = inDim-i < VLEN ? inDim-i : VLEN ;

	__vr vrw = _vel_vldu_vssl(outDim*4, pWeight+i*outDim+o, vl) ;

	__vr vri_b0 = _vel_vldu_vssl(4, pIn+(n+0)*inDim+i, vl) ;
	__vr vri_b1 = _vel_vldu_vssl(4, pIn+(n+1)*inDim+i, vl) ;
	__vr vri_b2 = _vel_vldu_vssl(4, pIn+(n+2)*inDim+i, vl) ;

	vrsum_b0 = _vel_vfmads_vvvvvl(vrsum_b0, vrw, vri_b0, vrsum_b0, vl) ;
	vrsum_b1 = _vel_vfmads_vvvvvl(vrsum_b1, vrw, vri_b1, vrsum_b1, vl) ;
	vrsum_b2 = _vel_vfmads_vvvvvl(vrsum_b2, vrw, vri_b2, vrsum_b2, vl) ;
      }
      vrsum_b0 = _vel_vfsums_vvl(vrsum_b0, maxvl) ;
      vrsum_b1 = _vel_vfsums_vvl(vrsum_b1, maxvl) ;
      vrsum_b2 = _vel_vfsums_vvl(vrsum_b2, maxvl) ;

      _vel_vstu_vssl(vrsum_b0, 4, pOut+(n+0)*outDim+o, 1) ;
      _vel_vstu_vssl(vrsum_b1, 4, pOut+(n+1)*outDim+o, 1) ;
      _vel_vstu_vssl(vrsum_b2, 4, pOut+(n+2)*outDim+o, 1) ;
    }
    n+=3 ;
    break ;
  default :
    break ;
  }

  for(; n<nBatch; n+=4) {
    for(int64_t o=0; o<outDim; o++) {

      __vr vrsum_b0 = _vel_vbrds_vsl(0.0f, maxvl) ;
      __vr vrsum_b1 = _vel_vbrds_vsl(0.0f, maxvl) ;
      __vr vrsum_b2 = _vel_vbrds_vsl(0.0f, maxvl) ;
      __vr vrsum_b3 = _vel_vbrds_vsl(0.0f, maxvl) ;

      for(int64_t i=0; i<inDim; i+=VLEN ) {
	const int64_t vl = inDim-i < VLEN ? inDim-i : VLEN ;

	__vr vrw = _vel_vldu_vssl(outDim*4, pWeight+i*outDim+o, vl) ;

	__vr vri_b0 = _vel_vldu_vssl(4, pIn+(n+0)*inDim+i, vl) ;
	__vr vri_b1 = _vel_vldu_vssl(4, pIn+(n+1)*inDim+i, vl) ;
	__vr vri_b2 = _vel_vldu_vssl(4, pIn+(n+2)*inDim+i, vl) ;
	__vr vri_b3 = _vel_vldu_vssl(4, pIn+(n+3)*inDim+i, vl) ;

	vrsum_b0 = _vel_vfmads_vvvvvl(vrsum_b0, vrw, vri_b0, vrsum_b0, vl) ;
	vrsum_b1 = _vel_vfmads_vvvvvl(vrsum_b1, vrw, vri_b1, vrsum_b1, vl) ;
	vrsum_b2 = _vel_vfmads_vvvvvl(vrsum_b2, vrw, vri_b2, vrsum_b2, vl) ;
	vrsum_b3 = _vel_vfmads_vvvvvl(vrsum_b3, vrw, vri_b3, vrsum_b3, vl) ;
      }
      vrsum_b0 = _vel_vfsums_vvl(vrsum_b0, maxvl) ;
      vrsum_b1 = _vel_vfsums_vvl(vrsum_b1, maxvl) ;
      vrsum_b2 = _vel_vfsums_vvl(vrsum_b2, maxvl) ;
      vrsum_b3 = _vel_vfsums_vvl(vrsum_b3, maxvl) ;

      _vel_vstu_vssl(vrsum_b0, 4, pOut+(n+0)*outDim+o, 1) ;
      _vel_vstu_vssl(vrsum_b1, 4, pOut+(n+1)*outDim+o, 1) ;
      _vel_vstu_vssl(vrsum_b2, 4, pOut+(n+2)*outDim+o, 1) ;
      _vel_vstu_vssl(vrsum_b3, 4, pOut+(n+3)*outDim+o, 1) ;
    }
  }

  return VEDNN_SUCCESS ;
}


