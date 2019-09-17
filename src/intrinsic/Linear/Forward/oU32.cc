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
  float * 		pOut,
  const int64_t         maxvl
)
{
  for(int64_t o=0; o<outDim; o++) {

    __vr vrsum_b0 = _vel_vbrds_vsl(0.0f, maxvl) ;
    __vr vrsum_b1 = _vel_vbrds_vsl(0.0f, maxvl) ;
    __vr vrsum_b2 = _vel_vbrds_vsl(0.0f, maxvl) ;
    __vr vrsum_b3 = _vel_vbrds_vsl(0.0f, maxvl) ;

    for(int64_t i=0; i<inDim; i+=VLEN ) {
	const int64_t vl = inDim-i < VLEN ? inDim-i : VLEN ;

	__vr vrw = _vel_vldu_vssl(outDim*4, pWeight+i*outDim+o, vl) ;

	__vr vri_b0 = _vel_vldu_vssl(4, pIn+0*inDim+i, vl) ;
	__vr vri_b1 = _vel_vldu_vssl(4, pIn+1*inDim+i, vl) ;
	__vr vri_b2 = _vel_vldu_vssl(4, pIn+2*inDim+i, vl) ;
	__vr vri_b3 = _vel_vldu_vssl(4, pIn+3*inDim+i, vl) ;

	if(BATCH >=1) vrsum_b0 = _vel_vfmads_vvvvvl(vrsum_b0, vrw, vri_b0, vrsum_b0, vl) ;
	if(BATCH >=2) vrsum_b1 = _vel_vfmads_vvvvvl(vrsum_b1, vrw, vri_b1, vrsum_b1, vl) ;
	if(BATCH >=3) vrsum_b2 = _vel_vfmads_vvvvvl(vrsum_b2, vrw, vri_b2, vrsum_b2, vl) ;
	if(BATCH >=4) vrsum_b3 = _vel_vfmads_vvvvvl(vrsum_b3, vrw, vri_b3, vrsum_b3, vl) ;
    }
    if(BATCH >=1) vrsum_b0 = _vel_vfsums_vvl(vrsum_b0, maxvl) ;
    if(BATCH >=2) vrsum_b1 = _vel_vfsums_vvl(vrsum_b1, maxvl) ;
    if(BATCH >=3) vrsum_b2 = _vel_vfsums_vvl(vrsum_b2, maxvl) ;
    if(BATCH >=4) vrsum_b3 = _vel_vfsums_vvl(vrsum_b3, maxvl) ;

    if(BATCH >=1) _vel_vstu_vssl(vrsum_b0, 4, pOut+0*outDim+o, 1) ;
    if(BATCH >=2) _vel_vstu_vssl(vrsum_b1, 4, pOut+1*outDim+o, 1) ;
    if(BATCH >=3) _vel_vstu_vssl(vrsum_b2, 4, pOut+2*outDim+o, 1) ;
    if(BATCH >=4) _vel_vstu_vssl(vrsum_b3, 4, pOut+3*outDim+o, 1) ;
  }
}

extern "C"
vednnError_t vednnLinearForward_oU32(
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

  const int64_t maxvl = inDim < VLEN ? inDim : VLEN ;

  switch( batchRemain ) {
  case 1:
    func<1>(inDim, outDim,pIn+n*inDim, pWeight, pOut+n*outDim, maxvl) ;
    n+=1 ;
    break ;
  case 2:
    func<2>(inDim, outDim,pIn+n*inDim, pWeight, pOut+n*outDim, maxvl) ;
    n+=2 ;
    break ;
  case 3:
    func<3>(inDim, outDim,pIn+n*inDim, pWeight, pOut+n*outDim, maxvl) ;
    n+=3 ;
    break ;
  default :
    break ;
  }

  for(; n<nBatch; n+=4) {
    func<4>(inDim, outDim,pIn+n*inDim, pWeight, pOut+n*outDim, maxvl) ;
  }

  return VEDNN_SUCCESS ;
}


