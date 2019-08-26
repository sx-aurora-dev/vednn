#include <stdint.h>
#include <float.h>

#include <stdio.h>

#include "vednn.h"

#include "velintrin.h"
#define VLEN	(256)


vednnError_t vednnActivationForward_Relu(
    const void 				*pDataIn,
    void 				*pDataOut,
    const uint64_t			nElements
)
{
  const float * restrict pIn     = pDataIn;
  float * restrict const pOut    = pDataOut;

  const uint64_t alignIn  = ((uint64_t)pDataIn) & 0x07 ;
  const uint64_t alignOut = ((uint64_t)pDataOut) & 0x07 ;

  if( alignIn == 0 && alignOut == 0 ) {
    const uint64_t halfElements = nElements >> 1 ;
    for(int64_t i=0; i<halfElements; i+=VLEN) {
      const int64_t vl = halfElements - i < VLEN ? halfElements - i : VLEN ;

      __vr vrin  = _vel_vld_vssl(8, pIn+2*i, vl) ;
      __vr vrout = _vel_pvfmax_vsvl(0UL, vrin, vl) ;

      _vel_vst_vssl(vrout, 8, pOut+2*i, vl) ;
    }
    if( (nElements & 0x01) == 1 ) {
      pOut[nElements-1] = pIn[nElements-1] > 0.0f ? pIn[nElements-1] : 0.0f ;
    }
  }
  else if( alignIn == 4 && alignOut == 4 ) {

    const uint64_t halfElements = (nElements-1) >> 1 ;
    pOut[0] = pIn[0] > 0.0f ? pIn[0] : 0.0f ;
    for(int64_t i=0; i<halfElements; i+=VLEN) {
      const int64_t vl = halfElements - i < VLEN ? halfElements - i : VLEN ;

      __vr vrin  = _vel_vld_vssl(8, pIn+2*i+1, vl) ;
      __vr vrout = _vel_pvfmax_vsvl(0UL, vrin, vl) ;

      _vel_vst_vssl(vrout, 8, pOut+2*i+1, vl) ;
    }
    if( (nElements & 0x01) == 0 ) {
      pOut[nElements-1] = pIn[nElements-1] > 0.0f ? pIn[nElements-1] : 0.0f ;
    }
  }
  else {
    for(int64_t i=0; i<nElements; i+=VLEN) {
      const int64_t vl = nElements - i < VLEN ? nElements - i : VLEN ;

      __vr vrin  = _vel_vldu_vssl(4, pIn+i, vl) ;
      __vr vrout = _vel_vfmaxs_vsvl(0.0f, vrin, vl) ;

      _vel_vstu_vssl(vrout, 4, pOut+i, vl) ;
    }
  }

  return VEDNN_SUCCESS ;
}



