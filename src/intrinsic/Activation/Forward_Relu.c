#include <stdint.h>
#include <float.h>

#include <stdio.h>

#include "vednn.h"

#include "veintrin.h"
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

      _ve_lvl(vl) ;

      __vr vrin  = _ve_vld_vss(8, pIn+2*i) ;
      __vr vrout = _ve_pvfmax_vsv(0UL, vrin) ;

      _ve_vst_vss(vrout, 8, pOut+2*i) ;
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

      _ve_lvl(vl) ;

      __vr vrin  = _ve_vld_vss(8, pIn+2*i+1) ;
      __vr vrout = _ve_pvfmax_vsv(0UL, vrin) ;

      _ve_vst_vss(vrout, 8, pOut+2*i+1) ;
    }
    if( (nElements & 0x01) == 0 ) {
      pOut[nElements-1] = pIn[nElements-1] > 0.0f ? pIn[nElements-1] : 0.0f ;
    }
  }
  else {
    for(int64_t i=0; i<nElements; i+=VLEN) {
      const int64_t vl = nElements - i < VLEN ? nElements - i : VLEN ;

      _ve_lvl(vl) ;

      __vr vrin  = _ve_vldu_vss(4, pIn+i) ;
      __vr vrout = _ve_vfmaxs_vsv(0.0f, vrin) ;

      _ve_vstu_vss(vrout, 4, pOut+i) ;
    }
  }

  return VEDNN_SUCCESS ;
}



