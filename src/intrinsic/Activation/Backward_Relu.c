#include <stdint.h>
#include <float.h>

#include <stdio.h>

#include "vednn.h"

#include "veintrin.h"
#define VLEN	(256)


vednnError_t vednnActivationBackward_Relu(
    const void 				*pDataGradOut,
    const void 				*pDataIn,
    void 				*pDataGradIn,
    const uint64_t			nElements
)
{
  const float * restrict pGOut   = pDataGradOut;
  const float * restrict pIn     = pDataIn;
  float * restrict const pGIn    = pDataGradIn;

  const uint64_t alignGOut = ((uint64_t)pDataGradOut) & 0x07 ;
  const uint64_t alignIn   = ((uint64_t)pDataIn) & 0x07 ;
  const uint64_t alignGIn  = ((uint64_t)pDataGradIn) & 0x07 ;

  if( alignGOut == 0 && alignIn == 0 &&  alignGIn ==0 ) {
    const uint64_t halfElements = nElements >> 1 ;

    for(int64_t i=0; i<halfElements; i+=VLEN) {
      const int64_t vl = halfElements - i < VLEN ? halfElements - i : VLEN ;

      _ve_lvl(vl) ;

      __vr vrin    = _ve_vld_vss(8, pIn+2*i) ;
      __vr vrgout  = _ve_vld_vss(8, pGOut+2*i) ;

      __vm512 vm = _ve_pvfmks_Mcv(VECC_G, vrin) ;

      __vr vrgin = _ve_vmrgw_vvvM(_ve_pvbrd_vs_i64(0UL), vrgout, vm) ;

      _ve_vst_vss(vrgin, 8, pGIn+2*i) ;
    }
    if( (nElements & 0x01) == 1 ) {
      pGIn[nElements-1] = pGOut[nElements-1] * ( pIn[nElements-1] > 0 ) ;
    }
  }
  else if( alignGOut == 4 && alignIn == 4 &&  alignGIn ==4 ) {
    const uint64_t halfElements = (nElements-1) >> 1 ;

    pGIn[0] = pGOut[0] * ( pIn[0] > 0 ) ;
    for(int64_t i=0; i<halfElements; i+=VLEN) {
      const int64_t vl = halfElements - i < VLEN ? halfElements - i : VLEN ;

      _ve_lvl(vl) ;

      __vr vrin    = _ve_vld_vss(8, pIn+2*i+1) ;
      __vr vrgout  = _ve_vld_vss(8, pGOut+2*i+1) ;

      __vm512 vm = _ve_pvfmks_Mcv(VECC_G, vrin) ;

      __vr vrgin = _ve_vmrgw_vvvM(_ve_pvbrd_vs_i64(0UL), vrgout, vm) ;

      _ve_vst_vss(vrgin, 8, pGIn+2*i+1) ;
    }
    if( (nElements & 0x01) == 0 ) {
      pGIn[nElements-1] = pGOut[nElements-1] * ( pIn[nElements-1] > 0 ) ;
    }
  }
  else {
    for(int64_t i=0; i<nElements; i+=VLEN) {
      const int64_t vl = nElements - i < VLEN ? nElements - i : VLEN ;

      _ve_lvl(vl) ;

      __vr vrin    = _ve_vldu_vss(4, pIn+i) ;
      __vr vrgout  = _ve_vldu_vss(4, pGOut+i) ;

      __vm256 vm = _ve_vfmks_mcv(VECC_G, vrin) ;

      __vr vrgin = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), vrgout, vm) ;

      _ve_vstu_vss(vrgin, 4, pGIn+i) ;
    }
  }

  return VEDNN_SUCCESS ;
}



