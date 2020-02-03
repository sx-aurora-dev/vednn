#include "vednn.h"
#include "velintrin.h"
#include <stdint.h>
#include <float.h>
//#include <stdio.h>

#define VLEN  (256)

vednnError_t vednnActivationBackward_Relu(
    const void     *pDataGradOut,
    const void     *pDataIn,
    void           *pDataGradIn,
    const uint64_t nElements
)
{
  const float * restrict pGOut   = (float const* restrict)pDataGradOut;
  const float * restrict pIn     = (float const* restrict)pDataIn;
  float * restrict const pGIn    = (float      * restrict)pDataGradIn;

  const uint64_t alignGOut = ((uint64_t)pDataGradOut) & 0x07 ;
  const uint64_t alignIn   = ((uint64_t)pDataIn) & 0x07 ;
  const uint64_t alignGIn  = ((uint64_t)pDataGradIn) & 0x07 ;

  if( alignGOut == 0 && alignIn == 0 &&  alignGIn ==0 ) {
    const uint64_t halfElements = nElements >> 1 ;

    for(int64_t i=0; i<halfElements; i+=VLEN) {
      const int64_t vl = halfElements - i < VLEN ? halfElements - i : VLEN ;

      __vr vrin    = _vel_vld_vssl(8, pIn+2*i, vl) ;
      __vr vrgout  = _vel_vld_vssl(8, pGOut+2*i, vl) ;

      __vm512 vm = _vel_pvfmksgt_Mvl(vrin, vl) ;

      __vr vrgin = _vel_vmrgw_vvvMl(_vel_pvbrd_vsl(0UL, vl), vrgout, vm, vl) ;

      _vel_vst_vssl(vrgin, 8, pGIn+2*i, vl) ;
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

       ;

      __vr vrin    = _vel_vld_vssl(8, pIn+2*i+1, vl) ;
      __vr vrgout  = _vel_vld_vssl(8, pGOut+2*i+1, vl) ;

      __vm512 vm = _vel_pvfmksgt_Mvl(vrin, vl) ;

      __vr vrgin = _vel_vmrgw_vvvMl(_vel_pvbrd_vsl(0UL, vl), vrgout, vm, vl) ;

      _vel_vst_vssl(vrgin, 8, pGIn+2*i+1, vl) ;
    }
    if( (nElements & 0x01) == 0 ) {
      pGIn[nElements-1] = pGOut[nElements-1] * ( pIn[nElements-1] > 0 ) ;
    }
  }
  else {
    for(int64_t i=0; i<nElements; i+=VLEN) {
      const int64_t vl = nElements - i < VLEN ? nElements - i : VLEN ;

       ;

      __vr vrin    = _vel_vldu_vssl(4, pIn+i, vl) ;
      __vr vrgout  = _vel_vldu_vssl(4, pGOut+i, vl) ;

      __vm256 vm =  _vel_vfmksgt_mvl(vrin, vl) ;

      __vr vrgin = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrgout, vm, vl) ;

      _vel_vstu_vssl(vrgin, 4, pGIn+i, vl) ;
    }
  }

  return VEDNN_SUCCESS ;
}
// vim: sw=2 ts=2 et
