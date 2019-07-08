#include <stdint.h>
#include <float.h>

#include <stdio.h>

#include "vednn.h"

#include "velintrin.h"
#define VLEN	(256)

#define NCHW_IDX(n,c,h,w,cl,hl,wl) ((((n)*(cl)+(c))*(hl)+(h))*(wl)+(w))

vednnError_t vednnMaxPoolingForward_regular_ww2X_owU128_ialigned(
    const vednnTensorParam_t 		*pParamIn,
    const void 				*pDataIn,
    const vednnTensorParam_t 		*pParamOut,
    void 				*pDataOut,
    const vednnPoolingParam_t		*pParamPool
)
{
  const int64_t batch      = pParamIn->batch;
  const int64_t inChannel  = pParamIn->channel;
  const int64_t inWidth    = pParamIn->width;
  const int64_t inHeight   = pParamIn->height;
  const int64_t outChannel = pParamOut->channel;
  const int64_t outWidth   = pParamOut->width;
  const int64_t outHeight  = pParamOut->height;

  const int64_t windowWidth  = pParamPool->windowWidth;
  const int64_t windowHeight = pParamPool->windowHeight;
  const int64_t strideWidth  = pParamPool->strideWidth;;
  const int64_t strideHeight = pParamPool->strideHeight;
//  const int64_t padWidth     = pParamPool->padWidth;		// must be 0
//  const int64_t padHeight    = pParamPool->padHeight;		// must be 0

  const float * restrict pIn     = pDataIn;
  float * restrict const pOut    = pDataOut ;

  {
    const int64_t nH = VLEN / outWidth  ;

    __vr vrseq = _vel_vseq_vl(VLEN) ;	// dh*dw

    __vr vrdh  = _vel_vdivsl_vvsl(vrseq, outWidth, VLEN) ;
    __vr vrdw  = _vel_vsubsl_vvvl(vrseq, _vel_vmulul_vsvl(outWidth,vrdh, VLEN), VLEN) ;

    __vr vrdy  = _vel_vmulsl_vsvl(strideHeight,vrdh, VLEN) ;
    __vr vrdx  = _vel_vmulsl_vsvl(strideWidth,vrdw, VLEN) ;

    __vr vri_idx    = _vel_vaddsl_vvvl(_vel_vmulsl_vsvl(inWidth, vrdy, VLEN), vrdx, VLEN) ;

    for(int64_t n=0; n<batch; n++) {
      for(int64_t c=0; c<outChannel; c++) {
	for (int64_t h0=0; h0<outHeight; h0+=nH) {
	  const int64_t w0 = 0 ;
	  const int64_t vlen = outWidth * (outHeight - h0 < nH ? outHeight - h0 : nH) ;

	  const int64_t outIndex  = NCHW_IDX(n,c,h0,w0,outChannel,outHeight,outWidth) ;

	  const float minus_flt_max = -FLT_MAX ;
	  __vr vrmax = _vel_vbrdl_vsl(_vel_pack_f32a(&minus_flt_max), vlen) ;

	  for(int64_t ph=0; ph<windowHeight; ph++) {
	    const int64_t y0 = h0*strideHeight + ph ;

	    for(int64_t pw=0; pw<windowWidth; pw+=2) {
	      const int64_t x0 = w0*strideWidth + pw ;
	      const int64_t inIndex = NCHW_IDX(n,c,y0,x0,inChannel,inHeight,inWidth) ;

	      __vr vrin_addr  = _vel_vsfa_vvssl(vri_idx, 2, (unsigned long)&pIn[inIndex], vlen) ;
	      __vr vrin = _vel_vgt_vvssl(vrin_addr, 0, 0, vlen) ;

	      vrmax = _vel_pvfmax_vvvl(vrin,vrmax, vlen) ;
	    } // windowWidth
	  } // windowHeight

	  vrmax = _vel_vfmaxs_vvvl(vrmax, _vel_vsll_vvsl(vrmax,32, vlen), vlen) ;

	  _vel_vstu_vssl(vrmax, 4, pOut+outIndex, vlen) ;
	} // outHeight
      } // channel
    } // batch
  }

  return VEDNN_SUCCESS ;
}



