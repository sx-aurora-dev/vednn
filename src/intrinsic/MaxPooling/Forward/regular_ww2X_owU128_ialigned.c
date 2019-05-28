#include <stdint.h>
#include <float.h>

#include <stdio.h>

#include "vednn.h"

#include "veintrin.h"
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

    _ve_lvl(VLEN) ;

    __vr vrseq = _ve_vseq_v() ;	// dh*dw

    __vr vrdh  = _ve_vdivsl_vvs(vrseq, outWidth) ;
    __vr vrdw  = _ve_vsubsl_vvv(vrseq, _ve_vmulul_vsv(outWidth,vrdh)) ;

    __vr vrdy  = _ve_vmulsl_vsv(strideHeight,vrdh) ;
    __vr vrdx  = _ve_vmulsl_vsv(strideWidth,vrdw) ;

    __vr vri_idx    = _ve_vaddsl_vvv(_ve_vmulsl_vsv(inWidth, vrdy), vrdx) ;

    for(int64_t n=0; n<batch; n++) {
      for(int64_t c=0; c<outChannel; c++) {
	for (int64_t h0=0; h0<outHeight; h0+=nH) {
	  const int64_t w0 = 0 ;
	  const int64_t vlen = outWidth * (outHeight - h0 < nH ? outHeight - h0 : nH) ;

	  const int64_t outIndex  = NCHW_IDX(n,c,h0,w0,outChannel,outHeight,outWidth) ;

	  _ve_lvl(vlen) ;

	  const float minus_flt_max = -FLT_MAX ;
	  __vr vrmax = _ve_vbrd_vs_i64(_ve_pack_f32a(&minus_flt_max)) ;

	  for(int64_t ph=0; ph<windowHeight; ph++) {
	    const int64_t y0 = h0*strideHeight + ph ;

	    for(int64_t pw=0; pw<windowWidth; pw+=2) {
	      const int64_t x0 = w0*strideWidth + pw ;
	      const int64_t inIndex = NCHW_IDX(n,c,y0,x0,inChannel,inHeight,inWidth) ;

	      __vr vrin_addr  = _ve_vsfa_vvss(vri_idx, 2, (unsigned long)&pIn[inIndex]) ;
	      __vr vrin = _ve_vgt_vv(vrin_addr) ;

	      vrmax = _ve_pvfmax_vvv(vrin,vrmax) ;
	    } // windowWidth
	  } // windowHeight

	  vrmax = _ve_vfmaxs_vvv(vrmax, _ve_vsll_vvs(vrmax,32)) ;

	  _ve_vstu_vss(vrmax, 4, pOut+outIndex) ;
	} // outHeight
      } // channel
    } // batch
  }

  return VEDNN_SUCCESS ;
}



