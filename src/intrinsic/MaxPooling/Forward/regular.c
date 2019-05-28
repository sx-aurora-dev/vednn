#include <stdint.h>
#include <float.h>

#include <stdio.h>

#include "vednn.h"

#include "veintrin.h"
#define VLEN	(256)

#define NCHW_IDX(n,c,h,w,cl,hl,wl) ((((n)*(cl)+(c))*(hl)+(h))*(wl)+(w))

vednnError_t vednnMaxPoolingForward_regular(
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
  float * restrict const pOut    = pDataOut;

  {
    for(int64_t n=0; n<batch; n++) {
      for(int64_t c=0; c<outChannel; c++) {
	for(int64_t h=0; h<outHeight; h++) {
	  for(int64_t w=0; w<outWidth; w+=VLEN) {
	    const int64_t vlen = outWidth-w < VLEN ? outWidth-w : VLEN ;

	    const int64_t outIndex  = NCHW_IDX(n,c,h,w,outChannel,outHeight,outWidth) ;

	    _ve_lvl(vlen) ;

	    __vr vrout = _ve_vbrdu_vs_f32(-FLT_MAX) ;

	    for(int64_t ph=0; ph<windowHeight; ph++) {
	      const int64_t y = h*strideHeight + ph ;

	      for(int64_t pw=0; pw<windowWidth; pw++) {
		const int64_t x = w*strideWidth + pw ;
		const int64_t inIndex = NCHW_IDX(n,c,y,x,inChannel,inHeight,inWidth) ;

		__vr vrin = _ve_vldu_vss(4*strideWidth,pIn+inIndex) ;

		vrout = _ve_vfmaxs_vvv(vrin,vrout) ;


	      } // windowWidth
	    } // windowHeight

	    _ve_vstu_vss(vrout, 4, pOut+outIndex) ;
	  } // outWidth
	} // outHeight
      } // channel
    } // batch
  }

  return VEDNN_SUCCESS ;
}



