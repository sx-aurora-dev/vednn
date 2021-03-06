#include <stdint.h>
#include <float.h>

#include "vednn.h"

#define NCHW_IDX(n,c,h,w,cl,hl,wl) ((((n)*(cl)+(c))*(hl)+(h))*(wl)+(w))

vednnError_t maxpooling_forward(
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
  const int64_t padWidth     = pParamPool->padWidth;
  const int64_t padHeight    = pParamPool->padHeight;

  const float * restrict pIn     = pDataIn;
  float * restrict const pOut    = pDataOut;

  {

    for(int64_t n=0; n<batch; n++) {
      for(int64_t c=0; c<outChannel; c++) {
	// Initialize
	for(int64_t h=0; h<outHeight; h++) {
	  for(int64_t w=0; w<outWidth; w++) {
	    const int64_t outIndex = NCHW_IDX(n,c,h,w,outChannel,outHeight,outWidth) ;
	    pOut[outIndex] = -FLT_MAX ;
	  }
	}

	for(int64_t ph=0; ph<windowHeight; ph++) {
	  for(int64_t pw=0; pw<windowWidth; pw++) {
	    for(int64_t h=0; h<outHeight; h++) {
	      const int64_t y = h*strideHeight - padHeight + ph ;
	      if( y < 0 || inHeight <= y) continue ;

	      for(int64_t w=0; w<outWidth; w++) {
		const int64_t x = w*strideWidth - padWidth + pw ;
		if( x < 0 || inWidth <= x) continue ;

		const int64_t inIndex = NCHW_IDX(n,c,y,x,inChannel,inHeight,inWidth) ;
		const float   inValue = pIn[inIndex] ;

		const int64_t outIndex = NCHW_IDX(n,c,h,w,outChannel,outHeight,outWidth) ;

		if( inValue > pOut[outIndex] ) pOut[outIndex] = inValue ;

	      } // outWidth
	    } // outHeight
	  } // windowWidth
	} // windowHeight
      } // channel
    } // batch
  }

  return VEDNN_SUCCESS ;
}

vednnError_t maxpooling_backward(
    const vednnTensorParam_t 		*pParamGradOut,
    const void 				*pDataGradOut,
    const vednnTensorParam_t 		*pParamOut,
    const void 				*pDataOut,
    const vednnTensorParam_t 		*pParamIn,
    const void 				*pDataIn,
    const vednnTensorParam_t 		*pParamGradIn,
    void 				*pDataGradIn,
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
  const int64_t padWidth     = pParamPool->padWidth;
  const int64_t padHeight    = pParamPool->padHeight;

  const float * restrict pGOut   = pDataGradOut;
  const float * restrict pOut    = pDataOut;
  const float * restrict pIn     = pDataIn;
  float * restrict const pGIn    = pDataGradIn ;

  {
    int64_t nIn = batch * inChannel * inHeight * inWidth ;
    for(int64_t i=0; i<nIn; i++) {
      pGIn[nIn] = 0.f ;
    }

    for(int64_t n=0; n<batch; n++) {
      for(int64_t c=0; c<outChannel; c++) {
	for(int64_t h=0; h<outHeight; h++) {
	  for(int64_t w=0; w<outWidth; w++) {

	    const int64_t outIndex  = NCHW_IDX(n,c,h,w,outChannel,outHeight,outWidth) ;
	    const float   maxValue  = pOut[outIndex] ;
	    const float   gOutValue = pGOut[outIndex] ;

	    int found = 0 ;

	    for(int64_t ph=0; ph<windowHeight; ph++) {
	      const int64_t y = h*strideHeight - padHeight + ph ;
	      if( y < 0 || inHeight <= y) continue ;

	      for(int64_t pw=0; pw<windowWidth; pw++) {
		const int64_t x = w*strideWidth - padWidth + pw ;
		if( x < 0 || inWidth <= x) continue ;

		const int64_t inIndex = NCHW_IDX(n,c,y,x,inChannel,inHeight,inWidth) ;
		const float   inValue = pIn[inIndex] ;

		if( (found == 0)  && (inValue == maxValue) ) {
		  pGIn[inIndex] +=  gOutValue ;
		  found = 1 ;
		}
	      } // windowWidth
	    } // windowHeight
	  } // outWidth
	} // outHeight
      } // channel
    } // batch
  }

  return VEDNN_SUCCESS ;
}

