#include "vednnMaxPoolingForward.h"
#include <stdint.h>
#include <float.h>

#if 1
// base version.1
vednnError_t vednnMaxPoolingForward_default( VEDNN_MAXPOOLINGFWD_ARGS )
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
#define NCHW_IDX(n,c,h,w,cl,hl,wl) ((((n)*(cl)+(c))*(hl)+(h))*(wl)+(w))



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

#else
// base version.0
vednnError_t vednnMaxPoolingForward_default( VEDNN_MAXPOOLINGFWD_ARGS )
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
#define NCHW_IDX(n,c,h,w,cl,hl,wl) ((((n)*(cl)+(c))*(hl)+(h))*(wl)+(w))

    for(int64_t n=0; n<batch; n++) {
      for(int64_t c=0; c<outChannel; c++) {
        for(int64_t h=0; h<outHeight; h++) {
          for(int64_t w=0; w<outWidth; w++) {

            const int64_t outIndex = NCHW_IDX(n,c,h,w,outChannel,outHeight,outWidth) ;
            float maxValue = -FLT_MAX ;

            for(int64_t ph=0; ph<windowHeight; ph++) {
              const int64_t y = h*strideHeight - padHeight + ph ;
              if( y < 0 || inHeight <= y) continue ;

              for(int64_t pw=0; pw<windowWidth; pw++) {
                const int64_t x = w*strideWidth - padWidth + pw ;
                if( x < 0 || inWidth <= x) continue ;

                const int64_t inIndex = NCHW_IDX(n,c,y,x,inChannel,inHeight,inWidth) ;
                const float   inValue = pIn[inIndex] ;

                if( inValue > maxValue ) maxValue = inValue ;

              } // windowWidth
            } // windowHeight

            pOut[outIndex] = maxValue ;

          } // outWidth
        } // outHeight
      } // channel
    } // batch
  }

  return VEDNN_SUCCESS ;
}
#endif
// vim: et sw=2 ts=2
