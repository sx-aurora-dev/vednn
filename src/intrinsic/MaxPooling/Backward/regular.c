#include <stdint.h>
#include <float.h>

#include <stdio.h>

#include "vednn.h"

#include "velintrin.h"
#define VLEN	(256)

#define NCHW_IDX(n,c,h,w,cl,hl,wl) ((((n)*(cl)+(c))*(hl)+(h))*(wl)+(w))

vednnError_t vednnMaxPoolingBackward_regular(
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
//  const int64_t padWidth     = pParamPool->padWidth;		// must be 0
//  const int64_t padHeight    = pParamPool->padHeight;		// must be 0

  const float * restrict pGOut   = pDataGradOut;
  const float * restrict pOut    = pDataOut;
  const float * restrict pIn     = pDataIn;
  float * restrict const pGIn    = pDataGradIn ;

  {
    for(int64_t n=0; n<batch; n++) {
      for(int64_t c=0; c<outChannel; c++) {
	for(int64_t h=0; h<outHeight; h++) {
	  for(int64_t w=0; w<outWidth; w+=VLEN) {
	    const int64_t vlen = outWidth-w < VLEN ? outWidth-w : VLEN ;

	    const int64_t outIndex  = NCHW_IDX(n,c,h,w,outChannel,outHeight,outWidth) ;

	    __vr vrout  = _vel_vldu_vssl(4, pOut+outIndex, vlen) ;
	    __vr vrgout = _vel_vldu_vssl(4, pGOut+outIndex, vlen) ;

	    __vm256 vm_not_found = _vel_vfmklat_ml(vlen) ;

	    for(int64_t ph=0; ph<windowHeight; ph++) {
	      const int64_t y = h*strideHeight + ph ;

	      for(int64_t pw=0; pw<windowWidth; pw++) {
		const int64_t x = w*strideWidth + pw ;
		const int64_t inIndex = NCHW_IDX(n,c,y,x,inChannel,inHeight,inWidth) ;

		__vr vrin = _vel_vldu_vssl(4*strideWidth,pIn+inIndex, vlen) ;

		__vm256 vm_equal =  _vel_vfmkseq_mvl(_vel_vfcmps_vvvl(vrout,vrin, vlen), vlen) ;
		__vm256 vm_and   = _vel_andm_mmm(vm_equal, vm_not_found) ;
		vm_not_found = _vel_nndm_mmm(vm_equal, vm_not_found) ;

		__vr vrgin = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.f, vlen), vrgout, vm_and, vlen) ;

		_vel_vstu_vssl(vrgin, 4*strideWidth, pGIn+inIndex, vlen) ;

	      } // windowWidth
	    } // windowHeight
	  } // outWidth
	} // outHeight
	{
	  const int64_t y = outHeight*strideHeight ;
	  if( y < inHeight ) {
	    const int64_t inIndex = NCHW_IDX(n,c,y,0,inChannel,inHeight,inWidth) ;
	    for(int64_t xy=0; xy<(inHeight-y)*inWidth; xy+=VLEN) {
	      const int64_t vl = (inHeight-y)*inWidth - xy <= VLEN ? (inHeight-y)*inWidth - xy : VLEN ;
	      _vel_vstu_vssl(_vel_vbrds_vsl(0.f, vl), 4, pGIn+inIndex+xy, vl) ;
	    }
	  }
	}
      } // channel
    } // batch
  }

  return VEDNN_SUCCESS ;
}



