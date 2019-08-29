#include <stdint.h>
#include <stdlib.h>

#include "vednn.h"

#include "velintrin.h"
#define VLEN	(256)


vednnError_t
vednnConvolutionForwardAddBias_direct_dil1_str1_pad0_ker1_c1024x(
    const vednnTensorParam_t * restrict 	pParamIn,
    const void * restrict 			pDataIn,
    const vednnFilterParam_t * restrict 	pParamKernel,
    const void * restrict 			pDataKernel,
    const vednnBiasParam_t * restrict 		pParamBias,
    const void * restrict 			pDataBias,
    const vednnConvolutionParam_t * restrict 	pParamConv,
    const vednnTensorParam_t * restrict 	pParamOut,
    void * restrict 				pDataOut
)
{
  const int64_t batch      = pParamIn->batch;
  const int64_t inChannel  = pParamIn->channel;
  const int64_t inWidth    = pParamIn->width;
  const int64_t inHeight   = pParamIn->height;
  const int64_t outChannel = pParamOut->channel;
  const int64_t outWidth   = pParamOut->width;
  const int64_t outHeight  = pParamOut->height;
//  const int64_t kernWidth  = pParamKernel->width;		/* must be 1 */
//  const int64_t kernHeight = pParamKernel->height;		/* must be 1 */

  const int64_t group          = pParamConv->group;
//  const int64_t strideWidth    = pParamConv->strideWidth;	/* must be 1 */
//  const int64_t strideHeight   = pParamConv->strideHeight;	/* must be 1 */
//  const int64_t padWidth       = pParamConv->padWidth;	/* must be 0 */
//  const int64_t padHeight      = pParamConv->padHeight;	/* must be 0 */
//  const int64_t dilationWidth  = pParamConv->dilationWidth;	/* must be 1 */
//  const int64_t dilationHeight = pParamConv->dilationHeight;	/* must be 1 */

  const int64_t inChannelGroup  = inChannel  / group;   // equal to pDataKernel->inChannel
  const int64_t outChannelGroup = outChannel / group;   // equal to pDataKernel->outChannel

  const float * restrict pIn     = pDataIn;
  const float * restrict pKernel = pDataKernel;
  const float * restrict pBias   = pDataBias;
  float * restrict const pOut    = pDataOut;

  const int oPixels= outHeight*outWidth ;

  {
    for (int64_t n = 0; n < batch; n++) {
      for (int64_t g = 0; g < group; g++) {
	const int64_t inGroupOffset   = g * inChannelGroup * inHeight * inWidth;
	const int64_t outGroupOffset  = g * outChannelGroup * outHeight * outWidth;
	const int64_t kernGroupOffset = g * outChannelGroup * inChannelGroup ;

	int k = 0 ;
	if ( (outChannelGroup & 0x01) == 1 ) {
	  int64_t outIndex = outGroupOffset + (n * outChannel + k) * oPixels ;

	  const float bias = (pBias == NULL) ? 0.0 : pBias[g * outChannelGroup + k];

	  for (int64_t op = 0; op < oPixels; op+=VLEN) {
	    const int64_t vl = oPixels - op < VLEN ? oPixels - op : VLEN ;

	    __vr vrsum = _vel_vbrds_vsl(bias, vl) ;

	    for(int64_t c = 0 ; c < inChannelGroup ; c+=2 ) {
	      const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c  ) * inHeight * inWidth ) ;

	      __vr vrin0  = _vel_vldu_vssl(4,&pInChannel[op], vl) ;
	      __vr vrin1  = _vel_vldu_vssl(4,&pInChannel[op + inHeight * inWidth ], vl) ;

	      const float *pKerValue = pKernel + kernGroupOffset + (k * inChannelGroup + c) ;

#define FILTER2C(VRSUM, N)				\
{							\
  VRSUM = _vel_vfmads_vvsvl(VRSUM, pKerValue[0], vrin0, vl) ;	\
  VRSUM = _vel_vfmads_vvsvl(VRSUM, pKerValue[1], vrin1, vl) ;	\
}
	      FILTER2C(vrsum, 0) ; pKerValue += inChannelGroup ;
#undef FILTER2C

	    } // inChannel

	    _vel_vstu_vssl(vrsum, 4, pOut+outIndex, vl) ;

	    outIndex += vl ;
	  } // outPixels
	  k++ ;
	}
	if ( ((outChannelGroup >> 1) & 0x01) == 1 ) {
	  int64_t outIndex = outGroupOffset + (n * outChannel + k  ) * oPixels ;

	  const float bias0 = (pBias == NULL) ? 0.0 : pBias[g * outChannelGroup + k  ];
	  const float bias1 = (pBias == NULL) ? 0.0 : pBias[g * outChannelGroup + k+1];

	  const uint64_t bias01 = _vel_pack_f32p(&bias0, &bias1) ;

	  for (int64_t op = 0; op < oPixels; op+=VLEN) {
	    const int64_t vl = oPixels - op < VLEN ? oPixels - op : VLEN ;

	    __vr vrsum01 = _vel_pvbrd_vsl(bias01, vl) ;

	    for(int64_t c = 0 ; c < inChannelGroup ; c+=2 ) {
	      const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c  ) * inHeight * inWidth ) ;

	      __vr vrin0  = _vel_vldu_vssl(4,&pInChannel[op], vl) ;
	      __vr vrin1  = _vel_vldu_vssl(4,&pInChannel[op + inHeight * inWidth ], vl) ;
	      __vr vrin0P = _vel_vshf_vvvsl(vrin0, vrin0, VE_VSHUFFLE_YUZU, vl) ;
	      __vr vrin1P = _vel_vshf_vvvsl(vrin1, vrin1, VE_VSHUFFLE_YUZU, vl) ;

	      const float *pKerValue = pKernel + kernGroupOffset + (k * inChannelGroup + c) ;

#define FILTER2C(VRSUM, N)							\
{										\
  const uint64_t kerValue_0 = _vel_pack_f32p(pKerValue,				\
					    pKerValue   + inChannelGroup ) ;	\
  const uint64_t kerValue_1 = _vel_pack_f32p(pKerValue+1,			\
					    pKerValue+1 + inChannelGroup ) ;	\
  VRSUM = _vel_pvfmad_vvsvl(VRSUM, kerValue_0, vrin0P, vl) ;				\
  VRSUM = _vel_pvfmad_vvsvl(VRSUM, kerValue_1, vrin1P, vl) ;				\
}
	      FILTER2C(vrsum01, 0) ; pKerValue += 2 * inChannelGroup ;
#undef FILTER2C

	    } // inChannel

	    _vel_vstu_vssl(vrsum01, 4, pOut+outIndex, vl) ;
	    _vel_vstl_vssl(vrsum01, 4, pOut+outIndex+   oPixels, vl) ;

	    outIndex += vl ;
	  } // outPixels

	  k+=2 ;
	}
	if ( ((outChannelGroup >> 2) & 0x01) == 1 ) {
	  int64_t outIndex = outGroupOffset + (n * outChannel + k  ) * oPixels ;

	  const float bias0 = (pBias == NULL) ? 0.0 : pBias[g * outChannelGroup + k  ];
	  const float bias1 = (pBias == NULL) ? 0.0 : pBias[g * outChannelGroup + k+1];
	  const float bias2 = (pBias == NULL) ? 0.0 : pBias[g * outChannelGroup + k+2];
	  const float bias3 = (pBias == NULL) ? 0.0 : pBias[g * outChannelGroup + k+3];

	  const uint64_t bias01 = _vel_pack_f32p(&bias0, &bias1) ;
	  const uint64_t bias23 = _vel_pack_f32p(&bias2, &bias3) ;

	  for (int64_t op = 0; op < oPixels; op+=VLEN) {
	    const int64_t vl = oPixels - op < VLEN ? oPixels - op : VLEN ;

	    __vr vrsum01 = _vel_pvbrd_vsl(bias01, vl) ;
	    __vr vrsum23 = _vel_pvbrd_vsl(bias23, vl) ;

	    for(int64_t c = 0 ; c < inChannelGroup ; c+=2 ) {
	      const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c  ) * inHeight * inWidth ) ;

	      __vr vrin0  = _vel_vldu_vssl(4,&pInChannel[op], vl) ;
	      __vr vrin1  = _vel_vldu_vssl(4,&pInChannel[op + inHeight * inWidth ], vl) ;
	      __vr vrin0P = _vel_vshf_vvvsl(vrin0, vrin0, VE_VSHUFFLE_YUZU, vl) ;
	      __vr vrin1P = _vel_vshf_vvvsl(vrin1, vrin1, VE_VSHUFFLE_YUZU, vl) ;

	      const float *pKerValue = pKernel + kernGroupOffset + (k * inChannelGroup + c) ;

#define FILTER2C(VRSUM, N)							\
{										\
  const uint64_t kerValue_0 = _vel_pack_f32p(pKerValue,				\
					    pKerValue   + inChannelGroup ) ;	\
  const uint64_t kerValue_1 = _vel_pack_f32p(pKerValue+1,			\
					    pKerValue+1 + inChannelGroup ) ;	\
  VRSUM = _vel_pvfmad_vvsvl(VRSUM, kerValue_0, vrin0P, vl) ;				\
  VRSUM = _vel_pvfmad_vvsvl(VRSUM, kerValue_1, vrin1P, vl) ;				\
}
	      FILTER2C(vrsum01, 0) ; pKerValue += 2 * inChannelGroup ;
	      FILTER2C(vrsum23, 2) ; pKerValue += 2 * inChannelGroup ;
#undef FILTER2C

	    } // inChannel

	    _vel_vstu_vssl(vrsum01, 4, pOut+outIndex, vl) ;
	    _vel_vstl_vssl(vrsum01, 4, pOut+outIndex+   oPixels, vl) ;
	    _vel_vstu_vssl(vrsum23, 4, pOut+outIndex+ 2*oPixels, vl) ;
	    _vel_vstl_vssl(vrsum23, 4, pOut+outIndex+ 3*oPixels, vl) ;

	    outIndex += vl ;
	  } // outPixels

	  k+=4 ;
	}
	for ( ; k < outChannelGroup; k+=8) {
	  int64_t outIndex = outGroupOffset + (n * outChannel + k  ) * oPixels ;

	  const float bias0 = (pBias == NULL) ? 0.0 : pBias[g * outChannelGroup + k  ];
	  const float bias1 = (pBias == NULL) ? 0.0 : pBias[g * outChannelGroup + k+1];
	  const float bias2 = (pBias == NULL) ? 0.0 : pBias[g * outChannelGroup + k+2];
	  const float bias3 = (pBias == NULL) ? 0.0 : pBias[g * outChannelGroup + k+3];
	  const float bias4 = (pBias == NULL) ? 0.0 : pBias[g * outChannelGroup + k+4];
	  const float bias5 = (pBias == NULL) ? 0.0 : pBias[g * outChannelGroup + k+5];
	  const float bias6 = (pBias == NULL) ? 0.0 : pBias[g * outChannelGroup + k+6];
	  const float bias7 = (pBias == NULL) ? 0.0 : pBias[g * outChannelGroup + k+7];

	  const uint64_t bias01 = _vel_pack_f32p(&bias0, &bias1) ;
	  const uint64_t bias23 = _vel_pack_f32p(&bias2, &bias3) ;
	  const uint64_t bias45 = _vel_pack_f32p(&bias4, &bias5) ;
	  const uint64_t bias67 = _vel_pack_f32p(&bias6, &bias7) ;

	  for (int64_t op = 0; op < oPixels; op+=VLEN) {
	    const int64_t vl = oPixels - op < VLEN ? oPixels - op : VLEN ;

	    __vr vrsum01 = _vel_pvbrd_vsl(bias01, vl) ;
	    __vr vrsum23 = _vel_pvbrd_vsl(bias23, vl) ;
	    __vr vrsum45 = _vel_pvbrd_vsl(bias45, vl) ;
	    __vr vrsum67 = _vel_pvbrd_vsl(bias67, vl) ;

	    for(int64_t c = 0 ; c < inChannelGroup ; c+=2 ) {
	      const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c  ) * inHeight * inWidth ) ;

	      __vr vrin0  = _vel_vldu_vssl(4,&pInChannel[op], vl) ;
	      __vr vrin1  = _vel_vldu_vssl(4,&pInChannel[op + inHeight * inWidth ], vl) ;
	      __vr vrin0P = _vel_vshf_vvvsl(vrin0, vrin0, VE_VSHUFFLE_YUZU, vl) ;
	      __vr vrin1P = _vel_vshf_vvvsl(vrin1, vrin1, VE_VSHUFFLE_YUZU, vl) ;

	      const float *pKerValue = pKernel + kernGroupOffset + (k * inChannelGroup + c) ;

#define FILTER2C(VRSUM, N)							\
{										\
  const uint64_t kerValue_0 = _vel_pack_f32p(pKerValue,				\
					    pKerValue   + inChannelGroup ) ;	\
  const uint64_t kerValue_1 = _vel_pack_f32p(pKerValue+1,			\
					    pKerValue+1 + inChannelGroup ) ;	\
  VRSUM = _vel_pvfmad_vvsvl(VRSUM, kerValue_0, vrin0P, vl) ;				\
  VRSUM = _vel_pvfmad_vvsvl(VRSUM, kerValue_1, vrin1P, vl) ;				\
}
	      FILTER2C(vrsum01, 0) ; pKerValue += 2 * inChannelGroup ;
	      FILTER2C(vrsum23, 2) ; pKerValue += 2 * inChannelGroup ;
	      FILTER2C(vrsum45, 4) ; pKerValue += 2 * inChannelGroup ;
	      FILTER2C(vrsum67, 6) ; pKerValue += 2 * inChannelGroup ;
#undef FILTER2C

	    } // inChannel
	    _vel_vstu_vssl(vrsum01, 4, pOut+outIndex, vl) ;
	    _vel_vstl_vssl(vrsum01, 4, pOut+outIndex+   oPixels, vl) ;
	    _vel_vstu_vssl(vrsum23, 4, pOut+outIndex+ 2*oPixels, vl) ;
	    _vel_vstl_vssl(vrsum23, 4, pOut+outIndex+ 3*oPixels, vl) ;
	    _vel_vstu_vssl(vrsum45, 4, pOut+outIndex+ 4*oPixels, vl) ;
	    _vel_vstl_vssl(vrsum45, 4, pOut+outIndex+ 5*oPixels, vl) ;
	    _vel_vstu_vssl(vrsum67, 4, pOut+outIndex+ 6*oPixels, vl) ;
	    _vel_vstl_vssl(vrsum67, 4, pOut+outIndex+ 7*oPixels, vl) ;

	    outIndex += vl ;
	  } // outPixels
	} // outChannel
      } // group
    } // batch
  }

  return VEDNN_SUCCESS;
}
