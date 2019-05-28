#include <stdint.h>
#include <stdlib.h>

#include "vednn.h"

#include "veintrin.h"
#define VLEN	(256)


vednnError_t
vednnConvolutionForward_direct_dil1_str1_pad0_ker1(
    const vednnTensorParam_t * restrict 	pParamIn,
    const void * restrict 			pDataIn,
    const vednnFilterParam_t * restrict 	pParamKernel,
    const void * restrict 			pDataKernel,
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

	  for (int64_t op = 0; op < oPixels; op+=VLEN) {
	    const int64_t vl = oPixels - op < VLEN ? oPixels - op : VLEN ;

	    _ve_lvl(vl) ;

	    __vr vrsum = _ve_vbrdu_vs_f32(0.0f) ;

	    int64_t c = 0 ;
	    if( ( inChannelGroup & 0x01 ) == 1 ) {
	      const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;

	      __vr vrin  = _ve_vldu_vss(4,&pInChannel[op]) ;
	      const float *pKerValue = pKernel + kernGroupOffset + (k * inChannelGroup + c);

#define FILTER1C(VRSUM, N)				\
{							\
  VRSUM = _ve_vfmads_vvsv(VRSUM, pKerValue[0], vrin) ;	\
}
	      FILTER1C(vrsum, 0) ; pKerValue += inChannelGroup ;
#undef FILTER1C

	      c++ ;
	    }
	    for( ; c < inChannelGroup ; c+=2 ) {
	      const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c  ) * inHeight * inWidth ) ;

	      __vr vrin0  = _ve_vldu_vss(4,&pInChannel[op]) ;
	      __vr vrin1  = _ve_vldu_vss(4,&pInChannel[op + inHeight * inWidth ]) ;

	      const float *pKerValue = pKernel + kernGroupOffset + (k * inChannelGroup + c) ;

#define FILTER2C(VRSUM, N)				\
{							\
  VRSUM = _ve_vfmads_vvsv(VRSUM, pKerValue[0], vrin0) ;	\
  VRSUM = _ve_vfmads_vvsv(VRSUM, pKerValue[1], vrin1) ;	\
}
	      FILTER2C(vrsum, 0) ; pKerValue += inChannelGroup ;
#undef FILTER2C

	    } // inChannel

	    _ve_vstu_vss(vrsum, 4, pOut+outIndex) ;

	    outIndex += vl ;
	  } // outPixels
	  k++ ;
	}
	if ( ((outChannelGroup >> 1) & 0x01) == 1 ) {
	  int64_t outIndex = outGroupOffset + (n * outChannel + k  ) * oPixels ;

	  for (int64_t op = 0; op < oPixels; op+=VLEN) {
	    const int64_t vl = oPixels - op < VLEN ? oPixels - op : VLEN ;

	    _ve_lvl(vl) ;

	    __vr vrsum01 = _ve_pvbrd_vs_i64(0UL) ;

	    int64_t c = 0 ;
	    if( ( inChannelGroup & 0x01 ) == 1 ) {
	      const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;

	      __vr vrin  = _ve_vldu_vss(4,&pInChannel[op]) ;
	      __vr vrinP = _ve_vshf_vvvs(vrin, vrin, VE_VSHUFFLE_YUZU) ;

	      const float *pKerValue = pKernel + kernGroupOffset + (k * inChannelGroup + c) ;

#define FILTER1C(VRSUM, N)							\
{										\
  const uint64_t kerValue = _ve_pack_f32p(pKerValue,				\
					    pKerValue   + inChannelGroup ) ;	\
  VRSUM = _ve_pvfmad_vvsv(VRSUM, kerValue, vrinP) ;				\
}
	      FILTER1C(vrsum01, 0) ; pKerValue += 2 * inChannelGroup ;

	      c++ ;
	    }
	    for( ; c < inChannelGroup ; c+=2 ) {
	      const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c  ) * inHeight * inWidth ) ;

	      __vr vrin0  = _ve_vldu_vss(4,&pInChannel[op]) ;
	      __vr vrin1  = _ve_vldu_vss(4,&pInChannel[op + inHeight * inWidth ]) ;
	      __vr vrin0P = _ve_vshf_vvvs(vrin0, vrin0, VE_VSHUFFLE_YUZU) ;
	      __vr vrin1P = _ve_vshf_vvvs(vrin1, vrin1, VE_VSHUFFLE_YUZU) ;

	      const float *pKerValue = pKernel + kernGroupOffset + (k * inChannelGroup + c) ;

#define FILTER2C(VRSUM, N)							\
{										\
  const uint64_t kerValue_0 = _ve_pack_f32p(pKerValue,				\
					    pKerValue   + inChannelGroup ) ;	\
  const uint64_t kerValue_1 = _ve_pack_f32p(pKerValue+1,			\
					    pKerValue+1 + inChannelGroup ) ;	\
  VRSUM = _ve_pvfmad_vvsv(VRSUM, kerValue_0, vrin0P) ;				\
  VRSUM = _ve_pvfmad_vvsv(VRSUM, kerValue_1, vrin1P) ;				\
}
	      FILTER2C(vrsum01, 0) ; pKerValue += 2 * inChannelGroup ;
#undef FILTER2C

	    } // inChannel

	    _ve_vstu_vss(vrsum01, 4, pOut+outIndex) ;
	    _ve_vstl_vss(vrsum01, 4, pOut+outIndex+   oPixels) ;

	    outIndex += vl ;
	  } // outPixels

	  k+=2 ;
	}
	if ( ((outChannelGroup >> 2) & 0x01) == 1 ) {
	  int64_t outIndex = outGroupOffset + (n * outChannel + k  ) * oPixels ;

	  for (int64_t op = 0; op < oPixels; op+=VLEN) {
	    const int64_t vl = oPixels - op < VLEN ? oPixels - op : VLEN ;

	    _ve_lvl(vl) ;

	    __vr vrsum01 = _ve_pvbrd_vs_i64(0UL) ;
	    __vr vrsum23 = _ve_pvbrd_vs_i64(0UL) ;

	    int64_t c = 0 ;
	    if( ( inChannelGroup & 0x01 ) == 1 ) {
	      const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;

	      __vr vrin  = _ve_vldu_vss(4,&pInChannel[op]) ;
	      __vr vrinP = _ve_vshf_vvvs(vrin, vrin, VE_VSHUFFLE_YUZU) ;

	      const float *pKerValue = pKernel + kernGroupOffset + (k * inChannelGroup + c) ;

#define FILTER1C(VRSUM, N)							\
{										\
  const uint64_t kerValue = _ve_pack_f32p(pKerValue,				\
					    pKerValue   + inChannelGroup ) ;	\
  VRSUM = _ve_pvfmad_vvsv(VRSUM, kerValue, vrinP) ;				\
}
	      FILTER1C(vrsum01, 0) ; pKerValue += 2 * inChannelGroup ;
	      FILTER1C(vrsum23, 2) ; pKerValue += 2 * inChannelGroup ;

	      c++ ;
	    }
	    for( ; c < inChannelGroup ; c+=2 ) {
	      const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c  ) * inHeight * inWidth ) ;

	      __vr vrin0  = _ve_vldu_vss(4,&pInChannel[op]) ;
	      __vr vrin1  = _ve_vldu_vss(4,&pInChannel[op + inHeight * inWidth ]) ;
	      __vr vrin0P = _ve_vshf_vvvs(vrin0, vrin0, VE_VSHUFFLE_YUZU) ;
	      __vr vrin1P = _ve_vshf_vvvs(vrin1, vrin1, VE_VSHUFFLE_YUZU) ;

	      const float *pKerValue = pKernel + kernGroupOffset + (k * inChannelGroup + c) ;

#define FILTER2C(VRSUM, N)							\
{										\
  const uint64_t kerValue_0 = _ve_pack_f32p(pKerValue,				\
					    pKerValue   + inChannelGroup ) ;	\
  const uint64_t kerValue_1 = _ve_pack_f32p(pKerValue+1,			\
					    pKerValue+1 + inChannelGroup ) ;	\
  VRSUM = _ve_pvfmad_vvsv(VRSUM, kerValue_0, vrin0P) ;				\
  VRSUM = _ve_pvfmad_vvsv(VRSUM, kerValue_1, vrin1P) ;				\
}
	      FILTER2C(vrsum01, 0) ; pKerValue += 2 * inChannelGroup ;
	      FILTER2C(vrsum23, 2) ; pKerValue += 2 * inChannelGroup ;
#undef FILTER2C

	    } // inChannel

	    _ve_vstu_vss(vrsum01, 4, pOut+outIndex) ;
	    _ve_vstl_vss(vrsum01, 4, pOut+outIndex+   oPixels) ;
	    _ve_vstu_vss(vrsum23, 4, pOut+outIndex+ 2*oPixels) ;
	    _ve_vstl_vss(vrsum23, 4, pOut+outIndex+ 3*oPixels) ;

	    outIndex += vl ;
	  } // outPixels

	  k+=4 ;
	}
	if ( ((outChannelGroup >> 3) & 0x01) == 1 ) {
	  int64_t outIndex = outGroupOffset + (n * outChannel + k  ) * oPixels ;

	  for (int64_t op = 0; op < oPixels; op+=VLEN) {
	    const int64_t vl = oPixels - op < VLEN ? oPixels - op : VLEN ;

	    _ve_lvl(vl) ;

	    __vr vrsum01 = _ve_pvbrd_vs_i64(0UL) ;
	    __vr vrsum23 = _ve_pvbrd_vs_i64(0UL) ;
	    __vr vrsum45 = _ve_pvbrd_vs_i64(0UL) ;
	    __vr vrsum67 = _ve_pvbrd_vs_i64(0UL) ;

	    int64_t c = 0 ;
	    if( ( inChannelGroup & 0x01 ) == 1 ) {
	      const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;

	      __vr vrin  = _ve_vldu_vss(4,&pInChannel[op]) ;
	      __vr vrinP = _ve_vshf_vvvs(vrin, vrin, VE_VSHUFFLE_YUZU) ;

	      const float *pKerValue = pKernel + kernGroupOffset + (k * inChannelGroup + c) ;

#define FILTER1C(VRSUM, N)							\
{										\
  const uint64_t kerValue = _ve_pack_f32p(pKerValue,				\
					    pKerValue   + inChannelGroup ) ;	\
  VRSUM = _ve_pvfmad_vvsv(VRSUM, kerValue, vrinP) ;				\
}
	      FILTER1C(vrsum01, 0) ; pKerValue += 2 * inChannelGroup ;
	      FILTER1C(vrsum23, 2) ; pKerValue += 2 * inChannelGroup ;
	      FILTER1C(vrsum45, 4) ; pKerValue += 2 * inChannelGroup ;
	      FILTER1C(vrsum67, 6) ; pKerValue += 2 * inChannelGroup ;

	      c++ ;
	    }
	    for( ; c < inChannelGroup ; c+=2 ) {
	      const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c  ) * inHeight * inWidth ) ;

	      __vr vrin0  = _ve_vldu_vss(4,&pInChannel[op]) ;
	      __vr vrin1  = _ve_vldu_vss(4,&pInChannel[op + inHeight * inWidth ]) ;
	      __vr vrin0P = _ve_vshf_vvvs(vrin0, vrin0, VE_VSHUFFLE_YUZU) ;
	      __vr vrin1P = _ve_vshf_vvvs(vrin1, vrin1, VE_VSHUFFLE_YUZU) ;

	      const float *pKerValue = pKernel + kernGroupOffset + (k * inChannelGroup + c) ;

#define FILTER2C(VRSUM, N)							\
{										\
  const uint64_t kerValue_0 = _ve_pack_f32p(pKerValue,				\
					    pKerValue   + inChannelGroup ) ;	\
  const uint64_t kerValue_1 = _ve_pack_f32p(pKerValue+1,			\
					    pKerValue+1 + inChannelGroup ) ;	\
  VRSUM = _ve_pvfmad_vvsv(VRSUM, kerValue_0, vrin0P) ;				\
  VRSUM = _ve_pvfmad_vvsv(VRSUM, kerValue_1, vrin1P) ;				\
}
	      FILTER2C(vrsum01, 0) ; pKerValue += 2 * inChannelGroup ;
	      FILTER2C(vrsum23, 2) ; pKerValue += 2 * inChannelGroup ;
	      FILTER2C(vrsum45, 4) ; pKerValue += 2 * inChannelGroup ;
	      FILTER2C(vrsum67, 6) ; pKerValue += 2 * inChannelGroup ;
#undef FILTER2C

	    } // inChannel


	    _ve_vstu_vss(vrsum01, 4, pOut+outIndex) ;
	    _ve_vstl_vss(vrsum01, 4, pOut+outIndex+   oPixels) ;
	    _ve_vstu_vss(vrsum23, 4, pOut+outIndex+ 2*oPixels) ;
	    _ve_vstl_vss(vrsum23, 4, pOut+outIndex+ 3*oPixels) ;
	    _ve_vstu_vss(vrsum45, 4, pOut+outIndex+ 4*oPixels) ;
	    _ve_vstl_vss(vrsum45, 4, pOut+outIndex+ 5*oPixels) ;
	    _ve_vstu_vss(vrsum67, 4, pOut+outIndex+ 6*oPixels) ;
	    _ve_vstl_vss(vrsum67, 4, pOut+outIndex+ 7*oPixels) ;

	    outIndex += vl ;
	  } // outPixels
	  k+=8 ;
	}
	for ( ; k < outChannelGroup; k+=16) {
	  int64_t outIndex = outGroupOffset + (n * outChannel + k  ) * oPixels ;

	  for (int64_t op = 0; op < oPixels; op+=VLEN) {
	    const int64_t vl = oPixels - op < VLEN ? oPixels - op : VLEN ;

	    _ve_lvl(vl) ;

	    __vr vrsum01 = _ve_pvbrd_vs_i64(0UL) ;
	    __vr vrsum23 = _ve_pvbrd_vs_i64(0UL) ;
	    __vr vrsum45 = _ve_pvbrd_vs_i64(0UL) ;
	    __vr vrsum67 = _ve_pvbrd_vs_i64(0UL) ;
	    __vr vrsum89 = _ve_pvbrd_vs_i64(0UL) ;
	    __vr vrsumAB = _ve_pvbrd_vs_i64(0UL) ;
	    __vr vrsumCD = _ve_pvbrd_vs_i64(0UL) ;
	    __vr vrsumEF = _ve_pvbrd_vs_i64(0UL) ;

	    int64_t c = 0 ;
	    if( ( inChannelGroup & 0x01 ) == 1 ) {
	      const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;

	      __vr vrin  = _ve_vldu_vss(4,&pInChannel[op]) ;
	      __vr vrinP = _ve_vshf_vvvs(vrin, vrin, VE_VSHUFFLE_YUZU) ;

	      const float *pKerValue = pKernel + kernGroupOffset + (k * inChannelGroup + c) ;

#define FILTER1C(VRSUM, N)							\
{										\
  const uint64_t kerValue = _ve_pack_f32p(pKerValue,				\
					    pKerValue   + inChannelGroup ) ;	\
  VRSUM = _ve_pvfmad_vvsv(VRSUM, kerValue, vrinP) ;				\
}
	      FILTER1C(vrsum01, 0) ; pKerValue += 2 * inChannelGroup ;
	      FILTER1C(vrsum23, 2) ; pKerValue += 2 * inChannelGroup ;
	      FILTER1C(vrsum45, 4) ; pKerValue += 2 * inChannelGroup ;
	      FILTER1C(vrsum67, 6) ; pKerValue += 2 * inChannelGroup ;
	      FILTER1C(vrsum89, 8) ; pKerValue += 2 * inChannelGroup ;
	      FILTER1C(vrsumAB, 10) ; pKerValue += 2 * inChannelGroup ;
	      FILTER1C(vrsumCD, 12) ; pKerValue += 2 * inChannelGroup ;
	      FILTER1C(vrsumEF, 14) ; pKerValue += 2 * inChannelGroup ;

	      c++ ;
	    }
	    for( ; c < inChannelGroup ; c+=2 ) {
	      const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c  ) * inHeight * inWidth ) ;

	      __vr vrin0  = _ve_vldu_vss(4,&pInChannel[op]) ;
	      __vr vrin1  = _ve_vldu_vss(4,&pInChannel[op + inHeight * inWidth ]) ;
	      __vr vrin0P = _ve_vshf_vvvs(vrin0, vrin0, VE_VSHUFFLE_YUZU) ;
	      __vr vrin1P = _ve_vshf_vvvs(vrin1, vrin1, VE_VSHUFFLE_YUZU) ;

	      const float *pKerValue = pKernel + kernGroupOffset + (k * inChannelGroup + c) ;

#define FILTER2C(VRSUM, N)							\
{										\
  const uint64_t kerValue_0 = _ve_pack_f32p(pKerValue,				\
					    pKerValue   + inChannelGroup ) ;	\
  const uint64_t kerValue_1 = _ve_pack_f32p(pKerValue+1,			\
					    pKerValue+1 + inChannelGroup ) ;	\
  VRSUM = _ve_pvfmad_vvsv(VRSUM, kerValue_0, vrin0P) ;				\
  VRSUM = _ve_pvfmad_vvsv(VRSUM, kerValue_1, vrin1P) ;				\
}
	      FILTER2C(vrsum01, 0) ; pKerValue += 2 * inChannelGroup ;
	      FILTER2C(vrsum23, 2) ; pKerValue += 2 * inChannelGroup ;
	      FILTER2C(vrsum45, 4) ; pKerValue += 2 * inChannelGroup ;
	      FILTER2C(vrsum67, 6) ; pKerValue += 2 * inChannelGroup ;
	      FILTER2C(vrsum89, 8) ; pKerValue += 2 * inChannelGroup ;
	      FILTER2C(vrsumAB, 10) ; pKerValue += 2 * inChannelGroup ;
	      FILTER2C(vrsumCD, 12) ; pKerValue += 2 * inChannelGroup ;
	      FILTER2C(vrsumEF, 14) ; pKerValue += 2 * inChannelGroup ;
#undef FILTER2C

	    } // inChannel

	    _ve_vstu_vss(vrsum01, 4, pOut+outIndex) ;
	    _ve_vstl_vss(vrsum01, 4, pOut+outIndex+   oPixels) ;
	    _ve_vstu_vss(vrsum23, 4, pOut+outIndex+ 2*oPixels) ;
	    _ve_vstl_vss(vrsum23, 4, pOut+outIndex+ 3*oPixels) ;
	    _ve_vstu_vss(vrsum45, 4, pOut+outIndex+ 4*oPixels) ;
	    _ve_vstl_vss(vrsum45, 4, pOut+outIndex+ 5*oPixels) ;
	    _ve_vstu_vss(vrsum67, 4, pOut+outIndex+ 6*oPixels) ;
	    _ve_vstl_vss(vrsum67, 4, pOut+outIndex+ 7*oPixels) ;
	    _ve_vstu_vss(vrsum89, 4, pOut+outIndex+ 8*oPixels) ;
	    _ve_vstl_vss(vrsum89, 4, pOut+outIndex+ 9*oPixels) ;
	    _ve_vstu_vss(vrsumAB, 4, pOut+outIndex+10*oPixels) ;
	    _ve_vstl_vss(vrsumAB, 4, pOut+outIndex+11*oPixels) ;
	    _ve_vstu_vss(vrsumCD, 4, pOut+outIndex+12*oPixels) ;
	    _ve_vstl_vss(vrsumCD, 4, pOut+outIndex+13*oPixels) ;
	    _ve_vstu_vss(vrsumEF, 4, pOut+outIndex+14*oPixels) ;
	    _ve_vstl_vss(vrsumEF, 4, pOut+outIndex+15*oPixels) ;

	    outIndex += vl ;
	  } // outPixels
	} // outChannel
      } // group
    } // batch
  }

  return VEDNN_SUCCESS;
}
