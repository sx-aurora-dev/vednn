#include <stdint.h>
#include <stdlib.h>

#include "vednn.h"

#include "veintrin.h"
#define VLEN	(256)

static inline void k1(
    const float * restrict pIn,
    const float * restrict pKernel,
    float * restrict const pOut,
    const int64_t inChannel,
    const int64_t inWidth,
    const int64_t inHeight,
    const int64_t outChannel,
    const int64_t outWidth,
    const int64_t outHeight,
    const int64_t inChannelGroup,
    const int64_t outChannelGroup,
    const int64_t strideHeight,
    const int64_t strideWidth,
    const int64_t inGroupOffset,
    const int64_t outGroupOffset,
    const int64_t kernGroupOffset,
    const int64_t oPixels,
    const int64_t n,
    const int64_t k
)
{

  int64_t outIndex0 = outGroupOffset + (n * outChannel + k  ) * outHeight*outWidth ;

  for (int64_t y=0; y<outHeight; y++) {
    int64_t h = y * strideHeight ;

    for ( int64_t x0=0; x0<outWidth; x0+=VLEN) {

      const int64_t vl = outWidth - x0 < VLEN ? outWidth - x0 : VLEN ;

      _ve_lvl(vl) ;

      __vr vrsum = _ve_vbrdu_vs_f32(0.f) ;

      const float *pKerValue = pKernel + kernGroupOffset + k * inChannelGroup ;

      for (int64_t c = 0; c < inChannelGroup; c++) {
	const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;

	__vr vrin = _ve_vldu_vss(4*strideWidth,&pInChannel[h*inWidth+x0*strideWidth]) ;

	const float *pKerValue = pKernel + kernGroupOffset + (k * inChannelGroup + c)  ;

	const uint64_t kerValue01 = _ve_pack_f32p(pKerValue,
						  pKerValue+      inChannelGroup) ;

	vrsum = _ve_vfmads_vvsv(vrsum, pKerValue[0], vrin) ;
      } // inChannel

      _ve_vstu_vss(vrsum, 4, pOut+outIndex0+y*outHeight+x0) ;
    } // x
  } // y
}

static inline void k2(
    const float * restrict pIn,
    const float * restrict pKernel,
    float * restrict const pOut,
    const int64_t inChannel,
    const int64_t inWidth,
    const int64_t inHeight,
    const int64_t outChannel,
    const int64_t outWidth,
    const int64_t outHeight,
    const int64_t inChannelGroup,
    const int64_t outChannelGroup,
    const int64_t strideHeight,
    const int64_t strideWidth,
    const int64_t inGroupOffset,
    const int64_t outGroupOffset,
    const int64_t kernGroupOffset,
    const int64_t oPixels,
    const int64_t n,
    const int64_t k
)
{

  int64_t outIndex0 = outGroupOffset + (n * outChannel + k  ) * outHeight*outWidth ;
  int64_t outIndex1 = outGroupOffset + (n * outChannel + k+1) * outHeight*outWidth ;

  for (int64_t y=0; y<outHeight; y++) {
    int64_t h = y * strideHeight ;

    for ( int64_t x0=0; x0<outWidth; x0+=VLEN) {

      const int64_t vl = outWidth - x0 < VLEN ? outWidth - x0 : VLEN ;

      _ve_lvl(vl) ;

      __vr vrsum01 = _ve_pvbrd_vs_i64(0UL) ;

      const float *pKerValue = pKernel + kernGroupOffset + k * inChannelGroup ;

      for (int64_t c = 0; c < inChannelGroup; c++) {
	const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;

	__vr vrin = _ve_vldu_vss(4*strideWidth,&pInChannel[h*inWidth+x0*strideWidth]) ;
	__vr vrinP = _ve_vshf_vvvs(vrin, vrin, VE_VSHUFFLE_YUZU) ;

	const float *pKerValue = pKernel + kernGroupOffset + (k * inChannelGroup + c)  ;

	const uint64_t kerValue01 = _ve_pack_f32p(pKerValue,
						  pKerValue+      inChannelGroup) ;

	vrsum01 = _ve_pvfmad_vvsv(vrsum01, kerValue01, vrinP) ;
      } // inChannel

      _ve_vstu_vss(vrsum01, 4, pOut+outIndex0+y*outHeight+x0) ;
      _ve_vstl_vss(vrsum01, 4, pOut+outIndex1+y*outHeight+x0) ;
    } // x
  } // y
}

static inline void k4(
    const float * restrict pIn,
    const float * restrict pKernel,
    float * restrict const pOut,
    const int64_t inChannel,
    const int64_t inWidth,
    const int64_t inHeight,
    const int64_t outChannel,
    const int64_t outWidth,
    const int64_t outHeight,
    const int64_t inChannelGroup,
    const int64_t outChannelGroup,
    const int64_t strideHeight,
    const int64_t strideWidth,
    const int64_t inGroupOffset,
    const int64_t outGroupOffset,
    const int64_t kernGroupOffset,
    const int64_t oPixels,
    const int64_t n,
    const int64_t k
)
{

  int64_t outIndex0 = outGroupOffset + (n * outChannel + k  ) * outHeight*outWidth ;
  int64_t outIndex1 = outGroupOffset + (n * outChannel + k+1) * outHeight*outWidth ;
  int64_t outIndex2 = outGroupOffset + (n * outChannel + k+2) * outHeight*outWidth ;
  int64_t outIndex3 = outGroupOffset + (n * outChannel + k+3) * outHeight*outWidth ;

  for (int64_t y=0; y<outHeight; y++) {
    int64_t h = y * strideHeight ;

    for ( int64_t x0=0; x0<outWidth; x0+=VLEN) {

      const int64_t vl = outWidth - x0 < VLEN ? outWidth - x0 : VLEN ;

      _ve_lvl(vl) ;

      __vr vrsum01 = _ve_pvbrd_vs_i64(0UL) ;
      __vr vrsum23 = _ve_pvbrd_vs_i64(0UL) ;

      const float *pKerValue = pKernel + kernGroupOffset + k * inChannelGroup ;

      for (int64_t c = 0; c < inChannelGroup; c++) {
	const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;

	__vr vrin = _ve_vldu_vss(4*strideWidth,&pInChannel[h*inWidth+x0*strideWidth]) ;
	__vr vrinP = _ve_vshf_vvvs(vrin, vrin, VE_VSHUFFLE_YUZU) ;

	const float *pKerValue = pKernel + kernGroupOffset + (k * inChannelGroup + c)  ;

	const uint64_t kerValue01 = _ve_pack_f32p(pKerValue,
						  pKerValue+      inChannelGroup) ;
	const uint64_t kerValue23 = _ve_pack_f32p(pKerValue + 2 * inChannelGroup,
						  pKerValue + 3 * inChannelGroup) ;

	vrsum01 = _ve_pvfmad_vvsv(vrsum01, kerValue01, vrinP) ;
	vrsum23 = _ve_pvfmad_vvsv(vrsum23, kerValue23, vrinP) ;
      } // inChannel

      _ve_vstu_vss(vrsum01, 4, pOut+outIndex0+y*outHeight+x0) ;
      _ve_vstl_vss(vrsum01, 4, pOut+outIndex1+y*outHeight+x0) ;
      _ve_vstu_vss(vrsum23, 4, pOut+outIndex2+y*outHeight+x0) ;
      _ve_vstl_vss(vrsum23, 4, pOut+outIndex3+y*outHeight+x0) ;
    } // x
  } // y
}

static inline void k8(
    const float * restrict pIn,
    const float * restrict pKernel,
    float * restrict const pOut,
    const int64_t inChannel,
    const int64_t inWidth,
    const int64_t inHeight,
    const int64_t outChannel,
    const int64_t outWidth,
    const int64_t outHeight,
    const int64_t inChannelGroup,
    const int64_t outChannelGroup,
    const int64_t strideHeight,
    const int64_t strideWidth,
    const int64_t inGroupOffset,
    const int64_t outGroupOffset,
    const int64_t kernGroupOffset,
    const int64_t oPixels,
    const int64_t n,
    const int64_t k
)
{

  int64_t outIndex0 = outGroupOffset + (n * outChannel + k  ) * outHeight*outWidth ;
  int64_t outIndex1 = outGroupOffset + (n * outChannel + k+1) * outHeight*outWidth ;
  int64_t outIndex2 = outGroupOffset + (n * outChannel + k+2) * outHeight*outWidth ;
  int64_t outIndex3 = outGroupOffset + (n * outChannel + k+3) * outHeight*outWidth ;
  int64_t outIndex4 = outGroupOffset + (n * outChannel + k+4) * outHeight*outWidth ;
  int64_t outIndex5 = outGroupOffset + (n * outChannel + k+5) * outHeight*outWidth ;
  int64_t outIndex6 = outGroupOffset + (n * outChannel + k+6) * outHeight*outWidth ;
  int64_t outIndex7 = outGroupOffset + (n * outChannel + k+7) * outHeight*outWidth ;

  for (int64_t y=0; y<outHeight; y++) {
    int64_t h = y * strideHeight ;

    for ( int64_t x0=0; x0<outWidth; x0+=VLEN) {

      const int64_t vl = outWidth - x0 < VLEN ? outWidth - x0 : VLEN ;

      _ve_lvl(vl) ;

      __vr vrsum01 = _ve_pvbrd_vs_i64(0UL) ;
      __vr vrsum23 = _ve_pvbrd_vs_i64(0UL) ;
      __vr vrsum45 = _ve_pvbrd_vs_i64(0UL) ;
      __vr vrsum67 = _ve_pvbrd_vs_i64(0UL) ;

      const float *pKerValue = pKernel + kernGroupOffset + k * inChannelGroup ;

      for (int64_t c = 0; c < inChannelGroup; c++) {
	const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;

	__vr vrin = _ve_vldu_vss(4*strideWidth,&pInChannel[h*inWidth+x0*strideWidth]) ;
	__vr vrinP = _ve_vshf_vvvs(vrin, vrin, VE_VSHUFFLE_YUZU) ;

	const float *pKerValue = pKernel + kernGroupOffset + (k * inChannelGroup + c)  ;

	const uint64_t kerValue01 = _ve_pack_f32p(pKerValue,
						  pKerValue+      inChannelGroup) ;
	const uint64_t kerValue23 = _ve_pack_f32p(pKerValue + 2 * inChannelGroup,
						  pKerValue + 3 * inChannelGroup) ;
	const uint64_t kerValue45 = _ve_pack_f32p(pKerValue + 4 * inChannelGroup,
						  pKerValue + 5 * inChannelGroup) ;
	const uint64_t kerValue67 = _ve_pack_f32p(pKerValue + 6 * inChannelGroup,
						  pKerValue + 7 * inChannelGroup) ;

	vrsum01 = _ve_pvfmad_vvsv(vrsum01, kerValue01, vrinP) ;
	vrsum23 = _ve_pvfmad_vvsv(vrsum23, kerValue23, vrinP) ;
	vrsum45 = _ve_pvfmad_vvsv(vrsum45, kerValue45, vrinP) ;
	vrsum67 = _ve_pvfmad_vvsv(vrsum67, kerValue67, vrinP) ;
      } // inChannel

      _ve_vstu_vss(vrsum01, 4, pOut+outIndex0+y*outHeight+x0) ;
      _ve_vstl_vss(vrsum01, 4, pOut+outIndex1+y*outHeight+x0) ;
      _ve_vstu_vss(vrsum23, 4, pOut+outIndex2+y*outHeight+x0) ;
      _ve_vstl_vss(vrsum23, 4, pOut+outIndex3+y*outHeight+x0) ;
      _ve_vstu_vss(vrsum45, 4, pOut+outIndex4+y*outHeight+x0) ;
      _ve_vstl_vss(vrsum45, 4, pOut+outIndex5+y*outHeight+x0) ;
      _ve_vstu_vss(vrsum67, 4, pOut+outIndex6+y*outHeight+x0) ;
      _ve_vstl_vss(vrsum67, 4, pOut+outIndex7+y*outHeight+x0) ;
    } // x
  } // y
}

static inline void k16(
    const float * restrict pIn,
    const float * restrict pKernel,
    float * restrict const pOut,
    const int64_t inChannel,
    const int64_t inWidth,
    const int64_t inHeight,
    const int64_t outChannel,
    const int64_t outWidth,
    const int64_t outHeight,
    const int64_t inChannelGroup,
    const int64_t outChannelGroup,
    const int64_t strideHeight,
    const int64_t strideWidth,
    const int64_t inGroupOffset,
    const int64_t outGroupOffset,
    const int64_t kernGroupOffset,
    const int64_t oPixels,
    const int64_t n,
    const int64_t k
)
{

  int64_t outIndex0 = outGroupOffset + (n * outChannel + k  ) * outHeight*outWidth ;
  int64_t outIndex1 = outGroupOffset + (n * outChannel + k+1) * outHeight*outWidth ;
  int64_t outIndex2 = outGroupOffset + (n * outChannel + k+2) * outHeight*outWidth ;
  int64_t outIndex3 = outGroupOffset + (n * outChannel + k+3) * outHeight*outWidth ;
  int64_t outIndex4 = outGroupOffset + (n * outChannel + k+4) * outHeight*outWidth ;
  int64_t outIndex5 = outGroupOffset + (n * outChannel + k+5) * outHeight*outWidth ;
  int64_t outIndex6 = outGroupOffset + (n * outChannel + k+6) * outHeight*outWidth ;
  int64_t outIndex7 = outGroupOffset + (n * outChannel + k+7) * outHeight*outWidth ;
  int64_t outIndex8 = outGroupOffset + (n * outChannel + k+8) * outHeight*outWidth ;
  int64_t outIndex9 = outGroupOffset + (n * outChannel + k+9) * outHeight*outWidth ;
  int64_t outIndexA = outGroupOffset + (n * outChannel + k+10) * outHeight*outWidth ;
  int64_t outIndexB = outGroupOffset + (n * outChannel + k+11) * outHeight*outWidth ;
  int64_t outIndexC = outGroupOffset + (n * outChannel + k+12) * outHeight*outWidth ;
  int64_t outIndexD = outGroupOffset + (n * outChannel + k+13) * outHeight*outWidth ;
  int64_t outIndexE = outGroupOffset + (n * outChannel + k+14) * outHeight*outWidth ;
  int64_t outIndexF = outGroupOffset + (n * outChannel + k+15) * outHeight*outWidth ;


  for (int64_t y=0; y<outHeight; y++) {
    int64_t h = y * strideHeight ;

    for ( int64_t x0=0; x0<outWidth; x0+=VLEN) {

      const int64_t vl = outWidth - x0 < VLEN ? outWidth - x0 : VLEN ;

      _ve_lvl(vl) ;

      __vr vrsum01 = _ve_pvbrd_vs_i64(0UL) ;
      __vr vrsum23 = _ve_pvbrd_vs_i64(0UL) ;
      __vr vrsum45 = _ve_pvbrd_vs_i64(0UL) ;
      __vr vrsum67 = _ve_pvbrd_vs_i64(0UL) ;
      __vr vrsum89 = _ve_pvbrd_vs_i64(0UL) ;
      __vr vrsumAB = _ve_pvbrd_vs_i64(0UL) ;
      __vr vrsumCD = _ve_pvbrd_vs_i64(0UL) ;
      __vr vrsumEF = _ve_pvbrd_vs_i64(0UL) ;

      const float *pKerValue = pKernel + kernGroupOffset + k * inChannelGroup ;

      for (int64_t c = 0; c < inChannelGroup; c++) {
	const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;

	__vr vrin = _ve_vldu_vss(4*strideWidth,&pInChannel[h*inWidth+x0*strideWidth]) ;
	__vr vrinP = _ve_vshf_vvvs(vrin, vrin, VE_VSHUFFLE_YUZU) ;

	const float *pKerValue = pKernel + kernGroupOffset + (k * inChannelGroup + c)  ;

	const uint64_t kerValue01 = _ve_pack_f32p(pKerValue,
						  pKerValue+      inChannelGroup) ;
	const uint64_t kerValue23 = _ve_pack_f32p(pKerValue + 2 * inChannelGroup,
						  pKerValue + 3 * inChannelGroup) ;
	const uint64_t kerValue45 = _ve_pack_f32p(pKerValue + 4 * inChannelGroup,
						  pKerValue + 5 * inChannelGroup) ;
	const uint64_t kerValue67 = _ve_pack_f32p(pKerValue + 6 * inChannelGroup,
						  pKerValue + 7 * inChannelGroup) ;
	const uint64_t kerValue89 = _ve_pack_f32p(pKerValue + 8 * inChannelGroup,
						  pKerValue + 9 * inChannelGroup) ;
	const uint64_t kerValueAB = _ve_pack_f32p(pKerValue +10 * inChannelGroup,
						  pKerValue +11 * inChannelGroup) ;
	const uint64_t kerValueCD = _ve_pack_f32p(pKerValue +12 * inChannelGroup,
						  pKerValue +13 * inChannelGroup) ;
	const uint64_t kerValueEF = _ve_pack_f32p(pKerValue +14 * inChannelGroup,
						  pKerValue +15 * inChannelGroup) ;

	vrsum01 = _ve_pvfmad_vvsv(vrsum01, kerValue01, vrinP) ;
	vrsum23 = _ve_pvfmad_vvsv(vrsum23, kerValue23, vrinP) ;
	vrsum45 = _ve_pvfmad_vvsv(vrsum45, kerValue45, vrinP) ;
	vrsum67 = _ve_pvfmad_vvsv(vrsum67, kerValue67, vrinP) ;
	vrsum89 = _ve_pvfmad_vvsv(vrsum89, kerValue89, vrinP) ;
	vrsumAB = _ve_pvfmad_vvsv(vrsumAB, kerValueAB, vrinP) ;
	vrsumCD = _ve_pvfmad_vvsv(vrsumCD, kerValueCD, vrinP) ;
	vrsumEF = _ve_pvfmad_vvsv(vrsumEF, kerValueEF, vrinP) ;
      } // inChannel

      _ve_vstu_vss(vrsum01, 4, pOut+outIndex0+y*outHeight+x0) ;
      _ve_vstl_vss(vrsum01, 4, pOut+outIndex1+y*outHeight+x0) ;
      _ve_vstu_vss(vrsum23, 4, pOut+outIndex2+y*outHeight+x0) ;
      _ve_vstl_vss(vrsum23, 4, pOut+outIndex3+y*outHeight+x0) ;
      _ve_vstu_vss(vrsum45, 4, pOut+outIndex4+y*outHeight+x0) ;
      _ve_vstl_vss(vrsum45, 4, pOut+outIndex5+y*outHeight+x0) ;
      _ve_vstu_vss(vrsum67, 4, pOut+outIndex6+y*outHeight+x0) ;
      _ve_vstl_vss(vrsum67, 4, pOut+outIndex7+y*outHeight+x0) ;
      _ve_vstu_vss(vrsum89, 4, pOut+outIndex8+y*outHeight+x0) ;
      _ve_vstl_vss(vrsum89, 4, pOut+outIndex9+y*outHeight+x0) ;
      _ve_vstu_vss(vrsumAB, 4, pOut+outIndexA+y*outHeight+x0) ;
      _ve_vstl_vss(vrsumAB, 4, pOut+outIndexB+y*outHeight+x0) ;
      _ve_vstu_vss(vrsumCD, 4, pOut+outIndexC+y*outHeight+x0) ;
      _ve_vstl_vss(vrsumCD, 4, pOut+outIndexD+y*outHeight+x0) ;
      _ve_vstu_vss(vrsumEF, 4, pOut+outIndexE+y*outHeight+x0) ;
      _ve_vstl_vss(vrsumEF, 4, pOut+outIndexF+y*outHeight+x0) ;
    } // x
  } // y
}


vednnError_t
vednnConvolutionForward_direct_dil1_pad0_ker1(
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
  const int64_t kernWidth  = pParamKernel->width;
  const int64_t kernHeight = pParamKernel->height;

  const int64_t group          = pParamConv->group;
  const int64_t strideWidth    = pParamConv->strideWidth;;
  const int64_t strideHeight   = pParamConv->strideHeight;
//  const int64_t padWidth       = pParamConv->padWidth;
//  const int64_t padHeight      = pParamConv->padHeight;
//  const int64_t dilationWidth  = pParamConv->dilationWidth;
//  const int64_t dilationHeight = pParamConv->dilationHeight;

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
	const int64_t kernGroupOffset = g * outChannelGroup * inChannelGroup * kernHeight * kernWidth;

	int k = 0 ;

	if ( (outChannelGroup & 0x01) == 1 ) {
	  k1(pIn, pKernel, pOut,
	     inChannel, inWidth, inHeight,
	     outChannel, outWidth, outHeight,
	     inChannelGroup, outChannelGroup,
	     strideHeight, strideWidth,
	     inGroupOffset, outGroupOffset,
	     kernGroupOffset, oPixels, n, k) ;

	  k++ ;
	}
	if ( ((outChannelGroup >> 1) & 0x01) == 1 ) {
	  k2(pIn, pKernel, pOut,
	     inChannel, inWidth, inHeight,
	     outChannel, outWidth, outHeight,
	     inChannelGroup, outChannelGroup,
	     strideHeight, strideWidth,
	     inGroupOffset, outGroupOffset,
	     kernGroupOffset, oPixels, n, k) ;

	  k+=2 ;
	}
	if ( ((outChannelGroup >> 2) & 0x01) == 1 ) {
	  k4(pIn, pKernel, pOut,
	     inChannel, inWidth, inHeight,
	     outChannel, outWidth, outHeight,
	     inChannelGroup, outChannelGroup,
	     strideHeight, strideWidth,
	     inGroupOffset, outGroupOffset,
	     kernGroupOffset, oPixels, n, k) ;

	  k+=4 ;
	}
	if ( ((outChannelGroup >> 3) & 0x01) == 1 ) {
	  k8(pIn, pKernel, pOut,
	     inChannel, inWidth, inHeight,
	     outChannel, outWidth, outHeight,
	     inChannelGroup, outChannelGroup,
	     strideHeight, strideWidth,
	     inGroupOffset, outGroupOffset,
	     kernGroupOffset, oPixels, n, k) ;
	  k+=8 ;
	} // outChannel
	for (; k < outChannelGroup; k+=16) {
	  k16(pIn, pKernel, pOut,
	     inChannel, inWidth, inHeight,
	     outChannel, outWidth, outHeight,
	     inChannelGroup, outChannelGroup,
	     strideHeight, strideWidth,
	     inGroupOffset, outGroupOffset,
	     kernGroupOffset, oPixels, n, k) ;
	} // outChannel

      } // group
    } // batch
  }

  return VEDNN_SUCCESS;
}
