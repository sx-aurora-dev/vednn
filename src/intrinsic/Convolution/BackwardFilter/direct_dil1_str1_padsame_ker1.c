#include <stdint.h>
#include <stdlib.h>

#include "vednn.h"

#include "veintrin.h"
#define VLEN	(256)

vednnError_t
vednnConvolutionBackwardFilter_direct_dil1_str1_padsame_ker1(
    const vednnTensorParam_t * restrict 	pParamIn,
    const void * restrict 			pDataIn,
    const vednnTensorParam_t * restrict 	pParamGradOut,
    const void * restrict 			pDataGradOut,
    const vednnConvolutionParam_t * restrict 	pParamConv,
    const vednnFilterParam_t * restrict 	pParamGradKernel,
    void * restrict 				pDataGradKernel
#ifdef VEDNN_USE_OPENMP
    ,
    const int64_t				beginOChannel,
    const int64_t				nOChannel
#endif
)
{
  const int64_t inChannel   = pParamIn->channel;
  const int64_t inWidth     = pParamIn->width;
  const int64_t inHeight    = pParamIn->height;
  const int64_t batch       = pParamGradOut->batch;
  const int64_t gOutChannel = pParamGradOut->channel;
  const int64_t gOutWidth   = pParamGradOut->width;
  const int64_t gOutHeight  = pParamGradOut->height;
//  const int64_t gKernWidth  = pParamGradKernel->width;		/* must be 1 */
//  const int64_t gKernHeight = pParamGradKernel->height;		/* must be 1 */

  const int64_t group          = pParamConv->group;
//  const int64_t strideWidth    = pParamConv->strideWidth;	/* must be 1 */
//  const int64_t strideHeight   = pParamConv->strideHeight;	/* must be 1 */
//  const int64_t padWidth       = pParamConv->padWidth;	/* must be 1 */
//  const int64_t padHeight      = pParamConv->padHeight;	/* must be 1 */
//  const int64_t dilationWidth  = pParamConv->dilationWidth;	/* must be 1 */
//  const int64_t dilationHeight = pParamConv->dilationHeight;	/* must be 1 */

  const int64_t inChannelGroup   =  inChannel   / group;
  const int64_t gOutChannelGroup = gOutChannel  / group;

  const float * restrict pIn      = pDataIn;
  const float * restrict pGOut    = pDataGradOut;
  float * restrict const pGKernel = pDataGradKernel;

  const int gOutPixels= gOutHeight*gOutWidth ;

#ifndef VEDNN_USE_OPENMP
  const int64_t beginOChannel = 0 ;
  const int64_t nOChannel     = gOutChannelGroup ;
#endif
  {
    {
      for (int64_t g = 0; g < group; g++) {
	int64_t inGroupOffset   = g * inChannelGroup  * inHeight  * inWidth;
	int64_t outGroupOffset  = (g * gOutChannelGroup + beginOChannel) * gOutHeight * gOutWidth;
	int64_t kernGroupOffset = (g * gOutChannelGroup + beginOChannel) * inChannelGroup ;

	int64_t k = 0 ;
	if ( (nOChannel & 0x01) == 1 ) {
	  for (int64_t c=0; c<inChannelGroup; c++) {
	    const int64_t kernelIndex = kernGroupOffset + (k * inChannelGroup + c) ;

	    float sum = pGKernel[kernelIndex] ;

	    _ve_lvl(VLEN) ;
	    __vr vrsum = _ve_vbrdu_vs_f32(0.f) ;

	    for (int64_t n=0; n<batch; n++) {
	      for (int64_t gop = 0; gop < gOutPixels; gop+=VLEN) {
		const int64_t vl = gOutPixels - gop < VLEN ? gOutPixels - gop : VLEN ;

		_ve_lvl(vl) ;

		const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;
		const int64_t gOutIndex  = outGroupOffset + ((n * gOutChannel + k) * gOutHeight ) * gOutWidth + gop;

		/* memory access errors mihgt be caused (vrin) */
		__vr vrin   = _ve_vldu_vss(4,&pInChannel[gop]) ;
		__vr vrgout = _ve_vldu_vss(4, pGOut+gOutIndex) ;

		vrsum = _ve_vfmads_vvvv(vrsum, vrin, vrgout) ;

	      } // gOutPixels
	    } // batch

	    _ve_lvl(VLEN) ;
	    vrsum = _ve_vfsums_vv(vrsum) ;

	    _ve_lvl(1) ;
	    _ve_vstu_vss(vrsum,4,pGKernel+kernelIndex) ;
	  } // inChannel

	  k+=1;
	}
	if ( ((nOChannel >> 1) & 0x01) == 1 ) {
	  for (int64_t c=0; c<inChannelGroup; c++) {
	    const int64_t kernelIndex0 = kernGroupOffset + (k     * inChannelGroup + c) ;
	    const int64_t kernelIndex1 = kernGroupOffset + ((k+1) * inChannelGroup + c) ;

	    uint64_t sum01 = _ve_pack_f32p(pGKernel+kernelIndex0, pGKernel+kernelIndex1) ;

	    _ve_lvl(VLEN) ;
	    __vr vrsum01 = _ve_vbrd_vs_i64(0UL) ;

	    for (int64_t n=0; n<batch; n++) {
	      for (int64_t gop = 0; gop < gOutPixels; gop+=VLEN) {
		const int64_t vl = gOutPixels - gop < VLEN ? gOutPixels - gop : VLEN ;

		_ve_lvl(vl) ;

		const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;

		const int64_t gOutIndex = outGroupOffset + ((n * gOutChannel + k  ) * gOutHeight ) * gOutWidth + gop;

		/* memory access errors mihgt be caused (vrin) */
		__vr vrin    = _ve_vldu_vss(4,&pInChannel[gop]) ;
		__vr vrgout0 = _ve_vldu_vss(4, pGOut+gOutIndex+0*gOutPixels) ;
		__vr vrgout1 = _ve_vldu_vss(4, pGOut+gOutIndex+1*gOutPixels) ;

		__vr vrinP    = _ve_vshf_vvvs(vrin, vrin, VE_VSHUFFLE_YUZU) ;
		__vr vrgout01 = _ve_vshf_vvvs(vrgout0, vrgout1, VE_VSHUFFLE_YUZU) ;

		vrsum01 = _ve_pvfmad_vvvv(vrsum01, vrinP, vrgout01) ;
	      } // gOutPixels
	    } // batch

	    _ve_lvl(VLEN) ;
	    __vr vrsum0 = _ve_vfsums_vv(vrsum01) ;
	    __vr vrsum1 = _ve_vfsums_vv(_ve_vsll_vvs(vrsum01,32));

	    _ve_lvl(1) ;
	    _ve_vstu_vss(vrsum0,4,pGKernel+kernelIndex0) ;
	    _ve_vstu_vss(vrsum1,4,pGKernel+kernelIndex1) ;
	  } // inChannel
	  k+=2;
	}
	if ( ((nOChannel >> 2) & 0x01) == 1 ) {
	  for (int64_t c=0; c<inChannelGroup; c++) {
	    const int64_t kernelIndex0 = kernGroupOffset + (k     * inChannelGroup + c) ;
	    const int64_t kernelIndex1 = kernGroupOffset + ((k+1) * inChannelGroup + c) ;
	    const int64_t kernelIndex2 = kernGroupOffset + ((k+2) * inChannelGroup + c) ;
	    const int64_t kernelIndex3 = kernGroupOffset + ((k+3) * inChannelGroup + c) ;

	    uint64_t sum01 = _ve_pack_f32p(pGKernel+kernelIndex0, pGKernel+kernelIndex1) ;
	    uint64_t sum23 = _ve_pack_f32p(pGKernel+kernelIndex2, pGKernel+kernelIndex3) ;

	    _ve_lvl(VLEN) ;
	    __vr vrsum01 = _ve_vbrd_vs_i64(0UL) ;
	    __vr vrsum23 = _ve_vbrd_vs_i64(0UL) ;

	    for (int64_t n=0; n<batch; n++) {
	      for (int64_t gop = 0; gop < gOutPixels; gop+=VLEN) {
		const int64_t vl = gOutPixels - gop < VLEN ? gOutPixels - gop : VLEN ;

		_ve_lvl(vl) ;

		const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;

		const int64_t gOutIndex  = outGroupOffset + ((n * gOutChannel + k  ) * gOutHeight ) * gOutWidth + gop;

		/* memory access errors mihgt be caused (vrin) */
		__vr vrin   = _ve_vldu_vss(4,&pInChannel[gop]) ;
		__vr vrgout0 = _ve_vldu_vss(4, pGOut+gOutIndex+0*gOutPixels) ;
		__vr vrgout1 = _ve_vldu_vss(4, pGOut+gOutIndex+1*gOutPixels) ;
		__vr vrgout2 = _ve_vldu_vss(4, pGOut+gOutIndex+2*gOutPixels) ;
		__vr vrgout3 = _ve_vldu_vss(4, pGOut+gOutIndex+3*gOutPixels) ;

		__vr vrinP    = _ve_vshf_vvvs(vrin, vrin, VE_VSHUFFLE_YUZU) ;
		__vr vrgout01 = _ve_vshf_vvvs(vrgout0, vrgout1, VE_VSHUFFLE_YUZU) ;
		__vr vrgout23 = _ve_vshf_vvvs(vrgout2, vrgout3, VE_VSHUFFLE_YUZU) ;

		vrsum01 = _ve_pvfmad_vvvv(vrsum01, vrinP, vrgout01) ;
		vrsum23 = _ve_pvfmad_vvvv(vrsum23, vrinP, vrgout23) ;
	      } // gOutPixels
	    } // batch

	    _ve_lvl(VLEN) ;
	    __vr vrsum0 = _ve_vfsums_vv(vrsum01) ;
	    __vr vrsum1 = _ve_vfsums_vv(_ve_vsll_vvs(vrsum01,32));
	    __vr vrsum2 = _ve_vfsums_vv(vrsum23) ;
	    __vr vrsum3 = _ve_vfsums_vv(_ve_vsll_vvs(vrsum23,32));

	    _ve_lvl(1) ;
	    _ve_vstu_vss(vrsum0,4,pGKernel+kernelIndex0) ;
	    _ve_vstu_vss(vrsum1,4,pGKernel+kernelIndex1) ;
	    _ve_vstu_vss(vrsum2,4,pGKernel+kernelIndex2) ;
	    _ve_vstu_vss(vrsum3,4,pGKernel+kernelIndex3) ;

	  } // inChannel
	  k+=4;
	}
	for ( ;k<nOChannel; k+=8) {
	  for (int64_t c=0; c<inChannelGroup; c++) {
	    const int64_t kernelIndex0 = kernGroupOffset + (k     * inChannelGroup + c) ;
	    const int64_t kernelIndex1 = kernGroupOffset + ((k+1) * inChannelGroup + c) ;
	    const int64_t kernelIndex2 = kernGroupOffset + ((k+2) * inChannelGroup + c) ;
	    const int64_t kernelIndex3 = kernGroupOffset + ((k+3) * inChannelGroup + c) ;
	    const int64_t kernelIndex4 = kernGroupOffset + ((k+4) * inChannelGroup + c) ;
	    const int64_t kernelIndex5 = kernGroupOffset + ((k+5) * inChannelGroup + c) ;
	    const int64_t kernelIndex6 = kernGroupOffset + ((k+6) * inChannelGroup + c) ;
	    const int64_t kernelIndex7 = kernGroupOffset + ((k+7) * inChannelGroup + c) ;

	    uint64_t sum01 = _ve_pack_f32p(pGKernel+kernelIndex0, pGKernel+kernelIndex1) ;
	    uint64_t sum23 = _ve_pack_f32p(pGKernel+kernelIndex2, pGKernel+kernelIndex3) ;
	    uint64_t sum45 = _ve_pack_f32p(pGKernel+kernelIndex4, pGKernel+kernelIndex5) ;
	    uint64_t sum67 = _ve_pack_f32p(pGKernel+kernelIndex6, pGKernel+kernelIndex7) ;

	    _ve_lvl(VLEN) ;
	    __vr vrsum01 = _ve_vbrd_vs_i64(0UL) ;
	    __vr vrsum23 = _ve_vbrd_vs_i64(0UL) ;
	    __vr vrsum45 = _ve_vbrd_vs_i64(0UL) ;
	    __vr vrsum67 = _ve_vbrd_vs_i64(0UL) ;

	    for (int64_t n=0; n<batch; n++) {
	      for (int64_t gop = 0; gop < gOutPixels; gop+=VLEN) {
		const int64_t vl = gOutPixels - gop < VLEN ? gOutPixels - gop : VLEN ;

		_ve_lvl(vl) ;

		const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;

		const int64_t gOutIndex = outGroupOffset + ((n * gOutChannel + k  ) * gOutHeight ) * gOutWidth + gop;

		/* memory access errors mihgt be caused (vrin) */
		__vr vrin    = _ve_vldu_vss(4,&pInChannel[gop]) ;
		__vr vrgout0 = _ve_vldu_vss(4, pGOut+gOutIndex+0*gOutPixels) ;
		__vr vrgout1 = _ve_vldu_vss(4, pGOut+gOutIndex+1*gOutPixels) ;
		__vr vrgout2 = _ve_vldu_vss(4, pGOut+gOutIndex+2*gOutPixels) ;
		__vr vrgout3 = _ve_vldu_vss(4, pGOut+gOutIndex+3*gOutPixels) ;
		__vr vrgout4 = _ve_vldu_vss(4, pGOut+gOutIndex+4*gOutPixels) ;
		__vr vrgout5 = _ve_vldu_vss(4, pGOut+gOutIndex+5*gOutPixels) ;
		__vr vrgout6 = _ve_vldu_vss(4, pGOut+gOutIndex+6*gOutPixels) ;
		__vr vrgout7 = _ve_vldu_vss(4, pGOut+gOutIndex+7*gOutPixels) ;

		__vr vrinP    = _ve_vshf_vvvs(vrin, vrin, VE_VSHUFFLE_YUZU) ;
		__vr vrgout01 = _ve_vshf_vvvs(vrgout0, vrgout1, VE_VSHUFFLE_YUZU) ;
		__vr vrgout23 = _ve_vshf_vvvs(vrgout2, vrgout3, VE_VSHUFFLE_YUZU) ;
		__vr vrgout45 = _ve_vshf_vvvs(vrgout4, vrgout5, VE_VSHUFFLE_YUZU) ;
		__vr vrgout67 = _ve_vshf_vvvs(vrgout6, vrgout7, VE_VSHUFFLE_YUZU) ;

		vrsum01 = _ve_pvfmad_vvvv(vrsum01, vrinP, vrgout01) ;
		vrsum23 = _ve_pvfmad_vvvv(vrsum23, vrinP, vrgout23) ;
		vrsum45 = _ve_pvfmad_vvvv(vrsum45, vrinP, vrgout45) ;
		vrsum67 = _ve_pvfmad_vvvv(vrsum67, vrinP, vrgout67) ;

	      } // gOutPixels
	    } // batch

	    _ve_lvl(VLEN) ;
	    __vr vrsum0 = _ve_vfsums_vv(vrsum01) ;
	    __vr vrsum1 = _ve_vfsums_vv(_ve_vsll_vvs(vrsum01,32));
	    __vr vrsum2 = _ve_vfsums_vv(vrsum23) ;
	    __vr vrsum3 = _ve_vfsums_vv(_ve_vsll_vvs(vrsum23,32));
	    __vr vrsum4 = _ve_vfsums_vv(vrsum45) ;
	    __vr vrsum5 = _ve_vfsums_vv(_ve_vsll_vvs(vrsum45,32));
	    __vr vrsum6 = _ve_vfsums_vv(vrsum67) ;
	    __vr vrsum7 = _ve_vfsums_vv(_ve_vsll_vvs(vrsum67,32));

	    _ve_lvl(1) ;
	    _ve_vstu_vss(vrsum0,4,pGKernel+kernelIndex0) ;
	    _ve_vstu_vss(vrsum1,4,pGKernel+kernelIndex1) ;
	    _ve_vstu_vss(vrsum2,4,pGKernel+kernelIndex2) ;
	    _ve_vstu_vss(vrsum3,4,pGKernel+kernelIndex3) ;
	    _ve_vstu_vss(vrsum4,4,pGKernel+kernelIndex4) ;
	    _ve_vstu_vss(vrsum5,4,pGKernel+kernelIndex5) ;
	    _ve_vstu_vss(vrsum6,4,pGKernel+kernelIndex6) ;
	    _ve_vstu_vss(vrsum7,4,pGKernel+kernelIndex7) ;

	  } // inChannel
	} // outChannel
      } // group
    }
  }


  return VEDNN_SUCCESS;
}
