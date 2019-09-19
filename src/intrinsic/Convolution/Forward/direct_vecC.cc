#include <stdint.h>
#include <stdlib.h>

#include "vednn.h"

#include "velintrin.h"
#define VLEN	(256)


template<filterLayout_t FLAYOUT, bool ADDBIAS, int NUMKERNEL, int Y, int X>
static inline void func(
    const float * __restrict__ pIn,
    const float * __restrict__ pKernel,
    const float * __restrict__ pBias,
    float * __restrict__ const pOut,
    const int64_t inChannel,
    const int64_t inWidth,
    const int64_t inHeight,
    const int64_t outChannel,
    const int64_t outWidth,
    const int64_t outHeight,
    const int64_t kernWidth,
    const int64_t kernHeight,
    const int64_t inChannelGroup,
    const int64_t outChannelGroup,
    const int64_t strideHeight,
    const int64_t strideWidth,
    const int64_t padHeight,
    const int64_t padWidth,
    const int64_t dilationHeight,
    const int64_t dilationWidth,
    const int64_t inGroupOffset,
    const int64_t outGroupOffset,
    const int64_t biasGroupOffset,
    const int64_t kernGroupOffset,
    const int64_t n,
    const int64_t k,
    const int64_t y,
    const int64_t x
)
{

  int64_t outIndex  = outGroupOffset + ((n * outChannel + k) * outHeight + y) * outWidth + x;

  const float bias0 = pBias[biasGroupOffset+k+ 0] ;
  const float bias1 = pBias[biasGroupOffset+k+ 1] ;
  const float bias2 = pBias[biasGroupOffset+k+ 2] ;
  const float bias3 = pBias[biasGroupOffset+k+ 3] ;
  const float bias4 = pBias[biasGroupOffset+k+ 4] ;
  const float bias5 = pBias[biasGroupOffset+k+ 5] ;
  const float bias6 = pBias[biasGroupOffset+k+ 6] ;
  const float bias7 = pBias[biasGroupOffset+k+ 7] ;
  const float bias8 = pBias[biasGroupOffset+k+ 8] ;
  const float bias9 = pBias[biasGroupOffset+k+ 9] ;
  const float biasA = pBias[biasGroupOffset+k+10] ;
  const float biasB = pBias[biasGroupOffset+k+11] ;
  const float biasC = pBias[biasGroupOffset+k+12] ;
  const float biasD = pBias[biasGroupOffset+k+13] ;
  const float biasE = pBias[biasGroupOffset+k+14] ;
  const float biasF = pBias[biasGroupOffset+k+15] ;

  __vr vrsum01_y0x0 = _vel_vbrdl_vsl(0UL, VLEN) ;
  __vr vrsum01_y0x1 = _vel_vbrdl_vsl(0UL, VLEN) ;
  __vr vrsum01_y1x0 = _vel_vbrdl_vsl(0UL, VLEN) ;
  __vr vrsum01_y1x1 = _vel_vbrdl_vsl(0UL, VLEN) ;

  __vr vrsum23_y0x0 = _vel_vbrdl_vsl(0UL, VLEN) ;
  __vr vrsum23_y0x1 = _vel_vbrdl_vsl(0UL, VLEN) ;
  __vr vrsum23_y1x0 = _vel_vbrdl_vsl(0UL, VLEN) ;
  __vr vrsum23_y1x1 = _vel_vbrdl_vsl(0UL, VLEN) ;

  __vr vrsum45_y0x0 = _vel_vbrdl_vsl(0UL, VLEN) ;
  __vr vrsum45_y0x1 = _vel_vbrdl_vsl(0UL, VLEN) ;
  __vr vrsum45_y1x0 = _vel_vbrdl_vsl(0UL, VLEN) ;
  __vr vrsum45_y1x1 = _vel_vbrdl_vsl(0UL, VLEN) ;

  __vr vrsum67_y0x0 = _vel_vbrdl_vsl(0UL, VLEN) ;
  __vr vrsum67_y0x1 = _vel_vbrdl_vsl(0UL, VLEN) ;
  __vr vrsum67_y1x0 = _vel_vbrdl_vsl(0UL, VLEN) ;
  __vr vrsum67_y1x1 = _vel_vbrdl_vsl(0UL, VLEN) ;

  __vr vrsum89_y0x0 = _vel_vbrdl_vsl(0UL, VLEN) ;
  __vr vrsum89_y0x1 = _vel_vbrdl_vsl(0UL, VLEN) ;
  __vr vrsum89_y1x0 = _vel_vbrdl_vsl(0UL, VLEN) ;
  __vr vrsum89_y1x1 = _vel_vbrdl_vsl(0UL, VLEN) ;

  __vr vrsumAB_y0x0 = _vel_vbrdl_vsl(0UL, VLEN) ;
  __vr vrsumAB_y0x1 = _vel_vbrdl_vsl(0UL, VLEN) ;
  __vr vrsumAB_y1x0 = _vel_vbrdl_vsl(0UL, VLEN) ;
  __vr vrsumAB_y1x1 = _vel_vbrdl_vsl(0UL, VLEN) ;

  __vr vrsumCD_y0x0 = _vel_vbrdl_vsl(0UL, VLEN) ;
  __vr vrsumCD_y0x1 = _vel_vbrdl_vsl(0UL, VLEN) ;
  __vr vrsumCD_y1x0 = _vel_vbrdl_vsl(0UL, VLEN) ;
  __vr vrsumCD_y1x1 = _vel_vbrdl_vsl(0UL, VLEN) ;

  __vr vrsumEF_y0x0 = _vel_vbrdl_vsl(0UL, VLEN) ;
  __vr vrsumEF_y0x1 = _vel_vbrdl_vsl(0UL, VLEN) ;
  __vr vrsumEF_y1x0 = _vel_vbrdl_vsl(0UL, VLEN) ;
  __vr vrsumEF_y1x1 = _vel_vbrdl_vsl(0UL, VLEN) ;


  for (int64_t c=0; c<inChannelGroup; c+= VLEN) {
    const int64_t vl = inChannelGroup - c < VLEN ? inChannelGroup - c : VLEN ;

    for (int64_t r=0; r<kernHeight; r++) {
      int64_t h0 = (y+0) * strideHeight - padHeight + r * dilationHeight;
      int64_t h1 = (y+1) * strideHeight - padHeight + r * dilationHeight;

      int64_t h0_valid = (Y>=1 && h0 >= 0 && h0 < inHeight ) ;
      int64_t h1_valid = (Y>=2 && h1 >= 0 && h1 < inHeight ) ;

      if( h0_valid && h1_valid ) {
	for (int64_t s=0; s<kernWidth; s++) {
	  int64_t w0 = (x+0) * strideWidth - padWidth + s * dilationWidth;
	  int64_t w1 = (x+1) * strideWidth - padWidth + s * dilationWidth;

	  int64_t w0_valid = (X>=1 && w0 >= 0 && w0 < inWidth ) ;
	  int64_t w1_valid = (X>=2 && w1 >= 0 && w1 < inWidth ) ;

	  if( w0_valid && w1_valid ) {
	    int64_t inputIndex_h0w0  = inGroupOffset + ((n * inChannel + c) * inHeight + h0) * inWidth + w0;
	    int64_t inputIndex_h0w1  = inGroupOffset + ((n * inChannel + c) * inHeight + h0) * inWidth + w1;
	    int64_t inputIndex_h1w0  = inGroupOffset + ((n * inChannel + c) * inHeight + h1) * inWidth + w0;
	    int64_t inputIndex_h1w1  = inGroupOffset + ((n * inChannel + c) * inHeight + h1) * inWidth + w1;

	    int64_t kernelIndex = kernGroupOffset + ((k * inChannelGroup + c) * kernHeight + r) * kernWidth + s;

	    __vr vri_h0w0 = _vel_vldu_vssl(4*inHeight*inWidth, &pIn[inputIndex_h0w0], vl) ;
	    __vr vri_h0w1 = _vel_vldu_vssl(4*inHeight*inWidth, &pIn[inputIndex_h0w1], vl) ;
	    __vr vri_h1w0 = _vel_vldu_vssl(4*inHeight*inWidth, &pIn[inputIndex_h1w0], vl) ;
	    __vr vri_h1w1 = _vel_vldu_vssl(4*inHeight*inWidth, &pIn[inputIndex_h1w1], vl) ;

	    const float *pKerValue = ( FLAYOUT == VEDNN_FILTER_LAYOUT_NCHW ) ?
				      pKernel + kernGroupOffset + ((k * inChannelGroup + c) * kernHeight + r) * kernWidth + s :
				      pKernel + kernGroupOffset + ( ( r * kernWidth + s ) * inChannelGroup + c ) * outChannelGroup + k ;

	    const int64_t kernelDistance = ( FLAYOUT == VEDNN_FILTER_LAYOUT_NCHW ) ?
					   inChannelGroup * kernHeight * kernWidth :
					   1 ;

	    const int64_t kernelStride = ( FLAYOUT == VEDNN_FILTER_LAYOUT_NCHW ) ?
			                 kernHeight * kernWidth :
					 outChannelGroup ;
	    __vr vrk0 = _vel_vldu_vssl(4*kernelStride, pKerValue+0*kernelDistance, vl) ;
	    __vr vrk1 = _vel_vldu_vssl(4*kernelStride, pKerValue+1*kernelDistance, vl) ;
	    __vr vrk2 = _vel_vldu_vssl(4*kernelStride, pKerValue+2*kernelDistance, vl) ;
	    __vr vrk3 = _vel_vldu_vssl(4*kernelStride, pKerValue+3*kernelDistance, vl) ;
	    __vr vrk4 = _vel_vldu_vssl(4*kernelStride, pKerValue+4*kernelDistance, vl) ;
	    __vr vrk5 = _vel_vldu_vssl(4*kernelStride, pKerValue+5*kernelDistance, vl) ;
	    __vr vrk6 = _vel_vldu_vssl(4*kernelStride, pKerValue+6*kernelDistance, vl) ;
	    __vr vrk7 = _vel_vldu_vssl(4*kernelStride, pKerValue+7*kernelDistance, vl) ;
	    __vr vrk8 = _vel_vldu_vssl(4*kernelStride, pKerValue+8*kernelDistance, vl) ;
	    __vr vrk9 = _vel_vldu_vssl(4*kernelStride, pKerValue+9*kernelDistance, vl) ;
	    __vr vrkA = _vel_vldu_vssl(4*kernelStride, pKerValue+10*kernelDistance, vl) ;
	    __vr vrkB = _vel_vldu_vssl(4*kernelStride, pKerValue+11*kernelDistance, vl) ;
	    __vr vrkC = _vel_vldu_vssl(4*kernelStride, pKerValue+12*kernelDistance, vl) ;
	    __vr vrkD = _vel_vldu_vssl(4*kernelStride, pKerValue+13*kernelDistance, vl) ;
	    __vr vrkE = _vel_vldu_vssl(4*kernelStride, pKerValue+14*kernelDistance, vl) ;
	    __vr vrkF = _vel_vldu_vssl(4*kernelStride, pKerValue+15*kernelDistance, vl) ;

	    __vr vriP_h0w0  = _vel_vshf_vvvsl(vri_h0w0, vri_h0w0, VE_VSHUFFLE_YUZU, vl) ;
	    __vr vriP_h0w1  = _vel_vshf_vvvsl(vri_h0w1, vri_h0w1, VE_VSHUFFLE_YUZU, vl) ;
	    __vr vriP_h1w0  = _vel_vshf_vvvsl(vri_h1w0, vri_h1w0, VE_VSHUFFLE_YUZU, vl) ;
	    __vr vriP_h1w1  = _vel_vshf_vvvsl(vri_h1w1, vri_h1w1, VE_VSHUFFLE_YUZU, vl) ;

	    __vr vrk01 = _vel_vshf_vvvsl(vrk0, vrk1, VE_VSHUFFLE_YUZU, vl) ;
	    vrsum01_y0x0 = _vel_pvfmad_vvvvvl(vrsum01_y0x0, vriP_h0w0, vrk01, vrsum01_y0x0, vl) ;
	    vrsum01_y0x1 = _vel_pvfmad_vvvvvl(vrsum01_y0x1, vriP_h0w1, vrk01, vrsum01_y0x1, vl) ;
	    vrsum01_y1x0 = _vel_pvfmad_vvvvvl(vrsum01_y1x0, vriP_h1w0, vrk01, vrsum01_y1x0, vl) ;
	    vrsum01_y1x1 = _vel_pvfmad_vvvvvl(vrsum01_y1x1, vriP_h1w1, vrk01, vrsum01_y1x1, vl) ;

	    __vr vrk23 = _vel_vshf_vvvsl(vrk2, vrk3, VE_VSHUFFLE_YUZU, vl) ;
	    vrsum23_y0x0 = _vel_pvfmad_vvvvvl(vrsum23_y0x0, vriP_h0w0, vrk23, vrsum23_y0x0, vl) ;
	    vrsum23_y0x1 = _vel_pvfmad_vvvvvl(vrsum23_y0x1, vriP_h0w1, vrk23, vrsum23_y0x1, vl) ;
	    vrsum23_y1x0 = _vel_pvfmad_vvvvvl(vrsum23_y1x0, vriP_h1w0, vrk23, vrsum23_y1x0, vl) ;
	    vrsum23_y1x1 = _vel_pvfmad_vvvvvl(vrsum23_y1x1, vriP_h1w1, vrk23, vrsum23_y1x1, vl) ;

	    __vr vrk45 = _vel_vshf_vvvsl(vrk4, vrk5, VE_VSHUFFLE_YUZU, vl) ;
	    vrsum45_y0x0 = _vel_pvfmad_vvvvvl(vrsum45_y0x0, vriP_h0w0, vrk45, vrsum45_y0x0, vl) ;
	    vrsum45_y0x1 = _vel_pvfmad_vvvvvl(vrsum45_y0x1, vriP_h0w1, vrk45, vrsum45_y0x1, vl) ;
	    vrsum45_y1x0 = _vel_pvfmad_vvvvvl(vrsum45_y1x0, vriP_h1w0, vrk45, vrsum45_y1x0, vl) ;
	    vrsum45_y1x1 = _vel_pvfmad_vvvvvl(vrsum45_y1x1, vriP_h1w1, vrk45, vrsum45_y1x1, vl) ;

	    __vr vrk67 = _vel_vshf_vvvsl(vrk6, vrk7, VE_VSHUFFLE_YUZU, vl) ;
	    vrsum67_y0x0 = _vel_pvfmad_vvvvvl(vrsum67_y0x0, vriP_h0w0, vrk67, vrsum67_y0x0, vl) ;
	    vrsum67_y0x1 = _vel_pvfmad_vvvvvl(vrsum67_y0x1, vriP_h0w1, vrk67, vrsum67_y0x1, vl) ;
	    vrsum67_y1x0 = _vel_pvfmad_vvvvvl(vrsum67_y1x0, vriP_h1w0, vrk67, vrsum67_y1x0, vl) ;
	    vrsum67_y1x1 = _vel_pvfmad_vvvvvl(vrsum67_y1x1, vriP_h1w1, vrk67, vrsum67_y1x1, vl) ;

	    __vr vrk89 = _vel_vshf_vvvsl(vrk8, vrk9, VE_VSHUFFLE_YUZU, vl) ;
	    vrsum89_y0x0 = _vel_pvfmad_vvvvvl(vrsum89_y0x0, vriP_h0w0, vrk89, vrsum89_y0x0, vl) ;
	    vrsum89_y0x1 = _vel_pvfmad_vvvvvl(vrsum89_y0x1, vriP_h0w1, vrk89, vrsum89_y0x1, vl) ;
	    vrsum89_y1x0 = _vel_pvfmad_vvvvvl(vrsum89_y1x0, vriP_h1w0, vrk89, vrsum89_y1x0, vl) ;
	    vrsum89_y1x1 = _vel_pvfmad_vvvvvl(vrsum89_y1x1, vriP_h1w1, vrk89, vrsum89_y1x1, vl) ;

	    __vr vrkAB = _vel_vshf_vvvsl(vrkA, vrkB, VE_VSHUFFLE_YUZU, vl) ;
	    vrsumAB_y0x0 = _vel_pvfmad_vvvvvl(vrsumAB_y0x0, vriP_h0w0, vrkAB, vrsumAB_y0x0, vl) ;
	    vrsumAB_y0x1 = _vel_pvfmad_vvvvvl(vrsumAB_y0x1, vriP_h0w1, vrkAB, vrsumAB_y0x1, vl) ;
	    vrsumAB_y1x0 = _vel_pvfmad_vvvvvl(vrsumAB_y1x0, vriP_h1w0, vrkAB, vrsumAB_y1x0, vl) ;
	    vrsumAB_y1x1 = _vel_pvfmad_vvvvvl(vrsumAB_y1x1, vriP_h1w1, vrkAB, vrsumAB_y1x1, vl) ;

	    __vr vrkCD = _vel_vshf_vvvsl(vrkC, vrkD, VE_VSHUFFLE_YUZU, vl) ;
	    vrsumCD_y0x0 = _vel_pvfmad_vvvvvl(vrsumCD_y0x0, vriP_h0w0, vrkCD, vrsumCD_y0x0, vl) ;
	    vrsumCD_y0x1 = _vel_pvfmad_vvvvvl(vrsumCD_y0x1, vriP_h0w1, vrkCD, vrsumCD_y0x1, vl) ;
	    vrsumCD_y1x0 = _vel_pvfmad_vvvvvl(vrsumCD_y1x0, vriP_h1w0, vrkCD, vrsumCD_y1x0, vl) ;
	    vrsumCD_y1x1 = _vel_pvfmad_vvvvvl(vrsumCD_y1x1, vriP_h1w1, vrkCD, vrsumCD_y1x1, vl) ;

	    __vr vrkEF = _vel_vshf_vvvsl(vrkE, vrkF, VE_VSHUFFLE_YUZU, vl) ;
	    vrsumEF_y0x0 = _vel_pvfmad_vvvvvl(vrsumEF_y0x0, vriP_h0w0, vrkEF, vrsumEF_y0x0, vl) ;
	    vrsumEF_y0x1 = _vel_pvfmad_vvvvvl(vrsumEF_y0x1, vriP_h0w1, vrkEF, vrsumEF_y0x1, vl) ;
	    vrsumEF_y1x0 = _vel_pvfmad_vvvvvl(vrsumEF_y1x0, vriP_h1w0, vrkEF, vrsumEF_y1x0, vl) ;
	    vrsumEF_y1x1 = _vel_pvfmad_vvvvvl(vrsumEF_y1x1, vriP_h1w1, vrkEF, vrsumEF_y1x1, vl) ;
	  }
	  else if( w0_valid ) {
	    int64_t inputIndex_h0w0  = inGroupOffset + ((n * inChannel + c) * inHeight + h0) * inWidth + w0;
	    int64_t inputIndex_h1w0  = inGroupOffset + ((n * inChannel + c) * inHeight + h1) * inWidth + w0;

	    int64_t kernelIndex = kernGroupOffset + ((k * inChannelGroup + c) * kernHeight + r) * kernWidth + s;

	    __vr vri_h0w0 = _vel_vldu_vssl(4*inHeight*inWidth, &pIn[inputIndex_h0w0], vl) ;
	    __vr vri_h1w0 = _vel_vldu_vssl(4*inHeight*inWidth, &pIn[inputIndex_h1w0], vl) ;

	    const float *pKerValue = ( FLAYOUT == VEDNN_FILTER_LAYOUT_NCHW ) ?
				      pKernel + kernGroupOffset + ((k * inChannelGroup + c) * kernHeight + r) * kernWidth + s :
				      pKernel + kernGroupOffset + ( ( r * kernWidth + s ) * inChannelGroup + c ) * outChannelGroup + k ;

	    const int64_t kernelDistance = ( FLAYOUT == VEDNN_FILTER_LAYOUT_NCHW ) ?
					   inChannelGroup * kernHeight * kernWidth :
					   1 ;

	    const int64_t kernelStride = ( FLAYOUT == VEDNN_FILTER_LAYOUT_NCHW ) ?
			                 kernHeight * kernWidth :
					 outChannelGroup ;
	    __vr vrk0 = _vel_vldu_vssl(4*kernelStride, pKerValue+0*kernelDistance, vl) ;
	    __vr vrk1 = _vel_vldu_vssl(4*kernelStride, pKerValue+1*kernelDistance, vl) ;
	    __vr vrk2 = _vel_vldu_vssl(4*kernelStride, pKerValue+2*kernelDistance, vl) ;
	    __vr vrk3 = _vel_vldu_vssl(4*kernelStride, pKerValue+3*kernelDistance, vl) ;
	    __vr vrk4 = _vel_vldu_vssl(4*kernelStride, pKerValue+4*kernelDistance, vl) ;
	    __vr vrk5 = _vel_vldu_vssl(4*kernelStride, pKerValue+5*kernelDistance, vl) ;
	    __vr vrk6 = _vel_vldu_vssl(4*kernelStride, pKerValue+6*kernelDistance, vl) ;
	    __vr vrk7 = _vel_vldu_vssl(4*kernelStride, pKerValue+7*kernelDistance, vl) ;
	    __vr vrk8 = _vel_vldu_vssl(4*kernelStride, pKerValue+8*kernelDistance, vl) ;
	    __vr vrk9 = _vel_vldu_vssl(4*kernelStride, pKerValue+9*kernelDistance, vl) ;
	    __vr vrkA = _vel_vldu_vssl(4*kernelStride, pKerValue+10*kernelDistance, vl) ;
	    __vr vrkB = _vel_vldu_vssl(4*kernelStride, pKerValue+11*kernelDistance, vl) ;
	    __vr vrkC = _vel_vldu_vssl(4*kernelStride, pKerValue+12*kernelDistance, vl) ;
	    __vr vrkD = _vel_vldu_vssl(4*kernelStride, pKerValue+13*kernelDistance, vl) ;
	    __vr vrkE = _vel_vldu_vssl(4*kernelStride, pKerValue+14*kernelDistance, vl) ;
	    __vr vrkF = _vel_vldu_vssl(4*kernelStride, pKerValue+15*kernelDistance, vl) ;

	    __vr vriP_h0w0  = _vel_vshf_vvvsl(vri_h0w0, vri_h0w0, VE_VSHUFFLE_YUZU, vl) ;
	    __vr vriP_h1w0  = _vel_vshf_vvvsl(vri_h1w0, vri_h1w0, VE_VSHUFFLE_YUZU, vl) ;

	    __vr vrk01 = _vel_vshf_vvvsl(vrk0, vrk1, VE_VSHUFFLE_YUZU, vl) ;
	    vrsum01_y0x0 = _vel_pvfmad_vvvvvl(vrsum01_y0x0, vriP_h0w0, vrk01, vrsum01_y0x0, vl) ;
	    vrsum01_y1x0 = _vel_pvfmad_vvvvvl(vrsum01_y1x0, vriP_h1w0, vrk01, vrsum01_y1x0, vl) ;

	    __vr vrk23 = _vel_vshf_vvvsl(vrk2, vrk3, VE_VSHUFFLE_YUZU, vl) ;
	    vrsum23_y0x0 = _vel_pvfmad_vvvvvl(vrsum23_y0x0, vriP_h0w0, vrk23, vrsum23_y0x0, vl) ;
	    vrsum23_y1x0 = _vel_pvfmad_vvvvvl(vrsum23_y1x0, vriP_h1w0, vrk23, vrsum23_y1x0, vl) ;

	    __vr vrk45 = _vel_vshf_vvvsl(vrk4, vrk5, VE_VSHUFFLE_YUZU, vl) ;
	    vrsum45_y0x0 = _vel_pvfmad_vvvvvl(vrsum45_y0x0, vriP_h0w0, vrk45, vrsum45_y0x0, vl) ;
	    vrsum45_y1x0 = _vel_pvfmad_vvvvvl(vrsum45_y1x0, vriP_h1w0, vrk45, vrsum45_y1x0, vl) ;

	    __vr vrk67 = _vel_vshf_vvvsl(vrk6, vrk7, VE_VSHUFFLE_YUZU, vl) ;
	    vrsum67_y0x0 = _vel_pvfmad_vvvvvl(vrsum67_y0x0, vriP_h0w0, vrk67, vrsum67_y0x0, vl) ;
	    vrsum67_y1x0 = _vel_pvfmad_vvvvvl(vrsum67_y1x0, vriP_h1w0, vrk67, vrsum67_y1x0, vl) ;

	    __vr vrk89 = _vel_vshf_vvvsl(vrk8, vrk9, VE_VSHUFFLE_YUZU, vl) ;
	    vrsum89_y0x0 = _vel_pvfmad_vvvvvl(vrsum89_y0x0, vriP_h0w0, vrk89, vrsum89_y0x0, vl) ;
	    vrsum89_y1x0 = _vel_pvfmad_vvvvvl(vrsum89_y1x0, vriP_h1w0, vrk89, vrsum89_y1x0, vl) ;

	    __vr vrkAB = _vel_vshf_vvvsl(vrkA, vrkB, VE_VSHUFFLE_YUZU, vl) ;
	    vrsumAB_y0x0 = _vel_pvfmad_vvvvvl(vrsumAB_y0x0, vriP_h0w0, vrkAB, vrsumAB_y0x0, vl) ;
	    vrsumAB_y1x0 = _vel_pvfmad_vvvvvl(vrsumAB_y1x0, vriP_h1w0, vrkAB, vrsumAB_y1x0, vl) ;

	    __vr vrkCD = _vel_vshf_vvvsl(vrkC, vrkD, VE_VSHUFFLE_YUZU, vl) ;
	    vrsumCD_y0x0 = _vel_pvfmad_vvvvvl(vrsumCD_y0x0, vriP_h0w0, vrkCD, vrsumCD_y0x0, vl) ;
	    vrsumCD_y1x0 = _vel_pvfmad_vvvvvl(vrsumCD_y1x0, vriP_h1w0, vrkCD, vrsumCD_y1x0, vl) ;

	    __vr vrkEF = _vel_vshf_vvvsl(vrkE, vrkF, VE_VSHUFFLE_YUZU, vl) ;
	    vrsumEF_y0x0 = _vel_pvfmad_vvvvvl(vrsumEF_y0x0, vriP_h0w0, vrkEF, vrsumEF_y0x0, vl) ;
	    vrsumEF_y1x0 = _vel_pvfmad_vvvvvl(vrsumEF_y1x0, vriP_h1w0, vrkEF, vrsumEF_y1x0, vl) ;
	  }
	  else if( w1_valid ) {
	    int64_t inputIndex_h0w1  = inGroupOffset + ((n * inChannel + c) * inHeight + h0) * inWidth + w1;
	    int64_t inputIndex_h1w1  = inGroupOffset + ((n * inChannel + c) * inHeight + h1) * inWidth + w1;

	    int64_t kernelIndex = kernGroupOffset + ((k * inChannelGroup + c) * kernHeight + r) * kernWidth + s;

	    __vr vri_h0w1 = _vel_vldu_vssl(4*inHeight*inWidth, &pIn[inputIndex_h0w1], vl) ;
	    __vr vri_h1w1 = _vel_vldu_vssl(4*inHeight*inWidth, &pIn[inputIndex_h1w1], vl) ;

	    const float *pKerValue = ( FLAYOUT == VEDNN_FILTER_LAYOUT_NCHW ) ?
				      pKernel + kernGroupOffset + ((k * inChannelGroup + c) * kernHeight + r) * kernWidth + s :
				      pKernel + kernGroupOffset + ( ( r * kernWidth + s ) * inChannelGroup + c ) * outChannelGroup + k ;

	    const int64_t kernelDistance = ( FLAYOUT == VEDNN_FILTER_LAYOUT_NCHW ) ?
					   inChannelGroup * kernHeight * kernWidth :
					   1 ;

	    const int64_t kernelStride = ( FLAYOUT == VEDNN_FILTER_LAYOUT_NCHW ) ?
			                 kernHeight * kernWidth :
					 outChannelGroup ;
	    __vr vrk0 = _vel_vldu_vssl(4*kernelStride, pKerValue+0*kernelDistance, vl) ;
	    __vr vrk1 = _vel_vldu_vssl(4*kernelStride, pKerValue+1*kernelDistance, vl) ;
	    __vr vrk2 = _vel_vldu_vssl(4*kernelStride, pKerValue+2*kernelDistance, vl) ;
	    __vr vrk3 = _vel_vldu_vssl(4*kernelStride, pKerValue+3*kernelDistance, vl) ;
	    __vr vrk4 = _vel_vldu_vssl(4*kernelStride, pKerValue+4*kernelDistance, vl) ;
	    __vr vrk5 = _vel_vldu_vssl(4*kernelStride, pKerValue+5*kernelDistance, vl) ;
	    __vr vrk6 = _vel_vldu_vssl(4*kernelStride, pKerValue+6*kernelDistance, vl) ;
	    __vr vrk7 = _vel_vldu_vssl(4*kernelStride, pKerValue+7*kernelDistance, vl) ;
	    __vr vrk8 = _vel_vldu_vssl(4*kernelStride, pKerValue+8*kernelDistance, vl) ;
	    __vr vrk9 = _vel_vldu_vssl(4*kernelStride, pKerValue+9*kernelDistance, vl) ;
	    __vr vrkA = _vel_vldu_vssl(4*kernelStride, pKerValue+10*kernelDistance, vl) ;
	    __vr vrkB = _vel_vldu_vssl(4*kernelStride, pKerValue+11*kernelDistance, vl) ;
	    __vr vrkC = _vel_vldu_vssl(4*kernelStride, pKerValue+12*kernelDistance, vl) ;
	    __vr vrkD = _vel_vldu_vssl(4*kernelStride, pKerValue+13*kernelDistance, vl) ;
	    __vr vrkE = _vel_vldu_vssl(4*kernelStride, pKerValue+14*kernelDistance, vl) ;
	    __vr vrkF = _vel_vldu_vssl(4*kernelStride, pKerValue+15*kernelDistance, vl) ;

	    __vr vriP_h0w1  = _vel_vshf_vvvsl(vri_h0w1, vri_h0w1, VE_VSHUFFLE_YUZU, vl) ;
	    __vr vriP_h1w1  = _vel_vshf_vvvsl(vri_h1w1, vri_h1w1, VE_VSHUFFLE_YUZU, vl) ;

	    __vr vrk01 = _vel_vshf_vvvsl(vrk0, vrk1, VE_VSHUFFLE_YUZU, vl) ;
	    vrsum01_y0x1 = _vel_pvfmad_vvvvvl(vrsum01_y0x1, vriP_h0w1, vrk01, vrsum01_y0x1, vl) ;
	    vrsum01_y1x1 = _vel_pvfmad_vvvvvl(vrsum01_y1x1, vriP_h1w1, vrk01, vrsum01_y1x1, vl) ;

	    __vr vrk23 = _vel_vshf_vvvsl(vrk2, vrk3, VE_VSHUFFLE_YUZU, vl) ;
	    vrsum23_y0x1 = _vel_pvfmad_vvvvvl(vrsum23_y0x1, vriP_h0w1, vrk23, vrsum23_y0x1, vl) ;
	    vrsum23_y1x1 = _vel_pvfmad_vvvvvl(vrsum23_y1x1, vriP_h1w1, vrk23, vrsum23_y1x1, vl) ;

	    __vr vrk45 = _vel_vshf_vvvsl(vrk4, vrk5, VE_VSHUFFLE_YUZU, vl) ;
	    vrsum45_y0x1 = _vel_pvfmad_vvvvvl(vrsum45_y0x1, vriP_h0w1, vrk45, vrsum45_y0x1, vl) ;
	    vrsum45_y1x1 = _vel_pvfmad_vvvvvl(vrsum45_y1x1, vriP_h1w1, vrk45, vrsum45_y1x1, vl) ;

	    __vr vrk67 = _vel_vshf_vvvsl(vrk6, vrk7, VE_VSHUFFLE_YUZU, vl) ;
	    vrsum67_y0x1 = _vel_pvfmad_vvvvvl(vrsum67_y0x1, vriP_h0w1, vrk67, vrsum67_y0x1, vl) ;
	    vrsum67_y1x1 = _vel_pvfmad_vvvvvl(vrsum67_y1x1, vriP_h1w1, vrk67, vrsum67_y1x1, vl) ;

	    __vr vrk89 = _vel_vshf_vvvsl(vrk8, vrk9, VE_VSHUFFLE_YUZU, vl) ;
	    vrsum89_y0x1 = _vel_pvfmad_vvvvvl(vrsum89_y0x1, vriP_h0w1, vrk89, vrsum89_y0x1, vl) ;
	    vrsum89_y1x1 = _vel_pvfmad_vvvvvl(vrsum89_y1x1, vriP_h1w1, vrk89, vrsum89_y1x1, vl) ;

	    __vr vrkAB = _vel_vshf_vvvsl(vrkA, vrkB, VE_VSHUFFLE_YUZU, vl) ;
	    vrsumAB_y0x1 = _vel_pvfmad_vvvvvl(vrsumAB_y0x1, vriP_h0w1, vrkAB, vrsumAB_y0x1, vl) ;
	    vrsumAB_y1x1 = _vel_pvfmad_vvvvvl(vrsumAB_y1x1, vriP_h1w1, vrkAB, vrsumAB_y1x1, vl) ;

	    __vr vrkCD = _vel_vshf_vvvsl(vrkC, vrkD, VE_VSHUFFLE_YUZU, vl) ;
	    vrsumCD_y0x1 = _vel_pvfmad_vvvvvl(vrsumCD_y0x1, vriP_h0w1, vrkCD, vrsumCD_y0x1, vl) ;
	    vrsumCD_y1x1 = _vel_pvfmad_vvvvvl(vrsumCD_y1x1, vriP_h1w1, vrkCD, vrsumCD_y1x1, vl) ;

	    __vr vrkEF = _vel_vshf_vvvsl(vrkE, vrkF, VE_VSHUFFLE_YUZU, vl) ;
	    vrsumEF_y0x1 = _vel_pvfmad_vvvvvl(vrsumEF_y0x1, vriP_h0w1, vrkEF, vrsumEF_y0x1, vl) ;
	    vrsumEF_y1x1 = _vel_pvfmad_vvvvvl(vrsumEF_y1x1, vriP_h1w1, vrkEF, vrsumEF_y1x1, vl) ;
	  }
	} // kernWidth
      }
      else if( h0_valid  ) {
	for (int64_t s=0; s<kernWidth; s++) {
	  int64_t w0 = (x+0) * strideWidth - padWidth + s * dilationWidth;
	  int64_t w1 = (x+1) * strideWidth - padWidth + s * dilationWidth;

	  int64_t w0_valid = ( w0 >= 0 && w0 < inWidth ) ;
	  int64_t w1_valid = ( w1 >= 0 && w1 < inWidth ) ;

	  if( w0_valid && w1_valid ) {
	    int64_t inputIndex_h0w0  = inGroupOffset + ((n * inChannel + c) * inHeight + h0) * inWidth + w0;
	    int64_t inputIndex_h0w1  = inGroupOffset + ((n * inChannel + c) * inHeight + h0) * inWidth + w1;

	    int64_t kernelIndex = kernGroupOffset + ((k * inChannelGroup + c) * kernHeight + r) * kernWidth + s;

	    __vr vri_h0w0 = _vel_vldu_vssl(4*inHeight*inWidth, &pIn[inputIndex_h0w0], vl) ;
	    __vr vri_h0w1 = _vel_vldu_vssl(4*inHeight*inWidth, &pIn[inputIndex_h0w1], vl) ;

	    const float *pKerValue = ( FLAYOUT == VEDNN_FILTER_LAYOUT_NCHW ) ?
				      pKernel + kernGroupOffset + ((k * inChannelGroup + c) * kernHeight + r) * kernWidth + s :
				      pKernel + kernGroupOffset + ( ( r * kernWidth + s ) * inChannelGroup + c ) * outChannelGroup + k ;

	    const int64_t kernelDistance = ( FLAYOUT == VEDNN_FILTER_LAYOUT_NCHW ) ?
					   inChannelGroup * kernHeight * kernWidth :
					   1 ;

	    const int64_t kernelStride = ( FLAYOUT == VEDNN_FILTER_LAYOUT_NCHW ) ?
			                 kernHeight * kernWidth :
					 outChannelGroup ;
	    __vr vrk0 = _vel_vldu_vssl(4*kernelStride, pKerValue+0*kernelDistance, vl) ;
	    __vr vrk1 = _vel_vldu_vssl(4*kernelStride, pKerValue+1*kernelDistance, vl) ;
	    __vr vrk2 = _vel_vldu_vssl(4*kernelStride, pKerValue+2*kernelDistance, vl) ;
	    __vr vrk3 = _vel_vldu_vssl(4*kernelStride, pKerValue+3*kernelDistance, vl) ;
	    __vr vrk4 = _vel_vldu_vssl(4*kernelStride, pKerValue+4*kernelDistance, vl) ;
	    __vr vrk5 = _vel_vldu_vssl(4*kernelStride, pKerValue+5*kernelDistance, vl) ;
	    __vr vrk6 = _vel_vldu_vssl(4*kernelStride, pKerValue+6*kernelDistance, vl) ;
	    __vr vrk7 = _vel_vldu_vssl(4*kernelStride, pKerValue+7*kernelDistance, vl) ;
	    __vr vrk8 = _vel_vldu_vssl(4*kernelStride, pKerValue+8*kernelDistance, vl) ;
	    __vr vrk9 = _vel_vldu_vssl(4*kernelStride, pKerValue+9*kernelDistance, vl) ;
	    __vr vrkA = _vel_vldu_vssl(4*kernelStride, pKerValue+10*kernelDistance, vl) ;
	    __vr vrkB = _vel_vldu_vssl(4*kernelStride, pKerValue+11*kernelDistance, vl) ;
	    __vr vrkC = _vel_vldu_vssl(4*kernelStride, pKerValue+12*kernelDistance, vl) ;
	    __vr vrkD = _vel_vldu_vssl(4*kernelStride, pKerValue+13*kernelDistance, vl) ;
	    __vr vrkE = _vel_vldu_vssl(4*kernelStride, pKerValue+14*kernelDistance, vl) ;
	    __vr vrkF = _vel_vldu_vssl(4*kernelStride, pKerValue+15*kernelDistance, vl) ;

	    __vr vriP_h0w0  = _vel_vshf_vvvsl(vri_h0w0, vri_h0w0, VE_VSHUFFLE_YUZU, vl) ;
	    __vr vriP_h0w1  = _vel_vshf_vvvsl(vri_h0w1, vri_h0w1, VE_VSHUFFLE_YUZU, vl) ;

	    __vr vrk01 = _vel_vshf_vvvsl(vrk0, vrk1, VE_VSHUFFLE_YUZU, vl) ;
	    vrsum01_y0x0 = _vel_pvfmad_vvvvvl(vrsum01_y0x0, vriP_h0w0, vrk01, vrsum01_y0x0, vl) ;
	    vrsum01_y0x1 = _vel_pvfmad_vvvvvl(vrsum01_y0x1, vriP_h0w1, vrk01, vrsum01_y0x1, vl) ;

	    __vr vrk23 = _vel_vshf_vvvsl(vrk2, vrk3, VE_VSHUFFLE_YUZU, vl) ;
	    vrsum23_y0x0 = _vel_pvfmad_vvvvvl(vrsum23_y0x0, vriP_h0w0, vrk23, vrsum23_y0x0, vl) ;
	    vrsum23_y0x1 = _vel_pvfmad_vvvvvl(vrsum23_y0x1, vriP_h0w1, vrk23, vrsum23_y0x1, vl) ;

	    __vr vrk45 = _vel_vshf_vvvsl(vrk4, vrk5, VE_VSHUFFLE_YUZU, vl) ;
	    vrsum45_y0x0 = _vel_pvfmad_vvvvvl(vrsum45_y0x0, vriP_h0w0, vrk45, vrsum45_y0x0, vl) ;
	    vrsum45_y0x1 = _vel_pvfmad_vvvvvl(vrsum45_y0x1, vriP_h0w1, vrk45, vrsum45_y0x1, vl) ;

	    __vr vrk67 = _vel_vshf_vvvsl(vrk6, vrk7, VE_VSHUFFLE_YUZU, vl) ;
	    vrsum67_y0x0 = _vel_pvfmad_vvvvvl(vrsum67_y0x0, vriP_h0w0, vrk67, vrsum67_y0x0, vl) ;
	    vrsum67_y0x1 = _vel_pvfmad_vvvvvl(vrsum67_y0x1, vriP_h0w1, vrk67, vrsum67_y0x1, vl) ;

	    __vr vrk89 = _vel_vshf_vvvsl(vrk8, vrk9, VE_VSHUFFLE_YUZU, vl) ;
	    vrsum89_y0x0 = _vel_pvfmad_vvvvvl(vrsum89_y0x0, vriP_h0w0, vrk89, vrsum89_y0x0, vl) ;
	    vrsum89_y0x1 = _vel_pvfmad_vvvvvl(vrsum89_y0x1, vriP_h0w1, vrk89, vrsum89_y0x1, vl) ;

	    __vr vrkAB = _vel_vshf_vvvsl(vrkA, vrkB, VE_VSHUFFLE_YUZU, vl) ;
	    vrsumAB_y0x0 = _vel_pvfmad_vvvvvl(vrsumAB_y0x0, vriP_h0w0, vrkAB, vrsumAB_y0x0, vl) ;
	    vrsumAB_y0x1 = _vel_pvfmad_vvvvvl(vrsumAB_y0x1, vriP_h0w1, vrkAB, vrsumAB_y0x1, vl) ;

	    __vr vrkCD = _vel_vshf_vvvsl(vrkC, vrkD, VE_VSHUFFLE_YUZU, vl) ;
	    vrsumCD_y0x0 = _vel_pvfmad_vvvvvl(vrsumCD_y0x0, vriP_h0w0, vrkCD, vrsumCD_y0x0, vl) ;
	    vrsumCD_y0x1 = _vel_pvfmad_vvvvvl(vrsumCD_y0x1, vriP_h0w1, vrkCD, vrsumCD_y0x1, vl) ;

	    __vr vrkEF = _vel_vshf_vvvsl(vrkE, vrkF, VE_VSHUFFLE_YUZU, vl) ;
	    vrsumEF_y0x0 = _vel_pvfmad_vvvvvl(vrsumEF_y0x0, vriP_h0w0, vrkEF, vrsumEF_y0x0, vl) ;
	    vrsumEF_y0x1 = _vel_pvfmad_vvvvvl(vrsumEF_y0x1, vriP_h0w1, vrkEF, vrsumEF_y0x1, vl) ;
	  }
	  else if( w0_valid ) {
	    int64_t inputIndex_h0w0  = inGroupOffset + ((n * inChannel + c) * inHeight + h0) * inWidth + w0;

	    int64_t kernelIndex = kernGroupOffset + ((k * inChannelGroup + c) * kernHeight + r) * kernWidth + s;

	    __vr vri_h0w0 = _vel_vldu_vssl(4*inHeight*inWidth, &pIn[inputIndex_h0w0], vl) ;

	    const float *pKerValue = ( FLAYOUT == VEDNN_FILTER_LAYOUT_NCHW ) ?
				      pKernel + kernGroupOffset + ((k * inChannelGroup + c) * kernHeight + r) * kernWidth + s :
				      pKernel + kernGroupOffset + ( ( r * kernWidth + s ) * inChannelGroup + c ) * outChannelGroup + k ;

	    const int64_t kernelDistance = ( FLAYOUT == VEDNN_FILTER_LAYOUT_NCHW ) ?
					   inChannelGroup * kernHeight * kernWidth :
					   1 ;

	    const int64_t kernelStride = ( FLAYOUT == VEDNN_FILTER_LAYOUT_NCHW ) ?
			                 kernHeight * kernWidth :
					 outChannelGroup ;
	    __vr vrk0 = _vel_vldu_vssl(4*kernelStride, pKerValue+0*kernelDistance, vl) ;
	    __vr vrk1 = _vel_vldu_vssl(4*kernelStride, pKerValue+1*kernelDistance, vl) ;
	    __vr vrk2 = _vel_vldu_vssl(4*kernelStride, pKerValue+2*kernelDistance, vl) ;
	    __vr vrk3 = _vel_vldu_vssl(4*kernelStride, pKerValue+3*kernelDistance, vl) ;
	    __vr vrk4 = _vel_vldu_vssl(4*kernelStride, pKerValue+4*kernelDistance, vl) ;
	    __vr vrk5 = _vel_vldu_vssl(4*kernelStride, pKerValue+5*kernelDistance, vl) ;
	    __vr vrk6 = _vel_vldu_vssl(4*kernelStride, pKerValue+6*kernelDistance, vl) ;
	    __vr vrk7 = _vel_vldu_vssl(4*kernelStride, pKerValue+7*kernelDistance, vl) ;
	    __vr vrk8 = _vel_vldu_vssl(4*kernelStride, pKerValue+8*kernelDistance, vl) ;
	    __vr vrk9 = _vel_vldu_vssl(4*kernelStride, pKerValue+9*kernelDistance, vl) ;
	    __vr vrkA = _vel_vldu_vssl(4*kernelStride, pKerValue+10*kernelDistance, vl) ;
	    __vr vrkB = _vel_vldu_vssl(4*kernelStride, pKerValue+11*kernelDistance, vl) ;
	    __vr vrkC = _vel_vldu_vssl(4*kernelStride, pKerValue+12*kernelDistance, vl) ;
	    __vr vrkD = _vel_vldu_vssl(4*kernelStride, pKerValue+13*kernelDistance, vl) ;
	    __vr vrkE = _vel_vldu_vssl(4*kernelStride, pKerValue+14*kernelDistance, vl) ;
	    __vr vrkF = _vel_vldu_vssl(4*kernelStride, pKerValue+15*kernelDistance, vl) ;

	    __vr vriP_h0w0  = _vel_vshf_vvvsl(vri_h0w0, vri_h0w0, VE_VSHUFFLE_YUZU, vl) ;

	    __vr vrk01 = _vel_vshf_vvvsl(vrk0, vrk1, VE_VSHUFFLE_YUZU, vl) ;
	    vrsum01_y0x0 = _vel_pvfmad_vvvvvl(vrsum01_y0x0, vriP_h0w0, vrk01, vrsum01_y0x0, vl) ;

	    __vr vrk23 = _vel_vshf_vvvsl(vrk2, vrk3, VE_VSHUFFLE_YUZU, vl) ;
	    vrsum23_y0x0 = _vel_pvfmad_vvvvvl(vrsum23_y0x0, vriP_h0w0, vrk23, vrsum23_y0x0, vl) ;

	    __vr vrk45 = _vel_vshf_vvvsl(vrk4, vrk5, VE_VSHUFFLE_YUZU, vl) ;
	    vrsum45_y0x0 = _vel_pvfmad_vvvvvl(vrsum45_y0x0, vriP_h0w0, vrk45, vrsum45_y0x0, vl) ;

	    __vr vrk67 = _vel_vshf_vvvsl(vrk6, vrk7, VE_VSHUFFLE_YUZU, vl) ;
	    vrsum67_y0x0 = _vel_pvfmad_vvvvvl(vrsum67_y0x0, vriP_h0w0, vrk67, vrsum67_y0x0, vl) ;

	    __vr vrk89 = _vel_vshf_vvvsl(vrk8, vrk9, VE_VSHUFFLE_YUZU, vl) ;
	    vrsum89_y0x0 = _vel_pvfmad_vvvvvl(vrsum89_y0x0, vriP_h0w0, vrk89, vrsum89_y0x0, vl) ;

	    __vr vrkAB = _vel_vshf_vvvsl(vrkA, vrkB, VE_VSHUFFLE_YUZU, vl) ;
	    vrsumAB_y0x0 = _vel_pvfmad_vvvvvl(vrsumAB_y0x0, vriP_h0w0, vrkAB, vrsumAB_y0x0, vl) ;

	    __vr vrkCD = _vel_vshf_vvvsl(vrkC, vrkD, VE_VSHUFFLE_YUZU, vl) ;
	    vrsumCD_y0x0 = _vel_pvfmad_vvvvvl(vrsumCD_y0x0, vriP_h0w0, vrkCD, vrsumCD_y0x0, vl) ;

	    __vr vrkEF = _vel_vshf_vvvsl(vrkE, vrkF, VE_VSHUFFLE_YUZU, vl) ;
	    vrsumEF_y0x0 = _vel_pvfmad_vvvvvl(vrsumEF_y0x0, vriP_h0w0, vrkEF, vrsumEF_y0x0, vl) ;
	  }
	  else if( w1_valid ) {
	    int64_t inputIndex_h0w1  = inGroupOffset + ((n * inChannel + c) * inHeight + h0) * inWidth + w1;

	    int64_t kernelIndex = kernGroupOffset + ((k * inChannelGroup + c) * kernHeight + r) * kernWidth + s;

	    __vr vri_h0w1 = _vel_vldu_vssl(4*inHeight*inWidth, &pIn[inputIndex_h0w1], vl) ;

	    const float *pKerValue = ( FLAYOUT == VEDNN_FILTER_LAYOUT_NCHW ) ?
				      pKernel + kernGroupOffset + ((k * inChannelGroup + c) * kernHeight + r) * kernWidth + s :
				      pKernel + kernGroupOffset + ( ( r * kernWidth + s ) * inChannelGroup + c ) * outChannelGroup + k ;

	    const int64_t kernelDistance = ( FLAYOUT == VEDNN_FILTER_LAYOUT_NCHW ) ?
					   inChannelGroup * kernHeight * kernWidth :
					   1 ;

	    const int64_t kernelStride = ( FLAYOUT == VEDNN_FILTER_LAYOUT_NCHW ) ?
			                 kernHeight * kernWidth :
					 outChannelGroup ;
	    __vr vrk0 = _vel_vldu_vssl(4*kernelStride, pKerValue+0*kernelDistance, vl) ;
	    __vr vrk1 = _vel_vldu_vssl(4*kernelStride, pKerValue+1*kernelDistance, vl) ;
	    __vr vrk2 = _vel_vldu_vssl(4*kernelStride, pKerValue+2*kernelDistance, vl) ;
	    __vr vrk3 = _vel_vldu_vssl(4*kernelStride, pKerValue+3*kernelDistance, vl) ;
	    __vr vrk4 = _vel_vldu_vssl(4*kernelStride, pKerValue+4*kernelDistance, vl) ;
	    __vr vrk5 = _vel_vldu_vssl(4*kernelStride, pKerValue+5*kernelDistance, vl) ;
	    __vr vrk6 = _vel_vldu_vssl(4*kernelStride, pKerValue+6*kernelDistance, vl) ;
	    __vr vrk7 = _vel_vldu_vssl(4*kernelStride, pKerValue+7*kernelDistance, vl) ;
	    __vr vrk8 = _vel_vldu_vssl(4*kernelStride, pKerValue+8*kernelDistance, vl) ;
	    __vr vrk9 = _vel_vldu_vssl(4*kernelStride, pKerValue+9*kernelDistance, vl) ;
	    __vr vrkA = _vel_vldu_vssl(4*kernelStride, pKerValue+10*kernelDistance, vl) ;
	    __vr vrkB = _vel_vldu_vssl(4*kernelStride, pKerValue+11*kernelDistance, vl) ;
	    __vr vrkC = _vel_vldu_vssl(4*kernelStride, pKerValue+12*kernelDistance, vl) ;
	    __vr vrkD = _vel_vldu_vssl(4*kernelStride, pKerValue+13*kernelDistance, vl) ;
	    __vr vrkE = _vel_vldu_vssl(4*kernelStride, pKerValue+14*kernelDistance, vl) ;
	    __vr vrkF = _vel_vldu_vssl(4*kernelStride, pKerValue+15*kernelDistance, vl) ;

	    __vr vriP_h0w1  = _vel_vshf_vvvsl(vri_h0w1, vri_h0w1, VE_VSHUFFLE_YUZU, vl) ;

	    __vr vrk01 = _vel_vshf_vvvsl(vrk0, vrk1, VE_VSHUFFLE_YUZU, vl) ;
	    vrsum01_y0x1 = _vel_pvfmad_vvvvvl(vrsum01_y0x1, vriP_h0w1, vrk01, vrsum01_y0x1, vl) ;

	    __vr vrk23 = _vel_vshf_vvvsl(vrk2, vrk3, VE_VSHUFFLE_YUZU, vl) ;
	    vrsum23_y0x1 = _vel_pvfmad_vvvvvl(vrsum23_y0x1, vriP_h0w1, vrk23, vrsum23_y0x1, vl) ;

	    __vr vrk45 = _vel_vshf_vvvsl(vrk4, vrk5, VE_VSHUFFLE_YUZU, vl) ;
	    vrsum45_y0x1 = _vel_pvfmad_vvvvvl(vrsum45_y0x1, vriP_h0w1, vrk45, vrsum45_y0x1, vl) ;

	    __vr vrk67 = _vel_vshf_vvvsl(vrk6, vrk7, VE_VSHUFFLE_YUZU, vl) ;
	    vrsum67_y0x1 = _vel_pvfmad_vvvvvl(vrsum67_y0x1, vriP_h0w1, vrk67, vrsum67_y0x1, vl) ;

	    __vr vrk89 = _vel_vshf_vvvsl(vrk8, vrk9, VE_VSHUFFLE_YUZU, vl) ;
	    vrsum89_y0x1 = _vel_pvfmad_vvvvvl(vrsum89_y0x1, vriP_h0w1, vrk89, vrsum89_y0x1, vl) ;

	    __vr vrkAB = _vel_vshf_vvvsl(vrkA, vrkB, VE_VSHUFFLE_YUZU, vl) ;
	    vrsumAB_y0x1 = _vel_pvfmad_vvvvvl(vrsumAB_y0x1, vriP_h0w1, vrkAB, vrsumAB_y0x1, vl) ;

	    __vr vrkCD = _vel_vshf_vvvsl(vrkC, vrkD, VE_VSHUFFLE_YUZU, vl) ;
	    vrsumCD_y0x1 = _vel_pvfmad_vvvvvl(vrsumCD_y0x1, vriP_h0w1, vrkCD, vrsumCD_y0x1, vl) ;

	    __vr vrkEF = _vel_vshf_vvvsl(vrkE, vrkF, VE_VSHUFFLE_YUZU, vl) ;
	    vrsumEF_y0x1 = _vel_pvfmad_vvvvvl(vrsumEF_y0x1, vriP_h0w1, vrkEF, vrsumEF_y0x1, vl) ;
	  }
	} // kernWidth
      }
      else if( h1_valid ) {
	for (int64_t s=0; s<kernWidth; s++) {
	  int64_t w0 = (x+0) * strideWidth - padWidth + s * dilationWidth;
	  int64_t w1 = (x+1) * strideWidth - padWidth + s * dilationWidth;

	  int64_t w0_valid = ( w0 >= 0 && w0 < inWidth ) ;
	  int64_t w1_valid = ( w1 >= 0 && w1 < inWidth ) ;

	  if( w0_valid && w1_valid ) {
	    int64_t inputIndex_h1w0  = inGroupOffset + ((n * inChannel + c) * inHeight + h1) * inWidth + w0;
	    int64_t inputIndex_h1w1  = inGroupOffset + ((n * inChannel + c) * inHeight + h1) * inWidth + w1;

	    int64_t kernelIndex = kernGroupOffset + ((k * inChannelGroup + c) * kernHeight + r) * kernWidth + s;

	    __vr vri_h1w0 = _vel_vldu_vssl(4*inHeight*inWidth, &pIn[inputIndex_h1w0], vl) ;
	    __vr vri_h1w1 = _vel_vldu_vssl(4*inHeight*inWidth, &pIn[inputIndex_h1w1], vl) ;

	    const float *pKerValue = ( FLAYOUT == VEDNN_FILTER_LAYOUT_NCHW ) ?
				      pKernel + kernGroupOffset + ((k * inChannelGroup + c) * kernHeight + r) * kernWidth + s :
				      pKernel + kernGroupOffset + ( ( r * kernWidth + s ) * inChannelGroup + c ) * outChannelGroup + k ;

	    const int64_t kernelDistance = ( FLAYOUT == VEDNN_FILTER_LAYOUT_NCHW ) ?
					   inChannelGroup * kernHeight * kernWidth :
					   1 ;

	    const int64_t kernelStride = ( FLAYOUT == VEDNN_FILTER_LAYOUT_NCHW ) ?
			                 kernHeight * kernWidth :
					 outChannelGroup ;
	    __vr vrk0 = _vel_vldu_vssl(4*kernelStride, pKerValue+0*kernelDistance, vl) ;
	    __vr vrk1 = _vel_vldu_vssl(4*kernelStride, pKerValue+1*kernelDistance, vl) ;
	    __vr vrk2 = _vel_vldu_vssl(4*kernelStride, pKerValue+2*kernelDistance, vl) ;
	    __vr vrk3 = _vel_vldu_vssl(4*kernelStride, pKerValue+3*kernelDistance, vl) ;
	    __vr vrk4 = _vel_vldu_vssl(4*kernelStride, pKerValue+4*kernelDistance, vl) ;
	    __vr vrk5 = _vel_vldu_vssl(4*kernelStride, pKerValue+5*kernelDistance, vl) ;
	    __vr vrk6 = _vel_vldu_vssl(4*kernelStride, pKerValue+6*kernelDistance, vl) ;
	    __vr vrk7 = _vel_vldu_vssl(4*kernelStride, pKerValue+7*kernelDistance, vl) ;
	    __vr vrk8 = _vel_vldu_vssl(4*kernelStride, pKerValue+8*kernelDistance, vl) ;
	    __vr vrk9 = _vel_vldu_vssl(4*kernelStride, pKerValue+9*kernelDistance, vl) ;
	    __vr vrkA = _vel_vldu_vssl(4*kernelStride, pKerValue+10*kernelDistance, vl) ;
	    __vr vrkB = _vel_vldu_vssl(4*kernelStride, pKerValue+11*kernelDistance, vl) ;
	    __vr vrkC = _vel_vldu_vssl(4*kernelStride, pKerValue+12*kernelDistance, vl) ;
	    __vr vrkD = _vel_vldu_vssl(4*kernelStride, pKerValue+13*kernelDistance, vl) ;
	    __vr vrkE = _vel_vldu_vssl(4*kernelStride, pKerValue+14*kernelDistance, vl) ;
	    __vr vrkF = _vel_vldu_vssl(4*kernelStride, pKerValue+15*kernelDistance, vl) ;

	    __vr vriP_h1w0  = _vel_vshf_vvvsl(vri_h1w0, vri_h1w0, VE_VSHUFFLE_YUZU, vl) ;
	    __vr vriP_h1w1  = _vel_vshf_vvvsl(vri_h1w1, vri_h1w1, VE_VSHUFFLE_YUZU, vl) ;

	    __vr vrk01 = _vel_vshf_vvvsl(vrk0, vrk1, VE_VSHUFFLE_YUZU, vl) ;
	    vrsum01_y1x0 = _vel_pvfmad_vvvvvl(vrsum01_y1x0, vriP_h1w0, vrk01, vrsum01_y1x0, vl) ;
	    vrsum01_y1x1 = _vel_pvfmad_vvvvvl(vrsum01_y1x1, vriP_h1w1, vrk01, vrsum01_y1x1, vl) ;

	    __vr vrk23 = _vel_vshf_vvvsl(vrk2, vrk3, VE_VSHUFFLE_YUZU, vl) ;
	    vrsum23_y1x0 = _vel_pvfmad_vvvvvl(vrsum23_y1x0, vriP_h1w0, vrk23, vrsum23_y1x0, vl) ;
	    vrsum23_y1x1 = _vel_pvfmad_vvvvvl(vrsum23_y1x1, vriP_h1w1, vrk23, vrsum23_y1x1, vl) ;

	    __vr vrk45 = _vel_vshf_vvvsl(vrk4, vrk5, VE_VSHUFFLE_YUZU, vl) ;
	    vrsum45_y1x0 = _vel_pvfmad_vvvvvl(vrsum45_y1x0, vriP_h1w0, vrk45, vrsum45_y1x0, vl) ;
	    vrsum45_y1x1 = _vel_pvfmad_vvvvvl(vrsum45_y1x1, vriP_h1w1, vrk45, vrsum45_y1x1, vl) ;

	    __vr vrk67 = _vel_vshf_vvvsl(vrk6, vrk7, VE_VSHUFFLE_YUZU, vl) ;
	    vrsum67_y1x0 = _vel_pvfmad_vvvvvl(vrsum67_y1x0, vriP_h1w0, vrk67, vrsum67_y1x0, vl) ;
	    vrsum67_y1x1 = _vel_pvfmad_vvvvvl(vrsum67_y1x1, vriP_h1w1, vrk67, vrsum67_y1x1, vl) ;

	    __vr vrk89 = _vel_vshf_vvvsl(vrk8, vrk9, VE_VSHUFFLE_YUZU, vl) ;
	    vrsum89_y1x0 = _vel_pvfmad_vvvvvl(vrsum89_y1x0, vriP_h1w0, vrk89, vrsum89_y1x0, vl) ;
	    vrsum89_y1x1 = _vel_pvfmad_vvvvvl(vrsum89_y1x1, vriP_h1w1, vrk89, vrsum89_y1x1, vl) ;

	    __vr vrkAB = _vel_vshf_vvvsl(vrkA, vrkB, VE_VSHUFFLE_YUZU, vl) ;
	    vrsumAB_y1x0 = _vel_pvfmad_vvvvvl(vrsumAB_y1x0, vriP_h1w0, vrkAB, vrsumAB_y1x0, vl) ;
	    vrsumAB_y1x1 = _vel_pvfmad_vvvvvl(vrsumAB_y1x1, vriP_h1w1, vrkAB, vrsumAB_y1x1, vl) ;

	    __vr vrkCD = _vel_vshf_vvvsl(vrkC, vrkD, VE_VSHUFFLE_YUZU, vl) ;
	    vrsumCD_y1x0 = _vel_pvfmad_vvvvvl(vrsumCD_y1x0, vriP_h1w0, vrkCD, vrsumCD_y1x0, vl) ;
	    vrsumCD_y1x1 = _vel_pvfmad_vvvvvl(vrsumCD_y1x1, vriP_h1w1, vrkCD, vrsumCD_y1x1, vl) ;

	    __vr vrkEF = _vel_vshf_vvvsl(vrkE, vrkF, VE_VSHUFFLE_YUZU, vl) ;
	    vrsumEF_y1x0 = _vel_pvfmad_vvvvvl(vrsumEF_y1x0, vriP_h1w0, vrkEF, vrsumEF_y1x0, vl) ;
	    vrsumEF_y1x1 = _vel_pvfmad_vvvvvl(vrsumEF_y1x1, vriP_h1w1, vrkEF, vrsumEF_y1x1, vl) ;
	  }
	  else if( w0_valid ) {
	    int64_t inputIndex_h1w0  = inGroupOffset + ((n * inChannel + c) * inHeight + h1) * inWidth + w0;

	    int64_t kernelIndex = kernGroupOffset + ((k * inChannelGroup + c) * kernHeight + r) * kernWidth + s;

	    __vr vri_h1w0 = _vel_vldu_vssl(4*inHeight*inWidth, &pIn[inputIndex_h1w0], vl) ;

	    const float *pKerValue = ( FLAYOUT == VEDNN_FILTER_LAYOUT_NCHW ) ?
				      pKernel + kernGroupOffset + ((k * inChannelGroup + c) * kernHeight + r) * kernWidth + s :
				      pKernel + kernGroupOffset + ( ( r * kernWidth + s ) * inChannelGroup + c ) * outChannelGroup + k ;

	    const int64_t kernelDistance = ( FLAYOUT == VEDNN_FILTER_LAYOUT_NCHW ) ?
					   inChannelGroup * kernHeight * kernWidth :
					   1 ;

	    const int64_t kernelStride = ( FLAYOUT == VEDNN_FILTER_LAYOUT_NCHW ) ?
			                 kernHeight * kernWidth :
					 outChannelGroup ;
	    __vr vrk0 = _vel_vldu_vssl(4*kernelStride, pKerValue+0*kernelDistance, vl) ;
	    __vr vrk1 = _vel_vldu_vssl(4*kernelStride, pKerValue+1*kernelDistance, vl) ;
	    __vr vrk2 = _vel_vldu_vssl(4*kernelStride, pKerValue+2*kernelDistance, vl) ;
	    __vr vrk3 = _vel_vldu_vssl(4*kernelStride, pKerValue+3*kernelDistance, vl) ;
	    __vr vrk4 = _vel_vldu_vssl(4*kernelStride, pKerValue+4*kernelDistance, vl) ;
	    __vr vrk5 = _vel_vldu_vssl(4*kernelStride, pKerValue+5*kernelDistance, vl) ;
	    __vr vrk6 = _vel_vldu_vssl(4*kernelStride, pKerValue+6*kernelDistance, vl) ;
	    __vr vrk7 = _vel_vldu_vssl(4*kernelStride, pKerValue+7*kernelDistance, vl) ;
	    __vr vrk8 = _vel_vldu_vssl(4*kernelStride, pKerValue+8*kernelDistance, vl) ;
	    __vr vrk9 = _vel_vldu_vssl(4*kernelStride, pKerValue+9*kernelDistance, vl) ;
	    __vr vrkA = _vel_vldu_vssl(4*kernelStride, pKerValue+10*kernelDistance, vl) ;
	    __vr vrkB = _vel_vldu_vssl(4*kernelStride, pKerValue+11*kernelDistance, vl) ;
	    __vr vrkC = _vel_vldu_vssl(4*kernelStride, pKerValue+12*kernelDistance, vl) ;
	    __vr vrkD = _vel_vldu_vssl(4*kernelStride, pKerValue+13*kernelDistance, vl) ;
	    __vr vrkE = _vel_vldu_vssl(4*kernelStride, pKerValue+14*kernelDistance, vl) ;
	    __vr vrkF = _vel_vldu_vssl(4*kernelStride, pKerValue+15*kernelDistance, vl) ;

	    __vr vriP_h1w0  = _vel_vshf_vvvsl(vri_h1w0, vri_h1w0, VE_VSHUFFLE_YUZU, vl) ;

	    __vr vrk01 = _vel_vshf_vvvsl(vrk0, vrk1, VE_VSHUFFLE_YUZU, vl) ;
	    vrsum01_y1x0 = _vel_pvfmad_vvvvvl(vrsum01_y1x0, vriP_h1w0, vrk01, vrsum01_y1x0, vl) ;

	    __vr vrk23 = _vel_vshf_vvvsl(vrk2, vrk3, VE_VSHUFFLE_YUZU, vl) ;
	    vrsum23_y1x0 = _vel_pvfmad_vvvvvl(vrsum23_y1x0, vriP_h1w0, vrk23, vrsum23_y1x0, vl) ;

	    __vr vrk45 = _vel_vshf_vvvsl(vrk4, vrk5, VE_VSHUFFLE_YUZU, vl) ;
	    vrsum45_y1x0 = _vel_pvfmad_vvvvvl(vrsum45_y1x0, vriP_h1w0, vrk45, vrsum45_y1x0, vl) ;

	    __vr vrk67 = _vel_vshf_vvvsl(vrk6, vrk7, VE_VSHUFFLE_YUZU, vl) ;
	    vrsum67_y1x0 = _vel_pvfmad_vvvvvl(vrsum67_y1x0, vriP_h1w0, vrk67, vrsum67_y1x0, vl) ;

	    __vr vrk89 = _vel_vshf_vvvsl(vrk8, vrk9, VE_VSHUFFLE_YUZU, vl) ;
	    vrsum89_y1x0 = _vel_pvfmad_vvvvvl(vrsum89_y1x0, vriP_h1w0, vrk89, vrsum89_y1x0, vl) ;

	    __vr vrkAB = _vel_vshf_vvvsl(vrkA, vrkB, VE_VSHUFFLE_YUZU, vl) ;
	    vrsumAB_y1x0 = _vel_pvfmad_vvvvvl(vrsumAB_y1x0, vriP_h1w0, vrkAB, vrsumAB_y1x0, vl) ;

	    __vr vrkCD = _vel_vshf_vvvsl(vrkC, vrkD, VE_VSHUFFLE_YUZU, vl) ;
	    vrsumCD_y1x0 = _vel_pvfmad_vvvvvl(vrsumCD_y1x0, vriP_h1w0, vrkCD, vrsumCD_y1x0, vl) ;

	    __vr vrkEF = _vel_vshf_vvvsl(vrkE, vrkF, VE_VSHUFFLE_YUZU, vl) ;
	    vrsumEF_y1x0 = _vel_pvfmad_vvvvvl(vrsumEF_y1x0, vriP_h1w0, vrkEF, vrsumEF_y1x0, vl) ;
	  }
	  else if( w1_valid ) {
	    int64_t inputIndex_h1w1  = inGroupOffset + ((n * inChannel + c) * inHeight + h1) * inWidth + w1;

	    int64_t kernelIndex = kernGroupOffset + ((k * inChannelGroup + c) * kernHeight + r) * kernWidth + s;

	    __vr vri_h1w1 = _vel_vldu_vssl(4*inHeight*inWidth, &pIn[inputIndex_h1w1], vl) ;

	    const float *pKerValue = ( FLAYOUT == VEDNN_FILTER_LAYOUT_NCHW ) ?
				      pKernel + kernGroupOffset + ((k * inChannelGroup + c) * kernHeight + r) * kernWidth + s :
				      pKernel + kernGroupOffset + ( ( r * kernWidth + s ) * inChannelGroup + c ) * outChannelGroup + k ;

	    const int64_t kernelDistance = ( FLAYOUT == VEDNN_FILTER_LAYOUT_NCHW ) ?
					   inChannelGroup * kernHeight * kernWidth :
					   1 ;

	    const int64_t kernelStride = ( FLAYOUT == VEDNN_FILTER_LAYOUT_NCHW ) ?
			                 kernHeight * kernWidth :
					 outChannelGroup ;
	    __vr vrk0 = _vel_vldu_vssl(4*kernelStride, pKerValue+0*kernelDistance, vl) ;
	    __vr vrk1 = _vel_vldu_vssl(4*kernelStride, pKerValue+1*kernelDistance, vl) ;
	    __vr vrk2 = _vel_vldu_vssl(4*kernelStride, pKerValue+2*kernelDistance, vl) ;
	    __vr vrk3 = _vel_vldu_vssl(4*kernelStride, pKerValue+3*kernelDistance, vl) ;
	    __vr vrk4 = _vel_vldu_vssl(4*kernelStride, pKerValue+4*kernelDistance, vl) ;
	    __vr vrk5 = _vel_vldu_vssl(4*kernelStride, pKerValue+5*kernelDistance, vl) ;
	    __vr vrk6 = _vel_vldu_vssl(4*kernelStride, pKerValue+6*kernelDistance, vl) ;
	    __vr vrk7 = _vel_vldu_vssl(4*kernelStride, pKerValue+7*kernelDistance, vl) ;
	    __vr vrk8 = _vel_vldu_vssl(4*kernelStride, pKerValue+8*kernelDistance, vl) ;
	    __vr vrk9 = _vel_vldu_vssl(4*kernelStride, pKerValue+9*kernelDistance, vl) ;
	    __vr vrkA = _vel_vldu_vssl(4*kernelStride, pKerValue+10*kernelDistance, vl) ;
	    __vr vrkB = _vel_vldu_vssl(4*kernelStride, pKerValue+11*kernelDistance, vl) ;
	    __vr vrkC = _vel_vldu_vssl(4*kernelStride, pKerValue+12*kernelDistance, vl) ;
	    __vr vrkD = _vel_vldu_vssl(4*kernelStride, pKerValue+13*kernelDistance, vl) ;
	    __vr vrkE = _vel_vldu_vssl(4*kernelStride, pKerValue+14*kernelDistance, vl) ;
	    __vr vrkF = _vel_vldu_vssl(4*kernelStride, pKerValue+15*kernelDistance, vl) ;

	    __vr vriP_h1w1  = _vel_vshf_vvvsl(vri_h1w1, vri_h1w1, VE_VSHUFFLE_YUZU, vl) ;

	    __vr vrk01 = _vel_vshf_vvvsl(vrk0, vrk1, VE_VSHUFFLE_YUZU, vl) ;
	    vrsum01_y1x1 = _vel_pvfmad_vvvvvl(vrsum01_y1x1, vriP_h1w1, vrk01, vrsum01_y1x1, vl) ;

	    __vr vrk23 = _vel_vshf_vvvsl(vrk2, vrk3, VE_VSHUFFLE_YUZU, vl) ;
	    vrsum23_y1x1 = _vel_pvfmad_vvvvvl(vrsum23_y1x1, vriP_h1w1, vrk23, vrsum23_y1x1, vl) ;

	    __vr vrk45 = _vel_vshf_vvvsl(vrk4, vrk5, VE_VSHUFFLE_YUZU, vl) ;
	    vrsum45_y1x1 = _vel_pvfmad_vvvvvl(vrsum45_y1x1, vriP_h1w1, vrk45, vrsum45_y1x1, vl) ;

	    __vr vrk67 = _vel_vshf_vvvsl(vrk6, vrk7, VE_VSHUFFLE_YUZU, vl) ;
	    vrsum67_y1x1 = _vel_pvfmad_vvvvvl(vrsum67_y1x1, vriP_h1w1, vrk67, vrsum67_y1x1, vl) ;

	    __vr vrk89 = _vel_vshf_vvvsl(vrk8, vrk9, VE_VSHUFFLE_YUZU, vl) ;
	    vrsum89_y1x1 = _vel_pvfmad_vvvvvl(vrsum89_y1x1, vriP_h1w1, vrk89, vrsum89_y1x1, vl) ;

	    __vr vrkAB = _vel_vshf_vvvsl(vrkA, vrkB, VE_VSHUFFLE_YUZU, vl) ;
	    vrsumAB_y1x1 = _vel_pvfmad_vvvvvl(vrsumAB_y1x1, vriP_h1w1, vrkAB, vrsumAB_y1x1, vl) ;

	    __vr vrkCD = _vel_vshf_vvvsl(vrkC, vrkD, VE_VSHUFFLE_YUZU, vl) ;
	    vrsumCD_y1x1 = _vel_pvfmad_vvvvvl(vrsumCD_y1x1, vriP_h1w1, vrkCD, vrsumCD_y1x1, vl) ;

	    __vr vrkEF = _vel_vshf_vvvsl(vrkE, vrkF, VE_VSHUFFLE_YUZU, vl) ;
	    vrsumEF_y1x1 = _vel_pvfmad_vvvvvl(vrsumEF_y1x1, vriP_h1w1, vrkEF, vrsumEF_y1x1, vl) ;
	  }
	} // kernWidth
      }
    } // kernHeight
  } // inChannel

  if(Y>=1 && X>=1) {
    __vr vrsum0_y0x0 = _vel_vfsums_vvl(vrsum01_y0x0, VLEN) ;
    if(ADDBIAS) vrsum0_y0x0 = _vel_vfadds_vsvl(bias0, vrsum0_y0x0, 1) ;
    __vr vrsum1_y0x0 = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsum01_y0x0,32, VLEN), VLEN) ;
    if(ADDBIAS) vrsum1_y0x0 = _vel_vfadds_vsvl(bias1, vrsum1_y0x0, 1) ;
    __vr vrsum2_y0x0 = _vel_vfsums_vvl(vrsum23_y0x0, VLEN) ;
    if(ADDBIAS) vrsum2_y0x0 = _vel_vfadds_vsvl(bias2, vrsum2_y0x0, 1) ;
    __vr vrsum3_y0x0 = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsum23_y0x0,32, VLEN), VLEN) ;
    if(ADDBIAS) vrsum3_y0x0 = _vel_vfadds_vsvl(bias3, vrsum3_y0x0, 1) ;
    __vr vrsum4_y0x0 = _vel_vfsums_vvl(vrsum45_y0x0, VLEN) ;
    if(ADDBIAS) vrsum4_y0x0 = _vel_vfadds_vsvl(bias4, vrsum4_y0x0, 1) ;
    __vr vrsum5_y0x0 = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsum45_y0x0,32, VLEN), VLEN) ;
    if(ADDBIAS) vrsum5_y0x0 = _vel_vfadds_vsvl(bias5, vrsum5_y0x0, 1) ;
    __vr vrsum6_y0x0 = _vel_vfsums_vvl(vrsum67_y0x0, VLEN) ;
    if(ADDBIAS) vrsum6_y0x0 = _vel_vfadds_vsvl(bias6, vrsum6_y0x0, 1) ;
    __vr vrsum7_y0x0 = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsum67_y0x0,32, VLEN), VLEN) ;
    if(ADDBIAS) vrsum7_y0x0 = _vel_vfadds_vsvl(bias7, vrsum7_y0x0, 1) ;
    __vr vrsum8_y0x0 = _vel_vfsums_vvl(vrsum89_y0x0, VLEN) ;
    if(ADDBIAS) vrsum8_y0x0 = _vel_vfadds_vsvl(bias8, vrsum8_y0x0, 1) ;
    __vr vrsum9_y0x0 = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsum89_y0x0,32, VLEN), VLEN) ;
    if(ADDBIAS) vrsum9_y0x0 = _vel_vfadds_vsvl(bias9, vrsum9_y0x0, 1) ;
    __vr vrsumA_y0x0 = _vel_vfsums_vvl(vrsumAB_y0x0, VLEN) ;
    if(ADDBIAS) vrsumA_y0x0 = _vel_vfadds_vsvl(biasA, vrsumA_y0x0, 1) ;
    __vr vrsumB_y0x0 = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsumAB_y0x0,32, VLEN), VLEN) ;
    if(ADDBIAS) vrsumB_y0x0 = _vel_vfadds_vsvl(biasB, vrsumB_y0x0, 1) ;
    __vr vrsumC_y0x0 = _vel_vfsums_vvl(vrsumCD_y0x0, VLEN) ;
    if(ADDBIAS) vrsumC_y0x0 = _vel_vfadds_vsvl(biasC, vrsumC_y0x0, 1) ;
    __vr vrsumD_y0x0 = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsumCD_y0x0,32, VLEN), VLEN) ;
    if(ADDBIAS) vrsumD_y0x0 = _vel_vfadds_vsvl(biasD, vrsumD_y0x0, 1) ;
    __vr vrsumE_y0x0 = _vel_vfsums_vvl(vrsumEF_y0x0, VLEN) ;
    if(ADDBIAS) vrsumE_y0x0 = _vel_vfadds_vsvl(biasE, vrsumE_y0x0, 1) ;
    __vr vrsumF_y0x0 = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsumEF_y0x0,32, VLEN), VLEN) ;
    if(ADDBIAS) vrsumF_y0x0 = _vel_vfadds_vsvl(biasF, vrsumF_y0x0, 1) ;

    if(NUMKERNEL >= 1) _vel_vstu_vssl(vrsum0_y0x0, 4, &pOut[outIndex+0*outHeight*outWidth], 1) ;
    if(NUMKERNEL >= 2) _vel_vstu_vssl(vrsum1_y0x0, 4, &pOut[outIndex+1*outHeight*outWidth], 1) ;
    if(NUMKERNEL >= 3) _vel_vstu_vssl(vrsum2_y0x0, 4, &pOut[outIndex+2*outHeight*outWidth], 1) ;
    if(NUMKERNEL >= 4) _vel_vstu_vssl(vrsum3_y0x0, 4, &pOut[outIndex+3*outHeight*outWidth], 1) ;
    if(NUMKERNEL >= 5) _vel_vstu_vssl(vrsum4_y0x0, 4, &pOut[outIndex+4*outHeight*outWidth], 1) ;
    if(NUMKERNEL >= 6) _vel_vstu_vssl(vrsum5_y0x0, 4, &pOut[outIndex+5*outHeight*outWidth], 1) ;
    if(NUMKERNEL >= 7) _vel_vstu_vssl(vrsum6_y0x0, 4, &pOut[outIndex+6*outHeight*outWidth], 1) ;
    if(NUMKERNEL >= 8) _vel_vstu_vssl(vrsum7_y0x0, 4, &pOut[outIndex+7*outHeight*outWidth], 1) ;
    if(NUMKERNEL >= 9) _vel_vstu_vssl(vrsum8_y0x0, 4, &pOut[outIndex+8*outHeight*outWidth], 1) ;
    if(NUMKERNEL >=10) _vel_vstu_vssl(vrsum9_y0x0, 4, &pOut[outIndex+9*outHeight*outWidth], 1) ;
    if(NUMKERNEL >=11) _vel_vstu_vssl(vrsumA_y0x0, 4, &pOut[outIndex+10*outHeight*outWidth], 1) ;
    if(NUMKERNEL >=12) _vel_vstu_vssl(vrsumB_y0x0, 4, &pOut[outIndex+11*outHeight*outWidth], 1) ;
    if(NUMKERNEL >=13) _vel_vstu_vssl(vrsumC_y0x0, 4, &pOut[outIndex+12*outHeight*outWidth], 1) ;
    if(NUMKERNEL >=14) _vel_vstu_vssl(vrsumD_y0x0, 4, &pOut[outIndex+13*outHeight*outWidth], 1) ;
    if(NUMKERNEL >=15) _vel_vstu_vssl(vrsumE_y0x0, 4, &pOut[outIndex+14*outHeight*outWidth], 1) ;
    if(NUMKERNEL >=16) _vel_vstu_vssl(vrsumF_y0x0, 4, &pOut[outIndex+15*outHeight*outWidth], 1) ;
  }
  if(Y>=1 && X>=2) {
    __vr vrsum0_y0x1 = _vel_vfsums_vvl(vrsum01_y0x1, VLEN) ;
    if(ADDBIAS) vrsum0_y0x1 = _vel_vfadds_vsvl(bias0, vrsum0_y0x1, 1) ;
    __vr vrsum1_y0x1 = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsum01_y0x1,32, VLEN), VLEN) ;
    if(ADDBIAS) vrsum1_y0x1 = _vel_vfadds_vsvl(bias1, vrsum1_y0x1, 1) ;
    __vr vrsum2_y0x1 = _vel_vfsums_vvl(vrsum23_y0x1, VLEN) ;
    if(ADDBIAS) vrsum2_y0x1 = _vel_vfadds_vsvl(bias2, vrsum2_y0x1, 1) ;
    __vr vrsum3_y0x1 = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsum23_y0x1,32, VLEN), VLEN) ;
    if(ADDBIAS) vrsum3_y0x1 = _vel_vfadds_vsvl(bias3, vrsum3_y0x1, 1) ;
    __vr vrsum4_y0x1 = _vel_vfsums_vvl(vrsum45_y0x1, VLEN) ;
    if(ADDBIAS) vrsum4_y0x1 = _vel_vfadds_vsvl(bias4, vrsum4_y0x1, 1) ;
    __vr vrsum5_y0x1 = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsum45_y0x1,32, VLEN), VLEN) ;
    if(ADDBIAS) vrsum5_y0x1 = _vel_vfadds_vsvl(bias5, vrsum5_y0x1, 1) ;
    __vr vrsum6_y0x1 = _vel_vfsums_vvl(vrsum67_y0x1, VLEN) ;
    if(ADDBIAS) vrsum6_y0x1 = _vel_vfadds_vsvl(bias6, vrsum6_y0x1, 1) ;
    __vr vrsum7_y0x1 = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsum67_y0x1,32, VLEN), VLEN) ;
    if(ADDBIAS) vrsum7_y0x1 = _vel_vfadds_vsvl(bias7, vrsum7_y0x1, 1) ;
    __vr vrsum8_y0x1 = _vel_vfsums_vvl(vrsum89_y0x1, VLEN) ;
    if(ADDBIAS) vrsum8_y0x1 = _vel_vfadds_vsvl(bias8, vrsum8_y0x1, 1) ;
    __vr vrsum9_y0x1 = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsum89_y0x1,32, VLEN), VLEN) ;
    if(ADDBIAS) vrsum9_y0x1 = _vel_vfadds_vsvl(bias9, vrsum9_y0x1, 1) ;
    __vr vrsumA_y0x1 = _vel_vfsums_vvl(vrsumAB_y0x1, VLEN) ;
    if(ADDBIAS) vrsumA_y0x1 = _vel_vfadds_vsvl(biasA, vrsumA_y0x1, 1) ;
    __vr vrsumB_y0x1 = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsumAB_y0x1,32, VLEN), VLEN) ;
    if(ADDBIAS) vrsumB_y0x1 = _vel_vfadds_vsvl(biasB, vrsumB_y0x1, 1) ;
    __vr vrsumC_y0x1 = _vel_vfsums_vvl(vrsumCD_y0x1, VLEN) ;
    if(ADDBIAS) vrsumC_y0x1 = _vel_vfadds_vsvl(biasC, vrsumC_y0x1, 1) ;
    __vr vrsumD_y0x1 = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsumCD_y0x1,32, VLEN), VLEN) ;
    if(ADDBIAS) vrsumD_y0x1 = _vel_vfadds_vsvl(biasD, vrsumD_y0x1, 1) ;
    __vr vrsumE_y0x1 = _vel_vfsums_vvl(vrsumEF_y0x1, VLEN) ;
    if(ADDBIAS) vrsumE_y0x1 = _vel_vfadds_vsvl(biasE, vrsumE_y0x1, 1) ;
    __vr vrsumF_y0x1 = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsumEF_y0x1,32, VLEN), VLEN) ;
    if(ADDBIAS) vrsumF_y0x1 = _vel_vfadds_vsvl(biasF, vrsumF_y0x1, 1) ;

    if(NUMKERNEL >= 1) _vel_vstu_vssl(vrsum0_y0x1, 4, &pOut[outIndex+0*outHeight*outWidth+1], 1) ;
    if(NUMKERNEL >= 2) _vel_vstu_vssl(vrsum1_y0x1, 4, &pOut[outIndex+1*outHeight*outWidth+1], 1) ;
    if(NUMKERNEL >= 3) _vel_vstu_vssl(vrsum2_y0x1, 4, &pOut[outIndex+2*outHeight*outWidth+1], 1) ;
    if(NUMKERNEL >= 4) _vel_vstu_vssl(vrsum3_y0x1, 4, &pOut[outIndex+3*outHeight*outWidth+1], 1) ;
    if(NUMKERNEL >= 5) _vel_vstu_vssl(vrsum4_y0x1, 4, &pOut[outIndex+4*outHeight*outWidth+1], 1) ;
    if(NUMKERNEL >= 6) _vel_vstu_vssl(vrsum5_y0x1, 4, &pOut[outIndex+5*outHeight*outWidth+1], 1) ;
    if(NUMKERNEL >= 7) _vel_vstu_vssl(vrsum6_y0x1, 4, &pOut[outIndex+6*outHeight*outWidth+1], 1) ;
    if(NUMKERNEL >= 8) _vel_vstu_vssl(vrsum7_y0x1, 4, &pOut[outIndex+7*outHeight*outWidth+1], 1) ;
    if(NUMKERNEL >= 9) _vel_vstu_vssl(vrsum8_y0x1, 4, &pOut[outIndex+8*outHeight*outWidth+1], 1) ;
    if(NUMKERNEL >=10) _vel_vstu_vssl(vrsum9_y0x1, 4, &pOut[outIndex+9*outHeight*outWidth+1], 1) ;
    if(NUMKERNEL >=11) _vel_vstu_vssl(vrsumA_y0x1, 4, &pOut[outIndex+10*outHeight*outWidth+1], 1) ;
    if(NUMKERNEL >=12) _vel_vstu_vssl(vrsumB_y0x1, 4, &pOut[outIndex+11*outHeight*outWidth+1], 1) ;
    if(NUMKERNEL >=13) _vel_vstu_vssl(vrsumC_y0x1, 4, &pOut[outIndex+12*outHeight*outWidth+1], 1) ;
    if(NUMKERNEL >=14) _vel_vstu_vssl(vrsumD_y0x1, 4, &pOut[outIndex+13*outHeight*outWidth+1], 1) ;
    if(NUMKERNEL >=15) _vel_vstu_vssl(vrsumE_y0x1, 4, &pOut[outIndex+14*outHeight*outWidth+1], 1) ;
    if(NUMKERNEL >=16) _vel_vstu_vssl(vrsumF_y0x1, 4, &pOut[outIndex+15*outHeight*outWidth+1], 1) ;
  }
  if(Y>=2 && X>=1) {
    __vr vrsum0_y1x0 = _vel_vfsums_vvl(vrsum01_y1x0, VLEN) ;
    if(ADDBIAS) vrsum0_y1x0 = _vel_vfadds_vsvl(bias0, vrsum0_y1x0, 1) ;
    __vr vrsum1_y1x0 = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsum01_y1x0,32, VLEN), VLEN) ;
    if(ADDBIAS) vrsum1_y1x0 = _vel_vfadds_vsvl(bias1, vrsum1_y1x0, 1) ;
    __vr vrsum2_y1x0 = _vel_vfsums_vvl(vrsum23_y1x0, VLEN) ;
    if(ADDBIAS) vrsum2_y1x0 = _vel_vfadds_vsvl(bias2, vrsum2_y1x0, 1) ;
    __vr vrsum3_y1x0 = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsum23_y1x0,32, VLEN), VLEN) ;
    if(ADDBIAS) vrsum3_y1x0 = _vel_vfadds_vsvl(bias3, vrsum3_y1x0, 1) ;
    __vr vrsum4_y1x0 = _vel_vfsums_vvl(vrsum45_y1x0, VLEN) ;
    if(ADDBIAS) vrsum4_y1x0 = _vel_vfadds_vsvl(bias4, vrsum4_y1x0, 1) ;
    __vr vrsum5_y1x0 = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsum45_y1x0,32, VLEN), VLEN) ;
    if(ADDBIAS) vrsum5_y1x0 = _vel_vfadds_vsvl(bias5, vrsum5_y1x0, 1) ;
    __vr vrsum6_y1x0 = _vel_vfsums_vvl(vrsum67_y1x0, VLEN) ;
    if(ADDBIAS) vrsum6_y1x0 = _vel_vfadds_vsvl(bias6, vrsum6_y1x0, 1) ;
    __vr vrsum7_y1x0 = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsum67_y1x0,32, VLEN), VLEN) ;
    if(ADDBIAS) vrsum7_y1x0 = _vel_vfadds_vsvl(bias7, vrsum7_y1x0, 1) ;
    __vr vrsum8_y1x0 = _vel_vfsums_vvl(vrsum89_y1x0, VLEN) ;
    if(ADDBIAS) vrsum8_y1x0 = _vel_vfadds_vsvl(bias8, vrsum8_y1x0, 1) ;
    __vr vrsum9_y1x0 = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsum89_y1x0,32, VLEN), VLEN) ;
    if(ADDBIAS) vrsum9_y1x0 = _vel_vfadds_vsvl(bias9, vrsum9_y1x0, 1) ;
    __vr vrsumA_y1x0 = _vel_vfsums_vvl(vrsumAB_y1x0, VLEN) ;
    if(ADDBIAS) vrsumA_y1x0 = _vel_vfadds_vsvl(biasA, vrsumA_y1x0, 1) ;
    __vr vrsumB_y1x0 = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsumAB_y1x0,32, VLEN), VLEN) ;
    if(ADDBIAS) vrsumB_y1x0 = _vel_vfadds_vsvl(biasB, vrsumB_y1x0, 1) ;
    __vr vrsumC_y1x0 = _vel_vfsums_vvl(vrsumCD_y1x0, VLEN) ;
    if(ADDBIAS) vrsumC_y1x0 = _vel_vfadds_vsvl(biasC, vrsumC_y1x0, 1) ;
    __vr vrsumD_y1x0 = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsumCD_y1x0,32, VLEN), VLEN) ;
    if(ADDBIAS) vrsumD_y1x0 = _vel_vfadds_vsvl(biasD, vrsumD_y1x0, 1) ;
    __vr vrsumE_y1x0 = _vel_vfsums_vvl(vrsumEF_y1x0, VLEN) ;
    if(ADDBIAS) vrsumE_y1x0 = _vel_vfadds_vsvl(biasE, vrsumE_y1x0, 1) ;
    __vr vrsumF_y1x0 = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsumEF_y1x0,32, VLEN), VLEN) ;
    if(ADDBIAS) vrsumF_y1x0 = _vel_vfadds_vsvl(biasF, vrsumF_y1x0, 1) ;

    if(NUMKERNEL >= 1) _vel_vstu_vssl(vrsum0_y1x0, 4, &pOut[outIndex+0*outHeight*outWidth+outWidth], 1) ;
    if(NUMKERNEL >= 2) _vel_vstu_vssl(vrsum1_y1x0, 4, &pOut[outIndex+1*outHeight*outWidth+outWidth], 1) ;
    if(NUMKERNEL >= 3) _vel_vstu_vssl(vrsum2_y1x0, 4, &pOut[outIndex+2*outHeight*outWidth+outWidth], 1) ;
    if(NUMKERNEL >= 4) _vel_vstu_vssl(vrsum3_y1x0, 4, &pOut[outIndex+3*outHeight*outWidth+outWidth], 1) ;
    if(NUMKERNEL >= 5) _vel_vstu_vssl(vrsum4_y1x0, 4, &pOut[outIndex+4*outHeight*outWidth+outWidth], 1) ;
    if(NUMKERNEL >= 6) _vel_vstu_vssl(vrsum5_y1x0, 4, &pOut[outIndex+5*outHeight*outWidth+outWidth], 1) ;
    if(NUMKERNEL >= 7) _vel_vstu_vssl(vrsum6_y1x0, 4, &pOut[outIndex+6*outHeight*outWidth+outWidth], 1) ;
    if(NUMKERNEL >= 8) _vel_vstu_vssl(vrsum7_y1x0, 4, &pOut[outIndex+7*outHeight*outWidth+outWidth], 1) ;
    if(NUMKERNEL >= 9) _vel_vstu_vssl(vrsum8_y1x0, 4, &pOut[outIndex+8*outHeight*outWidth+outWidth], 1) ;
    if(NUMKERNEL >=10) _vel_vstu_vssl(vrsum9_y1x0, 4, &pOut[outIndex+9*outHeight*outWidth+outWidth], 1) ;
    if(NUMKERNEL >=11) _vel_vstu_vssl(vrsumA_y1x0, 4, &pOut[outIndex+10*outHeight*outWidth+outWidth], 1) ;
    if(NUMKERNEL >=12) _vel_vstu_vssl(vrsumB_y1x0, 4, &pOut[outIndex+11*outHeight*outWidth+outWidth], 1) ;
    if(NUMKERNEL >=13) _vel_vstu_vssl(vrsumC_y1x0, 4, &pOut[outIndex+12*outHeight*outWidth+outWidth], 1) ;
    if(NUMKERNEL >=14) _vel_vstu_vssl(vrsumD_y1x0, 4, &pOut[outIndex+13*outHeight*outWidth+outWidth], 1) ;
    if(NUMKERNEL >=15) _vel_vstu_vssl(vrsumE_y1x0, 4, &pOut[outIndex+14*outHeight*outWidth+outWidth], 1) ;
    if(NUMKERNEL >=16) _vel_vstu_vssl(vrsumF_y1x0, 4, &pOut[outIndex+15*outHeight*outWidth+outWidth], 1) ;
  }
  if(Y>=2 && X>=2) {
    __vr vrsum0_y1x1 = _vel_vfsums_vvl(vrsum01_y1x1, VLEN) ;
    if(ADDBIAS) vrsum0_y1x1 = _vel_vfadds_vsvl(bias0, vrsum0_y1x1, 1) ;
    __vr vrsum1_y1x1 = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsum01_y1x1,32, VLEN), VLEN) ;
    if(ADDBIAS) vrsum1_y1x1 = _vel_vfadds_vsvl(bias1, vrsum1_y1x1, 1) ;
    __vr vrsum2_y1x1 = _vel_vfsums_vvl(vrsum23_y1x1, VLEN) ;
    if(ADDBIAS) vrsum2_y1x1 = _vel_vfadds_vsvl(bias2, vrsum2_y1x1, 1) ;
    __vr vrsum3_y1x1 = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsum23_y1x1,32, VLEN), VLEN) ;
    if(ADDBIAS) vrsum3_y1x1 = _vel_vfadds_vsvl(bias3, vrsum3_y1x1, 1) ;
    __vr vrsum4_y1x1 = _vel_vfsums_vvl(vrsum45_y1x1, VLEN) ;
    if(ADDBIAS) vrsum4_y1x1 = _vel_vfadds_vsvl(bias4, vrsum4_y1x1, 1) ;
    __vr vrsum5_y1x1 = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsum45_y1x1,32, VLEN), VLEN) ;
    if(ADDBIAS) vrsum5_y1x1 = _vel_vfadds_vsvl(bias5, vrsum5_y1x1, 1) ;
    __vr vrsum6_y1x1 = _vel_vfsums_vvl(vrsum67_y1x1, VLEN) ;
    if(ADDBIAS) vrsum6_y1x1 = _vel_vfadds_vsvl(bias6, vrsum6_y1x1, 1) ;
    __vr vrsum7_y1x1 = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsum67_y1x1,32, VLEN), VLEN) ;
    if(ADDBIAS) vrsum7_y1x1 = _vel_vfadds_vsvl(bias7, vrsum7_y1x1, 1) ;
    __vr vrsum8_y1x1 = _vel_vfsums_vvl(vrsum89_y1x1, VLEN) ;
    if(ADDBIAS) vrsum8_y1x1 = _vel_vfadds_vsvl(bias8, vrsum8_y1x1, 1) ;
    __vr vrsum9_y1x1 = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsum89_y1x1,32, VLEN), VLEN) ;
    if(ADDBIAS) vrsum9_y1x1 = _vel_vfadds_vsvl(bias9, vrsum9_y1x1, 1) ;
    __vr vrsumA_y1x1 = _vel_vfsums_vvl(vrsumAB_y1x1, VLEN) ;
    if(ADDBIAS) vrsumA_y1x1 = _vel_vfadds_vsvl(biasA, vrsumA_y1x1, 1) ;
    __vr vrsumB_y1x1 = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsumAB_y1x1,32, VLEN), VLEN) ;
    if(ADDBIAS) vrsumB_y1x1 = _vel_vfadds_vsvl(biasB, vrsumB_y1x1, 1) ;
    __vr vrsumC_y1x1 = _vel_vfsums_vvl(vrsumCD_y1x1, VLEN) ;
    if(ADDBIAS) vrsumC_y1x1 = _vel_vfadds_vsvl(biasC, vrsumC_y1x1, 1) ;
    __vr vrsumD_y1x1 = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsumCD_y1x1,32, VLEN), VLEN) ;
    if(ADDBIAS) vrsumD_y1x1 = _vel_vfadds_vsvl(biasD, vrsumD_y1x1, 1) ;
    __vr vrsumE_y1x1 = _vel_vfsums_vvl(vrsumEF_y1x1, VLEN) ;
    if(ADDBIAS) vrsumE_y1x1 = _vel_vfadds_vsvl(biasE, vrsumE_y1x1, 1) ;
    __vr vrsumF_y1x1 = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsumEF_y1x1,32, VLEN), VLEN) ;
    if(ADDBIAS) vrsumF_y1x1 = _vel_vfadds_vsvl(biasF, vrsumF_y1x1, 1) ;

    if(NUMKERNEL >= 1) _vel_vstu_vssl(vrsum0_y1x1, 4, &pOut[outIndex+0*outHeight*outWidth+outWidth+1], 1) ;
    if(NUMKERNEL >= 2) _vel_vstu_vssl(vrsum1_y1x1, 4, &pOut[outIndex+1*outHeight*outWidth+outWidth+1], 1) ;
    if(NUMKERNEL >= 3) _vel_vstu_vssl(vrsum2_y1x1, 4, &pOut[outIndex+2*outHeight*outWidth+outWidth+1], 1) ;
    if(NUMKERNEL >= 4) _vel_vstu_vssl(vrsum3_y1x1, 4, &pOut[outIndex+3*outHeight*outWidth+outWidth+1], 1) ;
    if(NUMKERNEL >= 5) _vel_vstu_vssl(vrsum4_y1x1, 4, &pOut[outIndex+4*outHeight*outWidth+outWidth+1], 1) ;
    if(NUMKERNEL >= 6) _vel_vstu_vssl(vrsum5_y1x1, 4, &pOut[outIndex+5*outHeight*outWidth+outWidth+1], 1) ;
    if(NUMKERNEL >= 7) _vel_vstu_vssl(vrsum6_y1x1, 4, &pOut[outIndex+6*outHeight*outWidth+outWidth+1], 1) ;
    if(NUMKERNEL >= 8) _vel_vstu_vssl(vrsum7_y1x1, 4, &pOut[outIndex+7*outHeight*outWidth+outWidth+1], 1) ;
    if(NUMKERNEL >= 9) _vel_vstu_vssl(vrsum8_y1x1, 4, &pOut[outIndex+8*outHeight*outWidth+outWidth+1], 1) ;
    if(NUMKERNEL >=10) _vel_vstu_vssl(vrsum9_y1x1, 4, &pOut[outIndex+9*outHeight*outWidth+outWidth+1], 1) ;
    if(NUMKERNEL >=11) _vel_vstu_vssl(vrsumA_y1x1, 4, &pOut[outIndex+10*outHeight*outWidth+outWidth+1], 1) ;
    if(NUMKERNEL >=12) _vel_vstu_vssl(vrsumB_y1x1, 4, &pOut[outIndex+11*outHeight*outWidth+outWidth+1], 1) ;
    if(NUMKERNEL >=13) _vel_vstu_vssl(vrsumC_y1x1, 4, &pOut[outIndex+12*outHeight*outWidth+outWidth+1], 1) ;
    if(NUMKERNEL >=14) _vel_vstu_vssl(vrsumD_y1x1, 4, &pOut[outIndex+13*outHeight*outWidth+outWidth+1], 1) ;
    if(NUMKERNEL >=15) _vel_vstu_vssl(vrsumE_y1x1, 4, &pOut[outIndex+14*outHeight*outWidth+outWidth+1], 1) ;
    if(NUMKERNEL >=16) _vel_vstu_vssl(vrsumF_y1x1, 4, &pOut[outIndex+15*outHeight*outWidth+outWidth+1], 1) ;
  }
}

template<filterLayout_t FLAYOUT,int NUMKERNEL, bool ADDBIAS>
static inline void convloopXY(
    const float * __restrict__ pIn,
    const float * __restrict__ pKernel,
    const float * __restrict__ pBias,
    float * __restrict__ const pOut,
    const int64_t inChannel,
    const int64_t inWidth,
    const int64_t inHeight,
    const int64_t outChannel,
    const int64_t outWidth,
    const int64_t outHeight,
    const int64_t kernWidth,
    const int64_t kernHeight,
    const int64_t inChannelGroup,
    const int64_t outChannelGroup,
    const int64_t strideHeight,
    const int64_t strideWidth,
    const int64_t padHeight,
    const int64_t padWidth,
    const int64_t dilationHeight,
    const int64_t dilationWidth,
    const int64_t inGroupOffset,
    const int64_t outGroupOffset,
    const int64_t biasGroupOffset,
    const int64_t kernGroupOffset,
    const int64_t n,
    const int64_t k
)
{
  int64_t y=0 ;
  if( (outHeight & 0x01) == 1 ) {
    int64_t x=0;
    if( (outWidth & 0x01) == 1 ) {
      func<FLAYOUT,ADDBIAS,NUMKERNEL,1,1>(pIn, pKernel, pBias, pOut,
	 inChannel, inWidth, inHeight,
	 outChannel, outWidth, outHeight,
	 kernWidth, kernHeight,
	 inChannelGroup, outChannelGroup,
	 strideHeight, strideWidth,
	 padHeight, padWidth,
	 dilationHeight, dilationWidth,
	 inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	 n, k, y, x) ;
      x+=1 ;
    }
    for (; x<outWidth; ) {
      func<FLAYOUT,ADDBIAS,NUMKERNEL,1,2>(pIn, pKernel, pBias, pOut,
	 inChannel, inWidth, inHeight,
	 outChannel, outWidth, outHeight,
	 kernWidth, kernHeight,
	 inChannelGroup, outChannelGroup,
	 strideHeight, strideWidth,
	 padHeight, padWidth,
	 dilationHeight, dilationWidth,
	 inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	 n, k, y, x) ;
      x+=2 ;
    } // outWidth
    y+=1 ;
  }
  for (; y<outHeight; ) {
    int64_t x=0;
    if( (outWidth & 0x01) == 1 ) {
      func<FLAYOUT,ADDBIAS,NUMKERNEL,2,1>(pIn, pKernel, pBias, pOut,
	 inChannel, inWidth, inHeight,
	 outChannel, outWidth, outHeight,
	 kernWidth, kernHeight,
	 inChannelGroup, outChannelGroup,
	 strideHeight, strideWidth,
	 padHeight, padWidth,
	 dilationHeight, dilationWidth,
	 inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	 n, k, y, x) ;
      x+=1 ;
    }
    for (; x<outWidth; ) {
      func<FLAYOUT,ADDBIAS,NUMKERNEL,2,2>(pIn, pKernel, pBias, pOut,
	 inChannel, inWidth, inHeight,
	 outChannel, outWidth, outHeight,
	 kernWidth, kernHeight,
	 inChannelGroup, outChannelGroup,
	 strideHeight, strideWidth,
	 padHeight, padWidth,
	 dilationHeight, dilationWidth,
	 inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	 n, k, y, x) ;
      x+=2 ;
    } // outWidth
    y+=2 ;
  } // outHeight
}

template<filterLayout_t FLAYOUT, bool ADDBIAS>
static inline void convloop(
    const float * __restrict__ pIn,
    const float * __restrict__ pKernel,
    const float * __restrict__ pBias,
    float * __restrict__ const pOut,
    const int64_t batch,
    const int64_t group,
    const int64_t inChannel,
    const int64_t inWidth,
    const int64_t inHeight,
    const int64_t outChannel,
    const int64_t outWidth,
    const int64_t outHeight,
    const int64_t kernWidth,
    const int64_t kernHeight,
    const int64_t inChannelGroup,
    const int64_t outChannelGroup,
    const int64_t strideHeight,
    const int64_t strideWidth,
    const int64_t padHeight,
    const int64_t padWidth,
    const int64_t dilationHeight,
    const int64_t dilationWidth
)
{
  for (int64_t n = 0; n < batch; n++) {
    for (int64_t g = 0; g < group; g++) {
      const int64_t inGroupOffset   = g * inChannelGroup * inHeight * inWidth;
      const int64_t outGroupOffset  = g * outChannelGroup * outHeight * outWidth;
      const int64_t biasGroupOffset = g * outChannelGroup;
      const int64_t kernGroupOffset = g * outChannelGroup * inChannelGroup * kernHeight * kernWidth;

      const int64_t remain = outChannelGroup & 0xf ;

      int k = 0 ;
      switch( remain ) {
      case 1 :
	convloopXY<FLAYOUT, 1, ADDBIAS>(pIn, pKernel, pBias, pOut,
	   inChannel, inWidth, inHeight,
	   outChannel, outWidth, outHeight,
	   kernWidth, kernHeight,
	   inChannelGroup, outChannelGroup,
	   strideHeight, strideWidth,
	   padHeight, padWidth,
	   dilationHeight, dilationWidth,
	   inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	   n, k );
	k+=1 ;
	break ;
      case 2 :
	convloopXY<FLAYOUT, 2, ADDBIAS>(pIn, pKernel, pBias, pOut,
	   inChannel, inWidth, inHeight,
	   outChannel, outWidth, outHeight,
	   kernWidth, kernHeight,
	   inChannelGroup, outChannelGroup,
	   strideHeight, strideWidth,
	   padHeight, padWidth,
	   dilationHeight, dilationWidth,
	   inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	   n, k );
	k+=2 ;
	break ;
      case 3 :
	convloopXY<FLAYOUT, 3, ADDBIAS>(pIn, pKernel, pBias, pOut,
	   inChannel, inWidth, inHeight,
	   outChannel, outWidth, outHeight,
	   kernWidth, kernHeight,
	   inChannelGroup, outChannelGroup,
	   strideHeight, strideWidth,
	   padHeight, padWidth,
	   dilationHeight, dilationWidth,
	   inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	   n, k );
	k+=3 ;
	break ;
      case 4 :
	convloopXY<FLAYOUT, 4, ADDBIAS>(pIn, pKernel, pBias, pOut,
	   inChannel, inWidth, inHeight,
	   outChannel, outWidth, outHeight,
	   kernWidth, kernHeight,
	   inChannelGroup, outChannelGroup,
	   strideHeight, strideWidth,
	   padHeight, padWidth,
	   dilationHeight, dilationWidth,
	   inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	   n, k );
	k+=4 ;
	break ;
      case 5 :
	convloopXY<FLAYOUT, 5, ADDBIAS>(pIn, pKernel, pBias, pOut,
	   inChannel, inWidth, inHeight,
	   outChannel, outWidth, outHeight,
	   kernWidth, kernHeight,
	   inChannelGroup, outChannelGroup,
	   strideHeight, strideWidth,
	   padHeight, padWidth,
	   dilationHeight, dilationWidth,
	   inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	   n, k );
	k+=5 ;
	break ;
      case 6 :
	convloopXY<FLAYOUT, 6, ADDBIAS>(pIn, pKernel, pBias, pOut,
	   inChannel, inWidth, inHeight,
	   outChannel, outWidth, outHeight,
	   kernWidth, kernHeight,
	   inChannelGroup, outChannelGroup,
	   strideHeight, strideWidth,
	   padHeight, padWidth,
	   dilationHeight, dilationWidth,
	   inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	   n, k );
	k+=6 ;
	break ;
      case 7 :
	convloopXY<FLAYOUT, 7, ADDBIAS>(pIn, pKernel, pBias, pOut,
	   inChannel, inWidth, inHeight,
	   outChannel, outWidth, outHeight,
	   kernWidth, kernHeight,
	   inChannelGroup, outChannelGroup,
	   strideHeight, strideWidth,
	   padHeight, padWidth,
	   dilationHeight, dilationWidth,
	   inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	   n, k );
	k+=7 ;
	break ;
      case 8 :
	convloopXY<FLAYOUT, 8, ADDBIAS>(pIn, pKernel, pBias, pOut,
	   inChannel, inWidth, inHeight,
	   outChannel, outWidth, outHeight,
	   kernWidth, kernHeight,
	   inChannelGroup, outChannelGroup,
	   strideHeight, strideWidth,
	   padHeight, padWidth,
	   dilationHeight, dilationWidth,
	   inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	   n, k );
	k+=8 ;
	break ;
      case 9 :
	convloopXY<FLAYOUT, 9, ADDBIAS>(pIn, pKernel, pBias, pOut,
	   inChannel, inWidth, inHeight,
	   outChannel, outWidth, outHeight,
	   kernWidth, kernHeight,
	   inChannelGroup, outChannelGroup,
	   strideHeight, strideWidth,
	   padHeight, padWidth,
	   dilationHeight, dilationWidth,
	   inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	   n, k );
	k+=9 ;
	break ;
      case 10 :
	convloopXY<FLAYOUT, 10, ADDBIAS>(pIn, pKernel, pBias, pOut,
	   inChannel, inWidth, inHeight,
	   outChannel, outWidth, outHeight,
	   kernWidth, kernHeight,
	   inChannelGroup, outChannelGroup,
	   strideHeight, strideWidth,
	   padHeight, padWidth,
	   dilationHeight, dilationWidth,
	   inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	   n, k );
	k+=10 ;
	break ;
      case 11 :
	convloopXY<FLAYOUT, 11, ADDBIAS>(pIn, pKernel, pBias, pOut,
	   inChannel, inWidth, inHeight,
	   outChannel, outWidth, outHeight,
	   kernWidth, kernHeight,
	   inChannelGroup, outChannelGroup,
	   strideHeight, strideWidth,
	   padHeight, padWidth,
	   dilationHeight, dilationWidth,
	   inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	   n, k );
	k+=11 ;
	break ;
      case 12 :
	convloopXY<FLAYOUT, 12, ADDBIAS>(pIn, pKernel, pBias, pOut,
	   inChannel, inWidth, inHeight,
	   outChannel, outWidth, outHeight,
	   kernWidth, kernHeight,
	   inChannelGroup, outChannelGroup,
	   strideHeight, strideWidth,
	   padHeight, padWidth,
	   dilationHeight, dilationWidth,
	   inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	   n, k );
	k+=12 ;
	break ;
      case 13 :
	convloopXY<FLAYOUT, 13, ADDBIAS>(pIn, pKernel, pBias, pOut,
	   inChannel, inWidth, inHeight,
	   outChannel, outWidth, outHeight,
	   kernWidth, kernHeight,
	   inChannelGroup, outChannelGroup,
	   strideHeight, strideWidth,
	   padHeight, padWidth,
	   dilationHeight, dilationWidth,
	   inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	   n, k );
	k+=13 ;
	break ;
      case 14 :
	convloopXY<FLAYOUT, 14, ADDBIAS>(pIn, pKernel, pBias, pOut,
	   inChannel, inWidth, inHeight,
	   outChannel, outWidth, outHeight,
	   kernWidth, kernHeight,
	   inChannelGroup, outChannelGroup,
	   strideHeight, strideWidth,
	   padHeight, padWidth,
	   dilationHeight, dilationWidth,
	   inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	   n, k );
	k+=14 ;
	break ;
      case 15 :
	convloopXY<FLAYOUT, 15, ADDBIAS>(pIn, pKernel, pBias, pOut,
	   inChannel, inWidth, inHeight,
	   outChannel, outWidth, outHeight,
	   kernWidth, kernHeight,
	   inChannelGroup, outChannelGroup,
	   strideHeight, strideWidth,
	   padHeight, padWidth,
	   dilationHeight, dilationWidth,
	   inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	   n, k );
	k+=15 ;
	break ;
      default :
	break ;
      }
      for (; k < outChannelGroup; k+=16) {
	convloopXY<FLAYOUT, 16, ADDBIAS>(pIn, pKernel, pBias, pOut,
	   inChannel, inWidth, inHeight,
	   outChannel, outWidth, outHeight,
	   kernWidth, kernHeight,
	   inChannelGroup, outChannelGroup,
	   strideHeight, strideWidth,
	   padHeight, padWidth,
	   dilationHeight, dilationWidth,
	   inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	   n, k );
      } // outChannel
    } // group
  } // batch
}

extern "C" vednnError_t
vednnConvolutionForward_direct_vecC(
    const vednnTensorParam_t *  	pParamIn,
    const void *  			pDataIn,
    const vednnFilterParam_t *  	pParamKernel,
    const void *  			pDataKernel,
    const vednnBiasParam_t * 		pParamBias,
    const void * 			pDataBias,
    const vednnConvolutionParam_t *  	pParamConv,
    const vednnTensorParam_t *  	pParamOut,
    void *  				pDataOut
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

  const int64_t filter_layout = pParamKernel->layout ;

  const int64_t group          = pParamConv->group;
  const int64_t strideWidth    = pParamConv->strideWidth;;
  const int64_t strideHeight   = pParamConv->strideHeight;
  const int64_t padWidth       = pParamConv->padWidth;
  const int64_t padHeight      = pParamConv->padHeight;
  const int64_t dilationWidth  = pParamConv->dilationWidth;
  const int64_t dilationHeight = pParamConv->dilationHeight;

  const int64_t inChannelGroup  = inChannel  / group;   // equal to pDataKernel->inChannel
  const int64_t outChannelGroup = outChannel / group;   // equal to pDataKernel->outChannel

  const float * pIn     = (const float *) pDataIn;
  const float * pKernel = (const float *) pDataKernel;
  const float * pBias   = (const float *) pDataBias;
  float * const pOut    = (float * const) pDataOut;

  if( filter_layout == VEDNN_FILTER_LAYOUT_NCHW) {
    if( pDataBias == NULL ) {
      convloop<VEDNN_FILTER_LAYOUT_NCHW, false>(pIn, pKernel, pBias, pOut,
		   batch, group,
		   inChannel, inWidth, inHeight,
		   outChannel, outWidth, outHeight,
		   kernWidth, kernHeight,
		   inChannelGroup, outChannelGroup,
		   strideHeight, strideWidth,
		   padHeight, padWidth,
		   dilationHeight, dilationWidth ) ;
    }
    else {
      convloop<VEDNN_FILTER_LAYOUT_NCHW, true>(pIn, pKernel, pBias, pOut,
		   batch, group,
		   inChannel, inWidth, inHeight,
		   outChannel, outWidth, outHeight,
		   kernWidth, kernHeight,
		   inChannelGroup, outChannelGroup,
		   strideHeight, strideWidth,
		   padHeight, padWidth,
		   dilationHeight, dilationWidth ) ;
    }
  }
  else {
    if( pDataBias == NULL ) {
      convloop<VEDNN_FILTER_LAYOUT_HWCN, false>(pIn, pKernel, pBias, pOut,
		   batch, group,
		   inChannel, inWidth, inHeight,
		   outChannel, outWidth, outHeight,
		   kernWidth, kernHeight,
		   inChannelGroup, outChannelGroup,
		   strideHeight, strideWidth,
		   padHeight, padWidth,
		   dilationHeight, dilationWidth ) ;
    }
    else {
      convloop<VEDNN_FILTER_LAYOUT_HWCN, true>(pIn, pKernel, pBias, pOut,
		   batch, group,
		   inChannel, inWidth, inHeight,
		   outChannel, outWidth, outHeight,
		   kernWidth, kernHeight,
		   inChannelGroup, outChannelGroup,
		   strideHeight, strideWidth,
		   padHeight, padWidth,
		   dilationHeight, dilationWidth ) ;
    }
  }


  return VEDNN_SUCCESS;
}
