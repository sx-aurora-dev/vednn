#include <stdint.h>
#include <stdlib.h>

#include "vednn.h"

#include "velintrin.h"
#define VLEN	(256)

template<filterLayout_t FLAYOUT>
static inline void k1(
    const float * __restrict__ pIn,
    const float * __restrict__ pKernel,
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
    const int64_t kernGroupOffset,
    const int64_t n,
    const int64_t k
)
{
  for (int64_t y=0; y<outHeight; y++) {
    int64_t i = y * strideHeight - padHeight;
    for (int64_t x=0; x<outWidth; x++) {
      int64_t j = x * strideWidth - padWidth;
      int64_t outIndex  = outGroupOffset + ((n * outChannel + k) * outHeight + y) * outWidth + x;

      __vr vrsum = _vel_vbrds_vsl(0.f, VLEN) ;

      for (int64_t c=0; c<inChannelGroup; c+= VLEN) {
	const int64_t vl = inChannelGroup - c < VLEN ? inChannelGroup - c : VLEN ;

	for (int64_t r=0; r<kernHeight; r++) {
	  for (int64_t s=0; s<kernWidth; s++) {
	    int64_t h = i + r * dilationHeight;
	    int64_t w = j + s * dilationWidth;
	    if (h < 0 || inHeight <= h) {
	      continue;
	    }
	    if (w < 0 || inWidth <= w) {
	      continue;
	    }
	    int64_t inputIndex  = inGroupOffset + ((n * inChannel + c) * inHeight + h) * inWidth + w;
	    __vr vri = _vel_vldu_vssl(4*inHeight*inWidth, &pIn[inputIndex], vl) ;

	    const float *pKerValue = ( FLAYOUT == VEDNN_FILTER_LAYOUT_NCHW ) ?
				      pKernel + kernGroupOffset + ((k * inChannelGroup + c) * kernHeight + r) * kernWidth + s :
				      pKernel + kernGroupOffset + ( ( r * kernWidth + s ) * inChannelGroup + c ) * outChannelGroup + k ;

	    const int64_t kernelStride = ( FLAYOUT == VEDNN_FILTER_LAYOUT_NCHW ) ?
			                 kernHeight * kernWidth :
					 outChannelGroup ;
	    __vr vrk = _vel_vldu_vssl(4*kernelStride, pKerValue, vl) ;

	    vrsum = _vel_vfmads_vvvvvl(vrsum, vri, vrk, vrsum, vl) ;
	  } // kernWidth
	} // kernHeight
      } // inChannel

      vrsum = _vel_vfsums_vvl(vrsum, VLEN) ;
      _vel_vstu_vssl(vrsum, 4, &pOut[outIndex], 1) ;
    } // outWidth
  } // outHeight
}

template<filterLayout_t FLAYOUT>
static inline void k2(
    const float * __restrict__ pIn,
    const float * __restrict__ pKernel,
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
    const int64_t kernGroupOffset,
    const int64_t n,
    const int64_t k
)
{
  for (int64_t y=0; y<outHeight; y++) {
    int64_t i = y * strideHeight - padHeight;
    for (int64_t x=0; x<outWidth; x++) {
      int64_t j = x * strideWidth - padWidth;
      int64_t outIndex  = outGroupOffset + ((n * outChannel + k) * outHeight + y) * outWidth + x;

       __vr vrsum01 = _vel_vbrdl_vsl(0UL, VLEN) ;

      for (int64_t c=0; c<inChannelGroup; c+= VLEN) {
	const int64_t vl = inChannelGroup - c < VLEN ? inChannelGroup - c : VLEN ;

	for (int64_t r=0; r<kernHeight; r++) {
	  for (int64_t s=0; s<kernWidth; s++) {
	    int64_t h = i + r * dilationHeight;
	    int64_t w = j + s * dilationWidth;
	    if (h < 0 || inHeight <= h) {
	      continue;
	    }
	    if (w < 0 || inWidth <= w) {
	      continue;
	    }
	    int64_t inputIndex  = inGroupOffset + ((n * inChannel + c) * inHeight + h) * inWidth + w;
	    __vr vri = _vel_vldu_vssl(4*inHeight*inWidth, &pIn[inputIndex], vl) ;

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

	    __vr vriP  = _vel_vshf_vvvsl(vri, vri, VE_VSHUFFLE_YUZU, vl) ;

	    __vr vrk01 = _vel_vshf_vvvsl(vrk0, vrk1, VE_VSHUFFLE_YUZU, vl) ;
	    vrsum01 = _vel_pvfmad_vvvvvl(vrsum01, vriP, vrk01, vrsum01, vl) ;
	  } // kernWidth
	} // kernHeight
      } // inChannel

      __vr vrsum0 = _vel_vfsums_vvl(vrsum01, VLEN) ;
      __vr vrsum1 = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsum01,32, VLEN), VLEN) ;

      _vel_vstu_vssl(vrsum0, 4, &pOut[outIndex+0*outHeight*outWidth], 1) ;
      _vel_vstu_vssl(vrsum1, 4, &pOut[outIndex+1*outHeight*outWidth], 1) ;
    } // outWidth
  } // outHeight
}

template<filterLayout_t FLAYOUT>
static inline void k4(
    const float * __restrict__ pIn,
    const float * __restrict__ pKernel,
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
    const int64_t kernGroupOffset,
    const int64_t n,
    const int64_t k
)
{
  for (int64_t y=0; y<outHeight; y++) {
    int64_t i = y * strideHeight - padHeight;
    for (int64_t x=0; x<outWidth; x++) {
      int64_t j = x * strideWidth - padWidth;
      int64_t outIndex  = outGroupOffset + ((n * outChannel + k) * outHeight + y) * outWidth + x;

      __vr vrsum01 = _vel_vbrdl_vsl(0UL, VLEN) ;
      __vr vrsum23 = _vel_vbrdl_vsl(0UL, VLEN) ;

      for (int64_t c=0; c<inChannelGroup; c+= VLEN) {
	const int64_t vl = inChannelGroup - c < VLEN ? inChannelGroup - c : VLEN ;

	for (int64_t r=0; r<kernHeight; r++) {
	  for (int64_t s=0; s<kernWidth; s++) {
	    int64_t h = i + r * dilationHeight;
	    int64_t w = j + s * dilationWidth;
	    if (h < 0 || inHeight <= h) {
	      continue;
	    }
	    if (w < 0 || inWidth <= w) {
	      continue;
	    }
	    int64_t inputIndex  = inGroupOffset + ((n * inChannel + c) * inHeight + h) * inWidth + w;
	    __vr vri = _vel_vldu_vssl(4*inHeight*inWidth, &pIn[inputIndex], vl) ;

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

	    __vr vriP  = _vel_vshf_vvvsl(vri, vri, VE_VSHUFFLE_YUZU, vl) ;

	    __vr vrk01 = _vel_vshf_vvvsl(vrk0, vrk1, VE_VSHUFFLE_YUZU, vl) ;
	    vrsum01 = _vel_pvfmad_vvvvvl(vrsum01, vriP, vrk01, vrsum01, vl) ;
	    __vr vrk23 = _vel_vshf_vvvsl(vrk2, vrk3, VE_VSHUFFLE_YUZU, vl) ;
	    vrsum23 = _vel_pvfmad_vvvvvl(vrsum23, vriP, vrk23, vrsum23, vl) ;
	  } // kernWidth
	} // kernHeight
      } // inChannel

      __vr vrsum0 = _vel_vfsums_vvl(vrsum01, VLEN) ;
      __vr vrsum1 = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsum01,32, VLEN), VLEN) ;
      __vr vrsum2 = _vel_vfsums_vvl(vrsum23, VLEN) ;
      __vr vrsum3 = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsum23,32, VLEN), VLEN) ;

      _vel_vstu_vssl(vrsum0, 4, &pOut[outIndex+0*outHeight*outWidth], 1) ;
      _vel_vstu_vssl(vrsum1, 4, &pOut[outIndex+1*outHeight*outWidth], 1) ;
      _vel_vstu_vssl(vrsum2, 4, &pOut[outIndex+2*outHeight*outWidth], 1) ;
      _vel_vstu_vssl(vrsum3, 4, &pOut[outIndex+3*outHeight*outWidth], 1) ;

    } // outWidth
  } // outHeight
}

template<filterLayout_t FLAYOUT>
static inline void k8y1x1(
    const float * __restrict__ pIn,
    const float * __restrict__ pKernel,
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
    const int64_t kernGroupOffset,
    const int64_t n,
    const int64_t k,
    const int64_t y,
    const int64_t x
)
{

  int64_t outIndex  = outGroupOffset + ((n * outChannel + k) * outHeight + y) * outWidth + x;

  __vr vrsum01_y0x0 = _vel_vbrdl_vsl(0UL, VLEN) ;
  __vr vrsum23_y0x0 = _vel_vbrdl_vsl(0UL, VLEN) ;
  __vr vrsum45_y0x0 = _vel_vbrdl_vsl(0UL, VLEN) ;
  __vr vrsum67_y0x0 = _vel_vbrdl_vsl(0UL, VLEN) ;

  for (int64_t c=0; c<inChannelGroup; c+= VLEN) {
    const int64_t vl = inChannelGroup - c < VLEN ? inChannelGroup - c : VLEN ;

    for (int64_t r=0; r<kernHeight; r++) {
      int64_t h0 = (y+0) * strideHeight - padHeight + r * dilationHeight;

      int64_t h0_valid = ( h0 >= 0 && h0 < inHeight ) ;

      if( h0_valid ) {
	for (int64_t s=0; s<kernWidth; s++) {
	  int64_t w0 = (x+0) * strideWidth - padWidth + s * dilationWidth;

	  int64_t w0_valid = ( w0 >= 0 && w0 < inWidth ) ;

	  if( w0_valid ) {
	    int64_t inputIndex_h0w0  = inGroupOffset + ((n * inChannel + c) * inHeight + h0) * inWidth + w0;
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

	    __vr vriP_h0w0  = _vel_vshf_vvvsl(vri_h0w0, vri_h0w0, VE_VSHUFFLE_YUZU, vl) ;

	    __vr vrk01 = _vel_vshf_vvvsl(vrk0, vrk1, VE_VSHUFFLE_YUZU, vl) ;
	    vrsum01_y0x0 = _vel_pvfmad_vvvvvl(vrsum01_y0x0, vriP_h0w0, vrk01, vrsum01_y0x0, vl) ;

	    __vr vrk23 = _vel_vshf_vvvsl(vrk2, vrk3, VE_VSHUFFLE_YUZU, vl) ;
	    vrsum23_y0x0 = _vel_pvfmad_vvvvvl(vrsum23_y0x0, vriP_h0w0, vrk23, vrsum23_y0x0, vl) ;

	    __vr vrk45 = _vel_vshf_vvvsl(vrk4, vrk5, VE_VSHUFFLE_YUZU, vl) ;
	    vrsum45_y0x0 = _vel_pvfmad_vvvvvl(vrsum45_y0x0, vriP_h0w0, vrk45, vrsum45_y0x0, vl) ;

	    __vr vrk67 = _vel_vshf_vvvsl(vrk6, vrk7, VE_VSHUFFLE_YUZU, vl) ;
	    vrsum67_y0x0 = _vel_pvfmad_vvvvvl(vrsum67_y0x0, vriP_h0w0, vrk67, vrsum67_y0x0, vl) ;
	  }
	} // kernWidth
      }
    } // kernHeight
  } // inChannel

  {
    __vr vrsum0_y0x0 = _vel_vfsums_vvl(vrsum01_y0x0, VLEN) ;
    __vr vrsum1_y0x0 = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsum01_y0x0,32, VLEN), VLEN) ;
    __vr vrsum2_y0x0 = _vel_vfsums_vvl(vrsum23_y0x0, VLEN) ;
    __vr vrsum3_y0x0 = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsum23_y0x0,32, VLEN), VLEN) ;
    __vr vrsum4_y0x0 = _vel_vfsums_vvl(vrsum45_y0x0, VLEN) ;
    __vr vrsum5_y0x0 = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsum45_y0x0,32, VLEN), VLEN) ;
    __vr vrsum6_y0x0 = _vel_vfsums_vvl(vrsum67_y0x0, VLEN) ;
    __vr vrsum7_y0x0 = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsum67_y0x0,32, VLEN), VLEN) ;

    _vel_vstu_vssl(vrsum0_y0x0, 4, &pOut[outIndex+0*outHeight*outWidth], 1) ;
    _vel_vstu_vssl(vrsum1_y0x0, 4, &pOut[outIndex+1*outHeight*outWidth], 1) ;
    _vel_vstu_vssl(vrsum2_y0x0, 4, &pOut[outIndex+2*outHeight*outWidth], 1) ;
    _vel_vstu_vssl(vrsum3_y0x0, 4, &pOut[outIndex+3*outHeight*outWidth], 1) ;
    _vel_vstu_vssl(vrsum4_y0x0, 4, &pOut[outIndex+4*outHeight*outWidth], 1) ;
    _vel_vstu_vssl(vrsum5_y0x0, 4, &pOut[outIndex+5*outHeight*outWidth], 1) ;
    _vel_vstu_vssl(vrsum6_y0x0, 4, &pOut[outIndex+6*outHeight*outWidth], 1) ;
    _vel_vstu_vssl(vrsum7_y0x0, 4, &pOut[outIndex+7*outHeight*outWidth], 1) ;
  }
}

template<filterLayout_t FLAYOUT>
static inline void k8y1x2(
    const float * __restrict__ pIn,
    const float * __restrict__ pKernel,
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
    const int64_t kernGroupOffset,
    const int64_t n,
    const int64_t k,
    const int64_t y,
    const int64_t x
)
{

  int64_t outIndex  = outGroupOffset + ((n * outChannel + k) * outHeight + y) * outWidth + x;

  __vr vrsum01_y0x0 = _vel_vbrdl_vsl(0UL, VLEN) ;
  __vr vrsum01_y0x1 = _vel_vbrdl_vsl(0UL, VLEN) ;

  __vr vrsum23_y0x0 = _vel_vbrdl_vsl(0UL, VLEN) ;
  __vr vrsum23_y0x1 = _vel_vbrdl_vsl(0UL, VLEN) ;

  __vr vrsum45_y0x0 = _vel_vbrdl_vsl(0UL, VLEN) ;
  __vr vrsum45_y0x1 = _vel_vbrdl_vsl(0UL, VLEN) ;

  __vr vrsum67_y0x0 = _vel_vbrdl_vsl(0UL, VLEN) ;
  __vr vrsum67_y0x1 = _vel_vbrdl_vsl(0UL, VLEN) ;

  for (int64_t c=0; c<inChannelGroup; c+= VLEN) {
    const int64_t vl = inChannelGroup - c < VLEN ? inChannelGroup - c : VLEN ;

    for (int64_t r=0; r<kernHeight; r++) {
      int64_t h0 = (y+0) * strideHeight - padHeight + r * dilationHeight;

      int64_t h0_valid = ( h0 >= 0 && h0 < inHeight ) ;

      if( h0_valid  ) {
	for (int64_t s=0; s<kernWidth; s++) {
	  int64_t w0 = (x+0) * strideWidth - padWidth + s * dilationWidth;
	  int64_t w1 = (x+1) * strideWidth - padWidth + s * dilationWidth;

	  int64_t w0_valid = ( w0 >= 0 && w0 < inWidth ) ;
	  int64_t w1_valid = ( w1 >= 0 && w1 < inWidth ) ;

	  if( w0_valid && w1_valid ) {
	    int64_t inputIndex_h0w0  = inGroupOffset + ((n * inChannel + c) * inHeight + h0) * inWidth + w0;
	    int64_t inputIndex_h0w1  = inGroupOffset + ((n * inChannel + c) * inHeight + h0) * inWidth + w1;
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
	  }
	  else if( w0_valid ) {
	    int64_t inputIndex_h0w0  = inGroupOffset + ((n * inChannel + c) * inHeight + h0) * inWidth + w0;
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

	    __vr vriP_h0w0  = _vel_vshf_vvvsl(vri_h0w0, vri_h0w0, VE_VSHUFFLE_YUZU, vl) ;

	    __vr vrk01 = _vel_vshf_vvvsl(vrk0, vrk1, VE_VSHUFFLE_YUZU, vl) ;
	    vrsum01_y0x0 = _vel_pvfmad_vvvvvl(vrsum01_y0x0, vriP_h0w0, vrk01, vrsum01_y0x0, vl) ;

	    __vr vrk23 = _vel_vshf_vvvsl(vrk2, vrk3, VE_VSHUFFLE_YUZU, vl) ;
	    vrsum23_y0x0 = _vel_pvfmad_vvvvvl(vrsum23_y0x0, vriP_h0w0, vrk23, vrsum23_y0x0, vl) ;

	    __vr vrk45 = _vel_vshf_vvvsl(vrk4, vrk5, VE_VSHUFFLE_YUZU, vl) ;
	    vrsum45_y0x0 = _vel_pvfmad_vvvvvl(vrsum45_y0x0, vriP_h0w0, vrk45, vrsum45_y0x0, vl) ;

	    __vr vrk67 = _vel_vshf_vvvsl(vrk6, vrk7, VE_VSHUFFLE_YUZU, vl) ;
	    vrsum67_y0x0 = _vel_pvfmad_vvvvvl(vrsum67_y0x0, vriP_h0w0, vrk67, vrsum67_y0x0, vl) ;
	  }
	  else if( w1_valid ) {
	    int64_t inputIndex_h0w1  = inGroupOffset + ((n * inChannel + c) * inHeight + h0) * inWidth + w1;

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

	    __vr vriP_h0w1  = _vel_vshf_vvvsl(vri_h0w1, vri_h0w1, VE_VSHUFFLE_YUZU, vl) ;

	    __vr vrk01 = _vel_vshf_vvvsl(vrk0, vrk1, VE_VSHUFFLE_YUZU, vl) ;
	    vrsum01_y0x1 = _vel_pvfmad_vvvvvl(vrsum01_y0x1, vriP_h0w1, vrk01, vrsum01_y0x1, vl) ;

	    __vr vrk23 = _vel_vshf_vvvsl(vrk2, vrk3, VE_VSHUFFLE_YUZU, vl) ;
	    vrsum23_y0x1 = _vel_pvfmad_vvvvvl(vrsum23_y0x1, vriP_h0w1, vrk23, vrsum23_y0x1, vl) ;

	    __vr vrk45 = _vel_vshf_vvvsl(vrk4, vrk5, VE_VSHUFFLE_YUZU, vl) ;
	    vrsum45_y0x1 = _vel_pvfmad_vvvvvl(vrsum45_y0x1, vriP_h0w1, vrk45, vrsum45_y0x1, vl) ;

	    __vr vrk67 = _vel_vshf_vvvsl(vrk6, vrk7, VE_VSHUFFLE_YUZU, vl) ;
	    vrsum67_y0x1 = _vel_pvfmad_vvvvvl(vrsum67_y0x1, vriP_h0w1, vrk67, vrsum67_y0x1, vl) ;
	  }
	} // kernWidth
      }
    } // kernHeight
  } // inChannel

  {
    __vr vrsum0_y0x0 = _vel_vfsums_vvl(vrsum01_y0x0, VLEN) ;
    __vr vrsum1_y0x0 = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsum01_y0x0,32, VLEN), VLEN) ;
    __vr vrsum2_y0x0 = _vel_vfsums_vvl(vrsum23_y0x0, VLEN) ;
    __vr vrsum3_y0x0 = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsum23_y0x0,32, VLEN), VLEN) ;
    __vr vrsum4_y0x0 = _vel_vfsums_vvl(vrsum45_y0x0, VLEN) ;
    __vr vrsum5_y0x0 = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsum45_y0x0,32, VLEN), VLEN) ;
    __vr vrsum6_y0x0 = _vel_vfsums_vvl(vrsum67_y0x0, VLEN) ;
    __vr vrsum7_y0x0 = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsum67_y0x0,32, VLEN), VLEN) ;

    _vel_vstu_vssl(vrsum0_y0x0, 4, &pOut[outIndex+0*outHeight*outWidth], 1) ;
    _vel_vstu_vssl(vrsum1_y0x0, 4, &pOut[outIndex+1*outHeight*outWidth], 1) ;
    _vel_vstu_vssl(vrsum2_y0x0, 4, &pOut[outIndex+2*outHeight*outWidth], 1) ;
    _vel_vstu_vssl(vrsum3_y0x0, 4, &pOut[outIndex+3*outHeight*outWidth], 1) ;
    _vel_vstu_vssl(vrsum4_y0x0, 4, &pOut[outIndex+4*outHeight*outWidth], 1) ;
    _vel_vstu_vssl(vrsum5_y0x0, 4, &pOut[outIndex+5*outHeight*outWidth], 1) ;
    _vel_vstu_vssl(vrsum6_y0x0, 4, &pOut[outIndex+6*outHeight*outWidth], 1) ;
    _vel_vstu_vssl(vrsum7_y0x0, 4, &pOut[outIndex+7*outHeight*outWidth], 1) ;
  }
  {
    __vr vrsum0_y0x1 = _vel_vfsums_vvl(vrsum01_y0x1, VLEN) ;
    __vr vrsum1_y0x1 = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsum01_y0x1,32, VLEN), VLEN) ;
    __vr vrsum2_y0x1 = _vel_vfsums_vvl(vrsum23_y0x1, VLEN) ;
    __vr vrsum3_y0x1 = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsum23_y0x1,32, VLEN), VLEN) ;
    __vr vrsum4_y0x1 = _vel_vfsums_vvl(vrsum45_y0x1, VLEN) ;
    __vr vrsum5_y0x1 = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsum45_y0x1,32, VLEN), VLEN) ;
    __vr vrsum6_y0x1 = _vel_vfsums_vvl(vrsum67_y0x1, VLEN) ;
    __vr vrsum7_y0x1 = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsum67_y0x1,32, VLEN), VLEN) ;

    _vel_vstu_vssl(vrsum0_y0x1, 4, &pOut[outIndex+0*outHeight*outWidth+1], 1) ;
    _vel_vstu_vssl(vrsum1_y0x1, 4, &pOut[outIndex+1*outHeight*outWidth+1], 1) ;
    _vel_vstu_vssl(vrsum2_y0x1, 4, &pOut[outIndex+2*outHeight*outWidth+1], 1) ;
    _vel_vstu_vssl(vrsum3_y0x1, 4, &pOut[outIndex+3*outHeight*outWidth+1], 1) ;
    _vel_vstu_vssl(vrsum4_y0x1, 4, &pOut[outIndex+4*outHeight*outWidth+1], 1) ;
    _vel_vstu_vssl(vrsum5_y0x1, 4, &pOut[outIndex+5*outHeight*outWidth+1], 1) ;
    _vel_vstu_vssl(vrsum6_y0x1, 4, &pOut[outIndex+6*outHeight*outWidth+1], 1) ;
    _vel_vstu_vssl(vrsum7_y0x1, 4, &pOut[outIndex+7*outHeight*outWidth+1], 1) ;
  }
}


template<filterLayout_t FLAYOUT>
static inline void k8y2x1(
    const float * __restrict__ pIn,
    const float * __restrict__ pKernel,
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
    const int64_t kernGroupOffset,
    const int64_t n,
    const int64_t k,
    const int64_t y,
    const int64_t x
)
{

  int64_t outIndex  = outGroupOffset + ((n * outChannel + k) * outHeight + y) * outWidth + x;

  __vr vrsum01_y0x0 = _vel_vbrdl_vsl(0UL, VLEN) ;
  __vr vrsum01_y1x0 = _vel_vbrdl_vsl(0UL, VLEN) ;

  __vr vrsum23_y0x0 = _vel_vbrdl_vsl(0UL, VLEN) ;
  __vr vrsum23_y1x0 = _vel_vbrdl_vsl(0UL, VLEN) ;

  __vr vrsum45_y0x0 = _vel_vbrdl_vsl(0UL, VLEN) ;
  __vr vrsum45_y1x0 = _vel_vbrdl_vsl(0UL, VLEN) ;

  __vr vrsum67_y0x0 = _vel_vbrdl_vsl(0UL, VLEN) ;
  __vr vrsum67_y1x0 = _vel_vbrdl_vsl(0UL, VLEN) ;

  for (int64_t c=0; c<inChannelGroup; c+= VLEN) {
    const int64_t vl = inChannelGroup - c < VLEN ? inChannelGroup - c : VLEN ;

    for (int64_t r=0; r<kernHeight; r++) {
      int64_t h0 = (y+0) * strideHeight - padHeight + r * dilationHeight;
      int64_t h1 = (y+1) * strideHeight - padHeight + r * dilationHeight;

      int64_t h0_valid = ( h0 >= 0 && h0 < inHeight ) ;
      int64_t h1_valid = ( h1 >= 0 && h1 < inHeight ) ;

      if( h0_valid && h1_valid ) {
	for (int64_t s=0; s<kernWidth; s++) {
	  int64_t w0 = (x+0) * strideWidth - padWidth + s * dilationWidth;

	  int64_t w0_valid = ( w0 >= 0 && w0 < inWidth ) ;

	  if( w0_valid ) {
	    int64_t inputIndex_h0w0  = inGroupOffset + ((n * inChannel + c) * inHeight + h0) * inWidth + w0;
	    int64_t inputIndex_h1w0  = inGroupOffset + ((n * inChannel + c) * inHeight + h1) * inWidth + w0;
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
	  }
	} // kernWidth
      }
      else if( h0_valid ) {
	for (int64_t s=0; s<kernWidth; s++) {
	  int64_t w0 = (x+0) * strideWidth - padWidth + s * dilationWidth;

	  int64_t w0_valid = ( w0 >= 0 && w0 < inWidth ) ;

	  if( w0_valid ) {
	    int64_t inputIndex_h0w0  = inGroupOffset + ((n * inChannel + c) * inHeight + h0) * inWidth + w0;
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

	    __vr vriP_h0w0  = _vel_vshf_vvvsl(vri_h0w0, vri_h0w0, VE_VSHUFFLE_YUZU, vl) ;

	    __vr vrk01 = _vel_vshf_vvvsl(vrk0, vrk1, VE_VSHUFFLE_YUZU, vl) ;
	    vrsum01_y0x0 = _vel_pvfmad_vvvvvl(vrsum01_y0x0, vriP_h0w0, vrk01, vrsum01_y0x0, vl) ;

	    __vr vrk23 = _vel_vshf_vvvsl(vrk2, vrk3, VE_VSHUFFLE_YUZU, vl) ;
	    vrsum23_y0x0 = _vel_pvfmad_vvvvvl(vrsum23_y0x0, vriP_h0w0, vrk23, vrsum23_y0x0, vl) ;

	    __vr vrk45 = _vel_vshf_vvvsl(vrk4, vrk5, VE_VSHUFFLE_YUZU, vl) ;
	    vrsum45_y0x0 = _vel_pvfmad_vvvvvl(vrsum45_y0x0, vriP_h0w0, vrk45, vrsum45_y0x0, vl) ;

	    __vr vrk67 = _vel_vshf_vvvsl(vrk6, vrk7, VE_VSHUFFLE_YUZU, vl) ;
	    vrsum67_y0x0 = _vel_pvfmad_vvvvvl(vrsum67_y0x0, vriP_h0w0, vrk67, vrsum67_y0x0, vl) ;
	  }
	} // kernWidth
      }
      else if( h1_valid ) {
	for (int64_t s=0; s<kernWidth; s++) {
	  int64_t w0 = (x+0) * strideWidth - padWidth + s * dilationWidth;

	  int64_t w0_valid = ( w0 >= 0 && w0 < inWidth ) ;

	  if( w0_valid ) {
	    int64_t inputIndex_h1w0  = inGroupOffset + ((n * inChannel + c) * inHeight + h1) * inWidth + w0;
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

	    __vr vriP_h1w0  = _vel_vshf_vvvsl(vri_h1w0, vri_h1w0, VE_VSHUFFLE_YUZU, vl) ;

	    __vr vrk01 = _vel_vshf_vvvsl(vrk0, vrk1, VE_VSHUFFLE_YUZU, vl) ;
	    vrsum01_y1x0 = _vel_pvfmad_vvvvvl(vrsum01_y1x0, vriP_h1w0, vrk01, vrsum01_y1x0, vl) ;

	    __vr vrk23 = _vel_vshf_vvvsl(vrk2, vrk3, VE_VSHUFFLE_YUZU, vl) ;
	    vrsum23_y1x0 = _vel_pvfmad_vvvvvl(vrsum23_y1x0, vriP_h1w0, vrk23, vrsum23_y1x0, vl) ;

	    __vr vrk45 = _vel_vshf_vvvsl(vrk4, vrk5, VE_VSHUFFLE_YUZU, vl) ;
	    vrsum45_y1x0 = _vel_pvfmad_vvvvvl(vrsum45_y1x0, vriP_h1w0, vrk45, vrsum45_y1x0, vl) ;

	    __vr vrk67 = _vel_vshf_vvvsl(vrk6, vrk7, VE_VSHUFFLE_YUZU, vl) ;
	    vrsum67_y1x0 = _vel_pvfmad_vvvvvl(vrsum67_y1x0, vriP_h1w0, vrk67, vrsum67_y1x0, vl) ;
	  }
	} // kernWidth
      }
    } // kernHeight
  } // inChannel

  {
    __vr vrsum0_y0x0 = _vel_vfsums_vvl(vrsum01_y0x0, VLEN) ;
    __vr vrsum1_y0x0 = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsum01_y0x0,32, VLEN), VLEN) ;
    __vr vrsum2_y0x0 = _vel_vfsums_vvl(vrsum23_y0x0, VLEN) ;
    __vr vrsum3_y0x0 = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsum23_y0x0,32, VLEN), VLEN) ;
    __vr vrsum4_y0x0 = _vel_vfsums_vvl(vrsum45_y0x0, VLEN) ;
    __vr vrsum5_y0x0 = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsum45_y0x0,32, VLEN), VLEN) ;
    __vr vrsum6_y0x0 = _vel_vfsums_vvl(vrsum67_y0x0, VLEN) ;
    __vr vrsum7_y0x0 = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsum67_y0x0,32, VLEN), VLEN) ;

    _vel_vstu_vssl(vrsum0_y0x0, 4, &pOut[outIndex+0*outHeight*outWidth], 1) ;
    _vel_vstu_vssl(vrsum1_y0x0, 4, &pOut[outIndex+1*outHeight*outWidth], 1) ;
    _vel_vstu_vssl(vrsum2_y0x0, 4, &pOut[outIndex+2*outHeight*outWidth], 1) ;
    _vel_vstu_vssl(vrsum3_y0x0, 4, &pOut[outIndex+3*outHeight*outWidth], 1) ;
    _vel_vstu_vssl(vrsum4_y0x0, 4, &pOut[outIndex+4*outHeight*outWidth], 1) ;
    _vel_vstu_vssl(vrsum5_y0x0, 4, &pOut[outIndex+5*outHeight*outWidth], 1) ;
    _vel_vstu_vssl(vrsum6_y0x0, 4, &pOut[outIndex+6*outHeight*outWidth], 1) ;
    _vel_vstu_vssl(vrsum7_y0x0, 4, &pOut[outIndex+7*outHeight*outWidth], 1) ;
  }
  {
    __vr vrsum0_y1x0 = _vel_vfsums_vvl(vrsum01_y1x0, VLEN) ;
    __vr vrsum1_y1x0 = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsum01_y1x0,32, VLEN), VLEN) ;
    __vr vrsum2_y1x0 = _vel_vfsums_vvl(vrsum23_y1x0, VLEN) ;
    __vr vrsum3_y1x0 = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsum23_y1x0,32, VLEN), VLEN) ;
    __vr vrsum4_y1x0 = _vel_vfsums_vvl(vrsum45_y1x0, VLEN) ;
    __vr vrsum5_y1x0 = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsum45_y1x0,32, VLEN), VLEN) ;
    __vr vrsum6_y1x0 = _vel_vfsums_vvl(vrsum67_y1x0, VLEN) ;
    __vr vrsum7_y1x0 = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsum67_y1x0,32, VLEN), VLEN) ;

    _vel_vstu_vssl(vrsum0_y1x0, 4, &pOut[outIndex+0*outHeight*outWidth+outWidth], 1) ;
    _vel_vstu_vssl(vrsum1_y1x0, 4, &pOut[outIndex+1*outHeight*outWidth+outWidth], 1) ;
    _vel_vstu_vssl(vrsum2_y1x0, 4, &pOut[outIndex+2*outHeight*outWidth+outWidth], 1) ;
    _vel_vstu_vssl(vrsum3_y1x0, 4, &pOut[outIndex+3*outHeight*outWidth+outWidth], 1) ;
    _vel_vstu_vssl(vrsum4_y1x0, 4, &pOut[outIndex+4*outHeight*outWidth+outWidth], 1) ;
    _vel_vstu_vssl(vrsum5_y1x0, 4, &pOut[outIndex+5*outHeight*outWidth+outWidth], 1) ;
    _vel_vstu_vssl(vrsum6_y1x0, 4, &pOut[outIndex+6*outHeight*outWidth+outWidth], 1) ;
    _vel_vstu_vssl(vrsum7_y1x0, 4, &pOut[outIndex+7*outHeight*outWidth+outWidth], 1) ;
  }
}

template<filterLayout_t FLAYOUT>
static inline void k8y2x2(
    const float * __restrict__ pIn,
    const float * __restrict__ pKernel,
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
    const int64_t kernGroupOffset,
    const int64_t n,
    const int64_t k,
    const int64_t y,
    const int64_t x
)
{

  int64_t outIndex  = outGroupOffset + ((n * outChannel + k) * outHeight + y) * outWidth + x;

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

  for (int64_t c=0; c<inChannelGroup; c+= VLEN) {
    const int64_t vl = inChannelGroup - c < VLEN ? inChannelGroup - c : VLEN ;

    for (int64_t r=0; r<kernHeight; r++) {
      int64_t h0 = (y+0) * strideHeight - padHeight + r * dilationHeight;
      int64_t h1 = (y+1) * strideHeight - padHeight + r * dilationHeight;

      int64_t h0_valid = ( h0 >= 0 && h0 < inHeight ) ;
      int64_t h1_valid = ( h1 >= 0 && h1 < inHeight ) ;

      if( h0_valid && h1_valid ) {
	for (int64_t s=0; s<kernWidth; s++) {
	  int64_t w0 = (x+0) * strideWidth - padWidth + s * dilationWidth;
	  int64_t w1 = (x+1) * strideWidth - padWidth + s * dilationWidth;

	  int64_t w0_valid = ( w0 >= 0 && w0 < inWidth ) ;
	  int64_t w1_valid = ( w1 >= 0 && w1 < inWidth ) ;

	  if( w0_valid && w1_valid ) {
	    int64_t inputIndex_h0w0  = inGroupOffset + ((n * inChannel + c) * inHeight + h0) * inWidth + w0;
	    int64_t inputIndex_h0w1  = inGroupOffset + ((n * inChannel + c) * inHeight + h0) * inWidth + w1;
	    int64_t inputIndex_h1w0  = inGroupOffset + ((n * inChannel + c) * inHeight + h1) * inWidth + w0;
	    int64_t inputIndex_h1w1  = inGroupOffset + ((n * inChannel + c) * inHeight + h1) * inWidth + w1;
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
	  }
	  else if( w0_valid ) {
	    int64_t inputIndex_h0w0  = inGroupOffset + ((n * inChannel + c) * inHeight + h0) * inWidth + w0;
	    int64_t inputIndex_h1w0  = inGroupOffset + ((n * inChannel + c) * inHeight + h1) * inWidth + w0;
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
	  }
	  else if( w1_valid ) {
	    int64_t inputIndex_h0w1  = inGroupOffset + ((n * inChannel + c) * inHeight + h0) * inWidth + w1;
	    int64_t inputIndex_h1w1  = inGroupOffset + ((n * inChannel + c) * inHeight + h1) * inWidth + w1;
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
	  }
	  else if( w0_valid ) {
	    int64_t inputIndex_h0w0  = inGroupOffset + ((n * inChannel + c) * inHeight + h0) * inWidth + w0;
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

	    __vr vriP_h0w0  = _vel_vshf_vvvsl(vri_h0w0, vri_h0w0, VE_VSHUFFLE_YUZU, vl) ;

	    __vr vrk01 = _vel_vshf_vvvsl(vrk0, vrk1, VE_VSHUFFLE_YUZU, vl) ;
	    vrsum01_y0x0 = _vel_pvfmad_vvvvvl(vrsum01_y0x0, vriP_h0w0, vrk01, vrsum01_y0x0, vl) ;

	    __vr vrk23 = _vel_vshf_vvvsl(vrk2, vrk3, VE_VSHUFFLE_YUZU, vl) ;
	    vrsum23_y0x0 = _vel_pvfmad_vvvvvl(vrsum23_y0x0, vriP_h0w0, vrk23, vrsum23_y0x0, vl) ;

	    __vr vrk45 = _vel_vshf_vvvsl(vrk4, vrk5, VE_VSHUFFLE_YUZU, vl) ;
	    vrsum45_y0x0 = _vel_pvfmad_vvvvvl(vrsum45_y0x0, vriP_h0w0, vrk45, vrsum45_y0x0, vl) ;

	    __vr vrk67 = _vel_vshf_vvvsl(vrk6, vrk7, VE_VSHUFFLE_YUZU, vl) ;
	    vrsum67_y0x0 = _vel_pvfmad_vvvvvl(vrsum67_y0x0, vriP_h0w0, vrk67, vrsum67_y0x0, vl) ;
	  }
	  else if( w1_valid ) {
	    int64_t inputIndex_h0w1  = inGroupOffset + ((n * inChannel + c) * inHeight + h0) * inWidth + w1;
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

	    __vr vriP_h0w1  = _vel_vshf_vvvsl(vri_h0w1, vri_h0w1, VE_VSHUFFLE_YUZU, vl) ;

	    __vr vrk01 = _vel_vshf_vvvsl(vrk0, vrk1, VE_VSHUFFLE_YUZU, vl) ;
	    vrsum01_y0x1 = _vel_pvfmad_vvvvvl(vrsum01_y0x1, vriP_h0w1, vrk01, vrsum01_y0x1, vl) ;

	    __vr vrk23 = _vel_vshf_vvvsl(vrk2, vrk3, VE_VSHUFFLE_YUZU, vl) ;
	    vrsum23_y0x1 = _vel_pvfmad_vvvvvl(vrsum23_y0x1, vriP_h0w1, vrk23, vrsum23_y0x1, vl) ;

	    __vr vrk45 = _vel_vshf_vvvsl(vrk4, vrk5, VE_VSHUFFLE_YUZU, vl) ;
	    vrsum45_y0x1 = _vel_pvfmad_vvvvvl(vrsum45_y0x1, vriP_h0w1, vrk45, vrsum45_y0x1, vl) ;

	    __vr vrk67 = _vel_vshf_vvvsl(vrk6, vrk7, VE_VSHUFFLE_YUZU, vl) ;
	    vrsum67_y0x1 = _vel_pvfmad_vvvvvl(vrsum67_y0x1, vriP_h0w1, vrk67, vrsum67_y0x1, vl) ;
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
	  }
	  else if( w0_valid ) {
	    int64_t inputIndex_h1w0  = inGroupOffset + ((n * inChannel + c) * inHeight + h1) * inWidth + w0;
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

	    __vr vriP_h1w0  = _vel_vshf_vvvsl(vri_h1w0, vri_h1w0, VE_VSHUFFLE_YUZU, vl) ;

	    __vr vrk01 = _vel_vshf_vvvsl(vrk0, vrk1, VE_VSHUFFLE_YUZU, vl) ;
	    vrsum01_y1x0 = _vel_pvfmad_vvvvvl(vrsum01_y1x0, vriP_h1w0, vrk01, vrsum01_y1x0, vl) ;

	    __vr vrk23 = _vel_vshf_vvvsl(vrk2, vrk3, VE_VSHUFFLE_YUZU, vl) ;
	    vrsum23_y1x0 = _vel_pvfmad_vvvvvl(vrsum23_y1x0, vriP_h1w0, vrk23, vrsum23_y1x0, vl) ;

	    __vr vrk45 = _vel_vshf_vvvsl(vrk4, vrk5, VE_VSHUFFLE_YUZU, vl) ;
	    vrsum45_y1x0 = _vel_pvfmad_vvvvvl(vrsum45_y1x0, vriP_h1w0, vrk45, vrsum45_y1x0, vl) ;

	    __vr vrk67 = _vel_vshf_vvvsl(vrk6, vrk7, VE_VSHUFFLE_YUZU, vl) ;
	    vrsum67_y1x0 = _vel_pvfmad_vvvvvl(vrsum67_y1x0, vriP_h1w0, vrk67, vrsum67_y1x0, vl) ;
	  }
	  else if( w1_valid ) {
	    int64_t inputIndex_h1w1  = inGroupOffset + ((n * inChannel + c) * inHeight + h1) * inWidth + w1;
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

	    __vr vriP_h1w1  = _vel_vshf_vvvsl(vri_h1w1, vri_h1w1, VE_VSHUFFLE_YUZU, vl) ;

	    __vr vrk01 = _vel_vshf_vvvsl(vrk0, vrk1, VE_VSHUFFLE_YUZU, vl) ;
	    vrsum01_y1x1 = _vel_pvfmad_vvvvvl(vrsum01_y1x1, vriP_h1w1, vrk01, vrsum01_y1x1, vl) ;

	    __vr vrk23 = _vel_vshf_vvvsl(vrk2, vrk3, VE_VSHUFFLE_YUZU, vl) ;
	    vrsum23_y1x1 = _vel_pvfmad_vvvvvl(vrsum23_y1x1, vriP_h1w1, vrk23, vrsum23_y1x1, vl) ;

	    __vr vrk45 = _vel_vshf_vvvsl(vrk4, vrk5, VE_VSHUFFLE_YUZU, vl) ;
	    vrsum45_y1x1 = _vel_pvfmad_vvvvvl(vrsum45_y1x1, vriP_h1w1, vrk45, vrsum45_y1x1, vl) ;

	    __vr vrk67 = _vel_vshf_vvvsl(vrk6, vrk7, VE_VSHUFFLE_YUZU, vl) ;
	    vrsum67_y1x1 = _vel_pvfmad_vvvvvl(vrsum67_y1x1, vriP_h1w1, vrk67, vrsum67_y1x1, vl) ;
	  }
	} // kernWidth
      }
    } // kernHeight
  } // inChannel

  {
    __vr vrsum0_y0x0 = _vel_vfsums_vvl(vrsum01_y0x0, VLEN) ;
    __vr vrsum1_y0x0 = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsum01_y0x0,32, VLEN), VLEN) ;
    __vr vrsum2_y0x0 = _vel_vfsums_vvl(vrsum23_y0x0, VLEN) ;
    __vr vrsum3_y0x0 = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsum23_y0x0,32, VLEN), VLEN) ;
    __vr vrsum4_y0x0 = _vel_vfsums_vvl(vrsum45_y0x0, VLEN) ;
    __vr vrsum5_y0x0 = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsum45_y0x0,32, VLEN), VLEN) ;
    __vr vrsum6_y0x0 = _vel_vfsums_vvl(vrsum67_y0x0, VLEN) ;
    __vr vrsum7_y0x0 = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsum67_y0x0,32, VLEN), VLEN) ;

    _vel_vstu_vssl(vrsum0_y0x0, 4, &pOut[outIndex+0*outHeight*outWidth], 1) ;
    _vel_vstu_vssl(vrsum1_y0x0, 4, &pOut[outIndex+1*outHeight*outWidth], 1) ;
    _vel_vstu_vssl(vrsum2_y0x0, 4, &pOut[outIndex+2*outHeight*outWidth], 1) ;
    _vel_vstu_vssl(vrsum3_y0x0, 4, &pOut[outIndex+3*outHeight*outWidth], 1) ;
    _vel_vstu_vssl(vrsum4_y0x0, 4, &pOut[outIndex+4*outHeight*outWidth], 1) ;
    _vel_vstu_vssl(vrsum5_y0x0, 4, &pOut[outIndex+5*outHeight*outWidth], 1) ;
    _vel_vstu_vssl(vrsum6_y0x0, 4, &pOut[outIndex+6*outHeight*outWidth], 1) ;
    _vel_vstu_vssl(vrsum7_y0x0, 4, &pOut[outIndex+7*outHeight*outWidth], 1) ;
  }
  {
    __vr vrsum0_y0x1 = _vel_vfsums_vvl(vrsum01_y0x1, VLEN) ;
    __vr vrsum1_y0x1 = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsum01_y0x1,32, VLEN), VLEN) ;
    __vr vrsum2_y0x1 = _vel_vfsums_vvl(vrsum23_y0x1, VLEN) ;
    __vr vrsum3_y0x1 = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsum23_y0x1,32, VLEN), VLEN) ;
    __vr vrsum4_y0x1 = _vel_vfsums_vvl(vrsum45_y0x1, VLEN) ;
    __vr vrsum5_y0x1 = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsum45_y0x1,32, VLEN), VLEN) ;
    __vr vrsum6_y0x1 = _vel_vfsums_vvl(vrsum67_y0x1, VLEN) ;
    __vr vrsum7_y0x1 = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsum67_y0x1,32, VLEN), VLEN) ;

    _vel_vstu_vssl(vrsum0_y0x1, 4, &pOut[outIndex+0*outHeight*outWidth+1], 1) ;
    _vel_vstu_vssl(vrsum1_y0x1, 4, &pOut[outIndex+1*outHeight*outWidth+1], 1) ;
    _vel_vstu_vssl(vrsum2_y0x1, 4, &pOut[outIndex+2*outHeight*outWidth+1], 1) ;
    _vel_vstu_vssl(vrsum3_y0x1, 4, &pOut[outIndex+3*outHeight*outWidth+1], 1) ;
    _vel_vstu_vssl(vrsum4_y0x1, 4, &pOut[outIndex+4*outHeight*outWidth+1], 1) ;
    _vel_vstu_vssl(vrsum5_y0x1, 4, &pOut[outIndex+5*outHeight*outWidth+1], 1) ;
    _vel_vstu_vssl(vrsum6_y0x1, 4, &pOut[outIndex+6*outHeight*outWidth+1], 1) ;
    _vel_vstu_vssl(vrsum7_y0x1, 4, &pOut[outIndex+7*outHeight*outWidth+1], 1) ;
  }
  {
    __vr vrsum0_y1x0 = _vel_vfsums_vvl(vrsum01_y1x0, VLEN) ;
    __vr vrsum1_y1x0 = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsum01_y1x0,32, VLEN), VLEN) ;
    __vr vrsum2_y1x0 = _vel_vfsums_vvl(vrsum23_y1x0, VLEN) ;
    __vr vrsum3_y1x0 = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsum23_y1x0,32, VLEN), VLEN) ;
    __vr vrsum4_y1x0 = _vel_vfsums_vvl(vrsum45_y1x0, VLEN) ;
    __vr vrsum5_y1x0 = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsum45_y1x0,32, VLEN), VLEN) ;
    __vr vrsum6_y1x0 = _vel_vfsums_vvl(vrsum67_y1x0, VLEN) ;
    __vr vrsum7_y1x0 = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsum67_y1x0,32, VLEN), VLEN) ;

    _vel_vstu_vssl(vrsum0_y1x0, 4, &pOut[outIndex+0*outHeight*outWidth+outWidth], 1) ;
    _vel_vstu_vssl(vrsum1_y1x0, 4, &pOut[outIndex+1*outHeight*outWidth+outWidth], 1) ;
    _vel_vstu_vssl(vrsum2_y1x0, 4, &pOut[outIndex+2*outHeight*outWidth+outWidth], 1) ;
    _vel_vstu_vssl(vrsum3_y1x0, 4, &pOut[outIndex+3*outHeight*outWidth+outWidth], 1) ;
    _vel_vstu_vssl(vrsum4_y1x0, 4, &pOut[outIndex+4*outHeight*outWidth+outWidth], 1) ;
    _vel_vstu_vssl(vrsum5_y1x0, 4, &pOut[outIndex+5*outHeight*outWidth+outWidth], 1) ;
    _vel_vstu_vssl(vrsum6_y1x0, 4, &pOut[outIndex+6*outHeight*outWidth+outWidth], 1) ;
    _vel_vstu_vssl(vrsum7_y1x0, 4, &pOut[outIndex+7*outHeight*outWidth+outWidth], 1) ;
  }
  {
    __vr vrsum0_y1x1 = _vel_vfsums_vvl(vrsum01_y1x1, VLEN) ;
    __vr vrsum1_y1x1 = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsum01_y1x1,32, VLEN), VLEN) ;
    __vr vrsum2_y1x1 = _vel_vfsums_vvl(vrsum23_y1x1, VLEN) ;
    __vr vrsum3_y1x1 = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsum23_y1x1,32, VLEN), VLEN) ;
    __vr vrsum4_y1x1 = _vel_vfsums_vvl(vrsum45_y1x1, VLEN) ;
    __vr vrsum5_y1x1 = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsum45_y1x1,32, VLEN), VLEN) ;
    __vr vrsum6_y1x1 = _vel_vfsums_vvl(vrsum67_y1x1, VLEN) ;
    __vr vrsum7_y1x1 = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsum67_y1x1,32, VLEN), VLEN) ;

    _vel_vstu_vssl(vrsum0_y1x1, 4, &pOut[outIndex+0*outHeight*outWidth+outWidth+1], 1) ;
    _vel_vstu_vssl(vrsum1_y1x1, 4, &pOut[outIndex+1*outHeight*outWidth+outWidth+1], 1) ;
    _vel_vstu_vssl(vrsum2_y1x1, 4, &pOut[outIndex+2*outHeight*outWidth+outWidth+1], 1) ;
    _vel_vstu_vssl(vrsum3_y1x1, 4, &pOut[outIndex+3*outHeight*outWidth+outWidth+1], 1) ;
    _vel_vstu_vssl(vrsum4_y1x1, 4, &pOut[outIndex+4*outHeight*outWidth+outWidth+1], 1) ;
    _vel_vstu_vssl(vrsum5_y1x1, 4, &pOut[outIndex+5*outHeight*outWidth+outWidth+1], 1) ;
    _vel_vstu_vssl(vrsum6_y1x1, 4, &pOut[outIndex+6*outHeight*outWidth+outWidth+1], 1) ;
    _vel_vstu_vssl(vrsum7_y1x1, 4, &pOut[outIndex+7*outHeight*outWidth+outWidth+1], 1) ;
  }
}

template<filterLayout_t FLAYOUT>
static inline void k16y1x1(
    const float * __restrict__ pIn,
    const float * __restrict__ pKernel,
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
    const int64_t kernGroupOffset,
    const int64_t n,
    const int64_t k,
    const int64_t y,
    const int64_t x
)
{

  int64_t outIndex  = outGroupOffset + ((n * outChannel + k) * outHeight + y) * outWidth + x;

  __vr vrsum01_y0x0 = _vel_vbrdl_vsl(0UL, VLEN) ;
  __vr vrsum23_y0x0 = _vel_vbrdl_vsl(0UL, VLEN) ;
  __vr vrsum45_y0x0 = _vel_vbrdl_vsl(0UL, VLEN) ;
  __vr vrsum67_y0x0 = _vel_vbrdl_vsl(0UL, VLEN) ;
  __vr vrsum89_y0x0 = _vel_vbrdl_vsl(0UL, VLEN) ;
  __vr vrsumAB_y0x0 = _vel_vbrdl_vsl(0UL, VLEN) ;
  __vr vrsumCD_y0x0 = _vel_vbrdl_vsl(0UL, VLEN) ;
  __vr vrsumEF_y0x0 = _vel_vbrdl_vsl(0UL, VLEN) ;

  for (int64_t c=0; c<inChannelGroup; c+= VLEN) {
    const int64_t vl = inChannelGroup - c < VLEN ? inChannelGroup - c : VLEN ;

    for (int64_t r=0; r<kernHeight; r++) {
      int64_t h0 = (y+0) * strideHeight - padHeight + r * dilationHeight;

      int64_t h0_valid = ( h0 >= 0 && h0 < inHeight ) ;

      if( h0_valid ) {
	for (int64_t s=0; s<kernWidth; s++) {
	  int64_t w0 = (x+0) * strideWidth - padWidth + s * dilationWidth;

	  int64_t w0_valid = ( w0 >= 0 && w0 < inWidth ) ;

	  if( w0_valid ) {
	    int64_t inputIndex_h0w0  = inGroupOffset + ((n * inChannel + c) * inHeight + h0) * inWidth + w0;

	    int64_t kernelIndex = kernGroupOffset + ((k * inChannelGroup + c) * kernHeight + r) * kernWidth + s;

	    __vr vri_h0w0 = _vel_vldu_vssl(4*inHeight*inWidth, &pIn[inputIndex_h0w0], vl) ;

	    __vr vrk0 = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+0*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrk1 = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+1*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrk2 = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+2*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrk3 = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+3*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrk4 = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+4*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrk5 = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+5*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrk6 = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+6*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrk7 = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+7*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrk8 = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+8*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrk9 = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+9*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrkA = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+10*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrkB = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+11*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrkC = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+12*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrkD = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+13*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrkE = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+14*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrkF = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+15*inChannelGroup*kernHeight*kernWidth], vl) ;

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
	} // kernWidth
      }
    } // kernHeight
  } // inChannel

  {
    __vr vrsum0_y0x0 = _vel_vfsums_vvl(vrsum01_y0x0, VLEN) ;
    __vr vrsum1_y0x0 = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsum01_y0x0,32, VLEN), VLEN) ;
    __vr vrsum2_y0x0 = _vel_vfsums_vvl(vrsum23_y0x0, VLEN) ;
    __vr vrsum3_y0x0 = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsum23_y0x0,32, VLEN), VLEN) ;
    __vr vrsum4_y0x0 = _vel_vfsums_vvl(vrsum45_y0x0, VLEN) ;
    __vr vrsum5_y0x0 = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsum45_y0x0,32, VLEN), VLEN) ;
    __vr vrsum6_y0x0 = _vel_vfsums_vvl(vrsum67_y0x0, VLEN) ;
    __vr vrsum7_y0x0 = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsum67_y0x0,32, VLEN), VLEN) ;
    __vr vrsum8_y0x0 = _vel_vfsums_vvl(vrsum89_y0x0, VLEN) ;
    __vr vrsum9_y0x0 = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsum89_y0x0,32, VLEN), VLEN) ;
    __vr vrsumA_y0x0 = _vel_vfsums_vvl(vrsumAB_y0x0, VLEN) ;
    __vr vrsumB_y0x0 = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsumAB_y0x0,32, VLEN), VLEN) ;
    __vr vrsumC_y0x0 = _vel_vfsums_vvl(vrsumCD_y0x0, VLEN) ;
    __vr vrsumD_y0x0 = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsumCD_y0x0,32, VLEN), VLEN) ;
    __vr vrsumE_y0x0 = _vel_vfsums_vvl(vrsumEF_y0x0, VLEN) ;
    __vr vrsumF_y0x0 = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsumEF_y0x0,32, VLEN), VLEN) ;

    _vel_vstu_vssl(vrsum0_y0x0, 4, &pOut[outIndex+0*outHeight*outWidth], 1) ;
    _vel_vstu_vssl(vrsum1_y0x0, 4, &pOut[outIndex+1*outHeight*outWidth], 1) ;
    _vel_vstu_vssl(vrsum2_y0x0, 4, &pOut[outIndex+2*outHeight*outWidth], 1) ;
    _vel_vstu_vssl(vrsum3_y0x0, 4, &pOut[outIndex+3*outHeight*outWidth], 1) ;
    _vel_vstu_vssl(vrsum4_y0x0, 4, &pOut[outIndex+4*outHeight*outWidth], 1) ;
    _vel_vstu_vssl(vrsum5_y0x0, 4, &pOut[outIndex+5*outHeight*outWidth], 1) ;
    _vel_vstu_vssl(vrsum6_y0x0, 4, &pOut[outIndex+6*outHeight*outWidth], 1) ;
    _vel_vstu_vssl(vrsum7_y0x0, 4, &pOut[outIndex+7*outHeight*outWidth], 1) ;
    _vel_vstu_vssl(vrsum8_y0x0, 4, &pOut[outIndex+8*outHeight*outWidth], 1) ;
    _vel_vstu_vssl(vrsum9_y0x0, 4, &pOut[outIndex+9*outHeight*outWidth], 1) ;
    _vel_vstu_vssl(vrsumA_y0x0, 4, &pOut[outIndex+10*outHeight*outWidth], 1) ;
    _vel_vstu_vssl(vrsumB_y0x0, 4, &pOut[outIndex+11*outHeight*outWidth], 1) ;
    _vel_vstu_vssl(vrsumC_y0x0, 4, &pOut[outIndex+12*outHeight*outWidth], 1) ;
    _vel_vstu_vssl(vrsumD_y0x0, 4, &pOut[outIndex+13*outHeight*outWidth], 1) ;
    _vel_vstu_vssl(vrsumE_y0x0, 4, &pOut[outIndex+14*outHeight*outWidth], 1) ;
    _vel_vstu_vssl(vrsumF_y0x0, 4, &pOut[outIndex+15*outHeight*outWidth], 1) ;
  }
}

template<filterLayout_t FLAYOUT>
static inline void k16y1x2(
    const float * __restrict__ pIn,
    const float * __restrict__ pKernel,
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
    const int64_t kernGroupOffset,
    const int64_t n,
    const int64_t k,
    const int64_t y,
    const int64_t x
)
{

  int64_t outIndex  = outGroupOffset + ((n * outChannel + k) * outHeight + y) * outWidth + x;

  __vr vrsum01_y0x0 = _vel_vbrdl_vsl(0UL, VLEN) ;
  __vr vrsum01_y0x1 = _vel_vbrdl_vsl(0UL, VLEN) ;

  __vr vrsum23_y0x0 = _vel_vbrdl_vsl(0UL, VLEN) ;
  __vr vrsum23_y0x1 = _vel_vbrdl_vsl(0UL, VLEN) ;

  __vr vrsum45_y0x0 = _vel_vbrdl_vsl(0UL, VLEN) ;
  __vr vrsum45_y0x1 = _vel_vbrdl_vsl(0UL, VLEN) ;

  __vr vrsum67_y0x0 = _vel_vbrdl_vsl(0UL, VLEN) ;
  __vr vrsum67_y0x1 = _vel_vbrdl_vsl(0UL, VLEN) ;

  __vr vrsum89_y0x0 = _vel_vbrdl_vsl(0UL, VLEN) ;
  __vr vrsum89_y0x1 = _vel_vbrdl_vsl(0UL, VLEN) ;

  __vr vrsumAB_y0x0 = _vel_vbrdl_vsl(0UL, VLEN) ;
  __vr vrsumAB_y0x1 = _vel_vbrdl_vsl(0UL, VLEN) ;

  __vr vrsumCD_y0x0 = _vel_vbrdl_vsl(0UL, VLEN) ;
  __vr vrsumCD_y0x1 = _vel_vbrdl_vsl(0UL, VLEN) ;

  __vr vrsumEF_y0x0 = _vel_vbrdl_vsl(0UL, VLEN) ;
  __vr vrsumEF_y0x1 = _vel_vbrdl_vsl(0UL, VLEN) ;

  for (int64_t c=0; c<inChannelGroup; c+= VLEN) {
    const int64_t vl = inChannelGroup - c < VLEN ? inChannelGroup - c : VLEN ;

    for (int64_t r=0; r<kernHeight; r++) {
      int64_t h0 = (y+0) * strideHeight - padHeight + r * dilationHeight;

      int64_t h0_valid = ( h0 >= 0 && h0 < inHeight ) ;

      if( h0_valid  ) {
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

	    __vr vrk0 = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+0*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrk1 = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+1*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrk2 = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+2*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrk3 = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+3*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrk4 = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+4*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrk5 = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+5*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrk6 = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+6*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrk7 = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+7*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrk8 = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+8*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrk9 = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+9*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrkA = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+10*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrkB = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+11*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrkC = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+12*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrkD = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+13*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrkE = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+14*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrkF = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+15*inChannelGroup*kernHeight*kernWidth], vl) ;

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

	    __vr vrk0 = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+0*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrk1 = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+1*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrk2 = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+2*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrk3 = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+3*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrk4 = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+4*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrk5 = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+5*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrk6 = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+6*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrk7 = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+7*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrk8 = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+8*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrk9 = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+9*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrkA = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+10*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrkB = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+11*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrkC = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+12*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrkD = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+13*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrkE = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+14*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrkF = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+15*inChannelGroup*kernHeight*kernWidth], vl) ;

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

	    __vr vrk0 = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+0*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrk1 = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+1*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrk2 = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+2*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrk3 = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+3*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrk4 = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+4*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrk5 = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+5*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrk6 = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+6*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrk7 = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+7*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrk8 = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+8*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrk9 = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+9*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrkA = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+10*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrkB = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+11*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrkC = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+12*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrkD = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+13*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrkE = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+14*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrkF = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+15*inChannelGroup*kernHeight*kernWidth], vl) ;

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
    } // kernHeight
  } // inChannel

  {
    __vr vrsum0_y0x0 = _vel_vfsums_vvl(vrsum01_y0x0, VLEN) ;
    __vr vrsum1_y0x0 = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsum01_y0x0,32, VLEN), VLEN) ;
    __vr vrsum2_y0x0 = _vel_vfsums_vvl(vrsum23_y0x0, VLEN) ;
    __vr vrsum3_y0x0 = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsum23_y0x0,32, VLEN), VLEN) ;
    __vr vrsum4_y0x0 = _vel_vfsums_vvl(vrsum45_y0x0, VLEN) ;
    __vr vrsum5_y0x0 = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsum45_y0x0,32, VLEN), VLEN) ;
    __vr vrsum6_y0x0 = _vel_vfsums_vvl(vrsum67_y0x0, VLEN) ;
    __vr vrsum7_y0x0 = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsum67_y0x0,32, VLEN), VLEN) ;
    __vr vrsum8_y0x0 = _vel_vfsums_vvl(vrsum89_y0x0, VLEN) ;
    __vr vrsum9_y0x0 = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsum89_y0x0,32, VLEN), VLEN) ;
    __vr vrsumA_y0x0 = _vel_vfsums_vvl(vrsumAB_y0x0, VLEN) ;
    __vr vrsumB_y0x0 = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsumAB_y0x0,32, VLEN), VLEN) ;
    __vr vrsumC_y0x0 = _vel_vfsums_vvl(vrsumCD_y0x0, VLEN) ;
    __vr vrsumD_y0x0 = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsumCD_y0x0,32, VLEN), VLEN) ;
    __vr vrsumE_y0x0 = _vel_vfsums_vvl(vrsumEF_y0x0, VLEN) ;
    __vr vrsumF_y0x0 = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsumEF_y0x0,32, VLEN), VLEN) ;

    _vel_vstu_vssl(vrsum0_y0x0, 4, &pOut[outIndex+0*outHeight*outWidth], 1) ;
    _vel_vstu_vssl(vrsum1_y0x0, 4, &pOut[outIndex+1*outHeight*outWidth], 1) ;
    _vel_vstu_vssl(vrsum2_y0x0, 4, &pOut[outIndex+2*outHeight*outWidth], 1) ;
    _vel_vstu_vssl(vrsum3_y0x0, 4, &pOut[outIndex+3*outHeight*outWidth], 1) ;
    _vel_vstu_vssl(vrsum4_y0x0, 4, &pOut[outIndex+4*outHeight*outWidth], 1) ;
    _vel_vstu_vssl(vrsum5_y0x0, 4, &pOut[outIndex+5*outHeight*outWidth], 1) ;
    _vel_vstu_vssl(vrsum6_y0x0, 4, &pOut[outIndex+6*outHeight*outWidth], 1) ;
    _vel_vstu_vssl(vrsum7_y0x0, 4, &pOut[outIndex+7*outHeight*outWidth], 1) ;
    _vel_vstu_vssl(vrsum8_y0x0, 4, &pOut[outIndex+8*outHeight*outWidth], 1) ;
    _vel_vstu_vssl(vrsum9_y0x0, 4, &pOut[outIndex+9*outHeight*outWidth], 1) ;
    _vel_vstu_vssl(vrsumA_y0x0, 4, &pOut[outIndex+10*outHeight*outWidth], 1) ;
    _vel_vstu_vssl(vrsumB_y0x0, 4, &pOut[outIndex+11*outHeight*outWidth], 1) ;
    _vel_vstu_vssl(vrsumC_y0x0, 4, &pOut[outIndex+12*outHeight*outWidth], 1) ;
    _vel_vstu_vssl(vrsumD_y0x0, 4, &pOut[outIndex+13*outHeight*outWidth], 1) ;
    _vel_vstu_vssl(vrsumE_y0x0, 4, &pOut[outIndex+14*outHeight*outWidth], 1) ;
    _vel_vstu_vssl(vrsumF_y0x0, 4, &pOut[outIndex+15*outHeight*outWidth], 1) ;
  }
  {
    __vr vrsum0_y0x1 = _vel_vfsums_vvl(vrsum01_y0x1, VLEN) ;
    __vr vrsum1_y0x1 = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsum01_y0x1,32, VLEN), VLEN) ;
    __vr vrsum2_y0x1 = _vel_vfsums_vvl(vrsum23_y0x1, VLEN) ;
    __vr vrsum3_y0x1 = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsum23_y0x1,32, VLEN), VLEN) ;
    __vr vrsum4_y0x1 = _vel_vfsums_vvl(vrsum45_y0x1, VLEN) ;
    __vr vrsum5_y0x1 = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsum45_y0x1,32, VLEN), VLEN) ;
    __vr vrsum6_y0x1 = _vel_vfsums_vvl(vrsum67_y0x1, VLEN) ;
    __vr vrsum7_y0x1 = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsum67_y0x1,32, VLEN), VLEN) ;
    __vr vrsum8_y0x1 = _vel_vfsums_vvl(vrsum89_y0x1, VLEN) ;
    __vr vrsum9_y0x1 = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsum89_y0x1,32, VLEN), VLEN) ;
    __vr vrsumA_y0x1 = _vel_vfsums_vvl(vrsumAB_y0x1, VLEN) ;
    __vr vrsumB_y0x1 = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsumAB_y0x1,32, VLEN), VLEN) ;
    __vr vrsumC_y0x1 = _vel_vfsums_vvl(vrsumCD_y0x1, VLEN) ;
    __vr vrsumD_y0x1 = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsumCD_y0x1,32, VLEN), VLEN) ;
    __vr vrsumE_y0x1 = _vel_vfsums_vvl(vrsumEF_y0x1, VLEN) ;
    __vr vrsumF_y0x1 = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsumEF_y0x1,32, VLEN), VLEN) ;

    _vel_vstu_vssl(vrsum0_y0x1, 4, &pOut[outIndex+0*outHeight*outWidth+1], 1) ;
    _vel_vstu_vssl(vrsum1_y0x1, 4, &pOut[outIndex+1*outHeight*outWidth+1], 1) ;
    _vel_vstu_vssl(vrsum2_y0x1, 4, &pOut[outIndex+2*outHeight*outWidth+1], 1) ;
    _vel_vstu_vssl(vrsum3_y0x1, 4, &pOut[outIndex+3*outHeight*outWidth+1], 1) ;
    _vel_vstu_vssl(vrsum4_y0x1, 4, &pOut[outIndex+4*outHeight*outWidth+1], 1) ;
    _vel_vstu_vssl(vrsum5_y0x1, 4, &pOut[outIndex+5*outHeight*outWidth+1], 1) ;
    _vel_vstu_vssl(vrsum6_y0x1, 4, &pOut[outIndex+6*outHeight*outWidth+1], 1) ;
    _vel_vstu_vssl(vrsum7_y0x1, 4, &pOut[outIndex+7*outHeight*outWidth+1], 1) ;
    _vel_vstu_vssl(vrsum8_y0x1, 4, &pOut[outIndex+8*outHeight*outWidth+1], 1) ;
    _vel_vstu_vssl(vrsum9_y0x1, 4, &pOut[outIndex+9*outHeight*outWidth+1], 1) ;
    _vel_vstu_vssl(vrsumA_y0x1, 4, &pOut[outIndex+10*outHeight*outWidth+1], 1) ;
    _vel_vstu_vssl(vrsumB_y0x1, 4, &pOut[outIndex+11*outHeight*outWidth+1], 1) ;
    _vel_vstu_vssl(vrsumC_y0x1, 4, &pOut[outIndex+12*outHeight*outWidth+1], 1) ;
    _vel_vstu_vssl(vrsumD_y0x1, 4, &pOut[outIndex+13*outHeight*outWidth+1], 1) ;
    _vel_vstu_vssl(vrsumE_y0x1, 4, &pOut[outIndex+14*outHeight*outWidth+1], 1) ;
    _vel_vstu_vssl(vrsumF_y0x1, 4, &pOut[outIndex+15*outHeight*outWidth+1], 1) ;
  }
}

template<filterLayout_t FLAYOUT>
static inline void k16y2x1(
    const float * __restrict__ pIn,
    const float * __restrict__ pKernel,
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
    const int64_t kernGroupOffset,
    const int64_t n,
    const int64_t k,
    const int64_t y,
    const int64_t x
)
{

  int64_t outIndex  = outGroupOffset + ((n * outChannel + k) * outHeight + y) * outWidth + x;

  __vr vrsum01_y0x0 = _vel_vbrdl_vsl(0UL, VLEN) ;
  __vr vrsum01_y1x0 = _vel_vbrdl_vsl(0UL, VLEN) ;

  __vr vrsum23_y0x0 = _vel_vbrdl_vsl(0UL, VLEN) ;
  __vr vrsum23_y1x0 = _vel_vbrdl_vsl(0UL, VLEN) ;

  __vr vrsum45_y0x0 = _vel_vbrdl_vsl(0UL, VLEN) ;
  __vr vrsum45_y1x0 = _vel_vbrdl_vsl(0UL, VLEN) ;

  __vr vrsum67_y0x0 = _vel_vbrdl_vsl(0UL, VLEN) ;
  __vr vrsum67_y1x0 = _vel_vbrdl_vsl(0UL, VLEN) ;

  __vr vrsum89_y0x0 = _vel_vbrdl_vsl(0UL, VLEN) ;
  __vr vrsum89_y1x0 = _vel_vbrdl_vsl(0UL, VLEN) ;

  __vr vrsumAB_y0x0 = _vel_vbrdl_vsl(0UL, VLEN) ;
  __vr vrsumAB_y1x0 = _vel_vbrdl_vsl(0UL, VLEN) ;

  __vr vrsumCD_y0x0 = _vel_vbrdl_vsl(0UL, VLEN) ;
  __vr vrsumCD_y1x0 = _vel_vbrdl_vsl(0UL, VLEN) ;

  __vr vrsumEF_y0x0 = _vel_vbrdl_vsl(0UL, VLEN) ;
  __vr vrsumEF_y1x0 = _vel_vbrdl_vsl(0UL, VLEN) ;

  for (int64_t c=0; c<inChannelGroup; c+= VLEN) {
    const int64_t vl = inChannelGroup - c < VLEN ? inChannelGroup - c : VLEN ;

    for (int64_t r=0; r<kernHeight; r++) {
      int64_t h0 = (y+0) * strideHeight - padHeight + r * dilationHeight;
      int64_t h1 = (y+1) * strideHeight - padHeight + r * dilationHeight;

      int64_t h0_valid = ( h0 >= 0 && h0 < inHeight ) ;
      int64_t h1_valid = ( h1 >= 0 && h1 < inHeight ) ;

      if( h0_valid && h1_valid ) {
	for (int64_t s=0; s<kernWidth; s++) {
	  int64_t w0 = (x+0) * strideWidth - padWidth + s * dilationWidth;

	  int64_t w0_valid = ( w0 >= 0 && w0 < inWidth ) ;

	  if( w0_valid ) {
	    int64_t inputIndex_h0w0  = inGroupOffset + ((n * inChannel + c) * inHeight + h0) * inWidth + w0;
	    int64_t inputIndex_h1w0  = inGroupOffset + ((n * inChannel + c) * inHeight + h1) * inWidth + w0;

	    int64_t kernelIndex = kernGroupOffset + ((k * inChannelGroup + c) * kernHeight + r) * kernWidth + s;

	    __vr vri_h0w0 = _vel_vldu_vssl(4*inHeight*inWidth, &pIn[inputIndex_h0w0], vl) ;
	    __vr vri_h1w0 = _vel_vldu_vssl(4*inHeight*inWidth, &pIn[inputIndex_h1w0], vl) ;

	    __vr vrk0 = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+0*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrk1 = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+1*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrk2 = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+2*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrk3 = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+3*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrk4 = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+4*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrk5 = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+5*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrk6 = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+6*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrk7 = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+7*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrk8 = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+8*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrk9 = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+9*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrkA = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+10*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrkB = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+11*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrkC = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+12*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrkD = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+13*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrkE = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+14*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrkF = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+15*inChannelGroup*kernHeight*kernWidth], vl) ;

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
	} // kernWidth
      }
      else if( h0_valid ) {
	for (int64_t s=0; s<kernWidth; s++) {
	  int64_t w0 = (x+0) * strideWidth - padWidth + s * dilationWidth;

	  int64_t w0_valid = ( w0 >= 0 && w0 < inWidth ) ;

	  if( w0_valid ) {
	    int64_t inputIndex_h0w0  = inGroupOffset + ((n * inChannel + c) * inHeight + h0) * inWidth + w0;

	    int64_t kernelIndex = kernGroupOffset + ((k * inChannelGroup + c) * kernHeight + r) * kernWidth + s;

	    __vr vri_h0w0 = _vel_vldu_vssl(4*inHeight*inWidth, &pIn[inputIndex_h0w0], vl) ;

	    __vr vrk0 = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+0*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrk1 = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+1*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrk2 = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+2*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrk3 = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+3*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrk4 = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+4*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrk5 = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+5*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrk6 = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+6*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrk7 = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+7*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrk8 = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+8*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrk9 = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+9*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrkA = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+10*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrkB = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+11*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrkC = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+12*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrkD = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+13*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrkE = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+14*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrkF = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+15*inChannelGroup*kernHeight*kernWidth], vl) ;

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
	} // kernWidth
      }
      else if( h1_valid ) {
	for (int64_t s=0; s<kernWidth; s++) {
	  int64_t w0 = (x+0) * strideWidth - padWidth + s * dilationWidth;

	  int64_t w0_valid = ( w0 >= 0 && w0 < inWidth ) ;

	  if( w0_valid ) {
	    int64_t inputIndex_h1w0  = inGroupOffset + ((n * inChannel + c) * inHeight + h1) * inWidth + w0;

	    int64_t kernelIndex = kernGroupOffset + ((k * inChannelGroup + c) * kernHeight + r) * kernWidth + s;

	    __vr vri_h1w0 = _vel_vldu_vssl(4*inHeight*inWidth, &pIn[inputIndex_h1w0], vl) ;

	    __vr vrk0 = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+0*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrk1 = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+1*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrk2 = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+2*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrk3 = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+3*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrk4 = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+4*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrk5 = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+5*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrk6 = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+6*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrk7 = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+7*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrk8 = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+8*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrk9 = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+9*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrkA = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+10*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrkB = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+11*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrkC = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+12*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrkD = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+13*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrkE = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+14*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrkF = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+15*inChannelGroup*kernHeight*kernWidth], vl) ;

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
	} // kernWidth
      }
    } // kernHeight
  } // inChannel

  {
    __vr vrsum0_y0x0 = _vel_vfsums_vvl(vrsum01_y0x0, VLEN) ;
    __vr vrsum1_y0x0 = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsum01_y0x0,32, VLEN), VLEN) ;
    __vr vrsum2_y0x0 = _vel_vfsums_vvl(vrsum23_y0x0, VLEN) ;
    __vr vrsum3_y0x0 = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsum23_y0x0,32, VLEN), VLEN) ;
    __vr vrsum4_y0x0 = _vel_vfsums_vvl(vrsum45_y0x0, VLEN) ;
    __vr vrsum5_y0x0 = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsum45_y0x0,32, VLEN), VLEN) ;
    __vr vrsum6_y0x0 = _vel_vfsums_vvl(vrsum67_y0x0, VLEN) ;
    __vr vrsum7_y0x0 = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsum67_y0x0,32, VLEN), VLEN) ;
    __vr vrsum8_y0x0 = _vel_vfsums_vvl(vrsum89_y0x0, VLEN) ;
    __vr vrsum9_y0x0 = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsum89_y0x0,32, VLEN), VLEN) ;
    __vr vrsumA_y0x0 = _vel_vfsums_vvl(vrsumAB_y0x0, VLEN) ;
    __vr vrsumB_y0x0 = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsumAB_y0x0,32, VLEN), VLEN) ;
    __vr vrsumC_y0x0 = _vel_vfsums_vvl(vrsumCD_y0x0, VLEN) ;
    __vr vrsumD_y0x0 = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsumCD_y0x0,32, VLEN), VLEN) ;
    __vr vrsumE_y0x0 = _vel_vfsums_vvl(vrsumEF_y0x0, VLEN) ;
    __vr vrsumF_y0x0 = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsumEF_y0x0,32, VLEN), VLEN) ;

    _vel_vstu_vssl(vrsum0_y0x0, 4, &pOut[outIndex+0*outHeight*outWidth], 1) ;
    _vel_vstu_vssl(vrsum1_y0x0, 4, &pOut[outIndex+1*outHeight*outWidth], 1) ;
    _vel_vstu_vssl(vrsum2_y0x0, 4, &pOut[outIndex+2*outHeight*outWidth], 1) ;
    _vel_vstu_vssl(vrsum3_y0x0, 4, &pOut[outIndex+3*outHeight*outWidth], 1) ;
    _vel_vstu_vssl(vrsum4_y0x0, 4, &pOut[outIndex+4*outHeight*outWidth], 1) ;
    _vel_vstu_vssl(vrsum5_y0x0, 4, &pOut[outIndex+5*outHeight*outWidth], 1) ;
    _vel_vstu_vssl(vrsum6_y0x0, 4, &pOut[outIndex+6*outHeight*outWidth], 1) ;
    _vel_vstu_vssl(vrsum7_y0x0, 4, &pOut[outIndex+7*outHeight*outWidth], 1) ;
    _vel_vstu_vssl(vrsum8_y0x0, 4, &pOut[outIndex+8*outHeight*outWidth], 1) ;
    _vel_vstu_vssl(vrsum9_y0x0, 4, &pOut[outIndex+9*outHeight*outWidth], 1) ;
    _vel_vstu_vssl(vrsumA_y0x0, 4, &pOut[outIndex+10*outHeight*outWidth], 1) ;
    _vel_vstu_vssl(vrsumB_y0x0, 4, &pOut[outIndex+11*outHeight*outWidth], 1) ;
    _vel_vstu_vssl(vrsumC_y0x0, 4, &pOut[outIndex+12*outHeight*outWidth], 1) ;
    _vel_vstu_vssl(vrsumD_y0x0, 4, &pOut[outIndex+13*outHeight*outWidth], 1) ;
    _vel_vstu_vssl(vrsumE_y0x0, 4, &pOut[outIndex+14*outHeight*outWidth], 1) ;
    _vel_vstu_vssl(vrsumF_y0x0, 4, &pOut[outIndex+15*outHeight*outWidth], 1) ;
  }
  {
    __vr vrsum0_y1x0 = _vel_vfsums_vvl(vrsum01_y1x0, VLEN) ;
    __vr vrsum1_y1x0 = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsum01_y1x0,32, VLEN), VLEN) ;
    __vr vrsum2_y1x0 = _vel_vfsums_vvl(vrsum23_y1x0, VLEN) ;
    __vr vrsum3_y1x0 = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsum23_y1x0,32, VLEN), VLEN) ;
    __vr vrsum4_y1x0 = _vel_vfsums_vvl(vrsum45_y1x0, VLEN) ;
    __vr vrsum5_y1x0 = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsum45_y1x0,32, VLEN), VLEN) ;
    __vr vrsum6_y1x0 = _vel_vfsums_vvl(vrsum67_y1x0, VLEN) ;
    __vr vrsum7_y1x0 = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsum67_y1x0,32, VLEN), VLEN) ;
    __vr vrsum8_y1x0 = _vel_vfsums_vvl(vrsum89_y1x0, VLEN) ;
    __vr vrsum9_y1x0 = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsum89_y1x0,32, VLEN), VLEN) ;
    __vr vrsumA_y1x0 = _vel_vfsums_vvl(vrsumAB_y1x0, VLEN) ;
    __vr vrsumB_y1x0 = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsumAB_y1x0,32, VLEN), VLEN) ;
    __vr vrsumC_y1x0 = _vel_vfsums_vvl(vrsumCD_y1x0, VLEN) ;
    __vr vrsumD_y1x0 = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsumCD_y1x0,32, VLEN), VLEN) ;
    __vr vrsumE_y1x0 = _vel_vfsums_vvl(vrsumEF_y1x0, VLEN) ;
    __vr vrsumF_y1x0 = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsumEF_y1x0,32, VLEN), VLEN) ;

    _vel_vstu_vssl(vrsum0_y1x0, 4, &pOut[outIndex+0*outHeight*outWidth+outWidth], 1) ;
    _vel_vstu_vssl(vrsum1_y1x0, 4, &pOut[outIndex+1*outHeight*outWidth+outWidth], 1) ;
    _vel_vstu_vssl(vrsum2_y1x0, 4, &pOut[outIndex+2*outHeight*outWidth+outWidth], 1) ;
    _vel_vstu_vssl(vrsum3_y1x0, 4, &pOut[outIndex+3*outHeight*outWidth+outWidth], 1) ;
    _vel_vstu_vssl(vrsum4_y1x0, 4, &pOut[outIndex+4*outHeight*outWidth+outWidth], 1) ;
    _vel_vstu_vssl(vrsum5_y1x0, 4, &pOut[outIndex+5*outHeight*outWidth+outWidth], 1) ;
    _vel_vstu_vssl(vrsum6_y1x0, 4, &pOut[outIndex+6*outHeight*outWidth+outWidth], 1) ;
    _vel_vstu_vssl(vrsum7_y1x0, 4, &pOut[outIndex+7*outHeight*outWidth+outWidth], 1) ;
    _vel_vstu_vssl(vrsum8_y1x0, 4, &pOut[outIndex+8*outHeight*outWidth+outWidth], 1) ;
    _vel_vstu_vssl(vrsum9_y1x0, 4, &pOut[outIndex+9*outHeight*outWidth+outWidth], 1) ;
    _vel_vstu_vssl(vrsumA_y1x0, 4, &pOut[outIndex+10*outHeight*outWidth+outWidth], 1) ;
    _vel_vstu_vssl(vrsumB_y1x0, 4, &pOut[outIndex+11*outHeight*outWidth+outWidth], 1) ;
    _vel_vstu_vssl(vrsumC_y1x0, 4, &pOut[outIndex+12*outHeight*outWidth+outWidth], 1) ;
    _vel_vstu_vssl(vrsumD_y1x0, 4, &pOut[outIndex+13*outHeight*outWidth+outWidth], 1) ;
    _vel_vstu_vssl(vrsumE_y1x0, 4, &pOut[outIndex+14*outHeight*outWidth+outWidth], 1) ;
    _vel_vstu_vssl(vrsumF_y1x0, 4, &pOut[outIndex+15*outHeight*outWidth+outWidth], 1) ;
  }
}

template<filterLayout_t FLAYOUT>
static inline void k16y2x2(
    const float * __restrict__ pIn,
    const float * __restrict__ pKernel,
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
    const int64_t kernGroupOffset,
    const int64_t n,
    const int64_t k,
    const int64_t y,
    const int64_t x
)
{

  int64_t outIndex  = outGroupOffset + ((n * outChannel + k) * outHeight + y) * outWidth + x;

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

      int64_t h0_valid = ( h0 >= 0 && h0 < inHeight ) ;
      int64_t h1_valid = ( h1 >= 0 && h1 < inHeight ) ;

      if( h0_valid && h1_valid ) {
	for (int64_t s=0; s<kernWidth; s++) {
	  int64_t w0 = (x+0) * strideWidth - padWidth + s * dilationWidth;
	  int64_t w1 = (x+1) * strideWidth - padWidth + s * dilationWidth;

	  int64_t w0_valid = ( w0 >= 0 && w0 < inWidth ) ;
	  int64_t w1_valid = ( w1 >= 0 && w1 < inWidth ) ;

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

	    __vr vrk0 = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+0*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrk1 = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+1*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrk2 = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+2*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrk3 = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+3*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrk4 = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+4*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrk5 = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+5*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrk6 = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+6*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrk7 = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+7*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrk8 = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+8*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrk9 = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+9*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrkA = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+10*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrkB = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+11*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrkC = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+12*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrkD = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+13*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrkE = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+14*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrkF = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+15*inChannelGroup*kernHeight*kernWidth], vl) ;

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

	    __vr vrk0 = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+0*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrk1 = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+1*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrk2 = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+2*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrk3 = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+3*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrk4 = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+4*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrk5 = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+5*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrk6 = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+6*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrk7 = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+7*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrk8 = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+8*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrk9 = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+9*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrkA = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+10*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrkB = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+11*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrkC = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+12*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrkD = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+13*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrkE = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+14*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrkF = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+15*inChannelGroup*kernHeight*kernWidth], vl) ;

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

	    __vr vrk0 = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+0*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrk1 = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+1*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrk2 = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+2*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrk3 = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+3*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrk4 = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+4*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrk5 = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+5*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrk6 = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+6*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrk7 = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+7*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrk8 = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+8*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrk9 = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+9*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrkA = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+10*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrkB = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+11*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrkC = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+12*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrkD = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+13*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrkE = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+14*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrkF = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+15*inChannelGroup*kernHeight*kernWidth], vl) ;

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

	    __vr vrk0 = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+0*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrk1 = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+1*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrk2 = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+2*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrk3 = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+3*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrk4 = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+4*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrk5 = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+5*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrk6 = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+6*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrk7 = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+7*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrk8 = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+8*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrk9 = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+9*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrkA = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+10*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrkB = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+11*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrkC = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+12*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrkD = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+13*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrkE = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+14*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrkF = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+15*inChannelGroup*kernHeight*kernWidth], vl) ;

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

	    __vr vrk0 = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+0*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrk1 = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+1*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrk2 = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+2*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrk3 = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+3*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrk4 = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+4*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrk5 = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+5*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrk6 = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+6*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrk7 = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+7*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrk8 = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+8*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrk9 = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+9*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrkA = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+10*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrkB = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+11*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrkC = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+12*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrkD = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+13*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrkE = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+14*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrkF = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+15*inChannelGroup*kernHeight*kernWidth], vl) ;

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

	    __vr vrk0 = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+0*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrk1 = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+1*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrk2 = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+2*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrk3 = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+3*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrk4 = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+4*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrk5 = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+5*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrk6 = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+6*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrk7 = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+7*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrk8 = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+8*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrk9 = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+9*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrkA = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+10*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrkB = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+11*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrkC = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+12*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrkD = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+13*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrkE = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+14*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrkF = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+15*inChannelGroup*kernHeight*kernWidth], vl) ;

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

	    __vr vrk0 = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+0*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrk1 = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+1*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrk2 = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+2*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrk3 = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+3*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrk4 = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+4*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrk5 = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+5*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrk6 = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+6*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrk7 = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+7*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrk8 = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+8*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrk9 = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+9*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrkA = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+10*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrkB = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+11*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrkC = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+12*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrkD = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+13*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrkE = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+14*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrkF = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+15*inChannelGroup*kernHeight*kernWidth], vl) ;

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

	    __vr vrk0 = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+0*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrk1 = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+1*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrk2 = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+2*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrk3 = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+3*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrk4 = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+4*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrk5 = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+5*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrk6 = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+6*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrk7 = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+7*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrk8 = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+8*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrk9 = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+9*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrkA = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+10*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrkB = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+11*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrkC = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+12*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrkD = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+13*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrkE = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+14*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrkF = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+15*inChannelGroup*kernHeight*kernWidth], vl) ;

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

	    __vr vrk0 = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+0*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrk1 = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+1*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrk2 = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+2*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrk3 = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+3*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrk4 = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+4*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrk5 = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+5*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrk6 = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+6*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrk7 = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+7*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrk8 = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+8*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrk9 = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+9*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrkA = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+10*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrkB = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+11*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrkC = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+12*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrkD = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+13*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrkE = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+14*inChannelGroup*kernHeight*kernWidth], vl) ;
	    __vr vrkF = _vel_vldu_vssl(4*kernHeight*kernWidth, &pKernel[kernelIndex+15*inChannelGroup*kernHeight*kernWidth], vl) ;

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

  {
    __vr vrsum0_y0x0 = _vel_vfsums_vvl(vrsum01_y0x0, VLEN) ;
    __vr vrsum1_y0x0 = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsum01_y0x0,32, VLEN), VLEN) ;
    __vr vrsum2_y0x0 = _vel_vfsums_vvl(vrsum23_y0x0, VLEN) ;
    __vr vrsum3_y0x0 = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsum23_y0x0,32, VLEN), VLEN) ;
    __vr vrsum4_y0x0 = _vel_vfsums_vvl(vrsum45_y0x0, VLEN) ;
    __vr vrsum5_y0x0 = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsum45_y0x0,32, VLEN), VLEN) ;
    __vr vrsum6_y0x0 = _vel_vfsums_vvl(vrsum67_y0x0, VLEN) ;
    __vr vrsum7_y0x0 = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsum67_y0x0,32, VLEN), VLEN) ;
    __vr vrsum8_y0x0 = _vel_vfsums_vvl(vrsum89_y0x0, VLEN) ;
    __vr vrsum9_y0x0 = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsum89_y0x0,32, VLEN), VLEN) ;
    __vr vrsumA_y0x0 = _vel_vfsums_vvl(vrsumAB_y0x0, VLEN) ;
    __vr vrsumB_y0x0 = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsumAB_y0x0,32, VLEN), VLEN) ;
    __vr vrsumC_y0x0 = _vel_vfsums_vvl(vrsumCD_y0x0, VLEN) ;
    __vr vrsumD_y0x0 = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsumCD_y0x0,32, VLEN), VLEN) ;
    __vr vrsumE_y0x0 = _vel_vfsums_vvl(vrsumEF_y0x0, VLEN) ;
    __vr vrsumF_y0x0 = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsumEF_y0x0,32, VLEN), VLEN) ;

    _vel_vstu_vssl(vrsum0_y0x0, 4, &pOut[outIndex+0*outHeight*outWidth], 1) ;
    _vel_vstu_vssl(vrsum1_y0x0, 4, &pOut[outIndex+1*outHeight*outWidth], 1) ;
    _vel_vstu_vssl(vrsum2_y0x0, 4, &pOut[outIndex+2*outHeight*outWidth], 1) ;
    _vel_vstu_vssl(vrsum3_y0x0, 4, &pOut[outIndex+3*outHeight*outWidth], 1) ;
    _vel_vstu_vssl(vrsum4_y0x0, 4, &pOut[outIndex+4*outHeight*outWidth], 1) ;
    _vel_vstu_vssl(vrsum5_y0x0, 4, &pOut[outIndex+5*outHeight*outWidth], 1) ;
    _vel_vstu_vssl(vrsum6_y0x0, 4, &pOut[outIndex+6*outHeight*outWidth], 1) ;
    _vel_vstu_vssl(vrsum7_y0x0, 4, &pOut[outIndex+7*outHeight*outWidth], 1) ;
    _vel_vstu_vssl(vrsum8_y0x0, 4, &pOut[outIndex+8*outHeight*outWidth], 1) ;
    _vel_vstu_vssl(vrsum9_y0x0, 4, &pOut[outIndex+9*outHeight*outWidth], 1) ;
    _vel_vstu_vssl(vrsumA_y0x0, 4, &pOut[outIndex+10*outHeight*outWidth], 1) ;
    _vel_vstu_vssl(vrsumB_y0x0, 4, &pOut[outIndex+11*outHeight*outWidth], 1) ;
    _vel_vstu_vssl(vrsumC_y0x0, 4, &pOut[outIndex+12*outHeight*outWidth], 1) ;
    _vel_vstu_vssl(vrsumD_y0x0, 4, &pOut[outIndex+13*outHeight*outWidth], 1) ;
    _vel_vstu_vssl(vrsumE_y0x0, 4, &pOut[outIndex+14*outHeight*outWidth], 1) ;
    _vel_vstu_vssl(vrsumF_y0x0, 4, &pOut[outIndex+15*outHeight*outWidth], 1) ;
  }
  {
    __vr vrsum0_y0x1 = _vel_vfsums_vvl(vrsum01_y0x1, VLEN) ;
    __vr vrsum1_y0x1 = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsum01_y0x1,32, VLEN), VLEN) ;
    __vr vrsum2_y0x1 = _vel_vfsums_vvl(vrsum23_y0x1, VLEN) ;
    __vr vrsum3_y0x1 = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsum23_y0x1,32, VLEN), VLEN) ;
    __vr vrsum4_y0x1 = _vel_vfsums_vvl(vrsum45_y0x1, VLEN) ;
    __vr vrsum5_y0x1 = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsum45_y0x1,32, VLEN), VLEN) ;
    __vr vrsum6_y0x1 = _vel_vfsums_vvl(vrsum67_y0x1, VLEN) ;
    __vr vrsum7_y0x1 = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsum67_y0x1,32, VLEN), VLEN) ;
    __vr vrsum8_y0x1 = _vel_vfsums_vvl(vrsum89_y0x1, VLEN) ;
    __vr vrsum9_y0x1 = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsum89_y0x1,32, VLEN), VLEN) ;
    __vr vrsumA_y0x1 = _vel_vfsums_vvl(vrsumAB_y0x1, VLEN) ;
    __vr vrsumB_y0x1 = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsumAB_y0x1,32, VLEN), VLEN) ;
    __vr vrsumC_y0x1 = _vel_vfsums_vvl(vrsumCD_y0x1, VLEN) ;
    __vr vrsumD_y0x1 = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsumCD_y0x1,32, VLEN), VLEN) ;
    __vr vrsumE_y0x1 = _vel_vfsums_vvl(vrsumEF_y0x1, VLEN) ;
    __vr vrsumF_y0x1 = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsumEF_y0x1,32, VLEN), VLEN) ;

    _vel_vstu_vssl(vrsum0_y0x1, 4, &pOut[outIndex+0*outHeight*outWidth+1], 1) ;
    _vel_vstu_vssl(vrsum1_y0x1, 4, &pOut[outIndex+1*outHeight*outWidth+1], 1) ;
    _vel_vstu_vssl(vrsum2_y0x1, 4, &pOut[outIndex+2*outHeight*outWidth+1], 1) ;
    _vel_vstu_vssl(vrsum3_y0x1, 4, &pOut[outIndex+3*outHeight*outWidth+1], 1) ;
    _vel_vstu_vssl(vrsum4_y0x1, 4, &pOut[outIndex+4*outHeight*outWidth+1], 1) ;
    _vel_vstu_vssl(vrsum5_y0x1, 4, &pOut[outIndex+5*outHeight*outWidth+1], 1) ;
    _vel_vstu_vssl(vrsum6_y0x1, 4, &pOut[outIndex+6*outHeight*outWidth+1], 1) ;
    _vel_vstu_vssl(vrsum7_y0x1, 4, &pOut[outIndex+7*outHeight*outWidth+1], 1) ;
    _vel_vstu_vssl(vrsum8_y0x1, 4, &pOut[outIndex+8*outHeight*outWidth+1], 1) ;
    _vel_vstu_vssl(vrsum9_y0x1, 4, &pOut[outIndex+9*outHeight*outWidth+1], 1) ;
    _vel_vstu_vssl(vrsumA_y0x1, 4, &pOut[outIndex+10*outHeight*outWidth+1], 1) ;
    _vel_vstu_vssl(vrsumB_y0x1, 4, &pOut[outIndex+11*outHeight*outWidth+1], 1) ;
    _vel_vstu_vssl(vrsumC_y0x1, 4, &pOut[outIndex+12*outHeight*outWidth+1], 1) ;
    _vel_vstu_vssl(vrsumD_y0x1, 4, &pOut[outIndex+13*outHeight*outWidth+1], 1) ;
    _vel_vstu_vssl(vrsumE_y0x1, 4, &pOut[outIndex+14*outHeight*outWidth+1], 1) ;
    _vel_vstu_vssl(vrsumF_y0x1, 4, &pOut[outIndex+15*outHeight*outWidth+1], 1) ;
  }
  {
    __vr vrsum0_y1x0 = _vel_vfsums_vvl(vrsum01_y1x0, VLEN) ;
    __vr vrsum1_y1x0 = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsum01_y1x0,32, VLEN), VLEN) ;
    __vr vrsum2_y1x0 = _vel_vfsums_vvl(vrsum23_y1x0, VLEN) ;
    __vr vrsum3_y1x0 = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsum23_y1x0,32, VLEN), VLEN) ;
    __vr vrsum4_y1x0 = _vel_vfsums_vvl(vrsum45_y1x0, VLEN) ;
    __vr vrsum5_y1x0 = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsum45_y1x0,32, VLEN), VLEN) ;
    __vr vrsum6_y1x0 = _vel_vfsums_vvl(vrsum67_y1x0, VLEN) ;
    __vr vrsum7_y1x0 = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsum67_y1x0,32, VLEN), VLEN) ;
    __vr vrsum8_y1x0 = _vel_vfsums_vvl(vrsum89_y1x0, VLEN) ;
    __vr vrsum9_y1x0 = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsum89_y1x0,32, VLEN), VLEN) ;
    __vr vrsumA_y1x0 = _vel_vfsums_vvl(vrsumAB_y1x0, VLEN) ;
    __vr vrsumB_y1x0 = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsumAB_y1x0,32, VLEN), VLEN) ;
    __vr vrsumC_y1x0 = _vel_vfsums_vvl(vrsumCD_y1x0, VLEN) ;
    __vr vrsumD_y1x0 = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsumCD_y1x0,32, VLEN), VLEN) ;
    __vr vrsumE_y1x0 = _vel_vfsums_vvl(vrsumEF_y1x0, VLEN) ;
    __vr vrsumF_y1x0 = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsumEF_y1x0,32, VLEN), VLEN) ;

    _vel_vstu_vssl(vrsum0_y1x0, 4, &pOut[outIndex+0*outHeight*outWidth+outWidth], 1) ;
    _vel_vstu_vssl(vrsum1_y1x0, 4, &pOut[outIndex+1*outHeight*outWidth+outWidth], 1) ;
    _vel_vstu_vssl(vrsum2_y1x0, 4, &pOut[outIndex+2*outHeight*outWidth+outWidth], 1) ;
    _vel_vstu_vssl(vrsum3_y1x0, 4, &pOut[outIndex+3*outHeight*outWidth+outWidth], 1) ;
    _vel_vstu_vssl(vrsum4_y1x0, 4, &pOut[outIndex+4*outHeight*outWidth+outWidth], 1) ;
    _vel_vstu_vssl(vrsum5_y1x0, 4, &pOut[outIndex+5*outHeight*outWidth+outWidth], 1) ;
    _vel_vstu_vssl(vrsum6_y1x0, 4, &pOut[outIndex+6*outHeight*outWidth+outWidth], 1) ;
    _vel_vstu_vssl(vrsum7_y1x0, 4, &pOut[outIndex+7*outHeight*outWidth+outWidth], 1) ;
    _vel_vstu_vssl(vrsum8_y1x0, 4, &pOut[outIndex+8*outHeight*outWidth+outWidth], 1) ;
    _vel_vstu_vssl(vrsum9_y1x0, 4, &pOut[outIndex+9*outHeight*outWidth+outWidth], 1) ;
    _vel_vstu_vssl(vrsumA_y1x0, 4, &pOut[outIndex+10*outHeight*outWidth+outWidth], 1) ;
    _vel_vstu_vssl(vrsumB_y1x0, 4, &pOut[outIndex+11*outHeight*outWidth+outWidth], 1) ;
    _vel_vstu_vssl(vrsumC_y1x0, 4, &pOut[outIndex+12*outHeight*outWidth+outWidth], 1) ;
    _vel_vstu_vssl(vrsumD_y1x0, 4, &pOut[outIndex+13*outHeight*outWidth+outWidth], 1) ;
    _vel_vstu_vssl(vrsumE_y1x0, 4, &pOut[outIndex+14*outHeight*outWidth+outWidth], 1) ;
    _vel_vstu_vssl(vrsumF_y1x0, 4, &pOut[outIndex+15*outHeight*outWidth+outWidth], 1) ;
  }
  {
    __vr vrsum0_y1x1 = _vel_vfsums_vvl(vrsum01_y1x1, VLEN) ;
    __vr vrsum1_y1x1 = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsum01_y1x1,32, VLEN), VLEN) ;
    __vr vrsum2_y1x1 = _vel_vfsums_vvl(vrsum23_y1x1, VLEN) ;
    __vr vrsum3_y1x1 = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsum23_y1x1,32, VLEN), VLEN) ;
    __vr vrsum4_y1x1 = _vel_vfsums_vvl(vrsum45_y1x1, VLEN) ;
    __vr vrsum5_y1x1 = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsum45_y1x1,32, VLEN), VLEN) ;
    __vr vrsum6_y1x1 = _vel_vfsums_vvl(vrsum67_y1x1, VLEN) ;
    __vr vrsum7_y1x1 = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsum67_y1x1,32, VLEN), VLEN) ;
    __vr vrsum8_y1x1 = _vel_vfsums_vvl(vrsum89_y1x1, VLEN) ;
    __vr vrsum9_y1x1 = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsum89_y1x1,32, VLEN), VLEN) ;
    __vr vrsumA_y1x1 = _vel_vfsums_vvl(vrsumAB_y1x1, VLEN) ;
    __vr vrsumB_y1x1 = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsumAB_y1x1,32, VLEN), VLEN) ;
    __vr vrsumC_y1x1 = _vel_vfsums_vvl(vrsumCD_y1x1, VLEN) ;
    __vr vrsumD_y1x1 = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsumCD_y1x1,32, VLEN), VLEN) ;
    __vr vrsumE_y1x1 = _vel_vfsums_vvl(vrsumEF_y1x1, VLEN) ;
    __vr vrsumF_y1x1 = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsumEF_y1x1,32, VLEN), VLEN) ;

    _vel_vstu_vssl(vrsum0_y1x1, 4, &pOut[outIndex+0*outHeight*outWidth+outWidth+1], 1) ;
    _vel_vstu_vssl(vrsum1_y1x1, 4, &pOut[outIndex+1*outHeight*outWidth+outWidth+1], 1) ;
    _vel_vstu_vssl(vrsum2_y1x1, 4, &pOut[outIndex+2*outHeight*outWidth+outWidth+1], 1) ;
    _vel_vstu_vssl(vrsum3_y1x1, 4, &pOut[outIndex+3*outHeight*outWidth+outWidth+1], 1) ;
    _vel_vstu_vssl(vrsum4_y1x1, 4, &pOut[outIndex+4*outHeight*outWidth+outWidth+1], 1) ;
    _vel_vstu_vssl(vrsum5_y1x1, 4, &pOut[outIndex+5*outHeight*outWidth+outWidth+1], 1) ;
    _vel_vstu_vssl(vrsum6_y1x1, 4, &pOut[outIndex+6*outHeight*outWidth+outWidth+1], 1) ;
    _vel_vstu_vssl(vrsum7_y1x1, 4, &pOut[outIndex+7*outHeight*outWidth+outWidth+1], 1) ;
    _vel_vstu_vssl(vrsum8_y1x1, 4, &pOut[outIndex+8*outHeight*outWidth+outWidth+1], 1) ;
    _vel_vstu_vssl(vrsum9_y1x1, 4, &pOut[outIndex+9*outHeight*outWidth+outWidth+1], 1) ;
    _vel_vstu_vssl(vrsumA_y1x1, 4, &pOut[outIndex+10*outHeight*outWidth+outWidth+1], 1) ;
    _vel_vstu_vssl(vrsumB_y1x1, 4, &pOut[outIndex+11*outHeight*outWidth+outWidth+1], 1) ;
    _vel_vstu_vssl(vrsumC_y1x1, 4, &pOut[outIndex+12*outHeight*outWidth+outWidth+1], 1) ;
    _vel_vstu_vssl(vrsumD_y1x1, 4, &pOut[outIndex+13*outHeight*outWidth+outWidth+1], 1) ;
    _vel_vstu_vssl(vrsumE_y1x1, 4, &pOut[outIndex+14*outHeight*outWidth+outWidth+1], 1) ;
    _vel_vstu_vssl(vrsumF_y1x1, 4, &pOut[outIndex+15*outHeight*outWidth+outWidth+1], 1) ;
  }
}

template<filterLayout_t FLAYOUT>
static inline void convloop(
    const float * __restrict__ pIn,
    const float * __restrict__ pKernel,
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
  for (int64_t n=0; n<batch; n++) {
    for (int64_t g = 0; g < group; g++) {
	int64_t inGroupOffset   = g * inChannelGroup  * inHeight  * inWidth;
	int64_t outGroupOffset  = g * outChannelGroup * outHeight * outWidth;
	int64_t kernGroupOffset = g * outChannelGroup * inChannelGroup * kernHeight * kernWidth;

	int64_t k=0 ;
	if( (outChannelGroup & 0x01) == 1 ) {
	  k1<FLAYOUT>(pIn, pKernel, pOut,
	     inChannel, inWidth, inHeight,
	     outChannel, outWidth, outHeight,
	     kernWidth, kernHeight,
	     inChannelGroup, outChannelGroup,
	     strideHeight, strideWidth,
	     padHeight, padWidth,
	     dilationHeight, dilationWidth,
	     inGroupOffset, outGroupOffset, kernGroupOffset,
	     n, k) ;
	  k+=1 ;
	}
	if( ((outChannelGroup >> 1) & 0x01) == 1 ) {
	  k2<FLAYOUT>(pIn, pKernel, pOut,
	     inChannel, inWidth, inHeight,
	     outChannel, outWidth, outHeight,
	     kernWidth, kernHeight,
	     inChannelGroup, outChannelGroup,
	     strideHeight, strideWidth,
	     padHeight, padWidth,
	     dilationHeight, dilationWidth,
	     inGroupOffset, outGroupOffset, kernGroupOffset,
	     n, k) ;
	  k+=2 ;
	}
	if( ((outChannelGroup >> 2) & 0x01) == 1 ) {
	  k4<FLAYOUT>(pIn, pKernel, pOut,
	     inChannel, inWidth, inHeight,
	     outChannel, outWidth, outHeight,
	     kernWidth, kernHeight,
	     inChannelGroup, outChannelGroup,
	     strideHeight, strideWidth,
	     padHeight, padWidth,
	     dilationHeight, dilationWidth,
	     inGroupOffset, outGroupOffset, kernGroupOffset,
	     n, k) ;
	  k+=4 ;
	}
	if( ((outChannelGroup >> 3) & 0x01) == 1 ) {
	  int64_t y=0 ;
	  if( (outHeight & 0x01) == 1 ) {
	    int64_t x=0;
	    if( (outWidth & 0x01) == 1 ) {
	      k8y1x1<FLAYOUT>(pIn, pKernel, pOut,
		 inChannel, inWidth, inHeight,
		 outChannel, outWidth, outHeight,
		 kernWidth, kernHeight,
		 inChannelGroup, outChannelGroup,
		 strideHeight, strideWidth,
		 padHeight, padWidth,
		 dilationHeight, dilationWidth,
		 inGroupOffset, outGroupOffset, kernGroupOffset,
		 n, k, y, x) ;
	      x+=1 ;
	    }
	    for (; x<outWidth; ) {
	      k8y1x2<FLAYOUT>(pIn, pKernel, pOut,
		 inChannel, inWidth, inHeight,
		 outChannel, outWidth, outHeight,
		 kernWidth, kernHeight,
		 inChannelGroup, outChannelGroup,
		 strideHeight, strideWidth,
		 padHeight, padWidth,
		 dilationHeight, dilationWidth,
		 inGroupOffset, outGroupOffset, kernGroupOffset,
		 n, k, y, x) ;
	      x+=2 ;
	    } // outWidth
	    y+=1 ;
	  }
	  for (; y<outHeight; ) {
	    int64_t x=0;
	    if( (outWidth & 0x01) == 1 ) {
	      k8y2x1<FLAYOUT>(pIn, pKernel, pOut,
		 inChannel, inWidth, inHeight,
		 outChannel, outWidth, outHeight,
		 kernWidth, kernHeight,
		 inChannelGroup, outChannelGroup,
		 strideHeight, strideWidth,
		 padHeight, padWidth,
		 dilationHeight, dilationWidth,
		 inGroupOffset, outGroupOffset, kernGroupOffset,
		 n, k, y, x) ;
	      x+=1 ;
	    }
	    for (; x<outWidth; ) {
	      k8y2x2<FLAYOUT>(pIn, pKernel, pOut,
		 inChannel, inWidth, inHeight,
		 outChannel, outWidth, outHeight,
		 kernWidth, kernHeight,
		 inChannelGroup, outChannelGroup,
		 strideHeight, strideWidth,
		 padHeight, padWidth,
		 dilationHeight, dilationWidth,
		 inGroupOffset, outGroupOffset, kernGroupOffset,
		 n, k, y, x) ;
	      x+=2 ;
	    } // outWidth
	    y+=2 ;
	  } // outHeight
	  k+=8 ;
	}
	for (; k<outChannelGroup; ) {
	  int64_t y=0 ;
	  if( (outHeight & 0x01) == 1 ) {
	    int64_t x=0;
	    if( (outWidth & 0x01) == 1 ) {
	      k16y1x1<FLAYOUT>(pIn, pKernel, pOut,
		 inChannel, inWidth, inHeight,
		 outChannel, outWidth, outHeight,
		 kernWidth, kernHeight,
		 inChannelGroup, outChannelGroup,
		 strideHeight, strideWidth,
		 padHeight, padWidth,
		 dilationHeight, dilationWidth,
		 inGroupOffset, outGroupOffset, kernGroupOffset,
		 n, k, y, x) ;
	      x+=1 ;
	    }
	    for (; x<outWidth; ) {
	      k16y1x2<FLAYOUT>(pIn, pKernel, pOut,
		 inChannel, inWidth, inHeight,
		 outChannel, outWidth, outHeight,
		 kernWidth, kernHeight,
		 inChannelGroup, outChannelGroup,
		 strideHeight, strideWidth,
		 padHeight, padWidth,
		 dilationHeight, dilationWidth,
		 inGroupOffset, outGroupOffset, kernGroupOffset,
		 n, k, y, x) ;
	      x+=2 ;
	    } // outWidth
	    y+=1 ;
	  }
	  for (; y<outHeight; ) {
	    int64_t x=0;
	    if( (outWidth & 0x01) == 1 ) {
	      k16y2x1<FLAYOUT>(pIn, pKernel, pOut,
		 inChannel, inWidth, inHeight,
		 outChannel, outWidth, outHeight,
		 kernWidth, kernHeight,
		 inChannelGroup, outChannelGroup,
		 strideHeight, strideWidth,
		 padHeight, padWidth,
		 dilationHeight, dilationWidth,
		 inGroupOffset, outGroupOffset, kernGroupOffset,
		 n, k, y, x) ;
	      x+=1 ;
	    }
	    for (; x<outWidth; ) {
	      k16y2x2<FLAYOUT>(pIn, pKernel, pOut,
		 inChannel, inWidth, inHeight,
		 outChannel, outWidth, outHeight,
		 kernWidth, kernHeight,
		 inChannelGroup, outChannelGroup,
		 strideHeight, strideWidth,
		 padHeight, padWidth,
		 dilationHeight, dilationWidth,
		 inGroupOffset, outGroupOffset, kernGroupOffset,
		 n, k, y, x) ;
	      x+=2 ;
	    } // outWidth
	    y+=2 ;
	  } // outHeight
	  k+=16 ;
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
  float * const pOut    = (float * const) pDataOut;

  if( filter_layout == VEDNN_FILTER_LAYOUT_NCHW) {
    convloop<VEDNN_FILTER_LAYOUT_NCHW>(pIn, pKernel, pOut,
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
    convloop<VEDNN_FILTER_LAYOUT_HWCN>(pIn, pKernel, pOut,
	         batch, group,
    		 inChannel, inWidth, inHeight,
    		 outChannel, outWidth, outHeight,
    		 kernWidth, kernHeight,
    		 inChannelGroup, outChannelGroup,
    		 strideHeight, strideWidth,
    		 padHeight, padWidth,
    		 dilationHeight, dilationWidth ) ;
  }


  return VEDNN_SUCCESS;
}
