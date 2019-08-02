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

      _ve_lvl(VLEN) ;
      __vr vrsum = _ve_vbrdu_vs_f32(0.f) ;

      for (int64_t c=0; c<inChannelGroup; c+= VLEN) {
	const int64_t vl = inChannelGroup - c < VLEN ? inChannelGroup - c : VLEN ;
	_ve_lvl(vl) ;

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
	    int64_t kernelIndex = kernGroupOffset + ((k * inChannelGroup + c) * kernHeight + r) * kernWidth + s;

	    __vr vri = _ve_vldu_vss(4*inHeight*inWidth, &pIn[inputIndex]) ;
	    __vr vrk = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex]) ;

	    vrsum = _ve_vfmads_vvvv(vrsum, vri, vrk) ;
	  } // kernWidth
	} // kernHeight
      } // inChannel

      _ve_lvl(VLEN) ;
      vrsum = _ve_vfsums_vv(vrsum) ;

      _ve_lvl(1) ;
      _ve_vstu_vss(vrsum, 4, &pOut[outIndex]) ;
    } // outWidth
  } // outHeight
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

      _ve_lvl(VLEN) ;
      __vr vrsum01 = _ve_vbrd_vs_i64(0UL) ;

      for (int64_t c=0; c<inChannelGroup; c+= VLEN) {
	const int64_t vl = inChannelGroup - c < VLEN ? inChannelGroup - c : VLEN ;
	_ve_lvl(vl) ;

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
	    int64_t kernelIndex = kernGroupOffset + ((k * inChannelGroup + c) * kernHeight + r) * kernWidth + s;

	    __vr vri = _ve_vldu_vss(4*inHeight*inWidth, &pIn[inputIndex]) ;

	    __vr vrk0 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+0*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk1 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+1*inChannelGroup*kernHeight*kernWidth]) ;

	    __vr vriP  = _ve_vshf_vvvs(vri, vri, VE_VSHUFFLE_YUZU) ;

	    __vr vrk01 = _ve_vshf_vvvs(vrk0, vrk1, VE_VSHUFFLE_YUZU) ;
	    vrsum01 = _ve_pvfmad_vvvv(vrsum01, vriP, vrk01) ;
	  } // kernWidth
	} // kernHeight
      } // inChannel

      _ve_lvl(VLEN) ;
      __vr vrsum0 = _ve_vfsums_vv(vrsum01) ;
      __vr vrsum1 = _ve_vfsums_vv(_ve_vsll_vvs(vrsum01,32)) ;

      _ve_lvl(1) ;
      _ve_vstu_vss(vrsum0, 4, &pOut[outIndex+0*outHeight*outWidth]) ;
      _ve_vstu_vss(vrsum1, 4, &pOut[outIndex+1*outHeight*outWidth]) ;
    } // outWidth
  } // outHeight
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

      _ve_lvl(VLEN) ;
      __vr vrsum01 = _ve_vbrd_vs_i64(0UL) ;
      __vr vrsum23 = _ve_vbrd_vs_i64(0UL) ;

      for (int64_t c=0; c<inChannelGroup; c+= VLEN) {
	const int64_t vl = inChannelGroup - c < VLEN ? inChannelGroup - c : VLEN ;
	_ve_lvl(vl) ;

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
	    int64_t kernelIndex = kernGroupOffset + ((k * inChannelGroup + c) * kernHeight + r) * kernWidth + s;

	    __vr vri = _ve_vldu_vss(4*inHeight*inWidth, &pIn[inputIndex]) ;

	    __vr vrk0 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+0*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk1 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+1*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk2 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+2*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk3 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+3*inChannelGroup*kernHeight*kernWidth]) ;

	    __vr vriP  = _ve_vshf_vvvs(vri, vri, VE_VSHUFFLE_YUZU) ;

	    __vr vrk01 = _ve_vshf_vvvs(vrk0, vrk1, VE_VSHUFFLE_YUZU) ;
	    vrsum01 = _ve_pvfmad_vvvv(vrsum01, vriP, vrk01) ;
	    __vr vrk23 = _ve_vshf_vvvs(vrk2, vrk3, VE_VSHUFFLE_YUZU) ;
	    vrsum23 = _ve_pvfmad_vvvv(vrsum23, vriP, vrk23) ;
	  } // kernWidth
	} // kernHeight
      } // inChannel

      _ve_lvl(VLEN) ;
      __vr vrsum0 = _ve_vfsums_vv(vrsum01) ;
      __vr vrsum1 = _ve_vfsums_vv(_ve_vsll_vvs(vrsum01,32)) ;
      __vr vrsum2 = _ve_vfsums_vv(vrsum23) ;
      __vr vrsum3 = _ve_vfsums_vv(_ve_vsll_vvs(vrsum23,32)) ;

      _ve_lvl(1) ;
      _ve_vstu_vss(vrsum0, 4, &pOut[outIndex+0*outHeight*outWidth]) ;
      _ve_vstu_vss(vrsum1, 4, &pOut[outIndex+1*outHeight*outWidth]) ;
      _ve_vstu_vss(vrsum2, 4, &pOut[outIndex+2*outHeight*outWidth]) ;
      _ve_vstu_vss(vrsum3, 4, &pOut[outIndex+3*outHeight*outWidth]) ;

    } // outWidth
  } // outHeight
}

static inline void k8y1x1(
    const float * restrict pIn,
    const float * restrict pKernel,
    float * restrict const pOut,
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

  _ve_lvl(VLEN) ;
  __vr vrsum01_y0x0 = _ve_vbrd_vs_i64(0UL) ;
  __vr vrsum23_y0x0 = _ve_vbrd_vs_i64(0UL) ;
  __vr vrsum45_y0x0 = _ve_vbrd_vs_i64(0UL) ;
  __vr vrsum67_y0x0 = _ve_vbrd_vs_i64(0UL) ;

  for (int64_t c=0; c<inChannelGroup; c+= VLEN) {
    const int64_t vl = inChannelGroup - c < VLEN ? inChannelGroup - c : VLEN ;
    _ve_lvl(vl) ;

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

	    __vr vri_h0w0 = _ve_vldu_vss(4*inHeight*inWidth, &pIn[inputIndex_h0w0]) ;

	    __vr vrk0 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+0*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk1 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+1*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk2 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+2*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk3 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+3*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk4 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+4*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk5 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+5*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk6 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+6*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk7 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+7*inChannelGroup*kernHeight*kernWidth]) ;

	    __vr vriP_h0w0  = _ve_vshf_vvvs(vri_h0w0, vri_h0w0, VE_VSHUFFLE_YUZU) ;

	    __vr vrk01 = _ve_vshf_vvvs(vrk0, vrk1, VE_VSHUFFLE_YUZU) ;
	    vrsum01_y0x0 = _ve_pvfmad_vvvv(vrsum01_y0x0, vriP_h0w0, vrk01) ;

	    __vr vrk23 = _ve_vshf_vvvs(vrk2, vrk3, VE_VSHUFFLE_YUZU) ;
	    vrsum23_y0x0 = _ve_pvfmad_vvvv(vrsum23_y0x0, vriP_h0w0, vrk23) ;

	    __vr vrk45 = _ve_vshf_vvvs(vrk4, vrk5, VE_VSHUFFLE_YUZU) ;
	    vrsum45_y0x0 = _ve_pvfmad_vvvv(vrsum45_y0x0, vriP_h0w0, vrk45) ;

	    __vr vrk67 = _ve_vshf_vvvs(vrk6, vrk7, VE_VSHUFFLE_YUZU) ;
	    vrsum67_y0x0 = _ve_pvfmad_vvvv(vrsum67_y0x0, vriP_h0w0, vrk67) ;
	  }
	} // kernWidth
      }
    } // kernHeight
  } // inChannel

  {
    _ve_lvl(VLEN) ;
    __vr vrsum0_y0x0 = _ve_vfsums_vv(vrsum01_y0x0) ;
    __vr vrsum1_y0x0 = _ve_vfsums_vv(_ve_vsll_vvs(vrsum01_y0x0,32)) ;
    __vr vrsum2_y0x0 = _ve_vfsums_vv(vrsum23_y0x0) ;
    __vr vrsum3_y0x0 = _ve_vfsums_vv(_ve_vsll_vvs(vrsum23_y0x0,32)) ;
    __vr vrsum4_y0x0 = _ve_vfsums_vv(vrsum45_y0x0) ;
    __vr vrsum5_y0x0 = _ve_vfsums_vv(_ve_vsll_vvs(vrsum45_y0x0,32)) ;
    __vr vrsum6_y0x0 = _ve_vfsums_vv(vrsum67_y0x0) ;
    __vr vrsum7_y0x0 = _ve_vfsums_vv(_ve_vsll_vvs(vrsum67_y0x0,32)) ;

    _ve_lvl(1) ;
    _ve_vstu_vss(vrsum0_y0x0, 4, &pOut[outIndex+0*outHeight*outWidth]) ;
    _ve_vstu_vss(vrsum1_y0x0, 4, &pOut[outIndex+1*outHeight*outWidth]) ;
    _ve_vstu_vss(vrsum2_y0x0, 4, &pOut[outIndex+2*outHeight*outWidth]) ;
    _ve_vstu_vss(vrsum3_y0x0, 4, &pOut[outIndex+3*outHeight*outWidth]) ;
    _ve_vstu_vss(vrsum4_y0x0, 4, &pOut[outIndex+4*outHeight*outWidth]) ;
    _ve_vstu_vss(vrsum5_y0x0, 4, &pOut[outIndex+5*outHeight*outWidth]) ;
    _ve_vstu_vss(vrsum6_y0x0, 4, &pOut[outIndex+6*outHeight*outWidth]) ;
    _ve_vstu_vss(vrsum7_y0x0, 4, &pOut[outIndex+7*outHeight*outWidth]) ;
  }
}

static inline void k8y1x2(
    const float * restrict pIn,
    const float * restrict pKernel,
    float * restrict const pOut,
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

  _ve_lvl(VLEN) ;
  __vr vrsum01_y0x0 = _ve_vbrd_vs_i64(0UL) ;
  __vr vrsum01_y0x1 = _ve_vbrd_vs_i64(0UL) ;

  __vr vrsum23_y0x0 = _ve_vbrd_vs_i64(0UL) ;
  __vr vrsum23_y0x1 = _ve_vbrd_vs_i64(0UL) ;

  __vr vrsum45_y0x0 = _ve_vbrd_vs_i64(0UL) ;
  __vr vrsum45_y0x1 = _ve_vbrd_vs_i64(0UL) ;

  __vr vrsum67_y0x0 = _ve_vbrd_vs_i64(0UL) ;
  __vr vrsum67_y0x1 = _ve_vbrd_vs_i64(0UL) ;

  for (int64_t c=0; c<inChannelGroup; c+= VLEN) {
    const int64_t vl = inChannelGroup - c < VLEN ? inChannelGroup - c : VLEN ;
    _ve_lvl(vl) ;

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

	    __vr vri_h0w0 = _ve_vldu_vss(4*inHeight*inWidth, &pIn[inputIndex_h0w0]) ;
	    __vr vri_h0w1 = _ve_vldu_vss(4*inHeight*inWidth, &pIn[inputIndex_h0w1]) ;

	    __vr vrk0 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+0*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk1 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+1*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk2 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+2*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk3 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+3*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk4 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+4*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk5 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+5*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk6 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+6*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk7 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+7*inChannelGroup*kernHeight*kernWidth]) ;

	    __vr vriP_h0w0  = _ve_vshf_vvvs(vri_h0w0, vri_h0w0, VE_VSHUFFLE_YUZU) ;
	    __vr vriP_h0w1  = _ve_vshf_vvvs(vri_h0w1, vri_h0w1, VE_VSHUFFLE_YUZU) ;

	    __vr vrk01 = _ve_vshf_vvvs(vrk0, vrk1, VE_VSHUFFLE_YUZU) ;
	    vrsum01_y0x0 = _ve_pvfmad_vvvv(vrsum01_y0x0, vriP_h0w0, vrk01) ;
	    vrsum01_y0x1 = _ve_pvfmad_vvvv(vrsum01_y0x1, vriP_h0w1, vrk01) ;

	    __vr vrk23 = _ve_vshf_vvvs(vrk2, vrk3, VE_VSHUFFLE_YUZU) ;
	    vrsum23_y0x0 = _ve_pvfmad_vvvv(vrsum23_y0x0, vriP_h0w0, vrk23) ;
	    vrsum23_y0x1 = _ve_pvfmad_vvvv(vrsum23_y0x1, vriP_h0w1, vrk23) ;

	    __vr vrk45 = _ve_vshf_vvvs(vrk4, vrk5, VE_VSHUFFLE_YUZU) ;
	    vrsum45_y0x0 = _ve_pvfmad_vvvv(vrsum45_y0x0, vriP_h0w0, vrk45) ;
	    vrsum45_y0x1 = _ve_pvfmad_vvvv(vrsum45_y0x1, vriP_h0w1, vrk45) ;

	    __vr vrk67 = _ve_vshf_vvvs(vrk6, vrk7, VE_VSHUFFLE_YUZU) ;
	    vrsum67_y0x0 = _ve_pvfmad_vvvv(vrsum67_y0x0, vriP_h0w0, vrk67) ;
	    vrsum67_y0x1 = _ve_pvfmad_vvvv(vrsum67_y0x1, vriP_h0w1, vrk67) ;
	  }
	  else if( w0_valid ) {
	    int64_t inputIndex_h0w0  = inGroupOffset + ((n * inChannel + c) * inHeight + h0) * inWidth + w0;

	    int64_t kernelIndex = kernGroupOffset + ((k * inChannelGroup + c) * kernHeight + r) * kernWidth + s;

	    __vr vri_h0w0 = _ve_vldu_vss(4*inHeight*inWidth, &pIn[inputIndex_h0w0]) ;

	    __vr vrk0 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+0*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk1 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+1*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk2 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+2*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk3 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+3*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk4 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+4*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk5 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+5*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk6 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+6*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk7 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+7*inChannelGroup*kernHeight*kernWidth]) ;

	    __vr vriP_h0w0  = _ve_vshf_vvvs(vri_h0w0, vri_h0w0, VE_VSHUFFLE_YUZU) ;

	    __vr vrk01 = _ve_vshf_vvvs(vrk0, vrk1, VE_VSHUFFLE_YUZU) ;
	    vrsum01_y0x0 = _ve_pvfmad_vvvv(vrsum01_y0x0, vriP_h0w0, vrk01) ;

	    __vr vrk23 = _ve_vshf_vvvs(vrk2, vrk3, VE_VSHUFFLE_YUZU) ;
	    vrsum23_y0x0 = _ve_pvfmad_vvvv(vrsum23_y0x0, vriP_h0w0, vrk23) ;

	    __vr vrk45 = _ve_vshf_vvvs(vrk4, vrk5, VE_VSHUFFLE_YUZU) ;
	    vrsum45_y0x0 = _ve_pvfmad_vvvv(vrsum45_y0x0, vriP_h0w0, vrk45) ;

	    __vr vrk67 = _ve_vshf_vvvs(vrk6, vrk7, VE_VSHUFFLE_YUZU) ;
	    vrsum67_y0x0 = _ve_pvfmad_vvvv(vrsum67_y0x0, vriP_h0w0, vrk67) ;
	  }
	  else if( w1_valid ) {
	    int64_t inputIndex_h0w1  = inGroupOffset + ((n * inChannel + c) * inHeight + h0) * inWidth + w1;

	    int64_t kernelIndex = kernGroupOffset + ((k * inChannelGroup + c) * kernHeight + r) * kernWidth + s;

	    __vr vri_h0w1 = _ve_vldu_vss(4*inHeight*inWidth, &pIn[inputIndex_h0w1]) ;

	    __vr vrk0 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+0*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk1 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+1*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk2 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+2*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk3 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+3*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk4 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+4*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk5 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+5*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk6 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+6*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk7 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+7*inChannelGroup*kernHeight*kernWidth]) ;

	    __vr vriP_h0w1  = _ve_vshf_vvvs(vri_h0w1, vri_h0w1, VE_VSHUFFLE_YUZU) ;

	    __vr vrk01 = _ve_vshf_vvvs(vrk0, vrk1, VE_VSHUFFLE_YUZU) ;
	    vrsum01_y0x1 = _ve_pvfmad_vvvv(vrsum01_y0x1, vriP_h0w1, vrk01) ;

	    __vr vrk23 = _ve_vshf_vvvs(vrk2, vrk3, VE_VSHUFFLE_YUZU) ;
	    vrsum23_y0x1 = _ve_pvfmad_vvvv(vrsum23_y0x1, vriP_h0w1, vrk23) ;

	    __vr vrk45 = _ve_vshf_vvvs(vrk4, vrk5, VE_VSHUFFLE_YUZU) ;
	    vrsum45_y0x1 = _ve_pvfmad_vvvv(vrsum45_y0x1, vriP_h0w1, vrk45) ;

	    __vr vrk67 = _ve_vshf_vvvs(vrk6, vrk7, VE_VSHUFFLE_YUZU) ;
	    vrsum67_y0x1 = _ve_pvfmad_vvvv(vrsum67_y0x1, vriP_h0w1, vrk67) ;
	  }
	} // kernWidth
      }
    } // kernHeight
  } // inChannel

  {
    _ve_lvl(VLEN) ;
    __vr vrsum0_y0x0 = _ve_vfsums_vv(vrsum01_y0x0) ;
    __vr vrsum1_y0x0 = _ve_vfsums_vv(_ve_vsll_vvs(vrsum01_y0x0,32)) ;
    __vr vrsum2_y0x0 = _ve_vfsums_vv(vrsum23_y0x0) ;
    __vr vrsum3_y0x0 = _ve_vfsums_vv(_ve_vsll_vvs(vrsum23_y0x0,32)) ;
    __vr vrsum4_y0x0 = _ve_vfsums_vv(vrsum45_y0x0) ;
    __vr vrsum5_y0x0 = _ve_vfsums_vv(_ve_vsll_vvs(vrsum45_y0x0,32)) ;
    __vr vrsum6_y0x0 = _ve_vfsums_vv(vrsum67_y0x0) ;
    __vr vrsum7_y0x0 = _ve_vfsums_vv(_ve_vsll_vvs(vrsum67_y0x0,32)) ;

    _ve_lvl(1) ;
    _ve_vstu_vss(vrsum0_y0x0, 4, &pOut[outIndex+0*outHeight*outWidth]) ;
    _ve_vstu_vss(vrsum1_y0x0, 4, &pOut[outIndex+1*outHeight*outWidth]) ;
    _ve_vstu_vss(vrsum2_y0x0, 4, &pOut[outIndex+2*outHeight*outWidth]) ;
    _ve_vstu_vss(vrsum3_y0x0, 4, &pOut[outIndex+3*outHeight*outWidth]) ;
    _ve_vstu_vss(vrsum4_y0x0, 4, &pOut[outIndex+4*outHeight*outWidth]) ;
    _ve_vstu_vss(vrsum5_y0x0, 4, &pOut[outIndex+5*outHeight*outWidth]) ;
    _ve_vstu_vss(vrsum6_y0x0, 4, &pOut[outIndex+6*outHeight*outWidth]) ;
    _ve_vstu_vss(vrsum7_y0x0, 4, &pOut[outIndex+7*outHeight*outWidth]) ;
  }
  {
    _ve_lvl(VLEN) ;
    __vr vrsum0_y0x1 = _ve_vfsums_vv(vrsum01_y0x1) ;
    __vr vrsum1_y0x1 = _ve_vfsums_vv(_ve_vsll_vvs(vrsum01_y0x1,32)) ;
    __vr vrsum2_y0x1 = _ve_vfsums_vv(vrsum23_y0x1) ;
    __vr vrsum3_y0x1 = _ve_vfsums_vv(_ve_vsll_vvs(vrsum23_y0x1,32)) ;
    __vr vrsum4_y0x1 = _ve_vfsums_vv(vrsum45_y0x1) ;
    __vr vrsum5_y0x1 = _ve_vfsums_vv(_ve_vsll_vvs(vrsum45_y0x1,32)) ;
    __vr vrsum6_y0x1 = _ve_vfsums_vv(vrsum67_y0x1) ;
    __vr vrsum7_y0x1 = _ve_vfsums_vv(_ve_vsll_vvs(vrsum67_y0x1,32)) ;

    _ve_lvl(1) ;
    _ve_vstu_vss(vrsum0_y0x1, 4, &pOut[outIndex+0*outHeight*outWidth+1]) ;
    _ve_vstu_vss(vrsum1_y0x1, 4, &pOut[outIndex+1*outHeight*outWidth+1]) ;
    _ve_vstu_vss(vrsum2_y0x1, 4, &pOut[outIndex+2*outHeight*outWidth+1]) ;
    _ve_vstu_vss(vrsum3_y0x1, 4, &pOut[outIndex+3*outHeight*outWidth+1]) ;
    _ve_vstu_vss(vrsum4_y0x1, 4, &pOut[outIndex+4*outHeight*outWidth+1]) ;
    _ve_vstu_vss(vrsum5_y0x1, 4, &pOut[outIndex+5*outHeight*outWidth+1]) ;
    _ve_vstu_vss(vrsum6_y0x1, 4, &pOut[outIndex+6*outHeight*outWidth+1]) ;
    _ve_vstu_vss(vrsum7_y0x1, 4, &pOut[outIndex+7*outHeight*outWidth+1]) ;
  }
}


static inline void k8y2x1(
    const float * restrict pIn,
    const float * restrict pKernel,
    float * restrict const pOut,
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

  _ve_lvl(VLEN) ;
  __vr vrsum01_y0x0 = _ve_vbrd_vs_i64(0UL) ;
  __vr vrsum01_y1x0 = _ve_vbrd_vs_i64(0UL) ;

  __vr vrsum23_y0x0 = _ve_vbrd_vs_i64(0UL) ;
  __vr vrsum23_y1x0 = _ve_vbrd_vs_i64(0UL) ;

  __vr vrsum45_y0x0 = _ve_vbrd_vs_i64(0UL) ;
  __vr vrsum45_y1x0 = _ve_vbrd_vs_i64(0UL) ;

  __vr vrsum67_y0x0 = _ve_vbrd_vs_i64(0UL) ;
  __vr vrsum67_y1x0 = _ve_vbrd_vs_i64(0UL) ;

  for (int64_t c=0; c<inChannelGroup; c+= VLEN) {
    const int64_t vl = inChannelGroup - c < VLEN ? inChannelGroup - c : VLEN ;
    _ve_lvl(vl) ;

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

	    __vr vri_h0w0 = _ve_vldu_vss(4*inHeight*inWidth, &pIn[inputIndex_h0w0]) ;
	    __vr vri_h1w0 = _ve_vldu_vss(4*inHeight*inWidth, &pIn[inputIndex_h1w0]) ;

	    __vr vrk0 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+0*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk1 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+1*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk2 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+2*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk3 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+3*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk4 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+4*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk5 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+5*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk6 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+6*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk7 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+7*inChannelGroup*kernHeight*kernWidth]) ;

	    __vr vriP_h0w0  = _ve_vshf_vvvs(vri_h0w0, vri_h0w0, VE_VSHUFFLE_YUZU) ;
	    __vr vriP_h1w0  = _ve_vshf_vvvs(vri_h1w0, vri_h1w0, VE_VSHUFFLE_YUZU) ;

	    __vr vrk01 = _ve_vshf_vvvs(vrk0, vrk1, VE_VSHUFFLE_YUZU) ;
	    vrsum01_y0x0 = _ve_pvfmad_vvvv(vrsum01_y0x0, vriP_h0w0, vrk01) ;
	    vrsum01_y1x0 = _ve_pvfmad_vvvv(vrsum01_y1x0, vriP_h1w0, vrk01) ;

	    __vr vrk23 = _ve_vshf_vvvs(vrk2, vrk3, VE_VSHUFFLE_YUZU) ;
	    vrsum23_y0x0 = _ve_pvfmad_vvvv(vrsum23_y0x0, vriP_h0w0, vrk23) ;
	    vrsum23_y1x0 = _ve_pvfmad_vvvv(vrsum23_y1x0, vriP_h1w0, vrk23) ;

	    __vr vrk45 = _ve_vshf_vvvs(vrk4, vrk5, VE_VSHUFFLE_YUZU) ;
	    vrsum45_y0x0 = _ve_pvfmad_vvvv(vrsum45_y0x0, vriP_h0w0, vrk45) ;
	    vrsum45_y1x0 = _ve_pvfmad_vvvv(vrsum45_y1x0, vriP_h1w0, vrk45) ;

	    __vr vrk67 = _ve_vshf_vvvs(vrk6, vrk7, VE_VSHUFFLE_YUZU) ;
	    vrsum67_y0x0 = _ve_pvfmad_vvvv(vrsum67_y0x0, vriP_h0w0, vrk67) ;
	    vrsum67_y1x0 = _ve_pvfmad_vvvv(vrsum67_y1x0, vriP_h1w0, vrk67) ;
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

	    __vr vri_h0w0 = _ve_vldu_vss(4*inHeight*inWidth, &pIn[inputIndex_h0w0]) ;

	    __vr vrk0 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+0*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk1 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+1*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk2 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+2*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk3 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+3*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk4 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+4*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk5 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+5*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk6 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+6*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk7 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+7*inChannelGroup*kernHeight*kernWidth]) ;

	    __vr vriP_h0w0  = _ve_vshf_vvvs(vri_h0w0, vri_h0w0, VE_VSHUFFLE_YUZU) ;

	    __vr vrk01 = _ve_vshf_vvvs(vrk0, vrk1, VE_VSHUFFLE_YUZU) ;
	    vrsum01_y0x0 = _ve_pvfmad_vvvv(vrsum01_y0x0, vriP_h0w0, vrk01) ;

	    __vr vrk23 = _ve_vshf_vvvs(vrk2, vrk3, VE_VSHUFFLE_YUZU) ;
	    vrsum23_y0x0 = _ve_pvfmad_vvvv(vrsum23_y0x0, vriP_h0w0, vrk23) ;

	    __vr vrk45 = _ve_vshf_vvvs(vrk4, vrk5, VE_VSHUFFLE_YUZU) ;
	    vrsum45_y0x0 = _ve_pvfmad_vvvv(vrsum45_y0x0, vriP_h0w0, vrk45) ;

	    __vr vrk67 = _ve_vshf_vvvs(vrk6, vrk7, VE_VSHUFFLE_YUZU) ;
	    vrsum67_y0x0 = _ve_pvfmad_vvvv(vrsum67_y0x0, vriP_h0w0, vrk67) ;
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

	    __vr vri_h1w0 = _ve_vldu_vss(4*inHeight*inWidth, &pIn[inputIndex_h1w0]) ;

	    __vr vrk0 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+0*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk1 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+1*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk2 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+2*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk3 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+3*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk4 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+4*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk5 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+5*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk6 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+6*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk7 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+7*inChannelGroup*kernHeight*kernWidth]) ;

	    __vr vriP_h1w0  = _ve_vshf_vvvs(vri_h1w0, vri_h1w0, VE_VSHUFFLE_YUZU) ;

	    __vr vrk01 = _ve_vshf_vvvs(vrk0, vrk1, VE_VSHUFFLE_YUZU) ;
	    vrsum01_y1x0 = _ve_pvfmad_vvvv(vrsum01_y1x0, vriP_h1w0, vrk01) ;

	    __vr vrk23 = _ve_vshf_vvvs(vrk2, vrk3, VE_VSHUFFLE_YUZU) ;
	    vrsum23_y1x0 = _ve_pvfmad_vvvv(vrsum23_y1x0, vriP_h1w0, vrk23) ;

	    __vr vrk45 = _ve_vshf_vvvs(vrk4, vrk5, VE_VSHUFFLE_YUZU) ;
	    vrsum45_y1x0 = _ve_pvfmad_vvvv(vrsum45_y1x0, vriP_h1w0, vrk45) ;

	    __vr vrk67 = _ve_vshf_vvvs(vrk6, vrk7, VE_VSHUFFLE_YUZU) ;
	    vrsum67_y1x0 = _ve_pvfmad_vvvv(vrsum67_y1x0, vriP_h1w0, vrk67) ;
	  }
	} // kernWidth
      }
    } // kernHeight
  } // inChannel

  {
    _ve_lvl(VLEN) ;
    __vr vrsum0_y0x0 = _ve_vfsums_vv(vrsum01_y0x0) ;
    __vr vrsum1_y0x0 = _ve_vfsums_vv(_ve_vsll_vvs(vrsum01_y0x0,32)) ;
    __vr vrsum2_y0x0 = _ve_vfsums_vv(vrsum23_y0x0) ;
    __vr vrsum3_y0x0 = _ve_vfsums_vv(_ve_vsll_vvs(vrsum23_y0x0,32)) ;
    __vr vrsum4_y0x0 = _ve_vfsums_vv(vrsum45_y0x0) ;
    __vr vrsum5_y0x0 = _ve_vfsums_vv(_ve_vsll_vvs(vrsum45_y0x0,32)) ;
    __vr vrsum6_y0x0 = _ve_vfsums_vv(vrsum67_y0x0) ;
    __vr vrsum7_y0x0 = _ve_vfsums_vv(_ve_vsll_vvs(vrsum67_y0x0,32)) ;

    _ve_lvl(1) ;
    _ve_vstu_vss(vrsum0_y0x0, 4, &pOut[outIndex+0*outHeight*outWidth]) ;
    _ve_vstu_vss(vrsum1_y0x0, 4, &pOut[outIndex+1*outHeight*outWidth]) ;
    _ve_vstu_vss(vrsum2_y0x0, 4, &pOut[outIndex+2*outHeight*outWidth]) ;
    _ve_vstu_vss(vrsum3_y0x0, 4, &pOut[outIndex+3*outHeight*outWidth]) ;
    _ve_vstu_vss(vrsum4_y0x0, 4, &pOut[outIndex+4*outHeight*outWidth]) ;
    _ve_vstu_vss(vrsum5_y0x0, 4, &pOut[outIndex+5*outHeight*outWidth]) ;
    _ve_vstu_vss(vrsum6_y0x0, 4, &pOut[outIndex+6*outHeight*outWidth]) ;
    _ve_vstu_vss(vrsum7_y0x0, 4, &pOut[outIndex+7*outHeight*outWidth]) ;
  }
  {
    _ve_lvl(VLEN) ;
    __vr vrsum0_y1x0 = _ve_vfsums_vv(vrsum01_y1x0) ;
    __vr vrsum1_y1x0 = _ve_vfsums_vv(_ve_vsll_vvs(vrsum01_y1x0,32)) ;
    __vr vrsum2_y1x0 = _ve_vfsums_vv(vrsum23_y1x0) ;
    __vr vrsum3_y1x0 = _ve_vfsums_vv(_ve_vsll_vvs(vrsum23_y1x0,32)) ;
    __vr vrsum4_y1x0 = _ve_vfsums_vv(vrsum45_y1x0) ;
    __vr vrsum5_y1x0 = _ve_vfsums_vv(_ve_vsll_vvs(vrsum45_y1x0,32)) ;
    __vr vrsum6_y1x0 = _ve_vfsums_vv(vrsum67_y1x0) ;
    __vr vrsum7_y1x0 = _ve_vfsums_vv(_ve_vsll_vvs(vrsum67_y1x0,32)) ;

    _ve_lvl(1) ;
    _ve_vstu_vss(vrsum0_y1x0, 4, &pOut[outIndex+0*outHeight*outWidth+outWidth]) ;
    _ve_vstu_vss(vrsum1_y1x0, 4, &pOut[outIndex+1*outHeight*outWidth+outWidth]) ;
    _ve_vstu_vss(vrsum2_y1x0, 4, &pOut[outIndex+2*outHeight*outWidth+outWidth]) ;
    _ve_vstu_vss(vrsum3_y1x0, 4, &pOut[outIndex+3*outHeight*outWidth+outWidth]) ;
    _ve_vstu_vss(vrsum4_y1x0, 4, &pOut[outIndex+4*outHeight*outWidth+outWidth]) ;
    _ve_vstu_vss(vrsum5_y1x0, 4, &pOut[outIndex+5*outHeight*outWidth+outWidth]) ;
    _ve_vstu_vss(vrsum6_y1x0, 4, &pOut[outIndex+6*outHeight*outWidth+outWidth]) ;
    _ve_vstu_vss(vrsum7_y1x0, 4, &pOut[outIndex+7*outHeight*outWidth+outWidth]) ;
  }
}

static inline void k8y2x2(
    const float * restrict pIn,
    const float * restrict pKernel,
    float * restrict const pOut,
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

  _ve_lvl(VLEN) ;
  __vr vrsum01_y0x0 = _ve_vbrd_vs_i64(0UL) ;
  __vr vrsum01_y0x1 = _ve_vbrd_vs_i64(0UL) ;
  __vr vrsum01_y1x0 = _ve_vbrd_vs_i64(0UL) ;
  __vr vrsum01_y1x1 = _ve_vbrd_vs_i64(0UL) ;

  __vr vrsum23_y0x0 = _ve_vbrd_vs_i64(0UL) ;
  __vr vrsum23_y0x1 = _ve_vbrd_vs_i64(0UL) ;
  __vr vrsum23_y1x0 = _ve_vbrd_vs_i64(0UL) ;
  __vr vrsum23_y1x1 = _ve_vbrd_vs_i64(0UL) ;

  __vr vrsum45_y0x0 = _ve_vbrd_vs_i64(0UL) ;
  __vr vrsum45_y0x1 = _ve_vbrd_vs_i64(0UL) ;
  __vr vrsum45_y1x0 = _ve_vbrd_vs_i64(0UL) ;
  __vr vrsum45_y1x1 = _ve_vbrd_vs_i64(0UL) ;

  __vr vrsum67_y0x0 = _ve_vbrd_vs_i64(0UL) ;
  __vr vrsum67_y0x1 = _ve_vbrd_vs_i64(0UL) ;
  __vr vrsum67_y1x0 = _ve_vbrd_vs_i64(0UL) ;
  __vr vrsum67_y1x1 = _ve_vbrd_vs_i64(0UL) ;

  for (int64_t c=0; c<inChannelGroup; c+= VLEN) {
    const int64_t vl = inChannelGroup - c < VLEN ? inChannelGroup - c : VLEN ;
    _ve_lvl(vl) ;

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

	    __vr vri_h0w0 = _ve_vldu_vss(4*inHeight*inWidth, &pIn[inputIndex_h0w0]) ;
	    __vr vri_h0w1 = _ve_vldu_vss(4*inHeight*inWidth, &pIn[inputIndex_h0w1]) ;
	    __vr vri_h1w0 = _ve_vldu_vss(4*inHeight*inWidth, &pIn[inputIndex_h1w0]) ;
	    __vr vri_h1w1 = _ve_vldu_vss(4*inHeight*inWidth, &pIn[inputIndex_h1w1]) ;

	    __vr vrk0 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+0*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk1 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+1*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk2 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+2*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk3 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+3*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk4 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+4*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk5 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+5*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk6 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+6*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk7 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+7*inChannelGroup*kernHeight*kernWidth]) ;

	    __vr vriP_h0w0  = _ve_vshf_vvvs(vri_h0w0, vri_h0w0, VE_VSHUFFLE_YUZU) ;
	    __vr vriP_h0w1  = _ve_vshf_vvvs(vri_h0w1, vri_h0w1, VE_VSHUFFLE_YUZU) ;
	    __vr vriP_h1w0  = _ve_vshf_vvvs(vri_h1w0, vri_h1w0, VE_VSHUFFLE_YUZU) ;
	    __vr vriP_h1w1  = _ve_vshf_vvvs(vri_h1w1, vri_h1w1, VE_VSHUFFLE_YUZU) ;

	    __vr vrk01 = _ve_vshf_vvvs(vrk0, vrk1, VE_VSHUFFLE_YUZU) ;
	    vrsum01_y0x0 = _ve_pvfmad_vvvv(vrsum01_y0x0, vriP_h0w0, vrk01) ;
	    vrsum01_y0x1 = _ve_pvfmad_vvvv(vrsum01_y0x1, vriP_h0w1, vrk01) ;
	    vrsum01_y1x0 = _ve_pvfmad_vvvv(vrsum01_y1x0, vriP_h1w0, vrk01) ;
	    vrsum01_y1x1 = _ve_pvfmad_vvvv(vrsum01_y1x1, vriP_h1w1, vrk01) ;

	    __vr vrk23 = _ve_vshf_vvvs(vrk2, vrk3, VE_VSHUFFLE_YUZU) ;
	    vrsum23_y0x0 = _ve_pvfmad_vvvv(vrsum23_y0x0, vriP_h0w0, vrk23) ;
	    vrsum23_y0x1 = _ve_pvfmad_vvvv(vrsum23_y0x1, vriP_h0w1, vrk23) ;
	    vrsum23_y1x0 = _ve_pvfmad_vvvv(vrsum23_y1x0, vriP_h1w0, vrk23) ;
	    vrsum23_y1x1 = _ve_pvfmad_vvvv(vrsum23_y1x1, vriP_h1w1, vrk23) ;

	    __vr vrk45 = _ve_vshf_vvvs(vrk4, vrk5, VE_VSHUFFLE_YUZU) ;
	    vrsum45_y0x0 = _ve_pvfmad_vvvv(vrsum45_y0x0, vriP_h0w0, vrk45) ;
	    vrsum45_y0x1 = _ve_pvfmad_vvvv(vrsum45_y0x1, vriP_h0w1, vrk45) ;
	    vrsum45_y1x0 = _ve_pvfmad_vvvv(vrsum45_y1x0, vriP_h1w0, vrk45) ;
	    vrsum45_y1x1 = _ve_pvfmad_vvvv(vrsum45_y1x1, vriP_h1w1, vrk45) ;

	    __vr vrk67 = _ve_vshf_vvvs(vrk6, vrk7, VE_VSHUFFLE_YUZU) ;
	    vrsum67_y0x0 = _ve_pvfmad_vvvv(vrsum67_y0x0, vriP_h0w0, vrk67) ;
	    vrsum67_y0x1 = _ve_pvfmad_vvvv(vrsum67_y0x1, vriP_h0w1, vrk67) ;
	    vrsum67_y1x0 = _ve_pvfmad_vvvv(vrsum67_y1x0, vriP_h1w0, vrk67) ;
	    vrsum67_y1x1 = _ve_pvfmad_vvvv(vrsum67_y1x1, vriP_h1w1, vrk67) ;
	  }
	  else if( w0_valid ) {
	    int64_t inputIndex_h0w0  = inGroupOffset + ((n * inChannel + c) * inHeight + h0) * inWidth + w0;
	    int64_t inputIndex_h1w0  = inGroupOffset + ((n * inChannel + c) * inHeight + h1) * inWidth + w0;

	    int64_t kernelIndex = kernGroupOffset + ((k * inChannelGroup + c) * kernHeight + r) * kernWidth + s;

	    __vr vri_h0w0 = _ve_vldu_vss(4*inHeight*inWidth, &pIn[inputIndex_h0w0]) ;
	    __vr vri_h1w0 = _ve_vldu_vss(4*inHeight*inWidth, &pIn[inputIndex_h1w0]) ;

	    __vr vrk0 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+0*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk1 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+1*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk2 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+2*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk3 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+3*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk4 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+4*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk5 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+5*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk6 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+6*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk7 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+7*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk8 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+8*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk9 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+9*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrkA = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+10*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrkB = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+11*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrkC = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+12*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrkD = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+13*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrkE = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+14*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrkF = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+15*inChannelGroup*kernHeight*kernWidth]) ;

	    __vr vriP_h0w0  = _ve_vshf_vvvs(vri_h0w0, vri_h0w0, VE_VSHUFFLE_YUZU) ;
	    __vr vriP_h1w0  = _ve_vshf_vvvs(vri_h1w0, vri_h1w0, VE_VSHUFFLE_YUZU) ;

	    __vr vrk01 = _ve_vshf_vvvs(vrk0, vrk1, VE_VSHUFFLE_YUZU) ;
	    vrsum01_y0x0 = _ve_pvfmad_vvvv(vrsum01_y0x0, vriP_h0w0, vrk01) ;
	    vrsum01_y1x0 = _ve_pvfmad_vvvv(vrsum01_y1x0, vriP_h1w0, vrk01) ;

	    __vr vrk23 = _ve_vshf_vvvs(vrk2, vrk3, VE_VSHUFFLE_YUZU) ;
	    vrsum23_y0x0 = _ve_pvfmad_vvvv(vrsum23_y0x0, vriP_h0w0, vrk23) ;
	    vrsum23_y1x0 = _ve_pvfmad_vvvv(vrsum23_y1x0, vriP_h1w0, vrk23) ;

	    __vr vrk45 = _ve_vshf_vvvs(vrk4, vrk5, VE_VSHUFFLE_YUZU) ;
	    vrsum45_y0x0 = _ve_pvfmad_vvvv(vrsum45_y0x0, vriP_h0w0, vrk45) ;
	    vrsum45_y1x0 = _ve_pvfmad_vvvv(vrsum45_y1x0, vriP_h1w0, vrk45) ;

	    __vr vrk67 = _ve_vshf_vvvs(vrk6, vrk7, VE_VSHUFFLE_YUZU) ;
	    vrsum67_y0x0 = _ve_pvfmad_vvvv(vrsum67_y0x0, vriP_h0w0, vrk67) ;
	    vrsum67_y1x0 = _ve_pvfmad_vvvv(vrsum67_y1x0, vriP_h1w0, vrk67) ;
	  }
	  else if( w1_valid ) {
	    int64_t inputIndex_h0w1  = inGroupOffset + ((n * inChannel + c) * inHeight + h0) * inWidth + w1;
	    int64_t inputIndex_h1w1  = inGroupOffset + ((n * inChannel + c) * inHeight + h1) * inWidth + w1;

	    int64_t kernelIndex = kernGroupOffset + ((k * inChannelGroup + c) * kernHeight + r) * kernWidth + s;

	    __vr vri_h0w1 = _ve_vldu_vss(4*inHeight*inWidth, &pIn[inputIndex_h0w1]) ;
	    __vr vri_h1w1 = _ve_vldu_vss(4*inHeight*inWidth, &pIn[inputIndex_h1w1]) ;

	    __vr vrk0 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+0*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk1 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+1*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk2 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+2*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk3 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+3*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk4 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+4*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk5 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+5*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk6 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+6*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk7 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+7*inChannelGroup*kernHeight*kernWidth]) ;

	    __vr vriP_h0w1  = _ve_vshf_vvvs(vri_h0w1, vri_h0w1, VE_VSHUFFLE_YUZU) ;
	    __vr vriP_h1w1  = _ve_vshf_vvvs(vri_h1w1, vri_h1w1, VE_VSHUFFLE_YUZU) ;

	    __vr vrk01 = _ve_vshf_vvvs(vrk0, vrk1, VE_VSHUFFLE_YUZU) ;
	    vrsum01_y0x1 = _ve_pvfmad_vvvv(vrsum01_y0x1, vriP_h0w1, vrk01) ;
	    vrsum01_y1x1 = _ve_pvfmad_vvvv(vrsum01_y1x1, vriP_h1w1, vrk01) ;

	    __vr vrk23 = _ve_vshf_vvvs(vrk2, vrk3, VE_VSHUFFLE_YUZU) ;
	    vrsum23_y0x1 = _ve_pvfmad_vvvv(vrsum23_y0x1, vriP_h0w1, vrk23) ;
	    vrsum23_y1x1 = _ve_pvfmad_vvvv(vrsum23_y1x1, vriP_h1w1, vrk23) ;

	    __vr vrk45 = _ve_vshf_vvvs(vrk4, vrk5, VE_VSHUFFLE_YUZU) ;
	    vrsum45_y0x1 = _ve_pvfmad_vvvv(vrsum45_y0x1, vriP_h0w1, vrk45) ;
	    vrsum45_y1x1 = _ve_pvfmad_vvvv(vrsum45_y1x1, vriP_h1w1, vrk45) ;

	    __vr vrk67 = _ve_vshf_vvvs(vrk6, vrk7, VE_VSHUFFLE_YUZU) ;
	    vrsum67_y0x1 = _ve_pvfmad_vvvv(vrsum67_y0x1, vriP_h0w1, vrk67) ;
	    vrsum67_y1x1 = _ve_pvfmad_vvvv(vrsum67_y1x1, vriP_h1w1, vrk67) ;
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

	    __vr vri_h0w0 = _ve_vldu_vss(4*inHeight*inWidth, &pIn[inputIndex_h0w0]) ;
	    __vr vri_h0w1 = _ve_vldu_vss(4*inHeight*inWidth, &pIn[inputIndex_h0w1]) ;

	    __vr vrk0 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+0*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk1 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+1*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk2 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+2*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk3 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+3*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk4 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+4*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk5 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+5*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk6 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+6*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk7 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+7*inChannelGroup*kernHeight*kernWidth]) ;

	    __vr vriP_h0w0  = _ve_vshf_vvvs(vri_h0w0, vri_h0w0, VE_VSHUFFLE_YUZU) ;
	    __vr vriP_h0w1  = _ve_vshf_vvvs(vri_h0w1, vri_h0w1, VE_VSHUFFLE_YUZU) ;

	    __vr vrk01 = _ve_vshf_vvvs(vrk0, vrk1, VE_VSHUFFLE_YUZU) ;
	    vrsum01_y0x0 = _ve_pvfmad_vvvv(vrsum01_y0x0, vriP_h0w0, vrk01) ;
	    vrsum01_y0x1 = _ve_pvfmad_vvvv(vrsum01_y0x1, vriP_h0w1, vrk01) ;

	    __vr vrk23 = _ve_vshf_vvvs(vrk2, vrk3, VE_VSHUFFLE_YUZU) ;
	    vrsum23_y0x0 = _ve_pvfmad_vvvv(vrsum23_y0x0, vriP_h0w0, vrk23) ;
	    vrsum23_y0x1 = _ve_pvfmad_vvvv(vrsum23_y0x1, vriP_h0w1, vrk23) ;

	    __vr vrk45 = _ve_vshf_vvvs(vrk4, vrk5, VE_VSHUFFLE_YUZU) ;
	    vrsum45_y0x0 = _ve_pvfmad_vvvv(vrsum45_y0x0, vriP_h0w0, vrk45) ;
	    vrsum45_y0x1 = _ve_pvfmad_vvvv(vrsum45_y0x1, vriP_h0w1, vrk45) ;

	    __vr vrk67 = _ve_vshf_vvvs(vrk6, vrk7, VE_VSHUFFLE_YUZU) ;
	    vrsum67_y0x0 = _ve_pvfmad_vvvv(vrsum67_y0x0, vriP_h0w0, vrk67) ;
	    vrsum67_y0x1 = _ve_pvfmad_vvvv(vrsum67_y0x1, vriP_h0w1, vrk67) ;
	  }
	  else if( w0_valid ) {
	    int64_t inputIndex_h0w0  = inGroupOffset + ((n * inChannel + c) * inHeight + h0) * inWidth + w0;

	    int64_t kernelIndex = kernGroupOffset + ((k * inChannelGroup + c) * kernHeight + r) * kernWidth + s;

	    __vr vri_h0w0 = _ve_vldu_vss(4*inHeight*inWidth, &pIn[inputIndex_h0w0]) ;

	    __vr vrk0 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+0*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk1 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+1*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk2 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+2*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk3 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+3*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk4 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+4*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk5 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+5*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk6 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+6*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk7 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+7*inChannelGroup*kernHeight*kernWidth]) ;

	    __vr vriP_h0w0  = _ve_vshf_vvvs(vri_h0w0, vri_h0w0, VE_VSHUFFLE_YUZU) ;

	    __vr vrk01 = _ve_vshf_vvvs(vrk0, vrk1, VE_VSHUFFLE_YUZU) ;
	    vrsum01_y0x0 = _ve_pvfmad_vvvv(vrsum01_y0x0, vriP_h0w0, vrk01) ;

	    __vr vrk23 = _ve_vshf_vvvs(vrk2, vrk3, VE_VSHUFFLE_YUZU) ;
	    vrsum23_y0x0 = _ve_pvfmad_vvvv(vrsum23_y0x0, vriP_h0w0, vrk23) ;

	    __vr vrk45 = _ve_vshf_vvvs(vrk4, vrk5, VE_VSHUFFLE_YUZU) ;
	    vrsum45_y0x0 = _ve_pvfmad_vvvv(vrsum45_y0x0, vriP_h0w0, vrk45) ;

	    __vr vrk67 = _ve_vshf_vvvs(vrk6, vrk7, VE_VSHUFFLE_YUZU) ;
	    vrsum67_y0x0 = _ve_pvfmad_vvvv(vrsum67_y0x0, vriP_h0w0, vrk67) ;
	  }
	  else if( w1_valid ) {
	    int64_t inputIndex_h0w1  = inGroupOffset + ((n * inChannel + c) * inHeight + h0) * inWidth + w1;

	    int64_t kernelIndex = kernGroupOffset + ((k * inChannelGroup + c) * kernHeight + r) * kernWidth + s;

	    __vr vri_h0w1 = _ve_vldu_vss(4*inHeight*inWidth, &pIn[inputIndex_h0w1]) ;

	    __vr vrk0 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+0*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk1 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+1*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk2 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+2*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk3 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+3*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk4 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+4*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk5 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+5*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk6 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+6*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk7 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+7*inChannelGroup*kernHeight*kernWidth]) ;

	    __vr vriP_h0w1  = _ve_vshf_vvvs(vri_h0w1, vri_h0w1, VE_VSHUFFLE_YUZU) ;

	    __vr vrk01 = _ve_vshf_vvvs(vrk0, vrk1, VE_VSHUFFLE_YUZU) ;
	    vrsum01_y0x1 = _ve_pvfmad_vvvv(vrsum01_y0x1, vriP_h0w1, vrk01) ;

	    __vr vrk23 = _ve_vshf_vvvs(vrk2, vrk3, VE_VSHUFFLE_YUZU) ;
	    vrsum23_y0x1 = _ve_pvfmad_vvvv(vrsum23_y0x1, vriP_h0w1, vrk23) ;

	    __vr vrk45 = _ve_vshf_vvvs(vrk4, vrk5, VE_VSHUFFLE_YUZU) ;
	    vrsum45_y0x1 = _ve_pvfmad_vvvv(vrsum45_y0x1, vriP_h0w1, vrk45) ;

	    __vr vrk67 = _ve_vshf_vvvs(vrk6, vrk7, VE_VSHUFFLE_YUZU) ;
	    vrsum67_y0x1 = _ve_pvfmad_vvvv(vrsum67_y0x1, vriP_h0w1, vrk67) ;
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

	    __vr vri_h1w0 = _ve_vldu_vss(4*inHeight*inWidth, &pIn[inputIndex_h1w0]) ;
	    __vr vri_h1w1 = _ve_vldu_vss(4*inHeight*inWidth, &pIn[inputIndex_h1w1]) ;

	    __vr vrk0 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+0*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk1 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+1*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk2 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+2*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk3 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+3*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk4 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+4*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk5 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+5*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk6 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+6*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk7 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+7*inChannelGroup*kernHeight*kernWidth]) ;

	    __vr vriP_h1w0  = _ve_vshf_vvvs(vri_h1w0, vri_h1w0, VE_VSHUFFLE_YUZU) ;
	    __vr vriP_h1w1  = _ve_vshf_vvvs(vri_h1w1, vri_h1w1, VE_VSHUFFLE_YUZU) ;

	    __vr vrk01 = _ve_vshf_vvvs(vrk0, vrk1, VE_VSHUFFLE_YUZU) ;
	    vrsum01_y1x0 = _ve_pvfmad_vvvv(vrsum01_y1x0, vriP_h1w0, vrk01) ;
	    vrsum01_y1x1 = _ve_pvfmad_vvvv(vrsum01_y1x1, vriP_h1w1, vrk01) ;

	    __vr vrk23 = _ve_vshf_vvvs(vrk2, vrk3, VE_VSHUFFLE_YUZU) ;
	    vrsum23_y1x0 = _ve_pvfmad_vvvv(vrsum23_y1x0, vriP_h1w0, vrk23) ;
	    vrsum23_y1x1 = _ve_pvfmad_vvvv(vrsum23_y1x1, vriP_h1w1, vrk23) ;

	    __vr vrk45 = _ve_vshf_vvvs(vrk4, vrk5, VE_VSHUFFLE_YUZU) ;
	    vrsum45_y1x0 = _ve_pvfmad_vvvv(vrsum45_y1x0, vriP_h1w0, vrk45) ;
	    vrsum45_y1x1 = _ve_pvfmad_vvvv(vrsum45_y1x1, vriP_h1w1, vrk45) ;

	    __vr vrk67 = _ve_vshf_vvvs(vrk6, vrk7, VE_VSHUFFLE_YUZU) ;
	    vrsum67_y1x0 = _ve_pvfmad_vvvv(vrsum67_y1x0, vriP_h1w0, vrk67) ;
	    vrsum67_y1x1 = _ve_pvfmad_vvvv(vrsum67_y1x1, vriP_h1w1, vrk67) ;
	  }
	  else if( w0_valid ) {
	    int64_t inputIndex_h1w0  = inGroupOffset + ((n * inChannel + c) * inHeight + h1) * inWidth + w0;

	    int64_t kernelIndex = kernGroupOffset + ((k * inChannelGroup + c) * kernHeight + r) * kernWidth + s;

	    __vr vri_h1w0 = _ve_vldu_vss(4*inHeight*inWidth, &pIn[inputIndex_h1w0]) ;

	    __vr vrk0 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+0*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk1 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+1*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk2 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+2*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk3 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+3*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk4 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+4*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk5 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+5*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk6 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+6*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk7 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+7*inChannelGroup*kernHeight*kernWidth]) ;

	    __vr vriP_h1w0  = _ve_vshf_vvvs(vri_h1w0, vri_h1w0, VE_VSHUFFLE_YUZU) ;

	    __vr vrk01 = _ve_vshf_vvvs(vrk0, vrk1, VE_VSHUFFLE_YUZU) ;
	    vrsum01_y1x0 = _ve_pvfmad_vvvv(vrsum01_y1x0, vriP_h1w0, vrk01) ;

	    __vr vrk23 = _ve_vshf_vvvs(vrk2, vrk3, VE_VSHUFFLE_YUZU) ;
	    vrsum23_y1x0 = _ve_pvfmad_vvvv(vrsum23_y1x0, vriP_h1w0, vrk23) ;

	    __vr vrk45 = _ve_vshf_vvvs(vrk4, vrk5, VE_VSHUFFLE_YUZU) ;
	    vrsum45_y1x0 = _ve_pvfmad_vvvv(vrsum45_y1x0, vriP_h1w0, vrk45) ;

	    __vr vrk67 = _ve_vshf_vvvs(vrk6, vrk7, VE_VSHUFFLE_YUZU) ;
	    vrsum67_y1x0 = _ve_pvfmad_vvvv(vrsum67_y1x0, vriP_h1w0, vrk67) ;
	  }
	  else if( w1_valid ) {
	    int64_t inputIndex_h1w1  = inGroupOffset + ((n * inChannel + c) * inHeight + h1) * inWidth + w1;

	    int64_t kernelIndex = kernGroupOffset + ((k * inChannelGroup + c) * kernHeight + r) * kernWidth + s;

	    __vr vri_h1w1 = _ve_vldu_vss(4*inHeight*inWidth, &pIn[inputIndex_h1w1]) ;

	    __vr vrk0 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+0*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk1 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+1*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk2 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+2*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk3 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+3*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk4 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+4*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk5 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+5*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk6 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+6*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk7 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+7*inChannelGroup*kernHeight*kernWidth]) ;

	    __vr vriP_h1w1  = _ve_vshf_vvvs(vri_h1w1, vri_h1w1, VE_VSHUFFLE_YUZU) ;

	    __vr vrk01 = _ve_vshf_vvvs(vrk0, vrk1, VE_VSHUFFLE_YUZU) ;
	    vrsum01_y1x1 = _ve_pvfmad_vvvv(vrsum01_y1x1, vriP_h1w1, vrk01) ;

	    __vr vrk23 = _ve_vshf_vvvs(vrk2, vrk3, VE_VSHUFFLE_YUZU) ;
	    vrsum23_y1x1 = _ve_pvfmad_vvvv(vrsum23_y1x1, vriP_h1w1, vrk23) ;

	    __vr vrk45 = _ve_vshf_vvvs(vrk4, vrk5, VE_VSHUFFLE_YUZU) ;
	    vrsum45_y1x1 = _ve_pvfmad_vvvv(vrsum45_y1x1, vriP_h1w1, vrk45) ;

	    __vr vrk67 = _ve_vshf_vvvs(vrk6, vrk7, VE_VSHUFFLE_YUZU) ;
	    vrsum67_y1x1 = _ve_pvfmad_vvvv(vrsum67_y1x1, vriP_h1w1, vrk67) ;
	  }
	} // kernWidth
      }
    } // kernHeight
  } // inChannel

  {
    _ve_lvl(VLEN) ;
    __vr vrsum0_y0x0 = _ve_vfsums_vv(vrsum01_y0x0) ;
    __vr vrsum1_y0x0 = _ve_vfsums_vv(_ve_vsll_vvs(vrsum01_y0x0,32)) ;
    __vr vrsum2_y0x0 = _ve_vfsums_vv(vrsum23_y0x0) ;
    __vr vrsum3_y0x0 = _ve_vfsums_vv(_ve_vsll_vvs(vrsum23_y0x0,32)) ;
    __vr vrsum4_y0x0 = _ve_vfsums_vv(vrsum45_y0x0) ;
    __vr vrsum5_y0x0 = _ve_vfsums_vv(_ve_vsll_vvs(vrsum45_y0x0,32)) ;
    __vr vrsum6_y0x0 = _ve_vfsums_vv(vrsum67_y0x0) ;
    __vr vrsum7_y0x0 = _ve_vfsums_vv(_ve_vsll_vvs(vrsum67_y0x0,32)) ;

    _ve_lvl(1) ;
    _ve_vstu_vss(vrsum0_y0x0, 4, &pOut[outIndex+0*outHeight*outWidth]) ;
    _ve_vstu_vss(vrsum1_y0x0, 4, &pOut[outIndex+1*outHeight*outWidth]) ;
    _ve_vstu_vss(vrsum2_y0x0, 4, &pOut[outIndex+2*outHeight*outWidth]) ;
    _ve_vstu_vss(vrsum3_y0x0, 4, &pOut[outIndex+3*outHeight*outWidth]) ;
    _ve_vstu_vss(vrsum4_y0x0, 4, &pOut[outIndex+4*outHeight*outWidth]) ;
    _ve_vstu_vss(vrsum5_y0x0, 4, &pOut[outIndex+5*outHeight*outWidth]) ;
    _ve_vstu_vss(vrsum6_y0x0, 4, &pOut[outIndex+6*outHeight*outWidth]) ;
    _ve_vstu_vss(vrsum7_y0x0, 4, &pOut[outIndex+7*outHeight*outWidth]) ;
  }
  {
    _ve_lvl(VLEN) ;
    __vr vrsum0_y0x1 = _ve_vfsums_vv(vrsum01_y0x1) ;
    __vr vrsum1_y0x1 = _ve_vfsums_vv(_ve_vsll_vvs(vrsum01_y0x1,32)) ;
    __vr vrsum2_y0x1 = _ve_vfsums_vv(vrsum23_y0x1) ;
    __vr vrsum3_y0x1 = _ve_vfsums_vv(_ve_vsll_vvs(vrsum23_y0x1,32)) ;
    __vr vrsum4_y0x1 = _ve_vfsums_vv(vrsum45_y0x1) ;
    __vr vrsum5_y0x1 = _ve_vfsums_vv(_ve_vsll_vvs(vrsum45_y0x1,32)) ;
    __vr vrsum6_y0x1 = _ve_vfsums_vv(vrsum67_y0x1) ;
    __vr vrsum7_y0x1 = _ve_vfsums_vv(_ve_vsll_vvs(vrsum67_y0x1,32)) ;

    _ve_lvl(1) ;
    _ve_vstu_vss(vrsum0_y0x1, 4, &pOut[outIndex+0*outHeight*outWidth+1]) ;
    _ve_vstu_vss(vrsum1_y0x1, 4, &pOut[outIndex+1*outHeight*outWidth+1]) ;
    _ve_vstu_vss(vrsum2_y0x1, 4, &pOut[outIndex+2*outHeight*outWidth+1]) ;
    _ve_vstu_vss(vrsum3_y0x1, 4, &pOut[outIndex+3*outHeight*outWidth+1]) ;
    _ve_vstu_vss(vrsum4_y0x1, 4, &pOut[outIndex+4*outHeight*outWidth+1]) ;
    _ve_vstu_vss(vrsum5_y0x1, 4, &pOut[outIndex+5*outHeight*outWidth+1]) ;
    _ve_vstu_vss(vrsum6_y0x1, 4, &pOut[outIndex+6*outHeight*outWidth+1]) ;
    _ve_vstu_vss(vrsum7_y0x1, 4, &pOut[outIndex+7*outHeight*outWidth+1]) ;
  }
  {
    _ve_lvl(VLEN) ;
    __vr vrsum0_y1x0 = _ve_vfsums_vv(vrsum01_y1x0) ;
    __vr vrsum1_y1x0 = _ve_vfsums_vv(_ve_vsll_vvs(vrsum01_y1x0,32)) ;
    __vr vrsum2_y1x0 = _ve_vfsums_vv(vrsum23_y1x0) ;
    __vr vrsum3_y1x0 = _ve_vfsums_vv(_ve_vsll_vvs(vrsum23_y1x0,32)) ;
    __vr vrsum4_y1x0 = _ve_vfsums_vv(vrsum45_y1x0) ;
    __vr vrsum5_y1x0 = _ve_vfsums_vv(_ve_vsll_vvs(vrsum45_y1x0,32)) ;
    __vr vrsum6_y1x0 = _ve_vfsums_vv(vrsum67_y1x0) ;
    __vr vrsum7_y1x0 = _ve_vfsums_vv(_ve_vsll_vvs(vrsum67_y1x0,32)) ;

    _ve_lvl(1) ;
    _ve_vstu_vss(vrsum0_y1x0, 4, &pOut[outIndex+0*outHeight*outWidth+outWidth]) ;
    _ve_vstu_vss(vrsum1_y1x0, 4, &pOut[outIndex+1*outHeight*outWidth+outWidth]) ;
    _ve_vstu_vss(vrsum2_y1x0, 4, &pOut[outIndex+2*outHeight*outWidth+outWidth]) ;
    _ve_vstu_vss(vrsum3_y1x0, 4, &pOut[outIndex+3*outHeight*outWidth+outWidth]) ;
    _ve_vstu_vss(vrsum4_y1x0, 4, &pOut[outIndex+4*outHeight*outWidth+outWidth]) ;
    _ve_vstu_vss(vrsum5_y1x0, 4, &pOut[outIndex+5*outHeight*outWidth+outWidth]) ;
    _ve_vstu_vss(vrsum6_y1x0, 4, &pOut[outIndex+6*outHeight*outWidth+outWidth]) ;
    _ve_vstu_vss(vrsum7_y1x0, 4, &pOut[outIndex+7*outHeight*outWidth+outWidth]) ;
  }
  {
    _ve_lvl(VLEN) ;
    __vr vrsum0_y1x1 = _ve_vfsums_vv(vrsum01_y1x1) ;
    __vr vrsum1_y1x1 = _ve_vfsums_vv(_ve_vsll_vvs(vrsum01_y1x1,32)) ;
    __vr vrsum2_y1x1 = _ve_vfsums_vv(vrsum23_y1x1) ;
    __vr vrsum3_y1x1 = _ve_vfsums_vv(_ve_vsll_vvs(vrsum23_y1x1,32)) ;
    __vr vrsum4_y1x1 = _ve_vfsums_vv(vrsum45_y1x1) ;
    __vr vrsum5_y1x1 = _ve_vfsums_vv(_ve_vsll_vvs(vrsum45_y1x1,32)) ;
    __vr vrsum6_y1x1 = _ve_vfsums_vv(vrsum67_y1x1) ;
    __vr vrsum7_y1x1 = _ve_vfsums_vv(_ve_vsll_vvs(vrsum67_y1x1,32)) ;

    _ve_lvl(1) ;
    _ve_vstu_vss(vrsum0_y1x1, 4, &pOut[outIndex+0*outHeight*outWidth+outWidth+1]) ;
    _ve_vstu_vss(vrsum1_y1x1, 4, &pOut[outIndex+1*outHeight*outWidth+outWidth+1]) ;
    _ve_vstu_vss(vrsum2_y1x1, 4, &pOut[outIndex+2*outHeight*outWidth+outWidth+1]) ;
    _ve_vstu_vss(vrsum3_y1x1, 4, &pOut[outIndex+3*outHeight*outWidth+outWidth+1]) ;
    _ve_vstu_vss(vrsum4_y1x1, 4, &pOut[outIndex+4*outHeight*outWidth+outWidth+1]) ;
    _ve_vstu_vss(vrsum5_y1x1, 4, &pOut[outIndex+5*outHeight*outWidth+outWidth+1]) ;
    _ve_vstu_vss(vrsum6_y1x1, 4, &pOut[outIndex+6*outHeight*outWidth+outWidth+1]) ;
    _ve_vstu_vss(vrsum7_y1x1, 4, &pOut[outIndex+7*outHeight*outWidth+outWidth+1]) ;
  }
}

static inline void k16y1x1(
    const float * restrict pIn,
    const float * restrict pKernel,
    float * restrict const pOut,
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

  _ve_lvl(VLEN) ;
  __vr vrsum01_y0x0 = _ve_vbrd_vs_i64(0UL) ;
  __vr vrsum23_y0x0 = _ve_vbrd_vs_i64(0UL) ;
  __vr vrsum45_y0x0 = _ve_vbrd_vs_i64(0UL) ;
  __vr vrsum67_y0x0 = _ve_vbrd_vs_i64(0UL) ;
  __vr vrsum89_y0x0 = _ve_vbrd_vs_i64(0UL) ;
  __vr vrsumAB_y0x0 = _ve_vbrd_vs_i64(0UL) ;
  __vr vrsumCD_y0x0 = _ve_vbrd_vs_i64(0UL) ;
  __vr vrsumEF_y0x0 = _ve_vbrd_vs_i64(0UL) ;

  for (int64_t c=0; c<inChannelGroup; c+= VLEN) {
    const int64_t vl = inChannelGroup - c < VLEN ? inChannelGroup - c : VLEN ;
    _ve_lvl(vl) ;

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

	    __vr vri_h0w0 = _ve_vldu_vss(4*inHeight*inWidth, &pIn[inputIndex_h0w0]) ;

	    __vr vrk0 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+0*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk1 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+1*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk2 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+2*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk3 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+3*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk4 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+4*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk5 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+5*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk6 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+6*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk7 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+7*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk8 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+8*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk9 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+9*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrkA = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+10*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrkB = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+11*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrkC = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+12*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrkD = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+13*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrkE = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+14*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrkF = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+15*inChannelGroup*kernHeight*kernWidth]) ;

	    __vr vriP_h0w0  = _ve_vshf_vvvs(vri_h0w0, vri_h0w0, VE_VSHUFFLE_YUZU) ;

	    __vr vrk01 = _ve_vshf_vvvs(vrk0, vrk1, VE_VSHUFFLE_YUZU) ;
	    vrsum01_y0x0 = _ve_pvfmad_vvvv(vrsum01_y0x0, vriP_h0w0, vrk01) ;

	    __vr vrk23 = _ve_vshf_vvvs(vrk2, vrk3, VE_VSHUFFLE_YUZU) ;
	    vrsum23_y0x0 = _ve_pvfmad_vvvv(vrsum23_y0x0, vriP_h0w0, vrk23) ;

	    __vr vrk45 = _ve_vshf_vvvs(vrk4, vrk5, VE_VSHUFFLE_YUZU) ;
	    vrsum45_y0x0 = _ve_pvfmad_vvvv(vrsum45_y0x0, vriP_h0w0, vrk45) ;

	    __vr vrk67 = _ve_vshf_vvvs(vrk6, vrk7, VE_VSHUFFLE_YUZU) ;
	    vrsum67_y0x0 = _ve_pvfmad_vvvv(vrsum67_y0x0, vriP_h0w0, vrk67) ;

	    __vr vrk89 = _ve_vshf_vvvs(vrk8, vrk9, VE_VSHUFFLE_YUZU) ;
	    vrsum89_y0x0 = _ve_pvfmad_vvvv(vrsum89_y0x0, vriP_h0w0, vrk89) ;

	    __vr vrkAB = _ve_vshf_vvvs(vrkA, vrkB, VE_VSHUFFLE_YUZU) ;
	    vrsumAB_y0x0 = _ve_pvfmad_vvvv(vrsumAB_y0x0, vriP_h0w0, vrkAB) ;

	    __vr vrkCD = _ve_vshf_vvvs(vrkC, vrkD, VE_VSHUFFLE_YUZU) ;
	    vrsumCD_y0x0 = _ve_pvfmad_vvvv(vrsumCD_y0x0, vriP_h0w0, vrkCD) ;

	    __vr vrkEF = _ve_vshf_vvvs(vrkE, vrkF, VE_VSHUFFLE_YUZU) ;
	    vrsumEF_y0x0 = _ve_pvfmad_vvvv(vrsumEF_y0x0, vriP_h0w0, vrkEF) ;
	  }
	} // kernWidth
      }
    } // kernHeight
  } // inChannel

  {
    _ve_lvl(VLEN) ;
    __vr vrsum0_y0x0 = _ve_vfsums_vv(vrsum01_y0x0) ;
    __vr vrsum1_y0x0 = _ve_vfsums_vv(_ve_vsll_vvs(vrsum01_y0x0,32)) ;
    __vr vrsum2_y0x0 = _ve_vfsums_vv(vrsum23_y0x0) ;
    __vr vrsum3_y0x0 = _ve_vfsums_vv(_ve_vsll_vvs(vrsum23_y0x0,32)) ;
    __vr vrsum4_y0x0 = _ve_vfsums_vv(vrsum45_y0x0) ;
    __vr vrsum5_y0x0 = _ve_vfsums_vv(_ve_vsll_vvs(vrsum45_y0x0,32)) ;
    __vr vrsum6_y0x0 = _ve_vfsums_vv(vrsum67_y0x0) ;
    __vr vrsum7_y0x0 = _ve_vfsums_vv(_ve_vsll_vvs(vrsum67_y0x0,32)) ;
    __vr vrsum8_y0x0 = _ve_vfsums_vv(vrsum89_y0x0) ;
    __vr vrsum9_y0x0 = _ve_vfsums_vv(_ve_vsll_vvs(vrsum89_y0x0,32)) ;
    __vr vrsumA_y0x0 = _ve_vfsums_vv(vrsumAB_y0x0) ;
    __vr vrsumB_y0x0 = _ve_vfsums_vv(_ve_vsll_vvs(vrsumAB_y0x0,32)) ;
    __vr vrsumC_y0x0 = _ve_vfsums_vv(vrsumCD_y0x0) ;
    __vr vrsumD_y0x0 = _ve_vfsums_vv(_ve_vsll_vvs(vrsumCD_y0x0,32)) ;
    __vr vrsumE_y0x0 = _ve_vfsums_vv(vrsumEF_y0x0) ;
    __vr vrsumF_y0x0 = _ve_vfsums_vv(_ve_vsll_vvs(vrsumEF_y0x0,32)) ;

    _ve_lvl(1) ;
    _ve_vstu_vss(vrsum0_y0x0, 4, &pOut[outIndex+0*outHeight*outWidth]) ;
    _ve_vstu_vss(vrsum1_y0x0, 4, &pOut[outIndex+1*outHeight*outWidth]) ;
    _ve_vstu_vss(vrsum2_y0x0, 4, &pOut[outIndex+2*outHeight*outWidth]) ;
    _ve_vstu_vss(vrsum3_y0x0, 4, &pOut[outIndex+3*outHeight*outWidth]) ;
    _ve_vstu_vss(vrsum4_y0x0, 4, &pOut[outIndex+4*outHeight*outWidth]) ;
    _ve_vstu_vss(vrsum5_y0x0, 4, &pOut[outIndex+5*outHeight*outWidth]) ;
    _ve_vstu_vss(vrsum6_y0x0, 4, &pOut[outIndex+6*outHeight*outWidth]) ;
    _ve_vstu_vss(vrsum7_y0x0, 4, &pOut[outIndex+7*outHeight*outWidth]) ;
    _ve_vstu_vss(vrsum8_y0x0, 4, &pOut[outIndex+8*outHeight*outWidth]) ;
    _ve_vstu_vss(vrsum9_y0x0, 4, &pOut[outIndex+9*outHeight*outWidth]) ;
    _ve_vstu_vss(vrsumA_y0x0, 4, &pOut[outIndex+10*outHeight*outWidth]) ;
    _ve_vstu_vss(vrsumB_y0x0, 4, &pOut[outIndex+11*outHeight*outWidth]) ;
    _ve_vstu_vss(vrsumC_y0x0, 4, &pOut[outIndex+12*outHeight*outWidth]) ;
    _ve_vstu_vss(vrsumD_y0x0, 4, &pOut[outIndex+13*outHeight*outWidth]) ;
    _ve_vstu_vss(vrsumE_y0x0, 4, &pOut[outIndex+14*outHeight*outWidth]) ;
    _ve_vstu_vss(vrsumF_y0x0, 4, &pOut[outIndex+15*outHeight*outWidth]) ;
  }
}

static inline void k16y1x2(
    const float * restrict pIn,
    const float * restrict pKernel,
    float * restrict const pOut,
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

  _ve_lvl(VLEN) ;
  __vr vrsum01_y0x0 = _ve_vbrd_vs_i64(0UL) ;
  __vr vrsum01_y0x1 = _ve_vbrd_vs_i64(0UL) ;

  __vr vrsum23_y0x0 = _ve_vbrd_vs_i64(0UL) ;
  __vr vrsum23_y0x1 = _ve_vbrd_vs_i64(0UL) ;

  __vr vrsum45_y0x0 = _ve_vbrd_vs_i64(0UL) ;
  __vr vrsum45_y0x1 = _ve_vbrd_vs_i64(0UL) ;

  __vr vrsum67_y0x0 = _ve_vbrd_vs_i64(0UL) ;
  __vr vrsum67_y0x1 = _ve_vbrd_vs_i64(0UL) ;

  __vr vrsum89_y0x0 = _ve_vbrd_vs_i64(0UL) ;
  __vr vrsum89_y0x1 = _ve_vbrd_vs_i64(0UL) ;

  __vr vrsumAB_y0x0 = _ve_vbrd_vs_i64(0UL) ;
  __vr vrsumAB_y0x1 = _ve_vbrd_vs_i64(0UL) ;

  __vr vrsumCD_y0x0 = _ve_vbrd_vs_i64(0UL) ;
  __vr vrsumCD_y0x1 = _ve_vbrd_vs_i64(0UL) ;

  __vr vrsumEF_y0x0 = _ve_vbrd_vs_i64(0UL) ;
  __vr vrsumEF_y0x1 = _ve_vbrd_vs_i64(0UL) ;

  for (int64_t c=0; c<inChannelGroup; c+= VLEN) {
    const int64_t vl = inChannelGroup - c < VLEN ? inChannelGroup - c : VLEN ;
    _ve_lvl(vl) ;

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

	    __vr vri_h0w0 = _ve_vldu_vss(4*inHeight*inWidth, &pIn[inputIndex_h0w0]) ;
	    __vr vri_h0w1 = _ve_vldu_vss(4*inHeight*inWidth, &pIn[inputIndex_h0w1]) ;

	    __vr vrk0 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+0*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk1 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+1*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk2 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+2*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk3 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+3*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk4 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+4*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk5 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+5*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk6 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+6*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk7 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+7*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk8 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+8*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk9 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+9*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrkA = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+10*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrkB = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+11*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrkC = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+12*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrkD = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+13*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrkE = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+14*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrkF = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+15*inChannelGroup*kernHeight*kernWidth]) ;

	    __vr vriP_h0w0  = _ve_vshf_vvvs(vri_h0w0, vri_h0w0, VE_VSHUFFLE_YUZU) ;
	    __vr vriP_h0w1  = _ve_vshf_vvvs(vri_h0w1, vri_h0w1, VE_VSHUFFLE_YUZU) ;

	    __vr vrk01 = _ve_vshf_vvvs(vrk0, vrk1, VE_VSHUFFLE_YUZU) ;
	    vrsum01_y0x0 = _ve_pvfmad_vvvv(vrsum01_y0x0, vriP_h0w0, vrk01) ;
	    vrsum01_y0x1 = _ve_pvfmad_vvvv(vrsum01_y0x1, vriP_h0w1, vrk01) ;

	    __vr vrk23 = _ve_vshf_vvvs(vrk2, vrk3, VE_VSHUFFLE_YUZU) ;
	    vrsum23_y0x0 = _ve_pvfmad_vvvv(vrsum23_y0x0, vriP_h0w0, vrk23) ;
	    vrsum23_y0x1 = _ve_pvfmad_vvvv(vrsum23_y0x1, vriP_h0w1, vrk23) ;

	    __vr vrk45 = _ve_vshf_vvvs(vrk4, vrk5, VE_VSHUFFLE_YUZU) ;
	    vrsum45_y0x0 = _ve_pvfmad_vvvv(vrsum45_y0x0, vriP_h0w0, vrk45) ;
	    vrsum45_y0x1 = _ve_pvfmad_vvvv(vrsum45_y0x1, vriP_h0w1, vrk45) ;

	    __vr vrk67 = _ve_vshf_vvvs(vrk6, vrk7, VE_VSHUFFLE_YUZU) ;
	    vrsum67_y0x0 = _ve_pvfmad_vvvv(vrsum67_y0x0, vriP_h0w0, vrk67) ;
	    vrsum67_y0x1 = _ve_pvfmad_vvvv(vrsum67_y0x1, vriP_h0w1, vrk67) ;

	    __vr vrk89 = _ve_vshf_vvvs(vrk8, vrk9, VE_VSHUFFLE_YUZU) ;
	    vrsum89_y0x0 = _ve_pvfmad_vvvv(vrsum89_y0x0, vriP_h0w0, vrk89) ;
	    vrsum89_y0x1 = _ve_pvfmad_vvvv(vrsum89_y0x1, vriP_h0w1, vrk89) ;

	    __vr vrkAB = _ve_vshf_vvvs(vrkA, vrkB, VE_VSHUFFLE_YUZU) ;
	    vrsumAB_y0x0 = _ve_pvfmad_vvvv(vrsumAB_y0x0, vriP_h0w0, vrkAB) ;
	    vrsumAB_y0x1 = _ve_pvfmad_vvvv(vrsumAB_y0x1, vriP_h0w1, vrkAB) ;

	    __vr vrkCD = _ve_vshf_vvvs(vrkC, vrkD, VE_VSHUFFLE_YUZU) ;
	    vrsumCD_y0x0 = _ve_pvfmad_vvvv(vrsumCD_y0x0, vriP_h0w0, vrkCD) ;
	    vrsumCD_y0x1 = _ve_pvfmad_vvvv(vrsumCD_y0x1, vriP_h0w1, vrkCD) ;

	    __vr vrkEF = _ve_vshf_vvvs(vrkE, vrkF, VE_VSHUFFLE_YUZU) ;
	    vrsumEF_y0x0 = _ve_pvfmad_vvvv(vrsumEF_y0x0, vriP_h0w0, vrkEF) ;
	    vrsumEF_y0x1 = _ve_pvfmad_vvvv(vrsumEF_y0x1, vriP_h0w1, vrkEF) ;
	  }
	  else if( w0_valid ) {
	    int64_t inputIndex_h0w0  = inGroupOffset + ((n * inChannel + c) * inHeight + h0) * inWidth + w0;

	    int64_t kernelIndex = kernGroupOffset + ((k * inChannelGroup + c) * kernHeight + r) * kernWidth + s;

	    __vr vri_h0w0 = _ve_vldu_vss(4*inHeight*inWidth, &pIn[inputIndex_h0w0]) ;

	    __vr vrk0 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+0*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk1 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+1*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk2 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+2*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk3 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+3*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk4 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+4*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk5 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+5*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk6 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+6*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk7 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+7*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk8 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+8*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk9 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+9*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrkA = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+10*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrkB = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+11*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrkC = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+12*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrkD = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+13*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrkE = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+14*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrkF = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+15*inChannelGroup*kernHeight*kernWidth]) ;

	    __vr vriP_h0w0  = _ve_vshf_vvvs(vri_h0w0, vri_h0w0, VE_VSHUFFLE_YUZU) ;

	    __vr vrk01 = _ve_vshf_vvvs(vrk0, vrk1, VE_VSHUFFLE_YUZU) ;
	    vrsum01_y0x0 = _ve_pvfmad_vvvv(vrsum01_y0x0, vriP_h0w0, vrk01) ;

	    __vr vrk23 = _ve_vshf_vvvs(vrk2, vrk3, VE_VSHUFFLE_YUZU) ;
	    vrsum23_y0x0 = _ve_pvfmad_vvvv(vrsum23_y0x0, vriP_h0w0, vrk23) ;

	    __vr vrk45 = _ve_vshf_vvvs(vrk4, vrk5, VE_VSHUFFLE_YUZU) ;
	    vrsum45_y0x0 = _ve_pvfmad_vvvv(vrsum45_y0x0, vriP_h0w0, vrk45) ;

	    __vr vrk67 = _ve_vshf_vvvs(vrk6, vrk7, VE_VSHUFFLE_YUZU) ;
	    vrsum67_y0x0 = _ve_pvfmad_vvvv(vrsum67_y0x0, vriP_h0w0, vrk67) ;

	    __vr vrk89 = _ve_vshf_vvvs(vrk8, vrk9, VE_VSHUFFLE_YUZU) ;
	    vrsum89_y0x0 = _ve_pvfmad_vvvv(vrsum89_y0x0, vriP_h0w0, vrk89) ;

	    __vr vrkAB = _ve_vshf_vvvs(vrkA, vrkB, VE_VSHUFFLE_YUZU) ;
	    vrsumAB_y0x0 = _ve_pvfmad_vvvv(vrsumAB_y0x0, vriP_h0w0, vrkAB) ;

	    __vr vrkCD = _ve_vshf_vvvs(vrkC, vrkD, VE_VSHUFFLE_YUZU) ;
	    vrsumCD_y0x0 = _ve_pvfmad_vvvv(vrsumCD_y0x0, vriP_h0w0, vrkCD) ;

	    __vr vrkEF = _ve_vshf_vvvs(vrkE, vrkF, VE_VSHUFFLE_YUZU) ;
	    vrsumEF_y0x0 = _ve_pvfmad_vvvv(vrsumEF_y0x0, vriP_h0w0, vrkEF) ;
	  }
	  else if( w1_valid ) {
	    int64_t inputIndex_h0w1  = inGroupOffset + ((n * inChannel + c) * inHeight + h0) * inWidth + w1;

	    int64_t kernelIndex = kernGroupOffset + ((k * inChannelGroup + c) * kernHeight + r) * kernWidth + s;

	    __vr vri_h0w1 = _ve_vldu_vss(4*inHeight*inWidth, &pIn[inputIndex_h0w1]) ;

	    __vr vrk0 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+0*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk1 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+1*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk2 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+2*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk3 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+3*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk4 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+4*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk5 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+5*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk6 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+6*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk7 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+7*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk8 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+8*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk9 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+9*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrkA = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+10*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrkB = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+11*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrkC = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+12*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrkD = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+13*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrkE = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+14*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrkF = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+15*inChannelGroup*kernHeight*kernWidth]) ;

	    __vr vriP_h0w1  = _ve_vshf_vvvs(vri_h0w1, vri_h0w1, VE_VSHUFFLE_YUZU) ;

	    __vr vrk01 = _ve_vshf_vvvs(vrk0, vrk1, VE_VSHUFFLE_YUZU) ;
	    vrsum01_y0x1 = _ve_pvfmad_vvvv(vrsum01_y0x1, vriP_h0w1, vrk01) ;

	    __vr vrk23 = _ve_vshf_vvvs(vrk2, vrk3, VE_VSHUFFLE_YUZU) ;
	    vrsum23_y0x1 = _ve_pvfmad_vvvv(vrsum23_y0x1, vriP_h0w1, vrk23) ;

	    __vr vrk45 = _ve_vshf_vvvs(vrk4, vrk5, VE_VSHUFFLE_YUZU) ;
	    vrsum45_y0x1 = _ve_pvfmad_vvvv(vrsum45_y0x1, vriP_h0w1, vrk45) ;

	    __vr vrk67 = _ve_vshf_vvvs(vrk6, vrk7, VE_VSHUFFLE_YUZU) ;
	    vrsum67_y0x1 = _ve_pvfmad_vvvv(vrsum67_y0x1, vriP_h0w1, vrk67) ;

	    __vr vrk89 = _ve_vshf_vvvs(vrk8, vrk9, VE_VSHUFFLE_YUZU) ;
	    vrsum89_y0x1 = _ve_pvfmad_vvvv(vrsum89_y0x1, vriP_h0w1, vrk89) ;

	    __vr vrkAB = _ve_vshf_vvvs(vrkA, vrkB, VE_VSHUFFLE_YUZU) ;
	    vrsumAB_y0x1 = _ve_pvfmad_vvvv(vrsumAB_y0x1, vriP_h0w1, vrkAB) ;

	    __vr vrkCD = _ve_vshf_vvvs(vrkC, vrkD, VE_VSHUFFLE_YUZU) ;
	    vrsumCD_y0x1 = _ve_pvfmad_vvvv(vrsumCD_y0x1, vriP_h0w1, vrkCD) ;

	    __vr vrkEF = _ve_vshf_vvvs(vrkE, vrkF, VE_VSHUFFLE_YUZU) ;
	    vrsumEF_y0x1 = _ve_pvfmad_vvvv(vrsumEF_y0x1, vriP_h0w1, vrkEF) ;
	  }
	} // kernWidth
      }
    } // kernHeight
  } // inChannel

  {
    _ve_lvl(VLEN) ;
    __vr vrsum0_y0x0 = _ve_vfsums_vv(vrsum01_y0x0) ;
    __vr vrsum1_y0x0 = _ve_vfsums_vv(_ve_vsll_vvs(vrsum01_y0x0,32)) ;
    __vr vrsum2_y0x0 = _ve_vfsums_vv(vrsum23_y0x0) ;
    __vr vrsum3_y0x0 = _ve_vfsums_vv(_ve_vsll_vvs(vrsum23_y0x0,32)) ;
    __vr vrsum4_y0x0 = _ve_vfsums_vv(vrsum45_y0x0) ;
    __vr vrsum5_y0x0 = _ve_vfsums_vv(_ve_vsll_vvs(vrsum45_y0x0,32)) ;
    __vr vrsum6_y0x0 = _ve_vfsums_vv(vrsum67_y0x0) ;
    __vr vrsum7_y0x0 = _ve_vfsums_vv(_ve_vsll_vvs(vrsum67_y0x0,32)) ;
    __vr vrsum8_y0x0 = _ve_vfsums_vv(vrsum89_y0x0) ;
    __vr vrsum9_y0x0 = _ve_vfsums_vv(_ve_vsll_vvs(vrsum89_y0x0,32)) ;
    __vr vrsumA_y0x0 = _ve_vfsums_vv(vrsumAB_y0x0) ;
    __vr vrsumB_y0x0 = _ve_vfsums_vv(_ve_vsll_vvs(vrsumAB_y0x0,32)) ;
    __vr vrsumC_y0x0 = _ve_vfsums_vv(vrsumCD_y0x0) ;
    __vr vrsumD_y0x0 = _ve_vfsums_vv(_ve_vsll_vvs(vrsumCD_y0x0,32)) ;
    __vr vrsumE_y0x0 = _ve_vfsums_vv(vrsumEF_y0x0) ;
    __vr vrsumF_y0x0 = _ve_vfsums_vv(_ve_vsll_vvs(vrsumEF_y0x0,32)) ;

    _ve_lvl(1) ;
    _ve_vstu_vss(vrsum0_y0x0, 4, &pOut[outIndex+0*outHeight*outWidth]) ;
    _ve_vstu_vss(vrsum1_y0x0, 4, &pOut[outIndex+1*outHeight*outWidth]) ;
    _ve_vstu_vss(vrsum2_y0x0, 4, &pOut[outIndex+2*outHeight*outWidth]) ;
    _ve_vstu_vss(vrsum3_y0x0, 4, &pOut[outIndex+3*outHeight*outWidth]) ;
    _ve_vstu_vss(vrsum4_y0x0, 4, &pOut[outIndex+4*outHeight*outWidth]) ;
    _ve_vstu_vss(vrsum5_y0x0, 4, &pOut[outIndex+5*outHeight*outWidth]) ;
    _ve_vstu_vss(vrsum6_y0x0, 4, &pOut[outIndex+6*outHeight*outWidth]) ;
    _ve_vstu_vss(vrsum7_y0x0, 4, &pOut[outIndex+7*outHeight*outWidth]) ;
    _ve_vstu_vss(vrsum8_y0x0, 4, &pOut[outIndex+8*outHeight*outWidth]) ;
    _ve_vstu_vss(vrsum9_y0x0, 4, &pOut[outIndex+9*outHeight*outWidth]) ;
    _ve_vstu_vss(vrsumA_y0x0, 4, &pOut[outIndex+10*outHeight*outWidth]) ;
    _ve_vstu_vss(vrsumB_y0x0, 4, &pOut[outIndex+11*outHeight*outWidth]) ;
    _ve_vstu_vss(vrsumC_y0x0, 4, &pOut[outIndex+12*outHeight*outWidth]) ;
    _ve_vstu_vss(vrsumD_y0x0, 4, &pOut[outIndex+13*outHeight*outWidth]) ;
    _ve_vstu_vss(vrsumE_y0x0, 4, &pOut[outIndex+14*outHeight*outWidth]) ;
    _ve_vstu_vss(vrsumF_y0x0, 4, &pOut[outIndex+15*outHeight*outWidth]) ;
  }
  {
    _ve_lvl(VLEN) ;
    __vr vrsum0_y0x1 = _ve_vfsums_vv(vrsum01_y0x1) ;
    __vr vrsum1_y0x1 = _ve_vfsums_vv(_ve_vsll_vvs(vrsum01_y0x1,32)) ;
    __vr vrsum2_y0x1 = _ve_vfsums_vv(vrsum23_y0x1) ;
    __vr vrsum3_y0x1 = _ve_vfsums_vv(_ve_vsll_vvs(vrsum23_y0x1,32)) ;
    __vr vrsum4_y0x1 = _ve_vfsums_vv(vrsum45_y0x1) ;
    __vr vrsum5_y0x1 = _ve_vfsums_vv(_ve_vsll_vvs(vrsum45_y0x1,32)) ;
    __vr vrsum6_y0x1 = _ve_vfsums_vv(vrsum67_y0x1) ;
    __vr vrsum7_y0x1 = _ve_vfsums_vv(_ve_vsll_vvs(vrsum67_y0x1,32)) ;
    __vr vrsum8_y0x1 = _ve_vfsums_vv(vrsum89_y0x1) ;
    __vr vrsum9_y0x1 = _ve_vfsums_vv(_ve_vsll_vvs(vrsum89_y0x1,32)) ;
    __vr vrsumA_y0x1 = _ve_vfsums_vv(vrsumAB_y0x1) ;
    __vr vrsumB_y0x1 = _ve_vfsums_vv(_ve_vsll_vvs(vrsumAB_y0x1,32)) ;
    __vr vrsumC_y0x1 = _ve_vfsums_vv(vrsumCD_y0x1) ;
    __vr vrsumD_y0x1 = _ve_vfsums_vv(_ve_vsll_vvs(vrsumCD_y0x1,32)) ;
    __vr vrsumE_y0x1 = _ve_vfsums_vv(vrsumEF_y0x1) ;
    __vr vrsumF_y0x1 = _ve_vfsums_vv(_ve_vsll_vvs(vrsumEF_y0x1,32)) ;

    _ve_lvl(1) ;
    _ve_vstu_vss(vrsum0_y0x1, 4, &pOut[outIndex+0*outHeight*outWidth+1]) ;
    _ve_vstu_vss(vrsum1_y0x1, 4, &pOut[outIndex+1*outHeight*outWidth+1]) ;
    _ve_vstu_vss(vrsum2_y0x1, 4, &pOut[outIndex+2*outHeight*outWidth+1]) ;
    _ve_vstu_vss(vrsum3_y0x1, 4, &pOut[outIndex+3*outHeight*outWidth+1]) ;
    _ve_vstu_vss(vrsum4_y0x1, 4, &pOut[outIndex+4*outHeight*outWidth+1]) ;
    _ve_vstu_vss(vrsum5_y0x1, 4, &pOut[outIndex+5*outHeight*outWidth+1]) ;
    _ve_vstu_vss(vrsum6_y0x1, 4, &pOut[outIndex+6*outHeight*outWidth+1]) ;
    _ve_vstu_vss(vrsum7_y0x1, 4, &pOut[outIndex+7*outHeight*outWidth+1]) ;
    _ve_vstu_vss(vrsum8_y0x1, 4, &pOut[outIndex+8*outHeight*outWidth+1]) ;
    _ve_vstu_vss(vrsum9_y0x1, 4, &pOut[outIndex+9*outHeight*outWidth+1]) ;
    _ve_vstu_vss(vrsumA_y0x1, 4, &pOut[outIndex+10*outHeight*outWidth+1]) ;
    _ve_vstu_vss(vrsumB_y0x1, 4, &pOut[outIndex+11*outHeight*outWidth+1]) ;
    _ve_vstu_vss(vrsumC_y0x1, 4, &pOut[outIndex+12*outHeight*outWidth+1]) ;
    _ve_vstu_vss(vrsumD_y0x1, 4, &pOut[outIndex+13*outHeight*outWidth+1]) ;
    _ve_vstu_vss(vrsumE_y0x1, 4, &pOut[outIndex+14*outHeight*outWidth+1]) ;
    _ve_vstu_vss(vrsumF_y0x1, 4, &pOut[outIndex+15*outHeight*outWidth+1]) ;
  }
}


static inline void k16y2x1(
    const float * restrict pIn,
    const float * restrict pKernel,
    float * restrict const pOut,
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

  _ve_lvl(VLEN) ;
  __vr vrsum01_y0x0 = _ve_vbrd_vs_i64(0UL) ;
  __vr vrsum01_y1x0 = _ve_vbrd_vs_i64(0UL) ;

  __vr vrsum23_y0x0 = _ve_vbrd_vs_i64(0UL) ;
  __vr vrsum23_y1x0 = _ve_vbrd_vs_i64(0UL) ;

  __vr vrsum45_y0x0 = _ve_vbrd_vs_i64(0UL) ;
  __vr vrsum45_y1x0 = _ve_vbrd_vs_i64(0UL) ;

  __vr vrsum67_y0x0 = _ve_vbrd_vs_i64(0UL) ;
  __vr vrsum67_y1x0 = _ve_vbrd_vs_i64(0UL) ;

  __vr vrsum89_y0x0 = _ve_vbrd_vs_i64(0UL) ;
  __vr vrsum89_y1x0 = _ve_vbrd_vs_i64(0UL) ;

  __vr vrsumAB_y0x0 = _ve_vbrd_vs_i64(0UL) ;
  __vr vrsumAB_y1x0 = _ve_vbrd_vs_i64(0UL) ;

  __vr vrsumCD_y0x0 = _ve_vbrd_vs_i64(0UL) ;
  __vr vrsumCD_y1x0 = _ve_vbrd_vs_i64(0UL) ;

  __vr vrsumEF_y0x0 = _ve_vbrd_vs_i64(0UL) ;
  __vr vrsumEF_y1x0 = _ve_vbrd_vs_i64(0UL) ;

  for (int64_t c=0; c<inChannelGroup; c+= VLEN) {
    const int64_t vl = inChannelGroup - c < VLEN ? inChannelGroup - c : VLEN ;
    _ve_lvl(vl) ;

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

	    __vr vri_h0w0 = _ve_vldu_vss(4*inHeight*inWidth, &pIn[inputIndex_h0w0]) ;
	    __vr vri_h1w0 = _ve_vldu_vss(4*inHeight*inWidth, &pIn[inputIndex_h1w0]) ;

	    __vr vrk0 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+0*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk1 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+1*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk2 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+2*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk3 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+3*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk4 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+4*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk5 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+5*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk6 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+6*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk7 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+7*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk8 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+8*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk9 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+9*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrkA = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+10*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrkB = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+11*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrkC = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+12*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrkD = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+13*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrkE = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+14*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrkF = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+15*inChannelGroup*kernHeight*kernWidth]) ;

	    __vr vriP_h0w0  = _ve_vshf_vvvs(vri_h0w0, vri_h0w0, VE_VSHUFFLE_YUZU) ;
	    __vr vriP_h1w0  = _ve_vshf_vvvs(vri_h1w0, vri_h1w0, VE_VSHUFFLE_YUZU) ;

	    __vr vrk01 = _ve_vshf_vvvs(vrk0, vrk1, VE_VSHUFFLE_YUZU) ;
	    vrsum01_y0x0 = _ve_pvfmad_vvvv(vrsum01_y0x0, vriP_h0w0, vrk01) ;
	    vrsum01_y1x0 = _ve_pvfmad_vvvv(vrsum01_y1x0, vriP_h1w0, vrk01) ;

	    __vr vrk23 = _ve_vshf_vvvs(vrk2, vrk3, VE_VSHUFFLE_YUZU) ;
	    vrsum23_y0x0 = _ve_pvfmad_vvvv(vrsum23_y0x0, vriP_h0w0, vrk23) ;
	    vrsum23_y1x0 = _ve_pvfmad_vvvv(vrsum23_y1x0, vriP_h1w0, vrk23) ;

	    __vr vrk45 = _ve_vshf_vvvs(vrk4, vrk5, VE_VSHUFFLE_YUZU) ;
	    vrsum45_y0x0 = _ve_pvfmad_vvvv(vrsum45_y0x0, vriP_h0w0, vrk45) ;
	    vrsum45_y1x0 = _ve_pvfmad_vvvv(vrsum45_y1x0, vriP_h1w0, vrk45) ;

	    __vr vrk67 = _ve_vshf_vvvs(vrk6, vrk7, VE_VSHUFFLE_YUZU) ;
	    vrsum67_y0x0 = _ve_pvfmad_vvvv(vrsum67_y0x0, vriP_h0w0, vrk67) ;
	    vrsum67_y1x0 = _ve_pvfmad_vvvv(vrsum67_y1x0, vriP_h1w0, vrk67) ;

	    __vr vrk89 = _ve_vshf_vvvs(vrk8, vrk9, VE_VSHUFFLE_YUZU) ;
	    vrsum89_y0x0 = _ve_pvfmad_vvvv(vrsum89_y0x0, vriP_h0w0, vrk89) ;
	    vrsum89_y1x0 = _ve_pvfmad_vvvv(vrsum89_y1x0, vriP_h1w0, vrk89) ;

	    __vr vrkAB = _ve_vshf_vvvs(vrkA, vrkB, VE_VSHUFFLE_YUZU) ;
	    vrsumAB_y0x0 = _ve_pvfmad_vvvv(vrsumAB_y0x0, vriP_h0w0, vrkAB) ;
	    vrsumAB_y1x0 = _ve_pvfmad_vvvv(vrsumAB_y1x0, vriP_h1w0, vrkAB) ;

	    __vr vrkCD = _ve_vshf_vvvs(vrkC, vrkD, VE_VSHUFFLE_YUZU) ;
	    vrsumCD_y0x0 = _ve_pvfmad_vvvv(vrsumCD_y0x0, vriP_h0w0, vrkCD) ;
	    vrsumCD_y1x0 = _ve_pvfmad_vvvv(vrsumCD_y1x0, vriP_h1w0, vrkCD) ;

	    __vr vrkEF = _ve_vshf_vvvs(vrkE, vrkF, VE_VSHUFFLE_YUZU) ;
	    vrsumEF_y0x0 = _ve_pvfmad_vvvv(vrsumEF_y0x0, vriP_h0w0, vrkEF) ;
	    vrsumEF_y1x0 = _ve_pvfmad_vvvv(vrsumEF_y1x0, vriP_h1w0, vrkEF) ;
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

	    __vr vri_h0w0 = _ve_vldu_vss(4*inHeight*inWidth, &pIn[inputIndex_h0w0]) ;

	    __vr vrk0 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+0*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk1 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+1*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk2 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+2*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk3 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+3*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk4 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+4*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk5 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+5*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk6 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+6*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk7 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+7*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk8 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+8*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk9 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+9*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrkA = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+10*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrkB = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+11*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrkC = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+12*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrkD = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+13*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrkE = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+14*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrkF = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+15*inChannelGroup*kernHeight*kernWidth]) ;

	    __vr vriP_h0w0  = _ve_vshf_vvvs(vri_h0w0, vri_h0w0, VE_VSHUFFLE_YUZU) ;

	    __vr vrk01 = _ve_vshf_vvvs(vrk0, vrk1, VE_VSHUFFLE_YUZU) ;
	    vrsum01_y0x0 = _ve_pvfmad_vvvv(vrsum01_y0x0, vriP_h0w0, vrk01) ;

	    __vr vrk23 = _ve_vshf_vvvs(vrk2, vrk3, VE_VSHUFFLE_YUZU) ;
	    vrsum23_y0x0 = _ve_pvfmad_vvvv(vrsum23_y0x0, vriP_h0w0, vrk23) ;

	    __vr vrk45 = _ve_vshf_vvvs(vrk4, vrk5, VE_VSHUFFLE_YUZU) ;
	    vrsum45_y0x0 = _ve_pvfmad_vvvv(vrsum45_y0x0, vriP_h0w0, vrk45) ;

	    __vr vrk67 = _ve_vshf_vvvs(vrk6, vrk7, VE_VSHUFFLE_YUZU) ;
	    vrsum67_y0x0 = _ve_pvfmad_vvvv(vrsum67_y0x0, vriP_h0w0, vrk67) ;

	    __vr vrk89 = _ve_vshf_vvvs(vrk8, vrk9, VE_VSHUFFLE_YUZU) ;
	    vrsum89_y0x0 = _ve_pvfmad_vvvv(vrsum89_y0x0, vriP_h0w0, vrk89) ;

	    __vr vrkAB = _ve_vshf_vvvs(vrkA, vrkB, VE_VSHUFFLE_YUZU) ;
	    vrsumAB_y0x0 = _ve_pvfmad_vvvv(vrsumAB_y0x0, vriP_h0w0, vrkAB) ;

	    __vr vrkCD = _ve_vshf_vvvs(vrkC, vrkD, VE_VSHUFFLE_YUZU) ;
	    vrsumCD_y0x0 = _ve_pvfmad_vvvv(vrsumCD_y0x0, vriP_h0w0, vrkCD) ;

	    __vr vrkEF = _ve_vshf_vvvs(vrkE, vrkF, VE_VSHUFFLE_YUZU) ;
	    vrsumEF_y0x0 = _ve_pvfmad_vvvv(vrsumEF_y0x0, vriP_h0w0, vrkEF) ;
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

	    __vr vri_h1w0 = _ve_vldu_vss(4*inHeight*inWidth, &pIn[inputIndex_h1w0]) ;

	    __vr vrk0 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+0*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk1 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+1*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk2 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+2*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk3 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+3*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk4 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+4*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk5 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+5*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk6 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+6*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk7 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+7*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk8 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+8*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk9 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+9*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrkA = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+10*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrkB = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+11*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrkC = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+12*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrkD = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+13*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrkE = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+14*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrkF = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+15*inChannelGroup*kernHeight*kernWidth]) ;

	    __vr vriP_h1w0  = _ve_vshf_vvvs(vri_h1w0, vri_h1w0, VE_VSHUFFLE_YUZU) ;

	    __vr vrk01 = _ve_vshf_vvvs(vrk0, vrk1, VE_VSHUFFLE_YUZU) ;
	    vrsum01_y1x0 = _ve_pvfmad_vvvv(vrsum01_y1x0, vriP_h1w0, vrk01) ;

	    __vr vrk23 = _ve_vshf_vvvs(vrk2, vrk3, VE_VSHUFFLE_YUZU) ;
	    vrsum23_y1x0 = _ve_pvfmad_vvvv(vrsum23_y1x0, vriP_h1w0, vrk23) ;

	    __vr vrk45 = _ve_vshf_vvvs(vrk4, vrk5, VE_VSHUFFLE_YUZU) ;
	    vrsum45_y1x0 = _ve_pvfmad_vvvv(vrsum45_y1x0, vriP_h1w0, vrk45) ;

	    __vr vrk67 = _ve_vshf_vvvs(vrk6, vrk7, VE_VSHUFFLE_YUZU) ;
	    vrsum67_y1x0 = _ve_pvfmad_vvvv(vrsum67_y1x0, vriP_h1w0, vrk67) ;

	    __vr vrk89 = _ve_vshf_vvvs(vrk8, vrk9, VE_VSHUFFLE_YUZU) ;
	    vrsum89_y1x0 = _ve_pvfmad_vvvv(vrsum89_y1x0, vriP_h1w0, vrk89) ;

	    __vr vrkAB = _ve_vshf_vvvs(vrkA, vrkB, VE_VSHUFFLE_YUZU) ;
	    vrsumAB_y1x0 = _ve_pvfmad_vvvv(vrsumAB_y1x0, vriP_h1w0, vrkAB) ;

	    __vr vrkCD = _ve_vshf_vvvs(vrkC, vrkD, VE_VSHUFFLE_YUZU) ;
	    vrsumCD_y1x0 = _ve_pvfmad_vvvv(vrsumCD_y1x0, vriP_h1w0, vrkCD) ;

	    __vr vrkEF = _ve_vshf_vvvs(vrkE, vrkF, VE_VSHUFFLE_YUZU) ;
	    vrsumEF_y1x0 = _ve_pvfmad_vvvv(vrsumEF_y1x0, vriP_h1w0, vrkEF) ;
	  }
	} // kernWidth
      }
    } // kernHeight
  } // inChannel

  {
    _ve_lvl(VLEN) ;
    __vr vrsum0_y0x0 = _ve_vfsums_vv(vrsum01_y0x0) ;
    __vr vrsum1_y0x0 = _ve_vfsums_vv(_ve_vsll_vvs(vrsum01_y0x0,32)) ;
    __vr vrsum2_y0x0 = _ve_vfsums_vv(vrsum23_y0x0) ;
    __vr vrsum3_y0x0 = _ve_vfsums_vv(_ve_vsll_vvs(vrsum23_y0x0,32)) ;
    __vr vrsum4_y0x0 = _ve_vfsums_vv(vrsum45_y0x0) ;
    __vr vrsum5_y0x0 = _ve_vfsums_vv(_ve_vsll_vvs(vrsum45_y0x0,32)) ;
    __vr vrsum6_y0x0 = _ve_vfsums_vv(vrsum67_y0x0) ;
    __vr vrsum7_y0x0 = _ve_vfsums_vv(_ve_vsll_vvs(vrsum67_y0x0,32)) ;
    __vr vrsum8_y0x0 = _ve_vfsums_vv(vrsum89_y0x0) ;
    __vr vrsum9_y0x0 = _ve_vfsums_vv(_ve_vsll_vvs(vrsum89_y0x0,32)) ;
    __vr vrsumA_y0x0 = _ve_vfsums_vv(vrsumAB_y0x0) ;
    __vr vrsumB_y0x0 = _ve_vfsums_vv(_ve_vsll_vvs(vrsumAB_y0x0,32)) ;
    __vr vrsumC_y0x0 = _ve_vfsums_vv(vrsumCD_y0x0) ;
    __vr vrsumD_y0x0 = _ve_vfsums_vv(_ve_vsll_vvs(vrsumCD_y0x0,32)) ;
    __vr vrsumE_y0x0 = _ve_vfsums_vv(vrsumEF_y0x0) ;
    __vr vrsumF_y0x0 = _ve_vfsums_vv(_ve_vsll_vvs(vrsumEF_y0x0,32)) ;

    _ve_lvl(1) ;
    _ve_vstu_vss(vrsum0_y0x0, 4, &pOut[outIndex+0*outHeight*outWidth]) ;
    _ve_vstu_vss(vrsum1_y0x0, 4, &pOut[outIndex+1*outHeight*outWidth]) ;
    _ve_vstu_vss(vrsum2_y0x0, 4, &pOut[outIndex+2*outHeight*outWidth]) ;
    _ve_vstu_vss(vrsum3_y0x0, 4, &pOut[outIndex+3*outHeight*outWidth]) ;
    _ve_vstu_vss(vrsum4_y0x0, 4, &pOut[outIndex+4*outHeight*outWidth]) ;
    _ve_vstu_vss(vrsum5_y0x0, 4, &pOut[outIndex+5*outHeight*outWidth]) ;
    _ve_vstu_vss(vrsum6_y0x0, 4, &pOut[outIndex+6*outHeight*outWidth]) ;
    _ve_vstu_vss(vrsum7_y0x0, 4, &pOut[outIndex+7*outHeight*outWidth]) ;
    _ve_vstu_vss(vrsum8_y0x0, 4, &pOut[outIndex+8*outHeight*outWidth]) ;
    _ve_vstu_vss(vrsum9_y0x0, 4, &pOut[outIndex+9*outHeight*outWidth]) ;
    _ve_vstu_vss(vrsumA_y0x0, 4, &pOut[outIndex+10*outHeight*outWidth]) ;
    _ve_vstu_vss(vrsumB_y0x0, 4, &pOut[outIndex+11*outHeight*outWidth]) ;
    _ve_vstu_vss(vrsumC_y0x0, 4, &pOut[outIndex+12*outHeight*outWidth]) ;
    _ve_vstu_vss(vrsumD_y0x0, 4, &pOut[outIndex+13*outHeight*outWidth]) ;
    _ve_vstu_vss(vrsumE_y0x0, 4, &pOut[outIndex+14*outHeight*outWidth]) ;
    _ve_vstu_vss(vrsumF_y0x0, 4, &pOut[outIndex+15*outHeight*outWidth]) ;
  }
  {
    _ve_lvl(VLEN) ;
    __vr vrsum0_y1x0 = _ve_vfsums_vv(vrsum01_y1x0) ;
    __vr vrsum1_y1x0 = _ve_vfsums_vv(_ve_vsll_vvs(vrsum01_y1x0,32)) ;
    __vr vrsum2_y1x0 = _ve_vfsums_vv(vrsum23_y1x0) ;
    __vr vrsum3_y1x0 = _ve_vfsums_vv(_ve_vsll_vvs(vrsum23_y1x0,32)) ;
    __vr vrsum4_y1x0 = _ve_vfsums_vv(vrsum45_y1x0) ;
    __vr vrsum5_y1x0 = _ve_vfsums_vv(_ve_vsll_vvs(vrsum45_y1x0,32)) ;
    __vr vrsum6_y1x0 = _ve_vfsums_vv(vrsum67_y1x0) ;
    __vr vrsum7_y1x0 = _ve_vfsums_vv(_ve_vsll_vvs(vrsum67_y1x0,32)) ;
    __vr vrsum8_y1x0 = _ve_vfsums_vv(vrsum89_y1x0) ;
    __vr vrsum9_y1x0 = _ve_vfsums_vv(_ve_vsll_vvs(vrsum89_y1x0,32)) ;
    __vr vrsumA_y1x0 = _ve_vfsums_vv(vrsumAB_y1x0) ;
    __vr vrsumB_y1x0 = _ve_vfsums_vv(_ve_vsll_vvs(vrsumAB_y1x0,32)) ;
    __vr vrsumC_y1x0 = _ve_vfsums_vv(vrsumCD_y1x0) ;
    __vr vrsumD_y1x0 = _ve_vfsums_vv(_ve_vsll_vvs(vrsumCD_y1x0,32)) ;
    __vr vrsumE_y1x0 = _ve_vfsums_vv(vrsumEF_y1x0) ;
    __vr vrsumF_y1x0 = _ve_vfsums_vv(_ve_vsll_vvs(vrsumEF_y1x0,32)) ;

    _ve_lvl(1) ;
    _ve_vstu_vss(vrsum0_y1x0, 4, &pOut[outIndex+0*outHeight*outWidth+outWidth]) ;
    _ve_vstu_vss(vrsum1_y1x0, 4, &pOut[outIndex+1*outHeight*outWidth+outWidth]) ;
    _ve_vstu_vss(vrsum2_y1x0, 4, &pOut[outIndex+2*outHeight*outWidth+outWidth]) ;
    _ve_vstu_vss(vrsum3_y1x0, 4, &pOut[outIndex+3*outHeight*outWidth+outWidth]) ;
    _ve_vstu_vss(vrsum4_y1x0, 4, &pOut[outIndex+4*outHeight*outWidth+outWidth]) ;
    _ve_vstu_vss(vrsum5_y1x0, 4, &pOut[outIndex+5*outHeight*outWidth+outWidth]) ;
    _ve_vstu_vss(vrsum6_y1x0, 4, &pOut[outIndex+6*outHeight*outWidth+outWidth]) ;
    _ve_vstu_vss(vrsum7_y1x0, 4, &pOut[outIndex+7*outHeight*outWidth+outWidth]) ;
    _ve_vstu_vss(vrsum8_y1x0, 4, &pOut[outIndex+8*outHeight*outWidth+outWidth]) ;
    _ve_vstu_vss(vrsum9_y1x0, 4, &pOut[outIndex+9*outHeight*outWidth+outWidth]) ;
    _ve_vstu_vss(vrsumA_y1x0, 4, &pOut[outIndex+10*outHeight*outWidth+outWidth]) ;
    _ve_vstu_vss(vrsumB_y1x0, 4, &pOut[outIndex+11*outHeight*outWidth+outWidth]) ;
    _ve_vstu_vss(vrsumC_y1x0, 4, &pOut[outIndex+12*outHeight*outWidth+outWidth]) ;
    _ve_vstu_vss(vrsumD_y1x0, 4, &pOut[outIndex+13*outHeight*outWidth+outWidth]) ;
    _ve_vstu_vss(vrsumE_y1x0, 4, &pOut[outIndex+14*outHeight*outWidth+outWidth]) ;
    _ve_vstu_vss(vrsumF_y1x0, 4, &pOut[outIndex+15*outHeight*outWidth+outWidth]) ;
  }
}

static inline void k16y2x2(
    const float * restrict pIn,
    const float * restrict pKernel,
    float * restrict const pOut,
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

  _ve_lvl(VLEN) ;
  __vr vrsum01_y0x0 = _ve_vbrd_vs_i64(0UL) ;
  __vr vrsum01_y0x1 = _ve_vbrd_vs_i64(0UL) ;
  __vr vrsum01_y1x0 = _ve_vbrd_vs_i64(0UL) ;
  __vr vrsum01_y1x1 = _ve_vbrd_vs_i64(0UL) ;

  __vr vrsum23_y0x0 = _ve_vbrd_vs_i64(0UL) ;
  __vr vrsum23_y0x1 = _ve_vbrd_vs_i64(0UL) ;
  __vr vrsum23_y1x0 = _ve_vbrd_vs_i64(0UL) ;
  __vr vrsum23_y1x1 = _ve_vbrd_vs_i64(0UL) ;

  __vr vrsum45_y0x0 = _ve_vbrd_vs_i64(0UL) ;
  __vr vrsum45_y0x1 = _ve_vbrd_vs_i64(0UL) ;
  __vr vrsum45_y1x0 = _ve_vbrd_vs_i64(0UL) ;
  __vr vrsum45_y1x1 = _ve_vbrd_vs_i64(0UL) ;

  __vr vrsum67_y0x0 = _ve_vbrd_vs_i64(0UL) ;
  __vr vrsum67_y0x1 = _ve_vbrd_vs_i64(0UL) ;
  __vr vrsum67_y1x0 = _ve_vbrd_vs_i64(0UL) ;
  __vr vrsum67_y1x1 = _ve_vbrd_vs_i64(0UL) ;

  __vr vrsum89_y0x0 = _ve_vbrd_vs_i64(0UL) ;
  __vr vrsum89_y0x1 = _ve_vbrd_vs_i64(0UL) ;
  __vr vrsum89_y1x0 = _ve_vbrd_vs_i64(0UL) ;
  __vr vrsum89_y1x1 = _ve_vbrd_vs_i64(0UL) ;

  __vr vrsumAB_y0x0 = _ve_vbrd_vs_i64(0UL) ;
  __vr vrsumAB_y0x1 = _ve_vbrd_vs_i64(0UL) ;
  __vr vrsumAB_y1x0 = _ve_vbrd_vs_i64(0UL) ;
  __vr vrsumAB_y1x1 = _ve_vbrd_vs_i64(0UL) ;

  __vr vrsumCD_y0x0 = _ve_vbrd_vs_i64(0UL) ;
  __vr vrsumCD_y0x1 = _ve_vbrd_vs_i64(0UL) ;
  __vr vrsumCD_y1x0 = _ve_vbrd_vs_i64(0UL) ;
  __vr vrsumCD_y1x1 = _ve_vbrd_vs_i64(0UL) ;

  __vr vrsumEF_y0x0 = _ve_vbrd_vs_i64(0UL) ;
  __vr vrsumEF_y0x1 = _ve_vbrd_vs_i64(0UL) ;
  __vr vrsumEF_y1x0 = _ve_vbrd_vs_i64(0UL) ;
  __vr vrsumEF_y1x1 = _ve_vbrd_vs_i64(0UL) ;

  for (int64_t c=0; c<inChannelGroup; c+= VLEN) {
    const int64_t vl = inChannelGroup - c < VLEN ? inChannelGroup - c : VLEN ;
    _ve_lvl(vl) ;

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

	    __vr vri_h0w0 = _ve_vldu_vss(4*inHeight*inWidth, &pIn[inputIndex_h0w0]) ;
	    __vr vri_h0w1 = _ve_vldu_vss(4*inHeight*inWidth, &pIn[inputIndex_h0w1]) ;
	    __vr vri_h1w0 = _ve_vldu_vss(4*inHeight*inWidth, &pIn[inputIndex_h1w0]) ;
	    __vr vri_h1w1 = _ve_vldu_vss(4*inHeight*inWidth, &pIn[inputIndex_h1w1]) ;

	    __vr vrk0 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+0*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk1 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+1*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk2 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+2*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk3 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+3*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk4 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+4*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk5 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+5*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk6 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+6*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk7 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+7*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk8 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+8*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk9 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+9*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrkA = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+10*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrkB = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+11*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrkC = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+12*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrkD = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+13*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrkE = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+14*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrkF = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+15*inChannelGroup*kernHeight*kernWidth]) ;

	    __vr vriP_h0w0  = _ve_vshf_vvvs(vri_h0w0, vri_h0w0, VE_VSHUFFLE_YUZU) ;
	    __vr vriP_h0w1  = _ve_vshf_vvvs(vri_h0w1, vri_h0w1, VE_VSHUFFLE_YUZU) ;
	    __vr vriP_h1w0  = _ve_vshf_vvvs(vri_h1w0, vri_h1w0, VE_VSHUFFLE_YUZU) ;
	    __vr vriP_h1w1  = _ve_vshf_vvvs(vri_h1w1, vri_h1w1, VE_VSHUFFLE_YUZU) ;

	    __vr vrk01 = _ve_vshf_vvvs(vrk0, vrk1, VE_VSHUFFLE_YUZU) ;
	    vrsum01_y0x0 = _ve_pvfmad_vvvv(vrsum01_y0x0, vriP_h0w0, vrk01) ;
	    vrsum01_y0x1 = _ve_pvfmad_vvvv(vrsum01_y0x1, vriP_h0w1, vrk01) ;
	    vrsum01_y1x0 = _ve_pvfmad_vvvv(vrsum01_y1x0, vriP_h1w0, vrk01) ;
	    vrsum01_y1x1 = _ve_pvfmad_vvvv(vrsum01_y1x1, vriP_h1w1, vrk01) ;

	    __vr vrk23 = _ve_vshf_vvvs(vrk2, vrk3, VE_VSHUFFLE_YUZU) ;
	    vrsum23_y0x0 = _ve_pvfmad_vvvv(vrsum23_y0x0, vriP_h0w0, vrk23) ;
	    vrsum23_y0x1 = _ve_pvfmad_vvvv(vrsum23_y0x1, vriP_h0w1, vrk23) ;
	    vrsum23_y1x0 = _ve_pvfmad_vvvv(vrsum23_y1x0, vriP_h1w0, vrk23) ;
	    vrsum23_y1x1 = _ve_pvfmad_vvvv(vrsum23_y1x1, vriP_h1w1, vrk23) ;

	    __vr vrk45 = _ve_vshf_vvvs(vrk4, vrk5, VE_VSHUFFLE_YUZU) ;
	    vrsum45_y0x0 = _ve_pvfmad_vvvv(vrsum45_y0x0, vriP_h0w0, vrk45) ;
	    vrsum45_y0x1 = _ve_pvfmad_vvvv(vrsum45_y0x1, vriP_h0w1, vrk45) ;
	    vrsum45_y1x0 = _ve_pvfmad_vvvv(vrsum45_y1x0, vriP_h1w0, vrk45) ;
	    vrsum45_y1x1 = _ve_pvfmad_vvvv(vrsum45_y1x1, vriP_h1w1, vrk45) ;

	    __vr vrk67 = _ve_vshf_vvvs(vrk6, vrk7, VE_VSHUFFLE_YUZU) ;
	    vrsum67_y0x0 = _ve_pvfmad_vvvv(vrsum67_y0x0, vriP_h0w0, vrk67) ;
	    vrsum67_y0x1 = _ve_pvfmad_vvvv(vrsum67_y0x1, vriP_h0w1, vrk67) ;
	    vrsum67_y1x0 = _ve_pvfmad_vvvv(vrsum67_y1x0, vriP_h1w0, vrk67) ;
	    vrsum67_y1x1 = _ve_pvfmad_vvvv(vrsum67_y1x1, vriP_h1w1, vrk67) ;

	    __vr vrk89 = _ve_vshf_vvvs(vrk8, vrk9, VE_VSHUFFLE_YUZU) ;
	    vrsum89_y0x0 = _ve_pvfmad_vvvv(vrsum89_y0x0, vriP_h0w0, vrk89) ;
	    vrsum89_y0x1 = _ve_pvfmad_vvvv(vrsum89_y0x1, vriP_h0w1, vrk89) ;
	    vrsum89_y1x0 = _ve_pvfmad_vvvv(vrsum89_y1x0, vriP_h1w0, vrk89) ;
	    vrsum89_y1x1 = _ve_pvfmad_vvvv(vrsum89_y1x1, vriP_h1w1, vrk89) ;

	    __vr vrkAB = _ve_vshf_vvvs(vrkA, vrkB, VE_VSHUFFLE_YUZU) ;
	    vrsumAB_y0x0 = _ve_pvfmad_vvvv(vrsumAB_y0x0, vriP_h0w0, vrkAB) ;
	    vrsumAB_y0x1 = _ve_pvfmad_vvvv(vrsumAB_y0x1, vriP_h0w1, vrkAB) ;
	    vrsumAB_y1x0 = _ve_pvfmad_vvvv(vrsumAB_y1x0, vriP_h1w0, vrkAB) ;
	    vrsumAB_y1x1 = _ve_pvfmad_vvvv(vrsumAB_y1x1, vriP_h1w1, vrkAB) ;

	    __vr vrkCD = _ve_vshf_vvvs(vrkC, vrkD, VE_VSHUFFLE_YUZU) ;
	    vrsumCD_y0x0 = _ve_pvfmad_vvvv(vrsumCD_y0x0, vriP_h0w0, vrkCD) ;
	    vrsumCD_y0x1 = _ve_pvfmad_vvvv(vrsumCD_y0x1, vriP_h0w1, vrkCD) ;
	    vrsumCD_y1x0 = _ve_pvfmad_vvvv(vrsumCD_y1x0, vriP_h1w0, vrkCD) ;
	    vrsumCD_y1x1 = _ve_pvfmad_vvvv(vrsumCD_y1x1, vriP_h1w1, vrkCD) ;

	    __vr vrkEF = _ve_vshf_vvvs(vrkE, vrkF, VE_VSHUFFLE_YUZU) ;
	    vrsumEF_y0x0 = _ve_pvfmad_vvvv(vrsumEF_y0x0, vriP_h0w0, vrkEF) ;
	    vrsumEF_y0x1 = _ve_pvfmad_vvvv(vrsumEF_y0x1, vriP_h0w1, vrkEF) ;
	    vrsumEF_y1x0 = _ve_pvfmad_vvvv(vrsumEF_y1x0, vriP_h1w0, vrkEF) ;
	    vrsumEF_y1x1 = _ve_pvfmad_vvvv(vrsumEF_y1x1, vriP_h1w1, vrkEF) ;
	  }
	  else if( w0_valid ) {
	    int64_t inputIndex_h0w0  = inGroupOffset + ((n * inChannel + c) * inHeight + h0) * inWidth + w0;
	    int64_t inputIndex_h1w0  = inGroupOffset + ((n * inChannel + c) * inHeight + h1) * inWidth + w0;

	    int64_t kernelIndex = kernGroupOffset + ((k * inChannelGroup + c) * kernHeight + r) * kernWidth + s;

	    __vr vri_h0w0 = _ve_vldu_vss(4*inHeight*inWidth, &pIn[inputIndex_h0w0]) ;
	    __vr vri_h1w0 = _ve_vldu_vss(4*inHeight*inWidth, &pIn[inputIndex_h1w0]) ;

	    __vr vrk0 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+0*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk1 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+1*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk2 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+2*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk3 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+3*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk4 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+4*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk5 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+5*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk6 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+6*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk7 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+7*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk8 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+8*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk9 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+9*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrkA = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+10*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrkB = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+11*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrkC = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+12*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrkD = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+13*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrkE = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+14*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrkF = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+15*inChannelGroup*kernHeight*kernWidth]) ;

	    __vr vriP_h0w0  = _ve_vshf_vvvs(vri_h0w0, vri_h0w0, VE_VSHUFFLE_YUZU) ;
	    __vr vriP_h1w0  = _ve_vshf_vvvs(vri_h1w0, vri_h1w0, VE_VSHUFFLE_YUZU) ;

	    __vr vrk01 = _ve_vshf_vvvs(vrk0, vrk1, VE_VSHUFFLE_YUZU) ;
	    vrsum01_y0x0 = _ve_pvfmad_vvvv(vrsum01_y0x0, vriP_h0w0, vrk01) ;
	    vrsum01_y1x0 = _ve_pvfmad_vvvv(vrsum01_y1x0, vriP_h1w0, vrk01) ;

	    __vr vrk23 = _ve_vshf_vvvs(vrk2, vrk3, VE_VSHUFFLE_YUZU) ;
	    vrsum23_y0x0 = _ve_pvfmad_vvvv(vrsum23_y0x0, vriP_h0w0, vrk23) ;
	    vrsum23_y1x0 = _ve_pvfmad_vvvv(vrsum23_y1x0, vriP_h1w0, vrk23) ;

	    __vr vrk45 = _ve_vshf_vvvs(vrk4, vrk5, VE_VSHUFFLE_YUZU) ;
	    vrsum45_y0x0 = _ve_pvfmad_vvvv(vrsum45_y0x0, vriP_h0w0, vrk45) ;
	    vrsum45_y1x0 = _ve_pvfmad_vvvv(vrsum45_y1x0, vriP_h1w0, vrk45) ;

	    __vr vrk67 = _ve_vshf_vvvs(vrk6, vrk7, VE_VSHUFFLE_YUZU) ;
	    vrsum67_y0x0 = _ve_pvfmad_vvvv(vrsum67_y0x0, vriP_h0w0, vrk67) ;
	    vrsum67_y1x0 = _ve_pvfmad_vvvv(vrsum67_y1x0, vriP_h1w0, vrk67) ;

	    __vr vrk89 = _ve_vshf_vvvs(vrk8, vrk9, VE_VSHUFFLE_YUZU) ;
	    vrsum89_y0x0 = _ve_pvfmad_vvvv(vrsum89_y0x0, vriP_h0w0, vrk89) ;
	    vrsum89_y1x0 = _ve_pvfmad_vvvv(vrsum89_y1x0, vriP_h1w0, vrk89) ;

	    __vr vrkAB = _ve_vshf_vvvs(vrkA, vrkB, VE_VSHUFFLE_YUZU) ;
	    vrsumAB_y0x0 = _ve_pvfmad_vvvv(vrsumAB_y0x0, vriP_h0w0, vrkAB) ;
	    vrsumAB_y1x0 = _ve_pvfmad_vvvv(vrsumAB_y1x0, vriP_h1w0, vrkAB) ;

	    __vr vrkCD = _ve_vshf_vvvs(vrkC, vrkD, VE_VSHUFFLE_YUZU) ;
	    vrsumCD_y0x0 = _ve_pvfmad_vvvv(vrsumCD_y0x0, vriP_h0w0, vrkCD) ;
	    vrsumCD_y1x0 = _ve_pvfmad_vvvv(vrsumCD_y1x0, vriP_h1w0, vrkCD) ;

	    __vr vrkEF = _ve_vshf_vvvs(vrkE, vrkF, VE_VSHUFFLE_YUZU) ;
	    vrsumEF_y0x0 = _ve_pvfmad_vvvv(vrsumEF_y0x0, vriP_h0w0, vrkEF) ;
	    vrsumEF_y1x0 = _ve_pvfmad_vvvv(vrsumEF_y1x0, vriP_h1w0, vrkEF) ;
	  }
	  else if( w1_valid ) {
	    int64_t inputIndex_h0w1  = inGroupOffset + ((n * inChannel + c) * inHeight + h0) * inWidth + w1;
	    int64_t inputIndex_h1w1  = inGroupOffset + ((n * inChannel + c) * inHeight + h1) * inWidth + w1;

	    int64_t kernelIndex = kernGroupOffset + ((k * inChannelGroup + c) * kernHeight + r) * kernWidth + s;

	    __vr vri_h0w1 = _ve_vldu_vss(4*inHeight*inWidth, &pIn[inputIndex_h0w1]) ;
	    __vr vri_h1w1 = _ve_vldu_vss(4*inHeight*inWidth, &pIn[inputIndex_h1w1]) ;

	    __vr vrk0 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+0*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk1 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+1*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk2 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+2*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk3 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+3*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk4 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+4*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk5 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+5*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk6 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+6*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk7 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+7*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk8 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+8*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk9 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+9*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrkA = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+10*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrkB = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+11*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrkC = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+12*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrkD = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+13*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrkE = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+14*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrkF = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+15*inChannelGroup*kernHeight*kernWidth]) ;

	    __vr vriP_h0w1  = _ve_vshf_vvvs(vri_h0w1, vri_h0w1, VE_VSHUFFLE_YUZU) ;
	    __vr vriP_h1w1  = _ve_vshf_vvvs(vri_h1w1, vri_h1w1, VE_VSHUFFLE_YUZU) ;

	    __vr vrk01 = _ve_vshf_vvvs(vrk0, vrk1, VE_VSHUFFLE_YUZU) ;
	    vrsum01_y0x1 = _ve_pvfmad_vvvv(vrsum01_y0x1, vriP_h0w1, vrk01) ;
	    vrsum01_y1x1 = _ve_pvfmad_vvvv(vrsum01_y1x1, vriP_h1w1, vrk01) ;

	    __vr vrk23 = _ve_vshf_vvvs(vrk2, vrk3, VE_VSHUFFLE_YUZU) ;
	    vrsum23_y0x1 = _ve_pvfmad_vvvv(vrsum23_y0x1, vriP_h0w1, vrk23) ;
	    vrsum23_y1x1 = _ve_pvfmad_vvvv(vrsum23_y1x1, vriP_h1w1, vrk23) ;

	    __vr vrk45 = _ve_vshf_vvvs(vrk4, vrk5, VE_VSHUFFLE_YUZU) ;
	    vrsum45_y0x1 = _ve_pvfmad_vvvv(vrsum45_y0x1, vriP_h0w1, vrk45) ;
	    vrsum45_y1x1 = _ve_pvfmad_vvvv(vrsum45_y1x1, vriP_h1w1, vrk45) ;

	    __vr vrk67 = _ve_vshf_vvvs(vrk6, vrk7, VE_VSHUFFLE_YUZU) ;
	    vrsum67_y0x1 = _ve_pvfmad_vvvv(vrsum67_y0x1, vriP_h0w1, vrk67) ;
	    vrsum67_y1x1 = _ve_pvfmad_vvvv(vrsum67_y1x1, vriP_h1w1, vrk67) ;

	    __vr vrk89 = _ve_vshf_vvvs(vrk8, vrk9, VE_VSHUFFLE_YUZU) ;
	    vrsum89_y0x1 = _ve_pvfmad_vvvv(vrsum89_y0x1, vriP_h0w1, vrk89) ;
	    vrsum89_y1x1 = _ve_pvfmad_vvvv(vrsum89_y1x1, vriP_h1w1, vrk89) ;

	    __vr vrkAB = _ve_vshf_vvvs(vrkA, vrkB, VE_VSHUFFLE_YUZU) ;
	    vrsumAB_y0x1 = _ve_pvfmad_vvvv(vrsumAB_y0x1, vriP_h0w1, vrkAB) ;
	    vrsumAB_y1x1 = _ve_pvfmad_vvvv(vrsumAB_y1x1, vriP_h1w1, vrkAB) ;

	    __vr vrkCD = _ve_vshf_vvvs(vrkC, vrkD, VE_VSHUFFLE_YUZU) ;
	    vrsumCD_y0x1 = _ve_pvfmad_vvvv(vrsumCD_y0x1, vriP_h0w1, vrkCD) ;
	    vrsumCD_y1x1 = _ve_pvfmad_vvvv(vrsumCD_y1x1, vriP_h1w1, vrkCD) ;

	    __vr vrkEF = _ve_vshf_vvvs(vrkE, vrkF, VE_VSHUFFLE_YUZU) ;
	    vrsumEF_y0x1 = _ve_pvfmad_vvvv(vrsumEF_y0x1, vriP_h0w1, vrkEF) ;
	    vrsumEF_y1x1 = _ve_pvfmad_vvvv(vrsumEF_y1x1, vriP_h1w1, vrkEF) ;
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

	    __vr vri_h0w0 = _ve_vldu_vss(4*inHeight*inWidth, &pIn[inputIndex_h0w0]) ;
	    __vr vri_h0w1 = _ve_vldu_vss(4*inHeight*inWidth, &pIn[inputIndex_h0w1]) ;

	    __vr vrk0 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+0*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk1 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+1*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk2 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+2*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk3 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+3*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk4 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+4*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk5 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+5*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk6 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+6*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk7 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+7*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk8 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+8*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk9 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+9*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrkA = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+10*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrkB = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+11*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrkC = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+12*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrkD = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+13*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrkE = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+14*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrkF = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+15*inChannelGroup*kernHeight*kernWidth]) ;

	    __vr vriP_h0w0  = _ve_vshf_vvvs(vri_h0w0, vri_h0w0, VE_VSHUFFLE_YUZU) ;
	    __vr vriP_h0w1  = _ve_vshf_vvvs(vri_h0w1, vri_h0w1, VE_VSHUFFLE_YUZU) ;

	    __vr vrk01 = _ve_vshf_vvvs(vrk0, vrk1, VE_VSHUFFLE_YUZU) ;
	    vrsum01_y0x0 = _ve_pvfmad_vvvv(vrsum01_y0x0, vriP_h0w0, vrk01) ;
	    vrsum01_y0x1 = _ve_pvfmad_vvvv(vrsum01_y0x1, vriP_h0w1, vrk01) ;

	    __vr vrk23 = _ve_vshf_vvvs(vrk2, vrk3, VE_VSHUFFLE_YUZU) ;
	    vrsum23_y0x0 = _ve_pvfmad_vvvv(vrsum23_y0x0, vriP_h0w0, vrk23) ;
	    vrsum23_y0x1 = _ve_pvfmad_vvvv(vrsum23_y0x1, vriP_h0w1, vrk23) ;

	    __vr vrk45 = _ve_vshf_vvvs(vrk4, vrk5, VE_VSHUFFLE_YUZU) ;
	    vrsum45_y0x0 = _ve_pvfmad_vvvv(vrsum45_y0x0, vriP_h0w0, vrk45) ;
	    vrsum45_y0x1 = _ve_pvfmad_vvvv(vrsum45_y0x1, vriP_h0w1, vrk45) ;

	    __vr vrk67 = _ve_vshf_vvvs(vrk6, vrk7, VE_VSHUFFLE_YUZU) ;
	    vrsum67_y0x0 = _ve_pvfmad_vvvv(vrsum67_y0x0, vriP_h0w0, vrk67) ;
	    vrsum67_y0x1 = _ve_pvfmad_vvvv(vrsum67_y0x1, vriP_h0w1, vrk67) ;

	    __vr vrk89 = _ve_vshf_vvvs(vrk8, vrk9, VE_VSHUFFLE_YUZU) ;
	    vrsum89_y0x0 = _ve_pvfmad_vvvv(vrsum89_y0x0, vriP_h0w0, vrk89) ;
	    vrsum89_y0x1 = _ve_pvfmad_vvvv(vrsum89_y0x1, vriP_h0w1, vrk89) ;

	    __vr vrkAB = _ve_vshf_vvvs(vrkA, vrkB, VE_VSHUFFLE_YUZU) ;
	    vrsumAB_y0x0 = _ve_pvfmad_vvvv(vrsumAB_y0x0, vriP_h0w0, vrkAB) ;
	    vrsumAB_y0x1 = _ve_pvfmad_vvvv(vrsumAB_y0x1, vriP_h0w1, vrkAB) ;

	    __vr vrkCD = _ve_vshf_vvvs(vrkC, vrkD, VE_VSHUFFLE_YUZU) ;
	    vrsumCD_y0x0 = _ve_pvfmad_vvvv(vrsumCD_y0x0, vriP_h0w0, vrkCD) ;
	    vrsumCD_y0x1 = _ve_pvfmad_vvvv(vrsumCD_y0x1, vriP_h0w1, vrkCD) ;

	    __vr vrkEF = _ve_vshf_vvvs(vrkE, vrkF, VE_VSHUFFLE_YUZU) ;
	    vrsumEF_y0x0 = _ve_pvfmad_vvvv(vrsumEF_y0x0, vriP_h0w0, vrkEF) ;
	    vrsumEF_y0x1 = _ve_pvfmad_vvvv(vrsumEF_y0x1, vriP_h0w1, vrkEF) ;
	  }
	  else if( w0_valid ) {
	    int64_t inputIndex_h0w0  = inGroupOffset + ((n * inChannel + c) * inHeight + h0) * inWidth + w0;

	    int64_t kernelIndex = kernGroupOffset + ((k * inChannelGroup + c) * kernHeight + r) * kernWidth + s;

	    __vr vri_h0w0 = _ve_vldu_vss(4*inHeight*inWidth, &pIn[inputIndex_h0w0]) ;

	    __vr vrk0 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+0*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk1 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+1*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk2 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+2*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk3 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+3*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk4 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+4*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk5 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+5*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk6 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+6*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk7 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+7*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk8 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+8*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk9 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+9*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrkA = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+10*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrkB = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+11*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrkC = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+12*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrkD = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+13*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrkE = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+14*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrkF = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+15*inChannelGroup*kernHeight*kernWidth]) ;

	    __vr vriP_h0w0  = _ve_vshf_vvvs(vri_h0w0, vri_h0w0, VE_VSHUFFLE_YUZU) ;

	    __vr vrk01 = _ve_vshf_vvvs(vrk0, vrk1, VE_VSHUFFLE_YUZU) ;
	    vrsum01_y0x0 = _ve_pvfmad_vvvv(vrsum01_y0x0, vriP_h0w0, vrk01) ;

	    __vr vrk23 = _ve_vshf_vvvs(vrk2, vrk3, VE_VSHUFFLE_YUZU) ;
	    vrsum23_y0x0 = _ve_pvfmad_vvvv(vrsum23_y0x0, vriP_h0w0, vrk23) ;

	    __vr vrk45 = _ve_vshf_vvvs(vrk4, vrk5, VE_VSHUFFLE_YUZU) ;
	    vrsum45_y0x0 = _ve_pvfmad_vvvv(vrsum45_y0x0, vriP_h0w0, vrk45) ;

	    __vr vrk67 = _ve_vshf_vvvs(vrk6, vrk7, VE_VSHUFFLE_YUZU) ;
	    vrsum67_y0x0 = _ve_pvfmad_vvvv(vrsum67_y0x0, vriP_h0w0, vrk67) ;

	    __vr vrk89 = _ve_vshf_vvvs(vrk8, vrk9, VE_VSHUFFLE_YUZU) ;
	    vrsum89_y0x0 = _ve_pvfmad_vvvv(vrsum89_y0x0, vriP_h0w0, vrk89) ;

	    __vr vrkAB = _ve_vshf_vvvs(vrkA, vrkB, VE_VSHUFFLE_YUZU) ;
	    vrsumAB_y0x0 = _ve_pvfmad_vvvv(vrsumAB_y0x0, vriP_h0w0, vrkAB) ;

	    __vr vrkCD = _ve_vshf_vvvs(vrkC, vrkD, VE_VSHUFFLE_YUZU) ;
	    vrsumCD_y0x0 = _ve_pvfmad_vvvv(vrsumCD_y0x0, vriP_h0w0, vrkCD) ;

	    __vr vrkEF = _ve_vshf_vvvs(vrkE, vrkF, VE_VSHUFFLE_YUZU) ;
	    vrsumEF_y0x0 = _ve_pvfmad_vvvv(vrsumEF_y0x0, vriP_h0w0, vrkEF) ;
	  }
	  else if( w1_valid ) {
	    int64_t inputIndex_h0w1  = inGroupOffset + ((n * inChannel + c) * inHeight + h0) * inWidth + w1;

	    int64_t kernelIndex = kernGroupOffset + ((k * inChannelGroup + c) * kernHeight + r) * kernWidth + s;

	    __vr vri_h0w1 = _ve_vldu_vss(4*inHeight*inWidth, &pIn[inputIndex_h0w1]) ;

	    __vr vrk0 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+0*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk1 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+1*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk2 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+2*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk3 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+3*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk4 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+4*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk5 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+5*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk6 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+6*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk7 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+7*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk8 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+8*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk9 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+9*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrkA = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+10*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrkB = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+11*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrkC = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+12*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrkD = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+13*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrkE = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+14*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrkF = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+15*inChannelGroup*kernHeight*kernWidth]) ;

	    __vr vriP_h0w1  = _ve_vshf_vvvs(vri_h0w1, vri_h0w1, VE_VSHUFFLE_YUZU) ;

	    __vr vrk01 = _ve_vshf_vvvs(vrk0, vrk1, VE_VSHUFFLE_YUZU) ;
	    vrsum01_y0x1 = _ve_pvfmad_vvvv(vrsum01_y0x1, vriP_h0w1, vrk01) ;

	    __vr vrk23 = _ve_vshf_vvvs(vrk2, vrk3, VE_VSHUFFLE_YUZU) ;
	    vrsum23_y0x1 = _ve_pvfmad_vvvv(vrsum23_y0x1, vriP_h0w1, vrk23) ;

	    __vr vrk45 = _ve_vshf_vvvs(vrk4, vrk5, VE_VSHUFFLE_YUZU) ;
	    vrsum45_y0x1 = _ve_pvfmad_vvvv(vrsum45_y0x1, vriP_h0w1, vrk45) ;

	    __vr vrk67 = _ve_vshf_vvvs(vrk6, vrk7, VE_VSHUFFLE_YUZU) ;
	    vrsum67_y0x1 = _ve_pvfmad_vvvv(vrsum67_y0x1, vriP_h0w1, vrk67) ;

	    __vr vrk89 = _ve_vshf_vvvs(vrk8, vrk9, VE_VSHUFFLE_YUZU) ;
	    vrsum89_y0x1 = _ve_pvfmad_vvvv(vrsum89_y0x1, vriP_h0w1, vrk89) ;

	    __vr vrkAB = _ve_vshf_vvvs(vrkA, vrkB, VE_VSHUFFLE_YUZU) ;
	    vrsumAB_y0x1 = _ve_pvfmad_vvvv(vrsumAB_y0x1, vriP_h0w1, vrkAB) ;

	    __vr vrkCD = _ve_vshf_vvvs(vrkC, vrkD, VE_VSHUFFLE_YUZU) ;
	    vrsumCD_y0x1 = _ve_pvfmad_vvvv(vrsumCD_y0x1, vriP_h0w1, vrkCD) ;

	    __vr vrkEF = _ve_vshf_vvvs(vrkE, vrkF, VE_VSHUFFLE_YUZU) ;
	    vrsumEF_y0x1 = _ve_pvfmad_vvvv(vrsumEF_y0x1, vriP_h0w1, vrkEF) ;
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

	    __vr vri_h1w0 = _ve_vldu_vss(4*inHeight*inWidth, &pIn[inputIndex_h1w0]) ;
	    __vr vri_h1w1 = _ve_vldu_vss(4*inHeight*inWidth, &pIn[inputIndex_h1w1]) ;

	    __vr vrk0 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+0*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk1 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+1*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk2 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+2*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk3 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+3*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk4 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+4*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk5 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+5*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk6 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+6*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk7 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+7*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk8 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+8*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk9 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+9*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrkA = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+10*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrkB = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+11*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrkC = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+12*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrkD = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+13*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrkE = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+14*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrkF = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+15*inChannelGroup*kernHeight*kernWidth]) ;

	    __vr vriP_h1w0  = _ve_vshf_vvvs(vri_h1w0, vri_h1w0, VE_VSHUFFLE_YUZU) ;
	    __vr vriP_h1w1  = _ve_vshf_vvvs(vri_h1w1, vri_h1w1, VE_VSHUFFLE_YUZU) ;

	    __vr vrk01 = _ve_vshf_vvvs(vrk0, vrk1, VE_VSHUFFLE_YUZU) ;
	    vrsum01_y1x0 = _ve_pvfmad_vvvv(vrsum01_y1x0, vriP_h1w0, vrk01) ;
	    vrsum01_y1x1 = _ve_pvfmad_vvvv(vrsum01_y1x1, vriP_h1w1, vrk01) ;

	    __vr vrk23 = _ve_vshf_vvvs(vrk2, vrk3, VE_VSHUFFLE_YUZU) ;
	    vrsum23_y1x0 = _ve_pvfmad_vvvv(vrsum23_y1x0, vriP_h1w0, vrk23) ;
	    vrsum23_y1x1 = _ve_pvfmad_vvvv(vrsum23_y1x1, vriP_h1w1, vrk23) ;

	    __vr vrk45 = _ve_vshf_vvvs(vrk4, vrk5, VE_VSHUFFLE_YUZU) ;
	    vrsum45_y1x0 = _ve_pvfmad_vvvv(vrsum45_y1x0, vriP_h1w0, vrk45) ;
	    vrsum45_y1x1 = _ve_pvfmad_vvvv(vrsum45_y1x1, vriP_h1w1, vrk45) ;

	    __vr vrk67 = _ve_vshf_vvvs(vrk6, vrk7, VE_VSHUFFLE_YUZU) ;
	    vrsum67_y1x0 = _ve_pvfmad_vvvv(vrsum67_y1x0, vriP_h1w0, vrk67) ;
	    vrsum67_y1x1 = _ve_pvfmad_vvvv(vrsum67_y1x1, vriP_h1w1, vrk67) ;

	    __vr vrk89 = _ve_vshf_vvvs(vrk8, vrk9, VE_VSHUFFLE_YUZU) ;
	    vrsum89_y1x0 = _ve_pvfmad_vvvv(vrsum89_y1x0, vriP_h1w0, vrk89) ;
	    vrsum89_y1x1 = _ve_pvfmad_vvvv(vrsum89_y1x1, vriP_h1w1, vrk89) ;

	    __vr vrkAB = _ve_vshf_vvvs(vrkA, vrkB, VE_VSHUFFLE_YUZU) ;
	    vrsumAB_y1x0 = _ve_pvfmad_vvvv(vrsumAB_y1x0, vriP_h1w0, vrkAB) ;
	    vrsumAB_y1x1 = _ve_pvfmad_vvvv(vrsumAB_y1x1, vriP_h1w1, vrkAB) ;

	    __vr vrkCD = _ve_vshf_vvvs(vrkC, vrkD, VE_VSHUFFLE_YUZU) ;
	    vrsumCD_y1x0 = _ve_pvfmad_vvvv(vrsumCD_y1x0, vriP_h1w0, vrkCD) ;
	    vrsumCD_y1x1 = _ve_pvfmad_vvvv(vrsumCD_y1x1, vriP_h1w1, vrkCD) ;

	    __vr vrkEF = _ve_vshf_vvvs(vrkE, vrkF, VE_VSHUFFLE_YUZU) ;
	    vrsumEF_y1x0 = _ve_pvfmad_vvvv(vrsumEF_y1x0, vriP_h1w0, vrkEF) ;
	    vrsumEF_y1x1 = _ve_pvfmad_vvvv(vrsumEF_y1x1, vriP_h1w1, vrkEF) ;
	  }
	  else if( w0_valid ) {
	    int64_t inputIndex_h1w0  = inGroupOffset + ((n * inChannel + c) * inHeight + h1) * inWidth + w0;

	    int64_t kernelIndex = kernGroupOffset + ((k * inChannelGroup + c) * kernHeight + r) * kernWidth + s;

	    __vr vri_h1w0 = _ve_vldu_vss(4*inHeight*inWidth, &pIn[inputIndex_h1w0]) ;

	    __vr vrk0 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+0*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk1 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+1*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk2 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+2*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk3 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+3*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk4 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+4*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk5 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+5*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk6 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+6*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk7 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+7*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk8 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+8*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk9 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+9*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrkA = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+10*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrkB = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+11*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrkC = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+12*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrkD = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+13*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrkE = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+14*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrkF = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+15*inChannelGroup*kernHeight*kernWidth]) ;

	    __vr vriP_h1w0  = _ve_vshf_vvvs(vri_h1w0, vri_h1w0, VE_VSHUFFLE_YUZU) ;

	    __vr vrk01 = _ve_vshf_vvvs(vrk0, vrk1, VE_VSHUFFLE_YUZU) ;
	    vrsum01_y1x0 = _ve_pvfmad_vvvv(vrsum01_y1x0, vriP_h1w0, vrk01) ;

	    __vr vrk23 = _ve_vshf_vvvs(vrk2, vrk3, VE_VSHUFFLE_YUZU) ;
	    vrsum23_y1x0 = _ve_pvfmad_vvvv(vrsum23_y1x0, vriP_h1w0, vrk23) ;

	    __vr vrk45 = _ve_vshf_vvvs(vrk4, vrk5, VE_VSHUFFLE_YUZU) ;
	    vrsum45_y1x0 = _ve_pvfmad_vvvv(vrsum45_y1x0, vriP_h1w0, vrk45) ;

	    __vr vrk67 = _ve_vshf_vvvs(vrk6, vrk7, VE_VSHUFFLE_YUZU) ;
	    vrsum67_y1x0 = _ve_pvfmad_vvvv(vrsum67_y1x0, vriP_h1w0, vrk67) ;

	    __vr vrk89 = _ve_vshf_vvvs(vrk8, vrk9, VE_VSHUFFLE_YUZU) ;
	    vrsum89_y1x0 = _ve_pvfmad_vvvv(vrsum89_y1x0, vriP_h1w0, vrk89) ;

	    __vr vrkAB = _ve_vshf_vvvs(vrkA, vrkB, VE_VSHUFFLE_YUZU) ;
	    vrsumAB_y1x0 = _ve_pvfmad_vvvv(vrsumAB_y1x0, vriP_h1w0, vrkAB) ;

	    __vr vrkCD = _ve_vshf_vvvs(vrkC, vrkD, VE_VSHUFFLE_YUZU) ;
	    vrsumCD_y1x0 = _ve_pvfmad_vvvv(vrsumCD_y1x0, vriP_h1w0, vrkCD) ;

	    __vr vrkEF = _ve_vshf_vvvs(vrkE, vrkF, VE_VSHUFFLE_YUZU) ;
	    vrsumEF_y1x0 = _ve_pvfmad_vvvv(vrsumEF_y1x0, vriP_h1w0, vrkEF) ;
	  }
	  else if( w1_valid ) {
	    int64_t inputIndex_h1w1  = inGroupOffset + ((n * inChannel + c) * inHeight + h1) * inWidth + w1;

	    int64_t kernelIndex = kernGroupOffset + ((k * inChannelGroup + c) * kernHeight + r) * kernWidth + s;

	    __vr vri_h1w1 = _ve_vldu_vss(4*inHeight*inWidth, &pIn[inputIndex_h1w1]) ;

	    __vr vrk0 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+0*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk1 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+1*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk2 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+2*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk3 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+3*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk4 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+4*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk5 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+5*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk6 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+6*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk7 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+7*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk8 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+8*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk9 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+9*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrkA = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+10*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrkB = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+11*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrkC = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+12*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrkD = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+13*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrkE = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+14*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrkF = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+15*inChannelGroup*kernHeight*kernWidth]) ;

	    __vr vriP_h1w1  = _ve_vshf_vvvs(vri_h1w1, vri_h1w1, VE_VSHUFFLE_YUZU) ;

	    __vr vrk01 = _ve_vshf_vvvs(vrk0, vrk1, VE_VSHUFFLE_YUZU) ;
	    vrsum01_y1x1 = _ve_pvfmad_vvvv(vrsum01_y1x1, vriP_h1w1, vrk01) ;

	    __vr vrk23 = _ve_vshf_vvvs(vrk2, vrk3, VE_VSHUFFLE_YUZU) ;
	    vrsum23_y1x1 = _ve_pvfmad_vvvv(vrsum23_y1x1, vriP_h1w1, vrk23) ;

	    __vr vrk45 = _ve_vshf_vvvs(vrk4, vrk5, VE_VSHUFFLE_YUZU) ;
	    vrsum45_y1x1 = _ve_pvfmad_vvvv(vrsum45_y1x1, vriP_h1w1, vrk45) ;

	    __vr vrk67 = _ve_vshf_vvvs(vrk6, vrk7, VE_VSHUFFLE_YUZU) ;
	    vrsum67_y1x1 = _ve_pvfmad_vvvv(vrsum67_y1x1, vriP_h1w1, vrk67) ;

	    __vr vrk89 = _ve_vshf_vvvs(vrk8, vrk9, VE_VSHUFFLE_YUZU) ;
	    vrsum89_y1x1 = _ve_pvfmad_vvvv(vrsum89_y1x1, vriP_h1w1, vrk89) ;

	    __vr vrkAB = _ve_vshf_vvvs(vrkA, vrkB, VE_VSHUFFLE_YUZU) ;
	    vrsumAB_y1x1 = _ve_pvfmad_vvvv(vrsumAB_y1x1, vriP_h1w1, vrkAB) ;

	    __vr vrkCD = _ve_vshf_vvvs(vrkC, vrkD, VE_VSHUFFLE_YUZU) ;
	    vrsumCD_y1x1 = _ve_pvfmad_vvvv(vrsumCD_y1x1, vriP_h1w1, vrkCD) ;

	    __vr vrkEF = _ve_vshf_vvvs(vrkE, vrkF, VE_VSHUFFLE_YUZU) ;
	    vrsumEF_y1x1 = _ve_pvfmad_vvvv(vrsumEF_y1x1, vriP_h1w1, vrkEF) ;
	  }
	} // kernWidth
      }
    } // kernHeight
  } // inChannel

  {
    _ve_lvl(VLEN) ;
    __vr vrsum0_y0x0 = _ve_vfsums_vv(vrsum01_y0x0) ;
    __vr vrsum1_y0x0 = _ve_vfsums_vv(_ve_vsll_vvs(vrsum01_y0x0,32)) ;
    __vr vrsum2_y0x0 = _ve_vfsums_vv(vrsum23_y0x0) ;
    __vr vrsum3_y0x0 = _ve_vfsums_vv(_ve_vsll_vvs(vrsum23_y0x0,32)) ;
    __vr vrsum4_y0x0 = _ve_vfsums_vv(vrsum45_y0x0) ;
    __vr vrsum5_y0x0 = _ve_vfsums_vv(_ve_vsll_vvs(vrsum45_y0x0,32)) ;
    __vr vrsum6_y0x0 = _ve_vfsums_vv(vrsum67_y0x0) ;
    __vr vrsum7_y0x0 = _ve_vfsums_vv(_ve_vsll_vvs(vrsum67_y0x0,32)) ;
    __vr vrsum8_y0x0 = _ve_vfsums_vv(vrsum89_y0x0) ;
    __vr vrsum9_y0x0 = _ve_vfsums_vv(_ve_vsll_vvs(vrsum89_y0x0,32)) ;
    __vr vrsumA_y0x0 = _ve_vfsums_vv(vrsumAB_y0x0) ;
    __vr vrsumB_y0x0 = _ve_vfsums_vv(_ve_vsll_vvs(vrsumAB_y0x0,32)) ;
    __vr vrsumC_y0x0 = _ve_vfsums_vv(vrsumCD_y0x0) ;
    __vr vrsumD_y0x0 = _ve_vfsums_vv(_ve_vsll_vvs(vrsumCD_y0x0,32)) ;
    __vr vrsumE_y0x0 = _ve_vfsums_vv(vrsumEF_y0x0) ;
    __vr vrsumF_y0x0 = _ve_vfsums_vv(_ve_vsll_vvs(vrsumEF_y0x0,32)) ;

    _ve_lvl(1) ;
    _ve_vstu_vss(vrsum0_y0x0, 4, &pOut[outIndex+0*outHeight*outWidth]) ;
    _ve_vstu_vss(vrsum1_y0x0, 4, &pOut[outIndex+1*outHeight*outWidth]) ;
    _ve_vstu_vss(vrsum2_y0x0, 4, &pOut[outIndex+2*outHeight*outWidth]) ;
    _ve_vstu_vss(vrsum3_y0x0, 4, &pOut[outIndex+3*outHeight*outWidth]) ;
    _ve_vstu_vss(vrsum4_y0x0, 4, &pOut[outIndex+4*outHeight*outWidth]) ;
    _ve_vstu_vss(vrsum5_y0x0, 4, &pOut[outIndex+5*outHeight*outWidth]) ;
    _ve_vstu_vss(vrsum6_y0x0, 4, &pOut[outIndex+6*outHeight*outWidth]) ;
    _ve_vstu_vss(vrsum7_y0x0, 4, &pOut[outIndex+7*outHeight*outWidth]) ;
    _ve_vstu_vss(vrsum8_y0x0, 4, &pOut[outIndex+8*outHeight*outWidth]) ;
    _ve_vstu_vss(vrsum9_y0x0, 4, &pOut[outIndex+9*outHeight*outWidth]) ;
    _ve_vstu_vss(vrsumA_y0x0, 4, &pOut[outIndex+10*outHeight*outWidth]) ;
    _ve_vstu_vss(vrsumB_y0x0, 4, &pOut[outIndex+11*outHeight*outWidth]) ;
    _ve_vstu_vss(vrsumC_y0x0, 4, &pOut[outIndex+12*outHeight*outWidth]) ;
    _ve_vstu_vss(vrsumD_y0x0, 4, &pOut[outIndex+13*outHeight*outWidth]) ;
    _ve_vstu_vss(vrsumE_y0x0, 4, &pOut[outIndex+14*outHeight*outWidth]) ;
    _ve_vstu_vss(vrsumF_y0x0, 4, &pOut[outIndex+15*outHeight*outWidth]) ;
  }
  {
    _ve_lvl(VLEN) ;
    __vr vrsum0_y0x1 = _ve_vfsums_vv(vrsum01_y0x1) ;
    __vr vrsum1_y0x1 = _ve_vfsums_vv(_ve_vsll_vvs(vrsum01_y0x1,32)) ;
    __vr vrsum2_y0x1 = _ve_vfsums_vv(vrsum23_y0x1) ;
    __vr vrsum3_y0x1 = _ve_vfsums_vv(_ve_vsll_vvs(vrsum23_y0x1,32)) ;
    __vr vrsum4_y0x1 = _ve_vfsums_vv(vrsum45_y0x1) ;
    __vr vrsum5_y0x1 = _ve_vfsums_vv(_ve_vsll_vvs(vrsum45_y0x1,32)) ;
    __vr vrsum6_y0x1 = _ve_vfsums_vv(vrsum67_y0x1) ;
    __vr vrsum7_y0x1 = _ve_vfsums_vv(_ve_vsll_vvs(vrsum67_y0x1,32)) ;
    __vr vrsum8_y0x1 = _ve_vfsums_vv(vrsum89_y0x1) ;
    __vr vrsum9_y0x1 = _ve_vfsums_vv(_ve_vsll_vvs(vrsum89_y0x1,32)) ;
    __vr vrsumA_y0x1 = _ve_vfsums_vv(vrsumAB_y0x1) ;
    __vr vrsumB_y0x1 = _ve_vfsums_vv(_ve_vsll_vvs(vrsumAB_y0x1,32)) ;
    __vr vrsumC_y0x1 = _ve_vfsums_vv(vrsumCD_y0x1) ;
    __vr vrsumD_y0x1 = _ve_vfsums_vv(_ve_vsll_vvs(vrsumCD_y0x1,32)) ;
    __vr vrsumE_y0x1 = _ve_vfsums_vv(vrsumEF_y0x1) ;
    __vr vrsumF_y0x1 = _ve_vfsums_vv(_ve_vsll_vvs(vrsumEF_y0x1,32)) ;

    _ve_lvl(1) ;
    _ve_vstu_vss(vrsum0_y0x1, 4, &pOut[outIndex+0*outHeight*outWidth+1]) ;
    _ve_vstu_vss(vrsum1_y0x1, 4, &pOut[outIndex+1*outHeight*outWidth+1]) ;
    _ve_vstu_vss(vrsum2_y0x1, 4, &pOut[outIndex+2*outHeight*outWidth+1]) ;
    _ve_vstu_vss(vrsum3_y0x1, 4, &pOut[outIndex+3*outHeight*outWidth+1]) ;
    _ve_vstu_vss(vrsum4_y0x1, 4, &pOut[outIndex+4*outHeight*outWidth+1]) ;
    _ve_vstu_vss(vrsum5_y0x1, 4, &pOut[outIndex+5*outHeight*outWidth+1]) ;
    _ve_vstu_vss(vrsum6_y0x1, 4, &pOut[outIndex+6*outHeight*outWidth+1]) ;
    _ve_vstu_vss(vrsum7_y0x1, 4, &pOut[outIndex+7*outHeight*outWidth+1]) ;
    _ve_vstu_vss(vrsum8_y0x1, 4, &pOut[outIndex+8*outHeight*outWidth+1]) ;
    _ve_vstu_vss(vrsum9_y0x1, 4, &pOut[outIndex+9*outHeight*outWidth+1]) ;
    _ve_vstu_vss(vrsumA_y0x1, 4, &pOut[outIndex+10*outHeight*outWidth+1]) ;
    _ve_vstu_vss(vrsumB_y0x1, 4, &pOut[outIndex+11*outHeight*outWidth+1]) ;
    _ve_vstu_vss(vrsumC_y0x1, 4, &pOut[outIndex+12*outHeight*outWidth+1]) ;
    _ve_vstu_vss(vrsumD_y0x1, 4, &pOut[outIndex+13*outHeight*outWidth+1]) ;
    _ve_vstu_vss(vrsumE_y0x1, 4, &pOut[outIndex+14*outHeight*outWidth+1]) ;
    _ve_vstu_vss(vrsumF_y0x1, 4, &pOut[outIndex+15*outHeight*outWidth+1]) ;
  }
  {
    _ve_lvl(VLEN) ;
    __vr vrsum0_y1x0 = _ve_vfsums_vv(vrsum01_y1x0) ;
    __vr vrsum1_y1x0 = _ve_vfsums_vv(_ve_vsll_vvs(vrsum01_y1x0,32)) ;
    __vr vrsum2_y1x0 = _ve_vfsums_vv(vrsum23_y1x0) ;
    __vr vrsum3_y1x0 = _ve_vfsums_vv(_ve_vsll_vvs(vrsum23_y1x0,32)) ;
    __vr vrsum4_y1x0 = _ve_vfsums_vv(vrsum45_y1x0) ;
    __vr vrsum5_y1x0 = _ve_vfsums_vv(_ve_vsll_vvs(vrsum45_y1x0,32)) ;
    __vr vrsum6_y1x0 = _ve_vfsums_vv(vrsum67_y1x0) ;
    __vr vrsum7_y1x0 = _ve_vfsums_vv(_ve_vsll_vvs(vrsum67_y1x0,32)) ;
    __vr vrsum8_y1x0 = _ve_vfsums_vv(vrsum89_y1x0) ;
    __vr vrsum9_y1x0 = _ve_vfsums_vv(_ve_vsll_vvs(vrsum89_y1x0,32)) ;
    __vr vrsumA_y1x0 = _ve_vfsums_vv(vrsumAB_y1x0) ;
    __vr vrsumB_y1x0 = _ve_vfsums_vv(_ve_vsll_vvs(vrsumAB_y1x0,32)) ;
    __vr vrsumC_y1x0 = _ve_vfsums_vv(vrsumCD_y1x0) ;
    __vr vrsumD_y1x0 = _ve_vfsums_vv(_ve_vsll_vvs(vrsumCD_y1x0,32)) ;
    __vr vrsumE_y1x0 = _ve_vfsums_vv(vrsumEF_y1x0) ;
    __vr vrsumF_y1x0 = _ve_vfsums_vv(_ve_vsll_vvs(vrsumEF_y1x0,32)) ;

    _ve_lvl(1) ;
    _ve_vstu_vss(vrsum0_y1x0, 4, &pOut[outIndex+0*outHeight*outWidth+outWidth]) ;
    _ve_vstu_vss(vrsum1_y1x0, 4, &pOut[outIndex+1*outHeight*outWidth+outWidth]) ;
    _ve_vstu_vss(vrsum2_y1x0, 4, &pOut[outIndex+2*outHeight*outWidth+outWidth]) ;
    _ve_vstu_vss(vrsum3_y1x0, 4, &pOut[outIndex+3*outHeight*outWidth+outWidth]) ;
    _ve_vstu_vss(vrsum4_y1x0, 4, &pOut[outIndex+4*outHeight*outWidth+outWidth]) ;
    _ve_vstu_vss(vrsum5_y1x0, 4, &pOut[outIndex+5*outHeight*outWidth+outWidth]) ;
    _ve_vstu_vss(vrsum6_y1x0, 4, &pOut[outIndex+6*outHeight*outWidth+outWidth]) ;
    _ve_vstu_vss(vrsum7_y1x0, 4, &pOut[outIndex+7*outHeight*outWidth+outWidth]) ;
    _ve_vstu_vss(vrsum8_y1x0, 4, &pOut[outIndex+8*outHeight*outWidth+outWidth]) ;
    _ve_vstu_vss(vrsum9_y1x0, 4, &pOut[outIndex+9*outHeight*outWidth+outWidth]) ;
    _ve_vstu_vss(vrsumA_y1x0, 4, &pOut[outIndex+10*outHeight*outWidth+outWidth]) ;
    _ve_vstu_vss(vrsumB_y1x0, 4, &pOut[outIndex+11*outHeight*outWidth+outWidth]) ;
    _ve_vstu_vss(vrsumC_y1x0, 4, &pOut[outIndex+12*outHeight*outWidth+outWidth]) ;
    _ve_vstu_vss(vrsumD_y1x0, 4, &pOut[outIndex+13*outHeight*outWidth+outWidth]) ;
    _ve_vstu_vss(vrsumE_y1x0, 4, &pOut[outIndex+14*outHeight*outWidth+outWidth]) ;
    _ve_vstu_vss(vrsumF_y1x0, 4, &pOut[outIndex+15*outHeight*outWidth+outWidth]) ;
  }
  {
    _ve_lvl(VLEN) ;
    __vr vrsum0_y1x1 = _ve_vfsums_vv(vrsum01_y1x1) ;
    __vr vrsum1_y1x1 = _ve_vfsums_vv(_ve_vsll_vvs(vrsum01_y1x1,32)) ;
    __vr vrsum2_y1x1 = _ve_vfsums_vv(vrsum23_y1x1) ;
    __vr vrsum3_y1x1 = _ve_vfsums_vv(_ve_vsll_vvs(vrsum23_y1x1,32)) ;
    __vr vrsum4_y1x1 = _ve_vfsums_vv(vrsum45_y1x1) ;
    __vr vrsum5_y1x1 = _ve_vfsums_vv(_ve_vsll_vvs(vrsum45_y1x1,32)) ;
    __vr vrsum6_y1x1 = _ve_vfsums_vv(vrsum67_y1x1) ;
    __vr vrsum7_y1x1 = _ve_vfsums_vv(_ve_vsll_vvs(vrsum67_y1x1,32)) ;
    __vr vrsum8_y1x1 = _ve_vfsums_vv(vrsum89_y1x1) ;
    __vr vrsum9_y1x1 = _ve_vfsums_vv(_ve_vsll_vvs(vrsum89_y1x1,32)) ;
    __vr vrsumA_y1x1 = _ve_vfsums_vv(vrsumAB_y1x1) ;
    __vr vrsumB_y1x1 = _ve_vfsums_vv(_ve_vsll_vvs(vrsumAB_y1x1,32)) ;
    __vr vrsumC_y1x1 = _ve_vfsums_vv(vrsumCD_y1x1) ;
    __vr vrsumD_y1x1 = _ve_vfsums_vv(_ve_vsll_vvs(vrsumCD_y1x1,32)) ;
    __vr vrsumE_y1x1 = _ve_vfsums_vv(vrsumEF_y1x1) ;
    __vr vrsumF_y1x1 = _ve_vfsums_vv(_ve_vsll_vvs(vrsumEF_y1x1,32)) ;

    _ve_lvl(1) ;
    _ve_vstu_vss(vrsum0_y1x1, 4, &pOut[outIndex+0*outHeight*outWidth+outWidth+1]) ;
    _ve_vstu_vss(vrsum1_y1x1, 4, &pOut[outIndex+1*outHeight*outWidth+outWidth+1]) ;
    _ve_vstu_vss(vrsum2_y1x1, 4, &pOut[outIndex+2*outHeight*outWidth+outWidth+1]) ;
    _ve_vstu_vss(vrsum3_y1x1, 4, &pOut[outIndex+3*outHeight*outWidth+outWidth+1]) ;
    _ve_vstu_vss(vrsum4_y1x1, 4, &pOut[outIndex+4*outHeight*outWidth+outWidth+1]) ;
    _ve_vstu_vss(vrsum5_y1x1, 4, &pOut[outIndex+5*outHeight*outWidth+outWidth+1]) ;
    _ve_vstu_vss(vrsum6_y1x1, 4, &pOut[outIndex+6*outHeight*outWidth+outWidth+1]) ;
    _ve_vstu_vss(vrsum7_y1x1, 4, &pOut[outIndex+7*outHeight*outWidth+outWidth+1]) ;
    _ve_vstu_vss(vrsum8_y1x1, 4, &pOut[outIndex+8*outHeight*outWidth+outWidth+1]) ;
    _ve_vstu_vss(vrsum9_y1x1, 4, &pOut[outIndex+9*outHeight*outWidth+outWidth+1]) ;
    _ve_vstu_vss(vrsumA_y1x1, 4, &pOut[outIndex+10*outHeight*outWidth+outWidth+1]) ;
    _ve_vstu_vss(vrsumB_y1x1, 4, &pOut[outIndex+11*outHeight*outWidth+outWidth+1]) ;
    _ve_vstu_vss(vrsumC_y1x1, 4, &pOut[outIndex+12*outHeight*outWidth+outWidth+1]) ;
    _ve_vstu_vss(vrsumD_y1x1, 4, &pOut[outIndex+13*outHeight*outWidth+outWidth+1]) ;
    _ve_vstu_vss(vrsumE_y1x1, 4, &pOut[outIndex+14*outHeight*outWidth+outWidth+1]) ;
    _ve_vstu_vss(vrsumF_y1x1, 4, &pOut[outIndex+15*outHeight*outWidth+outWidth+1]) ;
  }
}


vednnError_t
vednnConvolutionForward_direct_vecC(
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
  const int64_t padWidth       = pParamConv->padWidth;
  const int64_t padHeight      = pParamConv->padHeight;
  const int64_t dilationWidth  = pParamConv->dilationWidth;
  const int64_t dilationHeight = pParamConv->dilationHeight;

  const int64_t inChannelGroup  = inChannel  / group;   // equal to pDataKernel->inChannel
  const int64_t outChannelGroup = outChannel / group;   // equal to pDataKernel->outChannel

  const float * restrict pIn     = pDataIn;
  const float * restrict pKernel = pDataKernel;
  float * restrict const pOut    = pDataOut;

  for (int64_t n=0; n<batch; n++) {
    for (int64_t g = 0; g < group; g++) {
      int64_t inGroupOffset   = g * inChannelGroup  * inHeight  * inWidth;
      int64_t outGroupOffset  = g * outChannelGroup * outHeight * outWidth;
      int64_t kernGroupOffset = g * outChannelGroup * inChannelGroup * kernHeight * kernWidth;

      int64_t k=0 ;
      if( (outChannelGroup & 0x01) == 1 ) {
	k1(pIn, pKernel, pOut,
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
	k2(pIn, pKernel, pOut,
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
	k4(pIn, pKernel, pOut,
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
	    k8y1x1(pIn, pKernel, pOut,
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
	    k8y1x2(pIn, pKernel, pOut,
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
	    k8y2x1(pIn, pKernel, pOut,
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
	    k8y2x2(pIn, pKernel, pOut,
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
	    k16y1x1(pIn, pKernel, pOut,
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
	    k16y1x2(pIn, pKernel, pOut,
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
	    k16y2x1(pIn, pKernel, pOut,
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
	    k16y2x2(pIn, pKernel, pOut,
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

  return VEDNN_SUCCESS;
}
