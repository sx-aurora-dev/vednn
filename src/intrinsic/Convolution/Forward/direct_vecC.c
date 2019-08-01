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
  for (int64_t p=0; p<outHeight; p++) {
    int64_t i = p * strideHeight - padHeight;
    for (int64_t q=0; q<outWidth; q++) {
      int64_t j = q * strideWidth - padWidth;
      int64_t outIndex  = outGroupOffset + ((n * outChannel + k) * outHeight + p) * outWidth + q;

      _ve_lvl(VLEN) ;
      __vr vrsum = _ve_vbrdu_vs_f32(0.f) ;

      for (int64_t c=0; c<inChannelGroup; c+= VLEN) {
	const int64_t vl = inChannelGroup - c < VLEN ? inChannelGroup - c : VLEN ;
	_ve_lvl(vl) ;

	for (int64_t h=0; h<kernHeight; h++) {
	  for (int64_t w=0; w<kernWidth; w++) {
	    int64_t y = i + h * dilationHeight;
	    int64_t x = j + w * dilationWidth;
	    if (y < 0 || inHeight <= y) {
	      continue;
	    }
	    if (x < 0 || inWidth <= x) {
	      continue;
	    }
	    int64_t inputIndex  = inGroupOffset + ((n * inChannel + c) * inHeight + y) * inWidth + x;
	    int64_t kernelIndex = kernGroupOffset + ((k * inChannelGroup + c) * kernHeight + h) * kernWidth + w;

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
  for (int64_t p=0; p<outHeight; p++) {
    int64_t i = p * strideHeight - padHeight;
    for (int64_t q=0; q<outWidth; q++) {
      int64_t j = q * strideWidth - padWidth;
      int64_t outIndex  = outGroupOffset + ((n * outChannel + k) * outHeight + p) * outWidth + q;

      _ve_lvl(VLEN) ;
      __vr vrsum01 = _ve_vbrd_vs_i64(0UL) ;

      for (int64_t c=0; c<inChannelGroup; c+= VLEN) {
	const int64_t vl = inChannelGroup - c < VLEN ? inChannelGroup - c : VLEN ;
	_ve_lvl(vl) ;

	for (int64_t h=0; h<kernHeight; h++) {
	  for (int64_t w=0; w<kernWidth; w++) {
	    int64_t y = i + h * dilationHeight;
	    int64_t x = j + w * dilationWidth;
	    if (y < 0 || inHeight <= y) {
	      continue;
	    }
	    if (x < 0 || inWidth <= x) {
	      continue;
	    }
	    int64_t inputIndex  = inGroupOffset + ((n * inChannel + c) * inHeight + y) * inWidth + x;
	    int64_t kernelIndex = kernGroupOffset + ((k * inChannelGroup + c) * kernHeight + h) * kernWidth + w;

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
  for (int64_t p=0; p<outHeight; p++) {
    int64_t i = p * strideHeight - padHeight;
    for (int64_t q=0; q<outWidth; q++) {
      int64_t j = q * strideWidth - padWidth;
      int64_t outIndex  = outGroupOffset + ((n * outChannel + k) * outHeight + p) * outWidth + q;

      _ve_lvl(VLEN) ;
      __vr vrsum01 = _ve_vbrd_vs_i64(0UL) ;
      __vr vrsum23 = _ve_vbrd_vs_i64(0UL) ;

      for (int64_t c=0; c<inChannelGroup; c+= VLEN) {
	const int64_t vl = inChannelGroup - c < VLEN ? inChannelGroup - c : VLEN ;
	_ve_lvl(vl) ;

	for (int64_t h=0; h<kernHeight; h++) {
	  for (int64_t w=0; w<kernWidth; w++) {
	    int64_t y = i + h * dilationHeight;
	    int64_t x = j + w * dilationWidth;
	    if (y < 0 || inHeight <= y) {
	      continue;
	    }
	    if (x < 0 || inWidth <= x) {
	      continue;
	    }
	    int64_t inputIndex  = inGroupOffset + ((n * inChannel + c) * inHeight + y) * inWidth + x;
	    int64_t kernelIndex = kernGroupOffset + ((k * inChannelGroup + c) * kernHeight + h) * kernWidth + w;

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
  for (int64_t p=0; p<outHeight; p++) {
    int64_t i = p * strideHeight - padHeight;
    for (int64_t q=0; q<outWidth; q++) {
      int64_t j = q * strideWidth - padWidth;
      int64_t outIndex  = outGroupOffset + ((n * outChannel + k) * outHeight + p) * outWidth + q;

      _ve_lvl(VLEN) ;
      __vr vrsum01 = _ve_vbrd_vs_i64(0UL) ;
      __vr vrsum23 = _ve_vbrd_vs_i64(0UL) ;
      __vr vrsum45 = _ve_vbrd_vs_i64(0UL) ;
      __vr vrsum67 = _ve_vbrd_vs_i64(0UL) ;

      for (int64_t c=0; c<inChannelGroup; c+= VLEN) {
	const int64_t vl = inChannelGroup - c < VLEN ? inChannelGroup - c : VLEN ;
	_ve_lvl(vl) ;

	for (int64_t h=0; h<kernHeight; h++) {
	  for (int64_t w=0; w<kernWidth; w++) {
	    int64_t y = i + h * dilationHeight;
	    int64_t x = j + w * dilationWidth;
	    if (y < 0 || inHeight <= y) {
	      continue;
	    }
	    if (x < 0 || inWidth <= x) {
	      continue;
	    }
	    int64_t inputIndex  = inGroupOffset + ((n * inChannel + c) * inHeight + y) * inWidth + x;
	    int64_t kernelIndex = kernGroupOffset + ((k * inChannelGroup + c) * kernHeight + h) * kernWidth + w;

	    __vr vri = _ve_vldu_vss(4*inHeight*inWidth, &pIn[inputIndex]) ;

	    __vr vrk0 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+0*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk1 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+1*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk2 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+2*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk3 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+3*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk4 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+4*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk5 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+5*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk6 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+6*inChannelGroup*kernHeight*kernWidth]) ;
	    __vr vrk7 = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex+7*inChannelGroup*kernHeight*kernWidth]) ;

	    __vr vriP  = _ve_vshf_vvvs(vri, vri, VE_VSHUFFLE_YUZU) ;

	    __vr vrk01 = _ve_vshf_vvvs(vrk0, vrk1, VE_VSHUFFLE_YUZU) ;
	    vrsum01 = _ve_pvfmad_vvvv(vrsum01, vriP, vrk01) ;
	    __vr vrk23 = _ve_vshf_vvvs(vrk2, vrk3, VE_VSHUFFLE_YUZU) ;
	    vrsum23 = _ve_pvfmad_vvvv(vrsum23, vriP, vrk23) ;
	    __vr vrk45 = _ve_vshf_vvvs(vrk4, vrk5, VE_VSHUFFLE_YUZU) ;
	    vrsum45 = _ve_pvfmad_vvvv(vrsum45, vriP, vrk45) ;
	    __vr vrk67 = _ve_vshf_vvvs(vrk6, vrk7, VE_VSHUFFLE_YUZU) ;
	    vrsum67 = _ve_pvfmad_vvvv(vrsum67, vriP, vrk67) ;
	  } // kernWidth
	} // kernHeight
      } // inChannel

      _ve_lvl(VLEN) ;
      __vr vrsum0 = _ve_vfsums_vv(vrsum01) ;
      __vr vrsum1 = _ve_vfsums_vv(_ve_vsll_vvs(vrsum01,32)) ;
      __vr vrsum2 = _ve_vfsums_vv(vrsum23) ;
      __vr vrsum3 = _ve_vfsums_vv(_ve_vsll_vvs(vrsum23,32)) ;
      __vr vrsum4 = _ve_vfsums_vv(vrsum45) ;
      __vr vrsum5 = _ve_vfsums_vv(_ve_vsll_vvs(vrsum45,32)) ;
      __vr vrsum6 = _ve_vfsums_vv(vrsum67) ;
      __vr vrsum7 = _ve_vfsums_vv(_ve_vsll_vvs(vrsum67,32)) ;

      _ve_lvl(1) ;
      _ve_vstu_vss(vrsum0, 4, &pOut[outIndex+0*outHeight*outWidth]) ;
      _ve_vstu_vss(vrsum1, 4, &pOut[outIndex+1*outHeight*outWidth]) ;
      _ve_vstu_vss(vrsum2, 4, &pOut[outIndex+2*outHeight*outWidth]) ;
      _ve_vstu_vss(vrsum3, 4, &pOut[outIndex+3*outHeight*outWidth]) ;
      _ve_vstu_vss(vrsum4, 4, &pOut[outIndex+4*outHeight*outWidth]) ;
      _ve_vstu_vss(vrsum5, 4, &pOut[outIndex+5*outHeight*outWidth]) ;
      _ve_vstu_vss(vrsum6, 4, &pOut[outIndex+6*outHeight*outWidth]) ;
      _ve_vstu_vss(vrsum7, 4, &pOut[outIndex+7*outHeight*outWidth]) ;

    } // outWidth
  } // outHeight
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
  for (int64_t p=0; p<outHeight; p++) {
    int64_t i = p * strideHeight - padHeight;
    for (int64_t q=0; q<outWidth; q++) {
      int64_t j = q * strideWidth - padWidth;
      int64_t outIndex  = outGroupOffset + ((n * outChannel + k) * outHeight + p) * outWidth + q;

      _ve_lvl(VLEN) ;
      __vr vrsum01 = _ve_vbrd_vs_i64(0UL) ;
      __vr vrsum23 = _ve_vbrd_vs_i64(0UL) ;
      __vr vrsum45 = _ve_vbrd_vs_i64(0UL) ;
      __vr vrsum67 = _ve_vbrd_vs_i64(0UL) ;
      __vr vrsum89 = _ve_vbrd_vs_i64(0UL) ;
      __vr vrsumAB = _ve_vbrd_vs_i64(0UL) ;
      __vr vrsumCD = _ve_vbrd_vs_i64(0UL) ;
      __vr vrsumEF = _ve_vbrd_vs_i64(0UL) ;

      for (int64_t c=0; c<inChannelGroup; c+= VLEN) {
	const int64_t vl = inChannelGroup - c < VLEN ? inChannelGroup - c : VLEN ;
	_ve_lvl(vl) ;

	for (int64_t h=0; h<kernHeight; h++) {
	  for (int64_t w=0; w<kernWidth; w++) {
	    int64_t y = i + h * dilationHeight;
	    int64_t x = j + w * dilationWidth;
	    if (y < 0 || inHeight <= y) {
	      continue;
	    }
	    if (x < 0 || inWidth <= x) {
	      continue;
	    }
	    int64_t inputIndex  = inGroupOffset + ((n * inChannel + c) * inHeight + y) * inWidth + x;
	    int64_t kernelIndex = kernGroupOffset + ((k * inChannelGroup + c) * kernHeight + h) * kernWidth + w;

	    __vr vri = _ve_vldu_vss(4*inHeight*inWidth, &pIn[inputIndex]) ;

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


	    __vr vriP  = _ve_vshf_vvvs(vri, vri, VE_VSHUFFLE_YUZU) ;

	    __vr vrk01 = _ve_vshf_vvvs(vrk0, vrk1, VE_VSHUFFLE_YUZU) ;
	    vrsum01 = _ve_pvfmad_vvvv(vrsum01, vriP, vrk01) ;
	    __vr vrk23 = _ve_vshf_vvvs(vrk2, vrk3, VE_VSHUFFLE_YUZU) ;
	    vrsum23 = _ve_pvfmad_vvvv(vrsum23, vriP, vrk23) ;
	    __vr vrk45 = _ve_vshf_vvvs(vrk4, vrk5, VE_VSHUFFLE_YUZU) ;
	    vrsum45 = _ve_pvfmad_vvvv(vrsum45, vriP, vrk45) ;
	    __vr vrk67 = _ve_vshf_vvvs(vrk6, vrk7, VE_VSHUFFLE_YUZU) ;
	    vrsum67 = _ve_pvfmad_vvvv(vrsum67, vriP, vrk67) ;
	    __vr vrk89 = _ve_vshf_vvvs(vrk8, vrk9, VE_VSHUFFLE_YUZU) ;
	    vrsum89 = _ve_pvfmad_vvvv(vrsum89, vriP, vrk89) ;
	    __vr vrkAB = _ve_vshf_vvvs(vrkA, vrkB, VE_VSHUFFLE_YUZU) ;
	    vrsumAB = _ve_pvfmad_vvvv(vrsumAB, vriP, vrkAB) ;
	    __vr vrkCD = _ve_vshf_vvvs(vrkC, vrkD, VE_VSHUFFLE_YUZU) ;
	    vrsumCD = _ve_pvfmad_vvvv(vrsumCD, vriP, vrkCD) ;
	    __vr vrkEF = _ve_vshf_vvvs(vrkE, vrkF, VE_VSHUFFLE_YUZU) ;
	    vrsumEF = _ve_pvfmad_vvvv(vrsumEF, vriP, vrkEF) ;
	  } // kernWidth
	} // kernHeight
      } // inChannel

      _ve_lvl(VLEN) ;
      __vr vrsum0 = _ve_vfsums_vv(vrsum01) ;
      __vr vrsum1 = _ve_vfsums_vv(_ve_vsll_vvs(vrsum01,32)) ;
      __vr vrsum2 = _ve_vfsums_vv(vrsum23) ;
      __vr vrsum3 = _ve_vfsums_vv(_ve_vsll_vvs(vrsum23,32)) ;
      __vr vrsum4 = _ve_vfsums_vv(vrsum45) ;
      __vr vrsum5 = _ve_vfsums_vv(_ve_vsll_vvs(vrsum45,32)) ;
      __vr vrsum6 = _ve_vfsums_vv(vrsum67) ;
      __vr vrsum7 = _ve_vfsums_vv(_ve_vsll_vvs(vrsum67,32)) ;
      __vr vrsum8 = _ve_vfsums_vv(vrsum89) ;
      __vr vrsum9 = _ve_vfsums_vv(_ve_vsll_vvs(vrsum89,32)) ;
      __vr vrsumA = _ve_vfsums_vv(vrsumAB) ;
      __vr vrsumB = _ve_vfsums_vv(_ve_vsll_vvs(vrsumAB,32)) ;
      __vr vrsumC = _ve_vfsums_vv(vrsumCD) ;
      __vr vrsumD = _ve_vfsums_vv(_ve_vsll_vvs(vrsumCD,32)) ;
      __vr vrsumE = _ve_vfsums_vv(vrsumEF) ;
      __vr vrsumF = _ve_vfsums_vv(_ve_vsll_vvs(vrsumEF,32)) ;

      _ve_lvl(1) ;
      _ve_vstu_vss(vrsum0, 4, &pOut[outIndex+0*outHeight*outWidth]) ;
      _ve_vstu_vss(vrsum1, 4, &pOut[outIndex+1*outHeight*outWidth]) ;
      _ve_vstu_vss(vrsum2, 4, &pOut[outIndex+2*outHeight*outWidth]) ;
      _ve_vstu_vss(vrsum3, 4, &pOut[outIndex+3*outHeight*outWidth]) ;
      _ve_vstu_vss(vrsum4, 4, &pOut[outIndex+4*outHeight*outWidth]) ;
      _ve_vstu_vss(vrsum5, 4, &pOut[outIndex+5*outHeight*outWidth]) ;
      _ve_vstu_vss(vrsum6, 4, &pOut[outIndex+6*outHeight*outWidth]) ;
      _ve_vstu_vss(vrsum7, 4, &pOut[outIndex+7*outHeight*outWidth]) ;
      _ve_vstu_vss(vrsum8, 4, &pOut[outIndex+8*outHeight*outWidth]) ;
      _ve_vstu_vss(vrsum9, 4, &pOut[outIndex+9*outHeight*outWidth]) ;
      _ve_vstu_vss(vrsumA, 4, &pOut[outIndex+10*outHeight*outWidth]) ;
      _ve_vstu_vss(vrsumB, 4, &pOut[outIndex+11*outHeight*outWidth]) ;
      _ve_vstu_vss(vrsumC, 4, &pOut[outIndex+12*outHeight*outWidth]) ;
      _ve_vstu_vss(vrsumD, 4, &pOut[outIndex+13*outHeight*outWidth]) ;
      _ve_vstu_vss(vrsumE, 4, &pOut[outIndex+14*outHeight*outWidth]) ;
      _ve_vstu_vss(vrsumF, 4, &pOut[outIndex+15*outHeight*outWidth]) ;


    } // outWidth
  } // outHeight
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
	k8(pIn, pKernel, pOut,
	   inChannel, inWidth, inHeight,
	   outChannel, outWidth, outHeight,
	   kernWidth, kernHeight,
	   inChannelGroup, outChannelGroup,
	   strideHeight, strideWidth,
	   padHeight, padWidth,
	   dilationHeight, dilationWidth,
	   inGroupOffset, outGroupOffset, kernGroupOffset,
	   n, k) ;
	k+=8 ;
      }
      for (; k<outChannelGroup; ) {
	k16(pIn, pKernel, pOut,
	   inChannel, inWidth, inHeight,
	   outChannel, outWidth, outHeight,
	   kernWidth, kernHeight,
	   inChannelGroup, outChannelGroup,
	   strideHeight, strideWidth,
	   padHeight, padWidth,
	   dilationHeight, dilationWidth,
	   inGroupOffset, outGroupOffset, kernGroupOffset,
	   n, k) ;
	k+=16 ;
      } // outChannel
    } // group
  } // batch

  return VEDNN_SUCCESS;
}
