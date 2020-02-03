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

  const int64_t remain  = NUMKERNEL & 0x1 ;
  const int64_t nPacked = NUMKERNEL >> 1 ;

  float bias[NUMKERNEL] ;
#pragma clang loop unroll(full)
  for(int64_t kk=0; kk<NUMKERNEL; kk++) {
    bias[kk] = pBias[biasGroupOffset+k+kk] ;
  }


  __vr vrsum0_y0x0 = _vel_vbrds_vsl(0.f, VLEN) ;
  __vr vrsum0_y0x1 = _vel_vbrds_vsl(0.f, VLEN) ;
  __vr vrsum0_y1x0 = _vel_vbrds_vsl(0.f, VLEN) ;
  __vr vrsum0_y1x1 = _vel_vbrds_vsl(0.f, VLEN) ;

  __vr vrsum_y0x0[nPacked] ;
  __vr vrsum_y0x1[nPacked] ;
  __vr vrsum_y1x0[nPacked] ;
  __vr vrsum_y1x1[nPacked] ;
#pragma clang loop unroll(full)
  for(int64_t kk=0; kk<nPacked; kk++) {
    vrsum_y0x0[kk] = _vel_vbrdl_vsl(0UL, VLEN) ;
    vrsum_y0x1[kk] = _vel_vbrdl_vsl(0UL, VLEN) ;
    vrsum_y1x0[kk] = _vel_vbrdl_vsl(0UL, VLEN) ;
    vrsum_y1x1[kk] = _vel_vbrdl_vsl(0UL, VLEN) ;
  }

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

	    __vr vrk[NUMKERNEL] ;
#pragma clang loop unroll(full)
	    for(int64_t kk=0; kk<NUMKERNEL; kk++) {
	      vrk[kk] = _vel_vldu_vssl(4*kernelStride, pKerValue+kk*kernelDistance, vl) ;
	    }

	    __vr vriP_h0w0  = _vel_vshf_vvvsl(vri_h0w0, vri_h0w0, VE_VSHUFFLE_YUZU, vl) ;
	    __vr vriP_h0w1  = _vel_vshf_vvvsl(vri_h0w1, vri_h0w1, VE_VSHUFFLE_YUZU, vl) ;
	    __vr vriP_h1w0  = _vel_vshf_vvvsl(vri_h1w0, vri_h1w0, VE_VSHUFFLE_YUZU, vl) ;
	    __vr vriP_h1w1  = _vel_vshf_vvvsl(vri_h1w1, vri_h1w1, VE_VSHUFFLE_YUZU, vl) ;

	    if( remain ) {
	      vrsum0_y0x0 = _vel_vfmads_vvvvvl(vrsum0_y0x0, vri_h0w0, vrk[0], vrsum0_y0x0, vl) ;
	      vrsum0_y0x1 = _vel_vfmads_vvvvvl(vrsum0_y0x1, vri_h0w1, vrk[0], vrsum0_y0x1, vl) ;
	      vrsum0_y1x0 = _vel_vfmads_vvvvvl(vrsum0_y1x0, vri_h1w0, vrk[0], vrsum0_y1x0, vl) ;
	      vrsum0_y1x1 = _vel_vfmads_vvvvvl(vrsum0_y1x1, vri_h1w1, vrk[0], vrsum0_y1x1, vl) ;
	    }

#pragma clang loop unroll(full)
	    for(int64_t kk=0; kk<nPacked; kk++) {
	      __vr vrkp = _vel_vshf_vvvsl(vrk[2*kk+remain], vrk[2*kk+remain+1], VE_VSHUFFLE_YUZU, vl) ;
	      vrsum_y0x0[kk] = _vel_pvfmad_vvvvvl(vrsum_y0x0[kk], vriP_h0w0, vrkp, vrsum_y0x0[kk], vl) ;
	      vrsum_y0x1[kk] = _vel_pvfmad_vvvvvl(vrsum_y0x1[kk], vriP_h0w1, vrkp, vrsum_y0x1[kk], vl) ;
	      vrsum_y1x0[kk] = _vel_pvfmad_vvvvvl(vrsum_y1x0[kk], vriP_h1w0, vrkp, vrsum_y1x0[kk], vl) ;
	      vrsum_y1x1[kk] = _vel_pvfmad_vvvvvl(vrsum_y1x1[kk], vriP_h1w1, vrkp, vrsum_y1x1[kk], vl) ;
	    }
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

	    __vr vrk[NUMKERNEL] ;
#pragma clang loop unroll(full)
	    for(int64_t kk=0; kk<NUMKERNEL; kk++) {
	      vrk[kk] = _vel_vldu_vssl(4*kernelStride, pKerValue+kk*kernelDistance, vl) ;
	    }

	    __vr vriP_h0w0  = _vel_vshf_vvvsl(vri_h0w0, vri_h0w0, VE_VSHUFFLE_YUZU, vl) ;
	    __vr vriP_h1w0  = _vel_vshf_vvvsl(vri_h1w0, vri_h1w0, VE_VSHUFFLE_YUZU, vl) ;

	    if( remain ) {
	      vrsum0_y0x0 = _vel_vfmads_vvvvvl(vrsum0_y0x0, vri_h0w0, vrk[0], vrsum0_y0x0, vl) ;
	      vrsum0_y1x0 = _vel_vfmads_vvvvvl(vrsum0_y1x0, vri_h1w0, vrk[0], vrsum0_y1x0, vl) ;
	    }

#pragma clang loop unroll(full)
	    for(int64_t kk=0; kk<nPacked; kk++) {
	      __vr vrkp = _vel_vshf_vvvsl(vrk[2*kk+remain], vrk[2*kk+remain+1], VE_VSHUFFLE_YUZU, vl) ;
	      vrsum_y0x0[kk] = _vel_pvfmad_vvvvvl(vrsum_y0x0[kk], vriP_h0w0, vrkp, vrsum_y0x0[kk], vl) ;
	      vrsum_y1x0[kk] = _vel_pvfmad_vvvvvl(vrsum_y1x0[kk], vriP_h1w0, vrkp, vrsum_y1x0[kk], vl) ;
	    }
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

	    __vr vrk[NUMKERNEL] ;
#pragma clang loop unroll(full)
	    for(int64_t kk=0; kk<NUMKERNEL; kk++) {
	      vrk[kk] = _vel_vldu_vssl(4*kernelStride, pKerValue+kk*kernelDistance, vl) ;
	    }

	    __vr vriP_h0w1  = _vel_vshf_vvvsl(vri_h0w1, vri_h0w1, VE_VSHUFFLE_YUZU, vl) ;
	    __vr vriP_h1w1  = _vel_vshf_vvvsl(vri_h1w1, vri_h1w1, VE_VSHUFFLE_YUZU, vl) ;

	    if( remain ) {
	      vrsum0_y0x1 = _vel_vfmads_vvvvvl(vrsum0_y0x1, vri_h0w1, vrk[0], vrsum0_y0x1, vl) ;
	      vrsum0_y1x1 = _vel_vfmads_vvvvvl(vrsum0_y1x1, vri_h1w1, vrk[0], vrsum0_y1x1, vl) ;
	    }

#pragma clang loop unroll(full)
	    for(int64_t kk=0; kk<nPacked; kk++) {
	      __vr vrkp = _vel_vshf_vvvsl(vrk[2*kk+remain], vrk[2*kk+remain+1], VE_VSHUFFLE_YUZU, vl) ;
	      vrsum_y0x1[kk] = _vel_pvfmad_vvvvvl(vrsum_y0x1[kk], vriP_h0w1, vrkp, vrsum_y0x1[kk], vl) ;
	      vrsum_y1x1[kk] = _vel_pvfmad_vvvvvl(vrsum_y1x1[kk], vriP_h1w1, vrkp, vrsum_y1x1[kk], vl) ;
	    }
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
	    __vr vrk[NUMKERNEL] ;
#pragma clang loop unroll(full)
	    for(int64_t kk=0; kk<NUMKERNEL; kk++) {
	      vrk[kk] = _vel_vldu_vssl(4*kernelStride, pKerValue+kk*kernelDistance, vl) ;
	    }

	    __vr vriP_h0w0  = _vel_vshf_vvvsl(vri_h0w0, vri_h0w0, VE_VSHUFFLE_YUZU, vl) ;
	    __vr vriP_h0w1  = _vel_vshf_vvvsl(vri_h0w1, vri_h0w1, VE_VSHUFFLE_YUZU, vl) ;

	    if( remain ) {
	      vrsum0_y0x0 = _vel_vfmads_vvvvvl(vrsum0_y0x0, vri_h0w0, vrk[0], vrsum0_y0x0, vl) ;
	      vrsum0_y0x1 = _vel_vfmads_vvvvvl(vrsum0_y0x1, vri_h0w1, vrk[0], vrsum0_y0x1, vl) ;
	    }

#pragma clang loop unroll(full)
	    for(int64_t kk=0; kk<nPacked; kk++) {
	      __vr vrkp = _vel_vshf_vvvsl(vrk[2*kk+remain], vrk[2*kk+remain+1], VE_VSHUFFLE_YUZU, vl) ;
	      vrsum_y0x0[kk] = _vel_pvfmad_vvvvvl(vrsum_y0x0[kk], vriP_h0w0, vrkp, vrsum_y0x0[kk], vl) ;
	      vrsum_y0x1[kk] = _vel_pvfmad_vvvvvl(vrsum_y0x1[kk], vriP_h0w1, vrkp, vrsum_y0x1[kk], vl) ;
	    }
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

	    __vr vrk[NUMKERNEL] ;
#pragma clang loop unroll(full)
	    for(int64_t kk=0; kk<NUMKERNEL; kk++) {
	      vrk[kk] = _vel_vldu_vssl(4*kernelStride, pKerValue+kk*kernelDistance, vl) ;
	    }

	    __vr vriP_h0w0  = _vel_vshf_vvvsl(vri_h0w0, vri_h0w0, VE_VSHUFFLE_YUZU, vl) ;

	    if( remain ) {
	      vrsum0_y0x0 = _vel_vfmads_vvvvvl(vrsum0_y0x0, vri_h0w0, vrk[0], vrsum0_y0x0, vl) ;
	    }

#pragma clang loop unroll(full)
	    for(int64_t kk=0; kk<nPacked; kk++) {
	      __vr vrkp = _vel_vshf_vvvsl(vrk[2*kk+remain], vrk[2*kk+remain+1], VE_VSHUFFLE_YUZU, vl) ;
	      vrsum_y0x0[kk] = _vel_pvfmad_vvvvvl(vrsum_y0x0[kk], vriP_h0w0, vrkp, vrsum_y0x0[kk], vl) ;
	    }
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

	    __vr vrk[NUMKERNEL] ;
#pragma clang loop unroll(full)
	    for(int64_t kk=0; kk<NUMKERNEL; kk++) {
	      vrk[kk] = _vel_vldu_vssl(4*kernelStride, pKerValue+kk*kernelDistance, vl) ;
	    }

	    __vr vriP_h0w1  = _vel_vshf_vvvsl(vri_h0w1, vri_h0w1, VE_VSHUFFLE_YUZU, vl) ;

	    if( remain ) {
	      vrsum0_y0x1 = _vel_vfmads_vvvvvl(vrsum0_y0x1, vri_h0w1, vrk[0], vrsum0_y0x1, vl) ;
	    }

#pragma clang loop unroll(full)
	    for(int64_t kk=0; kk<nPacked; kk++) {
	      __vr vrkp = _vel_vshf_vvvsl(vrk[2*kk+remain], vrk[2*kk+remain+1], VE_VSHUFFLE_YUZU, vl) ;
	      vrsum_y0x1[kk] = _vel_pvfmad_vvvvvl(vrsum_y0x1[kk], vriP_h0w1, vrkp, vrsum_y0x1[kk], vl) ;
	    }
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

	    __vr vrk[NUMKERNEL] ;
#pragma clang loop unroll(full)
	    for(int64_t kk=0; kk<NUMKERNEL; kk++) {
	      vrk[kk] = _vel_vldu_vssl(4*kernelStride, pKerValue+kk*kernelDistance, vl) ;
	    }

	    __vr vriP_h1w0  = _vel_vshf_vvvsl(vri_h1w0, vri_h1w0, VE_VSHUFFLE_YUZU, vl) ;
	    __vr vriP_h1w1  = _vel_vshf_vvvsl(vri_h1w1, vri_h1w1, VE_VSHUFFLE_YUZU, vl) ;

	    if( remain ) {
	      vrsum0_y1x0 = _vel_vfmads_vvvvvl(vrsum0_y1x0, vri_h1w0, vrk[0], vrsum0_y1x0, vl) ;
	      vrsum0_y1x1 = _vel_vfmads_vvvvvl(vrsum0_y1x1, vri_h1w1, vrk[0], vrsum0_y1x1, vl) ;
	    }

#pragma clang loop unroll(full)
	    for(int64_t kk=0; kk<nPacked; kk++) {
	      __vr vrkp = _vel_vshf_vvvsl(vrk[2*kk+remain], vrk[2*kk+remain+1], VE_VSHUFFLE_YUZU, vl) ;
	      vrsum_y1x0[kk] = _vel_pvfmad_vvvvvl(vrsum_y1x0[kk], vriP_h1w0, vrkp, vrsum_y1x0[kk], vl) ;
	      vrsum_y1x1[kk] = _vel_pvfmad_vvvvvl(vrsum_y1x1[kk], vriP_h1w1, vrkp, vrsum_y1x1[kk], vl) ;
	    }
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

	    __vr vrk[NUMKERNEL] ;
#pragma clang loop unroll(full)
	    for(int64_t kk=0; kk<NUMKERNEL; kk++) {
	      vrk[kk] = _vel_vldu_vssl(4*kernelStride, pKerValue+kk*kernelDistance, vl) ;
	    }

	    __vr vriP_h1w0  = _vel_vshf_vvvsl(vri_h1w0, vri_h1w0, VE_VSHUFFLE_YUZU, vl) ;

	    if( remain ) {
	      vrsum0_y1x0 = _vel_vfmads_vvvvvl(vrsum0_y1x0, vri_h1w0, vrk[0], vrsum0_y1x0, vl) ;
	    }

#pragma clang loop unroll(full)
	    for(int64_t kk=0; kk<nPacked; kk++) {
	      __vr vrkp = _vel_vshf_vvvsl(vrk[2*kk+remain], vrk[2*kk+remain+1], VE_VSHUFFLE_YUZU, vl) ;
	      vrsum_y1x0[kk] = _vel_pvfmad_vvvvvl(vrsum_y1x0[kk], vriP_h1w0, vrkp, vrsum_y1x0[kk], vl) ;
	    }
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

	    __vr vrk[NUMKERNEL] ;
#pragma clang loop unroll(full)
	    for(int64_t kk=0; kk<NUMKERNEL; kk++) {
	      vrk[kk] = _vel_vldu_vssl(4*kernelStride, pKerValue+kk*kernelDistance, vl) ;
	    }

	    __vr vriP_h1w1  = _vel_vshf_vvvsl(vri_h1w1, vri_h1w1, VE_VSHUFFLE_YUZU, vl) ;

	    if( remain ) {
	      vrsum0_y1x1 = _vel_vfmads_vvvvvl(vrsum0_y1x1, vri_h1w1, vrk[0], vrsum0_y1x1, vl) ;
	    }

#pragma clang loop unroll(full)
	    for(int64_t kk=0; kk<nPacked; kk++) {
	      __vr vrkp = _vel_vshf_vvvsl(vrk[2*kk+remain], vrk[2*kk+remain+1], VE_VSHUFFLE_YUZU, vl) ;
	      vrsum_y1x1[kk] = _vel_pvfmad_vvvvvl(vrsum_y1x1[kk], vriP_h1w1, vrkp, vrsum_y1x1[kk], vl) ;
	    }
	  }
	} // kernWidth
      }
    } // kernHeight
  } // inChannel

  if(Y>=1 && X>=1) {
    if( remain ) {
      vrsum0_y0x0 = _vel_vfsums_vvl(vrsum0_y0x0, VLEN) ;
      if(ADDBIAS) vrsum0_y0x0 = _vel_vfadds_vsvl(bias[0], vrsum0_y0x0, 1) ;
      _vel_vstu_vssl(vrsum0_y0x0, 4, &pOut[outIndex+0*outHeight*outWidth], 1) ;
    }

#pragma clang loop unroll(full)
    for(int64_t kk=0; kk<nPacked; kk++) {
      __vr vrsumU_y0x0 = _vel_vfsums_vvl(vrsum_y0x0[kk], VLEN) ;
      if(ADDBIAS) vrsumU_y0x0 = _vel_vfadds_vsvl(bias[2*kk+remain], vrsumU_y0x0, 1) ;
      _vel_vstu_vssl(vrsumU_y0x0, 4, &pOut[outIndex+(2*kk+remain)*outHeight*outWidth], 1) ;
      __vr vrsumL_y0x0 = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsum_y0x0[kk],32, VLEN), VLEN) ;
      if(ADDBIAS) vrsumL_y0x0 = _vel_vfadds_vsvl(bias[2*kk+remain+1], vrsumL_y0x0, 1) ;
      _vel_vstu_vssl(vrsumL_y0x0, 4, &pOut[outIndex+(2*kk+remain+1)*outHeight*outWidth], 1) ;
    }
  }
  if(Y>=1 && X>=2) {
    if( remain ) {
      vrsum0_y0x1 = _vel_vfsums_vvl(vrsum0_y0x1, VLEN) ;
      if(ADDBIAS) vrsum0_y0x1 = _vel_vfadds_vsvl(bias[0], vrsum0_y0x1, 1) ;
      _vel_vstu_vssl(vrsum0_y0x1, 4, &pOut[outIndex+0*outHeight*outWidth+1], 1) ;
    }

#pragma clang loop unroll(full)
    for(int64_t kk=0; kk<nPacked; kk++) {
      __vr vrsumU_y0x1 = _vel_vfsums_vvl(vrsum_y0x1[kk], VLEN) ;
      if(ADDBIAS) vrsumU_y0x1 = _vel_vfadds_vsvl(bias[2*kk+remain], vrsumU_y0x1, 1) ;
      _vel_vstu_vssl(vrsumU_y0x1, 4, &pOut[outIndex+(2*kk+remain)*outHeight*outWidth+1], 1) ;
      __vr vrsumL_y0x1 = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsum_y0x1[kk],32, VLEN), VLEN) ;
      if(ADDBIAS) vrsumL_y0x1 = _vel_vfadds_vsvl(bias[2*kk+remain+1], vrsumL_y0x1, 1) ;
      _vel_vstu_vssl(vrsumL_y0x1, 4, &pOut[outIndex+(2*kk+remain+1)*outHeight*outWidth+1], 1) ;
    }
  }
  if(Y>=2 && X>=1) {
    if( remain ) {
      vrsum0_y1x0 = _vel_vfsums_vvl(vrsum0_y1x0, VLEN) ;
      if(ADDBIAS) vrsum0_y1x0 = _vel_vfadds_vsvl(bias[0], vrsum0_y1x0, 1) ;
      _vel_vstu_vssl(vrsum0_y1x0, 4, &pOut[outIndex+0*outHeight*outWidth+outWidth], 1) ;
    }

#pragma clang loop unroll(full)
    for(int64_t kk=0; kk<nPacked; kk++) {
      __vr vrsumU_y1x0 = _vel_vfsums_vvl(vrsum_y1x0[kk], VLEN) ;
      if(ADDBIAS) vrsumU_y1x0 = _vel_vfadds_vsvl(bias[2*kk+remain], vrsumU_y1x0, 1) ;
      _vel_vstu_vssl(vrsumU_y1x0, 4, &pOut[outIndex+(2*kk+remain)*outHeight*outWidth+outWidth], 1) ;
      __vr vrsumL_y1x0 = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsum_y1x0[kk],32, VLEN), VLEN) ;
      if(ADDBIAS) vrsumL_y1x0 = _vel_vfadds_vsvl(bias[2*kk+remain+1], vrsumL_y1x0, 1) ;
      _vel_vstu_vssl(vrsumL_y1x0, 4, &pOut[outIndex+(2*kk+remain+1)*outHeight*outWidth+outWidth], 1) ;
    }
  }
  if(Y>=2 && X>=2) {
    if( remain ) {
      vrsum0_y1x1 = _vel_vfsums_vvl(vrsum0_y1x1, VLEN) ;
      if(ADDBIAS) vrsum0_y1x1 = _vel_vfadds_vsvl(bias[0], vrsum0_y1x1, 1) ;
      _vel_vstu_vssl(vrsum0_y1x1, 4, &pOut[outIndex+0*outHeight*outWidth+outWidth+1], 1) ;
    }

#pragma clang loop unroll(full)
    for(int64_t kk=0; kk<nPacked; kk++) {
      __vr vrsumU_y1x1 = _vel_vfsums_vvl(vrsum_y1x1[kk], VLEN) ;
      if(ADDBIAS) vrsumU_y1x1 = _vel_vfadds_vsvl(bias[2*kk+remain], vrsumU_y1x1, 1) ;
      _vel_vstu_vssl(vrsumU_y1x1, 4, &pOut[outIndex+(2*kk+remain)*outHeight*outWidth+outWidth+1], 1) ;
      __vr vrsumL_y1x1 = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsum_y1x1[kk],32, VLEN), VLEN) ;
      if(ADDBIAS) vrsumL_y1x1 = _vel_vfadds_vsvl(bias[2*kk+remain+1], vrsumL_y1x1, 1) ;
      _vel_vstu_vssl(vrsumL_y1x1, 4, &pOut[outIndex+(2*kk+remain+1)*outHeight*outWidth+outWidth+1], 1) ;
    }
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
