#include <stdint.h>
#include <stdlib.h>

#include "vednn.h"

#include "veintrin.h"
#define VLEN	(256)

static inline void h1w1(
    const float * restrict pGOut,
    const float * restrict pKernel,
    float * restrict const pGIn,
    const int64_t gOutChannel,
    const int64_t gOutWidth,
    const int64_t gOutHeight,
    const int64_t gInChannel,
    const int64_t gInWidth,
    const int64_t gInHeight,
    const int64_t kernWidth,
    const int64_t kernHeight,
    const int64_t gInChannelGroup,
    const int64_t gOutChannelGroup,
    const int64_t strideHeight,
    const int64_t strideWidth,
    const int64_t padHeight,
    const int64_t padWidth,
    const int64_t dilationHeight,
    const int64_t dilationWidth,
    const int64_t gInGroupOffset,
    const int64_t gOutGroupOffset,
    const int64_t kernGroupOffset,
    const int64_t n,
    const int64_t h,
    const int64_t w
)
{

  for (int64_t c=0; c<gInChannelGroup; c+=VLEN) {
    int64_t gInIndex = gInGroupOffset + ((n * gInChannel + c) * gInHeight + h) * gInWidth + w;

    const int64_t vl = gInChannelGroup - c < VLEN ? gInChannelGroup - c : VLEN ;

    _ve_lvl(vl) ;
    __vr vrsum = _ve_vbrdu_vs_f32(0.f) ;

    for (int64_t r=0; r<kernHeight; r++) {
      int64_t i = h - r * dilationHeight + padHeight ;
      int64_t y = i/strideHeight;
      if ( y*strideHeight != i || y < 0 || gOutHeight <= y)  continue ;

      for (int64_t s=0; s<kernWidth; s++) {
	int64_t j = w - s * dilationWidth  + padWidth ;
	int64_t x = j/strideWidth ;
	if (x*strideWidth !=j || x < 0 || gOutWidth <= x) continue;

	for (int64_t k=0; k<gOutChannelGroup; k++) {
	  int64_t gOutIndex   = gOutGroupOffset + ((n * gOutChannel + k) * gOutHeight + y) * gOutWidth + x;
	  int64_t kernelIndex = kernGroupOffset + ((k * gInChannelGroup + c) * kernHeight + r) * kernWidth + s;

	  __vr vrk = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex]) ;

	  vrsum = _ve_vfmads_vvsv(vrsum, pGOut[gOutIndex], vrk) ;
	} // gOutChannel

      } // kernWidth
    } // kernHeight

    _ve_vstu_vss(vrsum, 4*gInHeight*gInWidth, &pGIn[gInIndex]) ;
  } // gInChannel
}

static inline void h1w2(
    const float * restrict pGOut,
    const float * restrict pKernel,
    float * restrict const pGIn,
    const int64_t gOutChannel,
    const int64_t gOutWidth,
    const int64_t gOutHeight,
    const int64_t gInChannel,
    const int64_t gInWidth,
    const int64_t gInHeight,
    const int64_t kernWidth,
    const int64_t kernHeight,
    const int64_t gInChannelGroup,
    const int64_t gOutChannelGroup,
    const int64_t strideHeight,
    const int64_t strideWidth,
    const int64_t padHeight,
    const int64_t padWidth,
    const int64_t dilationHeight,
    const int64_t dilationWidth,
    const int64_t gInGroupOffset,
    const int64_t gOutGroupOffset,
    const int64_t kernGroupOffset,
    const int64_t n,
    const int64_t h,
    const int64_t w
)
{

  for (int64_t c=0; c<gInChannelGroup; c+=VLEN) {
    int64_t gInIndex = gInGroupOffset + ((n * gInChannel + c) * gInHeight + h) * gInWidth + w;

    const int64_t vl = gInChannelGroup - c < VLEN ? gInChannelGroup - c : VLEN ;

    _ve_lvl(vl) ;
    __vr vrsum_w0 = _ve_vbrdu_vs_f32(0.f) ;
    __vr vrsum_w1 = _ve_vbrdu_vs_f32(0.f) ;

    for (int64_t r=0; r<kernHeight; r++) {
      int64_t i = h - r * dilationHeight + padHeight ;
      int64_t y = i/strideHeight;
      if ( y*strideHeight != i || y < 0 || gOutHeight <= y)  continue ;

      for (int64_t s=0; s<kernWidth; s++) {
	int64_t j = w - s * dilationWidth  + padWidth ;
	int64_t x0 = (j+0)/strideWidth ;
	int64_t x1 = (j+1)/strideWidth ;

	int64_t x0_valid  = ( x0*strideHeight == j   && x0 >= 0 &&  x0 < gOutWidth)  ;
	int64_t x1_valid  = ( x1*strideHeight == j+1 && x1 >= 0 &&  x1 < gOutWidth)  ;

	if( x0_valid && x1_valid) {
	  for (int64_t k=0; k<gOutChannelGroup; k++) {
	    int64_t gOutIndex_x0   = gOutGroupOffset + ((n * gOutChannel + k) * gOutHeight + y) * gOutWidth + x0;
	    int64_t gOutIndex_x1   = gOutGroupOffset + ((n * gOutChannel + k) * gOutHeight + y) * gOutWidth + x1;

	    int64_t kernelIndex = kernGroupOffset + ((k * gInChannelGroup + c) * kernHeight + r) * kernWidth + s;

	    __vr vrk = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex]) ;

	    vrsum_w0 = _ve_vfmads_vvsv(vrsum_w0, pGOut[gOutIndex_x0], vrk) ;
	    vrsum_w1 = _ve_vfmads_vvsv(vrsum_w1, pGOut[gOutIndex_x1], vrk) ;
	  } // gOutChannel
	}
	else if( x0_valid ) {
	  for (int64_t k=0; k<gOutChannelGroup; k++) {
	    int64_t gOutIndex_x0   = gOutGroupOffset + ((n * gOutChannel + k) * gOutHeight + y) * gOutWidth + x0;

	    int64_t kernelIndex = kernGroupOffset + ((k * gInChannelGroup + c) * kernHeight + r) * kernWidth + s;

	    __vr vrk = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex]) ;

	    vrsum_w0 = _ve_vfmads_vvsv(vrsum_w0, pGOut[gOutIndex_x0], vrk) ;
	  } // gOutChannel
	}
	else if( x1_valid ) {
	  for (int64_t k=0; k<gOutChannelGroup; k++) {
	    int64_t gOutIndex_x1   = gOutGroupOffset + ((n * gOutChannel + k) * gOutHeight + y) * gOutWidth + x1;

	    int64_t kernelIndex = kernGroupOffset + ((k * gInChannelGroup + c) * kernHeight + r) * kernWidth + s;

	    __vr vrk = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex]) ;

	    vrsum_w1 = _ve_vfmads_vvsv(vrsum_w1, pGOut[gOutIndex_x1], vrk) ;
	  } // gOutChannel
	}
      } // kernWidth
    } // kernHeight

    _ve_vstu_vss(vrsum_w0, 4*gInHeight*gInWidth, &pGIn[gInIndex]) ;
    _ve_vstu_vss(vrsum_w1, 4*gInHeight*gInWidth, &pGIn[gInIndex+1]) ;
  } // gInChannel
}

static inline void h2w1(
    const float * restrict pGOut,
    const float * restrict pKernel,
    float * restrict const pGIn,
    const int64_t gOutChannel,
    const int64_t gOutWidth,
    const int64_t gOutHeight,
    const int64_t gInChannel,
    const int64_t gInWidth,
    const int64_t gInHeight,
    const int64_t kernWidth,
    const int64_t kernHeight,
    const int64_t gInChannelGroup,
    const int64_t gOutChannelGroup,
    const int64_t strideHeight,
    const int64_t strideWidth,
    const int64_t padHeight,
    const int64_t padWidth,
    const int64_t dilationHeight,
    const int64_t dilationWidth,
    const int64_t gInGroupOffset,
    const int64_t gOutGroupOffset,
    const int64_t kernGroupOffset,
    const int64_t n,
    const int64_t h,
    const int64_t w
)
{

  for (int64_t c=0; c<gInChannelGroup; c+=VLEN) {
    int64_t gInIndex = gInGroupOffset + ((n * gInChannel + c) * gInHeight + h) * gInWidth + w;

    const int64_t vl = gInChannelGroup - c < VLEN ? gInChannelGroup - c : VLEN ;

    _ve_lvl(vl) ;
    __vr vrsum_h0 = _ve_vbrdu_vs_f32(0.f) ;
    __vr vrsum_h1 = _ve_vbrdu_vs_f32(0.f) ;

    for (int64_t r=0; r<kernHeight; r++) {
      int64_t i = h - r * dilationHeight + padHeight ;
      int64_t y0 = (i+0)/strideHeight;
      int64_t y1 = (i+1)/strideHeight;

      int64_t y0_valid  = ( y0*strideHeight == i   && y0 >= 0 &&  y0 < gOutHeight)  ;
      int64_t y1_valid  = ( y1*strideHeight == i+1 && y1 >= 0 &&  y1 < gOutHeight)  ;

      if( y0_valid && y1_valid ) {
	for (int64_t s=0; s<kernWidth; s++) {
	  int64_t j = w - s * dilationWidth  + padWidth ;
	  int64_t x = j/strideWidth ;
	  if (x*strideWidth !=j || x < 0 || gOutWidth <= x) continue;

	  for (int64_t k=0; k<gOutChannelGroup; k++) {
	    int64_t gOutIndex_y0 = gOutGroupOffset + ((n * gOutChannel + k) * gOutHeight + y0) * gOutWidth + x;
	    int64_t gOutIndex_y1 = gOutGroupOffset + ((n * gOutChannel + k) * gOutHeight + y1) * gOutWidth + x;

	    int64_t kernelIndex  = kernGroupOffset + ((k * gInChannelGroup + c) * kernHeight + r) * kernWidth + s;

	    __vr vrk = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex]) ;

	    vrsum_h0 = _ve_vfmads_vvsv(vrsum_h0, pGOut[gOutIndex_y0], vrk) ;
	    vrsum_h1 = _ve_vfmads_vvsv(vrsum_h1, pGOut[gOutIndex_y1], vrk) ;
	  } // gOutChannel
	} // kernWidth
      }
      else if( y0_valid ) {
	for (int64_t s=0; s<kernWidth; s++) {
	  int64_t j = w - s * dilationWidth  + padWidth ;
	  int64_t x = j/strideWidth ;
	  if (x*strideWidth !=j || x < 0 || gOutWidth <= x) continue;

	  for (int64_t k=0; k<gOutChannelGroup; k++) {
	    int64_t gOutIndex_y0 = gOutGroupOffset + ((n * gOutChannel + k) * gOutHeight + y0) * gOutWidth + x;

	    int64_t kernelIndex  = kernGroupOffset + ((k * gInChannelGroup + c) * kernHeight + r) * kernWidth + s;

	    __vr vrk = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex]) ;

	    vrsum_h0 = _ve_vfmads_vvsv(vrsum_h0, pGOut[gOutIndex_y0], vrk) ;
	  } // gOutChannel
	} // kernWidth
      }
      else if( y1_valid ) {
	for (int64_t s=0; s<kernWidth; s++) {
	  int64_t j = w - s * dilationWidth  + padWidth ;
	  int64_t x = j/strideWidth ;
	  if (x*strideWidth !=j || x < 0 || gOutWidth <= x) continue;

	  for (int64_t k=0; k<gOutChannelGroup; k++) {
	    int64_t gOutIndex_y1 = gOutGroupOffset + ((n * gOutChannel + k) * gOutHeight + y1) * gOutWidth + x;

	    int64_t kernelIndex  = kernGroupOffset + ((k * gInChannelGroup + c) * kernHeight + r) * kernWidth + s;

	    __vr vrk = _ve_vldu_vss(4*kernHeight*kernWidth, &pKernel[kernelIndex]) ;

	    vrsum_h1 = _ve_vfmads_vvsv(vrsum_h1, pGOut[gOutIndex_y1], vrk) ;
	  } // gOutChannel
	} // kernWidth
      }
    } // kernHeight

    _ve_vstu_vss(vrsum_h0, 4*gInHeight*gInWidth, &pGIn[gInIndex]) ;
    _ve_vstu_vss(vrsum_h1, 4*gInHeight*gInWidth, &pGIn[gInIndex+gInWidth]) ;
  } // gInChannel
}


vednnError_t
vednnConvolutionBackwardData_direct_vecC(
    const vednnTensorParam_t * restrict 	pParamGradOut,
    const void * restrict 			pDataGradOut,
    const vednnFilterParam_t * restrict 	pParamKernel,
    const void * restrict 			pDataKernel,
    const vednnConvolutionParam_t * restrict 	pParamConv,
    const vednnTensorParam_t * restrict 	pParamGradIn,
    void * restrict 				pDataGradIn
)
{
  const int64_t batch       = pParamGradOut->batch;
  const int64_t gOutChannel = pParamGradOut->channel;
  const int64_t gOutWidth   = pParamGradOut->width;
  const int64_t gOutHeight  = pParamGradOut->height;
  const int64_t gInChannel  = pParamGradIn->channel;
  const int64_t gInWidth    = pParamGradIn->width;
  const int64_t gInHeight   = pParamGradIn->height;
  const int64_t kernWidth   = pParamKernel->width;
  const int64_t kernHeight  = pParamKernel->height;

  const int64_t group          = pParamConv->group;
  const int64_t strideWidth    = pParamConv->strideWidth;;
  const int64_t strideHeight   = pParamConv->strideHeight;
  const int64_t padWidth       = pParamConv->padWidth;
  const int64_t padHeight      = pParamConv->padHeight;
  const int64_t dilationWidth  = pParamConv->dilationWidth;
  const int64_t dilationHeight = pParamConv->dilationHeight;

  const int64_t gOutChannelGroup = gOutChannel  / group;
  const int64_t gInChannelGroup  = gInChannel / group;

  const float * restrict pGOut   = pDataGradOut;
  const float * restrict pKernel = pDataKernel;
  float * restrict const pGIn   = pDataGradIn;

  const int64_t gInPixels= gInHeight*gInWidth ;


  for (int64_t n = 0; n<batch; n++) {
    for (int64_t g = 0; g < group; g++) {
      int64_t gInGroupOffset  = g * gInChannelGroup * gInHeight * gInWidth;
      int64_t gOutGroupOffset = g * gOutChannelGroup * gOutHeight  * gOutWidth;
      int64_t kernGroupOffset = g * gOutChannelGroup * gInChannelGroup * kernHeight * kernWidth;

      int64_t h=0 ;
      if( (gInHeight & 0x1) == 1) {
	int64_t w=0 ;
	if( (gInWidth & 0x1) == 1) {
	  h1w1(pGOut, pKernel, pGIn,
	       gOutChannel, gOutWidth, gOutHeight,
	       gInChannel, gInWidth, gInHeight,
	       kernWidth, kernHeight,
	       gInChannelGroup, gOutChannelGroup,
	       strideHeight, strideWidth,
	       padHeight, padWidth,
	       dilationHeight, dilationWidth,
	       gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	       n, h, w) ;
	  w+=1 ;
	}
        for ( ; w<gInWidth; w+=2) {
	  h1w2(pGOut, pKernel, pGIn,
	       gOutChannel, gOutWidth, gOutHeight,
	       gInChannel, gInWidth, gInHeight,
	       kernWidth, kernHeight,
	       gInChannelGroup, gOutChannelGroup,
	       strideHeight, strideWidth,
	       padHeight, padWidth,
	       dilationHeight, dilationWidth,
	       gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	       n, h, w) ;
        } // gInHeight
	h+=1 ;
      }
      for (; h<gInHeight; h+=2) {
        for (int64_t w=0; w<gInWidth; w++) {
	  h2w1(pGOut, pKernel, pGIn,
	       gOutChannel, gOutWidth, gOutHeight,
	       gInChannel, gInWidth, gInHeight,
	       kernWidth, kernHeight,
	       gInChannelGroup, gOutChannelGroup,
	       strideHeight, strideWidth,
	       padHeight, padWidth,
	       dilationHeight, dilationWidth,
	       gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	       n, h, w) ;
        } // gInHeight
      } // gInWidth
    } // group
  } // batch

  return VEDNN_SUCCESS;
}
