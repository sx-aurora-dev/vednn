#include <stdint.h>
#include <stdlib.h>

#include "vednn.h"

#include "velintrin.h"
#define VLEN	(256)


template <filterLayout_t FLAYOUT, int H, int W>
static inline void func(
    const float * __restrict__ pGOut,
    const float * __restrict__ pKernel,
    float * __restrict__ const pGIn,
    const int64_t gOutChannel,
    const int64_t gOutWidth,
    const int64_t gOutHeight,
    const int64_t gInChannel,
    const int64_t gInWidth,
    const int64_t gInHeight,
    const int64_t kernWidth,
    const int64_t kernHeight,
    const int64_t strideWidth,
    const int64_t strideHeight,
    const int64_t padWidth,
    const int64_t padHeight,
    const int64_t dilationWidth,
    const int64_t dilationHeight,
    const int64_t gInChannelGroup,
    const int64_t gOutChannelGroup,
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

    __vr vrsum_h0w0 = _vel_vbrds_vsl(0.f, vl) ;
    __vr vrsum_h0w1 = _vel_vbrds_vsl(0.f, vl) ;
    __vr vrsum_h0w2 = _vel_vbrds_vsl(0.f, vl) ;
    __vr vrsum_h0w3 = _vel_vbrds_vsl(0.f, vl) ;
    __vr vrsum_h1w0 = _vel_vbrds_vsl(0.f, vl) ;
    __vr vrsum_h1w1 = _vel_vbrds_vsl(0.f, vl) ;
    __vr vrsum_h1w2 = _vel_vbrds_vsl(0.f, vl) ;
    __vr vrsum_h1w3 = _vel_vbrds_vsl(0.f, vl) ;
    __vr vrsum_h2w0 = _vel_vbrds_vsl(0.f, vl) ;
    __vr vrsum_h2w1 = _vel_vbrds_vsl(0.f, vl) ;
    __vr vrsum_h2w2 = _vel_vbrds_vsl(0.f, vl) ;
    __vr vrsum_h2w3 = _vel_vbrds_vsl(0.f, vl) ;
    __vr vrsum_h3w0 = _vel_vbrds_vsl(0.f, vl) ;
    __vr vrsum_h3w1 = _vel_vbrds_vsl(0.f, vl) ;
    __vr vrsum_h3w2 = _vel_vbrds_vsl(0.f, vl) ;
    __vr vrsum_h3w3 = _vel_vbrds_vsl(0.f, vl) ;

    for (int64_t r=0; r<kernHeight; r++) {
      int64_t i = h - r * dilationHeight + padHeight ;
      int64_t y0 = (i+0)/strideHeight;
      int64_t y1 = (i+1)/strideHeight;
      int64_t y2 = (i+2)/strideHeight;
      int64_t y3 = (i+3)/strideHeight;

      int64_t y0_valid  = ( H >=1 && y0*strideHeight == i   && y0 >= 0 &&  y0 < gOutHeight)  ;
      int64_t y1_valid  = ( H >=2 && y1*strideHeight == i+1 && y1 >= 0 &&  y1 < gOutHeight)  ;
      int64_t y2_valid  = ( H >=3 && y2*strideHeight == i+2 && y2 >= 0 &&  y2 < gOutHeight)  ;
      int64_t y3_valid  = ( H >=4 && y3*strideHeight == i+3 && y3 >= 0 &&  y3 < gOutHeight)  ;

      if( y0_valid || y1_valid || y2_valid || y3_valid ) {
	for (int64_t s=0; s<kernWidth; s++) {
	  int64_t j = w - s * dilationWidth  + padWidth ;
	  int64_t x0 = (j+0)/strideWidth ;
	  int64_t x1 = (j+1)/strideWidth ;
	  int64_t x2 = (j+2)/strideWidth ;
	  int64_t x3 = (j+3)/strideWidth ;

	  int64_t x0_valid  = ( W >=1 && x0*strideHeight == j   && x0 >= 0 &&  x0 < gOutWidth)  ;
	  int64_t x1_valid  = ( W >=2 && x1*strideHeight == j+1 && x1 >= 0 &&  x1 < gOutWidth)  ;
	  int64_t x2_valid  = ( W >=3 && x2*strideHeight == j+2 && x2 >= 0 &&  x2 < gOutWidth)  ;
	  int64_t x3_valid  = ( W >=4 && x3*strideHeight == j+3 && x3 >= 0 &&  x3 < gOutWidth)  ;

	  if( x0_valid || x1_valid || x2_valid || x3_valid ) {
	    for (int64_t k=0; k<gOutChannelGroup; k++) {
	      const float *pKerValue = ( FLAYOUT == VEDNN_FILTER_LAYOUT_NCHW ) ?
					pKernel + kernGroupOffset + ((k * gInChannelGroup + c) * kernHeight + r) * kernWidth + s :
					pKernel + kernGroupOffset + ( ( r * kernWidth + s ) * gInChannelGroup + c ) * gOutChannelGroup + k ;

	      const int64_t kernelDistance = ( FLAYOUT == VEDNN_FILTER_LAYOUT_NCHW ) ?
					     gInChannelGroup * kernHeight * kernWidth :
					     1 ;

	      const int64_t kernelStride = ( FLAYOUT == VEDNN_FILTER_LAYOUT_NCHW ) ?
					   kernHeight * kernWidth :
					   gOutChannelGroup ;

	      __vr vrk = _vel_vldu_vssl(4*kernelStride, pKerValue+0*kernelDistance, vl) ;

	      if( y0_valid ) {
		int64_t gOutIndex_y0x0 = gOutGroupOffset + ((n * gOutChannel + k) * gOutHeight + y0) * gOutWidth + x0;
		int64_t gOutIndex_y0x1 = gOutGroupOffset + ((n * gOutChannel + k) * gOutHeight + y0) * gOutWidth + x1;
		int64_t gOutIndex_y0x2 = gOutGroupOffset + ((n * gOutChannel + k) * gOutHeight + y0) * gOutWidth + x2;
		int64_t gOutIndex_y0x3 = gOutGroupOffset + ((n * gOutChannel + k) * gOutHeight + y0) * gOutWidth + x3;
		if( x0_valid ) vrsum_h0w0 = _vel_vfmads_vvsvl(vrsum_h0w0, pGOut[gOutIndex_y0x0], vrk, vl) ;
		if( x1_valid ) vrsum_h0w1 = _vel_vfmads_vvsvl(vrsum_h0w1, pGOut[gOutIndex_y0x1], vrk, vl) ;
		if( x2_valid ) vrsum_h0w2 = _vel_vfmads_vvsvl(vrsum_h0w2, pGOut[gOutIndex_y0x2], vrk, vl) ;
		if( x3_valid ) vrsum_h0w3 = _vel_vfmads_vvsvl(vrsum_h0w3, pGOut[gOutIndex_y0x3], vrk, vl) ;
	      }
	      if( y1_valid ) {
		int64_t gOutIndex_y1x0 = gOutGroupOffset + ((n * gOutChannel + k) * gOutHeight + y1) * gOutWidth + x0;
		int64_t gOutIndex_y1x1 = gOutGroupOffset + ((n * gOutChannel + k) * gOutHeight + y1) * gOutWidth + x1;
		int64_t gOutIndex_y1x2 = gOutGroupOffset + ((n * gOutChannel + k) * gOutHeight + y1) * gOutWidth + x2;
		int64_t gOutIndex_y1x3 = gOutGroupOffset + ((n * gOutChannel + k) * gOutHeight + y1) * gOutWidth + x3;
		if( x0_valid ) vrsum_h1w0 = _vel_vfmads_vvsvl(vrsum_h1w0, pGOut[gOutIndex_y1x0], vrk, vl) ;
		if( x1_valid ) vrsum_h1w1 = _vel_vfmads_vvsvl(vrsum_h1w1, pGOut[gOutIndex_y1x1], vrk, vl) ;
		if( x2_valid ) vrsum_h1w2 = _vel_vfmads_vvsvl(vrsum_h1w2, pGOut[gOutIndex_y1x2], vrk, vl) ;
		if( x3_valid ) vrsum_h1w3 = _vel_vfmads_vvsvl(vrsum_h1w3, pGOut[gOutIndex_y1x3], vrk, vl) ;
	      }
	      if( y2_valid ) {
		int64_t gOutIndex_y2x0 = gOutGroupOffset + ((n * gOutChannel + k) * gOutHeight + y2) * gOutWidth + x0;
		int64_t gOutIndex_y2x1 = gOutGroupOffset + ((n * gOutChannel + k) * gOutHeight + y2) * gOutWidth + x1;
		int64_t gOutIndex_y2x2 = gOutGroupOffset + ((n * gOutChannel + k) * gOutHeight + y2) * gOutWidth + x2;
		int64_t gOutIndex_y2x3 = gOutGroupOffset + ((n * gOutChannel + k) * gOutHeight + y2) * gOutWidth + x3;
		if( x0_valid ) vrsum_h2w0 = _vel_vfmads_vvsvl(vrsum_h2w0, pGOut[gOutIndex_y2x0], vrk, vl) ;
		if( x1_valid ) vrsum_h2w1 = _vel_vfmads_vvsvl(vrsum_h2w1, pGOut[gOutIndex_y2x1], vrk, vl) ;
		if( x2_valid ) vrsum_h2w2 = _vel_vfmads_vvsvl(vrsum_h2w2, pGOut[gOutIndex_y2x2], vrk, vl) ;
		if( x3_valid ) vrsum_h2w3 = _vel_vfmads_vvsvl(vrsum_h2w3, pGOut[gOutIndex_y2x3], vrk, vl) ;
	      }
	      if( y3_valid ) {
		int64_t gOutIndex_y3x0 = gOutGroupOffset + ((n * gOutChannel + k) * gOutHeight + y3) * gOutWidth + x0;
		int64_t gOutIndex_y3x1 = gOutGroupOffset + ((n * gOutChannel + k) * gOutHeight + y3) * gOutWidth + x1;
		int64_t gOutIndex_y3x2 = gOutGroupOffset + ((n * gOutChannel + k) * gOutHeight + y3) * gOutWidth + x2;
		int64_t gOutIndex_y3x3 = gOutGroupOffset + ((n * gOutChannel + k) * gOutHeight + y3) * gOutWidth + x3;
		if( x0_valid ) vrsum_h3w0 = _vel_vfmads_vvsvl(vrsum_h3w0, pGOut[gOutIndex_y3x0], vrk, vl) ;
		if( x1_valid ) vrsum_h3w1 = _vel_vfmads_vvsvl(vrsum_h3w1, pGOut[gOutIndex_y3x1], vrk, vl) ;
		if( x2_valid ) vrsum_h3w2 = _vel_vfmads_vvsvl(vrsum_h3w2, pGOut[gOutIndex_y3x2], vrk, vl) ;
		if( x3_valid ) vrsum_h3w3 = _vel_vfmads_vvsvl(vrsum_h3w3, pGOut[gOutIndex_y3x3], vrk, vl) ;
	      }
	    } // gOutChannel
	  }
	} // kernWidth
      }
    } // kernHeight

    if( H >=1 && W >=1 ) _vel_vstu_vssl(vrsum_h0w0, 4*gInHeight*gInWidth, &pGIn[gInIndex+0*gInWidth+0], vl) ;
    if( H >=1 && W >=2 ) _vel_vstu_vssl(vrsum_h0w1, 4*gInHeight*gInWidth, &pGIn[gInIndex+0*gInWidth+1], vl) ;
    if( H >=1 && W >=3 ) _vel_vstu_vssl(vrsum_h0w2, 4*gInHeight*gInWidth, &pGIn[gInIndex+0*gInWidth+2], vl) ;
    if( H >=1 && W >=4 ) _vel_vstu_vssl(vrsum_h0w3, 4*gInHeight*gInWidth, &pGIn[gInIndex+0*gInWidth+3], vl) ;

    if( H >=2 && W >=1 ) _vel_vstu_vssl(vrsum_h1w0, 4*gInHeight*gInWidth, &pGIn[gInIndex+1*gInWidth+0], vl) ;
    if( H >=2 && W >=2 ) _vel_vstu_vssl(vrsum_h1w1, 4*gInHeight*gInWidth, &pGIn[gInIndex+1*gInWidth+1], vl) ;
    if( H >=2 && W >=3 ) _vel_vstu_vssl(vrsum_h1w2, 4*gInHeight*gInWidth, &pGIn[gInIndex+1*gInWidth+2], vl) ;
    if( H >=2 && W >=4 ) _vel_vstu_vssl(vrsum_h1w3, 4*gInHeight*gInWidth, &pGIn[gInIndex+1*gInWidth+3], vl) ;

    if( H >=3 && W >=1 ) _vel_vstu_vssl(vrsum_h2w0, 4*gInHeight*gInWidth, &pGIn[gInIndex+2*gInWidth+0], vl) ;
    if( H >=3 && W >=2 ) _vel_vstu_vssl(vrsum_h2w1, 4*gInHeight*gInWidth, &pGIn[gInIndex+2*gInWidth+1], vl) ;
    if( H >=3 && W >=3 ) _vel_vstu_vssl(vrsum_h2w2, 4*gInHeight*gInWidth, &pGIn[gInIndex+2*gInWidth+2], vl) ;
    if( H >=3 && W >=4 ) _vel_vstu_vssl(vrsum_h2w3, 4*gInHeight*gInWidth, &pGIn[gInIndex+2*gInWidth+3], vl) ;

    if( H >=4 && W >=1 ) _vel_vstu_vssl(vrsum_h3w0, 4*gInHeight*gInWidth, &pGIn[gInIndex+3*gInWidth+0], vl) ;
    if( H >=4 && W >=2 ) _vel_vstu_vssl(vrsum_h3w1, 4*gInHeight*gInWidth, &pGIn[gInIndex+3*gInWidth+1], vl) ;
    if( H >=4 && W >=3 ) _vel_vstu_vssl(vrsum_h3w2, 4*gInHeight*gInWidth, &pGIn[gInIndex+3*gInWidth+2], vl) ;
    if( H >=4 && W >=4 ) _vel_vstu_vssl(vrsum_h3w3, 4*gInHeight*gInWidth, &pGIn[gInIndex+3*gInWidth+3], vl) ;

  } // gInChannel
}

template <filterLayout_t FLAYOUT, int H>
static inline void convloopW(
    const float * __restrict__ pGOut,
    const float * __restrict__ pKernel,
    float * __restrict__ const pGIn,
    const int64_t gOutChannel,
    const int64_t gOutWidth,
    const int64_t gOutHeight,
    const int64_t gInChannel,
    const int64_t gInWidth,
    const int64_t gInHeight,
    const int64_t kernWidth,
    const int64_t kernHeight,
    const int64_t strideWidth,
    const int64_t strideHeight,
    const int64_t padWidth,
    const int64_t padHeight,
    const int64_t dilationWidth,
    const int64_t dilationHeight,
    const int64_t gInChannelGroup,
    const int64_t gOutChannelGroup,
    const int64_t gInGroupOffset,
    const int64_t gOutGroupOffset,
    const int64_t kernGroupOffset,
    const int64_t n,
    const int64_t h
)
{
  int64_t w=0 ;
  switch(gInWidth % 4) {
  case 1:
    func<FLAYOUT, H, 1>(pGOut, pKernel, pGIn,
	 gOutChannel, gOutWidth, gOutHeight,
	 gInChannel, gInWidth, gInHeight,
	 kernWidth, kernHeight,
	 strideWidth, strideHeight,
	 padWidth, padHeight,
	 dilationWidth, dilationHeight,
	 gInChannelGroup, gOutChannelGroup,
	 gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	 n, h, w) ;
    w+=1 ;
    break ;
  case 2 :
    func<FLAYOUT, H, 2>(pGOut, pKernel, pGIn,
	 gOutChannel, gOutWidth, gOutHeight,
	 gInChannel, gInWidth, gInHeight,
	 kernWidth, kernHeight,
	 strideWidth, strideHeight,
	 padWidth, padHeight,
	 dilationWidth, dilationHeight,
	 gInChannelGroup, gOutChannelGroup,
	 gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	 n, h, w) ;
    w+=2 ;
    break ;
  case 3 :
    func<FLAYOUT, H, 3>(pGOut, pKernel, pGIn,
	 gOutChannel, gOutWidth, gOutHeight,
	 gInChannel, gInWidth, gInHeight,
	 kernWidth, kernHeight,
	 strideWidth, strideHeight,
	 padWidth, padHeight,
	 dilationWidth, dilationHeight,
	 gInChannelGroup, gOutChannelGroup,
	 gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	 n, h, w) ;
    w+=3 ;
    break ;
  default :
    break ;
  }
  for( ; w < gInWidth ; w+=4 ) {
    func<FLAYOUT, H, 4>(pGOut, pKernel, pGIn,
	 gOutChannel, gOutWidth, gOutHeight,
	 gInChannel, gInWidth, gInHeight,
	 kernWidth, kernHeight,
	 strideWidth, strideHeight,
	 padWidth, padHeight,
	 dilationWidth, dilationHeight,
	 gInChannelGroup, gOutChannelGroup,
	 gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	 n, h, w) ;
    w+=4 ;
  }
}

template <filterLayout_t FLAYOUT>
static inline void convloopH(
    const float * __restrict__ pGOut,
    const float * __restrict__ pKernel,
    float * __restrict__ const pGIn,
    const int64_t gOutChannel,
    const int64_t gOutWidth,
    const int64_t gOutHeight,
    const int64_t gInChannel,
    const int64_t gInWidth,
    const int64_t gInHeight,
    const int64_t kernWidth,
    const int64_t kernHeight,
    const int64_t strideWidth,
    const int64_t strideHeight,
    const int64_t padWidth,
    const int64_t padHeight,
    const int64_t dilationWidth,
    const int64_t dilationHeight,
    const int64_t gInChannelGroup,
    const int64_t gOutChannelGroup,
    const int64_t gInGroupOffset,
    const int64_t gOutGroupOffset,
    const int64_t kernGroupOffset,
    const int64_t n
)
{
  int64_t h=0 ;
  switch(gInHeight % 4) {
  case 1 :
    convloopW<FLAYOUT, 1>(pGOut, pKernel, pGIn,
	 gOutChannel, gOutWidth, gOutHeight,
	 gInChannel, gInWidth, gInHeight,
	 kernWidth, kernHeight,
	 strideWidth, strideHeight,
	 padWidth, padHeight,
	 dilationWidth, dilationHeight,
	 gInChannelGroup, gOutChannelGroup,
	 gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	 n, h) ;
    h+=1 ;
    break ;
  case 2 :
    convloopW<FLAYOUT, 2>(pGOut, pKernel, pGIn,
	 gOutChannel, gOutWidth, gOutHeight,
	 gInChannel, gInWidth, gInHeight,
	 kernWidth, kernHeight,
	 strideWidth, strideHeight,
	 padWidth, padHeight,
	 dilationWidth, dilationHeight,
	 gInChannelGroup, gOutChannelGroup,
	 gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	 n, h) ;
    h+=2 ;
    break ;
  case 3 :
    convloopW<FLAYOUT, 3>(pGOut, pKernel, pGIn,
	 gOutChannel, gOutWidth, gOutHeight,
	 gInChannel, gInWidth, gInHeight,
	 kernWidth, kernHeight,
	 strideWidth, strideHeight,
	 padWidth, padHeight,
	 dilationWidth, dilationHeight,
	 gInChannelGroup, gOutChannelGroup,
	 gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	 n, h) ;
    h+=3 ;
    break ;
  default :
    break ;
  }
  for( ; h < gInHeight ; h+=4 ) {
    convloopW<FLAYOUT, 4>(pGOut, pKernel, pGIn,
	 gOutChannel, gOutWidth, gOutHeight,
	 gInChannel, gInWidth, gInHeight,
	 kernWidth, kernHeight,
	 strideWidth, strideHeight,
	 padWidth, padHeight,
	 dilationWidth, dilationHeight,
	 gInChannelGroup, gOutChannelGroup,
	 gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	 n, h) ;
  }
}

template<filterLayout_t FLAYOUT>
static inline void convloop(
    const float * __restrict__ pGOut,
    const float * __restrict__ pKernel,
    float * __restrict__ const pGIn,
    const int64_t batch,
    const int64_t group,
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
    const int64_t strideWidth,
    const int64_t strideHeight,
    const int64_t padWidth,
    const int64_t padHeight,
    const int64_t dilationWidth,
    const int64_t dilationHeight
)
{
  for (int64_t n=0; n<batch; n++) {
    for (int64_t g = 0; g < group; g++) {

      int64_t gInGroupOffset  = g * gInChannelGroup * gInHeight * gInWidth;
      int64_t gOutGroupOffset = g * gOutChannelGroup * gOutHeight * gOutWidth;
      int64_t kernGroupOffset = g * gOutChannelGroup * gInChannelGroup * kernHeight * kernWidth;

      convloopH<FLAYOUT>(pGOut, pKernel, pGIn,
	 gOutChannel, gOutWidth, gOutHeight,
	 gInChannel, gInWidth, gInHeight,
	 kernWidth, kernHeight,
	 strideWidth, strideHeight,
	 padWidth, padHeight,
	 dilationWidth, dilationHeight,
	 gInChannelGroup, gOutChannelGroup,
	 gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	 n ) ;
    } // group
  } // batch
}

extern "C"
vednnError_t
vednnConvolutionBackwardData_direct_vecC(
    const vednnTensorParam_t * 		pParamGradOut,
    const void *			pDataGradOut,
    const vednnFilterParam_t *	 	pParamKernel,
    const void * 			pDataKernel,
    const vednnConvolutionParam_t * 	pParamConv,
    const vednnTensorParam_t * 		pParamGradIn,
    void * 				pDataGradIn
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

  const int64_t filter_layout = pParamKernel->layout ;

  const int64_t group          = pParamConv->group;
  const int64_t strideWidth    = pParamConv->strideWidth;;
  const int64_t strideHeight   = pParamConv->strideHeight;
  const int64_t padWidth       = pParamConv->padWidth;
  const int64_t padHeight      = pParamConv->padHeight;
  const int64_t dilationWidth  = pParamConv->dilationWidth;
  const int64_t dilationHeight = pParamConv->dilationHeight;

  const int64_t gOutChannelGroup = gOutChannel  / group;
  const int64_t gInChannelGroup  = gInChannel / group;

  const float *  pGOut   = (const float *) pDataGradOut;
  const float *  pKernel = (const float *) pDataKernel;
  float *  const pGIn    = (float * const) pDataGradIn;


  if( filter_layout == VEDNN_FILTER_LAYOUT_NCHW) {
    convloop<VEDNN_FILTER_LAYOUT_NCHW>(pGOut, pKernel, pGIn,
	       batch, group,
	       gOutChannel, gOutWidth, gOutHeight,
	       gInChannel, gInWidth, gInHeight,
	       kernWidth, kernHeight,
	       gInChannelGroup, gOutChannelGroup,
	       strideWidth, strideHeight,
	       padWidth, padHeight,
	       dilationWidth, dilationHeight) ;
  }
  else {
    convloop<VEDNN_FILTER_LAYOUT_HWCN>(pGOut, pKernel, pGIn,
	       batch, group,
	       gOutChannel, gOutWidth, gOutHeight,
	       gInChannel, gInWidth, gInHeight,
	       kernWidth, kernHeight,
	       gInChannelGroup, gOutChannelGroup,
	       strideWidth, strideHeight,
	       padWidth, padHeight,
	       dilationWidth, dilationHeight) ;
  }

  return VEDNN_SUCCESS;
}
