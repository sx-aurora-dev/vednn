#include <stdint.h>
#include <stdlib.h>

#include "vednn.h"

#include "velintrin.h"
#define VLEN	(256)

vednnError_t
vednnConvolutionBackwardFilter_direct_default(
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
  const int64_t gKernWidth  = pParamGradKernel->width;
  const int64_t gKernHeight = pParamGradKernel->height;

  const int64_t group          = pParamConv->group;
  const int64_t strideWidth    = pParamConv->strideWidth;;
  const int64_t strideHeight   = pParamConv->strideHeight;
  const int64_t padWidth       = pParamConv->padWidth;
  const int64_t padHeight      = pParamConv->padHeight;
  const int64_t dilationWidth  = pParamConv->dilationWidth;
  const int64_t dilationHeight = pParamConv->dilationHeight;

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
    for (int64_t g = 0; g < group; g++) {
      int64_t inGroupOffset   = g * inChannelGroup  * inHeight  * inWidth;
      int64_t outGroupOffset  = (g * gOutChannelGroup + beginOChannel) * gOutHeight * gOutWidth;
      int64_t kernGroupOffset = (g * gOutChannelGroup + beginOChannel) * inChannelGroup * gKernHeight * gKernWidth;

      int64_t k=0;
      if ( (nOChannel & 0x01) == 1 ) {
	for (int64_t c=0; c<inChannelGroup; c++) {
	  for (int64_t r=0; r<gKernHeight; r++) {
	    for (int64_t s=0; s<gKernWidth; s++) {
	      const int64_t kernelIndex = kernGroupOffset + ((k * inChannelGroup + c) * gKernHeight + r) * gKernWidth + s;

	      __vr vrsum = _vel_vbrds_vsl(0.0f, VLEN) ;

	      for (int64_t n=0; n<batch; n++) {
		for (int64_t y = 0; y < gOutHeight ; y ++ ) {
		  int64_t i = y * strideHeight - padHeight;
		  int64_t h = i + r * dilationHeight;
		  if (h < 0 || inHeight <= h) {
		    continue;
		  }
		  for (int64_t x = 0; x < gOutWidth ; x += VLEN ) {
		    const int64_t vl = gOutWidth - x < VLEN ? gOutWidth - x  : VLEN ;


		    __vr vrw = _vel_vaddsl_vsvl(s*dilationWidth+x*strideWidth-padWidth,  _vel_vmulsl_vsvl(strideWidth, _vel_vseq_vl(vl), vl), vl) ;

		    __vm256 vmw0 =  _vel_vfmklge_mvl(vrw, vl) ;				// condition(0 <= w)
		    __vm256 vmw1 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(inWidth,vrw, vl), vl) ;	// condition(w < inWidth)

		    __vm256 vmw  = _vel_andm_mmm(vmw0, vmw1) ;

		    const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;

		    const int64_t gOutIndex  = outGroupOffset + ((n * gOutChannel + k  ) * gOutHeight + y) * gOutWidth + x ;

#if 0
		    __vr vrpin = _vel_vsfa_vvssl(vrw, 2, (uint64_t)&pInChannel[h*inWidth], vl) ;
		    __vr vrin = _vel_vgtu_vvssml(vrpin, 0, 0, vmw, vl) ;
#else
		    /* memory access errors mihgt be caused */
		    __vr vrin = _vel_vldu_vssl(4*strideWidth,&pInChannel[h*inWidth+x*strideWidth+s*dilationWidth-padWidth], vl) ;
#endif
		    vrin = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrin, vmw, vl) ;

		    __vr vrgout = _vel_vldu_vssl(4, pGOut+gOutIndex, vl) ;

		    vrsum = _vel_vfmads_vvvvvl(vrsum, vrin, vrgout, vrsum, vl) ;

		  } // gOutWidth
		} // gOutHeight
	      } // batch

	      vrsum = _vel_vfsums_vvl(vrsum, VLEN) ;
	      float sum = _vel_lvss_svs(vrsum, 0) ;
	      pGKernel[kernelIndex] += sum ;
	    } // kernWidth
	  } // kernHeight
	} // inChannel
	k++ ;
      }
      if ( ((nOChannel >> 1) & 0x01) == 1 ) {
	for (int64_t c=0; c<inChannelGroup; c++) {
	  for (int64_t r=0; r<gKernHeight; r++) {
	    for (int64_t s=0; s<gKernWidth; s++) {
	      const int64_t kernelIndex0 = kernGroupOffset + ((k     * inChannelGroup + c) * gKernHeight + r) * gKernWidth + s;
	      const int64_t kernelIndex1 = kernGroupOffset + (((k+1) * inChannelGroup + c) * gKernHeight + r) * gKernWidth + s;

	      __vr vrsum01 = _vel_vbrdl_vsl(0UL, VLEN) ;

	      for (int64_t n=0; n<batch; n++) {
		for (int64_t y = 0; y < gOutHeight ; y ++ ) {
		  int64_t i = y * strideHeight - padHeight;
		  int64_t h = i + r * dilationHeight;
		  if (h < 0 || inHeight <= h) {
		    continue;
		  }
		  for (int64_t x = 0; x < gOutWidth ; x += VLEN ) {
		    const int64_t vl = gOutWidth - x < VLEN ? gOutWidth - x  : VLEN ;

		    __vr vrw = _vel_vaddsl_vsvl(s*dilationWidth+x*strideWidth-padWidth,  _vel_vmulsl_vsvl(strideWidth, _vel_vseq_vl(vl), vl), vl) ;

		    __vm256 vmw0 =  _vel_vfmklge_mvl(vrw, vl) ;				// condition(0 <= w)
		    __vm256 vmw1 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(inWidth,vrw, vl), vl) ;	// condition(w < inWidth)

		    __vm256 vmw  = _vel_andm_mmm(vmw0, vmw1) ;

		    const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;

		    const int64_t gOutIndex0  = outGroupOffset + ((n * gOutChannel + k  ) * gOutHeight + y) * gOutWidth + x ;
		    const int64_t gOutIndex1  = outGroupOffset + ((n * gOutChannel + k+1) * gOutHeight + y) * gOutWidth + x ;

#if 0
		    __vr vrpin = _vel_vsfa_vvssl(vrw, 2, (uint64_t)&pInChannel[h*inWidth], vl) ;
		    __vr vrin = _vel_vgtu_vvssml(vrpin, 0, 0, vmw, vl) ;
#else
		    /* memory access errors mihgt be caused */
		    __vr vrin = _vel_vldu_vssl(4*strideWidth,&pInChannel[h*inWidth+x*strideWidth+s*dilationWidth-padWidth], vl) ;
#endif
		    vrin = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrin, vmw, vl) ;

		    __vr vrinP = _vel_vshf_vvvsl(vrin, vrin, VE_VSHUFFLE_YUZU, vl) ;

		    __vr vrgout0 = _vel_vldu_vssl(4, pGOut+gOutIndex0, vl) ;
		    __vr vrgout1 = _vel_vldu_vssl(4, pGOut+gOutIndex1, vl) ;

		    __vr vrgout01 = _vel_vshf_vvvsl(vrgout0, vrgout1, VE_VSHUFFLE_YUZU, vl) ;

		    vrsum01 = _vel_pvfmad_vvvvvl(vrsum01, vrinP, vrgout01, vrsum01, vl) ;

		  } // gOutWidth
		} // gOutHeight
	      } // batch

	      __vr vrsum0 = _vel_vfsums_vvl(vrsum01, VLEN) ;
	      __vr vrsum1 = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsum01,32, VLEN), VLEN);


	      pGKernel[kernelIndex0] += _vel_lvss_svs(vrsum0,0) ;
	      pGKernel[kernelIndex1] += _vel_lvss_svs(vrsum1,0) ;

	    } // kernWidth
	  } // kernHeight
	} // inChannel
	k+=2;
      }
      if ( ((nOChannel >> 2) & 0x01) == 1 ) {
	for (int64_t c=0; c<inChannelGroup; c++) {
	  for (int64_t r=0; r<gKernHeight; r++) {
	    for (int64_t s=0; s<gKernWidth; s++) {
	      const int64_t kernelIndex0 = kernGroupOffset + ((k     * inChannelGroup + c) * gKernHeight + r) * gKernWidth + s;
	      const int64_t kernelIndex1 = kernGroupOffset + (((k+1) * inChannelGroup + c) * gKernHeight + r) * gKernWidth + s;
	      const int64_t kernelIndex2 = kernGroupOffset + (((k+2) * inChannelGroup + c) * gKernHeight + r) * gKernWidth + s;
	      const int64_t kernelIndex3 = kernGroupOffset + (((k+3) * inChannelGroup + c) * gKernHeight + r) * gKernWidth + s;

	      __vr vrsum01 = _vel_vbrdl_vsl(0UL, VLEN) ;
	      __vr vrsum23 = _vel_vbrdl_vsl(0UL, VLEN) ;

	      for (int64_t n=0; n<batch; n++) {
		for (int64_t y = 0; y < gOutHeight ; y ++ ) {
		  int64_t i = y * strideHeight - padHeight;
		  int64_t h = i + r * dilationHeight;
		  if (h < 0 || inHeight <= h) {
		    continue;
		  }
		  for (int64_t x = 0; x < gOutWidth ; x += VLEN ) {
		    const int64_t vl = gOutWidth - x < VLEN ? gOutWidth - x  : VLEN ;

		     ;

		    __vr vrw = _vel_vaddsl_vsvl(s*dilationWidth+x*strideWidth-padWidth,  _vel_vmulsl_vsvl(strideWidth, _vel_vseq_vl(vl), vl), vl) ;

		    __vm256 vmw0 =  _vel_vfmklge_mvl(vrw, vl) ;				// condition(0 <= w)
		    __vm256 vmw1 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(inWidth,vrw, vl), vl) ;	// condition(w < inWidth)

		    __vm256 vmw  = _vel_andm_mmm(vmw0, vmw1) ;

		    const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;

		    const int64_t gOutIndex0  = outGroupOffset + ((n * gOutChannel + k  ) * gOutHeight + y) * gOutWidth + x ;
		    const int64_t gOutIndex1  = outGroupOffset + ((n * gOutChannel + k+1) * gOutHeight + y) * gOutWidth + x ;
		    const int64_t gOutIndex2  = outGroupOffset + ((n * gOutChannel + k+2) * gOutHeight + y) * gOutWidth + x ;
		    const int64_t gOutIndex3  = outGroupOffset + ((n * gOutChannel + k+3) * gOutHeight + y) * gOutWidth + x ;

#if 0
		    __vr vrpin = _vel_vsfa_vvssl(vrw, 2, (uint64_t)&pInChannel[h*inWidth], vl) ;
		    __vr vrin = _vel_vgtu_vvssml(vrpin, 0, 0, vmw, vl) ;
#else
		    /* memory access errors mihgt be caused */
		    __vr vrin = _vel_vldu_vssl(4*strideWidth,&pInChannel[h*inWidth+x*strideWidth+s*dilationWidth-padWidth], vl) ;
#endif
		    vrin = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrin, vmw, vl) ;

		    __vr vrinP = _vel_vshf_vvvsl(vrin, vrin, VE_VSHUFFLE_YUZU, vl) ;

		    __vr vrgout0 = _vel_vldu_vssl(4, pGOut+gOutIndex0, vl) ;
		    __vr vrgout1 = _vel_vldu_vssl(4, pGOut+gOutIndex1, vl) ;
		    __vr vrgout2 = _vel_vldu_vssl(4, pGOut+gOutIndex2, vl) ;
		    __vr vrgout3 = _vel_vldu_vssl(4, pGOut+gOutIndex3, vl) ;

		    __vr vrgout01 = _vel_vshf_vvvsl(vrgout0, vrgout1, VE_VSHUFFLE_YUZU, vl) ;
		    __vr vrgout23 = _vel_vshf_vvvsl(vrgout2, vrgout3, VE_VSHUFFLE_YUZU, vl) ;

		    vrsum01 = _vel_pvfmad_vvvvvl(vrsum01, vrinP, vrgout01, vrsum01, vl) ;
		    vrsum23 = _vel_pvfmad_vvvvvl(vrsum23, vrinP, vrgout23, vrsum23, vl) ;

		  } // gOutWidth
		} // gOutHeight
	      } // batch

	      __vr vrsum0 = _vel_vfsums_vvl(vrsum01, VLEN) ;
	      __vr vrsum1 = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsum01,32, VLEN), VLEN);
	      __vr vrsum2 = _vel_vfsums_vvl(vrsum23, VLEN) ;
	      __vr vrsum3 = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsum23,32, VLEN), VLEN);


	      pGKernel[kernelIndex0] += _vel_lvss_svs(vrsum0,0) ;
	      pGKernel[kernelIndex1] += _vel_lvss_svs(vrsum1,0) ;
	      pGKernel[kernelIndex2] += _vel_lvss_svs(vrsum2,0) ;
	      pGKernel[kernelIndex3] += _vel_lvss_svs(vrsum3,0) ;

	    } // kernWidth
	  } // kernHeight
	} // inChannel
	k+=4;
      }
      for ( ;k<nOChannel; k+=8) {
	for (int64_t c=0; c<inChannelGroup; c++) {
	  for (int64_t r=0; r<gKernHeight; r++) {
	    for (int64_t s=0; s<gKernWidth; s++) {
	      const int64_t kernelIndex0 = kernGroupOffset + ((k     * inChannelGroup + c) * gKernHeight + r) * gKernWidth + s;
	      const int64_t kernelIndex1 = kernGroupOffset + (((k+1) * inChannelGroup + c) * gKernHeight + r) * gKernWidth + s;
	      const int64_t kernelIndex2 = kernGroupOffset + (((k+2) * inChannelGroup + c) * gKernHeight + r) * gKernWidth + s;
	      const int64_t kernelIndex3 = kernGroupOffset + (((k+3) * inChannelGroup + c) * gKernHeight + r) * gKernWidth + s;
	      const int64_t kernelIndex4 = kernGroupOffset + (((k+4) * inChannelGroup + c) * gKernHeight + r) * gKernWidth + s;
	      const int64_t kernelIndex5 = kernGroupOffset + (((k+5) * inChannelGroup + c) * gKernHeight + r) * gKernWidth + s;
	      const int64_t kernelIndex6 = kernGroupOffset + (((k+6) * inChannelGroup + c) * gKernHeight + r) * gKernWidth + s;
	      const int64_t kernelIndex7 = kernGroupOffset + (((k+7) * inChannelGroup + c) * gKernHeight + r) * gKernWidth + s;

	       ;
	      __vr vrsum01 = _vel_vbrdl_vsl(0UL, VLEN) ;
	      __vr vrsum23 = _vel_vbrdl_vsl(0UL, VLEN) ;
	      __vr vrsum45= _vel_vbrdl_vsl(0UL, VLEN) ;
	      __vr vrsum67 = _vel_vbrdl_vsl(0UL, VLEN) ;

	      for (int64_t n=0; n<batch; n++) {
		for (int64_t y = 0; y < gOutHeight ; y ++ ) {
		  int64_t i = y * strideHeight - padHeight;
		  int64_t h = i + r * dilationHeight;
		  if (h < 0 || inHeight <= h) {
		    continue;
		  }
		  for (int64_t x = 0; x < gOutWidth ; x += VLEN ) {
		    const int64_t vl = gOutWidth - x < VLEN ? gOutWidth - x  : VLEN ;

		    __vr vrw = _vel_vaddsl_vsvl(s*dilationWidth+x*strideWidth-padWidth,  _vel_vmulsl_vsvl(strideWidth, _vel_vseq_vl(vl), vl), vl) ;

		    __vm256 vmw0 =  _vel_vfmklge_mvl(vrw, vl) ;				// condition(0 <= w)
		    __vm256 vmw1 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(inWidth,vrw, vl), vl) ;	// condition(w < inWidth)

		    __vm256 vmw  = _vel_andm_mmm(vmw0, vmw1) ;

		    const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;

		    const int64_t gOutIndex0  = outGroupOffset + ((n * gOutChannel + k  ) * gOutHeight + y) * gOutWidth + x ;
		    const int64_t gOutIndex1  = outGroupOffset + ((n * gOutChannel + k+1) * gOutHeight + y) * gOutWidth + x ;
		    const int64_t gOutIndex2  = outGroupOffset + ((n * gOutChannel + k+2) * gOutHeight + y) * gOutWidth + x ;
		    const int64_t gOutIndex3  = outGroupOffset + ((n * gOutChannel + k+3) * gOutHeight + y) * gOutWidth + x ;
		    const int64_t gOutIndex4  = outGroupOffset + ((n * gOutChannel + k+4) * gOutHeight + y) * gOutWidth + x ;
		    const int64_t gOutIndex5  = outGroupOffset + ((n * gOutChannel + k+5) * gOutHeight + y) * gOutWidth + x ;
		    const int64_t gOutIndex6  = outGroupOffset + ((n * gOutChannel + k+6) * gOutHeight + y) * gOutWidth + x ;
		    const int64_t gOutIndex7  = outGroupOffset + ((n * gOutChannel + k+7) * gOutHeight + y) * gOutWidth + x ;

#if 0
		    __vr vrpin = _vel_vsfa_vvssl(vrw, 2, (uint64_t)&pInChannel[h*inWidth], vl) ;
		    __vr vrin = _vel_vgtu_vvssml(vrpin, 0, 0, vmw, vl) ;
#else
		    /* memory access errors mihgt be caused */
		    __vr vrin = _vel_vldu_vssl(4*strideWidth,&pInChannel[h*inWidth+x*strideWidth+s*dilationWidth-padWidth], vl) ;
#endif
		    vrin = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrin, vmw, vl) ;

		    __vr vrinP = _vel_vshf_vvvsl(vrin, vrin, VE_VSHUFFLE_YUZU, vl) ;

		    __vr vrgout0 = _vel_vldu_vssl(4, pGOut+gOutIndex0, vl) ;
		    __vr vrgout1 = _vel_vldu_vssl(4, pGOut+gOutIndex1, vl) ;
		    __vr vrgout2 = _vel_vldu_vssl(4, pGOut+gOutIndex2, vl) ;
		    __vr vrgout3 = _vel_vldu_vssl(4, pGOut+gOutIndex3, vl) ;
		    __vr vrgout4 = _vel_vldu_vssl(4, pGOut+gOutIndex4, vl) ;
		    __vr vrgout5 = _vel_vldu_vssl(4, pGOut+gOutIndex5, vl) ;
		    __vr vrgout6 = _vel_vldu_vssl(4, pGOut+gOutIndex6, vl) ;
		    __vr vrgout7 = _vel_vldu_vssl(4, pGOut+gOutIndex7, vl) ;

		    __vr vrgout01 = _vel_vshf_vvvsl(vrgout0, vrgout1, VE_VSHUFFLE_YUZU, vl) ;
		    __vr vrgout23 = _vel_vshf_vvvsl(vrgout2, vrgout3, VE_VSHUFFLE_YUZU, vl) ;
		    __vr vrgout45 = _vel_vshf_vvvsl(vrgout4, vrgout5, VE_VSHUFFLE_YUZU, vl) ;
		    __vr vrgout67 = _vel_vshf_vvvsl(vrgout6, vrgout7, VE_VSHUFFLE_YUZU, vl) ;

		    vrsum01 = _vel_pvfmad_vvvvvl(vrsum01, vrinP, vrgout01, vrsum01, vl) ;
		    vrsum23 = _vel_pvfmad_vvvvvl(vrsum23, vrinP, vrgout23, vrsum23, vl) ;
		    vrsum45 = _vel_pvfmad_vvvvvl(vrsum45, vrinP, vrgout45, vrsum45, vl) ;
		    vrsum67 = _vel_pvfmad_vvvvvl(vrsum67, vrinP, vrgout67, vrsum67, vl) ;

		  } // gOutWidth
		} // gOutHeight
	      } // batch

	       ;
	      __vr vrsum0 = _vel_vfsums_vvl(vrsum01, VLEN) ;
	      __vr vrsum1 = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsum01,32, VLEN), VLEN);
	      __vr vrsum2 = _vel_vfsums_vvl(vrsum23, VLEN) ;
	      __vr vrsum3 = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsum23,32, VLEN), VLEN);
	      __vr vrsum4 = _vel_vfsums_vvl(vrsum45, VLEN) ;
	      __vr vrsum5 = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsum45,32, VLEN), VLEN);
	      __vr vrsum6 = _vel_vfsums_vvl(vrsum67, VLEN) ;
	      __vr vrsum7 = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsum67,32, VLEN), VLEN);

	      pGKernel[kernelIndex0] += _vel_lvss_svs(vrsum0,0) ;
	      pGKernel[kernelIndex1] += _vel_lvss_svs(vrsum1,0) ;
	      pGKernel[kernelIndex2] += _vel_lvss_svs(vrsum2,0) ;
	      pGKernel[kernelIndex3] += _vel_lvss_svs(vrsum3,0) ;
	      pGKernel[kernelIndex4] += _vel_lvss_svs(vrsum4,0) ;
	      pGKernel[kernelIndex5] += _vel_lvss_svs(vrsum5,0) ;
	      pGKernel[kernelIndex6] += _vel_lvss_svs(vrsum6,0) ;
	      pGKernel[kernelIndex7] += _vel_lvss_svs(vrsum7,0) ;

	    } // kernWidth
	  } // kernHeight
	} // inChannel
      } // outChannel
    } // group
  }

  return VEDNN_SUCCESS;
}

#if 0	// reference version
vednnError_t
vednnConvolutionBackwardFilter_direct_default(
    const vednnTensorParam_t * restrict 	pParamIn,
    const void * restrict 			pDataIn,
    const vednnTensorParam_t * restrict 	pParamGradOut,
    const void * restrict 			pDataGradOut,
    const vednnConvolutionParam_t * restrict 	pParamConv,
    const vednnFilterParam_t * restrict 	pParamGradKernel,
    void * restrict 				pDataGradKernel
)
{
  const int64_t inChannel   = pParamIn->channel;
  const int64_t inWidth     = pParamIn->width;
  const int64_t inHeight    = pParamIn->height;
  const int64_t batch       = pParamGradOut->batch;
  const int64_t gOutChannel = pParamGradOut->channel;
  const int64_t gOutWidth   = pParamGradOut->width;
  const int64_t gOutHeight  = pParamGradOut->height;
  const int64_t gKernWidth  = pParamGradKernel->width;
  const int64_t gKernHeight = pParamGradKernel->height;

  const int64_t group          = pParamConv->group;
  const int64_t strideWidth    = pParamConv->strideWidth;;
  const int64_t strideHeight   = pParamConv->strideHeight;
  const int64_t padWidth       = pParamConv->padWidth;
  const int64_t padHeight      = pParamConv->padHeight;
  const int64_t dilationWidth  = pParamConv->dilationWidth;
  const int64_t dilationHeight = pParamConv->dilationHeight;

  const int64_t inChannelGroup   =  inChannel   / group;
  const int64_t gOutChannelGroup = gOutChannel  / group;

  const float * restrict pIn      = pDataIn;
  const float * restrict pGOut    = pDataGradOut;
  float * restrict const pGKernel = pDataGradKernel;

#if 1
  /* version 1 : base version */
  {
    for (int64_t n=0; n<batch; n++) {
      for (int64_t g = 0; g < group; g++) {
	int64_t inGroupOffset   = g * inChannelGroup  * inHeight  * inWidth;
	int64_t outGroupOffset  = g * gOutChannelGroup * gOutHeight * gOutWidth;
	int64_t kernGroupOffset = g * gOutChannelGroup * inChannelGroup * gKernHeight * gKernWidth;

	for (int64_t k=0; k<gOutChannelGroup; k++) {
	  for (int64_t c=0; c<inChannelGroup; c++) {
	    for (int64_t r=0; r<gKernHeight; r++) {
	      for (int64_t s=0; s<gKernWidth; s++) {
		int64_t kernelIndex = kernGroupOffset + ((k * inChannelGroup + c) * gKernHeight + r) * gKernWidth + s;
		float sum = 0.0f ;
		for (int64_t y=0; y<gOutHeight; y++) {
		  int64_t i = y * strideHeight - padHeight;
		  for (int64_t x=0; x<gOutWidth; x++) {
		    int64_t j = x * strideWidth - padWidth;
		    int64_t outIndex  = outGroupOffset + ((n * gOutChannel + k) * gOutHeight + y) * gOutWidth + x;

		    int64_t h = i + r * dilationHeight;
		    int64_t w = j + s * dilationWidth;
		    if (h < 0 || inHeight <= h) {
		      continue;
		    }
		    if (w < 0 || inWidth <= w) {
		      continue;
		    }
		    int64_t inputIndex  = inGroupOffset + ((n * inChannel + c) * inHeight + h) * inWidth + w;

		    sum += (pGOut[outIndex] * pIn[inputIndex]);

		  } // outWidth
		} // outHeight
		pGKernel[kernelIndex] += sum ;
	      } // kernWidth
	    } // kernHeight
	  } // inChannel
	} // outChannel
      } // group
    } // batch
  }
#else
  /* version 0 : generated from forward propagation */
  {
    float * restrict pKernel = (float * restrict) pDataGradKernel;
    float * restrict pOut    = (float * restrict) pDataGradOut;

    const int64_t outChannel = gOutChannel ;

    const int64_t outChannelGroup = gOutChannelGroup ;

    const int64_t outHeight = gOutHeight ;
    const int64_t outWidth  = gOutWidth ;

    const int64_t kernHeight = gKernHeight ;
    const int64_t kernWidth  = gKernWidth ;

    for(int64_t i=0; i<inChannel*outChannel*kernHeight*kernWidth; i++) pKernel[i] = 0.0f ;

    for (int64_t n=0; n<batch; n++) {
      for (int64_t g = 0; g < group; g++) {
	int64_t inGroupOffset   = g * inChannelGroup  * inHeight  * inWidth;
	int64_t outGroupOffset  = g * outChannelGroup * outHeight * outWidth;
	int64_t kernGroupOffset = g * outChannelGroup * inChannelGroup * kernHeight * kernWidth;

	for (int64_t k=0; k<outChannelGroup; k++) {
	  for (int64_t p=0; p<outHeight; p++) {
	    int64_t i = p * strideHeight - padHeight;
	    for (int64_t q=0; q<outWidth; q++) {
	      int64_t j = q * strideWidth - padWidth;
	      int64_t outIndex  = outGroupOffset + ((n * outChannel + k) * outHeight + p) * outWidth + q;
	      for (int64_t c=0; c<inChannelGroup; c++) {
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
		    pKernel[kernelIndex] += (pOut[outIndex] * pIn[inputIndex]);
		  } // kernWidth
		} // kernHeight
	      } // inChannel
	    } // outWidth
	  } // outHeight
	} // outChannel
      } // group
    } // batch
  }
#endif

  return VEDNN_SUCCESS;
}
#endif
