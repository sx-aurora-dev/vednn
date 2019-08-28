#include <stdint.h>
#include <stdlib.h>

#include "vednn.h"

#include "velintrin.h"
#define VLEN	(256)

vednnError_t
vednnConvolutionBackwardFilter_direct_dil1_str1_pad0_ker3_ow2X_iw2XU256_igoaligned(
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
//  const int64_t strideWidth    = pParamConv->strideWidth;	// must be 1
//  const int64_t strideHeight   = pParamConv->strideHeight;	// must be 1
//  const int64_t padWidth       = pParamConv->padWidth;	// must be 0
//  const int64_t padHeight      = pParamConv->padHeight;	// must be 0
//  const int64_t dilationWidth  = pParamConv->dilationWidth;	// must be 1
//  const int64_t dilationHeight = pParamConv->dilationHeight;	// must be 1

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
    const int64_t inWidthHalf   = inWidth >> 1 ;
    const int64_t gOutWidthHalf = gOutWidth >> 1 ;
    const int64_t nY = VLEN / inWidthHalf ;

    __vr vrseq = _vel_vseq_vl(VLEN) ;	// xy
    __vr vry_s0  = _vel_vdivsl_vvsl(vrseq, inWidthHalf, VLEN) ;
    __vr vrx_s0  = _vel_vsubsl_vvvl(vrseq, _vel_vmulul_vsvl(inWidthHalf,vry_s0, VLEN), VLEN) ;
    __vm256 vm_s0 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutWidthHalf, vrx_s0, VLEN), VLEN) ; // condition(x<gOutWidthHalf)

    __vr vrseq2  = _vel_vaddsl_vsvl(inWidthHalf-1, vrseq, VLEN) ;
    __vr vry_s2  = _vel_vdivsl_vvsl(vrseq2, inWidthHalf, VLEN) ;
    __vr vrx_s2  = _vel_vsubsl_vvvl(vrseq2, _vel_vmulul_vsvl(inWidthHalf,vry_s2, VLEN), VLEN) ;
    __vm256 vm_s2 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutWidthHalf, vrx_s2, VLEN), VLEN) ; // condition(x<gOutWidthHalf)

    {
      for (int64_t g = 0; g < group; g++) {
	int64_t inGroupOffset   = g * inChannelGroup  * inHeight  * inWidth;
	int64_t outGroupOffset  = (g * gOutChannelGroup + beginOChannel) * gOutHeight * gOutWidth;
	int64_t kernGroupOffset = (g * gOutChannelGroup + beginOChannel) * inChannelGroup * gKernHeight * gKernWidth;

	int64_t k=0 ;
	switch( nOChannel % 5 ) {
	case 1 :
	  for (int64_t c=0; c<inChannelGroup; c++) {
	    const int64_t kernelIndex = kernGroupOffset + ((k     * inChannelGroup + c) * gKernHeight) * gKernWidth;

	    const int vl = nY*gOutWidthHalf ;

#define INIT_VRSUM_3X3(VRSUMTOKEN)			\
__vr VRSUMTOKEN ## _r0s0 = _vel_vbrdl_vsl(0UL, vl) ;	\
__vr VRSUMTOKEN ## _r0s1 = _vel_vbrdl_vsl(0UL, vl) ;	\
__vr VRSUMTOKEN ## _r0s2 = _vel_vbrdl_vsl(0UL, vl) ;	\
__vr VRSUMTOKEN ## _r1s0 = _vel_vbrdl_vsl(0UL, vl) ;	\
__vr VRSUMTOKEN ## _r1s1 = _vel_vbrdl_vsl(0UL, vl) ;	\
__vr VRSUMTOKEN ## _r1s2 = _vel_vbrdl_vsl(0UL, vl) ;	\
__vr VRSUMTOKEN ## _r2s0 = _vel_vbrdl_vsl(0UL, vl) ;	\
__vr VRSUMTOKEN ## _r2s1 = _vel_vbrdl_vsl(0UL, vl) ;	\
__vr VRSUMTOKEN ## _r2s2 = _vel_vbrdl_vsl(0UL, vl) ;	\

	    INIT_VRSUM_3X3(vrsum) ;

	    for (int64_t y=0; y<gOutHeight; y+=nY) {

	      const int64_t vl0 = inWidthHalf * (gOutHeight - y < nY ? gOutHeight - y : nY) ;
	      const int64_t vl1 = gOutWidthHalf * (gOutHeight - y < nY ? gOutHeight - y : nY) ;

	      const int64_t gop = y * gOutWidth ;

	      for (int64_t n=0; n<batch; n++) {
		const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;

		const int64_t gOutIndex  = outGroupOffset + ((n * gOutChannel + k  ) * gOutHeight ) * gOutWidth + gop;

		__vr vrin_r0 = _vel_vld_vssl(8, pInChannel+(y+0)*inWidth, vl0) ;
		__vr vrin_r1 = _vel_vld_vssl(8, pInChannel+(y+1)*inWidth, vl0) ;
		__vr vrin_r2 = _vel_vld_vssl(8, pInChannel+(y+2)*inWidth, vl0) ;

		__vr vrgout = _vel_vld_vssl(8, pGOut+gOutIndex+0*gOutPixels, vl1) ;

		__vr vrin_r0s0 = _vel_vcp_vvmvl(vrin_r0, vm_s0, _vel_vbrdl_vsl(0UL, vl0), vl0) ; //vrin_r0s0) ;
		__vr vrin_r0s2 = _vel_vcp_vvmvl(vrin_r0, vm_s2, _vel_vbrdl_vsl(0UL, vl0), vl0) ; //vrin_r0s2) ;
		__vr vrin_r1s0 = _vel_vcp_vvmvl(vrin_r1, vm_s0, _vel_vbrdl_vsl(0UL, vl0), vl0) ; //vrin_r1s0) ;
		__vr vrin_r1s2 = _vel_vcp_vvmvl(vrin_r1, vm_s2, _vel_vbrdl_vsl(0UL, vl0), vl0) ; //vrin_r1s2) ;
		__vr vrin_r2s0 = _vel_vcp_vvmvl(vrin_r2, vm_s0, _vel_vbrdl_vsl(0UL, vl0), vl0) ; //vrin_r2s0) ;
		__vr vrin_r2s2 = _vel_vcp_vvmvl(vrin_r2, vm_s2, _vel_vbrdl_vsl(0UL, vl0), vl0) ; //vrin_r2s2) ;

		__vr vrin_r0s1 = _vel_vshf_vvvsl(vrin_r0s0, vrin_r0s2, VE_VSHUFFLE_ZLYU, vl1) ;
		__vr vrin_r1s1 = _vel_vshf_vvvsl(vrin_r1s0, vrin_r1s2, VE_VSHUFFLE_ZLYU, vl1) ;
		__vr vrin_r2s1 = _vel_vshf_vvvsl(vrin_r2s0, vrin_r2s2, VE_VSHUFFLE_ZLYU, vl1) ;

#define VFADD_3X3(VRSUMTOKEN, VRGOUT)										\
{														\
VRSUMTOKEN ## _r0s0  = _vel_pvfmad_vvvvvl(VRSUMTOKEN ## _r0s0, vrin_r0s0, VRGOUT, VRSUMTOKEN ## _r0s0, vl1) ;	\
VRSUMTOKEN ## _r0s1  = _vel_pvfmad_vvvvvl(VRSUMTOKEN ## _r0s1, vrin_r0s1, VRGOUT, VRSUMTOKEN ## _r0s1, vl1) ;	\
VRSUMTOKEN ## _r0s2  = _vel_pvfmad_vvvvvl(VRSUMTOKEN ## _r0s2, vrin_r0s2, VRGOUT, VRSUMTOKEN ## _r0s2, vl1) ;	\
VRSUMTOKEN ## _r1s0  = _vel_pvfmad_vvvvvl(VRSUMTOKEN ## _r1s0, vrin_r1s0, VRGOUT, VRSUMTOKEN ## _r1s0, vl1) ;	\
VRSUMTOKEN ## _r1s1  = _vel_pvfmad_vvvvvl(VRSUMTOKEN ## _r1s1, vrin_r1s1, VRGOUT, VRSUMTOKEN ## _r1s1, vl1) ;	\
VRSUMTOKEN ## _r1s2  = _vel_pvfmad_vvvvvl(VRSUMTOKEN ## _r1s2, vrin_r1s2, VRGOUT, VRSUMTOKEN ## _r1s2, vl1) ;	\
VRSUMTOKEN ## _r2s0  = _vel_pvfmad_vvvvvl(VRSUMTOKEN ## _r2s0, vrin_r2s0, VRGOUT, VRSUMTOKEN ## _r2s0, vl1) ;	\
VRSUMTOKEN ## _r2s1  = _vel_pvfmad_vvvvvl(VRSUMTOKEN ## _r2s1, vrin_r2s1, VRGOUT, VRSUMTOKEN ## _r2s1, vl1) ;	\
VRSUMTOKEN ## _r2s2  = _vel_pvfmad_vvvvvl(VRSUMTOKEN ## _r2s2, vrin_r2s2, VRGOUT, VRSUMTOKEN ## _r2s2, vl1) ;	\
}
		VFADD_3X3(vrsum, vrgout) ;
	      } // nBatch
	    } // gOutPixels

#define VSUM_STORE_3X3(VRSUMTOKEN, KERNELINDEX, VL)						\
{												\
	VRSUMTOKEN ##_r0s0 = _vel_vfadds_vvvl(VRSUMTOKEN ## _r0s0,				\
			                     _vel_vsll_vvsl(VRSUMTOKEN ## _r0s0,32, VL),	\
	                                     VL) ;						\
	VRSUMTOKEN ##_r0s1 = _vel_vfadds_vvvl(VRSUMTOKEN ## _r0s1, 				\
	                                     _vel_vsll_vvsl(VRSUMTOKEN ## _r0s1,32, VL),	\
	                                     VL) ;						\
	VRSUMTOKEN ##_r0s2 = _vel_vfadds_vvvl(VRSUMTOKEN ## _r0s2,				\
					     _vel_vsll_vvsl(VRSUMTOKEN ## _r0s2,32, VL),	\
					     VL) ;						\
	VRSUMTOKEN ##_r1s0 = _vel_vfadds_vvvl(VRSUMTOKEN ## _r1s0,				\
					     _vel_vsll_vvsl(VRSUMTOKEN ## _r1s0,32, VL),	\
					     VL) ;						\
	VRSUMTOKEN ##_r1s1 = _vel_vfadds_vvvl(VRSUMTOKEN ## _r1s1, 				\
					     _vel_vsll_vvsl(VRSUMTOKEN ## _r1s1,32, VL),	\
					     VL) ;						\
	VRSUMTOKEN ##_r1s2 = _vel_vfadds_vvvl(VRSUMTOKEN ## _r1s2,				\
					     _vel_vsll_vvsl(VRSUMTOKEN ## _r1s2,32, VL),	\
					     VL) ;						\
	VRSUMTOKEN ##_r2s0 = _vel_vfadds_vvvl(VRSUMTOKEN ## _r2s0,				\
					     _vel_vsll_vvsl(VRSUMTOKEN ## _r2s0,32, VL),	\
					     VL) ;						\
	VRSUMTOKEN ##_r2s1 = _vel_vfadds_vvvl(VRSUMTOKEN ## _r2s1, 				\
					     _vel_vsll_vvsl(VRSUMTOKEN ## _r2s1,32, VL),	\
					     VL) ;						\
	VRSUMTOKEN ##_r2s2 = _vel_vfadds_vvvl(VRSUMTOKEN ## _r2s2,				\
					     _vel_vsll_vvsl(VRSUMTOKEN ## _r2s2,32, VL),	\
					     VL) ;						\
	VRSUMTOKEN ## _r0s0 = _vel_vfsums_vvl(VRSUMTOKEN ## _r0s0, VL) ;			\
	VRSUMTOKEN ## _r0s1 = _vel_vfsums_vvl(VRSUMTOKEN ## _r0s1, VL) ;			\
	VRSUMTOKEN ## _r0s2 = _vel_vfsums_vvl(VRSUMTOKEN ## _r0s2, VL) ;			\
	VRSUMTOKEN ## _r1s0 = _vel_vfsums_vvl(VRSUMTOKEN ## _r1s0, VL) ;			\
	VRSUMTOKEN ## _r1s1 = _vel_vfsums_vvl(VRSUMTOKEN ## _r1s1, VL) ;			\
	VRSUMTOKEN ## _r1s2 = _vel_vfsums_vvl(VRSUMTOKEN ## _r1s2, VL) ;			\
	VRSUMTOKEN ## _r2s0 = _vel_vfsums_vvl(VRSUMTOKEN ## _r2s0, VL) ;			\
	VRSUMTOKEN ## _r2s1 = _vel_vfsums_vvl(VRSUMTOKEN ## _r2s1, VL) ;			\
	VRSUMTOKEN ## _r2s2 = _vel_vfsums_vvl(VRSUMTOKEN ## _r2s2, VL) ;			\
	_vel_vstu_vssl(VRSUMTOKEN ## _r0s0,4,pGKernel+KERNELINDEX+0, 1) ;			\
	_vel_vstu_vssl(VRSUMTOKEN ## _r0s1,4,pGKernel+KERNELINDEX+1, 1) ;			\
	_vel_vstu_vssl(VRSUMTOKEN ## _r0s2,4,pGKernel+KERNELINDEX+2, 1) ;			\
	_vel_vstu_vssl(VRSUMTOKEN ## _r1s0,4,pGKernel+KERNELINDEX+3, 1) ;			\
	_vel_vstu_vssl(VRSUMTOKEN ## _r1s1,4,pGKernel+KERNELINDEX+4, 1) ;			\
	_vel_vstu_vssl(VRSUMTOKEN ## _r1s2,4,pGKernel+KERNELINDEX+5, 1) ;			\
	_vel_vstu_vssl(VRSUMTOKEN ## _r2s0,4,pGKernel+KERNELINDEX+6, 1) ;			\
	_vel_vstu_vssl(VRSUMTOKEN ## _r2s1,4,pGKernel+KERNELINDEX+7, 1) ;			\
	_vel_vstu_vssl(VRSUMTOKEN ## _r2s2,4,pGKernel+KERNELINDEX+8, 1) ;			\
}

	    VSUM_STORE_3X3(vrsum, kernelIndex,vl) ;

	  } // inChannel
	  k++ ;
	  break ;
	case 2 :
	  for (int64_t c=0; c<inChannelGroup; c++) {
	    const int64_t kernelIndex0 = kernGroupOffset + (((k  ) * inChannelGroup + c) * gKernHeight) * gKernWidth;
	    const int64_t kernelIndex1 = kernGroupOffset + (((k+1) * inChannelGroup + c) * gKernHeight) * gKernWidth;

	    const int vl = nY*gOutWidthHalf ;

	    INIT_VRSUM_3X3(vrsum0) ;
	    INIT_VRSUM_3X3(vrsum1) ;

	    for (int64_t y=0; y<gOutHeight; y+=nY) {

	      const int64_t vl0 = inWidthHalf * (gOutHeight - y < nY ? gOutHeight - y : nY) ;
	      const int64_t vl1 = gOutWidthHalf * (gOutHeight - y < nY ? gOutHeight - y : nY) ;

	      const int64_t gop = y * gOutWidth ;

	      for (int64_t n=0; n<batch; n++) {
		const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;

		const int64_t gOutIndex  = outGroupOffset + ((n * gOutChannel + k  ) * gOutHeight ) * gOutWidth + gop;

		__vr vrin_r0 = _vel_vld_vssl(8, pInChannel+(y+0)*inWidth, vl0) ;
		__vr vrin_r1 = _vel_vld_vssl(8, pInChannel+(y+1)*inWidth, vl0) ;
		__vr vrin_r2 = _vel_vld_vssl(8, pInChannel+(y+2)*inWidth, vl0) ;

		__vr vrgout0 = _vel_vld_vssl(8, pGOut+gOutIndex+0*gOutPixels, vl1) ;
		__vr vrgout1 = _vel_vld_vssl(8, pGOut+gOutIndex+1*gOutPixels, vl1) ;

		__vr vrin_r0s0 = _vel_vcp_vvmvl(vrin_r0, vm_s0, _vel_vbrdl_vsl(0UL, vl0), vl0) ; //vrin_r0s0) ;
		__vr vrin_r0s2 = _vel_vcp_vvmvl(vrin_r0, vm_s2, _vel_vbrdl_vsl(0UL, vl0), vl0) ; //vrin_r0s2) ;
		__vr vrin_r1s0 = _vel_vcp_vvmvl(vrin_r1, vm_s0, _vel_vbrdl_vsl(0UL, vl0), vl0) ; //vrin_r1s0) ;
		__vr vrin_r1s2 = _vel_vcp_vvmvl(vrin_r1, vm_s2, _vel_vbrdl_vsl(0UL, vl0), vl0) ; //vrin_r1s2) ;
		__vr vrin_r2s0 = _vel_vcp_vvmvl(vrin_r2, vm_s0, _vel_vbrdl_vsl(0UL, vl0), vl0) ; //vrin_r2s0) ;
		__vr vrin_r2s2 = _vel_vcp_vvmvl(vrin_r2, vm_s2, _vel_vbrdl_vsl(0UL, vl0), vl0) ; //vrin_r2s2) ;

		__vr vrin_r0s1 = _vel_vshf_vvvsl(vrin_r0s0, vrin_r0s2, VE_VSHUFFLE_ZLYU, vl1) ;
		__vr vrin_r1s1 = _vel_vshf_vvvsl(vrin_r1s0, vrin_r1s2, VE_VSHUFFLE_ZLYU, vl1) ;
		__vr vrin_r2s1 = _vel_vshf_vvvsl(vrin_r2s0, vrin_r2s2, VE_VSHUFFLE_ZLYU, vl1) ;

		VFADD_3X3(vrsum0, vrgout0) ;
		VFADD_3X3(vrsum1, vrgout1) ;

	      } // nBatch
	    } // gOutPixels

	    VSUM_STORE_3X3(vrsum0, kernelIndex0, vl) ;
	    VSUM_STORE_3X3(vrsum1, kernelIndex1, vl) ;

	  } // inChannel

	  k+=2 ;
	  break ;
	case 3 :
	  for (int64_t c=0; c<inChannelGroup; c++) {
	    const int64_t kernelIndex0 = kernGroupOffset + (((k  ) * inChannelGroup + c) * gKernHeight) * gKernWidth;
	    const int64_t kernelIndex1 = kernGroupOffset + (((k+1) * inChannelGroup + c) * gKernHeight) * gKernWidth;
	    const int64_t kernelIndex2 = kernGroupOffset + (((k+2) * inChannelGroup + c) * gKernHeight) * gKernWidth;

	    const int vl = nY*gOutWidthHalf ;

	     ;
	    INIT_VRSUM_3X3(vrsum0) ;
	    INIT_VRSUM_3X3(vrsum1) ;
	    INIT_VRSUM_3X3(vrsum2) ;

	    for (int64_t y=0; y<gOutHeight; y+=nY) {

	      const int64_t vl0 = inWidthHalf * (gOutHeight - y < nY ? gOutHeight - y : nY) ;
	      const int64_t vl1 = gOutWidthHalf * (gOutHeight - y < nY ? gOutHeight - y : nY) ;

	      const int64_t gop = y * gOutWidth ;

	      for (int64_t n=0; n<batch; n++) {
		const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;

		const int64_t gOutIndex  = outGroupOffset + ((n * gOutChannel + k  ) * gOutHeight ) * gOutWidth + gop;

		 ;
		__vr vrin_r0 = _vel_vld_vssl(8, pInChannel+(y+0)*inWidth, vl0) ;
		__vr vrin_r1 = _vel_vld_vssl(8, pInChannel+(y+1)*inWidth, vl0) ;
		__vr vrin_r2 = _vel_vld_vssl(8, pInChannel+(y+2)*inWidth, vl0) ;

		 ;
		__vr vrgout0 = _vel_vld_vssl(8, pGOut+gOutIndex+0*gOutPixels, vl1) ;
		__vr vrgout1 = _vel_vld_vssl(8, pGOut+gOutIndex+1*gOutPixels, vl1) ;
		__vr vrgout2 = _vel_vld_vssl(8, pGOut+gOutIndex+2*gOutPixels, vl1) ;

		 ;
		__vr vrin_r0s0 = _vel_vcp_vvmvl(vrin_r0, vm_s0, _vel_vbrdl_vsl(0UL, vl0), vl0) ; //vrin_r0s0) ;
		__vr vrin_r0s2 = _vel_vcp_vvmvl(vrin_r0, vm_s2, _vel_vbrdl_vsl(0UL, vl0), vl0) ; //vrin_r0s2) ;
		__vr vrin_r1s0 = _vel_vcp_vvmvl(vrin_r1, vm_s0, _vel_vbrdl_vsl(0UL, vl0), vl0) ; //vrin_r1s0) ;
		__vr vrin_r1s2 = _vel_vcp_vvmvl(vrin_r1, vm_s2, _vel_vbrdl_vsl(0UL, vl0), vl0) ; //vrin_r1s2) ;
		__vr vrin_r2s0 = _vel_vcp_vvmvl(vrin_r2, vm_s0, _vel_vbrdl_vsl(0UL, vl0), vl0) ; //vrin_r2s0) ;
		__vr vrin_r2s2 = _vel_vcp_vvmvl(vrin_r2, vm_s2, _vel_vbrdl_vsl(0UL, vl0), vl0) ; //vrin_r2s2) ;

		 ;
		__vr vrin_r0s1 = _vel_vshf_vvvsl(vrin_r0s0, vrin_r0s2, VE_VSHUFFLE_ZLYU, vl1) ;
		__vr vrin_r1s1 = _vel_vshf_vvvsl(vrin_r1s0, vrin_r1s2, VE_VSHUFFLE_ZLYU, vl1) ;
		__vr vrin_r2s1 = _vel_vshf_vvvsl(vrin_r2s0, vrin_r2s2, VE_VSHUFFLE_ZLYU, vl1) ;

		VFADD_3X3(vrsum0, vrgout0) ;
		VFADD_3X3(vrsum1, vrgout1) ;
		VFADD_3X3(vrsum2, vrgout2) ;

	      } // nBatch
	    } // gOutPixels

	    VSUM_STORE_3X3(vrsum0, kernelIndex0, vl) ;
	    VSUM_STORE_3X3(vrsum1, kernelIndex1, vl) ;
	    VSUM_STORE_3X3(vrsum2, kernelIndex2, vl) ;

	  } // inChannel
	  k+=3 ;
	  break ;
	case 4 :
	  for (int64_t c=0; c<inChannelGroup; c++) {
	    const int64_t kernelIndex0 = kernGroupOffset + (((k  ) * inChannelGroup + c) * gKernHeight) * gKernWidth;
	    const int64_t kernelIndex1 = kernGroupOffset + (((k+1) * inChannelGroup + c) * gKernHeight) * gKernWidth;
	    const int64_t kernelIndex2 = kernGroupOffset + (((k+2) * inChannelGroup + c) * gKernHeight) * gKernWidth;
	    const int64_t kernelIndex3 = kernGroupOffset + (((k+3) * inChannelGroup + c) * gKernHeight) * gKernWidth;

	    const int vl = nY*gOutWidthHalf ;

	    INIT_VRSUM_3X3(vrsum0) ;
	    INIT_VRSUM_3X3(vrsum1) ;
	    INIT_VRSUM_3X3(vrsum2) ;
	    INIT_VRSUM_3X3(vrsum3) ;

	    for (int64_t y=0; y<gOutHeight; y+=nY) {

	      const int64_t vl0 = inWidthHalf * (gOutHeight - y < nY ? gOutHeight - y : nY) ;
	      const int64_t vl1 = gOutWidthHalf * (gOutHeight - y < nY ? gOutHeight - y : nY) ;

	      const int64_t gop = y * gOutWidth ;

	      for (int64_t n=0; n<batch; n++) {
		const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;

		const int64_t gOutIndex  = outGroupOffset + ((n * gOutChannel + k  ) * gOutHeight ) * gOutWidth + gop;

		__vr vrin_r0 = _vel_vld_vssl(8, pInChannel+(y+0)*inWidth, vl0) ;
		__vr vrin_r1 = _vel_vld_vssl(8, pInChannel+(y+1)*inWidth, vl0) ;
		__vr vrin_r2 = _vel_vld_vssl(8, pInChannel+(y+2)*inWidth, vl0) ;

		__vr vrgout0 = _vel_vld_vssl(8, pGOut+gOutIndex+0*gOutPixels, vl1) ;
		__vr vrgout1 = _vel_vld_vssl(8, pGOut+gOutIndex+1*gOutPixels, vl1) ;
		__vr vrgout2 = _vel_vld_vssl(8, pGOut+gOutIndex+2*gOutPixels, vl1) ;
		__vr vrgout3 = _vel_vld_vssl(8, pGOut+gOutIndex+3*gOutPixels, vl1) ;

		__vr vrin_r0s0 = _vel_vcp_vvmvl(vrin_r0, vm_s0, _vel_vbrdl_vsl(0UL, vl0), vl0) ; //vrin_r0s0) ;
		__vr vrin_r0s2 = _vel_vcp_vvmvl(vrin_r0, vm_s2, _vel_vbrdl_vsl(0UL, vl0), vl0) ; //vrin_r0s2) ;
		__vr vrin_r1s0 = _vel_vcp_vvmvl(vrin_r1, vm_s0, _vel_vbrdl_vsl(0UL, vl0), vl0) ; //vrin_r1s0) ;
		__vr vrin_r1s2 = _vel_vcp_vvmvl(vrin_r1, vm_s2, _vel_vbrdl_vsl(0UL, vl0), vl0) ; //vrin_r1s2) ;
		__vr vrin_r2s0 = _vel_vcp_vvmvl(vrin_r2, vm_s0, _vel_vbrdl_vsl(0UL, vl0), vl0) ; //vrin_r2s0) ;
		__vr vrin_r2s2 = _vel_vcp_vvmvl(vrin_r2, vm_s2, _vel_vbrdl_vsl(0UL, vl0), vl0) ; //vrin_r2s2) ;

		__vr vrin_r0s1 = _vel_vshf_vvvsl(vrin_r0s0, vrin_r0s2, VE_VSHUFFLE_ZLYU, vl1) ;
		__vr vrin_r1s1 = _vel_vshf_vvvsl(vrin_r1s0, vrin_r1s2, VE_VSHUFFLE_ZLYU, vl1) ;
		__vr vrin_r2s1 = _vel_vshf_vvvsl(vrin_r2s0, vrin_r2s2, VE_VSHUFFLE_ZLYU, vl1) ;

		VFADD_3X3(vrsum0, vrgout0) ;
		VFADD_3X3(vrsum1, vrgout1) ;
		VFADD_3X3(vrsum2, vrgout2) ;
		VFADD_3X3(vrsum3, vrgout3) ;

	      } // nBatch
	    } // gOutPixels

	    VSUM_STORE_3X3(vrsum0, kernelIndex0, vl) ;
	    VSUM_STORE_3X3(vrsum1, kernelIndex1, vl) ;
	    VSUM_STORE_3X3(vrsum2, kernelIndex2, vl) ;
	    VSUM_STORE_3X3(vrsum3, kernelIndex3, vl) ;

	  } // inChannel

	  k+=4 ;
	  break ;
	defualt :
	  break ;
	}
	for ( ;k<nOChannel; k+=5) {
	  for (int64_t c=0; c<inChannelGroup; c++) {
	    const int64_t kernelIndex0 = kernGroupOffset + (((k  ) * inChannelGroup + c) * gKernHeight) * gKernWidth;
	    const int64_t kernelIndex1 = kernGroupOffset + (((k+1) * inChannelGroup + c) * gKernHeight) * gKernWidth;
	    const int64_t kernelIndex2 = kernGroupOffset + (((k+2) * inChannelGroup + c) * gKernHeight) * gKernWidth;
	    const int64_t kernelIndex3 = kernGroupOffset + (((k+3) * inChannelGroup + c) * gKernHeight) * gKernWidth;
	    const int64_t kernelIndex4 = kernGroupOffset + (((k+4) * inChannelGroup + c) * gKernHeight) * gKernWidth;

	    const int vl = nY*gOutWidthHalf ;

	    INIT_VRSUM_3X3(vrsum0) ;
	    INIT_VRSUM_3X3(vrsum1) ;
	    INIT_VRSUM_3X3(vrsum2) ;
	    INIT_VRSUM_3X3(vrsum3) ;
	    INIT_VRSUM_3X3(vrsum4) ;

	    for (int64_t y=0; y<gOutHeight; y+=nY) {

	      const int64_t vl0 = inWidthHalf * (gOutHeight - y < nY ? gOutHeight - y : nY) ;
	      const int64_t vl1 = gOutWidthHalf * (gOutHeight - y < nY ? gOutHeight - y : nY) ;

	      const int64_t gop = y * gOutWidth ;

	      for (int64_t n=0; n<batch; n++) {
		const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;

		const int64_t gOutIndex  = outGroupOffset + ((n * gOutChannel + k  ) * gOutHeight ) * gOutWidth + gop;

		__vr vrin_r0 = _vel_vld_vssl(8, pInChannel+(y+0)*inWidth, vl0) ;
		__vr vrin_r1 = _vel_vld_vssl(8, pInChannel+(y+1)*inWidth, vl0) ;
		__vr vrin_r2 = _vel_vld_vssl(8, pInChannel+(y+2)*inWidth, vl0) ;

		__vr vrgout0 = _vel_vld_vssl(8, pGOut+gOutIndex+0*gOutPixels, vl1) ;
		__vr vrgout1 = _vel_vld_vssl(8, pGOut+gOutIndex+1*gOutPixels, vl1) ;
		__vr vrgout2 = _vel_vld_vssl(8, pGOut+gOutIndex+2*gOutPixels, vl1) ;
		__vr vrgout3 = _vel_vld_vssl(8, pGOut+gOutIndex+3*gOutPixels, vl1) ;
		__vr vrgout4 = _vel_vld_vssl(8, pGOut+gOutIndex+4*gOutPixels, vl1) ;

		__vr vrin_r0s0 = _vel_vcp_vvmvl(vrin_r0, vm_s0, _vel_vbrdl_vsl(0UL, vl0), vl0) ; //vrin_r0s0) ;
		__vr vrin_r0s2 = _vel_vcp_vvmvl(vrin_r0, vm_s2, _vel_vbrdl_vsl(0UL, vl0), vl0) ; //vrin_r0s2) ;
		__vr vrin_r1s0 = _vel_vcp_vvmvl(vrin_r1, vm_s0, _vel_vbrdl_vsl(0UL, vl0), vl0) ; //vrin_r1s0) ;
		__vr vrin_r1s2 = _vel_vcp_vvmvl(vrin_r1, vm_s2, _vel_vbrdl_vsl(0UL, vl0), vl0) ; //vrin_r1s2) ;
		__vr vrin_r2s0 = _vel_vcp_vvmvl(vrin_r2, vm_s0, _vel_vbrdl_vsl(0UL, vl0), vl0) ; //vrin_r2s0) ;
		__vr vrin_r2s2 = _vel_vcp_vvmvl(vrin_r2, vm_s2, _vel_vbrdl_vsl(0UL, vl0), vl0) ; //vrin_r2s2) ;

		__vr vrin_r0s1 = _vel_vshf_vvvsl(vrin_r0s0, vrin_r0s2, VE_VSHUFFLE_ZLYU, vl1) ;
		__vr vrin_r1s1 = _vel_vshf_vvvsl(vrin_r1s0, vrin_r1s2, VE_VSHUFFLE_ZLYU, vl1) ;
		__vr vrin_r2s1 = _vel_vshf_vvvsl(vrin_r2s0, vrin_r2s2, VE_VSHUFFLE_ZLYU, vl1) ;

		VFADD_3X3(vrsum0, vrgout0) ;
		VFADD_3X3(vrsum1, vrgout1) ;
		VFADD_3X3(vrsum2, vrgout2) ;
		VFADD_3X3(vrsum3, vrgout3) ;
		VFADD_3X3(vrsum4, vrgout4) ;

	      } // nBatch
	    } // gOutPixels

	    VSUM_STORE_3X3(vrsum0, kernelIndex0, vl) ;
	    VSUM_STORE_3X3(vrsum1, kernelIndex1, vl) ;
	    VSUM_STORE_3X3(vrsum2, kernelIndex2, vl) ;
	    VSUM_STORE_3X3(vrsum3, kernelIndex3, vl) ;
	    VSUM_STORE_3X3(vrsum4, kernelIndex4, vl) ;

	  } // inChannel
	} // outChannel
      } // group
    }
  }

  return VEDNN_SUCCESS;
}
