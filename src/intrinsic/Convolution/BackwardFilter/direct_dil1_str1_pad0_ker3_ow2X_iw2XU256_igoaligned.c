#include <stdint.h>
#include <stdlib.h>

#include "vednn.h"

#include "veintrin.h"
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

    _ve_lvl(VLEN) ;
    __vr vrseq = _ve_vseq_v() ;	// xy
    __vr vry_s0  = _ve_vdivsl_vvs(vrseq, inWidthHalf) ;
    __vr vrx_s0  = _ve_vsubsl_vvv(vrseq, _ve_vmulul_vsv(inWidthHalf,vry_s0)) ;
    __vm256 vm_s0 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(gOutWidthHalf, vrx_s0)) ; // condition(x<gOutWidthHalf)

    __vr vrseq2  = _ve_vaddsl_vsv(inWidthHalf-1, vrseq) ;
    __vr vry_s2  = _ve_vdivsl_vvs(vrseq2, inWidthHalf) ;
    __vr vrx_s2  = _ve_vsubsl_vvv(vrseq2, _ve_vmulul_vsv(inWidthHalf,vry_s2)) ;
    __vm256 vm_s2 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(gOutWidthHalf, vrx_s2)) ; // condition(x<gOutWidthHalf)

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

	    _ve_lvl(vl) ;

#define INIT_VRSUM_3X3(VRSUMTOKEN)			\
__vr VRSUMTOKEN ## _r0s0 = _ve_vbrd_vs_i64(0UL) ;	\
__vr VRSUMTOKEN ## _r0s1 = _ve_vbrd_vs_i64(0UL) ;	\
__vr VRSUMTOKEN ## _r0s2 = _ve_vbrd_vs_i64(0UL) ;	\
__vr VRSUMTOKEN ## _r1s0 = _ve_vbrd_vs_i64(0UL) ;	\
__vr VRSUMTOKEN ## _r1s1 = _ve_vbrd_vs_i64(0UL) ;	\
__vr VRSUMTOKEN ## _r1s2 = _ve_vbrd_vs_i64(0UL) ;	\
__vr VRSUMTOKEN ## _r2s0 = _ve_vbrd_vs_i64(0UL) ;	\
__vr VRSUMTOKEN ## _r2s1 = _ve_vbrd_vs_i64(0UL) ;	\
__vr VRSUMTOKEN ## _r2s2 = _ve_vbrd_vs_i64(0UL) ;	\

	    INIT_VRSUM_3X3(vrsum) ;

	    for (int64_t y=0; y<gOutHeight; y+=nY) {

	      const int64_t vl0 = inWidthHalf * (gOutHeight - y < nY ? gOutHeight - y : nY) ;
	      const int64_t vl1 = gOutWidthHalf * (gOutHeight - y < nY ? gOutHeight - y : nY) ;

	      const int64_t gop = y * gOutWidth ;

	      for (int64_t n=0; n<batch; n++) {
		const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;

		const int64_t gOutIndex  = outGroupOffset + ((n * gOutChannel + k  ) * gOutHeight ) * gOutWidth + gop;

		_ve_lvl(vl0) ;
		__vr vrin_r0 = _ve_vld_vss(8, pInChannel+(y+0)*inWidth) ;
		__vr vrin_r1 = _ve_vld_vss(8, pInChannel+(y+1)*inWidth) ;
		__vr vrin_r2 = _ve_vld_vss(8, pInChannel+(y+2)*inWidth) ;

		_ve_lvl(vl1) ;
		__vr vrgout = _ve_vld_vss(8, pGOut+gOutIndex+0*gOutPixels) ;

		_ve_lvl(vl0) ;
		__vr vrin_r0s0 = _ve_vcp_vvmv(vrin_r0, vm_s0, _ve_vbrd_vs_i64(0UL)) ; //vrin_r0s0) ;
		__vr vrin_r0s2 = _ve_vcp_vvmv(vrin_r0, vm_s2, _ve_vbrd_vs_i64(0UL)) ; //vrin_r0s2) ;
		__vr vrin_r1s0 = _ve_vcp_vvmv(vrin_r1, vm_s0, _ve_vbrd_vs_i64(0UL)) ; //vrin_r1s0) ;
		__vr vrin_r1s2 = _ve_vcp_vvmv(vrin_r1, vm_s2, _ve_vbrd_vs_i64(0UL)) ; //vrin_r1s2) ;
		__vr vrin_r2s0 = _ve_vcp_vvmv(vrin_r2, vm_s0, _ve_vbrd_vs_i64(0UL)) ; //vrin_r2s0) ;
		__vr vrin_r2s2 = _ve_vcp_vvmv(vrin_r2, vm_s2, _ve_vbrd_vs_i64(0UL)) ; //vrin_r2s2) ;

		_ve_lvl(vl1) ;
		__vr vrin_r0s1 = _ve_vshf_vvvs(vrin_r0s0, vrin_r0s2, VE_VSHUFFLE_ZLYU) ;
		__vr vrin_r1s1 = _ve_vshf_vvvs(vrin_r1s0, vrin_r1s2, VE_VSHUFFLE_ZLYU) ;
		__vr vrin_r2s1 = _ve_vshf_vvvs(vrin_r2s0, vrin_r2s2, VE_VSHUFFLE_ZLYU) ;

#define VFADD_3X3(VRSUMTOKEN, VRGOUT)							\
{											\
VRSUMTOKEN ## _r0s0  = _ve_pvfmad_vvvv(VRSUMTOKEN ## _r0s0, vrin_r0s0, VRGOUT) ;	\
VRSUMTOKEN ## _r0s1  = _ve_pvfmad_vvvv(VRSUMTOKEN ## _r0s1, vrin_r0s1, VRGOUT) ;	\
VRSUMTOKEN ## _r0s2  = _ve_pvfmad_vvvv(VRSUMTOKEN ## _r0s2, vrin_r0s2, VRGOUT) ;	\
VRSUMTOKEN ## _r1s0  = _ve_pvfmad_vvvv(VRSUMTOKEN ## _r1s0, vrin_r1s0, VRGOUT) ;	\
VRSUMTOKEN ## _r1s1  = _ve_pvfmad_vvvv(VRSUMTOKEN ## _r1s1, vrin_r1s1, VRGOUT) ;	\
VRSUMTOKEN ## _r1s2  = _ve_pvfmad_vvvv(VRSUMTOKEN ## _r1s2, vrin_r1s2, VRGOUT) ;	\
VRSUMTOKEN ## _r2s0  = _ve_pvfmad_vvvv(VRSUMTOKEN ## _r2s0, vrin_r2s0, VRGOUT) ;	\
VRSUMTOKEN ## _r2s1  = _ve_pvfmad_vvvv(VRSUMTOKEN ## _r2s1, vrin_r2s1, VRGOUT) ;	\
VRSUMTOKEN ## _r2s2  = _ve_pvfmad_vvvv(VRSUMTOKEN ## _r2s2, vrin_r2s2, VRGOUT) ;	\
}
		VFADD_3X3(vrsum, vrgout) ;
	      } // nBatch
	    } // gOutPixels

#define VSUM_STORE_3X3(VRSUMTOKEN, KERNELINDEX, VL)				\
{										\
_ve_lvl(VL) ;									\
VRSUMTOKEN ##_r0s0 = _ve_vfadds_vvv(VRSUMTOKEN ## _r0s0,			\
				    _ve_vsll_vvs(VRSUMTOKEN ## _r0s0,32)) ;	\
VRSUMTOKEN ##_r0s1 = _ve_vfadds_vvv(VRSUMTOKEN ## _r0s1, 			\
                                    _ve_vsll_vvs(VRSUMTOKEN ## _r0s1,32)) ;	\
VRSUMTOKEN ##_r0s2 = _ve_vfadds_vvv(VRSUMTOKEN ## _r0s2,			\
				    _ve_vsll_vvs(VRSUMTOKEN ## _r0s2,32)) ;	\
VRSUMTOKEN ##_r1s0 = _ve_vfadds_vvv(VRSUMTOKEN ## _r1s0,			\
				    _ve_vsll_vvs(VRSUMTOKEN ## _r1s0,32)) ;	\
VRSUMTOKEN ##_r1s1 = _ve_vfadds_vvv(VRSUMTOKEN ## _r1s1,			\
				    _ve_vsll_vvs(VRSUMTOKEN ## _r1s1,32)) ;	\
VRSUMTOKEN ##_r1s2 = _ve_vfadds_vvv(VRSUMTOKEN ## _r1s2,			\
				    _ve_vsll_vvs(VRSUMTOKEN ## _r1s2,32)) ;	\
VRSUMTOKEN ##_r2s0 = _ve_vfadds_vvv(VRSUMTOKEN ## _r2s0,			\
				    _ve_vsll_vvs(VRSUMTOKEN ## _r2s0,32)) ;	\
VRSUMTOKEN ##_r2s1 = _ve_vfadds_vvv(VRSUMTOKEN ## _r2s1,			\
				    _ve_vsll_vvs(VRSUMTOKEN ## _r2s1,32)) ;	\
VRSUMTOKEN ##_r2s2 = _ve_vfadds_vvv(VRSUMTOKEN ## _r2s2,			\
				    _ve_vsll_vvs(VRSUMTOKEN ## _r2s2,32)) ;	\
VRSUMTOKEN ## _r0s0 = _ve_vfsums_vv(VRSUMTOKEN ## _r0s0) ;			\
VRSUMTOKEN ## _r0s1 = _ve_vfsums_vv(VRSUMTOKEN ## _r0s1) ;			\
VRSUMTOKEN ## _r0s2 = _ve_vfsums_vv(VRSUMTOKEN ## _r0s2) ;			\
VRSUMTOKEN ## _r1s0 = _ve_vfsums_vv(VRSUMTOKEN ## _r1s0) ;			\
VRSUMTOKEN ## _r1s1 = _ve_vfsums_vv(VRSUMTOKEN ## _r1s1) ;			\
VRSUMTOKEN ## _r1s2 = _ve_vfsums_vv(VRSUMTOKEN ## _r1s2) ;			\
VRSUMTOKEN ## _r2s0 = _ve_vfsums_vv(VRSUMTOKEN ## _r2s0) ;			\
VRSUMTOKEN ## _r2s1 = _ve_vfsums_vv(VRSUMTOKEN ## _r2s1) ;			\
VRSUMTOKEN ## _r2s2 = _ve_vfsums_vv(VRSUMTOKEN ## _r2s2) ;			\
_ve_lvl(1) ;									\
_ve_vstu_vss(VRSUMTOKEN ## _r0s0,4,pGKernel+KERNELINDEX+0) ;			\
_ve_vstu_vss(VRSUMTOKEN ## _r0s1,4,pGKernel+KERNELINDEX+1) ;			\
_ve_vstu_vss(VRSUMTOKEN ## _r0s2,4,pGKernel+KERNELINDEX+2) ;			\
_ve_vstu_vss(VRSUMTOKEN ## _r1s0,4,pGKernel+KERNELINDEX+3) ;			\
_ve_vstu_vss(VRSUMTOKEN ## _r1s1,4,pGKernel+KERNELINDEX+4) ;			\
_ve_vstu_vss(VRSUMTOKEN ## _r1s2,4,pGKernel+KERNELINDEX+5) ;			\
_ve_vstu_vss(VRSUMTOKEN ## _r2s0,4,pGKernel+KERNELINDEX+6) ;			\
_ve_vstu_vss(VRSUMTOKEN ## _r2s1,4,pGKernel+KERNELINDEX+7) ;			\
_ve_vstu_vss(VRSUMTOKEN ## _r2s2,4,pGKernel+KERNELINDEX+8) ;			\
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

	    _ve_lvl(vl) ;
	    INIT_VRSUM_3X3(vrsum0) ;
	    INIT_VRSUM_3X3(vrsum1) ;

	    for (int64_t y=0; y<gOutHeight; y+=nY) {

	      const int64_t vl0 = inWidthHalf * (gOutHeight - y < nY ? gOutHeight - y : nY) ;
	      const int64_t vl1 = gOutWidthHalf * (gOutHeight - y < nY ? gOutHeight - y : nY) ;

	      const int64_t gop = y * gOutWidth ;

	      for (int64_t n=0; n<batch; n++) {
		const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;

		const int64_t gOutIndex  = outGroupOffset + ((n * gOutChannel + k  ) * gOutHeight ) * gOutWidth + gop;

		_ve_lvl(vl0) ;
		__vr vrin_r0 = _ve_vld_vss(8, pInChannel+(y+0)*inWidth) ;
		__vr vrin_r1 = _ve_vld_vss(8, pInChannel+(y+1)*inWidth) ;
		__vr vrin_r2 = _ve_vld_vss(8, pInChannel+(y+2)*inWidth) ;

		_ve_lvl(vl1) ;
		__vr vrgout0 = _ve_vld_vss(8, pGOut+gOutIndex+0*gOutPixels) ;
		__vr vrgout1 = _ve_vld_vss(8, pGOut+gOutIndex+1*gOutPixels) ;

		_ve_lvl(vl0) ;
		__vr vrin_r0s0 = _ve_vcp_vvmv(vrin_r0, vm_s0, _ve_vbrd_vs_i64(0UL)) ; //vrin_r0s0) ;
		__vr vrin_r0s2 = _ve_vcp_vvmv(vrin_r0, vm_s2, _ve_vbrd_vs_i64(0UL)) ; //vrin_r0s2) ;
		__vr vrin_r1s0 = _ve_vcp_vvmv(vrin_r1, vm_s0, _ve_vbrd_vs_i64(0UL)) ; //vrin_r1s0) ;
		__vr vrin_r1s2 = _ve_vcp_vvmv(vrin_r1, vm_s2, _ve_vbrd_vs_i64(0UL)) ; //vrin_r1s2) ;
		__vr vrin_r2s0 = _ve_vcp_vvmv(vrin_r2, vm_s0, _ve_vbrd_vs_i64(0UL)) ; //vrin_r2s0) ;
		__vr vrin_r2s2 = _ve_vcp_vvmv(vrin_r2, vm_s2, _ve_vbrd_vs_i64(0UL)) ; //vrin_r2s2) ;

		_ve_lvl(vl1) ;
		__vr vrin_r0s1 = _ve_vshf_vvvs(vrin_r0s0, vrin_r0s2, VE_VSHUFFLE_ZLYU) ;
		__vr vrin_r1s1 = _ve_vshf_vvvs(vrin_r1s0, vrin_r1s2, VE_VSHUFFLE_ZLYU) ;
		__vr vrin_r2s1 = _ve_vshf_vvvs(vrin_r2s0, vrin_r2s2, VE_VSHUFFLE_ZLYU) ;

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

	    _ve_lvl(vl) ;
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

		_ve_lvl(vl0) ;
		__vr vrin_r0 = _ve_vld_vss(8, pInChannel+(y+0)*inWidth) ;
		__vr vrin_r1 = _ve_vld_vss(8, pInChannel+(y+1)*inWidth) ;
		__vr vrin_r2 = _ve_vld_vss(8, pInChannel+(y+2)*inWidth) ;

		_ve_lvl(vl1) ;
		__vr vrgout0 = _ve_vld_vss(8, pGOut+gOutIndex+0*gOutPixels) ;
		__vr vrgout1 = _ve_vld_vss(8, pGOut+gOutIndex+1*gOutPixels) ;
		__vr vrgout2 = _ve_vld_vss(8, pGOut+gOutIndex+2*gOutPixels) ;

		_ve_lvl(vl0) ;
		__vr vrin_r0s0 = _ve_vcp_vvmv(vrin_r0, vm_s0, _ve_vbrd_vs_i64(0UL)) ; //vrin_r0s0) ;
		__vr vrin_r0s2 = _ve_vcp_vvmv(vrin_r0, vm_s2, _ve_vbrd_vs_i64(0UL)) ; //vrin_r0s2) ;
		__vr vrin_r1s0 = _ve_vcp_vvmv(vrin_r1, vm_s0, _ve_vbrd_vs_i64(0UL)) ; //vrin_r1s0) ;
		__vr vrin_r1s2 = _ve_vcp_vvmv(vrin_r1, vm_s2, _ve_vbrd_vs_i64(0UL)) ; //vrin_r1s2) ;
		__vr vrin_r2s0 = _ve_vcp_vvmv(vrin_r2, vm_s0, _ve_vbrd_vs_i64(0UL)) ; //vrin_r2s0) ;
		__vr vrin_r2s2 = _ve_vcp_vvmv(vrin_r2, vm_s2, _ve_vbrd_vs_i64(0UL)) ; //vrin_r2s2) ;

		_ve_lvl(vl1) ;
		__vr vrin_r0s1 = _ve_vshf_vvvs(vrin_r0s0, vrin_r0s2, VE_VSHUFFLE_ZLYU) ;
		__vr vrin_r1s1 = _ve_vshf_vvvs(vrin_r1s0, vrin_r1s2, VE_VSHUFFLE_ZLYU) ;
		__vr vrin_r2s1 = _ve_vshf_vvvs(vrin_r2s0, vrin_r2s2, VE_VSHUFFLE_ZLYU) ;

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

	    _ve_lvl(vl) ;
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

		_ve_lvl(vl0) ;
		__vr vrin_r0 = _ve_vld_vss(8, pInChannel+(y+0)*inWidth) ;
		__vr vrin_r1 = _ve_vld_vss(8, pInChannel+(y+1)*inWidth) ;
		__vr vrin_r2 = _ve_vld_vss(8, pInChannel+(y+2)*inWidth) ;

		_ve_lvl(vl1) ;
		__vr vrgout0 = _ve_vld_vss(8, pGOut+gOutIndex+0*gOutPixels) ;
		__vr vrgout1 = _ve_vld_vss(8, pGOut+gOutIndex+1*gOutPixels) ;
		__vr vrgout2 = _ve_vld_vss(8, pGOut+gOutIndex+2*gOutPixels) ;
		__vr vrgout3 = _ve_vld_vss(8, pGOut+gOutIndex+3*gOutPixels) ;

		_ve_lvl(vl0) ;
		__vr vrin_r0s0 = _ve_vcp_vvmv(vrin_r0, vm_s0, _ve_vbrd_vs_i64(0UL)) ; //vrin_r0s0) ;
		__vr vrin_r0s2 = _ve_vcp_vvmv(vrin_r0, vm_s2, _ve_vbrd_vs_i64(0UL)) ; //vrin_r0s2) ;
		__vr vrin_r1s0 = _ve_vcp_vvmv(vrin_r1, vm_s0, _ve_vbrd_vs_i64(0UL)) ; //vrin_r1s0) ;
		__vr vrin_r1s2 = _ve_vcp_vvmv(vrin_r1, vm_s2, _ve_vbrd_vs_i64(0UL)) ; //vrin_r1s2) ;
		__vr vrin_r2s0 = _ve_vcp_vvmv(vrin_r2, vm_s0, _ve_vbrd_vs_i64(0UL)) ; //vrin_r2s0) ;
		__vr vrin_r2s2 = _ve_vcp_vvmv(vrin_r2, vm_s2, _ve_vbrd_vs_i64(0UL)) ; //vrin_r2s2) ;

		_ve_lvl(vl1) ;
		__vr vrin_r0s1 = _ve_vshf_vvvs(vrin_r0s0, vrin_r0s2, VE_VSHUFFLE_ZLYU) ;
		__vr vrin_r1s1 = _ve_vshf_vvvs(vrin_r1s0, vrin_r1s2, VE_VSHUFFLE_ZLYU) ;
		__vr vrin_r2s1 = _ve_vshf_vvvs(vrin_r2s0, vrin_r2s2, VE_VSHUFFLE_ZLYU) ;

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

	    _ve_lvl(vl) ;
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

		_ve_lvl(vl0) ;
		__vr vrin_r0 = _ve_vld_vss(8, pInChannel+(y+0)*inWidth) ;
		__vr vrin_r1 = _ve_vld_vss(8, pInChannel+(y+1)*inWidth) ;
		__vr vrin_r2 = _ve_vld_vss(8, pInChannel+(y+2)*inWidth) ;

		_ve_lvl(vl1) ;
		__vr vrgout0 = _ve_vld_vss(8, pGOut+gOutIndex+0*gOutPixels) ;
		__vr vrgout1 = _ve_vld_vss(8, pGOut+gOutIndex+1*gOutPixels) ;
		__vr vrgout2 = _ve_vld_vss(8, pGOut+gOutIndex+2*gOutPixels) ;
		__vr vrgout3 = _ve_vld_vss(8, pGOut+gOutIndex+3*gOutPixels) ;
		__vr vrgout4 = _ve_vld_vss(8, pGOut+gOutIndex+4*gOutPixels) ;

		_ve_lvl(vl0) ;
		__vr vrin_r0s0 = _ve_vcp_vvmv(vrin_r0, vm_s0, _ve_vbrd_vs_i64(0UL)) ; //vrin_r0s0) ;
		__vr vrin_r0s2 = _ve_vcp_vvmv(vrin_r0, vm_s2, _ve_vbrd_vs_i64(0UL)) ; //vrin_r0s2) ;
		__vr vrin_r1s0 = _ve_vcp_vvmv(vrin_r1, vm_s0, _ve_vbrd_vs_i64(0UL)) ; //vrin_r1s0) ;
		__vr vrin_r1s2 = _ve_vcp_vvmv(vrin_r1, vm_s2, _ve_vbrd_vs_i64(0UL)) ; //vrin_r1s2) ;
		__vr vrin_r2s0 = _ve_vcp_vvmv(vrin_r2, vm_s0, _ve_vbrd_vs_i64(0UL)) ; //vrin_r2s0) ;
		__vr vrin_r2s2 = _ve_vcp_vvmv(vrin_r2, vm_s2, _ve_vbrd_vs_i64(0UL)) ; //vrin_r2s2) ;

		_ve_lvl(vl1) ;
		__vr vrin_r0s1 = _ve_vshf_vvvs(vrin_r0s0, vrin_r0s2, VE_VSHUFFLE_ZLYU) ;
		__vr vrin_r1s1 = _ve_vshf_vvvs(vrin_r1s0, vrin_r1s2, VE_VSHUFFLE_ZLYU) ;
		__vr vrin_r2s1 = _ve_vshf_vvvs(vrin_r2s0, vrin_r2s2, VE_VSHUFFLE_ZLYU) ;

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
