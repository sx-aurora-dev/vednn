#include <stdint.h>
#include <stdlib.h>

#include "vednn.h"

#include "veintrin.h"
#define VLEN	(256)

vednnError_t
vednnConvolutionBackwardFilter_direct_dil1_str1_padsame_ker3_owU128(
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
  const int64_t gKernWidth  = pParamGradKernel->width;		/* must be 3 */
  const int64_t gKernHeight = pParamGradKernel->height;		/* must be 3 */

  const int64_t group          = pParamConv->group;
//  const int64_t strideWidth    = pParamConv->strideWidth;	/* must be 1 */
//  const int64_t strideHeight   = pParamConv->strideHeight;	/* must be 1 */
//  const int64_t padWidth       = pParamConv->padWidth;	/* must be 1 */
//  const int64_t padHeight      = pParamConv->padHeight;	/* must be 1 */
//  const int64_t dilationWidth  = pParamConv->dilationWidth;	/* must be 1 */
//  const int64_t dilationHeight = pParamConv->dilationHeight;	/* must be 1 */

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
    const int64_t nY = VLEN / gOutWidth ;

    const int64_t y_remain = gOutHeight % nY ;
    const int64_t last_y = y_remain == 0 ? gOutHeight - nY : gOutHeight - y_remain ;

    _ve_lvl(VLEN) ;

    __vr vrseq = _ve_vseq_v() ;			// xy

    __vr vry   = _ve_vdivsl_vvs(vrseq, gOutWidth) ;
    __vr vrx   = _ve_vsubsl_vvv(vrseq, _ve_vmulul_vsv(gOutWidth,vry)) ;

    __vm256 vmw0_s0 = _ve_vfmkl_mcv(VECC_GE, _ve_vaddsl_vsv(-1, vrx)) ;		// condition(  1<=x)
    __vm256 vmw1_s2 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(inWidth-1,vrx)) ;	// condition(x+1< inWidth)

    __vm256 vmh0_r0 = _ve_vfmkl_mcv(VECC_GE, _ve_vaddsl_vsv(-1, vry) ) ;	// condition(  1<=y)
    __vm256 vmh1_r2 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(inHeight-1-last_y,vry)) ;	// condition(y+1< inHeight)

    __vm256 vmall_r0s0 = _ve_andm_mmm(vmh0_r0, vmw0_s0) ;
    __vm256 vmall_r0s1 = vmh0_r0 ;
    __vm256 vmall_r0s2 = _ve_andm_mmm(vmh0_r0, vmw1_s2) ;

    __vm256 vmall_r2s0 = _ve_andm_mmm(vmh1_r2, vmw0_s0) ;
    __vm256 vmall_r2s1 = vmh1_r2 ;
    __vm256 vmall_r2s2 = _ve_andm_mmm(vmh1_r2, vmw1_s2) ;


    {
      for (int64_t g = 0; g < group; g++) {
	int64_t inGroupOffset   = g * inChannelGroup  * inHeight  * inWidth;
	int64_t outGroupOffset  = (g * gOutChannelGroup + beginOChannel) * gOutHeight * gOutWidth;
	int64_t kernGroupOffset = (g * gOutChannelGroup + beginOChannel) * inChannelGroup * gKernHeight * gKernWidth;

	int64_t k = 0 ;
	if ( (nOChannel & 0x01) == 1 ) {
	  for (int64_t c=0; c<inChannelGroup; c++) {
	    const int64_t kernelIndex = kernGroupOffset + ((k     * inChannelGroup + c) * gKernHeight) * gKernWidth;

	    _ve_lvl(VLEN) ;
	    __vr vrsum_r0s0 = _ve_vbrdu_vs_f32(0.0f) ;
	    __vr vrsum_r0s1 = _ve_vbrdu_vs_f32(0.0f) ;
	    __vr vrsum_r0s2 = _ve_vbrdu_vs_f32(0.0f) ;
	    __vr vrsum_r1s0 = _ve_vbrdu_vs_f32(0.0f) ;
	    __vr vrsum_r1s1 = _ve_vbrdu_vs_f32(0.0f) ;
	    __vr vrsum_r1s2 = _ve_vbrdu_vs_f32(0.0f) ;
	    __vr vrsum_r2s0 = _ve_vbrdu_vs_f32(0.0f) ;
	    __vr vrsum_r2s1 = _ve_vbrdu_vs_f32(0.0f) ;
	    __vr vrsum_r2s2 = _ve_vbrdu_vs_f32(0.0f) ;

	    for (int64_t n=0; n<batch; n++) {
	      const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;
	      const int64_t gOutIndex  = outGroupOffset + ((n * gOutChannel + k  ) * gOutHeight ) * gOutWidth ;

	      int64_t y=0 ;
	      {
		const int64_t vl = gOutWidth * nY ;
		const int64_t gop = y * gOutWidth ;

		_ve_lvl(vl) ;

		const int64_t gOutIndex  = outGroupOffset + ((n * gOutChannel + k  ) * gOutHeight ) * gOutWidth + gop;

		/* memory access errors mihgt be caused (vrin) */
		__vr vrin_r0s0    = _ve_vldu_vss(4,&pInChannel[gop-inWidth-1]) ;
		__vr vrin_r0s1    = _ve_vldu_vss(4,&pInChannel[gop-inWidth  ]) ;
		__vr vrin_r0s2    = _ve_vldu_vss(4,&pInChannel[gop-inWidth+1]) ;
		__vr vrin_r1s0    = _ve_vldu_vss(4,&pInChannel[gop        -1]) ;
		__vr vrin_r1s1    = _ve_vldu_vss(4,&pInChannel[gop          ]) ;
		__vr vrin_r1s2    = _ve_vldu_vss(4,&pInChannel[gop        +1]) ;
		__vr vrin_r2s0    = _ve_vldu_vss(4,&pInChannel[gop+inWidth-1]) ;
		__vr vrin_r2s1    = _ve_vldu_vss(4,&pInChannel[gop+inWidth  ]) ;
		__vr vrin_r2s2    = _ve_vldu_vss(4,&pInChannel[gop+inWidth+1]) ;

		__vr vrgout = _ve_vldu_vss(4, pGOut+gOutIndex+gop+0*gOutPixels) ;

		vrin_r0s0 = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), vrin_r0s0, vmall_r0s0) ;
		vrsum_r0s0 = _ve_vfmads_vvvv(vrsum_r0s0, vrin_r0s0, vrgout) ;
		vrin_r0s1 = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), vrin_r0s1, vmall_r0s1) ;
		vrsum_r0s1 = _ve_vfmads_vvvv(vrsum_r0s1, vrin_r0s1, vrgout) ;
		vrin_r0s2 = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), vrin_r0s2, vmall_r0s2) ;
		vrsum_r0s2 = _ve_vfmads_vvvv(vrsum_r0s2, vrin_r0s2, vrgout) ;

		vrsum_r1s0 = _ve_vfmads_vvvv(vrsum_r1s0, vrin_r1s0, vrgout) ;
		vrsum_r1s1 = _ve_vfmads_vvvv(vrsum_r1s1, vrin_r1s1, vrgout) ;
		vrsum_r1s2 = _ve_vfmads_vvvv(vrsum_r1s2, vrin_r1s2, vrgout) ;

		vrsum_r2s0 = _ve_vfmads_vvvv(vrsum_r2s0, vrin_r2s0, vrgout) ;
		vrsum_r2s1 = _ve_vfmads_vvvv(vrsum_r2s1, vrin_r2s1, vrgout) ;
		vrsum_r2s2 = _ve_vfmads_vvvv(vrsum_r2s2, vrin_r2s2, vrgout) ;
	      }
	      for (y=nY; y+nY<gOutHeight; y+=nY)
	      {
		const int64_t vl = gOutWidth * nY ;
		const int64_t gop = y * gOutWidth ;

		_ve_lvl(vl) ;

		/* memory access errors mihgt be caused (vrin) */
		__vr vrin_r0s0    = _ve_vldu_vss(4,&pInChannel[gop-inWidth-1]) ;
		__vr vrin_r0s1    = _ve_vldu_vss(4,&pInChannel[gop-inWidth  ]) ;
		__vr vrin_r0s2    = _ve_vldu_vss(4,&pInChannel[gop-inWidth+1]) ;
		__vr vrin_r1s0    = _ve_vldu_vss(4,&pInChannel[gop        -1]) ;
		__vr vrin_r1s1    = _ve_vldu_vss(4,&pInChannel[gop          ]) ;
		__vr vrin_r1s2    = _ve_vldu_vss(4,&pInChannel[gop        +1]) ;
		__vr vrin_r2s0    = _ve_vldu_vss(4,&pInChannel[gop+inWidth-1]) ;
		__vr vrin_r2s1    = _ve_vldu_vss(4,&pInChannel[gop+inWidth  ]) ;
		__vr vrin_r2s2    = _ve_vldu_vss(4,&pInChannel[gop+inWidth+1]) ;

		__vr vrgout = _ve_vldu_vss(4, pGOut+gOutIndex+gop+0*gOutPixels) ;

		vrsum_r0s0 = _ve_vfmads_vvvv(vrsum_r0s0, vrin_r0s0, vrgout) ;
		vrsum_r0s1 = _ve_vfmads_vvvv(vrsum_r0s1, vrin_r0s1, vrgout) ;
		vrsum_r0s2 = _ve_vfmads_vvvv(vrsum_r0s2, vrin_r0s2, vrgout) ;

		vrsum_r1s0 = _ve_vfmads_vvvv(vrsum_r1s0, vrin_r1s0, vrgout) ;
		vrsum_r1s1 = _ve_vfmads_vvvv(vrsum_r1s1, vrin_r1s1, vrgout) ;
		vrsum_r1s2 = _ve_vfmads_vvvv(vrsum_r1s2, vrin_r1s2, vrgout) ;

		vrsum_r2s0 = _ve_vfmads_vvvv(vrsum_r2s0, vrin_r2s0, vrgout) ;
		vrsum_r2s1 = _ve_vfmads_vvvv(vrsum_r2s1, vrin_r2s1, vrgout) ;
		vrsum_r2s2 = _ve_vfmads_vvvv(vrsum_r2s2, vrin_r2s2, vrgout) ;
	      }
	      {
		const int64_t vl = gOutWidth * (gOutHeight - y) ;
		const int64_t gop = y * gOutWidth ;

		_ve_lvl(vl) ;

		/* memory access errors mihgt be caused (vrin) */
		__vr vrin_r0s0    = _ve_vldu_vss(4,&pInChannel[gop-inWidth-1]) ;
		__vr vrin_r0s1    = _ve_vldu_vss(4,&pInChannel[gop-inWidth  ]) ;
		__vr vrin_r0s2    = _ve_vldu_vss(4,&pInChannel[gop-inWidth+1]) ;
		__vr vrin_r1s0    = _ve_vldu_vss(4,&pInChannel[gop        -1]) ;
		__vr vrin_r1s1    = _ve_vldu_vss(4,&pInChannel[gop          ]) ;
		__vr vrin_r1s2    = _ve_vldu_vss(4,&pInChannel[gop        +1]) ;
		__vr vrin_r2s0    = _ve_vldu_vss(4,&pInChannel[gop+inWidth-1]) ;
		__vr vrin_r2s1    = _ve_vldu_vss(4,&pInChannel[gop+inWidth  ]) ;
		__vr vrin_r2s2    = _ve_vldu_vss(4,&pInChannel[gop+inWidth+1]) ;

		__vr vrgout = _ve_vldu_vss(4, pGOut+gOutIndex+gop+0*gOutPixels) ;

		vrsum_r0s0 = _ve_vfmads_vvvv(vrsum_r0s0, vrin_r0s0, vrgout) ;
		vrsum_r0s1 = _ve_vfmads_vvvv(vrsum_r0s1, vrin_r0s1, vrgout) ;
		vrsum_r0s2 = _ve_vfmads_vvvv(vrsum_r0s2, vrin_r0s2, vrgout) ;

		vrsum_r1s0 = _ve_vfmads_vvvv(vrsum_r1s0, vrin_r1s0, vrgout) ;
		vrsum_r1s1 = _ve_vfmads_vvvv(vrsum_r1s1, vrin_r1s1, vrgout) ;
		vrsum_r1s2 = _ve_vfmads_vvvv(vrsum_r1s2, vrin_r1s2, vrgout) ;

		vrin_r2s0 = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), vrin_r2s0, vmall_r2s0) ;
		vrsum_r2s0 = _ve_vfmads_vvvv(vrsum_r2s0, vrin_r2s0, vrgout) ;
		vrin_r2s1 = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), vrin_r2s1, vmall_r2s1) ;
		vrsum_r2s1 = _ve_vfmads_vvvv(vrsum_r2s1, vrin_r2s1, vrgout) ;
		vrin_r2s2 = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), vrin_r2s2, vmall_r2s2) ;
		vrsum_r2s2 = _ve_vfmads_vvvv(vrsum_r2s2, vrin_r2s2, vrgout) ;
	      }

	    } // batch

	    _ve_lvl(VLEN) ;
	    vrsum_r0s0 = _ve_vfsums_vvm(vrsum_r0s0, vmw0_s0) ;
	    vrsum_r0s1 = _ve_vfsums_vv(vrsum_r0s1) ;
	    vrsum_r0s2 = _ve_vfsums_vvm(vrsum_r0s2, vmw1_s2) ;
	    vrsum_r1s0 = _ve_vfsums_vvm(vrsum_r1s0, vmw0_s0) ;
	    vrsum_r1s1 = _ve_vfsums_vv(vrsum_r1s1) ;
	    vrsum_r1s2 = _ve_vfsums_vvm(vrsum_r1s2, vmw1_s2) ;
	    vrsum_r2s0 = _ve_vfsums_vvm(vrsum_r2s0, vmw0_s0) ;
	    vrsum_r2s1 = _ve_vfsums_vv(vrsum_r2s1) ;
	    vrsum_r2s2 = _ve_vfsums_vvm(vrsum_r2s2, vmw1_s2) ;

	    _ve_lvl(1) ;
	    _ve_vstu_vss(vrsum_r0s0,4,pGKernel+kernelIndex+0) ;
	    _ve_vstu_vss(vrsum_r0s1,4,pGKernel+kernelIndex+1) ;
	    _ve_vstu_vss(vrsum_r0s2,4,pGKernel+kernelIndex+2) ;
	    _ve_vstu_vss(vrsum_r1s0,4,pGKernel+kernelIndex+3) ;
	    _ve_vstu_vss(vrsum_r1s1,4,pGKernel+kernelIndex+4) ;
	    _ve_vstu_vss(vrsum_r1s2,4,pGKernel+kernelIndex+5) ;
	    _ve_vstu_vss(vrsum_r2s0,4,pGKernel+kernelIndex+6) ;
	    _ve_vstu_vss(vrsum_r2s1,4,pGKernel+kernelIndex+7) ;
	    _ve_vstu_vss(vrsum_r2s2,4,pGKernel+kernelIndex+8) ;
	  } // inChannel


	  k+=1;
	}
	if ( ((nOChannel >> 1) & 0x01) == 1 ) {
	  for (int64_t c=0; c<inChannelGroup; c++) {
	    const int64_t kernelIndex0 = kernGroupOffset + ((k     * inChannelGroup + c) * gKernHeight) * gKernWidth;
	    const int64_t kernelIndex1 = kernGroupOffset + (((k+1) * inChannelGroup + c) * gKernHeight) * gKernWidth;

	    _ve_lvl(VLEN) ;
#define INIT_VRSUM_2(TOKEN, INDEX)	\
		__vr vrsum01_ ## TOKEN = _ve_vbrd_vs_i64(0UL) ;

	    INIT_VRSUM_2(r0s0, 0) ;
	    INIT_VRSUM_2(r0s1, 1) ;
	    INIT_VRSUM_2(r0s2, 2) ;
	    INIT_VRSUM_2(r1s0, 3) ;
	    INIT_VRSUM_2(r1s1, 4) ;
	    INIT_VRSUM_2(r1s2, 5) ;
	    INIT_VRSUM_2(r2s0, 6) ;
	    INIT_VRSUM_2(r2s1, 7) ;
	    INIT_VRSUM_2(r2s2, 8) ;
#undef INIT_VRSUM_2

	    for (int64_t n=0; n<batch; n++)  {
	      const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;
	      const int64_t gOutIndex  = outGroupOffset + ((n * gOutChannel + k  ) * gOutHeight ) * gOutWidth ;

	      int64_t y=0 ;
	      {
		const int64_t vl = gOutWidth * nY ;
		const int64_t gop = y * gOutWidth ;

		_ve_lvl(vl) ;

		const int64_t gOutIndex  = outGroupOffset + ((n * gOutChannel + k  ) * gOutHeight ) * gOutWidth + gop;

		/* memory access errors mihgt be caused (vrin) */
		__vr vrin_r0s0    = _ve_vldu_vss(4,&pInChannel[gop-inWidth-1]) ;
		__vr vrin_r0s1    = _ve_vldu_vss(4,&pInChannel[gop-inWidth  ]) ;
		__vr vrin_r0s2    = _ve_vldu_vss(4,&pInChannel[gop-inWidth+1]) ;
		__vr vrin_r1s0    = _ve_vldu_vss(4,&pInChannel[gop        -1]) ;
		__vr vrin_r1s1    = _ve_vldu_vss(4,&pInChannel[gop          ]) ;
		__vr vrin_r1s2    = _ve_vldu_vss(4,&pInChannel[gop        +1]) ;
		__vr vrin_r2s0    = _ve_vldu_vss(4,&pInChannel[gop+inWidth-1]) ;
		__vr vrin_r2s1    = _ve_vldu_vss(4,&pInChannel[gop+inWidth  ]) ;
		__vr vrin_r2s2    = _ve_vldu_vss(4,&pInChannel[gop+inWidth+1]) ;

		__vr vrgout0 = _ve_vldu_vss(4, pGOut+gOutIndex+gop+0*gOutPixels) ;
		__vr vrgout1 = _ve_vldu_vss(4, pGOut+gOutIndex+gop+1*gOutPixels) ;

		__vr vrgout01 = _ve_vshf_vvvs(vrgout0, vrgout1, VE_VSHUFFLE_YUZU) ;

		vrin_r0s0 = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), vrin_r0s0, vmall_r0s0) ;
		__vr vrinP_r0s0 = _ve_vshf_vvvs(vrin_r0s0, vrin_r0s0, VE_VSHUFFLE_YUZU) ;
		vrsum01_r0s0 = _ve_pvfmad_vvvv(vrsum01_r0s0, vrinP_r0s0, vrgout01) ;

		vrin_r0s1 = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), vrin_r0s1, vmall_r0s1) ;
		__vr vrinP_r0s1 = _ve_vshf_vvvs(vrin_r0s1, vrin_r0s1, VE_VSHUFFLE_YUZU) ;
		vrsum01_r0s1 = _ve_pvfmad_vvvv(vrsum01_r0s1, vrinP_r0s1, vrgout01) ;

		vrin_r0s2 = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), vrin_r0s2, vmall_r0s2) ;
		__vr vrinP_r0s2 = _ve_vshf_vvvs(vrin_r0s2, vrin_r0s2, VE_VSHUFFLE_YUZU) ;
		vrsum01_r0s2 = _ve_pvfmad_vvvv(vrsum01_r0s2, vrinP_r0s2, vrgout01) ;


		__vr vrinP_r1s0 = _ve_vshf_vvvs(vrin_r1s0, vrin_r1s0, VE_VSHUFFLE_YUZU) ;
		vrsum01_r1s0 = _ve_pvfmad_vvvv(vrsum01_r1s0, vrinP_r1s0, vrgout01) ;

		__vr vrinP_r1s1 = _ve_vshf_vvvs(vrin_r1s1, vrin_r1s1, VE_VSHUFFLE_YUZU) ;
		vrsum01_r1s1 = _ve_pvfmad_vvvv(vrsum01_r1s1, vrinP_r1s1, vrgout01) ;

		__vr vrinP_r1s2 = _ve_vshf_vvvs(vrin_r1s2, vrin_r1s2, VE_VSHUFFLE_YUZU) ;
		vrsum01_r1s2 = _ve_pvfmad_vvvv(vrsum01_r1s2, vrinP_r1s2, vrgout01) ;


		__vr vrinP_r2s0 = _ve_vshf_vvvs(vrin_r2s0, vrin_r2s0, VE_VSHUFFLE_YUZU) ;
		vrsum01_r2s0 = _ve_pvfmad_vvvv(vrsum01_r2s0, vrinP_r2s0, vrgout01) ;

		__vr vrinP_r2s1 = _ve_vshf_vvvs(vrin_r2s1, vrin_r2s1, VE_VSHUFFLE_YUZU) ;
		vrsum01_r2s1 = _ve_pvfmad_vvvv(vrsum01_r2s1, vrinP_r2s1, vrgout01) ;

		__vr vrinP_r2s2 = _ve_vshf_vvvs(vrin_r2s2, vrin_r2s2, VE_VSHUFFLE_YUZU) ;
		vrsum01_r2s2 = _ve_pvfmad_vvvv(vrsum01_r2s2, vrinP_r2s2, vrgout01) ;
	      }
	      for (y=nY; y+nY<gOutHeight; y+=nY)
	      {
		const int64_t vl = gOutWidth * nY ;
		const int64_t gop = y * gOutWidth ;

		_ve_lvl(vl) ;

		/* memory access errors mihgt be caused (vrin) */
		__vr vrin_r0s0    = _ve_vldu_vss(4,&pInChannel[gop-inWidth-1]) ;
		__vr vrin_r0s1    = _ve_vldu_vss(4,&pInChannel[gop-inWidth  ]) ;
		__vr vrin_r0s2    = _ve_vldu_vss(4,&pInChannel[gop-inWidth+1]) ;
		__vr vrin_r1s0    = _ve_vldu_vss(4,&pInChannel[gop        -1]) ;
		__vr vrin_r1s1    = _ve_vldu_vss(4,&pInChannel[gop          ]) ;
		__vr vrin_r1s2    = _ve_vldu_vss(4,&pInChannel[gop        +1]) ;
		__vr vrin_r2s0    = _ve_vldu_vss(4,&pInChannel[gop+inWidth-1]) ;
		__vr vrin_r2s1    = _ve_vldu_vss(4,&pInChannel[gop+inWidth  ]) ;
		__vr vrin_r2s2    = _ve_vldu_vss(4,&pInChannel[gop+inWidth+1]) ;

		__vr vrgout0 = _ve_vldu_vss(4, pGOut+gOutIndex+gop+0*gOutPixels) ;
		__vr vrgout1 = _ve_vldu_vss(4, pGOut+gOutIndex+gop+1*gOutPixels) ;

		__vr vrgout01 = _ve_vshf_vvvs(vrgout0, vrgout1, VE_VSHUFFLE_YUZU) ;

		__vr vrinP_r0s0 = _ve_vshf_vvvs(vrin_r0s0, vrin_r0s0, VE_VSHUFFLE_YUZU) ;
		vrsum01_r0s0 = _ve_pvfmad_vvvv(vrsum01_r0s0, vrinP_r0s0, vrgout01) ;

		__vr vrinP_r0s1 = _ve_vshf_vvvs(vrin_r0s1, vrin_r0s1, VE_VSHUFFLE_YUZU) ;
		vrsum01_r0s1 = _ve_pvfmad_vvvv(vrsum01_r0s1, vrinP_r0s1, vrgout01) ;

		__vr vrinP_r0s2 = _ve_vshf_vvvs(vrin_r0s2, vrin_r0s2, VE_VSHUFFLE_YUZU) ;
		vrsum01_r0s2 = _ve_pvfmad_vvvv(vrsum01_r0s2, vrinP_r0s2, vrgout01) ;


		__vr vrinP_r1s0 = _ve_vshf_vvvs(vrin_r1s0, vrin_r1s0, VE_VSHUFFLE_YUZU) ;
		vrsum01_r1s0 = _ve_pvfmad_vvvv(vrsum01_r1s0, vrinP_r1s0, vrgout01) ;

		__vr vrinP_r1s1 = _ve_vshf_vvvs(vrin_r1s1, vrin_r1s1, VE_VSHUFFLE_YUZU) ;
		vrsum01_r1s1 = _ve_pvfmad_vvvv(vrsum01_r1s1, vrinP_r1s1, vrgout01) ;

		__vr vrinP_r1s2 = _ve_vshf_vvvs(vrin_r1s2, vrin_r1s2, VE_VSHUFFLE_YUZU) ;
		vrsum01_r1s2 = _ve_pvfmad_vvvv(vrsum01_r1s2, vrinP_r1s2, vrgout01) ;


		__vr vrinP_r2s0 = _ve_vshf_vvvs(vrin_r2s0, vrin_r2s0, VE_VSHUFFLE_YUZU) ;
		vrsum01_r2s0 = _ve_pvfmad_vvvv(vrsum01_r2s0, vrinP_r2s0, vrgout01) ;

		__vr vrinP_r2s1 = _ve_vshf_vvvs(vrin_r2s1, vrin_r2s1, VE_VSHUFFLE_YUZU) ;
		vrsum01_r2s1 = _ve_pvfmad_vvvv(vrsum01_r2s1, vrinP_r2s1, vrgout01) ;

		__vr vrinP_r2s2 = _ve_vshf_vvvs(vrin_r2s2, vrin_r2s2, VE_VSHUFFLE_YUZU) ;
		vrsum01_r2s2 = _ve_pvfmad_vvvv(vrsum01_r2s2, vrinP_r2s2, vrgout01) ;
	      }
	      {
		const int64_t vl = gOutWidth * (gOutHeight - y) ;
		const int64_t gop = y * gOutWidth ;

		_ve_lvl(vl) ;

		/* memory access errors mihgt be caused (vrin) */
		__vr vrin_r0s0    = _ve_vldu_vss(4,&pInChannel[gop-inWidth-1]) ;
		__vr vrin_r0s1    = _ve_vldu_vss(4,&pInChannel[gop-inWidth  ]) ;
		__vr vrin_r0s2    = _ve_vldu_vss(4,&pInChannel[gop-inWidth+1]) ;
		__vr vrin_r1s0    = _ve_vldu_vss(4,&pInChannel[gop        -1]) ;
		__vr vrin_r1s1    = _ve_vldu_vss(4,&pInChannel[gop          ]) ;
		__vr vrin_r1s2    = _ve_vldu_vss(4,&pInChannel[gop        +1]) ;
		__vr vrin_r2s0    = _ve_vldu_vss(4,&pInChannel[gop+inWidth-1]) ;
		__vr vrin_r2s1    = _ve_vldu_vss(4,&pInChannel[gop+inWidth  ]) ;
		__vr vrin_r2s2    = _ve_vldu_vss(4,&pInChannel[gop+inWidth+1]) ;

		__vr vrgout0 = _ve_vldu_vss(4, pGOut+gOutIndex+gop+0*gOutPixels) ;
		__vr vrgout1 = _ve_vldu_vss(4, pGOut+gOutIndex+gop+1*gOutPixels) ;

		__vr vrgout01 = _ve_vshf_vvvs(vrgout0, vrgout1, VE_VSHUFFLE_YUZU) ;

		__vr vrinP_r0s0 = _ve_vshf_vvvs(vrin_r0s0, vrin_r0s0, VE_VSHUFFLE_YUZU) ;
		vrsum01_r0s0 = _ve_pvfmad_vvvv(vrsum01_r0s0, vrinP_r0s0, vrgout01) ;

		__vr vrinP_r0s1 = _ve_vshf_vvvs(vrin_r0s1, vrin_r0s1, VE_VSHUFFLE_YUZU) ;
		vrsum01_r0s1 = _ve_pvfmad_vvvv(vrsum01_r0s1, vrinP_r0s1, vrgout01) ;

		__vr vrinP_r0s2 = _ve_vshf_vvvs(vrin_r0s2, vrin_r0s2, VE_VSHUFFLE_YUZU) ;
		vrsum01_r0s2 = _ve_pvfmad_vvvv(vrsum01_r0s2, vrinP_r0s2, vrgout01) ;


		__vr vrinP_r1s0 = _ve_vshf_vvvs(vrin_r1s0, vrin_r1s0, VE_VSHUFFLE_YUZU) ;
		vrsum01_r1s0 = _ve_pvfmad_vvvv(vrsum01_r1s0, vrinP_r1s0, vrgout01) ;

		__vr vrinP_r1s1 = _ve_vshf_vvvs(vrin_r1s1, vrin_r1s1, VE_VSHUFFLE_YUZU) ;
		vrsum01_r1s1 = _ve_pvfmad_vvvv(vrsum01_r1s1, vrinP_r1s1, vrgout01) ;

		__vr vrinP_r1s2 = _ve_vshf_vvvs(vrin_r1s2, vrin_r1s2, VE_VSHUFFLE_YUZU) ;
		vrsum01_r1s2 = _ve_pvfmad_vvvv(vrsum01_r1s2, vrinP_r1s2, vrgout01) ;


		vrin_r2s0 = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), vrin_r2s0, vmall_r2s0) ;
		__vr vrinP_r2s0 = _ve_vshf_vvvs(vrin_r2s0, vrin_r2s0, VE_VSHUFFLE_YUZU) ;
		vrsum01_r2s0 = _ve_pvfmad_vvvv(vrsum01_r2s0, vrinP_r2s0, vrgout01) ;

		vrin_r2s1 = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), vrin_r2s1, vmall_r2s1) ;
		__vr vrinP_r2s1 = _ve_vshf_vvvs(vrin_r2s1, vrin_r2s1, VE_VSHUFFLE_YUZU) ;
		vrsum01_r2s1 = _ve_pvfmad_vvvv(vrsum01_r2s1, vrinP_r2s1, vrgout01) ;

		vrin_r2s2 = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), vrin_r2s2, vmall_r2s2) ;
		__vr vrinP_r2s2 = _ve_vshf_vvvs(vrin_r2s2, vrin_r2s2, VE_VSHUFFLE_YUZU) ;
		vrsum01_r2s2 = _ve_pvfmad_vvvv(vrsum01_r2s2, vrinP_r2s2, vrgout01) ;
	      }
	    } // batch

#define VSUM_STORE_3X3_UPPER(VRSUMTOKEN, KERNELINDEX)			\
{									\
  _ve_lvl(VLEN) ;							\
  __vr vrsumU_r0s0 = _ve_vfsums_vvm(VRSUMTOKEN ## _r0s0, vmw0_s0) ;	\
  __vr vrsumU_r0s1 = _ve_vfsums_vv(VRSUMTOKEN ## _r0s1) ;		\
  __vr vrsumU_r0s2 = _ve_vfsums_vvm(VRSUMTOKEN ## _r0s2, vmw1_s2) ;	\
  __vr vrsumU_r1s0 = _ve_vfsums_vvm(VRSUMTOKEN ## _r1s0, vmw0_s0) ;	\
  __vr vrsumU_r1s1 = _ve_vfsums_vv(VRSUMTOKEN ## _r1s1) ;		\
  __vr vrsumU_r1s2 = _ve_vfsums_vvm(VRSUMTOKEN ## _r1s2, vmw1_s2) ;	\
  __vr vrsumU_r2s0 = _ve_vfsums_vvm(VRSUMTOKEN ## _r2s0, vmw0_s0) ;	\
  __vr vrsumU_r2s1 = _ve_vfsums_vv(VRSUMTOKEN ## _r2s1) ;		\
  __vr vrsumU_r2s2 = _ve_vfsums_vvm(VRSUMTOKEN ## _r2s2, vmw1_s2) ;	\
  _ve_lvl(1) ;								\
  _ve_vstu_vss(vrsumU_r0s0,4,pGKernel+(KERNELINDEX)+0) ;		\
  _ve_vstu_vss(vrsumU_r0s1,4,pGKernel+(KERNELINDEX)+1) ;		\
  _ve_vstu_vss(vrsumU_r0s2,4,pGKernel+(KERNELINDEX)+2) ;		\
  _ve_vstu_vss(vrsumU_r1s0,4,pGKernel+(KERNELINDEX)+3) ;		\
  _ve_vstu_vss(vrsumU_r1s1,4,pGKernel+(KERNELINDEX)+4) ;		\
  _ve_vstu_vss(vrsumU_r1s2,4,pGKernel+(KERNELINDEX)+5) ;		\
  _ve_vstu_vss(vrsumU_r2s0,4,pGKernel+(KERNELINDEX)+6) ;		\
  _ve_vstu_vss(vrsumU_r2s1,4,pGKernel+(KERNELINDEX)+7) ;		\
  _ve_vstu_vss(vrsumU_r2s2,4,pGKernel+(KERNELINDEX)+8) ;		\
  }
#define VSUM_STORE_3X3_LOWER(VRSUMTOKEN, KERNELINDEX)					\
  {											\
  _ve_lvl(VLEN) ;									\
  __vr vrsumL_r0s0 = _ve_vfsums_vvm(_ve_vsll_vvs(VRSUMTOKEN ## _r0s0,32), vmw0_s0) ;	\
  __vr vrsumL_r0s1 = _ve_vfsums_vv(_ve_vsll_vvs(VRSUMTOKEN ## _r0s1,32)) ;		\
  __vr vrsumL_r0s2 = _ve_vfsums_vvm(_ve_vsll_vvs(VRSUMTOKEN ## _r0s2,32), vmw1_s2) ;	\
  __vr vrsumL_r1s0 = _ve_vfsums_vvm(_ve_vsll_vvs(VRSUMTOKEN ## _r1s0,32), vmw0_s0) ;	\
  __vr vrsumL_r1s1 = _ve_vfsums_vv(_ve_vsll_vvs(VRSUMTOKEN ## _r1s1,32)) ;		\
  __vr vrsumL_r1s2 = _ve_vfsums_vvm(_ve_vsll_vvs(VRSUMTOKEN ## _r1s2,32), vmw1_s2) ;	\
  __vr vrsumL_r2s0 = _ve_vfsums_vvm(_ve_vsll_vvs(VRSUMTOKEN ## _r2s0,32), vmw0_s0) ;	\
  __vr vrsumL_r2s1 = _ve_vfsums_vv(_ve_vsll_vvs(VRSUMTOKEN ## _r2s1,32)) ;		\
  __vr vrsumL_r2s2 = _ve_vfsums_vvm(_ve_vsll_vvs(VRSUMTOKEN ## _r2s2,32), vmw1_s2) ;	\
  _ve_lvl(1) ;										\
  _ve_vstu_vss(vrsumL_r0s0,4,pGKernel+(KERNELINDEX)+0) ;				\
  _ve_vstu_vss(vrsumL_r0s1,4,pGKernel+(KERNELINDEX)+1) ;				\
  _ve_vstu_vss(vrsumL_r0s2,4,pGKernel+(KERNELINDEX)+2) ;				\
  _ve_vstu_vss(vrsumL_r1s0,4,pGKernel+(KERNELINDEX)+3) ;				\
  _ve_vstu_vss(vrsumL_r1s1,4,pGKernel+(KERNELINDEX)+4) ;				\
  _ve_vstu_vss(vrsumL_r1s2,4,pGKernel+(KERNELINDEX)+5) ;				\
  _ve_vstu_vss(vrsumL_r2s0,4,pGKernel+(KERNELINDEX)+6) ;				\
  _ve_vstu_vss(vrsumL_r2s1,4,pGKernel+(KERNELINDEX)+7) ;				\
  _ve_vstu_vss(vrsumL_r2s2,4,pGKernel+(KERNELINDEX)+8) ;				\
}

	  VSUM_STORE_3X3_UPPER(vrsum01, kernelIndex0) ;
	  VSUM_STORE_3X3_LOWER(vrsum01, kernelIndex1) ;

	  } // inChannel
	  k+=2;
	}
	if ( ((nOChannel >> 2) & 0x01) == 1 ) {
	  for (int64_t c=0; c<inChannelGroup; c++) {
	    const int64_t kernelIndex0 = kernGroupOffset + ((k     * inChannelGroup + c) * gKernHeight) * gKernWidth;
	    const int64_t kernelIndex1 = kernGroupOffset + (((k+1) * inChannelGroup + c) * gKernHeight) * gKernWidth;
	    const int64_t kernelIndex2 = kernGroupOffset + (((k+2) * inChannelGroup + c) * gKernHeight) * gKernWidth;
	    const int64_t kernelIndex3 = kernGroupOffset + (((k+3) * inChannelGroup + c) * gKernHeight) * gKernWidth;

	    _ve_lvl(VLEN) ;
#define INIT_VRSUM_4(TOKEN, INDEX)	\
  __vr vrsum01_ ## TOKEN = _ve_vbrd_vs_i64(0UL) ;	\
  __vr vrsum23_ ## TOKEN = _ve_vbrd_vs_i64(0UL) ;

	    INIT_VRSUM_4(r0s0, 0) ;
	    INIT_VRSUM_4(r0s1, 1) ;
	    INIT_VRSUM_4(r0s2, 2) ;
	    INIT_VRSUM_4(r1s0, 3) ;
	    INIT_VRSUM_4(r1s1, 4) ;
	    INIT_VRSUM_4(r1s2, 5) ;
	    INIT_VRSUM_4(r2s0, 6) ;
	    INIT_VRSUM_4(r2s1, 7) ;
	    INIT_VRSUM_4(r2s2, 8) ;
#undef INIT_VRSUM_4

	    for (int64_t n=0; n<batch; n++) {
	      const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;
	      const int64_t gOutIndex  = outGroupOffset + ((n * gOutChannel + k  ) * gOutHeight ) * gOutWidth ;

	      int64_t y=0 ;
	      {
		const int64_t vl = gOutWidth * nY ;
		const int64_t gop = y * gOutWidth ;

		_ve_lvl(vl) ;

		const int64_t gOutIndex  = outGroupOffset + ((n * gOutChannel + k  ) * gOutHeight ) * gOutWidth + gop;

		/* memory access errors mihgt be caused (vrin) */
		__vr vrin_r0s0    = _ve_vldu_vss(4,&pInChannel[gop-inWidth-1]) ;
		__vr vrin_r0s1    = _ve_vldu_vss(4,&pInChannel[gop-inWidth  ]) ;
		__vr vrin_r0s2    = _ve_vldu_vss(4,&pInChannel[gop-inWidth+1]) ;
		__vr vrin_r1s0    = _ve_vldu_vss(4,&pInChannel[gop        -1]) ;
		__vr vrin_r1s1    = _ve_vldu_vss(4,&pInChannel[gop          ]) ;
		__vr vrin_r1s2    = _ve_vldu_vss(4,&pInChannel[gop        +1]) ;
		__vr vrin_r2s0    = _ve_vldu_vss(4,&pInChannel[gop+inWidth-1]) ;
		__vr vrin_r2s1    = _ve_vldu_vss(4,&pInChannel[gop+inWidth  ]) ;
		__vr vrin_r2s2    = _ve_vldu_vss(4,&pInChannel[gop+inWidth+1]) ;

		__vr vrgout0 = _ve_vldu_vss(4, pGOut+gOutIndex+gop+0*gOutPixels) ;
		__vr vrgout1 = _ve_vldu_vss(4, pGOut+gOutIndex+gop+1*gOutPixels) ;
		__vr vrgout2 = _ve_vldu_vss(4, pGOut+gOutIndex+gop+2*gOutPixels) ;
		__vr vrgout3 = _ve_vldu_vss(4, pGOut+gOutIndex+gop+3*gOutPixels) ;

		__vr vrgout01 = _ve_vshf_vvvs(vrgout0, vrgout1, VE_VSHUFFLE_YUZU) ;
		__vr vrgout23 = _ve_vshf_vvvs(vrgout2, vrgout3, VE_VSHUFFLE_YUZU) ;

		vrin_r0s0 = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), vrin_r0s0, vmall_r0s0) ;
		__vr vrinP_r0s0 = _ve_vshf_vvvs(vrin_r0s0, vrin_r0s0, VE_VSHUFFLE_YUZU) ;
		vrsum01_r0s0 = _ve_pvfmad_vvvv(vrsum01_r0s0, vrinP_r0s0, vrgout01) ;
		vrsum23_r0s0 = _ve_pvfmad_vvvv(vrsum23_r0s0, vrinP_r0s0, vrgout23) ;

		vrin_r0s1 = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), vrin_r0s1, vmall_r0s1) ;
		__vr vrinP_r0s1 = _ve_vshf_vvvs(vrin_r0s1, vrin_r0s1, VE_VSHUFFLE_YUZU) ;
		vrsum01_r0s1 = _ve_pvfmad_vvvv(vrsum01_r0s1, vrinP_r0s1, vrgout01) ;
		vrsum23_r0s1 = _ve_pvfmad_vvvv(vrsum23_r0s1, vrinP_r0s1, vrgout23) ;

		vrin_r0s2 = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), vrin_r0s2, vmall_r0s2) ;
		__vr vrinP_r0s2 = _ve_vshf_vvvs(vrin_r0s2, vrin_r0s2, VE_VSHUFFLE_YUZU) ;
		vrsum01_r0s2 = _ve_pvfmad_vvvv(vrsum01_r0s2, vrinP_r0s2, vrgout01) ;
		vrsum23_r0s2 = _ve_pvfmad_vvvv(vrsum23_r0s2, vrinP_r0s2, vrgout23) ;


		__vr vrinP_r1s0 = _ve_vshf_vvvs(vrin_r1s0, vrin_r1s0, VE_VSHUFFLE_YUZU) ;
		vrsum01_r1s0 = _ve_pvfmad_vvvv(vrsum01_r1s0, vrinP_r1s0, vrgout01) ;
		vrsum23_r1s0 = _ve_pvfmad_vvvv(vrsum23_r1s0, vrinP_r1s0, vrgout23) ;

		__vr vrinP_r1s1 = _ve_vshf_vvvs(vrin_r1s1, vrin_r1s1, VE_VSHUFFLE_YUZU) ;
		vrsum01_r1s1 = _ve_pvfmad_vvvv(vrsum01_r1s1, vrinP_r1s1, vrgout01) ;
		vrsum23_r1s1 = _ve_pvfmad_vvvv(vrsum23_r1s1, vrinP_r1s1, vrgout23) ;

		__vr vrinP_r1s2 = _ve_vshf_vvvs(vrin_r1s2, vrin_r1s2, VE_VSHUFFLE_YUZU) ;
		vrsum01_r1s2 = _ve_pvfmad_vvvv(vrsum01_r1s2, vrinP_r1s2, vrgout01) ;
		vrsum23_r1s2 = _ve_pvfmad_vvvv(vrsum23_r1s2, vrinP_r1s2, vrgout23) ;


		__vr vrinP_r2s0 = _ve_vshf_vvvs(vrin_r2s0, vrin_r2s0, VE_VSHUFFLE_YUZU) ;
		vrsum01_r2s0 = _ve_pvfmad_vvvv(vrsum01_r2s0, vrinP_r2s0, vrgout01) ;
		vrsum23_r2s0 = _ve_pvfmad_vvvv(vrsum23_r2s0, vrinP_r2s0, vrgout23) ;

		__vr vrinP_r2s1 = _ve_vshf_vvvs(vrin_r2s1, vrin_r2s1, VE_VSHUFFLE_YUZU) ;
		vrsum01_r2s1 = _ve_pvfmad_vvvv(vrsum01_r2s1, vrinP_r2s1, vrgout01) ;
		vrsum23_r2s1 = _ve_pvfmad_vvvv(vrsum23_r2s1, vrinP_r2s1, vrgout23) ;

		__vr vrinP_r2s2 = _ve_vshf_vvvs(vrin_r2s2, vrin_r2s2, VE_VSHUFFLE_YUZU) ;
		vrsum01_r2s2 = _ve_pvfmad_vvvv(vrsum01_r2s2, vrinP_r2s2, vrgout01) ;
		vrsum23_r2s2 = _ve_pvfmad_vvvv(vrsum23_r2s2, vrinP_r2s2, vrgout23) ;
	      }
	      for (y=nY; y+nY<gOutHeight; y+=nY)
	      {
		const int64_t vl = gOutWidth * nY ;
		const int64_t gop = y * gOutWidth ;

		_ve_lvl(vl) ;

		/* memory access errors mihgt be caused (vrin) */
		__vr vrin_r0s0    = _ve_vldu_vss(4,&pInChannel[gop-inWidth-1]) ;
		__vr vrin_r0s1    = _ve_vldu_vss(4,&pInChannel[gop-inWidth  ]) ;
		__vr vrin_r0s2    = _ve_vldu_vss(4,&pInChannel[gop-inWidth+1]) ;
		__vr vrin_r1s0    = _ve_vldu_vss(4,&pInChannel[gop        -1]) ;
		__vr vrin_r1s1    = _ve_vldu_vss(4,&pInChannel[gop          ]) ;
		__vr vrin_r1s2    = _ve_vldu_vss(4,&pInChannel[gop        +1]) ;
		__vr vrin_r2s0    = _ve_vldu_vss(4,&pInChannel[gop+inWidth-1]) ;
		__vr vrin_r2s1    = _ve_vldu_vss(4,&pInChannel[gop+inWidth  ]) ;
		__vr vrin_r2s2    = _ve_vldu_vss(4,&pInChannel[gop+inWidth+1]) ;

		__vr vrgout0 = _ve_vldu_vss(4, pGOut+gOutIndex+gop+0*gOutPixels) ;
		__vr vrgout1 = _ve_vldu_vss(4, pGOut+gOutIndex+gop+1*gOutPixels) ;
		__vr vrgout2 = _ve_vldu_vss(4, pGOut+gOutIndex+gop+2*gOutPixels) ;
		__vr vrgout3 = _ve_vldu_vss(4, pGOut+gOutIndex+gop+3*gOutPixels) ;

		__vr vrgout01 = _ve_vshf_vvvs(vrgout0, vrgout1, VE_VSHUFFLE_YUZU) ;
		__vr vrgout23 = _ve_vshf_vvvs(vrgout2, vrgout3, VE_VSHUFFLE_YUZU) ;

		__vr vrinP_r0s0 = _ve_vshf_vvvs(vrin_r0s0, vrin_r0s0, VE_VSHUFFLE_YUZU) ;
		vrsum01_r0s0 = _ve_pvfmad_vvvv(vrsum01_r0s0, vrinP_r0s0, vrgout01) ;
		vrsum23_r0s0 = _ve_pvfmad_vvvv(vrsum23_r0s0, vrinP_r0s0, vrgout23) ;

		__vr vrinP_r0s1 = _ve_vshf_vvvs(vrin_r0s1, vrin_r0s1, VE_VSHUFFLE_YUZU) ;
		vrsum01_r0s1 = _ve_pvfmad_vvvv(vrsum01_r0s1, vrinP_r0s1, vrgout01) ;
		vrsum23_r0s1 = _ve_pvfmad_vvvv(vrsum23_r0s1, vrinP_r0s1, vrgout23) ;

		__vr vrinP_r0s2 = _ve_vshf_vvvs(vrin_r0s2, vrin_r0s2, VE_VSHUFFLE_YUZU) ;
		vrsum01_r0s2 = _ve_pvfmad_vvvv(vrsum01_r0s2, vrinP_r0s2, vrgout01) ;
		vrsum23_r0s2 = _ve_pvfmad_vvvv(vrsum23_r0s2, vrinP_r0s2, vrgout23) ;


		__vr vrinP_r1s0 = _ve_vshf_vvvs(vrin_r1s0, vrin_r1s0, VE_VSHUFFLE_YUZU) ;
		vrsum01_r1s0 = _ve_pvfmad_vvvv(vrsum01_r1s0, vrinP_r1s0, vrgout01) ;
		vrsum23_r1s0 = _ve_pvfmad_vvvv(vrsum23_r1s0, vrinP_r1s0, vrgout23) ;

		__vr vrinP_r1s1 = _ve_vshf_vvvs(vrin_r1s1, vrin_r1s1, VE_VSHUFFLE_YUZU) ;
		vrsum01_r1s1 = _ve_pvfmad_vvvv(vrsum01_r1s1, vrinP_r1s1, vrgout01) ;
		vrsum23_r1s1 = _ve_pvfmad_vvvv(vrsum23_r1s1, vrinP_r1s1, vrgout23) ;

		__vr vrinP_r1s2 = _ve_vshf_vvvs(vrin_r1s2, vrin_r1s2, VE_VSHUFFLE_YUZU) ;
		vrsum01_r1s2 = _ve_pvfmad_vvvv(vrsum01_r1s2, vrinP_r1s2, vrgout01) ;
		vrsum23_r1s2 = _ve_pvfmad_vvvv(vrsum23_r1s2, vrinP_r1s2, vrgout23) ;


		__vr vrinP_r2s0 = _ve_vshf_vvvs(vrin_r2s0, vrin_r2s0, VE_VSHUFFLE_YUZU) ;
		vrsum01_r2s0 = _ve_pvfmad_vvvv(vrsum01_r2s0, vrinP_r2s0, vrgout01) ;
		vrsum23_r2s0 = _ve_pvfmad_vvvv(vrsum23_r2s0, vrinP_r2s0, vrgout23) ;

		__vr vrinP_r2s1 = _ve_vshf_vvvs(vrin_r2s1, vrin_r2s1, VE_VSHUFFLE_YUZU) ;
		vrsum01_r2s1 = _ve_pvfmad_vvvv(vrsum01_r2s1, vrinP_r2s1, vrgout01) ;
		vrsum23_r2s1 = _ve_pvfmad_vvvv(vrsum23_r2s1, vrinP_r2s1, vrgout23) ;

		__vr vrinP_r2s2 = _ve_vshf_vvvs(vrin_r2s2, vrin_r2s2, VE_VSHUFFLE_YUZU) ;
		vrsum01_r2s2 = _ve_pvfmad_vvvv(vrsum01_r2s2, vrinP_r2s2, vrgout01) ;
		vrsum23_r2s2 = _ve_pvfmad_vvvv(vrsum23_r2s2, vrinP_r2s2, vrgout23) ;
	      }
	      {
		const int64_t vl = gOutWidth * (gOutHeight - y) ;
		const int64_t gop = y * gOutWidth ;

		_ve_lvl(vl) ;

		/* memory access errors mihgt be caused (vrin) */
		__vr vrin_r0s0    = _ve_vldu_vss(4,&pInChannel[gop-inWidth-1]) ;
		__vr vrin_r0s1    = _ve_vldu_vss(4,&pInChannel[gop-inWidth  ]) ;
		__vr vrin_r0s2    = _ve_vldu_vss(4,&pInChannel[gop-inWidth+1]) ;
		__vr vrin_r1s0    = _ve_vldu_vss(4,&pInChannel[gop        -1]) ;
		__vr vrin_r1s1    = _ve_vldu_vss(4,&pInChannel[gop          ]) ;
		__vr vrin_r1s2    = _ve_vldu_vss(4,&pInChannel[gop        +1]) ;
		__vr vrin_r2s0    = _ve_vldu_vss(4,&pInChannel[gop+inWidth-1]) ;
		__vr vrin_r2s1    = _ve_vldu_vss(4,&pInChannel[gop+inWidth  ]) ;
		__vr vrin_r2s2    = _ve_vldu_vss(4,&pInChannel[gop+inWidth+1]) ;

		__vr vrgout0 = _ve_vldu_vss(4, pGOut+gOutIndex+gop+0*gOutPixels) ;
		__vr vrgout1 = _ve_vldu_vss(4, pGOut+gOutIndex+gop+1*gOutPixels) ;
		__vr vrgout2 = _ve_vldu_vss(4, pGOut+gOutIndex+gop+2*gOutPixels) ;
		__vr vrgout3 = _ve_vldu_vss(4, pGOut+gOutIndex+gop+3*gOutPixels) ;

		__vr vrgout01 = _ve_vshf_vvvs(vrgout0, vrgout1, VE_VSHUFFLE_YUZU) ;
		__vr vrgout23 = _ve_vshf_vvvs(vrgout2, vrgout3, VE_VSHUFFLE_YUZU) ;

		__vr vrinP_r0s0 = _ve_vshf_vvvs(vrin_r0s0, vrin_r0s0, VE_VSHUFFLE_YUZU) ;
		vrsum01_r0s0 = _ve_pvfmad_vvvv(vrsum01_r0s0, vrinP_r0s0, vrgout01) ;
		vrsum23_r0s0 = _ve_pvfmad_vvvv(vrsum23_r0s0, vrinP_r0s0, vrgout23) ;

		__vr vrinP_r0s1 = _ve_vshf_vvvs(vrin_r0s1, vrin_r0s1, VE_VSHUFFLE_YUZU) ;
		vrsum01_r0s1 = _ve_pvfmad_vvvv(vrsum01_r0s1, vrinP_r0s1, vrgout01) ;
		vrsum23_r0s1 = _ve_pvfmad_vvvv(vrsum23_r0s1, vrinP_r0s1, vrgout23) ;

		__vr vrinP_r0s2 = _ve_vshf_vvvs(vrin_r0s2, vrin_r0s2, VE_VSHUFFLE_YUZU) ;
		vrsum01_r0s2 = _ve_pvfmad_vvvv(vrsum01_r0s2, vrinP_r0s2, vrgout01) ;
		vrsum23_r0s2 = _ve_pvfmad_vvvv(vrsum23_r0s2, vrinP_r0s2, vrgout23) ;


		__vr vrinP_r1s0 = _ve_vshf_vvvs(vrin_r1s0, vrin_r1s0, VE_VSHUFFLE_YUZU) ;
		vrsum01_r1s0 = _ve_pvfmad_vvvv(vrsum01_r1s0, vrinP_r1s0, vrgout01) ;
		vrsum23_r1s0 = _ve_pvfmad_vvvv(vrsum23_r1s0, vrinP_r1s0, vrgout23) ;

		__vr vrinP_r1s1 = _ve_vshf_vvvs(vrin_r1s1, vrin_r1s1, VE_VSHUFFLE_YUZU) ;
		vrsum01_r1s1 = _ve_pvfmad_vvvv(vrsum01_r1s1, vrinP_r1s1, vrgout01) ;
		vrsum23_r1s1 = _ve_pvfmad_vvvv(vrsum23_r1s1, vrinP_r1s1, vrgout23) ;

		__vr vrinP_r1s2 = _ve_vshf_vvvs(vrin_r1s2, vrin_r1s2, VE_VSHUFFLE_YUZU) ;
		vrsum01_r1s2 = _ve_pvfmad_vvvv(vrsum01_r1s2, vrinP_r1s2, vrgout01) ;
		vrsum23_r1s2 = _ve_pvfmad_vvvv(vrsum23_r1s2, vrinP_r1s2, vrgout23) ;


		vrin_r2s0 = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), vrin_r2s0, vmall_r2s0) ;
		__vr vrinP_r2s0 = _ve_vshf_vvvs(vrin_r2s0, vrin_r2s0, VE_VSHUFFLE_YUZU) ;
		vrsum01_r2s0 = _ve_pvfmad_vvvv(vrsum01_r2s0, vrinP_r2s0, vrgout01) ;
		vrsum23_r2s0 = _ve_pvfmad_vvvv(vrsum23_r2s0, vrinP_r2s0, vrgout23) ;

		vrin_r2s1 = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), vrin_r2s1, vmall_r2s1) ;
		__vr vrinP_r2s1 = _ve_vshf_vvvs(vrin_r2s1, vrin_r2s1, VE_VSHUFFLE_YUZU) ;
		vrsum01_r2s1 = _ve_pvfmad_vvvv(vrsum01_r2s1, vrinP_r2s1, vrgout01) ;
		vrsum23_r2s1 = _ve_pvfmad_vvvv(vrsum23_r2s1, vrinP_r2s1, vrgout23) ;

		vrin_r2s2 = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), vrin_r2s2, vmall_r2s2) ;
		__vr vrinP_r2s2 = _ve_vshf_vvvs(vrin_r2s2, vrin_r2s2, VE_VSHUFFLE_YUZU) ;
		vrsum01_r2s2 = _ve_pvfmad_vvvv(vrsum01_r2s2, vrinP_r2s2, vrgout01) ;
		vrsum23_r2s2 = _ve_pvfmad_vvvv(vrsum23_r2s2, vrinP_r2s2, vrgout23) ;
	      }
	    } // batch

	    VSUM_STORE_3X3_UPPER(vrsum01, kernelIndex0) ;
	    VSUM_STORE_3X3_LOWER(vrsum01, kernelIndex1) ;
	    VSUM_STORE_3X3_UPPER(vrsum23, kernelIndex2) ;
	    VSUM_STORE_3X3_LOWER(vrsum23, kernelIndex3) ;

	  } // inChannel

	  k+=4;
	}
	for ( ;k<nOChannel; k+=8) {
	  for (int64_t c=0; c<inChannelGroup; c++) {
	    const int64_t kernelIndex0 = kernGroupOffset + ((k     * inChannelGroup + c) * gKernHeight) * gKernWidth;
	    const int64_t kernelIndex1 = kernGroupOffset + (((k+1) * inChannelGroup + c) * gKernHeight) * gKernWidth;
	    const int64_t kernelIndex2 = kernGroupOffset + (((k+2) * inChannelGroup + c) * gKernHeight) * gKernWidth;
	    const int64_t kernelIndex3 = kernGroupOffset + (((k+3) * inChannelGroup + c) * gKernHeight) * gKernWidth;
	    const int64_t kernelIndex4 = kernGroupOffset + (((k+4) * inChannelGroup + c) * gKernHeight) * gKernWidth;
	    const int64_t kernelIndex5 = kernGroupOffset + (((k+5) * inChannelGroup + c) * gKernHeight) * gKernWidth;
	    const int64_t kernelIndex6 = kernGroupOffset + (((k+6) * inChannelGroup + c) * gKernHeight) * gKernWidth;
	    const int64_t kernelIndex7 = kernGroupOffset + (((k+7) * inChannelGroup + c) * gKernHeight) * gKernWidth;

	    _ve_lvl(VLEN) ;
#define INIT_VRSUM_8(TOKEN, INDEX)	\
  __vr vrsum01_ ## TOKEN = _ve_vbrd_vs_i64(0UL) ;	\
  __vr vrsum23_ ## TOKEN = _ve_vbrd_vs_i64(0UL) ;	\
  __vr vrsum45_ ## TOKEN = _ve_vbrd_vs_i64(0UL) ;	\
  __vr vrsum67_ ## TOKEN = _ve_vbrd_vs_i64(0UL) ;

	    INIT_VRSUM_8(r0s0, 0) ;
	    INIT_VRSUM_8(r0s1, 1) ;
	    INIT_VRSUM_8(r0s2, 2) ;
	    INIT_VRSUM_8(r1s0, 3) ;
	    INIT_VRSUM_8(r1s1, 4) ;
	    INIT_VRSUM_8(r1s2, 5) ;
	    INIT_VRSUM_8(r2s0, 6) ;
	    INIT_VRSUM_8(r2s1, 7) ;
	    INIT_VRSUM_8(r2s2, 8) ;
#undef INIT_VRSUM_8

	    for (int64_t n=0; n<batch; n++) {

	      const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;
	      const int64_t gOutIndex  = outGroupOffset + ((n * gOutChannel + k  ) * gOutHeight ) * gOutWidth ;

	      int64_t y=0 ;
	      {
		const int64_t vl = gOutWidth * nY ;
		const int64_t gop = y * gOutWidth ;

		_ve_lvl(vl) ;

		const int64_t gOutIndex  = outGroupOffset + ((n * gOutChannel + k  ) * gOutHeight ) * gOutWidth + gop;

		/* memory access errors mihgt be caused (vrin) */
		__vr vrin_r0s0    = _ve_vldu_vss(4,&pInChannel[gop-inWidth-1]) ;
		__vr vrin_r0s1    = _ve_vldu_vss(4,&pInChannel[gop-inWidth  ]) ;
		__vr vrin_r0s2    = _ve_vldu_vss(4,&pInChannel[gop-inWidth+1]) ;
		__vr vrin_r1s0    = _ve_vldu_vss(4,&pInChannel[gop        -1]) ;
		__vr vrin_r1s1    = _ve_vldu_vss(4,&pInChannel[gop          ]) ;
		__vr vrin_r1s2    = _ve_vldu_vss(4,&pInChannel[gop        +1]) ;
		__vr vrin_r2s0    = _ve_vldu_vss(4,&pInChannel[gop+inWidth-1]) ;
		__vr vrin_r2s1    = _ve_vldu_vss(4,&pInChannel[gop+inWidth  ]) ;
		__vr vrin_r2s2    = _ve_vldu_vss(4,&pInChannel[gop+inWidth+1]) ;

		__vr vrgout0 = _ve_vldu_vss(4, pGOut+gOutIndex+gop+0*gOutPixels) ;
		__vr vrgout1 = _ve_vldu_vss(4, pGOut+gOutIndex+gop+1*gOutPixels) ;
		__vr vrgout2 = _ve_vldu_vss(4, pGOut+gOutIndex+gop+2*gOutPixels) ;
		__vr vrgout3 = _ve_vldu_vss(4, pGOut+gOutIndex+gop+3*gOutPixels) ;
		__vr vrgout4 = _ve_vldu_vss(4, pGOut+gOutIndex+gop+4*gOutPixels) ;
		__vr vrgout5 = _ve_vldu_vss(4, pGOut+gOutIndex+gop+5*gOutPixels) ;
		__vr vrgout6 = _ve_vldu_vss(4, pGOut+gOutIndex+gop+6*gOutPixels) ;
		__vr vrgout7 = _ve_vldu_vss(4, pGOut+gOutIndex+gop+7*gOutPixels) ;

		__vr vrgout01 = _ve_vshf_vvvs(vrgout0, vrgout1, VE_VSHUFFLE_YUZU) ;
		__vr vrgout23 = _ve_vshf_vvvs(vrgout2, vrgout3, VE_VSHUFFLE_YUZU) ;
		__vr vrgout45 = _ve_vshf_vvvs(vrgout4, vrgout5, VE_VSHUFFLE_YUZU) ;
		__vr vrgout67 = _ve_vshf_vvvs(vrgout6, vrgout7, VE_VSHUFFLE_YUZU) ;

		vrin_r0s0 = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), vrin_r0s0, vmall_r0s0) ;
		__vr vrinP_r0s0 = _ve_vshf_vvvs(vrin_r0s0, vrin_r0s0, VE_VSHUFFLE_YUZU) ;
		vrsum01_r0s0 = _ve_pvfmad_vvvv(vrsum01_r0s0, vrinP_r0s0, vrgout01) ;
		vrsum23_r0s0 = _ve_pvfmad_vvvv(vrsum23_r0s0, vrinP_r0s0, vrgout23) ;
		vrsum45_r0s0 = _ve_pvfmad_vvvv(vrsum45_r0s0, vrinP_r0s0, vrgout45) ;
		vrsum67_r0s0 = _ve_pvfmad_vvvv(vrsum67_r0s0, vrinP_r0s0, vrgout67) ;

		vrin_r0s1 = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), vrin_r0s1, vmall_r0s1) ;
		__vr vrinP_r0s1 = _ve_vshf_vvvs(vrin_r0s1, vrin_r0s1, VE_VSHUFFLE_YUZU) ;
		vrsum01_r0s1 = _ve_pvfmad_vvvv(vrsum01_r0s1, vrinP_r0s1, vrgout01) ;
		vrsum23_r0s1 = _ve_pvfmad_vvvv(vrsum23_r0s1, vrinP_r0s1, vrgout23) ;
		vrsum45_r0s1 = _ve_pvfmad_vvvv(vrsum45_r0s1, vrinP_r0s1, vrgout45) ;
		vrsum67_r0s1 = _ve_pvfmad_vvvv(vrsum67_r0s1, vrinP_r0s1, vrgout67) ;

		vrin_r0s2 = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), vrin_r0s2, vmall_r0s2) ;
		__vr vrinP_r0s2 = _ve_vshf_vvvs(vrin_r0s2, vrin_r0s2, VE_VSHUFFLE_YUZU) ;
		vrsum01_r0s2 = _ve_pvfmad_vvvv(vrsum01_r0s2, vrinP_r0s2, vrgout01) ;
		vrsum23_r0s2 = _ve_pvfmad_vvvv(vrsum23_r0s2, vrinP_r0s2, vrgout23) ;
		vrsum45_r0s2 = _ve_pvfmad_vvvv(vrsum45_r0s2, vrinP_r0s2, vrgout45) ;
		vrsum67_r0s2 = _ve_pvfmad_vvvv(vrsum67_r0s2, vrinP_r0s2, vrgout67) ;


		__vr vrinP_r1s0 = _ve_vshf_vvvs(vrin_r1s0, vrin_r1s0, VE_VSHUFFLE_YUZU) ;
		vrsum01_r1s0 = _ve_pvfmad_vvvv(vrsum01_r1s0, vrinP_r1s0, vrgout01) ;
		vrsum23_r1s0 = _ve_pvfmad_vvvv(vrsum23_r1s0, vrinP_r1s0, vrgout23) ;
		vrsum45_r1s0 = _ve_pvfmad_vvvv(vrsum45_r1s0, vrinP_r1s0, vrgout45) ;
		vrsum67_r1s0 = _ve_pvfmad_vvvv(vrsum67_r1s0, vrinP_r1s0, vrgout67) ;

		__vr vrinP_r1s1 = _ve_vshf_vvvs(vrin_r1s1, vrin_r1s1, VE_VSHUFFLE_YUZU) ;
		vrsum01_r1s1 = _ve_pvfmad_vvvv(vrsum01_r1s1, vrinP_r1s1, vrgout01) ;
		vrsum23_r1s1 = _ve_pvfmad_vvvv(vrsum23_r1s1, vrinP_r1s1, vrgout23) ;
		vrsum45_r1s1 = _ve_pvfmad_vvvv(vrsum45_r1s1, vrinP_r1s1, vrgout45) ;
		vrsum67_r1s1 = _ve_pvfmad_vvvv(vrsum67_r1s1, vrinP_r1s1, vrgout67) ;

		__vr vrinP_r1s2 = _ve_vshf_vvvs(vrin_r1s2, vrin_r1s2, VE_VSHUFFLE_YUZU) ;
		vrsum01_r1s2 = _ve_pvfmad_vvvv(vrsum01_r1s2, vrinP_r1s2, vrgout01) ;
		vrsum23_r1s2 = _ve_pvfmad_vvvv(vrsum23_r1s2, vrinP_r1s2, vrgout23) ;
		vrsum45_r1s2 = _ve_pvfmad_vvvv(vrsum45_r1s2, vrinP_r1s2, vrgout45) ;
		vrsum67_r1s2 = _ve_pvfmad_vvvv(vrsum67_r1s2, vrinP_r1s2, vrgout67) ;


		__vr vrinP_r2s0 = _ve_vshf_vvvs(vrin_r2s0, vrin_r2s0, VE_VSHUFFLE_YUZU) ;
		vrsum01_r2s0 = _ve_pvfmad_vvvv(vrsum01_r2s0, vrinP_r2s0, vrgout01) ;
		vrsum23_r2s0 = _ve_pvfmad_vvvv(vrsum23_r2s0, vrinP_r2s0, vrgout23) ;
		vrsum45_r2s0 = _ve_pvfmad_vvvv(vrsum45_r2s0, vrinP_r2s0, vrgout45) ;
		vrsum67_r2s0 = _ve_pvfmad_vvvv(vrsum67_r2s0, vrinP_r2s0, vrgout67) ;

		__vr vrinP_r2s1 = _ve_vshf_vvvs(vrin_r2s1, vrin_r2s1, VE_VSHUFFLE_YUZU) ;
		vrsum01_r2s1 = _ve_pvfmad_vvvv(vrsum01_r2s1, vrinP_r2s1, vrgout01) ;
		vrsum23_r2s1 = _ve_pvfmad_vvvv(vrsum23_r2s1, vrinP_r2s1, vrgout23) ;
		vrsum45_r2s1 = _ve_pvfmad_vvvv(vrsum45_r2s1, vrinP_r2s1, vrgout45) ;
		vrsum67_r2s1 = _ve_pvfmad_vvvv(vrsum67_r2s1, vrinP_r2s1, vrgout67) ;

		__vr vrinP_r2s2 = _ve_vshf_vvvs(vrin_r2s2, vrin_r2s2, VE_VSHUFFLE_YUZU) ;
		vrsum01_r2s2 = _ve_pvfmad_vvvv(vrsum01_r2s2, vrinP_r2s2, vrgout01) ;
		vrsum23_r2s2 = _ve_pvfmad_vvvv(vrsum23_r2s2, vrinP_r2s2, vrgout23) ;
		vrsum45_r2s2 = _ve_pvfmad_vvvv(vrsum45_r2s2, vrinP_r2s2, vrgout45) ;
		vrsum67_r2s2 = _ve_pvfmad_vvvv(vrsum67_r2s2, vrinP_r2s2, vrgout67) ;
	      }
	      for (y=nY; y+nY<gOutHeight; y+=nY)
	      {
		const int64_t vl = gOutWidth * nY ;
		const int64_t gop = y * gOutWidth ;

		_ve_lvl(vl) ;

		/* memory access errors mihgt be caused (vrin) */
		__vr vrin_r0s0    = _ve_vldu_vss(4,&pInChannel[gop-inWidth-1]) ;
		__vr vrin_r0s1    = _ve_vldu_vss(4,&pInChannel[gop-inWidth  ]) ;
		__vr vrin_r0s2    = _ve_vldu_vss(4,&pInChannel[gop-inWidth+1]) ;
		__vr vrin_r1s0    = _ve_vldu_vss(4,&pInChannel[gop        -1]) ;
		__vr vrin_r1s1    = _ve_vldu_vss(4,&pInChannel[gop          ]) ;
		__vr vrin_r1s2    = _ve_vldu_vss(4,&pInChannel[gop        +1]) ;
		__vr vrin_r2s0    = _ve_vldu_vss(4,&pInChannel[gop+inWidth-1]) ;
		__vr vrin_r2s1    = _ve_vldu_vss(4,&pInChannel[gop+inWidth  ]) ;
		__vr vrin_r2s2    = _ve_vldu_vss(4,&pInChannel[gop+inWidth+1]) ;

		__vr vrgout0 = _ve_vldu_vss(4, pGOut+gOutIndex+gop+0*gOutPixels) ;
		__vr vrgout1 = _ve_vldu_vss(4, pGOut+gOutIndex+gop+1*gOutPixels) ;
		__vr vrgout2 = _ve_vldu_vss(4, pGOut+gOutIndex+gop+2*gOutPixels) ;
		__vr vrgout3 = _ve_vldu_vss(4, pGOut+gOutIndex+gop+3*gOutPixels) ;
		__vr vrgout4 = _ve_vldu_vss(4, pGOut+gOutIndex+gop+4*gOutPixels) ;
		__vr vrgout5 = _ve_vldu_vss(4, pGOut+gOutIndex+gop+5*gOutPixels) ;
		__vr vrgout6 = _ve_vldu_vss(4, pGOut+gOutIndex+gop+6*gOutPixels) ;
		__vr vrgout7 = _ve_vldu_vss(4, pGOut+gOutIndex+gop+7*gOutPixels) ;

		__vr vrgout01 = _ve_vshf_vvvs(vrgout0, vrgout1, VE_VSHUFFLE_YUZU) ;
		__vr vrgout23 = _ve_vshf_vvvs(vrgout2, vrgout3, VE_VSHUFFLE_YUZU) ;
		__vr vrgout45 = _ve_vshf_vvvs(vrgout4, vrgout5, VE_VSHUFFLE_YUZU) ;
		__vr vrgout67 = _ve_vshf_vvvs(vrgout6, vrgout7, VE_VSHUFFLE_YUZU) ;

		__vr vrinP_r0s0 = _ve_vshf_vvvs(vrin_r0s0, vrin_r0s0, VE_VSHUFFLE_YUZU) ;
		vrsum01_r0s0 = _ve_pvfmad_vvvv(vrsum01_r0s0, vrinP_r0s0, vrgout01) ;
		vrsum23_r0s0 = _ve_pvfmad_vvvv(vrsum23_r0s0, vrinP_r0s0, vrgout23) ;
		vrsum45_r0s0 = _ve_pvfmad_vvvv(vrsum45_r0s0, vrinP_r0s0, vrgout45) ;
		vrsum67_r0s0 = _ve_pvfmad_vvvv(vrsum67_r0s0, vrinP_r0s0, vrgout67) ;

		__vr vrinP_r0s1 = _ve_vshf_vvvs(vrin_r0s1, vrin_r0s1, VE_VSHUFFLE_YUZU) ;
		vrsum01_r0s1 = _ve_pvfmad_vvvv(vrsum01_r0s1, vrinP_r0s1, vrgout01) ;
		vrsum23_r0s1 = _ve_pvfmad_vvvv(vrsum23_r0s1, vrinP_r0s1, vrgout23) ;
		vrsum45_r0s1 = _ve_pvfmad_vvvv(vrsum45_r0s1, vrinP_r0s1, vrgout45) ;
		vrsum67_r0s1 = _ve_pvfmad_vvvv(vrsum67_r0s1, vrinP_r0s1, vrgout67) ;

		__vr vrinP_r0s2 = _ve_vshf_vvvs(vrin_r0s2, vrin_r0s2, VE_VSHUFFLE_YUZU) ;
		vrsum01_r0s2 = _ve_pvfmad_vvvv(vrsum01_r0s2, vrinP_r0s2, vrgout01) ;
		vrsum23_r0s2 = _ve_pvfmad_vvvv(vrsum23_r0s2, vrinP_r0s2, vrgout23) ;
		vrsum45_r0s2 = _ve_pvfmad_vvvv(vrsum45_r0s2, vrinP_r0s2, vrgout45) ;
		vrsum67_r0s2 = _ve_pvfmad_vvvv(vrsum67_r0s2, vrinP_r0s2, vrgout67) ;


		__vr vrinP_r1s0 = _ve_vshf_vvvs(vrin_r1s0, vrin_r1s0, VE_VSHUFFLE_YUZU) ;
		vrsum01_r1s0 = _ve_pvfmad_vvvv(vrsum01_r1s0, vrinP_r1s0, vrgout01) ;
		vrsum23_r1s0 = _ve_pvfmad_vvvv(vrsum23_r1s0, vrinP_r1s0, vrgout23) ;
		vrsum45_r1s0 = _ve_pvfmad_vvvv(vrsum45_r1s0, vrinP_r1s0, vrgout45) ;
		vrsum67_r1s0 = _ve_pvfmad_vvvv(vrsum67_r1s0, vrinP_r1s0, vrgout67) ;

		__vr vrinP_r1s1 = _ve_vshf_vvvs(vrin_r1s1, vrin_r1s1, VE_VSHUFFLE_YUZU) ;
		vrsum01_r1s1 = _ve_pvfmad_vvvv(vrsum01_r1s1, vrinP_r1s1, vrgout01) ;
		vrsum23_r1s1 = _ve_pvfmad_vvvv(vrsum23_r1s1, vrinP_r1s1, vrgout23) ;
		vrsum45_r1s1 = _ve_pvfmad_vvvv(vrsum45_r1s1, vrinP_r1s1, vrgout45) ;
		vrsum67_r1s1 = _ve_pvfmad_vvvv(vrsum67_r1s1, vrinP_r1s1, vrgout67) ;

		__vr vrinP_r1s2 = _ve_vshf_vvvs(vrin_r1s2, vrin_r1s2, VE_VSHUFFLE_YUZU) ;
		vrsum01_r1s2 = _ve_pvfmad_vvvv(vrsum01_r1s2, vrinP_r1s2, vrgout01) ;
		vrsum23_r1s2 = _ve_pvfmad_vvvv(vrsum23_r1s2, vrinP_r1s2, vrgout23) ;
		vrsum45_r1s2 = _ve_pvfmad_vvvv(vrsum45_r1s2, vrinP_r1s2, vrgout45) ;
		vrsum67_r1s2 = _ve_pvfmad_vvvv(vrsum67_r1s2, vrinP_r1s2, vrgout67) ;


		__vr vrinP_r2s0 = _ve_vshf_vvvs(vrin_r2s0, vrin_r2s0, VE_VSHUFFLE_YUZU) ;
		vrsum01_r2s0 = _ve_pvfmad_vvvv(vrsum01_r2s0, vrinP_r2s0, vrgout01) ;
		vrsum23_r2s0 = _ve_pvfmad_vvvv(vrsum23_r2s0, vrinP_r2s0, vrgout23) ;
		vrsum45_r2s0 = _ve_pvfmad_vvvv(vrsum45_r2s0, vrinP_r2s0, vrgout45) ;
		vrsum67_r2s0 = _ve_pvfmad_vvvv(vrsum67_r2s0, vrinP_r2s0, vrgout67) ;

		__vr vrinP_r2s1 = _ve_vshf_vvvs(vrin_r2s1, vrin_r2s1, VE_VSHUFFLE_YUZU) ;
		vrsum01_r2s1 = _ve_pvfmad_vvvv(vrsum01_r2s1, vrinP_r2s1, vrgout01) ;
		vrsum23_r2s1 = _ve_pvfmad_vvvv(vrsum23_r2s1, vrinP_r2s1, vrgout23) ;
		vrsum45_r2s1 = _ve_pvfmad_vvvv(vrsum45_r2s1, vrinP_r2s1, vrgout45) ;
		vrsum67_r2s1 = _ve_pvfmad_vvvv(vrsum67_r2s1, vrinP_r2s1, vrgout67) ;

		__vr vrinP_r2s2 = _ve_vshf_vvvs(vrin_r2s2, vrin_r2s2, VE_VSHUFFLE_YUZU) ;
		vrsum01_r2s2 = _ve_pvfmad_vvvv(vrsum01_r2s2, vrinP_r2s2, vrgout01) ;
		vrsum23_r2s2 = _ve_pvfmad_vvvv(vrsum23_r2s2, vrinP_r2s2, vrgout23) ;
		vrsum45_r2s2 = _ve_pvfmad_vvvv(vrsum45_r2s2, vrinP_r2s2, vrgout45) ;
		vrsum67_r2s2 = _ve_pvfmad_vvvv(vrsum67_r2s2, vrinP_r2s2, vrgout67) ;
	      }
	      {
		const int64_t vl = gOutWidth * (gOutHeight - y) ;
		const int64_t gop = y * gOutWidth ;

		_ve_lvl(vl) ;

		/* memory access errors mihgt be caused (vrin) */
		__vr vrin_r0s0    = _ve_vldu_vss(4,&pInChannel[gop-inWidth-1]) ;
		__vr vrin_r0s1    = _ve_vldu_vss(4,&pInChannel[gop-inWidth  ]) ;
		__vr vrin_r0s2    = _ve_vldu_vss(4,&pInChannel[gop-inWidth+1]) ;
		__vr vrin_r1s0    = _ve_vldu_vss(4,&pInChannel[gop        -1]) ;
		__vr vrin_r1s1    = _ve_vldu_vss(4,&pInChannel[gop          ]) ;
		__vr vrin_r1s2    = _ve_vldu_vss(4,&pInChannel[gop        +1]) ;
		__vr vrin_r2s0    = _ve_vldu_vss(4,&pInChannel[gop+inWidth-1]) ;
		__vr vrin_r2s1    = _ve_vldu_vss(4,&pInChannel[gop+inWidth  ]) ;
		__vr vrin_r2s2    = _ve_vldu_vss(4,&pInChannel[gop+inWidth+1]) ;

		__vr vrgout0 = _ve_vldu_vss(4, pGOut+gOutIndex+gop+0*gOutPixels) ;
		__vr vrgout1 = _ve_vldu_vss(4, pGOut+gOutIndex+gop+1*gOutPixels) ;
		__vr vrgout2 = _ve_vldu_vss(4, pGOut+gOutIndex+gop+2*gOutPixels) ;
		__vr vrgout3 = _ve_vldu_vss(4, pGOut+gOutIndex+gop+3*gOutPixels) ;
		__vr vrgout4 = _ve_vldu_vss(4, pGOut+gOutIndex+gop+4*gOutPixels) ;
		__vr vrgout5 = _ve_vldu_vss(4, pGOut+gOutIndex+gop+5*gOutPixels) ;
		__vr vrgout6 = _ve_vldu_vss(4, pGOut+gOutIndex+gop+6*gOutPixels) ;
		__vr vrgout7 = _ve_vldu_vss(4, pGOut+gOutIndex+gop+7*gOutPixels) ;

		__vr vrgout01 = _ve_vshf_vvvs(vrgout0, vrgout1, VE_VSHUFFLE_YUZU) ;
		__vr vrgout23 = _ve_vshf_vvvs(vrgout2, vrgout3, VE_VSHUFFLE_YUZU) ;
		__vr vrgout45 = _ve_vshf_vvvs(vrgout4, vrgout5, VE_VSHUFFLE_YUZU) ;
		__vr vrgout67 = _ve_vshf_vvvs(vrgout6, vrgout7, VE_VSHUFFLE_YUZU) ;

		__vr vrinP_r0s0 = _ve_vshf_vvvs(vrin_r0s0, vrin_r0s0, VE_VSHUFFLE_YUZU) ;
		vrsum01_r0s0 = _ve_pvfmad_vvvv(vrsum01_r0s0, vrinP_r0s0, vrgout01) ;
		vrsum23_r0s0 = _ve_pvfmad_vvvv(vrsum23_r0s0, vrinP_r0s0, vrgout23) ;
		vrsum45_r0s0 = _ve_pvfmad_vvvv(vrsum45_r0s0, vrinP_r0s0, vrgout45) ;
		vrsum67_r0s0 = _ve_pvfmad_vvvv(vrsum67_r0s0, vrinP_r0s0, vrgout67) ;

		__vr vrinP_r0s1 = _ve_vshf_vvvs(vrin_r0s1, vrin_r0s1, VE_VSHUFFLE_YUZU) ;
		vrsum01_r0s1 = _ve_pvfmad_vvvv(vrsum01_r0s1, vrinP_r0s1, vrgout01) ;
		vrsum23_r0s1 = _ve_pvfmad_vvvv(vrsum23_r0s1, vrinP_r0s1, vrgout23) ;
		vrsum45_r0s1 = _ve_pvfmad_vvvv(vrsum45_r0s1, vrinP_r0s1, vrgout45) ;
		vrsum67_r0s1 = _ve_pvfmad_vvvv(vrsum67_r0s1, vrinP_r0s1, vrgout67) ;

		__vr vrinP_r0s2 = _ve_vshf_vvvs(vrin_r0s2, vrin_r0s2, VE_VSHUFFLE_YUZU) ;
		vrsum01_r0s2 = _ve_pvfmad_vvvv(vrsum01_r0s2, vrinP_r0s2, vrgout01) ;
		vrsum23_r0s2 = _ve_pvfmad_vvvv(vrsum23_r0s2, vrinP_r0s2, vrgout23) ;
		vrsum45_r0s2 = _ve_pvfmad_vvvv(vrsum45_r0s2, vrinP_r0s2, vrgout45) ;
		vrsum67_r0s2 = _ve_pvfmad_vvvv(vrsum67_r0s2, vrinP_r0s2, vrgout67) ;


		__vr vrinP_r1s0 = _ve_vshf_vvvs(vrin_r1s0, vrin_r1s0, VE_VSHUFFLE_YUZU) ;
		vrsum01_r1s0 = _ve_pvfmad_vvvv(vrsum01_r1s0, vrinP_r1s0, vrgout01) ;
		vrsum23_r1s0 = _ve_pvfmad_vvvv(vrsum23_r1s0, vrinP_r1s0, vrgout23) ;
		vrsum45_r1s0 = _ve_pvfmad_vvvv(vrsum45_r1s0, vrinP_r1s0, vrgout45) ;
		vrsum67_r1s0 = _ve_pvfmad_vvvv(vrsum67_r1s0, vrinP_r1s0, vrgout67) ;

		__vr vrinP_r1s1 = _ve_vshf_vvvs(vrin_r1s1, vrin_r1s1, VE_VSHUFFLE_YUZU) ;
		vrsum01_r1s1 = _ve_pvfmad_vvvv(vrsum01_r1s1, vrinP_r1s1, vrgout01) ;
		vrsum23_r1s1 = _ve_pvfmad_vvvv(vrsum23_r1s1, vrinP_r1s1, vrgout23) ;
		vrsum45_r1s1 = _ve_pvfmad_vvvv(vrsum45_r1s1, vrinP_r1s1, vrgout45) ;
		vrsum67_r1s1 = _ve_pvfmad_vvvv(vrsum67_r1s1, vrinP_r1s1, vrgout67) ;

		__vr vrinP_r1s2 = _ve_vshf_vvvs(vrin_r1s2, vrin_r1s2, VE_VSHUFFLE_YUZU) ;
		vrsum01_r1s2 = _ve_pvfmad_vvvv(vrsum01_r1s2, vrinP_r1s2, vrgout01) ;
		vrsum23_r1s2 = _ve_pvfmad_vvvv(vrsum23_r1s2, vrinP_r1s2, vrgout23) ;
		vrsum45_r1s2 = _ve_pvfmad_vvvv(vrsum45_r1s2, vrinP_r1s2, vrgout45) ;
		vrsum67_r1s2 = _ve_pvfmad_vvvv(vrsum67_r1s2, vrinP_r1s2, vrgout67) ;


		vrin_r2s0 = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), vrin_r2s0, vmall_r2s0) ;
		__vr vrinP_r2s0 = _ve_vshf_vvvs(vrin_r2s0, vrin_r2s0, VE_VSHUFFLE_YUZU) ;
		vrsum01_r2s0 = _ve_pvfmad_vvvv(vrsum01_r2s0, vrinP_r2s0, vrgout01) ;
		vrsum23_r2s0 = _ve_pvfmad_vvvv(vrsum23_r2s0, vrinP_r2s0, vrgout23) ;
		vrsum45_r2s0 = _ve_pvfmad_vvvv(vrsum45_r2s0, vrinP_r2s0, vrgout45) ;
		vrsum67_r2s0 = _ve_pvfmad_vvvv(vrsum67_r2s0, vrinP_r2s0, vrgout67) ;

		vrin_r2s1 = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), vrin_r2s1, vmall_r2s1) ;
		__vr vrinP_r2s1 = _ve_vshf_vvvs(vrin_r2s1, vrin_r2s1, VE_VSHUFFLE_YUZU) ;
		vrsum01_r2s1 = _ve_pvfmad_vvvv(vrsum01_r2s1, vrinP_r2s1, vrgout01) ;
		vrsum23_r2s1 = _ve_pvfmad_vvvv(vrsum23_r2s1, vrinP_r2s1, vrgout23) ;
		vrsum45_r2s1 = _ve_pvfmad_vvvv(vrsum45_r2s1, vrinP_r2s1, vrgout45) ;
		vrsum67_r2s1 = _ve_pvfmad_vvvv(vrsum67_r2s1, vrinP_r2s1, vrgout67) ;

		vrin_r2s2 = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), vrin_r2s2, vmall_r2s2) ;
		__vr vrinP_r2s2 = _ve_vshf_vvvs(vrin_r2s2, vrin_r2s2, VE_VSHUFFLE_YUZU) ;
		vrsum01_r2s2 = _ve_pvfmad_vvvv(vrsum01_r2s2, vrinP_r2s2, vrgout01) ;
		vrsum23_r2s2 = _ve_pvfmad_vvvv(vrsum23_r2s2, vrinP_r2s2, vrgout23) ;
		vrsum45_r2s2 = _ve_pvfmad_vvvv(vrsum45_r2s2, vrinP_r2s2, vrgout45) ;
		vrsum67_r2s2 = _ve_pvfmad_vvvv(vrsum67_r2s2, vrinP_r2s2, vrgout67) ;
	      }
	    } // batch

	    VSUM_STORE_3X3_UPPER(vrsum01, kernelIndex0) ;
	    VSUM_STORE_3X3_LOWER(vrsum01, kernelIndex1) ;
	    VSUM_STORE_3X3_UPPER(vrsum23, kernelIndex2) ;
	    VSUM_STORE_3X3_LOWER(vrsum23, kernelIndex3) ;
	    VSUM_STORE_3X3_UPPER(vrsum45, kernelIndex4) ;
	    VSUM_STORE_3X3_LOWER(vrsum45, kernelIndex5) ;
	    VSUM_STORE_3X3_UPPER(vrsum67, kernelIndex6) ;
	    VSUM_STORE_3X3_LOWER(vrsum67, kernelIndex7) ;

#undef VSUM_STORE_3X3_UPPER
#undef VSUM_STORE_3X3_LOWER

	  } // inChannel
	} // outChannel
      } // group
    }
  }


  return VEDNN_SUCCESS;
}
