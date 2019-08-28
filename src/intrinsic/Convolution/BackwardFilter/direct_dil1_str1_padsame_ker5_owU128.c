#include <stdint.h>
#include <stdlib.h>

#include "vednn.h"

#include "velintrin.h"
#define VLEN	(256)

vednnError_t
vednnConvolutionBackwardFilter_direct_dil1_str1_padsame_ker5_owU128(
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
  const int64_t gKernWidth  = pParamGradKernel->width;		/* must be 2*padWidth  + 1 */
  const int64_t gKernHeight = pParamGradKernel->height;		/* must be 2*padHeight + 1 */

  const int64_t group          = pParamConv->group;
//  const int64_t strideWidth    = pParamConv->strideWidth;	/* must be 1 */
//  const int64_t strideHeight   = pParamConv->strideHeight;	/* must be 1 */
  const int64_t padWidth       = pParamConv->padWidth;
  const int64_t padHeight      = pParamConv->padHeight;
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

    __vr vrseq = _vel_vseq_vl(VLEN) ;			// xy

    __vr vry   = _vel_vdivsl_vvsl(vrseq, gOutWidth, VLEN) ;
    __vr vrx   = _vel_vsubsl_vvvl(vrseq, _vel_vmulul_vsvl(gOutWidth,vry, VLEN), VLEN) ;

    __vm256 vmw_s0 =  _vel_vfmklge_mvl(_vel_vaddsl_vsvl(-2, vrx, VLEN), VLEN) ;		// condition(  2<=x)
    __vm256 vmw_s1 =  _vel_vfmklge_mvl(_vel_vaddsl_vsvl(-1, vrx, VLEN), VLEN) ;		// condition(  2<=x)
    __vm256 vmw_s3 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(inWidth-1,vrx, VLEN), VLEN) ;	// condition(x+1< inWidth)
    __vm256 vmw_s4 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(inWidth-2,vrx, VLEN), VLEN) ;	// condition(x+2< inWidth)

    for (int64_t g = 0; g < group; g++) {
      int64_t inGroupOffset   = g * inChannelGroup  * inHeight  * inWidth;
      int64_t outGroupOffset  = (g * gOutChannelGroup + beginOChannel) * gOutHeight * gOutWidth;
      int64_t kernGroupOffset = (g * gOutChannelGroup + beginOChannel) * inChannelGroup * gKernHeight * gKernWidth;

      int64_t k = 0 ;
	if ( (nOChannel & 0x01) == 1 ) {
	for (int64_t c=0; c<inChannelGroup; c++) {
	  const int64_t kernelIndex = kernGroupOffset + ((k     * inChannelGroup + c) * gKernHeight ) * gKernWidth ;

	  __vr vrsum_r0s0 = _vel_vbrds_vsl(0.0f, VLEN) ;
	  __vr vrsum_r0s1 = _vel_vbrds_vsl(0.0f, VLEN) ;
	  __vr vrsum_r0s2 = _vel_vbrds_vsl(0.0f, VLEN) ;
	  __vr vrsum_r0s3 = _vel_vbrds_vsl(0.0f, VLEN) ;
	  __vr vrsum_r0s4 = _vel_vbrds_vsl(0.0f, VLEN) ;

	  __vr vrsum_r1s0 = _vel_vbrds_vsl(0.0f, VLEN) ;
	  __vr vrsum_r1s1 = _vel_vbrds_vsl(0.0f, VLEN) ;
	  __vr vrsum_r1s2 = _vel_vbrds_vsl(0.0f, VLEN) ;
	  __vr vrsum_r1s3 = _vel_vbrds_vsl(0.0f, VLEN) ;
	  __vr vrsum_r1s4 = _vel_vbrds_vsl(0.0f, VLEN) ;

	  __vr vrsum_r2s0 = _vel_vbrds_vsl(0.0f, VLEN) ;
	  __vr vrsum_r2s1 = _vel_vbrds_vsl(0.0f, VLEN) ;
	  __vr vrsum_r2s2 = _vel_vbrds_vsl(0.0f, VLEN) ;
	  __vr vrsum_r2s3 = _vel_vbrds_vsl(0.0f, VLEN) ;
	  __vr vrsum_r2s4 = _vel_vbrds_vsl(0.0f, VLEN) ;

	  __vr vrsum_r3s0 = _vel_vbrds_vsl(0.0f, VLEN) ;
	  __vr vrsum_r3s1 = _vel_vbrds_vsl(0.0f, VLEN) ;
	  __vr vrsum_r3s2 = _vel_vbrds_vsl(0.0f, VLEN) ;
	  __vr vrsum_r3s3 = _vel_vbrds_vsl(0.0f, VLEN) ;
	  __vr vrsum_r3s4 = _vel_vbrds_vsl(0.0f, VLEN) ;

	  __vr vrsum_r4s0 = _vel_vbrds_vsl(0.0f, VLEN) ;
	  __vr vrsum_r4s1 = _vel_vbrds_vsl(0.0f, VLEN) ;
	  __vr vrsum_r4s2 = _vel_vbrds_vsl(0.0f, VLEN) ;
	  __vr vrsum_r4s3 = _vel_vbrds_vsl(0.0f, VLEN) ;
	  __vr vrsum_r4s4 = _vel_vbrds_vsl(0.0f, VLEN) ;


	  for (int64_t n=0; n<batch; n++) {
	    for (int64_t y=0; y<gOutHeight; y+=nY)
	    {
	      const int64_t vl = gOutWidth * (gOutHeight - y < nY ? gOutHeight - y : nY) ;
	      const int64_t gop = y * gOutWidth ;

	      __vm256 vmh0_r0 =  _vel_vfmklge_mvl(_vel_vaddsl_vsvl(y-2, vry, vl), vl) ;		// condition(0 <= h)
	      __vm256 vmh0_r1 =  _vel_vfmklge_mvl(_vel_vaddsl_vsvl(y-1, vry, vl), vl) ;		// condition(0 <= h)
	      __vm256 vmh1_r3 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(inHeight-1-y,vry, vl), vl) ;	// condition(h < inHeight)
	      __vm256 vmh1_r4 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(inHeight-2-y,vry, vl), vl) ;	// condition(h < inHeight)

	      __vm256 vmh_r0  = vmh0_r0 ;
	      __vm256 vmh_r1  = vmh0_r1 ;
	      __vm256 vmh_r3  = vmh1_r3 ;
	      __vm256 vmh_r4  = vmh1_r4 ;

	      __vm256 vmall_r0s0 = _vel_andm_mmm(vmh_r0, vmw_s0) ;
	      __vm256 vmall_r0s1 = _vel_andm_mmm(vmh_r0, vmw_s1) ;
	      __vm256 vmall_r0s2 = vmh_r0 ;
	      __vm256 vmall_r0s3 = _vel_andm_mmm(vmh_r0, vmw_s3) ;
	      __vm256 vmall_r0s4 = _vel_andm_mmm(vmh_r0, vmw_s4) ;

	      __vm256 vmall_r1s0 = _vel_andm_mmm(vmh_r1, vmw_s0) ;
	      __vm256 vmall_r1s1 = _vel_andm_mmm(vmh_r1, vmw_s1) ;
	      __vm256 vmall_r1s2 = vmh_r1 ;
	      __vm256 vmall_r1s3 = _vel_andm_mmm(vmh_r1, vmw_s3) ;
	      __vm256 vmall_r1s4 = _vel_andm_mmm(vmh_r1, vmw_s4) ;

	      __vm256 vmall_r2s0 = vmw_s0 ;
	      __vm256 vmall_r2s1 = vmw_s1 ;
	      __vm256 vmall_r2s3 = vmw_s3 ;
	      __vm256 vmall_r2s4 = vmw_s4 ;

	      __vm256 vmall_r3s0 = _vel_andm_mmm(vmh_r3, vmw_s0) ;
	      __vm256 vmall_r3s1 = _vel_andm_mmm(vmh_r3, vmw_s1) ;
	      __vm256 vmall_r3s2 = vmh_r3 ;
	      __vm256 vmall_r3s3 = _vel_andm_mmm(vmh_r3, vmw_s3) ;
	      __vm256 vmall_r3s4 = _vel_andm_mmm(vmh_r3, vmw_s4) ;

	      __vm256 vmall_r4s0 = _vel_andm_mmm(vmh_r4, vmw_s0) ;
	      __vm256 vmall_r4s1 = _vel_andm_mmm(vmh_r4, vmw_s1) ;
	      __vm256 vmall_r4s2 = vmh_r4 ;
	      __vm256 vmall_r4s3 = _vel_andm_mmm(vmh_r4, vmw_s3) ;
	      __vm256 vmall_r4s4 = _vel_andm_mmm(vmh_r4, vmw_s4) ;

	      const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;
	      const int64_t gOutIndex  = outGroupOffset + ((n * gOutChannel + k  ) * gOutHeight ) * gOutWidth + gop;

	      /* memory access errors mihgt be caused (vrin) */
	      __vr vrin_r0s0    = _vel_vldu_vssl(4,&pInChannel[gop-2*inWidth-2], vl) ;
	      __vr vrin_r0s1    = _vel_vldu_vssl(4,&pInChannel[gop-2*inWidth-1], vl) ;
	      __vr vrin_r0s2    = _vel_vldu_vssl(4,&pInChannel[gop-2*inWidth  ], vl) ;
	      __vr vrin_r0s3    = _vel_vldu_vssl(4,&pInChannel[gop-2*inWidth+1], vl) ;
	      __vr vrin_r0s4    = _vel_vldu_vssl(4,&pInChannel[gop-2*inWidth+2], vl) ;
	      __vr vrin_r1s0    = _vel_vldu_vssl(4,&pInChannel[gop-  inWidth-2], vl) ;
	      __vr vrin_r1s1    = _vel_vldu_vssl(4,&pInChannel[gop-  inWidth-1], vl) ;
	      __vr vrin_r1s2    = _vel_vldu_vssl(4,&pInChannel[gop-  inWidth  ], vl) ;
	      __vr vrin_r1s3    = _vel_vldu_vssl(4,&pInChannel[gop-  inWidth+1], vl) ;
	      __vr vrin_r1s4    = _vel_vldu_vssl(4,&pInChannel[gop-  inWidth+2], vl) ;
	      __vr vrin_r2s0    = _vel_vldu_vssl(4,&pInChannel[gop          -2], vl) ;
	      __vr vrin_r2s1    = _vel_vldu_vssl(4,&pInChannel[gop          -1], vl) ;
	      __vr vrin_r2s2    = _vel_vldu_vssl(4,&pInChannel[gop            ], vl) ;
	      __vr vrin_r2s3    = _vel_vldu_vssl(4,&pInChannel[gop          +1], vl) ;
	      __vr vrin_r2s4    = _vel_vldu_vssl(4,&pInChannel[gop          +2], vl) ;
	      __vr vrin_r3s0    = _vel_vldu_vssl(4,&pInChannel[gop+  inWidth-2], vl) ;
	      __vr vrin_r3s1    = _vel_vldu_vssl(4,&pInChannel[gop+  inWidth-1], vl) ;
	      __vr vrin_r3s2    = _vel_vldu_vssl(4,&pInChannel[gop+  inWidth  ], vl) ;
	      __vr vrin_r3s3    = _vel_vldu_vssl(4,&pInChannel[gop+  inWidth+1], vl) ;
	      __vr vrin_r3s4    = _vel_vldu_vssl(4,&pInChannel[gop+  inWidth+2], vl) ;
	      __vr vrin_r4s0    = _vel_vldu_vssl(4,&pInChannel[gop+2*inWidth-2], vl) ;
	      __vr vrin_r4s1    = _vel_vldu_vssl(4,&pInChannel[gop+2*inWidth-1], vl) ;
	      __vr vrin_r4s2    = _vel_vldu_vssl(4,&pInChannel[gop+2*inWidth  ], vl) ;
	      __vr vrin_r4s3    = _vel_vldu_vssl(4,&pInChannel[gop+2*inWidth+1], vl) ;
	      __vr vrin_r4s4    = _vel_vldu_vssl(4,&pInChannel[gop+2*inWidth+2], vl) ;

	      __vr vrgout = _vel_vldu_vssl(4, pGOut+gOutIndex+0*gOutPixels, vl) ;

#define VFMAD1(VRSUM, VRIN, VMR) {					\
  VRIN = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), VRIN, VMR, vl) ;	\
  VRSUM = _vel_vfmads_vvvvvl(VRSUM, VRIN, vrgout, VRSUM, vl) ;		\
}

	      VFMAD1(vrsum_r0s0, vrin_r0s0, vmall_r0s0) ;
	      VFMAD1(vrsum_r0s1, vrin_r0s1, vmall_r0s1) ;
	      VFMAD1(vrsum_r0s2, vrin_r0s2, vmall_r0s2) ;
	      VFMAD1(vrsum_r0s3, vrin_r0s3, vmall_r0s3) ;
	      VFMAD1(vrsum_r0s4, vrin_r0s4, vmall_r0s4) ;

	      VFMAD1(vrsum_r1s0, vrin_r1s0, vmall_r1s0) ;
	      VFMAD1(vrsum_r1s1, vrin_r1s1, vmall_r1s1) ;
	      VFMAD1(vrsum_r1s2, vrin_r1s2, vmall_r1s2) ;
	      VFMAD1(vrsum_r1s3, vrin_r1s3, vmall_r1s3) ;
	      VFMAD1(vrsum_r1s4, vrin_r1s4, vmall_r1s4) ;

	      VFMAD1(vrsum_r2s0, vrin_r2s0, vmall_r2s0) ;
	      VFMAD1(vrsum_r2s1, vrin_r2s1, vmall_r2s1) ;
	      vrsum_r2s2 = _vel_vfmads_vvvvvl(vrsum_r2s2, vrin_r2s2, vrgout, vrsum_r2s2, vl) ;
	      VFMAD1(vrsum_r2s3, vrin_r2s3, vmall_r2s3) ;
	      VFMAD1(vrsum_r2s4, vrin_r2s4, vmall_r2s4) ;

	      VFMAD1(vrsum_r3s0, vrin_r3s0, vmall_r3s0) ;
	      VFMAD1(vrsum_r3s1, vrin_r3s1, vmall_r3s1) ;
	      VFMAD1(vrsum_r3s2, vrin_r3s2, vmall_r3s2) ;
	      VFMAD1(vrsum_r3s3, vrin_r3s3, vmall_r3s3) ;
	      VFMAD1(vrsum_r3s4, vrin_r3s4, vmall_r3s4) ;

	      VFMAD1(vrsum_r4s0, vrin_r4s0, vmall_r4s0) ;
	      VFMAD1(vrsum_r4s1, vrin_r4s1, vmall_r4s1) ;
	      VFMAD1(vrsum_r4s2, vrin_r4s2, vmall_r4s2) ;
	      VFMAD1(vrsum_r4s3, vrin_r4s3, vmall_r4s3) ;
	      VFMAD1(vrsum_r4s4, vrin_r4s4, vmall_r4s4) ;
#undef VFMAD1

	    } // gOutPixels
	  } // batch

#define VSUM_STORE_5(VRSUMTOKEN, KERNELINDEX)			\
{								\
  VRSUMTOKEN ## s0 = _vel_vfsums_vvl(VRSUMTOKEN ## s0, VLEN) ;	\
  VRSUMTOKEN ## s1 = _vel_vfsums_vvl(VRSUMTOKEN ## s1, VLEN) ;	\
  VRSUMTOKEN ## s2 = _vel_vfsums_vvl(VRSUMTOKEN ## s2, VLEN) ;	\
  VRSUMTOKEN ## s3 = _vel_vfsums_vvl(VRSUMTOKEN ## s3, VLEN) ;	\
  VRSUMTOKEN ## s4 = _vel_vfsums_vvl(VRSUMTOKEN ## s4, VLEN) ;	\
  _vel_vstu_vssl(VRSUMTOKEN ## s0,4,pGKernel+(KERNELINDEX)+0, 1) ;	\
  _vel_vstu_vssl(VRSUMTOKEN ## s1,4,pGKernel+(KERNELINDEX)+1, 1) ;	\
  _vel_vstu_vssl(VRSUMTOKEN ## s2,4,pGKernel+(KERNELINDEX)+2, 1) ;	\
  _vel_vstu_vssl(VRSUMTOKEN ## s3,4,pGKernel+(KERNELINDEX)+3, 1) ;	\
  _vel_vstu_vssl(VRSUMTOKEN ## s4,4,pGKernel+(KERNELINDEX)+4, 1) ;	\
}

	  VSUM_STORE_5(vrsum_r0, kernelIndex) ;
	  VSUM_STORE_5(vrsum_r1, kernelIndex+5) ;
	  VSUM_STORE_5(vrsum_r2, kernelIndex+10) ;
	  VSUM_STORE_5(vrsum_r3, kernelIndex+15) ;
	  VSUM_STORE_5(vrsum_r4, kernelIndex+20) ;

#undef VSUM_STORE_5

	} // inChannel

	k+=1;
      }
      if ( ((nOChannel >> 1) & 0x01) == 1 ) {
	for (int64_t c=0; c<inChannelGroup; c++) {
	  const int64_t kernelIndex0 = kernGroupOffset + ((k     * inChannelGroup + c) * gKernHeight) * gKernWidth ;
	  const int64_t kernelIndex1 = kernGroupOffset + (((k+1) * inChannelGroup + c) * gKernHeight) * gKernWidth ;

#define INIT_VRSUM_2(TOKEN, INDEX)	\
  __vr vrsum01_ ## TOKEN = _vel_vbrdl_vsl(0UL, VLEN) ;

	  INIT_VRSUM_2(r0s0, 0) ;
	  INIT_VRSUM_2(r0s1, 1) ;
	  INIT_VRSUM_2(r0s2, 2) ;
	  INIT_VRSUM_2(r0s3, 3) ;
	  INIT_VRSUM_2(r0s4, 4) ;
	  INIT_VRSUM_2(r1s0, 5) ;
	  INIT_VRSUM_2(r1s1, 6) ;
	  INIT_VRSUM_2(r1s2, 7) ;
	  INIT_VRSUM_2(r1s3, 8) ;
	  INIT_VRSUM_2(r1s4, 9) ;
	  INIT_VRSUM_2(r2s0,10) ;
	  INIT_VRSUM_2(r2s1,11) ;
	  INIT_VRSUM_2(r2s2,12) ;
	  INIT_VRSUM_2(r2s3,13) ;
	  INIT_VRSUM_2(r2s4,14) ;
	  INIT_VRSUM_2(r3s0,15) ;
	  INIT_VRSUM_2(r3s1,16) ;
	  INIT_VRSUM_2(r3s2,17) ;
	  INIT_VRSUM_2(r3s3,18) ;
	  INIT_VRSUM_2(r3s4,19) ;
	  INIT_VRSUM_2(r4s0,20) ;
	  INIT_VRSUM_2(r4s1,21) ;
	  INIT_VRSUM_2(r4s2,22) ;
	  INIT_VRSUM_2(r4s3,23) ;
	  INIT_VRSUM_2(r4s4,24) ;
#undef INIT_VRSUM_2

	  for (int64_t n=0; n<batch; n++) {
	    for (int64_t y=0; y<gOutHeight; y+=nY)
	    {
	      const int64_t vl = gOutWidth * (gOutHeight - y < nY ? gOutHeight - y : nY) ;
	      const int64_t gop = y * gOutWidth ;

	      __vm256 vmh0_r0 =  _vel_vfmklge_mvl(_vel_vaddsl_vsvl(y-2, vry, vl), vl) ;		// condition(0 <= h)
	      __vm256 vmh0_r1 =  _vel_vfmklge_mvl(_vel_vaddsl_vsvl(y-1, vry, vl), vl) ;		// condition(0 <= h)
	      __vm256 vmh1_r3 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(inHeight-1-y,vry, vl), vl) ;	// condition(h < inHeight)
	      __vm256 vmh1_r4 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(inHeight-2-y,vry, vl), vl) ;	// condition(h < inHeight)

	      __vm256 vmh_r0  = vmh0_r0 ;
	      __vm256 vmh_r1  = vmh0_r1 ;
	      __vm256 vmh_r3  = vmh1_r3 ;
	      __vm256 vmh_r4  = vmh1_r4 ;

	      __vm256 vmall_r0s0 = _vel_andm_mmm(vmh_r0, vmw_s0) ;
	      __vm256 vmall_r0s1 = _vel_andm_mmm(vmh_r0, vmw_s1) ;
	      __vm256 vmall_r0s2 = vmh_r0 ;
	      __vm256 vmall_r0s3 = _vel_andm_mmm(vmh_r0, vmw_s3) ;
	      __vm256 vmall_r0s4 = _vel_andm_mmm(vmh_r0, vmw_s4) ;

	      __vm256 vmall_r1s0 = _vel_andm_mmm(vmh_r1, vmw_s0) ;
	      __vm256 vmall_r1s1 = _vel_andm_mmm(vmh_r1, vmw_s1) ;
	      __vm256 vmall_r1s2 = vmh_r1 ;
	      __vm256 vmall_r1s3 = _vel_andm_mmm(vmh_r1, vmw_s3) ;
	      __vm256 vmall_r1s4 = _vel_andm_mmm(vmh_r1, vmw_s4) ;

	      __vm256 vmall_r2s0 = vmw_s0 ;
	      __vm256 vmall_r2s1 = vmw_s1 ;
	      __vm256 vmall_r2s3 = vmw_s3 ;
	      __vm256 vmall_r2s4 = vmw_s4 ;

	      __vm256 vmall_r3s0 = _vel_andm_mmm(vmh_r3, vmw_s0) ;
	      __vm256 vmall_r3s1 = _vel_andm_mmm(vmh_r3, vmw_s1) ;
	      __vm256 vmall_r3s2 = vmh_r3 ;
	      __vm256 vmall_r3s3 = _vel_andm_mmm(vmh_r3, vmw_s3) ;
	      __vm256 vmall_r3s4 = _vel_andm_mmm(vmh_r3, vmw_s4) ;

	      __vm256 vmall_r4s0 = _vel_andm_mmm(vmh_r4, vmw_s0) ;
	      __vm256 vmall_r4s1 = _vel_andm_mmm(vmh_r4, vmw_s1) ;
	      __vm256 vmall_r4s2 = vmh_r4 ;
	      __vm256 vmall_r4s3 = _vel_andm_mmm(vmh_r4, vmw_s3) ;
	      __vm256 vmall_r4s4 = _vel_andm_mmm(vmh_r4, vmw_s4) ;

	      const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;
	      const int64_t gOutIndex  = outGroupOffset + ((n * gOutChannel + k  ) * gOutHeight ) * gOutWidth + gop;

	      /* memory access errors mihgt be caused (vrin) */
	      __vr vrin_r0s0    = _vel_vldu_vssl(4,&pInChannel[gop-2*inWidth-2], vl) ;
	      __vr vrin_r0s1    = _vel_vldu_vssl(4,&pInChannel[gop-2*inWidth-1], vl) ;
	      __vr vrin_r0s2    = _vel_vldu_vssl(4,&pInChannel[gop-2*inWidth  ], vl) ;
	      __vr vrin_r0s3    = _vel_vldu_vssl(4,&pInChannel[gop-2*inWidth+1], vl) ;
	      __vr vrin_r0s4    = _vel_vldu_vssl(4,&pInChannel[gop-2*inWidth+2], vl) ;
	      __vr vrin_r1s0    = _vel_vldu_vssl(4,&pInChannel[gop-  inWidth-2], vl) ;
	      __vr vrin_r1s1    = _vel_vldu_vssl(4,&pInChannel[gop-  inWidth-1], vl) ;
	      __vr vrin_r1s2    = _vel_vldu_vssl(4,&pInChannel[gop-  inWidth  ], vl) ;
	      __vr vrin_r1s3    = _vel_vldu_vssl(4,&pInChannel[gop-  inWidth+1], vl) ;
	      __vr vrin_r1s4    = _vel_vldu_vssl(4,&pInChannel[gop-  inWidth+2], vl) ;
	      __vr vrin_r2s0    = _vel_vldu_vssl(4,&pInChannel[gop          -2], vl) ;
	      __vr vrin_r2s1    = _vel_vldu_vssl(4,&pInChannel[gop          -1], vl) ;
	      __vr vrin_r2s2    = _vel_vldu_vssl(4,&pInChannel[gop            ], vl) ;
	      __vr vrin_r2s3    = _vel_vldu_vssl(4,&pInChannel[gop          +1], vl) ;
	      __vr vrin_r2s4    = _vel_vldu_vssl(4,&pInChannel[gop          +2], vl) ;
	      __vr vrin_r3s0    = _vel_vldu_vssl(4,&pInChannel[gop+  inWidth-2], vl) ;
	      __vr vrin_r3s1    = _vel_vldu_vssl(4,&pInChannel[gop+  inWidth-1], vl) ;
	      __vr vrin_r3s2    = _vel_vldu_vssl(4,&pInChannel[gop+  inWidth  ], vl) ;
	      __vr vrin_r3s3    = _vel_vldu_vssl(4,&pInChannel[gop+  inWidth+1], vl) ;
	      __vr vrin_r3s4    = _vel_vldu_vssl(4,&pInChannel[gop+  inWidth+2], vl) ;
	      __vr vrin_r4s0    = _vel_vldu_vssl(4,&pInChannel[gop+2*inWidth-2], vl) ;
	      __vr vrin_r4s1    = _vel_vldu_vssl(4,&pInChannel[gop+2*inWidth-1], vl) ;
	      __vr vrin_r4s2    = _vel_vldu_vssl(4,&pInChannel[gop+2*inWidth  ], vl) ;
	      __vr vrin_r4s3    = _vel_vldu_vssl(4,&pInChannel[gop+2*inWidth+1], vl) ;
	      __vr vrin_r4s4    = _vel_vldu_vssl(4,&pInChannel[gop+2*inWidth+2], vl) ;

	      __vr vrgout0 = _vel_vldu_vssl(4, pGOut+gOutIndex+0*gOutPixels, vl) ;
	      __vr vrgout1 = _vel_vldu_vssl(4, pGOut+gOutIndex+1*gOutPixels, vl) ;

	      __vr vrgout01 = _vel_vshf_vvvsl(vrgout0, vrgout1, VE_VSHUFFLE_YUZU, vl) ;

#define PVFMAD2(VRSUM, VRIN, VMR) {				\
  VRIN = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), VRIN, VMR, vl) ;	\
  __vr vrinP = _vel_vshf_vvvsl(VRIN, VRIN, VE_VSHUFFLE_YUZU, vl) ;	\
  VRSUM = _vel_pvfmad_vvvvvl(VRSUM, vrinP, vrgout01, VRSUM, vl) ;	\
}

	      PVFMAD2(vrsum01_r0s0, vrin_r0s0, vmall_r0s0) ;
	      PVFMAD2(vrsum01_r0s1, vrin_r0s1, vmall_r0s1) ;
	      PVFMAD2(vrsum01_r0s2, vrin_r0s2, vmall_r0s2) ;
	      PVFMAD2(vrsum01_r0s3, vrin_r0s3, vmall_r0s3) ;
	      PVFMAD2(vrsum01_r0s4, vrin_r0s4, vmall_r0s4) ;

	      PVFMAD2(vrsum01_r1s0, vrin_r1s0, vmall_r1s0) ;
	      PVFMAD2(vrsum01_r1s1, vrin_r1s1, vmall_r1s1) ;
	      PVFMAD2(vrsum01_r1s2, vrin_r1s2, vmall_r1s2) ;
	      PVFMAD2(vrsum01_r1s3, vrin_r1s3, vmall_r1s3) ;
	      PVFMAD2(vrsum01_r1s4, vrin_r1s4, vmall_r1s4) ;

	      PVFMAD2(vrsum01_r2s0, vrin_r2s0, vmall_r2s0) ;
	      PVFMAD2(vrsum01_r2s1, vrin_r2s1, vmall_r2s1) ;
	      {
	        __vr vrinP = _vel_vshf_vvvsl(vrin_r2s2, vrin_r2s2, VE_VSHUFFLE_YUZU, vl) ;
	        vrsum01_r2s2 = _vel_pvfmad_vvvvvl(vrsum01_r2s2, vrinP, vrgout01, vrsum01_r2s2, vl) ;
	      }
	      PVFMAD2(vrsum01_r2s3, vrin_r2s3, vmall_r2s3) ;
	      PVFMAD2(vrsum01_r2s4, vrin_r2s4, vmall_r2s4) ;

	      PVFMAD2(vrsum01_r3s0, vrin_r3s0, vmall_r3s0) ;
	      PVFMAD2(vrsum01_r3s1, vrin_r3s1, vmall_r3s1) ;
	      PVFMAD2(vrsum01_r3s2, vrin_r3s2, vmall_r3s2) ;
	      PVFMAD2(vrsum01_r3s3, vrin_r3s3, vmall_r3s3) ;
	      PVFMAD2(vrsum01_r3s4, vrin_r3s4, vmall_r3s4) ;

	      PVFMAD2(vrsum01_r4s0, vrin_r4s0, vmall_r4s0) ;
	      PVFMAD2(vrsum01_r4s1, vrin_r4s1, vmall_r4s1) ;
	      PVFMAD2(vrsum01_r4s2, vrin_r4s2, vmall_r4s2) ;
	      PVFMAD2(vrsum01_r4s3, vrin_r4s3, vmall_r4s3) ;
	      PVFMAD2(vrsum01_r4s4, vrin_r4s4, vmall_r4s4) ;

#undef PVFMAD2
	    } // gOutPixels
	  } // batch

#define VSUM_STORE_5_UPPER(VRSUMTOKEN, KERNELINDEX)	\
{							\
  __vr vrsumU_s0 = _vel_vfsums_vvl(VRSUMTOKEN ## s0, VLEN) ;	\
  __vr vrsumU_s1 = _vel_vfsums_vvl(VRSUMTOKEN ## s1, VLEN) ;	\
  __vr vrsumU_s2 = _vel_vfsums_vvl(VRSUMTOKEN ## s2, VLEN) ;	\
  __vr vrsumU_s3 = _vel_vfsums_vvl(VRSUMTOKEN ## s3, VLEN) ;	\
  __vr vrsumU_s4 = _vel_vfsums_vvl(VRSUMTOKEN ## s4, VLEN) ;	\
  _vel_vstu_vssl(vrsumU_s0,4,pGKernel+(KERNELINDEX)+0, 1) ;	\
  _vel_vstu_vssl(vrsumU_s1,4,pGKernel+(KERNELINDEX)+1, 1) ;	\
  _vel_vstu_vssl(vrsumU_s2,4,pGKernel+(KERNELINDEX)+2, 1) ;	\
  _vel_vstu_vssl(vrsumU_s3,4,pGKernel+(KERNELINDEX)+3, 1) ;	\
  _vel_vstu_vssl(vrsumU_s4,4,pGKernel+(KERNELINDEX)+4, 1) ;	\
}
#define VSUM_STORE_5_LOWER(VRSUMTOKEN, KERNELINDEX)			\
{									\
  __vr vrsumL_s0 = _vel_vfsums_vvl(_vel_vsll_vvsl(VRSUMTOKEN ## s0,32, VLEN), VLEN) ;	\
  __vr vrsumL_s1 = _vel_vfsums_vvl(_vel_vsll_vvsl(VRSUMTOKEN ## s1,32, VLEN), VLEN) ;	\
  __vr vrsumL_s2 = _vel_vfsums_vvl(_vel_vsll_vvsl(VRSUMTOKEN ## s2,32, VLEN), VLEN) ;	\
  __vr vrsumL_s3 = _vel_vfsums_vvl(_vel_vsll_vvsl(VRSUMTOKEN ## s3,32, VLEN), VLEN) ;	\
  __vr vrsumL_s4 = _vel_vfsums_vvl(_vel_vsll_vvsl(VRSUMTOKEN ## s4,32, VLEN), VLEN) ;	\
  _vel_vstu_vssl(vrsumL_s0,4,pGKernel+(KERNELINDEX)+0, 1) ;		\
  _vel_vstu_vssl(vrsumL_s1,4,pGKernel+(KERNELINDEX)+1, 1) ;		\
  _vel_vstu_vssl(vrsumL_s2,4,pGKernel+(KERNELINDEX)+2, 1) ;		\
  _vel_vstu_vssl(vrsumL_s3,4,pGKernel+(KERNELINDEX)+3, 1) ;		\
  _vel_vstu_vssl(vrsumL_s4,4,pGKernel+(KERNELINDEX)+4, 1) ;		\
}

	  VSUM_STORE_5_UPPER(vrsum01_r0, kernelIndex0) ;
	  VSUM_STORE_5_LOWER(vrsum01_r0, kernelIndex1) ;

	  VSUM_STORE_5_UPPER(vrsum01_r1, kernelIndex0+5) ;
	  VSUM_STORE_5_LOWER(vrsum01_r1, kernelIndex1+5) ;

	  VSUM_STORE_5_UPPER(vrsum01_r2, kernelIndex0+10) ;
	  VSUM_STORE_5_LOWER(vrsum01_r2, kernelIndex1+10) ;

	  VSUM_STORE_5_UPPER(vrsum01_r3, kernelIndex0+15) ;
	  VSUM_STORE_5_LOWER(vrsum01_r3, kernelIndex1+15) ;

	  VSUM_STORE_5_UPPER(vrsum01_r4, kernelIndex0+20) ;
	  VSUM_STORE_5_LOWER(vrsum01_r4, kernelIndex1+20) ;


	} // inChannel
	k+=2;
      }
      if ( ((nOChannel >> 2) & 0x01) == 1 ) {
	for (int64_t c=0; c<inChannelGroup; c++) {
	  const int64_t kernelIndex0 = kernGroupOffset + ((k     * inChannelGroup + c) * gKernHeight) * gKernWidth ;
	  const int64_t kernelIndex1 = kernGroupOffset + (((k+1) * inChannelGroup + c) * gKernHeight) * gKernWidth ;
	  const int64_t kernelIndex2 = kernGroupOffset + (((k+2) * inChannelGroup + c) * gKernHeight) * gKernWidth ;
	  const int64_t kernelIndex3 = kernGroupOffset + (((k+3) * inChannelGroup + c) * gKernHeight) * gKernWidth ;

	  {
#define INIT_VRSUM_4(TOKEN, INDEX)	\
  __vr vrsum01_ ## TOKEN = _vel_vbrdl_vsl(0UL, VLEN) ;	\
  __vr vrsum23_ ## TOKEN = _vel_vbrdl_vsl(0UL, VLEN) ;

	    INIT_VRSUM_4(r0s0, 0) ;
	    INIT_VRSUM_4(r0s1, 1) ;
	    INIT_VRSUM_4(r0s2, 2) ;
	    INIT_VRSUM_4(r0s3, 3) ;
	    INIT_VRSUM_4(r0s4, 4) ;
	    INIT_VRSUM_4(r1s0, 5) ;
	    INIT_VRSUM_4(r1s1, 6) ;
	    INIT_VRSUM_4(r1s2, 7) ;
	    INIT_VRSUM_4(r1s3, 8) ;
	    INIT_VRSUM_4(r1s4, 9) ;
	    INIT_VRSUM_4(r2s0,10) ;
	    INIT_VRSUM_4(r2s1,11) ;
	    INIT_VRSUM_4(r2s2,12) ;
	    INIT_VRSUM_4(r2s3,13) ;
	    INIT_VRSUM_4(r2s4,14) ;
#undef INIT_VRSUM_4

	    for (int64_t n=0; n<batch; n++) {
	      for (int64_t y=0; y<gOutHeight; y+=nY)
	      {
		const int64_t vl = gOutWidth * (gOutHeight - y < nY ? gOutHeight - y : nY) ;
		const int64_t gop = y * gOutWidth ;

		__vm256 vmh0_r0 =  _vel_vfmklge_mvl(_vel_vaddsl_vsvl(y-2, vry, vl), vl) ;		// condition(0 <= h)
		__vm256 vmh0_r1 =  _vel_vfmklge_mvl(_vel_vaddsl_vsvl(y-1, vry, vl), vl) ;		// condition(0 <= h)

		__vm256 vmh_r0  = vmh0_r0 ;
		__vm256 vmh_r1  = vmh0_r1 ;

		__vm256 vmall_r0s0 = _vel_andm_mmm(vmh_r0, vmw_s0) ;
		__vm256 vmall_r0s1 = _vel_andm_mmm(vmh_r0, vmw_s1) ;
		__vm256 vmall_r0s2 = vmh_r0 ;
		__vm256 vmall_r0s3 = _vel_andm_mmm(vmh_r0, vmw_s3) ;
		__vm256 vmall_r0s4 = _vel_andm_mmm(vmh_r0, vmw_s4) ;

		__vm256 vmall_r1s0 = _vel_andm_mmm(vmh_r1, vmw_s0) ;
		__vm256 vmall_r1s1 = _vel_andm_mmm(vmh_r1, vmw_s1) ;
		__vm256 vmall_r1s2 = vmh_r1 ;
		__vm256 vmall_r1s3 = _vel_andm_mmm(vmh_r1, vmw_s3) ;
		__vm256 vmall_r1s4 = _vel_andm_mmm(vmh_r1, vmw_s4) ;

		__vm256 vmall_r2s0 = vmw_s0 ;
		__vm256 vmall_r2s1 = vmw_s1 ;
		__vm256 vmall_r2s3 = vmw_s3 ;
		__vm256 vmall_r2s4 = vmw_s4 ;

		const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;
		const int64_t gOutIndex  = outGroupOffset + ((n * gOutChannel + k  ) * gOutHeight ) * gOutWidth + gop;

		/* memory access errors mihgt be caused (vrin) */
		__vr vrin_r0s0    = _vel_vldu_vssl(4,&pInChannel[gop-2*inWidth-2], vl) ;
		__vr vrin_r0s1    = _vel_vldu_vssl(4,&pInChannel[gop-2*inWidth-1], vl) ;
		__vr vrin_r0s2    = _vel_vldu_vssl(4,&pInChannel[gop-2*inWidth  ], vl) ;
		__vr vrin_r0s3    = _vel_vldu_vssl(4,&pInChannel[gop-2*inWidth+1], vl) ;
		__vr vrin_r0s4    = _vel_vldu_vssl(4,&pInChannel[gop-2*inWidth+2], vl) ;
		__vr vrin_r1s0    = _vel_vldu_vssl(4,&pInChannel[gop-  inWidth-2], vl) ;
		__vr vrin_r1s1    = _vel_vldu_vssl(4,&pInChannel[gop-  inWidth-1], vl) ;
		__vr vrin_r1s2    = _vel_vldu_vssl(4,&pInChannel[gop-  inWidth  ], vl) ;
		__vr vrin_r1s3    = _vel_vldu_vssl(4,&pInChannel[gop-  inWidth+1], vl) ;
		__vr vrin_r1s4    = _vel_vldu_vssl(4,&pInChannel[gop-  inWidth+2], vl) ;
		__vr vrin_r2s0    = _vel_vldu_vssl(4,&pInChannel[gop          -2], vl) ;
		__vr vrin_r2s1    = _vel_vldu_vssl(4,&pInChannel[gop          -1], vl) ;
		__vr vrin_r2s2    = _vel_vldu_vssl(4,&pInChannel[gop            ], vl) ;
		__vr vrin_r2s3    = _vel_vldu_vssl(4,&pInChannel[gop          +1], vl) ;
		__vr vrin_r2s4    = _vel_vldu_vssl(4,&pInChannel[gop          +2], vl) ;

		__vr vrgout0 = _vel_vldu_vssl(4, pGOut+gOutIndex+0*gOutPixels, vl) ;
		__vr vrgout1 = _vel_vldu_vssl(4, pGOut+gOutIndex+1*gOutPixels, vl) ;
		__vr vrgout2 = _vel_vldu_vssl(4, pGOut+gOutIndex+2*gOutPixels, vl) ;
		__vr vrgout3 = _vel_vldu_vssl(4, pGOut+gOutIndex+3*gOutPixels, vl) ;

		__vr vrgout01 = _vel_vshf_vvvsl(vrgout0, vrgout1, VE_VSHUFFLE_YUZU, vl) ;
		__vr vrgout23 = _vel_vshf_vvvsl(vrgout2, vrgout3, VE_VSHUFFLE_YUZU, vl) ;

#define PVFMAD4(VRSUM01, VRSUM23, VRIN,  VMR) { 			\
  VRIN = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), VRIN, VMR, vl) ;	\
  __vr vrinP = _vel_vshf_vvvsl(VRIN, VRIN, VE_VSHUFFLE_YUZU, vl) ;	\
  VRSUM01 = _vel_pvfmad_vvvvvl(VRSUM01, vrinP, vrgout01, VRSUM01, vl) ;	\
  VRSUM23 = _vel_pvfmad_vvvvvl(VRSUM23, vrinP, vrgout23, VRSUM23, vl) ;	\
}

		PVFMAD4(vrsum01_r0s0, vrsum23_r0s0, vrin_r0s0, vmall_r0s0) ;
		PVFMAD4(vrsum01_r0s1, vrsum23_r0s1, vrin_r0s1, vmall_r0s1) ;
		PVFMAD4(vrsum01_r0s2, vrsum23_r0s2, vrin_r0s2, vmall_r0s2) ;
		PVFMAD4(vrsum01_r0s3, vrsum23_r0s3, vrin_r0s3, vmall_r0s3) ;
		PVFMAD4(vrsum01_r0s4, vrsum23_r0s4, vrin_r0s4, vmall_r0s4) ;

		PVFMAD4(vrsum01_r1s0, vrsum23_r1s0, vrin_r1s0, vmall_r1s0) ;
		PVFMAD4(vrsum01_r1s1, vrsum23_r1s1, vrin_r1s1, vmall_r1s1) ;
		PVFMAD4(vrsum01_r1s2, vrsum23_r1s2, vrin_r1s2, vmall_r1s2) ;
		PVFMAD4(vrsum01_r1s3, vrsum23_r1s3, vrin_r1s3, vmall_r1s3) ;
		PVFMAD4(vrsum01_r1s4, vrsum23_r1s4, vrin_r1s4, vmall_r1s4) ;

		PVFMAD4(vrsum01_r2s0, vrsum23_r2s0, vrin_r2s0, vmall_r2s0) ;
		PVFMAD4(vrsum01_r2s1, vrsum23_r2s1, vrin_r2s1, vmall_r2s1) ;
		{
		  __vr vrinP = _vel_vshf_vvvsl(vrin_r2s2, vrin_r2s2, VE_VSHUFFLE_YUZU, vl) ;
		  vrsum01_r2s2 = _vel_pvfmad_vvvvvl(vrsum01_r2s2, vrinP, vrgout01, vrsum01_r2s2, vl) ;
		  vrsum23_r2s2 = _vel_pvfmad_vvvvvl(vrsum23_r2s2, vrinP, vrgout23, vrsum23_r2s2, vl) ;
		}
		PVFMAD4(vrsum01_r2s3, vrsum23_r2s3, vrin_r2s3, vmall_r2s3) ;
		PVFMAD4(vrsum01_r2s4, vrsum23_r2s4, vrin_r2s4, vmall_r2s4) ;
#undef PVFMAD4

	      } // gOutPixels
	    } // batch

	    VSUM_STORE_5_UPPER(vrsum01_r0, kernelIndex0) ;
	    VSUM_STORE_5_LOWER(vrsum01_r0, kernelIndex1) ;
	    VSUM_STORE_5_UPPER(vrsum23_r0, kernelIndex2) ;
	    VSUM_STORE_5_LOWER(vrsum23_r0, kernelIndex3) ;

	    VSUM_STORE_5_UPPER(vrsum01_r1, kernelIndex0+5) ;
	    VSUM_STORE_5_LOWER(vrsum01_r1, kernelIndex1+5) ;
	    VSUM_STORE_5_UPPER(vrsum23_r1, kernelIndex2+5) ;
	    VSUM_STORE_5_LOWER(vrsum23_r1, kernelIndex3+5) ;

	    VSUM_STORE_5_UPPER(vrsum01_r2, kernelIndex0+10) ;
	    VSUM_STORE_5_LOWER(vrsum01_r2, kernelIndex1+10) ;
	    VSUM_STORE_5_UPPER(vrsum23_r2, kernelIndex2+10) ;
	    VSUM_STORE_5_LOWER(vrsum23_r2, kernelIndex3+10) ;

	  } // r=0,1,2
	  {
#define INIT_VRSUM_4(TOKEN, INDEX)	\
  __vr vrsum01_ ## TOKEN = _vel_vbrdl_vsl(0UL, VLEN) ;	\
  __vr vrsum23_ ## TOKEN = _vel_vbrdl_vsl(0UL, VLEN) ;

	    INIT_VRSUM_4(r3s0,15) ;
	    INIT_VRSUM_4(r3s1,16) ;
	    INIT_VRSUM_4(r3s2,17) ;
	    INIT_VRSUM_4(r3s3,18) ;
	    INIT_VRSUM_4(r3s4,19) ;
	    INIT_VRSUM_4(r4s0,20) ;
	    INIT_VRSUM_4(r4s1,21) ;
	    INIT_VRSUM_4(r4s2,22) ;
	    INIT_VRSUM_4(r4s3,23) ;
	    INIT_VRSUM_4(r4s4,24) ;
#undef INIT_VRSUM_4

	    for (int64_t n=0; n<batch; n++) {
	      for (int64_t y=0; y<gOutHeight; y+=nY)
	      {
		const int64_t vl = gOutWidth * (gOutHeight - y < nY ? gOutHeight - y : nY) ;
		const int64_t gop = y * gOutWidth ;

		__vm256 vmh1_r3 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(inHeight-1-y,vry, vl), vl) ;	// condition(h < inHeight)
		__vm256 vmh1_r4 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(inHeight-2-y,vry, vl), vl) ;	// condition(h < inHeight)

		__vm256 vmh_r3  = vmh1_r3 ;
		__vm256 vmh_r4  = vmh1_r4 ;

		__vm256 vmall_r3s0 = _vel_andm_mmm(vmh_r3, vmw_s0) ;
		__vm256 vmall_r3s1 = _vel_andm_mmm(vmh_r3, vmw_s1) ;
		__vm256 vmall_r3s2 = vmh_r3 ;
		__vm256 vmall_r3s3 = _vel_andm_mmm(vmh_r3, vmw_s3) ;
		__vm256 vmall_r3s4 = _vel_andm_mmm(vmh_r3, vmw_s4) ;

		__vm256 vmall_r4s0 = _vel_andm_mmm(vmh_r4, vmw_s0) ;
		__vm256 vmall_r4s1 = _vel_andm_mmm(vmh_r4, vmw_s1) ;
		__vm256 vmall_r4s2 = vmh_r4 ;
		__vm256 vmall_r4s3 = _vel_andm_mmm(vmh_r4, vmw_s3) ;
		__vm256 vmall_r4s4 = _vel_andm_mmm(vmh_r4, vmw_s4) ;

		const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;
		const int64_t gOutIndex  = outGroupOffset + ((n * gOutChannel + k  ) * gOutHeight ) * gOutWidth + gop;

		/* memory access errors mihgt be caused (vrin) */
		__vr vrin_r3s0    = _vel_vldu_vssl(4,&pInChannel[gop+  inWidth-2], vl) ;
		__vr vrin_r3s1    = _vel_vldu_vssl(4,&pInChannel[gop+  inWidth-1], vl) ;
		__vr vrin_r3s2    = _vel_vldu_vssl(4,&pInChannel[gop+  inWidth  ], vl) ;
		__vr vrin_r3s3    = _vel_vldu_vssl(4,&pInChannel[gop+  inWidth+1], vl) ;
		__vr vrin_r3s4    = _vel_vldu_vssl(4,&pInChannel[gop+  inWidth+2], vl) ;
		__vr vrin_r4s0    = _vel_vldu_vssl(4,&pInChannel[gop+2*inWidth-2], vl) ;
		__vr vrin_r4s1    = _vel_vldu_vssl(4,&pInChannel[gop+2*inWidth-1], vl) ;
		__vr vrin_r4s2    = _vel_vldu_vssl(4,&pInChannel[gop+2*inWidth  ], vl) ;
		__vr vrin_r4s3    = _vel_vldu_vssl(4,&pInChannel[gop+2*inWidth+1], vl) ;
		__vr vrin_r4s4    = _vel_vldu_vssl(4,&pInChannel[gop+2*inWidth+2], vl) ;

		__vr vrgout0 = _vel_vldu_vssl(4, pGOut+gOutIndex+0*gOutPixels, vl) ;
		__vr vrgout1 = _vel_vldu_vssl(4, pGOut+gOutIndex+1*gOutPixels, vl) ;
		__vr vrgout2 = _vel_vldu_vssl(4, pGOut+gOutIndex+2*gOutPixels, vl) ;
		__vr vrgout3 = _vel_vldu_vssl(4, pGOut+gOutIndex+3*gOutPixels, vl) ;

		__vr vrgout01 = _vel_vshf_vvvsl(vrgout0, vrgout1, VE_VSHUFFLE_YUZU, vl) ;
		__vr vrgout23 = _vel_vshf_vvvsl(vrgout2, vrgout3, VE_VSHUFFLE_YUZU, vl) ;

#define PVFMAD4(VRSUM01, VRSUM23, VRIN,  VMR) { 			\
  VRIN = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), VRIN, VMR, vl) ;	\
  __vr vrinP = _vel_vshf_vvvsl(VRIN, VRIN, VE_VSHUFFLE_YUZU, vl) ;	\
  VRSUM01 = _vel_pvfmad_vvvvvl(VRSUM01, vrinP, vrgout01, VRSUM01, vl) ;	\
  VRSUM23 = _vel_pvfmad_vvvvvl(VRSUM23, vrinP, vrgout23, VRSUM23, vl) ;	\
}

		PVFMAD4(vrsum01_r3s0, vrsum23_r3s0, vrin_r3s0, vmall_r3s0) ;
		PVFMAD4(vrsum01_r3s1, vrsum23_r3s1, vrin_r3s1, vmall_r3s1) ;
		PVFMAD4(vrsum01_r3s2, vrsum23_r3s2, vrin_r3s2, vmall_r3s2) ;
		PVFMAD4(vrsum01_r3s3, vrsum23_r3s3, vrin_r3s3, vmall_r3s3) ;
		PVFMAD4(vrsum01_r3s4, vrsum23_r3s4, vrin_r3s4, vmall_r3s4) ;

		PVFMAD4(vrsum01_r4s0, vrsum23_r4s0, vrin_r4s0, vmall_r4s0) ;
		PVFMAD4(vrsum01_r4s1, vrsum23_r4s1, vrin_r4s1, vmall_r4s1) ;
		PVFMAD4(vrsum01_r4s2, vrsum23_r4s2, vrin_r4s2, vmall_r4s2) ;
		PVFMAD4(vrsum01_r4s3, vrsum23_r4s3, vrin_r4s3, vmall_r4s3) ;
		PVFMAD4(vrsum01_r4s4, vrsum23_r4s4, vrin_r4s4, vmall_r4s4) ;
#undef PVFMAD4

	      } // gOutPixels
	    } // batch

	    VSUM_STORE_5_UPPER(vrsum01_r3, kernelIndex0+15) ;
	    VSUM_STORE_5_LOWER(vrsum01_r3, kernelIndex1+15) ;
	    VSUM_STORE_5_UPPER(vrsum23_r3, kernelIndex2+15) ;
	    VSUM_STORE_5_LOWER(vrsum23_r3, kernelIndex3+15) ;

	    VSUM_STORE_5_UPPER(vrsum01_r4, kernelIndex0+20) ;
	    VSUM_STORE_5_LOWER(vrsum01_r4, kernelIndex1+20) ;
	    VSUM_STORE_5_UPPER(vrsum23_r4, kernelIndex2+20) ;
	    VSUM_STORE_5_LOWER(vrsum23_r4, kernelIndex3+20) ;
	  }  // r=3,4
	} // inChannel
	k+=4;
      }
      for ( ;k<nOChannel; k+=8) {
	for (int64_t c=0; c<inChannelGroup; c++) {
	  const int64_t kernelIndex0 = kernGroupOffset + ((k     * inChannelGroup + c) * gKernHeight) * gKernWidth ;
	  const int64_t kernelIndex1 = kernGroupOffset + (((k+1) * inChannelGroup + c) * gKernHeight) * gKernWidth ;
	  const int64_t kernelIndex2 = kernGroupOffset + (((k+2) * inChannelGroup + c) * gKernHeight) * gKernWidth ;
	  const int64_t kernelIndex3 = kernGroupOffset + (((k+3) * inChannelGroup + c) * gKernHeight) * gKernWidth ;
	  const int64_t kernelIndex4 = kernGroupOffset + (((k+4) * inChannelGroup + c) * gKernHeight) * gKernWidth ;
	  const int64_t kernelIndex5 = kernGroupOffset + (((k+5) * inChannelGroup + c) * gKernHeight) * gKernWidth ;
	  const int64_t kernelIndex6 = kernGroupOffset + (((k+6) * inChannelGroup + c) * gKernHeight) * gKernWidth ;
	  const int64_t kernelIndex7 = kernGroupOffset + (((k+7) * inChannelGroup + c) * gKernHeight) * gKernWidth ;
	  {
#define INIT_VRSUM_8(TOKEN, INDEX)	\
  __vr vrsum01_ ## TOKEN = _vel_vbrdl_vsl(0UL, VLEN) ;	\
  __vr vrsum23_ ## TOKEN = _vel_vbrdl_vsl(0UL, VLEN) ;	\
  __vr vrsum45_ ## TOKEN = _vel_vbrdl_vsl(0UL, VLEN) ;	\
  __vr vrsum67_ ## TOKEN = _vel_vbrdl_vsl(0UL, VLEN) ;

	    INIT_VRSUM_8(r0s0, 0) ;
	    INIT_VRSUM_8(r0s1, 1) ;
	    INIT_VRSUM_8(r0s2, 2) ;
	    INIT_VRSUM_8(r0s3, 3) ;
	    INIT_VRSUM_8(r0s4, 4) ;
	    INIT_VRSUM_8(r1s0, 5) ;
	    INIT_VRSUM_8(r1s1, 6) ;
	    INIT_VRSUM_8(r1s2, 7) ;
	    INIT_VRSUM_8(r1s3, 8) ;
	    INIT_VRSUM_8(r1s4, 9) ;
#undef INIT_VRSUM_8

	    for (int64_t n=0; n<batch; n++) {
	      for (int64_t y=0; y<gOutHeight; y+=nY)
	      {
		const int64_t vl = gOutWidth * (gOutHeight - y < nY ? gOutHeight - y : nY) ;
		const int64_t gop = y * gOutWidth ;

		__vm256 vmh0_r0 =  _vel_vfmklge_mvl(_vel_vaddsl_vsvl(y-2, vry, vl), vl) ;		// condition(0 <= h)
		__vm256 vmh0_r1 =  _vel_vfmklge_mvl(_vel_vaddsl_vsvl(y-1, vry, vl), vl) ;		// condition(0 <= h)

		__vm256 vmh_r0  = vmh0_r0 ;
		__vm256 vmh_r1  = vmh0_r1 ;

		__vm256 vmall_r0s0 = _vel_andm_mmm(vmh_r0, vmw_s0) ;
		__vm256 vmall_r0s1 = _vel_andm_mmm(vmh_r0, vmw_s1) ;
		__vm256 vmall_r0s2 = vmh_r0 ;
		__vm256 vmall_r0s3 = _vel_andm_mmm(vmh_r0, vmw_s3) ;
		__vm256 vmall_r0s4 = _vel_andm_mmm(vmh_r0, vmw_s4) ;

		__vm256 vmall_r1s0 = _vel_andm_mmm(vmh_r1, vmw_s0) ;
		__vm256 vmall_r1s1 = _vel_andm_mmm(vmh_r1, vmw_s1) ;
		__vm256 vmall_r1s2 = vmh_r1 ;
		__vm256 vmall_r1s3 = _vel_andm_mmm(vmh_r1, vmw_s3) ;
		__vm256 vmall_r1s4 = _vel_andm_mmm(vmh_r1, vmw_s4) ;

		const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;
		const int64_t gOutIndex  = outGroupOffset + ((n * gOutChannel + k  ) * gOutHeight ) * gOutWidth + gop;

		/* memory access errors mihgt be caused (vrin) */
		__vr vrin_r0s0    = _vel_vldu_vssl(4,&pInChannel[gop-2*inWidth-2], vl) ;
		__vr vrin_r0s1    = _vel_vldu_vssl(4,&pInChannel[gop-2*inWidth-1], vl) ;
		__vr vrin_r0s2    = _vel_vldu_vssl(4,&pInChannel[gop-2*inWidth  ], vl) ;
		__vr vrin_r0s3    = _vel_vldu_vssl(4,&pInChannel[gop-2*inWidth+1], vl) ;
		__vr vrin_r0s4    = _vel_vldu_vssl(4,&pInChannel[gop-2*inWidth+2], vl) ;
		__vr vrin_r1s0    = _vel_vldu_vssl(4,&pInChannel[gop-  inWidth-2], vl) ;
		__vr vrin_r1s1    = _vel_vldu_vssl(4,&pInChannel[gop-  inWidth-1], vl) ;
		__vr vrin_r1s2    = _vel_vldu_vssl(4,&pInChannel[gop-  inWidth  ], vl) ;
		__vr vrin_r1s3    = _vel_vldu_vssl(4,&pInChannel[gop-  inWidth+1], vl) ;
		__vr vrin_r1s4    = _vel_vldu_vssl(4,&pInChannel[gop-  inWidth+2], vl) ;

		__vr vrgout0 = _vel_vldu_vssl(4, pGOut+gOutIndex+0*gOutPixels, vl) ;
		__vr vrgout1 = _vel_vldu_vssl(4, pGOut+gOutIndex+1*gOutPixels, vl) ;
		__vr vrgout2 = _vel_vldu_vssl(4, pGOut+gOutIndex+2*gOutPixels, vl) ;
		__vr vrgout3 = _vel_vldu_vssl(4, pGOut+gOutIndex+3*gOutPixels, vl) ;
		__vr vrgout4 = _vel_vldu_vssl(4, pGOut+gOutIndex+4*gOutPixels, vl) ;
		__vr vrgout5 = _vel_vldu_vssl(4, pGOut+gOutIndex+5*gOutPixels, vl) ;
		__vr vrgout6 = _vel_vldu_vssl(4, pGOut+gOutIndex+6*gOutPixels, vl) ;
		__vr vrgout7 = _vel_vldu_vssl(4, pGOut+gOutIndex+7*gOutPixels, vl) ;

		__vr vrgout01 = _vel_vshf_vvvsl(vrgout0, vrgout1, VE_VSHUFFLE_YUZU, vl) ;
		__vr vrgout23 = _vel_vshf_vvvsl(vrgout2, vrgout3, VE_VSHUFFLE_YUZU, vl) ;
		__vr vrgout45 = _vel_vshf_vvvsl(vrgout4, vrgout5, VE_VSHUFFLE_YUZU, vl) ;
		__vr vrgout67 = _vel_vshf_vvvsl(vrgout6, vrgout7, VE_VSHUFFLE_YUZU, vl) ;

#define PVFMAD8(VRSUM01, VRSUM23, VRSUM45, VRSUM67, VRIN,  VMR) {	\
  VRIN = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), VRIN, VMR, vl) ;	\
  __vr vrinP = _vel_vshf_vvvsl(VRIN, VRIN, VE_VSHUFFLE_YUZU, vl) ;	\
  VRSUM01 = _vel_pvfmad_vvvvvl(VRSUM01, vrinP, vrgout01, VRSUM01, vl) ;	\
  VRSUM23 = _vel_pvfmad_vvvvvl(VRSUM23, vrinP, vrgout23, VRSUM23, vl) ;	\
  VRSUM45 = _vel_pvfmad_vvvvvl(VRSUM45, vrinP, vrgout45, VRSUM45, vl) ;	\
  VRSUM67 = _vel_pvfmad_vvvvvl(VRSUM67, vrinP, vrgout67, VRSUM67, vl) ;	\
}

		PVFMAD8(vrsum01_r0s0, vrsum23_r0s0, vrsum45_r0s0, vrsum67_r0s0, vrin_r0s0, vmall_r0s0) ;
		PVFMAD8(vrsum01_r0s1, vrsum23_r0s1, vrsum45_r0s1, vrsum67_r0s1, vrin_r0s1, vmall_r0s1) ;
		PVFMAD8(vrsum01_r0s2, vrsum23_r0s2, vrsum45_r0s2, vrsum67_r0s2, vrin_r0s2, vmall_r0s2) ;
		PVFMAD8(vrsum01_r0s3, vrsum23_r0s3, vrsum45_r0s3, vrsum67_r0s3, vrin_r0s3, vmall_r0s3) ;
		PVFMAD8(vrsum01_r0s4, vrsum23_r0s4, vrsum45_r0s4, vrsum67_r0s4, vrin_r0s4, vmall_r0s4) ;

		PVFMAD8(vrsum01_r1s0, vrsum23_r1s0, vrsum45_r1s0, vrsum67_r1s0, vrin_r1s0, vmall_r1s0) ;
		PVFMAD8(vrsum01_r1s1, vrsum23_r1s1, vrsum45_r1s1, vrsum67_r1s1, vrin_r1s1, vmall_r1s1) ;
		PVFMAD8(vrsum01_r1s2, vrsum23_r1s2, vrsum45_r1s2, vrsum67_r1s2, vrin_r1s2, vmall_r1s2) ;
		PVFMAD8(vrsum01_r1s3, vrsum23_r1s3, vrsum45_r1s3, vrsum67_r1s3, vrin_r1s3, vmall_r1s3) ;
		PVFMAD8(vrsum01_r1s4, vrsum23_r1s4, vrsum45_r1s4, vrsum67_r1s4, vrin_r1s4, vmall_r1s4) ;
#undef PVFMAD8

	      } // gOutPixels
	    } // batch

	    VSUM_STORE_5_UPPER(vrsum01_r0, kernelIndex0) ;
	    VSUM_STORE_5_LOWER(vrsum01_r0, kernelIndex1) ;
	    VSUM_STORE_5_UPPER(vrsum23_r0, kernelIndex2) ;
	    VSUM_STORE_5_LOWER(vrsum23_r0, kernelIndex3) ;
	    VSUM_STORE_5_UPPER(vrsum45_r0, kernelIndex4) ;
	    VSUM_STORE_5_LOWER(vrsum45_r0, kernelIndex5) ;
	    VSUM_STORE_5_UPPER(vrsum67_r0, kernelIndex6) ;
	    VSUM_STORE_5_LOWER(vrsum67_r0, kernelIndex7) ;

	    VSUM_STORE_5_UPPER(vrsum01_r1, kernelIndex0+5) ;
	    VSUM_STORE_5_LOWER(vrsum01_r1, kernelIndex1+5) ;
	    VSUM_STORE_5_UPPER(vrsum23_r1, kernelIndex2+5) ;
	    VSUM_STORE_5_LOWER(vrsum23_r1, kernelIndex3+5) ;
	    VSUM_STORE_5_UPPER(vrsum45_r1, kernelIndex4+5) ;
	    VSUM_STORE_5_LOWER(vrsum45_r1, kernelIndex5+5) ;
	    VSUM_STORE_5_UPPER(vrsum67_r1, kernelIndex6+5) ;
	    VSUM_STORE_5_LOWER(vrsum67_r1, kernelIndex7+5) ;
	  } // r=0,1
	  {
#define INIT_VRSUM_8(TOKEN, INDEX)	\
  __vr vrsum01_ ## TOKEN = _vel_vbrdl_vsl(0UL, VLEN) ;	\
  __vr vrsum23_ ## TOKEN = _vel_vbrdl_vsl(0UL, VLEN) ;	\
  __vr vrsum45_ ## TOKEN = _vel_vbrdl_vsl(0UL, VLEN) ;	\
  __vr vrsum67_ ## TOKEN = _vel_vbrdl_vsl(0UL, VLEN) ;

	    INIT_VRSUM_8(r2s0,10) ;
	    INIT_VRSUM_8(r2s1,11) ;
	    INIT_VRSUM_8(r2s2,12) ;
	    INIT_VRSUM_8(r2s3,13) ;
	    INIT_VRSUM_8(r2s4,14) ;
#undef INIT_VRSUM_8

	    for (int64_t n=0; n<batch; n++) {
	      for (int64_t y=0; y<gOutHeight; y+=nY)
	      {
		const int64_t vl = gOutWidth * (gOutHeight - y < nY ? gOutHeight - y : nY) ;
		const int64_t gop = y * gOutWidth ;

		__vm256 vmall_r2s0 = vmw_s0 ;
		__vm256 vmall_r2s1 = vmw_s1 ;
		__vm256 vmall_r2s3 = vmw_s3 ;
		__vm256 vmall_r2s4 = vmw_s4 ;

		const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;
		const int64_t gOutIndex  = outGroupOffset + ((n * gOutChannel + k  ) * gOutHeight ) * gOutWidth + gop;

		/* memory access errors mihgt be caused (vrin) */
		__vr vrin_r2s0    = _vel_vldu_vssl(4,&pInChannel[gop          -2], vl) ;
		__vr vrin_r2s1    = _vel_vldu_vssl(4,&pInChannel[gop          -1], vl) ;
		__vr vrin_r2s2    = _vel_vldu_vssl(4,&pInChannel[gop            ], vl) ;
		__vr vrin_r2s3    = _vel_vldu_vssl(4,&pInChannel[gop          +1], vl) ;
		__vr vrin_r2s4    = _vel_vldu_vssl(4,&pInChannel[gop          +2], vl) ;

		__vr vrgout0 = _vel_vldu_vssl(4, pGOut+gOutIndex+0*gOutPixels, vl) ;
		__vr vrgout1 = _vel_vldu_vssl(4, pGOut+gOutIndex+1*gOutPixels, vl) ;
		__vr vrgout2 = _vel_vldu_vssl(4, pGOut+gOutIndex+2*gOutPixels, vl) ;
		__vr vrgout3 = _vel_vldu_vssl(4, pGOut+gOutIndex+3*gOutPixels, vl) ;
		__vr vrgout4 = _vel_vldu_vssl(4, pGOut+gOutIndex+4*gOutPixels, vl) ;
		__vr vrgout5 = _vel_vldu_vssl(4, pGOut+gOutIndex+5*gOutPixels, vl) ;
		__vr vrgout6 = _vel_vldu_vssl(4, pGOut+gOutIndex+6*gOutPixels, vl) ;
		__vr vrgout7 = _vel_vldu_vssl(4, pGOut+gOutIndex+7*gOutPixels, vl) ;

		__vr vrgout01 = _vel_vshf_vvvsl(vrgout0, vrgout1, VE_VSHUFFLE_YUZU, vl) ;
		__vr vrgout23 = _vel_vshf_vvvsl(vrgout2, vrgout3, VE_VSHUFFLE_YUZU, vl) ;
		__vr vrgout45 = _vel_vshf_vvvsl(vrgout4, vrgout5, VE_VSHUFFLE_YUZU, vl) ;
		__vr vrgout67 = _vel_vshf_vvvsl(vrgout6, vrgout7, VE_VSHUFFLE_YUZU, vl) ;

#define PVFMAD8(VRSUM01, VRSUM23, VRSUM45, VRSUM67, VRIN,  VMR) {	\
  VRIN = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), VRIN, VMR, vl) ;	\
  __vr vrinP = _vel_vshf_vvvsl(VRIN, VRIN, VE_VSHUFFLE_YUZU, vl) ;	\
  VRSUM01 = _vel_pvfmad_vvvvvl(VRSUM01, vrinP, vrgout01, VRSUM01, vl) ;	\
  VRSUM23 = _vel_pvfmad_vvvvvl(VRSUM23, vrinP, vrgout23, VRSUM23, vl) ;	\
  VRSUM45 = _vel_pvfmad_vvvvvl(VRSUM45, vrinP, vrgout45, VRSUM45, vl) ;	\
  VRSUM67 = _vel_pvfmad_vvvvvl(VRSUM67, vrinP, vrgout67, VRSUM67, vl) ;	\
}

		PVFMAD8(vrsum01_r2s0, vrsum23_r2s0, vrsum45_r2s0, vrsum67_r2s0, vrin_r2s0, vmall_r2s0) ;
		PVFMAD8(vrsum01_r2s1, vrsum23_r2s1, vrsum45_r2s1, vrsum67_r2s1, vrin_r2s1, vmall_r2s1) ;
		{
		  __vr vrinP = _vel_vshf_vvvsl(vrin_r2s2, vrin_r2s2, VE_VSHUFFLE_YUZU, vl) ;
		  vrsum01_r2s2 = _vel_pvfmad_vvvvvl(vrsum01_r2s2, vrinP, vrgout01, vrsum01_r2s2, vl) ;
		  vrsum23_r2s2 = _vel_pvfmad_vvvvvl(vrsum23_r2s2, vrinP, vrgout23, vrsum23_r2s2, vl) ;
		  vrsum45_r2s2 = _vel_pvfmad_vvvvvl(vrsum45_r2s2, vrinP, vrgout45, vrsum45_r2s2, vl) ;
		  vrsum67_r2s2 = _vel_pvfmad_vvvvvl(vrsum67_r2s2, vrinP, vrgout67, vrsum67_r2s2, vl) ;
		}
		PVFMAD8(vrsum01_r2s3, vrsum23_r2s3, vrsum45_r2s3, vrsum67_r2s3, vrin_r2s3, vmall_r2s3) ;
		PVFMAD8(vrsum01_r2s4, vrsum23_r2s4, vrsum45_r2s4, vrsum67_r2s4, vrin_r2s4, vmall_r2s4) ;
#undef PVFMAD8

	      } // gOutPixels
	    } // batch

	    VSUM_STORE_5_UPPER(vrsum01_r2, kernelIndex0+10) ;
	    VSUM_STORE_5_LOWER(vrsum01_r2, kernelIndex1+10) ;
	    VSUM_STORE_5_UPPER(vrsum23_r2, kernelIndex2+10) ;
	    VSUM_STORE_5_LOWER(vrsum23_r2, kernelIndex3+10) ;
	    VSUM_STORE_5_UPPER(vrsum45_r2, kernelIndex4+10) ;
	    VSUM_STORE_5_LOWER(vrsum45_r2, kernelIndex5+10) ;
	    VSUM_STORE_5_UPPER(vrsum67_r2, kernelIndex6+10) ;
	    VSUM_STORE_5_LOWER(vrsum67_r2, kernelIndex7+10) ;

	  } // r=2
	  {
#define INIT_VRSUM_8(TOKEN, INDEX)	\
  __vr vrsum01_ ## TOKEN = _vel_vbrdl_vsl(0UL, VLEN) ;	\
  __vr vrsum23_ ## TOKEN = _vel_vbrdl_vsl(0UL, VLEN) ;	\
  __vr vrsum45_ ## TOKEN = _vel_vbrdl_vsl(0UL, VLEN) ;	\
  __vr vrsum67_ ## TOKEN = _vel_vbrdl_vsl(0UL, VLEN) ;

	    INIT_VRSUM_8(r3s0,15) ;
	    INIT_VRSUM_8(r3s1,16) ;
	    INIT_VRSUM_8(r3s2,17) ;
	    INIT_VRSUM_8(r3s3,18) ;
	    INIT_VRSUM_8(r3s4,19) ;
	    INIT_VRSUM_8(r4s0,20) ;
	    INIT_VRSUM_8(r4s1,21) ;
	    INIT_VRSUM_8(r4s2,22) ;
	    INIT_VRSUM_8(r4s3,23) ;
	    INIT_VRSUM_8(r4s4,24) ;
#undef INIT_VRSUM_8

	    for (int64_t n=0; n<batch; n++) {
	      for (int64_t y=0; y<gOutHeight; y+=nY)
	      {
		const int64_t vl = gOutWidth * (gOutHeight - y < nY ? gOutHeight - y : nY) ;
		const int64_t gop = y * gOutWidth ;

		__vm256 vmh1_r3 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(inHeight-1-y,vry, vl), vl) ;	// condition(h < inHeight)
		__vm256 vmh1_r4 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(inHeight-2-y,vry, vl), vl) ;	// condition(h < inHeight)

		__vm256 vmh_r3  = vmh1_r3 ;
		__vm256 vmh_r4  = vmh1_r4 ;

		__vm256 vmall_r3s0 = _vel_andm_mmm(vmh_r3, vmw_s0) ;
		__vm256 vmall_r3s1 = _vel_andm_mmm(vmh_r3, vmw_s1) ;
		__vm256 vmall_r3s2 = vmh_r3 ;
		__vm256 vmall_r3s3 = _vel_andm_mmm(vmh_r3, vmw_s3) ;
		__vm256 vmall_r3s4 = _vel_andm_mmm(vmh_r3, vmw_s4) ;

		__vm256 vmall_r4s0 = _vel_andm_mmm(vmh_r4, vmw_s0) ;
		__vm256 vmall_r4s1 = _vel_andm_mmm(vmh_r4, vmw_s1) ;
		__vm256 vmall_r4s2 = vmh_r4 ;
		__vm256 vmall_r4s3 = _vel_andm_mmm(vmh_r4, vmw_s3) ;
		__vm256 vmall_r4s4 = _vel_andm_mmm(vmh_r4, vmw_s4) ;

		const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;
		const int64_t gOutIndex  = outGroupOffset + ((n * gOutChannel + k  ) * gOutHeight ) * gOutWidth + gop;

		/* memory access errors mihgt be caused (vrin) */
		__vr vrin_r3s0    = _vel_vldu_vssl(4,&pInChannel[gop+  inWidth-2], vl) ;
		__vr vrin_r3s1    = _vel_vldu_vssl(4,&pInChannel[gop+  inWidth-1], vl) ;
		__vr vrin_r3s2    = _vel_vldu_vssl(4,&pInChannel[gop+  inWidth  ], vl) ;
		__vr vrin_r3s3    = _vel_vldu_vssl(4,&pInChannel[gop+  inWidth+1], vl) ;
		__vr vrin_r3s4    = _vel_vldu_vssl(4,&pInChannel[gop+  inWidth+2], vl) ;
		__vr vrin_r4s0    = _vel_vldu_vssl(4,&pInChannel[gop+2*inWidth-2], vl) ;
		__vr vrin_r4s1    = _vel_vldu_vssl(4,&pInChannel[gop+2*inWidth-1], vl) ;
		__vr vrin_r4s2    = _vel_vldu_vssl(4,&pInChannel[gop+2*inWidth  ], vl) ;
		__vr vrin_r4s3    = _vel_vldu_vssl(4,&pInChannel[gop+2*inWidth+1], vl) ;
		__vr vrin_r4s4    = _vel_vldu_vssl(4,&pInChannel[gop+2*inWidth+2], vl) ;

		__vr vrgout0 = _vel_vldu_vssl(4, pGOut+gOutIndex+0*gOutPixels, vl) ;
		__vr vrgout1 = _vel_vldu_vssl(4, pGOut+gOutIndex+1*gOutPixels, vl) ;
		__vr vrgout2 = _vel_vldu_vssl(4, pGOut+gOutIndex+2*gOutPixels, vl) ;
		__vr vrgout3 = _vel_vldu_vssl(4, pGOut+gOutIndex+3*gOutPixels, vl) ;
		__vr vrgout4 = _vel_vldu_vssl(4, pGOut+gOutIndex+4*gOutPixels, vl) ;
		__vr vrgout5 = _vel_vldu_vssl(4, pGOut+gOutIndex+5*gOutPixels, vl) ;
		__vr vrgout6 = _vel_vldu_vssl(4, pGOut+gOutIndex+6*gOutPixels, vl) ;
		__vr vrgout7 = _vel_vldu_vssl(4, pGOut+gOutIndex+7*gOutPixels, vl) ;

		__vr vrgout01 = _vel_vshf_vvvsl(vrgout0, vrgout1, VE_VSHUFFLE_YUZU, vl) ;
		__vr vrgout23 = _vel_vshf_vvvsl(vrgout2, vrgout3, VE_VSHUFFLE_YUZU, vl) ;
		__vr vrgout45 = _vel_vshf_vvvsl(vrgout4, vrgout5, VE_VSHUFFLE_YUZU, vl) ;
		__vr vrgout67 = _vel_vshf_vvvsl(vrgout6, vrgout7, VE_VSHUFFLE_YUZU, vl) ;

#define PVFMAD8(VRSUM01, VRSUM23, VRSUM45, VRSUM67, VRIN,  VMR) {	\
  VRIN = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), VRIN, VMR, vl) ;	\
  __vr vrinP = _vel_vshf_vvvsl(VRIN, VRIN, VE_VSHUFFLE_YUZU, vl) ;	\
  VRSUM01 = _vel_pvfmad_vvvvvl(VRSUM01, vrinP, vrgout01, VRSUM01, vl) ;	\
  VRSUM23 = _vel_pvfmad_vvvvvl(VRSUM23, vrinP, vrgout23, VRSUM23, vl) ;	\
  VRSUM45 = _vel_pvfmad_vvvvvl(VRSUM45, vrinP, vrgout45, VRSUM45, vl) ;	\
  VRSUM67 = _vel_pvfmad_vvvvvl(VRSUM67, vrinP, vrgout67, VRSUM67, vl) ;	\
}
		PVFMAD8(vrsum01_r3s0, vrsum23_r3s0, vrsum45_r3s0, vrsum67_r3s0, vrin_r3s0, vmall_r3s0) ;
		PVFMAD8(vrsum01_r3s1, vrsum23_r3s1, vrsum45_r3s1, vrsum67_r3s1, vrin_r3s1, vmall_r3s1) ;
		PVFMAD8(vrsum01_r3s2, vrsum23_r3s2, vrsum45_r3s2, vrsum67_r3s2, vrin_r3s2, vmall_r3s2) ;
		PVFMAD8(vrsum01_r3s3, vrsum23_r3s3, vrsum45_r3s3, vrsum67_r3s3, vrin_r3s3, vmall_r3s3) ;
		PVFMAD8(vrsum01_r3s4, vrsum23_r3s4, vrsum45_r3s4, vrsum67_r3s4, vrin_r3s4, vmall_r3s4) ;

		PVFMAD8(vrsum01_r4s0, vrsum23_r4s0, vrsum45_r4s0, vrsum67_r4s0, vrin_r4s0, vmall_r4s0) ;
		PVFMAD8(vrsum01_r4s1, vrsum23_r4s1, vrsum45_r4s1, vrsum67_r4s1, vrin_r4s1, vmall_r4s1) ;
		PVFMAD8(vrsum01_r4s2, vrsum23_r4s2, vrsum45_r4s2, vrsum67_r4s2, vrin_r4s2, vmall_r4s2) ;
		PVFMAD8(vrsum01_r4s3, vrsum23_r4s3, vrsum45_r4s3, vrsum67_r4s3, vrin_r4s3, vmall_r4s3) ;
		PVFMAD8(vrsum01_r4s4, vrsum23_r4s4, vrsum45_r4s4, vrsum67_r4s4, vrin_r4s4, vmall_r4s4) ;
#undef PVFMAD8

	      } // gOutPixels
	    } // batch

	    VSUM_STORE_5_UPPER(vrsum01_r3, kernelIndex0+15) ;
	    VSUM_STORE_5_LOWER(vrsum01_r3, kernelIndex1+15) ;
	    VSUM_STORE_5_UPPER(vrsum23_r3, kernelIndex2+15) ;
	    VSUM_STORE_5_LOWER(vrsum23_r3, kernelIndex3+15) ;
	    VSUM_STORE_5_UPPER(vrsum45_r3, kernelIndex4+15) ;
	    VSUM_STORE_5_LOWER(vrsum45_r3, kernelIndex5+15) ;
	    VSUM_STORE_5_UPPER(vrsum67_r3, kernelIndex6+15) ;
	    VSUM_STORE_5_LOWER(vrsum67_r3, kernelIndex7+15) ;


	    VSUM_STORE_5_UPPER(vrsum01_r4, kernelIndex0+20) ;
	    VSUM_STORE_5_LOWER(vrsum01_r4, kernelIndex1+20) ;
	    VSUM_STORE_5_UPPER(vrsum23_r4, kernelIndex2+20) ;
	    VSUM_STORE_5_LOWER(vrsum23_r4, kernelIndex3+20) ;
	    VSUM_STORE_5_UPPER(vrsum45_r4, kernelIndex4+20) ;
	    VSUM_STORE_5_LOWER(vrsum45_r4, kernelIndex5+20) ;
	    VSUM_STORE_5_UPPER(vrsum67_r4, kernelIndex6+20) ;
	    VSUM_STORE_5_LOWER(vrsum67_r4, kernelIndex7+20) ;

#undef VSUM_STORE_5_UPPER
#undef VSUM_STORE_5_LOWER
	  }  // r=3,4
	} // inChannel
      } // outChannel
    } // group
  }


  return VEDNN_SUCCESS;
}
