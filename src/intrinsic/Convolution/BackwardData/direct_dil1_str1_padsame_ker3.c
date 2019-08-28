#include <stdint.h>
#include <stdlib.h>

#include "vednn.h"

#include "velintrin.h"
#define VLEN	(256)

vednnError_t
vednnConvolutionBackwardData_direct_dil1_str1_padsame_ker3 (
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
  const int64_t kernWidth   = pParamKernel->width;		/* must be 2*padWidth  + 1 */
  const int64_t kernHeight  = pParamKernel->height;		/* must be 2*padHeight + 1 */

  const int64_t group          = pParamConv->group;
//  const int64_t strideWidth    = pParamConv->strideWidth; 	/* must be 1 */
//  const int64_t strideHeight   = pParamConv->strideHeight;	/* must be 1 */
  const int64_t padWidth       = pParamConv->padWidth;
  const int64_t padHeight      = pParamConv->padHeight;

//  const int64_t dilationWidth  = pParamConv->dilationWidth;	/* must be 1 */
//  const int64_t dilationHeight = pParamConv->dilationHeight;	/* must be 1 */

  const int64_t gOutChannelGroup = gOutChannel / group;
  const int64_t gInChannelGroup  = gInChannel  / group;

  const float * restrict pGOut   = pDataGradOut;
  const float * restrict pKernel = pDataKernel;
  float * restrict const pGIn    = pDataGradIn;

  const int gInPixels= gInHeight*gInWidth ;

  {

    for (int64_t n=0; n<batch; n++) {
      for (int64_t g = 0; g < group; g++) {

	int64_t gInGroupOffset  = g * gInChannelGroup * gInHeight * gInWidth;
	int64_t gOutGroupOffset = g * gOutChannelGroup  * gOutHeight  * gOutWidth;
	int64_t kernGroupOffset = g * gOutChannelGroup  * gInChannelGroup * kernHeight * kernWidth;

	int k=0;
	if ( (gInChannelGroup & 0x01) == 1 ) {
	  for (int64_t gip = 0; gip < gInPixels; gip+=VLEN) {
	    const int64_t vl = gInPixels - gip < VLEN ? gInPixels - gip : VLEN ;

	    const int64_t gInIndex = gInGroupOffset + ((n * gInChannel + k) * gInHeight ) * gInWidth + gip ;

	    __vr vrseq = _vel_vseq_vl(vl) ;			// hw
	    __vr vridx = _vel_vaddsl_vsvl(gip, vrseq, vl) ;	// op + hw

	    __vr vrsum = _vel_vbrds_vsl(0.f, vl) ;

	    __vr vrh   = _vel_vdivsl_vvsl(vridx, gInWidth, vl) ;
	    __vr vrix   = _vel_vsubsl_vvvl(vridx, _vel_vmulul_vsvl(gInWidth,vrh, vl), vl) ;

	    __vr vry_r0 = _vel_vaddsl_vsvl(1, vrh, vl) ;
	    __vr vry_r2 = _vel_vaddsl_vsvl(-1, vrh, vl) ;

	    __vr vrx_s0 = _vel_vaddsl_vsvl(1, vrix, vl) ;
	    __vr vrx_s2 = _vel_vaddsl_vsvl(-1, vrix, vl) ;

	    __vm256 vmy_r0 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutHeight,vry_r0, vl), vl) ;
	    __vm256 vmy_r2 =  _vel_vfmklge_mvl(vry_r2, vl) ;

	    __vm256 vmx_s0 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutWidth,vrx_s0, vl), vl) ;
	    __vm256 vmx_s2 =  _vel_vfmklge_mvl(vrx_s2, vl) ;

	    __vm256 vmall_r0s0 = _vel_andm_mmm(vmy_r0,vmx_s0) ;
	    __vm256 vmall_r0s1 = vmy_r0 ;
	    __vm256 vmall_r0s2 = _vel_andm_mmm(vmy_r0,vmx_s2) ;

	    __vm256 vmall_r1s0 = vmx_s0 ;
	    __vm256 vmall_r1s2 = vmx_s2 ;

	    __vm256 vmall_s0_r2 = _vel_andm_mmm(vmy_r2,vmx_s0) ;
	    __vm256 vmall_s1_r2 = vmy_r2 ;
	    __vm256 vmall_s2_r2 = _vel_andm_mmm(vmy_r2,vmx_s2) ;

	    for (int64_t c=0; c<gOutChannelGroup; c++) {
	      const float *pGOutChannel = pGOut + gOutGroupOffset + ((n * gOutChannel + c) * gOutHeight * gOutWidth ) ;
	      const float *pKerValue = pKernel + kernGroupOffset + ((c * gInChannelGroup + k) * kernHeight) * kernWidth ;

	      /* memory access errors mihgt be caused */
	      __vr vrgout_r0s0 = _vel_vldu_vssl(4,&pGOutChannel[gip+(padHeight-0)*gOutWidth+(padWidth-0)], vl) ;
	      __vr vrgout_r0s1 = _vel_vldu_vssl(4,&pGOutChannel[gip+(padHeight-0)*gOutWidth+(padWidth-1)], vl) ;
	      __vr vrgout_r0s2 = _vel_vldu_vssl(4,&pGOutChannel[gip+(padHeight-0)*gOutWidth+(padWidth-2)], vl) ;
	      __vr vrgout_r1s0 = _vel_vldu_vssl(4,&pGOutChannel[gip+(padHeight-1)*gOutWidth+(padWidth-0)], vl) ;
	      __vr vrgout_r1s1 = _vel_vldu_vssl(4,&pGOutChannel[gip+(padHeight-1)*gOutWidth+(padWidth-1)], vl) ;
	      __vr vrgout_r1s2 = _vel_vldu_vssl(4,&pGOutChannel[gip+(padHeight-1)*gOutWidth+(padWidth-2)], vl) ;
	      __vr vrgout_r2s0 = _vel_vldu_vssl(4,&pGOutChannel[gip+(padHeight-2)*gOutWidth+(padWidth-0)], vl) ;
	      __vr vrgout_r2s1 = _vel_vldu_vssl(4,&pGOutChannel[gip+(padHeight-2)*gOutWidth+(padWidth-1)], vl) ;
	      __vr vrgout_r2s2 = _vel_vldu_vssl(4,&pGOutChannel[gip+(padHeight-2)*gOutWidth+(padWidth-2)], vl) ;

	      vrgout_r0s0 = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrgout_r0s0, vmall_r0s0, vl) ;
	      vrgout_r0s1 = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrgout_r0s1, vmall_r0s1, vl) ;
	      vrgout_r0s2 = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrgout_r0s2, vmall_r0s2, vl) ;

	      vrsum = _vel_vfmads_vvsvl(vrsum, *pKerValue, vrgout_r0s0, vl) ; pKerValue++ ;
	      vrsum = _vel_vfmads_vvsvl(vrsum, *pKerValue, vrgout_r0s1, vl) ; pKerValue++ ;
	      vrsum = _vel_vfmads_vvsvl(vrsum, *pKerValue, vrgout_r0s2, vl) ; pKerValue++ ;


	      vrgout_r1s0 = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrgout_r1s0, vmall_r1s0, vl) ;
	      vrgout_r1s2 = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrgout_r1s2, vmall_r1s2, vl) ;

	      vrsum = _vel_vfmads_vvsvl(vrsum, *pKerValue, vrgout_r1s0, vl) ; pKerValue++ ;
	      vrsum = _vel_vfmads_vvsvl(vrsum, *pKerValue, vrgout_r1s1, vl) ; pKerValue++ ;
	      vrsum = _vel_vfmads_vvsvl(vrsum, *pKerValue, vrgout_r1s2, vl) ; pKerValue++ ;


	      vrgout_r2s0 = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrgout_r2s0, vmall_s0_r2, vl) ;
	      vrgout_r2s1 = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrgout_r2s1, vmall_s1_r2, vl) ;
	      vrgout_r2s2 = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrgout_r2s2, vmall_s2_r2, vl) ;

	      vrsum = _vel_vfmads_vvsvl(vrsum, *pKerValue, vrgout_r2s0, vl) ; pKerValue++ ;
	      vrsum = _vel_vfmads_vvsvl(vrsum, *pKerValue, vrgout_r2s1, vl) ; pKerValue++ ;
	      vrsum = _vel_vfmads_vvsvl(vrsum, *pKerValue, vrgout_r2s2, vl) ; pKerValue++ ;

	    } // gInChannel

	    _vel_vstu_vssl(vrsum, 4, pGIn+gInIndex, vl) ;

	  } // gInPixels

	  k+=1 ;
	}
	if ( ((gInChannelGroup >> 1) & 0x01) == 1 ) {
	  for (int64_t gip = 0; gip < gInPixels; gip+=VLEN) {
	    const int64_t vl = gInPixels - gip < VLEN ? gInPixels - gip : VLEN ;

	    const int64_t gInIndex = gInGroupOffset + ((n * gInChannel + k) * gInHeight ) * gInWidth + gip ;

	    __vr vrseq = _vel_vseq_vl(vl) ;			// hw
	    __vr vridx = _vel_vaddsl_vsvl(gip, vrseq, vl) ;	// op + hw

	    __vr vrsum01 = _vel_pvbrd_vsl(0UL, vl) ;

	    __vr vrh   = _vel_vdivsl_vvsl(vridx, gInWidth, vl) ;
	    __vr vrix   = _vel_vsubsl_vvvl(vridx, _vel_vmulul_vsvl(gInWidth,vrh, vl), vl) ;

	    __vr vry_r0 = _vel_vaddsl_vsvl(1, vrh, vl) ;
	    __vr vry_r2 = _vel_vaddsl_vsvl(-1, vrh, vl) ;

	    __vr vrx_s0 = _vel_vaddsl_vsvl(1, vrix, vl) ;
	    __vr vrx_s2 = _vel_vaddsl_vsvl(-1, vrix, vl) ;

	    __vm256 vmy_r0 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutHeight,vry_r0, vl), vl) ;
	    __vm256 vmy_r2 =  _vel_vfmklge_mvl(vry_r2, vl) ;

	    __vm256 vmx_s0 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutWidth,vrx_s0, vl), vl) ;
	    __vm256 vmx_s2 =  _vel_vfmklge_mvl(vrx_s2, vl) ;

	    __vm256 vmall_r0s0 = _vel_andm_mmm(vmy_r0,vmx_s0) ;
	    __vm256 vmall_r0s1 = vmy_r0 ;
	    __vm256 vmall_r0s2 = _vel_andm_mmm(vmy_r0,vmx_s2) ;

	    __vm256 vmall_r1s0 = vmx_s0 ;
	    __vm256 vmall_r1s2 = vmx_s2 ;

	    __vm256 vmall_s0_r2 = _vel_andm_mmm(vmy_r2,vmx_s0) ;
	    __vm256 vmall_s1_r2 = vmy_r2 ;
	    __vm256 vmall_s2_r2 = _vel_andm_mmm(vmy_r2,vmx_s2) ;

	    for (int64_t c=0; c<gOutChannelGroup; c++) {
	      const float *pGOutChannel = pGOut + gOutGroupOffset + ((n * gOutChannel + c) * gOutHeight * gOutWidth ) ;
	      const float *pKerValue = pKernel + kernGroupOffset + ((c * gInChannelGroup + k) * kernHeight) * kernWidth ;

	      /* memory access errors mihgt be caused */
	      __vr vrgout_r0s0 = _vel_vldu_vssl(4,&pGOutChannel[gip+(padHeight-0)*gOutWidth+(padWidth-0)], vl) ;
	      __vr vrgout_r0s1 = _vel_vldu_vssl(4,&pGOutChannel[gip+(padHeight-0)*gOutWidth+(padWidth-1)], vl) ;
	      __vr vrgout_r0s2 = _vel_vldu_vssl(4,&pGOutChannel[gip+(padHeight-0)*gOutWidth+(padWidth-2)], vl) ;
	      vrgout_r0s0 = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrgout_r0s0, vmall_r0s0, vl) ;
	      vrgout_r0s1 = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrgout_r0s1, vmall_r0s1, vl) ;
	      vrgout_r0s2 = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrgout_r0s2, vmall_r0s2, vl) ;
	      __vr vrgoutP_r0s0 = _vel_vshf_vvvsl(vrgout_r0s0, vrgout_r0s0, VE_VSHUFFLE_YUZU, vl) ;
	      __vr vrgoutP_r0s1 = _vel_vshf_vvvsl(vrgout_r0s1, vrgout_r0s1, VE_VSHUFFLE_YUZU, vl) ;
	      __vr vrgoutP_r0s2 = _vel_vshf_vvvsl(vrgout_r0s2, vrgout_r0s2, VE_VSHUFFLE_YUZU, vl) ;

#define FILTER_IC2(VROUT)								\
{											\
  const uint64_t kerValue01 = _vel_pack_f32p(pKerValue,					\
					    pKerValue+    kernHeight * kernWidth) ;	\
  vrsum01 = _vel_pvfmad_vvsvl(vrsum01, kerValue01, VROUT, vl) ;				\
}

	      FILTER_IC2(vrgoutP_r0s0) ; pKerValue++ ;
	      FILTER_IC2(vrgoutP_r0s1) ; pKerValue++ ;
	      FILTER_IC2(vrgoutP_r0s2) ; pKerValue++ ;

	      __vr vrgout_r1s0 = _vel_vldu_vssl(4,&pGOutChannel[gip+(padHeight-1)*gOutWidth+(padWidth-0)], vl) ;
	      __vr vrgout_r1s1 = _vel_vldu_vssl(4,&pGOutChannel[gip+(padHeight-1)*gOutWidth+(padWidth-1)], vl) ;
	      __vr vrgout_r1s2 = _vel_vldu_vssl(4,&pGOutChannel[gip+(padHeight-1)*gOutWidth+(padWidth-2)], vl) ;
	      vrgout_r1s0 = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrgout_r1s0, vmall_r1s0, vl) ;
	      vrgout_r1s2 = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrgout_r1s2, vmall_r1s2, vl) ;
	      __vr vrgoutP_r1s0 = _vel_vshf_vvvsl(vrgout_r1s0, vrgout_r1s0, VE_VSHUFFLE_YUZU, vl) ;
	      __vr vrgoutP_r1s1 = _vel_vshf_vvvsl(vrgout_r1s1, vrgout_r1s1, VE_VSHUFFLE_YUZU, vl) ;
	      __vr vrgoutP_r1s2 = _vel_vshf_vvvsl(vrgout_r1s2, vrgout_r1s2, VE_VSHUFFLE_YUZU, vl) ;

	      FILTER_IC2(vrgoutP_r1s0) ; pKerValue++ ;
	      FILTER_IC2(vrgoutP_r1s1) ; pKerValue++ ;
	      FILTER_IC2(vrgoutP_r1s2) ; pKerValue++ ;


	      __vr vrgout_r2s0 = _vel_vldu_vssl(4,&pGOutChannel[gip+(padHeight-2)*gOutWidth+(padWidth-0)], vl) ;
	      __vr vrgout_r2s1 = _vel_vldu_vssl(4,&pGOutChannel[gip+(padHeight-2)*gOutWidth+(padWidth-1)], vl) ;
	      __vr vrgout_r2s2 = _vel_vldu_vssl(4,&pGOutChannel[gip+(padHeight-2)*gOutWidth+(padWidth-2)], vl) ;
	      vrgout_r2s0 = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrgout_r2s0, vmall_s0_r2, vl) ;
	      vrgout_r2s1 = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrgout_r2s1, vmall_s1_r2, vl) ;
	      vrgout_r2s2 = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrgout_r2s2, vmall_s2_r2, vl) ;
	      __vr vrgoutP_r2s0 = _vel_vshf_vvvsl(vrgout_r2s0, vrgout_r2s0, VE_VSHUFFLE_YUZU, vl) ;
	      __vr vrgoutP_r2s1 = _vel_vshf_vvvsl(vrgout_r2s1, vrgout_r2s1, VE_VSHUFFLE_YUZU, vl) ;
	      __vr vrgoutP_r2s2 = _vel_vshf_vvvsl(vrgout_r2s2, vrgout_r2s2, VE_VSHUFFLE_YUZU, vl) ;

	      FILTER_IC2(vrgoutP_r2s0) ; pKerValue++ ;
	      FILTER_IC2(vrgoutP_r2s1) ; pKerValue++ ;
	      FILTER_IC2(vrgoutP_r2s2) ; pKerValue++ ;
#undef FILTER_IC2

	    } // gInChannel

	    _vel_vstu_vssl(vrsum01, 4, pGIn+gInIndex, vl) ;
	    _vel_vstl_vssl(vrsum01, 4, pGIn+gInIndex+  gInPixels, vl) ;

	  } // gInPixels

	  k+=2 ;
	}
	if ( ((gInChannelGroup >> 2) & 0x01) == 1 ) {
	  for (int64_t gip = 0; gip < gInPixels; gip+=VLEN) {
	    const int64_t vl = gInPixels - gip < VLEN ? gInPixels - gip : VLEN ;

	    const int64_t gInIndex = gInGroupOffset + ((n * gInChannel + k) * gInHeight ) * gInWidth + gip ;

	    __vr vrseq = _vel_vseq_vl(vl) ;			// hw
	    __vr vridx = _vel_vaddsl_vsvl(gip, vrseq, vl) ;	// op + hw

	    __vr vrsum01 = _vel_pvbrd_vsl(0UL, vl) ;
	    __vr vrsum23 = _vel_pvbrd_vsl(0UL, vl) ;

	    __vr vrh   = _vel_vdivsl_vvsl(vridx, gInWidth, vl) ;
	    __vr vrix   = _vel_vsubsl_vvvl(vridx, _vel_vmulul_vsvl(gInWidth,vrh, vl), vl) ;

	    __vr vry_r0 = _vel_vaddsl_vsvl(1, vrh, vl) ;
	    __vr vry_r2 = _vel_vaddsl_vsvl(-1, vrh, vl) ;

	    __vr vrx_s0 = _vel_vaddsl_vsvl(1, vrix, vl) ;
	    __vr vrx_s2 = _vel_vaddsl_vsvl(-1, vrix, vl) ;

	    __vm256 vmy_r0 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutHeight,vry_r0, vl), vl) ;
	    __vm256 vmy_r2 =  _vel_vfmklge_mvl(vry_r2, vl) ;

	    __vm256 vmx_s0 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutWidth,vrx_s0, vl), vl) ;
	    __vm256 vmx_s2 =  _vel_vfmklge_mvl(vrx_s2, vl) ;

	    __vm256 vmall_r0s0 = _vel_andm_mmm(vmy_r0,vmx_s0) ;
	    __vm256 vmall_r0s1 = vmy_r0 ;
	    __vm256 vmall_r0s2 = _vel_andm_mmm(vmy_r0,vmx_s2) ;

	    __vm256 vmall_r1s0 = vmx_s0 ;
	    __vm256 vmall_r1s2 = vmx_s2 ;

	    __vm256 vmall_s0_r2 = _vel_andm_mmm(vmy_r2,vmx_s0) ;
	    __vm256 vmall_s1_r2 = vmy_r2 ;
	    __vm256 vmall_s2_r2 = _vel_andm_mmm(vmy_r2,vmx_s2) ;

	    for (int64_t c=0; c<gOutChannelGroup; c++) {
	      const float *pGOutChannel = pGOut + gOutGroupOffset + ((n * gOutChannel + c) * gOutHeight * gOutWidth ) ;
	      const float *pKerValue = pKernel + kernGroupOffset + ((c * gInChannelGroup + k) * kernHeight) * kernWidth ;

	      /* memory access errors mihgt be caused */
	      __vr vrgout_r0s0 = _vel_vldu_vssl(4,&pGOutChannel[gip+(padHeight-0)*gOutWidth+(padWidth-0)], vl) ;
	      __vr vrgout_r0s1 = _vel_vldu_vssl(4,&pGOutChannel[gip+(padHeight-0)*gOutWidth+(padWidth-1)], vl) ;
	      __vr vrgout_r0s2 = _vel_vldu_vssl(4,&pGOutChannel[gip+(padHeight-0)*gOutWidth+(padWidth-2)], vl) ;
	      vrgout_r0s0 = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrgout_r0s0, vmall_r0s0, vl) ;
	      vrgout_r0s1 = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrgout_r0s1, vmall_r0s1, vl) ;
	      vrgout_r0s2 = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrgout_r0s2, vmall_r0s2, vl) ;
	      __vr vrgoutP_r0s0 = _vel_vshf_vvvsl(vrgout_r0s0, vrgout_r0s0, VE_VSHUFFLE_YUZU, vl) ;
	      __vr vrgoutP_r0s1 = _vel_vshf_vvvsl(vrgout_r0s1, vrgout_r0s1, VE_VSHUFFLE_YUZU, vl) ;
	      __vr vrgoutP_r0s2 = _vel_vshf_vvvsl(vrgout_r0s2, vrgout_r0s2, VE_VSHUFFLE_YUZU, vl) ;

#define FILTER_IC4(VROUT)								\
{											\
  const uint64_t kerValue01 = _vel_pack_f32p(pKerValue,					\
					    pKerValue+    kernHeight * kernWidth) ;	\
  const uint64_t kerValue23 = _vel_pack_f32p(pKerValue+ 2* kernHeight * kernWidth,	\
					    pKerValue+ 3* kernHeight * kernWidth) ;	\
  vrsum01 = _vel_pvfmad_vvsvl(vrsum01, kerValue01, VROUT, vl) ;				\
  vrsum23 = _vel_pvfmad_vvsvl(vrsum23, kerValue23, VROUT, vl) ;				\
}

	      FILTER_IC4(vrgoutP_r0s0) ; pKerValue++ ;
	      FILTER_IC4(vrgoutP_r0s1) ; pKerValue++ ;
	      FILTER_IC4(vrgoutP_r0s2) ; pKerValue++ ;

	      __vr vrgout_r1s0 = _vel_vldu_vssl(4,&pGOutChannel[gip+(padHeight-1)*gOutWidth+(padWidth-0)], vl) ;
	      __vr vrgout_r1s1 = _vel_vldu_vssl(4,&pGOutChannel[gip+(padHeight-1)*gOutWidth+(padWidth-1)], vl) ;
	      __vr vrgout_r1s2 = _vel_vldu_vssl(4,&pGOutChannel[gip+(padHeight-1)*gOutWidth+(padWidth-2)], vl) ;
	      vrgout_r1s0 = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrgout_r1s0, vmall_r1s0, vl) ;
	      vrgout_r1s2 = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrgout_r1s2, vmall_r1s2, vl) ;
	      __vr vrgoutP_r1s0 = _vel_vshf_vvvsl(vrgout_r1s0, vrgout_r1s0, VE_VSHUFFLE_YUZU, vl) ;
	      __vr vrgoutP_r1s1 = _vel_vshf_vvvsl(vrgout_r1s1, vrgout_r1s1, VE_VSHUFFLE_YUZU, vl) ;
	      __vr vrgoutP_r1s2 = _vel_vshf_vvvsl(vrgout_r1s2, vrgout_r1s2, VE_VSHUFFLE_YUZU, vl) ;

	      FILTER_IC4(vrgoutP_r1s0) ; pKerValue++ ;
	      FILTER_IC4(vrgoutP_r1s1) ; pKerValue++ ;
	      FILTER_IC4(vrgoutP_r1s2) ; pKerValue++ ;


	      __vr vrgout_r2s0 = _vel_vldu_vssl(4,&pGOutChannel[gip+(padHeight-2)*gOutWidth+(padWidth-0)], vl) ;
	      __vr vrgout_r2s1 = _vel_vldu_vssl(4,&pGOutChannel[gip+(padHeight-2)*gOutWidth+(padWidth-1)], vl) ;
	      __vr vrgout_r2s2 = _vel_vldu_vssl(4,&pGOutChannel[gip+(padHeight-2)*gOutWidth+(padWidth-2)], vl) ;
	      vrgout_r2s0 = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrgout_r2s0, vmall_s0_r2, vl) ;
	      vrgout_r2s1 = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrgout_r2s1, vmall_s1_r2, vl) ;
	      vrgout_r2s2 = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrgout_r2s2, vmall_s2_r2, vl) ;
	      __vr vrgoutP_r2s0 = _vel_vshf_vvvsl(vrgout_r2s0, vrgout_r2s0, VE_VSHUFFLE_YUZU, vl) ;
	      __vr vrgoutP_r2s1 = _vel_vshf_vvvsl(vrgout_r2s1, vrgout_r2s1, VE_VSHUFFLE_YUZU, vl) ;
	      __vr vrgoutP_r2s2 = _vel_vshf_vvvsl(vrgout_r2s2, vrgout_r2s2, VE_VSHUFFLE_YUZU, vl) ;

	      FILTER_IC4(vrgoutP_r2s0) ; pKerValue++ ;
	      FILTER_IC4(vrgoutP_r2s1) ; pKerValue++ ;
	      FILTER_IC4(vrgoutP_r2s2) ; pKerValue++ ;
#undef FILTER_IC4

	    } // gInChannel

	    _vel_vstu_vssl(vrsum01, 4, pGIn+gInIndex, vl) ;
	    _vel_vstl_vssl(vrsum01, 4, pGIn+gInIndex+  gInPixels, vl) ;
	    _vel_vstu_vssl(vrsum23, 4, pGIn+gInIndex+2*gInPixels, vl) ;
	    _vel_vstl_vssl(vrsum23, 4, pGIn+gInIndex+3*gInPixels, vl) ;

	  } // gInPixels

	  k+=4 ;
	}
	if ( ((gInChannelGroup >> 3) & 0x01) == 1 ) {
	  for (int64_t gip = 0; gip < gInPixels; gip+=VLEN) {
	    const int64_t vl = gInPixels - gip < VLEN ? gInPixels - gip : VLEN ;

	    const int64_t gInIndex = gInGroupOffset + ((n * gInChannel + k) * gInHeight ) * gInWidth + gip ;

	    __vr vrseq = _vel_vseq_vl(vl) ;			// hw
	    __vr vridx = _vel_vaddsl_vsvl(gip, vrseq, vl) ;	// op + hw

	    __vr vrsum01 = _vel_pvbrd_vsl(0UL, vl) ;
	    __vr vrsum23 = _vel_pvbrd_vsl(0UL, vl) ;
	    __vr vrsum45 = _vel_pvbrd_vsl(0UL, vl) ;
	    __vr vrsum67 = _vel_pvbrd_vsl(0UL, vl) ;

	    __vr vrh   = _vel_vdivsl_vvsl(vridx, gInWidth, vl) ;
	    __vr vrix   = _vel_vsubsl_vvvl(vridx, _vel_vmulul_vsvl(gInWidth,vrh, vl), vl) ;

	    __vr vry_r0 = _vel_vaddsl_vsvl(1, vrh, vl) ;
	    __vr vry_r2 = _vel_vaddsl_vsvl(-1, vrh, vl) ;

	    __vr vrx_s0 = _vel_vaddsl_vsvl(1, vrix, vl) ;
	    __vr vrx_s2 = _vel_vaddsl_vsvl(-1, vrix, vl) ;

	    __vm256 vmy_r0 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutHeight,vry_r0, vl), vl) ;
	    __vm256 vmy_r2 =  _vel_vfmklge_mvl(vry_r2, vl) ;

	    __vm256 vmx_s0 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutWidth,vrx_s0, vl), vl) ;
	    __vm256 vmx_s2 =  _vel_vfmklge_mvl(vrx_s2, vl) ;

	    __vm256 vmall_r0s0 = _vel_andm_mmm(vmy_r0,vmx_s0) ;
	    __vm256 vmall_r0s1 = vmy_r0 ;
	    __vm256 vmall_r0s2 = _vel_andm_mmm(vmy_r0,vmx_s2) ;

	    __vm256 vmall_r1s0 = vmx_s0 ;
	    __vm256 vmall_r1s2 = vmx_s2 ;

	    __vm256 vmall_s0_r2 = _vel_andm_mmm(vmy_r2,vmx_s0) ;
	    __vm256 vmall_s1_r2 = vmy_r2 ;
	    __vm256 vmall_s2_r2 = _vel_andm_mmm(vmy_r2,vmx_s2) ;

	    for (int64_t c=0; c<gOutChannelGroup; c++) {
	      const float *pGOutChannel = pGOut + gOutGroupOffset + ((n * gOutChannel + c) * gOutHeight * gOutWidth ) ;
	      const float *pKerValue = pKernel + kernGroupOffset + ((c * gInChannelGroup + k) * kernHeight) * kernWidth ;

	      /* memory access errors mihgt be caused */
	      __vr vrgout_r0s0 = _vel_vldu_vssl(4,&pGOutChannel[gip+(padHeight-0)*gOutWidth+(padWidth-0)], vl) ;
	      __vr vrgout_r0s1 = _vel_vldu_vssl(4,&pGOutChannel[gip+(padHeight-0)*gOutWidth+(padWidth-1)], vl) ;
	      __vr vrgout_r0s2 = _vel_vldu_vssl(4,&pGOutChannel[gip+(padHeight-0)*gOutWidth+(padWidth-2)], vl) ;
	      vrgout_r0s0 = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrgout_r0s0, vmall_r0s0, vl) ;
	      vrgout_r0s1 = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrgout_r0s1, vmall_r0s1, vl) ;
	      vrgout_r0s2 = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrgout_r0s2, vmall_r0s2, vl) ;
	      __vr vrgoutP_r0s0 = _vel_vshf_vvvsl(vrgout_r0s0, vrgout_r0s0, VE_VSHUFFLE_YUZU, vl) ;
	      __vr vrgoutP_r0s1 = _vel_vshf_vvvsl(vrgout_r0s1, vrgout_r0s1, VE_VSHUFFLE_YUZU, vl) ;
	      __vr vrgoutP_r0s2 = _vel_vshf_vvvsl(vrgout_r0s2, vrgout_r0s2, VE_VSHUFFLE_YUZU, vl) ;

#define FILTER_IC8(VROUT)								\
{											\
  const uint64_t kerValue01 = _vel_pack_f32p(pKerValue,					\
					     pKerValue+    kernHeight * kernWidth) ;	\
  const uint64_t kerValue23 = _vel_pack_f32p(pKerValue+ 2* kernHeight * kernWidth,	\
					     pKerValue+ 3* kernHeight * kernWidth) ;	\
  vrsum01 = _vel_pvfmad_vvsvl(vrsum01, kerValue01, VROUT, vl) ;				\
  vrsum23 = _vel_pvfmad_vvsvl(vrsum23, kerValue23, VROUT, vl) ;				\
  const uint64_t kerValue45 = _vel_pack_f32p(pKerValue+ 4* kernHeight * kernWidth,	\
					     pKerValue+ 5* kernHeight * kernWidth) ;	\
  const uint64_t kerValue67 = _vel_pack_f32p(pKerValue+ 6* kernHeight * kernWidth,	\
					     pKerValue+ 7* kernHeight * kernWidth) ;	\
  vrsum45 = _vel_pvfmad_vvsvl(vrsum45, kerValue45, VROUT, vl) ;				\
  vrsum67 = _vel_pvfmad_vvsvl(vrsum67, kerValue67, VROUT, vl) ;				\
}

	      FILTER_IC8(vrgoutP_r0s0) ; pKerValue++ ;
	      FILTER_IC8(vrgoutP_r0s1) ; pKerValue++ ;
	      FILTER_IC8(vrgoutP_r0s2) ; pKerValue++ ;

	      __vr vrgout_r1s0 = _vel_vldu_vssl(4,&pGOutChannel[gip+(padHeight-1)*gOutWidth+(padWidth-0)], vl) ;
	      __vr vrgout_r1s1 = _vel_vldu_vssl(4,&pGOutChannel[gip+(padHeight-1)*gOutWidth+(padWidth-1)], vl) ;
	      __vr vrgout_r1s2 = _vel_vldu_vssl(4,&pGOutChannel[gip+(padHeight-1)*gOutWidth+(padWidth-2)], vl) ;
	      vrgout_r1s0 = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrgout_r1s0, vmall_r1s0, vl) ;
	      vrgout_r1s2 = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrgout_r1s2, vmall_r1s2, vl) ;
	      __vr vrgoutP_r1s0 = _vel_vshf_vvvsl(vrgout_r1s0, vrgout_r1s0, VE_VSHUFFLE_YUZU, vl) ;
	      __vr vrgoutP_r1s1 = _vel_vshf_vvvsl(vrgout_r1s1, vrgout_r1s1, VE_VSHUFFLE_YUZU, vl) ;
	      __vr vrgoutP_r1s2 = _vel_vshf_vvvsl(vrgout_r1s2, vrgout_r1s2, VE_VSHUFFLE_YUZU, vl) ;

	      FILTER_IC8(vrgoutP_r1s0) ; pKerValue++ ;
	      FILTER_IC8(vrgoutP_r1s1) ; pKerValue++ ;
	      FILTER_IC8(vrgoutP_r1s2) ; pKerValue++ ;


	      __vr vrgout_r2s0 = _vel_vldu_vssl(4,&pGOutChannel[gip+(padHeight-2)*gOutWidth+(padWidth-0)], vl) ;
	      __vr vrgout_r2s1 = _vel_vldu_vssl(4,&pGOutChannel[gip+(padHeight-2)*gOutWidth+(padWidth-1)], vl) ;
	      __vr vrgout_r2s2 = _vel_vldu_vssl(4,&pGOutChannel[gip+(padHeight-2)*gOutWidth+(padWidth-2)], vl) ;
	      vrgout_r2s0 = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrgout_r2s0, vmall_s0_r2, vl) ;
	      vrgout_r2s1 = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrgout_r2s1, vmall_s1_r2, vl) ;
	      vrgout_r2s2 = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrgout_r2s2, vmall_s2_r2, vl) ;
	      __vr vrgoutP_r2s0 = _vel_vshf_vvvsl(vrgout_r2s0, vrgout_r2s0, VE_VSHUFFLE_YUZU, vl) ;
	      __vr vrgoutP_r2s1 = _vel_vshf_vvvsl(vrgout_r2s1, vrgout_r2s1, VE_VSHUFFLE_YUZU, vl) ;
	      __vr vrgoutP_r2s2 = _vel_vshf_vvvsl(vrgout_r2s2, vrgout_r2s2, VE_VSHUFFLE_YUZU, vl) ;

	      FILTER_IC8(vrgoutP_r2s0) ; pKerValue++ ;
	      FILTER_IC8(vrgoutP_r2s1) ; pKerValue++ ;
	      FILTER_IC8(vrgoutP_r2s2) ; pKerValue++ ;
#undef FILTER_IC8

	    } // gInChannel

	    _vel_vstu_vssl(vrsum01, 4, pGIn+gInIndex, vl) ;
	    _vel_vstl_vssl(vrsum01, 4, pGIn+gInIndex+  gInPixels, vl) ;
	    _vel_vstu_vssl(vrsum23, 4, pGIn+gInIndex+2*gInPixels, vl) ;
	    _vel_vstl_vssl(vrsum23, 4, pGIn+gInIndex+3*gInPixels, vl) ;
	    _vel_vstu_vssl(vrsum45, 4, pGIn+gInIndex+4*gInPixels, vl) ;
	    _vel_vstl_vssl(vrsum45, 4, pGIn+gInIndex+5*gInPixels, vl) ;
	    _vel_vstu_vssl(vrsum67, 4, pGIn+gInIndex+6*gInPixels, vl) ;
	    _vel_vstl_vssl(vrsum67, 4, pGIn+gInIndex+7*gInPixels, vl) ;

	  } // gInPixels

	  k+=8 ;
	}
	for (; k<gInChannelGroup; k+=16) {
	  for (int64_t gip = 0; gip < gInPixels; gip+=VLEN) {
	    const int64_t vl = gInPixels - gip < VLEN ? gInPixels - gip : VLEN ;

	    const int64_t gInIndex = gInGroupOffset + ((n * gInChannel + k) * gInHeight ) * gInWidth + gip ;

	    __vr vrseq = _vel_vseq_vl(vl) ;			// hw
	    __vr vridx = _vel_vaddsl_vsvl(gip, vrseq, vl) ;	// op + hw

	    __vr vrsum01 = _vel_pvbrd_vsl(0UL, vl) ;
	    __vr vrsum23 = _vel_pvbrd_vsl(0UL, vl) ;
	    __vr vrsum45 = _vel_pvbrd_vsl(0UL, vl) ;
	    __vr vrsum67 = _vel_pvbrd_vsl(0UL, vl) ;
	    __vr vrsum89 = _vel_pvbrd_vsl(0UL, vl) ;
	    __vr vrsumAB = _vel_pvbrd_vsl(0UL, vl) ;
	    __vr vrsumCD = _vel_pvbrd_vsl(0UL, vl) ;
	    __vr vrsumEF = _vel_pvbrd_vsl(0UL, vl) ;

	    __vr vrh   = _vel_vdivsl_vvsl(vridx, gInWidth, vl) ;
	    __vr vrix   = _vel_vsubsl_vvvl(vridx, _vel_vmulul_vsvl(gInWidth,vrh, vl), vl) ;

	    __vr vry_r0 = _vel_vaddsl_vsvl(1, vrh, vl) ;
	    __vr vry_r2 = _vel_vaddsl_vsvl(-1, vrh, vl) ;

	    __vr vrx_s0 = _vel_vaddsl_vsvl(1, vrix, vl) ;
	    __vr vrx_s2 = _vel_vaddsl_vsvl(-1, vrix, vl) ;

	    __vm256 vmy_r0 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutHeight,vry_r0, vl), vl) ;
	    __vm256 vmy_r2 =  _vel_vfmklge_mvl(vry_r2, vl) ;

	    __vm256 vmx_s0 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutWidth,vrx_s0, vl), vl) ;
	    __vm256 vmx_s2 =  _vel_vfmklge_mvl(vrx_s2, vl) ;

	    __vm256 vmall_r0s0 = _vel_andm_mmm(vmy_r0,vmx_s0) ;
	    __vm256 vmall_r0s1 = vmy_r0 ;
	    __vm256 vmall_r0s2 = _vel_andm_mmm(vmy_r0,vmx_s2) ;

	    __vm256 vmall_r1s0 = vmx_s0 ;
	    __vm256 vmall_r1s2 = vmx_s2 ;

	    __vm256 vmall_s0_r2 = _vel_andm_mmm(vmy_r2,vmx_s0) ;
	    __vm256 vmall_s1_r2 = vmy_r2 ;
	    __vm256 vmall_s2_r2 = _vel_andm_mmm(vmy_r2,vmx_s2) ;

	    for (int64_t c=0; c<gOutChannelGroup; c++) {
	      const float *pGOutChannel = pGOut + gOutGroupOffset + ((n * gOutChannel + c) * gOutHeight * gOutWidth ) ;
	      const float *pKerValue = pKernel + kernGroupOffset + ((c * gInChannelGroup + k) * kernHeight) * kernWidth ;

	      /* memory access errors mihgt be caused */
	      __vr vrgout_r0s0 = _vel_vldu_vssl(4,&pGOutChannel[gip+(padHeight-0)*gOutWidth+(padWidth-0)], vl) ;
	      __vr vrgout_r0s1 = _vel_vldu_vssl(4,&pGOutChannel[gip+(padHeight-0)*gOutWidth+(padWidth-1)], vl) ;
	      __vr vrgout_r0s2 = _vel_vldu_vssl(4,&pGOutChannel[gip+(padHeight-0)*gOutWidth+(padWidth-2)], vl) ;
	      vrgout_r0s0 = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrgout_r0s0, vmall_r0s0, vl) ;
	      vrgout_r0s1 = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrgout_r0s1, vmall_r0s1, vl) ;
	      vrgout_r0s2 = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrgout_r0s2, vmall_r0s2, vl) ;
	      __vr vrgoutP_r0s0 = _vel_vshf_vvvsl(vrgout_r0s0, vrgout_r0s0, VE_VSHUFFLE_YUZU, vl) ;
	      __vr vrgoutP_r0s1 = _vel_vshf_vvvsl(vrgout_r0s1, vrgout_r0s1, VE_VSHUFFLE_YUZU, vl) ;
	      __vr vrgoutP_r0s2 = _vel_vshf_vvvsl(vrgout_r0s2, vrgout_r0s2, VE_VSHUFFLE_YUZU, vl) ;

#define FILTER_IC16(VROUT)								\
{											\
  const uint64_t kerValue01 = _vel_pack_f32p(pKerValue,					\
					     pKerValue+    kernHeight * kernWidth) ;	\
  const uint64_t kerValue23 = _vel_pack_f32p(pKerValue+ 2* kernHeight * kernWidth,	\
					     pKerValue+ 3* kernHeight * kernWidth) ;	\
  vrsum01 = _vel_pvfmad_vvsvl(vrsum01, kerValue01, VROUT, vl) ;				\
  vrsum23 = _vel_pvfmad_vvsvl(vrsum23, kerValue23, VROUT, vl) ;				\
  const uint64_t kerValue45 = _vel_pack_f32p(pKerValue+ 4* kernHeight * kernWidth,	\
				             pKerValue+ 5* kernHeight * kernWidth) ;	\
  const uint64_t kerValue67 = _vel_pack_f32p(pKerValue+ 6* kernHeight * kernWidth,	\
					     pKerValue+ 7* kernHeight * kernWidth) ;	\
  vrsum45 = _vel_pvfmad_vvsvl(vrsum45, kerValue45, VROUT, vl) ;				\
  vrsum67 = _vel_pvfmad_vvsvl(vrsum67, kerValue67, VROUT, vl) ;				\
  const uint64_t kerValue89 = _vel_pack_f32p(pKerValue+ 8* kernHeight * kernWidth,	\
					     pKerValue+ 9* kernHeight * kernWidth) ;	\
  const uint64_t kerValueAB = _vel_pack_f32p(pKerValue+10* kernHeight * kernWidth,	\
					     pKerValue+11* kernHeight * kernWidth) ;	\
  vrsum89 = _vel_pvfmad_vvsvl(vrsum89, kerValue89, VROUT, vl) ;				\
  vrsumAB = _vel_pvfmad_vvsvl(vrsumAB, kerValueAB, VROUT, vl) ;				\
  const uint64_t kerValueCD = _vel_pack_f32p(pKerValue+12* kernHeight * kernWidth,	\
					     pKerValue+13* kernHeight * kernWidth) ;	\
  const uint64_t kerValueEF = _vel_pack_f32p(pKerValue+14* kernHeight * kernWidth,	\
					     pKerValue+15* kernHeight * kernWidth) ;	\
  vrsumCD = _vel_pvfmad_vvsvl(vrsumCD, kerValueCD, VROUT, vl) ;				\
  vrsumEF = _vel_pvfmad_vvsvl(vrsumEF, kerValueEF, VROUT, vl) ;				\
}

	      FILTER_IC16(vrgoutP_r0s0) ; pKerValue++ ;
	      FILTER_IC16(vrgoutP_r0s1) ; pKerValue++ ;
	      FILTER_IC16(vrgoutP_r0s2) ; pKerValue++ ;

	      __vr vrgout_r1s0 = _vel_vldu_vssl(4,&pGOutChannel[gip+(padHeight-1)*gOutWidth+(padWidth-0)], vl) ;
	      __vr vrgout_r1s1 = _vel_vldu_vssl(4,&pGOutChannel[gip+(padHeight-1)*gOutWidth+(padWidth-1)], vl) ;
	      __vr vrgout_r1s2 = _vel_vldu_vssl(4,&pGOutChannel[gip+(padHeight-1)*gOutWidth+(padWidth-2)], vl) ;
	      vrgout_r1s0 = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrgout_r1s0, vmall_r1s0, vl) ;
	      vrgout_r1s2 = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrgout_r1s2, vmall_r1s2, vl) ;
	      __vr vrgoutP_r1s0 = _vel_vshf_vvvsl(vrgout_r1s0, vrgout_r1s0, VE_VSHUFFLE_YUZU, vl) ;
	      __vr vrgoutP_r1s1 = _vel_vshf_vvvsl(vrgout_r1s1, vrgout_r1s1, VE_VSHUFFLE_YUZU, vl) ;
	      __vr vrgoutP_r1s2 = _vel_vshf_vvvsl(vrgout_r1s2, vrgout_r1s2, VE_VSHUFFLE_YUZU, vl) ;

	      FILTER_IC16(vrgoutP_r1s0) ; pKerValue++ ;
	      FILTER_IC16(vrgoutP_r1s1) ; pKerValue++ ;
	      FILTER_IC16(vrgoutP_r1s2) ; pKerValue++ ;


	      __vr vrgout_r2s0 = _vel_vldu_vssl(4,&pGOutChannel[gip+(padHeight-2)*gOutWidth+(padWidth-0)], vl) ;
	      __vr vrgout_r2s1 = _vel_vldu_vssl(4,&pGOutChannel[gip+(padHeight-2)*gOutWidth+(padWidth-1)], vl) ;
	      __vr vrgout_r2s2 = _vel_vldu_vssl(4,&pGOutChannel[gip+(padHeight-2)*gOutWidth+(padWidth-2)], vl) ;
	      vrgout_r2s0 = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrgout_r2s0, vmall_s0_r2, vl) ;
	      vrgout_r2s1 = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrgout_r2s1, vmall_s1_r2, vl) ;
	      vrgout_r2s2 = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrgout_r2s2, vmall_s2_r2, vl) ;
	      __vr vrgoutP_r2s0 = _vel_vshf_vvvsl(vrgout_r2s0, vrgout_r2s0, VE_VSHUFFLE_YUZU, vl) ;
	      __vr vrgoutP_r2s1 = _vel_vshf_vvvsl(vrgout_r2s1, vrgout_r2s1, VE_VSHUFFLE_YUZU, vl) ;
	      __vr vrgoutP_r2s2 = _vel_vshf_vvvsl(vrgout_r2s2, vrgout_r2s2, VE_VSHUFFLE_YUZU, vl) ;

	      FILTER_IC16(vrgoutP_r2s0) ; pKerValue++ ;
	      FILTER_IC16(vrgoutP_r2s1) ; pKerValue++ ;
	      FILTER_IC16(vrgoutP_r2s2) ; pKerValue++ ;
#undef FILTER_IC16

	    } // gInChannel

	    _vel_vstu_vssl(vrsum01, 4, pGIn+gInIndex, vl) ;
	    _vel_vstl_vssl(vrsum01, 4, pGIn+gInIndex+  gInPixels, vl) ;
	    _vel_vstu_vssl(vrsum23, 4, pGIn+gInIndex+2*gInPixels, vl) ;
	    _vel_vstl_vssl(vrsum23, 4, pGIn+gInIndex+3*gInPixels, vl) ;
	    _vel_vstu_vssl(vrsum45, 4, pGIn+gInIndex+4*gInPixels, vl) ;
	    _vel_vstl_vssl(vrsum45, 4, pGIn+gInIndex+5*gInPixels, vl) ;
	    _vel_vstu_vssl(vrsum67, 4, pGIn+gInIndex+6*gInPixels, vl) ;
	    _vel_vstl_vssl(vrsum67, 4, pGIn+gInIndex+7*gInPixels, vl) ;
	    _vel_vstu_vssl(vrsum89, 4, pGIn+gInIndex+8*gInPixels, vl) ;
	    _vel_vstl_vssl(vrsum89, 4, pGIn+gInIndex+9*gInPixels, vl) ;
	    _vel_vstu_vssl(vrsumAB, 4, pGIn+gInIndex+10*gInPixels, vl) ;
	    _vel_vstl_vssl(vrsumAB, 4, pGIn+gInIndex+11*gInPixels, vl) ;
	    _vel_vstu_vssl(vrsumCD, 4, pGIn+gInIndex+12*gInPixels, vl) ;
	    _vel_vstl_vssl(vrsumCD, 4, pGIn+gInIndex+13*gInPixels, vl) ;
	    _vel_vstu_vssl(vrsumEF, 4, pGIn+gInIndex+14*gInPixels, vl) ;
	    _vel_vstl_vssl(vrsumEF, 4, pGIn+gInIndex+15*gInPixels, vl) ;
	  } // gInPixels

	} // gOutChannel

      } // group
    } // batch
  }


  return VEDNN_SUCCESS;
}
