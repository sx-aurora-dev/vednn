#include <stdint.h>
#include <stdlib.h>

#include "vednn.h"

#include "velintrin.h"
#define VLEN	(256)


vednnError_t
vednnConvolutionForwardAddBias_direct_dil1_str1_padsame_ker3_c1024x(
    const vednnTensorParam_t * restrict 	pParamIn,
    const void * restrict 			pDataIn,
    const vednnFilterParam_t * restrict 	pParamKernel,
    const void * restrict 			pDataKernel,
    const vednnBiasParam_t * restrict 		pParamBias,
    const void * restrict 			pDataBias,
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
  const int64_t kernWidth  = pParamKernel->width;		/* must be 3 */
  const int64_t kernHeight = pParamKernel->height;		/* must be 3 */

  const int64_t group          = pParamConv->group;
//  const int64_t strideWidth    = pParamConv->strideWidth;	/* must be 1 */
//  const int64_t strideHeight   = pParamConv->strideHeight;	/* must be 1 */
  const int64_t padWidth       = pParamConv->padWidth;
  const int64_t padHeight      = pParamConv->padHeight;
//  const int64_t dilationWidth  = pParamConv->dilationWidth;	/* must be 1 */
//  const int64_t dilationHeight = pParamConv->dilationHeight;	/* must be 1 */

  const int64_t inChannelGroup  = inChannel  / group;   // equal to pDataKernel->inChannel
  const int64_t outChannelGroup = outChannel / group;   // equal to pDataKernel->outChannel

  const float * restrict pIn     = pDataIn;
  const float * restrict pKernel = pDataKernel;
  const float * restrict pBias   = pDataBias;
  float * restrict const pOut    = pDataOut;

  const int oPixels= outHeight*outWidth ;

  {
    for (int64_t n = 0; n < batch; n++) {
      for (int64_t g = 0; g < group; g++) {
	const int64_t inGroupOffset   = g * inChannelGroup * inHeight * inWidth;
	const int64_t outGroupOffset  = g * outChannelGroup * outHeight * outWidth;
	const int64_t kernGroupOffset = g * outChannelGroup * inChannelGroup * kernHeight * kernWidth;

	int k = 0 ;
	if ( (outChannelGroup & 0x01) == 1 ) {
	  int64_t outIndex = outGroupOffset + (n * outChannel + k) * oPixels ;

	  const float bias = (pBias == NULL) ? 0.0 : pBias[g * outChannelGroup + k];

	  for (int64_t op = 0; op < oPixels; op+=VLEN) {
	    const int64_t vl = oPixels - op < VLEN ? oPixels - op : VLEN ;

	    __vr vrsum = _vel_vbrds_vsl(bias, vl) ;

	    __vr vrseq = _vel_vseq_vl(vl) ;			// xy
	    __vr vridx = _vel_vaddsl_vsvl(op, vrseq, vl) ;	// op + xy

	    __vr vry   = _vel_vdivsl_vvsl(vridx, outWidth, vl) ;
	    __vr vrx   = _vel_vsubsl_vvvl(vridx, _vel_vmulul_vsvl(outWidth,vry, vl), vl) ;

	    __vr vrh_r0 = _vel_vaddsl_vsvl( -padHeight, vry, vl) ;
	    __vr vrh_r2 = _vel_vaddsl_vsvl(2-padHeight, vry, vl) ;

	    __vr vrw_s0 = _vel_vaddsl_vsvl( -padWidth, vrx, vl) ;
	    __vr vrw_s2 = _vel_vaddsl_vsvl(2-padWidth, vrx, vl) ;

	    __vm256 vm01_r0 =  _vel_vfmklge_mvl(vrh_r0, vl) ;
	    __vm256 vm01_r2 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(inHeight,vrh_r2, vl), vl) ;

	    __vm256 vm23_s0  =  _vel_vfmklge_mvl(vrw_s0, vl) ;
	    __vm256 vm23_s2  =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(inWidth,vrw_s2, vl), vl) ;


	    __vm256 vmall_r0s0 = _vel_andm_mmm(vm01_r0,vm23_s0) ;
	    __vm256 vmall_r0s1 = vm01_r0 ;
	    __vm256 vmall_r0s2 = _vel_andm_mmm(vm01_r0, vm23_s2) ;

	    __vm256 vmall_r1s0 = vm23_s0 ;
	    __vm256 vmall_r1s2 = vm23_s2 ;

	    __vm256 vmall_r2s0 = _vel_andm_mmm(vm01_r2,vm23_s0) ;
	    __vm256 vmall_r2s1 = vm01_r2 ;
	    __vm256 vmall_r2s2 = _vel_andm_mmm(vm01_r2, vm23_s2) ;

	    for (int64_t c = 0; c < inChannelGroup; c++) {
	      const float *pKerValue = pKernel + kernGroupOffset + ((k * inChannelGroup + c) * kernHeight) * kernWidth ;

	      const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;

	      /* memory access errors mihgt be caused */
	      __vr vrin_r0s0 = _vel_vldu_vssl(4,&pInChannel[op-inWidth-1], vl) ;
	      __vr vrin_r0s1 = _vel_vldu_vssl(4,&pInChannel[op-inWidth  ], vl) ;
	      __vr vrin_r0s2 = _vel_vldu_vssl(4,&pInChannel[op-inWidth+1], vl) ;
	      __vr vrin_r1s0 = _vel_vldu_vssl(4,&pInChannel[op+       -1], vl) ;
	      __vr vrin_r1s1 = _vel_vldu_vssl(4,&pInChannel[op          ], vl) ;
	      __vr vrin_r1s2 = _vel_vldu_vssl(4,&pInChannel[op+       +1], vl) ;
	      __vr vrin_r2s0 = _vel_vldu_vssl(4,&pInChannel[op+inWidth-1], vl) ;
	      __vr vrin_r2s1 = _vel_vldu_vssl(4,&pInChannel[op+inWidth  ], vl) ;
	      __vr vrin_r2s2 = _vel_vldu_vssl(4,&pInChannel[op+inWidth+1], vl) ;

	      __vr vrzerof = _vel_vbrds_vsl(0.0f, vl) ;

	      vrin_r0s0 = _vel_vmrg_vvvml(vrzerof, vrin_r0s0, vmall_r0s0, vl) ;
	      vrin_r0s1 = _vel_vmrg_vvvml(vrzerof, vrin_r0s1, vmall_r0s1, vl) ;
	      vrin_r0s2 = _vel_vmrg_vvvml(vrzerof, vrin_r0s2, vmall_r0s2, vl) ;

	      vrin_r1s0 = _vel_vmrg_vvvml(vrzerof, vrin_r1s0, vmall_r1s0, vl) ;
	      vrin_r1s2 = _vel_vmrg_vvvml(vrzerof, vrin_r1s2, vmall_r1s2, vl) ;

	      vrin_r2s0 = _vel_vmrg_vvvml(vrzerof, vrin_r2s0, vmall_r2s0, vl) ;
	      vrin_r2s1 = _vel_vmrg_vvvml(vrzerof, vrin_r2s1, vmall_r2s1, vl) ;
	      vrin_r2s2 = _vel_vmrg_vvvml(vrzerof, vrin_r2s2, vmall_r2s2, vl) ;

#define FILTER_OC1(VRIN,VRSUM)									\
{												\
  VRSUM = _vel_vfmads_vvsvl(VRSUM, *pKerValue, VRIN, vl) ;					\
}
		FILTER_OC1(vrin_r0s0, vrsum) ; pKerValue++ ;
		FILTER_OC1(vrin_r0s1, vrsum) ; pKerValue++ ;
		FILTER_OC1(vrin_r0s2, vrsum) ; pKerValue++ ;
		FILTER_OC1(vrin_r1s0, vrsum) ; pKerValue++ ;
		FILTER_OC1(vrin_r1s1, vrsum) ; pKerValue++ ;
		FILTER_OC1(vrin_r1s2, vrsum) ; pKerValue++ ;
		FILTER_OC1(vrin_r2s0, vrsum) ; pKerValue++ ;
		FILTER_OC1(vrin_r2s1, vrsum) ; pKerValue++ ;
		FILTER_OC1(vrin_r2s2, vrsum) ; pKerValue++ ;
#undef FILTER_OC1

	    } // inChannel

	    _vel_vstu_vssl(vrsum, 4, pOut+outIndex, vl) ;

	    outIndex += vl ;
	  } // outPixels
	  k++ ;
	}
	if ( ((outChannelGroup >> 1) & 0x01) == 1 ) {
	  int64_t outIndex = outGroupOffset + (n * outChannel + k  ) * oPixels ;

	  const float bias0 = (pBias == NULL) ? 0.0 : pBias[g * outChannelGroup + k  ];
	  const float bias1 = (pBias == NULL) ? 0.0 : pBias[g * outChannelGroup + k+1];

	  const uint64_t bias01 = _vel_pack_f32p(&bias0, &bias1) ;

	  for (int64_t op = 0; op < oPixels; op+=VLEN) {
	    const int64_t vl = oPixels - op < VLEN ? oPixels - op : VLEN ;

	    __vr vrsum01 = _vel_pvbrd_vsl(bias01, vl) ;

	    __vr vrseq = _vel_vseq_vl(vl) ;			// xy
	    __vr vridx = _vel_vaddsl_vsvl(op, vrseq, vl) ;	// op + xy

	    __vr vry   = _vel_vdivsl_vvsl(vridx, outWidth, vl) ;
	    __vr vrx   = _vel_vsubsl_vvvl(vridx, _vel_vmulul_vsvl(outWidth,vry, vl), vl) ;

	    __vr vrh_r0 = _vel_vaddsl_vsvl( -padHeight, vry, vl) ;
	    __vr vrh_r2 = _vel_vaddsl_vsvl(2-padHeight, vry, vl) ;

	    __vr vrw_s0 = _vel_vaddsl_vsvl( -padWidth, vrx, vl) ;
	    __vr vrw_s2 = _vel_vaddsl_vsvl(2-padWidth, vrx, vl) ;

	    __vm256 vm01_r0 =  _vel_vfmklge_mvl(vrh_r0, vl) ;
	    __vm256 vm01_r2 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(inHeight,vrh_r2, vl), vl) ;

	    __vm256 vm23_s0  =  _vel_vfmklge_mvl(vrw_s0, vl) ;
	    __vm256 vm23_s2  =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(inWidth,vrw_s2, vl), vl) ;


	    __vm256 vmall_r0s0 = _vel_andm_mmm(vm01_r0,vm23_s0) ;
	    __vm256 vmall_r0s1 = vm01_r0 ;
	    __vm256 vmall_r0s2 = _vel_andm_mmm(vm01_r0, vm23_s2) ;

	    __vm256 vmall_r1s0 = vm23_s0 ;
	    __vm256 vmall_r1s2 = vm23_s2 ;

	    __vm256 vmall_r2s0 = _vel_andm_mmm(vm01_r2,vm23_s0) ;
	    __vm256 vmall_r2s1 = vm01_r2 ;
	    __vm256 vmall_r2s2 = _vel_andm_mmm(vm01_r2, vm23_s2) ;

	    for (int64_t c = 0; c < inChannelGroup; c++) {
	      const float *pKerValue = pKernel + kernGroupOffset + ((k * inChannelGroup + c) * kernHeight) * kernWidth ;

	      const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;

	      /* memory access errors mihgt be caused */
	      __vr vrin_r0s0 = _vel_vldu_vssl(4,&pInChannel[op-inWidth-1], vl) ;
	      __vr vrin_r0s1 = _vel_vldu_vssl(4,&pInChannel[op-inWidth  ], vl) ;
	      __vr vrin_r0s2 = _vel_vldu_vssl(4,&pInChannel[op-inWidth+1], vl) ;
	      __vr vrin_r1s0 = _vel_vldu_vssl(4,&pInChannel[op+       -1], vl) ;
	      __vr vrin_r1s1 = _vel_vldu_vssl(4,&pInChannel[op          ], vl) ;
	      __vr vrin_r1s2 = _vel_vldu_vssl(4,&pInChannel[op+       +1], vl) ;
	      __vr vrin_r2s0 = _vel_vldu_vssl(4,&pInChannel[op+inWidth-1], vl) ;
	      __vr vrin_r2s1 = _vel_vldu_vssl(4,&pInChannel[op+inWidth  ], vl) ;
	      __vr vrin_r2s2 = _vel_vldu_vssl(4,&pInChannel[op+inWidth+1], vl) ;

	      __vr vrzerof = _vel_vbrds_vsl(0.0f, vl) ;

	      vrin_r0s0 = _vel_vmrg_vvvml(vrzerof, vrin_r0s0, vmall_r0s0, vl) ;
	      vrin_r0s1 = _vel_vmrg_vvvml(vrzerof, vrin_r0s1, vmall_r0s1, vl) ;
	      vrin_r0s2 = _vel_vmrg_vvvml(vrzerof, vrin_r0s2, vmall_r0s2, vl) ;
	      __vr vrinP_r0s0 = _vel_vshf_vvvsl(vrin_r0s0, vrin_r0s0, VE_VSHUFFLE_YUZU, vl) ;
	      __vr vrinP_r0s1 = _vel_vshf_vvvsl(vrin_r0s1, vrin_r0s1, VE_VSHUFFLE_YUZU, vl) ;
	      __vr vrinP_r0s2 = _vel_vshf_vvvsl(vrin_r0s2, vrin_r0s2, VE_VSHUFFLE_YUZU, vl) ;

	      vrin_r1s0 = _vel_vmrg_vvvml(vrzerof, vrin_r1s0, vmall_r1s0, vl) ;
	      vrin_r1s2 = _vel_vmrg_vvvml(vrzerof, vrin_r1s2, vmall_r1s2, vl) ;
	      __vr vrinP_r1s0 = _vel_vshf_vvvsl(vrin_r1s0, vrin_r1s0, VE_VSHUFFLE_YUZU, vl) ;
	      __vr vrinP_r1s1 = _vel_vshf_vvvsl(vrin_r1s1, vrin_r1s1, VE_VSHUFFLE_YUZU, vl) ;
	      __vr vrinP_r1s2 = _vel_vshf_vvvsl(vrin_r1s2, vrin_r1s2, VE_VSHUFFLE_YUZU, vl) ;

	      vrin_r2s0 = _vel_vmrg_vvvml(vrzerof, vrin_r2s0, vmall_r2s0, vl) ;
	      vrin_r2s1 = _vel_vmrg_vvvml(vrzerof, vrin_r2s1, vmall_r2s1, vl) ;
	      vrin_r2s2 = _vel_vmrg_vvvml(vrzerof, vrin_r2s2, vmall_r2s2, vl) ;
	      __vr vrinP_r2s0 = _vel_vshf_vvvsl(vrin_r2s0, vrin_r2s0, VE_VSHUFFLE_YUZU, vl) ;
	      __vr vrinP_r2s1 = _vel_vshf_vvvsl(vrin_r2s1, vrin_r2s1, VE_VSHUFFLE_YUZU, vl) ;
	      __vr vrinP_r2s2 = _vel_vshf_vvvsl(vrin_r2s2, vrin_r2s2, VE_VSHUFFLE_YUZU, vl) ;

#define FILTER_OC2(VRIN,VRSUM)									\
{												\
  const uint64_t kerValue = _vel_pack_f32p(pKerValue,						\
					  pKerValue + inChannelGroup * kernHeight * kernWidth) ;\
  VRSUM = _vel_pvfmad_vvsvl(VRSUM, kerValue, VRIN, vl) ;						\
}
		FILTER_OC2(vrinP_r0s0, vrsum01) ; pKerValue++ ;
		FILTER_OC2(vrinP_r0s1, vrsum01) ; pKerValue++ ;
		FILTER_OC2(vrinP_r0s2, vrsum01) ; pKerValue++ ;
		FILTER_OC2(vrinP_r1s0, vrsum01) ; pKerValue++ ;
		FILTER_OC2(vrinP_r1s1, vrsum01) ; pKerValue++ ;
		FILTER_OC2(vrinP_r1s2, vrsum01) ; pKerValue++ ;
		FILTER_OC2(vrinP_r2s0, vrsum01) ; pKerValue++ ;
		FILTER_OC2(vrinP_r2s1, vrsum01) ; pKerValue++ ;
		FILTER_OC2(vrinP_r2s2, vrsum01) ; pKerValue++ ;
#undef FILTER_OC2

	    } // inChannel

	    _vel_vstu_vssl(vrsum01, 4, pOut+outIndex, vl) ;
	    _vel_vstl_vssl(vrsum01, 4, pOut+outIndex+   oPixels, vl) ;

	    outIndex += vl ;
	  } // outPixels

	  k+=2 ;
	}
	if ( ((outChannelGroup >> 2) & 0x01) == 1 ) {
	  int64_t outIndex = outGroupOffset + (n * outChannel + k  ) * oPixels ;

	  const float bias0 = (pBias == NULL) ? 0.0 : pBias[g * outChannelGroup + k  ];
	  const float bias1 = (pBias == NULL) ? 0.0 : pBias[g * outChannelGroup + k+1];
	  const float bias2 = (pBias == NULL) ? 0.0 : pBias[g * outChannelGroup + k+2];
	  const float bias3 = (pBias == NULL) ? 0.0 : pBias[g * outChannelGroup + k+3];

	  const uint64_t bias01 = _vel_pack_f32p(&bias0, &bias1) ;
	  const uint64_t bias23 = _vel_pack_f32p(&bias2, &bias3) ;

	  for (int64_t op = 0; op < oPixels; op+=VLEN) {
	    const int64_t vl = oPixels - op < VLEN ? oPixels - op : VLEN ;

	    __vr vrsum01 = _vel_pvbrd_vsl(bias01, vl) ;
	    __vr vrsum23 = _vel_pvbrd_vsl(bias23, vl) ;

	    __vr vrseq = _vel_vseq_vl(vl) ;			// xy
	    __vr vridx = _vel_vaddsl_vsvl(op, vrseq, vl) ;	// op + xy

	    __vr vry   = _vel_vdivsl_vvsl(vridx, outWidth, vl) ;
	    __vr vrx   = _vel_vsubsl_vvvl(vridx, _vel_vmulul_vsvl(outWidth,vry, vl), vl) ;

	    __vr vrh_r0 = _vel_vaddsl_vsvl( -padHeight, vry, vl) ;
	    __vr vrh_r2 = _vel_vaddsl_vsvl(2-padHeight, vry, vl) ;

	    __vr vrw_s0 = _vel_vaddsl_vsvl( -padWidth, vrx, vl) ;
	    __vr vrw_s2 = _vel_vaddsl_vsvl(2-padWidth, vrx, vl) ;

	    __vm256 vm01_r0 =  _vel_vfmklge_mvl(vrh_r0, vl) ;
	    __vm256 vm01_r2 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(inHeight,vrh_r2, vl), vl) ;

	    __vm256 vm23_s0  =  _vel_vfmklge_mvl(vrw_s0, vl) ;
	    __vm256 vm23_s2  =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(inWidth,vrw_s2, vl), vl) ;


	    __vm256 vmall_r0s0 = _vel_andm_mmm(vm01_r0,vm23_s0) ;
	    __vm256 vmall_r0s1 = vm01_r0 ;
	    __vm256 vmall_r0s2 = _vel_andm_mmm(vm01_r0, vm23_s2) ;

	    __vm256 vmall_r1s0 = vm23_s0 ;
	    __vm256 vmall_r1s2 = vm23_s2 ;

	    __vm256 vmall_r2s0 = _vel_andm_mmm(vm01_r2,vm23_s0) ;
	    __vm256 vmall_r2s1 = vm01_r2 ;
	    __vm256 vmall_r2s2 = _vel_andm_mmm(vm01_r2, vm23_s2) ;

	    for (int64_t c = 0; c < inChannelGroup; c++) {
	      const float *pKerValue = pKernel + kernGroupOffset + ((k * inChannelGroup + c) * kernHeight) * kernWidth ;

	      const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;

	      /* memory access errors mihgt be caused */
	      __vr vrin_r0s0 = _vel_vldu_vssl(4,&pInChannel[op-inWidth-1], vl) ;
	      __vr vrin_r0s1 = _vel_vldu_vssl(4,&pInChannel[op-inWidth  ], vl) ;
	      __vr vrin_r0s2 = _vel_vldu_vssl(4,&pInChannel[op-inWidth+1], vl) ;
	      __vr vrin_r1s0 = _vel_vldu_vssl(4,&pInChannel[op+       -1], vl) ;
	      __vr vrin_r1s1 = _vel_vldu_vssl(4,&pInChannel[op          ], vl) ;
	      __vr vrin_r1s2 = _vel_vldu_vssl(4,&pInChannel[op+       +1], vl) ;
	      __vr vrin_r2s0 = _vel_vldu_vssl(4,&pInChannel[op+inWidth-1], vl) ;
	      __vr vrin_r2s1 = _vel_vldu_vssl(4,&pInChannel[op+inWidth  ], vl) ;
	      __vr vrin_r2s2 = _vel_vldu_vssl(4,&pInChannel[op+inWidth+1], vl) ;

	      __vr vrzerof = _vel_vbrds_vsl(0.0f, vl) ;

	      vrin_r0s0 = _vel_vmrg_vvvml(vrzerof, vrin_r0s0, vmall_r0s0, vl) ;
	      vrin_r0s1 = _vel_vmrg_vvvml(vrzerof, vrin_r0s1, vmall_r0s1, vl) ;
	      vrin_r0s2 = _vel_vmrg_vvvml(vrzerof, vrin_r0s2, vmall_r0s2, vl) ;
	      __vr vrinP_r0s0 = _vel_vshf_vvvsl(vrin_r0s0, vrin_r0s0, VE_VSHUFFLE_YUZU, vl) ;
	      __vr vrinP_r0s1 = _vel_vshf_vvvsl(vrin_r0s1, vrin_r0s1, VE_VSHUFFLE_YUZU, vl) ;
	      __vr vrinP_r0s2 = _vel_vshf_vvvsl(vrin_r0s2, vrin_r0s2, VE_VSHUFFLE_YUZU, vl) ;

	      vrin_r1s0 = _vel_vmrg_vvvml(vrzerof, vrin_r1s0, vmall_r1s0, vl) ;
	      vrin_r1s2 = _vel_vmrg_vvvml(vrzerof, vrin_r1s2, vmall_r1s2, vl) ;
	      __vr vrinP_r1s0 = _vel_vshf_vvvsl(vrin_r1s0, vrin_r1s0, VE_VSHUFFLE_YUZU, vl) ;
	      __vr vrinP_r1s1 = _vel_vshf_vvvsl(vrin_r1s1, vrin_r1s1, VE_VSHUFFLE_YUZU, vl) ;
	      __vr vrinP_r1s2 = _vel_vshf_vvvsl(vrin_r1s2, vrin_r1s2, VE_VSHUFFLE_YUZU, vl) ;

	      vrin_r2s0 = _vel_vmrg_vvvml(vrzerof, vrin_r2s0, vmall_r2s0, vl) ;
	      vrin_r2s1 = _vel_vmrg_vvvml(vrzerof, vrin_r2s1, vmall_r2s1, vl) ;
	      vrin_r2s2 = _vel_vmrg_vvvml(vrzerof, vrin_r2s2, vmall_r2s2, vl) ;
	      __vr vrinP_r2s0 = _vel_vshf_vvvsl(vrin_r2s0, vrin_r2s0, VE_VSHUFFLE_YUZU, vl) ;
	      __vr vrinP_r2s1 = _vel_vshf_vvvsl(vrin_r2s1, vrin_r2s1, VE_VSHUFFLE_YUZU, vl) ;
	      __vr vrinP_r2s2 = _vel_vshf_vvvsl(vrin_r2s2, vrin_r2s2, VE_VSHUFFLE_YUZU, vl) ;

#define FILTER_OC4(VRIN,VRSUM0,VRSUM1)									\
{													\
  const uint64_t kerValue0 = _vel_pack_f32p(pKerValue,							\
					   pKerValue +     inChannelGroup * kernHeight * kernWidth) ;	\
  const uint64_t kerValue1 = _vel_pack_f32p(pKerValue + 2 * inChannelGroup * kernHeight * kernWidth,	\
					   pKerValue + 3 * inChannelGroup * kernHeight * kernWidth) ;	\
  VRSUM0 = _vel_pvfmad_vvsvl(VRSUM0, kerValue0, VRIN, vl) ;								\
  VRSUM1 = _vel_pvfmad_vvsvl(VRSUM1, kerValue1, VRIN, vl) ;								\
}
		FILTER_OC4(vrinP_r0s0, vrsum01,vrsum23) ; pKerValue++ ;
		FILTER_OC4(vrinP_r0s1, vrsum01,vrsum23) ; pKerValue++ ;
		FILTER_OC4(vrinP_r0s2, vrsum01,vrsum23) ; pKerValue++ ;
		FILTER_OC4(vrinP_r1s0, vrsum01,vrsum23) ; pKerValue++ ;
		FILTER_OC4(vrinP_r1s1, vrsum01,vrsum23) ; pKerValue++ ;
		FILTER_OC4(vrinP_r1s2, vrsum01,vrsum23) ; pKerValue++ ;
		FILTER_OC4(vrinP_r2s0, vrsum01,vrsum23) ; pKerValue++ ;
		FILTER_OC4(vrinP_r2s1, vrsum01,vrsum23) ; pKerValue++ ;
		FILTER_OC4(vrinP_r2s2, vrsum01,vrsum23) ; pKerValue++ ;
#undef FILTER_OC4

	    } // inChannel

	    _vel_vstu_vssl(vrsum01, 4, pOut+outIndex, vl) ;
	    _vel_vstl_vssl(vrsum01, 4, pOut+outIndex+   oPixels, vl) ;
	    _vel_vstu_vssl(vrsum23, 4, pOut+outIndex+ 2*oPixels, vl) ;
	    _vel_vstl_vssl(vrsum23, 4, pOut+outIndex+ 3*oPixels, vl) ;

	    outIndex += vl ;
	  } // outPixels

	  k+=4 ;
	}
	for ( ; k < outChannelGroup; k+=8) {
	  int64_t outIndex = outGroupOffset + (n * outChannel + k  ) * oPixels ;

	  const float bias0 = (pBias == NULL) ? 0.0 : pBias[g * outChannelGroup + k  ];
	  const float bias1 = (pBias == NULL) ? 0.0 : pBias[g * outChannelGroup + k+1];
	  const float bias2 = (pBias == NULL) ? 0.0 : pBias[g * outChannelGroup + k+2];
	  const float bias3 = (pBias == NULL) ? 0.0 : pBias[g * outChannelGroup + k+3];
	  const float bias4 = (pBias == NULL) ? 0.0 : pBias[g * outChannelGroup + k+4];
	  const float bias5 = (pBias == NULL) ? 0.0 : pBias[g * outChannelGroup + k+5];
	  const float bias6 = (pBias == NULL) ? 0.0 : pBias[g * outChannelGroup + k+6];
	  const float bias7 = (pBias == NULL) ? 0.0 : pBias[g * outChannelGroup + k+7];

	  const uint64_t bias01 = _vel_pack_f32p(&bias0, &bias1) ;
	  const uint64_t bias23 = _vel_pack_f32p(&bias2, &bias3) ;
	  const uint64_t bias45 = _vel_pack_f32p(&bias4, &bias5) ;
	  const uint64_t bias67 = _vel_pack_f32p(&bias6, &bias7) ;

	  for (int64_t op = 0; op < oPixels; op+=VLEN) {
	    const int64_t vl = oPixels - op < VLEN ? oPixels - op : VLEN ;

	    __vr vrsum01 = _vel_pvbrd_vsl(bias01, vl) ;
	    __vr vrsum23 = _vel_pvbrd_vsl(bias23, vl) ;
	    __vr vrsum45 = _vel_pvbrd_vsl(bias45, vl) ;
	    __vr vrsum67 = _vel_pvbrd_vsl(bias67, vl) ;

	    __vr vrseq = _vel_vseq_vl(vl) ;			// xy
	    __vr vridx = _vel_vaddsl_vsvl(op, vrseq, vl) ;	// op + xy

	    __vr vry   = _vel_vdivsl_vvsl(vridx, outWidth, vl) ;
	    __vr vrx   = _vel_vsubsl_vvvl(vridx, _vel_vmulul_vsvl(outWidth,vry, vl), vl) ;

	    __vr vrh_r0 = _vel_vaddsl_vsvl( -padHeight, vry, vl) ;
	    __vr vrh_r2 = _vel_vaddsl_vsvl(2-padHeight, vry, vl) ;

	    __vr vrw_s0 = _vel_vaddsl_vsvl( -padWidth, vrx, vl) ;
	    __vr vrw_s2 = _vel_vaddsl_vsvl(2-padWidth, vrx, vl) ;

	    __vm256 vm01_r0 =  _vel_vfmklge_mvl(vrh_r0, vl) ;
	    __vm256 vm01_r2 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(inHeight,vrh_r2, vl), vl) ;

	    __vm256 vm23_s0  =  _vel_vfmklge_mvl(vrw_s0, vl) ;
	    __vm256 vm23_s2  =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(inWidth,vrw_s2, vl), vl) ;


	    __vm256 vmall_r0s0 = _vel_andm_mmm(vm01_r0,vm23_s0) ;
	    __vm256 vmall_r0s1 = vm01_r0 ;
	    __vm256 vmall_r0s2 = _vel_andm_mmm(vm01_r0, vm23_s2) ;

	    __vm256 vmall_r1s0 = vm23_s0 ;
	    __vm256 vmall_r1s2 = vm23_s2 ;

	    __vm256 vmall_r2s0 = _vel_andm_mmm(vm01_r2,vm23_s0) ;
	    __vm256 vmall_r2s1 = vm01_r2 ;
	    __vm256 vmall_r2s2 = _vel_andm_mmm(vm01_r2, vm23_s2) ;

	    for (int64_t c = 0; c < inChannelGroup; c++) {
	      const float *pKerValue = pKernel + kernGroupOffset + ((k * inChannelGroup + c) * kernHeight) * kernWidth ;

	      const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;

	      /* memory access errors mihgt be caused */
	      __vr vrin_r0s0 = _vel_vldu_vssl(4,&pInChannel[op-inWidth-1], vl) ;
	      __vr vrin_r0s1 = _vel_vldu_vssl(4,&pInChannel[op-inWidth  ], vl) ;
	      __vr vrin_r0s2 = _vel_vldu_vssl(4,&pInChannel[op-inWidth+1], vl) ;
	      __vr vrin_r1s0 = _vel_vldu_vssl(4,&pInChannel[op+       -1], vl) ;
	      __vr vrin_r1s1 = _vel_vldu_vssl(4,&pInChannel[op          ], vl) ;
	      __vr vrin_r1s2 = _vel_vldu_vssl(4,&pInChannel[op+       +1], vl) ;
	      __vr vrin_r2s0 = _vel_vldu_vssl(4,&pInChannel[op+inWidth-1], vl) ;
	      __vr vrin_r2s1 = _vel_vldu_vssl(4,&pInChannel[op+inWidth  ], vl) ;
	      __vr vrin_r2s2 = _vel_vldu_vssl(4,&pInChannel[op+inWidth+1], vl) ;

	      __vr vrzerof = _vel_vbrds_vsl(0.0f, vl) ;

	      vrin_r0s0 = _vel_vmrg_vvvml(vrzerof, vrin_r0s0, vmall_r0s0, vl) ;
	      vrin_r0s1 = _vel_vmrg_vvvml(vrzerof, vrin_r0s1, vmall_r0s1, vl) ;
	      vrin_r0s2 = _vel_vmrg_vvvml(vrzerof, vrin_r0s2, vmall_r0s2, vl) ;
	      __vr vrinP_r0s0 = _vel_vshf_vvvsl(vrin_r0s0, vrin_r0s0, VE_VSHUFFLE_YUZU, vl) ;
	      __vr vrinP_r0s1 = _vel_vshf_vvvsl(vrin_r0s1, vrin_r0s1, VE_VSHUFFLE_YUZU, vl) ;
	      __vr vrinP_r0s2 = _vel_vshf_vvvsl(vrin_r0s2, vrin_r0s2, VE_VSHUFFLE_YUZU, vl) ;

	      vrin_r1s0 = _vel_vmrg_vvvml(vrzerof, vrin_r1s0, vmall_r1s0, vl) ;
	      vrin_r1s2 = _vel_vmrg_vvvml(vrzerof, vrin_r1s2, vmall_r1s2, vl) ;
	      __vr vrinP_r1s0 = _vel_vshf_vvvsl(vrin_r1s0, vrin_r1s0, VE_VSHUFFLE_YUZU, vl) ;
	      __vr vrinP_r1s1 = _vel_vshf_vvvsl(vrin_r1s1, vrin_r1s1, VE_VSHUFFLE_YUZU, vl) ;
	      __vr vrinP_r1s2 = _vel_vshf_vvvsl(vrin_r1s2, vrin_r1s2, VE_VSHUFFLE_YUZU, vl) ;

	      vrin_r2s0 = _vel_vmrg_vvvml(vrzerof, vrin_r2s0, vmall_r2s0, vl) ;
	      vrin_r2s1 = _vel_vmrg_vvvml(vrzerof, vrin_r2s1, vmall_r2s1, vl) ;
	      vrin_r2s2 = _vel_vmrg_vvvml(vrzerof, vrin_r2s2, vmall_r2s2, vl) ;
	      __vr vrinP_r2s0 = _vel_vshf_vvvsl(vrin_r2s0, vrin_r2s0, VE_VSHUFFLE_YUZU, vl) ;
	      __vr vrinP_r2s1 = _vel_vshf_vvvsl(vrin_r2s1, vrin_r2s1, VE_VSHUFFLE_YUZU, vl) ;
	      __vr vrinP_r2s2 = _vel_vshf_vvvsl(vrin_r2s2, vrin_r2s2, VE_VSHUFFLE_YUZU, vl) ;

#define FILTER_OC4(VRIN,VRSUM0,VRSUM1,N)									\
{														\
  const uint64_t kerValue0 = _vel_pack_f32p(pKerValue +  (N)    * inChannelGroup * kernHeight * kernWidth,	\
					   pKerValue + ((N)+1) * inChannelGroup * kernHeight * kernWidth) ;	\
  const uint64_t kerValue1 = _vel_pack_f32p(pKerValue + ((N)+2) * inChannelGroup * kernHeight * kernWidth,	\
					   pKerValue + ((N)+3) * inChannelGroup * kernHeight * kernWidth) ;	\
  VRSUM0 = _vel_pvfmad_vvsvl(VRSUM0, kerValue0, VRIN, vl) ;								\
  VRSUM1 = _vel_pvfmad_vvsvl(VRSUM1, kerValue1, VRIN, vl) ;								\
}
#define FILTER_OC8(VRIN)		\
{					\
  FILTER_OC4(VRIN,vrsum01,vrsum23,0) ;	\
  FILTER_OC4(VRIN,vrsum45,vrsum67,4) ;	\
}
		FILTER_OC8(vrinP_r0s0) ; pKerValue++ ;
		FILTER_OC8(vrinP_r0s1) ; pKerValue++ ;
		FILTER_OC8(vrinP_r0s2) ; pKerValue++ ;
		FILTER_OC8(vrinP_r1s0) ; pKerValue++ ;
		FILTER_OC8(vrinP_r1s1) ; pKerValue++ ;
		FILTER_OC8(vrinP_r1s2) ; pKerValue++ ;
		FILTER_OC8(vrinP_r2s0) ; pKerValue++ ;
		FILTER_OC8(vrinP_r2s1) ; pKerValue++ ;
		FILTER_OC8(vrinP_r2s2) ; pKerValue++ ;
#undef FILTER_OC4
#undef FILTER_OC8
	    } // inChannel

	    _vel_vstu_vssl(vrsum01, 4, pOut+outIndex, vl) ;
	    _vel_vstl_vssl(vrsum01, 4, pOut+outIndex+   oPixels, vl) ;
	    _vel_vstu_vssl(vrsum23, 4, pOut+outIndex+ 2*oPixels, vl) ;
	    _vel_vstl_vssl(vrsum23, 4, pOut+outIndex+ 3*oPixels, vl) ;
	    _vel_vstu_vssl(vrsum45, 4, pOut+outIndex+ 4*oPixels, vl) ;
	    _vel_vstl_vssl(vrsum45, 4, pOut+outIndex+ 5*oPixels, vl) ;
	    _vel_vstu_vssl(vrsum67, 4, pOut+outIndex+ 6*oPixels, vl) ;
	    _vel_vstl_vssl(vrsum67, 4, pOut+outIndex+ 7*oPixels, vl) ;

	    outIndex += vl ;
	  } // outPixels
	} // outChannel
      } // group
    } // batch
  }

  return VEDNN_SUCCESS;
}

