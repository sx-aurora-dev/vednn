#include <stdint.h>
#include <stdlib.h>

#include "vednn.h"

#include "veintrin.h"
#define VLEN	(256)


vednnError_t
vednnConvolutionForward_direct_dil1_str1_padsame_ker3_c1_owU128(
    const vednnTensorParam_t * restrict 	pParamIn,
    const void * restrict 			pDataIn,
    const vednnFilterParam_t * restrict 	pParamKernel,
    const void * restrict 			pDataKernel,
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
  const int64_t outWidth   = pParamOut->width;			/* must be <= 128 */
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

//  const int64_t inChannelGroup  = inChannel  / group;   // equal to pDataKernel->inChannel ( must be 1 )
  const int64_t outChannelGroup = outChannel / group;   // equal to pDataKernel->outChannel

  const float * restrict pIn     = pDataIn;
  const float * restrict pKernel = pDataKernel;
  float * restrict const pOut    = pDataOut;

  const int oPixels= outHeight*outWidth ;

  {
    const int64_t nY = VLEN / outWidth ;

    _ve_lvl(VLEN) ;

    __vr vrseq = _ve_vseq_v() ;			// xy

    __vr vry   = _ve_vdivsl_vvs(vrseq, outWidth) ;
    __vr vrx   = _ve_vsubsl_vvv(vrseq, _ve_vmulul_vsv(outWidth,vry)) ;

    __vr vrw_s0 = _ve_vaddsl_vsv( -padWidth, vrx) ;
    __vr vrw_s2 = _ve_vaddsl_vsv(2-padWidth, vrx) ;

    __vm256 vm23_s0  = _ve_vfmkl_mcv(VECC_GE, vrw_s0) ;
    __vm256 vm23_s2  = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(inWidth,vrw_s2)) ;

    __vr vrzerof = _ve_vbrdu_vs_f32(0.0f) ;

    for (int64_t n = 0; n < batch; n++) {
      for (int64_t g = 0; g < group; g++) {

	const int64_t inGroupOffset   = g * inHeight * inWidth;
	const int64_t outGroupOffset  = g * outChannelGroup * outHeight * outWidth;
	const int64_t kernGroupOffset = g * outChannelGroup * kernHeight * kernWidth;

	int k = 0 ;
	if ( (outChannelGroup & 0x01) == 1 ) {

	  _ve_lvl(VLEN) ;
	  const __vr vrzero = _ve_vbrdu_vs_f32(0.0f) ;

	  for (int64_t y=0; y<outHeight; y+=nY)
	  {
	    const int64_t vl = outWidth * (outHeight - y < nY ? outHeight - y : nY) ;
	    const int64_t op = y * outWidth ;

	    const float *pKerValue  = pKernel + kernGroupOffset + (k * kernHeight    ) * kernWidth ;
	    const float *pInChannel = pIn + inGroupOffset + ((n * inChannel) * inHeight * inWidth ) ;

	    _ve_lvl(vl) ;

	    _ve_pfchv_ss(4,&pInChannel[op-inWidth-1]) ; // prefetch

	    __vr vrh_r0 = _ve_vaddsl_vsv(y  -padHeight, vry) ;
	    __vr vrh_r2 = _ve_vaddsl_vsv(y+2-padHeight, vry) ;

	    __vm256 vm01_r0 = _ve_vfmkl_mcv(VECC_GE, vrh_r0) ;
	    __vm256 vm01_r2 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(inHeight,vrh_r2)) ;

	    __vm256 vmall_r0s0 = _ve_andm_mmm(vm01_r0,vm23_s0) ;
	    __vm256 vmall_r0s1 = vm01_r0 ;
	    __vm256 vmall_r0s2 = _ve_andm_mmm(vm01_r0, vm23_s2) ;

	    __vm256 vmall_r1s0 = vm23_s0 ;
	    __vm256 vmall_r1s2 = vm23_s2 ;

	    __vm256 vmall_r2s0 = _ve_andm_mmm(vm01_r2,vm23_s0) ;
	    __vm256 vmall_r2s1 = vm01_r2 ;
	    __vm256 vmall_r2s2 = _ve_andm_mmm(vm01_r2, vm23_s2) ;

	    int64_t outIndex = outGroupOffset + (n * outChannel + k) * oPixels + op ;


	      /* memory access errors mihgt be caused */
	    __vr vrin_r0s0 = _ve_vldu_vss(4,&pInChannel[op-inWidth-1]) ;
	    __vr vrin_r0s1 = _ve_vldu_vss(4,&pInChannel[op-inWidth  ]) ;
	    __vr vrin_r0s2 = _ve_vldu_vss(4,&pInChannel[op-inWidth+1]) ;
	    __vr vrin_r1s0 = _ve_vldu_vss(4,&pInChannel[op+       -1]) ;
	    __vr vrin_r1s1 = _ve_vldu_vss(4,&pInChannel[op          ]) ;
	    __vr vrin_r1s2 = _ve_vldu_vss(4,&pInChannel[op+       +1]) ;
	    __vr vrin_r2s0 = _ve_vldu_vss(4,&pInChannel[op+inWidth-1]) ;
	    __vr vrin_r2s1 = _ve_vldu_vss(4,&pInChannel[op+inWidth  ]) ;
	    __vr vrin_r2s2 = _ve_vldu_vss(4,&pInChannel[op+inWidth+1]) ;


	    vrin_r0s0 = _ve_vmrg_vvvm(vrzerof, vrin_r0s0, vmall_r0s0) ;
	    vrin_r0s1 = _ve_vmrg_vvvm(vrzerof, vrin_r0s1, vmall_r0s1) ;
	    vrin_r0s2 = _ve_vmrg_vvvm(vrzerof, vrin_r0s2, vmall_r0s2) ;

	    vrin_r1s0 = _ve_vmrg_vvvm(vrzerof, vrin_r1s0, vmall_r1s0) ;
	    vrin_r1s2 = _ve_vmrg_vvvm(vrzerof, vrin_r1s2, vmall_r1s2) ;

	    vrin_r2s0 = _ve_vmrg_vvvm(vrzerof, vrin_r2s0, vmall_r2s0) ;
	    vrin_r2s1 = _ve_vmrg_vvvm(vrzerof, vrin_r2s1, vmall_r2s1) ;
	    vrin_r2s2 = _ve_vmrg_vvvm(vrzerof, vrin_r2s2, vmall_r2s2) ;


	    __vr vrsum = vrzero ;
	    vrsum = _ve_vfmads_vvsv(vrsum, pKerValue[0], vrin_r0s0) ;
	    vrsum = _ve_vfmads_vvsv(vrsum, pKerValue[1], vrin_r0s1) ;
	    vrsum = _ve_vfmads_vvsv(vrsum, pKerValue[2], vrin_r0s2) ;
	    vrsum = _ve_vfmads_vvsv(vrsum, pKerValue[3], vrin_r1s0) ;
	    vrsum = _ve_vfmads_vvsv(vrsum, pKerValue[4], vrin_r1s1) ;
	    vrsum = _ve_vfmads_vvsv(vrsum, pKerValue[5], vrin_r1s2) ;
	    vrsum = _ve_vfmads_vvsv(vrsum, pKerValue[6], vrin_r2s0) ;
	    vrsum = _ve_vfmads_vvsv(vrsum, pKerValue[7], vrin_r2s1) ;
	    vrsum = _ve_vfmads_vvsv(vrsum, pKerValue[8], vrin_r2s2) ;
	    _ve_vstu_vss(vrsum, 4, pOut+outIndex) ;

	  } // outPixels
	  k++ ;
	}
	if ( ((outChannelGroup >> 1) & 0x01) == 1 ) {

	  _ve_lvl(VLEN) ;
	  const __vr vrzero = _ve_pvbrd_vs_i64(0UL) ;

	  uint64_t kerValue[9] ;
	  const float *pKerValue  = pKernel + kernGroupOffset + (k * kernHeight    ) * kernWidth ;
	  kerValue[0] = _ve_pack_f32p(pKerValue, pKerValue+9); pKerValue++ ;
	  kerValue[1] = _ve_pack_f32p(pKerValue, pKerValue+9); pKerValue++ ;
	  kerValue[2] = _ve_pack_f32p(pKerValue, pKerValue+9); pKerValue++ ;
	  kerValue[3] = _ve_pack_f32p(pKerValue, pKerValue+9); pKerValue++ ;
	  kerValue[4] = _ve_pack_f32p(pKerValue, pKerValue+9); pKerValue++ ;
	  kerValue[5] = _ve_pack_f32p(pKerValue, pKerValue+9); pKerValue++ ;
	  kerValue[6] = _ve_pack_f32p(pKerValue, pKerValue+9); pKerValue++ ;
	  kerValue[7] = _ve_pack_f32p(pKerValue, pKerValue+9); pKerValue++ ;
	  kerValue[8] = _ve_pack_f32p(pKerValue, pKerValue+9); pKerValue++ ;


	  for (int64_t y=0; y<outHeight; y+=nY)
	  {
	    const int64_t vl = outWidth * (outHeight - y < nY ? outHeight - y : nY) ;
	    const int64_t op = y * outWidth ;

	    _ve_lvl(vl) ;

	    __vr vrh_r0 = _ve_vaddsl_vsv(y  -padHeight, vry) ;
	    __vr vrh_r2 = _ve_vaddsl_vsv(y+2-padHeight, vry) ;

	    __vm256 vm01_r0 = _ve_vfmkl_mcv(VECC_GE, vrh_r0) ;
	    __vm256 vm01_r2 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(inHeight,vrh_r2)) ;

	    __vm256 vmall_r0s0 = _ve_andm_mmm(vm01_r0,vm23_s0) ;
	    __vm256 vmall_r0s1 = vm01_r0 ;
	    __vm256 vmall_r0s2 = _ve_andm_mmm(vm01_r0, vm23_s2) ;

	    __vm256 vmall_r1s0 = vm23_s0 ;
	    __vm256 vmall_r1s2 = vm23_s2 ;

	    __vm256 vmall_r2s0 = _ve_andm_mmm(vm01_r2,vm23_s0) ;
	    __vm256 vmall_r2s1 = vm01_r2 ;
	    __vm256 vmall_r2s2 = _ve_andm_mmm(vm01_r2, vm23_s2) ;


	    int64_t outIndex = outGroupOffset + (n * outChannel + k  ) * oPixels + op ;

	    const float *pInChannel = pIn + inGroupOffset + ((n * inChannel) * inHeight * inWidth ) ;

	      /* memory access errors mihgt be caused */
	    __vr vrin_r0s0 = _ve_vldu_vss(4,&pInChannel[op-inWidth-1]) ;
	    __vr vrin_r0s1 = _ve_vldu_vss(4,&pInChannel[op-inWidth  ]) ;
	    __vr vrin_r0s2 = _ve_vldu_vss(4,&pInChannel[op-inWidth+1]) ;
	    __vr vrin_r1s0 = _ve_vldu_vss(4,&pInChannel[op+       -1]) ;
	    __vr vrin_r1s1 = _ve_vldu_vss(4,&pInChannel[op          ]) ;
	    __vr vrin_r1s2 = _ve_vldu_vss(4,&pInChannel[op+       +1]) ;
	    __vr vrin_r2s0 = _ve_vldu_vss(4,&pInChannel[op+inWidth-1]) ;
	    __vr vrin_r2s1 = _ve_vldu_vss(4,&pInChannel[op+inWidth  ]) ;
	    __vr vrin_r2s2 = _ve_vldu_vss(4,&pInChannel[op+inWidth+1]) ;

	    vrin_r0s0 = _ve_vmrg_vvvm(vrzerof, vrin_r0s0, vmall_r0s0) ;
	    vrin_r0s1 = _ve_vmrg_vvvm(vrzerof, vrin_r0s1, vmall_r0s1) ;
	    vrin_r0s2 = _ve_vmrg_vvvm(vrzerof, vrin_r0s2, vmall_r0s2) ;
	    __vr vrinP_r0s0 = _ve_vshf_vvvs(vrin_r0s0, vrin_r0s0, VE_VSHUFFLE_YUZU) ;
	    __vr vrinP_r0s1 = _ve_vshf_vvvs(vrin_r0s1, vrin_r0s1, VE_VSHUFFLE_YUZU) ;
	    __vr vrinP_r0s2 = _ve_vshf_vvvs(vrin_r0s2, vrin_r0s2, VE_VSHUFFLE_YUZU) ;

	    vrin_r1s0 = _ve_vmrg_vvvm(vrzerof, vrin_r1s0, vmall_r1s0) ;
	    vrin_r1s2 = _ve_vmrg_vvvm(vrzerof, vrin_r1s2, vmall_r1s2) ;
	    __vr vrinP_r1s0 = _ve_vshf_vvvs(vrin_r1s0, vrin_r1s0, VE_VSHUFFLE_YUZU) ;
	    __vr vrinP_r1s1 = _ve_vshf_vvvs(vrin_r1s1, vrin_r1s1, VE_VSHUFFLE_YUZU) ;
	    __vr vrinP_r1s2 = _ve_vshf_vvvs(vrin_r1s2, vrin_r1s2, VE_VSHUFFLE_YUZU) ;

	    vrin_r2s0 = _ve_vmrg_vvvm(vrzerof, vrin_r2s0, vmall_r2s0) ;
	    vrin_r2s1 = _ve_vmrg_vvvm(vrzerof, vrin_r2s1, vmall_r2s1) ;
	    vrin_r2s2 = _ve_vmrg_vvvm(vrzerof, vrin_r2s2, vmall_r2s2) ;
	    __vr vrinP_r2s0 = _ve_vshf_vvvs(vrin_r2s0, vrin_r2s0, VE_VSHUFFLE_YUZU) ;
	    __vr vrinP_r2s1 = _ve_vshf_vvvs(vrin_r2s1, vrin_r2s1, VE_VSHUFFLE_YUZU) ;
	    __vr vrinP_r2s2 = _ve_vshf_vvvs(vrin_r2s2, vrin_r2s2, VE_VSHUFFLE_YUZU) ;

	    __vr vrsum = vrzero ;
	    vrsum = _ve_pvfmad_vvsv(vrsum, kerValue[0], vrinP_r0s0) ;
	    vrsum = _ve_pvfmad_vvsv(vrsum, kerValue[1], vrinP_r0s1) ;
	    vrsum = _ve_pvfmad_vvsv(vrsum, kerValue[2], vrinP_r0s2) ;
	    vrsum = _ve_pvfmad_vvsv(vrsum, kerValue[3], vrinP_r1s0) ;
	    vrsum = _ve_pvfmad_vvsv(vrsum, kerValue[4], vrinP_r1s1) ;
	    vrsum = _ve_pvfmad_vvsv(vrsum, kerValue[5], vrinP_r1s2) ;
	    vrsum = _ve_pvfmad_vvsv(vrsum, kerValue[6], vrinP_r2s0) ;
	    vrsum = _ve_pvfmad_vvsv(vrsum, kerValue[7], vrinP_r2s1) ;
	    vrsum = _ve_pvfmad_vvsv(vrsum, kerValue[8], vrinP_r2s2) ;
	    _ve_vstu_vss(vrsum, 4, pOut+outIndex) ;
	    _ve_vstl_vss(vrsum, 4, pOut+outIndex+ oPixels) ;

	  } // outPixels

	  k+=2 ;
	}
	if ( ((outChannelGroup >> 2) & 0x01) == 1 ) {

	  _ve_lvl(VLEN) ;
	  const __vr vrzero = _ve_pvbrd_vs_i64(0UL) ;

	  uint64_t kerValue[2*9] ;
	  {
	    const float *pKerValue  = pKernel + kernGroupOffset + (k * kernHeight    ) * kernWidth ;
	    _ve_lvl(4*9) ;
	    __vr vrker0 = _ve_vldu_vss(4, pKerValue    ) ;
	    __vr vrker1 = _ve_vldu_vss(4, pKerValue+2*9) ;
	    _ve_vst_vss(_ve_vshf_vvvs(vrker0, vrker1, VE_VSHUFFLE_YUZU) ,8, kerValue) ;
	  }

	  for (int64_t y=0; y<outHeight; y+=nY)
	  {
	    const int64_t vl = outWidth * (outHeight - y < nY ? outHeight - y : nY) ;
	    const int64_t op = y * outWidth ;

	    _ve_lvl(vl) ;

	    __vr vrh_r0 = _ve_vaddsl_vsv(y  -padHeight, vry) ;
	    __vr vrh_r2 = _ve_vaddsl_vsv(y+2-padHeight, vry) ;

	    __vm256 vm01_r0 = _ve_vfmkl_mcv(VECC_GE, vrh_r0) ;
	    __vm256 vm01_r2 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(inHeight,vrh_r2)) ;

	    __vm256 vmall_r0s0 = _ve_andm_mmm(vm01_r0,vm23_s0) ;
	    __vm256 vmall_r0s1 = vm01_r0 ;
	    __vm256 vmall_r0s2 = _ve_andm_mmm(vm01_r0, vm23_s2) ;

	    __vm256 vmall_r1s0 = vm23_s0 ;
	    __vm256 vmall_r1s2 = vm23_s2 ;

	    __vm256 vmall_r2s0 = _ve_andm_mmm(vm01_r2,vm23_s0) ;
	    __vm256 vmall_r2s1 = vm01_r2 ;
	    __vm256 vmall_r2s2 = _ve_andm_mmm(vm01_r2, vm23_s2) ;

	    int64_t outIndex = outGroupOffset + (n * outChannel + k  ) * oPixels + op ;

	    const float *pInChannel = pIn + inGroupOffset + ((n * inChannel) * inHeight * inWidth ) ;

	      /* memory access errors mihgt be caused */
	    __vr vrin_r0s0 = _ve_vldu_vss(4,&pInChannel[op-inWidth-1]) ;
	    __vr vrin_r0s1 = _ve_vldu_vss(4,&pInChannel[op-inWidth  ]) ;
	    __vr vrin_r0s2 = _ve_vldu_vss(4,&pInChannel[op-inWidth+1]) ;
	    __vr vrin_r1s0 = _ve_vldu_vss(4,&pInChannel[op+       -1]) ;
	    __vr vrin_r1s1 = _ve_vldu_vss(4,&pInChannel[op          ]) ;
	    __vr vrin_r1s2 = _ve_vldu_vss(4,&pInChannel[op+       +1]) ;
	    __vr vrin_r2s0 = _ve_vldu_vss(4,&pInChannel[op+inWidth-1]) ;
	    __vr vrin_r2s1 = _ve_vldu_vss(4,&pInChannel[op+inWidth  ]) ;
	    __vr vrin_r2s2 = _ve_vldu_vss(4,&pInChannel[op+inWidth+1]) ;


	    vrin_r0s0 = _ve_vmrg_vvvm(vrzerof, vrin_r0s0, vmall_r0s0) ;
	    vrin_r0s1 = _ve_vmrg_vvvm(vrzerof, vrin_r0s1, vmall_r0s1) ;
	    vrin_r0s2 = _ve_vmrg_vvvm(vrzerof, vrin_r0s2, vmall_r0s2) ;
	    __vr vrinP_r0s0 = _ve_vshf_vvvs(vrin_r0s0, vrin_r0s0, VE_VSHUFFLE_YUZU) ;
	    __vr vrinP_r0s1 = _ve_vshf_vvvs(vrin_r0s1, vrin_r0s1, VE_VSHUFFLE_YUZU) ;
	    __vr vrinP_r0s2 = _ve_vshf_vvvs(vrin_r0s2, vrin_r0s2, VE_VSHUFFLE_YUZU) ;

	    vrin_r1s0 = _ve_vmrg_vvvm(vrzerof, vrin_r1s0, vmall_r1s0) ;
	    vrin_r1s2 = _ve_vmrg_vvvm(vrzerof, vrin_r1s2, vmall_r1s2) ;
	    __vr vrinP_r1s0 = _ve_vshf_vvvs(vrin_r1s0, vrin_r1s0, VE_VSHUFFLE_YUZU) ;
	    __vr vrinP_r1s1 = _ve_vshf_vvvs(vrin_r1s1, vrin_r1s1, VE_VSHUFFLE_YUZU) ;
	    __vr vrinP_r1s2 = _ve_vshf_vvvs(vrin_r1s2, vrin_r1s2, VE_VSHUFFLE_YUZU) ;

	    vrin_r2s0 = _ve_vmrg_vvvm(vrzerof, vrin_r2s0, vmall_r2s0) ;
	    vrin_r2s1 = _ve_vmrg_vvvm(vrzerof, vrin_r2s1, vmall_r2s1) ;
	    vrin_r2s2 = _ve_vmrg_vvvm(vrzerof, vrin_r2s2, vmall_r2s2) ;
	    __vr vrinP_r2s0 = _ve_vshf_vvvs(vrin_r2s0, vrin_r2s0, VE_VSHUFFLE_YUZU) ;
	    __vr vrinP_r2s1 = _ve_vshf_vvvs(vrin_r2s1, vrin_r2s1, VE_VSHUFFLE_YUZU) ;
	    __vr vrinP_r2s2 = _ve_vshf_vvvs(vrin_r2s2, vrin_r2s2, VE_VSHUFFLE_YUZU) ;

#define FILTER_R3S3(VRBIAS, N)						\
{									\
  __vr vrsum = (VRBIAS) ;						\
  vrsum = _ve_pvfmad_vvsv(vrsum, kerValue[9*(N)+0], vrinP_r0s0) ;	\
  vrsum = _ve_pvfmad_vvsv(vrsum, kerValue[9*(N)+1], vrinP_r0s1) ;	\
  vrsum = _ve_pvfmad_vvsv(vrsum, kerValue[9*(N)+2], vrinP_r0s2) ;	\
  vrsum = _ve_pvfmad_vvsv(vrsum, kerValue[9*(N)+3], vrinP_r1s0) ;	\
  vrsum = _ve_pvfmad_vvsv(vrsum, kerValue[9*(N)+4], vrinP_r1s1) ;	\
  vrsum = _ve_pvfmad_vvsv(vrsum, kerValue[9*(N)+5], vrinP_r1s2) ;	\
  vrsum = _ve_pvfmad_vvsv(vrsum, kerValue[9*(N)+6], vrinP_r2s0) ;	\
  vrsum = _ve_pvfmad_vvsv(vrsum, kerValue[9*(N)+7], vrinP_r2s1) ;	\
  vrsum = _ve_pvfmad_vvsv(vrsum, kerValue[9*(N)+8], vrinP_r2s2) ;	\
  _ve_vstu_vss(vrsum, 4, pOut+outIndex+ ((N)  )*oPixels) ;		\
  _ve_vstl_vss(vrsum, 4, pOut+outIndex+ ((N)+2)*oPixels) ;		\
}
	    FILTER_R3S3(vrzero, 0) ;
	    FILTER_R3S3(vrzero, 1) ;
#undef FILTER_R3S3
	  } // outPixels

	  k+=4 ;
	}
	if ( ((outChannelGroup >> 3) & 0x01) == 1 ) {

	  _ve_lvl(VLEN) ;
	  const __vr vrzero = _ve_pvbrd_vs_i64(0UL) ;

	  uint64_t kerValue[4*9] ;
	  {
	    const float *pKerValue  = pKernel + kernGroupOffset + (k * kernHeight    ) * kernWidth ;
	    _ve_lvl(4*9) ;
	    __vr vrker0 = _ve_vldu_vss(4, pKerValue    ) ;
	    __vr vrker1 = _ve_vldu_vss(4, pKerValue+4*9) ;
	    _ve_vst_vss(_ve_vshf_vvvs(vrker0, vrker1, VE_VSHUFFLE_YUZU) ,8, kerValue) ;
	  }

	  for (int64_t y=0; y<outHeight; y+=nY)
	  {
	    const int64_t vl = outWidth * (outHeight - y < nY ? outHeight - y : nY) ;
	    const int64_t op = y * outWidth ;

	    _ve_lvl(vl) ;

	    __vr vrh_r0 = _ve_vaddsl_vsv(y  -padHeight, vry) ;
	    __vr vrh_r2 = _ve_vaddsl_vsv(y+2-padHeight, vry) ;

	    __vm256 vm01_r0 = _ve_vfmkl_mcv(VECC_GE, vrh_r0) ;
	    __vm256 vm01_r2 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(inHeight,vrh_r2)) ;

	    __vm256 vmall_r0s0 = _ve_andm_mmm(vm01_r0,vm23_s0) ;
	    __vm256 vmall_r0s1 = vm01_r0 ;
	    __vm256 vmall_r0s2 = _ve_andm_mmm(vm01_r0, vm23_s2) ;

	    __vm256 vmall_r1s0 = vm23_s0 ;
	    __vm256 vmall_r1s2 = vm23_s2 ;

	    __vm256 vmall_r2s0 = _ve_andm_mmm(vm01_r2,vm23_s0) ;
	    __vm256 vmall_r2s1 = vm01_r2 ;
	    __vm256 vmall_r2s2 = _ve_andm_mmm(vm01_r2, vm23_s2) ;

	    int64_t outIndex = outGroupOffset + (n * outChannel + k  ) * oPixels + op ;

	    const float *pInChannel = pIn + inGroupOffset + ((n * inChannel) * inHeight * inWidth ) ;

	      /* memory access errors mihgt be caused */
	    __vr vrin_r0s0 = _ve_vldu_vss(4,&pInChannel[op-inWidth-1]) ;
	    __vr vrin_r0s1 = _ve_vldu_vss(4,&pInChannel[op-inWidth  ]) ;
	    __vr vrin_r0s2 = _ve_vldu_vss(4,&pInChannel[op-inWidth+1]) ;
	    __vr vrin_r1s0 = _ve_vldu_vss(4,&pInChannel[op+       -1]) ;
	    __vr vrin_r1s1 = _ve_vldu_vss(4,&pInChannel[op          ]) ;
	    __vr vrin_r1s2 = _ve_vldu_vss(4,&pInChannel[op+       +1]) ;
	    __vr vrin_r2s0 = _ve_vldu_vss(4,&pInChannel[op+inWidth-1]) ;
	    __vr vrin_r2s1 = _ve_vldu_vss(4,&pInChannel[op+inWidth  ]) ;
	    __vr vrin_r2s2 = _ve_vldu_vss(4,&pInChannel[op+inWidth+1]) ;

	    vrin_r0s0 = _ve_vmrg_vvvm(vrzerof, vrin_r0s0, vmall_r0s0) ;
	    vrin_r0s1 = _ve_vmrg_vvvm(vrzerof, vrin_r0s1, vmall_r0s1) ;
	    vrin_r0s2 = _ve_vmrg_vvvm(vrzerof, vrin_r0s2, vmall_r0s2) ;
	    __vr vrinP_r0s0 = _ve_vshf_vvvs(vrin_r0s0, vrin_r0s0, VE_VSHUFFLE_YUZU) ;
	    __vr vrinP_r0s1 = _ve_vshf_vvvs(vrin_r0s1, vrin_r0s1, VE_VSHUFFLE_YUZU) ;
	    __vr vrinP_r0s2 = _ve_vshf_vvvs(vrin_r0s2, vrin_r0s2, VE_VSHUFFLE_YUZU) ;

	    vrin_r1s0 = _ve_vmrg_vvvm(vrzerof, vrin_r1s0, vmall_r1s0) ;
	    vrin_r1s2 = _ve_vmrg_vvvm(vrzerof, vrin_r1s2, vmall_r1s2) ;
	    __vr vrinP_r1s0 = _ve_vshf_vvvs(vrin_r1s0, vrin_r1s0, VE_VSHUFFLE_YUZU) ;
	    __vr vrinP_r1s1 = _ve_vshf_vvvs(vrin_r1s1, vrin_r1s1, VE_VSHUFFLE_YUZU) ;
	    __vr vrinP_r1s2 = _ve_vshf_vvvs(vrin_r1s2, vrin_r1s2, VE_VSHUFFLE_YUZU) ;

	    vrin_r2s0 = _ve_vmrg_vvvm(vrzerof, vrin_r2s0, vmall_r2s0) ;
	    vrin_r2s1 = _ve_vmrg_vvvm(vrzerof, vrin_r2s1, vmall_r2s1) ;
	    vrin_r2s2 = _ve_vmrg_vvvm(vrzerof, vrin_r2s2, vmall_r2s2) ;
	    __vr vrinP_r2s0 = _ve_vshf_vvvs(vrin_r2s0, vrin_r2s0, VE_VSHUFFLE_YUZU) ;
	    __vr vrinP_r2s1 = _ve_vshf_vvvs(vrin_r2s1, vrin_r2s1, VE_VSHUFFLE_YUZU) ;
	    __vr vrinP_r2s2 = _ve_vshf_vvvs(vrin_r2s2, vrin_r2s2, VE_VSHUFFLE_YUZU) ;

#define FILTER_R3S3(VRBIAS, N)						\
{									\
  __vr vrsum = (VRBIAS) ;						\
  vrsum = _ve_pvfmad_vvsv(vrsum, kerValue[9*(N)+0], vrinP_r0s0) ;	\
  vrsum = _ve_pvfmad_vvsv(vrsum, kerValue[9*(N)+1], vrinP_r0s1) ;	\
  vrsum = _ve_pvfmad_vvsv(vrsum, kerValue[9*(N)+2], vrinP_r0s2) ;	\
  vrsum = _ve_pvfmad_vvsv(vrsum, kerValue[9*(N)+3], vrinP_r1s0) ;	\
  vrsum = _ve_pvfmad_vvsv(vrsum, kerValue[9*(N)+4], vrinP_r1s1) ;	\
  vrsum = _ve_pvfmad_vvsv(vrsum, kerValue[9*(N)+5], vrinP_r1s2) ;	\
  vrsum = _ve_pvfmad_vvsv(vrsum, kerValue[9*(N)+6], vrinP_r2s0) ;	\
  vrsum = _ve_pvfmad_vvsv(vrsum, kerValue[9*(N)+7], vrinP_r2s1) ;	\
  vrsum = _ve_pvfmad_vvsv(vrsum, kerValue[9*(N)+8], vrinP_r2s2) ;	\
  _ve_vstu_vss(vrsum, 4, pOut+outIndex+ ((N)  )*oPixels) ;		\
  _ve_vstl_vss(vrsum, 4, pOut+outIndex+ ((N)+4)*oPixels) ;		\
}
	    FILTER_R3S3(vrzero, 0) ;
	    FILTER_R3S3(vrzero, 1) ;
	    FILTER_R3S3(vrzero, 2) ;
	    FILTER_R3S3(vrzero, 3) ;
#undef FILTER_R3S3
	  } // outPixels
	  k+=8 ;
	}
	for ( ; k < outChannelGroup; k+=16) {

	  _ve_lvl(VLEN) ;
	  const __vr vrzero = _ve_pvbrd_vs_i64(0UL) ;

	  uint64_t kerValue[8*9] ;
	  {
	    const float *pKerValue  = pKernel + kernGroupOffset + (k * kernHeight    ) * kernWidth ;
	    _ve_lvl(8*9) ;
	    __vr vrker0 = _ve_vldu_vss(4, pKerValue    ) ;
	    __vr vrker1 = _ve_vldu_vss(4, pKerValue+8*9) ;
	    _ve_vst_vss(_ve_vshf_vvvs(vrker0, vrker1, VE_VSHUFFLE_YUZU) ,8, kerValue) ;
	  }

	  for (int64_t y=0; y<outHeight; y+=nY)
	  {
	    const int64_t vl = outWidth * (outHeight - y < nY ? outHeight - y : nY) ;
	    const int64_t op = y * outWidth ;

	    _ve_lvl(vl) ;

	    __vr vrh_r0 = _ve_vaddsl_vsv(y  -padHeight, vry) ;
	    __vr vrh_r2 = _ve_vaddsl_vsv(y+2-padHeight, vry) ;

	    __vm256 vm01_r0 = _ve_vfmkl_mcv(VECC_GE, vrh_r0) ;
	    __vm256 vm01_r2 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(inHeight,vrh_r2)) ;

	    __vm256 vmall_r0s0 = _ve_andm_mmm(vm01_r0,vm23_s0) ;
	    __vm256 vmall_r0s1 = vm01_r0 ;
	    __vm256 vmall_r0s2 = _ve_andm_mmm(vm01_r0, vm23_s2) ;

	    __vm256 vmall_r1s0 = vm23_s0 ;
	    __vm256 vmall_r1s2 = vm23_s2 ;

	    __vm256 vmall_r2s0 = _ve_andm_mmm(vm01_r2,vm23_s0) ;
	    __vm256 vmall_r2s1 = vm01_r2 ;
	    __vm256 vmall_r2s2 = _ve_andm_mmm(vm01_r2, vm23_s2) ;

	    int64_t outIndex  = outGroupOffset + (n * outChannel + k  ) * oPixels + op;

	    const float *pInChannel = pIn + inGroupOffset + ((n * inChannel) * inHeight * inWidth ) ;

	      /* memory access errors mihgt be caused */
	    __vr vrin_r0s0 = _ve_vldu_vss(4,&pInChannel[op-inWidth-1]) ;
	    __vr vrin_r0s1 = _ve_vldu_vss(4,&pInChannel[op-inWidth  ]) ;
	    __vr vrin_r0s2 = _ve_vldu_vss(4,&pInChannel[op-inWidth+1]) ;
	    __vr vrin_r1s0 = _ve_vldu_vss(4,&pInChannel[op+       -1]) ;
	    __vr vrin_r1s1 = _ve_vldu_vss(4,&pInChannel[op          ]) ;
	    __vr vrin_r1s2 = _ve_vldu_vss(4,&pInChannel[op+       +1]) ;
	    __vr vrin_r2s0 = _ve_vldu_vss(4,&pInChannel[op+inWidth-1]) ;
	    __vr vrin_r2s1 = _ve_vldu_vss(4,&pInChannel[op+inWidth  ]) ;
	    __vr vrin_r2s2 = _ve_vldu_vss(4,&pInChannel[op+inWidth+1]) ;

	    vrin_r0s0 = _ve_vmrg_vvvm(vrzerof, vrin_r0s0, vmall_r0s0) ;
	    vrin_r0s1 = _ve_vmrg_vvvm(vrzerof, vrin_r0s1, vmall_r0s1) ;
	    vrin_r0s2 = _ve_vmrg_vvvm(vrzerof, vrin_r0s2, vmall_r0s2) ;
	    __vr vrinP_r0s0 = _ve_vshf_vvvs(vrin_r0s0, vrin_r0s0, VE_VSHUFFLE_YUZU) ;
	    __vr vrinP_r0s1 = _ve_vshf_vvvs(vrin_r0s1, vrin_r0s1, VE_VSHUFFLE_YUZU) ;
	    __vr vrinP_r0s2 = _ve_vshf_vvvs(vrin_r0s2, vrin_r0s2, VE_VSHUFFLE_YUZU) ;

	    vrin_r1s0 = _ve_vmrg_vvvm(vrzerof, vrin_r1s0, vmall_r1s0) ;
	    vrin_r1s2 = _ve_vmrg_vvvm(vrzerof, vrin_r1s2, vmall_r1s2) ;
	    __vr vrinP_r1s0 = _ve_vshf_vvvs(vrin_r1s0, vrin_r1s0, VE_VSHUFFLE_YUZU) ;
	    __vr vrinP_r1s1 = _ve_vshf_vvvs(vrin_r1s1, vrin_r1s1, VE_VSHUFFLE_YUZU) ;
	    __vr vrinP_r1s2 = _ve_vshf_vvvs(vrin_r1s2, vrin_r1s2, VE_VSHUFFLE_YUZU) ;

	    vrin_r2s0 = _ve_vmrg_vvvm(vrzerof, vrin_r2s0, vmall_r2s0) ;
	    vrin_r2s1 = _ve_vmrg_vvvm(vrzerof, vrin_r2s1, vmall_r2s1) ;
	    vrin_r2s2 = _ve_vmrg_vvvm(vrzerof, vrin_r2s2, vmall_r2s2) ;
	    __vr vrinP_r2s0 = _ve_vshf_vvvs(vrin_r2s0, vrin_r2s0, VE_VSHUFFLE_YUZU) ;
	    __vr vrinP_r2s1 = _ve_vshf_vvvs(vrin_r2s1, vrin_r2s1, VE_VSHUFFLE_YUZU) ;
	    __vr vrinP_r2s2 = _ve_vshf_vvvs(vrin_r2s2, vrin_r2s2, VE_VSHUFFLE_YUZU) ;

#define FILTER_R3S3(VRBIAS, N)						\
{									\
  __vr vrsum = (VRBIAS) ;						\
  vrsum = _ve_pvfmad_vvsv(vrsum, kerValue[9*(N)+0], vrinP_r0s0) ;	\
  vrsum = _ve_pvfmad_vvsv(vrsum, kerValue[9*(N)+1], vrinP_r0s1) ;	\
  vrsum = _ve_pvfmad_vvsv(vrsum, kerValue[9*(N)+2], vrinP_r0s2) ;	\
  vrsum = _ve_pvfmad_vvsv(vrsum, kerValue[9*(N)+3], vrinP_r1s0) ;	\
  vrsum = _ve_pvfmad_vvsv(vrsum, kerValue[9*(N)+4], vrinP_r1s1) ;	\
  vrsum = _ve_pvfmad_vvsv(vrsum, kerValue[9*(N)+5], vrinP_r1s2) ;	\
  vrsum = _ve_pvfmad_vvsv(vrsum, kerValue[9*(N)+6], vrinP_r2s0) ;	\
  vrsum = _ve_pvfmad_vvsv(vrsum, kerValue[9*(N)+7], vrinP_r2s1) ;	\
  vrsum = _ve_pvfmad_vvsv(vrsum, kerValue[9*(N)+8], vrinP_r2s2) ;	\
  _ve_vstu_vss(vrsum, 4, pOut+outIndex+ ((N)  )*oPixels) ;		\
  _ve_vstl_vss(vrsum, 4, pOut+outIndex+ ((N)+8)*oPixels) ;		\
}
	    FILTER_R3S3(vrzero, 0) ;
	    FILTER_R3S3(vrzero, 1) ;
	    FILTER_R3S3(vrzero, 2) ;
	    FILTER_R3S3(vrzero, 3) ;
	    FILTER_R3S3(vrzero, 4) ;
	    FILTER_R3S3(vrzero, 5) ;
	    FILTER_R3S3(vrzero, 6) ;
	    FILTER_R3S3(vrzero, 7) ;
#undef FILTER_R3S3
	  } // outPixels
	} // outChannel
      } // group
    } // batch
  }

  return VEDNN_SUCCESS;
}

