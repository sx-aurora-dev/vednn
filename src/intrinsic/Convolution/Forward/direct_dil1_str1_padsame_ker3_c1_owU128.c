#include <stdint.h>
#include <stdlib.h>

#include "vednn.h"

#include "velintrin.h"
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

    __vr vrseq = _vel_vseq_vl(VLEN) ;			// xy

    __vr vry   = _vel_vdivsl_vvsl(vrseq, outWidth, VLEN) ;
    __vr vrx   = _vel_vsubsl_vvvl(vrseq, _vel_vmulul_vsvl(outWidth,vry, VLEN), VLEN) ;

    __vr vrw_s0 = _vel_vaddsl_vsvl( -padWidth, vrx, VLEN) ;
    __vr vrw_s2 = _vel_vaddsl_vsvl(2-padWidth, vrx, VLEN) ;

    __vm256 vm23_s0  =  _vel_vfmklge_mvl(vrw_s0, VLEN) ;
    __vm256 vm23_s2  =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(inWidth,vrw_s2, VLEN), VLEN) ;

    __vr vrzerof = _vel_vbrds_vsl(0.0f, VLEN) ;

    for (int64_t n = 0; n < batch; n++) {
      for (int64_t g = 0; g < group; g++) {

	const int64_t inGroupOffset   = g * inHeight * inWidth;
	const int64_t outGroupOffset  = g * outChannelGroup * outHeight * outWidth;
	const int64_t kernGroupOffset = g * outChannelGroup * kernHeight * kernWidth;

	int k = 0 ;
	if ( (outChannelGroup & 0x01) == 1 ) {
	  const __vr vrzero = _vel_vbrds_vsl(0.0f, VLEN) ;

	  for (int64_t y=0; y<outHeight; y+=nY)
	  {
	    const int64_t vl = outWidth * (outHeight - y < nY ? outHeight - y : nY) ;
	    const int64_t op = y * outWidth ;

	    const float *pKerValue  = pKernel + kernGroupOffset + (k * kernHeight    ) * kernWidth ;
	    const float *pInChannel = pIn + inGroupOffset + ((n * inChannel) * inHeight * inWidth ) ;

	    _vel_pfchv_ssl(4,&pInChannel[op-inWidth-1], vl) ; // prefetch

	    __vr vrh_r0 = _vel_vaddsl_vsvl(y  -padHeight, vry, vl) ;
	    __vr vrh_r2 = _vel_vaddsl_vsvl(y+2-padHeight, vry, vl) ;

	    __vm256 vm01_r0 =  _vel_vfmklge_mvl(vrh_r0, vl) ;
	    __vm256 vm01_r2 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(inHeight,vrh_r2, vl), vl) ;

	    __vm256 vmall_r0s0 = _vel_andm_mmm(vm01_r0,vm23_s0) ;
	    __vm256 vmall_r0s1 = vm01_r0 ;
	    __vm256 vmall_r0s2 = _vel_andm_mmm(vm01_r0, vm23_s2) ;

	    __vm256 vmall_r1s0 = vm23_s0 ;
	    __vm256 vmall_r1s2 = vm23_s2 ;

	    __vm256 vmall_r2s0 = _vel_andm_mmm(vm01_r2,vm23_s0) ;
	    __vm256 vmall_r2s1 = vm01_r2 ;
	    __vm256 vmall_r2s2 = _vel_andm_mmm(vm01_r2, vm23_s2) ;

	    int64_t outIndex = outGroupOffset + (n * outChannel + k) * oPixels + op ;


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


	    vrin_r0s0 = _vel_vmrg_vvvml(vrzerof, vrin_r0s0, vmall_r0s0, vl) ;
	    vrin_r0s1 = _vel_vmrg_vvvml(vrzerof, vrin_r0s1, vmall_r0s1, vl) ;
	    vrin_r0s2 = _vel_vmrg_vvvml(vrzerof, vrin_r0s2, vmall_r0s2, vl) ;

	    vrin_r1s0 = _vel_vmrg_vvvml(vrzerof, vrin_r1s0, vmall_r1s0, vl) ;
	    vrin_r1s2 = _vel_vmrg_vvvml(vrzerof, vrin_r1s2, vmall_r1s2, vl) ;

	    vrin_r2s0 = _vel_vmrg_vvvml(vrzerof, vrin_r2s0, vmall_r2s0, vl) ;
	    vrin_r2s1 = _vel_vmrg_vvvml(vrzerof, vrin_r2s1, vmall_r2s1, vl) ;
	    vrin_r2s2 = _vel_vmrg_vvvml(vrzerof, vrin_r2s2, vmall_r2s2, vl) ;


	    __vr vrsum = vrzero ;
	    vrsum = _vel_vfmads_vvsvl(vrsum, pKerValue[0], vrin_r0s0, vl) ;
	    vrsum = _vel_vfmads_vvsvl(vrsum, pKerValue[1], vrin_r0s1, vl) ;
	    vrsum = _vel_vfmads_vvsvl(vrsum, pKerValue[2], vrin_r0s2, vl) ;
	    vrsum = _vel_vfmads_vvsvl(vrsum, pKerValue[3], vrin_r1s0, vl) ;
	    vrsum = _vel_vfmads_vvsvl(vrsum, pKerValue[4], vrin_r1s1, vl) ;
	    vrsum = _vel_vfmads_vvsvl(vrsum, pKerValue[5], vrin_r1s2, vl) ;
	    vrsum = _vel_vfmads_vvsvl(vrsum, pKerValue[6], vrin_r2s0, vl) ;
	    vrsum = _vel_vfmads_vvsvl(vrsum, pKerValue[7], vrin_r2s1, vl) ;
	    vrsum = _vel_vfmads_vvsvl(vrsum, pKerValue[8], vrin_r2s2, vl) ;
	    _vel_vstu_vssl(vrsum, 4, pOut+outIndex, vl) ;

	  } // outPixels
	  k++ ;
	}
	if ( ((outChannelGroup >> 1) & 0x01) == 1 ) {
	  const __vr vrzero = _vel_pvbrd_vsl(0UL, VLEN) ;

	  uint64_t kerValue[9] ;
	  const float *pKerValue  = pKernel + kernGroupOffset + (k * kernHeight    ) * kernWidth ;
	  kerValue[0] = _vel_pack_f32p(pKerValue, pKerValue+9); pKerValue++ ;
	  kerValue[1] = _vel_pack_f32p(pKerValue, pKerValue+9); pKerValue++ ;
	  kerValue[2] = _vel_pack_f32p(pKerValue, pKerValue+9); pKerValue++ ;
	  kerValue[3] = _vel_pack_f32p(pKerValue, pKerValue+9); pKerValue++ ;
	  kerValue[4] = _vel_pack_f32p(pKerValue, pKerValue+9); pKerValue++ ;
	  kerValue[5] = _vel_pack_f32p(pKerValue, pKerValue+9); pKerValue++ ;
	  kerValue[6] = _vel_pack_f32p(pKerValue, pKerValue+9); pKerValue++ ;
	  kerValue[7] = _vel_pack_f32p(pKerValue, pKerValue+9); pKerValue++ ;
	  kerValue[8] = _vel_pack_f32p(pKerValue, pKerValue+9); pKerValue++ ;


	  for (int64_t y=0; y<outHeight; y+=nY)
	  {
	    const int64_t vl = outWidth * (outHeight - y < nY ? outHeight - y : nY) ;
	    const int64_t op = y * outWidth ;

	    __vr vrh_r0 = _vel_vaddsl_vsvl(y  -padHeight, vry, vl) ;
	    __vr vrh_r2 = _vel_vaddsl_vsvl(y+2-padHeight, vry, vl) ;

	    __vm256 vm01_r0 =  _vel_vfmklge_mvl(vrh_r0, vl) ;
	    __vm256 vm01_r2 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(inHeight,vrh_r2, vl), vl) ;

	    __vm256 vmall_r0s0 = _vel_andm_mmm(vm01_r0,vm23_s0) ;
	    __vm256 vmall_r0s1 = vm01_r0 ;
	    __vm256 vmall_r0s2 = _vel_andm_mmm(vm01_r0, vm23_s2) ;

	    __vm256 vmall_r1s0 = vm23_s0 ;
	    __vm256 vmall_r1s2 = vm23_s2 ;

	    __vm256 vmall_r2s0 = _vel_andm_mmm(vm01_r2,vm23_s0) ;
	    __vm256 vmall_r2s1 = vm01_r2 ;
	    __vm256 vmall_r2s2 = _vel_andm_mmm(vm01_r2, vm23_s2) ;


	    int64_t outIndex = outGroupOffset + (n * outChannel + k  ) * oPixels + op ;

	    const float *pInChannel = pIn + inGroupOffset + ((n * inChannel) * inHeight * inWidth ) ;

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

	    __vr vrsum = vrzero ;
	    vrsum = _vel_pvfmad_vvsvl(vrsum, kerValue[0], vrinP_r0s0, vl) ;
	    vrsum = _vel_pvfmad_vvsvl(vrsum, kerValue[1], vrinP_r0s1, vl) ;
	    vrsum = _vel_pvfmad_vvsvl(vrsum, kerValue[2], vrinP_r0s2, vl) ;
	    vrsum = _vel_pvfmad_vvsvl(vrsum, kerValue[3], vrinP_r1s0, vl) ;
	    vrsum = _vel_pvfmad_vvsvl(vrsum, kerValue[4], vrinP_r1s1, vl) ;
	    vrsum = _vel_pvfmad_vvsvl(vrsum, kerValue[5], vrinP_r1s2, vl) ;
	    vrsum = _vel_pvfmad_vvsvl(vrsum, kerValue[6], vrinP_r2s0, vl) ;
	    vrsum = _vel_pvfmad_vvsvl(vrsum, kerValue[7], vrinP_r2s1, vl) ;
	    vrsum = _vel_pvfmad_vvsvl(vrsum, kerValue[8], vrinP_r2s2, vl) ;
	    _vel_vstu_vssl(vrsum, 4, pOut+outIndex, vl) ;
	    _vel_vstl_vssl(vrsum, 4, pOut+outIndex+ oPixels, vl) ;

	  } // outPixels

	  k+=2 ;
	}
	if ( ((outChannelGroup >> 2) & 0x01) == 1 ) {
	  const __vr vrzero = _vel_pvbrd_vsl(0UL, VLEN) ;

	  uint64_t kerValue[2*9] ;
	  {
	    const float *pKerValue  = pKernel + kernGroupOffset + (k * kernHeight    ) * kernWidth ;
	    __vr vrker0 = _vel_vldu_vssl(4, pKerValue    , 4*9) ;
	    __vr vrker1 = _vel_vldu_vssl(4, pKerValue+2*9, 4*9) ;
	    _vel_vst_vssl(_vel_vshf_vvvsl(vrker0, vrker1, VE_VSHUFFLE_YUZU, 4*9) ,8, kerValue, 4*9) ;
	  }

	  for (int64_t y=0; y<outHeight; y+=nY)
	  {
	    const int64_t vl = outWidth * (outHeight - y < nY ? outHeight - y : nY) ;
	    const int64_t op = y * outWidth ;

	    __vr vrh_r0 = _vel_vaddsl_vsvl(y  -padHeight, vry, vl) ;
	    __vr vrh_r2 = _vel_vaddsl_vsvl(y+2-padHeight, vry, vl) ;

	    __vm256 vm01_r0 =  _vel_vfmklge_mvl(vrh_r0, vl) ;
	    __vm256 vm01_r2 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(inHeight,vrh_r2, vl), vl) ;

	    __vm256 vmall_r0s0 = _vel_andm_mmm(vm01_r0,vm23_s0) ;
	    __vm256 vmall_r0s1 = vm01_r0 ;
	    __vm256 vmall_r0s2 = _vel_andm_mmm(vm01_r0, vm23_s2) ;

	    __vm256 vmall_r1s0 = vm23_s0 ;
	    __vm256 vmall_r1s2 = vm23_s2 ;

	    __vm256 vmall_r2s0 = _vel_andm_mmm(vm01_r2,vm23_s0) ;
	    __vm256 vmall_r2s1 = vm01_r2 ;
	    __vm256 vmall_r2s2 = _vel_andm_mmm(vm01_r2, vm23_s2) ;

	    int64_t outIndex = outGroupOffset + (n * outChannel + k  ) * oPixels + op ;

	    const float *pInChannel = pIn + inGroupOffset + ((n * inChannel) * inHeight * inWidth ) ;

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

#define FILTER_R3S3(VRBIAS, N)						\
{									\
  __vr vrsum = (VRBIAS) ;						\
  vrsum = _vel_pvfmad_vvsvl(vrsum, kerValue[9*(N)+0], vrinP_r0s0, vl) ;	\
  vrsum = _vel_pvfmad_vvsvl(vrsum, kerValue[9*(N)+1], vrinP_r0s1, vl) ;	\
  vrsum = _vel_pvfmad_vvsvl(vrsum, kerValue[9*(N)+2], vrinP_r0s2, vl) ;	\
  vrsum = _vel_pvfmad_vvsvl(vrsum, kerValue[9*(N)+3], vrinP_r1s0, vl) ;	\
  vrsum = _vel_pvfmad_vvsvl(vrsum, kerValue[9*(N)+4], vrinP_r1s1, vl) ;	\
  vrsum = _vel_pvfmad_vvsvl(vrsum, kerValue[9*(N)+5], vrinP_r1s2, vl) ;	\
  vrsum = _vel_pvfmad_vvsvl(vrsum, kerValue[9*(N)+6], vrinP_r2s0, vl) ;	\
  vrsum = _vel_pvfmad_vvsvl(vrsum, kerValue[9*(N)+7], vrinP_r2s1, vl) ;	\
  vrsum = _vel_pvfmad_vvsvl(vrsum, kerValue[9*(N)+8], vrinP_r2s2, vl) ;	\
  _vel_vstu_vssl(vrsum, 4, pOut+outIndex+ ((N)  )*oPixels, vl) ;		\
  _vel_vstl_vssl(vrsum, 4, pOut+outIndex+ ((N)+2)*oPixels, vl) ;		\
}
	    FILTER_R3S3(vrzero, 0) ;
	    FILTER_R3S3(vrzero, 1) ;
#undef FILTER_R3S3
	  } // outPixels

	  k+=4 ;
	}
	if ( ((outChannelGroup >> 3) & 0x01) == 1 ) {
	  const __vr vrzero = _vel_pvbrd_vsl(0UL, VLEN) ;

	  uint64_t kerValue[4*9] ;
	  {
	    const float *pKerValue  = pKernel + kernGroupOffset + (k * kernHeight    ) * kernWidth ;
	    __vr vrker0 = _vel_vldu_vssl(4, pKerValue    , 4*9) ;
	    __vr vrker1 = _vel_vldu_vssl(4, pKerValue+4*9, 4*9) ;
	    _vel_vst_vssl(_vel_vshf_vvvsl(vrker0, vrker1, VE_VSHUFFLE_YUZU, 4*9) ,8, kerValue, 4*9) ;
	  }

	  for (int64_t y=0; y<outHeight; y+=nY)
	  {
	    const int64_t vl = outWidth * (outHeight - y < nY ? outHeight - y : nY) ;
	    const int64_t op = y * outWidth ;

	    __vr vrh_r0 = _vel_vaddsl_vsvl(y  -padHeight, vry, vl) ;
	    __vr vrh_r2 = _vel_vaddsl_vsvl(y+2-padHeight, vry, vl) ;

	    __vm256 vm01_r0 =  _vel_vfmklge_mvl(vrh_r0, vl) ;
	    __vm256 vm01_r2 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(inHeight,vrh_r2, vl), vl) ;

	    __vm256 vmall_r0s0 = _vel_andm_mmm(vm01_r0,vm23_s0) ;
	    __vm256 vmall_r0s1 = vm01_r0 ;
	    __vm256 vmall_r0s2 = _vel_andm_mmm(vm01_r0, vm23_s2) ;

	    __vm256 vmall_r1s0 = vm23_s0 ;
	    __vm256 vmall_r1s2 = vm23_s2 ;

	    __vm256 vmall_r2s0 = _vel_andm_mmm(vm01_r2,vm23_s0) ;
	    __vm256 vmall_r2s1 = vm01_r2 ;
	    __vm256 vmall_r2s2 = _vel_andm_mmm(vm01_r2, vm23_s2) ;

	    int64_t outIndex = outGroupOffset + (n * outChannel + k  ) * oPixels + op ;

	    const float *pInChannel = pIn + inGroupOffset + ((n * inChannel) * inHeight * inWidth ) ;

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

#define FILTER_R3S3(VRBIAS, N)						\
{									\
  __vr vrsum = (VRBIAS) ;						\
  vrsum = _vel_pvfmad_vvsvl(vrsum, kerValue[9*(N)+0], vrinP_r0s0, vl) ;	\
  vrsum = _vel_pvfmad_vvsvl(vrsum, kerValue[9*(N)+1], vrinP_r0s1, vl) ;	\
  vrsum = _vel_pvfmad_vvsvl(vrsum, kerValue[9*(N)+2], vrinP_r0s2, vl) ;	\
  vrsum = _vel_pvfmad_vvsvl(vrsum, kerValue[9*(N)+3], vrinP_r1s0, vl) ;	\
  vrsum = _vel_pvfmad_vvsvl(vrsum, kerValue[9*(N)+4], vrinP_r1s1, vl) ;	\
  vrsum = _vel_pvfmad_vvsvl(vrsum, kerValue[9*(N)+5], vrinP_r1s2, vl) ;	\
  vrsum = _vel_pvfmad_vvsvl(vrsum, kerValue[9*(N)+6], vrinP_r2s0, vl) ;	\
  vrsum = _vel_pvfmad_vvsvl(vrsum, kerValue[9*(N)+7], vrinP_r2s1, vl) ;	\
  vrsum = _vel_pvfmad_vvsvl(vrsum, kerValue[9*(N)+8], vrinP_r2s2, vl) ;	\
  _vel_vstu_vssl(vrsum, 4, pOut+outIndex+ ((N)  )*oPixels, vl) ;		\
  _vel_vstl_vssl(vrsum, 4, pOut+outIndex+ ((N)+4)*oPixels, vl) ;		\
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
	  const __vr vrzero = _vel_pvbrd_vsl(0UL, VLEN) ;

	  uint64_t kerValue[8*9] ;
	  {
	    const float *pKerValue  = pKernel + kernGroupOffset + (k * kernHeight    ) * kernWidth ;
	    __vr vrker0 = _vel_vldu_vssl(4, pKerValue    , 8*9) ;
	    __vr vrker1 = _vel_vldu_vssl(4, pKerValue+8*9, 8*9) ;
	    _vel_vst_vssl(_vel_vshf_vvvsl(vrker0, vrker1, VE_VSHUFFLE_YUZU, 8*9) ,8, kerValue, 8*9) ;
	  }

	  for (int64_t y=0; y<outHeight; y+=nY)
	  {
	    const int64_t vl = outWidth * (outHeight - y < nY ? outHeight - y : nY) ;
	    const int64_t op = y * outWidth ;

	    __vr vrh_r0 = _vel_vaddsl_vsvl(y  -padHeight, vry, vl) ;
	    __vr vrh_r2 = _vel_vaddsl_vsvl(y+2-padHeight, vry, vl) ;

	    __vm256 vm01_r0 =  _vel_vfmklge_mvl(vrh_r0, vl) ;
	    __vm256 vm01_r2 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(inHeight,vrh_r2, vl), vl) ;

	    __vm256 vmall_r0s0 = _vel_andm_mmm(vm01_r0,vm23_s0) ;
	    __vm256 vmall_r0s1 = vm01_r0 ;
	    __vm256 vmall_r0s2 = _vel_andm_mmm(vm01_r0, vm23_s2) ;

	    __vm256 vmall_r1s0 = vm23_s0 ;
	    __vm256 vmall_r1s2 = vm23_s2 ;

	    __vm256 vmall_r2s0 = _vel_andm_mmm(vm01_r2,vm23_s0) ;
	    __vm256 vmall_r2s1 = vm01_r2 ;
	    __vm256 vmall_r2s2 = _vel_andm_mmm(vm01_r2, vm23_s2) ;

	    int64_t outIndex  = outGroupOffset + (n * outChannel + k  ) * oPixels + op;

	    const float *pInChannel = pIn + inGroupOffset + ((n * inChannel) * inHeight * inWidth ) ;

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

#define FILTER_R3S3(VRBIAS, N)						\
{									\
  __vr vrsum = (VRBIAS) ;						\
  vrsum = _vel_pvfmad_vvsvl(vrsum, kerValue[9*(N)+0], vrinP_r0s0, vl) ;	\
  vrsum = _vel_pvfmad_vvsvl(vrsum, kerValue[9*(N)+1], vrinP_r0s1, vl) ;	\
  vrsum = _vel_pvfmad_vvsvl(vrsum, kerValue[9*(N)+2], vrinP_r0s2, vl) ;	\
  vrsum = _vel_pvfmad_vvsvl(vrsum, kerValue[9*(N)+3], vrinP_r1s0, vl) ;	\
  vrsum = _vel_pvfmad_vvsvl(vrsum, kerValue[9*(N)+4], vrinP_r1s1, vl) ;	\
  vrsum = _vel_pvfmad_vvsvl(vrsum, kerValue[9*(N)+5], vrinP_r1s2, vl) ;	\
  vrsum = _vel_pvfmad_vvsvl(vrsum, kerValue[9*(N)+6], vrinP_r2s0, vl) ;	\
  vrsum = _vel_pvfmad_vvsvl(vrsum, kerValue[9*(N)+7], vrinP_r2s1, vl) ;	\
  vrsum = _vel_pvfmad_vvsvl(vrsum, kerValue[9*(N)+8], vrinP_r2s2, vl) ;	\
  _vel_vstu_vssl(vrsum, 4, pOut+outIndex+ ((N)  )*oPixels, vl) ;		\
  _vel_vstl_vssl(vrsum, 4, pOut+outIndex+ ((N)+8)*oPixels, vl) ;		\
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

