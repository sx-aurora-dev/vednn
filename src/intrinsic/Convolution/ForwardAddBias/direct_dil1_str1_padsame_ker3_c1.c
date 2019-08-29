#include <stdint.h>
#include <stdlib.h>

#include "vednn.h"

#include "velintrin.h"
#define VLEN	(256)


vednnError_t
vednnConvolutionForwardAddBias_direct_dil1_str1_padsame_ker3_c1(
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

//  const int64_t inChannelGroup  = inChannel  / group;   // equal to pDataKernel->inChannel ( must be 1 )
  const int64_t outChannelGroup = outChannel / group;   // equal to pDataKernel->outChannel

  const float * restrict pIn     = pDataIn;
  const float * restrict pKernel = pDataKernel;
  const float * restrict pBias   = pDataBias;
  float * restrict const pOut    = pDataOut;

  const int oPixels= outHeight*outWidth ;

  {
    for (int64_t n = 0; n < batch; n++) {
      for (int64_t g = 0; g < group; g++) {

	const int64_t inGroupOffset   = g * inHeight * inWidth;
	const int64_t outGroupOffset  = g * outChannelGroup * outHeight * outWidth;
	const int64_t kernGroupOffset = g * outChannelGroup * kernHeight * kernWidth;

	int k = 0 ;
	if ( (outChannelGroup & 0x01) == 1 ) {

	  const float bias = (pBias == NULL) ? 0.0 : pBias[g * outChannelGroup + k];

	  for (int64_t op = 0; op < oPixels; op+=VLEN) {
	    const int64_t vl = oPixels - op < VLEN ? oPixels - op : VLEN ;

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

	    int64_t outIndex = outGroupOffset + (n * outChannel + k) * oPixels + op ;

	    const float *pKerValue  = pKernel + kernGroupOffset + (k * kernHeight    ) * kernWidth ;
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

	    __vr vrzerof = _vel_vbrds_vsl(0.0f, vl) ;

	    vrin_r0s0 = _vel_vmrg_vvvml(vrzerof, vrin_r0s0, vmall_r0s0, vl) ;
	    vrin_r0s1 = _vel_vmrg_vvvml(vrzerof, vrin_r0s1, vmall_r0s1, vl) ;
	    vrin_r0s2 = _vel_vmrg_vvvml(vrzerof, vrin_r0s2, vmall_r0s2, vl) ;

	    vrin_r1s0 = _vel_vmrg_vvvml(vrzerof, vrin_r1s0, vmall_r1s0, vl) ;
	    vrin_r1s2 = _vel_vmrg_vvvml(vrzerof, vrin_r1s2, vmall_r1s2, vl) ;

	    vrin_r2s0 = _vel_vmrg_vvvml(vrzerof, vrin_r2s0, vmall_r2s0, vl) ;
	    vrin_r2s1 = _vel_vmrg_vvvml(vrzerof, vrin_r2s1, vmall_r2s1, vl) ;
	    vrin_r2s2 = _vel_vmrg_vvvml(vrzerof, vrin_r2s2, vmall_r2s2, vl) ;


	    __vr vrsum = _vel_vbrds_vsl(bias, vl) ;
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
	  const float bias0 = (pBias == NULL) ? 0.0 : pBias[g * outChannelGroup + k  ];
	  const float bias1 = (pBias == NULL) ? 0.0 : pBias[g * outChannelGroup + k+1];

	  const uint64_t bias01 = _vel_pack_f32p(&bias0, &bias1) ;

	  for (int64_t op = 0; op < oPixels; op+=VLEN) {
	    const int64_t vl = oPixels - op < VLEN ? oPixels - op : VLEN ;

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

	    int64_t outIndex = outGroupOffset + (n * outChannel + k  ) * oPixels + op ;

	    const float *pKerValue  = pKernel + kernGroupOffset + (k * kernHeight    ) * kernWidth ;
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

#define FILTER_R3S3(BIAS, N)					\
{								\
  __vr vrsum = _vel_pvbrd_vsl(BIAS, vl) ;			\
  const uint64_t kerValue_r0s0 = _vel_pack_f32p(pKerValue,	\
					       pKerValue  +9);	\
  const uint64_t kerValue_r0s1 = _vel_pack_f32p(pKerValue+1,	\
					       pKerValue+1+9);	\
  const uint64_t kerValue_r0s2 = _vel_pack_f32p(pKerValue+2,	\
					       pKerValue+2+9);	\
  const uint64_t kerValue_r1s0 = _vel_pack_f32p(pKerValue+3,	\
					       pKerValue+3+9);	\
  const uint64_t kerValue_r1s1 = _vel_pack_f32p(pKerValue+4,	\
					       pKerValue+4+9);	\
  const uint64_t kerValue_r1s2 = _vel_pack_f32p(pKerValue+5,	\
					       pKerValue+5+9);	\
  const uint64_t kerValue_r2s0 = _vel_pack_f32p(pKerValue+6,	\
					       pKerValue+6+9);	\
  const uint64_t kerValue_r2s1 = _vel_pack_f32p(pKerValue+7,	\
					       pKerValue+7+9);	\
  const uint64_t kerValue_r2s2 = _vel_pack_f32p(pKerValue+8,	\
					       pKerValue+8+9);	\
  vrsum = _vel_pvfmad_vvsvl(vrsum, kerValue_r0s0, vrinP_r0s0, vl) ;	\
  vrsum = _vel_pvfmad_vvsvl(vrsum, kerValue_r0s1, vrinP_r0s1, vl) ;	\
  vrsum = _vel_pvfmad_vvsvl(vrsum, kerValue_r0s2, vrinP_r0s2, vl) ;	\
  vrsum = _vel_pvfmad_vvsvl(vrsum, kerValue_r1s0, vrinP_r1s0, vl) ;	\
  vrsum = _vel_pvfmad_vvsvl(vrsum, kerValue_r1s1, vrinP_r1s1, vl) ;	\
  vrsum = _vel_pvfmad_vvsvl(vrsum, kerValue_r1s2, vrinP_r1s2, vl) ;	\
  vrsum = _vel_pvfmad_vvsvl(vrsum, kerValue_r2s0, vrinP_r2s0, vl) ;	\
  vrsum = _vel_pvfmad_vvsvl(vrsum, kerValue_r2s1, vrinP_r2s1, vl) ;	\
  vrsum = _vel_pvfmad_vvsvl(vrsum, kerValue_r2s2, vrinP_r2s2, vl) ;	\
  _vel_vstu_vssl(vrsum, 4, pOut+outIndex+ (N)    *oPixels, vl) ;	\
  _vel_vstl_vssl(vrsum, 4, pOut+outIndex+ ((N)+1)*oPixels, vl) ;	\
}
	    FILTER_R3S3(bias01, 0) ; pKerValue += 18 ;
#undef FILTER_R3S3

	  } // outPixels

	  k+=2 ;
	}
	if ( ((outChannelGroup >> 2) & 0x01) == 1 ) {
	  const float bias0 = (pBias == NULL) ? 0.0 : pBias[g * outChannelGroup + k  ];
	  const float bias1 = (pBias == NULL) ? 0.0 : pBias[g * outChannelGroup + k+1];
	  const float bias2 = (pBias == NULL) ? 0.0 : pBias[g * outChannelGroup + k+2];
	  const float bias3 = (pBias == NULL) ? 0.0 : pBias[g * outChannelGroup + k+3];

	  const uint64_t bias01 = _vel_pack_f32p(&bias0, &bias1) ;
	  const uint64_t bias23 = _vel_pack_f32p(&bias2, &bias3) ;

	  for (int64_t op = 0; op < oPixels; op+=VLEN) {
	    const int64_t vl = oPixels - op < VLEN ? oPixels - op : VLEN ;

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


	    int64_t outIndex = outGroupOffset + (n * outChannel + k  ) * oPixels + op ;

	    const float *pKerValue  = pKernel + kernGroupOffset + (k * kernHeight    ) * kernWidth ;
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

#define FILTER_R3S3(BIAS, N)					\
{								\
  __vr vrsum = _vel_pvbrd_vsl(BIAS, vl) ;				\
  const uint64_t kerValue_r0s0 = _vel_pack_f32p(pKerValue,	\
					       pKerValue  +9);	\
  const uint64_t kerValue_r0s1 = _vel_pack_f32p(pKerValue+1,	\
					       pKerValue+1+9);	\
  const uint64_t kerValue_r0s2 = _vel_pack_f32p(pKerValue+2,	\
					       pKerValue+2+9);	\
  const uint64_t kerValue_r1s0 = _vel_pack_f32p(pKerValue+3,	\
					       pKerValue+3+9);	\
  const uint64_t kerValue_r1s1 = _vel_pack_f32p(pKerValue+4,	\
					       pKerValue+4+9);	\
  const uint64_t kerValue_r1s2 = _vel_pack_f32p(pKerValue+5,	\
					       pKerValue+5+9);	\
  const uint64_t kerValue_r2s0 = _vel_pack_f32p(pKerValue+6,	\
					       pKerValue+6+9);	\
  const uint64_t kerValue_r2s1 = _vel_pack_f32p(pKerValue+7,	\
					       pKerValue+7+9);	\
  const uint64_t kerValue_r2s2 = _vel_pack_f32p(pKerValue+8,	\
					       pKerValue+8+9);	\
  vrsum = _vel_pvfmad_vvsvl(vrsum, kerValue_r0s0, vrinP_r0s0, vl) ;	\
  vrsum = _vel_pvfmad_vvsvl(vrsum, kerValue_r0s1, vrinP_r0s1, vl) ;	\
  vrsum = _vel_pvfmad_vvsvl(vrsum, kerValue_r0s2, vrinP_r0s2, vl) ;	\
  vrsum = _vel_pvfmad_vvsvl(vrsum, kerValue_r1s0, vrinP_r1s0, vl) ;	\
  vrsum = _vel_pvfmad_vvsvl(vrsum, kerValue_r1s1, vrinP_r1s1, vl) ;	\
  vrsum = _vel_pvfmad_vvsvl(vrsum, kerValue_r1s2, vrinP_r1s2, vl) ;	\
  vrsum = _vel_pvfmad_vvsvl(vrsum, kerValue_r2s0, vrinP_r2s0, vl) ;	\
  vrsum = _vel_pvfmad_vvsvl(vrsum, kerValue_r2s1, vrinP_r2s1, vl) ;	\
  vrsum = _vel_pvfmad_vvsvl(vrsum, kerValue_r2s2, vrinP_r2s2, vl) ;	\
  _vel_vstu_vssl(vrsum, 4, pOut+outIndex+ (N)    *oPixels, vl) ;	\
  _vel_vstl_vssl(vrsum, 4, pOut+outIndex+ ((N)+1)*oPixels, vl) ;	\
}
	    FILTER_R3S3(bias01, 0) ; pKerValue += 18 ;
	    FILTER_R3S3(bias23, 2) ; pKerValue += 18 ;
#undef FILTER_R3S3
	  } // outPixels


	  k+=4 ;
	}
	if ( ((outChannelGroup >> 3) & 0x01) == 1 ) {
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

	    int64_t outIndex = outGroupOffset + (n * outChannel + k  ) * oPixels + op ;

	    const float *pKerValue  = pKernel + kernGroupOffset + (k * kernHeight    ) * kernWidth ;
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

#define FILTER_R3S3(BIAS, N)					\
{								\
  __vr vrsum = _vel_pvbrd_vsl(BIAS, vl) ;			\
  const uint64_t kerValue_r0s0 = _vel_pack_f32p(pKerValue,	\
					       pKerValue  +9);	\
  const uint64_t kerValue_r0s1 = _vel_pack_f32p(pKerValue+1,	\
					       pKerValue+1+9);	\
  const uint64_t kerValue_r0s2 = _vel_pack_f32p(pKerValue+2,	\
					       pKerValue+2+9);	\
  const uint64_t kerValue_r1s0 = _vel_pack_f32p(pKerValue+3,	\
					       pKerValue+3+9);	\
  const uint64_t kerValue_r1s1 = _vel_pack_f32p(pKerValue+4,	\
					       pKerValue+4+9);	\
  const uint64_t kerValue_r1s2 = _vel_pack_f32p(pKerValue+5,	\
					       pKerValue+5+9);	\
  const uint64_t kerValue_r2s0 = _vel_pack_f32p(pKerValue+6,	\
					       pKerValue+6+9);	\
  const uint64_t kerValue_r2s1 = _vel_pack_f32p(pKerValue+7,	\
					       pKerValue+7+9);	\
  const uint64_t kerValue_r2s2 = _vel_pack_f32p(pKerValue+8,	\
					       pKerValue+8+9);	\
  vrsum = _vel_pvfmad_vvsvl(vrsum, kerValue_r0s0, vrinP_r0s0, vl) ;	\
  vrsum = _vel_pvfmad_vvsvl(vrsum, kerValue_r0s1, vrinP_r0s1, vl) ;	\
  vrsum = _vel_pvfmad_vvsvl(vrsum, kerValue_r0s2, vrinP_r0s2, vl) ;	\
  vrsum = _vel_pvfmad_vvsvl(vrsum, kerValue_r1s0, vrinP_r1s0, vl) ;	\
  vrsum = _vel_pvfmad_vvsvl(vrsum, kerValue_r1s1, vrinP_r1s1, vl) ;	\
  vrsum = _vel_pvfmad_vvsvl(vrsum, kerValue_r1s2, vrinP_r1s2, vl) ;	\
  vrsum = _vel_pvfmad_vvsvl(vrsum, kerValue_r2s0, vrinP_r2s0, vl) ;	\
  vrsum = _vel_pvfmad_vvsvl(vrsum, kerValue_r2s1, vrinP_r2s1, vl) ;	\
  vrsum = _vel_pvfmad_vvsvl(vrsum, kerValue_r2s2, vrinP_r2s2, vl) ;	\
  _vel_vstu_vssl(vrsum, 4, pOut+outIndex+ (N)    *oPixels, vl) ;	\
  _vel_vstl_vssl(vrsum, 4, pOut+outIndex+ ((N)+1)*oPixels, vl) ;	\
}
	    FILTER_R3S3(bias01, 0) ; pKerValue += 18 ;
	    FILTER_R3S3(bias23, 2) ; pKerValue += 18 ;
	    FILTER_R3S3(bias45, 4) ; pKerValue += 18 ;
	    FILTER_R3S3(bias67, 6) ; pKerValue += 18 ;
#undef FILTER_R3S3
	  } // outPixels
	  k+=8 ;
	}
	for ( ; k < outChannelGroup; k+=16) {
	  const float bias0 = (pBias == NULL) ? 0.0 : pBias[g * outChannelGroup + k  ];
	  const float bias1 = (pBias == NULL) ? 0.0 : pBias[g * outChannelGroup + k+1];
	  const float bias2 = (pBias == NULL) ? 0.0 : pBias[g * outChannelGroup + k+2];
	  const float bias3 = (pBias == NULL) ? 0.0 : pBias[g * outChannelGroup + k+3];
	  const float bias4 = (pBias == NULL) ? 0.0 : pBias[g * outChannelGroup + k+4];
	  const float bias5 = (pBias == NULL) ? 0.0 : pBias[g * outChannelGroup + k+5];
	  const float bias6 = (pBias == NULL) ? 0.0 : pBias[g * outChannelGroup + k+6];
	  const float bias7 = (pBias == NULL) ? 0.0 : pBias[g * outChannelGroup + k+7];
	  const float bias8 = (pBias == NULL) ? 0.0 : pBias[g * outChannelGroup + k+8];
	  const float bias9 = (pBias == NULL) ? 0.0 : pBias[g * outChannelGroup + k+9];
	  const float biasA = (pBias == NULL) ? 0.0 : pBias[g * outChannelGroup + k+10];
	  const float biasB = (pBias == NULL) ? 0.0 : pBias[g * outChannelGroup + k+11];
	  const float biasC = (pBias == NULL) ? 0.0 : pBias[g * outChannelGroup + k+12];
	  const float biasD = (pBias == NULL) ? 0.0 : pBias[g * outChannelGroup + k+13];
	  const float biasE = (pBias == NULL) ? 0.0 : pBias[g * outChannelGroup + k+14];
	  const float biasF = (pBias == NULL) ? 0.0 : pBias[g * outChannelGroup + k+15];


	  const uint64_t bias01 = _vel_pack_f32p(&bias0, &bias1) ;
	  const uint64_t bias23 = _vel_pack_f32p(&bias2, &bias3) ;
	  const uint64_t bias45 = _vel_pack_f32p(&bias4, &bias5) ;
	  const uint64_t bias67 = _vel_pack_f32p(&bias6, &bias7) ;
	  const uint64_t bias89 = _vel_pack_f32p(&bias8, &bias9) ;
	  const uint64_t biasAB = _vel_pack_f32p(&biasA, &biasB) ;
	  const uint64_t biasCD = _vel_pack_f32p(&biasC, &biasD) ;
	  const uint64_t biasEF = _vel_pack_f32p(&biasE, &biasF) ;

	  for (int64_t op = 0; op < oPixels; op+=VLEN) {
	    const int64_t vl = oPixels - op < VLEN ? oPixels - op : VLEN ;

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

	    int64_t outIndex  = outGroupOffset + (n * outChannel + k  ) * oPixels + op;

	    const float *pKerValue  = pKernel + kernGroupOffset + (k * kernHeight    ) * kernWidth ;
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

#define FILTER_R3S3(BIAS, N)					\
{								\
  __vr vrsum = _vel_pvbrd_vsl(BIAS, vl) ;			\
  const uint64_t kerValue_r0s0 = _vel_pack_f32p(pKerValue,	\
					       pKerValue  +9);	\
  const uint64_t kerValue_r0s1 = _vel_pack_f32p(pKerValue+1,	\
					       pKerValue+1+9);	\
  const uint64_t kerValue_r0s2 = _vel_pack_f32p(pKerValue+2,	\
					       pKerValue+2+9);	\
  const uint64_t kerValue_r1s0 = _vel_pack_f32p(pKerValue+3,	\
					       pKerValue+3+9);	\
  const uint64_t kerValue_r1s1 = _vel_pack_f32p(pKerValue+4,	\
					       pKerValue+4+9);	\
  const uint64_t kerValue_r1s2 = _vel_pack_f32p(pKerValue+5,	\
					       pKerValue+5+9);	\
  const uint64_t kerValue_r2s0 = _vel_pack_f32p(pKerValue+6,	\
					       pKerValue+6+9);	\
  const uint64_t kerValue_r2s1 = _vel_pack_f32p(pKerValue+7,	\
					       pKerValue+7+9);	\
  const uint64_t kerValue_r2s2 = _vel_pack_f32p(pKerValue+8,	\
					       pKerValue+8+9);	\
  vrsum = _vel_pvfmad_vvsvl(vrsum, kerValue_r0s0, vrinP_r0s0, vl) ;	\
  vrsum = _vel_pvfmad_vvsvl(vrsum, kerValue_r0s1, vrinP_r0s1, vl) ;	\
  vrsum = _vel_pvfmad_vvsvl(vrsum, kerValue_r0s2, vrinP_r0s2, vl) ;	\
  vrsum = _vel_pvfmad_vvsvl(vrsum, kerValue_r1s0, vrinP_r1s0, vl) ;	\
  vrsum = _vel_pvfmad_vvsvl(vrsum, kerValue_r1s1, vrinP_r1s1, vl) ;	\
  vrsum = _vel_pvfmad_vvsvl(vrsum, kerValue_r1s2, vrinP_r1s2, vl) ;	\
  vrsum = _vel_pvfmad_vvsvl(vrsum, kerValue_r2s0, vrinP_r2s0, vl) ;	\
  vrsum = _vel_pvfmad_vvsvl(vrsum, kerValue_r2s1, vrinP_r2s1, vl) ;	\
  vrsum = _vel_pvfmad_vvsvl(vrsum, kerValue_r2s2, vrinP_r2s2, vl) ;	\
  _vel_vstu_vssl(vrsum, 4, pOut+outIndex+ (N)    *oPixels, vl) ;	\
  _vel_vstl_vssl(vrsum, 4, pOut+outIndex+ ((N)+1)*oPixels, vl) ;	\
}
	    FILTER_R3S3(bias01, 0) ; pKerValue += 18 ;
	    FILTER_R3S3(bias23, 2) ; pKerValue += 18 ;
	    FILTER_R3S3(bias45, 4) ; pKerValue += 18 ;
	    FILTER_R3S3(bias67, 6) ; pKerValue += 18 ;
	    FILTER_R3S3(bias89, 8) ; pKerValue += 18 ;
	    FILTER_R3S3(biasAB, 10) ; pKerValue += 18 ;
	    FILTER_R3S3(biasCD, 12) ; pKerValue += 18 ;
	    FILTER_R3S3(biasEF, 14) ; pKerValue += 18 ;

#undef FILTER_R3S3
	  } // outPixels
	} // outChannel
      } // group
    } // batch
  }

  return VEDNN_SUCCESS;
}

