#include <stdint.h>
#include <stdlib.h>

#include "vednn.h"

#include "velintrin.h"
#define VLEN	(256)

vednnError_t
vednnConvolutionForwardAddBias_direct_dil1_str1_padsame(
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
  const int64_t kernWidth  = pParamKernel->width;		/* must be 2*padWidth  + 1 */
  const int64_t kernHeight = pParamKernel->height;		/* must be 2*padHeight + 1 */

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

	    __vr vrseq = _vel_vseq_vl(vl) ;			// xy
	    __vr vridx = _vel_vaddsl_vsvl(op, vrseq, vl) ;	// op + xy

	    __vr vrsum = _vel_vbrds_vsl(bias, vl) ;
	    __vr vry   = _vel_vdivsl_vvsl(vridx, outWidth, vl) ;
	    __vr vrx   = _vel_vsubsl_vvvl(vridx, _vel_vmulul_vsvl(outWidth,vry, vl), vl) ;

	    for (int64_t r = 0; r < kernHeight; r++) {
	      for (int64_t s = 0; s < kernWidth; s++) {
		__vr vrh = _vel_vaddsl_vsvl(r-padHeight, vry, vl) ;
		__vr vrw = _vel_vaddsl_vsvl(s-padWidth,  vrx, vl) ;

		__vm256 vm0 =  _vel_vfmklge_mvl(vrh, vl) ;				// condition(0 <= h)
		__vm256 vm1 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(inHeight,vrh, vl), vl) ;	// condition(h < inHeight)
		__vm256 vm2 =  _vel_vfmklge_mvl(vrw, vl) ;				// condition(0 <= w)
		__vm256 vm3 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(inWidth,vrw, vl), vl) ;	// condition(w < inWidth)

		__vm256 vm01  = _vel_andm_mmm(vm0, vm1) ;
		__vm256 vm23  = _vel_andm_mmm(vm2, vm3) ;
		__vm256 vmall = _vel_andm_mmm(vm01,vm23) ;

		for (int64_t c = 0; c < inChannelGroup; c++) {
		  const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;

		  /* memory access errors mihgt be caused */
		  __vr vrin = _vel_vldu_vssl(4,&pInChannel[op+(r-padHeight)*inWidth+s-padWidth], vl) ;
		  vrin = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrin, vmall, vl) ;

		  const float *pKerValue = pKernel + kernGroupOffset + ((k * inChannelGroup + c) * kernHeight + r) * kernWidth + s;

		  vrsum = _vel_vfmads_vvsvl(vrsum, *pKerValue, vrin, vl) ;

	        } // kernWidth
	      } // kernHeight
	    } // inChannel

	    _vel_vstu_vssl(vrsum, 4, pOut+outIndex, vl) ;

	    outIndex += vl ;
	  } // outPixels
	  k++ ;
	}
	if ( ((outChannelGroup >> 1) & 0x01) == 1 ) {
	  int64_t outIndex0 = outGroupOffset + (n * outChannel + k  ) * oPixels ;
	  int64_t outIndex1 = outGroupOffset + (n * outChannel + k+1) * oPixels ;

	  const float bias0 = (pBias == NULL) ? 0.0 : pBias[g * outChannelGroup + k  ];
	  const float bias1 = (pBias == NULL) ? 0.0 : pBias[g * outChannelGroup + k+1];

	  const uint64_t bias01 = _vel_pack_f32p(&bias0, &bias1) ;

	  for (int64_t op = 0; op < oPixels; op+=VLEN) {
	    const int64_t vl = oPixels - op < VLEN ? oPixels - op : VLEN ;

	    __vr vrseq = _vel_vseq_vl(vl) ;			// xy
	    __vr vridx = _vel_vaddsl_vsvl(op, vrseq, vl) ;	// op + xy

	    __vr vrsum01 = _vel_pvbrd_vsl(bias01, vl) ;

	    __vr vry   = _vel_vdivsl_vvsl(vridx, outWidth, vl) ;
	    __vr vrx   = _vel_vsubsl_vvvl(vridx, _vel_vmulul_vsvl(outWidth,vry, vl), vl) ;

	    for (int64_t r = 0; r < kernHeight; r++) {
	      for (int64_t s = 0; s < kernWidth; s++) {
		__vr vrh = _vel_vaddsl_vsvl(r-padHeight, vry, vl) ;
		__vr vrw = _vel_vaddsl_vsvl(s-padWidth,  vrx, vl) ;

		__vm256 vm0 =  _vel_vfmklge_mvl(vrh, vl) ;				// condition(0 <= h)
		__vm256 vm1 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(inHeight,vrh, vl), vl) ;	// condition(h < inHeight)
		__vm256 vm2 =  _vel_vfmklge_mvl(vrw, vl) ;				// condition(0 <= w)
		__vm256 vm3 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(inWidth,vrw, vl), vl) ;	// condition(w < inWidth)

		__vm256 vm01  = _vel_andm_mmm(vm0, vm1) ;
		__vm256 vm23  = _vel_andm_mmm(vm2, vm3) ;
		__vm256 vmall = _vel_andm_mmm(vm01,vm23) ;

		for (int64_t c = 0; c < inChannelGroup; c++) {
		  const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;

		  /* memory access errors mihgt be caused */
		  __vr vrin = _vel_vldu_vssl(4,&pInChannel[op+(r-padHeight)*inWidth+s-padWidth], vl) ;
		  vrin = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrin, vmall, vl) ;

		  __vr vrinP = _vel_vshf_vvvsl(vrin, vrin, VE_VSHUFFLE_YUZU, vl) ;

		  const float *pKerValue = pKernel + kernGroupOffset + ((k * inChannelGroup + c) * kernHeight + r) * kernWidth + s;

		  const uint64_t kerValue01 = _vel_pack_f32p(pKerValue,
							    pKerValue+      inChannelGroup * kernHeight * kernWidth) ;

		  vrsum01 = _vel_pvfmad_vvsvl(vrsum01, kerValue01, vrinP, vl) ;
		} // inChannel
	      } // kernWidth
	    } // kernHeight

	    _vel_vstu_vssl(vrsum01, 4, pOut+outIndex0, vl) ;
	    _vel_vstl_vssl(vrsum01, 4, pOut+outIndex1, vl) ;

	    outIndex0 += vl ;
	    outIndex1 += vl ;
	  } // outPixels

	  k+=2 ;
	}
	if ( ((outChannelGroup >> 2) & 0x01) == 1 ) {
	  int64_t outIndex0 = outGroupOffset + (n * outChannel + k  ) * oPixels ;
	  int64_t outIndex1 = outGroupOffset + (n * outChannel + k+1) * oPixels ;
	  int64_t outIndex2 = outGroupOffset + (n * outChannel + k+2) * oPixels ;
	  int64_t outIndex3 = outGroupOffset + (n * outChannel + k+3) * oPixels ;

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

	    __vr vrsum01 = _vel_pvbrd_vsl(bias01, vl) ;
	    __vr vrsum23 = _vel_pvbrd_vsl(bias23, vl) ;

	    __vr vry   = _vel_vdivsl_vvsl(vridx, outWidth, vl) ;
	    __vr vrx   = _vel_vsubsl_vvvl(vridx, _vel_vmulul_vsvl(outWidth,vry, vl), vl) ;

	    for (int64_t r = 0; r < kernHeight; r++) {
	      for (int64_t s = 0; s < kernWidth; s++) {
		__vr vrh = _vel_vaddsl_vsvl(r-padHeight, vry, vl) ;
		__vr vrw = _vel_vaddsl_vsvl(s-padWidth,  vrx, vl) ;

		__vm256 vm0 =  _vel_vfmklge_mvl(vrh, vl) ;				// condition(0 <= h)
		__vm256 vm1 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(inHeight,vrh, vl), vl) ;	// condition(h < inHeight)
		__vm256 vm2 =  _vel_vfmklge_mvl(vrw, vl) ;				// condition(0 <= w)
		__vm256 vm3 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(inWidth,vrw, vl), vl) ;	// condition(w < inWidth)

		__vm256 vm01  = _vel_andm_mmm(vm0, vm1) ;
		__vm256 vm23  = _vel_andm_mmm(vm2, vm3) ;
		__vm256 vmall = _vel_andm_mmm(vm01,vm23) ;

		for (int64_t c = 0; c < inChannelGroup; c++) {
		  const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;

		  /* memory access errors mihgt be caused */
		  __vr vrin = _vel_vldu_vssl(4,&pInChannel[op+(r-padHeight)*inWidth+s-padWidth], vl) ;
		  vrin = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrin, vmall, vl) ;

		  __vr vrinP = _vel_vshf_vvvsl(vrin, vrin, VE_VSHUFFLE_YUZU, vl) ;

		  const float *pKerValue = pKernel + kernGroupOffset + ((k * inChannelGroup + c) * kernHeight + r) * kernWidth + s;

		  const uint64_t kerValue01 = _vel_pack_f32p(pKerValue,
							    pKerValue+      inChannelGroup * kernHeight * kernWidth) ;
		  const uint64_t kerValue23 = _vel_pack_f32p(pKerValue + 2 * inChannelGroup * kernHeight * kernWidth,
							    pKerValue + 3 * inChannelGroup * kernHeight * kernWidth) ;

		  vrsum01 = _vel_pvfmad_vvsvl(vrsum01, kerValue01, vrinP, vl) ;
		  vrsum23 = _vel_pvfmad_vvsvl(vrsum23, kerValue23, vrinP, vl) ;
		} // inChannel

	      } // kernWidth
	    } // kernHeight

	    _vel_vstu_vssl(vrsum01, 4, pOut+outIndex0, vl) ;
	    _vel_vstl_vssl(vrsum01, 4, pOut+outIndex1, vl) ;
	    _vel_vstu_vssl(vrsum23, 4, pOut+outIndex2, vl) ;
	    _vel_vstl_vssl(vrsum23, 4, pOut+outIndex3, vl) ;

	    outIndex0 += vl ;
	    outIndex1 += vl ;
	    outIndex2 += vl ;
	    outIndex3 += vl ;
	  } // outPixels

	  k+=4 ;
	}
	for ( ; k < outChannelGroup; k+=8) {
	  int64_t outIndex0 = outGroupOffset + (n * outChannel + k  ) * oPixels ;
	  int64_t outIndex1 = outGroupOffset + (n * outChannel + k+1) * oPixels ;
	  int64_t outIndex2 = outGroupOffset + (n * outChannel + k+2) * oPixels ;
	  int64_t outIndex3 = outGroupOffset + (n * outChannel + k+3) * oPixels ;
	  int64_t outIndex4 = outGroupOffset + (n * outChannel + k+4) * oPixels ;
	  int64_t outIndex5 = outGroupOffset + (n * outChannel + k+5) * oPixels ;
	  int64_t outIndex6 = outGroupOffset + (n * outChannel + k+6) * oPixels ;
	  int64_t outIndex7 = outGroupOffset + (n * outChannel + k+7) * oPixels ;

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

	    __vr vrsum01 = _vel_pvbrd_vsl(bias01, vl) ;
	    __vr vrsum23 = _vel_pvbrd_vsl(bias23, vl) ;
	    __vr vrsum45 = _vel_pvbrd_vsl(bias45, vl) ;
	    __vr vrsum67 = _vel_pvbrd_vsl(bias67, vl) ;

	    __vr vry   = _vel_vdivsl_vvsl(vridx, outWidth, vl) ;
	    __vr vrx   = _vel_vsubsl_vvvl(vridx, _vel_vmulul_vsvl(outWidth,vry, vl), vl) ;

	    for (int64_t r = 0; r < kernHeight; r++) {
	      for (int64_t s = 0; s < kernWidth; s++) {
		__vr vrh = _vel_vaddsl_vsvl(r-padHeight, vry, vl) ;
		__vr vrw = _vel_vaddsl_vsvl(s-padWidth,  vrx, vl) ;

		__vm256 vm0 =  _vel_vfmklge_mvl(vrh, vl) ;				// condition(0 <= h)
		__vm256 vm1 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(inHeight,vrh, vl), vl) ;	// condition(h < inHeight)
		__vm256 vm2 =  _vel_vfmklge_mvl(vrw, vl) ;				// condition(0 <= w)
		__vm256 vm3 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(inWidth,vrw, vl), vl) ;	// condition(w < inWidth)

		__vm256 vm01  = _vel_andm_mmm(vm0, vm1) ;
		__vm256 vm23  = _vel_andm_mmm(vm2, vm3) ;
		__vm256 vmall = _vel_andm_mmm(vm01,vm23) ;

		for (int64_t c = 0; c < inChannelGroup; c++) {
		  const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;

		  /* memory access errors mihgt be caused */
		  __vr vrin = _vel_vldu_vssl(4,&pInChannel[op+(r-padHeight)*inWidth+s-padWidth], vl) ;
		  vrin = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrin, vmall, vl) ;

		  __vr vrinP = _vel_vshf_vvvsl(vrin, vrin, VE_VSHUFFLE_YUZU, vl) ;

		  const float *pKerValue = pKernel + kernGroupOffset + ((k * inChannelGroup + c) * kernHeight + r) * kernWidth + s;

		  const uint64_t kerValue01 = _vel_pack_f32p(pKerValue,
							    pKerValue+      inChannelGroup * kernHeight * kernWidth) ;
		  const uint64_t kerValue23 = _vel_pack_f32p(pKerValue + 2 * inChannelGroup * kernHeight * kernWidth,
							    pKerValue + 3 * inChannelGroup * kernHeight * kernWidth) ;
		  const uint64_t kerValue45 = _vel_pack_f32p(pKerValue + 4 * inChannelGroup * kernHeight * kernWidth,
							    pKerValue + 5 * inChannelGroup * kernHeight * kernWidth) ;
		  const uint64_t kerValue67 = _vel_pack_f32p(pKerValue + 6 * inChannelGroup * kernHeight * kernWidth,
							    pKerValue + 7 * inChannelGroup * kernHeight * kernWidth) ;

		  vrsum01 = _vel_pvfmad_vvsvl(vrsum01, kerValue01, vrinP, vl) ;
		  vrsum23 = _vel_pvfmad_vvsvl(vrsum23, kerValue23, vrinP, vl) ;
		  vrsum45 = _vel_pvfmad_vvsvl(vrsum45, kerValue45, vrinP, vl) ;
		  vrsum67 = _vel_pvfmad_vvsvl(vrsum67, kerValue67, vrinP, vl) ;
		} // inChannel
	      } // kernWidth
	    } // kernHeight

	    _vel_vstu_vssl(vrsum01, 4, pOut+outIndex0, vl) ;
	    _vel_vstl_vssl(vrsum01, 4, pOut+outIndex1, vl) ;
	    _vel_vstu_vssl(vrsum23, 4, pOut+outIndex2, vl) ;
	    _vel_vstl_vssl(vrsum23, 4, pOut+outIndex3, vl) ;
	    _vel_vstu_vssl(vrsum45, 4, pOut+outIndex4, vl) ;
	    _vel_vstl_vssl(vrsum45, 4, pOut+outIndex5, vl) ;
	    _vel_vstu_vssl(vrsum67, 4, pOut+outIndex6, vl) ;
	    _vel_vstl_vssl(vrsum67, 4, pOut+outIndex7, vl) ;

	    outIndex0 += vl ;
	    outIndex1 += vl ;
	    outIndex2 += vl ;
	    outIndex3 += vl ;
	    outIndex4 += vl ;
	    outIndex5 += vl ;
	    outIndex6 += vl ;
	    outIndex7 += vl ;
	  } // outPixels
	} // outChannel
      } // group
    } // batch
  }

  return VEDNN_SUCCESS;
}
