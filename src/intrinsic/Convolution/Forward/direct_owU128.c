#include <stdint.h>
#include <stdlib.h>

#include "vednn.h"

#include "veintrin.h"
#define VLEN	(256)

vednnError_t
vednnConvolutionForward_direct_owU128(
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
  const int64_t outWidth   = pParamOut->width;
  const int64_t outHeight  = pParamOut->height;
  const int64_t kernWidth  = pParamKernel->width;
  const int64_t kernHeight = pParamKernel->height;

  const int64_t group          = pParamConv->group;
  const int64_t strideWidth    = pParamConv->strideWidth;;
  const int64_t strideHeight   = pParamConv->strideHeight;
  const int64_t padWidth       = pParamConv->padWidth;
  const int64_t padHeight      = pParamConv->padHeight;
  const int64_t dilationWidth  = pParamConv->dilationWidth;
  const int64_t dilationHeight = pParamConv->dilationHeight;

  const int64_t inChannelGroup  = inChannel  / group;   // equal to pDataKernel->inChannel
  const int64_t outChannelGroup = outChannel / group;   // equal to pDataKernel->outChannel

  const float * restrict pIn     = pDataIn;
  const float * restrict pKernel = pDataKernel;
  float * restrict const pOut    = pDataOut;

  const int oPixels= outHeight*outWidth ;

  {
    const int64_t nY = VLEN / outWidth ;

    _ve_lvl(nY*outWidth) ;

    __vr vrseq = _ve_vseq_v() ;
    __vr vry  = _ve_vdivsl_vvs(vrseq, outWidth) ;
    __vr vrx  = _ve_vsubsl_vvv(vrseq, _ve_vmulul_vsv(outWidth,vry)) ;

    __vr vri   = _ve_vaddsl_vsv(-padHeight, _ve_vmulsl_vsv(strideHeight, vry)) ;
    __vr vrj   = _ve_vaddsl_vsv(-padWidth,  _ve_vmulsl_vsv(strideWidth,  vrx)) ;

    for (int64_t n = 0; n < batch; n++) {
      for (int64_t g = 0; g < group; g++) {
	const int64_t inGroupOffset   = g * inChannelGroup * inHeight * inWidth;
	const int64_t outGroupOffset  = g * outChannelGroup * outHeight * outWidth;
	const int64_t kernGroupOffset = g * outChannelGroup * inChannelGroup * kernHeight * kernWidth;

	int k = 0 ;
	if ( (outChannelGroup & 0x01) == 1 ) {
	  int64_t outIndex = outGroupOffset + (n * outChannel + k) * oPixels ;

	  for (int64_t y=0; y<outHeight; y+=nY) {
	    const int64_t vl = outWidth * (outHeight - y < nY ? outHeight - y : nY) ;
	    const int64_t op = y * outWidth ;

	    _ve_lvl(vl) ;
	    __vr vrsum = _ve_vbrdu_vs_f32(0.0f) ;

	    for (int64_t r = 0; r < kernHeight; r++) {
	      for (int64_t s = 0; s < kernWidth; s++) {
		__vr vrh = _ve_vaddsl_vsv(r*dilationHeight+y*strideHeight, vri) ;
		__vr vrw = _ve_vaddsl_vsv(s*dilationWidth,                 vrj) ;

		__vm256 vm0 = _ve_vfmkl_mcv(VECC_GE, vrh) ;				// condition(0 <= h)
		__vm256 vm1 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(inHeight,vrh)) ;	// condition(h < inHeight)
		__vm256 vm2 = _ve_vfmkl_mcv(VECC_GE, vrw) ;				// condition(0 <= w)
		__vm256 vm3 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(inWidth,vrw)) ;	// condition(w < inWidth)

		__vm256 vm01  = _ve_andm_mmm(vm0, vm1) ;
		__vm256 vm23  = _ve_andm_mmm(vm2, vm3) ;
		__vm256 vmall = _ve_andm_mmm(vm01,vm23) ;

		for (int64_t c = 0; c < inChannelGroup; c++) {
		  const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;
		  __vr vrpin = _ve_vsfa_vvss(_ve_vaddul_vvv(vrw, _ve_vmulul_vsv(inWidth,vrh)),
					     2,
					     (uint64_t)pInChannel) ;

		  __vr vrin = _ve_vgtu_vvm(vrpin, vmall) ;
		  vrin = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), vrin, vmall) ;

		  const float *pKerValue = pKernel + kernGroupOffset + ((k * inChannelGroup + c) * kernHeight + r) * kernWidth + s;

		  vrsum = _ve_vfmads_vvsv(vrsum, *pKerValue, vrin) ;

	        } // kernWidth
	      } // kernHeight
	    } // inChannel

	    _ve_vstu_vss(vrsum, 4, pOut+outIndex) ;

	    outIndex += vl ;
	  } // outPixels
	  k++ ;
	}
	if ( ((outChannelGroup >> 1) & 0x01) == 1 ) {
	  int64_t outIndex = outGroupOffset + (n * outChannel + k  ) * oPixels ;

	  for (int64_t y=0; y<outHeight; y+=nY) {
	    const int64_t vl = outWidth * (outHeight - y < nY ? outHeight - y : nY) ;
	    const int64_t op = y * outWidth ;

	    _ve_lvl(vl) ;

	    __vr vrsum01 = _ve_pvbrd_vs_i64(0UL) ;

	    for (int64_t r = 0; r < kernHeight; r++) {
	      for (int64_t s = 0; s < kernWidth; s++) {
		__vr vrh = _ve_vaddsl_vsv(r*dilationHeight+y*strideHeight, vri) ;
		__vr vrw = _ve_vaddsl_vsv(s*dilationWidth,                 vrj) ;

		__vm256 vm0 = _ve_vfmkl_mcv(VECC_GE, vrh) ;				// condition(0 <= h)
		__vm256 vm1 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(inHeight,vrh)) ;	// condition(h < inHeight)
		__vm256 vm2 = _ve_vfmkl_mcv(VECC_GE, vrw) ;				// condition(0 <= w)
		__vm256 vm3 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(inWidth,vrw)) ;	// condition(w < inWidth)

		__vm256 vm01  = _ve_andm_mmm(vm0, vm1) ;
		__vm256 vm23  = _ve_andm_mmm(vm2, vm3) ;
		__vm256 vmall = _ve_andm_mmm(vm01,vm23) ;

		for (int64_t c = 0; c < inChannelGroup; c++) {
		  const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;
		  __vr vrpin = _ve_vsfa_vvss(_ve_vaddul_vvv(vrw, _ve_vmulul_vsv(inWidth,vrh)),
					     2,
					     (uint64_t)pInChannel) ;

		  __vr vrin = _ve_vgtu_vvm(vrpin, vmall) ;
		  vrin = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), vrin, vmall) ;

		  __vr vrinP = _ve_vshf_vvvs(vrin, vrin, VE_VSHUFFLE_YUZU) ;

		  const float *pKerValue = pKernel + kernGroupOffset + ((k * inChannelGroup + c) * kernHeight + r) * kernWidth + s;

		  const uint64_t kerValue01 = _ve_pack_f32p(pKerValue,
							    pKerValue+      inChannelGroup * kernHeight * kernWidth) ;

		  vrsum01 = _ve_pvfmad_vvsv(vrsum01, kerValue01, vrinP) ;
		} // inChannel
	      } // kernWidth
	    } // kernHeight

	    _ve_vstu_vss(vrsum01, 4, pOut+outIndex) ;
	    _ve_vstl_vss(vrsum01, 4, pOut+outIndex+ 1*oPixels) ;

	    outIndex += vl ;
	  } // outPixels

	  k+=2 ;
	}
	if ( ((outChannelGroup >> 2) & 0x01) == 1 ) {
	  int64_t outIndex = outGroupOffset + (n * outChannel + k  ) * oPixels ;

	  for (int64_t y=0; y<outHeight; y+=nY) {
	    const int64_t vl = outWidth * (outHeight - y < nY ? outHeight - y : nY) ;
	    const int64_t op = y * outWidth ;

	    _ve_lvl(vl) ;

	    __vr vrsum01 = _ve_pvbrd_vs_i64(0UL) ;
	    __vr vrsum23 = _ve_pvbrd_vs_i64(0UL) ;

	    for (int64_t r = 0; r < kernHeight; r++) {
	      for (int64_t s = 0; s < kernWidth; s++) {
		__vr vrh = _ve_vaddsl_vsv(r*dilationHeight+y*strideHeight, vri) ;
		__vr vrw = _ve_vaddsl_vsv(s*dilationWidth,                 vrj) ;

		__vm256 vm0 = _ve_vfmkl_mcv(VECC_GE, vrh) ;				// condition(0 <= h)
		__vm256 vm1 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(inHeight,vrh)) ;	// condition(h < inHeight)
		__vm256 vm2 = _ve_vfmkl_mcv(VECC_GE, vrw) ;				// condition(0 <= w)
		__vm256 vm3 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(inWidth,vrw)) ;	// condition(w < inWidth)

		__vm256 vm01  = _ve_andm_mmm(vm0, vm1) ;
		__vm256 vm23  = _ve_andm_mmm(vm2, vm3) ;
		__vm256 vmall = _ve_andm_mmm(vm01,vm23) ;

		for (int64_t c = 0; c < inChannelGroup; c++) {
		  const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;
		  __vr vrpin = _ve_vsfa_vvss(_ve_vaddul_vvv(vrw, _ve_vmulul_vsv(inWidth,vrh)),
					     2,
					     (uint64_t)pInChannel) ;

		  __vr vrin = _ve_vgtu_vvm(vrpin, vmall) ;
		  vrin = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), vrin, vmall) ;

		  __vr vrinP = _ve_vshf_vvvs(vrin, vrin, VE_VSHUFFLE_YUZU) ;

		  const float *pKerValue = pKernel + kernGroupOffset + ((k * inChannelGroup + c) * kernHeight + r) * kernWidth + s;

		  const uint64_t kerValue01 = _ve_pack_f32p(pKerValue,
							    pKerValue+      inChannelGroup * kernHeight * kernWidth) ;
		  const uint64_t kerValue23 = _ve_pack_f32p(pKerValue + 2 * inChannelGroup * kernHeight * kernWidth,
							    pKerValue + 3 * inChannelGroup * kernHeight * kernWidth) ;

		  vrsum01 = _ve_pvfmad_vvsv(vrsum01, kerValue01, vrinP) ;
		  vrsum23 = _ve_pvfmad_vvsv(vrsum23, kerValue23, vrinP) ;
		} // inChannel

	      } // kernWidth
	    } // kernHeight

	    _ve_vstu_vss(vrsum01, 4, pOut+outIndex) ;
	    _ve_vstl_vss(vrsum01, 4, pOut+outIndex+ 1*oPixels) ;
	    _ve_vstu_vss(vrsum23, 4, pOut+outIndex+ 2*oPixels) ;
	    _ve_vstl_vss(vrsum23, 4, pOut+outIndex+ 3*oPixels) ;

	    outIndex += vl ;
	  } // outPixels

	  k+=4 ;
	}
	if ( ((outChannelGroup >> 3) & 0x01) == 1 ) {
	  int64_t outIndex = outGroupOffset + (n * outChannel + k  ) * oPixels ;

	  for (int64_t y=0; y<outHeight; y+=nY)
	  {
	    const int64_t vl = outWidth * (outHeight - y < nY ? outHeight - y : nY) ;
	    const int64_t op = y * outWidth ;

	    _ve_lvl(vl) ;

	    __vr vrsum01 = _ve_pvbrd_vs_i64(0UL) ;
	    __vr vrsum23 = _ve_pvbrd_vs_i64(0UL) ;
	    __vr vrsum45 = _ve_pvbrd_vs_i64(0UL) ;
	    __vr vrsum67 = _ve_pvbrd_vs_i64(0UL) ;

	    for (int64_t r = 0; r < kernHeight; r++) {
	      for (int64_t s = 0; s < kernWidth; s++) {
		__vr vrh = _ve_vaddsl_vsv(r*dilationHeight+y*strideHeight, vri) ;
		__vr vrw = _ve_vaddsl_vsv(s*dilationWidth,                 vrj) ;

		__vm256 vm0 = _ve_vfmkl_mcv(VECC_GE, vrh) ;				// condition(0 <= h)
		__vm256 vm1 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(inHeight,vrh)) ;	// condition(h < inHeight)
		__vm256 vm2 = _ve_vfmkl_mcv(VECC_GE, vrw) ;				// condition(0 <= w)
		__vm256 vm3 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(inWidth,vrw)) ;	// condition(w < inWidth)

		__vm256 vm01  = _ve_andm_mmm(vm0, vm1) ;
		__vm256 vm23  = _ve_andm_mmm(vm2, vm3) ;
		__vm256 vmall = _ve_andm_mmm(vm01,vm23) ;

		for (int64_t c = 0; c < inChannelGroup; c++) {
		  const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;
		  __vr vrpin = _ve_vsfa_vvss(_ve_vaddul_vvv(vrw, _ve_vmulul_vsv(inWidth,vrh)),
					     2,
		                             (uint64_t)pInChannel) ;

		  __vr vrin = _ve_vgtu_vvm(vrpin, vmall) ;
		  vrin = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), vrin, vmall) ;

		  __vr vrinP = _ve_vshf_vvvs(vrin, vrin, VE_VSHUFFLE_YUZU) ;

		  const float *pKerValue = pKernel + kernGroupOffset + ((k * inChannelGroup + c) * kernHeight + r) * kernWidth + s;

		  const uint64_t kerValue01 = _ve_pack_f32p(pKerValue,
							    pKerValue+      inChannelGroup * kernHeight * kernWidth) ;
		  const uint64_t kerValue23 = _ve_pack_f32p(pKerValue + 2 * inChannelGroup * kernHeight * kernWidth,
							    pKerValue + 3 * inChannelGroup * kernHeight * kernWidth) ;
		  const uint64_t kerValue45 = _ve_pack_f32p(pKerValue + 4 * inChannelGroup * kernHeight * kernWidth,
							    pKerValue + 5 * inChannelGroup * kernHeight * kernWidth) ;
		  const uint64_t kerValue67 = _ve_pack_f32p(pKerValue + 6 * inChannelGroup * kernHeight * kernWidth,
							    pKerValue + 7 * inChannelGroup * kernHeight * kernWidth) ;

		  vrsum01 = _ve_pvfmad_vvsv(vrsum01, kerValue01, vrinP) ;
		  vrsum23 = _ve_pvfmad_vvsv(vrsum23, kerValue23, vrinP) ;
		  vrsum45 = _ve_pvfmad_vvsv(vrsum45, kerValue45, vrinP) ;
		  vrsum67 = _ve_pvfmad_vvsv(vrsum67, kerValue67, vrinP) ;
		} // inChannel
	      } // kernWidth
	    } // kernHeight

	    _ve_vstu_vss(vrsum01, 4, pOut+outIndex) ;
	    _ve_vstl_vss(vrsum01, 4, pOut+outIndex+ 1*oPixels) ;
	    _ve_vstu_vss(vrsum23, 4, pOut+outIndex+ 2*oPixels) ;
	    _ve_vstl_vss(vrsum23, 4, pOut+outIndex+ 3*oPixels) ;
	    _ve_vstu_vss(vrsum45, 4, pOut+outIndex+ 4*oPixels) ;
	    _ve_vstl_vss(vrsum45, 4, pOut+outIndex+ 5*oPixels) ;
	    _ve_vstu_vss(vrsum67, 4, pOut+outIndex+ 6*oPixels) ;
	    _ve_vstl_vss(vrsum67, 4, pOut+outIndex+ 7*oPixels) ;

	    outIndex += vl ;
	  } // outPixels

	  k+=8 ;
	}
	for ( ; k < outChannelGroup; k+=16) {
	  int64_t outIndex = outGroupOffset + (n * outChannel + k  ) * oPixels ;

	  for (int64_t y=0; y<outHeight; y+=nY)
	  {
	    const int64_t vl = outWidth * (outHeight - y < nY ? outHeight - y : nY) ;
	    const int64_t op = y * outWidth ;

	    _ve_lvl(vl) ;

	    __vr vrsum01 = _ve_pvbrd_vs_i64(0UL) ;
	    __vr vrsum23 = _ve_pvbrd_vs_i64(0UL) ;
	    __vr vrsum45 = _ve_pvbrd_vs_i64(0UL) ;
	    __vr vrsum67 = _ve_pvbrd_vs_i64(0UL) ;
	    __vr vrsum89 = _ve_pvbrd_vs_i64(0UL) ;
	    __vr vrsumAB = _ve_pvbrd_vs_i64(0UL) ;
	    __vr vrsumCD = _ve_pvbrd_vs_i64(0UL) ;
	    __vr vrsumEF = _ve_pvbrd_vs_i64(0UL) ;

	    for (int64_t r = 0; r < kernHeight; r++) {
	      for (int64_t s = 0; s < kernWidth; s++) {
		__vr vrh = _ve_vaddsl_vsv(r*dilationHeight+y*strideHeight, vri) ;
		__vr vrw = _ve_vaddsl_vsv(s*dilationWidth,                 vrj) ;

		__vm256 vm0 = _ve_vfmkl_mcv(VECC_GE, vrh) ;				// condition(0 <= h)
		__vm256 vm1 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(inHeight,vrh)) ;	// condition(h < inHeight)
		__vm256 vm2 = _ve_vfmkl_mcv(VECC_GE, vrw) ;				// condition(0 <= w)
		__vm256 vm3 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(inWidth,vrw)) ;	// condition(w < inWidth)

		__vm256 vm01  = _ve_andm_mmm(vm0, vm1) ;
		__vm256 vm23  = _ve_andm_mmm(vm2, vm3) ;
		__vm256 vmall = _ve_andm_mmm(vm01,vm23) ;

		for (int64_t c = 0; c < inChannelGroup; c++) {
		  const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;
		  __vr vrpin = _ve_vsfa_vvss(_ve_vaddul_vvv(vrw, _ve_vmulul_vsv(inWidth,vrh)),
					     2,
		                             (uint64_t)pInChannel) ;

		  __vr vrin = _ve_vgtu_vvm(vrpin, vmall) ;
		  vrin = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), vrin, vmall) ;

		  __vr vrinP = _ve_vshf_vvvs(vrin, vrin, VE_VSHUFFLE_YUZU) ;

		  const float *pKerValue = pKernel + kernGroupOffset + ((k * inChannelGroup + c) * kernHeight + r) * kernWidth + s;

		  const uint64_t kerValue01 = _ve_pack_f32p(pKerValue,
							    pKerValue+      inChannelGroup * kernHeight * kernWidth) ;
		  const uint64_t kerValue23 = _ve_pack_f32p(pKerValue + 2 * inChannelGroup * kernHeight * kernWidth,
							    pKerValue + 3 * inChannelGroup * kernHeight * kernWidth) ;
		  const uint64_t kerValue45 = _ve_pack_f32p(pKerValue + 4 * inChannelGroup * kernHeight * kernWidth,
							    pKerValue + 5 * inChannelGroup * kernHeight * kernWidth) ;
		  const uint64_t kerValue67 = _ve_pack_f32p(pKerValue + 6 * inChannelGroup * kernHeight * kernWidth,
							    pKerValue + 7 * inChannelGroup * kernHeight * kernWidth) ;
		  const uint64_t kerValue89 = _ve_pack_f32p(pKerValue + 8 * inChannelGroup * kernHeight * kernWidth,
							    pKerValue + 9 * inChannelGroup * kernHeight * kernWidth) ;
		  const uint64_t kerValueAB = _ve_pack_f32p(pKerValue +10 * inChannelGroup * kernHeight * kernWidth,
							    pKerValue +11 * inChannelGroup * kernHeight * kernWidth) ;
		  const uint64_t kerValueCD = _ve_pack_f32p(pKerValue +12 * inChannelGroup * kernHeight * kernWidth,
							    pKerValue +13 * inChannelGroup * kernHeight * kernWidth) ;
		  const uint64_t kerValueEF = _ve_pack_f32p(pKerValue +14 * inChannelGroup * kernHeight * kernWidth,
							    pKerValue +15 * inChannelGroup * kernHeight * kernWidth) ;

		  vrsum01 = _ve_pvfmad_vvsv(vrsum01, kerValue01, vrinP) ;
		  vrsum23 = _ve_pvfmad_vvsv(vrsum23, kerValue23, vrinP) ;
		  vrsum45 = _ve_pvfmad_vvsv(vrsum45, kerValue45, vrinP) ;
		  vrsum67 = _ve_pvfmad_vvsv(vrsum67, kerValue67, vrinP) ;
		  vrsum89 = _ve_pvfmad_vvsv(vrsum89, kerValue89, vrinP) ;
		  vrsumAB = _ve_pvfmad_vvsv(vrsumAB, kerValueAB, vrinP) ;
		  vrsumCD = _ve_pvfmad_vvsv(vrsumCD, kerValueCD, vrinP) ;
		  vrsumEF = _ve_pvfmad_vvsv(vrsumEF, kerValueEF, vrinP) ;
		} // inChannel
	      } // kernWidth
	    } // kernHeight

	    _ve_vstu_vss(vrsum01, 4, pOut+outIndex) ;
	    _ve_vstl_vss(vrsum01, 4, pOut+outIndex+ 1*oPixels) ;
	    _ve_vstu_vss(vrsum23, 4, pOut+outIndex+ 2*oPixels) ;
	    _ve_vstl_vss(vrsum23, 4, pOut+outIndex+ 3*oPixels) ;
	    _ve_vstu_vss(vrsum45, 4, pOut+outIndex+ 4*oPixels) ;
	    _ve_vstl_vss(vrsum45, 4, pOut+outIndex+ 5*oPixels) ;
	    _ve_vstu_vss(vrsum67, 4, pOut+outIndex+ 6*oPixels) ;
	    _ve_vstl_vss(vrsum67, 4, pOut+outIndex+ 7*oPixels) ;
	    _ve_vstu_vss(vrsum89, 4, pOut+outIndex+ 8*oPixels) ;
	    _ve_vstl_vss(vrsum89, 4, pOut+outIndex+ 9*oPixels) ;
	    _ve_vstu_vss(vrsumAB, 4, pOut+outIndex+10*oPixels) ;
	    _ve_vstl_vss(vrsumAB, 4, pOut+outIndex+11*oPixels) ;
	    _ve_vstu_vss(vrsumCD, 4, pOut+outIndex+12*oPixels) ;
	    _ve_vstl_vss(vrsumCD, 4, pOut+outIndex+13*oPixels) ;
	    _ve_vstu_vss(vrsumEF, 4, pOut+outIndex+14*oPixels) ;
	    _ve_vstl_vss(vrsumEF, 4, pOut+outIndex+15*oPixels) ;

	    outIndex += vl ;
	  } // outPixels
	} // outChannel
      } // group
    } // batch
  }

  return VEDNN_SUCCESS;
}

