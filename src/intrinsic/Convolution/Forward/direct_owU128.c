#include <stdint.h>
#include <stdlib.h>

#include "vednn.h"

#include "velintrin.h"
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

    const int64_t maxvl = nY * outWidth ;

    __vr vrseq = _vel_vseq_vl(maxvl) ;
    __vr vry  = _vel_vdivsl_vvsl(vrseq, outWidth, maxvl) ;
    __vr vrx  = _vel_vsubsl_vvvl(vrseq, _vel_vmulul_vsvl(outWidth,vry, maxvl), maxvl) ;

    __vr vri   = _vel_vaddsl_vsvl(-padHeight, _vel_vmulsl_vsvl(strideHeight, vry, maxvl), maxvl) ;
    __vr vrj   = _vel_vaddsl_vsvl(-padWidth,  _vel_vmulsl_vsvl(strideWidth,  vrx, maxvl), maxvl) ;

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

	    __vr vrsum = _vel_vbrds_vsl(0.0f, vl) ;

	    for (int64_t r = 0; r < kernHeight; r++) {
	      for (int64_t s = 0; s < kernWidth; s++) {
		__vr vrh = _vel_vaddsl_vsvl(r*dilationHeight+y*strideHeight, vri, vl) ;
		__vr vrw = _vel_vaddsl_vsvl(s*dilationWidth,                 vrj, vl) ;

		__vm256 vm0 =  _vel_vfmklge_mvl(vrh, vl) ;					// condition(0 <= h)
		__vm256 vm1 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(inHeight,vrh, vl), vl) ;	// condition(h < inHeight)
		__vm256 vm2 =  _vel_vfmklge_mvl(vrw, vl) ;					// condition(0 <= w)
		__vm256 vm3 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(inWidth,vrw, vl), vl) ;	// condition(w < inWidth)

		__vm256 vm01  = _vel_andm_mmm(vm0, vm1) ;
		__vm256 vm23  = _vel_andm_mmm(vm2, vm3) ;
		__vm256 vmall = _vel_andm_mmm(vm01,vm23) ;

		for (int64_t c = 0; c < inChannelGroup; c++) {
		  const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;
		  __vr vrpin = _vel_vsfa_vvssl(_vel_vaddul_vvvl(vrw, _vel_vmulul_vsvl(inWidth,vrh, vl), vl),
					       2,
					       (uint64_t)pInChannel, vl) ;

		  __vr vrin = _vel_vgtu_vvssml(vrpin, 0, 0, vmall, vl) ;
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
	  int64_t outIndex = outGroupOffset + (n * outChannel + k  ) * oPixels ;

	  for (int64_t y=0; y<outHeight; y+=nY) {
	    const int64_t vl = outWidth * (outHeight - y < nY ? outHeight - y : nY) ;
	    const int64_t op = y * outWidth ;

	    __vr vrsum01 = _vel_pvbrd_vsl(0UL, vl) ;

	    for (int64_t r = 0; r < kernHeight; r++) {
	      for (int64_t s = 0; s < kernWidth; s++) {
		__vr vrh = _vel_vaddsl_vsvl(r*dilationHeight+y*strideHeight, vri, vl) ;
		__vr vrw = _vel_vaddsl_vsvl(s*dilationWidth,                 vrj, vl) ;

		__vm256 vm0 =  _vel_vfmklge_mvl(vrh, vl) ;					// condition(0 <= h)
		__vm256 vm1 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(inHeight,vrh, vl), vl) ;	// condition(h < inHeight)
		__vm256 vm2 =  _vel_vfmklge_mvl(vrw, vl) ;					// condition(0 <= w)
		__vm256 vm3 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(inWidth,vrw, vl), vl) ;	// condition(w < inWidth)

		__vm256 vm01  = _vel_andm_mmm(vm0, vm1) ;
		__vm256 vm23  = _vel_andm_mmm(vm2, vm3) ;
		__vm256 vmall = _vel_andm_mmm(vm01,vm23) ;

		for (int64_t c = 0; c < inChannelGroup; c++) {
		  const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;
		  __vr vrpin = _vel_vsfa_vvssl(_vel_vaddul_vvvl(vrw, _vel_vmulul_vsvl(inWidth,vrh, vl), vl),
					       2,
					       (uint64_t)pInChannel, vl) ;

		  __vr vrin = _vel_vgtu_vvssml(vrpin, 0, 0, vmall, vl) ;
		  vrin = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrin, vmall, vl) ;

		  __vr vrinP = _vel_vshf_vvvsl(vrin, vrin, VE_VSHUFFLE_YUZU, vl) ;

		  const float *pKerValue = pKernel + kernGroupOffset + ((k * inChannelGroup + c) * kernHeight + r) * kernWidth + s;

		  const uint64_t kerValue01 = _vel_pack_f32p(pKerValue,
							     pKerValue+      inChannelGroup * kernHeight * kernWidth) ;

		  vrsum01 = _vel_pvfmad_vvsvl(vrsum01, kerValue01, vrinP, vl) ;
		} // inChannel
	      } // kernWidth
	    } // kernHeight

	    _vel_vstu_vssl(vrsum01, 4, pOut+outIndex, vl) ;
	    _vel_vstl_vssl(vrsum01, 4, pOut+outIndex+ 1*oPixels, vl) ;

	    outIndex += vl ;
	  } // outPixels

	  k+=2 ;
	}
	if ( ((outChannelGroup >> 2) & 0x01) == 1 ) {
	  int64_t outIndex = outGroupOffset + (n * outChannel + k  ) * oPixels ;

	  for (int64_t y=0; y<outHeight; y+=nY) {
	    const int64_t vl = outWidth * (outHeight - y < nY ? outHeight - y : nY) ;
	    const int64_t op = y * outWidth ;

	    __vr vrsum01 = _vel_pvbrd_vsl(0UL, vl) ;
	    __vr vrsum23 = _vel_pvbrd_vsl(0UL, vl) ;

	    for (int64_t r = 0; r < kernHeight; r++) {
	      for (int64_t s = 0; s < kernWidth; s++) {
		__vr vrh = _vel_vaddsl_vsvl(r*dilationHeight+y*strideHeight, vri, vl) ;
		__vr vrw = _vel_vaddsl_vsvl(s*dilationWidth,                 vrj, vl) ;

		__vm256 vm0 =  _vel_vfmklge_mvl(vrh, vl) ;					// condition(0 <= h)
		__vm256 vm1 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(inHeight,vrh, vl), vl) ;	// condition(h < inHeight)
		__vm256 vm2 =  _vel_vfmklge_mvl(vrw, vl) ;					// condition(0 <= w)
		__vm256 vm3 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(inWidth,vrw, vl), vl) ;	// condition(w < inWidth)

		__vm256 vm01  = _vel_andm_mmm(vm0, vm1) ;
		__vm256 vm23  = _vel_andm_mmm(vm2, vm3) ;
		__vm256 vmall = _vel_andm_mmm(vm01,vm23) ;

		for (int64_t c = 0; c < inChannelGroup; c++) {
		  const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;
		  __vr vrpin = _vel_vsfa_vvssl(_vel_vaddul_vvvl(vrw, _vel_vmulul_vsvl(inWidth,vrh, vl), vl),
					       2,
					       (uint64_t)pInChannel, vl) ;

		  __vr vrin = _vel_vgtu_vvssml(vrpin, 0, 0, vmall, vl) ;
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

	    _vel_vstu_vssl(vrsum01, 4, pOut+outIndex, vl) ;
	    _vel_vstl_vssl(vrsum01, 4, pOut+outIndex+ 1*oPixels, vl) ;
	    _vel_vstu_vssl(vrsum23, 4, pOut+outIndex+ 2*oPixels, vl) ;
	    _vel_vstl_vssl(vrsum23, 4, pOut+outIndex+ 3*oPixels, vl) ;

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

	    __vr vrsum01 = _vel_pvbrd_vsl(0UL, vl) ;
	    __vr vrsum23 = _vel_pvbrd_vsl(0UL, vl) ;
	    __vr vrsum45 = _vel_pvbrd_vsl(0UL, vl) ;
	    __vr vrsum67 = _vel_pvbrd_vsl(0UL, vl) ;

	    for (int64_t r = 0; r < kernHeight; r++) {
	      for (int64_t s = 0; s < kernWidth; s++) {
		__vr vrh = _vel_vaddsl_vsvl(r*dilationHeight+y*strideHeight, vri, vl) ;
		__vr vrw = _vel_vaddsl_vsvl(s*dilationWidth,                 vrj, vl) ;

		__vm256 vm0 =  _vel_vfmklge_mvl(vrh, vl) ;					// condition(0 <= h)
		__vm256 vm1 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(inHeight,vrh, vl), vl) ;	// condition(h < inHeight)
		__vm256 vm2 =  _vel_vfmklge_mvl(vrw, vl) ;					// condition(0 <= w)
		__vm256 vm3 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(inWidth,vrw, vl), vl) ;	// condition(w < inWidth)

		__vm256 vm01  = _vel_andm_mmm(vm0, vm1) ;
		__vm256 vm23  = _vel_andm_mmm(vm2, vm3) ;
		__vm256 vmall = _vel_andm_mmm(vm01,vm23) ;

		for (int64_t c = 0; c < inChannelGroup; c++) {
		  const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;
		  __vr vrpin = _vel_vsfa_vvssl(_vel_vaddul_vvvl(vrw, _vel_vmulul_vsvl(inWidth,vrh, vl), vl),
					       2,
		                               (uint64_t)pInChannel, vl) ;

		  __vr vrin = _vel_vgtu_vvssml(vrpin, 0, 0, vmall, vl) ;
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

	    _vel_vstu_vssl(vrsum01, 4, pOut+outIndex, vl) ;
	    _vel_vstl_vssl(vrsum01, 4, pOut+outIndex+ 1*oPixels, vl) ;
	    _vel_vstu_vssl(vrsum23, 4, pOut+outIndex+ 2*oPixels, vl) ;
	    _vel_vstl_vssl(vrsum23, 4, pOut+outIndex+ 3*oPixels, vl) ;
	    _vel_vstu_vssl(vrsum45, 4, pOut+outIndex+ 4*oPixels, vl) ;
	    _vel_vstl_vssl(vrsum45, 4, pOut+outIndex+ 5*oPixels, vl) ;
	    _vel_vstu_vssl(vrsum67, 4, pOut+outIndex+ 6*oPixels, vl) ;
	    _vel_vstl_vssl(vrsum67, 4, pOut+outIndex+ 7*oPixels, vl) ;

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

	    __vr vrsum01 = _vel_pvbrd_vsl(0UL, vl) ;
	    __vr vrsum23 = _vel_pvbrd_vsl(0UL, vl) ;
	    __vr vrsum45 = _vel_pvbrd_vsl(0UL, vl) ;
	    __vr vrsum67 = _vel_pvbrd_vsl(0UL, vl) ;
	    __vr vrsum89 = _vel_pvbrd_vsl(0UL, vl) ;
	    __vr vrsumAB = _vel_pvbrd_vsl(0UL, vl) ;
	    __vr vrsumCD = _vel_pvbrd_vsl(0UL, vl) ;
	    __vr vrsumEF = _vel_pvbrd_vsl(0UL, vl) ;

	    for (int64_t r = 0; r < kernHeight; r++) {
	      for (int64_t s = 0; s < kernWidth; s++) {
		__vr vrh = _vel_vaddsl_vsvl(r*dilationHeight+y*strideHeight, vri, vl) ;
		__vr vrw = _vel_vaddsl_vsvl(s*dilationWidth,                 vrj, vl) ;

		__vm256 vm0 =  _vel_vfmklge_mvl(vrh, vl) ;					// condition(0 <= h)
		__vm256 vm1 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(inHeight,vrh, vl), vl) ;	// condition(h < inHeight)
		__vm256 vm2 =  _vel_vfmklge_mvl(vrw, vl) ;					// condition(0 <= w)
		__vm256 vm3 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(inWidth,vrw, vl), vl) ;	// condition(w < inWidth)

		__vm256 vm01  = _vel_andm_mmm(vm0, vm1) ;
		__vm256 vm23  = _vel_andm_mmm(vm2, vm3) ;
		__vm256 vmall = _vel_andm_mmm(vm01,vm23) ;

		for (int64_t c = 0; c < inChannelGroup; c++) {
		  const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;
		  __vr vrpin = _vel_vsfa_vvssl(_vel_vaddul_vvvl(vrw, _vel_vmulul_vsvl(inWidth,vrh, vl), vl),
					       2,
		                               (uint64_t)pInChannel, vl) ;

		  __vr vrin = _vel_vgtu_vvssml(vrpin, 0, 0, vmall, vl) ;
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
		  const uint64_t kerValue89 = _vel_pack_f32p(pKerValue + 8 * inChannelGroup * kernHeight * kernWidth,
							     pKerValue + 9 * inChannelGroup * kernHeight * kernWidth) ;
		  const uint64_t kerValueAB = _vel_pack_f32p(pKerValue +10 * inChannelGroup * kernHeight * kernWidth,
							     pKerValue +11 * inChannelGroup * kernHeight * kernWidth) ;
		  const uint64_t kerValueCD = _vel_pack_f32p(pKerValue +12 * inChannelGroup * kernHeight * kernWidth,
							     pKerValue +13 * inChannelGroup * kernHeight * kernWidth) ;
		  const uint64_t kerValueEF = _vel_pack_f32p(pKerValue +14 * inChannelGroup * kernHeight * kernWidth,
							     pKerValue +15 * inChannelGroup * kernHeight * kernWidth) ;

		  vrsum01 = _vel_pvfmad_vvsvl(vrsum01, kerValue01, vrinP, vl) ;
		  vrsum23 = _vel_pvfmad_vvsvl(vrsum23, kerValue23, vrinP, vl) ;
		  vrsum45 = _vel_pvfmad_vvsvl(vrsum45, kerValue45, vrinP, vl) ;
		  vrsum67 = _vel_pvfmad_vvsvl(vrsum67, kerValue67, vrinP, vl) ;
		  vrsum89 = _vel_pvfmad_vvsvl(vrsum89, kerValue89, vrinP, vl) ;
		  vrsumAB = _vel_pvfmad_vvsvl(vrsumAB, kerValueAB, vrinP, vl) ;
		  vrsumCD = _vel_pvfmad_vvsvl(vrsumCD, kerValueCD, vrinP, vl) ;
		  vrsumEF = _vel_pvfmad_vvsvl(vrsumEF, kerValueEF, vrinP, vl) ;
		} // inChannel
	      } // kernWidth
	    } // kernHeight

	    _vel_vstu_vssl(vrsum01, 4, pOut+outIndex, vl) ;
	    _vel_vstl_vssl(vrsum01, 4, pOut+outIndex+ 1*oPixels, vl) ;
	    _vel_vstu_vssl(vrsum23, 4, pOut+outIndex+ 2*oPixels, vl) ;
	    _vel_vstl_vssl(vrsum23, 4, pOut+outIndex+ 3*oPixels, vl) ;
	    _vel_vstu_vssl(vrsum45, 4, pOut+outIndex+ 4*oPixels, vl) ;
	    _vel_vstl_vssl(vrsum45, 4, pOut+outIndex+ 5*oPixels, vl) ;
	    _vel_vstu_vssl(vrsum67, 4, pOut+outIndex+ 6*oPixels, vl) ;
	    _vel_vstl_vssl(vrsum67, 4, pOut+outIndex+ 7*oPixels, vl) ;
	    _vel_vstu_vssl(vrsum89, 4, pOut+outIndex+ 8*oPixels, vl) ;
	    _vel_vstl_vssl(vrsum89, 4, pOut+outIndex+ 9*oPixels, vl) ;
	    _vel_vstu_vssl(vrsumAB, 4, pOut+outIndex+10*oPixels, vl) ;
	    _vel_vstl_vssl(vrsumAB, 4, pOut+outIndex+11*oPixels, vl) ;
	    _vel_vstu_vssl(vrsumCD, 4, pOut+outIndex+12*oPixels, vl) ;
	    _vel_vstl_vssl(vrsumCD, 4, pOut+outIndex+13*oPixels, vl) ;
	    _vel_vstu_vssl(vrsumEF, 4, pOut+outIndex+14*oPixels, vl) ;
	    _vel_vstl_vssl(vrsumEF, 4, pOut+outIndex+15*oPixels, vl) ;

	    outIndex += vl ;
	  } // outPixels
	} // outChannel
      } // group
    } // batch
  }

  return VEDNN_SUCCESS;
}

