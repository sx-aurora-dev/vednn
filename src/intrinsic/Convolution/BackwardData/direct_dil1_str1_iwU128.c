#include <stdint.h>
#include <stdlib.h>

#include "vednn.h"

#include "velintrin.h"
#define VLEN	(256)

vednnError_t
vednnConvolutionBackwardData_direct_dil1_str1_iwU128(
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
  const int64_t kernWidth   = pParamKernel->width;
  const int64_t kernHeight  = pParamKernel->height;

  const int64_t group          = pParamConv->group;
//  const int64_t strideWidth    = pParamConv->strideWidth;		// must be 1
//  const int64_t strideHeight   = pParamConv->strideHeight;		// must be 1
  const int64_t padWidth       = pParamConv->padWidth;
  const int64_t padHeight      = pParamConv->padHeight;
//  const int64_t dilationWidth  = pParamConv->dilationWidth;		// must be 1
//  const int64_t dilationHeight = pParamConv->dilationHeight;		// must be 1

  const int64_t gOutChannelGroup = gOutChannel  / group;
  const int64_t gInChannelGroup  = gInChannel / group;

  const float * restrict pGOut   = pDataGradOut;
  const float * restrict pKernel = pDataKernel;
  float * restrict const pGIn   = pDataGradIn;

  const int gInPixels= gInHeight*gInWidth ;

  /* intrinsic version 1 */
  {
    const int64_t nH = VLEN / gInWidth ;

    __vr vrseq = _vel_vseq_vl(nH*gInWidth) ;
    __vr vrh  = _vel_vdivsl_vvsl(vrseq, gInWidth, nH*gInWidth) ;
    __vr vrw  = _vel_vsubsl_vvvl(vrseq, _vel_vmulul_vsvl(gInWidth,vrh, nH*gInWidth), nH*gInWidth) ;

    for (int64_t n=0; n<batch; n++) {
      for (int64_t g = 0; g < group; g++) {

	int64_t gInGroupOffset  = g * gInChannelGroup * gInHeight * gInWidth;
	int64_t gOutGroupOffset = g * gOutChannelGroup * gOutHeight * gOutWidth;
	int64_t kernGroupOffset = g * gOutChannelGroup * gInChannelGroup * kernHeight * kernWidth;

	int64_t k=0;
	if( (gInChannelGroup & 0x01 ) == 1 ) {

	  for (int64_t h=0; h<gInHeight; h+=nH) {
	    const int64_t vl = gInWidth * (gInHeight - h < nH ? gInHeight - h : nH) ;
	    const int64_t gip = h * gInWidth ;

	    const int64_t gInIndex = gInGroupOffset + ((n * gInChannel + k) * gInHeight ) * gInWidth + gip ;

	    __vr vrsum = _vel_vbrds_vsl(0.0f, vl) ;

	    for (int64_t r=0; r<kernHeight; r++) {
	      __vr vry = _vel_vaddsl_vsvl(padHeight-r+h, vrh, vl) ;

	      __vm256 vmy1 =  _vel_vfmklge_mvl(vry, vl) ;
	      __vm256 vmy2 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutHeight,vry, vl), vl) ;

	      __vm256 vmy = _vel_andm_mmm(vmy1, vmy2) ;

	      for (int64_t s=0; s<kernWidth; s++) {
		__vr vrx = _vel_vaddsl_vsvl(padWidth-s, vrw, vl) ;

		__vm256 vmx1 =  _vel_vfmklge_mvl(vrx, vl) ;
		__vm256 vmx2 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutWidth,vrx, vl), vl) ;

		__vm256 vmx = _vel_andm_mmm(vmx1, vmx2) ;

		__vm256 vmall = _vel_andm_mmm(vmy,vmx) ;

		for (int64_t c=0; c<gOutChannelGroup; c++) {
		  int64_t gOutIndex    = gOutGroupOffset + ((n * gOutChannel + c) * gOutHeight) * gOutWidth ;
		  __vr vrgout_ptr = _vel_vsfa_vvssl(_vel_vaddsl_vvvl(_vel_vmulsl_vsvl(gOutWidth, vry, vl), vrx, vl),
						2,
						(unsigned long)(pGOut+gOutIndex), vl) ;

		  __vr vrgout = _vel_vgtu_vvssml(vrgout_ptr, 0, 0, vmall, vl) ;
		  vrgout = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrgout, vmall, vl) ;

		  int64_t kernelIndex = kernGroupOffset + ((c * gInChannelGroup + k) * kernHeight + r) * kernWidth + s;

		  vrsum = _vel_vfmads_vvsvl(vrsum, pKernel[kernelIndex], vrgout, vl) ;
		} // gOutChannel

	      } // kernWidth
	    } // kernHeight

	    _vel_vstu_vssl(vrsum, 4, pGIn+gInIndex, vl) ;

	  } // gOutPixels

	  k++ ;
	}
	if( ((gInChannelGroup>>1) & 0x01 ) == 1 ) {

	  for (int64_t h=0; h<gInHeight; h+=nH) {
	    const int64_t vl = gInWidth * (gInHeight - h < nH ? gInHeight - h : nH) ;
	    const int64_t gip = h * gInWidth ;

	    const int64_t gInIndex = gInGroupOffset + ((n * gInChannel + k) * gInHeight ) * gInWidth + gip ;

	    __vr vrsum01 = _vel_pvbrd_vsl(0UL, vl) ;

	    for (int64_t r=0; r<kernHeight; r++) {
	      __vr vry = _vel_vaddsl_vsvl(padHeight-r+h, vrh, vl) ;

	      __vm256 vmy1 =  _vel_vfmklge_mvl(vry, vl) ;
	      __vm256 vmy2 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutHeight,vry, vl), vl) ;

	      __vm256 vmy = _vel_andm_mmm(vmy1, vmy2) ;

	      for (int64_t s=0; s<kernWidth; s++) {
		__vr vrx = _vel_vaddsl_vsvl(padWidth-s, vrw, vl) ;

		__vm256 vmx1 =  _vel_vfmklge_mvl(vrx, vl) ;
		__vm256 vmx2 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutWidth,vrx, vl), vl) ;

		__vm256 vmx = _vel_andm_mmm(vmx1, vmx2) ;

		__vm256 vmall = _vel_andm_mmm(vmy,vmx) ;

		for (int64_t c=0; c<gOutChannelGroup; c++) {
		  int64_t gOutIndex    = gOutGroupOffset + ((n * gOutChannel + c) * gOutHeight) * gOutWidth ;
		  __vr vrgout_ptr = _vel_vsfa_vvssl(_vel_vaddsl_vvvl(_vel_vmulsl_vsvl(gOutWidth, vry, vl), vrx, vl),
						2,
						(unsigned long)(pGOut+gOutIndex), vl) ;

		  __vr vrgout = _vel_vgtu_vvssml(vrgout_ptr, 0, 0, vmall, vl) ;
		  vrgout = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrgout, vmall, vl) ;

		  __vr vrgoutP = _vel_vshf_vvvsl(vrgout, vrgout, VE_VSHUFFLE_YUZU, vl) ;

		  const float *pKerValue = pKernel + kernGroupOffset + ((c * gInChannelGroup + k) * kernHeight + r) * kernWidth + s;

		  const uint64_t kerValue01 = _vel_pack_f32p(pKerValue,
							     pKerValue + kernHeight * kernWidth ) ;

		  vrsum01 = _vel_pvfmad_vvsvl(vrsum01, kerValue01, vrgoutP, vl) ;
		} // gOutChannel

	      } // kernWidth
	    } // kernHeight

	    _vel_vstu_vssl(vrsum01, 4, pGIn+gInIndex, vl) ;
	    _vel_vstl_vssl(vrsum01, 4, pGIn+gInIndex+gInPixels, vl) ;

	  } // gOutPixels

	  k+=2 ;
	}
	if( ((gInChannelGroup>>2) & 0x01 ) == 1 ) {

	  for (int64_t h=0; h<gInHeight; h+=nH) {
	    const int64_t vl = gInWidth * (gInHeight - h < nH ? gInHeight - h : nH) ;
	    const int64_t gip = h * gInWidth ;

	    const int64_t gInIndex = gInGroupOffset + ((n * gInChannel + k) * gInHeight ) * gInWidth + gip ;

	    __vr vrsum01 = _vel_pvbrd_vsl(0UL, vl) ;
	    __vr vrsum23 = _vel_pvbrd_vsl(0UL, vl) ;

	    for (int64_t r=0; r<kernHeight; r++) {
	      __vr vry = _vel_vaddsl_vsvl(padHeight-r+h, vrh, vl) ;

	      __vm256 vmy1 =  _vel_vfmklge_mvl(vry, vl) ;
	      __vm256 vmy2 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutHeight,vry, vl), vl) ;

	      __vm256 vmy = _vel_andm_mmm(vmy1, vmy2) ;

	      for (int64_t s=0; s<kernWidth; s++) {
		__vr vrx = _vel_vaddsl_vsvl(padWidth-s, vrw, vl) ;

		__vm256 vmx1 =  _vel_vfmklge_mvl(vrx, vl) ;
		__vm256 vmx2 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutWidth,vrx, vl), vl) ;

		__vm256 vmx = _vel_andm_mmm(vmx1, vmx2) ;

		__vm256 vmall = _vel_andm_mmm(vmy,vmx) ;

		for (int64_t c=0; c<gOutChannelGroup; c++) {
		  int64_t gOutIndex    = gOutGroupOffset + ((n * gOutChannel + c) * gOutHeight) * gOutWidth ;
		  __vr vrgout_ptr = _vel_vsfa_vvssl(_vel_vaddsl_vvvl(_vel_vmulsl_vsvl(gOutWidth, vry, vl), vrx, vl),
						2,
						(unsigned long)(pGOut+gOutIndex), vl) ;

		  __vr vrgout = _vel_vgtu_vvssml(vrgout_ptr, 0, 0, vmall, vl) ;
		  vrgout = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrgout, vmall, vl) ;

		  __vr vrgoutP = _vel_vshf_vvvsl(vrgout, vrgout, VE_VSHUFFLE_YUZU, vl) ;

		  const float *pKerValue = pKernel + kernGroupOffset + ((c * gInChannelGroup + k) * kernHeight + r) * kernWidth + s;

		  const uint64_t kerValue01 = _vel_pack_f32p(pKerValue,
							     pKerValue + kernHeight * kernWidth ) ;
		  const uint64_t kerValue23 = _vel_pack_f32p(pKerValue + 2 * kernHeight * kernWidth,
							     pKerValue + 3 * kernHeight * kernWidth ) ;

		  vrsum01 = _vel_pvfmad_vvsvl(vrsum01, kerValue01, vrgoutP, vl) ;
		  vrsum23 = _vel_pvfmad_vvsvl(vrsum23, kerValue23, vrgoutP, vl) ;
		} // gOutChannel

	      } // kernWidth
	    } // kernHeight

	    _vel_vstu_vssl(vrsum01, 4, pGIn+gInIndex, vl) ;
	    _vel_vstl_vssl(vrsum01, 4, pGIn+gInIndex+  gInPixels, vl) ;
	    _vel_vstu_vssl(vrsum23, 4, pGIn+gInIndex+2*gInPixels, vl) ;
	    _vel_vstl_vssl(vrsum23, 4, pGIn+gInIndex+3*gInPixels, vl) ;

	  } // gOutPixels

	  k+=4 ;
	}
	if( ((gInChannelGroup>>3) & 0x01 ) == 1 ) {

	  for (int64_t h=0; h<gInHeight; h+=nH) {
	    const int64_t vl = gInWidth * (gInHeight - h < nH ? gInHeight - h : nH) ;
	    const int64_t gip = h * gInWidth ;

	    const int64_t gInIndex = gInGroupOffset + ((n * gInChannel + k) * gInHeight ) * gInWidth + gip ;

	    __vr vrsum01 = _vel_pvbrd_vsl(0UL, vl) ;
	    __vr vrsum23 = _vel_pvbrd_vsl(0UL, vl) ;
	    __vr vrsum45 = _vel_pvbrd_vsl(0UL, vl) ;
	    __vr vrsum67 = _vel_pvbrd_vsl(0UL, vl) ;

	    for (int64_t r=0; r<kernHeight; r++) {
	      __vr vry = _vel_vaddsl_vsvl(padHeight-r+h, vrh, vl) ;

	      __vm256 vmy1 =  _vel_vfmklge_mvl(vry, vl) ;
	      __vm256 vmy2 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutHeight,vry, vl), vl) ;

	      __vm256 vmy = _vel_andm_mmm(vmy1, vmy2) ;

	      for (int64_t s=0; s<kernWidth; s++) {
		__vr vrx = _vel_vaddsl_vsvl(padWidth-s, vrw, vl) ;

		__vm256 vmx1 =  _vel_vfmklge_mvl(vrx, vl) ;
		__vm256 vmx2 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutWidth,vrx, vl), vl) ;

		__vm256 vmx = _vel_andm_mmm(vmx1, vmx2) ;

		__vm256 vmall = _vel_andm_mmm(vmy,vmx) ;

		for (int64_t c=0; c<gOutChannelGroup; c++) {
		  int64_t gOutIndex    = gOutGroupOffset + ((n * gOutChannel + c) * gOutHeight) * gOutWidth ;
		  __vr vrgout_ptr = _vel_vsfa_vvssl(_vel_vaddsl_vvvl(_vel_vmulsl_vsvl(gOutWidth, vry, vl), vrx, vl),
						2,
						(unsigned long)(pGOut+gOutIndex), vl) ;

		  __vr vrgout = _vel_vgtu_vvssml(vrgout_ptr, 0, 0, vmall, vl) ;
		  vrgout = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrgout, vmall, vl) ;

		  __vr vrgoutP = _vel_vshf_vvvsl(vrgout, vrgout, VE_VSHUFFLE_YUZU, vl) ;

		  const float *pKerValue = pKernel + kernGroupOffset + ((c * gInChannelGroup + k) * kernHeight + r) * kernWidth + s;

		  const uint64_t kerValue01 = _vel_pack_f32p(pKerValue,
							     pKerValue + kernHeight * kernWidth ) ;
		  const uint64_t kerValue23 = _vel_pack_f32p(pKerValue + 2 * kernHeight * kernWidth,
							     pKerValue + 3 * kernHeight * kernWidth ) ;
		  const uint64_t kerValue45 = _vel_pack_f32p(pKerValue + 4 * kernHeight * kernWidth,
							     pKerValue + 5 * kernHeight * kernWidth ) ;
		  const uint64_t kerValue67 = _vel_pack_f32p(pKerValue + 6 * kernHeight * kernWidth,
							     pKerValue + 7 * kernHeight * kernWidth ) ;


		  vrsum01 = _vel_pvfmad_vvsvl(vrsum01, kerValue01, vrgoutP, vl) ;
		  vrsum23 = _vel_pvfmad_vvsvl(vrsum23, kerValue23, vrgoutP, vl) ;
		  vrsum45 = _vel_pvfmad_vvsvl(vrsum45, kerValue45, vrgoutP, vl) ;
		  vrsum67 = _vel_pvfmad_vvsvl(vrsum67, kerValue67, vrgoutP, vl) ;
		} // gOutChannel

	      } // kernWidth
	    } // kernHeight

	    _vel_vstu_vssl(vrsum01, 4, pGIn+gInIndex, vl) ;
	    _vel_vstl_vssl(vrsum01, 4, pGIn+gInIndex+  gInPixels, vl) ;
	    _vel_vstu_vssl(vrsum23, 4, pGIn+gInIndex+2*gInPixels, vl) ;
	    _vel_vstl_vssl(vrsum23, 4, pGIn+gInIndex+3*gInPixels, vl) ;
	    _vel_vstu_vssl(vrsum45, 4, pGIn+gInIndex+4*gInPixels, vl) ;
	    _vel_vstl_vssl(vrsum45, 4, pGIn+gInIndex+5*gInPixels, vl) ;
	    _vel_vstu_vssl(vrsum67, 4, pGIn+gInIndex+6*gInPixels, vl) ;
	    _vel_vstl_vssl(vrsum67, 4, pGIn+gInIndex+7*gInPixels, vl) ;
	  } // gOutPixels

	  k+=8 ;
	}
	for (; k<gInChannelGroup; k+=16) {

	  for (int64_t h=0; h<gInHeight; h+=nH) {
	    const int64_t vl = gInWidth * (gInHeight - h < nH ? gInHeight - h : nH) ;
	    const int64_t gip = h * gInWidth ;

	    const int64_t gInIndex = gInGroupOffset + ((n * gInChannel + k) * gInHeight ) * gInWidth + gip ;

	    __vr vrsum01 = _vel_pvbrd_vsl(0UL, vl) ;
	    __vr vrsum23 = _vel_pvbrd_vsl(0UL, vl) ;
	    __vr vrsum45 = _vel_pvbrd_vsl(0UL, vl) ;
	    __vr vrsum67 = _vel_pvbrd_vsl(0UL, vl) ;
	    __vr vrsum89 = _vel_pvbrd_vsl(0UL, vl) ;
	    __vr vrsumAB = _vel_pvbrd_vsl(0UL, vl) ;
	    __vr vrsumCD = _vel_pvbrd_vsl(0UL, vl) ;
	    __vr vrsumEF = _vel_pvbrd_vsl(0UL, vl) ;

	    for (int64_t r=0; r<kernHeight; r++) {
	      __vr vry = _vel_vaddsl_vsvl(padHeight-r+h, vrh, vl) ;

	      __vm256 vmy1 =  _vel_vfmklge_mvl(vry, vl) ;
	      __vm256 vmy2 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutHeight,vry, vl), vl) ;

	      __vm256 vmy = _vel_andm_mmm(vmy1, vmy2) ;

	      for (int64_t s=0; s<kernWidth; s++) {
		__vr vrx = _vel_vaddsl_vsvl(padWidth-s, vrw, vl) ;

		__vm256 vmx1 =  _vel_vfmklge_mvl(vrx, vl) ;
		__vm256 vmx2 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutWidth,vrx, vl), vl) ;

		__vm256 vmx = _vel_andm_mmm(vmx1, vmx2) ;

		__vm256 vmall = _vel_andm_mmm(vmy,vmx) ;

		for (int64_t c=0; c<gOutChannelGroup; c++) {
		  int64_t gOutIndex    = gOutGroupOffset + ((n * gOutChannel + c) * gOutHeight) * gOutWidth ;
		  __vr vrgout_ptr = _vel_vsfa_vvssl(_vel_vaddsl_vvvl(_vel_vmulsl_vsvl(gOutWidth, vry, vl), vrx, vl),
						2,
						(unsigned long)(pGOut+gOutIndex), vl) ;

		  __vr vrgout = _vel_vgtu_vvssml(vrgout_ptr, 0, 0, vmall, vl) ;
		  vrgout = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrgout, vmall, vl) ;

		  __vr vrgoutP = _vel_vshf_vvvsl(vrgout, vrgout, VE_VSHUFFLE_YUZU, vl) ;

		  const float *pKerValue = pKernel + kernGroupOffset + ((c * gInChannelGroup + k) * kernHeight + r) * kernWidth + s;

		  const uint64_t kerValue01 = _vel_pack_f32p(pKerValue,
							     pKerValue + kernHeight * kernWidth ) ;
		  const uint64_t kerValue23 = _vel_pack_f32p(pKerValue + 2 * kernHeight * kernWidth,
							     pKerValue + 3 * kernHeight * kernWidth ) ;
		  const uint64_t kerValue45 = _vel_pack_f32p(pKerValue + 4 * kernHeight * kernWidth,
							     pKerValue + 5 * kernHeight * kernWidth ) ;
		  const uint64_t kerValue67 = _vel_pack_f32p(pKerValue + 6 * kernHeight * kernWidth,
							     pKerValue + 7 * kernHeight * kernWidth ) ;
		  const uint64_t kerValue89 = _vel_pack_f32p(pKerValue + 8 * kernHeight * kernWidth,
							     pKerValue + 9 * kernHeight * kernWidth ) ;
		  const uint64_t kerValueAB = _vel_pack_f32p(pKerValue +10 * kernHeight * kernWidth,
							     pKerValue +11 * kernHeight * kernWidth ) ;
		  const uint64_t kerValueCD = _vel_pack_f32p(pKerValue +12 * kernHeight * kernWidth,
							     pKerValue +13 * kernHeight * kernWidth ) ;
		  const uint64_t kerValueEF = _vel_pack_f32p(pKerValue +14 * kernHeight * kernWidth,
							     pKerValue +15 * kernHeight * kernWidth ) ;

		  vrsum01 = _vel_pvfmad_vvsvl(vrsum01, kerValue01, vrgoutP, vl) ;
		  vrsum23 = _vel_pvfmad_vvsvl(vrsum23, kerValue23, vrgoutP, vl) ;
		  vrsum45 = _vel_pvfmad_vvsvl(vrsum45, kerValue45, vrgoutP, vl) ;
		  vrsum67 = _vel_pvfmad_vvsvl(vrsum67, kerValue67, vrgoutP, vl) ;
		  vrsum89 = _vel_pvfmad_vvsvl(vrsum89, kerValue89, vrgoutP, vl) ;
		  vrsumAB = _vel_pvfmad_vvsvl(vrsumAB, kerValueAB, vrgoutP, vl) ;
		  vrsumCD = _vel_pvfmad_vvsvl(vrsumCD, kerValueCD, vrgoutP, vl) ;
		  vrsumEF = _vel_pvfmad_vvsvl(vrsumEF, kerValueEF, vrgoutP, vl) ;
		} // gOutChannel

	      } // kernWidth
	    } // kernHeight

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


	  } // gOutPixels
	} // gInChannel
      } // group
    } // batch
  }


  return VEDNN_SUCCESS;
}
