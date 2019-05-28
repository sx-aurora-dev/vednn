#include <stdint.h>
#include <stdlib.h>

#include "vednn.h"

#include "veintrin.h"
#define VLEN	(256)

vednnError_t
vednnConvolutionForward_direct_dil1_str1_pad0_ker3_iw2XU256_ow2X_ioaligned(
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
//  const int64_t strideWidth    = pParamConv->strideWidth;		// must be 1
//  const int64_t strideHeight   = pParamConv->strideHeight;		// must be 1
//  const int64_t padWidth       = pParamConv->padWidth;		// must be 0
//  const int64_t padHeight      = pParamConv->padHeight;		// must be 0
//  const int64_t dilationWidth  = pParamConv->dilationWidth;		// must be 1
//  const int64_t dilationHeight = pParamConv->dilationHeight;		// must be 1

  const int64_t inChannelGroup  = inChannel  / group;   // equal to pDataKernel->inChannel
  const int64_t outChannelGroup = outChannel / group;   // equal to pDataKernel->outChannel

  const float * restrict pIn     = pDataIn;
  const float * restrict pKernel = pDataKernel;
  float * restrict const pOut    = pDataOut;

  const int oPixels= outHeight*outWidth ;
  {
    const int64_t inWidthHalf  = inWidth >> 1 ;
    const int64_t outWidthHalf = outWidth >> 1 ;
    const int64_t nY = VLEN / inWidthHalf ;

    _ve_lvl(VLEN) ;

    __vr vrseq = _ve_vseq_v() ;
    __vm256 vm_s0, vm_s2 ;
    {
      __vr vry_s0  = _ve_vdivsl_vvs(vrseq, inWidthHalf) ;
      __vr vrx_s0  = _ve_vsubsl_vvv(vrseq, _ve_vmulul_vsv(inWidthHalf,vry_s0)) ;
      vm_s0 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(outWidthHalf, vrx_s0)) ; // condition(x<outWidthHalf)

      __vr vrseq2  = _ve_vaddsl_vsv(inWidthHalf-1, vrseq) ;
      __vr vry_s2  = _ve_vdivsl_vvs(vrseq2, inWidthHalf) ;
      __vr vrx_s2  = _ve_vsubsl_vvv(vrseq2, _ve_vmulul_vsv(inWidthHalf,vry_s2)) ;
      vm_s2 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(outWidthHalf, vrx_s2)) ; // condition(x<outWidthHalf)
    }

    for (int64_t n = 0; n < batch; n++) {
      for (int64_t g = 0; g < group; g++) {
	const int64_t inGroupOffset   = g * inChannelGroup * inHeight * inWidth;
	const int64_t outGroupOffset  = g * outChannelGroup * outHeight * outWidth;
	const int64_t kernGroupOffset = g * outChannelGroup * inChannelGroup * kernHeight * kernWidth;

	int k = 0 ;
	if ( (outChannelGroup & 0x01) == 1 ) {
	  int64_t outIndex = outGroupOffset + (n * outChannel + k) * oPixels ;

	  for (int64_t y=0; y<outHeight; y+=nY) {
	    const int64_t vl0 = inWidthHalf * (outHeight - y < nY ? outHeight - y : nY) ;
	    const int64_t vl1 = outWidthHalf * (outHeight - y < nY ? outHeight - y : nY) ;
	    const int64_t op = y * outWidth ;

	    _ve_lvl(vl1) ;
	    __vr vrsum = _ve_vbrd_vs_i64(0UL) ;

	    for (int64_t c = 0; c < inChannelGroup; c++) {

	      const float *pInChannel = pIn + inGroupOffset + (((n * inChannel + c) * inHeight + y) * inWidth ) ;

	      _ve_lvl(vl0) ;
	      __vr vrin_r0 = _ve_vld_vss(8, pInChannel+0*inWidth) ;
	      __vr vrin_r1 = _ve_vld_vss(8, pInChannel+1*inWidth) ;
	      __vr vrin_r2 = _ve_vld_vss(8, pInChannel+2*inWidth) ;

	      __vr vrin_r0s0 = _ve_vcp_vvmv(vrin_r0, vm_s0, _ve_vbrd_vs_i64(0UL)) ;
	      __vr vrin_r0s2 = _ve_vcp_vvmv(vrin_r0, vm_s2, _ve_vbrd_vs_i64(0UL)) ;
	      __vr vrin_r1s0 = _ve_vcp_vvmv(vrin_r1, vm_s0, _ve_vbrd_vs_i64(0UL)) ;
	      __vr vrin_r1s2 = _ve_vcp_vvmv(vrin_r1, vm_s2, _ve_vbrd_vs_i64(0UL)) ;
	      __vr vrin_r2s0 = _ve_vcp_vvmv(vrin_r2, vm_s0, _ve_vbrd_vs_i64(0UL)) ;
	      __vr vrin_r2s2 = _ve_vcp_vvmv(vrin_r2, vm_s2, _ve_vbrd_vs_i64(0UL)) ;

	      _ve_lvl(vl1) ;
	      __vr vrin_r0s1 = _ve_vshf_vvvs(vrin_r0s0, vrin_r0s2,VE_VSHUFFLE_ZLYU) ;
	      __vr vrin_r1s1 = _ve_vshf_vvvs(vrin_r1s0, vrin_r1s2,VE_VSHUFFLE_ZLYU) ;
	      __vr vrin_r2s1 = _ve_vshf_vvvs(vrin_r2s0, vrin_r2s2,VE_VSHUFFLE_ZLYU) ;


	      const float *pKerValue = pKernel+kernGroupOffset+((k*inChannelGroup+c)*kernHeight)*kernWidth;
	      vrsum = _ve_pvfmad_vvsv(vrsum, _ve_pack_f32a(pKerValue), vrin_r0s0) ; pKerValue++ ;
	      vrsum = _ve_pvfmad_vvsv(vrsum, _ve_pack_f32a(pKerValue), vrin_r0s1) ; pKerValue++ ;
	      vrsum = _ve_pvfmad_vvsv(vrsum, _ve_pack_f32a(pKerValue), vrin_r0s2) ; pKerValue++ ;
	      vrsum = _ve_pvfmad_vvsv(vrsum, _ve_pack_f32a(pKerValue), vrin_r1s0) ; pKerValue++ ;
	      vrsum = _ve_pvfmad_vvsv(vrsum, _ve_pack_f32a(pKerValue), vrin_r1s1) ; pKerValue++ ;
	      vrsum = _ve_pvfmad_vvsv(vrsum, _ve_pack_f32a(pKerValue), vrin_r1s2) ; pKerValue++ ;
	      vrsum = _ve_pvfmad_vvsv(vrsum, _ve_pack_f32a(pKerValue), vrin_r2s0) ; pKerValue++ ;
	      vrsum = _ve_pvfmad_vvsv(vrsum, _ve_pack_f32a(pKerValue), vrin_r2s1) ; pKerValue++ ;
	      vrsum = _ve_pvfmad_vvsv(vrsum, _ve_pack_f32a(pKerValue), vrin_r2s2) ; pKerValue++ ;

	    } // inChannel

	    _ve_vst_vss(vrsum, 8, pOut+outIndex) ;

	    outIndex += 2*vl1 ;
	  } // outPixels
	  k++ ;
	}
	if ( ((outChannelGroup >> 1) & 0x01) == 1 ) {
	  int64_t outIndex = outGroupOffset + (n * outChannel + k) * oPixels ;

	  for (int64_t y=0; y<outHeight; y+=nY) {
	    const int64_t vl0 = inWidthHalf * (outHeight - y < nY ? outHeight - y : nY) ;
	    const int64_t vl1 = outWidthHalf * (outHeight - y < nY ? outHeight - y : nY) ;
	    const int64_t op = y * outWidth ;

	    _ve_lvl(vl1) ;
	    __vr vrsum0 = _ve_vbrd_vs_i64(0UL) ;
	    __vr vrsum1 = _ve_vbrd_vs_i64(0UL) ;

	    for (int64_t c = 0; c < inChannelGroup; c++) {

	      const float *pInChannel = pIn + inGroupOffset + (((n * inChannel + c) * inHeight + y) * inWidth ) ;

	      _ve_lvl(vl0) ;
	      __vr vrin_r0 = _ve_vld_vss(8, pInChannel+0*inWidth) ;
	      __vr vrin_r1 = _ve_vld_vss(8, pInChannel+1*inWidth) ;
	      __vr vrin_r2 = _ve_vld_vss(8, pInChannel+2*inWidth) ;

	      __vr vrin_r0s0 = _ve_vcp_vvmv(vrin_r0, vm_s0, _ve_vbrd_vs_i64(0UL)) ;
	      __vr vrin_r0s2 = _ve_vcp_vvmv(vrin_r0, vm_s2, _ve_vbrd_vs_i64(0UL)) ;
	      __vr vrin_r1s0 = _ve_vcp_vvmv(vrin_r1, vm_s0, _ve_vbrd_vs_i64(0UL)) ;
	      __vr vrin_r1s2 = _ve_vcp_vvmv(vrin_r1, vm_s2, _ve_vbrd_vs_i64(0UL)) ;
	      __vr vrin_r2s0 = _ve_vcp_vvmv(vrin_r2, vm_s0, _ve_vbrd_vs_i64(0UL)) ;
	      __vr vrin_r2s2 = _ve_vcp_vvmv(vrin_r2, vm_s2, _ve_vbrd_vs_i64(0UL)) ;

	      _ve_lvl(vl1) ;
	      __vr vrin_r0s1 = _ve_vshf_vvvs(vrin_r0s0, vrin_r0s2,VE_VSHUFFLE_ZLYU) ;
	      __vr vrin_r1s1 = _ve_vshf_vvvs(vrin_r1s0, vrin_r1s2,VE_VSHUFFLE_ZLYU) ;
	      __vr vrin_r2s1 = _ve_vshf_vvvs(vrin_r2s0, vrin_r2s2,VE_VSHUFFLE_ZLYU) ;

	      const float *pKerValue = pKernel+kernGroupOffset+((k*inChannelGroup+c)*kernHeight)*kernWidth;
#define VFADD2(VRIN)											\
{													\
  const uint64_t kerValue0 = _ve_pack_f32a(pKerValue) ;							\
  const uint64_t kerValue1 = _ve_pack_f32a(pKerValue+ inChannelGroup * kernHeight * kernWidth) ;	\
  vrsum0 = _ve_pvfmad_vvsv(vrsum0, kerValue0, VRIN) ;							\
  vrsum1 = _ve_pvfmad_vvsv(vrsum1, kerValue1, VRIN) ;							\
}
	      VFADD2(vrin_r0s0) ; pKerValue++ ;
	      VFADD2(vrin_r0s1) ; pKerValue++ ;
	      VFADD2(vrin_r0s2) ; pKerValue++ ;
	      VFADD2(vrin_r1s0) ; pKerValue++ ;
	      VFADD2(vrin_r1s1) ; pKerValue++ ;
	      VFADD2(vrin_r1s2) ; pKerValue++ ;
	      VFADD2(vrin_r2s0) ; pKerValue++ ;
	      VFADD2(vrin_r2s1) ; pKerValue++ ;
	      VFADD2(vrin_r2s2) ; pKerValue++ ;
#undef VFADD2
	    } // inChannel

	    _ve_vst_vss(vrsum0, 8, pOut+outIndex) ;
	    _ve_vst_vss(vrsum1, 8, pOut+outIndex+ 1*oPixels) ;

	    outIndex += 2*vl1 ;
	  } // outPixels

	  k+=2 ;
	}
	if ( ((outChannelGroup >> 2) & 0x01) == 1 ) {
	  int64_t outIndex = outGroupOffset + (n * outChannel + k) * oPixels ;

	  for (int64_t y=0; y<outHeight; y+=nY) {
	    const int64_t vl0 = inWidthHalf * (outHeight - y < nY ? outHeight - y : nY) ;
	    const int64_t vl1 = outWidthHalf * (outHeight - y < nY ? outHeight - y : nY) ;
	    const int64_t op = y * outWidth ;

	    _ve_lvl(vl1) ;
	    __vr vrsum0 = _ve_vbrd_vs_i64(0UL) ;
	    __vr vrsum1 = _ve_vbrd_vs_i64(0UL) ;
	    __vr vrsum2 = _ve_vbrd_vs_i64(0UL) ;
	    __vr vrsum3 = _ve_vbrd_vs_i64(0UL) ;

	    for (int64_t c = 0; c < inChannelGroup; c++) {

	      const float *pInChannel = pIn + inGroupOffset + (((n * inChannel + c) * inHeight + y) * inWidth ) ;

	      _ve_lvl(vl0) ;
	      __vr vrin_r0 = _ve_vld_vss(8, pInChannel+0*inWidth) ;
	      __vr vrin_r1 = _ve_vld_vss(8, pInChannel+1*inWidth) ;
	      __vr vrin_r2 = _ve_vld_vss(8, pInChannel+2*inWidth) ;

	      __vr vrin_r0s0 = _ve_vcp_vvmv(vrin_r0, vm_s0, _ve_vbrd_vs_i64(0UL)) ;
	      __vr vrin_r0s2 = _ve_vcp_vvmv(vrin_r0, vm_s2, _ve_vbrd_vs_i64(0UL)) ;
	      __vr vrin_r1s0 = _ve_vcp_vvmv(vrin_r1, vm_s0, _ve_vbrd_vs_i64(0UL)) ;
	      __vr vrin_r1s2 = _ve_vcp_vvmv(vrin_r1, vm_s2, _ve_vbrd_vs_i64(0UL)) ;
	      __vr vrin_r2s0 = _ve_vcp_vvmv(vrin_r2, vm_s0, _ve_vbrd_vs_i64(0UL)) ;
	      __vr vrin_r2s2 = _ve_vcp_vvmv(vrin_r2, vm_s2, _ve_vbrd_vs_i64(0UL)) ;

	      _ve_lvl(vl1) ;
	      __vr vrin_r0s1 = _ve_vshf_vvvs(vrin_r0s0, vrin_r0s2,VE_VSHUFFLE_ZLYU) ;
	      __vr vrin_r1s1 = _ve_vshf_vvvs(vrin_r1s0, vrin_r1s2,VE_VSHUFFLE_ZLYU) ;
	      __vr vrin_r2s1 = _ve_vshf_vvvs(vrin_r2s0, vrin_r2s2,VE_VSHUFFLE_ZLYU) ;

	      const float *pKerValue = pKernel+kernGroupOffset+((k*inChannelGroup+c)*kernHeight)*kernWidth;
#define VFADD4(VRIN)											\
{													\
  const uint64_t kerValue0 = _ve_pack_f32a(pKerValue) ;							\
  const uint64_t kerValue1 = _ve_pack_f32a(pKerValue+     inChannelGroup * kernHeight * kernWidth) ;	\
  const uint64_t kerValue2 = _ve_pack_f32a(pKerValue+ 2 * inChannelGroup * kernHeight * kernWidth) ;	\
  const uint64_t kerValue3 = _ve_pack_f32a(pKerValue+ 3 * inChannelGroup * kernHeight * kernWidth) ;	\
  vrsum0 = _ve_pvfmad_vvsv(vrsum0, kerValue0, VRIN) ;							\
  vrsum1 = _ve_pvfmad_vvsv(vrsum1, kerValue1, VRIN) ;							\
  vrsum2 = _ve_pvfmad_vvsv(vrsum2, kerValue2, VRIN) ;							\
  vrsum3 = _ve_pvfmad_vvsv(vrsum3, kerValue3, VRIN) ;							\
}
	      VFADD4(vrin_r0s0) ; pKerValue++ ;
	      VFADD4(vrin_r0s1) ; pKerValue++ ;
	      VFADD4(vrin_r0s2) ; pKerValue++ ;
	      VFADD4(vrin_r1s0) ; pKerValue++ ;
	      VFADD4(vrin_r1s1) ; pKerValue++ ;
	      VFADD4(vrin_r1s2) ; pKerValue++ ;
	      VFADD4(vrin_r2s0) ; pKerValue++ ;
	      VFADD4(vrin_r2s1) ; pKerValue++ ;
	      VFADD4(vrin_r2s2) ; pKerValue++ ;
#undef VFADD4
	    } // inChannel

	    _ve_vst_vss(vrsum0, 8, pOut+outIndex) ;
	    _ve_vst_vss(vrsum1, 8, pOut+outIndex+ 1*oPixels) ;
	    _ve_vst_vss(vrsum2, 8, pOut+outIndex+ 2*oPixels) ;
	    _ve_vst_vss(vrsum3, 8, pOut+outIndex+ 3*oPixels) ;

	    outIndex += 2*vl1 ;
	  } // outPixels

	  k+=4 ;
	}
	if ( ((outChannelGroup >> 3) & 0x01) == 1 ) {
	  int64_t outIndex = outGroupOffset + (n * outChannel + k) * oPixels ;

	  for (int64_t y=0; y<outHeight; y+=nY) {
	    const int64_t vl0 = inWidthHalf * (outHeight - y < nY ? outHeight - y : nY) ;
	    const int64_t vl1 = outWidthHalf * (outHeight - y < nY ? outHeight - y : nY) ;
	    const int64_t op = y * outWidth ;

	    _ve_lvl(vl1) ;
	    __vr vrsum0 = _ve_vbrd_vs_i64(0UL) ;
	    __vr vrsum1 = _ve_vbrd_vs_i64(0UL) ;
	    __vr vrsum2 = _ve_vbrd_vs_i64(0UL) ;
	    __vr vrsum3 = _ve_vbrd_vs_i64(0UL) ;
	    __vr vrsum4 = _ve_vbrd_vs_i64(0UL) ;
	    __vr vrsum5 = _ve_vbrd_vs_i64(0UL) ;
	    __vr vrsum6 = _ve_vbrd_vs_i64(0UL) ;
	    __vr vrsum7 = _ve_vbrd_vs_i64(0UL) ;

	    for (int64_t c = 0; c < inChannelGroup; c++) {

	      const float *pInChannel = pIn + inGroupOffset + (((n * inChannel + c) * inHeight + y) * inWidth ) ;

	      _ve_lvl(vl0) ;
	      __vr vrin_r0 = _ve_vld_vss(8, pInChannel+0*inWidth) ;
	      __vr vrin_r1 = _ve_vld_vss(8, pInChannel+1*inWidth) ;
	      __vr vrin_r2 = _ve_vld_vss(8, pInChannel+2*inWidth) ;

	      __vr vrin_r0s0 = _ve_vcp_vvmv(vrin_r0, vm_s0, _ve_vbrd_vs_i64(0UL)) ;
	      __vr vrin_r0s2 = _ve_vcp_vvmv(vrin_r0, vm_s2, _ve_vbrd_vs_i64(0UL)) ;
	      __vr vrin_r1s0 = _ve_vcp_vvmv(vrin_r1, vm_s0, _ve_vbrd_vs_i64(0UL)) ;
	      __vr vrin_r1s2 = _ve_vcp_vvmv(vrin_r1, vm_s2, _ve_vbrd_vs_i64(0UL)) ;
	      __vr vrin_r2s0 = _ve_vcp_vvmv(vrin_r2, vm_s0, _ve_vbrd_vs_i64(0UL)) ;
	      __vr vrin_r2s2 = _ve_vcp_vvmv(vrin_r2, vm_s2, _ve_vbrd_vs_i64(0UL)) ;

	      _ve_lvl(vl1) ;
	      __vr vrin_r0s1 = _ve_vshf_vvvs(vrin_r0s0, vrin_r0s2,VE_VSHUFFLE_ZLYU) ;
	      __vr vrin_r1s1 = _ve_vshf_vvvs(vrin_r1s0, vrin_r1s2,VE_VSHUFFLE_ZLYU) ;
	      __vr vrin_r2s1 = _ve_vshf_vvvs(vrin_r2s0, vrin_r2s2,VE_VSHUFFLE_ZLYU) ;

	      const float *pKerValue = pKernel+kernGroupOffset+((k*inChannelGroup+c)*kernHeight)*kernWidth;
#define VFADD8(VRIN)											\
{													\
  const uint64_t kerValue0 = _ve_pack_f32a(pKerValue) ;							\
  const uint64_t kerValue1 = _ve_pack_f32a(pKerValue+     inChannelGroup * kernHeight * kernWidth) ;	\
  const uint64_t kerValue2 = _ve_pack_f32a(pKerValue+ 2 * inChannelGroup * kernHeight * kernWidth) ;	\
  const uint64_t kerValue3 = _ve_pack_f32a(pKerValue+ 3 * inChannelGroup * kernHeight * kernWidth) ;	\
  const uint64_t kerValue4 = _ve_pack_f32a(pKerValue+ 4 * inChannelGroup * kernHeight * kernWidth) ;	\
  const uint64_t kerValue5 = _ve_pack_f32a(pKerValue+ 5 * inChannelGroup * kernHeight * kernWidth) ;	\
  const uint64_t kerValue6 = _ve_pack_f32a(pKerValue+ 6 * inChannelGroup * kernHeight * kernWidth) ;	\
  const uint64_t kerValue7 = _ve_pack_f32a(pKerValue+ 7 * inChannelGroup * kernHeight * kernWidth) ;	\
  vrsum0 = _ve_pvfmad_vvsv(vrsum0, kerValue0, VRIN) ;							\
  vrsum1 = _ve_pvfmad_vvsv(vrsum1, kerValue1, VRIN) ;							\
  vrsum2 = _ve_pvfmad_vvsv(vrsum2, kerValue2, VRIN) ;							\
  vrsum3 = _ve_pvfmad_vvsv(vrsum3, kerValue3, VRIN) ;							\
  vrsum4 = _ve_pvfmad_vvsv(vrsum4, kerValue4, VRIN) ;							\
  vrsum5 = _ve_pvfmad_vvsv(vrsum5, kerValue5, VRIN) ;							\
  vrsum6 = _ve_pvfmad_vvsv(vrsum6, kerValue6, VRIN) ;							\
  vrsum7 = _ve_pvfmad_vvsv(vrsum7, kerValue7, VRIN) ;							\
}
	      VFADD8(vrin_r0s0) ; pKerValue++ ;
	      VFADD8(vrin_r0s1) ; pKerValue++ ;
	      VFADD8(vrin_r0s2) ; pKerValue++ ;
	      VFADD8(vrin_r1s0) ; pKerValue++ ;
	      VFADD8(vrin_r1s1) ; pKerValue++ ;
	      VFADD8(vrin_r1s2) ; pKerValue++ ;
	      VFADD8(vrin_r2s0) ; pKerValue++ ;
	      VFADD8(vrin_r2s1) ; pKerValue++ ;
	      VFADD8(vrin_r2s2) ; pKerValue++ ;
#undef VFADD8
	    } // inChannel

	    _ve_vst_vss(vrsum0, 8, pOut+outIndex) ;
	    _ve_vst_vss(vrsum1, 8, pOut+outIndex+ 1*oPixels) ;
	    _ve_vst_vss(vrsum2, 8, pOut+outIndex+ 2*oPixels) ;
	    _ve_vst_vss(vrsum3, 8, pOut+outIndex+ 3*oPixels) ;
	    _ve_vst_vss(vrsum4, 8, pOut+outIndex+ 4*oPixels) ;
	    _ve_vst_vss(vrsum5, 8, pOut+outIndex+ 5*oPixels) ;
	    _ve_vst_vss(vrsum6, 8, pOut+outIndex+ 6*oPixels) ;
	    _ve_vst_vss(vrsum7, 8, pOut+outIndex+ 7*oPixels) ;

	    outIndex += 2*vl1 ;
	  } // outPixels

	  k+=8 ;
	}
	for ( ; k < outChannelGroup; k+=16) {
	  int64_t outIndex = outGroupOffset + (n * outChannel + k) * oPixels ;

	  for (int64_t y=0; y<outHeight; y+=nY) {
	    const int64_t vl0 = inWidthHalf * (outHeight - y < nY ? outHeight - y : nY) ;
	    const int64_t vl1 = outWidthHalf * (outHeight - y < nY ? outHeight - y : nY) ;
	    const int64_t op = y * outWidth ;

	    _ve_lvl(vl1) ;
	    __vr vrsum0 = _ve_vbrd_vs_i64(0UL) ;
	    __vr vrsum1 = _ve_vbrd_vs_i64(0UL) ;
	    __vr vrsum2 = _ve_vbrd_vs_i64(0UL) ;
	    __vr vrsum3 = _ve_vbrd_vs_i64(0UL) ;
	    __vr vrsum4 = _ve_vbrd_vs_i64(0UL) ;
	    __vr vrsum5 = _ve_vbrd_vs_i64(0UL) ;
	    __vr vrsum6 = _ve_vbrd_vs_i64(0UL) ;
	    __vr vrsum7 = _ve_vbrd_vs_i64(0UL) ;
	    __vr vrsum8 = _ve_vbrd_vs_i64(0UL) ;
	    __vr vrsum9 = _ve_vbrd_vs_i64(0UL) ;
	    __vr vrsumA = _ve_vbrd_vs_i64(0UL) ;
	    __vr vrsumB = _ve_vbrd_vs_i64(0UL) ;
	    __vr vrsumC = _ve_vbrd_vs_i64(0UL) ;
	    __vr vrsumD = _ve_vbrd_vs_i64(0UL) ;
	    __vr vrsumE = _ve_vbrd_vs_i64(0UL) ;
	    __vr vrsumF = _ve_vbrd_vs_i64(0UL) ;

	    for (int64_t c = 0; c < inChannelGroup; c++) {

	      const float *pInChannel = pIn + inGroupOffset + (((n * inChannel + c) * inHeight + y) * inWidth ) ;

	      _ve_lvl(vl0) ;
	      __vr vrin_r0 = _ve_vld_vss(8, pInChannel+0*inWidth) ;
	      __vr vrin_r1 = _ve_vld_vss(8, pInChannel+1*inWidth) ;
	      __vr vrin_r2 = _ve_vld_vss(8, pInChannel+2*inWidth) ;

	      __vr vrin_r0s0 = _ve_vcp_vvmv(vrin_r0, vm_s0, _ve_vbrd_vs_i64(0UL)) ;
	      __vr vrin_r0s2 = _ve_vcp_vvmv(vrin_r0, vm_s2, _ve_vbrd_vs_i64(0UL)) ;
	      __vr vrin_r1s0 = _ve_vcp_vvmv(vrin_r1, vm_s0, _ve_vbrd_vs_i64(0UL)) ;
	      __vr vrin_r1s2 = _ve_vcp_vvmv(vrin_r1, vm_s2, _ve_vbrd_vs_i64(0UL)) ;
	      __vr vrin_r2s0 = _ve_vcp_vvmv(vrin_r2, vm_s0, _ve_vbrd_vs_i64(0UL)) ;
	      __vr vrin_r2s2 = _ve_vcp_vvmv(vrin_r2, vm_s2, _ve_vbrd_vs_i64(0UL)) ;

	      _ve_lvl(vl1) ;
	      __vr vrin_r0s1 = _ve_vshf_vvvs(vrin_r0s0, vrin_r0s2,VE_VSHUFFLE_ZLYU) ;
	      __vr vrin_r1s1 = _ve_vshf_vvvs(vrin_r1s0, vrin_r1s2,VE_VSHUFFLE_ZLYU) ;
	      __vr vrin_r2s1 = _ve_vshf_vvvs(vrin_r2s0, vrin_r2s2,VE_VSHUFFLE_ZLYU) ;

	      const float *pKerValue = pKernel+kernGroupOffset+((k*inChannelGroup+c)*kernHeight)*kernWidth;
#define VFADD16(VRIN)											\
{													\
  const uint64_t kerValue0 = _ve_pack_f32a(pKerValue) ;							\
  const uint64_t kerValue1 = _ve_pack_f32a(pKerValue+     inChannelGroup * kernHeight * kernWidth) ;	\
  const uint64_t kerValue2 = _ve_pack_f32a(pKerValue+ 2 * inChannelGroup * kernHeight * kernWidth) ;	\
  const uint64_t kerValue3 = _ve_pack_f32a(pKerValue+ 3 * inChannelGroup * kernHeight * kernWidth) ;	\
  const uint64_t kerValue4 = _ve_pack_f32a(pKerValue+ 4 * inChannelGroup * kernHeight * kernWidth) ;	\
  const uint64_t kerValue5 = _ve_pack_f32a(pKerValue+ 5 * inChannelGroup * kernHeight * kernWidth) ;	\
  const uint64_t kerValue6 = _ve_pack_f32a(pKerValue+ 6 * inChannelGroup * kernHeight * kernWidth) ;	\
  const uint64_t kerValue7 = _ve_pack_f32a(pKerValue+ 7 * inChannelGroup * kernHeight * kernWidth) ;	\
  const uint64_t kerValue8 = _ve_pack_f32a(pKerValue+ 8 * inChannelGroup * kernHeight * kernWidth) ;	\
  const uint64_t kerValue9 = _ve_pack_f32a(pKerValue+ 9 * inChannelGroup * kernHeight * kernWidth) ;	\
  const uint64_t kerValueA = _ve_pack_f32a(pKerValue+10 * inChannelGroup * kernHeight * kernWidth) ;	\
  const uint64_t kerValueB = _ve_pack_f32a(pKerValue+11 * inChannelGroup * kernHeight * kernWidth) ;	\
  const uint64_t kerValueC = _ve_pack_f32a(pKerValue+12 * inChannelGroup * kernHeight * kernWidth) ;	\
  const uint64_t kerValueD = _ve_pack_f32a(pKerValue+13 * inChannelGroup * kernHeight * kernWidth) ;	\
  const uint64_t kerValueE = _ve_pack_f32a(pKerValue+14 * inChannelGroup * kernHeight * kernWidth) ;	\
  const uint64_t kerValueF = _ve_pack_f32a(pKerValue+15 * inChannelGroup * kernHeight * kernWidth) ;	\
  vrsum0 = _ve_pvfmad_vvsv(vrsum0, kerValue0, VRIN) ;							\
  vrsum1 = _ve_pvfmad_vvsv(vrsum1, kerValue1, VRIN) ;							\
  vrsum2 = _ve_pvfmad_vvsv(vrsum2, kerValue2, VRIN) ;							\
  vrsum3 = _ve_pvfmad_vvsv(vrsum3, kerValue3, VRIN) ;							\
  vrsum4 = _ve_pvfmad_vvsv(vrsum4, kerValue4, VRIN) ;							\
  vrsum5 = _ve_pvfmad_vvsv(vrsum5, kerValue5, VRIN) ;							\
  vrsum6 = _ve_pvfmad_vvsv(vrsum6, kerValue6, VRIN) ;							\
  vrsum7 = _ve_pvfmad_vvsv(vrsum7, kerValue7, VRIN) ;							\
  vrsum8 = _ve_pvfmad_vvsv(vrsum8, kerValue8, VRIN) ;							\
  vrsum9 = _ve_pvfmad_vvsv(vrsum9, kerValue9, VRIN) ;							\
  vrsumA = _ve_pvfmad_vvsv(vrsumA, kerValueA, VRIN) ;							\
  vrsumB = _ve_pvfmad_vvsv(vrsumB, kerValueB, VRIN) ;							\
  vrsumC = _ve_pvfmad_vvsv(vrsumC, kerValueC, VRIN) ;							\
  vrsumD = _ve_pvfmad_vvsv(vrsumD, kerValueD, VRIN) ;							\
  vrsumE = _ve_pvfmad_vvsv(vrsumE, kerValueE, VRIN) ;							\
  vrsumF = _ve_pvfmad_vvsv(vrsumF, kerValueF, VRIN) ;							\
}
	      VFADD16(vrin_r0s0) ; pKerValue++ ;
	      VFADD16(vrin_r0s1) ; pKerValue++ ;
	      VFADD16(vrin_r0s2) ; pKerValue++ ;
	      VFADD16(vrin_r1s0) ; pKerValue++ ;
	      VFADD16(vrin_r1s1) ; pKerValue++ ;
	      VFADD16(vrin_r1s2) ; pKerValue++ ;
	      VFADD16(vrin_r2s0) ; pKerValue++ ;
	      VFADD16(vrin_r2s1) ; pKerValue++ ;
	      VFADD16(vrin_r2s2) ; pKerValue++ ;
#undef VFADD8
	    } // inChannel

	    _ve_vst_vss(vrsum0, 8, pOut+outIndex) ;
	    _ve_vst_vss(vrsum1, 8, pOut+outIndex+ 1*oPixels) ;
	    _ve_vst_vss(vrsum2, 8, pOut+outIndex+ 2*oPixels) ;
	    _ve_vst_vss(vrsum3, 8, pOut+outIndex+ 3*oPixels) ;
	    _ve_vst_vss(vrsum4, 8, pOut+outIndex+ 4*oPixels) ;
	    _ve_vst_vss(vrsum5, 8, pOut+outIndex+ 5*oPixels) ;
	    _ve_vst_vss(vrsum6, 8, pOut+outIndex+ 6*oPixels) ;
	    _ve_vst_vss(vrsum7, 8, pOut+outIndex+ 7*oPixels) ;
	    _ve_vst_vss(vrsum8, 8, pOut+outIndex+ 8*oPixels) ;
	    _ve_vst_vss(vrsum9, 8, pOut+outIndex+ 9*oPixels) ;
	    _ve_vst_vss(vrsumA, 8, pOut+outIndex+10*oPixels) ;
	    _ve_vst_vss(vrsumB, 8, pOut+outIndex+11*oPixels) ;
	    _ve_vst_vss(vrsumC, 8, pOut+outIndex+12*oPixels) ;
	    _ve_vst_vss(vrsumD, 8, pOut+outIndex+13*oPixels) ;
	    _ve_vst_vss(vrsumE, 8, pOut+outIndex+14*oPixels) ;
	    _ve_vst_vss(vrsumF, 8, pOut+outIndex+15*oPixels) ;

	    outIndex += 2*vl1 ;
	  } // outPixels

	} // outChannel
      } // group
    } // batch
  }

  return VEDNN_SUCCESS;
}

