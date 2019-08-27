#include <stdint.h>
#include <stdlib.h>

#include "vednn.h"

#include "velintrin.h"
#define VLEN	(256)

vednnError_t
vednnConvolutionForward_direct_dil1_str1_padsame_ker5_owU128(
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
  float * restrict const pOut    = pDataOut;

  const int oPixels= outHeight*outWidth ;

  {
    const int64_t nY = VLEN / outWidth ;

    __vr vrseq = _vel_vseq_vl(VLEN) ;			// xy
    __vr vry   = _vel_vdivsl_vvsl(vrseq, outWidth, VLEN) ;
    __vr vrx   = _vel_vsubsl_vvvl(vrseq, _vel_vmulul_vsvl(outWidth,vry, VLEN), VLEN) ;

    __vm256 vmw_s0 =  _vel_vfmklge_mvl(_vel_vaddsl_vsvl(-2,  vrx, VLEN), VLEN) ;		// condition(0 <= w)
    __vm256 vmw_s1 =  _vel_vfmklge_mvl(_vel_vaddsl_vsvl(-1,  vrx, VLEN), VLEN) ;		// condition(0 <= w)

    __vm256 vmw_s3 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(inWidth-1,vrx, VLEN), VLEN) ;	// condition(w < inWidth)
    __vm256 vmw_s4 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(inWidth-2,vrx, VLEN), VLEN) ;	// condition(w < inWidth)

    for (int64_t n = 0; n < batch; n++) {
      for (int64_t g = 0; g < group; g++) {
	const int64_t inGroupOffset   = g * inChannelGroup * inHeight * inWidth;
	const int64_t outGroupOffset  = g * outChannelGroup * outHeight * outWidth;
	const int64_t kernGroupOffset = g * outChannelGroup * inChannelGroup * kernHeight * kernWidth;

	int k = 0 ;
	if ( (outChannelGroup & 0x01) == 1 ) {
	  int64_t outIndex = outGroupOffset + (n * outChannel + k) * oPixels ;

	  for (int64_t y=0; y<outHeight; y+=nY)
	  {
	    const int64_t vl = outWidth * (outHeight - y < nY ? outHeight - y : nY) ;
	    const int64_t op = y * outWidth ;

	    __vr vrsum = _vel_vbrds_vsl(0.0f, vl) ;

	    for (int64_t r = 0; r < kernHeight; r++) {
	      __vr vrh = _vel_vaddsl_vsvl(y+r-padHeight, vry, vl) ;

	      __vm256 vmh0 = _vel_vfmklge_mvl(vrh, vl) ;				// condition(0 <= h)
	      __vm256 vmh1 = _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(inHeight,vrh, vl), vl) ;	// condition(h < inHeight)
	      __vm256 vmh  = _vel_andm_mmm(vmh0, vmh1) ;

	      __vm256 vmall_s0 = _vel_andm_mmm(vmh,vmw_s0) ;
	      __vm256 vmall_s1 = _vel_andm_mmm(vmh,vmw_s1) ;
	      __vm256 vmall_s2 = vmh ;
	      __vm256 vmall_s3 = _vel_andm_mmm(vmh,vmw_s3) ;
	      __vm256 vmall_s4 = _vel_andm_mmm(vmh,vmw_s4) ;

	      for (int64_t c = 0; c < inChannelGroup; c++) {
		const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;

		/* memory access errors mihgt be caused */
		__vr vrin_s0 = _vel_vldu_vssl(4,&pInChannel[op+(r-padHeight)*inWidth-2], vl) ;
		__vr vrin_s1 = _vel_vldu_vssl(4,&pInChannel[op+(r-padHeight)*inWidth-1], vl) ;
		__vr vrin_s2 = _vel_vldu_vssl(4,&pInChannel[op+(r-padHeight)*inWidth  ], vl) ;
		__vr vrin_s3 = _vel_vldu_vssl(4,&pInChannel[op+(r-padHeight)*inWidth+1], vl) ;
		__vr vrin_s4 = _vel_vldu_vssl(4,&pInChannel[op+(r-padHeight)*inWidth+2], vl) ;

		const float *pKerValue = pKernel + kernGroupOffset + ((k * inChannelGroup + c) * kernHeight + r) * kernWidth ;

#define VFMAD1(VRIN, VMR, PKERVALUE) {				\
  VRIN = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), VRIN, VMR, vl) ;	\
  vrsum = _vel_vfmads_vvsvl(vrsum, *(PKERVALUE), VRIN, vl) ;		\
}
		VFMAD1(vrin_s0, vmall_s0, pKerValue) ; pKerValue++ ;
		VFMAD1(vrin_s1, vmall_s1, pKerValue) ; pKerValue++ ;
		VFMAD1(vrin_s2, vmall_s2, pKerValue) ; pKerValue++ ;
		VFMAD1(vrin_s3, vmall_s3, pKerValue) ; pKerValue++ ;
		VFMAD1(vrin_s4, vmall_s4, pKerValue) ; pKerValue++ ;
#undef VFMAD1

	      } // inChannel
	    } // kernHeight

	    _vel_vstu_vssl(vrsum, 4, pOut+outIndex, vl) ;

	    outIndex += vl ;
	  } // outPixels
	  k++ ;
	}
	if ( ((outChannelGroup >> 1) & 0x01) == 1 ) {
	  int64_t outIndex0 = outGroupOffset + (n * outChannel + k  ) * oPixels ;
	  int64_t outIndex1 = outGroupOffset + (n * outChannel + k+1) * oPixels ;

	  for (int64_t y=0; y<outHeight; y+=nY)
	  {
	    const int64_t vl = outWidth * (outHeight - y < nY ? outHeight - y : nY) ;
	    const int64_t op = y * outWidth ;

	    __vr vrsum01 = _vel_pvbrd_vsl(0UL, vl) ;

	    for (int64_t r = 0; r < kernHeight; r++) {
	      __vr vrh = _vel_vaddsl_vsvl(y+r-padHeight, vry, vl) ;

	      __vm256 vmh0 = _vel_vfmklge_mvl(vrh, vl) ;				// condition(0 <= h)
	      __vm256 vmh1 = _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(inHeight,vrh, vl), vl) ;	// condition(h < inHeight)
	      __vm256 vmh  = _vel_andm_mmm(vmh0, vmh1) ;

	      __vm256 vmall_s0 = _vel_andm_mmm(vmh,vmw_s0) ;
	      __vm256 vmall_s1 = _vel_andm_mmm(vmh,vmw_s1) ;
	      __vm256 vmall_s2 = vmh ;
	      __vm256 vmall_s3 = _vel_andm_mmm(vmh,vmw_s3) ;
	      __vm256 vmall_s4 = _vel_andm_mmm(vmh,vmw_s4) ;

	      for (int64_t c = 0; c < inChannelGroup; c++) {
		const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;

		/* memory access errors mihgt be caused */
		__vr vrin_s0 = _vel_vldu_vssl(4,&pInChannel[op+(r-padHeight)*inWidth-2], vl) ;
		__vr vrin_s1 = _vel_vldu_vssl(4,&pInChannel[op+(r-padHeight)*inWidth-1], vl) ;
		__vr vrin_s2 = _vel_vldu_vssl(4,&pInChannel[op+(r-padHeight)*inWidth  ], vl) ;
		__vr vrin_s3 = _vel_vldu_vssl(4,&pInChannel[op+(r-padHeight)*inWidth+1], vl) ;
		__vr vrin_s4 = _vel_vldu_vssl(4,&pInChannel[op+(r-padHeight)*inWidth+2], vl) ;

		const float *pKerValue = pKernel + kernGroupOffset + ((k * inChannelGroup + c) * kernHeight + r) * kernWidth ;

#define PVFMAD2(VRIN, VMR, PKERVALUE) {									\
  VRIN = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), VRIN, VMR, vl) ;					\
  __vr vrinP = _vel_vshf_vvvsl(VRIN, VRIN, VE_VSHUFFLE_YUZU, vl) ;					\
  const uint64_t kerValue01 = _vel_pack_f32p(PKERVALUE,							\
					     PKERVALUE +     inChannelGroup * kernHeight * kernWidth) ;	\
  vrsum01 = _vel_pvfmad_vvsvl(vrsum01, kerValue01, vrinP, vl) ;						\
}
		PVFMAD2(vrin_s0, vmall_s0, pKerValue) ; pKerValue++ ;
		PVFMAD2(vrin_s1, vmall_s1, pKerValue) ; pKerValue++ ;
		PVFMAD2(vrin_s2, vmall_s2, pKerValue) ; pKerValue++ ;
		PVFMAD2(vrin_s3, vmall_s3, pKerValue) ; pKerValue++ ;
		PVFMAD2(vrin_s4, vmall_s4, pKerValue) ; pKerValue++ ;
#undef PVFMAD2

	      } // inChannel
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

	  for (int64_t y=0; y<outHeight; y+=nY)
	  {
	    const int64_t vl = outWidth * (outHeight - y < nY ? outHeight - y : nY) ;
	    const int64_t op = y * outWidth ;

	    __vr vrsum01 = _vel_pvbrd_vsl(0UL, vl) ;
	    __vr vrsum23 = _vel_pvbrd_vsl(0UL, vl) ;

	    for (int64_t r = 0; r < kernHeight; r++) {
	      __vr vrh = _vel_vaddsl_vsvl(y+r-padHeight, vry, vl) ;

	      __vm256 vmh0 = _vel_vfmklge_mvl(vrh, vl) ;				// condition(0 <= h)
	      __vm256 vmh1 = _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(inHeight,vrh, vl), vl) ;	// condition(h < inHeight)
	      __vm256 vmh  = _vel_andm_mmm(vmh0, vmh1) ;

	      __vm256 vmall_s0 = _vel_andm_mmm(vmh,vmw_s0) ;
	      __vm256 vmall_s1 = _vel_andm_mmm(vmh,vmw_s1) ;
	      __vm256 vmall_s2 = vmh ;
	      __vm256 vmall_s3 = _vel_andm_mmm(vmh,vmw_s3) ;
	      __vm256 vmall_s4 = _vel_andm_mmm(vmh,vmw_s4) ;

	      for (int64_t c = 0; c < inChannelGroup; c++) {
		const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;

		/* memory access errors mihgt be caused */
		__vr vrin_s0 = _vel_vldu_vssl(4,&pInChannel[op+(r-padHeight)*inWidth-2], vl) ;
		__vr vrin_s1 = _vel_vldu_vssl(4,&pInChannel[op+(r-padHeight)*inWidth-1], vl) ;
		__vr vrin_s2 = _vel_vldu_vssl(4,&pInChannel[op+(r-padHeight)*inWidth  ], vl) ;
		__vr vrin_s3 = _vel_vldu_vssl(4,&pInChannel[op+(r-padHeight)*inWidth+1], vl) ;
		__vr vrin_s4 = _vel_vldu_vssl(4,&pInChannel[op+(r-padHeight)*inWidth+2], vl) ;

		const float *pKerValue = pKernel + kernGroupOffset + ((k * inChannelGroup + c) * kernHeight + r) * kernWidth ;

#define PVFMAD4(VRIN, VMR, PKERVALUE) {									\
  VRIN = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), VRIN, VMR, vl) ;					\
  __vr vrinP = _vel_vshf_vvvsl(VRIN, VRIN, VE_VSHUFFLE_YUZU, vl) ;					\
  const uint64_t kerValue01 = _vel_pack_f32p(PKERVALUE,							\
					     PKERVALUE +     inChannelGroup * kernHeight * kernWidth) ;	\
  const uint64_t kerValue23 = _vel_pack_f32p(PKERVALUE + 2 * inChannelGroup * kernHeight * kernWidth,	\
					     PKERVALUE + 3 * inChannelGroup * kernHeight * kernWidth) ;	\
  vrsum01 = _vel_pvfmad_vvsvl(vrsum01, kerValue01, vrinP, vl) ;						\
  vrsum23 = _vel_pvfmad_vvsvl(vrsum23, kerValue23, vrinP, vl) ;						\
}
		PVFMAD4(vrin_s0, vmall_s0, pKerValue) ; pKerValue++ ;
		PVFMAD4(vrin_s1, vmall_s1, pKerValue) ; pKerValue++ ;
		PVFMAD4(vrin_s2, vmall_s2, pKerValue) ; pKerValue++ ;
		PVFMAD4(vrin_s3, vmall_s3, pKerValue) ; pKerValue++ ;
		PVFMAD4(vrin_s4, vmall_s4, pKerValue) ; pKerValue++ ;
#undef PVFMAD4

	      } // inChannel
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

	  for (int64_t y=0; y<outHeight; y+=nY)
	  {
	    const int64_t vl = outWidth * (outHeight - y < nY ? outHeight - y : nY) ;
	    const int64_t op = y * outWidth ;

	    __vr vrsum01 = _vel_pvbrd_vsl(0UL, vl) ;
	    __vr vrsum23 = _vel_pvbrd_vsl(0UL, vl) ;
	    __vr vrsum45 = _vel_pvbrd_vsl(0UL, vl) ;
	    __vr vrsum67 = _vel_pvbrd_vsl(0UL, vl) ;

	    for (int64_t r = 0; r < kernHeight; r++) {
	      __vr vrh = _vel_vaddsl_vsvl(y+r-padHeight, vry, vl) ;

	      __vm256 vmh0 = _vel_vfmklge_mvl(vrh, vl) ;				// condition(0 <= h)
	      __vm256 vmh1 = _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(inHeight,vrh, vl), vl) ;	// condition(h < inHeight)
	      __vm256 vmh  = _vel_andm_mmm(vmh0, vmh1) ;

	      __vm256 vmall_s0 = _vel_andm_mmm(vmh,vmw_s0) ;
	      __vm256 vmall_s1 = _vel_andm_mmm(vmh,vmw_s1) ;
	      __vm256 vmall_s2 = vmh ;
	      __vm256 vmall_s3 = _vel_andm_mmm(vmh,vmw_s3) ;
	      __vm256 vmall_s4 = _vel_andm_mmm(vmh,vmw_s4) ;

	      for (int64_t c = 0; c < inChannelGroup; c++) {
		const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;

		/* memory access errors mihgt be caused */
		__vr vrin_s0 = _vel_vldu_vssl(4,&pInChannel[op+(r-padHeight)*inWidth-2], vl) ;
		__vr vrin_s1 = _vel_vldu_vssl(4,&pInChannel[op+(r-padHeight)*inWidth-1], vl) ;
		__vr vrin_s2 = _vel_vldu_vssl(4,&pInChannel[op+(r-padHeight)*inWidth  ], vl) ;
		__vr vrin_s3 = _vel_vldu_vssl(4,&pInChannel[op+(r-padHeight)*inWidth+1], vl) ;
		__vr vrin_s4 = _vel_vldu_vssl(4,&pInChannel[op+(r-padHeight)*inWidth+2], vl) ;

		const float *pKerValue = pKernel + kernGroupOffset + ((k * inChannelGroup + c) * kernHeight + r) * kernWidth ;

#define PVFMAD8(VRIN, VMR, PKERVALUE) {									\
  VRIN = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), VRIN, VMR, vl) ;					\
  __vr vrinP = _vel_vshf_vvvsl(VRIN, VRIN, VE_VSHUFFLE_YUZU, vl) ;					\
  const uint64_t kerValue01 = _vel_pack_f32p(PKERVALUE,							\
					     PKERVALUE +     inChannelGroup * kernHeight * kernWidth) ;	\
  const uint64_t kerValue23 = _vel_pack_f32p(PKERVALUE + 2 * inChannelGroup * kernHeight * kernWidth,	\
					     PKERVALUE + 3 * inChannelGroup * kernHeight * kernWidth) ;	\
  const uint64_t kerValue45 = _vel_pack_f32p(PKERVALUE + 4 * inChannelGroup * kernHeight * kernWidth,	\
					     PKERVALUE + 5 * inChannelGroup * kernHeight * kernWidth) ;	\
  const uint64_t kerValue67 = _vel_pack_f32p(PKERVALUE + 6 * inChannelGroup * kernHeight * kernWidth,	\
					     PKERVALUE + 7 * inChannelGroup * kernHeight * kernWidth) ;	\
  vrsum01 = _vel_pvfmad_vvsvl(vrsum01, kerValue01, vrinP, vl) ;						\
  vrsum23 = _vel_pvfmad_vvsvl(vrsum23, kerValue23, vrinP, vl) ;						\
  vrsum45 = _vel_pvfmad_vvsvl(vrsum45, kerValue45, vrinP, vl) ;						\
  vrsum67 = _vel_pvfmad_vvsvl(vrsum67, kerValue67, vrinP, vl) ;						\
}
		PVFMAD8(vrin_s0, vmall_s0, pKerValue) ; pKerValue++ ;
		PVFMAD8(vrin_s1, vmall_s1, pKerValue) ; pKerValue++ ;
		PVFMAD8(vrin_s2, vmall_s2, pKerValue) ; pKerValue++ ;
		PVFMAD8(vrin_s3, vmall_s3, pKerValue) ; pKerValue++ ;
		PVFMAD8(vrin_s4, vmall_s4, pKerValue) ; pKerValue++ ;
#undef PVFMAD8
	      } // inChannel
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

