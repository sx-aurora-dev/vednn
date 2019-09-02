#include <stdint.h>
#include <stdlib.h>

#include "vednn.h"

#include "velintrin.h"
#define VLEN	(256)

static inline void k1(
  const float * restrict pIn,
  const float * restrict pKernel,
  float * restrict const pOut,
  const int64_t inChannel,
  const int64_t inWidth,
  const int64_t inHeight,
  const int64_t outChannel,
  const int64_t outWidth,
  const int64_t outHeight,
  const int64_t kernWidth,
  const int64_t kernHeight,
  const int64_t inChannelGroup,
  const int64_t outChannelGroup,
  const int64_t strideHeight,
  const int64_t strideWidth,
  const int64_t padHeight,
  const int64_t padWidth,
  const int64_t dilationHeight,
  const int64_t dilationWidth,
  const int64_t inGroupOffset,
  const int64_t outGroupOffset,
  const int64_t kernGroupOffset,
  const int64_t n,
  const int64_t k
)
{
  int64_t outIndex0 = outGroupOffset + (n * outChannel + k  ) * outHeight*outWidth ;

  for (int64_t y=0; y<outHeight; y++) {
    int64_t i = y * strideHeight - padHeight;

    for ( int64_t x0=0; x0<outWidth; x0+=VLEN) {

      const int64_t vl = outWidth - x0 < VLEN ? outWidth - x0 : VLEN ;

      __vr vrsum = _vel_vbrds_vsl(0.f, vl) ;

      __vr vrseq = _vel_vseq_vl(vl) ;
      __vr vrj   = _vel_vaddsl_vsvl(-padWidth,  _vel_vmulsl_vsvl(strideWidth, _vel_vaddsl_vsvl(x0, vrseq, vl), vl), vl) ;

      const float *pKerValue = pKernel + kernGroupOffset + k * inChannelGroup * kernHeight * kernWidth ;

      for (int64_t r = 0; r < kernHeight; r++) {
	for (int64_t s = 0; s < kernWidth; s++) {

	  const int64_t h = i + r * dilationHeight;
	  if ( !(h < 0 || inHeight <= h) ) {
	    __vr vrw = _vel_vaddsl_vsvl(s*dilationWidth,  vrj, vl) ;

	    __vm256 vm2 =  _vel_vfmklge_mvl(vrw, vl) ;				// condition(0 <= w)
	    __vm256 vm3 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(inWidth,vrw, vl), vl) ;	// condition(w < inWidth)

	    __vm256 vm23  = _vel_andm_mmm(vm2, vm3) ;

	    for (int64_t c = 0; c < inChannelGroup; c++) {
	      const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;

#if 0
	      __vr vrpin = _vel_vsfa_vvssl(vrw, 2, (uint64_t)&pInChannel[h*inWidth], vl) ;
	      __vr vrin = _vel_vgtu_vvssml(vrpin, 0, 0, vm23, vl) ;
#else
	      /* memory access errors mihgt be caused */
	      __vr vrin = _vel_vldu_vssl(4*strideWidth,&pInChannel[h*inWidth+x0*strideWidth+s*dilationWidth-padWidth], vl) ;
#endif
	      vrin = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrin, vm23, vl) ;
	      __vr vrinP = _vel_vshf_vvvsl(vrin, vrin, VE_VSHUFFLE_YUZU, vl) ;

	      const float *pKerValue = pKernel + kernGroupOffset + ((k * inChannelGroup + c) * kernHeight + r) * kernWidth + s;


	      vrsum = _vel_vfmads_vvsvl(vrsum, pKerValue[0], vrinP, vl) ;
	    } // inChannel
	  }
	} // kernWidth
      } // kernHeight

      _vel_vstu_vssl(vrsum, 4, pOut+outIndex0, vl) ;

      outIndex0 += vl ;
    } // x
  } // y
}


static inline void k2(
  const float * restrict pIn,
  const float * restrict pKernel,
  float * restrict const pOut,
  const int64_t inChannel,
  const int64_t inWidth,
  const int64_t inHeight,
  const int64_t outChannel,
  const int64_t outWidth,
  const int64_t outHeight,
  const int64_t kernWidth,
  const int64_t kernHeight,
  const int64_t inChannelGroup,
  const int64_t outChannelGroup,
  const int64_t strideHeight,
  const int64_t strideWidth,
  const int64_t padHeight,
  const int64_t padWidth,
  const int64_t dilationHeight,
  const int64_t dilationWidth,
  const int64_t inGroupOffset,
  const int64_t outGroupOffset,
  const int64_t kernGroupOffset,
  const int64_t n,
  const int64_t k
)
{
  int64_t outIndex0 = outGroupOffset + (n * outChannel + k  ) * outHeight*outWidth ;
  int64_t outIndex1 = outGroupOffset + (n * outChannel + k+1) * outHeight*outWidth ;

  for (int64_t y=0; y<outHeight; y++) {
    int64_t i = y * strideHeight - padHeight;

    for ( int64_t x0=0; x0<outWidth; x0+=VLEN) {

      const int64_t vl = outWidth - x0 < VLEN ? outWidth - x0 : VLEN ;

      __vr vrsum01 = _vel_pvbrd_vsl(0UL, vl) ;

      __vr vrseq = _vel_vseq_vl(vl) ;
      __vr vrj   = _vel_vaddsl_vsvl(-padWidth,  _vel_vmulsl_vsvl(strideWidth, _vel_vaddsl_vsvl(x0, vrseq, vl), vl), vl) ;

      const float *pKerValue = pKernel + kernGroupOffset + k * inChannelGroup * kernHeight * kernWidth ;

      for (int64_t r = 0; r < kernHeight; r++) {
	for (int64_t s = 0; s < kernWidth; s++) {

	  const int64_t h = i + r * dilationHeight;
	  if ( !(h < 0 || inHeight <= h) ) {
	    __vr vrw = _vel_vaddsl_vsvl(s*dilationWidth,  vrj, vl) ;

	    __vm256 vm2 =  _vel_vfmklge_mvl(vrw, vl) ;				// condition(0 <= w)
	    __vm256 vm3 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(inWidth,vrw, vl), vl) ;	// condition(w < inWidth)

	    __vm256 vm23  = _vel_andm_mmm(vm2, vm3) ;

	    for (int64_t c = 0; c < inChannelGroup; c++) {
	      const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;

#if 0
	      __vr vrpin = _vel_vsfa_vvssl(vrw, 2, (uint64_t)&pInChannel[h*inWidth], vl) ;
	      __vr vrin = _vel_vgtu_vvssml(vrpin, 0, 0, vm23, vl) ;
#else
	      /* memory access errors mihgt be caused */
	      __vr vrin = _vel_vldu_vssl(4*strideWidth,&pInChannel[h*inWidth+x0*strideWidth+s*dilationWidth-padWidth], vl) ;
#endif
	      vrin = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrin, vm23, vl) ;
	      __vr vrinP = _vel_vshf_vvvsl(vrin, vrin, VE_VSHUFFLE_YUZU, vl) ;

	      const float *pKerValue = pKernel + kernGroupOffset + ((k * inChannelGroup + c) * kernHeight + r) * kernWidth + s;

	      const uint64_t kerValue01 = _vel_pack_f32p(pKerValue,
		                                        pKerValue+      inChannelGroup * kernHeight * kernWidth) ;

	      vrsum01 = _vel_pvfmad_vvsvl(vrsum01, kerValue01, vrinP, vl) ;
	    } // inChannel
	  }
	} // kernWidth
      } // kernHeight

      _vel_vstu_vssl(vrsum01, 4, pOut+outIndex0, vl) ;
      _vel_vstl_vssl(vrsum01, 4, pOut+outIndex1, vl) ;

      outIndex0 += vl ;
      outIndex1 += vl ;
    } // x
  } // y
}


static inline void k4(
  const float * restrict pIn,
  const float * restrict pKernel,
  float * restrict const pOut,
  const int64_t inChannel,
  const int64_t inWidth,
  const int64_t inHeight,
  const int64_t outChannel,
  const int64_t outWidth,
  const int64_t outHeight,
  const int64_t kernWidth,
  const int64_t kernHeight,
  const int64_t inChannelGroup,
  const int64_t outChannelGroup,
  const int64_t strideHeight,
  const int64_t strideWidth,
  const int64_t padHeight,
  const int64_t padWidth,
  const int64_t dilationHeight,
  const int64_t dilationWidth,
  const int64_t inGroupOffset,
  const int64_t outGroupOffset,
  const int64_t kernGroupOffset,
  const int64_t n,
  const int64_t k
)
{
  int64_t outIndex0 = outGroupOffset + (n * outChannel + k  ) * outHeight*outWidth ;
  int64_t outIndex1 = outGroupOffset + (n * outChannel + k+1) * outHeight*outWidth ;
  int64_t outIndex2 = outGroupOffset + (n * outChannel + k+2) * outHeight*outWidth ;
  int64_t outIndex3 = outGroupOffset + (n * outChannel + k+3) * outHeight*outWidth ;

  for (int64_t y=0; y<outHeight; y++) {
    int64_t i = y * strideHeight - padHeight;

    for ( int64_t x0=0; x0<outWidth; x0+=VLEN) {

      const int64_t vl = outWidth - x0 < VLEN ? outWidth - x0 : VLEN ;

      __vr vrsum01 = _vel_pvbrd_vsl(0UL, vl) ;
      __vr vrsum23 = _vel_pvbrd_vsl(0UL, vl) ;

      __vr vrseq = _vel_vseq_vl(vl) ;
      __vr vrj   = _vel_vaddsl_vsvl(-padWidth,  _vel_vmulsl_vsvl(strideWidth, _vel_vaddsl_vsvl(x0, vrseq, vl), vl), vl) ;

      const float *pKerValue = pKernel + kernGroupOffset + k * inChannelGroup * kernHeight * kernWidth ;

      for (int64_t r = 0; r < kernHeight; r++) {
	for (int64_t s = 0; s < kernWidth; s++) {

	  const int64_t h = i + r * dilationHeight;
	  if ( !(h < 0 || inHeight <= h) ) {
	    __vr vrw = _vel_vaddsl_vsvl(s*dilationWidth,  vrj, vl) ;

	    __vm256 vm2 =  _vel_vfmklge_mvl(vrw, vl) ;				// condition(0 <= w)
	    __vm256 vm3 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(inWidth,vrw, vl), vl) ;	// condition(w < inWidth)

	    __vm256 vm23  = _vel_andm_mmm(vm2, vm3) ;

	    for (int64_t c = 0; c < inChannelGroup; c++) {
	      const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;

#if 0
	      __vr vrpin = _vel_vsfa_vvssl(vrw, 2, (uint64_t)&pInChannel[h*inWidth], vl) ;
	      __vr vrin = _vel_vgtu_vvssml(vrpin, 0, 0, vm23, vl) ;
#else
	      /* memory access errors mihgt be caused */
	      __vr vrin = _vel_vldu_vssl(4*strideWidth,&pInChannel[h*inWidth+x0*strideWidth+s*dilationWidth-padWidth], vl) ;
#endif
	      vrin = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrin, vm23, vl) ;
	      __vr vrinP = _vel_vshf_vvvsl(vrin, vrin, VE_VSHUFFLE_YUZU, vl) ;

	      const float *pKerValue = pKernel + kernGroupOffset + ((k * inChannelGroup + c) * kernHeight + r) * kernWidth + s;

	      const uint64_t kerValue01 = _vel_pack_f32p(pKerValue,
		                                        pKerValue+      inChannelGroup * kernHeight * kernWidth) ;
	      const uint64_t kerValue23 = _vel_pack_f32p(pKerValue + 2 * inChannelGroup * kernHeight * kernWidth,
		                                        pKerValue + 3 * inChannelGroup * kernHeight * kernWidth) ;

	      vrsum01 = _vel_pvfmad_vvsvl(vrsum01, kerValue01, vrinP, vl) ;
	      vrsum23 = _vel_pvfmad_vvsvl(vrsum23, kerValue23, vrinP, vl) ;
	    } // inChannel
	  }
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
    } // x
  } // y
}


static inline void k8(
  const float * restrict pIn,
  const float * restrict pKernel,
  float * restrict const pOut,
  const int64_t inChannel,
  const int64_t inWidth,
  const int64_t inHeight,
  const int64_t outChannel,
  const int64_t outWidth,
  const int64_t outHeight,
  const int64_t kernWidth,
  const int64_t kernHeight,
  const int64_t inChannelGroup,
  const int64_t outChannelGroup,
  const int64_t strideHeight,
  const int64_t strideWidth,
  const int64_t padHeight,
  const int64_t padWidth,
  const int64_t dilationHeight,
  const int64_t dilationWidth,
  const int64_t inGroupOffset,
  const int64_t outGroupOffset,
  const int64_t kernGroupOffset,
  const int64_t n,
  const int64_t k
)
{
  int64_t outIndex = outGroupOffset + (n * outChannel + k  ) * outHeight*outWidth ;

  for (int64_t y=0; y<outHeight; y++) {
    int64_t i = y * strideHeight - padHeight;

    for ( int64_t x0=0; x0<outWidth; x0+=VLEN) {

      const int64_t vl = outWidth - x0 < VLEN ? outWidth - x0 : VLEN ;

      __vr vrsum01 = _vel_pvbrd_vsl(0UL, vl) ;
      __vr vrsum23 = _vel_pvbrd_vsl(0UL, vl) ;
      __vr vrsum45 = _vel_pvbrd_vsl(0UL, vl) ;
      __vr vrsum67 = _vel_pvbrd_vsl(0UL, vl) ;

      __vr vrseq = _vel_vseq_vl(vl) ;
      __vr vrj   = _vel_vaddsl_vsvl(-padWidth,  _vel_vmulsl_vsvl(strideWidth, _vel_vaddsl_vsvl(x0, vrseq, vl), vl), vl) ;

      const float *pKerValue = pKernel + kernGroupOffset + k * inChannelGroup * kernHeight * kernWidth ;

      for (int64_t r = 0; r < kernHeight; r++)
      {
	for (int64_t s = 0; s < kernWidth; s++) {

	  const int64_t h = i + r * dilationHeight;
	  if ( !(h < 0 || inHeight <= h) ) {
	    __vr vrw = _vel_vaddsl_vsvl(s*dilationWidth,  vrj, vl) ;

	    __vm256 vm2 =  _vel_vfmklge_mvl(vrw, vl) ;				// condition(0 <= w)
	    __vm256 vm3 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(inWidth,vrw, vl), vl) ;	// condition(w < inWidth)

	    __vm256 vm23  = _vel_andm_mmm(vm2, vm3) ;

	    for (int64_t c = 0; c < inChannelGroup; c++) {
	      const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;

#if 0
	      __vr vrpin = _vel_vsfa_vvssl(vrw, 2, (uint64_t)&pInChannel[h*inWidth], vl) ;
	      __vr vrin = _vel_vgtu_vvssml(vrpin, 0, 0, vm23, vl) ;
#else
	      /* memory access errors might be caused */
	      __vr vrin = _vel_vldu_vssl(4*strideWidth,&pInChannel[h*inWidth+x0*strideWidth+s*dilationWidth-padWidth], vl) ;
#endif
	      vrin = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrin, vm23, vl) ;
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
	  }
	} // kernWidth
      } // kernHeight

      _vel_vstu_vssl(vrsum01, 4, pOut+outIndex + 0 * outHeight*outWidth, vl) ;
      _vel_vstl_vssl(vrsum01, 4, pOut+outIndex + 1 * outHeight*outWidth, vl) ;
      _vel_vstu_vssl(vrsum23, 4, pOut+outIndex + 2 * outHeight*outWidth, vl) ;
      _vel_vstl_vssl(vrsum23, 4, pOut+outIndex + 3 * outHeight*outWidth, vl) ;
      _vel_vstu_vssl(vrsum45, 4, pOut+outIndex + 4 * outHeight*outWidth, vl) ;
      _vel_vstl_vssl(vrsum45, 4, pOut+outIndex + 5 * outHeight*outWidth, vl) ;
      _vel_vstu_vssl(vrsum67, 4, pOut+outIndex + 6 * outHeight*outWidth, vl) ;
      _vel_vstl_vssl(vrsum67, 4, pOut+outIndex + 7 * outHeight*outWidth, vl) ;

      outIndex += vl ;
    } // x
  } // y
}

static inline void k16(
  const float * restrict pIn,
  const float * restrict pKernel,
  float * restrict const pOut,
  const int64_t inChannel,
  const int64_t inWidth,
  const int64_t inHeight,
  const int64_t outChannel,
  const int64_t outWidth,
  const int64_t outHeight,
  const int64_t kernWidth,
  const int64_t kernHeight,
  const int64_t inChannelGroup,
  const int64_t outChannelGroup,
  const int64_t strideHeight,
  const int64_t strideWidth,
  const int64_t padHeight,
  const int64_t padWidth,
  const int64_t dilationHeight,
  const int64_t dilationWidth,
  const int64_t inGroupOffset,
  const int64_t outGroupOffset,
  const int64_t kernGroupOffset,
  const int64_t n,
  const int64_t k
)
{
  int64_t outIndex = outGroupOffset + (n * outChannel + k  ) * outHeight * outWidth ;

  for (int64_t y=0; y<outHeight; y++) {
    int64_t i = y * strideHeight - padHeight;

    for ( int64_t x0=0; x0<outWidth; x0+=VLEN) {

      const int64_t vl = outWidth - x0 < VLEN ? outWidth - x0 : VLEN ;

      __vr vrsum01 = _vel_pvbrd_vsl(0UL, vl) ;
      __vr vrsum23 = _vel_pvbrd_vsl(0UL, vl) ;
      __vr vrsum45 = _vel_pvbrd_vsl(0UL, vl) ;
      __vr vrsum67 = _vel_pvbrd_vsl(0UL, vl) ;
      __vr vrsum89 = _vel_pvbrd_vsl(0UL, vl) ;
      __vr vrsumAB = _vel_pvbrd_vsl(0UL, vl) ;
      __vr vrsumCD = _vel_pvbrd_vsl(0UL, vl) ;
      __vr vrsumEF = _vel_pvbrd_vsl(0UL, vl) ;

      __vr vrseq = _vel_vseq_vl(vl) ;
      __vr vrj   = _vel_vaddsl_vsvl(-padWidth,  _vel_vmulsl_vsvl(strideWidth, _vel_vaddsl_vsvl(x0, vrseq, vl), vl), vl) ;

      const float *pKerValue = pKernel + kernGroupOffset + k * inChannelGroup * kernHeight * kernWidth ;

      for (int64_t r = 0; r < kernHeight; r++)
      {
	for (int64_t s = 0; s < kernWidth; s++) {

	  const int64_t h = i + r * dilationHeight;
	  if ( !(h < 0 || inHeight <= h) ) {
	    __vr vrw = _vel_vaddsl_vsvl(s*dilationWidth,  vrj, vl) ;

	    __vm256 vm2 =  _vel_vfmklge_mvl(vrw, vl) ;				// condition(0 <= w)
	    __vm256 vm3 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(inWidth,vrw, vl), vl) ;	// condition(w < inWidth)

	    __vm256 vm23  = _vel_andm_mmm(vm2, vm3) ;

	    for (int64_t c = 0; c < inChannelGroup; c++) {
	      const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;

#if 0
	      __vr vrpin = _vel_vsfa_vvssl(vrw, 2, (uint64_t)&pInChannel[h*inWidth], vl) ;
	      __vr vrin = _vel_vgtu_vvssml(vrpin, 0, 0, vm23, vl) ;
#else
	      /* memory access errors might be caused */
	      __vr vrin = _vel_vldu_vssl(4*strideWidth,&pInChannel[h*inWidth+x0*strideWidth+s*dilationWidth-padWidth], vl) ;
#endif
	      vrin = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrin, vm23, vl) ;
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
	  }
	} // kernWidth
      } // kernHeight

      _vel_vstu_vssl(vrsum01, 4, pOut+outIndex + 0 * outHeight*outWidth, vl) ;
      _vel_vstl_vssl(vrsum01, 4, pOut+outIndex + 1 * outHeight*outWidth, vl) ;
      _vel_vstu_vssl(vrsum23, 4, pOut+outIndex + 2 * outHeight*outWidth, vl) ;
      _vel_vstl_vssl(vrsum23, 4, pOut+outIndex + 3 * outHeight*outWidth, vl) ;
      _vel_vstu_vssl(vrsum45, 4, pOut+outIndex + 4 * outHeight*outWidth, vl) ;
      _vel_vstl_vssl(vrsum45, 4, pOut+outIndex + 5 * outHeight*outWidth, vl) ;
      _vel_vstu_vssl(vrsum67, 4, pOut+outIndex + 6 * outHeight*outWidth, vl) ;
      _vel_vstl_vssl(vrsum67, 4, pOut+outIndex + 7 * outHeight*outWidth, vl) ;
      _vel_vstu_vssl(vrsum89, 4, pOut+outIndex + 8 * outHeight*outWidth, vl) ;
      _vel_vstl_vssl(vrsum89, 4, pOut+outIndex + 9 * outHeight*outWidth, vl) ;
      _vel_vstu_vssl(vrsumAB, 4, pOut+outIndex +10 * outHeight*outWidth, vl) ;
      _vel_vstl_vssl(vrsumAB, 4, pOut+outIndex +11 * outHeight*outWidth, vl) ;
      _vel_vstu_vssl(vrsumCD, 4, pOut+outIndex +12 * outHeight*outWidth, vl) ;
      _vel_vstl_vssl(vrsumCD, 4, pOut+outIndex +13 * outHeight*outWidth, vl) ;
      _vel_vstu_vssl(vrsumEF, 4, pOut+outIndex +14 * outHeight*outWidth, vl) ;
      _vel_vstl_vssl(vrsumEF, 4, pOut+outIndex +15 * outHeight*outWidth, vl) ;


      outIndex += vl ;
    } // x
  } // y
}

vednnError_t
vednnConvolutionForward_direct_default(
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

  {
    for (int64_t n = 0; n < batch; n++) {
      for (int64_t g = 0; g < group; g++) {
	const int64_t inGroupOffset   = g * inChannelGroup * inHeight * inWidth;
	const int64_t outGroupOffset  = g * outChannelGroup * outHeight * outWidth;
	const int64_t kernGroupOffset = g * outChannelGroup * inChannelGroup * kernHeight * kernWidth;

	int k = 0 ;
	if ( (outChannelGroup & 0x01) == 1 ) {
	  k1(pIn, pKernel, pOut,
	     inChannel, inWidth, inHeight,
	     outChannel, outWidth, outHeight,
	     kernWidth, kernHeight,
	     inChannelGroup, outChannelGroup,
	     strideHeight, strideWidth,
	     padHeight, padWidth,
	     dilationHeight, dilationWidth,
	     inGroupOffset, outGroupOffset, kernGroupOffset,
	     n, k );

	  k++ ;
	}
	if ( ((outChannelGroup >> 1) & 0x01) == 1 ) {
	  k2(pIn, pKernel, pOut,
	     inChannel, inWidth, inHeight,
	     outChannel, outWidth, outHeight,
	     kernWidth, kernHeight,
	     inChannelGroup, outChannelGroup,
	     strideHeight, strideWidth,
	     padHeight, padWidth,
	     dilationHeight, dilationWidth,
	     inGroupOffset, outGroupOffset, kernGroupOffset,
	     n, k );

	  k+=2 ;
	}
	if ( ((outChannelGroup >> 2) & 0x01) == 1 ) {
	  k4(pIn, pKernel, pOut,
	     inChannel, inWidth, inHeight,
	     outChannel, outWidth, outHeight,
	     kernWidth, kernHeight,
	     inChannelGroup, outChannelGroup,
	     strideHeight, strideWidth,
	     padHeight, padWidth,
	     dilationHeight, dilationWidth,
	     inGroupOffset, outGroupOffset, kernGroupOffset,
	     n, k );

	  k+=4 ;
	}
	if ( ((outChannelGroup >> 3) & 0x01) == 1 ) {
	  k8(pIn, pKernel, pOut,
	     inChannel, inWidth, inHeight,
	     outChannel, outWidth, outHeight,
	     kernWidth, kernHeight,
	     inChannelGroup, outChannelGroup,
	     strideHeight, strideWidth,
	     padHeight, padWidth,
	     dilationHeight, dilationWidth,
	     inGroupOffset, outGroupOffset, kernGroupOffset,
	     n, k );

	  k+=8 ;
	}
	for (; k < outChannelGroup; k+=16) {
	  k16(pIn, pKernel, pOut,
	     inChannel, inWidth, inHeight,
	     outChannel, outWidth, outHeight,
	     kernWidth, kernHeight,
	     inChannelGroup, outChannelGroup,
	     strideHeight, strideWidth,
	     padHeight, padWidth,
	     dilationHeight, dilationWidth,
	     inGroupOffset, outGroupOffset, kernGroupOffset,
	     n, k );
	} // outChannel
      } // group
    } // batch
  }

  return VEDNN_SUCCESS;
}
