#include <stdint.h>
#include <stdlib.h>

#include "vednn.h"
#include "vednn_util.h"

#include "velintrin.h"
#define VLEN	(256)

template<filterLayout_t FLAYOUT>
static inline void k1(
  const float * __restrict__ pIn,
  const float * __restrict__ pKernel,
  float * __restrict__ const pOut,
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
  const int64_t inGroupOffset,
  const int64_t outGroupOffset,
  const int64_t kernGroupOffset,
  const int64_t oPixels,
  const int64_t nY,
  const __vr    vrij,
  const int64_t n,
  const int64_t k
)
{
  int64_t outIndex = outGroupOffset + (n * outChannel + k  ) * oPixels ;

  for (int64_t y=0; y<outHeight; y+=nY)
  {
    const int64_t vl = outWidth * (outHeight - y < nY ? outHeight - y : nY) ;
    const int64_t op = y * outWidth ;

    __vr vrsum = _vel_vbrds_vsl(0.f, vl) ;

    for (int64_t r = 0; r < kernHeight; r++) {
      for (int64_t s = 0; s < kernWidth; s++) {
	for (int64_t c = 0; c < inChannelGroup; c++) {
	  const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;
	  __vr vrpin = _vel_vsfa_vvssl(vrij, 2, (uint64_t)(pInChannel+(y*strideHeight+r)*inWidth+s), vl) ;

	  __vr vrin = _vel_vgtu_vvssl(vrpin, 0, 0, vl) ;

#define FILTER_OFFSET(k,c,r,s) ( kernGroupOffset + filter_index<FLAYOUT>(k,c,r,s, inChannelGroup, outChannelGroup, kernHeight, kernWidth) )
	  vrsum = _vel_vfmads_vvsvl(vrsum, pKernel[FILTER_OFFSET(k,c,r,s)], vrin, vl) ;
#undef FILTER_OFFSET
	} // inChannel
      } // kernWidth
    } // kernHeight

    _vel_vstu_vssl(vrsum, 4, pOut+outIndex, vl) ;

    outIndex += vl ;
  } // outPixels
}


template<filterLayout_t FLAYOUT>
static inline void k2(
  const float * __restrict__ pIn,
  const float * __restrict__ pKernel,
  float * __restrict__ const pOut,
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
  const int64_t inGroupOffset,
  const int64_t outGroupOffset,
  const int64_t kernGroupOffset,
  const int64_t oPixels,
  const int64_t nY,
  const __vr    vrij,
  const int64_t n,
  const int64_t k
)
{
  int64_t outIndex = outGroupOffset + (n * outChannel + k  ) * oPixels ;

  for (int64_t y=0; y<outHeight; y+=nY)
  {
    const int64_t vl = outWidth * (outHeight - y < nY ? outHeight - y : nY) ;
    const int64_t op = y * outWidth ;

    __vr vrsum01 = _vel_pvbrd_vsl(0UL, vl) ;

    for (int64_t r = 0; r < kernHeight; r++) {
      for (int64_t s = 0; s < kernWidth; s++) {
	for (int64_t c = 0; c < inChannelGroup; c++) {
	  const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;
	  __vr vrpin = _vel_vsfa_vvssl(vrij, 2, (uint64_t)(pInChannel+(y*strideHeight+r)*inWidth+s), vl) ;

	  __vr vrin = _vel_vgtu_vvssl(vrpin, 0, 0, vl) ;
	  __vr vrinP = _vel_vshf_vvvsl(vrin, vrin, VE_VSHUFFLE_YUZU, vl) ;

#define FILTER_OFFSET(k,c,r,s) ( kernGroupOffset + filter_index<FLAYOUT>(k,c,r,s, inChannelGroup, outChannelGroup, kernHeight, kernWidth) )

	  const uint64_t kerValue01 = _vel_pack_f32p(pKernel + FILTER_OFFSET(k+ 0,c,r,s),
						     pKernel + FILTER_OFFSET(k+ 1,c,r,s)) ;
	  vrsum01 = _vel_pvfmad_vvsvl(vrsum01, kerValue01, vrinP, vl) ;
#undef FILTER_OFFSET
	} // inChannel
      } // kernWidth
    } // kernHeight

    _vel_vstu_vssl(vrsum01, 4, pOut+outIndex, vl) ;
    _vel_vstl_vssl(vrsum01, 4, pOut+outIndex+ 1*oPixels, vl) ;

    outIndex += vl ;
  } // outPixels
}


template<filterLayout_t FLAYOUT>
static inline void k4(
  const float * __restrict__ pIn,
  const float * __restrict__ pKernel,
  float * __restrict__ const pOut,
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
  const int64_t inGroupOffset,
  const int64_t outGroupOffset,
  const int64_t kernGroupOffset,
  const int64_t oPixels,
  const int64_t nY,
  const __vr    vrij,
  const int64_t n,
  const int64_t k
)
{
  int64_t outIndex = outGroupOffset + (n * outChannel + k  ) * oPixels ;

  for (int64_t y=0; y<outHeight; y+=nY)
  {
    const int64_t vl = outWidth * (outHeight - y < nY ? outHeight - y : nY) ;
    const int64_t op = y * outWidth ;

    __vr vrsum01 = _vel_pvbrd_vsl(0UL, vl) ;
    __vr vrsum23 = _vel_pvbrd_vsl(0UL, vl) ;

    for (int64_t r = 0; r < kernHeight; r++) {
      for (int64_t s = 0; s < kernWidth; s++) {
	for (int64_t c = 0; c < inChannelGroup; c++) {
	  const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;
	  __vr vrpin = _vel_vsfa_vvssl(vrij, 2, (uint64_t)(pInChannel+(y*strideHeight+r)*inWidth+s), vl) ;

	  __vr vrin = _vel_vgtu_vvssl(vrpin, 0, 0, vl) ;
	  __vr vrinP = _vel_vshf_vvvsl(vrin, vrin, VE_VSHUFFLE_YUZU, vl) ;

#define FILTER_OFFSET(k,c,r,s) ( kernGroupOffset + filter_index<FLAYOUT>(k,c,r,s, inChannelGroup, outChannelGroup, kernHeight, kernWidth) )

	  const uint64_t kerValue01 = _vel_pack_f32p(pKernel + FILTER_OFFSET(k+ 0,c,r,s),
						     pKernel + FILTER_OFFSET(k+ 1,c,r,s)) ;
	  const uint64_t kerValue23 = _vel_pack_f32p(pKernel + FILTER_OFFSET(k+ 2,c,r,s),
						     pKernel + FILTER_OFFSET(k+ 3,c,r,s)) ;

	  vrsum01 = _vel_pvfmad_vvsvl(vrsum01, kerValue01, vrinP, vl) ;
	  vrsum23 = _vel_pvfmad_vvsvl(vrsum23, kerValue23, vrinP, vl) ;
#undef FILTER_OFFSET
	} // inChannel
      } // kernWidth
    } // kernHeight

    _vel_vstu_vssl(vrsum01, 4, pOut+outIndex, vl) ;
    _vel_vstl_vssl(vrsum01, 4, pOut+outIndex+ 1*oPixels, vl) ;
    _vel_vstu_vssl(vrsum23, 4, pOut+outIndex+ 2*oPixels, vl) ;
    _vel_vstl_vssl(vrsum23, 4, pOut+outIndex+ 3*oPixels, vl) ;

    outIndex += vl ;
  } // outPixels
}

template<filterLayout_t FLAYOUT>
static inline void k8(
  const float * __restrict__ pIn,
  const float * __restrict__ pKernel,
  float * __restrict__ const pOut,
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
  const int64_t inGroupOffset,
  const int64_t outGroupOffset,
  const int64_t kernGroupOffset,
  const int64_t oPixels,
  const int64_t nY,
  const __vr    vrij,
  const int64_t n,
  const int64_t k
)
{
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
	for (int64_t c = 0; c < inChannelGroup; c++) {
	  const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;
	  __vr vrpin = _vel_vsfa_vvssl(vrij, 2, (uint64_t)(pInChannel+(y*strideHeight+r)*inWidth+s), vl) ;

	  __vr vrin = _vel_vgtu_vvssl(vrpin, 0, 0, vl) ;
	  __vr vrinP = _vel_vshf_vvvsl(vrin, vrin, VE_VSHUFFLE_YUZU, vl) ;

#define FILTER_OFFSET(k,c,r,s) ( kernGroupOffset + filter_index<FLAYOUT>(k,c,r,s, inChannelGroup, outChannelGroup, kernHeight, kernWidth) )

	  const uint64_t kerValue01 = _vel_pack_f32p(pKernel + FILTER_OFFSET(k+ 0,c,r,s),
						     pKernel + FILTER_OFFSET(k+ 1,c,r,s)) ;
	  const uint64_t kerValue23 = _vel_pack_f32p(pKernel + FILTER_OFFSET(k+ 2,c,r,s),
						     pKernel + FILTER_OFFSET(k+ 3,c,r,s)) ;
	  const uint64_t kerValue45 = _vel_pack_f32p(pKernel + FILTER_OFFSET(k+ 4,c,r,s),
						     pKernel + FILTER_OFFSET(k+ 5,c,r,s)) ;
	  const uint64_t kerValue67 = _vel_pack_f32p(pKernel + FILTER_OFFSET(k+ 6,c,r,s),
						     pKernel + FILTER_OFFSET(k+ 7,c,r,s)) ;

	  vrsum01 = _vel_pvfmad_vvsvl(vrsum01, kerValue01, vrinP, vl) ;
	  vrsum23 = _vel_pvfmad_vvsvl(vrsum23, kerValue23, vrinP, vl) ;
	  vrsum45 = _vel_pvfmad_vvsvl(vrsum45, kerValue45, vrinP, vl) ;
	  vrsum67 = _vel_pvfmad_vvsvl(vrsum67, kerValue67, vrinP, vl) ;
#undef FILTER_OFFSET
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
}

template<filterLayout_t FLAYOUT>
static inline void k16(
  const float * __restrict__ pIn,
  const float * __restrict__ pKernel,
  float * __restrict__ const pOut,
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
  const int64_t inGroupOffset,
  const int64_t outGroupOffset,
  const int64_t kernGroupOffset,
  const int64_t oPixels,
  const int64_t nY,
  const __vr    vrij,
  const int64_t n,
  const int64_t k
)
{
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
	for (int64_t c = 0; c < inChannelGroup; c++) {
	  const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;
	  __vr vrpin = _vel_vsfa_vvssl(vrij, 2, (uint64_t)(pInChannel+(y*strideHeight+r)*inWidth+s), vl) ;

	  __vr vrin = _vel_vgtu_vvssl(vrpin, 0, 0, vl) ;
	  __vr vrinP = _vel_vshf_vvvsl(vrin, vrin, VE_VSHUFFLE_YUZU, vl) ;

#define FILTER_OFFSET(k,c,r,s) ( kernGroupOffset + filter_index<FLAYOUT>(k,c,r,s, inChannelGroup, outChannelGroup, kernHeight, kernWidth) )

	  const uint64_t kerValue01 = _vel_pack_f32p(pKernel + FILTER_OFFSET(k+ 0,c,r,s),
						     pKernel + FILTER_OFFSET(k+ 1,c,r,s)) ;
	  const uint64_t kerValue23 = _vel_pack_f32p(pKernel + FILTER_OFFSET(k+ 2,c,r,s),
						     pKernel + FILTER_OFFSET(k+ 3,c,r,s)) ;
	  const uint64_t kerValue45 = _vel_pack_f32p(pKernel + FILTER_OFFSET(k+ 4,c,r,s),
						     pKernel + FILTER_OFFSET(k+ 5,c,r,s)) ;
	  const uint64_t kerValue67 = _vel_pack_f32p(pKernel + FILTER_OFFSET(k+ 6,c,r,s),
						     pKernel + FILTER_OFFSET(k+ 7,c,r,s)) ;
	  const uint64_t kerValue89 = _vel_pack_f32p(pKernel + FILTER_OFFSET(k+ 8,c,r,s),
						     pKernel + FILTER_OFFSET(k+ 9,c,r,s) ) ;
	  const uint64_t kerValueAB = _vel_pack_f32p(pKernel + FILTER_OFFSET(k+10,c,r,s),
						     pKernel + FILTER_OFFSET(k+11,c,r,s) ) ;
	  const uint64_t kerValueCD = _vel_pack_f32p(pKernel + FILTER_OFFSET(k+12,c,r,s),
						     pKernel + FILTER_OFFSET(k+13,c,r,s) ) ;
	  const uint64_t kerValueEF = _vel_pack_f32p(pKernel + FILTER_OFFSET(k+14,c,r,s),
						     pKernel + FILTER_OFFSET(k+15,c,r,s)) ;

	  vrsum01 = _vel_pvfmad_vvsvl(vrsum01, kerValue01, vrinP, vl) ;
	  vrsum23 = _vel_pvfmad_vvsvl(vrsum23, kerValue23, vrinP, vl) ;
	  vrsum45 = _vel_pvfmad_vvsvl(vrsum45, kerValue45, vrinP, vl) ;
	  vrsum67 = _vel_pvfmad_vvsvl(vrsum67, kerValue67, vrinP, vl) ;
	  vrsum89 = _vel_pvfmad_vvsvl(vrsum89, kerValue89, vrinP, vl) ;
	  vrsumAB = _vel_pvfmad_vvsvl(vrsumAB, kerValueAB, vrinP, vl) ;
	  vrsumCD = _vel_pvfmad_vvsvl(vrsumCD, kerValueCD, vrinP, vl) ;
	  vrsumEF = _vel_pvfmad_vvsvl(vrsumEF, kerValueEF, vrinP, vl) ;
#undef FILTER_OFFSET
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
}

extern "C" vednnError_t
vednnConvolutionForward_direct_dil1_pad0_owU128(
    const vednnTensorParam_t *  	pParamIn,
    const void *  			pDataIn,
    const vednnFilterParam_t *  	pParamKernel,
    const void *  			pDataKernel,
    const vednnConvolutionParam_t *  	pParamConv,
    const vednnTensorParam_t *  	pParamOut,
    void *  				pDataOut
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

  const int64_t filter_layout = pParamKernel->layout ;

  const int64_t group          = pParamConv->group;
  const int64_t strideWidth    = pParamConv->strideWidth;;
  const int64_t strideHeight   = pParamConv->strideHeight;
//  const int64_t padWidth       = pParamConv->padWidth;
//  const int64_t padHeight      = pParamConv->padHeight;
//  const int64_t dilationWidth  = pParamConv->dilationWidth;
//  const int64_t dilationHeight = pParamConv->dilationHeight;

  const int64_t inChannelGroup  = inChannel  / group;   // equal to pDataKernel->inChannel
  const int64_t outChannelGroup = outChannel / group;   // equal to pDataKernel->outChannel

  const float * pIn     = (const float *) pDataIn;
  const float * pKernel = (const float *) pDataKernel;
  float * const pOut    = (float * const) pDataOut;

  const int oPixels= outHeight*outWidth ;

  {
    const int64_t nY = VLEN / outWidth ;

    __vr vrseq = _vel_vseq_vl(nY*outWidth) ;
    __vr vry  = _vel_vdivsl_vvsl(vrseq, outWidth, nY*outWidth) ;
    __vr vrx  = _vel_vsubsl_vvvl(vrseq, _vel_vmulul_vsvl(outWidth,vry, nY*outWidth), nY*outWidth) ;

    __vr vri   = _vel_vmulsl_vsvl(strideHeight, vry, nY*outWidth) ;
    __vr vrj   = _vel_vmulsl_vsvl(strideWidth,  vrx, nY*outWidth) ;
    __vr vrij = _vel_vaddul_vvvl(vrj, _vel_vmulul_vsvl(inWidth, vri, nY*outWidth), nY*outWidth) ;

    for (int64_t n = 0; n < batch; n++) {
      for (int64_t g = 0; g < group; g++) {
	const int64_t inGroupOffset   = g * inChannelGroup * inHeight * inWidth;
	const int64_t outGroupOffset  = g * outChannelGroup * outHeight * outWidth;
	const int64_t kernGroupOffset = g * outChannelGroup * inChannelGroup * kernHeight * kernWidth;

	int k = 0 ;
	if ( (outChannelGroup & 0x01) == 1 ) {
	  if( filter_layout == VEDNN_FILTER_LAYOUT_NCHW) {
	    k1<VEDNN_FILTER_LAYOUT_NCHW>(pIn, pKernel, pOut,
	       inChannel, inWidth, inHeight,
	       outChannel, outWidth, outHeight,
	       kernWidth, kernHeight,
	       inChannelGroup, outChannelGroup,
	       strideHeight, strideWidth,
	       inGroupOffset, outGroupOffset, kernGroupOffset,
	       oPixels, nY,
	       vrij,
	       n, k );
	  }
	  else {
	    k1<VEDNN_FILTER_LAYOUT_HWCN>(pIn, pKernel, pOut,
	       inChannel, inWidth, inHeight,
	       outChannel, outWidth, outHeight,
	       kernWidth, kernHeight,
	       inChannelGroup, outChannelGroup,
	       strideHeight, strideWidth,
	       inGroupOffset, outGroupOffset, kernGroupOffset,
	       oPixels, nY,
	       vrij,
	       n, k );
	  }

	  k++ ;
	}
	if ( ((outChannelGroup >> 1) & 0x01) == 1 ) {
	  if( filter_layout == VEDNN_FILTER_LAYOUT_NCHW) {
	    k2<VEDNN_FILTER_LAYOUT_NCHW>(pIn, pKernel, pOut,
	       inChannel, inWidth, inHeight,
	       outChannel, outWidth, outHeight,
	       kernWidth, kernHeight,
	       inChannelGroup, outChannelGroup,
	       strideHeight, strideWidth,
	       inGroupOffset, outGroupOffset, kernGroupOffset,
	       oPixels, nY,
	       vrij,
	       n, k );
	  }
	  else {
	    k2<VEDNN_FILTER_LAYOUT_HWCN>(pIn, pKernel, pOut,
	       inChannel, inWidth, inHeight,
	       outChannel, outWidth, outHeight,
	       kernWidth, kernHeight,
	       inChannelGroup, outChannelGroup,
	       strideHeight, strideWidth,
	       inGroupOffset, outGroupOffset, kernGroupOffset,
	       oPixels, nY,
	       vrij,
	       n, k );
	  }

	  k+=2 ;
	}
	if ( ((outChannelGroup >> 2) & 0x01) == 1 ) {
	  if( filter_layout == VEDNN_FILTER_LAYOUT_NCHW) {
	    k4<VEDNN_FILTER_LAYOUT_NCHW>(pIn, pKernel, pOut,
	       inChannel, inWidth, inHeight,
	       outChannel, outWidth, outHeight,
	       kernWidth, kernHeight,
	       inChannelGroup, outChannelGroup,
	       strideHeight, strideWidth,
	       inGroupOffset, outGroupOffset, kernGroupOffset,
	       oPixels, nY,
	       vrij,
	       n, k );
	  }
	  else {
	    k4<VEDNN_FILTER_LAYOUT_HWCN>(pIn, pKernel, pOut,
	       inChannel, inWidth, inHeight,
	       outChannel, outWidth, outHeight,
	       kernWidth, kernHeight,
	       inChannelGroup, outChannelGroup,
	       strideHeight, strideWidth,
	       inGroupOffset, outGroupOffset, kernGroupOffset,
	       oPixels, nY,
	       vrij,
	       n, k );
	  }

	  k+=4 ;
	}
	if ( ((outChannelGroup >> 3) & 0x01) == 1 ) {
	  if( filter_layout == VEDNN_FILTER_LAYOUT_NCHW) {
	    k8<VEDNN_FILTER_LAYOUT_NCHW>(pIn, pKernel, pOut,
	       inChannel, inWidth, inHeight,
	       outChannel, outWidth, outHeight,
	       kernWidth, kernHeight,
	       inChannelGroup, outChannelGroup,
	       strideHeight, strideWidth,
	       inGroupOffset, outGroupOffset, kernGroupOffset,
	       oPixels, nY,
	       vrij,
	       n, k );
	  }
	  else {
	    k8<VEDNN_FILTER_LAYOUT_HWCN>(pIn, pKernel, pOut,
	       inChannel, inWidth, inHeight,
	       outChannel, outWidth, outHeight,
	       kernWidth, kernHeight,
	       inChannelGroup, outChannelGroup,
	       strideHeight, strideWidth,
	       inGroupOffset, outGroupOffset, kernGroupOffset,
	       oPixels, nY,
	       vrij,
	       n, k );
	  }

	  k+=8 ;
	}
	for (; k < outChannelGroup; k+=16) {
	  if( filter_layout == VEDNN_FILTER_LAYOUT_NCHW) {
	    k16<VEDNN_FILTER_LAYOUT_NCHW>(pIn, pKernel, pOut,
	       inChannel, inWidth, inHeight,
	       outChannel, outWidth, outHeight,
	       kernWidth, kernHeight,
	       inChannelGroup, outChannelGroup,
	       strideHeight, strideWidth,
	       inGroupOffset, outGroupOffset, kernGroupOffset,
	       oPixels, nY,
	       vrij,
	       n, k );
	  }
	  else {
	    k16<VEDNN_FILTER_LAYOUT_HWCN>(pIn, pKernel, pOut,
	       inChannel, inWidth, inHeight,
	       outChannel, outWidth, outHeight,
	       kernWidth, kernHeight,
	       inChannelGroup, outChannelGroup,
	       strideHeight, strideWidth,
	       inGroupOffset, outGroupOffset, kernGroupOffset,
	       oPixels, nY,
	       vrij,
	       n, k );
	  }
	} // outChannel
      } // group
    } // batch
  }

  return VEDNN_SUCCESS;
}

