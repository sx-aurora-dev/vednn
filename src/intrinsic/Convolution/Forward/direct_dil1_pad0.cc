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
  const int64_t n,
  const int64_t k
)
{
  int64_t outIndex = outGroupOffset + (n * outChannel + k  ) * outHeight * outWidth ;

  for (int64_t y=0; y<outHeight; y++) {
    int64_t i = y * strideHeight ;

    for ( int64_t x0=0; x0<outWidth; x0+=VLEN) {

      const int64_t vl = outWidth - x0 < VLEN ? outWidth - x0 : VLEN ;

      __vr vrsum = _vel_vbrds_vsl(0.f, vl) ;

      __vr vrseq = _vel_vseq_vl(vl) ;
      __vr vrj   = _vel_vmulsl_vsvl(strideWidth, _vel_vaddsl_vsvl(x0, vrseq, vl), vl) ;

      const float *pKerValue = pKernel + kernGroupOffset + k * inChannelGroup * kernHeight * kernWidth ;

      for (int64_t r = 0; r < kernHeight; r++) {
	for (int64_t s = 0; s < kernWidth; s++) {

	  const int64_t h = i + r;
	  __vr vrw = _vel_vaddsl_vsvl(s,  vrj, vl) ;

	  for (int64_t c = 0; c < inChannelGroup; c++) {
	    const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;

	    __vr vrin = _vel_vldu_vssl(4*strideWidth,&pInChannel[h*inWidth+x0*strideWidth+s], vl) ;

#define FILTER_OFFSET(k,c,r,s) ( kernGroupOffset + filter_index<FLAYOUT>(k,c,r,s, inChannelGroup, outChannelGroup, kernHeight, kernWidth) )

	    vrsum = _vel_vfmads_vvsvl(vrsum, pKerValue[FILTER_OFFSET(k,c,r,s)], vrin, vl) ;

#undef FILTER_OFFSET
	  } // inChannel
	} // kernWidth
      } // kernHeight

      _vel_vstu_vssl(vrsum, 4, pOut+outIndex + 0 * outHeight*outWidth, vl) ;

      outIndex += vl ;
    } // x
  } // y
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
  const int64_t n,
  const int64_t k
)
{
  int64_t outIndex = outGroupOffset + (n * outChannel + k  ) * outHeight * outWidth ;

  for (int64_t y=0; y<outHeight; y++) {
    int64_t i = y * strideHeight ;

    for ( int64_t x0=0; x0<outWidth; x0+=VLEN) {

      const int64_t vl = outWidth - x0 < VLEN ? outWidth - x0 : VLEN ;

      __vr vrsum01 = _vel_pvbrd_vsl(0UL, vl) ;

      __vr vrseq = _vel_vseq_vl(vl) ;
      __vr vrj   = _vel_vmulsl_vsvl(strideWidth, _vel_vaddsl_vsvl(x0, vrseq, vl), vl) ;

      const float *pKerValue = pKernel + kernGroupOffset + k * inChannelGroup * kernHeight * kernWidth ;

      for (int64_t r = 0; r < kernHeight; r++) {
	for (int64_t s = 0; s < kernWidth; s++) {

	  const int64_t h = i + r;
	  __vr vrw = _vel_vaddsl_vsvl(s,  vrj, vl) ;

	  for (int64_t c = 0; c < inChannelGroup; c++) {
	    const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;

	    __vr vrin = _vel_vldu_vssl(4*strideWidth,&pInChannel[h*inWidth+x0*strideWidth+s], vl) ;
	    __vr vrinP = _vel_vshf_vvvsl(vrin, vrin, VE_VSHUFFLE_YUZU, vl) ;

#define FILTER_OFFSET(k,c,r,s) ( kernGroupOffset + filter_index<FLAYOUT>(k,c,r,s, inChannelGroup, outChannelGroup, kernHeight, kernWidth) )

	      const uint64_t kerValue01 = _vel_pack_f32p(pKernel + FILTER_OFFSET(k+ 0,c,r,s),
		                                         pKernel + FILTER_OFFSET(k+ 1,c,r,s)) ;

	      vrsum01 = _vel_pvfmad_vvsvl(vrsum01, kerValue01, vrinP, vl) ;
#undef FILTER_OFFSET
	  } // inChannel
	} // kernWidth
      } // kernHeight

      _vel_vstu_vssl(vrsum01, 4, pOut+outIndex + 0 * outHeight*outWidth, vl) ;
      _vel_vstl_vssl(vrsum01, 4, pOut+outIndex + 1 * outHeight*outWidth, vl) ;

      outIndex += vl ;
    } // x
  } // y
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
  const int64_t n,
  const int64_t k
)
{
  int64_t outIndex = outGroupOffset + (n * outChannel + k  ) * outHeight * outWidth ;

  for (int64_t y=0; y<outHeight; y++) {
    int64_t i = y * strideHeight ;

    for ( int64_t x0=0; x0<outWidth; x0+=VLEN) {

      const int64_t vl = outWidth - x0 < VLEN ? outWidth - x0 : VLEN ;

      __vr vrsum01 = _vel_pvbrd_vsl(0UL, vl) ;
      __vr vrsum23 = _vel_pvbrd_vsl(0UL, vl) ;

      __vr vrseq = _vel_vseq_vl(vl) ;
      __vr vrj   = _vel_vmulsl_vsvl(strideWidth, _vel_vaddsl_vsvl(x0, vrseq, vl), vl) ;

      const float *pKerValue = pKernel + kernGroupOffset + k * inChannelGroup * kernHeight * kernWidth ;

      for (int64_t r = 0; r < kernHeight; r++) {
	for (int64_t s = 0; s < kernWidth; s++) {

	  const int64_t h = i + r;
	  __vr vrw = _vel_vaddsl_vsvl(s,  vrj, vl) ;

	  for (int64_t c = 0; c < inChannelGroup; c++) {
	    const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;

	    __vr vrin = _vel_vldu_vssl(4*strideWidth,&pInChannel[h*inWidth+x0*strideWidth+s], vl) ;
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

      _vel_vstu_vssl(vrsum01, 4, pOut+outIndex + 0 * outHeight*outWidth, vl) ;
      _vel_vstl_vssl(vrsum01, 4, pOut+outIndex + 1 * outHeight*outWidth, vl) ;
      _vel_vstu_vssl(vrsum23, 4, pOut+outIndex + 2 * outHeight*outWidth, vl) ;
      _vel_vstl_vssl(vrsum23, 4, pOut+outIndex + 3 * outHeight*outWidth, vl) ;

      outIndex += vl ;
    } // x
  } // y
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
  const int64_t n,
  const int64_t k
)
{
  int64_t outIndex = outGroupOffset + (n * outChannel + k  ) * outHeight * outWidth ;

  for (int64_t y=0; y<outHeight; y++) {
    int64_t i = y * strideHeight ;

    for ( int64_t x0=0; x0<outWidth; x0+=VLEN) {

      const int64_t vl = outWidth - x0 < VLEN ? outWidth - x0 : VLEN ;

      __vr vrsum01 = _vel_pvbrd_vsl(0UL, vl) ;
      __vr vrsum23 = _vel_pvbrd_vsl(0UL, vl) ;
      __vr vrsum45 = _vel_pvbrd_vsl(0UL, vl) ;
      __vr vrsum67 = _vel_pvbrd_vsl(0UL, vl) ;

      __vr vrseq = _vel_vseq_vl(vl) ;
      __vr vrj   = _vel_vmulsl_vsvl(strideWidth, _vel_vaddsl_vsvl(x0, vrseq, vl), vl) ;

      const float *pKerValue = pKernel + kernGroupOffset + k * inChannelGroup * kernHeight * kernWidth ;

      for (int64_t r = 0; r < kernHeight; r++) {
	for (int64_t s = 0; s < kernWidth; s++) {

	  const int64_t h = i + r;
	  __vr vrw = _vel_vaddsl_vsvl(s,  vrj, vl) ;

	  for (int64_t c = 0; c < inChannelGroup; c++) {
	    const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;

	    __vr vrin = _vel_vldu_vssl(4*strideWidth,&pInChannel[h*inWidth+x0*strideWidth+s], vl) ;
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


extern "C" vednnError_t
vednnConvolutionForward_direct_dil1_pad0(
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

  {
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
	       n, k );
	  }

	  k+=4 ;
	}
	for (; k < outChannelGroup; k+=8) {
	  if( filter_layout == VEDNN_FILTER_LAYOUT_NCHW) {
	    k8<VEDNN_FILTER_LAYOUT_NCHW>(pIn, pKernel, pOut,
	       inChannel, inWidth, inHeight,
	       outChannel, outWidth, outHeight,
	       kernWidth, kernHeight,
	       inChannelGroup, outChannelGroup,
	       strideHeight, strideWidth,
	       inGroupOffset, outGroupOffset, kernGroupOffset,
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
	       n, k );
	  }
	} // outChannel
      } // group
    } // batch
  }

  return VEDNN_SUCCESS;
}
