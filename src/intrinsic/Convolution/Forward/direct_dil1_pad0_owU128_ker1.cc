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
    const int64_t inChannelGroup,
    const int64_t outChannelGroup,
    const int64_t strideHeight,
    const int64_t strideWidth,
    const int64_t inGroupOffset,
    const int64_t outGroupOffset,
    const int64_t kernGroupOffset,
    const int64_t oPixels,
    const int64_t n,
    const int64_t k,
    const int64_t nY,
    const __vr vrij

)
{
  int64_t outIndex = outGroupOffset + (n * outChannel + k  ) * oPixels ;

  for (int64_t y=0; y<outHeight; y+=nY)
  {
    const int64_t vl = outWidth * (outHeight - y < nY ? outHeight - y : nY) ;
    const int64_t op = y * outWidth ;

    __vr vrsum = _vel_vbrds_vsl(0.f, vl) ;

    for (int64_t c = 0; c < inChannelGroup; c++) {
      const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;
      __vr vrpin = _vel_vsfa_vvssl(vrij, 2, (uint64_t)(pInChannel+y*strideHeight*inWidth), vl) ;

      __vr vrin = _vel_vgtu_vvssl(vrpin, 0, 0, vl) ;

#define FILTER_OFFSET(k,c) ( kernGroupOffset + filter_index<FLAYOUT>(k,c,0,0, inChannelGroup, outChannelGroup, 1, 1) )
      vrsum = _vel_vfmads_vvsvl(vrsum, pKernel[FILTER_OFFSET(k,c)], vrin, vl) ;
#undef FILTER_OFFSET
    } // inChannel

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
    const int64_t inChannelGroup,
    const int64_t outChannelGroup,
    const int64_t strideHeight,
    const int64_t strideWidth,
    const int64_t inGroupOffset,
    const int64_t outGroupOffset,
    const int64_t kernGroupOffset,
    const int64_t oPixels,
    const int64_t n,
    const int64_t k,
    const int64_t nY,
    const __vr vrij

)
{
  int64_t outIndex = outGroupOffset + (n * outChannel + k  ) * oPixels ;

  for (int64_t y=0; y<outHeight; y+=nY)
  {
    const int64_t vl = outWidth * (outHeight - y < nY ? outHeight - y : nY) ;
    const int64_t op = y * outWidth ;

    __vr vrsum01 = _vel_pvbrd_vsl(0UL, vl) ;

    for (int64_t c = 0; c < inChannelGroup; c++) {
      const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;
      __vr vrpin = _vel_vsfa_vvssl(vrij, 2, (uint64_t)(pInChannel+y*strideHeight*inWidth), vl) ;

      __vr vrin = _vel_vgtu_vvssl(vrpin, 0, 0, vl) ;
      __vr vrinP = _vel_vshf_vvvsl(vrin, vrin, VE_VSHUFFLE_YUZU, vl) ;

#define FILTER_OFFSET(k,c) ( kernGroupOffset + filter_index<FLAYOUT>(k,c,0,0, inChannelGroup, outChannelGroup, 1, 1) )

      const uint64_t kerValue01 = _vel_pack_f32p(pKernel + FILTER_OFFSET(k+ 0,c),
						 pKernel + FILTER_OFFSET(k+ 1,c)) ;

      vrsum01 = _vel_pvfmad_vvsvl(vrsum01, kerValue01, vrinP, vl) ;
#undef FILTER_OFFSET
    } // inChannel

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
    const int64_t inChannelGroup,
    const int64_t outChannelGroup,
    const int64_t strideHeight,
    const int64_t strideWidth,
    const int64_t inGroupOffset,
    const int64_t outGroupOffset,
    const int64_t kernGroupOffset,
    const int64_t oPixels,
    const int64_t n,
    const int64_t k,
    const int64_t nY,
    const __vr vrij

)
{
  int64_t outIndex = outGroupOffset + (n * outChannel + k  ) * oPixels ;

  for (int64_t y=0; y<outHeight; y+=nY)
  {
    const int64_t vl = outWidth * (outHeight - y < nY ? outHeight - y : nY) ;
    const int64_t op = y * outWidth ;

    __vr vrsum01 = _vel_pvbrd_vsl(0UL, vl) ;
    __vr vrsum23 = _vel_pvbrd_vsl(0UL, vl) ;

    for (int64_t c = 0; c < inChannelGroup; c++) {
      const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;
      __vr vrpin = _vel_vsfa_vvssl(vrij, 2, (uint64_t)(pInChannel+y*strideHeight*inWidth), vl) ;

      __vr vrin = _vel_vgtu_vvssl(vrpin, 0, 0, vl) ;
      __vr vrinP = _vel_vshf_vvvsl(vrin, vrin, VE_VSHUFFLE_YUZU, vl) ;

#define FILTER_OFFSET(k,c) ( kernGroupOffset + filter_index<FLAYOUT>(k,c,0,0, inChannelGroup, outChannelGroup, 1, 1) )

      const uint64_t kerValue01 = _vel_pack_f32p(pKernel + FILTER_OFFSET(k+ 0,c),
						 pKernel + FILTER_OFFSET(k+ 1,c)) ;
      const uint64_t kerValue23 = _vel_pack_f32p(pKernel + FILTER_OFFSET(k+ 2,c),
						 pKernel + FILTER_OFFSET(k+ 3,c)) ;

      vrsum01 = _vel_pvfmad_vvsvl(vrsum01, kerValue01, vrinP, vl) ;
      vrsum23 = _vel_pvfmad_vvsvl(vrsum23, kerValue23, vrinP, vl) ;
#undef FILTER_OFFSET
    } // inChannel

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
    const int64_t inChannelGroup,
    const int64_t outChannelGroup,
    const int64_t strideHeight,
    const int64_t strideWidth,
    const int64_t inGroupOffset,
    const int64_t outGroupOffset,
    const int64_t kernGroupOffset,
    const int64_t oPixels,
    const int64_t n,
    const int64_t k,
    const int64_t nY,
    const __vr vrij

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

    for (int64_t c = 0; c < inChannelGroup; c++) {
      const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;
      __vr vrpin = _vel_vsfa_vvssl(vrij, 2, (uint64_t)(pInChannel+y*strideHeight*inWidth), vl) ;

      __vr vrin = _vel_vgtu_vvssl(vrpin, 0, 0, vl) ;
      __vr vrinP = _vel_vshf_vvvsl(vrin, vrin, VE_VSHUFFLE_YUZU, vl) ;

#define FILTER_OFFSET(k,c) ( kernGroupOffset + filter_index<FLAYOUT>(k,c,0,0, inChannelGroup, outChannelGroup, 1, 1) )

      const uint64_t kerValue01 = _vel_pack_f32p(pKernel + FILTER_OFFSET(k+ 0,c),
						 pKernel + FILTER_OFFSET(k+ 1,c)) ;
      const uint64_t kerValue23 = _vel_pack_f32p(pKernel + FILTER_OFFSET(k+ 2,c),
						 pKernel + FILTER_OFFSET(k+ 3,c)) ;
      const uint64_t kerValue45 = _vel_pack_f32p(pKernel + FILTER_OFFSET(k+ 4,c),
						 pKernel + FILTER_OFFSET(k+ 5,c)) ;
      const uint64_t kerValue67 = _vel_pack_f32p(pKernel + FILTER_OFFSET(k+ 6,c),
						 pKernel + FILTER_OFFSET(k+ 7,c)) ;

      vrsum01 = _vel_pvfmad_vvsvl(vrsum01, kerValue01, vrinP, vl) ;
      vrsum23 = _vel_pvfmad_vvsvl(vrsum23, kerValue23, vrinP, vl) ;
      vrsum45 = _vel_pvfmad_vvsvl(vrsum45, kerValue45, vrinP, vl) ;
      vrsum67 = _vel_pvfmad_vvsvl(vrsum67, kerValue67, vrinP, vl) ;
#undef FILTER_OFFSET
    } // inChannel

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
    const int64_t inChannelGroup,
    const int64_t outChannelGroup,
    const int64_t strideHeight,
    const int64_t strideWidth,
    const int64_t inGroupOffset,
    const int64_t outGroupOffset,
    const int64_t kernGroupOffset,
    const int64_t oPixels,
    const int64_t n,
    const int64_t k,
    const int64_t nY,
    const __vr vrij

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

    int c = 0 ;
    if ( (inChannelGroup & 0x01) == 1 ) {
      const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;
      __vr vrpin_c0 = _vel_vsfa_vvssl(vrij, 2, (uint64_t)(pInChannel+y*strideHeight*inWidth), vl) ;

      __vr vrin_c0 = _vel_vgtu_vvssl(vrpin_c0, 0, 0, vl) ;

#define FILTER_OFFSET(K,C) ( kernGroupOffset + filter_index<FLAYOUT>(K,C,0,0, inChannelGroup, outChannelGroup, 1, 1) )
#define VFMAD16(VRIN, C)									\
      {												\
	__vr vrinP = _vel_vshf_vvvsl(VRIN, VRIN, VE_VSHUFFLE_YUZU, vl) ;			\
	const uint64_t kerValue01 = _vel_pack_f32p(pKernel + FILTER_OFFSET(k+ 0,(C)),		\
						   pKernel + FILTER_OFFSET(k+ 1,(C))) ;		\
	const uint64_t kerValue23 = _vel_pack_f32p(pKernel + FILTER_OFFSET(k+ 2,(C)),		\
						   pKernel + FILTER_OFFSET(k+ 3,(C))) ;		\
	const uint64_t kerValue45 = _vel_pack_f32p(pKernel + FILTER_OFFSET(k+ 4,(C)),		\
						   pKernel + FILTER_OFFSET(k+ 5,(C))) ;		\
	const uint64_t kerValue67 = _vel_pack_f32p(pKernel + FILTER_OFFSET(k+ 6,(C)),		\
						   pKernel + FILTER_OFFSET(k+ 7,(C))) ;		\
	const uint64_t kerValue89 = _vel_pack_f32p(pKernel + FILTER_OFFSET(k+ 8,(C)),		\
						   pKernel + FILTER_OFFSET(k+ 9,(C))) ;		\
	const uint64_t kerValueAB = _vel_pack_f32p(pKernel + FILTER_OFFSET(k+10,(C)),		\
						   pKernel + FILTER_OFFSET(k+11,(C))) ;		\
	const uint64_t kerValueCD = _vel_pack_f32p(pKernel + FILTER_OFFSET(k+12,(C)),		\
						   pKernel + FILTER_OFFSET(k+13,(C))) ;		\
	const uint64_t kerValueEF = _vel_pack_f32p(pKernel + FILTER_OFFSET(k+14,(C)),		\
						   pKernel + FILTER_OFFSET(k+15,(C))) ;		\
	vrsum01 = _vel_pvfmad_vvsvl(vrsum01, kerValue01, vrinP, vl) ;				\
	vrsum23 = _vel_pvfmad_vvsvl(vrsum23, kerValue23, vrinP, vl) ;				\
	vrsum45 = _vel_pvfmad_vvsvl(vrsum45, kerValue45, vrinP, vl) ;				\
	vrsum67 = _vel_pvfmad_vvsvl(vrsum67, kerValue67, vrinP, vl) ;				\
	vrsum89 = _vel_pvfmad_vvsvl(vrsum89, kerValue89, vrinP, vl) ;				\
	vrsumAB = _vel_pvfmad_vvsvl(vrsumAB, kerValueAB, vrinP, vl) ;				\
	vrsumCD = _vel_pvfmad_vvsvl(vrsumCD, kerValueCD, vrinP, vl) ;				\
	vrsumEF = _vel_pvfmad_vvsvl(vrsumEF, kerValueEF, vrinP, vl) ;				\
      }

      VFMAD16(vrin_c0, c+0) ;

      c+=1 ;
    }
    if ( ((inChannelGroup >> 1) & 0x01) == 1 ) {
      const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;
      __vr vrpin_c0 = _vel_vsfa_vvssl(vrij, 2, (uint64_t)(pInChannel+y*strideHeight*inWidth), vl) ;
      __vr vrpin_c1 = _vel_vaddul_vsvl(  4*inHeight*inWidth,vrpin_c0, vl) ;

      __vr vrin_c0 = _vel_vgtu_vvssl(vrpin_c0, 0, 0, vl) ;
      __vr vrin_c1 = _vel_vgtu_vvssl(vrpin_c1, 0, 0, vl) ;

      VFMAD16(vrin_c0, c+0) ;
      VFMAD16(vrin_c1, c+1) ;


      c+=2 ;
    } // inChannel
    if ( ((inChannelGroup >> 2) & 0x01) == 1 ) {
      const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;
      __vr vrpin_c0 = _vel_vsfa_vvssl(vrij, 2, (uint64_t)(pInChannel+y*strideHeight*inWidth), vl) ;
      __vr vrpin_c1 = _vel_vaddul_vsvl(  4*inHeight*inWidth,vrpin_c0, vl) ;
      __vr vrpin_c2 = _vel_vaddul_vsvl(2*4*inHeight*inWidth,vrpin_c0, vl) ;
      __vr vrpin_c3 = _vel_vaddul_vsvl(3*4*inHeight*inWidth,vrpin_c0, vl) ;

      __vr vrin_c0 = _vel_vgtu_vvssl(vrpin_c0, 0, 0, vl) ;
      __vr vrin_c1 = _vel_vgtu_vvssl(vrpin_c1, 0, 0, vl) ;
      __vr vrin_c2 = _vel_vgtu_vvssl(vrpin_c2, 0, 0, vl) ;
      __vr vrin_c3 = _vel_vgtu_vvssl(vrpin_c3, 0, 0, vl) ;

      VFMAD16(vrin_c0, c+0) ;
      VFMAD16(vrin_c1, c+1) ;
      VFMAD16(vrin_c2, c+2) ;
      VFMAD16(vrin_c3, c+3) ;

      c+=4 ;
    }
    for (; c < inChannelGroup; c+=8) {
      const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;
      __vr vrpin_c0 = _vel_vsfa_vvssl(vrij, 2, (uint64_t)(pInChannel+y*strideHeight*inWidth), vl) ;
      __vr vrpin_c1 = _vel_vaddul_vsvl(  4*inHeight*inWidth,vrpin_c0, vl) ;
      __vr vrpin_c2 = _vel_vaddul_vsvl(2*4*inHeight*inWidth,vrpin_c0, vl) ;
      __vr vrpin_c3 = _vel_vaddul_vsvl(3*4*inHeight*inWidth,vrpin_c0, vl) ;
      __vr vrpin_c4 = _vel_vaddul_vsvl(4*4*inHeight*inWidth,vrpin_c0, vl) ;
      __vr vrpin_c5 = _vel_vaddul_vsvl(5*4*inHeight*inWidth,vrpin_c0, vl) ;
      __vr vrpin_c6 = _vel_vaddul_vsvl(6*4*inHeight*inWidth,vrpin_c0, vl) ;
      __vr vrpin_c7 = _vel_vaddul_vsvl(7*4*inHeight*inWidth,vrpin_c0, vl) ;

      __vr vrin_c0 = _vel_vgtu_vvssl(vrpin_c0, 0, 0, vl) ;
      __vr vrin_c1 = _vel_vgtu_vvssl(vrpin_c1, 0, 0, vl) ;
      __vr vrin_c2 = _vel_vgtu_vvssl(vrpin_c2, 0, 0, vl) ;
      __vr vrin_c3 = _vel_vgtu_vvssl(vrpin_c3, 0, 0, vl) ;
      __vr vrin_c4 = _vel_vgtu_vvssl(vrpin_c4, 0, 0, vl) ;
      __vr vrin_c5 = _vel_vgtu_vvssl(vrpin_c5, 0, 0, vl) ;
      __vr vrin_c6 = _vel_vgtu_vvssl(vrpin_c6, 0, 0, vl) ;
      __vr vrin_c7 = _vel_vgtu_vvssl(vrpin_c7, 0, 0, vl) ;

      VFMAD16(vrin_c0, c+0) ;
      VFMAD16(vrin_c1, c+1) ;
      VFMAD16(vrin_c2, c+2) ;
      VFMAD16(vrin_c3, c+3) ;
      VFMAD16(vrin_c4, c+4) ;
      VFMAD16(vrin_c5, c+5) ;
      VFMAD16(vrin_c6, c+6) ;
      VFMAD16(vrin_c7, c+7) ;

#undef VFMAD16
#undef FILTER_OFFSET
    }

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

static inline void k16_c1024x(
    const float * __restrict__ pIn,
    const float * __restrict__ pKernel,
    float * __restrict__ const pOut,
    const int64_t inChannel,
    const int64_t inWidth,
    const int64_t inHeight,
    const int64_t outChannel,
    const int64_t outWidth,
    const int64_t outHeight,
    const int64_t inChannelGroup,
    const int64_t outChannelGroup,
    const int64_t strideHeight,
    const int64_t strideWidth,
    const int64_t inGroupOffset,
    const int64_t outGroupOffset,
    const int64_t kernGroupOffset,
    const int64_t oPixels,
    const int64_t n,
    const int64_t k,
    const int64_t nY,
    const __vr vrij,
    float * __restrict__ const filter

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


    for(int64_t c0=0; c0<inChannelGroup; c0+=512) {
      const float *pKerValue = pKernel + kernGroupOffset + (k * inChannelGroup + c0) ;

      __vr vr0 = _vel_vld_vssl(8, pKerValue+ 0*inChannelGroup, 256) ;
      __vr vr1 = _vel_vld_vssl(8, pKerValue+ 1*inChannelGroup, 256) ;
      __vr vr2 = _vel_vld_vssl(8, pKerValue+ 2*inChannelGroup, 256) ;
      __vr vr3 = _vel_vld_vssl(8, pKerValue+ 3*inChannelGroup, 256) ;
      __vr vr4 = _vel_vld_vssl(8, pKerValue+ 4*inChannelGroup, 256) ;
      __vr vr5 = _vel_vld_vssl(8, pKerValue+ 5*inChannelGroup, 256) ;
      __vr vr6 = _vel_vld_vssl(8, pKerValue+ 6*inChannelGroup, 256) ;
      __vr vr7 = _vel_vld_vssl(8, pKerValue+ 7*inChannelGroup, 256) ;
      __vr vr8 = _vel_vld_vssl(8, pKerValue+ 8*inChannelGroup, 256) ;
      __vr vr9 = _vel_vld_vssl(8, pKerValue+ 9*inChannelGroup, 256) ;
      __vr vrA = _vel_vld_vssl(8, pKerValue+10*inChannelGroup, 256) ;
      __vr vrB = _vel_vld_vssl(8, pKerValue+11*inChannelGroup, 256) ;
      __vr vrC = _vel_vld_vssl(8, pKerValue+12*inChannelGroup, 256) ;
      __vr vrD = _vel_vld_vssl(8, pKerValue+13*inChannelGroup, 256) ;
      __vr vrE = _vel_vld_vssl(8, pKerValue+14*inChannelGroup, 256) ;
      __vr vrF = _vel_vld_vssl(8, pKerValue+15*inChannelGroup, 256) ;

      __vr vr01_c0 = _vel_vshf_vvvsl(vr0,vr1,VE_VSHUFFLE_YLZL, 256) ;
      __vr vr23_c0 = _vel_vshf_vvvsl(vr2,vr3,VE_VSHUFFLE_YLZL, 256) ;
      __vr vr45_c0 = _vel_vshf_vvvsl(vr4,vr5,VE_VSHUFFLE_YLZL, 256) ;
      __vr vr67_c0 = _vel_vshf_vvvsl(vr6,vr7,VE_VSHUFFLE_YLZL, 256) ;
      __vr vr89_c0 = _vel_vshf_vvvsl(vr8,vr9,VE_VSHUFFLE_YLZL, 256) ;
      __vr vrAB_c0 = _vel_vshf_vvvsl(vrA,vrB,VE_VSHUFFLE_YLZL, 256) ;
      __vr vrCD_c0 = _vel_vshf_vvvsl(vrC,vrD,VE_VSHUFFLE_YLZL, 256) ;
      __vr vrEF_c0 = _vel_vshf_vvvsl(vrE,vrF,VE_VSHUFFLE_YLZL, 256) ;

      _vel_vst_vssl(vr01_c0, 8, filter, 256) ;
      _vel_vst_vssl(vr23_c0, 8, filter+1*512, 256) ;
      _vel_vst_vssl(vr45_c0, 8, filter+2*512, 256) ;
      _vel_vst_vssl(vr67_c0, 8, filter+3*512, 256) ;
      _vel_vst_vssl(vr89_c0, 8, filter+4*512, 256) ;
      _vel_vst_vssl(vrAB_c0, 8, filter+5*512, 256) ;
      _vel_vst_vssl(vrCD_c0, 8, filter+6*512, 256) ;
      _vel_vst_vssl(vrEF_c0, 8, filter+7*512, 256) ;

      __vr vr01_c1 = _vel_vshf_vvvsl(vr0,vr1,VE_VSHUFFLE_YUZU, 256) ;
      __vr vr23_c1 = _vel_vshf_vvvsl(vr2,vr3,VE_VSHUFFLE_YUZU, 256) ;
      __vr vr45_c1 = _vel_vshf_vvvsl(vr4,vr5,VE_VSHUFFLE_YUZU, 256) ;
      __vr vr67_c1 = _vel_vshf_vvvsl(vr6,vr7,VE_VSHUFFLE_YUZU, 256) ;
      __vr vr89_c1 = _vel_vshf_vvvsl(vr8,vr9,VE_VSHUFFLE_YUZU, 256) ;
      __vr vrAB_c1 = _vel_vshf_vvvsl(vrA,vrB,VE_VSHUFFLE_YUZU, 256) ;
      __vr vrCD_c1 = _vel_vshf_vvvsl(vrC,vrD,VE_VSHUFFLE_YUZU, 256) ;
      __vr vrEF_c1 = _vel_vshf_vvvsl(vrE,vrF,VE_VSHUFFLE_YUZU, 256) ;

      _vel_vst_vssl(vr01_c1, 8, filter+ 8*512, 256) ;
      _vel_vst_vssl(vr23_c1, 8, filter+ 9*512, 256) ;
      _vel_vst_vssl(vr45_c1, 8, filter+10*512, 256) ;
      _vel_vst_vssl(vr67_c1, 8, filter+11*512, 256) ;
      _vel_vst_vssl(vr89_c1, 8, filter+12*512, 256) ;
      _vel_vst_vssl(vrAB_c1, 8, filter+13*512, 256) ;
      _vel_vst_vssl(vrCD_c1, 8, filter+14*512, 256) ;
      _vel_vst_vssl(vrEF_c1, 8, filter+15*512, 256) ;

      for(int64_t c1 = 0; c1 < 512 ; c1+=8 ) {
	const int64_t c = c0 + c1 ;

	const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;
	const uint64_t* filter_u64 = (const uint64_t*)(filter+c1) ;

	__vr vrpin_c0 = _vel_vsfa_vvssl(vrij, 2, (uint64_t)(pInChannel+y*strideHeight*inWidth), vl) ;
	__vr vrpin_c1 = _vel_vaddul_vsvl(  4*inHeight*inWidth,vrpin_c0, vl) ;
	__vr vrpin_c2 = _vel_vaddul_vsvl(2*4*inHeight*inWidth,vrpin_c0, vl) ;
	__vr vrpin_c3 = _vel_vaddul_vsvl(3*4*inHeight*inWidth,vrpin_c0, vl) ;
	__vr vrpin_c4 = _vel_vaddul_vsvl(4*4*inHeight*inWidth,vrpin_c0, vl) ;
	__vr vrpin_c5 = _vel_vaddul_vsvl(5*4*inHeight*inWidth,vrpin_c0, vl) ;
	__vr vrpin_c6 = _vel_vaddul_vsvl(6*4*inHeight*inWidth,vrpin_c0, vl) ;
	__vr vrpin_c7 = _vel_vaddul_vsvl(7*4*inHeight*inWidth,vrpin_c0, vl) ;

	__vr vrin_c0 = _vel_vgtu_vvssl(vrpin_c0, 0, 0, vl) ;
	__vr vrin_c1 = _vel_vgtu_vvssl(vrpin_c1, 0, 0, vl) ;
	__vr vrin_c2 = _vel_vgtu_vvssl(vrpin_c2, 0, 0, vl) ;
	__vr vrin_c3 = _vel_vgtu_vvssl(vrpin_c3, 0, 0, vl) ;
	__vr vrin_c4 = _vel_vgtu_vvssl(vrpin_c4, 0, 0, vl) ;
	__vr vrin_c5 = _vel_vgtu_vvssl(vrpin_c5, 0, 0, vl) ;
	__vr vrin_c6 = _vel_vgtu_vvssl(vrpin_c6, 0, 0, vl) ;
	__vr vrin_c7 = _vel_vgtu_vvssl(vrpin_c7, 0, 0, vl) ;

	const float *pKerValue = pKernel + kernGroupOffset + (k * inChannelGroup + c);

	__vr vrin_c0P = _vel_vshf_vvvsl(vrin_c0, vrin_c0, VE_VSHUFFLE_YUZU, vl) ;
	vrsum01 = _vel_pvfmad_vvsvl(vrsum01, filter_u64[0*256], vrin_c0P, vl) ;
	vrsum23 = _vel_pvfmad_vvsvl(vrsum23, filter_u64[1*256], vrin_c0P, vl) ;
	vrsum45 = _vel_pvfmad_vvsvl(vrsum45, filter_u64[2*256], vrin_c0P, vl) ;
	vrsum67 = _vel_pvfmad_vvsvl(vrsum67, filter_u64[3*256], vrin_c0P, vl) ;
	vrsum89 = _vel_pvfmad_vvsvl(vrsum89, filter_u64[4*256], vrin_c0P, vl) ;
	vrsumAB = _vel_pvfmad_vvsvl(vrsumAB, filter_u64[5*256], vrin_c0P, vl) ;
	vrsumCD = _vel_pvfmad_vvsvl(vrsumCD, filter_u64[6*256], vrin_c0P, vl) ;
	vrsumEF = _vel_pvfmad_vvsvl(vrsumEF, filter_u64[7*256], vrin_c0P, vl) ;

	__vr vrin_c1P = _vel_vshf_vvvsl(vrin_c1, vrin_c1, VE_VSHUFFLE_YUZU, vl) ;
	vrsum01 = _vel_pvfmad_vvsvl(vrsum01, filter_u64[8*256],  vrin_c1P, vl) ;
	vrsum23 = _vel_pvfmad_vvsvl(vrsum23, filter_u64[9*256],  vrin_c1P, vl) ;
	vrsum45 = _vel_pvfmad_vvsvl(vrsum45, filter_u64[10*256], vrin_c1P, vl) ;
	vrsum67 = _vel_pvfmad_vvsvl(vrsum67, filter_u64[11*256], vrin_c1P, vl) ;
	vrsum89 = _vel_pvfmad_vvsvl(vrsum89, filter_u64[12*256], vrin_c1P, vl) ;
	vrsumAB = _vel_pvfmad_vvsvl(vrsumAB, filter_u64[13*256], vrin_c1P, vl) ;
	vrsumCD = _vel_pvfmad_vvsvl(vrsumCD, filter_u64[14*256], vrin_c1P, vl) ;
	vrsumEF = _vel_pvfmad_vvsvl(vrsumEF, filter_u64[15*256], vrin_c1P, vl) ;

	__vr vrin_c2P = _vel_vshf_vvvsl(vrin_c2, vrin_c2, VE_VSHUFFLE_YUZU, vl) ;
	vrsum01 = _vel_pvfmad_vvsvl(vrsum01, filter_u64[0*256+1], vrin_c2P, vl) ;
	vrsum23 = _vel_pvfmad_vvsvl(vrsum23, filter_u64[1*256+1], vrin_c2P, vl) ;
	vrsum45 = _vel_pvfmad_vvsvl(vrsum45, filter_u64[2*256+1], vrin_c2P, vl) ;
	vrsum67 = _vel_pvfmad_vvsvl(vrsum67, filter_u64[3*256+1], vrin_c2P, vl) ;
	vrsum89 = _vel_pvfmad_vvsvl(vrsum89, filter_u64[4*256+1], vrin_c2P, vl) ;
	vrsumAB = _vel_pvfmad_vvsvl(vrsumAB, filter_u64[5*256+1], vrin_c2P, vl) ;
	vrsumCD = _vel_pvfmad_vvsvl(vrsumCD, filter_u64[6*256+1], vrin_c2P, vl) ;
	vrsumEF = _vel_pvfmad_vvsvl(vrsumEF, filter_u64[7*256+1], vrin_c2P, vl) ;

	__vr vrin_c3P = _vel_vshf_vvvsl(vrin_c3, vrin_c3, VE_VSHUFFLE_YUZU, vl) ;
	vrsum01 = _vel_pvfmad_vvsvl(vrsum01, filter_u64[8*256+1],  vrin_c3P, vl) ;
	vrsum23 = _vel_pvfmad_vvsvl(vrsum23, filter_u64[9*256+1],  vrin_c3P, vl) ;
	vrsum45 = _vel_pvfmad_vvsvl(vrsum45, filter_u64[10*256+1], vrin_c3P, vl) ;
	vrsum67 = _vel_pvfmad_vvsvl(vrsum67, filter_u64[11*256+1], vrin_c3P, vl) ;
	vrsum89 = _vel_pvfmad_vvsvl(vrsum89, filter_u64[12*256+1], vrin_c3P, vl) ;
	vrsumAB = _vel_pvfmad_vvsvl(vrsumAB, filter_u64[13*256+1], vrin_c3P, vl) ;
	vrsumCD = _vel_pvfmad_vvsvl(vrsumCD, filter_u64[14*256+1], vrin_c3P, vl) ;
	vrsumEF = _vel_pvfmad_vvsvl(vrsumEF, filter_u64[15*256+1], vrin_c3P, vl) ;

	__vr vrin_c4P = _vel_vshf_vvvsl(vrin_c4, vrin_c4, VE_VSHUFFLE_YUZU, vl) ;
	vrsum01 = _vel_pvfmad_vvsvl(vrsum01, filter_u64[0*256+2], vrin_c4P, vl) ;
	vrsum23 = _vel_pvfmad_vvsvl(vrsum23, filter_u64[1*256+2], vrin_c4P, vl) ;
	vrsum45 = _vel_pvfmad_vvsvl(vrsum45, filter_u64[2*256+2], vrin_c4P, vl) ;
	vrsum67 = _vel_pvfmad_vvsvl(vrsum67, filter_u64[3*256+2], vrin_c4P, vl) ;
	vrsum89 = _vel_pvfmad_vvsvl(vrsum89, filter_u64[4*256+2], vrin_c4P, vl) ;
	vrsumAB = _vel_pvfmad_vvsvl(vrsumAB, filter_u64[5*256+2], vrin_c4P, vl) ;
	vrsumCD = _vel_pvfmad_vvsvl(vrsumCD, filter_u64[6*256+2], vrin_c4P, vl) ;
	vrsumEF = _vel_pvfmad_vvsvl(vrsumEF, filter_u64[7*256+2], vrin_c4P, vl) ;

	__vr vrin_c5P = _vel_vshf_vvvsl(vrin_c5, vrin_c5, VE_VSHUFFLE_YUZU, vl) ;
	vrsum01 = _vel_pvfmad_vvsvl(vrsum01, filter_u64[8*256+2],  vrin_c5P, vl) ;
	vrsum23 = _vel_pvfmad_vvsvl(vrsum23, filter_u64[9*256+2],  vrin_c5P, vl) ;
	vrsum45 = _vel_pvfmad_vvsvl(vrsum45, filter_u64[10*256+2], vrin_c5P, vl) ;
	vrsum67 = _vel_pvfmad_vvsvl(vrsum67, filter_u64[11*256+2], vrin_c5P, vl) ;
	vrsum89 = _vel_pvfmad_vvsvl(vrsum89, filter_u64[12*256+2], vrin_c5P, vl) ;
	vrsumAB = _vel_pvfmad_vvsvl(vrsumAB, filter_u64[13*256+2], vrin_c5P, vl) ;
	vrsumCD = _vel_pvfmad_vvsvl(vrsumCD, filter_u64[14*256+2], vrin_c5P, vl) ;
	vrsumEF = _vel_pvfmad_vvsvl(vrsumEF, filter_u64[15*256+2], vrin_c5P, vl) ;

	__vr vrin_c6P = _vel_vshf_vvvsl(vrin_c6, vrin_c6, VE_VSHUFFLE_YUZU, vl) ;
	vrsum01 = _vel_pvfmad_vvsvl(vrsum01, filter_u64[0*256+3], vrin_c6P, vl) ;
	vrsum23 = _vel_pvfmad_vvsvl(vrsum23, filter_u64[1*256+3], vrin_c6P, vl) ;
	vrsum45 = _vel_pvfmad_vvsvl(vrsum45, filter_u64[2*256+3], vrin_c6P, vl) ;
	vrsum67 = _vel_pvfmad_vvsvl(vrsum67, filter_u64[3*256+3], vrin_c6P, vl) ;
	vrsum89 = _vel_pvfmad_vvsvl(vrsum89, filter_u64[4*256+3], vrin_c6P, vl) ;
	vrsumAB = _vel_pvfmad_vvsvl(vrsumAB, filter_u64[5*256+3], vrin_c6P, vl) ;
	vrsumCD = _vel_pvfmad_vvsvl(vrsumCD, filter_u64[6*256+3], vrin_c6P, vl) ;
	vrsumEF = _vel_pvfmad_vvsvl(vrsumEF, filter_u64[7*256+3], vrin_c6P, vl) ;

	__vr vrin_c7P = _vel_vshf_vvvsl(vrin_c7, vrin_c7, VE_VSHUFFLE_YUZU, vl) ;
	vrsum01 = _vel_pvfmad_vvsvl(vrsum01, filter_u64[8*256+3],  vrin_c7P, vl) ;
	vrsum23 = _vel_pvfmad_vvsvl(vrsum23, filter_u64[9*256+3],  vrin_c7P, vl) ;
	vrsum45 = _vel_pvfmad_vvsvl(vrsum45, filter_u64[10*256+3], vrin_c7P, vl) ;
	vrsum67 = _vel_pvfmad_vvsvl(vrsum67, filter_u64[11*256+3], vrin_c7P, vl) ;
	vrsum89 = _vel_pvfmad_vvsvl(vrsum89, filter_u64[12*256+3], vrin_c7P, vl) ;
	vrsumAB = _vel_pvfmad_vvsvl(vrsumAB, filter_u64[13*256+3], vrin_c7P, vl) ;
	vrsumCD = _vel_pvfmad_vvsvl(vrsumCD, filter_u64[14*256+3], vrin_c7P, vl) ;
	vrsumEF = _vel_pvfmad_vvsvl(vrsumEF, filter_u64[15*256+3], vrin_c7P, vl) ;
      } // inChannel
    }

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
vednnConvolutionForward_direct_dil1_pad0_owU128_ker1(
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

  float __attribute__ ((aligned(8))) filter[16*512] ;

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
	       inChannelGroup, outChannelGroup,
	       strideHeight, strideWidth,
	       inGroupOffset, outGroupOffset,
	       kernGroupOffset, oPixels, n, k,
	       nY, vrij) ;
	  }
	  else {
	    k1<VEDNN_FILTER_LAYOUT_HWCN>(pIn, pKernel, pOut,
	       inChannel, inWidth, inHeight,
	       outChannel, outWidth, outHeight,
	       inChannelGroup, outChannelGroup,
	       strideHeight, strideWidth,
	       inGroupOffset, outGroupOffset,
	       kernGroupOffset, oPixels, n, k,
	       nY, vrij) ;
	  }
	  k++ ;
	}
	if ( ((outChannelGroup >> 1) & 0x01) == 1 ) {
	  if( filter_layout == VEDNN_FILTER_LAYOUT_NCHW) {
	    k2<VEDNN_FILTER_LAYOUT_NCHW>(pIn, pKernel, pOut,
	       inChannel, inWidth, inHeight,
	       outChannel, outWidth, outHeight,
	       inChannelGroup, outChannelGroup,
	       strideHeight, strideWidth,
	       inGroupOffset, outGroupOffset,
	       kernGroupOffset, oPixels, n, k,
	       nY, vrij) ;
	  }
	  else {
	    k2<VEDNN_FILTER_LAYOUT_HWCN>(pIn, pKernel, pOut,
	       inChannel, inWidth, inHeight,
	       outChannel, outWidth, outHeight,
	       inChannelGroup, outChannelGroup,
	       strideHeight, strideWidth,
	       inGroupOffset, outGroupOffset,
	       kernGroupOffset, oPixels, n, k,
	       nY, vrij) ;
	  }

	  k+=2 ;
	}
	if ( ((outChannelGroup >> 2) & 0x01) == 1 ) {
	  if( filter_layout == VEDNN_FILTER_LAYOUT_NCHW) {
	    k4<VEDNN_FILTER_LAYOUT_NCHW>(pIn, pKernel, pOut,
	       inChannel, inWidth, inHeight,
	       outChannel, outWidth, outHeight,
	       inChannelGroup, outChannelGroup,
	       strideHeight, strideWidth,
	       inGroupOffset, outGroupOffset,
	       kernGroupOffset, oPixels, n, k,
	       nY, vrij) ;
	  }
	  else {
	    k4<VEDNN_FILTER_LAYOUT_HWCN>(pIn, pKernel, pOut,
	       inChannel, inWidth, inHeight,
	       outChannel, outWidth, outHeight,
	       inChannelGroup, outChannelGroup,
	       strideHeight, strideWidth,
	       inGroupOffset, outGroupOffset,
	       kernGroupOffset, oPixels, n, k,
	       nY, vrij) ;
	  }

	  k+=4 ;
	}
	if ( ((outChannelGroup >> 3) & 0x01) == 1 ) {
	  if( filter_layout == VEDNN_FILTER_LAYOUT_NCHW) {
	    k8<VEDNN_FILTER_LAYOUT_NCHW>(pIn, pKernel, pOut,
	       inChannel, inWidth, inHeight,
	       outChannel, outWidth, outHeight,
	       inChannelGroup, outChannelGroup,
	       strideHeight, strideWidth,
	       inGroupOffset, outGroupOffset,
	       kernGroupOffset, oPixels, n, k,
	       nY, vrij) ;
	  }
	  else {
	    k8<VEDNN_FILTER_LAYOUT_HWCN>(pIn, pKernel, pOut,
	       inChannel, inWidth, inHeight,
	       outChannel, outWidth, outHeight,
	       inChannelGroup, outChannelGroup,
	       strideHeight, strideWidth,
	       inGroupOffset, outGroupOffset,
	       kernGroupOffset, oPixels, n, k,
	       nY, vrij) ;
	  }

	  k+=8 ;
	}
	for ( ; k < outChannelGroup; k+=16) {
	  if( filter_layout == VEDNN_FILTER_LAYOUT_NCHW) {
	    if ( inChannelGroup % 1024 == 0 && (((uint64_t)pDataKernel) & 0x7) == 0 ) {
	      k16_c1024x(pIn, pKernel, pOut,
			 inChannel, inWidth, inHeight,
			 outChannel, outWidth, outHeight,
			 inChannelGroup, outChannelGroup,
			 strideHeight, strideWidth,
			 inGroupOffset, outGroupOffset,
			 kernGroupOffset, oPixels, n, k,
			 nY, vrij, filter) ;
	    }
	    else {
	      k16<VEDNN_FILTER_LAYOUT_NCHW>(pIn, pKernel, pOut,
		 inChannel, inWidth, inHeight,
		 outChannel, outWidth, outHeight,
		 inChannelGroup, outChannelGroup,
		 strideHeight, strideWidth,
		 inGroupOffset, outGroupOffset,
		 kernGroupOffset, oPixels, n, k,
		 nY, vrij) ;
	    }
	  }
	  else {
	    k16<VEDNN_FILTER_LAYOUT_HWCN>(pIn, pKernel, pOut,
	       inChannel, inWidth, inHeight,
	       outChannel, outWidth, outHeight,
	       inChannelGroup, outChannelGroup,
	       strideHeight, strideWidth,
	       inGroupOffset, outGroupOffset,
	       kernGroupOffset, oPixels, n, k,
	       nY, vrij) ;
	  }

	} // outChannel
      } // group
    } // batch
  }

  return VEDNN_SUCCESS;
}

