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
    const int64_t inChannelGroup,
    const int64_t outChannelGroup,
    const int64_t inGroupOffset,
    const int64_t outGroupOffset,
    const int64_t kernGroupOffset,
    const int64_t oPixels,
    const int64_t n,
    const int64_t k,
    const int64_t op
)
{
  int64_t outIndex = outGroupOffset + (n * outChannel + k  ) * oPixels + op;

    const int64_t vl = oPixels - op < VLEN ? oPixels - op : VLEN ;

    __vr vrsum = _vel_vbrds_vsl(0.0f, vl) ;

    int64_t c = 0 ;
    if( ( inChannelGroup & 0x01 ) == 1 ) {
      const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;

      __vr vrin  = _vel_vldu_vssl(4,&pInChannel[op], vl) ;
      const float *pKerValue = pKernel + kernGroupOffset + (k * inChannelGroup + c);

      vrsum = _vel_vfmads_vvsvl(vrsum, pKerValue[0], vrin, vl) ;

      c++ ;
    }
    for( ; c < inChannelGroup ; c+=2 ) {
      const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c  ) * inHeight * inWidth ) ;

      __vr vrin0  = _vel_vldu_vssl(4,&pInChannel[op], vl) ;
      __vr vrin1  = _vel_vldu_vssl(4,&pInChannel[op + inHeight * inWidth ], vl) ;

      const float *pKerValue = pKernel + kernGroupOffset + (k * inChannelGroup + c) ;

      vrsum = _vel_vfmads_vvsvl(vrsum, pKerValue[0], vrin0, vl) ;
      vrsum = _vel_vfmads_vvsvl(vrsum, pKerValue[1], vrin1, vl) ;
    } // inChannel

    _vel_vstu_vssl(vrsum, 4, pOut+outIndex, vl) ;
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
    const int64_t inChannelGroup,
    const int64_t outChannelGroup,
    const int64_t inGroupOffset,
    const int64_t outGroupOffset,
    const int64_t kernGroupOffset,
    const int64_t oPixels,
    const int64_t n,
    const int64_t k,
    const int64_t op
)
{
  int64_t outIndex = outGroupOffset + (n * outChannel + k  ) * oPixels + op;

    const int64_t vl = oPixels - op < VLEN ? oPixels - op : VLEN ;

    __vr vrsum01 = _vel_pvbrd_vsl(0UL, vl) ;

    int64_t c = 0 ;

    if( ( inChannelGroup & 0x01 ) == 1 ) {
      const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;


      __vr vrin_c0  = _vel_vldu_vssl(4,&pInChannel[op], vl) ;
      __vr vrin_c0P = _vel_vshf_vvvsl(vrin_c0, vrin_c0, VE_VSHUFFLE_YUZU, vl) ;

      const float *pKerValue = pKernel + kernGroupOffset + (k * inChannelGroup + c) ;


      const uint64_t kerValue01_c0 = _vel_pack_f32p(pKerValue   + 0*inChannelGroup,
      				                    pKerValue   + 1*inChannelGroup ) ;
      vrsum01 = _vel_pvfmad_vvsvl(vrsum01, kerValue01_c0, vrin_c0P, vl) ;

      c++ ;
    }
    for( ; c < inChannelGroup ; c+=2 ) {
      const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c  ) * inHeight * inWidth ) ;

      __vr vrin_c0  = _vel_vldu_vssl(4,&pInChannel[op], vl) ;
      __vr vrin_c1  = _vel_vldu_vssl(4,&pInChannel[op + inHeight * inWidth ], vl) ;
      __vr vrin_c0P = _vel_vshf_vvvsl(vrin_c0, vrin_c0, VE_VSHUFFLE_YUZU, vl) ;
      __vr vrin_c1P = _vel_vshf_vvvsl(vrin_c1, vrin_c1, VE_VSHUFFLE_YUZU, vl) ;

      const float *pKerValue = pKernel + kernGroupOffset + (k * inChannelGroup + c) ;


      const uint64_t kerValue01_c0 = _vel_pack_f32p(pKerValue   + 0*inChannelGroup,
      				                    pKerValue   + 1*inChannelGroup ) ;
      const uint64_t kerValue01_c1 = _vel_pack_f32p(pKerValue+1 + 0*inChannelGroup,
      				                    pKerValue+1 + 1*inChannelGroup ) ;
      vrsum01 = _vel_pvfmad_vvsvl(vrsum01, kerValue01_c0, vrin_c0P, vl) ;
      vrsum01 = _vel_pvfmad_vvsvl(vrsum01, kerValue01_c1, vrin_c1P, vl) ;

    } // inChannel

    _vel_vstu_vssl(vrsum01, 4, pOut+outIndex, vl) ;
    _vel_vstl_vssl(vrsum01, 4, pOut+outIndex+   oPixels, vl) ;
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
    const int64_t inChannelGroup,
    const int64_t outChannelGroup,
    const int64_t inGroupOffset,
    const int64_t outGroupOffset,
    const int64_t kernGroupOffset,
    const int64_t oPixels,
    const int64_t n,
    const int64_t k,
    const int64_t op
)
{
  int64_t outIndex = outGroupOffset + (n * outChannel + k  ) * oPixels + op;

    const int64_t vl = oPixels - op < VLEN ? oPixels - op : VLEN ;

    __vr vrsum01 = _vel_pvbrd_vsl(0UL, vl) ;
    __vr vrsum23 = _vel_pvbrd_vsl(0UL, vl) ;

    int64_t c = 0 ;

    if( ( inChannelGroup & 0x01 ) == 1 ) {
      const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;


      __vr vrin_c0  = _vel_vldu_vssl(4,&pInChannel[op], vl) ;
      __vr vrin_c0P = _vel_vshf_vvvsl(vrin_c0, vrin_c0, VE_VSHUFFLE_YUZU, vl) ;

      const float *pKerValue = pKernel + kernGroupOffset + (k * inChannelGroup + c) ;


      const uint64_t kerValue01_c0 = _vel_pack_f32p(pKerValue   + 0*inChannelGroup,
      				                    pKerValue   + 1*inChannelGroup ) ;
      vrsum01 = _vel_pvfmad_vvsvl(vrsum01, kerValue01_c0, vrin_c0P, vl) ;

      const uint64_t kerValue23_c0 = _vel_pack_f32p(pKerValue   + 2*inChannelGroup,
      				                    pKerValue   + 3*inChannelGroup ) ;
      vrsum23 = _vel_pvfmad_vvsvl(vrsum23, kerValue23_c0, vrin_c0P, vl) ;

      c++ ;
    }
    for( ; c < inChannelGroup ; c+=2 ) {
      const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c  ) * inHeight * inWidth ) ;

      __vr vrin_c0  = _vel_vldu_vssl(4,&pInChannel[op], vl) ;
      __vr vrin_c1  = _vel_vldu_vssl(4,&pInChannel[op + inHeight * inWidth ], vl) ;
      __vr vrin_c0P = _vel_vshf_vvvsl(vrin_c0, vrin_c0, VE_VSHUFFLE_YUZU, vl) ;
      __vr vrin_c1P = _vel_vshf_vvvsl(vrin_c1, vrin_c1, VE_VSHUFFLE_YUZU, vl) ;

      const float *pKerValue = pKernel + kernGroupOffset + (k * inChannelGroup + c) ;


      const uint64_t kerValue01_c0 = _vel_pack_f32p(pKerValue   + 0*inChannelGroup,
      				                    pKerValue   + 1*inChannelGroup ) ;
      const uint64_t kerValue01_c1 = _vel_pack_f32p(pKerValue+1 + 0*inChannelGroup,
      				                    pKerValue+1 + 1*inChannelGroup ) ;
      vrsum01 = _vel_pvfmad_vvsvl(vrsum01, kerValue01_c0, vrin_c0P, vl) ;
      vrsum01 = _vel_pvfmad_vvsvl(vrsum01, kerValue01_c1, vrin_c1P, vl) ;

      const uint64_t kerValue23_c0 = _vel_pack_f32p(pKerValue   + 2*inChannelGroup,
      				                    pKerValue   + 3*inChannelGroup ) ;
      const uint64_t kerValue23_c1 = _vel_pack_f32p(pKerValue+1 + 2*inChannelGroup,
      				                    pKerValue+1 + 3*inChannelGroup ) ;
      vrsum23 = _vel_pvfmad_vvsvl(vrsum23, kerValue23_c0, vrin_c0P, vl) ;
      vrsum23 = _vel_pvfmad_vvsvl(vrsum23, kerValue23_c1, vrin_c1P, vl) ;
    } // inChannel

    _vel_vstu_vssl(vrsum01, 4, pOut+outIndex, vl) ;
    _vel_vstl_vssl(vrsum01, 4, pOut+outIndex+   oPixels, vl) ;
    _vel_vstu_vssl(vrsum23, 4, pOut+outIndex+ 2*oPixels, vl) ;
    _vel_vstl_vssl(vrsum23, 4, pOut+outIndex+ 3*oPixels, vl) ;
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
    const int64_t inChannelGroup,
    const int64_t outChannelGroup,
    const int64_t inGroupOffset,
    const int64_t outGroupOffset,
    const int64_t kernGroupOffset,
    const int64_t oPixels,
    const int64_t n,
    const int64_t k,
    const int64_t op
)
{
  int64_t outIndex = outGroupOffset + (n * outChannel + k  ) * oPixels + op;

    const int64_t vl = oPixels - op < VLEN ? oPixels - op : VLEN ;

    __vr vrsum01 = _vel_pvbrd_vsl(0UL, vl) ;
    __vr vrsum23 = _vel_pvbrd_vsl(0UL, vl) ;
    __vr vrsum45 = _vel_pvbrd_vsl(0UL, vl) ;
    __vr vrsum67 = _vel_pvbrd_vsl(0UL, vl) ;

    int64_t c = 0 ;

    if( ( inChannelGroup & 0x01 ) == 1 ) {
      const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;


      __vr vrin_c0  = _vel_vldu_vssl(4,&pInChannel[op], vl) ;
      __vr vrin_c0P = _vel_vshf_vvvsl(vrin_c0, vrin_c0, VE_VSHUFFLE_YUZU, vl) ;

      const float *pKerValue = pKernel + kernGroupOffset + (k * inChannelGroup + c) ;


      const uint64_t kerValue01_c0 = _vel_pack_f32p(pKerValue   + 0*inChannelGroup,
      				                    pKerValue   + 1*inChannelGroup ) ;
      vrsum01 = _vel_pvfmad_vvsvl(vrsum01, kerValue01_c0, vrin_c0P, vl) ;

      const uint64_t kerValue23_c0 = _vel_pack_f32p(pKerValue   + 2*inChannelGroup,
      				                    pKerValue   + 3*inChannelGroup ) ;
      vrsum23 = _vel_pvfmad_vvsvl(vrsum23, kerValue23_c0, vrin_c0P, vl) ;

      const uint64_t kerValue45_c0 = _vel_pack_f32p(pKerValue   + 4*inChannelGroup,
      				                    pKerValue   + 5*inChannelGroup ) ;
      vrsum45 = _vel_pvfmad_vvsvl(vrsum45, kerValue45_c0, vrin_c0P, vl) ;

      const uint64_t kerValue67_c0 = _vel_pack_f32p(pKerValue   + 6*inChannelGroup,
      				                    pKerValue   + 7*inChannelGroup ) ;
      vrsum67 = _vel_pvfmad_vvsvl(vrsum67, kerValue67_c0, vrin_c0P, vl) ;

      c++ ;
    }
    for( ; c < inChannelGroup ; c+=2 ) {
      const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c  ) * inHeight * inWidth ) ;

      __vr vrin_c0  = _vel_vldu_vssl(4,&pInChannel[op], vl) ;
      __vr vrin_c1  = _vel_vldu_vssl(4,&pInChannel[op + inHeight * inWidth ], vl) ;
      __vr vrin_c0P = _vel_vshf_vvvsl(vrin_c0, vrin_c0, VE_VSHUFFLE_YUZU, vl) ;
      __vr vrin_c1P = _vel_vshf_vvvsl(vrin_c1, vrin_c1, VE_VSHUFFLE_YUZU, vl) ;

      const float *pKerValue = pKernel + kernGroupOffset + (k * inChannelGroup + c) ;


      const uint64_t kerValue01_c0 = _vel_pack_f32p(pKerValue   + 0*inChannelGroup,
      				                    pKerValue   + 1*inChannelGroup ) ;
      const uint64_t kerValue01_c1 = _vel_pack_f32p(pKerValue+1 + 0*inChannelGroup,
      				                    pKerValue+1 + 1*inChannelGroup ) ;
      vrsum01 = _vel_pvfmad_vvsvl(vrsum01, kerValue01_c0, vrin_c0P, vl) ;
      vrsum01 = _vel_pvfmad_vvsvl(vrsum01, kerValue01_c1, vrin_c1P, vl) ;

      const uint64_t kerValue23_c0 = _vel_pack_f32p(pKerValue   + 2*inChannelGroup,
      				                    pKerValue   + 3*inChannelGroup ) ;
      const uint64_t kerValue23_c1 = _vel_pack_f32p(pKerValue+1 + 2*inChannelGroup,
      				                    pKerValue+1 + 3*inChannelGroup ) ;
      vrsum23 = _vel_pvfmad_vvsvl(vrsum23, kerValue23_c0, vrin_c0P, vl) ;
      vrsum23 = _vel_pvfmad_vvsvl(vrsum23, kerValue23_c1, vrin_c1P, vl) ;

      const uint64_t kerValue45_c0 = _vel_pack_f32p(pKerValue   + 4*inChannelGroup,
      				                    pKerValue   + 5*inChannelGroup ) ;
      const uint64_t kerValue45_c1 = _vel_pack_f32p(pKerValue+1 + 4*inChannelGroup,
      				                    pKerValue+1 + 5*inChannelGroup ) ;
      vrsum45 = _vel_pvfmad_vvsvl(vrsum45, kerValue45_c0, vrin_c0P, vl) ;
      vrsum45 = _vel_pvfmad_vvsvl(vrsum45, kerValue45_c1, vrin_c1P, vl) ;

      const uint64_t kerValue67_c0 = _vel_pack_f32p(pKerValue   + 6*inChannelGroup,
      				                    pKerValue   + 7*inChannelGroup ) ;
      const uint64_t kerValue67_c1 = _vel_pack_f32p(pKerValue+1 + 6*inChannelGroup,
      				                    pKerValue+1 + 7*inChannelGroup ) ;
      vrsum67 = _vel_pvfmad_vvsvl(vrsum67, kerValue67_c0, vrin_c0P, vl) ;
      vrsum67 = _vel_pvfmad_vvsvl(vrsum67, kerValue67_c1, vrin_c1P, vl) ;

    } // inChannel

    _vel_vstu_vssl(vrsum01, 4, pOut+outIndex, vl) ;
    _vel_vstl_vssl(vrsum01, 4, pOut+outIndex+   oPixels, vl) ;
    _vel_vstu_vssl(vrsum23, 4, pOut+outIndex+ 2*oPixels, vl) ;
    _vel_vstl_vssl(vrsum23, 4, pOut+outIndex+ 3*oPixels, vl) ;
    _vel_vstu_vssl(vrsum45, 4, pOut+outIndex+ 4*oPixels, vl) ;
    _vel_vstl_vssl(vrsum45, 4, pOut+outIndex+ 5*oPixels, vl) ;
    _vel_vstu_vssl(vrsum67, 4, pOut+outIndex+ 6*oPixels, vl) ;
    _vel_vstl_vssl(vrsum67, 4, pOut+outIndex+ 7*oPixels, vl) ;
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
    const int64_t inChannelGroup,
    const int64_t outChannelGroup,
    const int64_t inGroupOffset,
    const int64_t outGroupOffset,
    const int64_t kernGroupOffset,
    const int64_t oPixels,
    const int64_t n,
    const int64_t k,
    const int64_t op
)
{
  int64_t outIndex = outGroupOffset + (n * outChannel + k  ) * oPixels + op;

    const int64_t vl = oPixels - op < VLEN ? oPixels - op : VLEN ;

    __vr vrsum01 = _vel_pvbrd_vsl(0UL, vl) ;
    __vr vrsum23 = _vel_pvbrd_vsl(0UL, vl) ;
    __vr vrsum45 = _vel_pvbrd_vsl(0UL, vl) ;
    __vr vrsum67 = _vel_pvbrd_vsl(0UL, vl) ;
    __vr vrsum89 = _vel_pvbrd_vsl(0UL, vl) ;
    __vr vrsumAB = _vel_pvbrd_vsl(0UL, vl) ;
    __vr vrsumCD = _vel_pvbrd_vsl(0UL, vl) ;
    __vr vrsumEF = _vel_pvbrd_vsl(0UL, vl) ;

    int64_t c = 0 ;

    if( ( inChannelGroup & 0x01 ) == 1 ) {
      const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;


      __vr vrin_c0  = _vel_vldu_vssl(4,&pInChannel[op], vl) ;
      __vr vrin_c0P = _vel_vshf_vvvsl(vrin_c0, vrin_c0, VE_VSHUFFLE_YUZU, vl) ;

      const float *pKerValue = pKernel + kernGroupOffset + (k * inChannelGroup + c) ;


      const uint64_t kerValue01_c0 = _vel_pack_f32p(pKerValue   + 0*inChannelGroup,
      				                    pKerValue   + 1*inChannelGroup ) ;
      vrsum01 = _vel_pvfmad_vvsvl(vrsum01, kerValue01_c0, vrin_c0P, vl) ;

      const uint64_t kerValue23_c0 = _vel_pack_f32p(pKerValue   + 2*inChannelGroup,
      				                    pKerValue   + 3*inChannelGroup ) ;
      vrsum23 = _vel_pvfmad_vvsvl(vrsum23, kerValue23_c0, vrin_c0P, vl) ;

      const uint64_t kerValue45_c0 = _vel_pack_f32p(pKerValue   + 4*inChannelGroup,
      				                    pKerValue   + 5*inChannelGroup ) ;
      vrsum45 = _vel_pvfmad_vvsvl(vrsum45, kerValue45_c0, vrin_c0P, vl) ;

      const uint64_t kerValue67_c0 = _vel_pack_f32p(pKerValue   + 6*inChannelGroup,
      				                    pKerValue   + 7*inChannelGroup ) ;
      vrsum67 = _vel_pvfmad_vvsvl(vrsum67, kerValue67_c0, vrin_c0P, vl) ;

      const uint64_t kerValue89_c0 = _vel_pack_f32p(pKerValue   + 8*inChannelGroup,
      				                    pKerValue   + 9*inChannelGroup ) ;
      vrsum89 = _vel_pvfmad_vvsvl(vrsum89, kerValue89_c0, vrin_c0P, vl) ;

      const uint64_t kerValueAB_c0 = _vel_pack_f32p(pKerValue   +10*inChannelGroup,
      				                    pKerValue   +11*inChannelGroup ) ;
      vrsumAB = _vel_pvfmad_vvsvl(vrsumAB, kerValueAB_c0, vrin_c0P, vl) ;

      const uint64_t kerValueCD_c0 = _vel_pack_f32p(pKerValue   +12*inChannelGroup,
      				                    pKerValue   +13*inChannelGroup ) ;
      vrsumCD = _vel_pvfmad_vvsvl(vrsumCD, kerValueCD_c0, vrin_c0P, vl) ;

      const uint64_t kerValueEF_c0 = _vel_pack_f32p(pKerValue   +14*inChannelGroup,
      				                    pKerValue   +15*inChannelGroup ) ;
      vrsumEF = _vel_pvfmad_vvsvl(vrsumEF, kerValueEF_c0, vrin_c0P, vl) ;

      c++ ;
    }
    for( ; c < inChannelGroup ; c+=2 ) {
      const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c  ) * inHeight * inWidth ) ;

      __vr vrin_c0  = _vel_vldu_vssl(4,&pInChannel[op], vl) ;
      __vr vrin_c1  = _vel_vldu_vssl(4,&pInChannel[op + inHeight * inWidth ], vl) ;
      __vr vrin_c0P = _vel_vshf_vvvsl(vrin_c0, vrin_c0, VE_VSHUFFLE_YUZU, vl) ;
      __vr vrin_c1P = _vel_vshf_vvvsl(vrin_c1, vrin_c1, VE_VSHUFFLE_YUZU, vl) ;

      const float *pKerValue = pKernel + kernGroupOffset + (k * inChannelGroup + c) ;


      const uint64_t kerValue01_c0 = _vel_pack_f32p(pKerValue   + 0*inChannelGroup,
      				                    pKerValue   + 1*inChannelGroup ) ;
      const uint64_t kerValue01_c1 = _vel_pack_f32p(pKerValue+1 + 0*inChannelGroup,
      				                    pKerValue+1 + 1*inChannelGroup ) ;
      vrsum01 = _vel_pvfmad_vvsvl(vrsum01, kerValue01_c0, vrin_c0P, vl) ;
      vrsum01 = _vel_pvfmad_vvsvl(vrsum01, kerValue01_c1, vrin_c1P, vl) ;

      const uint64_t kerValue23_c0 = _vel_pack_f32p(pKerValue   + 2*inChannelGroup,
      				                    pKerValue   + 3*inChannelGroup ) ;
      const uint64_t kerValue23_c1 = _vel_pack_f32p(pKerValue+1 + 2*inChannelGroup,
      				                    pKerValue+1 + 3*inChannelGroup ) ;
      vrsum23 = _vel_pvfmad_vvsvl(vrsum23, kerValue23_c0, vrin_c0P, vl) ;
      vrsum23 = _vel_pvfmad_vvsvl(vrsum23, kerValue23_c1, vrin_c1P, vl) ;

      const uint64_t kerValue45_c0 = _vel_pack_f32p(pKerValue   + 4*inChannelGroup,
      				                    pKerValue   + 5*inChannelGroup ) ;
      const uint64_t kerValue45_c1 = _vel_pack_f32p(pKerValue+1 + 4*inChannelGroup,
      				                    pKerValue+1 + 5*inChannelGroup ) ;
      vrsum45 = _vel_pvfmad_vvsvl(vrsum45, kerValue45_c0, vrin_c0P, vl) ;
      vrsum45 = _vel_pvfmad_vvsvl(vrsum45, kerValue45_c1, vrin_c1P, vl) ;

      const uint64_t kerValue67_c0 = _vel_pack_f32p(pKerValue   + 6*inChannelGroup,
      				                    pKerValue   + 7*inChannelGroup ) ;
      const uint64_t kerValue67_c1 = _vel_pack_f32p(pKerValue+1 + 6*inChannelGroup,
      				                    pKerValue+1 + 7*inChannelGroup ) ;
      vrsum67 = _vel_pvfmad_vvsvl(vrsum67, kerValue67_c0, vrin_c0P, vl) ;
      vrsum67 = _vel_pvfmad_vvsvl(vrsum67, kerValue67_c1, vrin_c1P, vl) ;

      const uint64_t kerValue89_c0 = _vel_pack_f32p(pKerValue   + 8*inChannelGroup,
      				                    pKerValue   + 9*inChannelGroup ) ;
      const uint64_t kerValue89_c1 = _vel_pack_f32p(pKerValue+1 + 8*inChannelGroup,
      				                    pKerValue+1 + 9*inChannelGroup ) ;
      vrsum89 = _vel_pvfmad_vvsvl(vrsum89, kerValue89_c0, vrin_c0P, vl) ;
      vrsum89 = _vel_pvfmad_vvsvl(vrsum89, kerValue89_c1, vrin_c1P, vl) ;

      const uint64_t kerValueAB_c0 = _vel_pack_f32p(pKerValue   +10*inChannelGroup,
      				                    pKerValue   +11*inChannelGroup ) ;
      const uint64_t kerValueAB_c1 = _vel_pack_f32p(pKerValue+1 +10*inChannelGroup,
      				                    pKerValue+1 +11*inChannelGroup ) ;
      vrsumAB = _vel_pvfmad_vvsvl(vrsumAB, kerValueAB_c0, vrin_c0P, vl) ;
      vrsumAB = _vel_pvfmad_vvsvl(vrsumAB, kerValueAB_c1, vrin_c1P, vl) ;

      const uint64_t kerValueCD_c0 = _vel_pack_f32p(pKerValue   +12*inChannelGroup,
      				                    pKerValue   +13*inChannelGroup ) ;
      const uint64_t kerValueCD_c1 = _vel_pack_f32p(pKerValue+1 +12*inChannelGroup,
      				                    pKerValue+1 +13*inChannelGroup ) ;
      vrsumCD = _vel_pvfmad_vvsvl(vrsumCD, kerValueCD_c0, vrin_c0P, vl) ;
      vrsumCD = _vel_pvfmad_vvsvl(vrsumCD, kerValueCD_c1, vrin_c1P, vl) ;

      const uint64_t kerValueEF_c0 = _vel_pack_f32p(pKerValue   +14*inChannelGroup,
      				                    pKerValue   +15*inChannelGroup ) ;
      const uint64_t kerValueEF_c1 = _vel_pack_f32p(pKerValue+1 +14*inChannelGroup,
      				                    pKerValue+1 +15*inChannelGroup ) ;
      vrsumEF = _vel_pvfmad_vvsvl(vrsumEF, kerValueEF_c0, vrin_c0P, vl) ;
      vrsumEF = _vel_pvfmad_vvsvl(vrsumEF, kerValueEF_c1, vrin_c1P, vl) ;
    } // inChannel

    _vel_vstu_vssl(vrsum01, 4, pOut+outIndex, vl) ;
    _vel_vstl_vssl(vrsum01, 4, pOut+outIndex+   oPixels, vl) ;
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
}

static inline void k16_c1024x(
    const float * restrict pIn,
    const float * restrict pKernel,
    float * restrict const pOut,
    const int64_t inChannel,
    const int64_t inWidth,
    const int64_t inHeight,
    const int64_t outChannel,
    const int64_t outWidth,
    const int64_t outHeight,
    const int64_t inChannelGroup,
    const int64_t outChannelGroup,
    const int64_t inGroupOffset,
    const int64_t outGroupOffset,
    const int64_t kernGroupOffset,
    const int64_t oPixels,
    const int64_t n,
    const int64_t k,
    const int64_t op
)
{

  float __attribute__ ((aligned(8))) filter[16*512] ;

  int64_t outIndex = outGroupOffset + (n * outChannel + k  ) * oPixels + op;

    const int64_t vl = oPixels - op < VLEN ? oPixels - op : VLEN ;

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


      for(int64_t c1 = 0; c1 < 512 ; c1+=2 ) {
	const int64_t c = c0 + c1 ;

	const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;

	const uint64_t* filter_u64 = (const uint64_t*)(filter+c1) ;

	__vr vrin_c0  = _vel_vldu_vssl(4,&pInChannel[op], vl) ;
	__vr vrin_c1  = _vel_vldu_vssl(4,&pInChannel[op +    inHeight * inWidth ], vl) ;

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
      } // inChannel
    }

    _vel_vstu_vssl(vrsum01, 4, pOut+outIndex,            vl) ;
    _vel_vstl_vssl(vrsum01, 4, pOut+outIndex+   oPixels, vl) ;
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
}


vednnError_t
vednnConvolutionForward_direct_dil1_str1_pad0_ker1_T_subkernel(
    const vednnTensorParam_t * restrict 	pParamIn,
    const void * restrict 			pDataIn,
    const vednnFilterParam_t * restrict 	pParamKernel,
    const void * restrict 			pDataKernel,
    const vednnConvolutionParam_t * restrict 	pParamConv,
    const vednnTensorParam_t * restrict 	pParamOut,
    void * restrict 				pDataOut,
    const int                                   n,
    const int                                   g,
    const int                                   curOutChannelGroupPrime,
    const int                                   curOutPixelPrime
)
{
  const int64_t batch      = pParamIn->batch;
  const int64_t inChannel  = pParamIn->channel;
  const int64_t inWidth    = pParamIn->width;
  const int64_t inHeight   = pParamIn->height;
  const int64_t outChannel = pParamOut->channel;
  const int64_t outWidth   = pParamOut->width;
  const int64_t outHeight  = pParamOut->height;
//  const int64_t kernWidth  = pParamKernel->width;		/* must be 1 */
//  const int64_t kernHeight = pParamKernel->height;		/* must be 1 */

  const int64_t group          = pParamConv->group;
//  const int64_t strideWidth    = pParamConv->strideWidth;	/* must be 1 */
//  const int64_t strideHeight   = pParamConv->strideHeight;	/* must be 1 */
//  const int64_t padWidth       = pParamConv->padWidth;	/* must be 0 */
//  const int64_t padHeight      = pParamConv->padHeight;	/* must be 0 */
//  const int64_t dilationWidth  = pParamConv->dilationWidth;	/* must be 1 */
//  const int64_t dilationHeight = pParamConv->dilationHeight;	/* must be 1 */

  const int64_t inChannelGroup  = inChannel  / group;   // equal to pDataKernel->inChannel
  const int64_t outChannelGroup = outChannel / group;   // equal to pDataKernel->outChannel

  const float * restrict pIn     = pDataIn;
  const float * restrict pKernel = pDataKernel;
  float * restrict const pOut    = pDataOut;

  const int oPixels= outHeight*outWidth ;

  {
    const int64_t inGroupOffset   = g * inChannelGroup * inHeight * inWidth;
    const int64_t outGroupOffset  = g * outChannelGroup * outHeight * outWidth;
    const int64_t kernGroupOffset = g * outChannelGroup * inChannelGroup ;

    int64_t op = curOutPixelPrime * VLEN;
    int ocgRemainder = outChannelGroup % 16;

    if(curOutChannelGroupPrime == 0 && ocgRemainder) {
	int k = 0 ;
	if ( (outChannelGroup & 0x01) == 1 ) {
	  k1(pIn, pKernel, pOut,
	     inChannel, inWidth, inHeight,
	     outChannel, outWidth, outHeight,
	     inChannelGroup, outChannelGroup,
	     inGroupOffset, outGroupOffset,
	     kernGroupOffset, oPixels, n, k, op) ;
	  k++ ;
	}
	if ( ((outChannelGroup >> 1) & 0x01) == 1 ) {
	  k2(pIn, pKernel, pOut,
	     inChannel, inWidth, inHeight,
	     outChannel, outWidth, outHeight,
	     inChannelGroup, outChannelGroup,
	     inGroupOffset, outGroupOffset,
	     kernGroupOffset, oPixels, n, k, op) ;
	  k+=2 ;
	}
	if ( ((outChannelGroup >> 2) & 0x01) == 1 ) {
	  k4(pIn, pKernel, pOut,
	     inChannel, inWidth, inHeight,
	     outChannel, outWidth, outHeight,
	     inChannelGroup, outChannelGroup,
	     inGroupOffset, outGroupOffset,
	     kernGroupOffset, oPixels, n, k, op) ;
	  k+=4 ;
	}
	if ( ((outChannelGroup >> 3) & 0x01) == 1 ) {
	  k8(pIn, pKernel, pOut,
	     inChannel, inWidth, inHeight,
	     outChannel, outWidth, outHeight,
	     inChannelGroup, outChannelGroup,
	     inGroupOffset, outGroupOffset,
	     kernGroupOffset, oPixels, n, k, op) ;
	  k+=8 ;
	}
      return VEDNN_SUCCESS;
    }

    int k = ocgRemainder ?
            (curOutChannelGroupPrime-1) * 16 + ocgRemainder :
            curOutChannelGroupPrime * 16 ;

	  if ( inChannelGroup % 1024 == 0 && (((uint64_t)pDataKernel) & 0x7) == 0 ) {
	    k16_c1024x(pIn, pKernel, pOut,
	       inChannel, inWidth, inHeight,
	       outChannel, outWidth, outHeight,
	       inChannelGroup, outChannelGroup,
	       inGroupOffset, outGroupOffset,
	       kernGroupOffset, oPixels, n, k, op);
	  }
	  else
	  {
	    k16(pIn, pKernel, pOut,
	       inChannel, inWidth, inHeight,
	       outChannel, outWidth, outHeight,
	       inChannelGroup, outChannelGroup,
	       inGroupOffset, outGroupOffset,
	       kernGroupOffset, oPixels, n, k, op) ;
	  }
  }

  return VEDNN_SUCCESS;
}
