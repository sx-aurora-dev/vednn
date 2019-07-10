#include <stdint.h>
#include <stdlib.h>

#include "vednn.h"

#include "veintrin.h"
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
    const int64_t k
)
{
  int64_t outIndex = outGroupOffset + (n * outChannel + k) * oPixels ;

  for (int64_t op = 0; op < oPixels; op+=VLEN) {
    const int64_t vl = oPixels - op < VLEN ? oPixels - op : VLEN ;

    _ve_lvl(vl) ;

    __vr vrsum = _ve_vbrdu_vs_f32(0.0f) ;

    int64_t c = 0 ;
    if( ( inChannelGroup & 0x01 ) == 1 ) {
      const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;

      __vr vrin  = _ve_vldu_vss(4,&pInChannel[op]) ;
      const float *pKerValue = pKernel + kernGroupOffset + (k * inChannelGroup + c);

      vrsum = _ve_vfmads_vvsv(vrsum, pKerValue[0], vrin) ;

      c++ ;
    }
    for( ; c < inChannelGroup ; c+=2 ) {
      const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c  ) * inHeight * inWidth ) ;

      __vr vrin0  = _ve_vldu_vss(4,&pInChannel[op]) ;
      __vr vrin1  = _ve_vldu_vss(4,&pInChannel[op + inHeight * inWidth ]) ;

      const float *pKerValue = pKernel + kernGroupOffset + (k * inChannelGroup + c) ;

      vrsum = _ve_vfmads_vvsv(vrsum, pKerValue[0], vrin0) ;
      vrsum = _ve_vfmads_vvsv(vrsum, pKerValue[1], vrin1) ;
    } // inChannel

    _ve_vstu_vss(vrsum, 4, pOut+outIndex) ;

    outIndex += vl ;
  } // outPixels
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
    const int64_t k
)
{
  int64_t outIndex = outGroupOffset + (n * outChannel + k  ) * oPixels ;

  for (int64_t op = 0; op < oPixels; op+=VLEN) {
    const int64_t vl = oPixels - op < VLEN ? oPixels - op : VLEN ;

    _ve_lvl(vl) ;

    __vr vrsum01 = _ve_pvbrd_vs_i64(0UL) ;

    int64_t c = 0 ;

    if( ( inChannelGroup & 0x01 ) == 1 ) {
      const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;


      __vr vrin_c0  = _ve_vldu_vss(4,&pInChannel[op]) ;
      __vr vrin_c0P = _ve_vshf_vvvs(vrin_c0, vrin_c0, VE_VSHUFFLE_YUZU) ;

      const float *pKerValue = pKernel + kernGroupOffset + (k * inChannelGroup + c) ;


      const uint64_t kerValue01_c0 = _ve_pack_f32p(pKerValue   + 0*inChannelGroup,
      				                   pKerValue   + 1*inChannelGroup ) ;
      vrsum01 = _ve_pvfmad_vvsv(vrsum01, kerValue01_c0, vrin_c0P) ;

      c++ ;
    }
    for( ; c < inChannelGroup ; c+=2 ) {
      const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c  ) * inHeight * inWidth ) ;

      __vr vrin_c0  = _ve_vldu_vss(4,&pInChannel[op]) ;
      __vr vrin_c1  = _ve_vldu_vss(4,&pInChannel[op + inHeight * inWidth ]) ;
      __vr vrin_c0P = _ve_vshf_vvvs(vrin_c0, vrin_c0, VE_VSHUFFLE_YUZU) ;
      __vr vrin_c1P = _ve_vshf_vvvs(vrin_c1, vrin_c1, VE_VSHUFFLE_YUZU) ;

      const float *pKerValue = pKernel + kernGroupOffset + (k * inChannelGroup + c) ;


      const uint64_t kerValue01_c0 = _ve_pack_f32p(pKerValue   + 0*inChannelGroup,
      				                   pKerValue   + 1*inChannelGroup ) ;
      const uint64_t kerValue01_c1 = _ve_pack_f32p(pKerValue+1 + 0*inChannelGroup,
      				                   pKerValue+1 + 1*inChannelGroup ) ;
      vrsum01 = _ve_pvfmad_vvsv(vrsum01, kerValue01_c0, vrin_c0P) ;
      vrsum01 = _ve_pvfmad_vvsv(vrsum01, kerValue01_c1, vrin_c1P) ;

    } // inChannel

    _ve_vstu_vss(vrsum01, 4, pOut+outIndex) ;
    _ve_vstl_vss(vrsum01, 4, pOut+outIndex+   oPixels) ;

    outIndex += vl ;
  } // outPixels
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
    const int64_t k
)
{
  int64_t outIndex = outGroupOffset + (n * outChannel + k  ) * oPixels ;

  for (int64_t op = 0; op < oPixels; op+=VLEN) {
    const int64_t vl = oPixels - op < VLEN ? oPixels - op : VLEN ;

    _ve_lvl(vl) ;

    __vr vrsum01 = _ve_pvbrd_vs_i64(0UL) ;
    __vr vrsum23 = _ve_pvbrd_vs_i64(0UL) ;

    int64_t c = 0 ;

    if( ( inChannelGroup & 0x01 ) == 1 ) {
      const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;


      __vr vrin_c0  = _ve_vldu_vss(4,&pInChannel[op]) ;
      __vr vrin_c0P = _ve_vshf_vvvs(vrin_c0, vrin_c0, VE_VSHUFFLE_YUZU) ;

      const float *pKerValue = pKernel + kernGroupOffset + (k * inChannelGroup + c) ;


      const uint64_t kerValue01_c0 = _ve_pack_f32p(pKerValue   + 0*inChannelGroup,
      				                   pKerValue   + 1*inChannelGroup ) ;
      vrsum01 = _ve_pvfmad_vvsv(vrsum01, kerValue01_c0, vrin_c0P) ;

      const uint64_t kerValue23_c0 = _ve_pack_f32p(pKerValue   + 2*inChannelGroup,
      				                   pKerValue   + 3*inChannelGroup ) ;
      vrsum23 = _ve_pvfmad_vvsv(vrsum23, kerValue23_c0, vrin_c0P) ;

      c++ ;
    }
    for( ; c < inChannelGroup ; c+=2 ) {
      const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c  ) * inHeight * inWidth ) ;

      __vr vrin_c0  = _ve_vldu_vss(4,&pInChannel[op]) ;
      __vr vrin_c1  = _ve_vldu_vss(4,&pInChannel[op + inHeight * inWidth ]) ;
      __vr vrin_c0P = _ve_vshf_vvvs(vrin_c0, vrin_c0, VE_VSHUFFLE_YUZU) ;
      __vr vrin_c1P = _ve_vshf_vvvs(vrin_c1, vrin_c1, VE_VSHUFFLE_YUZU) ;

      const float *pKerValue = pKernel + kernGroupOffset + (k * inChannelGroup + c) ;


      const uint64_t kerValue01_c0 = _ve_pack_f32p(pKerValue   + 0*inChannelGroup,
      				                   pKerValue   + 1*inChannelGroup ) ;
      const uint64_t kerValue01_c1 = _ve_pack_f32p(pKerValue+1 + 0*inChannelGroup,
      				                   pKerValue+1 + 1*inChannelGroup ) ;
      vrsum01 = _ve_pvfmad_vvsv(vrsum01, kerValue01_c0, vrin_c0P) ;
      vrsum01 = _ve_pvfmad_vvsv(vrsum01, kerValue01_c1, vrin_c1P) ;

      const uint64_t kerValue23_c0 = _ve_pack_f32p(pKerValue   + 2*inChannelGroup,
      				                   pKerValue   + 3*inChannelGroup ) ;
      const uint64_t kerValue23_c1 = _ve_pack_f32p(pKerValue+1 + 2*inChannelGroup,
      				                   pKerValue+1 + 3*inChannelGroup ) ;
      vrsum23 = _ve_pvfmad_vvsv(vrsum23, kerValue23_c0, vrin_c0P) ;
      vrsum23 = _ve_pvfmad_vvsv(vrsum23, kerValue23_c1, vrin_c1P) ;
    } // inChannel

    _ve_vstu_vss(vrsum01, 4, pOut+outIndex) ;
    _ve_vstl_vss(vrsum01, 4, pOut+outIndex+   oPixels) ;
    _ve_vstu_vss(vrsum23, 4, pOut+outIndex+ 2*oPixels) ;
    _ve_vstl_vss(vrsum23, 4, pOut+outIndex+ 3*oPixels) ;

    outIndex += vl ;
  } // outPixels
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
    const int64_t k
)
{
  int64_t outIndex = outGroupOffset + (n * outChannel + k  ) * oPixels ;

  for (int64_t op = 0; op < oPixels; op+=VLEN) {
    const int64_t vl = oPixels - op < VLEN ? oPixels - op : VLEN ;

    _ve_lvl(vl) ;

    __vr vrsum01 = _ve_pvbrd_vs_i64(0UL) ;
    __vr vrsum23 = _ve_pvbrd_vs_i64(0UL) ;
    __vr vrsum45 = _ve_pvbrd_vs_i64(0UL) ;
    __vr vrsum67 = _ve_pvbrd_vs_i64(0UL) ;

    int64_t c = 0 ;

    if( ( inChannelGroup & 0x01 ) == 1 ) {
      const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;


      __vr vrin_c0  = _ve_vldu_vss(4,&pInChannel[op]) ;
      __vr vrin_c0P = _ve_vshf_vvvs(vrin_c0, vrin_c0, VE_VSHUFFLE_YUZU) ;

      const float *pKerValue = pKernel + kernGroupOffset + (k * inChannelGroup + c) ;


      const uint64_t kerValue01_c0 = _ve_pack_f32p(pKerValue   + 0*inChannelGroup,
      				                   pKerValue   + 1*inChannelGroup ) ;
      vrsum01 = _ve_pvfmad_vvsv(vrsum01, kerValue01_c0, vrin_c0P) ;

      const uint64_t kerValue23_c0 = _ve_pack_f32p(pKerValue   + 2*inChannelGroup,
      				                   pKerValue   + 3*inChannelGroup ) ;
      vrsum23 = _ve_pvfmad_vvsv(vrsum23, kerValue23_c0, vrin_c0P) ;

      const uint64_t kerValue45_c0 = _ve_pack_f32p(pKerValue   + 4*inChannelGroup,
      				                   pKerValue   + 5*inChannelGroup ) ;
      vrsum45 = _ve_pvfmad_vvsv(vrsum45, kerValue45_c0, vrin_c0P) ;

      const uint64_t kerValue67_c0 = _ve_pack_f32p(pKerValue   + 6*inChannelGroup,
      				                   pKerValue   + 7*inChannelGroup ) ;
      vrsum67 = _ve_pvfmad_vvsv(vrsum67, kerValue67_c0, vrin_c0P) ;

      c++ ;
    }
    for( ; c < inChannelGroup ; c+=2 ) {
      const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c  ) * inHeight * inWidth ) ;

      __vr vrin_c0  = _ve_vldu_vss(4,&pInChannel[op]) ;
      __vr vrin_c1  = _ve_vldu_vss(4,&pInChannel[op + inHeight * inWidth ]) ;
      __vr vrin_c0P = _ve_vshf_vvvs(vrin_c0, vrin_c0, VE_VSHUFFLE_YUZU) ;
      __vr vrin_c1P = _ve_vshf_vvvs(vrin_c1, vrin_c1, VE_VSHUFFLE_YUZU) ;

      const float *pKerValue = pKernel + kernGroupOffset + (k * inChannelGroup + c) ;


      const uint64_t kerValue01_c0 = _ve_pack_f32p(pKerValue   + 0*inChannelGroup,
      				                   pKerValue   + 1*inChannelGroup ) ;
      const uint64_t kerValue01_c1 = _ve_pack_f32p(pKerValue+1 + 0*inChannelGroup,
      				                   pKerValue+1 + 1*inChannelGroup ) ;
      vrsum01 = _ve_pvfmad_vvsv(vrsum01, kerValue01_c0, vrin_c0P) ;
      vrsum01 = _ve_pvfmad_vvsv(vrsum01, kerValue01_c1, vrin_c1P) ;

      const uint64_t kerValue23_c0 = _ve_pack_f32p(pKerValue   + 2*inChannelGroup,
      				                   pKerValue   + 3*inChannelGroup ) ;
      const uint64_t kerValue23_c1 = _ve_pack_f32p(pKerValue+1 + 2*inChannelGroup,
      				                   pKerValue+1 + 3*inChannelGroup ) ;
      vrsum23 = _ve_pvfmad_vvsv(vrsum23, kerValue23_c0, vrin_c0P) ;
      vrsum23 = _ve_pvfmad_vvsv(vrsum23, kerValue23_c1, vrin_c1P) ;

      const uint64_t kerValue45_c0 = _ve_pack_f32p(pKerValue   + 4*inChannelGroup,
      				                   pKerValue   + 5*inChannelGroup ) ;
      const uint64_t kerValue45_c1 = _ve_pack_f32p(pKerValue+1 + 4*inChannelGroup,
      				                   pKerValue+1 + 5*inChannelGroup ) ;
      vrsum45 = _ve_pvfmad_vvsv(vrsum45, kerValue45_c0, vrin_c0P) ;
      vrsum45 = _ve_pvfmad_vvsv(vrsum45, kerValue45_c1, vrin_c1P) ;

      const uint64_t kerValue67_c0 = _ve_pack_f32p(pKerValue   + 6*inChannelGroup,
      				                   pKerValue   + 7*inChannelGroup ) ;
      const uint64_t kerValue67_c1 = _ve_pack_f32p(pKerValue+1 + 6*inChannelGroup,
      				                   pKerValue+1 + 7*inChannelGroup ) ;
      vrsum67 = _ve_pvfmad_vvsv(vrsum67, kerValue67_c0, vrin_c0P) ;
      vrsum67 = _ve_pvfmad_vvsv(vrsum67, kerValue67_c1, vrin_c1P) ;

    } // inChannel

    _ve_vstu_vss(vrsum01, 4, pOut+outIndex) ;
    _ve_vstl_vss(vrsum01, 4, pOut+outIndex+   oPixels) ;
    _ve_vstu_vss(vrsum23, 4, pOut+outIndex+ 2*oPixels) ;
    _ve_vstl_vss(vrsum23, 4, pOut+outIndex+ 3*oPixels) ;
    _ve_vstu_vss(vrsum45, 4, pOut+outIndex+ 4*oPixels) ;
    _ve_vstl_vss(vrsum45, 4, pOut+outIndex+ 5*oPixels) ;
    _ve_vstu_vss(vrsum67, 4, pOut+outIndex+ 6*oPixels) ;
    _ve_vstl_vss(vrsum67, 4, pOut+outIndex+ 7*oPixels) ;

    outIndex += vl ;
  } // outPixels
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
    const int64_t k
)
{
  int64_t outIndex = outGroupOffset + (n * outChannel + k  ) * oPixels ;

  for (int64_t op = 0; op < oPixels; op+=VLEN) {
    const int64_t vl = oPixels - op < VLEN ? oPixels - op : VLEN ;

    _ve_lvl(vl) ;

    __vr vrsum01 = _ve_pvbrd_vs_i64(0UL) ;
    __vr vrsum23 = _ve_pvbrd_vs_i64(0UL) ;
    __vr vrsum45 = _ve_pvbrd_vs_i64(0UL) ;
    __vr vrsum67 = _ve_pvbrd_vs_i64(0UL) ;
    __vr vrsum89 = _ve_pvbrd_vs_i64(0UL) ;
    __vr vrsumAB = _ve_pvbrd_vs_i64(0UL) ;
    __vr vrsumCD = _ve_pvbrd_vs_i64(0UL) ;
    __vr vrsumEF = _ve_pvbrd_vs_i64(0UL) ;

    int64_t c = 0 ;

    if( ( inChannelGroup & 0x01 ) == 1 ) {
      const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;


      __vr vrin_c0  = _ve_vldu_vss(4,&pInChannel[op]) ;
      __vr vrin_c0P = _ve_vshf_vvvs(vrin_c0, vrin_c0, VE_VSHUFFLE_YUZU) ;

      const float *pKerValue = pKernel + kernGroupOffset + (k * inChannelGroup + c) ;


      const uint64_t kerValue01_c0 = _ve_pack_f32p(pKerValue   + 0*inChannelGroup,
      				                   pKerValue   + 1*inChannelGroup ) ;
      vrsum01 = _ve_pvfmad_vvsv(vrsum01, kerValue01_c0, vrin_c0P) ;

      const uint64_t kerValue23_c0 = _ve_pack_f32p(pKerValue   + 2*inChannelGroup,
      				                   pKerValue   + 3*inChannelGroup ) ;
      vrsum23 = _ve_pvfmad_vvsv(vrsum23, kerValue23_c0, vrin_c0P) ;

      const uint64_t kerValue45_c0 = _ve_pack_f32p(pKerValue   + 4*inChannelGroup,
      				                   pKerValue   + 5*inChannelGroup ) ;
      vrsum45 = _ve_pvfmad_vvsv(vrsum45, kerValue45_c0, vrin_c0P) ;

      const uint64_t kerValue67_c0 = _ve_pack_f32p(pKerValue   + 6*inChannelGroup,
      				                   pKerValue   + 7*inChannelGroup ) ;
      vrsum67 = _ve_pvfmad_vvsv(vrsum67, kerValue67_c0, vrin_c0P) ;

      const uint64_t kerValue89_c0 = _ve_pack_f32p(pKerValue   + 8*inChannelGroup,
      				                   pKerValue   + 9*inChannelGroup ) ;
      vrsum89 = _ve_pvfmad_vvsv(vrsum89, kerValue89_c0, vrin_c0P) ;

      const uint64_t kerValueAB_c0 = _ve_pack_f32p(pKerValue   +10*inChannelGroup,
      				                   pKerValue   +11*inChannelGroup ) ;
      vrsumAB = _ve_pvfmad_vvsv(vrsumAB, kerValueAB_c0, vrin_c0P) ;

      const uint64_t kerValueCD_c0 = _ve_pack_f32p(pKerValue   +12*inChannelGroup,
      				                   pKerValue   +13*inChannelGroup ) ;
      vrsumCD = _ve_pvfmad_vvsv(vrsumCD, kerValueCD_c0, vrin_c0P) ;

      const uint64_t kerValueEF_c0 = _ve_pack_f32p(pKerValue   +14*inChannelGroup,
      				                   pKerValue   +15*inChannelGroup ) ;
      vrsumEF = _ve_pvfmad_vvsv(vrsumEF, kerValueEF_c0, vrin_c0P) ;

      c++ ;
    }
    for( ; c < inChannelGroup ; c+=2 ) {
      const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c  ) * inHeight * inWidth ) ;

      __vr vrin_c0  = _ve_vldu_vss(4,&pInChannel[op]) ;
      __vr vrin_c1  = _ve_vldu_vss(4,&pInChannel[op + inHeight * inWidth ]) ;
      __vr vrin_c0P = _ve_vshf_vvvs(vrin_c0, vrin_c0, VE_VSHUFFLE_YUZU) ;
      __vr vrin_c1P = _ve_vshf_vvvs(vrin_c1, vrin_c1, VE_VSHUFFLE_YUZU) ;

      const float *pKerValue = pKernel + kernGroupOffset + (k * inChannelGroup + c) ;


      const uint64_t kerValue01_c0 = _ve_pack_f32p(pKerValue   + 0*inChannelGroup,
      				                   pKerValue   + 1*inChannelGroup ) ;
      const uint64_t kerValue01_c1 = _ve_pack_f32p(pKerValue+1 + 0*inChannelGroup,
      				                   pKerValue+1 + 1*inChannelGroup ) ;
      vrsum01 = _ve_pvfmad_vvsv(vrsum01, kerValue01_c0, vrin_c0P) ;
      vrsum01 = _ve_pvfmad_vvsv(vrsum01, kerValue01_c1, vrin_c1P) ;

      const uint64_t kerValue23_c0 = _ve_pack_f32p(pKerValue   + 2*inChannelGroup,
      				                   pKerValue   + 3*inChannelGroup ) ;
      const uint64_t kerValue23_c1 = _ve_pack_f32p(pKerValue+1 + 2*inChannelGroup,
      				                   pKerValue+1 + 3*inChannelGroup ) ;
      vrsum23 = _ve_pvfmad_vvsv(vrsum23, kerValue23_c0, vrin_c0P) ;
      vrsum23 = _ve_pvfmad_vvsv(vrsum23, kerValue23_c1, vrin_c1P) ;

      const uint64_t kerValue45_c0 = _ve_pack_f32p(pKerValue   + 4*inChannelGroup,
      				                   pKerValue   + 5*inChannelGroup ) ;
      const uint64_t kerValue45_c1 = _ve_pack_f32p(pKerValue+1 + 4*inChannelGroup,
      				                   pKerValue+1 + 5*inChannelGroup ) ;
      vrsum45 = _ve_pvfmad_vvsv(vrsum45, kerValue45_c0, vrin_c0P) ;
      vrsum45 = _ve_pvfmad_vvsv(vrsum45, kerValue45_c1, vrin_c1P) ;

      const uint64_t kerValue67_c0 = _ve_pack_f32p(pKerValue   + 6*inChannelGroup,
      				                   pKerValue   + 7*inChannelGroup ) ;
      const uint64_t kerValue67_c1 = _ve_pack_f32p(pKerValue+1 + 6*inChannelGroup,
      				                   pKerValue+1 + 7*inChannelGroup ) ;
      vrsum67 = _ve_pvfmad_vvsv(vrsum67, kerValue67_c0, vrin_c0P) ;
      vrsum67 = _ve_pvfmad_vvsv(vrsum67, kerValue67_c1, vrin_c1P) ;

      const uint64_t kerValue89_c0 = _ve_pack_f32p(pKerValue   + 8*inChannelGroup,
      				                   pKerValue   + 9*inChannelGroup ) ;
      const uint64_t kerValue89_c1 = _ve_pack_f32p(pKerValue+1 + 8*inChannelGroup,
      				                   pKerValue+1 + 9*inChannelGroup ) ;
      vrsum89 = _ve_pvfmad_vvsv(vrsum89, kerValue89_c0, vrin_c0P) ;
      vrsum89 = _ve_pvfmad_vvsv(vrsum89, kerValue89_c1, vrin_c1P) ;

      const uint64_t kerValueAB_c0 = _ve_pack_f32p(pKerValue   +10*inChannelGroup,
      				                   pKerValue   +11*inChannelGroup ) ;
      const uint64_t kerValueAB_c1 = _ve_pack_f32p(pKerValue+1 +10*inChannelGroup,
      				                   pKerValue+1 +11*inChannelGroup ) ;
      vrsumAB = _ve_pvfmad_vvsv(vrsumAB, kerValueAB_c0, vrin_c0P) ;
      vrsumAB = _ve_pvfmad_vvsv(vrsumAB, kerValueAB_c1, vrin_c1P) ;

      const uint64_t kerValueCD_c0 = _ve_pack_f32p(pKerValue   +12*inChannelGroup,
      				                   pKerValue   +13*inChannelGroup ) ;
      const uint64_t kerValueCD_c1 = _ve_pack_f32p(pKerValue+1 +12*inChannelGroup,
      				                   pKerValue+1 +13*inChannelGroup ) ;
      vrsumCD = _ve_pvfmad_vvsv(vrsumCD, kerValueCD_c0, vrin_c0P) ;
      vrsumCD = _ve_pvfmad_vvsv(vrsumCD, kerValueCD_c1, vrin_c1P) ;

      const uint64_t kerValueEF_c0 = _ve_pack_f32p(pKerValue   +14*inChannelGroup,
      				                   pKerValue   +15*inChannelGroup ) ;
      const uint64_t kerValueEF_c1 = _ve_pack_f32p(pKerValue+1 +14*inChannelGroup,
      				                   pKerValue+1 +15*inChannelGroup ) ;
      vrsumEF = _ve_pvfmad_vvsv(vrsumEF, kerValueEF_c0, vrin_c0P) ;
      vrsumEF = _ve_pvfmad_vvsv(vrsumEF, kerValueEF_c1, vrin_c1P) ;
    } // inChannel

    _ve_vstu_vss(vrsum01, 4, pOut+outIndex) ;
    _ve_vstl_vss(vrsum01, 4, pOut+outIndex+   oPixels) ;
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
    float * restrict const filter
)
{

  int64_t outIndex = outGroupOffset + (n * outChannel + k  ) * oPixels ;

  for (int64_t op = 0; op < oPixels; op+=VLEN) {
    const int64_t vl = oPixels - op < VLEN ? oPixels - op : VLEN ;

    _ve_lvl(vl) ;

    __vr vrsum01 = _ve_pvbrd_vs_i64(0UL) ;
    __vr vrsum23 = _ve_pvbrd_vs_i64(0UL) ;
    __vr vrsum45 = _ve_pvbrd_vs_i64(0UL) ;
    __vr vrsum67 = _ve_pvbrd_vs_i64(0UL) ;
    __vr vrsum89 = _ve_pvbrd_vs_i64(0UL) ;
    __vr vrsumAB = _ve_pvbrd_vs_i64(0UL) ;
    __vr vrsumCD = _ve_pvbrd_vs_i64(0UL) ;
    __vr vrsumEF = _ve_pvbrd_vs_i64(0UL) ;

    for(int64_t c0=0; c0<inChannelGroup; c0+=512) {

      const float *pKerValue = pKernel + kernGroupOffset + (k * inChannelGroup + c0) ;
      _ve_lvl(256) ;

      __vr vr0 = _ve_vld_vss(8, pKerValue+ 0*inChannelGroup) ;
      __vr vr1 = _ve_vld_vss(8, pKerValue+ 1*inChannelGroup) ;
      __vr vr2 = _ve_vld_vss(8, pKerValue+ 2*inChannelGroup) ;
      __vr vr3 = _ve_vld_vss(8, pKerValue+ 3*inChannelGroup) ;
      __vr vr4 = _ve_vld_vss(8, pKerValue+ 4*inChannelGroup) ;
      __vr vr5 = _ve_vld_vss(8, pKerValue+ 5*inChannelGroup) ;
      __vr vr6 = _ve_vld_vss(8, pKerValue+ 6*inChannelGroup) ;
      __vr vr7 = _ve_vld_vss(8, pKerValue+ 7*inChannelGroup) ;
      __vr vr8 = _ve_vld_vss(8, pKerValue+ 8*inChannelGroup) ;
      __vr vr9 = _ve_vld_vss(8, pKerValue+ 9*inChannelGroup) ;
      __vr vrA = _ve_vld_vss(8, pKerValue+10*inChannelGroup) ;
      __vr vrB = _ve_vld_vss(8, pKerValue+11*inChannelGroup) ;
      __vr vrC = _ve_vld_vss(8, pKerValue+12*inChannelGroup) ;
      __vr vrD = _ve_vld_vss(8, pKerValue+13*inChannelGroup) ;
      __vr vrE = _ve_vld_vss(8, pKerValue+14*inChannelGroup) ;
      __vr vrF = _ve_vld_vss(8, pKerValue+15*inChannelGroup) ;

      __vr vr01_c0 = _ve_vshf_vvvs(vr0,vr1,VE_VSHUFFLE_YLZL) ;
      __vr vr23_c0 = _ve_vshf_vvvs(vr2,vr3,VE_VSHUFFLE_YLZL) ;
      __vr vr45_c0 = _ve_vshf_vvvs(vr4,vr5,VE_VSHUFFLE_YLZL) ;
      __vr vr67_c0 = _ve_vshf_vvvs(vr6,vr7,VE_VSHUFFLE_YLZL) ;
      __vr vr89_c0 = _ve_vshf_vvvs(vr8,vr9,VE_VSHUFFLE_YLZL) ;
      __vr vrAB_c0 = _ve_vshf_vvvs(vrA,vrB,VE_VSHUFFLE_YLZL) ;
      __vr vrCD_c0 = _ve_vshf_vvvs(vrC,vrD,VE_VSHUFFLE_YLZL) ;
      __vr vrEF_c0 = _ve_vshf_vvvs(vrE,vrF,VE_VSHUFFLE_YLZL) ;

      _ve_vst_vss(vr01_c0, 8, filter) ;
      _ve_vst_vss(vr23_c0, 8, filter+1*512) ;
      _ve_vst_vss(vr45_c0, 8, filter+2*512) ;
      _ve_vst_vss(vr67_c0, 8, filter+3*512) ;
      _ve_vst_vss(vr89_c0, 8, filter+4*512) ;
      _ve_vst_vss(vrAB_c0, 8, filter+5*512) ;
      _ve_vst_vss(vrCD_c0, 8, filter+6*512) ;
      _ve_vst_vss(vrEF_c0, 8, filter+7*512) ;

      __vr vr01_c1 = _ve_vshf_vvvs(vr0,vr1,VE_VSHUFFLE_YUZU) ;
      __vr vr23_c1 = _ve_vshf_vvvs(vr2,vr3,VE_VSHUFFLE_YUZU) ;
      __vr vr45_c1 = _ve_vshf_vvvs(vr4,vr5,VE_VSHUFFLE_YUZU) ;
      __vr vr67_c1 = _ve_vshf_vvvs(vr6,vr7,VE_VSHUFFLE_YUZU) ;
      __vr vr89_c1 = _ve_vshf_vvvs(vr8,vr9,VE_VSHUFFLE_YUZU) ;
      __vr vrAB_c1 = _ve_vshf_vvvs(vrA,vrB,VE_VSHUFFLE_YUZU) ;
      __vr vrCD_c1 = _ve_vshf_vvvs(vrC,vrD,VE_VSHUFFLE_YUZU) ;
      __vr vrEF_c1 = _ve_vshf_vvvs(vrE,vrF,VE_VSHUFFLE_YUZU) ;

      _ve_vst_vss(vr01_c1, 8, filter+ 8*512) ;
      _ve_vst_vss(vr23_c1, 8, filter+ 9*512) ;
      _ve_vst_vss(vr45_c1, 8, filter+10*512) ;
      _ve_vst_vss(vr67_c1, 8, filter+11*512) ;
      _ve_vst_vss(vr89_c1, 8, filter+12*512) ;
      _ve_vst_vss(vrAB_c1, 8, filter+13*512) ;
      _ve_vst_vss(vrCD_c1, 8, filter+14*512) ;
      _ve_vst_vss(vrEF_c1, 8, filter+15*512) ;

      _ve_lvl(vl) ;

      for(int64_t c1 = 0; c1 < 512 ; c1+=2 ) {
	const int64_t c = c0 + c1 ;

	const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;

	const uint64_t* filter_u64 = (const uint64_t*)(filter+c1) ;

	// _ve_lvl(vl) ;
	__vr vrin_c0  = _ve_vldu_vss(4,&pInChannel[op]) ;
	__vr vrin_c1  = _ve_vldu_vss(4,&pInChannel[op +    inHeight * inWidth ]) ;

	__vr vrin_c0P = _ve_vshf_vvvs(vrin_c0, vrin_c0, VE_VSHUFFLE_YUZU) ;
	vrsum01 = _ve_pvfmad_vvsv(vrsum01, filter_u64[0*256], vrin_c0P) ;
	vrsum23 = _ve_pvfmad_vvsv(vrsum23, filter_u64[1*256], vrin_c0P) ;
	vrsum45 = _ve_pvfmad_vvsv(vrsum45, filter_u64[2*256], vrin_c0P) ;
	vrsum67 = _ve_pvfmad_vvsv(vrsum67, filter_u64[3*256], vrin_c0P) ;
	vrsum89 = _ve_pvfmad_vvsv(vrsum89, filter_u64[4*256], vrin_c0P) ;
	vrsumAB = _ve_pvfmad_vvsv(vrsumAB, filter_u64[5*256], vrin_c0P) ;
	vrsumCD = _ve_pvfmad_vvsv(vrsumCD, filter_u64[6*256], vrin_c0P) ;
	vrsumEF = _ve_pvfmad_vvsv(vrsumEF, filter_u64[7*256], vrin_c0P) ;

	__vr vrin_c1P = _ve_vshf_vvvs(vrin_c1, vrin_c1, VE_VSHUFFLE_YUZU) ;
	vrsum01 = _ve_pvfmad_vvsv(vrsum01, filter_u64[8*256],  vrin_c1P) ;
	vrsum23 = _ve_pvfmad_vvsv(vrsum23, filter_u64[9*256],  vrin_c1P) ;
	vrsum45 = _ve_pvfmad_vvsv(vrsum45, filter_u64[10*256], vrin_c1P) ;
	vrsum67 = _ve_pvfmad_vvsv(vrsum67, filter_u64[11*256], vrin_c1P) ;
	vrsum89 = _ve_pvfmad_vvsv(vrsum89, filter_u64[12*256], vrin_c1P) ;
	vrsumAB = _ve_pvfmad_vvsv(vrsumAB, filter_u64[13*256], vrin_c1P) ;
	vrsumCD = _ve_pvfmad_vvsv(vrsumCD, filter_u64[14*256], vrin_c1P) ;
	vrsumEF = _ve_pvfmad_vvsv(vrsumEF, filter_u64[15*256], vrin_c1P) ;
      } // inChannel
    }

    // _ve_lvl(vl) ;
    _ve_vstu_vss(vrsum01, 4, pOut+outIndex) ;
    _ve_vstl_vss(vrsum01, 4, pOut+outIndex+   oPixels) ;
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
}


vednnError_t
vednnConvolutionForward_direct_dil1_str1_pad0_ker1(
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

  float __attribute__ ((aligned(8))) filter[16*512] ;

  {
    for (int64_t n = 0; n < batch; n++) {
      for (int64_t g = 0; g < group; g++) {
	const int64_t inGroupOffset   = g * inChannelGroup * inHeight * inWidth;
	const int64_t outGroupOffset  = g * outChannelGroup * outHeight * outWidth;
	const int64_t kernGroupOffset = g * outChannelGroup * inChannelGroup ;

	int k = 0 ;
	if ( (outChannelGroup & 0x01) == 1 ) {
	  k1(pIn, pKernel, pOut,
	     inChannel, inWidth, inHeight,
	     outChannel, outWidth, outHeight,
	     inChannelGroup, outChannelGroup,
	     inGroupOffset, outGroupOffset,
	     kernGroupOffset, oPixels, n, k) ;
	  k++ ;
	}
	if ( ((outChannelGroup >> 1) & 0x01) == 1 ) {
	  k2(pIn, pKernel, pOut,
	     inChannel, inWidth, inHeight,
	     outChannel, outWidth, outHeight,
	     inChannelGroup, outChannelGroup,
	     inGroupOffset, outGroupOffset,
	     kernGroupOffset, oPixels, n, k) ;
	  k+=2 ;
	}
	if ( ((outChannelGroup >> 2) & 0x01) == 1 ) {
	  k4(pIn, pKernel, pOut,
	     inChannel, inWidth, inHeight,
	     outChannel, outWidth, outHeight,
	     inChannelGroup, outChannelGroup,
	     inGroupOffset, outGroupOffset,
	     kernGroupOffset, oPixels, n, k) ;
	  k+=4 ;
	}
	if ( ((outChannelGroup >> 3) & 0x01) == 1 ) {
	  k8(pIn, pKernel, pOut,
	     inChannel, inWidth, inHeight,
	     outChannel, outWidth, outHeight,
	     inChannelGroup, outChannelGroup,
	     inGroupOffset, outGroupOffset,
	     kernGroupOffset, oPixels, n, k) ;
	  k+=8 ;
	}
	for ( ; k < outChannelGroup; k+=16 ) {
	  if ( inChannelGroup % 1024 == 0 && (((uint64_t)pDataKernel) & 0x7) == 0 ) {
	    k16_c1024x(pIn, pKernel, pOut,
	       inChannel, inWidth, inHeight,
	       outChannel, outWidth, outHeight,
	       inChannelGroup, outChannelGroup,
	       inGroupOffset, outGroupOffset,
	       kernGroupOffset, oPixels, n, k,
	       filter) ;
	  }
	  else
	  {
	    k16(pIn, pKernel, pOut,
	       inChannel, inWidth, inHeight,
	       outChannel, outWidth, outHeight,
	       inChannelGroup, outChannelGroup,
	       inGroupOffset, outGroupOffset,
	       kernGroupOffset, oPixels, n, k) ;
	  }
	} // outChannel
      } // group
    } // batch
  }

  return VEDNN_SUCCESS;
}
