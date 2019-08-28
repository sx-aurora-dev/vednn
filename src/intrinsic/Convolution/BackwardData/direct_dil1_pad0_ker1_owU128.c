#include <stdint.h>
#include <stdlib.h>

#include "vednn.h"

#include "velintrin.h"
#define VLEN	(256)

static inline void c1(
    const float * restrict pGOut,
    const float * restrict pKernel,
    float * restrict const pGIn,
    const int64_t gOutChannel,
    const int64_t gOutWidth,
    const int64_t gOutHeight,
    const int64_t gInChannel,
    const int64_t gInWidth,
    const int64_t gInHeight,
    const int64_t strideWidth,
    const int64_t strideHeight,
    const int64_t gInChannelGroup,
    const int64_t gOutChannelGroup,
    const int64_t gInGroupOffset,
    const int64_t gOutGroupOffset,
    const int64_t kernGroupOffset,
    const int64_t n,
    const int64_t c,
    const int64_t nY,
    const __vr vrij
)
{
  const float *pInChannel = pGIn + gInGroupOffset + ((n * gInChannel + c) * gInHeight * gInWidth ) ;
  {
    // zero filling
    const int64_t nPixs = 1 * gInHeight * gInWidth ;
    __vr vrzero = _vel_vbrds_vsl(0.f, VLEN) ;
    for(int64_t ij=0; ij<nPixs ; ij+=VLEN) {
      const int64_t vl = nPixs - ij < VLEN ? nPixs - ij : VLEN ;
      _vel_vstu_vssl(vrzero, 4, (void*)(pInChannel+ij), vl) ;
    }
  }

  for (int64_t y=0; y<gOutHeight; y+=nY)
  {
    const int64_t vl = gOutWidth * (gOutHeight - y < nY ? gOutHeight - y : nY) ;
    const int64_t op = y * gOutWidth ;

    __vr vrsum = _vel_vbrds_vsl(0.f, vl) ;

    for(int64_t k=0; k<gOutChannelGroup; k++) {
      int64_t outIndex = gOutGroupOffset + (n * gOutChannel + k  ) *  gOutHeight * gOutWidth + op ;

      const float *pKerValue_k0 = pKernel + kernGroupOffset + ((k  ) * gInChannelGroup + c) ;

      __vr vrgout_k0 = _vel_vldu_vssl(4, pGOut+outIndex, vl) ;

      vrsum = _vel_vfmads_vvsvl(vrsum, pKerValue_k0[0], vrgout_k0, vl) ;

    }

    __vr vrpgin_c0 = _vel_vsfa_vvssl(vrij, 2, (uint64_t)(pInChannel+y*strideHeight*gInWidth), vl) ;

    _vel_svob() ;

    _vel_vscuot_vvssl(vrsum, vrpgin_c0, 0, 0, vl) ;
  }

  _vel_svob() ;
}


static inline void c2(
    const float * restrict pGOut,
    const float * restrict pKernel,
    float * restrict const pGIn,
    const int64_t gOutChannel,
    const int64_t gOutWidth,
    const int64_t gOutHeight,
    const int64_t gInChannel,
    const int64_t gInWidth,
    const int64_t gInHeight,
    const int64_t strideWidth,
    const int64_t strideHeight,
    const int64_t gInChannelGroup,
    const int64_t gOutChannelGroup,
    const int64_t gInGroupOffset,
    const int64_t gOutGroupOffset,
    const int64_t kernGroupOffset,
    const int64_t n,
    const int64_t c,
    const int64_t nY,
    const __vr vrij
)
{
  const float *pInChannel = pGIn + gInGroupOffset + ((n * gInChannel + c) * gInHeight * gInWidth ) ;
  {
    // zero filling
    const int64_t nPixs = 2 * gInHeight * gInWidth ;
    __vr vrzero = _vel_vbrds_vsl(0.f, VLEN) ;
    for(int64_t ij=0; ij<nPixs ; ij+=VLEN) {
      const int64_t vl = nPixs - ij < VLEN ? nPixs - ij : VLEN ;
      _vel_vstu_vssl(vrzero, 4, (void*)(pInChannel+ij), vl) ;
    }
  }

  for (int64_t y=0; y<gOutHeight; y+=nY)
  {
    const int64_t vl = gOutWidth * (gOutHeight - y < nY ? gOutHeight - y : nY) ;
    const int64_t op = y * gOutWidth ;

    __vr vrsum01 = _vel_pvbrd_vsl(0UL, vl) ;

    for(int64_t k=0; k<gOutChannelGroup; k++) {
      int64_t outIndex = gOutGroupOffset + (n * gOutChannel + k  ) *  gOutHeight * gOutWidth + op ;

      const float *pKerValue_k0 = pKernel + kernGroupOffset + ((k  ) * gInChannelGroup + c) ;

      __vr vrgout_k0 = _vel_vldu_vssl(4, pGOut+outIndex, vl) ;

      const uint64_t kerValue01_k0 = _vel_pack_f32p(pKerValue_k0,
						    pKerValue_k0 + 1) ;
      __vr vrgoutP_k0 = _vel_vshf_vvvsl(vrgout_k0, vrgout_k0, VE_VSHUFFLE_YUZU, vl) ;
      vrsum01 = _vel_pvfmad_vvsvl(vrsum01, kerValue01_k0, vrgoutP_k0, vl) ;

    }

    __vr vrpgin_c0 = _vel_vsfa_vvssl(vrij, 2, (uint64_t)(pInChannel+y*strideHeight*gInWidth), vl) ;
    __vr vrpgin_c1 = _vel_vaddul_vsvl(  4*gInHeight*gInWidth,vrpgin_c0, vl) ;

    _vel_svob() ;

    _vel_vscuot_vvssl(vrsum01, vrpgin_c0, 0, 0, vl) ;
    _vel_vsclot_vvssl(vrsum01, vrpgin_c1, 0, 0, vl) ;
  }

  _vel_svob() ;
}


static inline void c4(
    const float * restrict pGOut,
    const float * restrict pKernel,
    float * restrict const pGIn,
    const int64_t gOutChannel,
    const int64_t gOutWidth,
    const int64_t gOutHeight,
    const int64_t gInChannel,
    const int64_t gInWidth,
    const int64_t gInHeight,
    const int64_t strideWidth,
    const int64_t strideHeight,
    const int64_t gInChannelGroup,
    const int64_t gOutChannelGroup,
    const int64_t gInGroupOffset,
    const int64_t gOutGroupOffset,
    const int64_t kernGroupOffset,
    const int64_t n,
    const int64_t c,
    const int64_t nY,
    const __vr vrij
)
{
  const float *pInChannel = pGIn + gInGroupOffset + ((n * gInChannel + c) * gInHeight * gInWidth ) ;
  {
    // zero filling
    const int64_t nPixs = 4 * gInHeight * gInWidth ;
    __vr vrzero = _vel_vbrds_vsl(0.f, VLEN) ;
    for(int64_t ij=0; ij<nPixs ; ij+=VLEN) {
      const int64_t vl = nPixs - ij < VLEN ? nPixs - ij : VLEN ;
      _vel_vstu_vssl(vrzero, 4, (void*)(pInChannel+ij), vl) ;
    }
  }

  for (int64_t y=0; y<gOutHeight; y+=nY)
  {
    const int64_t vl = gOutWidth * (gOutHeight - y < nY ? gOutHeight - y : nY) ;
    const int64_t op = y * gOutWidth ;

    __vr vrsum01 = _vel_pvbrd_vsl(0UL, vl) ;
    __vr vrsum23 = _vel_pvbrd_vsl(0UL, vl) ;

    for(int64_t k=0; k<gOutChannelGroup; k++) {
      int64_t outIndex = gOutGroupOffset + (n * gOutChannel + k  ) *  gOutHeight * gOutWidth + op ;

      const float *pKerValue_k0 = pKernel + kernGroupOffset + ((k  ) * gInChannelGroup + c) ;

      __vr vrgout_k0 = _vel_vldu_vssl(4, pGOut+outIndex, vl) ;

      const uint64_t kerValue01_k0 = _vel_pack_f32p(pKerValue_k0,
						    pKerValue_k0 + 1) ;
      const uint64_t kerValue23_k0 = _vel_pack_f32p(pKerValue_k0 + 2,
						    pKerValue_k0 + 3) ;
      __vr vrgoutP_k0 = _vel_vshf_vvvsl(vrgout_k0, vrgout_k0, VE_VSHUFFLE_YUZU, vl) ;
      vrsum01 = _vel_pvfmad_vvsvl(vrsum01, kerValue01_k0, vrgoutP_k0, vl) ;
      vrsum23 = _vel_pvfmad_vvsvl(vrsum23, kerValue23_k0, vrgoutP_k0, vl) ;
    }

    __vr vrpgin_c0 = _vel_vsfa_vvssl(vrij, 2, (uint64_t)(pInChannel+y*strideHeight*gInWidth), vl) ;
    __vr vrpgin_c1 = _vel_vaddul_vsvl(  4*gInHeight*gInWidth,vrpgin_c0, vl) ;
    __vr vrpgin_c2 = _vel_vaddul_vsvl(2*4*gInHeight*gInWidth,vrpgin_c0, vl) ;
    __vr vrpgin_c3 = _vel_vaddul_vsvl(3*4*gInHeight*gInWidth,vrpgin_c0, vl) ;

    _vel_svob() ;

    _vel_vscuot_vvssl(vrsum01, vrpgin_c0, 0, 0, vl) ;
    _vel_vsclot_vvssl(vrsum01, vrpgin_c1, 0, 0, vl) ;
    _vel_vscuot_vvssl(vrsum23, vrpgin_c2, 0, 0, vl) ;
    _vel_vsclot_vvssl(vrsum23, vrpgin_c3, 0, 0, vl) ;
  }

  _vel_svob() ;
}


static inline void c8(
    const float * restrict pGOut,
    const float * restrict pKernel,
    float * restrict const pGIn,
    const int64_t gOutChannel,
    const int64_t gOutWidth,
    const int64_t gOutHeight,
    const int64_t gInChannel,
    const int64_t gInWidth,
    const int64_t gInHeight,
    const int64_t strideWidth,
    const int64_t strideHeight,
    const int64_t gInChannelGroup,
    const int64_t gOutChannelGroup,
    const int64_t gInGroupOffset,
    const int64_t gOutGroupOffset,
    const int64_t kernGroupOffset,
    const int64_t n,
    const int64_t c,
    const int64_t nY,
    const __vr vrij
)
{
  const float *pInChannel = pGIn + gInGroupOffset + ((n * gInChannel + c) * gInHeight * gInWidth ) ;
  {
    // zero filling
    const int64_t nPixs = 8 * gInHeight * gInWidth ;
    __vr vrzero = _vel_vbrds_vsl(0.f, VLEN) ;
    for(int64_t ij=0; ij<nPixs ; ij+=VLEN) {
      const int64_t vl = nPixs - ij < VLEN ? nPixs - ij : VLEN ;
      _vel_vstu_vssl(vrzero, 4, (void*)(pInChannel+ij), vl) ;
    }
  }

  for (int64_t y=0; y<gOutHeight; y+=nY)
  {
    const int64_t vl = gOutWidth * (gOutHeight - y < nY ? gOutHeight - y : nY) ;
    const int64_t op = y * gOutWidth ;

    __vr vrsum01 = _vel_pvbrd_vsl(0UL, vl) ;
    __vr vrsum23 = _vel_pvbrd_vsl(0UL, vl) ;
    __vr vrsum45 = _vel_pvbrd_vsl(0UL, vl) ;
    __vr vrsum67 = _vel_pvbrd_vsl(0UL, vl) ;

    for(int64_t k=0; k<gOutChannelGroup; k++) {
      int64_t outIndex = gOutGroupOffset + (n * gOutChannel + k  ) *  gOutHeight * gOutWidth + op ;

      const float *pKerValue_k0 = pKernel + kernGroupOffset + ((k  ) * gInChannelGroup + c) ;

      __vr vrgout_k0 = _vel_vldu_vssl(4, pGOut+outIndex, vl) ;

      const uint64_t kerValue01_k0 = _vel_pack_f32p(pKerValue_k0,
						    pKerValue_k0 + 1) ;
      const uint64_t kerValue23_k0 = _vel_pack_f32p(pKerValue_k0 + 2,
						    pKerValue_k0 + 3) ;
      const uint64_t kerValue45_k0 = _vel_pack_f32p(pKerValue_k0 + 4,
						    pKerValue_k0 + 5) ;
      const uint64_t kerValue67_k0 = _vel_pack_f32p(pKerValue_k0 + 6,
						    pKerValue_k0 + 7) ;
      __vr vrgoutP_k0 = _vel_vshf_vvvsl(vrgout_k0, vrgout_k0, VE_VSHUFFLE_YUZU, vl) ;
      vrsum01 = _vel_pvfmad_vvsvl(vrsum01, kerValue01_k0, vrgoutP_k0, vl) ;
      vrsum23 = _vel_pvfmad_vvsvl(vrsum23, kerValue23_k0, vrgoutP_k0, vl) ;
      vrsum45 = _vel_pvfmad_vvsvl(vrsum45, kerValue45_k0, vrgoutP_k0, vl) ;
      vrsum67 = _vel_pvfmad_vvsvl(vrsum67, kerValue67_k0, vrgoutP_k0, vl) ;
    }

    __vr vrpgin_c0 = _vel_vsfa_vvssl(vrij, 2, (uint64_t)(pInChannel+y*strideHeight*gInWidth), vl) ;
    __vr vrpgin_c1 = _vel_vaddul_vsvl(  4*gInHeight*gInWidth,vrpgin_c0, vl) ;
    __vr vrpgin_c2 = _vel_vaddul_vsvl(2*4*gInHeight*gInWidth,vrpgin_c0, vl) ;
    __vr vrpgin_c3 = _vel_vaddul_vsvl(3*4*gInHeight*gInWidth,vrpgin_c0, vl) ;
    __vr vrpgin_c4 = _vel_vaddul_vsvl(4*4*gInHeight*gInWidth,vrpgin_c0, vl) ;
    __vr vrpgin_c5 = _vel_vaddul_vsvl(5*4*gInHeight*gInWidth,vrpgin_c0, vl) ;
    __vr vrpgin_c6 = _vel_vaddul_vsvl(6*4*gInHeight*gInWidth,vrpgin_c0, vl) ;
    __vr vrpgin_c7 = _vel_vaddul_vsvl(7*4*gInHeight*gInWidth,vrpgin_c0, vl) ;

    _vel_svob() ;

    _vel_vscuot_vvssl(vrsum01, vrpgin_c0, 0, 0, vl) ;
    _vel_vsclot_vvssl(vrsum01, vrpgin_c1, 0, 0, vl) ;
    _vel_vscuot_vvssl(vrsum23, vrpgin_c2, 0, 0, vl) ;
    _vel_vsclot_vvssl(vrsum23, vrpgin_c3, 0, 0, vl) ;
    _vel_vscuot_vvssl(vrsum45, vrpgin_c4, 0, 0, vl) ;
    _vel_vsclot_vvssl(vrsum45, vrpgin_c5, 0, 0, vl) ;
    _vel_vscuot_vvssl(vrsum67, vrpgin_c6, 0, 0, vl) ;
    _vel_vsclot_vvssl(vrsum67, vrpgin_c7, 0, 0, vl) ;
  }

  _vel_svob() ;
}


static inline void c16(
    const float * restrict pGOut,
    const float * restrict pKernel,
    float * restrict const pGIn,
    const int64_t gOutChannel,
    const int64_t gOutWidth,
    const int64_t gOutHeight,
    const int64_t gInChannel,
    const int64_t gInWidth,
    const int64_t gInHeight,
    const int64_t strideWidth,
    const int64_t strideHeight,
    const int64_t gInChannelGroup,
    const int64_t gOutChannelGroup,
    const int64_t gInGroupOffset,
    const int64_t gOutGroupOffset,
    const int64_t kernGroupOffset,
    const int64_t n,
    const int64_t c,
    const int64_t nY,
    const __vr vrij
)
{
  const float *pInChannel = pGIn + gInGroupOffset + ((n * gInChannel + c) * gInHeight * gInWidth ) ;
  {
    // zero filling
    const int64_t nPixs = 16 * gInHeight * gInWidth ;
    __vr vrzero = _vel_vbrds_vsl(0.f, VLEN) ;
    for(int64_t ij=0; ij<nPixs ; ij+=VLEN) {
      const int64_t vl = nPixs - ij < VLEN ? nPixs - ij : VLEN ;
      _vel_vstu_vssl(vrzero, 4, (void*)(pInChannel+ij), vl) ;
    }
  }

  for (int64_t y=0; y<gOutHeight; y+=nY)
  {
    const int64_t vl = gOutWidth * (gOutHeight - y < nY ? gOutHeight - y : nY) ;
    const int64_t op = y * gOutWidth ;

    __vr vrsum01 = _vel_pvbrd_vsl(0UL, vl) ;
    __vr vrsum23 = _vel_pvbrd_vsl(0UL, vl) ;
    __vr vrsum45 = _vel_pvbrd_vsl(0UL, vl) ;
    __vr vrsum67 = _vel_pvbrd_vsl(0UL, vl) ;
    __vr vrsum89 = _vel_pvbrd_vsl(0UL, vl) ;
    __vr vrsumAB = _vel_pvbrd_vsl(0UL, vl) ;
    __vr vrsumCD = _vel_pvbrd_vsl(0UL, vl) ;
    __vr vrsumEF = _vel_pvbrd_vsl(0UL, vl) ;


    for(int64_t k=0; k<gOutChannelGroup; k++) {
      int64_t outIndex = gOutGroupOffset + (n * gOutChannel + k  ) *  gOutHeight * gOutWidth + op ;

      const float *pKerValue_k0 = pKernel + kernGroupOffset + ((k  ) * gInChannelGroup + c) ;

      __vr vrgout_k0 = _vel_vldu_vssl(4, pGOut+outIndex, vl) ;

      const uint64_t kerValue01_k0 = _vel_pack_f32p(pKerValue_k0,
						    pKerValue_k0 + 1) ;
      const uint64_t kerValue23_k0 = _vel_pack_f32p(pKerValue_k0 + 2,
						    pKerValue_k0 + 3) ;
      const uint64_t kerValue45_k0 = _vel_pack_f32p(pKerValue_k0 + 4,
						    pKerValue_k0 + 5) ;
      const uint64_t kerValue67_k0 = _vel_pack_f32p(pKerValue_k0 + 6,
						    pKerValue_k0 + 7) ;
      const uint64_t kerValue89_k0 = _vel_pack_f32p(pKerValue_k0 + 8,
						    pKerValue_k0 + 9) ;
      const uint64_t kerValueAB_k0 = _vel_pack_f32p(pKerValue_k0 +10,
						    pKerValue_k0 +11) ;
      const uint64_t kerValueCD_k0 = _vel_pack_f32p(pKerValue_k0 +12,
						    pKerValue_k0 +13) ;
      const uint64_t kerValueEF_k0 = _vel_pack_f32p(pKerValue_k0 +14,
						    pKerValue_k0 +15) ;
      __vr vrgoutP_k0 = _vel_vshf_vvvsl(vrgout_k0, vrgout_k0, VE_VSHUFFLE_YUZU, vl) ;
      vrsum01 = _vel_pvfmad_vvsvl(vrsum01, kerValue01_k0, vrgoutP_k0, vl) ;
      vrsum23 = _vel_pvfmad_vvsvl(vrsum23, kerValue23_k0, vrgoutP_k0, vl) ;
      vrsum45 = _vel_pvfmad_vvsvl(vrsum45, kerValue45_k0, vrgoutP_k0, vl) ;
      vrsum67 = _vel_pvfmad_vvsvl(vrsum67, kerValue67_k0, vrgoutP_k0, vl) ;
      vrsum89 = _vel_pvfmad_vvsvl(vrsum89, kerValue89_k0, vrgoutP_k0, vl) ;
      vrsumAB = _vel_pvfmad_vvsvl(vrsumAB, kerValueAB_k0, vrgoutP_k0, vl) ;
      vrsumCD = _vel_pvfmad_vvsvl(vrsumCD, kerValueCD_k0, vrgoutP_k0, vl) ;
      vrsumEF = _vel_pvfmad_vvsvl(vrsumEF, kerValueEF_k0, vrgoutP_k0, vl) ;
    }

    __vr vrpgin_c0 = _vel_vsfa_vvssl(vrij, 2, (uint64_t)(pInChannel+y*strideHeight*gInWidth), vl) ;
    __vr vrpgin_c1 = _vel_vaddul_vsvl(  4*gInHeight*gInWidth,vrpgin_c0, vl) ;
    __vr vrpgin_c2 = _vel_vaddul_vsvl(2*4*gInHeight*gInWidth,vrpgin_c0, vl) ;
    __vr vrpgin_c3 = _vel_vaddul_vsvl(3*4*gInHeight*gInWidth,vrpgin_c0, vl) ;
    __vr vrpgin_c4 = _vel_vaddul_vsvl(4*4*gInHeight*gInWidth,vrpgin_c0, vl) ;
    __vr vrpgin_c5 = _vel_vaddul_vsvl(5*4*gInHeight*gInWidth,vrpgin_c0, vl) ;
    __vr vrpgin_c6 = _vel_vaddul_vsvl(6*4*gInHeight*gInWidth,vrpgin_c0, vl) ;
    __vr vrpgin_c7 = _vel_vaddul_vsvl(7*4*gInHeight*gInWidth,vrpgin_c0, vl) ;
    __vr vrpgin_c8 = _vel_vaddul_vsvl(8*4*gInHeight*gInWidth,vrpgin_c0, vl) ;
    __vr vrpgin_c9 = _vel_vaddul_vsvl(9*4*gInHeight*gInWidth,vrpgin_c0, vl) ;
    __vr vrpgin_cA = _vel_vaddul_vsvl(10*4*gInHeight*gInWidth,vrpgin_c0, vl) ;
    __vr vrpgin_cB = _vel_vaddul_vsvl(11*4*gInHeight*gInWidth,vrpgin_c0, vl) ;
    __vr vrpgin_cC = _vel_vaddul_vsvl(12*4*gInHeight*gInWidth,vrpgin_c0, vl) ;
    __vr vrpgin_cD = _vel_vaddul_vsvl(13*4*gInHeight*gInWidth,vrpgin_c0, vl) ;
    __vr vrpgin_cE = _vel_vaddul_vsvl(14*4*gInHeight*gInWidth,vrpgin_c0, vl) ;
    __vr vrpgin_cF = _vel_vaddul_vsvl(15*4*gInHeight*gInWidth,vrpgin_c0, vl) ;

    _vel_svob() ;

    _vel_vscuot_vvssl(vrsum01, vrpgin_c0, 0, 0, vl) ;
    _vel_vsclot_vvssl(vrsum01, vrpgin_c1, 0, 0, vl) ;
    _vel_vscuot_vvssl(vrsum23, vrpgin_c2, 0, 0, vl) ;
    _vel_vsclot_vvssl(vrsum23, vrpgin_c3, 0, 0, vl) ;
    _vel_vscuot_vvssl(vrsum45, vrpgin_c4, 0, 0, vl) ;
    _vel_vsclot_vvssl(vrsum45, vrpgin_c5, 0, 0, vl) ;
    _vel_vscuot_vvssl(vrsum67, vrpgin_c6, 0, 0, vl) ;
    _vel_vsclot_vvssl(vrsum67, vrpgin_c7, 0, 0, vl) ;
    _vel_vscuot_vvssl(vrsum89, vrpgin_c8, 0, 0, vl) ;
    _vel_vsclot_vvssl(vrsum89, vrpgin_c9, 0, 0, vl) ;
    _vel_vscuot_vvssl(vrsumAB, vrpgin_cA, 0, 0, vl) ;
    _vel_vsclot_vvssl(vrsumAB, vrpgin_cB, 0, 0, vl) ;
    _vel_vscuot_vvssl(vrsumCD, vrpgin_cC, 0, 0, vl) ;
    _vel_vsclot_vvssl(vrsumCD, vrpgin_cD, 0, 0, vl) ;
    _vel_vscuot_vvssl(vrsumEF, vrpgin_cE, 0, 0, vl) ;
    _vel_vsclot_vvssl(vrsumEF, vrpgin_cF, 0, 0, vl) ;
  }

  _vel_svob() ;
}


static inline void c16p(
    const float * restrict pGOut,
    const float * restrict pKernel,
    float * restrict const pGIn,
    const int64_t gOutChannel,
    const int64_t gOutWidth,
    const int64_t gOutHeight,
    const int64_t gInChannel,
    const int64_t gInWidth,
    const int64_t gInHeight,
    const int64_t strideWidth,
    const int64_t strideHeight,
    const int64_t gInChannelGroup,
    const int64_t gOutChannelGroup,
    const int64_t gInGroupOffset,
    const int64_t gOutGroupOffset,
    const int64_t kernGroupOffset,
    const int64_t n,
    const int64_t c,
    const int64_t nY,
    const __vr vrij
)
{
  const float *pInChannel = pGIn + gInGroupOffset + ((n * gInChannel + c) * gInHeight * gInWidth ) ;
  {
    // zero filling
    const int64_t nPixs = 16 * gInHeight * gInWidth ;
    __vr vrzero = _vel_vbrds_vsl(0.f, VLEN) ;
    for(int64_t ij=0; ij<nPixs ; ij+=VLEN) {
      const int64_t vl = nPixs - ij < VLEN ? nPixs - ij : VLEN ;
      _vel_vstu_vssl(vrzero, 4, (void*)(pInChannel+ij), vl) ;
    }
  }

  for (int64_t y=0; y<gOutHeight; y+=nY)
  {
    const int64_t vl = gOutWidth * (gOutHeight - y < nY ? gOutHeight - y : nY) ;
    const int64_t op = y * gOutWidth ;

    __vr vrsum01 = _vel_pvbrd_vsl(0UL, vl) ;
    __vr vrsum23 = _vel_pvbrd_vsl(0UL, vl) ;
    __vr vrsum45 = _vel_pvbrd_vsl(0UL, vl) ;
    __vr vrsum67 = _vel_pvbrd_vsl(0UL, vl) ;
    __vr vrsum89 = _vel_pvbrd_vsl(0UL, vl) ;
    __vr vrsumAB = _vel_pvbrd_vsl(0UL, vl) ;
    __vr vrsumCD = _vel_pvbrd_vsl(0UL, vl) ;
    __vr vrsumEF = _vel_pvbrd_vsl(0UL, vl) ;


    for(int64_t k=0; k<gOutChannelGroup; k++) {
      int64_t outIndex = gOutGroupOffset + (n * gOutChannel + k  ) *  gOutHeight * gOutWidth + op ;

      const float *pKerValue_k0 = pKernel + kernGroupOffset + ((k  ) * gInChannelGroup + c) ;
      const uint64_t *pKerValue_u64_k0 = (const uint64_t*) pKerValue_k0 ;

      __vr vrgout_k0 = _vel_vldu_vssl(4, pGOut+outIndex, vl) ;

      __vr vrgoutP_k0 = _vel_vshf_vvvsl(vrgout_k0, vrgout_k0, VE_VSHUFFLE_YUZU, vl) ;
      vrsum01 = _vel_pvfmad_vvsvl(vrsum01, pKerValue_u64_k0[0], vrgoutP_k0, vl) ;
      vrsum23 = _vel_pvfmad_vvsvl(vrsum23, pKerValue_u64_k0[1], vrgoutP_k0, vl) ;
      vrsum45 = _vel_pvfmad_vvsvl(vrsum45, pKerValue_u64_k0[2], vrgoutP_k0, vl) ;
      vrsum67 = _vel_pvfmad_vvsvl(vrsum67, pKerValue_u64_k0[3], vrgoutP_k0, vl) ;
      vrsum89 = _vel_pvfmad_vvsvl(vrsum89, pKerValue_u64_k0[4], vrgoutP_k0, vl) ;
      vrsumAB = _vel_pvfmad_vvsvl(vrsumAB, pKerValue_u64_k0[5], vrgoutP_k0, vl) ;
      vrsumCD = _vel_pvfmad_vvsvl(vrsumCD, pKerValue_u64_k0[6], vrgoutP_k0, vl) ;
      vrsumEF = _vel_pvfmad_vvsvl(vrsumEF, pKerValue_u64_k0[7], vrgoutP_k0, vl) ;
    }

    __vr vrpgin_c0 = _vel_vsfa_vvssl(vrij, 2, (uint64_t)(pInChannel+y*strideHeight*gInWidth), vl) ;
    __vr vrpgin_c1 = _vel_vaddul_vsvl(  4*gInHeight*gInWidth,vrpgin_c0, vl) ;
    __vr vrpgin_c2 = _vel_vaddul_vsvl(2*4*gInHeight*gInWidth,vrpgin_c0, vl) ;
    __vr vrpgin_c3 = _vel_vaddul_vsvl(3*4*gInHeight*gInWidth,vrpgin_c0, vl) ;
    __vr vrpgin_c4 = _vel_vaddul_vsvl(4*4*gInHeight*gInWidth,vrpgin_c0, vl) ;
    __vr vrpgin_c5 = _vel_vaddul_vsvl(5*4*gInHeight*gInWidth,vrpgin_c0, vl) ;
    __vr vrpgin_c6 = _vel_vaddul_vsvl(6*4*gInHeight*gInWidth,vrpgin_c0, vl) ;
    __vr vrpgin_c7 = _vel_vaddul_vsvl(7*4*gInHeight*gInWidth,vrpgin_c0, vl) ;
    __vr vrpgin_c8 = _vel_vaddul_vsvl(8*4*gInHeight*gInWidth,vrpgin_c0, vl) ;
    __vr vrpgin_c9 = _vel_vaddul_vsvl(9*4*gInHeight*gInWidth,vrpgin_c0, vl) ;
    __vr vrpgin_cA = _vel_vaddul_vsvl(10*4*gInHeight*gInWidth,vrpgin_c0, vl) ;
    __vr vrpgin_cB = _vel_vaddul_vsvl(11*4*gInHeight*gInWidth,vrpgin_c0, vl) ;
    __vr vrpgin_cC = _vel_vaddul_vsvl(12*4*gInHeight*gInWidth,vrpgin_c0, vl) ;
    __vr vrpgin_cD = _vel_vaddul_vsvl(13*4*gInHeight*gInWidth,vrpgin_c0, vl) ;
    __vr vrpgin_cE = _vel_vaddul_vsvl(14*4*gInHeight*gInWidth,vrpgin_c0, vl) ;
    __vr vrpgin_cF = _vel_vaddul_vsvl(15*4*gInHeight*gInWidth,vrpgin_c0, vl) ;

    _vel_svob() ;

    _vel_vsclot_vvssl(vrsum01, vrpgin_c0, 0, 0, vl) ;
    _vel_vscuot_vvssl(vrsum01, vrpgin_c1, 0, 0, vl) ;
    _vel_vsclot_vvssl(vrsum23, vrpgin_c2, 0, 0, vl) ;
    _vel_vscuot_vvssl(vrsum23, vrpgin_c3, 0, 0, vl) ;
    _vel_vsclot_vvssl(vrsum45, vrpgin_c4, 0, 0, vl) ;
    _vel_vscuot_vvssl(vrsum45, vrpgin_c5, 0, 0, vl) ;
    _vel_vsclot_vvssl(vrsum67, vrpgin_c6, 0, 0, vl) ;
    _vel_vscuot_vvssl(vrsum67, vrpgin_c7, 0, 0, vl) ;
    _vel_vsclot_vvssl(vrsum89, vrpgin_c8, 0, 0, vl) ;
    _vel_vscuot_vvssl(vrsum89, vrpgin_c9, 0, 0, vl) ;
    _vel_vsclot_vvssl(vrsumAB, vrpgin_cA, 0, 0, vl) ;
    _vel_vscuot_vvssl(vrsumAB, vrpgin_cB, 0, 0, vl) ;
    _vel_vsclot_vvssl(vrsumCD, vrpgin_cC, 0, 0, vl) ;
    _vel_vscuot_vvssl(vrsumCD, vrpgin_cD, 0, 0, vl) ;
    _vel_vsclot_vvssl(vrsumEF, vrpgin_cE, 0, 0, vl) ;
    _vel_vscuot_vvssl(vrsumEF, vrpgin_cF, 0, 0, vl) ;
  }

  _vel_svob() ;
}



vednnError_t
vednnConvolutionBackwardData_direct_dil1_pad0_ker1_owU128(
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
  const int64_t strideWidth    = pParamConv->strideWidth;;
  const int64_t strideHeight   = pParamConv->strideHeight;
//  const int64_t padWidth       = pParamConv->padWidth;	// must be 0
//  const int64_t padHeight      = pParamConv->padHeight;	// must be 0
//  const int64_t dilationWidth  = pParamConv->dilationWidth;	// must be 1
//  const int64_t dilationHeight = pParamConv->dilationHeight;	// must be 1

  const int64_t gOutChannelGroup = gOutChannel  / group;
  const int64_t gInChannelGroup  = gInChannel / group;

  const float * restrict pGOut   = pDataGradOut;
  const float * restrict pKernel = pDataKernel;
  float * restrict const pGIn   = pDataGradIn;

  {
    const int64_t nY = VLEN / gOutWidth ;
     ;

    __vr vrseq = _vel_vseq_vl(nY*gOutWidth) ;
    __vr vry  = _vel_vdivsl_vvsl(vrseq, gOutWidth, nY*gOutWidth) ;
    __vr vrx  = _vel_vsubsl_vvvl(vrseq, _vel_vmulul_vsvl(gOutWidth,vry, nY*gOutWidth), nY*gOutWidth) ;

    __vr vri   = _vel_vmulsl_vsvl(strideHeight, vry, nY*gOutWidth) ;
    __vr vrj   = _vel_vmulsl_vsvl(strideWidth,  vrx, nY*gOutWidth) ;
    __vr vrij = _vel_vaddul_vvvl(vrj, _vel_vmulul_vsvl(gInWidth, vri, nY*gOutWidth), nY*gOutWidth) ;

    const int64_t usePackedKernel = (((uint64_t)pKernel) & 0x07) == 0 && (gInChannelGroup & 0x01) == 0 ?  1 : 0  ;

    for (int64_t n=0; n<batch; n++) {
      for (int64_t g = 0; g < group; g++) {

	int64_t gInGroupOffset  = g * gInChannelGroup * gInHeight * gInWidth;
	int64_t gOutGroupOffset = g * gOutChannelGroup * gOutHeight * gOutWidth;
	int64_t kernGroupOffset = g * gOutChannelGroup * gInChannelGroup * kernHeight * kernWidth;

	int64_t c=0;

	if( (gInChannelGroup & 0x01 ) == 1 ) {
	  c1(pGOut, pKernel, pGIn,
	     gOutChannel, gOutWidth, gOutHeight,
             gInChannel, gInWidth, gInHeight,
             strideWidth, strideHeight,
	     gInChannelGroup, gOutChannelGroup,
	     gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	     n, c,
	     nY, vrij) ;

	  c++ ;
	}
	if( ((gInChannelGroup>>1) & 0x01 ) == 1 ) {
	  c2(pGOut, pKernel, pGIn,
	     gOutChannel, gOutWidth, gOutHeight,
             gInChannel, gInWidth, gInHeight,
             strideWidth, strideHeight,
	     gInChannelGroup, gOutChannelGroup,
	     gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	     n, c,
	     nY, vrij) ;

	  c+=2 ;
	}
	if( ((gInChannelGroup>>2) & 0x01 ) == 1 ) {
	  c4(pGOut, pKernel, pGIn,
	     gOutChannel, gOutWidth, gOutHeight,
             gInChannel, gInWidth, gInHeight,
             strideWidth, strideHeight,
	     gInChannelGroup, gOutChannelGroup,
	     gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	     n, c,
	     nY, vrij) ;

	  c+=4 ;
	}
	if( ((gInChannelGroup>>3) & 0x01 ) == 1 ) {
	  c8(pGOut, pKernel, pGIn,
	     gOutChannel, gOutWidth, gOutHeight,
             gInChannel, gInWidth, gInHeight,
             strideWidth, strideHeight,
	     gInChannelGroup, gOutChannelGroup,
	     gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	     n, c,
	     nY, vrij) ;

	  c+=8 ;
	}
	for (; c<gInChannelGroup; c+=16) {
	  if( usePackedKernel ) {
	    c16p(pGOut, pKernel, pGIn,
	       gOutChannel, gOutWidth, gOutHeight,
	       gInChannel, gInWidth, gInHeight,
	       strideWidth, strideHeight,
	       gInChannelGroup, gOutChannelGroup,
	       gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	       n, c,
	       nY, vrij) ;
	  }
	  else {
	    c16(pGOut, pKernel, pGIn,
	       gOutChannel, gOutWidth, gOutHeight,
	       gInChannel, gInWidth, gInHeight,
	       strideWidth, strideHeight,
	       gInChannelGroup, gOutChannelGroup,
	       gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	       n, c,
	       nY, vrij) ;
	  }

	} // gInChannel
      } // group
    } // batch
  }


  return VEDNN_SUCCESS;
}
