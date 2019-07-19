#include <stdint.h>
#include <stdlib.h>

#include "vednn.h"

#include "veintrin.h"
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
    _ve_lvl(VLEN) ;
    __vr vrzero = _ve_vbrdu_vs_f32(0.f) ;
    for(int64_t ij=0; ij<nPixs ; ij+=VLEN) {
      const int64_t vl = nPixs - ij < VLEN ? nPixs - ij : VLEN ;
      _ve_lvl(vl) ;
      _ve_vstu_vss(vrzero, 4, (void*)(pInChannel+ij)) ;
    }
  }

  for (int64_t y=0; y<gOutHeight; y+=nY)
  {
    const int64_t vl = gOutWidth * (gOutHeight - y < nY ? gOutHeight - y : nY) ;
    const int64_t op = y * gOutWidth ;

    _ve_lvl(vl) ;

    __vr vrsum = _ve_vbrdu_vs_f32(0.f) ;

    for(int64_t k=0; k<gOutChannelGroup; k++) {
      int64_t outIndex = gOutGroupOffset + (n * gOutChannel + k  ) *  gOutHeight * gOutWidth + op ;

      const float *pKerValue_k0 = pKernel + kernGroupOffset + ((k  ) * gInChannelGroup + c) ;

      __vr vrgout_k0 = _ve_vldu_vss(4, pGOut+outIndex) ;

      vrsum = _ve_vfmads_vvsv(vrsum, pKerValue_k0[0], vrgout_k0) ;

    }

    __vr vrpgin_c0 = _ve_vsfa_vvss(vrij, 2, (uint64_t)(pInChannel+y*strideHeight*gInWidth)) ;

    _ve_svob() ;

    _ve_vscuot_vv(vrsum, vrpgin_c0) ;
  }

  _ve_svob() ;
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
    _ve_lvl(VLEN) ;
    __vr vrzero = _ve_vbrdu_vs_f32(0.f) ;
    for(int64_t ij=0; ij<nPixs ; ij+=VLEN) {
      const int64_t vl = nPixs - ij < VLEN ? nPixs - ij : VLEN ;
      _ve_lvl(vl) ;
      _ve_vstu_vss(vrzero, 4, (void*)(pInChannel+ij)) ;
    }
  }

  for (int64_t y=0; y<gOutHeight; y+=nY)
  {
    const int64_t vl = gOutWidth * (gOutHeight - y < nY ? gOutHeight - y : nY) ;
    const int64_t op = y * gOutWidth ;

    _ve_lvl(vl) ;

    __vr vrsum01 = _ve_pvbrd_vs_i64(0UL) ;

    for(int64_t k=0; k<gOutChannelGroup; k++) {
      int64_t outIndex = gOutGroupOffset + (n * gOutChannel + k  ) *  gOutHeight * gOutWidth + op ;

      const float *pKerValue_k0 = pKernel + kernGroupOffset + ((k  ) * gInChannelGroup + c) ;

      __vr vrgout_k0 = _ve_vldu_vss(4, pGOut+outIndex) ;

      const uint64_t kerValue01_k0 = _ve_pack_f32p(pKerValue_k0,
						   pKerValue_k0 + 1) ;
      __vr vrgoutP_k0 = _ve_vshf_vvvs(vrgout_k0, vrgout_k0, VE_VSHUFFLE_YUZU) ;
      vrsum01 = _ve_pvfmad_vvsv(vrsum01, kerValue01_k0, vrgoutP_k0) ;

    }

    __vr vrpgin_c0 = _ve_vsfa_vvss(vrij, 2, (uint64_t)(pInChannel+y*strideHeight*gInWidth)) ;
    __vr vrpgin_c1 = _ve_vaddul_vsv(  4*gInHeight*gInWidth,vrpgin_c0) ;

    _ve_svob() ;

    _ve_vscuot_vv(vrsum01, vrpgin_c0) ;
    _ve_vsclot_vv(vrsum01, vrpgin_c1) ;
  }

  _ve_svob() ;
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
    _ve_lvl(VLEN) ;
    __vr vrzero = _ve_vbrdu_vs_f32(0.f) ;
    for(int64_t ij=0; ij<nPixs ; ij+=VLEN) {
      const int64_t vl = nPixs - ij < VLEN ? nPixs - ij : VLEN ;
      _ve_lvl(vl) ;
      _ve_vstu_vss(vrzero, 4, (void*)(pInChannel+ij)) ;
    }
  }

  for (int64_t y=0; y<gOutHeight; y+=nY)
  {
    const int64_t vl = gOutWidth * (gOutHeight - y < nY ? gOutHeight - y : nY) ;
    const int64_t op = y * gOutWidth ;

    _ve_lvl(vl) ;

    __vr vrsum01 = _ve_pvbrd_vs_i64(0UL) ;
    __vr vrsum23 = _ve_pvbrd_vs_i64(0UL) ;

    for(int64_t k=0; k<gOutChannelGroup; k++) {
      int64_t outIndex = gOutGroupOffset + (n * gOutChannel + k  ) *  gOutHeight * gOutWidth + op ;

      const float *pKerValue_k0 = pKernel + kernGroupOffset + ((k  ) * gInChannelGroup + c) ;

      __vr vrgout_k0 = _ve_vldu_vss(4, pGOut+outIndex) ;

      const uint64_t kerValue01_k0 = _ve_pack_f32p(pKerValue_k0,
						   pKerValue_k0 + 1) ;
      const uint64_t kerValue23_k0 = _ve_pack_f32p(pKerValue_k0 + 2,
						   pKerValue_k0 + 3) ;
      __vr vrgoutP_k0 = _ve_vshf_vvvs(vrgout_k0, vrgout_k0, VE_VSHUFFLE_YUZU) ;
      vrsum01 = _ve_pvfmad_vvsv(vrsum01, kerValue01_k0, vrgoutP_k0) ;
      vrsum23 = _ve_pvfmad_vvsv(vrsum23, kerValue23_k0, vrgoutP_k0) ;
    }

    __vr vrpgin_c0 = _ve_vsfa_vvss(vrij, 2, (uint64_t)(pInChannel+y*strideHeight*gInWidth)) ;
    __vr vrpgin_c1 = _ve_vaddul_vsv(  4*gInHeight*gInWidth,vrpgin_c0) ;
    __vr vrpgin_c2 = _ve_vaddul_vsv(2*4*gInHeight*gInWidth,vrpgin_c0) ;
    __vr vrpgin_c3 = _ve_vaddul_vsv(3*4*gInHeight*gInWidth,vrpgin_c0) ;

    _ve_svob() ;

    _ve_vscuot_vv(vrsum01, vrpgin_c0) ;
    _ve_vsclot_vv(vrsum01, vrpgin_c1) ;
    _ve_vscuot_vv(vrsum23, vrpgin_c2) ;
    _ve_vsclot_vv(vrsum23, vrpgin_c3) ;
  }

  _ve_svob() ;
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
    _ve_lvl(VLEN) ;
    __vr vrzero = _ve_vbrdu_vs_f32(0.f) ;
    for(int64_t ij=0; ij<nPixs ; ij+=VLEN) {
      const int64_t vl = nPixs - ij < VLEN ? nPixs - ij : VLEN ;
      _ve_lvl(vl) ;
      _ve_vstu_vss(vrzero, 4, (void*)(pInChannel+ij)) ;
    }
  }

  for (int64_t y=0; y<gOutHeight; y+=nY)
  {
    const int64_t vl = gOutWidth * (gOutHeight - y < nY ? gOutHeight - y : nY) ;
    const int64_t op = y * gOutWidth ;

    _ve_lvl(vl) ;

    __vr vrsum01 = _ve_pvbrd_vs_i64(0UL) ;
    __vr vrsum23 = _ve_pvbrd_vs_i64(0UL) ;
    __vr vrsum45 = _ve_pvbrd_vs_i64(0UL) ;
    __vr vrsum67 = _ve_pvbrd_vs_i64(0UL) ;

    for(int64_t k=0; k<gOutChannelGroup; k++) {
      int64_t outIndex = gOutGroupOffset + (n * gOutChannel + k  ) *  gOutHeight * gOutWidth + op ;

      const float *pKerValue_k0 = pKernel + kernGroupOffset + ((k  ) * gInChannelGroup + c) ;

      __vr vrgout_k0 = _ve_vldu_vss(4, pGOut+outIndex) ;

      const uint64_t kerValue01_k0 = _ve_pack_f32p(pKerValue_k0,
						   pKerValue_k0 + 1) ;
      const uint64_t kerValue23_k0 = _ve_pack_f32p(pKerValue_k0 + 2,
						   pKerValue_k0 + 3) ;
      const uint64_t kerValue45_k0 = _ve_pack_f32p(pKerValue_k0 + 4,
						   pKerValue_k0 + 5) ;
      const uint64_t kerValue67_k0 = _ve_pack_f32p(pKerValue_k0 + 6,
						   pKerValue_k0 + 7) ;
      __vr vrgoutP_k0 = _ve_vshf_vvvs(vrgout_k0, vrgout_k0, VE_VSHUFFLE_YUZU) ;
      vrsum01 = _ve_pvfmad_vvsv(vrsum01, kerValue01_k0, vrgoutP_k0) ;
      vrsum23 = _ve_pvfmad_vvsv(vrsum23, kerValue23_k0, vrgoutP_k0) ;
      vrsum45 = _ve_pvfmad_vvsv(vrsum45, kerValue45_k0, vrgoutP_k0) ;
      vrsum67 = _ve_pvfmad_vvsv(vrsum67, kerValue67_k0, vrgoutP_k0) ;
    }

    __vr vrpgin_c0 = _ve_vsfa_vvss(vrij, 2, (uint64_t)(pInChannel+y*strideHeight*gInWidth)) ;
    __vr vrpgin_c1 = _ve_vaddul_vsv(  4*gInHeight*gInWidth,vrpgin_c0) ;
    __vr vrpgin_c2 = _ve_vaddul_vsv(2*4*gInHeight*gInWidth,vrpgin_c0) ;
    __vr vrpgin_c3 = _ve_vaddul_vsv(3*4*gInHeight*gInWidth,vrpgin_c0) ;
    __vr vrpgin_c4 = _ve_vaddul_vsv(4*4*gInHeight*gInWidth,vrpgin_c0) ;
    __vr vrpgin_c5 = _ve_vaddul_vsv(5*4*gInHeight*gInWidth,vrpgin_c0) ;
    __vr vrpgin_c6 = _ve_vaddul_vsv(6*4*gInHeight*gInWidth,vrpgin_c0) ;
    __vr vrpgin_c7 = _ve_vaddul_vsv(7*4*gInHeight*gInWidth,vrpgin_c0) ;

    _ve_svob() ;

    _ve_vscuot_vv(vrsum01, vrpgin_c0) ;
    _ve_vsclot_vv(vrsum01, vrpgin_c1) ;
    _ve_vscuot_vv(vrsum23, vrpgin_c2) ;
    _ve_vsclot_vv(vrsum23, vrpgin_c3) ;
    _ve_vscuot_vv(vrsum45, vrpgin_c4) ;
    _ve_vsclot_vv(vrsum45, vrpgin_c5) ;
    _ve_vscuot_vv(vrsum67, vrpgin_c6) ;
    _ve_vsclot_vv(vrsum67, vrpgin_c7) ;
  }

  _ve_svob() ;
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
    _ve_lvl(VLEN) ;
    __vr vrzero = _ve_vbrdu_vs_f32(0.f) ;
    for(int64_t ij=0; ij<nPixs ; ij+=VLEN) {
      const int64_t vl = nPixs - ij < VLEN ? nPixs - ij : VLEN ;
      _ve_lvl(vl) ;
      _ve_vstu_vss(vrzero, 4, (void*)(pInChannel+ij)) ;
    }
  }

  for (int64_t y=0; y<gOutHeight; y+=nY)
  {
    const int64_t vl = gOutWidth * (gOutHeight - y < nY ? gOutHeight - y : nY) ;
    const int64_t op = y * gOutWidth ;

    _ve_lvl(vl) ;

    __vr vrsum01 = _ve_pvbrd_vs_i64(0UL) ;
    __vr vrsum23 = _ve_pvbrd_vs_i64(0UL) ;
    __vr vrsum45 = _ve_pvbrd_vs_i64(0UL) ;
    __vr vrsum67 = _ve_pvbrd_vs_i64(0UL) ;
    __vr vrsum89 = _ve_pvbrd_vs_i64(0UL) ;
    __vr vrsumAB = _ve_pvbrd_vs_i64(0UL) ;
    __vr vrsumCD = _ve_pvbrd_vs_i64(0UL) ;
    __vr vrsumEF = _ve_pvbrd_vs_i64(0UL) ;


    for(int64_t k=0; k<gOutChannelGroup; k++) {
      int64_t outIndex = gOutGroupOffset + (n * gOutChannel + k  ) *  gOutHeight * gOutWidth + op ;

      const float *pKerValue_k0 = pKernel + kernGroupOffset + ((k  ) * gInChannelGroup + c) ;

      __vr vrgout_k0 = _ve_vldu_vss(4, pGOut+outIndex) ;

      const uint64_t kerValue01_k0 = _ve_pack_f32p(pKerValue_k0,
						   pKerValue_k0 + 1) ;
      const uint64_t kerValue23_k0 = _ve_pack_f32p(pKerValue_k0 + 2,
						   pKerValue_k0 + 3) ;
      const uint64_t kerValue45_k0 = _ve_pack_f32p(pKerValue_k0 + 4,
						   pKerValue_k0 + 5) ;
      const uint64_t kerValue67_k0 = _ve_pack_f32p(pKerValue_k0 + 6,
						   pKerValue_k0 + 7) ;
      const uint64_t kerValue89_k0 = _ve_pack_f32p(pKerValue_k0 + 8,
						   pKerValue_k0 + 9) ;
      const uint64_t kerValueAB_k0 = _ve_pack_f32p(pKerValue_k0 +10,
						   pKerValue_k0 +11) ;
      const uint64_t kerValueCD_k0 = _ve_pack_f32p(pKerValue_k0 +12,
						   pKerValue_k0 +13) ;
      const uint64_t kerValueEF_k0 = _ve_pack_f32p(pKerValue_k0 +14,
						   pKerValue_k0 +15) ;
      __vr vrgoutP_k0 = _ve_vshf_vvvs(vrgout_k0, vrgout_k0, VE_VSHUFFLE_YUZU) ;
      vrsum01 = _ve_pvfmad_vvsv(vrsum01, kerValue01_k0, vrgoutP_k0) ;
      vrsum23 = _ve_pvfmad_vvsv(vrsum23, kerValue23_k0, vrgoutP_k0) ;
      vrsum45 = _ve_pvfmad_vvsv(vrsum45, kerValue45_k0, vrgoutP_k0) ;
      vrsum67 = _ve_pvfmad_vvsv(vrsum67, kerValue67_k0, vrgoutP_k0) ;
      vrsum89 = _ve_pvfmad_vvsv(vrsum89, kerValue89_k0, vrgoutP_k0) ;
      vrsumAB = _ve_pvfmad_vvsv(vrsumAB, kerValueAB_k0, vrgoutP_k0) ;
      vrsumCD = _ve_pvfmad_vvsv(vrsumCD, kerValueCD_k0, vrgoutP_k0) ;
      vrsumEF = _ve_pvfmad_vvsv(vrsumEF, kerValueEF_k0, vrgoutP_k0) ;
    }

    __vr vrpgin_c0 = _ve_vsfa_vvss(vrij, 2, (uint64_t)(pInChannel+y*strideHeight*gInWidth)) ;
    __vr vrpgin_c1 = _ve_vaddul_vsv(  4*gInHeight*gInWidth,vrpgin_c0) ;
    __vr vrpgin_c2 = _ve_vaddul_vsv(2*4*gInHeight*gInWidth,vrpgin_c0) ;
    __vr vrpgin_c3 = _ve_vaddul_vsv(3*4*gInHeight*gInWidth,vrpgin_c0) ;
    __vr vrpgin_c4 = _ve_vaddul_vsv(4*4*gInHeight*gInWidth,vrpgin_c0) ;
    __vr vrpgin_c5 = _ve_vaddul_vsv(5*4*gInHeight*gInWidth,vrpgin_c0) ;
    __vr vrpgin_c6 = _ve_vaddul_vsv(6*4*gInHeight*gInWidth,vrpgin_c0) ;
    __vr vrpgin_c7 = _ve_vaddul_vsv(7*4*gInHeight*gInWidth,vrpgin_c0) ;
    __vr vrpgin_c8 = _ve_vaddul_vsv(8*4*gInHeight*gInWidth,vrpgin_c0) ;
    __vr vrpgin_c9 = _ve_vaddul_vsv(9*4*gInHeight*gInWidth,vrpgin_c0) ;
    __vr vrpgin_cA = _ve_vaddul_vsv(10*4*gInHeight*gInWidth,vrpgin_c0) ;
    __vr vrpgin_cB = _ve_vaddul_vsv(11*4*gInHeight*gInWidth,vrpgin_c0) ;
    __vr vrpgin_cC = _ve_vaddul_vsv(12*4*gInHeight*gInWidth,vrpgin_c0) ;
    __vr vrpgin_cD = _ve_vaddul_vsv(13*4*gInHeight*gInWidth,vrpgin_c0) ;
    __vr vrpgin_cE = _ve_vaddul_vsv(14*4*gInHeight*gInWidth,vrpgin_c0) ;
    __vr vrpgin_cF = _ve_vaddul_vsv(15*4*gInHeight*gInWidth,vrpgin_c0) ;

    _ve_svob() ;

    _ve_vscuot_vv(vrsum01, vrpgin_c0) ;
    _ve_vsclot_vv(vrsum01, vrpgin_c1) ;
    _ve_vscuot_vv(vrsum23, vrpgin_c2) ;
    _ve_vsclot_vv(vrsum23, vrpgin_c3) ;
    _ve_vscuot_vv(vrsum45, vrpgin_c4) ;
    _ve_vsclot_vv(vrsum45, vrpgin_c5) ;
    _ve_vscuot_vv(vrsum67, vrpgin_c6) ;
    _ve_vsclot_vv(vrsum67, vrpgin_c7) ;
    _ve_vscuot_vv(vrsum89, vrpgin_c8) ;
    _ve_vsclot_vv(vrsum89, vrpgin_c9) ;
    _ve_vscuot_vv(vrsumAB, vrpgin_cA) ;
    _ve_vsclot_vv(vrsumAB, vrpgin_cB) ;
    _ve_vscuot_vv(vrsumCD, vrpgin_cC) ;
    _ve_vsclot_vv(vrsumCD, vrpgin_cD) ;
    _ve_vscuot_vv(vrsumEF, vrpgin_cE) ;
    _ve_vsclot_vv(vrsumEF, vrpgin_cF) ;
  }

  _ve_svob() ;
}


vednnError_t
vednnConvolutionBackwardData_direct_dil1_pad0_ker1_iwU128(
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
  const int64_t dilationWidth  = pParamConv->dilationWidth;
  const int64_t dilationHeight = pParamConv->dilationHeight;

  const int64_t gOutChannelGroup = gOutChannel  / group;
  const int64_t gInChannelGroup  = gInChannel / group;

  const float * restrict pGOut   = pDataGradOut;
  const float * restrict pKernel = pDataKernel;
  float * restrict const pGIn   = pDataGradIn;

  {
    const int64_t nY = VLEN / gOutWidth ;
    _ve_lvl(nY*gOutWidth) ;

    __vr vrseq = _ve_vseq_v() ;
    __vr vry  = _ve_vdivsl_vvs(vrseq, gOutWidth) ;
    __vr vrx  = _ve_vsubsl_vvv(vrseq, _ve_vmulul_vsv(gOutWidth,vry)) ;

    __vr vri   = _ve_vmulsl_vsv(strideHeight, vry) ;
    __vr vrj   = _ve_vmulsl_vsv(strideWidth,  vrx) ;
    __vr vrij = _ve_vaddul_vvv(vrj, _ve_vmulul_vsv(gInWidth, vri)) ;

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
	  c16(pGOut, pKernel, pGIn,
	     gOutChannel, gOutWidth, gOutHeight,
             gInChannel, gInWidth, gInHeight,
             strideWidth, strideHeight,
	     gInChannelGroup, gOutChannelGroup,
	     gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	     n, c,
	     nY, vrij) ;

	} // gInChannel
      } // group
    } // batch
  }


  return VEDNN_SUCCESS;
}
