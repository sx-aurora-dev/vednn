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
  const int64_t gInChannelGroup,
  const int64_t gOutChannelGroup,
  const int64_t gInGroupOffset,
  const int64_t gOutGroupOffset,
  const int64_t kernGroupOffset,
  const int64_t gInPixels,
  const int64_t n,
  const int64_t c
)
{
  for (int64_t gip = 0; gip < gInPixels; gip+=VLEN) {
    const int64_t vl = gInPixels - gip < VLEN ? gInPixels - gip : VLEN ;

    const int64_t gInIndex = gInGroupOffset + ((n * gInChannel + c) * gInHeight ) * gInWidth + gip ;

    __vr vrseq = _vel_vseq_vl(vl) ;			// hw
    __vr vridx = _vel_vaddsl_vsvl(gip, vrseq, vl) ;	// op + hw

    __vr vrsum = _vel_vbrdl_vsl(0UL, vl) ;

    for (int64_t k=0; k<gOutChannelGroup; k++) {

      const float *pGOutChannel = pGOut + gOutGroupOffset + ((n * gOutChannel + k) * gOutHeight * gOutWidth ) ;
      const float *pKerValue    = pKernel + kernGroupOffset + (k * gInChannelGroup + c) ;

      /* memory access errors might be caused */
      __vr vrgout = _vel_vldu_vssl(4,&pGOutChannel[gip], vl) ;
      vrsum = _vel_vfmads_vvsvl(vrsum, pKerValue[0], vrgout, vl) ;

    } // gInChannel

    _vel_vstu_vssl(vrsum, 4, pGIn+gInIndex, vl) ;

  } // gInPixels
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
  const int64_t gInChannelGroup,
  const int64_t gOutChannelGroup,
  const int64_t gInGroupOffset,
  const int64_t gOutGroupOffset,
  const int64_t kernGroupOffset,
  const int64_t gInPixels,
  const int64_t n,
  const int64_t c
)
{
  for (int64_t gip = 0; gip < gInPixels; gip+=VLEN) {
    const int64_t vl = gInPixels - gip < VLEN ? gInPixels - gip : VLEN ;

    const int64_t gInIndex = gInGroupOffset + ((n * gInChannel + c) * gInHeight ) * gInWidth + gip ;

    __vr vrseq = _vel_vseq_vl(vl) ;			// hw
    __vr vridx = _vel_vaddsl_vsvl(gip, vrseq, vl) ;	// op + hw

    __vr vrsum01 = _vel_pvbrd_vsl(0UL, vl) ;

    for (int64_t k=0; k<gOutChannelGroup; k++) {

      const float *pGOutChannel = pGOut + gOutGroupOffset + ((n * gOutChannel + k) * gOutHeight * gOutWidth ) ;
      const float *pKerValue    = pKernel + kernGroupOffset + (k * gInChannelGroup + c) ;

      /* memory access errors might be caused */
      __vr vrgout = _vel_vldu_vssl(4,&pGOutChannel[gip], vl) ;
      __vr vrgoutP = _vel_vshf_vvvsl(vrgout, vrgout, VE_VSHUFFLE_YUZU, vl) ;
      const uint64_t kerValue01 = _vel_pack_f32p(pKerValue,
						 pKerValue+ 1) ;
      vrsum01 = _vel_pvfmad_vvsvl(vrsum01, kerValue01, vrgoutP, vl) ;

    } // gInChannel

    _vel_vstu_vssl(vrsum01, 4, pGIn+gInIndex, vl) ;
    _vel_vstl_vssl(vrsum01, 4, pGIn+gInIndex+  gInPixels, vl) ;

  } // gInPixels
}

static inline void c2p(
  const float * restrict pGOut,
  const float * restrict pKernel,
  float * restrict const pGIn,
  const int64_t gOutChannel,
  const int64_t gOutWidth,
  const int64_t gOutHeight,
  const int64_t gInChannel,
  const int64_t gInWidth,
  const int64_t gInHeight,
  const int64_t gInChannelGroup,
  const int64_t gOutChannelGroup,
  const int64_t gInGroupOffset,
  const int64_t gOutGroupOffset,
  const int64_t kernGroupOffset,
  const int64_t gInPixels,
  const int64_t n,
  const int64_t c
)
{
  for (int64_t gip = 0; gip < gInPixels; gip+=VLEN) {
    const int64_t vl = gInPixels - gip < VLEN ? gInPixels - gip : VLEN ;

    const int64_t gInIndex = gInGroupOffset + ((n * gInChannel + c) * gInHeight ) * gInWidth + gip ;

    __vr vrseq = _vel_vseq_vl(vl) ;			// hw
    __vr vridx = _vel_vaddsl_vsvl(gip, vrseq, vl) ;	// op + hw

    __vr vrsum01 = _vel_pvbrd_vsl(0UL, vl) ;

    for (int64_t k=0; k<gOutChannelGroup; k++) {

      const float *pGOutChannel = pGOut + gOutGroupOffset + ((n * gOutChannel + k) * gOutHeight * gOutWidth ) ;
      const float *pKerValue    = pKernel + kernGroupOffset + (k * gInChannelGroup + c) ;
      const uint64_t *pKerValue_u64 = (const uint64_t *) pKerValue ;

      /* memory access errors might be caused */
      __vr vrgout = _vel_vldu_vssl(4,&pGOutChannel[gip], vl) ;
      __vr vrgoutP = _vel_vshf_vvvsl(vrgout, vrgout, VE_VSHUFFLE_YUZU, vl) ;

      vrsum01 = _vel_pvfmad_vvsvl(vrsum01, pKerValue_u64[0], vrgoutP, vl) ;

    } // gInChannel

    _vel_vstl_vssl(vrsum01, 4, pGIn+gInIndex, vl) ;
    _vel_vstu_vssl(vrsum01, 4, pGIn+gInIndex+  gInPixels, vl) ;

  } // gInPixels
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
  const int64_t gInChannelGroup,
  const int64_t gOutChannelGroup,
  const int64_t gInGroupOffset,
  const int64_t gOutGroupOffset,
  const int64_t kernGroupOffset,
  const int64_t gInPixels,
  const int64_t n,
  const int64_t c
)
{
  for (int64_t gip = 0; gip < gInPixels; gip+=VLEN) {
    const int64_t vl = gInPixels - gip < VLEN ? gInPixels - gip : VLEN ;

    const int64_t gInIndex = gInGroupOffset + ((n * gInChannel + c) * gInHeight ) * gInWidth + gip ;

    __vr vrseq = _vel_vseq_vl(vl) ;			// hw
    __vr vridx = _vel_vaddsl_vsvl(gip, vrseq, vl) ;	// op + hw

    __vr vrsum01 = _vel_pvbrd_vsl(0UL, vl) ;
    __vr vrsum23 = _vel_pvbrd_vsl(0UL, vl) ;

    for (int64_t k=0; k<gOutChannelGroup; k++) {

      const float *pGOutChannel = pGOut + gOutGroupOffset + ((n * gOutChannel + k) * gOutHeight * gOutWidth ) ;
      const float *pKerValue    = pKernel + kernGroupOffset + (k * gInChannelGroup + c) ;

      /* memory access errors might be caused */
      __vr vrgout = _vel_vldu_vssl(4,&pGOutChannel[gip], vl) ;
      __vr vrgoutP = _vel_vshf_vvvsl(vrgout, vrgout, VE_VSHUFFLE_YUZU, vl) ;
      const uint64_t kerValue01 = _vel_pack_f32p(pKerValue,
						 pKerValue+ 1) ;
      const uint64_t kerValue23 = _vel_pack_f32p(pKerValue+ 2,
						 pKerValue+ 3) ;
      vrsum01 = _vel_pvfmad_vvsvl(vrsum01, kerValue01, vrgoutP, vl) ;
      vrsum23 = _vel_pvfmad_vvsvl(vrsum23, kerValue23, vrgoutP, vl) ;

    } // gInChannel

    _vel_vstu_vssl(vrsum01, 4, pGIn+gInIndex, vl) ;
    _vel_vstl_vssl(vrsum01, 4, pGIn+gInIndex+  gInPixels, vl) ;
    _vel_vstu_vssl(vrsum23, 4, pGIn+gInIndex+ 2*gInPixels, vl) ;
    _vel_vstl_vssl(vrsum23, 4, pGIn+gInIndex+ 3*gInPixels, vl) ;

  } // gInPixels
}


static inline void c4p(
  const float * restrict pGOut,
  const float * restrict pKernel,
  float * restrict const pGIn,
  const int64_t gOutChannel,
  const int64_t gOutWidth,
  const int64_t gOutHeight,
  const int64_t gInChannel,
  const int64_t gInWidth,
  const int64_t gInHeight,
  const int64_t gInChannelGroup,
  const int64_t gOutChannelGroup,
  const int64_t gInGroupOffset,
  const int64_t gOutGroupOffset,
  const int64_t kernGroupOffset,
  const int64_t gInPixels,
  const int64_t n,
  const int64_t c
)
{
  for (int64_t gip = 0; gip < gInPixels; gip+=VLEN) {
    const int64_t vl = gInPixels - gip < VLEN ? gInPixels - gip : VLEN ;

    const int64_t gInIndex = gInGroupOffset + ((n * gInChannel + c) * gInHeight ) * gInWidth + gip ;

    __vr vrseq = _vel_vseq_vl(vl) ;			// hw
    __vr vridx = _vel_vaddsl_vsvl(gip, vrseq, vl) ;	// op + hw

    __vr vrsum01 = _vel_pvbrd_vsl(0UL, vl) ;
    __vr vrsum23 = _vel_pvbrd_vsl(0UL, vl) ;

    for (int64_t k=0; k<gOutChannelGroup; k++) {

      const float *pGOutChannel = pGOut + gOutGroupOffset + ((n * gOutChannel + k) * gOutHeight * gOutWidth ) ;
      const float *pKerValue    = pKernel + kernGroupOffset + (k * gInChannelGroup + c) ;
      const uint64_t *pKerValue_u64 = (const uint64_t *) pKerValue ;

      /* memory access errors might be caused */
      __vr vrgout = _vel_vldu_vssl(4,&pGOutChannel[gip], vl) ;
      __vr vrgoutP = _vel_vshf_vvvsl(vrgout, vrgout, VE_VSHUFFLE_YUZU, vl) ;

      vrsum01 = _vel_pvfmad_vvsvl(vrsum01, pKerValue_u64[0], vrgoutP, vl) ;
      vrsum23 = _vel_pvfmad_vvsvl(vrsum23, pKerValue_u64[1], vrgoutP, vl) ;

    } // gInChannel

    _vel_vstl_vssl(vrsum01, 4, pGIn+gInIndex, vl) ;
    _vel_vstu_vssl(vrsum01, 4, pGIn+gInIndex+  gInPixels, vl) ;
    _vel_vstl_vssl(vrsum23, 4, pGIn+gInIndex+ 2*gInPixels, vl) ;
    _vel_vstu_vssl(vrsum23, 4, pGIn+gInIndex+ 3*gInPixels, vl) ;

  } // gInPixels
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
  const int64_t gInChannelGroup,
  const int64_t gOutChannelGroup,
  const int64_t gInGroupOffset,
  const int64_t gOutGroupOffset,
  const int64_t kernGroupOffset,
  const int64_t gInPixels,
  const int64_t n,
  const int64_t c
)
{
  for (int64_t gip = 0; gip < gInPixels; gip+=VLEN) {
    const int64_t vl = gInPixels - gip < VLEN ? gInPixels - gip : VLEN ;

    const int64_t gInIndex = gInGroupOffset + ((n * gInChannel + c) * gInHeight ) * gInWidth + gip ;

    __vr vrseq = _vel_vseq_vl(vl) ;			// hw
    __vr vridx = _vel_vaddsl_vsvl(gip, vrseq, vl) ;	// op + hw

    __vr vrsum01 = _vel_pvbrd_vsl(0UL, vl) ;
    __vr vrsum23 = _vel_pvbrd_vsl(0UL, vl) ;
    __vr vrsum45 = _vel_pvbrd_vsl(0UL, vl) ;
    __vr vrsum67 = _vel_pvbrd_vsl(0UL, vl) ;

    for (int64_t k=0; k<gOutChannelGroup; k++) {

      const float *pGOutChannel = pGOut + gOutGroupOffset + ((n * gOutChannel + k) * gOutHeight * gOutWidth ) ;
      const float *pKerValue    = pKernel + kernGroupOffset + (k * gInChannelGroup + c) ;

      /* memory access errors might be caused */
      __vr vrgout = _vel_vldu_vssl(4,&pGOutChannel[gip], vl) ;
      __vr vrgoutP = _vel_vshf_vvvsl(vrgout, vrgout, VE_VSHUFFLE_YUZU, vl) ;
      const uint64_t kerValue01 = _vel_pack_f32p(pKerValue,
						 pKerValue+ 1) ;
      const uint64_t kerValue23 = _vel_pack_f32p(pKerValue+ 2,
						 pKerValue+ 3) ;
      const uint64_t kerValue45 = _vel_pack_f32p(pKerValue+ 4,
						 pKerValue+ 5) ;
      const uint64_t kerValue67 = _vel_pack_f32p(pKerValue+ 6,
						 pKerValue+ 7) ;
      vrsum01 = _vel_pvfmad_vvsvl(vrsum01, kerValue01, vrgoutP, vl) ;
      vrsum23 = _vel_pvfmad_vvsvl(vrsum23, kerValue23, vrgoutP, vl) ;
      vrsum45 = _vel_pvfmad_vvsvl(vrsum45, kerValue45, vrgoutP, vl) ;
      vrsum67 = _vel_pvfmad_vvsvl(vrsum67, kerValue67, vrgoutP, vl) ;

    } // gInChannel

    _vel_vstu_vssl(vrsum01, 4, pGIn+gInIndex, vl) ;
    _vel_vstl_vssl(vrsum01, 4, pGIn+gInIndex+  gInPixels, vl) ;
    _vel_vstu_vssl(vrsum23, 4, pGIn+gInIndex+ 2*gInPixels, vl) ;
    _vel_vstl_vssl(vrsum23, 4, pGIn+gInIndex+ 3*gInPixels, vl) ;
    _vel_vstu_vssl(vrsum45, 4, pGIn+gInIndex+ 4*gInPixels, vl) ;
    _vel_vstl_vssl(vrsum45, 4, pGIn+gInIndex+ 5*gInPixels, vl) ;
    _vel_vstu_vssl(vrsum67, 4, pGIn+gInIndex+ 6*gInPixels, vl) ;
    _vel_vstl_vssl(vrsum67, 4, pGIn+gInIndex+ 7*gInPixels, vl) ;

  } // gInPixels
}


static inline void c8p(
  const float * restrict pGOut,
  const float * restrict pKernel,
  float * restrict const pGIn,
  const int64_t gOutChannel,
  const int64_t gOutWidth,
  const int64_t gOutHeight,
  const int64_t gInChannel,
  const int64_t gInWidth,
  const int64_t gInHeight,
  const int64_t gInChannelGroup,
  const int64_t gOutChannelGroup,
  const int64_t gInGroupOffset,
  const int64_t gOutGroupOffset,
  const int64_t kernGroupOffset,
  const int64_t gInPixels,
  const int64_t n,
  const int64_t c
)
{
  for (int64_t gip = 0; gip < gInPixels; gip+=VLEN) {
    const int64_t vl = gInPixels - gip < VLEN ? gInPixels - gip : VLEN ;

    const int64_t gInIndex = gInGroupOffset + ((n * gInChannel + c) * gInHeight ) * gInWidth + gip ;

    __vr vrseq = _vel_vseq_vl(vl) ;			// hw
    __vr vridx = _vel_vaddsl_vsvl(gip, vrseq, vl) ;	// op + hw

    __vr vrsum01 = _vel_pvbrd_vsl(0UL, vl) ;
    __vr vrsum23 = _vel_pvbrd_vsl(0UL, vl) ;
    __vr vrsum45 = _vel_pvbrd_vsl(0UL, vl) ;
    __vr vrsum67 = _vel_pvbrd_vsl(0UL, vl) ;

    for (int64_t k=0; k<gOutChannelGroup; k++) {

      const float *pGOutChannel = pGOut + gOutGroupOffset + ((n * gOutChannel + k) * gOutHeight * gOutWidth ) ;
      const float *pKerValue    = pKernel + kernGroupOffset + (k * gInChannelGroup + c) ;
      const uint64_t *pKerValue_u64 = (const uint64_t *) pKerValue ;

      /* memory access errors might be caused */
      __vr vrgout = _vel_vldu_vssl(4,&pGOutChannel[gip], vl) ;
      __vr vrgoutP = _vel_vshf_vvvsl(vrgout, vrgout, VE_VSHUFFLE_YUZU, vl) ;

      vrsum01 = _vel_pvfmad_vvsvl(vrsum01, pKerValue_u64[0], vrgoutP, vl) ;
      vrsum23 = _vel_pvfmad_vvsvl(vrsum23, pKerValue_u64[1], vrgoutP, vl) ;
      vrsum45 = _vel_pvfmad_vvsvl(vrsum45, pKerValue_u64[2], vrgoutP, vl) ;
      vrsum67 = _vel_pvfmad_vvsvl(vrsum67, pKerValue_u64[3], vrgoutP, vl) ;

    } // gInChannel

    _vel_vstl_vssl(vrsum01, 4, pGIn+gInIndex, vl) ;
    _vel_vstu_vssl(vrsum01, 4, pGIn+gInIndex+  gInPixels, vl) ;
    _vel_vstl_vssl(vrsum23, 4, pGIn+gInIndex+ 2*gInPixels, vl) ;
    _vel_vstu_vssl(vrsum23, 4, pGIn+gInIndex+ 3*gInPixels, vl) ;
    _vel_vstl_vssl(vrsum45, 4, pGIn+gInIndex+ 4*gInPixels, vl) ;
    _vel_vstu_vssl(vrsum45, 4, pGIn+gInIndex+ 5*gInPixels, vl) ;
    _vel_vstl_vssl(vrsum67, 4, pGIn+gInIndex+ 6*gInPixels, vl) ;
    _vel_vstu_vssl(vrsum67, 4, pGIn+gInIndex+ 7*gInPixels, vl) ;

  } // gInPixels
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
  const int64_t gInChannelGroup,
  const int64_t gOutChannelGroup,
  const int64_t gInGroupOffset,
  const int64_t gOutGroupOffset,
  const int64_t kernGroupOffset,
  const int64_t gInPixels,
  const int64_t n,
  const int64_t c
)
{
  for (int64_t gip = 0; gip < gInPixels; gip+=VLEN) {
    const int64_t vl = gInPixels - gip < VLEN ? gInPixels - gip : VLEN ;

    const int64_t gInIndex = gInGroupOffset + ((n * gInChannel + c) * gInHeight ) * gInWidth + gip ;

    __vr vrseq = _vel_vseq_vl(vl) ;			// hw
    __vr vridx = _vel_vaddsl_vsvl(gip, vrseq, vl) ;	// op + hw

    __vr vrsum01 = _vel_pvbrd_vsl(0UL, vl) ;
    __vr vrsum23 = _vel_pvbrd_vsl(0UL, vl) ;
    __vr vrsum45 = _vel_pvbrd_vsl(0UL, vl) ;
    __vr vrsum67 = _vel_pvbrd_vsl(0UL, vl) ;
    __vr vrsum89 = _vel_pvbrd_vsl(0UL, vl) ;
    __vr vrsumAB = _vel_pvbrd_vsl(0UL, vl) ;
    __vr vrsumCD = _vel_pvbrd_vsl(0UL, vl) ;
    __vr vrsumEF = _vel_pvbrd_vsl(0UL, vl) ;

    for (int64_t k=0; k<gOutChannelGroup; k++) {

      const float *pGOutChannel = pGOut + gOutGroupOffset + ((n * gOutChannel + k) * gOutHeight * gOutWidth ) ;
      const float *pKerValue    = pKernel + kernGroupOffset + (k * gInChannelGroup + c) ;

      /* memory access errors might be caused */
      __vr vrgout = _vel_vldu_vssl(4,&pGOutChannel[gip], vl) ;
      __vr vrgoutP = _vel_vshf_vvvsl(vrgout, vrgout, VE_VSHUFFLE_YUZU, vl) ;
      const uint64_t kerValue01 = _vel_pack_f32p(pKerValue,
						 pKerValue+ 1) ;
      const uint64_t kerValue23 = _vel_pack_f32p(pKerValue+ 2,
						 pKerValue+ 3) ;
      const uint64_t kerValue45 = _vel_pack_f32p(pKerValue+ 4,
						 pKerValue+ 5) ;
      const uint64_t kerValue67 = _vel_pack_f32p(pKerValue+ 6,
						 pKerValue+ 7) ;
      const uint64_t kerValue89 = _vel_pack_f32p(pKerValue+ 8,
						 pKerValue+ 9) ;
      const uint64_t kerValueAB = _vel_pack_f32p(pKerValue+10,
						 pKerValue+11) ;
      const uint64_t kerValueCD = _vel_pack_f32p(pKerValue+12,
						 pKerValue+13) ;
      const uint64_t kerValueEF = _vel_pack_f32p(pKerValue+14,
						 pKerValue+15) ;
      vrsum01 = _vel_pvfmad_vvsvl(vrsum01, kerValue01, vrgoutP, vl) ;
      vrsum23 = _vel_pvfmad_vvsvl(vrsum23, kerValue23, vrgoutP, vl) ;
      vrsum45 = _vel_pvfmad_vvsvl(vrsum45, kerValue45, vrgoutP, vl) ;
      vrsum67 = _vel_pvfmad_vvsvl(vrsum67, kerValue67, vrgoutP, vl) ;
      vrsum89 = _vel_pvfmad_vvsvl(vrsum89, kerValue89, vrgoutP, vl) ;
      vrsumAB = _vel_pvfmad_vvsvl(vrsumAB, kerValueAB, vrgoutP, vl) ;
      vrsumCD = _vel_pvfmad_vvsvl(vrsumCD, kerValueCD, vrgoutP, vl) ;
      vrsumEF = _vel_pvfmad_vvsvl(vrsumEF, kerValueEF, vrgoutP, vl) ;

    } // gInChannel

    _vel_vstu_vssl(vrsum01, 4, pGIn+gInIndex, vl) ;
    _vel_vstl_vssl(vrsum01, 4, pGIn+gInIndex+  gInPixels, vl) ;
    _vel_vstu_vssl(vrsum23, 4, pGIn+gInIndex+ 2*gInPixels, vl) ;
    _vel_vstl_vssl(vrsum23, 4, pGIn+gInIndex+ 3*gInPixels, vl) ;
    _vel_vstu_vssl(vrsum45, 4, pGIn+gInIndex+ 4*gInPixels, vl) ;
    _vel_vstl_vssl(vrsum45, 4, pGIn+gInIndex+ 5*gInPixels, vl) ;
    _vel_vstu_vssl(vrsum67, 4, pGIn+gInIndex+ 6*gInPixels, vl) ;
    _vel_vstl_vssl(vrsum67, 4, pGIn+gInIndex+ 7*gInPixels, vl) ;
    _vel_vstu_vssl(vrsum89, 4, pGIn+gInIndex+ 8*gInPixels, vl) ;
    _vel_vstl_vssl(vrsum89, 4, pGIn+gInIndex+ 9*gInPixels, vl) ;
    _vel_vstu_vssl(vrsumAB, 4, pGIn+gInIndex+10*gInPixels, vl) ;
    _vel_vstl_vssl(vrsumAB, 4, pGIn+gInIndex+11*gInPixels, vl) ;
    _vel_vstu_vssl(vrsumCD, 4, pGIn+gInIndex+12*gInPixels, vl) ;
    _vel_vstl_vssl(vrsumCD, 4, pGIn+gInIndex+13*gInPixels, vl) ;
    _vel_vstu_vssl(vrsumEF, 4, pGIn+gInIndex+14*gInPixels, vl) ;
    _vel_vstl_vssl(vrsumEF, 4, pGIn+gInIndex+15*gInPixels, vl) ;

  } // gInPixels
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
  const int64_t gInChannelGroup,
  const int64_t gOutChannelGroup,
  const int64_t gInGroupOffset,
  const int64_t gOutGroupOffset,
  const int64_t kernGroupOffset,
  const int64_t gInPixels,
  const int64_t n,
  const int64_t c
)
{
  for (int64_t gip = 0; gip < gInPixels; gip+=VLEN) {
    const int64_t vl = gInPixels - gip < VLEN ? gInPixels - gip : VLEN ;

    const int64_t gInIndex = gInGroupOffset + ((n * gInChannel + c) * gInHeight ) * gInWidth + gip ;

    __vr vrseq = _vel_vseq_vl(vl) ;			// hw
    __vr vridx = _vel_vaddsl_vsvl(gip, vrseq, vl) ;	// op + hw

    __vr vrsum01 = _vel_pvbrd_vsl(0UL, vl) ;
    __vr vrsum23 = _vel_pvbrd_vsl(0UL, vl) ;
    __vr vrsum45 = _vel_pvbrd_vsl(0UL, vl) ;
    __vr vrsum67 = _vel_pvbrd_vsl(0UL, vl) ;
    __vr vrsum89 = _vel_pvbrd_vsl(0UL, vl) ;
    __vr vrsumAB = _vel_pvbrd_vsl(0UL, vl) ;
    __vr vrsumCD = _vel_pvbrd_vsl(0UL, vl) ;
    __vr vrsumEF = _vel_pvbrd_vsl(0UL, vl) ;

    for (int64_t k=0; k<gOutChannelGroup; k++) {

      const float *pGOutChannel = pGOut + gOutGroupOffset + ((n * gOutChannel + k) * gOutHeight * gOutWidth ) ;
      const float *pKerValue    = pKernel + kernGroupOffset + (k * gInChannelGroup + c) ;
      const uint64_t *pKerValue_u64 = (const uint64_t *) pKerValue ;

      /* memory access errors might be caused */
      __vr vrgout = _vel_vldu_vssl(4,&pGOutChannel[gip], vl) ;
      __vr vrgoutP = _vel_vshf_vvvsl(vrgout, vrgout, VE_VSHUFFLE_YUZU, vl) ;

      vrsum01 = _vel_pvfmad_vvsvl(vrsum01, pKerValue_u64[0], vrgoutP, vl) ;
      vrsum23 = _vel_pvfmad_vvsvl(vrsum23, pKerValue_u64[1], vrgoutP, vl) ;
      vrsum45 = _vel_pvfmad_vvsvl(vrsum45, pKerValue_u64[2], vrgoutP, vl) ;
      vrsum67 = _vel_pvfmad_vvsvl(vrsum67, pKerValue_u64[3], vrgoutP, vl) ;
      vrsum89 = _vel_pvfmad_vvsvl(vrsum89, pKerValue_u64[4], vrgoutP, vl) ;
      vrsumAB = _vel_pvfmad_vvsvl(vrsumAB, pKerValue_u64[5], vrgoutP, vl) ;
      vrsumCD = _vel_pvfmad_vvsvl(vrsumCD, pKerValue_u64[6], vrgoutP, vl) ;
      vrsumEF = _vel_pvfmad_vvsvl(vrsumEF, pKerValue_u64[7], vrgoutP, vl) ;

    } // gInChannel

    _vel_vstl_vssl(vrsum01, 4, pGIn+gInIndex, vl) ;
    _vel_vstu_vssl(vrsum01, 4, pGIn+gInIndex+  gInPixels, vl) ;
    _vel_vstl_vssl(vrsum23, 4, pGIn+gInIndex+ 2*gInPixels, vl) ;
    _vel_vstu_vssl(vrsum23, 4, pGIn+gInIndex+ 3*gInPixels, vl) ;
    _vel_vstl_vssl(vrsum45, 4, pGIn+gInIndex+ 4*gInPixels, vl) ;
    _vel_vstu_vssl(vrsum45, 4, pGIn+gInIndex+ 5*gInPixels, vl) ;
    _vel_vstl_vssl(vrsum67, 4, pGIn+gInIndex+ 6*gInPixels, vl) ;
    _vel_vstu_vssl(vrsum67, 4, pGIn+gInIndex+ 7*gInPixels, vl) ;
    _vel_vstl_vssl(vrsum89, 4, pGIn+gInIndex+ 8*gInPixels, vl) ;
    _vel_vstu_vssl(vrsum89, 4, pGIn+gInIndex+ 9*gInPixels, vl) ;
    _vel_vstl_vssl(vrsumAB, 4, pGIn+gInIndex+10*gInPixels, vl) ;
    _vel_vstu_vssl(vrsumAB, 4, pGIn+gInIndex+11*gInPixels, vl) ;
    _vel_vstl_vssl(vrsumCD, 4, pGIn+gInIndex+12*gInPixels, vl) ;
    _vel_vstu_vssl(vrsumCD, 4, pGIn+gInIndex+13*gInPixels, vl) ;
    _vel_vstl_vssl(vrsumEF, 4, pGIn+gInIndex+14*gInPixels, vl) ;
    _vel_vstu_vssl(vrsumEF, 4, pGIn+gInIndex+15*gInPixels, vl) ;

  } // gInPixels
}

vednnError_t
vednnConvolutionBackwardData_direct_dil1_str1_padsame_ker1 (
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
//  const int64_t kernWidth   = pParamKernel->width;		/* must be 1 */
//  const int64_t kernHeight  = pParamKernel->height;		/* must be 1 */

  const int64_t group          = pParamConv->group;
//  const int64_t strideWidth    = pParamConv->strideWidth; 	/* must be 1 */
//  const int64_t strideHeight   = pParamConv->strideHeight;	/* must be 1 */
  const int64_t padWidth       = pParamConv->padWidth;
  const int64_t padHeight      = pParamConv->padHeight;

//  const int64_t dilationWidth  = pParamConv->dilationWidth;	/* must be 1 */
//  const int64_t dilationHeight = pParamConv->dilationHeight;	/* must be 1 */

  const int64_t gOutChannelGroup = gOutChannel / group;
  const int64_t gInChannelGroup  = gInChannel  / group;

  const float * restrict pGOut   = pDataGradOut;
  const float * restrict pKernel = pDataKernel;
  float * restrict const pGIn    = pDataGradIn;

  const int gInPixels= gInHeight*gInWidth ;

  /* intrinsic version 1 */
  {

    const int64_t usePackedKernel = (((uint64_t)pKernel) & 0x07) == 0 && (gOutChannelGroup & 0x01) == 0 ?  1 : 0  ;

    for (int64_t n=0; n<batch; n++) {
      for (int64_t g = 0; g < group; g++) {

	int64_t gInGroupOffset  = g * gInChannelGroup * gInHeight * gInWidth;
	int64_t gOutGroupOffset = g * gOutChannelGroup  * gOutHeight  * gOutWidth;
	int64_t kernGroupOffset = g * gOutChannelGroup  * gInChannelGroup ;

	int c=0;
	if ( (gInChannelGroup & 0x01) == 1 ) {
	  c1(pGOut, pKernel, pGIn,
	     gOutChannel, gOutWidth, gOutHeight,
	     gInChannel, gInWidth, gInHeight,
	     gInChannelGroup, gOutChannelGroup,
	     gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	     gInPixels, n, c) ;

	  c+=1 ;
	}
	if ( ((gInChannelGroup >> 1) & 0x01) == 1 ) {
	  if( usePackedKernel ) {
	    c2p(pGOut, pKernel, pGIn,
	       gOutChannel, gOutWidth, gOutHeight,
	       gInChannel, gInWidth, gInHeight,
	       gInChannelGroup, gOutChannelGroup,
	       gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	       gInPixels, n, c) ;
	  }
	  else {
	    c2(pGOut, pKernel, pGIn,
	       gOutChannel, gOutWidth, gOutHeight,
	       gInChannel, gInWidth, gInHeight,
	       gInChannelGroup, gOutChannelGroup,
	       gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	       gInPixels, n, c) ;
	  }


	  c+=2 ;
	}
	if ( ((gInChannelGroup >> 2) & 0x01) == 1 ) {
	  if( usePackedKernel ) {
	    c4p(pGOut, pKernel, pGIn,
	       gOutChannel, gOutWidth, gOutHeight,
	       gInChannel, gInWidth, gInHeight,
	       gInChannelGroup, gOutChannelGroup,
	       gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	       gInPixels, n, c) ;
	  }
	  else {
	    c4(pGOut, pKernel, pGIn,
	       gOutChannel, gOutWidth, gOutHeight,
	       gInChannel, gInWidth, gInHeight,
	       gInChannelGroup, gOutChannelGroup,
	       gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	       gInPixels, n, c) ;
	  }

	  c+=4 ;
	}
	if ( ((gInChannelGroup >> 3) & 0x01) == 1 ) {
	  if( usePackedKernel ) {
	    c8p(pGOut, pKernel, pGIn,
	       gOutChannel, gOutWidth, gOutHeight,
	       gInChannel, gInWidth, gInHeight,
	       gInChannelGroup, gOutChannelGroup,
	       gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	       gInPixels, n, c) ;
	  }
	  else {
	    c8(pGOut, pKernel, pGIn,
	       gOutChannel, gOutWidth, gOutHeight,
	       gInChannel, gInWidth, gInHeight,
	       gInChannelGroup, gOutChannelGroup,
	       gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	       gInPixels, n, c) ;
	  }

	  c+=8 ;
	}
	for (; c<gInChannelGroup; c+=16) {
	  if( usePackedKernel ) {
	    c16p(pGOut, pKernel, pGIn,
	       gOutChannel, gOutWidth, gOutHeight,
	       gInChannel, gInWidth, gInHeight,
	       gInChannelGroup, gOutChannelGroup,
	       gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	       gInPixels, n, c) ;
	  }
	  else {
	    c16(pGOut, pKernel, pGIn,
	       gOutChannel, gOutWidth, gOutHeight,
	       gInChannel, gInWidth, gInHeight,
	       gInChannelGroup, gOutChannelGroup,
	       gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	       gInPixels, n, c) ;
	  }
	} // gOutChannel
      } // group
    } // batch
  }


  return VEDNN_SUCCESS;
}
