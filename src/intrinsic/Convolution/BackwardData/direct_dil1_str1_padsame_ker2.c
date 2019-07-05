#include <stdint.h>
#include <stdlib.h>

#include "vednn.h"

#include "veintrin.h"
#define VLEN	(256)


static inline void k1(
  const float * restrict pGOut,
  const float * restrict pKernel,
  float * restrict const pGIn,
  const int64_t gOutChannel,
  const int64_t gOutWidth,
  const int64_t gOutHeight,
  const int64_t gInChannel,
  const int64_t gInWidth,
  const int64_t gInHeight,
  const int64_t kernWidth,
  const int64_t kernHeight,
  const int64_t gInChannelGroup,
  const int64_t gOutChannelGroup,
  const int64_t gInGroupOffset,
  const int64_t gOutGroupOffset,
  const int64_t kernGroupOffset,
  const int64_t gInPixels,
  const int64_t n,
  const int64_t k
)
{
  for (int64_t gip = 0; gip < gInPixels; gip+=VLEN) {
    const int64_t vl = gInPixels - gip < VLEN ? gInPixels - gip : VLEN ;

    const int64_t gInIndex = gInGroupOffset + ((n * gInChannel + k) * gInHeight ) * gInWidth + gip ;

    _ve_lvl(vl) ;

    __vr vrseq = _ve_vseq_v() ;			// hw
    __vr vridx = _ve_vaddsl_vsv(gip, vrseq) ;	// op + hw

    __vr vrsum = _ve_vbrdu_vs_f32(0.f) ;

    __vr vrh   = _ve_vdivsl_vvs(vridx, gInWidth) ;
    __vr vrw   = _ve_vsubsl_vvv(vridx, _ve_vmulul_vsv(gInWidth,vrh)) ;

    __vm256 vm_r1 = _ve_vfmkl_mcv(VECC_GE, _ve_vaddsl_vsv(0-1, vrh) ) ;
    __vm256 vm_s1 = _ve_vfmkl_mcv(VECC_GE, _ve_vaddsl_vsv(-1, vrw)) ;

    __vm256 vm_r0s1 = vm_s1 ;
    __vm256 vm_r1s0 = vm_r1 ;
    __vm256 vm_r1s1 = _ve_andm_mmm(vm_r1,vm_s1) ;

    for (int64_t c=0; c<gOutChannelGroup; c++) {

      const float *pGOutChannel = pGOut + gOutGroupOffset + ((n * gOutChannel + c) * gOutHeight * gOutWidth ) ;
      const float *pKerValue    = pKernel + kernGroupOffset + ((c * gInChannelGroup + k) * kernHeight ) * kernWidth;

      /* memory access errors mihgt be caused */
      __vr vrgout_r0s0 = _ve_vldu_vss(4,&pGOutChannel[gip+(0-0)*gOutWidth+(0-0)]) ;
      vrsum = _ve_vfmads_vvsv(vrsum, pKerValue[0], vrgout_r0s0) ;

      __vr vrgout_r0s1 = _ve_vldu_vss(4,&pGOutChannel[gip+(0-0)*gOutWidth+(0-1)]) ;
      vrgout_r0s1 = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), vrgout_r0s1, vm_r0s1) ;
      vrsum = _ve_vfmads_vvsv(vrsum, pKerValue[1], vrgout_r0s1) ;

      __vr vrgout_r1s0 = _ve_vldu_vss(4,&pGOutChannel[gip+(0-1)*gOutWidth+(0-0)]) ;
      vrgout_r1s0 = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), vrgout_r1s0, vm_r1s0) ;
      vrsum = _ve_vfmads_vvsv(vrsum, pKerValue[2], vrgout_r1s0) ;

      __vr vrgout_r1s1 = _ve_vldu_vss(4,&pGOutChannel[gip+(0-1)*gOutWidth+(0-1)]) ;
      vrgout_r1s1 = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), vrgout_r1s1, vm_r1s1) ;
      vrsum = _ve_vfmads_vvsv(vrsum, pKerValue[3], vrgout_r1s1) ;
    } // gInChannel

    _ve_vstu_vss(vrsum, 4, pGIn+gInIndex) ;

  } // gInPixels
}


static inline void k2(
  const float * restrict pGOut,
  const float * restrict pKernel,
  float * restrict const pGIn,
  const int64_t gOutChannel,
  const int64_t gOutWidth,
  const int64_t gOutHeight,
  const int64_t gInChannel,
  const int64_t gInWidth,
  const int64_t gInHeight,
  const int64_t kernWidth,
  const int64_t kernHeight,
  const int64_t gInChannelGroup,
  const int64_t gOutChannelGroup,
  const int64_t gInGroupOffset,
  const int64_t gOutGroupOffset,
  const int64_t kernGroupOffset,
  const int64_t gInPixels,
  const int64_t n,
  const int64_t k
)
{
  for (int64_t gip = 0; gip < gInPixels; gip+=VLEN) {
    const int64_t vl = gInPixels - gip < VLEN ? gInPixels - gip : VLEN ;

    const int64_t gInIndex = gInGroupOffset + ((n * gInChannel + k) * gInHeight ) * gInWidth + gip ;

    _ve_lvl(vl) ;

    __vr vrseq = _ve_vseq_v() ;			// hw
    __vr vridx = _ve_vaddsl_vsv(gip, vrseq) ;	// op + hw

    __vr vrsum01 = _ve_pvbrd_vs_i64(0UL) ;

    __vr vrh   = _ve_vdivsl_vvs(vridx, gInWidth) ;
    __vr vrw   = _ve_vsubsl_vvv(vridx, _ve_vmulul_vsv(gInWidth,vrh)) ;

    __vm256 vm_r1 = _ve_vfmkl_mcv(VECC_GE, _ve_vaddsl_vsv(0-1, vrh) ) ;
    __vm256 vm_s1 = _ve_vfmkl_mcv(VECC_GE, _ve_vaddsl_vsv(-1, vrw)) ;

    __vm256 vm_r0s1 = vm_s1 ;
    __vm256 vm_r1s0 = vm_r1 ;
    __vm256 vm_r1s1 = _ve_andm_mmm(vm_r1,vm_s1) ;

    for (int64_t c=0; c<gOutChannelGroup; c++) {

      const float *pGOutChannel = pGOut + gOutGroupOffset + ((n * gOutChannel + c) * gOutHeight * gOutWidth ) ;
      const float *pKerValue    = pKernel + kernGroupOffset + ((c * gInChannelGroup + k) * kernHeight ) * kernWidth;

      /* memory access errors might be caused */
      __vr vrgout_r0s0 = _ve_vldu_vss(4,&pGOutChannel[gip+(0-0)*gOutWidth+(0-0)]) ;
      __vr vrgoutP_r0s0 = _ve_vshf_vvvs(vrgout_r0s0, vrgout_r0s0, VE_VSHUFFLE_YUZU) ;
      const uint64_t kerValue01_r0s0 = _ve_pack_f32p(pKerValue,
						     pKerValue+    kernHeight * kernWidth) ;
      vrsum01 = _ve_pvfmad_vvsv(vrsum01, kerValue01_r0s0, vrgoutP_r0s0) ;

      /* memory access errors might be caused */
      __vr vrgout_r0s1 = _ve_vldu_vss(4,&pGOutChannel[gip+(0-0)*gOutWidth+(0-1)]) ;
      vrgout_r0s1 = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), vrgout_r0s1, vm_r0s1) ;
      __vr vrgoutP_r0s1 = _ve_vshf_vvvs(vrgout_r0s1, vrgout_r0s1, VE_VSHUFFLE_YUZU) ;
      const uint64_t kerValue01_r0s1 = _ve_pack_f32p(pKerValue                             +1,
						     pKerValue+    kernHeight * kernWidth  +1) ;
      vrsum01 = _ve_pvfmad_vvsv(vrsum01, kerValue01_r0s1, vrgoutP_r0s1) ;

      /* memory access errors might be caused */
      __vr vrgout_r1s0 = _ve_vldu_vss(4,&pGOutChannel[gip+(0-1)*gOutWidth+(0-0)]) ;
      vrgout_r1s0 = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), vrgout_r1s0, vm_r1s0) ;
      __vr vrgoutP_r1s0 = _ve_vshf_vvvs(vrgout_r1s0, vrgout_r1s0, VE_VSHUFFLE_YUZU) ;
      const uint64_t kerValue01_r1s0 = _ve_pack_f32p(pKerValue                             +2,
						     pKerValue+    kernHeight * kernWidth  +2) ;
      vrsum01 = _ve_pvfmad_vvsv(vrsum01, kerValue01_r1s0, vrgoutP_r1s0) ;

      /* memory access errors might be caused */
      __vr vrgout_r1s1 = _ve_vldu_vss(4,&pGOutChannel[gip+(0-1)*gOutWidth+(0-1)]) ;
      vrgout_r1s1 = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), vrgout_r1s1, vm_r1s1) ;
      __vr vrgoutP_r1s1 = _ve_vshf_vvvs(vrgout_r1s1, vrgout_r1s1, VE_VSHUFFLE_YUZU) ;
      const uint64_t kerValue01_r1s1 = _ve_pack_f32p(pKerValue                             +3,
						     pKerValue+    kernHeight * kernWidth  +3) ;
      vrsum01 = _ve_pvfmad_vvsv(vrsum01, kerValue01_r1s1, vrgoutP_r1s1) ;
    } // gInChannel

    _ve_vstu_vss(vrsum01, 4, pGIn+gInIndex) ;
    _ve_vstl_vss(vrsum01, 4, pGIn+gInIndex+  gInPixels) ;

  } // gInPixels
}


static inline void k4(
  const float * restrict pGOut,
  const float * restrict pKernel,
  float * restrict const pGIn,
  const int64_t gOutChannel,
  const int64_t gOutWidth,
  const int64_t gOutHeight,
  const int64_t gInChannel,
  const int64_t gInWidth,
  const int64_t gInHeight,
  const int64_t kernWidth,
  const int64_t kernHeight,
  const int64_t gInChannelGroup,
  const int64_t gOutChannelGroup,
  const int64_t gInGroupOffset,
  const int64_t gOutGroupOffset,
  const int64_t kernGroupOffset,
  const int64_t gInPixels,
  const int64_t n,
  const int64_t k
)
{
  for (int64_t gip = 0; gip < gInPixels; gip+=VLEN) {
    const int64_t vl = gInPixels - gip < VLEN ? gInPixels - gip : VLEN ;

    const int64_t gInIndex = gInGroupOffset + ((n * gInChannel + k) * gInHeight ) * gInWidth + gip ;

    _ve_lvl(vl) ;

    __vr vrseq = _ve_vseq_v() ;			// hw
    __vr vridx = _ve_vaddsl_vsv(gip, vrseq) ;	// op + hw

    __vr vrsum01 = _ve_pvbrd_vs_i64(0UL) ;
    __vr vrsum23 = _ve_pvbrd_vs_i64(0UL) ;

    __vr vrh   = _ve_vdivsl_vvs(vridx, gInWidth) ;
    __vr vrw   = _ve_vsubsl_vvv(vridx, _ve_vmulul_vsv(gInWidth,vrh)) ;

    __vm256 vm_r1 = _ve_vfmkl_mcv(VECC_GE, _ve_vaddsl_vsv(0-1, vrh) ) ;
    __vm256 vm_s1 = _ve_vfmkl_mcv(VECC_GE, _ve_vaddsl_vsv(-1, vrw)) ;

    __vm256 vm_r0s1 = vm_s1 ;
    __vm256 vm_r1s0 = vm_r1 ;
    __vm256 vm_r1s1 = _ve_andm_mmm(vm_r1,vm_s1) ;

    for (int64_t c=0; c<gOutChannelGroup; c++) {

      const float *pGOutChannel = pGOut + gOutGroupOffset + ((n * gOutChannel + c) * gOutHeight * gOutWidth ) ;
      const float *pKerValue    = pKernel + kernGroupOffset + ((c * gInChannelGroup + k) * kernHeight ) * kernWidth;

      /* memory access errors might be caused */
      __vr vrgout_r0s0 = _ve_vldu_vss(4,&pGOutChannel[gip+(0-0)*gOutWidth+(0-0)]) ;
      __vr vrgoutP_r0s0 = _ve_vshf_vvvs(vrgout_r0s0, vrgout_r0s0, VE_VSHUFFLE_YUZU) ;
      const uint64_t kerValue01_r0s0 = _ve_pack_f32p(pKerValue,
						     pKerValue+    kernHeight * kernWidth) ;
      const uint64_t kerValue23_r0s0 = _ve_pack_f32p(pKerValue+ 2* kernHeight * kernWidth,
						     pKerValue+ 3* kernHeight * kernWidth) ;
      vrsum01 = _ve_pvfmad_vvsv(vrsum01, kerValue01_r0s0, vrgoutP_r0s0) ;
      vrsum23 = _ve_pvfmad_vvsv(vrsum23, kerValue23_r0s0, vrgoutP_r0s0) ;

      /* memory access errors might be caused */
      __vr vrgout_r0s1 = _ve_vldu_vss(4,&pGOutChannel[gip+(0-0)*gOutWidth+(0-1)]) ;
      vrgout_r0s1 = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), vrgout_r0s1, vm_r0s1) ;
      __vr vrgoutP_r0s1 = _ve_vshf_vvvs(vrgout_r0s1, vrgout_r0s1, VE_VSHUFFLE_YUZU) ;
      const uint64_t kerValue01_r0s1 = _ve_pack_f32p(pKerValue                             +1,
						     pKerValue+    kernHeight * kernWidth  +1) ;
      const uint64_t kerValue23_r0s1 = _ve_pack_f32p(pKerValue+ 2* kernHeight * kernWidth  +1,
						     pKerValue+ 3* kernHeight * kernWidth  +1) ;
      vrsum01 = _ve_pvfmad_vvsv(vrsum01, kerValue01_r0s1, vrgoutP_r0s1) ;
      vrsum23 = _ve_pvfmad_vvsv(vrsum23, kerValue23_r0s1, vrgoutP_r0s1) ;

      /* memory access errors might be caused */
      __vr vrgout_r1s0 = _ve_vldu_vss(4,&pGOutChannel[gip+(0-1)*gOutWidth+(0-0)]) ;
      vrgout_r1s0 = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), vrgout_r1s0, vm_r1s0) ;
      __vr vrgoutP_r1s0 = _ve_vshf_vvvs(vrgout_r1s0, vrgout_r1s0, VE_VSHUFFLE_YUZU) ;
      const uint64_t kerValue01_r1s0 = _ve_pack_f32p(pKerValue                             +2,
						     pKerValue+    kernHeight * kernWidth  +2) ;
      const uint64_t kerValue23_r1s0 = _ve_pack_f32p(pKerValue+ 2* kernHeight * kernWidth  +2,
						     pKerValue+ 3* kernHeight * kernWidth  +2) ;
      vrsum01 = _ve_pvfmad_vvsv(vrsum01, kerValue01_r1s0, vrgoutP_r1s0) ;
      vrsum23 = _ve_pvfmad_vvsv(vrsum23, kerValue23_r1s0, vrgoutP_r1s0) ;

      /* memory access errors might be caused */
      __vr vrgout_r1s1 = _ve_vldu_vss(4,&pGOutChannel[gip+(0-1)*gOutWidth+(0-1)]) ;
      vrgout_r1s1 = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), vrgout_r1s1, vm_r1s1) ;
      __vr vrgoutP_r1s1 = _ve_vshf_vvvs(vrgout_r1s1, vrgout_r1s1, VE_VSHUFFLE_YUZU) ;
      const uint64_t kerValue01_r1s1 = _ve_pack_f32p(pKerValue                             +3,
						     pKerValue+    kernHeight * kernWidth  +3) ;
      const uint64_t kerValue23_r1s1 = _ve_pack_f32p(pKerValue+ 2* kernHeight * kernWidth  +3,
						     pKerValue+ 3* kernHeight * kernWidth  +3) ;
      vrsum01 = _ve_pvfmad_vvsv(vrsum01, kerValue01_r1s1, vrgoutP_r1s1) ;
      vrsum23 = _ve_pvfmad_vvsv(vrsum23, kerValue23_r1s1, vrgoutP_r1s1) ;
    } // gInChannel

    _ve_vstu_vss(vrsum01, 4, pGIn+gInIndex) ;
    _ve_vstl_vss(vrsum01, 4, pGIn+gInIndex+  gInPixels) ;
    _ve_vstu_vss(vrsum23, 4, pGIn+gInIndex+2*gInPixels) ;
    _ve_vstl_vss(vrsum23, 4, pGIn+gInIndex+3*gInPixels) ;

  } // gInPixels
}

static inline void k8(
  const float * restrict pGOut,
  const float * restrict pKernel,
  float * restrict const pGIn,
  const int64_t gOutChannel,
  const int64_t gOutWidth,
  const int64_t gOutHeight,
  const int64_t gInChannel,
  const int64_t gInWidth,
  const int64_t gInHeight,
  const int64_t kernWidth,
  const int64_t kernHeight,
  const int64_t gInChannelGroup,
  const int64_t gOutChannelGroup,
  const int64_t gInGroupOffset,
  const int64_t gOutGroupOffset,
  const int64_t kernGroupOffset,
  const int64_t gInPixels,
  const int64_t n,
  const int64_t k
)
{
  for (int64_t gip = 0; gip < gInPixels; gip+=VLEN) {
    const int64_t vl = gInPixels - gip < VLEN ? gInPixels - gip : VLEN ;

    const int64_t gInIndex = gInGroupOffset + ((n * gInChannel + k) * gInHeight ) * gInWidth + gip ;

    _ve_lvl(vl) ;

    __vr vrseq = _ve_vseq_v() ;			// hw
    __vr vridx = _ve_vaddsl_vsv(gip, vrseq) ;	// op + hw

    __vr vrsum01 = _ve_pvbrd_vs_i64(0UL) ;
    __vr vrsum23 = _ve_pvbrd_vs_i64(0UL) ;
    __vr vrsum45 = _ve_pvbrd_vs_i64(0UL) ;
    __vr vrsum67 = _ve_pvbrd_vs_i64(0UL) ;

    __vr vrh   = _ve_vdivsl_vvs(vridx, gInWidth) ;
    __vr vrw   = _ve_vsubsl_vvv(vridx, _ve_vmulul_vsv(gInWidth,vrh)) ;

    __vm256 vm_r1 = _ve_vfmkl_mcv(VECC_GE, _ve_vaddsl_vsv(0-1, vrh) ) ;
    __vm256 vm_s1 = _ve_vfmkl_mcv(VECC_GE, _ve_vaddsl_vsv(-1, vrw)) ;

    __vm256 vm_r0s1 = vm_s1 ;
    __vm256 vm_r1s0 = vm_r1 ;
    __vm256 vm_r1s1 = _ve_andm_mmm(vm_r1,vm_s1) ;

    for (int64_t c=0; c<gOutChannelGroup; c++) {

      const float *pGOutChannel = pGOut + gOutGroupOffset + ((n * gOutChannel + c) * gOutHeight * gOutWidth ) ;
      const float *pKerValue    = pKernel + kernGroupOffset + ((c * gInChannelGroup + k) * kernHeight ) * kernWidth;

      /* memory access errors might be caused */
      __vr vrgout_r0s0 = _ve_vldu_vss(4,&pGOutChannel[gip+(0-0)*gOutWidth+(0-0)]) ;
      __vr vrgoutP_r0s0 = _ve_vshf_vvvs(vrgout_r0s0, vrgout_r0s0, VE_VSHUFFLE_YUZU) ;
      const uint64_t kerValue01_r0s0 = _ve_pack_f32p(pKerValue,
						     pKerValue+    kernHeight * kernWidth) ;
      const uint64_t kerValue23_r0s0 = _ve_pack_f32p(pKerValue+ 2* kernHeight * kernWidth,
						     pKerValue+ 3* kernHeight * kernWidth) ;
      const uint64_t kerValue45_r0s0 = _ve_pack_f32p(pKerValue+ 4* kernHeight * kernWidth,
						     pKerValue+ 5* kernHeight * kernWidth) ;
      const uint64_t kerValue67_r0s0 = _ve_pack_f32p(pKerValue+ 6* kernHeight * kernWidth,
						     pKerValue+ 7* kernHeight * kernWidth) ;
      vrsum01 = _ve_pvfmad_vvsv(vrsum01, kerValue01_r0s0, vrgoutP_r0s0) ;
      vrsum23 = _ve_pvfmad_vvsv(vrsum23, kerValue23_r0s0, vrgoutP_r0s0) ;
      vrsum45 = _ve_pvfmad_vvsv(vrsum45, kerValue45_r0s0, vrgoutP_r0s0) ;
      vrsum67 = _ve_pvfmad_vvsv(vrsum67, kerValue67_r0s0, vrgoutP_r0s0) ;

      /* memory access errors might be caused */
      __vr vrgout_r0s1 = _ve_vldu_vss(4,&pGOutChannel[gip+(0-0)*gOutWidth+(0-1)]) ;
      vrgout_r0s1 = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), vrgout_r0s1, vm_r0s1) ;
      __vr vrgoutP_r0s1 = _ve_vshf_vvvs(vrgout_r0s1, vrgout_r0s1, VE_VSHUFFLE_YUZU) ;
      const uint64_t kerValue01_r0s1 = _ve_pack_f32p(pKerValue                             +1,
						     pKerValue+    kernHeight * kernWidth  +1) ;
      const uint64_t kerValue23_r0s1 = _ve_pack_f32p(pKerValue+ 2* kernHeight * kernWidth  +1,
						     pKerValue+ 3* kernHeight * kernWidth  +1) ;
      const uint64_t kerValue45_r0s1 = _ve_pack_f32p(pKerValue+ 4* kernHeight * kernWidth  +1,
						     pKerValue+ 5* kernHeight * kernWidth  +1) ;
      const uint64_t kerValue67_r0s1 = _ve_pack_f32p(pKerValue+ 6* kernHeight * kernWidth  +1,
						     pKerValue+ 7* kernHeight * kernWidth  +1) ;
      vrsum01 = _ve_pvfmad_vvsv(vrsum01, kerValue01_r0s1, vrgoutP_r0s1) ;
      vrsum23 = _ve_pvfmad_vvsv(vrsum23, kerValue23_r0s1, vrgoutP_r0s1) ;
      vrsum45 = _ve_pvfmad_vvsv(vrsum45, kerValue45_r0s1, vrgoutP_r0s1) ;
      vrsum67 = _ve_pvfmad_vvsv(vrsum67, kerValue67_r0s1, vrgoutP_r0s1) ;

      /* memory access errors might be caused */
      __vr vrgout_r1s0 = _ve_vldu_vss(4,&pGOutChannel[gip+(0-1)*gOutWidth+(0-0)]) ;
      vrgout_r1s0 = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), vrgout_r1s0, vm_r1s0) ;
      __vr vrgoutP_r1s0 = _ve_vshf_vvvs(vrgout_r1s0, vrgout_r1s0, VE_VSHUFFLE_YUZU) ;
      const uint64_t kerValue01_r1s0 = _ve_pack_f32p(pKerValue                             +2,
						     pKerValue+    kernHeight * kernWidth  +2) ;
      const uint64_t kerValue23_r1s0 = _ve_pack_f32p(pKerValue+ 2* kernHeight * kernWidth  +2,
						     pKerValue+ 3* kernHeight * kernWidth  +2) ;
      const uint64_t kerValue45_r1s0 = _ve_pack_f32p(pKerValue+ 4* kernHeight * kernWidth  +2,
						     pKerValue+ 5* kernHeight * kernWidth  +2) ;
      const uint64_t kerValue67_r1s0 = _ve_pack_f32p(pKerValue+ 6* kernHeight * kernWidth  +2,
						     pKerValue+ 7* kernHeight * kernWidth  +2) ;
      vrsum01 = _ve_pvfmad_vvsv(vrsum01, kerValue01_r1s0, vrgoutP_r1s0) ;
      vrsum23 = _ve_pvfmad_vvsv(vrsum23, kerValue23_r1s0, vrgoutP_r1s0) ;
      vrsum45 = _ve_pvfmad_vvsv(vrsum45, kerValue45_r1s0, vrgoutP_r1s0) ;
      vrsum67 = _ve_pvfmad_vvsv(vrsum67, kerValue67_r1s0, vrgoutP_r1s0) ;

      /* memory access errors might be caused */
      __vr vrgout_r1s1 = _ve_vldu_vss(4,&pGOutChannel[gip+(0-1)*gOutWidth+(0-1)]) ;
      vrgout_r1s1 = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), vrgout_r1s1, vm_r1s1) ;
      __vr vrgoutP_r1s1 = _ve_vshf_vvvs(vrgout_r1s1, vrgout_r1s1, VE_VSHUFFLE_YUZU) ;
      const uint64_t kerValue01_r1s1 = _ve_pack_f32p(pKerValue                             +3,
						     pKerValue+    kernHeight * kernWidth  +3) ;
      const uint64_t kerValue23_r1s1 = _ve_pack_f32p(pKerValue+ 2* kernHeight * kernWidth  +3,
						     pKerValue+ 3* kernHeight * kernWidth  +3) ;
      const uint64_t kerValue45_r1s1 = _ve_pack_f32p(pKerValue+ 4* kernHeight * kernWidth  +3,
						     pKerValue+ 5* kernHeight * kernWidth  +3) ;
      const uint64_t kerValue67_r1s1 = _ve_pack_f32p(pKerValue+ 6* kernHeight * kernWidth  +3,
						     pKerValue+ 7* kernHeight * kernWidth  +3) ;
      vrsum01 = _ve_pvfmad_vvsv(vrsum01, kerValue01_r1s1, vrgoutP_r1s1) ;
      vrsum23 = _ve_pvfmad_vvsv(vrsum23, kerValue23_r1s1, vrgoutP_r1s1) ;
      vrsum45 = _ve_pvfmad_vvsv(vrsum45, kerValue45_r1s1, vrgoutP_r1s1) ;
      vrsum67 = _ve_pvfmad_vvsv(vrsum67, kerValue67_r1s1, vrgoutP_r1s1) ;
    } // gInChannel

    _ve_vstu_vss(vrsum01, 4, pGIn+gInIndex) ;
    _ve_vstl_vss(vrsum01, 4, pGIn+gInIndex+  gInPixels) ;
    _ve_vstu_vss(vrsum23, 4, pGIn+gInIndex+2*gInPixels) ;
    _ve_vstl_vss(vrsum23, 4, pGIn+gInIndex+3*gInPixels) ;
    _ve_vstu_vss(vrsum45, 4, pGIn+gInIndex+4*gInPixels) ;
    _ve_vstl_vss(vrsum45, 4, pGIn+gInIndex+5*gInPixels) ;
    _ve_vstu_vss(vrsum67, 4, pGIn+gInIndex+6*gInPixels) ;
    _ve_vstl_vss(vrsum67, 4, pGIn+gInIndex+7*gInPixels) ;

  } // gInPixels
}

static inline void k16(
  const float * restrict pGOut,
  const float * restrict pKernel,
  float * restrict const pGIn,
  const int64_t gOutChannel,
  const int64_t gOutWidth,
  const int64_t gOutHeight,
  const int64_t gInChannel,
  const int64_t gInWidth,
  const int64_t gInHeight,
  const int64_t kernWidth,
  const int64_t kernHeight,
  const int64_t gInChannelGroup,
  const int64_t gOutChannelGroup,
  const int64_t gInGroupOffset,
  const int64_t gOutGroupOffset,
  const int64_t kernGroupOffset,
  const int64_t gInPixels,
  const int64_t n,
  const int64_t k
)
{
  for (int64_t gip = 0; gip < gInPixels; gip+=VLEN) {
    const int64_t vl = gInPixels - gip < VLEN ? gInPixels - gip : VLEN ;

    const int64_t gInIndex = gInGroupOffset + ((n * gInChannel + k) * gInHeight ) * gInWidth + gip ;

    _ve_lvl(vl) ;

    __vr vrseq = _ve_vseq_v() ;			// hw
    __vr vridx = _ve_vaddsl_vsv(gip, vrseq) ;	// op + hw

    __vr vrsum01 = _ve_pvbrd_vs_i64(0UL) ;
    __vr vrsum23 = _ve_pvbrd_vs_i64(0UL) ;
    __vr vrsum45 = _ve_pvbrd_vs_i64(0UL) ;
    __vr vrsum67 = _ve_pvbrd_vs_i64(0UL) ;
    __vr vrsum89 = _ve_pvbrd_vs_i64(0UL) ;
    __vr vrsumAB = _ve_pvbrd_vs_i64(0UL) ;
    __vr vrsumCD = _ve_pvbrd_vs_i64(0UL) ;
    __vr vrsumEF = _ve_pvbrd_vs_i64(0UL) ;

    __vr vrh   = _ve_vdivsl_vvs(vridx, gInWidth) ;
    __vr vrw   = _ve_vsubsl_vvv(vridx, _ve_vmulul_vsv(gInWidth,vrh)) ;

    __vm256 vm_r1 = _ve_vfmkl_mcv(VECC_GE, _ve_vaddsl_vsv(0-1, vrh) ) ;
    __vm256 vm_s1 = _ve_vfmkl_mcv(VECC_GE, _ve_vaddsl_vsv(-1, vrw)) ;

    __vm256 vm_r0s1 = vm_s1 ;
    __vm256 vm_r1s0 = vm_r1 ;
    __vm256 vm_r1s1 = _ve_andm_mmm(vm_r1,vm_s1) ;

    for (int64_t c=0; c<gOutChannelGroup; c++) {

      const float *pGOutChannel = pGOut + gOutGroupOffset + ((n * gOutChannel + c) * gOutHeight * gOutWidth ) ;
      const float *pKerValue    = pKernel + kernGroupOffset + ((c * gInChannelGroup + k) * kernHeight ) * kernWidth;

      /* memory access errors might be caused */
      __vr vrgout_r0s0 = _ve_vldu_vss(4,&pGOutChannel[gip+(0-0)*gOutWidth+(0-0)]) ;
      __vr vrgoutP_r0s0 = _ve_vshf_vvvs(vrgout_r0s0, vrgout_r0s0, VE_VSHUFFLE_YUZU) ;
      const uint64_t kerValue01_r0s0 = _ve_pack_f32p(pKerValue,
						     pKerValue+    kernHeight * kernWidth) ;
      const uint64_t kerValue23_r0s0 = _ve_pack_f32p(pKerValue+ 2* kernHeight * kernWidth,
						     pKerValue+ 3* kernHeight * kernWidth) ;
      const uint64_t kerValue45_r0s0 = _ve_pack_f32p(pKerValue+ 4* kernHeight * kernWidth,
						     pKerValue+ 5* kernHeight * kernWidth) ;
      const uint64_t kerValue67_r0s0 = _ve_pack_f32p(pKerValue+ 6* kernHeight * kernWidth,
						     pKerValue+ 7* kernHeight * kernWidth) ;
      const uint64_t kerValue89_r0s0 = _ve_pack_f32p(pKerValue+ 8* kernHeight * kernWidth,
						     pKerValue+ 9* kernHeight * kernWidth) ;
      const uint64_t kerValueAB_r0s0 = _ve_pack_f32p(pKerValue+10* kernHeight * kernWidth,
						     pKerValue+11* kernHeight * kernWidth) ;
      const uint64_t kerValueCD_r0s0 = _ve_pack_f32p(pKerValue+12* kernHeight * kernWidth,
						     pKerValue+13* kernHeight * kernWidth) ;
      const uint64_t kerValueEF_r0s0 = _ve_pack_f32p(pKerValue+14* kernHeight * kernWidth,
						     pKerValue+15* kernHeight * kernWidth) ;
      vrsum01 = _ve_pvfmad_vvsv(vrsum01, kerValue01_r0s0, vrgoutP_r0s0) ;
      vrsum23 = _ve_pvfmad_vvsv(vrsum23, kerValue23_r0s0, vrgoutP_r0s0) ;
      vrsum45 = _ve_pvfmad_vvsv(vrsum45, kerValue45_r0s0, vrgoutP_r0s0) ;
      vrsum67 = _ve_pvfmad_vvsv(vrsum67, kerValue67_r0s0, vrgoutP_r0s0) ;
      vrsum89 = _ve_pvfmad_vvsv(vrsum89, kerValue89_r0s0, vrgoutP_r0s0) ;
      vrsumAB = _ve_pvfmad_vvsv(vrsumAB, kerValueAB_r0s0, vrgoutP_r0s0) ;
      vrsumCD = _ve_pvfmad_vvsv(vrsumCD, kerValueCD_r0s0, vrgoutP_r0s0) ;
      vrsumEF = _ve_pvfmad_vvsv(vrsumEF, kerValueEF_r0s0, vrgoutP_r0s0) ;

      /* memory access errors might be caused */
      __vr vrgout_r0s1 = _ve_vldu_vss(4,&pGOutChannel[gip+(0-0)*gOutWidth+(0-1)]) ;
      vrgout_r0s1 = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), vrgout_r0s1, vm_r0s1) ;
      __vr vrgoutP_r0s1 = _ve_vshf_vvvs(vrgout_r0s1, vrgout_r0s1, VE_VSHUFFLE_YUZU) ;
      const uint64_t kerValue01_r0s1 = _ve_pack_f32p(pKerValue                             +1,
						     pKerValue+    kernHeight * kernWidth  +1) ;
      const uint64_t kerValue23_r0s1 = _ve_pack_f32p(pKerValue+ 2* kernHeight * kernWidth  +1,
						     pKerValue+ 3* kernHeight * kernWidth  +1) ;
      const uint64_t kerValue45_r0s1 = _ve_pack_f32p(pKerValue+ 4* kernHeight * kernWidth  +1,
						     pKerValue+ 5* kernHeight * kernWidth  +1) ;
      const uint64_t kerValue67_r0s1 = _ve_pack_f32p(pKerValue+ 6* kernHeight * kernWidth  +1,
						     pKerValue+ 7* kernHeight * kernWidth  +1) ;
      const uint64_t kerValue89_r0s1 = _ve_pack_f32p(pKerValue+ 8* kernHeight * kernWidth  +1,
						     pKerValue+ 9* kernHeight * kernWidth  +1) ;
      const uint64_t kerValueAB_r0s1 = _ve_pack_f32p(pKerValue+10* kernHeight * kernWidth  +1,
						     pKerValue+11* kernHeight * kernWidth  +1) ;
      const uint64_t kerValueCD_r0s1 = _ve_pack_f32p(pKerValue+12* kernHeight * kernWidth  +1,
						     pKerValue+13* kernHeight * kernWidth  +1) ;
      const uint64_t kerValueEF_r0s1 = _ve_pack_f32p(pKerValue+14* kernHeight * kernWidth  +1,
						     pKerValue+15* kernHeight * kernWidth  +1) ;
      vrsum01 = _ve_pvfmad_vvsv(vrsum01, kerValue01_r0s1, vrgoutP_r0s1) ;
      vrsum23 = _ve_pvfmad_vvsv(vrsum23, kerValue23_r0s1, vrgoutP_r0s1) ;
      vrsum45 = _ve_pvfmad_vvsv(vrsum45, kerValue45_r0s1, vrgoutP_r0s1) ;
      vrsum67 = _ve_pvfmad_vvsv(vrsum67, kerValue67_r0s1, vrgoutP_r0s1) ;
      vrsum89 = _ve_pvfmad_vvsv(vrsum89, kerValue89_r0s1, vrgoutP_r0s1) ;
      vrsumAB = _ve_pvfmad_vvsv(vrsumAB, kerValueAB_r0s1, vrgoutP_r0s1) ;
      vrsumCD = _ve_pvfmad_vvsv(vrsumCD, kerValueCD_r0s1, vrgoutP_r0s1) ;
      vrsumEF = _ve_pvfmad_vvsv(vrsumEF, kerValueEF_r0s1, vrgoutP_r0s1) ;

      /* memory access errors might be caused */
      __vr vrgout_r1s0 = _ve_vldu_vss(4,&pGOutChannel[gip+(0-1)*gOutWidth+(0-0)]) ;
      vrgout_r1s0 = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), vrgout_r1s0, vm_r1s0) ;
      __vr vrgoutP_r1s0 = _ve_vshf_vvvs(vrgout_r1s0, vrgout_r1s0, VE_VSHUFFLE_YUZU) ;
      const uint64_t kerValue01_r1s0 = _ve_pack_f32p(pKerValue                             +2,
						     pKerValue+    kernHeight * kernWidth  +2) ;
      const uint64_t kerValue23_r1s0 = _ve_pack_f32p(pKerValue+ 2* kernHeight * kernWidth  +2,
						     pKerValue+ 3* kernHeight * kernWidth  +2) ;
      const uint64_t kerValue45_r1s0 = _ve_pack_f32p(pKerValue+ 4* kernHeight * kernWidth  +2,
						     pKerValue+ 5* kernHeight * kernWidth  +2) ;
      const uint64_t kerValue67_r1s0 = _ve_pack_f32p(pKerValue+ 6* kernHeight * kernWidth  +2,
						     pKerValue+ 7* kernHeight * kernWidth  +2) ;
      const uint64_t kerValue89_r1s0 = _ve_pack_f32p(pKerValue+ 8* kernHeight * kernWidth  +2,
						     pKerValue+ 9* kernHeight * kernWidth  +2) ;
      const uint64_t kerValueAB_r1s0 = _ve_pack_f32p(pKerValue+10* kernHeight * kernWidth  +2,
						     pKerValue+11* kernHeight * kernWidth  +2) ;
      const uint64_t kerValueCD_r1s0 = _ve_pack_f32p(pKerValue+12* kernHeight * kernWidth  +2,
						     pKerValue+13* kernHeight * kernWidth  +2) ;
      const uint64_t kerValueEF_r1s0 = _ve_pack_f32p(pKerValue+14* kernHeight * kernWidth  +2,
						     pKerValue+15* kernHeight * kernWidth  +2) ;
      vrsum01 = _ve_pvfmad_vvsv(vrsum01, kerValue01_r1s0, vrgoutP_r1s0) ;
      vrsum23 = _ve_pvfmad_vvsv(vrsum23, kerValue23_r1s0, vrgoutP_r1s0) ;
      vrsum45 = _ve_pvfmad_vvsv(vrsum45, kerValue45_r1s0, vrgoutP_r1s0) ;
      vrsum67 = _ve_pvfmad_vvsv(vrsum67, kerValue67_r1s0, vrgoutP_r1s0) ;
      vrsum89 = _ve_pvfmad_vvsv(vrsum89, kerValue89_r1s0, vrgoutP_r1s0) ;
      vrsumAB = _ve_pvfmad_vvsv(vrsumAB, kerValueAB_r1s0, vrgoutP_r1s0) ;
      vrsumCD = _ve_pvfmad_vvsv(vrsumCD, kerValueCD_r1s0, vrgoutP_r1s0) ;
      vrsumEF = _ve_pvfmad_vvsv(vrsumEF, kerValueEF_r1s0, vrgoutP_r1s0) ;

      /* memory access errors might be caused */
      __vr vrgout_r1s1 = _ve_vldu_vss(4,&pGOutChannel[gip+(0-1)*gOutWidth+(0-1)]) ;
      vrgout_r1s1 = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), vrgout_r1s1, vm_r1s1) ;
      __vr vrgoutP_r1s1 = _ve_vshf_vvvs(vrgout_r1s1, vrgout_r1s1, VE_VSHUFFLE_YUZU) ;
      const uint64_t kerValue01_r1s1 = _ve_pack_f32p(pKerValue                             +3,
						     pKerValue+    kernHeight * kernWidth  +3) ;
      const uint64_t kerValue23_r1s1 = _ve_pack_f32p(pKerValue+ 2* kernHeight * kernWidth  +3,
						     pKerValue+ 3* kernHeight * kernWidth  +3) ;
      const uint64_t kerValue45_r1s1 = _ve_pack_f32p(pKerValue+ 4* kernHeight * kernWidth  +3,
						     pKerValue+ 5* kernHeight * kernWidth  +3) ;
      const uint64_t kerValue67_r1s1 = _ve_pack_f32p(pKerValue+ 6* kernHeight * kernWidth  +3,
						     pKerValue+ 7* kernHeight * kernWidth  +3) ;
      const uint64_t kerValue89_r1s1 = _ve_pack_f32p(pKerValue+ 8* kernHeight * kernWidth  +3,
						     pKerValue+ 9* kernHeight * kernWidth  +3) ;
      const uint64_t kerValueAB_r1s1 = _ve_pack_f32p(pKerValue+10* kernHeight * kernWidth  +3,
						     pKerValue+11* kernHeight * kernWidth  +3) ;
      const uint64_t kerValueCD_r1s1 = _ve_pack_f32p(pKerValue+12* kernHeight * kernWidth  +3,
						     pKerValue+13* kernHeight * kernWidth  +3) ;
      const uint64_t kerValueEF_r1s1 = _ve_pack_f32p(pKerValue+14* kernHeight * kernWidth  +3,
						     pKerValue+15* kernHeight * kernWidth  +3) ;
      vrsum01 = _ve_pvfmad_vvsv(vrsum01, kerValue01_r1s1, vrgoutP_r1s1) ;
      vrsum23 = _ve_pvfmad_vvsv(vrsum23, kerValue23_r1s1, vrgoutP_r1s1) ;
      vrsum45 = _ve_pvfmad_vvsv(vrsum45, kerValue45_r1s1, vrgoutP_r1s1) ;
      vrsum67 = _ve_pvfmad_vvsv(vrsum67, kerValue67_r1s1, vrgoutP_r1s1) ;
      vrsum89 = _ve_pvfmad_vvsv(vrsum89, kerValue89_r1s1, vrgoutP_r1s1) ;
      vrsumAB = _ve_pvfmad_vvsv(vrsumAB, kerValueAB_r1s1, vrgoutP_r1s1) ;
      vrsumCD = _ve_pvfmad_vvsv(vrsumCD, kerValueCD_r1s1, vrgoutP_r1s1) ;
      vrsumEF = _ve_pvfmad_vvsv(vrsumEF, kerValueEF_r1s1, vrgoutP_r1s1) ;
    } // gInChannel

    _ve_vstu_vss(vrsum01, 4, pGIn+gInIndex) ;
    _ve_vstl_vss(vrsum01, 4, pGIn+gInIndex+  gInPixels) ;
    _ve_vstu_vss(vrsum23, 4, pGIn+gInIndex+2*gInPixels) ;
    _ve_vstl_vss(vrsum23, 4, pGIn+gInIndex+3*gInPixels) ;
    _ve_vstu_vss(vrsum45, 4, pGIn+gInIndex+4*gInPixels) ;
    _ve_vstl_vss(vrsum45, 4, pGIn+gInIndex+5*gInPixels) ;
    _ve_vstu_vss(vrsum67, 4, pGIn+gInIndex+6*gInPixels) ;
    _ve_vstl_vss(vrsum67, 4, pGIn+gInIndex+7*gInPixels) ;
    _ve_vstu_vss(vrsum89, 4, pGIn+gInIndex+8*gInPixels) ;
    _ve_vstl_vss(vrsum89, 4, pGIn+gInIndex+9*gInPixels) ;
    _ve_vstu_vss(vrsumAB, 4, pGIn+gInIndex+10*gInPixels) ;
    _ve_vstl_vss(vrsumAB, 4, pGIn+gInIndex+11*gInPixels) ;
    _ve_vstu_vss(vrsumCD, 4, pGIn+gInIndex+12*gInPixels) ;
    _ve_vstl_vss(vrsumCD, 4, pGIn+gInIndex+13*gInPixels) ;
    _ve_vstu_vss(vrsumEF, 4, pGIn+gInIndex+14*gInPixels) ;
    _ve_vstl_vss(vrsumEF, 4, pGIn+gInIndex+15*gInPixels) ;

  } // gInPixels
}



vednnError_t
vednnConvolutionBackwardData_direct_dil1_str1_padsame_ker2 (
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
  const int64_t kernWidth   = pParamKernel->width;		/* must be 2*padWidth  + 1 */
  const int64_t kernHeight  = pParamKernel->height;		/* must be 2*padHeight + 1 */

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

    for (int64_t n=0; n<batch; n++) {
      for (int64_t g = 0; g < group; g++) {

	int64_t gInGroupOffset  = g * gInChannelGroup * gInHeight * gInWidth;
	int64_t gOutGroupOffset = g * gOutChannelGroup  * gOutHeight  * gOutWidth;
	int64_t kernGroupOffset = g * gOutChannelGroup  * gInChannelGroup * kernHeight * kernWidth;

	int k=0;
	if ( (gInChannelGroup & 0x01) == 1 ) {
	  k1(pGOut, pKernel, pGIn,
	     gOutChannel, gOutWidth, gOutHeight,
	     gInChannel, gInWidth, gInHeight,
	     kernWidth, kernHeight,
	     gInChannelGroup, gOutChannelGroup,
	     gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	     gInPixels, n, k) ;

	  k+=1 ;
	}
	if ( ((gInChannelGroup >> 1) & 0x01) == 1 ) {
	  k2(pGOut, pKernel, pGIn,
	     gOutChannel, gOutWidth, gOutHeight,
	     gInChannel, gInWidth, gInHeight,
	     kernWidth, kernHeight,
	     gInChannelGroup, gOutChannelGroup,
	     gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	     gInPixels, n, k) ;

	  k+=2 ;
	}
	if ( ((gInChannelGroup >> 2) & 0x01) == 1 ) {
	  k4(pGOut, pKernel, pGIn,
	     gOutChannel, gOutWidth, gOutHeight,
	     gInChannel, gInWidth, gInHeight,
	     kernWidth, kernHeight,
	     gInChannelGroup, gOutChannelGroup,
	     gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	     gInPixels, n, k) ;

	  k+=4 ;
	}
	if ( ((gInChannelGroup >> 3) & 0x01) == 1 ) {
	  k8(pGOut, pKernel, pGIn,
	     gOutChannel, gOutWidth, gOutHeight,
	     gInChannel, gInWidth, gInHeight,
	     kernWidth, kernHeight,
	     gInChannelGroup, gOutChannelGroup,
	     gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	     gInPixels, n, k) ;

	  k+=8 ;
	}
	for (; k<gInChannelGroup; k+=16) {
	  k16(pGOut, pKernel, pGIn,
	     gOutChannel, gOutWidth, gOutHeight,
	     gInChannel, gInWidth, gInHeight,
	     kernWidth, kernHeight,
	     gInChannelGroup, gOutChannelGroup,
	     gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	     gInPixels, n, k) ;
	} // gOutChannel
      } // group
    } // batch
  }


  return VEDNN_SUCCESS;
}
