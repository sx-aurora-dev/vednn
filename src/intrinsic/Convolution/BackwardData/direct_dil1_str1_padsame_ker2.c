#include <stdint.h>
#include <stdlib.h>

#include "vednn.h"

#include "velintrin.h"
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

    __vr vrseq = _vel_vseq_vl(vl) ;			// hw
    __vr vridx = _vel_vaddsl_vsvl(gip, vrseq, vl) ;	// op + hw

    __vr vrsum = _vel_vbrds_vsl(0.f, vl) ;

    __vr vrh   = _vel_vdivsl_vvsl(vridx, gInWidth, vl) ;
    __vr vrw   = _vel_vsubsl_vvvl(vridx, _vel_vmulul_vsvl(gInWidth,vrh, vl), vl) ;

    __vm256 vm_r0 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutHeight-1, vrh, vl), vl);
    __vm256 vm_s0 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutWidth-1,  vrw, vl), vl);

    __vm256 vm_r0s0 = _vel_andm_mmm(vm_r0,vm_s0) ;
    __vm256 vm_r0s1 = vm_r0 ;
    __vm256 vm_r1s0 = vm_s0 ;

    for (int64_t c=0; c<gOutChannelGroup; c++) {

      const float *pGOutChannel = pGOut + gOutGroupOffset + ((n * gOutChannel + c) * gOutHeight * gOutWidth ) ;
      const float *pKerValue    = pKernel + kernGroupOffset + ((c * gInChannelGroup + k) * kernHeight ) * kernWidth;

      /* memory access errors mihgt be caused */
      __vr vrgout_r0s0 = _vel_vldu_vssl(4,&pGOutChannel[gip+(1-0)*gOutWidth+(1-0)], vl) ;
      vrgout_r0s0 = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrgout_r0s0, vm_r0s0, vl) ;
      vrsum = _vel_vfmads_vvsvl(vrsum, pKerValue[0], vrgout_r0s0, vl) ;

      __vr vrgout_r0s1 = _vel_vldu_vssl(4,&pGOutChannel[gip+(1-0)*gOutWidth+(1-1)], vl) ;
      vrgout_r0s1 = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrgout_r0s1, vm_r0s1, vl) ;
      vrsum = _vel_vfmads_vvsvl(vrsum, pKerValue[1], vrgout_r0s1, vl) ;

      __vr vrgout_r1s0 = _vel_vldu_vssl(4,&pGOutChannel[gip+(1-1)*gOutWidth+(1-0)], vl) ;
      vrgout_r1s0 = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrgout_r1s0, vm_r1s0, vl) ;
      vrsum = _vel_vfmads_vvsvl(vrsum, pKerValue[2], vrgout_r1s0, vl) ;

      __vr vrgout_r1s1 = _vel_vldu_vssl(4,&pGOutChannel[gip+(1-1)*gOutWidth+(1-1)], vl) ;
      vrsum = _vel_vfmads_vvsvl(vrsum, pKerValue[3], vrgout_r1s1, vl) ;
    } // gInChannel

    _vel_vstu_vssl(vrsum, 4, pGIn+gInIndex, vl) ;

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

    __vr vrseq = _vel_vseq_vl(vl) ;			// hw
    __vr vridx = _vel_vaddsl_vsvl(gip, vrseq, vl) ;	// op + hw

    __vr vrsum01 = _vel_pvbrd_vsl(0UL, vl) ;

    __vr vrh   = _vel_vdivsl_vvsl(vridx, gInWidth, vl) ;
    __vr vrw   = _vel_vsubsl_vvvl(vridx, _vel_vmulul_vsvl(gInWidth,vrh, vl), vl) ;

    __vm256 vm_r0 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutHeight-1, vrh, vl), vl);
    __vm256 vm_s0 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutWidth-1,  vrw, vl), vl);

    __vm256 vm_r0s0 = _vel_andm_mmm(vm_r0,vm_s0) ;
    __vm256 vm_r0s1 = vm_r0 ;
    __vm256 vm_r1s0 = vm_s0 ;

    for (int64_t c=0; c<gOutChannelGroup; c++) {

      const float *pGOutChannel = pGOut + gOutGroupOffset + ((n * gOutChannel + c) * gOutHeight * gOutWidth ) ;
      const float *pKerValue    = pKernel + kernGroupOffset + ((c * gInChannelGroup + k) * kernHeight ) * kernWidth;

      /* memory access errors might be caused */
      __vr vrgout_r0s0 = _vel_vldu_vssl(4,&pGOutChannel[gip+(1-0)*gOutWidth+(1-0)], vl) ;
      vrgout_r0s0 = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrgout_r0s0, vm_r0s0, vl) ;
      __vr vrgoutP_r0s0 = _vel_vshf_vvvsl(vrgout_r0s0, vrgout_r0s0, VE_VSHUFFLE_YUZU, vl) ;
      const uint64_t kerValue01_r0s0 = _vel_pack_f32p(pKerValue,
						      pKerValue+    kernHeight * kernWidth) ;
      vrsum01 = _vel_pvfmad_vvsvl(vrsum01, kerValue01_r0s0, vrgoutP_r0s0, vl) ;

      /* memory access errors might be caused */
      __vr vrgout_r0s1 = _vel_vldu_vssl(4,&pGOutChannel[gip+(1-0)*gOutWidth+(1-1)], vl) ;
      vrgout_r0s1 = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrgout_r0s1, vm_r0s1, vl) ;
      __vr vrgoutP_r0s1 = _vel_vshf_vvvsl(vrgout_r0s1, vrgout_r0s1, VE_VSHUFFLE_YUZU, vl) ;
      const uint64_t kerValue01_r0s1 = _vel_pack_f32p(pKerValue                             +1,
						      pKerValue+    kernHeight * kernWidth  +1) ;
      vrsum01 = _vel_pvfmad_vvsvl(vrsum01, kerValue01_r0s1, vrgoutP_r0s1, vl) ;

      /* memory access errors might be caused */
      __vr vrgout_r1s0 = _vel_vldu_vssl(4,&pGOutChannel[gip+(1-1)*gOutWidth+(1-0)], vl) ;
      vrgout_r1s0 = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrgout_r1s0, vm_r1s0, vl) ;
      __vr vrgoutP_r1s0 = _vel_vshf_vvvsl(vrgout_r1s0, vrgout_r1s0, VE_VSHUFFLE_YUZU, vl) ;
      const uint64_t kerValue01_r1s0 = _vel_pack_f32p(pKerValue                             +2,
						      pKerValue+    kernHeight * kernWidth  +2) ;
      vrsum01 = _vel_pvfmad_vvsvl(vrsum01, kerValue01_r1s0, vrgoutP_r1s0, vl) ;

      /* memory access errors might be caused */
      __vr vrgout_r1s1 = _vel_vldu_vssl(4,&pGOutChannel[gip+(1-1)*gOutWidth+(1-1)], vl) ;
      __vr vrgoutP_r1s1 = _vel_vshf_vvvsl(vrgout_r1s1, vrgout_r1s1, VE_VSHUFFLE_YUZU, vl) ;
      const uint64_t kerValue01_r1s1 = _vel_pack_f32p(pKerValue                             +3,
						      pKerValue+    kernHeight * kernWidth  +3) ;
      vrsum01 = _vel_pvfmad_vvsvl(vrsum01, kerValue01_r1s1, vrgoutP_r1s1, vl) ;
    } // gInChannel

    _vel_vstu_vssl(vrsum01, 4, pGIn+gInIndex, vl) ;
    _vel_vstl_vssl(vrsum01, 4, pGIn+gInIndex+  gInPixels, vl) ;

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

    __vr vrseq = _vel_vseq_vl(vl) ;			// hw
    __vr vridx = _vel_vaddsl_vsvl(gip, vrseq, vl) ;	// op + hw

    __vr vrsum01 = _vel_pvbrd_vsl(0UL, vl) ;
    __vr vrsum23 = _vel_pvbrd_vsl(0UL, vl) ;

    __vr vrh   = _vel_vdivsl_vvsl(vridx, gInWidth, vl) ;
    __vr vrw   = _vel_vsubsl_vvvl(vridx, _vel_vmulul_vsvl(gInWidth,vrh, vl), vl) ;

    __vm256 vm_r0 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutHeight-1, vrh, vl), vl);
    __vm256 vm_s0 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutWidth-1,  vrw, vl), vl);

    __vm256 vm_r0s0 = _vel_andm_mmm(vm_r0,vm_s0) ;
    __vm256 vm_r0s1 = vm_r0 ;
    __vm256 vm_r1s0 = vm_s0 ;

    for (int64_t c=0; c<gOutChannelGroup; c++) {

      const float *pGOutChannel = pGOut + gOutGroupOffset + ((n * gOutChannel + c) * gOutHeight * gOutWidth ) ;
      const float *pKerValue    = pKernel + kernGroupOffset + ((c * gInChannelGroup + k) * kernHeight ) * kernWidth;

      /* memory access errors might be caused */
      __vr vrgout_r0s0 = _vel_vldu_vssl(4,&pGOutChannel[gip+(1-0)*gOutWidth+(1-0)], vl) ;
      vrgout_r0s0 = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrgout_r0s0, vm_r0s0, vl) ;
      __vr vrgoutP_r0s0 = _vel_vshf_vvvsl(vrgout_r0s0, vrgout_r0s0, VE_VSHUFFLE_YUZU, vl) ;
      const uint64_t kerValue01_r0s0 = _vel_pack_f32p(pKerValue,
						      pKerValue+    kernHeight * kernWidth) ;
      const uint64_t kerValue23_r0s0 = _vel_pack_f32p(pKerValue+ 2* kernHeight * kernWidth,
						      pKerValue+ 3* kernHeight * kernWidth) ;
      vrsum01 = _vel_pvfmad_vvsvl(vrsum01, kerValue01_r0s0, vrgoutP_r0s0, vl) ;
      vrsum23 = _vel_pvfmad_vvsvl(vrsum23, kerValue23_r0s0, vrgoutP_r0s0, vl) ;

      /* memory access errors might be caused */
      __vr vrgout_r0s1 = _vel_vldu_vssl(4,&pGOutChannel[gip+(1-0)*gOutWidth+(1-1)], vl) ;
      vrgout_r0s1 = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrgout_r0s1, vm_r0s1, vl) ;
      __vr vrgoutP_r0s1 = _vel_vshf_vvvsl(vrgout_r0s1, vrgout_r0s1, VE_VSHUFFLE_YUZU, vl) ;
      const uint64_t kerValue01_r0s1 = _vel_pack_f32p(pKerValue                             +1,
						      pKerValue+    kernHeight * kernWidth  +1) ;
      const uint64_t kerValue23_r0s1 = _vel_pack_f32p(pKerValue+ 2* kernHeight * kernWidth  +1,
						      pKerValue+ 3* kernHeight * kernWidth  +1) ;
      vrsum01 = _vel_pvfmad_vvsvl(vrsum01, kerValue01_r0s1, vrgoutP_r0s1, vl) ;
      vrsum23 = _vel_pvfmad_vvsvl(vrsum23, kerValue23_r0s1, vrgoutP_r0s1, vl) ;

      /* memory access errors might be caused */
      __vr vrgout_r1s0 = _vel_vldu_vssl(4,&pGOutChannel[gip+(1-1)*gOutWidth+(1-0)], vl) ;
      vrgout_r1s0 = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrgout_r1s0, vm_r1s0, vl) ;
      __vr vrgoutP_r1s0 = _vel_vshf_vvvsl(vrgout_r1s0, vrgout_r1s0, VE_VSHUFFLE_YUZU, vl) ;
      const uint64_t kerValue01_r1s0 = _vel_pack_f32p(pKerValue                             +2,
						      pKerValue+    kernHeight * kernWidth  +2) ;
      const uint64_t kerValue23_r1s0 = _vel_pack_f32p(pKerValue+ 2* kernHeight * kernWidth  +2,
						      pKerValue+ 3* kernHeight * kernWidth  +2) ;
      vrsum01 = _vel_pvfmad_vvsvl(vrsum01, kerValue01_r1s0, vrgoutP_r1s0, vl) ;
      vrsum23 = _vel_pvfmad_vvsvl(vrsum23, kerValue23_r1s0, vrgoutP_r1s0, vl) ;

      /* memory access errors might be caused */
      __vr vrgout_r1s1 = _vel_vldu_vssl(4,&pGOutChannel[gip+(1-1)*gOutWidth+(1-1)], vl) ;
      __vr vrgoutP_r1s1 = _vel_vshf_vvvsl(vrgout_r1s1, vrgout_r1s1, VE_VSHUFFLE_YUZU, vl) ;
      const uint64_t kerValue01_r1s1 = _vel_pack_f32p(pKerValue                             +3,
						      pKerValue+    kernHeight * kernWidth  +3) ;
      const uint64_t kerValue23_r1s1 = _vel_pack_f32p(pKerValue+ 2* kernHeight * kernWidth  +3,
						      pKerValue+ 3* kernHeight * kernWidth  +3) ;
      vrsum01 = _vel_pvfmad_vvsvl(vrsum01, kerValue01_r1s1, vrgoutP_r1s1, vl) ;
      vrsum23 = _vel_pvfmad_vvsvl(vrsum23, kerValue23_r1s1, vrgoutP_r1s1, vl) ;
    } // gInChannel

    _vel_vstu_vssl(vrsum01, 4, pGIn+gInIndex, vl) ;
    _vel_vstl_vssl(vrsum01, 4, pGIn+gInIndex+  gInPixels, vl) ;
    _vel_vstu_vssl(vrsum23, 4, pGIn+gInIndex+2*gInPixels, vl) ;
    _vel_vstl_vssl(vrsum23, 4, pGIn+gInIndex+3*gInPixels, vl) ;

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

    __vr vrseq = _vel_vseq_vl(vl) ;			// hw
    __vr vridx = _vel_vaddsl_vsvl(gip, vrseq, vl) ;	// op + hw

    __vr vrsum01 = _vel_pvbrd_vsl(0UL, vl) ;
    __vr vrsum23 = _vel_pvbrd_vsl(0UL, vl) ;
    __vr vrsum45 = _vel_pvbrd_vsl(0UL, vl) ;
    __vr vrsum67 = _vel_pvbrd_vsl(0UL, vl) ;

    __vr vrh   = _vel_vdivsl_vvsl(vridx, gInWidth, vl) ;
    __vr vrw   = _vel_vsubsl_vvvl(vridx, _vel_vmulul_vsvl(gInWidth,vrh, vl), vl) ;

    __vm256 vm_r0 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutHeight-1, vrh, vl), vl);
    __vm256 vm_s0 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutWidth-1,  vrw, vl), vl);

    __vm256 vm_r0s0 = _vel_andm_mmm(vm_r0,vm_s0) ;
    __vm256 vm_r0s1 = vm_r0 ;
    __vm256 vm_r1s0 = vm_s0 ;

    for (int64_t c=0; c<gOutChannelGroup; c++) {

      const float *pGOutChannel = pGOut + gOutGroupOffset + ((n * gOutChannel + c) * gOutHeight * gOutWidth ) ;
      const float *pKerValue    = pKernel + kernGroupOffset + ((c * gInChannelGroup + k) * kernHeight ) * kernWidth;

      /* memory access errors might be caused */
      __vr vrgout_r0s0 = _vel_vldu_vssl(4,&pGOutChannel[gip+(1-0)*gOutWidth+(1-0)], vl) ;
      vrgout_r0s0 = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrgout_r0s0, vm_r0s0, vl) ;
      __vr vrgoutP_r0s0 = _vel_vshf_vvvsl(vrgout_r0s0, vrgout_r0s0, VE_VSHUFFLE_YUZU, vl) ;
      const uint64_t kerValue01_r0s0 = _vel_pack_f32p(pKerValue,
						      pKerValue+    kernHeight * kernWidth) ;
      const uint64_t kerValue23_r0s0 = _vel_pack_f32p(pKerValue+ 2* kernHeight * kernWidth,
						      pKerValue+ 3* kernHeight * kernWidth) ;
      const uint64_t kerValue45_r0s0 = _vel_pack_f32p(pKerValue+ 4* kernHeight * kernWidth,
						      pKerValue+ 5* kernHeight * kernWidth) ;
      const uint64_t kerValue67_r0s0 = _vel_pack_f32p(pKerValue+ 6* kernHeight * kernWidth,
						      pKerValue+ 7* kernHeight * kernWidth) ;
      vrsum01 = _vel_pvfmad_vvsvl(vrsum01, kerValue01_r0s0, vrgoutP_r0s0, vl) ;
      vrsum23 = _vel_pvfmad_vvsvl(vrsum23, kerValue23_r0s0, vrgoutP_r0s0, vl) ;
      vrsum45 = _vel_pvfmad_vvsvl(vrsum45, kerValue45_r0s0, vrgoutP_r0s0, vl) ;
      vrsum67 = _vel_pvfmad_vvsvl(vrsum67, kerValue67_r0s0, vrgoutP_r0s0, vl) ;

      /* memory access errors might be caused */
      __vr vrgout_r0s1 = _vel_vldu_vssl(4,&pGOutChannel[gip+(1-0)*gOutWidth+(1-1)], vl) ;
      vrgout_r0s1 = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrgout_r0s1, vm_r0s1, vl) ;
      __vr vrgoutP_r0s1 = _vel_vshf_vvvsl(vrgout_r0s1, vrgout_r0s1, VE_VSHUFFLE_YUZU, vl) ;
      const uint64_t kerValue01_r0s1 = _vel_pack_f32p(pKerValue                             +1,
						      pKerValue+    kernHeight * kernWidth  +1) ;
      const uint64_t kerValue23_r0s1 = _vel_pack_f32p(pKerValue+ 2* kernHeight * kernWidth  +1,
						      pKerValue+ 3* kernHeight * kernWidth  +1) ;
      const uint64_t kerValue45_r0s1 = _vel_pack_f32p(pKerValue+ 4* kernHeight * kernWidth  +1,
						      pKerValue+ 5* kernHeight * kernWidth  +1) ;
      const uint64_t kerValue67_r0s1 = _vel_pack_f32p(pKerValue+ 6* kernHeight * kernWidth  +1,
						      pKerValue+ 7* kernHeight * kernWidth  +1) ;
      vrsum01 = _vel_pvfmad_vvsvl(vrsum01, kerValue01_r0s1, vrgoutP_r0s1, vl) ;
      vrsum23 = _vel_pvfmad_vvsvl(vrsum23, kerValue23_r0s1, vrgoutP_r0s1, vl) ;
      vrsum45 = _vel_pvfmad_vvsvl(vrsum45, kerValue45_r0s1, vrgoutP_r0s1, vl) ;
      vrsum67 = _vel_pvfmad_vvsvl(vrsum67, kerValue67_r0s1, vrgoutP_r0s1, vl) ;

      /* memory access errors might be caused */
      __vr vrgout_r1s0 = _vel_vldu_vssl(4,&pGOutChannel[gip+(1-1)*gOutWidth+(1-0)], vl) ;
      vrgout_r1s0 = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrgout_r1s0, vm_r1s0, vl) ;
      __vr vrgoutP_r1s0 = _vel_vshf_vvvsl(vrgout_r1s0, vrgout_r1s0, VE_VSHUFFLE_YUZU, vl) ;
      const uint64_t kerValue01_r1s0 = _vel_pack_f32p(pKerValue                             +2,
						      pKerValue+    kernHeight * kernWidth  +2) ;
      const uint64_t kerValue23_r1s0 = _vel_pack_f32p(pKerValue+ 2* kernHeight * kernWidth  +2,
						      pKerValue+ 3* kernHeight * kernWidth  +2) ;
      const uint64_t kerValue45_r1s0 = _vel_pack_f32p(pKerValue+ 4* kernHeight * kernWidth  +2,
						      pKerValue+ 5* kernHeight * kernWidth  +2) ;
      const uint64_t kerValue67_r1s0 = _vel_pack_f32p(pKerValue+ 6* kernHeight * kernWidth  +2,
						      pKerValue+ 7* kernHeight * kernWidth  +2) ;
      vrsum01 = _vel_pvfmad_vvsvl(vrsum01, kerValue01_r1s0, vrgoutP_r1s0, vl) ;
      vrsum23 = _vel_pvfmad_vvsvl(vrsum23, kerValue23_r1s0, vrgoutP_r1s0, vl) ;
      vrsum45 = _vel_pvfmad_vvsvl(vrsum45, kerValue45_r1s0, vrgoutP_r1s0, vl) ;
      vrsum67 = _vel_pvfmad_vvsvl(vrsum67, kerValue67_r1s0, vrgoutP_r1s0, vl) ;

      /* memory access errors might be caused */
      __vr vrgout_r1s1 = _vel_vldu_vssl(4,&pGOutChannel[gip+(1-1)*gOutWidth+(1-1)], vl) ;
      __vr vrgoutP_r1s1 = _vel_vshf_vvvsl(vrgout_r1s1, vrgout_r1s1, VE_VSHUFFLE_YUZU, vl) ;
      const uint64_t kerValue01_r1s1 = _vel_pack_f32p(pKerValue                             +3,
						      pKerValue+    kernHeight * kernWidth  +3) ;
      const uint64_t kerValue23_r1s1 = _vel_pack_f32p(pKerValue+ 2* kernHeight * kernWidth  +3,
						      pKerValue+ 3* kernHeight * kernWidth  +3) ;
      const uint64_t kerValue45_r1s1 = _vel_pack_f32p(pKerValue+ 4* kernHeight * kernWidth  +3,
						      pKerValue+ 5* kernHeight * kernWidth  +3) ;
      const uint64_t kerValue67_r1s1 = _vel_pack_f32p(pKerValue+ 6* kernHeight * kernWidth  +3,
						      pKerValue+ 7* kernHeight * kernWidth  +3) ;
      vrsum01 = _vel_pvfmad_vvsvl(vrsum01, kerValue01_r1s1, vrgoutP_r1s1, vl) ;
      vrsum23 = _vel_pvfmad_vvsvl(vrsum23, kerValue23_r1s1, vrgoutP_r1s1, vl) ;
      vrsum45 = _vel_pvfmad_vvsvl(vrsum45, kerValue45_r1s1, vrgoutP_r1s1, vl) ;
      vrsum67 = _vel_pvfmad_vvsvl(vrsum67, kerValue67_r1s1, vrgoutP_r1s1, vl) ;
    } // gInChannel

    _vel_vstu_vssl(vrsum01, 4, pGIn+gInIndex, vl) ;
    _vel_vstl_vssl(vrsum01, 4, pGIn+gInIndex+  gInPixels, vl) ;
    _vel_vstu_vssl(vrsum23, 4, pGIn+gInIndex+2*gInPixels, vl) ;
    _vel_vstl_vssl(vrsum23, 4, pGIn+gInIndex+3*gInPixels, vl) ;
    _vel_vstu_vssl(vrsum45, 4, pGIn+gInIndex+4*gInPixels, vl) ;
    _vel_vstl_vssl(vrsum45, 4, pGIn+gInIndex+5*gInPixels, vl) ;
    _vel_vstu_vssl(vrsum67, 4, pGIn+gInIndex+6*gInPixels, vl) ;
    _vel_vstl_vssl(vrsum67, 4, pGIn+gInIndex+7*gInPixels, vl) ;

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

    __vr vrh   = _vel_vdivsl_vvsl(vridx, gInWidth, vl) ;
    __vr vrw   = _vel_vsubsl_vvvl(vridx, _vel_vmulul_vsvl(gInWidth,vrh, vl), vl) ;

    __vm256 vm_r0 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutHeight-1, vrh, vl), vl);
    __vm256 vm_s0 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutWidth-1,  vrw, vl), vl);

    __vm256 vm_r0s0 = _vel_andm_mmm(vm_r0,vm_s0) ;
    __vm256 vm_r0s1 = vm_r0 ;
    __vm256 vm_r1s0 = vm_s0 ;

    for (int64_t c=0; c<gOutChannelGroup; c++) {

      const float *pGOutChannel = pGOut + gOutGroupOffset + ((n * gOutChannel + c) * gOutHeight * gOutWidth ) ;
      const float *pKerValue    = pKernel + kernGroupOffset + ((c * gInChannelGroup + k) * kernHeight ) * kernWidth;

      /* memory access errors might be caused */
      __vr vrgout_r0s0 = _vel_vldu_vssl(4,&pGOutChannel[gip+(1-0)*gOutWidth+(1-0)], vl) ;
      vrgout_r0s0 = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrgout_r0s0, vm_r0s0, vl) ;
      __vr vrgoutP_r0s0 = _vel_vshf_vvvsl(vrgout_r0s0, vrgout_r0s0, VE_VSHUFFLE_YUZU, vl) ;
      const uint64_t kerValue01_r0s0 = _vel_pack_f32p(pKerValue,
						      pKerValue+    kernHeight * kernWidth) ;
      const uint64_t kerValue23_r0s0 = _vel_pack_f32p(pKerValue+ 2* kernHeight * kernWidth,
						      pKerValue+ 3* kernHeight * kernWidth) ;
      const uint64_t kerValue45_r0s0 = _vel_pack_f32p(pKerValue+ 4* kernHeight * kernWidth,
						      pKerValue+ 5* kernHeight * kernWidth) ;
      const uint64_t kerValue67_r0s0 = _vel_pack_f32p(pKerValue+ 6* kernHeight * kernWidth,
						      pKerValue+ 7* kernHeight * kernWidth) ;
      const uint64_t kerValue89_r0s0 = _vel_pack_f32p(pKerValue+ 8* kernHeight * kernWidth,
						      pKerValue+ 9* kernHeight * kernWidth) ;
      const uint64_t kerValueAB_r0s0 = _vel_pack_f32p(pKerValue+10* kernHeight * kernWidth,
						      pKerValue+11* kernHeight * kernWidth) ;
      const uint64_t kerValueCD_r0s0 = _vel_pack_f32p(pKerValue+12* kernHeight * kernWidth,
						      pKerValue+13* kernHeight * kernWidth) ;
      const uint64_t kerValueEF_r0s0 = _vel_pack_f32p(pKerValue+14* kernHeight * kernWidth,
						      pKerValue+15* kernHeight * kernWidth) ;
      vrsum01 = _vel_pvfmad_vvsvl(vrsum01, kerValue01_r0s0, vrgoutP_r0s0, vl) ;
      vrsum23 = _vel_pvfmad_vvsvl(vrsum23, kerValue23_r0s0, vrgoutP_r0s0, vl) ;
      vrsum45 = _vel_pvfmad_vvsvl(vrsum45, kerValue45_r0s0, vrgoutP_r0s0, vl) ;
      vrsum67 = _vel_pvfmad_vvsvl(vrsum67, kerValue67_r0s0, vrgoutP_r0s0, vl) ;
      vrsum89 = _vel_pvfmad_vvsvl(vrsum89, kerValue89_r0s0, vrgoutP_r0s0, vl) ;
      vrsumAB = _vel_pvfmad_vvsvl(vrsumAB, kerValueAB_r0s0, vrgoutP_r0s0, vl) ;
      vrsumCD = _vel_pvfmad_vvsvl(vrsumCD, kerValueCD_r0s0, vrgoutP_r0s0, vl) ;
      vrsumEF = _vel_pvfmad_vvsvl(vrsumEF, kerValueEF_r0s0, vrgoutP_r0s0, vl) ;

      /* memory access errors might be caused */
      __vr vrgout_r0s1 = _vel_vldu_vssl(4,&pGOutChannel[gip+(1-0)*gOutWidth+(1-1)], vl) ;
      vrgout_r0s1 = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrgout_r0s1, vm_r0s1, vl) ;
      __vr vrgoutP_r0s1 = _vel_vshf_vvvsl(vrgout_r0s1, vrgout_r0s1, VE_VSHUFFLE_YUZU, vl) ;
      const uint64_t kerValue01_r0s1 = _vel_pack_f32p(pKerValue                             +1,
						      pKerValue+    kernHeight * kernWidth  +1) ;
      const uint64_t kerValue23_r0s1 = _vel_pack_f32p(pKerValue+ 2* kernHeight * kernWidth  +1,
						      pKerValue+ 3* kernHeight * kernWidth  +1) ;
      const uint64_t kerValue45_r0s1 = _vel_pack_f32p(pKerValue+ 4* kernHeight * kernWidth  +1,
						      pKerValue+ 5* kernHeight * kernWidth  +1) ;
      const uint64_t kerValue67_r0s1 = _vel_pack_f32p(pKerValue+ 6* kernHeight * kernWidth  +1,
						      pKerValue+ 7* kernHeight * kernWidth  +1) ;
      const uint64_t kerValue89_r0s1 = _vel_pack_f32p(pKerValue+ 8* kernHeight * kernWidth  +1,
						      pKerValue+ 9* kernHeight * kernWidth  +1) ;
      const uint64_t kerValueAB_r0s1 = _vel_pack_f32p(pKerValue+10* kernHeight * kernWidth  +1,
						      pKerValue+11* kernHeight * kernWidth  +1) ;
      const uint64_t kerValueCD_r0s1 = _vel_pack_f32p(pKerValue+12* kernHeight * kernWidth  +1,
						      pKerValue+13* kernHeight * kernWidth  +1) ;
      const uint64_t kerValueEF_r0s1 = _vel_pack_f32p(pKerValue+14* kernHeight * kernWidth  +1,
						      pKerValue+15* kernHeight * kernWidth  +1) ;
      vrsum01 = _vel_pvfmad_vvsvl(vrsum01, kerValue01_r0s1, vrgoutP_r0s1, vl) ;
      vrsum23 = _vel_pvfmad_vvsvl(vrsum23, kerValue23_r0s1, vrgoutP_r0s1, vl) ;
      vrsum45 = _vel_pvfmad_vvsvl(vrsum45, kerValue45_r0s1, vrgoutP_r0s1, vl) ;
      vrsum67 = _vel_pvfmad_vvsvl(vrsum67, kerValue67_r0s1, vrgoutP_r0s1, vl) ;
      vrsum89 = _vel_pvfmad_vvsvl(vrsum89, kerValue89_r0s1, vrgoutP_r0s1, vl) ;
      vrsumAB = _vel_pvfmad_vvsvl(vrsumAB, kerValueAB_r0s1, vrgoutP_r0s1, vl) ;
      vrsumCD = _vel_pvfmad_vvsvl(vrsumCD, kerValueCD_r0s1, vrgoutP_r0s1, vl) ;
      vrsumEF = _vel_pvfmad_vvsvl(vrsumEF, kerValueEF_r0s1, vrgoutP_r0s1, vl) ;

      /* memory access errors might be caused */
      __vr vrgout_r1s0 = _vel_vldu_vssl(4,&pGOutChannel[gip+(1-1)*gOutWidth+(1-0)], vl) ;
      vrgout_r1s0 = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrgout_r1s0, vm_r1s0, vl) ;
      __vr vrgoutP_r1s0 = _vel_vshf_vvvsl(vrgout_r1s0, vrgout_r1s0, VE_VSHUFFLE_YUZU, vl) ;
      const uint64_t kerValue01_r1s0 = _vel_pack_f32p(pKerValue                             +2,
						      pKerValue+    kernHeight * kernWidth  +2) ;
      const uint64_t kerValue23_r1s0 = _vel_pack_f32p(pKerValue+ 2* kernHeight * kernWidth  +2,
						      pKerValue+ 3* kernHeight * kernWidth  +2) ;
      const uint64_t kerValue45_r1s0 = _vel_pack_f32p(pKerValue+ 4* kernHeight * kernWidth  +2,
						      pKerValue+ 5* kernHeight * kernWidth  +2) ;
      const uint64_t kerValue67_r1s0 = _vel_pack_f32p(pKerValue+ 6* kernHeight * kernWidth  +2,
						      pKerValue+ 7* kernHeight * kernWidth  +2) ;
      const uint64_t kerValue89_r1s0 = _vel_pack_f32p(pKerValue+ 8* kernHeight * kernWidth  +2,
						      pKerValue+ 9* kernHeight * kernWidth  +2) ;
      const uint64_t kerValueAB_r1s0 = _vel_pack_f32p(pKerValue+10* kernHeight * kernWidth  +2,
						      pKerValue+11* kernHeight * kernWidth  +2) ;
      const uint64_t kerValueCD_r1s0 = _vel_pack_f32p(pKerValue+12* kernHeight * kernWidth  +2,
						      pKerValue+13* kernHeight * kernWidth  +2) ;
      const uint64_t kerValueEF_r1s0 = _vel_pack_f32p(pKerValue+14* kernHeight * kernWidth  +2,
						      pKerValue+15* kernHeight * kernWidth  +2) ;
      vrsum01 = _vel_pvfmad_vvsvl(vrsum01, kerValue01_r1s0, vrgoutP_r1s0, vl) ;
      vrsum23 = _vel_pvfmad_vvsvl(vrsum23, kerValue23_r1s0, vrgoutP_r1s0, vl) ;
      vrsum45 = _vel_pvfmad_vvsvl(vrsum45, kerValue45_r1s0, vrgoutP_r1s0, vl) ;
      vrsum67 = _vel_pvfmad_vvsvl(vrsum67, kerValue67_r1s0, vrgoutP_r1s0, vl) ;
      vrsum89 = _vel_pvfmad_vvsvl(vrsum89, kerValue89_r1s0, vrgoutP_r1s0, vl) ;
      vrsumAB = _vel_pvfmad_vvsvl(vrsumAB, kerValueAB_r1s0, vrgoutP_r1s0, vl) ;
      vrsumCD = _vel_pvfmad_vvsvl(vrsumCD, kerValueCD_r1s0, vrgoutP_r1s0, vl) ;
      vrsumEF = _vel_pvfmad_vvsvl(vrsumEF, kerValueEF_r1s0, vrgoutP_r1s0, vl) ;

      /* memory access errors might be caused */
      __vr vrgout_r1s1 = _vel_vldu_vssl(4,&pGOutChannel[gip+(1-1)*gOutWidth+(1-1)], vl) ;
      __vr vrgoutP_r1s1 = _vel_vshf_vvvsl(vrgout_r1s1, vrgout_r1s1, VE_VSHUFFLE_YUZU, vl) ;
      const uint64_t kerValue01_r1s1 = _vel_pack_f32p(pKerValue                             +3,
						      pKerValue+    kernHeight * kernWidth  +3) ;
      const uint64_t kerValue23_r1s1 = _vel_pack_f32p(pKerValue+ 2* kernHeight * kernWidth  +3,
						      pKerValue+ 3* kernHeight * kernWidth  +3) ;
      const uint64_t kerValue45_r1s1 = _vel_pack_f32p(pKerValue+ 4* kernHeight * kernWidth  +3,
						      pKerValue+ 5* kernHeight * kernWidth  +3) ;
      const uint64_t kerValue67_r1s1 = _vel_pack_f32p(pKerValue+ 6* kernHeight * kernWidth  +3,
						      pKerValue+ 7* kernHeight * kernWidth  +3) ;
      const uint64_t kerValue89_r1s1 = _vel_pack_f32p(pKerValue+ 8* kernHeight * kernWidth  +3,
						      pKerValue+ 9* kernHeight * kernWidth  +3) ;
      const uint64_t kerValueAB_r1s1 = _vel_pack_f32p(pKerValue+10* kernHeight * kernWidth  +3,
						      pKerValue+11* kernHeight * kernWidth  +3) ;
      const uint64_t kerValueCD_r1s1 = _vel_pack_f32p(pKerValue+12* kernHeight * kernWidth  +3,
						      pKerValue+13* kernHeight * kernWidth  +3) ;
      const uint64_t kerValueEF_r1s1 = _vel_pack_f32p(pKerValue+14* kernHeight * kernWidth  +3,
						      pKerValue+15* kernHeight * kernWidth  +3) ;
      vrsum01 = _vel_pvfmad_vvsvl(vrsum01, kerValue01_r1s1, vrgoutP_r1s1, vl) ;
      vrsum23 = _vel_pvfmad_vvsvl(vrsum23, kerValue23_r1s1, vrgoutP_r1s1, vl) ;
      vrsum45 = _vel_pvfmad_vvsvl(vrsum45, kerValue45_r1s1, vrgoutP_r1s1, vl) ;
      vrsum67 = _vel_pvfmad_vvsvl(vrsum67, kerValue67_r1s1, vrgoutP_r1s1, vl) ;
      vrsum89 = _vel_pvfmad_vvsvl(vrsum89, kerValue89_r1s1, vrgoutP_r1s1, vl) ;
      vrsumAB = _vel_pvfmad_vvsvl(vrsumAB, kerValueAB_r1s1, vrgoutP_r1s1, vl) ;
      vrsumCD = _vel_pvfmad_vvsvl(vrsumCD, kerValueCD_r1s1, vrgoutP_r1s1, vl) ;
      vrsumEF = _vel_pvfmad_vvsvl(vrsumEF, kerValueEF_r1s1, vrgoutP_r1s1, vl) ;
    } // gInChannel

    _vel_vstu_vssl(vrsum01, 4, pGIn+gInIndex, vl) ;
    _vel_vstl_vssl(vrsum01, 4, pGIn+gInIndex+  gInPixels, vl) ;
    _vel_vstu_vssl(vrsum23, 4, pGIn+gInIndex+2*gInPixels, vl) ;
    _vel_vstl_vssl(vrsum23, 4, pGIn+gInIndex+3*gInPixels, vl) ;
    _vel_vstu_vssl(vrsum45, 4, pGIn+gInIndex+4*gInPixels, vl) ;
    _vel_vstl_vssl(vrsum45, 4, pGIn+gInIndex+5*gInPixels, vl) ;
    _vel_vstu_vssl(vrsum67, 4, pGIn+gInIndex+6*gInPixels, vl) ;
    _vel_vstl_vssl(vrsum67, 4, pGIn+gInIndex+7*gInPixels, vl) ;
    _vel_vstu_vssl(vrsum89, 4, pGIn+gInIndex+8*gInPixels, vl) ;
    _vel_vstl_vssl(vrsum89, 4, pGIn+gInIndex+9*gInPixels, vl) ;
    _vel_vstu_vssl(vrsumAB, 4, pGIn+gInIndex+10*gInPixels, vl) ;
    _vel_vstl_vssl(vrsumAB, 4, pGIn+gInIndex+11*gInPixels, vl) ;
    _vel_vstu_vssl(vrsumCD, 4, pGIn+gInIndex+12*gInPixels, vl) ;
    _vel_vstl_vssl(vrsumCD, 4, pGIn+gInIndex+13*gInPixels, vl) ;
    _vel_vstu_vssl(vrsumEF, 4, pGIn+gInIndex+14*gInPixels, vl) ;
    _vel_vstl_vssl(vrsumEF, 4, pGIn+gInIndex+15*gInPixels, vl) ;

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
