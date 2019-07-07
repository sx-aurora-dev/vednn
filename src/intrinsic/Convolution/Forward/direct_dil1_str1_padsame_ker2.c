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
    const int64_t kernWidth,
    const int64_t kernHeight,
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

    __vr vrseq = _ve_vseq_v() ;			// xy
    __vr vridx = _ve_vaddsl_vsv(op, vrseq) ;	// op + xy

    __vr vrsum = _ve_vbrdu_vs_f32(0.0f) ;
    __vr vry   = _ve_vdivsl_vvs(vridx, outWidth) ;
    __vr vrx   = _ve_vsubsl_vvv(vridx, _ve_vmulul_vsv(outWidth,vry)) ;

    __vm256 vm_r1 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(inHeight-1,vry)) ;	// condition(y+1 < inHeight)
    __vm256 vm_s1 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(inWidth-1,vrx)) ;	// condition(x+1 < inWidth)

    __vm256 vm_r1s0 = vm_r1 ;
    __vm256 vm_r0s1 = vm_s1 ;
    __vm256 vm_r1s1 = _ve_andm_mmm(vm_r1,vm_s1) ;

    for (int64_t c = 0; c < inChannelGroup; c++) {

      const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;
      const float *pKerValue  = pKernel + kernGroupOffset + ((k * inChannelGroup + c) * kernHeight ) * kernWidth ;

      /* memory access errors mihgt be caused */
      __vr vrin_r0s0 = _ve_vldu_vss(4, pInChannel+op) ;
      __vr vrin_r0s1 = _ve_vldu_vss(4, pInChannel+op+1) ;
      __vr vrin_r1s0 = _ve_vldu_vss(4, pInChannel+op+inWidth) ;
      __vr vrin_r1s1 = _ve_vldu_vss(4, pInChannel+op+inWidth+1) ;

      vrin_r0s1 = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), vrin_r0s1, vm_r0s1) ;
      vrin_r1s0 = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), vrin_r1s0, vm_r1s0) ;
      vrin_r1s1 = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), vrin_r1s1, vm_r1s1) ;


      vrsum = _ve_vfmads_vvsv(vrsum, pKerValue[0], vrin_r0s0) ;
      vrsum = _ve_vfmads_vvsv(vrsum, pKerValue[1], vrin_r0s1) ;
      vrsum = _ve_vfmads_vvsv(vrsum, pKerValue[2], vrin_r1s0) ;
      vrsum = _ve_vfmads_vvsv(vrsum, pKerValue[3], vrin_r1s1) ;
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
    const int64_t kernWidth,
    const int64_t kernHeight,
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
  int64_t outIndex0 = outGroupOffset + (n * outChannel + k  ) * oPixels ;
  int64_t outIndex1 = outGroupOffset + (n * outChannel + k+1) * oPixels ;

  for (int64_t op = 0; op < oPixels; op+=VLEN) {
    const int64_t vl = oPixels - op < VLEN ? oPixels - op : VLEN ;

    _ve_lvl(vl) ;

    __vr vrseq = _ve_vseq_v() ;			// xy
    __vr vridx = _ve_vaddsl_vsv(op, vrseq) ;	// op + xy

    __vr vrsum01 = _ve_pvbrd_vs_i64(0UL) ;

    __vr vry   = _ve_vdivsl_vvs(vridx, outWidth) ;
    __vr vrx   = _ve_vsubsl_vvv(vridx, _ve_vmulul_vsv(outWidth,vry)) ;

    __vm256 vm_r1 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(inHeight-1,vry)) ;	// condition(y+1 < inHeight)
    __vm256 vm_s1 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(inWidth-1,vrx)) ;	// condition(x+1 < inWidth)

    __vm256 vm_r1s0 = vm_r1 ;
    __vm256 vm_r0s1 = vm_s1 ;
    __vm256 vm_r1s1 = _ve_andm_mmm(vm_r1,vm_s1) ;

    for (int64_t c = 0; c < inChannelGroup; c++) {

      const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;
      const float *pKerValue  = pKernel + kernGroupOffset + ((k * inChannelGroup + c) * kernHeight ) * kernWidth ;

      /* memory access errors mihgt be caused */
      __vr vrin_r0s0 = _ve_vldu_vss(4, pInChannel+op) ;
      __vr vrin_r0s1 = _ve_vldu_vss(4, pInChannel+op+1) ;
      __vr vrin_r1s0 = _ve_vldu_vss(4, pInChannel+op+inWidth) ;
      __vr vrin_r1s1 = _ve_vldu_vss(4, pInChannel+op+inWidth+1) ;

      vrin_r0s1 = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), vrin_r0s1, vm_r0s1) ;
      vrin_r1s0 = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), vrin_r1s0, vm_r1s0) ;
      vrin_r1s1 = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), vrin_r1s1, vm_r1s1) ;

      __vr vrinP_r0s0 = _ve_vshf_vvvs(vrin_r0s0, vrin_r0s0, VE_VSHUFFLE_YUZU) ;
      __vr vrinP_r0s1 = _ve_vshf_vvvs(vrin_r0s1, vrin_r0s1, VE_VSHUFFLE_YUZU) ;
      __vr vrinP_r1s0 = _ve_vshf_vvvs(vrin_r1s0, vrin_r1s0, VE_VSHUFFLE_YUZU) ;
      __vr vrinP_r1s1 = _ve_vshf_vvvs(vrin_r1s1, vrin_r1s1, VE_VSHUFFLE_YUZU) ;

      const uint64_t kerValue01_r0s0 = _ve_pack_f32p(pKerValue,
						     pKerValue+      inChannelGroup * kernHeight * kernWidth) ;
      vrsum01 = _ve_pvfmad_vvsv(vrsum01, kerValue01_r0s0, vrinP_r0s0) ;


      const uint64_t kerValue01_r0s1 = _ve_pack_f32p(pKerValue                                               +1,
						     pKerValue+      inChannelGroup * kernHeight * kernWidth +1) ;
      vrsum01 = _ve_pvfmad_vvsv(vrsum01, kerValue01_r0s1, vrinP_r0s1) ;


      const uint64_t kerValue01_r1s0 = _ve_pack_f32p(pKerValue                                               +2,
						     pKerValue+      inChannelGroup * kernHeight * kernWidth +2) ;
      vrsum01 = _ve_pvfmad_vvsv(vrsum01, kerValue01_r1s0, vrinP_r1s0) ;


      const uint64_t kerValue01_r1s1 = _ve_pack_f32p(pKerValue                                               +3,
						     pKerValue+      inChannelGroup * kernHeight * kernWidth +3) ;
      vrsum01 = _ve_pvfmad_vvsv(vrsum01, kerValue01_r1s1, vrinP_r1s1) ;
    } // inChannel

    _ve_vstu_vss(vrsum01, 4, pOut+outIndex0) ;
    _ve_vstl_vss(vrsum01, 4, pOut+outIndex1) ;

    outIndex0 += vl ;
    outIndex1 += vl ;
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
    const int64_t kernWidth,
    const int64_t kernHeight,
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
  int64_t outIndex0 = outGroupOffset + (n * outChannel + k  ) * oPixels ;
  int64_t outIndex1 = outGroupOffset + (n * outChannel + k+1) * oPixels ;
  int64_t outIndex2 = outGroupOffset + (n * outChannel + k+2) * oPixels ;
  int64_t outIndex3 = outGroupOffset + (n * outChannel + k+3) * oPixels ;

  for (int64_t op = 0; op < oPixels; op+=VLEN) {
    const int64_t vl = oPixels - op < VLEN ? oPixels - op : VLEN ;

    _ve_lvl(vl) ;

    __vr vrseq = _ve_vseq_v() ;			// xy
    __vr vridx = _ve_vaddsl_vsv(op, vrseq) ;	// op + xy

    __vr vrsum01 = _ve_pvbrd_vs_i64(0UL) ;
    __vr vrsum23 = _ve_pvbrd_vs_i64(0UL) ;

    __vr vry   = _ve_vdivsl_vvs(vridx, outWidth) ;
    __vr vrx   = _ve_vsubsl_vvv(vridx, _ve_vmulul_vsv(outWidth,vry)) ;

    __vm256 vm_r1 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(inHeight-1,vry)) ;	// condition(y+1 < inHeight)
    __vm256 vm_s1 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(inWidth-1,vrx)) ;	// condition(x+1 < inWidth)

    __vm256 vm_r1s0 = vm_r1 ;
    __vm256 vm_r0s1 = vm_s1 ;
    __vm256 vm_r1s1 = _ve_andm_mmm(vm_r1,vm_s1) ;

    for (int64_t c = 0; c < inChannelGroup; c++) {

      const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;
      const float *pKerValue  = pKernel + kernGroupOffset + ((k * inChannelGroup + c) * kernHeight ) * kernWidth ;

      /* memory access errors mihgt be caused */
      __vr vrin_r0s0 = _ve_vldu_vss(4, pInChannel+op) ;
      __vr vrin_r0s1 = _ve_vldu_vss(4, pInChannel+op+1) ;
      __vr vrin_r1s0 = _ve_vldu_vss(4, pInChannel+op+inWidth) ;
      __vr vrin_r1s1 = _ve_vldu_vss(4, pInChannel+op+inWidth+1) ;

      vrin_r0s1 = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), vrin_r0s1, vm_r0s1) ;
      vrin_r1s0 = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), vrin_r1s0, vm_r1s0) ;
      vrin_r1s1 = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), vrin_r1s1, vm_r1s1) ;

      __vr vrinP_r0s0 = _ve_vshf_vvvs(vrin_r0s0, vrin_r0s0, VE_VSHUFFLE_YUZU) ;
      __vr vrinP_r0s1 = _ve_vshf_vvvs(vrin_r0s1, vrin_r0s1, VE_VSHUFFLE_YUZU) ;
      __vr vrinP_r1s0 = _ve_vshf_vvvs(vrin_r1s0, vrin_r1s0, VE_VSHUFFLE_YUZU) ;
      __vr vrinP_r1s1 = _ve_vshf_vvvs(vrin_r1s1, vrin_r1s1, VE_VSHUFFLE_YUZU) ;

      const uint64_t kerValue01_r0s0 = _ve_pack_f32p(pKerValue,
						     pKerValue+      inChannelGroup * kernHeight * kernWidth) ;
      const uint64_t kerValue23_r0s0 = _ve_pack_f32p(pKerValue + 2 * inChannelGroup * kernHeight * kernWidth,
						     pKerValue + 3 * inChannelGroup * kernHeight * kernWidth) ;
      vrsum01 = _ve_pvfmad_vvsv(vrsum01, kerValue01_r0s0, vrinP_r0s0) ;
      vrsum23 = _ve_pvfmad_vvsv(vrsum23, kerValue23_r0s0, vrinP_r0s0) ;


      const uint64_t kerValue01_r0s1 = _ve_pack_f32p(pKerValue                                               +1,
						     pKerValue+      inChannelGroup * kernHeight * kernWidth +1) ;
      const uint64_t kerValue23_r0s1 = _ve_pack_f32p(pKerValue + 2 * inChannelGroup * kernHeight * kernWidth +1,
						     pKerValue + 3 * inChannelGroup * kernHeight * kernWidth +1) ;
      vrsum01 = _ve_pvfmad_vvsv(vrsum01, kerValue01_r0s1, vrinP_r0s1) ;
      vrsum23 = _ve_pvfmad_vvsv(vrsum23, kerValue23_r0s1, vrinP_r0s1) ;


      const uint64_t kerValue01_r1s0 = _ve_pack_f32p(pKerValue                                               +2,
						     pKerValue+      inChannelGroup * kernHeight * kernWidth +2) ;
      const uint64_t kerValue23_r1s0 = _ve_pack_f32p(pKerValue + 2 * inChannelGroup * kernHeight * kernWidth +2,
						     pKerValue + 3 * inChannelGroup * kernHeight * kernWidth +2) ;
      vrsum01 = _ve_pvfmad_vvsv(vrsum01, kerValue01_r1s0, vrinP_r1s0) ;
      vrsum23 = _ve_pvfmad_vvsv(vrsum23, kerValue23_r1s0, vrinP_r1s0) ;


      const uint64_t kerValue01_r1s1 = _ve_pack_f32p(pKerValue                                               +3,
						     pKerValue+      inChannelGroup * kernHeight * kernWidth +3) ;
      const uint64_t kerValue23_r1s1 = _ve_pack_f32p(pKerValue + 2 * inChannelGroup * kernHeight * kernWidth +3,
						     pKerValue + 3 * inChannelGroup * kernHeight * kernWidth +3) ;
      vrsum01 = _ve_pvfmad_vvsv(vrsum01, kerValue01_r1s1, vrinP_r1s1) ;
      vrsum23 = _ve_pvfmad_vvsv(vrsum23, kerValue23_r1s1, vrinP_r1s1) ;
    } // inChannel

    _ve_vstu_vss(vrsum01, 4, pOut+outIndex0) ;
    _ve_vstl_vss(vrsum01, 4, pOut+outIndex1) ;
    _ve_vstu_vss(vrsum23, 4, pOut+outIndex2) ;
    _ve_vstl_vss(vrsum23, 4, pOut+outIndex3) ;

    outIndex0 += vl ;
    outIndex1 += vl ;
    outIndex2 += vl ;
    outIndex3 += vl ;
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
    const int64_t kernWidth,
    const int64_t kernHeight,
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
  int64_t outIndex0 = outGroupOffset + (n * outChannel + k  ) * oPixels ;
  int64_t outIndex1 = outGroupOffset + (n * outChannel + k+1) * oPixels ;
  int64_t outIndex2 = outGroupOffset + (n * outChannel + k+2) * oPixels ;
  int64_t outIndex3 = outGroupOffset + (n * outChannel + k+3) * oPixels ;
  int64_t outIndex4 = outGroupOffset + (n * outChannel + k+4) * oPixels ;
  int64_t outIndex5 = outGroupOffset + (n * outChannel + k+5) * oPixels ;
  int64_t outIndex6 = outGroupOffset + (n * outChannel + k+6) * oPixels ;
  int64_t outIndex7 = outGroupOffset + (n * outChannel + k+7) * oPixels ;

  for (int64_t op = 0; op < oPixels; op+=VLEN) {
    const int64_t vl = oPixels - op < VLEN ? oPixels - op : VLEN ;

    _ve_lvl(vl) ;

    __vr vrseq = _ve_vseq_v() ;			// xy
    __vr vridx = _ve_vaddsl_vsv(op, vrseq) ;	// op + xy

    __vr vrsum01 = _ve_pvbrd_vs_i64(0UL) ;
    __vr vrsum23 = _ve_pvbrd_vs_i64(0UL) ;
    __vr vrsum45 = _ve_pvbrd_vs_i64(0UL) ;
    __vr vrsum67 = _ve_pvbrd_vs_i64(0UL) ;

    __vr vry   = _ve_vdivsl_vvs(vridx, outWidth) ;
    __vr vrx   = _ve_vsubsl_vvv(vridx, _ve_vmulul_vsv(outWidth,vry)) ;

    __vm256 vm_r1 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(inHeight-1,vry)) ;	// condition(y+1 < inHeight)
    __vm256 vm_s1 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(inWidth-1,vrx)) ;	// condition(x+1 < inWidth)

    __vm256 vm_r1s0 = vm_r1 ;
    __vm256 vm_r0s1 = vm_s1 ;
    __vm256 vm_r1s1 = _ve_andm_mmm(vm_r1,vm_s1) ;

    for (int64_t c = 0; c < inChannelGroup; c++) {

      const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;
      const float *pKerValue  = pKernel + kernGroupOffset + ((k * inChannelGroup + c) * kernHeight ) * kernWidth ;

      /* memory access errors mihgt be caused */
      __vr vrin_r0s0 = _ve_vldu_vss(4, pInChannel+op) ;
      __vr vrin_r0s1 = _ve_vldu_vss(4, pInChannel+op+1) ;
      __vr vrin_r1s0 = _ve_vldu_vss(4, pInChannel+op+inWidth) ;
      __vr vrin_r1s1 = _ve_vldu_vss(4, pInChannel+op+inWidth+1) ;

      vrin_r0s1 = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), vrin_r0s1, vm_r0s1) ;
      vrin_r1s0 = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), vrin_r1s0, vm_r1s0) ;
      vrin_r1s1 = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), vrin_r1s1, vm_r1s1) ;

      __vr vrinP_r0s0 = _ve_vshf_vvvs(vrin_r0s0, vrin_r0s0, VE_VSHUFFLE_YUZU) ;
      __vr vrinP_r0s1 = _ve_vshf_vvvs(vrin_r0s1, vrin_r0s1, VE_VSHUFFLE_YUZU) ;
      __vr vrinP_r1s0 = _ve_vshf_vvvs(vrin_r1s0, vrin_r1s0, VE_VSHUFFLE_YUZU) ;
      __vr vrinP_r1s1 = _ve_vshf_vvvs(vrin_r1s1, vrin_r1s1, VE_VSHUFFLE_YUZU) ;

      const uint64_t kerValue01_r0s0 = _ve_pack_f32p(pKerValue,
						     pKerValue+      inChannelGroup * kernHeight * kernWidth) ;
      const uint64_t kerValue23_r0s0 = _ve_pack_f32p(pKerValue + 2 * inChannelGroup * kernHeight * kernWidth,
						     pKerValue + 3 * inChannelGroup * kernHeight * kernWidth) ;
      const uint64_t kerValue45_r0s0 = _ve_pack_f32p(pKerValue + 4 * inChannelGroup * kernHeight * kernWidth,
						     pKerValue + 5 * inChannelGroup * kernHeight * kernWidth) ;
      const uint64_t kerValue67_r0s0 = _ve_pack_f32p(pKerValue + 6 * inChannelGroup * kernHeight * kernWidth,
						     pKerValue + 7 * inChannelGroup * kernHeight * kernWidth) ;
      vrsum01 = _ve_pvfmad_vvsv(vrsum01, kerValue01_r0s0, vrinP_r0s0) ;
      vrsum23 = _ve_pvfmad_vvsv(vrsum23, kerValue23_r0s0, vrinP_r0s0) ;
      vrsum45 = _ve_pvfmad_vvsv(vrsum45, kerValue45_r0s0, vrinP_r0s0) ;
      vrsum67 = _ve_pvfmad_vvsv(vrsum67, kerValue67_r0s0, vrinP_r0s0) ;


      const uint64_t kerValue01_r0s1 = _ve_pack_f32p(pKerValue                                               +1,
						     pKerValue+      inChannelGroup * kernHeight * kernWidth +1) ;
      const uint64_t kerValue23_r0s1 = _ve_pack_f32p(pKerValue + 2 * inChannelGroup * kernHeight * kernWidth +1,
						     pKerValue + 3 * inChannelGroup * kernHeight * kernWidth +1) ;
      const uint64_t kerValue45_r0s1 = _ve_pack_f32p(pKerValue + 4 * inChannelGroup * kernHeight * kernWidth +1,
						     pKerValue + 5 * inChannelGroup * kernHeight * kernWidth +1) ;
      const uint64_t kerValue67_r0s1 = _ve_pack_f32p(pKerValue + 6 * inChannelGroup * kernHeight * kernWidth +1,
						     pKerValue + 7 * inChannelGroup * kernHeight * kernWidth +1) ;
      vrsum01 = _ve_pvfmad_vvsv(vrsum01, kerValue01_r0s1, vrinP_r0s1) ;
      vrsum23 = _ve_pvfmad_vvsv(vrsum23, kerValue23_r0s1, vrinP_r0s1) ;
      vrsum45 = _ve_pvfmad_vvsv(vrsum45, kerValue45_r0s1, vrinP_r0s1) ;
      vrsum67 = _ve_pvfmad_vvsv(vrsum67, kerValue67_r0s1, vrinP_r0s1) ;


      const uint64_t kerValue01_r1s0 = _ve_pack_f32p(pKerValue                                               +2,
						     pKerValue+      inChannelGroup * kernHeight * kernWidth +2) ;
      const uint64_t kerValue23_r1s0 = _ve_pack_f32p(pKerValue + 2 * inChannelGroup * kernHeight * kernWidth +2,
						     pKerValue + 3 * inChannelGroup * kernHeight * kernWidth +2) ;
      const uint64_t kerValue45_r1s0 = _ve_pack_f32p(pKerValue + 4 * inChannelGroup * kernHeight * kernWidth +2,
						     pKerValue + 5 * inChannelGroup * kernHeight * kernWidth +2) ;
      const uint64_t kerValue67_r1s0 = _ve_pack_f32p(pKerValue + 6 * inChannelGroup * kernHeight * kernWidth +2,
						     pKerValue + 7 * inChannelGroup * kernHeight * kernWidth +2) ;
      vrsum01 = _ve_pvfmad_vvsv(vrsum01, kerValue01_r1s0, vrinP_r1s0) ;
      vrsum23 = _ve_pvfmad_vvsv(vrsum23, kerValue23_r1s0, vrinP_r1s0) ;
      vrsum45 = _ve_pvfmad_vvsv(vrsum45, kerValue45_r1s0, vrinP_r1s0) ;
      vrsum67 = _ve_pvfmad_vvsv(vrsum67, kerValue67_r1s0, vrinP_r1s0) ;


      const uint64_t kerValue01_r1s1 = _ve_pack_f32p(pKerValue                                               +3,
						     pKerValue+      inChannelGroup * kernHeight * kernWidth +3) ;
      const uint64_t kerValue23_r1s1 = _ve_pack_f32p(pKerValue + 2 * inChannelGroup * kernHeight * kernWidth +3,
						     pKerValue + 3 * inChannelGroup * kernHeight * kernWidth +3) ;
      const uint64_t kerValue45_r1s1 = _ve_pack_f32p(pKerValue + 4 * inChannelGroup * kernHeight * kernWidth +3,
						     pKerValue + 5 * inChannelGroup * kernHeight * kernWidth +3) ;
      const uint64_t kerValue67_r1s1 = _ve_pack_f32p(pKerValue + 6 * inChannelGroup * kernHeight * kernWidth +3,
						     pKerValue + 7 * inChannelGroup * kernHeight * kernWidth +3) ;
      vrsum01 = _ve_pvfmad_vvsv(vrsum01, kerValue01_r1s1, vrinP_r1s1) ;
      vrsum23 = _ve_pvfmad_vvsv(vrsum23, kerValue23_r1s1, vrinP_r1s1) ;
      vrsum45 = _ve_pvfmad_vvsv(vrsum45, kerValue45_r1s1, vrinP_r1s1) ;
      vrsum67 = _ve_pvfmad_vvsv(vrsum67, kerValue67_r1s1, vrinP_r1s1) ;
    } // inChannel

    _ve_vstu_vss(vrsum01, 4, pOut+outIndex0) ;
    _ve_vstl_vss(vrsum01, 4, pOut+outIndex1) ;
    _ve_vstu_vss(vrsum23, 4, pOut+outIndex2) ;
    _ve_vstl_vss(vrsum23, 4, pOut+outIndex3) ;
    _ve_vstu_vss(vrsum45, 4, pOut+outIndex4) ;
    _ve_vstl_vss(vrsum45, 4, pOut+outIndex5) ;
    _ve_vstu_vss(vrsum67, 4, pOut+outIndex6) ;
    _ve_vstl_vss(vrsum67, 4, pOut+outIndex7) ;

    outIndex0 += vl ;
    outIndex1 += vl ;
    outIndex2 += vl ;
    outIndex3 += vl ;
    outIndex4 += vl ;
    outIndex5 += vl ;
    outIndex6 += vl ;
    outIndex7 += vl ;
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
    const int64_t kernWidth,
    const int64_t kernHeight,
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
  int64_t outIndex0 = outGroupOffset + (n * outChannel + k  ) * oPixels ;
  int64_t outIndex1 = outGroupOffset + (n * outChannel + k+1) * oPixels ;
  int64_t outIndex2 = outGroupOffset + (n * outChannel + k+2) * oPixels ;
  int64_t outIndex3 = outGroupOffset + (n * outChannel + k+3) * oPixels ;
  int64_t outIndex4 = outGroupOffset + (n * outChannel + k+4) * oPixels ;
  int64_t outIndex5 = outGroupOffset + (n * outChannel + k+5) * oPixels ;
  int64_t outIndex6 = outGroupOffset + (n * outChannel + k+6) * oPixels ;
  int64_t outIndex7 = outGroupOffset + (n * outChannel + k+7) * oPixels ;
  int64_t outIndex8 = outGroupOffset + (n * outChannel + k+8) * oPixels ;
  int64_t outIndex9 = outGroupOffset + (n * outChannel + k+9) * oPixels ;
  int64_t outIndexA = outGroupOffset + (n * outChannel + k+10) * oPixels ;
  int64_t outIndexB = outGroupOffset + (n * outChannel + k+11) * oPixels ;
  int64_t outIndexC = outGroupOffset + (n * outChannel + k+12) * oPixels ;
  int64_t outIndexD = outGroupOffset + (n * outChannel + k+13) * oPixels ;
  int64_t outIndexE = outGroupOffset + (n * outChannel + k+14) * oPixels ;
  int64_t outIndexF = outGroupOffset + (n * outChannel + k+15) * oPixels ;

  for (int64_t op = 0; op < oPixels; op+=VLEN) {
    const int64_t vl = oPixels - op < VLEN ? oPixels - op : VLEN ;

    _ve_lvl(vl) ;

    __vr vrseq = _ve_vseq_v() ;			// xy
    __vr vridx = _ve_vaddsl_vsv(op, vrseq) ;	// op + xy

    __vr vrsum01 = _ve_pvbrd_vs_i64(0UL) ;
    __vr vrsum23 = _ve_pvbrd_vs_i64(0UL) ;
    __vr vrsum45 = _ve_pvbrd_vs_i64(0UL) ;
    __vr vrsum67 = _ve_pvbrd_vs_i64(0UL) ;
    __vr vrsum89 = _ve_pvbrd_vs_i64(0UL) ;
    __vr vrsumAB = _ve_pvbrd_vs_i64(0UL) ;
    __vr vrsumCD = _ve_pvbrd_vs_i64(0UL) ;
    __vr vrsumEF = _ve_pvbrd_vs_i64(0UL) ;

    __vr vry   = _ve_vdivsl_vvs(vridx, outWidth) ;
    __vr vrx   = _ve_vsubsl_vvv(vridx, _ve_vmulul_vsv(outWidth,vry)) ;

    __vm256 vm_r1 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(inHeight-1,vry)) ;	// condition(y+1 < inHeight)
    __vm256 vm_s1 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(inWidth-1,vrx)) ;	// condition(x+1 < inWidth)

    __vm256 vm_r1s0 = vm_r1 ;
    __vm256 vm_r0s1 = vm_s1 ;
    __vm256 vm_r1s1 = _ve_andm_mmm(vm_r1,vm_s1) ;

    for (int64_t c = 0; c < inChannelGroup; c++) {

      const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;
      const float *pKerValue  = pKernel + kernGroupOffset + ((k * inChannelGroup + c) * kernHeight ) * kernWidth ;

      /* memory access errors mihgt be caused */
      __vr vrin_r0s0 = _ve_vldu_vss(4, pInChannel+op) ;
      __vr vrin_r0s1 = _ve_vldu_vss(4, pInChannel+op+1) ;
      __vr vrin_r1s0 = _ve_vldu_vss(4, pInChannel+op+inWidth) ;
      __vr vrin_r1s1 = _ve_vldu_vss(4, pInChannel+op+inWidth+1) ;

      vrin_r0s1 = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), vrin_r0s1, vm_r0s1) ;
      vrin_r1s0 = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), vrin_r1s0, vm_r1s0) ;
      vrin_r1s1 = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), vrin_r1s1, vm_r1s1) ;

      __vr vrinP_r0s0 = _ve_vshf_vvvs(vrin_r0s0, vrin_r0s0, VE_VSHUFFLE_YUZU) ;
      __vr vrinP_r0s1 = _ve_vshf_vvvs(vrin_r0s1, vrin_r0s1, VE_VSHUFFLE_YUZU) ;
      __vr vrinP_r1s0 = _ve_vshf_vvvs(vrin_r1s0, vrin_r1s0, VE_VSHUFFLE_YUZU) ;
      __vr vrinP_r1s1 = _ve_vshf_vvvs(vrin_r1s1, vrin_r1s1, VE_VSHUFFLE_YUZU) ;

      const uint64_t kerValue01_r0s0 = _ve_pack_f32p(pKerValue,
						     pKerValue+      inChannelGroup * kernHeight * kernWidth) ;
      const uint64_t kerValue23_r0s0 = _ve_pack_f32p(pKerValue + 2 * inChannelGroup * kernHeight * kernWidth,
						     pKerValue + 3 * inChannelGroup * kernHeight * kernWidth) ;
      const uint64_t kerValue45_r0s0 = _ve_pack_f32p(pKerValue + 4 * inChannelGroup * kernHeight * kernWidth,
						     pKerValue + 5 * inChannelGroup * kernHeight * kernWidth) ;
      const uint64_t kerValue67_r0s0 = _ve_pack_f32p(pKerValue + 6 * inChannelGroup * kernHeight * kernWidth,
						     pKerValue + 7 * inChannelGroup * kernHeight * kernWidth) ;
      const uint64_t kerValue89_r0s0 = _ve_pack_f32p(pKerValue + 8 * inChannelGroup * kernHeight * kernWidth,
						     pKerValue + 9 * inChannelGroup * kernHeight * kernWidth) ;
      const uint64_t kerValueAB_r0s0 = _ve_pack_f32p(pKerValue +10 * inChannelGroup * kernHeight * kernWidth,
						     pKerValue +11 * inChannelGroup * kernHeight * kernWidth) ;
      const uint64_t kerValueCD_r0s0 = _ve_pack_f32p(pKerValue +12 * inChannelGroup * kernHeight * kernWidth,
						     pKerValue +13 * inChannelGroup * kernHeight * kernWidth) ;
      const uint64_t kerValueEF_r0s0 = _ve_pack_f32p(pKerValue +14 * inChannelGroup * kernHeight * kernWidth,
						     pKerValue +15 * inChannelGroup * kernHeight * kernWidth) ;
      vrsum01 = _ve_pvfmad_vvsv(vrsum01, kerValue01_r0s0, vrinP_r0s0) ;
      vrsum23 = _ve_pvfmad_vvsv(vrsum23, kerValue23_r0s0, vrinP_r0s0) ;
      vrsum45 = _ve_pvfmad_vvsv(vrsum45, kerValue45_r0s0, vrinP_r0s0) ;
      vrsum67 = _ve_pvfmad_vvsv(vrsum67, kerValue67_r0s0, vrinP_r0s0) ;
      vrsum89 = _ve_pvfmad_vvsv(vrsum89, kerValue89_r0s0, vrinP_r0s0) ;
      vrsumAB = _ve_pvfmad_vvsv(vrsumAB, kerValueAB_r0s0, vrinP_r0s0) ;
      vrsumCD = _ve_pvfmad_vvsv(vrsumCD, kerValueCD_r0s0, vrinP_r0s0) ;
      vrsumEF = _ve_pvfmad_vvsv(vrsumEF, kerValueEF_r0s0, vrinP_r0s0) ;


      const uint64_t kerValue01_r0s1 = _ve_pack_f32p(pKerValue                                               +1,
						     pKerValue+      inChannelGroup * kernHeight * kernWidth +1) ;
      const uint64_t kerValue23_r0s1 = _ve_pack_f32p(pKerValue + 2 * inChannelGroup * kernHeight * kernWidth +1,
						     pKerValue + 3 * inChannelGroup * kernHeight * kernWidth +1) ;
      const uint64_t kerValue45_r0s1 = _ve_pack_f32p(pKerValue + 4 * inChannelGroup * kernHeight * kernWidth +1,
						     pKerValue + 5 * inChannelGroup * kernHeight * kernWidth +1) ;
      const uint64_t kerValue67_r0s1 = _ve_pack_f32p(pKerValue + 6 * inChannelGroup * kernHeight * kernWidth +1,
						     pKerValue + 7 * inChannelGroup * kernHeight * kernWidth +1) ;
      const uint64_t kerValue89_r0s1 = _ve_pack_f32p(pKerValue + 8 * inChannelGroup * kernHeight * kernWidth +1,
						     pKerValue + 9 * inChannelGroup * kernHeight * kernWidth +1) ;
      const uint64_t kerValueAB_r0s1 = _ve_pack_f32p(pKerValue +10 * inChannelGroup * kernHeight * kernWidth +1,
						     pKerValue +11 * inChannelGroup * kernHeight * kernWidth +1) ;
      const uint64_t kerValueCD_r0s1 = _ve_pack_f32p(pKerValue +12 * inChannelGroup * kernHeight * kernWidth +1,
						     pKerValue +13 * inChannelGroup * kernHeight * kernWidth +1) ;
      const uint64_t kerValueEF_r0s1 = _ve_pack_f32p(pKerValue +14 * inChannelGroup * kernHeight * kernWidth +1,
						     pKerValue +15 * inChannelGroup * kernHeight * kernWidth +1) ;
      vrsum01 = _ve_pvfmad_vvsv(vrsum01, kerValue01_r0s1, vrinP_r0s1) ;
      vrsum23 = _ve_pvfmad_vvsv(vrsum23, kerValue23_r0s1, vrinP_r0s1) ;
      vrsum45 = _ve_pvfmad_vvsv(vrsum45, kerValue45_r0s1, vrinP_r0s1) ;
      vrsum67 = _ve_pvfmad_vvsv(vrsum67, kerValue67_r0s1, vrinP_r0s1) ;
      vrsum89 = _ve_pvfmad_vvsv(vrsum89, kerValue89_r0s1, vrinP_r0s1) ;
      vrsumAB = _ve_pvfmad_vvsv(vrsumAB, kerValueAB_r0s1, vrinP_r0s1) ;
      vrsumCD = _ve_pvfmad_vvsv(vrsumCD, kerValueCD_r0s1, vrinP_r0s1) ;
      vrsumEF = _ve_pvfmad_vvsv(vrsumEF, kerValueEF_r0s1, vrinP_r0s1) ;


      const uint64_t kerValue01_r1s0 = _ve_pack_f32p(pKerValue                                               +2,
						     pKerValue+      inChannelGroup * kernHeight * kernWidth +2) ;
      const uint64_t kerValue23_r1s0 = _ve_pack_f32p(pKerValue + 2 * inChannelGroup * kernHeight * kernWidth +2,
						     pKerValue + 3 * inChannelGroup * kernHeight * kernWidth +2) ;
      const uint64_t kerValue45_r1s0 = _ve_pack_f32p(pKerValue + 4 * inChannelGroup * kernHeight * kernWidth +2,
						     pKerValue + 5 * inChannelGroup * kernHeight * kernWidth +2) ;
      const uint64_t kerValue67_r1s0 = _ve_pack_f32p(pKerValue + 6 * inChannelGroup * kernHeight * kernWidth +2,
						     pKerValue + 7 * inChannelGroup * kernHeight * kernWidth +2) ;
      const uint64_t kerValue89_r1s0 = _ve_pack_f32p(pKerValue + 8 * inChannelGroup * kernHeight * kernWidth +2,
						     pKerValue + 9 * inChannelGroup * kernHeight * kernWidth +2) ;
      const uint64_t kerValueAB_r1s0 = _ve_pack_f32p(pKerValue +10 * inChannelGroup * kernHeight * kernWidth +2,
						     pKerValue +11 * inChannelGroup * kernHeight * kernWidth +2) ;
      const uint64_t kerValueCD_r1s0 = _ve_pack_f32p(pKerValue +12 * inChannelGroup * kernHeight * kernWidth +2,
						     pKerValue +13 * inChannelGroup * kernHeight * kernWidth +2) ;
      const uint64_t kerValueEF_r1s0 = _ve_pack_f32p(pKerValue +14 * inChannelGroup * kernHeight * kernWidth +2,
						     pKerValue +15 * inChannelGroup * kernHeight * kernWidth +2) ;
      vrsum01 = _ve_pvfmad_vvsv(vrsum01, kerValue01_r1s0, vrinP_r1s0) ;
      vrsum23 = _ve_pvfmad_vvsv(vrsum23, kerValue23_r1s0, vrinP_r1s0) ;
      vrsum45 = _ve_pvfmad_vvsv(vrsum45, kerValue45_r1s0, vrinP_r1s0) ;
      vrsum67 = _ve_pvfmad_vvsv(vrsum67, kerValue67_r1s0, vrinP_r1s0) ;
      vrsum89 = _ve_pvfmad_vvsv(vrsum89, kerValue89_r1s0, vrinP_r1s0) ;
      vrsumAB = _ve_pvfmad_vvsv(vrsumAB, kerValueAB_r1s0, vrinP_r1s0) ;
      vrsumCD = _ve_pvfmad_vvsv(vrsumCD, kerValueCD_r1s0, vrinP_r1s0) ;
      vrsumEF = _ve_pvfmad_vvsv(vrsumEF, kerValueEF_r1s0, vrinP_r1s0) ;


      const uint64_t kerValue01_r1s1 = _ve_pack_f32p(pKerValue                                               +3,
						     pKerValue+      inChannelGroup * kernHeight * kernWidth +3) ;
      const uint64_t kerValue23_r1s1 = _ve_pack_f32p(pKerValue + 2 * inChannelGroup * kernHeight * kernWidth +3,
						     pKerValue + 3 * inChannelGroup * kernHeight * kernWidth +3) ;
      const uint64_t kerValue45_r1s1 = _ve_pack_f32p(pKerValue + 4 * inChannelGroup * kernHeight * kernWidth +3,
						     pKerValue + 5 * inChannelGroup * kernHeight * kernWidth +3) ;
      const uint64_t kerValue67_r1s1 = _ve_pack_f32p(pKerValue + 6 * inChannelGroup * kernHeight * kernWidth +3,
						     pKerValue + 7 * inChannelGroup * kernHeight * kernWidth +3) ;
      const uint64_t kerValue89_r1s1 = _ve_pack_f32p(pKerValue + 8 * inChannelGroup * kernHeight * kernWidth +3,
						     pKerValue + 9 * inChannelGroup * kernHeight * kernWidth +3) ;
      const uint64_t kerValueAB_r1s1 = _ve_pack_f32p(pKerValue +10 * inChannelGroup * kernHeight * kernWidth +3,
						     pKerValue +11 * inChannelGroup * kernHeight * kernWidth +3) ;
      const uint64_t kerValueCD_r1s1 = _ve_pack_f32p(pKerValue +12 * inChannelGroup * kernHeight * kernWidth +3,
						     pKerValue +13 * inChannelGroup * kernHeight * kernWidth +3) ;
      const uint64_t kerValueEF_r1s1 = _ve_pack_f32p(pKerValue +14 * inChannelGroup * kernHeight * kernWidth +3,
						     pKerValue +15 * inChannelGroup * kernHeight * kernWidth +3) ;
      vrsum01 = _ve_pvfmad_vvsv(vrsum01, kerValue01_r1s1, vrinP_r1s1) ;
      vrsum23 = _ve_pvfmad_vvsv(vrsum23, kerValue23_r1s1, vrinP_r1s1) ;
      vrsum45 = _ve_pvfmad_vvsv(vrsum45, kerValue45_r1s1, vrinP_r1s1) ;
      vrsum67 = _ve_pvfmad_vvsv(vrsum67, kerValue67_r1s1, vrinP_r1s1) ;
      vrsum89 = _ve_pvfmad_vvsv(vrsum89, kerValue89_r1s1, vrinP_r1s1) ;
      vrsumAB = _ve_pvfmad_vvsv(vrsumAB, kerValueAB_r1s1, vrinP_r1s1) ;
      vrsumCD = _ve_pvfmad_vvsv(vrsumCD, kerValueCD_r1s1, vrinP_r1s1) ;
      vrsumEF = _ve_pvfmad_vvsv(vrsumEF, kerValueEF_r1s1, vrinP_r1s1) ;
    } // inChannel

    _ve_vstu_vss(vrsum01, 4, pOut+outIndex0) ;
    _ve_vstl_vss(vrsum01, 4, pOut+outIndex1) ;
    _ve_vstu_vss(vrsum23, 4, pOut+outIndex2) ;
    _ve_vstl_vss(vrsum23, 4, pOut+outIndex3) ;
    _ve_vstu_vss(vrsum45, 4, pOut+outIndex4) ;
    _ve_vstl_vss(vrsum45, 4, pOut+outIndex5) ;
    _ve_vstu_vss(vrsum67, 4, pOut+outIndex6) ;
    _ve_vstl_vss(vrsum67, 4, pOut+outIndex7) ;
    _ve_vstu_vss(vrsum89, 4, pOut+outIndex8) ;
    _ve_vstl_vss(vrsum89, 4, pOut+outIndex9) ;
    _ve_vstu_vss(vrsumAB, 4, pOut+outIndexA) ;
    _ve_vstl_vss(vrsumAB, 4, pOut+outIndexB) ;
    _ve_vstu_vss(vrsumCD, 4, pOut+outIndexC) ;
    _ve_vstl_vss(vrsumCD, 4, pOut+outIndexD) ;
    _ve_vstu_vss(vrsumEF, 4, pOut+outIndexE) ;
    _ve_vstl_vss(vrsumEF, 4, pOut+outIndexF) ;

    outIndex0 += vl ;
    outIndex1 += vl ;
    outIndex2 += vl ;
    outIndex3 += vl ;
    outIndex4 += vl ;
    outIndex5 += vl ;
    outIndex6 += vl ;
    outIndex7 += vl ;
    outIndex8 += vl ;
    outIndex9 += vl ;
    outIndexA += vl ;
    outIndexB += vl ;
    outIndexC += vl ;
    outIndexD += vl ;
    outIndexE += vl ;
    outIndexF += vl ;
  } // outPixels
}

static inline void k8_c512x(
    const float * restrict pIn,
    const float * restrict pKernel,
    float * restrict const pOut,
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
    const int64_t inGroupOffset,
    const int64_t outGroupOffset,
    const int64_t kernGroupOffset,
    const int64_t oPixels,
    const int64_t n,
    const int64_t k,
    float * restrict const filter
)
{
  int64_t outIndex0 = outGroupOffset + (n * outChannel + k  ) * oPixels ;
  int64_t outIndex1 = outGroupOffset + (n * outChannel + k+1) * oPixels ;
  int64_t outIndex2 = outGroupOffset + (n * outChannel + k+2) * oPixels ;
  int64_t outIndex3 = outGroupOffset + (n * outChannel + k+3) * oPixels ;
  int64_t outIndex4 = outGroupOffset + (n * outChannel + k+4) * oPixels ;
  int64_t outIndex5 = outGroupOffset + (n * outChannel + k+5) * oPixels ;
  int64_t outIndex6 = outGroupOffset + (n * outChannel + k+6) * oPixels ;
  int64_t outIndex7 = outGroupOffset + (n * outChannel + k+7) * oPixels ;

  for (int64_t op = 0; op < oPixels; op+=VLEN) {
    const int64_t vl = oPixels - op < VLEN ? oPixels - op : VLEN ;

    _ve_lvl(vl) ;

    __vr vrseq = _ve_vseq_v() ;			// xy
    __vr vridx = _ve_vaddsl_vsv(op, vrseq) ;	// op + xy

    __vr vrsum01 = _ve_pvbrd_vs_i64(0UL) ;
    __vr vrsum23 = _ve_pvbrd_vs_i64(0UL) ;
    __vr vrsum45 = _ve_pvbrd_vs_i64(0UL) ;
    __vr vrsum67 = _ve_pvbrd_vs_i64(0UL) ;

    __vr vry   = _ve_vdivsl_vvs(vridx, outWidth) ;
    __vr vrx   = _ve_vsubsl_vvv(vridx, _ve_vmulul_vsv(outWidth,vry)) ;

    __vm256 vm_r1 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(inHeight-1,vry)) ;	// condition(y+1 < inHeight)
    __vm256 vm_s1 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(inWidth-1,vrx)) ;	// condition(x+1 < inWidth)

    __vm256 vm_r1s0 = vm_r1 ;
    __vm256 vm_r0s1 = vm_s1 ;
    __vm256 vm_r1s1 = _ve_andm_mmm(vm_r1,vm_s1) ;

    for (int64_t c0 = 0; c0 < inChannelGroup; c0+=256) {
      const float *pKerValue  = pKernel + kernGroupOffset + ((k * inChannelGroup + c0) * kernHeight ) * kernWidth ;
      {
	_ve_lvl(256) ;

	__vr vr0_r0 = _ve_vld_vss(4*4, pKerValue +0*inChannelGroup*4) ;
	__vr vr1_r0 = _ve_vld_vss(4*4, pKerValue +1*inChannelGroup*4) ;
	__vr vr2_r0 = _ve_vld_vss(4*4, pKerValue +2*inChannelGroup*4) ;
	__vr vr3_r0 = _ve_vld_vss(4*4, pKerValue +3*inChannelGroup*4) ;
	__vr vr4_r0 = _ve_vld_vss(4*4, pKerValue +4*inChannelGroup*4) ;
	__vr vr5_r0 = _ve_vld_vss(4*4, pKerValue +5*inChannelGroup*4) ;
	__vr vr6_r0 = _ve_vld_vss(4*4, pKerValue +6*inChannelGroup*4) ;
	__vr vr7_r0 = _ve_vld_vss(4*4, pKerValue +7*inChannelGroup*4) ;

	__vr vr0_r1 = _ve_vld_vss(4*4, pKerValue +0*inChannelGroup*4+2) ;
	__vr vr1_r1 = _ve_vld_vss(4*4, pKerValue +1*inChannelGroup*4+2) ;
	__vr vr2_r1 = _ve_vld_vss(4*4, pKerValue +2*inChannelGroup*4+2) ;
	__vr vr3_r1 = _ve_vld_vss(4*4, pKerValue +3*inChannelGroup*4+2) ;
	__vr vr4_r1 = _ve_vld_vss(4*4, pKerValue +4*inChannelGroup*4+2) ;
	__vr vr5_r1 = _ve_vld_vss(4*4, pKerValue +5*inChannelGroup*4+2) ;
	__vr vr6_r1 = _ve_vld_vss(4*4, pKerValue +6*inChannelGroup*4+2) ;
	__vr vr7_r1 = _ve_vld_vss(4*4, pKerValue +7*inChannelGroup*4+2) ;

	__vr vr01_r0s0 = _ve_vshf_vvvs(vr0_r0,vr1_r0,VE_VSHUFFLE_YLZL) ;
	__vr vr23_r0s0 = _ve_vshf_vvvs(vr2_r0,vr3_r0,VE_VSHUFFLE_YLZL) ;
	__vr vr45_r0s0 = _ve_vshf_vvvs(vr4_r0,vr5_r0,VE_VSHUFFLE_YLZL) ;
	__vr vr67_r0s0 = _ve_vshf_vvvs(vr6_r0,vr7_r0,VE_VSHUFFLE_YLZL) ;
	_ve_vst_vss(vr01_r0s0, 4*4*8, filter) ;
	_ve_vst_vss(vr23_r0s0, 4*4*8, filter+2) ;
	_ve_vst_vss(vr45_r0s0, 4*4*8, filter+4) ;
	_ve_vst_vss(vr67_r0s0, 4*4*8, filter+6) ;

	__vr vr01_r0s1 = _ve_vshf_vvvs(vr0_r0,vr1_r0,VE_VSHUFFLE_YUZU) ;
	__vr vr23_r0s1 = _ve_vshf_vvvs(vr2_r0,vr3_r0,VE_VSHUFFLE_YUZU) ;
	__vr vr45_r0s1 = _ve_vshf_vvvs(vr4_r0,vr5_r0,VE_VSHUFFLE_YUZU) ;
	__vr vr67_r0s1 = _ve_vshf_vvvs(vr6_r0,vr7_r0,VE_VSHUFFLE_YUZU) ;
	_ve_vst_vss(vr01_r0s1, 4*4*8, filter+8+0) ;
	_ve_vst_vss(vr23_r0s1, 4*4*8, filter+8+2) ;
	_ve_vst_vss(vr45_r0s1, 4*4*8, filter+8+4) ;
	_ve_vst_vss(vr67_r0s1, 4*4*8, filter+8+6) ;

	__vr vr01_r1s0 = _ve_vshf_vvvs(vr0_r1,vr1_r1,VE_VSHUFFLE_YLZL) ;
	__vr vr23_r1s0 = _ve_vshf_vvvs(vr2_r1,vr3_r1,VE_VSHUFFLE_YLZL) ;
	__vr vr45_r1s0 = _ve_vshf_vvvs(vr4_r1,vr5_r1,VE_VSHUFFLE_YLZL) ;
	__vr vr67_r1s0 = _ve_vshf_vvvs(vr6_r1,vr7_r1,VE_VSHUFFLE_YLZL) ;
	_ve_vst_vss(vr01_r1s0, 4*4*8, filter+2*8+0) ;
	_ve_vst_vss(vr23_r1s0, 4*4*8, filter+2*8+2) ;
	_ve_vst_vss(vr45_r1s0, 4*4*8, filter+2*8+4) ;
	_ve_vst_vss(vr67_r1s0, 4*4*8, filter+2*8+6) ;

	__vr vr01_r1s1 = _ve_vshf_vvvs(vr0_r1,vr1_r1,VE_VSHUFFLE_YUZU) ;
	__vr vr23_r1s1 = _ve_vshf_vvvs(vr2_r1,vr3_r1,VE_VSHUFFLE_YUZU) ;
	__vr vr45_r1s1 = _ve_vshf_vvvs(vr4_r1,vr5_r1,VE_VSHUFFLE_YUZU) ;
	__vr vr67_r1s1 = _ve_vshf_vvvs(vr6_r1,vr7_r1,VE_VSHUFFLE_YUZU) ;
	_ve_vst_vss(vr01_r1s1, 4*4*8, filter+3*8+0) ;
	_ve_vst_vss(vr23_r1s1, 4*4*8, filter+3*8+2) ;
	_ve_vst_vss(vr45_r1s1, 4*4*8, filter+3*8+4) ;
	_ve_vst_vss(vr67_r1s1, 4*4*8, filter+3*8+6) ;
      }
      for (int64_t c1 = 0 ; c1 < 256 ; c1++) {
	_ve_lvl(vl) ;
	const int64_t c = c0 + c1 ;

	const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;
	const uint64_t* filter_u64 = (const uint64_t*)(filter+c1*32) ;

	/* memory access errors mihgt be caused */
	__vr vrin_r0s0 = _ve_vldu_vss(4, pInChannel+op) ;
	__vr vrin_r0s1 = _ve_vldu_vss(4, pInChannel+op+1) ;
	__vr vrin_r1s0 = _ve_vldu_vss(4, pInChannel+op+inWidth) ;
	__vr vrin_r1s1 = _ve_vldu_vss(4, pInChannel+op+inWidth+1) ;

	vrin_r0s1 = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), vrin_r0s1, vm_r0s1) ;
	vrin_r1s0 = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), vrin_r1s0, vm_r1s0) ;
	vrin_r1s1 = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), vrin_r1s1, vm_r1s1) ;

	__vr vrinP_r0s0 = _ve_vshf_vvvs(vrin_r0s0, vrin_r0s0, VE_VSHUFFLE_YUZU) ;
	__vr vrinP_r0s1 = _ve_vshf_vvvs(vrin_r0s1, vrin_r0s1, VE_VSHUFFLE_YUZU) ;
	__vr vrinP_r1s0 = _ve_vshf_vvvs(vrin_r1s0, vrin_r1s0, VE_VSHUFFLE_YUZU) ;
	__vr vrinP_r1s1 = _ve_vshf_vvvs(vrin_r1s1, vrin_r1s1, VE_VSHUFFLE_YUZU) ;

	vrsum01 = _ve_pvfmad_vvsv(vrsum01, filter_u64[0], vrinP_r0s0) ;
	vrsum23 = _ve_pvfmad_vvsv(vrsum23, filter_u64[1], vrinP_r0s0) ;
	vrsum45 = _ve_pvfmad_vvsv(vrsum45, filter_u64[2], vrinP_r0s0) ;
	vrsum67 = _ve_pvfmad_vvsv(vrsum67, filter_u64[3], vrinP_r0s0) ;

	vrsum01 = _ve_pvfmad_vvsv(vrsum01, filter_u64[4], vrinP_r0s1) ;
	vrsum23 = _ve_pvfmad_vvsv(vrsum23, filter_u64[5], vrinP_r0s1) ;
	vrsum45 = _ve_pvfmad_vvsv(vrsum45, filter_u64[6], vrinP_r0s1) ;
	vrsum67 = _ve_pvfmad_vvsv(vrsum67, filter_u64[7], vrinP_r0s1) ;

	vrsum01 = _ve_pvfmad_vvsv(vrsum01, filter_u64[8], vrinP_r1s0) ;
	vrsum23 = _ve_pvfmad_vvsv(vrsum23, filter_u64[9], vrinP_r1s0) ;
	vrsum45 = _ve_pvfmad_vvsv(vrsum45, filter_u64[10], vrinP_r1s0) ;
	vrsum67 = _ve_pvfmad_vvsv(vrsum67, filter_u64[11], vrinP_r1s0) ;

	vrsum01 = _ve_pvfmad_vvsv(vrsum01, filter_u64[12], vrinP_r1s1) ;
	vrsum23 = _ve_pvfmad_vvsv(vrsum23, filter_u64[13], vrinP_r1s1) ;
	vrsum45 = _ve_pvfmad_vvsv(vrsum45, filter_u64[14], vrinP_r1s1) ;
	vrsum67 = _ve_pvfmad_vvsv(vrsum67, filter_u64[15], vrinP_r1s1) ;
      }
    } // inChannel

    _ve_vstu_vss(vrsum01, 4, pOut+outIndex0) ;
    _ve_vstl_vss(vrsum01, 4, pOut+outIndex1) ;
    _ve_vstu_vss(vrsum23, 4, pOut+outIndex2) ;
    _ve_vstl_vss(vrsum23, 4, pOut+outIndex3) ;
    _ve_vstu_vss(vrsum45, 4, pOut+outIndex4) ;
    _ve_vstl_vss(vrsum45, 4, pOut+outIndex5) ;
    _ve_vstu_vss(vrsum67, 4, pOut+outIndex6) ;
    _ve_vstl_vss(vrsum67, 4, pOut+outIndex7) ;

    outIndex0 += vl ;
    outIndex1 += vl ;
    outIndex2 += vl ;
    outIndex3 += vl ;
    outIndex4 += vl ;
    outIndex5 += vl ;
    outIndex6 += vl ;
    outIndex7 += vl ;
  } // outPixels
}

static inline void k16_c256x(
    const float * restrict pIn,
    const float * restrict pKernel,
    float * restrict const pOut,
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
    const int64_t inGroupOffset,
    const int64_t outGroupOffset,
    const int64_t kernGroupOffset,
    const int64_t oPixels,
    const int64_t n,
    const int64_t k,
    float * restrict const filter
)
{
  int64_t outIndex0 = outGroupOffset + (n * outChannel + k  ) * oPixels ;
  int64_t outIndex1 = outGroupOffset + (n * outChannel + k+1) * oPixels ;
  int64_t outIndex2 = outGroupOffset + (n * outChannel + k+2) * oPixels ;
  int64_t outIndex3 = outGroupOffset + (n * outChannel + k+3) * oPixels ;
  int64_t outIndex4 = outGroupOffset + (n * outChannel + k+4) * oPixels ;
  int64_t outIndex5 = outGroupOffset + (n * outChannel + k+5) * oPixels ;
  int64_t outIndex6 = outGroupOffset + (n * outChannel + k+6) * oPixels ;
  int64_t outIndex7 = outGroupOffset + (n * outChannel + k+7) * oPixels ;
  int64_t outIndex8 = outGroupOffset + (n * outChannel + k+8) * oPixels ;
  int64_t outIndex9 = outGroupOffset + (n * outChannel + k+9) * oPixels ;
  int64_t outIndexA = outGroupOffset + (n * outChannel + k+10) * oPixels ;
  int64_t outIndexB = outGroupOffset + (n * outChannel + k+11) * oPixels ;
  int64_t outIndexC = outGroupOffset + (n * outChannel + k+12) * oPixels ;
  int64_t outIndexD = outGroupOffset + (n * outChannel + k+13) * oPixels ;
  int64_t outIndexE = outGroupOffset + (n * outChannel + k+14) * oPixels ;
  int64_t outIndexF = outGroupOffset + (n * outChannel + k+15) * oPixels ;

  for (int64_t op = 0; op < oPixels; op+=VLEN) {
    const int64_t vl = oPixels - op < VLEN ? oPixels - op : VLEN ;

    _ve_lvl(vl) ;

    __vr vrseq = _ve_vseq_v() ;			// xy
    __vr vridx = _ve_vaddsl_vsv(op, vrseq) ;	// op + xy

    __vr vrsum01 = _ve_pvbrd_vs_i64(0UL) ;
    __vr vrsum23 = _ve_pvbrd_vs_i64(0UL) ;
    __vr vrsum45 = _ve_pvbrd_vs_i64(0UL) ;
    __vr vrsum67 = _ve_pvbrd_vs_i64(0UL) ;
    __vr vrsum89 = _ve_pvbrd_vs_i64(0UL) ;
    __vr vrsumAB = _ve_pvbrd_vs_i64(0UL) ;
    __vr vrsumCD = _ve_pvbrd_vs_i64(0UL) ;
    __vr vrsumEF = _ve_pvbrd_vs_i64(0UL) ;

    __vr vry   = _ve_vdivsl_vvs(vridx, outWidth) ;
    __vr vrx   = _ve_vsubsl_vvv(vridx, _ve_vmulul_vsv(outWidth,vry)) ;

    __vm256 vm_r1 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(inHeight-1,vry)) ;	// condition(y+1 < inHeight)
    __vm256 vm_s1 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(inWidth-1,vrx)) ;	// condition(x+1 < inWidth)

    __vm256 vm_r1s0 = vm_r1 ;
    __vm256 vm_r0s1 = vm_s1 ;
    __vm256 vm_r1s1 = _ve_andm_mmm(vm_r1,vm_s1) ;

    for (int64_t c0 = 0; c0 < inChannelGroup; c0+=256) {
      const float *pKerValue  = pKernel + kernGroupOffset + ((k * inChannelGroup + c0) * kernHeight ) * kernWidth ;
      {
	_ve_lvl(256) ;

	__vr vr0_r0 = _ve_vld_vss(4*4, pKerValue +0*inChannelGroup*4) ;
	__vr vr1_r0 = _ve_vld_vss(4*4, pKerValue +1*inChannelGroup*4) ;
	__vr vr2_r0 = _ve_vld_vss(4*4, pKerValue +2*inChannelGroup*4) ;
	__vr vr3_r0 = _ve_vld_vss(4*4, pKerValue +3*inChannelGroup*4) ;
	__vr vr4_r0 = _ve_vld_vss(4*4, pKerValue +4*inChannelGroup*4) ;
	__vr vr5_r0 = _ve_vld_vss(4*4, pKerValue +5*inChannelGroup*4) ;
	__vr vr6_r0 = _ve_vld_vss(4*4, pKerValue +6*inChannelGroup*4) ;
	__vr vr7_r0 = _ve_vld_vss(4*4, pKerValue +7*inChannelGroup*4) ;
	__vr vr8_r0 = _ve_vld_vss(4*4, pKerValue +8*inChannelGroup*4) ;
	__vr vr9_r0 = _ve_vld_vss(4*4, pKerValue +9*inChannelGroup*4) ;
	__vr vrA_r0 = _ve_vld_vss(4*4, pKerValue +10*inChannelGroup*4) ;
	__vr vrB_r0 = _ve_vld_vss(4*4, pKerValue +11*inChannelGroup*4) ;
	__vr vrC_r0 = _ve_vld_vss(4*4, pKerValue +12*inChannelGroup*4) ;
	__vr vrD_r0 = _ve_vld_vss(4*4, pKerValue +13*inChannelGroup*4) ;
	__vr vrE_r0 = _ve_vld_vss(4*4, pKerValue +14*inChannelGroup*4) ;
	__vr vrF_r0 = _ve_vld_vss(4*4, pKerValue +15*inChannelGroup*4) ;

	__vr vr0_r1 = _ve_vld_vss(4*4, pKerValue +0*inChannelGroup*4+2) ;
	__vr vr1_r1 = _ve_vld_vss(4*4, pKerValue +1*inChannelGroup*4+2) ;
	__vr vr2_r1 = _ve_vld_vss(4*4, pKerValue +2*inChannelGroup*4+2) ;
	__vr vr3_r1 = _ve_vld_vss(4*4, pKerValue +3*inChannelGroup*4+2) ;
	__vr vr4_r1 = _ve_vld_vss(4*4, pKerValue +4*inChannelGroup*4+2) ;
	__vr vr5_r1 = _ve_vld_vss(4*4, pKerValue +5*inChannelGroup*4+2) ;
	__vr vr6_r1 = _ve_vld_vss(4*4, pKerValue +6*inChannelGroup*4+2) ;
	__vr vr7_r1 = _ve_vld_vss(4*4, pKerValue +7*inChannelGroup*4+2) ;
	__vr vr8_r1 = _ve_vld_vss(4*4, pKerValue +8*inChannelGroup*4+2) ;
	__vr vr9_r1 = _ve_vld_vss(4*4, pKerValue +9*inChannelGroup*4+2) ;
	__vr vrA_r1 = _ve_vld_vss(4*4, pKerValue +10*inChannelGroup*4+2) ;
	__vr vrB_r1 = _ve_vld_vss(4*4, pKerValue +11*inChannelGroup*4+2) ;
	__vr vrC_r1 = _ve_vld_vss(4*4, pKerValue +12*inChannelGroup*4+2) ;
	__vr vrD_r1 = _ve_vld_vss(4*4, pKerValue +13*inChannelGroup*4+2) ;
	__vr vrE_r1 = _ve_vld_vss(4*4, pKerValue +14*inChannelGroup*4+2) ;
	__vr vrF_r1 = _ve_vld_vss(4*4, pKerValue +15*inChannelGroup*4+2) ;

	__vr vr01_r0s0 = _ve_vshf_vvvs(vr0_r0,vr1_r0,VE_VSHUFFLE_YLZL) ;
	__vr vr23_r0s0 = _ve_vshf_vvvs(vr2_r0,vr3_r0,VE_VSHUFFLE_YLZL) ;
	__vr vr45_r0s0 = _ve_vshf_vvvs(vr4_r0,vr5_r0,VE_VSHUFFLE_YLZL) ;
	__vr vr67_r0s0 = _ve_vshf_vvvs(vr6_r0,vr7_r0,VE_VSHUFFLE_YLZL) ;
	__vr vr89_r0s0 = _ve_vshf_vvvs(vr8_r0,vr9_r0,VE_VSHUFFLE_YLZL) ;
	__vr vrAB_r0s0 = _ve_vshf_vvvs(vrA_r0,vrB_r0,VE_VSHUFFLE_YLZL) ;
	__vr vrCD_r0s0 = _ve_vshf_vvvs(vrC_r0,vrD_r0,VE_VSHUFFLE_YLZL) ;
	__vr vrEF_r0s0 = _ve_vshf_vvvs(vrE_r0,vrF_r0,VE_VSHUFFLE_YLZL) ;
	_ve_vst_vss(vr01_r0s0, 4*4*16, filter) ;
	_ve_vst_vss(vr23_r0s0, 4*4*16, filter+2) ;
	_ve_vst_vss(vr45_r0s0, 4*4*16, filter+4) ;
	_ve_vst_vss(vr67_r0s0, 4*4*16, filter+6) ;
	_ve_vst_vss(vr89_r0s0, 4*4*16, filter+8) ;
	_ve_vst_vss(vrAB_r0s0, 4*4*16, filter+10) ;
	_ve_vst_vss(vrCD_r0s0, 4*4*16, filter+12) ;
	_ve_vst_vss(vrEF_r0s0, 4*4*16, filter+14) ;

	__vr vr01_r0s1 = _ve_vshf_vvvs(vr0_r0,vr1_r0,VE_VSHUFFLE_YUZU) ;
	__vr vr23_r0s1 = _ve_vshf_vvvs(vr2_r0,vr3_r0,VE_VSHUFFLE_YUZU) ;
	__vr vr45_r0s1 = _ve_vshf_vvvs(vr4_r0,vr5_r0,VE_VSHUFFLE_YUZU) ;
	__vr vr67_r0s1 = _ve_vshf_vvvs(vr6_r0,vr7_r0,VE_VSHUFFLE_YUZU) ;
	__vr vr89_r0s1 = _ve_vshf_vvvs(vr8_r0,vr9_r0,VE_VSHUFFLE_YUZU) ;
	__vr vrAB_r0s1 = _ve_vshf_vvvs(vrA_r0,vrB_r0,VE_VSHUFFLE_YUZU) ;
	__vr vrCD_r0s1 = _ve_vshf_vvvs(vrC_r0,vrD_r0,VE_VSHUFFLE_YUZU) ;
	__vr vrEF_r0s1 = _ve_vshf_vvvs(vrE_r0,vrF_r0,VE_VSHUFFLE_YUZU) ;
	_ve_vst_vss(vr01_r0s1, 4*4*16, filter+16+0) ;
	_ve_vst_vss(vr23_r0s1, 4*4*16, filter+16+2) ;
	_ve_vst_vss(vr45_r0s1, 4*4*16, filter+16+4) ;
	_ve_vst_vss(vr67_r0s1, 4*4*16, filter+16+6) ;
	_ve_vst_vss(vr89_r0s1, 4*4*16, filter+16+8) ;
	_ve_vst_vss(vrAB_r0s1, 4*4*16, filter+16+10) ;
	_ve_vst_vss(vrCD_r0s1, 4*4*16, filter+16+12) ;
	_ve_vst_vss(vrEF_r0s1, 4*4*16, filter+16+14) ;

	__vr vr01_r1s0 = _ve_vshf_vvvs(vr0_r1,vr1_r1,VE_VSHUFFLE_YLZL) ;
	__vr vr23_r1s0 = _ve_vshf_vvvs(vr2_r1,vr3_r1,VE_VSHUFFLE_YLZL) ;
	__vr vr45_r1s0 = _ve_vshf_vvvs(vr4_r1,vr5_r1,VE_VSHUFFLE_YLZL) ;
	__vr vr67_r1s0 = _ve_vshf_vvvs(vr6_r1,vr7_r1,VE_VSHUFFLE_YLZL) ;
	__vr vr89_r1s0 = _ve_vshf_vvvs(vr8_r1,vr9_r1,VE_VSHUFFLE_YLZL) ;
	__vr vrAB_r1s0 = _ve_vshf_vvvs(vrA_r1,vrB_r1,VE_VSHUFFLE_YLZL) ;
	__vr vrCD_r1s0 = _ve_vshf_vvvs(vrC_r1,vrD_r1,VE_VSHUFFLE_YLZL) ;
	__vr vrEF_r1s0 = _ve_vshf_vvvs(vrE_r1,vrF_r1,VE_VSHUFFLE_YLZL) ;
	_ve_vst_vss(vr01_r1s0, 4*4*16, filter+2*16+0) ;
	_ve_vst_vss(vr23_r1s0, 4*4*16, filter+2*16+2) ;
	_ve_vst_vss(vr45_r1s0, 4*4*16, filter+2*16+4) ;
	_ve_vst_vss(vr67_r1s0, 4*4*16, filter+2*16+6) ;
	_ve_vst_vss(vr89_r1s0, 4*4*16, filter+2*16+8) ;
	_ve_vst_vss(vrAB_r1s0, 4*4*16, filter+2*16+10) ;
	_ve_vst_vss(vrCD_r1s0, 4*4*16, filter+2*16+12) ;
	_ve_vst_vss(vrEF_r1s0, 4*4*16, filter+2*16+14) ;

	__vr vr01_r1s1 = _ve_vshf_vvvs(vr0_r1,vr1_r1,VE_VSHUFFLE_YUZU) ;
	__vr vr23_r1s1 = _ve_vshf_vvvs(vr2_r1,vr3_r1,VE_VSHUFFLE_YUZU) ;
	__vr vr45_r1s1 = _ve_vshf_vvvs(vr4_r1,vr5_r1,VE_VSHUFFLE_YUZU) ;
	__vr vr67_r1s1 = _ve_vshf_vvvs(vr6_r1,vr7_r1,VE_VSHUFFLE_YUZU) ;
	__vr vr89_r1s1 = _ve_vshf_vvvs(vr8_r1,vr9_r1,VE_VSHUFFLE_YUZU) ;
	__vr vrAB_r1s1 = _ve_vshf_vvvs(vrA_r1,vrB_r1,VE_VSHUFFLE_YUZU) ;
	__vr vrCD_r1s1 = _ve_vshf_vvvs(vrC_r1,vrD_r1,VE_VSHUFFLE_YUZU) ;
	__vr vrEF_r1s1 = _ve_vshf_vvvs(vrE_r1,vrF_r1,VE_VSHUFFLE_YUZU) ;
	_ve_vst_vss(vr01_r1s1, 4*4*16, filter+3*16+0) ;
	_ve_vst_vss(vr23_r1s1, 4*4*16, filter+3*16+2) ;
	_ve_vst_vss(vr45_r1s1, 4*4*16, filter+3*16+4) ;
	_ve_vst_vss(vr67_r1s1, 4*4*16, filter+3*16+6) ;
	_ve_vst_vss(vr89_r1s1, 4*4*16, filter+3*16+8) ;
	_ve_vst_vss(vrAB_r1s1, 4*4*16, filter+3*16+10) ;
	_ve_vst_vss(vrCD_r1s1, 4*4*16, filter+3*16+12) ;
	_ve_vst_vss(vrEF_r1s1, 4*4*16, filter+3*16+14) ;
      }
      for (int64_t c1 = 0 ; c1 < 256 ; c1++) {
	_ve_lvl(vl) ;
	const int64_t c = c0 + c1 ;

	const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;
	const uint64_t* filter_u64 = (const uint64_t*)(filter+c1*64) ;

	/* memory access errors mihgt be caused */
	__vr vrin_r0s0 = _ve_vldu_vss(4, pInChannel+op) ;
	__vr vrin_r0s1 = _ve_vldu_vss(4, pInChannel+op+1) ;
	__vr vrin_r1s0 = _ve_vldu_vss(4, pInChannel+op+inWidth) ;
	__vr vrin_r1s1 = _ve_vldu_vss(4, pInChannel+op+inWidth+1) ;

	vrin_r0s1 = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), vrin_r0s1, vm_r0s1) ;
	vrin_r1s0 = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), vrin_r1s0, vm_r1s0) ;
	vrin_r1s1 = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), vrin_r1s1, vm_r1s1) ;

	__vr vrinP_r0s0 = _ve_vshf_vvvs(vrin_r0s0, vrin_r0s0, VE_VSHUFFLE_YUZU) ;
	__vr vrinP_r0s1 = _ve_vshf_vvvs(vrin_r0s1, vrin_r0s1, VE_VSHUFFLE_YUZU) ;
	__vr vrinP_r1s0 = _ve_vshf_vvvs(vrin_r1s0, vrin_r1s0, VE_VSHUFFLE_YUZU) ;
	__vr vrinP_r1s1 = _ve_vshf_vvvs(vrin_r1s1, vrin_r1s1, VE_VSHUFFLE_YUZU) ;

	vrsum01 = _ve_pvfmad_vvsv(vrsum01, filter_u64[0], vrinP_r0s0) ;
	vrsum23 = _ve_pvfmad_vvsv(vrsum23, filter_u64[1], vrinP_r0s0) ;
	vrsum45 = _ve_pvfmad_vvsv(vrsum45, filter_u64[2], vrinP_r0s0) ;
	vrsum67 = _ve_pvfmad_vvsv(vrsum67, filter_u64[3], vrinP_r0s0) ;
	vrsum89 = _ve_pvfmad_vvsv(vrsum89, filter_u64[4], vrinP_r0s0) ;
	vrsumAB = _ve_pvfmad_vvsv(vrsumAB, filter_u64[5], vrinP_r0s0) ;
	vrsumCD = _ve_pvfmad_vvsv(vrsumCD, filter_u64[6], vrinP_r0s0) ;
	vrsumEF = _ve_pvfmad_vvsv(vrsumEF, filter_u64[7], vrinP_r0s0) ;

	vrsum01 = _ve_pvfmad_vvsv(vrsum01, filter_u64[8], vrinP_r0s1) ;
	vrsum23 = _ve_pvfmad_vvsv(vrsum23, filter_u64[9], vrinP_r0s1) ;
	vrsum45 = _ve_pvfmad_vvsv(vrsum45, filter_u64[10], vrinP_r0s1) ;
	vrsum67 = _ve_pvfmad_vvsv(vrsum67, filter_u64[11], vrinP_r0s1) ;
	vrsum89 = _ve_pvfmad_vvsv(vrsum89, filter_u64[12], vrinP_r0s1) ;
	vrsumAB = _ve_pvfmad_vvsv(vrsumAB, filter_u64[13], vrinP_r0s1) ;
	vrsumCD = _ve_pvfmad_vvsv(vrsumCD, filter_u64[14], vrinP_r0s1) ;
	vrsumEF = _ve_pvfmad_vvsv(vrsumEF, filter_u64[15], vrinP_r0s1) ;

	vrsum01 = _ve_pvfmad_vvsv(vrsum01, filter_u64[16], vrinP_r1s0) ;
	vrsum23 = _ve_pvfmad_vvsv(vrsum23, filter_u64[17], vrinP_r1s0) ;
	vrsum45 = _ve_pvfmad_vvsv(vrsum45, filter_u64[18], vrinP_r1s0) ;
	vrsum67 = _ve_pvfmad_vvsv(vrsum67, filter_u64[19], vrinP_r1s0) ;
	vrsum89 = _ve_pvfmad_vvsv(vrsum89, filter_u64[20], vrinP_r1s0) ;
	vrsumAB = _ve_pvfmad_vvsv(vrsumAB, filter_u64[21], vrinP_r1s0) ;
	vrsumCD = _ve_pvfmad_vvsv(vrsumCD, filter_u64[22], vrinP_r1s0) ;
	vrsumEF = _ve_pvfmad_vvsv(vrsumEF, filter_u64[23], vrinP_r1s0) ;

	vrsum01 = _ve_pvfmad_vvsv(vrsum01, filter_u64[24], vrinP_r1s1) ;
	vrsum23 = _ve_pvfmad_vvsv(vrsum23, filter_u64[25], vrinP_r1s1) ;
	vrsum45 = _ve_pvfmad_vvsv(vrsum45, filter_u64[26], vrinP_r1s1) ;
	vrsum67 = _ve_pvfmad_vvsv(vrsum67, filter_u64[27], vrinP_r1s1) ;
	vrsum89 = _ve_pvfmad_vvsv(vrsum89, filter_u64[28], vrinP_r1s1) ;
	vrsumAB = _ve_pvfmad_vvsv(vrsumAB, filter_u64[29], vrinP_r1s1) ;
	vrsumCD = _ve_pvfmad_vvsv(vrsumCD, filter_u64[30], vrinP_r1s1) ;
	vrsumEF = _ve_pvfmad_vvsv(vrsumEF, filter_u64[31], vrinP_r1s1) ;
      }
    } // inChannel

    _ve_vstu_vss(vrsum01, 4, pOut+outIndex0) ;
    _ve_vstl_vss(vrsum01, 4, pOut+outIndex1) ;
    _ve_vstu_vss(vrsum23, 4, pOut+outIndex2) ;
    _ve_vstl_vss(vrsum23, 4, pOut+outIndex3) ;
    _ve_vstu_vss(vrsum45, 4, pOut+outIndex4) ;
    _ve_vstl_vss(vrsum45, 4, pOut+outIndex5) ;
    _ve_vstu_vss(vrsum67, 4, pOut+outIndex6) ;
    _ve_vstl_vss(vrsum67, 4, pOut+outIndex7) ;
    _ve_vstu_vss(vrsum89, 4, pOut+outIndex8) ;
    _ve_vstl_vss(vrsum89, 4, pOut+outIndex9) ;
    _ve_vstu_vss(vrsumAB, 4, pOut+outIndexA) ;
    _ve_vstl_vss(vrsumAB, 4, pOut+outIndexB) ;
    _ve_vstu_vss(vrsumCD, 4, pOut+outIndexC) ;
    _ve_vstl_vss(vrsumCD, 4, pOut+outIndexD) ;
    _ve_vstu_vss(vrsumEF, 4, pOut+outIndexE) ;
    _ve_vstl_vss(vrsumEF, 4, pOut+outIndexF) ;

    outIndex0 += vl ;
    outIndex1 += vl ;
    outIndex2 += vl ;
    outIndex3 += vl ;
    outIndex4 += vl ;
    outIndex5 += vl ;
    outIndex6 += vl ;
    outIndex7 += vl ;
    outIndex8 += vl ;
    outIndex9 += vl ;
    outIndexA += vl ;
    outIndexB += vl ;
    outIndexC += vl ;
    outIndexD += vl ;
    outIndexE += vl ;
    outIndexF += vl ;
  } // outPixels
}

vednnError_t
vednnConvolutionForward_direct_dil1_str1_padsame_ker2(
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
  const int64_t outWidth   = pParamOut->width;		/* must be equal to inWidth */
  const int64_t outHeight  = pParamOut->height;		/* must be equal to inHeight */
  const int64_t kernWidth  = pParamKernel->width;	/* must be 2 */
  const int64_t kernHeight = pParamKernel->height;	/* must be 2 */

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

  float __attribute__ ((aligned(8))) filter[4*16*256] ;

  {
    for (int64_t n = 0; n < batch; n++) {
      for (int64_t g = 0; g < group; g++) {
	const int64_t inGroupOffset   = g * inChannelGroup * inHeight * inWidth;
	const int64_t outGroupOffset  = g * outChannelGroup * outHeight * outWidth;
	const int64_t kernGroupOffset = g * outChannelGroup * inChannelGroup * kernHeight * kernWidth;

	int k = 0 ;

	if ( (outChannelGroup & 0x01) == 1 ) {
	  k1(pIn, pKernel, pOut,
	     inChannel, inWidth, inHeight,
	     outChannel, outWidth, outHeight,
	     kernWidth, kernHeight,
	     inChannelGroup, outChannelGroup,
	     inGroupOffset, outGroupOffset,
	     kernGroupOffset, oPixels, n, k) ;
	  k++ ;
	}
	if ( ((outChannelGroup >> 1) & 0x01) == 1 ) {
	  k2(pIn, pKernel, pOut,
	     inChannel, inWidth, inHeight,
	     outChannel, outWidth, outHeight,
	     kernWidth, kernHeight,
	     inChannelGroup, outChannelGroup,
	     inGroupOffset, outGroupOffset,
	     kernGroupOffset, oPixels, n, k) ;

	  k+=2 ;
	}
	if ( ((outChannelGroup >> 2) & 0x01) == 1 ) {
	  k4(pIn, pKernel, pOut,
	     inChannel, inWidth, inHeight,
	     outChannel, outWidth, outHeight,
	     kernWidth, kernHeight,
	     inChannelGroup, outChannelGroup,
	     inGroupOffset, outGroupOffset,
	     kernGroupOffset, oPixels, n, k) ;

	  k+=4 ;
	}
	if ( ((outChannelGroup >> 3) & 0x01) == 1 ) {
	  if( inChannel % 512 == 0 && (((uint64_t)pDataKernel) & 0x7) == 0 ) {
	    k8_c512x(pIn, pKernel, pOut,
	       inChannel, inWidth, inHeight,
	       outChannel, outWidth, outHeight,
	       kernWidth, kernHeight,
	       inChannelGroup, outChannelGroup,
	       inGroupOffset, outGroupOffset,
	       kernGroupOffset, oPixels, n, k,
	       filter ) ;
	  }
	  else {
	    k8(pIn, pKernel, pOut,
	       inChannel, inWidth, inHeight,
	       outChannel, outWidth, outHeight,
	       kernWidth, kernHeight,
	       inChannelGroup, outChannelGroup,
	       inGroupOffset, outGroupOffset,
	       kernGroupOffset, oPixels, n, k) ;
	  }
	  k+=8 ;
	}
	for ( ; k < outChannelGroup; k+=16) {
	  if( inChannel % 256 == 0 && (((uint64_t)pDataKernel) & 0x7) == 0 ) {
	    k16_c256x(pIn, pKernel, pOut,
	       inChannel, inWidth, inHeight,
	       outChannel, outWidth, outHeight,
	       kernWidth, kernHeight,
	       inChannelGroup, outChannelGroup,
	       inGroupOffset, outGroupOffset,
	       kernGroupOffset, oPixels, n, k,
	       filter) ;
	  }
	  else {
	    k16(pIn, pKernel, pOut,
	       inChannel, inWidth, inHeight,
	       outChannel, outWidth, outHeight,
	       kernWidth, kernHeight,
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
