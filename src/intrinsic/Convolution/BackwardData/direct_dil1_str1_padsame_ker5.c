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
    const int64_t kernWidth,
    const int64_t kernHeight,
    const int64_t padWidth,
    const int64_t padHeight,
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

    _ve_lvl(vl) ;

    __vr vrseq = _ve_vseq_v() ;			// hw
    __vr vridx = _ve_vaddsl_vsv(gip, vrseq) ;	// op + hw

    __vr vrsum = _ve_vbrdu_vs_f32(0.f) ;

    __vr vrh   = _ve_vdivsl_vvs(vridx, gInWidth) ;
    __vr vrix  = _ve_vsubsl_vvv(vridx, _ve_vmulul_vsv(gInWidth,vrh)) ;

    __vr vrx_s0 = _ve_vaddsl_vsv(2, vrix) ;
    __vr vrx_s1 = _ve_vaddsl_vsv(1, vrix) ;

    __vm256 vmx1_s3 = _ve_vfmkl_mcv(VECC_GE, _ve_vaddsl_vsv(-1, vrix)) ;
    __vm256 vmx1_s4 = _ve_vfmkl_mcv(VECC_GE, _ve_vaddsl_vsv(-2, vrix)) ;

    __vm256 vmx2_s0 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(gOutWidth-2,vrix)) ;
    __vm256 vmx2_s1 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(gOutWidth-1,vrix)) ;

    __vm256 vmx_s0  = vmx2_s0 ;
    __vm256 vmx_s1  = vmx2_s1 ;
    __vm256 vmx_s3  = vmx1_s3 ;
    __vm256 vmx_s4  = vmx1_s4 ;

    for (int64_t r=0; r<kernHeight; r++) {
      __vr vry = _ve_vaddsl_vsv(padHeight-r, vrh) ;

      __vm256 vmy1 = _ve_vfmkl_mcv(VECC_GE, vry) ;
      __vm256 vmy2 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(gOutHeight,vry)) ;
      __vm256 vmy  = _ve_andm_mmm(vmy1, vmy2) ;

      __vm256 vmall_s0 = _ve_andm_mmm(vmy,vmx_s0) ;
      __vm256 vmall_s1 = _ve_andm_mmm(vmy,vmx_s1) ;
      __vm256 vmall_s2 = vmy ;
      __vm256 vmall_s3 = _ve_andm_mmm(vmy,vmx_s3) ;
      __vm256 vmall_s4 = _ve_andm_mmm(vmy,vmx_s4) ;

      for (int64_t k=0; k<gOutChannelGroup; k++) {

	const float *pGOutChannel = pGOut + gOutGroupOffset + ((n * gOutChannel + k) * gOutHeight * gOutWidth ) ;

	/* memory access errors mihgt be caused */
	__vr vrgout_s0 = _ve_vldu_vss(4,&pGOutChannel[gip+(padHeight-r)*gOutWidth+(padWidth-0)]) ;
	__vr vrgout_s1 = _ve_vldu_vss(4,&pGOutChannel[gip+(padHeight-r)*gOutWidth+(padWidth-1)]) ;
	__vr vrgout_s2 = _ve_vldu_vss(4,&pGOutChannel[gip+(padHeight-r)*gOutWidth+(padWidth-2)]) ;
	__vr vrgout_s3 = _ve_vldu_vss(4,&pGOutChannel[gip+(padHeight-r)*gOutWidth+(padWidth-3)]) ;
	__vr vrgout_s4 = _ve_vldu_vss(4,&pGOutChannel[gip+(padHeight-r)*gOutWidth+(padWidth-4)]) ;

	const float *pKerValue = pKernel + kernGroupOffset + ((k * gInChannelGroup + c) * kernHeight + r) * kernWidth ;
#define VFMAD1(PKERVALUE, VRGOUT, VMR) {					\
	  VRGOUT = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), VRGOUT, VMR) ;		\
	  const float kerValue = *(PKERVALUE) ;					\
	  vrsum = _ve_vfmads_vvsv(vrsum, kerValue, VRGOUT) ;			\
	}

	VFMAD1(pKerValue,vrgout_s0,vmall_s0) ; pKerValue++ ;
	VFMAD1(pKerValue,vrgout_s1,vmall_s1) ; pKerValue++ ;
	VFMAD1(pKerValue,vrgout_s2,vmall_s2) ; pKerValue++ ;
	VFMAD1(pKerValue,vrgout_s3,vmall_s3) ; pKerValue++ ;
	VFMAD1(pKerValue,vrgout_s4,vmall_s4) ; pKerValue++ ;
#undef VFMAD1
      } // gInChannel
    } // kernHeight

    _ve_vstu_vss(vrsum, 4, pGIn+gInIndex) ;

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
    const int64_t kernWidth,
    const int64_t kernHeight,
    const int64_t padWidth,
    const int64_t padHeight,
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

    _ve_lvl(vl) ;

    __vr vrseq = _ve_vseq_v() ;			// hw
    __vr vridx = _ve_vaddsl_vsv(gip, vrseq) ;	// op + hw

    __vr vrsum01 = _ve_pvbrd_vs_i64(0UL) ;

    __vr vrh   = _ve_vdivsl_vvs(vridx, gInWidth) ;
    __vr vrix  = _ve_vsubsl_vvv(vridx, _ve_vmulul_vsv(gInWidth,vrh)) ;

    __vr vrx_s0 = _ve_vaddsl_vsv(2, vrix) ;
    __vr vrx_s1 = _ve_vaddsl_vsv(1, vrix) ;

    __vm256 vmx1_s3 = _ve_vfmkl_mcv(VECC_GE, _ve_vaddsl_vsv(-1, vrix)) ;
    __vm256 vmx1_s4 = _ve_vfmkl_mcv(VECC_GE, _ve_vaddsl_vsv(-2, vrix)) ;

    __vm256 vmx2_s0 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(gOutWidth-2,vrix)) ;
    __vm256 vmx2_s1 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(gOutWidth-1,vrix)) ;

    __vm256 vmx_s0  = vmx2_s0 ;
    __vm256 vmx_s1  = vmx2_s1 ;
    __vm256 vmx_s3  = vmx1_s3 ;
    __vm256 vmx_s4  = vmx1_s4 ;

    for (int64_t r=0; r<kernHeight; r++) {
      __vr vry = _ve_vaddsl_vsv(padHeight-r, vrh) ;

      __vm256 vmy1 = _ve_vfmkl_mcv(VECC_GE, vry) ;
      __vm256 vmy2 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(gOutHeight,vry)) ;
      __vm256 vmy  = _ve_andm_mmm(vmy1, vmy2) ;

      __vm256 vmall_s0 = _ve_andm_mmm(vmy,vmx_s0) ;
      __vm256 vmall_s1 = _ve_andm_mmm(vmy,vmx_s1) ;
      __vm256 vmall_s2 = vmy ;
      __vm256 vmall_s3 = _ve_andm_mmm(vmy,vmx_s3) ;
      __vm256 vmall_s4 = _ve_andm_mmm(vmy,vmx_s4) ;

      for (int64_t k=0; k<gOutChannelGroup; k++) {

	const float *pGOutChannel = pGOut + gOutGroupOffset + ((n * gOutChannel + k) * gOutHeight * gOutWidth ) ;

	/* memory access errors mihgt be caused */
	__vr vrgout_s0 = _ve_vldu_vss(4,&pGOutChannel[gip+(padHeight-r)*gOutWidth+(padWidth-0)]) ;
	__vr vrgout_s1 = _ve_vldu_vss(4,&pGOutChannel[gip+(padHeight-r)*gOutWidth+(padWidth-1)]) ;
	__vr vrgout_s2 = _ve_vldu_vss(4,&pGOutChannel[gip+(padHeight-r)*gOutWidth+(padWidth-2)]) ;
	__vr vrgout_s3 = _ve_vldu_vss(4,&pGOutChannel[gip+(padHeight-r)*gOutWidth+(padWidth-3)]) ;
	__vr vrgout_s4 = _ve_vldu_vss(4,&pGOutChannel[gip+(padHeight-r)*gOutWidth+(padWidth-4)]) ;

	const float *pKerValue = pKernel + kernGroupOffset + ((k * gInChannelGroup + c) * kernHeight + r) * kernWidth ;
#define PVFMAD2(PKERVALUE, VRGOUT, VMR) {						\
	  VRGOUT = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), VRGOUT, VMR) ;				\
	  __vr vrgoutP = _ve_vshf_vvvs(VRGOUT, VRGOUT, VE_VSHUFFLE_YUZU) ;			\
	  const uint64_t kerValue01 = _ve_pack_f32p(PKERVALUE,					\
					            PKERVALUE+    kernHeight * kernWidth) ;	\
	  vrsum01 = _ve_pvfmad_vvsv(vrsum01, kerValue01, vrgoutP) ;	\
	}

	PVFMAD2(pKerValue,vrgout_s0,vmall_s0) ; pKerValue++ ;
	PVFMAD2(pKerValue,vrgout_s1,vmall_s1) ; pKerValue++ ;
	PVFMAD2(pKerValue,vrgout_s2,vmall_s2) ; pKerValue++ ;
	PVFMAD2(pKerValue,vrgout_s3,vmall_s3) ; pKerValue++ ;
	PVFMAD2(pKerValue,vrgout_s4,vmall_s4) ; pKerValue++ ;
#undef PVFMAD2
      } // gInChannel
    } // kernHeight

    _ve_vstu_vss(vrsum01, 4, pGIn+gInIndex) ;
    _ve_vstl_vss(vrsum01, 4, pGIn+gInIndex+  gInPixels) ;

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
    const int64_t kernWidth,
    const int64_t kernHeight,
    const int64_t padWidth,
    const int64_t padHeight,
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

    _ve_lvl(vl) ;

    __vr vrseq = _ve_vseq_v() ;			// hw
    __vr vridx = _ve_vaddsl_vsv(gip, vrseq) ;	// op + hw

    __vr vrsum01 = _ve_pvbrd_vs_i64(0UL) ;
    __vr vrsum23 = _ve_pvbrd_vs_i64(0UL) ;

    __vr vrh   = _ve_vdivsl_vvs(vridx, gInWidth) ;
    __vr vrix  = _ve_vsubsl_vvv(vridx, _ve_vmulul_vsv(gInWidth,vrh)) ;

    __vr vrx_s0 = _ve_vaddsl_vsv(2, vrix) ;
    __vr vrx_s1 = _ve_vaddsl_vsv(1, vrix) ;

    __vm256 vmx1_s3 = _ve_vfmkl_mcv(VECC_GE, _ve_vaddsl_vsv(-1, vrix)) ;
    __vm256 vmx1_s4 = _ve_vfmkl_mcv(VECC_GE, _ve_vaddsl_vsv(-2, vrix)) ;

    __vm256 vmx2_s0 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(gOutWidth-2,vrix)) ;
    __vm256 vmx2_s1 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(gOutWidth-1,vrix)) ;

    __vm256 vmx_s0  = vmx2_s0 ;
    __vm256 vmx_s1  = vmx2_s1 ;
    __vm256 vmx_s3  = vmx1_s3 ;
    __vm256 vmx_s4  = vmx1_s4 ;

    for (int64_t r=0; r<kernHeight; r++) {
      __vr vry = _ve_vaddsl_vsv(padHeight-r, vrh) ;

      __vm256 vmy1 = _ve_vfmkl_mcv(VECC_GE, vry) ;
      __vm256 vmy2 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(gOutHeight,vry)) ;
      __vm256 vmy  = _ve_andm_mmm(vmy1, vmy2) ;

      __vm256 vmall_s0 = _ve_andm_mmm(vmy,vmx_s0) ;
      __vm256 vmall_s1 = _ve_andm_mmm(vmy,vmx_s1) ;
      __vm256 vmall_s2 = vmy ;
      __vm256 vmall_s3 = _ve_andm_mmm(vmy,vmx_s3) ;
      __vm256 vmall_s4 = _ve_andm_mmm(vmy,vmx_s4) ;

      for (int64_t k=0; k<gOutChannelGroup; k++) {

	const float *pGOutChannel = pGOut + gOutGroupOffset + ((n * gOutChannel + k) * gOutHeight * gOutWidth ) ;

	/* memory access errors mihgt be caused */
	__vr vrgout_s0 = _ve_vldu_vss(4,&pGOutChannel[gip+(padHeight-r)*gOutWidth+(padWidth-0)]) ;
	__vr vrgout_s1 = _ve_vldu_vss(4,&pGOutChannel[gip+(padHeight-r)*gOutWidth+(padWidth-1)]) ;
	__vr vrgout_s2 = _ve_vldu_vss(4,&pGOutChannel[gip+(padHeight-r)*gOutWidth+(padWidth-2)]) ;
	__vr vrgout_s3 = _ve_vldu_vss(4,&pGOutChannel[gip+(padHeight-r)*gOutWidth+(padWidth-3)]) ;
	__vr vrgout_s4 = _ve_vldu_vss(4,&pGOutChannel[gip+(padHeight-r)*gOutWidth+(padWidth-4)]) ;

	const float *pKerValue = pKernel + kernGroupOffset + ((k * gInChannelGroup + c) * kernHeight + r) * kernWidth ;
#define PVFMAD4(PKERVALUE, VRGOUT, VMR) {						\
	  VRGOUT = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), VRGOUT, VMR) ;				\
	  __vr vrgoutP = _ve_vshf_vvvs(VRGOUT, VRGOUT, VE_VSHUFFLE_YUZU) ;			\
	  const uint64_t kerValue01 = _ve_pack_f32p(PKERVALUE,					\
					            PKERVALUE+    kernHeight * kernWidth) ;	\
	  const uint64_t kerValue23 = _ve_pack_f32p(PKERVALUE+ 2* kernHeight * kernWidth,	\
					            PKERVALUE+ 3* kernHeight * kernWidth) ;	\
	  vrsum01 = _ve_pvfmad_vvsv(vrsum01, kerValue01, vrgoutP) ;	\
	  vrsum23 = _ve_pvfmad_vvsv(vrsum23, kerValue23, vrgoutP) ;	\
	}

	PVFMAD4(pKerValue,vrgout_s0,vmall_s0) ; pKerValue++ ;
	PVFMAD4(pKerValue,vrgout_s1,vmall_s1) ; pKerValue++ ;
	PVFMAD4(pKerValue,vrgout_s2,vmall_s2) ; pKerValue++ ;
	PVFMAD4(pKerValue,vrgout_s3,vmall_s3) ; pKerValue++ ;
	PVFMAD4(pKerValue,vrgout_s4,vmall_s4) ; pKerValue++ ;
#undef PVFMAD4
      } // gInChannel
    } // kernHeight

    _ve_vstu_vss(vrsum01, 4, pGIn+gInIndex) ;
    _ve_vstl_vss(vrsum01, 4, pGIn+gInIndex+  gInPixels) ;
    _ve_vstu_vss(vrsum23, 4, pGIn+gInIndex+2*gInPixels) ;
    _ve_vstl_vss(vrsum23, 4, pGIn+gInIndex+3*gInPixels) ;

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
    const int64_t kernWidth,
    const int64_t kernHeight,
    const int64_t padWidth,
    const int64_t padHeight,
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

    _ve_lvl(vl) ;

    __vr vrseq = _ve_vseq_v() ;			// hw
    __vr vridx = _ve_vaddsl_vsv(gip, vrseq) ;	// op + hw

    __vr vrsum01 = _ve_pvbrd_vs_i64(0UL) ;
    __vr vrsum23 = _ve_pvbrd_vs_i64(0UL) ;
    __vr vrsum45 = _ve_pvbrd_vs_i64(0UL) ;
    __vr vrsum67 = _ve_pvbrd_vs_i64(0UL) ;

    __vr vrh   = _ve_vdivsl_vvs(vridx, gInWidth) ;
    __vr vrix  = _ve_vsubsl_vvv(vridx, _ve_vmulul_vsv(gInWidth,vrh)) ;

    __vr vrx_s0 = _ve_vaddsl_vsv(2, vrix) ;
    __vr vrx_s1 = _ve_vaddsl_vsv(1, vrix) ;

    __vm256 vmx1_s3 = _ve_vfmkl_mcv(VECC_GE, _ve_vaddsl_vsv(-1, vrix)) ;
    __vm256 vmx1_s4 = _ve_vfmkl_mcv(VECC_GE, _ve_vaddsl_vsv(-2, vrix)) ;

    __vm256 vmx2_s0 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(gOutWidth-2,vrix)) ;
    __vm256 vmx2_s1 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(gOutWidth-1,vrix)) ;

    __vm256 vmx_s0  = vmx2_s0 ;
    __vm256 vmx_s1  = vmx2_s1 ;
    __vm256 vmx_s3  = vmx1_s3 ;
    __vm256 vmx_s4  = vmx1_s4 ;

    for (int64_t r=0; r<kernHeight; r++) {
      __vr vry = _ve_vaddsl_vsv(padHeight-r, vrh) ;

      __vm256 vmy1 = _ve_vfmkl_mcv(VECC_GE, vry) ;
      __vm256 vmy2 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(gOutHeight,vry)) ;
      __vm256 vmy  = _ve_andm_mmm(vmy1, vmy2) ;

      __vm256 vmall_s0 = _ve_andm_mmm(vmy,vmx_s0) ;
      __vm256 vmall_s1 = _ve_andm_mmm(vmy,vmx_s1) ;
      __vm256 vmall_s2 = vmy ;
      __vm256 vmall_s3 = _ve_andm_mmm(vmy,vmx_s3) ;
      __vm256 vmall_s4 = _ve_andm_mmm(vmy,vmx_s4) ;

      for (int64_t k=0; k<gOutChannelGroup; k++) {

	const float *pGOutChannel = pGOut + gOutGroupOffset + ((n * gOutChannel + k) * gOutHeight * gOutWidth ) ;

	/* memory access errors mihgt be caused */
	__vr vrgout_s0 = _ve_vldu_vss(4,&pGOutChannel[gip+(padHeight-r)*gOutWidth+(padWidth-0)]) ;
	__vr vrgout_s1 = _ve_vldu_vss(4,&pGOutChannel[gip+(padHeight-r)*gOutWidth+(padWidth-1)]) ;
	__vr vrgout_s2 = _ve_vldu_vss(4,&pGOutChannel[gip+(padHeight-r)*gOutWidth+(padWidth-2)]) ;
	__vr vrgout_s3 = _ve_vldu_vss(4,&pGOutChannel[gip+(padHeight-r)*gOutWidth+(padWidth-3)]) ;
	__vr vrgout_s4 = _ve_vldu_vss(4,&pGOutChannel[gip+(padHeight-r)*gOutWidth+(padWidth-4)]) ;

	const float *pKerValue = pKernel + kernGroupOffset + ((k * gInChannelGroup + c) * kernHeight + r) * kernWidth ;
#define PVFMAD8(PKERVALUE, VRGOUT, VMR) {						\
	  VRGOUT = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), VRGOUT, VMR) ;				\
	  __vr vrgoutP = _ve_vshf_vvvs(VRGOUT, VRGOUT, VE_VSHUFFLE_YUZU) ;			\
	  const uint64_t kerValue01 = _ve_pack_f32p(PKERVALUE,					\
					            PKERVALUE+    kernHeight * kernWidth) ;	\
	  const uint64_t kerValue23 = _ve_pack_f32p(PKERVALUE+ 2* kernHeight * kernWidth,	\
					            PKERVALUE+ 3* kernHeight * kernWidth) ;	\
	  const uint64_t kerValue45 = _ve_pack_f32p(PKERVALUE+ 4* kernHeight * kernWidth,	\
					            PKERVALUE+ 5* kernHeight * kernWidth) ;	\
	  const uint64_t kerValue67 = _ve_pack_f32p(PKERVALUE+ 6* kernHeight * kernWidth,	\
					            PKERVALUE+ 7* kernHeight * kernWidth) ;	\
	  vrsum01 = _ve_pvfmad_vvsv(vrsum01, kerValue01, vrgoutP) ;	\
	  vrsum23 = _ve_pvfmad_vvsv(vrsum23, kerValue23, vrgoutP) ;	\
	  vrsum45 = _ve_pvfmad_vvsv(vrsum45, kerValue45, vrgoutP) ;	\
	  vrsum67 = _ve_pvfmad_vvsv(vrsum67, kerValue67, vrgoutP) ;	\
	}

	PVFMAD8(pKerValue,vrgout_s0,vmall_s0) ; pKerValue++ ;
	PVFMAD8(pKerValue,vrgout_s1,vmall_s1) ; pKerValue++ ;
	PVFMAD8(pKerValue,vrgout_s2,vmall_s2) ; pKerValue++ ;
	PVFMAD8(pKerValue,vrgout_s3,vmall_s3) ; pKerValue++ ;
	PVFMAD8(pKerValue,vrgout_s4,vmall_s4) ; pKerValue++ ;
#undef PVFMAD8
      } // gInChannel
    } // kernHeight

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
    const int64_t kernWidth,
    const int64_t kernHeight,
    const int64_t padWidth,
    const int64_t padHeight,
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
    __vr vrix  = _ve_vsubsl_vvv(vridx, _ve_vmulul_vsv(gInWidth,vrh)) ;

    __vr vrx_s0 = _ve_vaddsl_vsv(2, vrix) ;
    __vr vrx_s1 = _ve_vaddsl_vsv(1, vrix) ;

    __vm256 vmx1_s3 = _ve_vfmkl_mcv(VECC_GE, _ve_vaddsl_vsv(-1, vrix)) ;
    __vm256 vmx1_s4 = _ve_vfmkl_mcv(VECC_GE, _ve_vaddsl_vsv(-2, vrix)) ;

    __vm256 vmx2_s0 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(gOutWidth-2,vrix)) ;
    __vm256 vmx2_s1 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(gOutWidth-1,vrix)) ;

    __vm256 vmx_s0  = vmx2_s0 ;
    __vm256 vmx_s1  = vmx2_s1 ;
    __vm256 vmx_s3  = vmx1_s3 ;
    __vm256 vmx_s4  = vmx1_s4 ;

    for (int64_t r=0; r<kernHeight; r++) {
      __vr vry = _ve_vaddsl_vsv(padHeight-r, vrh) ;

      __vm256 vmy1 = _ve_vfmkl_mcv(VECC_GE, vry) ;
      __vm256 vmy2 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(gOutHeight,vry)) ;
      __vm256 vmy  = _ve_andm_mmm(vmy1, vmy2) ;

      __vm256 vmall_s0 = _ve_andm_mmm(vmy,vmx_s0) ;
      __vm256 vmall_s1 = _ve_andm_mmm(vmy,vmx_s1) ;
      __vm256 vmall_s2 = vmy ;
      __vm256 vmall_s3 = _ve_andm_mmm(vmy,vmx_s3) ;
      __vm256 vmall_s4 = _ve_andm_mmm(vmy,vmx_s4) ;

      for (int64_t k=0; k<gOutChannelGroup; k++) {

	const float *pGOutChannel = pGOut + gOutGroupOffset + ((n * gOutChannel + k) * gOutHeight * gOutWidth ) ;

	/* memory access errors mihgt be caused */
	__vr vrgout_s0 = _ve_vldu_vss(4,&pGOutChannel[gip+(padHeight-r)*gOutWidth+(padWidth-0)]) ;
	__vr vrgout_s1 = _ve_vldu_vss(4,&pGOutChannel[gip+(padHeight-r)*gOutWidth+(padWidth-1)]) ;
	__vr vrgout_s2 = _ve_vldu_vss(4,&pGOutChannel[gip+(padHeight-r)*gOutWidth+(padWidth-2)]) ;
	__vr vrgout_s3 = _ve_vldu_vss(4,&pGOutChannel[gip+(padHeight-r)*gOutWidth+(padWidth-3)]) ;
	__vr vrgout_s4 = _ve_vldu_vss(4,&pGOutChannel[gip+(padHeight-r)*gOutWidth+(padWidth-4)]) ;

	const float *pKerValue = pKernel + kernGroupOffset + ((k * gInChannelGroup + c) * kernHeight + r) * kernWidth ;
#define PVFMAD16(PKERVALUE, VRGOUT, VMR) {							\
	  VRGOUT = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), VRGOUT, VMR) ;				\
	  __vr vrgoutP = _ve_vshf_vvvs(VRGOUT, VRGOUT, VE_VSHUFFLE_YUZU) ;			\
	  const uint64_t kerValue01 = _ve_pack_f32p(PKERVALUE,					\
					            PKERVALUE+    kernHeight * kernWidth) ;	\
	  const uint64_t kerValue23 = _ve_pack_f32p(PKERVALUE+ 2* kernHeight * kernWidth,	\
					            PKERVALUE+ 3* kernHeight * kernWidth) ;	\
	  const uint64_t kerValue45 = _ve_pack_f32p(PKERVALUE+ 4* kernHeight * kernWidth,	\
					            PKERVALUE+ 5* kernHeight * kernWidth) ;	\
	  const uint64_t kerValue67 = _ve_pack_f32p(PKERVALUE+ 6* kernHeight * kernWidth,	\
					            PKERVALUE+ 7* kernHeight * kernWidth) ;	\
	  const uint64_t kerValue89 = _ve_pack_f32p(PKERVALUE+ 8* kernHeight * kernWidth,	\
					            PKERVALUE+ 9* kernHeight * kernWidth) ;	\
	  const uint64_t kerValueAB = _ve_pack_f32p(PKERVALUE+10* kernHeight * kernWidth,	\
					            PKERVALUE+11* kernHeight * kernWidth) ;	\
	  const uint64_t kerValueCD = _ve_pack_f32p(PKERVALUE+12* kernHeight * kernWidth,	\
					            PKERVALUE+13* kernHeight * kernWidth) ;	\
	  const uint64_t kerValueEF = _ve_pack_f32p(PKERVALUE+14* kernHeight * kernWidth,	\
					            PKERVALUE+15* kernHeight * kernWidth) ;	\
	  vrsum01 = _ve_pvfmad_vvsv(vrsum01, kerValue01, vrgoutP) ;	\
	  vrsum23 = _ve_pvfmad_vvsv(vrsum23, kerValue23, vrgoutP) ;	\
	  vrsum45 = _ve_pvfmad_vvsv(vrsum45, kerValue45, vrgoutP) ;	\
	  vrsum67 = _ve_pvfmad_vvsv(vrsum67, kerValue67, vrgoutP) ;	\
	  vrsum89 = _ve_pvfmad_vvsv(vrsum89, kerValue89, vrgoutP) ;	\
	  vrsumAB = _ve_pvfmad_vvsv(vrsumAB, kerValueAB, vrgoutP) ;	\
	  vrsumCD = _ve_pvfmad_vvsv(vrsumCD, kerValueCD, vrgoutP) ;	\
	  vrsumEF = _ve_pvfmad_vvsv(vrsumEF, kerValueEF, vrgoutP) ;	\
        }

	PVFMAD16(pKerValue,vrgout_s0,vmall_s0) ; pKerValue++ ;
	PVFMAD16(pKerValue,vrgout_s1,vmall_s1) ; pKerValue++ ;
	PVFMAD16(pKerValue,vrgout_s2,vmall_s2) ; pKerValue++ ;
	PVFMAD16(pKerValue,vrgout_s3,vmall_s3) ; pKerValue++ ;
	PVFMAD16(pKerValue,vrgout_s4,vmall_s4) ; pKerValue++ ;
#undef PVFMAD16
      } // gInChannel
    } // kernHeight

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

static inline void c32(
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
    const int64_t padWidth,
    const int64_t padHeight,
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
    __vr vrsumGH = _ve_pvbrd_vs_i64(0UL) ;
    __vr vrsumIJ = _ve_pvbrd_vs_i64(0UL) ;
    __vr vrsumKL = _ve_pvbrd_vs_i64(0UL) ;
    __vr vrsumMN = _ve_pvbrd_vs_i64(0UL) ;
    __vr vrsumOP = _ve_pvbrd_vs_i64(0UL) ;
    __vr vrsumQR = _ve_pvbrd_vs_i64(0UL) ;
    __vr vrsumST = _ve_pvbrd_vs_i64(0UL) ;
    __vr vrsumUV = _ve_pvbrd_vs_i64(0UL) ;

    __vr vrh   = _ve_vdivsl_vvs(vridx, gInWidth) ;
    __vr vrix  = _ve_vsubsl_vvv(vridx, _ve_vmulul_vsv(gInWidth,vrh)) ;

    __vr vrx_s0 = _ve_vaddsl_vsv(2, vrix) ;
    __vr vrx_s1 = _ve_vaddsl_vsv(1, vrix) ;

    __vm256 vmx1_s3 = _ve_vfmkl_mcv(VECC_GE, _ve_vaddsl_vsv(-1, vrix)) ;
    __vm256 vmx1_s4 = _ve_vfmkl_mcv(VECC_GE, _ve_vaddsl_vsv(-2, vrix)) ;

    __vm256 vmx2_s0 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(gOutWidth-2,vrix)) ;
    __vm256 vmx2_s1 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(gOutWidth-1,vrix)) ;

    __vm256 vmx_s0  = vmx2_s0 ;
    __vm256 vmx_s1  = vmx2_s1 ;
    __vm256 vmx_s3  = vmx1_s3 ;
    __vm256 vmx_s4  = vmx1_s4 ;

    for (int64_t r=0; r<kernHeight; r++) {
      __vr vry = _ve_vaddsl_vsv(padHeight-r, vrh) ;

      __vm256 vmy1 = _ve_vfmkl_mcv(VECC_GE, vry) ;
      __vm256 vmy2 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(gOutHeight,vry)) ;
      __vm256 vmy  = _ve_andm_mmm(vmy1, vmy2) ;

      __vm256 vmall_s0 = _ve_andm_mmm(vmy,vmx_s0) ;
      __vm256 vmall_s1 = _ve_andm_mmm(vmy,vmx_s1) ;
      __vm256 vmall_s2 = vmy ;
      __vm256 vmall_s3 = _ve_andm_mmm(vmy,vmx_s3) ;
      __vm256 vmall_s4 = _ve_andm_mmm(vmy,vmx_s4) ;

      for (int64_t k=0; k<gOutChannelGroup; k++) {

	const float *pGOutChannel = pGOut + gOutGroupOffset + ((n * gOutChannel + k) * gOutHeight * gOutWidth ) ;

	/* memory access errors mihgt be caused */
	__vr vrgout_s0 = _ve_vldu_vss(4,&pGOutChannel[gip+(padHeight-r)*gOutWidth+(padWidth-0)]) ;
	__vr vrgout_s1 = _ve_vldu_vss(4,&pGOutChannel[gip+(padHeight-r)*gOutWidth+(padWidth-1)]) ;
	__vr vrgout_s2 = _ve_vldu_vss(4,&pGOutChannel[gip+(padHeight-r)*gOutWidth+(padWidth-2)]) ;
	__vr vrgout_s3 = _ve_vldu_vss(4,&pGOutChannel[gip+(padHeight-r)*gOutWidth+(padWidth-3)]) ;
	__vr vrgout_s4 = _ve_vldu_vss(4,&pGOutChannel[gip+(padHeight-r)*gOutWidth+(padWidth-4)]) ;

	const float *pKerValue = pKernel + kernGroupOffset + ((k * gInChannelGroup + c) * kernHeight + r) * kernWidth ;
#define PVFMAD32(PKERVALUE, VRGOUT, VMR) {							\
	  VRGOUT = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), VRGOUT, VMR) ;				\
	  __vr vrgoutP = _ve_vshf_vvvs(VRGOUT, VRGOUT, VE_VSHUFFLE_YUZU) ;			\
	  const uint64_t kerValue01 = _ve_pack_f32p(PKERVALUE,					\
					            PKERVALUE+    kernHeight * kernWidth) ;	\
	  const uint64_t kerValue23 = _ve_pack_f32p(PKERVALUE+ 2* kernHeight * kernWidth,	\
					            PKERVALUE+ 3* kernHeight * kernWidth) ;	\
	  const uint64_t kerValue45 = _ve_pack_f32p(PKERVALUE+ 4* kernHeight * kernWidth,	\
					            PKERVALUE+ 5* kernHeight * kernWidth) ;	\
	  const uint64_t kerValue67 = _ve_pack_f32p(PKERVALUE+ 6* kernHeight * kernWidth,	\
					            PKERVALUE+ 7* kernHeight * kernWidth) ;	\
	  const uint64_t kerValue89 = _ve_pack_f32p(PKERVALUE+ 8* kernHeight * kernWidth,	\
					            PKERVALUE+ 9* kernHeight * kernWidth) ;	\
	  const uint64_t kerValueAB = _ve_pack_f32p(PKERVALUE+10* kernHeight * kernWidth,	\
					            PKERVALUE+11* kernHeight * kernWidth) ;	\
	  const uint64_t kerValueCD = _ve_pack_f32p(PKERVALUE+12* kernHeight * kernWidth,	\
					            PKERVALUE+13* kernHeight * kernWidth) ;	\
	  const uint64_t kerValueEF = _ve_pack_f32p(PKERVALUE+14* kernHeight * kernWidth,	\
					            PKERVALUE+15* kernHeight * kernWidth) ;	\
	  vrsum01 = _ve_pvfmad_vvsv(vrsum01, kerValue01, vrgoutP) ;	\
	  vrsum23 = _ve_pvfmad_vvsv(vrsum23, kerValue23, vrgoutP) ;	\
	  vrsum45 = _ve_pvfmad_vvsv(vrsum45, kerValue45, vrgoutP) ;	\
	  vrsum67 = _ve_pvfmad_vvsv(vrsum67, kerValue67, vrgoutP) ;	\
	  vrsum89 = _ve_pvfmad_vvsv(vrsum89, kerValue89, vrgoutP) ;	\
	  vrsumAB = _ve_pvfmad_vvsv(vrsumAB, kerValueAB, vrgoutP) ;	\
	  vrsumCD = _ve_pvfmad_vvsv(vrsumCD, kerValueCD, vrgoutP) ;	\
	  vrsumEF = _ve_pvfmad_vvsv(vrsumEF, kerValueEF, vrgoutP) ;	\
	  const uint64_t kerValueGH = _ve_pack_f32p(PKERVALUE+16* kernHeight * kernWidth,	\
					            PKERVALUE+17* kernHeight * kernWidth) ;	\
	  const uint64_t kerValueIJ = _ve_pack_f32p(PKERVALUE+18* kernHeight * kernWidth,	\
					            PKERVALUE+19* kernHeight * kernWidth) ;	\
	  const uint64_t kerValueKL = _ve_pack_f32p(PKERVALUE+20* kernHeight * kernWidth,	\
					            PKERVALUE+21* kernHeight * kernWidth) ;	\
	  const uint64_t kerValueMN = _ve_pack_f32p(PKERVALUE+22* kernHeight * kernWidth,	\
					            PKERVALUE+23* kernHeight * kernWidth) ;	\
	  const uint64_t kerValueOP = _ve_pack_f32p(PKERVALUE+24* kernHeight * kernWidth,	\
					            PKERVALUE+25* kernHeight * kernWidth) ;	\
	  const uint64_t kerValueQR = _ve_pack_f32p(PKERVALUE+26* kernHeight * kernWidth,	\
					            PKERVALUE+27* kernHeight * kernWidth) ;	\
	  const uint64_t kerValueST = _ve_pack_f32p(PKERVALUE+28* kernHeight * kernWidth,	\
					            PKERVALUE+29* kernHeight * kernWidth) ;	\
	  const uint64_t kerValueUV = _ve_pack_f32p(PKERVALUE+30* kernHeight * kernWidth,	\
					            PKERVALUE+31* kernHeight * kernWidth) ;	\
	  vrsumGH = _ve_pvfmad_vvsv(vrsumGH, kerValueGH, vrgoutP) ;	\
	  vrsumIJ = _ve_pvfmad_vvsv(vrsumIJ, kerValueIJ, vrgoutP) ;	\
	  vrsumKL = _ve_pvfmad_vvsv(vrsumKL, kerValueKL, vrgoutP) ;	\
	  vrsumMN = _ve_pvfmad_vvsv(vrsumMN, kerValueMN, vrgoutP) ;	\
	  vrsumOP = _ve_pvfmad_vvsv(vrsumOP, kerValueOP, vrgoutP) ;	\
	  vrsumQR = _ve_pvfmad_vvsv(vrsumQR, kerValueQR, vrgoutP) ;	\
	  vrsumST = _ve_pvfmad_vvsv(vrsumST, kerValueST, vrgoutP) ;	\
	  vrsumUV = _ve_pvfmad_vvsv(vrsumUV, kerValueUV, vrgoutP) ;	\
        }

	PVFMAD32(pKerValue,vrgout_s0,vmall_s0) ; pKerValue++ ;
	PVFMAD32(pKerValue,vrgout_s1,vmall_s1) ; pKerValue++ ;
	PVFMAD32(pKerValue,vrgout_s2,vmall_s2) ; pKerValue++ ;
	PVFMAD32(pKerValue,vrgout_s3,vmall_s3) ; pKerValue++ ;
	PVFMAD32(pKerValue,vrgout_s4,vmall_s4) ; pKerValue++ ;
#undef PVFMAD32
      } // gInChannel
    } // kernHeight

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
    _ve_vstu_vss(vrsumGH, 4, pGIn+gInIndex+16*gInPixels) ;
    _ve_vstl_vss(vrsumGH, 4, pGIn+gInIndex+17*gInPixels) ;
    _ve_vstu_vss(vrsumIJ, 4, pGIn+gInIndex+18*gInPixels) ;
    _ve_vstl_vss(vrsumIJ, 4, pGIn+gInIndex+19*gInPixels) ;
    _ve_vstu_vss(vrsumKL, 4, pGIn+gInIndex+20*gInPixels) ;
    _ve_vstl_vss(vrsumKL, 4, pGIn+gInIndex+21*gInPixels) ;
    _ve_vstu_vss(vrsumMN, 4, pGIn+gInIndex+22*gInPixels) ;
    _ve_vstl_vss(vrsumMN, 4, pGIn+gInIndex+23*gInPixels) ;
    _ve_vstu_vss(vrsumOP, 4, pGIn+gInIndex+24*gInPixels) ;
    _ve_vstl_vss(vrsumOP, 4, pGIn+gInIndex+25*gInPixels) ;
    _ve_vstu_vss(vrsumQR, 4, pGIn+gInIndex+26*gInPixels) ;
    _ve_vstl_vss(vrsumQR, 4, pGIn+gInIndex+27*gInPixels) ;
    _ve_vstu_vss(vrsumST, 4, pGIn+gInIndex+28*gInPixels) ;
    _ve_vstl_vss(vrsumST, 4, pGIn+gInIndex+29*gInPixels) ;
    _ve_vstu_vss(vrsumUV, 4, pGIn+gInIndex+30*gInPixels) ;
    _ve_vstl_vss(vrsumUV, 4, pGIn+gInIndex+31*gInPixels) ;


  } // gInPixels

}


vednnError_t
vednnConvolutionBackwardData_direct_dil1_str1_padsame_ker5 (
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

  for (int64_t n=0; n<batch; n++) {
    for (int64_t g = 0; g < group; g++) {

      int64_t gInGroupOffset  = g * gInChannelGroup * gInHeight * gInWidth;
      int64_t gOutGroupOffset = g * gOutChannelGroup  * gOutHeight  * gOutWidth;
      int64_t kernGroupOffset = g * gOutChannelGroup  * gInChannelGroup * kernHeight * kernWidth;

      int c=0;
      if ( (gInChannelGroup & 0x01) == 1 ) {
	c1(pGOut, pKernel, pGIn,
	   gOutChannel, gOutWidth, gOutHeight,
	   gInChannel, gInWidth, gInHeight,
	   kernWidth, kernHeight,
	   padWidth, padHeight,
	   gInChannelGroup, gOutChannelGroup,
	   gInGroupOffset, gOutGroupOffset,
	   kernGroupOffset, gInPixels,
	   n, c ) ;
	c+=1 ;
      }
      if ( ((gInChannelGroup >> 1) & 0x01) == 1 ) {
	c2(pGOut, pKernel, pGIn,
	   gOutChannel, gOutWidth, gOutHeight,
	   gInChannel, gInWidth, gInHeight,
	   kernWidth, kernHeight,
	   padWidth, padHeight,
	   gInChannelGroup, gOutChannelGroup,
	   gInGroupOffset, gOutGroupOffset,
	   kernGroupOffset, gInPixels,
	   n, c ) ;
	c+=2 ;
      }
      if ( ((gInChannelGroup >> 2) & 0x01) == 1 ) {
	c4(pGOut, pKernel, pGIn,
	   gOutChannel, gOutWidth, gOutHeight,
	   gInChannel, gInWidth, gInHeight,
	   kernWidth, kernHeight,
	   padWidth, padHeight,
	   gInChannelGroup, gOutChannelGroup,
	   gInGroupOffset, gOutGroupOffset,
	   kernGroupOffset, gInPixels,
	   n, c ) ;
	c+=4 ;
      }
      if ( ((gInChannelGroup >> 3) & 0x01) == 1 ) {
	c8(pGOut, pKernel, pGIn,
	   gOutChannel, gOutWidth, gOutHeight,
	   gInChannel, gInWidth, gInHeight,
	   kernWidth, kernHeight,
	   padWidth, padHeight,
	   gInChannelGroup, gOutChannelGroup,
	   gInGroupOffset, gOutGroupOffset,
	   kernGroupOffset, gInPixels,
	   n, c ) ;
	c+=8 ;
      }
      if ( ((gInChannelGroup >> 4) & 0x01) == 1 ) {
	c16(pGOut, pKernel, pGIn,
	   gOutChannel, gOutWidth, gOutHeight,
	   gInChannel, gInWidth, gInHeight,
	   kernWidth, kernHeight,
	   padWidth, padHeight,
	   gInChannelGroup, gOutChannelGroup,
	   gInGroupOffset, gOutGroupOffset,
	   kernGroupOffset, gInPixels,
	   n, c ) ;
	c+=16 ;
      }
      for (; c<gInChannelGroup; ) {

	c32(pGOut, pKernel, pGIn,
	   gOutChannel, gOutWidth, gOutHeight,
	   gInChannel, gInWidth, gInHeight,
	   kernWidth, kernHeight,
	   padWidth, padHeight,
	   gInChannelGroup, gOutChannelGroup,
	   gInGroupOffset, gOutGroupOffset,
	   kernGroupOffset, gInPixels,
	   n, c ) ;
	c+=32 ;

      } // gInChannelGroup
    } // group
  } // batch

  return VEDNN_SUCCESS;
}
