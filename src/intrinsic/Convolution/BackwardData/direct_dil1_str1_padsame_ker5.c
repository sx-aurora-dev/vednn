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

    __vr vrseq = _vel_vseq_vl(vl) ;			// hw
    __vr vridx = _vel_vaddsl_vsvl(gip, vrseq, vl) ;	// op + hw

    __vr vrsum = _vel_vbrds_vsl(0.f, vl) ;

    __vr vrh   = _vel_vdivsl_vvsl(vridx, gInWidth, vl) ;
    __vr vrix  = _vel_vsubsl_vvvl(vridx, _vel_vmulul_vsvl(gInWidth,vrh, vl), vl) ;

    __vr vrx_s0 = _vel_vaddsl_vsvl(2, vrix, vl) ;
    __vr vrx_s1 = _vel_vaddsl_vsvl(1, vrix, vl) ;

    __vm256 vmx1_s3 =  _vel_vfmklge_mvl(_vel_vaddsl_vsvl(-1, vrix, vl), vl) ;
    __vm256 vmx1_s4 =  _vel_vfmklge_mvl(_vel_vaddsl_vsvl(-2, vrix, vl), vl) ;

    __vm256 vmx2_s0 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutWidth-2,vrix, vl), vl) ;
    __vm256 vmx2_s1 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutWidth-1,vrix, vl), vl) ;

    __vm256 vmx_s0  = vmx2_s0 ;
    __vm256 vmx_s1  = vmx2_s1 ;
    __vm256 vmx_s3  = vmx1_s3 ;
    __vm256 vmx_s4  = vmx1_s4 ;

    for (int64_t r=0; r<kernHeight; r++) {
      __vr vry = _vel_vaddsl_vsvl(padHeight-r, vrh, vl) ;

      __vm256 vmy1 =  _vel_vfmklge_mvl(vry, vl) ;
      __vm256 vmy2 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutHeight,vry, vl), vl) ;
      __vm256 vmy  = _vel_andm_mmm(vmy1, vmy2) ;

      __vm256 vmall_s0 = _vel_andm_mmm(vmy,vmx_s0) ;
      __vm256 vmall_s1 = _vel_andm_mmm(vmy,vmx_s1) ;
      __vm256 vmall_s2 = vmy ;
      __vm256 vmall_s3 = _vel_andm_mmm(vmy,vmx_s3) ;
      __vm256 vmall_s4 = _vel_andm_mmm(vmy,vmx_s4) ;

      for (int64_t k=0; k<gOutChannelGroup; k++) {

	const float *pGOutChannel = pGOut + gOutGroupOffset + ((n * gOutChannel + k) * gOutHeight * gOutWidth ) ;

	/* memory access errors mihgt be caused */
	__vr vrgout_s0 = _vel_vldu_vssl(4,&pGOutChannel[gip+(padHeight-r)*gOutWidth+(padWidth-0)], vl) ;
	__vr vrgout_s1 = _vel_vldu_vssl(4,&pGOutChannel[gip+(padHeight-r)*gOutWidth+(padWidth-1)], vl) ;
	__vr vrgout_s2 = _vel_vldu_vssl(4,&pGOutChannel[gip+(padHeight-r)*gOutWidth+(padWidth-2)], vl) ;
	__vr vrgout_s3 = _vel_vldu_vssl(4,&pGOutChannel[gip+(padHeight-r)*gOutWidth+(padWidth-3)], vl) ;
	__vr vrgout_s4 = _vel_vldu_vssl(4,&pGOutChannel[gip+(padHeight-r)*gOutWidth+(padWidth-4)], vl) ;

	const float *pKerValue = pKernel + kernGroupOffset + ((k * gInChannelGroup + c) * kernHeight + r) * kernWidth ;
#define VFMAD1(PKERVALUE, VRGOUT, VMR) {					\
	  VRGOUT = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), VRGOUT, VMR, vl) ;	\
	  const float kerValue = *(PKERVALUE) ;					\
	  vrsum = _vel_vfmads_vvsvl(vrsum, kerValue, VRGOUT, vl) ;		\
	}

	VFMAD1(pKerValue,vrgout_s0,vmall_s0) ; pKerValue++ ;
	VFMAD1(pKerValue,vrgout_s1,vmall_s1) ; pKerValue++ ;
	VFMAD1(pKerValue,vrgout_s2,vmall_s2) ; pKerValue++ ;
	VFMAD1(pKerValue,vrgout_s3,vmall_s3) ; pKerValue++ ;
	VFMAD1(pKerValue,vrgout_s4,vmall_s4) ; pKerValue++ ;
#undef VFMAD1
      } // gInChannel
    } // kernHeight

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

    __vr vrseq = _vel_vseq_vl(vl) ;			// hw
    __vr vridx = _vel_vaddsl_vsvl(gip, vrseq, vl) ;	// op + hw

    __vr vrsum01 = _vel_pvbrd_vsl(0UL, vl) ;

    __vr vrh   = _vel_vdivsl_vvsl(vridx, gInWidth, vl) ;
    __vr vrix  = _vel_vsubsl_vvvl(vridx, _vel_vmulul_vsvl(gInWidth,vrh, vl), vl) ;

    __vr vrx_s0 = _vel_vaddsl_vsvl(2, vrix, vl) ;
    __vr vrx_s1 = _vel_vaddsl_vsvl(1, vrix, vl) ;

    __vm256 vmx1_s3 =  _vel_vfmklge_mvl(_vel_vaddsl_vsvl(-1, vrix, vl), vl) ;
    __vm256 vmx1_s4 =  _vel_vfmklge_mvl(_vel_vaddsl_vsvl(-2, vrix, vl), vl) ;

    __vm256 vmx2_s0 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutWidth-2,vrix, vl), vl) ;
    __vm256 vmx2_s1 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutWidth-1,vrix, vl), vl) ;

    __vm256 vmx_s0  = vmx2_s0 ;
    __vm256 vmx_s1  = vmx2_s1 ;
    __vm256 vmx_s3  = vmx1_s3 ;
    __vm256 vmx_s4  = vmx1_s4 ;

    for (int64_t r=0; r<kernHeight; r++) {
      __vr vry = _vel_vaddsl_vsvl(padHeight-r, vrh, vl) ;

      __vm256 vmy1 =  _vel_vfmklge_mvl(vry, vl) ;
      __vm256 vmy2 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutHeight,vry, vl), vl) ;
      __vm256 vmy  = _vel_andm_mmm(vmy1, vmy2) ;

      __vm256 vmall_s0 = _vel_andm_mmm(vmy,vmx_s0) ;
      __vm256 vmall_s1 = _vel_andm_mmm(vmy,vmx_s1) ;
      __vm256 vmall_s2 = vmy ;
      __vm256 vmall_s3 = _vel_andm_mmm(vmy,vmx_s3) ;
      __vm256 vmall_s4 = _vel_andm_mmm(vmy,vmx_s4) ;

      for (int64_t k=0; k<gOutChannelGroup; k++) {

	const float *pGOutChannel = pGOut + gOutGroupOffset + ((n * gOutChannel + k) * gOutHeight * gOutWidth ) ;

	/* memory access errors mihgt be caused */
	__vr vrgout_s0 = _vel_vldu_vssl(4,&pGOutChannel[gip+(padHeight-r)*gOutWidth+(padWidth-0)], vl) ;
	__vr vrgout_s1 = _vel_vldu_vssl(4,&pGOutChannel[gip+(padHeight-r)*gOutWidth+(padWidth-1)], vl) ;
	__vr vrgout_s2 = _vel_vldu_vssl(4,&pGOutChannel[gip+(padHeight-r)*gOutWidth+(padWidth-2)], vl) ;
	__vr vrgout_s3 = _vel_vldu_vssl(4,&pGOutChannel[gip+(padHeight-r)*gOutWidth+(padWidth-3)], vl) ;
	__vr vrgout_s4 = _vel_vldu_vssl(4,&pGOutChannel[gip+(padHeight-r)*gOutWidth+(padWidth-4)], vl) ;

	const float *pKerValue = pKernel + kernGroupOffset + ((k * gInChannelGroup + c) * kernHeight + r) * kernWidth ;
#define PVFMAD2(PKERVALUE, VRGOUT, VMR) {							\
	  VRGOUT = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), VRGOUT, VMR, vl) ;			\
	  __vr vrgoutP = _vel_vshf_vvvsl(VRGOUT, VRGOUT, VE_VSHUFFLE_YUZU, vl) ;		\
	  const uint64_t kerValue01 = _vel_pack_f32p(PKERVALUE,					\
					             PKERVALUE+    kernHeight * kernWidth) ;	\
	  vrsum01 = _vel_pvfmad_vvsvl(vrsum01, kerValue01, vrgoutP, vl) ;			\
	}

	PVFMAD2(pKerValue,vrgout_s0,vmall_s0) ; pKerValue++ ;
	PVFMAD2(pKerValue,vrgout_s1,vmall_s1) ; pKerValue++ ;
	PVFMAD2(pKerValue,vrgout_s2,vmall_s2) ; pKerValue++ ;
	PVFMAD2(pKerValue,vrgout_s3,vmall_s3) ; pKerValue++ ;
	PVFMAD2(pKerValue,vrgout_s4,vmall_s4) ; pKerValue++ ;
#undef PVFMAD2
      } // gInChannel
    } // kernHeight

    _vel_vstu_vssl(vrsum01, 4, pGIn+gInIndex, vl) ;
    _vel_vstl_vssl(vrsum01, 4, pGIn+gInIndex+  gInPixels, vl) ;

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

    __vr vrseq = _vel_vseq_vl(vl) ;			// hw
    __vr vridx = _vel_vaddsl_vsvl(gip, vrseq, vl) ;	// op + hw

    __vr vrsum01 = _vel_pvbrd_vsl(0UL, vl) ;
    __vr vrsum23 = _vel_pvbrd_vsl(0UL, vl) ;

    __vr vrh   = _vel_vdivsl_vvsl(vridx, gInWidth, vl) ;
    __vr vrix  = _vel_vsubsl_vvvl(vridx, _vel_vmulul_vsvl(gInWidth,vrh, vl), vl) ;

    __vr vrx_s0 = _vel_vaddsl_vsvl(2, vrix, vl) ;
    __vr vrx_s1 = _vel_vaddsl_vsvl(1, vrix, vl) ;

    __vm256 vmx1_s3 =  _vel_vfmklge_mvl(_vel_vaddsl_vsvl(-1, vrix, vl), vl) ;
    __vm256 vmx1_s4 =  _vel_vfmklge_mvl(_vel_vaddsl_vsvl(-2, vrix, vl), vl) ;

    __vm256 vmx2_s0 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutWidth-2,vrix, vl), vl) ;
    __vm256 vmx2_s1 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutWidth-1,vrix, vl), vl) ;

    __vm256 vmx_s0  = vmx2_s0 ;
    __vm256 vmx_s1  = vmx2_s1 ;
    __vm256 vmx_s3  = vmx1_s3 ;
    __vm256 vmx_s4  = vmx1_s4 ;

    for (int64_t r=0; r<kernHeight; r++) {
      __vr vry = _vel_vaddsl_vsvl(padHeight-r, vrh, vl) ;

      __vm256 vmy1 =  _vel_vfmklge_mvl(vry, vl) ;
      __vm256 vmy2 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutHeight,vry, vl), vl) ;
      __vm256 vmy  = _vel_andm_mmm(vmy1, vmy2) ;

      __vm256 vmall_s0 = _vel_andm_mmm(vmy,vmx_s0) ;
      __vm256 vmall_s1 = _vel_andm_mmm(vmy,vmx_s1) ;
      __vm256 vmall_s2 = vmy ;
      __vm256 vmall_s3 = _vel_andm_mmm(vmy,vmx_s3) ;
      __vm256 vmall_s4 = _vel_andm_mmm(vmy,vmx_s4) ;

      for (int64_t k=0; k<gOutChannelGroup; k++) {

	const float *pGOutChannel = pGOut + gOutGroupOffset + ((n * gOutChannel + k) * gOutHeight * gOutWidth ) ;

	/* memory access errors mihgt be caused */
	__vr vrgout_s0 = _vel_vldu_vssl(4,&pGOutChannel[gip+(padHeight-r)*gOutWidth+(padWidth-0)], vl) ;
	__vr vrgout_s1 = _vel_vldu_vssl(4,&pGOutChannel[gip+(padHeight-r)*gOutWidth+(padWidth-1)], vl) ;
	__vr vrgout_s2 = _vel_vldu_vssl(4,&pGOutChannel[gip+(padHeight-r)*gOutWidth+(padWidth-2)], vl) ;
	__vr vrgout_s3 = _vel_vldu_vssl(4,&pGOutChannel[gip+(padHeight-r)*gOutWidth+(padWidth-3)], vl) ;
	__vr vrgout_s4 = _vel_vldu_vssl(4,&pGOutChannel[gip+(padHeight-r)*gOutWidth+(padWidth-4)], vl) ;

	const float *pKerValue = pKernel + kernGroupOffset + ((k * gInChannelGroup + c) * kernHeight + r) * kernWidth ;
#define PVFMAD4(PKERVALUE, VRGOUT, VMR) {							\
	  VRGOUT = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), VRGOUT, VMR, vl) ;			\
	  __vr vrgoutP = _vel_vshf_vvvsl(VRGOUT, VRGOUT, VE_VSHUFFLE_YUZU, vl) ;		\
	  const uint64_t kerValue01 = _vel_pack_f32p(PKERVALUE,					\
					             PKERVALUE+    kernHeight * kernWidth) ;	\
	  const uint64_t kerValue23 = _vel_pack_f32p(PKERVALUE+ 2* kernHeight * kernWidth,	\
					             PKERVALUE+ 3* kernHeight * kernWidth) ;	\
	  vrsum01 = _vel_pvfmad_vvsvl(vrsum01, kerValue01, vrgoutP, vl) ;	\
	  vrsum23 = _vel_pvfmad_vvsvl(vrsum23, kerValue23, vrgoutP, vl) ;	\
	}

	PVFMAD4(pKerValue,vrgout_s0,vmall_s0) ; pKerValue++ ;
	PVFMAD4(pKerValue,vrgout_s1,vmall_s1) ; pKerValue++ ;
	PVFMAD4(pKerValue,vrgout_s2,vmall_s2) ; pKerValue++ ;
	PVFMAD4(pKerValue,vrgout_s3,vmall_s3) ; pKerValue++ ;
	PVFMAD4(pKerValue,vrgout_s4,vmall_s4) ; pKerValue++ ;
#undef PVFMAD4
      } // gInChannel
    } // kernHeight

    _vel_vstu_vssl(vrsum01, 4, pGIn+gInIndex, vl) ;
    _vel_vstl_vssl(vrsum01, 4, pGIn+gInIndex+  gInPixels, vl) ;
    _vel_vstu_vssl(vrsum23, 4, pGIn+gInIndex+2*gInPixels, vl) ;
    _vel_vstl_vssl(vrsum23, 4, pGIn+gInIndex+3*gInPixels, vl) ;

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

    __vr vrseq = _vel_vseq_vl(vl) ;			// hw
    __vr vridx = _vel_vaddsl_vsvl(gip, vrseq, vl) ;	// op + hw

    __vr vrsum01 = _vel_pvbrd_vsl(0UL, vl) ;
    __vr vrsum23 = _vel_pvbrd_vsl(0UL, vl) ;
    __vr vrsum45 = _vel_pvbrd_vsl(0UL, vl) ;
    __vr vrsum67 = _vel_pvbrd_vsl(0UL, vl) ;

    __vr vrh   = _vel_vdivsl_vvsl(vridx, gInWidth, vl) ;
    __vr vrix  = _vel_vsubsl_vvvl(vridx, _vel_vmulul_vsvl(gInWidth,vrh, vl), vl) ;

    __vr vrx_s0 = _vel_vaddsl_vsvl(2, vrix, vl) ;
    __vr vrx_s1 = _vel_vaddsl_vsvl(1, vrix, vl) ;

    __vm256 vmx1_s3 =  _vel_vfmklge_mvl(_vel_vaddsl_vsvl(-1, vrix, vl), vl) ;
    __vm256 vmx1_s4 =  _vel_vfmklge_mvl(_vel_vaddsl_vsvl(-2, vrix, vl), vl) ;

    __vm256 vmx2_s0 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutWidth-2,vrix, vl), vl) ;
    __vm256 vmx2_s1 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutWidth-1,vrix, vl), vl) ;

    __vm256 vmx_s0  = vmx2_s0 ;
    __vm256 vmx_s1  = vmx2_s1 ;
    __vm256 vmx_s3  = vmx1_s3 ;
    __vm256 vmx_s4  = vmx1_s4 ;

    for (int64_t r=0; r<kernHeight; r++) {
      __vr vry = _vel_vaddsl_vsvl(padHeight-r, vrh, vl) ;

      __vm256 vmy1 =  _vel_vfmklge_mvl(vry, vl) ;
      __vm256 vmy2 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutHeight,vry, vl), vl) ;
      __vm256 vmy  = _vel_andm_mmm(vmy1, vmy2) ;

      __vm256 vmall_s0 = _vel_andm_mmm(vmy,vmx_s0) ;
      __vm256 vmall_s1 = _vel_andm_mmm(vmy,vmx_s1) ;
      __vm256 vmall_s2 = vmy ;
      __vm256 vmall_s3 = _vel_andm_mmm(vmy,vmx_s3) ;
      __vm256 vmall_s4 = _vel_andm_mmm(vmy,vmx_s4) ;

      for (int64_t k=0; k<gOutChannelGroup; k++) {

	const float *pGOutChannel = pGOut + gOutGroupOffset + ((n * gOutChannel + k) * gOutHeight * gOutWidth ) ;

	/* memory access errors mihgt be caused */
	__vr vrgout_s0 = _vel_vldu_vssl(4,&pGOutChannel[gip+(padHeight-r)*gOutWidth+(padWidth-0)], vl) ;
	__vr vrgout_s1 = _vel_vldu_vssl(4,&pGOutChannel[gip+(padHeight-r)*gOutWidth+(padWidth-1)], vl) ;
	__vr vrgout_s2 = _vel_vldu_vssl(4,&pGOutChannel[gip+(padHeight-r)*gOutWidth+(padWidth-2)], vl) ;
	__vr vrgout_s3 = _vel_vldu_vssl(4,&pGOutChannel[gip+(padHeight-r)*gOutWidth+(padWidth-3)], vl) ;
	__vr vrgout_s4 = _vel_vldu_vssl(4,&pGOutChannel[gip+(padHeight-r)*gOutWidth+(padWidth-4)], vl) ;

	const float *pKerValue = pKernel + kernGroupOffset + ((k * gInChannelGroup + c) * kernHeight + r) * kernWidth ;
#define PVFMAD8(PKERVALUE, VRGOUT, VMR) {							\
	  VRGOUT = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), VRGOUT, VMR, vl) ;			\
	  __vr vrgoutP = _vel_vshf_vvvsl(VRGOUT, VRGOUT, VE_VSHUFFLE_YUZU, vl) ;		\
	  const uint64_t kerValue01 = _vel_pack_f32p(PKERVALUE,					\
					             PKERVALUE+    kernHeight * kernWidth) ;	\
	  const uint64_t kerValue23 = _vel_pack_f32p(PKERVALUE+ 2* kernHeight * kernWidth,	\
					             PKERVALUE+ 3* kernHeight * kernWidth) ;	\
	  const uint64_t kerValue45 = _vel_pack_f32p(PKERVALUE+ 4* kernHeight * kernWidth,	\
					             PKERVALUE+ 5* kernHeight * kernWidth) ;	\
	  const uint64_t kerValue67 = _vel_pack_f32p(PKERVALUE+ 6* kernHeight * kernWidth,	\
					             PKERVALUE+ 7* kernHeight * kernWidth) ;	\
	  vrsum01 = _vel_pvfmad_vvsvl(vrsum01, kerValue01, vrgoutP, vl) ;	\
	  vrsum23 = _vel_pvfmad_vvsvl(vrsum23, kerValue23, vrgoutP, vl) ;	\
	  vrsum45 = _vel_pvfmad_vvsvl(vrsum45, kerValue45, vrgoutP, vl) ;	\
	  vrsum67 = _vel_pvfmad_vvsvl(vrsum67, kerValue67, vrgoutP, vl) ;	\
	}

	PVFMAD8(pKerValue,vrgout_s0,vmall_s0) ; pKerValue++ ;
	PVFMAD8(pKerValue,vrgout_s1,vmall_s1) ; pKerValue++ ;
	PVFMAD8(pKerValue,vrgout_s2,vmall_s2) ; pKerValue++ ;
	PVFMAD8(pKerValue,vrgout_s3,vmall_s3) ; pKerValue++ ;
	PVFMAD8(pKerValue,vrgout_s4,vmall_s4) ; pKerValue++ ;
#undef PVFMAD8
      } // gInChannel
    } // kernHeight

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
    __vr vrix  = _vel_vsubsl_vvvl(vridx, _vel_vmulul_vsvl(gInWidth,vrh, vl), vl) ;

    __vr vrx_s0 = _vel_vaddsl_vsvl(2, vrix, vl) ;
    __vr vrx_s1 = _vel_vaddsl_vsvl(1, vrix, vl) ;

    __vm256 vmx1_s3 =  _vel_vfmklge_mvl(_vel_vaddsl_vsvl(-1, vrix, vl), vl) ;
    __vm256 vmx1_s4 =  _vel_vfmklge_mvl(_vel_vaddsl_vsvl(-2, vrix, vl), vl) ;

    __vm256 vmx2_s0 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutWidth-2,vrix, vl), vl) ;
    __vm256 vmx2_s1 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutWidth-1,vrix, vl), vl) ;

    __vm256 vmx_s0  = vmx2_s0 ;
    __vm256 vmx_s1  = vmx2_s1 ;
    __vm256 vmx_s3  = vmx1_s3 ;
    __vm256 vmx_s4  = vmx1_s4 ;

    for (int64_t r=0; r<kernHeight; r++) {
      __vr vry = _vel_vaddsl_vsvl(padHeight-r, vrh, vl) ;

      __vm256 vmy1 =  _vel_vfmklge_mvl(vry, vl) ;
      __vm256 vmy2 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutHeight,vry, vl), vl) ;
      __vm256 vmy  = _vel_andm_mmm(vmy1, vmy2) ;

      __vm256 vmall_s0 = _vel_andm_mmm(vmy,vmx_s0) ;
      __vm256 vmall_s1 = _vel_andm_mmm(vmy,vmx_s1) ;
      __vm256 vmall_s2 = vmy ;
      __vm256 vmall_s3 = _vel_andm_mmm(vmy,vmx_s3) ;
      __vm256 vmall_s4 = _vel_andm_mmm(vmy,vmx_s4) ;

      for (int64_t k=0; k<gOutChannelGroup; k++) {

	const float *pGOutChannel = pGOut + gOutGroupOffset + ((n * gOutChannel + k) * gOutHeight * gOutWidth ) ;

	/* memory access errors mihgt be caused */
	__vr vrgout_s0 = _vel_vldu_vssl(4,&pGOutChannel[gip+(padHeight-r)*gOutWidth+(padWidth-0)], vl) ;
	__vr vrgout_s1 = _vel_vldu_vssl(4,&pGOutChannel[gip+(padHeight-r)*gOutWidth+(padWidth-1)], vl) ;
	__vr vrgout_s2 = _vel_vldu_vssl(4,&pGOutChannel[gip+(padHeight-r)*gOutWidth+(padWidth-2)], vl) ;
	__vr vrgout_s3 = _vel_vldu_vssl(4,&pGOutChannel[gip+(padHeight-r)*gOutWidth+(padWidth-3)], vl) ;
	__vr vrgout_s4 = _vel_vldu_vssl(4,&pGOutChannel[gip+(padHeight-r)*gOutWidth+(padWidth-4)], vl) ;

	const float *pKerValue = pKernel + kernGroupOffset + ((k * gInChannelGroup + c) * kernHeight + r) * kernWidth ;
#define PVFMAD16(PKERVALUE, VRGOUT, VMR) {							\
	  VRGOUT = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), VRGOUT, VMR, vl) ;			\
	  __vr vrgoutP = _vel_vshf_vvvsl(VRGOUT, VRGOUT, VE_VSHUFFLE_YUZU, vl) ;		\
	  const uint64_t kerValue01 = _vel_pack_f32p(PKERVALUE,					\
					             PKERVALUE+    kernHeight * kernWidth) ;	\
	  const uint64_t kerValue23 = _vel_pack_f32p(PKERVALUE+ 2* kernHeight * kernWidth,	\
					             PKERVALUE+ 3* kernHeight * kernWidth) ;	\
	  const uint64_t kerValue45 = _vel_pack_f32p(PKERVALUE+ 4* kernHeight * kernWidth,	\
					             PKERVALUE+ 5* kernHeight * kernWidth) ;	\
	  const uint64_t kerValue67 = _vel_pack_f32p(PKERVALUE+ 6* kernHeight * kernWidth,	\
					             PKERVALUE+ 7* kernHeight * kernWidth) ;	\
	  const uint64_t kerValue89 = _vel_pack_f32p(PKERVALUE+ 8* kernHeight * kernWidth,	\
					             PKERVALUE+ 9* kernHeight * kernWidth) ;	\
	  const uint64_t kerValueAB = _vel_pack_f32p(PKERVALUE+10* kernHeight * kernWidth,	\
					             PKERVALUE+11* kernHeight * kernWidth) ;	\
	  const uint64_t kerValueCD = _vel_pack_f32p(PKERVALUE+12* kernHeight * kernWidth,	\
					             PKERVALUE+13* kernHeight * kernWidth) ;	\
	  const uint64_t kerValueEF = _vel_pack_f32p(PKERVALUE+14* kernHeight * kernWidth,	\
					             PKERVALUE+15* kernHeight * kernWidth) ;	\
	  vrsum01 = _vel_pvfmad_vvsvl(vrsum01, kerValue01, vrgoutP, vl) ;	\
	  vrsum23 = _vel_pvfmad_vvsvl(vrsum23, kerValue23, vrgoutP, vl) ;	\
	  vrsum45 = _vel_pvfmad_vvsvl(vrsum45, kerValue45, vrgoutP, vl) ;	\
	  vrsum67 = _vel_pvfmad_vvsvl(vrsum67, kerValue67, vrgoutP, vl) ;	\
	  vrsum89 = _vel_pvfmad_vvsvl(vrsum89, kerValue89, vrgoutP, vl) ;	\
	  vrsumAB = _vel_pvfmad_vvsvl(vrsumAB, kerValueAB, vrgoutP, vl) ;	\
	  vrsumCD = _vel_pvfmad_vvsvl(vrsumCD, kerValueCD, vrgoutP, vl) ;	\
	  vrsumEF = _vel_pvfmad_vvsvl(vrsumEF, kerValueEF, vrgoutP, vl) ;	\
        }

	PVFMAD16(pKerValue,vrgout_s0,vmall_s0) ; pKerValue++ ;
	PVFMAD16(pKerValue,vrgout_s1,vmall_s1) ; pKerValue++ ;
	PVFMAD16(pKerValue,vrgout_s2,vmall_s2) ; pKerValue++ ;
	PVFMAD16(pKerValue,vrgout_s3,vmall_s3) ; pKerValue++ ;
	PVFMAD16(pKerValue,vrgout_s4,vmall_s4) ; pKerValue++ ;
#undef PVFMAD16
      } // gInChannel
    } // kernHeight

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
    __vr vrsumGH = _vel_pvbrd_vsl(0UL, vl) ;
    __vr vrsumIJ = _vel_pvbrd_vsl(0UL, vl) ;
    __vr vrsumKL = _vel_pvbrd_vsl(0UL, vl) ;
    __vr vrsumMN = _vel_pvbrd_vsl(0UL, vl) ;
    __vr vrsumOP = _vel_pvbrd_vsl(0UL, vl) ;
    __vr vrsumQR = _vel_pvbrd_vsl(0UL, vl) ;
    __vr vrsumST = _vel_pvbrd_vsl(0UL, vl) ;
    __vr vrsumUV = _vel_pvbrd_vsl(0UL, vl) ;

    __vr vrh   = _vel_vdivsl_vvsl(vridx, gInWidth, vl) ;
    __vr vrix  = _vel_vsubsl_vvvl(vridx, _vel_vmulul_vsvl(gInWidth,vrh, vl), vl) ;

    __vr vrx_s0 = _vel_vaddsl_vsvl(2, vrix, vl) ;
    __vr vrx_s1 = _vel_vaddsl_vsvl(1, vrix, vl) ;

    __vm256 vmx1_s3 =  _vel_vfmklge_mvl(_vel_vaddsl_vsvl(-1, vrix, vl), vl) ;
    __vm256 vmx1_s4 =  _vel_vfmklge_mvl(_vel_vaddsl_vsvl(-2, vrix, vl), vl) ;

    __vm256 vmx2_s0 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutWidth-2,vrix, vl), vl) ;
    __vm256 vmx2_s1 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutWidth-1,vrix, vl), vl) ;

    __vm256 vmx_s0  = vmx2_s0 ;
    __vm256 vmx_s1  = vmx2_s1 ;
    __vm256 vmx_s3  = vmx1_s3 ;
    __vm256 vmx_s4  = vmx1_s4 ;

    for (int64_t r=0; r<kernHeight; r++) {
      __vr vry = _vel_vaddsl_vsvl(padHeight-r, vrh, vl) ;

      __vm256 vmy1 =  _vel_vfmklge_mvl(vry, vl) ;
      __vm256 vmy2 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutHeight,vry, vl), vl) ;
      __vm256 vmy  = _vel_andm_mmm(vmy1, vmy2) ;

      __vm256 vmall_s0 = _vel_andm_mmm(vmy,vmx_s0) ;
      __vm256 vmall_s1 = _vel_andm_mmm(vmy,vmx_s1) ;
      __vm256 vmall_s2 = vmy ;
      __vm256 vmall_s3 = _vel_andm_mmm(vmy,vmx_s3) ;
      __vm256 vmall_s4 = _vel_andm_mmm(vmy,vmx_s4) ;

      for (int64_t k=0; k<gOutChannelGroup; k++) {

	const float *pGOutChannel = pGOut + gOutGroupOffset + ((n * gOutChannel + k) * gOutHeight * gOutWidth ) ;

	/* memory access errors mihgt be caused */
	__vr vrgout_s0 = _vel_vldu_vssl(4,&pGOutChannel[gip+(padHeight-r)*gOutWidth+(padWidth-0)], vl) ;
	__vr vrgout_s1 = _vel_vldu_vssl(4,&pGOutChannel[gip+(padHeight-r)*gOutWidth+(padWidth-1)], vl) ;
	__vr vrgout_s2 = _vel_vldu_vssl(4,&pGOutChannel[gip+(padHeight-r)*gOutWidth+(padWidth-2)], vl) ;
	__vr vrgout_s3 = _vel_vldu_vssl(4,&pGOutChannel[gip+(padHeight-r)*gOutWidth+(padWidth-3)], vl) ;
	__vr vrgout_s4 = _vel_vldu_vssl(4,&pGOutChannel[gip+(padHeight-r)*gOutWidth+(padWidth-4)], vl) ;

	const float *pKerValue = pKernel + kernGroupOffset + ((k * gInChannelGroup + c) * kernHeight + r) * kernWidth ;
#define PVFMAD32(PKERVALUE, VRGOUT, VMR) {							\
	  VRGOUT = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), VRGOUT, VMR, vl) ;			\
	  __vr vrgoutP = _vel_vshf_vvvsl(VRGOUT, VRGOUT, VE_VSHUFFLE_YUZU, vl) ;		\
	  const uint64_t kerValue01 = _vel_pack_f32p(PKERVALUE,					\
					             PKERVALUE+    kernHeight * kernWidth) ;	\
	  const uint64_t kerValue23 = _vel_pack_f32p(PKERVALUE+ 2* kernHeight * kernWidth,	\
					             PKERVALUE+ 3* kernHeight * kernWidth) ;	\
	  const uint64_t kerValue45 = _vel_pack_f32p(PKERVALUE+ 4* kernHeight * kernWidth,	\
					             PKERVALUE+ 5* kernHeight * kernWidth) ;	\
	  const uint64_t kerValue67 = _vel_pack_f32p(PKERVALUE+ 6* kernHeight * kernWidth,	\
					             PKERVALUE+ 7* kernHeight * kernWidth) ;	\
	  const uint64_t kerValue89 = _vel_pack_f32p(PKERVALUE+ 8* kernHeight * kernWidth,	\
					             PKERVALUE+ 9* kernHeight * kernWidth) ;	\
	  const uint64_t kerValueAB = _vel_pack_f32p(PKERVALUE+10* kernHeight * kernWidth,	\
					             PKERVALUE+11* kernHeight * kernWidth) ;	\
	  const uint64_t kerValueCD = _vel_pack_f32p(PKERVALUE+12* kernHeight * kernWidth,	\
					             PKERVALUE+13* kernHeight * kernWidth) ;	\
	  const uint64_t kerValueEF = _vel_pack_f32p(PKERVALUE+14* kernHeight * kernWidth,	\
					             PKERVALUE+15* kernHeight * kernWidth) ;	\
	  vrsum01 = _vel_pvfmad_vvsvl(vrsum01, kerValue01, vrgoutP, vl) ;	\
	  vrsum23 = _vel_pvfmad_vvsvl(vrsum23, kerValue23, vrgoutP, vl) ;	\
	  vrsum45 = _vel_pvfmad_vvsvl(vrsum45, kerValue45, vrgoutP, vl) ;	\
	  vrsum67 = _vel_pvfmad_vvsvl(vrsum67, kerValue67, vrgoutP, vl) ;	\
	  vrsum89 = _vel_pvfmad_vvsvl(vrsum89, kerValue89, vrgoutP, vl) ;	\
	  vrsumAB = _vel_pvfmad_vvsvl(vrsumAB, kerValueAB, vrgoutP, vl) ;	\
	  vrsumCD = _vel_pvfmad_vvsvl(vrsumCD, kerValueCD, vrgoutP, vl) ;	\
	  vrsumEF = _vel_pvfmad_vvsvl(vrsumEF, kerValueEF, vrgoutP, vl) ;	\
	  const uint64_t kerValueGH = _vel_pack_f32p(PKERVALUE+16* kernHeight * kernWidth,	\
					             PKERVALUE+17* kernHeight * kernWidth) ;	\
	  const uint64_t kerValueIJ = _vel_pack_f32p(PKERVALUE+18* kernHeight * kernWidth,	\
					             PKERVALUE+19* kernHeight * kernWidth) ;	\
	  const uint64_t kerValueKL = _vel_pack_f32p(PKERVALUE+20* kernHeight * kernWidth,	\
					             PKERVALUE+21* kernHeight * kernWidth) ;	\
	  const uint64_t kerValueMN = _vel_pack_f32p(PKERVALUE+22* kernHeight * kernWidth,	\
					             PKERVALUE+23* kernHeight * kernWidth) ;	\
	  const uint64_t kerValueOP = _vel_pack_f32p(PKERVALUE+24* kernHeight * kernWidth,	\
					             PKERVALUE+25* kernHeight * kernWidth) ;	\
	  const uint64_t kerValueQR = _vel_pack_f32p(PKERVALUE+26* kernHeight * kernWidth,	\
					             PKERVALUE+27* kernHeight * kernWidth) ;	\
	  const uint64_t kerValueST = _vel_pack_f32p(PKERVALUE+28* kernHeight * kernWidth,	\
					             PKERVALUE+29* kernHeight * kernWidth) ;	\
	  const uint64_t kerValueUV = _vel_pack_f32p(PKERVALUE+30* kernHeight * kernWidth,	\
					             PKERVALUE+31* kernHeight * kernWidth) ;	\
	  vrsumGH = _vel_pvfmad_vvsvl(vrsumGH, kerValueGH, vrgoutP, vl) ;	\
	  vrsumIJ = _vel_pvfmad_vvsvl(vrsumIJ, kerValueIJ, vrgoutP, vl) ;	\
	  vrsumKL = _vel_pvfmad_vvsvl(vrsumKL, kerValueKL, vrgoutP, vl) ;	\
	  vrsumMN = _vel_pvfmad_vvsvl(vrsumMN, kerValueMN, vrgoutP, vl) ;	\
	  vrsumOP = _vel_pvfmad_vvsvl(vrsumOP, kerValueOP, vrgoutP, vl) ;	\
	  vrsumQR = _vel_pvfmad_vvsvl(vrsumQR, kerValueQR, vrgoutP, vl) ;	\
	  vrsumST = _vel_pvfmad_vvsvl(vrsumST, kerValueST, vrgoutP, vl) ;	\
	  vrsumUV = _vel_pvfmad_vvsvl(vrsumUV, kerValueUV, vrgoutP, vl) ;	\
        }

	PVFMAD32(pKerValue,vrgout_s0,vmall_s0) ; pKerValue++ ;
	PVFMAD32(pKerValue,vrgout_s1,vmall_s1) ; pKerValue++ ;
	PVFMAD32(pKerValue,vrgout_s2,vmall_s2) ; pKerValue++ ;
	PVFMAD32(pKerValue,vrgout_s3,vmall_s3) ; pKerValue++ ;
	PVFMAD32(pKerValue,vrgout_s4,vmall_s4) ; pKerValue++ ;
#undef PVFMAD32
      } // gInChannel
    } // kernHeight

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
    _vel_vstu_vssl(vrsumGH, 4, pGIn+gInIndex+16*gInPixels, vl) ;
    _vel_vstl_vssl(vrsumGH, 4, pGIn+gInIndex+17*gInPixels, vl) ;
    _vel_vstu_vssl(vrsumIJ, 4, pGIn+gInIndex+18*gInPixels, vl) ;
    _vel_vstl_vssl(vrsumIJ, 4, pGIn+gInIndex+19*gInPixels, vl) ;
    _vel_vstu_vssl(vrsumKL, 4, pGIn+gInIndex+20*gInPixels, vl) ;
    _vel_vstl_vssl(vrsumKL, 4, pGIn+gInIndex+21*gInPixels, vl) ;
    _vel_vstu_vssl(vrsumMN, 4, pGIn+gInIndex+22*gInPixels, vl) ;
    _vel_vstl_vssl(vrsumMN, 4, pGIn+gInIndex+23*gInPixels, vl) ;
    _vel_vstu_vssl(vrsumOP, 4, pGIn+gInIndex+24*gInPixels, vl) ;
    _vel_vstl_vssl(vrsumOP, 4, pGIn+gInIndex+25*gInPixels, vl) ;
    _vel_vstu_vssl(vrsumQR, 4, pGIn+gInIndex+26*gInPixels, vl) ;
    _vel_vstl_vssl(vrsumQR, 4, pGIn+gInIndex+27*gInPixels, vl) ;
    _vel_vstu_vssl(vrsumST, 4, pGIn+gInIndex+28*gInPixels, vl) ;
    _vel_vstl_vssl(vrsumST, 4, pGIn+gInIndex+29*gInPixels, vl) ;
    _vel_vstu_vssl(vrsumUV, 4, pGIn+gInIndex+30*gInPixels, vl) ;
    _vel_vstl_vssl(vrsumUV, 4, pGIn+gInIndex+31*gInPixels, vl) ;


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
