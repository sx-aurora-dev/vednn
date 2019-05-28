#include <stdint.h>
#include <stdlib.h>

#include "vednn.h"

#include "veintrin.h"
#define VLEN	(256)

template<int R_UNROLL, int S_UNROLL>
static inline void f1(
    const __vr vrij,
    const float * pIn, const int64_t inWidth, const int64_t inHeight,
    const float * pGOut, const int64_t gOutWidth, const int64_t gOutHeight,
    float * const pGKernel, const int64_t gKernWidth , const int64_t gKernHeight,
    const int64_t strideHeight,
    const int64_t inChannelGroup, const int64_t inChannel, const int64_t gOutChannel,
    const int64_t inGroupOffset, const int64_t outGroupOffset, const int64_t kernGroupOffset,
    const int64_t nY,
    const int64_t batch, const int64_t k, const int64_t c,
    const int64_t r, const int64_t s
)
{
  const int64_t kernelIndex = kernGroupOffset + ((k     * inChannelGroup + c) * gKernHeight + r) * gKernWidth + s;

  _ve_lvl(VLEN) ;

#define INIT_VRSUM(R, S, RS)			\
  __vr vrsum_##RS ;				\
  if( (R_UNROLL > R) && (S_UNROLL > S) ) {	\
    vrsum_##RS = _ve_vbrdu_vs_f32(0.f) ;	\
  }

  INIT_VRSUM(0,0,r0s0) ;
  INIT_VRSUM(0,1,r0s1) ;
  INIT_VRSUM(0,2,r0s2) ;
  INIT_VRSUM(1,0,r1s0) ;
  INIT_VRSUM(1,1,r1s1) ;
  INIT_VRSUM(1,2,r1s2) ;
  INIT_VRSUM(2,0,r2s0) ;
  INIT_VRSUM(2,1,r2s1) ;
  INIT_VRSUM(2,2,r2s2) ;
#undef INIT_VRSUM

  for (int64_t n=0; n<batch; n++) {
    for (int64_t y=0; y<gOutHeight; y+=nY) {

      const int64_t vl = gOutWidth * (gOutHeight - y < nY ? gOutHeight - y : nY) ;

      _ve_lvl(vl) ;

      const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;

      const int64_t gOutIndex  = outGroupOffset + ((n * gOutChannel + k  ) * gOutHeight + y) * gOutWidth ;

      __vr vrpin = _ve_vsfa_vvss(vrij,2,(int64_t)(pInChannel+(y*strideHeight+r)*inWidth+s)) ;
      __vr vrin_r0s0 = _ve_vgtu_vv(vrpin) ;

#define VLOAD_VRIN(R, S, RS)			\
  __vr vrin_##RS ;				\
  if( (R_UNROLL > R) && (S_UNROLL > S) ) {	\
    vrin_##RS = _ve_vgtu_vv(_ve_vaddsl_vsv(4*(R*inWidth+S), vrpin)) ;	\
  }						\

      VLOAD_VRIN(0,1,r0s1) ;
      VLOAD_VRIN(0,2,r0s2) ;
      VLOAD_VRIN(1,0,r1s0) ;
      VLOAD_VRIN(1,1,r1s1) ;
      VLOAD_VRIN(1,2,r1s2) ;
      VLOAD_VRIN(2,0,r2s0) ;
      VLOAD_VRIN(2,1,r2s1) ;
      VLOAD_VRIN(2,2,r2s2) ;
#undef VLOAD_VRIN

      __vr vrgout = _ve_vldu_vss(4, pGOut+gOutIndex) ;

#define VFMAD(R,S,RS)							\
  if( (R_UNROLL > R) && (S_UNROLL > S) ) {				\
    vrsum_##RS = _ve_vfmads_vvvv(vrsum_##RS, vrin_##RS, vrgout) ;	\
  }

      VFMAD(0,0,r0s0) ;
      VFMAD(0,1,r0s1) ;
      VFMAD(0,2,r0s2) ;
      VFMAD(1,0,r1s0) ;
      VFMAD(1,1,r1s1) ;
      VFMAD(1,2,r1s2) ;
      VFMAD(2,0,r2s0) ;
      VFMAD(2,1,r2s1) ;
      VFMAD(2,2,r2s2) ;
#undef VFMAD
    } // gOutPixel
  } // batch

  _ve_lvl(VLEN) ;

#define VSUM(R,S,RS)							\
  if( (R_UNROLL > R) && (S_UNROLL > S) ) {				\
    vrsum_##RS = _ve_vfsums_vv(vrsum_##RS) ;				\
  }
  VSUM(0,0,r0s0) ;
  VSUM(0,1,r0s1) ;
  VSUM(0,2,r0s2) ;
  VSUM(1,0,r1s0) ;
  VSUM(1,1,r1s1) ;
  VSUM(1,2,r1s2) ;
  VSUM(2,0,r2s0) ;
  VSUM(2,1,r2s1) ;
  VSUM(2,2,r2s2) ;
#undef VSUM

  _ve_lvl(1) ;
#define VST(R,S,RS)							\
  if( (R_UNROLL > R) && (S_UNROLL > S) ) {				\
    _ve_vstu_vss(vrsum_##RS, 4, pGKernel+kernelIndex+R*gKernWidth+S) ;	\
  }

  VST(0,0,r0s0) ;
  VST(0,1,r0s1) ;
  VST(0,2,r0s2) ;
  VST(1,0,r1s0) ;
  VST(1,1,r1s1) ;
  VST(1,2,r1s2) ;
  VST(2,0,r2s0) ;
  VST(2,1,r2s1) ;
  VST(2,2,r2s2) ;

#undef VST

}


template<int R_UNROLL, int S_UNROLL>
static inline void f2(
  const __vr vrij,
  const float * pIn, const int64_t inWidth, const int64_t inHeight,
  const float * pGOut, const int64_t gOutWidth, const int64_t gOutHeight,
  float * const pGKernel, const int64_t gKernWidth , const int64_t gKernHeight,
  const int64_t strideHeight,
  const int64_t inChannelGroup, const int64_t inChannel, const int64_t gOutChannel,
  const int64_t inGroupOffset, const int64_t outGroupOffset, const int64_t kernGroupOffset,
  const int64_t nY,
  const int64_t batch, const int64_t k, const int64_t c,
  const int64_t r, const int64_t s
)
{
  const int64_t kernelIndex0 = kernGroupOffset + ((k     * inChannelGroup + c) * gKernHeight + r) * gKernWidth + s;
  const int64_t kernelIndex1 = kernGroupOffset + (((k+1) * inChannelGroup + c) * gKernHeight + r) * gKernWidth + s;

  _ve_lvl(VLEN) ;

#define INIT_VRSUM2(R, S, RS)			\
  __vr vrsum01_##RS ;				\
  if( (R_UNROLL > R) && (S_UNROLL > S) ) {	\
    vrsum01_##RS = _ve_vbrd_vs_i64(0UL) ;	\
  }

  INIT_VRSUM2(0,0,r0s0) ;
  INIT_VRSUM2(0,1,r0s1) ;
  INIT_VRSUM2(0,2,r0s2) ;
  INIT_VRSUM2(1,0,r1s0) ;
  INIT_VRSUM2(1,1,r1s1) ;
  INIT_VRSUM2(1,2,r1s2) ;
  INIT_VRSUM2(2,0,r2s0) ;
  INIT_VRSUM2(2,1,r2s1) ;
  INIT_VRSUM2(2,2,r2s2) ;
  INIT_VRSUM2(2,3,r2s3) ;
#undef INIT_VRSUM2

  for (int64_t n=0; n<batch; n++) {
    for (int64_t y=0; y<gOutHeight; y+=nY) {

      const int64_t vl = gOutWidth * (gOutHeight - y < nY ? gOutHeight - y : nY) ;

      _ve_lvl(vl) ;

      const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;

      const int64_t gOutIndex0  = outGroupOffset + ((n * gOutChannel + k  ) * gOutHeight + y) * gOutWidth ;
      const int64_t gOutIndex1  = outGroupOffset + ((n * gOutChannel + k+1) * gOutHeight + y) * gOutWidth ;

      __vr vrpin = _ve_vsfa_vvss(vrij,2,(int64_t)(pInChannel+(y*strideHeight+r)*inWidth+s)) ;
      __vr vrin_r0s0 = _ve_vgtu_vv(vrpin) ;

#define VLOAD_VRIN(R, S, RS)			\
  __vr vrin_##RS ;				\
  if( (R_UNROLL > R) && (S_UNROLL > S) ) {	\
    vrin_##RS = _ve_vgtu_vv(_ve_vaddsl_vsv(4*(R*inWidth+S), vrpin)) ;	\
  }						\

      VLOAD_VRIN(0,1,r0s1) ;
      VLOAD_VRIN(0,2,r0s2) ;
      VLOAD_VRIN(1,0,r1s0) ;
      VLOAD_VRIN(1,1,r1s1) ;
      VLOAD_VRIN(1,2,r1s2) ;
      VLOAD_VRIN(2,0,r2s0) ;
      VLOAD_VRIN(2,1,r2s1) ;
      VLOAD_VRIN(2,2,r2s2) ;
#undef VLOAD_VRIN

      __vr vrgout0 = _ve_vldu_vss(4, pGOut+gOutIndex0) ;
      __vr vrgout1 = _ve_vldu_vss(4, pGOut+gOutIndex1) ;

      __vr vrgout01 = _ve_vshf_vvvs(vrgout0, vrgout1, VE_VSHUFFLE_YUZU) ;

#define VFMAD2(R,S,RS)								\
  if( (R_UNROLL > R) && (S_UNROLL > S) ) {					\
    __vr vrinP_##RS = _ve_vshf_vvvs(vrin_##RS, vrin_##RS, VE_VSHUFFLE_YUZU) ;	\
    vrsum01_##RS = _ve_pvfmad_vvvv(vrsum01_##RS, vrinP_##RS, vrgout01) ;	\
  }

      VFMAD2(0,0,r0s0) ;
      VFMAD2(0,1,r0s1) ;
      VFMAD2(0,2,r0s2) ;
      VFMAD2(1,0,r1s0) ;
      VFMAD2(1,1,r1s1) ;
      VFMAD2(1,2,r1s2) ;
      VFMAD2(2,0,r2s0) ;
      VFMAD2(2,1,r2s1) ;
      VFMAD2(2,2,r2s2) ;
#undef VFMAD2

    } // gOutPixels
  } // batch

#define VSUM_AND_VST2(R,S,RS)						\
  if( (R_UNROLL > R) && (S_UNROLL > S) ) {				\
    _ve_lvl(VLEN) ;							\
    __vr vrsum0 = _ve_vfsums_vv(vrsum01_##RS) ;				\
    __vr vrsum1 = _ve_vfsums_vv(_ve_vsll_vvs(vrsum01_##RS,32));		\
    _ve_lvl(1) ;							\
    _ve_vstu_vss(vrsum0, 4, pGKernel+kernelIndex0+R*gKernWidth+S) ;	\
    _ve_vstu_vss(vrsum1, 4, pGKernel+kernelIndex1+R*gKernWidth+S) ;	\
  }

  VSUM_AND_VST2(0,0,r0s0) ;
  VSUM_AND_VST2(0,1,r0s1) ;
  VSUM_AND_VST2(0,2,r0s2) ;
  VSUM_AND_VST2(1,0,r1s0) ;
  VSUM_AND_VST2(1,1,r1s1) ;
  VSUM_AND_VST2(1,2,r1s2) ;
  VSUM_AND_VST2(2,0,r2s0) ;
  VSUM_AND_VST2(2,1,r2s1) ;
  VSUM_AND_VST2(2,2,r2s2) ;

#undef VSUM_AND_VST2

}

template<int R_UNROLL, int S_UNROLL>
static inline void f4(
  const __vr vrij,
  const float * pIn, const int64_t inWidth, const int64_t inHeight,
  const float * pGOut, const int64_t gOutWidth, const int64_t gOutHeight,
  float * const pGKernel, const int64_t gKernWidth , const int64_t gKernHeight,
  const int64_t strideHeight,
  const int64_t inChannelGroup, const int64_t inChannel, const int64_t gOutChannel,
  const int64_t inGroupOffset, const int64_t outGroupOffset, const int64_t kernGroupOffset,
  const int64_t nY,
  const int64_t batch, const int64_t k, const int64_t c,
  const int64_t r, const int64_t s
)
{
  const int64_t kernelIndex0 = kernGroupOffset + ((k     * inChannelGroup + c) * gKernHeight + r) * gKernWidth + s;
  const int64_t kernelIndex1 = kernGroupOffset + (((k+1) * inChannelGroup + c) * gKernHeight + r) * gKernWidth + s;
  const int64_t kernelIndex2 = kernGroupOffset + (((k+2) * inChannelGroup + c) * gKernHeight + r) * gKernWidth + s;
  const int64_t kernelIndex3 = kernGroupOffset + (((k+3) * inChannelGroup + c) * gKernHeight + r) * gKernWidth + s;

  _ve_lvl(VLEN) ;

#define INIT_VRSUM4(R, S, RS)			\
  __vr vrsum01_##RS ;				\
  __vr vrsum23_##RS ;				\
  if( (R_UNROLL > R) && (S_UNROLL > S) ) {	\
    vrsum01_##RS = _ve_vbrd_vs_i64(0UL) ;	\
    vrsum23_##RS = _ve_vbrd_vs_i64(0UL) ;	\
  }

  INIT_VRSUM4(0,0,r0s0) ;
  INIT_VRSUM4(0,1,r0s1) ;
  INIT_VRSUM4(0,2,r0s2) ;
  INIT_VRSUM4(1,0,r1s0) ;
  INIT_VRSUM4(1,1,r1s1) ;
  INIT_VRSUM4(1,2,r1s2) ;
  INIT_VRSUM4(2,0,r2s0) ;
  INIT_VRSUM4(2,1,r2s1) ;
  INIT_VRSUM4(2,2,r2s2) ;
  INIT_VRSUM4(2,3,r2s3) ;
#undef INIT_VRSUM4

  for (int64_t n=0; n<batch; n++) {
    for (int64_t y=0; y<gOutHeight; y+=nY) {

      const int64_t vl = gOutWidth * (gOutHeight - y < nY ? gOutHeight - y : nY) ;

      _ve_lvl(vl) ;

      const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;

      const int64_t gOutIndex0  = outGroupOffset + ((n * gOutChannel + k  ) * gOutHeight + y) * gOutWidth ;
      const int64_t gOutIndex1  = outGroupOffset + ((n * gOutChannel + k+1) * gOutHeight + y) * gOutWidth ;
      const int64_t gOutIndex2  = outGroupOffset + ((n * gOutChannel + k+2) * gOutHeight + y) * gOutWidth ;
      const int64_t gOutIndex3  = outGroupOffset + ((n * gOutChannel + k+3) * gOutHeight + y) * gOutWidth ;

      __vr vrpin = _ve_vsfa_vvss(vrij,2,(int64_t)(pInChannel+(y*strideHeight+r)*inWidth+s)) ;
      __vr vrin_r0s0 = _ve_vgtu_vv(vrpin) ;

#define VLOAD_VRIN(R, S, RS)			\
  __vr vrin_##RS ;				\
  if( (R_UNROLL > R) && (S_UNROLL > S) ) {	\
    vrin_##RS = _ve_vgtu_vv(_ve_vaddsl_vsv(4*(R*inWidth+S), vrpin)) ;	\
  }						\

      VLOAD_VRIN(0,1,r0s1) ;
      VLOAD_VRIN(0,2,r0s2) ;
      VLOAD_VRIN(1,0,r1s0) ;
      VLOAD_VRIN(1,1,r1s1) ;
      VLOAD_VRIN(1,2,r1s2) ;
      VLOAD_VRIN(2,0,r2s0) ;
      VLOAD_VRIN(2,1,r2s1) ;
      VLOAD_VRIN(2,2,r2s2) ;
#undef VLOAD_VRIN

      __vr vrgout0 = _ve_vldu_vss(4, pGOut+gOutIndex0) ;
      __vr vrgout1 = _ve_vldu_vss(4, pGOut+gOutIndex1) ;
      __vr vrgout2 = _ve_vldu_vss(4, pGOut+gOutIndex2) ;
      __vr vrgout3 = _ve_vldu_vss(4, pGOut+gOutIndex3) ;

      __vr vrgout01 = _ve_vshf_vvvs(vrgout0, vrgout1, VE_VSHUFFLE_YUZU) ;
      __vr vrgout23 = _ve_vshf_vvvs(vrgout2, vrgout3, VE_VSHUFFLE_YUZU) ;

#define VFMAD4(R,S,RS)								\
  if( (R_UNROLL > R) && (S_UNROLL > S) ) {					\
    __vr vrinP_##RS = _ve_vshf_vvvs(vrin_##RS, vrin_##RS, VE_VSHUFFLE_YUZU) ;	\
    vrsum01_##RS = _ve_pvfmad_vvvv(vrsum01_##RS, vrinP_##RS, vrgout01) ;	\
    vrsum23_##RS = _ve_pvfmad_vvvv(vrsum23_##RS, vrinP_##RS, vrgout23) ;	\
  }

      VFMAD4(0,0,r0s0) ;
      VFMAD4(0,1,r0s1) ;
      VFMAD4(0,2,r0s2) ;
      VFMAD4(1,0,r1s0) ;
      VFMAD4(1,1,r1s1) ;
      VFMAD4(1,2,r1s2) ;
      VFMAD4(2,0,r2s0) ;
      VFMAD4(2,1,r2s1) ;
      VFMAD4(2,2,r2s2) ;
#undef VFMAD4

    } // gOutPixels
  } // batch

#define VSUM_AND_VST4(R,S,RS)						\
  if( (R_UNROLL > R) && (S_UNROLL > S) ) {				\
    _ve_lvl(VLEN) ;							\
    __vr vrsum0 = _ve_vfsums_vv(vrsum01_##RS) ;				\
    __vr vrsum1 = _ve_vfsums_vv(_ve_vsll_vvs(vrsum01_##RS,32));		\
    __vr vrsum2 = _ve_vfsums_vv(vrsum23_##RS) ;				\
    __vr vrsum3 = _ve_vfsums_vv(_ve_vsll_vvs(vrsum23_##RS,32));		\
    _ve_lvl(1) ;							\
    _ve_vstu_vss(vrsum0, 4, pGKernel+kernelIndex0+R*gKernWidth+S) ;	\
    _ve_vstu_vss(vrsum1, 4, pGKernel+kernelIndex1+R*gKernWidth+S) ;	\
    _ve_vstu_vss(vrsum2, 4, pGKernel+kernelIndex2+R*gKernWidth+S) ;	\
    _ve_vstu_vss(vrsum3, 4, pGKernel+kernelIndex3+R*gKernWidth+S) ;	\
  }

  VSUM_AND_VST4(0,0,r0s0) ;
  VSUM_AND_VST4(0,1,r0s1) ;
  VSUM_AND_VST4(0,2,r0s2) ;
  VSUM_AND_VST4(1,0,r1s0) ;
  VSUM_AND_VST4(1,1,r1s1) ;
  VSUM_AND_VST4(1,2,r1s2) ;
  VSUM_AND_VST4(2,0,r2s0) ;
  VSUM_AND_VST4(2,1,r2s1) ;
  VSUM_AND_VST4(2,2,r2s2) ;

#undef VSUM_AND_VST4

}

template<int R_UNROLL, int S_UNROLL>
static inline void f8(
  const __vr vrij,
  const float * pIn, const int64_t inWidth, const int64_t inHeight,
  const float * pGOut, const int64_t gOutWidth, const int64_t gOutHeight,
  float * const pGKernel, const int64_t gKernWidth , const int64_t gKernHeight,
  const int64_t strideHeight,
  const int64_t inChannelGroup, const int64_t inChannel, const int64_t gOutChannel,
  const int64_t inGroupOffset, const int64_t outGroupOffset, const int64_t kernGroupOffset,
  const int64_t nY,
  const int64_t batch, const int64_t k, const int64_t c,
  const int64_t r, const int64_t s
)
{
  const int64_t kernelIndex0 = kernGroupOffset + ((k     * inChannelGroup + c) * gKernHeight + r) * gKernWidth + s;
  const int64_t kernelIndex1 = kernGroupOffset + (((k+1) * inChannelGroup + c) * gKernHeight + r) * gKernWidth + s;
  const int64_t kernelIndex2 = kernGroupOffset + (((k+2) * inChannelGroup + c) * gKernHeight + r) * gKernWidth + s;
  const int64_t kernelIndex3 = kernGroupOffset + (((k+3) * inChannelGroup + c) * gKernHeight + r) * gKernWidth + s;
  const int64_t kernelIndex4 = kernGroupOffset + (((k+4) * inChannelGroup + c) * gKernHeight + r) * gKernWidth + s;
  const int64_t kernelIndex5 = kernGroupOffset + (((k+5) * inChannelGroup + c) * gKernHeight + r) * gKernWidth + s;
  const int64_t kernelIndex6 = kernGroupOffset + (((k+6) * inChannelGroup + c) * gKernHeight + r) * gKernWidth + s;
  const int64_t kernelIndex7 = kernGroupOffset + (((k+7) * inChannelGroup + c) * gKernHeight + r) * gKernWidth + s;

  _ve_lvl(VLEN) ;

#define INIT_VRSUM8(R, S, RS)			\
  __vr vrsum01_##RS ;				\
  __vr vrsum23_##RS ;				\
  __vr vrsum45_##RS ;				\
  __vr vrsum67_##RS ;				\
  if( (R_UNROLL > R) && (S_UNROLL > S) ) {	\
    vrsum01_##RS = _ve_vbrd_vs_i64(0UL) ;	\
    vrsum23_##RS = _ve_vbrd_vs_i64(0UL) ;	\
    vrsum45_##RS = _ve_vbrd_vs_i64(0UL) ;	\
    vrsum67_##RS = _ve_vbrd_vs_i64(0UL) ;	\
  }

  INIT_VRSUM8(0,0,r0s0) ;
  INIT_VRSUM8(0,1,r0s1) ;
  INIT_VRSUM8(0,2,r0s2) ;
  INIT_VRSUM8(1,0,r1s0) ;
  INIT_VRSUM8(1,1,r1s1) ;
  INIT_VRSUM8(1,2,r1s2) ;
  INIT_VRSUM8(2,0,r2s0) ;
  INIT_VRSUM8(2,1,r2s1) ;
  INIT_VRSUM8(2,2,r2s2) ;
  INIT_VRSUM8(2,3,r2s3) ;
#undef INIT_VRSUM8

  for (int64_t n=0; n<batch; n++) {
    for (int64_t y=0; y<gOutHeight; y+=nY) {

      const int64_t vl = gOutWidth * (gOutHeight - y < nY ? gOutHeight - y : nY) ;

      _ve_lvl(vl) ;

      const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;

      const int64_t gOutIndex0  = outGroupOffset + ((n * gOutChannel + k  ) * gOutHeight + y) * gOutWidth ;
      const int64_t gOutIndex1  = outGroupOffset + ((n * gOutChannel + k+1) * gOutHeight + y) * gOutWidth ;
      const int64_t gOutIndex2  = outGroupOffset + ((n * gOutChannel + k+2) * gOutHeight + y) * gOutWidth ;
      const int64_t gOutIndex3  = outGroupOffset + ((n * gOutChannel + k+3) * gOutHeight + y) * gOutWidth ;
      const int64_t gOutIndex4  = outGroupOffset + ((n * gOutChannel + k+4) * gOutHeight + y) * gOutWidth ;
      const int64_t gOutIndex5  = outGroupOffset + ((n * gOutChannel + k+5) * gOutHeight + y) * gOutWidth ;
      const int64_t gOutIndex6  = outGroupOffset + ((n * gOutChannel + k+6) * gOutHeight + y) * gOutWidth ;
      const int64_t gOutIndex7  = outGroupOffset + ((n * gOutChannel + k+7) * gOutHeight + y) * gOutWidth ;


      __vr vrpin = _ve_vsfa_vvss(vrij,2,(int64_t)(pInChannel+(y*strideHeight+r)*inWidth+s)) ;
      __vr vrin_r0s0 = _ve_vgtu_vv(vrpin) ;

#define VLOAD_VRIN(R, S, RS)			\
  __vr vrin_##RS ;				\
  if( (R_UNROLL > R) && (S_UNROLL > S) ) {	\
    vrin_##RS = _ve_vgtu_vv(_ve_vaddsl_vsv(4*(R*inWidth+S), vrpin)) ;	\
  }						\

      VLOAD_VRIN(0,1,r0s1) ;
      VLOAD_VRIN(0,2,r0s2) ;
      VLOAD_VRIN(1,0,r1s0) ;
      VLOAD_VRIN(1,1,r1s1) ;
      VLOAD_VRIN(1,2,r1s2) ;
      VLOAD_VRIN(2,0,r2s0) ;
      VLOAD_VRIN(2,1,r2s1) ;
      VLOAD_VRIN(2,2,r2s2) ;
#undef VLOAD_VRIN

      __vr vrgout0 = _ve_vldu_vss(4, pGOut+gOutIndex0) ;
      __vr vrgout1 = _ve_vldu_vss(4, pGOut+gOutIndex1) ;
      __vr vrgout2 = _ve_vldu_vss(4, pGOut+gOutIndex2) ;
      __vr vrgout3 = _ve_vldu_vss(4, pGOut+gOutIndex3) ;
      __vr vrgout4 = _ve_vldu_vss(4, pGOut+gOutIndex4) ;
      __vr vrgout5 = _ve_vldu_vss(4, pGOut+gOutIndex5) ;
      __vr vrgout6 = _ve_vldu_vss(4, pGOut+gOutIndex6) ;
      __vr vrgout7 = _ve_vldu_vss(4, pGOut+gOutIndex7) ;

      __vr vrgout01 = _ve_vshf_vvvs(vrgout0, vrgout1, VE_VSHUFFLE_YUZU) ;
      __vr vrgout23 = _ve_vshf_vvvs(vrgout2, vrgout3, VE_VSHUFFLE_YUZU) ;
      __vr vrgout45 = _ve_vshf_vvvs(vrgout4, vrgout5, VE_VSHUFFLE_YUZU) ;
	__vr vrgout67 = _ve_vshf_vvvs(vrgout6, vrgout7, VE_VSHUFFLE_YUZU) ;

#define VFMAD8(R,S,RS)								\
  if( (R_UNROLL > R) && (S_UNROLL > S) ) {					\
    __vr vrinP_##RS = _ve_vshf_vvvs(vrin_##RS, vrin_##RS, VE_VSHUFFLE_YUZU) ;	\
    vrsum01_##RS = _ve_pvfmad_vvvv(vrsum01_##RS, vrinP_##RS, vrgout01) ;	\
    vrsum23_##RS = _ve_pvfmad_vvvv(vrsum23_##RS, vrinP_##RS, vrgout23) ;	\
    vrsum45_##RS = _ve_pvfmad_vvvv(vrsum45_##RS, vrinP_##RS, vrgout45) ;	\
    vrsum67_##RS = _ve_pvfmad_vvvv(vrsum67_##RS, vrinP_##RS, vrgout67) ;	\
  }

      VFMAD8(0,0,r0s0) ;
      VFMAD8(0,1,r0s1) ;
      VFMAD8(0,2,r0s2) ;
      VFMAD8(1,0,r1s0) ;
      VFMAD8(1,1,r1s1) ;
      VFMAD8(1,2,r1s2) ;
      VFMAD8(2,0,r2s0) ;
      VFMAD8(2,1,r2s1) ;
      VFMAD8(2,2,r2s2) ;
#undef VFMAD8

    } // gOutPixels
  } // batch

#define VSUM_AND_VST8(R,S,RS)						\
  if( (R_UNROLL > R) && (S_UNROLL > S) ) {				\
    _ve_lvl(VLEN) ;							\
    __vr vrsum0 = _ve_vfsums_vv(vrsum01_##RS) ;				\
    __vr vrsum1 = _ve_vfsums_vv(_ve_vsll_vvs(vrsum01_##RS,32));		\
    __vr vrsum2 = _ve_vfsums_vv(vrsum23_##RS) ;				\
    __vr vrsum3 = _ve_vfsums_vv(_ve_vsll_vvs(vrsum23_##RS,32));		\
    __vr vrsum4 = _ve_vfsums_vv(vrsum45_##RS) ;				\
    __vr vrsum5 = _ve_vfsums_vv(_ve_vsll_vvs(vrsum45_##RS,32));		\
    __vr vrsum6 = _ve_vfsums_vv(vrsum67_##RS) ;				\
    __vr vrsum7 = _ve_vfsums_vv(_ve_vsll_vvs(vrsum67_##RS,32));		\
    _ve_lvl(1) ;							\
    _ve_vstu_vss(vrsum0, 4, pGKernel+kernelIndex0+R*gKernWidth+S) ;	\
    _ve_vstu_vss(vrsum1, 4, pGKernel+kernelIndex1+R*gKernWidth+S) ;	\
    _ve_vstu_vss(vrsum2, 4, pGKernel+kernelIndex2+R*gKernWidth+S) ;	\
    _ve_vstu_vss(vrsum3, 4, pGKernel+kernelIndex3+R*gKernWidth+S) ;	\
    _ve_vstu_vss(vrsum4, 4, pGKernel+kernelIndex4+R*gKernWidth+S) ;	\
    _ve_vstu_vss(vrsum5, 4, pGKernel+kernelIndex5+R*gKernWidth+S) ;	\
    _ve_vstu_vss(vrsum6, 4, pGKernel+kernelIndex6+R*gKernWidth+S) ;	\
    _ve_vstu_vss(vrsum7, 4, pGKernel+kernelIndex7+R*gKernWidth+S) ;	\
  }

  VSUM_AND_VST8(0,0,r0s0) ;
  VSUM_AND_VST8(0,1,r0s1) ;
  VSUM_AND_VST8(0,2,r0s2) ;
  VSUM_AND_VST8(1,0,r1s0) ;
  VSUM_AND_VST8(1,1,r1s1) ;
  VSUM_AND_VST8(1,2,r1s2) ;
  VSUM_AND_VST8(2,0,r2s0) ;
  VSUM_AND_VST8(2,1,r2s1) ;
  VSUM_AND_VST8(2,2,r2s2) ;

#undef VSUM_AND_VST8

}

extern "C" {

vednnError_t
vednnConvolutionBackwardFilter_direct_dil1_pad0_owU32(
    const vednnTensorParam_t *  	pParamIn,
    const void *  			pDataIn,
    const vednnTensorParam_t *  	pParamGradOut,
    const void *  			pDataGradOut,
    const vednnConvolutionParam_t *  	pParamConv,
    const vednnFilterParam_t *  	pParamGradKernel,
    void *  				pDataGradKernel
#ifdef VEDNN_USE_OPENMP
    ,
    const int64_t				beginOChannel,
    const int64_t				nOChannel
#endif
)
{
  const int64_t inChannel   = pParamIn->channel;
  const int64_t inWidth     = pParamIn->width;
  const int64_t inHeight    = pParamIn->height;
  const int64_t batch       = pParamGradOut->batch;
  const int64_t gOutChannel = pParamGradOut->channel;
  const int64_t gOutWidth   = pParamGradOut->width;
  const int64_t gOutHeight  = pParamGradOut->height;
  const int64_t gKernWidth  = pParamGradKernel->width;
  const int64_t gKernHeight = pParamGradKernel->height;

  const int64_t group          = pParamConv->group;
  const int64_t strideWidth    = pParamConv->strideWidth;;
  const int64_t strideHeight   = pParamConv->strideHeight;
//  const int64_t padWidth       = pParamConv->padWidth;	// must be 0
//  const int64_t padHeight      = pParamConv->padHeight;	// must be 0
//  const int64_t dilationWidth  = pParamConv->dilationWidth;	// must be 1
//  const int64_t dilationHeight = pParamConv->dilationHeight;	// must be 1

  const int64_t inChannelGroup   =  inChannel   / group;
  const int64_t gOutChannelGroup = gOutChannel  / group;

  const float *  pIn      = (const float *) pDataIn;
  const float *  pGOut    = (const float *) pDataGradOut;
  float *  const pGKernel = (float * const) pDataGradKernel;

  const int gOutPixels= gOutHeight*gOutWidth ;

#ifndef VEDNN_USE_OPENMP
  const int64_t beginOChannel = 0 ;
  const int64_t nOChannel     = gOutChannelGroup ;
#endif
  {
    const int64_t nY = VLEN / gOutWidth ;

    _ve_lvl(nY*gOutWidth) ;
    __vr vrseq = _ve_vseq_v() ;			// xy
    __vr vry  = _ve_vdivsl_vvs(vrseq, gOutWidth) ;
    __vr vrx  = _ve_vsubsl_vvv(vrseq, _ve_vmulul_vsv(gOutWidth,vry)) ;
    __vr vri  = _ve_vmulsl_vsv(strideHeight, vry) ;
    __vr vrj  = _ve_vmulsl_vsv(strideWidth,  vrx) ;
    __vr vrij = _ve_vaddul_vvv(vrj, _ve_vmulul_vsv(inWidth,vri)) ;


    for (int64_t g = 0; g < group; g++) {
      int64_t inGroupOffset   = g * inChannelGroup  * inHeight  * inWidth;
      int64_t outGroupOffset  = (g * gOutChannelGroup + beginOChannel) * gOutHeight * gOutWidth;
      int64_t kernGroupOffset = (g * gOutChannelGroup + beginOChannel) * inChannelGroup * gKernHeight * gKernWidth;

      int64_t k=0;
      if ( (nOChannel & 0x01) == 1 ) {
	for (int64_t c=0; c<inChannelGroup; c++) {

#define F1(R,S,M,N)					\
f1<M,N>(vrij,						\
      pIn, inWidth, inHeight,				\
      pGOut, gOutWidth, gOutHeight,			\
      pGKernel, gKernWidth, gKernHeight,		\
      strideHeight,					\
      inChannelGroup, inChannel, gOutChannel,		\
      inGroupOffset, outGroupOffset, kernGroupOffset,	\
      nY,						\
      batch, k, c,					\
      R, S						\
  )
	  int64_t r = 0;

	  switch(gKernHeight % 3) {
	  case 1 :
	    {
	      int64_t s = 0;
	      switch( gKernWidth % 3 ) {
	      case 1 : F1(r,s,1,1) ; s+=1 ; break;
	      case 2 : F1(r,s,1,2) ; s+=2 ; break;
	      default : ;
	      }
	      for (; s<gKernWidth; s+=3) {
		F1(r,s,1,3) ;
	      }
	      r+=1 ;
	    }
	    break ;
	  case 2 :
	    {
	      int64_t s = 0;
	      switch( gKernWidth % 3 ) {
	      case 1 : F1(r,s,2,1) ; s+=1 ; break;
	      case 2 : F1(r,s,2,2) ; s+=2 ; break;
	      default : ;
	      }
	      for (; s<gKernWidth; s+=3) {
		F1(r,s,2,3) ;
	      }
	      r+=2 ;
	    }
	    break ;
	  default :
	    break ;
	  }
	  for(; r<gKernHeight; r+=3) {
	    int64_t s = 0;
	    switch( gKernWidth % 3 ) {
	    case 1 : F1(r,s,3,1) ; s+=1 ; break;
	    case 2 : F1(r,s,3,2) ; s+=2 ; break;
	    default : ;
	    }
	    for (; s<gKernWidth; s+=3) {
	      F1(r,s,3,3) ;
	    }
	  }
#undef F1
	} // inChannel

	k++ ;
      }
      if ( ((nOChannel >> 1) & 0x01) == 1 ) {
	for (int64_t c=0; c<inChannelGroup; c++) {

#define F2(R,S,M,N)					\
f2<M,N>(vrij,						\
      pIn, inWidth, inHeight,				\
      pGOut, gOutWidth, gOutHeight,			\
      pGKernel, gKernWidth, gKernHeight,		\
      strideHeight,					\
      inChannelGroup, inChannel, gOutChannel,		\
      inGroupOffset, outGroupOffset, kernGroupOffset,	\
      nY,						\
      batch, k, c,					\
      R, S						\
)
	  int64_t r = 0;

	  switch(gKernHeight % 3) {
	  case 1 :
	    {
	      int64_t s = 0;
	      switch( gKernWidth % 3 ) {
	      case 1 : F2(r,s,1,1) ; s+=1 ; break;
	      case 2 : F2(r,s,1,2) ; s+=2 ; break;
	      default : ;
	      }
	      for (; s<gKernWidth; s+=3) {
		F2(r,s,1,3) ;
	      }
	      r+=1 ;
	    }
	    break ;
	  case 2 :
	    {
	      int64_t s = 0;
	      switch( gKernWidth % 3 ) {
	      case 1 : F2(r,s,2,1) ; s+=1 ; break;
	      case 2 : F2(r,s,2,2) ; s+=2 ; break;
	      default : ;
	      }
	      for (; s<gKernWidth; s+=3) {
		F2(r,s,2,3) ;
	      }
	      r+=2 ;
	    }
	    break ;
	  default :
	    break ;
	  }
	  for(; r<gKernHeight; r+=3) {
	    int64_t s = 0;
	    switch( gKernWidth % 3 ) {
	    case 1 : F2(r,s,3,1) ; s+=1 ; break;
	    case 2 : F2(r,s,3,2) ; s+=2 ; break;
	    default : ;
	    }
	    for (; s<gKernWidth; s+=3) {
	      F2(r,s,3,3) ;
	    }
	  }
#undef F2
	} // inChannel
	k+=2;
      }
      if ( ((nOChannel >> 2) & 0x01) == 1 ) {
	for (int64_t c=0; c<inChannelGroup; c++) {

#define F4(R,S,M,N)					\
f4<M,N>(vrij,						\
      pIn, inWidth, inHeight,				\
      pGOut, gOutWidth, gOutHeight,			\
      pGKernel, gKernWidth, gKernHeight,		\
      strideHeight,					\
      inChannelGroup, inChannel, gOutChannel,		\
      inGroupOffset, outGroupOffset, kernGroupOffset,	\
      nY,						\
      batch, k, c,					\
      R, S						\
)
	  int64_t r = 0;

	  switch(gKernHeight % 3) {
	  case 1 :
	    {
	      int64_t s = 0;
	      switch( gKernWidth % 3 ) {
	      case 1 : F4(r,s,1,1) ; s+=1 ; break;
	      case 2 : F4(r,s,1,2) ; s+=2 ; break;
	      default : ;
	      }
	      for (; s<gKernWidth; s+=3) {
		F4(r,s,1,3) ;
	      }
	      r+=1 ;
	    }
	    break ;
	  case 2 :
	    {
	      int64_t s = 0;
	      switch( gKernWidth % 3 ) {
	      case 1 : F4(r,s,2,1) ; s+=1 ; break;
	      case 2 : F4(r,s,2,2) ; s+=2 ; break;
	      default : ;
	      }
	      for (; s<gKernWidth; s+=3) {
		F4(r,s,2,3) ;
	      }
	      r+=2 ;
	    }
	    break ;
	  default :
	    break ;
	  }
	  for(; r<gKernHeight; r+=3) {
	    int64_t s = 0;
	    switch( gKernWidth % 3 ) {
	    case 1 : F4(r,s,3,1) ; s+=1 ; break;
	    case 2 : F4(r,s,3,2) ; s+=2 ; break;
	    default : ;
	    }
	    for (; s<gKernWidth; s+=3) {
	      F4(r,s,3,3) ;
	    }
	  }
#undef F4
	} // inChannel
	k+=4;
      }
      for ( ;k<nOChannel; k+=8) {
	for (int64_t c=0; c<inChannelGroup; c++) {

#define F8(R,S,M,N)					\
f8<M,N>(vrij,						\
      pIn, inWidth, inHeight,				\
      pGOut, gOutWidth, gOutHeight,			\
      pGKernel, gKernWidth, gKernHeight,		\
      strideHeight,					\
      inChannelGroup, inChannel, gOutChannel,		\
      inGroupOffset, outGroupOffset, kernGroupOffset,	\
      nY,						\
      batch, k, c,					\
      R, S						\
)
	  int64_t r = 0;

	  switch(gKernHeight % 3) {
	  case 1 :
	    {
	      int64_t s = 0;
	      switch( gKernWidth % 3 ) {
	      case 1 : F8(r,s,1,1) ; s+=1 ; break;
	      case 2 : F8(r,s,1,2) ; s+=2 ; break;
	      default : ;
	      }
	      for (; s<gKernWidth; s+=3) {
		F8(r,s,1,3) ;
	      }
	      r+=1 ;
	    }
	    break ;
	  case 2 :
	    {
	      int64_t s = 0;
	      switch( gKernWidth % 3 ) {
	      case 1 : F8(r,s,2,1) ; s+=1 ; break;
	      case 2 : F8(r,s,2,2) ; s+=2 ; break;
	      default : ;
	      }
	      for (; s<gKernWidth; s+=3) {
		F8(r,s,2,3) ;
	      }
	      r+=2 ;
	    }
	    break ;
	  default :
	    break ;
	  }
	  for(; r<gKernHeight; r+=3) {
	    int64_t s = 0;
	    switch( gKernWidth % 3 ) {
	    case 1 : F8(r,s,3,1) ; s+=1 ; break;
	    case 2 : F8(r,s,3,2) ; s+=2 ; break;
	    default : ;
	    }
	    for (; s<gKernWidth; s+=3) {
	      F8(r,s,3,3) ;
	    }
	  }
#undef F8
	} // inChannel
      } // outChannel
    } // group
  }


  return VEDNN_SUCCESS;
}

}
