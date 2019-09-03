#include <stdint.h>
#include <stdlib.h>

#include "vednn.h"

#include "velintrin.h"
#define VLEN	(256)


template<filterLayout_t FLAYOUT>
static inline void k1(
  const float * __restrict__ pIn,
  const float * __restrict__ pKernel,
  float * __restrict__ const pOut,
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
  const int64_t nY,
  const int64_t inWidthHalf,
  const int64_t outWidthHalf,
  const __vm256 vm_s0,
  const __vm256 vm_s2,
  const int64_t n,
  const int64_t k
)
{
  int64_t outIndex = outGroupOffset + (n * outChannel + k) * oPixels ;

  for (int64_t y=0; y<outHeight; y+=nY) {
    const int64_t vl0 = inWidthHalf * (outHeight - y < nY ? outHeight - y : nY) ;
    const int64_t vl1 = outWidthHalf * (outHeight - y < nY ? outHeight - y : nY) ;
    const int64_t op = y * outWidth ;

    __vr vrsum0 = _vel_vbrdl_vsl(0UL, vl1) ;

    for (int64_t c = 0; c < inChannelGroup; c++) {

      const float *pInChannel = pIn + inGroupOffset + (((n * inChannel + c) * inHeight + y) * inWidth ) ;

      __vr vrin_r0 = _vel_vld_vssl(8, pInChannel+0*inWidth, vl0) ;
      __vr vrin_r1 = _vel_vld_vssl(8, pInChannel+1*inWidth, vl0) ;
      __vr vrin_r2 = _vel_vld_vssl(8, pInChannel+2*inWidth, vl0) ;

      __vr vrin_r0s0 = _vel_vcp_vvmvl(vrin_r0, vm_s0, _vel_vbrdl_vsl(0UL, vl0), vl0) ;
      __vr vrin_r0s2 = _vel_vcp_vvmvl(vrin_r0, vm_s2, _vel_vbrdl_vsl(0UL, vl0), vl0) ;
      __vr vrin_r1s0 = _vel_vcp_vvmvl(vrin_r1, vm_s0, _vel_vbrdl_vsl(0UL, vl0), vl0) ;
      __vr vrin_r1s2 = _vel_vcp_vvmvl(vrin_r1, vm_s2, _vel_vbrdl_vsl(0UL, vl0), vl0) ;
      __vr vrin_r2s0 = _vel_vcp_vvmvl(vrin_r2, vm_s0, _vel_vbrdl_vsl(0UL, vl0), vl0) ;
      __vr vrin_r2s2 = _vel_vcp_vvmvl(vrin_r2, vm_s2, _vel_vbrdl_vsl(0UL, vl0), vl0) ;

      __vr vrin_r0s1 = _vel_vshf_vvvsl(vrin_r0s0, vrin_r0s2,VE_VSHUFFLE_ZLYU, vl1) ;
      __vr vrin_r1s1 = _vel_vshf_vvvsl(vrin_r1s0, vrin_r1s2,VE_VSHUFFLE_ZLYU, vl1) ;
      __vr vrin_r2s1 = _vel_vshf_vvvsl(vrin_r2s0, vrin_r2s2,VE_VSHUFFLE_ZLYU, vl1) ;

      const float *pKerValue = ( FLAYOUT == VEDNN_FILTER_LAYOUT_NCHW ) ?
				pKernel + kernGroupOffset + (k * inChannelGroup + c) * kernHeight * kernWidth :
				pKernel + kernGroupOffset + c * outChannelGroup + k ;

      const int64_t kernelDistance = ( FLAYOUT == VEDNN_FILTER_LAYOUT_NCHW ) ?
				     inChannelGroup * kernHeight * kernWidth :
				     1 ;

#define VFADD1(VRIN)								\
{										\
const uint64_t kerValue0 = _vel_pack_f32a(pKerValue+ 0 * kernelDistance) ;	\
vrsum0 = _vel_pvfmad_vvsvl(vrsum0, kerValue0, VRIN, vl1) ;			\
}
      if ( FLAYOUT == VEDNN_FILTER_LAYOUT_NCHW ) {
	VFADD1(vrin_r0s0) ; pKerValue++ ;
	VFADD1(vrin_r0s1) ; pKerValue++ ;
	VFADD1(vrin_r0s2) ; pKerValue++ ;
	VFADD1(vrin_r1s0) ; pKerValue++ ;
	VFADD1(vrin_r1s1) ; pKerValue++ ;
	VFADD1(vrin_r1s2) ; pKerValue++ ;
	VFADD1(vrin_r2s0) ; pKerValue++ ;
	VFADD1(vrin_r2s1) ; pKerValue++ ;
	VFADD1(vrin_r2s2) ; pKerValue++ ;
      }
      else {
	VFADD1(vrin_r0s0) ; pKerValue += inChannelGroup*outChannelGroup;
	VFADD1(vrin_r0s1) ; pKerValue += inChannelGroup*outChannelGroup;
	VFADD1(vrin_r0s2) ; pKerValue += inChannelGroup*outChannelGroup;
	VFADD1(vrin_r1s0) ; pKerValue += inChannelGroup*outChannelGroup;
	VFADD1(vrin_r1s1) ; pKerValue += inChannelGroup*outChannelGroup;
	VFADD1(vrin_r1s2) ; pKerValue += inChannelGroup*outChannelGroup;
	VFADD1(vrin_r2s0) ; pKerValue += inChannelGroup*outChannelGroup;
	VFADD1(vrin_r2s1) ; pKerValue += inChannelGroup*outChannelGroup;
	VFADD1(vrin_r2s2) ; pKerValue += inChannelGroup*outChannelGroup;
      }
#undef VFADD1
    } // inChannel

    _vel_vst_vssl(vrsum0, 8, pOut+outIndex, vl1) ;

    outIndex += 2*vl1 ;
  } // outPixels
}

template<filterLayout_t FLAYOUT>
static inline void k2(
  const float * __restrict__ pIn,
  const float * __restrict__ pKernel,
  float * __restrict__ const pOut,
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
  const int64_t nY,
  const int64_t inWidthHalf,
  const int64_t outWidthHalf,
  const __vm256 vm_s0,
  const __vm256 vm_s2,
  const int64_t n,
  const int64_t k
)
{
  int64_t outIndex = outGroupOffset + (n * outChannel + k) * oPixels ;

  for (int64_t y=0; y<outHeight; y+=nY) {
    const int64_t vl0 = inWidthHalf * (outHeight - y < nY ? outHeight - y : nY) ;
    const int64_t vl1 = outWidthHalf * (outHeight - y < nY ? outHeight - y : nY) ;
    const int64_t op = y * outWidth ;

    __vr vrsum0 = _vel_vbrdl_vsl(0UL, vl1) ;
    __vr vrsum1 = _vel_vbrdl_vsl(0UL, vl1) ;

    for (int64_t c = 0; c < inChannelGroup; c++) {

      const float *pInChannel = pIn + inGroupOffset + (((n * inChannel + c) * inHeight + y) * inWidth ) ;

      __vr vrin_r0 = _vel_vld_vssl(8, pInChannel+0*inWidth, vl0) ;
      __vr vrin_r1 = _vel_vld_vssl(8, pInChannel+1*inWidth, vl0) ;
      __vr vrin_r2 = _vel_vld_vssl(8, pInChannel+2*inWidth, vl0) ;

      __vr vrin_r0s0 = _vel_vcp_vvmvl(vrin_r0, vm_s0, _vel_vbrdl_vsl(0UL, vl0), vl0) ;
      __vr vrin_r0s2 = _vel_vcp_vvmvl(vrin_r0, vm_s2, _vel_vbrdl_vsl(0UL, vl0), vl0) ;
      __vr vrin_r1s0 = _vel_vcp_vvmvl(vrin_r1, vm_s0, _vel_vbrdl_vsl(0UL, vl0), vl0) ;
      __vr vrin_r1s2 = _vel_vcp_vvmvl(vrin_r1, vm_s2, _vel_vbrdl_vsl(0UL, vl0), vl0) ;
      __vr vrin_r2s0 = _vel_vcp_vvmvl(vrin_r2, vm_s0, _vel_vbrdl_vsl(0UL, vl0), vl0) ;
      __vr vrin_r2s2 = _vel_vcp_vvmvl(vrin_r2, vm_s2, _vel_vbrdl_vsl(0UL, vl0), vl0) ;

      __vr vrin_r0s1 = _vel_vshf_vvvsl(vrin_r0s0, vrin_r0s2,VE_VSHUFFLE_ZLYU, vl1) ;
      __vr vrin_r1s1 = _vel_vshf_vvvsl(vrin_r1s0, vrin_r1s2,VE_VSHUFFLE_ZLYU, vl1) ;
      __vr vrin_r2s1 = _vel_vshf_vvvsl(vrin_r2s0, vrin_r2s2,VE_VSHUFFLE_ZLYU, vl1) ;

      const float *pKerValue = ( FLAYOUT == VEDNN_FILTER_LAYOUT_NCHW ) ?
				pKernel + kernGroupOffset + (k * inChannelGroup + c) * kernHeight * kernWidth :
				pKernel + kernGroupOffset + c * outChannelGroup + k ;

      const int64_t kernelDistance = ( FLAYOUT == VEDNN_FILTER_LAYOUT_NCHW ) ?
				     inChannelGroup * kernHeight * kernWidth :
				     1 ;

#define VFADD2(VRIN)								\
{										\
const uint64_t kerValue0 = _vel_pack_f32a(pKerValue+ 0 * kernelDistance) ;	\
const uint64_t kerValue1 = _vel_pack_f32a(pKerValue+ 1 * kernelDistance) ;	\
vrsum0 = _vel_pvfmad_vvsvl(vrsum0, kerValue0, VRIN, vl1) ;			\
vrsum1 = _vel_pvfmad_vvsvl(vrsum1, kerValue1, VRIN, vl1) ;			\
}
      if ( FLAYOUT == VEDNN_FILTER_LAYOUT_NCHW ) {
	VFADD2(vrin_r0s0) ; pKerValue++ ;
	VFADD2(vrin_r0s1) ; pKerValue++ ;
	VFADD2(vrin_r0s2) ; pKerValue++ ;
	VFADD2(vrin_r1s0) ; pKerValue++ ;
	VFADD2(vrin_r1s1) ; pKerValue++ ;
	VFADD2(vrin_r1s2) ; pKerValue++ ;
	VFADD2(vrin_r2s0) ; pKerValue++ ;
	VFADD2(vrin_r2s1) ; pKerValue++ ;
	VFADD2(vrin_r2s2) ; pKerValue++ ;
      }
      else {
	VFADD2(vrin_r0s0) ; pKerValue += inChannelGroup*outChannelGroup;
	VFADD2(vrin_r0s1) ; pKerValue += inChannelGroup*outChannelGroup;
	VFADD2(vrin_r0s2) ; pKerValue += inChannelGroup*outChannelGroup;
	VFADD2(vrin_r1s0) ; pKerValue += inChannelGroup*outChannelGroup;
	VFADD2(vrin_r1s1) ; pKerValue += inChannelGroup*outChannelGroup;
	VFADD2(vrin_r1s2) ; pKerValue += inChannelGroup*outChannelGroup;
	VFADD2(vrin_r2s0) ; pKerValue += inChannelGroup*outChannelGroup;
	VFADD2(vrin_r2s1) ; pKerValue += inChannelGroup*outChannelGroup;
	VFADD2(vrin_r2s2) ; pKerValue += inChannelGroup*outChannelGroup;
      }
#undef VFADD2
    } // inChannel

    _vel_vst_vssl(vrsum0, 8, pOut+outIndex, vl1) ;
    _vel_vst_vssl(vrsum1, 8, pOut+outIndex+ 1*oPixels, vl1) ;

    outIndex += 2*vl1 ;
  } // outPixels
}


template<filterLayout_t FLAYOUT>
static inline void k4(
  const float * __restrict__ pIn,
  const float * __restrict__ pKernel,
  float * __restrict__ const pOut,
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
  const int64_t nY,
  const int64_t inWidthHalf,
  const int64_t outWidthHalf,
  const __vm256 vm_s0,
  const __vm256 vm_s2,
  const int64_t n,
  const int64_t k
)
{
  int64_t outIndex = outGroupOffset + (n * outChannel + k) * oPixels ;

  for (int64_t y=0; y<outHeight; y+=nY) {
    const int64_t vl0 = inWidthHalf * (outHeight - y < nY ? outHeight - y : nY) ;
    const int64_t vl1 = outWidthHalf * (outHeight - y < nY ? outHeight - y : nY) ;
    const int64_t op = y * outWidth ;

    __vr vrsum0 = _vel_vbrdl_vsl(0UL, vl1) ;
    __vr vrsum1 = _vel_vbrdl_vsl(0UL, vl1) ;
    __vr vrsum2 = _vel_vbrdl_vsl(0UL, vl1) ;
    __vr vrsum3 = _vel_vbrdl_vsl(0UL, vl1) ;

    for (int64_t c = 0; c < inChannelGroup; c++) {

      const float *pInChannel = pIn + inGroupOffset + (((n * inChannel + c) * inHeight + y) * inWidth ) ;

      __vr vrin_r0 = _vel_vld_vssl(8, pInChannel+0*inWidth, vl0) ;
      __vr vrin_r1 = _vel_vld_vssl(8, pInChannel+1*inWidth, vl0) ;
      __vr vrin_r2 = _vel_vld_vssl(8, pInChannel+2*inWidth, vl0) ;

      __vr vrin_r0s0 = _vel_vcp_vvmvl(vrin_r0, vm_s0, _vel_vbrdl_vsl(0UL, vl0), vl0) ;
      __vr vrin_r0s2 = _vel_vcp_vvmvl(vrin_r0, vm_s2, _vel_vbrdl_vsl(0UL, vl0), vl0) ;
      __vr vrin_r1s0 = _vel_vcp_vvmvl(vrin_r1, vm_s0, _vel_vbrdl_vsl(0UL, vl0), vl0) ;
      __vr vrin_r1s2 = _vel_vcp_vvmvl(vrin_r1, vm_s2, _vel_vbrdl_vsl(0UL, vl0), vl0) ;
      __vr vrin_r2s0 = _vel_vcp_vvmvl(vrin_r2, vm_s0, _vel_vbrdl_vsl(0UL, vl0), vl0) ;
      __vr vrin_r2s2 = _vel_vcp_vvmvl(vrin_r2, vm_s2, _vel_vbrdl_vsl(0UL, vl0), vl0) ;

      __vr vrin_r0s1 = _vel_vshf_vvvsl(vrin_r0s0, vrin_r0s2,VE_VSHUFFLE_ZLYU, vl1) ;
      __vr vrin_r1s1 = _vel_vshf_vvvsl(vrin_r1s0, vrin_r1s2,VE_VSHUFFLE_ZLYU, vl1) ;
      __vr vrin_r2s1 = _vel_vshf_vvvsl(vrin_r2s0, vrin_r2s2,VE_VSHUFFLE_ZLYU, vl1) ;

      const float *pKerValue = ( FLAYOUT == VEDNN_FILTER_LAYOUT_NCHW ) ?
				pKernel + kernGroupOffset + (k * inChannelGroup + c) * kernHeight * kernWidth :
				pKernel + kernGroupOffset + c * outChannelGroup + k ;

      const int64_t kernelDistance = ( FLAYOUT == VEDNN_FILTER_LAYOUT_NCHW ) ?
				     inChannelGroup * kernHeight * kernWidth :
				     1 ;

#define VFADD4(VRIN)								\
{										\
const uint64_t kerValue0 = _vel_pack_f32a(pKerValue+ 0 * kernelDistance) ;	\
const uint64_t kerValue1 = _vel_pack_f32a(pKerValue+ 1 * kernelDistance) ;	\
const uint64_t kerValue2 = _vel_pack_f32a(pKerValue+ 2 * kernelDistance) ;	\
const uint64_t kerValue3 = _vel_pack_f32a(pKerValue+ 3 * kernelDistance) ;	\
vrsum0 = _vel_pvfmad_vvsvl(vrsum0, kerValue0, VRIN, vl1) ;			\
vrsum1 = _vel_pvfmad_vvsvl(vrsum1, kerValue1, VRIN, vl1) ;			\
vrsum2 = _vel_pvfmad_vvsvl(vrsum2, kerValue2, VRIN, vl1) ;			\
vrsum3 = _vel_pvfmad_vvsvl(vrsum3, kerValue3, VRIN, vl1) ;			\
}
      if ( FLAYOUT == VEDNN_FILTER_LAYOUT_NCHW ) {
	VFADD4(vrin_r0s0) ; pKerValue++ ;
	VFADD4(vrin_r0s1) ; pKerValue++ ;
	VFADD4(vrin_r0s2) ; pKerValue++ ;
	VFADD4(vrin_r1s0) ; pKerValue++ ;
	VFADD4(vrin_r1s1) ; pKerValue++ ;
	VFADD4(vrin_r1s2) ; pKerValue++ ;
	VFADD4(vrin_r2s0) ; pKerValue++ ;
	VFADD4(vrin_r2s1) ; pKerValue++ ;
	VFADD4(vrin_r2s2) ; pKerValue++ ;
      }
      else {
	VFADD4(vrin_r0s0) ; pKerValue += inChannelGroup*outChannelGroup;
	VFADD4(vrin_r0s1) ; pKerValue += inChannelGroup*outChannelGroup;
	VFADD4(vrin_r0s2) ; pKerValue += inChannelGroup*outChannelGroup;
	VFADD4(vrin_r1s0) ; pKerValue += inChannelGroup*outChannelGroup;
	VFADD4(vrin_r1s1) ; pKerValue += inChannelGroup*outChannelGroup;
	VFADD4(vrin_r1s2) ; pKerValue += inChannelGroup*outChannelGroup;
	VFADD4(vrin_r2s0) ; pKerValue += inChannelGroup*outChannelGroup;
	VFADD4(vrin_r2s1) ; pKerValue += inChannelGroup*outChannelGroup;
	VFADD4(vrin_r2s2) ; pKerValue += inChannelGroup*outChannelGroup;
      }
#undef VFADD4
    } // inChannel

    _vel_vst_vssl(vrsum0, 8, pOut+outIndex, vl1) ;
    _vel_vst_vssl(vrsum1, 8, pOut+outIndex+ 1*oPixels, vl1) ;
    _vel_vst_vssl(vrsum2, 8, pOut+outIndex+ 2*oPixels, vl1) ;
    _vel_vst_vssl(vrsum3, 8, pOut+outIndex+ 3*oPixels, vl1) ;

    outIndex += 2*vl1 ;
  } // outPixels
}


template<filterLayout_t FLAYOUT>
static inline void k8(
  const float * __restrict__ pIn,
  const float * __restrict__ pKernel,
  float * __restrict__ const pOut,
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
  const int64_t nY,
  const int64_t inWidthHalf,
  const int64_t outWidthHalf,
  const __vm256 vm_s0,
  const __vm256 vm_s2,
  const int64_t n,
  const int64_t k
)
{
  int64_t outIndex = outGroupOffset + (n * outChannel + k) * oPixels ;

  for (int64_t y=0; y<outHeight; y+=nY) {
    const int64_t vl0 = inWidthHalf * (outHeight - y < nY ? outHeight - y : nY) ;
    const int64_t vl1 = outWidthHalf * (outHeight - y < nY ? outHeight - y : nY) ;
    const int64_t op = y * outWidth ;

    __vr vrsum0 = _vel_vbrdl_vsl(0UL, vl1) ;
    __vr vrsum1 = _vel_vbrdl_vsl(0UL, vl1) ;
    __vr vrsum2 = _vel_vbrdl_vsl(0UL, vl1) ;
    __vr vrsum3 = _vel_vbrdl_vsl(0UL, vl1) ;
    __vr vrsum4 = _vel_vbrdl_vsl(0UL, vl1) ;
    __vr vrsum5 = _vel_vbrdl_vsl(0UL, vl1) ;
    __vr vrsum6 = _vel_vbrdl_vsl(0UL, vl1) ;
    __vr vrsum7 = _vel_vbrdl_vsl(0UL, vl1) ;

    for (int64_t c = 0; c < inChannelGroup; c++) {

      const float *pInChannel = pIn + inGroupOffset + (((n * inChannel + c) * inHeight + y) * inWidth ) ;

      __vr vrin_r0 = _vel_vld_vssl(8, pInChannel+0*inWidth, vl0) ;
      __vr vrin_r1 = _vel_vld_vssl(8, pInChannel+1*inWidth, vl0) ;
      __vr vrin_r2 = _vel_vld_vssl(8, pInChannel+2*inWidth, vl0) ;

      __vr vrin_r0s0 = _vel_vcp_vvmvl(vrin_r0, vm_s0, _vel_vbrdl_vsl(0UL, vl0), vl0) ;
      __vr vrin_r0s2 = _vel_vcp_vvmvl(vrin_r0, vm_s2, _vel_vbrdl_vsl(0UL, vl0), vl0) ;
      __vr vrin_r1s0 = _vel_vcp_vvmvl(vrin_r1, vm_s0, _vel_vbrdl_vsl(0UL, vl0), vl0) ;
      __vr vrin_r1s2 = _vel_vcp_vvmvl(vrin_r1, vm_s2, _vel_vbrdl_vsl(0UL, vl0), vl0) ;
      __vr vrin_r2s0 = _vel_vcp_vvmvl(vrin_r2, vm_s0, _vel_vbrdl_vsl(0UL, vl0), vl0) ;
      __vr vrin_r2s2 = _vel_vcp_vvmvl(vrin_r2, vm_s2, _vel_vbrdl_vsl(0UL, vl0), vl0) ;

      __vr vrin_r0s1 = _vel_vshf_vvvsl(vrin_r0s0, vrin_r0s2,VE_VSHUFFLE_ZLYU, vl1) ;
      __vr vrin_r1s1 = _vel_vshf_vvvsl(vrin_r1s0, vrin_r1s2,VE_VSHUFFLE_ZLYU, vl1) ;
      __vr vrin_r2s1 = _vel_vshf_vvvsl(vrin_r2s0, vrin_r2s2,VE_VSHUFFLE_ZLYU, vl1) ;

      const float *pKerValue = ( FLAYOUT == VEDNN_FILTER_LAYOUT_NCHW ) ?
				pKernel + kernGroupOffset + (k * inChannelGroup + c) * kernHeight * kernWidth :
				pKernel + kernGroupOffset + c * outChannelGroup + k ;

      const int64_t kernelDistance = ( FLAYOUT == VEDNN_FILTER_LAYOUT_NCHW ) ?
				     inChannelGroup * kernHeight * kernWidth :
				     1 ;

#define VFADD8(VRIN)								\
{										\
const uint64_t kerValue0 = _vel_pack_f32a(pKerValue+ 0 * kernelDistance) ;	\
const uint64_t kerValue1 = _vel_pack_f32a(pKerValue+ 1 * kernelDistance) ;	\
const uint64_t kerValue2 = _vel_pack_f32a(pKerValue+ 2 * kernelDistance) ;	\
const uint64_t kerValue3 = _vel_pack_f32a(pKerValue+ 3 * kernelDistance) ;	\
const uint64_t kerValue4 = _vel_pack_f32a(pKerValue+ 4 * kernelDistance) ;	\
const uint64_t kerValue5 = _vel_pack_f32a(pKerValue+ 5 * kernelDistance) ;	\
const uint64_t kerValue6 = _vel_pack_f32a(pKerValue+ 6 * kernelDistance) ;	\
const uint64_t kerValue7 = _vel_pack_f32a(pKerValue+ 7 * kernelDistance) ;	\
vrsum0 = _vel_pvfmad_vvsvl(vrsum0, kerValue0, VRIN, vl1) ;			\
vrsum1 = _vel_pvfmad_vvsvl(vrsum1, kerValue1, VRIN, vl1) ;			\
vrsum2 = _vel_pvfmad_vvsvl(vrsum2, kerValue2, VRIN, vl1) ;			\
vrsum3 = _vel_pvfmad_vvsvl(vrsum3, kerValue3, VRIN, vl1) ;			\
vrsum4 = _vel_pvfmad_vvsvl(vrsum4, kerValue4, VRIN, vl1) ;			\
vrsum5 = _vel_pvfmad_vvsvl(vrsum5, kerValue5, VRIN, vl1) ;			\
vrsum6 = _vel_pvfmad_vvsvl(vrsum6, kerValue6, VRIN, vl1) ;			\
vrsum7 = _vel_pvfmad_vvsvl(vrsum7, kerValue7, VRIN, vl1) ;			\
}
      if ( FLAYOUT == VEDNN_FILTER_LAYOUT_NCHW ) {
	VFADD8(vrin_r0s0) ; pKerValue++ ;
	VFADD8(vrin_r0s1) ; pKerValue++ ;
	VFADD8(vrin_r0s2) ; pKerValue++ ;
	VFADD8(vrin_r1s0) ; pKerValue++ ;
	VFADD8(vrin_r1s1) ; pKerValue++ ;
	VFADD8(vrin_r1s2) ; pKerValue++ ;
	VFADD8(vrin_r2s0) ; pKerValue++ ;
	VFADD8(vrin_r2s1) ; pKerValue++ ;
	VFADD8(vrin_r2s2) ; pKerValue++ ;
      }
      else {
	VFADD8(vrin_r0s0) ; pKerValue += inChannelGroup*outChannelGroup;
	VFADD8(vrin_r0s1) ; pKerValue += inChannelGroup*outChannelGroup;
	VFADD8(vrin_r0s2) ; pKerValue += inChannelGroup*outChannelGroup;
	VFADD8(vrin_r1s0) ; pKerValue += inChannelGroup*outChannelGroup;
	VFADD8(vrin_r1s1) ; pKerValue += inChannelGroup*outChannelGroup;
	VFADD8(vrin_r1s2) ; pKerValue += inChannelGroup*outChannelGroup;
	VFADD8(vrin_r2s0) ; pKerValue += inChannelGroup*outChannelGroup;
	VFADD8(vrin_r2s1) ; pKerValue += inChannelGroup*outChannelGroup;
	VFADD8(vrin_r2s2) ; pKerValue += inChannelGroup*outChannelGroup;
      }
#undef VFADD8
    } // inChannel

    _vel_vst_vssl(vrsum0, 8, pOut+outIndex, vl1) ;
    _vel_vst_vssl(vrsum1, 8, pOut+outIndex+ 1*oPixels, vl1) ;
    _vel_vst_vssl(vrsum2, 8, pOut+outIndex+ 2*oPixels, vl1) ;
    _vel_vst_vssl(vrsum3, 8, pOut+outIndex+ 3*oPixels, vl1) ;
    _vel_vst_vssl(vrsum4, 8, pOut+outIndex+ 4*oPixels, vl1) ;
    _vel_vst_vssl(vrsum5, 8, pOut+outIndex+ 5*oPixels, vl1) ;
    _vel_vst_vssl(vrsum6, 8, pOut+outIndex+ 6*oPixels, vl1) ;
    _vel_vst_vssl(vrsum7, 8, pOut+outIndex+ 7*oPixels, vl1) ;

    outIndex += 2*vl1 ;
  } // outPixels
}

template<filterLayout_t FLAYOUT>
static inline void k16(
  const float * __restrict__ pIn,
  const float * __restrict__ pKernel,
  float * __restrict__ const pOut,
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
  const int64_t nY,
  const int64_t inWidthHalf,
  const int64_t outWidthHalf,
  const __vm256 vm_s0,
  const __vm256 vm_s2,
  const int64_t n,
  const int64_t k
)
{
  int64_t outIndex = outGroupOffset + (n * outChannel + k) * oPixels ;

  for (int64_t y=0; y<outHeight; y+=nY) {
    const int64_t vl0 = inWidthHalf * (outHeight - y < nY ? outHeight - y : nY) ;
    const int64_t vl1 = outWidthHalf * (outHeight - y < nY ? outHeight - y : nY) ;
    const int64_t op = y * outWidth ;

    __vr vrsum0 = _vel_vbrdl_vsl(0UL, vl1) ;
    __vr vrsum1 = _vel_vbrdl_vsl(0UL, vl1) ;
    __vr vrsum2 = _vel_vbrdl_vsl(0UL, vl1) ;
    __vr vrsum3 = _vel_vbrdl_vsl(0UL, vl1) ;
    __vr vrsum4 = _vel_vbrdl_vsl(0UL, vl1) ;
    __vr vrsum5 = _vel_vbrdl_vsl(0UL, vl1) ;
    __vr vrsum6 = _vel_vbrdl_vsl(0UL, vl1) ;
    __vr vrsum7 = _vel_vbrdl_vsl(0UL, vl1) ;
    __vr vrsum8 = _vel_vbrdl_vsl(0UL, vl1) ;
    __vr vrsum9 = _vel_vbrdl_vsl(0UL, vl1) ;
    __vr vrsumA = _vel_vbrdl_vsl(0UL, vl1) ;
    __vr vrsumB = _vel_vbrdl_vsl(0UL, vl1) ;
    __vr vrsumC = _vel_vbrdl_vsl(0UL, vl1) ;
    __vr vrsumD = _vel_vbrdl_vsl(0UL, vl1) ;
    __vr vrsumE = _vel_vbrdl_vsl(0UL, vl1) ;
    __vr vrsumF = _vel_vbrdl_vsl(0UL, vl1) ;

    for (int64_t c = 0; c < inChannelGroup; c++) {

      const float *pInChannel = pIn + inGroupOffset + (((n * inChannel + c) * inHeight + y) * inWidth ) ;

      __vr vrin_r0 = _vel_vld_vssl(8, pInChannel+0*inWidth, vl0) ;
      __vr vrin_r1 = _vel_vld_vssl(8, pInChannel+1*inWidth, vl0) ;
      __vr vrin_r2 = _vel_vld_vssl(8, pInChannel+2*inWidth, vl0) ;

      __vr vrin_r0s0 = _vel_vcp_vvmvl(vrin_r0, vm_s0, _vel_vbrdl_vsl(0UL, vl0), vl0) ;
      __vr vrin_r0s2 = _vel_vcp_vvmvl(vrin_r0, vm_s2, _vel_vbrdl_vsl(0UL, vl0), vl0) ;
      __vr vrin_r1s0 = _vel_vcp_vvmvl(vrin_r1, vm_s0, _vel_vbrdl_vsl(0UL, vl0), vl0) ;
      __vr vrin_r1s2 = _vel_vcp_vvmvl(vrin_r1, vm_s2, _vel_vbrdl_vsl(0UL, vl0), vl0) ;
      __vr vrin_r2s0 = _vel_vcp_vvmvl(vrin_r2, vm_s0, _vel_vbrdl_vsl(0UL, vl0), vl0) ;
      __vr vrin_r2s2 = _vel_vcp_vvmvl(vrin_r2, vm_s2, _vel_vbrdl_vsl(0UL, vl0), vl0) ;

      __vr vrin_r0s1 = _vel_vshf_vvvsl(vrin_r0s0, vrin_r0s2,VE_VSHUFFLE_ZLYU, vl1) ;
      __vr vrin_r1s1 = _vel_vshf_vvvsl(vrin_r1s0, vrin_r1s2,VE_VSHUFFLE_ZLYU, vl1) ;
      __vr vrin_r2s1 = _vel_vshf_vvvsl(vrin_r2s0, vrin_r2s2,VE_VSHUFFLE_ZLYU, vl1) ;

      const float *pKerValue = ( FLAYOUT == VEDNN_FILTER_LAYOUT_NCHW ) ?
				pKernel + kernGroupOffset + (k * inChannelGroup + c) * kernHeight * kernWidth :
				pKernel + kernGroupOffset + c * outChannelGroup + k ;

      const int64_t kernelDistance = ( FLAYOUT == VEDNN_FILTER_LAYOUT_NCHW ) ?
				     inChannelGroup * kernHeight * kernWidth :
				     1 ;

#define VFADD16(VRIN)								\
{										\
const uint64_t kerValue0 = _vel_pack_f32a(pKerValue+ 0 * kernelDistance) ;	\
const uint64_t kerValue1 = _vel_pack_f32a(pKerValue+ 1 * kernelDistance) ;	\
const uint64_t kerValue2 = _vel_pack_f32a(pKerValue+ 2 * kernelDistance) ;	\
const uint64_t kerValue3 = _vel_pack_f32a(pKerValue+ 3 * kernelDistance) ;	\
const uint64_t kerValue4 = _vel_pack_f32a(pKerValue+ 4 * kernelDistance) ;	\
const uint64_t kerValue5 = _vel_pack_f32a(pKerValue+ 5 * kernelDistance) ;	\
const uint64_t kerValue6 = _vel_pack_f32a(pKerValue+ 6 * kernelDistance) ;	\
const uint64_t kerValue7 = _vel_pack_f32a(pKerValue+ 7 * kernelDistance) ;	\
const uint64_t kerValue8 = _vel_pack_f32a(pKerValue+ 8 * kernelDistance) ;	\
const uint64_t kerValue9 = _vel_pack_f32a(pKerValue+ 9 * kernelDistance) ;	\
const uint64_t kerValueA = _vel_pack_f32a(pKerValue+10 * kernelDistance) ;	\
const uint64_t kerValueB = _vel_pack_f32a(pKerValue+11 * kernelDistance) ;	\
const uint64_t kerValueC = _vel_pack_f32a(pKerValue+12 * kernelDistance) ;	\
const uint64_t kerValueD = _vel_pack_f32a(pKerValue+13 * kernelDistance) ;	\
const uint64_t kerValueE = _vel_pack_f32a(pKerValue+14 * kernelDistance) ;	\
const uint64_t kerValueF = _vel_pack_f32a(pKerValue+15 * kernelDistance) ;	\
vrsum0 = _vel_pvfmad_vvsvl(vrsum0, kerValue0, VRIN, vl1) ;			\
vrsum1 = _vel_pvfmad_vvsvl(vrsum1, kerValue1, VRIN, vl1) ;			\
vrsum2 = _vel_pvfmad_vvsvl(vrsum2, kerValue2, VRIN, vl1) ;			\
vrsum3 = _vel_pvfmad_vvsvl(vrsum3, kerValue3, VRIN, vl1) ;			\
vrsum4 = _vel_pvfmad_vvsvl(vrsum4, kerValue4, VRIN, vl1) ;			\
vrsum5 = _vel_pvfmad_vvsvl(vrsum5, kerValue5, VRIN, vl1) ;			\
vrsum6 = _vel_pvfmad_vvsvl(vrsum6, kerValue6, VRIN, vl1) ;			\
vrsum7 = _vel_pvfmad_vvsvl(vrsum7, kerValue7, VRIN, vl1) ;			\
vrsum8 = _vel_pvfmad_vvsvl(vrsum8, kerValue8, VRIN, vl1) ;			\
vrsum9 = _vel_pvfmad_vvsvl(vrsum9, kerValue9, VRIN, vl1) ;			\
vrsumA = _vel_pvfmad_vvsvl(vrsumA, kerValueA, VRIN, vl1) ;			\
vrsumB = _vel_pvfmad_vvsvl(vrsumB, kerValueB, VRIN, vl1) ;			\
vrsumC = _vel_pvfmad_vvsvl(vrsumC, kerValueC, VRIN, vl1) ;			\
vrsumD = _vel_pvfmad_vvsvl(vrsumD, kerValueD, VRIN, vl1) ;			\
vrsumE = _vel_pvfmad_vvsvl(vrsumE, kerValueE, VRIN, vl1) ;			\
vrsumF = _vel_pvfmad_vvsvl(vrsumF, kerValueF, VRIN, vl1) ;			\
}
      if ( FLAYOUT == VEDNN_FILTER_LAYOUT_NCHW ) {
	VFADD16(vrin_r0s0) ; pKerValue++ ;
	VFADD16(vrin_r0s1) ; pKerValue++ ;
	VFADD16(vrin_r0s2) ; pKerValue++ ;
	VFADD16(vrin_r1s0) ; pKerValue++ ;
	VFADD16(vrin_r1s1) ; pKerValue++ ;
	VFADD16(vrin_r1s2) ; pKerValue++ ;
	VFADD16(vrin_r2s0) ; pKerValue++ ;
	VFADD16(vrin_r2s1) ; pKerValue++ ;
	VFADD16(vrin_r2s2) ; pKerValue++ ;
      }
      else {
	VFADD16(vrin_r0s0) ; pKerValue += inChannelGroup*outChannelGroup;
	VFADD16(vrin_r0s1) ; pKerValue += inChannelGroup*outChannelGroup;
	VFADD16(vrin_r0s2) ; pKerValue += inChannelGroup*outChannelGroup;
	VFADD16(vrin_r1s0) ; pKerValue += inChannelGroup*outChannelGroup;
	VFADD16(vrin_r1s1) ; pKerValue += inChannelGroup*outChannelGroup;
	VFADD16(vrin_r1s2) ; pKerValue += inChannelGroup*outChannelGroup;
	VFADD16(vrin_r2s0) ; pKerValue += inChannelGroup*outChannelGroup;
	VFADD16(vrin_r2s1) ; pKerValue += inChannelGroup*outChannelGroup;
	VFADD16(vrin_r2s2) ; pKerValue += inChannelGroup*outChannelGroup;
      }
#undef VFADD16
    } // inChannel

    _vel_vst_vssl(vrsum0, 8, pOut+outIndex, vl1) ;
    _vel_vst_vssl(vrsum1, 8, pOut+outIndex+ 1*oPixels, vl1) ;
    _vel_vst_vssl(vrsum2, 8, pOut+outIndex+ 2*oPixels, vl1) ;
    _vel_vst_vssl(vrsum3, 8, pOut+outIndex+ 3*oPixels, vl1) ;
    _vel_vst_vssl(vrsum4, 8, pOut+outIndex+ 4*oPixels, vl1) ;
    _vel_vst_vssl(vrsum5, 8, pOut+outIndex+ 5*oPixels, vl1) ;
    _vel_vst_vssl(vrsum6, 8, pOut+outIndex+ 6*oPixels, vl1) ;
    _vel_vst_vssl(vrsum7, 8, pOut+outIndex+ 7*oPixels, vl1) ;
    _vel_vst_vssl(vrsum8, 8, pOut+outIndex+ 8*oPixels, vl1) ;
    _vel_vst_vssl(vrsum9, 8, pOut+outIndex+ 9*oPixels, vl1) ;
    _vel_vst_vssl(vrsumA, 8, pOut+outIndex+10*oPixels, vl1) ;
    _vel_vst_vssl(vrsumB, 8, pOut+outIndex+11*oPixels, vl1) ;
    _vel_vst_vssl(vrsumC, 8, pOut+outIndex+12*oPixels, vl1) ;
    _vel_vst_vssl(vrsumD, 8, pOut+outIndex+13*oPixels, vl1) ;
    _vel_vst_vssl(vrsumE, 8, pOut+outIndex+14*oPixels, vl1) ;
    _vel_vst_vssl(vrsumF, 8, pOut+outIndex+15*oPixels, vl1) ;

    outIndex += 2*vl1 ;
  } // outPixels
}

extern "C" vednnError_t
vednnConvolutionForward_direct_dil1_str1_pad0_ker3_iw2XU256_ow2X_ioaligned(
    const vednnTensorParam_t *  	pParamIn,
    const void *  			pDataIn,
    const vednnFilterParam_t *  	pParamKernel,
    const void *  			pDataKernel,
    const vednnConvolutionParam_t *  	pParamConv,
    const vednnTensorParam_t *  	pParamOut,
    void *  				pDataOut
)
{
  const int64_t batch      = pParamIn->batch;
  const int64_t inChannel  = pParamIn->channel;
  const int64_t inWidth    = pParamIn->width;
  const int64_t inHeight   = pParamIn->height;
  const int64_t outChannel = pParamOut->channel;
  const int64_t outWidth   = pParamOut->width;
  const int64_t outHeight  = pParamOut->height;
  const int64_t kernWidth  = pParamKernel->width;
  const int64_t kernHeight = pParamKernel->height;

  const int64_t filter_layout = pParamKernel->layout ;

  const int64_t group          = pParamConv->group;
//  const int64_t strideWidth    = pParamConv->strideWidth;		// must be 1
//  const int64_t strideHeight   = pParamConv->strideHeight;		// must be 1
//  const int64_t padWidth       = pParamConv->padWidth;		// must be 0
//  const int64_t padHeight      = pParamConv->padHeight;		// must be 0
//  const int64_t dilationWidth  = pParamConv->dilationWidth;		// must be 1
//  const int64_t dilationHeight = pParamConv->dilationHeight;		// must be 1

  const int64_t inChannelGroup  = inChannel  / group;   // equal to pDataKernel->inChannel
  const int64_t outChannelGroup = outChannel / group;   // equal to pDataKernel->outChannel

  const float * pIn     = (const float *) pDataIn;
  const float * pKernel = (const float *) pDataKernel;
  float * const pOut    = (float * const) pDataOut;


  const int oPixels= outHeight*outWidth ;
  {
    const int64_t inWidthHalf  = inWidth >> 1 ;
    const int64_t outWidthHalf = outWidth >> 1 ;
    const int64_t nY = VLEN / inWidthHalf ;

    __vr vrseq = _vel_vseq_vl(VLEN) ;
    __vm256 vm_s0, vm_s2 ;
    {
      __vr vry_s0  = _vel_vdivsl_vvsl(vrseq, inWidthHalf, VLEN) ;
      __vr vrx_s0  = _vel_vsubsl_vvvl(vrseq, _vel_vmulul_vsvl(inWidthHalf,vry_s0, VLEN), VLEN) ;
      vm_s0 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(outWidthHalf, vrx_s0, VLEN), VLEN) ; // condition(x<outWidthHalf)

      __vr vrseq2  = _vel_vaddsl_vsvl(inWidthHalf-1, vrseq, VLEN) ;
      __vr vry_s2  = _vel_vdivsl_vvsl(vrseq2, inWidthHalf, VLEN) ;
      __vr vrx_s2  = _vel_vsubsl_vvvl(vrseq2, _vel_vmulul_vsvl(inWidthHalf,vry_s2, VLEN), VLEN) ;
      vm_s2 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(outWidthHalf, vrx_s2, VLEN), VLEN) ; // condition(x<outWidthHalf)
    }

    for (int64_t n = 0; n < batch; n++) {
      for (int64_t g = 0; g < group; g++) {
	const int64_t inGroupOffset   = g * inChannelGroup * inHeight * inWidth;
	const int64_t outGroupOffset  = g * outChannelGroup * outHeight * outWidth;
	const int64_t kernGroupOffset = g * outChannelGroup * inChannelGroup * kernHeight * kernWidth;

	int k = 0 ;
	if ( (outChannelGroup & 0x01) == 1 ) {
	  if( filter_layout == VEDNN_FILTER_LAYOUT_NCHW) {
	    k1<VEDNN_FILTER_LAYOUT_NCHW>(pIn, pKernel, pOut,
	       inChannel, inWidth, inHeight,
	       outChannel, outWidth, outHeight,
	       kernWidth, kernHeight,
	       inChannelGroup, outChannelGroup,
	       inGroupOffset, outGroupOffset, kernGroupOffset,
	       oPixels, nY,
	       inWidthHalf, outWidthHalf,
	       vm_s0, vm_s2,
	       n, k );
	  }
	  else {
	    k1<VEDNN_FILTER_LAYOUT_HWCN>(pIn, pKernel, pOut,
	       inChannel, inWidth, inHeight,
	       outChannel, outWidth, outHeight,
	       kernWidth, kernHeight,
	       inChannelGroup, outChannelGroup,
	       inGroupOffset, outGroupOffset, kernGroupOffset,
	       oPixels, nY,
	       inWidthHalf, outWidthHalf,
	       vm_s0, vm_s2,
	       n, k );
	  }

	  k++ ;
	}
	if ( ((outChannelGroup >> 1) & 0x01) == 1 ) {
	  if( filter_layout == VEDNN_FILTER_LAYOUT_NCHW) {
	    k2<VEDNN_FILTER_LAYOUT_NCHW>(pIn, pKernel, pOut,
	       inChannel, inWidth, inHeight,
	       outChannel, outWidth, outHeight,
	       kernWidth, kernHeight,
	       inChannelGroup, outChannelGroup,
	       inGroupOffset, outGroupOffset, kernGroupOffset,
	       oPixels, nY,
	       inWidthHalf, outWidthHalf,
	       vm_s0, vm_s2,
	       n, k );
	  }
	  else {
	    k2<VEDNN_FILTER_LAYOUT_HWCN>(pIn, pKernel, pOut,
	       inChannel, inWidth, inHeight,
	       outChannel, outWidth, outHeight,
	       kernWidth, kernHeight,
	       inChannelGroup, outChannelGroup,
	       inGroupOffset, outGroupOffset, kernGroupOffset,
	       oPixels, nY,
	       inWidthHalf, outWidthHalf,
	       vm_s0, vm_s2,
	       n, k );
	  }

	  k+=2 ;
	}
	if ( ((outChannelGroup >> 2) & 0x01) == 1 ) {
	  if( filter_layout == VEDNN_FILTER_LAYOUT_NCHW) {
	    k4<VEDNN_FILTER_LAYOUT_NCHW>(pIn, pKernel, pOut,
	       inChannel, inWidth, inHeight,
	       outChannel, outWidth, outHeight,
	       kernWidth, kernHeight,
	       inChannelGroup, outChannelGroup,
	       inGroupOffset, outGroupOffset, kernGroupOffset,
	       oPixels, nY,
	       inWidthHalf, outWidthHalf,
	       vm_s0, vm_s2,
	       n, k );
	  }
	  else {
	    k4<VEDNN_FILTER_LAYOUT_HWCN>(pIn, pKernel, pOut,
	       inChannel, inWidth, inHeight,
	       outChannel, outWidth, outHeight,
	       kernWidth, kernHeight,
	       inChannelGroup, outChannelGroup,
	       inGroupOffset, outGroupOffset, kernGroupOffset,
	       oPixels, nY,
	       inWidthHalf, outWidthHalf,
	       vm_s0, vm_s2,
	       n, k );
	  }

	  k+=4 ;
	}
	if ( ((outChannelGroup >> 3) & 0x01) == 1 ) {
	  if( filter_layout == VEDNN_FILTER_LAYOUT_NCHW) {
	    k8<VEDNN_FILTER_LAYOUT_NCHW>(pIn, pKernel, pOut,
	       inChannel, inWidth, inHeight,
	       outChannel, outWidth, outHeight,
	       kernWidth, kernHeight,
	       inChannelGroup, outChannelGroup,
	       inGroupOffset, outGroupOffset, kernGroupOffset,
	       oPixels, nY,
	       inWidthHalf, outWidthHalf,
	       vm_s0, vm_s2,
	       n, k );
	  }
	  else {
	    k8<VEDNN_FILTER_LAYOUT_HWCN>(pIn, pKernel, pOut,
	       inChannel, inWidth, inHeight,
	       outChannel, outWidth, outHeight,
	       kernWidth, kernHeight,
	       inChannelGroup, outChannelGroup,
	       inGroupOffset, outGroupOffset, kernGroupOffset,
	       oPixels, nY,
	       inWidthHalf, outWidthHalf,
	       vm_s0, vm_s2,
	       n, k );
	  }

	  k+=8 ;
	}
	for ( ; k < outChannelGroup; k+=16) {
	  if( filter_layout == VEDNN_FILTER_LAYOUT_NCHW) {
	    k16<VEDNN_FILTER_LAYOUT_NCHW>(pIn, pKernel, pOut,
	       inChannel, inWidth, inHeight,
	       outChannel, outWidth, outHeight,
	       kernWidth, kernHeight,
	       inChannelGroup, outChannelGroup,
	       inGroupOffset, outGroupOffset, kernGroupOffset,
	       oPixels, nY,
	       inWidthHalf, outWidthHalf,
	       vm_s0, vm_s2,
	       n, k );
	  }
	  else {
	    k16<VEDNN_FILTER_LAYOUT_HWCN>(pIn, pKernel, pOut,
	       inChannel, inWidth, inHeight,
	       outChannel, outWidth, outHeight,
	       kernWidth, kernHeight,
	       inChannelGroup, outChannelGroup,
	       inGroupOffset, outGroupOffset, kernGroupOffset,
	       oPixels, nY,
	       inWidthHalf, outWidthHalf,
	       vm_s0, vm_s2,
	       n, k );
	  }
	} // outChannel
      } // group
    } // batch
  }

  return VEDNN_SUCCESS;
}

