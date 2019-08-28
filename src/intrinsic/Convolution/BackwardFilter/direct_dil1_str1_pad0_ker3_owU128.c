#include <stdint.h>
#include <stdlib.h>

#include "vednn.h"

#include "velintrin.h"
#define VLEN	(256)


#define VSUM_STORE_3X3_UPPER(VRSUMTOKEN, KERNELINDEX)		\
{								\
__vr vrsumU_r0s0 = _vel_vfsums_vvl(VRSUMTOKEN ## _r0s0, VLEN) ;	\
__vr vrsumU_r0s1 = _vel_vfsums_vvl(VRSUMTOKEN ## _r0s1, VLEN) ;	\
__vr vrsumU_r0s2 = _vel_vfsums_vvl(VRSUMTOKEN ## _r0s2, VLEN) ;	\
__vr vrsumU_r1s0 = _vel_vfsums_vvl(VRSUMTOKEN ## _r1s0, VLEN) ;	\
__vr vrsumU_r1s1 = _vel_vfsums_vvl(VRSUMTOKEN ## _r1s1, VLEN) ;	\
__vr vrsumU_r1s2 = _vel_vfsums_vvl(VRSUMTOKEN ## _r1s2, VLEN) ;	\
__vr vrsumU_r2s0 = _vel_vfsums_vvl(VRSUMTOKEN ## _r2s0, VLEN) ;	\
__vr vrsumU_r2s1 = _vel_vfsums_vvl(VRSUMTOKEN ## _r2s1, VLEN) ;	\
__vr vrsumU_r2s2 = _vel_vfsums_vvl(VRSUMTOKEN ## _r2s2, VLEN) ;	\
_vel_vstu_vssl(vrsumU_r0s0,4,pGKernel+(KERNELINDEX)+0, 1) ;	\
_vel_vstu_vssl(vrsumU_r0s1,4,pGKernel+(KERNELINDEX)+1, 1) ;	\
_vel_vstu_vssl(vrsumU_r0s2,4,pGKernel+(KERNELINDEX)+2, 1) ;	\
_vel_vstu_vssl(vrsumU_r1s0,4,pGKernel+(KERNELINDEX)+3, 1) ;	\
_vel_vstu_vssl(vrsumU_r1s1,4,pGKernel+(KERNELINDEX)+4, 1) ;	\
_vel_vstu_vssl(vrsumU_r1s2,4,pGKernel+(KERNELINDEX)+5, 1) ;	\
_vel_vstu_vssl(vrsumU_r2s0,4,pGKernel+(KERNELINDEX)+6, 1) ;	\
_vel_vstu_vssl(vrsumU_r2s1,4,pGKernel+(KERNELINDEX)+7, 1) ;	\
_vel_vstu_vssl(vrsumU_r2s2,4,pGKernel+(KERNELINDEX)+8, 1) ;	\
}
#define VSUM_STORE_3X3_LOWER(VRSUMTOKEN, KERNELINDEX)				\
{										\
__vr vrsumL_r0s0 = _vel_vfsums_vvl(_vel_vsll_vvsl(VRSUMTOKEN ## _r0s0,32, VLEN), VLEN) ;	\
__vr vrsumL_r0s1 = _vel_vfsums_vvl(_vel_vsll_vvsl(VRSUMTOKEN ## _r0s1,32, VLEN), VLEN) ;	\
__vr vrsumL_r0s2 = _vel_vfsums_vvl(_vel_vsll_vvsl(VRSUMTOKEN ## _r0s2,32, VLEN), VLEN) ;	\
__vr vrsumL_r1s0 = _vel_vfsums_vvl(_vel_vsll_vvsl(VRSUMTOKEN ## _r1s0,32, VLEN), VLEN) ;	\
__vr vrsumL_r1s1 = _vel_vfsums_vvl(_vel_vsll_vvsl(VRSUMTOKEN ## _r1s1,32, VLEN), VLEN) ;	\
__vr vrsumL_r1s2 = _vel_vfsums_vvl(_vel_vsll_vvsl(VRSUMTOKEN ## _r1s2,32, VLEN), VLEN) ;	\
__vr vrsumL_r2s0 = _vel_vfsums_vvl(_vel_vsll_vvsl(VRSUMTOKEN ## _r2s0,32, VLEN), VLEN) ;	\
__vr vrsumL_r2s1 = _vel_vfsums_vvl(_vel_vsll_vvsl(VRSUMTOKEN ## _r2s1,32, VLEN), VLEN) ;	\
__vr vrsumL_r2s2 = _vel_vfsums_vvl(_vel_vsll_vvsl(VRSUMTOKEN ## _r2s2,32, VLEN), VLEN) ;	\
_vel_vstu_vssl(vrsumL_r0s0,4,pGKernel+(KERNELINDEX)+0, 1) ;			\
_vel_vstu_vssl(vrsumL_r0s1,4,pGKernel+(KERNELINDEX)+1, 1) ;			\
_vel_vstu_vssl(vrsumL_r0s2,4,pGKernel+(KERNELINDEX)+2, 1) ;			\
_vel_vstu_vssl(vrsumL_r1s0,4,pGKernel+(KERNELINDEX)+3, 1) ;			\
_vel_vstu_vssl(vrsumL_r1s1,4,pGKernel+(KERNELINDEX)+4, 1) ;			\
_vel_vstu_vssl(vrsumL_r1s2,4,pGKernel+(KERNELINDEX)+5, 1) ;			\
_vel_vstu_vssl(vrsumL_r2s0,4,pGKernel+(KERNELINDEX)+6, 1) ;			\
_vel_vstu_vssl(vrsumL_r2s1,4,pGKernel+(KERNELINDEX)+7, 1) ;			\
_vel_vstu_vssl(vrsumL_r2s2,4,pGKernel+(KERNELINDEX)+8, 1) ;			\
}


static inline void k1(
  const float * restrict pIn,
  const float * restrict pGOut,
  float * restrict const pGKernel,
  const int64_t inChannel,
  const int64_t inWidth,
  const int64_t inHeight,
  const int64_t gOutChannel,
  const int64_t gOutWidth,
  const int64_t gOutHeight,
  const int64_t gKernWidth,
  const int64_t gKernHeight,
  const int64_t inChannelGroup,
  const int64_t inGroupOffset,
  const int64_t outGroupOffset,
  const int64_t kernGroupOffset,
  const int64_t batch,
  const int64_t k,
  const __vm256 vm_s0,
  const __vm256 vm_s1,
  const __vm256 vm_s2,
  const int64_t nY
)
{
  for (int64_t c=0; c<inChannelGroup; c++) {
    const int64_t kernelIndex0 = kernGroupOffset + ((k     * inChannelGroup + c) * gKernHeight ) * gKernWidth ;

    __vr vrsum_r0s0 = _vel_vbrds_vsl(0.0f, VLEN) ;
    __vr vrsum_r0s1 = _vel_vbrds_vsl(0.0f, VLEN) ;
    __vr vrsum_r0s2 = _vel_vbrds_vsl(0.0f, VLEN) ;
    __vr vrsum_r1s0 = _vel_vbrds_vsl(0.0f, VLEN) ;
    __vr vrsum_r1s1 = _vel_vbrds_vsl(0.0f, VLEN) ;
    __vr vrsum_r1s2 = _vel_vbrds_vsl(0.0f, VLEN) ;
    __vr vrsum_r2s0 = _vel_vbrds_vsl(0.0f, VLEN) ;
    __vr vrsum_r2s1 = _vel_vbrds_vsl(0.0f, VLEN) ;
    __vr vrsum_r2s2 = _vel_vbrds_vsl(0.0f, VLEN) ;

    for (int64_t y=0; y<gOutHeight; y+=nY) {

      const int64_t vl0 = inWidth   * (gOutHeight - y < nY ? gOutHeight - y : nY) ;
      const int64_t vl1 = gOutWidth * (gOutHeight - y < nY ? gOutHeight - y : nY) ;

      const int64_t gop = y * gOutWidth ;

      for (int64_t n=0; n<batch; n++) {

	const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;

	const int64_t gOutIndex0  = outGroupOffset + ((n * gOutChannel + k  ) * gOutHeight ) * gOutWidth + gop;

	__vr vrin_r0 = _vel_vldu_vssl(4, pInChannel+(y+0)*inWidth, vl0) ;
	__vr vrin_r1 = _vel_vldu_vssl(4, pInChannel+(y+1)*inWidth, vl0) ;
	__vr vrin_r2 = _vel_vldu_vssl(4, pInChannel+(y+2)*inWidth, vl0) ;

	__vr vrgout0 = _vel_vldu_vssl(4, pGOut+gOutIndex0, vl1) ;

	__vr vrin_r0s0  = _vel_vcp_vvmvl(vrin_r0, vm_s0, _vel_vbrdl_vsl(0UL, vl0), vl0) ;
	__vr vrin_r0s1  = _vel_vcp_vvmvl(vrin_r0, vm_s1, _vel_vbrdl_vsl(0UL, vl0), vl0) ;
	__vr vrin_r0s2  = _vel_vcp_vvmvl(vrin_r0, vm_s2, _vel_vbrdl_vsl(0UL, vl0), vl0) ;
	__vr vrin_r1s0  = _vel_vcp_vvmvl(vrin_r1, vm_s0, _vel_vbrdl_vsl(0UL, vl0), vl0) ;
	__vr vrin_r1s1  = _vel_vcp_vvmvl(vrin_r1, vm_s1, _vel_vbrdl_vsl(0UL, vl0), vl0) ;
	__vr vrin_r1s2  = _vel_vcp_vvmvl(vrin_r1, vm_s2, _vel_vbrdl_vsl(0UL, vl0), vl0) ;
	__vr vrin_r2s0  = _vel_vcp_vvmvl(vrin_r2, vm_s0, _vel_vbrdl_vsl(0UL, vl0), vl0) ;
	__vr vrin_r2s1  = _vel_vcp_vvmvl(vrin_r2, vm_s1, _vel_vbrdl_vsl(0UL, vl0), vl0) ;
	__vr vrin_r2s2  = _vel_vcp_vvmvl(vrin_r2, vm_s2, _vel_vbrdl_vsl(0UL, vl0), vl0) ;

	vrsum_r0s0 = _vel_vfmads_vvvvvl(vrsum_r0s0, vrin_r0s0, vrgout0, vrsum_r0s0, vl1) ;
	vrsum_r0s1 = _vel_vfmads_vvvvvl(vrsum_r0s1, vrin_r0s1, vrgout0, vrsum_r0s1, vl1) ;
	vrsum_r0s2 = _vel_vfmads_vvvvvl(vrsum_r0s2, vrin_r0s2, vrgout0, vrsum_r0s2, vl1) ;
	vrsum_r1s0 = _vel_vfmads_vvvvvl(vrsum_r1s0, vrin_r1s0, vrgout0, vrsum_r1s0, vl1) ;
	vrsum_r1s1 = _vel_vfmads_vvvvvl(vrsum_r1s1, vrin_r1s1, vrgout0, vrsum_r1s1, vl1) ;
	vrsum_r1s2 = _vel_vfmads_vvvvvl(vrsum_r1s2, vrin_r1s2, vrgout0, vrsum_r1s2, vl1) ;
	vrsum_r2s0 = _vel_vfmads_vvvvvl(vrsum_r2s0, vrin_r2s0, vrgout0, vrsum_r2s0, vl1) ;
	vrsum_r2s1 = _vel_vfmads_vvvvvl(vrsum_r2s1, vrin_r2s1, vrgout0, vrsum_r2s1, vl1) ;
	vrsum_r2s2 = _vel_vfmads_vvvvvl(vrsum_r2s2, vrin_r2s2, vrgout0, vrsum_r2s2, vl1) ;

      } // batch
    } // gOutPixels

    VSUM_STORE_3X3_UPPER(vrsum, kernelIndex0) ;

  } // inChannel
}

static inline void k2(
  const float * restrict pIn,
  const float * restrict pGOut,
  float * restrict const pGKernel,
  const int64_t inChannel,
  const int64_t inWidth,
  const int64_t inHeight,
  const int64_t gOutChannel,
  const int64_t gOutWidth,
  const int64_t gOutHeight,
  const int64_t gKernWidth,
  const int64_t gKernHeight,
  const int64_t inChannelGroup,
  const int64_t inGroupOffset,
  const int64_t outGroupOffset,
  const int64_t kernGroupOffset,
  const int64_t batch,
  const int64_t k,
  const __vm256 vm_s0,
  const __vm256 vm_s1,
  const __vm256 vm_s2,
  const int64_t nY
)
{
  for (int64_t c=0; c<inChannelGroup; c++) {
    const int64_t kernelIndex0 = kernGroupOffset + ((k     * inChannelGroup + c) * gKernHeight ) * gKernWidth ;
    const int64_t kernelIndex1 = kernGroupOffset + (((k+1) * inChannelGroup + c) * gKernHeight ) * gKernWidth ;

#define INIT_VRSUM_2(TOKEN, INDEX)	\
__vr vrsum01_ ## TOKEN = _vel_vbrdl_vsl(0UL, VLEN) ;

    INIT_VRSUM_2(r0s0, 0) ;
    INIT_VRSUM_2(r0s1, 1) ;
    INIT_VRSUM_2(r0s2, 2) ;
    INIT_VRSUM_2(r1s0, 3) ;
    INIT_VRSUM_2(r1s1, 4) ;
    INIT_VRSUM_2(r1s2, 5) ;
    INIT_VRSUM_2(r2s0, 6) ;
    INIT_VRSUM_2(r2s1, 7) ;
    INIT_VRSUM_2(r2s2, 8) ;
#undef INIT_VRSUM_2

    for (int64_t y=0; y<gOutHeight; y+=nY) {

      const int64_t vl0 = inWidth   * (gOutHeight - y < nY ? gOutHeight - y : nY) ;
      const int64_t vl1 = gOutWidth * (gOutHeight - y < nY ? gOutHeight - y : nY) ;

      const int64_t gop = y * gOutWidth ;

      for (int64_t n=0; n<batch; n++) {

	const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;

	const int64_t gOutIndex0  = outGroupOffset + ((n * gOutChannel + k  ) * gOutHeight ) * gOutWidth + gop;
	const int64_t gOutIndex1  = outGroupOffset + ((n * gOutChannel + k+1) * gOutHeight ) * gOutWidth + gop;

	__vr vrin_r0 = _vel_vldu_vssl(4, pInChannel+(y+0)*inWidth, vl0) ;
	__vr vrin_r1 = _vel_vldu_vssl(4, pInChannel+(y+1)*inWidth, vl0) ;
	__vr vrin_r2 = _vel_vldu_vssl(4, pInChannel+(y+2)*inWidth, vl0) ;

	__vr vrgout0 = _vel_vldu_vssl(4, pGOut+gOutIndex0, vl1) ;
	__vr vrgout1 = _vel_vldu_vssl(4, pGOut+gOutIndex1, vl1) ;

	__vr vrin_r0s0  = _vel_vcp_vvmvl(vrin_r0, vm_s0, _vel_vbrdl_vsl(0UL, vl0), vl0) ;
	__vr vrinP_r0s0 = _vel_vshf_vvvsl(vrin_r0s0, vrin_r0s0, VE_VSHUFFLE_YUZU, vl0) ;
	__vr vrin_r0s1  = _vel_vcp_vvmvl(vrin_r0, vm_s1, _vel_vbrdl_vsl(0UL, vl0), vl0) ;
	__vr vrinP_r0s1 = _vel_vshf_vvvsl(vrin_r0s1, vrin_r0s1, VE_VSHUFFLE_YUZU, vl0) ;
	__vr vrin_r0s2  = _vel_vcp_vvmvl(vrin_r0, vm_s2, _vel_vbrdl_vsl(0UL, vl0), vl0) ;
	__vr vrinP_r0s2 = _vel_vshf_vvvsl(vrin_r0s2, vrin_r0s2, VE_VSHUFFLE_YUZU, vl0) ;
	__vr vrin_r1s0  = _vel_vcp_vvmvl(vrin_r1, vm_s0, _vel_vbrdl_vsl(0UL, vl0), vl0) ;
	__vr vrinP_r1s0 = _vel_vshf_vvvsl(vrin_r1s0, vrin_r1s0, VE_VSHUFFLE_YUZU, vl0) ;
	__vr vrin_r1s1  = _vel_vcp_vvmvl(vrin_r1, vm_s1, _vel_vbrdl_vsl(0UL, vl0), vl0) ;
	__vr vrinP_r1s1 = _vel_vshf_vvvsl(vrin_r1s1, vrin_r1s1, VE_VSHUFFLE_YUZU, vl0) ;
	__vr vrin_r1s2  = _vel_vcp_vvmvl(vrin_r1, vm_s2, _vel_vbrdl_vsl(0UL, vl0), vl0) ;
	__vr vrinP_r1s2 = _vel_vshf_vvvsl(vrin_r1s2, vrin_r1s2, VE_VSHUFFLE_YUZU, vl0) ;
	__vr vrin_r2s0  = _vel_vcp_vvmvl(vrin_r2, vm_s0, _vel_vbrdl_vsl(0UL, vl0), vl0) ;
	__vr vrinP_r2s0 = _vel_vshf_vvvsl(vrin_r2s0, vrin_r2s0, VE_VSHUFFLE_YUZU, vl0) ;
	__vr vrin_r2s1  = _vel_vcp_vvmvl(vrin_r2, vm_s1, _vel_vbrdl_vsl(0UL, vl0), vl0) ;
	__vr vrinP_r2s1 = _vel_vshf_vvvsl(vrin_r2s1, vrin_r2s1, VE_VSHUFFLE_YUZU, vl0) ;
	__vr vrin_r2s2  = _vel_vcp_vvmvl(vrin_r2, vm_s2, _vel_vbrdl_vsl(0UL, vl0), vl0) ;
	__vr vrinP_r2s2 = _vel_vshf_vvvsl(vrin_r2s2, vrin_r2s2, VE_VSHUFFLE_YUZU, vl0) ;

	__vr vrgout01 = _vel_vshf_vvvsl(vrgout0, vrgout1, VE_VSHUFFLE_YUZU, vl1) ;
	vrsum01_r0s0 = _vel_pvfmad_vvvvvl(vrsum01_r0s0, vrinP_r0s0, vrgout01, vrsum01_r0s0, vl1) ;
	vrsum01_r0s1 = _vel_pvfmad_vvvvvl(vrsum01_r0s1, vrinP_r0s1, vrgout01, vrsum01_r0s1, vl1) ;
	vrsum01_r0s2 = _vel_pvfmad_vvvvvl(vrsum01_r0s2, vrinP_r0s2, vrgout01, vrsum01_r0s2, vl1) ;
	vrsum01_r1s0 = _vel_pvfmad_vvvvvl(vrsum01_r1s0, vrinP_r1s0, vrgout01, vrsum01_r1s0, vl1) ;
	vrsum01_r1s1 = _vel_pvfmad_vvvvvl(vrsum01_r1s1, vrinP_r1s1, vrgout01, vrsum01_r1s1, vl1) ;
	vrsum01_r1s2 = _vel_pvfmad_vvvvvl(vrsum01_r1s2, vrinP_r1s2, vrgout01, vrsum01_r1s2, vl1) ;
	vrsum01_r2s0 = _vel_pvfmad_vvvvvl(vrsum01_r2s0, vrinP_r2s0, vrgout01, vrsum01_r2s0, vl1) ;
	vrsum01_r2s1 = _vel_pvfmad_vvvvvl(vrsum01_r2s1, vrinP_r2s1, vrgout01, vrsum01_r2s1, vl1) ;
	vrsum01_r2s2 = _vel_pvfmad_vvvvvl(vrsum01_r2s2, vrinP_r2s2, vrgout01, vrsum01_r2s2, vl1) ;

      } // batch
    } // gOutPixels

    VSUM_STORE_3X3_UPPER(vrsum01, kernelIndex0) ;
    VSUM_STORE_3X3_LOWER(vrsum01, kernelIndex1) ;

  } // inChannel
}

static inline void k4(
  const float * restrict pIn,
  const float * restrict pGOut,
  float * restrict const pGKernel,
  const int64_t inChannel,
  const int64_t inWidth,
  const int64_t inHeight,
  const int64_t gOutChannel,
  const int64_t gOutWidth,
  const int64_t gOutHeight,
  const int64_t gKernWidth,
  const int64_t gKernHeight,
  const int64_t inChannelGroup,
  const int64_t inGroupOffset,
  const int64_t outGroupOffset,
  const int64_t kernGroupOffset,
  const int64_t batch,
  const int64_t k,
  const __vm256 vm_s0,
  const __vm256 vm_s1,
  const __vm256 vm_s2,
  const int64_t nY
)
{
  for (int64_t c=0; c<inChannelGroup; c++) {
    const int64_t kernelIndex0 = kernGroupOffset + ((k     * inChannelGroup + c) * gKernHeight ) * gKernWidth ;
    const int64_t kernelIndex1 = kernGroupOffset + (((k+1) * inChannelGroup + c) * gKernHeight ) * gKernWidth ;
    const int64_t kernelIndex2 = kernGroupOffset + (((k+2) * inChannelGroup + c) * gKernHeight ) * gKernWidth ;
    const int64_t kernelIndex3 = kernGroupOffset + (((k+3) * inChannelGroup + c) * gKernHeight ) * gKernWidth ;

#define INIT_VRSUM_4(TOKEN, INDEX)	\
__vr vrsum01_ ## TOKEN = _vel_vbrdl_vsl(0UL, VLEN) ;	\
__vr vrsum23_ ## TOKEN = _vel_vbrdl_vsl(0UL, VLEN) ;

    INIT_VRSUM_4(r0s0, 0) ;
    INIT_VRSUM_4(r0s1, 1) ;
    INIT_VRSUM_4(r0s2, 2) ;
    INIT_VRSUM_4(r1s0, 3) ;
    INIT_VRSUM_4(r1s1, 4) ;
    INIT_VRSUM_4(r1s2, 5) ;
    INIT_VRSUM_4(r2s0, 6) ;
    INIT_VRSUM_4(r2s1, 7) ;
    INIT_VRSUM_4(r2s2, 8) ;
#undef INIT_VRSUM_4

    for (int64_t y=0; y<gOutHeight; y+=nY) {

      const int64_t vl0 = inWidth   * (gOutHeight - y < nY ? gOutHeight - y : nY) ;
      const int64_t vl1 = gOutWidth * (gOutHeight - y < nY ? gOutHeight - y : nY) ;

      const int64_t gop = y * gOutWidth ;

      for (int64_t n=0; n<batch; n++) {

	const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;

	const int64_t gOutIndex0  = outGroupOffset + ((n * gOutChannel + k  ) * gOutHeight ) * gOutWidth + gop;
	const int64_t gOutIndex1  = outGroupOffset + ((n * gOutChannel + k+1) * gOutHeight ) * gOutWidth + gop;
	const int64_t gOutIndex2  = outGroupOffset + ((n * gOutChannel + k+2) * gOutHeight ) * gOutWidth + gop;
	const int64_t gOutIndex3  = outGroupOffset + ((n * gOutChannel + k+3) * gOutHeight ) * gOutWidth + gop;

	__vr vrin_r0 = _vel_vldu_vssl(4, pInChannel+(y+0)*inWidth, vl0) ;
	__vr vrin_r1 = _vel_vldu_vssl(4, pInChannel+(y+1)*inWidth, vl0) ;
	__vr vrin_r2 = _vel_vldu_vssl(4, pInChannel+(y+2)*inWidth, vl0) ;

	__vr vrgout0 = _vel_vldu_vssl(4, pGOut+gOutIndex0, vl1) ;
	__vr vrgout1 = _vel_vldu_vssl(4, pGOut+gOutIndex1, vl1) ;
	__vr vrgout2 = _vel_vldu_vssl(4, pGOut+gOutIndex2, vl1) ;
	__vr vrgout3 = _vel_vldu_vssl(4, pGOut+gOutIndex3, vl1) ;

	__vr vrin_r0s0  = _vel_vcp_vvmvl(vrin_r0, vm_s0, _vel_vbrdl_vsl(0UL, vl0), vl0) ;
	__vr vrinP_r0s0 = _vel_vshf_vvvsl(vrin_r0s0, vrin_r0s0, VE_VSHUFFLE_YUZU, vl0) ;
	__vr vrin_r0s1  = _vel_vcp_vvmvl(vrin_r0, vm_s1, _vel_vbrdl_vsl(0UL, vl0), vl0) ;
	__vr vrinP_r0s1 = _vel_vshf_vvvsl(vrin_r0s1, vrin_r0s1, VE_VSHUFFLE_YUZU, vl0) ;
	__vr vrin_r0s2  = _vel_vcp_vvmvl(vrin_r0, vm_s2, _vel_vbrdl_vsl(0UL, vl0), vl0) ;
	__vr vrinP_r0s2 = _vel_vshf_vvvsl(vrin_r0s2, vrin_r0s2, VE_VSHUFFLE_YUZU, vl0) ;
	__vr vrin_r1s0  = _vel_vcp_vvmvl(vrin_r1, vm_s0, _vel_vbrdl_vsl(0UL, vl0), vl0) ;
	__vr vrinP_r1s0 = _vel_vshf_vvvsl(vrin_r1s0, vrin_r1s0, VE_VSHUFFLE_YUZU, vl0) ;
	__vr vrin_r1s1  = _vel_vcp_vvmvl(vrin_r1, vm_s1, _vel_vbrdl_vsl(0UL, vl0), vl0) ;
	__vr vrinP_r1s1 = _vel_vshf_vvvsl(vrin_r1s1, vrin_r1s1, VE_VSHUFFLE_YUZU, vl0) ;
	__vr vrin_r1s2  = _vel_vcp_vvmvl(vrin_r1, vm_s2, _vel_vbrdl_vsl(0UL, vl0), vl0) ;
	__vr vrinP_r1s2 = _vel_vshf_vvvsl(vrin_r1s2, vrin_r1s2, VE_VSHUFFLE_YUZU, vl0) ;
	__vr vrin_r2s0  = _vel_vcp_vvmvl(vrin_r2, vm_s0, _vel_vbrdl_vsl(0UL, vl0), vl0) ;
	__vr vrinP_r2s0 = _vel_vshf_vvvsl(vrin_r2s0, vrin_r2s0, VE_VSHUFFLE_YUZU, vl0) ;
	__vr vrin_r2s1  = _vel_vcp_vvmvl(vrin_r2, vm_s1, _vel_vbrdl_vsl(0UL, vl0), vl0) ;
	__vr vrinP_r2s1 = _vel_vshf_vvvsl(vrin_r2s1, vrin_r2s1, VE_VSHUFFLE_YUZU, vl0) ;
	__vr vrin_r2s2  = _vel_vcp_vvmvl(vrin_r2, vm_s2, _vel_vbrdl_vsl(0UL, vl0), vl0) ;
	__vr vrinP_r2s2 = _vel_vshf_vvvsl(vrin_r2s2, vrin_r2s2, VE_VSHUFFLE_YUZU, vl0) ;

	__vr vrgout01 = _vel_vshf_vvvsl(vrgout0, vrgout1, VE_VSHUFFLE_YUZU, vl1) ;
	vrsum01_r0s0 = _vel_pvfmad_vvvvvl(vrsum01_r0s0, vrinP_r0s0, vrgout01, vrsum01_r0s0, vl1) ;
	vrsum01_r0s1 = _vel_pvfmad_vvvvvl(vrsum01_r0s1, vrinP_r0s1, vrgout01, vrsum01_r0s1, vl1) ;
	vrsum01_r0s2 = _vel_pvfmad_vvvvvl(vrsum01_r0s2, vrinP_r0s2, vrgout01, vrsum01_r0s2, vl1) ;
	vrsum01_r1s0 = _vel_pvfmad_vvvvvl(vrsum01_r1s0, vrinP_r1s0, vrgout01, vrsum01_r1s0, vl1) ;
	vrsum01_r1s1 = _vel_pvfmad_vvvvvl(vrsum01_r1s1, vrinP_r1s1, vrgout01, vrsum01_r1s1, vl1) ;
	vrsum01_r1s2 = _vel_pvfmad_vvvvvl(vrsum01_r1s2, vrinP_r1s2, vrgout01, vrsum01_r1s2, vl1) ;
	vrsum01_r2s0 = _vel_pvfmad_vvvvvl(vrsum01_r2s0, vrinP_r2s0, vrgout01, vrsum01_r2s0, vl1) ;
	vrsum01_r2s1 = _vel_pvfmad_vvvvvl(vrsum01_r2s1, vrinP_r2s1, vrgout01, vrsum01_r2s1, vl1) ;
	vrsum01_r2s2 = _vel_pvfmad_vvvvvl(vrsum01_r2s2, vrinP_r2s2, vrgout01, vrsum01_r2s2, vl1) ;

	__vr vrgout23 = _vel_vshf_vvvsl(vrgout2, vrgout3, VE_VSHUFFLE_YUZU, vl1) ;
	vrsum23_r0s0 = _vel_pvfmad_vvvvvl(vrsum23_r0s0, vrinP_r0s0, vrgout23, vrsum23_r0s0, vl1) ;
	vrsum23_r0s1 = _vel_pvfmad_vvvvvl(vrsum23_r0s1, vrinP_r0s1, vrgout23, vrsum23_r0s1, vl1) ;
	vrsum23_r0s2 = _vel_pvfmad_vvvvvl(vrsum23_r0s2, vrinP_r0s2, vrgout23, vrsum23_r0s2, vl1) ;
	vrsum23_r1s0 = _vel_pvfmad_vvvvvl(vrsum23_r1s0, vrinP_r1s0, vrgout23, vrsum23_r1s0, vl1) ;
	vrsum23_r1s1 = _vel_pvfmad_vvvvvl(vrsum23_r1s1, vrinP_r1s1, vrgout23, vrsum23_r1s1, vl1) ;
	vrsum23_r1s2 = _vel_pvfmad_vvvvvl(vrsum23_r1s2, vrinP_r1s2, vrgout23, vrsum23_r1s2, vl1) ;
	vrsum23_r2s0 = _vel_pvfmad_vvvvvl(vrsum23_r2s0, vrinP_r2s0, vrgout23, vrsum23_r2s0, vl1) ;
	vrsum23_r2s1 = _vel_pvfmad_vvvvvl(vrsum23_r2s1, vrinP_r2s1, vrgout23, vrsum23_r2s1, vl1) ;
	vrsum23_r2s2 = _vel_pvfmad_vvvvvl(vrsum23_r2s2, vrinP_r2s2, vrgout23, vrsum23_r2s2, vl1) ;

      } // batch
    } // gOutPixels

    VSUM_STORE_3X3_UPPER(vrsum01, kernelIndex0) ;
    VSUM_STORE_3X3_LOWER(vrsum01, kernelIndex1) ;
    VSUM_STORE_3X3_UPPER(vrsum23, kernelIndex2) ;
    VSUM_STORE_3X3_LOWER(vrsum23, kernelIndex3) ;

  } // inChannel
}

static inline void k8(
  const float * restrict pIn,
  const float * restrict pGOut,
  float * restrict const pGKernel,
  const int64_t inChannel,
  const int64_t inWidth,
  const int64_t inHeight,
  const int64_t gOutChannel,
  const int64_t gOutWidth,
  const int64_t gOutHeight,
  const int64_t gKernWidth,
  const int64_t gKernHeight,
  const int64_t inChannelGroup,
  const int64_t inGroupOffset,
  const int64_t outGroupOffset,
  const int64_t kernGroupOffset,
  const int64_t batch,
  const int64_t k,
  const __vm256 vm_s0,
  const __vm256 vm_s1,
  const __vm256 vm_s2,
  const int64_t nY
)
{
  for (int64_t c=0; c<inChannelGroup; c++) {
    const int64_t kernelIndex0 = kernGroupOffset + ((k     * inChannelGroup + c) * gKernHeight ) * gKernWidth ;
    const int64_t kernelIndex1 = kernGroupOffset + (((k+1) * inChannelGroup + c) * gKernHeight ) * gKernWidth ;
    const int64_t kernelIndex2 = kernGroupOffset + (((k+2) * inChannelGroup + c) * gKernHeight ) * gKernWidth ;
    const int64_t kernelIndex3 = kernGroupOffset + (((k+3) * inChannelGroup + c) * gKernHeight ) * gKernWidth ;
    const int64_t kernelIndex4 = kernGroupOffset + (((k+4) * inChannelGroup + c) * gKernHeight ) * gKernWidth ;
    const int64_t kernelIndex5 = kernGroupOffset + (((k+5) * inChannelGroup + c) * gKernHeight ) * gKernWidth ;
    const int64_t kernelIndex6 = kernGroupOffset + (((k+6) * inChannelGroup + c) * gKernHeight ) * gKernWidth ;
    const int64_t kernelIndex7 = kernGroupOffset + (((k+7) * inChannelGroup + c) * gKernHeight ) * gKernWidth ;

#define INIT_VRSUM_8(TOKEN, INDEX)	\
__vr vrsum01_ ## TOKEN = _vel_vbrdl_vsl(0UL, VLEN) ;	\
__vr vrsum23_ ## TOKEN = _vel_vbrdl_vsl(0UL, VLEN) ;	\
__vr vrsum45_ ## TOKEN = _vel_vbrdl_vsl(0UL, VLEN) ;	\
__vr vrsum67_ ## TOKEN = _vel_vbrdl_vsl(0UL, VLEN) ;

    INIT_VRSUM_8(r0s0, 0) ;
    INIT_VRSUM_8(r0s1, 1) ;
    INIT_VRSUM_8(r0s2, 2) ;
    INIT_VRSUM_8(r1s0, 3) ;
    INIT_VRSUM_8(r1s1, 4) ;
    INIT_VRSUM_8(r1s2, 5) ;
    INIT_VRSUM_8(r2s0, 6) ;
    INIT_VRSUM_8(r2s1, 7) ;
    INIT_VRSUM_8(r2s2, 8) ;
#undef INIT_VRSUM_8

    for (int64_t y=0; y<gOutHeight; y+=nY) {

      const int64_t vl0 = inWidth   * (gOutHeight - y < nY ? gOutHeight - y : nY) ;
      const int64_t vl1 = gOutWidth * (gOutHeight - y < nY ? gOutHeight - y : nY) ;

      const int64_t gop = y * gOutWidth ;

      for (int64_t n=0; n<batch; n++) {

	const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;

	const int64_t gOutIndex0  = outGroupOffset + ((n * gOutChannel + k  ) * gOutHeight ) * gOutWidth + gop;
	const int64_t gOutIndex1  = outGroupOffset + ((n * gOutChannel + k+1) * gOutHeight ) * gOutWidth + gop;
	const int64_t gOutIndex2  = outGroupOffset + ((n * gOutChannel + k+2) * gOutHeight ) * gOutWidth + gop;
	const int64_t gOutIndex3  = outGroupOffset + ((n * gOutChannel + k+3) * gOutHeight ) * gOutWidth + gop;
	const int64_t gOutIndex4  = outGroupOffset + ((n * gOutChannel + k+4) * gOutHeight ) * gOutWidth + gop;
	const int64_t gOutIndex5  = outGroupOffset + ((n * gOutChannel + k+5) * gOutHeight ) * gOutWidth + gop;
	const int64_t gOutIndex6  = outGroupOffset + ((n * gOutChannel + k+6) * gOutHeight ) * gOutWidth + gop;
	const int64_t gOutIndex7  = outGroupOffset + ((n * gOutChannel + k+7) * gOutHeight ) * gOutWidth + gop;

	__vr vrin_r0 = _vel_vldu_vssl(4, pInChannel+(y+0)*inWidth, vl0) ;
	__vr vrin_r1 = _vel_vldu_vssl(4, pInChannel+(y+1)*inWidth, vl0) ;
	__vr vrin_r2 = _vel_vldu_vssl(4, pInChannel+(y+2)*inWidth, vl0) ;

	__vr vrgout0 = _vel_vldu_vssl(4, pGOut+gOutIndex0, vl1) ;
	__vr vrgout1 = _vel_vldu_vssl(4, pGOut+gOutIndex1, vl1) ;
	__vr vrgout2 = _vel_vldu_vssl(4, pGOut+gOutIndex2, vl1) ;
	__vr vrgout3 = _vel_vldu_vssl(4, pGOut+gOutIndex3, vl1) ;
	__vr vrgout4 = _vel_vldu_vssl(4, pGOut+gOutIndex4, vl1) ;
	__vr vrgout5 = _vel_vldu_vssl(4, pGOut+gOutIndex5, vl1) ;
	__vr vrgout6 = _vel_vldu_vssl(4, pGOut+gOutIndex6, vl1) ;
	__vr vrgout7 = _vel_vldu_vssl(4, pGOut+gOutIndex7, vl1) ;

	__vr vrin_r0s0  = _vel_vcp_vvmvl(vrin_r0, vm_s0, _vel_vbrdl_vsl(0UL, vl0), vl0) ;
	__vr vrinP_r0s0 = _vel_vshf_vvvsl(vrin_r0s0, vrin_r0s0, VE_VSHUFFLE_YUZU, vl0) ;
	__vr vrin_r0s1  = _vel_vcp_vvmvl(vrin_r0, vm_s1, _vel_vbrdl_vsl(0UL, vl0), vl0) ;
	__vr vrinP_r0s1 = _vel_vshf_vvvsl(vrin_r0s1, vrin_r0s1, VE_VSHUFFLE_YUZU, vl0) ;
	__vr vrin_r0s2  = _vel_vcp_vvmvl(vrin_r0, vm_s2, _vel_vbrdl_vsl(0UL, vl0), vl0) ;
	__vr vrinP_r0s2 = _vel_vshf_vvvsl(vrin_r0s2, vrin_r0s2, VE_VSHUFFLE_YUZU, vl0) ;
	__vr vrin_r1s0  = _vel_vcp_vvmvl(vrin_r1, vm_s0, _vel_vbrdl_vsl(0UL, vl0), vl0) ;
	__vr vrinP_r1s0 = _vel_vshf_vvvsl(vrin_r1s0, vrin_r1s0, VE_VSHUFFLE_YUZU, vl0) ;
	__vr vrin_r1s1  = _vel_vcp_vvmvl(vrin_r1, vm_s1, _vel_vbrdl_vsl(0UL, vl0), vl0) ;
	__vr vrinP_r1s1 = _vel_vshf_vvvsl(vrin_r1s1, vrin_r1s1, VE_VSHUFFLE_YUZU, vl0) ;
	__vr vrin_r1s2  = _vel_vcp_vvmvl(vrin_r1, vm_s2, _vel_vbrdl_vsl(0UL, vl0), vl0) ;
	__vr vrinP_r1s2 = _vel_vshf_vvvsl(vrin_r1s2, vrin_r1s2, VE_VSHUFFLE_YUZU, vl0) ;
	__vr vrin_r2s0  = _vel_vcp_vvmvl(vrin_r2, vm_s0, _vel_vbrdl_vsl(0UL, vl0), vl0) ;
	__vr vrinP_r2s0 = _vel_vshf_vvvsl(vrin_r2s0, vrin_r2s0, VE_VSHUFFLE_YUZU, vl0) ;
	__vr vrin_r2s1  = _vel_vcp_vvmvl(vrin_r2, vm_s1, _vel_vbrdl_vsl(0UL, vl0), vl0) ;
	__vr vrinP_r2s1 = _vel_vshf_vvvsl(vrin_r2s1, vrin_r2s1, VE_VSHUFFLE_YUZU, vl0) ;
	__vr vrin_r2s2  = _vel_vcp_vvmvl(vrin_r2, vm_s2, _vel_vbrdl_vsl(0UL, vl0), vl0) ;
	__vr vrinP_r2s2 = _vel_vshf_vvvsl(vrin_r2s2, vrin_r2s2, VE_VSHUFFLE_YUZU, vl0) ;

	__vr vrgout01 = _vel_vshf_vvvsl(vrgout0, vrgout1, VE_VSHUFFLE_YUZU, vl1) ;
	vrsum01_r0s0 = _vel_pvfmad_vvvvvl(vrsum01_r0s0, vrinP_r0s0, vrgout01, vrsum01_r0s0, vl1) ;
	vrsum01_r0s1 = _vel_pvfmad_vvvvvl(vrsum01_r0s1, vrinP_r0s1, vrgout01, vrsum01_r0s1, vl1) ;
	vrsum01_r0s2 = _vel_pvfmad_vvvvvl(vrsum01_r0s2, vrinP_r0s2, vrgout01, vrsum01_r0s2, vl1) ;
	vrsum01_r1s0 = _vel_pvfmad_vvvvvl(vrsum01_r1s0, vrinP_r1s0, vrgout01, vrsum01_r1s0, vl1) ;
	vrsum01_r1s1 = _vel_pvfmad_vvvvvl(vrsum01_r1s1, vrinP_r1s1, vrgout01, vrsum01_r1s1, vl1) ;
	vrsum01_r1s2 = _vel_pvfmad_vvvvvl(vrsum01_r1s2, vrinP_r1s2, vrgout01, vrsum01_r1s2, vl1) ;
	vrsum01_r2s0 = _vel_pvfmad_vvvvvl(vrsum01_r2s0, vrinP_r2s0, vrgout01, vrsum01_r2s0, vl1) ;
	vrsum01_r2s1 = _vel_pvfmad_vvvvvl(vrsum01_r2s1, vrinP_r2s1, vrgout01, vrsum01_r2s1, vl1) ;
	vrsum01_r2s2 = _vel_pvfmad_vvvvvl(vrsum01_r2s2, vrinP_r2s2, vrgout01, vrsum01_r2s2, vl1) ;

	__vr vrgout23 = _vel_vshf_vvvsl(vrgout2, vrgout3, VE_VSHUFFLE_YUZU, vl1) ;
	vrsum23_r0s0 = _vel_pvfmad_vvvvvl(vrsum23_r0s0, vrinP_r0s0, vrgout23, vrsum23_r0s0, vl1) ;
	vrsum23_r0s1 = _vel_pvfmad_vvvvvl(vrsum23_r0s1, vrinP_r0s1, vrgout23, vrsum23_r0s1, vl1) ;
	vrsum23_r0s2 = _vel_pvfmad_vvvvvl(vrsum23_r0s2, vrinP_r0s2, vrgout23, vrsum23_r0s2, vl1) ;
	vrsum23_r1s0 = _vel_pvfmad_vvvvvl(vrsum23_r1s0, vrinP_r1s0, vrgout23, vrsum23_r1s0, vl1) ;
	vrsum23_r1s1 = _vel_pvfmad_vvvvvl(vrsum23_r1s1, vrinP_r1s1, vrgout23, vrsum23_r1s1, vl1) ;
	vrsum23_r1s2 = _vel_pvfmad_vvvvvl(vrsum23_r1s2, vrinP_r1s2, vrgout23, vrsum23_r1s2, vl1) ;
	vrsum23_r2s0 = _vel_pvfmad_vvvvvl(vrsum23_r2s0, vrinP_r2s0, vrgout23, vrsum23_r2s0, vl1) ;
	vrsum23_r2s1 = _vel_pvfmad_vvvvvl(vrsum23_r2s1, vrinP_r2s1, vrgout23, vrsum23_r2s1, vl1) ;
	vrsum23_r2s2 = _vel_pvfmad_vvvvvl(vrsum23_r2s2, vrinP_r2s2, vrgout23, vrsum23_r2s2, vl1) ;

	__vr vrgout45 = _vel_vshf_vvvsl(vrgout4, vrgout5, VE_VSHUFFLE_YUZU, vl1) ;
	vrsum45_r0s0 = _vel_pvfmad_vvvvvl(vrsum45_r0s0, vrinP_r0s0, vrgout45, vrsum45_r0s0, vl1) ;
	vrsum45_r0s1 = _vel_pvfmad_vvvvvl(vrsum45_r0s1, vrinP_r0s1, vrgout45, vrsum45_r0s1, vl1) ;
	vrsum45_r0s2 = _vel_pvfmad_vvvvvl(vrsum45_r0s2, vrinP_r0s2, vrgout45, vrsum45_r0s2, vl1) ;
	vrsum45_r1s0 = _vel_pvfmad_vvvvvl(vrsum45_r1s0, vrinP_r1s0, vrgout45, vrsum45_r1s0, vl1) ;
	vrsum45_r1s1 = _vel_pvfmad_vvvvvl(vrsum45_r1s1, vrinP_r1s1, vrgout45, vrsum45_r1s1, vl1) ;
	vrsum45_r1s2 = _vel_pvfmad_vvvvvl(vrsum45_r1s2, vrinP_r1s2, vrgout45, vrsum45_r1s2, vl1) ;
	vrsum45_r2s0 = _vel_pvfmad_vvvvvl(vrsum45_r2s0, vrinP_r2s0, vrgout45, vrsum45_r2s0, vl1) ;
	vrsum45_r2s1 = _vel_pvfmad_vvvvvl(vrsum45_r2s1, vrinP_r2s1, vrgout45, vrsum45_r2s1, vl1) ;
	vrsum45_r2s2 = _vel_pvfmad_vvvvvl(vrsum45_r2s2, vrinP_r2s2, vrgout45, vrsum45_r2s2, vl1) ;

	__vr vrgout67 = _vel_vshf_vvvsl(vrgout6, vrgout7, VE_VSHUFFLE_YUZU, vl1) ;
	vrsum67_r0s0 = _vel_pvfmad_vvvvvl(vrsum67_r0s0, vrinP_r0s0, vrgout67, vrsum67_r0s0, vl1) ;
	vrsum67_r0s1 = _vel_pvfmad_vvvvvl(vrsum67_r0s1, vrinP_r0s1, vrgout67, vrsum67_r0s1, vl1) ;
	vrsum67_r0s2 = _vel_pvfmad_vvvvvl(vrsum67_r0s2, vrinP_r0s2, vrgout67, vrsum67_r0s2, vl1) ;
	vrsum67_r1s0 = _vel_pvfmad_vvvvvl(vrsum67_r1s0, vrinP_r1s0, vrgout67, vrsum67_r1s0, vl1) ;
	vrsum67_r1s1 = _vel_pvfmad_vvvvvl(vrsum67_r1s1, vrinP_r1s1, vrgout67, vrsum67_r1s1, vl1) ;
	vrsum67_r1s2 = _vel_pvfmad_vvvvvl(vrsum67_r1s2, vrinP_r1s2, vrgout67, vrsum67_r1s2, vl1) ;
	vrsum67_r2s0 = _vel_pvfmad_vvvvvl(vrsum67_r2s0, vrinP_r2s0, vrgout67, vrsum67_r2s0, vl1) ;
	vrsum67_r2s1 = _vel_pvfmad_vvvvvl(vrsum67_r2s1, vrinP_r2s1, vrgout67, vrsum67_r2s1, vl1) ;
	vrsum67_r2s2 = _vel_pvfmad_vvvvvl(vrsum67_r2s2, vrinP_r2s2, vrgout67, vrsum67_r2s2, vl1) ;
      } // batch
    } // gOutPixels

    VSUM_STORE_3X3_UPPER(vrsum01, kernelIndex0) ;
    VSUM_STORE_3X3_LOWER(vrsum01, kernelIndex1) ;
    VSUM_STORE_3X3_UPPER(vrsum23, kernelIndex2) ;
    VSUM_STORE_3X3_LOWER(vrsum23, kernelIndex3) ;
    VSUM_STORE_3X3_UPPER(vrsum45, kernelIndex4) ;
    VSUM_STORE_3X3_LOWER(vrsum45, kernelIndex5) ;
    VSUM_STORE_3X3_UPPER(vrsum67, kernelIndex6) ;
    VSUM_STORE_3X3_LOWER(vrsum67, kernelIndex7) ;

//#undef VSUM_STORE_3X3_UPPER
//#undef VSUM_STORE_3X3_LOWER

  } // inChannel
}

vednnError_t
vednnConvolutionBackwardFilter_direct_dil1_str1_pad0_ker3_owU128(
    const vednnTensorParam_t * restrict 	pParamIn,
    const void * restrict 			pDataIn,
    const vednnTensorParam_t * restrict 	pParamGradOut,
    const void * restrict 			pDataGradOut,
    const vednnConvolutionParam_t * restrict 	pParamConv,
    const vednnFilterParam_t * restrict 	pParamGradKernel,
    void * restrict 				pDataGradKernel
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

  const float * restrict pIn      = pDataIn;
  const float * restrict pGOut    = pDataGradOut;
  float * restrict const pGKernel = pDataGradKernel;

  const int gOutPixels= gOutHeight*gOutWidth ;

#ifndef VEDNN_USE_OPENMP
  const int64_t beginOChannel = 0 ;
  const int64_t nOChannel     = gOutChannelGroup ;
#endif

  const int64_t nY = VLEN / inWidth ;

  __vr vrseq = _vel_vseq_vl(VLEN) ;			// xy
  __vr vry_s0  = _vel_vdivsl_vvsl(vrseq, inWidth, VLEN) ;
  __vr vrx_s0  = _vel_vsubsl_vvvl(vrseq, _vel_vmulul_vsvl(inWidth,vry_s0, VLEN), VLEN) ;
  __vm256 vm_s0 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutWidth, vrx_s0, VLEN), VLEN) ; // condition(x<gOutWidth)

  __vr vrseq1  = _vel_vaddsl_vsvl(inWidth-1, vrseq, VLEN) ;
  __vr vry_s1  = _vel_vdivsl_vvsl(vrseq1, inWidth, VLEN) ;
  __vr vrx_s1  = _vel_vsubsl_vvvl(vrseq1, _vel_vmulul_vsvl(inWidth,vry_s1, VLEN), VLEN) ;
  __vm256 vm_s1 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutWidth, vrx_s1, VLEN), VLEN) ; // condition(x<gOutWidth)

  __vr vrseq2  = _vel_vaddsl_vsvl(inWidth-2, vrseq, VLEN) ;
  __vr vry_s2  = _vel_vdivsl_vvsl(vrseq2, inWidth, VLEN) ;
  __vr vrx_s2  = _vel_vsubsl_vvvl(vrseq2, _vel_vmulul_vsvl(inWidth,vry_s2, VLEN), VLEN) ;
  __vm256 vm_s2 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutWidth, vrx_s2, VLEN), VLEN) ; // condition(x<gOutWidth)

  for (int64_t g = 0; g < group; g++) {
    int64_t inGroupOffset   = g * inChannelGroup  * inHeight  * inWidth;
    int64_t outGroupOffset  = (g * gOutChannelGroup + beginOChannel) * gOutHeight * gOutWidth;
    int64_t kernGroupOffset = (g * gOutChannelGroup + beginOChannel) * inChannelGroup * gKernHeight * gKernWidth;

    int64_t k=0;
    if ( (nOChannel & 0x01) == 1 ) {
      k1(pIn, pGOut, pGKernel,
	 inChannel, inWidth, inHeight,
	 gOutChannel, gOutWidth, gOutHeight,
	 gKernWidth, gKernHeight,
	 inChannelGroup,
	 inGroupOffset, outGroupOffset, kernGroupOffset,
	 batch,
	 k,
	 vm_s0, vm_s1, vm_s2,
	 nY ) ;
      k++ ;
    }
    if ( ((nOChannel >> 1) & 0x01) == 1 ) {
      k2(pIn, pGOut, pGKernel,
	 inChannel, inWidth, inHeight,
	 gOutChannel, gOutWidth, gOutHeight,
	 gKernWidth, gKernHeight,
	 inChannelGroup,
	 inGroupOffset, outGroupOffset, kernGroupOffset,
	 batch,
	 k,
	 vm_s0, vm_s1, vm_s2,
	 nY ) ;
      k+=2;
    }
    if ( ((nOChannel >> 2) & 0x01) == 1 ) {
      k4(pIn, pGOut, pGKernel,
	 inChannel, inWidth, inHeight,
	 gOutChannel, gOutWidth, gOutHeight,
	 gKernWidth, gKernHeight,
	 inChannelGroup,
	 inGroupOffset, outGroupOffset, kernGroupOffset,
	 batch,
	 k,
	 vm_s0, vm_s1, vm_s2,
	 nY ) ;
      k+=4;
    }
    for ( ;k<nOChannel; k+=8) {
      k8(pIn, pGOut, pGKernel,
	 inChannel, inWidth, inHeight,
	 gOutChannel, gOutWidth, gOutHeight,
	 gKernWidth, gKernHeight,
	 inChannelGroup,
	 inGroupOffset, outGroupOffset, kernGroupOffset,
	 batch,
	 k,
	 vm_s0, vm_s1, vm_s2,
	 nY ) ;
    } // outChannel
  } // group

  return VEDNN_SUCCESS;
}
