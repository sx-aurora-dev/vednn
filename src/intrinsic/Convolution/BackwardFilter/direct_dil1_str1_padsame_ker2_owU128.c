#include <stdint.h>
#include <stdlib.h>

#include "vednn.h"

#include "veintrin.h"
#define VLEN	(256)


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
  const int64_t gOutPixels,
  const int64_t batch,
  const int64_t k
)
{
  const int64_t nY = VLEN / gOutWidth ;

  const int64_t y_remain = gOutHeight % nY ;
  const int64_t last_y = y_remain == 0 ? gOutHeight - nY : gOutHeight - y_remain ;

  _ve_lvl(VLEN) ;
  __vr vrseq = _ve_vseq_v() ; // xy
  __vr vry   = _ve_vdivsl_vvs(vrseq, gOutWidth) ;
  __vr vrx   = _ve_vsubsl_vvv(vrseq, _ve_vmulul_vsv(gOutWidth,vry)) ;

  __vm256 vm_r0 = _ve_vfmkl_mcv(VECC_G, vry) ;	// condition(y-1>=0)
  __vm256 vm_s0 = _ve_vfmkl_mcv(VECC_G, vrx) ;	// condition(x-1>=0)

  for (int64_t c=0; c<inChannelGroup; c++) {
    const int64_t kernelIndex0 = kernGroupOffset + ((k     * inChannelGroup + c) * gKernHeight) * gKernWidth;

    _ve_lvl(VLEN) ;
    __vr vrsum_r0s0 = _ve_vbrdu_vs_f32(0.f) ;
    __vr vrsum_r0s1 = _ve_vbrdu_vs_f32(0.f) ;
    __vr vrsum_r1s0 = _ve_vbrdu_vs_f32(0.f) ;
    __vr vrsum_r1s1 = _ve_vbrdu_vs_f32(0.f) ;

    for (int64_t n=0; n<batch; n++) {
      int64_t y=0;
      for (; y<gOutHeight; y+=nY) {
	const int64_t vl = gOutWidth * (gOutHeight - y < nY ? gOutHeight - y : nY ) ;

	_ve_lvl(vl) ;

	__vm256 vm_r0s0 = _ve_andm_mmm(vm_r0,vm_s0) ;
	__vm256 vm_r0s1 = vm_r0 ;
	__vm256 vm_r1s0 = vm_s0 ;

	const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;

	const int64_t gOutIndex0  = outGroupOffset + ((n * gOutChannel + k  ) * gOutHeight + y ) * gOutWidth ;

	/* memory access errors mihgt be caused (vrin) */
	__vr vrin_r0s0    = _ve_vldu_vss(4,&pInChannel[(y-1)*inWidth-1]) ;
	__vr vrin_r0s1    = _ve_vldu_vss(4,&pInChannel[(y-1)*inWidth  ]) ;
	__vr vrin_r1s0    = _ve_vldu_vss(4,&pInChannel[(y  )*inWidth-1]) ;
	__vr vrin_r1s1    = _ve_vldu_vss(4,&pInChannel[(y  )*inWidth  ]) ;
	__vr vrgout0 = _ve_vldu_vss(4, pGOut+gOutIndex0) ;

	vrin_r0s0 = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), vrin_r0s0, vm_r0s0) ;
	vrsum_r0s0 = _ve_vfmads_vvvv(vrsum_r0s0, vrin_r0s0, vrgout0) ;

	vrin_r0s1 = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), vrin_r0s1, vm_r0s1) ;
	vrsum_r0s1 = _ve_vfmads_vvvv(vrsum_r0s1, vrin_r0s1, vrgout0) ;

	vrin_r1s0 = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), vrin_r1s0, vm_r1s0) ;
	vrsum_r1s0 = _ve_vfmads_vvvv(vrsum_r1s0, vrin_r1s0, vrgout0) ;

	// vrin_r1s1 : no need to use mask
	vrsum_r1s1 = _ve_vfmads_vvvv(vrsum_r1s1, vrin_r1s1, vrgout0) ;
      }
    } // batch

    _ve_lvl(VLEN) ;
    vrsum_r0s0 = _ve_vfsums_vv(vrsum_r0s0) ;
    vrsum_r0s1 = _ve_vfsums_vv(vrsum_r0s1) ;
    vrsum_r1s0 = _ve_vfsums_vv(vrsum_r1s0) ;
    vrsum_r1s1 = _ve_vfsums_vv(vrsum_r1s1) ;
    _ve_lvl(1) ;
    _ve_vstu_vss(vrsum_r0s0,4,pGKernel+kernelIndex0+0) ;
    _ve_vstu_vss(vrsum_r0s1,4,pGKernel+kernelIndex0+1) ;
    _ve_vstu_vss(vrsum_r1s0,4,pGKernel+kernelIndex0+2) ;
    _ve_vstu_vss(vrsum_r1s1,4,pGKernel+kernelIndex0+3) ;
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
  const int64_t gOutPixels,
  const int64_t batch,
  const int64_t k
)
{
  const int64_t nY = VLEN / gOutWidth ;

  const int64_t y_remain = gOutHeight % nY ;
  const int64_t last_y = y_remain == 0 ? gOutHeight - nY : gOutHeight - y_remain ;

  _ve_lvl(VLEN) ;
  __vr vrseq = _ve_vseq_v() ; // xy
  __vr vry   = _ve_vdivsl_vvs(vrseq, gOutWidth) ;
  __vr vrx   = _ve_vsubsl_vvv(vrseq, _ve_vmulul_vsv(gOutWidth,vry)) ;

  __vm256 vm_r0 = _ve_vfmkl_mcv(VECC_G, vry) ;	// condition(y-1>=0)
  __vm256 vm_s0 = _ve_vfmkl_mcv(VECC_G, vrx) ;	// condition(x-1>=0)

  for (int64_t c=0; c<inChannelGroup; c++) {
    const int64_t kernelIndex0 = kernGroupOffset + ((k     * inChannelGroup + c) * gKernHeight) * gKernWidth;
    const int64_t kernelIndex1 = kernGroupOffset + (((k+1) * inChannelGroup + c) * gKernHeight) * gKernWidth;

    _ve_lvl(VLEN) ;
    __vr vrsum01_r0s0 = _ve_vbrd_vs_i64(0UL) ;
    __vr vrsum01_r0s1 = _ve_vbrd_vs_i64(0UL) ;
    __vr vrsum01_r1s0 = _ve_vbrd_vs_i64(0UL) ;
    __vr vrsum01_r1s1 = _ve_vbrd_vs_i64(0UL) ;

    for (int64_t n=0; n<batch; n++) {
      int64_t y=0;
      for (; y<gOutHeight; y+=nY) {
	const int64_t vl = gOutWidth * (gOutHeight - y < nY ? gOutHeight - y : nY ) ;

	_ve_lvl(vl) ;

	__vm256 vm_r0s0 = _ve_andm_mmm(vm_r0,vm_s0) ;
	__vm256 vm_r0s1 = vm_r0 ;
	__vm256 vm_r1s0 = vm_s0 ;

	const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;

	const int64_t gOutIndex0  = outGroupOffset + ((n * gOutChannel + k  ) * gOutHeight + y ) * gOutWidth ;
	const int64_t gOutIndex1  = outGroupOffset + ((n * gOutChannel + k+1) * gOutHeight + y ) * gOutWidth ;

	/* memory access errors mihgt be caused (vrin) */
	__vr vrin_r0s0    = _ve_vldu_vss(4,&pInChannel[(y-1)*inWidth-1]) ;
	__vr vrin_r0s1    = _ve_vldu_vss(4,&pInChannel[(y-1)*inWidth  ]) ;
	__vr vrin_r1s0    = _ve_vldu_vss(4,&pInChannel[(y  )*inWidth-1]) ;
	__vr vrin_r1s1    = _ve_vldu_vss(4,&pInChannel[(y  )*inWidth  ]) ;
	__vr vrgout0 = _ve_vldu_vss(4, pGOut+gOutIndex0) ;
	__vr vrgout1 = _ve_vldu_vss(4, pGOut+gOutIndex1) ;

	__vr vrgout01 = _ve_vshf_vvvs(vrgout0, vrgout1, VE_VSHUFFLE_YUZU) ;

	vrin_r0s0 = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), vrin_r0s0, vm_r0s0) ;
	__vr vrinP_r0s0    = _ve_vshf_vvvs(vrin_r0s0, vrin_r0s0, VE_VSHUFFLE_YUZU) ;
	vrsum01_r0s0 = _ve_pvfmad_vvvv(vrsum01_r0s0, vrinP_r0s0, vrgout01) ;

	vrin_r0s1 = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), vrin_r0s1, vm_r0s1) ;
	__vr vrinP_r0s1    = _ve_vshf_vvvs(vrin_r0s1, vrin_r0s1, VE_VSHUFFLE_YUZU) ;
	vrsum01_r0s1 = _ve_pvfmad_vvvv(vrsum01_r0s1, vrinP_r0s1, vrgout01) ;

	vrin_r1s0 = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), vrin_r1s0, vm_r1s0) ;
	__vr vrinP_r1s0    = _ve_vshf_vvvs(vrin_r1s0, vrin_r1s0, VE_VSHUFFLE_YUZU) ;
	vrsum01_r1s0 = _ve_pvfmad_vvvv(vrsum01_r1s0, vrinP_r1s0, vrgout01) ;

	// vrin_r1s1 : no need to use mask
	__vr vrinP_r1s1    = _ve_vshf_vvvs(vrin_r1s1, vrin_r1s1, VE_VSHUFFLE_YUZU) ;
	vrsum01_r1s1 = _ve_pvfmad_vvvv(vrsum01_r1s1, vrinP_r1s1, vrgout01) ;
      }
    } // batch

#define VSUM_STORE_2X2_UPPER(VRSUMTOKEN, KERNELINDEX)		\
{								\
  _ve_lvl(VLEN) ;						\
  __vr vrsumU_r0s0 = _ve_vfsums_vv(VRSUMTOKEN ## _r0s0) ;	\
  __vr vrsumU_r0s1 = _ve_vfsums_vv(VRSUMTOKEN ## _r0s1) ;	\
  __vr vrsumU_r1s0 = _ve_vfsums_vv(VRSUMTOKEN ## _r1s0) ;	\
  __vr vrsumU_r1s1 = _ve_vfsums_vv(VRSUMTOKEN ## _r1s1) ;	\
  _ve_lvl(1) ;							\
  _ve_vstu_vss(vrsumU_r0s0,4,pGKernel+(KERNELINDEX)+0) ;	\
  _ve_vstu_vss(vrsumU_r0s1,4,pGKernel+(KERNELINDEX)+1) ;	\
  _ve_vstu_vss(vrsumU_r1s0,4,pGKernel+(KERNELINDEX)+2) ;	\
  _ve_vstu_vss(vrsumU_r1s1,4,pGKernel+(KERNELINDEX)+3) ;	\
}
#define VSUM_STORE_2X2_LOWER(VRSUMTOKEN, KERNELINDEX)				\
{										\
  _ve_lvl(VLEN) ;								\
  __vr vrsumL_r0s0 = _ve_vfsums_vv(_ve_vsll_vvs(VRSUMTOKEN ## _r0s0,32)) ;	\
  __vr vrsumL_r0s1 = _ve_vfsums_vv(_ve_vsll_vvs(VRSUMTOKEN ## _r0s1,32)) ;	\
  __vr vrsumL_r1s0 = _ve_vfsums_vv(_ve_vsll_vvs(VRSUMTOKEN ## _r1s0,32)) ;	\
  __vr vrsumL_r1s1 = _ve_vfsums_vv(_ve_vsll_vvs(VRSUMTOKEN ## _r1s1,32)) ;	\
  _ve_lvl(1) ;									\
  _ve_vstu_vss(vrsumL_r0s0,4,pGKernel+(KERNELINDEX)+0) ;			\
  _ve_vstu_vss(vrsumL_r0s1,4,pGKernel+(KERNELINDEX)+1) ;			\
  _ve_vstu_vss(vrsumL_r1s0,4,pGKernel+(KERNELINDEX)+2) ;			\
  _ve_vstu_vss(vrsumL_r1s1,4,pGKernel+(KERNELINDEX)+3) ;			\
}

    VSUM_STORE_2X2_UPPER(vrsum01, kernelIndex0) ;
    VSUM_STORE_2X2_LOWER(vrsum01, kernelIndex1) ;
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
  const int64_t gOutPixels,
  const int64_t batch,
  const int64_t k
)
{
  const int64_t nY = VLEN / gOutWidth ;

  const int64_t y_remain = gOutHeight % nY ;
  const int64_t last_y = y_remain == 0 ? gOutHeight - nY : gOutHeight - y_remain ;

  _ve_lvl(VLEN) ;
  __vr vrseq = _ve_vseq_v() ; // xy
  __vr vry   = _ve_vdivsl_vvs(vrseq, gOutWidth) ;
  __vr vrx   = _ve_vsubsl_vvv(vrseq, _ve_vmulul_vsv(gOutWidth,vry)) ;

  __vm256 vm_r0 = _ve_vfmkl_mcv(VECC_G, vry) ;	// condition(y-1>=0)
  __vm256 vm_s0 = _ve_vfmkl_mcv(VECC_G, vrx) ;	// condition(x-1>=0)

  for (int64_t c=0; c<inChannelGroup; c++) {
    const int64_t kernelIndex0 = kernGroupOffset + ((k     * inChannelGroup + c) * gKernHeight) * gKernWidth;
    const int64_t kernelIndex1 = kernGroupOffset + (((k+1) * inChannelGroup + c) * gKernHeight) * gKernWidth;
    const int64_t kernelIndex2 = kernGroupOffset + (((k+2) * inChannelGroup + c) * gKernHeight) * gKernWidth;
    const int64_t kernelIndex3 = kernGroupOffset + (((k+3) * inChannelGroup + c) * gKernHeight) * gKernWidth;

    _ve_lvl(VLEN) ;
    __vr vrsum01_r0s0 = _ve_vbrd_vs_i64(0UL) ;
    __vr vrsum23_r0s0 = _ve_vbrd_vs_i64(0UL) ;

    __vr vrsum01_r0s1 = _ve_vbrd_vs_i64(0UL) ;
    __vr vrsum23_r0s1 = _ve_vbrd_vs_i64(0UL) ;

    __vr vrsum01_r1s0 = _ve_vbrd_vs_i64(0UL) ;
    __vr vrsum23_r1s0 = _ve_vbrd_vs_i64(0UL) ;

    __vr vrsum01_r1s1 = _ve_vbrd_vs_i64(0UL) ;
    __vr vrsum23_r1s1 = _ve_vbrd_vs_i64(0UL) ;

    for (int64_t n=0; n<batch; n++) {
      int64_t y=0;
      for (; y<gOutHeight; y+=nY) {
	const int64_t vl = gOutWidth * (gOutHeight - y < nY ? gOutHeight - y : nY ) ;

	_ve_lvl(vl) ;

	__vm256 vm_r0s0 = _ve_andm_mmm(vm_r0,vm_s0) ;
	__vm256 vm_r0s1 = vm_r0 ;
	__vm256 vm_r1s0 = vm_s0 ;

	const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;

	const int64_t gOutIndex0  = outGroupOffset + ((n * gOutChannel + k  ) * gOutHeight + y ) * gOutWidth ;
	const int64_t gOutIndex1  = outGroupOffset + ((n * gOutChannel + k+1) * gOutHeight + y ) * gOutWidth ;
	const int64_t gOutIndex2  = outGroupOffset + ((n * gOutChannel + k+2) * gOutHeight + y ) * gOutWidth ;
	const int64_t gOutIndex3  = outGroupOffset + ((n * gOutChannel + k+3) * gOutHeight + y ) * gOutWidth ;

	/* memory access errors mihgt be caused (vrin) */
	__vr vrin_r0s0    = _ve_vldu_vss(4,&pInChannel[(y-1)*inWidth-1]) ;
	__vr vrin_r0s1    = _ve_vldu_vss(4,&pInChannel[(y-1)*inWidth  ]) ;
	__vr vrin_r1s0    = _ve_vldu_vss(4,&pInChannel[(y  )*inWidth-1]) ;
	__vr vrin_r1s1    = _ve_vldu_vss(4,&pInChannel[(y  )*inWidth  ]) ;
	__vr vrgout0 = _ve_vldu_vss(4, pGOut+gOutIndex0) ;
	__vr vrgout1 = _ve_vldu_vss(4, pGOut+gOutIndex1) ;
	__vr vrgout2 = _ve_vldu_vss(4, pGOut+gOutIndex2) ;
	__vr vrgout3 = _ve_vldu_vss(4, pGOut+gOutIndex3) ;

	__vr vrgout01 = _ve_vshf_vvvs(vrgout0, vrgout1, VE_VSHUFFLE_YUZU) ;
	__vr vrgout23 = _ve_vshf_vvvs(vrgout2, vrgout3, VE_VSHUFFLE_YUZU) ;

	vrin_r0s0 = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), vrin_r0s0, vm_r0s0) ;
	__vr vrinP_r0s0    = _ve_vshf_vvvs(vrin_r0s0, vrin_r0s0, VE_VSHUFFLE_YUZU) ;
	vrsum01_r0s0 = _ve_pvfmad_vvvv(vrsum01_r0s0, vrinP_r0s0, vrgout01) ;
	vrsum23_r0s0 = _ve_pvfmad_vvvv(vrsum23_r0s0, vrinP_r0s0, vrgout23) ;

	vrin_r0s1 = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), vrin_r0s1, vm_r0s1) ;
	__vr vrinP_r0s1    = _ve_vshf_vvvs(vrin_r0s1, vrin_r0s1, VE_VSHUFFLE_YUZU) ;
	vrsum01_r0s1 = _ve_pvfmad_vvvv(vrsum01_r0s1, vrinP_r0s1, vrgout01) ;
	vrsum23_r0s1 = _ve_pvfmad_vvvv(vrsum23_r0s1, vrinP_r0s1, vrgout23) ;

	vrin_r1s0 = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), vrin_r1s0, vm_r1s0) ;
	__vr vrinP_r1s0    = _ve_vshf_vvvs(vrin_r1s0, vrin_r1s0, VE_VSHUFFLE_YUZU) ;
	vrsum01_r1s0 = _ve_pvfmad_vvvv(vrsum01_r1s0, vrinP_r1s0, vrgout01) ;
	vrsum23_r1s0 = _ve_pvfmad_vvvv(vrsum23_r1s0, vrinP_r1s0, vrgout23) ;

	// vrin_r1s1 : no need to use mask
	__vr vrinP_r1s1    = _ve_vshf_vvvs(vrin_r1s1, vrin_r1s1, VE_VSHUFFLE_YUZU) ;
	vrsum01_r1s1 = _ve_pvfmad_vvvv(vrsum01_r1s1, vrinP_r1s1, vrgout01) ;
	vrsum23_r1s1 = _ve_pvfmad_vvvv(vrsum23_r1s1, vrinP_r1s1, vrgout23) ;
      }
    } // batch

    VSUM_STORE_2X2_UPPER(vrsum01, kernelIndex0) ;
    VSUM_STORE_2X2_LOWER(vrsum01, kernelIndex1) ;
    VSUM_STORE_2X2_UPPER(vrsum23, kernelIndex2) ;
    VSUM_STORE_2X2_LOWER(vrsum23, kernelIndex3) ;

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
  const int64_t gOutPixels,
  const int64_t batch,
  const int64_t k
)
{
  const int64_t nY = VLEN / gOutWidth ;

  const int64_t y_remain = gOutHeight % nY ;
  const int64_t last_y = y_remain == 0 ? gOutHeight - nY : gOutHeight - y_remain ;

  _ve_lvl(VLEN) ;
  __vr vrseq = _ve_vseq_v() ; // xy
  __vr vry   = _ve_vdivsl_vvs(vrseq, gOutWidth) ;
  __vr vrx   = _ve_vsubsl_vvv(vrseq, _ve_vmulul_vsv(gOutWidth,vry)) ;

  __vm256 vm_r0 = _ve_vfmkl_mcv(VECC_G, vry) ;	// condition(y-1>=0)
  __vm256 vm_s0 = _ve_vfmkl_mcv(VECC_G, vrx) ;	// condition(x-1>=0)

  for (int64_t c=0; c<inChannelGroup; c++) {
    const int64_t kernelIndex0 = kernGroupOffset + ((k     * inChannelGroup + c) * gKernHeight) * gKernWidth;
    const int64_t kernelIndex1 = kernGroupOffset + (((k+1) * inChannelGroup + c) * gKernHeight) * gKernWidth;
    const int64_t kernelIndex2 = kernGroupOffset + (((k+2) * inChannelGroup + c) * gKernHeight) * gKernWidth;
    const int64_t kernelIndex3 = kernGroupOffset + (((k+3) * inChannelGroup + c) * gKernHeight) * gKernWidth;
    const int64_t kernelIndex4 = kernGroupOffset + (((k+4) * inChannelGroup + c) * gKernHeight) * gKernWidth;
    const int64_t kernelIndex5 = kernGroupOffset + (((k+5) * inChannelGroup + c) * gKernHeight) * gKernWidth;
    const int64_t kernelIndex6 = kernGroupOffset + (((k+6) * inChannelGroup + c) * gKernHeight) * gKernWidth;
    const int64_t kernelIndex7 = kernGroupOffset + (((k+7) * inChannelGroup + c) * gKernHeight) * gKernWidth;

    _ve_lvl(VLEN) ;
    __vr vrsum01_r0s0 = _ve_vbrd_vs_i64(0UL) ;
    __vr vrsum23_r0s0 = _ve_vbrd_vs_i64(0UL) ;
    __vr vrsum45_r0s0 = _ve_vbrd_vs_i64(0UL) ;
    __vr vrsum67_r0s0 = _ve_vbrd_vs_i64(0UL) ;

    __vr vrsum01_r0s1 = _ve_vbrd_vs_i64(0UL) ;
    __vr vrsum23_r0s1 = _ve_vbrd_vs_i64(0UL) ;
    __vr vrsum45_r0s1 = _ve_vbrd_vs_i64(0UL) ;
    __vr vrsum67_r0s1 = _ve_vbrd_vs_i64(0UL) ;

    __vr vrsum01_r1s0 = _ve_vbrd_vs_i64(0UL) ;
    __vr vrsum23_r1s0 = _ve_vbrd_vs_i64(0UL) ;
    __vr vrsum45_r1s0 = _ve_vbrd_vs_i64(0UL) ;
    __vr vrsum67_r1s0 = _ve_vbrd_vs_i64(0UL) ;

    __vr vrsum01_r1s1 = _ve_vbrd_vs_i64(0UL) ;
    __vr vrsum23_r1s1 = _ve_vbrd_vs_i64(0UL) ;
    __vr vrsum45_r1s1 = _ve_vbrd_vs_i64(0UL) ;
    __vr vrsum67_r1s1 = _ve_vbrd_vs_i64(0UL) ;

    for (int64_t n=0; n<batch; n++) {
      int64_t y=0;
      for (; y<gOutHeight; y+=nY) {
	const int64_t vl = gOutWidth * (gOutHeight - y < nY ? gOutHeight - y : nY ) ;

	_ve_lvl(vl) ;

	__vm256 vm_r0s0 = _ve_andm_mmm(vm_r0,vm_s0) ;
	__vm256 vm_r0s1 = vm_r0 ;
	__vm256 vm_r1s0 = vm_s0 ;

	const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;

	const int64_t gOutIndex0  = outGroupOffset + ((n * gOutChannel + k  ) * gOutHeight + y ) * gOutWidth ;
	const int64_t gOutIndex1  = outGroupOffset + ((n * gOutChannel + k+1) * gOutHeight + y ) * gOutWidth ;
	const int64_t gOutIndex2  = outGroupOffset + ((n * gOutChannel + k+2) * gOutHeight + y ) * gOutWidth ;
	const int64_t gOutIndex3  = outGroupOffset + ((n * gOutChannel + k+3) * gOutHeight + y ) * gOutWidth ;
	const int64_t gOutIndex4  = outGroupOffset + ((n * gOutChannel + k+4) * gOutHeight + y ) * gOutWidth ;
	const int64_t gOutIndex5  = outGroupOffset + ((n * gOutChannel + k+5) * gOutHeight + y ) * gOutWidth ;
	const int64_t gOutIndex6  = outGroupOffset + ((n * gOutChannel + k+6) * gOutHeight + y ) * gOutWidth ;
	const int64_t gOutIndex7  = outGroupOffset + ((n * gOutChannel + k+7) * gOutHeight + y ) * gOutWidth ;

	/* memory access errors mihgt be caused (vrin) */
	__vr vrin_r0s0    = _ve_vldu_vss(4,&pInChannel[(y-1)*inWidth-1]) ;
	__vr vrin_r0s1    = _ve_vldu_vss(4,&pInChannel[(y-1)*inWidth  ]) ;
	__vr vrin_r1s0    = _ve_vldu_vss(4,&pInChannel[(y  )*inWidth-1]) ;
	__vr vrin_r1s1    = _ve_vldu_vss(4,&pInChannel[(y  )*inWidth  ]) ;
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

	vrin_r0s0 = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), vrin_r0s0, vm_r0s0) ;
	__vr vrinP_r0s0    = _ve_vshf_vvvs(vrin_r0s0, vrin_r0s0, VE_VSHUFFLE_YUZU) ;
	vrsum01_r0s0 = _ve_pvfmad_vvvv(vrsum01_r0s0, vrinP_r0s0, vrgout01) ;
	vrsum23_r0s0 = _ve_pvfmad_vvvv(vrsum23_r0s0, vrinP_r0s0, vrgout23) ;
	vrsum45_r0s0 = _ve_pvfmad_vvvv(vrsum45_r0s0, vrinP_r0s0, vrgout45) ;
	vrsum67_r0s0 = _ve_pvfmad_vvvv(vrsum67_r0s0, vrinP_r0s0, vrgout67) ;

	vrin_r0s1 = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), vrin_r0s1, vm_r0s1) ;
	__vr vrinP_r0s1    = _ve_vshf_vvvs(vrin_r0s1, vrin_r0s1, VE_VSHUFFLE_YUZU) ;
	vrsum01_r0s1 = _ve_pvfmad_vvvv(vrsum01_r0s1, vrinP_r0s1, vrgout01) ;
	vrsum23_r0s1 = _ve_pvfmad_vvvv(vrsum23_r0s1, vrinP_r0s1, vrgout23) ;
	vrsum45_r0s1 = _ve_pvfmad_vvvv(vrsum45_r0s1, vrinP_r0s1, vrgout45) ;
	vrsum67_r0s1 = _ve_pvfmad_vvvv(vrsum67_r0s1, vrinP_r0s1, vrgout67) ;

	vrin_r1s0 = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), vrin_r1s0, vm_r1s0) ;
	__vr vrinP_r1s0    = _ve_vshf_vvvs(vrin_r1s0, vrin_r1s0, VE_VSHUFFLE_YUZU) ;
	vrsum01_r1s0 = _ve_pvfmad_vvvv(vrsum01_r1s0, vrinP_r1s0, vrgout01) ;
	vrsum23_r1s0 = _ve_pvfmad_vvvv(vrsum23_r1s0, vrinP_r1s0, vrgout23) ;
	vrsum45_r1s0 = _ve_pvfmad_vvvv(vrsum45_r1s0, vrinP_r1s0, vrgout45) ;
	vrsum67_r1s0 = _ve_pvfmad_vvvv(vrsum67_r1s0, vrinP_r1s0, vrgout67) ;

	// vrin_r1s1 : no need to use mask
	__vr vrinP_r1s1    = _ve_vshf_vvvs(vrin_r1s1, vrin_r1s1, VE_VSHUFFLE_YUZU) ;
	vrsum01_r1s1 = _ve_pvfmad_vvvv(vrsum01_r1s1, vrinP_r1s1, vrgout01) ;
	vrsum23_r1s1 = _ve_pvfmad_vvvv(vrsum23_r1s1, vrinP_r1s1, vrgout23) ;
	vrsum45_r1s1 = _ve_pvfmad_vvvv(vrsum45_r1s1, vrinP_r1s1, vrgout45) ;
	vrsum67_r1s1 = _ve_pvfmad_vvvv(vrsum67_r1s1, vrinP_r1s1, vrgout67) ;

      }
    } // batch

    VSUM_STORE_2X2_UPPER(vrsum01, kernelIndex0) ;
    VSUM_STORE_2X2_LOWER(vrsum01, kernelIndex1) ;
    VSUM_STORE_2X2_UPPER(vrsum23, kernelIndex2) ;
    VSUM_STORE_2X2_LOWER(vrsum23, kernelIndex3) ;
    VSUM_STORE_2X2_UPPER(vrsum45, kernelIndex4) ;
    VSUM_STORE_2X2_LOWER(vrsum45, kernelIndex5) ;
    VSUM_STORE_2X2_UPPER(vrsum67, kernelIndex6) ;
    VSUM_STORE_2X2_LOWER(vrsum67, kernelIndex7) ;

  } // inChannel
}

vednnError_t
vednnConvolutionBackwardFilter_direct_dil1_str1_padsame_ker2_owU128(
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
  const int64_t gKernWidth  = pParamGradKernel->width;		/* must be 2*padWidth  + 1 */
  const int64_t gKernHeight = pParamGradKernel->height;		/* must be 2*padHeight + 1 */

  const int64_t group          = pParamConv->group;
//  const int64_t strideWidth    = pParamConv->strideWidth;	/* must be 1 */
//  const int64_t strideHeight   = pParamConv->strideHeight;	/* must be 1 */
  const int64_t padWidth       = pParamConv->padWidth;
  const int64_t padHeight      = pParamConv->padHeight;
//  const int64_t dilationWidth  = pParamConv->dilationWidth;	/* must be 1 */
//  const int64_t dilationHeight = pParamConv->dilationHeight;	/* must be 1 */

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

  {
    for (int64_t g = 0; g < group; g++) {
      int64_t inGroupOffset   = g * inChannelGroup  * inHeight  * inWidth;
      int64_t outGroupOffset  = (g * gOutChannelGroup + beginOChannel) * gOutHeight * gOutWidth;
      int64_t kernGroupOffset = (g * gOutChannelGroup + beginOChannel) * inChannelGroup * gKernHeight * gKernWidth;

      int64_t k = 0 ;

      if ( (nOChannel & 0x01) == 1 ) {
	k1(pIn, pGOut, pGKernel,
	   inChannel, inWidth, inHeight,
	   gOutChannel, gOutWidth, gOutHeight,
	   gKernWidth, gKernHeight,
	   inChannelGroup,
	   inGroupOffset, outGroupOffset, kernGroupOffset,
	   gOutPixels,
	   batch,
	   k) ;
	k+=1;
      }
      if ( ((nOChannel >> 1) & 0x01) == 1 ) {
	k2(pIn, pGOut, pGKernel,
	   inChannel, inWidth, inHeight,
	   gOutChannel, gOutWidth, gOutHeight,
	   gKernWidth, gKernHeight,
	   inChannelGroup,
	   inGroupOffset, outGroupOffset, kernGroupOffset,
	   gOutPixels,
	   batch,
	   k) ;
	k+=2;
      }
      if ( ((nOChannel >> 2) & 0x01) == 1 ) {
	k4(pIn, pGOut, pGKernel,
	   inChannel, inWidth, inHeight,
	   gOutChannel, gOutWidth, gOutHeight,
	   gKernWidth, gKernHeight,
	   inChannelGroup,
	   inGroupOffset, outGroupOffset, kernGroupOffset,
	   gOutPixels,
	   batch,
	   k) ;
	k+=4 ;
      }
      for ( ;k<nOChannel; k+=8) {
	k8(pIn, pGOut, pGKernel,
	   inChannel, inWidth, inHeight,
	   gOutChannel, gOutWidth, gOutHeight,
	   gKernWidth, gKernHeight,
	   inChannelGroup,
	   inGroupOffset, outGroupOffset, kernGroupOffset,
	   gOutPixels,
	   batch,
	   k) ;
      } // outChannel
    } // group
  }


  return VEDNN_SUCCESS;
}
