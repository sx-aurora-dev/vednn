#include <stdint.h>
#include <stdlib.h>

#include "vednn.h"

#include "velintrin.h"
#define VLEN	(256)

static inline void f1(
  const float * pIn, const int64_t inWidth, const int64_t inHeight,
  const float * pGOut, const int64_t gOutWidth, const int64_t gOutHeight,
  float * const pGKernel, const int64_t gKernWidth , const int64_t gKernHeight,
  const int64_t strideHeight, const int64_t strideWidth,
  const int64_t padHeight, const int64_t padWidth,
  const int64_t dilationHeight, const int64_t dilationWidth,
  const int64_t inChannelGroup, const int64_t inChannel, const int64_t gOutChannel,
  const int64_t inGroupOffset, const int64_t outGroupOffset, const int64_t kernGroupOffset,
  const int64_t batch, const int64_t k,
  const int64_t nY,
  const __vr vri,
  const __vr vrj
)
{
  __vr vrw_s0 = _vel_vaddsl_vsvl(0*dilationWidth-padWidth,  vrj, nY*gOutWidth) ;
  __vr vrw_s1 = _vel_vaddsl_vsvl(1*dilationWidth-padWidth,  vrj, nY*gOutWidth) ;
  __vr vrw_s2 = _vel_vaddsl_vsvl(2*dilationWidth-padWidth,  vrj, nY*gOutWidth) ;

  __vm256 vmw0_s0 =  _vel_vfmklge_mvl(vrw_s0, nY*gOutWidth) ;						// condition(0 <= w)
  __vm256 vmw1_s0 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(inWidth,vrw_s0, nY*gOutWidth), nY*gOutWidth) ;	// condition(w < inWidth)
  __vm256 vmw_s0  = _vel_andm_mmm(vmw0_s0, vmw1_s0) ;

  __vm256 vmw0_s1 =  _vel_vfmklge_mvl(vrw_s1, nY*gOutWidth) ;						// condition(0 <= w)
  __vm256 vmw1_s1 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(inWidth,vrw_s1, nY*gOutWidth), nY*gOutWidth) ;	// condition(w < inWidth)
  __vm256 vmw_s1  = _vel_andm_mmm(vmw0_s1, vmw1_s1) ;

  __vm256 vmw0_s2 =  _vel_vfmklge_mvl(vrw_s2, nY*gOutWidth) ;						// condition(0 <= w)
  __vm256 vmw1_s2 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(inWidth,vrw_s2, nY*gOutWidth), nY*gOutWidth) ;	// condition(w < inWidth)
  __vm256 vmw_s2  = _vel_andm_mmm(vmw0_s2, vmw1_s2) ;

  for (int64_t c=0; c<inChannelGroup; c++) {
    const int64_t kernelIndex = kernGroupOffset + ((k     * inChannelGroup + c) * gKernHeight ) * gKernWidth ;

    __vr vrsum0_r0s0 = _vel_vbrds_vsl(0.f, VLEN) ;
    __vr vrsum0_r0s1 = _vel_vbrds_vsl(0.f, VLEN) ;
    __vr vrsum0_r0s2 = _vel_vbrds_vsl(0.f, VLEN) ;
    __vr vrsum0_r1s0 = _vel_vbrds_vsl(0.f, VLEN) ;
    __vr vrsum0_r1s1 = _vel_vbrds_vsl(0.f, VLEN) ;
    __vr vrsum0_r1s2 = _vel_vbrds_vsl(0.f, VLEN) ;
    __vr vrsum0_r2s0 = _vel_vbrds_vsl(0.f, VLEN) ;
    __vr vrsum0_r2s1 = _vel_vbrds_vsl(0.f, VLEN) ;
    __vr vrsum0_r2s2 = _vel_vbrds_vsl(0.f, VLEN) ;

    for (int64_t n=0; n<batch; n++) {
      for (int64_t y=0; y<gOutHeight; y+=nY) {

	const int64_t vl = gOutWidth * (gOutHeight - y < nY ? gOutHeight - y : nY) ;
	const int64_t gop = y * gOutWidth ;

	__vr vrh_r0 = _vel_vaddsl_vsvl(0*dilationHeight-padHeight+y*strideHeight, vri, vl) ;
	__vr vrh_r1 = _vel_vaddsl_vsvl(1*dilationHeight-padHeight+y*strideHeight, vri, vl) ;
	__vr vrh_r2 = _vel_vaddsl_vsvl(2*dilationHeight-padHeight+y*strideHeight, vri, vl) ;


	__vm256 vmh0_r0 =  _vel_vfmklge_mvl(vrh_r0, vl) ;					// condition(0 <= h)
	__vm256 vmh1_r0 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(inHeight,vrh_r0, vl), vl) ;	// condition(h < inHeight)
	__vm256 vmh_r0  = _vel_andm_mmm(vmh0_r0, vmh1_r0) ;
	__vm256 vmhw_r0s0 = _vel_andm_mmm(vmh_r0, vmw_s0) ;
	__vm256 vmhw_r0s1 = _vel_andm_mmm(vmh_r0, vmw_s1) ;
	__vm256 vmhw_r0s2 = _vel_andm_mmm(vmh_r0, vmw_s2) ;

	__vm256 vmh0_r1 =  _vel_vfmklge_mvl(vrh_r1, vl) ;					// condition(0 <= h)
	__vm256 vmh1_r1 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(inHeight,vrh_r1, vl), vl) ;	// condition(h < inHeight)
	__vm256 vmh_r1  = _vel_andm_mmm(vmh0_r1, vmh1_r1) ;
	__vm256 vmhw_r1s0 = _vel_andm_mmm(vmh_r1, vmw_s0) ;
	__vm256 vmhw_r1s1 = _vel_andm_mmm(vmh_r1, vmw_s1) ;
	__vm256 vmhw_r1s2 = _vel_andm_mmm(vmh_r1, vmw_s2) ;

	__vm256 vmh0_r2 =  _vel_vfmklge_mvl(vrh_r2, vl) ;					// condition(0 <= h)
	__vm256 vmh1_r2 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(inHeight,vrh_r2, vl), vl) ;	// condition(h < inHeight)
	__vm256 vmh_r2  = _vel_andm_mmm(vmh0_r2, vmh1_r2) ;
	__vm256 vmhw_r2s0 = _vel_andm_mmm(vmh_r2, vmw_s0) ;
	__vm256 vmhw_r2s1 = _vel_andm_mmm(vmh_r2, vmw_s1) ;
	__vm256 vmhw_r2s2 = _vel_andm_mmm(vmh_r2, vmw_s2) ;


	const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;


	__vr vrpin_r0s0 = _vel_vsfa_vvssl(_vel_vaddul_vvvl(vrw_s0, _vel_vmulul_vsvl(inWidth,vrh_r0, vl), vl), 2, (uint64_t)pInChannel, vl) ;
	__vr vrin_r0s0 = _vel_vgtu_vvssml(vrpin_r0s0, 0, 0, vmhw_r0s0, vl) ;
	__vr vrpin_r0s1 = _vel_vsfa_vvssl(_vel_vaddul_vvvl(vrw_s1, _vel_vmulul_vsvl(inWidth,vrh_r0, vl), vl), 2, (uint64_t)pInChannel, vl) ;
	__vr vrin_r0s1 = _vel_vgtu_vvssml(vrpin_r0s1, 0, 0, vmhw_r0s1, vl) ;
	__vr vrpin_r0s2 = _vel_vsfa_vvssl(_vel_vaddul_vvvl(vrw_s2, _vel_vmulul_vsvl(inWidth,vrh_r0, vl), vl), 2, (uint64_t)pInChannel, vl) ;
	__vr vrin_r0s2 = _vel_vgtu_vvssml(vrpin_r0s2, 0, 0, vmhw_r0s2, vl) ;

	__vr vrpin_r1s0 = _vel_vsfa_vvssl(_vel_vaddul_vvvl(vrw_s0, _vel_vmulul_vsvl(inWidth,vrh_r1, vl), vl), 2, (uint64_t)pInChannel, vl) ;
	__vr vrin_r1s0 = _vel_vgtu_vvssml(vrpin_r1s0, 0, 0, vmhw_r1s0, vl) ;
	__vr vrpin_r1s1 = _vel_vsfa_vvssl(_vel_vaddul_vvvl(vrw_s1, _vel_vmulul_vsvl(inWidth,vrh_r1, vl), vl), 2, (uint64_t)pInChannel, vl) ;
	__vr vrin_r1s1 = _vel_vgtu_vvssml(vrpin_r1s1, 0, 0, vmhw_r1s1, vl) ;
	__vr vrpin_r1s2 = _vel_vsfa_vvssl(_vel_vaddul_vvvl(vrw_s2, _vel_vmulul_vsvl(inWidth,vrh_r1, vl), vl), 2, (uint64_t)pInChannel, vl) ;
	__vr vrin_r1s2 = _vel_vgtu_vvssml(vrpin_r1s2, 0, 0, vmhw_r1s2, vl) ;

	__vr vrpin_r2s0 = _vel_vsfa_vvssl(_vel_vaddul_vvvl(vrw_s0, _vel_vmulul_vsvl(inWidth,vrh_r2, vl), vl), 2, (uint64_t)pInChannel, vl) ;
	__vr vrin_r2s0 = _vel_vgtu_vvssml(vrpin_r2s0, 0, 0, vmhw_r2s0, vl) ;
	__vr vrpin_r2s1 = _vel_vsfa_vvssl(_vel_vaddul_vvvl(vrw_s1, _vel_vmulul_vsvl(inWidth,vrh_r2, vl), vl), 2, (uint64_t)pInChannel, vl) ;
	__vr vrin_r2s1 = _vel_vgtu_vvssml(vrpin_r2s1, 0, 0, vmhw_r2s1, vl) ;
	__vr vrpin_r2s2 = _vel_vsfa_vvssl(_vel_vaddul_vvvl(vrw_s2, _vel_vmulul_vsvl(inWidth,vrh_r2, vl), vl), 2, (uint64_t)pInChannel, vl) ;
	__vr vrin_r2s2 = _vel_vgtu_vvssml(vrpin_r2s2, 0, 0, vmhw_r2s2, vl) ;

	__vr vrgout0 = _vel_vldu_vssl(4, pGOut+outGroupOffset + ((n * gOutChannel + k  ) * gOutHeight ) * gOutWidth + gop, vl) ;
	__vr vrgout1 = _vel_vldu_vssl(4, pGOut+outGroupOffset + ((n * gOutChannel + k+1) * gOutHeight ) * gOutWidth + gop, vl) ;
	__vr vrgout2 = _vel_vldu_vssl(4, pGOut+outGroupOffset + ((n * gOutChannel + k+2) * gOutHeight ) * gOutWidth + gop, vl) ;
	__vr vrgout3 = _vel_vldu_vssl(4, pGOut+outGroupOffset + ((n * gOutChannel + k+3) * gOutHeight ) * gOutWidth + gop, vl) ;
	__vr vrgout4 = _vel_vldu_vssl(4, pGOut+outGroupOffset + ((n * gOutChannel + k+4) * gOutHeight ) * gOutWidth + gop, vl) ;
	__vr vrgout5 = _vel_vldu_vssl(4, pGOut+outGroupOffset + ((n * gOutChannel + k+5) * gOutHeight ) * gOutWidth + gop, vl) ;
	__vr vrgout6 = _vel_vldu_vssl(4, pGOut+outGroupOffset + ((n * gOutChannel + k+6) * gOutHeight ) * gOutWidth + gop, vl) ;
	__vr vrgout7 = _vel_vldu_vssl(4, pGOut+outGroupOffset + ((n * gOutChannel + k+7) * gOutHeight ) * gOutWidth + gop, vl) ;

	__vr vrgout01 = _vel_vshf_vvvsl(vrgout0, vrgout1, VE_VSHUFFLE_YUZU, vl) ;
	__vr vrgout23 = _vel_vshf_vvvsl(vrgout2, vrgout3, VE_VSHUFFLE_YUZU, vl) ;
	__vr vrgout45 = _vel_vshf_vvvsl(vrgout4, vrgout5, VE_VSHUFFLE_YUZU, vl) ;
	__vr vrgout67 = _vel_vshf_vvvsl(vrgout6, vrgout7, VE_VSHUFFLE_YUZU, vl) ;

	vrin_r0s0 = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrin_r0s0, vmhw_r0s0, vl) ;
	vrsum0_r0s0 = _vel_vfmads_vvvvvl(vrsum0_r0s0, vrin_r0s0, vrgout01, vrsum0_r0s0, vl) ;

	vrin_r0s1 = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrin_r0s1, vmhw_r0s1, vl) ;
	vrsum0_r0s1 = _vel_vfmads_vvvvvl(vrsum0_r0s1, vrin_r0s1, vrgout01, vrsum0_r0s1, vl) ;

	vrin_r0s2 = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrin_r0s2, vmhw_r0s2, vl) ;
	vrsum0_r0s2 = _vel_vfmads_vvvvvl(vrsum0_r0s2, vrin_r0s2, vrgout01, vrsum0_r0s2, vl) ;

	vrin_r1s0 = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrin_r1s0, vmhw_r1s0, vl) ;
	vrsum0_r1s0 = _vel_vfmads_vvvvvl(vrsum0_r1s0, vrin_r1s0, vrgout01, vrsum0_r1s0, vl) ;

	vrin_r1s1 = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrin_r1s1, vmhw_r1s1, vl) ;
	vrsum0_r1s1 = _vel_vfmads_vvvvvl(vrsum0_r1s1, vrin_r1s1, vrgout01, vrsum0_r1s1, vl) ;

	vrin_r1s2 = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrin_r1s2, vmhw_r1s2, vl) ;
	vrsum0_r1s2 = _vel_vfmads_vvvvvl(vrsum0_r1s2, vrin_r1s2, vrgout01, vrsum0_r1s2, vl) ;

	vrin_r2s0 = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrin_r2s0, vmhw_r2s0, vl) ;
	vrsum0_r2s0 = _vel_vfmads_vvvvvl(vrsum0_r2s0, vrin_r2s0, vrgout01, vrsum0_r2s0, vl) ;

	vrin_r2s1 = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrin_r2s1, vmhw_r2s1, vl) ;
	vrsum0_r2s1 = _vel_vfmads_vvvvvl(vrsum0_r2s1, vrin_r2s1, vrgout01, vrsum0_r2s1, vl) ;

	vrin_r2s2 = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrin_r2s2, vmhw_r2s2, vl) ;
	vrsum0_r2s2 = _vel_vfmads_vvvvvl(vrsum0_r2s2, vrin_r2s2, vrgout01, vrsum0_r2s2, vl) ;

      } // gOutPixels
    } // batch

#define SUM_AND_STORE(R, S, RSTOKEN)							\
    {											\
      __vr vrsum0 = _vel_vfsums_vvl(vrsum0_##RSTOKEN, VLEN) ;				\
      _vel_vstu_vssl(vrsum0, 4, pGKernel+kernelIndex + 0 * inChannelGroup * gKernHeight * gKernWidth + (R) * gKernWidth + (S), 1) ;	\
    }
    SUM_AND_STORE(0,0, r0s0) ;
    SUM_AND_STORE(0,1, r0s1) ;
    SUM_AND_STORE(0,2, r0s2) ;
    SUM_AND_STORE(1,0, r1s0) ;
    SUM_AND_STORE(1,1, r1s1) ;
    SUM_AND_STORE(1,2, r1s2) ;
    SUM_AND_STORE(2,0, r2s0) ;
    SUM_AND_STORE(2,1, r2s1) ;
    SUM_AND_STORE(2,2, r2s2) ;
#undef SUM_AND_STORE

  } // inChannel
}

static inline void f2(
  const float * pIn, const int64_t inWidth, const int64_t inHeight,
  const float * pGOut, const int64_t gOutWidth, const int64_t gOutHeight,
  float * const pGKernel, const int64_t gKernWidth , const int64_t gKernHeight,
  const int64_t strideHeight, const int64_t strideWidth,
  const int64_t padHeight, const int64_t padWidth,
  const int64_t dilationHeight, const int64_t dilationWidth,
  const int64_t inChannelGroup, const int64_t inChannel, const int64_t gOutChannel,
  const int64_t inGroupOffset, const int64_t outGroupOffset, const int64_t kernGroupOffset,
  const int64_t batch, const int64_t k,
  const int64_t nY,
  const __vr vri,
  const __vr vrj
)
{
  __vr vrw_s0 = _vel_vaddsl_vsvl(0*dilationWidth-padWidth,  vrj, nY*gOutWidth) ;
  __vr vrw_s1 = _vel_vaddsl_vsvl(1*dilationWidth-padWidth,  vrj, nY*gOutWidth) ;
  __vr vrw_s2 = _vel_vaddsl_vsvl(2*dilationWidth-padWidth,  vrj, nY*gOutWidth) ;

  __vm256 vmw0_s0 =  _vel_vfmklge_mvl(vrw_s0, nY*gOutWidth) ;						// condition(0 <= w)
  __vm256 vmw1_s0 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(inWidth,vrw_s0, nY*gOutWidth), nY*gOutWidth) ;	// condition(w < inWidth)
  __vm256 vmw_s0  = _vel_andm_mmm(vmw0_s0, vmw1_s0) ;

  __vm256 vmw0_s1 =  _vel_vfmklge_mvl(vrw_s1, nY*gOutWidth) ;						// condition(0 <= w)
  __vm256 vmw1_s1 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(inWidth,vrw_s1, nY*gOutWidth), nY*gOutWidth) ;	// condition(w < inWidth)
  __vm256 vmw_s1  = _vel_andm_mmm(vmw0_s1, vmw1_s1) ;

  __vm256 vmw0_s2 =  _vel_vfmklge_mvl(vrw_s2, nY*gOutWidth) ;						// condition(0 <= w)
  __vm256 vmw1_s2 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(inWidth,vrw_s2, nY*gOutWidth), nY*gOutWidth) ;	// condition(w < inWidth)
  __vm256 vmw_s2  = _vel_andm_mmm(vmw0_s2, vmw1_s2) ;

  for (int64_t c=0; c<inChannelGroup; c++) {
    const int64_t kernelIndex = kernGroupOffset + ((k     * inChannelGroup + c) * gKernHeight ) * gKernWidth ;

    __vr vrsum01_r0s0 = _vel_pvbrd_vsl(0UL, VLEN) ;
    __vr vrsum01_r0s1 = _vel_pvbrd_vsl(0UL, VLEN) ;
    __vr vrsum01_r0s2 = _vel_pvbrd_vsl(0UL, VLEN) ;
    __vr vrsum01_r1s0 = _vel_pvbrd_vsl(0UL, VLEN) ;
    __vr vrsum01_r1s1 = _vel_pvbrd_vsl(0UL, VLEN) ;
    __vr vrsum01_r1s2 = _vel_pvbrd_vsl(0UL, VLEN) ;
    __vr vrsum01_r2s0 = _vel_pvbrd_vsl(0UL, VLEN) ;
    __vr vrsum01_r2s1 = _vel_pvbrd_vsl(0UL, VLEN) ;
    __vr vrsum01_r2s2 = _vel_pvbrd_vsl(0UL, VLEN) ;

    for (int64_t n=0; n<batch; n++) {
      for (int64_t y=0; y<gOutHeight; y+=nY) {

	const int64_t vl = gOutWidth * (gOutHeight - y < nY ? gOutHeight - y : nY) ;
	const int64_t gop = y * gOutWidth ;

	__vr vrh_r0 = _vel_vaddsl_vsvl(0*dilationHeight-padHeight+y*strideHeight, vri, vl) ;
	__vr vrh_r1 = _vel_vaddsl_vsvl(1*dilationHeight-padHeight+y*strideHeight, vri, vl) ;
	__vr vrh_r2 = _vel_vaddsl_vsvl(2*dilationHeight-padHeight+y*strideHeight, vri, vl) ;


	__vm256 vmh0_r0 =  _vel_vfmklge_mvl(vrh_r0, vl) ;					// condition(0 <= h)
	__vm256 vmh1_r0 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(inHeight,vrh_r0, vl), vl) ;	// condition(h < inHeight)
	__vm256 vmh_r0  = _vel_andm_mmm(vmh0_r0, vmh1_r0) ;
	__vm256 vmhw_r0s0 = _vel_andm_mmm(vmh_r0, vmw_s0) ;
	__vm256 vmhw_r0s1 = _vel_andm_mmm(vmh_r0, vmw_s1) ;
	__vm256 vmhw_r0s2 = _vel_andm_mmm(vmh_r0, vmw_s2) ;

	__vm256 vmh0_r1 =  _vel_vfmklge_mvl(vrh_r1, vl) ;					// condition(0 <= h)
	__vm256 vmh1_r1 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(inHeight,vrh_r1, vl), vl) ;	// condition(h < inHeight)
	__vm256 vmh_r1  = _vel_andm_mmm(vmh0_r1, vmh1_r1) ;
	__vm256 vmhw_r1s0 = _vel_andm_mmm(vmh_r1, vmw_s0) ;
	__vm256 vmhw_r1s1 = _vel_andm_mmm(vmh_r1, vmw_s1) ;
	__vm256 vmhw_r1s2 = _vel_andm_mmm(vmh_r1, vmw_s2) ;

	__vm256 vmh0_r2 =  _vel_vfmklge_mvl(vrh_r2, vl) ;					// condition(0 <= h)
	__vm256 vmh1_r2 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(inHeight,vrh_r2, vl), vl) ;	// condition(h < inHeight)
	__vm256 vmh_r2  = _vel_andm_mmm(vmh0_r2, vmh1_r2) ;
	__vm256 vmhw_r2s0 = _vel_andm_mmm(vmh_r2, vmw_s0) ;
	__vm256 vmhw_r2s1 = _vel_andm_mmm(vmh_r2, vmw_s1) ;
	__vm256 vmhw_r2s2 = _vel_andm_mmm(vmh_r2, vmw_s2) ;


	const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;


	__vr vrpin_r0s0 = _vel_vsfa_vvssl(_vel_vaddul_vvvl(vrw_s0, _vel_vmulul_vsvl(inWidth,vrh_r0, vl), vl), 2, (uint64_t)pInChannel, vl) ;
	__vr vrin_r0s0 = _vel_vgtu_vvssml(vrpin_r0s0, 0, 0, vmhw_r0s0, vl) ;
	__vr vrpin_r0s1 = _vel_vsfa_vvssl(_vel_vaddul_vvvl(vrw_s1, _vel_vmulul_vsvl(inWidth,vrh_r0, vl), vl), 2, (uint64_t)pInChannel, vl) ;
	__vr vrin_r0s1 = _vel_vgtu_vvssml(vrpin_r0s1, 0, 0, vmhw_r0s1, vl) ;
	__vr vrpin_r0s2 = _vel_vsfa_vvssl(_vel_vaddul_vvvl(vrw_s2, _vel_vmulul_vsvl(inWidth,vrh_r0, vl), vl), 2, (uint64_t)pInChannel, vl) ;
	__vr vrin_r0s2 = _vel_vgtu_vvssml(vrpin_r0s2, 0, 0, vmhw_r0s2, vl) ;

	__vr vrpin_r1s0 = _vel_vsfa_vvssl(_vel_vaddul_vvvl(vrw_s0, _vel_vmulul_vsvl(inWidth,vrh_r1, vl), vl), 2, (uint64_t)pInChannel, vl) ;
	__vr vrin_r1s0 = _vel_vgtu_vvssml(vrpin_r1s0, 0, 0, vmhw_r1s0, vl) ;
	__vr vrpin_r1s1 = _vel_vsfa_vvssl(_vel_vaddul_vvvl(vrw_s1, _vel_vmulul_vsvl(inWidth,vrh_r1, vl), vl), 2, (uint64_t)pInChannel, vl) ;
	__vr vrin_r1s1 = _vel_vgtu_vvssml(vrpin_r1s1, 0, 0, vmhw_r1s1, vl) ;
	__vr vrpin_r1s2 = _vel_vsfa_vvssl(_vel_vaddul_vvvl(vrw_s2, _vel_vmulul_vsvl(inWidth,vrh_r1, vl), vl), 2, (uint64_t)pInChannel, vl) ;
	__vr vrin_r1s2 = _vel_vgtu_vvssml(vrpin_r1s2, 0, 0, vmhw_r1s2, vl) ;

	__vr vrpin_r2s0 = _vel_vsfa_vvssl(_vel_vaddul_vvvl(vrw_s0, _vel_vmulul_vsvl(inWidth,vrh_r2, vl), vl), 2, (uint64_t)pInChannel, vl) ;
	__vr vrin_r2s0 = _vel_vgtu_vvssml(vrpin_r2s0, 0, 0, vmhw_r2s0, vl) ;
	__vr vrpin_r2s1 = _vel_vsfa_vvssl(_vel_vaddul_vvvl(vrw_s1, _vel_vmulul_vsvl(inWidth,vrh_r2, vl), vl), 2, (uint64_t)pInChannel, vl) ;
	__vr vrin_r2s1 = _vel_vgtu_vvssml(vrpin_r2s1, 0, 0, vmhw_r2s1, vl) ;
	__vr vrpin_r2s2 = _vel_vsfa_vvssl(_vel_vaddul_vvvl(vrw_s2, _vel_vmulul_vsvl(inWidth,vrh_r2, vl), vl), 2, (uint64_t)pInChannel, vl) ;
	__vr vrin_r2s2 = _vel_vgtu_vvssml(vrpin_r2s2, 0, 0, vmhw_r2s2, vl) ;

	__vr vrgout0 = _vel_vldu_vssl(4, pGOut+outGroupOffset + ((n * gOutChannel + k  ) * gOutHeight ) * gOutWidth + gop, vl) ;
	__vr vrgout1 = _vel_vldu_vssl(4, pGOut+outGroupOffset + ((n * gOutChannel + k+1) * gOutHeight ) * gOutWidth + gop, vl) ;
	__vr vrgout2 = _vel_vldu_vssl(4, pGOut+outGroupOffset + ((n * gOutChannel + k+2) * gOutHeight ) * gOutWidth + gop, vl) ;
	__vr vrgout3 = _vel_vldu_vssl(4, pGOut+outGroupOffset + ((n * gOutChannel + k+3) * gOutHeight ) * gOutWidth + gop, vl) ;
	__vr vrgout4 = _vel_vldu_vssl(4, pGOut+outGroupOffset + ((n * gOutChannel + k+4) * gOutHeight ) * gOutWidth + gop, vl) ;
	__vr vrgout5 = _vel_vldu_vssl(4, pGOut+outGroupOffset + ((n * gOutChannel + k+5) * gOutHeight ) * gOutWidth + gop, vl) ;
	__vr vrgout6 = _vel_vldu_vssl(4, pGOut+outGroupOffset + ((n * gOutChannel + k+6) * gOutHeight ) * gOutWidth + gop, vl) ;
	__vr vrgout7 = _vel_vldu_vssl(4, pGOut+outGroupOffset + ((n * gOutChannel + k+7) * gOutHeight ) * gOutWidth + gop, vl) ;

	__vr vrgout01 = _vel_vshf_vvvsl(vrgout0, vrgout1, VE_VSHUFFLE_YUZU, vl) ;
	__vr vrgout23 = _vel_vshf_vvvsl(vrgout2, vrgout3, VE_VSHUFFLE_YUZU, vl) ;
	__vr vrgout45 = _vel_vshf_vvvsl(vrgout4, vrgout5, VE_VSHUFFLE_YUZU, vl) ;
	__vr vrgout67 = _vel_vshf_vvvsl(vrgout6, vrgout7, VE_VSHUFFLE_YUZU, vl) ;

	vrin_r0s0 = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrin_r0s0, vmhw_r0s0, vl) ;
	__vr vrinP_r0s0 = _vel_vshf_vvvsl(vrin_r0s0, vrin_r0s0, VE_VSHUFFLE_YUZU, vl) ;
	vrsum01_r0s0 = _vel_pvfmad_vvvvvl(vrsum01_r0s0, vrinP_r0s0, vrgout01, vrsum01_r0s0, vl) ;

	vrin_r0s1 = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrin_r0s1, vmhw_r0s1, vl) ;
	__vr vrinP_r0s1 = _vel_vshf_vvvsl(vrin_r0s1, vrin_r0s1, VE_VSHUFFLE_YUZU, vl) ;
	vrsum01_r0s1 = _vel_pvfmad_vvvvvl(vrsum01_r0s1, vrinP_r0s1, vrgout01, vrsum01_r0s1, vl) ;

	vrin_r0s2 = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrin_r0s2, vmhw_r0s2, vl) ;
	__vr vrinP_r0s2 = _vel_vshf_vvvsl(vrin_r0s2, vrin_r0s2, VE_VSHUFFLE_YUZU, vl) ;
	vrsum01_r0s2 = _vel_pvfmad_vvvvvl(vrsum01_r0s2, vrinP_r0s2, vrgout01, vrsum01_r0s2, vl) ;

	vrin_r1s0 = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrin_r1s0, vmhw_r1s0, vl) ;
	__vr vrinP_r1s0 = _vel_vshf_vvvsl(vrin_r1s0, vrin_r1s0, VE_VSHUFFLE_YUZU, vl) ;
	vrsum01_r1s0 = _vel_pvfmad_vvvvvl(vrsum01_r1s0, vrinP_r1s0, vrgout01, vrsum01_r1s0, vl) ;

	vrin_r1s1 = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrin_r1s1, vmhw_r1s1, vl) ;
	__vr vrinP_r1s1 = _vel_vshf_vvvsl(vrin_r1s1, vrin_r1s1, VE_VSHUFFLE_YUZU, vl) ;
	vrsum01_r1s1 = _vel_pvfmad_vvvvvl(vrsum01_r1s1, vrinP_r1s1, vrgout01, vrsum01_r1s1, vl) ;

	vrin_r1s2 = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrin_r1s2, vmhw_r1s2, vl) ;
	__vr vrinP_r1s2 = _vel_vshf_vvvsl(vrin_r1s2, vrin_r1s2, VE_VSHUFFLE_YUZU, vl) ;
	vrsum01_r1s2 = _vel_pvfmad_vvvvvl(vrsum01_r1s2, vrinP_r1s2, vrgout01, vrsum01_r1s2, vl) ;

	vrin_r2s0 = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrin_r2s0, vmhw_r2s0, vl) ;
	__vr vrinP_r2s0 = _vel_vshf_vvvsl(vrin_r2s0, vrin_r2s0, VE_VSHUFFLE_YUZU, vl) ;
	vrsum01_r2s0 = _vel_pvfmad_vvvvvl(vrsum01_r2s0, vrinP_r2s0, vrgout01, vrsum01_r2s0, vl) ;

	vrin_r2s1 = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrin_r2s1, vmhw_r2s1, vl) ;
	__vr vrinP_r2s1 = _vel_vshf_vvvsl(vrin_r2s1, vrin_r2s1, VE_VSHUFFLE_YUZU, vl) ;
	vrsum01_r2s1 = _vel_pvfmad_vvvvvl(vrsum01_r2s1, vrinP_r2s1, vrgout01, vrsum01_r2s1, vl) ;

	vrin_r2s2 = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrin_r2s2, vmhw_r2s2, vl) ;
	__vr vrinP_r2s2 = _vel_vshf_vvvsl(vrin_r2s2, vrin_r2s2, VE_VSHUFFLE_YUZU, vl) ;
	vrsum01_r2s2 = _vel_pvfmad_vvvvvl(vrsum01_r2s2, vrinP_r2s2, vrgout01, vrsum01_r2s2, vl) ;

      } // gOutPixels
    } // batch

#define SUM_AND_STORE(R, S, RSTOKEN)							\
    {											\
      __vr vrsum0 = _vel_vfsums_vvl(vrsum01_##RSTOKEN, VLEN) ;				\
      __vr vrsum1 = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsum01_##RSTOKEN,32, VLEN), VLEN);	\
      _vel_vstu_vssl(vrsum0, 4, pGKernel+kernelIndex + 0 * inChannelGroup * gKernHeight * gKernWidth + (R) * gKernWidth + (S), 1) ;	\
      _vel_vstu_vssl(vrsum1, 4, pGKernel+kernelIndex + 1 * inChannelGroup * gKernHeight * gKernWidth + (R) * gKernWidth + (S), 1) ;	\
    }
    SUM_AND_STORE(0,0, r0s0) ;
    SUM_AND_STORE(0,1, r0s1) ;
    SUM_AND_STORE(0,2, r0s2) ;
    SUM_AND_STORE(1,0, r1s0) ;
    SUM_AND_STORE(1,1, r1s1) ;
    SUM_AND_STORE(1,2, r1s2) ;
    SUM_AND_STORE(2,0, r2s0) ;
    SUM_AND_STORE(2,1, r2s1) ;
    SUM_AND_STORE(2,2, r2s2) ;
#undef SUM_AND_STORE

  } // inChannel
}

static inline void f4(
  const float * pIn, const int64_t inWidth, const int64_t inHeight,
  const float * pGOut, const int64_t gOutWidth, const int64_t gOutHeight,
  float * const pGKernel, const int64_t gKernWidth , const int64_t gKernHeight,
  const int64_t strideHeight, const int64_t strideWidth,
  const int64_t padHeight, const int64_t padWidth,
  const int64_t dilationHeight, const int64_t dilationWidth,
  const int64_t inChannelGroup, const int64_t inChannel, const int64_t gOutChannel,
  const int64_t inGroupOffset, const int64_t outGroupOffset, const int64_t kernGroupOffset,
  const int64_t batch, const int64_t k,
  const int64_t nY,
  const __vr vri,
  const __vr vrj
)
{
  __vr vrw_s0 = _vel_vaddsl_vsvl(0*dilationWidth-padWidth,  vrj, nY*gOutWidth) ;
  __vr vrw_s1 = _vel_vaddsl_vsvl(1*dilationWidth-padWidth,  vrj, nY*gOutWidth) ;
  __vr vrw_s2 = _vel_vaddsl_vsvl(2*dilationWidth-padWidth,  vrj, nY*gOutWidth) ;

  __vm256 vmw0_s0 =  _vel_vfmklge_mvl(vrw_s0, nY*gOutWidth) ;						// condition(0 <= w)
  __vm256 vmw1_s0 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(inWidth,vrw_s0, nY*gOutWidth), nY*gOutWidth) ;	// condition(w < inWidth)
  __vm256 vmw_s0  = _vel_andm_mmm(vmw0_s0, vmw1_s0) ;

  __vm256 vmw0_s1 =  _vel_vfmklge_mvl(vrw_s1, nY*gOutWidth) ;						// condition(0 <= w)
  __vm256 vmw1_s1 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(inWidth,vrw_s1, nY*gOutWidth), nY*gOutWidth) ;	// condition(w < inWidth)
  __vm256 vmw_s1  = _vel_andm_mmm(vmw0_s1, vmw1_s1) ;

  __vm256 vmw0_s2 =  _vel_vfmklge_mvl(vrw_s2, nY*gOutWidth) ;						// condition(0 <= w)
  __vm256 vmw1_s2 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(inWidth,vrw_s2, nY*gOutWidth), nY*gOutWidth) ;	// condition(w < inWidth)
  __vm256 vmw_s2  = _vel_andm_mmm(vmw0_s2, vmw1_s2) ;

  for (int64_t c=0; c<inChannelGroup; c++) {
    const int64_t kernelIndex = kernGroupOffset + ((k     * inChannelGroup + c) * gKernHeight ) * gKernWidth ;

    __vr vrsum01_r0s0 = _vel_pvbrd_vsl(0UL, VLEN) ;
    __vr vrsum23_r0s0 = _vel_pvbrd_vsl(0UL, VLEN) ;

    __vr vrsum01_r0s1 = _vel_pvbrd_vsl(0UL, VLEN) ;
    __vr vrsum23_r0s1 = _vel_pvbrd_vsl(0UL, VLEN) ;

    __vr vrsum01_r0s2 = _vel_pvbrd_vsl(0UL, VLEN) ;
    __vr vrsum23_r0s2 = _vel_pvbrd_vsl(0UL, VLEN) ;

    __vr vrsum01_r1s0 = _vel_pvbrd_vsl(0UL, VLEN) ;
    __vr vrsum23_r1s0 = _vel_pvbrd_vsl(0UL, VLEN) ;

    __vr vrsum01_r1s1 = _vel_pvbrd_vsl(0UL, VLEN) ;
    __vr vrsum23_r1s1 = _vel_pvbrd_vsl(0UL, VLEN) ;

    __vr vrsum01_r1s2 = _vel_pvbrd_vsl(0UL, VLEN) ;
    __vr vrsum23_r1s2 = _vel_pvbrd_vsl(0UL, VLEN) ;

    __vr vrsum01_r2s0 = _vel_pvbrd_vsl(0UL, VLEN) ;
    __vr vrsum23_r2s0 = _vel_pvbrd_vsl(0UL, VLEN) ;

    __vr vrsum01_r2s1 = _vel_pvbrd_vsl(0UL, VLEN) ;
    __vr vrsum23_r2s1 = _vel_pvbrd_vsl(0UL, VLEN) ;

    __vr vrsum01_r2s2 = _vel_pvbrd_vsl(0UL, VLEN) ;
    __vr vrsum23_r2s2 = _vel_pvbrd_vsl(0UL, VLEN) ;

    for (int64_t n=0; n<batch; n++) {
      for (int64_t y=0; y<gOutHeight; y+=nY) {

	const int64_t vl = gOutWidth * (gOutHeight - y < nY ? gOutHeight - y : nY) ;
	const int64_t gop = y * gOutWidth ;

	__vr vrh_r0 = _vel_vaddsl_vsvl(0*dilationHeight-padHeight+y*strideHeight, vri, vl) ;
	__vr vrh_r1 = _vel_vaddsl_vsvl(1*dilationHeight-padHeight+y*strideHeight, vri, vl) ;
	__vr vrh_r2 = _vel_vaddsl_vsvl(2*dilationHeight-padHeight+y*strideHeight, vri, vl) ;


	__vm256 vmh0_r0 =  _vel_vfmklge_mvl(vrh_r0, vl) ;					// condition(0 <= h)
	__vm256 vmh1_r0 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(inHeight,vrh_r0, vl), vl) ;	// condition(h < inHeight)
	__vm256 vmh_r0  = _vel_andm_mmm(vmh0_r0, vmh1_r0) ;
	__vm256 vmhw_r0s0 = _vel_andm_mmm(vmh_r0, vmw_s0) ;
	__vm256 vmhw_r0s1 = _vel_andm_mmm(vmh_r0, vmw_s1) ;
	__vm256 vmhw_r0s2 = _vel_andm_mmm(vmh_r0, vmw_s2) ;

	__vm256 vmh0_r1 =  _vel_vfmklge_mvl(vrh_r1, vl) ;					// condition(0 <= h)
	__vm256 vmh1_r1 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(inHeight,vrh_r1, vl), vl) ;	// condition(h < inHeight)
	__vm256 vmh_r1  = _vel_andm_mmm(vmh0_r1, vmh1_r1) ;
	__vm256 vmhw_r1s0 = _vel_andm_mmm(vmh_r1, vmw_s0) ;
	__vm256 vmhw_r1s1 = _vel_andm_mmm(vmh_r1, vmw_s1) ;
	__vm256 vmhw_r1s2 = _vel_andm_mmm(vmh_r1, vmw_s2) ;

	__vm256 vmh0_r2 =  _vel_vfmklge_mvl(vrh_r2, vl) ;					// condition(0 <= h)
	__vm256 vmh1_r2 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(inHeight,vrh_r2, vl), vl) ;	// condition(h < inHeight)
	__vm256 vmh_r2  = _vel_andm_mmm(vmh0_r2, vmh1_r2) ;
	__vm256 vmhw_r2s0 = _vel_andm_mmm(vmh_r2, vmw_s0) ;
	__vm256 vmhw_r2s1 = _vel_andm_mmm(vmh_r2, vmw_s1) ;
	__vm256 vmhw_r2s2 = _vel_andm_mmm(vmh_r2, vmw_s2) ;


	const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;


	__vr vrpin_r0s0 = _vel_vsfa_vvssl(_vel_vaddul_vvvl(vrw_s0, _vel_vmulul_vsvl(inWidth,vrh_r0, vl), vl), 2, (uint64_t)pInChannel, vl) ;
	__vr vrin_r0s0 = _vel_vgtu_vvssml(vrpin_r0s0, 0, 0, vmhw_r0s0, vl) ;
	__vr vrpin_r0s1 = _vel_vsfa_vvssl(_vel_vaddul_vvvl(vrw_s1, _vel_vmulul_vsvl(inWidth,vrh_r0, vl), vl), 2, (uint64_t)pInChannel, vl) ;
	__vr vrin_r0s1 = _vel_vgtu_vvssml(vrpin_r0s1, 0, 0, vmhw_r0s1, vl) ;
	__vr vrpin_r0s2 = _vel_vsfa_vvssl(_vel_vaddul_vvvl(vrw_s2, _vel_vmulul_vsvl(inWidth,vrh_r0, vl), vl), 2, (uint64_t)pInChannel, vl) ;
	__vr vrin_r0s2 = _vel_vgtu_vvssml(vrpin_r0s2, 0, 0, vmhw_r0s2, vl) ;

	__vr vrpin_r1s0 = _vel_vsfa_vvssl(_vel_vaddul_vvvl(vrw_s0, _vel_vmulul_vsvl(inWidth,vrh_r1, vl), vl), 2, (uint64_t)pInChannel, vl) ;
	__vr vrin_r1s0 = _vel_vgtu_vvssml(vrpin_r1s0, 0, 0, vmhw_r1s0, vl) ;
	__vr vrpin_r1s1 = _vel_vsfa_vvssl(_vel_vaddul_vvvl(vrw_s1, _vel_vmulul_vsvl(inWidth,vrh_r1, vl), vl), 2, (uint64_t)pInChannel, vl) ;
	__vr vrin_r1s1 = _vel_vgtu_vvssml(vrpin_r1s1, 0, 0, vmhw_r1s1, vl) ;
	__vr vrpin_r1s2 = _vel_vsfa_vvssl(_vel_vaddul_vvvl(vrw_s2, _vel_vmulul_vsvl(inWidth,vrh_r1, vl), vl), 2, (uint64_t)pInChannel, vl) ;
	__vr vrin_r1s2 = _vel_vgtu_vvssml(vrpin_r1s2, 0, 0, vmhw_r1s2, vl) ;

	__vr vrpin_r2s0 = _vel_vsfa_vvssl(_vel_vaddul_vvvl(vrw_s0, _vel_vmulul_vsvl(inWidth,vrh_r2, vl), vl), 2, (uint64_t)pInChannel, vl) ;
	__vr vrin_r2s0 = _vel_vgtu_vvssml(vrpin_r2s0, 0, 0, vmhw_r2s0, vl) ;
	__vr vrpin_r2s1 = _vel_vsfa_vvssl(_vel_vaddul_vvvl(vrw_s1, _vel_vmulul_vsvl(inWidth,vrh_r2, vl), vl), 2, (uint64_t)pInChannel, vl) ;
	__vr vrin_r2s1 = _vel_vgtu_vvssml(vrpin_r2s1, 0, 0, vmhw_r2s1, vl) ;
	__vr vrpin_r2s2 = _vel_vsfa_vvssl(_vel_vaddul_vvvl(vrw_s2, _vel_vmulul_vsvl(inWidth,vrh_r2, vl), vl), 2, (uint64_t)pInChannel, vl) ;
	__vr vrin_r2s2 = _vel_vgtu_vvssml(vrpin_r2s2, 0, 0, vmhw_r2s2, vl) ;

	__vr vrgout0 = _vel_vldu_vssl(4, pGOut+outGroupOffset + ((n * gOutChannel + k  ) * gOutHeight ) * gOutWidth + gop, vl) ;
	__vr vrgout1 = _vel_vldu_vssl(4, pGOut+outGroupOffset + ((n * gOutChannel + k+1) * gOutHeight ) * gOutWidth + gop, vl) ;
	__vr vrgout2 = _vel_vldu_vssl(4, pGOut+outGroupOffset + ((n * gOutChannel + k+2) * gOutHeight ) * gOutWidth + gop, vl) ;
	__vr vrgout3 = _vel_vldu_vssl(4, pGOut+outGroupOffset + ((n * gOutChannel + k+3) * gOutHeight ) * gOutWidth + gop, vl) ;
	__vr vrgout4 = _vel_vldu_vssl(4, pGOut+outGroupOffset + ((n * gOutChannel + k+4) * gOutHeight ) * gOutWidth + gop, vl) ;
	__vr vrgout5 = _vel_vldu_vssl(4, pGOut+outGroupOffset + ((n * gOutChannel + k+5) * gOutHeight ) * gOutWidth + gop, vl) ;
	__vr vrgout6 = _vel_vldu_vssl(4, pGOut+outGroupOffset + ((n * gOutChannel + k+6) * gOutHeight ) * gOutWidth + gop, vl) ;
	__vr vrgout7 = _vel_vldu_vssl(4, pGOut+outGroupOffset + ((n * gOutChannel + k+7) * gOutHeight ) * gOutWidth + gop, vl) ;

	__vr vrgout01 = _vel_vshf_vvvsl(vrgout0, vrgout1, VE_VSHUFFLE_YUZU, vl) ;
	__vr vrgout23 = _vel_vshf_vvvsl(vrgout2, vrgout3, VE_VSHUFFLE_YUZU, vl) ;
	__vr vrgout45 = _vel_vshf_vvvsl(vrgout4, vrgout5, VE_VSHUFFLE_YUZU, vl) ;
	__vr vrgout67 = _vel_vshf_vvvsl(vrgout6, vrgout7, VE_VSHUFFLE_YUZU, vl) ;

	vrin_r0s0 = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrin_r0s0, vmhw_r0s0, vl) ;
	__vr vrinP_r0s0 = _vel_vshf_vvvsl(vrin_r0s0, vrin_r0s0, VE_VSHUFFLE_YUZU, vl) ;
	vrsum01_r0s0 = _vel_pvfmad_vvvvvl(vrsum01_r0s0, vrinP_r0s0, vrgout01, vrsum01_r0s0, vl) ;
	vrsum23_r0s0 = _vel_pvfmad_vvvvvl(vrsum23_r0s0, vrinP_r0s0, vrgout23, vrsum23_r0s0, vl) ;

	vrin_r0s1 = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrin_r0s1, vmhw_r0s1, vl) ;
	__vr vrinP_r0s1 = _vel_vshf_vvvsl(vrin_r0s1, vrin_r0s1, VE_VSHUFFLE_YUZU, vl) ;
	vrsum01_r0s1 = _vel_pvfmad_vvvvvl(vrsum01_r0s1, vrinP_r0s1, vrgout01, vrsum01_r0s1, vl) ;
	vrsum23_r0s1 = _vel_pvfmad_vvvvvl(vrsum23_r0s1, vrinP_r0s1, vrgout23, vrsum23_r0s1, vl) ;

	vrin_r0s2 = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrin_r0s2, vmhw_r0s2, vl) ;
	__vr vrinP_r0s2 = _vel_vshf_vvvsl(vrin_r0s2, vrin_r0s2, VE_VSHUFFLE_YUZU, vl) ;
	vrsum01_r0s2 = _vel_pvfmad_vvvvvl(vrsum01_r0s2, vrinP_r0s2, vrgout01, vrsum01_r0s2, vl) ;
	vrsum23_r0s2 = _vel_pvfmad_vvvvvl(vrsum23_r0s2, vrinP_r0s2, vrgout23, vrsum23_r0s2, vl) ;

	vrin_r1s0 = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrin_r1s0, vmhw_r1s0, vl) ;
	__vr vrinP_r1s0 = _vel_vshf_vvvsl(vrin_r1s0, vrin_r1s0, VE_VSHUFFLE_YUZU, vl) ;
	vrsum01_r1s0 = _vel_pvfmad_vvvvvl(vrsum01_r1s0, vrinP_r1s0, vrgout01, vrsum01_r1s0, vl) ;
	vrsum23_r1s0 = _vel_pvfmad_vvvvvl(vrsum23_r1s0, vrinP_r1s0, vrgout23, vrsum23_r1s0, vl) ;

	vrin_r1s1 = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrin_r1s1, vmhw_r1s1, vl) ;
	__vr vrinP_r1s1 = _vel_vshf_vvvsl(vrin_r1s1, vrin_r1s1, VE_VSHUFFLE_YUZU, vl) ;
	vrsum01_r1s1 = _vel_pvfmad_vvvvvl(vrsum01_r1s1, vrinP_r1s1, vrgout01, vrsum01_r1s1, vl) ;
	vrsum23_r1s1 = _vel_pvfmad_vvvvvl(vrsum23_r1s1, vrinP_r1s1, vrgout23, vrsum23_r1s1, vl) ;

	vrin_r1s2 = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrin_r1s2, vmhw_r1s2, vl) ;
	__vr vrinP_r1s2 = _vel_vshf_vvvsl(vrin_r1s2, vrin_r1s2, VE_VSHUFFLE_YUZU, vl) ;
	vrsum01_r1s2 = _vel_pvfmad_vvvvvl(vrsum01_r1s2, vrinP_r1s2, vrgout01, vrsum01_r1s2, vl) ;
	vrsum23_r1s2 = _vel_pvfmad_vvvvvl(vrsum23_r1s2, vrinP_r1s2, vrgout23, vrsum23_r1s2, vl) ;

	vrin_r2s0 = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrin_r2s0, vmhw_r2s0, vl) ;
	__vr vrinP_r2s0 = _vel_vshf_vvvsl(vrin_r2s0, vrin_r2s0, VE_VSHUFFLE_YUZU, vl) ;
	vrsum01_r2s0 = _vel_pvfmad_vvvvvl(vrsum01_r2s0, vrinP_r2s0, vrgout01, vrsum01_r2s0, vl) ;
	vrsum23_r2s0 = _vel_pvfmad_vvvvvl(vrsum23_r2s0, vrinP_r2s0, vrgout23, vrsum23_r2s0, vl) ;

	vrin_r2s1 = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrin_r2s1, vmhw_r2s1, vl) ;
	__vr vrinP_r2s1 = _vel_vshf_vvvsl(vrin_r2s1, vrin_r2s1, VE_VSHUFFLE_YUZU, vl) ;
	vrsum01_r2s1 = _vel_pvfmad_vvvvvl(vrsum01_r2s1, vrinP_r2s1, vrgout01, vrsum01_r2s1, vl) ;
	vrsum23_r2s1 = _vel_pvfmad_vvvvvl(vrsum23_r2s1, vrinP_r2s1, vrgout23, vrsum23_r2s1, vl) ;

	vrin_r2s2 = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrin_r2s2, vmhw_r2s2, vl) ;
	__vr vrinP_r2s2 = _vel_vshf_vvvsl(vrin_r2s2, vrin_r2s2, VE_VSHUFFLE_YUZU, vl) ;
	vrsum01_r2s2 = _vel_pvfmad_vvvvvl(vrsum01_r2s2, vrinP_r2s2, vrgout01, vrsum01_r2s2, vl) ;
	vrsum23_r2s2 = _vel_pvfmad_vvvvvl(vrsum23_r2s2, vrinP_r2s2, vrgout23, vrsum23_r2s2, vl) ;

      } // gOutPixels
    } // batch

#define SUM_AND_STORE(R, S, RSTOKEN)							\
    {											\
      __vr vrsum0 = _vel_vfsums_vvl(vrsum01_##RSTOKEN, VLEN) ;				\
      __vr vrsum1 = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsum01_##RSTOKEN,32, VLEN), VLEN);	\
      __vr vrsum2 = _vel_vfsums_vvl(vrsum23_##RSTOKEN, VLEN) ;				\
      __vr vrsum3 = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsum23_##RSTOKEN,32, VLEN), VLEN);	\
      _vel_vstu_vssl(vrsum0, 4, pGKernel+kernelIndex + 0 * inChannelGroup * gKernHeight * gKernWidth + (R) * gKernWidth + (S), 1) ;	\
      _vel_vstu_vssl(vrsum1, 4, pGKernel+kernelIndex + 1 * inChannelGroup * gKernHeight * gKernWidth + (R) * gKernWidth + (S), 1) ;	\
      _vel_vstu_vssl(vrsum2, 4, pGKernel+kernelIndex + 2 * inChannelGroup * gKernHeight * gKernWidth + (R) * gKernWidth + (S), 1) ;	\
      _vel_vstu_vssl(vrsum3, 4, pGKernel+kernelIndex + 3 * inChannelGroup * gKernHeight * gKernWidth + (R) * gKernWidth + (S), 1) ;	\
    }
    SUM_AND_STORE(0,0, r0s0) ;
    SUM_AND_STORE(0,1, r0s1) ;
    SUM_AND_STORE(0,2, r0s2) ;
    SUM_AND_STORE(1,0, r1s0) ;
    SUM_AND_STORE(1,1, r1s1) ;
    SUM_AND_STORE(1,2, r1s2) ;
    SUM_AND_STORE(2,0, r2s0) ;
    SUM_AND_STORE(2,1, r2s1) ;
    SUM_AND_STORE(2,2, r2s2) ;
#undef SUM_AND_STORE

  } // inChannel
}

static inline void f8(
  const float * pIn, const int64_t inWidth, const int64_t inHeight,
  const float * pGOut, const int64_t gOutWidth, const int64_t gOutHeight,
  float * const pGKernel, const int64_t gKernWidth , const int64_t gKernHeight,
  const int64_t strideHeight, const int64_t strideWidth,
  const int64_t padHeight, const int64_t padWidth,
  const int64_t dilationHeight, const int64_t dilationWidth,
  const int64_t inChannelGroup, const int64_t inChannel, const int64_t gOutChannel,
  const int64_t inGroupOffset, const int64_t outGroupOffset, const int64_t kernGroupOffset,
  const int64_t batch, const int64_t k,
  const int64_t nY,
  const __vr vri,
  const __vr vrj
)
{
  __vr vrw_s0 = _vel_vaddsl_vsvl(0*dilationWidth-padWidth,  vrj, nY*gOutWidth) ;
  __vr vrw_s1 = _vel_vaddsl_vsvl(1*dilationWidth-padWidth,  vrj, nY*gOutWidth) ;
  __vr vrw_s2 = _vel_vaddsl_vsvl(2*dilationWidth-padWidth,  vrj, nY*gOutWidth) ;

  __vm256 vmw0_s0 =  _vel_vfmklge_mvl(vrw_s0, nY*gOutWidth) ;						// condition(0 <= w)
  __vm256 vmw1_s0 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(inWidth,vrw_s0, nY*gOutWidth), nY*gOutWidth) ;	// condition(w < inWidth)
  __vm256 vmw_s0  = _vel_andm_mmm(vmw0_s0, vmw1_s0) ;

  __vm256 vmw0_s1 =  _vel_vfmklge_mvl(vrw_s1, nY*gOutWidth) ;						// condition(0 <= w)
  __vm256 vmw1_s1 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(inWidth,vrw_s1, nY*gOutWidth), nY*gOutWidth) ;	// condition(w < inWidth)
  __vm256 vmw_s1  = _vel_andm_mmm(vmw0_s1, vmw1_s1) ;

  __vm256 vmw0_s2 =  _vel_vfmklge_mvl(vrw_s2, nY*gOutWidth) ;						// condition(0 <= w)
  __vm256 vmw1_s2 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(inWidth,vrw_s2, nY*gOutWidth), nY*gOutWidth) ;	// condition(w < inWidth)
  __vm256 vmw_s2  = _vel_andm_mmm(vmw0_s2, vmw1_s2) ;

  for (int64_t c=0; c<inChannelGroup; c++) {
    const int64_t kernelIndex = kernGroupOffset + ((k     * inChannelGroup + c) * gKernHeight ) * gKernWidth ;

    __vr vrsum01_r0s0 = _vel_pvbrd_vsl(0UL, VLEN) ;
    __vr vrsum23_r0s0 = _vel_pvbrd_vsl(0UL, VLEN) ;
    __vr vrsum45_r0s0 = _vel_pvbrd_vsl(0UL, VLEN) ;
    __vr vrsum67_r0s0 = _vel_pvbrd_vsl(0UL, VLEN) ;

    __vr vrsum01_r0s1 = _vel_pvbrd_vsl(0UL, VLEN) ;
    __vr vrsum23_r0s1 = _vel_pvbrd_vsl(0UL, VLEN) ;
    __vr vrsum45_r0s1 = _vel_pvbrd_vsl(0UL, VLEN) ;
    __vr vrsum67_r0s1 = _vel_pvbrd_vsl(0UL, VLEN) ;

    __vr vrsum01_r0s2 = _vel_pvbrd_vsl(0UL, VLEN) ;
    __vr vrsum23_r0s2 = _vel_pvbrd_vsl(0UL, VLEN) ;
    __vr vrsum45_r0s2 = _vel_pvbrd_vsl(0UL, VLEN) ;
    __vr vrsum67_r0s2 = _vel_pvbrd_vsl(0UL, VLEN) ;

    __vr vrsum01_r1s0 = _vel_pvbrd_vsl(0UL, VLEN) ;
    __vr vrsum23_r1s0 = _vel_pvbrd_vsl(0UL, VLEN) ;
    __vr vrsum45_r1s0 = _vel_pvbrd_vsl(0UL, VLEN) ;
    __vr vrsum67_r1s0 = _vel_pvbrd_vsl(0UL, VLEN) ;

    __vr vrsum01_r1s1 = _vel_pvbrd_vsl(0UL, VLEN) ;
    __vr vrsum23_r1s1 = _vel_pvbrd_vsl(0UL, VLEN) ;
    __vr vrsum45_r1s1 = _vel_pvbrd_vsl(0UL, VLEN) ;
    __vr vrsum67_r1s1 = _vel_pvbrd_vsl(0UL, VLEN) ;

    __vr vrsum01_r1s2 = _vel_pvbrd_vsl(0UL, VLEN) ;
    __vr vrsum23_r1s2 = _vel_pvbrd_vsl(0UL, VLEN) ;
    __vr vrsum45_r1s2 = _vel_pvbrd_vsl(0UL, VLEN) ;
    __vr vrsum67_r1s2 = _vel_pvbrd_vsl(0UL, VLEN) ;

    __vr vrsum01_r2s0 = _vel_pvbrd_vsl(0UL, VLEN) ;
    __vr vrsum23_r2s0 = _vel_pvbrd_vsl(0UL, VLEN) ;
    __vr vrsum45_r2s0 = _vel_pvbrd_vsl(0UL, VLEN) ;
    __vr vrsum67_r2s0 = _vel_pvbrd_vsl(0UL, VLEN) ;

    __vr vrsum01_r2s1 = _vel_pvbrd_vsl(0UL, VLEN) ;
    __vr vrsum23_r2s1 = _vel_pvbrd_vsl(0UL, VLEN) ;
    __vr vrsum45_r2s1 = _vel_pvbrd_vsl(0UL, VLEN) ;
    __vr vrsum67_r2s1 = _vel_pvbrd_vsl(0UL, VLEN) ;

    __vr vrsum01_r2s2 = _vel_pvbrd_vsl(0UL, VLEN) ;
    __vr vrsum23_r2s2 = _vel_pvbrd_vsl(0UL, VLEN) ;
    __vr vrsum45_r2s2 = _vel_pvbrd_vsl(0UL, VLEN) ;
    __vr vrsum67_r2s2 = _vel_pvbrd_vsl(0UL, VLEN) ;

    for (int64_t n=0; n<batch; n++) {
      for (int64_t y=0; y<gOutHeight; y+=nY) {

	const int64_t vl = gOutWidth * (gOutHeight - y < nY ? gOutHeight - y : nY) ;
	const int64_t gop = y * gOutWidth ;

	__vr vrh_r0 = _vel_vaddsl_vsvl(0*dilationHeight-padHeight+y*strideHeight, vri, vl) ;
	__vr vrh_r1 = _vel_vaddsl_vsvl(1*dilationHeight-padHeight+y*strideHeight, vri, vl) ;
	__vr vrh_r2 = _vel_vaddsl_vsvl(2*dilationHeight-padHeight+y*strideHeight, vri, vl) ;


	__vm256 vmh0_r0 =  _vel_vfmklge_mvl(vrh_r0, vl) ;					// condition(0 <= h)
	__vm256 vmh1_r0 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(inHeight,vrh_r0, vl), vl) ;	// condition(h < inHeight)
	__vm256 vmh_r0  = _vel_andm_mmm(vmh0_r0, vmh1_r0) ;
	__vm256 vmhw_r0s0 = _vel_andm_mmm(vmh_r0, vmw_s0) ;
	__vm256 vmhw_r0s1 = _vel_andm_mmm(vmh_r0, vmw_s1) ;
	__vm256 vmhw_r0s2 = _vel_andm_mmm(vmh_r0, vmw_s2) ;

	__vm256 vmh0_r1 =  _vel_vfmklge_mvl(vrh_r1, vl) ;					// condition(0 <= h)
	__vm256 vmh1_r1 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(inHeight,vrh_r1, vl), vl) ;	// condition(h < inHeight)
	__vm256 vmh_r1  = _vel_andm_mmm(vmh0_r1, vmh1_r1) ;
	__vm256 vmhw_r1s0 = _vel_andm_mmm(vmh_r1, vmw_s0) ;
	__vm256 vmhw_r1s1 = _vel_andm_mmm(vmh_r1, vmw_s1) ;
	__vm256 vmhw_r1s2 = _vel_andm_mmm(vmh_r1, vmw_s2) ;

	__vm256 vmh0_r2 =  _vel_vfmklge_mvl(vrh_r2, vl) ;					// condition(0 <= h)
	__vm256 vmh1_r2 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(inHeight,vrh_r2, vl), vl) ;	// condition(h < inHeight)
	__vm256 vmh_r2  = _vel_andm_mmm(vmh0_r2, vmh1_r2) ;
	__vm256 vmhw_r2s0 = _vel_andm_mmm(vmh_r2, vmw_s0) ;
	__vm256 vmhw_r2s1 = _vel_andm_mmm(vmh_r2, vmw_s1) ;
	__vm256 vmhw_r2s2 = _vel_andm_mmm(vmh_r2, vmw_s2) ;


	const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;


	__vr vrpin_r0s0 = _vel_vsfa_vvssl(_vel_vaddul_vvvl(vrw_s0, _vel_vmulul_vsvl(inWidth,vrh_r0, vl), vl), 2, (uint64_t)pInChannel, vl) ;
	__vr vrin_r0s0 = _vel_vgtu_vvssml(vrpin_r0s0, 0, 0, vmhw_r0s0, vl) ;
	__vr vrpin_r0s1 = _vel_vsfa_vvssl(_vel_vaddul_vvvl(vrw_s1, _vel_vmulul_vsvl(inWidth,vrh_r0, vl), vl), 2, (uint64_t)pInChannel, vl) ;
	__vr vrin_r0s1 = _vel_vgtu_vvssml(vrpin_r0s1, 0, 0, vmhw_r0s1, vl) ;
	__vr vrpin_r0s2 = _vel_vsfa_vvssl(_vel_vaddul_vvvl(vrw_s2, _vel_vmulul_vsvl(inWidth,vrh_r0, vl), vl), 2, (uint64_t)pInChannel, vl) ;
	__vr vrin_r0s2 = _vel_vgtu_vvssml(vrpin_r0s2, 0, 0, vmhw_r0s2, vl) ;

	__vr vrpin_r1s0 = _vel_vsfa_vvssl(_vel_vaddul_vvvl(vrw_s0, _vel_vmulul_vsvl(inWidth,vrh_r1, vl), vl), 2, (uint64_t)pInChannel, vl) ;
	__vr vrin_r1s0 = _vel_vgtu_vvssml(vrpin_r1s0, 0, 0, vmhw_r1s0, vl) ;
	__vr vrpin_r1s1 = _vel_vsfa_vvssl(_vel_vaddul_vvvl(vrw_s1, _vel_vmulul_vsvl(inWidth,vrh_r1, vl), vl), 2, (uint64_t)pInChannel, vl) ;
	__vr vrin_r1s1 = _vel_vgtu_vvssml(vrpin_r1s1, 0, 0, vmhw_r1s1, vl) ;
	__vr vrpin_r1s2 = _vel_vsfa_vvssl(_vel_vaddul_vvvl(vrw_s2, _vel_vmulul_vsvl(inWidth,vrh_r1, vl), vl), 2, (uint64_t)pInChannel, vl) ;
	__vr vrin_r1s2 = _vel_vgtu_vvssml(vrpin_r1s2, 0, 0, vmhw_r1s2, vl) ;

	__vr vrpin_r2s0 = _vel_vsfa_vvssl(_vel_vaddul_vvvl(vrw_s0, _vel_vmulul_vsvl(inWidth,vrh_r2, vl), vl), 2, (uint64_t)pInChannel, vl) ;
	__vr vrin_r2s0 = _vel_vgtu_vvssml(vrpin_r2s0, 0, 0, vmhw_r2s0, vl) ;
	__vr vrpin_r2s1 = _vel_vsfa_vvssl(_vel_vaddul_vvvl(vrw_s1, _vel_vmulul_vsvl(inWidth,vrh_r2, vl), vl), 2, (uint64_t)pInChannel, vl) ;
	__vr vrin_r2s1 = _vel_vgtu_vvssml(vrpin_r2s1, 0, 0, vmhw_r2s1, vl) ;
	__vr vrpin_r2s2 = _vel_vsfa_vvssl(_vel_vaddul_vvvl(vrw_s2, _vel_vmulul_vsvl(inWidth,vrh_r2, vl), vl), 2, (uint64_t)pInChannel, vl) ;
	__vr vrin_r2s2 = _vel_vgtu_vvssml(vrpin_r2s2, 0, 0, vmhw_r2s2, vl) ;

	__vr vrgout0 = _vel_vldu_vssl(4, pGOut+outGroupOffset + ((n * gOutChannel + k  ) * gOutHeight ) * gOutWidth + gop, vl) ;
	__vr vrgout1 = _vel_vldu_vssl(4, pGOut+outGroupOffset + ((n * gOutChannel + k+1) * gOutHeight ) * gOutWidth + gop, vl) ;
	__vr vrgout2 = _vel_vldu_vssl(4, pGOut+outGroupOffset + ((n * gOutChannel + k+2) * gOutHeight ) * gOutWidth + gop, vl) ;
	__vr vrgout3 = _vel_vldu_vssl(4, pGOut+outGroupOffset + ((n * gOutChannel + k+3) * gOutHeight ) * gOutWidth + gop, vl) ;
	__vr vrgout4 = _vel_vldu_vssl(4, pGOut+outGroupOffset + ((n * gOutChannel + k+4) * gOutHeight ) * gOutWidth + gop, vl) ;
	__vr vrgout5 = _vel_vldu_vssl(4, pGOut+outGroupOffset + ((n * gOutChannel + k+5) * gOutHeight ) * gOutWidth + gop, vl) ;
	__vr vrgout6 = _vel_vldu_vssl(4, pGOut+outGroupOffset + ((n * gOutChannel + k+6) * gOutHeight ) * gOutWidth + gop, vl) ;
	__vr vrgout7 = _vel_vldu_vssl(4, pGOut+outGroupOffset + ((n * gOutChannel + k+7) * gOutHeight ) * gOutWidth + gop, vl) ;

	__vr vrgout01 = _vel_vshf_vvvsl(vrgout0, vrgout1, VE_VSHUFFLE_YUZU, vl) ;
	__vr vrgout23 = _vel_vshf_vvvsl(vrgout2, vrgout3, VE_VSHUFFLE_YUZU, vl) ;
	__vr vrgout45 = _vel_vshf_vvvsl(vrgout4, vrgout5, VE_VSHUFFLE_YUZU, vl) ;
	__vr vrgout67 = _vel_vshf_vvvsl(vrgout6, vrgout7, VE_VSHUFFLE_YUZU, vl) ;

	vrin_r0s0 = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrin_r0s0, vmhw_r0s0, vl) ;
	__vr vrinP_r0s0 = _vel_vshf_vvvsl(vrin_r0s0, vrin_r0s0, VE_VSHUFFLE_YUZU, vl) ;
	vrsum01_r0s0 = _vel_pvfmad_vvvvvl(vrsum01_r0s0, vrinP_r0s0, vrgout01, vrsum01_r0s0, vl) ;
	vrsum23_r0s0 = _vel_pvfmad_vvvvvl(vrsum23_r0s0, vrinP_r0s0, vrgout23, vrsum23_r0s0, vl) ;
	vrsum45_r0s0 = _vel_pvfmad_vvvvvl(vrsum45_r0s0, vrinP_r0s0, vrgout45, vrsum45_r0s0, vl) ;
	vrsum67_r0s0 = _vel_pvfmad_vvvvvl(vrsum67_r0s0, vrinP_r0s0, vrgout67, vrsum67_r0s0, vl) ;

	vrin_r0s1 = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrin_r0s1, vmhw_r0s1, vl) ;
	__vr vrinP_r0s1 = _vel_vshf_vvvsl(vrin_r0s1, vrin_r0s1, VE_VSHUFFLE_YUZU, vl) ;
	vrsum01_r0s1 = _vel_pvfmad_vvvvvl(vrsum01_r0s1, vrinP_r0s1, vrgout01, vrsum01_r0s1, vl) ;
	vrsum23_r0s1 = _vel_pvfmad_vvvvvl(vrsum23_r0s1, vrinP_r0s1, vrgout23, vrsum23_r0s1, vl) ;
	vrsum45_r0s1 = _vel_pvfmad_vvvvvl(vrsum45_r0s1, vrinP_r0s1, vrgout45, vrsum45_r0s1, vl) ;
	vrsum67_r0s1 = _vel_pvfmad_vvvvvl(vrsum67_r0s1, vrinP_r0s1, vrgout67, vrsum67_r0s1, vl) ;

	vrin_r0s2 = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrin_r0s2, vmhw_r0s2, vl) ;
	__vr vrinP_r0s2 = _vel_vshf_vvvsl(vrin_r0s2, vrin_r0s2, VE_VSHUFFLE_YUZU, vl) ;
	vrsum01_r0s2 = _vel_pvfmad_vvvvvl(vrsum01_r0s2, vrinP_r0s2, vrgout01, vrsum01_r0s2, vl) ;
	vrsum23_r0s2 = _vel_pvfmad_vvvvvl(vrsum23_r0s2, vrinP_r0s2, vrgout23, vrsum23_r0s2, vl) ;
	vrsum45_r0s2 = _vel_pvfmad_vvvvvl(vrsum45_r0s2, vrinP_r0s2, vrgout45, vrsum45_r0s2, vl) ;
	vrsum67_r0s2 = _vel_pvfmad_vvvvvl(vrsum67_r0s2, vrinP_r0s2, vrgout67, vrsum67_r0s2, vl) ;

	vrin_r1s0 = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrin_r1s0, vmhw_r1s0, vl) ;
	__vr vrinP_r1s0 = _vel_vshf_vvvsl(vrin_r1s0, vrin_r1s0, VE_VSHUFFLE_YUZU, vl) ;
	vrsum01_r1s0 = _vel_pvfmad_vvvvvl(vrsum01_r1s0, vrinP_r1s0, vrgout01, vrsum01_r1s0, vl) ;
	vrsum23_r1s0 = _vel_pvfmad_vvvvvl(vrsum23_r1s0, vrinP_r1s0, vrgout23, vrsum23_r1s0, vl) ;
	vrsum45_r1s0 = _vel_pvfmad_vvvvvl(vrsum45_r1s0, vrinP_r1s0, vrgout45, vrsum45_r1s0, vl) ;
	vrsum67_r1s0 = _vel_pvfmad_vvvvvl(vrsum67_r1s0, vrinP_r1s0, vrgout67, vrsum67_r1s0, vl) ;

	vrin_r1s1 = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrin_r1s1, vmhw_r1s1, vl) ;
	__vr vrinP_r1s1 = _vel_vshf_vvvsl(vrin_r1s1, vrin_r1s1, VE_VSHUFFLE_YUZU, vl) ;
	vrsum01_r1s1 = _vel_pvfmad_vvvvvl(vrsum01_r1s1, vrinP_r1s1, vrgout01, vrsum01_r1s1, vl) ;
	vrsum23_r1s1 = _vel_pvfmad_vvvvvl(vrsum23_r1s1, vrinP_r1s1, vrgout23, vrsum23_r1s1, vl) ;
	vrsum45_r1s1 = _vel_pvfmad_vvvvvl(vrsum45_r1s1, vrinP_r1s1, vrgout45, vrsum45_r1s1, vl) ;
	vrsum67_r1s1 = _vel_pvfmad_vvvvvl(vrsum67_r1s1, vrinP_r1s1, vrgout67, vrsum67_r1s1, vl) ;

	vrin_r1s2 = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrin_r1s2, vmhw_r1s2, vl) ;
	__vr vrinP_r1s2 = _vel_vshf_vvvsl(vrin_r1s2, vrin_r1s2, VE_VSHUFFLE_YUZU, vl) ;
	vrsum01_r1s2 = _vel_pvfmad_vvvvvl(vrsum01_r1s2, vrinP_r1s2, vrgout01, vrsum01_r1s2, vl) ;
	vrsum23_r1s2 = _vel_pvfmad_vvvvvl(vrsum23_r1s2, vrinP_r1s2, vrgout23, vrsum23_r1s2, vl) ;
	vrsum45_r1s2 = _vel_pvfmad_vvvvvl(vrsum45_r1s2, vrinP_r1s2, vrgout45, vrsum45_r1s2, vl) ;
	vrsum67_r1s2 = _vel_pvfmad_vvvvvl(vrsum67_r1s2, vrinP_r1s2, vrgout67, vrsum67_r1s2, vl) ;

	vrin_r2s0 = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrin_r2s0, vmhw_r2s0, vl) ;
	__vr vrinP_r2s0 = _vel_vshf_vvvsl(vrin_r2s0, vrin_r2s0, VE_VSHUFFLE_YUZU, vl) ;
	vrsum01_r2s0 = _vel_pvfmad_vvvvvl(vrsum01_r2s0, vrinP_r2s0, vrgout01, vrsum01_r2s0, vl) ;
	vrsum23_r2s0 = _vel_pvfmad_vvvvvl(vrsum23_r2s0, vrinP_r2s0, vrgout23, vrsum23_r2s0, vl) ;
	vrsum45_r2s0 = _vel_pvfmad_vvvvvl(vrsum45_r2s0, vrinP_r2s0, vrgout45, vrsum45_r2s0, vl) ;
	vrsum67_r2s0 = _vel_pvfmad_vvvvvl(vrsum67_r2s0, vrinP_r2s0, vrgout67, vrsum67_r2s0, vl) ;

	vrin_r2s1 = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrin_r2s1, vmhw_r2s1, vl) ;
	__vr vrinP_r2s1 = _vel_vshf_vvvsl(vrin_r2s1, vrin_r2s1, VE_VSHUFFLE_YUZU, vl) ;
	vrsum01_r2s1 = _vel_pvfmad_vvvvvl(vrsum01_r2s1, vrinP_r2s1, vrgout01, vrsum01_r2s1, vl) ;
	vrsum23_r2s1 = _vel_pvfmad_vvvvvl(vrsum23_r2s1, vrinP_r2s1, vrgout23, vrsum23_r2s1, vl) ;
	vrsum45_r2s1 = _vel_pvfmad_vvvvvl(vrsum45_r2s1, vrinP_r2s1, vrgout45, vrsum45_r2s1, vl) ;
	vrsum67_r2s1 = _vel_pvfmad_vvvvvl(vrsum67_r2s1, vrinP_r2s1, vrgout67, vrsum67_r2s1, vl) ;

	vrin_r2s2 = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrin_r2s2, vmhw_r2s2, vl) ;
	__vr vrinP_r2s2 = _vel_vshf_vvvsl(vrin_r2s2, vrin_r2s2, VE_VSHUFFLE_YUZU, vl) ;
	vrsum01_r2s2 = _vel_pvfmad_vvvvvl(vrsum01_r2s2, vrinP_r2s2, vrgout01, vrsum01_r2s2, vl) ;
	vrsum23_r2s2 = _vel_pvfmad_vvvvvl(vrsum23_r2s2, vrinP_r2s2, vrgout23, vrsum23_r2s2, vl) ;
	vrsum45_r2s2 = _vel_pvfmad_vvvvvl(vrsum45_r2s2, vrinP_r2s2, vrgout45, vrsum45_r2s2, vl) ;
	vrsum67_r2s2 = _vel_pvfmad_vvvvvl(vrsum67_r2s2, vrinP_r2s2, vrgout67, vrsum67_r2s2, vl) ;

      } // gOutPixels
    } // batch

#define SUM_AND_STORE(R, S, RSTOKEN)							\
    {											\
      __vr vrsum0 = _vel_vfsums_vvl(vrsum01_##RSTOKEN, VLEN) ;				\
      __vr vrsum1 = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsum01_##RSTOKEN,32, VLEN), VLEN);	\
      __vr vrsum2 = _vel_vfsums_vvl(vrsum23_##RSTOKEN, VLEN) ;				\
      __vr vrsum3 = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsum23_##RSTOKEN,32, VLEN), VLEN);	\
      __vr vrsum4 = _vel_vfsums_vvl(vrsum45_##RSTOKEN, VLEN) ;				\
      __vr vrsum5 = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsum45_##RSTOKEN,32, VLEN), VLEN);	\
      __vr vrsum6 = _vel_vfsums_vvl(vrsum67_##RSTOKEN, VLEN) ;				\
      __vr vrsum7 = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsum67_##RSTOKEN,32, VLEN), VLEN);	\
      _vel_vstu_vssl(vrsum0, 4, pGKernel+kernelIndex + 0 * inChannelGroup * gKernHeight * gKernWidth + (R) * gKernWidth + (S), 1) ;	\
      _vel_vstu_vssl(vrsum1, 4, pGKernel+kernelIndex + 1 * inChannelGroup * gKernHeight * gKernWidth + (R) * gKernWidth + (S), 1) ;	\
      _vel_vstu_vssl(vrsum2, 4, pGKernel+kernelIndex + 2 * inChannelGroup * gKernHeight * gKernWidth + (R) * gKernWidth + (S), 1) ;	\
      _vel_vstu_vssl(vrsum3, 4, pGKernel+kernelIndex + 3 * inChannelGroup * gKernHeight * gKernWidth + (R) * gKernWidth + (S), 1) ;	\
      _vel_vstu_vssl(vrsum4, 4, pGKernel+kernelIndex + 4 * inChannelGroup * gKernHeight * gKernWidth + (R) * gKernWidth + (S), 1) ;	\
      _vel_vstu_vssl(vrsum5, 4, pGKernel+kernelIndex + 5 * inChannelGroup * gKernHeight * gKernWidth + (R) * gKernWidth + (S), 1) ;	\
      _vel_vstu_vssl(vrsum6, 4, pGKernel+kernelIndex + 6 * inChannelGroup * gKernHeight * gKernWidth + (R) * gKernWidth + (S), 1) ;	\
      _vel_vstu_vssl(vrsum7, 4, pGKernel+kernelIndex + 7 * inChannelGroup * gKernHeight * gKernWidth + (R) * gKernWidth + (S), 1) ;	\
    }
    SUM_AND_STORE(0,0, r0s0) ;
    SUM_AND_STORE(0,1, r0s1) ;
    SUM_AND_STORE(0,2, r0s2) ;
    SUM_AND_STORE(1,0, r1s0) ;
    SUM_AND_STORE(1,1, r1s1) ;
    SUM_AND_STORE(1,2, r1s2) ;
    SUM_AND_STORE(2,0, r2s0) ;
    SUM_AND_STORE(2,1, r2s1) ;
    SUM_AND_STORE(2,2, r2s2) ;
#undef SUM_AND_STORE

  } // inChannel
}

vednnError_t
vednnConvolutionBackwardFilter_direct_ker3_owU128(
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
  const int64_t padWidth       = pParamConv->padWidth;
  const int64_t padHeight      = pParamConv->padHeight;
  const int64_t dilationWidth  = pParamConv->dilationWidth;
  const int64_t dilationHeight = pParamConv->dilationHeight;

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
    const int64_t nY = VLEN / gOutWidth ;

    __vr vrseq = _vel_vseq_vl(nY*gOutWidth) ;			// xy
    __vr vry  = _vel_vdivsl_vvsl(vrseq, gOutWidth, nY*gOutWidth) ;
    __vr vrx  = _vel_vsubsl_vvvl(vrseq, _vel_vmulul_vsvl(gOutWidth,vry, nY*gOutWidth), nY*gOutWidth) ;
    __vr vri  = _vel_vmulsl_vsvl(strideHeight, vry, nY*gOutWidth) ;
    __vr vrj  = _vel_vmulsl_vsvl(strideWidth,  vrx, nY*gOutWidth) ;

    for (int64_t g = 0; g < group; g++) {
      int64_t inGroupOffset   = g * inChannelGroup  * inHeight  * inWidth;
      int64_t outGroupOffset  = (g * gOutChannelGroup + beginOChannel) * gOutHeight * gOutWidth;
      int64_t kernGroupOffset = (g * gOutChannelGroup + beginOChannel) * inChannelGroup * gKernHeight * gKernWidth;

      int64_t k=0;
      if ( (nOChannel & 0x01) == 1 ) {
	f1(pIn, inWidth,inHeight,
	   pGOut, gOutWidth, gOutHeight,
	   pGKernel, gKernWidth, gKernHeight,
	   strideHeight, strideWidth,
	   padHeight, padWidth,
	   dilationHeight, dilationWidth,
	   inChannelGroup, inChannel, gOutChannel,
	   inGroupOffset,  outGroupOffset,  kernGroupOffset,
	   batch,  k,
	   nY, vri, vrj ) ;
	k+=1;
      }
      if ( ((nOChannel >> 1) & 0x01) == 1 ) {
	f2(pIn, inWidth,inHeight,
	   pGOut, gOutWidth, gOutHeight,
	   pGKernel, gKernWidth, gKernHeight,
	   strideHeight, strideWidth,
	   padHeight, padWidth,
	   dilationHeight, dilationWidth,
	   inChannelGroup, inChannel, gOutChannel,
	   inGroupOffset,  outGroupOffset,  kernGroupOffset,
	   batch,  k,
	   nY, vri, vrj ) ;
	k+=2;
      }
      if ( ((nOChannel >> 2) & 0x01) == 1 ) {
	f4(pIn, inWidth,inHeight,
	   pGOut, gOutWidth, gOutHeight,
	   pGKernel, gKernWidth, gKernHeight,
	   strideHeight, strideWidth,
	   padHeight, padWidth,
	   dilationHeight, dilationWidth,
	   inChannelGroup, inChannel, gOutChannel,
	   inGroupOffset,  outGroupOffset,  kernGroupOffset,
	   batch,  k,
	   nY, vri, vrj ) ;
	k+=4;
      }
      for ( ;k<nOChannel; k+=8) {
	f8(pIn, inWidth,inHeight,
	   pGOut, gOutWidth, gOutHeight,
	   pGKernel, gKernWidth, gKernHeight,
	   strideHeight, strideWidth,
	   padHeight, padWidth,
	   dilationHeight, dilationWidth,
	   inChannelGroup, inChannel, gOutChannel,
	   inGroupOffset,  outGroupOffset,  kernGroupOffset,
	   batch,  k,
	   nY, vri, vrj ) ;
      } // outChannel
    } // group
  } // batch

  return VEDNN_SUCCESS;
}
