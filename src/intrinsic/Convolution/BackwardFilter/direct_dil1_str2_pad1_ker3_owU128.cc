#include <stdint.h>
#include <stdlib.h>

#include "vednn.h"

#include "velintrin.h"
#define VLEN	(256)


template <int NUM_KERNEL>
static inline void func_odd(
  const float * pIn, const int64_t inWidth, const int64_t inHeight,
  const float * pGOut, const int64_t gOutWidth, const int64_t gOutHeight,
  float * const pGKernel, const int64_t gKernWidth , const int64_t gKernHeight,
  const int64_t inChannelGroup, const int64_t inChannel, const int64_t gOutChannel,
  const int64_t inGroupOffset, const int64_t outGroupOffset, const int64_t kernGroupOffset,
  const int64_t batch, const int64_t k,
  const int64_t nY,
  const __vr vri,
  const __vr vrj
)
{
  __vr vrin_ptr_ost = _vel_vsfa_vvssl(_vel_vaddul_vvvl(vrj, _vel_vmulul_vsvl(inWidth,vri, nY*gOutWidth), nY*gOutWidth), 2, (uint64_t)(pIn + inGroupOffset), nY*gOutWidth) ;

  __vr vrw_s0 = _vel_vaddsl_vsvl(-1,  vrj, nY*gOutWidth) ;
  __vr vrw_s2 = _vel_vaddsl_vsvl( 1,  vrj, nY*gOutWidth) ;

  __vm256 vmw_s0 =  _vel_vfmklge_mvl(vrw_s0, nY*gOutWidth) ;						// condition(0 <= w)
  __vm256 vmw_s2  = _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(inWidth,vrw_s2, nY*gOutWidth), nY*gOutWidth) ;	// condition(w < inWidth)

  for (int64_t c=0; c<inChannelGroup; c++) {
    const int64_t kernelIndex = kernGroupOffset + ((k     * inChannelGroup + c) * gKernHeight ) * gKernWidth ;

    __vr vrsum0_r0s0  = _vel_vbrds_vsl(0.f, VLEN) ;
    __vr vrsum12_r0s0 = _vel_pvbrd_vsl(0UL, VLEN) ;
    __vr vrsum34_r0s0 = _vel_pvbrd_vsl(0UL, VLEN) ;
    __vr vrsum56_r0s0 = _vel_pvbrd_vsl(0UL, VLEN) ;

    __vr vrsum0_r0s1  = _vel_vbrds_vsl(0.f, VLEN) ;
    __vr vrsum12_r0s1 = _vel_pvbrd_vsl(0UL, VLEN) ;
    __vr vrsum34_r0s1 = _vel_pvbrd_vsl(0UL, VLEN) ;
    __vr vrsum56_r0s1 = _vel_pvbrd_vsl(0UL, VLEN) ;

    __vr vrsum0_r0s2  = _vel_vbrds_vsl(0.f, VLEN) ;
    __vr vrsum12_r0s2 = _vel_pvbrd_vsl(0UL, VLEN) ;
    __vr vrsum34_r0s2 = _vel_pvbrd_vsl(0UL, VLEN) ;
    __vr vrsum56_r0s2 = _vel_pvbrd_vsl(0UL, VLEN) ;

    __vr vrsum0_r1s0  = _vel_vbrds_vsl(0.f, VLEN) ;
    __vr vrsum12_r1s0 = _vel_pvbrd_vsl(0UL, VLEN) ;
    __vr vrsum34_r1s0 = _vel_pvbrd_vsl(0UL, VLEN) ;
    __vr vrsum56_r1s0 = _vel_pvbrd_vsl(0UL, VLEN) ;

    __vr vrsum0_r1s1  = _vel_vbrds_vsl(0.f, VLEN) ;
    __vr vrsum12_r1s1 = _vel_pvbrd_vsl(0UL, VLEN) ;
    __vr vrsum34_r1s1 = _vel_pvbrd_vsl(0UL, VLEN) ;
    __vr vrsum56_r1s1 = _vel_pvbrd_vsl(0UL, VLEN) ;

    __vr vrsum0_r1s2  = _vel_vbrds_vsl(0.f, VLEN) ;
    __vr vrsum12_r1s2 = _vel_pvbrd_vsl(0UL, VLEN) ;
    __vr vrsum34_r1s2 = _vel_pvbrd_vsl(0UL, VLEN) ;
    __vr vrsum56_r1s2 = _vel_pvbrd_vsl(0UL, VLEN) ;

    __vr vrsum0_r2s0  = _vel_vbrds_vsl(0.f, VLEN) ;
    __vr vrsum12_r2s0 = _vel_pvbrd_vsl(0UL, VLEN) ;
    __vr vrsum34_r2s0 = _vel_pvbrd_vsl(0UL, VLEN) ;
    __vr vrsum56_r2s0 = _vel_pvbrd_vsl(0UL, VLEN) ;

    __vr vrsum0_r2s1  = _vel_vbrds_vsl(0.f, VLEN) ;
    __vr vrsum12_r2s1 = _vel_pvbrd_vsl(0UL, VLEN) ;
    __vr vrsum34_r2s1 = _vel_pvbrd_vsl(0UL, VLEN) ;
    __vr vrsum56_r2s1 = _vel_pvbrd_vsl(0UL, VLEN) ;

    __vr vrsum0_r2s2  = _vel_vbrds_vsl(0.f, VLEN) ;
    __vr vrsum12_r2s2 = _vel_pvbrd_vsl(0UL, VLEN) ;
    __vr vrsum34_r2s2 = _vel_pvbrd_vsl(0UL, VLEN) ;
    __vr vrsum56_r2s2 = _vel_pvbrd_vsl(0UL, VLEN) ;

    for (int64_t n=0; n<batch; n++) {
      for (int64_t y=0; y<gOutHeight; y+=nY) {

	const int64_t vl = gOutWidth * (gOutHeight - y < nY ? gOutHeight - y : nY) ;
	const int64_t gop = y * gOutWidth ;

	__vr vrh_r0 = _vel_vaddsl_vsvl(-1+y*2, vri, vl) ;
	__vr vrh_r2 = _vel_vaddsl_vsvl( 1+y*2, vri, vl) ;

	__vm256 vmh_r0 =  _vel_vfmklge_mvl(vrh_r0, vl) ;				// condition(0 <= h)
	__vm256 vmh_r2 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(inHeight,vrh_r2, vl), vl) ;	// condition(h < inHeight)

	__vm256 vmhw_r0s0 = _vel_andm_mmm(vmh_r0, vmw_s0) ;
	__vm256 vmhw_r0s1 = vmh_r0 ;
	__vm256 vmhw_r0s2 = _vel_andm_mmm(vmh_r0, vmw_s2) ;

	__vm256 vmhw_r1s0 = vmw_s0 ;
	__vm256 vmhw_r1s2 = vmw_s2 ;

	__vm256 vmhw_r2s0 = _vel_andm_mmm(vmh_r2, vmw_s0) ;
	__vm256 vmhw_r2s1 = vmh_r2 ;
	__vm256 vmhw_r2s2 = _vel_andm_mmm(vmh_r2, vmw_s2) ;

	__vr vrin_ptr_r0s0 = _vel_vaddsl_vsvl(4*(((n*inChannel+c)*inHeight+2*y-1)*inWidth-1), vrin_ptr_ost, vl) ;
	__vr vrin_r0s0  = _vel_vgtu_vvssml(vrin_ptr_r0s0, 0, 0, vmh_r0, vl) ;
	__vr vrin_ptr_r0s1 = _vel_vaddsl_vsvl(4*(((n*inChannel+c)*inHeight+2*y-1)*inWidth  ), vrin_ptr_ost, vl) ;
	__vr vrin_r0s1  = _vel_vgtu_vvssml(vrin_ptr_r0s1, 0, 0, vmh_r0, vl) ;
	__vr vrin_ptr_r0s2 = _vel_vaddsl_vsvl(4*(((n*inChannel+c)*inHeight+2*y-1)*inWidth+1), vrin_ptr_ost, vl) ;
	__vr vrin_r0s2  = _vel_vgtu_vvssml(vrin_ptr_r0s2, 0, 0, vmh_r0, vl) ;

	__vr vrin_ptr_r1s0 = _vel_vaddsl_vsvl(4*(((n*inChannel+c)*inHeight+2*y  )*inWidth-1), vrin_ptr_ost, vl) ;
	__vr vrin_r1s0  = _vel_vgtu_vvssl(vrin_ptr_r1s0, 0, 0, vl) ;
	__vr vrin_ptr_r1s1 = _vel_vaddsl_vsvl(4*(((n*inChannel+c)*inHeight+2*y  )*inWidth  ), vrin_ptr_ost, vl) ;
	__vr vrin_r1s1  = _vel_vgtu_vvssl(vrin_ptr_r1s1, 0, 0, vl) ;
	__vr vrin_ptr_r1s2 = _vel_vaddsl_vsvl(4*(((n*inChannel+c)*inHeight+2*y  )*inWidth+1), vrin_ptr_ost, vl) ;
	__vr vrin_r1s2  = _vel_vgtu_vvssl(vrin_ptr_r1s2, 0, 0, vl) ;

	__vr vrin_ptr_r2s0 = _vel_vaddsl_vsvl(4*(((n*inChannel+c)*inHeight+2*y+1)*inWidth-1), vrin_ptr_ost, vl) ;
	__vr vrin_r2s0  = _vel_vgtu_vvssml(vrin_ptr_r2s0, 0, 0, vmh_r2, vl) ;
	__vr vrin_ptr_r2s1 = _vel_vaddsl_vsvl(4*(((n*inChannel+c)*inHeight+2*y+1)*inWidth  ), vrin_ptr_ost, vl) ;
	__vr vrin_r2s1  = _vel_vgtu_vvssml(vrin_ptr_r2s1, 0, 0, vmh_r2, vl) ;
	__vr vrin_ptr_r2s2 = _vel_vaddsl_vsvl(4*(((n*inChannel+c)*inHeight+2*y+1)*inWidth+1), vrin_ptr_ost, vl) ;
	__vr vrin_r2s2  = _vel_vgtu_vvssml(vrin_ptr_r2s2, 0, 0, vmh_r2, vl) ;

	__vr vrgout0 = _vel_vldu_vssl(4, pGOut+outGroupOffset + ((n * gOutChannel + k  ) * gOutHeight ) * gOutWidth + gop, vl) ;
	__vr vrgout1 = _vel_vldu_vssl(4, pGOut+outGroupOffset + ((n * gOutChannel + k+1) * gOutHeight ) * gOutWidth + gop, vl) ;
	__vr vrgout2 = _vel_vldu_vssl(4, pGOut+outGroupOffset + ((n * gOutChannel + k+2) * gOutHeight ) * gOutWidth + gop, vl) ;
	__vr vrgout3 = _vel_vldu_vssl(4, pGOut+outGroupOffset + ((n * gOutChannel + k+3) * gOutHeight ) * gOutWidth + gop, vl) ;
	__vr vrgout4 = _vel_vldu_vssl(4, pGOut+outGroupOffset + ((n * gOutChannel + k+4) * gOutHeight ) * gOutWidth + gop, vl) ;
	__vr vrgout5 = _vel_vldu_vssl(4, pGOut+outGroupOffset + ((n * gOutChannel + k+5) * gOutHeight ) * gOutWidth + gop, vl) ;
	__vr vrgout6 = _vel_vldu_vssl(4, pGOut+outGroupOffset + ((n * gOutChannel + k+6) * gOutHeight ) * gOutWidth + gop, vl) ;

	__vr vrgout12 = _vel_vshf_vvvsl(vrgout1, vrgout2, VE_VSHUFFLE_YUZU, vl) ;
	__vr vrgout34 = _vel_vshf_vvvsl(vrgout3, vrgout4, VE_VSHUFFLE_YUZU, vl) ;
	__vr vrgout56 = _vel_vshf_vvvsl(vrgout5, vrgout6, VE_VSHUFFLE_YUZU, vl) ;

#define VFADD(VRIN, VRINP, RS)												\
	{														\
	  vrsum0_##RS = _vel_vfmads_vvvvvl(vrsum0_##RS, VRIN, vrgout0, vrsum0_##RS, vl) ;				\
	  if(NUM_KERNEL >= 3) vrsum12_##RS = _vel_pvfmad_vvvvvl(vrsum12_##RS, VRINP, vrgout12, vrsum12_##RS, vl) ;	\
	  if(NUM_KERNEL >= 5) vrsum34_##RS = _vel_pvfmad_vvvvvl(vrsum34_##RS, VRINP, vrgout34, vrsum34_##RS, vl) ;	\
	  if(NUM_KERNEL >= 7) vrsum56_##RS = _vel_pvfmad_vvvvvl(vrsum56_##RS, VRINP, vrgout56, vrsum56_##RS, vl) ;	\
	}

	vrin_r0s0 = _vel_vmrg_vsvml(0.f, vrin_r0s0, vmhw_r0s0, vl) ;
	__vr vrinP_r0s0 = _vel_vshf_vvvsl(vrin_r0s0, vrin_r0s0, VE_VSHUFFLE_YUZU, vl) ;
	VFADD(vrin_r0s0, vrinP_r0s0, r0s0) ;

	vrin_r0s1 = _vel_vmrg_vsvml(0.f, vrin_r0s1, vmhw_r0s1, vl) ;
	__vr vrinP_r0s1 = _vel_vshf_vvvsl(vrin_r0s1, vrin_r0s1, VE_VSHUFFLE_YUZU, vl) ;
	VFADD(vrin_r0s1, vrinP_r0s1, r0s1) ;

	vrin_r0s2 = _vel_vmrg_vsvml(0.f, vrin_r0s2, vmhw_r0s2, vl) ;
	__vr vrinP_r0s2 = _vel_vshf_vvvsl(vrin_r0s2, vrin_r0s2, VE_VSHUFFLE_YUZU, vl) ;
	VFADD(vrin_r0s2, vrinP_r0s2, r0s2) ;

	vrin_r1s0 = _vel_vmrg_vsvml(0.f, vrin_r1s0, vmhw_r1s0, vl) ;
	__vr vrinP_r1s0 = _vel_vshf_vvvsl(vrin_r1s0, vrin_r1s0, VE_VSHUFFLE_YUZU, vl) ;
	VFADD(vrin_r1s0, vrinP_r1s0, r1s0) ;

	// need not to use mask
	__vr vrinP_r1s1 = _vel_vshf_vvvsl(vrin_r1s1, vrin_r1s1, VE_VSHUFFLE_YUZU, vl) ;
	VFADD(vrin_r1s1, vrinP_r1s1, r1s1) ;

	vrin_r1s2 = _vel_vmrg_vsvml(0.f, vrin_r1s2, vmhw_r1s2, vl) ;
	__vr vrinP_r1s2 = _vel_vshf_vvvsl(vrin_r1s2, vrin_r1s2, VE_VSHUFFLE_YUZU, vl) ;
	VFADD(vrin_r1s2, vrinP_r1s2, r1s2) ;

	vrin_r2s0 = _vel_vmrg_vsvml(0.f, vrin_r2s0, vmhw_r2s0, vl) ;
	__vr vrinP_r2s0 = _vel_vshf_vvvsl(vrin_r2s0, vrin_r2s0, VE_VSHUFFLE_YUZU, vl) ;
	VFADD(vrin_r2s0, vrinP_r2s0, r2s0) ;

	vrin_r2s1 = _vel_vmrg_vsvml(0.f, vrin_r2s1, vmhw_r2s1, vl) ;
	__vr vrinP_r2s1 = _vel_vshf_vvvsl(vrin_r2s1, vrin_r2s1, VE_VSHUFFLE_YUZU, vl) ;
	VFADD(vrin_r2s1, vrinP_r2s1, r2s1) ;

	vrin_r2s2 = _vel_vmrg_vsvml(0.f, vrin_r2s2, vmhw_r2s2, vl) ;
	__vr vrinP_r2s2 = _vel_vshf_vvvsl(vrin_r2s2, vrin_r2s2, VE_VSHUFFLE_YUZU, vl) ;
	VFADD(vrin_r2s2, vrinP_r2s2, r2s2) ;

#undef VFADD
      } // gOutPixels
    } // batch

#define SUM_AND_STORE(VRSUM_R0S0, VRSUM_R0S1, VRSUM_R0S2, VRSUM_R1S0, VRSUM_R1S1, VRSUM_R1S2, VRSUM_R2S0, VRSUM_R2S1, VRSUM_R2S2, K)		\
    {																		\
      __vr vrsum0_r0s0 = _vel_vfsums_vvl(VRSUM_R0S0, VLEN) ;											\
      __vr vrsum1_r0s0 = _vel_vfsums_vvl(_vel_vsll_vvsl(VRSUM_R0S0, 32, VLEN), VLEN);								\
      _vel_vstu_vssl(vrsum0_r0s0, 4, pGKernel+kernelIndex + ((K)  ) * inChannelGroup * gKernHeight * gKernWidth + 0 * gKernWidth + 0, 1) ;	\
      _vel_vstu_vssl(vrsum1_r0s0, 4, pGKernel+kernelIndex + ((K)+1) * inChannelGroup * gKernHeight * gKernWidth + 0 * gKernWidth + 0, 1) ;	\
      __vr vrsum0_r0s1 = _vel_vfsums_vvl(VRSUM_R0S1, VLEN) ;											\
      __vr vrsum1_r0s1 = _vel_vfsums_vvl(_vel_vsll_vvsl(VRSUM_R0S1, 32, VLEN), VLEN);								\
      _vel_vstu_vssl(vrsum0_r0s1, 4, pGKernel+kernelIndex + ((K)  ) * inChannelGroup * gKernHeight * gKernWidth + 0 * gKernWidth + 1, 1) ;	\
      _vel_vstu_vssl(vrsum1_r0s1, 4, pGKernel+kernelIndex + ((K)+1) * inChannelGroup * gKernHeight * gKernWidth + 0 * gKernWidth + 1, 1) ;	\
      __vr vrsum0_r0s2 = _vel_vfsums_vvl(VRSUM_R0S2, VLEN) ;											\
      __vr vrsum1_r0s2 = _vel_vfsums_vvl(_vel_vsll_vvsl(VRSUM_R0S2, 32, VLEN), VLEN);								\
      _vel_vstu_vssl(vrsum0_r0s2, 4, pGKernel+kernelIndex + ((K)  ) * inChannelGroup * gKernHeight * gKernWidth + 0 * gKernWidth + 2, 1) ;	\
      _vel_vstu_vssl(vrsum1_r0s2, 4, pGKernel+kernelIndex + ((K)+1) * inChannelGroup * gKernHeight * gKernWidth + 0 * gKernWidth + 2, 1) ;	\
      __vr vrsum0_r1s0 = _vel_vfsums_vvl(VRSUM_R1S0, VLEN) ;											\
      __vr vrsum1_r1s0 = _vel_vfsums_vvl(_vel_vsll_vvsl(VRSUM_R1S0, 32, VLEN), VLEN);								\
      _vel_vstu_vssl(vrsum0_r1s0, 4, pGKernel+kernelIndex + ((K)  ) * inChannelGroup * gKernHeight * gKernWidth + 1 * gKernWidth + 0, 1) ;	\
      _vel_vstu_vssl(vrsum1_r1s0, 4, pGKernel+kernelIndex + ((K)+1) * inChannelGroup * gKernHeight * gKernWidth + 1 * gKernWidth + 0, 1) ;	\
      __vr vrsum0_r1s1 = _vel_vfsums_vvl(VRSUM_R1S1, VLEN) ;											\
      __vr vrsum1_r1s1 = _vel_vfsums_vvl(_vel_vsll_vvsl(VRSUM_R1S1, 32, VLEN), VLEN);								\
      _vel_vstu_vssl(vrsum0_r1s1, 4, pGKernel+kernelIndex + ((K)  ) * inChannelGroup * gKernHeight * gKernWidth + 1 * gKernWidth + 1, 1) ;	\
      _vel_vstu_vssl(vrsum1_r1s1, 4, pGKernel+kernelIndex + ((K)+1) * inChannelGroup * gKernHeight * gKernWidth + 1 * gKernWidth + 1, 1) ;	\
      __vr vrsum0_r1s2 = _vel_vfsums_vvl(VRSUM_R1S2, VLEN) ;											\
      __vr vrsum1_r1s2 = _vel_vfsums_vvl(_vel_vsll_vvsl(VRSUM_R1S2, 32, VLEN), VLEN);								\
      _vel_vstu_vssl(vrsum0_r1s2, 4, pGKernel+kernelIndex + ((K)  ) * inChannelGroup * gKernHeight * gKernWidth + 1 * gKernWidth + 2, 1) ;	\
      _vel_vstu_vssl(vrsum1_r1s2, 4, pGKernel+kernelIndex + ((K)+1) * inChannelGroup * gKernHeight * gKernWidth + 1 * gKernWidth + 2, 1) ;	\
      __vr vrsum0_r2s0 = _vel_vfsums_vvl(VRSUM_R2S0, VLEN) ;											\
      __vr vrsum1_r2s0 = _vel_vfsums_vvl(_vel_vsll_vvsl(VRSUM_R2S0, 32, VLEN), VLEN);								\
      _vel_vstu_vssl(vrsum0_r2s0, 4, pGKernel+kernelIndex + ((K)  ) * inChannelGroup * gKernHeight * gKernWidth + 2 * gKernWidth + 0, 1) ;	\
      _vel_vstu_vssl(vrsum1_r2s0, 4, pGKernel+kernelIndex + ((K)+1) * inChannelGroup * gKernHeight * gKernWidth + 2 * gKernWidth + 0, 1) ;	\
      __vr vrsum0_r2s1 = _vel_vfsums_vvl(VRSUM_R2S1, VLEN) ;											\
      __vr vrsum1_r2s1 = _vel_vfsums_vvl(_vel_vsll_vvsl(VRSUM_R2S1, 32, VLEN), VLEN);								\
      _vel_vstu_vssl(vrsum0_r2s1, 4, pGKernel+kernelIndex + ((K)  ) * inChannelGroup * gKernHeight * gKernWidth + 2 * gKernWidth + 1, 1) ;	\
      _vel_vstu_vssl(vrsum1_r2s1, 4, pGKernel+kernelIndex + ((K)+1) * inChannelGroup * gKernHeight * gKernWidth + 2 * gKernWidth + 1, 1) ;	\
      __vr vrsum0_r2s2 = _vel_vfsums_vvl(VRSUM_R2S2, VLEN) ;											\
      __vr vrsum1_r2s2 = _vel_vfsums_vvl(_vel_vsll_vvsl(VRSUM_R2S2, 32, VLEN), VLEN);								\
      _vel_vstu_vssl(vrsum0_r2s2, 4, pGKernel+kernelIndex + ((K)  ) * inChannelGroup * gKernHeight * gKernWidth + 2 * gKernWidth + 2, 1) ;	\
      _vel_vstu_vssl(vrsum1_r2s2, 4, pGKernel+kernelIndex + ((K)+1) * inChannelGroup * gKernHeight * gKernWidth + 2 * gKernWidth + 2, 1) ;	\
    }

    {											\
      vrsum0_r0s0 = _vel_vfsums_vvl(vrsum0_r0s0, VLEN) ;				\
      _vel_vstu_vssl(vrsum0_r0s0, 4, pGKernel+kernelIndex + 0 * gKernWidth + 0, 1) ;	\
      vrsum0_r0s1 = _vel_vfsums_vvl(vrsum0_r0s1, VLEN) ;				\
      _vel_vstu_vssl(vrsum0_r0s1, 4, pGKernel+kernelIndex + 0 * gKernWidth + 1, 1) ;	\
      vrsum0_r0s2 = _vel_vfsums_vvl(vrsum0_r0s2, VLEN) ;				\
      _vel_vstu_vssl(vrsum0_r0s2, 4, pGKernel+kernelIndex + 0 * gKernWidth + 2, 1) ;	\
      vrsum0_r1s0 = _vel_vfsums_vvl(vrsum0_r1s0, VLEN) ;				\
      _vel_vstu_vssl(vrsum0_r1s0, 4, pGKernel+kernelIndex + 1 * gKernWidth + 0, 1) ;	\
      vrsum0_r1s1 = _vel_vfsums_vvl(vrsum0_r1s1, VLEN) ;				\
      _vel_vstu_vssl(vrsum0_r1s1, 4, pGKernel+kernelIndex + 1 * gKernWidth + 1, 1) ;	\
      vrsum0_r1s2 = _vel_vfsums_vvl(vrsum0_r1s2, VLEN) ;				\
      _vel_vstu_vssl(vrsum0_r1s2, 4, pGKernel+kernelIndex + 1 * gKernWidth + 2, 1) ;	\
      vrsum0_r2s0 = _vel_vfsums_vvl(vrsum0_r2s0, VLEN) ;				\
      _vel_vstu_vssl(vrsum0_r2s0, 4, pGKernel+kernelIndex + 2 * gKernWidth + 0, 1) ;	\
      vrsum0_r2s1 = _vel_vfsums_vvl(vrsum0_r2s1, VLEN) ;				\
      _vel_vstu_vssl(vrsum0_r2s1, 4, pGKernel+kernelIndex + 2 * gKernWidth + 1, 1) ;	\
      vrsum0_r2s2 = _vel_vfsums_vvl(vrsum0_r2s2, VLEN) ;				\
      _vel_vstu_vssl(vrsum0_r2s2, 4, pGKernel+kernelIndex + 2 * gKernWidth + 2, 1) ;	\
    }

    if(NUM_KERNEL >= 3) SUM_AND_STORE(vrsum12_r0s0, vrsum12_r0s1, vrsum12_r0s2, vrsum12_r1s0, vrsum12_r1s1, vrsum12_r1s2, vrsum12_r2s0, vrsum12_r2s1, vrsum12_r2s2, 1) ;
    if(NUM_KERNEL >= 5) SUM_AND_STORE(vrsum34_r0s0, vrsum34_r0s1, vrsum34_r0s2, vrsum34_r1s0, vrsum34_r1s1, vrsum34_r1s2, vrsum34_r2s0, vrsum34_r2s1, vrsum34_r2s2, 3) ;
    if(NUM_KERNEL >= 7) SUM_AND_STORE(vrsum56_r0s0, vrsum56_r0s1, vrsum56_r0s2, vrsum56_r1s0, vrsum56_r1s1, vrsum56_r1s2, vrsum56_r2s0, vrsum56_r2s1, vrsum56_r2s2, 5) ;
#undef SUM_AND_STORE

  } // inChannel
}

template <int NUM_KERNEL>
static inline void func_odd_ialigned(
  const float * pIn,
  const int64_t inWidth,
  const int64_t inHeight,
  const float * pGOut,
  const int64_t gOutWidth,
  const int64_t gOutHeight,
  float * const pGKernel,
  const int64_t gKernWidth,
  const int64_t gKernHeight,
  const int64_t inChannelGroup,
  const int64_t inChannel,
  const int64_t gOutChannel,
  const int64_t inGroupOffset,
  const int64_t outGroupOffset,
  const int64_t kernGroupOffset,
  const int64_t batch,
  const int64_t k,
  const int64_t nY,
  const __vr vri,
  const __vr vrj
)
{
  __vr vrin_ptr_ost = _vel_vsfa_vvssl(_vel_vaddul_vvvl(vrj, _vel_vmulul_vsvl(inWidth,vri, nY*gOutWidth), nY*gOutWidth), 2, (uint64_t)(pIn + inGroupOffset), nY*gOutWidth) ;

  __vr vrw_s0 = _vel_vaddsl_vsvl(-1,  vrj, nY*gOutWidth) ;
  __vm256 vmw_s0 =  _vel_vfmklge_mvl(vrw_s0, nY*gOutWidth) ;						// condition(0 <= w)

  for (int64_t c=0; c<inChannelGroup; c++) {
    const int64_t kernelIndex = kernGroupOffset + ((k     * inChannelGroup + c) * gKernHeight ) * gKernWidth ;

    __vr vrsum0_r0s0  = _vel_vbrds_vsl(0.f, VLEN) ;
    __vr vrsum12_r0s0 = _vel_pvbrd_vsl(0UL, VLEN) ;
    __vr vrsum34_r0s0 = _vel_pvbrd_vsl(0UL, VLEN) ;
    __vr vrsum56_r0s0 = _vel_pvbrd_vsl(0UL, VLEN) ;
    __vr vrsum78_r0s0 = _vel_pvbrd_vsl(0UL, VLEN) ;

    __vr vrsum0_r0s1  = _vel_vbrds_vsl(0.f, VLEN) ;
    __vr vrsum12_r0s1 = _vel_pvbrd_vsl(0UL, VLEN) ;
    __vr vrsum34_r0s1 = _vel_pvbrd_vsl(0UL, VLEN) ;
    __vr vrsum56_r0s1 = _vel_pvbrd_vsl(0UL, VLEN) ;
    __vr vrsum78_r0s1 = _vel_pvbrd_vsl(0UL, VLEN) ;

    __vr vrsum0_r0s2  = _vel_vbrds_vsl(0.f, VLEN) ;
    __vr vrsum12_r0s2 = _vel_pvbrd_vsl(0UL, VLEN) ;
    __vr vrsum34_r0s2 = _vel_pvbrd_vsl(0UL, VLEN) ;
    __vr vrsum56_r0s2 = _vel_pvbrd_vsl(0UL, VLEN) ;
    __vr vrsum78_r0s2 = _vel_pvbrd_vsl(0UL, VLEN) ;

    __vr vrsum0_r1s0  = _vel_vbrds_vsl(0.f, VLEN) ;
    __vr vrsum12_r1s0 = _vel_pvbrd_vsl(0UL, VLEN) ;
    __vr vrsum34_r1s0 = _vel_pvbrd_vsl(0UL, VLEN) ;
    __vr vrsum56_r1s0 = _vel_pvbrd_vsl(0UL, VLEN) ;
    __vr vrsum78_r1s0 = _vel_pvbrd_vsl(0UL, VLEN) ;

    __vr vrsum0_r1s1  = _vel_vbrds_vsl(0.f, VLEN) ;
    __vr vrsum12_r1s1 = _vel_pvbrd_vsl(0UL, VLEN) ;
    __vr vrsum34_r1s1 = _vel_pvbrd_vsl(0UL, VLEN) ;
    __vr vrsum56_r1s1 = _vel_pvbrd_vsl(0UL, VLEN) ;
    __vr vrsum78_r1s1 = _vel_pvbrd_vsl(0UL, VLEN) ;

    __vr vrsum0_r1s2  = _vel_vbrds_vsl(0.f, VLEN) ;
    __vr vrsum12_r1s2 = _vel_pvbrd_vsl(0UL, VLEN) ;
    __vr vrsum34_r1s2 = _vel_pvbrd_vsl(0UL, VLEN) ;
    __vr vrsum56_r1s2 = _vel_pvbrd_vsl(0UL, VLEN) ;
    __vr vrsum78_r1s2 = _vel_pvbrd_vsl(0UL, VLEN) ;

    __vr vrsum0_r2s0  = _vel_vbrds_vsl(0.f, VLEN) ;
    __vr vrsum12_r2s0 = _vel_pvbrd_vsl(0UL, VLEN) ;
    __vr vrsum34_r2s0 = _vel_pvbrd_vsl(0UL, VLEN) ;
    __vr vrsum56_r2s0 = _vel_pvbrd_vsl(0UL, VLEN) ;
    __vr vrsum78_r2s0 = _vel_pvbrd_vsl(0UL, VLEN) ;

    __vr vrsum0_r2s1  = _vel_vbrds_vsl(0.f, VLEN) ;
    __vr vrsum12_r2s1 = _vel_pvbrd_vsl(0UL, VLEN) ;
    __vr vrsum34_r2s1 = _vel_pvbrd_vsl(0UL, VLEN) ;
    __vr vrsum56_r2s1 = _vel_pvbrd_vsl(0UL, VLEN) ;
    __vr vrsum78_r2s1 = _vel_pvbrd_vsl(0UL, VLEN) ;

    __vr vrsum0_r2s2  = _vel_vbrds_vsl(0.f, VLEN) ;
    __vr vrsum12_r2s2 = _vel_pvbrd_vsl(0UL, VLEN) ;
    __vr vrsum34_r2s2 = _vel_pvbrd_vsl(0UL, VLEN) ;
    __vr vrsum56_r2s2 = _vel_pvbrd_vsl(0UL, VLEN) ;
    __vr vrsum78_r2s2 = _vel_pvbrd_vsl(0UL, VLEN) ;

    for (int64_t n=0; n<batch; n++) {
      for (int64_t y=0; y<gOutHeight; y+=nY) {

	const int64_t vl = gOutWidth * (gOutHeight - y < nY ? gOutHeight - y : nY) ;
	const int64_t gop = y * gOutWidth ;

	__vr vrh_r0 = _vel_vaddsl_vsvl(-1+y*2, vri, vl) ;
	__vr vrh_r2 = _vel_vaddsl_vsvl( 1+y*2, vri, vl) ;

	__vm256 vmh_r0 =  _vel_vfmklge_mvl(vrh_r0, vl) ;				// condition(0 <= h)
	__vm256 vmh_r2 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(inHeight,vrh_r2, vl), vl) ;	// condition(h < inHeight)

	__vr vrin_ptr_r0 = _vel_vaddsl_vsvl(4*(((n*inChannel+c)*inHeight+2*y-1)*inWidth), vrin_ptr_ost, vl) ;
	__vr vrin_r0  = _vel_vgt_vvssml(vrin_ptr_r0, 0, 0, vmh_r0, vl) ;

	__vr vrin_ptr_r1 = _vel_vaddsl_vsvl(4*(((n*inChannel+c)*inHeight+2*y  )*inWidth), vrin_ptr_ost, vl) ;
	__vr vrin_r1  = _vel_vgt_vvssl(vrin_ptr_r1, 0, 0, vl) ;

	__vr vrin_ptr_r2 = _vel_vaddsl_vsvl(4*(((n*inChannel+c)*inHeight+2*y+1)*inWidth), vrin_ptr_ost, vl) ;
	__vr vrin_r2  = _vel_vgt_vvssml(vrin_ptr_r2, 0, 0, vmh_r2, vl) ;

	__vr vrgout0 = _vel_vldu_vssl(4, pGOut+outGroupOffset + ((n * gOutChannel + k  ) * gOutHeight ) * gOutWidth + gop, vl) ;
	__vr vrgout1 = _vel_vldu_vssl(4, pGOut+outGroupOffset + ((n * gOutChannel + k+1) * gOutHeight ) * gOutWidth + gop, vl) ;
	__vr vrgout2 = _vel_vldu_vssl(4, pGOut+outGroupOffset + ((n * gOutChannel + k+2) * gOutHeight ) * gOutWidth + gop, vl) ;
	__vr vrgout3 = _vel_vldu_vssl(4, pGOut+outGroupOffset + ((n * gOutChannel + k+3) * gOutHeight ) * gOutWidth + gop, vl) ;
	__vr vrgout4 = _vel_vldu_vssl(4, pGOut+outGroupOffset + ((n * gOutChannel + k+4) * gOutHeight ) * gOutWidth + gop, vl) ;
	__vr vrgout5 = _vel_vldu_vssl(4, pGOut+outGroupOffset + ((n * gOutChannel + k+5) * gOutHeight ) * gOutWidth + gop, vl) ;
	__vr vrgout6 = _vel_vldu_vssl(4, pGOut+outGroupOffset + ((n * gOutChannel + k+6) * gOutHeight ) * gOutWidth + gop, vl) ;
	__vr vrgout7 = _vel_vldu_vssl(4, pGOut+outGroupOffset + ((n * gOutChannel + k+7) * gOutHeight ) * gOutWidth + gop, vl) ;
	__vr vrgout8 = _vel_vldu_vssl(4, pGOut+outGroupOffset + ((n * gOutChannel + k+8) * gOutHeight ) * gOutWidth + gop, vl) ;
	__vr vrgout9 = _vel_vldu_vssl(4, pGOut+outGroupOffset + ((n * gOutChannel + k+9) * gOutHeight ) * gOutWidth + gop, vl) ;

	__vr vrgout12 = _vel_vshf_vvvsl(vrgout1, vrgout2, VE_VSHUFFLE_YUZU, vl) ;
	__vr vrgout34 = _vel_vshf_vvvsl(vrgout3, vrgout4, VE_VSHUFFLE_YUZU, vl) ;
	__vr vrgout56 = _vel_vshf_vvvsl(vrgout5, vrgout6, VE_VSHUFFLE_YUZU, vl) ;
	__vr vrgout78 = _vel_vshf_vvvsl(vrgout7, vrgout8, VE_VSHUFFLE_YUZU, vl) ;

#define VFADD(VRINP, RS)												\
	{														\
	  vrsum0_##RS = _vel_vfmads_vvvvvl(vrsum0_##RS, VRINP, vrgout0, vrsum0_##RS, vl) ;				\
	  if(NUM_KERNEL >= 3) vrsum12_##RS = _vel_pvfmad_vvvvvl(vrsum12_##RS, VRINP, vrgout12, vrsum12_##RS, vl) ;	\
	  if(NUM_KERNEL >= 5) vrsum34_##RS = _vel_pvfmad_vvvvvl(vrsum34_##RS, VRINP, vrgout34, vrsum34_##RS, vl) ;	\
	  if(NUM_KERNEL >= 7) vrsum56_##RS = _vel_pvfmad_vvvvvl(vrsum56_##RS, VRINP, vrgout56, vrsum56_##RS, vl) ;	\
	  if(NUM_KERNEL >= 9) vrsum78_##RS = _vel_pvfmad_vvvvvl(vrsum56_##RS, VRINP, vrgout78, vrsum78_##RS, vl) ;	\
	}

	vrin_r0 = _vel_vmrg_vsvml(0.f, vrin_r0, vmh_r0, vl) ;
	__vr vrinP_r0s2 = _vel_vshf_vvvsl(vrin_r0, vrin_r0, VE_VSHUFFLE_YUZU, vl) ;
	__vr vrinP_r0s1 = _vel_vshf_vvvsl(vrin_r0, vrin_r0, VE_VSHUFFLE_YLZL, vl) ;
	__vr vrinP_r0s0 = _vel_vmv_vsvl(-1, vrinP_r0s2, vl) ;
	VFADD(vrinP_r0s2, r0s2) ;
	VFADD(vrinP_r0s1, r0s1) ;
	VFADD(vrinP_r0s0, r0s0) ;

	// need not to use mask
	__vr vrinP_r1s2 = _vel_vshf_vvvsl(vrin_r1, vrin_r1, VE_VSHUFFLE_YUZU, vl) ;
	__vr vrinP_r1s1 = _vel_vshf_vvvsl(vrin_r1, vrin_r1, VE_VSHUFFLE_YLZL, vl) ;
	__vr vrinP_r1s0 = _vel_vmv_vsvl(-1, vrinP_r1s2, vl) ;
	VFADD(vrinP_r1s2, r1s2) ;
	VFADD(vrinP_r1s1, r1s1) ;
	VFADD(vrinP_r1s0, r1s0) ;

	vrin_r2 = _vel_vmrg_vsvml(0.f, vrin_r2, vmh_r2, vl) ;
	__vr vrinP_r2s2 = _vel_vshf_vvvsl(vrin_r2, vrin_r2, VE_VSHUFFLE_YUZU, vl) ;
	__vr vrinP_r2s1 = _vel_vshf_vvvsl(vrin_r2, vrin_r2, VE_VSHUFFLE_YLZL, vl) ;
	__vr vrinP_r2s0 = _vel_vmv_vsvl(-1, vrinP_r2s2, vl) ;
	VFADD(vrinP_r2s2, r2s2) ;
	VFADD(vrinP_r2s1, r2s1) ;
	VFADD(vrinP_r2s0, r2s0) ;

#undef VFADD
      } // gOutPixels
    } // batch

#define SUM_AND_STORE(VRSUM_R0S0, VRSUM_R0S1, VRSUM_R0S2, VRSUM_R1S0, VRSUM_R1S1, VRSUM_R1S2, VRSUM_R2S0, VRSUM_R2S1, VRSUM_R2S2, K)		\
    {																		\
      __vr vrsum0_r0s0 = _vel_vfsums_vvml(VRSUM_R0S0, vmw_s0, VLEN) ;										\
      __vr vrsum1_r0s0 = _vel_vfsums_vvml(_vel_vsll_vvsl(VRSUM_R0S0, 32, VLEN), vmw_s0, VLEN);							\
      _vel_vstu_vssl(vrsum0_r0s0, 4, pGKernel+kernelIndex + ((K)  ) * inChannelGroup * gKernHeight * gKernWidth + 0 * gKernWidth + 0, 1) ;	\
      _vel_vstu_vssl(vrsum1_r0s0, 4, pGKernel+kernelIndex + ((K)+1) * inChannelGroup * gKernHeight * gKernWidth + 0 * gKernWidth + 0, 1) ;	\
      __vr vrsum0_r0s1 = _vel_vfsums_vvl(VRSUM_R0S1, VLEN) ;											\
      __vr vrsum1_r0s1 = _vel_vfsums_vvl(_vel_vsll_vvsl(VRSUM_R0S1, 32, VLEN), VLEN);								\
      _vel_vstu_vssl(vrsum0_r0s1, 4, pGKernel+kernelIndex + ((K)  ) * inChannelGroup * gKernHeight * gKernWidth + 0 * gKernWidth + 1, 1) ;	\
      _vel_vstu_vssl(vrsum1_r0s1, 4, pGKernel+kernelIndex + ((K)+1) * inChannelGroup * gKernHeight * gKernWidth + 0 * gKernWidth + 1, 1) ;	\
      __vr vrsum0_r0s2 = _vel_vfsums_vvl(VRSUM_R0S2, VLEN) ;											\
      __vr vrsum1_r0s2 = _vel_vfsums_vvl(_vel_vsll_vvsl(VRSUM_R0S2, 32, VLEN), VLEN);								\
      _vel_vstu_vssl(vrsum0_r0s2, 4, pGKernel+kernelIndex + ((K)  ) * inChannelGroup * gKernHeight * gKernWidth + 0 * gKernWidth + 2, 1) ;	\
      _vel_vstu_vssl(vrsum1_r0s2, 4, pGKernel+kernelIndex + ((K)+1) * inChannelGroup * gKernHeight * gKernWidth + 0 * gKernWidth + 2, 1) ;	\
      __vr vrsum0_r1s0 = _vel_vfsums_vvml(VRSUM_R1S0, vmw_s0, VLEN) ;										\
      __vr vrsum1_r1s0 = _vel_vfsums_vvml(_vel_vsll_vvsl(VRSUM_R1S0, 32, VLEN), vmw_s0, VLEN);							\
      _vel_vstu_vssl(vrsum0_r1s0, 4, pGKernel+kernelIndex + ((K)  ) * inChannelGroup * gKernHeight * gKernWidth + 1 * gKernWidth + 0, 1) ;	\
      _vel_vstu_vssl(vrsum1_r1s0, 4, pGKernel+kernelIndex + ((K)+1) * inChannelGroup * gKernHeight * gKernWidth + 1 * gKernWidth + 0, 1) ;	\
      __vr vrsum0_r1s1 = _vel_vfsums_vvl(VRSUM_R1S1, VLEN) ;											\
      __vr vrsum1_r1s1 = _vel_vfsums_vvl(_vel_vsll_vvsl(VRSUM_R1S1, 32, VLEN), VLEN);								\
      _vel_vstu_vssl(vrsum0_r1s1, 4, pGKernel+kernelIndex + ((K)  ) * inChannelGroup * gKernHeight * gKernWidth + 1 * gKernWidth + 1, 1) ;	\
      _vel_vstu_vssl(vrsum1_r1s1, 4, pGKernel+kernelIndex + ((K)+1) * inChannelGroup * gKernHeight * gKernWidth + 1 * gKernWidth + 1, 1) ;	\
      __vr vrsum0_r1s2 = _vel_vfsums_vvl(VRSUM_R1S2, VLEN) ;											\
      __vr vrsum1_r1s2 = _vel_vfsums_vvl(_vel_vsll_vvsl(VRSUM_R1S2, 32, VLEN), VLEN);								\
      _vel_vstu_vssl(vrsum0_r1s2, 4, pGKernel+kernelIndex + ((K)  ) * inChannelGroup * gKernHeight * gKernWidth + 1 * gKernWidth + 2, 1) ;	\
      _vel_vstu_vssl(vrsum1_r1s2, 4, pGKernel+kernelIndex + ((K)+1) * inChannelGroup * gKernHeight * gKernWidth + 1 * gKernWidth + 2, 1) ;	\
      __vr vrsum0_r2s0 = _vel_vfsums_vvml(VRSUM_R2S0, vmw_s0, VLEN) ;										\
      __vr vrsum1_r2s0 = _vel_vfsums_vvml(_vel_vsll_vvsl(VRSUM_R2S0, 32, VLEN), vmw_s0, VLEN);							\
      _vel_vstu_vssl(vrsum0_r2s0, 4, pGKernel+kernelIndex + ((K)  ) * inChannelGroup * gKernHeight * gKernWidth + 2 * gKernWidth + 0, 1) ;	\
      _vel_vstu_vssl(vrsum1_r2s0, 4, pGKernel+kernelIndex + ((K)+1) * inChannelGroup * gKernHeight * gKernWidth + 2 * gKernWidth + 0, 1) ;	\
      __vr vrsum0_r2s1 = _vel_vfsums_vvl(VRSUM_R2S1, VLEN) ;											\
      __vr vrsum1_r2s1 = _vel_vfsums_vvl(_vel_vsll_vvsl(VRSUM_R2S1, 32, VLEN), VLEN);								\
      _vel_vstu_vssl(vrsum0_r2s1, 4, pGKernel+kernelIndex + ((K)  ) * inChannelGroup * gKernHeight * gKernWidth + 2 * gKernWidth + 1, 1) ;	\
      _vel_vstu_vssl(vrsum1_r2s1, 4, pGKernel+kernelIndex + ((K)+1) * inChannelGroup * gKernHeight * gKernWidth + 2 * gKernWidth + 1, 1) ;	\
      __vr vrsum0_r2s2 = _vel_vfsums_vvl(VRSUM_R2S2, VLEN) ;											\
      __vr vrsum1_r2s2 = _vel_vfsums_vvl(_vel_vsll_vvsl(VRSUM_R2S2, 32, VLEN), VLEN);								\
      _vel_vstu_vssl(vrsum0_r2s2, 4, pGKernel+kernelIndex + ((K)  ) * inChannelGroup * gKernHeight * gKernWidth + 2 * gKernWidth + 2, 1) ;	\
      _vel_vstu_vssl(vrsum1_r2s2, 4, pGKernel+kernelIndex + ((K)+1) * inChannelGroup * gKernHeight * gKernWidth + 2 * gKernWidth + 2, 1) ;	\
    }

    {											\
      vrsum0_r0s0 = _vel_vfsums_vvml(vrsum0_r0s0, vmw_s0, VLEN) ;			\
      _vel_vstu_vssl(vrsum0_r0s0, 4, pGKernel+kernelIndex + 0 * gKernWidth + 0, 1) ;	\
      vrsum0_r0s1 = _vel_vfsums_vvl(vrsum0_r0s1, VLEN) ;				\
      _vel_vstu_vssl(vrsum0_r0s1, 4, pGKernel+kernelIndex + 0 * gKernWidth + 1, 1) ;	\
      vrsum0_r0s2 = _vel_vfsums_vvl(vrsum0_r0s2, VLEN) ;				\
      _vel_vstu_vssl(vrsum0_r0s2, 4, pGKernel+kernelIndex + 0 * gKernWidth + 2, 1) ;	\
      vrsum0_r1s0 = _vel_vfsums_vvml(vrsum0_r1s0, vmw_s0, VLEN) ;			\
      _vel_vstu_vssl(vrsum0_r1s0, 4, pGKernel+kernelIndex + 1 * gKernWidth + 0, 1) ;	\
      vrsum0_r1s1 = _vel_vfsums_vvl(vrsum0_r1s1, VLEN) ;				\
      _vel_vstu_vssl(vrsum0_r1s1, 4, pGKernel+kernelIndex + 1 * gKernWidth + 1, 1) ;	\
      vrsum0_r1s2 = _vel_vfsums_vvl(vrsum0_r1s2, VLEN) ;				\
      _vel_vstu_vssl(vrsum0_r1s2, 4, pGKernel+kernelIndex + 1 * gKernWidth + 2, 1) ;	\
      vrsum0_r2s0 = _vel_vfsums_vvml(vrsum0_r2s0, vmw_s0, VLEN) ;			\
      _vel_vstu_vssl(vrsum0_r2s0, 4, pGKernel+kernelIndex + 2 * gKernWidth + 0, 1) ;	\
      vrsum0_r2s1 = _vel_vfsums_vvl(vrsum0_r2s1, VLEN) ;				\
      _vel_vstu_vssl(vrsum0_r2s1, 4, pGKernel+kernelIndex + 2 * gKernWidth + 1, 1) ;	\
      vrsum0_r2s2 = _vel_vfsums_vvl(vrsum0_r2s2, VLEN) ;				\
      _vel_vstu_vssl(vrsum0_r2s2, 4, pGKernel+kernelIndex + 2 * gKernWidth + 2, 1) ;	\
    }
    if(NUM_KERNEL >= 3) SUM_AND_STORE(vrsum12_r0s0, vrsum12_r0s1, vrsum12_r0s2, vrsum12_r1s0, vrsum12_r1s1, vrsum12_r1s2, vrsum12_r2s0, vrsum12_r2s1, vrsum12_r2s2, 1) ;
    if(NUM_KERNEL >= 5) SUM_AND_STORE(vrsum34_r0s0, vrsum34_r0s1, vrsum34_r0s2, vrsum34_r1s0, vrsum34_r1s1, vrsum34_r1s2, vrsum34_r2s0, vrsum34_r2s1, vrsum34_r2s2, 3) ;
    if(NUM_KERNEL >= 7) SUM_AND_STORE(vrsum56_r0s0, vrsum56_r0s1, vrsum56_r0s2, vrsum56_r1s0, vrsum56_r1s1, vrsum56_r1s2, vrsum56_r2s0, vrsum56_r2s1, vrsum56_r2s2, 5) ;
    if(NUM_KERNEL >= 9) SUM_AND_STORE(vrsum78_r0s0, vrsum78_r0s1, vrsum78_r0s2, vrsum78_r1s0, vrsum78_r1s1, vrsum78_r1s2, vrsum78_r2s0, vrsum78_r2s1, vrsum78_r2s2, 7) ;

#undef SUM_AND_STORE


  } // inChannel
}

template <int NUM_KERNEL>
static inline void func_even(
  const float * pIn, const int64_t inWidth, const int64_t inHeight,
  const float * pGOut, const int64_t gOutWidth, const int64_t gOutHeight,
  float * const pGKernel, const int64_t gKernWidth , const int64_t gKernHeight,
  const int64_t inChannelGroup, const int64_t inChannel, const int64_t gOutChannel,
  const int64_t inGroupOffset, const int64_t outGroupOffset, const int64_t kernGroupOffset,
  const int64_t batch, const int64_t k,
  const int64_t nY,
  const __vr vri,
  const __vr vrj
)
{
  __vr vrin_ptr_ost = _vel_vsfa_vvssl(_vel_vaddul_vvvl(vrj, _vel_vmulul_vsvl(inWidth,vri, nY*gOutWidth), nY*gOutWidth), 2, (uint64_t)(pIn + inGroupOffset), nY*gOutWidth) ;

  __vr vrw_s0 = _vel_vaddsl_vsvl(-1,  vrj, nY*gOutWidth) ;
  __vr vrw_s2 = _vel_vaddsl_vsvl( 1,  vrj, nY*gOutWidth) ;

  __vm256 vmw_s0 =  _vel_vfmklge_mvl(vrw_s0, nY*gOutWidth) ;						// condition(0 <= w)
  __vm256 vmw_s2  = _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(inWidth,vrw_s2, nY*gOutWidth), nY*gOutWidth) ;	// condition(w < inWidth)

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

	__vr vrh_r0 = _vel_vaddsl_vsvl(-1+y*2, vri, vl) ;
	__vr vrh_r2 = _vel_vaddsl_vsvl( 1+y*2, vri, vl) ;

	__vm256 vmh_r0 =  _vel_vfmklge_mvl(vrh_r0, vl) ;				// condition(0 <= h)
	__vm256 vmh_r2 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(inHeight,vrh_r2, vl), vl) ;	// condition(h < inHeight)

	__vm256 vmhw_r0s0 = _vel_andm_mmm(vmh_r0, vmw_s0) ;
	__vm256 vmhw_r0s1 = vmh_r0 ;
	__vm256 vmhw_r0s2 = _vel_andm_mmm(vmh_r0, vmw_s2) ;

	__vm256 vmhw_r1s0 = vmw_s0 ;
	__vm256 vmhw_r1s2 = vmw_s2 ;

	__vm256 vmhw_r2s0 = _vel_andm_mmm(vmh_r2, vmw_s0) ;
	__vm256 vmhw_r2s1 = vmh_r2 ;
	__vm256 vmhw_r2s2 = _vel_andm_mmm(vmh_r2, vmw_s2) ;

	__vr vrin_ptr_r0s0 = _vel_vaddsl_vsvl(4*(((n*inChannel+c)*inHeight+2*y-1)*inWidth-1), vrin_ptr_ost, vl) ;
	__vr vrin_r0s0  = _vel_vgtu_vvssml(vrin_ptr_r0s0, 0, 0, vmh_r0, vl) ;
	__vr vrin_ptr_r0s1 = _vel_vaddsl_vsvl(4*(((n*inChannel+c)*inHeight+2*y-1)*inWidth  ), vrin_ptr_ost, vl) ;
	__vr vrin_r0s1  = _vel_vgtu_vvssml(vrin_ptr_r0s1, 0, 0, vmh_r0, vl) ;
	__vr vrin_ptr_r0s2 = _vel_vaddsl_vsvl(4*(((n*inChannel+c)*inHeight+2*y-1)*inWidth+1), vrin_ptr_ost, vl) ;
	__vr vrin_r0s2  = _vel_vgtu_vvssml(vrin_ptr_r0s2, 0, 0, vmh_r0, vl) ;

	__vr vrin_ptr_r1s0 = _vel_vaddsl_vsvl(4*(((n*inChannel+c)*inHeight+2*y  )*inWidth-1), vrin_ptr_ost, vl) ;
	__vr vrin_r1s0  = _vel_vgtu_vvssl(vrin_ptr_r1s0, 0, 0, vl) ;
	__vr vrin_ptr_r1s1 = _vel_vaddsl_vsvl(4*(((n*inChannel+c)*inHeight+2*y  )*inWidth  ), vrin_ptr_ost, vl) ;
	__vr vrin_r1s1  = _vel_vgtu_vvssl(vrin_ptr_r1s1, 0, 0, vl) ;
	__vr vrin_ptr_r1s2 = _vel_vaddsl_vsvl(4*(((n*inChannel+c)*inHeight+2*y  )*inWidth+1), vrin_ptr_ost, vl) ;
	__vr vrin_r1s2  = _vel_vgtu_vvssl(vrin_ptr_r1s2, 0, 0, vl) ;

	__vr vrin_ptr_r2s0 = _vel_vaddsl_vsvl(4*(((n*inChannel+c)*inHeight+2*y+1)*inWidth-1), vrin_ptr_ost, vl) ;
	__vr vrin_r2s0  = _vel_vgtu_vvssml(vrin_ptr_r2s0, 0, 0, vmh_r2, vl) ;
	__vr vrin_ptr_r2s1 = _vel_vaddsl_vsvl(4*(((n*inChannel+c)*inHeight+2*y+1)*inWidth  ), vrin_ptr_ost, vl) ;
	__vr vrin_r2s1  = _vel_vgtu_vvssml(vrin_ptr_r2s1, 0, 0, vmh_r2, vl) ;
	__vr vrin_ptr_r2s2 = _vel_vaddsl_vsvl(4*(((n*inChannel+c)*inHeight+2*y+1)*inWidth+1), vrin_ptr_ost, vl) ;
	__vr vrin_r2s2  = _vel_vgtu_vvssml(vrin_ptr_r2s2, 0, 0, vmh_r2, vl) ;

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

#define VFADD(VRIN, RS)										\
	{											\
	  if(NUM_KERNEL >= 2) vrsum01_##RS = _vel_pvfmad_vvvvvl(vrsum01_##RS, VRIN, vrgout01, vrsum01_##RS, vl) ;	\
	  if(NUM_KERNEL >= 4) vrsum23_##RS = _vel_pvfmad_vvvvvl(vrsum23_##RS, VRIN, vrgout23, vrsum23_##RS, vl) ;	\
	  if(NUM_KERNEL >= 6) vrsum45_##RS = _vel_pvfmad_vvvvvl(vrsum45_##RS, VRIN, vrgout45, vrsum45_##RS, vl) ;	\
	  if(NUM_KERNEL >= 8) vrsum67_##RS = _vel_pvfmad_vvvvvl(vrsum67_##RS, VRIN, vrgout67, vrsum67_##RS, vl) ;	\
	}

	vrin_r0s0 = _vel_vmrg_vsvml(0.f, vrin_r0s0, vmhw_r0s0, vl) ;
	__vr vrinP_r0s0 = _vel_vshf_vvvsl(vrin_r0s0, vrin_r0s0, VE_VSHUFFLE_YUZU, vl) ;
	VFADD(vrinP_r0s0, r0s0) ;

	vrin_r0s1 = _vel_vmrg_vsvml(0.f, vrin_r0s1, vmhw_r0s1, vl) ;
	__vr vrinP_r0s1 = _vel_vshf_vvvsl(vrin_r0s1, vrin_r0s1, VE_VSHUFFLE_YUZU, vl) ;
	VFADD(vrinP_r0s1, r0s1) ;

	vrin_r0s2 = _vel_vmrg_vsvml(0.f, vrin_r0s2, vmhw_r0s2, vl) ;
	__vr vrinP_r0s2 = _vel_vshf_vvvsl(vrin_r0s2, vrin_r0s2, VE_VSHUFFLE_YUZU, vl) ;
	VFADD(vrinP_r0s2, r0s2) ;

	vrin_r1s0 = _vel_vmrg_vsvml(0.f, vrin_r1s0, vmhw_r1s0, vl) ;
	__vr vrinP_r1s0 = _vel_vshf_vvvsl(vrin_r1s0, vrin_r1s0, VE_VSHUFFLE_YUZU, vl) ;
	VFADD(vrinP_r1s0, r1s0) ;

	// need not to use mask
	__vr vrinP_r1s1 = _vel_vshf_vvvsl(vrin_r1s1, vrin_r1s1, VE_VSHUFFLE_YUZU, vl) ;
	VFADD(vrinP_r1s1, r1s1) ;

	vrin_r1s2 = _vel_vmrg_vsvml(0.f, vrin_r1s2, vmhw_r1s2, vl) ;
	__vr vrinP_r1s2 = _vel_vshf_vvvsl(vrin_r1s2, vrin_r1s2, VE_VSHUFFLE_YUZU, vl) ;
	VFADD(vrinP_r1s2, r1s2) ;

	vrin_r2s0 = _vel_vmrg_vsvml(0.f, vrin_r2s0, vmhw_r2s0, vl) ;
	__vr vrinP_r2s0 = _vel_vshf_vvvsl(vrin_r2s0, vrin_r2s0, VE_VSHUFFLE_YUZU, vl) ;
	VFADD(vrinP_r2s0, r2s0) ;

	vrin_r2s1 = _vel_vmrg_vsvml(0.f, vrin_r2s1, vmhw_r2s1, vl) ;
	__vr vrinP_r2s1 = _vel_vshf_vvvsl(vrin_r2s1, vrin_r2s1, VE_VSHUFFLE_YUZU, vl) ;
	VFADD(vrinP_r2s1, r2s1) ;

	vrin_r2s2 = _vel_vmrg_vsvml(0.f, vrin_r2s2, vmhw_r2s2, vl) ;
	__vr vrinP_r2s2 = _vel_vshf_vvvsl(vrin_r2s2, vrin_r2s2, VE_VSHUFFLE_YUZU, vl) ;
	VFADD(vrinP_r2s2, r2s2) ;

#undef VFADD
      } // gOutPixels
    } // batch

#define SUM_AND_STORE(VRSUM_R0S0, VRSUM_R0S1, VRSUM_R0S2, VRSUM_R1S0, VRSUM_R1S1, VRSUM_R1S2, VRSUM_R2S0, VRSUM_R2S1, VRSUM_R2S2, K)		\
    {																		\
      __vr vrsum0_r0s0 = _vel_vfsums_vvl(VRSUM_R0S0, VLEN) ;											\
      __vr vrsum1_r0s0 = _vel_vfsums_vvl(_vel_vsll_vvsl(VRSUM_R0S0, 32, VLEN), VLEN);								\
      _vel_vstu_vssl(vrsum0_r0s0, 4, pGKernel+kernelIndex + ((K)  ) * inChannelGroup * gKernHeight * gKernWidth + 0 * gKernWidth + 0, 1) ;	\
      _vel_vstu_vssl(vrsum1_r0s0, 4, pGKernel+kernelIndex + ((K)+1) * inChannelGroup * gKernHeight * gKernWidth + 0 * gKernWidth + 0, 1) ;	\
      __vr vrsum0_r0s1 = _vel_vfsums_vvl(VRSUM_R0S1, VLEN) ;											\
      __vr vrsum1_r0s1 = _vel_vfsums_vvl(_vel_vsll_vvsl(VRSUM_R0S1, 32, VLEN), VLEN);								\
      _vel_vstu_vssl(vrsum0_r0s1, 4, pGKernel+kernelIndex + ((K)  ) * inChannelGroup * gKernHeight * gKernWidth + 0 * gKernWidth + 1, 1) ;	\
      _vel_vstu_vssl(vrsum1_r0s1, 4, pGKernel+kernelIndex + ((K)+1) * inChannelGroup * gKernHeight * gKernWidth + 0 * gKernWidth + 1, 1) ;	\
      __vr vrsum0_r0s2 = _vel_vfsums_vvl(VRSUM_R0S2, VLEN) ;											\
      __vr vrsum1_r0s2 = _vel_vfsums_vvl(_vel_vsll_vvsl(VRSUM_R0S2, 32, VLEN), VLEN);								\
      _vel_vstu_vssl(vrsum0_r0s2, 4, pGKernel+kernelIndex + ((K)  ) * inChannelGroup * gKernHeight * gKernWidth + 0 * gKernWidth + 2, 1) ;	\
      _vel_vstu_vssl(vrsum1_r0s2, 4, pGKernel+kernelIndex + ((K)+1) * inChannelGroup * gKernHeight * gKernWidth + 0 * gKernWidth + 2, 1) ;	\
      __vr vrsum0_r1s0 = _vel_vfsums_vvl(VRSUM_R1S0, VLEN) ;											\
      __vr vrsum1_r1s0 = _vel_vfsums_vvl(_vel_vsll_vvsl(VRSUM_R1S0, 32, VLEN), VLEN);								\
      _vel_vstu_vssl(vrsum0_r1s0, 4, pGKernel+kernelIndex + ((K)  ) * inChannelGroup * gKernHeight * gKernWidth + 1 * gKernWidth + 0, 1) ;	\
      _vel_vstu_vssl(vrsum1_r1s0, 4, pGKernel+kernelIndex + ((K)+1) * inChannelGroup * gKernHeight * gKernWidth + 1 * gKernWidth + 0, 1) ;	\
      __vr vrsum0_r1s1 = _vel_vfsums_vvl(VRSUM_R1S1, VLEN) ;											\
      __vr vrsum1_r1s1 = _vel_vfsums_vvl(_vel_vsll_vvsl(VRSUM_R1S1, 32, VLEN), VLEN);								\
      _vel_vstu_vssl(vrsum0_r1s1, 4, pGKernel+kernelIndex + ((K)  ) * inChannelGroup * gKernHeight * gKernWidth + 1 * gKernWidth + 1, 1) ;	\
      _vel_vstu_vssl(vrsum1_r1s1, 4, pGKernel+kernelIndex + ((K)+1) * inChannelGroup * gKernHeight * gKernWidth + 1 * gKernWidth + 1, 1) ;	\
      __vr vrsum0_r1s2 = _vel_vfsums_vvl(VRSUM_R1S2, VLEN) ;											\
      __vr vrsum1_r1s2 = _vel_vfsums_vvl(_vel_vsll_vvsl(VRSUM_R1S2, 32, VLEN), VLEN);								\
      _vel_vstu_vssl(vrsum0_r1s2, 4, pGKernel+kernelIndex + ((K)  ) * inChannelGroup * gKernHeight * gKernWidth + 1 * gKernWidth + 2, 1) ;	\
      _vel_vstu_vssl(vrsum1_r1s2, 4, pGKernel+kernelIndex + ((K)+1) * inChannelGroup * gKernHeight * gKernWidth + 1 * gKernWidth + 2, 1) ;	\
      __vr vrsum0_r2s0 = _vel_vfsums_vvl(VRSUM_R2S0, VLEN) ;											\
      __vr vrsum1_r2s0 = _vel_vfsums_vvl(_vel_vsll_vvsl(VRSUM_R2S0, 32, VLEN), VLEN);								\
      _vel_vstu_vssl(vrsum0_r2s0, 4, pGKernel+kernelIndex + ((K)  ) * inChannelGroup * gKernHeight * gKernWidth + 2 * gKernWidth + 0, 1) ;	\
      _vel_vstu_vssl(vrsum1_r2s0, 4, pGKernel+kernelIndex + ((K)+1) * inChannelGroup * gKernHeight * gKernWidth + 2 * gKernWidth + 0, 1) ;	\
      __vr vrsum0_r2s1 = _vel_vfsums_vvl(VRSUM_R2S1, VLEN) ;											\
      __vr vrsum1_r2s1 = _vel_vfsums_vvl(_vel_vsll_vvsl(VRSUM_R2S1, 32, VLEN), VLEN);								\
      _vel_vstu_vssl(vrsum0_r2s1, 4, pGKernel+kernelIndex + ((K)  ) * inChannelGroup * gKernHeight * gKernWidth + 2 * gKernWidth + 1, 1) ;	\
      _vel_vstu_vssl(vrsum1_r2s1, 4, pGKernel+kernelIndex + ((K)+1) * inChannelGroup * gKernHeight * gKernWidth + 2 * gKernWidth + 1, 1) ;	\
      __vr vrsum0_r2s2 = _vel_vfsums_vvl(VRSUM_R2S2, VLEN) ;											\
      __vr vrsum1_r2s2 = _vel_vfsums_vvl(_vel_vsll_vvsl(VRSUM_R2S2, 32, VLEN), VLEN);								\
      _vel_vstu_vssl(vrsum0_r2s2, 4, pGKernel+kernelIndex + ((K)  ) * inChannelGroup * gKernHeight * gKernWidth + 2 * gKernWidth + 2, 1) ;	\
      _vel_vstu_vssl(vrsum1_r2s2, 4, pGKernel+kernelIndex + ((K)+1) * inChannelGroup * gKernHeight * gKernWidth + 2 * gKernWidth + 2, 1) ;	\
    }

    if(NUM_KERNEL >= 2) SUM_AND_STORE(vrsum01_r0s0, vrsum01_r0s1, vrsum01_r0s2, vrsum01_r1s0, vrsum01_r1s1, vrsum01_r1s2, vrsum01_r2s0, vrsum01_r2s1, vrsum01_r2s2, 0) ;
    if(NUM_KERNEL >= 4) SUM_AND_STORE(vrsum23_r0s0, vrsum23_r0s1, vrsum23_r0s2, vrsum23_r1s0, vrsum23_r1s1, vrsum23_r1s2, vrsum23_r2s0, vrsum23_r2s1, vrsum23_r2s2, 2) ;
    if(NUM_KERNEL >= 6) SUM_AND_STORE(vrsum45_r0s0, vrsum45_r0s1, vrsum45_r0s2, vrsum45_r1s0, vrsum45_r1s1, vrsum45_r1s2, vrsum45_r2s0, vrsum45_r2s1, vrsum45_r2s2, 4) ;
    if(NUM_KERNEL >= 8) SUM_AND_STORE(vrsum67_r0s0, vrsum67_r0s1, vrsum67_r0s2, vrsum67_r1s0, vrsum67_r1s1, vrsum67_r1s2, vrsum67_r2s0, vrsum67_r2s1, vrsum67_r2s2, 6) ;

#undef SUM_AND_STORE

  } // inChannel
}

template <int NUM_KERNEL>
static inline void func_even_ialigned(
  const float * pIn,
  const int64_t inWidth,
  const int64_t inHeight,
  const float * pGOut,
  const int64_t gOutWidth,
  const int64_t gOutHeight,
  float * const pGKernel,
  const int64_t gKernWidth,
  const int64_t gKernHeight,
  const int64_t inChannelGroup,
  const int64_t inChannel,
  const int64_t gOutChannel,
  const int64_t inGroupOffset,
  const int64_t outGroupOffset,
  const int64_t kernGroupOffset,
  const int64_t batch,
  const int64_t k,
  const int64_t nY,
  const __vr vri,
  const __vr vrj
)
{
  __vr vrin_ptr_ost = _vel_vsfa_vvssl(_vel_vaddul_vvvl(vrj, _vel_vmulul_vsvl(inWidth,vri, nY*gOutWidth), nY*gOutWidth), 2, (uint64_t)(pIn + inGroupOffset), nY*gOutWidth) ;

  __vr vrw_s0 = _vel_vaddsl_vsvl(-1,  vrj, nY*gOutWidth) ;
  __vm256 vmw_s0 =  _vel_vfmklge_mvl(vrw_s0, nY*gOutWidth) ;						// condition(0 <= w)

  for (int64_t c=0; c<inChannelGroup; c++) {
    const int64_t kernelIndex = kernGroupOffset + ((k     * inChannelGroup + c) * gKernHeight ) * gKernWidth ;

    __vr vrsum01_r0s0 = _vel_pvbrd_vsl(0UL, VLEN) ;
    __vr vrsum23_r0s0 = _vel_pvbrd_vsl(0UL, VLEN) ;
    __vr vrsum45_r0s0 = _vel_pvbrd_vsl(0UL, VLEN) ;
    __vr vrsum67_r0s0 = _vel_pvbrd_vsl(0UL, VLEN) ;
    __vr vrsum89_r0s0 = _vel_pvbrd_vsl(0UL, VLEN) ;

    __vr vrsum01_r0s1 = _vel_pvbrd_vsl(0UL, VLEN) ;
    __vr vrsum23_r0s1 = _vel_pvbrd_vsl(0UL, VLEN) ;
    __vr vrsum45_r0s1 = _vel_pvbrd_vsl(0UL, VLEN) ;
    __vr vrsum67_r0s1 = _vel_pvbrd_vsl(0UL, VLEN) ;
    __vr vrsum89_r0s1 = _vel_pvbrd_vsl(0UL, VLEN) ;

    __vr vrsum01_r0s2 = _vel_pvbrd_vsl(0UL, VLEN) ;
    __vr vrsum23_r0s2 = _vel_pvbrd_vsl(0UL, VLEN) ;
    __vr vrsum45_r0s2 = _vel_pvbrd_vsl(0UL, VLEN) ;
    __vr vrsum67_r0s2 = _vel_pvbrd_vsl(0UL, VLEN) ;
    __vr vrsum89_r0s2 = _vel_pvbrd_vsl(0UL, VLEN) ;

    __vr vrsum01_r1s0 = _vel_pvbrd_vsl(0UL, VLEN) ;
    __vr vrsum23_r1s0 = _vel_pvbrd_vsl(0UL, VLEN) ;
    __vr vrsum45_r1s0 = _vel_pvbrd_vsl(0UL, VLEN) ;
    __vr vrsum67_r1s0 = _vel_pvbrd_vsl(0UL, VLEN) ;
    __vr vrsum89_r1s0 = _vel_pvbrd_vsl(0UL, VLEN) ;

    __vr vrsum01_r1s1 = _vel_pvbrd_vsl(0UL, VLEN) ;
    __vr vrsum23_r1s1 = _vel_pvbrd_vsl(0UL, VLEN) ;
    __vr vrsum45_r1s1 = _vel_pvbrd_vsl(0UL, VLEN) ;
    __vr vrsum67_r1s1 = _vel_pvbrd_vsl(0UL, VLEN) ;
    __vr vrsum89_r1s1 = _vel_pvbrd_vsl(0UL, VLEN) ;

    __vr vrsum01_r1s2 = _vel_pvbrd_vsl(0UL, VLEN) ;
    __vr vrsum23_r1s2 = _vel_pvbrd_vsl(0UL, VLEN) ;
    __vr vrsum45_r1s2 = _vel_pvbrd_vsl(0UL, VLEN) ;
    __vr vrsum67_r1s2 = _vel_pvbrd_vsl(0UL, VLEN) ;
    __vr vrsum89_r1s2 = _vel_pvbrd_vsl(0UL, VLEN) ;

    __vr vrsum01_r2s0 = _vel_pvbrd_vsl(0UL, VLEN) ;
    __vr vrsum23_r2s0 = _vel_pvbrd_vsl(0UL, VLEN) ;
    __vr vrsum45_r2s0 = _vel_pvbrd_vsl(0UL, VLEN) ;
    __vr vrsum67_r2s0 = _vel_pvbrd_vsl(0UL, VLEN) ;
    __vr vrsum89_r2s0 = _vel_pvbrd_vsl(0UL, VLEN) ;

    __vr vrsum01_r2s1 = _vel_pvbrd_vsl(0UL, VLEN) ;
    __vr vrsum23_r2s1 = _vel_pvbrd_vsl(0UL, VLEN) ;
    __vr vrsum45_r2s1 = _vel_pvbrd_vsl(0UL, VLEN) ;
    __vr vrsum67_r2s1 = _vel_pvbrd_vsl(0UL, VLEN) ;
    __vr vrsum89_r2s1 = _vel_pvbrd_vsl(0UL, VLEN) ;

    __vr vrsum01_r2s2 = _vel_pvbrd_vsl(0UL, VLEN) ;
    __vr vrsum23_r2s2 = _vel_pvbrd_vsl(0UL, VLEN) ;
    __vr vrsum45_r2s2 = _vel_pvbrd_vsl(0UL, VLEN) ;
    __vr vrsum67_r2s2 = _vel_pvbrd_vsl(0UL, VLEN) ;
    __vr vrsum89_r2s2 = _vel_pvbrd_vsl(0UL, VLEN) ;

    for (int64_t n=0; n<batch; n++) {
      for (int64_t y=0; y<gOutHeight; y+=nY) {

	const int64_t vl = gOutWidth * (gOutHeight - y < nY ? gOutHeight - y : nY) ;
	const int64_t gop = y * gOutWidth ;

	__vr vrh_r0 = _vel_vaddsl_vsvl(-1+y*2, vri, vl) ;
	__vr vrh_r2 = _vel_vaddsl_vsvl( 1+y*2, vri, vl) ;

	__vm256 vmh_r0 =  _vel_vfmklge_mvl(vrh_r0, vl) ;				// condition(0 <= h)
	__vm256 vmh_r2 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(inHeight,vrh_r2, vl), vl) ;	// condition(h < inHeight)

	__vr vrin_ptr_r0 = _vel_vaddsl_vsvl(4*(((n*inChannel+c)*inHeight+2*y-1)*inWidth), vrin_ptr_ost, vl) ;
	__vr vrin_r0  = _vel_vgt_vvssml(vrin_ptr_r0, 0, 0, vmh_r0, vl) ;

	__vr vrin_ptr_r1 = _vel_vaddsl_vsvl(4*(((n*inChannel+c)*inHeight+2*y  )*inWidth), vrin_ptr_ost, vl) ;
	__vr vrin_r1  = _vel_vgt_vvssl(vrin_ptr_r1, 0, 0, vl) ;

	__vr vrin_ptr_r2 = _vel_vaddsl_vsvl(4*(((n*inChannel+c)*inHeight+2*y+1)*inWidth), vrin_ptr_ost, vl) ;
	__vr vrin_r2  = _vel_vgt_vvssml(vrin_ptr_r2, 0, 0, vmh_r2, vl) ;

	__vr vrgout0 = _vel_vldu_vssl(4, pGOut+outGroupOffset + ((n * gOutChannel + k  ) * gOutHeight ) * gOutWidth + gop, vl) ;
	__vr vrgout1 = _vel_vldu_vssl(4, pGOut+outGroupOffset + ((n * gOutChannel + k+1) * gOutHeight ) * gOutWidth + gop, vl) ;
	__vr vrgout2 = _vel_vldu_vssl(4, pGOut+outGroupOffset + ((n * gOutChannel + k+2) * gOutHeight ) * gOutWidth + gop, vl) ;
	__vr vrgout3 = _vel_vldu_vssl(4, pGOut+outGroupOffset + ((n * gOutChannel + k+3) * gOutHeight ) * gOutWidth + gop, vl) ;
	__vr vrgout4 = _vel_vldu_vssl(4, pGOut+outGroupOffset + ((n * gOutChannel + k+4) * gOutHeight ) * gOutWidth + gop, vl) ;
	__vr vrgout5 = _vel_vldu_vssl(4, pGOut+outGroupOffset + ((n * gOutChannel + k+5) * gOutHeight ) * gOutWidth + gop, vl) ;
	__vr vrgout6 = _vel_vldu_vssl(4, pGOut+outGroupOffset + ((n * gOutChannel + k+6) * gOutHeight ) * gOutWidth + gop, vl) ;
	__vr vrgout7 = _vel_vldu_vssl(4, pGOut+outGroupOffset + ((n * gOutChannel + k+7) * gOutHeight ) * gOutWidth + gop, vl) ;
	__vr vrgout8 = _vel_vldu_vssl(4, pGOut+outGroupOffset + ((n * gOutChannel + k+8) * gOutHeight ) * gOutWidth + gop, vl) ;
	__vr vrgout9 = _vel_vldu_vssl(4, pGOut+outGroupOffset + ((n * gOutChannel + k+9) * gOutHeight ) * gOutWidth + gop, vl) ;

	__vr vrgout01 = _vel_vshf_vvvsl(vrgout0, vrgout1, VE_VSHUFFLE_YUZU, vl) ;
	__vr vrgout23 = _vel_vshf_vvvsl(vrgout2, vrgout3, VE_VSHUFFLE_YUZU, vl) ;
	__vr vrgout45 = _vel_vshf_vvvsl(vrgout4, vrgout5, VE_VSHUFFLE_YUZU, vl) ;
	__vr vrgout67 = _vel_vshf_vvvsl(vrgout6, vrgout7, VE_VSHUFFLE_YUZU, vl) ;
	__vr vrgout89 = _vel_vshf_vvvsl(vrgout8, vrgout9, VE_VSHUFFLE_YUZU, vl) ;

#define VFADD(VRIN, RS)										\
	{											\
	  if(NUM_KERNEL >= 2) vrsum01_##RS = _vel_pvfmad_vvvvvl(vrsum01_##RS, VRIN, vrgout01, vrsum01_##RS, vl) ;	\
	  if(NUM_KERNEL >= 4) vrsum23_##RS = _vel_pvfmad_vvvvvl(vrsum23_##RS, VRIN, vrgout23, vrsum23_##RS, vl) ;	\
	  if(NUM_KERNEL >= 6) vrsum45_##RS = _vel_pvfmad_vvvvvl(vrsum45_##RS, VRIN, vrgout45, vrsum45_##RS, vl) ;	\
	  if(NUM_KERNEL >= 8) vrsum67_##RS = _vel_pvfmad_vvvvvl(vrsum67_##RS, VRIN, vrgout67, vrsum67_##RS, vl) ;	\
	  if(NUM_KERNEL >=10) vrsum89_##RS = _vel_pvfmad_vvvvvl(vrsum89_##RS, VRIN, vrgout89, vrsum89_##RS, vl) ;	\
	}

	vrin_r0 = _vel_vmrg_vsvml(0.f, vrin_r0, vmh_r0, vl) ;
	__vr vrinP_r0s2 = _vel_vshf_vvvsl(vrin_r0, vrin_r0, VE_VSHUFFLE_YUZU, vl) ;
	__vr vrinP_r0s1 = _vel_vshf_vvvsl(vrin_r0, vrin_r0, VE_VSHUFFLE_YLZL, vl) ;
	__vr vrinP_r0s0 = _vel_vmv_vsvl(-1, vrinP_r0s2, vl) ;
	VFADD(vrinP_r0s2, r0s2) ;
	VFADD(vrinP_r0s1, r0s1) ;
	VFADD(vrinP_r0s0, r0s0) ;

	// need not to use mask
	__vr vrinP_r1s2 = _vel_vshf_vvvsl(vrin_r1, vrin_r1, VE_VSHUFFLE_YUZU, vl) ;
	__vr vrinP_r1s1 = _vel_vshf_vvvsl(vrin_r1, vrin_r1, VE_VSHUFFLE_YLZL, vl) ;
	__vr vrinP_r1s0 = _vel_vmv_vsvl(-1, vrinP_r1s2, vl) ;
	VFADD(vrinP_r1s2, r1s2) ;
	VFADD(vrinP_r1s1, r1s1) ;
	VFADD(vrinP_r1s0, r1s0) ;

	vrin_r2 = _vel_vmrg_vsvml(0.f, vrin_r2, vmh_r2, vl) ;
	__vr vrinP_r2s2 = _vel_vshf_vvvsl(vrin_r2, vrin_r2, VE_VSHUFFLE_YUZU, vl) ;
	__vr vrinP_r2s1 = _vel_vshf_vvvsl(vrin_r2, vrin_r2, VE_VSHUFFLE_YLZL, vl) ;
	__vr vrinP_r2s0 = _vel_vmv_vsvl(-1, vrinP_r2s2, vl) ;
	VFADD(vrinP_r2s2, r2s2) ;
	VFADD(vrinP_r2s1, r2s1) ;
	VFADD(vrinP_r2s0, r2s0) ;

#undef VFADD
      } // gOutPixels
    } // batch

#define SUM_AND_STORE(VRSUM_R0S0, VRSUM_R0S1, VRSUM_R0S2, VRSUM_R1S0, VRSUM_R1S1, VRSUM_R1S2, VRSUM_R2S0, VRSUM_R2S1, VRSUM_R2S2, K)		\
    {																		\
      __vr vrsum0_r0s0 = _vel_vfsums_vvml(VRSUM_R0S0, vmw_s0, VLEN) ;										\
      __vr vrsum1_r0s0 = _vel_vfsums_vvml(_vel_vsll_vvsl(VRSUM_R0S0, 32, VLEN), vmw_s0, VLEN);							\
      _vel_vstu_vssl(vrsum0_r0s0, 4, pGKernel+kernelIndex + ((K)  ) * inChannelGroup * gKernHeight * gKernWidth + 0 * gKernWidth + 0, 1) ;	\
      _vel_vstu_vssl(vrsum1_r0s0, 4, pGKernel+kernelIndex + ((K)+1) * inChannelGroup * gKernHeight * gKernWidth + 0 * gKernWidth + 0, 1) ;	\
      __vr vrsum0_r0s1 = _vel_vfsums_vvl(VRSUM_R0S1, VLEN) ;											\
      __vr vrsum1_r0s1 = _vel_vfsums_vvl(_vel_vsll_vvsl(VRSUM_R0S1, 32, VLEN), VLEN);								\
      _vel_vstu_vssl(vrsum0_r0s1, 4, pGKernel+kernelIndex + ((K)  ) * inChannelGroup * gKernHeight * gKernWidth + 0 * gKernWidth + 1, 1) ;	\
      _vel_vstu_vssl(vrsum1_r0s1, 4, pGKernel+kernelIndex + ((K)+1) * inChannelGroup * gKernHeight * gKernWidth + 0 * gKernWidth + 1, 1) ;	\
      __vr vrsum0_r0s2 = _vel_vfsums_vvl(VRSUM_R0S2, VLEN) ;											\
      __vr vrsum1_r0s2 = _vel_vfsums_vvl(_vel_vsll_vvsl(VRSUM_R0S2, 32, VLEN), VLEN);								\
      _vel_vstu_vssl(vrsum0_r0s2, 4, pGKernel+kernelIndex + ((K)  ) * inChannelGroup * gKernHeight * gKernWidth + 0 * gKernWidth + 2, 1) ;	\
      _vel_vstu_vssl(vrsum1_r0s2, 4, pGKernel+kernelIndex + ((K)+1) * inChannelGroup * gKernHeight * gKernWidth + 0 * gKernWidth + 2, 1) ;	\
      __vr vrsum0_r1s0 = _vel_vfsums_vvml(VRSUM_R1S0, vmw_s0, VLEN) ;										\
      __vr vrsum1_r1s0 = _vel_vfsums_vvml(_vel_vsll_vvsl(VRSUM_R1S0, 32, VLEN), vmw_s0, VLEN);							\
      _vel_vstu_vssl(vrsum0_r1s0, 4, pGKernel+kernelIndex + ((K)  ) * inChannelGroup * gKernHeight * gKernWidth + 1 * gKernWidth + 0, 1) ;	\
      _vel_vstu_vssl(vrsum1_r1s0, 4, pGKernel+kernelIndex + ((K)+1) * inChannelGroup * gKernHeight * gKernWidth + 1 * gKernWidth + 0, 1) ;	\
      __vr vrsum0_r1s1 = _vel_vfsums_vvl(VRSUM_R1S1, VLEN) ;											\
      __vr vrsum1_r1s1 = _vel_vfsums_vvl(_vel_vsll_vvsl(VRSUM_R1S1, 32, VLEN), VLEN);								\
      _vel_vstu_vssl(vrsum0_r1s1, 4, pGKernel+kernelIndex + ((K)  ) * inChannelGroup * gKernHeight * gKernWidth + 1 * gKernWidth + 1, 1) ;	\
      _vel_vstu_vssl(vrsum1_r1s1, 4, pGKernel+kernelIndex + ((K)+1) * inChannelGroup * gKernHeight * gKernWidth + 1 * gKernWidth + 1, 1) ;	\
      __vr vrsum0_r1s2 = _vel_vfsums_vvl(VRSUM_R1S2, VLEN) ;											\
      __vr vrsum1_r1s2 = _vel_vfsums_vvl(_vel_vsll_vvsl(VRSUM_R1S2, 32, VLEN), VLEN);								\
      _vel_vstu_vssl(vrsum0_r1s2, 4, pGKernel+kernelIndex + ((K)  ) * inChannelGroup * gKernHeight * gKernWidth + 1 * gKernWidth + 2, 1) ;	\
      _vel_vstu_vssl(vrsum1_r1s2, 4, pGKernel+kernelIndex + ((K)+1) * inChannelGroup * gKernHeight * gKernWidth + 1 * gKernWidth + 2, 1) ;	\
      __vr vrsum0_r2s0 = _vel_vfsums_vvml(VRSUM_R2S0, vmw_s0, VLEN) ;										\
      __vr vrsum1_r2s0 = _vel_vfsums_vvml(_vel_vsll_vvsl(VRSUM_R2S0, 32, VLEN), vmw_s0, VLEN);							\
      _vel_vstu_vssl(vrsum0_r2s0, 4, pGKernel+kernelIndex + ((K)  ) * inChannelGroup * gKernHeight * gKernWidth + 2 * gKernWidth + 0, 1) ;	\
      _vel_vstu_vssl(vrsum1_r2s0, 4, pGKernel+kernelIndex + ((K)+1) * inChannelGroup * gKernHeight * gKernWidth + 2 * gKernWidth + 0, 1) ;	\
      __vr vrsum0_r2s1 = _vel_vfsums_vvl(VRSUM_R2S1, VLEN) ;											\
      __vr vrsum1_r2s1 = _vel_vfsums_vvl(_vel_vsll_vvsl(VRSUM_R2S1, 32, VLEN), VLEN);								\
      _vel_vstu_vssl(vrsum0_r2s1, 4, pGKernel+kernelIndex + ((K)  ) * inChannelGroup * gKernHeight * gKernWidth + 2 * gKernWidth + 1, 1) ;	\
      _vel_vstu_vssl(vrsum1_r2s1, 4, pGKernel+kernelIndex + ((K)+1) * inChannelGroup * gKernHeight * gKernWidth + 2 * gKernWidth + 1, 1) ;	\
      __vr vrsum0_r2s2 = _vel_vfsums_vvl(VRSUM_R2S2, VLEN) ;											\
      __vr vrsum1_r2s2 = _vel_vfsums_vvl(_vel_vsll_vvsl(VRSUM_R2S2, 32, VLEN), VLEN);								\
      _vel_vstu_vssl(vrsum0_r2s2, 4, pGKernel+kernelIndex + ((K)  ) * inChannelGroup * gKernHeight * gKernWidth + 2 * gKernWidth + 2, 1) ;	\
      _vel_vstu_vssl(vrsum1_r2s2, 4, pGKernel+kernelIndex + ((K)+1) * inChannelGroup * gKernHeight * gKernWidth + 2 * gKernWidth + 2, 1) ;	\
    }

    if(NUM_KERNEL >= 2) SUM_AND_STORE(vrsum01_r0s0, vrsum01_r0s1, vrsum01_r0s2, vrsum01_r1s0, vrsum01_r1s1, vrsum01_r1s2, vrsum01_r2s0, vrsum01_r2s1, vrsum01_r2s2, 0) ;
    if(NUM_KERNEL >= 4) SUM_AND_STORE(vrsum23_r0s0, vrsum23_r0s1, vrsum23_r0s2, vrsum23_r1s0, vrsum23_r1s1, vrsum23_r1s2, vrsum23_r2s0, vrsum23_r2s1, vrsum23_r2s2, 2) ;
    if(NUM_KERNEL >= 6) SUM_AND_STORE(vrsum45_r0s0, vrsum45_r0s1, vrsum45_r0s2, vrsum45_r1s0, vrsum45_r1s1, vrsum45_r1s2, vrsum45_r2s0, vrsum45_r2s1, vrsum45_r2s2, 4) ;
    if(NUM_KERNEL >= 8) SUM_AND_STORE(vrsum67_r0s0, vrsum67_r0s1, vrsum67_r0s2, vrsum67_r1s0, vrsum67_r1s1, vrsum67_r1s2, vrsum67_r2s0, vrsum67_r2s1, vrsum67_r2s2, 6) ;
    if(NUM_KERNEL >=10) SUM_AND_STORE(vrsum89_r0s0, vrsum89_r0s1, vrsum89_r0s2, vrsum89_r1s0, vrsum89_r1s1, vrsum89_r1s2, vrsum89_r2s0, vrsum89_r2s1, vrsum89_r2s2, 8) ;

#undef SUM_AND_STORE


  } // inChannel
}

static inline void convloop(
  const float * pIn,
  const float * pGOut,
  float * const pGKernel,
  const int64_t batch,
  const int64_t group,
  const int64_t inChannel,
  const int64_t inWidth,
  const int64_t inHeight,
  const int64_t gOutChannel,
  const int64_t gOutWidth,
  const int64_t gOutHeight,
  const int64_t gKernWidth,
  const int64_t gKernHeight,
  const int64_t strideWidth,
  const int64_t strideHeight,
  const int64_t padWidth,
  const int64_t padHeight,
  const int64_t dilationWidth,
  const int64_t dilationHeight,
  const int64_t inChannelGroup,
  const int64_t gOutChannelGroup,
  const int64_t beginOChannel,
  const int64_t nOChannel
)
{
  const int64_t nY = VLEN / gOutWidth ;

  __vr vrseq = _vel_vseq_vl(nY*gOutWidth) ;			// xy
  __vr vry  = _vel_vdivsl_vvsl(vrseq, gOutWidth, nY*gOutWidth) ;
  __vr vrx  = _vel_vsubsl_vvvl(vrseq, _vel_vmulul_vsvl(gOutWidth,vry, nY*gOutWidth), nY*gOutWidth) ;
  __vr vri  = _vel_vmulsl_vsvl(2, vry, nY*gOutWidth) ;
  __vr vrj  = _vel_vmulsl_vsvl(2,  vrx, nY*gOutWidth) ;

  const int64_t ialigned = ((((uint64_t) pIn) & 0x7) == 0) && ((inWidth & 0x1) == 0) ;

  for (int64_t g = 0; g < group; g++) {
    int64_t inGroupOffset   = g * inChannelGroup  * inHeight  * inWidth;
    int64_t outGroupOffset  = (g * gOutChannelGroup + beginOChannel) * gOutHeight * gOutWidth;
    int64_t kernGroupOffset = (g * gOutChannelGroup + beginOChannel) * inChannelGroup * gKernHeight * gKernWidth;

    int64_t k=0;

    if( ialigned ) {
      int64_t kremain = nOChannel % 10 ;

      switch(kremain) {
      case 1 :
        func_odd_ialigned<1>(pIn, inWidth,inHeight,
  	 pGOut, gOutWidth, gOutHeight,
  	 pGKernel, gKernWidth, gKernHeight,
  	 inChannelGroup, inChannel, gOutChannel,
  	 inGroupOffset,  outGroupOffset,  kernGroupOffset,
  	 batch,  k,
  	 nY, vri, vrj ) ;
        k+=1 ;
        break ;
      case 2 :
  	func_even_ialigned<2>(pIn, inWidth,inHeight,
  	   pGOut, gOutWidth, gOutHeight,
  	   pGKernel, gKernWidth, gKernHeight,
  	   inChannelGroup, inChannel, gOutChannel,
  	   inGroupOffset,  outGroupOffset,  kernGroupOffset,
  	   batch,  k,
  	   nY, vri, vrj ) ;
        k+=2 ;
        break ;
      case 3 :
        func_odd_ialigned<3>(pIn, inWidth,inHeight,
  	 pGOut, gOutWidth, gOutHeight,
  	 pGKernel, gKernWidth, gKernHeight,
  	 inChannelGroup, inChannel, gOutChannel,
  	 inGroupOffset,  outGroupOffset,  kernGroupOffset,
  	 batch,  k,
  	 nY, vri, vrj ) ;
        k+=3 ;
        break ;
      case 4 :
  	func_even_ialigned<4>(pIn, inWidth,inHeight,
  	   pGOut, gOutWidth, gOutHeight,
  	   pGKernel, gKernWidth, gKernHeight,
  	   inChannelGroup, inChannel, gOutChannel,
  	   inGroupOffset,  outGroupOffset,  kernGroupOffset,
  	   batch,  k,
  	   nY, vri, vrj ) ;
        k+=4 ;
        break ;
      case 5 :
        func_odd_ialigned<5>(pIn, inWidth,inHeight,
  	 pGOut, gOutWidth, gOutHeight,
  	 pGKernel, gKernWidth, gKernHeight,
  	 inChannelGroup, inChannel, gOutChannel,
  	 inGroupOffset,  outGroupOffset,  kernGroupOffset,
  	 batch,  k,
  	 nY, vri, vrj ) ;
        k+=5 ;
        break ;
      case 6 :
  	func_even_ialigned<6>(pIn, inWidth,inHeight,
  	   pGOut, gOutWidth, gOutHeight,
  	   pGKernel, gKernWidth, gKernHeight,
  	   inChannelGroup, inChannel, gOutChannel,
  	   inGroupOffset,  outGroupOffset,  kernGroupOffset,
  	   batch,  k,
  	   nY, vri, vrj ) ;
        k+=6 ;
        break ;
      case 7 :
        func_odd_ialigned<7>(pIn, inWidth,inHeight,
  	 pGOut, gOutWidth, gOutHeight,
  	 pGKernel, gKernWidth, gKernHeight,
  	 inChannelGroup, inChannel, gOutChannel,
  	 inGroupOffset,  outGroupOffset,  kernGroupOffset,
  	 batch,  k,
  	 nY, vri, vrj ) ;
        k+=7 ;
        break ;
      case 8 :
  	func_even_ialigned<8>(pIn, inWidth,inHeight,
  	   pGOut, gOutWidth, gOutHeight,
  	   pGKernel, gKernWidth, gKernHeight,
  	   inChannelGroup, inChannel, gOutChannel,
  	   inGroupOffset,  outGroupOffset,  kernGroupOffset,
  	   batch,  k,
  	   nY, vri, vrj ) ;
        k+=8 ;
        break ;
      case 9 :
        func_odd_ialigned<9>(pIn, inWidth,inHeight,
  	 pGOut, gOutWidth, gOutHeight,
  	 pGKernel, gKernWidth, gKernHeight,
  	 inChannelGroup, inChannel, gOutChannel,
  	 inGroupOffset,  outGroupOffset,  kernGroupOffset,
  	 batch,  k,
  	 nY, vri, vrj ) ;
        k+=9 ;
        break ;
      default :
        break ;
      }
      for ( ;k<nOChannel; k+=10) {
  	func_even_ialigned<10>(pIn, inWidth,inHeight,
  	   pGOut, gOutWidth, gOutHeight,
  	   pGKernel, gKernWidth, gKernHeight,
  	   inChannelGroup, inChannel, gOutChannel,
  	   inGroupOffset,  outGroupOffset,  kernGroupOffset,
  	   batch,  k,
  	   nY, vri, vrj ) ;
      } // outChannel
    }
    else {
      int64_t kremain = nOChannel % 8 ;

      switch(kremain) {
      case 1 :
        func_odd<1>(pIn, inWidth,inHeight,
  	 pGOut, gOutWidth, gOutHeight,
  	 pGKernel, gKernWidth, gKernHeight,
  	 inChannelGroup, inChannel, gOutChannel,
  	 inGroupOffset,  outGroupOffset,  kernGroupOffset,
  	 batch,  k,
  	 nY, vri, vrj ) ;
        k+=1 ;
        break ;
      case 2 :
  	func_even<2>(pIn, inWidth,inHeight,
  	   pGOut, gOutWidth, gOutHeight,
  	   pGKernel, gKernWidth, gKernHeight,
  	   inChannelGroup, inChannel, gOutChannel,
  	   inGroupOffset,  outGroupOffset,  kernGroupOffset,
  	   batch,  k,
  	   nY, vri, vrj ) ;
        k+=2 ;
        break ;
      case 3 :
        func_odd<3>(pIn, inWidth,inHeight,
  	 pGOut, gOutWidth, gOutHeight,
  	 pGKernel, gKernWidth, gKernHeight,
  	 inChannelGroup, inChannel, gOutChannel,
  	 inGroupOffset,  outGroupOffset,  kernGroupOffset,
  	 batch,  k,
  	 nY, vri, vrj ) ;
        k+=3 ;
        break ;
      case 4 :
  	func_even<4>(pIn, inWidth,inHeight,
  	   pGOut, gOutWidth, gOutHeight,
  	   pGKernel, gKernWidth, gKernHeight,
  	   inChannelGroup, inChannel, gOutChannel,
  	   inGroupOffset,  outGroupOffset,  kernGroupOffset,
  	   batch,  k,
  	   nY, vri, vrj ) ;
        k+=4 ;
        break ;
      case 5 :
        func_odd<5>(pIn, inWidth,inHeight,
  	 pGOut, gOutWidth, gOutHeight,
  	 pGKernel, gKernWidth, gKernHeight,
  	 inChannelGroup, inChannel, gOutChannel,
  	 inGroupOffset,  outGroupOffset,  kernGroupOffset,
  	 batch,  k,
  	 nY, vri, vrj ) ;
        k+=5 ;
        break ;
      case 6 :
  	func_even<6>(pIn, inWidth,inHeight,
  	   pGOut, gOutWidth, gOutHeight,
  	   pGKernel, gKernWidth, gKernHeight,
  	   inChannelGroup, inChannel, gOutChannel,
  	   inGroupOffset,  outGroupOffset,  kernGroupOffset,
  	   batch,  k,
  	   nY, vri, vrj ) ;
        k+=6 ;
        break ;
      case 7 :
        func_odd<7>(pIn, inWidth,inHeight,
  	 pGOut, gOutWidth, gOutHeight,
  	 pGKernel, gKernWidth, gKernHeight,
  	 inChannelGroup, inChannel, gOutChannel,
  	 inGroupOffset,  outGroupOffset,  kernGroupOffset,
  	 batch,  k,
  	 nY, vri, vrj ) ;
        k+=7 ;
        break ;
      default :
        break ;
      }
      for ( ;k<nOChannel; k+=8) {
  	func_even<8>(pIn, inWidth,inHeight,
  	   pGOut, gOutWidth, gOutHeight,
  	   pGKernel, gKernWidth, gKernHeight,
  	   inChannelGroup, inChannel, gOutChannel,
  	   inGroupOffset,  outGroupOffset,  kernGroupOffset,
  	   batch,  k,
  	   nY, vri, vrj ) ;
      } // outChannel
    }


  } // group
}

extern "C"
vednnError_t
vednnConvolutionBackwardFilter_direct_dil1_str2_pad1_ker3_owU128(
    const vednnTensorParam_t * __restrict__ 		pParamIn,
    const void * __restrict__ 				pDataIn,
    const vednnTensorParam_t * __restrict__ 		pParamGradOut,
    const void * __restrict__ 				pDataGradOut,
    const vednnConvolutionParam_t * __restrict__ 	pParamConv,
    const vednnFilterParam_t * __restrict__ 		pParamGradKernel,
    void * __restrict__ 				pDataGradKernel
#ifdef VEDNN_USE_OPENMP
    ,
    const int64_t					beginOChannel,
    const int64_t					nOChannel
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

  const float * pIn      = (const float *) pDataIn;
  const float * pGOut    = (const float *) pDataGradOut;
  float * const pGKernel = (float * const) pDataGradKernel;

#ifndef VEDNN_USE_OPENMP
  const int64_t beginOChannel = 0 ;
  const int64_t nOChannel     = gOutChannelGroup ;
#endif

  convloop(pIn, pGOut, pGKernel,
	   batch, group,
           inChannel, inWidth, inHeight,
           gOutChannel, gOutWidth, gOutHeight,
           gKernWidth, gKernHeight,
	   strideWidth, strideHeight,
	   padWidth, padHeight,
	   dilationWidth, dilationHeight,
	   inChannelGroup, gOutChannelGroup,
	   beginOChannel, nOChannel ) ;

  return VEDNN_SUCCESS;
}
