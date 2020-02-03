#include <stdint.h>
#include <stdlib.h>

#include "vednn.h"

#include "velintrin.h"
#define VLEN	(256)


template<filterLayout_t FLAYOUT, int NUMKERNEL>
static inline void func(
    const float * __restrict__ pIn,
    const float * __restrict__ pGOut,
    float * __restrict__ const pGKernel,
    const int64_t gOutChannel,
    const int64_t gOutWidth,
    const int64_t gOutHeight,
    const int64_t inChannel,
    const int64_t inWidth,
    const int64_t inHeight,
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
    const int64_t inGroupOffset,
    const int64_t gOutGroupOffset,
    const int64_t gKernGroupOffset,
    const int64_t batch,
    const int64_t k,
    const int64_t nY,
    const __vr vri,
    const __vr vrj
)
{
  const int64_t remain  = NUMKERNEL & 0x1 ;
  const int64_t nPacked = NUMKERNEL >> 1 ;

  __vr vrin_ptr_ost = _vel_vsfa_vvssl(_vel_vaddul_vvvl(vrj, _vel_vmulul_vsvl(inWidth,vri, nY*gOutWidth), nY*gOutWidth), 2, (uint64_t)(pIn + inGroupOffset), nY*gOutWidth) ;

  __vr vrw_s0 = _vel_vaddsl_vsvl(-1,  vrj, nY*gOutWidth) ;
  __vr vrw_s2 = _vel_vaddsl_vsvl( 1,  vrj, nY*gOutWidth) ;

  __vm256 vmw_s0 =  _vel_vfmklge_mvl(vrw_s0, nY*gOutWidth) ;						// condition(0 <= w)
  __vm256 vmw_s2  = _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(inWidth,vrw_s2, nY*gOutWidth), nY*gOutWidth) ;	// condition(w < inWidth)

  for (int64_t c=0; c<inChannelGroup; c++) {

    __vr vrsum0_r0s0 = _vel_vbrds_vsl(0.f, VLEN) ;
    __vr vrsum0_r0s1 = _vel_vbrds_vsl(0.f, VLEN) ;
    __vr vrsum0_r0s2 = _vel_vbrds_vsl(0.f, VLEN) ;
    __vr vrsum0_r1s0 = _vel_vbrds_vsl(0.f, VLEN) ;
    __vr vrsum0_r1s1 = _vel_vbrds_vsl(0.f, VLEN) ;
    __vr vrsum0_r1s2 = _vel_vbrds_vsl(0.f, VLEN) ;
    __vr vrsum0_r2s0 = _vel_vbrds_vsl(0.f, VLEN) ;
    __vr vrsum0_r2s1 = _vel_vbrds_vsl(0.f, VLEN) ;
    __vr vrsum0_r2s2 = _vel_vbrds_vsl(0.f, VLEN) ;


    __vr vrsum_r0s0[nPacked] ;
    __vr vrsum_r0s1[nPacked] ;
    __vr vrsum_r0s2[nPacked] ;
    __vr vrsum_r1s0[nPacked] ;
    __vr vrsum_r1s1[nPacked] ;
    __vr vrsum_r1s2[nPacked] ;
    __vr vrsum_r2s0[nPacked] ;
    __vr vrsum_r2s1[nPacked] ;
    __vr vrsum_r2s2[nPacked] ;
#pragma clang loop unroll(full)
    for(int64_t kk=0; kk<nPacked; kk++) {
      vrsum_r0s0[kk] = _vel_pvbrd_vsl(0UL, VLEN) ;
      vrsum_r0s1[kk] = _vel_pvbrd_vsl(0UL, VLEN) ;
      vrsum_r0s2[kk] = _vel_pvbrd_vsl(0UL, VLEN) ;
      vrsum_r1s0[kk] = _vel_pvbrd_vsl(0UL, VLEN) ;
      vrsum_r1s1[kk] = _vel_pvbrd_vsl(0UL, VLEN) ;
      vrsum_r1s2[kk] = _vel_pvbrd_vsl(0UL, VLEN) ;
      vrsum_r2s0[kk] = _vel_pvbrd_vsl(0UL, VLEN) ;
      vrsum_r2s1[kk] = _vel_pvbrd_vsl(0UL, VLEN) ;
      vrsum_r2s2[kk] = _vel_pvbrd_vsl(0UL, VLEN) ;
    }

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

	const int64_t outIndex = gOutGroupOffset + ((n * gOutChannel + k) * gOutHeight + y) * gOutWidth ;

	__vr vrgout[NUMKERNEL] ;
#pragma clang loop unroll(full)
	for(int64_t kk=0; kk<NUMKERNEL; kk++) {
	  vrgout[kk] = _vel_vldu_vssl(4, pGOut+outIndex+kk*gOutHeight*gOutWidth, vl) ;
	}

	__vr vrgoutp[NUMKERNEL]  ;
#pragma clang loop unroll(full)
	for(int64_t kk=0; kk<nPacked; kk++) {
	  vrgoutp[kk] = _vel_vshf_vvvsl(vrgout[2*kk+remain], vrgout[2*kk+remain+1], VE_VSHUFFLE_YUZU, vl) ;
	}

#define VFADD(VRIN, VRINP, RS)												\
	{														\
	  if( remain ) {												\
	    vrsum0_##RS = _vel_vfmads_vvvvvl(vrsum0_##RS, vrin_##RS, vrgout[0], vrsum0_##RS, vl) ;			\
	  }														\
	  _Pragma("clang loop unroll(full)")										\
	  for(int64_t kk=0; kk<nPacked; kk++) {										\
	    vrsum_##RS[kk] = _vel_pvfmad_vvvvvl(vrsum_##RS[kk], vrinP_##RS, vrgoutp[kk], vrsum_##RS[kk], vl) ;		\
	  }														\
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

#define FILTER_OFFSET(k,c,r,s) ( gKernGroupOffset + filter_index<FLAYOUT>(k,c,r,s, inChannelGroup, gOutChannelGroup, gKernHeight, gKernWidth) )

    if( remain ){
      vrsum0_r0s0 = _vel_vfsums_vvl(vrsum0_r0s0, VLEN) ;
      _vel_vstu_vssl(vrsum0_r0s0, 4, pGKernel+FILTER_OFFSET(k, c, 0, 0), 1) ;
      vrsum0_r0s1 = _vel_vfsums_vvl(vrsum0_r0s1, VLEN) ;
      _vel_vstu_vssl(vrsum0_r0s1, 4, pGKernel+FILTER_OFFSET(k, c, 0, 1), 1) ;
      vrsum0_r0s2 = _vel_vfsums_vvl(vrsum0_r0s2, VLEN) ;
      _vel_vstu_vssl(vrsum0_r0s2, 4, pGKernel+FILTER_OFFSET(k, c, 0, 2), 1) ;
      vrsum0_r1s0 = _vel_vfsums_vvl(vrsum0_r1s0, VLEN) ;
      _vel_vstu_vssl(vrsum0_r1s0, 4, pGKernel+FILTER_OFFSET(k, c, 1, 0), 1) ;
      vrsum0_r1s1 = _vel_vfsums_vvl(vrsum0_r1s1, VLEN) ;
      _vel_vstu_vssl(vrsum0_r1s1, 4, pGKernel+FILTER_OFFSET(k, c, 1, 1), 1) ;
      vrsum0_r1s2 = _vel_vfsums_vvl(vrsum0_r1s2, VLEN) ;
      _vel_vstu_vssl(vrsum0_r1s2, 4, pGKernel+FILTER_OFFSET(k, c, 1, 2), 1) ;
      vrsum0_r2s0 = _vel_vfsums_vvl(vrsum0_r2s0, VLEN) ;
      _vel_vstu_vssl(vrsum0_r2s0, 4, pGKernel+FILTER_OFFSET(k, c, 2, 0), 1) ;
      vrsum0_r2s1 = _vel_vfsums_vvl(vrsum0_r2s1, VLEN) ;
      _vel_vstu_vssl(vrsum0_r2s1, 4, pGKernel+FILTER_OFFSET(k, c, 2, 1), 1) ;
      vrsum0_r2s2 = _vel_vfsums_vvl(vrsum0_r2s2, VLEN) ;
      _vel_vstu_vssl(vrsum0_r2s2, 4, pGKernel+FILTER_OFFSET(k, c, 2, 2), 1) ;
    }

#pragma clang loop unroll(full)
    for(int64_t kk=0; kk<nPacked; kk++) {
      int64_t K = k+2*kk+remain ;
      __vr vrsum0_r0s0 = _vel_vfsums_vvl(vrsum_r0s0[kk], VLEN) ;
      __vr vrsum1_r0s0 = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsum_r0s0[kk], 32, VLEN), VLEN);
      _vel_vstu_vssl(vrsum0_r0s0, 4, pGKernel+FILTER_OFFSET(K  , c, 0, 0), 1) ;
      _vel_vstu_vssl(vrsum1_r0s0, 4, pGKernel+FILTER_OFFSET(K+1, c, 0, 0), 1) ;
      __vr vrsum0_r0s1 = _vel_vfsums_vvl(vrsum_r0s1[kk], VLEN) ;
      __vr vrsum1_r0s1 = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsum_r0s1[kk], 32, VLEN), VLEN);
      _vel_vstu_vssl(vrsum0_r0s1, 4, pGKernel+FILTER_OFFSET(K  , c, 0, 1), 1) ;
      _vel_vstu_vssl(vrsum1_r0s1, 4, pGKernel+FILTER_OFFSET(K+1, c, 0, 1), 1) ;
      __vr vrsum0_r0s2 = _vel_vfsums_vvl(vrsum_r0s2[kk], VLEN) ;
      __vr vrsum1_r0s2 = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsum_r0s2[kk], 32, VLEN), VLEN);
      _vel_vstu_vssl(vrsum0_r0s2, 4, pGKernel+FILTER_OFFSET(K  , c, 0, 2), 1) ;
      _vel_vstu_vssl(vrsum1_r0s2, 4, pGKernel+FILTER_OFFSET(K+1, c, 0, 2), 1) ;
      __vr vrsum0_r1s0 = _vel_vfsums_vvl(vrsum_r1s0[kk], VLEN) ;
      __vr vrsum1_r1s0 = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsum_r1s0[kk], 32, VLEN), VLEN);
      _vel_vstu_vssl(vrsum0_r1s0, 4, pGKernel+FILTER_OFFSET(K  , c, 1, 0), 1) ;
      _vel_vstu_vssl(vrsum1_r1s0, 4, pGKernel+FILTER_OFFSET(K+1, c, 1, 0), 1) ;
      __vr vrsum0_r1s1 = _vel_vfsums_vvl(vrsum_r1s1[kk], VLEN) ;
      __vr vrsum1_r1s1 = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsum_r1s1[kk], 32, VLEN), VLEN);
      _vel_vstu_vssl(vrsum0_r1s1, 4, pGKernel+FILTER_OFFSET(K  , c, 1, 1), 1) ;
      _vel_vstu_vssl(vrsum1_r1s1, 4, pGKernel+FILTER_OFFSET(K+1, c, 1, 1), 1) ;
      __vr vrsum0_r1s2 = _vel_vfsums_vvl(vrsum_r1s2[kk], VLEN) ;
      __vr vrsum1_r1s2 = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsum_r1s2[kk], 32, VLEN), VLEN);
      _vel_vstu_vssl(vrsum0_r1s2, 4, pGKernel+FILTER_OFFSET(K  , c, 1, 2), 1) ;
      _vel_vstu_vssl(vrsum1_r1s2, 4, pGKernel+FILTER_OFFSET(K+1, c, 1, 2), 1) ;
      __vr vrsum0_r2s0 = _vel_vfsums_vvl(vrsum_r2s0[kk], VLEN) ;
      __vr vrsum1_r2s0 = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsum_r2s0[kk], 32, VLEN), VLEN);
      _vel_vstu_vssl(vrsum0_r2s0, 4, pGKernel+FILTER_OFFSET(K  , c, 2, 0), 1) ;
      _vel_vstu_vssl(vrsum1_r2s0, 4, pGKernel+FILTER_OFFSET(K+1, c, 2, 0), 1) ;
      __vr vrsum0_r2s1 = _vel_vfsums_vvl(vrsum_r2s1[kk], VLEN) ;
      __vr vrsum1_r2s1 = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsum_r2s1[kk], 32, VLEN), VLEN);
      _vel_vstu_vssl(vrsum0_r2s1, 4, pGKernel+FILTER_OFFSET(K  , c, 2, 1), 1) ;
      _vel_vstu_vssl(vrsum1_r2s1, 4, pGKernel+FILTER_OFFSET(K+1, c, 2, 1), 1) ;
      __vr vrsum0_r2s2 = _vel_vfsums_vvl(vrsum_r2s2[kk], VLEN) ;
      __vr vrsum1_r2s2 = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsum_r2s2[kk], 32, VLEN), VLEN);
      _vel_vstu_vssl(vrsum0_r2s2, 4, pGKernel+FILTER_OFFSET(K  , c, 2, 2), 1) ;
      _vel_vstu_vssl(vrsum1_r2s2, 4, pGKernel+FILTER_OFFSET(K+1, c, 2, 2), 1) ;
    }

#undef SUM_AND_STORE

  } // inChannel
}


template<filterLayout_t FLAYOUT, int NUMKERNEL>
static inline void func_ialigned(
    const float * __restrict__ pIn,
    const float * __restrict__ pGOut,
    float * __restrict__ const pGKernel,
    const int64_t gOutChannel,
    const int64_t gOutWidth,
    const int64_t gOutHeight,
    const int64_t inChannel,
    const int64_t inWidth,
    const int64_t inHeight,
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
    const int64_t inGroupOffset,
    const int64_t gOutGroupOffset,
    const int64_t gKernGroupOffset,
    const int64_t batch,
    const int64_t k,
    const int64_t nY,
    const __vr vri,
    const __vr vrj
)
{
  const int64_t remain  = NUMKERNEL & 0x1 ;
  const int64_t nPacked = NUMKERNEL >> 1 ;

  __vr vrin_ptr_ost = _vel_vsfa_vvssl(_vel_vaddul_vvvl(vrj, _vel_vmulul_vsvl(inWidth,vri, nY*gOutWidth), nY*gOutWidth), 2, (uint64_t)(pIn + inGroupOffset), nY*gOutWidth) ;

  __vr vrw_s0 = _vel_vaddsl_vsvl(-1,  vrj, nY*gOutWidth) ;
  __vm256 vmw_s0 =  _vel_vfmklge_mvl(vrw_s0, nY*gOutWidth) ;						// condition(0 <= w)

  for (int64_t c=0; c<inChannelGroup; c++) {

    __vr vrsum0_r0s0 = _vel_vbrds_vsl(0.f, VLEN) ;
    __vr vrsum0_r0s1 = _vel_vbrds_vsl(0.f, VLEN) ;
    __vr vrsum0_r0s2 = _vel_vbrds_vsl(0.f, VLEN) ;
    __vr vrsum0_r1s0 = _vel_vbrds_vsl(0.f, VLEN) ;
    __vr vrsum0_r1s1 = _vel_vbrds_vsl(0.f, VLEN) ;
    __vr vrsum0_r1s2 = _vel_vbrds_vsl(0.f, VLEN) ;
    __vr vrsum0_r2s0 = _vel_vbrds_vsl(0.f, VLEN) ;
    __vr vrsum0_r2s1 = _vel_vbrds_vsl(0.f, VLEN) ;
    __vr vrsum0_r2s2 = _vel_vbrds_vsl(0.f, VLEN) ;


    __vr vrsum_r0s0[nPacked] ;
    __vr vrsum_r0s1[nPacked] ;
    __vr vrsum_r0s2[nPacked] ;
    __vr vrsum_r1s0[nPacked] ;
    __vr vrsum_r1s1[nPacked] ;
    __vr vrsum_r1s2[nPacked] ;
    __vr vrsum_r2s0[nPacked] ;
    __vr vrsum_r2s1[nPacked] ;
    __vr vrsum_r2s2[nPacked] ;
#pragma clang loop unroll(full)
    for(int64_t kk=0; kk<nPacked; kk++) {
      vrsum_r0s0[kk] = _vel_pvbrd_vsl(0UL, VLEN) ;
      vrsum_r0s1[kk] = _vel_pvbrd_vsl(0UL, VLEN) ;
      vrsum_r0s2[kk] = _vel_pvbrd_vsl(0UL, VLEN) ;
      vrsum_r1s0[kk] = _vel_pvbrd_vsl(0UL, VLEN) ;
      vrsum_r1s1[kk] = _vel_pvbrd_vsl(0UL, VLEN) ;
      vrsum_r1s2[kk] = _vel_pvbrd_vsl(0UL, VLEN) ;
      vrsum_r2s0[kk] = _vel_pvbrd_vsl(0UL, VLEN) ;
      vrsum_r2s1[kk] = _vel_pvbrd_vsl(0UL, VLEN) ;
      vrsum_r2s2[kk] = _vel_pvbrd_vsl(0UL, VLEN) ;
    }

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

	const int64_t outIndex = gOutGroupOffset + ((n * gOutChannel + k) * gOutHeight + y) * gOutWidth ;

	__vr vrgout[NUMKERNEL] ;
#pragma clang loop unroll(full)
	for(int64_t kk=0; kk<NUMKERNEL; kk++) {
	  vrgout[kk] = _vel_vldu_vssl(4, pGOut+outIndex+kk*gOutHeight*gOutWidth, vl) ;
	}

	__vr vrgoutp[NUMKERNEL]  ;
#pragma clang loop unroll(full)
	for(int64_t kk=0; kk<nPacked; kk++) {
	  vrgoutp[kk] = _vel_vshf_vvvsl(vrgout[2*kk+remain], vrgout[2*kk+remain+1], VE_VSHUFFLE_YUZU, vl) ;
	}

#define VFADD(VRINP, RS)												\
	{														\
	  if( remain ) {												\
	    vrsum0_##RS = _vel_vfmads_vvvvvl(vrsum0_##RS, vrinP_##RS, vrgout[0], vrsum0_##RS, vl) ;			\
	  }														\
	  _Pragma("clang loop unroll(full)")										\
	  for(int64_t kk=0; kk<nPacked; kk++) {										\
	    vrsum_##RS[kk] = _vel_pvfmad_vvvvvl(vrsum_##RS[kk], vrinP_##RS, vrgoutp[kk], vrsum_##RS[kk], vl) ;		\
	  }														\
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

#define FILTER_OFFSET(k,c,r,s) ( gKernGroupOffset + filter_index<FLAYOUT>(k,c,r,s, inChannelGroup, gOutChannelGroup, gKernHeight, gKernWidth) )

    if( remain ){
      vrsum0_r0s0 = _vel_vfsums_vvml(vrsum0_r0s0, vmw_s0, VLEN) ;
      _vel_vstu_vssl(vrsum0_r0s0, 4, pGKernel+FILTER_OFFSET(k, c, 0, 0), 1) ;
      vrsum0_r0s1 = _vel_vfsums_vvl(vrsum0_r0s1, VLEN) ;
      _vel_vstu_vssl(vrsum0_r0s1, 4, pGKernel+FILTER_OFFSET(k, c, 0, 1), 1) ;
      vrsum0_r0s2 = _vel_vfsums_vvl(vrsum0_r0s2, VLEN) ;
      _vel_vstu_vssl(vrsum0_r0s2, 4, pGKernel+FILTER_OFFSET(k, c, 0, 2), 1) ;
      vrsum0_r1s0 = _vel_vfsums_vvml(vrsum0_r1s0, vmw_s0, VLEN) ;
      _vel_vstu_vssl(vrsum0_r1s0, 4, pGKernel+FILTER_OFFSET(k, c, 1, 0), 1) ;
      vrsum0_r1s1 = _vel_vfsums_vvl(vrsum0_r1s1, VLEN) ;
      _vel_vstu_vssl(vrsum0_r1s1, 4, pGKernel+FILTER_OFFSET(k, c, 1, 1), 1) ;
      vrsum0_r1s2 = _vel_vfsums_vvl(vrsum0_r1s2, VLEN) ;
      _vel_vstu_vssl(vrsum0_r1s2, 4, pGKernel+FILTER_OFFSET(k, c, 1, 2), 1) ;
      vrsum0_r2s0 = _vel_vfsums_vvml(vrsum0_r2s0, vmw_s0, VLEN) ;
      _vel_vstu_vssl(vrsum0_r2s0, 4, pGKernel+FILTER_OFFSET(k, c, 2, 0), 1) ;
      vrsum0_r2s1 = _vel_vfsums_vvl(vrsum0_r2s1, VLEN) ;
      _vel_vstu_vssl(vrsum0_r2s1, 4, pGKernel+FILTER_OFFSET(k, c, 2, 1), 1) ;
      vrsum0_r2s2 = _vel_vfsums_vvl(vrsum0_r2s2, VLEN) ;
      _vel_vstu_vssl(vrsum0_r2s2, 4, pGKernel+FILTER_OFFSET(k, c, 2, 2), 1) ;
    }

#pragma clang loop unroll(full)
    for(int64_t kk=0; kk<nPacked; kk++) {
      int64_t K = k+2*kk+remain ;
      __vr vrsum0_r0s0 = _vel_vfsums_vvml(vrsum_r0s0[kk], vmw_s0, VLEN) ;
      __vr vrsum1_r0s0 = _vel_vfsums_vvml(_vel_vsll_vvsl(vrsum_r0s0[kk], 32, VLEN), vmw_s0, VLEN);
      _vel_vstu_vssl(vrsum0_r0s0, 4, pGKernel+FILTER_OFFSET(K  , c, 0, 0), 1) ;
      _vel_vstu_vssl(vrsum1_r0s0, 4, pGKernel+FILTER_OFFSET(K+1, c, 0, 0), 1) ;
      __vr vrsum0_r0s1 = _vel_vfsums_vvl(vrsum_r0s1[kk], VLEN) ;
      __vr vrsum1_r0s1 = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsum_r0s1[kk], 32, VLEN), VLEN);
      _vel_vstu_vssl(vrsum0_r0s1, 4, pGKernel+FILTER_OFFSET(K  , c, 0, 1), 1) ;
      _vel_vstu_vssl(vrsum1_r0s1, 4, pGKernel+FILTER_OFFSET(K+1, c, 0, 1), 1) ;
      __vr vrsum0_r0s2 = _vel_vfsums_vvl(vrsum_r0s2[kk], VLEN) ;
      __vr vrsum1_r0s2 = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsum_r0s2[kk], 32, VLEN), VLEN);
      _vel_vstu_vssl(vrsum0_r0s2, 4, pGKernel+FILTER_OFFSET(K  , c, 0, 2), 1) ;
      _vel_vstu_vssl(vrsum1_r0s2, 4, pGKernel+FILTER_OFFSET(K+1, c, 0, 2), 1) ;
      __vr vrsum0_r1s0 = _vel_vfsums_vvml(vrsum_r1s0[kk], vmw_s0, VLEN) ;
      __vr vrsum1_r1s0 = _vel_vfsums_vvml(_vel_vsll_vvsl(vrsum_r1s0[kk], 32, VLEN), vmw_s0, VLEN);
      _vel_vstu_vssl(vrsum0_r1s0, 4, pGKernel+FILTER_OFFSET(K  , c, 1, 0), 1) ;
      _vel_vstu_vssl(vrsum1_r1s0, 4, pGKernel+FILTER_OFFSET(K+1, c, 1, 0), 1) ;
      __vr vrsum0_r1s1 = _vel_vfsums_vvl(vrsum_r1s1[kk], VLEN) ;
      __vr vrsum1_r1s1 = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsum_r1s1[kk], 32, VLEN), VLEN);
      _vel_vstu_vssl(vrsum0_r1s1, 4, pGKernel+FILTER_OFFSET(K  , c, 1, 1), 1) ;
      _vel_vstu_vssl(vrsum1_r1s1, 4, pGKernel+FILTER_OFFSET(K+1, c, 1, 1), 1) ;
      __vr vrsum0_r1s2 = _vel_vfsums_vvl(vrsum_r1s2[kk], VLEN) ;
      __vr vrsum1_r1s2 = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsum_r1s2[kk], 32, VLEN), VLEN);
      _vel_vstu_vssl(vrsum0_r1s2, 4, pGKernel+FILTER_OFFSET(K  , c, 1, 2), 1) ;
      _vel_vstu_vssl(vrsum1_r1s2, 4, pGKernel+FILTER_OFFSET(K+1, c, 1, 2), 1) ;
      __vr vrsum0_r2s0 = _vel_vfsums_vvml(vrsum_r2s0[kk], vmw_s0, VLEN) ;
      __vr vrsum1_r2s0 = _vel_vfsums_vvml(_vel_vsll_vvsl(vrsum_r2s0[kk], 32, VLEN), vmw_s0, VLEN);
      _vel_vstu_vssl(vrsum0_r2s0, 4, pGKernel+FILTER_OFFSET(K  , c, 2, 0), 1) ;
      _vel_vstu_vssl(vrsum1_r2s0, 4, pGKernel+FILTER_OFFSET(K+1, c, 2, 0), 1) ;
      __vr vrsum0_r2s1 = _vel_vfsums_vvl(vrsum_r2s1[kk], VLEN) ;
      __vr vrsum1_r2s1 = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsum_r2s1[kk], 32, VLEN), VLEN);
      _vel_vstu_vssl(vrsum0_r2s1, 4, pGKernel+FILTER_OFFSET(K  , c, 2, 1), 1) ;
      _vel_vstu_vssl(vrsum1_r2s1, 4, pGKernel+FILTER_OFFSET(K+1, c, 2, 1), 1) ;
      __vr vrsum0_r2s2 = _vel_vfsums_vvl(vrsum_r2s2[kk], VLEN) ;
      __vr vrsum1_r2s2 = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsum_r2s2[kk], 32, VLEN), VLEN);
      _vel_vstu_vssl(vrsum0_r2s2, 4, pGKernel+FILTER_OFFSET(K  , c, 2, 2), 1) ;
      _vel_vstu_vssl(vrsum1_r2s2, 4, pGKernel+FILTER_OFFSET(K+1, c, 2, 2), 1) ;
    }

#undef SUM_AND_STORE

  } // inChannel
}


template<filterLayout_t FLAYOUT>
static inline void convloop(
    const float * __restrict__ pIn,
    const float * __restrict__ pGOut,
    float * __restrict__ const pGKernel,
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
    const int64_t inChannelGroup,
    const int64_t gOutChannelGroup,
    const int64_t beginOChannel,
    const int64_t nOChannel,
    const int64_t beginGroup,
    const int64_t nGroup,
    const int64_t strideWidth,
    const int64_t strideHeight,
    const int64_t padWidth,
    const int64_t padHeight,
    const int64_t dilationWidth,
    const int64_t dilationHeight
)
{
  const int64_t nY = VLEN / gOutWidth ;

  __vr vrseq = _vel_vseq_vl(nY*gOutWidth) ;			// xy
  __vr vry  = _vel_vdivsl_vvsl(vrseq, gOutWidth, nY*gOutWidth) ;
  __vr vrx  = _vel_vsubsl_vvvl(vrseq, _vel_vmulul_vsvl(gOutWidth,vry, nY*gOutWidth), nY*gOutWidth) ;
  __vr vri  = _vel_vmulsl_vsvl(strideHeight, vry, nY*gOutWidth) ;
  __vr vrj  = _vel_vmulsl_vsvl(strideWidth,  vrx, nY*gOutWidth) ;

  const int64_t ialigned = ((((uint64_t) pIn) & 0x7) == 0) && ((inWidth & 0x1) == 0) ;

  for (int64_t g = beginGroup; g < nGroup; g++) {
    int64_t inGroupOffset    = g * inChannelGroup  * inHeight  * inWidth;
    int64_t gOutGroupOffset  = g * gOutChannelGroup * gOutHeight * gOutWidth;
    int64_t gKernGroupOffset = g * gOutChannelGroup * inChannelGroup * gKernHeight * gKernWidth;

    int64_t k=0;

    if( ialigned ) {
      const int64_t remain = nOChannel & 0x7 ;

      switch(remain) {
      case 1:
	func_ialigned<FLAYOUT, 1>(pIn, pGOut, pGKernel,
	   gOutChannel, gOutWidth, gOutHeight,
	   inChannel, inWidth, inHeight,
	   gKernWidth, gKernHeight,
	   strideWidth, strideHeight,
	   padWidth, padHeight,
	   dilationWidth, dilationHeight,
	   inChannelGroup, gOutChannelGroup,
	   inGroupOffset, gOutGroupOffset, gKernGroupOffset,
	   batch, beginOChannel + k,
	   nY, vri, vrj ) ;
	k+=1 ;
	break ;
      case 2:
	func_ialigned<FLAYOUT, 2>(pIn, pGOut, pGKernel,
	   gOutChannel, gOutWidth, gOutHeight,
	   inChannel, inWidth, inHeight,
	   gKernWidth, gKernHeight,
	   strideWidth, strideHeight,
	   padWidth, padHeight,
	   dilationWidth, dilationHeight,
	   inChannelGroup, gOutChannelGroup,
	   inGroupOffset, gOutGroupOffset, gKernGroupOffset,
	   batch, beginOChannel + k,
	   nY, vri, vrj ) ;
	k+=2 ;
	break ;
      case 3:
	func_ialigned<FLAYOUT, 3>(pIn, pGOut, pGKernel,
	   gOutChannel, gOutWidth, gOutHeight,
	   inChannel, inWidth, inHeight,
	   gKernWidth, gKernHeight,
	   strideWidth, strideHeight,
	   padWidth, padHeight,
	   dilationWidth, dilationHeight,
	   inChannelGroup, gOutChannelGroup,
	   inGroupOffset, gOutGroupOffset, gKernGroupOffset,
	   batch, beginOChannel + k,
	   nY, vri, vrj ) ;
	k+=3 ;
	break ;
      case 4:
	func_ialigned<FLAYOUT, 4>(pIn, pGOut, pGKernel,
	   gOutChannel, gOutWidth, gOutHeight,
	   inChannel, inWidth, inHeight,
	   gKernWidth, gKernHeight,
	   strideWidth, strideHeight,
	   padWidth, padHeight,
	   dilationWidth, dilationHeight,
	   inChannelGroup, gOutChannelGroup,
	   inGroupOffset, gOutGroupOffset, gKernGroupOffset,
	   batch, beginOChannel + k,
	   nY, vri, vrj ) ;
	k+=4 ;
	break ;
      case 5:
	func_ialigned<FLAYOUT, 5>(pIn, pGOut, pGKernel,
	   gOutChannel, gOutWidth, gOutHeight,
	   inChannel, inWidth, inHeight,
	   gKernWidth, gKernHeight,
	   strideWidth, strideHeight,
	   padWidth, padHeight,
	   dilationWidth, dilationHeight,
	   inChannelGroup, gOutChannelGroup,
	   inGroupOffset, gOutGroupOffset, gKernGroupOffset,
	   batch, beginOChannel + k,
	   nY, vri, vrj ) ;
	k+=5 ;
	break ;
      case 6:
	func_ialigned<FLAYOUT, 6>(pIn, pGOut, pGKernel,
	   gOutChannel, gOutWidth, gOutHeight,
	   inChannel, inWidth, inHeight,
	   gKernWidth, gKernHeight,
	   strideWidth, strideHeight,
	   padWidth, padHeight,
	   dilationWidth, dilationHeight,
	   inChannelGroup, gOutChannelGroup,
	   inGroupOffset, gOutGroupOffset, gKernGroupOffset,
	   batch, beginOChannel + k,
	   nY, vri, vrj ) ;
	k+=6 ;
	break ;
      case 7:
	func_ialigned<FLAYOUT, 7>(pIn, pGOut, pGKernel,
	   gOutChannel, gOutWidth, gOutHeight,
	   inChannel, inWidth, inHeight,
	   gKernWidth, gKernHeight,
	   strideWidth, strideHeight,
	   padWidth, padHeight,
	   dilationWidth, dilationHeight,
	   inChannelGroup, gOutChannelGroup,
	   inGroupOffset, gOutGroupOffset, gKernGroupOffset,
	   batch, beginOChannel + k,
	   nY, vri, vrj ) ;
	k+=7 ;
	break ;
      default :
	break ;
      }
      for (; k<nOChannel; ) {
	func_ialigned<FLAYOUT, 8>(pIn, pGOut, pGKernel,
	   gOutChannel, gOutWidth, gOutHeight,
	   inChannel, inWidth, inHeight,
	   gKernWidth, gKernHeight,
	   strideWidth, strideHeight,
	   padWidth, padHeight,
	   dilationWidth, dilationHeight,
	   inChannelGroup, gOutChannelGroup,
	   inGroupOffset, gOutGroupOffset, gKernGroupOffset,
	   batch, beginOChannel + k,
	   nY, vri, vrj ) ;
	k+=8 ;
      } // gOutChannel
    }
    else {
      const int64_t remain = nOChannel & 0x7 ;

      switch(remain) {
      case 1:
	func<FLAYOUT, 1>(pIn, pGOut, pGKernel,
	   gOutChannel, gOutWidth, gOutHeight,
	   inChannel, inWidth, inHeight,
	   gKernWidth, gKernHeight,
	   strideWidth, strideHeight,
	   padWidth, padHeight,
	   dilationWidth, dilationHeight,
	   inChannelGroup, gOutChannelGroup,
	   inGroupOffset, gOutGroupOffset, gKernGroupOffset,
	   batch, beginOChannel + k,
	   nY, vri, vrj ) ;
	k+=1 ;
	break ;
      case 2:
	func<FLAYOUT, 2>(pIn, pGOut, pGKernel,
	   gOutChannel, gOutWidth, gOutHeight,
	   inChannel, inWidth, inHeight,
	   gKernWidth, gKernHeight,
	   strideWidth, strideHeight,
	   padWidth, padHeight,
	   dilationWidth, dilationHeight,
	   inChannelGroup, gOutChannelGroup,
	   inGroupOffset, gOutGroupOffset, gKernGroupOffset,
	   batch, beginOChannel + k,
	   nY, vri, vrj ) ;
	k+=2 ;
	break ;
      case 3:
	func<FLAYOUT, 3>(pIn, pGOut, pGKernel,
	   gOutChannel, gOutWidth, gOutHeight,
	   inChannel, inWidth, inHeight,
	   gKernWidth, gKernHeight,
	   strideWidth, strideHeight,
	   padWidth, padHeight,
	   dilationWidth, dilationHeight,
	   inChannelGroup, gOutChannelGroup,
	   inGroupOffset, gOutGroupOffset, gKernGroupOffset,
	   batch, beginOChannel + k,
	   nY, vri, vrj ) ;
	k+=3 ;
	break ;
      case 4:
	func<FLAYOUT, 4>(pIn, pGOut, pGKernel,
	   gOutChannel, gOutWidth, gOutHeight,
	   inChannel, inWidth, inHeight,
	   gKernWidth, gKernHeight,
	   strideWidth, strideHeight,
	   padWidth, padHeight,
	   dilationWidth, dilationHeight,
	   inChannelGroup, gOutChannelGroup,
	   inGroupOffset, gOutGroupOffset, gKernGroupOffset,
	   batch, beginOChannel + k,
	   nY, vri, vrj ) ;
	k+=4 ;
	break ;
      case 5:
	func<FLAYOUT, 5>(pIn, pGOut, pGKernel,
	   gOutChannel, gOutWidth, gOutHeight,
	   inChannel, inWidth, inHeight,
	   gKernWidth, gKernHeight,
	   strideWidth, strideHeight,
	   padWidth, padHeight,
	   dilationWidth, dilationHeight,
	   inChannelGroup, gOutChannelGroup,
	   inGroupOffset, gOutGroupOffset, gKernGroupOffset,
	   batch, beginOChannel + k,
	   nY, vri, vrj ) ;
	k+=5 ;
	break ;
      case 6:
	func<FLAYOUT, 6>(pIn, pGOut, pGKernel,
	   gOutChannel, gOutWidth, gOutHeight,
	   inChannel, inWidth, inHeight,
	   gKernWidth, gKernHeight,
	   strideWidth, strideHeight,
	   padWidth, padHeight,
	   dilationWidth, dilationHeight,
	   inChannelGroup, gOutChannelGroup,
	   inGroupOffset, gOutGroupOffset, gKernGroupOffset,
	   batch, beginOChannel + k,
	   nY, vri, vrj ) ;
	k+=6 ;
	break ;
      case 7:
	func<FLAYOUT, 7>(pIn, pGOut, pGKernel,
	   gOutChannel, gOutWidth, gOutHeight,
	   inChannel, inWidth, inHeight,
	   gKernWidth, gKernHeight,
	   strideWidth, strideHeight,
	   padWidth, padHeight,
	   dilationWidth, dilationHeight,
	   inChannelGroup, gOutChannelGroup,
	   inGroupOffset, gOutGroupOffset, gKernGroupOffset,
	   batch, beginOChannel + k,
	   nY, vri, vrj ) ;
	k+=7 ;
	break ;
      default :
	break ;
      }
      for (; k<nOChannel; ) {
	func<FLAYOUT, 8>(pIn, pGOut, pGKernel,
	   gOutChannel, gOutWidth, gOutHeight,
	   inChannel, inWidth, inHeight,
	   gKernWidth, gKernHeight,
	   strideWidth, strideHeight,
	   padWidth, padHeight,
	   dilationWidth, dilationHeight,
	   inChannelGroup, gOutChannelGroup,
	   inGroupOffset, gOutGroupOffset, gKernGroupOffset,
	   batch, beginOChannel + k,
	   nY, vri, vrj ) ;
	k+=8 ;
      } // gOutChannel
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
    const int64_t				beginOChannel,
    const int64_t				nOChannel
#ifdef VEDNN_OMP_GROUP_PARALLEL
    ,
    const int64_t				beginGroup,
    const int64_t				nGroup
#endif
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

  const int64_t filter_layout = pParamGradKernel->layout ;

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
#ifndef VEDNN_OMP_GROUP_PARALLEL
  const int64_t beginGroup = 0 ;
  const int64_t nGroup     = group ;
#endif

  if( filter_layout == VEDNN_FILTER_LAYOUT_NCHW) {
    convloop<VEDNN_FILTER_LAYOUT_NCHW>(pIn, pGOut, pGKernel,
	       batch, group,
	       inChannel, inWidth, inHeight,
	       gOutChannel, gOutWidth, gOutHeight,
	       gKernWidth, gKernHeight,
	       inChannelGroup, gOutChannelGroup,
	       beginOChannel, nOChannel,
	       beginGroup, nGroup,
	       strideWidth, strideHeight,
	       padWidth, padHeight,
	       dilationWidth, dilationHeight) ;
  }
  else {
    convloop<VEDNN_FILTER_LAYOUT_HWCN>(pIn, pGOut, pGKernel,
	       batch, group,
	       inChannel, inWidth, inHeight,
	       gOutChannel, gOutWidth, gOutHeight,
	       gKernWidth, gKernHeight,
	       inChannelGroup, gOutChannelGroup,
	       beginOChannel, nOChannel,
	       beginGroup, nGroup,
	       strideWidth, strideHeight,
	       padWidth, padHeight,
	       dilationWidth, dilationHeight) ;
  }

  return VEDNN_SUCCESS;
}
