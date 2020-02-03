#include <stdint.h>
#include <stdlib.h>

#include "vednn.h"

#include "velintrin.h"
#define VLEN	(256)

template<int NUMKERNEL, bool ADDBIAS>
static inline void func_hwcn(
  const float * __restrict__ pIn,
  const float * __restrict__ pKernel,
  const float * __restrict__ pBias,
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
  const int64_t biasGroupOffset,
  const int64_t kernGroupOffset,
  const int64_t nY,
  const __vr    vri,
  const __vr    vrj,
  const int64_t n,
  const int64_t k
)
{
  int64_t outIndex = outGroupOffset + (n * outChannel + k  ) * outHeight*outWidth ;

  const int64_t remain  = NUMKERNEL & 0x1 ;
  const int64_t nPacked = NUMKERNEL >> 1 ;

  __vr vrw_s0 = _vel_vaddsl_vsvl(0,vrj, nY*outWidth) ;
  __vr vrw_s1 = _vel_vaddsl_vsvl(1,vrj, nY*outWidth) ;
  __vr vrw_s2 = _vel_vaddsl_vsvl(2,vrj, nY*outWidth) ;
  __vr vrw_s3 = _vel_vaddsl_vsvl(3,vrj, nY*outWidth) ;

  __vm256 vmw0_s0 = _vel_vfmklge_mvl(vrw_s0, nY*outWidth) ;						// condition(0 <= w)
  __vm256 vmw1_s0 = _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(inWidth,vrw_s0, nY*outWidth), nY*outWidth) ;	// condition(w < inWidth)
  __vm256 vmw_s0  = _vel_andm_mmm(vmw0_s0, vmw1_s0) ;

  __vm256 vmw0_s1 = _vel_vfmklge_mvl(vrw_s1, nY*outWidth) ;						// condition(0 <= w)
  __vm256 vmw1_s1 = _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(inWidth,vrw_s1, nY*outWidth), nY*outWidth) ;	// condition(w < inWidth)
  __vm256 vmw_s1  = _vel_andm_mmm(vmw0_s1, vmw1_s1) ;

  __vm256 vmw0_s2 = _vel_vfmklge_mvl(vrw_s2, nY*outWidth) ;						// condition(0 <= w)
  __vm256 vmw1_s2 = _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(inWidth,vrw_s2, nY*outWidth), nY*outWidth) ;	// condition(w < inWidth)
  __vm256 vmw_s2  = _vel_andm_mmm(vmw0_s2, vmw1_s2) ;

  __vm256 vmw0_s3 = _vel_vfmklge_mvl(vrw_s3, nY*outWidth) ;						// condition(0 <= w)
  __vm256 vmw1_s3 = _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(inWidth,vrw_s3, nY*outWidth), nY*outWidth) ;	// condition(w < inWidth)
  __vm256 vmw_s3  = _vel_andm_mmm(vmw0_s3, vmw1_s3) ;

  const float   bias0  = ADDBIAS ?  pBias[biasGroupOffset+k+ 0] : 0.f ;
  int64_t bias[nPacked] ;
#pragma clang loop unroll(full)
  for(int64_t kk=0; kk<nPacked; kk++) bias[kk] = ADDBIAS ?  _vel_pack_f32p(pBias+biasGroupOffset+k+2*kk+remain, pBias+biasGroupOffset+k+2*kk+remain+1) : 0UL ;

  for (int64_t y=0; y<outHeight; y+=nY)
  {
    const int64_t vl = outWidth * (outHeight - y < nY ? outHeight - y : nY) ;
    const int64_t op = y * outWidth ;

    __vr vrsum0_s0  = _vel_vbrds_vsl(0.f, vl) ;
    __vr vrsum0_s1  = _vel_vbrds_vsl(0.f, vl) ;
    __vr vrsum0_s2  = _vel_vbrds_vsl(0.f, vl) ;
    __vr vrsum0_s3  = _vel_vbrds_vsl(0.f, vl) ;
    __vr vrsum_s0[nPacked] ;
    __vr vrsum_s1[nPacked] ;
    __vr vrsum_s2[nPacked] ;
    __vr vrsum_s3[nPacked] ;
#pragma clang loop unroll(full)
    for(int64_t kk=0; kk<nPacked; kk++) {
      vrsum_s0[kk] = _vel_pvbrd_vsl(0UL, vl) ;
      vrsum_s1[kk] = _vel_pvbrd_vsl(0UL, vl) ;
      vrsum_s2[kk] = _vel_pvbrd_vsl(0UL, vl) ;
      vrsum_s3[kk] = _vel_pvbrd_vsl(0UL, vl) ;
    }

    __vr vrh_r0 = _vel_vaddsl_vsvl(0+y*2, vri, vl) ;
    __vr vrh_r1 = _vel_vaddsl_vsvl(1+y*2, vri, vl) ;
    __vr vrh_r2 = _vel_vaddsl_vsvl(2+y*2, vri, vl) ;
    __vr vrh_r3 = _vel_vaddsl_vsvl(3+y*2, vri, vl) ;

    __vm256 vmh0_r0 = _vel_vfmklge_mvl(vrh_r0, vl) ;				// condition(0 <= h)
    __vm256 vmh1_r0 = _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(inHeight,vrh_r0, vl), vl) ;	// condition(h < inHeight)
    __vm256 vmh_r0  = _vel_andm_mmm(vmh0_r0, vmh1_r0) ;

    __vm256 vmh0_r1 = _vel_vfmklge_mvl(vrh_r1, vl) ;				// condition(0 <= h)
    __vm256 vmh1_r1 = _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(inHeight,vrh_r1, vl), vl) ;	// condition(h < inHeight)
    __vm256 vmh_r1  = _vel_andm_mmm(vmh0_r1, vmh1_r1) ;

    __vm256 vmh0_r2 = _vel_vfmklge_mvl(vrh_r2, vl) ;				// condition(0 <= h)
    __vm256 vmh1_r2 = _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(inHeight,vrh_r2, vl), vl) ;	// condition(h < inHeight)
    __vm256 vmh_r2  = _vel_andm_mmm(vmh0_r2, vmh1_r2) ;

    __vm256 vmh0_r3 = _vel_vfmklge_mvl(vrh_r3, vl) ;				// condition(0 <= h)
    __vm256 vmh1_r3 = _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(inHeight,vrh_r3, vl), vl) ;	// condition(h < inHeight)
    __vm256 vmh_r3  = _vel_andm_mmm(vmh0_r3, vmh1_r3) ;


    for (int64_t c = 0; c < inChannelGroup; c++) {
      const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;
      __vr vrpin_r0s1 = _vel_vsfa_vvssl(_vel_vaddul_vvvl(vrw_s1, _vel_vmulul_vsvl(inWidth,vrh_r0, vl), vl), 2, (uint64_t)pInChannel, vl) ;
      __vr vrin_r0s1 = _vel_vgtu_vvssml(vrpin_r0s1, 0, 0, vmh_r0, vl) ;
      __vr vrpin_r0s2 = _vel_vsfa_vvssl(_vel_vaddul_vvvl(vrw_s2, _vel_vmulul_vsvl(inWidth,vrh_r0, vl), vl), 2, (uint64_t)pInChannel, vl) ;
      __vr vrin_r0s2 = _vel_vgtu_vvssml(vrpin_r0s2, 0, 0, vmh_r0, vl) ;

      __vr vrpin_r1s1 = _vel_vsfa_vvssl(_vel_vaddul_vvvl(vrw_s1, _vel_vmulul_vsvl(inWidth,vrh_r1, vl), vl), 2, (uint64_t)pInChannel, vl) ;
      __vr vrin_r1s1 = _vel_vgtu_vvssml(vrpin_r1s1, 0, 0, vmh_r1, vl) ;
      __vr vrpin_r1s2 = _vel_vsfa_vvssl(_vel_vaddul_vvvl(vrw_s2, _vel_vmulul_vsvl(inWidth,vrh_r1, vl), vl), 2, (uint64_t)pInChannel, vl) ;
      __vr vrin_r1s2 = _vel_vgtu_vvssml(vrpin_r1s2, 0, 0, vmh_r1, vl) ;

      __vr vrpin_r2s1 = _vel_vsfa_vvssl(_vel_vaddul_vvvl(vrw_s1, _vel_vmulul_vsvl(inWidth,vrh_r2, vl), vl), 2, (uint64_t)pInChannel, vl) ;
      __vr vrin_r2s1 = _vel_vgtu_vvssml(vrpin_r2s1, 0, 0, vmh_r2, vl) ;
      __vr vrpin_r2s2 = _vel_vsfa_vvssl(_vel_vaddul_vvvl(vrw_s2, _vel_vmulul_vsvl(inWidth,vrh_r2, vl), vl), 2, (uint64_t)pInChannel, vl) ;
      __vr vrin_r2s2 = _vel_vgtu_vvssml(vrpin_r2s2, 0, 0, vmh_r2, vl) ;

      __vr vrpin_r3s1 = _vel_vsfa_vvssl(_vel_vaddul_vvvl(vrw_s1, _vel_vmulul_vsvl(inWidth,vrh_r3, vl), vl), 2, (uint64_t)pInChannel, vl) ;
      __vr vrin_r3s1 = _vel_vgtu_vvssml(vrpin_r3s1, 0, 0, vmh_r3, vl) ;
      __vr vrpin_r3s2 = _vel_vsfa_vvssl(_vel_vaddul_vvvl(vrw_s2, _vel_vmulul_vsvl(inWidth,vrh_r3, vl), vl), 2, (uint64_t)pInChannel, vl) ;
      __vr vrin_r3s2 = _vel_vgtu_vvssml(vrpin_r3s2, 0, 0, vmh_r3, vl) ;

#define FILTER_OFFSET(k,c,r,s) ( kernGroupOffset + filter_index<VEDNN_FILTER_LAYOUT_HWCN>(k,c,r,s, inChannelGroup, outChannelGroup, 4, 4) )

#define VFADD(VRIN, VRSUM, VRSUM0, R, S)								\
      {													\
	__vr vrinP = _vel_vshf_vvvsl(VRIN, VRIN, VE_VSHUFFLE_YUZU, vl) ;				\
	if( remain ) {											\
	  const float  kerValue0  = pKernel[FILTER_OFFSET(k+ 0,c,R,S)] ;				\
	  VRSUM0 = _vel_vfmads_vvsvl(VRSUM0, kerValue0, VRIN, vl) ;					\
	}												\
	_Pragma("clang loop unroll(full)")								\
	for(int64_t kk=0; kk<nPacked; kk++) {								\
	  const uint64_t kerValue = _vel_pack_f32p(pKernel + FILTER_OFFSET(k+2*kk+remain,  c,R,S),	\
						   pKernel + FILTER_OFFSET(k+2*kk+remain+1,c,R,S)) ;	\
	  VRSUM[kk]= _vel_pvfmad_vvsvl(VRSUM[kk], kerValue, vrinP, vl) ;				\
	}												\
      }

      vrin_r0s1 = _vel_vmrg_vsvml(0.f, vrin_r0s1, vmh_r0, vl) ;
      VFADD(vrin_r0s1, vrsum_s1, vrsum0_s1, 0, 1) ;
      VFADD(vrin_r0s1, vrsum_s3, vrsum0_s3, 0, 3) ;
      vrin_r0s2 = _vel_vmrg_vsvml(0.f, vrin_r0s2, vmh_r0, vl) ;
      VFADD(vrin_r0s2, vrsum_s0, vrsum0_s0, 0, 0) ;
      VFADD(vrin_r0s2, vrsum_s2, vrsum0_s2, 0, 2) ;

      vrin_r1s1 = _vel_vmrg_vsvml(0.f, vrin_r1s1, vmh_r1, vl) ;
      VFADD(vrin_r1s1, vrsum_s1, vrsum0_s1, 1, 1) ;
      VFADD(vrin_r1s1, vrsum_s3, vrsum0_s3, 1, 3) ;
      vrin_r1s2 = _vel_vmrg_vsvml(0.f, vrin_r1s2, vmh_r1, vl) ;
      VFADD(vrin_r1s2, vrsum_s0, vrsum0_s0, 1, 0) ;
      VFADD(vrin_r1s2, vrsum_s2, vrsum0_s2, 1, 2) ;

      vrin_r2s1 = _vel_vmrg_vsvml(0.f, vrin_r2s1, vmh_r2, vl) ;
      VFADD(vrin_r2s1, vrsum_s1, vrsum0_s1, 2, 1) ;
      VFADD(vrin_r2s1, vrsum_s3, vrsum0_s3, 2, 3) ;
      vrin_r2s2 = _vel_vmrg_vsvml(0.f, vrin_r2s2, vmh_r2, vl) ;
      VFADD(vrin_r2s2, vrsum_s0, vrsum0_s0, 2, 0) ;
      VFADD(vrin_r2s2, vrsum_s2, vrsum0_s2, 2, 2) ;

      vrin_r3s1 = _vel_vmrg_vsvml(0.f, vrin_r3s1, vmh_r3, vl) ;
      VFADD(vrin_r3s1, vrsum_s1, vrsum0_s1, 3, 1) ;
      VFADD(vrin_r3s1, vrsum_s3, vrsum0_s3, 3, 3) ;
      vrin_r3s2 = _vel_vmrg_vsvml(0.f, vrin_r3s2, vmh_r3, vl) ;
      VFADD(vrin_r3s2, vrsum_s0, vrsum0_s0, 3, 0) ;
      VFADD(vrin_r3s2, vrsum_s2, vrsum0_s2, 3, 2) ;

#undef VFADD
#undef FILTER_OFFSET
    } // inChannel

    if( remain ) {
      vrsum0_s0 = _vel_vmrg_vsvml(0UL, _vel_vmv_vsvl(-1, vrsum0_s0,vl), vmw_s0, vl) ;
      vrsum0_s1 = _vel_vmrg_vsvml(0UL, vrsum0_s1, vmw_s1, vl) ;
      vrsum0_s2 = _vel_vmrg_vsvml(0UL, vrsum0_s2, vmw_s2, vl) ;
      vrsum0_s3 = _vel_vmrg_vsvml(0UL, _vel_vmv_vsvl( 1, vrsum0_s3,vl), vmw_s3, vl) ;

      __vr vrsum = _vel_vfadds_vvvl(_vel_vfadds_vvvl(vrsum0_s0, vrsum0_s1, vl),
	                            _vel_vfadds_vvvl(vrsum0_s2, vrsum0_s3, vl), vl) ;
      if(ADDBIAS) vrsum = _vel_vfadds_vsvl(bias0, vrsum, vl) ;
      _vel_vstu_vssl(vrsum,  4, pOut+outIndex + 0 * outHeight*outWidth, vl) ;
    }
#pragma clang loop unroll(full)
    for(int64_t kk=0; kk<nPacked; kk++) {
      __vr _vrsum_s0 = _vel_vmrg_vsvml(0UL, _vel_vmv_vsvl(-1, vrsum_s0[kk],vl), vmw_s0, vl) ;
      __vr _vrsum_s1 = _vel_vmrg_vsvml(0UL, vrsum_s1[kk], vmw_s1, vl) ;
      __vr _vrsum_s2 = _vel_vmrg_vsvml(0UL, vrsum_s2[kk], vmw_s2, vl) ;
      __vr _vrsum_s3 = _vel_vmrg_vsvml(0UL, _vel_vmv_vsvl( 1, vrsum_s3[kk],vl), vmw_s3, vl) ;

      __vr vrsum = _vel_pvfadd_vvvl(_vel_pvfadd_vvvl(_vrsum_s0, _vrsum_s1, vl),
	                            _vel_pvfadd_vvvl(_vrsum_s2, _vrsum_s3, vl), vl) ;
      if(ADDBIAS) vrsum = _vel_pvfadd_vsvl(bias[kk], vrsum, vl) ;
      _vel_vstu_vssl(vrsum, 4, pOut+outIndex + (2*kk+remain)   * outHeight*outWidth, vl) ;
      _vel_vstl_vssl(vrsum, 4, pOut+outIndex + (2*kk+remain+1) * outHeight*outWidth, vl) ;
    }

    outIndex += vl ;
  } // outPixels
}

template<int NUMKERNEL, bool ADDBIAS>
static inline void func_nchw(
  const float * __restrict__ pIn,
  const float * __restrict__ pKernel,
  const float * __restrict__ pBias,
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
  const int64_t biasGroupOffset,
  const int64_t kernGroupOffset,
  const int64_t nY,
  const __vr    vri,
  const __vr    vrj,
  const int64_t n,
  const int64_t k
)
{
  float __attribute__ ((aligned(8))) filter[NUMKERNEL*16*64] ;
  uint64_t* filter_u64 = (uint64_t*) filter ;

  int64_t outIndex = outGroupOffset + (n * outChannel + k  ) * outHeight*outWidth ;

  const int64_t remain  = NUMKERNEL & 0x1 ;
  const int64_t nPacked = NUMKERNEL >> 1 ;

  __vr vrw_s0 = _vel_vaddsl_vsvl(0,vrj, nY*outWidth) ;
  __vr vrw_s1 = _vel_vaddsl_vsvl(1,vrj, nY*outWidth) ;
  __vr vrw_s2 = _vel_vaddsl_vsvl(2,vrj, nY*outWidth) ;
  __vr vrw_s3 = _vel_vaddsl_vsvl(3,vrj, nY*outWidth) ;

  __vm256 vmw0_s0 = _vel_vfmklge_mvl(vrw_s0, nY*outWidth) ;						// condition(0 <= w)
  __vm256 vmw1_s0 = _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(inWidth,vrw_s0, nY*outWidth), nY*outWidth) ;	// condition(w < inWidth)
  __vm256 vmw_s0  = _vel_andm_mmm(vmw0_s0, vmw1_s0) ;

  __vm256 vmw0_s1 = _vel_vfmklge_mvl(vrw_s1, nY*outWidth) ;						// condition(0 <= w)
  __vm256 vmw1_s1 = _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(inWidth,vrw_s1, nY*outWidth), nY*outWidth) ;	// condition(w < inWidth)
  __vm256 vmw_s1  = _vel_andm_mmm(vmw0_s1, vmw1_s1) ;

  __vm256 vmw0_s2 = _vel_vfmklge_mvl(vrw_s2, nY*outWidth) ;						// condition(0 <= w)
  __vm256 vmw1_s2 = _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(inWidth,vrw_s2, nY*outWidth), nY*outWidth) ;	// condition(w < inWidth)
  __vm256 vmw_s2  = _vel_andm_mmm(vmw0_s2, vmw1_s2) ;

  __vm256 vmw0_s3 = _vel_vfmklge_mvl(vrw_s3, nY*outWidth) ;						// condition(0 <= w)
  __vm256 vmw1_s3 = _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(inWidth,vrw_s3, nY*outWidth), nY*outWidth) ;	// condition(w < inWidth)
  __vm256 vmw_s3  = _vel_andm_mmm(vmw0_s3, vmw1_s3) ;

  const float   bias0  = ADDBIAS ?  pBias[biasGroupOffset+k+ 0] : 0.f ;
  int64_t bias[nPacked] ;
#pragma clang loop unroll(full)
  for(int64_t kk=0; kk<nPacked; kk++) bias[kk] = ADDBIAS ?  _vel_pack_f32p(pBias+biasGroupOffset+k+2*kk+remain, pBias+biasGroupOffset+k+2*kk+remain+1) : 0UL ;

  for (int64_t y=0; y<outHeight; y+=nY)
  {
    const int64_t vl = outWidth * (outHeight - y < nY ? outHeight - y : nY) ;
    const int64_t op = y * outWidth ;

    __vr vrsum0_s0  = _vel_vbrds_vsl(0.f, vl) ;
    __vr vrsum0_s1  = _vel_vbrds_vsl(0.f, vl) ;
    __vr vrsum0_s2  = _vel_vbrds_vsl(0.f, vl) ;
    __vr vrsum0_s3  = _vel_vbrds_vsl(0.f, vl) ;
    __vr vrsum_s0[nPacked] ;
    __vr vrsum_s1[nPacked] ;
    __vr vrsum_s2[nPacked] ;
    __vr vrsum_s3[nPacked] ;
#pragma clang loop unroll(full)
    for(int64_t kk=0; kk<nPacked; kk++) {
      vrsum_s0[kk] = _vel_pvbrd_vsl(0UL, vl) ;
      vrsum_s1[kk] = _vel_pvbrd_vsl(0UL, vl) ;
      vrsum_s2[kk] = _vel_pvbrd_vsl(0UL, vl) ;
      vrsum_s3[kk] = _vel_pvbrd_vsl(0UL, vl) ;
    }

    __vr vrh_r0 = _vel_vaddsl_vsvl(0+y*2, vri, vl) ;
    __vr vrh_r1 = _vel_vaddsl_vsvl(1+y*2, vri, vl) ;
    __vr vrh_r2 = _vel_vaddsl_vsvl(2+y*2, vri, vl) ;
    __vr vrh_r3 = _vel_vaddsl_vsvl(3+y*2, vri, vl) ;

    __vm256 vmh0_r0 = _vel_vfmklge_mvl(vrh_r0, vl) ;				// condition(0 <= h)
    __vm256 vmh1_r0 = _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(inHeight,vrh_r0, vl), vl) ;	// condition(h < inHeight)
    __vm256 vmh_r0  = _vel_andm_mmm(vmh0_r0, vmh1_r0) ;

    __vm256 vmh0_r1 = _vel_vfmklge_mvl(vrh_r1, vl) ;				// condition(0 <= h)
    __vm256 vmh1_r1 = _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(inHeight,vrh_r1, vl), vl) ;	// condition(h < inHeight)
    __vm256 vmh_r1  = _vel_andm_mmm(vmh0_r1, vmh1_r1) ;

    __vm256 vmh0_r2 = _vel_vfmklge_mvl(vrh_r2, vl) ;				// condition(0 <= h)
    __vm256 vmh1_r2 = _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(inHeight,vrh_r2, vl), vl) ;	// condition(h < inHeight)
    __vm256 vmh_r2  = _vel_andm_mmm(vmh0_r2, vmh1_r2) ;

    __vm256 vmh0_r3 = _vel_vfmklge_mvl(vrh_r3, vl) ;				// condition(0 <= h)
    __vm256 vmh1_r3 = _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(inHeight,vrh_r3, vl), vl) ;	// condition(h < inHeight)
    __vm256 vmh_r3  = _vel_andm_mmm(vmh0_r3, vmh1_r3) ;


    for (int64_t c0 = 0; c0 < inChannelGroup; c0+=64) {
      const int64_t clen = inChannelGroup - c0 < 64 ? inChannelGroup - c0 : 64 ;
      const float *pKerValue  = pKernel + kernGroupOffset + ((k * inChannelGroup + c0) * kernHeight ) * kernWidth ;

      for(int64_t t=0; t<16*clen; t+=256) {
	const int64_t tvl = 16*clen-t < 256 ? 16*clen-t : 256 ;

	__vr vr[NUMKERNEL] ;
#pragma clang loop unroll(full)
	for(int64_t kk=0; kk<NUMKERNEL; kk++) {
	  vr[kk] = _vel_vldu_vssl(4, pKerValue+ kk*16*inChannelGroup+t, tvl) ;
	}

#pragma clang loop unroll(full)
	for(int64_t kk=0; kk<nPacked; kk++) {
	  __vr vrp = _vel_vshf_vvvsl(vr[2*kk+remain], vr[2*kk+remain+1], VE_VSHUFFLE_YUZU, tvl) ;
	  _vel_vst_vssl(vrp, 8, filter_u64+kk*16*clen+t, tvl) ;
	}
	if( remain ) {
	  _vel_vstu_vssl(vr[0], 4, filter+(NUMKERNEL-1)*16*clen+t, tvl) ;
	}
      }
      for (int64_t c1 = 0 ; c1 < clen ; c1++) {
	const int64_t c = c0 + c1 ;
	const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;
	__vr vrpin_r0s1 = _vel_vsfa_vvssl(_vel_vaddul_vvvl(vrw_s1, _vel_vmulul_vsvl(inWidth,vrh_r0, vl), vl), 2, (uint64_t)pInChannel, vl) ;
	__vr vrin_r0s1 = _vel_vgtu_vvssml(vrpin_r0s1, 0, 0, vmh_r0, vl) ;
	__vr vrpin_r0s2 = _vel_vsfa_vvssl(_vel_vaddul_vvvl(vrw_s2, _vel_vmulul_vsvl(inWidth,vrh_r0, vl), vl), 2, (uint64_t)pInChannel, vl) ;
	__vr vrin_r0s2 = _vel_vgtu_vvssml(vrpin_r0s2, 0, 0, vmh_r0, vl) ;

	__vr vrpin_r1s1 = _vel_vsfa_vvssl(_vel_vaddul_vvvl(vrw_s1, _vel_vmulul_vsvl(inWidth,vrh_r1, vl), vl), 2, (uint64_t)pInChannel, vl) ;
	__vr vrin_r1s1 = _vel_vgtu_vvssml(vrpin_r1s1, 0, 0, vmh_r1, vl) ;
	__vr vrpin_r1s2 = _vel_vsfa_vvssl(_vel_vaddul_vvvl(vrw_s2, _vel_vmulul_vsvl(inWidth,vrh_r1, vl), vl), 2, (uint64_t)pInChannel, vl) ;
	__vr vrin_r1s2 = _vel_vgtu_vvssml(vrpin_r1s2, 0, 0, vmh_r1, vl) ;

	__vr vrpin_r2s1 = _vel_vsfa_vvssl(_vel_vaddul_vvvl(vrw_s1, _vel_vmulul_vsvl(inWidth,vrh_r2, vl), vl), 2, (uint64_t)pInChannel, vl) ;
	__vr vrin_r2s1 = _vel_vgtu_vvssml(vrpin_r2s1, 0, 0, vmh_r2, vl) ;
	__vr vrpin_r2s2 = _vel_vsfa_vvssl(_vel_vaddul_vvvl(vrw_s2, _vel_vmulul_vsvl(inWidth,vrh_r2, vl), vl), 2, (uint64_t)pInChannel, vl) ;
	__vr vrin_r2s2 = _vel_vgtu_vvssml(vrpin_r2s2, 0, 0, vmh_r2, vl) ;

	__vr vrpin_r3s1 = _vel_vsfa_vvssl(_vel_vaddul_vvvl(vrw_s1, _vel_vmulul_vsvl(inWidth,vrh_r3, vl), vl), 2, (uint64_t)pInChannel, vl) ;
	__vr vrin_r3s1 = _vel_vgtu_vvssml(vrpin_r3s1, 0, 0, vmh_r3, vl) ;
	__vr vrpin_r3s2 = _vel_vsfa_vvssl(_vel_vaddul_vvvl(vrw_s2, _vel_vmulul_vsvl(inWidth,vrh_r3, vl), vl), 2, (uint64_t)pInChannel, vl) ;
	__vr vrin_r3s2 = _vel_vgtu_vvssml(vrpin_r3s2, 0, 0, vmh_r3, vl) ;


#define VFADD(VRIN, VRSUM, VRSUM0, R, S)								\
	{												\
	  __vr vrinP = _vel_vshf_vvvsl(VRIN, VRIN, VE_VSHUFFLE_YUZU, vl) ;				\
	  if( remain ) {										\
	    const float kerValue0  = filter[(NUMKERNEL-1)*16*clen+16*(c1)+(4*(R)+(S))] ;		\
	    VRSUM0 = _vel_vfmads_vvsvl(VRSUM0, kerValue0, VRIN, vl) ;					\
	  }												\
	  _Pragma("clang loop unroll(full)")								\
	  for(int64_t kk=0; kk<nPacked; kk++) {								\
            const uint64_t kerValue = filter_u64[kk*16*clen+16*(c1)+(4*(R)+(S))] ;			\
	    VRSUM[kk]= _vel_pvfmad_vvsvl(VRSUM[kk], kerValue, vrinP, vl) ;				\
	  }												\
	}

	vrin_r0s1 = _vel_vmrg_vsvml(0.f, vrin_r0s1, vmh_r0, vl) ;
	VFADD(vrin_r0s1, vrsum_s1, vrsum0_s1, 0, 1) ;
	VFADD(vrin_r0s1, vrsum_s3, vrsum0_s3, 0, 3) ;
	vrin_r0s2 = _vel_vmrg_vsvml(0.f, vrin_r0s2, vmh_r0, vl) ;
	VFADD(vrin_r0s2, vrsum_s0, vrsum0_s0, 0, 0) ;
	VFADD(vrin_r0s2, vrsum_s2, vrsum0_s2, 0, 2) ;

	vrin_r1s1 = _vel_vmrg_vsvml(0.f, vrin_r1s1, vmh_r1, vl) ;
	VFADD(vrin_r1s1, vrsum_s1, vrsum0_s1, 1, 1) ;
	VFADD(vrin_r1s1, vrsum_s3, vrsum0_s3, 1, 3) ;
	vrin_r1s2 = _vel_vmrg_vsvml(0.f, vrin_r1s2, vmh_r1, vl) ;
	VFADD(vrin_r1s2, vrsum_s0, vrsum0_s0, 1, 0) ;
	VFADD(vrin_r1s2, vrsum_s2, vrsum0_s2, 1, 2) ;

	vrin_r2s1 = _vel_vmrg_vsvml(0.f, vrin_r2s1, vmh_r2, vl) ;
	VFADD(vrin_r2s1, vrsum_s1, vrsum0_s1, 2, 1) ;
	VFADD(vrin_r2s1, vrsum_s3, vrsum0_s3, 2, 3) ;
	vrin_r2s2 = _vel_vmrg_vsvml(0.f, vrin_r2s2, vmh_r2, vl) ;
	VFADD(vrin_r2s2, vrsum_s0, vrsum0_s0, 2, 0) ;
	VFADD(vrin_r2s2, vrsum_s2, vrsum0_s2, 2, 2) ;

	vrin_r3s1 = _vel_vmrg_vsvml(0.f, vrin_r3s1, vmh_r3, vl) ;
	VFADD(vrin_r3s1, vrsum_s1, vrsum0_s1, 3, 1) ;
	VFADD(vrin_r3s1, vrsum_s3, vrsum0_s3, 3, 3) ;
	vrin_r3s2 = _vel_vmrg_vsvml(0.f, vrin_r3s2, vmh_r3, vl) ;
	VFADD(vrin_r3s2, vrsum_s0, vrsum0_s0, 3, 0) ;
	VFADD(vrin_r3s2, vrsum_s2, vrsum0_s2, 3, 2) ;
#undef VFADD
      }
    } // inChannel

    if( remain ) {
      vrsum0_s0 = _vel_vmrg_vsvml(0UL, _vel_vmv_vsvl(-1, vrsum0_s0,vl), vmw_s0, vl) ;
      vrsum0_s1 = _vel_vmrg_vsvml(0UL, vrsum0_s1, vmw_s1, vl) ;
      vrsum0_s2 = _vel_vmrg_vsvml(0UL, vrsum0_s2, vmw_s2, vl) ;
      vrsum0_s3 = _vel_vmrg_vsvml(0UL, _vel_vmv_vsvl( 1, vrsum0_s3,vl), vmw_s3, vl) ;
      __vr vrsum = _vel_vfadds_vvvl(_vel_vfadds_vvvl(vrsum0_s0, vrsum0_s1, vl),
	                            _vel_vfadds_vvvl(vrsum0_s2, vrsum0_s3, vl), vl) ;
      if(ADDBIAS) vrsum = _vel_vfadds_vsvl(bias0, vrsum, vl) ;
      _vel_vstu_vssl(vrsum,  4, pOut+outIndex + 0 * outHeight*outWidth, vl) ;
    }
#pragma clang loop unroll(full)
    for(int64_t kk=0; kk<nPacked; kk++) {
      __vr _vrsum_s0 = _vel_vmrg_vsvml(0UL, _vel_vmv_vsvl(-1, vrsum_s0[kk],vl), vmw_s0, vl) ;
      __vr _vrsum_s1 = _vel_vmrg_vsvml(0UL, vrsum_s1[kk], vmw_s1, vl) ;
      __vr _vrsum_s2 = _vel_vmrg_vsvml(0UL, vrsum_s2[kk], vmw_s2, vl) ;
      __vr _vrsum_s3 = _vel_vmrg_vsvml(0UL, _vel_vmv_vsvl( 1, vrsum_s3[kk],vl), vmw_s3, vl) ;

      __vr vrsum = _vel_pvfadd_vvvl(_vel_pvfadd_vvvl(_vrsum_s0, _vrsum_s1, vl),
	                            _vel_pvfadd_vvvl(_vrsum_s2, _vrsum_s3, vl), vl) ;
      if(ADDBIAS) vrsum = _vel_pvfadd_vsvl(bias[kk], vrsum, vl) ;
      _vel_vstu_vssl(vrsum, 4, pOut+outIndex + (2*kk+remain)   * outHeight*outWidth, vl) ;
      _vel_vstl_vssl(vrsum, 4, pOut+outIndex + (2*kk+remain+1) * outHeight*outWidth, vl) ;
    }

    outIndex += vl ;
  } // outPixels
}

template<filterLayout_t FLAYOUT, int NUMKERNEL, bool ADDBIAS>
static inline void func(
  const float * __restrict__ pIn,
  const float * __restrict__ pKernel,
  const float * __restrict__ pBias,
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
  const int64_t biasGroupOffset,
  const int64_t kernGroupOffset,
  const int64_t nY,
  const __vr    vri,
  const __vr    vrj,
  const int64_t n,
  const int64_t k
)
{
  if( FLAYOUT == VEDNN_FILTER_LAYOUT_NCHW) {
      func_nchw<NUMKERNEL, ADDBIAS>(pIn, pKernel, pBias, pOut,
	 inChannel, inWidth, inHeight,
	 outChannel, outWidth, outHeight,
	 kernWidth, kernHeight,
	 inChannelGroup, outChannelGroup,
	 inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	 nY, vri, vrj, n, k ) ;
  }
  else {
    func_hwcn<NUMKERNEL, ADDBIAS>(pIn, pKernel, pBias, pOut,
	 inChannel, inWidth, inHeight,
	 outChannel, outWidth, outHeight,
	 kernWidth, kernHeight,
	 inChannelGroup, outChannelGroup,
	 inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	 nY, vri, vrj, n, k ) ;
  }
}

template<filterLayout_t FLAYOUT, bool ADDBIAS>
static inline void convloop(
    const float * __restrict__ pIn,
    const float * __restrict__ pKernel,
    const float * __restrict__ pBias,
    float * __restrict__ const pOut,
    const int64_t batch,
    const int64_t group,
    const int64_t inChannel,
    const int64_t inWidth,
    const int64_t inHeight,
    const int64_t outChannel,
    const int64_t outWidth,
    const int64_t outHeight,
    const int64_t kernWidth,		// 4
    const int64_t kernHeight,		// 4
    const int64_t inChannelGroup,
    const int64_t outChannelGroup,
    const int64_t strideHeight,		// 2
    const int64_t strideWidth,		// 2
    const int64_t padHeight,		// 1
    const int64_t padWidth,		// 1
    const int64_t dilationHeight,	// 1
    const int64_t dilationWidth		// 1
)
{
  const int64_t nY = VLEN / outWidth ;

  const int64_t maxvl = nY * outWidth ;

  __vr vrseq = _vel_vseq_vl(maxvl) ;
  __vr vry  = _vel_vdivsl_vvsl(vrseq, outWidth, maxvl) ;
  __vr vrx  = _vel_vsubsl_vvvl(vrseq, _vel_vmulul_vsvl(outWidth,vry, maxvl), maxvl) ;

  __vr vri   = _vel_vaddsl_vsvl(-1, _vel_vmulsl_vsvl(2, vry, maxvl), maxvl) ;
  __vr vrj   = _vel_vaddsl_vsvl(-1,  _vel_vmulsl_vsvl(2,  vrx, maxvl), maxvl) ;

  for (int64_t n = 0; n < batch; n++) {
    for (int64_t g = 0; g < group; g++) {
	const int64_t inGroupOffset   = g * inChannelGroup * inHeight * inWidth;
	const int64_t outGroupOffset  = g * outChannelGroup * outHeight * outWidth;
	const int64_t biasGroupOffset = g * outChannelGroup;
	const int64_t kernGroupOffset = g * outChannelGroup * inChannelGroup * kernHeight * kernWidth;

	const int64_t remain = outChannelGroup & 0xf ;

	int k = 0 ;
	switch( remain ) {
	case 1 :
	  func<FLAYOUT, 1, ADDBIAS>(pIn, pKernel, pBias, pOut,
	     inChannel, inWidth, inHeight,
	     outChannel, outWidth, outHeight,
	     kernWidth, kernHeight,
	     inChannelGroup, outChannelGroup,
	     inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	     nY, vri, vrj, n, k ) ;
	  k+=1 ;
	  break ;
	case 2 :
	  func<FLAYOUT, 2, ADDBIAS>(pIn, pKernel, pBias, pOut,
	     inChannel, inWidth, inHeight,
	     outChannel, outWidth, outHeight,
	     kernWidth, kernHeight,
	     inChannelGroup, outChannelGroup,
	     inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	     nY, vri, vrj, n, k ) ;
	  k+=2 ;
	  break ;
	case 3 :
	  func<FLAYOUT, 3, ADDBIAS>(pIn, pKernel, pBias, pOut,
	     inChannel, inWidth, inHeight,
	     outChannel, outWidth, outHeight,
	     kernWidth, kernHeight,
	     inChannelGroup, outChannelGroup,
	     inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	     nY, vri, vrj, n, k ) ;
	  k+=3 ;
	  break ;
	case 4 :
	  func<FLAYOUT, 4, ADDBIAS>(pIn, pKernel, pBias, pOut,
	     inChannel, inWidth, inHeight,
	     outChannel, outWidth, outHeight,
	     kernWidth, kernHeight,
	     inChannelGroup, outChannelGroup,
	     inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	     nY, vri, vrj, n, k ) ;
	  k+=4 ;
	  break ;
	case 5 :
	  func<FLAYOUT, 5, ADDBIAS>(pIn, pKernel, pBias, pOut,
	     inChannel, inWidth, inHeight,
	     outChannel, outWidth, outHeight,
	     kernWidth, kernHeight,
	     inChannelGroup, outChannelGroup,
	     inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	     nY, vri, vrj, n, k ) ;
	  k+=5 ;
	  break ;
	case 6 :
	  func<FLAYOUT, 6, ADDBIAS>(pIn, pKernel, pBias, pOut,
	     inChannel, inWidth, inHeight,
	     outChannel, outWidth, outHeight,
	     kernWidth, kernHeight,
	     inChannelGroup, outChannelGroup,
	     inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	     nY, vri, vrj, n, k ) ;
	  k+=6 ;
	  break ;
	case 7 :
	  func<FLAYOUT, 7, ADDBIAS>(pIn, pKernel, pBias, pOut,
	     inChannel, inWidth, inHeight,
	     outChannel, outWidth, outHeight,
	     kernWidth, kernHeight,
	     inChannelGroup, outChannelGroup,
	     inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	     nY, vri, vrj, n, k ) ;
	  k+=7 ;
	  break ;
	case 8 :
	  func<FLAYOUT, 8, ADDBIAS>(pIn, pKernel, pBias, pOut,
	     inChannel, inWidth, inHeight,
	     outChannel, outWidth, outHeight,
	     kernWidth, kernHeight,
	     inChannelGroup, outChannelGroup,
	     inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	     nY, vri, vrj, n, k ) ;
	  k+=8 ;
	  break ;
	case 9 :
	  func<FLAYOUT, 9, ADDBIAS>(pIn, pKernel, pBias, pOut,
	     inChannel, inWidth, inHeight,
	     outChannel, outWidth, outHeight,
	     kernWidth, kernHeight,
	     inChannelGroup, outChannelGroup,
	     inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	     nY, vri, vrj, n, k ) ;
	  k+=9 ;
	  break ;
	case 10 :
	  func<FLAYOUT, 10, ADDBIAS>(pIn, pKernel, pBias, pOut,
	     inChannel, inWidth, inHeight,
	     outChannel, outWidth, outHeight,
	     kernWidth, kernHeight,
	     inChannelGroup, outChannelGroup,
	     inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	     nY, vri, vrj, n, k ) ;
	  k+=10 ;
	  break ;
	case 11 :
	  func<FLAYOUT, 11, ADDBIAS>(pIn, pKernel, pBias, pOut,
	     inChannel, inWidth, inHeight,
	     outChannel, outWidth, outHeight,
	     kernWidth, kernHeight,
	     inChannelGroup, outChannelGroup,
	     inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	     nY, vri, vrj, n, k ) ;
	  k+=11 ;
	  break ;
	case 12 :
	  func<FLAYOUT, 12, ADDBIAS>(pIn, pKernel, pBias, pOut,
	     inChannel, inWidth, inHeight,
	     outChannel, outWidth, outHeight,
	     kernWidth, kernHeight,
	     inChannelGroup, outChannelGroup,
	     inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	     nY, vri, vrj, n, k ) ;
	  k+=12 ;
	  break ;
	case 13 :
	  func<FLAYOUT, 13, ADDBIAS>(pIn, pKernel, pBias, pOut,
	     inChannel, inWidth, inHeight,
	     outChannel, outWidth, outHeight,
	     kernWidth, kernHeight,
	     inChannelGroup, outChannelGroup,
	     inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	     nY, vri, vrj, n, k ) ;
	  k+=13 ;
	  break ;
	case 14 :
	  func<FLAYOUT, 14, ADDBIAS>(pIn, pKernel, pBias, pOut,
	     inChannel, inWidth, inHeight,
	     outChannel, outWidth, outHeight,
	     kernWidth, kernHeight,
	     inChannelGroup, outChannelGroup,
	     inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	     nY, vri, vrj, n, k ) ;
	  k+=14 ;
	  break ;
	case 15 :
	  func<FLAYOUT, 15, ADDBIAS>(pIn, pKernel, pBias, pOut,
	     inChannel, inWidth, inHeight,
	     outChannel, outWidth, outHeight,
	     kernWidth, kernHeight,
	     inChannelGroup, outChannelGroup,
	     inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	     nY, vri, vrj, n, k ) ;
	  k+=15 ;
	  break ;
	default :
	  break ;
	}
	for (; k < outChannelGroup; k+=16) {
	  func<FLAYOUT, 16, ADDBIAS>(pIn, pKernel, pBias, pOut,
	     inChannel, inWidth, inHeight,
	     outChannel, outWidth, outHeight,
	     kernWidth, kernHeight,
	     inChannelGroup, outChannelGroup,
	     inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	     nY, vri, vrj, n, k ) ;
	} // outChannel
    } // group
  } // batch
}

extern "C" vednnError_t
vednnConvolutionForward_direct_dil1_str2_pad1_ker4_owU128(
    const vednnTensorParam_t *  	pParamIn,
    const void *  			pDataIn,
    const vednnFilterParam_t *  	pParamKernel,
    const void *  			pDataKernel,
    const vednnBiasParam_t * 		pParamBias,
    const void * 			pDataBias,
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
  const int64_t strideWidth    = pParamConv->strideWidth;;
  const int64_t strideHeight   = pParamConv->strideHeight;
  const int64_t padWidth       = pParamConv->padWidth;
  const int64_t padHeight      = pParamConv->padHeight;
  const int64_t dilationWidth  = pParamConv->dilationWidth;
  const int64_t dilationHeight = pParamConv->dilationHeight;

  const int64_t inChannelGroup  = inChannel  / group;   // equal to pDataKernel->inChannel
  const int64_t outChannelGroup = outChannel / group;   // equal to pDataKernel->outChannel

  const float * pIn     = (const float *) pDataIn;
  const float * pKernel = (const float *) pDataKernel;
  const float * pBias   = (const float *) pDataBias;
  float * const pOut    = (float * const) pDataOut;


  if( filter_layout == VEDNN_FILTER_LAYOUT_NCHW) {
    if( pDataBias == NULL ) {
      convloop<VEDNN_FILTER_LAYOUT_NCHW, false>(pIn, pKernel, pBias, pOut,
		 batch, group,
		 inChannel, inWidth, inHeight,
		 outChannel, outWidth, outHeight,
		 kernWidth, kernHeight,
		 inChannelGroup, outChannelGroup,
		 strideHeight, strideWidth,
		 padHeight, padWidth,
		 dilationHeight, dilationWidth ) ;
    }
    else {
      convloop<VEDNN_FILTER_LAYOUT_NCHW, true>(pIn, pKernel, pBias, pOut,
		 batch, group,
		 inChannel, inWidth, inHeight,
		 outChannel, outWidth, outHeight,
		 kernWidth, kernHeight,
		 inChannelGroup, outChannelGroup,
		 strideHeight, strideWidth,
		 padHeight, padWidth,
		 dilationHeight, dilationWidth ) ;
    }
  }
  else {
    if( pDataBias == NULL ) {
      convloop<VEDNN_FILTER_LAYOUT_HWCN, false>(pIn, pKernel, pBias, pOut,
		 batch, group,
		 inChannel, inWidth, inHeight,
		 outChannel, outWidth, outHeight,
		 kernWidth, kernHeight,
		 inChannelGroup, outChannelGroup,
		 strideHeight, strideWidth,
		 padHeight, padWidth,
		 dilationHeight, dilationWidth ) ;
    }
    else {
      convloop<VEDNN_FILTER_LAYOUT_HWCN, true>(pIn, pKernel, pBias, pOut,
		 batch, group,
		 inChannel, inWidth, inHeight,
		 outChannel, outWidth, outHeight,
		 kernWidth, kernHeight,
		 inChannelGroup, outChannelGroup,
		 strideHeight, strideWidth,
		 padHeight, padWidth,
		 dilationHeight, dilationWidth ) ;
    }
  }

  return VEDNN_SUCCESS;
}

