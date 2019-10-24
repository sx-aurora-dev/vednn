#include <stdint.h>
#include <stdlib.h>

#include "vednn.h"

#include "velintrin.h"
#define VLEN	(256)

template<filterLayout_t FLAYOUT, int NUMKERNEL, bool ADDBIAS>
static __attribute__((noinline)) void func_inoaligned(
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
  const int64_t strideHeight,
  const int64_t strideWidth,
  const int64_t padHeight,
  const int64_t padWidth,
  const int64_t dilationHeight,
  const int64_t dilationWidth,
  const int64_t inGroupOffset,
  const int64_t outGroupOffset,
  const int64_t biasGroupOffset,
  const int64_t kernGroupOffset,
  const int64_t nY,
  const int64_t n,
  const int64_t k
)
{
  int64_t outIndex = outGroupOffset + (n * outChannel + k  ) * outHeight*outWidth ;

  const int64_t remain  = NUMKERNEL & 0x1 ;
  const int64_t nPacked = NUMKERNEL >> 1 ;

  const float   bias0  = ADDBIAS ?  pBias[biasGroupOffset+k+ 0] : 0.f ;
  int64_t bias[nPacked] ;
#pragma clang loop unroll(full)
  for(int64_t kk=0; kk<nPacked; kk++) bias[kk] = ADDBIAS ?  _vel_pack_f32p(pBias+biasGroupOffset+k+2*kk+remain, pBias+biasGroupOffset+k+2*kk+remain+1) : 0UL ;

  const int64_t maxvl = nY * outWidth ;

  __vr vrseq = _vel_vseq_vl(maxvl) ;
  __vr vry  = _vel_vdivsl_vvsl(vrseq, outWidth, maxvl) ;
  __vr vrx  = _vel_vsubsl_vvvl(vrseq, _vel_vmulul_vsvl(outWidth,vry, maxvl), maxvl) ;

  __vr vri   = _vel_vaddsl_vsvl(-padHeight, _vel_vmulsl_vsvl(strideHeight, vry, maxvl), maxvl) ;
  __vr vrj   = _vel_vaddsl_vsvl(-padWidth,  _vel_vmulsl_vsvl(strideWidth,  vrx, maxvl), maxvl) ;

  for (int64_t y=0; y<outHeight; y+=nY)
  {
    const int64_t vl = outWidth * (outHeight - y < nY ? outHeight - y : nY) ;
    const int64_t op = y * outWidth ;

    __vr vrsum0  = _vel_vbrds_vsl(bias0, vl) ;
    __vr vrsum[nPacked] ;
#pragma clang loop unroll(full)
    for(int64_t kk=0; kk<nPacked; kk++) vrsum[kk] = _vel_pvbrd_vsl(bias[kk], vl) ;

    __vr vrh_r0 = _vel_vaddsl_vsvl(0*dilationHeight+y*strideHeight, vri, vl) ;
    __vr vrh_r1 = _vel_vaddsl_vsvl(1*dilationHeight+y*strideHeight, vri, vl) ;
    __vr vrh_r2 = _vel_vaddsl_vsvl(2*dilationHeight+y*strideHeight, vri, vl) ;
    __vm256 vm_r0 =  _vel_vfmklge_mvl(vrh_r0, vl) ;					// condition(0 <= h)
    __vm256 vm_r2 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(inHeight,vrh_r2, vl), vl) ;	// condition(h < inHeight)

    __vr vrw_s0 = _vel_vaddsl_vsvl(0*dilationWidth,                 vrj, vl) ;
    __vr vrw_s1 = _vel_vaddsl_vsvl(1*dilationWidth,                 vrj, vl) ;
    __vr vrw_s2 = _vel_vaddsl_vsvl(2*dilationWidth,                 vrj, vl) ;
    __vm256 vm_s0 =  _vel_vfmklge_mvl(vrw_s0, vl) ;					// condition(0 <= w)
    __vm256 vm_s2 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(inWidth,vrw_s2, vl), vl) ;	// condition(w < inWidth)

    __vm256 vm_r0s0 = _vel_andm_mmm(vm_r0,vm_s0) ;
    __vm256 vm_r0s1 = vm_r0 ;
    __vm256 vm_r0s2 = _vel_andm_mmm(vm_r0,vm_s2) ;
    __vm256 vm_r1s0 = vm_s0 ;
    __vm256 vm_r1s1 = _vel_vfmklat_ml(vl) ;
    __vm256 vm_r1s2 = vm_s2 ;
    __vm256 vm_r2s0 = _vel_andm_mmm(vm_r2,vm_s0) ;
    __vm256 vm_r2s1 = vm_r2 ;
    __vm256 vm_r2s2 = _vel_andm_mmm(vm_r2,vm_s2) ;

    int64_t c=0 ;
    if( (inChannelGroup & 0x1) == 1 ) {
      const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;
      __vr vrpin_c0r0s0 = _vel_vsfa_vvssl(_vel_vaddul_vvvl(vrw_s0, _vel_vmulul_vsvl(inWidth,vrh_r0, vl), vl),
				   2,
				   (uint64_t)pInChannel, vl) ;
      __vr vrin_c0r0s0 = _vel_vgtu_vvssml(vrpin_c0r0s0, 0, 0, vm_r0s0, vl) ;

      __vr vrpin_c0r0s1 = _vel_vaddul_vsvl(4*1*dilationWidth, vrpin_c0r0s0, vl) ;
      __vr vrin_c0r0s1 = _vel_vgtu_vvssml(vrpin_c0r0s1, 0, 0, vm_r0s1, vl) ;

      __vr vrpin_c0r0s2 = _vel_vaddul_vsvl(4*2*dilationWidth, vrpin_c0r0s0, vl) ;
      __vr vrin_c0r0s2 = _vel_vgtu_vvssml(vrpin_c0r0s2, 0, 0, vm_r0s2, vl) ;

      __vr vrpin_c0r1s0 = _vel_vaddul_vsvl(4*1*dilationHeight*inWidth, vrpin_c0r0s0, vl) ;
      __vr vrin_c0r1s0 = _vel_vgtu_vvssml(vrpin_c0r1s0, 0, 0, vm_r1s0, vl) ;

      __vr vrpin_c0r1s1 = _vel_vaddul_vsvl(4*1*dilationWidth, vrpin_c0r1s0, vl) ;
      __vr vrin_c0r1s1 = _vel_vgtu_vvssml(vrpin_c0r1s1, 0, 0, vm_r1s1, vl) ;

      __vr vrpin_c0r1s2 = _vel_vaddul_vsvl(4*2*dilationWidth, vrpin_c0r1s0, vl) ;
      __vr vrin_c0r1s2 = _vel_vgtu_vvssml(vrpin_c0r1s2, 0, 0, vm_r1s2, vl) ;

      __vr vrpin_c0r2s0 = _vel_vaddul_vsvl(4*2*dilationHeight*inWidth, vrpin_c0r0s0, vl) ;
      __vr vrin_c0r2s0 = _vel_vgtu_vvssml(vrpin_c0r2s0, 0, 0, vm_r2s0, vl) ;

      __vr vrpin_c0r2s1 = _vel_vaddul_vsvl(4*1*dilationWidth, vrpin_c0r2s0, vl) ;
      __vr vrin_c0r2s1 = _vel_vgtu_vvssml(vrpin_c0r2s1, 0, 0, vm_r2s1, vl) ;

      __vr vrpin_c0r2s2 = _vel_vaddul_vsvl(4*2*dilationWidth, vrpin_c0r2s0, vl) ;
      __vr vrin_c0r2s2 = _vel_vgtu_vvssml(vrpin_c0r2s2, 0, 0, vm_r2s2, vl) ;

#define FILTER_OFFSET(k,c,r,s) ( kernGroupOffset + filter_index<FLAYOUT>(k,c,r,s, inChannelGroup, outChannelGroup, kernHeight, kernWidth) )

#define VFADD(VRIN, VE_VSHUFFLE_MODE, VM, C, R, S)							\
      {													\
	__vr vrinP = _vel_vshf_vvvsl(VRIN, VRIN, VE_VSHUFFLE_MODE, vl) ;				\
	vrinP = _vel_vmrg_vsvml(0UL, vrinP, VM, vl) ;							\
	if( remain ) {											\
	  const float    kerValue0  = pKernel[FILTER_OFFSET(k+ 0,C,R,S)] ;				\
	  vrsum0 = _vel_vfmads_vvsvl(vrsum0, kerValue0, vrinP, vl) ;					\
	}												\
        _Pragma("clang loop unroll(full)")								\
	for(int64_t kk=0; kk<nPacked; kk++) {								\
	  const uint64_t kerValue = _vel_pack_f32p(pKernel + FILTER_OFFSET(k+2*kk+remain,  c,R,S),	\
						   pKernel + FILTER_OFFSET(k+2*kk+remain+1,c,R,S)) ;	\
	  vrsum[kk]= _vel_pvfmad_vvsvl(vrsum[kk], kerValue, vrinP, vl) ;				\
	}												\
      }

      VFADD(vrin_c0r0s0, VE_VSHUFFLE_YUZU, vm_r0s0, c, 0, 0) ;
      VFADD(vrin_c0r0s1, VE_VSHUFFLE_YUZU, vm_r0s1, c, 0, 1) ;
      VFADD(vrin_c0r0s2, VE_VSHUFFLE_YUZU, vm_r0s2, c, 0, 2) ;

      VFADD(vrin_c0r1s0, VE_VSHUFFLE_YUZU, vm_r1s0, c, 1, 0) ;
      VFADD(vrin_c0r1s1, VE_VSHUFFLE_YUZU, vm_r1s1, c, 1, 1) ;
      VFADD(vrin_c0r1s2, VE_VSHUFFLE_YUZU, vm_r1s2, c, 1, 2) ;

      VFADD(vrin_c0r2s0, VE_VSHUFFLE_YUZU, vm_r2s0, c, 2, 0) ;
      VFADD(vrin_c0r2s1, VE_VSHUFFLE_YUZU, vm_r2s1, c, 2, 1) ;
      VFADD(vrin_c0r2s2, VE_VSHUFFLE_YUZU, vm_r2s2, c, 2, 2) ;

      c+=1 ;
    }
    for( ; c < inChannelGroup ; c+=2 ) {
	const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;
	__vr vrpin_c0r0s0 = _vel_vsfa_vvssl(_vel_vaddul_vvvl(vrw_s0, _vel_vmulul_vsvl(inWidth,vrh_r0, vl), vl),
				     2,
				     (uint64_t)pInChannel, vl) ;
	__vr vrin_c0r0s0 = _vel_vgtu_vvssml(vrpin_c0r0s0, 0, 0, vm_r0s0, vl) ;

	__vr vrpin_c0r0s1 = _vel_vaddul_vsvl(4*1*dilationWidth, vrpin_c0r0s0, vl) ;
	__vr vrin_c0r0s1 = _vel_vgtu_vvssml(vrpin_c0r0s1, 0, 0, vm_r0s1, vl) ;

	__vr vrpin_c0r0s2 = _vel_vaddul_vsvl(4*2*dilationWidth, vrpin_c0r0s0, vl) ;
	__vr vrin_c0r0s2 = _vel_vgtu_vvssml(vrpin_c0r0s2, 0, 0, vm_r0s2, vl) ;

	__vr vrpin_c0r1s0 = _vel_vaddul_vsvl(4*1*dilationHeight*inWidth, vrpin_c0r0s0, vl) ;
	__vr vrin_c0r1s0 = _vel_vgtu_vvssml(vrpin_c0r1s0, 0, 0, vm_r1s0, vl) ;

	__vr vrpin_c0r1s1 = _vel_vaddul_vsvl(4*1*dilationWidth, vrpin_c0r1s0, vl) ;
	__vr vrin_c0r1s1 = _vel_vgtu_vvssml(vrpin_c0r1s1, 0, 0, vm_r1s1, vl) ;

	__vr vrpin_c0r1s2 = _vel_vaddul_vsvl(4*2*dilationWidth, vrpin_c0r1s0, vl) ;
	__vr vrin_c0r1s2 = _vel_vgtu_vvssml(vrpin_c0r1s2, 0, 0, vm_r1s2, vl) ;

	__vr vrpin_c0r2s0 = _vel_vaddul_vsvl(4*2*dilationHeight*inWidth, vrpin_c0r0s0, vl) ;
	__vr vrin_c0r2s0 = _vel_vgtu_vvssml(vrpin_c0r2s0, 0, 0, vm_r2s0, vl) ;

	__vr vrpin_c0r2s1 = _vel_vaddul_vsvl(4*1*dilationWidth, vrpin_c0r2s0, vl) ;
	__vr vrin_c0r2s1 = _vel_vgtu_vvssml(vrpin_c0r2s1, 0, 0, vm_r2s1, vl) ;

	__vr vrpin_c0r2s2 = _vel_vaddul_vsvl(4*2*dilationWidth, vrpin_c0r2s0, vl) ;
	__vr vrin_c0r2s2 = _vel_vgtu_vvssml(vrpin_c0r2s2, 0, 0, vm_r2s2, vl) ;


	__vr vrpin_c1r0s0 = _vel_vaddul_vsvl(4*1*inHeight*inWidth, vrpin_c0r0s0, vl) ;
	__vr vrin_c1r0s0 = _vel_vgtu_vvssml(vrpin_c1r0s0, 0, 0, vm_r0s0, vl) ;

	__vr vrpin_c1r0s1 = _vel_vaddul_vsvl(4*1*dilationWidth, vrpin_c1r0s0, vl) ;
	__vr vrin_c1r0s1 = _vel_vgtu_vvssml(vrpin_c1r0s1, 0, 0, vm_r0s1, vl) ;

	__vr vrpin_c1r0s2 = _vel_vaddul_vsvl(4*2*dilationWidth, vrpin_c1r0s0, vl) ;
	__vr vrin_c1r0s2 = _vel_vgtu_vvssml(vrpin_c1r0s2, 0, 0, vm_r0s2, vl) ;

	__vr vrpin_c1r1s0 = _vel_vaddul_vsvl(4*1*dilationHeight*inWidth, vrpin_c1r0s0, vl) ;
	__vr vrin_c1r1s0 = _vel_vgtu_vvssml(vrpin_c1r1s0, 0, 0, vm_r1s0, vl) ;

	__vr vrpin_c1r1s1 = _vel_vaddul_vsvl(4*1*dilationWidth, vrpin_c1r1s0, vl) ;
	__vr vrin_c1r1s1 = _vel_vgtu_vvssml(vrpin_c1r1s1, 0, 0, vm_r1s1, vl) ;

	__vr vrpin_c1r1s2 = _vel_vaddul_vsvl(4*2*dilationWidth, vrpin_c1r1s0, vl) ;
	__vr vrin_c1r1s2 = _vel_vgtu_vvssml(vrpin_c1r1s2, 0, 0, vm_r1s2, vl) ;

	__vr vrpin_c1r2s0 = _vel_vaddul_vsvl(4*2*dilationHeight*inWidth, vrpin_c1r0s0, vl) ;
	__vr vrin_c1r2s0 = _vel_vgtu_vvssml(vrpin_c1r2s0, 0, 0, vm_r2s0, vl) ;

	__vr vrpin_c1r2s1 = _vel_vaddul_vsvl(4*1*dilationWidth, vrpin_c1r2s0, vl) ;
	__vr vrin_c1r2s1 = _vel_vgtu_vvssml(vrpin_c1r2s1, 0, 0, vm_r2s1, vl) ;

	__vr vrpin_c1r2s2 = _vel_vaddul_vsvl(4*2*dilationWidth, vrpin_c1r2s0, vl) ;
	__vr vrin_c1r2s2 = _vel_vgtu_vvssml(vrpin_c1r2s2, 0, 0, vm_r2s2, vl) ;

	VFADD(vrin_c0r0s0, VE_VSHUFFLE_YUZU, vm_r0s0, c, 0, 0) ;
	VFADD(vrin_c0r0s1, VE_VSHUFFLE_YUZU, vm_r0s1, c, 0, 1) ;
	VFADD(vrin_c0r0s2, VE_VSHUFFLE_YUZU, vm_r0s2, c, 0, 2) ;

	VFADD(vrin_c0r1s0, VE_VSHUFFLE_YUZU, vm_r1s0, c, 1, 0) ;
	VFADD(vrin_c0r1s1, VE_VSHUFFLE_YUZU, vm_r1s1, c, 1, 1) ;
	VFADD(vrin_c0r1s2, VE_VSHUFFLE_YUZU, vm_r1s2, c, 1, 2) ;

	VFADD(vrin_c0r2s0, VE_VSHUFFLE_YUZU, vm_r2s0, c, 2, 0) ;
	VFADD(vrin_c0r2s1, VE_VSHUFFLE_YUZU, vm_r2s1, c, 2, 1) ;
	VFADD(vrin_c0r2s2, VE_VSHUFFLE_YUZU, vm_r2s2, c, 2, 2) ;

	VFADD(vrin_c1r0s0, VE_VSHUFFLE_YUZU, vm_r0s0, c+1, 0, 0) ;
	VFADD(vrin_c1r0s1, VE_VSHUFFLE_YUZU, vm_r0s1, c+1, 0, 1) ;
	VFADD(vrin_c1r0s2, VE_VSHUFFLE_YUZU, vm_r0s2, c+1, 0, 2) ;

	VFADD(vrin_c1r1s0, VE_VSHUFFLE_YUZU, vm_r1s0, c+1, 1, 0) ;
	VFADD(vrin_c1r1s1, VE_VSHUFFLE_YUZU, vm_r1s1, c+1, 1, 1) ;
	VFADD(vrin_c1r1s2, VE_VSHUFFLE_YUZU, vm_r1s2, c+1, 1, 2) ;

	VFADD(vrin_c1r2s0, VE_VSHUFFLE_YUZU, vm_r2s0, c+1, 2, 0) ;
	VFADD(vrin_c1r2s1, VE_VSHUFFLE_YUZU, vm_r2s1, c+1, 2, 1) ;
	VFADD(vrin_c1r2s2, VE_VSHUFFLE_YUZU, vm_r2s2, c+1, 2, 2) ;

#undef VFADD
#undef FILTER_OFFSET
    } // inChannel

    if( remain ) {
      _vel_vstu_vssl(vrsum0,  4, pOut+outIndex + 0 * outHeight*outWidth, vl) ;
    }
    for(int64_t kk=0; kk<nPacked; kk++) {
      _vel_vstu_vssl(vrsum[kk], 4, pOut+outIndex + (2*kk+remain)   * outHeight*outWidth, vl) ;
      _vel_vstl_vssl(vrsum[kk], 4, pOut+outIndex + (2*kk+remain+1) * outHeight*outWidth, vl) ;
    }

    outIndex += vl ;
  } // outPixels

}


template<int NUMKERNEL, bool ADDBIAS>
static __attribute__((noinline)) void func_inoaligned_filternchw_avoid_l1m(
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
  const int64_t strideHeight,
  const int64_t strideWidth,
  const int64_t padHeight,
  const int64_t padWidth,
  const int64_t dilationHeight,
  const int64_t dilationWidth,
  const int64_t inGroupOffset,
  const int64_t outGroupOffset,
  const int64_t biasGroupOffset,
  const int64_t kernGroupOffset,
  const int64_t nY,
  const int64_t n,
  const int64_t k
)
{
  float __attribute__ ((aligned(8))) filter[NUMKERNEL*9*256] ;
  uint64_t* filter_u64 = (uint64_t*) filter ;

  int64_t outIndex = outGroupOffset + (n * outChannel + k  ) * outHeight*outWidth ;

  const int64_t remain  = NUMKERNEL & 0x1 ;
  const int64_t nPacked = NUMKERNEL >> 1 ;

  const float   bias0  = ADDBIAS ?  pBias[biasGroupOffset+k+ 0] : 0.f ;
  int64_t bias[nPacked] ;
#pragma clang loop unroll(full)
  for(int64_t kk=0; kk<nPacked; kk++) bias[kk] = ADDBIAS ?  _vel_pack_f32p(pBias+biasGroupOffset+k+2*kk+remain, pBias+biasGroupOffset+k+2*kk+remain+1) : 0UL ;

  const int64_t maxvl = nY * outWidth ;

  __vr vrseq = _vel_vseq_vl(maxvl) ;
  __vr vry  = _vel_vdivsl_vvsl(vrseq, outWidth, maxvl) ;
  __vr vrx  = _vel_vsubsl_vvvl(vrseq, _vel_vmulul_vsvl(outWidth,vry, maxvl), maxvl) ;

  __vr vri   = _vel_vaddsl_vsvl(-padHeight, _vel_vmulsl_vsvl(strideHeight, vry, maxvl), maxvl) ;
  __vr vrj   = _vel_vaddsl_vsvl(-padWidth,  _vel_vmulsl_vsvl(strideWidth,  vrx, maxvl), maxvl) ;

  for (int64_t y=0; y<outHeight; y+=nY)
  {
    const int64_t vl = outWidth * (outHeight - y < nY ? outHeight - y : nY) ;
    const int64_t op = y * outWidth ;

    __vr vrsum0  = _vel_vbrds_vsl(bias0, vl) ;
    __vr vrsum[nPacked] ;
#pragma clang loop unroll(full)
    for(int64_t kk=0; kk<nPacked; kk++) vrsum[kk] = _vel_pvbrd_vsl(bias[kk], vl) ;

    __vr vrh_r0 = _vel_vaddsl_vsvl(0*dilationHeight+y*strideHeight, vri, vl) ;
    __vr vrh_r1 = _vel_vaddsl_vsvl(1*dilationHeight+y*strideHeight, vri, vl) ;
    __vr vrh_r2 = _vel_vaddsl_vsvl(2*dilationHeight+y*strideHeight, vri, vl) ;
    __vm256 vm_r0 =  _vel_vfmklge_mvl(vrh_r0, vl) ;					// condition(0 <= h)
    __vm256 vm_r2 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(inHeight,vrh_r2, vl), vl) ;	// condition(h < inHeight)

    __vr vrw_s0 = _vel_vaddsl_vsvl(0*dilationWidth,                 vrj, vl) ;
    __vr vrw_s1 = _vel_vaddsl_vsvl(1*dilationWidth,                 vrj, vl) ;
    __vr vrw_s2 = _vel_vaddsl_vsvl(2*dilationWidth,                 vrj, vl) ;
    __vm256 vm_s0 =  _vel_vfmklge_mvl(vrw_s0, vl) ;					// condition(0 <= w)
    __vm256 vm_s2 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(inWidth,vrw_s2, vl), vl) ;	// condition(w < inWidth)

    __vm256 vm_r0s0 = _vel_andm_mmm(vm_r0,vm_s0) ;
    __vm256 vm_r0s1 = vm_r0 ;
    __vm256 vm_r0s2 = _vel_andm_mmm(vm_r0,vm_s2) ;
    __vm256 vm_r1s0 = vm_s0 ;
    __vm256 vm_r1s1 = _vel_vfmklat_ml(vl) ;
    __vm256 vm_r1s2 = vm_s2 ;
    __vm256 vm_r2s0 = _vel_andm_mmm(vm_r2,vm_s0) ;
    __vm256 vm_r2s1 = vm_r2 ;
    __vm256 vm_r2s2 = _vel_andm_mmm(vm_r2,vm_s2) ;

    for(int64_t c0=0; c0<inChannelGroup; c0+=256) {
      const int64_t clen = inChannelGroup - c0 < 256 ? inChannelGroup - c0 : 256 ;

      const float *pKerValue = pKernel + kernGroupOffset + (k * inChannelGroup + c0) * kernHeight * kernWidth ;

      for(int64_t t=0; t<9*clen; t+=256) {
	const int64_t tvl = 9*clen-t < 256 ? 9*clen-t : 256 ;

	__vr vr[NUMKERNEL] ;
#pragma clang loop unroll(full)
	for(int64_t kk=0; kk<NUMKERNEL; kk++) {
	  vr[kk] = _vel_vldu_vssl(4, pKerValue+ kk*9*inChannelGroup+t, tvl) ;
	}

#pragma clang loop unroll(full)
	for(int64_t kk=0; kk<nPacked; kk++) {
	  __vr vrp = _vel_vshf_vvvsl(vr[2*kk+remain], vr[2*kk+remain+1], VE_VSHUFFLE_YUZU, tvl) ;
	  _vel_vst_vssl(vrp, 8, filter_u64+kk*9*clen+t, tvl) ;
	}
	if( remain ) {
	  _vel_vstu_vssl(vr[0], 4, filter+(NUMKERNEL-1)*9*clen+t, tvl) ;
	}
      }

      int64_t c1 = 0 ;
      if( (clen & 0x01) == 1 ) {
	const int64_t c = c0 + c1 ;

	const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;
	__vr vrpin_c0r0s0 = _vel_vsfa_vvssl(_vel_vaddul_vvvl(vrw_s0, _vel_vmulul_vsvl(inWidth,vrh_r0, vl), vl),
				     2,
				     (uint64_t)pInChannel, vl) ;
	__vr vrin_c0r0s0 = _vel_vgtu_vvssml(vrpin_c0r0s0, 0, 0, vm_r0s0, vl) ;

	__vr vrpin_c0r0s1 = _vel_vaddul_vsvl(4*1*dilationWidth, vrpin_c0r0s0, vl) ;
	__vr vrin_c0r0s1 = _vel_vgtu_vvssml(vrpin_c0r0s1, 0, 0, vm_r0s1, vl) ;

	__vr vrpin_c0r0s2 = _vel_vaddul_vsvl(4*2*dilationWidth, vrpin_c0r0s0, vl) ;
	__vr vrin_c0r0s2 = _vel_vgtu_vvssml(vrpin_c0r0s2, 0, 0, vm_r0s2, vl) ;

	__vr vrpin_c0r1s0 = _vel_vaddul_vsvl(4*1*dilationHeight*inWidth, vrpin_c0r0s0, vl) ;
	__vr vrin_c0r1s0 = _vel_vgtu_vvssml(vrpin_c0r1s0, 0, 0, vm_r1s0, vl) ;

	__vr vrpin_c0r1s1 = _vel_vaddul_vsvl(4*1*dilationWidth, vrpin_c0r1s0, vl) ;
	__vr vrin_c0r1s1 = _vel_vgtu_vvssml(vrpin_c0r1s1, 0, 0, vm_r1s1, vl) ;

	__vr vrpin_c0r1s2 = _vel_vaddul_vsvl(4*2*dilationWidth, vrpin_c0r1s0, vl) ;
	__vr vrin_c0r1s2 = _vel_vgtu_vvssml(vrpin_c0r1s2, 0, 0, vm_r1s2, vl) ;

	__vr vrpin_c0r2s0 = _vel_vaddul_vsvl(4*2*dilationHeight*inWidth, vrpin_c0r0s0, vl) ;
	__vr vrin_c0r2s0 = _vel_vgtu_vvssml(vrpin_c0r2s0, 0, 0, vm_r2s0, vl) ;

	__vr vrpin_c0r2s1 = _vel_vaddul_vsvl(4*1*dilationWidth, vrpin_c0r2s0, vl) ;
	__vr vrin_c0r2s1 = _vel_vgtu_vvssml(vrpin_c0r2s1, 0, 0, vm_r2s1, vl) ;

	__vr vrpin_c0r2s2 = _vel_vaddul_vsvl(4*2*dilationWidth, vrpin_c0r2s0, vl) ;
	__vr vrin_c0r2s2 = _vel_vgtu_vvssml(vrpin_c0r2s2, 0, 0, vm_r2s2, vl) ;


#define VFADD(VRIN, VE_VSHUFFLE_MODE, VM, C, R, S)					\
	{										\
	  __vr vrinP = _vel_vshf_vvvsl(VRIN, VRIN, VE_VSHUFFLE_MODE, vl) ;		\
	  vrinP = _vel_vmrg_vsvml(0UL, vrinP, VM, vl) ;					\
	  if(remain) {									\
            const float kerValue0  = filter[(NUMKERNEL-1)*9*clen+9*(C)+(3*(R)+(S))] ;	\
	    vrsum0 = _vel_vfmads_vvsvl(vrsum0, kerValue0, vrinP, vl) ;			\
	  }										\
	  _Pragma("clang loop unroll(full)")						\
	  for(int64_t kk=0; kk<nPacked; kk++) {						\
	    const uint64_t kerValue = filter_u64[kk*9*clen+9*(C)+(3*(R)+(S))] ;		\
	    vrsum[kk] = _vel_pvfmad_vvsvl(vrsum[kk], kerValue, vrinP, vl) ;		\
	  }										\
	}

	VFADD(vrin_c0r0s0, VE_VSHUFFLE_YUZU, vm_r0s0, c1, 0, 0) ;
	VFADD(vrin_c0r0s1, VE_VSHUFFLE_YUZU, vm_r0s1, c1, 0, 1) ;
	VFADD(vrin_c0r0s2, VE_VSHUFFLE_YUZU, vm_r0s2, c1, 0, 2) ;

	VFADD(vrin_c0r1s0, VE_VSHUFFLE_YUZU, vm_r1s0, c1, 1, 0) ;
	VFADD(vrin_c0r1s1, VE_VSHUFFLE_YUZU, vm_r1s1, c1, 1, 1) ;
	VFADD(vrin_c0r1s2, VE_VSHUFFLE_YUZU, vm_r1s2, c1, 1, 2) ;

	VFADD(vrin_c0r2s0, VE_VSHUFFLE_YUZU, vm_r2s0, c1, 2, 0) ;
	VFADD(vrin_c0r2s1, VE_VSHUFFLE_YUZU, vm_r2s1, c1, 2, 1) ;
	VFADD(vrin_c0r2s2, VE_VSHUFFLE_YUZU, vm_r2s2, c1, 2, 2) ;

	c1++ ;
      }
      if( ((clen>>1) & 0x01) == 1 ) {
	const int64_t c = c0 + c1 ;

	const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;
	__vr vrpin_c0r0s0 = _vel_vsfa_vvssl(_vel_vaddul_vvvl(vrw_s0, _vel_vmulul_vsvl(inWidth,vrh_r0, vl), vl),
				     2,
				     (uint64_t)pInChannel, vl) ;
	__vr vrin_c0r0s0 = _vel_vgtu_vvssml(vrpin_c0r0s0, 0, 0, vm_r0s0, vl) ;

	__vr vrpin_c0r0s1 = _vel_vaddul_vsvl(4*1*dilationWidth, vrpin_c0r0s0, vl) ;
	__vr vrin_c0r0s1 = _vel_vgtu_vvssml(vrpin_c0r0s1, 0, 0, vm_r0s1, vl) ;

	__vr vrpin_c0r0s2 = _vel_vaddul_vsvl(4*2*dilationWidth, vrpin_c0r0s0, vl) ;
	__vr vrin_c0r0s2 = _vel_vgtu_vvssml(vrpin_c0r0s2, 0, 0, vm_r0s2, vl) ;

	__vr vrpin_c0r1s0 = _vel_vaddul_vsvl(4*1*dilationHeight*inWidth, vrpin_c0r0s0, vl) ;
	__vr vrin_c0r1s0 = _vel_vgtu_vvssml(vrpin_c0r1s0, 0, 0, vm_r1s0, vl) ;

	__vr vrpin_c0r1s1 = _vel_vaddul_vsvl(4*1*dilationWidth, vrpin_c0r1s0, vl) ;
	__vr vrin_c0r1s1 = _vel_vgtu_vvssml(vrpin_c0r1s1, 0, 0, vm_r1s1, vl) ;

	__vr vrpin_c0r1s2 = _vel_vaddul_vsvl(4*2*dilationWidth, vrpin_c0r1s0, vl) ;
	__vr vrin_c0r1s2 = _vel_vgtu_vvssml(vrpin_c0r1s2, 0, 0, vm_r1s2, vl) ;

	__vr vrpin_c0r2s0 = _vel_vaddul_vsvl(4*2*dilationHeight*inWidth, vrpin_c0r0s0, vl) ;
	__vr vrin_c0r2s0 = _vel_vgtu_vvssml(vrpin_c0r2s0, 0, 0, vm_r2s0, vl) ;

	__vr vrpin_c0r2s1 = _vel_vaddul_vsvl(4*1*dilationWidth, vrpin_c0r2s0, vl) ;
	__vr vrin_c0r2s1 = _vel_vgtu_vvssml(vrpin_c0r2s1, 0, 0, vm_r2s1, vl) ;

	__vr vrpin_c0r2s2 = _vel_vaddul_vsvl(4*2*dilationWidth, vrpin_c0r2s0, vl) ;
	__vr vrin_c0r2s2 = _vel_vgtu_vvssml(vrpin_c0r2s2, 0, 0, vm_r2s2, vl) ;


	__vr vrpin_c1r0s0 = _vel_vaddul_vsvl(4*1*inHeight*inWidth, vrpin_c0r0s0, vl) ;
	__vr vrin_c1r0s0 = _vel_vgtu_vvssml(vrpin_c1r0s0, 0, 0, vm_r0s0, vl) ;

	__vr vrpin_c1r0s1 = _vel_vaddul_vsvl(4*1*dilationWidth, vrpin_c1r0s0, vl) ;
	__vr vrin_c1r0s1 = _vel_vgtu_vvssml(vrpin_c1r0s1, 0, 0, vm_r0s1, vl) ;

	__vr vrpin_c1r0s2 = _vel_vaddul_vsvl(4*2*dilationWidth, vrpin_c1r0s0, vl) ;
	__vr vrin_c1r0s2 = _vel_vgtu_vvssml(vrpin_c1r0s2, 0, 0, vm_r0s2, vl) ;

	__vr vrpin_c1r1s0 = _vel_vaddul_vsvl(4*1*dilationHeight*inWidth, vrpin_c1r0s0, vl) ;
	__vr vrin_c1r1s0 = _vel_vgtu_vvssml(vrpin_c1r1s0, 0, 0, vm_r1s0, vl) ;

	__vr vrpin_c1r1s1 = _vel_vaddul_vsvl(4*1*dilationWidth, vrpin_c1r1s0, vl) ;
	__vr vrin_c1r1s1 = _vel_vgtu_vvssml(vrpin_c1r1s1, 0, 0, vm_r1s1, vl) ;

	__vr vrpin_c1r1s2 = _vel_vaddul_vsvl(4*2*dilationWidth, vrpin_c1r1s0, vl) ;
	__vr vrin_c1r1s2 = _vel_vgtu_vvssml(vrpin_c1r1s2, 0, 0, vm_r1s2, vl) ;

	__vr vrpin_c1r2s0 = _vel_vaddul_vsvl(4*2*dilationHeight*inWidth, vrpin_c1r0s0, vl) ;
	__vr vrin_c1r2s0 = _vel_vgtu_vvssml(vrpin_c1r2s0, 0, 0, vm_r2s0, vl) ;

	__vr vrpin_c1r2s1 = _vel_vaddul_vsvl(4*1*dilationWidth, vrpin_c1r2s0, vl) ;
	__vr vrin_c1r2s1 = _vel_vgtu_vvssml(vrpin_c1r2s1, 0, 0, vm_r2s1, vl) ;

	__vr vrpin_c1r2s2 = _vel_vaddul_vsvl(4*2*dilationWidth, vrpin_c1r2s0, vl) ;
	__vr vrin_c1r2s2 = _vel_vgtu_vvssml(vrpin_c1r2s2, 0, 0, vm_r2s2, vl) ;

	VFADD(vrin_c0r0s0, VE_VSHUFFLE_YUZU, vm_r0s0, c1, 0, 0) ;
	VFADD(vrin_c0r0s1, VE_VSHUFFLE_YUZU, vm_r0s1, c1, 0, 1) ;
	VFADD(vrin_c0r0s2, VE_VSHUFFLE_YUZU, vm_r0s2, c1, 0, 2) ;

	VFADD(vrin_c0r1s0, VE_VSHUFFLE_YUZU, vm_r1s0, c1, 1, 0) ;
	VFADD(vrin_c0r1s1, VE_VSHUFFLE_YUZU, vm_r1s1, c1, 1, 1) ;
	VFADD(vrin_c0r1s2, VE_VSHUFFLE_YUZU, vm_r1s2, c1, 1, 2) ;

	VFADD(vrin_c0r2s0, VE_VSHUFFLE_YUZU, vm_r2s0, c1, 2, 0) ;
	VFADD(vrin_c0r2s1, VE_VSHUFFLE_YUZU, vm_r2s1, c1, 2, 1) ;
	VFADD(vrin_c0r2s2, VE_VSHUFFLE_YUZU, vm_r2s2, c1, 2, 2) ;

	VFADD(vrin_c1r0s0, VE_VSHUFFLE_YUZU, vm_r0s0, c1+1, 0, 0) ;
	VFADD(vrin_c1r0s1, VE_VSHUFFLE_YUZU, vm_r0s1, c1+1, 0, 1) ;
	VFADD(vrin_c1r0s2, VE_VSHUFFLE_YUZU, vm_r0s2, c1+1, 0, 2) ;

	VFADD(vrin_c1r1s0, VE_VSHUFFLE_YUZU, vm_r1s0, c1+1, 1, 0) ;
	VFADD(vrin_c1r1s1, VE_VSHUFFLE_YUZU, vm_r1s1, c1+1, 1, 1) ;
	VFADD(vrin_c1r1s2, VE_VSHUFFLE_YUZU, vm_r1s2, c1+1, 1, 2) ;

	VFADD(vrin_c1r2s0, VE_VSHUFFLE_YUZU, vm_r2s0, c1+1, 2, 0) ;
	VFADD(vrin_c1r2s1, VE_VSHUFFLE_YUZU, vm_r2s1, c1+1, 2, 1) ;
	VFADD(vrin_c1r2s2, VE_VSHUFFLE_YUZU, vm_r2s2, c1+1, 2, 2) ;

	c1+=2 ;
      }
      for( ; c1 < clen ; c1+=4 ) {
	const int64_t c = c0 + c1 ;

	const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;
	__vr vrpin_c0r0s0 = _vel_vsfa_vvssl(_vel_vaddul_vvvl(vrw_s0, _vel_vmulul_vsvl(inWidth,vrh_r0, vl), vl),
				     2,
				     (uint64_t)pInChannel, vl) ;
	__vr vrin_c0r0s0 = _vel_vgtu_vvssml(vrpin_c0r0s0, 0, 0, vm_r0s0, vl) ;

	__vr vrpin_c0r0s1 = _vel_vaddul_vsvl(4*1*dilationWidth, vrpin_c0r0s0, vl) ;
	__vr vrin_c0r0s1 = _vel_vgtu_vvssml(vrpin_c0r0s1, 0, 0, vm_r0s1, vl) ;

	__vr vrpin_c0r0s2 = _vel_vaddul_vsvl(4*2*dilationWidth, vrpin_c0r0s0, vl) ;
	__vr vrin_c0r0s2 = _vel_vgtu_vvssml(vrpin_c0r0s2, 0, 0, vm_r0s2, vl) ;

	__vr vrpin_c0r1s0 = _vel_vaddul_vsvl(4*1*dilationHeight*inWidth, vrpin_c0r0s0, vl) ;
	__vr vrin_c0r1s0 = _vel_vgtu_vvssml(vrpin_c0r1s0, 0, 0, vm_r1s0, vl) ;

	__vr vrpin_c0r1s1 = _vel_vaddul_vsvl(4*1*dilationWidth, vrpin_c0r1s0, vl) ;
	__vr vrin_c0r1s1 = _vel_vgtu_vvssml(vrpin_c0r1s1, 0, 0, vm_r1s1, vl) ;

	__vr vrpin_c0r1s2 = _vel_vaddul_vsvl(4*2*dilationWidth, vrpin_c0r1s0, vl) ;
	__vr vrin_c0r1s2 = _vel_vgtu_vvssml(vrpin_c0r1s2, 0, 0, vm_r1s2, vl) ;

	__vr vrpin_c0r2s0 = _vel_vaddul_vsvl(4*2*dilationHeight*inWidth, vrpin_c0r0s0, vl) ;
	__vr vrin_c0r2s0 = _vel_vgtu_vvssml(vrpin_c0r2s0, 0, 0, vm_r2s0, vl) ;

	__vr vrpin_c0r2s1 = _vel_vaddul_vsvl(4*1*dilationWidth, vrpin_c0r2s0, vl) ;
	__vr vrin_c0r2s1 = _vel_vgtu_vvssml(vrpin_c0r2s1, 0, 0, vm_r2s1, vl) ;

	__vr vrpin_c0r2s2 = _vel_vaddul_vsvl(4*2*dilationWidth, vrpin_c0r2s0, vl) ;
	__vr vrin_c0r2s2 = _vel_vgtu_vvssml(vrpin_c0r2s2, 0, 0, vm_r2s2, vl) ;


	__vr vrpin_c1r0s0 = _vel_vaddul_vsvl(4*1*inHeight*inWidth, vrpin_c0r0s0, vl) ;
	__vr vrin_c1r0s0 = _vel_vgtu_vvssml(vrpin_c1r0s0, 0, 0, vm_r0s0, vl) ;

	__vr vrpin_c1r0s1 = _vel_vaddul_vsvl(4*1*dilationWidth, vrpin_c1r0s0, vl) ;
	__vr vrin_c1r0s1 = _vel_vgtu_vvssml(vrpin_c1r0s1, 0, 0, vm_r0s1, vl) ;

	__vr vrpin_c1r0s2 = _vel_vaddul_vsvl(4*2*dilationWidth, vrpin_c1r0s0, vl) ;
	__vr vrin_c1r0s2 = _vel_vgtu_vvssml(vrpin_c1r0s2, 0, 0, vm_r0s2, vl) ;

	__vr vrpin_c1r1s0 = _vel_vaddul_vsvl(4*1*dilationHeight*inWidth, vrpin_c1r0s0, vl) ;
	__vr vrin_c1r1s0 = _vel_vgtu_vvssml(vrpin_c1r1s0, 0, 0, vm_r1s0, vl) ;

	__vr vrpin_c1r1s1 = _vel_vaddul_vsvl(4*1*dilationWidth, vrpin_c1r1s0, vl) ;
	__vr vrin_c1r1s1 = _vel_vgtu_vvssml(vrpin_c1r1s1, 0, 0, vm_r1s1, vl) ;

	__vr vrpin_c1r1s2 = _vel_vaddul_vsvl(4*2*dilationWidth, vrpin_c1r1s0, vl) ;
	__vr vrin_c1r1s2 = _vel_vgtu_vvssml(vrpin_c1r1s2, 0, 0, vm_r1s2, vl) ;

	__vr vrpin_c1r2s0 = _vel_vaddul_vsvl(4*2*dilationHeight*inWidth, vrpin_c1r0s0, vl) ;
	__vr vrin_c1r2s0 = _vel_vgtu_vvssml(vrpin_c1r2s0, 0, 0, vm_r2s0, vl) ;

	__vr vrpin_c1r2s1 = _vel_vaddul_vsvl(4*1*dilationWidth, vrpin_c1r2s0, vl) ;
	__vr vrin_c1r2s1 = _vel_vgtu_vvssml(vrpin_c1r2s1, 0, 0, vm_r2s1, vl) ;

	__vr vrpin_c1r2s2 = _vel_vaddul_vsvl(4*2*dilationWidth, vrpin_c1r2s0, vl) ;
	__vr vrin_c1r2s2 = _vel_vgtu_vvssml(vrpin_c1r2s2, 0, 0, vm_r2s2, vl) ;


	__vr vrpin_c2r0s0 = _vel_vaddul_vsvl(4*1*inHeight*inWidth, vrpin_c1r0s0, vl) ;
	__vr vrin_c2r0s0 = _vel_vgtu_vvssml(vrpin_c2r0s0, 0, 0, vm_r0s0, vl) ;

	__vr vrpin_c2r0s1 = _vel_vaddul_vsvl(4*1*dilationWidth, vrpin_c2r0s0, vl) ;
	__vr vrin_c2r0s1 = _vel_vgtu_vvssml(vrpin_c2r0s1, 0, 0, vm_r0s1, vl) ;

	__vr vrpin_c2r0s2 = _vel_vaddul_vsvl(4*2*dilationWidth, vrpin_c2r0s0, vl) ;
	__vr vrin_c2r0s2 = _vel_vgtu_vvssml(vrpin_c2r0s2, 0, 0, vm_r0s2, vl) ;

	__vr vrpin_c2r1s0 = _vel_vaddul_vsvl(4*1*dilationHeight*inWidth, vrpin_c2r0s0, vl) ;
	__vr vrin_c2r1s0 = _vel_vgtu_vvssml(vrpin_c2r1s0, 0, 0, vm_r1s0, vl) ;

	__vr vrpin_c2r1s1 = _vel_vaddul_vsvl(4*1*dilationWidth, vrpin_c2r1s0, vl) ;
	__vr vrin_c2r1s1 = _vel_vgtu_vvssml(vrpin_c2r1s1, 0, 0, vm_r1s1, vl) ;

	__vr vrpin_c2r1s2 = _vel_vaddul_vsvl(4*2*dilationWidth, vrpin_c2r1s0, vl) ;
	__vr vrin_c2r1s2 = _vel_vgtu_vvssml(vrpin_c2r1s2, 0, 0, vm_r1s2, vl) ;

	__vr vrpin_c2r2s0 = _vel_vaddul_vsvl(4*2*dilationHeight*inWidth, vrpin_c2r0s0, vl) ;
	__vr vrin_c2r2s0 = _vel_vgtu_vvssml(vrpin_c2r2s0, 0, 0, vm_r2s0, vl) ;

	__vr vrpin_c2r2s1 = _vel_vaddul_vsvl(4*1*dilationWidth, vrpin_c2r2s0, vl) ;
	__vr vrin_c2r2s1 = _vel_vgtu_vvssml(vrpin_c2r2s1, 0, 0, vm_r2s1, vl) ;

	__vr vrpin_c2r2s2 = _vel_vaddul_vsvl(4*2*dilationWidth, vrpin_c2r2s0, vl) ;
	__vr vrin_c2r2s2 = _vel_vgtu_vvssml(vrpin_c2r2s2, 0, 0, vm_r2s2, vl) ;


	__vr vrpin_c3r0s0 = _vel_vaddul_vsvl(4*1*inHeight*inWidth, vrpin_c2r0s0, vl) ;
	__vr vrin_c3r0s0 = _vel_vgtu_vvssml(vrpin_c3r0s0, 0, 0, vm_r0s0, vl) ;

	__vr vrpin_c3r0s1 = _vel_vaddul_vsvl(4*1*dilationWidth, vrpin_c3r0s0, vl) ;
	__vr vrin_c3r0s1 = _vel_vgtu_vvssml(vrpin_c3r0s1, 0, 0, vm_r0s1, vl) ;

	__vr vrpin_c3r0s2 = _vel_vaddul_vsvl(4*2*dilationWidth, vrpin_c3r0s0, vl) ;
	__vr vrin_c3r0s2 = _vel_vgtu_vvssml(vrpin_c3r0s2, 0, 0, vm_r0s2, vl) ;

	__vr vrpin_c3r1s0 = _vel_vaddul_vsvl(4*1*dilationHeight*inWidth, vrpin_c3r0s0, vl) ;
	__vr vrin_c3r1s0 = _vel_vgtu_vvssml(vrpin_c3r1s0, 0, 0, vm_r1s0, vl) ;

	__vr vrpin_c3r1s1 = _vel_vaddul_vsvl(4*1*dilationWidth, vrpin_c3r1s0, vl) ;
	__vr vrin_c3r1s1 = _vel_vgtu_vvssml(vrpin_c3r1s1, 0, 0, vm_r1s1, vl) ;

	__vr vrpin_c3r1s2 = _vel_vaddul_vsvl(4*2*dilationWidth, vrpin_c3r1s0, vl) ;
	__vr vrin_c3r1s2 = _vel_vgtu_vvssml(vrpin_c3r1s2, 0, 0, vm_r1s2, vl) ;

	__vr vrpin_c3r2s0 = _vel_vaddul_vsvl(4*2*dilationHeight*inWidth, vrpin_c3r0s0, vl) ;
	__vr vrin_c3r2s0 = _vel_vgtu_vvssml(vrpin_c3r2s0, 0, 0, vm_r2s0, vl) ;

	__vr vrpin_c3r2s1 = _vel_vaddul_vsvl(4*1*dilationWidth, vrpin_c3r2s0, vl) ;
	__vr vrin_c3r2s1 = _vel_vgtu_vvssml(vrpin_c3r2s1, 0, 0, vm_r2s1, vl) ;

	__vr vrpin_c3r2s2 = _vel_vaddul_vsvl(4*2*dilationWidth, vrpin_c3r2s0, vl) ;
	__vr vrin_c3r2s2 = _vel_vgtu_vvssml(vrpin_c3r2s2, 0, 0, vm_r2s2, vl) ;


	VFADD(vrin_c0r0s0, VE_VSHUFFLE_YUZU, vm_r0s0, c1, 0, 0) ;
	VFADD(vrin_c0r0s1, VE_VSHUFFLE_YUZU, vm_r0s1, c1, 0, 1) ;
	VFADD(vrin_c0r0s2, VE_VSHUFFLE_YUZU, vm_r0s2, c1, 0, 2) ;

	VFADD(vrin_c0r1s0, VE_VSHUFFLE_YUZU, vm_r1s0, c1, 1, 0) ;
	VFADD(vrin_c0r1s1, VE_VSHUFFLE_YUZU, vm_r1s1, c1, 1, 1) ;
	VFADD(vrin_c0r1s2, VE_VSHUFFLE_YUZU, vm_r1s2, c1, 1, 2) ;

	VFADD(vrin_c0r2s0, VE_VSHUFFLE_YUZU, vm_r2s0, c1, 2, 0) ;
	VFADD(vrin_c0r2s1, VE_VSHUFFLE_YUZU, vm_r2s1, c1, 2, 1) ;
	VFADD(vrin_c0r2s2, VE_VSHUFFLE_YUZU, vm_r2s2, c1, 2, 2) ;

	VFADD(vrin_c1r0s0, VE_VSHUFFLE_YUZU, vm_r0s0, c1+1, 0, 0) ;
	VFADD(vrin_c1r0s1, VE_VSHUFFLE_YUZU, vm_r0s1, c1+1, 0, 1) ;
	VFADD(vrin_c1r0s2, VE_VSHUFFLE_YUZU, vm_r0s2, c1+1, 0, 2) ;

	VFADD(vrin_c1r1s0, VE_VSHUFFLE_YUZU, vm_r1s0, c1+1, 1, 0) ;
	VFADD(vrin_c1r1s1, VE_VSHUFFLE_YUZU, vm_r1s1, c1+1, 1, 1) ;
	VFADD(vrin_c1r1s2, VE_VSHUFFLE_YUZU, vm_r1s2, c1+1, 1, 2) ;

	VFADD(vrin_c1r2s0, VE_VSHUFFLE_YUZU, vm_r2s0, c1+1, 2, 0) ;
	VFADD(vrin_c1r2s1, VE_VSHUFFLE_YUZU, vm_r2s1, c1+1, 2, 1) ;
	VFADD(vrin_c1r2s2, VE_VSHUFFLE_YUZU, vm_r2s2, c1+1, 2, 2) ;

	VFADD(vrin_c2r0s0, VE_VSHUFFLE_YUZU, vm_r0s0, c1+2, 0, 0) ;
	VFADD(vrin_c2r0s1, VE_VSHUFFLE_YUZU, vm_r0s1, c1+2, 0, 1) ;
	VFADD(vrin_c2r0s2, VE_VSHUFFLE_YUZU, vm_r0s2, c1+2, 0, 2) ;

	VFADD(vrin_c2r1s0, VE_VSHUFFLE_YUZU, vm_r1s0, c1+2, 1, 0) ;
	VFADD(vrin_c2r1s1, VE_VSHUFFLE_YUZU, vm_r1s1, c1+2, 1, 1) ;
	VFADD(vrin_c2r1s2, VE_VSHUFFLE_YUZU, vm_r1s2, c1+2, 1, 2) ;

	VFADD(vrin_c2r2s0, VE_VSHUFFLE_YUZU, vm_r2s0, c1+2, 2, 0) ;
	VFADD(vrin_c2r2s1, VE_VSHUFFLE_YUZU, vm_r2s1, c1+2, 2, 1) ;
	VFADD(vrin_c2r2s2, VE_VSHUFFLE_YUZU, vm_r2s2, c1+2, 2, 2) ;

	VFADD(vrin_c3r0s0, VE_VSHUFFLE_YUZU, vm_r0s0, c1+3, 0, 0) ;
	VFADD(vrin_c3r0s1, VE_VSHUFFLE_YUZU, vm_r0s1, c1+3, 0, 1) ;
	VFADD(vrin_c3r0s2, VE_VSHUFFLE_YUZU, vm_r0s2, c1+3, 0, 2) ;

	VFADD(vrin_c3r1s0, VE_VSHUFFLE_YUZU, vm_r1s0, c1+3, 1, 0) ;
	VFADD(vrin_c3r1s1, VE_VSHUFFLE_YUZU, vm_r1s1, c1+3, 1, 1) ;
	VFADD(vrin_c3r1s2, VE_VSHUFFLE_YUZU, vm_r1s2, c1+3, 1, 2) ;

	VFADD(vrin_c3r2s0, VE_VSHUFFLE_YUZU, vm_r2s0, c1+3, 2, 0) ;
	VFADD(vrin_c3r2s1, VE_VSHUFFLE_YUZU, vm_r2s1, c1+3, 2, 1) ;
	VFADD(vrin_c3r2s2, VE_VSHUFFLE_YUZU, vm_r2s2, c1+3, 2, 2) ;

#undef VFADD
      }
    }

    if( remain ) {
	_vel_vstu_vssl(vrsum0,  4, pOut+outIndex + 0 * outHeight*outWidth, vl) ;
    }
    for(int64_t kk=0; kk<nPacked; kk++) {
	_vel_vstu_vssl(vrsum[kk], 4, pOut+outIndex + (2*kk+remain)   * outHeight*outWidth, vl) ;
	_vel_vstl_vssl(vrsum[kk], 4, pOut+outIndex + (2*kk+remain+1) * outHeight*outWidth, vl) ;
    }

    outIndex += vl ;
  } // outPixels
}


template<filterLayout_t FLAYOUT, int NUMKERNEL, bool ADDBIAS>
static __attribute__((noinline)) void func_ialigned(
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
  const int64_t strideHeight,
  const int64_t strideWidth,
  const int64_t padHeight,
  const int64_t padWidth,
  const int64_t dilationHeight,
  const int64_t dilationWidth,
  const int64_t inGroupOffset,
  const int64_t outGroupOffset,
  const int64_t biasGroupOffset,
  const int64_t kernGroupOffset,
  const int64_t nY,
  const int64_t n,
  const int64_t k
)
{
  int64_t outIndex = outGroupOffset + (n * outChannel + k  ) * outHeight*outWidth ;

  const int64_t remain  = NUMKERNEL & 0x1 ;
  const int64_t nPacked = NUMKERNEL >> 1 ;

  const float   bias0  = ADDBIAS ?  pBias[biasGroupOffset+k+ 0] : 0.f ;
  int64_t bias[nPacked] ;
#pragma clang loop unroll(full)
  for(int64_t kk=0; kk<nPacked; kk++) bias[kk] = ADDBIAS ?  _vel_pack_f32p(pBias+biasGroupOffset+k+2*kk+remain, pBias+biasGroupOffset+k+2*kk+remain+1) : 0UL ;

  const int64_t maxvl = nY * outWidth ;

  __vr vrseq = _vel_vseq_vl(maxvl) ;
  __vr vry  = _vel_vdivsl_vvsl(vrseq, outWidth, maxvl) ;
  __vr vrx  = _vel_vsubsl_vvvl(vrseq, _vel_vmulul_vsvl(outWidth,vry, maxvl), maxvl) ;

  __vr vri   = _vel_vaddsl_vsvl(-padHeight, _vel_vmulsl_vsvl(strideHeight, vry, maxvl), maxvl) ;
  __vr vrj   = _vel_vaddsl_vsvl(-padWidth,  _vel_vmulsl_vsvl(strideWidth,  vrx, maxvl), maxvl) ;

  for (int64_t y=0; y<outHeight; y+=nY)
  {
    const int64_t vl = outWidth * (outHeight - y < nY ? outHeight - y : nY) ;
    const int64_t op = y * outWidth ;

    __vr vrsum0  = _vel_vbrds_vsl(bias0, vl) ;
    __vr vrsum[nPacked] ;
#pragma clang loop unroll(full)
    for(int64_t kk=0; kk<nPacked; kk++) vrsum[kk] = _vel_pvbrd_vsl(bias[kk], vl) ;

    __vr vrh_r0 = _vel_vaddsl_vsvl(0*dilationHeight+y*strideHeight, vri, vl) ;
    __vr vrh_r1 = _vel_vaddsl_vsvl(1*dilationHeight+y*strideHeight, vri, vl) ;
    __vr vrh_r2 = _vel_vaddsl_vsvl(2*dilationHeight+y*strideHeight, vri, vl) ;
    __vm256 vm_r0 =  _vel_vfmklge_mvl(vrh_r0, vl) ;					// condition(0 <= h)
    __vm256 vm_r2 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(inHeight,vrh_r2, vl), vl) ;	// condition(h < inHeight)

    __vr vrw_s0 = _vel_vaddsl_vsvl(0*dilationWidth,                 vrj, vl) ;
    __vr vrw_s1 = _vel_vaddsl_vsvl(1*dilationWidth,                 vrj, vl) ;
    __vr vrw_s2 = _vel_vaddsl_vsvl(2*dilationWidth,                 vrj, vl) ;
    __vm256 vm_s0 =  _vel_vfmklge_mvl(vrw_s0, vl) ;					// condition(0 <= w)
    __vm256 vm_s2 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(inWidth,vrw_s2, vl), vl) ;	// condition(w < inWidth)

    __vm256 vm_r0s0 = _vel_andm_mmm(vm_r0,vm_s0) ;
    __vm256 vm_r0s1 = vm_r0 ;
    __vm256 vm_r0s2 = _vel_andm_mmm(vm_r0,vm_s2) ;
    __vm256 vm_r1s0 = vm_s0 ;
    __vm256 vm_r1s1 = _vel_vfmklat_ml(vl) ;
    __vm256 vm_r1s2 = vm_s2 ;
    __vm256 vm_r2s0 = _vel_andm_mmm(vm_r2,vm_s0) ;
    __vm256 vm_r2s1 = vm_r2 ;
    __vm256 vm_r2s2 = _vel_andm_mmm(vm_r2,vm_s2) ;

    int64_t c=0 ;
    if( (inChannelGroup & 0x1) == 1 ) {
      const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;
      __vr vrpin_c0r0s0 = _vel_vsfa_vvssl(_vel_vaddul_vvvl(vrw_s0, _vel_vmulul_vsvl(inWidth,vrh_r0, vl), vl),
				   2,
				   (uint64_t)pInChannel, vl) ;
      __vr vrin_c0r0s0 = _vel_vgtu_vvssml(vrpin_c0r0s0, 0, 0, vm_r0s0, vl) ;

      __vr vrpin_c0r0s1 = _vel_vaddul_vsvl(4*1*dilationWidth, vrpin_c0r0s0, vl) ;
      __vr vrin_c0r0s1 = _vel_vgt_vvssml(vrpin_c0r0s1, 0, 0, vm_r0s1, vl) ;

      __vr vrpin_c0r1s0 = _vel_vaddul_vsvl(4*1*dilationHeight*inWidth, vrpin_c0r0s0, vl) ;
      __vr vrin_c0r1s0 = _vel_vgtu_vvssml(vrpin_c0r1s0, 0, 0, vm_r1s0, vl) ;

      __vr vrpin_c0r1s1 = _vel_vaddul_vsvl(4*1*dilationWidth, vrpin_c0r1s0, vl) ;
      __vr vrin_c0r1s1 = _vel_vgt_vvssml(vrpin_c0r1s1, 0, 0, vm_r1s1, vl) ;

      __vr vrpin_c0r2s0 = _vel_vaddul_vsvl(4*2*dilationHeight*inWidth, vrpin_c0r0s0, vl) ;
      __vr vrin_c0r2s0 = _vel_vgtu_vvssml(vrpin_c0r2s0, 0, 0, vm_r2s0, vl) ;

      __vr vrpin_c0r2s1 = _vel_vaddul_vsvl(4*1*dilationWidth, vrpin_c0r2s0, vl) ;
      __vr vrin_c0r2s1 = _vel_vgt_vvssml(vrpin_c0r2s1, 0, 0, vm_r2s1, vl) ;


#define FILTER_OFFSET(k,c,r,s) ( kernGroupOffset + filter_index<FLAYOUT>(k,c,r,s, inChannelGroup, outChannelGroup, kernHeight, kernWidth) )

#define VFADD(VRIN, VE_VSHUFFLE_MODE, VM, C, R, S)							\
      {													\
	__vr vrinP = _vel_vshf_vvvsl(VRIN, VRIN, VE_VSHUFFLE_MODE, vl) ;				\
	vrinP = _vel_vmrg_vsvml(0UL, vrinP, VM, vl) ;							\
	if( remain ) {											\
	  const float    kerValue0  = pKernel[FILTER_OFFSET(k+ 0,C,R,S)] ;				\
	  vrsum0 = _vel_vfmads_vvsvl(vrsum0, kerValue0, vrinP, vl) ;					\
	}												\
        _Pragma("clang loop unroll(full)")								\
        for(int64_t kk=0; kk<nPacked; kk++) {								\
	  const uint64_t kerValue = _vel_pack_f32p(pKernel + FILTER_OFFSET(k+2*kk+remain,  C,R,S),	\
						   pKernel + FILTER_OFFSET(k+2*kk+remain+1,C,R,S)) ;	\
	  vrsum[kk] = _vel_pvfmad_vvsvl(vrsum[kk], kerValue, vrinP, vl) ;				\
        }												\
      }

      VFADD(vrin_c0r0s0, VE_VSHUFFLE_YUZU, vm_r0s0, c, 0, 0) ;
      VFADD(vrin_c0r0s1, VE_VSHUFFLE_YLZL, vm_r0s1, c, 0, 1) ;
      VFADD(vrin_c0r0s1, VE_VSHUFFLE_YUZU, vm_r0s2, c, 0, 2) ;

      VFADD(vrin_c0r1s0, VE_VSHUFFLE_YUZU, vm_r1s0, c, 1, 0) ;
      VFADD(vrin_c0r1s1, VE_VSHUFFLE_YLZL, vm_r1s1, c, 1, 1) ;
      VFADD(vrin_c0r1s1, VE_VSHUFFLE_YUZU, vm_r1s2, c, 1, 2) ;

      VFADD(vrin_c0r2s0, VE_VSHUFFLE_YUZU, vm_r2s0, c, 2, 0) ;
      VFADD(vrin_c0r2s1, VE_VSHUFFLE_YLZL, vm_r2s1, c, 2, 1) ;
      VFADD(vrin_c0r2s1, VE_VSHUFFLE_YUZU, vm_r2s2, c, 2, 2) ;

      c+=1 ;
    }
    for( ; c < inChannelGroup ; c+=2 ) {
	const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;
	__vr vrpin_c0r0s0 = _vel_vsfa_vvssl(_vel_vaddul_vvvl(vrw_s0, _vel_vmulul_vsvl(inWidth,vrh_r0, vl), vl),
				     2,
				     (uint64_t)pInChannel, vl) ;
	__vr vrin_c0r0s0 = _vel_vgtu_vvssml(vrpin_c0r0s0, 0, 0, vm_r0s0, vl) ;

	__vr vrpin_c0r0s1 = _vel_vaddul_vsvl(4*1*dilationWidth, vrpin_c0r0s0, vl) ;
	__vr vrin_c0r0s1 = _vel_vgt_vvssml(vrpin_c0r0s1, 0, 0, vm_r0s1, vl) ;

	__vr vrpin_c0r1s0 = _vel_vaddul_vsvl(4*1*dilationHeight*inWidth, vrpin_c0r0s0, vl) ;
	__vr vrin_c0r1s0 = _vel_vgtu_vvssml(vrpin_c0r1s0, 0, 0, vm_r1s0, vl) ;

	__vr vrpin_c0r1s1 = _vel_vaddul_vsvl(4*1*dilationWidth, vrpin_c0r1s0, vl) ;
	__vr vrin_c0r1s1 = _vel_vgt_vvssml(vrpin_c0r1s1, 0, 0, vm_r1s1, vl) ;

	__vr vrpin_c0r2s0 = _vel_vaddul_vsvl(4*2*dilationHeight*inWidth, vrpin_c0r0s0, vl) ;
	__vr vrin_c0r2s0 = _vel_vgtu_vvssml(vrpin_c0r2s0, 0, 0, vm_r2s0, vl) ;

	__vr vrpin_c0r2s1 = _vel_vaddul_vsvl(4*1*dilationWidth, vrpin_c0r2s0, vl) ;
	__vr vrin_c0r2s1 = _vel_vgt_vvssml(vrpin_c0r2s1, 0, 0, vm_r2s1, vl) ;


	__vr vrpin_c1r0s0 = _vel_vaddul_vsvl(4*1*inHeight*inWidth, vrpin_c0r0s0, vl) ;
	__vr vrin_c1r0s0 = _vel_vgtu_vvssml(vrpin_c1r0s0, 0, 0, vm_r0s0, vl) ;

	__vr vrpin_c1r0s1 = _vel_vaddul_vsvl(4*1*dilationWidth, vrpin_c1r0s0, vl) ;
	__vr vrin_c1r0s1 = _vel_vgt_vvssml(vrpin_c1r0s1, 0, 0, vm_r0s1, vl) ;

	__vr vrpin_c1r1s0 = _vel_vaddul_vsvl(4*1*dilationHeight*inWidth, vrpin_c1r0s0, vl) ;
	__vr vrin_c1r1s0 = _vel_vgtu_vvssml(vrpin_c1r1s0, 0, 0, vm_r1s0, vl) ;

	__vr vrpin_c1r1s1 = _vel_vaddul_vsvl(4*1*dilationWidth, vrpin_c1r1s0, vl) ;
	__vr vrin_c1r1s1 = _vel_vgt_vvssml(vrpin_c1r1s1, 0, 0, vm_r1s1, vl) ;

	__vr vrpin_c1r2s0 = _vel_vaddul_vsvl(4*2*dilationHeight*inWidth, vrpin_c1r0s0, vl) ;
	__vr vrin_c1r2s0 = _vel_vgtu_vvssml(vrpin_c1r2s0, 0, 0, vm_r2s0, vl) ;

	__vr vrpin_c1r2s1 = _vel_vaddul_vsvl(4*1*dilationWidth, vrpin_c1r2s0, vl) ;
	__vr vrin_c1r2s1 = _vel_vgt_vvssml(vrpin_c1r2s1, 0, 0, vm_r2s1, vl) ;


	VFADD(vrin_c0r0s0, VE_VSHUFFLE_YUZU, vm_r0s0, c, 0, 0) ;
	VFADD(vrin_c0r0s1, VE_VSHUFFLE_YLZL, vm_r0s1, c, 0, 1) ;
	VFADD(vrin_c0r0s1, VE_VSHUFFLE_YUZU, vm_r0s2, c, 0, 2) ;

	VFADD(vrin_c0r1s0, VE_VSHUFFLE_YUZU, vm_r1s0, c, 1, 0) ;
	VFADD(vrin_c0r1s1, VE_VSHUFFLE_YLZL, vm_r1s1, c, 1, 1) ;
	VFADD(vrin_c0r1s1, VE_VSHUFFLE_YUZU, vm_r1s2, c, 1, 2) ;

	VFADD(vrin_c0r2s0, VE_VSHUFFLE_YUZU, vm_r2s0, c, 2, 0) ;
	VFADD(vrin_c0r2s1, VE_VSHUFFLE_YLZL, vm_r2s1, c, 2, 1) ;
	VFADD(vrin_c0r2s1, VE_VSHUFFLE_YUZU, vm_r2s2, c, 2, 2) ;

	VFADD(vrin_c1r0s0, VE_VSHUFFLE_YUZU, vm_r0s0, c+1, 0, 0) ;
	VFADD(vrin_c1r0s1, VE_VSHUFFLE_YLZL, vm_r0s1, c+1, 0, 1) ;
	VFADD(vrin_c1r0s1, VE_VSHUFFLE_YUZU, vm_r0s2, c+1, 0, 2) ;

	VFADD(vrin_c1r1s0, VE_VSHUFFLE_YUZU, vm_r1s0, c+1, 1, 0) ;
	VFADD(vrin_c1r1s1, VE_VSHUFFLE_YLZL, vm_r1s1, c+1, 1, 1) ;
	VFADD(vrin_c1r1s1, VE_VSHUFFLE_YUZU, vm_r1s2, c+1, 1, 2) ;

	VFADD(vrin_c1r2s0, VE_VSHUFFLE_YUZU, vm_r2s0, c+1, 2, 0) ;
	VFADD(vrin_c1r2s1, VE_VSHUFFLE_YLZL, vm_r2s1, c+1, 2, 1) ;
	VFADD(vrin_c1r2s1, VE_VSHUFFLE_YUZU, vm_r2s2, c+1, 2, 2) ;

#undef VFADD
#undef FILTER_OFFSET
    } // inChannel

    if( remain ) {
      _vel_vstu_vssl(vrsum0,  4, pOut+outIndex + 0 * outHeight*outWidth, vl) ;
    }
    for(int64_t kk=0; kk<nPacked; kk++) {
      _vel_vstu_vssl(vrsum[kk], 4, pOut+outIndex + (2*kk+remain)   * outHeight*outWidth, vl) ;
      _vel_vstl_vssl(vrsum[kk], 4, pOut+outIndex + (2*kk+remain+1) * outHeight*outWidth, vl) ;
    }

    outIndex += vl ;
  } // outPixels
}


template<int NUMKERNEL, bool ADDBIAS>
static __attribute__((noinline)) void func_ialigned_filternchw_avoid_l1m(
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
  const int64_t strideHeight,
  const int64_t strideWidth,
  const int64_t padHeight,
  const int64_t padWidth,
  const int64_t dilationHeight,
  const int64_t dilationWidth,
  const int64_t inGroupOffset,
  const int64_t outGroupOffset,
  const int64_t biasGroupOffset,
  const int64_t kernGroupOffset,
  const int64_t nY,
  const int64_t n,
  const int64_t k
)
{
  float __attribute__ ((aligned(8))) filter[NUMKERNEL*9*256] ;
  uint64_t* filter_u64 = (uint64_t*) filter ;

  int64_t outIndex = outGroupOffset + (n * outChannel + k  ) * outHeight*outWidth ;

  const int64_t remain  = NUMKERNEL & 0x1 ;
  const int64_t nPacked = NUMKERNEL >> 1 ;

  const float   bias0  = ADDBIAS ?  pBias[biasGroupOffset+k+ 0] : 0.f ;
  int64_t bias[nPacked] ;
#pragma clang loop unroll(full)
  for(int64_t kk=0; kk<nPacked; kk++) bias[kk] = ADDBIAS ?  _vel_pack_f32p(pBias+biasGroupOffset+k+2*kk+remain, pBias+biasGroupOffset+k+2*kk+remain+1) : 0UL ;

  const int64_t maxvl = nY * outWidth ;

  __vr vrseq = _vel_vseq_vl(maxvl) ;
  __vr vry  = _vel_vdivsl_vvsl(vrseq, outWidth, maxvl) ;
  __vr vrx  = _vel_vsubsl_vvvl(vrseq, _vel_vmulul_vsvl(outWidth,vry, maxvl), maxvl) ;

  __vr vri   = _vel_vaddsl_vsvl(-padHeight, _vel_vmulsl_vsvl(strideHeight, vry, maxvl), maxvl) ;
  __vr vrj   = _vel_vaddsl_vsvl(-padWidth,  _vel_vmulsl_vsvl(strideWidth,  vrx, maxvl), maxvl) ;

  for (int64_t y=0; y<outHeight; y+=nY)
  {
    const int64_t vl = outWidth * (outHeight - y < nY ? outHeight - y : nY) ;
    const int64_t op = y * outWidth ;

    __vr vrsum0  = _vel_vbrds_vsl(bias0, vl) ;
    __vr vrsum[nPacked] ;
#pragma clang loop unroll(full)
    for(int64_t kk=0; kk<nPacked; kk++) vrsum[kk] = _vel_pvbrd_vsl(bias[kk], vl) ;

    __vr vrh_r0 = _vel_vaddsl_vsvl(0*dilationHeight+y*strideHeight, vri, vl) ;
    __vr vrh_r1 = _vel_vaddsl_vsvl(1*dilationHeight+y*strideHeight, vri, vl) ;
    __vr vrh_r2 = _vel_vaddsl_vsvl(2*dilationHeight+y*strideHeight, vri, vl) ;
    __vm256 vm_r0 =  _vel_vfmklge_mvl(vrh_r0, vl) ;					// condition(0 <= h)
    __vm256 vm_r2 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(inHeight,vrh_r2, vl), vl) ;	// condition(h < inHeight)

    __vr vrw_s0 = _vel_vaddsl_vsvl(0*dilationWidth,                 vrj, vl) ;
    __vr vrw_s1 = _vel_vaddsl_vsvl(1*dilationWidth,                 vrj, vl) ;
    __vr vrw_s2 = _vel_vaddsl_vsvl(2*dilationWidth,                 vrj, vl) ;
    __vm256 vm_s0 =  _vel_vfmklge_mvl(vrw_s0, vl) ;					// condition(0 <= w)
    __vm256 vm_s2 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(inWidth,vrw_s2, vl), vl) ;	// condition(w < inWidth)

    __vm256 vm_r0s0 = _vel_andm_mmm(vm_r0,vm_s0) ;
    __vm256 vm_r0s1 = vm_r0 ;
    __vm256 vm_r0s2 = _vel_andm_mmm(vm_r0,vm_s2) ;
    __vm256 vm_r1s0 = vm_s0 ;
    __vm256 vm_r1s1 = _vel_vfmklat_ml(vl) ;
    __vm256 vm_r1s2 = vm_s2 ;
    __vm256 vm_r2s0 = _vel_andm_mmm(vm_r2,vm_s0) ;
    __vm256 vm_r2s1 = vm_r2 ;
    __vm256 vm_r2s2 = _vel_andm_mmm(vm_r2,vm_s2) ;

    for(int64_t c0=0; c0<inChannelGroup; c0+=256) {
      const int64_t clen = inChannelGroup - c0 < 256 ? inChannelGroup - c0 : 256 ;

      const float *pKerValue = pKernel + kernGroupOffset + (k * inChannelGroup + c0) * kernHeight * kernWidth ;

      for(int64_t t=0; t<9*clen; t+=256) {
	const int64_t tvl = 9*clen-t < 256 ? 9*clen-t : 256 ;

	__vr vr[NUMKERNEL] ;
#pragma clang loop unroll(full)
	for(int64_t kk=0; kk<NUMKERNEL; kk++) {
	  vr[kk] = _vel_vldu_vssl(4, pKerValue+kk*9*inChannelGroup+t, tvl) ;
	}

#pragma clang loop unroll(full)
	for(int64_t kk=0; kk<nPacked; kk++) {
	  __vr vrp  = _vel_vshf_vvvsl(vr[2*kk+remain],vr[2*kk+remain+1],VE_VSHUFFLE_YUZU, tvl) ;
	}
	if( remain ) {
	  _vel_vstu_vssl(vr[0], 4, filter+(NUMKERNEL-1)*9*clen+t, tvl) ;
	}
      }

      int64_t c1 = 0 ;
      if( (clen & 0x01) == 1 ) {
	const int64_t c = c0 + c1 ;

	const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;
	__vr vrpin_c0r0s0 = _vel_vsfa_vvssl(_vel_vaddul_vvvl(vrw_s0, _vel_vmulul_vsvl(inWidth,vrh_r0, vl), vl),
				     2,
				     (uint64_t)pInChannel, vl) ;
	__vr vrin_c0r0s0 = _vel_vgtu_vvssml(vrpin_c0r0s0, 0, 0, vm_r0s0, vl) ;

	__vr vrpin_c0r0s1 = _vel_vaddul_vsvl(4*1*dilationWidth, vrpin_c0r0s0, vl) ;
	__vr vrin_c0r0s1 = _vel_vgt_vvssml(vrpin_c0r0s1, 0, 0, vm_r0s1, vl) ;

	__vr vrpin_c0r1s0 = _vel_vaddul_vsvl(4*1*dilationHeight*inWidth, vrpin_c0r0s0, vl) ;
	__vr vrin_c0r1s0 = _vel_vgtu_vvssml(vrpin_c0r1s0, 0, 0, vm_r1s0, vl) ;

	__vr vrpin_c0r1s1 = _vel_vaddul_vsvl(4*1*dilationWidth, vrpin_c0r1s0, vl) ;
	__vr vrin_c0r1s1 = _vel_vgt_vvssml(vrpin_c0r1s1, 0, 0, vm_r1s1, vl) ;

	__vr vrpin_c0r2s0 = _vel_vaddul_vsvl(4*2*dilationHeight*inWidth, vrpin_c0r0s0, vl) ;
	__vr vrin_c0r2s0 = _vel_vgtu_vvssml(vrpin_c0r2s0, 0, 0, vm_r2s0, vl) ;

	__vr vrpin_c0r2s1 = _vel_vaddul_vsvl(4*1*dilationWidth, vrpin_c0r2s0, vl) ;
	__vr vrin_c0r2s1 = _vel_vgt_vvssml(vrpin_c0r2s1, 0, 0, vm_r2s1, vl) ;


#define VFADD(VRIN, VE_VSHUFFLE_MODE, VM, C, R, S)						\
	{											\
	  __vr vrinP = _vel_vshf_vvvsl(VRIN, VRIN, VE_VSHUFFLE_MODE, vl) ;			\
	  vrinP = _vel_vmrg_vsvml(0UL, vrinP, VM, vl) ;						\
	  if( remain ) {									\
	    const float    kerValue0  = filter[(NUMKERNEL-1)*9*clen+9*(C)+(3*(R)+(S))] ;	\
	    vrsum0 = _vel_vfmads_vvsvl(vrsum0, kerValue0, vrinP, vl) ;				\
	  }											\
	  _Pragma("clang loop unroll(full)")							\
	  for(int64_t kk=0; kk<nPacked; kk++) {							\
	    const uint64_t kerValue = filter_u64[kk*9*clen+9*(C)+(3*(R)+(S))] ;			\
	    vrsum[kk] = _vel_pvfmad_vvsvl(vrsum[kk], kerValue, vrinP, vl) ;			\
	  }											\
	}

	VFADD(vrin_c0r0s0, VE_VSHUFFLE_YUZU, vm_r0s0, c1, 0, 0) ;
	VFADD(vrin_c0r0s1, VE_VSHUFFLE_YLZL, vm_r0s1, c1, 0, 1) ;
	VFADD(vrin_c0r0s1, VE_VSHUFFLE_YUZU, vm_r0s2, c1, 0, 2) ;

	VFADD(vrin_c0r1s0, VE_VSHUFFLE_YUZU, vm_r1s0, c1, 1, 0) ;
	VFADD(vrin_c0r1s1, VE_VSHUFFLE_YLZL, vm_r1s1, c1, 1, 1) ;
	VFADD(vrin_c0r1s1, VE_VSHUFFLE_YUZU, vm_r1s2, c1, 1, 2) ;

	VFADD(vrin_c0r2s0, VE_VSHUFFLE_YUZU, vm_r2s0, c1, 2, 0) ;
	VFADD(vrin_c0r2s1, VE_VSHUFFLE_YLZL, vm_r2s1, c1, 2, 1) ;
	VFADD(vrin_c0r2s1, VE_VSHUFFLE_YUZU, vm_r2s2, c1, 2, 2) ;

	c1++ ;
      }
      if( ((clen>>1) & 0x01) == 1 ) {
	const int64_t c = c0 + c1 ;

	const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;
	__vr vrpin_c0r0s0 = _vel_vsfa_vvssl(_vel_vaddul_vvvl(vrw_s0, _vel_vmulul_vsvl(inWidth,vrh_r0, vl), vl),
				     2,
				     (uint64_t)pInChannel, vl) ;
	__vr vrin_c0r0s0 = _vel_vgtu_vvssml(vrpin_c0r0s0, 0, 0, vm_r0s0, vl) ;

	__vr vrpin_c0r0s1 = _vel_vaddul_vsvl(4*1*dilationWidth, vrpin_c0r0s0, vl) ;
	__vr vrin_c0r0s1 = _vel_vgt_vvssml(vrpin_c0r0s1, 0, 0, vm_r0s1, vl) ;

	__vr vrpin_c0r1s0 = _vel_vaddul_vsvl(4*1*dilationHeight*inWidth, vrpin_c0r0s0, vl) ;
	__vr vrin_c0r1s0 = _vel_vgtu_vvssml(vrpin_c0r1s0, 0, 0, vm_r1s0, vl) ;

	__vr vrpin_c0r1s1 = _vel_vaddul_vsvl(4*1*dilationWidth, vrpin_c0r1s0, vl) ;
	__vr vrin_c0r1s1 = _vel_vgt_vvssml(vrpin_c0r1s1, 0, 0, vm_r1s1, vl) ;

	__vr vrpin_c0r2s0 = _vel_vaddul_vsvl(4*2*dilationHeight*inWidth, vrpin_c0r0s0, vl) ;
	__vr vrin_c0r2s0 = _vel_vgtu_vvssml(vrpin_c0r2s0, 0, 0, vm_r2s0, vl) ;

	__vr vrpin_c0r2s1 = _vel_vaddul_vsvl(4*1*dilationWidth, vrpin_c0r2s0, vl) ;
	__vr vrin_c0r2s1 = _vel_vgt_vvssml(vrpin_c0r2s1, 0, 0, vm_r2s1, vl) ;


	__vr vrpin_c1r0s0 = _vel_vaddul_vsvl(4*1*inHeight*inWidth, vrpin_c0r0s0, vl) ;
	__vr vrin_c1r0s0 = _vel_vgtu_vvssml(vrpin_c1r0s0, 0, 0, vm_r0s0, vl) ;

	__vr vrpin_c1r0s1 = _vel_vaddul_vsvl(4*1*dilationWidth, vrpin_c1r0s0, vl) ;
	__vr vrin_c1r0s1 = _vel_vgt_vvssml(vrpin_c1r0s1, 0, 0, vm_r0s1, vl) ;

	__vr vrpin_c1r1s0 = _vel_vaddul_vsvl(4*1*dilationHeight*inWidth, vrpin_c1r0s0, vl) ;
	__vr vrin_c1r1s0 = _vel_vgtu_vvssml(vrpin_c1r1s0, 0, 0, vm_r1s0, vl) ;

	__vr vrpin_c1r1s1 = _vel_vaddul_vsvl(4*1*dilationWidth, vrpin_c1r1s0, vl) ;
	__vr vrin_c1r1s1 = _vel_vgt_vvssml(vrpin_c1r1s1, 0, 0, vm_r1s1, vl) ;

	__vr vrpin_c1r2s0 = _vel_vaddul_vsvl(4*2*dilationHeight*inWidth, vrpin_c1r0s0, vl) ;
	__vr vrin_c1r2s0 = _vel_vgtu_vvssml(vrpin_c1r2s0, 0, 0, vm_r2s0, vl) ;

	__vr vrpin_c1r2s1 = _vel_vaddul_vsvl(4*1*dilationWidth, vrpin_c1r2s0, vl) ;
	__vr vrin_c1r2s1 = _vel_vgt_vvssml(vrpin_c1r2s1, 0, 0, vm_r2s1, vl) ;

	VFADD(vrin_c0r0s0, VE_VSHUFFLE_YUZU, vm_r0s0, c1, 0, 0) ;
	VFADD(vrin_c0r0s1, VE_VSHUFFLE_YLZL, vm_r0s1, c1, 0, 1) ;
	VFADD(vrin_c0r0s1, VE_VSHUFFLE_YUZU, vm_r0s2, c1, 0, 2) ;

	VFADD(vrin_c0r1s0, VE_VSHUFFLE_YUZU, vm_r1s0, c1, 1, 0) ;
	VFADD(vrin_c0r1s1, VE_VSHUFFLE_YLZL, vm_r1s1, c1, 1, 1) ;
	VFADD(vrin_c0r1s1, VE_VSHUFFLE_YUZU, vm_r1s2, c1, 1, 2) ;

	VFADD(vrin_c0r2s0, VE_VSHUFFLE_YUZU, vm_r2s0, c1, 2, 0) ;
	VFADD(vrin_c0r2s1, VE_VSHUFFLE_YLZL, vm_r2s1, c1, 2, 1) ;
	VFADD(vrin_c0r2s1, VE_VSHUFFLE_YUZU, vm_r2s2, c1, 2, 2) ;

	VFADD(vrin_c1r0s0, VE_VSHUFFLE_YUZU, vm_r0s0, c1+1, 0, 0) ;
	VFADD(vrin_c1r0s1, VE_VSHUFFLE_YLZL, vm_r0s1, c1+1, 0, 1) ;
	VFADD(vrin_c1r0s1, VE_VSHUFFLE_YUZU, vm_r0s2, c1+1, 0, 2) ;

	VFADD(vrin_c1r1s0, VE_VSHUFFLE_YUZU, vm_r1s0, c1+1, 1, 0) ;
	VFADD(vrin_c1r1s1, VE_VSHUFFLE_YLZL, vm_r1s1, c1+1, 1, 1) ;
	VFADD(vrin_c1r1s1, VE_VSHUFFLE_YUZU, vm_r1s2, c1+1, 1, 2) ;

	VFADD(vrin_c1r2s0, VE_VSHUFFLE_YUZU, vm_r2s0, c1+1, 2, 0) ;
	VFADD(vrin_c1r2s1, VE_VSHUFFLE_YLZL, vm_r2s1, c1+1, 2, 1) ;
	VFADD(vrin_c1r2s1, VE_VSHUFFLE_YUZU, vm_r2s2, c1+1, 2, 2) ;

	c1+=2 ;
      }
      for( ; c1 < clen ; c1+=4 ) {
	const int64_t c = c0 + c1 ;

	const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;
	__vr vrpin_c0r0s0 = _vel_vsfa_vvssl(_vel_vaddul_vvvl(vrw_s0, _vel_vmulul_vsvl(inWidth,vrh_r0, vl), vl),
				     2,
				     (uint64_t)pInChannel, vl) ;
	__vr vrin_c0r0s0 = _vel_vgtu_vvssml(vrpin_c0r0s0, 0, 0, vm_r0s0, vl) ;

	__vr vrpin_c0r0s1 = _vel_vaddul_vsvl(4*1*dilationWidth, vrpin_c0r0s0, vl) ;
	__vr vrin_c0r0s1 = _vel_vgt_vvssml(vrpin_c0r0s1, 0, 0, vm_r0s1, vl) ;

	__vr vrpin_c0r1s0 = _vel_vaddul_vsvl(4*1*dilationHeight*inWidth, vrpin_c0r0s0, vl) ;
	__vr vrin_c0r1s0 = _vel_vgtu_vvssml(vrpin_c0r1s0, 0, 0, vm_r1s0, vl) ;

	__vr vrpin_c0r1s1 = _vel_vaddul_vsvl(4*1*dilationWidth, vrpin_c0r1s0, vl) ;
	__vr vrin_c0r1s1 = _vel_vgt_vvssml(vrpin_c0r1s1, 0, 0, vm_r1s1, vl) ;

	__vr vrpin_c0r2s0 = _vel_vaddul_vsvl(4*2*dilationHeight*inWidth, vrpin_c0r0s0, vl) ;
	__vr vrin_c0r2s0 = _vel_vgtu_vvssml(vrpin_c0r2s0, 0, 0, vm_r2s0, vl) ;

	__vr vrpin_c0r2s1 = _vel_vaddul_vsvl(4*1*dilationWidth, vrpin_c0r2s0, vl) ;
	__vr vrin_c0r2s1 = _vel_vgt_vvssml(vrpin_c0r2s1, 0, 0, vm_r2s1, vl) ;


	__vr vrpin_c1r0s0 = _vel_vaddul_vsvl(4*1*inHeight*inWidth, vrpin_c0r0s0, vl) ;
	__vr vrin_c1r0s0 = _vel_vgtu_vvssml(vrpin_c1r0s0, 0, 0, vm_r0s0, vl) ;

	__vr vrpin_c1r0s1 = _vel_vaddul_vsvl(4*1*dilationWidth, vrpin_c1r0s0, vl) ;
	__vr vrin_c1r0s1 = _vel_vgt_vvssml(vrpin_c1r0s1, 0, 0, vm_r0s1, vl) ;

	__vr vrpin_c1r1s0 = _vel_vaddul_vsvl(4*1*dilationHeight*inWidth, vrpin_c1r0s0, vl) ;
	__vr vrin_c1r1s0 = _vel_vgtu_vvssml(vrpin_c1r1s0, 0, 0, vm_r1s0, vl) ;

	__vr vrpin_c1r1s1 = _vel_vaddul_vsvl(4*1*dilationWidth, vrpin_c1r1s0, vl) ;
	__vr vrin_c1r1s1 = _vel_vgt_vvssml(vrpin_c1r1s1, 0, 0, vm_r1s1, vl) ;

	__vr vrpin_c1r2s0 = _vel_vaddul_vsvl(4*2*dilationHeight*inWidth, vrpin_c1r0s0, vl) ;
	__vr vrin_c1r2s0 = _vel_vgtu_vvssml(vrpin_c1r2s0, 0, 0, vm_r2s0, vl) ;

	__vr vrpin_c1r2s1 = _vel_vaddul_vsvl(4*1*dilationWidth, vrpin_c1r2s0, vl) ;
	__vr vrin_c1r2s1 = _vel_vgt_vvssml(vrpin_c1r2s1, 0, 0, vm_r2s1, vl) ;


	__vr vrpin_c2r0s0 = _vel_vaddul_vsvl(4*1*inHeight*inWidth, vrpin_c1r0s0, vl) ;
	__vr vrin_c2r0s0 = _vel_vgtu_vvssml(vrpin_c2r0s0, 0, 0, vm_r0s0, vl) ;

	__vr vrpin_c2r0s1 = _vel_vaddul_vsvl(4*1*dilationWidth, vrpin_c2r0s0, vl) ;
	__vr vrin_c2r0s1 = _vel_vgt_vvssml(vrpin_c2r0s1, 0, 0, vm_r0s1, vl) ;

	__vr vrpin_c2r1s0 = _vel_vaddul_vsvl(4*1*dilationHeight*inWidth, vrpin_c2r0s0, vl) ;
	__vr vrin_c2r1s0 = _vel_vgtu_vvssml(vrpin_c2r1s0, 0, 0, vm_r1s0, vl) ;

	__vr vrpin_c2r1s1 = _vel_vaddul_vsvl(4*1*dilationWidth, vrpin_c2r1s0, vl) ;
	__vr vrin_c2r1s1 = _vel_vgt_vvssml(vrpin_c2r1s1, 0, 0, vm_r1s1, vl) ;

	__vr vrpin_c2r2s0 = _vel_vaddul_vsvl(4*2*dilationHeight*inWidth, vrpin_c2r0s0, vl) ;
	__vr vrin_c2r2s0 = _vel_vgtu_vvssml(vrpin_c2r2s0, 0, 0, vm_r2s0, vl) ;

	__vr vrpin_c2r2s1 = _vel_vaddul_vsvl(4*1*dilationWidth, vrpin_c2r2s0, vl) ;
	__vr vrin_c2r2s1 = _vel_vgt_vvssml(vrpin_c2r2s1, 0, 0, vm_r2s1, vl) ;


	__vr vrpin_c3r0s0 = _vel_vaddul_vsvl(4*1*inHeight*inWidth, vrpin_c2r0s0, vl) ;
	__vr vrin_c3r0s0 = _vel_vgtu_vvssml(vrpin_c3r0s0, 0, 0, vm_r0s0, vl) ;

	__vr vrpin_c3r0s1 = _vel_vaddul_vsvl(4*1*dilationWidth, vrpin_c3r0s0, vl) ;
	__vr vrin_c3r0s1 = _vel_vgt_vvssml(vrpin_c3r0s1, 0, 0, vm_r0s1, vl) ;

	__vr vrpin_c3r1s0 = _vel_vaddul_vsvl(4*1*dilationHeight*inWidth, vrpin_c3r0s0, vl) ;
	__vr vrin_c3r1s0 = _vel_vgtu_vvssml(vrpin_c3r1s0, 0, 0, vm_r1s0, vl) ;

	__vr vrpin_c3r1s1 = _vel_vaddul_vsvl(4*1*dilationWidth, vrpin_c3r1s0, vl) ;
	__vr vrin_c3r1s1 = _vel_vgt_vvssml(vrpin_c3r1s1, 0, 0, vm_r1s1, vl) ;

	__vr vrpin_c3r2s0 = _vel_vaddul_vsvl(4*2*dilationHeight*inWidth, vrpin_c3r0s0, vl) ;
	__vr vrin_c3r2s0 = _vel_vgtu_vvssml(vrpin_c3r2s0, 0, 0, vm_r2s0, vl) ;

	__vr vrpin_c3r2s1 = _vel_vaddul_vsvl(4*1*dilationWidth, vrpin_c3r2s0, vl) ;
	__vr vrin_c3r2s1 = _vel_vgt_vvssml(vrpin_c3r2s1, 0, 0, vm_r2s1, vl) ;

	VFADD(vrin_c0r0s0, VE_VSHUFFLE_YUZU, vm_r0s0, c1, 0, 0) ;
	VFADD(vrin_c0r0s1, VE_VSHUFFLE_YLZL, vm_r0s1, c1, 0, 1) ;
	VFADD(vrin_c0r0s1, VE_VSHUFFLE_YUZU, vm_r0s2, c1, 0, 2) ;

	VFADD(vrin_c0r1s0, VE_VSHUFFLE_YUZU, vm_r1s0, c1, 1, 0) ;
	VFADD(vrin_c0r1s1, VE_VSHUFFLE_YLZL, vm_r1s1, c1, 1, 1) ;
	VFADD(vrin_c0r1s1, VE_VSHUFFLE_YUZU, vm_r1s2, c1, 1, 2) ;

	VFADD(vrin_c0r2s0, VE_VSHUFFLE_YUZU, vm_r2s0, c1, 2, 0) ;
	VFADD(vrin_c0r2s1, VE_VSHUFFLE_YLZL, vm_r2s1, c1, 2, 1) ;
	VFADD(vrin_c0r2s1, VE_VSHUFFLE_YUZU, vm_r2s2, c1, 2, 2) ;

	VFADD(vrin_c1r0s0, VE_VSHUFFLE_YUZU, vm_r0s0, c1+1, 0, 0) ;
	VFADD(vrin_c1r0s1, VE_VSHUFFLE_YLZL, vm_r0s1, c1+1, 0, 1) ;
	VFADD(vrin_c1r0s1, VE_VSHUFFLE_YUZU, vm_r0s2, c1+1, 0, 2) ;

	VFADD(vrin_c1r1s0, VE_VSHUFFLE_YUZU, vm_r1s0, c1+1, 1, 0) ;
	VFADD(vrin_c1r1s1, VE_VSHUFFLE_YLZL, vm_r1s1, c1+1, 1, 1) ;
	VFADD(vrin_c1r1s1, VE_VSHUFFLE_YUZU, vm_r1s2, c1+1, 1, 2) ;

	VFADD(vrin_c1r2s0, VE_VSHUFFLE_YUZU, vm_r2s0, c1+1, 2, 0) ;
	VFADD(vrin_c1r2s1, VE_VSHUFFLE_YLZL, vm_r2s1, c1+1, 2, 1) ;
	VFADD(vrin_c1r2s1, VE_VSHUFFLE_YUZU, vm_r2s2, c1+1, 2, 2) ;

	VFADD(vrin_c2r0s0, VE_VSHUFFLE_YUZU, vm_r0s0, c1+2, 0, 0) ;
	VFADD(vrin_c2r0s1, VE_VSHUFFLE_YLZL, vm_r0s1, c1+2, 0, 1) ;
	VFADD(vrin_c2r0s1, VE_VSHUFFLE_YUZU, vm_r0s2, c1+2, 0, 2) ;

	VFADD(vrin_c2r1s0, VE_VSHUFFLE_YUZU, vm_r1s0, c1+2, 1, 0) ;
	VFADD(vrin_c2r1s1, VE_VSHUFFLE_YLZL, vm_r1s1, c1+2, 1, 1) ;
	VFADD(vrin_c2r1s1, VE_VSHUFFLE_YUZU, vm_r1s2, c1+2, 1, 2) ;

	VFADD(vrin_c2r2s0, VE_VSHUFFLE_YUZU, vm_r2s0, c1+2, 2, 0) ;
	VFADD(vrin_c2r2s1, VE_VSHUFFLE_YLZL, vm_r2s1, c1+2, 2, 1) ;
	VFADD(vrin_c2r2s1, VE_VSHUFFLE_YUZU, vm_r2s2, c1+2, 2, 2) ;

	VFADD(vrin_c3r0s0, VE_VSHUFFLE_YUZU, vm_r0s0, c1+3, 0, 0) ;
	VFADD(vrin_c3r0s1, VE_VSHUFFLE_YLZL, vm_r0s1, c1+3, 0, 1) ;
	VFADD(vrin_c3r0s1, VE_VSHUFFLE_YUZU, vm_r0s2, c1+3, 0, 2) ;

	VFADD(vrin_c3r1s0, VE_VSHUFFLE_YUZU, vm_r1s0, c1+3, 1, 0) ;
	VFADD(vrin_c3r1s1, VE_VSHUFFLE_YLZL, vm_r1s1, c1+3, 1, 1) ;
	VFADD(vrin_c3r1s1, VE_VSHUFFLE_YUZU, vm_r1s2, c1+3, 1, 2) ;

	VFADD(vrin_c3r2s0, VE_VSHUFFLE_YUZU, vm_r2s0, c1+3, 2, 0) ;
	VFADD(vrin_c3r2s1, VE_VSHUFFLE_YLZL, vm_r2s1, c1+3, 2, 1) ;
	VFADD(vrin_c3r2s1, VE_VSHUFFLE_YUZU, vm_r2s2, c1+3, 2, 2) ;

#undef VFADD
      }
    }

    if( remain ) {
      _vel_vstu_vssl(vrsum0,  4, pOut+outIndex + 0 * outHeight*outWidth, vl) ;
    }
    for(int64_t kk=0; kk<nPacked; kk++) {
      _vel_vstu_vssl(vrsum[kk], 4, pOut+outIndex + (2*kk+remain)   * outHeight*outWidth, vl) ;
      _vel_vstl_vssl(vrsum[kk], 4, pOut+outIndex + (2*kk+remain+1) * outHeight*outWidth, vl) ;
    }

    outIndex += vl ;
  } // outPixels
}


template<filterLayout_t FLAYOUT, int NUMKERNEL, bool ADDBIAS, bool IALIGNED>
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
  const int64_t strideHeight,
  const int64_t strideWidth,
  const int64_t padHeight,
  const int64_t padWidth,
  const int64_t dilationHeight,
  const int64_t dilationWidth,
  const int64_t inGroupOffset,
  const int64_t outGroupOffset,
  const int64_t biasGroupOffset,
  const int64_t kernGroupOffset,
  const int64_t nY,
  const int64_t n,
  const int64_t k
)
{
  if(IALIGNED) {
    if( FLAYOUT == VEDNN_FILTER_LAYOUT_NCHW && inChannelGroup >= 64 )
    {
      func_ialigned_filternchw_avoid_l1m<NUMKERNEL, ADDBIAS>(pIn, pKernel, pBias, pOut,
	 inChannel, inWidth, inHeight,
	 outChannel, outWidth, outHeight,
	 kernWidth, kernHeight,
	 inChannelGroup, outChannelGroup,
	 strideHeight, strideWidth,
	 padHeight, padWidth,
	 dilationHeight, dilationWidth,
	 inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	 nY, n, k ) ;
    }
    else {
      func_ialigned<FLAYOUT, NUMKERNEL, ADDBIAS>(pIn, pKernel, pBias, pOut,
	 inChannel, inWidth, inHeight,
	 outChannel, outWidth, outHeight,
	 kernWidth, kernHeight,
	 inChannelGroup, outChannelGroup,
	 strideHeight, strideWidth,
	 padHeight, padWidth,
	 dilationHeight, dilationWidth,
	 inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	 nY, n, k ) ;
    }
  }
  else {
    if( FLAYOUT == VEDNN_FILTER_LAYOUT_NCHW && inChannelGroup >= 64 )
    {
      func_inoaligned_filternchw_avoid_l1m<NUMKERNEL, ADDBIAS>(pIn, pKernel, pBias, pOut,
	 inChannel, inWidth, inHeight,
	 outChannel, outWidth, outHeight,
	 kernWidth, kernHeight,
	 inChannelGroup, outChannelGroup,
	 strideHeight, strideWidth,
	 padHeight, padWidth,
	 dilationHeight, dilationWidth,
	 inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	 nY, n, k ) ;
    }
    else {
      func_inoaligned<FLAYOUT, NUMKERNEL, ADDBIAS>(pIn, pKernel, pBias, pOut,
	 inChannel, inWidth, inHeight,
	 outChannel, outWidth, outHeight,
	 kernWidth, kernHeight,
	 inChannelGroup, outChannelGroup,
	 strideHeight, strideWidth,
	 padHeight, padWidth,
	 dilationHeight, dilationWidth,
	 inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	 nY, n, k ) ;
    }
  }
}

template<filterLayout_t FLAYOUT, bool ADDBIAS, bool IALIGNED>
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
    const int64_t inChannelGroup,
    const int64_t outChannelGroup
)
{
  const int64_t kernHeight = 3 ;
  const int64_t kernWidth  = 3 ;

  const int64_t strideHeight = 2 ;
  const int64_t strideWidth  = 2 ;
  const int64_t padHeight = 1 ;
  const int64_t padWidth  = 1 ;
  const int64_t dilationHeight = 1;
  const int64_t dilationWidth  = 1;

  const int64_t nY = VLEN / outWidth ;

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
	  func<FLAYOUT, 1, ADDBIAS, IALIGNED>(pIn, pKernel, pBias, pOut,
	     inChannel, inWidth, inHeight,
	     outChannel, outWidth, outHeight,
	     kernWidth, kernHeight,
	     inChannelGroup, outChannelGroup,
	     strideHeight, strideWidth,
	     padHeight, padWidth,
	     dilationHeight, dilationWidth,
	     inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	     nY, n, k ) ;
	  k+=1 ;
	  break ;
	case 2 :
	  func<FLAYOUT, 2, ADDBIAS, IALIGNED>(pIn, pKernel, pBias, pOut,
	     inChannel, inWidth, inHeight,
	     outChannel, outWidth, outHeight,
	     kernWidth, kernHeight,
	     inChannelGroup, outChannelGroup,
	     strideHeight, strideWidth,
	     padHeight, padWidth,
	     dilationHeight, dilationWidth,
	     inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	     nY, n, k ) ;
	  k+=2 ;
	  break ;
	case 3 :
	  func<FLAYOUT, 3, ADDBIAS, IALIGNED>(pIn, pKernel, pBias, pOut,
	     inChannel, inWidth, inHeight,
	     outChannel, outWidth, outHeight,
	     kernWidth, kernHeight,
	     inChannelGroup, outChannelGroup,
	     strideHeight, strideWidth,
	     padHeight, padWidth,
	     dilationHeight, dilationWidth,
	     inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	     nY, n, k ) ;
	  k+=3 ;
	  break ;
	case 4 :
	  func<FLAYOUT, 4, ADDBIAS, IALIGNED>(pIn, pKernel, pBias, pOut,
	     inChannel, inWidth, inHeight,
	     outChannel, outWidth, outHeight,
	     kernWidth, kernHeight,
	     inChannelGroup, outChannelGroup,
	     strideHeight, strideWidth,
	     padHeight, padWidth,
	     dilationHeight, dilationWidth,
	     inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	     nY, n, k ) ;
	  k+=4 ;
	  break ;
	case 5 :
	  func<FLAYOUT, 5, ADDBIAS, IALIGNED>(pIn, pKernel, pBias, pOut,
	     inChannel, inWidth, inHeight,
	     outChannel, outWidth, outHeight,
	     kernWidth, kernHeight,
	     inChannelGroup, outChannelGroup,
	     strideHeight, strideWidth,
	     padHeight, padWidth,
	     dilationHeight, dilationWidth,
	     inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	     nY, n, k ) ;
	  k+=5 ;
	  break ;
	case 6 :
	  func<FLAYOUT, 6, ADDBIAS, IALIGNED>(pIn, pKernel, pBias, pOut,
	     inChannel, inWidth, inHeight,
	     outChannel, outWidth, outHeight,
	     kernWidth, kernHeight,
	     inChannelGroup, outChannelGroup,
	     strideHeight, strideWidth,
	     padHeight, padWidth,
	     dilationHeight, dilationWidth,
	     inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	     nY, n, k ) ;
	  k+=6 ;
	  break ;
	case 7 :
	  func<FLAYOUT, 7, ADDBIAS, IALIGNED>(pIn, pKernel, pBias, pOut,
	     inChannel, inWidth, inHeight,
	     outChannel, outWidth, outHeight,
	     kernWidth, kernHeight,
	     inChannelGroup, outChannelGroup,
	     strideHeight, strideWidth,
	     padHeight, padWidth,
	     dilationHeight, dilationWidth,
	     inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	     nY, n, k ) ;
	  k+=7 ;
	  break ;
	case 8 :
	  func<FLAYOUT, 8, ADDBIAS, IALIGNED>(pIn, pKernel, pBias, pOut,
	     inChannel, inWidth, inHeight,
	     outChannel, outWidth, outHeight,
	     kernWidth, kernHeight,
	     inChannelGroup, outChannelGroup,
	     strideHeight, strideWidth,
	     padHeight, padWidth,
	     dilationHeight, dilationWidth,
	     inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	     nY, n, k ) ;
	  k+=8 ;
	  break ;
	case 9 :
	  func<FLAYOUT, 9, ADDBIAS, IALIGNED>(pIn, pKernel, pBias, pOut,
	     inChannel, inWidth, inHeight,
	     outChannel, outWidth, outHeight,
	     kernWidth, kernHeight,
	     inChannelGroup, outChannelGroup,
	     strideHeight, strideWidth,
	     padHeight, padWidth,
	     dilationHeight, dilationWidth,
	     inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	     nY, n, k ) ;
	  k+=9 ;
	  break ;
	case 10 :
	  func<FLAYOUT, 10, ADDBIAS, IALIGNED>(pIn, pKernel, pBias, pOut,
	     inChannel, inWidth, inHeight,
	     outChannel, outWidth, outHeight,
	     kernWidth, kernHeight,
	     inChannelGroup, outChannelGroup,
	     strideHeight, strideWidth,
	     padHeight, padWidth,
	     dilationHeight, dilationWidth,
	     inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	     nY, n, k ) ;
	  k+=10 ;
	  break ;
	case 11 :
	  func<FLAYOUT, 11, ADDBIAS, IALIGNED>(pIn, pKernel, pBias, pOut,
	     inChannel, inWidth, inHeight,
	     outChannel, outWidth, outHeight,
	     kernWidth, kernHeight,
	     inChannelGroup, outChannelGroup,
	     strideHeight, strideWidth,
	     padHeight, padWidth,
	     dilationHeight, dilationWidth,
	     inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	     nY, n, k ) ;
	  k+=11 ;
	  break ;
	case 12 :
	  func<FLAYOUT, 12, ADDBIAS, IALIGNED>(pIn, pKernel, pBias, pOut,
	     inChannel, inWidth, inHeight,
	     outChannel, outWidth, outHeight,
	     kernWidth, kernHeight,
	     inChannelGroup, outChannelGroup,
	     strideHeight, strideWidth,
	     padHeight, padWidth,
	     dilationHeight, dilationWidth,
	     inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	     nY, n, k ) ;
	  k+=12 ;
	  break ;
	case 13 :
	  func<FLAYOUT, 13, ADDBIAS, IALIGNED>(pIn, pKernel, pBias, pOut,
	     inChannel, inWidth, inHeight,
	     outChannel, outWidth, outHeight,
	     kernWidth, kernHeight,
	     inChannelGroup, outChannelGroup,
	     strideHeight, strideWidth,
	     padHeight, padWidth,
	     dilationHeight, dilationWidth,
	     inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	     nY, n, k ) ;
	  k+=13 ;
	  break ;
	case 14 :
	  func<FLAYOUT, 14, ADDBIAS, IALIGNED>(pIn, pKernel, pBias, pOut,
	     inChannel, inWidth, inHeight,
	     outChannel, outWidth, outHeight,
	     kernWidth, kernHeight,
	     inChannelGroup, outChannelGroup,
	     strideHeight, strideWidth,
	     padHeight, padWidth,
	     dilationHeight, dilationWidth,
	     inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	     nY, n, k ) ;
	  k+=14 ;
	  break ;
	case 15 :
	  func<FLAYOUT, 15, ADDBIAS, IALIGNED>(pIn, pKernel, pBias, pOut,
	     inChannel, inWidth, inHeight,
	     outChannel, outWidth, outHeight,
	     kernWidth, kernHeight,
	     inChannelGroup, outChannelGroup,
	     strideHeight, strideWidth,
	     padHeight, padWidth,
	     dilationHeight, dilationWidth,
	     inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	     nY, n, k ) ;
	  k+=15 ;
	  break ;
	default :
	  break ;
	}
	for (; k < outChannelGroup; k+=16) {
	  func<FLAYOUT, 16, ADDBIAS, IALIGNED>(pIn, pKernel, pBias, pOut,
	     inChannel, inWidth, inHeight,
	     outChannel, outWidth, outHeight,
	     kernWidth, kernHeight,
	     inChannelGroup, outChannelGroup,
	     strideHeight, strideWidth,
	     padHeight, padWidth,
	     dilationHeight, dilationWidth,
	     inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	     nY, n, k ) ;
	} // outChannel
    } // group
  } // batch
}


extern "C" vednnError_t
vednnConvolutionForward_direct_dil1_str2_pad1_ker3_owU128(
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
//  const int64_t kernWidth  = pParamKernel->width;			// 3
//  const int64_t kernHeight = pParamKernel->height;			// 3

  const int64_t filter_layout = pParamKernel->layout ;

  const int64_t group          = pParamConv->group;
//  const int64_t strideWidth    = pParamConv->strideWidth;;		// 2
//  const int64_t strideHeight   = pParamConv->strideHeight;		// 2
//  const int64_t padWidth       = pParamConv->padWidth;		// 1
//  const int64_t padHeight      = pParamConv->padHeight;		// 1
//  const int64_t dilationWidth  = pParamConv->dilationWidth;		// 1
//  const int64_t dilationHeight = pParamConv->dilationHeight;		// 1

  const int64_t inChannelGroup  = inChannel  / group;   // equal to pDataKernel->inChannel
  const int64_t outChannelGroup = outChannel / group;   // equal to pDataKernel->outChannel

  const float * pIn     = (const float *) pDataIn;
  const float * pKernel = (const float *) pDataKernel;
  const float * pBias   = (const float *) pDataBias;
  float * const pOut    = (float * const) pDataOut;

  if( (inWidth & 0x1) == 0 && (((uint64_t)pDataIn) & 0x07) == 0 ) {
    if( filter_layout == VEDNN_FILTER_LAYOUT_NCHW) {
      if( pDataBias == NULL ) {
	convloop<VEDNN_FILTER_LAYOUT_NCHW, false, true>(
	    pIn, pKernel, pBias, pOut,
	    batch, group,
	    inChannel, inWidth, inHeight,
	    outChannel, outWidth, outHeight,
	    inChannelGroup, outChannelGroup ) ;
      }
      else {
	convloop<VEDNN_FILTER_LAYOUT_NCHW, true, true>(
	    pIn, pKernel, pBias, pOut,
	    batch, group,
	    inChannel, inWidth, inHeight,
	    outChannel, outWidth, outHeight,
	    inChannelGroup, outChannelGroup ) ;
      }
    }
    else {
      if( pDataBias == NULL ) {
	convloop<VEDNN_FILTER_LAYOUT_HWCN, false, true>(
	    pIn, pKernel, pBias, pOut,
	    batch, group,
	    inChannel, inWidth, inHeight,
	    outChannel, outWidth, outHeight,
	    inChannelGroup, outChannelGroup ) ;
      }
      else {
	convloop<VEDNN_FILTER_LAYOUT_HWCN, true, true>(
	    pIn, pKernel, pBias, pOut,
	    batch, group,
	    inChannel, inWidth, inHeight,
	    outChannel, outWidth, outHeight,
	    inChannelGroup, outChannelGroup ) ;
      }
    }
  }
  else {
    if( filter_layout == VEDNN_FILTER_LAYOUT_NCHW) {
      if( pDataBias == NULL ) {
	convloop<VEDNN_FILTER_LAYOUT_NCHW, false, false>(
	    pIn, pKernel, pBias, pOut,
	    batch, group,
	    inChannel, inWidth, inHeight,
	    outChannel, outWidth, outHeight,
	    inChannelGroup, outChannelGroup ) ;
      }
      else {
	convloop<VEDNN_FILTER_LAYOUT_NCHW, true, false>(
	    pIn, pKernel, pBias, pOut,
	    batch, group,
	    inChannel, inWidth, inHeight,
	    outChannel, outWidth, outHeight,
	    inChannelGroup, outChannelGroup ) ;
      }
    }
    else {
      if( pDataBias == NULL ) {
	convloop<VEDNN_FILTER_LAYOUT_HWCN, false, false>(
	    pIn, pKernel, pBias, pOut,
	    batch, group,
	    inChannel, inWidth, inHeight,
	    outChannel, outWidth, outHeight,
	    inChannelGroup, outChannelGroup ) ;
      }
      else {
	convloop<VEDNN_FILTER_LAYOUT_HWCN, true, false>(
	    pIn, pKernel, pBias, pOut,
	    batch, group,
	    inChannel, inWidth, inHeight,
	    outChannel, outWidth, outHeight,
	    inChannelGroup, outChannelGroup ) ;
      }
    }
  }

  return VEDNN_SUCCESS;
}

