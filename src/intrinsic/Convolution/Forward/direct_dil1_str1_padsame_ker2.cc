#include <stdint.h>
#include <stdlib.h>

#include "vednn.h"
#include "vednn_util.h"

#include "velintrin.h"
#define VLEN	(256)

template<filterLayout_t FLAYOUT, int NUMKERNEL, bool ADDBIAS>
static inline void func_odd(
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
  const int64_t n,
  const int64_t k
)
{
  int64_t outIndex = outGroupOffset + (n * outChannel + k  ) * outHeight * outWidth ;

  const float   bias0  = ADDBIAS ?  pBias[biasGroupOffset+k+ 0] : 0.f ;
  const int64_t bias12 = ADDBIAS ?  _vel_pack_f32p(pBias+biasGroupOffset+k+ 1, pBias+biasGroupOffset+k+ 2) : 0UL ;
  const int64_t bias34 = ADDBIAS ?  _vel_pack_f32p(pBias+biasGroupOffset+k+ 3, pBias+biasGroupOffset+k+ 4) : 0UL ;
  const int64_t bias56 = ADDBIAS ?  _vel_pack_f32p(pBias+biasGroupOffset+k+ 5, pBias+biasGroupOffset+k+ 6) : 0UL ;
  const int64_t bias78 = ADDBIAS ?  _vel_pack_f32p(pBias+biasGroupOffset+k+ 7, pBias+biasGroupOffset+k+ 8) : 0UL ;
  const int64_t bias9A = ADDBIAS ?  _vel_pack_f32p(pBias+biasGroupOffset+k+ 9, pBias+biasGroupOffset+k+10) : 0UL ;
  const int64_t biasBC = ADDBIAS ?  _vel_pack_f32p(pBias+biasGroupOffset+k+11, pBias+biasGroupOffset+k+12) : 0UL ;
  const int64_t biasDE = ADDBIAS ?  _vel_pack_f32p(pBias+biasGroupOffset+k+13, pBias+biasGroupOffset+k+14) : 0UL ;

  for (int64_t op = 0; op < outHeight * outWidth; op+=VLEN) {
    const int64_t vl = outHeight * outWidth - op < VLEN ? outHeight * outWidth - op : VLEN ;

    __vr vrseq = _vel_vseq_vl(vl) ;			// xy
    __vr vridx = _vel_vaddsl_vsvl(op, vrseq, vl) ;	// op + xy

    __vr vrsum0  = _vel_vbrds_vsl(bias0, vl) ;
    __vr vrsum12 = _vel_pvbrd_vsl(bias12, vl) ;
    __vr vrsum34 = _vel_pvbrd_vsl(bias34, vl) ;
    __vr vrsum56 = _vel_pvbrd_vsl(bias56, vl) ;
    __vr vrsum78 = _vel_pvbrd_vsl(bias78, vl) ;
    __vr vrsum9A = _vel_pvbrd_vsl(bias9A, vl) ;
    __vr vrsumBC = _vel_pvbrd_vsl(biasBC, vl) ;
    __vr vrsumDE = _vel_pvbrd_vsl(biasDE, vl) ;

    __vr vry   = _vel_vdivsl_vvsl(vridx, outWidth, vl) ;
    __vr vrx   = _vel_vsubsl_vvvl(vridx, _vel_vmulul_vsvl(outWidth,vry, vl), vl) ;

    __vm256 vm_r0 =  _vel_vfmklgt_mvl(vry, vl) ;	// condition(y-1>=0)
    __vm256 vm_s0 =  _vel_vfmklgt_mvl(vrx, vl) ;	// condition(x-1>=0)

    __vm256 vm_r0s0 = _vel_andm_mmm(vm_r0,vm_s0) ;
    __vm256 vm_r0s1 = vm_r0 ;
    __vm256 vm_r1s0 = vm_s0 ;

    for (int64_t c = 0; c < inChannelGroup; c++) {

      const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;

      /* memory access errors mihgt be caused */
      __vr vrin_r0s0 = _vel_vldu_vssl(4, pInChannel+op-inWidth-1, vl) ;
      __vr vrin_r0s1 = _vel_vldu_vssl(4, pInChannel+op-inWidth  , vl) ;
      __vr vrin_r1s0 = _vel_vldu_vssl(4, pInChannel+op        -1, vl) ;
      __vr vrin_r1s1 = _vel_vldu_vssl(4, pInChannel+op          , vl) ;

      vrin_r0s0 = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrin_r0s0, vm_r0s0, vl) ;
      vrin_r0s1 = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrin_r0s1, vm_r0s1, vl) ;
      vrin_r1s0 = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrin_r1s0, vm_r1s0, vl) ;

      __vr vrinP_r0s0 = _vel_vshf_vvvsl(vrin_r0s0, vrin_r0s0, VE_VSHUFFLE_YUZU, vl) ;
      __vr vrinP_r0s1 = _vel_vshf_vvvsl(vrin_r0s1, vrin_r0s1, VE_VSHUFFLE_YUZU, vl) ;
      __vr vrinP_r1s0 = _vel_vshf_vvvsl(vrin_r1s0, vrin_r1s0, VE_VSHUFFLE_YUZU, vl) ;
      __vr vrinP_r1s1 = _vel_vshf_vvvsl(vrin_r1s1, vrin_r1s1, VE_VSHUFFLE_YUZU, vl) ;


#define FILTER_OFFSET(k,c,r,s) ( kernGroupOffset + filter_index<FLAYOUT>(k,c,r,s, inChannelGroup, outChannelGroup, 2, 2) )
#define VFMAD(VRIN, VRINP, R, S)									\
      {												\
	const float    kerValue0  = pKernel[FILTER_OFFSET(k+ 0,c,R,S)] ;			\
	const uint64_t kerValue12 = _vel_pack_f32p(pKernel + FILTER_OFFSET(k+ 1,c,R,S),		\
						   pKernel + FILTER_OFFSET(k+ 2,c,R,S)) ;	\
	const uint64_t kerValue34 = _vel_pack_f32p(pKernel + FILTER_OFFSET(k+ 3,c,R,S),		\
						   pKernel + FILTER_OFFSET(k+ 4,c,R,S)) ;	\
	const uint64_t kerValue56 = _vel_pack_f32p(pKernel + FILTER_OFFSET(k+ 5,c,R,S),		\
						   pKernel + FILTER_OFFSET(k+ 6,c,R,S)) ;	\
	const uint64_t kerValue78 = _vel_pack_f32p(pKernel + FILTER_OFFSET(k+ 7,c,R,S),		\
						   pKernel + FILTER_OFFSET(k+ 8,c,R,S)) ;	\
	const uint64_t kerValue9A = _vel_pack_f32p(pKernel + FILTER_OFFSET(k+ 9,c,R,S),		\
						   pKernel + FILTER_OFFSET(k+10,c,R,S)) ;	\
	const uint64_t kerValueBC = _vel_pack_f32p(pKernel + FILTER_OFFSET(k+11,c,R,S),		\
						   pKernel + FILTER_OFFSET(k+12,c,R,S)) ;	\
	const uint64_t kerValueDE = _vel_pack_f32p(pKernel + FILTER_OFFSET(k+13,c,R,S),		\
						   pKernel + FILTER_OFFSET(k+14,c,R,S)) ;	\
	vrsum0 = _vel_vfmads_vvsvl(vrsum0, kerValue0, VRIN, vl) ;				\
	if(NUMKERNEL>= 3) vrsum12 = _vel_pvfmad_vvsvl(vrsum12, kerValue12, VRINP, vl) ;	\
	if(NUMKERNEL>= 5) vrsum34 = _vel_pvfmad_vvsvl(vrsum34, kerValue34, VRINP, vl) ;	\
	if(NUMKERNEL>= 7) vrsum56 = _vel_pvfmad_vvsvl(vrsum56, kerValue56, VRINP, vl) ;	\
	if(NUMKERNEL>= 9) vrsum78 = _vel_pvfmad_vvsvl(vrsum78, kerValue78, VRINP, vl) ;	\
	if(NUMKERNEL>=11) vrsum9A = _vel_pvfmad_vvsvl(vrsum9A, kerValue9A, VRINP, vl) ;	\
	if(NUMKERNEL>=13) vrsumBC = _vel_pvfmad_vvsvl(vrsumBC, kerValueBC, VRINP, vl) ;	\
	if(NUMKERNEL>=15) vrsumDE = _vel_pvfmad_vvsvl(vrsumDE, kerValueDE, VRINP, vl) ;	\
      }

      VFMAD(vrin_r0s0, vrinP_r0s0, 0, 0) ;
      VFMAD(vrin_r0s1, vrinP_r0s1, 0, 1) ;
      VFMAD(vrin_r1s0, vrinP_r1s0, 1, 0) ;
      VFMAD(vrin_r1s1, vrinP_r1s1, 1, 1) ;
#undef VFMAD
#undef FILTER_OFFSET
    } // inChannel

    _vel_vstu_vssl(vrsum0,  4, pOut+outIndex + 0 * outHeight*outWidth, vl) ;
    if(NUMKERNEL>= 3) {
	_vel_vstu_vssl(vrsum12, 4, pOut+outIndex + 1 * outHeight*outWidth, vl) ;
	_vel_vstl_vssl(vrsum12, 4, pOut+outIndex + 2 * outHeight*outWidth, vl) ;
    }
    if(NUMKERNEL>= 5) {
	_vel_vstu_vssl(vrsum34, 4, pOut+outIndex + 3 * outHeight*outWidth, vl) ;
	_vel_vstl_vssl(vrsum34, 4, pOut+outIndex + 4 * outHeight*outWidth, vl) ;
    }
    if(NUMKERNEL>= 7) {
	_vel_vstu_vssl(vrsum56, 4, pOut+outIndex + 5 * outHeight*outWidth, vl) ;
	_vel_vstl_vssl(vrsum56, 4, pOut+outIndex + 6 * outHeight*outWidth, vl) ;
    }
    if(NUMKERNEL>= 9) {
	_vel_vstu_vssl(vrsum78, 4, pOut+outIndex + 7 * outHeight*outWidth, vl) ;
	_vel_vstl_vssl(vrsum78, 4, pOut+outIndex + 8 * outHeight*outWidth, vl) ;
    }
    if(NUMKERNEL>= 11) {
	_vel_vstu_vssl(vrsum9A, 4, pOut+outIndex + 9 * outHeight*outWidth, vl) ;
	_vel_vstl_vssl(vrsum9A, 4, pOut+outIndex +10 * outHeight*outWidth, vl) ;
    }
    if(NUMKERNEL>= 13) {
	_vel_vstu_vssl(vrsumBC, 4, pOut+outIndex +11 * outHeight*outWidth, vl) ;
	_vel_vstl_vssl(vrsumBC, 4, pOut+outIndex +12 * outHeight*outWidth, vl) ;
    }
    if(NUMKERNEL>= 15) {
	_vel_vstu_vssl(vrsumDE, 4, pOut+outIndex +13 * outHeight*outWidth, vl) ;
	_vel_vstl_vssl(vrsumDE, 4, pOut+outIndex +14 * outHeight*outWidth, vl) ;
    }

    outIndex += vl ;
  } // outPixels
}

template<filterLayout_t FLAYOUT, int NUMKERNEL, bool ADDBIAS>
static inline void func_even(
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
  const int64_t n,
  const int64_t k
)
{
  int64_t outIndex = outGroupOffset + (n * outChannel + k  ) * outHeight * outWidth ;

  const int64_t bias01 = ADDBIAS ?  _vel_pack_f32p(pBias+biasGroupOffset+k+ 0, pBias+biasGroupOffset+k+ 1) : 0UL ;
  const int64_t bias23 = ADDBIAS ?  _vel_pack_f32p(pBias+biasGroupOffset+k+ 2, pBias+biasGroupOffset+k+ 3) : 0UL ;
  const int64_t bias45 = ADDBIAS ?  _vel_pack_f32p(pBias+biasGroupOffset+k+ 4, pBias+biasGroupOffset+k+ 5) : 0UL ;
  const int64_t bias67 = ADDBIAS ?  _vel_pack_f32p(pBias+biasGroupOffset+k+ 6, pBias+biasGroupOffset+k+ 7) : 0UL ;
  const int64_t bias89 = ADDBIAS ?  _vel_pack_f32p(pBias+biasGroupOffset+k+ 8, pBias+biasGroupOffset+k+ 9) : 0UL ;
  const int64_t biasAB = ADDBIAS ?  _vel_pack_f32p(pBias+biasGroupOffset+k+10, pBias+biasGroupOffset+k+11) : 0UL ;
  const int64_t biasCD = ADDBIAS ?  _vel_pack_f32p(pBias+biasGroupOffset+k+12, pBias+biasGroupOffset+k+13) : 0UL ;
  const int64_t biasEF = ADDBIAS ?  _vel_pack_f32p(pBias+biasGroupOffset+k+14, pBias+biasGroupOffset+k+15) : 0UL ;

  for (int64_t op = 0; op < outHeight * outWidth; op+=VLEN) {
    const int64_t vl = outHeight * outWidth - op < VLEN ? outHeight * outWidth - op : VLEN ;

    __vr vrseq = _vel_vseq_vl(vl) ;			// xy
    __vr vridx = _vel_vaddsl_vsvl(op, vrseq, vl) ;	// op + xy

    __vr vrsum01 = _vel_pvbrd_vsl(bias01, vl) ;
    __vr vrsum23 = _vel_pvbrd_vsl(bias23, vl) ;
    __vr vrsum45 = _vel_pvbrd_vsl(bias45, vl) ;
    __vr vrsum67 = _vel_pvbrd_vsl(bias67, vl) ;
    __vr vrsum89 = _vel_pvbrd_vsl(bias89, vl) ;
    __vr vrsumAB = _vel_pvbrd_vsl(biasAB, vl) ;
    __vr vrsumCD = _vel_pvbrd_vsl(biasCD, vl) ;
    __vr vrsumEF = _vel_pvbrd_vsl(biasEF, vl) ;

    __vr vry   = _vel_vdivsl_vvsl(vridx, outWidth, vl) ;
    __vr vrx   = _vel_vsubsl_vvvl(vridx, _vel_vmulul_vsvl(outWidth,vry, vl), vl) ;

    __vm256 vm_r0 =  _vel_vfmklgt_mvl(vry, vl) ;	// condition(y-1>=0)
    __vm256 vm_s0 =  _vel_vfmklgt_mvl(vrx, vl) ;	// condition(x-1>=0)

    __vm256 vm_r0s0 = _vel_andm_mmm(vm_r0,vm_s0) ;
    __vm256 vm_r0s1 = vm_r0 ;
    __vm256 vm_r1s0 = vm_s0 ;

    for (int64_t c = 0; c < inChannelGroup; c++) {

      const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;

      /* memory access errors mihgt be caused */
      __vr vrin_r0s0 = _vel_vldu_vssl(4, pInChannel+op-inWidth-1, vl) ;
      __vr vrin_r0s1 = _vel_vldu_vssl(4, pInChannel+op-inWidth  , vl) ;
      __vr vrin_r1s0 = _vel_vldu_vssl(4, pInChannel+op        -1, vl) ;
      __vr vrin_r1s1 = _vel_vldu_vssl(4, pInChannel+op          , vl) ;

      vrin_r0s0 = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrin_r0s0, vm_r0s0, vl) ;
      vrin_r0s1 = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrin_r0s1, vm_r0s1, vl) ;
      vrin_r1s0 = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrin_r1s0, vm_r1s0, vl) ;

      __vr vrinP_r0s0 = _vel_vshf_vvvsl(vrin_r0s0, vrin_r0s0, VE_VSHUFFLE_YUZU, vl) ;
      __vr vrinP_r0s1 = _vel_vshf_vvvsl(vrin_r0s1, vrin_r0s1, VE_VSHUFFLE_YUZU, vl) ;
      __vr vrinP_r1s0 = _vel_vshf_vvvsl(vrin_r1s0, vrin_r1s0, VE_VSHUFFLE_YUZU, vl) ;
      __vr vrinP_r1s1 = _vel_vshf_vvvsl(vrin_r1s1, vrin_r1s1, VE_VSHUFFLE_YUZU, vl) ;


#define FILTER_OFFSET(k,c,r,s) ( kernGroupOffset + filter_index<FLAYOUT>(k,c,r,s, inChannelGroup, outChannelGroup, 2, 2) )
#define VFMAD(VRINP, R, S)									\
      {												\
	const uint64_t kerValue01 = _vel_pack_f32p(pKernel + FILTER_OFFSET(k+ 0,c,R,S),		\
						   pKernel + FILTER_OFFSET(k+ 1,c,R,S)) ;	\
	const uint64_t kerValue23 = _vel_pack_f32p(pKernel + FILTER_OFFSET(k+ 2,c,R,S),		\
						   pKernel + FILTER_OFFSET(k+ 3,c,R,S)) ;	\
	const uint64_t kerValue45 = _vel_pack_f32p(pKernel + FILTER_OFFSET(k+ 4,c,R,S),		\
						   pKernel + FILTER_OFFSET(k+ 5,c,R,S)) ;	\
	const uint64_t kerValue67 = _vel_pack_f32p(pKernel + FILTER_OFFSET(k+ 6,c,R,S),		\
						   pKernel + FILTER_OFFSET(k+ 7,c,R,S)) ;	\
	const uint64_t kerValue89 = _vel_pack_f32p(pKernel + FILTER_OFFSET(k+ 8,c,R,S),		\
						   pKernel + FILTER_OFFSET(k+ 9,c,R,S)) ;	\
	const uint64_t kerValueAB = _vel_pack_f32p(pKernel + FILTER_OFFSET(k+10,c,R,S),		\
						   pKernel + FILTER_OFFSET(k+11,c,R,S)) ;	\
	const uint64_t kerValueCD = _vel_pack_f32p(pKernel + FILTER_OFFSET(k+12,c,R,S),		\
						   pKernel + FILTER_OFFSET(k+13,c,R,S)) ;	\
	const uint64_t kerValueEF = _vel_pack_f32p(pKernel + FILTER_OFFSET(k+14,c,R,S),		\
						   pKernel + FILTER_OFFSET(k+15,c,R,S)) ;	\
	if(NUMKERNEL>= 2) vrsum01 = _vel_pvfmad_vvsvl(vrsum01, kerValue01, VRINP, vl) ;		\
	if(NUMKERNEL>= 4) vrsum23 = _vel_pvfmad_vvsvl(vrsum23, kerValue23, VRINP, vl) ;		\
	if(NUMKERNEL>= 6) vrsum45 = _vel_pvfmad_vvsvl(vrsum45, kerValue45, VRINP, vl) ;		\
	if(NUMKERNEL>= 8) vrsum67 = _vel_pvfmad_vvsvl(vrsum67, kerValue67, VRINP, vl) ;		\
	if(NUMKERNEL>=10) vrsum89 = _vel_pvfmad_vvsvl(vrsum89, kerValue89, VRINP, vl) ;		\
	if(NUMKERNEL>=12) vrsumAB = _vel_pvfmad_vvsvl(vrsumAB, kerValueAB, VRINP, vl) ;		\
	if(NUMKERNEL>=14) vrsumCD = _vel_pvfmad_vvsvl(vrsumCD, kerValueCD, VRINP, vl) ;		\
	if(NUMKERNEL>=16) vrsumEF = _vel_pvfmad_vvsvl(vrsumEF, kerValueEF, VRINP, vl) ;		\
      }

      VFMAD(vrinP_r0s0, 0, 0) ;
      VFMAD(vrinP_r0s1, 0, 1) ;
      VFMAD(vrinP_r1s0, 1, 0) ;
      VFMAD(vrinP_r1s1, 1, 1) ;
#undef VFMAD
#undef FILTER_OFFSET
    } // inChannel

    if(NUMKERNEL>= 2) {
	_vel_vstu_vssl(vrsum01, 4, pOut+outIndex + 0 * outHeight*outWidth, vl) ;
	_vel_vstl_vssl(vrsum01, 4, pOut+outIndex + 1 * outHeight*outWidth, vl) ;
    }
    if(NUMKERNEL>= 4) {
	_vel_vstu_vssl(vrsum23, 4, pOut+outIndex + 2 * outHeight*outWidth, vl) ;
	_vel_vstl_vssl(vrsum23, 4, pOut+outIndex + 3 * outHeight*outWidth, vl) ;
    }
    if(NUMKERNEL>= 6) {
	_vel_vstu_vssl(vrsum45, 4, pOut+outIndex + 4 * outHeight*outWidth, vl) ;
	_vel_vstl_vssl(vrsum45, 4, pOut+outIndex + 5 * outHeight*outWidth, vl) ;
    }
    if(NUMKERNEL>= 8) {
	_vel_vstu_vssl(vrsum67, 4, pOut+outIndex + 6 * outHeight*outWidth, vl) ;
	_vel_vstl_vssl(vrsum67, 4, pOut+outIndex + 7 * outHeight*outWidth, vl) ;
    }
    if(NUMKERNEL>= 10) {
	_vel_vstu_vssl(vrsum89, 4, pOut+outIndex + 8 * outHeight*outWidth, vl) ;
	_vel_vstl_vssl(vrsum89, 4, pOut+outIndex + 9 * outHeight*outWidth, vl) ;
    }
    if(NUMKERNEL>= 12) {
	_vel_vstu_vssl(vrsumAB, 4, pOut+outIndex +10 * outHeight*outWidth, vl) ;
	_vel_vstl_vssl(vrsumAB, 4, pOut+outIndex +11 * outHeight*outWidth, vl) ;
    }
    if(NUMKERNEL>= 14) {
	_vel_vstu_vssl(vrsumCD, 4, pOut+outIndex +12 * outHeight*outWidth, vl) ;
	_vel_vstl_vssl(vrsumCD, 4, pOut+outIndex +13 * outHeight*outWidth, vl) ;
    }
    if(NUMKERNEL>= 16) {
	_vel_vstu_vssl(vrsumEF, 4, pOut+outIndex +14 * outHeight*outWidth, vl) ;
	_vel_vstl_vssl(vrsumEF, 4, pOut+outIndex +15 * outHeight*outWidth, vl) ;
    }

    outIndex += vl ;
  } // outPixels
}

template<bool ADDBIAS>
static inline void k8_filter_nchw_c512x(
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
    const int64_t n,
    const int64_t k
)
{

  float __attribute__ ((aligned(8))) filter[4*16*256] ;
  int64_t outIndex = outGroupOffset + (n * outChannel + k  ) * outHeight * outWidth ;

  const int64_t bias01 = ADDBIAS ?  _vel_pack_f32p(pBias+biasGroupOffset+k+ 0, pBias+biasGroupOffset+k+ 1) : 0UL ;
  const int64_t bias23 = ADDBIAS ?  _vel_pack_f32p(pBias+biasGroupOffset+k+ 2, pBias+biasGroupOffset+k+ 3) : 0UL ;
  const int64_t bias45 = ADDBIAS ?  _vel_pack_f32p(pBias+biasGroupOffset+k+ 4, pBias+biasGroupOffset+k+ 5) : 0UL ;
  const int64_t bias67 = ADDBIAS ?  _vel_pack_f32p(pBias+biasGroupOffset+k+ 6, pBias+biasGroupOffset+k+ 7) : 0UL ;

  for (int64_t op = 0; op < outHeight * outWidth; op+=VLEN) {
    const int64_t vl = outHeight * outWidth - op < VLEN ? outHeight * outWidth - op : VLEN ;

    __vr vrseq = _vel_vseq_vl(vl) ;			// xy
    __vr vridx = _vel_vaddsl_vsvl(op, vrseq, vl) ;	// op + xy

    __vr vrsum01 = _vel_pvbrd_vsl(bias01, vl) ;
    __vr vrsum23 = _vel_pvbrd_vsl(bias23, vl) ;
    __vr vrsum45 = _vel_pvbrd_vsl(bias45, vl) ;
    __vr vrsum67 = _vel_pvbrd_vsl(bias67, vl) ;

    __vr vry   = _vel_vdivsl_vvsl(vridx, outWidth, vl) ;
    __vr vrx   = _vel_vsubsl_vvvl(vridx, _vel_vmulul_vsvl(outWidth,vry, vl), vl) ;

    __vm256 vm_r0 =  _vel_vfmklgt_mvl(vry, vl) ;	// condition(y-1>=0)
    __vm256 vm_s0 =  _vel_vfmklgt_mvl(vrx, vl) ;	// condition(x-1>=0)

    __vm256 vm_r0s0 = _vel_andm_mmm(vm_r0,vm_s0) ;
    __vm256 vm_r0s1 = vm_r0 ;
    __vm256 vm_r1s0 = vm_s0 ;

    for (int64_t c0 = 0; c0 < inChannelGroup; c0+=256) {
      const float *pKerValue  = pKernel + kernGroupOffset + ((k * inChannelGroup + c0) * kernHeight ) * kernWidth ;
      {
	__vr vr0_r0 = _vel_vld_vssl(4*4, pKerValue +0*inChannelGroup*4, 256) ;
	__vr vr1_r0 = _vel_vld_vssl(4*4, pKerValue +1*inChannelGroup*4, 256) ;
	__vr vr2_r0 = _vel_vld_vssl(4*4, pKerValue +2*inChannelGroup*4, 256) ;
	__vr vr3_r0 = _vel_vld_vssl(4*4, pKerValue +3*inChannelGroup*4, 256) ;
	__vr vr4_r0 = _vel_vld_vssl(4*4, pKerValue +4*inChannelGroup*4, 256) ;
	__vr vr5_r0 = _vel_vld_vssl(4*4, pKerValue +5*inChannelGroup*4, 256) ;
	__vr vr6_r0 = _vel_vld_vssl(4*4, pKerValue +6*inChannelGroup*4, 256) ;
	__vr vr7_r0 = _vel_vld_vssl(4*4, pKerValue +7*inChannelGroup*4, 256) ;

	__vr vr0_r1 = _vel_vld_vssl(4*4, pKerValue +0*inChannelGroup*4+2, 256) ;
	__vr vr1_r1 = _vel_vld_vssl(4*4, pKerValue +1*inChannelGroup*4+2, 256) ;
	__vr vr2_r1 = _vel_vld_vssl(4*4, pKerValue +2*inChannelGroup*4+2, 256) ;
	__vr vr3_r1 = _vel_vld_vssl(4*4, pKerValue +3*inChannelGroup*4+2, 256) ;
	__vr vr4_r1 = _vel_vld_vssl(4*4, pKerValue +4*inChannelGroup*4+2, 256) ;
	__vr vr5_r1 = _vel_vld_vssl(4*4, pKerValue +5*inChannelGroup*4+2, 256) ;
	__vr vr6_r1 = _vel_vld_vssl(4*4, pKerValue +6*inChannelGroup*4+2, 256) ;
	__vr vr7_r1 = _vel_vld_vssl(4*4, pKerValue +7*inChannelGroup*4+2, 256) ;

	__vr vr01_r0s0 = _vel_vshf_vvvsl(vr0_r0,vr1_r0,VE_VSHUFFLE_YLZL, 256) ;
	__vr vr23_r0s0 = _vel_vshf_vvvsl(vr2_r0,vr3_r0,VE_VSHUFFLE_YLZL, 256) ;
	__vr vr45_r0s0 = _vel_vshf_vvvsl(vr4_r0,vr5_r0,VE_VSHUFFLE_YLZL, 256) ;
	__vr vr67_r0s0 = _vel_vshf_vvvsl(vr6_r0,vr7_r0,VE_VSHUFFLE_YLZL, 256) ;
	_vel_vst_vssl(vr01_r0s0, 4*4*8, filter, 256) ;
	_vel_vst_vssl(vr23_r0s0, 4*4*8, filter+2, 256) ;
	_vel_vst_vssl(vr45_r0s0, 4*4*8, filter+4, 256) ;
	_vel_vst_vssl(vr67_r0s0, 4*4*8, filter+6, 256) ;

	__vr vr01_r0s1 = _vel_vshf_vvvsl(vr0_r0,vr1_r0,VE_VSHUFFLE_YUZU, 256) ;
	__vr vr23_r0s1 = _vel_vshf_vvvsl(vr2_r0,vr3_r0,VE_VSHUFFLE_YUZU, 256) ;
	__vr vr45_r0s1 = _vel_vshf_vvvsl(vr4_r0,vr5_r0,VE_VSHUFFLE_YUZU, 256) ;
	__vr vr67_r0s1 = _vel_vshf_vvvsl(vr6_r0,vr7_r0,VE_VSHUFFLE_YUZU, 256) ;
	_vel_vst_vssl(vr01_r0s1, 4*4*8, filter+8+0, 256) ;
	_vel_vst_vssl(vr23_r0s1, 4*4*8, filter+8+2, 256) ;
	_vel_vst_vssl(vr45_r0s1, 4*4*8, filter+8+4, 256) ;
	_vel_vst_vssl(vr67_r0s1, 4*4*8, filter+8+6, 256) ;

	__vr vr01_r1s0 = _vel_vshf_vvvsl(vr0_r1,vr1_r1,VE_VSHUFFLE_YLZL, 256) ;
	__vr vr23_r1s0 = _vel_vshf_vvvsl(vr2_r1,vr3_r1,VE_VSHUFFLE_YLZL, 256) ;
	__vr vr45_r1s0 = _vel_vshf_vvvsl(vr4_r1,vr5_r1,VE_VSHUFFLE_YLZL, 256) ;
	__vr vr67_r1s0 = _vel_vshf_vvvsl(vr6_r1,vr7_r1,VE_VSHUFFLE_YLZL, 256) ;
	_vel_vst_vssl(vr01_r1s0, 4*4*8, filter+2*8+0, 256) ;
	_vel_vst_vssl(vr23_r1s0, 4*4*8, filter+2*8+2, 256) ;
	_vel_vst_vssl(vr45_r1s0, 4*4*8, filter+2*8+4, 256) ;
	_vel_vst_vssl(vr67_r1s0, 4*4*8, filter+2*8+6, 256) ;

	__vr vr01_r1s1 = _vel_vshf_vvvsl(vr0_r1,vr1_r1,VE_VSHUFFLE_YUZU, 256) ;
	__vr vr23_r1s1 = _vel_vshf_vvvsl(vr2_r1,vr3_r1,VE_VSHUFFLE_YUZU, 256) ;
	__vr vr45_r1s1 = _vel_vshf_vvvsl(vr4_r1,vr5_r1,VE_VSHUFFLE_YUZU, 256) ;
	__vr vr67_r1s1 = _vel_vshf_vvvsl(vr6_r1,vr7_r1,VE_VSHUFFLE_YUZU, 256) ;
	_vel_vst_vssl(vr01_r1s1, 4*4*8, filter+3*8+0, 256) ;
	_vel_vst_vssl(vr23_r1s1, 4*4*8, filter+3*8+2, 256) ;
	_vel_vst_vssl(vr45_r1s1, 4*4*8, filter+3*8+4, 256) ;
	_vel_vst_vssl(vr67_r1s1, 4*4*8, filter+3*8+6, 256) ;
      }
      for (int64_t c1 = 0 ; c1 < 256 ; c1++) {
	const int64_t c = c0 + c1 ;

	const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;
	const uint64_t* filter_u64 = (const uint64_t*)(filter+c1*32) ;

	/* memory access errors mihgt be caused */
	__vr vrin_r0s0 = _vel_vldu_vssl(4, pInChannel+op-inWidth-1, vl) ;
	__vr vrin_r0s1 = _vel_vldu_vssl(4, pInChannel+op-inWidth  , vl) ;
	__vr vrin_r1s0 = _vel_vldu_vssl(4, pInChannel+op        -1, vl) ;
	__vr vrin_r1s1 = _vel_vldu_vssl(4, pInChannel+op          , vl) ;

	vrin_r0s0 = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrin_r0s0, vm_r0s0, vl) ;
	vrin_r0s1 = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrin_r0s1, vm_r0s1, vl) ;
	vrin_r1s0 = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrin_r1s0, vm_r1s0, vl) ;

	__vr vrinP_r0s0 = _vel_vshf_vvvsl(vrin_r0s0, vrin_r0s0, VE_VSHUFFLE_YUZU, vl) ;
	__vr vrinP_r0s1 = _vel_vshf_vvvsl(vrin_r0s1, vrin_r0s1, VE_VSHUFFLE_YUZU, vl) ;
	__vr vrinP_r1s0 = _vel_vshf_vvvsl(vrin_r1s0, vrin_r1s0, VE_VSHUFFLE_YUZU, vl) ;
	__vr vrinP_r1s1 = _vel_vshf_vvvsl(vrin_r1s1, vrin_r1s1, VE_VSHUFFLE_YUZU, vl) ;

	vrsum01 = _vel_pvfmad_vvsvl(vrsum01, filter_u64[0], vrinP_r0s0, vl) ;
	vrsum23 = _vel_pvfmad_vvsvl(vrsum23, filter_u64[1], vrinP_r0s0, vl) ;
	vrsum45 = _vel_pvfmad_vvsvl(vrsum45, filter_u64[2], vrinP_r0s0, vl) ;
	vrsum67 = _vel_pvfmad_vvsvl(vrsum67, filter_u64[3], vrinP_r0s0, vl) ;

	vrsum01 = _vel_pvfmad_vvsvl(vrsum01, filter_u64[4], vrinP_r0s1, vl) ;
	vrsum23 = _vel_pvfmad_vvsvl(vrsum23, filter_u64[5], vrinP_r0s1, vl) ;
	vrsum45 = _vel_pvfmad_vvsvl(vrsum45, filter_u64[6], vrinP_r0s1, vl) ;
	vrsum67 = _vel_pvfmad_vvsvl(vrsum67, filter_u64[7], vrinP_r0s1, vl) ;

	vrsum01 = _vel_pvfmad_vvsvl(vrsum01, filter_u64[8], vrinP_r1s0, vl) ;
	vrsum23 = _vel_pvfmad_vvsvl(vrsum23, filter_u64[9], vrinP_r1s0, vl) ;
	vrsum45 = _vel_pvfmad_vvsvl(vrsum45, filter_u64[10], vrinP_r1s0, vl) ;
	vrsum67 = _vel_pvfmad_vvsvl(vrsum67, filter_u64[11], vrinP_r1s0, vl) ;

	vrsum01 = _vel_pvfmad_vvsvl(vrsum01, filter_u64[12], vrinP_r1s1, vl) ;
	vrsum23 = _vel_pvfmad_vvsvl(vrsum23, filter_u64[13], vrinP_r1s1, vl) ;
	vrsum45 = _vel_pvfmad_vvsvl(vrsum45, filter_u64[14], vrinP_r1s1, vl) ;
	vrsum67 = _vel_pvfmad_vvsvl(vrsum67, filter_u64[15], vrinP_r1s1, vl) ;
      }
    } // inChannel

    _vel_vstu_vssl(vrsum01, 4, pOut+outIndex + 0 * outHeight*outWidth, vl) ;
    _vel_vstl_vssl(vrsum01, 4, pOut+outIndex + 1 * outHeight*outWidth, vl) ;
    _vel_vstu_vssl(vrsum23, 4, pOut+outIndex + 2 * outHeight*outWidth, vl) ;
    _vel_vstl_vssl(vrsum23, 4, pOut+outIndex + 3 * outHeight*outWidth, vl) ;
    _vel_vstu_vssl(vrsum45, 4, pOut+outIndex + 4 * outHeight*outWidth, vl) ;
    _vel_vstl_vssl(vrsum45, 4, pOut+outIndex + 5 * outHeight*outWidth, vl) ;
    _vel_vstu_vssl(vrsum67, 4, pOut+outIndex + 6 * outHeight*outWidth, vl) ;
    _vel_vstl_vssl(vrsum67, 4, pOut+outIndex + 7 * outHeight*outWidth, vl) ;

    outIndex += vl ;
  } // outPixels
}

template<bool ADDBIAS>
static inline void k16_filter_nchw_c256x(
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
    const int64_t n,
    const int64_t k
)
{
  float __attribute__ ((aligned(8))) filter[4*16*256] ;

  int64_t outIndex = outGroupOffset + (n * outChannel + k  ) * outHeight * outWidth ;

  const int64_t bias01 = ADDBIAS ?  _vel_pack_f32p(pBias+biasGroupOffset+k+ 0, pBias+biasGroupOffset+k+ 1) : 0UL ;
  const int64_t bias23 = ADDBIAS ?  _vel_pack_f32p(pBias+biasGroupOffset+k+ 2, pBias+biasGroupOffset+k+ 3) : 0UL ;
  const int64_t bias45 = ADDBIAS ?  _vel_pack_f32p(pBias+biasGroupOffset+k+ 4, pBias+biasGroupOffset+k+ 5) : 0UL ;
  const int64_t bias67 = ADDBIAS ?  _vel_pack_f32p(pBias+biasGroupOffset+k+ 6, pBias+biasGroupOffset+k+ 7) : 0UL ;
  const int64_t bias89 = ADDBIAS ?  _vel_pack_f32p(pBias+biasGroupOffset+k+ 8, pBias+biasGroupOffset+k+ 9) : 0UL ;
  const int64_t biasAB = ADDBIAS ?  _vel_pack_f32p(pBias+biasGroupOffset+k+10, pBias+biasGroupOffset+k+11) : 0UL ;
  const int64_t biasCD = ADDBIAS ?  _vel_pack_f32p(pBias+biasGroupOffset+k+12, pBias+biasGroupOffset+k+13) : 0UL ;
  const int64_t biasEF = ADDBIAS ?  _vel_pack_f32p(pBias+biasGroupOffset+k+14, pBias+biasGroupOffset+k+15) : 0UL ;

  for (int64_t op = 0; op < outHeight * outWidth; op+=VLEN) {
    const int64_t vl = outHeight * outWidth - op < VLEN ? outHeight * outWidth - op : VLEN ;

    __vr vrseq = _vel_vseq_vl(vl) ;			// xy
    __vr vridx = _vel_vaddsl_vsvl(op, vrseq, vl) ;	// op + xy

    __vr vrsum01 = _vel_pvbrd_vsl(bias01, vl) ;
    __vr vrsum23 = _vel_pvbrd_vsl(bias23, vl) ;
    __vr vrsum45 = _vel_pvbrd_vsl(bias45, vl) ;
    __vr vrsum67 = _vel_pvbrd_vsl(bias67, vl) ;
    __vr vrsum89 = _vel_pvbrd_vsl(bias89, vl) ;
    __vr vrsumAB = _vel_pvbrd_vsl(biasAB, vl) ;
    __vr vrsumCD = _vel_pvbrd_vsl(biasCD, vl) ;
    __vr vrsumEF = _vel_pvbrd_vsl(biasEF, vl) ;

    __vr vry   = _vel_vdivsl_vvsl(vridx, outWidth, vl) ;
    __vr vrx   = _vel_vsubsl_vvvl(vridx, _vel_vmulul_vsvl(outWidth,vry, vl), vl) ;

    __vm256 vm_r0 =  _vel_vfmklgt_mvl(vry, vl) ;	// condition(y-1>=0)
    __vm256 vm_s0 =  _vel_vfmklgt_mvl(vrx, vl) ;	// condition(x-1>=0)

    __vm256 vm_r0s0 = _vel_andm_mmm(vm_r0,vm_s0) ;
    __vm256 vm_r0s1 = vm_r0 ;
    __vm256 vm_r1s0 = vm_s0 ;

    for (int64_t c0 = 0; c0 < inChannelGroup; c0+=256) {
      const float *pKerValue  = pKernel + kernGroupOffset + ((k * inChannelGroup + c0) * kernHeight ) * kernWidth ;
      {
	__vr vr0_r0 = _vel_vld_vssl(4*4, pKerValue +0*inChannelGroup*4, 256) ;
	__vr vr1_r0 = _vel_vld_vssl(4*4, pKerValue +1*inChannelGroup*4, 256) ;
	__vr vr2_r0 = _vel_vld_vssl(4*4, pKerValue +2*inChannelGroup*4, 256) ;
	__vr vr3_r0 = _vel_vld_vssl(4*4, pKerValue +3*inChannelGroup*4, 256) ;
	__vr vr4_r0 = _vel_vld_vssl(4*4, pKerValue +4*inChannelGroup*4, 256) ;
	__vr vr5_r0 = _vel_vld_vssl(4*4, pKerValue +5*inChannelGroup*4, 256) ;
	__vr vr6_r0 = _vel_vld_vssl(4*4, pKerValue +6*inChannelGroup*4, 256) ;
	__vr vr7_r0 = _vel_vld_vssl(4*4, pKerValue +7*inChannelGroup*4, 256) ;
	__vr vr8_r0 = _vel_vld_vssl(4*4, pKerValue +8*inChannelGroup*4, 256) ;
	__vr vr9_r0 = _vel_vld_vssl(4*4, pKerValue +9*inChannelGroup*4, 256) ;
	__vr vrA_r0 = _vel_vld_vssl(4*4, pKerValue +10*inChannelGroup*4, 256) ;
	__vr vrB_r0 = _vel_vld_vssl(4*4, pKerValue +11*inChannelGroup*4, 256) ;
	__vr vrC_r0 = _vel_vld_vssl(4*4, pKerValue +12*inChannelGroup*4, 256) ;
	__vr vrD_r0 = _vel_vld_vssl(4*4, pKerValue +13*inChannelGroup*4, 256) ;
	__vr vrE_r0 = _vel_vld_vssl(4*4, pKerValue +14*inChannelGroup*4, 256) ;
	__vr vrF_r0 = _vel_vld_vssl(4*4, pKerValue +15*inChannelGroup*4, 256) ;

	__vr vr0_r1 = _vel_vld_vssl(4*4, pKerValue +0*inChannelGroup*4+2, 256) ;
	__vr vr1_r1 = _vel_vld_vssl(4*4, pKerValue +1*inChannelGroup*4+2, 256) ;
	__vr vr2_r1 = _vel_vld_vssl(4*4, pKerValue +2*inChannelGroup*4+2, 256) ;
	__vr vr3_r1 = _vel_vld_vssl(4*4, pKerValue +3*inChannelGroup*4+2, 256) ;
	__vr vr4_r1 = _vel_vld_vssl(4*4, pKerValue +4*inChannelGroup*4+2, 256) ;
	__vr vr5_r1 = _vel_vld_vssl(4*4, pKerValue +5*inChannelGroup*4+2, 256) ;
	__vr vr6_r1 = _vel_vld_vssl(4*4, pKerValue +6*inChannelGroup*4+2, 256) ;
	__vr vr7_r1 = _vel_vld_vssl(4*4, pKerValue +7*inChannelGroup*4+2, 256) ;
	__vr vr8_r1 = _vel_vld_vssl(4*4, pKerValue +8*inChannelGroup*4+2, 256) ;
	__vr vr9_r1 = _vel_vld_vssl(4*4, pKerValue +9*inChannelGroup*4+2, 256) ;
	__vr vrA_r1 = _vel_vld_vssl(4*4, pKerValue +10*inChannelGroup*4+2, 256) ;
	__vr vrB_r1 = _vel_vld_vssl(4*4, pKerValue +11*inChannelGroup*4+2, 256) ;
	__vr vrC_r1 = _vel_vld_vssl(4*4, pKerValue +12*inChannelGroup*4+2, 256) ;
	__vr vrD_r1 = _vel_vld_vssl(4*4, pKerValue +13*inChannelGroup*4+2, 256) ;
	__vr vrE_r1 = _vel_vld_vssl(4*4, pKerValue +14*inChannelGroup*4+2, 256) ;
	__vr vrF_r1 = _vel_vld_vssl(4*4, pKerValue +15*inChannelGroup*4+2, 256) ;

	__vr vr01_r0s0 = _vel_vshf_vvvsl(vr0_r0,vr1_r0,VE_VSHUFFLE_YLZL, 256) ;
	__vr vr23_r0s0 = _vel_vshf_vvvsl(vr2_r0,vr3_r0,VE_VSHUFFLE_YLZL, 256) ;
	__vr vr45_r0s0 = _vel_vshf_vvvsl(vr4_r0,vr5_r0,VE_VSHUFFLE_YLZL, 256) ;
	__vr vr67_r0s0 = _vel_vshf_vvvsl(vr6_r0,vr7_r0,VE_VSHUFFLE_YLZL, 256) ;
	__vr vr89_r0s0 = _vel_vshf_vvvsl(vr8_r0,vr9_r0,VE_VSHUFFLE_YLZL, 256) ;
	__vr vrAB_r0s0 = _vel_vshf_vvvsl(vrA_r0,vrB_r0,VE_VSHUFFLE_YLZL, 256) ;
	__vr vrCD_r0s0 = _vel_vshf_vvvsl(vrC_r0,vrD_r0,VE_VSHUFFLE_YLZL, 256) ;
	__vr vrEF_r0s0 = _vel_vshf_vvvsl(vrE_r0,vrF_r0,VE_VSHUFFLE_YLZL, 256) ;
	_vel_vst_vssl(vr01_r0s0, 4*4*16, filter, 256) ;
	_vel_vst_vssl(vr23_r0s0, 4*4*16, filter+2, 256) ;
	_vel_vst_vssl(vr45_r0s0, 4*4*16, filter+4, 256) ;
	_vel_vst_vssl(vr67_r0s0, 4*4*16, filter+6, 256) ;
	_vel_vst_vssl(vr89_r0s0, 4*4*16, filter+8, 256) ;
	_vel_vst_vssl(vrAB_r0s0, 4*4*16, filter+10, 256) ;
	_vel_vst_vssl(vrCD_r0s0, 4*4*16, filter+12, 256) ;
	_vel_vst_vssl(vrEF_r0s0, 4*4*16, filter+14, 256) ;

	__vr vr01_r0s1 = _vel_vshf_vvvsl(vr0_r0,vr1_r0,VE_VSHUFFLE_YUZU, 256) ;
	__vr vr23_r0s1 = _vel_vshf_vvvsl(vr2_r0,vr3_r0,VE_VSHUFFLE_YUZU, 256) ;
	__vr vr45_r0s1 = _vel_vshf_vvvsl(vr4_r0,vr5_r0,VE_VSHUFFLE_YUZU, 256) ;
	__vr vr67_r0s1 = _vel_vshf_vvvsl(vr6_r0,vr7_r0,VE_VSHUFFLE_YUZU, 256) ;
	__vr vr89_r0s1 = _vel_vshf_vvvsl(vr8_r0,vr9_r0,VE_VSHUFFLE_YUZU, 256) ;
	__vr vrAB_r0s1 = _vel_vshf_vvvsl(vrA_r0,vrB_r0,VE_VSHUFFLE_YUZU, 256) ;
	__vr vrCD_r0s1 = _vel_vshf_vvvsl(vrC_r0,vrD_r0,VE_VSHUFFLE_YUZU, 256) ;
	__vr vrEF_r0s1 = _vel_vshf_vvvsl(vrE_r0,vrF_r0,VE_VSHUFFLE_YUZU, 256) ;
	_vel_vst_vssl(vr01_r0s1, 4*4*16, filter+16+0, 256) ;
	_vel_vst_vssl(vr23_r0s1, 4*4*16, filter+16+2, 256) ;
	_vel_vst_vssl(vr45_r0s1, 4*4*16, filter+16+4, 256) ;
	_vel_vst_vssl(vr67_r0s1, 4*4*16, filter+16+6, 256) ;
	_vel_vst_vssl(vr89_r0s1, 4*4*16, filter+16+8, 256) ;
	_vel_vst_vssl(vrAB_r0s1, 4*4*16, filter+16+10, 256) ;
	_vel_vst_vssl(vrCD_r0s1, 4*4*16, filter+16+12, 256) ;
	_vel_vst_vssl(vrEF_r0s1, 4*4*16, filter+16+14, 256) ;

	__vr vr01_r1s0 = _vel_vshf_vvvsl(vr0_r1,vr1_r1,VE_VSHUFFLE_YLZL, 256) ;
	__vr vr23_r1s0 = _vel_vshf_vvvsl(vr2_r1,vr3_r1,VE_VSHUFFLE_YLZL, 256) ;
	__vr vr45_r1s0 = _vel_vshf_vvvsl(vr4_r1,vr5_r1,VE_VSHUFFLE_YLZL, 256) ;
	__vr vr67_r1s0 = _vel_vshf_vvvsl(vr6_r1,vr7_r1,VE_VSHUFFLE_YLZL, 256) ;
	__vr vr89_r1s0 = _vel_vshf_vvvsl(vr8_r1,vr9_r1,VE_VSHUFFLE_YLZL, 256) ;
	__vr vrAB_r1s0 = _vel_vshf_vvvsl(vrA_r1,vrB_r1,VE_VSHUFFLE_YLZL, 256) ;
	__vr vrCD_r1s0 = _vel_vshf_vvvsl(vrC_r1,vrD_r1,VE_VSHUFFLE_YLZL, 256) ;
	__vr vrEF_r1s0 = _vel_vshf_vvvsl(vrE_r1,vrF_r1,VE_VSHUFFLE_YLZL, 256) ;
	_vel_vst_vssl(vr01_r1s0, 4*4*16, filter+2*16+0, 256) ;
	_vel_vst_vssl(vr23_r1s0, 4*4*16, filter+2*16+2, 256) ;
	_vel_vst_vssl(vr45_r1s0, 4*4*16, filter+2*16+4, 256) ;
	_vel_vst_vssl(vr67_r1s0, 4*4*16, filter+2*16+6, 256) ;
	_vel_vst_vssl(vr89_r1s0, 4*4*16, filter+2*16+8, 256) ;
	_vel_vst_vssl(vrAB_r1s0, 4*4*16, filter+2*16+10, 256) ;
	_vel_vst_vssl(vrCD_r1s0, 4*4*16, filter+2*16+12, 256) ;
	_vel_vst_vssl(vrEF_r1s0, 4*4*16, filter+2*16+14, 256) ;

	__vr vr01_r1s1 = _vel_vshf_vvvsl(vr0_r1,vr1_r1,VE_VSHUFFLE_YUZU, 256) ;
	__vr vr23_r1s1 = _vel_vshf_vvvsl(vr2_r1,vr3_r1,VE_VSHUFFLE_YUZU, 256) ;
	__vr vr45_r1s1 = _vel_vshf_vvvsl(vr4_r1,vr5_r1,VE_VSHUFFLE_YUZU, 256) ;
	__vr vr67_r1s1 = _vel_vshf_vvvsl(vr6_r1,vr7_r1,VE_VSHUFFLE_YUZU, 256) ;
	__vr vr89_r1s1 = _vel_vshf_vvvsl(vr8_r1,vr9_r1,VE_VSHUFFLE_YUZU, 256) ;
	__vr vrAB_r1s1 = _vel_vshf_vvvsl(vrA_r1,vrB_r1,VE_VSHUFFLE_YUZU, 256) ;
	__vr vrCD_r1s1 = _vel_vshf_vvvsl(vrC_r1,vrD_r1,VE_VSHUFFLE_YUZU, 256) ;
	__vr vrEF_r1s1 = _vel_vshf_vvvsl(vrE_r1,vrF_r1,VE_VSHUFFLE_YUZU, 256) ;
	_vel_vst_vssl(vr01_r1s1, 4*4*16, filter+3*16+0, 256) ;
	_vel_vst_vssl(vr23_r1s1, 4*4*16, filter+3*16+2, 256) ;
	_vel_vst_vssl(vr45_r1s1, 4*4*16, filter+3*16+4, 256) ;
	_vel_vst_vssl(vr67_r1s1, 4*4*16, filter+3*16+6, 256) ;
	_vel_vst_vssl(vr89_r1s1, 4*4*16, filter+3*16+8, 256) ;
	_vel_vst_vssl(vrAB_r1s1, 4*4*16, filter+3*16+10, 256) ;
	_vel_vst_vssl(vrCD_r1s1, 4*4*16, filter+3*16+12, 256) ;
	_vel_vst_vssl(vrEF_r1s1, 4*4*16, filter+3*16+14, 256) ;
      }
      for (int64_t c1 = 0 ; c1 < 256 ; c1++) {
	const int64_t c = c0 + c1 ;

	const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;
	const uint64_t* filter_u64 = (const uint64_t*)(filter+c1*64) ;

	/* memory access errors mihgt be caused */
	__vr vrin_r0s0 = _vel_vldu_vssl(4, pInChannel+op-inWidth-1, vl) ;
	__vr vrin_r0s1 = _vel_vldu_vssl(4, pInChannel+op-inWidth  , vl) ;
	__vr vrin_r1s0 = _vel_vldu_vssl(4, pInChannel+op        -1, vl) ;
	__vr vrin_r1s1 = _vel_vldu_vssl(4, pInChannel+op          , vl) ;

	vrin_r0s0 = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrin_r0s0, vm_r0s0, vl) ;
	vrin_r0s1 = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrin_r0s1, vm_r0s1, vl) ;
	vrin_r1s0 = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrin_r1s0, vm_r1s0, vl) ;

	__vr vrinP_r0s0 = _vel_vshf_vvvsl(vrin_r0s0, vrin_r0s0, VE_VSHUFFLE_YUZU, vl) ;
	__vr vrinP_r0s1 = _vel_vshf_vvvsl(vrin_r0s1, vrin_r0s1, VE_VSHUFFLE_YUZU, vl) ;
	__vr vrinP_r1s0 = _vel_vshf_vvvsl(vrin_r1s0, vrin_r1s0, VE_VSHUFFLE_YUZU, vl) ;
	__vr vrinP_r1s1 = _vel_vshf_vvvsl(vrin_r1s1, vrin_r1s1, VE_VSHUFFLE_YUZU, vl) ;

	vrsum01 = _vel_pvfmad_vvsvl(vrsum01, filter_u64[0], vrinP_r0s0, vl) ;
	vrsum23 = _vel_pvfmad_vvsvl(vrsum23, filter_u64[1], vrinP_r0s0, vl) ;
	vrsum45 = _vel_pvfmad_vvsvl(vrsum45, filter_u64[2], vrinP_r0s0, vl) ;
	vrsum67 = _vel_pvfmad_vvsvl(vrsum67, filter_u64[3], vrinP_r0s0, vl) ;
	vrsum89 = _vel_pvfmad_vvsvl(vrsum89, filter_u64[4], vrinP_r0s0, vl) ;
	vrsumAB = _vel_pvfmad_vvsvl(vrsumAB, filter_u64[5], vrinP_r0s0, vl) ;
	vrsumCD = _vel_pvfmad_vvsvl(vrsumCD, filter_u64[6], vrinP_r0s0, vl) ;
	vrsumEF = _vel_pvfmad_vvsvl(vrsumEF, filter_u64[7], vrinP_r0s0, vl) ;

	vrsum01 = _vel_pvfmad_vvsvl(vrsum01, filter_u64[8], vrinP_r0s1, vl) ;
	vrsum23 = _vel_pvfmad_vvsvl(vrsum23, filter_u64[9], vrinP_r0s1, vl) ;
	vrsum45 = _vel_pvfmad_vvsvl(vrsum45, filter_u64[10], vrinP_r0s1, vl) ;
	vrsum67 = _vel_pvfmad_vvsvl(vrsum67, filter_u64[11], vrinP_r0s1, vl) ;
	vrsum89 = _vel_pvfmad_vvsvl(vrsum89, filter_u64[12], vrinP_r0s1, vl) ;
	vrsumAB = _vel_pvfmad_vvsvl(vrsumAB, filter_u64[13], vrinP_r0s1, vl) ;
	vrsumCD = _vel_pvfmad_vvsvl(vrsumCD, filter_u64[14], vrinP_r0s1, vl) ;
	vrsumEF = _vel_pvfmad_vvsvl(vrsumEF, filter_u64[15], vrinP_r0s1, vl) ;

	vrsum01 = _vel_pvfmad_vvsvl(vrsum01, filter_u64[16], vrinP_r1s0, vl) ;
	vrsum23 = _vel_pvfmad_vvsvl(vrsum23, filter_u64[17], vrinP_r1s0, vl) ;
	vrsum45 = _vel_pvfmad_vvsvl(vrsum45, filter_u64[18], vrinP_r1s0, vl) ;
	vrsum67 = _vel_pvfmad_vvsvl(vrsum67, filter_u64[19], vrinP_r1s0, vl) ;
	vrsum89 = _vel_pvfmad_vvsvl(vrsum89, filter_u64[20], vrinP_r1s0, vl) ;
	vrsumAB = _vel_pvfmad_vvsvl(vrsumAB, filter_u64[21], vrinP_r1s0, vl) ;
	vrsumCD = _vel_pvfmad_vvsvl(vrsumCD, filter_u64[22], vrinP_r1s0, vl) ;
	vrsumEF = _vel_pvfmad_vvsvl(vrsumEF, filter_u64[23], vrinP_r1s0, vl) ;

	vrsum01 = _vel_pvfmad_vvsvl(vrsum01, filter_u64[24], vrinP_r1s1, vl) ;
	vrsum23 = _vel_pvfmad_vvsvl(vrsum23, filter_u64[25], vrinP_r1s1, vl) ;
	vrsum45 = _vel_pvfmad_vvsvl(vrsum45, filter_u64[26], vrinP_r1s1, vl) ;
	vrsum67 = _vel_pvfmad_vvsvl(vrsum67, filter_u64[27], vrinP_r1s1, vl) ;
	vrsum89 = _vel_pvfmad_vvsvl(vrsum89, filter_u64[28], vrinP_r1s1, vl) ;
	vrsumAB = _vel_pvfmad_vvsvl(vrsumAB, filter_u64[29], vrinP_r1s1, vl) ;
	vrsumCD = _vel_pvfmad_vvsvl(vrsumCD, filter_u64[30], vrinP_r1s1, vl) ;
	vrsumEF = _vel_pvfmad_vvsvl(vrsumEF, filter_u64[31], vrinP_r1s1, vl) ;
      }
    } // inChannel

    _vel_vstu_vssl(vrsum01, 4, pOut+outIndex + 0 * outHeight*outWidth, vl) ;
    _vel_vstl_vssl(vrsum01, 4, pOut+outIndex + 1 * outHeight*outWidth, vl) ;
    _vel_vstu_vssl(vrsum23, 4, pOut+outIndex + 2 * outHeight*outWidth, vl) ;
    _vel_vstl_vssl(vrsum23, 4, pOut+outIndex + 3 * outHeight*outWidth, vl) ;
    _vel_vstu_vssl(vrsum45, 4, pOut+outIndex + 4 * outHeight*outWidth, vl) ;
    _vel_vstl_vssl(vrsum45, 4, pOut+outIndex + 5 * outHeight*outWidth, vl) ;
    _vel_vstu_vssl(vrsum67, 4, pOut+outIndex + 6 * outHeight*outWidth, vl) ;
    _vel_vstl_vssl(vrsum67, 4, pOut+outIndex + 7 * outHeight*outWidth, vl) ;
    _vel_vstu_vssl(vrsum89, 4, pOut+outIndex + 8 * outHeight*outWidth, vl) ;
    _vel_vstl_vssl(vrsum89, 4, pOut+outIndex + 9 * outHeight*outWidth, vl) ;
    _vel_vstu_vssl(vrsumAB, 4, pOut+outIndex +10 * outHeight*outWidth, vl) ;
    _vel_vstl_vssl(vrsumAB, 4, pOut+outIndex +11 * outHeight*outWidth, vl) ;
    _vel_vstu_vssl(vrsumCD, 4, pOut+outIndex +12 * outHeight*outWidth, vl) ;
    _vel_vstl_vssl(vrsumCD, 4, pOut+outIndex +13 * outHeight*outWidth, vl) ;
    _vel_vstu_vssl(vrsumEF, 4, pOut+outIndex +14 * outHeight*outWidth, vl) ;
    _vel_vstl_vssl(vrsumEF, 4, pOut+outIndex +15 * outHeight*outWidth, vl) ;

    outIndex += vl ;
  } // outPixels
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
    const int64_t kernWidth,
    const int64_t kernHeight,
    const int64_t inChannelGroup,
    const int64_t outChannelGroup
)
{
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
	  func_odd<FLAYOUT, 1, ADDBIAS>(pIn, pKernel, pBias, pOut,
	     inChannel, inWidth, inHeight,
	     outChannel, outWidth, outHeight,
	     kernWidth, kernHeight,
	     inChannelGroup, outChannelGroup,
	     inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	     n, k );
	  k+=1 ;
	  break ;
	case 2 :
	  func_even<FLAYOUT, 2, ADDBIAS>(pIn, pKernel, pBias, pOut,
	     inChannel, inWidth, inHeight,
	     outChannel, outWidth, outHeight,
	     kernWidth, kernHeight,
	     inChannelGroup, outChannelGroup,
	     inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	     n, k );
	  k+=2 ;
	  break ;
	case 3 :
	  func_odd<FLAYOUT, 3, ADDBIAS>(pIn, pKernel, pBias, pOut,
	     inChannel, inWidth, inHeight,
	     outChannel, outWidth, outHeight,
	     kernWidth, kernHeight,
	     inChannelGroup, outChannelGroup,
	     inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	     n, k );
	  k+=3 ;
	  break ;
	case 4 :
	  func_even<FLAYOUT, 4, ADDBIAS>(pIn, pKernel, pBias, pOut,
	     inChannel, inWidth, inHeight,
	     outChannel, outWidth, outHeight,
	     kernWidth, kernHeight,
	     inChannelGroup, outChannelGroup,
	     inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	     n, k );
	  k+=4 ;
	  break ;
	case 5 :
	  func_odd<FLAYOUT, 5, ADDBIAS>(pIn, pKernel, pBias, pOut,
	     inChannel, inWidth, inHeight,
	     outChannel, outWidth, outHeight,
	     kernWidth, kernHeight,
	     inChannelGroup, outChannelGroup,
	     inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	     n, k );
	  k+=5 ;
	  break ;
	case 6 :
	  func_even<FLAYOUT, 6, ADDBIAS>(pIn, pKernel, pBias, pOut,
	     inChannel, inWidth, inHeight,
	     outChannel, outWidth, outHeight,
	     kernWidth, kernHeight,
	     inChannelGroup, outChannelGroup,
	     inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	     n, k );
	  k+=6 ;
	  break ;
	case 7 :
	  func_odd<FLAYOUT, 7, ADDBIAS>(pIn, pKernel, pBias, pOut,
	     inChannel, inWidth, inHeight,
	     outChannel, outWidth, outHeight,
	     kernWidth, kernHeight,
	     inChannelGroup, outChannelGroup,
	     inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	     n, k );
	  k+=7 ;
	  break ;
	case 8 :
	  if( FLAYOUT == VEDNN_FILTER_LAYOUT_NCHW  &&
	      ( inChannel % 512 == 0 && (((uint64_t) pKernel) & 0x7) == 0 ) ) {
	    k8_filter_nchw_c512x<ADDBIAS>(pIn, pKernel, pBias, pOut,
	       inChannel, inWidth, inHeight,
	       outChannel, outWidth, outHeight,
	       kernWidth, kernHeight,
	       inChannelGroup, outChannelGroup,
	       inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	       n, k );
	  }
	  else {
	    func_even<FLAYOUT, 8, ADDBIAS>(pIn, pKernel, pBias, pOut,
	       inChannel, inWidth, inHeight,
	       outChannel, outWidth, outHeight,
	       kernWidth, kernHeight,
	       inChannelGroup, outChannelGroup,
	       inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	       n, k );
	  }
	  k+=8 ;
	  break ;
	case 9 :
	  func_odd<FLAYOUT, 9, ADDBIAS>(pIn, pKernel, pBias, pOut,
	     inChannel, inWidth, inHeight,
	     outChannel, outWidth, outHeight,
	     kernWidth, kernHeight,
	     inChannelGroup, outChannelGroup,
	     inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	     n, k );
	  k+=9 ;
	  break ;
	case 10 :
	  func_even<FLAYOUT, 10, ADDBIAS>(pIn, pKernel, pBias, pOut,
	     inChannel, inWidth, inHeight,
	     outChannel, outWidth, outHeight,
	     kernWidth, kernHeight,
	     inChannelGroup, outChannelGroup,
	     inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	     n, k );
	  k+=10 ;
	  break ;
	case 11 :
	  func_odd<FLAYOUT, 11, ADDBIAS>(pIn, pKernel, pBias, pOut,
	     inChannel, inWidth, inHeight,
	     outChannel, outWidth, outHeight,
	     kernWidth, kernHeight,
	     inChannelGroup, outChannelGroup,
	     inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	     n, k );
	  k+=11 ;
	  break ;
	case 12 :
	  func_even<FLAYOUT, 12, ADDBIAS>(pIn, pKernel, pBias, pOut,
	     inChannel, inWidth, inHeight,
	     outChannel, outWidth, outHeight,
	     kernWidth, kernHeight,
	     inChannelGroup, outChannelGroup,
	     inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	     n, k );
	  k+=12 ;
	  break ;
	case 13 :
	  func_odd<FLAYOUT, 13, ADDBIAS>(pIn, pKernel, pBias, pOut,
	     inChannel, inWidth, inHeight,
	     outChannel, outWidth, outHeight,
	     kernWidth, kernHeight,
	     inChannelGroup, outChannelGroup,
	     inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	     n, k );
	  k+=13 ;
	  break ;
	case 14 :
	  func_even<FLAYOUT, 14, ADDBIAS>(pIn, pKernel, pBias, pOut,
	     inChannel, inWidth, inHeight,
	     outChannel, outWidth, outHeight,
	     kernWidth, kernHeight,
	     inChannelGroup, outChannelGroup,
	     inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	     n, k );
	  k+=14 ;
	  break ;
	case 15 :
	  func_odd<FLAYOUT, 15, ADDBIAS>(pIn, pKernel, pBias, pOut,
	     inChannel, inWidth, inHeight,
	     outChannel, outWidth, outHeight,
	     kernWidth, kernHeight,
	     inChannelGroup, outChannelGroup,
	     inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	     n, k );
	  k+=15 ;
	  break ;
	default :
	  break ;
	}
	for (; k < outChannelGroup; k+=16) {
	  if( FLAYOUT == VEDNN_FILTER_LAYOUT_NCHW  &&
	      ( inChannel % 256 == 0 && (((uint64_t) pKernel) & 0x7) == 0 ) ) {
	    k16_filter_nchw_c256x<ADDBIAS>(pIn, pKernel, pBias, pOut,
	       inChannel, inWidth, inHeight,
	       outChannel, outWidth, outHeight,
	       kernWidth, kernHeight,
	       inChannelGroup, outChannelGroup,
	       inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	       n, k );
	  }
	  else {
	    func_even<FLAYOUT, 16, ADDBIAS>(pIn, pKernel, pBias, pOut,
	       inChannel, inWidth, inHeight,
	       outChannel, outWidth, outHeight,
	       kernWidth, kernHeight,
	       inChannelGroup, outChannelGroup,
	       inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	       n, k );
	  }
	} // outChannel
    } // group
  } // batch
}


extern "C" vednnError_t
vednnConvolutionForward_direct_dil1_str1_padsame_ker2(
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
  const int64_t outWidth   = pParamOut->width;		/* must be equal to inWidth */
  const int64_t outHeight  = pParamOut->height;		/* must be equal to inHeight */
  const int64_t kernWidth  = pParamKernel->width;	/* must be 2 */
  const int64_t kernHeight = pParamKernel->height;	/* must be 2 */

  const int64_t filter_layout = pParamKernel->layout ;

  const int64_t group          = pParamConv->group;
//  const int64_t strideWidth    = pParamConv->strideWidth;	/* must be 1 */
//  const int64_t strideHeight   = pParamConv->strideHeight;	/* must be 1 */
//  const int64_t padWidth       = pParamConv->padWidth;	/* must be 0 */
//  const int64_t padHeight      = pParamConv->padHeight;	/* must be 0 */
//  const int64_t dilationWidth  = pParamConv->dilationWidth;	/* must be 1 */
//  const int64_t dilationHeight = pParamConv->dilationHeight;	/* must be 1 */

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
		 inChannelGroup, outChannelGroup ) ;
    }
    else {
      convloop<VEDNN_FILTER_LAYOUT_NCHW, true>(pIn, pKernel, pBias, pOut,
		 batch, group,
		 inChannel, inWidth, inHeight,
		 outChannel, outWidth, outHeight,
		 kernWidth, kernHeight,
		 inChannelGroup, outChannelGroup ) ;
    }
  }
  else {
    if( pDataBias == NULL ) {
      convloop<VEDNN_FILTER_LAYOUT_HWCN, false>(pIn, pKernel, pBias, pOut,
		 batch, group,
		 inChannel, inWidth, inHeight,
		 outChannel, outWidth, outHeight,
		 kernWidth, kernHeight,
		 inChannelGroup, outChannelGroup ) ;
    }
    else {
      convloop<VEDNN_FILTER_LAYOUT_HWCN, true>(pIn, pKernel, pBias, pOut,
		 batch, group,
		 inChannel, inWidth, inHeight,
		 outChannel, outWidth, outHeight,
		 kernWidth, kernHeight,
		 inChannelGroup, outChannelGroup ) ;
    }
  }


//  const float * pIn     = (const float *) pDataIn;
//  const float * pKernel = (const float *) pDataKernel;
//  float * const pOut    = (float * const) pDataOut;
//
//  const int oPixels= outHeight*outWidth ;
//
//  float __attribute__ ((aligned(8))) filter[4*16*256] ;
//
//  {
//    for (int64_t n = 0; n < batch; n++) {
//      for (int64_t g = 0; g < group; g++) {
//	const int64_t inGroupOffset   = g * inChannelGroup * inHeight * inWidth;
//	const int64_t outGroupOffset  = g * outChannelGroup * outHeight * outWidth;
//	const int64_t kernGroupOffset = g * outChannelGroup * inChannelGroup * kernHeight * kernWidth;
//
//	int k = 0 ;
//
//	if ( (outChannelGroup & 0x01) == 1 ) {
//	  if( filter_layout == VEDNN_FILTER_LAYOUT_NCHW) {
//	    k1<VEDNN_FILTER_LAYOUT_NCHW>(pIn, pKernel, pOut,
//	       inChannel, inWidth, inHeight,
//	       outChannel, outWidth, outHeight,
//	       kernWidth, kernHeight,
//	       inChannelGroup, outChannelGroup,
//	       inGroupOffset, outGroupOffset,
//	       kernGroupOffset, oPixels, n, k) ;
//	  }
//	  else {
//	    k1<VEDNN_FILTER_LAYOUT_HWCN>(pIn, pKernel, pOut,
//	       inChannel, inWidth, inHeight,
//	       outChannel, outWidth, outHeight,
//	       kernWidth, kernHeight,
//	       inChannelGroup, outChannelGroup,
//	       inGroupOffset, outGroupOffset,
//	       kernGroupOffset, oPixels, n, k) ;
//	  }
//	  k++ ;
//	}
//	if ( ((outChannelGroup >> 1) & 0x01) == 1 ) {
//	  if( filter_layout == VEDNN_FILTER_LAYOUT_NCHW) {
//	    k2<VEDNN_FILTER_LAYOUT_NCHW>(pIn, pKernel, pOut,
//	       inChannel, inWidth, inHeight,
//	       outChannel, outWidth, outHeight,
//	       kernWidth, kernHeight,
//	       inChannelGroup, outChannelGroup,
//	       inGroupOffset, outGroupOffset,
//	       kernGroupOffset, oPixels, n, k) ;
//	  }
//	  else {
//	    k2<VEDNN_FILTER_LAYOUT_HWCN>(pIn, pKernel, pOut,
//	       inChannel, inWidth, inHeight,
//	       outChannel, outWidth, outHeight,
//	       kernWidth, kernHeight,
//	       inChannelGroup, outChannelGroup,
//	       inGroupOffset, outGroupOffset,
//	       kernGroupOffset, oPixels, n, k) ;
//	  }
//
//	  k+=2 ;
//	}
//	if ( ((outChannelGroup >> 2) & 0x01) == 1 ) {
//	  if( filter_layout == VEDNN_FILTER_LAYOUT_NCHW) {
//	    k4<VEDNN_FILTER_LAYOUT_NCHW>(pIn, pKernel, pOut,
//	       inChannel, inWidth, inHeight,
//	       outChannel, outWidth, outHeight,
//	       kernWidth, kernHeight,
//	       inChannelGroup, outChannelGroup,
//	       inGroupOffset, outGroupOffset,
//	       kernGroupOffset, oPixels, n, k) ;
//	  }
//	  else {
//	    k4<VEDNN_FILTER_LAYOUT_HWCN>(pIn, pKernel, pOut,
//	       inChannel, inWidth, inHeight,
//	       outChannel, outWidth, outHeight,
//	       kernWidth, kernHeight,
//	       inChannelGroup, outChannelGroup,
//	       inGroupOffset, outGroupOffset,
//	       kernGroupOffset, oPixels, n, k) ;
//	  }
//
//	  k+=4 ;
//	}
//	if ( ((outChannelGroup >> 3) & 0x01) == 1 ) {
//	  if( filter_layout == VEDNN_FILTER_LAYOUT_NCHW) {
//	    if( inChannel % 512 == 0 && (((uint64_t)pDataKernel) & 0x7) == 0 ) {
//	      k8_c512x(pIn, pKernel, pOut,
//		 inChannel, inWidth, inHeight,
//		 outChannel, outWidth, outHeight,
//		 kernWidth, kernHeight,
//		 inChannelGroup, outChannelGroup,
//		 inGroupOffset, outGroupOffset,
//		 kernGroupOffset, oPixels, n, k,
//		 filter ) ;
//	    }
//	    else {
//	      k8<VEDNN_FILTER_LAYOUT_NCHW>(pIn, pKernel, pOut,
//		 inChannel, inWidth, inHeight,
//		 outChannel, outWidth, outHeight,
//		 kernWidth, kernHeight,
//		 inChannelGroup, outChannelGroup,
//		 inGroupOffset, outGroupOffset,
//		 kernGroupOffset, oPixels, n, k) ;
//	    }
//	  }
//	  else {
//	    k8<VEDNN_FILTER_LAYOUT_HWCN>(pIn, pKernel, pOut,
//	       inChannel, inWidth, inHeight,
//	       outChannel, outWidth, outHeight,
//	       kernWidth, kernHeight,
//	       inChannelGroup, outChannelGroup,
//	       inGroupOffset, outGroupOffset,
//	       kernGroupOffset, oPixels, n, k) ;
//	  }
//	  k+=8 ;
//	}
//	for ( ; k < outChannelGroup; k+=16) {
//	  if( filter_layout == VEDNN_FILTER_LAYOUT_NCHW) {
//	    if( inChannel % 256 == 0 && (((uint64_t)pDataKernel) & 0x7) == 0 ) {
//	      k16_c256x(pIn, pKernel, pOut,
//		 inChannel, inWidth, inHeight,
//		 outChannel, outWidth, outHeight,
//		 kernWidth, kernHeight,
//		 inChannelGroup, outChannelGroup,
//		 inGroupOffset, outGroupOffset,
//		 kernGroupOffset, oPixels, n, k,
//		 filter) ;
//	    }
//	    else {
//	      k16<VEDNN_FILTER_LAYOUT_NCHW>(pIn, pKernel, pOut,
//		 inChannel, inWidth, inHeight,
//		 outChannel, outWidth, outHeight,
//		 kernWidth, kernHeight,
//		 inChannelGroup, outChannelGroup,
//		 inGroupOffset, outGroupOffset,
//		 kernGroupOffset, oPixels, n, k) ;
//	    }
//	  }
//	  else {
//	    k16<VEDNN_FILTER_LAYOUT_HWCN>(pIn, pKernel, pOut,
//	       inChannel, inWidth, inHeight,
//	       outChannel, outWidth, outHeight,
//	       kernWidth, kernHeight,
//	       inChannelGroup, outChannelGroup,
//	       inGroupOffset, outGroupOffset,
//	       kernGroupOffset, oPixels, n, k) ;
//	  }
//	} // outChannel
//      } // group
//    } // batch
//  }

  return VEDNN_SUCCESS;
}
