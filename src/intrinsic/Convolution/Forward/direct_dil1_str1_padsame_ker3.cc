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
  const int64_t padHeight,
  const int64_t padWidth,
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

    __vr vrsum0  = _vel_vbrds_vsl(bias0, vl) ;
    __vr vrsum12 = _vel_pvbrd_vsl(bias12, vl) ;
    __vr vrsum34 = _vel_pvbrd_vsl(bias34, vl) ;
    __vr vrsum56 = _vel_pvbrd_vsl(bias56, vl) ;
    __vr vrsum78 = _vel_pvbrd_vsl(bias78, vl) ;
    __vr vrsum9A = _vel_pvbrd_vsl(bias9A, vl) ;
    __vr vrsumBC = _vel_pvbrd_vsl(biasBC, vl) ;
    __vr vrsumDE = _vel_pvbrd_vsl(biasDE, vl) ;

    __vr vrseq = _vel_vseq_vl(vl) ;			// xy
    __vr vridx = _vel_vaddsl_vsvl(op, vrseq, vl) ;	// op + xy

    __vr vry   = _vel_vdivsl_vvsl(vridx, outWidth, vl) ;
    __vr vrx   = _vel_vsubsl_vvvl(vridx, _vel_vmulul_vsvl(outWidth,vry, vl), vl) ;

    __vr vrh_r0 = _vel_vaddsl_vsvl( -1, vry, vl) ;
    __vr vrh_r2 = _vel_vaddsl_vsvl(2-1, vry, vl) ;

    __vr vrw_s0 = _vel_vaddsl_vsvl( -1, vrx, vl) ;
    __vr vrw_s2 = _vel_vaddsl_vsvl(2-1, vrx, vl) ;

    __vm256 vm01_r0 =  _vel_vfmklge_mvl(vrh_r0, vl) ;
    __vm256 vm01_r2 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(inHeight,vrh_r2, vl), vl) ;

    __vm256 vm23_s0  =  _vel_vfmklge_mvl(vrw_s0, vl) ;
    __vm256 vm23_s2  =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(inWidth,vrw_s2, vl), vl) ;


    __vm256 vmall_r0s0 = _vel_andm_mmm(vm01_r0,vm23_s0) ;
    __vm256 vmall_r0s1 = vm01_r0 ;
    __vm256 vmall_r0s2 = _vel_andm_mmm(vm01_r0, vm23_s2) ;

    __vm256 vmall_r1s0 = vm23_s0 ;
    __vm256 vmall_r1s2 = vm23_s2 ;

    __vm256 vmall_r2s0 = _vel_andm_mmm(vm01_r2,vm23_s0) ;
    __vm256 vmall_r2s1 = vm01_r2 ;
    __vm256 vmall_r2s2 = _vel_andm_mmm(vm01_r2, vm23_s2) ;

    for (int64_t c = 0; c < inChannelGroup; c++) {
      const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;

      /* memory access errors mihgt be caused */
      __vr vrin_r0s0 = _vel_vldu_vssl(4,&pInChannel[op-inWidth-1], vl) ;
      __vr vrin_r0s1 = _vel_vldu_vssl(4,&pInChannel[op-inWidth  ], vl) ;
      __vr vrin_r0s2 = _vel_vldu_vssl(4,&pInChannel[op-inWidth+1], vl) ;
      __vr vrin_r1s0 = _vel_vldu_vssl(4,&pInChannel[op+       -1], vl) ;
      __vr vrin_r1s1 = _vel_vldu_vssl(4,&pInChannel[op          ], vl) ;
      __vr vrin_r1s2 = _vel_vldu_vssl(4,&pInChannel[op+       +1], vl) ;
      __vr vrin_r2s0 = _vel_vldu_vssl(4,&pInChannel[op+inWidth-1], vl) ;
      __vr vrin_r2s1 = _vel_vldu_vssl(4,&pInChannel[op+inWidth  ], vl) ;
      __vr vrin_r2s2 = _vel_vldu_vssl(4,&pInChannel[op+inWidth+1], vl) ;

      __vr vrzerof = _vel_vbrds_vsl(0.0f, vl) ;

      vrin_r0s0 = _vel_vmrg_vvvml(vrzerof, vrin_r0s0, vmall_r0s0, vl) ;
      vrin_r0s1 = _vel_vmrg_vvvml(vrzerof, vrin_r0s1, vmall_r0s1, vl) ;
      vrin_r0s2 = _vel_vmrg_vvvml(vrzerof, vrin_r0s2, vmall_r0s2, vl) ;
      __vr vrinP_r0s0 = _vel_vshf_vvvsl(vrin_r0s0, vrin_r0s0, VE_VSHUFFLE_YUZU, vl) ;
      __vr vrinP_r0s1 = _vel_vshf_vvvsl(vrin_r0s1, vrin_r0s1, VE_VSHUFFLE_YUZU, vl) ;
      __vr vrinP_r0s2 = _vel_vshf_vvvsl(vrin_r0s2, vrin_r0s2, VE_VSHUFFLE_YUZU, vl) ;

      vrin_r1s0 = _vel_vmrg_vvvml(vrzerof, vrin_r1s0, vmall_r1s0, vl) ;
      vrin_r1s2 = _vel_vmrg_vvvml(vrzerof, vrin_r1s2, vmall_r1s2, vl) ;
      __vr vrinP_r1s0 = _vel_vshf_vvvsl(vrin_r1s0, vrin_r1s0, VE_VSHUFFLE_YUZU, vl) ;
      __vr vrinP_r1s1 = _vel_vshf_vvvsl(vrin_r1s1, vrin_r1s1, VE_VSHUFFLE_YUZU, vl) ;
      __vr vrinP_r1s2 = _vel_vshf_vvvsl(vrin_r1s2, vrin_r1s2, VE_VSHUFFLE_YUZU, vl) ;

      vrin_r2s0 = _vel_vmrg_vvvml(vrzerof, vrin_r2s0, vmall_r2s0, vl) ;
      vrin_r2s1 = _vel_vmrg_vvvml(vrzerof, vrin_r2s1, vmall_r2s1, vl) ;
      vrin_r2s2 = _vel_vmrg_vvvml(vrzerof, vrin_r2s2, vmall_r2s2, vl) ;
      __vr vrinP_r2s0 = _vel_vshf_vvvsl(vrin_r2s0, vrin_r2s0, VE_VSHUFFLE_YUZU, vl) ;
      __vr vrinP_r2s1 = _vel_vshf_vvvsl(vrin_r2s1, vrin_r2s1, VE_VSHUFFLE_YUZU, vl) ;
      __vr vrinP_r2s2 = _vel_vshf_vvvsl(vrin_r2s2, vrin_r2s2, VE_VSHUFFLE_YUZU, vl) ;

#define FILTER_OFFSET(k,c,r,s) ( kernGroupOffset + filter_index<FLAYOUT>(k,c,r,s, inChannelGroup, outChannelGroup, kernHeight, kernWidth) )
#define VFMAD(VRIN, R, S) 								\
      {											\
	__vr vrinP = _vel_vshf_vvvsl(VRIN, VRIN, VE_VSHUFFLE_YUZU, vl) ;		\
	const float    kerValue0  = pKernel[FILTER_OFFSET(k+ 0,c,R,S)] ;		\
	const uint64_t kerValue12 = _vel_pack_f32p(pKernel+FILTER_OFFSET(k+ 1,c,R,S),	\
						   pKernel+FILTER_OFFSET(k+ 2,c,R,S)) ;	\
	vrsum0 = _vel_vfmads_vvsvl(vrsum0, kerValue0, VRIN, vl) ;			\
	if(NUMKERNEL>= 3) vrsum12 = _vel_pvfmad_vvsvl(vrsum12, kerValue12, vrinP, vl) ;	\
	const uint64_t kerValue34 = _vel_pack_f32p(pKernel+FILTER_OFFSET(k+ 3,c,R,S),	\
						   pKernel+FILTER_OFFSET(k+ 4,c,R,S)) ;	\
	const uint64_t kerValue56 = _vel_pack_f32p(pKernel+FILTER_OFFSET(k+ 5,c,R,S),	\
						   pKernel+FILTER_OFFSET(k+ 6,c,R,S)) ;	\
	if(NUMKERNEL>= 5) vrsum34 = _vel_pvfmad_vvsvl(vrsum34, kerValue34, vrinP, vl) ;	\
	if(NUMKERNEL>= 7) vrsum56 = _vel_pvfmad_vvsvl(vrsum56, kerValue56, vrinP, vl) ;	\
	const uint64_t kerValue78 = _vel_pack_f32p(pKernel+FILTER_OFFSET(k+ 7,c,R,S),	\
						   pKernel+FILTER_OFFSET(k+ 8,c,R,S)) ;	\
	const uint64_t kerValue9A = _vel_pack_f32p(pKernel+FILTER_OFFSET(k+ 9,c,R,S),	\
						   pKernel+FILTER_OFFSET(k+10,c,R,S)) ;	\
	if(NUMKERNEL>= 9) vrsum78 = _vel_pvfmad_vvsvl(vrsum78, kerValue78, vrinP, vl) ;	\
	if(NUMKERNEL>=11) vrsum9A = _vel_pvfmad_vvsvl(vrsum9A, kerValue9A, vrinP, vl) ;	\
	const uint64_t kerValueBC = _vel_pack_f32p(pKernel+FILTER_OFFSET(k+11,c,R,S),	\
						   pKernel+FILTER_OFFSET(k+12,c,R,S)) ;	\
	const uint64_t kerValueDE = _vel_pack_f32p(pKernel+FILTER_OFFSET(k+13,c,R,S),	\
						   pKernel+FILTER_OFFSET(k+14,c,R,S)) ;	\
	if(NUMKERNEL>=13) vrsumBC = _vel_pvfmad_vvsvl(vrsumBC, kerValueBC, vrinP, vl) ;	\
	if(NUMKERNEL>=15) vrsumDE = _vel_pvfmad_vvsvl(vrsumDE, kerValueDE, vrinP, vl) ;	\
      }


      VFMAD(vrinP_r0s0, 0, 0) ;
      VFMAD(vrinP_r0s1, 0, 1) ;
      VFMAD(vrinP_r0s2, 0, 2) ;
      VFMAD(vrinP_r1s0, 1, 0) ;
      VFMAD(vrinP_r1s1, 1, 1) ;
      VFMAD(vrinP_r1s2, 1, 2) ;
      VFMAD(vrinP_r2s0, 2, 0) ;
      VFMAD(vrinP_r2s1, 2, 1) ;
      VFMAD(vrinP_r2s2, 2, 2) ;

#undef VFMAD
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
  const int64_t padHeight,
  const int64_t padWidth,
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

    __vr vrsum01 = _vel_pvbrd_vsl(bias01, vl) ;
    __vr vrsum23 = _vel_pvbrd_vsl(bias23, vl) ;
    __vr vrsum45 = _vel_pvbrd_vsl(bias45, vl) ;
    __vr vrsum67 = _vel_pvbrd_vsl(bias67, vl) ;
    __vr vrsum89 = _vel_pvbrd_vsl(bias89, vl) ;
    __vr vrsumAB = _vel_pvbrd_vsl(biasAB, vl) ;
    __vr vrsumCD = _vel_pvbrd_vsl(biasCD, vl) ;
    __vr vrsumEF = _vel_pvbrd_vsl(biasEF, vl) ;

    __vr vrseq = _vel_vseq_vl(vl) ;			// xy
    __vr vridx = _vel_vaddsl_vsvl(op, vrseq, vl) ;	// op + xy

    __vr vry   = _vel_vdivsl_vvsl(vridx, outWidth, vl) ;
    __vr vrx   = _vel_vsubsl_vvvl(vridx, _vel_vmulul_vsvl(outWidth,vry, vl), vl) ;

    __vr vrh_r0 = _vel_vaddsl_vsvl( -1, vry, vl) ;
    __vr vrh_r2 = _vel_vaddsl_vsvl(2-1, vry, vl) ;

    __vr vrw_s0 = _vel_vaddsl_vsvl( -1, vrx, vl) ;
    __vr vrw_s2 = _vel_vaddsl_vsvl(2-1, vrx, vl) ;

    __vm256 vm01_r0 =  _vel_vfmklge_mvl(vrh_r0, vl) ;
    __vm256 vm01_r2 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(inHeight,vrh_r2, vl), vl) ;

    __vm256 vm23_s0  =  _vel_vfmklge_mvl(vrw_s0, vl) ;
    __vm256 vm23_s2  =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(inWidth,vrw_s2, vl), vl) ;


    __vm256 vmall_r0s0 = _vel_andm_mmm(vm01_r0,vm23_s0) ;
    __vm256 vmall_r0s1 = vm01_r0 ;
    __vm256 vmall_r0s2 = _vel_andm_mmm(vm01_r0, vm23_s2) ;

    __vm256 vmall_r1s0 = vm23_s0 ;
    __vm256 vmall_r1s2 = vm23_s2 ;

    __vm256 vmall_r2s0 = _vel_andm_mmm(vm01_r2,vm23_s0) ;
    __vm256 vmall_r2s1 = vm01_r2 ;
    __vm256 vmall_r2s2 = _vel_andm_mmm(vm01_r2, vm23_s2) ;

    for (int64_t c = 0; c < inChannelGroup; c++) {
      const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;

      /* memory access errors mihgt be caused */
      __vr vrin_r0s0 = _vel_vldu_vssl(4,&pInChannel[op-inWidth-1], vl) ;
      __vr vrin_r0s1 = _vel_vldu_vssl(4,&pInChannel[op-inWidth  ], vl) ;
      __vr vrin_r0s2 = _vel_vldu_vssl(4,&pInChannel[op-inWidth+1], vl) ;
      __vr vrin_r1s0 = _vel_vldu_vssl(4,&pInChannel[op+       -1], vl) ;
      __vr vrin_r1s1 = _vel_vldu_vssl(4,&pInChannel[op          ], vl) ;
      __vr vrin_r1s2 = _vel_vldu_vssl(4,&pInChannel[op+       +1], vl) ;
      __vr vrin_r2s0 = _vel_vldu_vssl(4,&pInChannel[op+inWidth-1], vl) ;
      __vr vrin_r2s1 = _vel_vldu_vssl(4,&pInChannel[op+inWidth  ], vl) ;
      __vr vrin_r2s2 = _vel_vldu_vssl(4,&pInChannel[op+inWidth+1], vl) ;

      __vr vrzerof = _vel_vbrds_vsl(0.0f, vl) ;

      vrin_r0s0 = _vel_vmrg_vvvml(vrzerof, vrin_r0s0, vmall_r0s0, vl) ;
      vrin_r0s1 = _vel_vmrg_vvvml(vrzerof, vrin_r0s1, vmall_r0s1, vl) ;
      vrin_r0s2 = _vel_vmrg_vvvml(vrzerof, vrin_r0s2, vmall_r0s2, vl) ;
      __vr vrinP_r0s0 = _vel_vshf_vvvsl(vrin_r0s0, vrin_r0s0, VE_VSHUFFLE_YUZU, vl) ;
      __vr vrinP_r0s1 = _vel_vshf_vvvsl(vrin_r0s1, vrin_r0s1, VE_VSHUFFLE_YUZU, vl) ;
      __vr vrinP_r0s2 = _vel_vshf_vvvsl(vrin_r0s2, vrin_r0s2, VE_VSHUFFLE_YUZU, vl) ;

      vrin_r1s0 = _vel_vmrg_vvvml(vrzerof, vrin_r1s0, vmall_r1s0, vl) ;
      vrin_r1s2 = _vel_vmrg_vvvml(vrzerof, vrin_r1s2, vmall_r1s2, vl) ;
      __vr vrinP_r1s0 = _vel_vshf_vvvsl(vrin_r1s0, vrin_r1s0, VE_VSHUFFLE_YUZU, vl) ;
      __vr vrinP_r1s1 = _vel_vshf_vvvsl(vrin_r1s1, vrin_r1s1, VE_VSHUFFLE_YUZU, vl) ;
      __vr vrinP_r1s2 = _vel_vshf_vvvsl(vrin_r1s2, vrin_r1s2, VE_VSHUFFLE_YUZU, vl) ;

      vrin_r2s0 = _vel_vmrg_vvvml(vrzerof, vrin_r2s0, vmall_r2s0, vl) ;
      vrin_r2s1 = _vel_vmrg_vvvml(vrzerof, vrin_r2s1, vmall_r2s1, vl) ;
      vrin_r2s2 = _vel_vmrg_vvvml(vrzerof, vrin_r2s2, vmall_r2s2, vl) ;
      __vr vrinP_r2s0 = _vel_vshf_vvvsl(vrin_r2s0, vrin_r2s0, VE_VSHUFFLE_YUZU, vl) ;
      __vr vrinP_r2s1 = _vel_vshf_vvvsl(vrin_r2s1, vrin_r2s1, VE_VSHUFFLE_YUZU, vl) ;
      __vr vrinP_r2s2 = _vel_vshf_vvvsl(vrin_r2s2, vrin_r2s2, VE_VSHUFFLE_YUZU, vl) ;

#define FILTER_OFFSET(k,c,r,s) ( kernGroupOffset + filter_index<FLAYOUT>(k,c,r,s, inChannelGroup, outChannelGroup, kernHeight, kernWidth) )
#define VFMAD(VRIN,R,S)								\
      {											\
	const uint64_t kerValue01 = _vel_pack_f32p(pKernel+FILTER_OFFSET(k+ 0,c,R,S),	\
						   pKernel+FILTER_OFFSET(k+ 1,c,R,S)) ;	\
	const uint64_t kerValue23 = _vel_pack_f32p(pKernel+FILTER_OFFSET(k+ 2,c,R,S),	\
						   pKernel+FILTER_OFFSET(k+ 3,c,R,S)) ;	\
	if(NUMKERNEL>= 2) vrsum01 = _vel_pvfmad_vvsvl(vrsum01, kerValue01, VRIN, vl) ;	\
	if(NUMKERNEL>= 4) vrsum23 = _vel_pvfmad_vvsvl(vrsum23, kerValue23, VRIN, vl) ;	\
	const uint64_t kerValue45 = _vel_pack_f32p(pKernel+FILTER_OFFSET(k+ 4,c,R,S),	\
						   pKernel+FILTER_OFFSET(k+ 5,c,R,S)) ;	\
	const uint64_t kerValue67 = _vel_pack_f32p(pKernel+FILTER_OFFSET(k+ 6,c,R,S),	\
						   pKernel+FILTER_OFFSET(k+ 7,c,R,S)) ;	\
	if(NUMKERNEL>= 6) vrsum45 = _vel_pvfmad_vvsvl(vrsum45, kerValue45, VRIN, vl) ;	\
	if(NUMKERNEL>= 8) vrsum67 = _vel_pvfmad_vvsvl(vrsum67, kerValue67, VRIN, vl) ;	\
	const uint64_t kerValue89 = _vel_pack_f32p(pKernel+FILTER_OFFSET(k+ 8,c,R,S),	\
						   pKernel+FILTER_OFFSET(k+ 9,c,R,S)) ;	\
	const uint64_t kerValueAB = _vel_pack_f32p(pKernel+FILTER_OFFSET(k+10,c,R,S),	\
						   pKernel+FILTER_OFFSET(k+11,c,R,S)) ;	\
	if(NUMKERNEL>=10) vrsum89 = _vel_pvfmad_vvsvl(vrsum89, kerValue89, VRIN, vl) ;	\
	if(NUMKERNEL>=12) vrsumAB = _vel_pvfmad_vvsvl(vrsumAB, kerValueAB, VRIN, vl) ;	\
	const uint64_t kerValueCD = _vel_pack_f32p(pKernel+FILTER_OFFSET(k+12,c,R,S),	\
						   pKernel+FILTER_OFFSET(k+13,c,R,S)) ;	\
	const uint64_t kerValueEF = _vel_pack_f32p(pKernel+FILTER_OFFSET(k+14,c,R,S),	\
						   pKernel+FILTER_OFFSET(k+15,c,R,S)) ;	\
	if(NUMKERNEL>=14) vrsumCD = _vel_pvfmad_vvsvl(vrsumCD, kerValueCD, VRIN, vl) ;	\
	if(NUMKERNEL>=16) vrsumEF = _vel_pvfmad_vvsvl(vrsumEF, kerValueEF, VRIN, vl) ;	\
      }

      VFMAD(vrinP_r0s0, 0, 0) ;
      VFMAD(vrinP_r0s1, 0, 1) ;
      VFMAD(vrinP_r0s2, 0, 2) ;
      VFMAD(vrinP_r1s0, 1, 0) ;
      VFMAD(vrinP_r1s1, 1, 1) ;
      VFMAD(vrinP_r1s2, 1, 2) ;
      VFMAD(vrinP_r2s0, 2, 0) ;
      VFMAD(vrinP_r2s1, 2, 1) ;
      VFMAD(vrinP_r2s2, 2, 2) ;

#undef VFMAD
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
    const int64_t outChannelGroup,
    const int64_t padHeight,
    const int64_t padWidth
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
	     padHeight, padWidth,
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
	     padHeight, padWidth,
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
	     padHeight, padWidth,
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
	     padHeight, padWidth,
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
	     padHeight, padWidth,
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
	     padHeight, padWidth,
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
	     padHeight, padWidth,
	     inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	     n, k );
	  k+=7 ;
	  break ;
	case 8 :
	  func_even<FLAYOUT, 8, ADDBIAS>(pIn, pKernel, pBias, pOut,
	     inChannel, inWidth, inHeight,
	     outChannel, outWidth, outHeight,
	     kernWidth, kernHeight,
	     inChannelGroup, outChannelGroup,
	     padHeight, padWidth,
	     inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	     n, k );
	  k+=8 ;
	  break ;
	case 9 :
	  func_odd<FLAYOUT, 9, ADDBIAS>(pIn, pKernel, pBias, pOut,
	     inChannel, inWidth, inHeight,
	     outChannel, outWidth, outHeight,
	     kernWidth, kernHeight,
	     inChannelGroup, outChannelGroup,
	     padHeight, padWidth,
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
	     padHeight, padWidth,
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
	     padHeight, padWidth,
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
	     padHeight, padWidth,
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
	     padHeight, padWidth,
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
	     padHeight, padWidth,
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
	     padHeight, padWidth,
	     inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	     n, k );
	  k+=15 ;
	  break ;
	default :
	  break ;
	}
	for (; k < outChannelGroup; k+=16) {
	  func_even<FLAYOUT, 16, ADDBIAS>(pIn, pKernel, pBias, pOut,
	     inChannel, inWidth, inHeight,
	     outChannel, outWidth, outHeight,
	     kernWidth, kernHeight,
	     inChannelGroup, outChannelGroup,
	     padHeight, padWidth,
	     inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	     n, k );
	} // outChannel
    } // group
  } // batch
}

extern "C" vednnError_t
vednnConvolutionForward_direct_dil1_str1_padsame_ker3(
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
  const int64_t kernWidth  = pParamKernel->width;		/* must be 3 */
  const int64_t kernHeight = pParamKernel->height;		/* must be 3 */

  const int64_t filter_layout = pParamKernel->layout ;

  const int64_t group          = pParamConv->group;
//  const int64_t strideWidth    = pParamConv->strideWidth;	/* must be 1 */
//  const int64_t strideHeight   = pParamConv->strideHeight;	/* must be 1 */
  const int64_t padWidth       = pParamConv->padWidth;
  const int64_t padHeight      = pParamConv->padHeight;
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
		 inChannelGroup, outChannelGroup,
		 padHeight, padWidth ) ;
    }
    else {
      convloop<VEDNN_FILTER_LAYOUT_NCHW, true>(pIn, pKernel, pBias, pOut,
		 batch, group,
		 inChannel, inWidth, inHeight,
		 outChannel, outWidth, outHeight,
		 kernWidth, kernHeight,
		 inChannelGroup, outChannelGroup,
		 padHeight, padWidth ) ;
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
		 padHeight, padWidth ) ;
    }
    else {
      convloop<VEDNN_FILTER_LAYOUT_HWCN, true>(pIn, pKernel, pBias, pOut,
		 batch, group,
		 inChannel, inWidth, inHeight,
		 outChannel, outWidth, outHeight,
		 kernWidth, kernHeight,
		 inChannelGroup, outChannelGroup,
		 padHeight, padWidth ) ;
    }
  }

  return VEDNN_SUCCESS;
}

