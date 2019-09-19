#include <stdint.h>
#include <stdlib.h>

#include "vednn.h"
#include "vednn_util.h"

#include "velintrin.h"
#define VLEN	(256)

#include <stdio.h>


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
  const int64_t outChannelGroup,
  const int64_t padHeight,
  const int64_t padWidth,
  const int64_t inGroupOffset,
  const int64_t outGroupOffset,
  const int64_t biasGroupOffset,
  const int64_t kernGroupOffset,
  const int64_t nY,
  const __vr    vry,
  const __vm256 vmw_s0,
  const __vm256 vmw_s2,
  const int64_t n,
  const int64_t k
)
{
  const float   bias0  = ADDBIAS ?  pBias[biasGroupOffset+k+ 0] : 0.f ;
  const int64_t bias12 = ADDBIAS ?  _vel_pack_f32p(pBias+biasGroupOffset+k+ 1, pBias+biasGroupOffset+k+ 2) : 0UL ;
  const int64_t bias34 = ADDBIAS ?  _vel_pack_f32p(pBias+biasGroupOffset+k+ 3, pBias+biasGroupOffset+k+ 4) : 0UL ;
  const int64_t bias56 = ADDBIAS ?  _vel_pack_f32p(pBias+biasGroupOffset+k+ 5, pBias+biasGroupOffset+k+ 6) : 0UL ;
  const int64_t bias78 = ADDBIAS ?  _vel_pack_f32p(pBias+biasGroupOffset+k+ 7, pBias+biasGroupOffset+k+ 8) : 0UL ;
  const int64_t bias9A = ADDBIAS ?  _vel_pack_f32p(pBias+biasGroupOffset+k+ 9, pBias+biasGroupOffset+k+10) : 0UL ;
  const int64_t biasBC = ADDBIAS ?  _vel_pack_f32p(pBias+biasGroupOffset+k+11, pBias+biasGroupOffset+k+12) : 0UL ;
  const int64_t biasDE = ADDBIAS ?  _vel_pack_f32p(pBias+biasGroupOffset+k+13, pBias+biasGroupOffset+k+14) : 0UL ;

  const __vr vrzerof = _vel_vbrds_vsl(0.f, VLEN) ;

  for (int64_t y=0; y<outHeight; y+=nY)
  {
    const int64_t vl = outWidth * (outHeight - y < nY ? outHeight - y : nY) ;
    const int64_t op = y * outWidth ;

    __vr vrh_r0 = _vel_vaddsl_vsvl(y  -1, vry, vl) ;
    __vr vrh_r2 = _vel_vaddsl_vsvl(y+2-1, vry, vl) ;

    __vm256 vmh_r0 =  _vel_vfmklge_mvl(vrh_r0, vl) ;
    __vm256 vmh_r2 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(inHeight,vrh_r2, vl), vl) ;

    __vm256 vmhw_r0s0 = _vel_andm_mmm(vmh_r0,vmw_s0) ;
    __vm256 vmhw_r0s1 = vmh_r0 ;
    __vm256 vmhw_r0s2 = _vel_andm_mmm(vmh_r0, vmw_s2) ;

    __vm256 vmhw_r1s0 = vmw_s0 ;
    __vm256 vmhw_r1s2 = vmw_s2 ;

    __vm256 vmhw_r2s0 = _vel_andm_mmm(vmh_r2,vmw_s0) ;
    __vm256 vmhw_r2s1 = vmh_r2 ;
    __vm256 vmhw_r2s2 = _vel_andm_mmm(vmh_r2, vmw_s2) ;

    int64_t outIndex  = outGroupOffset + (n * outChannel + k  ) * outHeight * outWidth + op;

    const float *pInChannel = pIn + inGroupOffset + ((n * inChannel) * inHeight * inWidth ) ;

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

    vrin_r0s0 = _vel_vmrg_vvvml(vrzerof, vrin_r0s0, vmhw_r0s0, vl) ;
    vrin_r0s1 = _vel_vmrg_vvvml(vrzerof, vrin_r0s1, vmhw_r0s1, vl) ;
    vrin_r0s2 = _vel_vmrg_vvvml(vrzerof, vrin_r0s2, vmhw_r0s2, vl) ;
    __vr vrinP_r0s0 = _vel_vshf_vvvsl(vrin_r0s0, vrin_r0s0, VE_VSHUFFLE_YUZU, vl) ;
    __vr vrinP_r0s1 = _vel_vshf_vvvsl(vrin_r0s1, vrin_r0s1, VE_VSHUFFLE_YUZU, vl) ;
    __vr vrinP_r0s2 = _vel_vshf_vvvsl(vrin_r0s2, vrin_r0s2, VE_VSHUFFLE_YUZU, vl) ;

    vrin_r1s0 = _vel_vmrg_vvvml(vrzerof, vrin_r1s0, vmhw_r1s0, vl) ;
    vrin_r1s2 = _vel_vmrg_vvvml(vrzerof, vrin_r1s2, vmhw_r1s2, vl) ;
    __vr vrinP_r1s0 = _vel_vshf_vvvsl(vrin_r1s0, vrin_r1s0, VE_VSHUFFLE_YUZU, vl) ;
    __vr vrinP_r1s1 = _vel_vshf_vvvsl(vrin_r1s1, vrin_r1s1, VE_VSHUFFLE_YUZU, vl) ;
    __vr vrinP_r1s2 = _vel_vshf_vvvsl(vrin_r1s2, vrin_r1s2, VE_VSHUFFLE_YUZU, vl) ;

    vrin_r2s0 = _vel_vmrg_vvvml(vrzerof, vrin_r2s0, vmhw_r2s0, vl) ;
    vrin_r2s1 = _vel_vmrg_vvvml(vrzerof, vrin_r2s1, vmhw_r2s1, vl) ;
    vrin_r2s2 = _vel_vmrg_vvvml(vrzerof, vrin_r2s2, vmhw_r2s2, vl) ;
    __vr vrinP_r2s0 = _vel_vshf_vvvsl(vrin_r2s0, vrin_r2s0, VE_VSHUFFLE_YUZU, vl) ;
    __vr vrinP_r2s1 = _vel_vshf_vvvsl(vrin_r2s1, vrin_r2s1, VE_VSHUFFLE_YUZU, vl) ;
    __vr vrinP_r2s2 = _vel_vshf_vvvsl(vrin_r2s2, vrin_r2s2, VE_VSHUFFLE_YUZU, vl) ;

#define FILTER_OFFSET(k,c,r,s) ( kernGroupOffset + filter_index<FLAYOUT>(k,c,r,s, 1, outChannelGroup, kernHeight, kernWidth) )

    {
      __vr vrsum = _vel_vbrds_vsl(bias0, vl) ;
      vrsum = _vel_vfmads_vvsvl(vrsum, pKernel[FILTER_OFFSET(k+0,0,0,0)], vrin_r0s0, vl) ;
      vrsum = _vel_vfmads_vvsvl(vrsum, pKernel[FILTER_OFFSET(k+0,0,0,1)], vrin_r0s1, vl) ;
      vrsum = _vel_vfmads_vvsvl(vrsum, pKernel[FILTER_OFFSET(k+0,0,0,2)], vrin_r0s2, vl) ;
      vrsum = _vel_vfmads_vvsvl(vrsum, pKernel[FILTER_OFFSET(k+0,0,1,0)], vrin_r1s0, vl) ;
      vrsum = _vel_vfmads_vvsvl(vrsum, pKernel[FILTER_OFFSET(k+0,0,1,1)], vrin_r1s1, vl) ;
      vrsum = _vel_vfmads_vvsvl(vrsum, pKernel[FILTER_OFFSET(k+0,0,1,2)], vrin_r1s2, vl) ;
      vrsum = _vel_vfmads_vvsvl(vrsum, pKernel[FILTER_OFFSET(k+0,0,2,0)], vrin_r2s0, vl) ;
      vrsum = _vel_vfmads_vvsvl(vrsum, pKernel[FILTER_OFFSET(k+0,0,2,1)], vrin_r2s1, vl) ;
      vrsum = _vel_vfmads_vvsvl(vrsum, pKernel[FILTER_OFFSET(k+0,0,2,2)], vrin_r2s2, vl) ;
      _vel_vstu_vssl(vrsum, 4, pOut+outIndex+ (0) * outHeight * outWidth, vl) ;
    }
#define FILTER_R3S3(BIAS, K)									\
    {												\
      __vr vrsum = _vel_pvbrd_vsl(BIAS, vl) ;							\
      const uint64_t kerValue_r0s0 = _vel_pack_f32p(pKernel + FILTER_OFFSET(k+K  ,0,0,0),	\
						    pKernel + FILTER_OFFSET(k+K+1,0,0,0));	\
      const uint64_t kerValue_r0s1 = _vel_pack_f32p(pKernel + FILTER_OFFSET(k+K  ,0,0,1),	\
						    pKernel + FILTER_OFFSET(k+K+1,0,0,1));	\
      const uint64_t kerValue_r0s2 = _vel_pack_f32p(pKernel + FILTER_OFFSET(k+K  ,0,0,2),	\
						    pKernel + FILTER_OFFSET(k+K+1,0,0,2));	\
      const uint64_t kerValue_r1s0 = _vel_pack_f32p(pKernel + FILTER_OFFSET(k+K  ,0,1,0),	\
						    pKernel + FILTER_OFFSET(k+K+1,0,1,0));	\
      const uint64_t kerValue_r1s1 = _vel_pack_f32p(pKernel + FILTER_OFFSET(k+K  ,0,1,1),	\
						    pKernel + FILTER_OFFSET(k+K+1,0,1,1));	\
      const uint64_t kerValue_r1s2 = _vel_pack_f32p(pKernel + FILTER_OFFSET(k+K  ,0,1,2),	\
						    pKernel + FILTER_OFFSET(k+K+1,0,1,2));	\
      const uint64_t kerValue_r2s0 = _vel_pack_f32p(pKernel + FILTER_OFFSET(k+K  ,0,2,0),	\
						    pKernel + FILTER_OFFSET(k+K+1,0,2,0));	\
      const uint64_t kerValue_r2s1 = _vel_pack_f32p(pKernel + FILTER_OFFSET(k+K  ,0,2,1),	\
						    pKernel + FILTER_OFFSET(k+K+1,0,2,1));	\
      const uint64_t kerValue_r2s2 = _vel_pack_f32p(pKernel + FILTER_OFFSET(k+K  ,0,2,2),	\
						    pKernel + FILTER_OFFSET(k+K+1,0,2,2));	\
      vrsum = _vel_pvfmad_vvsvl(vrsum, kerValue_r0s0, vrinP_r0s0, vl) ;	\
      vrsum = _vel_pvfmad_vvsvl(vrsum, kerValue_r0s1, vrinP_r0s1, vl) ;	\
      vrsum = _vel_pvfmad_vvsvl(vrsum, kerValue_r0s2, vrinP_r0s2, vl) ;	\
      vrsum = _vel_pvfmad_vvsvl(vrsum, kerValue_r1s0, vrinP_r1s0, vl) ;	\
      vrsum = _vel_pvfmad_vvsvl(vrsum, kerValue_r1s1, vrinP_r1s1, vl) ;	\
      vrsum = _vel_pvfmad_vvsvl(vrsum, kerValue_r1s2, vrinP_r1s2, vl) ;	\
      vrsum = _vel_pvfmad_vvsvl(vrsum, kerValue_r2s0, vrinP_r2s0, vl) ;	\
      vrsum = _vel_pvfmad_vvsvl(vrsum, kerValue_r2s1, vrinP_r2s1, vl) ;	\
      vrsum = _vel_pvfmad_vvsvl(vrsum, kerValue_r2s2, vrinP_r2s2, vl) ;	\
      _vel_vstu_vssl(vrsum, 4, pOut+outIndex+ (K)    * outHeight * outWidth, vl) ;	\
      _vel_vstl_vssl(vrsum, 4, pOut+outIndex+ ((K)+1)* outHeight * outWidth, vl) ;	\
    }
    if(NUMKERNEL>= 3) FILTER_R3S3(bias12, 1) ;
    if(NUMKERNEL>= 5) FILTER_R3S3(bias34, 3) ;
    if(NUMKERNEL>= 7) FILTER_R3S3(bias56, 5) ;
    if(NUMKERNEL>= 9) FILTER_R3S3(bias78, 7) ;
    if(NUMKERNEL>=11) FILTER_R3S3(bias9A, 9) ;
    if(NUMKERNEL>=13) FILTER_R3S3(biasBC, 11) ;
    if(NUMKERNEL>=15) FILTER_R3S3(biasDE, 13) ;
#undef FILTER_R3S3
#undef FILTER_OFFSET
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
  const int64_t outChannelGroup,
  const int64_t padHeight,
  const int64_t padWidth,
  const int64_t inGroupOffset,
  const int64_t outGroupOffset,
  const int64_t biasGroupOffset,
  const int64_t kernGroupOffset,
  const int64_t nY,
  const __vr    vry,
  const __vm256 vmw_s0,
  const __vm256 vmw_s2,
  const int64_t n,
  const int64_t k
)
{
  const int64_t bias01 = ADDBIAS ?  _vel_pack_f32p(pBias+biasGroupOffset+k+ 0, pBias+biasGroupOffset+k+ 1) : 0UL ;
  const int64_t bias23 = ADDBIAS ?  _vel_pack_f32p(pBias+biasGroupOffset+k+ 2, pBias+biasGroupOffset+k+ 3) : 0UL ;
  const int64_t bias45 = ADDBIAS ?  _vel_pack_f32p(pBias+biasGroupOffset+k+ 4, pBias+biasGroupOffset+k+ 5) : 0UL ;
  const int64_t bias67 = ADDBIAS ?  _vel_pack_f32p(pBias+biasGroupOffset+k+ 6, pBias+biasGroupOffset+k+ 7) : 0UL ;
  const int64_t bias89 = ADDBIAS ?  _vel_pack_f32p(pBias+biasGroupOffset+k+ 8, pBias+biasGroupOffset+k+ 9) : 0UL ;
  const int64_t biasAB = ADDBIAS ?  _vel_pack_f32p(pBias+biasGroupOffset+k+10, pBias+biasGroupOffset+k+11) : 0UL ;
  const int64_t biasCD = ADDBIAS ?  _vel_pack_f32p(pBias+biasGroupOffset+k+12, pBias+biasGroupOffset+k+13) : 0UL ;
  const int64_t biasEF = ADDBIAS ?  _vel_pack_f32p(pBias+biasGroupOffset+k+14, pBias+biasGroupOffset+k+15) : 0UL ;

  const __vr vrzerof = _vel_vbrds_vsl(0.f, VLEN) ;

  for (int64_t y=0; y<outHeight; y+=nY)
  {
    const int64_t vl = outWidth * (outHeight - y < nY ? outHeight - y : nY) ;
    const int64_t op = y * outWidth ;

    __vr vrh_r0 = _vel_vaddsl_vsvl(y  -1, vry, vl) ;
    __vr vrh_r2 = _vel_vaddsl_vsvl(y+2-1, vry, vl) ;

    __vm256 vmh_r0 =  _vel_vfmklge_mvl(vrh_r0, vl) ;
    __vm256 vmh_r2 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(inHeight,vrh_r2, vl), vl) ;

    __vm256 vmhw_r0s0 = _vel_andm_mmm(vmh_r0,vmw_s0) ;
    __vm256 vmhw_r0s1 = vmh_r0 ;
    __vm256 vmhw_r0s2 = _vel_andm_mmm(vmh_r0, vmw_s2) ;

    __vm256 vmhw_r1s0 = vmw_s0 ;
    __vm256 vmhw_r1s2 = vmw_s2 ;

    __vm256 vmhw_r2s0 = _vel_andm_mmm(vmh_r2,vmw_s0) ;
    __vm256 vmhw_r2s1 = vmh_r2 ;
    __vm256 vmhw_r2s2 = _vel_andm_mmm(vmh_r2, vmw_s2) ;

    int64_t outIndex  = outGroupOffset + (n * outChannel + k  ) * outHeight * outWidth + op;

    const float *pInChannel = pIn + inGroupOffset + ((n * inChannel) * inHeight * inWidth ) ;

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

    vrin_r0s0 = _vel_vmrg_vvvml(vrzerof, vrin_r0s0, vmhw_r0s0, vl) ;
    vrin_r0s1 = _vel_vmrg_vvvml(vrzerof, vrin_r0s1, vmhw_r0s1, vl) ;
    vrin_r0s2 = _vel_vmrg_vvvml(vrzerof, vrin_r0s2, vmhw_r0s2, vl) ;
    __vr vrinP_r0s0 = _vel_vshf_vvvsl(vrin_r0s0, vrin_r0s0, VE_VSHUFFLE_YUZU, vl) ;
    __vr vrinP_r0s1 = _vel_vshf_vvvsl(vrin_r0s1, vrin_r0s1, VE_VSHUFFLE_YUZU, vl) ;
    __vr vrinP_r0s2 = _vel_vshf_vvvsl(vrin_r0s2, vrin_r0s2, VE_VSHUFFLE_YUZU, vl) ;

    vrin_r1s0 = _vel_vmrg_vvvml(vrzerof, vrin_r1s0, vmhw_r1s0, vl) ;
    vrin_r1s2 = _vel_vmrg_vvvml(vrzerof, vrin_r1s2, vmhw_r1s2, vl) ;
    __vr vrinP_r1s0 = _vel_vshf_vvvsl(vrin_r1s0, vrin_r1s0, VE_VSHUFFLE_YUZU, vl) ;
    __vr vrinP_r1s1 = _vel_vshf_vvvsl(vrin_r1s1, vrin_r1s1, VE_VSHUFFLE_YUZU, vl) ;
    __vr vrinP_r1s2 = _vel_vshf_vvvsl(vrin_r1s2, vrin_r1s2, VE_VSHUFFLE_YUZU, vl) ;

    vrin_r2s0 = _vel_vmrg_vvvml(vrzerof, vrin_r2s0, vmhw_r2s0, vl) ;
    vrin_r2s1 = _vel_vmrg_vvvml(vrzerof, vrin_r2s1, vmhw_r2s1, vl) ;
    vrin_r2s2 = _vel_vmrg_vvvml(vrzerof, vrin_r2s2, vmhw_r2s2, vl) ;
    __vr vrinP_r2s0 = _vel_vshf_vvvsl(vrin_r2s0, vrin_r2s0, VE_VSHUFFLE_YUZU, vl) ;
    __vr vrinP_r2s1 = _vel_vshf_vvvsl(vrin_r2s1, vrin_r2s1, VE_VSHUFFLE_YUZU, vl) ;
    __vr vrinP_r2s2 = _vel_vshf_vvvsl(vrin_r2s2, vrin_r2s2, VE_VSHUFFLE_YUZU, vl) ;

#define FILTER_OFFSET(k,c,r,s) ( kernGroupOffset + filter_index<FLAYOUT>(k,c,r,s, 1, outChannelGroup, kernHeight, kernWidth) )
#define FILTER_R3S3(BIAS, K)									\
    {												\
      __vr vrsum = _vel_pvbrd_vsl(BIAS, vl) ;							\
      const uint64_t kerValue_r0s0 = _vel_pack_f32p(pKernel + FILTER_OFFSET(k+K  ,0,0,0),	\
						    pKernel + FILTER_OFFSET(k+K+1,0,0,0));	\
      const uint64_t kerValue_r0s1 = _vel_pack_f32p(pKernel + FILTER_OFFSET(k+K  ,0,0,1),	\
						    pKernel + FILTER_OFFSET(k+K+1,0,0,1));	\
      const uint64_t kerValue_r0s2 = _vel_pack_f32p(pKernel + FILTER_OFFSET(k+K  ,0,0,2),	\
						    pKernel + FILTER_OFFSET(k+K+1,0,0,2));	\
      const uint64_t kerValue_r1s0 = _vel_pack_f32p(pKernel + FILTER_OFFSET(k+K  ,0,1,0),	\
						    pKernel + FILTER_OFFSET(k+K+1,0,1,0));	\
      const uint64_t kerValue_r1s1 = _vel_pack_f32p(pKernel + FILTER_OFFSET(k+K  ,0,1,1),	\
						    pKernel + FILTER_OFFSET(k+K+1,0,1,1));	\
      const uint64_t kerValue_r1s2 = _vel_pack_f32p(pKernel + FILTER_OFFSET(k+K  ,0,1,2),	\
						    pKernel + FILTER_OFFSET(k+K+1,0,1,2));	\
      const uint64_t kerValue_r2s0 = _vel_pack_f32p(pKernel + FILTER_OFFSET(k+K  ,0,2,0),	\
						    pKernel + FILTER_OFFSET(k+K+1,0,2,0));	\
      const uint64_t kerValue_r2s1 = _vel_pack_f32p(pKernel + FILTER_OFFSET(k+K  ,0,2,1),	\
						    pKernel + FILTER_OFFSET(k+K+1,0,2,1));	\
      const uint64_t kerValue_r2s2 = _vel_pack_f32p(pKernel + FILTER_OFFSET(k+K  ,0,2,2),	\
						    pKernel + FILTER_OFFSET(k+K+1,0,2,2));	\
      vrsum = _vel_pvfmad_vvsvl(vrsum, kerValue_r0s0, vrinP_r0s0, vl) ;	\
      vrsum = _vel_pvfmad_vvsvl(vrsum, kerValue_r0s1, vrinP_r0s1, vl) ;	\
      vrsum = _vel_pvfmad_vvsvl(vrsum, kerValue_r0s2, vrinP_r0s2, vl) ;	\
      vrsum = _vel_pvfmad_vvsvl(vrsum, kerValue_r1s0, vrinP_r1s0, vl) ;	\
      vrsum = _vel_pvfmad_vvsvl(vrsum, kerValue_r1s1, vrinP_r1s1, vl) ;	\
      vrsum = _vel_pvfmad_vvsvl(vrsum, kerValue_r1s2, vrinP_r1s2, vl) ;	\
      vrsum = _vel_pvfmad_vvsvl(vrsum, kerValue_r2s0, vrinP_r2s0, vl) ;	\
      vrsum = _vel_pvfmad_vvsvl(vrsum, kerValue_r2s1, vrinP_r2s1, vl) ;	\
      vrsum = _vel_pvfmad_vvsvl(vrsum, kerValue_r2s2, vrinP_r2s2, vl) ;	\
      _vel_vstu_vssl(vrsum, 4, pOut+outIndex+ (K)    * outHeight * outWidth, vl) ;	\
      _vel_vstl_vssl(vrsum, 4, pOut+outIndex+ ((K)+1)* outHeight * outWidth, vl) ;	\
    }
    if(NUMKERNEL>= 2) FILTER_R3S3(bias01, 0) ;
    if(NUMKERNEL>= 4) FILTER_R3S3(bias23, 2) ;
    if(NUMKERNEL>= 6) FILTER_R3S3(bias45, 4) ;
    if(NUMKERNEL>= 8) FILTER_R3S3(bias67, 6) ;
    if(NUMKERNEL>=10) FILTER_R3S3(bias89, 8) ;
    if(NUMKERNEL>=12) FILTER_R3S3(biasAB, 10) ;
    if(NUMKERNEL>=14) FILTER_R3S3(biasCD, 12) ;
    if(NUMKERNEL>=16) FILTER_R3S3(biasEF, 14) ;
#undef FILTER_R3S3
#undef FILTER_OFFSET
  } // outPixels
}

template<int NUMKERNEL, bool ADDBIAS>
static inline void func_odd_filternchw_opt(
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
  const int64_t outChannelGroup,
  const int64_t padHeight,
  const int64_t padWidth,
  const int64_t inGroupOffset,
  const int64_t outGroupOffset,
  const int64_t biasGroupOffset,
  const int64_t kernGroupOffset,
  const int64_t nY,
  const __vr    vry,
  const __vm256 vmw_s0,
  const __vm256 vmw_s2,
  const int64_t n,
  const int64_t k
)
{
  const float   bias0 = ADDBIAS ?  pBias[biasGroupOffset+k+ 0] : 0.f ;
  const int64_t bias1 = ADDBIAS ?  _vel_pack_f32p(pBias+biasGroupOffset+k+1, pBias+biasGroupOffset+k+1+(NUMKERNEL/2)) : 0UL ;
  const int64_t bias2 = ADDBIAS ?  _vel_pack_f32p(pBias+biasGroupOffset+k+2, pBias+biasGroupOffset+k+2+(NUMKERNEL/2)) : 0UL ;
  const int64_t bias3 = ADDBIAS ?  _vel_pack_f32p(pBias+biasGroupOffset+k+3, pBias+biasGroupOffset+k+3+(NUMKERNEL/2)) : 0UL ;
  const int64_t bias4 = ADDBIAS ?  _vel_pack_f32p(pBias+biasGroupOffset+k+4, pBias+biasGroupOffset+k+4+(NUMKERNEL/2)) : 0UL ;
  const int64_t bias5 = ADDBIAS ?  _vel_pack_f32p(pBias+biasGroupOffset+k+5, pBias+biasGroupOffset+k+5+(NUMKERNEL/2)) : 0UL ;
  const int64_t bias6 = ADDBIAS ?  _vel_pack_f32p(pBias+biasGroupOffset+k+6, pBias+biasGroupOffset+k+6+(NUMKERNEL/2)) : 0UL ;
  const int64_t bias7 = ADDBIAS ?  _vel_pack_f32p(pBias+biasGroupOffset+k+7, pBias+biasGroupOffset+k+7+(NUMKERNEL/2)) : 0UL ;

  const __vr vrzerof = _vel_vbrds_vsl(0.f, VLEN) ;

  uint64_t kerValue[(NUMKERNEL/2)*9] ;
  {
    const float *pKerValue  = pKernel + kernGroupOffset + (k * kernHeight    ) * kernWidth ;
    __vr vrker0 = _vel_vldu_vssl(4, pKerValue+9                , (NUMKERNEL/2)*9) ;
    __vr vrker1 = _vel_vldu_vssl(4, pKerValue+9+(NUMKERNEL/2)*9, (NUMKERNEL/2)*9) ;
    _vel_vst_vssl(_vel_vshf_vvvsl(vrker0, vrker1, VE_VSHUFFLE_YUZU, (NUMKERNEL/2)*9) ,8, kerValue, (NUMKERNEL/2)*9) ;
  }

  for (int64_t y=0; y<outHeight; y+=nY)
  {
    const int64_t vl = outWidth * (outHeight - y < nY ? outHeight - y : nY) ;
    const int64_t op = y * outWidth ;

    __vr vrh_r0 = _vel_vaddsl_vsvl(y  -1, vry, vl) ;
    __vr vrh_r2 = _vel_vaddsl_vsvl(y+2-1, vry, vl) ;

    __vm256 vmh_r0 =  _vel_vfmklge_mvl(vrh_r0, vl) ;
    __vm256 vmh_r2 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(inHeight,vrh_r2, vl), vl) ;

    __vm256 vmhw_r0s0 = _vel_andm_mmm(vmh_r0,vmw_s0) ;
    __vm256 vmhw_r0s1 = vmh_r0 ;
    __vm256 vmhw_r0s2 = _vel_andm_mmm(vmh_r0, vmw_s2) ;

    __vm256 vmhw_r1s0 = vmw_s0 ;
    __vm256 vmhw_r1s2 = vmw_s2 ;

    __vm256 vmhw_r2s0 = _vel_andm_mmm(vmh_r2,vmw_s0) ;
    __vm256 vmhw_r2s1 = vmh_r2 ;
    __vm256 vmhw_r2s2 = _vel_andm_mmm(vmh_r2, vmw_s2) ;

    int64_t outIndex  = outGroupOffset + (n * outChannel + k  ) * outHeight * outWidth + op;

    const float *pInChannel = pIn + inGroupOffset + ((n * inChannel) * inHeight * inWidth ) ;

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

    vrin_r0s0 = _vel_vmrg_vvvml(vrzerof, vrin_r0s0, vmhw_r0s0, vl) ;
    vrin_r0s1 = _vel_vmrg_vvvml(vrzerof, vrin_r0s1, vmhw_r0s1, vl) ;
    vrin_r0s2 = _vel_vmrg_vvvml(vrzerof, vrin_r0s2, vmhw_r0s2, vl) ;
    __vr vrinP_r0s0 = _vel_vshf_vvvsl(vrin_r0s0, vrin_r0s0, VE_VSHUFFLE_YUZU, vl) ;
    __vr vrinP_r0s1 = _vel_vshf_vvvsl(vrin_r0s1, vrin_r0s1, VE_VSHUFFLE_YUZU, vl) ;
    __vr vrinP_r0s2 = _vel_vshf_vvvsl(vrin_r0s2, vrin_r0s2, VE_VSHUFFLE_YUZU, vl) ;

    vrin_r1s0 = _vel_vmrg_vvvml(vrzerof, vrin_r1s0, vmhw_r1s0, vl) ;
    vrin_r1s2 = _vel_vmrg_vvvml(vrzerof, vrin_r1s2, vmhw_r1s2, vl) ;
    __vr vrinP_r1s0 = _vel_vshf_vvvsl(vrin_r1s0, vrin_r1s0, VE_VSHUFFLE_YUZU, vl) ;
    __vr vrinP_r1s1 = _vel_vshf_vvvsl(vrin_r1s1, vrin_r1s1, VE_VSHUFFLE_YUZU, vl) ;
    __vr vrinP_r1s2 = _vel_vshf_vvvsl(vrin_r1s2, vrin_r1s2, VE_VSHUFFLE_YUZU, vl) ;

    vrin_r2s0 = _vel_vmrg_vvvml(vrzerof, vrin_r2s0, vmhw_r2s0, vl) ;
    vrin_r2s1 = _vel_vmrg_vvvml(vrzerof, vrin_r2s1, vmhw_r2s1, vl) ;
    vrin_r2s2 = _vel_vmrg_vvvml(vrzerof, vrin_r2s2, vmhw_r2s2, vl) ;
    __vr vrinP_r2s0 = _vel_vshf_vvvsl(vrin_r2s0, vrin_r2s0, VE_VSHUFFLE_YUZU, vl) ;
    __vr vrinP_r2s1 = _vel_vshf_vvvsl(vrin_r2s1, vrin_r2s1, VE_VSHUFFLE_YUZU, vl) ;
    __vr vrinP_r2s2 = _vel_vshf_vvvsl(vrin_r2s2, vrin_r2s2, VE_VSHUFFLE_YUZU, vl) ;

#define FILTER_OFFSET(k,c,r,s) ( kernGroupOffset + filter_index<VEDNN_FILTER_LAYOUT_NCHW>(k,c,r,s, 1, outChannelGroup, kernHeight, kernWidth) )
    {
      __vr vrsum = _vel_vbrds_vsl(bias0, vl) ;
      vrsum = _vel_vfmads_vvsvl(vrsum, pKernel[FILTER_OFFSET(k+0,0,0,0)], vrin_r0s0, vl) ;
      vrsum = _vel_vfmads_vvsvl(vrsum, pKernel[FILTER_OFFSET(k+0,0,0,1)], vrin_r0s1, vl) ;
      vrsum = _vel_vfmads_vvsvl(vrsum, pKernel[FILTER_OFFSET(k+0,0,0,2)], vrin_r0s2, vl) ;
      vrsum = _vel_vfmads_vvsvl(vrsum, pKernel[FILTER_OFFSET(k+0,0,1,0)], vrin_r1s0, vl) ;
      vrsum = _vel_vfmads_vvsvl(vrsum, pKernel[FILTER_OFFSET(k+0,0,1,1)], vrin_r1s1, vl) ;
      vrsum = _vel_vfmads_vvsvl(vrsum, pKernel[FILTER_OFFSET(k+0,0,1,2)], vrin_r1s2, vl) ;
      vrsum = _vel_vfmads_vvsvl(vrsum, pKernel[FILTER_OFFSET(k+0,0,2,0)], vrin_r2s0, vl) ;
      vrsum = _vel_vfmads_vvsvl(vrsum, pKernel[FILTER_OFFSET(k+0,0,2,1)], vrin_r2s1, vl) ;
      vrsum = _vel_vfmads_vvsvl(vrsum, pKernel[FILTER_OFFSET(k+0,0,2,2)], vrin_r2s2, vl) ;
      _vel_vstu_vssl(vrsum, 4, pOut+outIndex+ (0) * outHeight * outWidth, vl) ;
    }

#define FILTER_R3S3(BIAS, N)							\
    {										\
      __vr vrsum = _vel_pvbrd_vsl(BIAS, vl) ;					\
      vrsum = _vel_pvfmad_vvsvl(vrsum, kerValue[9*(N-1)+0], vrinP_r0s0, vl) ;	\
      vrsum = _vel_pvfmad_vvsvl(vrsum, kerValue[9*(N-1)+1], vrinP_r0s1, vl) ;	\
      vrsum = _vel_pvfmad_vvsvl(vrsum, kerValue[9*(N-1)+2], vrinP_r0s2, vl) ;	\
      vrsum = _vel_pvfmad_vvsvl(vrsum, kerValue[9*(N-1)+3], vrinP_r1s0, vl) ;	\
      vrsum = _vel_pvfmad_vvsvl(vrsum, kerValue[9*(N-1)+4], vrinP_r1s1, vl) ;	\
      vrsum = _vel_pvfmad_vvsvl(vrsum, kerValue[9*(N-1)+5], vrinP_r1s2, vl) ;	\
      vrsum = _vel_pvfmad_vvsvl(vrsum, kerValue[9*(N-1)+6], vrinP_r2s0, vl) ;	\
      vrsum = _vel_pvfmad_vvsvl(vrsum, kerValue[9*(N-1)+7], vrinP_r2s1, vl) ;	\
      vrsum = _vel_pvfmad_vvsvl(vrsum, kerValue[9*(N-1)+8], vrinP_r2s2, vl) ;	\
      _vel_vstu_vssl(vrsum, 4, pOut+outIndex+  (N)                * outHeight * outWidth, vl) ;	\
      _vel_vstl_vssl(vrsum, 4, pOut+outIndex+ ((N)+(NUMKERNEL/2)) * outHeight * outWidth, vl) ;	\
    }
    if(NUMKERNEL>= 3) FILTER_R3S3(bias1, 1) ;
    if(NUMKERNEL>= 5) FILTER_R3S3(bias2, 2) ;
    if(NUMKERNEL>= 7) FILTER_R3S3(bias3, 3) ;
    if(NUMKERNEL>= 9) FILTER_R3S3(bias4, 4) ;
    if(NUMKERNEL>=11) FILTER_R3S3(bias5, 5) ;
    if(NUMKERNEL>=13) FILTER_R3S3(bias6, 6) ;
    if(NUMKERNEL>=15) FILTER_R3S3(bias7, 7) ;
#undef FILTER_R3S3
#undef FILTER_OFFSET
  } // outPixels
}


template<int NUMKERNEL, bool ADDBIAS>
static inline void func_even_filternchw_opt(
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
  const int64_t outChannelGroup,
  const int64_t padHeight,
  const int64_t padWidth,
  const int64_t inGroupOffset,
  const int64_t outGroupOffset,
  const int64_t biasGroupOffset,
  const int64_t kernGroupOffset,
  const int64_t nY,
  const __vr    vry,
  const __vm256 vmw_s0,
  const __vm256 vmw_s2,
  const int64_t n,
  const int64_t k
)
{
  const int64_t bias0 = ADDBIAS ?  _vel_pack_f32p(pBias+biasGroupOffset+k+0, pBias+biasGroupOffset+k+0+(NUMKERNEL/2)) : 0UL ;
  const int64_t bias1 = ADDBIAS ?  _vel_pack_f32p(pBias+biasGroupOffset+k+1, pBias+biasGroupOffset+k+1+(NUMKERNEL/2)) : 0UL ;
  const int64_t bias2 = ADDBIAS ?  _vel_pack_f32p(pBias+biasGroupOffset+k+2, pBias+biasGroupOffset+k+2+(NUMKERNEL/2)) : 0UL ;
  const int64_t bias3 = ADDBIAS ?  _vel_pack_f32p(pBias+biasGroupOffset+k+3, pBias+biasGroupOffset+k+3+(NUMKERNEL/2)) : 0UL ;
  const int64_t bias4 = ADDBIAS ?  _vel_pack_f32p(pBias+biasGroupOffset+k+4, pBias+biasGroupOffset+k+4+(NUMKERNEL/2)) : 0UL ;
  const int64_t bias5 = ADDBIAS ?  _vel_pack_f32p(pBias+biasGroupOffset+k+5, pBias+biasGroupOffset+k+5+(NUMKERNEL/2)) : 0UL ;
  const int64_t bias6 = ADDBIAS ?  _vel_pack_f32p(pBias+biasGroupOffset+k+6, pBias+biasGroupOffset+k+6+(NUMKERNEL/2)) : 0UL ;
  const int64_t bias7 = ADDBIAS ?  _vel_pack_f32p(pBias+biasGroupOffset+k+7, pBias+biasGroupOffset+k+7+(NUMKERNEL/2)) : 0UL ;

  const __vr vrzerof = _vel_vbrds_vsl(0.f, VLEN) ;

  uint64_t kerValue[(NUMKERNEL/2)*9] ;
  {
    const float *pKerValue  = pKernel + kernGroupOffset + (k * kernHeight    ) * kernWidth ;
    __vr vrker0 = _vel_vldu_vssl(4, pKerValue                , (NUMKERNEL/2)*9) ;
    __vr vrker1 = _vel_vldu_vssl(4, pKerValue+(NUMKERNEL/2)*9, (NUMKERNEL/2)*9) ;
    _vel_vst_vssl(_vel_vshf_vvvsl(vrker0, vrker1, VE_VSHUFFLE_YUZU, (NUMKERNEL/2)*9) ,8, kerValue, (NUMKERNEL/2)*9) ;
  }

  for (int64_t y=0; y<outHeight; y+=nY)
  {
    const int64_t vl = outWidth * (outHeight - y < nY ? outHeight - y : nY) ;
    const int64_t op = y * outWidth ;

    __vr vrh_r0 = _vel_vaddsl_vsvl(y  -1, vry, vl) ;
    __vr vrh_r2 = _vel_vaddsl_vsvl(y+2-1, vry, vl) ;

    __vm256 vmh_r0 =  _vel_vfmklge_mvl(vrh_r0, vl) ;
    __vm256 vmh_r2 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(inHeight,vrh_r2, vl), vl) ;

    __vm256 vmhw_r0s0 = _vel_andm_mmm(vmh_r0,vmw_s0) ;
    __vm256 vmhw_r0s1 = vmh_r0 ;
    __vm256 vmhw_r0s2 = _vel_andm_mmm(vmh_r0, vmw_s2) ;

    __vm256 vmhw_r1s0 = vmw_s0 ;
    __vm256 vmhw_r1s2 = vmw_s2 ;

    __vm256 vmhw_r2s0 = _vel_andm_mmm(vmh_r2,vmw_s0) ;
    __vm256 vmhw_r2s1 = vmh_r2 ;
    __vm256 vmhw_r2s2 = _vel_andm_mmm(vmh_r2, vmw_s2) ;

    int64_t outIndex  = outGroupOffset + (n * outChannel + k  ) * outHeight * outWidth + op;

    const float *pInChannel = pIn + inGroupOffset + ((n * inChannel) * inHeight * inWidth ) ;

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

    vrin_r0s0 = _vel_vmrg_vvvml(vrzerof, vrin_r0s0, vmhw_r0s0, vl) ;
    vrin_r0s1 = _vel_vmrg_vvvml(vrzerof, vrin_r0s1, vmhw_r0s1, vl) ;
    vrin_r0s2 = _vel_vmrg_vvvml(vrzerof, vrin_r0s2, vmhw_r0s2, vl) ;
    __vr vrinP_r0s0 = _vel_vshf_vvvsl(vrin_r0s0, vrin_r0s0, VE_VSHUFFLE_YUZU, vl) ;
    __vr vrinP_r0s1 = _vel_vshf_vvvsl(vrin_r0s1, vrin_r0s1, VE_VSHUFFLE_YUZU, vl) ;
    __vr vrinP_r0s2 = _vel_vshf_vvvsl(vrin_r0s2, vrin_r0s2, VE_VSHUFFLE_YUZU, vl) ;

    vrin_r1s0 = _vel_vmrg_vvvml(vrzerof, vrin_r1s0, vmhw_r1s0, vl) ;
    vrin_r1s2 = _vel_vmrg_vvvml(vrzerof, vrin_r1s2, vmhw_r1s2, vl) ;
    __vr vrinP_r1s0 = _vel_vshf_vvvsl(vrin_r1s0, vrin_r1s0, VE_VSHUFFLE_YUZU, vl) ;
    __vr vrinP_r1s1 = _vel_vshf_vvvsl(vrin_r1s1, vrin_r1s1, VE_VSHUFFLE_YUZU, vl) ;
    __vr vrinP_r1s2 = _vel_vshf_vvvsl(vrin_r1s2, vrin_r1s2, VE_VSHUFFLE_YUZU, vl) ;

    vrin_r2s0 = _vel_vmrg_vvvml(vrzerof, vrin_r2s0, vmhw_r2s0, vl) ;
    vrin_r2s1 = _vel_vmrg_vvvml(vrzerof, vrin_r2s1, vmhw_r2s1, vl) ;
    vrin_r2s2 = _vel_vmrg_vvvml(vrzerof, vrin_r2s2, vmhw_r2s2, vl) ;
    __vr vrinP_r2s0 = _vel_vshf_vvvsl(vrin_r2s0, vrin_r2s0, VE_VSHUFFLE_YUZU, vl) ;
    __vr vrinP_r2s1 = _vel_vshf_vvvsl(vrin_r2s1, vrin_r2s1, VE_VSHUFFLE_YUZU, vl) ;
    __vr vrinP_r2s2 = _vel_vshf_vvvsl(vrin_r2s2, vrin_r2s2, VE_VSHUFFLE_YUZU, vl) ;

#define FILTER_R3S3(BIAS, N)							\
    {										\
      __vr vrsum = _vel_pvbrd_vsl(BIAS, vl) ;					\
      vrsum = _vel_pvfmad_vvsvl(vrsum, kerValue[9*(N)+0], vrinP_r0s0, vl) ;	\
      vrsum = _vel_pvfmad_vvsvl(vrsum, kerValue[9*(N)+1], vrinP_r0s1, vl) ;	\
      vrsum = _vel_pvfmad_vvsvl(vrsum, kerValue[9*(N)+2], vrinP_r0s2, vl) ;	\
      vrsum = _vel_pvfmad_vvsvl(vrsum, kerValue[9*(N)+3], vrinP_r1s0, vl) ;	\
      vrsum = _vel_pvfmad_vvsvl(vrsum, kerValue[9*(N)+4], vrinP_r1s1, vl) ;	\
      vrsum = _vel_pvfmad_vvsvl(vrsum, kerValue[9*(N)+5], vrinP_r1s2, vl) ;	\
      vrsum = _vel_pvfmad_vvsvl(vrsum, kerValue[9*(N)+6], vrinP_r2s0, vl) ;	\
      vrsum = _vel_pvfmad_vvsvl(vrsum, kerValue[9*(N)+7], vrinP_r2s1, vl) ;	\
      vrsum = _vel_pvfmad_vvsvl(vrsum, kerValue[9*(N)+8], vrinP_r2s2, vl) ;	\
      _vel_vstu_vssl(vrsum, 4, pOut+outIndex+  (N)                * outHeight * outWidth, vl) ;	\
      _vel_vstl_vssl(vrsum, 4, pOut+outIndex+ ((N)+(NUMKERNEL/2)) * outHeight * outWidth, vl) ;	\
    }
    if(NUMKERNEL>= 2) FILTER_R3S3(bias0, 0) ;
    if(NUMKERNEL>= 4) FILTER_R3S3(bias1, 1) ;
    if(NUMKERNEL>= 6) FILTER_R3S3(bias2, 2) ;
    if(NUMKERNEL>= 8) FILTER_R3S3(bias3, 3) ;
    if(NUMKERNEL>=10) FILTER_R3S3(bias4, 4) ;
    if(NUMKERNEL>=12) FILTER_R3S3(bias5, 5) ;
    if(NUMKERNEL>=14) FILTER_R3S3(bias6, 6) ;
    if(NUMKERNEL>=16) FILTER_R3S3(bias7, 7) ;
#undef FILTER_R3S3
#undef FILTER_OFFSET
  } // outPixels
}

template<int NUMKERNEL, bool ADDBIAS>
static inline void func_even_filterhwcn_opt(
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
  const int64_t outChannelGroup,
  const int64_t padHeight,
  const int64_t padWidth,
  const int64_t inGroupOffset,
  const int64_t outGroupOffset,
  const int64_t biasGroupOffset,
  const int64_t kernGroupOffset,
  const int64_t nY,
  const __vr    vry,
  const __vm256 vmw_s0,
  const __vm256 vmw_s2,
  const int64_t n,
  const int64_t k
)
{
  const __vr vrzerof = _vel_vbrds_vsl(0.f, VLEN) ;

  const int64_t bias01 = ADDBIAS ?  _vel_pack_f32p(pBias+biasGroupOffset+k+ 1, pBias+biasGroupOffset+k+ 0) : 0UL ;
  const int64_t bias23 = ADDBIAS ?  _vel_pack_f32p(pBias+biasGroupOffset+k+ 3, pBias+biasGroupOffset+k+ 2) : 0UL ;
  const int64_t bias45 = ADDBIAS ?  _vel_pack_f32p(pBias+biasGroupOffset+k+ 5, pBias+biasGroupOffset+k+ 4) : 0UL ;
  const int64_t bias67 = ADDBIAS ?  _vel_pack_f32p(pBias+biasGroupOffset+k+ 7, pBias+biasGroupOffset+k+ 6) : 0UL ;
  const int64_t bias89 = ADDBIAS ?  _vel_pack_f32p(pBias+biasGroupOffset+k+ 9, pBias+biasGroupOffset+k+ 8) : 0UL ;
  const int64_t biasAB = ADDBIAS ?  _vel_pack_f32p(pBias+biasGroupOffset+k+11, pBias+biasGroupOffset+k+10) : 0UL ;
  const int64_t biasCD = ADDBIAS ?  _vel_pack_f32p(pBias+biasGroupOffset+k+13, pBias+biasGroupOffset+k+12) : 0UL ;
  const int64_t biasEF = ADDBIAS ?  _vel_pack_f32p(pBias+biasGroupOffset+k+15, pBias+biasGroupOffset+k+14) : 0UL ;

  for (int64_t y=0; y<outHeight; y+=nY)
  {
    const int64_t vl = outWidth * (outHeight - y < nY ? outHeight - y : nY) ;
    const int64_t op = y * outWidth ;

    __vr vrh_r0 = _vel_vaddsl_vsvl(y  -1, vry, vl) ;
    __vr vrh_r2 = _vel_vaddsl_vsvl(y+2-1, vry, vl) ;

    __vm256 vmh_r0 =  _vel_vfmklge_mvl(vrh_r0, vl) ;
    __vm256 vmh_r2 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(inHeight,vrh_r2, vl), vl) ;

    __vm256 vmhw_r0s0 = _vel_andm_mmm(vmh_r0,vmw_s0) ;
    __vm256 vmhw_r0s1 = vmh_r0 ;
    __vm256 vmhw_r0s2 = _vel_andm_mmm(vmh_r0, vmw_s2) ;

    __vm256 vmhw_r1s0 = vmw_s0 ;
    __vm256 vmhw_r1s2 = vmw_s2 ;

    __vm256 vmhw_r2s0 = _vel_andm_mmm(vmh_r2,vmw_s0) ;
    __vm256 vmhw_r2s1 = vmh_r2 ;
    __vm256 vmhw_r2s2 = _vel_andm_mmm(vmh_r2, vmw_s2) ;

    int64_t outIndex  = outGroupOffset + (n * outChannel + k  ) * outHeight * outWidth + op;

    const float *pInChannel = pIn + inGroupOffset + ((n * inChannel) * inHeight * inWidth ) ;

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

    vrin_r0s0 = _vel_vmrg_vvvml(vrzerof, vrin_r0s0, vmhw_r0s0, vl) ;
    vrin_r0s1 = _vel_vmrg_vvvml(vrzerof, vrin_r0s1, vmhw_r0s1, vl) ;
    vrin_r0s2 = _vel_vmrg_vvvml(vrzerof, vrin_r0s2, vmhw_r0s2, vl) ;
    __vr vrinP_r0s0 = _vel_vshf_vvvsl(vrin_r0s0, vrin_r0s0, VE_VSHUFFLE_YUZU, vl) ;
    __vr vrinP_r0s1 = _vel_vshf_vvvsl(vrin_r0s1, vrin_r0s1, VE_VSHUFFLE_YUZU, vl) ;
    __vr vrinP_r0s2 = _vel_vshf_vvvsl(vrin_r0s2, vrin_r0s2, VE_VSHUFFLE_YUZU, vl) ;

    vrin_r1s0 = _vel_vmrg_vvvml(vrzerof, vrin_r1s0, vmhw_r1s0, vl) ;
    vrin_r1s2 = _vel_vmrg_vvvml(vrzerof, vrin_r1s2, vmhw_r1s2, vl) ;
    __vr vrinP_r1s0 = _vel_vshf_vvvsl(vrin_r1s0, vrin_r1s0, VE_VSHUFFLE_YUZU, vl) ;
    __vr vrinP_r1s1 = _vel_vshf_vvvsl(vrin_r1s1, vrin_r1s1, VE_VSHUFFLE_YUZU, vl) ;
    __vr vrinP_r1s2 = _vel_vshf_vvvsl(vrin_r1s2, vrin_r1s2, VE_VSHUFFLE_YUZU, vl) ;

    vrin_r2s0 = _vel_vmrg_vvvml(vrzerof, vrin_r2s0, vmhw_r2s0, vl) ;
    vrin_r2s1 = _vel_vmrg_vvvml(vrzerof, vrin_r2s1, vmhw_r2s1, vl) ;
    vrin_r2s2 = _vel_vmrg_vvvml(vrzerof, vrin_r2s2, vmhw_r2s2, vl) ;
    __vr vrinP_r2s0 = _vel_vshf_vvvsl(vrin_r2s0, vrin_r2s0, VE_VSHUFFLE_YUZU, vl) ;
    __vr vrinP_r2s1 = _vel_vshf_vvvsl(vrin_r2s1, vrin_r2s1, VE_VSHUFFLE_YUZU, vl) ;
    __vr vrinP_r2s2 = _vel_vshf_vvvsl(vrin_r2s2, vrin_r2s2, VE_VSHUFFLE_YUZU, vl) ;

    const uint64_t *pKerValue_u64 = (const uint64_t*) ( pKernel + kernGroupOffset + k ) ;

#define FILTER_R3S3(BIAS, N)										\
    {													\
      __vr vrsum = _vel_pvbrd_vsl(BIAS, vl) ;								\
      vrsum = _vel_pvfmad_vvsvl(vrsum, pKerValue_u64[(N/2)+0*outChannelGroup/2], vrinP_r0s0, vl) ;	\
      vrsum = _vel_pvfmad_vvsvl(vrsum, pKerValue_u64[(N/2)+1*outChannelGroup/2], vrinP_r0s1, vl) ;	\
      vrsum = _vel_pvfmad_vvsvl(vrsum, pKerValue_u64[(N/2)+2*outChannelGroup/2], vrinP_r0s2, vl) ;	\
      vrsum = _vel_pvfmad_vvsvl(vrsum, pKerValue_u64[(N/2)+3*outChannelGroup/2], vrinP_r1s0, vl) ;	\
      vrsum = _vel_pvfmad_vvsvl(vrsum, pKerValue_u64[(N/2)+4*outChannelGroup/2], vrinP_r1s1, vl) ;	\
      vrsum = _vel_pvfmad_vvsvl(vrsum, pKerValue_u64[(N/2)+5*outChannelGroup/2], vrinP_r1s2, vl) ;	\
      vrsum = _vel_pvfmad_vvsvl(vrsum, pKerValue_u64[(N/2)+6*outChannelGroup/2], vrinP_r2s0, vl) ;	\
      vrsum = _vel_pvfmad_vvsvl(vrsum, pKerValue_u64[(N/2)+7*outChannelGroup/2], vrinP_r2s1, vl) ;	\
      vrsum = _vel_pvfmad_vvsvl(vrsum, pKerValue_u64[(N/2)+8*outChannelGroup/2], vrinP_r2s2, vl) ;	\
      _vel_vstl_vssl(vrsum, 4, pOut+outIndex+ ((N)  ) * outHeight * outWidth , vl) ;			\
      _vel_vstu_vssl(vrsum, 4, pOut+outIndex+ ((N)+1) * outHeight * outWidth , vl) ;			\
    }
    if(NUMKERNEL>= 2) FILTER_R3S3(bias01, 0) ;
    if(NUMKERNEL>= 4) FILTER_R3S3(bias23, 2) ;
    if(NUMKERNEL>= 6) FILTER_R3S3(bias45, 4) ;
    if(NUMKERNEL>= 8) FILTER_R3S3(bias67, 6) ;
    if(NUMKERNEL>=10) FILTER_R3S3(bias89, 8) ;
    if(NUMKERNEL>=12) FILTER_R3S3(biasAB, 10) ;
    if(NUMKERNEL>=14) FILTER_R3S3(biasCD, 12) ;
    if(NUMKERNEL>=16) FILTER_R3S3(biasEF, 14) ;
#undef FILTER_R3S3
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
    const int64_t outChannelGroup,
    const int64_t padHeight,
    const int64_t padWidth
)
{
  const int64_t nY = VLEN / outWidth ;

  __vr vrseq = _vel_vseq_vl(VLEN) ; // xy

  __vr vry   = _vel_vdivsl_vvsl(vrseq, outWidth, VLEN) ;
  __vr vrx   = _vel_vsubsl_vvvl(vrseq, _vel_vmulul_vsvl(outWidth,vry, VLEN), VLEN) ;

  __vr vrw_s0 = _vel_vaddsl_vsvl( -padWidth, vrx, VLEN) ;
  __vr vrw_s2 = _vel_vaddsl_vsvl(2-padWidth, vrx, VLEN) ;

  __vm256 vmw_s0  =  _vel_vfmklge_mvl(vrw_s0, VLEN) ;
  __vm256 vmw_s2  =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(inWidth,vrw_s2, VLEN), VLEN) ;

  for (int64_t n = 0; n < batch; n++) {
    for (int64_t g = 0; g < group; g++) {
	const int64_t inGroupOffset   = g * 1 * inHeight * inWidth;
	const int64_t outGroupOffset  = g * outChannelGroup * outHeight * outWidth;
	const int64_t biasGroupOffset = g * outChannelGroup;
	const int64_t kernGroupOffset = g * outChannelGroup * 1 * kernHeight * kernWidth;

	const int64_t remain = outChannelGroup & 0xf ;

	int k = 0 ;
	switch( remain ) {
	case 1 :
	  func_odd<FLAYOUT, 1, ADDBIAS>(pIn, pKernel, pBias, pOut,
	     inChannel, inWidth, inHeight,
	     outChannel, outWidth, outHeight,
	     kernWidth, kernHeight,
	     outChannelGroup,
	     padHeight, padWidth,
	     inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	     nY, vry, vmw_s0, vmw_s2,
	     n, k );
	  k+=1 ;
	  break ;
	case 2 :
	  func_even<FLAYOUT, 2, ADDBIAS>(pIn, pKernel, pBias, pOut,
	     inChannel, inWidth, inHeight,
	     outChannel, outWidth, outHeight,
	     kernWidth, kernHeight,
	     outChannelGroup,
	     padHeight, padWidth,
	     inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	     nY, vry, vmw_s0, vmw_s2,
	     n, k );
	  k+=2 ;
	  break ;
	case 3 :
	  func_odd<FLAYOUT, 3, ADDBIAS>(pIn, pKernel, pBias, pOut,
	     inChannel, inWidth, inHeight,
	     outChannel, outWidth, outHeight,
	     kernWidth, kernHeight,
	     outChannelGroup,
	     padHeight, padWidth,
	     inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	     nY, vry, vmw_s0, vmw_s2,
	     n, k );
	  k+=3 ;
	  break ;
	case 4 :
	  func_even<FLAYOUT, 4, ADDBIAS>(pIn, pKernel, pBias, pOut,
	     inChannel, inWidth, inHeight,
	     outChannel, outWidth, outHeight,
	     kernWidth, kernHeight,
	     outChannelGroup,
	     padHeight, padWidth,
	     inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	     nY, vry, vmw_s0, vmw_s2,
	     n, k );
	  k+=4 ;
	  break ;
	case 5 :
	  func_odd<FLAYOUT, 5, ADDBIAS>(pIn, pKernel, pBias, pOut,
	     inChannel, inWidth, inHeight,
	     outChannel, outWidth, outHeight,
	     kernWidth, kernHeight,
	     outChannelGroup,
	     padHeight, padWidth,
	     inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	     nY, vry, vmw_s0, vmw_s2,
	     n, k );
	  k+=5 ;
	  break ;
	case 6 :
	  func_even<FLAYOUT, 6, ADDBIAS>(pIn, pKernel, pBias, pOut,
	     inChannel, inWidth, inHeight,
	     outChannel, outWidth, outHeight,
	     kernWidth, kernHeight,
	     outChannelGroup,
	     padHeight, padWidth,
	     inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	     nY, vry, vmw_s0, vmw_s2,
	     n, k );
	  k+=6 ;
	  break ;
	case 7 :
	  func_odd<FLAYOUT, 7, ADDBIAS>(pIn, pKernel, pBias, pOut,
	     inChannel, inWidth, inHeight,
	     outChannel, outWidth, outHeight,
	     kernWidth, kernHeight,
	     outChannelGroup,
	     padHeight, padWidth,
	     inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	     nY, vry, vmw_s0, vmw_s2,
	     n, k );
	  k+=7 ;
	  break ;
	case 8 :
	  if( FLAYOUT == VEDNN_FILTER_LAYOUT_NCHW ) {
	    func_even_filternchw_opt<8, ADDBIAS>(pIn, pKernel, pBias, pOut,
	       inChannel, inWidth, inHeight,
	       outChannel, outWidth, outHeight,
	       kernWidth, kernHeight,
	       outChannelGroup,
	       padHeight, padWidth,
	       inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	       nY, vry, vmw_s0, vmw_s2,
	       n, k );
	  }
	  else {
	    if( (((uint64_t)(pKernel + kernGroupOffset + k)) & 0x7) == 0
		&& (outChannelGroup & 0x1) == 0 ) {
	      func_even_filterhwcn_opt<8, ADDBIAS>(pIn, pKernel, pBias, pOut,
		 inChannel, inWidth, inHeight,
		 outChannel, outWidth, outHeight,
		 kernWidth, kernHeight,
		 outChannelGroup,
		 padHeight, padWidth,
		 inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
		 nY, vry, vmw_s0, vmw_s2,
		 n, k );
	    }
	    else {
	      func_even<FLAYOUT, 8, ADDBIAS>(pIn, pKernel, pBias, pOut,
		 inChannel, inWidth, inHeight,
		 outChannel, outWidth, outHeight,
		 kernWidth, kernHeight,
		 outChannelGroup,
		 padHeight, padWidth,
		 inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
		 nY, vry, vmw_s0, vmw_s2,
		 n, k );
	    }
	  }
	  k+=8 ;
	  break ;
	case 9 :
	  if( FLAYOUT == VEDNN_FILTER_LAYOUT_NCHW ) {
	    func_odd_filternchw_opt<9, ADDBIAS>(pIn, pKernel, pBias, pOut,
	       inChannel, inWidth, inHeight,
	       outChannel, outWidth, outHeight,
	       kernWidth, kernHeight,
	       outChannelGroup,
	       padHeight, padWidth,
	       inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	       nY, vry, vmw_s0, vmw_s2,
	       n, k );
	  }
	  else {
	    func_odd<FLAYOUT, 9, ADDBIAS>(pIn, pKernel, pBias, pOut,
	       inChannel, inWidth, inHeight,
	       outChannel, outWidth, outHeight,
	       kernWidth, kernHeight,
	       outChannelGroup,
	       padHeight, padWidth,
	       inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	       nY, vry, vmw_s0, vmw_s2,
	       n, k );
	  }
	  k+=9 ;
	  break ;
	case 10 :
	  if( FLAYOUT == VEDNN_FILTER_LAYOUT_NCHW ) {
	    func_even_filternchw_opt<10, ADDBIAS>(pIn, pKernel, pBias, pOut,
	       inChannel, inWidth, inHeight,
	       outChannel, outWidth, outHeight,
	       kernWidth, kernHeight,
	       outChannelGroup,
	       padHeight, padWidth,
	       inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	       nY, vry, vmw_s0, vmw_s2,
	       n, k );
	  }
	  else {
	    if( (((uint64_t)(pKernel + kernGroupOffset + k)) & 0x7) == 0
		&& (outChannelGroup & 0x1) == 0 ) {
	      func_even_filterhwcn_opt<10, ADDBIAS>(pIn, pKernel, pBias, pOut,
		 inChannel, inWidth, inHeight,
		 outChannel, outWidth, outHeight,
		 kernWidth, kernHeight,
		 outChannelGroup,
		 padHeight, padWidth,
		 inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
		 nY, vry, vmw_s0, vmw_s2,
		 n, k );
	    }
	    else {
	      func_even<FLAYOUT, 10, ADDBIAS>(pIn, pKernel, pBias, pOut,
		 inChannel, inWidth, inHeight,
		 outChannel, outWidth, outHeight,
		 kernWidth, kernHeight,
		 outChannelGroup,
		 padHeight, padWidth,
		 inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
		 nY, vry, vmw_s0, vmw_s2,
		 n, k );
	    }
	  }
	  k+=10 ;
	  break ;
	case 11 :
	  if( FLAYOUT == VEDNN_FILTER_LAYOUT_NCHW ) {
	    func_odd_filternchw_opt<11, ADDBIAS>(pIn, pKernel, pBias, pOut,
	       inChannel, inWidth, inHeight,
	       outChannel, outWidth, outHeight,
	       kernWidth, kernHeight,
	       outChannelGroup,
	       padHeight, padWidth,
	       inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	       nY, vry, vmw_s0, vmw_s2,
	       n, k );
	  }
	  else {
	    func_odd<FLAYOUT, 11, ADDBIAS>(pIn, pKernel, pBias, pOut,
	       inChannel, inWidth, inHeight,
	       outChannel, outWidth, outHeight,
	       kernWidth, kernHeight,
	       outChannelGroup,
	       padHeight, padWidth,
	       inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	       nY, vry, vmw_s0, vmw_s2,
	       n, k );
	  }
	  k+=11 ;
	  break ;
	case 12 :
	  if( FLAYOUT == VEDNN_FILTER_LAYOUT_NCHW ) {
	    func_even_filternchw_opt<12, ADDBIAS>(pIn, pKernel, pBias, pOut,
	       inChannel, inWidth, inHeight,
	       outChannel, outWidth, outHeight,
	       kernWidth, kernHeight,
	       outChannelGroup,
	       padHeight, padWidth,
	       inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	       nY, vry, vmw_s0, vmw_s2,
	       n, k );
	  }
	  else {
	    if( (((uint64_t)(pKernel + kernGroupOffset + k)) & 0x7) == 0
		&& (outChannelGroup & 0x1) == 0 ) {
	      func_even_filterhwcn_opt<12, ADDBIAS>(pIn, pKernel, pBias, pOut,
		 inChannel, inWidth, inHeight,
		 outChannel, outWidth, outHeight,
		 kernWidth, kernHeight,
		 outChannelGroup,
		 padHeight, padWidth,
		 inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
		 nY, vry, vmw_s0, vmw_s2,
		 n, k );
	    }
	    else {
	      func_even<FLAYOUT, 12, ADDBIAS>(pIn, pKernel, pBias, pOut,
		 inChannel, inWidth, inHeight,
		 outChannel, outWidth, outHeight,
		 kernWidth, kernHeight,
		 outChannelGroup,
		 padHeight, padWidth,
		 inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
		 nY, vry, vmw_s0, vmw_s2,
		 n, k );
	    }
	  }
	  k+=12 ;
	  break ;
	case 13 :
	  if( FLAYOUT == VEDNN_FILTER_LAYOUT_NCHW ) {
	    func_odd_filternchw_opt<13, ADDBIAS>(pIn, pKernel, pBias, pOut,
	       inChannel, inWidth, inHeight,
	       outChannel, outWidth, outHeight,
	       kernWidth, kernHeight,
	       outChannelGroup,
	       padHeight, padWidth,
	       inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	       nY, vry, vmw_s0, vmw_s2,
	       n, k );
	  }
	  else {
	    func_odd<FLAYOUT, 13, ADDBIAS>(pIn, pKernel, pBias, pOut,
	       inChannel, inWidth, inHeight,
	       outChannel, outWidth, outHeight,
	       kernWidth, kernHeight,
	       outChannelGroup,
	       padHeight, padWidth,
	       inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	       nY, vry, vmw_s0, vmw_s2,
	       n, k );
	  }
	  k+=13 ;
	  break ;
	case 14 :
	  if( FLAYOUT == VEDNN_FILTER_LAYOUT_NCHW ) {
	    func_even_filternchw_opt<14, ADDBIAS>(pIn, pKernel, pBias, pOut,
	       inChannel, inWidth, inHeight,
	       outChannel, outWidth, outHeight,
	       kernWidth, kernHeight,
	       outChannelGroup,
	       padHeight, padWidth,
	       inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	       nY, vry, vmw_s0, vmw_s2,
	       n, k );
	  }
	  else {
	    if( (((uint64_t)(pKernel + kernGroupOffset + k)) & 0x7) == 0
		&& (outChannelGroup & 0x1) == 0 ) {
	      func_even_filterhwcn_opt<14, ADDBIAS>(pIn, pKernel, pBias, pOut,
		 inChannel, inWidth, inHeight,
		 outChannel, outWidth, outHeight,
		 kernWidth, kernHeight,
		 outChannelGroup,
		 padHeight, padWidth,
		 inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
		 nY, vry, vmw_s0, vmw_s2,
		 n, k );
	    }
	    else {
	      func_even<FLAYOUT, 14, ADDBIAS>(pIn, pKernel, pBias, pOut,
		 inChannel, inWidth, inHeight,
		 outChannel, outWidth, outHeight,
		 kernWidth, kernHeight,
		 outChannelGroup,
		 padHeight, padWidth,
		 inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
		 nY, vry, vmw_s0, vmw_s2,
		 n, k );
	    }
	  }
	  k+=14 ;
	  break ;
	case 15 :
	  if( FLAYOUT == VEDNN_FILTER_LAYOUT_NCHW ) {
	    func_odd_filternchw_opt<15, ADDBIAS>(pIn, pKernel, pBias, pOut,
	       inChannel, inWidth, inHeight,
	       outChannel, outWidth, outHeight,
	       kernWidth, kernHeight,
	       outChannelGroup,
	       padHeight, padWidth,
	       inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	       nY, vry, vmw_s0, vmw_s2,
	       n, k );
	  }
	  else {
	    func_odd<FLAYOUT, 15, ADDBIAS>(pIn, pKernel, pBias, pOut,
	       inChannel, inWidth, inHeight,
	       outChannel, outWidth, outHeight,
	       kernWidth, kernHeight,
	       outChannelGroup,
	       padHeight, padWidth,
	       inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	       nY, vry, vmw_s0, vmw_s2,
	       n, k );
	  }
	  k+=15 ;
	  break ;
	default :
	  break ;
	}
	for (; k < outChannelGroup; k+=16) {
	  if( FLAYOUT == VEDNN_FILTER_LAYOUT_NCHW ) {
	    func_even_filternchw_opt<16, ADDBIAS>(pIn, pKernel, pBias, pOut,
	       inChannel, inWidth, inHeight,
	       outChannel, outWidth, outHeight,
	       kernWidth, kernHeight,
	       outChannelGroup,
	       padHeight, padWidth,
	       inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	       nY, vry, vmw_s0, vmw_s2,
	       n, k );
	  }
	  else {
	    if( (((uint64_t)(pKernel + kernGroupOffset + k)) & 0x7) == 0
		&& (outChannelGroup & 0x1) == 0 ) {
	      func_even_filterhwcn_opt<16, ADDBIAS>(pIn, pKernel, pBias, pOut,
		 inChannel, inWidth, inHeight,
		 outChannel, outWidth, outHeight,
		 kernWidth, kernHeight,
		 outChannelGroup,
		 padHeight, padWidth,
		 inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
		 nY, vry, vmw_s0, vmw_s2,
		 n, k );
	    }
	    else {
	      func_even<FLAYOUT, 16, ADDBIAS>(pIn, pKernel, pBias, pOut,
		 inChannel, inWidth, inHeight,
		 outChannel, outWidth, outHeight,
		 kernWidth, kernHeight,
		 outChannelGroup,
		 padHeight, padWidth,
		 inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
		 nY, vry, vmw_s0, vmw_s2,
		 n, k );
	    }
	  }
	} // outChannel
    } // group
  } // batch
}

extern "C" vednnError_t
vednnConvolutionForward_direct_dil1_str1_padsame_ker3_c1_owU128(
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
  const int64_t outWidth   = pParamOut->width;			/* must be <= 128 */
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

//  const int64_t inChannelGroup  = inChannel  / group;   // equal to pDataKernel->inChannel ( must be 1 )
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
		 outChannelGroup,
		 padHeight, padWidth ) ;
    }
    else {
      convloop<VEDNN_FILTER_LAYOUT_NCHW, true>(pIn, pKernel, pBias, pOut,
		 batch, group,
		 inChannel, inWidth, inHeight,
		 outChannel, outWidth, outHeight,
		 kernWidth, kernHeight,
		 outChannelGroup,
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
		 outChannelGroup,
		 padHeight, padWidth ) ;
    }
    else {
      convloop<VEDNN_FILTER_LAYOUT_HWCN, true>(pIn, pKernel, pBias, pOut,
		 batch, group,
		 inChannel, inWidth, inHeight,
		 outChannel, outWidth, outHeight,
		 kernWidth, kernHeight,
		 outChannelGroup,
		 padHeight, padWidth ) ;
    }
  }

  return VEDNN_SUCCESS;
}
