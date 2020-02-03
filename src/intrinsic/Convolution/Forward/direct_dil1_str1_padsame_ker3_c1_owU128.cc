#include <stdint.h>
#include <stdlib.h>

#include "vednn.h"

#include "velintrin.h"
#define VLEN	(256)

#include <stdio.h>


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
  const int64_t remain  = NUMKERNEL & 0x1 ;
  const int64_t nPacked = NUMKERNEL >> 1 ;

  const float   bias0  = ADDBIAS ?  pBias[biasGroupOffset+k+ 0] : 0.f ;
  int64_t bias[nPacked] ;
#pragma clang loop unroll(full)
  for(int64_t kk=0; kk<nPacked; kk++) bias[kk] = ADDBIAS ?  _vel_pack_f32p(pBias+biasGroupOffset+k+2*kk+remain, pBias+biasGroupOffset+k+2*kk+remain+1) : 0UL ;

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

    if( remain ) {
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
#pragma clang loop unroll(full)
    for(int64_t kk=0; kk<nPacked; kk++ ) {
      __vr vrsum = _vel_pvbrd_vsl(bias[kk], vl) ;
      const uint64_t kerValue_r0s0 = _vel_pack_f32p(pKernel + FILTER_OFFSET(k+2*kk+remain  ,0,0,0),
						    pKernel + FILTER_OFFSET(k+2*kk+remain+1,0,0,0));
      const uint64_t kerValue_r0s1 = _vel_pack_f32p(pKernel + FILTER_OFFSET(k+2*kk+remain  ,0,0,1),
						    pKernel + FILTER_OFFSET(k+2*kk+remain+1,0,0,1));
      const uint64_t kerValue_r0s2 = _vel_pack_f32p(pKernel + FILTER_OFFSET(k+2*kk+remain  ,0,0,2),
						    pKernel + FILTER_OFFSET(k+2*kk+remain+1,0,0,2));
      const uint64_t kerValue_r1s0 = _vel_pack_f32p(pKernel + FILTER_OFFSET(k+2*kk+remain  ,0,1,0),
						    pKernel + FILTER_OFFSET(k+2*kk+remain+1,0,1,0));
      const uint64_t kerValue_r1s1 = _vel_pack_f32p(pKernel + FILTER_OFFSET(k+2*kk+remain  ,0,1,1),
						    pKernel + FILTER_OFFSET(k+2*kk+remain+1,0,1,1));
      const uint64_t kerValue_r1s2 = _vel_pack_f32p(pKernel + FILTER_OFFSET(k+2*kk+remain  ,0,1,2),
						    pKernel + FILTER_OFFSET(k+2*kk+remain+1,0,1,2));
      const uint64_t kerValue_r2s0 = _vel_pack_f32p(pKernel + FILTER_OFFSET(k+2*kk+remain  ,0,2,0),
						    pKernel + FILTER_OFFSET(k+2*kk+remain+1,0,2,0));
      const uint64_t kerValue_r2s1 = _vel_pack_f32p(pKernel + FILTER_OFFSET(k+2*kk+remain  ,0,2,1),
						    pKernel + FILTER_OFFSET(k+2*kk+remain+1,0,2,1));
      const uint64_t kerValue_r2s2 = _vel_pack_f32p(pKernel + FILTER_OFFSET(k+2*kk+remain  ,0,2,2),
						    pKernel + FILTER_OFFSET(k+2*kk+remain+1,0,2,2));
      vrsum = _vel_pvfmad_vvsvl(vrsum, kerValue_r0s0, vrinP_r0s0, vl) ;
      vrsum = _vel_pvfmad_vvsvl(vrsum, kerValue_r0s1, vrinP_r0s1, vl) ;
      vrsum = _vel_pvfmad_vvsvl(vrsum, kerValue_r0s2, vrinP_r0s2, vl) ;
      vrsum = _vel_pvfmad_vvsvl(vrsum, kerValue_r1s0, vrinP_r1s0, vl) ;
      vrsum = _vel_pvfmad_vvsvl(vrsum, kerValue_r1s1, vrinP_r1s1, vl) ;
      vrsum = _vel_pvfmad_vvsvl(vrsum, kerValue_r1s2, vrinP_r1s2, vl) ;
      vrsum = _vel_pvfmad_vvsvl(vrsum, kerValue_r2s0, vrinP_r2s0, vl) ;
      vrsum = _vel_pvfmad_vvsvl(vrsum, kerValue_r2s1, vrinP_r2s1, vl) ;
      vrsum = _vel_pvfmad_vvsvl(vrsum, kerValue_r2s2, vrinP_r2s2, vl) ;
      _vel_vstu_vssl(vrsum, 4, pOut+outIndex+ (2*kk+remain  )* outHeight * outWidth, vl) ;
      _vel_vstl_vssl(vrsum, 4, pOut+outIndex+ (2*kk+remain+1)* outHeight * outWidth, vl) ;
    }

#undef FILTER_OFFSET
  } // outPixels
}


template<int NUMKERNEL, bool ADDBIAS>
static inline void func_filternchw_opt(
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
  const int64_t remain  = NUMKERNEL & 0x1 ;
  const int64_t nPacked = NUMKERNEL >> 1 ;

  const float   bias0  = ADDBIAS ?  pBias[biasGroupOffset+k+ 0] : 0.f ;
  int64_t bias[nPacked] ;
#pragma clang loop unroll(full)
  for(int64_t kk=0; kk<nPacked; kk++) bias[kk] = ADDBIAS ?  _vel_pack_f32p(pBias+biasGroupOffset+k+kk+remain, pBias+biasGroupOffset+k+kk+remain+nPacked) : 0UL ;

  const __vr vrzerof = _vel_vbrds_vsl(0.f, VLEN) ;

  uint64_t kerValue[nPacked*9] ;
  {
    const float *pKerValue  = pKernel + kernGroupOffset + (k * kernHeight    ) * kernWidth ;
    __vr vrker0 = _vel_vldu_vssl(4, pKerValue+remain*9          , nPacked*9) ;
    __vr vrker1 = _vel_vldu_vssl(4, pKerValue+(remain+nPacked)*9, nPacked*9) ;
    _vel_vst_vssl(_vel_vshf_vvvsl(vrker0, vrker1, VE_VSHUFFLE_YUZU, nPacked*9) ,8, kerValue, nPacked*9) ;
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

    if( remain ){
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

#pragma clang loop unroll(full)
    for(int64_t kk=0; kk<nPacked; kk++ ) {
      __vr vrsum = _vel_pvbrd_vsl(bias[kk], vl) ;
      vrsum = _vel_pvfmad_vvsvl(vrsum, kerValue[9*kk+0], vrinP_r0s0, vl) ;
      vrsum = _vel_pvfmad_vvsvl(vrsum, kerValue[9*kk+1], vrinP_r0s1, vl) ;
      vrsum = _vel_pvfmad_vvsvl(vrsum, kerValue[9*kk+2], vrinP_r0s2, vl) ;
      vrsum = _vel_pvfmad_vvsvl(vrsum, kerValue[9*kk+3], vrinP_r1s0, vl) ;
      vrsum = _vel_pvfmad_vvsvl(vrsum, kerValue[9*kk+4], vrinP_r1s1, vl) ;
      vrsum = _vel_pvfmad_vvsvl(vrsum, kerValue[9*kk+5], vrinP_r1s2, vl) ;
      vrsum = _vel_pvfmad_vvsvl(vrsum, kerValue[9*kk+6], vrinP_r2s0, vl) ;
      vrsum = _vel_pvfmad_vvsvl(vrsum, kerValue[9*kk+7], vrinP_r2s1, vl) ;
      vrsum = _vel_pvfmad_vvsvl(vrsum, kerValue[9*kk+8], vrinP_r2s2, vl) ;
      _vel_vstu_vssl(vrsum, 4, pOut+outIndex+ (kk+remain        )* outHeight * outWidth, vl) ;
      _vel_vstl_vssl(vrsum, 4, pOut+outIndex+ (kk+remain+nPacked)* outHeight * outWidth, vl) ;
    }

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

  const int64_t nPacked = NUMKERNEL >> 1 ;

  int64_t bias[nPacked] ;
#pragma clang loop unroll(full)
  for(int64_t kk=0; kk<nPacked; kk++) bias[kk] = ADDBIAS ?  _vel_pack_f32p(pBias+biasGroupOffset+k+2*kk+1, pBias+biasGroupOffset+k+2*kk) : 0UL ;

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

#pragma clang loop unroll(full)
    for(int64_t kk=0; kk<nPacked; kk++ ) {
      __vr vrsum = _vel_pvbrd_vsl(bias[kk], vl) ;
      vrsum = _vel_pvfmad_vvsvl(vrsum, pKerValue_u64[kk+0*outChannelGroup/2], vrinP_r0s0, vl) ;
      vrsum = _vel_pvfmad_vvsvl(vrsum, pKerValue_u64[kk+1*outChannelGroup/2], vrinP_r0s1, vl) ;
      vrsum = _vel_pvfmad_vvsvl(vrsum, pKerValue_u64[kk+2*outChannelGroup/2], vrinP_r0s2, vl) ;
      vrsum = _vel_pvfmad_vvsvl(vrsum, pKerValue_u64[kk+3*outChannelGroup/2], vrinP_r1s0, vl) ;
      vrsum = _vel_pvfmad_vvsvl(vrsum, pKerValue_u64[kk+4*outChannelGroup/2], vrinP_r1s1, vl) ;
      vrsum = _vel_pvfmad_vvsvl(vrsum, pKerValue_u64[kk+5*outChannelGroup/2], vrinP_r1s2, vl) ;
      vrsum = _vel_pvfmad_vvsvl(vrsum, pKerValue_u64[kk+6*outChannelGroup/2], vrinP_r2s0, vl) ;
      vrsum = _vel_pvfmad_vvsvl(vrsum, pKerValue_u64[kk+7*outChannelGroup/2], vrinP_r2s1, vl) ;
      vrsum = _vel_pvfmad_vvsvl(vrsum, pKerValue_u64[kk+8*outChannelGroup/2], vrinP_r2s2, vl) ;
      _vel_vstl_vssl(vrsum, 4, pOut+outIndex+ (2*kk  ) * outHeight * outWidth , vl) ;
      _vel_vstu_vssl(vrsum, 4, pOut+outIndex+ (2*kk+1) * outHeight * outWidth , vl) ;
    }

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
	  func<FLAYOUT, 1, ADDBIAS>(pIn, pKernel, pBias, pOut,
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
	  func<FLAYOUT, 2, ADDBIAS>(pIn, pKernel, pBias, pOut,
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
	  func<FLAYOUT, 3, ADDBIAS>(pIn, pKernel, pBias, pOut,
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
	  func<FLAYOUT, 4, ADDBIAS>(pIn, pKernel, pBias, pOut,
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
	  func<FLAYOUT, 5, ADDBIAS>(pIn, pKernel, pBias, pOut,
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
	  func<FLAYOUT, 6, ADDBIAS>(pIn, pKernel, pBias, pOut,
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
	  func<FLAYOUT, 7, ADDBIAS>(pIn, pKernel, pBias, pOut,
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
	    func_filternchw_opt<8, ADDBIAS>(pIn, pKernel, pBias, pOut,
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
	      func<FLAYOUT, 8, ADDBIAS>(pIn, pKernel, pBias, pOut,
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
	    func_filternchw_opt<9, ADDBIAS>(pIn, pKernel, pBias, pOut,
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
	    func<FLAYOUT, 9, ADDBIAS>(pIn, pKernel, pBias, pOut,
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
	    func_filternchw_opt<10, ADDBIAS>(pIn, pKernel, pBias, pOut,
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
	      func<FLAYOUT, 10, ADDBIAS>(pIn, pKernel, pBias, pOut,
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
	    func_filternchw_opt<11, ADDBIAS>(pIn, pKernel, pBias, pOut,
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
	    func<FLAYOUT, 11, ADDBIAS>(pIn, pKernel, pBias, pOut,
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
	    func_filternchw_opt<12, ADDBIAS>(pIn, pKernel, pBias, pOut,
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
	      func<FLAYOUT, 12, ADDBIAS>(pIn, pKernel, pBias, pOut,
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
	    func_filternchw_opt<13, ADDBIAS>(pIn, pKernel, pBias, pOut,
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
	    func<FLAYOUT, 13, ADDBIAS>(pIn, pKernel, pBias, pOut,
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
	    func_filternchw_opt<14, ADDBIAS>(pIn, pKernel, pBias, pOut,
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
	      func<FLAYOUT, 14, ADDBIAS>(pIn, pKernel, pBias, pOut,
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
	    func_filternchw_opt<15, ADDBIAS>(pIn, pKernel, pBias, pOut,
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
	    func<FLAYOUT, 15, ADDBIAS>(pIn, pKernel, pBias, pOut,
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
	    func_filternchw_opt<16, ADDBIAS>(pIn, pKernel, pBias, pOut,
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
	      func<FLAYOUT, 16, ADDBIAS>(pIn, pKernel, pBias, pOut,
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

