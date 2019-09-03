#include <stdint.h>
#include <stdlib.h>

#include "vednn.h"

#include "velintrin.h"
#define VLEN	(256)

#include <stdio.h>

template<filterLayout_t FLAYOUT>
static inline void k1(
    const float * __restrict__ pIn,
    const float * __restrict__ pKernel,
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
    const int64_t inGroupOffset,
    const int64_t outGroupOffset,
    const int64_t kernGroupOffset,
    const int64_t oPixels,
    const int64_t nY,
    const __vr    vry,
    const __vm256 vmw_s0,
    const __vm256 vmw_s2,
    const int64_t n,
    const int64_t k
)
{
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

    int64_t outIndex  = outGroupOffset + (n * outChannel + k  ) * oPixels + op;

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

    vrin_r1s0 = _vel_vmrg_vvvml(vrzerof, vrin_r1s0, vmhw_r1s0, vl) ;
    vrin_r1s2 = _vel_vmrg_vvvml(vrzerof, vrin_r1s2, vmhw_r1s2, vl) ;

    vrin_r2s0 = _vel_vmrg_vvvml(vrzerof, vrin_r2s0, vmhw_r2s0, vl) ;
    vrin_r2s1 = _vel_vmrg_vvvml(vrzerof, vrin_r2s1, vmhw_r2s1, vl) ;
    vrin_r2s2 = _vel_vmrg_vvvml(vrzerof, vrin_r2s2, vmhw_r2s2, vl) ;

    const float *pKerValue = ( FLAYOUT == VEDNN_FILTER_LAYOUT_NCHW ) ?
  			     pKernel + kernGroupOffset + k * kernHeight * kernWidth  :
                             pKernel + kernGroupOffset + k ;

//    const int64_t kernelDistance0 = ( FLAYOUT == VEDNN_FILTER_LAYOUT_NCHW ) ?
//  	                            kernHeight * kernWidth :
//  				    1 ;

    const int64_t kernelDistance1 = ( FLAYOUT == VEDNN_FILTER_LAYOUT_NCHW ) ?
  	                            1 :
  				    outChannelGroup ;

    __vr vrsum = _vel_vbrds_vsl(0.f, vl) ;
    vrsum = _vel_vfmads_vvsvl(vrsum, pKerValue[0*kernelDistance1], vrin_r0s0, vl) ;
    vrsum = _vel_vfmads_vvsvl(vrsum, pKerValue[1*kernelDistance1], vrin_r0s1, vl) ;
    vrsum = _vel_vfmads_vvsvl(vrsum, pKerValue[2*kernelDistance1], vrin_r0s2, vl) ;
    vrsum = _vel_vfmads_vvsvl(vrsum, pKerValue[3*kernelDistance1], vrin_r1s0, vl) ;
    vrsum = _vel_vfmads_vvsvl(vrsum, pKerValue[4*kernelDistance1], vrin_r1s1, vl) ;
    vrsum = _vel_vfmads_vvsvl(vrsum, pKerValue[5*kernelDistance1], vrin_r1s2, vl) ;
    vrsum = _vel_vfmads_vvsvl(vrsum, pKerValue[6*kernelDistance1], vrin_r2s0, vl) ;
    vrsum = _vel_vfmads_vvsvl(vrsum, pKerValue[7*kernelDistance1], vrin_r2s1, vl) ;
    vrsum = _vel_vfmads_vvsvl(vrsum, pKerValue[8*kernelDistance1], vrin_r2s2, vl) ;
    _vel_vstu_vssl(vrsum, 4, pOut+outIndex+ (0)    *oPixels, vl) ;

  } // outPixels
}

template<filterLayout_t FLAYOUT, bool useopt = false>
static inline void k2(
    const float * __restrict__ pIn,
    const float * __restrict__ pKernel,
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
    const int64_t inGroupOffset,
    const int64_t outGroupOffset,
    const int64_t kernGroupOffset,
    const int64_t oPixels,
    const int64_t nY,
    const __vr    vry,
    const __vm256 vmw_s0,
    const __vm256 vmw_s2,
    const int64_t n,
    const int64_t k
)
{
  const __vr vrzerof = _vel_vbrds_vsl(0.f, VLEN) ;
  const __vr vrzero  = _vel_pvbrd_vsl(0UL, VLEN) ;

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

    int64_t outIndex  = outGroupOffset + (n * outChannel + k  ) * oPixels + op;

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


    const float *pKerValue = ( FLAYOUT == VEDNN_FILTER_LAYOUT_NCHW ) ?
  			     pKernel + kernGroupOffset + k * kernHeight * kernWidth  :
                             pKernel + kernGroupOffset + k ;

    const int64_t kernelDistance0 = ( FLAYOUT == VEDNN_FILTER_LAYOUT_NCHW ) ?
  	                            kernHeight * kernWidth :
  				    1 ;

    const int64_t kernelDistance1 = ( FLAYOUT == VEDNN_FILTER_LAYOUT_NCHW ) ?
  	                            1 :
  				    outChannelGroup ;

#define FILTER_R3S3(BIAS, N)										\
{													\
__vr vrsum = _vel_pvbrd_vsl(BIAS, vl) ;									\
const uint64_t kerValue_r0s0 = _vel_pack_f32p(pKerValue + 0*kernelDistance1 + 0*kernelDistance0,	\
				              pKerValue + 0*kernelDistance1 + 1*kernelDistance0);	\
const uint64_t kerValue_r0s1 = _vel_pack_f32p(pKerValue + 1*kernelDistance1 + 0*kernelDistance0,	\
					      pKerValue + 1*kernelDistance1 + 1*kernelDistance0);	\
const uint64_t kerValue_r0s2 = _vel_pack_f32p(pKerValue + 2*kernelDistance1 + 0*kernelDistance0,	\
					      pKerValue + 2*kernelDistance1 + 1*kernelDistance0);	\
const uint64_t kerValue_r1s0 = _vel_pack_f32p(pKerValue + 3*kernelDistance1 + 0*kernelDistance0,	\
					      pKerValue + 3*kernelDistance1 + 1*kernelDistance0);	\
const uint64_t kerValue_r1s1 = _vel_pack_f32p(pKerValue + 4*kernelDistance1 + 0*kernelDistance0,	\
					      pKerValue + 4*kernelDistance1 + 1*kernelDistance0);	\
const uint64_t kerValue_r1s2 = _vel_pack_f32p(pKerValue + 5*kernelDistance1 + 0*kernelDistance0,	\
					      pKerValue + 5*kernelDistance1 + 1*kernelDistance0);	\
const uint64_t kerValue_r2s0 = _vel_pack_f32p(pKerValue + 6*kernelDistance1 + 0*kernelDistance0,	\
					      pKerValue + 6*kernelDistance1 + 1*kernelDistance0);	\
const uint64_t kerValue_r2s1 = _vel_pack_f32p(pKerValue + 7*kernelDistance1 + 0*kernelDistance0,	\
					      pKerValue + 7*kernelDistance1 + 1*kernelDistance0);	\
const uint64_t kerValue_r2s2 = _vel_pack_f32p(pKerValue + 8*kernelDistance1 + 0*kernelDistance0,	\
					      pKerValue + 8*kernelDistance1 + 1*kernelDistance0);	\
vrsum = _vel_pvfmad_vvsvl(vrsum, kerValue_r0s0, vrinP_r0s0, vl) ;	\
vrsum = _vel_pvfmad_vvsvl(vrsum, kerValue_r0s1, vrinP_r0s1, vl) ;	\
vrsum = _vel_pvfmad_vvsvl(vrsum, kerValue_r0s2, vrinP_r0s2, vl) ;	\
vrsum = _vel_pvfmad_vvsvl(vrsum, kerValue_r1s0, vrinP_r1s0, vl) ;	\
vrsum = _vel_pvfmad_vvsvl(vrsum, kerValue_r1s1, vrinP_r1s1, vl) ;	\
vrsum = _vel_pvfmad_vvsvl(vrsum, kerValue_r1s2, vrinP_r1s2, vl) ;	\
vrsum = _vel_pvfmad_vvsvl(vrsum, kerValue_r2s0, vrinP_r2s0, vl) ;	\
vrsum = _vel_pvfmad_vvsvl(vrsum, kerValue_r2s1, vrinP_r2s1, vl) ;	\
vrsum = _vel_pvfmad_vvsvl(vrsum, kerValue_r2s2, vrinP_r2s2, vl) ;	\
_vel_vstu_vssl(vrsum, 4, pOut+outIndex+ (N)    *oPixels, vl) ;		\
_vel_vstl_vssl(vrsum, 4, pOut+outIndex+ ((N)+1)*oPixels, vl) ;		\
}
    FILTER_R3S3(0UL, 0) ;  pKerValue += 2*kernelDistance0 ;
#undef FILTER_R3S3
  } // outPixels
}

template<>
 inline void k2<VEDNN_FILTER_LAYOUT_HWCN, true>(
    const float * __restrict__ pIn,
    const float * __restrict__ pKernel,
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
    const int64_t inGroupOffset,
    const int64_t outGroupOffset,
    const int64_t kernGroupOffset,
    const int64_t oPixels,
    const int64_t nY,
    const __vr    vry,
    const __vm256 vmw_s0,
    const __vm256 vmw_s2,
    const int64_t n,
    const int64_t k
)
{
  const __vr vrzerof = _vel_vbrds_vsl(0.f, VLEN) ;
  const __vr vrzero  = _vel_pvbrd_vsl(0UL, VLEN) ;

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

    int64_t outIndex  = outGroupOffset + (n * outChannel + k  ) * oPixels + op;

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

#define FILTER_R3S3(VRBIAS, N)									\
{												\
__vr vrsum = (VRBIAS) ;										\
vrsum = _vel_pvfmad_vvsvl(vrsum, pKerValue_u64[(N/2)+0*outChannelGroup/2], vrinP_r0s0, vl) ;	\
vrsum = _vel_pvfmad_vvsvl(vrsum, pKerValue_u64[(N/2)+1*outChannelGroup/2], vrinP_r0s1, vl) ;	\
vrsum = _vel_pvfmad_vvsvl(vrsum, pKerValue_u64[(N/2)+2*outChannelGroup/2], vrinP_r0s2, vl) ;	\
vrsum = _vel_pvfmad_vvsvl(vrsum, pKerValue_u64[(N/2)+3*outChannelGroup/2], vrinP_r1s0, vl) ;	\
vrsum = _vel_pvfmad_vvsvl(vrsum, pKerValue_u64[(N/2)+4*outChannelGroup/2], vrinP_r1s1, vl) ;	\
vrsum = _vel_pvfmad_vvsvl(vrsum, pKerValue_u64[(N/2)+5*outChannelGroup/2], vrinP_r1s2, vl) ;	\
vrsum = _vel_pvfmad_vvsvl(vrsum, pKerValue_u64[(N/2)+6*outChannelGroup/2], vrinP_r2s0, vl) ;	\
vrsum = _vel_pvfmad_vvsvl(vrsum, pKerValue_u64[(N/2)+7*outChannelGroup/2], vrinP_r2s1, vl) ;	\
vrsum = _vel_pvfmad_vvsvl(vrsum, pKerValue_u64[(N/2)+8*outChannelGroup/2], vrinP_r2s2, vl) ;	\
_vel_vstl_vssl(vrsum, 4, pOut+outIndex+ ((N)  )*oPixels, vl) ;					\
_vel_vstu_vssl(vrsum, 4, pOut+outIndex+ ((N)+1)*oPixels, vl) ;					\
}
    FILTER_R3S3(vrzero, 0) ;
#undef FILTER_R3S3
  } // outPixels
}

template<>
 inline void k2<VEDNN_FILTER_LAYOUT_NCHW, true>(
    const float * __restrict__ pIn,
    const float * __restrict__ pKernel,
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
    const int64_t inGroupOffset,
    const int64_t outGroupOffset,
    const int64_t kernGroupOffset,
    const int64_t oPixels,
    const int64_t nY,
    const __vr    vry,
    const __vm256 vmw_s0,
    const __vm256 vmw_s2,
    const int64_t n,
    const int64_t k
)
{
  const __vr vrzerof = _vel_vbrds_vsl(0.f, VLEN) ;
  const __vr vrzero  = _vel_pvbrd_vsl(0UL, VLEN) ;

  uint64_t kerValue[9] ;
  const float *pKerValue  = pKernel + kernGroupOffset + (k * kernHeight    ) * kernWidth ;
  kerValue[0] = _vel_pack_f32p(pKerValue, pKerValue+9); pKerValue++ ;
  kerValue[1] = _vel_pack_f32p(pKerValue, pKerValue+9); pKerValue++ ;
  kerValue[2] = _vel_pack_f32p(pKerValue, pKerValue+9); pKerValue++ ;
  kerValue[3] = _vel_pack_f32p(pKerValue, pKerValue+9); pKerValue++ ;
  kerValue[4] = _vel_pack_f32p(pKerValue, pKerValue+9); pKerValue++ ;
  kerValue[5] = _vel_pack_f32p(pKerValue, pKerValue+9); pKerValue++ ;
  kerValue[6] = _vel_pack_f32p(pKerValue, pKerValue+9); pKerValue++ ;
  kerValue[7] = _vel_pack_f32p(pKerValue, pKerValue+9); pKerValue++ ;
  kerValue[8] = _vel_pack_f32p(pKerValue, pKerValue+9); pKerValue++ ;


  for (int64_t y=0; y<outHeight; y+=nY)
  {
    const int64_t vl = outWidth * (outHeight - y < nY ? outHeight - y : nY) ;
    const int64_t op = y * outWidth ;

    __vr vrh_r0 = _vel_vaddsl_vsvl(y  -1, vry, vl) ;
    __vr vrh_r2 = _vel_vaddsl_vsvl(y+2-1, vry, vl) ;

    __vm256 vm01_r0 =  _vel_vfmklge_mvl(vrh_r0, vl) ;
    __vm256 vm01_r2 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(inHeight,vrh_r2, vl), vl) ;

    __vm256 vmall_r0s0 = _vel_andm_mmm(vm01_r0,vmw_s0) ;
    __vm256 vmall_r0s1 = vm01_r0 ;
    __vm256 vmall_r0s2 = _vel_andm_mmm(vm01_r0, vmw_s2) ;

    __vm256 vmall_r1s0 = vmw_s0 ;
    __vm256 vmall_r1s2 = vmw_s2 ;

    __vm256 vmall_r2s0 = _vel_andm_mmm(vm01_r2,vmw_s0) ;
    __vm256 vmall_r2s1 = vm01_r2 ;
    __vm256 vmall_r2s2 = _vel_andm_mmm(vm01_r2, vmw_s2) ;


    int64_t outIndex = outGroupOffset + (n * outChannel + k  ) * oPixels + op ;

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

    __vr vrsum = vrzero ;
    vrsum = _vel_pvfmad_vvsvl(vrsum, kerValue[0], vrinP_r0s0, vl) ;
    vrsum = _vel_pvfmad_vvsvl(vrsum, kerValue[1], vrinP_r0s1, vl) ;
    vrsum = _vel_pvfmad_vvsvl(vrsum, kerValue[2], vrinP_r0s2, vl) ;
    vrsum = _vel_pvfmad_vvsvl(vrsum, kerValue[3], vrinP_r1s0, vl) ;
    vrsum = _vel_pvfmad_vvsvl(vrsum, kerValue[4], vrinP_r1s1, vl) ;
    vrsum = _vel_pvfmad_vvsvl(vrsum, kerValue[5], vrinP_r1s2, vl) ;
    vrsum = _vel_pvfmad_vvsvl(vrsum, kerValue[6], vrinP_r2s0, vl) ;
    vrsum = _vel_pvfmad_vvsvl(vrsum, kerValue[7], vrinP_r2s1, vl) ;
    vrsum = _vel_pvfmad_vvsvl(vrsum, kerValue[8], vrinP_r2s2, vl) ;
    _vel_vstu_vssl(vrsum, 4, pOut+outIndex, vl) ;
    _vel_vstl_vssl(vrsum, 4, pOut+outIndex+ oPixels, vl) ;

  } // outPixels
}


template<filterLayout_t FLAYOUT, bool useopt = false>
static inline void k4(
    const float * __restrict__ pIn,
    const float * __restrict__ pKernel,
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
    const int64_t inGroupOffset,
    const int64_t outGroupOffset,
    const int64_t kernGroupOffset,
    const int64_t oPixels,
    const int64_t nY,
    const __vr    vry,
    const __vm256 vmw_s0,
    const __vm256 vmw_s2,
    const int64_t n,
    const int64_t k
)
{
  const __vr vrzerof = _vel_vbrds_vsl(0.f, VLEN) ;
  const __vr vrzero  = _vel_pvbrd_vsl(0UL, VLEN) ;

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

    int64_t outIndex  = outGroupOffset + (n * outChannel + k  ) * oPixels + op;

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


    const float *pKerValue = ( FLAYOUT == VEDNN_FILTER_LAYOUT_NCHW ) ?
  			     pKernel + kernGroupOffset + k * kernHeight * kernWidth  :
                             pKernel + kernGroupOffset + k ;

    const int64_t kernelDistance0 = ( FLAYOUT == VEDNN_FILTER_LAYOUT_NCHW ) ?
  	                            kernHeight * kernWidth :
  				    1 ;

    const int64_t kernelDistance1 = ( FLAYOUT == VEDNN_FILTER_LAYOUT_NCHW ) ?
  	                            1 :
  				    outChannelGroup ;

#define FILTER_R3S3(BIAS, N)										\
{													\
__vr vrsum = _vel_pvbrd_vsl(BIAS, vl) ;									\
const uint64_t kerValue_r0s0 = _vel_pack_f32p(pKerValue + 0*kernelDistance1 + 0*kernelDistance0,	\
				              pKerValue + 0*kernelDistance1 + 1*kernelDistance0);	\
const uint64_t kerValue_r0s1 = _vel_pack_f32p(pKerValue + 1*kernelDistance1 + 0*kernelDistance0,	\
					      pKerValue + 1*kernelDistance1 + 1*kernelDistance0);	\
const uint64_t kerValue_r0s2 = _vel_pack_f32p(pKerValue + 2*kernelDistance1 + 0*kernelDistance0,	\
					      pKerValue + 2*kernelDistance1 + 1*kernelDistance0);	\
const uint64_t kerValue_r1s0 = _vel_pack_f32p(pKerValue + 3*kernelDistance1 + 0*kernelDistance0,	\
					      pKerValue + 3*kernelDistance1 + 1*kernelDistance0);	\
const uint64_t kerValue_r1s1 = _vel_pack_f32p(pKerValue + 4*kernelDistance1 + 0*kernelDistance0,	\
					      pKerValue + 4*kernelDistance1 + 1*kernelDistance0);	\
const uint64_t kerValue_r1s2 = _vel_pack_f32p(pKerValue + 5*kernelDistance1 + 0*kernelDistance0,	\
					      pKerValue + 5*kernelDistance1 + 1*kernelDistance0);	\
const uint64_t kerValue_r2s0 = _vel_pack_f32p(pKerValue + 6*kernelDistance1 + 0*kernelDistance0,	\
					      pKerValue + 6*kernelDistance1 + 1*kernelDistance0);	\
const uint64_t kerValue_r2s1 = _vel_pack_f32p(pKerValue + 7*kernelDistance1 + 0*kernelDistance0,	\
					      pKerValue + 7*kernelDistance1 + 1*kernelDistance0);	\
const uint64_t kerValue_r2s2 = _vel_pack_f32p(pKerValue + 8*kernelDistance1 + 0*kernelDistance0,	\
					      pKerValue + 8*kernelDistance1 + 1*kernelDistance0);	\
vrsum = _vel_pvfmad_vvsvl(vrsum, kerValue_r0s0, vrinP_r0s0, vl) ;	\
vrsum = _vel_pvfmad_vvsvl(vrsum, kerValue_r0s1, vrinP_r0s1, vl) ;	\
vrsum = _vel_pvfmad_vvsvl(vrsum, kerValue_r0s2, vrinP_r0s2, vl) ;	\
vrsum = _vel_pvfmad_vvsvl(vrsum, kerValue_r1s0, vrinP_r1s0, vl) ;	\
vrsum = _vel_pvfmad_vvsvl(vrsum, kerValue_r1s1, vrinP_r1s1, vl) ;	\
vrsum = _vel_pvfmad_vvsvl(vrsum, kerValue_r1s2, vrinP_r1s2, vl) ;	\
vrsum = _vel_pvfmad_vvsvl(vrsum, kerValue_r2s0, vrinP_r2s0, vl) ;	\
vrsum = _vel_pvfmad_vvsvl(vrsum, kerValue_r2s1, vrinP_r2s1, vl) ;	\
vrsum = _vel_pvfmad_vvsvl(vrsum, kerValue_r2s2, vrinP_r2s2, vl) ;	\
_vel_vstu_vssl(vrsum, 4, pOut+outIndex+ (N)    *oPixels, vl) ;		\
_vel_vstl_vssl(vrsum, 4, pOut+outIndex+ ((N)+1)*oPixels, vl) ;		\
}
    FILTER_R3S3(0UL, 0) ;  pKerValue += 2*kernelDistance0 ;
    FILTER_R3S3(0UL, 2) ;  pKerValue += 2*kernelDistance0 ;
#undef FILTER_R3S3
  } // outPixels
}

template<>
 inline void k4<VEDNN_FILTER_LAYOUT_HWCN, true>(
    const float * __restrict__ pIn,
    const float * __restrict__ pKernel,
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
    const int64_t inGroupOffset,
    const int64_t outGroupOffset,
    const int64_t kernGroupOffset,
    const int64_t oPixels,
    const int64_t nY,
    const __vr    vry,
    const __vm256 vmw_s0,
    const __vm256 vmw_s2,
    const int64_t n,
    const int64_t k
)
{
  const __vr vrzerof = _vel_vbrds_vsl(0.f, VLEN) ;
  const __vr vrzero  = _vel_pvbrd_vsl(0UL, VLEN) ;

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

    int64_t outIndex  = outGroupOffset + (n * outChannel + k  ) * oPixels + op;

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

#define FILTER_R3S3(VRBIAS, N)									\
{												\
__vr vrsum = (VRBIAS) ;										\
vrsum = _vel_pvfmad_vvsvl(vrsum, pKerValue_u64[(N/2)+0*outChannelGroup/2], vrinP_r0s0, vl) ;	\
vrsum = _vel_pvfmad_vvsvl(vrsum, pKerValue_u64[(N/2)+1*outChannelGroup/2], vrinP_r0s1, vl) ;	\
vrsum = _vel_pvfmad_vvsvl(vrsum, pKerValue_u64[(N/2)+2*outChannelGroup/2], vrinP_r0s2, vl) ;	\
vrsum = _vel_pvfmad_vvsvl(vrsum, pKerValue_u64[(N/2)+3*outChannelGroup/2], vrinP_r1s0, vl) ;	\
vrsum = _vel_pvfmad_vvsvl(vrsum, pKerValue_u64[(N/2)+4*outChannelGroup/2], vrinP_r1s1, vl) ;	\
vrsum = _vel_pvfmad_vvsvl(vrsum, pKerValue_u64[(N/2)+5*outChannelGroup/2], vrinP_r1s2, vl) ;	\
vrsum = _vel_pvfmad_vvsvl(vrsum, pKerValue_u64[(N/2)+6*outChannelGroup/2], vrinP_r2s0, vl) ;	\
vrsum = _vel_pvfmad_vvsvl(vrsum, pKerValue_u64[(N/2)+7*outChannelGroup/2], vrinP_r2s1, vl) ;	\
vrsum = _vel_pvfmad_vvsvl(vrsum, pKerValue_u64[(N/2)+8*outChannelGroup/2], vrinP_r2s2, vl) ;	\
_vel_vstl_vssl(vrsum, 4, pOut+outIndex+ ((N)  )*oPixels, vl) ;					\
_vel_vstu_vssl(vrsum, 4, pOut+outIndex+ ((N)+1)*oPixels, vl) ;					\
}
    FILTER_R3S3(vrzero, 0) ;
    FILTER_R3S3(vrzero, 2) ;
#undef FILTER_R3S3
  } // outPixels
}

template<>
 inline void k4<VEDNN_FILTER_LAYOUT_NCHW, true>(
    const float * __restrict__ pIn,
    const float * __restrict__ pKernel,
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
    const int64_t inGroupOffset,
    const int64_t outGroupOffset,
    const int64_t kernGroupOffset,
    const int64_t oPixels,
    const int64_t nY,
    const __vr    vry,
    const __vm256 vmw_s0,
    const __vm256 vmw_s2,
    const int64_t n,
    const int64_t k
)
{
  const __vr vrzerof = _vel_vbrds_vsl(0.f, VLEN) ;
  const __vr vrzero  = _vel_pvbrd_vsl(0UL, VLEN) ;

  uint64_t kerValue[2*9] ;
  {
    const float *pKerValue  = pKernel + kernGroupOffset + (k * kernHeight    ) * kernWidth ;
    __vr vrker0 = _vel_vldu_vssl(4, pKerValue    , 2*9) ;
    __vr vrker1 = _vel_vldu_vssl(4, pKerValue+2*9, 2*9) ;
    _vel_vst_vssl(_vel_vshf_vvvsl(vrker0, vrker1, VE_VSHUFFLE_YUZU, 2*9) ,8, kerValue, 2*9) ;
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

    int64_t outIndex  = outGroupOffset + (n * outChannel + k  ) * oPixels + op;

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

#define FILTER_R3S3(VRBIAS, N)						\
{									\
__vr vrsum = (VRBIAS) ;							\
vrsum = _vel_pvfmad_vvsvl(vrsum, kerValue[9*(N)+0], vrinP_r0s0, vl) ;	\
vrsum = _vel_pvfmad_vvsvl(vrsum, kerValue[9*(N)+1], vrinP_r0s1, vl) ;	\
vrsum = _vel_pvfmad_vvsvl(vrsum, kerValue[9*(N)+2], vrinP_r0s2, vl) ;	\
vrsum = _vel_pvfmad_vvsvl(vrsum, kerValue[9*(N)+3], vrinP_r1s0, vl) ;	\
vrsum = _vel_pvfmad_vvsvl(vrsum, kerValue[9*(N)+4], vrinP_r1s1, vl) ;	\
vrsum = _vel_pvfmad_vvsvl(vrsum, kerValue[9*(N)+5], vrinP_r1s2, vl) ;	\
vrsum = _vel_pvfmad_vvsvl(vrsum, kerValue[9*(N)+6], vrinP_r2s0, vl) ;	\
vrsum = _vel_pvfmad_vvsvl(vrsum, kerValue[9*(N)+7], vrinP_r2s1, vl) ;	\
vrsum = _vel_pvfmad_vvsvl(vrsum, kerValue[9*(N)+8], vrinP_r2s2, vl) ;	\
_vel_vstu_vssl(vrsum, 4, pOut+outIndex+ ((N)  )*oPixels, vl) ;		\
_vel_vstl_vssl(vrsum, 4, pOut+outIndex+ ((N)+2)*oPixels, vl) ;		\
}
    FILTER_R3S3(vrzero, 0) ;
    FILTER_R3S3(vrzero, 1) ;
#undef FILTER_R3S3
  } // outPixels
}



template<filterLayout_t FLAYOUT, bool useopt = false>
static inline void k8(
    const float * __restrict__ pIn,
    const float * __restrict__ pKernel,
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
    const int64_t inGroupOffset,
    const int64_t outGroupOffset,
    const int64_t kernGroupOffset,
    const int64_t oPixels,
    const int64_t nY,
    const __vr    vry,
    const __vm256 vmw_s0,
    const __vm256 vmw_s2,
    const int64_t n,
    const int64_t k
)
{
  const __vr vrzerof = _vel_vbrds_vsl(0.f, VLEN) ;
  const __vr vrzero  = _vel_pvbrd_vsl(0UL, VLEN) ;

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

    int64_t outIndex  = outGroupOffset + (n * outChannel + k  ) * oPixels + op;

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


    const float *pKerValue = ( FLAYOUT == VEDNN_FILTER_LAYOUT_NCHW ) ?
  			     pKernel + kernGroupOffset + k * kernHeight * kernWidth  :
                             pKernel + kernGroupOffset + k ;

    const int64_t kernelDistance0 = ( FLAYOUT == VEDNN_FILTER_LAYOUT_NCHW ) ?
  	                            kernHeight * kernWidth :
  				    1 ;

    const int64_t kernelDistance1 = ( FLAYOUT == VEDNN_FILTER_LAYOUT_NCHW ) ?
  	                            1 :
  				    outChannelGroup ;

#define FILTER_R3S3(BIAS, N)										\
{													\
__vr vrsum = _vel_pvbrd_vsl(BIAS, vl) ;									\
const uint64_t kerValue_r0s0 = _vel_pack_f32p(pKerValue + 0*kernelDistance1 + 0*kernelDistance0,	\
				              pKerValue + 0*kernelDistance1 + 1*kernelDistance0);	\
const uint64_t kerValue_r0s1 = _vel_pack_f32p(pKerValue + 1*kernelDistance1 + 0*kernelDistance0,	\
					      pKerValue + 1*kernelDistance1 + 1*kernelDistance0);	\
const uint64_t kerValue_r0s2 = _vel_pack_f32p(pKerValue + 2*kernelDistance1 + 0*kernelDistance0,	\
					      pKerValue + 2*kernelDistance1 + 1*kernelDistance0);	\
const uint64_t kerValue_r1s0 = _vel_pack_f32p(pKerValue + 3*kernelDistance1 + 0*kernelDistance0,	\
					      pKerValue + 3*kernelDistance1 + 1*kernelDistance0);	\
const uint64_t kerValue_r1s1 = _vel_pack_f32p(pKerValue + 4*kernelDistance1 + 0*kernelDistance0,	\
					      pKerValue + 4*kernelDistance1 + 1*kernelDistance0);	\
const uint64_t kerValue_r1s2 = _vel_pack_f32p(pKerValue + 5*kernelDistance1 + 0*kernelDistance0,	\
					      pKerValue + 5*kernelDistance1 + 1*kernelDistance0);	\
const uint64_t kerValue_r2s0 = _vel_pack_f32p(pKerValue + 6*kernelDistance1 + 0*kernelDistance0,	\
					      pKerValue + 6*kernelDistance1 + 1*kernelDistance0);	\
const uint64_t kerValue_r2s1 = _vel_pack_f32p(pKerValue + 7*kernelDistance1 + 0*kernelDistance0,	\
					      pKerValue + 7*kernelDistance1 + 1*kernelDistance0);	\
const uint64_t kerValue_r2s2 = _vel_pack_f32p(pKerValue + 8*kernelDistance1 + 0*kernelDistance0,	\
					      pKerValue + 8*kernelDistance1 + 1*kernelDistance0);	\
vrsum = _vel_pvfmad_vvsvl(vrsum, kerValue_r0s0, vrinP_r0s0, vl) ;	\
vrsum = _vel_pvfmad_vvsvl(vrsum, kerValue_r0s1, vrinP_r0s1, vl) ;	\
vrsum = _vel_pvfmad_vvsvl(vrsum, kerValue_r0s2, vrinP_r0s2, vl) ;	\
vrsum = _vel_pvfmad_vvsvl(vrsum, kerValue_r1s0, vrinP_r1s0, vl) ;	\
vrsum = _vel_pvfmad_vvsvl(vrsum, kerValue_r1s1, vrinP_r1s1, vl) ;	\
vrsum = _vel_pvfmad_vvsvl(vrsum, kerValue_r1s2, vrinP_r1s2, vl) ;	\
vrsum = _vel_pvfmad_vvsvl(vrsum, kerValue_r2s0, vrinP_r2s0, vl) ;	\
vrsum = _vel_pvfmad_vvsvl(vrsum, kerValue_r2s1, vrinP_r2s1, vl) ;	\
vrsum = _vel_pvfmad_vvsvl(vrsum, kerValue_r2s2, vrinP_r2s2, vl) ;	\
_vel_vstu_vssl(vrsum, 4, pOut+outIndex+ (N)    *oPixels, vl) ;		\
_vel_vstl_vssl(vrsum, 4, pOut+outIndex+ ((N)+1)*oPixels, vl) ;		\
}
    FILTER_R3S3(0UL, 0) ;  pKerValue += 2*kernelDistance0 ;
    FILTER_R3S3(0UL, 2) ;  pKerValue += 2*kernelDistance0 ;
    FILTER_R3S3(0UL, 4) ;  pKerValue += 2*kernelDistance0 ;
    FILTER_R3S3(0UL, 6) ;  pKerValue += 2*kernelDistance0 ;
#undef FILTER_R3S3
  } // outPixels
}

template<>
 inline void k8<VEDNN_FILTER_LAYOUT_HWCN, true>(
    const float * __restrict__ pIn,
    const float * __restrict__ pKernel,
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
    const int64_t inGroupOffset,
    const int64_t outGroupOffset,
    const int64_t kernGroupOffset,
    const int64_t oPixels,
    const int64_t nY,
    const __vr    vry,
    const __vm256 vmw_s0,
    const __vm256 vmw_s2,
    const int64_t n,
    const int64_t k
)
{
  const __vr vrzerof = _vel_vbrds_vsl(0.f, VLEN) ;
  const __vr vrzero  = _vel_pvbrd_vsl(0UL, VLEN) ;

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

    int64_t outIndex  = outGroupOffset + (n * outChannel + k  ) * oPixels + op;

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

#define FILTER_R3S3(VRBIAS, N)									\
{												\
__vr vrsum = (VRBIAS) ;										\
vrsum = _vel_pvfmad_vvsvl(vrsum, pKerValue_u64[(N/2)+0*outChannelGroup/2], vrinP_r0s0, vl) ;	\
vrsum = _vel_pvfmad_vvsvl(vrsum, pKerValue_u64[(N/2)+1*outChannelGroup/2], vrinP_r0s1, vl) ;	\
vrsum = _vel_pvfmad_vvsvl(vrsum, pKerValue_u64[(N/2)+2*outChannelGroup/2], vrinP_r0s2, vl) ;	\
vrsum = _vel_pvfmad_vvsvl(vrsum, pKerValue_u64[(N/2)+3*outChannelGroup/2], vrinP_r1s0, vl) ;	\
vrsum = _vel_pvfmad_vvsvl(vrsum, pKerValue_u64[(N/2)+4*outChannelGroup/2], vrinP_r1s1, vl) ;	\
vrsum = _vel_pvfmad_vvsvl(vrsum, pKerValue_u64[(N/2)+5*outChannelGroup/2], vrinP_r1s2, vl) ;	\
vrsum = _vel_pvfmad_vvsvl(vrsum, pKerValue_u64[(N/2)+6*outChannelGroup/2], vrinP_r2s0, vl) ;	\
vrsum = _vel_pvfmad_vvsvl(vrsum, pKerValue_u64[(N/2)+7*outChannelGroup/2], vrinP_r2s1, vl) ;	\
vrsum = _vel_pvfmad_vvsvl(vrsum, pKerValue_u64[(N/2)+8*outChannelGroup/2], vrinP_r2s2, vl) ;	\
_vel_vstl_vssl(vrsum, 4, pOut+outIndex+ ((N)  )*oPixels, vl) ;					\
_vel_vstu_vssl(vrsum, 4, pOut+outIndex+ ((N)+1)*oPixels, vl) ;					\
}
    FILTER_R3S3(vrzero, 0) ;
    FILTER_R3S3(vrzero, 2) ;
    FILTER_R3S3(vrzero, 4) ;
    FILTER_R3S3(vrzero, 6) ;
#undef FILTER_R3S3
  } // outPixels
}

template<>
 inline void k8<VEDNN_FILTER_LAYOUT_NCHW, true>(
    const float * __restrict__ pIn,
    const float * __restrict__ pKernel,
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
    const int64_t inGroupOffset,
    const int64_t outGroupOffset,
    const int64_t kernGroupOffset,
    const int64_t oPixels,
    const int64_t nY,
    const __vr    vry,
    const __vm256 vmw_s0,
    const __vm256 vmw_s2,
    const int64_t n,
    const int64_t k
)
{
  const __vr vrzerof = _vel_vbrds_vsl(0.f, VLEN) ;
  const __vr vrzero  = _vel_pvbrd_vsl(0UL, VLEN) ;

  uint64_t kerValue[4*9] ;
  {
    const float *pKerValue  = pKernel + kernGroupOffset + (k * kernHeight    ) * kernWidth ;
    __vr vrker0 = _vel_vldu_vssl(4, pKerValue    , 4*9) ;
    __vr vrker1 = _vel_vldu_vssl(4, pKerValue+4*9, 4*9) ;
    _vel_vst_vssl(_vel_vshf_vvvsl(vrker0, vrker1, VE_VSHUFFLE_YUZU, 4*9) ,8, kerValue, 4*9) ;
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

    int64_t outIndex  = outGroupOffset + (n * outChannel + k  ) * oPixels + op;

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

#define FILTER_R3S3(VRBIAS, N)						\
{									\
__vr vrsum = (VRBIAS) ;							\
vrsum = _vel_pvfmad_vvsvl(vrsum, kerValue[9*(N)+0], vrinP_r0s0, vl) ;	\
vrsum = _vel_pvfmad_vvsvl(vrsum, kerValue[9*(N)+1], vrinP_r0s1, vl) ;	\
vrsum = _vel_pvfmad_vvsvl(vrsum, kerValue[9*(N)+2], vrinP_r0s2, vl) ;	\
vrsum = _vel_pvfmad_vvsvl(vrsum, kerValue[9*(N)+3], vrinP_r1s0, vl) ;	\
vrsum = _vel_pvfmad_vvsvl(vrsum, kerValue[9*(N)+4], vrinP_r1s1, vl) ;	\
vrsum = _vel_pvfmad_vvsvl(vrsum, kerValue[9*(N)+5], vrinP_r1s2, vl) ;	\
vrsum = _vel_pvfmad_vvsvl(vrsum, kerValue[9*(N)+6], vrinP_r2s0, vl) ;	\
vrsum = _vel_pvfmad_vvsvl(vrsum, kerValue[9*(N)+7], vrinP_r2s1, vl) ;	\
vrsum = _vel_pvfmad_vvsvl(vrsum, kerValue[9*(N)+8], vrinP_r2s2, vl) ;	\
_vel_vstu_vssl(vrsum, 4, pOut+outIndex+ ((N)  )*oPixels, vl) ;		\
_vel_vstl_vssl(vrsum, 4, pOut+outIndex+ ((N)+4)*oPixels, vl) ;		\
}
    FILTER_R3S3(vrzero, 0) ;
    FILTER_R3S3(vrzero, 1) ;
    FILTER_R3S3(vrzero, 2) ;
    FILTER_R3S3(vrzero, 3) ;
#undef FILTER_R3S3
  } // outPixels
}


template<filterLayout_t FLAYOUT, bool useopt = false>
static inline void k16(
    const float * __restrict__ pIn,
    const float * __restrict__ pKernel,
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
    const int64_t inGroupOffset,
    const int64_t outGroupOffset,
    const int64_t kernGroupOffset,
    const int64_t oPixels,
    const int64_t nY,
    const __vr    vry,
    const __vm256 vmw_s0,
    const __vm256 vmw_s2,
    const int64_t n,
    const int64_t k
)
{
  const __vr vrzerof = _vel_vbrds_vsl(0.f, VLEN) ;
  const __vr vrzero  = _vel_pvbrd_vsl(0UL, VLEN) ;

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

    int64_t outIndex  = outGroupOffset + (n * outChannel + k  ) * oPixels + op;

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


    const float *pKerValue = ( FLAYOUT == VEDNN_FILTER_LAYOUT_NCHW ) ?
  			     pKernel + kernGroupOffset + k * kernHeight * kernWidth  :
                             pKernel + kernGroupOffset + k ;

    const int64_t kernelDistance0 = ( FLAYOUT == VEDNN_FILTER_LAYOUT_NCHW ) ?
  	                            kernHeight * kernWidth :
  				    1 ;

    const int64_t kernelDistance1 = ( FLAYOUT == VEDNN_FILTER_LAYOUT_NCHW ) ?
  	                            1 :
  				    outChannelGroup ;

#define FILTER_R3S3(BIAS, N)										\
{													\
__vr vrsum = _vel_pvbrd_vsl(BIAS, vl) ;									\
const uint64_t kerValue_r0s0 = _vel_pack_f32p(pKerValue + 0*kernelDistance1 + 0*kernelDistance0,	\
				              pKerValue + 0*kernelDistance1 + 1*kernelDistance0);	\
const uint64_t kerValue_r0s1 = _vel_pack_f32p(pKerValue + 1*kernelDistance1 + 0*kernelDistance0,	\
					      pKerValue + 1*kernelDistance1 + 1*kernelDistance0);	\
const uint64_t kerValue_r0s2 = _vel_pack_f32p(pKerValue + 2*kernelDistance1 + 0*kernelDistance0,	\
					      pKerValue + 2*kernelDistance1 + 1*kernelDistance0);	\
const uint64_t kerValue_r1s0 = _vel_pack_f32p(pKerValue + 3*kernelDistance1 + 0*kernelDistance0,	\
					      pKerValue + 3*kernelDistance1 + 1*kernelDistance0);	\
const uint64_t kerValue_r1s1 = _vel_pack_f32p(pKerValue + 4*kernelDistance1 + 0*kernelDistance0,	\
					      pKerValue + 4*kernelDistance1 + 1*kernelDistance0);	\
const uint64_t kerValue_r1s2 = _vel_pack_f32p(pKerValue + 5*kernelDistance1 + 0*kernelDistance0,	\
					      pKerValue + 5*kernelDistance1 + 1*kernelDistance0);	\
const uint64_t kerValue_r2s0 = _vel_pack_f32p(pKerValue + 6*kernelDistance1 + 0*kernelDistance0,	\
					      pKerValue + 6*kernelDistance1 + 1*kernelDistance0);	\
const uint64_t kerValue_r2s1 = _vel_pack_f32p(pKerValue + 7*kernelDistance1 + 0*kernelDistance0,	\
					      pKerValue + 7*kernelDistance1 + 1*kernelDistance0);	\
const uint64_t kerValue_r2s2 = _vel_pack_f32p(pKerValue + 8*kernelDistance1 + 0*kernelDistance0,	\
					      pKerValue + 8*kernelDistance1 + 1*kernelDistance0);	\
vrsum = _vel_pvfmad_vvsvl(vrsum, kerValue_r0s0, vrinP_r0s0, vl) ;	\
vrsum = _vel_pvfmad_vvsvl(vrsum, kerValue_r0s1, vrinP_r0s1, vl) ;	\
vrsum = _vel_pvfmad_vvsvl(vrsum, kerValue_r0s2, vrinP_r0s2, vl) ;	\
vrsum = _vel_pvfmad_vvsvl(vrsum, kerValue_r1s0, vrinP_r1s0, vl) ;	\
vrsum = _vel_pvfmad_vvsvl(vrsum, kerValue_r1s1, vrinP_r1s1, vl) ;	\
vrsum = _vel_pvfmad_vvsvl(vrsum, kerValue_r1s2, vrinP_r1s2, vl) ;	\
vrsum = _vel_pvfmad_vvsvl(vrsum, kerValue_r2s0, vrinP_r2s0, vl) ;	\
vrsum = _vel_pvfmad_vvsvl(vrsum, kerValue_r2s1, vrinP_r2s1, vl) ;	\
vrsum = _vel_pvfmad_vvsvl(vrsum, kerValue_r2s2, vrinP_r2s2, vl) ;	\
_vel_vstu_vssl(vrsum, 4, pOut+outIndex+ (N)    *oPixels, vl) ;		\
_vel_vstl_vssl(vrsum, 4, pOut+outIndex+ ((N)+1)*oPixels, vl) ;		\
}
    FILTER_R3S3(0UL, 0) ;  pKerValue += 2*kernelDistance0 ;
    FILTER_R3S3(0UL, 2) ;  pKerValue += 2*kernelDistance0 ;
    FILTER_R3S3(0UL, 4) ;  pKerValue += 2*kernelDistance0 ;
    FILTER_R3S3(0UL, 6) ;  pKerValue += 2*kernelDistance0 ;
    FILTER_R3S3(0UL, 8) ;  pKerValue += 2*kernelDistance0 ;
    FILTER_R3S3(0UL,10) ;  pKerValue += 2*kernelDistance0 ;
    FILTER_R3S3(0UL,12) ;  pKerValue += 2*kernelDistance0 ;
    FILTER_R3S3(0UL,14) ;  pKerValue += 2*kernelDistance0 ;
#undef FILTER_R3S3
  } // outPixels
}

template<>
 inline void k16<VEDNN_FILTER_LAYOUT_HWCN, true>(
    const float * __restrict__ pIn,
    const float * __restrict__ pKernel,
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
    const int64_t inGroupOffset,
    const int64_t outGroupOffset,
    const int64_t kernGroupOffset,
    const int64_t oPixels,
    const int64_t nY,
    const __vr    vry,
    const __vm256 vmw_s0,
    const __vm256 vmw_s2,
    const int64_t n,
    const int64_t k
)
{
  const __vr vrzerof = _vel_vbrds_vsl(0.f, VLEN) ;
  const __vr vrzero  = _vel_pvbrd_vsl(0UL, VLEN) ;

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

    int64_t outIndex  = outGroupOffset + (n * outChannel + k  ) * oPixels + op;

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

#define FILTER_R3S3(VRBIAS, N)									\
{												\
__vr vrsum = (VRBIAS) ;										\
vrsum = _vel_pvfmad_vvsvl(vrsum, pKerValue_u64[(N/2)+0*outChannelGroup/2], vrinP_r0s0, vl) ;	\
vrsum = _vel_pvfmad_vvsvl(vrsum, pKerValue_u64[(N/2)+1*outChannelGroup/2], vrinP_r0s1, vl) ;	\
vrsum = _vel_pvfmad_vvsvl(vrsum, pKerValue_u64[(N/2)+2*outChannelGroup/2], vrinP_r0s2, vl) ;	\
vrsum = _vel_pvfmad_vvsvl(vrsum, pKerValue_u64[(N/2)+3*outChannelGroup/2], vrinP_r1s0, vl) ;	\
vrsum = _vel_pvfmad_vvsvl(vrsum, pKerValue_u64[(N/2)+4*outChannelGroup/2], vrinP_r1s1, vl) ;	\
vrsum = _vel_pvfmad_vvsvl(vrsum, pKerValue_u64[(N/2)+5*outChannelGroup/2], vrinP_r1s2, vl) ;	\
vrsum = _vel_pvfmad_vvsvl(vrsum, pKerValue_u64[(N/2)+6*outChannelGroup/2], vrinP_r2s0, vl) ;	\
vrsum = _vel_pvfmad_vvsvl(vrsum, pKerValue_u64[(N/2)+7*outChannelGroup/2], vrinP_r2s1, vl) ;	\
vrsum = _vel_pvfmad_vvsvl(vrsum, pKerValue_u64[(N/2)+8*outChannelGroup/2], vrinP_r2s2, vl) ;	\
_vel_vstl_vssl(vrsum, 4, pOut+outIndex+ ((N)  )*oPixels, vl) ;					\
_vel_vstu_vssl(vrsum, 4, pOut+outIndex+ ((N)+1)*oPixels, vl) ;					\
}
    FILTER_R3S3(vrzero, 0) ;
    FILTER_R3S3(vrzero, 2) ;
    FILTER_R3S3(vrzero, 4) ;
    FILTER_R3S3(vrzero, 6) ;
    FILTER_R3S3(vrzero, 8) ;
    FILTER_R3S3(vrzero, 10) ;
    FILTER_R3S3(vrzero, 12) ;
    FILTER_R3S3(vrzero, 14) ;
#undef FILTER_R3S3
  } // outPixels
}

template<>
 inline void k16<VEDNN_FILTER_LAYOUT_NCHW, true>(
    const float * __restrict__ pIn,
    const float * __restrict__ pKernel,
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
    const int64_t inGroupOffset,
    const int64_t outGroupOffset,
    const int64_t kernGroupOffset,
    const int64_t oPixels,
    const int64_t nY,
    const __vr    vry,
    const __vm256 vmw_s0,
    const __vm256 vmw_s2,
    const int64_t n,
    const int64_t k
)
{
  const __vr vrzerof = _vel_vbrds_vsl(0.f, VLEN) ;
  const __vr vrzero  = _vel_pvbrd_vsl(0UL, VLEN) ;

  uint64_t kerValue[8*9] ;
  {
    const float *pKerValue  = pKernel + kernGroupOffset + (k * kernHeight    ) * kernWidth ;
    __vr vrker0 = _vel_vldu_vssl(4, pKerValue    , 8*9) ;
    __vr vrker1 = _vel_vldu_vssl(4, pKerValue+8*9, 8*9) ;
    _vel_vst_vssl(_vel_vshf_vvvsl(vrker0, vrker1, VE_VSHUFFLE_YUZU, 8*9) ,8, kerValue, 8*9) ;
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

    int64_t outIndex  = outGroupOffset + (n * outChannel + k  ) * oPixels + op;

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

#define FILTER_R3S3(VRBIAS, N)						\
{									\
__vr vrsum = (VRBIAS) ;							\
vrsum = _vel_pvfmad_vvsvl(vrsum, kerValue[9*(N)+0], vrinP_r0s0, vl) ;	\
vrsum = _vel_pvfmad_vvsvl(vrsum, kerValue[9*(N)+1], vrinP_r0s1, vl) ;	\
vrsum = _vel_pvfmad_vvsvl(vrsum, kerValue[9*(N)+2], vrinP_r0s2, vl) ;	\
vrsum = _vel_pvfmad_vvsvl(vrsum, kerValue[9*(N)+3], vrinP_r1s0, vl) ;	\
vrsum = _vel_pvfmad_vvsvl(vrsum, kerValue[9*(N)+4], vrinP_r1s1, vl) ;	\
vrsum = _vel_pvfmad_vvsvl(vrsum, kerValue[9*(N)+5], vrinP_r1s2, vl) ;	\
vrsum = _vel_pvfmad_vvsvl(vrsum, kerValue[9*(N)+6], vrinP_r2s0, vl) ;	\
vrsum = _vel_pvfmad_vvsvl(vrsum, kerValue[9*(N)+7], vrinP_r2s1, vl) ;	\
vrsum = _vel_pvfmad_vvsvl(vrsum, kerValue[9*(N)+8], vrinP_r2s2, vl) ;	\
_vel_vstu_vssl(vrsum, 4, pOut+outIndex+ ((N)  )*oPixels, vl) ;		\
_vel_vstl_vssl(vrsum, 4, pOut+outIndex+ ((N)+8)*oPixels, vl) ;		\
}
    FILTER_R3S3(vrzero, 0) ;
    FILTER_R3S3(vrzero, 1) ;
    FILTER_R3S3(vrzero, 2) ;
    FILTER_R3S3(vrzero, 3) ;
    FILTER_R3S3(vrzero, 4) ;
    FILTER_R3S3(vrzero, 5) ;
    FILTER_R3S3(vrzero, 6) ;
    FILTER_R3S3(vrzero, 7) ;
#undef FILTER_R3S3
  } // outPixels
}

extern "C" vednnError_t
vednnConvolutionForward_direct_dil1_str1_padsame_ker3_c1_owU128(
    const vednnTensorParam_t *  	pParamIn,
    const void *  			pDataIn,
    const vednnFilterParam_t *  	pParamKernel,
    const void *  			pDataKernel,
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
  float * const pOut    = (float * const) pDataOut;

  const int oPixels= outHeight*outWidth ;

  {
    const int64_t nY = VLEN / outWidth ;

    __vr vrseq = _vel_vseq_vl(VLEN) ;			// xy

    __vr vry   = _vel_vdivsl_vvsl(vrseq, outWidth, VLEN) ;
    __vr vrx   = _vel_vsubsl_vvvl(vrseq, _vel_vmulul_vsvl(outWidth,vry, VLEN), VLEN) ;

    __vr vrw_s0 = _vel_vaddsl_vsvl( -padWidth, vrx, VLEN) ;
    __vr vrw_s2 = _vel_vaddsl_vsvl(2-padWidth, vrx, VLEN) ;

    __vm256 vmw_s0  =  _vel_vfmklge_mvl(vrw_s0, VLEN) ;
    __vm256 vmw_s2  =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(inWidth,vrw_s2, VLEN), VLEN) ;

    for (int64_t n = 0; n < batch; n++) {
      for (int64_t g = 0; g < group; g++) {

	const int64_t inGroupOffset   = g * inHeight * inWidth;
	const int64_t outGroupOffset  = g * outChannelGroup * outHeight * outWidth;
	const int64_t kernGroupOffset = g * outChannelGroup * kernHeight * kernWidth;

	int k = 0 ;
	if ( (outChannelGroup & 0x01) == 1 ) {
	  if( filter_layout == VEDNN_FILTER_LAYOUT_NCHW) {
	    k1<VEDNN_FILTER_LAYOUT_NCHW>(pIn, pKernel, pOut,
	       inChannel, inWidth, inHeight,
	       outChannel, outWidth, outHeight,
	       kernWidth, kernHeight,
	       outChannelGroup,
	       inGroupOffset, outGroupOffset,
	       kernGroupOffset, oPixels,
	       nY,
	       vry, vmw_s0, vmw_s2,
	       n, k) ;
	  }
	  else {
	    k1<VEDNN_FILTER_LAYOUT_HWCN>(pIn, pKernel, pOut,
	       inChannel, inWidth, inHeight,
	       outChannel, outWidth, outHeight,
	       kernWidth, kernHeight,
	       outChannelGroup,
	       inGroupOffset, outGroupOffset,
	       kernGroupOffset, oPixels,
	       nY,
	       vry, vmw_s0, vmw_s2,
	       n, k) ;
	  }
	  k++ ;
	}
	if ( ((outChannelGroup >> 1) & 0x01) == 1 ) {
	  if( filter_layout == VEDNN_FILTER_LAYOUT_NCHW) {
	    k2<VEDNN_FILTER_LAYOUT_NCHW, true>(pIn, pKernel, pOut,
	       inChannel, inWidth, inHeight,
	       outChannel, outWidth, outHeight,
	       kernWidth, kernHeight,
	       outChannelGroup,
	       inGroupOffset, outGroupOffset,
	       kernGroupOffset, oPixels,
	       nY,
	       vry, vmw_s0, vmw_s2,
	       n, k) ;
	  }
	  else {
	    if( (((uint64_t)(pKernel + kernGroupOffset + k)) & 0x7) == 0
		&& (outChannelGroup & 0x1) == 0 )
	    {
	      k2<VEDNN_FILTER_LAYOUT_HWCN, true>(pIn, pKernel, pOut,
		 inChannel, inWidth, inHeight,
		 outChannel, outWidth, outHeight,
		 kernWidth, kernHeight,
		 outChannelGroup,
		 inGroupOffset, outGroupOffset,
		 kernGroupOffset, oPixels,
		 nY,
		 vry, vmw_s0, vmw_s2,
		 n, k) ;
	    }
	    else {
	      k2<VEDNN_FILTER_LAYOUT_HWCN, false>(pIn, pKernel, pOut,
		 inChannel, inWidth, inHeight,
		 outChannel, outWidth, outHeight,
		 kernWidth, kernHeight,
		 outChannelGroup,
		 inGroupOffset, outGroupOffset,
		 kernGroupOffset, oPixels,
		 nY,
		 vry, vmw_s0, vmw_s2,
		 n, k) ;
	    }
	  }
	  k+=2 ;
	}
	if ( ((outChannelGroup >> 2) & 0x01) == 1 ) {
	  if( filter_layout == VEDNN_FILTER_LAYOUT_NCHW) {
	    k4<VEDNN_FILTER_LAYOUT_NCHW, true>(pIn, pKernel, pOut,
	       inChannel, inWidth, inHeight,
	       outChannel, outWidth, outHeight,
	       kernWidth, kernHeight,
	       outChannelGroup,
	       inGroupOffset, outGroupOffset,
	       kernGroupOffset, oPixels,
	       nY,
	       vry, vmw_s0, vmw_s2,
	       n, k) ;
	  }
	  else {
	    if( (((uint64_t)(pKernel + kernGroupOffset + k)) & 0x7) == 0
		&& (outChannelGroup & 0x1) == 0 )
	    {
	      k4<VEDNN_FILTER_LAYOUT_HWCN, true>(pIn, pKernel, pOut,
		 inChannel, inWidth, inHeight,
		 outChannel, outWidth, outHeight,
		 kernWidth, kernHeight,
		 outChannelGroup,
		 inGroupOffset, outGroupOffset,
		 kernGroupOffset, oPixels,
		 nY,
		 vry, vmw_s0, vmw_s2,
		 n, k) ;
	    }
	    else {
	      k4<VEDNN_FILTER_LAYOUT_HWCN, false>(pIn, pKernel, pOut,
		 inChannel, inWidth, inHeight,
		 outChannel, outWidth, outHeight,
		 kernWidth, kernHeight,
		 outChannelGroup,
		 inGroupOffset, outGroupOffset,
		 kernGroupOffset, oPixels,
		 nY,
		 vry, vmw_s0, vmw_s2,
		 n, k) ;
	    }
	  }

	  k+=4 ;
	}
	if ( ((outChannelGroup >> 3) & 0x01) == 1 ) {
	  if( filter_layout == VEDNN_FILTER_LAYOUT_NCHW) {
	    k8<VEDNN_FILTER_LAYOUT_NCHW, true>(pIn, pKernel, pOut,
	       inChannel, inWidth, inHeight,
	       outChannel, outWidth, outHeight,
	       kernWidth, kernHeight,
	       outChannelGroup,
	       inGroupOffset, outGroupOffset,
	       kernGroupOffset, oPixels,
	       nY,
	       vry, vmw_s0, vmw_s2,
	       n, k) ;
	  }
	  else {
	    if( (((uint64_t)(pKernel + kernGroupOffset + k)) & 0x7) == 0
		&& (outChannelGroup & 0x1) == 0 )
	    {
	      k8<VEDNN_FILTER_LAYOUT_HWCN, true>(pIn, pKernel, pOut,
		 inChannel, inWidth, inHeight,
		 outChannel, outWidth, outHeight,
		 kernWidth, kernHeight,
		 outChannelGroup,
		 inGroupOffset, outGroupOffset,
		 kernGroupOffset, oPixels,
		 nY,
		 vry, vmw_s0, vmw_s2,
		 n, k) ;
	    }
	    else {
	      k8<VEDNN_FILTER_LAYOUT_HWCN, false>(pIn, pKernel, pOut,
		 inChannel, inWidth, inHeight,
		 outChannel, outWidth, outHeight,
		 kernWidth, kernHeight,
		 outChannelGroup,
		 inGroupOffset, outGroupOffset,
		 kernGroupOffset, oPixels,
		 nY,
		 vry, vmw_s0, vmw_s2,
		 n, k) ;
	    }
	  }
	  k+=8 ;
	}
	for ( ; k < outChannelGroup; k+=16) {
	  if( filter_layout == VEDNN_FILTER_LAYOUT_NCHW) {
	    k16<VEDNN_FILTER_LAYOUT_NCHW, true>(pIn, pKernel, pOut,
	       inChannel, inWidth, inHeight,
	       outChannel, outWidth, outHeight,
	       kernWidth, kernHeight,
	       outChannelGroup,
	       inGroupOffset, outGroupOffset,
	       kernGroupOffset, oPixels,
	       nY,
	       vry, vmw_s0, vmw_s2,
	       n, k) ;
	  }
	  else {
	    if( (((uint64_t)(pKernel + kernGroupOffset + k)) & 0x7) == 0
		&& (outChannelGroup & 0x1) == 0 )
	    {
	      k16<VEDNN_FILTER_LAYOUT_HWCN, true>(pIn, pKernel, pOut,
		 inChannel, inWidth, inHeight,
		 outChannel, outWidth, outHeight,
		 kernWidth, kernHeight,
		 outChannelGroup,
		 inGroupOffset, outGroupOffset,
		 kernGroupOffset, oPixels,
		 nY,
		 vry, vmw_s0, vmw_s2,
		 n, k) ;
	    }
	    else {
	      k16<VEDNN_FILTER_LAYOUT_HWCN, false>(pIn, pKernel, pOut,
		 inChannel, inWidth, inHeight,
		 outChannel, outWidth, outHeight,
		 kernWidth, kernHeight,
		 outChannelGroup,
		 inGroupOffset, outGroupOffset,
		 kernGroupOffset, oPixels,
		 nY,
		 vry, vmw_s0, vmw_s2,
		 n, k) ;
	    }
	  }
	} // outChannel
      } // group
    } // batch
  }

  return VEDNN_SUCCESS;
}

