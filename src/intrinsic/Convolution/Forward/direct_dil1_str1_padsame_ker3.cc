#include <stdint.h>
#include <stdlib.h>

#include "vednn.h"

#include "velintrin.h"
#define VLEN	(256)

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

  const int64_t remain  = NUMKERNEL & 0x1 ;
  const int64_t nPacked = NUMKERNEL >> 1 ;

  const float   bias0  = ADDBIAS ?  pBias[biasGroupOffset+k+ 0] : 0.f ;
  int64_t bias[nPacked] ;
#pragma clang loop unroll(full)
  for(int64_t kk=0; kk<nPacked; kk++) bias[kk] = ADDBIAS ?  _vel_pack_f32p(pBias+biasGroupOffset+k+2*kk+remain, pBias+biasGroupOffset+k+2*kk+remain+1) : 0UL ;

  for (int64_t op = 0; op < outHeight * outWidth; op+=VLEN) {
    const int64_t vl = outHeight * outWidth - op < VLEN ? outHeight * outWidth - op : VLEN ;

    __vr vrsum0  = _vel_vbrds_vsl(bias0, vl) ;
    __vr vrsum[nPacked] ;
#pragma clang loop unroll(full)
    for(int64_t kk=0; kk<nPacked; kk++) vrsum[kk] = _vel_pvbrd_vsl(bias[kk], vl) ;

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
#define VFMAD(VRIN, R, S) 										\
      {													\
	__vr vrinP = _vel_vshf_vvvsl(VRIN, VRIN, VE_VSHUFFLE_YUZU, vl) ;				\
	if( remain ) {											\
	  const float    kerValue0  = pKernel[FILTER_OFFSET(k+ 0,c,R,S)] ;				\
	  vrsum0 = _vel_vfmads_vvsvl(vrsum0, kerValue0, VRIN, vl) ;					\
	}												\
	_Pragma("clang loop unroll(full)")								\
	for(int64_t kk=0; kk<nPacked; kk++) {								\
	  const uint64_t kerValue = _vel_pack_f32p(pKernel+FILTER_OFFSET(k+2*kk+remain,  c,R,S),	\
						   pKernel+FILTER_OFFSET(k+2*kk+remain+1,c,R,S)) ;	\
	  vrsum[kk] = _vel_pvfmad_vvsvl(vrsum[kk], kerValue, vrinP, vl) ;				\
        }												\
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
	  func<FLAYOUT, 1, ADDBIAS>(pIn, pKernel, pBias, pOut,
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
	  func<FLAYOUT, 2, ADDBIAS>(pIn, pKernel, pBias, pOut,
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
	  func<FLAYOUT, 3, ADDBIAS>(pIn, pKernel, pBias, pOut,
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
	  func<FLAYOUT, 4, ADDBIAS>(pIn, pKernel, pBias, pOut,
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
	  func<FLAYOUT, 5, ADDBIAS>(pIn, pKernel, pBias, pOut,
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
	  func<FLAYOUT, 6, ADDBIAS>(pIn, pKernel, pBias, pOut,
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
	  func<FLAYOUT, 7, ADDBIAS>(pIn, pKernel, pBias, pOut,
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
	  func<FLAYOUT, 8, ADDBIAS>(pIn, pKernel, pBias, pOut,
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
	  func<FLAYOUT, 9, ADDBIAS>(pIn, pKernel, pBias, pOut,
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
	  func<FLAYOUT, 10, ADDBIAS>(pIn, pKernel, pBias, pOut,
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
	  func<FLAYOUT, 11, ADDBIAS>(pIn, pKernel, pBias, pOut,
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
	  func<FLAYOUT, 12, ADDBIAS>(pIn, pKernel, pBias, pOut,
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
	  func<FLAYOUT, 13, ADDBIAS>(pIn, pKernel, pBias, pOut,
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
	  func<FLAYOUT, 14, ADDBIAS>(pIn, pKernel, pBias, pOut,
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
	  func<FLAYOUT, 15, ADDBIAS>(pIn, pKernel, pBias, pOut,
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
	  func<FLAYOUT, 16, ADDBIAS>(pIn, pKernel, pBias, pOut,
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

