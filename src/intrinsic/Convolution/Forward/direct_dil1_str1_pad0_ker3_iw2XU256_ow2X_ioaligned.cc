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
  const int64_t inGroupOffset,
  const int64_t outGroupOffset,
  const int64_t biasGroupOffset,
  const int64_t kernGroupOffset,
  const int64_t nY,
  const int64_t inWidthHalf,
  const int64_t outWidthHalf,
  const __vm256 vm_s0,
  const __vm256 vm_s2,
  const int64_t n,
  const int64_t k
)
{

  int64_t outIndex = outGroupOffset + (n * outChannel + k  ) * outHeight*outWidth ;

  int64_t bias[NUMKERNEL] ;
#pragma clang loop unroll(full)
  for(int64_t kk=0; kk<NUMKERNEL; kk++) {
    bias[kk] = ADDBIAS ?  _vel_pack_f32a(pBias+biasGroupOffset+k+kk) : 0UL ;
  }

  for (int64_t y=0; y<outHeight; y+=nY) {
    const int64_t vl0 = inWidthHalf * (outHeight - y < nY ? outHeight - y : nY) ;
    const int64_t vl1 = outWidthHalf * (outHeight - y < nY ? outHeight - y : nY) ;
    const int64_t op = y * outWidth ;

    __vr vrsum[NUMKERNEL] ;
#pragma clang loop unroll(full)
    for(int64_t kk=0; kk<NUMKERNEL; kk++) {
      vrsum[kk] = _vel_vbrdl_vsl(bias[kk], vl1) ;
    }

    for (int64_t c = 0; c < inChannelGroup; c++) {

      const float *pInChannel = pIn + inGroupOffset + (((n * inChannel + c) * inHeight + y) * inWidth ) ;

      __vr vrin_r0 = _vel_vld_vssl(8, pInChannel+0*inWidth, vl0) ;
      __vr vrin_r1 = _vel_vld_vssl(8, pInChannel+1*inWidth, vl0) ;
      __vr vrin_r2 = _vel_vld_vssl(8, pInChannel+2*inWidth, vl0) ;

      __vr vrin_r0s0 = _vel_vcp_vvmvl(vrin_r0, vm_s0, _vel_vbrdl_vsl(0UL, vl0), vl0) ;
      __vr vrin_r0s2 = _vel_vcp_vvmvl(vrin_r0, vm_s2, _vel_vbrdl_vsl(0UL, vl0), vl0) ;
      __vr vrin_r1s0 = _vel_vcp_vvmvl(vrin_r1, vm_s0, _vel_vbrdl_vsl(0UL, vl0), vl0) ;
      __vr vrin_r1s2 = _vel_vcp_vvmvl(vrin_r1, vm_s2, _vel_vbrdl_vsl(0UL, vl0), vl0) ;
      __vr vrin_r2s0 = _vel_vcp_vvmvl(vrin_r2, vm_s0, _vel_vbrdl_vsl(0UL, vl0), vl0) ;
      __vr vrin_r2s2 = _vel_vcp_vvmvl(vrin_r2, vm_s2, _vel_vbrdl_vsl(0UL, vl0), vl0) ;

      __vr vrin_r0s1 = _vel_vshf_vvvsl(vrin_r0s0, vrin_r0s2,VE_VSHUFFLE_ZLYU, vl1) ;
      __vr vrin_r1s1 = _vel_vshf_vvvsl(vrin_r1s0, vrin_r1s2,VE_VSHUFFLE_ZLYU, vl1) ;
      __vr vrin_r2s1 = _vel_vshf_vvvsl(vrin_r2s0, vrin_r2s2,VE_VSHUFFLE_ZLYU, vl1) ;


#define FILTER_OFFSET(k,c,r,s) ( kernGroupOffset + filter_index<FLAYOUT>(k,c,r,s, inChannelGroup, outChannelGroup, kernHeight, kernWidth) )
#define VFADD(VRIN, R, S) 									\
      {												\
	_Pragma("clang loop unroll(full)")							\
	for(int64_t kk=0; kk<NUMKERNEL; kk++) {							\
	  const uint64_t kerValue = _vel_pack_f32a(pKernel+ FILTER_OFFSET(k+kk,c,R,S)) ;	\
	  vrsum[kk] = _vel_pvfmad_vvsvl(vrsum[kk], kerValue, VRIN, vl1) ;			\
	}											\
      }

      VFADD(vrin_r0s0, 0, 0) ;
      VFADD(vrin_r0s1, 0, 1) ;
      VFADD(vrin_r0s2, 0, 2) ;
      VFADD(vrin_r1s0, 1, 0) ;
      VFADD(vrin_r1s1, 1, 1) ;
      VFADD(vrin_r1s2, 1, 2) ;
      VFADD(vrin_r2s0, 2, 0) ;
      VFADD(vrin_r2s1, 2, 1) ;
      VFADD(vrin_r2s2, 2, 2) ;
#undef VFADD
#undef FILTER_OFFSET

    } // inChannel

#pragma clang loop unroll(full)
    for(int64_t kk=0; kk<NUMKERNEL; kk++) {
      _vel_vst_vssl(vrsum[kk], 8, pOut+outIndex+kk*outHeight*outWidth, vl1) ;
    }

    outIndex += 2*vl1 ;
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
  const int64_t inWidthHalf  = inWidth >> 1 ;
  const int64_t outWidthHalf = outWidth >> 1 ;
  const int64_t nY = VLEN / inWidthHalf ;

  __vr vrseq = _vel_vseq_vl(VLEN) ;
  __vm256 vm_s0, vm_s2 ;
  {
    __vr vry_s0  = _vel_vdivsl_vvsl(vrseq, inWidthHalf, VLEN) ;
    __vr vrx_s0  = _vel_vsubsl_vvvl(vrseq, _vel_vmulul_vsvl(inWidthHalf,vry_s0, VLEN), VLEN) ;
    vm_s0 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(outWidthHalf, vrx_s0, VLEN), VLEN) ; // condition(x<outWidthHalf)

    __vr vrseq2  = _vel_vaddsl_vsvl(inWidthHalf-1, vrseq, VLEN) ;
    __vr vry_s2  = _vel_vdivsl_vvsl(vrseq2, inWidthHalf, VLEN) ;
    __vr vrx_s2  = _vel_vsubsl_vvvl(vrseq2, _vel_vmulul_vsvl(inWidthHalf,vry_s2, VLEN), VLEN) ;
    vm_s2 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(outWidthHalf, vrx_s2, VLEN), VLEN) ; // condition(x<outWidthHalf)
  }

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
	     nY,
	     inWidthHalf, outWidthHalf,
	     vm_s0, vm_s2,
	     n, k );
	  k+=1 ;
	  break ;
	case 2 :
	  func<FLAYOUT, 2, ADDBIAS>(pIn, pKernel, pBias, pOut,
	     inChannel, inWidth, inHeight,
	     outChannel, outWidth, outHeight,
	     kernWidth, kernHeight,
	     inChannelGroup, outChannelGroup,
	     inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	     nY,
	     inWidthHalf, outWidthHalf,
	     vm_s0, vm_s2,
	     n, k );
	  k+=2 ;
	  break ;
	case 3 :
	  func<FLAYOUT, 3, ADDBIAS>(pIn, pKernel, pBias, pOut,
	     inChannel, inWidth, inHeight,
	     outChannel, outWidth, outHeight,
	     kernWidth, kernHeight,
	     inChannelGroup, outChannelGroup,
	     inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	     nY,
	     inWidthHalf, outWidthHalf,
	     vm_s0, vm_s2,
	     n, k );
	  k+=3 ;
	  break ;
	case 4 :
	  func<FLAYOUT, 4, ADDBIAS>(pIn, pKernel, pBias, pOut,
	     inChannel, inWidth, inHeight,
	     outChannel, outWidth, outHeight,
	     kernWidth, kernHeight,
	     inChannelGroup, outChannelGroup,
	     inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	     nY,
	     inWidthHalf, outWidthHalf,
	     vm_s0, vm_s2,
	     n, k );
	  k+=4 ;
	  break ;
	case 5 :
	  func<FLAYOUT, 5, ADDBIAS>(pIn, pKernel, pBias, pOut,
	     inChannel, inWidth, inHeight,
	     outChannel, outWidth, outHeight,
	     kernWidth, kernHeight,
	     inChannelGroup, outChannelGroup,
	     inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	     nY,
	     inWidthHalf, outWidthHalf,
	     vm_s0, vm_s2,
	     n, k );
	  k+=5 ;
	  break ;
	case 6 :
	  func<FLAYOUT, 6, ADDBIAS>(pIn, pKernel, pBias, pOut,
	     inChannel, inWidth, inHeight,
	     outChannel, outWidth, outHeight,
	     kernWidth, kernHeight,
	     inChannelGroup, outChannelGroup,
	     inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	     nY,
	     inWidthHalf, outWidthHalf,
	     vm_s0, vm_s2,
	     n, k );
	  k+=6 ;
	  break ;
	case 7 :
	  func<FLAYOUT, 7, ADDBIAS>(pIn, pKernel, pBias, pOut,
	     inChannel, inWidth, inHeight,
	     outChannel, outWidth, outHeight,
	     kernWidth, kernHeight,
	     inChannelGroup, outChannelGroup,
	     inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	     nY,
	     inWidthHalf, outWidthHalf,
	     vm_s0, vm_s2,
	     n, k );
	  k+=7 ;
	  break ;
	case 8 :
	  func<FLAYOUT, 8, ADDBIAS>(pIn, pKernel, pBias, pOut,
	     inChannel, inWidth, inHeight,
	     outChannel, outWidth, outHeight,
	     kernWidth, kernHeight,
	     inChannelGroup, outChannelGroup,
	     inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	     nY,
	     inWidthHalf, outWidthHalf,
	     vm_s0, vm_s2,
	     n, k );
	  k+=8 ;
	  break ;
	case 9 :
	  func<FLAYOUT, 9, ADDBIAS>(pIn, pKernel, pBias, pOut,
	     inChannel, inWidth, inHeight,
	     outChannel, outWidth, outHeight,
	     kernWidth, kernHeight,
	     inChannelGroup, outChannelGroup,
	     inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	     nY,
	     inWidthHalf, outWidthHalf,
	     vm_s0, vm_s2,
	     n, k );
	  k+=9 ;
	  break ;
	case 10 :
	  func<FLAYOUT, 10, ADDBIAS>(pIn, pKernel, pBias, pOut,
	     inChannel, inWidth, inHeight,
	     outChannel, outWidth, outHeight,
	     kernWidth, kernHeight,
	     inChannelGroup, outChannelGroup,
	     inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	     nY,
	     inWidthHalf, outWidthHalf,
	     vm_s0, vm_s2,
	     n, k );
	  k+=10 ;
	  break ;
	case 11 :
	  func<FLAYOUT, 11, ADDBIAS>(pIn, pKernel, pBias, pOut,
	     inChannel, inWidth, inHeight,
	     outChannel, outWidth, outHeight,
	     kernWidth, kernHeight,
	     inChannelGroup, outChannelGroup,
	     inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	     nY,
	     inWidthHalf, outWidthHalf,
	     vm_s0, vm_s2,
	     n, k );
	  k+=11 ;
	  break ;
	case 12 :
	  func<FLAYOUT, 12, ADDBIAS>(pIn, pKernel, pBias, pOut,
	     inChannel, inWidth, inHeight,
	     outChannel, outWidth, outHeight,
	     kernWidth, kernHeight,
	     inChannelGroup, outChannelGroup,
	     inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	     nY,
	     inWidthHalf, outWidthHalf,
	     vm_s0, vm_s2,
	     n, k );
	  k+=12 ;
	  break ;
	case 13 :
	  func<FLAYOUT, 13, ADDBIAS>(pIn, pKernel, pBias, pOut,
	     inChannel, inWidth, inHeight,
	     outChannel, outWidth, outHeight,
	     kernWidth, kernHeight,
	     inChannelGroup, outChannelGroup,
	     inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	     nY,
	     inWidthHalf, outWidthHalf,
	     vm_s0, vm_s2,
	     n, k );
	  k+=13 ;
	  break ;
	case 14 :
	  func<FLAYOUT, 14, ADDBIAS>(pIn, pKernel, pBias, pOut,
	     inChannel, inWidth, inHeight,
	     outChannel, outWidth, outHeight,
	     kernWidth, kernHeight,
	     inChannelGroup, outChannelGroup,
	     inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	     nY,
	     inWidthHalf, outWidthHalf,
	     vm_s0, vm_s2,
	     n, k );
	  k+=14 ;
	  break ;
	case 15 :
	  func<FLAYOUT, 15, ADDBIAS>(pIn, pKernel, pBias, pOut,
	     inChannel, inWidth, inHeight,
	     outChannel, outWidth, outHeight,
	     kernWidth, kernHeight,
	     inChannelGroup, outChannelGroup,
	     inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	     nY,
	     inWidthHalf, outWidthHalf,
	     vm_s0, vm_s2,
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
	     inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	     nY,
	     inWidthHalf, outWidthHalf,
	     vm_s0, vm_s2,
	     n, k );
	} // outChannel
    } // group
  } // batch
}

extern "C" vednnError_t
vednnConvolutionForward_direct_dil1_str1_pad0_ker3_iw2XU256_ow2X_ioaligned(
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
//  const int64_t strideWidth    = pParamConv->strideWidth;		// must be 1
//  const int64_t strideHeight   = pParamConv->strideHeight;		// must be 1
//  const int64_t padWidth       = pParamConv->padWidth;		// must be 0
//  const int64_t padHeight      = pParamConv->padHeight;		// must be 0
//  const int64_t dilationWidth  = pParamConv->dilationWidth;		// must be 1
//  const int64_t dilationHeight = pParamConv->dilationHeight;		// must be 1

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

  return VEDNN_SUCCESS;
}

