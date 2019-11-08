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
    const int64_t inChannelGroup,
    const int64_t gOutChannelGroup,
    const int64_t inGroupOffset,
    const int64_t gOutGroupOffset,
    const int64_t gKernGroupOffset,
    const int64_t batch,
    const int64_t k,
    const int64_t nY,
    const __vm256 vm_s0,
    const __vm256 vm_s2
)
{
  const int64_t inWidthHalf   = inWidth >> 1 ;
  const int64_t gOutWidthHalf = gOutWidth >> 1 ;

  for (int64_t c=0; c<inChannelGroup; c++) {
    __vr vrsum_r0s0[NUMKERNEL] ;
    __vr vrsum_r0s1[NUMKERNEL] ;
    __vr vrsum_r0s2[NUMKERNEL] ;
    __vr vrsum_r1s0[NUMKERNEL] ;
    __vr vrsum_r1s1[NUMKERNEL] ;
    __vr vrsum_r1s2[NUMKERNEL] ;
    __vr vrsum_r2s0[NUMKERNEL] ;
    __vr vrsum_r2s1[NUMKERNEL] ;
    __vr vrsum_r2s2[NUMKERNEL] ;
#pragma clang loop unroll(full)
    for(int64_t kk=0; kk<NUMKERNEL; kk++) {
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

	const int64_t vl0 = inWidthHalf * (gOutHeight - y < nY ? gOutHeight - y : nY) ;
	const int64_t vl1 = gOutWidthHalf * (gOutHeight - y < nY ? gOutHeight - y : nY) ;

	const float *pInChannel = pIn + inGroupOffset + ( ((n * inChannel + c) * inHeight + y ) * inWidth ) ;

	const int64_t outIndex = gOutGroupOffset + ((n * gOutChannel + k) * gOutHeight + y) * gOutWidth ;

	__vr vrin_r0 = _vel_vld_vssl(8, pInChannel+0*inWidth, vl0) ;
	__vr vrin_r1 = _vel_vld_vssl(8, pInChannel+1*inWidth, vl0) ;
	__vr vrin_r2 = _vel_vld_vssl(8, pInChannel+2*inWidth, vl0) ;

	__vr vrgout[NUMKERNEL] ;
#pragma clang loop unroll(full)
	for(int64_t kk=0; kk<NUMKERNEL; kk++) {
	  vrgout[kk] = _vel_vld_vssl(8, pGOut+outIndex+kk*gOutHeight*gOutWidth, vl1) ;
	}

	__vr vrin_r0s0 = _vel_vcp_vvmvl(vrin_r0, vm_s0, _vel_vbrdl_vsl(0UL, vl0), vl0) ;
	__vr vrin_r0s2 = _vel_vcp_vvmvl(vrin_r0, vm_s2, _vel_vbrdl_vsl(0UL, vl0), vl0) ;
	__vr vrin_r1s0 = _vel_vcp_vvmvl(vrin_r1, vm_s0, _vel_vbrdl_vsl(0UL, vl0), vl0) ;
	__vr vrin_r1s2 = _vel_vcp_vvmvl(vrin_r1, vm_s2, _vel_vbrdl_vsl(0UL, vl0), vl0) ;
	__vr vrin_r2s0 = _vel_vcp_vvmvl(vrin_r2, vm_s0, _vel_vbrdl_vsl(0UL, vl0), vl0) ;
	__vr vrin_r2s2 = _vel_vcp_vvmvl(vrin_r2, vm_s2, _vel_vbrdl_vsl(0UL, vl0), vl0) ;

	__vr vrin_r0s1 = _vel_vshf_vvvsl(vrin_r0s0, vrin_r0s2, VE_VSHUFFLE_ZLYU, vl1) ;
	__vr vrin_r1s1 = _vel_vshf_vvvsl(vrin_r1s0, vrin_r1s2, VE_VSHUFFLE_ZLYU, vl1) ;
	__vr vrin_r2s1 = _vel_vshf_vvvsl(vrin_r2s0, vrin_r2s2, VE_VSHUFFLE_ZLYU, vl1) ;

#if 0
#pragma clang loop unroll(full)
	for(int64_t kk=0; kk<NUMKERNEL; kk++) {
	  vrsum_r0s0[kk] = _vel_pvfmad_vvvvvl(vrsum_r0s0[kk], vrin_r0s0, vrgout[kk], vrsum_r0s0[kk], vl1) ;
	}

#pragma clang loop unroll(full)
	for(int64_t kk=0; kk<NUMKERNEL; kk++) {
	  vrsum_r0s1[kk] = _vel_pvfmad_vvvvvl(vrsum_r0s1[kk], vrin_r0s1, vrgout[kk], vrsum_r0s1[kk], vl1) ;
	}

#pragma clang loop unroll(full)
	for(int64_t kk=0; kk<NUMKERNEL; kk++) {
	  vrsum_r0s2[kk] = _vel_pvfmad_vvvvvl(vrsum_r0s2[kk], vrin_r0s2, vrgout[kk], vrsum_r0s2[kk], vl1) ;
	}

#pragma clang loop unroll(full)
	for(int64_t kk=0; kk<NUMKERNEL; kk++) {
	  vrsum_r1s0[kk] = _vel_pvfmad_vvvvvl(vrsum_r1s0[kk], vrin_r1s0, vrgout[kk], vrsum_r1s0[kk], vl1) ;
	}

#pragma clang loop unroll(full)
	for(int64_t kk=0; kk<NUMKERNEL; kk++) {
	  vrsum_r1s1[kk] = _vel_pvfmad_vvvvvl(vrsum_r1s1[kk], vrin_r1s1, vrgout[kk], vrsum_r1s1[kk], vl1) ;
	}

#pragma clang loop unroll(full)
	for(int64_t kk=0; kk<NUMKERNEL; kk++) {
	  vrsum_r1s2[kk] = _vel_pvfmad_vvvvvl(vrsum_r1s2[kk], vrin_r1s2, vrgout[kk], vrsum_r1s2[kk], vl1) ;
	}

#pragma clang loop unroll(full)
	for(int64_t kk=0; kk<NUMKERNEL; kk++) {
	  vrsum_r2s0[kk] = _vel_pvfmad_vvvvvl(vrsum_r2s0[kk], vrin_r2s0, vrgout[kk], vrsum_r2s0[kk], vl1) ;
	}


#pragma clang loop unroll(full)
	for(int64_t kk=0; kk<NUMKERNEL; kk++) {
	  vrsum_r2s1[kk] = _vel_pvfmad_vvvvvl(vrsum_r2s1[kk], vrin_r2s1, vrgout[kk], vrsum_r2s1[kk], vl1) ;
	}

#pragma clang loop unroll(full)
	for(int64_t kk=0; kk<NUMKERNEL; kk++) {
	  vrsum_r2s2[kk] = _vel_pvfmad_vvvvvl(vrsum_r2s2[kk], vrin_r2s2, vrgout[kk], vrsum_r2s2[kk], vl1) ;
	}
#else
#pragma clang loop unroll(full)
	for(int64_t kk=0; kk<NUMKERNEL; kk++) {
	  vrsum_r0s0[kk] = _vel_pvfmad_vvvvvl(vrsum_r0s0[kk], vrin_r0s0, vrgout[kk], vrsum_r0s0[kk], vl1) ;
	  vrsum_r0s1[kk] = _vel_pvfmad_vvvvvl(vrsum_r0s1[kk], vrin_r0s1, vrgout[kk], vrsum_r0s1[kk], vl1) ;
	  vrsum_r0s2[kk] = _vel_pvfmad_vvvvvl(vrsum_r0s2[kk], vrin_r0s2, vrgout[kk], vrsum_r0s2[kk], vl1) ;
	  vrsum_r1s0[kk] = _vel_pvfmad_vvvvvl(vrsum_r1s0[kk], vrin_r1s0, vrgout[kk], vrsum_r1s0[kk], vl1) ;
	  vrsum_r1s1[kk] = _vel_pvfmad_vvvvvl(vrsum_r1s1[kk], vrin_r1s1, vrgout[kk], vrsum_r1s1[kk], vl1) ;
	  vrsum_r1s2[kk] = _vel_pvfmad_vvvvvl(vrsum_r1s2[kk], vrin_r1s2, vrgout[kk], vrsum_r1s2[kk], vl1) ;
	  vrsum_r2s0[kk] = _vel_pvfmad_vvvvvl(vrsum_r2s0[kk], vrin_r2s0, vrgout[kk], vrsum_r2s0[kk], vl1) ;
	  vrsum_r2s1[kk] = _vel_pvfmad_vvvvvl(vrsum_r2s1[kk], vrin_r2s1, vrgout[kk], vrsum_r2s1[kk], vl1) ;
	  vrsum_r2s2[kk] = _vel_pvfmad_vvvvvl(vrsum_r2s2[kk], vrin_r2s2, vrgout[kk], vrsum_r2s2[kk], vl1) ;
	}
#endif
      } // gOutPixels
    } // batch

#define FILTER_OFFSET(k,c,r,s) ( gKernGroupOffset + filter_index<FLAYOUT>(k,c,r,s, inChannelGroup, gOutChannelGroup, gKernHeight, gKernWidth) )

#pragma clang loop unroll(full)
    for(int64_t kk=0; kk<NUMKERNEL; kk++) {
      __vr vrsum00 = _vel_vfsums_vvl(_vel_vfadds_vvvl(vrsum_r0s0[kk], _vel_vsll_vvsl(vrsum_r0s0[kk],32,VLEN), VLEN), VLEN) ;
      _vel_vstu_vssl(vrsum00, 4, pGKernel+FILTER_OFFSET(k+kk, c, 0, 0), 1) ;
      __vr vrsum01 = _vel_vfsums_vvl(_vel_vfadds_vvvl(vrsum_r0s1[kk], _vel_vsll_vvsl(vrsum_r0s1[kk],32,VLEN), VLEN), VLEN) ;
      _vel_vstu_vssl(vrsum01, 4, pGKernel+FILTER_OFFSET(k+kk, c, 0, 1), 1) ;
      __vr vrsum02 = _vel_vfsums_vvl(_vel_vfadds_vvvl(vrsum_r0s2[kk], _vel_vsll_vvsl(vrsum_r0s2[kk],32,VLEN), VLEN), VLEN) ;
      _vel_vstu_vssl(vrsum02, 4, pGKernel+FILTER_OFFSET(k+kk, c, 0, 2), 1) ;
      __vr vrsum10 = _vel_vfsums_vvl(_vel_vfadds_vvvl(vrsum_r1s0[kk], _vel_vsll_vvsl(vrsum_r1s0[kk],32,VLEN), VLEN), VLEN) ;
      _vel_vstu_vssl(vrsum10, 4, pGKernel+FILTER_OFFSET(k+kk, c, 1, 0), 1) ;
      __vr vrsum11 = _vel_vfsums_vvl(_vel_vfadds_vvvl(vrsum_r1s1[kk], _vel_vsll_vvsl(vrsum_r1s1[kk],32,VLEN), VLEN), VLEN) ;
      _vel_vstu_vssl(vrsum11, 4, pGKernel+FILTER_OFFSET(k+kk, c, 1, 1), 1) ;
      __vr vrsum12 = _vel_vfsums_vvl(_vel_vfadds_vvvl(vrsum_r1s2[kk], _vel_vsll_vvsl(vrsum_r1s2[kk],32,VLEN), VLEN), VLEN) ;
      _vel_vstu_vssl(vrsum12, 4, pGKernel+FILTER_OFFSET(k+kk, c, 1, 2), 1) ;
      __vr vrsum20 = _vel_vfsums_vvl(_vel_vfadds_vvvl(vrsum_r2s0[kk], _vel_vsll_vvsl(vrsum_r2s0[kk],32,VLEN), VLEN), VLEN) ;
      _vel_vstu_vssl(vrsum20, 4, pGKernel+FILTER_OFFSET(k+kk, c, 2, 0), 1) ;
      __vr vrsum21 = _vel_vfsums_vvl(_vel_vfadds_vvvl(vrsum_r2s1[kk], _vel_vsll_vvsl(vrsum_r2s1[kk],32,VLEN), VLEN), VLEN) ;
      _vel_vstu_vssl(vrsum21, 4, pGKernel+FILTER_OFFSET(k+kk, c, 2, 1), 1) ;
      __vr vrsum22 = _vel_vfsums_vvl(_vel_vfadds_vvvl(vrsum_r2s2[kk], _vel_vsll_vvsl(vrsum_r2s2[kk],32,VLEN), VLEN), VLEN) ;
      _vel_vstu_vssl(vrsum22, 4, pGKernel+FILTER_OFFSET(k+kk, c, 2, 2), 1) ;
    }
#undef FILTER_OFFSET

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
    const int64_t strideWidth,		// 1
    const int64_t strideHeight,		// 1
    const int64_t padWidth,		// 0
    const int64_t padHeight,		// 0
    const int64_t dilationWidth,	// 1
    const int64_t dilationHeight	// 1
)
{

  const int64_t inWidthHalf   = inWidth >> 1 ;
  const int64_t gOutWidthHalf = gOutWidth >> 1 ;
  const int64_t nY = VLEN / inWidthHalf ;

  __vr vrseq = _vel_vseq_vl(VLEN) ;	// xy
  __vr vry_s0  = _vel_vdivsl_vvsl(vrseq, inWidthHalf, VLEN) ;
  __vr vrx_s0  = _vel_vsubsl_vvvl(vrseq, _vel_vmulul_vsvl(inWidthHalf,vry_s0, VLEN), VLEN) ;
  __vm256 vm_s0 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutWidthHalf, vrx_s0, VLEN), VLEN) ; // condition(x<gOutWidthHalf)

  __vr vrseq2  = _vel_vaddsl_vsvl(inWidthHalf-1, vrseq, VLEN) ;
  __vr vry_s2  = _vel_vdivsl_vvsl(vrseq2, inWidthHalf, VLEN) ;
  __vr vrx_s2  = _vel_vsubsl_vvvl(vrseq2, _vel_vmulul_vsvl(inWidthHalf,vry_s2, VLEN), VLEN) ;
  __vm256 vm_s2 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutWidthHalf, vrx_s2, VLEN), VLEN) ; // condition(x<gOutWidthHalf)

  for (int64_t g = beginGroup; g < nGroup; g++) {
    int64_t inGroupOffset    = g * inChannelGroup  * inHeight  * inWidth;
    int64_t gOutGroupOffset  = g * gOutChannelGroup * gOutHeight * gOutWidth;
    int64_t gKernGroupOffset = g * gOutChannelGroup * inChannelGroup * gKernHeight * gKernWidth;

    const int64_t remain = nOChannel % 5 ;

    int64_t k=0;
    switch(remain) {
    case 1:
      func<FLAYOUT, 1>(pIn, pGOut, pGKernel,
	 gOutChannel, gOutWidth, gOutHeight,
	 inChannel, inWidth, inHeight,
	 gKernWidth, gKernHeight,
	 inChannelGroup, gOutChannelGroup,
	 inGroupOffset, gOutGroupOffset, gKernGroupOffset,
	 batch, beginOChannel + k,
	 nY, vm_s0, vm_s2 ) ;
      k+=1 ;
      break ;
    case 2:
      func<FLAYOUT, 2>(pIn, pGOut, pGKernel,
	 gOutChannel, gOutWidth, gOutHeight,
	 inChannel, inWidth, inHeight,
	 gKernWidth, gKernHeight,
	 inChannelGroup, gOutChannelGroup,
	 inGroupOffset, gOutGroupOffset, gKernGroupOffset,
	 batch, beginOChannel + k,
	 nY, vm_s0, vm_s2 ) ;
      k+=2 ;
      break ;
    case 3:
      func<FLAYOUT, 3>(pIn, pGOut, pGKernel,
	 gOutChannel, gOutWidth, gOutHeight,
	 inChannel, inWidth, inHeight,
	 gKernWidth, gKernHeight,
	 inChannelGroup, gOutChannelGroup,
	 inGroupOffset, gOutGroupOffset, gKernGroupOffset,
	 batch, beginOChannel + k,
	 nY, vm_s0, vm_s2 ) ;
      k+=3 ;
      break ;
    case 4:
      func<FLAYOUT, 4>(pIn, pGOut, pGKernel,
	 gOutChannel, gOutWidth, gOutHeight,
	 inChannel, inWidth, inHeight,
	 gKernWidth, gKernHeight,
	 inChannelGroup, gOutChannelGroup,
	 inGroupOffset, gOutGroupOffset, gKernGroupOffset,
	 batch, beginOChannel + k,
	 nY, vm_s0, vm_s2 ) ;
      k+=4 ;
      break ;
    default :
      break ;
    }
    for (; k<nOChannel; ) {
      func<FLAYOUT, 5>(pIn, pGOut, pGKernel,
	 gOutChannel, gOutWidth, gOutHeight,
	 inChannel, inWidth, inHeight,
	 gKernWidth, gKernHeight,
	 inChannelGroup, gOutChannelGroup,
	 inGroupOffset, gOutGroupOffset, gKernGroupOffset,
	 batch, beginOChannel + k,
	 nY, vm_s0, vm_s2 ) ;
      k+=5 ;
    } // gOutChannel
  } // group
}


extern "C" vednnError_t
vednnConvolutionBackwardFilter_direct_dil1_str1_pad0_ker3_ow2X_iw2XU256_igoaligned (
    const vednnTensorParam_t * restrict 	pParamIn,
    const void * restrict 			pDataIn,
    const vednnTensorParam_t * restrict 	pParamGradOut,
    const void * restrict 			pDataGradOut,
    const vednnConvolutionParam_t * restrict 	pParamConv,
    const vednnFilterParam_t * restrict 	pParamGradKernel,
    void * restrict 				pDataGradKernel
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
