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
    const __vm256 vm_s1,
    const __vm256 vm_s2,
    const __vm256 vm_s3
)
{
  const int64_t remain  = NUMKERNEL & 0x1 ;
  const int64_t nPacked = NUMKERNEL >> 1 ;

  for (int64_t c=0; c<inChannelGroup; c++) {
    for (int64_t r=0; r<gKernHeight; r++) {

      __vr vrsum0_s0  = _vel_vbrds_vsl(0.f, VLEN) ;
      __vr vrsum0_s1  = _vel_vbrds_vsl(0.f, VLEN) ;
      __vr vrsum0_s2  = _vel_vbrds_vsl(0.f, VLEN) ;
      __vr vrsum0_s3  = _vel_vbrds_vsl(0.f, VLEN) ;
      __vr vrsum_s0[nPacked] ;
      __vr vrsum_s1[nPacked] ;
      __vr vrsum_s2[nPacked] ;
      __vr vrsum_s3[nPacked] ;
#pragma clang loop unroll(full)
      for(int64_t kk=0; kk<nPacked; kk++) {
	vrsum_s0[kk] = _vel_pvbrd_vsl(0UL, VLEN) ;
	vrsum_s1[kk] = _vel_pvbrd_vsl(0UL, VLEN) ;
	vrsum_s2[kk] = _vel_pvbrd_vsl(0UL, VLEN) ;
	vrsum_s3[kk] = _vel_pvbrd_vsl(0UL, VLEN) ;
      }

      for (int64_t n=0; n<batch; n++) {
	for (int64_t y=0; y<gOutHeight; y+=nY) {

	  const int64_t vl0 = inWidth   * (gOutHeight - y < nY ? gOutHeight - y : nY) ;
	  const int64_t vl1 = gOutWidth * (gOutHeight - y < nY ? gOutHeight - y : nY) ;

	  const int64_t gop = y * gOutWidth ;

	  const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;

	  const int64_t outIndex = gOutGroupOffset + ((n * gOutChannel + k) * gOutHeight + y) * gOutWidth ;

	  __vr vrin = _vel_vldu_vssl(4, pInChannel+(y+r)*inWidth, vl0) ;

	  __vr vrgout[NUMKERNEL] ;
#pragma clang loop unroll(full)
	  for(int64_t kk=0; kk<NUMKERNEL; kk++) {
	    vrgout[kk] = _vel_vldu_vssl(4, pGOut+outIndex+kk*gOutHeight*gOutWidth, vl1) ;
	  }

	  __vr vrin_s0  = _vel_vcp_vvmvl(vrin, vm_s0, _vel_vbrdl_vsl(0UL, vl0), vl0) ;
	  __vr vrin_s1  = _vel_vcp_vvmvl(vrin, vm_s1, _vel_vbrdl_vsl(0UL, vl0), vl0) ;
	  __vr vrin_s2  = _vel_vcp_vvmvl(vrin, vm_s2, _vel_vbrdl_vsl(0UL, vl0), vl0) ;
	  __vr vrin_s3  = _vel_vcp_vvmvl(vrin, vm_s3, _vel_vbrdl_vsl(0UL, vl0), vl0) ;

	  __vr vrgoutp[NUMKERNEL]  ;
#pragma clang loop unroll(full)
	  for(int64_t kk=0; kk<nPacked; kk++) {
	    vrgoutp[kk] = _vel_vshf_vvvsl(vrgout[2*kk+remain], vrgout[2*kk+remain+1], VE_VSHUFFLE_YUZU, vl1) ;
	  }

#define VFMAD(VRIN, VRSUM, VRSUM0)								\
	  {											\
	    __vr vrinP = _vel_vshf_vvvsl(VRIN, VRIN, VE_VSHUFFLE_YUZU, vl1) ;			\
	    if( remain ) {									\
	      VRSUM0  = _vel_vfmads_vvvvvl(VRSUM0, VRIN, vrgout[0], VRSUM0, vl1) ;		\
	    }											\
	    _Pragma("clang loop unroll(full)")							\
	    for(int64_t kk=0; kk<nPacked; kk++) {						\
	      VRSUM[kk] = _vel_pvfmad_vvvvvl(VRSUM[kk], vrinP, vrgoutp[kk], VRSUM[kk], vl1) ;	\
	    }											\
	  }

	  VFMAD(vrin_s0, vrsum_s0, vrsum0_s0)
	  VFMAD(vrin_s1, vrsum_s1, vrsum0_s1)
	  VFMAD(vrin_s2, vrsum_s2, vrsum0_s2)
	  VFMAD(vrin_s3, vrsum_s3, vrsum0_s3)

#undef VFMAD
	} // gOutPixels
      } // batch

#define FILTER_OFFSET(k,c,r,s) ( gKernGroupOffset + filter_index<FLAYOUT>(k,c,r,s, inChannelGroup, gOutChannelGroup, 4, 4) )

      if( remain ) {
	vrsum0_s0 = _vel_vfsums_vvl(vrsum0_s0, VLEN) ;
	_vel_vstu_vssl(vrsum0_s0, 4, pGKernel+FILTER_OFFSET(k+0,c,r,0), 1) ;
	vrsum0_s1 = _vel_vfsums_vvl(vrsum0_s1, VLEN) ;
	_vel_vstu_vssl(vrsum0_s1, 4, pGKernel+FILTER_OFFSET(k+0,c,r,1), 1) ;
	vrsum0_s2 = _vel_vfsums_vvl(vrsum0_s2, VLEN) ;
	_vel_vstu_vssl(vrsum0_s2, 4, pGKernel+FILTER_OFFSET(k+0,c,r,2), 1) ;
	vrsum0_s3 = _vel_vfsums_vvl(vrsum0_s3, VLEN) ;
	_vel_vstu_vssl(vrsum0_s3, 4, pGKernel+FILTER_OFFSET(k+0,c,r,3), 1) ;
      }

#pragma clang loop unroll(full)
      for(int64_t kk=0; kk<nPacked; kk++) {
	__vr vrsumU_s0 = _vel_vfsums_vvl(vrsum_s0[kk], VLEN) ;
	_vel_vstu_vssl(vrsumU_s0, 4, pGKernel + FILTER_OFFSET(k+2*kk+remain,  c,r,0), 1) ;
	__vr vrsumL_s0 = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsum_s0[kk],32, VLEN), VLEN);
	_vel_vstu_vssl(vrsumL_s0, 4, pGKernel + FILTER_OFFSET(k+2*kk+remain+1,c,r,0), 1) ;
	__vr vrsumU_s1 = _vel_vfsums_vvl(vrsum_s1[kk], VLEN) ;
	_vel_vstu_vssl(vrsumU_s1, 4, pGKernel + FILTER_OFFSET(k+2*kk+remain,  c,r,1), 1) ;
	__vr vrsumL_s1 = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsum_s1[kk],32, VLEN), VLEN);
	_vel_vstu_vssl(vrsumL_s1, 4, pGKernel + FILTER_OFFSET(k+2*kk+remain+1,c,r,1), 1) ;
	__vr vrsumU_s2 = _vel_vfsums_vvl(vrsum_s2[kk], VLEN) ;
	_vel_vstu_vssl(vrsumU_s2, 4, pGKernel + FILTER_OFFSET(k+2*kk+remain,  c,r,2), 1) ;
	__vr vrsumL_s2 = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsum_s2[kk],32, VLEN), VLEN);
	_vel_vstu_vssl(vrsumL_s2, 4, pGKernel + FILTER_OFFSET(k+2*kk+remain+1,c,r,2), 1) ;
	__vr vrsumU_s3 = _vel_vfsums_vvl(vrsum_s3[kk], VLEN) ;
	_vel_vstu_vssl(vrsumU_s3, 4, pGKernel + FILTER_OFFSET(k+2*kk+remain,  c,r,3), 1) ;
	__vr vrsumL_s3 = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsum_s3[kk],32, VLEN), VLEN);
	_vel_vstu_vssl(vrsumL_s3, 4, pGKernel + FILTER_OFFSET(k+2*kk+remain+1,c,r,3), 1) ;
      }

#undef FILTER_OFFSET

    } // kernHeight
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
  const int64_t nY = VLEN / inWidth ;

  __vr vrseq = _vel_vseq_vl(VLEN) ;			// xy
  __vr vry_s0  = _vel_vdivsl_vvsl(vrseq, inWidth, VLEN) ;
  __vr vrx_s0  = _vel_vsubsl_vvvl(vrseq, _vel_vmulul_vsvl(inWidth,vry_s0, VLEN), VLEN) ;
  __vm256 vm_s0 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutWidth, vrx_s0, VLEN), VLEN) ; // condition(x<gOutWidth)

  __vr vrseq1  = _vel_vaddsl_vsvl(inWidth-1, vrseq, VLEN) ;
  __vr vry_s1  = _vel_vdivsl_vvsl(vrseq1, inWidth, VLEN) ;
  __vr vrx_s1  = _vel_vsubsl_vvvl(vrseq1, _vel_vmulul_vsvl(inWidth,vry_s1, VLEN), VLEN) ;
  __vm256 vm_s1 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutWidth, vrx_s1, VLEN), VLEN) ; // condition(x<gOutWidth)

  __vr vrseq2  = _vel_vaddsl_vsvl(inWidth-2, vrseq, VLEN) ;
  __vr vry_s2  = _vel_vdivsl_vvsl(vrseq2, inWidth, VLEN) ;
  __vr vrx_s2  = _vel_vsubsl_vvvl(vrseq2, _vel_vmulul_vsvl(inWidth,vry_s2, VLEN), VLEN) ;
  __vm256 vm_s2 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutWidth, vrx_s2, VLEN), VLEN) ; // condition(x<gOutWidth)

  __vr vrseq3  = _vel_vaddsl_vsvl(inWidth-3, vrseq, VLEN) ;
  __vr vry_s3  = _vel_vdivsl_vvsl(vrseq3, inWidth, VLEN) ;
  __vr vrx_s3  = _vel_vsubsl_vvvl(vrseq3, _vel_vmulul_vsvl(inWidth,vry_s3, VLEN), VLEN) ;
  __vm256 vm_s3 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutWidth, vrx_s3, VLEN), VLEN) ; // condition(x<gOutWidth)


  for (int64_t g = beginGroup; g < beginGroup + nGroup; g++) {
    int64_t inGroupOffset    = g * inChannelGroup  * inHeight  * inWidth;
    int64_t gOutGroupOffset  = g * gOutChannelGroup * gOutHeight * gOutWidth;
    int64_t gKernGroupOffset = g * gOutChannelGroup * inChannelGroup * gKernHeight * gKernWidth;

    const int64_t remain = nOChannel & 0xf ;

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
	 nY, vm_s0, vm_s1, vm_s2, vm_s3 ) ;
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
	 nY, vm_s0, vm_s1, vm_s2, vm_s3 ) ;
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
	 nY, vm_s0, vm_s1, vm_s2, vm_s3 ) ;
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
	 nY, vm_s0, vm_s1, vm_s2, vm_s3 ) ;
      k+=4 ;
      break ;
    case 5:
      func<FLAYOUT, 5>(pIn, pGOut, pGKernel,
	 gOutChannel, gOutWidth, gOutHeight,
	 inChannel, inWidth, inHeight,
	 gKernWidth, gKernHeight,
	 inChannelGroup, gOutChannelGroup,
	 inGroupOffset, gOutGroupOffset, gKernGroupOffset,
	 batch, beginOChannel + k,
	 nY, vm_s0, vm_s1, vm_s2, vm_s3 ) ;
      k+=5 ;
      break ;
    case 6:
      func<FLAYOUT, 6>(pIn, pGOut, pGKernel,
	 gOutChannel, gOutWidth, gOutHeight,
	 inChannel, inWidth, inHeight,
	 gKernWidth, gKernHeight,
	 inChannelGroup, gOutChannelGroup,
	 inGroupOffset, gOutGroupOffset, gKernGroupOffset,
	 batch, beginOChannel + k,
	 nY, vm_s0, vm_s1, vm_s2, vm_s3 ) ;
      k+=6 ;
      break ;
    case 7:
      func<FLAYOUT, 7>(pIn, pGOut, pGKernel,
	 gOutChannel, gOutWidth, gOutHeight,
	 inChannel, inWidth, inHeight,
	 gKernWidth, gKernHeight,
	 inChannelGroup, gOutChannelGroup,
	 inGroupOffset, gOutGroupOffset, gKernGroupOffset,
	 batch, beginOChannel + k,
	 nY, vm_s0, vm_s1, vm_s2, vm_s3 ) ;
      k+=7 ;
      break ;
    case 8:
      func<FLAYOUT, 8>(pIn, pGOut, pGKernel,
	 gOutChannel, gOutWidth, gOutHeight,
	 inChannel, inWidth, inHeight,
	 gKernWidth, gKernHeight,
	 inChannelGroup, gOutChannelGroup,
	 inGroupOffset, gOutGroupOffset, gKernGroupOffset,
	 batch, beginOChannel + k,
	 nY, vm_s0, vm_s1, vm_s2, vm_s3 ) ;
      k+=8 ;
      break ;
    case 9:
      func<FLAYOUT, 9>(pIn, pGOut, pGKernel,
	 gOutChannel, gOutWidth, gOutHeight,
	 inChannel, inWidth, inHeight,
	 gKernWidth, gKernHeight,
	 inChannelGroup, gOutChannelGroup,
	 inGroupOffset, gOutGroupOffset, gKernGroupOffset,
	 batch, beginOChannel + k,
	 nY, vm_s0, vm_s1, vm_s2, vm_s3 ) ;
      k+=9 ;
      break ;
    case 10:
      func<FLAYOUT, 10>(pIn, pGOut, pGKernel,
	 gOutChannel, gOutWidth, gOutHeight,
	 inChannel, inWidth, inHeight,
	 gKernWidth, gKernHeight,
	 inChannelGroup, gOutChannelGroup,
	 inGroupOffset, gOutGroupOffset, gKernGroupOffset,
	 batch, beginOChannel + k,
	 nY, vm_s0, vm_s1, vm_s2, vm_s3 ) ;
      k+=10 ;
      break ;
    case 11:
      func<FLAYOUT, 11>(pIn, pGOut, pGKernel,
	 gOutChannel, gOutWidth, gOutHeight,
	 inChannel, inWidth, inHeight,
	 gKernWidth, gKernHeight,
	 inChannelGroup, gOutChannelGroup,
	 inGroupOffset, gOutGroupOffset, gKernGroupOffset,
	 batch, beginOChannel + k,
	 nY, vm_s0, vm_s1, vm_s2, vm_s3 ) ;
      k+=11 ;
      break ;
    case 12:
      func<FLAYOUT, 12>(pIn, pGOut, pGKernel,
	 gOutChannel, gOutWidth, gOutHeight,
	 inChannel, inWidth, inHeight,
	 gKernWidth, gKernHeight,
	 inChannelGroup, gOutChannelGroup,
	 inGroupOffset, gOutGroupOffset, gKernGroupOffset,
	 batch, beginOChannel + k,
	 nY, vm_s0, vm_s1, vm_s2, vm_s3 ) ;
      k+=12 ;
      break ;
    case 13:
      func<FLAYOUT, 13>(pIn, pGOut, pGKernel,
	 gOutChannel, gOutWidth, gOutHeight,
	 inChannel, inWidth, inHeight,
	 gKernWidth, gKernHeight,
	 inChannelGroup, gOutChannelGroup,
	 inGroupOffset, gOutGroupOffset, gKernGroupOffset,
	 batch, beginOChannel + k,
	 nY, vm_s0, vm_s1, vm_s2, vm_s3 ) ;
      k+=13 ;
      break ;
    case 14:
      func<FLAYOUT, 14>(pIn, pGOut, pGKernel,
	 gOutChannel, gOutWidth, gOutHeight,
	 inChannel, inWidth, inHeight,
	 gKernWidth, gKernHeight,
	 inChannelGroup, gOutChannelGroup,
	 inGroupOffset, gOutGroupOffset, gKernGroupOffset,
	 batch, beginOChannel + k,
	 nY, vm_s0, vm_s1, vm_s2, vm_s3 ) ;
      k+=14 ;
      break ;
    case 15:
      func<FLAYOUT, 15>(pIn, pGOut, pGKernel,
	 gOutChannel, gOutWidth, gOutHeight,
	 inChannel, inWidth, inHeight,
	 gKernWidth, gKernHeight,
	 inChannelGroup, gOutChannelGroup,
	 inGroupOffset, gOutGroupOffset, gKernGroupOffset,
	 batch, beginOChannel + k,
	 nY, vm_s0, vm_s1, vm_s2, vm_s3 ) ;;
      k+=15 ;
      break ;
    default :
      break ;
    }
    for (; k<nOChannel; ) {
      func<FLAYOUT, 16>(pIn, pGOut, pGKernel,
	 gOutChannel, gOutWidth, gOutHeight,
	 inChannel, inWidth, inHeight,
	 gKernWidth, gKernHeight,
	 inChannelGroup, gOutChannelGroup,
	 inGroupOffset, gOutGroupOffset, gKernGroupOffset,
	 batch, beginOChannel + k,
	 nY, vm_s0, vm_s1, vm_s2, vm_s3 ) ;
      k+=16 ;
    } // gOutChannel
  } // group
}


extern "C" vednnError_t
vednnConvolutionBackwardFilter_direct_dil1_str1_pad0_ker4_owU128(
    const vednnTensorParam_t *  	pParamIn,
    const void *  			pDataIn,
    const vednnTensorParam_t *  	pParamGradOut,
    const void *  			pDataGradOut,
    const vednnConvolutionParam_t *  	pParamConv,
    const vednnFilterParam_t *  	pParamGradKernel,
    void *  				pDataGradKernel
#ifdef VEDNN_USE_OPENMP
    ,
    const int64_t			beginOChannel,
    const int64_t			nOChannel
#ifdef VEDNN_OMP_GROUP_PARALLEL
    ,
    const int64_t			beginGroup,
    const int64_t			nGroup
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