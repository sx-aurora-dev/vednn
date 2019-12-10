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
    const __vr vri,
    const __vr vrj
)
{
  const int64_t remain  = NUMKERNEL & 0x1 ;
  const int64_t nPacked = NUMKERNEL >> 1 ;

  __vr vrin_ptr_ost = _vel_vsfa_vvssl(_vel_vaddul_vvvl(vrj, _vel_vmulul_vsvl(inWidth,vri, nY*gOutWidth), nY*gOutWidth), 2, (uint64_t)(pIn + inGroupOffset), nY*gOutWidth) ;

  __vr vrw_s0 = _vel_vaddsl_vsvl(-1, vrj, nY*gOutWidth) ;
  __vr vrw_s3 = _vel_vaddsl_vsvl( 2, vrj, nY*gOutWidth) ;

  __vm256 vmw_s0 = _vel_vfmklge_mvl(vrw_s0, nY*gOutWidth) ;						// condition(0 <= w)
  __vm256 vmw_s3 = _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(inWidth,vrw_s3, nY*gOutWidth), nY*gOutWidth) ;	// condition(w < inWidth)

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

	  const int64_t vl = gOutWidth * (gOutHeight - y < nY ? gOutHeight - y : nY) ;
	  const int64_t gop = y * gOutWidth ;

	  __vr vrh = _vel_vaddsl_vsvl(r-1+y*2, vri, vl) ;

	  __vm256 vmh0 = _vel_vfmklge_mvl(vrh, vl) ;					// condition(0 <= h)
	  __vm256 vmh1 = _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(inHeight,vrh, vl), vl) ;	// condition(h < inHeight)
	  __vm256 vmh  = _vel_andm_mmm(vmh0, vmh1) ;

	  __vm256 vmhw_s0 = _vel_andm_mmm(vmh, vmw_s0) ;
	  __vm256 vmhw_s1 = vmh ;
	  __vm256 vmhw_s2 = vmh ;
	  __vm256 vmhw_s3 = _vel_andm_mmm(vmh, vmw_s3) ;

	  const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;

	  const int64_t outIndex = gOutGroupOffset + ((n * gOutChannel + k) * gOutHeight + y) * gOutWidth ;

	  __vr vrin_ptr_s0 = _vel_vaddsl_vsvl(4*(((n*inChannel+c)*inHeight+2*y+r-1)*inWidth-1), vrin_ptr_ost, vl) ;
	  __vr vrin_ptr_s1 = _vel_vaddsl_vsvl(4*(((n*inChannel+c)*inHeight+2*y+r-1)*inWidth  ), vrin_ptr_ost, vl) ;
	  __vr vrin_ptr_s2 = _vel_vaddsl_vsvl(4*(((n*inChannel+c)*inHeight+2*y+r-1)*inWidth+1), vrin_ptr_ost, vl) ;
	  __vr vrin_ptr_s3 = _vel_vaddsl_vsvl(4*(((n*inChannel+c)*inHeight+2*y+r-1)*inWidth+2), vrin_ptr_ost, vl) ;

	  __vr vrin_s0 = _vel_vgtu_vvssml(vrin_ptr_s0, 0, 0, vmhw_s0, vl) ;
	  __vr vrin_s1 = _vel_vgtu_vvssml(vrin_ptr_s1, 0, 0, vmhw_s1, vl) ;
	  __vr vrin_s2 = _vel_vgtu_vvssml(vrin_ptr_s2, 0, 0, vmhw_s2, vl) ;
	  __vr vrin_s3 = _vel_vgtu_vvssml(vrin_ptr_s3, 0, 0, vmhw_s3, vl) ;

	  __vr vrgout[NUMKERNEL] ;
#pragma clang loop unroll(full)
	  for(int64_t kk=0; kk<NUMKERNEL; kk++) {
	    vrgout[kk] = _vel_vldu_vssl(4, pGOut+outIndex+kk*gOutHeight*gOutWidth, vl) ;
	  }

	  __vr vrgoutp[NUMKERNEL]  ;
#pragma clang loop unroll(full)
	  for(int64_t kk=0; kk<nPacked; kk++) {
	    vrgoutp[kk] = _vel_vshf_vvvsl(vrgout[2*kk+remain], vrgout[2*kk+remain+1], VE_VSHUFFLE_YUZU, vl) ;
	  }

#define VFMAD(VRIN, VM, VRSUM, VRSUM0 )								\
	  {											\
	    VRIN = _vel_vmrg_vsvml(0.f, VRIN, VM, vl) ;						\
	    __vr vrinP = _vel_vshf_vvvsl(VRIN, VRIN, VE_VSHUFFLE_YUZU, vl) ;			\
	    if( remain ) {									\
	      VRSUM0  = _vel_vfmads_vvvvvl(VRSUM0, VRIN, vrgout[0], VRSUM0, vl) ;		\
	    }											\
	    _Pragma("clang loop unroll(full)")							\
	    for(int64_t kk=0; kk<nPacked; kk++) {						\
	      VRSUM[kk] = _vel_pvfmad_vvvvvl(VRSUM[kk], vrinP, vrgoutp[kk], VRSUM[kk], vl) ;	\
	    }											\
	  }

	  VFMAD(vrin_s0, vmhw_s0, vrsum_s0, vrsum0_s0)
	  VFMAD(vrin_s1, vmhw_s1, vrsum_s1, vrsum0_s1)
	  VFMAD(vrin_s2, vmhw_s2, vrsum_s2, vrsum0_s2)
	  VFMAD(vrin_s3, vmhw_s3, vrsum_s3, vrsum0_s3)

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
    const int64_t gKernWidth,		// 4
    const int64_t gKernHeight,		// 4
    const int64_t inChannelGroup,
    const int64_t gOutChannelGroup,
    const int64_t beginOChannel,
    const int64_t nOChannel,
    const int64_t beginGroup,
    const int64_t nGroup,
    const int64_t strideWidth,		// 2
    const int64_t strideHeight,		// 2
    const int64_t padWidth,		// 1
    const int64_t padHeight,		// 1
    const int64_t dilationWidth,	// 1
    const int64_t dilationHeight	// 1
)
{
  const int64_t nY = VLEN / gOutWidth ;

  __vr vrseq = _vel_vseq_vl(nY*gOutWidth) ;			// xy
  __vr vry  = _vel_vdivsl_vvsl(vrseq, gOutWidth, nY*gOutWidth) ;
  __vr vrx  = _vel_vsubsl_vvvl(vrseq, _vel_vmulul_vsvl(gOutWidth,vry, nY*gOutWidth), nY*gOutWidth) ;
  __vr vri  = _vel_vmulsl_vsvl(2, vry, nY*gOutWidth) ;
  __vr vrj  = _vel_vmulsl_vsvl(2, vrx, nY*gOutWidth) ;

  for (int64_t g = beginGroup; g < nGroup; g++) {
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
	 nY, vri, vrj ) ;
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
	 nY, vri, vrj ) ;
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
	 nY, vri, vrj ) ;
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
	 nY, vri, vrj ) ;
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
	 nY, vri, vrj ) ;
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
	 nY, vri, vrj ) ;
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
	 nY, vri, vrj ) ;
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
	 nY, vri, vrj ) ;
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
	 nY, vri, vrj ) ;
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
	 nY, vri, vrj ) ;
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
	 nY, vri, vrj ) ;
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
	 nY, vri, vrj ) ;
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
	 nY, vri, vrj ) ;
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
	 nY, vri, vrj ) ;
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
	 nY, vri, vrj ) ;
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
	 nY, vri, vrj ) ;
      k+=16 ;
    } // gOutChannel
  } // group
}


extern "C" vednnError_t
vednnConvolutionBackwardFilter_direct_dil1_str2_pad1_ker4_owU128(
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
