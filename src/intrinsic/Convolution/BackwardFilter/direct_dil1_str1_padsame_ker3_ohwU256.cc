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
    const __vm256 vmh_r0,
    const __vm256 vmh_r2,
    const __vm256 vmw_s0,
    const __vm256 vmw_s2
)
{

  const int64_t remain  = NUMKERNEL & 0x1 ;
  const int64_t nPacked = NUMKERNEL >> 1 ;

  __vm256 vmhw_r0s0 = _vel_andm_mmm(vmh_r0, vmw_s0) ;
  __vm256 vmhw_r0s1 = vmh_r0 ;
  __vm256 vmhw_r0s2 = _vel_andm_mmm(vmh_r0, vmw_s2) ;

  __vm256 vmhw_r1s0 = vmw_s0 ;
  __vm256 vmhw_r1s2 = vmw_s2 ;

  __vm256 vmhw_r2s0 = _vel_andm_mmm(vmh_r2, vmw_s0) ;
  __vm256 vmhw_r2s1 = vmh_r2 ;
  __vm256 vmhw_r2s2 = _vel_andm_mmm(vmh_r2, vmw_s2) ;

  const int64_t vl = gOutHeight * gOutWidth  ;

  for (int64_t c=0; c<inChannelGroup; c++) {

    __vr vrsum0_r0s0  = _vel_vbrds_vsl(0.f, vl) ;
    __vr vrsum0_r0s1  = _vel_vbrds_vsl(0.f, vl) ;
    __vr vrsum0_r0s2  = _vel_vbrds_vsl(0.f, vl) ;
    __vr vrsum0_r1s0  = _vel_vbrds_vsl(0.f, vl) ;
    __vr vrsum0_r1s1  = _vel_vbrds_vsl(0.f, vl) ;
    __vr vrsum0_r1s2  = _vel_vbrds_vsl(0.f, vl) ;
    __vr vrsum0_r2s0  = _vel_vbrds_vsl(0.f, vl) ;
    __vr vrsum0_r2s1  = _vel_vbrds_vsl(0.f, vl) ;
    __vr vrsum0_r2s2  = _vel_vbrds_vsl(0.f, vl) ;

    __vr vrsum_r0s0[nPacked] ;
    __vr vrsum_r0s1[nPacked] ;
    __vr vrsum_r0s2[nPacked] ;
    __vr vrsum_r1s0[nPacked] ;
    __vr vrsum_r1s1[nPacked] ;
    __vr vrsum_r1s2[nPacked] ;
    __vr vrsum_r2s0[nPacked] ;
    __vr vrsum_r2s1[nPacked] ;
    __vr vrsum_r2s2[nPacked] ;

#pragma clang loop unroll(full)
    for(int64_t kk=0; kk<nPacked; kk++) {
      vrsum_r0s0[kk] = _vel_pvbrd_vsl(0UL, vl) ;
      vrsum_r0s1[kk] = _vel_pvbrd_vsl(0UL, vl) ;
      vrsum_r0s2[kk] = _vel_pvbrd_vsl(0UL, vl) ;
      vrsum_r1s0[kk] = _vel_pvbrd_vsl(0UL, vl) ;
      vrsum_r1s1[kk] = _vel_pvbrd_vsl(0UL, vl) ;
      vrsum_r1s2[kk] = _vel_pvbrd_vsl(0UL, vl) ;
      vrsum_r2s0[kk] = _vel_pvbrd_vsl(0UL, vl) ;
      vrsum_r2s1[kk] = _vel_pvbrd_vsl(0UL, vl) ;
      vrsum_r2s2[kk] = _vel_pvbrd_vsl(0UL, vl) ;
    }

    for (int64_t n=0; n<batch; n++) {

      const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;

      const int64_t outIndex = gOutGroupOffset + ((n * gOutChannel + k) * gOutHeight ) * gOutWidth ;

      __vr vrin_r0s0 = _vel_vldu_vssl(4,&pInChannel[-inWidth+0-1], vl) ;
      __vr vrin_r0s1 = _vel_vldu_vssl(4,&pInChannel[-inWidth+1-1], vl) ;
      __vr vrin_r0s2 = _vel_vldu_vssl(4,&pInChannel[-inWidth+2-1], vl) ;
      __vr vrin_r1s0 = _vel_vldu_vssl(4,&pInChannel[        +0-1], vl) ;
      __vr vrin_r1s1 = _vel_vldu_vssl(4,&pInChannel[        +1-1], vl) ;
      __vr vrin_r1s2 = _vel_vldu_vssl(4,&pInChannel[        +2-1], vl) ;
      __vr vrin_r2s0 = _vel_vldu_vssl(4,&pInChannel[ inWidth+0-1], vl) ;
      __vr vrin_r2s1 = _vel_vldu_vssl(4,&pInChannel[ inWidth+1-1], vl) ;
      __vr vrin_r2s2 = _vel_vldu_vssl(4,&pInChannel[ inWidth+2-1], vl) ;


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

#define VFADD(VRIN,VRSUM0,VRSUM)								\
      {											\
	__vr vrinP = _vel_vshf_vvvsl(VRIN, VRIN, VE_VSHUFFLE_YUZU, vl) ;			\
	if( remain ) {									\
	  VRSUM0  = _vel_vfmads_vvvvvl(VRSUM0, VRIN, vrgout[0], VRSUM0, vl) ;		\
	}											\
	_Pragma("clang loop unroll(full)")						\
	for(int64_t kk=0; kk<nPacked; kk++) {						\
	  VRSUM[kk] = _vel_pvfmad_vvvvvl(VRSUM[kk], vrinP, vrgoutp[kk], VRSUM[kk], vl) ;	\
	}											\
      }

      VFADD(vrin_r0s0, vrsum0_r0s0, vrsum_r0s0) ;
      VFADD(vrin_r0s1, vrsum0_r0s1, vrsum_r0s1) ;
      VFADD(vrin_r0s2, vrsum0_r0s2, vrsum_r0s2) ;

      VFADD(vrin_r1s0, vrsum0_r1s0, vrsum_r1s0) ;
      VFADD(vrin_r1s1, vrsum0_r1s1, vrsum_r1s1) ;
      VFADD(vrin_r1s2, vrsum0_r1s2, vrsum_r1s2) ;

      VFADD(vrin_r2s0, vrsum0_r2s0, vrsum_r2s0) ;
      VFADD(vrin_r2s1, vrsum0_r2s1, vrsum_r2s1) ;
      VFADD(vrin_r2s2, vrsum0_r2s2, vrsum_r2s2) ;

#undef VFADD
    } // batch

#define FILTER_OFFSET(k,c,r,s) ( gKernGroupOffset + filter_index<FLAYOUT>(k,c,r,s, inChannelGroup, gOutChannelGroup, gKernHeight, gKernWidth) )

    if( remain ) {
      vrsum0_r0s0 = _vel_vfsums_vvml(vrsum0_r0s0, vmhw_r0s0, vl) ;
      _vel_vstu_vssl(vrsum0_r0s0, 4, pGKernel+FILTER_OFFSET(k+0,c,0,0), 1) ;
      vrsum0_r0s1 = _vel_vfsums_vvml(vrsum0_r0s1, vmhw_r0s1, vl) ;
      _vel_vstu_vssl(vrsum0_r0s1, 4, pGKernel+FILTER_OFFSET(k+0,c,0,1), 1) ;
      vrsum0_r0s2 = _vel_vfsums_vvml(vrsum0_r0s2, vmhw_r0s2, vl) ;
      _vel_vstu_vssl(vrsum0_r0s2, 4, pGKernel+FILTER_OFFSET(k+0,c,0,2), 1) ;

      vrsum0_r1s0 = _vel_vfsums_vvml(vrsum0_r1s0, vmw_s0, vl) ;
      _vel_vstu_vssl(vrsum0_r1s0, 4, pGKernel+FILTER_OFFSET(k+0,c,1,0), 1) ;
      vrsum0_r1s1 = _vel_vfsums_vvl(vrsum0_r1s1, vl) ;
      _vel_vstu_vssl(vrsum0_r1s1, 4, pGKernel+FILTER_OFFSET(k+0,c,1,1), 1) ;
      vrsum0_r1s2 = _vel_vfsums_vvml(vrsum0_r1s2, vmw_s2, vl) ;
      _vel_vstu_vssl(vrsum0_r1s2, 4, pGKernel+FILTER_OFFSET(k+0,c,1,2), 1) ;

      vrsum0_r2s0 = _vel_vfsums_vvml(vrsum0_r2s0, vmhw_r2s0, vl) ;
      _vel_vstu_vssl(vrsum0_r2s0, 4, pGKernel+FILTER_OFFSET(k+0,c,2,0), 1) ;
      vrsum0_r2s1 = _vel_vfsums_vvml(vrsum0_r2s1, vmhw_r2s1, vl) ;
      _vel_vstu_vssl(vrsum0_r2s1, 4, pGKernel+FILTER_OFFSET(k+0,c,2,1), 1) ;
      vrsum0_r2s2 = _vel_vfsums_vvml(vrsum0_r2s2, vmhw_r2s2, vl) ;
      _vel_vstu_vssl(vrsum0_r2s2, 4, pGKernel+FILTER_OFFSET(k+0,c,2,2), 1) ;

    }

#pragma clang loop unroll(full)
    for(int64_t kk=0; kk<nPacked; kk++) {
      __vr vrsumU_r0s0 = _vel_vfsums_vvml(vrsum_r0s0[kk], vmhw_r0s0, vl) ;
      _vel_vstu_vssl(vrsumU_r0s0, 4, pGKernel + FILTER_OFFSET(k+2*kk+remain,  c,0,0), 1) ;
      __vr vrsumL_r0s0 = _vel_vfsums_vvml(_vel_vsll_vvsl(vrsum_r0s0[kk],32, vl), vmhw_r0s0, vl);
      _vel_vstu_vssl(vrsumL_r0s0, 4, pGKernel + FILTER_OFFSET(k+2*kk+remain+1,c,0,0), 1) ;
      __vr vrsumU_r0s1 = _vel_vfsums_vvml(vrsum_r0s1[kk], vmhw_r0s1, vl) ;
      _vel_vstu_vssl(vrsumU_r0s1, 4, pGKernel + FILTER_OFFSET(k+2*kk+remain,  c,0,1), 1) ;
      __vr vrsumL_r0s1 = _vel_vfsums_vvml(_vel_vsll_vvsl(vrsum_r0s1[kk],32, vl), vmhw_r0s1, vl);
      _vel_vstu_vssl(vrsumL_r0s1, 4, pGKernel + FILTER_OFFSET(k+2*kk+remain+1,c,0,1), 1) ;
      __vr vrsumU_r0s2 = _vel_vfsums_vvml(vrsum_r0s2[kk], vmhw_r0s2, vl) ;
      _vel_vstu_vssl(vrsumU_r0s2, 4, pGKernel + FILTER_OFFSET(k+2*kk+remain,  c,0,2), 1) ;
      __vr vrsumL_r0s2 = _vel_vfsums_vvml(_vel_vsll_vvsl(vrsum_r0s2[kk],32, vl), vmhw_r0s2, vl);
      _vel_vstu_vssl(vrsumL_r0s2, 4, pGKernel + FILTER_OFFSET(k+2*kk+remain+1,c,0,2), 1) ;
      __vr vrsumU_r1s0 = _vel_vfsums_vvml(vrsum_r1s0[kk], vmhw_r1s0, vl) ;
      _vel_vstu_vssl(vrsumU_r1s0, 4, pGKernel + FILTER_OFFSET(k+2*kk+remain,  c,1,0), 1) ;
      __vr vrsumL_r1s0 = _vel_vfsums_vvml(_vel_vsll_vvsl(vrsum_r1s0[kk],32, vl), vmhw_r1s0, vl);
      _vel_vstu_vssl(vrsumL_r1s0, 4, pGKernel + FILTER_OFFSET(k+2*kk+remain+1,c,1,0), 1) ;
      __vr vrsumU_r1s1 = _vel_vfsums_vvl(vrsum_r1s1[kk], vl) ;
      _vel_vstu_vssl(vrsumU_r1s1, 4, pGKernel + FILTER_OFFSET(k+2*kk+remain,  c,1,1), 1) ;
      __vr vrsumL_r1s1 = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsum_r1s1[kk],32, vl), vl);
      _vel_vstu_vssl(vrsumL_r1s1, 4, pGKernel + FILTER_OFFSET(k+2*kk+remain+1,c,1,1), 1) ;
      __vr vrsumU_r1s2 = _vel_vfsums_vvml(vrsum_r1s2[kk], vmhw_r1s2, vl) ;
      _vel_vstu_vssl(vrsumU_r1s2, 4, pGKernel + FILTER_OFFSET(k+2*kk+remain,  c,1,2), 1) ;
      __vr vrsumL_r1s2 = _vel_vfsums_vvml(_vel_vsll_vvsl(vrsum_r1s2[kk],32, vl), vmhw_r1s2, vl);
      _vel_vstu_vssl(vrsumL_r1s2, 4, pGKernel + FILTER_OFFSET(k+2*kk+remain+1,c,1,2), 1) ;
      __vr vrsumU_r2s0 = _vel_vfsums_vvml(vrsum_r2s0[kk], vmhw_r2s0, vl) ;
      _vel_vstu_vssl(vrsumU_r2s0, 4, pGKernel + FILTER_OFFSET(k+2*kk+remain,  c,2,0), 1) ;
      __vr vrsumL_r2s0 = _vel_vfsums_vvml(_vel_vsll_vvsl(vrsum_r2s0[kk],32, vl), vmhw_r2s0, vl);
      _vel_vstu_vssl(vrsumL_r2s0, 4, pGKernel + FILTER_OFFSET(k+2*kk+remain+1,c,2,0), 1) ;
      __vr vrsumU_r2s1 = _vel_vfsums_vvml(vrsum_r2s1[kk], vmhw_r2s1, vl) ;
      _vel_vstu_vssl(vrsumU_r2s1, 4, pGKernel + FILTER_OFFSET(k+2*kk+remain,  c,2,1), 1) ;
      __vr vrsumL_r2s1 = _vel_vfsums_vvml(_vel_vsll_vvsl(vrsum_r2s1[kk],32, vl), vmhw_r2s1, vl);
      _vel_vstu_vssl(vrsumL_r2s1, 4, pGKernel + FILTER_OFFSET(k+2*kk+remain+1,c,2,1), 1) ;
      __vr vrsumU_r2s2 = _vel_vfsums_vvml(vrsum_r2s2[kk], vmhw_r2s2, vl) ;
      _vel_vstu_vssl(vrsumU_r2s2, 4, pGKernel + FILTER_OFFSET(k+2*kk+remain,  c,2,2), 1) ;
      __vr vrsumL_r2s2 = _vel_vfsums_vvml(_vel_vsll_vvsl(vrsum_r2s2[kk],32, vl), vmhw_r2s2, vl);
      _vel_vstu_vssl(vrsumL_r2s2, 4, pGKernel + FILTER_OFFSET(k+2*kk+remain+1,c,2,2), 1) ;

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
    const int64_t gKernWidth,		// 3
    const int64_t gKernHeight,		// 3
    const int64_t inChannelGroup,
    const int64_t gOutChannelGroup,
    const int64_t beginOChannel,
    const int64_t nOChannel,
    const int64_t beginGroup,
    const int64_t nGroup,
    const int64_t strideWidth,		// 1
    const int64_t strideHeight,		// 1
    const int64_t padWidth,		// 1
    const int64_t padHeight,		// 1
    const int64_t dilationWidth,	// 1
    const int64_t dilationHeight	// 1
)
{
  const int64_t vl = gOutWidth * gOutHeight ;

  __vr vrseq = _vel_vseq_vl(vl) ;			// xy

  __vr vry   = _vel_vdivsl_vvsl(vrseq, gOutWidth, vl) ;
  __vr vrx   = _vel_vsubsl_vvvl(vrseq, _vel_vmulul_vsvl(gOutWidth,vry, vl), vl) ;

  __vm256 vmw_s0 =  _vel_vfmklge_mvl(_vel_vaddsl_vsvl(-1, vrx, vl), vl) ;		// condition(  1<=x)
  __vm256 vmw_s2 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(inWidth-1,vrx, vl), vl) ;	// condition(x+1< inWidth)

  __vm256 vmh_r0 =  _vel_vfmklge_mvl(_vel_vaddsl_vsvl(-1, vry, vl) , vl) ;		// condition(  1<=y)
  __vm256 vmh_r2 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(inHeight-1,vry, vl), vl) ;	// condition(y+1< inHeight)


  for (int64_t g = beginGroup; g < beginGroup + nGroup; g++) {
    int64_t inGroupOffset    = g * inChannelGroup  * inHeight  * inWidth;
    int64_t gOutGroupOffset  = g * gOutChannelGroup * gOutHeight * gOutWidth;
    int64_t gKernGroupOffset = g * gOutChannelGroup * inChannelGroup * gKernHeight * gKernWidth;

    const int64_t remain = nOChannel & 0x7 ;

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
	 vmh_r0, vmh_r2, vmw_s0, vmw_s2) ;
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
	 vmh_r0, vmh_r2, vmw_s0, vmw_s2) ;
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
	 vmh_r0, vmh_r2, vmw_s0, vmw_s2) ;
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
	 vmh_r0, vmh_r2, vmw_s0, vmw_s2) ;
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
	 vmh_r0, vmh_r2, vmw_s0, vmw_s2) ;
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
	 vmh_r0, vmh_r2, vmw_s0, vmw_s2) ;
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
	 vmh_r0, vmh_r2, vmw_s0, vmw_s2) ;
      k+=7 ;
      break ;
    default :
      break ;
    }
    for (; k<nOChannel; ) {
      func<FLAYOUT, 8>(pIn, pGOut, pGKernel,
	 gOutChannel, gOutWidth, gOutHeight,
	 inChannel, inWidth, inHeight,
	 gKernWidth, gKernHeight,
	 inChannelGroup, gOutChannelGroup,
	 inGroupOffset, gOutGroupOffset, gKernGroupOffset,
	 batch, beginOChannel + k,
	 vmh_r0, vmh_r2, vmw_s0, vmw_s2) ;
      k+=8 ;
    } // gOutChannel
  } // group
}


extern "C" vednnError_t
vednnConvolutionBackwardFilter_direct_dil1_str1_padsame_ker3_ohwU256(
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
