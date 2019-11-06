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
    const __vm256 vmh_r0,
    const __vm256 vmh_r2,
    const __vm256 vmw_s0,
    const __vm256 vmw_s2
)
{
  const int64_t remain  = NUMKERNEL & 0x1 ;
  const int64_t nPacked = NUMKERNEL >> 1 ;

  for (int64_t c=0; c<inChannelGroup; c++) {

      __vr vrsum0_r0s0  = _vel_vbrds_vsl(0.f, VLEN) ;
      __vr vrsum0_r0s1  = _vel_vbrds_vsl(0.f, VLEN) ;
      __vr vrsum0_r0s2  = _vel_vbrds_vsl(0.f, VLEN) ;
      __vr vrsum0_r1s0  = _vel_vbrds_vsl(0.f, VLEN) ;
      __vr vrsum0_r1s1  = _vel_vbrds_vsl(0.f, VLEN) ;
      __vr vrsum0_r1s2  = _vel_vbrds_vsl(0.f, VLEN) ;
      __vr vrsum0_r2s0  = _vel_vbrds_vsl(0.f, VLEN) ;
      __vr vrsum0_r2s1  = _vel_vbrds_vsl(0.f, VLEN) ;
      __vr vrsum0_r2s2  = _vel_vbrds_vsl(0.f, VLEN) ;

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
	int64_t y = 0 ;
	{
	  const int64_t vl = gOutWidth * nY ;

	  const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;

	  const int64_t outIndex = gOutGroupOffset + ((n * gOutChannel + k) * gOutHeight + y) * gOutWidth ;

	  __vr vrin_r0s0 = _vel_vldu_vssl(4,&pInChannel[(y-1)*inWidth+0-1], vl) ;
	  __vr vrin_r0s1 = _vel_vldu_vssl(4,&pInChannel[(y-1)*inWidth+1-1], vl) ;
	  __vr vrin_r0s2 = _vel_vldu_vssl(4,&pInChannel[(y-1)*inWidth+2-1], vl) ;
	  __vr vrin_r1s0 = _vel_vldu_vssl(4,&pInChannel[(y  )*inWidth+0-1], vl) ;
	  __vr vrin_r1s1 = _vel_vldu_vssl(4,&pInChannel[(y  )*inWidth+1-1], vl) ;
	  __vr vrin_r1s2 = _vel_vldu_vssl(4,&pInChannel[(y  )*inWidth+2-1], vl) ;
	  __vr vrin_r2s0 = _vel_vldu_vssl(4,&pInChannel[(y+1)*inWidth+0-1], vl) ;
	  __vr vrin_r2s1 = _vel_vldu_vssl(4,&pInChannel[(y+1)*inWidth+1-1], vl) ;
	  __vr vrin_r2s2 = _vel_vldu_vssl(4,&pInChannel[(y+1)*inWidth+2-1], vl) ;

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

	  vrin_r0s0 = _vel_vmrg_vsvml(0.f, vrin_r0s0, vmh_r0, vl) ;
	  VFADD(vrin_r0s0, vrsum0_r0s0, vrsum_r0s0) ;
	  vrin_r0s1 = _vel_vmrg_vsvml(0.f, vrin_r0s1, vmh_r0, vl) ;
	  VFADD(vrin_r0s1, vrsum0_r0s1, vrsum_r0s1) ;
	  vrin_r0s2 = _vel_vmrg_vsvml(0.f, vrin_r0s2, vmh_r0, vl) ;
	  VFADD(vrin_r0s2, vrsum0_r0s2, vrsum_r0s2) ;

	  VFADD(vrin_r1s0, vrsum0_r1s0, vrsum_r1s0) ;
	  VFADD(vrin_r1s1, vrsum0_r1s1, vrsum_r1s1) ;
	  VFADD(vrin_r1s2, vrsum0_r1s2, vrsum_r1s2) ;

	  VFADD(vrin_r2s0, vrsum0_r2s0, vrsum_r2s0) ;
	  VFADD(vrin_r2s1, vrsum0_r2s1, vrsum_r2s1) ;
	  VFADD(vrin_r2s2, vrsum0_r2s2, vrsum_r2s2) ;

	} // gOutHeight
	for (y=nY; y+nY<gOutHeight; y+=nY) {
	  const int64_t vl = gOutWidth * nY ;

	  const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;

	  const int64_t outIndex = gOutGroupOffset + ((n * gOutChannel + k) * gOutHeight + y) * gOutWidth ;

	  __vr vrin_r0s0 = _vel_vldu_vssl(4,&pInChannel[(y-1)*inWidth+0-1], vl) ;
	  __vr vrin_r0s1 = _vel_vldu_vssl(4,&pInChannel[(y-1)*inWidth+1-1], vl) ;
	  __vr vrin_r0s2 = _vel_vldu_vssl(4,&pInChannel[(y-1)*inWidth+2-1], vl) ;
	  __vr vrin_r1s0 = _vel_vldu_vssl(4,&pInChannel[(y  )*inWidth+0-1], vl) ;
	  __vr vrin_r1s1 = _vel_vldu_vssl(4,&pInChannel[(y  )*inWidth+1-1], vl) ;
	  __vr vrin_r1s2 = _vel_vldu_vssl(4,&pInChannel[(y  )*inWidth+2-1], vl) ;
	  __vr vrin_r2s0 = _vel_vldu_vssl(4,&pInChannel[(y+1)*inWidth+0-1], vl) ;
	  __vr vrin_r2s1 = _vel_vldu_vssl(4,&pInChannel[(y+1)*inWidth+1-1], vl) ;
	  __vr vrin_r2s2 = _vel_vldu_vssl(4,&pInChannel[(y+1)*inWidth+2-1], vl) ;


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

	  VFADD(vrin_r0s0, vrsum0_r0s0, vrsum_r0s0) ;
	  VFADD(vrin_r0s1, vrsum0_r0s1, vrsum_r0s1) ;
	  VFADD(vrin_r0s2, vrsum0_r0s2, vrsum_r0s2) ;

	  VFADD(vrin_r1s0, vrsum0_r1s0, vrsum_r1s0) ;
	  VFADD(vrin_r1s1, vrsum0_r1s1, vrsum_r1s1) ;
	  VFADD(vrin_r1s2, vrsum0_r1s2, vrsum_r1s2) ;

	  VFADD(vrin_r2s0, vrsum0_r2s0, vrsum_r2s0) ;
	  VFADD(vrin_r2s1, vrsum0_r2s1, vrsum_r2s1) ;
	  VFADD(vrin_r2s2, vrsum0_r2s2, vrsum_r2s2) ;

	} // gOutHeight
	{
	  const int64_t vl = gOutWidth * (gOutHeight - y) ;

	  int64_t y = gOutHeight - 1 ;

	  const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;

	  const int64_t outIndex = gOutGroupOffset + ((n * gOutChannel + k) * gOutHeight + y) * gOutWidth ;

	  __vr vrin_r0s0 = _vel_vldu_vssl(4,&pInChannel[(y-1)*inWidth+0-1], vl) ;
	  __vr vrin_r0s1 = _vel_vldu_vssl(4,&pInChannel[(y-1)*inWidth+1-1], vl) ;
	  __vr vrin_r0s2 = _vel_vldu_vssl(4,&pInChannel[(y-1)*inWidth+2-1], vl) ;
	  __vr vrin_r1s0 = _vel_vldu_vssl(4,&pInChannel[(y  )*inWidth+0-1], vl) ;
	  __vr vrin_r1s1 = _vel_vldu_vssl(4,&pInChannel[(y  )*inWidth+1-1], vl) ;
	  __vr vrin_r1s2 = _vel_vldu_vssl(4,&pInChannel[(y  )*inWidth+2-1], vl) ;
	  __vr vrin_r2s0 = _vel_vldu_vssl(4,&pInChannel[(y+1)*inWidth+0-1], vl) ;
	  __vr vrin_r2s1 = _vel_vldu_vssl(4,&pInChannel[(y+1)*inWidth+1-1], vl) ;
	  __vr vrin_r2s2 = _vel_vldu_vssl(4,&pInChannel[(y+1)*inWidth+2-1], vl) ;


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

	  VFADD(vrin_r0s0, vrsum0_r0s0, vrsum_r0s0) ;
	  VFADD(vrin_r0s1, vrsum0_r0s1, vrsum_r0s1) ;
	  VFADD(vrin_r0s2, vrsum0_r0s2, vrsum_r0s2) ;

	  VFADD(vrin_r1s0, vrsum0_r1s0, vrsum_r1s0) ;
	  VFADD(vrin_r1s1, vrsum0_r1s1, vrsum_r1s1) ;
	  VFADD(vrin_r1s2, vrsum0_r1s2, vrsum_r1s2) ;

	  vrin_r2s0 = _vel_vmrg_vsvml(0.f, vrin_r2s0, vmh_r2, vl) ;
	  VFADD(vrin_r2s0, vrsum0_r2s0, vrsum_r2s0) ;
	  vrin_r2s1 = _vel_vmrg_vsvml(0.f, vrin_r2s1, vmh_r2, vl) ;
	  VFADD(vrin_r2s1, vrsum0_r2s1, vrsum_r2s1) ;
	  vrin_r2s2 = _vel_vmrg_vsvml(0.f, vrin_r2s2, vmh_r2, vl) ;
	  VFADD(vrin_r2s2, vrsum0_r2s2, vrsum_r2s2) ;
#undef VFADD
	} // gOutHeight
      } // batch

#define FILTER_OFFSET(k,c,r,s) ( gKernGroupOffset + filter_index<FLAYOUT>(k,c,r,s, inChannelGroup, gOutChannelGroup, gKernHeight, gKernWidth) )

      if( remain ) {
	vrsum0_r0s0 = _vel_vfsums_vvml(vrsum0_r0s0, vmw_s0, VLEN) ;
	_vel_vstu_vssl(vrsum0_r0s0, 4, pGKernel+FILTER_OFFSET(k+0,c,0,0), 1) ;
	vrsum0_r0s1 = _vel_vfsums_vvl(vrsum0_r0s1, VLEN) ;
	_vel_vstu_vssl(vrsum0_r0s1, 4, pGKernel+FILTER_OFFSET(k+0,c,0,1), 1) ;
	vrsum0_r0s2 = _vel_vfsums_vvml(vrsum0_r0s2, vmw_s2, VLEN) ;
	_vel_vstu_vssl(vrsum0_r0s2, 4, pGKernel+FILTER_OFFSET(k+0,c,0,2), 1) ;

	vrsum0_r1s0 = _vel_vfsums_vvml(vrsum0_r1s0, vmw_s0, VLEN) ;
	_vel_vstu_vssl(vrsum0_r1s0, 4, pGKernel+FILTER_OFFSET(k+0,c,1,0), 1) ;
	vrsum0_r1s1 = _vel_vfsums_vvl(vrsum0_r1s1, VLEN) ;
	_vel_vstu_vssl(vrsum0_r1s1, 4, pGKernel+FILTER_OFFSET(k+0,c,1,1), 1) ;
	vrsum0_r1s2 = _vel_vfsums_vvml(vrsum0_r1s2, vmw_s2, VLEN) ;
	_vel_vstu_vssl(vrsum0_r1s2, 4, pGKernel+FILTER_OFFSET(k+0,c,1,2), 1) ;

	vrsum0_r2s0 = _vel_vfsums_vvml(vrsum0_r2s0, vmw_s0, VLEN) ;
	_vel_vstu_vssl(vrsum0_r2s0, 4, pGKernel+FILTER_OFFSET(k+0,c,2,0), 1) ;
	vrsum0_r2s1 = _vel_vfsums_vvl(vrsum0_r2s1, VLEN) ;
	_vel_vstu_vssl(vrsum0_r2s1, 4, pGKernel+FILTER_OFFSET(k+0,c,2,1), 1) ;
	vrsum0_r2s2 = _vel_vfsums_vvml(vrsum0_r2s2, vmw_s2, VLEN) ;
	_vel_vstu_vssl(vrsum0_r2s2, 4, pGKernel+FILTER_OFFSET(k+0,c,2,2), 1) ;

      }

#pragma clang loop unroll(full)
      for(int64_t kk=0; kk<nPacked; kk++) {
	__vr vrsumU_r0s0 = _vel_vfsums_vvml(vrsum_r0s0[kk], vmw_s0, VLEN) ;
	_vel_vstu_vssl(vrsumU_r0s0, 4, pGKernel + FILTER_OFFSET(k+2*kk+remain,  c,0,0), 1) ;
	__vr vrsumL_r0s0 = _vel_vfsums_vvml(_vel_vsll_vvsl(vrsum_r0s0[kk],32, VLEN), vmw_s0, VLEN);
	_vel_vstu_vssl(vrsumL_r0s0, 4, pGKernel + FILTER_OFFSET(k+2*kk+remain+1,c,0,0), 1) ;
	__vr vrsumU_r0s1 = _vel_vfsums_vvl(vrsum_r0s1[kk], VLEN) ;
	_vel_vstu_vssl(vrsumU_r0s1, 4, pGKernel + FILTER_OFFSET(k+2*kk+remain,  c,0,1), 1) ;
	__vr vrsumL_r0s1 = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsum_r0s1[kk],32, VLEN), VLEN);
	_vel_vstu_vssl(vrsumL_r0s1, 4, pGKernel + FILTER_OFFSET(k+2*kk+remain+1,c,0,1), 1) ;
	__vr vrsumU_r0s2 = _vel_vfsums_vvml(vrsum_r0s2[kk], vmw_s2, VLEN) ;
	_vel_vstu_vssl(vrsumU_r0s2, 4, pGKernel + FILTER_OFFSET(k+2*kk+remain,  c,0,2), 1) ;
	__vr vrsumL_r0s2 = _vel_vfsums_vvml(_vel_vsll_vvsl(vrsum_r0s2[kk],32, VLEN), vmw_s2, VLEN);
	_vel_vstu_vssl(vrsumL_r0s2, 4, pGKernel + FILTER_OFFSET(k+2*kk+remain+1,c,0,2), 1) ;
	__vr vrsumU_r1s0 = _vel_vfsums_vvml(vrsum_r1s0[kk], vmw_s0, VLEN) ;
	_vel_vstu_vssl(vrsumU_r1s0, 4, pGKernel + FILTER_OFFSET(k+2*kk+remain,  c,1,0), 1) ;
	__vr vrsumL_r1s0 = _vel_vfsums_vvml(_vel_vsll_vvsl(vrsum_r1s0[kk],32, VLEN), vmw_s0, VLEN);
	_vel_vstu_vssl(vrsumL_r1s0, 4, pGKernel + FILTER_OFFSET(k+2*kk+remain+1,c,1,0), 1) ;
	__vr vrsumU_r1s1 = _vel_vfsums_vvl(vrsum_r1s1[kk], VLEN) ;
	_vel_vstu_vssl(vrsumU_r1s1, 4, pGKernel + FILTER_OFFSET(k+2*kk+remain,  c,1,1), 1) ;
	__vr vrsumL_r1s1 = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsum_r1s1[kk],32, VLEN), VLEN);
	_vel_vstu_vssl(vrsumL_r1s1, 4, pGKernel + FILTER_OFFSET(k+2*kk+remain+1,c,1,1), 1) ;
	__vr vrsumU_r1s2 = _vel_vfsums_vvml(vrsum_r1s2[kk], vmw_s2, VLEN) ;
	_vel_vstu_vssl(vrsumU_r1s2, 4, pGKernel + FILTER_OFFSET(k+2*kk+remain,  c,1,2), 1) ;
	__vr vrsumL_r1s2 = _vel_vfsums_vvml(_vel_vsll_vvsl(vrsum_r1s2[kk],32, VLEN), vmw_s2, VLEN);
	_vel_vstu_vssl(vrsumL_r1s2, 4, pGKernel + FILTER_OFFSET(k+2*kk+remain+1,c,1,2), 1) ;
	__vr vrsumU_r2s0 = _vel_vfsums_vvml(vrsum_r2s0[kk], vmw_s0, VLEN) ;
	_vel_vstu_vssl(vrsumU_r2s0, 4, pGKernel + FILTER_OFFSET(k+2*kk+remain,  c,2,0), 1) ;
	__vr vrsumL_r2s0 = _vel_vfsums_vvml(_vel_vsll_vvsl(vrsum_r2s0[kk],32, VLEN), vmw_s0, VLEN);
	_vel_vstu_vssl(vrsumL_r2s0, 4, pGKernel + FILTER_OFFSET(k+2*kk+remain+1,c,2,0), 1) ;
	__vr vrsumU_r2s1 = _vel_vfsums_vvl(vrsum_r2s1[kk], VLEN) ;
	_vel_vstu_vssl(vrsumU_r2s1, 4, pGKernel + FILTER_OFFSET(k+2*kk+remain,  c,2,1), 1) ;
	__vr vrsumL_r2s1 = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsum_r2s1[kk],32, VLEN), VLEN);
	_vel_vstu_vssl(vrsumL_r2s1, 4, pGKernel + FILTER_OFFSET(k+2*kk+remain+1,c,2,1), 1) ;
	__vr vrsumU_r2s2 = _vel_vfsums_vvml(vrsum_r2s2[kk], vmw_s2, VLEN) ;
	_vel_vstu_vssl(vrsumU_r2s2, 4, pGKernel + FILTER_OFFSET(k+2*kk+remain,  c,2,2), 1) ;
	__vr vrsumL_r2s2 = _vel_vfsums_vvml(_vel_vsll_vvsl(vrsum_r2s2[kk],32, VLEN), vmw_s2, VLEN);
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
    const int64_t strideWidth,		// 1
    const int64_t strideHeight,		// 1
    const int64_t padWidth,		// 1
    const int64_t padHeight,		// 1
    const int64_t dilationWidth,	// 1
    const int64_t dilationHeight	// 1
)
{
  const int64_t nY = VLEN / gOutWidth ;

  const int64_t y_remain = gOutHeight % nY ;
  const int64_t last_y = y_remain == 0 ? gOutHeight - nY : gOutHeight - y_remain ;

  __vr vrseq = _vel_vseq_vl(VLEN) ;			// xy

  __vr vry   = _vel_vdivsl_vvsl(vrseq, gOutWidth, VLEN) ;
  __vr vrx   = _vel_vsubsl_vvvl(vrseq, _vel_vmulul_vsvl(gOutWidth,vry, VLEN), VLEN) ;

  __vm256 vmw_s0 =  _vel_vfmklge_mvl(_vel_vaddsl_vsvl(-1, vrx, VLEN), VLEN) ;			// condition(  1<=x)
  __vm256 vmw_s2 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(inWidth-1,vrx, VLEN), VLEN) ;		// condition(x+1< inWidth)

  __vm256 vmh_r0 =  _vel_vfmklge_mvl(_vel_vaddsl_vsvl(-1, vry, VLEN) , VLEN) ;			// condition(  1<=y)
  __vm256 vmh_r2 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(inHeight-1-last_y,vry, VLEN), VLEN) ;	// condition(y+1< inHeight)

  for (int64_t g = 0; g < group; g++) {
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
	 nY, vmh_r0, vmh_r2, vmw_s0, vmw_s2) ;
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
	 nY, vmh_r0, vmh_r2, vmw_s0, vmw_s2) ;
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
	 nY, vmh_r0, vmh_r2, vmw_s0, vmw_s2) ;
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
	 nY, vmh_r0, vmh_r2, vmw_s0, vmw_s2) ;
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
	 nY, vmh_r0, vmh_r2, vmw_s0, vmw_s2) ;
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
	 nY, vmh_r0, vmh_r2, vmw_s0, vmw_s2) ;
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
	 nY, vmh_r0, vmh_r2, vmw_s0, vmw_s2) ;
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
	 nY, vmh_r0, vmh_r2, vmw_s0, vmw_s2) ;
      k+=8 ;
    } // gOutChannel
  } // group
}


extern "C" vednnError_t
vednnConvolutionBackwardFilter_direct_dil1_str1_padsame_ker3_owU128(
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

  if( filter_layout == VEDNN_FILTER_LAYOUT_NCHW) {
    convloop<VEDNN_FILTER_LAYOUT_NCHW>(pIn, pGOut, pGKernel,
	       batch, group,
	       inChannel, inWidth, inHeight,
	       gOutChannel, gOutWidth, gOutHeight,
	       gKernWidth, gKernHeight,
	       inChannelGroup, gOutChannelGroup,
	       beginOChannel, nOChannel,
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
	       strideWidth, strideHeight,
	       padWidth, padHeight,
	       dilationWidth, dilationHeight) ;
  }

  return VEDNN_SUCCESS;
}



