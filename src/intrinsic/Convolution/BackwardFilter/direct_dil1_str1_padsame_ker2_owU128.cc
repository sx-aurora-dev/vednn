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
    const int64_t padWidth,
    const int64_t padHeight,
    const int64_t inChannelGroup,
    const int64_t gOutChannelGroup,
    const int64_t inGroupOffset,
    const int64_t gOutGroupOffset,
    const int64_t gKernGroupOffset,
    const int64_t batch,
    const int64_t k,
    const int64_t nY,
    const __vm256 vm_r0,
    const __vm256 vm_s0
)
{
  const int64_t remain  = NUMKERNEL & 0x1 ;
  const int64_t nPacked = NUMKERNEL >> 1 ;

  for (int64_t c=0; c<inChannelGroup; c++) {

    __vr vrsum0_r0s0 = _vel_vbrds_vsl(0.f, VLEN) ;
    __vr vrsum0_r0s1 = _vel_vbrds_vsl(0.f, VLEN) ;
    __vr vrsum0_r1s0 = _vel_vbrds_vsl(0.f, VLEN) ;
    __vr vrsum0_r1s1 = _vel_vbrds_vsl(0.f, VLEN) ;

    __vr vrsum_r0s0[nPacked] ;
    __vr vrsum_r0s1[nPacked] ;
    __vr vrsum_r1s0[nPacked] ;
    __vr vrsum_r1s1[nPacked] ;
#pragma clang loop unroll(full)
    for(int64_t kk=0; kk<nPacked; kk++) {
      vrsum_r0s0[kk] = _vel_pvbrd_vsl(0UL, VLEN) ;
      vrsum_r0s1[kk] = _vel_pvbrd_vsl(0UL, VLEN) ;
      vrsum_r1s0[kk] = _vel_pvbrd_vsl(0UL, VLEN) ;
      vrsum_r1s1[kk] = _vel_pvbrd_vsl(0UL, VLEN) ;
    }


    for (int64_t n=0; n<batch; n++) {
      {
	int64_t y = 0 ;

	const int64_t vl = gOutWidth * (gOutHeight - y < nY ? gOutHeight - y : nY ) ;

	__vm256 vm_r0s0 = _vel_andm_mmm(vm_r0,vm_s0) ;
	__vm256 vm_r0s1 = vm_r0 ;
	__vm256 vm_r1s0 = vm_s0 ;

	const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;

	const int64_t outIndex = gOutGroupOffset + ((n * gOutChannel + k) * gOutHeight + y) * gOutWidth ;

      	/* memory access errors mihgt be caused (vrin) */
      	__vr vrin_r0s0    = _vel_vldu_vssl(4,&pInChannel[(y-1)*inWidth-1], vl) ;
      	__vr vrin_r0s1    = _vel_vldu_vssl(4,&pInChannel[(y-1)*inWidth  ], vl) ;
      	__vr vrin_r1s0    = _vel_vldu_vssl(4,&pInChannel[(y  )*inWidth-1], vl) ;
      	__vr vrin_r1s1    = _vel_vldu_vssl(4,&pInChannel[(y  )*inWidth  ], vl) ;

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

      	vrin_r0s0 = _vel_vmrg_vsvml(0.f, vrin_r0s0, vm_r0s0, vl) ;
      	__vr vrinP_r0s0    = _vel_vshf_vvvsl(vrin_r0s0, vrin_r0s0, VE_VSHUFFLE_YUZU, vl) ;
	if( remain ) {
	  vrsum0_r0s0 = _vel_vfmads_vvvvvl(vrsum0_r0s0, vrin_r0s0, vrgout[0], vrsum0_r0s0, vl) ;
	}
#pragma clang loop unroll(full)
	for(int64_t kk=0; kk<nPacked; kk++) {
	  vrsum_r0s0[kk] = _vel_pvfmad_vvvvvl(vrsum_r0s0[kk], vrinP_r0s0, vrgoutp[kk], vrsum_r0s0[kk], vl) ;
	}

      	vrin_r0s1 = _vel_vmrg_vsvml(0.f, vrin_r0s1, vm_r0s1, vl) ;
      	__vr vrinP_r0s1    = _vel_vshf_vvvsl(vrin_r0s1, vrin_r0s1, VE_VSHUFFLE_YUZU, vl) ;
	if( remain ) {
	  vrsum0_r0s1 = _vel_vfmads_vvvvvl(vrsum0_r0s1, vrin_r0s1, vrgout[0], vrsum0_r0s1, vl) ;
	}
#pragma clang loop unroll(full)
	for(int64_t kk=0; kk<nPacked; kk++) {
	  vrsum_r0s1[kk] = _vel_pvfmad_vvvvvl(vrsum_r0s1[kk], vrinP_r0s1, vrgoutp[kk], vrsum_r0s1[kk], vl) ;
	}

	vrin_r1s0 = _vel_vmrg_vsvml(0.f, vrin_r1s0, vm_r1s0, vl) ;
	__vr vrinP_r1s0    = _vel_vshf_vvvsl(vrin_r1s0, vrin_r1s0, VE_VSHUFFLE_YUZU, vl) ;
	if( remain ) {
	  vrsum0_r1s0 = _vel_vfmads_vvvvvl(vrsum0_r1s0, vrin_r1s0, vrgout[0], vrsum0_r1s0, vl) ;
	}
#pragma clang loop unroll(full)
	for(int64_t kk=0; kk<nPacked; kk++) {
	  vrsum_r1s0[kk] = _vel_pvfmad_vvvvvl(vrsum_r1s0[kk], vrinP_r1s0, vrgoutp[kk], vrsum_r1s0[kk], vl) ;
	}

	// vrin_r1s1 : no need to use mask
	__vr vrinP_r1s1    = _vel_vshf_vvvsl(vrin_r1s1, vrin_r1s1, VE_VSHUFFLE_YUZU, vl) ;
	if( remain ) {
	  vrsum0_r1s1 = _vel_vfmads_vvvvvl(vrsum0_r1s1, vrin_r1s1, vrgout[0], vrsum0_r1s1, vl) ;
	}
#pragma clang loop unroll(full)
	for(int64_t kk=0; kk<nPacked; kk++) {
	  vrsum_r1s1[kk] = _vel_pvfmad_vvvvvl(vrsum_r1s1[kk], vrinP_r1s1, vrgoutp[kk], vrsum_r1s1[kk], vl) ;
	}
      }
      for (int64_t y = nY; y < gOutHeight ; y+=nY ) {
	const int64_t vl = gOutWidth * (gOutHeight - y < nY ? gOutHeight - y : nY ) ;

	__vm256 vm_r0s0 = vm_s0 ;
	__vm256 vm_r1s0 = vm_s0 ;

	const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;

	const int64_t outIndex = gOutGroupOffset + ((n * gOutChannel + k) * gOutHeight + y) * gOutWidth ;

	__vr vrin_r0s0    = _vel_vldu_vssl(4,&pInChannel[(y-1)*inWidth-1], vl) ;
	__vr vrin_r0s1    = _vel_vldu_vssl(4,&pInChannel[(y-1)*inWidth  ], vl) ;
	__vr vrin_r1s0    = _vel_vldu_vssl(4,&pInChannel[(y  )*inWidth-1], vl) ;
	__vr vrin_r1s1    = _vel_vldu_vssl(4,&pInChannel[(y  )*inWidth  ], vl) ;

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

	vrin_r0s0 = _vel_vmrg_vsvml(0.f, vrin_r0s0, vm_r0s0, vl) ;
	__vr vrinP_r0s0    = _vel_vshf_vvvsl(vrin_r0s0, vrin_r0s0, VE_VSHUFFLE_YUZU, vl) ;
	if( remain ) {
	  vrsum0_r0s0 = _vel_vfmads_vvvvvl(vrsum0_r0s0, vrin_r0s0, vrgout[0], vrsum0_r0s0, vl) ;
	}
#pragma clang loop unroll(full)
	for(int64_t kk=0; kk<nPacked; kk++) {
	  vrsum_r0s0[kk] = _vel_pvfmad_vvvvvl(vrsum_r0s0[kk], vrinP_r0s0, vrgoutp[kk], vrsum_r0s0[kk], vl) ;
	}

	// vrin_r0s1 : no need to use mask
	__vr vrinP_r0s1    = _vel_vshf_vvvsl(vrin_r0s1, vrin_r0s1, VE_VSHUFFLE_YUZU, vl) ;
	if( remain ) {
	  vrsum0_r0s1 = _vel_vfmads_vvvvvl(vrsum0_r0s1, vrin_r0s1, vrgout[0], vrsum0_r0s1, vl) ;
	}
#pragma clang loop unroll(full)
	for(int64_t kk=0; kk<nPacked; kk++) {
	  vrsum_r0s1[kk] = _vel_pvfmad_vvvvvl(vrsum_r0s1[kk], vrinP_r0s1, vrgoutp[kk], vrsum_r0s1[kk], vl) ;
	}

	vrin_r1s0 = _vel_vmrg_vsvml(0.f, vrin_r1s0, vm_r1s0, vl) ;
	__vr vrinP_r1s0    = _vel_vshf_vvvsl(vrin_r1s0, vrin_r1s0, VE_VSHUFFLE_YUZU, vl) ;
	if( remain ) {
	  vrsum0_r1s0 = _vel_vfmads_vvvvvl(vrsum0_r1s0, vrin_r1s0, vrgout[0], vrsum0_r1s0, vl) ;
	}
#pragma clang loop unroll(full)
	for(int64_t kk=0; kk<nPacked; kk++) {
	  vrsum_r1s0[kk] = _vel_pvfmad_vvvvvl(vrsum_r1s0[kk], vrinP_r1s0, vrgoutp[kk], vrsum_r1s0[kk], vl) ;
	}

	// vrin_r1s1 : no need to use mask
	__vr vrinP_r1s1    = _vel_vshf_vvvsl(vrin_r1s1, vrin_r1s1, VE_VSHUFFLE_YUZU, vl) ;
	if( remain ) {
	  vrsum0_r1s1 = _vel_vfmads_vvvvvl(vrsum0_r1s1, vrin_r1s1, vrgout[0], vrsum0_r1s1, vl) ;
	}
#pragma clang loop unroll(full)
	for(int64_t kk=0; kk<nPacked; kk++) {
	  vrsum_r1s1[kk] = _vel_pvfmad_vvvvvl(vrsum_r1s1[kk], vrinP_r1s1, vrgoutp[kk], vrsum_r1s1[kk], vl) ;
	}

      } // gOutHeight
    } // batch

#define FILTER_OFFSET(k,c,r,s) ( gKernGroupOffset + filter_index<FLAYOUT>(k,c,r,s, inChannelGroup, gOutChannelGroup, gKernHeight, gKernWidth) )

    if( remain ){
      vrsum0_r0s0 = _vel_vfsums_vvl(vrsum0_r0s0, VLEN) ;
      _vel_vstu_vssl(vrsum0_r0s0, 4, pGKernel+FILTER_OFFSET(k, c, 0, 0), 1) ;
      vrsum0_r0s1 = _vel_vfsums_vvl(vrsum0_r0s1, VLEN) ;
      _vel_vstu_vssl(vrsum0_r0s1, 4, pGKernel+FILTER_OFFSET(k, c, 0, 1), 1) ;
      vrsum0_r1s0 = _vel_vfsums_vvl(vrsum0_r1s0, VLEN) ;
      _vel_vstu_vssl(vrsum0_r1s0, 4, pGKernel+FILTER_OFFSET(k, c, 1, 0), 1) ;
      vrsum0_r1s1 = _vel_vfsums_vvl(vrsum0_r1s1, VLEN) ;
      _vel_vstu_vssl(vrsum0_r1s1, 4, pGKernel+FILTER_OFFSET(k, c, 1, 1), 1) ;
    }

#pragma clang loop unroll(full)
    for(int64_t kk=0; kk<nPacked; kk++) {
      int64_t K = k+2*kk+remain ;
      __vr vrsum0_r0s0 = _vel_vfsums_vvl(vrsum_r0s0[kk], VLEN) ;
      __vr vrsum1_r0s0 = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsum_r0s0[kk], 32, VLEN), VLEN);
      _vel_vstu_vssl(vrsum0_r0s0, 4, pGKernel+FILTER_OFFSET(K  , c, 0, 0), 1) ;
      _vel_vstu_vssl(vrsum1_r0s0, 4, pGKernel+FILTER_OFFSET(K+1, c, 0, 0), 1) ;
      __vr vrsum0_r0s1 = _vel_vfsums_vvl(vrsum_r0s1[kk], VLEN) ;
      __vr vrsum1_r0s1 = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsum_r0s1[kk], 32, VLEN), VLEN);
      _vel_vstu_vssl(vrsum0_r0s1, 4, pGKernel+FILTER_OFFSET(K  , c, 0, 1), 1) ;
      _vel_vstu_vssl(vrsum1_r0s1, 4, pGKernel+FILTER_OFFSET(K+1, c, 0, 1), 1) ;
      __vr vrsum0_r1s0 = _vel_vfsums_vvl(vrsum_r1s0[kk], VLEN) ;
      __vr vrsum1_r1s0 = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsum_r1s0[kk], 32, VLEN), VLEN);
      _vel_vstu_vssl(vrsum0_r1s0, 4, pGKernel+FILTER_OFFSET(K  , c, 1, 0), 1) ;
      _vel_vstu_vssl(vrsum1_r1s0, 4, pGKernel+FILTER_OFFSET(K+1, c, 1, 0), 1) ;
      __vr vrsum0_r1s1 = _vel_vfsums_vvl(vrsum_r1s1[kk], VLEN) ;
      __vr vrsum1_r1s1 = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsum_r1s1[kk], 32, VLEN), VLEN);
      _vel_vstu_vssl(vrsum0_r1s1, 4, pGKernel+FILTER_OFFSET(K  , c, 1, 1), 1) ;
      _vel_vstu_vssl(vrsum1_r1s1, 4, pGKernel+FILTER_OFFSET(K+1, c, 1, 1), 1) ;
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
    const int64_t gKernWidth,		// 2
    const int64_t gKernHeight,		// 2
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
  const int64_t nY = VLEN / gOutWidth ;

  __vr vrseq = _vel_vseq_vl(VLEN) ; // xy
  __vr vry   = _vel_vdivsl_vvsl(vrseq, gOutWidth, VLEN) ;
  __vr vrx   = _vel_vsubsl_vvvl(vrseq, _vel_vmulul_vsvl(gOutWidth,vry, VLEN), VLEN) ;

  __vm256 vm_r0 =  _vel_vfmklgt_mvl(vry, VLEN) ;	// condition(y-1>=0)
  __vm256 vm_s0 =  _vel_vfmklgt_mvl(vrx, VLEN) ;	// condition(x-1>=0)

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
	 padWidth, padHeight,
	 inChannelGroup, gOutChannelGroup,
	 inGroupOffset, gOutGroupOffset, gKernGroupOffset,
	 batch, beginOChannel + k,
	 nY, vm_r0, vm_s0) ;
      k+=1 ;
      break ;
    case 2:
      func<FLAYOUT, 2>(pIn, pGOut, pGKernel,
	 gOutChannel, gOutWidth, gOutHeight,
	 inChannel, inWidth, inHeight,
	 gKernWidth, gKernHeight,
	 padWidth, padHeight,
	 inChannelGroup, gOutChannelGroup,
	 inGroupOffset, gOutGroupOffset, gKernGroupOffset,
	 batch, beginOChannel + k,
	 nY, vm_r0, vm_s0) ;
      k+=2 ;
      break ;
    case 3:
      func<FLAYOUT, 3>(pIn, pGOut, pGKernel,
	 gOutChannel, gOutWidth, gOutHeight,
	 inChannel, inWidth, inHeight,
	 gKernWidth, gKernHeight,
	 padWidth, padHeight,
	 inChannelGroup, gOutChannelGroup,
	 inGroupOffset, gOutGroupOffset, gKernGroupOffset,
	 batch, beginOChannel + k,
	 nY, vm_r0, vm_s0) ;
      k+=3 ;
      break ;
    case 4:
      func<FLAYOUT, 4>(pIn, pGOut, pGKernel,
	 gOutChannel, gOutWidth, gOutHeight,
	 inChannel, inWidth, inHeight,
	 gKernWidth, gKernHeight,
	 padWidth, padHeight,
	 inChannelGroup, gOutChannelGroup,
	 inGroupOffset, gOutGroupOffset, gKernGroupOffset,
	 batch, beginOChannel + k,
	 nY, vm_r0, vm_s0) ;
      k+=4 ;
      break ;
    case 5:
      func<FLAYOUT, 5>(pIn, pGOut, pGKernel,
	 gOutChannel, gOutWidth, gOutHeight,
	 inChannel, inWidth, inHeight,
	 gKernWidth, gKernHeight,
	 padWidth, padHeight,
	 inChannelGroup, gOutChannelGroup,
	 inGroupOffset, gOutGroupOffset, gKernGroupOffset,
	 batch, beginOChannel + k,
	 nY, vm_r0, vm_s0) ;
      k+=5 ;
      break ;
    case 6:
      func<FLAYOUT, 6>(pIn, pGOut, pGKernel,
	 gOutChannel, gOutWidth, gOutHeight,
	 inChannel, inWidth, inHeight,
	 gKernWidth, gKernHeight,
	 padWidth, padHeight,
	 inChannelGroup, gOutChannelGroup,
	 inGroupOffset, gOutGroupOffset, gKernGroupOffset,
	 batch, beginOChannel + k,
	 nY, vm_r0, vm_s0) ;
      k+=6 ;
      break ;
    case 7:
      func<FLAYOUT, 7>(pIn, pGOut, pGKernel,
	 gOutChannel, gOutWidth, gOutHeight,
	 inChannel, inWidth, inHeight,
	 gKernWidth, gKernHeight,
	 padWidth, padHeight,
	 inChannelGroup, gOutChannelGroup,
	 inGroupOffset, gOutGroupOffset, gKernGroupOffset,
	 batch, beginOChannel + k,
	 nY, vm_r0, vm_s0) ;
      k+=7 ;
      break ;
    case 8:
      func<FLAYOUT, 8>(pIn, pGOut, pGKernel,
	 gOutChannel, gOutWidth, gOutHeight,
	 inChannel, inWidth, inHeight,
	 gKernWidth, gKernHeight,
	 padWidth, padHeight,
	 inChannelGroup, gOutChannelGroup,
	 inGroupOffset, gOutGroupOffset, gKernGroupOffset,
	 batch, beginOChannel + k,
	 nY, vm_r0, vm_s0) ;
      k+=8 ;
      break ;
    case 9:
      func<FLAYOUT, 9>(pIn, pGOut, pGKernel,
	 gOutChannel, gOutWidth, gOutHeight,
	 inChannel, inWidth, inHeight,
	 gKernWidth, gKernHeight,
	 padWidth, padHeight,
	 inChannelGroup, gOutChannelGroup,
	 inGroupOffset, gOutGroupOffset, gKernGroupOffset,
	 batch, beginOChannel + k,
	 nY, vm_r0, vm_s0) ;
      k+=9 ;
      break ;
    case 10:
      func<FLAYOUT, 10>(pIn, pGOut, pGKernel,
	 gOutChannel, gOutWidth, gOutHeight,
	 inChannel, inWidth, inHeight,
	 gKernWidth, gKernHeight,
	 padWidth, padHeight,
	 inChannelGroup, gOutChannelGroup,
	 inGroupOffset, gOutGroupOffset, gKernGroupOffset,
	 batch, beginOChannel + k,
	 nY, vm_r0, vm_s0) ;
      k+=10 ;
      break ;
    case 11:
      func<FLAYOUT, 11>(pIn, pGOut, pGKernel,
	 gOutChannel, gOutWidth, gOutHeight,
	 inChannel, inWidth, inHeight,
	 gKernWidth, gKernHeight,
	 padWidth, padHeight,
	 inChannelGroup, gOutChannelGroup,
	 inGroupOffset, gOutGroupOffset, gKernGroupOffset,
	 batch, beginOChannel + k,
	 nY, vm_r0, vm_s0) ;
      k+=11 ;
      break ;
    case 12:
      func<FLAYOUT, 12>(pIn, pGOut, pGKernel,
	 gOutChannel, gOutWidth, gOutHeight,
	 inChannel, inWidth, inHeight,
	 gKernWidth, gKernHeight,
	 padWidth, padHeight,
	 inChannelGroup, gOutChannelGroup,
	 inGroupOffset, gOutGroupOffset, gKernGroupOffset,
	 batch, beginOChannel + k,
	 nY, vm_r0, vm_s0) ;
      k+=12 ;
      break ;
    case 13:
      func<FLAYOUT, 13>(pIn, pGOut, pGKernel,
	 gOutChannel, gOutWidth, gOutHeight,
	 inChannel, inWidth, inHeight,
	 gKernWidth, gKernHeight,
	 padWidth, padHeight,
	 inChannelGroup, gOutChannelGroup,
	 inGroupOffset, gOutGroupOffset, gKernGroupOffset,
	 batch, beginOChannel + k,
	 nY, vm_r0, vm_s0) ;
      k+=13 ;
      break ;
    case 14:
      func<FLAYOUT, 14>(pIn, pGOut, pGKernel,
	 gOutChannel, gOutWidth, gOutHeight,
	 inChannel, inWidth, inHeight,
	 gKernWidth, gKernHeight,
	 padWidth, padHeight,
	 inChannelGroup, gOutChannelGroup,
	 inGroupOffset, gOutGroupOffset, gKernGroupOffset,
	 batch, beginOChannel + k,
	 nY, vm_r0, vm_s0) ;
      k+=14 ;
      break ;
    case 15:
      func<FLAYOUT, 15>(pIn, pGOut, pGKernel,
	 gOutChannel, gOutWidth, gOutHeight,
	 inChannel, inWidth, inHeight,
	 gKernWidth, gKernHeight,
	 padWidth, padHeight,
	 inChannelGroup, gOutChannelGroup,
	 inGroupOffset, gOutGroupOffset, gKernGroupOffset,
	 batch, beginOChannel + k,
	 nY, vm_r0, vm_s0) ;
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
	 padWidth, padHeight,
	 inChannelGroup, gOutChannelGroup,
	 inGroupOffset, gOutGroupOffset, gKernGroupOffset,
	 batch, beginOChannel + k,
	 nY, vm_r0, vm_s0) ;
      k+=16 ;
    } // gOutChannel
  } // group
}


extern "C" vednnError_t
vednnConvolutionBackwardFilter_direct_dil1_str1_padsame_ker2_owU128(
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
