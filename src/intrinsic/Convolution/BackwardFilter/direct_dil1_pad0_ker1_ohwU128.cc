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
    const int64_t strideWidth,
    const int64_t strideHeight,
    const int64_t inChannelGroup,
    const int64_t gOutChannelGroup,
    const int64_t inGroupOffset,
    const int64_t gOutGroupOffset,
    const int64_t gKernGroupOffset,
    const int64_t batch,
    const int64_t k,
    const int64_t nY,
    const __vr vrhw,
    const __vm vm0,
    const __vm vm1
)
{
  const int64_t d0 = NUMKERNEL & 0x1 ;
  const int64_t d1 = (NUMKERNEL >> 1) & 0x1 ;
  const int64_t d2 = (NUMKERNEL >> 2) ;

  int64_t c=0 ;
  if( (inChannelGroup & 0x1 ) == 1 ) {
    __vr vrsum_d0_c0 = _vel_vbrds_vsl(0.f, VLEN) ;
    __vr vrsum_d1_c0 = _vel_pvbrd_vsl(0UL, VLEN) ;
    __vr vrsum_d2_c0[d2] ;
#pragma clang loop unroll(full)
    for(int64_t kk=0; kk<d2; kk++) {
      vrsum_d2_c0[kk] = _vel_pvbrd_vsl(0UL, VLEN) ;
    }

    {
      const int64_t vl   = gOutWidth * gOutHeight ;
      const int64_t vl_i = NUMKERNEL >= 4 ? 2 * vl : vl ;

      for (int64_t n=0; n<batch; n++) {

	const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;

	const int64_t outIndex = gOutGroupOffset + ((n * gOutChannel + k) * gOutHeight ) * gOutWidth ;

	__vr vrpin_c0 = _vel_vsfa_vvssl(vrhw, 2, (uint64_t)(pInChannel+0*inHeight*inWidth), vl_i) ;
	__vr vrin_c0  = _vel_vgtu_vvssl(vrpin_c0, 0, 0, vl_i) ;

	__vr vrgout_d2[2*d2] ;
	__vr vrgout_d1[2] ;
	__vr vrgout_d0 ;
#pragma clang loop unroll(full)
	for(int64_t kk=0; kk<2*d2; kk++) {
	  vrgout_d2[kk] = _vel_vldu_vssl(4, pGOut+outIndex+2*kk*gOutHeight*gOutWidth, 2*vl) ;
	}
	if( d1 ) {
	  vrgout_d1[0] = _vel_vldu_vssl(4, pGOut+outIndex+(4*d2+0)*gOutHeight*gOutWidth, vl) ;
	  vrgout_d1[1] = _vel_vldu_vssl(4, pGOut+outIndex+(4*d2+1)*gOutHeight*gOutWidth, vl) ;
	}
	if( d0 ) {
	  vrgout_d0 = _vel_vldu_vssl(4, pGOut+outIndex+(4*d2+2*d1)*gOutHeight*gOutWidth, vl) ;
	}

	__vr vrgoutp_d2[d2] ;
	__vr vrgoutp_d1 ;
#pragma clang loop unroll(full)
	for(int64_t kk=0; kk<d2; kk++) {
	  vrgoutp_d2[kk] = _vel_vshf_vvvsl(vrgout_d2[2*kk+0], vrgout_d2[2*kk+1], VE_VSHUFFLE_YUZU, 2*vl) ;
	}
	if( d1 ) {
	  vrgoutp_d1 = _vel_vshf_vvvsl(vrgout_d1[0], vrgout_d1[1], VE_VSHUFFLE_YUZU, vl) ;
	}

	__vr vrinP_c0 = _vel_vshf_vvvsl(vrin_c0, vrin_c0, VE_VSHUFFLE_YUZU, vl_i) ;
#pragma clang loop unroll(full)
	for(int64_t kk=0; kk<d2; kk++) {
	  vrsum_d2_c0[kk] = _vel_pvfmad_vvvvvl(vrsum_d2_c0[kk], vrinP_c0, vrgoutp_d2[kk], vrsum_d2_c0[kk], 2*vl) ;
	}
	if( d1 ) {
	  vrsum_d1_c0 = _vel_pvfmad_vvvvvl(vrsum_d1_c0, vrinP_c0, vrgoutp_d1, vrsum_d1_c0, vl) ;
	}
	if( d0 ) {
	  vrsum_d0_c0  = _vel_vfmads_vvvvvl(vrsum_d0_c0, vrin_c0, vrgout_d0, vrsum_d0_c0, vl) ;
	}
      } // batch
    } // gOutHeight

#define FILTER_OFFSET(k,c,r,s) ( gKernGroupOffset + filter_index<FLAYOUT>(k,c,r,s, inChannelGroup, gOutChannelGroup, 1, 1) )

#pragma clang loop unroll(full)
    for(int64_t kk=0; kk<d2; kk++) {
      __vr vrsum0_c0 = _vel_vfsums_vvml(vrsum_d2_c0[kk], vm0, VLEN) ;
      _vel_vstu_vssl(vrsum0_c0, 4, pGKernel+FILTER_OFFSET(k+4*kk,  c+0,0,0), 1) ;
      __vr vrsum1_c0 = _vel_vfsums_vvml(vrsum_d2_c0[kk], vm1, VLEN) ;
      _vel_vstu_vssl(vrsum1_c0, 4, pGKernel+FILTER_OFFSET(k+4*kk+1,c+0,0,0), 1) ;
      __vr vrsum2_c0 = _vel_vfsums_vvml(_vel_vsll_vvsl(vrsum_d2_c0[kk],32, VLEN), vm0, VLEN);
      _vel_vstu_vssl(vrsum2_c0, 4, pGKernel+FILTER_OFFSET(k+4*kk+2,c+0,0,0), 1) ;
      __vr vrsum3_c0 = _vel_vfsums_vvml(_vel_vsll_vvsl(vrsum_d2_c0[kk],32, VLEN), vm1, VLEN);
      _vel_vstu_vssl(vrsum3_c0, 4, pGKernel+FILTER_OFFSET(k+4*kk+3,c+0,0,0), 1) ;
    }
    if( d1 ) {
      __vr vrsum0_c0 = _vel_vfsums_vvl(vrsum_d1_c0, VLEN) ;
      _vel_vstu_vssl(vrsum0_c0, 4, pGKernel + FILTER_OFFSET(k+4*d2,  c+0,0,0), 1) ;
      __vr vrsum1_c0 = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsum_d1_c0,32, VLEN), VLEN);
      _vel_vstu_vssl(vrsum1_c0, 4, pGKernel + FILTER_OFFSET(k+4*d2+1,c+0,0,0), 1) ;
    }
    if( d0 ) {
      vrsum_d0_c0 = _vel_vfsums_vvl(vrsum_d0_c0, VLEN) ;
      _vel_vstu_vssl(vrsum_d0_c0, 4, pGKernel+FILTER_OFFSET(k+4*d2+2*d1,c+0,0,0), 1) ;
    }

#undef FILTER_OFFSET

    c+=1;
  }
  if ( ((inChannelGroup >>1) & 0x1 ) == 1) {
    __vr vrsum_d0_c0 = _vel_vbrds_vsl(0.f, VLEN) ;
    __vr vrsum_d0_c1 = _vel_vbrds_vsl(0.f, VLEN) ;
    __vr vrsum_d1_c0 = _vel_pvbrd_vsl(0UL, VLEN) ;
    __vr vrsum_d1_c1 = _vel_pvbrd_vsl(0UL, VLEN) ;
    __vr vrsum_d2_c0[d2] ;
    __vr vrsum_d2_c1[d2] ;
#pragma clang loop unroll(full)
    for(int64_t kk=0; kk<d2; kk++) {
      vrsum_d2_c0[kk] = _vel_pvbrd_vsl(0UL, VLEN) ;
      vrsum_d2_c1[kk] = _vel_pvbrd_vsl(0UL, VLEN) ;
    }

    {
      const int64_t vl   = gOutWidth * gOutHeight ;
      const int64_t vl_i = NUMKERNEL >= 4 ? 2 * vl : vl ;

      for (int64_t n=0; n<batch; n++) {

	const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;

	const int64_t outIndex = gOutGroupOffset + ((n * gOutChannel + k) * gOutHeight ) * gOutWidth ;

	__vr vrpin_c0 = _vel_vsfa_vvssl(vrhw, 2, (uint64_t)(pInChannel+0*inHeight*inWidth), vl_i) ;
	__vr vrin_c0  = _vel_vgtu_vvssl(vrpin_c0, 0, 0, vl_i) ;
	__vr vrpin_c1 = _vel_vsfa_vvssl(vrhw, 2, (uint64_t)(pInChannel+1*inHeight*inWidth), vl_i) ;
	__vr vrin_c1  = _vel_vgtu_vvssl(vrpin_c1, 0, 0, vl_i) ;

	__vr vrgout_d2[2*d2] ;
	__vr vrgout_d1[2] ;
	__vr vrgout_d0 ;
#pragma clang loop unroll(full)
	for(int64_t kk=0; kk<2*d2; kk++) {
	  vrgout_d2[kk] = _vel_vldu_vssl(4, pGOut+outIndex+2*kk*gOutHeight*gOutWidth, 2*vl) ;
	}
	if( d1 ) {
	  vrgout_d1[0] = _vel_vldu_vssl(4, pGOut+outIndex+(4*d2+0)*gOutHeight*gOutWidth, vl) ;
	  vrgout_d1[1] = _vel_vldu_vssl(4, pGOut+outIndex+(4*d2+1)*gOutHeight*gOutWidth, vl) ;
	}
	if( d0 ) {
	  vrgout_d0 = _vel_vldu_vssl(4, pGOut+outIndex+(4*d2+2*d1)*gOutHeight*gOutWidth, vl) ;
	}

	__vr vrgoutp_d2[d2] ;
	__vr vrgoutp_d1 ;
#pragma clang loop unroll(full)
	for(int64_t kk=0; kk<d2; kk++) {
	  vrgoutp_d2[kk] = _vel_vshf_vvvsl(vrgout_d2[2*kk+0], vrgout_d2[2*kk+1], VE_VSHUFFLE_YUZU, 2*vl) ;
	}
	if( d1 ) {
	  vrgoutp_d1 = _vel_vshf_vvvsl(vrgout_d1[0], vrgout_d1[1], VE_VSHUFFLE_YUZU, vl) ;
	}

	__vr vrinP_c0 = _vel_vshf_vvvsl(vrin_c0, vrin_c0, VE_VSHUFFLE_YUZU, vl_i) ;
#pragma clang loop unroll(full)
	for(int64_t kk=0; kk<d2; kk++) {
	  vrsum_d2_c0[kk] = _vel_pvfmad_vvvvvl(vrsum_d2_c0[kk], vrinP_c0, vrgoutp_d2[kk], vrsum_d2_c0[kk], 2*vl) ;
	}
	if( d1 ) {
	  vrsum_d1_c0 = _vel_pvfmad_vvvvvl(vrsum_d1_c0, vrinP_c0, vrgoutp_d1, vrsum_d1_c0, vl) ;
	}
	if( d0 ) {
	  vrsum_d0_c0  = _vel_vfmads_vvvvvl(vrsum_d0_c0, vrin_c0, vrgout_d0, vrsum_d0_c0, vl) ;
	}

	__vr vrinP_c1 = _vel_vshf_vvvsl(vrin_c1, vrin_c1, VE_VSHUFFLE_YUZU, vl_i) ;
#pragma clang loop unroll(full)
	for(int64_t kk=0; kk<d2; kk++) {
	  vrsum_d2_c1[kk] = _vel_pvfmad_vvvvvl(vrsum_d2_c1[kk], vrinP_c1, vrgoutp_d2[kk], vrsum_d2_c1[kk], 2*vl) ;
	}
	if( d1 ) {
	  vrsum_d1_c1 = _vel_pvfmad_vvvvvl(vrsum_d1_c1, vrinP_c1, vrgoutp_d1, vrsum_d1_c1, vl) ;
	}
	if( d0 ) {
	  vrsum_d0_c1  = _vel_vfmads_vvvvvl(vrsum_d0_c1, vrin_c1, vrgout_d0, vrsum_d0_c1, vl) ;
	}

      } // batch
    } // gOutHeight

#define FILTER_OFFSET(k,c,r,s) ( gKernGroupOffset + filter_index<FLAYOUT>(k,c,r,s, inChannelGroup, gOutChannelGroup, 1, 1) )

#pragma clang loop unroll(full)
    for(int64_t kk=0; kk<d2; kk++) {
      __vr vrsum0_c0 = _vel_vfsums_vvml(vrsum_d2_c0[kk], vm0, VLEN) ;
      _vel_vstu_vssl(vrsum0_c0, 4, pGKernel+FILTER_OFFSET(k+4*kk,  c+0,0,0), 1) ;
      __vr vrsum1_c0 = _vel_vfsums_vvml(vrsum_d2_c0[kk], vm1, VLEN) ;
      _vel_vstu_vssl(vrsum1_c0, 4, pGKernel+FILTER_OFFSET(k+4*kk+1,c+0,0,0), 1) ;
      __vr vrsum2_c0 = _vel_vfsums_vvml(_vel_vsll_vvsl(vrsum_d2_c0[kk],32, VLEN), vm0, VLEN);
      _vel_vstu_vssl(vrsum2_c0, 4, pGKernel+FILTER_OFFSET(k+4*kk+2,c+0,0,0), 1) ;
      __vr vrsum3_c0 = _vel_vfsums_vvml(_vel_vsll_vvsl(vrsum_d2_c0[kk],32, VLEN), vm1, VLEN);
      _vel_vstu_vssl(vrsum3_c0, 4, pGKernel+FILTER_OFFSET(k+4*kk+3,c+0,0,0), 1) ;
    }
    if( d1 ) {
      __vr vrsum0_c0 = _vel_vfsums_vvl(vrsum_d1_c0, VLEN) ;
      _vel_vstu_vssl(vrsum0_c0, 4, pGKernel + FILTER_OFFSET(k+4*d2,  c+0,0,0), 1) ;
      __vr vrsum1_c0 = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsum_d1_c0,32, VLEN), VLEN);
      _vel_vstu_vssl(vrsum1_c0, 4, pGKernel + FILTER_OFFSET(k+4*d2+1,c+0,0,0), 1) ;
    }
    if( d0 ) {
      vrsum_d0_c0 = _vel_vfsums_vvl(vrsum_d0_c0, VLEN) ;
      _vel_vstu_vssl(vrsum_d0_c0, 4, pGKernel+FILTER_OFFSET(k+4*d2+2*d1,c+0,0,0), 1) ;
    }

#pragma clang loop unroll(full)
    for(int64_t kk=0; kk<d2; kk++) {
      __vr vrsum0_c1 = _vel_vfsums_vvml(vrsum_d2_c1[kk], vm0, VLEN) ;
      _vel_vstu_vssl(vrsum0_c1, 4, pGKernel+FILTER_OFFSET(k+4*kk,  c+1,0,0), 1) ;
      __vr vrsum1_c1 = _vel_vfsums_vvml(vrsum_d2_c1[kk], vm1, VLEN) ;
      _vel_vstu_vssl(vrsum1_c1, 4, pGKernel+FILTER_OFFSET(k+4*kk+1,c+1,0,0), 1) ;
      __vr vrsum2_c1 = _vel_vfsums_vvml(_vel_vsll_vvsl(vrsum_d2_c1[kk],32, VLEN), vm0, VLEN);
      _vel_vstu_vssl(vrsum2_c1, 4, pGKernel+FILTER_OFFSET(k+4*kk+2,c+1,0,0), 1) ;
      __vr vrsum3_c1 = _vel_vfsums_vvml(_vel_vsll_vvsl(vrsum_d2_c1[kk],32, VLEN), vm1, VLEN);
      _vel_vstu_vssl(vrsum3_c1, 4, pGKernel+FILTER_OFFSET(k+4*kk+3,c+1,0,0), 1) ;
    }
    if( d1 ) {
      __vr vrsum0_c1 = _vel_vfsums_vvl(vrsum_d1_c1, VLEN) ;
      _vel_vstu_vssl(vrsum0_c1, 4, pGKernel + FILTER_OFFSET(k+4*d2,  c+1,0,0), 1) ;
      __vr vrsum1_c1 = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsum_d1_c1,32, VLEN), VLEN);
      _vel_vstu_vssl(vrsum1_c1, 4, pGKernel + FILTER_OFFSET(k+4*d2+1,c+1,0,0), 1) ;
    }
    if( d0 ) {
      vrsum_d0_c1 = _vel_vfsums_vvl(vrsum_d0_c1, VLEN) ;
      _vel_vstu_vssl(vrsum_d0_c1, 4, pGKernel+FILTER_OFFSET(k+4*d2+2*d1,c+1,0,0), 1) ;
    }

#undef FILTER_OFFSET

    c+=2;

  } // inChannel
  for ( ; c<inChannelGroup; c+=4 ) {
    __vr vrsum_d0_c0 = _vel_vbrds_vsl(0.f, VLEN) ;
    __vr vrsum_d0_c1 = _vel_vbrds_vsl(0.f, VLEN) ;
    __vr vrsum_d0_c2 = _vel_vbrds_vsl(0.f, VLEN) ;
    __vr vrsum_d0_c3 = _vel_vbrds_vsl(0.f, VLEN) ;
    __vr vrsum_d1_c0 = _vel_pvbrd_vsl(0UL, VLEN) ;
    __vr vrsum_d1_c1 = _vel_pvbrd_vsl(0UL, VLEN) ;
    __vr vrsum_d1_c2 = _vel_pvbrd_vsl(0UL, VLEN) ;
    __vr vrsum_d1_c3 = _vel_pvbrd_vsl(0UL, VLEN) ;
    __vr vrsum_d2_c0[d2] ;
    __vr vrsum_d2_c1[d2] ;
    __vr vrsum_d2_c2[d2] ;
    __vr vrsum_d2_c3[d2] ;

#pragma clang loop unroll(full)
    for(int64_t kk=0; kk<d2; kk++) {
      vrsum_d2_c0[kk] = _vel_pvbrd_vsl(0UL, VLEN) ;
      vrsum_d2_c1[kk] = _vel_pvbrd_vsl(0UL, VLEN) ;
      vrsum_d2_c2[kk] = _vel_pvbrd_vsl(0UL, VLEN) ;
      vrsum_d2_c3[kk] = _vel_pvbrd_vsl(0UL, VLEN) ;

    }

    {
      const int64_t vl   = gOutWidth * gOutHeight ;
      const int64_t vl_i = NUMKERNEL >= 4 ? 2 * vl : vl ;

      for (int64_t n=0; n<batch; n++) {

	const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;

	const int64_t outIndex = gOutGroupOffset + ((n * gOutChannel + k) * gOutHeight ) * gOutWidth ;

	__vr vrpin_c0 = _vel_vsfa_vvssl(vrhw, 2, (uint64_t)(pInChannel+0*inHeight*inWidth), vl_i) ;
	__vr vrin_c0  = _vel_vgtu_vvssl(vrpin_c0, 0, 0, vl_i) ;
	__vr vrpin_c1 = _vel_vsfa_vvssl(vrhw, 2, (uint64_t)(pInChannel+1*inHeight*inWidth), vl_i) ;
	__vr vrin_c1  = _vel_vgtu_vvssl(vrpin_c1, 0, 0, vl_i) ;
	__vr vrpin_c2 = _vel_vsfa_vvssl(vrhw, 2, (uint64_t)(pInChannel+2*inHeight*inWidth), vl_i) ;
	__vr vrin_c2  = _vel_vgtu_vvssl(vrpin_c2, 0, 0, vl_i) ;
	__vr vrpin_c3 = _vel_vsfa_vvssl(vrhw, 2, (uint64_t)(pInChannel+3*inHeight*inWidth), vl_i) ;
	__vr vrin_c3  = _vel_vgtu_vvssl(vrpin_c3, 0, 0, vl_i) ;


	__vr vrgout_d2[2*d2] ;
	__vr vrgout_d1[2] ;
	__vr vrgout_d0 ;
#pragma clang loop unroll(full)
	for(int64_t kk=0; kk<2*d2; kk++) {
	  vrgout_d2[kk] = _vel_vldu_vssl(4, pGOut+outIndex+2*kk*gOutHeight*gOutWidth, 2*vl) ;
	}
	if( d1 ) {
	  vrgout_d1[0] = _vel_vldu_vssl(4, pGOut+outIndex+(4*d2+0)*gOutHeight*gOutWidth, vl) ;
	  vrgout_d1[1] = _vel_vldu_vssl(4, pGOut+outIndex+(4*d2+1)*gOutHeight*gOutWidth, vl) ;
	}
	if( d0 ) {
	  vrgout_d0 = _vel_vldu_vssl(4, pGOut+outIndex+(4*d2+2*d1)*gOutHeight*gOutWidth, vl) ;
	}

	__vr vrgoutp_d2[d2] ;
	__vr vrgoutp_d1 ;
#pragma clang loop unroll(full)
	for(int64_t kk=0; kk<d2; kk++) {
	  vrgoutp_d2[kk] = _vel_vshf_vvvsl(vrgout_d2[2*kk+0], vrgout_d2[2*kk+1], VE_VSHUFFLE_YUZU, 2*vl) ;
	}
	if( d1 ) {
	  vrgoutp_d1 = _vel_vshf_vvvsl(vrgout_d1[0], vrgout_d1[1], VE_VSHUFFLE_YUZU, vl) ;
	}

	__vr vrinP_c0 = _vel_vshf_vvvsl(vrin_c0, vrin_c0, VE_VSHUFFLE_YUZU, vl_i) ;
#pragma clang loop unroll(full)
	for(int64_t kk=0; kk<d2; kk++) {
	  vrsum_d2_c0[kk] = _vel_pvfmad_vvvvvl(vrsum_d2_c0[kk], vrinP_c0, vrgoutp_d2[kk], vrsum_d2_c0[kk], 2*vl) ;
	}
	if( d1 ) {
	  vrsum_d1_c0 = _vel_pvfmad_vvvvvl(vrsum_d1_c0, vrinP_c0, vrgoutp_d1, vrsum_d1_c0, vl) ;
	}
	if( d0 ) {
	  vrsum_d0_c0  = _vel_vfmads_vvvvvl(vrsum_d0_c0, vrin_c0, vrgout_d0, vrsum_d0_c0, vl) ;
	}

	__vr vrinP_c1 = _vel_vshf_vvvsl(vrin_c1, vrin_c1, VE_VSHUFFLE_YUZU, vl_i) ;
#pragma clang loop unroll(full)
	for(int64_t kk=0; kk<d2; kk++) {
	  vrsum_d2_c1[kk] = _vel_pvfmad_vvvvvl(vrsum_d2_c1[kk], vrinP_c1, vrgoutp_d2[kk], vrsum_d2_c1[kk], 2*vl) ;
	}
	if( d1 ) {
	  vrsum_d1_c1 = _vel_pvfmad_vvvvvl(vrsum_d1_c1, vrinP_c1, vrgoutp_d1, vrsum_d1_c1, vl) ;
	}
	if( d0 ) {
	  vrsum_d0_c1  = _vel_vfmads_vvvvvl(vrsum_d0_c1, vrin_c1, vrgout_d0, vrsum_d0_c1, vl) ;
	}

	__vr vrinP_c2 = _vel_vshf_vvvsl(vrin_c2, vrin_c2, VE_VSHUFFLE_YUZU, vl_i) ;
#pragma clang loop unroll(full)
	for(int64_t kk=0; kk<d2; kk++) {
	  vrsum_d2_c2[kk] = _vel_pvfmad_vvvvvl(vrsum_d2_c2[kk], vrinP_c2, vrgoutp_d2[kk], vrsum_d2_c2[kk], 2*vl) ;
	}
	if( d1 ) {
	  vrsum_d1_c2 = _vel_pvfmad_vvvvvl(vrsum_d1_c2, vrinP_c2, vrgoutp_d1, vrsum_d1_c2, vl) ;
	}
	if( d0 ) {
	  vrsum_d0_c2  = _vel_vfmads_vvvvvl(vrsum_d0_c2, vrin_c2, vrgout_d0, vrsum_d0_c2, vl) ;
	}

	__vr vrinP_c3 = _vel_vshf_vvvsl(vrin_c3, vrin_c3, VE_VSHUFFLE_YUZU, vl_i) ;
#pragma clang loop unroll(full)
	for(int64_t kk=0; kk<d2; kk++) {
	  vrsum_d2_c3[kk] = _vel_pvfmad_vvvvvl(vrsum_d2_c3[kk], vrinP_c3, vrgoutp_d2[kk], vrsum_d2_c3[kk], 2*vl) ;
	}
	if( d1 ) {
	  vrsum_d1_c3 = _vel_pvfmad_vvvvvl(vrsum_d1_c3, vrinP_c3, vrgoutp_d1, vrsum_d1_c3, vl) ;
	}
	if( d0 ) {
	  vrsum_d0_c3  = _vel_vfmads_vvvvvl(vrsum_d0_c3, vrin_c3, vrgout_d0, vrsum_d0_c3, vl) ;
	}

      } // batch
    } // gOutHeight

#define FILTER_OFFSET(k,c,r,s) ( gKernGroupOffset + filter_index<FLAYOUT>(k,c,r,s, inChannelGroup, gOutChannelGroup, 1, 1) )

#pragma clang loop unroll(full)
    for(int64_t kk=0; kk<d2; kk++) {
      __vr vrsum0_c0 = _vel_vfsums_vvml(vrsum_d2_c0[kk], vm0, VLEN) ;
      _vel_vstu_vssl(vrsum0_c0, 4, pGKernel+FILTER_OFFSET(k+4*kk,  c+0,0,0), 1) ;
      __vr vrsum1_c0 = _vel_vfsums_vvml(vrsum_d2_c0[kk], vm1, VLEN) ;
      _vel_vstu_vssl(vrsum1_c0, 4, pGKernel+FILTER_OFFSET(k+4*kk+1,c+0,0,0), 1) ;
      __vr vrsum2_c0 = _vel_vfsums_vvml(_vel_vsll_vvsl(vrsum_d2_c0[kk],32, VLEN), vm0, VLEN);
      _vel_vstu_vssl(vrsum2_c0, 4, pGKernel+FILTER_OFFSET(k+4*kk+2,c+0,0,0), 1) ;
      __vr vrsum3_c0 = _vel_vfsums_vvml(_vel_vsll_vvsl(vrsum_d2_c0[kk],32, VLEN), vm1, VLEN);
      _vel_vstu_vssl(vrsum3_c0, 4, pGKernel+FILTER_OFFSET(k+4*kk+3,c+0,0,0), 1) ;
    }
    if( d1 ) {
      __vr vrsum0_c0 = _vel_vfsums_vvl(vrsum_d1_c0, VLEN) ;
      _vel_vstu_vssl(vrsum0_c0, 4, pGKernel + FILTER_OFFSET(k+4*d2,  c+0,0,0), 1) ;
      __vr vrsum1_c0 = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsum_d1_c0,32, VLEN), VLEN);
      _vel_vstu_vssl(vrsum1_c0, 4, pGKernel + FILTER_OFFSET(k+4*d2+1,c+0,0,0), 1) ;
    }
    if( d0 ) {
      vrsum_d0_c0 = _vel_vfsums_vvl(vrsum_d0_c0, VLEN) ;
      _vel_vstu_vssl(vrsum_d0_c0, 4, pGKernel+FILTER_OFFSET(k+4*d2+2*d1,c+0,0,0), 1) ;
    }

#pragma clang loop unroll(full)
    for(int64_t kk=0; kk<d2; kk++) {
      __vr vrsum0_c1 = _vel_vfsums_vvml(vrsum_d2_c1[kk], vm0, VLEN) ;
      _vel_vstu_vssl(vrsum0_c1, 4, pGKernel+FILTER_OFFSET(k+4*kk,  c+1,0,0), 1) ;
      __vr vrsum1_c1 = _vel_vfsums_vvml(vrsum_d2_c1[kk], vm1, VLEN) ;
      _vel_vstu_vssl(vrsum1_c1, 4, pGKernel+FILTER_OFFSET(k+4*kk+1,c+1,0,0), 1) ;
      __vr vrsum2_c1 = _vel_vfsums_vvml(_vel_vsll_vvsl(vrsum_d2_c1[kk],32, VLEN), vm0, VLEN);
      _vel_vstu_vssl(vrsum2_c1, 4, pGKernel+FILTER_OFFSET(k+4*kk+2,c+1,0,0), 1) ;
      __vr vrsum3_c1 = _vel_vfsums_vvml(_vel_vsll_vvsl(vrsum_d2_c1[kk],32, VLEN), vm1, VLEN);
      _vel_vstu_vssl(vrsum3_c1, 4, pGKernel+FILTER_OFFSET(k+4*kk+3,c+1,0,0), 1) ;
    }
    if( d1 ) {
      __vr vrsum0_c1 = _vel_vfsums_vvl(vrsum_d1_c1, VLEN) ;
      _vel_vstu_vssl(vrsum0_c1, 4, pGKernel + FILTER_OFFSET(k+4*d2,  c+1,0,0), 1) ;
      __vr vrsum1_c1 = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsum_d1_c1,32, VLEN), VLEN);
      _vel_vstu_vssl(vrsum1_c1, 4, pGKernel + FILTER_OFFSET(k+4*d2+1,c+1,0,0), 1) ;
    }
    if( d0 ) {
      vrsum_d0_c1 = _vel_vfsums_vvl(vrsum_d0_c1, VLEN) ;
      _vel_vstu_vssl(vrsum_d0_c1, 4, pGKernel+FILTER_OFFSET(k+4*d2+2*d1,c+1,0,0), 1) ;
    }

#pragma clang loop unroll(full)
    for(int64_t kk=0; kk<d2; kk++) {
      __vr vrsum0_c2 = _vel_vfsums_vvml(vrsum_d2_c2[kk], vm0, VLEN) ;
      _vel_vstu_vssl(vrsum0_c2, 4, pGKernel+FILTER_OFFSET(k+4*kk,  c+2,0,0), 1) ;
      __vr vrsum1_c2 = _vel_vfsums_vvml(vrsum_d2_c2[kk], vm1, VLEN) ;
      _vel_vstu_vssl(vrsum1_c2, 4, pGKernel+FILTER_OFFSET(k+4*kk+1,c+2,0,0), 1) ;
      __vr vrsum2_c2 = _vel_vfsums_vvml(_vel_vsll_vvsl(vrsum_d2_c2[kk],32, VLEN), vm0, VLEN);
      _vel_vstu_vssl(vrsum2_c2, 4, pGKernel+FILTER_OFFSET(k+4*kk+2,c+2,0,0), 1) ;
      __vr vrsum3_c2 = _vel_vfsums_vvml(_vel_vsll_vvsl(vrsum_d2_c2[kk],32, VLEN), vm1, VLEN);
      _vel_vstu_vssl(vrsum3_c2, 4, pGKernel+FILTER_OFFSET(k+4*kk+3,c+2,0,0), 1) ;
    }
    if( d1 ) {
      __vr vrsum0_c2 = _vel_vfsums_vvl(vrsum_d1_c2, VLEN) ;
      _vel_vstu_vssl(vrsum0_c2, 4, pGKernel + FILTER_OFFSET(k+4*d2,  c+2,0,0), 1) ;
      __vr vrsum1_c2 = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsum_d1_c2,32, VLEN), VLEN);
      _vel_vstu_vssl(vrsum1_c2, 4, pGKernel + FILTER_OFFSET(k+4*d2+1,c+2,0,0), 1) ;
    }
    if( d0 ) {
      vrsum_d0_c2 = _vel_vfsums_vvl(vrsum_d0_c2, VLEN) ;
      _vel_vstu_vssl(vrsum_d0_c2, 4, pGKernel+FILTER_OFFSET(k+4*d2+2*d1,c+2,0,0), 1) ;
    }

#pragma clang loop unroll(full)
    for(int64_t kk=0; kk<d2; kk++) {
      __vr vrsum0_c3 = _vel_vfsums_vvml(vrsum_d2_c3[kk], vm0, VLEN) ;
      _vel_vstu_vssl(vrsum0_c3, 4, pGKernel+FILTER_OFFSET(k+4*kk,  c+3,0,0), 1) ;
      __vr vrsum1_c3 = _vel_vfsums_vvml(vrsum_d2_c3[kk], vm1, VLEN) ;
      _vel_vstu_vssl(vrsum1_c3, 4, pGKernel+FILTER_OFFSET(k+4*kk+1,c+3,0,0), 1) ;
      __vr vrsum2_c3 = _vel_vfsums_vvml(_vel_vsll_vvsl(vrsum_d2_c3[kk],32, VLEN), vm0, VLEN);
      _vel_vstu_vssl(vrsum2_c3, 4, pGKernel+FILTER_OFFSET(k+4*kk+2,c+3,0,0), 1) ;
      __vr vrsum3_c3 = _vel_vfsums_vvml(_vel_vsll_vvsl(vrsum_d2_c3[kk],32, VLEN), vm1, VLEN);
      _vel_vstu_vssl(vrsum3_c3, 4, pGKernel+FILTER_OFFSET(k+4*kk+3,c+3,0,0), 1) ;
    }
    if( d1 ) {
      __vr vrsum0_c3 = _vel_vfsums_vvl(vrsum_d1_c3, VLEN) ;
      _vel_vstu_vssl(vrsum0_c3, 4, pGKernel + FILTER_OFFSET(k+4*d2,  c+3,0,0), 1) ;
      __vr vrsum1_c3 = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsum_d1_c3,32, VLEN), VLEN);
      _vel_vstu_vssl(vrsum1_c3, 4, pGKernel + FILTER_OFFSET(k+4*d2+1,c+3,0,0), 1) ;
    }
    if( d0 ) {
      vrsum_d0_c3 = _vel_vfsums_vvl(vrsum_d0_c3, VLEN) ;
      _vel_vstu_vssl(vrsum_d0_c3, 4, pGKernel+FILTER_OFFSET(k+4*d2+2*d1,c+3,0,0), 1) ;
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
    const int64_t gKernWidth,		// 1
    const int64_t gKernHeight,		// 1
    const int64_t inChannelGroup,
    const int64_t gOutChannelGroup,
    const int64_t beginOChannel,
    const int64_t nOChannel,
    const int64_t beginGroup,
    const int64_t nGroup,
    const int64_t strideWidth,
    const int64_t strideHeight,
    const int64_t padWidth,		// 0
    const int64_t padHeight,		// 0
    const int64_t dilationWidth,	// 1
    const int64_t dilationHeight	// 1
)
{
  const int64_t nY = VLEN / gOutWidth ;

  __vr vrseq = _vel_vseq_vl(VLEN) ;			// xy
  __vr vry  = _vel_vdivsl_vvsl(vrseq, gOutWidth, VLEN) ;
  __vr vrx  = _vel_vsubsl_vvvl(vrseq, _vel_vmulul_vsvl(gOutWidth,vry, VLEN), VLEN) ;

  __vm256 vm0 =  _vel_vfmkllt_mvl(_vel_vaddsl_vsvl(-gOutHeight*gOutWidth, vrseq, VLEN), VLEN) ;
  __vm256 vm1 = _vel_negm_mm(vm0) ;

  __vr vri  = _vel_vmulsl_vsvl(strideHeight, vry, VLEN) ;
  __vr vrj  = _vel_vmulsl_vsvl(strideWidth,  vrx, VLEN) ;

  __vr vrhw = _vel_vaddul_vvvl(vrj, _vel_vmulul_vsvl(inWidth,vri, VLEN), VLEN) ;
  vrhw = _vel_vmrg_vvvml(vrhw, _vel_vmv_vsvl(-gOutHeight*gOutWidth,vrhw, VLEN), vm1, VLEN) ;

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
	 strideWidth, strideHeight,
	 inChannelGroup, gOutChannelGroup,
	 inGroupOffset, gOutGroupOffset, gKernGroupOffset,
	 batch, beginOChannel + k,
	 nY, vrhw, vm0, vm1 ) ;
      k+=1 ;
      break ;
    case 2:
      func<FLAYOUT, 2>(pIn, pGOut, pGKernel,
	 gOutChannel, gOutWidth, gOutHeight,
	 inChannel, inWidth, inHeight,
	 strideWidth, strideHeight,
	 inChannelGroup, gOutChannelGroup,
	 inGroupOffset, gOutGroupOffset, gKernGroupOffset,
	 batch, beginOChannel + k,
	 nY, vrhw, vm0, vm1 ) ;
      k+=2 ;
      break ;
    case 3:
      func<FLAYOUT, 3>(pIn, pGOut, pGKernel,
	 gOutChannel, gOutWidth, gOutHeight,
	 inChannel, inWidth, inHeight,
	 strideWidth, strideHeight,
	 inChannelGroup, gOutChannelGroup,
	 inGroupOffset, gOutGroupOffset, gKernGroupOffset,
	 batch, beginOChannel + k,
	 nY, vrhw, vm0, vm1 ) ;
      k+=3 ;
      break ;
    case 4:
      func<FLAYOUT, 4>(pIn, pGOut, pGKernel,
	 gOutChannel, gOutWidth, gOutHeight,
	 inChannel, inWidth, inHeight,
	 strideWidth, strideHeight,
	 inChannelGroup, gOutChannelGroup,
	 inGroupOffset, gOutGroupOffset, gKernGroupOffset,
	 batch, beginOChannel + k,
	 nY, vrhw, vm0, vm1 ) ;
      k+=4 ;
      break ;
    case 5:
      func<FLAYOUT, 5>(pIn, pGOut, pGKernel,
	 gOutChannel, gOutWidth, gOutHeight,
	 inChannel, inWidth, inHeight,
	 strideWidth, strideHeight,
	 inChannelGroup, gOutChannelGroup,
	 inGroupOffset, gOutGroupOffset, gKernGroupOffset,
	 batch, beginOChannel + k,
	 nY, vrhw, vm0, vm1 ) ;
      k+=5 ;
      break ;
    case 6:
      func<FLAYOUT, 6>(pIn, pGOut, pGKernel,
	 gOutChannel, gOutWidth, gOutHeight,
	 inChannel, inWidth, inHeight,
	 strideWidth, strideHeight,
	 inChannelGroup, gOutChannelGroup,
	 inGroupOffset, gOutGroupOffset, gKernGroupOffset,
	 batch, beginOChannel + k,
	 nY, vrhw, vm0, vm1 ) ;
      k+=6 ;
      break ;
    case 7:
      func<FLAYOUT, 7>(pIn, pGOut, pGKernel,
	 gOutChannel, gOutWidth, gOutHeight,
	 inChannel, inWidth, inHeight,
	 strideWidth, strideHeight,
	 inChannelGroup, gOutChannelGroup,
	 inGroupOffset, gOutGroupOffset, gKernGroupOffset,
	 batch, beginOChannel + k,
	 nY, vrhw, vm0, vm1 ) ;
      k+=7 ;
      break ;
    case 8:
      func<FLAYOUT, 8>(pIn, pGOut, pGKernel,
	 gOutChannel, gOutWidth, gOutHeight,
	 inChannel, inWidth, inHeight,
	 strideWidth, strideHeight,
	 inChannelGroup, gOutChannelGroup,
	 inGroupOffset, gOutGroupOffset, gKernGroupOffset,
	 batch, beginOChannel + k,
	 nY, vrhw, vm0, vm1 ) ;

      k+=8 ;
      break ;
    case 9:
      func<FLAYOUT, 9>(pIn, pGOut, pGKernel,
	 gOutChannel, gOutWidth, gOutHeight,
	 inChannel, inWidth, inHeight,
	 strideWidth, strideHeight,
	 inChannelGroup, gOutChannelGroup,
	 inGroupOffset, gOutGroupOffset, gKernGroupOffset,
	 batch, beginOChannel + k,
	 nY, vrhw, vm0, vm1 ) ;
      k+=9 ;
      break ;
    case 10:
      func<FLAYOUT, 10>(pIn, pGOut, pGKernel,
	 gOutChannel, gOutWidth, gOutHeight,
	 inChannel, inWidth, inHeight,
	 strideWidth, strideHeight,
	 inChannelGroup, gOutChannelGroup,
	 inGroupOffset, gOutGroupOffset, gKernGroupOffset,
	 batch, beginOChannel + k,
	 nY, vrhw, vm0, vm1 ) ;
      k+=10 ;
      break ;
    case 11:
      func<FLAYOUT, 11>(pIn, pGOut, pGKernel,
	 gOutChannel, gOutWidth, gOutHeight,
	 inChannel, inWidth, inHeight,
	 strideWidth, strideHeight,
	 inChannelGroup, gOutChannelGroup,
	 inGroupOffset, gOutGroupOffset, gKernGroupOffset,
	 batch, beginOChannel + k,
	 nY, vrhw, vm0, vm1 ) ;
      k+=11 ;
      break ;
    case 12:
      func<FLAYOUT, 12>(pIn, pGOut, pGKernel,
	 gOutChannel, gOutWidth, gOutHeight,
	 inChannel, inWidth, inHeight,
	 strideWidth, strideHeight,
	 inChannelGroup, gOutChannelGroup,
	 inGroupOffset, gOutGroupOffset, gKernGroupOffset,
	 batch, beginOChannel + k,
	 nY, vrhw, vm0, vm1 ) ;
      k+=12 ;
      break ;
    case 13:
      func<FLAYOUT, 13>(pIn, pGOut, pGKernel,
	 gOutChannel, gOutWidth, gOutHeight,
	 inChannel, inWidth, inHeight,
	 strideWidth, strideHeight,
	 inChannelGroup, gOutChannelGroup,
	 inGroupOffset, gOutGroupOffset, gKernGroupOffset,
	 batch, beginOChannel + k,
	 nY, vrhw, vm0, vm1 ) ;
      k+=13 ;
      break ;
    case 14:
      func<FLAYOUT, 14>(pIn, pGOut, pGKernel,
	 gOutChannel, gOutWidth, gOutHeight,
	 inChannel, inWidth, inHeight,
	 strideWidth, strideHeight,
	 inChannelGroup, gOutChannelGroup,
	 inGroupOffset, gOutGroupOffset, gKernGroupOffset,
	 batch, beginOChannel + k,
	 nY, vrhw, vm0, vm1 ) ;
      k+=14 ;
      break ;
    case 15:
      func<FLAYOUT, 15>(pIn, pGOut, pGKernel,
	 gOutChannel, gOutWidth, gOutHeight,
	 inChannel, inWidth, inHeight,
	 strideWidth, strideHeight,
	 inChannelGroup, gOutChannelGroup,
	 inGroupOffset, gOutGroupOffset, gKernGroupOffset,
	 batch, beginOChannel + k,
	 nY, vrhw, vm0, vm1 ) ;
      k+=15 ;
      break ;
    default :
      break ;
    }
    for (; k<nOChannel; ) {
      func<FLAYOUT, 16>(pIn, pGOut, pGKernel,
	 gOutChannel, gOutWidth, gOutHeight,
	 inChannel, inWidth, inHeight,
	 strideWidth, strideHeight,
	 inChannelGroup, gOutChannelGroup,
	 inGroupOffset, gOutGroupOffset, gKernGroupOffset,
	 batch, beginOChannel + k,
	 nY, vrhw, vm0, vm1 ) ;
      k+=16 ;
    } // gOutChannel
  } // group
}


extern "C" vednnError_t
vednnConvolutionBackwardFilter_direct_dil1_pad0_ker1_ohwU128(
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

