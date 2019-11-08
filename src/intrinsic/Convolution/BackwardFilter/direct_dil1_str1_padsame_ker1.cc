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
    const int64_t inChannelGroup,
    const int64_t gOutChannelGroup,
    const int64_t inGroupOffset,
    const int64_t gOutGroupOffset,
    const int64_t gKernGroupOffset,
    const int64_t batch,
    const int64_t k )
{
  const int64_t remain  = NUMKERNEL & 0x1 ;
  const int64_t nPacked = NUMKERNEL >> 1 ;

  int64_t c=0;

  if( (inChannelGroup & 0x1) == 1 ) {

    __vr vrsum0_c0  = _vel_vbrds_vsl(0.f, VLEN) ;
    __vr vrsum_c0[nPacked] ;
#pragma clang loop unroll(full)
    for(int64_t kk=0; kk<nPacked; kk++) {
      vrsum_c0[kk] = _vel_pvbrd_vsl(0UL, VLEN) ;
    }

    for (int64_t n=0; n<batch; n++) {
      for (int64_t xy = 0; xy < gOutHeight * gOutWidth ; xy += VLEN ) {

	const int64_t vl = gOutHeight * gOutWidth - xy < VLEN ? gOutHeight * gOutWidth - xy  : VLEN ;

	const float *pInChannel = pIn + inGroupOffset + n * inChannel * inHeight * inWidth  ;

	const int64_t outIndex = gOutGroupOffset + ((n * gOutChannel + k) * gOutHeight ) * gOutWidth + xy ;


	__vr vrin_c0  = _vel_vldu_vssl(4,&pInChannel[(c+0)*inHeight*inWidth+xy], vl) ;

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

#define VFADD(VRIN, VRSUM0, VRSUM)								\
	{											\
	  __vr vrinP = _vel_vshf_vvvsl(VRIN, VRIN, VE_VSHUFFLE_YUZU, vl) ;			\
	  if( remain ) {									\
	    VRSUM0  = _vel_vfmads_vvvvvl(VRSUM0, VRIN, vrgout[0], VRSUM0, vl) ;			\
	  }											\
          _Pragma("clang loop unroll(full)")							\
	  for(int64_t kk=0; kk<nPacked; kk++) {							\
            VRSUM[kk] = _vel_pvfmad_vvvvvl(VRSUM[kk], vrinP, vrgoutp[kk], VRSUM[kk], vl) ;	\
	  }											\
	}

	VFADD(vrin_c0, vrsum0_c0, vrsum_c0) ;

      } // gOutHeight * gOutWidth
    } // batch

#define FILTER_OFFSET(k,c,r,s) ( gKernGroupOffset + filter_index<FLAYOUT>(k,c,r,s, inChannelGroup, gOutChannelGroup, 1, 1) )


    if( remain ) {
      vrsum0_c0 = _vel_vfsums_vvl(vrsum0_c0, VLEN) ;
      _vel_vstu_vssl(vrsum0_c0, 4, pGKernel+FILTER_OFFSET(k+0,c+0,0,0), 1) ;
    }

#pragma clang loop unroll(full)
    for(int64_t kk=0; kk<nPacked; kk++) {
      __vr vrsumU_c0 = _vel_vfsums_vvl(vrsum_c0[kk], VLEN) ;
      _vel_vstu_vssl(vrsumU_c0, 4, pGKernel + FILTER_OFFSET(k+2*kk+remain,  c+0,0,0), 1) ;
      __vr vrsumL_c0 = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsum_c0[kk],32, VLEN), VLEN);
      _vel_vstu_vssl(vrsumL_c0, 4, pGKernel + FILTER_OFFSET(k+2*kk+remain+1,c+0,0,0), 1) ;
    }

#undef FILTER_OFFSET

    c+=1 ;
  }
  if( ((inChannelGroup >> 1) & 0x1) == 1 ) {

    __vr vrsum0_c0  = _vel_vbrds_vsl(0.f, VLEN) ;
    __vr vrsum0_c1  = _vel_vbrds_vsl(0.f, VLEN) ;
    __vr vrsum_c0[nPacked] ;
    __vr vrsum_c1[nPacked] ;
#pragma clang loop unroll(full)
    for(int64_t kk=0; kk<nPacked; kk++) {
      vrsum_c0[kk] = _vel_pvbrd_vsl(0UL, VLEN) ;
      vrsum_c1[kk] = _vel_pvbrd_vsl(0UL, VLEN) ;
    }

    for (int64_t n=0; n<batch; n++) {
      for (int64_t xy = 0; xy < gOutHeight * gOutWidth ; xy += VLEN ) {

	const int64_t vl = gOutHeight * gOutWidth - xy < VLEN ? gOutHeight * gOutWidth - xy  : VLEN ;

	const float *pInChannel = pIn + inGroupOffset + n * inChannel * inHeight * inWidth  ;

	const int64_t outIndex = gOutGroupOffset + ((n * gOutChannel + k) * gOutHeight ) * gOutWidth + xy ;


	__vr vrin_c0  = _vel_vldu_vssl(4,&pInChannel[(c+0)*inHeight*inWidth+xy], vl) ;
	__vr vrin_c1  = _vel_vldu_vssl(4,&pInChannel[(c+1)*inHeight*inWidth+xy], vl) ;

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

	VFADD(vrin_c0, vrsum0_c0, vrsum_c0) ;
	VFADD(vrin_c1, vrsum0_c1, vrsum_c1) ;

      } // gOutHeight * gOutWidth
    } // batch

#define FILTER_OFFSET(k,c,r,s) ( gKernGroupOffset + filter_index<FLAYOUT>(k,c,r,s, inChannelGroup, gOutChannelGroup, 1, 1) )


    if( remain ) {
      vrsum0_c0 = _vel_vfsums_vvl(vrsum0_c0, VLEN) ;
      _vel_vstu_vssl(vrsum0_c0, 4, pGKernel+FILTER_OFFSET(k+0,c+0,0,0), 1) ;
      vrsum0_c1 = _vel_vfsums_vvl(vrsum0_c1, VLEN) ;
      _vel_vstu_vssl(vrsum0_c1, 4, pGKernel+FILTER_OFFSET(k+0,c+1,0,0), 1) ;
    }

#pragma clang loop unroll(full)
    for(int64_t kk=0; kk<nPacked; kk++) {
      __vr vrsumU_c0 = _vel_vfsums_vvl(vrsum_c0[kk], VLEN) ;
      _vel_vstu_vssl(vrsumU_c0, 4, pGKernel + FILTER_OFFSET(k+2*kk+remain,  c+0,0,0), 1) ;
      __vr vrsumL_c0 = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsum_c0[kk],32, VLEN), VLEN);
      _vel_vstu_vssl(vrsumL_c0, 4, pGKernel + FILTER_OFFSET(k+2*kk+remain+1,c+0,0,0), 1) ;
      __vr vrsumU_c1 = _vel_vfsums_vvl(vrsum_c1[kk], VLEN) ;
      _vel_vstu_vssl(vrsumU_c1, 4, pGKernel + FILTER_OFFSET(k+2*kk+remain,  c+1,0,0), 1) ;
      __vr vrsumL_c1 = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsum_c1[kk],32, VLEN), VLEN);
      _vel_vstu_vssl(vrsumL_c1, 4, pGKernel + FILTER_OFFSET(k+2*kk+remain+1,c+1,0,0), 1) ;
    }

#undef FILTER_OFFSET


    c+=2 ;
  }
  if( ((inChannelGroup >> 2) & 0x1) == 1 ) {
    __vr vrsum0_c0  = _vel_vbrds_vsl(0.f, VLEN) ;
    __vr vrsum0_c1  = _vel_vbrds_vsl(0.f, VLEN) ;
    __vr vrsum0_c2  = _vel_vbrds_vsl(0.f, VLEN) ;
    __vr vrsum0_c3  = _vel_vbrds_vsl(0.f, VLEN) ;

    __vr vrsum_c0[nPacked] ;
    __vr vrsum_c1[nPacked] ;
    __vr vrsum_c2[nPacked] ;
    __vr vrsum_c3[nPacked] ;
#pragma clang loop unroll(full)
    for(int64_t kk=0; kk<nPacked; kk++) {
      vrsum_c0[kk] = _vel_pvbrd_vsl(0UL, VLEN) ;
      vrsum_c1[kk] = _vel_pvbrd_vsl(0UL, VLEN) ;
      vrsum_c2[kk] = _vel_pvbrd_vsl(0UL, VLEN) ;
      vrsum_c3[kk] = _vel_pvbrd_vsl(0UL, VLEN) ;
    }

    for (int64_t n=0; n<batch; n++) {
      for (int64_t xy = 0; xy < gOutHeight * gOutWidth ; xy += VLEN ) {

	const int64_t vl = gOutHeight * gOutWidth - xy < VLEN ? gOutHeight * gOutWidth - xy  : VLEN ;

	const float *pInChannel = pIn + inGroupOffset + n * inChannel * inHeight * inWidth  ;

	const int64_t outIndex = gOutGroupOffset + ((n * gOutChannel + k) * gOutHeight ) * gOutWidth + xy ;


	__vr vrin_c0  = _vel_vldu_vssl(4,&pInChannel[(c+0)*inHeight*inWidth+xy], vl) ;
	__vr vrin_c1  = _vel_vldu_vssl(4,&pInChannel[(c+1)*inHeight*inWidth+xy], vl) ;
	__vr vrin_c2  = _vel_vldu_vssl(4,&pInChannel[(c+2)*inHeight*inWidth+xy], vl) ;
	__vr vrin_c3  = _vel_vldu_vssl(4,&pInChannel[(c+3)*inHeight*inWidth+xy], vl) ;

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

	VFADD(vrin_c0, vrsum0_c0, vrsum_c0) ;
	VFADD(vrin_c1, vrsum0_c1, vrsum_c1) ;
	VFADD(vrin_c2, vrsum0_c2, vrsum_c2) ;
	VFADD(vrin_c3, vrsum0_c3, vrsum_c3) ;

      } // gOutHeight * gOutWidth
    } // batch

#define FILTER_OFFSET(k,c,r,s) ( gKernGroupOffset + filter_index<FLAYOUT>(k,c,r,s, inChannelGroup, gOutChannelGroup, 1, 1) )


    if( remain ) {
      vrsum0_c0 = _vel_vfsums_vvl(vrsum0_c0, VLEN) ;
      _vel_vstu_vssl(vrsum0_c0, 4, pGKernel+FILTER_OFFSET(k+0,c+0,0,0), 1) ;
      vrsum0_c1 = _vel_vfsums_vvl(vrsum0_c1, VLEN) ;
      _vel_vstu_vssl(vrsum0_c1, 4, pGKernel+FILTER_OFFSET(k+0,c+1,0,0), 1) ;
      vrsum0_c2 = _vel_vfsums_vvl(vrsum0_c2, VLEN) ;
      _vel_vstu_vssl(vrsum0_c2, 4, pGKernel+FILTER_OFFSET(k+0,c+2,0,0), 1) ;
      vrsum0_c3 = _vel_vfsums_vvl(vrsum0_c3, VLEN) ;
      _vel_vstu_vssl(vrsum0_c3, 4, pGKernel+FILTER_OFFSET(k+0,c+3,0,0), 1) ;

    }

#pragma clang loop unroll(full)
    for(int64_t kk=0; kk<nPacked; kk++) {
      __vr vrsumU_c0 = _vel_vfsums_vvl(vrsum_c0[kk], VLEN) ;
      _vel_vstu_vssl(vrsumU_c0, 4, pGKernel + FILTER_OFFSET(k+2*kk+remain,  c+0,0,0), 1) ;
      __vr vrsumL_c0 = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsum_c0[kk],32, VLEN), VLEN);
      _vel_vstu_vssl(vrsumL_c0, 4, pGKernel + FILTER_OFFSET(k+2*kk+remain+1,c+0,0,0), 1) ;
      __vr vrsumU_c1 = _vel_vfsums_vvl(vrsum_c1[kk], VLEN) ;
      _vel_vstu_vssl(vrsumU_c1, 4, pGKernel + FILTER_OFFSET(k+2*kk+remain,  c+1,0,0), 1) ;
      __vr vrsumL_c1 = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsum_c1[kk],32, VLEN), VLEN);
      _vel_vstu_vssl(vrsumL_c1, 4, pGKernel + FILTER_OFFSET(k+2*kk+remain+1,c+1,0,0), 1) ;
      __vr vrsumU_c2 = _vel_vfsums_vvl(vrsum_c2[kk], VLEN) ;
      _vel_vstu_vssl(vrsumU_c2, 4, pGKernel + FILTER_OFFSET(k+2*kk+remain,  c+2,0,0), 1) ;
      __vr vrsumL_c2 = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsum_c2[kk],32, VLEN), VLEN);
      _vel_vstu_vssl(vrsumL_c2, 4, pGKernel + FILTER_OFFSET(k+2*kk+remain+1,c+2,0,0), 1) ;
      __vr vrsumU_c3 = _vel_vfsums_vvl(vrsum_c3[kk], VLEN) ;
      _vel_vstu_vssl(vrsumU_c3, 4, pGKernel + FILTER_OFFSET(k+2*kk+remain,  c+3,0,0), 1) ;
      __vr vrsumL_c3 = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsum_c3[kk],32, VLEN), VLEN);
      _vel_vstu_vssl(vrsumL_c3, 4, pGKernel + FILTER_OFFSET(k+2*kk+remain+1,c+3,0,0), 1) ;

    }

#undef FILTER_OFFSET

    c+=4 ;
  }
  for ( ; c<inChannelGroup; c+=8) {

    __vr vrsum0_c0  = _vel_vbrds_vsl(0.f, VLEN) ;
    __vr vrsum0_c1  = _vel_vbrds_vsl(0.f, VLEN) ;
    __vr vrsum0_c2  = _vel_vbrds_vsl(0.f, VLEN) ;
    __vr vrsum0_c3  = _vel_vbrds_vsl(0.f, VLEN) ;
    __vr vrsum0_c4  = _vel_vbrds_vsl(0.f, VLEN) ;
    __vr vrsum0_c5  = _vel_vbrds_vsl(0.f, VLEN) ;
    __vr vrsum0_c6  = _vel_vbrds_vsl(0.f, VLEN) ;
    __vr vrsum0_c7  = _vel_vbrds_vsl(0.f, VLEN) ;

    __vr vrsum_c0[nPacked] ;
    __vr vrsum_c1[nPacked] ;
    __vr vrsum_c2[nPacked] ;
    __vr vrsum_c3[nPacked] ;
    __vr vrsum_c4[nPacked] ;
    __vr vrsum_c5[nPacked] ;
    __vr vrsum_c6[nPacked] ;
    __vr vrsum_c7[nPacked] ;
#pragma clang loop unroll(full)
    for(int64_t kk=0; kk<nPacked; kk++) {
      vrsum_c0[kk] = _vel_pvbrd_vsl(0UL, VLEN) ;
      vrsum_c1[kk] = _vel_pvbrd_vsl(0UL, VLEN) ;
      vrsum_c2[kk] = _vel_pvbrd_vsl(0UL, VLEN) ;
      vrsum_c3[kk] = _vel_pvbrd_vsl(0UL, VLEN) ;
      vrsum_c4[kk] = _vel_pvbrd_vsl(0UL, VLEN) ;
      vrsum_c5[kk] = _vel_pvbrd_vsl(0UL, VLEN) ;
      vrsum_c6[kk] = _vel_pvbrd_vsl(0UL, VLEN) ;
      vrsum_c7[kk] = _vel_pvbrd_vsl(0UL, VLEN) ;
    }

    for (int64_t n=0; n<batch; n++) {
      for (int64_t xy = 0; xy < gOutHeight * gOutWidth ; xy += VLEN ) {

	const int64_t vl = gOutHeight * gOutWidth - xy < VLEN ? gOutHeight * gOutWidth - xy  : VLEN ;

	const float *pInChannel = pIn + inGroupOffset + n * inChannel * inHeight * inWidth  ;

	const int64_t outIndex = gOutGroupOffset + ((n * gOutChannel + k) * gOutHeight ) * gOutWidth + xy ;

	__vr vrin_c0  = _vel_vldu_vssl(4,&pInChannel[(c+0)*inHeight*inWidth+xy], vl) ;
	__vr vrin_c1  = _vel_vldu_vssl(4,&pInChannel[(c+1)*inHeight*inWidth+xy], vl) ;
	__vr vrin_c2  = _vel_vldu_vssl(4,&pInChannel[(c+2)*inHeight*inWidth+xy], vl) ;
	__vr vrin_c3  = _vel_vldu_vssl(4,&pInChannel[(c+3)*inHeight*inWidth+xy], vl) ;
	__vr vrin_c4  = _vel_vldu_vssl(4,&pInChannel[(c+4)*inHeight*inWidth+xy], vl) ;
	__vr vrin_c5  = _vel_vldu_vssl(4,&pInChannel[(c+5)*inHeight*inWidth+xy], vl) ;
	__vr vrin_c6  = _vel_vldu_vssl(4,&pInChannel[(c+6)*inHeight*inWidth+xy], vl) ;
	__vr vrin_c7  = _vel_vldu_vssl(4,&pInChannel[(c+7)*inHeight*inWidth+xy], vl) ;

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

	VFADD(vrin_c0, vrsum0_c0, vrsum_c0) ;
	VFADD(vrin_c1, vrsum0_c1, vrsum_c1) ;
	VFADD(vrin_c2, vrsum0_c2, vrsum_c2) ;
	VFADD(vrin_c3, vrsum0_c3, vrsum_c3) ;
	VFADD(vrin_c4, vrsum0_c4, vrsum_c4) ;
	VFADD(vrin_c5, vrsum0_c5, vrsum_c5) ;
	VFADD(vrin_c6, vrsum0_c6, vrsum_c6) ;
	VFADD(vrin_c7, vrsum0_c7, vrsum_c7) ;

#undef VFADD
      } // gOutHeight * gOutWidth
    } // batch

#define FILTER_OFFSET(k,c,r,s) ( gKernGroupOffset + filter_index<FLAYOUT>(k,c,r,s, inChannelGroup, gOutChannelGroup, 1, 1) )


    if( remain ) {
      vrsum0_c0 = _vel_vfsums_vvl(vrsum0_c0, VLEN) ;
      _vel_vstu_vssl(vrsum0_c0, 4, pGKernel+FILTER_OFFSET(k+0,c+0,0,0), 1) ;
      vrsum0_c1 = _vel_vfsums_vvl(vrsum0_c1, VLEN) ;
      _vel_vstu_vssl(vrsum0_c1, 4, pGKernel+FILTER_OFFSET(k+0,c+1,0,0), 1) ;
      vrsum0_c2 = _vel_vfsums_vvl(vrsum0_c2, VLEN) ;
      _vel_vstu_vssl(vrsum0_c2, 4, pGKernel+FILTER_OFFSET(k+0,c+2,0,0), 1) ;
      vrsum0_c3 = _vel_vfsums_vvl(vrsum0_c3, VLEN) ;
      _vel_vstu_vssl(vrsum0_c3, 4, pGKernel+FILTER_OFFSET(k+0,c+3,0,0), 1) ;
      vrsum0_c4 = _vel_vfsums_vvl(vrsum0_c4, VLEN) ;
      _vel_vstu_vssl(vrsum0_c4, 4, pGKernel+FILTER_OFFSET(k+0,c+4,0,0), 1) ;
      vrsum0_c5 = _vel_vfsums_vvl(vrsum0_c5, VLEN) ;
      _vel_vstu_vssl(vrsum0_c5, 4, pGKernel+FILTER_OFFSET(k+0,c+5,0,0), 1) ;
      vrsum0_c6 = _vel_vfsums_vvl(vrsum0_c6, VLEN) ;
      _vel_vstu_vssl(vrsum0_c6, 4, pGKernel+FILTER_OFFSET(k+0,c+6,0,0), 1) ;
      vrsum0_c7 = _vel_vfsums_vvl(vrsum0_c7, VLEN) ;
      _vel_vstu_vssl(vrsum0_c7, 4, pGKernel+FILTER_OFFSET(k+0,c+7,0,0), 1) ;

    }

#pragma clang loop unroll(full)
    for(int64_t kk=0; kk<nPacked; kk++) {
      __vr vrsumU_c0 = _vel_vfsums_vvl(vrsum_c0[kk], VLEN) ;
      _vel_vstu_vssl(vrsumU_c0, 4, pGKernel + FILTER_OFFSET(k+2*kk+remain,  c+0,0,0), 1) ;
      __vr vrsumL_c0 = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsum_c0[kk],32, VLEN), VLEN);
      _vel_vstu_vssl(vrsumL_c0, 4, pGKernel + FILTER_OFFSET(k+2*kk+remain+1,c+0,0,0), 1) ;
      __vr vrsumU_c1 = _vel_vfsums_vvl(vrsum_c1[kk], VLEN) ;
      _vel_vstu_vssl(vrsumU_c1, 4, pGKernel + FILTER_OFFSET(k+2*kk+remain,  c+1,0,0), 1) ;
      __vr vrsumL_c1 = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsum_c1[kk],32, VLEN), VLEN);
      _vel_vstu_vssl(vrsumL_c1, 4, pGKernel + FILTER_OFFSET(k+2*kk+remain+1,c+1,0,0), 1) ;
      __vr vrsumU_c2 = _vel_vfsums_vvl(vrsum_c2[kk], VLEN) ;
      _vel_vstu_vssl(vrsumU_c2, 4, pGKernel + FILTER_OFFSET(k+2*kk+remain,  c+2,0,0), 1) ;
      __vr vrsumL_c2 = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsum_c2[kk],32, VLEN), VLEN);
      _vel_vstu_vssl(vrsumL_c2, 4, pGKernel + FILTER_OFFSET(k+2*kk+remain+1,c+2,0,0), 1) ;
      __vr vrsumU_c3 = _vel_vfsums_vvl(vrsum_c3[kk], VLEN) ;
      _vel_vstu_vssl(vrsumU_c3, 4, pGKernel + FILTER_OFFSET(k+2*kk+remain,  c+3,0,0), 1) ;
      __vr vrsumL_c3 = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsum_c3[kk],32, VLEN), VLEN);
      _vel_vstu_vssl(vrsumL_c3, 4, pGKernel + FILTER_OFFSET(k+2*kk+remain+1,c+3,0,0), 1) ;
      __vr vrsumU_c4 = _vel_vfsums_vvl(vrsum_c4[kk], VLEN) ;
      _vel_vstu_vssl(vrsumU_c4, 4, pGKernel + FILTER_OFFSET(k+2*kk+remain,  c+4,0,0), 1) ;
      __vr vrsumL_c4 = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsum_c4[kk],32, VLEN), VLEN);
      _vel_vstu_vssl(vrsumL_c4, 4, pGKernel + FILTER_OFFSET(k+2*kk+remain+1,c+4,0,0), 1) ;
      __vr vrsumU_c5 = _vel_vfsums_vvl(vrsum_c5[kk], VLEN) ;
      _vel_vstu_vssl(vrsumU_c5, 4, pGKernel + FILTER_OFFSET(k+2*kk+remain,  c+5,0,0), 1) ;
      __vr vrsumL_c5 = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsum_c5[kk],32, VLEN), VLEN);
      _vel_vstu_vssl(vrsumL_c5, 4, pGKernel + FILTER_OFFSET(k+2*kk+remain+1,c+5,0,0), 1) ;
      __vr vrsumU_c6 = _vel_vfsums_vvl(vrsum_c6[kk], VLEN) ;
      _vel_vstu_vssl(vrsumU_c6, 4, pGKernel + FILTER_OFFSET(k+2*kk+remain,  c+6,0,0), 1) ;
      __vr vrsumL_c6 = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsum_c6[kk],32, VLEN), VLEN);
      _vel_vstu_vssl(vrsumL_c6, 4, pGKernel + FILTER_OFFSET(k+2*kk+remain+1,c+6,0,0), 1) ;
      __vr vrsumU_c7 = _vel_vfsums_vvl(vrsum_c7[kk], VLEN) ;
      _vel_vstu_vssl(vrsumU_c7, 4, pGKernel + FILTER_OFFSET(k+2*kk+remain,  c+7,0,0), 1) ;
      __vr vrsumL_c7 = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsum_c7[kk],32, VLEN), VLEN);
      _vel_vstu_vssl(vrsumL_c7, 4, pGKernel + FILTER_OFFSET(k+2*kk+remain+1,c+7,0,0), 1) ;

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
    const int64_t strideWidth,		// 1
    const int64_t strideHeight,		// 1
    const int64_t padWidth,		// 0
    const int64_t padHeight,		// 0
    const int64_t dilationWidth,	// 1
    const int64_t dilationHeight	// 1
)
{
  for (int64_t g = beginGroup; g < nGroup; g++) {
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
	 inChannelGroup, gOutChannelGroup,
	 inGroupOffset, gOutGroupOffset, gKernGroupOffset,
	 batch, beginOChannel + k ) ;
      k+=1 ;
      break ;
    case 2:
      func<FLAYOUT, 2>(pIn, pGOut, pGKernel,
	 gOutChannel, gOutWidth, gOutHeight,
	 inChannel, inWidth, inHeight,
	 inChannelGroup, gOutChannelGroup,
	 inGroupOffset, gOutGroupOffset, gKernGroupOffset,
	 batch, beginOChannel + k ) ;
      k+=2 ;
      break ;
    case 3:
      func<FLAYOUT, 3>(pIn, pGOut, pGKernel,
	 gOutChannel, gOutWidth, gOutHeight,
	 inChannel, inWidth, inHeight,
	 inChannelGroup, gOutChannelGroup,
	 inGroupOffset, gOutGroupOffset, gKernGroupOffset,
	 batch, beginOChannel + k ) ;
      k+=3 ;
      break ;
    case 4:
      func<FLAYOUT, 4>(pIn, pGOut, pGKernel,
	 gOutChannel, gOutWidth, gOutHeight,
	 inChannel, inWidth, inHeight,
	 inChannelGroup, gOutChannelGroup,
	 inGroupOffset, gOutGroupOffset, gKernGroupOffset,
	 batch, beginOChannel + k ) ;
      k+=4 ;
      break ;
    case 5:
      func<FLAYOUT, 5>(pIn, pGOut, pGKernel,
	 gOutChannel, gOutWidth, gOutHeight,
	 inChannel, inWidth, inHeight,
	 inChannelGroup, gOutChannelGroup,
	 inGroupOffset, gOutGroupOffset, gKernGroupOffset,
	 batch, beginOChannel + k ) ;
      k+=5 ;
      break ;
    case 6:
      func<FLAYOUT, 6>(pIn, pGOut, pGKernel,
	 gOutChannel, gOutWidth, gOutHeight,
	 inChannel, inWidth, inHeight,
	 inChannelGroup, gOutChannelGroup,
	 inGroupOffset, gOutGroupOffset, gKernGroupOffset,
	 batch, beginOChannel + k ) ;
      k+=6 ;
      break ;
    case 7:
      func<FLAYOUT, 7>(pIn, pGOut, pGKernel,
	 gOutChannel, gOutWidth, gOutHeight,
	 inChannel, inWidth, inHeight,
	 inChannelGroup, gOutChannelGroup,
	 inGroupOffset, gOutGroupOffset, gKernGroupOffset,
	 batch, beginOChannel + k ) ;
      k+=7 ;
      break ;
    default :
      break ;
    }
    for (; k<nOChannel; ) {
      func<FLAYOUT, 8>(pIn, pGOut, pGKernel,
	 gOutChannel, gOutWidth, gOutHeight,
	 inChannel, inWidth, inHeight,
	 inChannelGroup, gOutChannelGroup,
	 inGroupOffset, gOutGroupOffset, gKernGroupOffset,
	 batch, beginOChannel + k ) ;
      k+=8 ;
    } // gOutChannel
  } // group
}


extern "C" vednnError_t
vednnConvolutionBackwardFilter_direct_dil1_str1_padsame_ker1(
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
