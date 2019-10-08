#include <stdint.h>
#include <stdlib.h>

#include "vednn.h"

#include "velintrin.h"
#define VLEN	(256)

template<filterLayout_t FLAYOUT, int NUMCHANNEL>
static inline void func_odd(
    const float * __restrict__ pGOut,
    const float * __restrict__ pKernel,
    float * __restrict__ const pGIn,
    const int64_t gOutChannel,
    const int64_t gOutWidth,
    const int64_t gOutHeight,
    const int64_t gInChannel,
    const int64_t gInWidth,
    const int64_t gInHeight,
    const int64_t strideWidth,
    const int64_t strideHeight,
    const int64_t gInChannelGroup,
    const int64_t gOutChannelGroup,
    const int64_t gInGroupOffset,
    const int64_t gOutGroupOffset,
    const int64_t kernGroupOffset,
    const int64_t n,
    const int64_t c,
    const int64_t nY,
    const __vr    vrij
)
{
  const float *pInChannel = pGIn + gInGroupOffset + ((n * gInChannel + c) * gInHeight * gInWidth ) ;
  {
    // zero filling
    const int64_t nPixs = NUMCHANNEL * gInHeight * gInWidth ;
    __vr vrzero = _vel_vbrds_vsl(0.f, VLEN) ;
    for(int64_t ij=0; ij<nPixs ; ij+=VLEN) {
      const int64_t vl = nPixs - ij < VLEN ? nPixs - ij : VLEN ;
      _vel_vstu_vssl(vrzero, 4, (void*)(pInChannel+ij), vl) ;
    }
  }

  for (int64_t y=0; y<gOutHeight; y+=nY)
  {
    const int64_t vl = gOutWidth * (gOutHeight - y < nY ? gOutHeight - y : nY) ;
    const int64_t op = y * gOutWidth ;

    __vr vrsum0  = _vel_vbrds_vsl(0.f, vl) ;
    __vr vrsum12 = _vel_pvbrd_vsl(0UL, vl) ;
    __vr vrsum34 = _vel_pvbrd_vsl(0UL, vl) ;
    __vr vrsum56 = _vel_pvbrd_vsl(0UL, vl) ;
    __vr vrsum78 = _vel_pvbrd_vsl(0UL, vl) ;
    __vr vrsum9A = _vel_pvbrd_vsl(0UL, vl) ;
    __vr vrsumBC = _vel_pvbrd_vsl(0UL, vl) ;
    __vr vrsumDE = _vel_pvbrd_vsl(0UL, vl) ;

    for(int64_t k=0; k<gOutChannelGroup; k++) {
      int64_t outIndex = gOutGroupOffset + (n * gOutChannel + k  ) *  gOutHeight * gOutWidth + op ;

      __vr vrgout = _vel_vldu_vssl(4, pGOut+outIndex, vl) ;

#define FILTER_OFFSET(k,c,r,s) ( kernGroupOffset + filter_index<FLAYOUT>(k,c,r,s, gInChannelGroup, gOutChannelGroup, 1, 1) )

#define VFADD(VRGOUT,K,R,S) {									\
	const float    kerValue0  = pKernel[FILTER_OFFSET(K,c+ 0,R,S)] ;			\
	const uint64_t kerValue12 = _vel_pack_f32p(pKernel + FILTER_OFFSET(K,c+ 1,R,S),		\
						   pKernel + FILTER_OFFSET(K,c+ 2,R,S)) ;	\
	const uint64_t kerValue34 = _vel_pack_f32p(pKernel + FILTER_OFFSET(K,c+ 3,R,S),		\
						   pKernel + FILTER_OFFSET(K,c+ 4,R,S)) ;	\
	const uint64_t kerValue56 = _vel_pack_f32p(pKernel + FILTER_OFFSET(K,c+ 5,R,S),		\
						   pKernel + FILTER_OFFSET(K,c+ 6,R,S)) ;	\
	const uint64_t kerValue78 = _vel_pack_f32p(pKernel + FILTER_OFFSET(K,c+ 7,R,S),		\
						   pKernel + FILTER_OFFSET(K,c+ 8,R,S)) ;	\
	const uint64_t kerValue9A = _vel_pack_f32p(pKernel + FILTER_OFFSET(K,c+ 9,R,S),		\
						   pKernel + FILTER_OFFSET(K,c+10,R,S)) ;	\
	const uint64_t kerValueBC = _vel_pack_f32p(pKernel + FILTER_OFFSET(K,c+11,R,S),		\
						   pKernel + FILTER_OFFSET(K,c+12,R,S)) ;	\
	const uint64_t kerValueDE = _vel_pack_f32p(pKernel + FILTER_OFFSET(K,c+13,R,S),		\
						   pKernel + FILTER_OFFSET(K,c+14,R,S)) ;	\
	__vr vrgoutP = _vel_vshf_vvvsl(VRGOUT, VRGOUT, VE_VSHUFFLE_YUZU, vl) ;			\
	vrsum0 = _vel_vfmads_vvsvl(vrsum0, kerValue0, VRGOUT, vl) ;				\
	if(NUMCHANNEL>= 3) vrsum12 = _vel_pvfmad_vvsvl(vrsum12, kerValue12, vrgoutP, vl) ;	\
	if(NUMCHANNEL>= 5) vrsum34 = _vel_pvfmad_vvsvl(vrsum34, kerValue34, vrgoutP, vl) ;	\
	if(NUMCHANNEL>= 7) vrsum56 = _vel_pvfmad_vvsvl(vrsum56, kerValue56, vrgoutP, vl) ;	\
	if(NUMCHANNEL>= 9) vrsum78 = _vel_pvfmad_vvsvl(vrsum78, kerValue78, vrgoutP, vl) ;	\
	if(NUMCHANNEL>=11) vrsum9A = _vel_pvfmad_vvsvl(vrsum9A, kerValue9A, vrgoutP, vl) ;	\
	if(NUMCHANNEL>=13) vrsumBC = _vel_pvfmad_vvsvl(vrsumBC, kerValueBC, vrgoutP, vl) ;	\
	if(NUMCHANNEL>=15) vrsumDE = _vel_pvfmad_vvsvl(vrsumDE, kerValueDE, vrgoutP, vl) ;	\
      }

      VFADD(vrgout, k, 0, 0) ;
#undef VFADD
#undef FILTER_OFFSET
    }

    __vr vrpgin_c0 = _vel_vsfa_vvssl(vrij, 2, (uint64_t)(pInChannel+y*strideHeight*gInWidth), vl) ;
    __vr vrpgin_c1 = _vel_vaddul_vsvl(  4*gInHeight*gInWidth,vrpgin_c0, vl) ;
    __vr vrpgin_c2 = _vel_vaddul_vsvl(2*4*gInHeight*gInWidth,vrpgin_c0, vl) ;
    __vr vrpgin_c3 = _vel_vaddul_vsvl(3*4*gInHeight*gInWidth,vrpgin_c0, vl) ;
    __vr vrpgin_c4 = _vel_vaddul_vsvl(4*4*gInHeight*gInWidth,vrpgin_c0, vl) ;
    __vr vrpgin_c5 = _vel_vaddul_vsvl(5*4*gInHeight*gInWidth,vrpgin_c0, vl) ;
    __vr vrpgin_c6 = _vel_vaddul_vsvl(6*4*gInHeight*gInWidth,vrpgin_c0, vl) ;
    __vr vrpgin_c7 = _vel_vaddul_vsvl(7*4*gInHeight*gInWidth,vrpgin_c0, vl) ;
    __vr vrpgin_c8 = _vel_vaddul_vsvl(8*4*gInHeight*gInWidth,vrpgin_c0, vl) ;
    __vr vrpgin_c9 = _vel_vaddul_vsvl(9*4*gInHeight*gInWidth,vrpgin_c0, vl) ;
    __vr vrpgin_cA = _vel_vaddul_vsvl(10*4*gInHeight*gInWidth,vrpgin_c0, vl) ;
    __vr vrpgin_cB = _vel_vaddul_vsvl(11*4*gInHeight*gInWidth,vrpgin_c0, vl) ;
    __vr vrpgin_cC = _vel_vaddul_vsvl(12*4*gInHeight*gInWidth,vrpgin_c0, vl) ;
    __vr vrpgin_cD = _vel_vaddul_vsvl(13*4*gInHeight*gInWidth,vrpgin_c0, vl) ;
    __vr vrpgin_cE = _vel_vaddul_vsvl(14*4*gInHeight*gInWidth,vrpgin_c0, vl) ;

    _vel_svob() ;

    _vel_vscuot_vvssl(vrsum0, vrpgin_c0, 0, 0, vl) ;

    if(NUMCHANNEL>= 3) {
      _vel_vscuot_vvssl(vrsum12, vrpgin_c1, 0, 0, vl) ;
      _vel_vsclot_vvssl(vrsum12, vrpgin_c2, 0, 0, vl) ;
    }
    if(NUMCHANNEL>= 5) {
      _vel_vscuot_vvssl(vrsum34, vrpgin_c3, 0, 0, vl) ;
      _vel_vsclot_vvssl(vrsum34, vrpgin_c4, 0, 0, vl) ;
    }
    if(NUMCHANNEL>= 6) {
      _vel_vscuot_vvssl(vrsum56, vrpgin_c5, 0, 0, vl) ;
      _vel_vsclot_vvssl(vrsum56, vrpgin_c6, 0, 0, vl) ;
    }
    if(NUMCHANNEL>= 9) {
      _vel_vscuot_vvssl(vrsum78, vrpgin_c7, 0, 0, vl) ;
      _vel_vsclot_vvssl(vrsum78, vrpgin_c8, 0, 0, vl) ;
    }
    if(NUMCHANNEL>=11) {
      _vel_vscuot_vvssl(vrsum9A, vrpgin_c9, 0, 0, vl) ;
      _vel_vsclot_vvssl(vrsum9A, vrpgin_cA, 0, 0, vl) ;
    }
    if(NUMCHANNEL>=13) {
      _vel_vscuot_vvssl(vrsumBC, vrpgin_cB, 0, 0, vl) ;
      _vel_vsclot_vvssl(vrsumBC, vrpgin_cC, 0, 0, vl) ;
    }
    if(NUMCHANNEL>=15) {
      _vel_vscuot_vvssl(vrsumDE, vrpgin_cD, 0, 0, vl) ;
      _vel_vsclot_vvssl(vrsumDE, vrpgin_cE, 0, 0, vl) ;
    }

  }

  _vel_svob() ;
}


template<filterLayout_t FLAYOUT, int NUMCHANNEL>
static inline void func_even(
    const float * __restrict__ pGOut,
    const float * __restrict__ pKernel,
    float * __restrict__ const pGIn,
    const int64_t gOutChannel,
    const int64_t gOutWidth,
    const int64_t gOutHeight,
    const int64_t gInChannel,
    const int64_t gInWidth,
    const int64_t gInHeight,
    const int64_t strideWidth,
    const int64_t strideHeight,
    const int64_t gInChannelGroup,
    const int64_t gOutChannelGroup,
    const int64_t gInGroupOffset,
    const int64_t gOutGroupOffset,
    const int64_t kernGroupOffset,
    const int64_t n,
    const int64_t c,
    const int64_t nY,
    const __vr    vrij
)
{
  const float *pInChannel = pGIn + gInGroupOffset + ((n * gInChannel + c) * gInHeight * gInWidth ) ;
  {
    // zero filling
    const int64_t nPixs = NUMCHANNEL * gInHeight * gInWidth ;
    __vr vrzero = _vel_vbrds_vsl(0.f, VLEN) ;
    for(int64_t ij=0; ij<nPixs ; ij+=VLEN) {
      const int64_t vl = nPixs - ij < VLEN ? nPixs - ij : VLEN ;
      _vel_vstu_vssl(vrzero, 4, (void*)(pInChannel+ij), vl) ;
    }
  }

  for (int64_t y=0; y<gOutHeight; y+=nY)
  {
    const int64_t vl = gOutWidth * (gOutHeight - y < nY ? gOutHeight - y : nY) ;
    const int64_t op = y * gOutWidth ;

    __vr vrsum01 = _vel_pvbrd_vsl(0UL, vl) ;
    __vr vrsum23 = _vel_pvbrd_vsl(0UL, vl) ;
    __vr vrsum45 = _vel_pvbrd_vsl(0UL, vl) ;
    __vr vrsum67 = _vel_pvbrd_vsl(0UL, vl) ;
    __vr vrsum89 = _vel_pvbrd_vsl(0UL, vl) ;
    __vr vrsumAB = _vel_pvbrd_vsl(0UL, vl) ;
    __vr vrsumCD = _vel_pvbrd_vsl(0UL, vl) ;
    __vr vrsumEF = _vel_pvbrd_vsl(0UL, vl) ;


    for(int64_t k=0; k<gOutChannelGroup; k++) {
      int64_t outIndex = gOutGroupOffset + (n * gOutChannel + k  ) *  gOutHeight * gOutWidth + op ;

      __vr vrgout = _vel_vldu_vssl(4, pGOut+outIndex, vl) ;

#define FILTER_OFFSET(k,c,r,s) ( kernGroupOffset + filter_index<FLAYOUT>(k,c,r,s, gInChannelGroup, gOutChannelGroup, 1, 1) )

#define VFADD(VRGOUT,K,R,S) {									\
	const uint64_t kerValue01 = _vel_pack_f32p(pKernel + FILTER_OFFSET(K,c+ 0,R,S),		\
						   pKernel + FILTER_OFFSET(K,c+ 1,R,S)) ;	\
	const uint64_t kerValue23 = _vel_pack_f32p(pKernel + FILTER_OFFSET(K,c+ 2,R,S),		\
						   pKernel + FILTER_OFFSET(K,c+ 3,R,S)) ;	\
	const uint64_t kerValue45 = _vel_pack_f32p(pKernel + FILTER_OFFSET(K,c+ 4,R,S),		\
						   pKernel + FILTER_OFFSET(K,c+ 5,R,S)) ;	\
	const uint64_t kerValue67 = _vel_pack_f32p(pKernel + FILTER_OFFSET(K,c+ 6,R,S),		\
						   pKernel + FILTER_OFFSET(K,c+ 7,R,S)) ;	\
	const uint64_t kerValue89 = _vel_pack_f32p(pKernel + FILTER_OFFSET(K,c+ 8,R,S),		\
						   pKernel + FILTER_OFFSET(K,c+ 9,R,S)) ;	\
	const uint64_t kerValueAB = _vel_pack_f32p(pKernel + FILTER_OFFSET(K,c+10,R,S),		\
						   pKernel + FILTER_OFFSET(K,c+11,R,S)) ;	\
	const uint64_t kerValueCD = _vel_pack_f32p(pKernel + FILTER_OFFSET(K,c+12,R,S),		\
						   pKernel + FILTER_OFFSET(K,c+13,R,S)) ;	\
	const uint64_t kerValueEF = _vel_pack_f32p(pKernel + FILTER_OFFSET(K,c+14,R,S),		\
						   pKernel + FILTER_OFFSET(K,c+15,R,S)) ;	\
	__vr vrgoutP = _vel_vshf_vvvsl(VRGOUT, VRGOUT, VE_VSHUFFLE_YUZU, vl) ;			\
	if(NUMCHANNEL>= 2) vrsum01 = _vel_pvfmad_vvsvl(vrsum01, kerValue01, vrgoutP, vl) ;	\
	if(NUMCHANNEL>= 4) vrsum23 = _vel_pvfmad_vvsvl(vrsum23, kerValue23, vrgoutP, vl) ;	\
	if(NUMCHANNEL>= 6) vrsum45 = _vel_pvfmad_vvsvl(vrsum45, kerValue45, vrgoutP, vl) ;	\
	if(NUMCHANNEL>= 8) vrsum67 = _vel_pvfmad_vvsvl(vrsum67, kerValue67, vrgoutP, vl) ;	\
	if(NUMCHANNEL>=10) vrsum89 = _vel_pvfmad_vvsvl(vrsum89, kerValue89, vrgoutP, vl) ;	\
	if(NUMCHANNEL>=12) vrsumAB = _vel_pvfmad_vvsvl(vrsumAB, kerValueAB, vrgoutP, vl) ;	\
	if(NUMCHANNEL>=14) vrsumCD = _vel_pvfmad_vvsvl(vrsumCD, kerValueCD, vrgoutP, vl) ;	\
	if(NUMCHANNEL>=16) vrsumEF = _vel_pvfmad_vvsvl(vrsumEF, kerValueEF, vrgoutP, vl) ;	\
      }

      VFADD(vrgout, k, 0, 0) ;
#undef VFADD
#undef FILTER_OFFSET
    }

    __vr vrpgin_c0 = _vel_vsfa_vvssl(vrij, 2, (uint64_t)(pInChannel+y*strideHeight*gInWidth), vl) ;
    __vr vrpgin_c1 = _vel_vaddul_vsvl(  4*gInHeight*gInWidth,vrpgin_c0, vl) ;
    __vr vrpgin_c2 = _vel_vaddul_vsvl(2*4*gInHeight*gInWidth,vrpgin_c0, vl) ;
    __vr vrpgin_c3 = _vel_vaddul_vsvl(3*4*gInHeight*gInWidth,vrpgin_c0, vl) ;
    __vr vrpgin_c4 = _vel_vaddul_vsvl(4*4*gInHeight*gInWidth,vrpgin_c0, vl) ;
    __vr vrpgin_c5 = _vel_vaddul_vsvl(5*4*gInHeight*gInWidth,vrpgin_c0, vl) ;
    __vr vrpgin_c6 = _vel_vaddul_vsvl(6*4*gInHeight*gInWidth,vrpgin_c0, vl) ;
    __vr vrpgin_c7 = _vel_vaddul_vsvl(7*4*gInHeight*gInWidth,vrpgin_c0, vl) ;
    __vr vrpgin_c8 = _vel_vaddul_vsvl(8*4*gInHeight*gInWidth,vrpgin_c0, vl) ;
    __vr vrpgin_c9 = _vel_vaddul_vsvl(9*4*gInHeight*gInWidth,vrpgin_c0, vl) ;
    __vr vrpgin_cA = _vel_vaddul_vsvl(10*4*gInHeight*gInWidth,vrpgin_c0, vl) ;
    __vr vrpgin_cB = _vel_vaddul_vsvl(11*4*gInHeight*gInWidth,vrpgin_c0, vl) ;
    __vr vrpgin_cC = _vel_vaddul_vsvl(12*4*gInHeight*gInWidth,vrpgin_c0, vl) ;
    __vr vrpgin_cD = _vel_vaddul_vsvl(13*4*gInHeight*gInWidth,vrpgin_c0, vl) ;
    __vr vrpgin_cE = _vel_vaddul_vsvl(14*4*gInHeight*gInWidth,vrpgin_c0, vl) ;
    __vr vrpgin_cF = _vel_vaddul_vsvl(15*4*gInHeight*gInWidth,vrpgin_c0, vl) ;

    _vel_svob() ;

    if(NUMCHANNEL>= 2) {
      _vel_vscuot_vvssl(vrsum01, vrpgin_c0, 0, 0, vl) ;
      _vel_vsclot_vvssl(vrsum01, vrpgin_c1, 0, 0, vl) ;
    }
    if(NUMCHANNEL>= 4) {
      _vel_vscuot_vvssl(vrsum23, vrpgin_c2, 0, 0, vl) ;
      _vel_vsclot_vvssl(vrsum23, vrpgin_c3, 0, 0, vl) ;
    }
    if(NUMCHANNEL>= 6) {
      _vel_vscuot_vvssl(vrsum45, vrpgin_c4, 0, 0, vl) ;
      _vel_vsclot_vvssl(vrsum45, vrpgin_c5, 0, 0, vl) ;
    }
    if(NUMCHANNEL>= 8) {
      _vel_vscuot_vvssl(vrsum67, vrpgin_c6, 0, 0, vl) ;
      _vel_vsclot_vvssl(vrsum67, vrpgin_c7, 0, 0, vl) ;
    }
    if(NUMCHANNEL>=10) {
      _vel_vscuot_vvssl(vrsum89, vrpgin_c8, 0, 0, vl) ;
      _vel_vsclot_vvssl(vrsum89, vrpgin_c9, 0, 0, vl) ;
    }
    if(NUMCHANNEL>=12) {
      _vel_vscuot_vvssl(vrsumAB, vrpgin_cA, 0, 0, vl) ;
      _vel_vsclot_vvssl(vrsumAB, vrpgin_cB, 0, 0, vl) ;
    }
    if(NUMCHANNEL>=14) {
      _vel_vscuot_vvssl(vrsumCD, vrpgin_cC, 0, 0, vl) ;
      _vel_vsclot_vvssl(vrsumCD, vrpgin_cD, 0, 0, vl) ;
    }
    if(NUMCHANNEL>=16) {
      _vel_vscuot_vvssl(vrsumEF, vrpgin_cE, 0, 0, vl) ;
      _vel_vsclot_vvssl(vrsumEF, vrpgin_cF, 0, 0, vl) ;
    }
  }

  _vel_svob() ;
}

template<int NUMCHANNEL>
static inline void func_even_filternchw_packedkernel(
    const float * __restrict__ pGOut,
    const float * __restrict__ pKernel,
    float * __restrict__ const pGIn,
    const int64_t gOutChannel,
    const int64_t gOutWidth,
    const int64_t gOutHeight,
    const int64_t gInChannel,
    const int64_t gInWidth,
    const int64_t gInHeight,
    const int64_t strideWidth,
    const int64_t strideHeight,
    const int64_t gInChannelGroup,
    const int64_t gOutChannelGroup,
    const int64_t gInGroupOffset,
    const int64_t gOutGroupOffset,
    const int64_t kernGroupOffset,
    const int64_t n,
    const int64_t c,
    const int64_t nY,
    const __vr    vrij
)
{
  const float *pInChannel = pGIn + gInGroupOffset + ((n * gInChannel + c) * gInHeight * gInWidth ) ;
  {
    // zero filling
    const int64_t nPixs = NUMCHANNEL * gInHeight * gInWidth ;
    __vr vrzero = _vel_vbrds_vsl(0.f, VLEN) ;
    for(int64_t ij=0; ij<nPixs ; ij+=VLEN) {
      const int64_t vl = nPixs - ij < VLEN ? nPixs - ij : VLEN ;
      _vel_vstu_vssl(vrzero, 4, (void*)(pInChannel+ij), vl) ;
    }
  }

  for (int64_t y=0; y<gOutHeight; y+=nY)
  {
    const int64_t vl = gOutWidth * (gOutHeight - y < nY ? gOutHeight - y : nY) ;
    const int64_t op = y * gOutWidth ;

    __vr vrsum01 = _vel_pvbrd_vsl(0UL, vl) ;
    __vr vrsum23 = _vel_pvbrd_vsl(0UL, vl) ;
    __vr vrsum45 = _vel_pvbrd_vsl(0UL, vl) ;
    __vr vrsum67 = _vel_pvbrd_vsl(0UL, vl) ;
    __vr vrsum89 = _vel_pvbrd_vsl(0UL, vl) ;
    __vr vrsumAB = _vel_pvbrd_vsl(0UL, vl) ;
    __vr vrsumCD = _vel_pvbrd_vsl(0UL, vl) ;
    __vr vrsumEF = _vel_pvbrd_vsl(0UL, vl) ;


    for(int64_t k=0; k<gOutChannelGroup; k++) {
      int64_t outIndex = gOutGroupOffset + (n * gOutChannel + k  ) *  gOutHeight * gOutWidth + op ;

      __vr vrgout = _vel_vldu_vssl(4, pGOut+outIndex, vl) ;

      const float *pKerValue = pKernel + kernGroupOffset + ((k  ) * gInChannelGroup + c) ;
      const uint64_t *pKerValue_u64 = (const uint64_t*) pKerValue ;
#define VFADD(VRGOUT,K,R,S) {									\
	__vr vrgoutP = _vel_vshf_vvvsl(VRGOUT, VRGOUT, VE_VSHUFFLE_YUZU, vl) ;			\
	if(NUMCHANNEL>= 2) vrsum01 = _vel_pvfmad_vvsvl(vrsum01, pKerValue_u64[0], vrgoutP, vl) ;	\
	if(NUMCHANNEL>= 4) vrsum23 = _vel_pvfmad_vvsvl(vrsum23, pKerValue_u64[1], vrgoutP, vl) ;	\
	if(NUMCHANNEL>= 6) vrsum45 = _vel_pvfmad_vvsvl(vrsum45, pKerValue_u64[2], vrgoutP, vl) ;	\
	if(NUMCHANNEL>= 8) vrsum67 = _vel_pvfmad_vvsvl(vrsum67, pKerValue_u64[3], vrgoutP, vl) ;	\
	if(NUMCHANNEL>=10) vrsum89 = _vel_pvfmad_vvsvl(vrsum89, pKerValue_u64[4], vrgoutP, vl) ;	\
	if(NUMCHANNEL>=12) vrsumAB = _vel_pvfmad_vvsvl(vrsumAB, pKerValue_u64[5], vrgoutP, vl) ;	\
	if(NUMCHANNEL>=14) vrsumCD = _vel_pvfmad_vvsvl(vrsumCD, pKerValue_u64[6], vrgoutP, vl) ;	\
	if(NUMCHANNEL>=16) vrsumEF = _vel_pvfmad_vvsvl(vrsumEF, pKerValue_u64[7], vrgoutP, vl) ;	\
      }

      VFADD(vrgout, k, 0, 0) ;
#undef VFADD
    }

    __vr vrpgin_c0 = _vel_vsfa_vvssl(vrij, 2, (uint64_t)(pInChannel+y*strideHeight*gInWidth), vl) ;
    __vr vrpgin_c1 = _vel_vaddul_vsvl(  4*gInHeight*gInWidth,vrpgin_c0, vl) ;
    __vr vrpgin_c2 = _vel_vaddul_vsvl(2*4*gInHeight*gInWidth,vrpgin_c0, vl) ;
    __vr vrpgin_c3 = _vel_vaddul_vsvl(3*4*gInHeight*gInWidth,vrpgin_c0, vl) ;
    __vr vrpgin_c4 = _vel_vaddul_vsvl(4*4*gInHeight*gInWidth,vrpgin_c0, vl) ;
    __vr vrpgin_c5 = _vel_vaddul_vsvl(5*4*gInHeight*gInWidth,vrpgin_c0, vl) ;
    __vr vrpgin_c6 = _vel_vaddul_vsvl(6*4*gInHeight*gInWidth,vrpgin_c0, vl) ;
    __vr vrpgin_c7 = _vel_vaddul_vsvl(7*4*gInHeight*gInWidth,vrpgin_c0, vl) ;
    __vr vrpgin_c8 = _vel_vaddul_vsvl(8*4*gInHeight*gInWidth,vrpgin_c0, vl) ;
    __vr vrpgin_c9 = _vel_vaddul_vsvl(9*4*gInHeight*gInWidth,vrpgin_c0, vl) ;
    __vr vrpgin_cA = _vel_vaddul_vsvl(10*4*gInHeight*gInWidth,vrpgin_c0, vl) ;
    __vr vrpgin_cB = _vel_vaddul_vsvl(11*4*gInHeight*gInWidth,vrpgin_c0, vl) ;
    __vr vrpgin_cC = _vel_vaddul_vsvl(12*4*gInHeight*gInWidth,vrpgin_c0, vl) ;
    __vr vrpgin_cD = _vel_vaddul_vsvl(13*4*gInHeight*gInWidth,vrpgin_c0, vl) ;
    __vr vrpgin_cE = _vel_vaddul_vsvl(14*4*gInHeight*gInWidth,vrpgin_c0, vl) ;
    __vr vrpgin_cF = _vel_vaddul_vsvl(15*4*gInHeight*gInWidth,vrpgin_c0, vl) ;

    _vel_svob() ;

    if(NUMCHANNEL>= 2) {
      _vel_vsclot_vvssl(vrsum01, vrpgin_c0, 0, 0, vl) ;
      _vel_vscuot_vvssl(vrsum01, vrpgin_c1, 0, 0, vl) ;
    }
    if(NUMCHANNEL>= 4) {
      _vel_vsclot_vvssl(vrsum23, vrpgin_c2, 0, 0, vl) ;
      _vel_vscuot_vvssl(vrsum23, vrpgin_c3, 0, 0, vl) ;
    }
    if(NUMCHANNEL>= 6) {
      _vel_vsclot_vvssl(vrsum45, vrpgin_c4, 0, 0, vl) ;
      _vel_vscuot_vvssl(vrsum45, vrpgin_c5, 0, 0, vl) ;
    }
    if(NUMCHANNEL>= 8) {
      _vel_vsclot_vvssl(vrsum67, vrpgin_c6, 0, 0, vl) ;
      _vel_vscuot_vvssl(vrsum67, vrpgin_c7, 0, 0, vl) ;
    }
    if(NUMCHANNEL>=10) {
      _vel_vsclot_vvssl(vrsum89, vrpgin_c8, 0, 0, vl) ;
      _vel_vscuot_vvssl(vrsum89, vrpgin_c9, 0, 0, vl) ;
    }
    if(NUMCHANNEL>=12) {
      _vel_vsclot_vvssl(vrsumAB, vrpgin_cA, 0, 0, vl) ;
      _vel_vscuot_vvssl(vrsumAB, vrpgin_cB, 0, 0, vl) ;
    }
    if(NUMCHANNEL>=14) {
      _vel_vsclot_vvssl(vrsumCD, vrpgin_cC, 0, 0, vl) ;
      _vel_vscuot_vvssl(vrsumCD, vrpgin_cD, 0, 0, vl) ;
    }
    if(NUMCHANNEL>=16) {
      _vel_vsclot_vvssl(vrsumEF, vrpgin_cE, 0, 0, vl) ;
      _vel_vscuot_vvssl(vrsumEF, vrpgin_cF, 0, 0, vl) ;
    }
  }

  _vel_svob() ;
}

template<filterLayout_t FLAYOUT>
static inline void convloop(
    const float * __restrict__ pGOut,
    const float * __restrict__ pKernel,
    float * __restrict__ const pGIn,
    const int64_t batch,
    const int64_t group,
    const int64_t gOutChannel,
    const int64_t gOutWidth,
    const int64_t gOutHeight,
    const int64_t gInChannel,
    const int64_t gInWidth,
    const int64_t gInHeight,
    const int64_t gInChannelGroup,
    const int64_t gOutChannelGroup,
    const int64_t strideWidth,
    const int64_t strideHeight
)
{

  const int64_t nY = VLEN / gOutWidth ;

  __vr vrseq = _vel_vseq_vl(nY*gOutWidth) ;
  __vr vry  = _vel_vdivsl_vvsl(vrseq, gOutWidth, nY*gOutWidth) ;
  __vr vrx  = _vel_vsubsl_vvvl(vrseq, _vel_vmulul_vsvl(gOutWidth,vry, nY*gOutWidth), nY*gOutWidth) ;

  __vr vri   = _vel_vmulsl_vsvl(strideHeight, vry, nY*gOutWidth) ;
  __vr vrj   = _vel_vmulsl_vsvl(strideWidth,  vrx, nY*gOutWidth) ;
  __vr vrij = _vel_vaddul_vvvl(vrj, _vel_vmulul_vsvl(gInWidth, vri, nY*gOutWidth), nY*gOutWidth) ;

  const int64_t usePackedKernel = (((uint64_t)pKernel) & 0x07) == 0 && (gInChannelGroup & 0x01) == 0 ?  1 : 0  ;

  for (int64_t n=0; n<batch; n++) {
    for (int64_t g = 0; g < group; g++) {

      int64_t gInGroupOffset  = g * gInChannelGroup * gInHeight * gInWidth;
      int64_t gOutGroupOffset = g * gOutChannelGroup * gOutHeight * gOutWidth;
      int64_t kernGroupOffset = g * gOutChannelGroup * gInChannelGroup ;

      const int64_t remain = gInChannelGroup & 0xf ;

      int64_t c=0;
      switch(remain) {
      case 1:
	func_odd<FLAYOUT, 1>(pGOut, pKernel, pGIn,
	   gOutChannel, gOutWidth, gOutHeight,
	   gInChannel, gInWidth, gInHeight,
	   strideWidth, strideHeight,
	   gInChannelGroup, gOutChannelGroup,
	   gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	   n, c, nY, vrij) ;
	c+=1 ;
	break ;
      case 2:
	if ( FLAYOUT ==  VEDNN_FILTER_LAYOUT_NCHW && usePackedKernel ) {
	  func_even_filternchw_packedkernel<2>(pGOut, pKernel, pGIn,
	     gOutChannel, gOutWidth, gOutHeight,
	     gInChannel, gInWidth, gInHeight,
	     strideWidth, strideHeight,
	     gInChannelGroup, gOutChannelGroup,
	     gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	     n, c, nY, vrij) ;
	}
	else {
	  func_even<FLAYOUT, 2>(pGOut, pKernel, pGIn,
	     gOutChannel, gOutWidth, gOutHeight,
	     gInChannel, gInWidth, gInHeight,
	     strideWidth, strideHeight,
	     gInChannelGroup, gOutChannelGroup,
	     gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	     n, c, nY, vrij) ;
	}
	c+=2 ;
	break ;
      case 3:
	func_odd<FLAYOUT, 3>(pGOut, pKernel, pGIn,
	   gOutChannel, gOutWidth, gOutHeight,
	   gInChannel, gInWidth, gInHeight,
	   strideWidth, strideHeight,
	   gInChannelGroup, gOutChannelGroup,
	   gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	   n, c, nY, vrij) ;
	c+=3 ;
	break ;
      case 4:
	if ( FLAYOUT ==  VEDNN_FILTER_LAYOUT_NCHW && usePackedKernel ) {
	  func_even_filternchw_packedkernel<4>(pGOut, pKernel, pGIn,
	     gOutChannel, gOutWidth, gOutHeight,
	     gInChannel, gInWidth, gInHeight,
	     strideWidth, strideHeight,
	     gInChannelGroup, gOutChannelGroup,
	     gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	     n, c, nY, vrij) ;
	}
	else {
	  func_even<FLAYOUT, 4>(pGOut, pKernel, pGIn,
	     gOutChannel, gOutWidth, gOutHeight,
	     gInChannel, gInWidth, gInHeight,
	     strideWidth, strideHeight,
	     gInChannelGroup, gOutChannelGroup,
	     gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	     n, c, nY, vrij) ;
	}
	c+=4 ;
	break ;
      case 5:
	func_odd<FLAYOUT, 5>(pGOut, pKernel, pGIn,
	   gOutChannel, gOutWidth, gOutHeight,
	   gInChannel, gInWidth, gInHeight,
	   strideWidth, strideHeight,
	   gInChannelGroup, gOutChannelGroup,
	   gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	   n, c, nY, vrij) ;
	c+=5 ;
	break ;
      case 6:
	if ( FLAYOUT ==  VEDNN_FILTER_LAYOUT_NCHW && usePackedKernel ) {
	  func_even_filternchw_packedkernel<6>(pGOut, pKernel, pGIn,
	     gOutChannel, gOutWidth, gOutHeight,
	     gInChannel, gInWidth, gInHeight,
	     strideWidth, strideHeight,
	     gInChannelGroup, gOutChannelGroup,
	     gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	     n, c, nY, vrij) ;
	}
	else {
	  func_even<FLAYOUT, 6>(pGOut, pKernel, pGIn,
	     gOutChannel, gOutWidth, gOutHeight,
	     gInChannel, gInWidth, gInHeight,
	     strideWidth, strideHeight,
	     gInChannelGroup, gOutChannelGroup,
	     gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	     n, c, nY, vrij) ;
	}
	c+=6 ;
	break ;
      case 7:
	func_odd<FLAYOUT, 7>(pGOut, pKernel, pGIn,
	   gOutChannel, gOutWidth, gOutHeight,
	   gInChannel, gInWidth, gInHeight,
	   strideWidth, strideHeight,
	   gInChannelGroup, gOutChannelGroup,
	   gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	   n, c, nY, vrij) ;
	c+=7 ;
	break ;
      case 8:
	if ( FLAYOUT ==  VEDNN_FILTER_LAYOUT_NCHW && usePackedKernel ) {
	  func_even_filternchw_packedkernel<8>(pGOut, pKernel, pGIn,
	     gOutChannel, gOutWidth, gOutHeight,
	     gInChannel, gInWidth, gInHeight,
	     strideWidth, strideHeight,
	     gInChannelGroup, gOutChannelGroup,
	     gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	     n, c, nY, vrij) ;
	}
	else {
	  func_even<FLAYOUT, 8>(pGOut, pKernel, pGIn,
	     gOutChannel, gOutWidth, gOutHeight,
	     gInChannel, gInWidth, gInHeight,
	     strideWidth, strideHeight,
	     gInChannelGroup, gOutChannelGroup,
	     gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	     n, c, nY, vrij) ;
	}
	c+=8 ;
	break ;
      case 9:
	func_odd<FLAYOUT, 9>(pGOut, pKernel, pGIn,
	   gOutChannel, gOutWidth, gOutHeight,
	   gInChannel, gInWidth, gInHeight,
	   strideWidth, strideHeight,
	   gInChannelGroup, gOutChannelGroup,
	   gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	   n, c, nY, vrij) ;
	c+=9 ;
	break ;
      case 10:
	if ( FLAYOUT ==  VEDNN_FILTER_LAYOUT_NCHW && usePackedKernel ) {
	  func_even_filternchw_packedkernel<10>(pGOut, pKernel, pGIn,
	     gOutChannel, gOutWidth, gOutHeight,
	     gInChannel, gInWidth, gInHeight,
	     strideWidth, strideHeight,
	     gInChannelGroup, gOutChannelGroup,
	     gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	     n, c, nY, vrij) ;
	}
	else {
	  func_even<FLAYOUT, 10>(pGOut, pKernel, pGIn,
	     gOutChannel, gOutWidth, gOutHeight,
	     gInChannel, gInWidth, gInHeight,
	     strideWidth, strideHeight,
	     gInChannelGroup, gOutChannelGroup,
	     gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	     n, c, nY, vrij) ;
	}
	c+=10 ;
	break ;
      case 11:
	func_odd<FLAYOUT, 11>(pGOut, pKernel, pGIn,
	   gOutChannel, gOutWidth, gOutHeight,
	   gInChannel, gInWidth, gInHeight,
	   strideWidth, strideHeight,
	   gInChannelGroup, gOutChannelGroup,
	   gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	   n, c, nY, vrij) ;
	c+=11 ;
	break ;
      case 12:
	if ( FLAYOUT ==  VEDNN_FILTER_LAYOUT_NCHW && usePackedKernel ) {
	  func_even_filternchw_packedkernel<12>(pGOut, pKernel, pGIn,
	     gOutChannel, gOutWidth, gOutHeight,
	     gInChannel, gInWidth, gInHeight,
	     strideWidth, strideHeight,
	     gInChannelGroup, gOutChannelGroup,
	     gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	     n, c, nY, vrij) ;
	}
	else {
	  func_even<FLAYOUT, 12>(pGOut, pKernel, pGIn,
	     gOutChannel, gOutWidth, gOutHeight,
	     gInChannel, gInWidth, gInHeight,
	     strideWidth, strideHeight,
	     gInChannelGroup, gOutChannelGroup,
	     gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	     n, c, nY, vrij) ;
	}
	c+=12 ;
	break ;
      case 13:
	func_odd<FLAYOUT, 13>(pGOut, pKernel, pGIn,
	   gOutChannel, gOutWidth, gOutHeight,
	   gInChannel, gInWidth, gInHeight,
	   strideWidth, strideHeight,
	   gInChannelGroup, gOutChannelGroup,
	   gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	   n, c, nY, vrij) ;
	c+=13 ;
	break ;
      case 14:
	if ( FLAYOUT ==  VEDNN_FILTER_LAYOUT_NCHW && usePackedKernel ) {
	  func_even_filternchw_packedkernel<14>(pGOut, pKernel, pGIn,
	     gOutChannel, gOutWidth, gOutHeight,
	     gInChannel, gInWidth, gInHeight,
	     strideWidth, strideHeight,
	     gInChannelGroup, gOutChannelGroup,
	     gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	     n, c, nY, vrij) ;
	}
	else {
	  func_even<FLAYOUT, 14>(pGOut, pKernel, pGIn,
	     gOutChannel, gOutWidth, gOutHeight,
	     gInChannel, gInWidth, gInHeight,
	     strideWidth, strideHeight,
	     gInChannelGroup, gOutChannelGroup,
	     gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	     n, c, nY, vrij) ;
	}
	c+=14 ;
	break ;
      case 15:
	func_odd<FLAYOUT, 15>(pGOut, pKernel, pGIn,
	   gOutChannel, gOutWidth, gOutHeight,
	   gInChannel, gInWidth, gInHeight,
	   strideWidth, strideHeight,
	   gInChannelGroup, gOutChannelGroup,
	   gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	   n, c, nY, vrij) ;
	c+=15 ;
	break ;
      default :
	break ;
      }
      for (; c<gInChannelGroup; ) {
	if ( FLAYOUT ==  VEDNN_FILTER_LAYOUT_NCHW && usePackedKernel ) {
	  func_even_filternchw_packedkernel<16>(pGOut, pKernel, pGIn,
	     gOutChannel, gOutWidth, gOutHeight,
	     gInChannel, gInWidth, gInHeight,
	     strideWidth, strideHeight,
	     gInChannelGroup, gOutChannelGroup,
	     gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	     n, c, nY, vrij) ;
	}
	else {
	  func_even<FLAYOUT, 16>(pGOut, pKernel, pGIn,
	     gOutChannel, gOutWidth, gOutHeight,
	     gInChannel, gInWidth, gInHeight,
	     strideWidth, strideHeight,
	     gInChannelGroup, gOutChannelGroup,
	     gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	     n, c, nY, vrij) ;
	}
	c+= 16 ;
      } // gInChannel
    } // group
  } // batch
}

extern "C"
vednnError_t
vednnConvolutionBackwardData_direct_dil1_pad0_ker1_owU128(
    const vednnTensorParam_t * 		pParamGradOut,
    const void *			pDataGradOut,
    const vednnFilterParam_t *	 	pParamKernel,
    const void * 			pDataKernel,
    const vednnConvolutionParam_t * 	pParamConv,
    const vednnTensorParam_t * 		pParamGradIn,
    void * 				pDataGradIn
)
{
  const int64_t batch       = pParamGradOut->batch;
  const int64_t gOutChannel = pParamGradOut->channel;
  const int64_t gOutWidth   = pParamGradOut->width;
  const int64_t gOutHeight  = pParamGradOut->height;
  const int64_t gInChannel  = pParamGradIn->channel;
  const int64_t gInWidth    = pParamGradIn->width;
  const int64_t gInHeight   = pParamGradIn->height;
//  const int64_t kernWidth   = pParamKernel->width;	// 1
//  const int64_t kernHeight  = pParamKernel->height;	// 1

  const int64_t filter_layout = pParamKernel->layout ;

  const int64_t group          = pParamConv->group;
  const int64_t strideWidth    = pParamConv->strideWidth;
  const int64_t strideHeight   = pParamConv->strideHeight;
//  const int64_t padWidth       = pParamConv->padWidth;	// 0
//  const int64_t padHeight      = pParamConv->padHeight;	// 0
//  const int64_t dilationWidth  = pParamConv->dilationWidth;	// 1
//  const int64_t dilationHeight = pParamConv->dilationHeight;	// 1

  const int64_t gOutChannelGroup = gOutChannel  / group;
  const int64_t gInChannelGroup  = gInChannel / group;

  const float *  pGOut   = (const float *) pDataGradOut;
  const float *  pKernel = (const float *) pDataKernel;
  float *  const pGIn    = (float * const) pDataGradIn;


  if( filter_layout == VEDNN_FILTER_LAYOUT_NCHW) {
    convloop<VEDNN_FILTER_LAYOUT_NCHW>(pGOut, pKernel, pGIn,
	       batch, group,
	       gOutChannel, gOutWidth, gOutHeight,
	       gInChannel, gInWidth, gInHeight,
	       gInChannelGroup, gOutChannelGroup,
	       strideWidth, strideHeight ) ;
  }
  else {
    convloop<VEDNN_FILTER_LAYOUT_HWCN>(pGOut, pKernel, pGIn,
	       batch, group,
	       gOutChannel, gOutWidth, gOutHeight,
	       gInChannel, gInWidth, gInHeight,
	       gInChannelGroup, gOutChannelGroup,
	       strideWidth, strideHeight ) ;
  }

  return VEDNN_SUCCESS;
}
