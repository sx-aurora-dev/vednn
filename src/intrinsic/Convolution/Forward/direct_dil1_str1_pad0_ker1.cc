#include <stdint.h>
#include <stdlib.h>

#include "vednn.h"
#include "vednn_util.h"

#include "velintrin.h"
#define VLEN	(256)

template<filterLayout_t FLAYOUT, int NUMKERNEL, bool ADDBIAS>
static inline void func_odd(
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
  const int64_t inChannelGroup,
  const int64_t outChannelGroup,
  const int64_t inGroupOffset,
  const int64_t outGroupOffset,
  const int64_t biasGroupOffset,
  const int64_t kernGroupOffset,
  const int64_t n,
  const int64_t k
)
{
  int64_t outIndex = outGroupOffset + (n * outChannel + k  ) * outHeight * outWidth ;

  const float   bias0  = ADDBIAS ?  pBias[biasGroupOffset+k+ 0] : 0.f ;
  const int64_t bias12 = ADDBIAS ?  _vel_pack_f32p(pBias+biasGroupOffset+k+ 1, pBias+biasGroupOffset+k+ 2) : 0UL ;
  const int64_t bias34 = ADDBIAS ?  _vel_pack_f32p(pBias+biasGroupOffset+k+ 3, pBias+biasGroupOffset+k+ 4) : 0UL ;
  const int64_t bias56 = ADDBIAS ?  _vel_pack_f32p(pBias+biasGroupOffset+k+ 5, pBias+biasGroupOffset+k+ 6) : 0UL ;
  const int64_t bias78 = ADDBIAS ?  _vel_pack_f32p(pBias+biasGroupOffset+k+ 7, pBias+biasGroupOffset+k+ 8) : 0UL ;
  const int64_t bias9A = ADDBIAS ?  _vel_pack_f32p(pBias+biasGroupOffset+k+ 9, pBias+biasGroupOffset+k+10) : 0UL ;
  const int64_t biasBC = ADDBIAS ?  _vel_pack_f32p(pBias+biasGroupOffset+k+11, pBias+biasGroupOffset+k+12) : 0UL ;
  const int64_t biasDE = ADDBIAS ?  _vel_pack_f32p(pBias+biasGroupOffset+k+13, pBias+biasGroupOffset+k+14) : 0UL ;



  for (int64_t op = 0; op < outHeight * outWidth; op+=VLEN) {
    const int64_t vl = outHeight * outWidth - op < VLEN ? outHeight * outWidth - op : VLEN ;

    __vr vrsum0  = _vel_vbrds_vsl(bias0, vl) ;
    __vr vrsum12 = _vel_pvbrd_vsl(bias12, vl) ;
    __vr vrsum34 = _vel_pvbrd_vsl(bias34, vl) ;
    __vr vrsum56 = _vel_pvbrd_vsl(bias56, vl) ;
    __vr vrsum78 = _vel_pvbrd_vsl(bias78, vl) ;
    __vr vrsum9A = _vel_pvbrd_vsl(bias9A, vl) ;
    __vr vrsumBC = _vel_pvbrd_vsl(biasBC, vl) ;
    __vr vrsumDE = _vel_pvbrd_vsl(biasDE, vl) ;

    int64_t c = 0 ;

    if( ( inChannelGroup & 0x01 ) == 1 ) {
      const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;


      __vr vrin_c0  = _vel_vldu_vssl(4,&pInChannel[op], vl) ;
      __vr vrin_c0P = _vel_vshf_vvvsl(vrin_c0, vrin_c0, VE_VSHUFFLE_YUZU, vl) ;

#define FILTER_OFFSET(k,c) ( kernGroupOffset + filter_index<FLAYOUT>(k,c,0,0, inChannelGroup, outChannelGroup, 1, 1) )

#define VFMAD(VRSUM, VRINP, K, C)								\
      {												\
	const uint64_t kerValue = _vel_pack_f32p(pKernel + FILTER_OFFSET(k+(K),   c+(C)),	\
						 pKernel + FILTER_OFFSET(k+(K)+1, c+(C))) ;	\
	VRSUM = _vel_pvfmad_vvsvl(VRSUM, kerValue, VRINP, vl) ;					\
      }

      {
	const float kerValue0_c0 = pKernel[FILTER_OFFSET(k+ 0, c+0)] ;
	vrsum0 = _vel_vfmads_vvsvl(vrsum0, kerValue0_c0, vrin_c0, vl) ;
      }
      if(NUMKERNEL>= 3) {
	VFMAD(vrsum12, vrin_c0P, 1, 0) ;
      }
      if(NUMKERNEL>= 5) {
	VFMAD(vrsum34, vrin_c0P, 3, 0) ;
      }
      if(NUMKERNEL>= 7) {
	VFMAD(vrsum56, vrin_c0P, 5, 0) ;
      }
      if(NUMKERNEL>= 9) {
	VFMAD(vrsum78, vrin_c0P, 7, 0) ;
      }
      if(NUMKERNEL>= 11) {
	VFMAD(vrsum9A, vrin_c0P, 9, 0) ;
      }
      if(NUMKERNEL>= 13) {
	VFMAD(vrsumBC, vrin_c0P, 11, 0) ;
      }
      if(NUMKERNEL>= 15) {
	VFMAD(vrsumDE, vrin_c0P, 13, 0) ;
      }

      c++ ;
    }
    for( ; c < inChannelGroup ; c+=2 ) {
      const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c  ) * inHeight * inWidth ) ;

      __vr vrin_c0  = _vel_vldu_vssl(4,&pInChannel[op], vl) ;
      __vr vrin_c1  = _vel_vldu_vssl(4,&pInChannel[op + inHeight * inWidth ], vl) ;
      __vr vrin_c0P = _vel_vshf_vvvsl(vrin_c0, vrin_c0, VE_VSHUFFLE_YUZU, vl) ;
      __vr vrin_c1P = _vel_vshf_vvvsl(vrin_c1, vrin_c1, VE_VSHUFFLE_YUZU, vl) ;

      {
	const float kerValue0_c0 = pKernel[FILTER_OFFSET(k+ 0, c+0)] ;
	const float kerValue0_c1 = pKernel[FILTER_OFFSET(k+ 0, c+1)] ;
	vrsum0 = _vel_vfmads_vvsvl(vrsum0, kerValue0_c0, vrin_c0, vl) ;
	vrsum0 = _vel_vfmads_vvsvl(vrsum0, kerValue0_c1, vrin_c1, vl) ;
      }
      if(NUMKERNEL>= 3) {
	VFMAD(vrsum12, vrin_c0P, 1, 0) ;
	VFMAD(vrsum12, vrin_c1P, 1, 1) ;
      }
      if(NUMKERNEL>= 5) {
	VFMAD(vrsum34, vrin_c0P, 3, 0) ;
	VFMAD(vrsum34, vrin_c1P, 3, 1) ;
      }
      if(NUMKERNEL>= 7) {
	VFMAD(vrsum56, vrin_c0P, 5, 0) ;
	VFMAD(vrsum56, vrin_c1P, 5, 1) ;
      }
      if(NUMKERNEL>= 9) {
	VFMAD(vrsum78, vrin_c0P, 7, 0) ;
	VFMAD(vrsum78, vrin_c1P, 7, 1) ;
      }
      if(NUMKERNEL>= 11) {
	VFMAD(vrsum9A, vrin_c0P, 9, 0) ;
	VFMAD(vrsum9A, vrin_c1P, 9, 1) ;
      }
      if(NUMKERNEL>= 13) {
	VFMAD(vrsumBC, vrin_c0P, 11, 0) ;
	VFMAD(vrsumBC, vrin_c1P, 11, 1) ;
      }
      if(NUMKERNEL>= 15) {
	VFMAD(vrsumDE, vrin_c0P, 13, 0) ;
	VFMAD(vrsumDE, vrin_c1P, 13, 1) ;
      }
#undef FILTER_OFFSET
    } // inChannel

    _vel_vstu_vssl(vrsum0,  4, pOut+outIndex + 0 * outHeight*outWidth, vl) ;
    if(NUMKERNEL>= 3) {
	_vel_vstu_vssl(vrsum12, 4, pOut+outIndex + 1 * outHeight*outWidth, vl) ;
	_vel_vstl_vssl(vrsum12, 4, pOut+outIndex + 2 * outHeight*outWidth, vl) ;
    }
    if(NUMKERNEL>= 5) {
	_vel_vstu_vssl(vrsum34, 4, pOut+outIndex + 3 * outHeight*outWidth, vl) ;
	_vel_vstl_vssl(vrsum34, 4, pOut+outIndex + 4 * outHeight*outWidth, vl) ;
    }
    if(NUMKERNEL>= 7) {
	_vel_vstu_vssl(vrsum56, 4, pOut+outIndex + 5 * outHeight*outWidth, vl) ;
	_vel_vstl_vssl(vrsum56, 4, pOut+outIndex + 6 * outHeight*outWidth, vl) ;
    }
    if(NUMKERNEL>= 9) {
	_vel_vstu_vssl(vrsum78, 4, pOut+outIndex + 7 * outHeight*outWidth, vl) ;
	_vel_vstl_vssl(vrsum78, 4, pOut+outIndex + 8 * outHeight*outWidth, vl) ;
    }
    if(NUMKERNEL>= 11) {
	_vel_vstu_vssl(vrsum9A, 4, pOut+outIndex + 9 * outHeight*outWidth, vl) ;
	_vel_vstl_vssl(vrsum9A, 4, pOut+outIndex +10 * outHeight*outWidth, vl) ;
    }
    if(NUMKERNEL>= 13) {
	_vel_vstu_vssl(vrsumBC, 4, pOut+outIndex +11 * outHeight*outWidth, vl) ;
	_vel_vstl_vssl(vrsumBC, 4, pOut+outIndex +12 * outHeight*outWidth, vl) ;
    }
    if(NUMKERNEL>= 15) {
	_vel_vstu_vssl(vrsumDE, 4, pOut+outIndex +13 * outHeight*outWidth, vl) ;
	_vel_vstl_vssl(vrsumDE, 4, pOut+outIndex +14 * outHeight*outWidth, vl) ;
    }

    outIndex += vl ;
  } // outPixels
}

template<filterLayout_t FLAYOUT, int NUMKERNEL, bool ADDBIAS>
static inline void func_even(
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
  const int64_t inChannelGroup,
  const int64_t outChannelGroup,
  const int64_t inGroupOffset,
  const int64_t outGroupOffset,
  const int64_t biasGroupOffset,
  const int64_t kernGroupOffset,
  const int64_t n,
  const int64_t k
)
{
  int64_t outIndex = outGroupOffset + (n * outChannel + k  ) * outHeight * outWidth ;

  const int64_t bias01 = ADDBIAS ?  _vel_pack_f32p(pBias+biasGroupOffset+k+ 0, pBias+biasGroupOffset+k+ 1) : 0UL ;
  const int64_t bias23 = ADDBIAS ?  _vel_pack_f32p(pBias+biasGroupOffset+k+ 2, pBias+biasGroupOffset+k+ 3) : 0UL ;
  const int64_t bias45 = ADDBIAS ?  _vel_pack_f32p(pBias+biasGroupOffset+k+ 4, pBias+biasGroupOffset+k+ 5) : 0UL ;
  const int64_t bias67 = ADDBIAS ?  _vel_pack_f32p(pBias+biasGroupOffset+k+ 6, pBias+biasGroupOffset+k+ 7) : 0UL ;
  const int64_t bias89 = ADDBIAS ?  _vel_pack_f32p(pBias+biasGroupOffset+k+ 8, pBias+biasGroupOffset+k+ 9) : 0UL ;
  const int64_t biasAB = ADDBIAS ?  _vel_pack_f32p(pBias+biasGroupOffset+k+10, pBias+biasGroupOffset+k+11) : 0UL ;
  const int64_t biasCD = ADDBIAS ?  _vel_pack_f32p(pBias+biasGroupOffset+k+12, pBias+biasGroupOffset+k+13) : 0UL ;
  const int64_t biasEF = ADDBIAS ?  _vel_pack_f32p(pBias+biasGroupOffset+k+14, pBias+biasGroupOffset+k+15) : 0UL ;


  for (int64_t op = 0; op < outHeight * outWidth; op+=VLEN) {
    const int64_t vl = outHeight * outWidth - op < VLEN ? outHeight * outWidth - op : VLEN ;

    __vr vrsum01 = _vel_pvbrd_vsl(bias01, vl) ;
    __vr vrsum23 = _vel_pvbrd_vsl(bias23, vl) ;
    __vr vrsum45 = _vel_pvbrd_vsl(bias45, vl) ;
    __vr vrsum67 = _vel_pvbrd_vsl(bias67, vl) ;
    __vr vrsum89 = _vel_pvbrd_vsl(bias89, vl) ;
    __vr vrsumAB = _vel_pvbrd_vsl(biasAB, vl) ;
    __vr vrsumCD = _vel_pvbrd_vsl(biasCD, vl) ;
    __vr vrsumEF = _vel_pvbrd_vsl(biasEF, vl) ;

    int64_t c = 0 ;

    if( ( inChannelGroup & 0x01 ) == 1 ) {
      const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;


      __vr vrin_c0  = _vel_vldu_vssl(4,&pInChannel[op], vl) ;
      __vr vrin_c0P = _vel_vshf_vvvsl(vrin_c0, vrin_c0, VE_VSHUFFLE_YUZU, vl) ;

#define FILTER_OFFSET(k,c) ( kernGroupOffset + filter_index<FLAYOUT>(k,c,0,0, inChannelGroup, outChannelGroup, 1, 1) )

#define VFMAD(VRSUM, VRINP, K, C)								\
      {												\
	const uint64_t kerValue = _vel_pack_f32p(pKernel + FILTER_OFFSET(k+(K),   c+(C)),	\
						 pKernel + FILTER_OFFSET(k+(K)+1, c+(C))) ;	\
	VRSUM = _vel_pvfmad_vvsvl(VRSUM, kerValue, VRINP, vl) ;					\
      }

      if(NUMKERNEL>= 2) {
	VFMAD(vrsum01, vrin_c0P, 0, 0) ;
      }
      if(NUMKERNEL>= 4) {
	VFMAD(vrsum23, vrin_c0P, 2, 0) ;
      }
      if(NUMKERNEL>= 6) {
	VFMAD(vrsum45, vrin_c0P, 4, 0) ;
      }
      if(NUMKERNEL>= 8) {
	VFMAD(vrsum67, vrin_c0P, 6, 0) ;
      }
      if(NUMKERNEL>= 10) {
	VFMAD(vrsum89, vrin_c0P, 8, 0) ;
      }
      if(NUMKERNEL>= 12) {
	VFMAD(vrsumAB, vrin_c0P, 10, 0) ;
      }
      if(NUMKERNEL>= 14) {
	VFMAD(vrsumCD, vrin_c0P, 12, 0) ;
      }
      if(NUMKERNEL>= 16) {
	VFMAD(vrsumEF, vrin_c0P, 14, 0) ;
      }

      c++ ;
    }
    for( ; c < inChannelGroup ; c+=2 ) {

      const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c  ) * inHeight * inWidth ) ;

      __vr vrin_c0  = _vel_vldu_vssl(4,&pInChannel[op], vl) ;
      __vr vrin_c1  = _vel_vldu_vssl(4,&pInChannel[op + inHeight * inWidth ], vl) ;
      __vr vrin_c0P = _vel_vshf_vvvsl(vrin_c0, vrin_c0, VE_VSHUFFLE_YUZU, vl) ;
      __vr vrin_c1P = _vel_vshf_vvvsl(vrin_c1, vrin_c1, VE_VSHUFFLE_YUZU, vl) ;

      if(NUMKERNEL>= 2) {
	VFMAD(vrsum01, vrin_c0P, 0, 0) ;
	VFMAD(vrsum01, vrin_c1P, 0, 1) ;
      }
      if(NUMKERNEL>= 4) {
	VFMAD(vrsum23, vrin_c0P, 2, 0) ;
	VFMAD(vrsum23, vrin_c1P, 2, 1) ;
      }
      if(NUMKERNEL>= 6) {
	VFMAD(vrsum45, vrin_c0P, 4, 0) ;
	VFMAD(vrsum45, vrin_c1P, 4, 1) ;
      }
      if(NUMKERNEL>= 8) {
	VFMAD(vrsum67, vrin_c0P, 6, 0) ;
	VFMAD(vrsum67, vrin_c1P, 6, 1) ;
      }
      if(NUMKERNEL>= 10) {
	VFMAD(vrsum89, vrin_c0P, 8, 0) ;
	VFMAD(vrsum89, vrin_c1P, 8, 1) ;
      }
      if(NUMKERNEL>= 12) {
	VFMAD(vrsumAB, vrin_c0P, 10, 0) ;
	VFMAD(vrsumAB, vrin_c1P, 10, 1) ;
      }
      if(NUMKERNEL>= 14) {
	VFMAD(vrsumCD, vrin_c0P, 12, 0) ;
	VFMAD(vrsumCD, vrin_c1P, 12, 1) ;
      }
      if(NUMKERNEL>= 16) {
	VFMAD(vrsumEF, vrin_c0P, 14, 0) ;
	VFMAD(vrsumEF, vrin_c1P, 14, 1) ;
      }
#undef VFMAD
#undef FILTER_OFFSET
    } // inChannel

    if(NUMKERNEL>= 2) {
	_vel_vstu_vssl(vrsum01, 4, pOut+outIndex + 0 * outHeight*outWidth, vl) ;
	_vel_vstl_vssl(vrsum01, 4, pOut+outIndex + 1 * outHeight*outWidth, vl) ;
    }
    if(NUMKERNEL>= 4) {
	_vel_vstu_vssl(vrsum23, 4, pOut+outIndex + 2 * outHeight*outWidth, vl) ;
	_vel_vstl_vssl(vrsum23, 4, pOut+outIndex + 3 * outHeight*outWidth, vl) ;
    }
    if(NUMKERNEL>= 6) {
	_vel_vstu_vssl(vrsum45, 4, pOut+outIndex + 4 * outHeight*outWidth, vl) ;
	_vel_vstl_vssl(vrsum45, 4, pOut+outIndex + 5 * outHeight*outWidth, vl) ;
    }
    if(NUMKERNEL>= 8) {
	_vel_vstu_vssl(vrsum67, 4, pOut+outIndex + 6 * outHeight*outWidth, vl) ;
	_vel_vstl_vssl(vrsum67, 4, pOut+outIndex + 7 * outHeight*outWidth, vl) ;
    }
    if(NUMKERNEL>= 10) {
	_vel_vstu_vssl(vrsum89, 4, pOut+outIndex + 8 * outHeight*outWidth, vl) ;
	_vel_vstl_vssl(vrsum89, 4, pOut+outIndex + 9 * outHeight*outWidth, vl) ;
    }
    if(NUMKERNEL>= 12) {
	_vel_vstu_vssl(vrsumAB, 4, pOut+outIndex +10 * outHeight*outWidth, vl) ;
	_vel_vstl_vssl(vrsumAB, 4, pOut+outIndex +11 * outHeight*outWidth, vl) ;
    }
    if(NUMKERNEL>= 14) {
	_vel_vstu_vssl(vrsumCD, 4, pOut+outIndex +12 * outHeight*outWidth, vl) ;
	_vel_vstl_vssl(vrsumCD, 4, pOut+outIndex +13 * outHeight*outWidth, vl) ;
    }
    if(NUMKERNEL>= 16) {
	_vel_vstu_vssl(vrsumEF, 4, pOut+outIndex +14 * outHeight*outWidth, vl) ;
	_vel_vstl_vssl(vrsumEF, 4, pOut+outIndex +15 * outHeight*outWidth, vl) ;
    }

    outIndex += vl ;
  } // outPixels
}


template<int NUMKERNEL, bool ADDBIAS>
static inline void func_odd_filternchw_avoid_l1m(
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
  const int64_t inChannelGroup,
  const int64_t outChannelGroup,
  const int64_t inGroupOffset,
  const int64_t outGroupOffset,
  const int64_t biasGroupOffset,
  const int64_t kernGroupOffset,
  const int64_t n,
  const int64_t k
)
{

  float __attribute__ ((aligned(8))) filter[NUMKERNEL*512] ;
  uint64_t* filter_u64 = (uint64_t*) filter ;

  int64_t outIndex = outGroupOffset + (n * outChannel + k  ) * outHeight * outWidth ;

  const float   bias0  = ADDBIAS ?  pBias[biasGroupOffset+k+ 0] : 0.f ;
  const int64_t bias12 = ADDBIAS ?  _vel_pack_f32p(pBias+biasGroupOffset+k+ 1, pBias+biasGroupOffset+k+ 2) : 0UL ;
  const int64_t bias34 = ADDBIAS ?  _vel_pack_f32p(pBias+biasGroupOffset+k+ 3, pBias+biasGroupOffset+k+ 4) : 0UL ;
  const int64_t bias56 = ADDBIAS ?  _vel_pack_f32p(pBias+biasGroupOffset+k+ 5, pBias+biasGroupOffset+k+ 6) : 0UL ;
  const int64_t bias78 = ADDBIAS ?  _vel_pack_f32p(pBias+biasGroupOffset+k+ 7, pBias+biasGroupOffset+k+ 8) : 0UL ;
  const int64_t bias9A = ADDBIAS ?  _vel_pack_f32p(pBias+biasGroupOffset+k+ 9, pBias+biasGroupOffset+k+10) : 0UL ;
  const int64_t biasBC = ADDBIAS ?  _vel_pack_f32p(pBias+biasGroupOffset+k+11, pBias+biasGroupOffset+k+12) : 0UL ;
  const int64_t biasDE = ADDBIAS ?  _vel_pack_f32p(pBias+biasGroupOffset+k+13, pBias+biasGroupOffset+k+14) : 0UL ;

  for (int64_t op = 0; op < outHeight * outWidth; op+=VLEN) {
    const int64_t vl = outHeight * outWidth - op < VLEN ? outHeight * outWidth - op : VLEN ;

    __vr vrsum0  = _vel_vbrds_vsl(bias0, vl) ;
    __vr vrsum12 = _vel_pvbrd_vsl(bias12, vl) ;
    __vr vrsum34 = _vel_pvbrd_vsl(bias34, vl) ;
    __vr vrsum56 = _vel_pvbrd_vsl(bias56, vl) ;
    __vr vrsum78 = _vel_pvbrd_vsl(bias78, vl) ;
    __vr vrsum9A = _vel_pvbrd_vsl(bias9A, vl) ;
    __vr vrsumBC = _vel_pvbrd_vsl(biasBC, vl) ;
    __vr vrsumDE = _vel_pvbrd_vsl(biasDE, vl) ;

    for(int64_t c0=0; c0<inChannelGroup; c0+=256) {
      const int64_t clen = inChannelGroup - c0 < 256 ? inChannelGroup - c0 : 256 ;

      const float *pKerValue = pKernel + kernGroupOffset + (k * inChannelGroup + c0) ;

      __vr vr0 = _vel_vldu_vssl(4, pKerValue+ 0*inChannelGroup, clen) ;
      __vr vr1 = _vel_vldu_vssl(4, pKerValue+ 1*inChannelGroup, clen) ;
      __vr vr2 = _vel_vldu_vssl(4, pKerValue+ 2*inChannelGroup, clen) ;
      __vr vr3 = _vel_vldu_vssl(4, pKerValue+ 3*inChannelGroup, clen) ;
      __vr vr4 = _vel_vldu_vssl(4, pKerValue+ 4*inChannelGroup, clen) ;
      __vr vr5 = _vel_vldu_vssl(4, pKerValue+ 5*inChannelGroup, clen) ;
      __vr vr6 = _vel_vldu_vssl(4, pKerValue+ 6*inChannelGroup, clen) ;
      __vr vr7 = _vel_vldu_vssl(4, pKerValue+ 7*inChannelGroup, clen) ;
      __vr vr8 = _vel_vldu_vssl(4, pKerValue+ 8*inChannelGroup, clen) ;
      __vr vr9 = _vel_vldu_vssl(4, pKerValue+ 9*inChannelGroup, clen) ;
      __vr vrA = _vel_vldu_vssl(4, pKerValue+10*inChannelGroup, clen) ;
      __vr vrB = _vel_vldu_vssl(4, pKerValue+11*inChannelGroup, clen) ;
      __vr vrC = _vel_vldu_vssl(4, pKerValue+12*inChannelGroup, clen) ;
      __vr vrD = _vel_vldu_vssl(4, pKerValue+13*inChannelGroup, clen) ;
      __vr vrE = _vel_vldu_vssl(4, pKerValue+14*inChannelGroup, clen) ;

      __vr vr12 = _vel_vshf_vvvsl(vr1,vr2,VE_VSHUFFLE_YUZU, clen) ;
      __vr vr34 = _vel_vshf_vvvsl(vr3,vr4,VE_VSHUFFLE_YUZU, clen) ;
      __vr vr56 = _vel_vshf_vvvsl(vr5,vr6,VE_VSHUFFLE_YUZU, clen) ;
      __vr vr78 = _vel_vshf_vvvsl(vr7,vr8,VE_VSHUFFLE_YUZU, clen) ;
      __vr vr9A = _vel_vshf_vvvsl(vr9,vrA,VE_VSHUFFLE_YUZU, clen) ;
      __vr vrBC = _vel_vshf_vvvsl(vrB,vrC,VE_VSHUFFLE_YUZU, clen) ;
      __vr vrDE = _vel_vshf_vvvsl(vrD,vrE,VE_VSHUFFLE_YUZU, clen) ;

      _vel_vstu_vssl(vr0, 4, filter+(NUMKERNEL-1)*clen, clen) ;
      if(NUMKERNEL>= 3) _vel_vst_vssl(vr12, 8, filter_u64+0*clen, clen) ;
      if(NUMKERNEL>= 5) _vel_vst_vssl(vr34, 8, filter_u64+1*clen, clen) ;
      if(NUMKERNEL>= 7) _vel_vst_vssl(vr56, 8, filter_u64+2*clen, clen) ;
      if(NUMKERNEL>= 9) _vel_vst_vssl(vr78, 8, filter_u64+3*clen, clen) ;
      if(NUMKERNEL>=11) _vel_vst_vssl(vr9A, 8, filter_u64+4*clen, clen) ;
      if(NUMKERNEL>=13) _vel_vst_vssl(vrBC, 8, filter_u64+5*clen, clen) ;
      if(NUMKERNEL>=15) _vel_vst_vssl(vrDE, 8, filter_u64+6*clen, clen) ;

      for(int64_t c1 = 0; c1 < clen ; c1++ ) {
	const int64_t c = c0 + c1 ;

	const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;

	__vr vrin  = _vel_vldu_vssl(4,&pInChannel[op], vl) ;
	__vr vrinP = _vel_vshf_vvvsl(vrin, vrin, VE_VSHUFFLE_YUZU, vl) ;

	vrsum0 = _vel_vfmads_vvsvl(vrsum0, filter[(NUMKERNEL-1)*clen+c1], vrin, vl) ;
	if(NUMKERNEL>= 1) vrsum12 = _vel_pvfmad_vvsvl(vrsum12, filter_u64[0*clen+c1], vrinP, vl) ;
	if(NUMKERNEL>= 3) vrsum34 = _vel_pvfmad_vvsvl(vrsum34, filter_u64[1*clen+c1], vrinP, vl) ;
	if(NUMKERNEL>= 5) vrsum56 = _vel_pvfmad_vvsvl(vrsum56, filter_u64[2*clen+c1], vrinP, vl) ;
	if(NUMKERNEL>= 7) vrsum78 = _vel_pvfmad_vvsvl(vrsum78, filter_u64[3*clen+c1], vrinP, vl) ;
	if(NUMKERNEL>=11) vrsum9A = _vel_pvfmad_vvsvl(vrsum9A, filter_u64[4*clen+c1], vrinP, vl) ;
	if(NUMKERNEL>=13) vrsumBC = _vel_pvfmad_vvsvl(vrsumBC, filter_u64[5*clen+c1], vrinP, vl) ;
	if(NUMKERNEL>=15) vrsumDE = _vel_pvfmad_vvsvl(vrsumDE, filter_u64[6*clen+c1], vrinP, vl) ;
      } // inChannel
    }

    _vel_vstu_vssl(vrsum0,  4, pOut+outIndex + 0 * outHeight*outWidth, vl) ;
    if(NUMKERNEL>= 3) {
	_vel_vstu_vssl(vrsum12, 4, pOut+outIndex + 1 * outHeight*outWidth, vl) ;
	_vel_vstl_vssl(vrsum12, 4, pOut+outIndex + 2 * outHeight*outWidth, vl) ;
    }
    if(NUMKERNEL>= 5) {
	_vel_vstu_vssl(vrsum34, 4, pOut+outIndex + 3 * outHeight*outWidth, vl) ;
	_vel_vstl_vssl(vrsum34, 4, pOut+outIndex + 4 * outHeight*outWidth, vl) ;
    }
    if(NUMKERNEL>= 7) {
	_vel_vstu_vssl(vrsum56, 4, pOut+outIndex + 5 * outHeight*outWidth, vl) ;
	_vel_vstl_vssl(vrsum56, 4, pOut+outIndex + 6 * outHeight*outWidth, vl) ;
    }
    if(NUMKERNEL>= 9) {
	_vel_vstu_vssl(vrsum78, 4, pOut+outIndex + 7 * outHeight*outWidth, vl) ;
	_vel_vstl_vssl(vrsum78, 4, pOut+outIndex + 8 * outHeight*outWidth, vl) ;
    }
    if(NUMKERNEL>= 11) {
	_vel_vstu_vssl(vrsum9A, 4, pOut+outIndex + 9 * outHeight*outWidth, vl) ;
	_vel_vstl_vssl(vrsum9A, 4, pOut+outIndex +10 * outHeight*outWidth, vl) ;
    }
    if(NUMKERNEL>= 13) {
	_vel_vstu_vssl(vrsumBC, 4, pOut+outIndex +11 * outHeight*outWidth, vl) ;
	_vel_vstl_vssl(vrsumBC, 4, pOut+outIndex +12 * outHeight*outWidth, vl) ;
    }
    if(NUMKERNEL>= 15) {
	_vel_vstu_vssl(vrsumDE, 4, pOut+outIndex +13 * outHeight*outWidth, vl) ;
	_vel_vstl_vssl(vrsumDE, 4, pOut+outIndex +14 * outHeight*outWidth, vl) ;
    }

    outIndex += vl ;
  } // outPixels
}


template<int NUMKERNEL, bool ADDBIAS>
static inline void func_even_filternchw_avoid_l1m(
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
  const int64_t inChannelGroup,
  const int64_t outChannelGroup,
  const int64_t inGroupOffset,
  const int64_t outGroupOffset,
  const int64_t biasGroupOffset,
  const int64_t kernGroupOffset,
  const int64_t n,
  const int64_t k
)
{

  float __attribute__ ((aligned(8))) filter[NUMKERNEL*512] ;
  uint64_t* filter_u64 = (uint64_t*) filter ;

  int64_t outIndex = outGroupOffset + (n * outChannel + k  ) * outHeight * outWidth ;

  const int64_t bias01 = ADDBIAS ?  _vel_pack_f32p(pBias+biasGroupOffset+k+ 0, pBias+biasGroupOffset+k+ 1) : 0UL ;
  const int64_t bias23 = ADDBIAS ?  _vel_pack_f32p(pBias+biasGroupOffset+k+ 2, pBias+biasGroupOffset+k+ 3) : 0UL ;
  const int64_t bias45 = ADDBIAS ?  _vel_pack_f32p(pBias+biasGroupOffset+k+ 4, pBias+biasGroupOffset+k+ 5) : 0UL ;
  const int64_t bias67 = ADDBIAS ?  _vel_pack_f32p(pBias+biasGroupOffset+k+ 6, pBias+biasGroupOffset+k+ 7) : 0UL ;
  const int64_t bias89 = ADDBIAS ?  _vel_pack_f32p(pBias+biasGroupOffset+k+ 8, pBias+biasGroupOffset+k+ 9) : 0UL ;
  const int64_t biasAB = ADDBIAS ?  _vel_pack_f32p(pBias+biasGroupOffset+k+10, pBias+biasGroupOffset+k+11) : 0UL ;
  const int64_t biasCD = ADDBIAS ?  _vel_pack_f32p(pBias+biasGroupOffset+k+12, pBias+biasGroupOffset+k+13) : 0UL ;
  const int64_t biasEF = ADDBIAS ?  _vel_pack_f32p(pBias+biasGroupOffset+k+14, pBias+biasGroupOffset+k+15) : 0UL ;


  for (int64_t op = 0; op < outHeight * outWidth; op+=VLEN) {
    const int64_t vl = outHeight * outWidth - op < VLEN ? outHeight * outWidth - op : VLEN ;

    __vr vrsum01 = _vel_pvbrd_vsl(bias01, vl) ;
    __vr vrsum23 = _vel_pvbrd_vsl(bias23, vl) ;
    __vr vrsum45 = _vel_pvbrd_vsl(bias45, vl) ;
    __vr vrsum67 = _vel_pvbrd_vsl(bias67, vl) ;
    __vr vrsum89 = _vel_pvbrd_vsl(bias89, vl) ;
    __vr vrsumAB = _vel_pvbrd_vsl(biasAB, vl) ;
    __vr vrsumCD = _vel_pvbrd_vsl(biasCD, vl) ;
    __vr vrsumEF = _vel_pvbrd_vsl(biasEF, vl) ;

    for(int64_t c0=0; c0<inChannelGroup; c0+=256) {
      const int64_t clen = inChannelGroup - c0 < 256 ? inChannelGroup - c0 : 256 ;

      const float *pKerValue = pKernel + kernGroupOffset + (k * inChannelGroup + c0) ;

      __vr vr0 = _vel_vldu_vssl(4, pKerValue+ 0*inChannelGroup, clen) ;
      __vr vr1 = _vel_vldu_vssl(4, pKerValue+ 1*inChannelGroup, clen) ;
      __vr vr2 = _vel_vldu_vssl(4, pKerValue+ 2*inChannelGroup, clen) ;
      __vr vr3 = _vel_vldu_vssl(4, pKerValue+ 3*inChannelGroup, clen) ;
      __vr vr4 = _vel_vldu_vssl(4, pKerValue+ 4*inChannelGroup, clen) ;
      __vr vr5 = _vel_vldu_vssl(4, pKerValue+ 5*inChannelGroup, clen) ;
      __vr vr6 = _vel_vldu_vssl(4, pKerValue+ 6*inChannelGroup, clen) ;
      __vr vr7 = _vel_vldu_vssl(4, pKerValue+ 7*inChannelGroup, clen) ;
      __vr vr8 = _vel_vldu_vssl(4, pKerValue+ 8*inChannelGroup, clen) ;
      __vr vr9 = _vel_vldu_vssl(4, pKerValue+ 9*inChannelGroup, clen) ;
      __vr vrA = _vel_vldu_vssl(4, pKerValue+10*inChannelGroup, clen) ;
      __vr vrB = _vel_vldu_vssl(4, pKerValue+11*inChannelGroup, clen) ;
      __vr vrC = _vel_vldu_vssl(4, pKerValue+12*inChannelGroup, clen) ;
      __vr vrD = _vel_vldu_vssl(4, pKerValue+13*inChannelGroup, clen) ;
      __vr vrE = _vel_vldu_vssl(4, pKerValue+14*inChannelGroup, clen) ;
      __vr vrF = _vel_vldu_vssl(4, pKerValue+15*inChannelGroup, clen) ;

      __vr vr01 = _vel_vshf_vvvsl(vr0,vr1,VE_VSHUFFLE_YUZU, clen) ;
      __vr vr23 = _vel_vshf_vvvsl(vr2,vr3,VE_VSHUFFLE_YUZU, clen) ;
      __vr vr45 = _vel_vshf_vvvsl(vr4,vr5,VE_VSHUFFLE_YUZU, clen) ;
      __vr vr67 = _vel_vshf_vvvsl(vr6,vr7,VE_VSHUFFLE_YUZU, clen) ;
      __vr vr89 = _vel_vshf_vvvsl(vr8,vr9,VE_VSHUFFLE_YUZU, clen) ;
      __vr vrAB = _vel_vshf_vvvsl(vrA,vrB,VE_VSHUFFLE_YUZU, clen) ;
      __vr vrCD = _vel_vshf_vvvsl(vrC,vrD,VE_VSHUFFLE_YUZU, clen) ;
      __vr vrEF = _vel_vshf_vvvsl(vrE,vrF,VE_VSHUFFLE_YUZU, clen) ;

      if(NUMKERNEL>= 2) _vel_vst_vssl(vr01, 8, filter_u64,        clen) ;
      if(NUMKERNEL>= 4) _vel_vst_vssl(vr23, 8, filter_u64+1*clen, clen) ;
      if(NUMKERNEL>= 6) _vel_vst_vssl(vr45, 8, filter_u64+2*clen, clen) ;
      if(NUMKERNEL>= 8) _vel_vst_vssl(vr67, 8, filter_u64+3*clen, clen) ;
      if(NUMKERNEL>=10) _vel_vst_vssl(vr89, 8, filter_u64+4*clen, clen) ;
      if(NUMKERNEL>=12) _vel_vst_vssl(vrAB, 8, filter_u64+5*clen, clen) ;
      if(NUMKERNEL>=14) _vel_vst_vssl(vrCD, 8, filter_u64+6*clen, clen) ;
      if(NUMKERNEL>=16) _vel_vst_vssl(vrEF, 8, filter_u64+7*clen, clen) ;

      for(int64_t c1 = 0; c1 < clen ; c1++ ) {
	const int64_t c = c0 + c1 ;

	const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;

	__vr vrin  = _vel_vldu_vssl(4,&pInChannel[op], vl) ;
	__vr vrinP = _vel_vshf_vvvsl(vrin, vrin, VE_VSHUFFLE_YUZU, vl) ;

	if(NUMKERNEL>= 2) vrsum01 = _vel_pvfmad_vvsvl(vrsum01, filter_u64[0*clen+c1], vrinP, vl) ;
	if(NUMKERNEL>= 4) vrsum23 = _vel_pvfmad_vvsvl(vrsum23, filter_u64[1*clen+c1], vrinP, vl) ;
	if(NUMKERNEL>= 6) vrsum45 = _vel_pvfmad_vvsvl(vrsum45, filter_u64[2*clen+c1], vrinP, vl) ;
	if(NUMKERNEL>= 8) vrsum67 = _vel_pvfmad_vvsvl(vrsum67, filter_u64[3*clen+c1], vrinP, vl) ;
	if(NUMKERNEL>=10) vrsum89 = _vel_pvfmad_vvsvl(vrsum89, filter_u64[4*clen+c1], vrinP, vl) ;
	if(NUMKERNEL>=12) vrsumAB = _vel_pvfmad_vvsvl(vrsumAB, filter_u64[5*clen+c1], vrinP, vl) ;
	if(NUMKERNEL>=14) vrsumCD = _vel_pvfmad_vvsvl(vrsumCD, filter_u64[6*clen+c1], vrinP, vl) ;
	if(NUMKERNEL>=16) vrsumEF = _vel_pvfmad_vvsvl(vrsumEF, filter_u64[7*clen+c1], vrinP, vl) ;
      } // inChannel
    }

    if(NUMKERNEL>= 2) {
	_vel_vstu_vssl(vrsum01, 4, pOut+outIndex + 0 * outHeight*outWidth, vl) ;
	_vel_vstl_vssl(vrsum01, 4, pOut+outIndex + 1 * outHeight*outWidth, vl) ;
    }
    if(NUMKERNEL>= 4) {
	_vel_vstu_vssl(vrsum23, 4, pOut+outIndex + 2 * outHeight*outWidth, vl) ;
	_vel_vstl_vssl(vrsum23, 4, pOut+outIndex + 3 * outHeight*outWidth, vl) ;
    }
    if(NUMKERNEL>= 6) {
	_vel_vstu_vssl(vrsum45, 4, pOut+outIndex + 4 * outHeight*outWidth, vl) ;
	_vel_vstl_vssl(vrsum45, 4, pOut+outIndex + 5 * outHeight*outWidth, vl) ;
    }
    if(NUMKERNEL>= 8) {
	_vel_vstu_vssl(vrsum67, 4, pOut+outIndex + 6 * outHeight*outWidth, vl) ;
	_vel_vstl_vssl(vrsum67, 4, pOut+outIndex + 7 * outHeight*outWidth, vl) ;
    }
    if(NUMKERNEL>= 10) {
	_vel_vstu_vssl(vrsum89, 4, pOut+outIndex + 8 * outHeight*outWidth, vl) ;
	_vel_vstl_vssl(vrsum89, 4, pOut+outIndex + 9 * outHeight*outWidth, vl) ;
    }
    if(NUMKERNEL>= 12) {
	_vel_vstu_vssl(vrsumAB, 4, pOut+outIndex +10 * outHeight*outWidth, vl) ;
	_vel_vstl_vssl(vrsumAB, 4, pOut+outIndex +11 * outHeight*outWidth, vl) ;
    }
    if(NUMKERNEL>= 14) {
	_vel_vstu_vssl(vrsumCD, 4, pOut+outIndex +12 * outHeight*outWidth, vl) ;
	_vel_vstl_vssl(vrsumCD, 4, pOut+outIndex +13 * outHeight*outWidth, vl) ;
    }
    if(NUMKERNEL>= 16) {
	_vel_vstu_vssl(vrsumEF, 4, pOut+outIndex +14 * outHeight*outWidth, vl) ;
	_vel_vstl_vssl(vrsumEF, 4, pOut+outIndex +15 * outHeight*outWidth, vl) ;
    }

    outIndex += vl ;
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
    const int64_t inChannelGroup,
    const int64_t outChannelGroup
)
{
  for (int64_t n = 0; n < batch; n++) {
    for (int64_t g = 0; g < group; g++) {
	const int64_t inGroupOffset   = g * inChannelGroup * inHeight * inWidth;
	const int64_t outGroupOffset  = g * outChannelGroup * outHeight * outWidth;
	const int64_t biasGroupOffset = g * outChannelGroup;
	const int64_t kernGroupOffset = g * outChannelGroup * inChannelGroup * 1 * 1;

	const int64_t remain = outChannelGroup & 0xf ;

	int k = 0 ;
	switch( remain ) {
	case 1 :
	  func_odd<FLAYOUT, 1, ADDBIAS>(pIn, pKernel, pBias, pOut,
	     inChannel, inWidth, inHeight,
	     outChannel, outWidth, outHeight,
	     inChannelGroup, outChannelGroup,
	     inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	     n, k );
	  k+=1 ;
	  break ;
	case 2 :
	  if( FLAYOUT == VEDNN_FILTER_LAYOUT_NCHW
	      && inChannelGroup >= 512
	      && ( (inChannelGroup % 8192) < 64 || (inChannelGroup % 8192) > 8192-64 ) )
	  {
	    func_even_filternchw_avoid_l1m<2, ADDBIAS>(pIn, pKernel, pBias, pOut,
	       inChannel, inWidth, inHeight,
	       outChannel, outWidth, outHeight,
	       inChannelGroup, outChannelGroup,
	       inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	       n, k );
	  }
	  else
	  {
	    func_even<FLAYOUT, 2, ADDBIAS>(pIn, pKernel, pBias, pOut,
	       inChannel, inWidth, inHeight,
	       outChannel, outWidth, outHeight,
	       inChannelGroup, outChannelGroup,
	       inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	       n, k );
	  }
	  k+=2 ;
	  break ;
	case 3 :
	  if( FLAYOUT == VEDNN_FILTER_LAYOUT_NCHW
	      && inChannelGroup >= 512
	      && ( (inChannelGroup % 4096) < 32 || (inChannelGroup % 4096) > 4096-32 ) )
	  {
	    func_odd_filternchw_avoid_l1m<3, ADDBIAS>(pIn, pKernel, pBias, pOut,
	       inChannel, inWidth, inHeight,
	       outChannel, outWidth, outHeight,
	       inChannelGroup, outChannelGroup,
	       inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	       n, k );
	  }
	  else
	  {
	    func_odd<FLAYOUT, 3, ADDBIAS>(pIn, pKernel, pBias, pOut,
	       inChannel, inWidth, inHeight,
	       outChannel, outWidth, outHeight,
	       inChannelGroup, outChannelGroup,
	       inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	       n, k );
	  }
	  k+=3 ;
	  break ;
	case 4 :
	  if( FLAYOUT == VEDNN_FILTER_LAYOUT_NCHW
	      && inChannelGroup >= 512
	      && ( (inChannelGroup % 4096) < 32 || (inChannelGroup % 4096) > 4096-32 ) )
	  {
	    func_even_filternchw_avoid_l1m<4, ADDBIAS>(pIn, pKernel, pBias, pOut,
	       inChannel, inWidth, inHeight,
	       outChannel, outWidth, outHeight,
	       inChannelGroup, outChannelGroup,
	       inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	       n, k );
	  }
	  else
	  {
	    func_even<FLAYOUT, 4, ADDBIAS>(pIn, pKernel, pBias, pOut,
	       inChannel, inWidth, inHeight,
	       outChannel, outWidth, outHeight,
	       inChannelGroup, outChannelGroup,
	       inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	       n, k );
	  }
	  k+=4 ;
	  break ;
	case 5 :
	  if( FLAYOUT == VEDNN_FILTER_LAYOUT_NCHW
	      && inChannelGroup >= 512
	      && ( (inChannelGroup % 2048) < 16 || (inChannelGroup % 2048) > 2048-16 ) )
	  {
	    func_odd_filternchw_avoid_l1m<5, ADDBIAS>(pIn, pKernel, pBias, pOut,
	       inChannel, inWidth, inHeight,
	       outChannel, outWidth, outHeight,
	       inChannelGroup, outChannelGroup,
	       inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	       n, k );
	  }
	  else
	  {
	    func_odd<FLAYOUT, 5, ADDBIAS>(pIn, pKernel, pBias, pOut,
	       inChannel, inWidth, inHeight,
	       outChannel, outWidth, outHeight,
	       inChannelGroup, outChannelGroup,
	       inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	       n, k );
	  }
	  k+=5 ;
	  break ;
	case 6 :
	  if( FLAYOUT == VEDNN_FILTER_LAYOUT_NCHW
	      && inChannelGroup >= 512
	      && ( (inChannelGroup % 2048) < 16 || (inChannelGroup % 2048) > 2048-16 ) )
	  {
	    func_even_filternchw_avoid_l1m<6, ADDBIAS>(pIn, pKernel, pBias, pOut,
	       inChannel, inWidth, inHeight,
	       outChannel, outWidth, outHeight,
	       inChannelGroup, outChannelGroup,
	       inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	       n, k );
	  }
	  else
	  {
	    func_even<FLAYOUT, 6, ADDBIAS>(pIn, pKernel, pBias, pOut,
	       inChannel, inWidth, inHeight,
	       outChannel, outWidth, outHeight,
	       inChannelGroup, outChannelGroup,
	       inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	       n, k );
	  }
	  k+=6 ;
	  break ;
	case 7 :
	  if( FLAYOUT == VEDNN_FILTER_LAYOUT_NCHW
	      && inChannelGroup >= 512
	      && ( (inChannelGroup % 2048) < 16 || (inChannelGroup % 2048) > 2048-16 ) )
	  {
	    func_odd_filternchw_avoid_l1m<7, ADDBIAS>(pIn, pKernel, pBias, pOut,
	       inChannel, inWidth, inHeight,
	       outChannel, outWidth, outHeight,
	       inChannelGroup, outChannelGroup,
	       inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	       n, k );
	  }
	  else
	  {
	    func_odd<FLAYOUT, 7, ADDBIAS>(pIn, pKernel, pBias, pOut,
	       inChannel, inWidth, inHeight,
	       outChannel, outWidth, outHeight,
	       inChannelGroup, outChannelGroup,
	       inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	       n, k );
	  }
	  k+=7 ;
	  break ;
	case 8 :
	  if( FLAYOUT == VEDNN_FILTER_LAYOUT_NCHW
	      && inChannelGroup >= 512
	      && ( (inChannelGroup % 2048) < 16 || (inChannelGroup % 2048) > 2048-16 ) )
	  {
	    func_even_filternchw_avoid_l1m<8, ADDBIAS>(pIn, pKernel, pBias, pOut,
	       inChannel, inWidth, inHeight,
	       outChannel, outWidth, outHeight,
	       inChannelGroup, outChannelGroup,
	       inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	       n, k );
	  }
	  else
	  {
	    func_even<FLAYOUT, 8, ADDBIAS>(pIn, pKernel, pBias, pOut,
	       inChannel, inWidth, inHeight,
	       outChannel, outWidth, outHeight,
	       inChannelGroup, outChannelGroup,
	       inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	       n, k );
	  }
	  k+=8 ;
	  break ;
	case 9 :
	  if( FLAYOUT == VEDNN_FILTER_LAYOUT_NCHW
	      && inChannelGroup >= 512
	      && ( (inChannelGroup % 1024) < 8 || (inChannelGroup % 1024) > 1024-8 ) )
	  {
	    func_odd_filternchw_avoid_l1m<9, ADDBIAS>(pIn, pKernel, pBias, pOut,
	       inChannel, inWidth, inHeight,
	       outChannel, outWidth, outHeight,
	       inChannelGroup, outChannelGroup,
	       inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	       n, k );
	  }
	  else
	  {
	    func_odd<FLAYOUT, 9, ADDBIAS>(pIn, pKernel, pBias, pOut,
	       inChannel, inWidth, inHeight,
	       outChannel, outWidth, outHeight,
	       inChannelGroup, outChannelGroup,
	       inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	       n, k );
	  }
	  k+=9 ;
	  break ;
	case 10 :
	  if( FLAYOUT == VEDNN_FILTER_LAYOUT_NCHW
	      && inChannelGroup >= 512
	      && ( (inChannelGroup % 1024) < 8 || (inChannelGroup % 1024) > 1024-8 ) )
	  {
	    func_even_filternchw_avoid_l1m<10, ADDBIAS>(pIn, pKernel, pBias, pOut,
	       inChannel, inWidth, inHeight,
	       outChannel, outWidth, outHeight,
	       inChannelGroup, outChannelGroup,
	       inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	       n, k );
	  }
	  else
	  {
	    func_even<FLAYOUT, 10, ADDBIAS>(pIn, pKernel, pBias, pOut,
	       inChannel, inWidth, inHeight,
	       outChannel, outWidth, outHeight,
	       inChannelGroup, outChannelGroup,
	       inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	       n, k );
	  }
	  k+=10 ;
	  break ;
	case 11 :
	  if( FLAYOUT == VEDNN_FILTER_LAYOUT_NCHW
	      && inChannelGroup >= 512
	      && ( (inChannelGroup % 1024) < 8 || (inChannelGroup % 1024) > 1024-8 ) )
	  {
	    func_odd_filternchw_avoid_l1m<11, ADDBIAS>(pIn, pKernel, pBias, pOut,
	       inChannel, inWidth, inHeight,
	       outChannel, outWidth, outHeight,
	       inChannelGroup, outChannelGroup,
	       inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	       n, k );
	  }
	  else
	  {
	    func_odd<FLAYOUT, 11, ADDBIAS>(pIn, pKernel, pBias, pOut,
	       inChannel, inWidth, inHeight,
	       outChannel, outWidth, outHeight,
	       inChannelGroup, outChannelGroup,
	       inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	       n, k );
	  }
	  k+=11 ;
	  break ;
	case 12 :
	  if( FLAYOUT == VEDNN_FILTER_LAYOUT_NCHW
	      && inChannelGroup >= 512
	      && ( (inChannelGroup % 1024) < 8 || (inChannelGroup % 1024) > 1024-8 ) )
	  {
	    func_even_filternchw_avoid_l1m<12, ADDBIAS>(pIn, pKernel, pBias, pOut,
	       inChannel, inWidth, inHeight,
	       outChannel, outWidth, outHeight,
	       inChannelGroup, outChannelGroup,
	       inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	       n, k );
	  }
	  else
	  {
	    func_even<FLAYOUT, 12, ADDBIAS>(pIn, pKernel, pBias, pOut,
	       inChannel, inWidth, inHeight,
	       outChannel, outWidth, outHeight,
	       inChannelGroup, outChannelGroup,
	       inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	       n, k );
	  }
	  k+=12 ;
	  break ;
	case 13 :
	  if( FLAYOUT == VEDNN_FILTER_LAYOUT_NCHW
	      && inChannelGroup >= 512
	      && ( (inChannelGroup % 1024) < 8 || (inChannelGroup % 1024) > 1024-8 ) )
	  {
	    func_odd_filternchw_avoid_l1m<13, ADDBIAS>(pIn, pKernel, pBias, pOut,
	       inChannel, inWidth, inHeight,
	       outChannel, outWidth, outHeight,
	       inChannelGroup, outChannelGroup,
	       inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	       n, k );
	  }
	  else
	  {
	    func_odd<FLAYOUT, 13, ADDBIAS>(pIn, pKernel, pBias, pOut,
	       inChannel, inWidth, inHeight,
	       outChannel, outWidth, outHeight,
	       inChannelGroup, outChannelGroup,
	       inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	       n, k );
	  }
	  k+=13 ;
	  break ;
	case 14 :
	  if( FLAYOUT == VEDNN_FILTER_LAYOUT_NCHW
	      && inChannelGroup >= 512
	      && ( (inChannelGroup % 1024) < 8 || (inChannelGroup % 1024) > 1024-8 ) )
	  {
	    func_even_filternchw_avoid_l1m<14, ADDBIAS>(pIn, pKernel, pBias, pOut,
	       inChannel, inWidth, inHeight,
	       outChannel, outWidth, outHeight,
	       inChannelGroup, outChannelGroup,
	       inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	       n, k );
	  }
	  else
	  {
	    func_even<FLAYOUT, 14, ADDBIAS>(pIn, pKernel, pBias, pOut,
	       inChannel, inWidth, inHeight,
	       outChannel, outWidth, outHeight,
	       inChannelGroup, outChannelGroup,
	       inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	       n, k );
	  }
	  k+=14 ;
	  break ;
	case 15 :
	  if( FLAYOUT == VEDNN_FILTER_LAYOUT_NCHW
	      && inChannelGroup >= 512
	      && ( (inChannelGroup % 1024) < 8 || (inChannelGroup % 1024) > 1024-8 ) )
	  {
	    func_odd_filternchw_avoid_l1m<15, ADDBIAS>(pIn, pKernel, pBias, pOut,
	       inChannel, inWidth, inHeight,
	       outChannel, outWidth, outHeight,
	       inChannelGroup, outChannelGroup,
	       inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	       n, k );
	  }
	  else
	  {
	    func_odd<FLAYOUT, 15, ADDBIAS>(pIn, pKernel, pBias, pOut,
	       inChannel, inWidth, inHeight,
	       outChannel, outWidth, outHeight,
	       inChannelGroup, outChannelGroup,
	       inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	       n, k );
	  }
	  k+=15 ;
	  break ;
	default :
	  break ;
	}
	for (; k < outChannelGroup; k+=16) {
	  if( FLAYOUT == VEDNN_FILTER_LAYOUT_NCHW
	      && inChannelGroup >= 512
	      && ( (inChannelGroup % 1024) < 8 || (inChannelGroup % 1024) > 1024-8 ) )
	  {
	    func_even_filternchw_avoid_l1m<16, ADDBIAS>(pIn, pKernel, pBias, pOut,
	       inChannel, inWidth, inHeight,
	       outChannel, outWidth, outHeight,
	       inChannelGroup, outChannelGroup,
	       inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	       n, k );
	  }
	  else
	  {
	    func_even<FLAYOUT, 16, ADDBIAS>(pIn, pKernel, pBias, pOut,
	       inChannel, inWidth, inHeight,
	       outChannel, outWidth, outHeight,
	       inChannelGroup, outChannelGroup,
	       inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	       n, k );
	  }
	} // outChannel
    } // group
  } // batch
}

extern "C" vednnError_t
vednnConvolutionForward_direct_dil1_str1_pad0_ker1(
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
//  const int64_t kernWidth  = pParamKernel->width;		/* must be 1 */
//  const int64_t kernHeight = pParamKernel->height;		/* must be 1 */

  const int64_t filter_layout = pParamKernel->layout ;

  const int64_t group          = pParamConv->group;
//  const int64_t strideWidth    = pParamConv->strideWidth;	/* must be 1 */
//  const int64_t strideHeight   = pParamConv->strideHeight;	/* must be 1 */
//  const int64_t padWidth       = pParamConv->padWidth;	/* must be 0 */
//  const int64_t padHeight      = pParamConv->padHeight;	/* must be 0 */
//  const int64_t dilationWidth  = pParamConv->dilationWidth;	/* must be 1 */
//  const int64_t dilationHeight = pParamConv->dilationHeight;	/* must be 1 */

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
		 inChannelGroup, outChannelGroup ) ;
    }
    else {
      convloop<VEDNN_FILTER_LAYOUT_NCHW, true>(pIn, pKernel, pBias, pOut,
		 batch, group,
		 inChannel, inWidth, inHeight,
		 outChannel, outWidth, outHeight,
		 inChannelGroup, outChannelGroup ) ;
    }
  }
  else {
    if( pDataBias == NULL ) {
      convloop<VEDNN_FILTER_LAYOUT_HWCN, false>(pIn, pKernel, pBias, pOut,
		 batch, group,
		 inChannel, inWidth, inHeight,
		 outChannel, outWidth, outHeight,
		 inChannelGroup, outChannelGroup ) ;
    }
    else {
      convloop<VEDNN_FILTER_LAYOUT_HWCN, true>(pIn, pKernel, pBias, pOut,
		 batch, group,
		 inChannel, inWidth, inHeight,
		 outChannel, outWidth, outHeight,
		 inChannelGroup, outChannelGroup ) ;
    }
  }

  return VEDNN_SUCCESS;
}
