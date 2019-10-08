#include <stdint.h>
#include <stdlib.h>

#include "vednn.h"

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
  const int64_t kernWidth,
  const int64_t kernHeight,
  const int64_t inChannelGroup,
  const int64_t outChannelGroup,
  const int64_t strideHeight,
  const int64_t strideWidth,
  const int64_t inGroupOffset,
  const int64_t outGroupOffset,
  const int64_t biasGroupOffset,
  const int64_t kernGroupOffset,
  const int64_t nY,
  const __vr    vrij,
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

  for (int64_t y=0; y<outHeight; y+=nY)
  {
    const int64_t vl = outWidth * (outHeight - y < nY ? outHeight - y : nY) ;
    const int64_t op = y * outWidth ;

    __vr vrsum0  = _vel_vbrds_vsl(bias0, vl) ;
    __vr vrsum12 = _vel_pvbrd_vsl(bias12, vl) ;
    __vr vrsum34 = _vel_pvbrd_vsl(bias34, vl) ;
    __vr vrsum56 = _vel_pvbrd_vsl(bias56, vl) ;
    __vr vrsum78 = _vel_pvbrd_vsl(bias78, vl) ;
    __vr vrsum9A = _vel_pvbrd_vsl(bias9A, vl) ;
    __vr vrsumBC = _vel_pvbrd_vsl(biasBC, vl) ;
    __vr vrsumDE = _vel_pvbrd_vsl(biasDE, vl) ;

    int c = 0 ;
    if ( (inChannelGroup & 0x01) == 1 ) {
      const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;
      __vr vrpin_c0 = _vel_vsfa_vvssl(vrij, 2, (uint64_t)(pInChannel+y*strideHeight*inWidth), vl) ;

      __vr vrin_c0 = _vel_vgtu_vvssl(vrpin_c0, 0, 0, vl) ;

#define FILTER_OFFSET(K,C) ( kernGroupOffset + filter_index<FLAYOUT>(K,C,0,0, inChannelGroup, outChannelGroup, 1, 1) )
#define VFMAD(VRIN, C)										\
      {												\
	__vr vrinP = _vel_vshf_vvvsl(VRIN, VRIN, VE_VSHUFFLE_YUZU, vl) ;			\
	const float    kerValue0  = pKernel[FILTER_OFFSET(k+0,(C))] ;				\
	const uint64_t kerValue12 = _vel_pack_f32p(pKernel + FILTER_OFFSET(k+ 1,(C)),		\
						   pKernel + FILTER_OFFSET(k+ 2,(C))) ;		\
	const uint64_t kerValue34 = _vel_pack_f32p(pKernel + FILTER_OFFSET(k+ 3,(C)),		\
						   pKernel + FILTER_OFFSET(k+ 4,(C))) ;		\
	const uint64_t kerValue56 = _vel_pack_f32p(pKernel + FILTER_OFFSET(k+ 5,(C)),		\
						   pKernel + FILTER_OFFSET(k+ 6,(C))) ;		\
	const uint64_t kerValue78 = _vel_pack_f32p(pKernel + FILTER_OFFSET(k+ 7,(C)),		\
						   pKernel + FILTER_OFFSET(k+ 8,(C))) ;		\
	const uint64_t kerValue9A = _vel_pack_f32p(pKernel + FILTER_OFFSET(k+ 9,(C)),		\
						   pKernel + FILTER_OFFSET(k+10,(C))) ;		\
	const uint64_t kerValueBC = _vel_pack_f32p(pKernel + FILTER_OFFSET(k+11,(C)),		\
						   pKernel + FILTER_OFFSET(k+12,(C))) ;		\
	const uint64_t kerValueDE = _vel_pack_f32p(pKernel + FILTER_OFFSET(k+13,(C)),		\
						   pKernel + FILTER_OFFSET(k+14,(C))) ;		\
	vrsum0 = _vel_vfmads_vvsvl(vrsum0, kerValue0, VRIN, vl) ;				\
	if(NUMKERNEL>= 3) vrsum12 = _vel_pvfmad_vvsvl(vrsum12, kerValue12, vrinP, vl) ;		\
	if(NUMKERNEL>= 5) vrsum34 = _vel_pvfmad_vvsvl(vrsum34, kerValue34, vrinP, vl) ;		\
	if(NUMKERNEL>= 7) vrsum56 = _vel_pvfmad_vvsvl(vrsum56, kerValue56, vrinP, vl) ;		\
	if(NUMKERNEL>= 9) vrsum78 = _vel_pvfmad_vvsvl(vrsum78, kerValue78, vrinP, vl) ;		\
	if(NUMKERNEL>=11) vrsum9A = _vel_pvfmad_vvsvl(vrsum9A, kerValue9A, vrinP, vl) ;		\
	if(NUMKERNEL>=13) vrsumBC = _vel_pvfmad_vvsvl(vrsumBC, kerValueBC, vrinP, vl) ;		\
	if(NUMKERNEL>=15) vrsumDE = _vel_pvfmad_vvsvl(vrsumDE, kerValueDE, vrinP, vl) ;		\
      }

      VFMAD(vrin_c0, c+0) ;

      c+=1 ;
    }
    if ( ((inChannelGroup >> 1) & 0x01) == 1 ) {
      const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;
      __vr vrpin_c0 = _vel_vsfa_vvssl(vrij, 2, (uint64_t)(pInChannel+y*strideHeight*inWidth), vl) ;
      __vr vrpin_c1 = _vel_vaddul_vsvl(  4*inHeight*inWidth,vrpin_c0, vl) ;

      __vr vrin_c0 = _vel_vgtu_vvssl(vrpin_c0, 0, 0, vl) ;
      __vr vrin_c1 = _vel_vgtu_vvssl(vrpin_c1, 0, 0, vl) ;

      VFMAD(vrin_c0, c+0) ;
      VFMAD(vrin_c1, c+1) ;


      c+=2 ;
    } // inChannel
    if ( ((inChannelGroup >> 2) & 0x01) == 1 ) {
      const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;
      __vr vrpin_c0 = _vel_vsfa_vvssl(vrij, 2, (uint64_t)(pInChannel+y*strideHeight*inWidth), vl) ;
      __vr vrpin_c1 = _vel_vaddul_vsvl(  4*inHeight*inWidth,vrpin_c0, vl) ;
      __vr vrpin_c2 = _vel_vaddul_vsvl(2*4*inHeight*inWidth,vrpin_c0, vl) ;
      __vr vrpin_c3 = _vel_vaddul_vsvl(3*4*inHeight*inWidth,vrpin_c0, vl) ;

      __vr vrin_c0 = _vel_vgtu_vvssl(vrpin_c0, 0, 0, vl) ;
      __vr vrin_c1 = _vel_vgtu_vvssl(vrpin_c1, 0, 0, vl) ;
      __vr vrin_c2 = _vel_vgtu_vvssl(vrpin_c2, 0, 0, vl) ;
      __vr vrin_c3 = _vel_vgtu_vvssl(vrpin_c3, 0, 0, vl) ;

      VFMAD(vrin_c0, c+0) ;
      VFMAD(vrin_c1, c+1) ;
      VFMAD(vrin_c2, c+2) ;
      VFMAD(vrin_c3, c+3) ;

      c+=4 ;
    }
    for (; c < inChannelGroup; c+=8) {
      const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;
      __vr vrpin_c0 = _vel_vsfa_vvssl(vrij, 2, (uint64_t)(pInChannel+y*strideHeight*inWidth), vl) ;
      __vr vrpin_c1 = _vel_vaddul_vsvl(  4*inHeight*inWidth,vrpin_c0, vl) ;
      __vr vrpin_c2 = _vel_vaddul_vsvl(2*4*inHeight*inWidth,vrpin_c0, vl) ;
      __vr vrpin_c3 = _vel_vaddul_vsvl(3*4*inHeight*inWidth,vrpin_c0, vl) ;
      __vr vrpin_c4 = _vel_vaddul_vsvl(4*4*inHeight*inWidth,vrpin_c0, vl) ;
      __vr vrpin_c5 = _vel_vaddul_vsvl(5*4*inHeight*inWidth,vrpin_c0, vl) ;
      __vr vrpin_c6 = _vel_vaddul_vsvl(6*4*inHeight*inWidth,vrpin_c0, vl) ;
      __vr vrpin_c7 = _vel_vaddul_vsvl(7*4*inHeight*inWidth,vrpin_c0, vl) ;

      __vr vrin_c0 = _vel_vgtu_vvssl(vrpin_c0, 0, 0, vl) ;
      __vr vrin_c1 = _vel_vgtu_vvssl(vrpin_c1, 0, 0, vl) ;
      __vr vrin_c2 = _vel_vgtu_vvssl(vrpin_c2, 0, 0, vl) ;
      __vr vrin_c3 = _vel_vgtu_vvssl(vrpin_c3, 0, 0, vl) ;
      __vr vrin_c4 = _vel_vgtu_vvssl(vrpin_c4, 0, 0, vl) ;
      __vr vrin_c5 = _vel_vgtu_vvssl(vrpin_c5, 0, 0, vl) ;
      __vr vrin_c6 = _vel_vgtu_vvssl(vrpin_c6, 0, 0, vl) ;
      __vr vrin_c7 = _vel_vgtu_vvssl(vrpin_c7, 0, 0, vl) ;

      VFMAD(vrin_c0, c+0) ;
      VFMAD(vrin_c1, c+1) ;
      VFMAD(vrin_c2, c+2) ;
      VFMAD(vrin_c3, c+3) ;
      VFMAD(vrin_c4, c+4) ;
      VFMAD(vrin_c5, c+5) ;
      VFMAD(vrin_c6, c+6) ;
      VFMAD(vrin_c7, c+7) ;

#undef VFMAD
#undef FILTER_OFFSET
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
  const int64_t kernWidth,
  const int64_t kernHeight,
  const int64_t inChannelGroup,
  const int64_t outChannelGroup,
  const int64_t strideHeight,
  const int64_t strideWidth,
  const int64_t inGroupOffset,
  const int64_t outGroupOffset,
  const int64_t biasGroupOffset,
  const int64_t kernGroupOffset,
  const int64_t nY,
  const __vr    vrij,
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

  for (int64_t y=0; y<outHeight; y+=nY)
  {
    const int64_t vl = outWidth * (outHeight - y < nY ? outHeight - y : nY) ;
    const int64_t op = y * outWidth ;

    __vr vrsum01 = _vel_pvbrd_vsl(bias01, vl) ;
    __vr vrsum23 = _vel_pvbrd_vsl(bias23, vl) ;
    __vr vrsum45 = _vel_pvbrd_vsl(bias45, vl) ;
    __vr vrsum67 = _vel_pvbrd_vsl(bias67, vl) ;
    __vr vrsum89 = _vel_pvbrd_vsl(bias89, vl) ;
    __vr vrsumAB = _vel_pvbrd_vsl(biasAB, vl) ;
    __vr vrsumCD = _vel_pvbrd_vsl(biasCD, vl) ;
    __vr vrsumEF = _vel_pvbrd_vsl(biasEF, vl) ;

    int c = 0 ;
    if ( (inChannelGroup & 0x01) == 1 ) {
      const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;
      __vr vrpin_c0 = _vel_vsfa_vvssl(vrij, 2, (uint64_t)(pInChannel+y*strideHeight*inWidth), vl) ;

      __vr vrin_c0 = _vel_vgtu_vvssl(vrpin_c0, 0, 0, vl) ;

#define FILTER_OFFSET(K,C) ( kernGroupOffset + filter_index<FLAYOUT>(K,C,0,0, inChannelGroup, outChannelGroup, 1, 1) )
#define VFMAD(VRIN, C)									\
      {												\
	__vr vrinP = _vel_vshf_vvvsl(VRIN, VRIN, VE_VSHUFFLE_YUZU, vl) ;			\
	const uint64_t kerValue01 = _vel_pack_f32p(pKernel + FILTER_OFFSET(k+ 0,(C)),		\
						   pKernel + FILTER_OFFSET(k+ 1,(C))) ;		\
	const uint64_t kerValue23 = _vel_pack_f32p(pKernel + FILTER_OFFSET(k+ 2,(C)),		\
						   pKernel + FILTER_OFFSET(k+ 3,(C))) ;		\
	const uint64_t kerValue45 = _vel_pack_f32p(pKernel + FILTER_OFFSET(k+ 4,(C)),		\
						   pKernel + FILTER_OFFSET(k+ 5,(C))) ;		\
	const uint64_t kerValue67 = _vel_pack_f32p(pKernel + FILTER_OFFSET(k+ 6,(C)),		\
						   pKernel + FILTER_OFFSET(k+ 7,(C))) ;		\
	const uint64_t kerValue89 = _vel_pack_f32p(pKernel + FILTER_OFFSET(k+ 8,(C)),		\
						   pKernel + FILTER_OFFSET(k+ 9,(C))) ;		\
	const uint64_t kerValueAB = _vel_pack_f32p(pKernel + FILTER_OFFSET(k+10,(C)),		\
						   pKernel + FILTER_OFFSET(k+11,(C))) ;		\
	const uint64_t kerValueCD = _vel_pack_f32p(pKernel + FILTER_OFFSET(k+12,(C)),		\
						   pKernel + FILTER_OFFSET(k+13,(C))) ;		\
	const uint64_t kerValueEF = _vel_pack_f32p(pKernel + FILTER_OFFSET(k+14,(C)),		\
						   pKernel + FILTER_OFFSET(k+15,(C))) ;		\
        if(NUMKERNEL>= 2) vrsum01 = _vel_pvfmad_vvsvl(vrsum01, kerValue01, vrinP, vl) ;		\
        if(NUMKERNEL>= 4) vrsum23 = _vel_pvfmad_vvsvl(vrsum23, kerValue23, vrinP, vl) ;		\
        if(NUMKERNEL>= 6) vrsum45 = _vel_pvfmad_vvsvl(vrsum45, kerValue45, vrinP, vl) ;		\
        if(NUMKERNEL>= 8) vrsum67 = _vel_pvfmad_vvsvl(vrsum67, kerValue67, vrinP, vl) ;		\
        if(NUMKERNEL>=10) vrsum89 = _vel_pvfmad_vvsvl(vrsum89, kerValue89, vrinP, vl) ;		\
        if(NUMKERNEL>=12) vrsumAB = _vel_pvfmad_vvsvl(vrsumAB, kerValueAB, vrinP, vl) ;		\
        if(NUMKERNEL>=14) vrsumCD = _vel_pvfmad_vvsvl(vrsumCD, kerValueCD, vrinP, vl) ;		\
        if(NUMKERNEL>=16) vrsumEF = _vel_pvfmad_vvsvl(vrsumEF, kerValueEF, vrinP, vl) ;		\
      }

      VFMAD(vrin_c0, c+0) ;

      c+=1 ;
    }
    if ( ((inChannelGroup >> 1) & 0x01) == 1 ) {
      const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;
      __vr vrpin_c0 = _vel_vsfa_vvssl(vrij, 2, (uint64_t)(pInChannel+y*strideHeight*inWidth), vl) ;
      __vr vrpin_c1 = _vel_vaddul_vsvl(  4*inHeight*inWidth,vrpin_c0, vl) ;

      __vr vrin_c0 = _vel_vgtu_vvssl(vrpin_c0, 0, 0, vl) ;
      __vr vrin_c1 = _vel_vgtu_vvssl(vrpin_c1, 0, 0, vl) ;

      VFMAD(vrin_c0, c+0) ;
      VFMAD(vrin_c1, c+1) ;


      c+=2 ;
    } // inChannel
    if ( ((inChannelGroup >> 2) & 0x01) == 1 ) {
      const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;
      __vr vrpin_c0 = _vel_vsfa_vvssl(vrij, 2, (uint64_t)(pInChannel+y*strideHeight*inWidth), vl) ;
      __vr vrpin_c1 = _vel_vaddul_vsvl(  4*inHeight*inWidth,vrpin_c0, vl) ;
      __vr vrpin_c2 = _vel_vaddul_vsvl(2*4*inHeight*inWidth,vrpin_c0, vl) ;
      __vr vrpin_c3 = _vel_vaddul_vsvl(3*4*inHeight*inWidth,vrpin_c0, vl) ;

      __vr vrin_c0 = _vel_vgtu_vvssl(vrpin_c0, 0, 0, vl) ;
      __vr vrin_c1 = _vel_vgtu_vvssl(vrpin_c1, 0, 0, vl) ;
      __vr vrin_c2 = _vel_vgtu_vvssl(vrpin_c2, 0, 0, vl) ;
      __vr vrin_c3 = _vel_vgtu_vvssl(vrpin_c3, 0, 0, vl) ;

      VFMAD(vrin_c0, c+0) ;
      VFMAD(vrin_c1, c+1) ;
      VFMAD(vrin_c2, c+2) ;
      VFMAD(vrin_c3, c+3) ;

      c+=4 ;
    }
    for (; c < inChannelGroup; c+=8) {
      const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;
      __vr vrpin_c0 = _vel_vsfa_vvssl(vrij, 2, (uint64_t)(pInChannel+y*strideHeight*inWidth), vl) ;
      __vr vrpin_c1 = _vel_vaddul_vsvl(  4*inHeight*inWidth,vrpin_c0, vl) ;
      __vr vrpin_c2 = _vel_vaddul_vsvl(2*4*inHeight*inWidth,vrpin_c0, vl) ;
      __vr vrpin_c3 = _vel_vaddul_vsvl(3*4*inHeight*inWidth,vrpin_c0, vl) ;
      __vr vrpin_c4 = _vel_vaddul_vsvl(4*4*inHeight*inWidth,vrpin_c0, vl) ;
      __vr vrpin_c5 = _vel_vaddul_vsvl(5*4*inHeight*inWidth,vrpin_c0, vl) ;
      __vr vrpin_c6 = _vel_vaddul_vsvl(6*4*inHeight*inWidth,vrpin_c0, vl) ;
      __vr vrpin_c7 = _vel_vaddul_vsvl(7*4*inHeight*inWidth,vrpin_c0, vl) ;

      __vr vrin_c0 = _vel_vgtu_vvssl(vrpin_c0, 0, 0, vl) ;
      __vr vrin_c1 = _vel_vgtu_vvssl(vrpin_c1, 0, 0, vl) ;
      __vr vrin_c2 = _vel_vgtu_vvssl(vrpin_c2, 0, 0, vl) ;
      __vr vrin_c3 = _vel_vgtu_vvssl(vrpin_c3, 0, 0, vl) ;
      __vr vrin_c4 = _vel_vgtu_vvssl(vrpin_c4, 0, 0, vl) ;
      __vr vrin_c5 = _vel_vgtu_vvssl(vrpin_c5, 0, 0, vl) ;
      __vr vrin_c6 = _vel_vgtu_vvssl(vrpin_c6, 0, 0, vl) ;
      __vr vrin_c7 = _vel_vgtu_vvssl(vrpin_c7, 0, 0, vl) ;

      VFMAD(vrin_c0, c+0) ;
      VFMAD(vrin_c1, c+1) ;
      VFMAD(vrin_c2, c+2) ;
      VFMAD(vrin_c3, c+3) ;
      VFMAD(vrin_c4, c+4) ;
      VFMAD(vrin_c5, c+5) ;
      VFMAD(vrin_c6, c+6) ;
      VFMAD(vrin_c7, c+7) ;

#undef VFMAD
#undef FILTER_OFFSET
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

template<bool ADDBIAS>
static inline void k16_filter_nchw_c1024x(
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
    const int64_t kernWidth,
    const int64_t kernHeight,
    const int64_t inChannelGroup,
    const int64_t outChannelGroup,
    const int64_t strideHeight,
    const int64_t strideWidth,
    const int64_t inGroupOffset,
    const int64_t outGroupOffset,
    const int64_t biasGroupOffset,
    const int64_t kernGroupOffset,
    const int64_t nY,
    const __vr    vrij,
    const int64_t n,
    const int64_t k
)
{
  float __attribute__ ((aligned(8))) filter[16*512] ;

  int64_t outIndex = outGroupOffset + (n * outChannel + k  ) * outHeight * outWidth ;

  const int64_t bias01 = ADDBIAS ?  _vel_pack_f32p(pBias+biasGroupOffset+k+ 0, pBias+biasGroupOffset+k+ 1) : 0UL ;
  const int64_t bias23 = ADDBIAS ?  _vel_pack_f32p(pBias+biasGroupOffset+k+ 2, pBias+biasGroupOffset+k+ 3) : 0UL ;
  const int64_t bias45 = ADDBIAS ?  _vel_pack_f32p(pBias+biasGroupOffset+k+ 4, pBias+biasGroupOffset+k+ 5) : 0UL ;
  const int64_t bias67 = ADDBIAS ?  _vel_pack_f32p(pBias+biasGroupOffset+k+ 6, pBias+biasGroupOffset+k+ 7) : 0UL ;
  const int64_t bias89 = ADDBIAS ?  _vel_pack_f32p(pBias+biasGroupOffset+k+ 8, pBias+biasGroupOffset+k+ 9) : 0UL ;
  const int64_t biasAB = ADDBIAS ?  _vel_pack_f32p(pBias+biasGroupOffset+k+10, pBias+biasGroupOffset+k+11) : 0UL ;
  const int64_t biasCD = ADDBIAS ?  _vel_pack_f32p(pBias+biasGroupOffset+k+12, pBias+biasGroupOffset+k+13) : 0UL ;
  const int64_t biasEF = ADDBIAS ?  _vel_pack_f32p(pBias+biasGroupOffset+k+14, pBias+biasGroupOffset+k+15) : 0UL ;

  for (int64_t y=0; y<outHeight; y+=nY)
  {
    const int64_t vl = outWidth * (outHeight - y < nY ? outHeight - y : nY) ;
    const int64_t op = y * outWidth ;

    __vr vrsum01 = _vel_pvbrd_vsl(bias01, vl) ;
    __vr vrsum23 = _vel_pvbrd_vsl(bias23, vl) ;
    __vr vrsum45 = _vel_pvbrd_vsl(bias45, vl) ;
    __vr vrsum67 = _vel_pvbrd_vsl(bias67, vl) ;
    __vr vrsum89 = _vel_pvbrd_vsl(bias89, vl) ;
    __vr vrsumAB = _vel_pvbrd_vsl(biasAB, vl) ;
    __vr vrsumCD = _vel_pvbrd_vsl(biasCD, vl) ;
    __vr vrsumEF = _vel_pvbrd_vsl(biasEF, vl) ;


    for(int64_t c0=0; c0<inChannelGroup; c0+=512) {
      const float *pKerValue = pKernel + kernGroupOffset + (k * inChannelGroup + c0) ;

      __vr vr0 = _vel_vld_vssl(8, pKerValue+ 0*inChannelGroup, 256) ;
      __vr vr1 = _vel_vld_vssl(8, pKerValue+ 1*inChannelGroup, 256) ;
      __vr vr2 = _vel_vld_vssl(8, pKerValue+ 2*inChannelGroup, 256) ;
      __vr vr3 = _vel_vld_vssl(8, pKerValue+ 3*inChannelGroup, 256) ;
      __vr vr4 = _vel_vld_vssl(8, pKerValue+ 4*inChannelGroup, 256) ;
      __vr vr5 = _vel_vld_vssl(8, pKerValue+ 5*inChannelGroup, 256) ;
      __vr vr6 = _vel_vld_vssl(8, pKerValue+ 6*inChannelGroup, 256) ;
      __vr vr7 = _vel_vld_vssl(8, pKerValue+ 7*inChannelGroup, 256) ;
      __vr vr8 = _vel_vld_vssl(8, pKerValue+ 8*inChannelGroup, 256) ;
      __vr vr9 = _vel_vld_vssl(8, pKerValue+ 9*inChannelGroup, 256) ;
      __vr vrA = _vel_vld_vssl(8, pKerValue+10*inChannelGroup, 256) ;
      __vr vrB = _vel_vld_vssl(8, pKerValue+11*inChannelGroup, 256) ;
      __vr vrC = _vel_vld_vssl(8, pKerValue+12*inChannelGroup, 256) ;
      __vr vrD = _vel_vld_vssl(8, pKerValue+13*inChannelGroup, 256) ;
      __vr vrE = _vel_vld_vssl(8, pKerValue+14*inChannelGroup, 256) ;
      __vr vrF = _vel_vld_vssl(8, pKerValue+15*inChannelGroup, 256) ;

      __vr vr01_c0 = _vel_vshf_vvvsl(vr0,vr1,VE_VSHUFFLE_YLZL, 256) ;
      __vr vr23_c0 = _vel_vshf_vvvsl(vr2,vr3,VE_VSHUFFLE_YLZL, 256) ;
      __vr vr45_c0 = _vel_vshf_vvvsl(vr4,vr5,VE_VSHUFFLE_YLZL, 256) ;
      __vr vr67_c0 = _vel_vshf_vvvsl(vr6,vr7,VE_VSHUFFLE_YLZL, 256) ;
      __vr vr89_c0 = _vel_vshf_vvvsl(vr8,vr9,VE_VSHUFFLE_YLZL, 256) ;
      __vr vrAB_c0 = _vel_vshf_vvvsl(vrA,vrB,VE_VSHUFFLE_YLZL, 256) ;
      __vr vrCD_c0 = _vel_vshf_vvvsl(vrC,vrD,VE_VSHUFFLE_YLZL, 256) ;
      __vr vrEF_c0 = _vel_vshf_vvvsl(vrE,vrF,VE_VSHUFFLE_YLZL, 256) ;

      _vel_vst_vssl(vr01_c0, 8, filter, 256) ;
      _vel_vst_vssl(vr23_c0, 8, filter+1*512, 256) ;
      _vel_vst_vssl(vr45_c0, 8, filter+2*512, 256) ;
      _vel_vst_vssl(vr67_c0, 8, filter+3*512, 256) ;
      _vel_vst_vssl(vr89_c0, 8, filter+4*512, 256) ;
      _vel_vst_vssl(vrAB_c0, 8, filter+5*512, 256) ;
      _vel_vst_vssl(vrCD_c0, 8, filter+6*512, 256) ;
      _vel_vst_vssl(vrEF_c0, 8, filter+7*512, 256) ;

      __vr vr01_c1 = _vel_vshf_vvvsl(vr0,vr1,VE_VSHUFFLE_YUZU, 256) ;
      __vr vr23_c1 = _vel_vshf_vvvsl(vr2,vr3,VE_VSHUFFLE_YUZU, 256) ;
      __vr vr45_c1 = _vel_vshf_vvvsl(vr4,vr5,VE_VSHUFFLE_YUZU, 256) ;
      __vr vr67_c1 = _vel_vshf_vvvsl(vr6,vr7,VE_VSHUFFLE_YUZU, 256) ;
      __vr vr89_c1 = _vel_vshf_vvvsl(vr8,vr9,VE_VSHUFFLE_YUZU, 256) ;
      __vr vrAB_c1 = _vel_vshf_vvvsl(vrA,vrB,VE_VSHUFFLE_YUZU, 256) ;
      __vr vrCD_c1 = _vel_vshf_vvvsl(vrC,vrD,VE_VSHUFFLE_YUZU, 256) ;
      __vr vrEF_c1 = _vel_vshf_vvvsl(vrE,vrF,VE_VSHUFFLE_YUZU, 256) ;

      _vel_vst_vssl(vr01_c1, 8, filter+ 8*512, 256) ;
      _vel_vst_vssl(vr23_c1, 8, filter+ 9*512, 256) ;
      _vel_vst_vssl(vr45_c1, 8, filter+10*512, 256) ;
      _vel_vst_vssl(vr67_c1, 8, filter+11*512, 256) ;
      _vel_vst_vssl(vr89_c1, 8, filter+12*512, 256) ;
      _vel_vst_vssl(vrAB_c1, 8, filter+13*512, 256) ;
      _vel_vst_vssl(vrCD_c1, 8, filter+14*512, 256) ;
      _vel_vst_vssl(vrEF_c1, 8, filter+15*512, 256) ;

      for(int64_t c1 = 0; c1 < 512 ; c1+=8 ) {
	const int64_t c = c0 + c1 ;

	const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;
	const uint64_t* filter_u64 = (const uint64_t*)(filter+c1) ;

	__vr vrpin_c0 = _vel_vsfa_vvssl(vrij, 2, (uint64_t)(pInChannel+y*strideHeight*inWidth), vl) ;
	__vr vrpin_c1 = _vel_vaddul_vsvl(  4*inHeight*inWidth,vrpin_c0, vl) ;
	__vr vrpin_c2 = _vel_vaddul_vsvl(2*4*inHeight*inWidth,vrpin_c0, vl) ;
	__vr vrpin_c3 = _vel_vaddul_vsvl(3*4*inHeight*inWidth,vrpin_c0, vl) ;
	__vr vrpin_c4 = _vel_vaddul_vsvl(4*4*inHeight*inWidth,vrpin_c0, vl) ;
	__vr vrpin_c5 = _vel_vaddul_vsvl(5*4*inHeight*inWidth,vrpin_c0, vl) ;
	__vr vrpin_c6 = _vel_vaddul_vsvl(6*4*inHeight*inWidth,vrpin_c0, vl) ;
	__vr vrpin_c7 = _vel_vaddul_vsvl(7*4*inHeight*inWidth,vrpin_c0, vl) ;

	__vr vrin_c0 = _vel_vgtu_vvssl(vrpin_c0, 0, 0, vl) ;
	__vr vrin_c1 = _vel_vgtu_vvssl(vrpin_c1, 0, 0, vl) ;
	__vr vrin_c2 = _vel_vgtu_vvssl(vrpin_c2, 0, 0, vl) ;
	__vr vrin_c3 = _vel_vgtu_vvssl(vrpin_c3, 0, 0, vl) ;
	__vr vrin_c4 = _vel_vgtu_vvssl(vrpin_c4, 0, 0, vl) ;
	__vr vrin_c5 = _vel_vgtu_vvssl(vrpin_c5, 0, 0, vl) ;
	__vr vrin_c6 = _vel_vgtu_vvssl(vrpin_c6, 0, 0, vl) ;
	__vr vrin_c7 = _vel_vgtu_vvssl(vrpin_c7, 0, 0, vl) ;

	const float *pKerValue = pKernel + kernGroupOffset + (k * inChannelGroup + c);

	__vr vrin_c0P = _vel_vshf_vvvsl(vrin_c0, vrin_c0, VE_VSHUFFLE_YUZU, vl) ;
	vrsum01 = _vel_pvfmad_vvsvl(vrsum01, filter_u64[0*256], vrin_c0P, vl) ;
	vrsum23 = _vel_pvfmad_vvsvl(vrsum23, filter_u64[1*256], vrin_c0P, vl) ;
	vrsum45 = _vel_pvfmad_vvsvl(vrsum45, filter_u64[2*256], vrin_c0P, vl) ;
	vrsum67 = _vel_pvfmad_vvsvl(vrsum67, filter_u64[3*256], vrin_c0P, vl) ;
	vrsum89 = _vel_pvfmad_vvsvl(vrsum89, filter_u64[4*256], vrin_c0P, vl) ;
	vrsumAB = _vel_pvfmad_vvsvl(vrsumAB, filter_u64[5*256], vrin_c0P, vl) ;
	vrsumCD = _vel_pvfmad_vvsvl(vrsumCD, filter_u64[6*256], vrin_c0P, vl) ;
	vrsumEF = _vel_pvfmad_vvsvl(vrsumEF, filter_u64[7*256], vrin_c0P, vl) ;

	__vr vrin_c1P = _vel_vshf_vvvsl(vrin_c1, vrin_c1, VE_VSHUFFLE_YUZU, vl) ;
	vrsum01 = _vel_pvfmad_vvsvl(vrsum01, filter_u64[8*256],  vrin_c1P, vl) ;
	vrsum23 = _vel_pvfmad_vvsvl(vrsum23, filter_u64[9*256],  vrin_c1P, vl) ;
	vrsum45 = _vel_pvfmad_vvsvl(vrsum45, filter_u64[10*256], vrin_c1P, vl) ;
	vrsum67 = _vel_pvfmad_vvsvl(vrsum67, filter_u64[11*256], vrin_c1P, vl) ;
	vrsum89 = _vel_pvfmad_vvsvl(vrsum89, filter_u64[12*256], vrin_c1P, vl) ;
	vrsumAB = _vel_pvfmad_vvsvl(vrsumAB, filter_u64[13*256], vrin_c1P, vl) ;
	vrsumCD = _vel_pvfmad_vvsvl(vrsumCD, filter_u64[14*256], vrin_c1P, vl) ;
	vrsumEF = _vel_pvfmad_vvsvl(vrsumEF, filter_u64[15*256], vrin_c1P, vl) ;

	__vr vrin_c2P = _vel_vshf_vvvsl(vrin_c2, vrin_c2, VE_VSHUFFLE_YUZU, vl) ;
	vrsum01 = _vel_pvfmad_vvsvl(vrsum01, filter_u64[0*256+1], vrin_c2P, vl) ;
	vrsum23 = _vel_pvfmad_vvsvl(vrsum23, filter_u64[1*256+1], vrin_c2P, vl) ;
	vrsum45 = _vel_pvfmad_vvsvl(vrsum45, filter_u64[2*256+1], vrin_c2P, vl) ;
	vrsum67 = _vel_pvfmad_vvsvl(vrsum67, filter_u64[3*256+1], vrin_c2P, vl) ;
	vrsum89 = _vel_pvfmad_vvsvl(vrsum89, filter_u64[4*256+1], vrin_c2P, vl) ;
	vrsumAB = _vel_pvfmad_vvsvl(vrsumAB, filter_u64[5*256+1], vrin_c2P, vl) ;
	vrsumCD = _vel_pvfmad_vvsvl(vrsumCD, filter_u64[6*256+1], vrin_c2P, vl) ;
	vrsumEF = _vel_pvfmad_vvsvl(vrsumEF, filter_u64[7*256+1], vrin_c2P, vl) ;

	__vr vrin_c3P = _vel_vshf_vvvsl(vrin_c3, vrin_c3, VE_VSHUFFLE_YUZU, vl) ;
	vrsum01 = _vel_pvfmad_vvsvl(vrsum01, filter_u64[8*256+1],  vrin_c3P, vl) ;
	vrsum23 = _vel_pvfmad_vvsvl(vrsum23, filter_u64[9*256+1],  vrin_c3P, vl) ;
	vrsum45 = _vel_pvfmad_vvsvl(vrsum45, filter_u64[10*256+1], vrin_c3P, vl) ;
	vrsum67 = _vel_pvfmad_vvsvl(vrsum67, filter_u64[11*256+1], vrin_c3P, vl) ;
	vrsum89 = _vel_pvfmad_vvsvl(vrsum89, filter_u64[12*256+1], vrin_c3P, vl) ;
	vrsumAB = _vel_pvfmad_vvsvl(vrsumAB, filter_u64[13*256+1], vrin_c3P, vl) ;
	vrsumCD = _vel_pvfmad_vvsvl(vrsumCD, filter_u64[14*256+1], vrin_c3P, vl) ;
	vrsumEF = _vel_pvfmad_vvsvl(vrsumEF, filter_u64[15*256+1], vrin_c3P, vl) ;

	__vr vrin_c4P = _vel_vshf_vvvsl(vrin_c4, vrin_c4, VE_VSHUFFLE_YUZU, vl) ;
	vrsum01 = _vel_pvfmad_vvsvl(vrsum01, filter_u64[0*256+2], vrin_c4P, vl) ;
	vrsum23 = _vel_pvfmad_vvsvl(vrsum23, filter_u64[1*256+2], vrin_c4P, vl) ;
	vrsum45 = _vel_pvfmad_vvsvl(vrsum45, filter_u64[2*256+2], vrin_c4P, vl) ;
	vrsum67 = _vel_pvfmad_vvsvl(vrsum67, filter_u64[3*256+2], vrin_c4P, vl) ;
	vrsum89 = _vel_pvfmad_vvsvl(vrsum89, filter_u64[4*256+2], vrin_c4P, vl) ;
	vrsumAB = _vel_pvfmad_vvsvl(vrsumAB, filter_u64[5*256+2], vrin_c4P, vl) ;
	vrsumCD = _vel_pvfmad_vvsvl(vrsumCD, filter_u64[6*256+2], vrin_c4P, vl) ;
	vrsumEF = _vel_pvfmad_vvsvl(vrsumEF, filter_u64[7*256+2], vrin_c4P, vl) ;

	__vr vrin_c5P = _vel_vshf_vvvsl(vrin_c5, vrin_c5, VE_VSHUFFLE_YUZU, vl) ;
	vrsum01 = _vel_pvfmad_vvsvl(vrsum01, filter_u64[8*256+2],  vrin_c5P, vl) ;
	vrsum23 = _vel_pvfmad_vvsvl(vrsum23, filter_u64[9*256+2],  vrin_c5P, vl) ;
	vrsum45 = _vel_pvfmad_vvsvl(vrsum45, filter_u64[10*256+2], vrin_c5P, vl) ;
	vrsum67 = _vel_pvfmad_vvsvl(vrsum67, filter_u64[11*256+2], vrin_c5P, vl) ;
	vrsum89 = _vel_pvfmad_vvsvl(vrsum89, filter_u64[12*256+2], vrin_c5P, vl) ;
	vrsumAB = _vel_pvfmad_vvsvl(vrsumAB, filter_u64[13*256+2], vrin_c5P, vl) ;
	vrsumCD = _vel_pvfmad_vvsvl(vrsumCD, filter_u64[14*256+2], vrin_c5P, vl) ;
	vrsumEF = _vel_pvfmad_vvsvl(vrsumEF, filter_u64[15*256+2], vrin_c5P, vl) ;

	__vr vrin_c6P = _vel_vshf_vvvsl(vrin_c6, vrin_c6, VE_VSHUFFLE_YUZU, vl) ;
	vrsum01 = _vel_pvfmad_vvsvl(vrsum01, filter_u64[0*256+3], vrin_c6P, vl) ;
	vrsum23 = _vel_pvfmad_vvsvl(vrsum23, filter_u64[1*256+3], vrin_c6P, vl) ;
	vrsum45 = _vel_pvfmad_vvsvl(vrsum45, filter_u64[2*256+3], vrin_c6P, vl) ;
	vrsum67 = _vel_pvfmad_vvsvl(vrsum67, filter_u64[3*256+3], vrin_c6P, vl) ;
	vrsum89 = _vel_pvfmad_vvsvl(vrsum89, filter_u64[4*256+3], vrin_c6P, vl) ;
	vrsumAB = _vel_pvfmad_vvsvl(vrsumAB, filter_u64[5*256+3], vrin_c6P, vl) ;
	vrsumCD = _vel_pvfmad_vvsvl(vrsumCD, filter_u64[6*256+3], vrin_c6P, vl) ;
	vrsumEF = _vel_pvfmad_vvsvl(vrsumEF, filter_u64[7*256+3], vrin_c6P, vl) ;

	__vr vrin_c7P = _vel_vshf_vvvsl(vrin_c7, vrin_c7, VE_VSHUFFLE_YUZU, vl) ;
	vrsum01 = _vel_pvfmad_vvsvl(vrsum01, filter_u64[8*256+3],  vrin_c7P, vl) ;
	vrsum23 = _vel_pvfmad_vvsvl(vrsum23, filter_u64[9*256+3],  vrin_c7P, vl) ;
	vrsum45 = _vel_pvfmad_vvsvl(vrsum45, filter_u64[10*256+3], vrin_c7P, vl) ;
	vrsum67 = _vel_pvfmad_vvsvl(vrsum67, filter_u64[11*256+3], vrin_c7P, vl) ;
	vrsum89 = _vel_pvfmad_vvsvl(vrsum89, filter_u64[12*256+3], vrin_c7P, vl) ;
	vrsumAB = _vel_pvfmad_vvsvl(vrsumAB, filter_u64[13*256+3], vrin_c7P, vl) ;
	vrsumCD = _vel_pvfmad_vvsvl(vrsumCD, filter_u64[14*256+3], vrin_c7P, vl) ;
	vrsumEF = _vel_pvfmad_vvsvl(vrsumEF, filter_u64[15*256+3], vrin_c7P, vl) ;
      } // inChannel
    }

    _vel_vstu_vssl(vrsum01, 4, pOut+outIndex + 0 * outHeight*outWidth, vl) ;
    _vel_vstl_vssl(vrsum01, 4, pOut+outIndex + 1 * outHeight*outWidth, vl) ;
    _vel_vstu_vssl(vrsum23, 4, pOut+outIndex + 2 * outHeight*outWidth, vl) ;
    _vel_vstl_vssl(vrsum23, 4, pOut+outIndex + 3 * outHeight*outWidth, vl) ;
    _vel_vstu_vssl(vrsum45, 4, pOut+outIndex + 4 * outHeight*outWidth, vl) ;
    _vel_vstl_vssl(vrsum45, 4, pOut+outIndex + 5 * outHeight*outWidth, vl) ;
    _vel_vstu_vssl(vrsum67, 4, pOut+outIndex + 6 * outHeight*outWidth, vl) ;
    _vel_vstl_vssl(vrsum67, 4, pOut+outIndex + 7 * outHeight*outWidth, vl) ;
    _vel_vstu_vssl(vrsum89, 4, pOut+outIndex + 8 * outHeight*outWidth, vl) ;
    _vel_vstl_vssl(vrsum89, 4, pOut+outIndex + 9 * outHeight*outWidth, vl) ;
    _vel_vstu_vssl(vrsumAB, 4, pOut+outIndex +10 * outHeight*outWidth, vl) ;
    _vel_vstl_vssl(vrsumAB, 4, pOut+outIndex +11 * outHeight*outWidth, vl) ;
    _vel_vstu_vssl(vrsumCD, 4, pOut+outIndex +12 * outHeight*outWidth, vl) ;
    _vel_vstl_vssl(vrsumCD, 4, pOut+outIndex +13 * outHeight*outWidth, vl) ;
    _vel_vstu_vssl(vrsumEF, 4, pOut+outIndex +14 * outHeight*outWidth, vl) ;
    _vel_vstl_vssl(vrsumEF, 4, pOut+outIndex +15 * outHeight*outWidth, vl) ;

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
    const int64_t kernWidth,
    const int64_t kernHeight,
    const int64_t inChannelGroup,
    const int64_t outChannelGroup,
    const int64_t strideHeight,
    const int64_t strideWidth
)
{
  const int64_t nY = VLEN / outWidth ;

  __vr vrseq = _vel_vseq_vl(nY*outWidth) ;
  __vr vry  = _vel_vdivsl_vvsl(vrseq, outWidth, nY*outWidth) ;
  __vr vrx  = _vel_vsubsl_vvvl(vrseq, _vel_vmulul_vsvl(outWidth,vry, nY*outWidth), nY*outWidth) ;

  __vr vri   = _vel_vmulsl_vsvl(strideHeight, vry, nY*outWidth) ;
  __vr vrj   = _vel_vmulsl_vsvl(strideWidth,  vrx, nY*outWidth) ;
  __vr vrij = _vel_vaddul_vvvl(vrj, _vel_vmulul_vsvl(inWidth, vri, nY*outWidth), nY*outWidth) ;

  for (int64_t n = 0; n < batch; n++) {
    for (int64_t g = 0; g < group; g++) {
	const int64_t inGroupOffset   = g * inChannelGroup * inHeight * inWidth;
	const int64_t outGroupOffset  = g * outChannelGroup * outHeight * outWidth;
	const int64_t biasGroupOffset = g * outChannelGroup;
	const int64_t kernGroupOffset = g * outChannelGroup * inChannelGroup * kernHeight * kernWidth;

	const int64_t remain = outChannelGroup & 0xf ;

	int k = 0 ;
	switch( remain ) {
	case 1 :
	  func_odd<FLAYOUT, 1, ADDBIAS>(pIn, pKernel, pBias, pOut,
	     inChannel, inWidth, inHeight,
	     outChannel, outWidth, outHeight,
	     kernWidth, kernHeight,
	     inChannelGroup, outChannelGroup,
	     strideHeight, strideWidth,
	     inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	     nY, vrij, n, k ) ;
	  k+=1 ;
	  break ;
	case 2 :
	  func_even<FLAYOUT, 2, ADDBIAS>(pIn, pKernel, pBias, pOut,
	     inChannel, inWidth, inHeight,
	     outChannel, outWidth, outHeight,
	     kernWidth, kernHeight,
	     inChannelGroup, outChannelGroup,
	     strideHeight, strideWidth,
	     inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	     nY, vrij, n, k ) ;
	  k+=2 ;
	  break ;
	case 3 :
	  func_odd<FLAYOUT, 3, ADDBIAS>(pIn, pKernel, pBias, pOut,
	     inChannel, inWidth, inHeight,
	     outChannel, outWidth, outHeight,
	     kernWidth, kernHeight,
	     inChannelGroup, outChannelGroup,
	     strideHeight, strideWidth,
	     inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	     nY, vrij, n, k ) ;
	  k+=3 ;
	  break ;
	case 4 :
	  func_even<FLAYOUT, 4, ADDBIAS>(pIn, pKernel, pBias, pOut,
	     inChannel, inWidth, inHeight,
	     outChannel, outWidth, outHeight,
	     kernWidth, kernHeight,
	     inChannelGroup, outChannelGroup,
	     strideHeight, strideWidth,
	     inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	     nY, vrij, n, k ) ;
	  k+=4 ;
	  break ;
	case 5 :
	  func_odd<FLAYOUT, 5, ADDBIAS>(pIn, pKernel, pBias, pOut,
	     inChannel, inWidth, inHeight,
	     outChannel, outWidth, outHeight,
	     kernWidth, kernHeight,
	     inChannelGroup, outChannelGroup,
	     strideHeight, strideWidth,
	     inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	     nY, vrij, n, k ) ;
	  k+=5 ;
	  break ;
	case 6 :
	  func_even<FLAYOUT, 6, ADDBIAS>(pIn, pKernel, pBias, pOut,
	     inChannel, inWidth, inHeight,
	     outChannel, outWidth, outHeight,
	     kernWidth, kernHeight,
	     inChannelGroup, outChannelGroup,
	     strideHeight, strideWidth,
	     inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	     nY, vrij, n, k ) ;
	  k+=6 ;
	  break ;
	case 7 :
	  func_odd<FLAYOUT, 7, ADDBIAS>(pIn, pKernel, pBias, pOut,
	     inChannel, inWidth, inHeight,
	     outChannel, outWidth, outHeight,
	     kernWidth, kernHeight,
	     inChannelGroup, outChannelGroup,
	     strideHeight, strideWidth,
	     inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	     nY, vrij, n, k ) ;
	  k+=7 ;
	  break ;
	case 8 :
	  func_even<FLAYOUT, 8, ADDBIAS>(pIn, pKernel, pBias, pOut,
	     inChannel, inWidth, inHeight,
	     outChannel, outWidth, outHeight,
	     kernWidth, kernHeight,
	     inChannelGroup, outChannelGroup,
	     strideHeight, strideWidth,
	     inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	     nY, vrij, n, k ) ;
	  k+=8 ;
	  break ;
	case 9 :
	  func_odd<FLAYOUT, 9, ADDBIAS>(pIn, pKernel, pBias, pOut,
	     inChannel, inWidth, inHeight,
	     outChannel, outWidth, outHeight,
	     kernWidth, kernHeight,
	     inChannelGroup, outChannelGroup,
	     strideHeight, strideWidth,
	     inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	     nY, vrij, n, k ) ;
	  k+=9 ;
	  break ;
	case 10 :
	  func_even<FLAYOUT, 10, ADDBIAS>(pIn, pKernel, pBias, pOut,
	     inChannel, inWidth, inHeight,
	     outChannel, outWidth, outHeight,
	     kernWidth, kernHeight,
	     inChannelGroup, outChannelGroup,
	     strideHeight, strideWidth,
	     inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	     nY, vrij, n, k ) ;
	  k+=10 ;
	  break ;
	case 11 :
	  func_odd<FLAYOUT, 11, ADDBIAS>(pIn, pKernel, pBias, pOut,
	     inChannel, inWidth, inHeight,
	     outChannel, outWidth, outHeight,
	     kernWidth, kernHeight,
	     inChannelGroup, outChannelGroup,
	     strideHeight, strideWidth,
	     inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	     nY, vrij, n, k ) ;
	  k+=11 ;
	  break ;
	case 12 :
	  func_even<FLAYOUT, 12, ADDBIAS>(pIn, pKernel, pBias, pOut,
	     inChannel, inWidth, inHeight,
	     outChannel, outWidth, outHeight,
	     kernWidth, kernHeight,
	     inChannelGroup, outChannelGroup,
	     strideHeight, strideWidth,
	     inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	     nY, vrij, n, k ) ;
	  k+=12 ;
	  break ;
	case 13 :
	  func_odd<FLAYOUT, 13, ADDBIAS>(pIn, pKernel, pBias, pOut,
	     inChannel, inWidth, inHeight,
	     outChannel, outWidth, outHeight,
	     kernWidth, kernHeight,
	     inChannelGroup, outChannelGroup,
	     strideHeight, strideWidth,
	     inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	     nY, vrij, n, k ) ;
	  k+=13 ;
	  break ;
	case 14 :
	  func_even<FLAYOUT, 14, ADDBIAS>(pIn, pKernel, pBias, pOut,
	     inChannel, inWidth, inHeight,
	     outChannel, outWidth, outHeight,
	     kernWidth, kernHeight,
	     inChannelGroup, outChannelGroup,
	     strideHeight, strideWidth,
	     inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	     nY, vrij, n, k ) ;
	  k+=14 ;
	  break ;
	case 15 :
	  func_odd<FLAYOUT, 15, ADDBIAS>(pIn, pKernel, pBias, pOut,
	     inChannel, inWidth, inHeight,
	     outChannel, outWidth, outHeight,
	     kernWidth, kernHeight,
	     inChannelGroup, outChannelGroup,
	     strideHeight, strideWidth,
	     inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	     nY, vrij, n, k ) ;
	  k+=15 ;
	  break ;
	default :
	  break ;
	}
	for (; k < outChannelGroup; k+=16) {
	  if( FLAYOUT == VEDNN_FILTER_LAYOUT_NCHW &&
	     ( inChannelGroup % 1024 == 0 && (((uint64_t)pKernel) & 0x7) == 0 ) )
	  {
	    k16_filter_nchw_c1024x<ADDBIAS>(pIn, pKernel, pBias, pOut,
		       inChannel, inWidth, inHeight,
		       outChannel, outWidth, outHeight,
		       kernWidth, kernHeight,
		       inChannelGroup, outChannelGroup,
		       strideHeight, strideWidth,
		       inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
		       nY, vrij, n, k ) ;
	  }
	  else {
	    func_even<FLAYOUT, 16, ADDBIAS>(pIn, pKernel, pBias, pOut,
	       inChannel, inWidth, inHeight,
	       outChannel, outWidth, outHeight,
	       kernWidth, kernHeight,
	       inChannelGroup, outChannelGroup,
	       strideHeight, strideWidth,
	       inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	       nY, vrij, n, k ) ;
	  }
	} // outChannel
    } // group
  } // batch
}


extern "C" vednnError_t
vednnConvolutionForward_direct_dil1_pad0_owU128_ker1(
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
  const int64_t kernWidth  = pParamKernel->width;
  const int64_t kernHeight = pParamKernel->height;

  const int64_t filter_layout = pParamKernel->layout ;

  const int64_t group          = pParamConv->group;
  const int64_t strideWidth    = pParamConv->strideWidth;;
  const int64_t strideHeight   = pParamConv->strideHeight;
//  const int64_t padWidth       = pParamConv->padWidth;
//  const int64_t padHeight      = pParamConv->padHeight;
//  const int64_t dilationWidth  = pParamConv->dilationWidth;
//  const int64_t dilationHeight = pParamConv->dilationHeight;

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
		 kernWidth, kernHeight,
		 inChannelGroup, outChannelGroup,
		 strideHeight, strideWidth ) ;
    }
    else {
      convloop<VEDNN_FILTER_LAYOUT_NCHW, true>(pIn, pKernel, pBias, pOut,
		 batch, group,
		 inChannel, inWidth, inHeight,
		 outChannel, outWidth, outHeight,
		 kernWidth, kernHeight,
		 inChannelGroup, outChannelGroup,
		 strideHeight, strideWidth ) ;
    }
  }
  else {
    if( pDataBias == NULL ) {
      convloop<VEDNN_FILTER_LAYOUT_HWCN, false>(pIn, pKernel, pBias, pOut,
		 batch, group,
		 inChannel, inWidth, inHeight,
		 outChannel, outWidth, outHeight,
		 kernWidth, kernHeight,
		 inChannelGroup, outChannelGroup,
		 strideHeight, strideWidth ) ;
    }
    else {
      convloop<VEDNN_FILTER_LAYOUT_HWCN, true>(pIn, pKernel, pBias, pOut,
		 batch, group,
		 inChannel, inWidth, inHeight,
		 outChannel, outWidth, outHeight,
		 kernWidth, kernHeight,
		 inChannelGroup, outChannelGroup,
		 strideHeight, strideWidth ) ;
    }
  }

  return VEDNN_SUCCESS;
}

