#include <stdint.h>
#include <stdlib.h>

#include "vednn.h"

#include "velintrin.h"
#define VLEN	(256)

template<filterLayout_t FLAYOUT, int NUMKERNEL, bool ADDBIAS>
static inline void func(
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

  const int64_t remain  = NUMKERNEL & 0x1 ;
  const int64_t nPacked = NUMKERNEL >> 1 ;

  const float   bias0  = ADDBIAS ?  pBias[biasGroupOffset+k+ 0] : 0.f ;
  int64_t bias[nPacked] ;
#pragma clang loop unroll(full)
  for(int64_t kk=0; kk<nPacked; kk++) bias[kk] = ADDBIAS ?  _vel_pack_f32p(pBias+biasGroupOffset+k+2*kk+remain, pBias+biasGroupOffset+k+2*kk+remain+1) : 0UL ;

  for (int64_t y=0; y<outHeight; y+=nY)
  {
    const int64_t vl = outWidth * (outHeight - y < nY ? outHeight - y : nY) ;
    const int64_t op = y * outWidth ;

    __vr vrsum0  = _vel_vbrds_vsl(bias0, vl) ;
    __vr vrsum[nPacked] ;
#pragma clang loop unroll(full)
    for(int64_t kk=0; kk<nPacked; kk++) vrsum[kk] = _vel_pvbrd_vsl(bias[kk], vl) ;

    int c = 0 ;
    if ( (inChannelGroup & 0x01) == 1 ) {
      const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;
      __vr vrpin_c0 = _vel_vsfa_vvssl(vrij, 2, (uint64_t)(pInChannel+y*strideHeight*inWidth), vl) ;
      __vr vrin_c0 = _vel_vgtu_vvssl(vrpin_c0, 0, 0, vl) ;

#define FILTER_OFFSET(K,C) ( kernGroupOffset + filter_index<FLAYOUT>(K,C,0,0, inChannelGroup, outChannelGroup, 1, 1) )
#define VFMAD(VRIN, C)											\
      {													\
	__vr vrinP = _vel_vshf_vvvsl(VRIN, VRIN, VE_VSHUFFLE_YUZU, vl) ;				\
	if( remain ) {											\
	  const float    kerValue0  = pKernel[FILTER_OFFSET(k+0,(C))] ;					\
	  vrsum0 = _vel_vfmads_vvsvl(vrsum0, kerValue0, VRIN, vl) ;					\
	}												\
	_Pragma("clang loop unroll(full)")								\
	for(int64_t kk=0; kk<nPacked; kk++) {								\
	  const uint64_t kerValue = _vel_pack_f32p(pKernel + FILTER_OFFSET(k+2*kk+remain,  (C)),	\
						   pKernel + FILTER_OFFSET(k+2*kk+remain+1,(C))) ;	\
	  vrsum[kk] = _vel_pvfmad_vvsvl(vrsum[kk], kerValue, vrinP, vl) ;				\
	}												\
      }

      VFMAD(vrin_c0, c+0) ;

      c+=1 ;
    }
    if ( ((inChannelGroup >> 1) & 0x01) == 1 ) {
      const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;
      __vr vrpin_c0 = _vel_vsfa_vvssl(vrij, 2, (uint64_t)(pInChannel+y*strideHeight*inWidth), vl) ;
      __vr vrin_c0 = _vel_vgtu_vvssl(vrpin_c0, 0, 0, vl) ;
      __vr vrpin_c1 = _vel_vaddul_vsvl(  4*inHeight*inWidth,vrpin_c0, vl) ;
      __vr vrin_c1 = _vel_vgtu_vvssl(vrpin_c1, 0, 0, vl) ;


      VFMAD(vrin_c0, c+0) ;
      VFMAD(vrin_c1, c+1) ;


      c+=2 ;
    } // inChannel
    if ( ((inChannelGroup >> 2) & 0x01) == 1 ) {
      const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;
      __vr vrpin_c0 = _vel_vsfa_vvssl(vrij, 2, (uint64_t)(pInChannel+y*strideHeight*inWidth), vl) ;
      __vr vrin_c0 = _vel_vgtu_vvssl(vrpin_c0, 0, 0, vl) ;
      __vr vrpin_c1 = _vel_vaddul_vsvl(  4*inHeight*inWidth,vrpin_c0, vl) ;
      __vr vrin_c1 = _vel_vgtu_vvssl(vrpin_c1, 0, 0, vl) ;
      __vr vrpin_c2 = _vel_vaddul_vsvl(2*4*inHeight*inWidth,vrpin_c0, vl) ;
      __vr vrin_c2 = _vel_vgtu_vvssl(vrpin_c2, 0, 0, vl) ;
      __vr vrpin_c3 = _vel_vaddul_vsvl(3*4*inHeight*inWidth,vrpin_c0, vl) ;
      __vr vrin_c3 = _vel_vgtu_vvssl(vrpin_c3, 0, 0, vl) ;

      VFMAD(vrin_c0, c+0) ;
      VFMAD(vrin_c1, c+1) ;
      VFMAD(vrin_c2, c+2) ;
      VFMAD(vrin_c3, c+3) ;

      c+=4 ;
    }
    if ( ((inChannelGroup >> 3) & 0x01) == 1 ) {
      const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;
      __vr vrpin_c0 = _vel_vsfa_vvssl(vrij, 2, (uint64_t)(pInChannel+y*strideHeight*inWidth), vl) ;
      __vr vrin_c0 = _vel_vgtu_vvssl(vrpin_c0, 0, 0, vl) ;
      __vr vrpin_c1 = _vel_vaddul_vsvl(  4*inHeight*inWidth,vrpin_c0, vl) ;
      __vr vrin_c1 = _vel_vgtu_vvssl(vrpin_c1, 0, 0, vl) ;
      __vr vrpin_c2 = _vel_vaddul_vsvl(2*4*inHeight*inWidth,vrpin_c0, vl) ;
      __vr vrin_c2 = _vel_vgtu_vvssl(vrpin_c2, 0, 0, vl) ;
      __vr vrpin_c3 = _vel_vaddul_vsvl(3*4*inHeight*inWidth,vrpin_c0, vl) ;
      __vr vrin_c3 = _vel_vgtu_vvssl(vrpin_c3, 0, 0, vl) ;
      __vr vrpin_c4 = _vel_vaddul_vsvl(4*4*inHeight*inWidth,vrpin_c0, vl) ;
      __vr vrin_c4 = _vel_vgtu_vvssl(vrpin_c4, 0, 0, vl) ;
      __vr vrpin_c5 = _vel_vaddul_vsvl(5*4*inHeight*inWidth,vrpin_c0, vl) ;
      __vr vrin_c5 = _vel_vgtu_vvssl(vrpin_c5, 0, 0, vl) ;
      __vr vrpin_c6 = _vel_vaddul_vsvl(6*4*inHeight*inWidth,vrpin_c0, vl) ;
      __vr vrin_c6 = _vel_vgtu_vvssl(vrpin_c6, 0, 0, vl) ;
      __vr vrpin_c7 = _vel_vaddul_vsvl(7*4*inHeight*inWidth,vrpin_c0, vl) ;
      __vr vrin_c7 = _vel_vgtu_vvssl(vrpin_c7, 0, 0, vl) ;

      VFMAD(vrin_c0, c+0) ;
      VFMAD(vrin_c1, c+1) ;
      VFMAD(vrin_c2, c+2) ;
      VFMAD(vrin_c3, c+3) ;
      VFMAD(vrin_c4, c+4) ;
      VFMAD(vrin_c5, c+5) ;
      VFMAD(vrin_c6, c+6) ;
      VFMAD(vrin_c7, c+7) ;

      c+=8 ;
    }
    for (; c < inChannelGroup; c+=16) {
      const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;
      __vr vrpin_c0 = _vel_vsfa_vvssl(vrij, 2, (uint64_t)(pInChannel+y*strideHeight*inWidth), vl) ;
      __vr vrin_c0 = _vel_vgtu_vvssl(vrpin_c0, 0, 0, vl) ;
      __vr vrpin_c1 = _vel_vaddul_vsvl(  4*inHeight*inWidth,vrpin_c0, vl) ;
      __vr vrin_c1 = _vel_vgtu_vvssl(vrpin_c1, 0, 0, vl) ;
      __vr vrpin_c2 = _vel_vaddul_vsvl(2*4*inHeight*inWidth,vrpin_c0, vl) ;
      __vr vrin_c2 = _vel_vgtu_vvssl(vrpin_c2, 0, 0, vl) ;
      __vr vrpin_c3 = _vel_vaddul_vsvl(3*4*inHeight*inWidth,vrpin_c0, vl) ;
      __vr vrin_c3 = _vel_vgtu_vvssl(vrpin_c3, 0, 0, vl) ;
      __vr vrpin_c4 = _vel_vaddul_vsvl(4*4*inHeight*inWidth,vrpin_c0, vl) ;
      __vr vrin_c4 = _vel_vgtu_vvssl(vrpin_c4, 0, 0, vl) ;
      __vr vrpin_c5 = _vel_vaddul_vsvl(5*4*inHeight*inWidth,vrpin_c0, vl) ;
      __vr vrin_c5 = _vel_vgtu_vvssl(vrpin_c5, 0, 0, vl) ;
      __vr vrpin_c6 = _vel_vaddul_vsvl(6*4*inHeight*inWidth,vrpin_c0, vl) ;
      __vr vrin_c6 = _vel_vgtu_vvssl(vrpin_c6, 0, 0, vl) ;
      __vr vrpin_c7 = _vel_vaddul_vsvl(7*4*inHeight*inWidth,vrpin_c0, vl) ;
      __vr vrin_c7 = _vel_vgtu_vvssl(vrpin_c7, 0, 0, vl) ;
      __vr vrpin_c8 = _vel_vaddul_vsvl(8*4*inHeight*inWidth,vrpin_c0, vl) ;
      __vr vrin_c8 = _vel_vgtu_vvssl(vrpin_c8, 0, 0, vl) ;
      __vr vrpin_c9 = _vel_vaddul_vsvl(9*4*inHeight*inWidth,vrpin_c0, vl) ;
      __vr vrin_c9 = _vel_vgtu_vvssl(vrpin_c9, 0, 0, vl) ;
      __vr vrpin_cA = _vel_vaddul_vsvl(10*4*inHeight*inWidth,vrpin_c0, vl) ;
      __vr vrin_cA = _vel_vgtu_vvssl(vrpin_cA, 0, 0, vl) ;
      __vr vrpin_cB = _vel_vaddul_vsvl(11*4*inHeight*inWidth,vrpin_c0, vl) ;
      __vr vrin_cB = _vel_vgtu_vvssl(vrpin_cB, 0, 0, vl) ;
      __vr vrpin_cC = _vel_vaddul_vsvl(12*4*inHeight*inWidth,vrpin_c0, vl) ;
      __vr vrin_cC = _vel_vgtu_vvssl(vrpin_cC, 0, 0, vl) ;
      __vr vrpin_cD = _vel_vaddul_vsvl(13*4*inHeight*inWidth,vrpin_c0, vl) ;
      __vr vrin_cD = _vel_vgtu_vvssl(vrpin_cD, 0, 0, vl) ;
      __vr vrpin_cE = _vel_vaddul_vsvl(14*4*inHeight*inWidth,vrpin_c0, vl) ;
      __vr vrin_cE = _vel_vgtu_vvssl(vrpin_cE, 0, 0, vl) ;
      __vr vrpin_cF = _vel_vaddul_vsvl(15*4*inHeight*inWidth,vrpin_c0, vl) ;
      __vr vrin_cF = _vel_vgtu_vvssl(vrpin_cF, 0, 0, vl) ;

      VFMAD(vrin_c0, c+0) ;
      VFMAD(vrin_c1, c+1) ;
      VFMAD(vrin_c2, c+2) ;
      VFMAD(vrin_c3, c+3) ;
      VFMAD(vrin_c4, c+4) ;
      VFMAD(vrin_c5, c+5) ;
      VFMAD(vrin_c6, c+6) ;
      VFMAD(vrin_c7, c+7) ;
      VFMAD(vrin_c8, c+8) ;
      VFMAD(vrin_c9, c+9) ;
      VFMAD(vrin_cA, c+10) ;
      VFMAD(vrin_cB, c+11) ;
      VFMAD(vrin_cC, c+12) ;
      VFMAD(vrin_cD, c+13) ;
      VFMAD(vrin_cE, c+14) ;
      VFMAD(vrin_cF, c+15) ;

#undef VFMAD
#undef FILTER_OFFSET
    }

    if( remain ) {
	_vel_vstu_vssl(vrsum0,  4, pOut+outIndex + 0 * outHeight*outWidth, vl) ;
    }
#pragma clang loop unroll(full)
    for(int64_t kk=0; kk<nPacked; kk++) {
	_vel_vstu_vssl(vrsum[kk], 4, pOut+outIndex + (2*kk+remain)   * outHeight*outWidth, vl) ;
	_vel_vstl_vssl(vrsum[kk], 4, pOut+outIndex + (2*kk+remain+1) * outHeight*outWidth, vl) ;
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

  int64_t bias[8] ;
#pragma clang loop unroll(full)
  for(int64_t kk=0; kk<8; kk++) bias[kk] = ADDBIAS ?  _vel_pack_f32p(pBias+biasGroupOffset+k+2*kk, pBias+biasGroupOffset+k+2*kk+1) : 0UL ;

  for (int64_t y=0; y<outHeight; y+=nY)
  {
    const int64_t vl = outWidth * (outHeight - y < nY ? outHeight - y : nY) ;
    const int64_t op = y * outWidth ;

    __vr vrsum[8] ;
#pragma clang loop unroll(full)
    for(int64_t kk=0; kk<8; kk++) vrsum[kk] = _vel_pvbrd_vsl(bias[kk], vl) ;


    for(int64_t c0=0; c0<inChannelGroup; c0+=512) {
      const float *pKerValue = pKernel + kernGroupOffset + (k * inChannelGroup + c0) ;

      __vr vr[16] ;
#pragma clang loop unroll(full)
      for(int64_t kk=0; kk<16; kk++) {
	vr[kk] = _vel_vld_vssl(8, pKerValue+ kk*inChannelGroup, 256) ;
      }

#pragma clang loop unroll(full)
      for(int64_t kk=0; kk<8; kk++) {
        __vr vrp = _vel_vshf_vvvsl(vr[2*kk],vr[2*kk+1],VE_VSHUFFLE_YLZL, 256) ;
        _vel_vst_vssl(vrp, 8, filter+kk*512, 256) ;
      }

#pragma clang loop unroll(full)
      for(int64_t kk=0; kk<8; kk++) {
        __vr vrp = _vel_vshf_vvvsl(vr[2*kk],vr[2*kk+1],VE_VSHUFFLE_YUZU, 256) ;
        _vel_vst_vssl(vrp, 8, filter+(8+kk)*512, 256) ;
      }

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
#pragma clang loop unroll(full)
	for(int64_t kk=0; kk<8; kk++) {
          vrsum[kk] = _vel_pvfmad_vvsvl(vrsum[kk], filter_u64[kk*256], vrin_c0P, vl) ;
	}

	__vr vrin_c1P = _vel_vshf_vvvsl(vrin_c1, vrin_c1, VE_VSHUFFLE_YUZU, vl) ;
#pragma clang loop unroll(full)
	for(int64_t kk=0; kk<8; kk++) {
          vrsum[kk] = _vel_pvfmad_vvsvl(vrsum[kk], filter_u64[(8+kk)*256], vrin_c1P, vl) ;
	}

	__vr vrin_c2P = _vel_vshf_vvvsl(vrin_c2, vrin_c2, VE_VSHUFFLE_YUZU, vl) ;
#pragma clang loop unroll(full)
	for(int64_t kk=0; kk<8; kk++) {
          vrsum[kk] = _vel_pvfmad_vvsvl(vrsum[kk], filter_u64[kk*256+1], vrin_c2P, vl) ;
	}

	__vr vrin_c3P = _vel_vshf_vvvsl(vrin_c3, vrin_c3, VE_VSHUFFLE_YUZU, vl) ;
#pragma clang loop unroll(full)
	for(int64_t kk=0; kk<8; kk++) {
          vrsum[kk] = _vel_pvfmad_vvsvl(vrsum[kk], filter_u64[(8+kk)*256+1], vrin_c3P, vl) ;
	}

	__vr vrin_c4P = _vel_vshf_vvvsl(vrin_c4, vrin_c4, VE_VSHUFFLE_YUZU, vl) ;
#pragma clang loop unroll(full)
	for(int64_t kk=0; kk<8; kk++) {
          vrsum[kk] = _vel_pvfmad_vvsvl(vrsum[kk], filter_u64[kk*256+2], vrin_c4P, vl) ;
	}

	__vr vrin_c5P = _vel_vshf_vvvsl(vrin_c5, vrin_c5, VE_VSHUFFLE_YUZU, vl) ;
#pragma clang loop unroll(full)
	for(int64_t kk=0; kk<8; kk++) {
          vrsum[kk] = _vel_pvfmad_vvsvl(vrsum[kk], filter_u64[(8+kk)*256+2], vrin_c5P, vl) ;
	}

	__vr vrin_c6P = _vel_vshf_vvvsl(vrin_c6, vrin_c6, VE_VSHUFFLE_YUZU, vl) ;
#pragma clang loop unroll(full)
	for(int64_t kk=0; kk<8; kk++) {
          vrsum[kk] = _vel_pvfmad_vvsvl(vrsum[kk], filter_u64[kk*256+3], vrin_c6P, vl) ;
	}

	__vr vrin_c7P = _vel_vshf_vvvsl(vrin_c7, vrin_c7, VE_VSHUFFLE_YUZU, vl) ;
#pragma clang loop unroll(full)
	for(int64_t kk=0; kk<8; kk++) {
          vrsum[kk] = _vel_pvfmad_vvsvl(vrsum[kk], filter_u64[(8+kk)*256+3], vrin_c7P, vl) ;
	}

      } // inChannel
    }

#pragma clang loop unroll(full)
    for(int64_t kk=0; kk<8; kk++) {
      _vel_vstu_vssl(vrsum[kk], 4, pOut+outIndex + (2*kk)   * outHeight*outWidth, vl) ;
      _vel_vstl_vssl(vrsum[kk], 4, pOut+outIndex + (2*kk+1) * outHeight*outWidth, vl) ;
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
	  func<FLAYOUT, 1, ADDBIAS>(pIn, pKernel, pBias, pOut,
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
	  func<FLAYOUT, 2, ADDBIAS>(pIn, pKernel, pBias, pOut,
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
	  func<FLAYOUT, 3, ADDBIAS>(pIn, pKernel, pBias, pOut,
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
	  func<FLAYOUT, 4, ADDBIAS>(pIn, pKernel, pBias, pOut,
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
	  func<FLAYOUT, 5, ADDBIAS>(pIn, pKernel, pBias, pOut,
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
	  func<FLAYOUT, 6, ADDBIAS>(pIn, pKernel, pBias, pOut,
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
	  func<FLAYOUT, 7, ADDBIAS>(pIn, pKernel, pBias, pOut,
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
	  func<FLAYOUT, 8, ADDBIAS>(pIn, pKernel, pBias, pOut,
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
	  func<FLAYOUT, 9, ADDBIAS>(pIn, pKernel, pBias, pOut,
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
	  func<FLAYOUT, 10, ADDBIAS>(pIn, pKernel, pBias, pOut,
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
	  func<FLAYOUT, 11, ADDBIAS>(pIn, pKernel, pBias, pOut,
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
	  func<FLAYOUT, 12, ADDBIAS>(pIn, pKernel, pBias, pOut,
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
	  func<FLAYOUT, 13, ADDBIAS>(pIn, pKernel, pBias, pOut,
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
	  func<FLAYOUT, 14, ADDBIAS>(pIn, pKernel, pBias, pOut,
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
	  func<FLAYOUT, 15, ADDBIAS>(pIn, pKernel, pBias, pOut,
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
	    func<FLAYOUT, 16, ADDBIAS>(pIn, pKernel, pBias, pOut,
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

