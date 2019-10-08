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
  const int64_t inGroupOffset,
  const int64_t outGroupOffset,
  const int64_t biasGroupOffset,
  const int64_t kernGroupOffset,
  const int64_t nY,
  const int64_t inWidthHalf,
  const int64_t outWidthHalf,
  const __vm256 vm_s0,
  const __vm256 vm_s2,
  const int64_t n,
  const int64_t k
)
{
  int64_t outIndex = outGroupOffset + (n * outChannel + k  ) * outHeight*outWidth ;

  const int64_t bias0 = ADDBIAS ?  _vel_pack_f32a(pBias+biasGroupOffset+k+ 0) : 0UL ;
  const int64_t bias1 = ADDBIAS ?  _vel_pack_f32a(pBias+biasGroupOffset+k+ 1) : 0UL ;
  const int64_t bias2 = ADDBIAS ?  _vel_pack_f32a(pBias+biasGroupOffset+k+ 2) : 0UL ;
  const int64_t bias3 = ADDBIAS ?  _vel_pack_f32a(pBias+biasGroupOffset+k+ 3) : 0UL ;
  const int64_t bias4 = ADDBIAS ?  _vel_pack_f32a(pBias+biasGroupOffset+k+ 4) : 0UL ;
  const int64_t bias5 = ADDBIAS ?  _vel_pack_f32a(pBias+biasGroupOffset+k+ 5) : 0UL ;
  const int64_t bias6 = ADDBIAS ?  _vel_pack_f32a(pBias+biasGroupOffset+k+ 6) : 0UL ;
  const int64_t bias7 = ADDBIAS ?  _vel_pack_f32a(pBias+biasGroupOffset+k+ 7) : 0UL ;
  const int64_t bias8 = ADDBIAS ?  _vel_pack_f32a(pBias+biasGroupOffset+k+ 8) : 0UL ;
  const int64_t bias9 = ADDBIAS ?  _vel_pack_f32a(pBias+biasGroupOffset+k+ 9) : 0UL ;
  const int64_t biasA = ADDBIAS ?  _vel_pack_f32a(pBias+biasGroupOffset+k+10) : 0UL ;
  const int64_t biasB = ADDBIAS ?  _vel_pack_f32a(pBias+biasGroupOffset+k+11) : 0UL ;
  const int64_t biasC = ADDBIAS ?  _vel_pack_f32a(pBias+biasGroupOffset+k+12) : 0UL ;
  const int64_t biasD = ADDBIAS ?  _vel_pack_f32a(pBias+biasGroupOffset+k+13) : 0UL ;
  const int64_t biasE = ADDBIAS ?  _vel_pack_f32a(pBias+biasGroupOffset+k+14) : 0UL ;
  const int64_t biasF = ADDBIAS ?  _vel_pack_f32a(pBias+biasGroupOffset+k+15) : 0UL ;

  for (int64_t y=0; y<outHeight; y+=nY) {
    const int64_t vl0 = inWidthHalf * (outHeight - y < nY ? outHeight - y : nY) ;
    const int64_t vl1 = outWidthHalf * (outHeight - y < nY ? outHeight - y : nY) ;
    const int64_t op = y * outWidth ;

    __vr vrsum0 = _vel_vbrdl_vsl(bias0, vl1) ;
    __vr vrsum1 = _vel_vbrdl_vsl(bias1, vl1) ;
    __vr vrsum2 = _vel_vbrdl_vsl(bias2, vl1) ;
    __vr vrsum3 = _vel_vbrdl_vsl(bias3, vl1) ;
    __vr vrsum4 = _vel_vbrdl_vsl(bias4, vl1) ;
    __vr vrsum5 = _vel_vbrdl_vsl(bias5, vl1) ;
    __vr vrsum6 = _vel_vbrdl_vsl(bias6, vl1) ;
    __vr vrsum7 = _vel_vbrdl_vsl(bias7, vl1) ;
    __vr vrsum8 = _vel_vbrdl_vsl(bias8, vl1) ;
    __vr vrsum9 = _vel_vbrdl_vsl(bias9, vl1) ;
    __vr vrsumA = _vel_vbrdl_vsl(biasA, vl1) ;
    __vr vrsumB = _vel_vbrdl_vsl(biasB, vl1) ;
    __vr vrsumC = _vel_vbrdl_vsl(biasC, vl1) ;
    __vr vrsumD = _vel_vbrdl_vsl(biasD, vl1) ;
    __vr vrsumE = _vel_vbrdl_vsl(biasE, vl1) ;
    __vr vrsumF = _vel_vbrdl_vsl(biasF, vl1) ;

    for (int64_t c = 0; c < inChannelGroup; c++) {

      const float *pInChannel = pIn + inGroupOffset + (((n * inChannel + c) * inHeight + y) * inWidth ) ;

      __vr vrin_r0 = _vel_vld_vssl(8, pInChannel+0*inWidth, vl0) ;
      __vr vrin_r1 = _vel_vld_vssl(8, pInChannel+1*inWidth, vl0) ;
      __vr vrin_r2 = _vel_vld_vssl(8, pInChannel+2*inWidth, vl0) ;

      __vr vrin_r0s0 = _vel_vcp_vvmvl(vrin_r0, vm_s0, _vel_vbrdl_vsl(0UL, vl0), vl0) ;
      __vr vrin_r0s2 = _vel_vcp_vvmvl(vrin_r0, vm_s2, _vel_vbrdl_vsl(0UL, vl0), vl0) ;
      __vr vrin_r1s0 = _vel_vcp_vvmvl(vrin_r1, vm_s0, _vel_vbrdl_vsl(0UL, vl0), vl0) ;
      __vr vrin_r1s2 = _vel_vcp_vvmvl(vrin_r1, vm_s2, _vel_vbrdl_vsl(0UL, vl0), vl0) ;
      __vr vrin_r2s0 = _vel_vcp_vvmvl(vrin_r2, vm_s0, _vel_vbrdl_vsl(0UL, vl0), vl0) ;
      __vr vrin_r2s2 = _vel_vcp_vvmvl(vrin_r2, vm_s2, _vel_vbrdl_vsl(0UL, vl0), vl0) ;

      __vr vrin_r0s1 = _vel_vshf_vvvsl(vrin_r0s0, vrin_r0s2,VE_VSHUFFLE_ZLYU, vl1) ;
      __vr vrin_r1s1 = _vel_vshf_vvvsl(vrin_r1s0, vrin_r1s2,VE_VSHUFFLE_ZLYU, vl1) ;
      __vr vrin_r2s1 = _vel_vshf_vvvsl(vrin_r2s0, vrin_r2s2,VE_VSHUFFLE_ZLYU, vl1) ;


#define FILTER_OFFSET(k,c,r,s) ( kernGroupOffset + filter_index<FLAYOUT>(k,c,r,s, inChannelGroup, outChannelGroup, kernHeight, kernWidth) )
#define VFADD(VRIN, R, S) 								\
{											\
  if(NUMKERNEL>= 1) {									\
    const uint64_t kerValue0 = _vel_pack_f32a(pKernel+ FILTER_OFFSET(k+ 0,c,R,S)) ;	\
    vrsum0 = _vel_pvfmad_vvsvl(vrsum0, kerValue0, VRIN, vl1) ;				\
  }											\
  if(NUMKERNEL>= 2) {									\
    const uint64_t kerValue1 = _vel_pack_f32a(pKernel+ FILTER_OFFSET(k+ 1,c,R,S)) ;	\
    vrsum1 = _vel_pvfmad_vvsvl(vrsum1, kerValue1, VRIN, vl1) ;				\
  }											\
  if(NUMKERNEL>= 3) {									\
    const uint64_t kerValue2 = _vel_pack_f32a(pKernel+ FILTER_OFFSET(k+ 2,c,R,S)) ;	\
    vrsum2 = _vel_pvfmad_vvsvl(vrsum2, kerValue2, VRIN, vl1) ;				\
  }											\
  if(NUMKERNEL>= 4) {									\
    const uint64_t kerValue3 = _vel_pack_f32a(pKernel+ FILTER_OFFSET(k+ 3,c,R,S)) ;	\
    vrsum3 = _vel_pvfmad_vvsvl(vrsum3, kerValue3, VRIN, vl1) ;				\
  }											\
  if(NUMKERNEL>= 5) {									\
    const uint64_t kerValue4 = _vel_pack_f32a(pKernel+ FILTER_OFFSET(k+ 4,c,R,S)) ;	\
    vrsum4 = _vel_pvfmad_vvsvl(vrsum4, kerValue4, VRIN, vl1) ;				\
  }											\
  if(NUMKERNEL>= 6) {									\
    const uint64_t kerValue5 = _vel_pack_f32a(pKernel+ FILTER_OFFSET(k+ 5,c,R,S)) ;	\
    vrsum5 = _vel_pvfmad_vvsvl(vrsum5, kerValue5, VRIN, vl1) ;				\
  }											\
  if(NUMKERNEL>= 7) {									\
    const uint64_t kerValue6 = _vel_pack_f32a(pKernel+ FILTER_OFFSET(k+ 6,c,R,S)) ;	\
    vrsum6 = _vel_pvfmad_vvsvl(vrsum6, kerValue6, VRIN, vl1) ;				\
  }											\
  if(NUMKERNEL>= 8) {									\
    const uint64_t kerValue7 = _vel_pack_f32a(pKernel+ FILTER_OFFSET(k+ 7,c,R,S)) ;	\
    vrsum7 = _vel_pvfmad_vvsvl(vrsum7, kerValue7, VRIN, vl1) ;				\
  }											\
  if(NUMKERNEL>= 9) {									\
    const uint64_t kerValue8 = _vel_pack_f32a(pKernel+ FILTER_OFFSET(k+ 8,c,R,S)) ;	\
    vrsum8 = _vel_pvfmad_vvsvl(vrsum8, kerValue8, VRIN, vl1) ;				\
  }											\
  if(NUMKERNEL>=10) {									\
    const uint64_t kerValue9 = _vel_pack_f32a(pKernel+ FILTER_OFFSET(k+ 9,c,R,S)) ;	\
    vrsum9 = _vel_pvfmad_vvsvl(vrsum9, kerValue9, VRIN, vl1) ;				\
  }											\
  if(NUMKERNEL>=11) {									\
    const uint64_t kerValueA = _vel_pack_f32a(pKernel+ FILTER_OFFSET(k+10,c,R,S)) ;	\
    vrsumA = _vel_pvfmad_vvsvl(vrsumA, kerValueA, VRIN, vl1) ;				\
  }											\
  if(NUMKERNEL>=12) {									\
    const uint64_t kerValueB = _vel_pack_f32a(pKernel+ FILTER_OFFSET(k+11,c,R,S)) ;	\
    vrsumB = _vel_pvfmad_vvsvl(vrsumB, kerValueB, VRIN, vl1) ;				\
  }											\
  if(NUMKERNEL>=13) {									\
    const uint64_t kerValueC = _vel_pack_f32a(pKernel+ FILTER_OFFSET(k+12,c,R,S)) ;	\
    vrsumC = _vel_pvfmad_vvsvl(vrsumC, kerValueC, VRIN, vl1) ;				\
  }											\
  if(NUMKERNEL>=14) {									\
    const uint64_t kerValueD = _vel_pack_f32a(pKernel+ FILTER_OFFSET(k+13,c,R,S)) ;	\
    vrsumD = _vel_pvfmad_vvsvl(vrsumD, kerValueD, VRIN, vl1) ;				\
  }											\
  if(NUMKERNEL>=15) {									\
    const uint64_t kerValueE = _vel_pack_f32a(pKernel+ FILTER_OFFSET(k+14,c,R,S)) ;	\
    vrsumE = _vel_pvfmad_vvsvl(vrsumE, kerValueE, VRIN, vl1) ;				\
  }											\
  if(NUMKERNEL>=16) {									\
    const uint64_t kerValueF = _vel_pack_f32a(pKernel+ FILTER_OFFSET(k+15,c,R,S)) ;	\
    vrsumF = _vel_pvfmad_vvsvl(vrsumF, kerValueF, VRIN, vl1) ;				\
  }											\
}

      VFADD(vrin_r0s0, 0, 0) ;
      VFADD(vrin_r0s1, 0, 1) ;
      VFADD(vrin_r0s2, 0, 2) ;
      VFADD(vrin_r1s0, 1, 0) ;
      VFADD(vrin_r1s1, 1, 1) ;
      VFADD(vrin_r1s2, 1, 2) ;
      VFADD(vrin_r2s0, 2, 0) ;
      VFADD(vrin_r2s1, 2, 1) ;
      VFADD(vrin_r2s2, 2, 2) ;
#undef VFADD
#undef FILTER_OFFSET

    } // inChannel

    if(NUMKERNEL>= 1) _vel_vst_vssl(vrsum0, 8, pOut+outIndex+ 0*outHeight*outWidth, vl1) ;
    if(NUMKERNEL>= 2) _vel_vst_vssl(vrsum1, 8, pOut+outIndex+ 1*outHeight*outWidth, vl1) ;
    if(NUMKERNEL>= 3) _vel_vst_vssl(vrsum2, 8, pOut+outIndex+ 2*outHeight*outWidth, vl1) ;
    if(NUMKERNEL>= 4) _vel_vst_vssl(vrsum3, 8, pOut+outIndex+ 3*outHeight*outWidth, vl1) ;
    if(NUMKERNEL>= 5) _vel_vst_vssl(vrsum4, 8, pOut+outIndex+ 4*outHeight*outWidth, vl1) ;
    if(NUMKERNEL>= 6) _vel_vst_vssl(vrsum5, 8, pOut+outIndex+ 5*outHeight*outWidth, vl1) ;
    if(NUMKERNEL>= 7) _vel_vst_vssl(vrsum6, 8, pOut+outIndex+ 6*outHeight*outWidth, vl1) ;
    if(NUMKERNEL>= 8) _vel_vst_vssl(vrsum7, 8, pOut+outIndex+ 7*outHeight*outWidth, vl1) ;
    if(NUMKERNEL>= 9) _vel_vst_vssl(vrsum8, 8, pOut+outIndex+ 8*outHeight*outWidth, vl1) ;
    if(NUMKERNEL>=10) _vel_vst_vssl(vrsum9, 8, pOut+outIndex+ 9*outHeight*outWidth, vl1) ;
    if(NUMKERNEL>=11) _vel_vst_vssl(vrsumA, 8, pOut+outIndex+10*outHeight*outWidth, vl1) ;
    if(NUMKERNEL>=12) _vel_vst_vssl(vrsumB, 8, pOut+outIndex+11*outHeight*outWidth, vl1) ;
    if(NUMKERNEL>=13) _vel_vst_vssl(vrsumC, 8, pOut+outIndex+12*outHeight*outWidth, vl1) ;
    if(NUMKERNEL>=14) _vel_vst_vssl(vrsumD, 8, pOut+outIndex+13*outHeight*outWidth, vl1) ;
    if(NUMKERNEL>=15) _vel_vst_vssl(vrsumE, 8, pOut+outIndex+14*outHeight*outWidth, vl1) ;
    if(NUMKERNEL>=16) _vel_vst_vssl(vrsumF, 8, pOut+outIndex+15*outHeight*outWidth, vl1) ;

    outIndex += 2*vl1 ;
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
    const int64_t outChannelGroup
)
{
  const int64_t inWidthHalf  = inWidth >> 1 ;
  const int64_t outWidthHalf = outWidth >> 1 ;
  const int64_t nY = VLEN / inWidthHalf ;

  __vr vrseq = _vel_vseq_vl(VLEN) ;
  __vm256 vm_s0, vm_s2 ;
  {
    __vr vry_s0  = _vel_vdivsl_vvsl(vrseq, inWidthHalf, VLEN) ;
    __vr vrx_s0  = _vel_vsubsl_vvvl(vrseq, _vel_vmulul_vsvl(inWidthHalf,vry_s0, VLEN), VLEN) ;
    vm_s0 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(outWidthHalf, vrx_s0, VLEN), VLEN) ; // condition(x<outWidthHalf)

    __vr vrseq2  = _vel_vaddsl_vsvl(inWidthHalf-1, vrseq, VLEN) ;
    __vr vry_s2  = _vel_vdivsl_vvsl(vrseq2, inWidthHalf, VLEN) ;
    __vr vrx_s2  = _vel_vsubsl_vvvl(vrseq2, _vel_vmulul_vsvl(inWidthHalf,vry_s2, VLEN), VLEN) ;
    vm_s2 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(outWidthHalf, vrx_s2, VLEN), VLEN) ; // condition(x<outWidthHalf)
  }

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
	     inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	     nY,
	     inWidthHalf, outWidthHalf,
	     vm_s0, vm_s2,
	     n, k );
	  k+=1 ;
	  break ;
	case 2 :
	  func<FLAYOUT, 2, ADDBIAS>(pIn, pKernel, pBias, pOut,
	     inChannel, inWidth, inHeight,
	     outChannel, outWidth, outHeight,
	     kernWidth, kernHeight,
	     inChannelGroup, outChannelGroup,
	     inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	     nY,
	     inWidthHalf, outWidthHalf,
	     vm_s0, vm_s2,
	     n, k );
	  k+=2 ;
	  break ;
	case 3 :
	  func<FLAYOUT, 3, ADDBIAS>(pIn, pKernel, pBias, pOut,
	     inChannel, inWidth, inHeight,
	     outChannel, outWidth, outHeight,
	     kernWidth, kernHeight,
	     inChannelGroup, outChannelGroup,
	     inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	     nY,
	     inWidthHalf, outWidthHalf,
	     vm_s0, vm_s2,
	     n, k );
	  k+=3 ;
	  break ;
	case 4 :
	  func<FLAYOUT, 4, ADDBIAS>(pIn, pKernel, pBias, pOut,
	     inChannel, inWidth, inHeight,
	     outChannel, outWidth, outHeight,
	     kernWidth, kernHeight,
	     inChannelGroup, outChannelGroup,
	     inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	     nY,
	     inWidthHalf, outWidthHalf,
	     vm_s0, vm_s2,
	     n, k );
	  k+=4 ;
	  break ;
	case 5 :
	  func<FLAYOUT, 5, ADDBIAS>(pIn, pKernel, pBias, pOut,
	     inChannel, inWidth, inHeight,
	     outChannel, outWidth, outHeight,
	     kernWidth, kernHeight,
	     inChannelGroup, outChannelGroup,
	     inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	     nY,
	     inWidthHalf, outWidthHalf,
	     vm_s0, vm_s2,
	     n, k );
	  k+=5 ;
	  break ;
	case 6 :
	  func<FLAYOUT, 6, ADDBIAS>(pIn, pKernel, pBias, pOut,
	     inChannel, inWidth, inHeight,
	     outChannel, outWidth, outHeight,
	     kernWidth, kernHeight,
	     inChannelGroup, outChannelGroup,
	     inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	     nY,
	     inWidthHalf, outWidthHalf,
	     vm_s0, vm_s2,
	     n, k );
	  k+=6 ;
	  break ;
	case 7 :
	  func<FLAYOUT, 7, ADDBIAS>(pIn, pKernel, pBias, pOut,
	     inChannel, inWidth, inHeight,
	     outChannel, outWidth, outHeight,
	     kernWidth, kernHeight,
	     inChannelGroup, outChannelGroup,
	     inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	     nY,
	     inWidthHalf, outWidthHalf,
	     vm_s0, vm_s2,
	     n, k );
	  k+=7 ;
	  break ;
	case 8 :
	  func<FLAYOUT, 8, ADDBIAS>(pIn, pKernel, pBias, pOut,
	     inChannel, inWidth, inHeight,
	     outChannel, outWidth, outHeight,
	     kernWidth, kernHeight,
	     inChannelGroup, outChannelGroup,
	     inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	     nY,
	     inWidthHalf, outWidthHalf,
	     vm_s0, vm_s2,
	     n, k );
	  k+=8 ;
	  break ;
	case 9 :
	  func<FLAYOUT, 9, ADDBIAS>(pIn, pKernel, pBias, pOut,
	     inChannel, inWidth, inHeight,
	     outChannel, outWidth, outHeight,
	     kernWidth, kernHeight,
	     inChannelGroup, outChannelGroup,
	     inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	     nY,
	     inWidthHalf, outWidthHalf,
	     vm_s0, vm_s2,
	     n, k );
	  k+=9 ;
	  break ;
	case 10 :
	  func<FLAYOUT, 10, ADDBIAS>(pIn, pKernel, pBias, pOut,
	     inChannel, inWidth, inHeight,
	     outChannel, outWidth, outHeight,
	     kernWidth, kernHeight,
	     inChannelGroup, outChannelGroup,
	     inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	     nY,
	     inWidthHalf, outWidthHalf,
	     vm_s0, vm_s2,
	     n, k );
	  k+=10 ;
	  break ;
	case 11 :
	  func<FLAYOUT, 11, ADDBIAS>(pIn, pKernel, pBias, pOut,
	     inChannel, inWidth, inHeight,
	     outChannel, outWidth, outHeight,
	     kernWidth, kernHeight,
	     inChannelGroup, outChannelGroup,
	     inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	     nY,
	     inWidthHalf, outWidthHalf,
	     vm_s0, vm_s2,
	     n, k );
	  k+=11 ;
	  break ;
	case 12 :
	  func<FLAYOUT, 12, ADDBIAS>(pIn, pKernel, pBias, pOut,
	     inChannel, inWidth, inHeight,
	     outChannel, outWidth, outHeight,
	     kernWidth, kernHeight,
	     inChannelGroup, outChannelGroup,
	     inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	     nY,
	     inWidthHalf, outWidthHalf,
	     vm_s0, vm_s2,
	     n, k );
	  k+=12 ;
	  break ;
	case 13 :
	  func<FLAYOUT, 13, ADDBIAS>(pIn, pKernel, pBias, pOut,
	     inChannel, inWidth, inHeight,
	     outChannel, outWidth, outHeight,
	     kernWidth, kernHeight,
	     inChannelGroup, outChannelGroup,
	     inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	     nY,
	     inWidthHalf, outWidthHalf,
	     vm_s0, vm_s2,
	     n, k );
	  k+=13 ;
	  break ;
	case 14 :
	  func<FLAYOUT, 14, ADDBIAS>(pIn, pKernel, pBias, pOut,
	     inChannel, inWidth, inHeight,
	     outChannel, outWidth, outHeight,
	     kernWidth, kernHeight,
	     inChannelGroup, outChannelGroup,
	     inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	     nY,
	     inWidthHalf, outWidthHalf,
	     vm_s0, vm_s2,
	     n, k );
	  k+=14 ;
	  break ;
	case 15 :
	  func<FLAYOUT, 15, ADDBIAS>(pIn, pKernel, pBias, pOut,
	     inChannel, inWidth, inHeight,
	     outChannel, outWidth, outHeight,
	     kernWidth, kernHeight,
	     inChannelGroup, outChannelGroup,
	     inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	     nY,
	     inWidthHalf, outWidthHalf,
	     vm_s0, vm_s2,
	     n, k );
	  k+=15 ;
	  break ;
	default :
	  break ;
	}
	for (; k < outChannelGroup; k+=16) {
	  func<FLAYOUT, 16, ADDBIAS>(pIn, pKernel, pBias, pOut,
	     inChannel, inWidth, inHeight,
	     outChannel, outWidth, outHeight,
	     kernWidth, kernHeight,
	     inChannelGroup, outChannelGroup,
	     inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	     nY,
	     inWidthHalf, outWidthHalf,
	     vm_s0, vm_s2,
	     n, k );
	} // outChannel
    } // group
  } // batch
}

extern "C" vednnError_t
vednnConvolutionForward_direct_dil1_str1_pad0_ker3_iw2XU256_ow2X_ioaligned(
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
//  const int64_t strideWidth    = pParamConv->strideWidth;		// must be 1
//  const int64_t strideHeight   = pParamConv->strideHeight;		// must be 1
//  const int64_t padWidth       = pParamConv->padWidth;		// must be 0
//  const int64_t padHeight      = pParamConv->padHeight;		// must be 0
//  const int64_t dilationWidth  = pParamConv->dilationWidth;		// must be 1
//  const int64_t dilationHeight = pParamConv->dilationHeight;		// must be 1

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
		 inChannelGroup, outChannelGroup ) ;
    }
    else {
      convloop<VEDNN_FILTER_LAYOUT_NCHW, true>(pIn, pKernel, pBias, pOut,
		 batch, group,
		 inChannel, inWidth, inHeight,
		 outChannel, outWidth, outHeight,
		 kernWidth, kernHeight,
		 inChannelGroup, outChannelGroup ) ;
    }
  }
  else {
    if( pDataBias == NULL ) {
      convloop<VEDNN_FILTER_LAYOUT_HWCN, false>(pIn, pKernel, pBias, pOut,
		 batch, group,
		 inChannel, inWidth, inHeight,
		 outChannel, outWidth, outHeight,
		 kernWidth, kernHeight,
		 inChannelGroup, outChannelGroup ) ;
    }
    else {
      convloop<VEDNN_FILTER_LAYOUT_HWCN, true>(pIn, pKernel, pBias, pOut,
		 batch, group,
		 inChannel, inWidth, inHeight,
		 outChannel, outWidth, outHeight,
		 kernWidth, kernHeight,
		 inChannelGroup, outChannelGroup ) ;
    }
  }

  return VEDNN_SUCCESS;
}

