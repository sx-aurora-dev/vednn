#include <stdint.h>
#include <stdlib.h>

#include "vednn.h"

#include "velintrin.h"
#define VLEN	(256)

#if 0
static inline void f1(
  const float * pIn, const int64_t inWidth, const int64_t inHeight,
  const float * pGOut, const int64_t gOutWidth, const int64_t gOutHeight,
  float * const pGKernel,
  const int64_t strideHeight, const int64_t strideWidth,
  const int64_t inChannelGroup, const int64_t inChannel, const int64_t gOutChannel,
  const int64_t inGroupOffset, const int64_t outGroupOffset, const int64_t kernGroupOffset,
  const int64_t batch, const int64_t k,
  const __vr vrhw
)
{
  int64_t c=0;
  if ( (inChannelGroup & 0x01) == 1 ) {
    const int64_t kernelIndex0 = kernGroupOffset + k     * inChannelGroup + c ;

#define INIT_VRSUM(CTOKEN) \
    __vr vrsum_##CTOKEN = _vel_vbrds_vsl(0UL, VLEN) ;

    INIT_VRSUM(c0)

    {

      const int64_t vl = gOutWidth * gOutHeight ;

      for (int64_t n=0; n<batch; n++) {
	const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;

	const int64_t gOutIndex0  = outGroupOffset + ((n * gOutChannel + k  ) * gOutHeight ) * gOutWidth ;

	__vr vrpin_c0 = _vel_vsfa_vvssl(vrhw, 2, (uint64_t)pInChannel, vl) ;
	__vr vrin_c0 = _vel_vgtu_vvssl(vrpin_c0, 0, 0, vl) ;

	__vr vrgout0 = _vel_vldu_vssl(4, pGOut+gOutIndex0, vl) ;

#define VFMAD_R1S1(CTOKEN) \
          vrsum_##CTOKEN = _vel_vfmads_vvvvvl(vrsum_##CTOKEN, vrin_##CTOKEN, vrgout0, vrsum_##CTOKEN, vl) ;

	  VFMAD_R1S1(c0)

      } // batch
    } // gOutHeight

#define VFSUM_STORE_R1S1(C,CTOKEN) \
    vrsum_##CTOKEN = _vel_vfsums_vvl(vrsum_##CTOKEN, VLEN) ; \
    _vel_vstu_vssl(vrsum_##CTOKEN, 4, pGKernel+kernelIndex0+(C), 1) ;

    VFSUM_STORE_R1S1(0,c0)

    c+=1;
  }
  if ( ((inChannelGroup >> 1) & 0x01) == 1 ) {
    const int64_t kernelIndex0 = kernGroupOffset + k     * inChannelGroup + c ;

    INIT_VRSUM(c0)
    INIT_VRSUM(c1)

    {

      const int64_t vl = gOutWidth * gOutHeight ;

      for (int64_t n=0; n<batch; n++) {
	const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;

	const int64_t gOutIndex0  = outGroupOffset + ((n * gOutChannel + k  ) * gOutHeight ) * gOutWidth ;

	__vr vrpin_c0 = _vel_vsfa_vvssl(vrhw, 2, (uint64_t)pInChannel, vl) ;
	__vr vrin_c0 = _vel_vgtu_vvssl(vrpin_c0, 0, 0, vl) ;
	__vr vrpin_c1 = _vel_vaddul_vsvl(1*4*inHeight*inWidth, vrpin_c0, vl) ;
	__vr vrin_c1 = _vel_vgtu_vvssl(vrpin_c1, 0, 0, vl) ;

	__vr vrgout0 = _vel_vldu_vssl(4, pGOut+gOutIndex0, vl) ;

	VFMAD_R1S1(c0)
	VFMAD_R1S1(c1)
      } // batch
    } // gOutHeight
    VFSUM_STORE_R1S1(0,c0)
    VFSUM_STORE_R1S1(1,c1)

    c+=2 ;
  }
  if ( ((inChannelGroup >> 1) & 0x01) == 1 ) {
    const int64_t kernelIndex0 = kernGroupOffset + k     * inChannelGroup + c ;

    INIT_VRSUM(c0)
    INIT_VRSUM(c1)
    INIT_VRSUM(c2)
    INIT_VRSUM(c3)

    {
      const int64_t vl = gOutWidth * gOutHeight ;

      for (int64_t n=0; n<batch; n++) {
	const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;

	const int64_t gOutIndex0  = outGroupOffset + ((n * gOutChannel + k  ) * gOutHeight ) * gOutWidth ;

	__vr vrpin_c0 = _vel_vsfa_vvssl(vrhw, 2, (uint64_t)pInChannel, vl) ;
	__vr vrin_c0 = _vel_vgtu_vvssl(vrpin_c0, 0, 0, vl) ;
	__vr vrpin_c1 = _vel_vaddul_vsvl(1*4*inHeight*inWidth, vrpin_c0, vl) ;
	__vr vrin_c1 = _vel_vgtu_vvssl(vrpin_c1, 0, 0, vl) ;
	__vr vrpin_c2 = _vel_vaddul_vsvl(2*4*inHeight*inWidth, vrpin_c0, vl) ;
	__vr vrin_c2 = _vel_vgtu_vvssl(vrpin_c2, 0, 0, vl) ;
	__vr vrpin_c3 = _vel_vaddul_vsvl(3*4*inHeight*inWidth, vrpin_c0, vl) ;
	__vr vrin_c3 = _vel_vgtu_vvssl(vrpin_c3, 0, 0, vl) ;

	__vr vrgout0 = _vel_vldu_vssl(4, pGOut+gOutIndex0, vl) ;

	VFMAD_R1S1(c0)
	VFMAD_R1S1(c1)
	VFMAD_R1S1(c2)
	VFMAD_R1S1(c3)
      } // batch
    } // gOutHeight


    VFSUM_STORE_R1S1(0,c0)
    VFSUM_STORE_R1S1(1,c1)
    VFSUM_STORE_R1S1(2,c2)
    VFSUM_STORE_R1S1(3,c3)

    c+=4 ;
  }
  for (; c<inChannelGroup; c+=8) {
    const int64_t kernelIndex0 = kernGroupOffset + k     * inChannelGroup + c ;

    INIT_VRSUM(c0)
    INIT_VRSUM(c1)
    INIT_VRSUM(c2)
    INIT_VRSUM(c3)
    INIT_VRSUM(c4)
    INIT_VRSUM(c5)
    INIT_VRSUM(c6)
    INIT_VRSUM(c7)
#undef INIT_VRSUM

    {

      const int64_t vl = gOutWidth * gOutHeight ;

      for (int64_t n=0; n<batch; n++) {
	const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;

	const int64_t gOutIndex0  = outGroupOffset + ((n * gOutChannel + k  ) * gOutHeight ) * gOutWidth ;

	__vr vrpin_c0 = _vel_vsfa_vvssl(vrhw, 2, (uint64_t)pInChannel, vl) ;
	__vr vrin_c0 = _vel_vgtu_vvssl(vrpin_c0, 0, 0, vl) ;
	__vr vrpin_c1 = _vel_vaddul_vsvl(1*4*inHeight*inWidth, vrpin_c0, vl) ;
	__vr vrin_c1 = _vel_vgtu_vvssl(vrpin_c1, 0, 0, vl) ;
	__vr vrpin_c2 = _vel_vaddul_vsvl(2*4*inHeight*inWidth, vrpin_c0, vl) ;
	__vr vrin_c2 = _vel_vgtu_vvssl(vrpin_c2, 0, 0, vl) ;
	__vr vrpin_c3 = _vel_vaddul_vsvl(3*4*inHeight*inWidth, vrpin_c0, vl) ;
	__vr vrin_c3 = _vel_vgtu_vvssl(vrpin_c3, 0, 0, vl) ;
	__vr vrpin_c4 = _vel_vaddul_vsvl(4*4*inHeight*inWidth, vrpin_c0, vl) ;
	__vr vrin_c4 = _vel_vgtu_vvssl(vrpin_c4, 0, 0, vl) ;
	__vr vrpin_c5 = _vel_vaddul_vsvl(5*4*inHeight*inWidth, vrpin_c0, vl) ;
	__vr vrin_c5 = _vel_vgtu_vvssl(vrpin_c5, 0, 0, vl) ;
	__vr vrpin_c6 = _vel_vaddul_vsvl(6*4*inHeight*inWidth, vrpin_c0, vl) ;
	__vr vrin_c6 = _vel_vgtu_vvssl(vrpin_c6, 0, 0, vl) ;
	__vr vrpin_c7 = _vel_vaddul_vsvl(7*4*inHeight*inWidth, vrpin_c0, vl) ;
	__vr vrin_c7 = _vel_vgtu_vvssl(vrpin_c7, 0, 0, vl) ;

	__vr vrgout0 = _vel_vldu_vssl(4, pGOut+gOutIndex0, vl) ;

	VFMAD_R1S1(c0)
	VFMAD_R1S1(c1)
	VFMAD_R1S1(c2)
	VFMAD_R1S1(c3)
	VFMAD_R1S1(c4)
	VFMAD_R1S1(c5)
	VFMAD_R1S1(c6)
	VFMAD_R1S1(c7)
#undef VFMAD_R1S1
      } // batch
    } // gOutHeight

    VFSUM_STORE_R1S1(0,c0)
    VFSUM_STORE_R1S1(1,c1)
    VFSUM_STORE_R1S1(2,c2)
    VFSUM_STORE_R1S1(3,c3)
    VFSUM_STORE_R1S1(4,c4)
    VFSUM_STORE_R1S1(5,c5)
    VFSUM_STORE_R1S1(6,c6)
    VFSUM_STORE_R1S1(7,c7)
#undef VFSUM_STORE_R1S1
  }
}


static inline void f2(
  const float * pIn, const int64_t inWidth, const int64_t inHeight,
  const float * pGOut, const int64_t gOutWidth, const int64_t gOutHeight,
  float * const pGKernel,
  const int64_t strideHeight, const int64_t strideWidth,
  const int64_t inChannelGroup, const int64_t inChannel, const int64_t gOutChannel,
  const int64_t inGroupOffset, const int64_t outGroupOffset, const int64_t kernGroupOffset,
  const int64_t batch, const int64_t k,
  const __vr vrhw
)
{
  int64_t c=0;
  if ( (inChannelGroup & 0x01) == 1 ) {
    const int64_t kernelIndex0 = kernGroupOffset + k     * inChannelGroup + c ;
    const int64_t kernelIndex1 = kernGroupOffset + (k+1) * inChannelGroup + c ;

#define INIT_VRSUM(CTOKEN) \
    __vr vrsum01_##CTOKEN = _vel_vbrdl_vsl(0UL, VLEN) ;

    INIT_VRSUM(c0)

    {

      const int64_t vl = gOutWidth * gOutHeight ;

      for (int64_t n=0; n<batch; n++) {
	const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;

	const int64_t gOutIndex0  = outGroupOffset + ((n * gOutChannel + k  ) * gOutHeight ) * gOutWidth ;
	const int64_t gOutIndex1  = outGroupOffset + ((n * gOutChannel + k+1) * gOutHeight ) * gOutWidth ;

	__vr vrpin_c0 = _vel_vsfa_vvssl(vrhw, 2, (uint64_t)pInChannel, vl) ;
	__vr vrin_c0 = _vel_vgtu_vvssl(vrpin_c0, 0, 0, vl) ;

	__vr vrgout0 = _vel_vldu_vssl(4, pGOut+gOutIndex0, vl) ;
	__vr vrgout1 = _vel_vldu_vssl(4, pGOut+gOutIndex1, vl) ;

	__vr vrgout01 = _vel_vshf_vvvsl(vrgout0, vrgout1, VE_VSHUFFLE_YUZU, vl) ;

#define VFMAD_R1S1(CTOKEN) \
          __vr vrinP_##CTOKEN = _vel_vshf_vvvsl(vrin_##CTOKEN, vrin_##CTOKEN, VE_VSHUFFLE_YUZU, vl) ; \
          vrsum01_##CTOKEN = _vel_pvfmad_vvvvvl(vrsum01_##CTOKEN, vrinP_##CTOKEN, vrgout01, vrsum01_##CTOKEN, vl) ;

	  VFMAD_R1S1(c0)

      } // batch
    } // gOutHeight

#define VFSUM_STORE_R1S1(C,CTOKEN) \
    __vr vrsum0_##CTOKEN = _vel_vfsums_vvl(vrsum01_##CTOKEN, VLEN) ; \
    __vr vrsum1_##CTOKEN = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsum01_##CTOKEN,32, VLEN), VLEN); \
    _vel_vstu_vssl(vrsum0_##CTOKEN, 4, pGKernel+kernelIndex0+(C), 1) ; \
    _vel_vstu_vssl(vrsum1_##CTOKEN, 4, pGKernel+kernelIndex1+(C), 1) ;

    VFSUM_STORE_R1S1(0,c0)

    c+=1;
  }
  if ( ((inChannelGroup >> 1) & 0x01) == 1 ) {
    const int64_t kernelIndex0 = kernGroupOffset + k     * inChannelGroup + c ;
    const int64_t kernelIndex1 = kernGroupOffset + (k+1) * inChannelGroup + c ;

    INIT_VRSUM(c0)
    INIT_VRSUM(c1)

    {

      const int64_t vl = gOutWidth * gOutHeight ;

      for (int64_t n=0; n<batch; n++) {
	const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;

	const int64_t gOutIndex0  = outGroupOffset + ((n * gOutChannel + k  ) * gOutHeight ) * gOutWidth ;
	const int64_t gOutIndex1  = outGroupOffset + ((n * gOutChannel + k+1) * gOutHeight ) * gOutWidth ;

	__vr vrpin_c0 = _vel_vsfa_vvssl(vrhw, 2, (uint64_t)pInChannel, vl) ;
	__vr vrin_c0 = _vel_vgtu_vvssl(vrpin_c0, 0, 0, vl) ;
	__vr vrpin_c1 = _vel_vaddul_vsvl(1*4*inHeight*inWidth, vrpin_c0, vl) ;
	__vr vrin_c1 = _vel_vgtu_vvssl(vrpin_c1, 0, 0, vl) ;

	__vr vrgout0 = _vel_vldu_vssl(4, pGOut+gOutIndex0, vl) ;
	__vr vrgout1 = _vel_vldu_vssl(4, pGOut+gOutIndex1, vl) ;

	__vr vrgout01 = _vel_vshf_vvvsl(vrgout0, vrgout1, VE_VSHUFFLE_YUZU, vl) ;

	VFMAD_R1S1(c0)
	VFMAD_R1S1(c1)
      } // batch
    } // gOutHeight
    VFSUM_STORE_R1S1(0,c0)
    VFSUM_STORE_R1S1(1,c1)

    c+=2 ;
  }
  if ( ((inChannelGroup >> 1) & 0x01) == 1 ) {
    const int64_t kernelIndex0 = kernGroupOffset + k     * inChannelGroup + c ;
    const int64_t kernelIndex1 = kernGroupOffset + (k+1) * inChannelGroup + c ;

    INIT_VRSUM(c0)
    INIT_VRSUM(c1)
    INIT_VRSUM(c2)
    INIT_VRSUM(c3)

    {

      const int64_t vl = gOutWidth * gOutHeight ;

      for (int64_t n=0; n<batch; n++) {
	const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;

	const int64_t gOutIndex0  = outGroupOffset + ((n * gOutChannel + k  ) * gOutHeight ) * gOutWidth ;
	const int64_t gOutIndex1  = outGroupOffset + ((n * gOutChannel + k+1) * gOutHeight ) * gOutWidth ;

	__vr vrpin_c0 = _vel_vsfa_vvssl(vrhw, 2, (uint64_t)pInChannel, vl) ;
	__vr vrin_c0 = _vel_vgtu_vvssl(vrpin_c0, 0, 0, vl) ;
	__vr vrpin_c1 = _vel_vaddul_vsvl(1*4*inHeight*inWidth, vrpin_c0, vl) ;
	__vr vrin_c1 = _vel_vgtu_vvssl(vrpin_c1, 0, 0, vl) ;
	__vr vrpin_c2 = _vel_vaddul_vsvl(2*4*inHeight*inWidth, vrpin_c0, vl) ;
	__vr vrin_c2 = _vel_vgtu_vvssl(vrpin_c2, 0, 0, vl) ;
	__vr vrpin_c3 = _vel_vaddul_vsvl(3*4*inHeight*inWidth, vrpin_c0, vl) ;
	__vr vrin_c3 = _vel_vgtu_vvssl(vrpin_c3, 0, 0, vl) ;

	__vr vrgout0 = _vel_vldu_vssl(4, pGOut+gOutIndex0, vl) ;
	__vr vrgout1 = _vel_vldu_vssl(4, pGOut+gOutIndex1, vl) ;

	__vr vrgout01 = _vel_vshf_vvvsl(vrgout0, vrgout1, VE_VSHUFFLE_YUZU, vl) ;

	VFMAD_R1S1(c0)
	VFMAD_R1S1(c1)
	VFMAD_R1S1(c2)
	VFMAD_R1S1(c3)
      } // batch
    } // gOutHeight


    VFSUM_STORE_R1S1(0,c0)
    VFSUM_STORE_R1S1(1,c1)
    VFSUM_STORE_R1S1(2,c2)
    VFSUM_STORE_R1S1(3,c3)

    c+=4 ;
  }
  for (; c<inChannelGroup; c+=8) {
    const int64_t kernelIndex0 = kernGroupOffset + k     * inChannelGroup + c ;
    const int64_t kernelIndex1 = kernGroupOffset + (k+1) * inChannelGroup + c ;

    INIT_VRSUM(c0)
    INIT_VRSUM(c1)
    INIT_VRSUM(c2)
    INIT_VRSUM(c3)
    INIT_VRSUM(c4)
    INIT_VRSUM(c5)
    INIT_VRSUM(c6)
    INIT_VRSUM(c7)
#undef INIT_VRSUM

    {

      const int64_t vl = gOutWidth * gOutHeight ;

      for (int64_t n=0; n<batch; n++) {
	const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;

	const int64_t gOutIndex0  = outGroupOffset + ((n * gOutChannel + k  ) * gOutHeight ) * gOutWidth ;
	const int64_t gOutIndex1  = outGroupOffset + ((n * gOutChannel + k+1) * gOutHeight ) * gOutWidth ;

	__vr vrpin_c0 = _vel_vsfa_vvssl(vrhw, 2, (uint64_t)pInChannel, vl) ;
	__vr vrin_c0 = _vel_vgtu_vvssl(vrpin_c0, 0, 0, vl) ;
	__vr vrpin_c1 = _vel_vaddul_vsvl(1*4*inHeight*inWidth, vrpin_c0, vl) ;
	__vr vrin_c1 = _vel_vgtu_vvssl(vrpin_c1, 0, 0, vl) ;
	__vr vrpin_c2 = _vel_vaddul_vsvl(2*4*inHeight*inWidth, vrpin_c0, vl) ;
	__vr vrin_c2 = _vel_vgtu_vvssl(vrpin_c2, 0, 0, vl) ;
	__vr vrpin_c3 = _vel_vaddul_vsvl(3*4*inHeight*inWidth, vrpin_c0, vl) ;
	__vr vrin_c3 = _vel_vgtu_vvssl(vrpin_c3, 0, 0, vl) ;
	__vr vrpin_c4 = _vel_vaddul_vsvl(4*4*inHeight*inWidth, vrpin_c0, vl) ;
	__vr vrin_c4 = _vel_vgtu_vvssl(vrpin_c4, 0, 0, vl) ;
	__vr vrpin_c5 = _vel_vaddul_vsvl(5*4*inHeight*inWidth, vrpin_c0, vl) ;
	__vr vrin_c5 = _vel_vgtu_vvssl(vrpin_c5, 0, 0, vl) ;
	__vr vrpin_c6 = _vel_vaddul_vsvl(6*4*inHeight*inWidth, vrpin_c0, vl) ;
	__vr vrin_c6 = _vel_vgtu_vvssl(vrpin_c6, 0, 0, vl) ;
	__vr vrpin_c7 = _vel_vaddul_vsvl(7*4*inHeight*inWidth, vrpin_c0, vl) ;
	__vr vrin_c7 = _vel_vgtu_vvssl(vrpin_c7, 0, 0, vl) ;

	__vr vrgout0 = _vel_vldu_vssl(4, pGOut+gOutIndex0, vl) ;
	__vr vrgout1 = _vel_vldu_vssl(4, pGOut+gOutIndex1, vl) ;

	__vr vrgout01 = _vel_vshf_vvvsl(vrgout0, vrgout1, VE_VSHUFFLE_YUZU, vl) ;

	VFMAD_R1S1(c0)
	VFMAD_R1S1(c1)
	VFMAD_R1S1(c2)
	VFMAD_R1S1(c3)
	VFMAD_R1S1(c4)
	VFMAD_R1S1(c5)
	VFMAD_R1S1(c6)
	VFMAD_R1S1(c7)
#undef VFMAD_R1S1
      } // batch
    } // gOutHeight

    VFSUM_STORE_R1S1(0,c0)
    VFSUM_STORE_R1S1(1,c1)
    VFSUM_STORE_R1S1(2,c2)
    VFSUM_STORE_R1S1(3,c3)
    VFSUM_STORE_R1S1(4,c4)
    VFSUM_STORE_R1S1(5,c5)
    VFSUM_STORE_R1S1(6,c6)
    VFSUM_STORE_R1S1(7,c7)
#undef VFSUM_STORE_R1S1
  }
}


static inline void f4(
  const float * pIn, const int64_t inWidth, const int64_t inHeight,
  const float * pGOut, const int64_t gOutWidth, const int64_t gOutHeight,
  float * const pGKernel,
  const int64_t strideHeight, const int64_t strideWidth,
  const int64_t inChannelGroup, const int64_t inChannel, const int64_t gOutChannel,
  const int64_t inGroupOffset, const int64_t outGroupOffset, const int64_t kernGroupOffset,
  const int64_t batch, const int64_t k,
  const __vr vrhw, const __vm256 vm_k0, const __vm256 vm_k1
)
{
  int64_t c=0;
  if ( (inChannelGroup & 0x01) == 1 ) {
    const int64_t kernelIndex0 = kernGroupOffset + k     * inChannelGroup + c ;
    const int64_t kernelIndex1 = kernGroupOffset + (k+1) * inChannelGroup + c ;
    const int64_t kernelIndex2 = kernGroupOffset + (k+2) * inChannelGroup + c ;
    const int64_t kernelIndex3 = kernGroupOffset + (k+3) * inChannelGroup + c ;

#define INIT_VRSUM(CTOKEN) \
    __vr vrsum0123_##CTOKEN = _vel_vbrdl_vsl(0UL, VLEN) ;

    INIT_VRSUM(c0)

    {
      const int64_t vl = gOutWidth * gOutHeight ;

      for (int64_t n=0; n<batch; n++) {
	const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;

	const int64_t gOutIndex01  = outGroupOffset + ((n * gOutChannel + k  ) * gOutHeight ) * gOutWidth ;
	const int64_t gOutIndex23  = outGroupOffset + ((n * gOutChannel + k+2) * gOutHeight ) * gOutWidth ;

	__vr vrpin_c0 = _vel_vsfa_vvssl(vrhw, 2, (uint64_t)pInChannel, 2*vl) ;
	__vr vrin_c0 = _vel_vgtu_vvssl(vrpin_c0, 0, 0, 2*vl) ;

	__vr vrgout01 = _vel_vldu_vssl(4, pGOut+gOutIndex01, 2*vl) ;
	__vr vrgout23 = _vel_vldu_vssl(4, pGOut+gOutIndex23, 2*vl) ;

	__vr vrgout0123 = _vel_vshf_vvvsl(vrgout01, vrgout23, VE_VSHUFFLE_YUZU, 2*vl) ;

#define VFMAD_R1S1(CTOKEN) \
          __vr vrinP_##CTOKEN = _vel_vshf_vvvsl(vrin_##CTOKEN, vrin_##CTOKEN, VE_VSHUFFLE_YUZU, 2*vl) ;  \
          vrsum0123_##CTOKEN = _vel_pvfmad_vvvvvl(vrsum0123_##CTOKEN, vrinP_##CTOKEN, vrgout0123, vrsum0123_##CTOKEN, 2*vl) ;

	  VFMAD_R1S1(c0)

      } // batch
    } // gOutHeight

#define VFSUM_STORE_R1S1(C,CTOKEN) \
    __vr vrsum0_##CTOKEN = _vel_vfsums_vvml(vrsum0123_##CTOKEN, vm_k0, VLEN) ; \
    __vr vrsum1_##CTOKEN = _vel_vfsums_vvml(vrsum0123_##CTOKEN, vm_k1, VLEN) ; \
    __vr vrsum2_##CTOKEN = _vel_vfsums_vvml(_vel_vsll_vvsl(vrsum0123_##CTOKEN,32, VLEN), vm_k0, VLEN); \
    __vr vrsum3_##CTOKEN = _vel_vfsums_vvml(_vel_vsll_vvsl(vrsum0123_##CTOKEN,32, VLEN), vm_k1, VLEN); \
    _vel_vstu_vssl(vrsum0_##CTOKEN, 4, pGKernel+kernelIndex0+(C), 1) ; \
    _vel_vstu_vssl(vrsum1_##CTOKEN, 4, pGKernel+kernelIndex1+(C), 1) ; \
    _vel_vstu_vssl(vrsum2_##CTOKEN, 4, pGKernel+kernelIndex2+(C), 1) ; \
    _vel_vstu_vssl(vrsum3_##CTOKEN, 4, pGKernel+kernelIndex3+(C), 1) ;

    VFSUM_STORE_R1S1(0,c0)

    c+=1;
  }
  if ( ((inChannelGroup >> 1) & 0x01) == 1 ) {
    const int64_t kernelIndex0 = kernGroupOffset + k     * inChannelGroup + c ;
    const int64_t kernelIndex1 = kernGroupOffset + (k+1) * inChannelGroup + c ;
    const int64_t kernelIndex2 = kernGroupOffset + (k+2) * inChannelGroup + c ;
    const int64_t kernelIndex3 = kernGroupOffset + (k+3) * inChannelGroup + c ;

    INIT_VRSUM(c0)
    INIT_VRSUM(c1)

    {
      const int64_t vl = gOutWidth * gOutHeight ;

      for (int64_t n=0; n<batch; n++) {
	const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;

	const int64_t gOutIndex01  = outGroupOffset + ((n * gOutChannel + k  ) * gOutHeight ) * gOutWidth ;
	const int64_t gOutIndex23  = outGroupOffset + ((n * gOutChannel + k+2) * gOutHeight ) * gOutWidth ;

	__vr vrpin_c0 = _vel_vsfa_vvssl(vrhw, 2, (uint64_t)pInChannel, 2*vl) ;
	__vr vrin_c0 = _vel_vgtu_vvssl(vrpin_c0, 0, 0, 2*vl) ;
	__vr vrpin_c1 = _vel_vaddul_vsvl(1*4*inHeight*inWidth, vrpin_c0, 2*vl) ;
	__vr vrin_c1 = _vel_vgtu_vvssl(vrpin_c1, 0, 0, 2*vl) ;

	__vr vrgout01 = _vel_vldu_vssl(4, pGOut+gOutIndex01, 2*vl) ;
	__vr vrgout23 = _vel_vldu_vssl(4, pGOut+gOutIndex23, 2*vl) ;

	__vr vrgout0123 = _vel_vshf_vvvsl(vrgout01, vrgout23, VE_VSHUFFLE_YUZU, 2*vl) ;


	VFMAD_R1S1(c0)
	VFMAD_R1S1(c1)

      } // batch
    } // gOutHeight

    VFSUM_STORE_R1S1(0,c0)
    VFSUM_STORE_R1S1(1,c1)

    c+=2 ;
  }
  if ( ((inChannelGroup >> 2) & 0x01) == 1 ) {
    const int64_t kernelIndex0 = kernGroupOffset + k     * inChannelGroup + c ;
    const int64_t kernelIndex1 = kernGroupOffset + (k+1) * inChannelGroup + c ;
    const int64_t kernelIndex2 = kernGroupOffset + (k+2) * inChannelGroup + c ;
    const int64_t kernelIndex3 = kernGroupOffset + (k+3) * inChannelGroup + c ;

    INIT_VRSUM(c0)
    INIT_VRSUM(c1)
    INIT_VRSUM(c2)
    INIT_VRSUM(c3)

    {
      const int64_t vl = gOutWidth * gOutHeight ;

      for (int64_t n=0; n<batch; n++) {
	const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;

	const int64_t gOutIndex01  = outGroupOffset + ((n * gOutChannel + k  ) * gOutHeight ) * gOutWidth ;
	const int64_t gOutIndex23  = outGroupOffset + ((n * gOutChannel + k+2) * gOutHeight ) * gOutWidth ;

	__vr vrpin_c0 = _vel_vsfa_vvssl(vrhw, 2, (uint64_t)pInChannel, 2*vl) ;
	__vr vrin_c0 = _vel_vgtu_vvssl(vrpin_c0, 0, 0, 2*vl) ;
	__vr vrpin_c1 = _vel_vaddul_vsvl(1*4*inHeight*inWidth, vrpin_c0, 2*vl) ;
	__vr vrin_c1 = _vel_vgtu_vvssl(vrpin_c1, 0, 0, 2*vl) ;
	__vr vrpin_c2 = _vel_vaddul_vsvl(2*4*inHeight*inWidth, vrpin_c0, 2*vl) ;
	__vr vrin_c2 = _vel_vgtu_vvssl(vrpin_c2, 0, 0, 2*vl) ;
	__vr vrpin_c3 = _vel_vaddul_vsvl(3*4*inHeight*inWidth, vrpin_c0, 2*vl) ;
	__vr vrin_c3 = _vel_vgtu_vvssl(vrpin_c3, 0, 0, 2*vl) ;

	__vr vrgout01 = _vel_vldu_vssl(4, pGOut+gOutIndex01, 2*vl) ;
	__vr vrgout23 = _vel_vldu_vssl(4, pGOut+gOutIndex23, 2*vl) ;

	__vr vrgout0123 = _vel_vshf_vvvsl(vrgout01, vrgout23, VE_VSHUFFLE_YUZU, 2*vl) ;

	VFMAD_R1S1(c0)
	VFMAD_R1S1(c1)
	VFMAD_R1S1(c2)
	VFMAD_R1S1(c3)

      } // batch
    } // gOutHeight

    VFSUM_STORE_R1S1(0,c0)
    VFSUM_STORE_R1S1(1,c1)
    VFSUM_STORE_R1S1(2,c2)
    VFSUM_STORE_R1S1(3,c3)

    c+=4 ;
  }
  for (; c<inChannelGroup; c+=8) {
    const int64_t kernelIndex0 = kernGroupOffset + k     * inChannelGroup + c ;
    const int64_t kernelIndex1 = kernGroupOffset + (k+1) * inChannelGroup + c ;
    const int64_t kernelIndex2 = kernGroupOffset + (k+2) * inChannelGroup + c ;
    const int64_t kernelIndex3 = kernGroupOffset + (k+3) * inChannelGroup + c ;

    INIT_VRSUM(c0)
    INIT_VRSUM(c1)
    INIT_VRSUM(c2)
    INIT_VRSUM(c3)
    INIT_VRSUM(c4)
    INIT_VRSUM(c5)
    INIT_VRSUM(c6)
    INIT_VRSUM(c7)
#undef INIT_VRSUM

    {
      const int64_t vl = gOutWidth * gOutHeight ;

      for (int64_t n=0; n<batch; n++) {
	const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;

	const int64_t gOutIndex01  = outGroupOffset + ((n * gOutChannel + k  ) * gOutHeight ) * gOutWidth ;
	const int64_t gOutIndex23  = outGroupOffset + ((n * gOutChannel + k+2) * gOutHeight ) * gOutWidth ;

	__vr vrpin_c0 = _vel_vsfa_vvssl(vrhw, 2, (uint64_t)pInChannel, 2*vl) ;
	__vr vrin_c0 = _vel_vgtu_vvssl(vrpin_c0, 0, 0, 2*vl) ;
	__vr vrpin_c1 = _vel_vaddul_vsvl(1*4*inHeight*inWidth, vrpin_c0, 2*vl) ;
	__vr vrin_c1 = _vel_vgtu_vvssl(vrpin_c1, 0, 0, 2*vl) ;
	__vr vrpin_c2 = _vel_vaddul_vsvl(2*4*inHeight*inWidth, vrpin_c0, 2*vl) ;
	__vr vrin_c2 = _vel_vgtu_vvssl(vrpin_c2, 0, 0, 2*vl) ;
	__vr vrpin_c3 = _vel_vaddul_vsvl(3*4*inHeight*inWidth, vrpin_c0, 2*vl) ;
	__vr vrin_c3 = _vel_vgtu_vvssl(vrpin_c3, 0, 0, 2*vl) ;
	__vr vrpin_c4 = _vel_vaddul_vsvl(4*4*inHeight*inWidth, vrpin_c0, 2*vl) ;
	__vr vrin_c4 = _vel_vgtu_vvssl(vrpin_c4, 0, 0, 2*vl) ;
	__vr vrpin_c5 = _vel_vaddul_vsvl(5*4*inHeight*inWidth, vrpin_c0, 2*vl) ;
	__vr vrin_c5 = _vel_vgtu_vvssl(vrpin_c5, 0, 0, 2*vl) ;
	__vr vrpin_c6 = _vel_vaddul_vsvl(6*4*inHeight*inWidth, vrpin_c0, 2*vl) ;
	__vr vrin_c6 = _vel_vgtu_vvssl(vrpin_c6, 0, 0, 2*vl) ;
	__vr vrpin_c7 = _vel_vaddul_vsvl(7*4*inHeight*inWidth, vrpin_c0, 2*vl) ;
	__vr vrin_c7 = _vel_vgtu_vvssl(vrpin_c7, 0, 0, 2*vl) ;

	__vr vrgout01 = _vel_vldu_vssl(4, pGOut+gOutIndex01, 2*vl) ;
	__vr vrgout23 = _vel_vldu_vssl(4, pGOut+gOutIndex23, 2*vl) ;

	__vr vrgout0123 = _vel_vshf_vvvsl(vrgout01, vrgout23, VE_VSHUFFLE_YUZU, 2*vl) ;

	VFMAD_R1S1(c0)
	VFMAD_R1S1(c1)
	VFMAD_R1S1(c2)
	VFMAD_R1S1(c3)
	VFMAD_R1S1(c4)
	VFMAD_R1S1(c5)
	VFMAD_R1S1(c6)
	VFMAD_R1S1(c7)
#undef VFMAD_R1S1
      } // batch
    } // gOutHeight

    VFSUM_STORE_R1S1(0,c0)
    VFSUM_STORE_R1S1(1,c1)
    VFSUM_STORE_R1S1(2,c2)
    VFSUM_STORE_R1S1(3,c3)
    VFSUM_STORE_R1S1(4,c4)
    VFSUM_STORE_R1S1(5,c5)
    VFSUM_STORE_R1S1(6,c6)
    VFSUM_STORE_R1S1(7,c7)
#undef VFSUM_STORE_R1S1
  }
}


static inline void f8(
  const float * pIn, const int64_t inWidth, const int64_t inHeight,
  const float * pGOut, const int64_t gOutWidth, const int64_t gOutHeight,
  float * const pGKernel,
  const int64_t strideHeight, const int64_t strideWidth,
  const int64_t inChannelGroup, const int64_t inChannel, const int64_t gOutChannel,
  const int64_t inGroupOffset, const int64_t outGroupOffset, const int64_t kernGroupOffset,
  const int64_t batch, const int64_t k,
  const __vr vrhw, const __vm256 vm_k0, const __vm256 vm_k1
)
{
  int64_t c=0;
  if ( (inChannelGroup & 0x01) == 1 ) {
    const int64_t kernelIndex0 = kernGroupOffset + k     * inChannelGroup + c ;
    const int64_t kernelIndex1 = kernGroupOffset + (k+1) * inChannelGroup + c ;
    const int64_t kernelIndex2 = kernGroupOffset + (k+2) * inChannelGroup + c ;
    const int64_t kernelIndex3 = kernGroupOffset + (k+3) * inChannelGroup + c ;
    const int64_t kernelIndex4 = kernGroupOffset + (k+4) * inChannelGroup + c ;
    const int64_t kernelIndex5 = kernGroupOffset + (k+5) * inChannelGroup + c ;
    const int64_t kernelIndex6 = kernGroupOffset + (k+6) * inChannelGroup + c ;
    const int64_t kernelIndex7 = kernGroupOffset + (k+7) * inChannelGroup + c ;

#define INIT_VRSUM(CTOKEN) \
    __vr vrsum0123_##CTOKEN = _vel_vbrdl_vsl(0UL, VLEN) ; \
    __vr vrsum4567_##CTOKEN = _vel_vbrdl_vsl(0UL, VLEN) ;

    INIT_VRSUM(c0)

    {
      const int64_t vl = gOutWidth * gOutHeight ;

      for (int64_t n=0; n<batch; n++) {
	const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;

	const int64_t gOutIndex01  = outGroupOffset + ((n * gOutChannel + k  ) * gOutHeight ) * gOutWidth ;
	const int64_t gOutIndex23  = outGroupOffset + ((n * gOutChannel + k+2) * gOutHeight ) * gOutWidth ;
	const int64_t gOutIndex45  = outGroupOffset + ((n * gOutChannel + k+4) * gOutHeight ) * gOutWidth ;
	const int64_t gOutIndex67  = outGroupOffset + ((n * gOutChannel + k+6) * gOutHeight ) * gOutWidth ;

	__vr vrpin_c0 = _vel_vsfa_vvssl(vrhw, 2, (uint64_t)pInChannel, 2*vl) ;
	__vr vrin_c0 = _vel_vgtu_vvssl(vrpin_c0, 0, 0, 2*vl) ;

	__vr vrgout01 = _vel_vldu_vssl(4, pGOut+gOutIndex01, 2*vl) ;
	__vr vrgout23 = _vel_vldu_vssl(4, pGOut+gOutIndex23, 2*vl) ;
	__vr vrgout45 = _vel_vldu_vssl(4, pGOut+gOutIndex45, 2*vl) ;
	__vr vrgout67 = _vel_vldu_vssl(4, pGOut+gOutIndex67, 2*vl) ;

	__vr vrgout0123 = _vel_vshf_vvvsl(vrgout01, vrgout23, VE_VSHUFFLE_YUZU, 2*vl) ;
	__vr vrgout4567 = _vel_vshf_vvvsl(vrgout45, vrgout67, VE_VSHUFFLE_YUZU, 2*vl) ;

#define VFMAD_R1S1(CTOKEN) \
          __vr vrinP_##CTOKEN = _vel_vshf_vvvsl(vrin_##CTOKEN, vrin_##CTOKEN, VE_VSHUFFLE_YUZU, 2*vl) ;  \
          vrsum0123_##CTOKEN = _vel_pvfmad_vvvvvl(vrsum0123_##CTOKEN, vrinP_##CTOKEN, vrgout0123, vrsum0123_##CTOKEN, 2*vl) ; \
          vrsum4567_##CTOKEN = _vel_pvfmad_vvvvvl(vrsum4567_##CTOKEN, vrinP_##CTOKEN, vrgout4567, vrsum4567_##CTOKEN, 2*vl) ;

	  VFMAD_R1S1(c0)

      } // batch
    } // gOutHeight

#define VFSUM_STORE_R1S1(C,CTOKEN) \
    __vr vrsum0_##CTOKEN = _vel_vfsums_vvml(vrsum0123_##CTOKEN, vm_k0, VLEN) ; \
    __vr vrsum1_##CTOKEN = _vel_vfsums_vvml(vrsum0123_##CTOKEN, vm_k1, VLEN) ; \
    __vr vrsum2_##CTOKEN = _vel_vfsums_vvml(_vel_vsll_vvsl(vrsum0123_##CTOKEN,32, VLEN), vm_k0, VLEN); \
    __vr vrsum3_##CTOKEN = _vel_vfsums_vvml(_vel_vsll_vvsl(vrsum0123_##CTOKEN,32, VLEN), vm_k1, VLEN); \
    __vr vrsum4_##CTOKEN = _vel_vfsums_vvml(vrsum4567_##CTOKEN, vm_k0, VLEN) ; \
    __vr vrsum5_##CTOKEN = _vel_vfsums_vvml(vrsum4567_##CTOKEN, vm_k1, VLEN) ; \
    __vr vrsum6_##CTOKEN = _vel_vfsums_vvml(_vel_vsll_vvsl(vrsum4567_##CTOKEN,32, VLEN), vm_k0, VLEN); \
    __vr vrsum7_##CTOKEN = _vel_vfsums_vvml(_vel_vsll_vvsl(vrsum4567_##CTOKEN,32, VLEN), vm_k1, VLEN); \
    _vel_vstu_vssl(vrsum0_##CTOKEN, 4, pGKernel+kernelIndex0+(C), 1) ; \
    _vel_vstu_vssl(vrsum1_##CTOKEN, 4, pGKernel+kernelIndex1+(C), 1) ; \
    _vel_vstu_vssl(vrsum2_##CTOKEN, 4, pGKernel+kernelIndex2+(C), 1) ; \
    _vel_vstu_vssl(vrsum3_##CTOKEN, 4, pGKernel+kernelIndex3+(C), 1) ; \
    _vel_vstu_vssl(vrsum4_##CTOKEN, 4, pGKernel+kernelIndex4+(C), 1) ; \
    _vel_vstu_vssl(vrsum5_##CTOKEN, 4, pGKernel+kernelIndex5+(C), 1) ; \
    _vel_vstu_vssl(vrsum6_##CTOKEN, 4, pGKernel+kernelIndex6+(C), 1) ; \
    _vel_vstu_vssl(vrsum7_##CTOKEN, 4, pGKernel+kernelIndex7+(C), 1) ;

    VFSUM_STORE_R1S1(0,c0)

    c+=1;
  }
  if ( ((inChannelGroup >> 1) & 0x01) == 1 ) {
    const int64_t kernelIndex0 = kernGroupOffset + k     * inChannelGroup + c ;
    const int64_t kernelIndex1 = kernGroupOffset + (k+1) * inChannelGroup + c ;
    const int64_t kernelIndex2 = kernGroupOffset + (k+2) * inChannelGroup + c ;
    const int64_t kernelIndex3 = kernGroupOffset + (k+3) * inChannelGroup + c ;
    const int64_t kernelIndex4 = kernGroupOffset + (k+4) * inChannelGroup + c ;
    const int64_t kernelIndex5 = kernGroupOffset + (k+5) * inChannelGroup + c ;
    const int64_t kernelIndex6 = kernGroupOffset + (k+6) * inChannelGroup + c ;
    const int64_t kernelIndex7 = kernGroupOffset + (k+7) * inChannelGroup + c ;

    INIT_VRSUM(c0)
    INIT_VRSUM(c1)

    {
      const int64_t vl = gOutWidth * gOutHeight ;

      for (int64_t n=0; n<batch; n++) {
	const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;

	const int64_t gOutIndex01  = outGroupOffset + ((n * gOutChannel + k  ) * gOutHeight ) * gOutWidth ;
	const int64_t gOutIndex23  = outGroupOffset + ((n * gOutChannel + k+2) * gOutHeight ) * gOutWidth ;
	const int64_t gOutIndex45  = outGroupOffset + ((n * gOutChannel + k+4) * gOutHeight ) * gOutWidth ;
	const int64_t gOutIndex67  = outGroupOffset + ((n * gOutChannel + k+6) * gOutHeight ) * gOutWidth ;

	__vr vrpin_c0 = _vel_vsfa_vvssl(vrhw, 2, (uint64_t)pInChannel, 2*vl) ;
	__vr vrin_c0 = _vel_vgtu_vvssl(vrpin_c0, 0, 0, 2*vl) ;
	__vr vrpin_c1 = _vel_vaddul_vsvl(1*4*inHeight*inWidth, vrpin_c0, 2*vl) ;
	__vr vrin_c1 = _vel_vgtu_vvssl(vrpin_c1, 0, 0, 2*vl) ;

	__vr vrgout01 = _vel_vldu_vssl(4, pGOut+gOutIndex01, 2*vl) ;
	__vr vrgout23 = _vel_vldu_vssl(4, pGOut+gOutIndex23, 2*vl) ;
	__vr vrgout45 = _vel_vldu_vssl(4, pGOut+gOutIndex45, 2*vl) ;
	__vr vrgout67 = _vel_vldu_vssl(4, pGOut+gOutIndex67, 2*vl) ;

	__vr vrgout0123 = _vel_vshf_vvvsl(vrgout01, vrgout23, VE_VSHUFFLE_YUZU, 2*vl) ;
	__vr vrgout4567 = _vel_vshf_vvvsl(vrgout45, vrgout67, VE_VSHUFFLE_YUZU, 2*vl) ;


	VFMAD_R1S1(c0)
	VFMAD_R1S1(c1)

      } // batch
    } // gOutHeight

    VFSUM_STORE_R1S1(0,c0)
    VFSUM_STORE_R1S1(1,c1)

    c+=2 ;
  }
  if ( ((inChannelGroup >> 2) & 0x01) == 1 ) {
    const int64_t kernelIndex0 = kernGroupOffset + k     * inChannelGroup + c ;
    const int64_t kernelIndex1 = kernGroupOffset + (k+1) * inChannelGroup + c ;
    const int64_t kernelIndex2 = kernGroupOffset + (k+2) * inChannelGroup + c ;
    const int64_t kernelIndex3 = kernGroupOffset + (k+3) * inChannelGroup + c ;
    const int64_t kernelIndex4 = kernGroupOffset + (k+4) * inChannelGroup + c ;
    const int64_t kernelIndex5 = kernGroupOffset + (k+5) * inChannelGroup + c ;
    const int64_t kernelIndex6 = kernGroupOffset + (k+6) * inChannelGroup + c ;
    const int64_t kernelIndex7 = kernGroupOffset + (k+7) * inChannelGroup + c ;

    INIT_VRSUM(c0)
    INIT_VRSUM(c1)
    INIT_VRSUM(c2)
    INIT_VRSUM(c3)

    {
      const int64_t vl = gOutWidth * gOutHeight ;

      for (int64_t n=0; n<batch; n++) {
	const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;

	const int64_t gOutIndex01  = outGroupOffset + ((n * gOutChannel + k  ) * gOutHeight ) * gOutWidth ;
	const int64_t gOutIndex23  = outGroupOffset + ((n * gOutChannel + k+2) * gOutHeight ) * gOutWidth ;
	const int64_t gOutIndex45  = outGroupOffset + ((n * gOutChannel + k+4) * gOutHeight ) * gOutWidth ;
	const int64_t gOutIndex67  = outGroupOffset + ((n * gOutChannel + k+6) * gOutHeight ) * gOutWidth ;

	__vr vrpin_c0 = _vel_vsfa_vvssl(vrhw, 2, (uint64_t)pInChannel, 2*vl) ;
	__vr vrin_c0 = _vel_vgtu_vvssl(vrpin_c0, 0, 0, 2*vl) ;
	__vr vrpin_c1 = _vel_vaddul_vsvl(1*4*inHeight*inWidth, vrpin_c0, 2*vl) ;
	__vr vrin_c1 = _vel_vgtu_vvssl(vrpin_c1, 0, 0, 2*vl) ;
	__vr vrpin_c2 = _vel_vaddul_vsvl(2*4*inHeight*inWidth, vrpin_c0, 2*vl) ;
	__vr vrin_c2 = _vel_vgtu_vvssl(vrpin_c2, 0, 0, 2*vl) ;
	__vr vrpin_c3 = _vel_vaddul_vsvl(3*4*inHeight*inWidth, vrpin_c0, 2*vl) ;
	__vr vrin_c3 = _vel_vgtu_vvssl(vrpin_c3, 0, 0, 2*vl) ;

	__vr vrgout01 = _vel_vldu_vssl(4, pGOut+gOutIndex01, 2*vl) ;
	__vr vrgout23 = _vel_vldu_vssl(4, pGOut+gOutIndex23, 2*vl) ;
	__vr vrgout45 = _vel_vldu_vssl(4, pGOut+gOutIndex45, 2*vl) ;
	__vr vrgout67 = _vel_vldu_vssl(4, pGOut+gOutIndex67, 2*vl) ;

	__vr vrgout0123 = _vel_vshf_vvvsl(vrgout01, vrgout23, VE_VSHUFFLE_YUZU, 2*vl) ;
	__vr vrgout4567 = _vel_vshf_vvvsl(vrgout45, vrgout67, VE_VSHUFFLE_YUZU, 2*vl) ;

	VFMAD_R1S1(c0)
	VFMAD_R1S1(c1)
	VFMAD_R1S1(c2)
	VFMAD_R1S1(c3)

      } // batch
    } // gOutHeight

    VFSUM_STORE_R1S1(0,c0)
    VFSUM_STORE_R1S1(1,c1)
    VFSUM_STORE_R1S1(2,c2)
    VFSUM_STORE_R1S1(3,c3)

    c+=4 ;
  }
  for (; c<inChannelGroup; c+=8) {
    const int64_t kernelIndex0 = kernGroupOffset + k     * inChannelGroup + c ;
    const int64_t kernelIndex1 = kernGroupOffset + (k+1) * inChannelGroup + c ;
    const int64_t kernelIndex2 = kernGroupOffset + (k+2) * inChannelGroup + c ;
    const int64_t kernelIndex3 = kernGroupOffset + (k+3) * inChannelGroup + c ;
    const int64_t kernelIndex4 = kernGroupOffset + (k+4) * inChannelGroup + c ;
    const int64_t kernelIndex5 = kernGroupOffset + (k+5) * inChannelGroup + c ;
    const int64_t kernelIndex6 = kernGroupOffset + (k+6) * inChannelGroup + c ;
    const int64_t kernelIndex7 = kernGroupOffset + (k+7) * inChannelGroup + c ;

    INIT_VRSUM(c0)
    INIT_VRSUM(c1)
    INIT_VRSUM(c2)
    INIT_VRSUM(c3)
    INIT_VRSUM(c4)
    INIT_VRSUM(c5)
    INIT_VRSUM(c6)
    INIT_VRSUM(c7)
#undef INIT_VRSUM

    {
      const int64_t vl = gOutWidth * gOutHeight ;

      for (int64_t n=0; n<batch; n++) {
	const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;

	const int64_t gOutIndex01  = outGroupOffset + ((n * gOutChannel + k  ) * gOutHeight ) * gOutWidth ;
	const int64_t gOutIndex23  = outGroupOffset + ((n * gOutChannel + k+2) * gOutHeight ) * gOutWidth ;
	const int64_t gOutIndex45  = outGroupOffset + ((n * gOutChannel + k+4) * gOutHeight ) * gOutWidth ;
	const int64_t gOutIndex67  = outGroupOffset + ((n * gOutChannel + k+6) * gOutHeight ) * gOutWidth ;

	__vr vrpin_c0 = _vel_vsfa_vvssl(vrhw, 2, (uint64_t)pInChannel, 2*vl) ;
	__vr vrin_c0 = _vel_vgtu_vvssl(vrpin_c0, 0, 0, 2*vl) ;
	__vr vrpin_c1 = _vel_vaddul_vsvl(1*4*inHeight*inWidth, vrpin_c0, 2*vl) ;
	__vr vrin_c1 = _vel_vgtu_vvssl(vrpin_c1, 0, 0, 2*vl) ;
	__vr vrpin_c2 = _vel_vaddul_vsvl(2*4*inHeight*inWidth, vrpin_c0, 2*vl) ;
	__vr vrin_c2 = _vel_vgtu_vvssl(vrpin_c2, 0, 0, 2*vl) ;
	__vr vrpin_c3 = _vel_vaddul_vsvl(3*4*inHeight*inWidth, vrpin_c0, 2*vl) ;
	__vr vrin_c3 = _vel_vgtu_vvssl(vrpin_c3, 0, 0, 2*vl) ;
	__vr vrpin_c4 = _vel_vaddul_vsvl(4*4*inHeight*inWidth, vrpin_c0, 2*vl) ;
	__vr vrin_c4 = _vel_vgtu_vvssl(vrpin_c4, 0, 0, 2*vl) ;
	__vr vrpin_c5 = _vel_vaddul_vsvl(5*4*inHeight*inWidth, vrpin_c0, 2*vl) ;
	__vr vrin_c5 = _vel_vgtu_vvssl(vrpin_c5, 0, 0, 2*vl) ;
	__vr vrpin_c6 = _vel_vaddul_vsvl(6*4*inHeight*inWidth, vrpin_c0, 2*vl) ;
	__vr vrin_c6 = _vel_vgtu_vvssl(vrpin_c6, 0, 0, 2*vl) ;
	__vr vrpin_c7 = _vel_vaddul_vsvl(7*4*inHeight*inWidth, vrpin_c0, 2*vl) ;
	__vr vrin_c7 = _vel_vgtu_vvssl(vrpin_c7, 0, 0, 2*vl) ;

	__vr vrgout01 = _vel_vldu_vssl(4, pGOut+gOutIndex01, 2*vl) ;
	__vr vrgout23 = _vel_vldu_vssl(4, pGOut+gOutIndex23, 2*vl) ;
	__vr vrgout45 = _vel_vldu_vssl(4, pGOut+gOutIndex45, 2*vl) ;
	__vr vrgout67 = _vel_vldu_vssl(4, pGOut+gOutIndex67, 2*vl) ;

	__vr vrgout0123 = _vel_vshf_vvvsl(vrgout01, vrgout23, VE_VSHUFFLE_YUZU, 2*vl) ;
	__vr vrgout4567 = _vel_vshf_vvvsl(vrgout45, vrgout67, VE_VSHUFFLE_YUZU, 2*vl) ;

	VFMAD_R1S1(c0)
	VFMAD_R1S1(c1)
	VFMAD_R1S1(c2)
	VFMAD_R1S1(c3)
	VFMAD_R1S1(c4)
	VFMAD_R1S1(c5)
	VFMAD_R1S1(c6)
	VFMAD_R1S1(c7)
#undef VFMAD_R1S1
      } // batch
    } // gOutHeight

    VFSUM_STORE_R1S1(0,c0)
    VFSUM_STORE_R1S1(1,c1)
    VFSUM_STORE_R1S1(2,c2)
    VFSUM_STORE_R1S1(3,c3)
    VFSUM_STORE_R1S1(4,c4)
    VFSUM_STORE_R1S1(5,c5)
    VFSUM_STORE_R1S1(6,c6)
    VFSUM_STORE_R1S1(7,c7)
#undef VFSUM_STORE_R1S1
  }
}



static inline void f16(
  const float * pIn, const int64_t inWidth, const int64_t inHeight,
  const float * pGOut, const int64_t gOutWidth, const int64_t gOutHeight,
  float * const pGKernel,
  const int64_t strideHeight, const int64_t strideWidth,
  const int64_t inChannelGroup, const int64_t inChannel, const int64_t gOutChannel,
  const int64_t inGroupOffset, const int64_t outGroupOffset, const int64_t kernGroupOffset,
  const int64_t batch, const int64_t k,
  const __vr vrhw,
  const __vm256 vm_k0, const __vm256 vm_k1, const __vm256 vm_k2, const __vm256 vm_k3
)
{
  int64_t c=0;
  if ( (inChannelGroup & 0x01) == 1 ) {
    const int64_t kernelIndex = kernGroupOffset + k     * inChannelGroup + c ;

#define INIT_VRSUM(CTOKEN) \
    __vr vrsum01234567_##CTOKEN = _vel_vbrdl_vsl(0UL, VLEN) ; \
    __vr vrsum89ABCDEF_##CTOKEN = _vel_vbrdl_vsl(0UL, VLEN) ;

    INIT_VRSUM(c0)

    {
      const int64_t vl = gOutWidth * gOutHeight ;

      for (int64_t n=0; n<batch; n++) {
	const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;

	const int64_t gOutIndex  = outGroupOffset + ((n * gOutChannel + k  ) * gOutHeight ) * gOutWidth ;

	__vr vrpin_c0 = _vel_vsfa_vvssl(vrhw, 2, (uint64_t)pInChannel, 4*vl) ;
	__vr vrin_c0 = _vel_vgtu_vvssl(vrpin_c0, 0, 0, 4*vl) ;

	__vr vrgout0123 = _vel_vldu_vssl(4, pGOut+gOutIndex , 4*vl) ;
	__vr vrgout4567 = _vel_vldu_vssl(4, pGOut+gOutIndex+ 4*gOutHeight*gOutWidth, 4*vl) ;
	__vr vrgout89AB = _vel_vldu_vssl(4, pGOut+gOutIndex+ 8*gOutHeight*gOutWidth, 4*vl) ;
	__vr vrgoutCDEF = _vel_vldu_vssl(4, pGOut+gOutIndex+12*gOutHeight*gOutWidth, 4*vl) ;

	__vr vrgout01234567 = _vel_vshf_vvvsl(vrgout0123, vrgout4567, VE_VSHUFFLE_YUZU, 4*vl) ;
	__vr vrgout89ABCDEF = _vel_vshf_vvvsl(vrgout89AB, vrgoutCDEF, VE_VSHUFFLE_YUZU, 4*vl) ;

#define VFMAD_R1S1(CTOKEN) \
          __vr vrinP_##CTOKEN = _vel_vshf_vvvsl(vrin_##CTOKEN, vrin_##CTOKEN, VE_VSHUFFLE_YUZU, 4*vl) ;  \
          vrsum01234567_##CTOKEN = _vel_pvfmad_vvvvvl(vrsum01234567_##CTOKEN, vrinP_##CTOKEN, vrgout01234567, vrsum01234567_##CTOKEN, 4*vl) ; \
          vrsum89ABCDEF_##CTOKEN = _vel_pvfmad_vvvvvl(vrsum89ABCDEF_##CTOKEN, vrinP_##CTOKEN, vrgout89ABCDEF, vrsum89ABCDEF_##CTOKEN, 4*vl) ;

	  VFMAD_R1S1(c0)

      } // batch
    } // gOutHeight

#define VFSUM_STORE_R1S1(C,CTOKEN) \
    __vr vrsum0_##CTOKEN = _vel_vfsums_vvml(vrsum01234567_##CTOKEN, vm_k0, VLEN) ; \
    __vr vrsum1_##CTOKEN = _vel_vfsums_vvml(vrsum01234567_##CTOKEN, vm_k1, VLEN) ; \
    __vr vrsum2_##CTOKEN = _vel_vfsums_vvml(vrsum01234567_##CTOKEN, vm_k2, VLEN) ; \
    __vr vrsum3_##CTOKEN = _vel_vfsums_vvml(vrsum01234567_##CTOKEN, vm_k3, VLEN) ; \
    __vr vrsum4_##CTOKEN = _vel_vfsums_vvml(_vel_vsll_vvsl(vrsum01234567_##CTOKEN,32, VLEN), vm_k0, VLEN); \
    __vr vrsum5_##CTOKEN = _vel_vfsums_vvml(_vel_vsll_vvsl(vrsum01234567_##CTOKEN,32, VLEN), vm_k1, VLEN); \
    __vr vrsum6_##CTOKEN = _vel_vfsums_vvml(_vel_vsll_vvsl(vrsum01234567_##CTOKEN,32, VLEN), vm_k2, VLEN); \
    __vr vrsum7_##CTOKEN = _vel_vfsums_vvml(_vel_vsll_vvsl(vrsum01234567_##CTOKEN,32, VLEN), vm_k3, VLEN); \
    _vel_vstu_vssl(vrsum0_##CTOKEN, 4, pGKernel+kernelIndex+0*inChannelGroup+(C), 1) ; \
    _vel_vstu_vssl(vrsum1_##CTOKEN, 4, pGKernel+kernelIndex+1*inChannelGroup+(C), 1) ; \
    _vel_vstu_vssl(vrsum2_##CTOKEN, 4, pGKernel+kernelIndex+2*inChannelGroup+(C), 1) ; \
    _vel_vstu_vssl(vrsum3_##CTOKEN, 4, pGKernel+kernelIndex+3*inChannelGroup+(C), 1) ; \
    _vel_vstu_vssl(vrsum4_##CTOKEN, 4, pGKernel+kernelIndex+4*inChannelGroup+(C), 1) ; \
    _vel_vstu_vssl(vrsum5_##CTOKEN, 4, pGKernel+kernelIndex+5*inChannelGroup+(C), 1) ; \
    _vel_vstu_vssl(vrsum6_##CTOKEN, 4, pGKernel+kernelIndex+6*inChannelGroup+(C), 1) ; \
    _vel_vstu_vssl(vrsum7_##CTOKEN, 4, pGKernel+kernelIndex+7*inChannelGroup+(C), 1) ; \
    __vr vrsum8_##CTOKEN = _vel_vfsums_vvml(vrsum89ABCDEF_##CTOKEN, vm_k0, VLEN) ; \
    __vr vrsum9_##CTOKEN = _vel_vfsums_vvml(vrsum89ABCDEF_##CTOKEN, vm_k1, VLEN) ; \
    __vr vrsumA_##CTOKEN = _vel_vfsums_vvml(vrsum89ABCDEF_##CTOKEN, vm_k2, VLEN) ; \
    __vr vrsumB_##CTOKEN = _vel_vfsums_vvml(vrsum89ABCDEF_##CTOKEN, vm_k3, VLEN) ; \
    __vr vrsumC_##CTOKEN = _vel_vfsums_vvml(_vel_vsll_vvsl(vrsum89ABCDEF_##CTOKEN,32, VLEN), vm_k0, VLEN); \
    __vr vrsumD_##CTOKEN = _vel_vfsums_vvml(_vel_vsll_vvsl(vrsum89ABCDEF_##CTOKEN,32, VLEN), vm_k1, VLEN); \
    __vr vrsumE_##CTOKEN = _vel_vfsums_vvml(_vel_vsll_vvsl(vrsum89ABCDEF_##CTOKEN,32, VLEN), vm_k2, VLEN); \
    __vr vrsumF_##CTOKEN = _vel_vfsums_vvml(_vel_vsll_vvsl(vrsum89ABCDEF_##CTOKEN,32, VLEN), vm_k3, VLEN); \
    _vel_vstu_vssl(vrsum8_##CTOKEN, 4, pGKernel+kernelIndex+8*inChannelGroup+(C), 1) ; \
    _vel_vstu_vssl(vrsum9_##CTOKEN, 4, pGKernel+kernelIndex+9*inChannelGroup+(C), 1) ; \
    _vel_vstu_vssl(vrsumA_##CTOKEN, 4, pGKernel+kernelIndex+10*inChannelGroup+(C), 1) ; \
    _vel_vstu_vssl(vrsumB_##CTOKEN, 4, pGKernel+kernelIndex+11*inChannelGroup+(C), 1) ; \
    _vel_vstu_vssl(vrsumC_##CTOKEN, 4, pGKernel+kernelIndex+12*inChannelGroup+(C), 1) ; \
    _vel_vstu_vssl(vrsumD_##CTOKEN, 4, pGKernel+kernelIndex+13*inChannelGroup+(C), 1) ; \
    _vel_vstu_vssl(vrsumE_##CTOKEN, 4, pGKernel+kernelIndex+14*inChannelGroup+(C), 1) ; \
    _vel_vstu_vssl(vrsumF_##CTOKEN, 4, pGKernel+kernelIndex+15*inChannelGroup+(C), 1) ;

    VFSUM_STORE_R1S1(0,c0)

    c+=1;
  }
  if ( ((inChannelGroup >> 1) & 0x01) == 1 ) {
    const int64_t kernelIndex = kernGroupOffset + k     * inChannelGroup + c ;

    INIT_VRSUM(c0)
    INIT_VRSUM(c1)

    {
      const int64_t vl = gOutWidth * gOutHeight ;

      for (int64_t n=0; n<batch; n++) {
	const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;

	const int64_t gOutIndex  = outGroupOffset + ((n * gOutChannel + k  ) * gOutHeight ) * gOutWidth ;

	__vr vrpin_c0 = _vel_vsfa_vvssl(vrhw, 2, (uint64_t)pInChannel, 4*vl) ;
	__vr vrin_c0 = _vel_vgtu_vvssl(vrpin_c0, 0, 0, 4*vl) ;
	__vr vrpin_c1 = _vel_vaddul_vsvl(1*4*inHeight*inWidth, vrpin_c0, 4*vl) ;
	__vr vrin_c1 = _vel_vgtu_vvssl(vrpin_c1, 0, 0, 4*vl) ;

	__vr vrgout0123 = _vel_vldu_vssl(4, pGOut+gOutIndex, 4*vl) ;
	__vr vrgout4567 = _vel_vldu_vssl(4, pGOut+gOutIndex+ 4*gOutHeight*gOutWidth, 4*vl) ;
	__vr vrgout89AB = _vel_vldu_vssl(4, pGOut+gOutIndex+ 8*gOutHeight*gOutWidth, 4*vl) ;
	__vr vrgoutCDEF = _vel_vldu_vssl(4, pGOut+gOutIndex+12*gOutHeight*gOutWidth, 4*vl) ;

	__vr vrgout01234567 = _vel_vshf_vvvsl(vrgout0123, vrgout4567, VE_VSHUFFLE_YUZU, 4*vl) ;
	__vr vrgout89ABCDEF = _vel_vshf_vvvsl(vrgout89AB, vrgoutCDEF, VE_VSHUFFLE_YUZU, 4*vl) ;

	VFMAD_R1S1(c0)
	VFMAD_R1S1(c1)

      } // batch
    } // gOutHeight

    VFSUM_STORE_R1S1(0,c0)
    VFSUM_STORE_R1S1(1,c1)

    c+=2 ;
  }
  if ( ((inChannelGroup >> 2) & 0x01) == 1 ) {
    const int64_t kernelIndex = kernGroupOffset + k     * inChannelGroup + c ;

    INIT_VRSUM(c0)
    INIT_VRSUM(c1)
    INIT_VRSUM(c2)
    INIT_VRSUM(c3)

    {
      const int64_t vl = gOutWidth * gOutHeight ;

      for (int64_t n=0; n<batch; n++) {
	const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;

	const int64_t gOutIndex  = outGroupOffset + ((n * gOutChannel + k  ) * gOutHeight ) * gOutWidth ;

	__vr vrpin_c0 = _vel_vsfa_vvssl(vrhw, 2, (uint64_t)pInChannel, 4*vl) ;
	__vr vrin_c0 = _vel_vgtu_vvssl(vrpin_c0, 0, 0, 4*vl) ;
	__vr vrpin_c1 = _vel_vaddul_vsvl(1*4*inHeight*inWidth, vrpin_c0, 4*vl) ;
	__vr vrin_c1 = _vel_vgtu_vvssl(vrpin_c1, 0, 0, 4*vl) ;
	__vr vrpin_c2 = _vel_vaddul_vsvl(2*4*inHeight*inWidth, vrpin_c0, 4*vl) ;
	__vr vrin_c2 = _vel_vgtu_vvssl(vrpin_c2, 0, 0, 4*vl) ;
	__vr vrpin_c3 = _vel_vaddul_vsvl(3*4*inHeight*inWidth, vrpin_c0, 4*vl) ;
	__vr vrin_c3 = _vel_vgtu_vvssl(vrpin_c3, 0, 0, 4*vl) ;

	__vr vrgout0123 = _vel_vldu_vssl(4, pGOut+gOutIndex, 4*vl) ;
	__vr vrgout4567 = _vel_vldu_vssl(4, pGOut+gOutIndex+ 4*gOutHeight*gOutWidth, 4*vl) ;
	__vr vrgout89AB = _vel_vldu_vssl(4, pGOut+gOutIndex+ 8*gOutHeight*gOutWidth, 4*vl) ;
	__vr vrgoutCDEF = _vel_vldu_vssl(4, pGOut+gOutIndex+12*gOutHeight*gOutWidth, 4*vl) ;

	__vr vrgout01234567 = _vel_vshf_vvvsl(vrgout0123, vrgout4567, VE_VSHUFFLE_YUZU, 4*vl) ;
	__vr vrgout89ABCDEF = _vel_vshf_vvvsl(vrgout89AB, vrgoutCDEF, VE_VSHUFFLE_YUZU, 4*vl) ;

	VFMAD_R1S1(c0)
	VFMAD_R1S1(c1)
	VFMAD_R1S1(c2)
	VFMAD_R1S1(c3)

      } // batch
    } // gOutHeight

    VFSUM_STORE_R1S1(0,c0)
    VFSUM_STORE_R1S1(1,c1)
    VFSUM_STORE_R1S1(2,c2)
    VFSUM_STORE_R1S1(3,c3)

    c+=4 ;
  }
  for (; c<inChannelGroup; c+=8) {
    const int64_t kernelIndex = kernGroupOffset + k     * inChannelGroup + c ;

    INIT_VRSUM(c0)
    INIT_VRSUM(c1)
    INIT_VRSUM(c2)
    INIT_VRSUM(c3)
    INIT_VRSUM(c4)
    INIT_VRSUM(c5)
    INIT_VRSUM(c6)
    INIT_VRSUM(c7)
#undef INIT_VRSUM

    {
      const int64_t vl = gOutWidth * gOutHeight ;

      for (int64_t n=0; n<batch; n++) {
	const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;

	const int64_t gOutIndex  = outGroupOffset + ((n * gOutChannel + k  ) * gOutHeight ) * gOutWidth ;

	__vr vrpin_c0 = _vel_vsfa_vvssl(vrhw, 2, (uint64_t)pInChannel, 4*vl) ;
	__vr vrin_c0 = _vel_vgtu_vvssl(vrpin_c0, 0, 0, 4*vl) ;
	__vr vrpin_c1 = _vel_vaddul_vsvl(1*4*inHeight*inWidth, vrpin_c0, 4*vl) ;
	__vr vrin_c1 = _vel_vgtu_vvssl(vrpin_c1, 0, 0, 4*vl) ;
	__vr vrpin_c2 = _vel_vaddul_vsvl(2*4*inHeight*inWidth, vrpin_c0, 4*vl) ;
	__vr vrin_c2 = _vel_vgtu_vvssl(vrpin_c2, 0, 0, 4*vl) ;
	__vr vrpin_c3 = _vel_vaddul_vsvl(3*4*inHeight*inWidth, vrpin_c0, 4*vl) ;
	__vr vrin_c3 = _vel_vgtu_vvssl(vrpin_c3, 0, 0, 4*vl) ;
	__vr vrpin_c4 = _vel_vaddul_vsvl(4*4*inHeight*inWidth, vrpin_c0, 4*vl) ;
	__vr vrin_c4 = _vel_vgtu_vvssl(vrpin_c4, 0, 0, 4*vl) ;
	__vr vrpin_c5 = _vel_vaddul_vsvl(5*4*inHeight*inWidth, vrpin_c0, 4*vl) ;
	__vr vrin_c5 = _vel_vgtu_vvssl(vrpin_c5, 0, 0, 4*vl) ;
	__vr vrpin_c6 = _vel_vaddul_vsvl(6*4*inHeight*inWidth, vrpin_c0, 4*vl) ;
	__vr vrin_c6 = _vel_vgtu_vvssl(vrpin_c6, 0, 0, 4*vl) ;
	__vr vrpin_c7 = _vel_vaddul_vsvl(7*4*inHeight*inWidth, vrpin_c0, 4*vl) ;
	__vr vrin_c7 = _vel_vgtu_vvssl(vrpin_c7, 0, 0, 4*vl) ;

	__vr vrgout0123 = _vel_vldu_vssl(4, pGOut+gOutIndex, 4*vl) ;
	__vr vrgout4567 = _vel_vldu_vssl(4, pGOut+gOutIndex+ 4*gOutHeight*gOutWidth, 4*vl) ;
	__vr vrgout89AB = _vel_vldu_vssl(4, pGOut+gOutIndex+ 8*gOutHeight*gOutWidth, 4*vl) ;
	__vr vrgoutCDEF = _vel_vldu_vssl(4, pGOut+gOutIndex+12*gOutHeight*gOutWidth, 4*vl) ;

	__vr vrgout01234567 = _vel_vshf_vvvsl(vrgout0123, vrgout4567, VE_VSHUFFLE_YUZU, 4*vl) ;
	__vr vrgout89ABCDEF = _vel_vshf_vvvsl(vrgout89AB, vrgoutCDEF, VE_VSHUFFLE_YUZU, 4*vl) ;

	VFMAD_R1S1(c0)
	VFMAD_R1S1(c1)
	VFMAD_R1S1(c2)
	VFMAD_R1S1(c3)
	VFMAD_R1S1(c4)
	VFMAD_R1S1(c5)
	VFMAD_R1S1(c6)
	VFMAD_R1S1(c7)
#undef VFMAD_R1S1
      } // batch
    } // gOutHeight

    VFSUM_STORE_R1S1(0,c0)
    VFSUM_STORE_R1S1(1,c1)
    VFSUM_STORE_R1S1(2,c2)
    VFSUM_STORE_R1S1(3,c3)
    VFSUM_STORE_R1S1(4,c4)
    VFSUM_STORE_R1S1(5,c5)
    VFSUM_STORE_R1S1(6,c6)
    VFSUM_STORE_R1S1(7,c7)
#undef VFSUM_STORE_R1S1
  }
}


static inline void f32(
  const float * pIn, const int64_t inWidth, const int64_t inHeight,
  const float * pGOut, const int64_t gOutWidth, const int64_t gOutHeight,
  float * const pGKernel,
  const int64_t strideHeight, const int64_t strideWidth,
  const int64_t inChannelGroup, const int64_t inChannel, const int64_t gOutChannel,
  const int64_t inGroupOffset, const int64_t outGroupOffset, const int64_t kernGroupOffset,
  const int64_t batch, const int64_t k,
  const __vr vrhw,
  const __vm256 vm_k0, const __vm256 vm_k1, const __vm256 vm_k2, const __vm256 vm_k3
)
{
  int64_t c=0;
  if ( (inChannelGroup & 0x01) == 1 ) {
    const int64_t kernelIndex = kernGroupOffset + k     * inChannelGroup + c ;

#define INIT_VRSUM(CTOKEN) \
    __vr vrsum01234567_##CTOKEN = _vel_vbrdl_vsl(0UL, VLEN) ; \
    __vr vrsum89ABCDEF_##CTOKEN = _vel_vbrdl_vsl(0UL, VLEN) ; \
    __vr vrsumGHIJKLMN_##CTOKEN = _vel_vbrdl_vsl(0UL, VLEN) ; \
    __vr vrsumOPQRSTUV_##CTOKEN = _vel_vbrdl_vsl(0UL, VLEN) ;

    INIT_VRSUM(c0)

    {
      const int64_t vl = gOutWidth * gOutHeight ;

      for (int64_t n=0; n<batch; n++) {
	const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;

	const int64_t gOutIndex0123  = outGroupOffset + ((n * gOutChannel + k  ) * gOutHeight ) * gOutWidth ;
	const int64_t gOutIndex4567  = outGroupOffset + ((n * gOutChannel + k+4) * gOutHeight ) * gOutWidth ;
	const int64_t gOutIndex89AB  = outGroupOffset + ((n * gOutChannel + k+8) * gOutHeight ) * gOutWidth ;
	const int64_t gOutIndexCDEF  = outGroupOffset + ((n * gOutChannel + k+12) * gOutHeight ) * gOutWidth ;
	const int64_t gOutIndexGHIJ  = outGroupOffset + ((n * gOutChannel + k+16) * gOutHeight ) * gOutWidth ;
	const int64_t gOutIndexKLMN  = outGroupOffset + ((n * gOutChannel + k+20) * gOutHeight ) * gOutWidth ;
	const int64_t gOutIndexOPQR  = outGroupOffset + ((n * gOutChannel + k+24) * gOutHeight ) * gOutWidth ;
	const int64_t gOutIndexSTUV  = outGroupOffset + ((n * gOutChannel + k+28) * gOutHeight ) * gOutWidth ;

	__vr vrpin_c0 = _vel_vsfa_vvssl(vrhw, 2, (uint64_t)pInChannel, 4*vl) ;
	__vr vrin_c0 = _vel_vgtu_vvssl(vrpin_c0, 0, 0, 4*vl) ;

	__vr vrgout0123 = _vel_vldu_vssl(4, pGOut+gOutIndex0123, 4*vl) ;
	__vr vrgout4567 = _vel_vldu_vssl(4, pGOut+gOutIndex4567, 4*vl) ;
	__vr vrgout89AB = _vel_vldu_vssl(4, pGOut+gOutIndex89AB, 4*vl) ;
	__vr vrgoutCDEF = _vel_vldu_vssl(4, pGOut+gOutIndexCDEF, 4*vl) ;
	__vr vrgoutGHIJ = _vel_vldu_vssl(4, pGOut+gOutIndexGHIJ, 4*vl) ;
	__vr vrgoutKLMN = _vel_vldu_vssl(4, pGOut+gOutIndexKLMN, 4*vl) ;
	__vr vrgoutOPQR = _vel_vldu_vssl(4, pGOut+gOutIndexOPQR, 4*vl) ;
	__vr vrgoutSTUV = _vel_vldu_vssl(4, pGOut+gOutIndexSTUV, 4*vl) ;

	__vr vrgout01234567 = _vel_vshf_vvvsl(vrgout0123, vrgout4567, VE_VSHUFFLE_YUZU, 4*vl) ;
	__vr vrgout89ABCDEF = _vel_vshf_vvvsl(vrgout89AB, vrgoutCDEF, VE_VSHUFFLE_YUZU, 4*vl) ;
	__vr vrgoutGHIJKLMN = _vel_vshf_vvvsl(vrgoutGHIJ, vrgoutKLMN, VE_VSHUFFLE_YUZU, 4*vl) ;
	__vr vrgoutOPQRSTUV = _vel_vshf_vvvsl(vrgoutOPQR, vrgoutSTUV, VE_VSHUFFLE_YUZU, 4*vl) ;

#define VFMAD_R1S1(CTOKEN) \
          __vr vrinP_##CTOKEN = _vel_vshf_vvvsl(vrin_##CTOKEN, vrin_##CTOKEN, VE_VSHUFFLE_YUZU, 4*vl) ;  \
          vrsum01234567_##CTOKEN = _vel_pvfmad_vvvvvl(vrsum01234567_##CTOKEN, vrinP_##CTOKEN, vrgout01234567, vrsum01234567_##CTOKEN, 4*vl) ; \
          vrsum89ABCDEF_##CTOKEN = _vel_pvfmad_vvvvvl(vrsum89ABCDEF_##CTOKEN, vrinP_##CTOKEN, vrgout89ABCDEF, vrsum89ABCDEF_##CTOKEN, 4*vl) ; \
          vrsumGHIJKLMN_##CTOKEN = _vel_pvfmad_vvvvvl(vrsumGHIJKLMN_##CTOKEN, vrinP_##CTOKEN, vrgoutGHIJKLMN, vrsumGHIJKLMN_##CTOKEN, 4*vl) ; \
          vrsumOPQRSTUV_##CTOKEN = _vel_pvfmad_vvvvvl(vrsumOPQRSTUV_##CTOKEN, vrinP_##CTOKEN, vrgoutOPQRSTUV, vrsumOPQRSTUV_##CTOKEN, 4*vl) ;

	  VFMAD_R1S1(c0)

      } // batch
    } // gOutHeight

#define VFSUM_STORE_R1S1(C,CTOKEN) \
    __vr vrsum0_##CTOKEN = _vel_vfsums_vvml(vrsum01234567_##CTOKEN, vm_k0, VLEN) ; \
    __vr vrsum1_##CTOKEN = _vel_vfsums_vvml(vrsum01234567_##CTOKEN, vm_k1, VLEN) ; \
    __vr vrsum2_##CTOKEN = _vel_vfsums_vvml(vrsum01234567_##CTOKEN, vm_k2, VLEN) ; \
    __vr vrsum3_##CTOKEN = _vel_vfsums_vvml(vrsum01234567_##CTOKEN, vm_k3, VLEN) ; \
    __vr vrsum4_##CTOKEN = _vel_vfsums_vvml(_vel_vsll_vvsl(vrsum01234567_##CTOKEN,32, VLEN), vm_k0, VLEN); \
    __vr vrsum5_##CTOKEN = _vel_vfsums_vvml(_vel_vsll_vvsl(vrsum01234567_##CTOKEN,32, VLEN), vm_k1, VLEN); \
    __vr vrsum6_##CTOKEN = _vel_vfsums_vvml(_vel_vsll_vvsl(vrsum01234567_##CTOKEN,32, VLEN), vm_k2, VLEN); \
    __vr vrsum7_##CTOKEN = _vel_vfsums_vvml(_vel_vsll_vvsl(vrsum01234567_##CTOKEN,32, VLEN), vm_k3, VLEN); \
    _vel_vstu_vssl(vrsum0_##CTOKEN, 4, pGKernel+kernelIndex+0*inChannelGroup+(C), 1) ; \
    _vel_vstu_vssl(vrsum1_##CTOKEN, 4, pGKernel+kernelIndex+1*inChannelGroup+(C), 1) ; \
    _vel_vstu_vssl(vrsum2_##CTOKEN, 4, pGKernel+kernelIndex+2*inChannelGroup+(C), 1) ; \
    _vel_vstu_vssl(vrsum3_##CTOKEN, 4, pGKernel+kernelIndex+3*inChannelGroup+(C), 1) ; \
    _vel_vstu_vssl(vrsum4_##CTOKEN, 4, pGKernel+kernelIndex+4*inChannelGroup+(C), 1) ; \
    _vel_vstu_vssl(vrsum5_##CTOKEN, 4, pGKernel+kernelIndex+5*inChannelGroup+(C), 1) ; \
    _vel_vstu_vssl(vrsum6_##CTOKEN, 4, pGKernel+kernelIndex+6*inChannelGroup+(C), 1) ; \
    _vel_vstu_vssl(vrsum7_##CTOKEN, 4, pGKernel+kernelIndex+7*inChannelGroup+(C), 1) ; \
    __vr vrsum8_##CTOKEN = _vel_vfsums_vvml(vrsum89ABCDEF_##CTOKEN, vm_k0, VLEN) ; \
    __vr vrsum9_##CTOKEN = _vel_vfsums_vvml(vrsum89ABCDEF_##CTOKEN, vm_k1, VLEN) ; \
    __vr vrsumA_##CTOKEN = _vel_vfsums_vvml(vrsum89ABCDEF_##CTOKEN, vm_k2, VLEN) ; \
    __vr vrsumB_##CTOKEN = _vel_vfsums_vvml(vrsum89ABCDEF_##CTOKEN, vm_k3, VLEN) ; \
    __vr vrsumC_##CTOKEN = _vel_vfsums_vvml(_vel_vsll_vvsl(vrsum89ABCDEF_##CTOKEN,32, VLEN), vm_k0, VLEN); \
    __vr vrsumD_##CTOKEN = _vel_vfsums_vvml(_vel_vsll_vvsl(vrsum89ABCDEF_##CTOKEN,32, VLEN), vm_k1, VLEN); \
    __vr vrsumE_##CTOKEN = _vel_vfsums_vvml(_vel_vsll_vvsl(vrsum89ABCDEF_##CTOKEN,32, VLEN), vm_k2, VLEN); \
    __vr vrsumF_##CTOKEN = _vel_vfsums_vvml(_vel_vsll_vvsl(vrsum89ABCDEF_##CTOKEN,32, VLEN), vm_k3, VLEN); \
    _vel_vstu_vssl(vrsum8_##CTOKEN, 4, pGKernel+kernelIndex+8*inChannelGroup+(C), 1) ; \
    _vel_vstu_vssl(vrsum9_##CTOKEN, 4, pGKernel+kernelIndex+9*inChannelGroup+(C), 1) ; \
    _vel_vstu_vssl(vrsumA_##CTOKEN, 4, pGKernel+kernelIndex+10*inChannelGroup+(C), 1) ; \
    _vel_vstu_vssl(vrsumB_##CTOKEN, 4, pGKernel+kernelIndex+11*inChannelGroup+(C), 1) ; \
    _vel_vstu_vssl(vrsumC_##CTOKEN, 4, pGKernel+kernelIndex+12*inChannelGroup+(C), 1) ; \
    _vel_vstu_vssl(vrsumD_##CTOKEN, 4, pGKernel+kernelIndex+13*inChannelGroup+(C), 1) ; \
    _vel_vstu_vssl(vrsumE_##CTOKEN, 4, pGKernel+kernelIndex+14*inChannelGroup+(C), 1) ; \
    _vel_vstu_vssl(vrsumF_##CTOKEN, 4, pGKernel+kernelIndex+15*inChannelGroup+(C), 1) ; \
    __vr vrsumG_##CTOKEN = _vel_vfsums_vvml(vrsumGHIJKLMN_##CTOKEN, vm_k0, VLEN) ; \
    __vr vrsumH_##CTOKEN = _vel_vfsums_vvml(vrsumGHIJKLMN_##CTOKEN, vm_k1, VLEN) ; \
    __vr vrsumI_##CTOKEN = _vel_vfsums_vvml(vrsumGHIJKLMN_##CTOKEN, vm_k2, VLEN) ; \
    __vr vrsumJ_##CTOKEN = _vel_vfsums_vvml(vrsumGHIJKLMN_##CTOKEN, vm_k3, VLEN) ; \
    __vr vrsumK_##CTOKEN = _vel_vfsums_vvml(_vel_vsll_vvsl(vrsumGHIJKLMN_##CTOKEN,32, VLEN), vm_k0, VLEN); \
    __vr vrsumL_##CTOKEN = _vel_vfsums_vvml(_vel_vsll_vvsl(vrsumGHIJKLMN_##CTOKEN,32, VLEN), vm_k1, VLEN); \
    __vr vrsumM_##CTOKEN = _vel_vfsums_vvml(_vel_vsll_vvsl(vrsumGHIJKLMN_##CTOKEN,32, VLEN), vm_k2, VLEN); \
    __vr vrsumN_##CTOKEN = _vel_vfsums_vvml(_vel_vsll_vvsl(vrsumGHIJKLMN_##CTOKEN,32, VLEN), vm_k3, VLEN); \
    _vel_vstu_vssl(vrsumG_##CTOKEN, 4, pGKernel+kernelIndex+16*inChannelGroup+(C), 1) ; \
    _vel_vstu_vssl(vrsumH_##CTOKEN, 4, pGKernel+kernelIndex+17*inChannelGroup+(C), 1) ; \
    _vel_vstu_vssl(vrsumI_##CTOKEN, 4, pGKernel+kernelIndex+18*inChannelGroup+(C), 1) ; \
    _vel_vstu_vssl(vrsumJ_##CTOKEN, 4, pGKernel+kernelIndex+19*inChannelGroup+(C), 1) ; \
    _vel_vstu_vssl(vrsumK_##CTOKEN, 4, pGKernel+kernelIndex+20*inChannelGroup+(C), 1) ; \
    _vel_vstu_vssl(vrsumL_##CTOKEN, 4, pGKernel+kernelIndex+21*inChannelGroup+(C), 1) ; \
    _vel_vstu_vssl(vrsumM_##CTOKEN, 4, pGKernel+kernelIndex+22*inChannelGroup+(C), 1) ; \
    _vel_vstu_vssl(vrsumN_##CTOKEN, 4, pGKernel+kernelIndex+23*inChannelGroup+(C), 1) ; \
    __vr vrsumO_##CTOKEN = _vel_vfsums_vvml(vrsumOPQRSTUV_##CTOKEN, vm_k0, VLEN) ; \
    __vr vrsumP_##CTOKEN = _vel_vfsums_vvml(vrsumOPQRSTUV_##CTOKEN, vm_k1, VLEN) ; \
    __vr vrsumQ_##CTOKEN = _vel_vfsums_vvml(vrsumOPQRSTUV_##CTOKEN, vm_k2, VLEN) ; \
    __vr vrsumR_##CTOKEN = _vel_vfsums_vvml(vrsumOPQRSTUV_##CTOKEN, vm_k3, VLEN) ; \
    __vr vrsumS_##CTOKEN = _vel_vfsums_vvml(_vel_vsll_vvsl(vrsumOPQRSTUV_##CTOKEN,32, VLEN), vm_k0, VLEN); \
    __vr vrsumT_##CTOKEN = _vel_vfsums_vvml(_vel_vsll_vvsl(vrsumOPQRSTUV_##CTOKEN,32, VLEN), vm_k1, VLEN); \
    __vr vrsumU_##CTOKEN = _vel_vfsums_vvml(_vel_vsll_vvsl(vrsumOPQRSTUV_##CTOKEN,32, VLEN), vm_k2, VLEN); \
    __vr vrsumV_##CTOKEN = _vel_vfsums_vvml(_vel_vsll_vvsl(vrsumOPQRSTUV_##CTOKEN,32, VLEN), vm_k3, VLEN); \
    _vel_vstu_vssl(vrsumO_##CTOKEN, 4, pGKernel+kernelIndex+24*inChannelGroup+(C), 1) ; \
    _vel_vstu_vssl(vrsumP_##CTOKEN, 4, pGKernel+kernelIndex+25*inChannelGroup+(C), 1) ; \
    _vel_vstu_vssl(vrsumQ_##CTOKEN, 4, pGKernel+kernelIndex+26*inChannelGroup+(C), 1) ; \
    _vel_vstu_vssl(vrsumR_##CTOKEN, 4, pGKernel+kernelIndex+27*inChannelGroup+(C), 1) ; \
    _vel_vstu_vssl(vrsumS_##CTOKEN, 4, pGKernel+kernelIndex+28*inChannelGroup+(C), 1) ; \
    _vel_vstu_vssl(vrsumT_##CTOKEN, 4, pGKernel+kernelIndex+29*inChannelGroup+(C), 1) ; \
    _vel_vstu_vssl(vrsumU_##CTOKEN, 4, pGKernel+kernelIndex+30*inChannelGroup+(C), 1) ; \
    _vel_vstu_vssl(vrsumV_##CTOKEN, 4, pGKernel+kernelIndex+31*inChannelGroup+(C), 1) ;

    VFSUM_STORE_R1S1(0,c0)

    c+=1;
  }
  if ( ((inChannelGroup >> 1) & 0x01) == 1 ) {
    const int64_t kernelIndex = kernGroupOffset + k     * inChannelGroup + c ;

    INIT_VRSUM(c0)
    INIT_VRSUM(c1)

    {
      const int64_t vl = gOutWidth * gOutHeight ;

      for (int64_t n=0; n<batch; n++) {
	const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;

	const int64_t gOutIndex  = outGroupOffset + ((n * gOutChannel + k  ) * gOutHeight ) * gOutWidth ;

	__vr vrpin_c0 = _vel_vsfa_vvssl(vrhw, 2, (uint64_t)pInChannel, 4*vl) ;
	__vr vrin_c0 = _vel_vgtu_vvssl(vrpin_c0, 0, 0, 4*vl) ;
	__vr vrpin_c1 = _vel_vaddul_vsvl(1*4*inHeight*inWidth, vrpin_c0, 4*vl) ;
	__vr vrin_c1 = _vel_vgtu_vvssl(vrpin_c1, 0, 0, 4*vl) ;

	__vr vrgout0123 = _vel_vldu_vssl(4, pGOut+gOutIndex, 4*vl) ;
	__vr vrgout4567 = _vel_vldu_vssl(4, pGOut+gOutIndex+ 4*gOutHeight*gOutWidth, 4*vl) ;
	__vr vrgout89AB = _vel_vldu_vssl(4, pGOut+gOutIndex+ 8*gOutHeight*gOutWidth, 4*vl) ;
	__vr vrgoutCDEF = _vel_vldu_vssl(4, pGOut+gOutIndex+12*gOutHeight*gOutWidth, 4*vl) ;
	__vr vrgoutGHIJ = _vel_vldu_vssl(4, pGOut+gOutIndex+16*gOutHeight*gOutWidth, 4*vl) ;
	__vr vrgoutKLMN = _vel_vldu_vssl(4, pGOut+gOutIndex+20*gOutHeight*gOutWidth, 4*vl) ;
	__vr vrgoutOPQR = _vel_vldu_vssl(4, pGOut+gOutIndex+24*gOutHeight*gOutWidth, 4*vl) ;
	__vr vrgoutSTUV = _vel_vldu_vssl(4, pGOut+gOutIndex+28*gOutHeight*gOutWidth, 4*vl) ;

	__vr vrgout01234567 = _vel_vshf_vvvsl(vrgout0123, vrgout4567, VE_VSHUFFLE_YUZU, 4*vl) ;
	__vr vrgout89ABCDEF = _vel_vshf_vvvsl(vrgout89AB, vrgoutCDEF, VE_VSHUFFLE_YUZU, 4*vl) ;
	__vr vrgoutGHIJKLMN = _vel_vshf_vvvsl(vrgoutGHIJ, vrgoutKLMN, VE_VSHUFFLE_YUZU, 4*vl) ;
	__vr vrgoutOPQRSTUV = _vel_vshf_vvvsl(vrgoutOPQR, vrgoutSTUV, VE_VSHUFFLE_YUZU, 4*vl) ;

	VFMAD_R1S1(c0)
	VFMAD_R1S1(c1)

      } // batch
    } // gOutHeight

    VFSUM_STORE_R1S1(0,c0)
    VFSUM_STORE_R1S1(1,c1)

    c+=2 ;
  }
  if ( ((inChannelGroup >> 2) & 0x01) == 1 ) {
    const int64_t kernelIndex = kernGroupOffset + k     * inChannelGroup + c ;

    INIT_VRSUM(c0)
    INIT_VRSUM(c1)
    INIT_VRSUM(c2)
    INIT_VRSUM(c3)

    {
      const int64_t vl = gOutWidth * gOutHeight ;

      for (int64_t n=0; n<batch; n++) {
	const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;

	const int64_t gOutIndex  = outGroupOffset + ((n * gOutChannel + k  ) * gOutHeight ) * gOutWidth ;

	__vr vrpin_c0 = _vel_vsfa_vvssl(vrhw, 2, (uint64_t)pInChannel, 4*vl) ;
	__vr vrin_c0 = _vel_vgtu_vvssl(vrpin_c0, 0, 0, 4*vl) ;
	__vr vrpin_c1 = _vel_vaddul_vsvl(1*4*inHeight*inWidth, vrpin_c0, 4*vl) ;
	__vr vrin_c1 = _vel_vgtu_vvssl(vrpin_c1, 0, 0, 4*vl) ;
	__vr vrpin_c2 = _vel_vaddul_vsvl(2*4*inHeight*inWidth, vrpin_c0, 4*vl) ;
	__vr vrin_c2 = _vel_vgtu_vvssl(vrpin_c2, 0, 0, 4*vl) ;
	__vr vrpin_c3 = _vel_vaddul_vsvl(3*4*inHeight*inWidth, vrpin_c0, 4*vl) ;
	__vr vrin_c3 = _vel_vgtu_vvssl(vrpin_c3, 0, 0, 4*vl) ;

	__vr vrgout0123 = _vel_vldu_vssl(4, pGOut+gOutIndex, 4*vl) ;
	__vr vrgout4567 = _vel_vldu_vssl(4, pGOut+gOutIndex+ 4*gOutHeight*gOutWidth, 4*vl) ;
	__vr vrgout89AB = _vel_vldu_vssl(4, pGOut+gOutIndex+ 8*gOutHeight*gOutWidth, 4*vl) ;
	__vr vrgoutCDEF = _vel_vldu_vssl(4, pGOut+gOutIndex+12*gOutHeight*gOutWidth, 4*vl) ;
	__vr vrgoutGHIJ = _vel_vldu_vssl(4, pGOut+gOutIndex+16*gOutHeight*gOutWidth, 4*vl) ;
	__vr vrgoutKLMN = _vel_vldu_vssl(4, pGOut+gOutIndex+20*gOutHeight*gOutWidth, 4*vl) ;
	__vr vrgoutOPQR = _vel_vldu_vssl(4, pGOut+gOutIndex+24*gOutHeight*gOutWidth, 4*vl) ;
	__vr vrgoutSTUV = _vel_vldu_vssl(4, pGOut+gOutIndex+28*gOutHeight*gOutWidth, 4*vl) ;

	__vr vrgout01234567 = _vel_vshf_vvvsl(vrgout0123, vrgout4567, VE_VSHUFFLE_YUZU, 4*vl) ;
	__vr vrgout89ABCDEF = _vel_vshf_vvvsl(vrgout89AB, vrgoutCDEF, VE_VSHUFFLE_YUZU, 4*vl) ;
	__vr vrgoutGHIJKLMN = _vel_vshf_vvvsl(vrgoutGHIJ, vrgoutKLMN, VE_VSHUFFLE_YUZU, 4*vl) ;
	__vr vrgoutOPQRSTUV = _vel_vshf_vvvsl(vrgoutOPQR, vrgoutSTUV, VE_VSHUFFLE_YUZU, 4*vl) ;

	VFMAD_R1S1(c0)
	VFMAD_R1S1(c1)
	VFMAD_R1S1(c2)
	VFMAD_R1S1(c3)

      } // batch
    } // gOutHeight

    VFSUM_STORE_R1S1(0,c0)
    VFSUM_STORE_R1S1(1,c1)
    VFSUM_STORE_R1S1(2,c2)
    VFSUM_STORE_R1S1(3,c3)

    c+=4 ;
  }
  for (; c<inChannelGroup; c+=8) {
    const int64_t kernelIndex = kernGroupOffset + k     * inChannelGroup + c ;

    INIT_VRSUM(c0)
    INIT_VRSUM(c1)
    INIT_VRSUM(c2)
    INIT_VRSUM(c3)
    INIT_VRSUM(c4)
    INIT_VRSUM(c5)
    INIT_VRSUM(c6)
    INIT_VRSUM(c7)
#undef INIT_VRSUM

    {
      const int64_t vl = gOutWidth * gOutHeight ;

      for (int64_t n=0; n<batch; n++) {
	const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;

	const int64_t gOutIndex  = outGroupOffset + ((n * gOutChannel + k  ) * gOutHeight ) * gOutWidth ;

	__vr vrpin_c0 = _vel_vsfa_vvssl(vrhw, 2, (uint64_t)pInChannel, 4*vl) ;
	__vr vrin_c0 = _vel_vgtu_vvssl(vrpin_c0, 0, 0, 4*vl) ;
	__vr vrpin_c1 = _vel_vaddul_vsvl(1*4*inHeight*inWidth, vrpin_c0, 4*vl) ;
	__vr vrin_c1 = _vel_vgtu_vvssl(vrpin_c1, 0, 0, 4*vl) ;
	__vr vrpin_c2 = _vel_vaddul_vsvl(2*4*inHeight*inWidth, vrpin_c0, 4*vl) ;
	__vr vrin_c2 = _vel_vgtu_vvssl(vrpin_c2, 0, 0, 4*vl) ;
	__vr vrpin_c3 = _vel_vaddul_vsvl(3*4*inHeight*inWidth, vrpin_c0, 4*vl) ;
	__vr vrin_c3 = _vel_vgtu_vvssl(vrpin_c3, 0, 0, 4*vl) ;
	__vr vrpin_c4 = _vel_vaddul_vsvl(4*4*inHeight*inWidth, vrpin_c0, 4*vl) ;
	__vr vrin_c4 = _vel_vgtu_vvssl(vrpin_c4, 0, 0, 4*vl) ;
	__vr vrpin_c5 = _vel_vaddul_vsvl(5*4*inHeight*inWidth, vrpin_c0, 4*vl) ;
	__vr vrin_c5 = _vel_vgtu_vvssl(vrpin_c5, 0, 0, 4*vl) ;
	__vr vrpin_c6 = _vel_vaddul_vsvl(6*4*inHeight*inWidth, vrpin_c0, 4*vl) ;
	__vr vrin_c6 = _vel_vgtu_vvssl(vrpin_c6, 0, 0, 4*vl) ;
	__vr vrpin_c7 = _vel_vaddul_vsvl(7*4*inHeight*inWidth, vrpin_c0, 4*vl) ;
	__vr vrin_c7 = _vel_vgtu_vvssl(vrpin_c7, 0, 0, 4*vl) ;

	__vr vrgout0123 = _vel_vldu_vssl(4, pGOut+gOutIndex, 4*vl) ;
	__vr vrgout4567 = _vel_vldu_vssl(4, pGOut+gOutIndex+ 4*gOutHeight*gOutWidth, 4*vl) ;
	__vr vrgout89AB = _vel_vldu_vssl(4, pGOut+gOutIndex+ 8*gOutHeight*gOutWidth, 4*vl) ;
	__vr vrgoutCDEF = _vel_vldu_vssl(4, pGOut+gOutIndex+12*gOutHeight*gOutWidth, 4*vl) ;
	__vr vrgoutGHIJ = _vel_vldu_vssl(4, pGOut+gOutIndex+16*gOutHeight*gOutWidth, 4*vl) ;
	__vr vrgoutKLMN = _vel_vldu_vssl(4, pGOut+gOutIndex+20*gOutHeight*gOutWidth, 4*vl) ;
	__vr vrgoutOPQR = _vel_vldu_vssl(4, pGOut+gOutIndex+24*gOutHeight*gOutWidth, 4*vl) ;
	__vr vrgoutSTUV = _vel_vldu_vssl(4, pGOut+gOutIndex+28*gOutHeight*gOutWidth, 4*vl) ;

	__vr vrgout01234567 = _vel_vshf_vvvsl(vrgout0123, vrgout4567, VE_VSHUFFLE_YUZU, 4*vl) ;
	__vr vrgout89ABCDEF = _vel_vshf_vvvsl(vrgout89AB, vrgoutCDEF, VE_VSHUFFLE_YUZU, 4*vl) ;
	__vr vrgoutGHIJKLMN = _vel_vshf_vvvsl(vrgoutGHIJ, vrgoutKLMN, VE_VSHUFFLE_YUZU, 4*vl) ;
	__vr vrgoutOPQRSTUV = _vel_vshf_vvvsl(vrgoutOPQR, vrgoutSTUV, VE_VSHUFFLE_YUZU, 4*vl) ;

	VFMAD_R1S1(c0)
	VFMAD_R1S1(c1)
	VFMAD_R1S1(c2)
	VFMAD_R1S1(c3)
	VFMAD_R1S1(c4)
	VFMAD_R1S1(c5)
	VFMAD_R1S1(c6)
	VFMAD_R1S1(c7)
#undef VFMAD_R1S1
      } // batch
    } // gOutHeight

    VFSUM_STORE_R1S1(0,c0)
    VFSUM_STORE_R1S1(1,c1)
    VFSUM_STORE_R1S1(2,c2)
    VFSUM_STORE_R1S1(3,c3)
    VFSUM_STORE_R1S1(4,c4)
    VFSUM_STORE_R1S1(5,c5)
    VFSUM_STORE_R1S1(6,c6)
    VFSUM_STORE_R1S1(7,c7)
#undef VFSUM_STORE_R1S1
  }
}

vednnError_t
vednnConvolutionBackwardFilter_direct_dil1_pad0_ker1_ohwU64(
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

  const int64_t group          = pParamConv->group;
  const int64_t strideWidth    = pParamConv->strideWidth;;
  const int64_t strideHeight   = pParamConv->strideHeight;
//  const int64_t padWidth       = pParamConv->padWidth;	// must be 0
//  const int64_t padHeight      = pParamConv->padHeight;	// must be 0
//  const int64_t dilationWidth  = pParamConv->dilationWidth;	// must be 1
//  const int64_t dilationHeight = pParamConv->dilationHeight;	// must be 1

  const int64_t inChannelGroup   =  inChannel   / group;
  const int64_t gOutChannelGroup = gOutChannel  / group;

  const float *  pIn      = (const float *) pDataIn;
  const float *  pGOut    = (const float *) pDataGradOut;
  float * const  pGKernel = (float * const) pDataGradKernel;

  const int gOutPixels= gOutHeight*gOutWidth ;

#ifndef VEDNN_USE_OPENMP
  const int64_t beginOChannel = 0 ;
  const int64_t nOChannel     = gOutChannelGroup ;
#endif
  {
    const int64_t nY = VLEN / gOutWidth ;

     ;

    __vr vrseq = _vel_vseq_vl(VLEN) ;			// xy
    __vr vry  = _vel_vdivsl_vvsl(vrseq, gOutWidth, VLEN) ;
    __vr vrx  = _vel_vsubsl_vvvl(vrseq, _vel_vmulul_vsvl(gOutWidth,vry, VLEN), VLEN) ;

    __vm256 vm_k0 =  _vel_vfmkllt_mvl(_vel_vaddsl_vsvl(-gOutPixels, vrseq, VLEN), VLEN) ;
    __vm256 vm_k1 =  _vel_vfmkllt_mvl(_vel_vaddsl_vsvl(-2*gOutPixels, vrseq, VLEN), VLEN) ;
    __vm256 vm_k2 =  _vel_vfmkllt_mvl(_vel_vaddsl_vsvl(-3*gOutPixels, vrseq, VLEN), VLEN) ;
    __vm256 vm_k3 = _vel_negm_mm(vm_k2) ;
    vm_k2 = _vel_andm_mmm(vm_k2, _vel_negm_mm(vm_k1)) ;
    vm_k1 = _vel_andm_mmm(vm_k1, _vel_negm_mm(vm_k0)) ;

    __vr vri  = _vel_vmulsl_vsvl(strideHeight, vry, VLEN) ;
    __vr vrj  = _vel_vmulsl_vsvl(strideWidth,  vrx, VLEN) ;

    __vr vrhw = _vel_vaddul_vvvl(vrj, _vel_vmulul_vsvl(inWidth,vri, VLEN), VLEN) ;
    vrhw = _vel_vmrg_vvvml(vrhw, _vel_vmv_vsvl(-gOutPixels,vrhw, VLEN), vm_k1, VLEN) ;
    vrhw = _vel_vmrg_vvvml(vrhw, _vel_vmv_vsvl(-gOutPixels,vrhw, VLEN), vm_k2, VLEN) ;
    vrhw = _vel_vmrg_vvvml(vrhw, _vel_vmv_vsvl(-gOutPixels,vrhw, VLEN), vm_k3, VLEN) ;

    for (int64_t g = 0; g < group; g++) {
      int64_t inGroupOffset   = g * inChannelGroup  * inHeight  * inWidth;
      int64_t outGroupOffset  = (g * gOutChannelGroup + beginOChannel) * gOutHeight * gOutWidth;
      int64_t kernGroupOffset = (g * gOutChannelGroup + beginOChannel) * inChannelGroup * gKernHeight * gKernWidth;

      int64_t k=0;
      if ( (nOChannel & 0x01) == 1 ) {
	f1( pIn, inWidth, inHeight,
	    pGOut, gOutWidth, gOutHeight,
	    pGKernel,
	    strideHeight, strideWidth,
	    inChannelGroup, inChannel, gOutChannel,
	    inGroupOffset, outGroupOffset, kernGroupOffset,
	    batch, k,
	    vrhw
	    ) ;

	k++ ;
      }
      if ( ((nOChannel >> 1) & 0x01) == 1 ) {
	f2( pIn, inWidth, inHeight,
	    pGOut, gOutWidth, gOutHeight,
	    pGKernel,
	    strideHeight, strideWidth,
	    inChannelGroup, inChannel, gOutChannel,
	    inGroupOffset, outGroupOffset, kernGroupOffset,
	    batch, k,
	    vrhw
	    ) ;
	k+=2;
      }
      if ( ((nOChannel >> 2) & 0x01) == 1 ) {
	f4( pIn, inWidth, inHeight,
	    pGOut, gOutWidth, gOutHeight,
	    pGKernel,
	    strideHeight, strideWidth,
	    inChannelGroup, inChannel, gOutChannel,
	    inGroupOffset, outGroupOffset, kernGroupOffset,
	    batch, k,
	    vrhw, vm_k0, vm_k1
	    ) ;
	k+=4;
      }
      if ( ((nOChannel >> 3) & 0x01) == 1 ) {
	f8( pIn, inWidth, inHeight,
	    pGOut, gOutWidth, gOutHeight,
	    pGKernel,
	    strideHeight, strideWidth,
	    inChannelGroup, inChannel, gOutChannel,
	    inGroupOffset, outGroupOffset, kernGroupOffset,
	    batch, k,
	    vrhw, vm_k0, vm_k1
	    ) ;
	k+=8;
      }
      if ( ((nOChannel >> 4) & 0x01) == 1 ) {
	f16( pIn, inWidth, inHeight,
	    pGOut, gOutWidth, gOutHeight,
	    pGKernel,
	    strideHeight, strideWidth,
	    inChannelGroup, inChannel, gOutChannel,
	    inGroupOffset, outGroupOffset, kernGroupOffset,
	    batch, k,
	    vrhw,
	    vm_k0, vm_k1, vm_k2, vm_k3
	    ) ;
	k+=16 ;
      }
      for ( ;k<nOChannel; k+=32) {
	f32( pIn, inWidth, inHeight,
	    pGOut, gOutWidth, gOutHeight,
	    pGKernel,
	    strideHeight, strideWidth,
	    inChannelGroup, inChannel, gOutChannel,
	    inGroupOffset, outGroupOffset, kernGroupOffset,
	    batch, k,
	    vrhw,
	    vm_k0, vm_k1, vm_k2, vm_k3
	    ) ;

      } // outChannel
    } // group
  }

  return VEDNN_SUCCESS;
}
#endif


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
    const __vm vm1,
    const __vm vm2,
    const __vm vm3
)
{
  const int64_t d0 = NUMKERNEL & 0x1 ;
  const int64_t d1 = (NUMKERNEL >> 1) & 0x1 ;
  const int64_t d2 = (NUMKERNEL >> 2) & 0x1 ;
  const int64_t d3 = (NUMKERNEL >> 3) ;

  int64_t c=0 ;
  if( (inChannelGroup & 0x1 ) == 1 ) {
    __vr vrsum_d0_c0 = _vel_vbrds_vsl(0.f, VLEN) ;
    __vr vrsum_d1_c0 = _vel_pvbrd_vsl(0UL, VLEN) ;
    __vr vrsum_d2_c0 = _vel_pvbrd_vsl(0UL, VLEN) ;
    __vr vrsum_d3_c0[d3] ;
#pragma clang loop unroll(full)
    for(int64_t kk=0; kk<d3; kk++) {
      vrsum_d3_c0[kk] = _vel_pvbrd_vsl(0UL, VLEN) ;
    }

    {
      const int64_t vl   = gOutWidth * gOutHeight ;
      const int64_t vl_i = NUMKERNEL >= 8 ? 4 * vl : (NUMKERNEL >= 4 ? 2 * vl : vl) ;

      for (int64_t n=0; n<batch; n++) {

	const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;

	const int64_t outIndex = gOutGroupOffset + ((n * gOutChannel + k) * gOutHeight ) * gOutWidth ;

	__vr vrpin_c0 = _vel_vsfa_vvssl(vrhw, 2, (uint64_t)(pInChannel+0*inHeight*inWidth), vl_i) ;
	__vr vrin_c0  = _vel_vgtu_vvssl(vrpin_c0, 0, 0, vl_i) ;

	__vr vrgout_d3[2*d3] ;
	__vr vrgout_d2[2] ;
	__vr vrgout_d1[2] ;
	__vr vrgout_d0 ;
#pragma clang loop unroll(full)
	for(int64_t kk=0; kk<8*d3; kk++) {
	  vrgout_d3[kk] = _vel_vldu_vssl(4, pGOut+outIndex+4*kk*gOutHeight*gOutWidth, 4*vl) ;
	}
	if( d2 ) {
	  vrgout_d2[0] = _vel_vldu_vssl(4, pGOut+outIndex+(8*d3+2*0)*gOutHeight*gOutWidth, 2*vl) ;
	  vrgout_d2[1] = _vel_vldu_vssl(4, pGOut+outIndex+(8*d3+2*1)*gOutHeight*gOutWidth, 2*vl) ;
	}
	if( d1 ) {
	  vrgout_d1[0] = _vel_vldu_vssl(4, pGOut+outIndex+(8*d3+4*d2+0)*gOutHeight*gOutWidth, vl) ;
	  vrgout_d1[1] = _vel_vldu_vssl(4, pGOut+outIndex+(8*d3+4*d2+1)*gOutHeight*gOutWidth, vl) ;
	}
	if( d0 ) {
	  vrgout_d0 = _vel_vldu_vssl(4, pGOut+outIndex+(8*d3+4*d2+2*d1)*gOutHeight*gOutWidth, vl) ;
	}

	__vr vrgoutp_d3[d3] ;
	__vr vrgoutp_d2 ;
	__vr vrgoutp_d1 ;
#pragma clang loop unroll(full)
	for(int64_t kk=0; kk<d3; kk++) {
	  vrgoutp_d3[kk] = _vel_vshf_vvvsl(vrgout_d3[2*kk+0], vrgout_d3[2*kk+1], VE_VSHUFFLE_YUZU, 4*vl) ;
	}
	if( d2 ) {
	  vrgoutp_d2 = _vel_vshf_vvvsl(vrgout_d2[0], vrgout_d2[1], VE_VSHUFFLE_YUZU, 2*vl) ;
	}
	if( d1 ) {
	  vrgoutp_d1 = _vel_vshf_vvvsl(vrgout_d1[0], vrgout_d1[1], VE_VSHUFFLE_YUZU, vl) ;
	}

#define VFADD_R1S1(CTOKEN)													\
	{															\
	  __vr vrinP_##CTOKEN = _vel_vshf_vvvsl(vrin_##CTOKEN, vrin_##CTOKEN, VE_VSHUFFLE_YUZU, vl_i) ;				\
  	  _Pragma("clang loop unroll(full)")											\
	  for(int64_t kk=0; kk<d3; kk++) {											\
	    vrsum_d3_##CTOKEN[kk] = _vel_pvfmad_vvvvvl(vrsum_d3_##CTOKEN[kk], vrinP_##CTOKEN, vrgoutp_d3[kk], vrsum_d3_##CTOKEN[kk], 4*vl) ;	\
	  }															\
	  if( d2 ) {														\
	    vrsum_d2_##CTOKEN = _vel_pvfmad_vvvvvl(vrsum_d2_##CTOKEN, vrinP_##CTOKEN, vrgoutp_d2, vrsum_d2_##CTOKEN, 2*vl) ;	\
	  }															\
	  if( d1 ) {														\
	    vrsum_d1_##CTOKEN = _vel_pvfmad_vvvvvl(vrsum_d1_##CTOKEN, vrinP_##CTOKEN, vrgoutp_d1, vrsum_d1_##CTOKEN, vl) ;	\
	  }															\
	  if( d0 ) {														\
	    vrsum_d0_##CTOKEN  = _vel_vfmads_vvvvvl(vrsum_d0_##CTOKEN, vrin_##CTOKEN, vrgout_d0, vrsum_d0_##CTOKEN, vl) ;	\
	  }															\
	}

	VFADD_R1S1(c0) ;

      } // batch
    } // gOutHeight

#define FILTER_OFFSET(k,c,r,s) ( gKernGroupOffset + filter_index<FLAYOUT>(k,c,r,s, inChannelGroup, gOutChannelGroup, 1, 1) )

#define VFSUM_STORE_R1S1(C,CTOKEN) 										\
    {														\
      _Pragma("clang loop unroll(full)")									\
      for(int64_t kk=0; kk<d3; kk++) {										\
	__vr vrsum0_##CTOKEN = _vel_vfsums_vvml(vrsum_d3_##CTOKEN[kk], vm0, VLEN) ; 				\
	_vel_vstu_vssl(vrsum0_##CTOKEN, 4, pGKernel+FILTER_OFFSET(k+8*kk+0,c+(C),0,0), 1) ;			\
	__vr vrsum1_##CTOKEN = _vel_vfsums_vvml(vrsum_d3_##CTOKEN[kk], vm1, VLEN) ; 				\
	_vel_vstu_vssl(vrsum1_##CTOKEN, 4, pGKernel+FILTER_OFFSET(k+8*kk+1,c+(C),0,0), 1) ;			\
	__vr vrsum2_##CTOKEN = _vel_vfsums_vvml(vrsum_d3_##CTOKEN[kk], vm2, VLEN) ; 				\
	_vel_vstu_vssl(vrsum2_##CTOKEN, 4, pGKernel+FILTER_OFFSET(k+8*kk+2,c+(C),0,0), 1) ;			\
	__vr vrsum3_##CTOKEN = _vel_vfsums_vvml(vrsum_d3_##CTOKEN[kk], vm3, VLEN) ; 				\
	_vel_vstu_vssl(vrsum3_##CTOKEN, 4, pGKernel+FILTER_OFFSET(k+8*kk+3,c+(C),0,0), 1) ;			\
	__vr vrsum4_##CTOKEN = _vel_vfsums_vvml(_vel_vsll_vvsl(vrsum_d3_##CTOKEN[kk],32, VLEN), vm0, VLEN); 	\
	_vel_vstu_vssl(vrsum4_##CTOKEN, 4, pGKernel+FILTER_OFFSET(k+8*kk+4,c+(C),0,0), 1) ;			\
	__vr vrsum5_##CTOKEN = _vel_vfsums_vvml(_vel_vsll_vvsl(vrsum_d3_##CTOKEN[kk],32, VLEN), vm1, VLEN); 	\
	_vel_vstu_vssl(vrsum5_##CTOKEN, 4, pGKernel+FILTER_OFFSET(k+8*kk+5,c+(C),0,0), 1) ;			\
	__vr vrsum6_##CTOKEN = _vel_vfsums_vvml(_vel_vsll_vvsl(vrsum_d3_##CTOKEN[kk],32, VLEN), vm2, VLEN); 	\
	_vel_vstu_vssl(vrsum6_##CTOKEN, 4, pGKernel+FILTER_OFFSET(k+8*kk+6,c+(C),0,0), 1) ;			\
	__vr vrsum7_##CTOKEN = _vel_vfsums_vvml(_vel_vsll_vvsl(vrsum_d3_##CTOKEN[kk],32, VLEN), vm3, VLEN); 	\
	_vel_vstu_vssl(vrsum7_##CTOKEN, 4, pGKernel+FILTER_OFFSET(k+8*kk+7,c+(C),0,0), 1) ;			\
      }														\
      if( d2 ) {												\
	__vr vrsum0_##CTOKEN = _vel_vfsums_vvml(vrsum_d2_##CTOKEN, vm0, VLEN) ;					\
	_vel_vstu_vssl(vrsum0_##CTOKEN, 4, pGKernel+FILTER_OFFSET(k+8*d3+0,c+(C),0,0), 1) ;			\
	__vr vrsum1_##CTOKEN = _vel_vfsums_vvml(vrsum_d2_##CTOKEN, vm1, VLEN) ;					\
	_vel_vstu_vssl(vrsum1_##CTOKEN, 4, pGKernel+FILTER_OFFSET(k+8*d3+1,c+(C),0,0), 1) ;			\
	__vr vrsum2_##CTOKEN = _vel_vfsums_vvml(_vel_vsll_vvsl(vrsum_d2_##CTOKEN,32, VLEN), vm0, VLEN);		\
	_vel_vstu_vssl(vrsum2_##CTOKEN, 4, pGKernel+FILTER_OFFSET(k+8*d3+2,c+(C),0,0), 1) ;			\
	__vr vrsum3_##CTOKEN = _vel_vfsums_vvml(_vel_vsll_vvsl(vrsum_d2_##CTOKEN,32, VLEN), vm1, VLEN);		\
	_vel_vstu_vssl(vrsum3_##CTOKEN, 4, pGKernel+FILTER_OFFSET(k+8*d3+3,c+(C),0,0), 1) ;			\
      }														\
      if( d1 ) {												\
	__vr vrsum0_##CTOKEN = _vel_vfsums_vvl(vrsum_d1_##CTOKEN, VLEN) ;					\
	_vel_vstu_vssl(vrsum0_##CTOKEN, 4, pGKernel + FILTER_OFFSET(k+8*d3+4*d2,  c+(C),0,0), 1) ;		\
	__vr vrsum1_##CTOKEN = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsum_d1_##CTOKEN,32, VLEN), VLEN);		\
	_vel_vstu_vssl(vrsum1_##CTOKEN, 4, pGKernel + FILTER_OFFSET(k+8*d3+4*d2+1,c+(C),0,0), 1) ;		\
      }														\
      if( d0 ) {												\
	vrsum_d0_##CTOKEN = _vel_vfsums_vvl(vrsum_d0_##CTOKEN, VLEN) ;						\
	_vel_vstu_vssl(vrsum_d0_##CTOKEN, 4, pGKernel+FILTER_OFFSET(k+8*d3+4*d2+2*d1,c+(C),0,0), 1) ;		\
      }														\
    }
    VFSUM_STORE_R1S1(0,c0) ;

    c+=1;
  }
  if ( ((inChannelGroup >>1) & 0x1 ) == 1) {
    __vr vrsum_d0_c0 = _vel_vbrds_vsl(0.f, VLEN) ;
    __vr vrsum_d0_c1 = _vel_vbrds_vsl(0.f, VLEN) ;
    __vr vrsum_d1_c0 = _vel_pvbrd_vsl(0UL, VLEN) ;
    __vr vrsum_d1_c1 = _vel_pvbrd_vsl(0UL, VLEN) ;
    __vr vrsum_d2_c0 = _vel_pvbrd_vsl(0UL, VLEN) ;
    __vr vrsum_d2_c1 = _vel_pvbrd_vsl(0UL, VLEN) ;
    __vr vrsum_d3_c0[d3] ;
    __vr vrsum_d3_c1[d3] ;
#pragma clang loop unroll(full)
    for(int64_t kk=0; kk<d3; kk++) {
      vrsum_d3_c0[kk] = _vel_pvbrd_vsl(0UL, VLEN) ;
      vrsum_d3_c1[kk] = _vel_pvbrd_vsl(0UL, VLEN) ;
    }

    {
      const int64_t vl   = gOutWidth * gOutHeight ;
      const int64_t vl_i = NUMKERNEL >= 8 ? 4 * vl : (NUMKERNEL >= 4 ? 2 * vl : vl) ;

      for (int64_t n=0; n<batch; n++) {

	const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;

	const int64_t outIndex = gOutGroupOffset + ((n * gOutChannel + k) * gOutHeight ) * gOutWidth ;

	__vr vrpin_c0 = _vel_vsfa_vvssl(vrhw, 2, (uint64_t)(pInChannel+0*inHeight*inWidth), vl_i) ;
	__vr vrin_c0  = _vel_vgtu_vvssl(vrpin_c0, 0, 0, vl_i) ;
	__vr vrpin_c1 = _vel_vsfa_vvssl(vrhw, 2, (uint64_t)(pInChannel+1*inHeight*inWidth), vl_i) ;
	__vr vrin_c1  = _vel_vgtu_vvssl(vrpin_c1, 0, 0, vl_i) ;


	__vr vrgout_d3[2*d3] ;
	__vr vrgout_d2[2] ;
	__vr vrgout_d1[2] ;
	__vr vrgout_d0 ;
#pragma clang loop unroll(full)
	for(int64_t kk=0; kk<8*d3; kk++) {
	  vrgout_d3[kk] = _vel_vldu_vssl(4, pGOut+outIndex+4*kk*gOutHeight*gOutWidth, 4*vl) ;
	}
	if( d2 ) {
	  vrgout_d2[0] = _vel_vldu_vssl(4, pGOut+outIndex+(8*d3+2*0)*gOutHeight*gOutWidth, 2*vl) ;
	  vrgout_d2[1] = _vel_vldu_vssl(4, pGOut+outIndex+(8*d3+2*1)*gOutHeight*gOutWidth, 2*vl) ;
	}
	if( d1 ) {
	  vrgout_d1[0] = _vel_vldu_vssl(4, pGOut+outIndex+(8*d3+4*d2+0)*gOutHeight*gOutWidth, vl) ;
	  vrgout_d1[1] = _vel_vldu_vssl(4, pGOut+outIndex+(8*d3+4*d2+1)*gOutHeight*gOutWidth, vl) ;
	}
	if( d0 ) {
	  vrgout_d0 = _vel_vldu_vssl(4, pGOut+outIndex+(8*d3+4*d2+2*d1)*gOutHeight*gOutWidth, vl) ;
	}

	__vr vrgoutp_d3[d3] ;
	__vr vrgoutp_d2 ;
	__vr vrgoutp_d1 ;
#pragma clang loop unroll(full)
	for(int64_t kk=0; kk<d3; kk++) {
	  vrgoutp_d3[kk] = _vel_vshf_vvvsl(vrgout_d3[2*kk+0], vrgout_d3[2*kk+1], VE_VSHUFFLE_YUZU, 4*vl) ;
	}
	if( d2 ) {
	  vrgoutp_d2 = _vel_vshf_vvvsl(vrgout_d2[0], vrgout_d2[1], VE_VSHUFFLE_YUZU, 2*vl) ;
	}
	if( d1 ) {
	  vrgoutp_d1 = _vel_vshf_vvvsl(vrgout_d1[0], vrgout_d1[1], VE_VSHUFFLE_YUZU, vl) ;
	}

	VFADD_R1S1(c0) ;
	VFADD_R1S1(c1) ;

      } // batch
    } // gOutHeight

    VFSUM_STORE_R1S1(0,c0) ;
    VFSUM_STORE_R1S1(1,c1) ;

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
    __vr vrsum_d2_c0 = _vel_pvbrd_vsl(0UL, VLEN) ;
    __vr vrsum_d2_c1 = _vel_pvbrd_vsl(0UL, VLEN) ;
    __vr vrsum_d2_c2 = _vel_pvbrd_vsl(0UL, VLEN) ;
    __vr vrsum_d2_c3 = _vel_pvbrd_vsl(0UL, VLEN) ;
    __vr vrsum_d3_c0[d3] ;
    __vr vrsum_d3_c1[d3] ;
    __vr vrsum_d3_c2[d3] ;
    __vr vrsum_d3_c3[d3] ;

#pragma clang loop unroll(full)
    for(int64_t kk=0; kk<d3; kk++) {
      vrsum_d3_c0[kk] = _vel_pvbrd_vsl(0UL, VLEN) ;
      vrsum_d3_c1[kk] = _vel_pvbrd_vsl(0UL, VLEN) ;
      vrsum_d3_c2[kk] = _vel_pvbrd_vsl(0UL, VLEN) ;
      vrsum_d3_c3[kk] = _vel_pvbrd_vsl(0UL, VLEN) ;
    }

    {
      const int64_t vl   = gOutWidth * gOutHeight ;
      const int64_t vl_i = NUMKERNEL >= 8 ? 4 * vl : (NUMKERNEL >= 4 ? 2 * vl : vl) ;

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


	__vr vrgout_d3[2*d3] ;
	__vr vrgout_d2[2] ;
	__vr vrgout_d1[2] ;
	__vr vrgout_d0 ;
#pragma clang loop unroll(full)
	for(int64_t kk=0; kk<8*d3; kk++) {
	  vrgout_d3[kk] = _vel_vldu_vssl(4, pGOut+outIndex+4*kk*gOutHeight*gOutWidth, 4*vl) ;
	}
	if( d2 ) {
	  vrgout_d2[0] = _vel_vldu_vssl(4, pGOut+outIndex+(8*d3+2*0)*gOutHeight*gOutWidth, 2*vl) ;
	  vrgout_d2[1] = _vel_vldu_vssl(4, pGOut+outIndex+(8*d3+2*1)*gOutHeight*gOutWidth, 2*vl) ;
	}
	if( d1 ) {
	  vrgout_d1[0] = _vel_vldu_vssl(4, pGOut+outIndex+(8*d3+4*d2+0)*gOutHeight*gOutWidth, vl) ;
	  vrgout_d1[1] = _vel_vldu_vssl(4, pGOut+outIndex+(8*d3+4*d2+1)*gOutHeight*gOutWidth, vl) ;
	}
	if( d0 ) {
	  vrgout_d0 = _vel_vldu_vssl(4, pGOut+outIndex+(8*d3+4*d2+2*d1)*gOutHeight*gOutWidth, vl) ;
	}

	__vr vrgoutp_d3[d3] ;
	__vr vrgoutp_d2 ;
	__vr vrgoutp_d1 ;
#pragma clang loop unroll(full)
	for(int64_t kk=0; kk<d3; kk++) {
	  vrgoutp_d3[kk] = _vel_vshf_vvvsl(vrgout_d3[2*kk+0], vrgout_d3[2*kk+1], VE_VSHUFFLE_YUZU, 4*vl) ;
	}
	if( d2 ) {
	  vrgoutp_d2 = _vel_vshf_vvvsl(vrgout_d2[0], vrgout_d2[1], VE_VSHUFFLE_YUZU, 2*vl) ;
	}
	if( d1 ) {
	  vrgoutp_d1 = _vel_vshf_vvvsl(vrgout_d1[0], vrgout_d1[1], VE_VSHUFFLE_YUZU, vl) ;
	}

	VFADD_R1S1(c0) ;
	VFADD_R1S1(c1) ;
	VFADD_R1S1(c2) ;
	VFADD_R1S1(c3) ;

#undef VFADD_R1S1
      } // batch
    } // gOutHeight

    VFSUM_STORE_R1S1(0,c0) ;
    VFSUM_STORE_R1S1(1,c1) ;
    VFSUM_STORE_R1S1(2,c2) ;
    VFSUM_STORE_R1S1(3,c3) ;

#undef VFSUM_STORE_R1S1
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
  __vm256 vm1 =  _vel_vfmkllt_mvl(_vel_vaddsl_vsvl(-2*gOutHeight*gOutWidth, vrseq, VLEN), VLEN) ;
  __vm256 vm2 =  _vel_vfmkllt_mvl(_vel_vaddsl_vsvl(-3*gOutHeight*gOutWidth, vrseq, VLEN), VLEN) ;
  __vm256 vm3 = _vel_negm_mm(vm2) ;
  vm2 = _vel_andm_mmm(vm2, _vel_negm_mm(vm1)) ;
  vm1 = _vel_andm_mmm(vm1, _vel_negm_mm(vm0)) ;

  __vr vri  = _vel_vmulsl_vsvl(strideHeight, vry, VLEN) ;
  __vr vrj  = _vel_vmulsl_vsvl(strideWidth,  vrx, VLEN) ;

  __vr vrhw = _vel_vaddul_vvvl(vrj, _vel_vmulul_vsvl(inWidth,vri, VLEN), VLEN) ;
  vrhw = _vel_vmrg_vvvml(vrhw, _vel_vmv_vsvl(-gOutHeight*gOutWidth,vrhw, VLEN), vm1, VLEN) ;
  vrhw = _vel_vmrg_vvvml(vrhw, _vel_vmv_vsvl(-gOutHeight*gOutWidth,vrhw, VLEN), vm2, VLEN) ;
  vrhw = _vel_vmrg_vvvml(vrhw, _vel_vmv_vsvl(-gOutHeight*gOutWidth,vrhw, VLEN), vm3, VLEN) ;

  for (int64_t g = 0; g < group; g++) {
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
	 nY, vrhw, vm0, vm1, vm2, vm3 ) ;
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
	 nY, vrhw, vm0, vm1, vm2, vm3 ) ;
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
	 nY, vrhw, vm0, vm1, vm2, vm3 ) ;
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
	 nY, vrhw, vm0, vm1, vm2, vm3 ) ;
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
	 nY, vrhw, vm0, vm1, vm2, vm3 ) ;
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
	 nY, vrhw, vm0, vm1, vm2, vm3 ) ;
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
	 nY, vrhw, vm0, vm1, vm2, vm3 ) ;
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
	 nY, vrhw, vm0, vm1, vm2, vm3 ) ;
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
	 nY, vrhw, vm0, vm1, vm2, vm3 ) ;
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
	 nY, vrhw, vm0, vm1, vm2, vm3 ) ;
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
	 nY, vrhw, vm0, vm1, vm2, vm3 ) ;
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
	 nY, vrhw, vm0, vm1, vm2, vm3 ) ;
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
	 nY, vrhw, vm0, vm1, vm2, vm3 ) ;
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
	 nY, vrhw, vm0, vm1, vm2, vm3 ) ;
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
	 nY, vrhw, vm0, vm1, vm2, vm3 ) ;
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
	 nY, vrhw, vm0, vm1, vm2, vm3 ) ;
      k+=16 ;
    } // gOutChannel
  } // group
}


extern "C" vednnError_t
vednnConvolutionBackwardFilter_direct_dil1_pad0_ker1_ohwU64(
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

