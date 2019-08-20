#include <stdint.h>
#include <stdlib.h>

#include "vednn.h"

#include "veintrin.h"
#define VLEN	(256)

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

    _ve_lvl(VLEN) ;

#define INIT_VRSUM(CTOKEN) \
    __vr vrsum_##CTOKEN = _ve_vbrdu_vs_f32(0UL) ;

    INIT_VRSUM(c0)

    {

      const int64_t vl = gOutWidth * gOutHeight ;

      _ve_lvl(vl) ;

      for (int64_t n=0; n<batch; n++) {
	const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;

	const int64_t gOutIndex0  = outGroupOffset + ((n * gOutChannel + k  ) * gOutHeight ) * gOutWidth ;

	__vr vrpin_c0 = _ve_vsfa_vvss(vrhw, 2, (uint64_t)pInChannel) ;
	__vr vrin_c0 = _ve_vgtu_vv(vrpin_c0) ;

	__vr vrgout0 = _ve_vldu_vss(4, pGOut+gOutIndex0) ;

#define VFMAD_R1S1(CTOKEN) \
          vrsum_##CTOKEN = _ve_vfmads_vvvv(vrsum_##CTOKEN, vrin_##CTOKEN, vrgout0) ;

	  VFMAD_R1S1(c0)

      } // batch
    } // gOutHeight

#define VFSUM_STORE_R1S1(C,CTOKEN) \
    _ve_lvl(VLEN) ; \
    vrsum_##CTOKEN = _ve_vfsums_vv(vrsum_##CTOKEN) ; \
    _ve_lvl(1) ; \
    _ve_vstu_vss(vrsum_##CTOKEN, 4, pGKernel+kernelIndex0+(C)) ;

    VFSUM_STORE_R1S1(0,c0)

    c+=1;
  }
  if ( ((inChannelGroup >> 1) & 0x01) == 1 ) {
    const int64_t kernelIndex0 = kernGroupOffset + k     * inChannelGroup + c ;

    _ve_lvl(VLEN) ;

    INIT_VRSUM(c0)
    INIT_VRSUM(c1)

    {

      const int64_t vl = gOutWidth * gOutHeight ;

      _ve_lvl(vl) ;

      for (int64_t n=0; n<batch; n++) {
	const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;

	const int64_t gOutIndex0  = outGroupOffset + ((n * gOutChannel + k  ) * gOutHeight ) * gOutWidth ;

	__vr vrpin_c0 = _ve_vsfa_vvss(vrhw, 2, (uint64_t)pInChannel) ;
	__vr vrin_c0 = _ve_vgtu_vv(vrpin_c0) ;
	__vr vrpin_c1 = _ve_vaddul_vsv(1*4*inHeight*inWidth, vrpin_c0) ;
	__vr vrin_c1 = _ve_vgtu_vv(vrpin_c1) ;

	__vr vrgout0 = _ve_vldu_vss(4, pGOut+gOutIndex0) ;

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

    _ve_lvl(VLEN) ;

    INIT_VRSUM(c0)
    INIT_VRSUM(c1)
    INIT_VRSUM(c2)
    INIT_VRSUM(c3)

    {
      const int64_t vl = gOutWidth * gOutHeight ;

      _ve_lvl(vl) ;

      for (int64_t n=0; n<batch; n++) {
	const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;

	const int64_t gOutIndex0  = outGroupOffset + ((n * gOutChannel + k  ) * gOutHeight ) * gOutWidth ;

	__vr vrpin_c0 = _ve_vsfa_vvss(vrhw, 2, (uint64_t)pInChannel) ;
	__vr vrin_c0 = _ve_vgtu_vv(vrpin_c0) ;
	__vr vrpin_c1 = _ve_vaddul_vsv(1*4*inHeight*inWidth, vrpin_c0) ;
	__vr vrin_c1 = _ve_vgtu_vv(vrpin_c1) ;
	__vr vrpin_c2 = _ve_vaddul_vsv(2*4*inHeight*inWidth, vrpin_c0) ;
	__vr vrin_c2 = _ve_vgtu_vv(vrpin_c2) ;
	__vr vrpin_c3 = _ve_vaddul_vsv(3*4*inHeight*inWidth, vrpin_c0) ;
	__vr vrin_c3 = _ve_vgtu_vv(vrpin_c3) ;

	__vr vrgout0 = _ve_vldu_vss(4, pGOut+gOutIndex0) ;

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

    _ve_lvl(VLEN) ;

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

      _ve_lvl(vl) ;

      for (int64_t n=0; n<batch; n++) {
	const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;

	const int64_t gOutIndex0  = outGroupOffset + ((n * gOutChannel + k  ) * gOutHeight ) * gOutWidth ;

	__vr vrpin_c0 = _ve_vsfa_vvss(vrhw, 2, (uint64_t)pInChannel) ;
	__vr vrin_c0 = _ve_vgtu_vv(vrpin_c0) ;
	__vr vrpin_c1 = _ve_vaddul_vsv(1*4*inHeight*inWidth, vrpin_c0) ;
	__vr vrin_c1 = _ve_vgtu_vv(vrpin_c1) ;
	__vr vrpin_c2 = _ve_vaddul_vsv(2*4*inHeight*inWidth, vrpin_c0) ;
	__vr vrin_c2 = _ve_vgtu_vv(vrpin_c2) ;
	__vr vrpin_c3 = _ve_vaddul_vsv(3*4*inHeight*inWidth, vrpin_c0) ;
	__vr vrin_c3 = _ve_vgtu_vv(vrpin_c3) ;
	__vr vrpin_c4 = _ve_vaddul_vsv(4*4*inHeight*inWidth, vrpin_c0) ;
	__vr vrin_c4 = _ve_vgtu_vv(vrpin_c4) ;
	__vr vrpin_c5 = _ve_vaddul_vsv(5*4*inHeight*inWidth, vrpin_c0) ;
	__vr vrin_c5 = _ve_vgtu_vv(vrpin_c5) ;
	__vr vrpin_c6 = _ve_vaddul_vsv(6*4*inHeight*inWidth, vrpin_c0) ;
	__vr vrin_c6 = _ve_vgtu_vv(vrpin_c6) ;
	__vr vrpin_c7 = _ve_vaddul_vsv(7*4*inHeight*inWidth, vrpin_c0) ;
	__vr vrin_c7 = _ve_vgtu_vv(vrpin_c7) ;

	__vr vrgout0 = _ve_vldu_vss(4, pGOut+gOutIndex0) ;

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

    _ve_lvl(VLEN) ;

#define INIT_VRSUM(CTOKEN) \
    __vr vrsum01_##CTOKEN = _ve_vbrd_vs_i64(0UL) ;

    INIT_VRSUM(c0)

    {

      const int64_t vl = gOutWidth * gOutHeight ;

      _ve_lvl(vl) ;

      for (int64_t n=0; n<batch; n++) {
	const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;

	const int64_t gOutIndex0  = outGroupOffset + ((n * gOutChannel + k  ) * gOutHeight ) * gOutWidth ;
	const int64_t gOutIndex1  = outGroupOffset + ((n * gOutChannel + k+1) * gOutHeight ) * gOutWidth ;

	__vr vrpin_c0 = _ve_vsfa_vvss(vrhw, 2, (uint64_t)pInChannel) ;
	__vr vrin_c0 = _ve_vgtu_vv(vrpin_c0) ;

	__vr vrgout0 = _ve_vldu_vss(4, pGOut+gOutIndex0) ;
	__vr vrgout1 = _ve_vldu_vss(4, pGOut+gOutIndex1) ;

	__vr vrgout01 = _ve_vshf_vvvs(vrgout0, vrgout1, VE_VSHUFFLE_YUZU) ;

#define VFMAD_R1S1(CTOKEN) \
          __vr vrinP_##CTOKEN = _ve_vshf_vvvs(vrin_##CTOKEN, vrin_##CTOKEN, VE_VSHUFFLE_YUZU) ; \
          vrsum01_##CTOKEN = _ve_pvfmad_vvvv(vrsum01_##CTOKEN, vrinP_##CTOKEN, vrgout01) ;

	  VFMAD_R1S1(c0)

      } // batch
    } // gOutHeight

#define VFSUM_STORE_R1S1(C,CTOKEN) \
    _ve_lvl(VLEN) ; \
    __vr vrsum0_##CTOKEN = _ve_vfsums_vv(vrsum01_##CTOKEN) ; \
    __vr vrsum1_##CTOKEN = _ve_vfsums_vv(_ve_vsll_vvs(vrsum01_##CTOKEN,32)); \
    _ve_lvl(1) ; \
    _ve_vstu_vss(vrsum0_##CTOKEN, 4, pGKernel+kernelIndex0+(C)) ; \
    _ve_vstu_vss(vrsum1_##CTOKEN, 4, pGKernel+kernelIndex1+(C)) ;

    VFSUM_STORE_R1S1(0,c0)

    c+=1;
  }
  if ( ((inChannelGroup >> 1) & 0x01) == 1 ) {
    const int64_t kernelIndex0 = kernGroupOffset + k     * inChannelGroup + c ;
    const int64_t kernelIndex1 = kernGroupOffset + (k+1) * inChannelGroup + c ;

    _ve_lvl(VLEN) ;

    INIT_VRSUM(c0)
    INIT_VRSUM(c1)

    {

      const int64_t vl = gOutWidth * gOutHeight ;

      _ve_lvl(vl) ;

      for (int64_t n=0; n<batch; n++) {
	const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;

	const int64_t gOutIndex0  = outGroupOffset + ((n * gOutChannel + k  ) * gOutHeight ) * gOutWidth ;
	const int64_t gOutIndex1  = outGroupOffset + ((n * gOutChannel + k+1) * gOutHeight ) * gOutWidth ;

	__vr vrpin_c0 = _ve_vsfa_vvss(vrhw, 2, (uint64_t)pInChannel) ;
	__vr vrin_c0 = _ve_vgtu_vv(vrpin_c0) ;
	__vr vrpin_c1 = _ve_vaddul_vsv(1*4*inHeight*inWidth, vrpin_c0) ;
	__vr vrin_c1 = _ve_vgtu_vv(vrpin_c1) ;

	__vr vrgout0 = _ve_vldu_vss(4, pGOut+gOutIndex0) ;
	__vr vrgout1 = _ve_vldu_vss(4, pGOut+gOutIndex1) ;

	__vr vrgout01 = _ve_vshf_vvvs(vrgout0, vrgout1, VE_VSHUFFLE_YUZU) ;

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

    _ve_lvl(VLEN) ;

    INIT_VRSUM(c0)
    INIT_VRSUM(c1)
    INIT_VRSUM(c2)
    INIT_VRSUM(c3)

    {

      const int64_t vl = gOutWidth * gOutHeight ;

      _ve_lvl(vl) ;

      for (int64_t n=0; n<batch; n++) {
	const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;

	const int64_t gOutIndex0  = outGroupOffset + ((n * gOutChannel + k  ) * gOutHeight ) * gOutWidth ;
	const int64_t gOutIndex1  = outGroupOffset + ((n * gOutChannel + k+1) * gOutHeight ) * gOutWidth ;

	__vr vrpin_c0 = _ve_vsfa_vvss(vrhw, 2, (uint64_t)pInChannel) ;
	__vr vrin_c0 = _ve_vgtu_vv(vrpin_c0) ;
	__vr vrpin_c1 = _ve_vaddul_vsv(1*4*inHeight*inWidth, vrpin_c0) ;
	__vr vrin_c1 = _ve_vgtu_vv(vrpin_c1) ;
	__vr vrpin_c2 = _ve_vaddul_vsv(2*4*inHeight*inWidth, vrpin_c0) ;
	__vr vrin_c2 = _ve_vgtu_vv(vrpin_c2) ;
	__vr vrpin_c3 = _ve_vaddul_vsv(3*4*inHeight*inWidth, vrpin_c0) ;
	__vr vrin_c3 = _ve_vgtu_vv(vrpin_c3) ;

	__vr vrgout0 = _ve_vldu_vss(4, pGOut+gOutIndex0) ;
	__vr vrgout1 = _ve_vldu_vss(4, pGOut+gOutIndex1) ;

	__vr vrgout01 = _ve_vshf_vvvs(vrgout0, vrgout1, VE_VSHUFFLE_YUZU) ;

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

    _ve_lvl(VLEN) ;

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

      _ve_lvl(vl) ;

      for (int64_t n=0; n<batch; n++) {
	const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;

	const int64_t gOutIndex0  = outGroupOffset + ((n * gOutChannel + k  ) * gOutHeight ) * gOutWidth ;
	const int64_t gOutIndex1  = outGroupOffset + ((n * gOutChannel + k+1) * gOutHeight ) * gOutWidth ;

	__vr vrpin_c0 = _ve_vsfa_vvss(vrhw, 2, (uint64_t)pInChannel) ;
	__vr vrin_c0 = _ve_vgtu_vv(vrpin_c0) ;
	__vr vrpin_c1 = _ve_vaddul_vsv(1*4*inHeight*inWidth, vrpin_c0) ;
	__vr vrin_c1 = _ve_vgtu_vv(vrpin_c1) ;
	__vr vrpin_c2 = _ve_vaddul_vsv(2*4*inHeight*inWidth, vrpin_c0) ;
	__vr vrin_c2 = _ve_vgtu_vv(vrpin_c2) ;
	__vr vrpin_c3 = _ve_vaddul_vsv(3*4*inHeight*inWidth, vrpin_c0) ;
	__vr vrin_c3 = _ve_vgtu_vv(vrpin_c3) ;
	__vr vrpin_c4 = _ve_vaddul_vsv(4*4*inHeight*inWidth, vrpin_c0) ;
	__vr vrin_c4 = _ve_vgtu_vv(vrpin_c4) ;
	__vr vrpin_c5 = _ve_vaddul_vsv(5*4*inHeight*inWidth, vrpin_c0) ;
	__vr vrin_c5 = _ve_vgtu_vv(vrpin_c5) ;
	__vr vrpin_c6 = _ve_vaddul_vsv(6*4*inHeight*inWidth, vrpin_c0) ;
	__vr vrin_c6 = _ve_vgtu_vv(vrpin_c6) ;
	__vr vrpin_c7 = _ve_vaddul_vsv(7*4*inHeight*inWidth, vrpin_c0) ;
	__vr vrin_c7 = _ve_vgtu_vv(vrpin_c7) ;

	__vr vrgout0 = _ve_vldu_vss(4, pGOut+gOutIndex0) ;
	__vr vrgout1 = _ve_vldu_vss(4, pGOut+gOutIndex1) ;

	__vr vrgout01 = _ve_vshf_vvvs(vrgout0, vrgout1, VE_VSHUFFLE_YUZU) ;

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

    _ve_lvl(VLEN) ;

#define INIT_VRSUM(CTOKEN) \
    __vr vrsum0123_##CTOKEN = _ve_vbrd_vs_i64(0UL) ;

    INIT_VRSUM(c0)

    {
      const int64_t vl = gOutWidth * gOutHeight ;

      _ve_lvl(2*vl) ;

      for (int64_t n=0; n<batch; n++) {
	const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;

	const int64_t gOutIndex01  = outGroupOffset + ((n * gOutChannel + k  ) * gOutHeight ) * gOutWidth ;
	const int64_t gOutIndex23  = outGroupOffset + ((n * gOutChannel + k+2) * gOutHeight ) * gOutWidth ;

	__vr vrpin_c0 = _ve_vsfa_vvss(vrhw, 2, (uint64_t)pInChannel) ;
	__vr vrin_c0 = _ve_vgtu_vv(vrpin_c0) ;

	__vr vrgout01 = _ve_vldu_vss(4, pGOut+gOutIndex01) ;
	__vr vrgout23 = _ve_vldu_vss(4, pGOut+gOutIndex23) ;

	__vr vrgout0123 = _ve_vshf_vvvs(vrgout01, vrgout23, VE_VSHUFFLE_YUZU) ;

#define VFMAD_R1S1(CTOKEN) \
          __vr vrinP_##CTOKEN = _ve_vshf_vvvs(vrin_##CTOKEN, vrin_##CTOKEN, VE_VSHUFFLE_YUZU) ;  \
          vrsum0123_##CTOKEN = _ve_pvfmad_vvvv(vrsum0123_##CTOKEN, vrinP_##CTOKEN, vrgout0123) ;

	  VFMAD_R1S1(c0)

      } // batch
    } // gOutHeight

#define VFSUM_STORE_R1S1(C,CTOKEN) \
    _ve_lvl(VLEN) ; \
    __vr vrsum0_##CTOKEN = _ve_vfsums_vvm(vrsum0123_##CTOKEN, vm_k0) ; \
    __vr vrsum1_##CTOKEN = _ve_vfsums_vvm(vrsum0123_##CTOKEN, vm_k1) ; \
    __vr vrsum2_##CTOKEN = _ve_vfsums_vvm(_ve_vsll_vvs(vrsum0123_##CTOKEN,32), vm_k0); \
    __vr vrsum3_##CTOKEN = _ve_vfsums_vvm(_ve_vsll_vvs(vrsum0123_##CTOKEN,32), vm_k1); \
    _ve_lvl(1) ; \
    _ve_vstu_vss(vrsum0_##CTOKEN, 4, pGKernel+kernelIndex0+(C)) ; \
    _ve_vstu_vss(vrsum1_##CTOKEN, 4, pGKernel+kernelIndex1+(C)) ; \
    _ve_vstu_vss(vrsum2_##CTOKEN, 4, pGKernel+kernelIndex2+(C)) ; \
    _ve_vstu_vss(vrsum3_##CTOKEN, 4, pGKernel+kernelIndex3+(C)) ;

    VFSUM_STORE_R1S1(0,c0)

    c+=1;
  }
  if ( ((inChannelGroup >> 1) & 0x01) == 1 ) {
    const int64_t kernelIndex0 = kernGroupOffset + k     * inChannelGroup + c ;
    const int64_t kernelIndex1 = kernGroupOffset + (k+1) * inChannelGroup + c ;
    const int64_t kernelIndex2 = kernGroupOffset + (k+2) * inChannelGroup + c ;
    const int64_t kernelIndex3 = kernGroupOffset + (k+3) * inChannelGroup + c ;

    _ve_lvl(VLEN) ;

    INIT_VRSUM(c0)
    INIT_VRSUM(c1)

    {
      const int64_t vl = gOutWidth * gOutHeight ;

      _ve_lvl(2*vl) ;

      for (int64_t n=0; n<batch; n++) {
	const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;

	const int64_t gOutIndex01  = outGroupOffset + ((n * gOutChannel + k  ) * gOutHeight ) * gOutWidth ;
	const int64_t gOutIndex23  = outGroupOffset + ((n * gOutChannel + k+2) * gOutHeight ) * gOutWidth ;

	__vr vrpin_c0 = _ve_vsfa_vvss(vrhw, 2, (uint64_t)pInChannel) ;
	__vr vrin_c0 = _ve_vgtu_vv(vrpin_c0) ;
	__vr vrpin_c1 = _ve_vaddul_vsv(1*4*inHeight*inWidth, vrpin_c0) ;
	__vr vrin_c1 = _ve_vgtu_vv(vrpin_c1) ;

	__vr vrgout01 = _ve_vldu_vss(4, pGOut+gOutIndex01) ;
	__vr vrgout23 = _ve_vldu_vss(4, pGOut+gOutIndex23) ;

	__vr vrgout0123 = _ve_vshf_vvvs(vrgout01, vrgout23, VE_VSHUFFLE_YUZU) ;


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

    _ve_lvl(VLEN) ;

    INIT_VRSUM(c0)
    INIT_VRSUM(c1)
    INIT_VRSUM(c2)
    INIT_VRSUM(c3)

    {
      const int64_t vl = gOutWidth * gOutHeight ;

      _ve_lvl(2*vl) ;

      for (int64_t n=0; n<batch; n++) {
	const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;

	const int64_t gOutIndex01  = outGroupOffset + ((n * gOutChannel + k  ) * gOutHeight ) * gOutWidth ;
	const int64_t gOutIndex23  = outGroupOffset + ((n * gOutChannel + k+2) * gOutHeight ) * gOutWidth ;

	__vr vrpin_c0 = _ve_vsfa_vvss(vrhw, 2, (uint64_t)pInChannel) ;
	__vr vrin_c0 = _ve_vgtu_vv(vrpin_c0) ;
	__vr vrpin_c1 = _ve_vaddul_vsv(1*4*inHeight*inWidth, vrpin_c0) ;
	__vr vrin_c1 = _ve_vgtu_vv(vrpin_c1) ;
	__vr vrpin_c2 = _ve_vaddul_vsv(2*4*inHeight*inWidth, vrpin_c0) ;
	__vr vrin_c2 = _ve_vgtu_vv(vrpin_c2) ;
	__vr vrpin_c3 = _ve_vaddul_vsv(3*4*inHeight*inWidth, vrpin_c0) ;
	__vr vrin_c3 = _ve_vgtu_vv(vrpin_c3) ;

	__vr vrgout01 = _ve_vldu_vss(4, pGOut+gOutIndex01) ;
	__vr vrgout23 = _ve_vldu_vss(4, pGOut+gOutIndex23) ;

	__vr vrgout0123 = _ve_vshf_vvvs(vrgout01, vrgout23, VE_VSHUFFLE_YUZU) ;

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

    _ve_lvl(VLEN) ;

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

      _ve_lvl(2*vl) ;

      for (int64_t n=0; n<batch; n++) {
	const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;

	const int64_t gOutIndex01  = outGroupOffset + ((n * gOutChannel + k  ) * gOutHeight ) * gOutWidth ;
	const int64_t gOutIndex23  = outGroupOffset + ((n * gOutChannel + k+2) * gOutHeight ) * gOutWidth ;

	__vr vrpin_c0 = _ve_vsfa_vvss(vrhw, 2, (uint64_t)pInChannel) ;
	__vr vrin_c0 = _ve_vgtu_vv(vrpin_c0) ;
	__vr vrpin_c1 = _ve_vaddul_vsv(1*4*inHeight*inWidth, vrpin_c0) ;
	__vr vrin_c1 = _ve_vgtu_vv(vrpin_c1) ;
	__vr vrpin_c2 = _ve_vaddul_vsv(2*4*inHeight*inWidth, vrpin_c0) ;
	__vr vrin_c2 = _ve_vgtu_vv(vrpin_c2) ;
	__vr vrpin_c3 = _ve_vaddul_vsv(3*4*inHeight*inWidth, vrpin_c0) ;
	__vr vrin_c3 = _ve_vgtu_vv(vrpin_c3) ;
	__vr vrpin_c4 = _ve_vaddul_vsv(4*4*inHeight*inWidth, vrpin_c0) ;
	__vr vrin_c4 = _ve_vgtu_vv(vrpin_c4) ;
	__vr vrpin_c5 = _ve_vaddul_vsv(5*4*inHeight*inWidth, vrpin_c0) ;
	__vr vrin_c5 = _ve_vgtu_vv(vrpin_c5) ;
	__vr vrpin_c6 = _ve_vaddul_vsv(6*4*inHeight*inWidth, vrpin_c0) ;
	__vr vrin_c6 = _ve_vgtu_vv(vrpin_c6) ;
	__vr vrpin_c7 = _ve_vaddul_vsv(7*4*inHeight*inWidth, vrpin_c0) ;
	__vr vrin_c7 = _ve_vgtu_vv(vrpin_c7) ;

	__vr vrgout01 = _ve_vldu_vss(4, pGOut+gOutIndex01) ;
	__vr vrgout23 = _ve_vldu_vss(4, pGOut+gOutIndex23) ;

	__vr vrgout0123 = _ve_vshf_vvvs(vrgout01, vrgout23, VE_VSHUFFLE_YUZU) ;

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

    _ve_lvl(VLEN) ;

#define INIT_VRSUM(CTOKEN) \
    __vr vrsum0123_##CTOKEN = _ve_vbrd_vs_i64(0UL) ; \
    __vr vrsum4567_##CTOKEN = _ve_vbrd_vs_i64(0UL) ;

    INIT_VRSUM(c0)

    {
      const int64_t vl = gOutWidth * gOutHeight ;

      _ve_lvl(2*vl) ;

      for (int64_t n=0; n<batch; n++) {
	const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;

	const int64_t gOutIndex01  = outGroupOffset + ((n * gOutChannel + k  ) * gOutHeight ) * gOutWidth ;
	const int64_t gOutIndex23  = outGroupOffset + ((n * gOutChannel + k+2) * gOutHeight ) * gOutWidth ;
	const int64_t gOutIndex45  = outGroupOffset + ((n * gOutChannel + k+4) * gOutHeight ) * gOutWidth ;
	const int64_t gOutIndex67  = outGroupOffset + ((n * gOutChannel + k+6) * gOutHeight ) * gOutWidth ;

	__vr vrpin_c0 = _ve_vsfa_vvss(vrhw, 2, (uint64_t)pInChannel) ;
	__vr vrin_c0 = _ve_vgtu_vv(vrpin_c0) ;

	__vr vrgout01 = _ve_vldu_vss(4, pGOut+gOutIndex01) ;
	__vr vrgout23 = _ve_vldu_vss(4, pGOut+gOutIndex23) ;
	__vr vrgout45 = _ve_vldu_vss(4, pGOut+gOutIndex45) ;
	__vr vrgout67 = _ve_vldu_vss(4, pGOut+gOutIndex67) ;

	__vr vrgout0123 = _ve_vshf_vvvs(vrgout01, vrgout23, VE_VSHUFFLE_YUZU) ;
	__vr vrgout4567 = _ve_vshf_vvvs(vrgout45, vrgout67, VE_VSHUFFLE_YUZU) ;

#define VFMAD_R1S1(CTOKEN) \
          __vr vrinP_##CTOKEN = _ve_vshf_vvvs(vrin_##CTOKEN, vrin_##CTOKEN, VE_VSHUFFLE_YUZU) ;  \
          vrsum0123_##CTOKEN = _ve_pvfmad_vvvv(vrsum0123_##CTOKEN, vrinP_##CTOKEN, vrgout0123) ; \
          vrsum4567_##CTOKEN = _ve_pvfmad_vvvv(vrsum4567_##CTOKEN, vrinP_##CTOKEN, vrgout4567) ;

	  VFMAD_R1S1(c0)

      } // batch
    } // gOutHeight

#define VFSUM_STORE_R1S1(C,CTOKEN) \
    _ve_lvl(VLEN) ; \
    __vr vrsum0_##CTOKEN = _ve_vfsums_vvm(vrsum0123_##CTOKEN, vm_k0) ; \
    __vr vrsum1_##CTOKEN = _ve_vfsums_vvm(vrsum0123_##CTOKEN, vm_k1) ; \
    __vr vrsum2_##CTOKEN = _ve_vfsums_vvm(_ve_vsll_vvs(vrsum0123_##CTOKEN,32), vm_k0); \
    __vr vrsum3_##CTOKEN = _ve_vfsums_vvm(_ve_vsll_vvs(vrsum0123_##CTOKEN,32), vm_k1); \
    __vr vrsum4_##CTOKEN = _ve_vfsums_vvm(vrsum4567_##CTOKEN, vm_k0) ; \
    __vr vrsum5_##CTOKEN = _ve_vfsums_vvm(vrsum4567_##CTOKEN, vm_k1) ; \
    __vr vrsum6_##CTOKEN = _ve_vfsums_vvm(_ve_vsll_vvs(vrsum4567_##CTOKEN,32), vm_k0); \
    __vr vrsum7_##CTOKEN = _ve_vfsums_vvm(_ve_vsll_vvs(vrsum4567_##CTOKEN,32), vm_k1); \
    _ve_lvl(1) ; \
    _ve_vstu_vss(vrsum0_##CTOKEN, 4, pGKernel+kernelIndex0+(C)) ; \
    _ve_vstu_vss(vrsum1_##CTOKEN, 4, pGKernel+kernelIndex1+(C)) ; \
    _ve_vstu_vss(vrsum2_##CTOKEN, 4, pGKernel+kernelIndex2+(C)) ; \
    _ve_vstu_vss(vrsum3_##CTOKEN, 4, pGKernel+kernelIndex3+(C)) ; \
    _ve_vstu_vss(vrsum4_##CTOKEN, 4, pGKernel+kernelIndex4+(C)) ; \
    _ve_vstu_vss(vrsum5_##CTOKEN, 4, pGKernel+kernelIndex5+(C)) ; \
    _ve_vstu_vss(vrsum6_##CTOKEN, 4, pGKernel+kernelIndex6+(C)) ; \
    _ve_vstu_vss(vrsum7_##CTOKEN, 4, pGKernel+kernelIndex7+(C)) ;

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

    _ve_lvl(VLEN) ;

    INIT_VRSUM(c0)
    INIT_VRSUM(c1)

    {
      const int64_t vl = gOutWidth * gOutHeight ;

      _ve_lvl(2*vl) ;

      for (int64_t n=0; n<batch; n++) {
	const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;

	const int64_t gOutIndex01  = outGroupOffset + ((n * gOutChannel + k  ) * gOutHeight ) * gOutWidth ;
	const int64_t gOutIndex23  = outGroupOffset + ((n * gOutChannel + k+2) * gOutHeight ) * gOutWidth ;
	const int64_t gOutIndex45  = outGroupOffset + ((n * gOutChannel + k+4) * gOutHeight ) * gOutWidth ;
	const int64_t gOutIndex67  = outGroupOffset + ((n * gOutChannel + k+6) * gOutHeight ) * gOutWidth ;

	__vr vrpin_c0 = _ve_vsfa_vvss(vrhw, 2, (uint64_t)pInChannel) ;
	__vr vrin_c0 = _ve_vgtu_vv(vrpin_c0) ;
	__vr vrpin_c1 = _ve_vaddul_vsv(1*4*inHeight*inWidth, vrpin_c0) ;
	__vr vrin_c1 = _ve_vgtu_vv(vrpin_c1) ;

	__vr vrgout01 = _ve_vldu_vss(4, pGOut+gOutIndex01) ;
	__vr vrgout23 = _ve_vldu_vss(4, pGOut+gOutIndex23) ;
	__vr vrgout45 = _ve_vldu_vss(4, pGOut+gOutIndex45) ;
	__vr vrgout67 = _ve_vldu_vss(4, pGOut+gOutIndex67) ;

	__vr vrgout0123 = _ve_vshf_vvvs(vrgout01, vrgout23, VE_VSHUFFLE_YUZU) ;
	__vr vrgout4567 = _ve_vshf_vvvs(vrgout45, vrgout67, VE_VSHUFFLE_YUZU) ;


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

    _ve_lvl(VLEN) ;

    INIT_VRSUM(c0)
    INIT_VRSUM(c1)
    INIT_VRSUM(c2)
    INIT_VRSUM(c3)

    {
      const int64_t vl = gOutWidth * gOutHeight ;

      _ve_lvl(2*vl) ;

      for (int64_t n=0; n<batch; n++) {
	const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;

	const int64_t gOutIndex01  = outGroupOffset + ((n * gOutChannel + k  ) * gOutHeight ) * gOutWidth ;
	const int64_t gOutIndex23  = outGroupOffset + ((n * gOutChannel + k+2) * gOutHeight ) * gOutWidth ;
	const int64_t gOutIndex45  = outGroupOffset + ((n * gOutChannel + k+4) * gOutHeight ) * gOutWidth ;
	const int64_t gOutIndex67  = outGroupOffset + ((n * gOutChannel + k+6) * gOutHeight ) * gOutWidth ;

	__vr vrpin_c0 = _ve_vsfa_vvss(vrhw, 2, (uint64_t)pInChannel) ;
	__vr vrin_c0 = _ve_vgtu_vv(vrpin_c0) ;
	__vr vrpin_c1 = _ve_vaddul_vsv(1*4*inHeight*inWidth, vrpin_c0) ;
	__vr vrin_c1 = _ve_vgtu_vv(vrpin_c1) ;
	__vr vrpin_c2 = _ve_vaddul_vsv(2*4*inHeight*inWidth, vrpin_c0) ;
	__vr vrin_c2 = _ve_vgtu_vv(vrpin_c2) ;
	__vr vrpin_c3 = _ve_vaddul_vsv(3*4*inHeight*inWidth, vrpin_c0) ;
	__vr vrin_c3 = _ve_vgtu_vv(vrpin_c3) ;

	__vr vrgout01 = _ve_vldu_vss(4, pGOut+gOutIndex01) ;
	__vr vrgout23 = _ve_vldu_vss(4, pGOut+gOutIndex23) ;
	__vr vrgout45 = _ve_vldu_vss(4, pGOut+gOutIndex45) ;
	__vr vrgout67 = _ve_vldu_vss(4, pGOut+gOutIndex67) ;

	__vr vrgout0123 = _ve_vshf_vvvs(vrgout01, vrgout23, VE_VSHUFFLE_YUZU) ;
	__vr vrgout4567 = _ve_vshf_vvvs(vrgout45, vrgout67, VE_VSHUFFLE_YUZU) ;

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

    _ve_lvl(VLEN) ;

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

      _ve_lvl(2*vl) ;

      for (int64_t n=0; n<batch; n++) {
	const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;

	const int64_t gOutIndex01  = outGroupOffset + ((n * gOutChannel + k  ) * gOutHeight ) * gOutWidth ;
	const int64_t gOutIndex23  = outGroupOffset + ((n * gOutChannel + k+2) * gOutHeight ) * gOutWidth ;
	const int64_t gOutIndex45  = outGroupOffset + ((n * gOutChannel + k+4) * gOutHeight ) * gOutWidth ;
	const int64_t gOutIndex67  = outGroupOffset + ((n * gOutChannel + k+6) * gOutHeight ) * gOutWidth ;

	__vr vrpin_c0 = _ve_vsfa_vvss(vrhw, 2, (uint64_t)pInChannel) ;
	__vr vrin_c0 = _ve_vgtu_vv(vrpin_c0) ;
	__vr vrpin_c1 = _ve_vaddul_vsv(1*4*inHeight*inWidth, vrpin_c0) ;
	__vr vrin_c1 = _ve_vgtu_vv(vrpin_c1) ;
	__vr vrpin_c2 = _ve_vaddul_vsv(2*4*inHeight*inWidth, vrpin_c0) ;
	__vr vrin_c2 = _ve_vgtu_vv(vrpin_c2) ;
	__vr vrpin_c3 = _ve_vaddul_vsv(3*4*inHeight*inWidth, vrpin_c0) ;
	__vr vrin_c3 = _ve_vgtu_vv(vrpin_c3) ;
	__vr vrpin_c4 = _ve_vaddul_vsv(4*4*inHeight*inWidth, vrpin_c0) ;
	__vr vrin_c4 = _ve_vgtu_vv(vrpin_c4) ;
	__vr vrpin_c5 = _ve_vaddul_vsv(5*4*inHeight*inWidth, vrpin_c0) ;
	__vr vrin_c5 = _ve_vgtu_vv(vrpin_c5) ;
	__vr vrpin_c6 = _ve_vaddul_vsv(6*4*inHeight*inWidth, vrpin_c0) ;
	__vr vrin_c6 = _ve_vgtu_vv(vrpin_c6) ;
	__vr vrpin_c7 = _ve_vaddul_vsv(7*4*inHeight*inWidth, vrpin_c0) ;
	__vr vrin_c7 = _ve_vgtu_vv(vrpin_c7) ;

	__vr vrgout01 = _ve_vldu_vss(4, pGOut+gOutIndex01) ;
	__vr vrgout23 = _ve_vldu_vss(4, pGOut+gOutIndex23) ;
	__vr vrgout45 = _ve_vldu_vss(4, pGOut+gOutIndex45) ;
	__vr vrgout67 = _ve_vldu_vss(4, pGOut+gOutIndex67) ;

	__vr vrgout0123 = _ve_vshf_vvvs(vrgout01, vrgout23, VE_VSHUFFLE_YUZU) ;
	__vr vrgout4567 = _ve_vshf_vvvs(vrgout45, vrgout67, VE_VSHUFFLE_YUZU) ;

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

    _ve_lvl(VLEN) ;

#define INIT_VRSUM(CTOKEN) \
    __vr vrsum01234567_##CTOKEN = _ve_vbrd_vs_i64(0UL) ; \
    __vr vrsum89ABCDEF_##CTOKEN = _ve_vbrd_vs_i64(0UL) ;

    INIT_VRSUM(c0)

    {
      const int64_t vl = gOutWidth * gOutHeight ;

      _ve_lvl(4*vl) ;

      for (int64_t n=0; n<batch; n++) {
	const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;

	const int64_t gOutIndex  = outGroupOffset + ((n * gOutChannel + k  ) * gOutHeight ) * gOutWidth ;

	__vr vrpin_c0 = _ve_vsfa_vvss(vrhw, 2, (uint64_t)pInChannel) ;
	__vr vrin_c0 = _ve_vgtu_vv(vrpin_c0) ;

	__vr vrgout0123 = _ve_vldu_vss(4, pGOut+gOutIndex ) ;
	__vr vrgout4567 = _ve_vldu_vss(4, pGOut+gOutIndex+ 4*gOutHeight*gOutWidth) ;
	__vr vrgout89AB = _ve_vldu_vss(4, pGOut+gOutIndex+ 8*gOutHeight*gOutWidth) ;
	__vr vrgoutCDEF = _ve_vldu_vss(4, pGOut+gOutIndex+12*gOutHeight*gOutWidth) ;

	__vr vrgout01234567 = _ve_vshf_vvvs(vrgout0123, vrgout4567, VE_VSHUFFLE_YUZU) ;
	__vr vrgout89ABCDEF = _ve_vshf_vvvs(vrgout89AB, vrgoutCDEF, VE_VSHUFFLE_YUZU) ;

#define VFMAD_R1S1(CTOKEN) \
          __vr vrinP_##CTOKEN = _ve_vshf_vvvs(vrin_##CTOKEN, vrin_##CTOKEN, VE_VSHUFFLE_YUZU) ;  \
          vrsum01234567_##CTOKEN = _ve_pvfmad_vvvv(vrsum01234567_##CTOKEN, vrinP_##CTOKEN, vrgout01234567) ; \
          vrsum89ABCDEF_##CTOKEN = _ve_pvfmad_vvvv(vrsum89ABCDEF_##CTOKEN, vrinP_##CTOKEN, vrgout89ABCDEF) ;

	  VFMAD_R1S1(c0)

      } // batch
    } // gOutHeight

#define VFSUM_STORE_R1S1(C,CTOKEN) \
    _ve_lvl(VLEN) ; \
    __vr vrsum0_##CTOKEN = _ve_vfsums_vvm(vrsum01234567_##CTOKEN, vm_k0) ; \
    __vr vrsum1_##CTOKEN = _ve_vfsums_vvm(vrsum01234567_##CTOKEN, vm_k1) ; \
    __vr vrsum2_##CTOKEN = _ve_vfsums_vvm(vrsum01234567_##CTOKEN, vm_k2) ; \
    __vr vrsum3_##CTOKEN = _ve_vfsums_vvm(vrsum01234567_##CTOKEN, vm_k3) ; \
    __vr vrsum4_##CTOKEN = _ve_vfsums_vvm(_ve_vsll_vvs(vrsum01234567_##CTOKEN,32), vm_k0); \
    __vr vrsum5_##CTOKEN = _ve_vfsums_vvm(_ve_vsll_vvs(vrsum01234567_##CTOKEN,32), vm_k1); \
    __vr vrsum6_##CTOKEN = _ve_vfsums_vvm(_ve_vsll_vvs(vrsum01234567_##CTOKEN,32), vm_k2); \
    __vr vrsum7_##CTOKEN = _ve_vfsums_vvm(_ve_vsll_vvs(vrsum01234567_##CTOKEN,32), vm_k3); \
    _ve_lvl(1) ; \
    _ve_vstu_vss(vrsum0_##CTOKEN, 4, pGKernel+kernelIndex+0*inChannelGroup+(C)) ; \
    _ve_vstu_vss(vrsum1_##CTOKEN, 4, pGKernel+kernelIndex+1*inChannelGroup+(C)) ; \
    _ve_vstu_vss(vrsum2_##CTOKEN, 4, pGKernel+kernelIndex+2*inChannelGroup+(C)) ; \
    _ve_vstu_vss(vrsum3_##CTOKEN, 4, pGKernel+kernelIndex+3*inChannelGroup+(C)) ; \
    _ve_vstu_vss(vrsum4_##CTOKEN, 4, pGKernel+kernelIndex+4*inChannelGroup+(C)) ; \
    _ve_vstu_vss(vrsum5_##CTOKEN, 4, pGKernel+kernelIndex+5*inChannelGroup+(C)) ; \
    _ve_vstu_vss(vrsum6_##CTOKEN, 4, pGKernel+kernelIndex+6*inChannelGroup+(C)) ; \
    _ve_vstu_vss(vrsum7_##CTOKEN, 4, pGKernel+kernelIndex+7*inChannelGroup+(C)) ; \
    _ve_lvl(VLEN) ; \
    __vr vrsum8_##CTOKEN = _ve_vfsums_vvm(vrsum89ABCDEF_##CTOKEN, vm_k0) ; \
    __vr vrsum9_##CTOKEN = _ve_vfsums_vvm(vrsum89ABCDEF_##CTOKEN, vm_k1) ; \
    __vr vrsumA_##CTOKEN = _ve_vfsums_vvm(vrsum89ABCDEF_##CTOKEN, vm_k2) ; \
    __vr vrsumB_##CTOKEN = _ve_vfsums_vvm(vrsum89ABCDEF_##CTOKEN, vm_k3) ; \
    __vr vrsumC_##CTOKEN = _ve_vfsums_vvm(_ve_vsll_vvs(vrsum89ABCDEF_##CTOKEN,32), vm_k0); \
    __vr vrsumD_##CTOKEN = _ve_vfsums_vvm(_ve_vsll_vvs(vrsum89ABCDEF_##CTOKEN,32), vm_k1); \
    __vr vrsumE_##CTOKEN = _ve_vfsums_vvm(_ve_vsll_vvs(vrsum89ABCDEF_##CTOKEN,32), vm_k2); \
    __vr vrsumF_##CTOKEN = _ve_vfsums_vvm(_ve_vsll_vvs(vrsum89ABCDEF_##CTOKEN,32), vm_k3); \
    _ve_lvl(1) ; \
    _ve_vstu_vss(vrsum8_##CTOKEN, 4, pGKernel+kernelIndex+8*inChannelGroup+(C)) ; \
    _ve_vstu_vss(vrsum9_##CTOKEN, 4, pGKernel+kernelIndex+9*inChannelGroup+(C)) ; \
    _ve_vstu_vss(vrsumA_##CTOKEN, 4, pGKernel+kernelIndex+10*inChannelGroup+(C)) ; \
    _ve_vstu_vss(vrsumB_##CTOKEN, 4, pGKernel+kernelIndex+11*inChannelGroup+(C)) ; \
    _ve_vstu_vss(vrsumC_##CTOKEN, 4, pGKernel+kernelIndex+12*inChannelGroup+(C)) ; \
    _ve_vstu_vss(vrsumD_##CTOKEN, 4, pGKernel+kernelIndex+13*inChannelGroup+(C)) ; \
    _ve_vstu_vss(vrsumE_##CTOKEN, 4, pGKernel+kernelIndex+14*inChannelGroup+(C)) ; \
    _ve_vstu_vss(vrsumF_##CTOKEN, 4, pGKernel+kernelIndex+15*inChannelGroup+(C)) ;

    VFSUM_STORE_R1S1(0,c0)

    c+=1;
  }
  if ( ((inChannelGroup >> 1) & 0x01) == 1 ) {
    const int64_t kernelIndex = kernGroupOffset + k     * inChannelGroup + c ;

    _ve_lvl(VLEN) ;

    INIT_VRSUM(c0)
    INIT_VRSUM(c1)

    {
      const int64_t vl = gOutWidth * gOutHeight ;

      _ve_lvl(4*vl) ;

      for (int64_t n=0; n<batch; n++) {
	const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;

	const int64_t gOutIndex  = outGroupOffset + ((n * gOutChannel + k  ) * gOutHeight ) * gOutWidth ;

	__vr vrpin_c0 = _ve_vsfa_vvss(vrhw, 2, (uint64_t)pInChannel) ;
	__vr vrin_c0 = _ve_vgtu_vv(vrpin_c0) ;
	__vr vrpin_c1 = _ve_vaddul_vsv(1*4*inHeight*inWidth, vrpin_c0) ;
	__vr vrin_c1 = _ve_vgtu_vv(vrpin_c1) ;

	__vr vrgout0123 = _ve_vldu_vss(4, pGOut+gOutIndex) ;
	__vr vrgout4567 = _ve_vldu_vss(4, pGOut+gOutIndex+ 4*gOutHeight*gOutWidth) ;
	__vr vrgout89AB = _ve_vldu_vss(4, pGOut+gOutIndex+ 8*gOutHeight*gOutWidth) ;
	__vr vrgoutCDEF = _ve_vldu_vss(4, pGOut+gOutIndex+12*gOutHeight*gOutWidth) ;

	__vr vrgout01234567 = _ve_vshf_vvvs(vrgout0123, vrgout4567, VE_VSHUFFLE_YUZU) ;
	__vr vrgout89ABCDEF = _ve_vshf_vvvs(vrgout89AB, vrgoutCDEF, VE_VSHUFFLE_YUZU) ;

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

    _ve_lvl(VLEN) ;

    INIT_VRSUM(c0)
    INIT_VRSUM(c1)
    INIT_VRSUM(c2)
    INIT_VRSUM(c3)

    {
      const int64_t vl = gOutWidth * gOutHeight ;

      _ve_lvl(4*vl) ;

      for (int64_t n=0; n<batch; n++) {
	const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;

	const int64_t gOutIndex  = outGroupOffset + ((n * gOutChannel + k  ) * gOutHeight ) * gOutWidth ;

	__vr vrpin_c0 = _ve_vsfa_vvss(vrhw, 2, (uint64_t)pInChannel) ;
	__vr vrin_c0 = _ve_vgtu_vv(vrpin_c0) ;
	__vr vrpin_c1 = _ve_vaddul_vsv(1*4*inHeight*inWidth, vrpin_c0) ;
	__vr vrin_c1 = _ve_vgtu_vv(vrpin_c1) ;
	__vr vrpin_c2 = _ve_vaddul_vsv(2*4*inHeight*inWidth, vrpin_c0) ;
	__vr vrin_c2 = _ve_vgtu_vv(vrpin_c2) ;
	__vr vrpin_c3 = _ve_vaddul_vsv(3*4*inHeight*inWidth, vrpin_c0) ;
	__vr vrin_c3 = _ve_vgtu_vv(vrpin_c3) ;

	__vr vrgout0123 = _ve_vldu_vss(4, pGOut+gOutIndex) ;
	__vr vrgout4567 = _ve_vldu_vss(4, pGOut+gOutIndex+ 4*gOutHeight*gOutWidth) ;
	__vr vrgout89AB = _ve_vldu_vss(4, pGOut+gOutIndex+ 8*gOutHeight*gOutWidth) ;
	__vr vrgoutCDEF = _ve_vldu_vss(4, pGOut+gOutIndex+12*gOutHeight*gOutWidth) ;

	__vr vrgout01234567 = _ve_vshf_vvvs(vrgout0123, vrgout4567, VE_VSHUFFLE_YUZU) ;
	__vr vrgout89ABCDEF = _ve_vshf_vvvs(vrgout89AB, vrgoutCDEF, VE_VSHUFFLE_YUZU) ;

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

    _ve_lvl(VLEN) ;

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

      _ve_lvl(4*vl) ;

      for (int64_t n=0; n<batch; n++) {
	const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;

	const int64_t gOutIndex  = outGroupOffset + ((n * gOutChannel + k  ) * gOutHeight ) * gOutWidth ;

	__vr vrpin_c0 = _ve_vsfa_vvss(vrhw, 2, (uint64_t)pInChannel) ;
	__vr vrin_c0 = _ve_vgtu_vv(vrpin_c0) ;
	__vr vrpin_c1 = _ve_vaddul_vsv(1*4*inHeight*inWidth, vrpin_c0) ;
	__vr vrin_c1 = _ve_vgtu_vv(vrpin_c1) ;
	__vr vrpin_c2 = _ve_vaddul_vsv(2*4*inHeight*inWidth, vrpin_c0) ;
	__vr vrin_c2 = _ve_vgtu_vv(vrpin_c2) ;
	__vr vrpin_c3 = _ve_vaddul_vsv(3*4*inHeight*inWidth, vrpin_c0) ;
	__vr vrin_c3 = _ve_vgtu_vv(vrpin_c3) ;
	__vr vrpin_c4 = _ve_vaddul_vsv(4*4*inHeight*inWidth, vrpin_c0) ;
	__vr vrin_c4 = _ve_vgtu_vv(vrpin_c4) ;
	__vr vrpin_c5 = _ve_vaddul_vsv(5*4*inHeight*inWidth, vrpin_c0) ;
	__vr vrin_c5 = _ve_vgtu_vv(vrpin_c5) ;
	__vr vrpin_c6 = _ve_vaddul_vsv(6*4*inHeight*inWidth, vrpin_c0) ;
	__vr vrin_c6 = _ve_vgtu_vv(vrpin_c6) ;
	__vr vrpin_c7 = _ve_vaddul_vsv(7*4*inHeight*inWidth, vrpin_c0) ;
	__vr vrin_c7 = _ve_vgtu_vv(vrpin_c7) ;

	__vr vrgout0123 = _ve_vldu_vss(4, pGOut+gOutIndex) ;
	__vr vrgout4567 = _ve_vldu_vss(4, pGOut+gOutIndex+ 4*gOutHeight*gOutWidth) ;
	__vr vrgout89AB = _ve_vldu_vss(4, pGOut+gOutIndex+ 8*gOutHeight*gOutWidth) ;
	__vr vrgoutCDEF = _ve_vldu_vss(4, pGOut+gOutIndex+12*gOutHeight*gOutWidth) ;

	__vr vrgout01234567 = _ve_vshf_vvvs(vrgout0123, vrgout4567, VE_VSHUFFLE_YUZU) ;
	__vr vrgout89ABCDEF = _ve_vshf_vvvs(vrgout89AB, vrgoutCDEF, VE_VSHUFFLE_YUZU) ;

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

    _ve_lvl(VLEN) ;

#define INIT_VRSUM(CTOKEN) \
    __vr vrsum01234567_##CTOKEN = _ve_vbrd_vs_i64(0UL) ; \
    __vr vrsum89ABCDEF_##CTOKEN = _ve_vbrd_vs_i64(0UL) ; \
    __vr vrsumGHIJKLMN_##CTOKEN = _ve_vbrd_vs_i64(0UL) ; \
    __vr vrsumOPQRSTUV_##CTOKEN = _ve_vbrd_vs_i64(0UL) ;

    INIT_VRSUM(c0)

    {
      const int64_t vl = gOutWidth * gOutHeight ;

      _ve_lvl(4*vl) ;

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

	__vr vrpin_c0 = _ve_vsfa_vvss(vrhw, 2, (uint64_t)pInChannel) ;
	__vr vrin_c0 = _ve_vgtu_vv(vrpin_c0) ;

	__vr vrgout0123 = _ve_vldu_vss(4, pGOut+gOutIndex0123) ;
	__vr vrgout4567 = _ve_vldu_vss(4, pGOut+gOutIndex4567) ;
	__vr vrgout89AB = _ve_vldu_vss(4, pGOut+gOutIndex89AB) ;
	__vr vrgoutCDEF = _ve_vldu_vss(4, pGOut+gOutIndexCDEF) ;
	__vr vrgoutGHIJ = _ve_vldu_vss(4, pGOut+gOutIndexGHIJ) ;
	__vr vrgoutKLMN = _ve_vldu_vss(4, pGOut+gOutIndexKLMN) ;
	__vr vrgoutOPQR = _ve_vldu_vss(4, pGOut+gOutIndexOPQR) ;
	__vr vrgoutSTUV = _ve_vldu_vss(4, pGOut+gOutIndexSTUV) ;

	__vr vrgout01234567 = _ve_vshf_vvvs(vrgout0123, vrgout4567, VE_VSHUFFLE_YUZU) ;
	__vr vrgout89ABCDEF = _ve_vshf_vvvs(vrgout89AB, vrgoutCDEF, VE_VSHUFFLE_YUZU) ;
	__vr vrgoutGHIJKLMN = _ve_vshf_vvvs(vrgoutGHIJ, vrgoutKLMN, VE_VSHUFFLE_YUZU) ;
	__vr vrgoutOPQRSTUV = _ve_vshf_vvvs(vrgoutOPQR, vrgoutSTUV, VE_VSHUFFLE_YUZU) ;

#define VFMAD_R1S1(CTOKEN) \
          __vr vrinP_##CTOKEN = _ve_vshf_vvvs(vrin_##CTOKEN, vrin_##CTOKEN, VE_VSHUFFLE_YUZU) ;  \
          vrsum01234567_##CTOKEN = _ve_pvfmad_vvvv(vrsum01234567_##CTOKEN, vrinP_##CTOKEN, vrgout01234567) ; \
          vrsum89ABCDEF_##CTOKEN = _ve_pvfmad_vvvv(vrsum89ABCDEF_##CTOKEN, vrinP_##CTOKEN, vrgout89ABCDEF) ; \
          vrsumGHIJKLMN_##CTOKEN = _ve_pvfmad_vvvv(vrsumGHIJKLMN_##CTOKEN, vrinP_##CTOKEN, vrgoutGHIJKLMN) ; \
          vrsumOPQRSTUV_##CTOKEN = _ve_pvfmad_vvvv(vrsumOPQRSTUV_##CTOKEN, vrinP_##CTOKEN, vrgoutOPQRSTUV) ;

	  VFMAD_R1S1(c0)

      } // batch
    } // gOutHeight

#define VFSUM_STORE_R1S1(C,CTOKEN) \
    _ve_lvl(VLEN) ; \
    __vr vrsum0_##CTOKEN = _ve_vfsums_vvm(vrsum01234567_##CTOKEN, vm_k0) ; \
    __vr vrsum1_##CTOKEN = _ve_vfsums_vvm(vrsum01234567_##CTOKEN, vm_k1) ; \
    __vr vrsum2_##CTOKEN = _ve_vfsums_vvm(vrsum01234567_##CTOKEN, vm_k2) ; \
    __vr vrsum3_##CTOKEN = _ve_vfsums_vvm(vrsum01234567_##CTOKEN, vm_k3) ; \
    __vr vrsum4_##CTOKEN = _ve_vfsums_vvm(_ve_vsll_vvs(vrsum01234567_##CTOKEN,32), vm_k0); \
    __vr vrsum5_##CTOKEN = _ve_vfsums_vvm(_ve_vsll_vvs(vrsum01234567_##CTOKEN,32), vm_k1); \
    __vr vrsum6_##CTOKEN = _ve_vfsums_vvm(_ve_vsll_vvs(vrsum01234567_##CTOKEN,32), vm_k2); \
    __vr vrsum7_##CTOKEN = _ve_vfsums_vvm(_ve_vsll_vvs(vrsum01234567_##CTOKEN,32), vm_k3); \
    _ve_lvl(1) ; \
    _ve_vstu_vss(vrsum0_##CTOKEN, 4, pGKernel+kernelIndex+0*inChannelGroup+(C)) ; \
    _ve_vstu_vss(vrsum1_##CTOKEN, 4, pGKernel+kernelIndex+1*inChannelGroup+(C)) ; \
    _ve_vstu_vss(vrsum2_##CTOKEN, 4, pGKernel+kernelIndex+2*inChannelGroup+(C)) ; \
    _ve_vstu_vss(vrsum3_##CTOKEN, 4, pGKernel+kernelIndex+3*inChannelGroup+(C)) ; \
    _ve_vstu_vss(vrsum4_##CTOKEN, 4, pGKernel+kernelIndex+4*inChannelGroup+(C)) ; \
    _ve_vstu_vss(vrsum5_##CTOKEN, 4, pGKernel+kernelIndex+5*inChannelGroup+(C)) ; \
    _ve_vstu_vss(vrsum6_##CTOKEN, 4, pGKernel+kernelIndex+6*inChannelGroup+(C)) ; \
    _ve_vstu_vss(vrsum7_##CTOKEN, 4, pGKernel+kernelIndex+7*inChannelGroup+(C)) ; \
    _ve_lvl(VLEN) ; \
    __vr vrsum8_##CTOKEN = _ve_vfsums_vvm(vrsum89ABCDEF_##CTOKEN, vm_k0) ; \
    __vr vrsum9_##CTOKEN = _ve_vfsums_vvm(vrsum89ABCDEF_##CTOKEN, vm_k1) ; \
    __vr vrsumA_##CTOKEN = _ve_vfsums_vvm(vrsum89ABCDEF_##CTOKEN, vm_k2) ; \
    __vr vrsumB_##CTOKEN = _ve_vfsums_vvm(vrsum89ABCDEF_##CTOKEN, vm_k3) ; \
    __vr vrsumC_##CTOKEN = _ve_vfsums_vvm(_ve_vsll_vvs(vrsum89ABCDEF_##CTOKEN,32), vm_k0); \
    __vr vrsumD_##CTOKEN = _ve_vfsums_vvm(_ve_vsll_vvs(vrsum89ABCDEF_##CTOKEN,32), vm_k1); \
    __vr vrsumE_##CTOKEN = _ve_vfsums_vvm(_ve_vsll_vvs(vrsum89ABCDEF_##CTOKEN,32), vm_k2); \
    __vr vrsumF_##CTOKEN = _ve_vfsums_vvm(_ve_vsll_vvs(vrsum89ABCDEF_##CTOKEN,32), vm_k3); \
    _ve_lvl(1) ; \
    _ve_vstu_vss(vrsum8_##CTOKEN, 4, pGKernel+kernelIndex+8*inChannelGroup+(C)) ; \
    _ve_vstu_vss(vrsum9_##CTOKEN, 4, pGKernel+kernelIndex+9*inChannelGroup+(C)) ; \
    _ve_vstu_vss(vrsumA_##CTOKEN, 4, pGKernel+kernelIndex+10*inChannelGroup+(C)) ; \
    _ve_vstu_vss(vrsumB_##CTOKEN, 4, pGKernel+kernelIndex+11*inChannelGroup+(C)) ; \
    _ve_vstu_vss(vrsumC_##CTOKEN, 4, pGKernel+kernelIndex+12*inChannelGroup+(C)) ; \
    _ve_vstu_vss(vrsumD_##CTOKEN, 4, pGKernel+kernelIndex+13*inChannelGroup+(C)) ; \
    _ve_vstu_vss(vrsumE_##CTOKEN, 4, pGKernel+kernelIndex+14*inChannelGroup+(C)) ; \
    _ve_vstu_vss(vrsumF_##CTOKEN, 4, pGKernel+kernelIndex+15*inChannelGroup+(C)) ; \
    _ve_lvl(VLEN) ; \
    __vr vrsumG_##CTOKEN = _ve_vfsums_vvm(vrsumGHIJKLMN_##CTOKEN, vm_k0) ; \
    __vr vrsumH_##CTOKEN = _ve_vfsums_vvm(vrsumGHIJKLMN_##CTOKEN, vm_k1) ; \
    __vr vrsumI_##CTOKEN = _ve_vfsums_vvm(vrsumGHIJKLMN_##CTOKEN, vm_k2) ; \
    __vr vrsumJ_##CTOKEN = _ve_vfsums_vvm(vrsumGHIJKLMN_##CTOKEN, vm_k3) ; \
    __vr vrsumK_##CTOKEN = _ve_vfsums_vvm(_ve_vsll_vvs(vrsumGHIJKLMN_##CTOKEN,32), vm_k0); \
    __vr vrsumL_##CTOKEN = _ve_vfsums_vvm(_ve_vsll_vvs(vrsumGHIJKLMN_##CTOKEN,32), vm_k1); \
    __vr vrsumM_##CTOKEN = _ve_vfsums_vvm(_ve_vsll_vvs(vrsumGHIJKLMN_##CTOKEN,32), vm_k2); \
    __vr vrsumN_##CTOKEN = _ve_vfsums_vvm(_ve_vsll_vvs(vrsumGHIJKLMN_##CTOKEN,32), vm_k3); \
    _ve_lvl(1) ; \
    _ve_vstu_vss(vrsumG_##CTOKEN, 4, pGKernel+kernelIndex+16*inChannelGroup+(C)) ; \
    _ve_vstu_vss(vrsumH_##CTOKEN, 4, pGKernel+kernelIndex+17*inChannelGroup+(C)) ; \
    _ve_vstu_vss(vrsumI_##CTOKEN, 4, pGKernel+kernelIndex+18*inChannelGroup+(C)) ; \
    _ve_vstu_vss(vrsumJ_##CTOKEN, 4, pGKernel+kernelIndex+19*inChannelGroup+(C)) ; \
    _ve_vstu_vss(vrsumK_##CTOKEN, 4, pGKernel+kernelIndex+20*inChannelGroup+(C)) ; \
    _ve_vstu_vss(vrsumL_##CTOKEN, 4, pGKernel+kernelIndex+21*inChannelGroup+(C)) ; \
    _ve_vstu_vss(vrsumM_##CTOKEN, 4, pGKernel+kernelIndex+22*inChannelGroup+(C)) ; \
    _ve_vstu_vss(vrsumN_##CTOKEN, 4, pGKernel+kernelIndex+23*inChannelGroup+(C)) ; \
    _ve_lvl(VLEN) ; \
    __vr vrsumO_##CTOKEN = _ve_vfsums_vvm(vrsumOPQRSTUV_##CTOKEN, vm_k0) ; \
    __vr vrsumP_##CTOKEN = _ve_vfsums_vvm(vrsumOPQRSTUV_##CTOKEN, vm_k1) ; \
    __vr vrsumQ_##CTOKEN = _ve_vfsums_vvm(vrsumOPQRSTUV_##CTOKEN, vm_k2) ; \
    __vr vrsumR_##CTOKEN = _ve_vfsums_vvm(vrsumOPQRSTUV_##CTOKEN, vm_k3) ; \
    __vr vrsumS_##CTOKEN = _ve_vfsums_vvm(_ve_vsll_vvs(vrsumOPQRSTUV_##CTOKEN,32), vm_k0); \
    __vr vrsumT_##CTOKEN = _ve_vfsums_vvm(_ve_vsll_vvs(vrsumOPQRSTUV_##CTOKEN,32), vm_k1); \
    __vr vrsumU_##CTOKEN = _ve_vfsums_vvm(_ve_vsll_vvs(vrsumOPQRSTUV_##CTOKEN,32), vm_k2); \
    __vr vrsumV_##CTOKEN = _ve_vfsums_vvm(_ve_vsll_vvs(vrsumOPQRSTUV_##CTOKEN,32), vm_k3); \
    _ve_lvl(1) ; \
    _ve_vstu_vss(vrsumO_##CTOKEN, 4, pGKernel+kernelIndex+24*inChannelGroup+(C)) ; \
    _ve_vstu_vss(vrsumP_##CTOKEN, 4, pGKernel+kernelIndex+25*inChannelGroup+(C)) ; \
    _ve_vstu_vss(vrsumQ_##CTOKEN, 4, pGKernel+kernelIndex+26*inChannelGroup+(C)) ; \
    _ve_vstu_vss(vrsumR_##CTOKEN, 4, pGKernel+kernelIndex+27*inChannelGroup+(C)) ; \
    _ve_vstu_vss(vrsumS_##CTOKEN, 4, pGKernel+kernelIndex+28*inChannelGroup+(C)) ; \
    _ve_vstu_vss(vrsumT_##CTOKEN, 4, pGKernel+kernelIndex+29*inChannelGroup+(C)) ; \
    _ve_vstu_vss(vrsumU_##CTOKEN, 4, pGKernel+kernelIndex+30*inChannelGroup+(C)) ; \
    _ve_vstu_vss(vrsumV_##CTOKEN, 4, pGKernel+kernelIndex+31*inChannelGroup+(C)) ;

    VFSUM_STORE_R1S1(0,c0)

    c+=1;
  }
  if ( ((inChannelGroup >> 1) & 0x01) == 1 ) {
    const int64_t kernelIndex = kernGroupOffset + k     * inChannelGroup + c ;

    _ve_lvl(VLEN) ;

    INIT_VRSUM(c0)
    INIT_VRSUM(c1)

    {
      const int64_t vl = gOutWidth * gOutHeight ;

      _ve_lvl(4*vl) ;

      for (int64_t n=0; n<batch; n++) {
	const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;

	const int64_t gOutIndex  = outGroupOffset + ((n * gOutChannel + k  ) * gOutHeight ) * gOutWidth ;

	__vr vrpin_c0 = _ve_vsfa_vvss(vrhw, 2, (uint64_t)pInChannel) ;
	__vr vrin_c0 = _ve_vgtu_vv(vrpin_c0) ;
	__vr vrpin_c1 = _ve_vaddul_vsv(1*4*inHeight*inWidth, vrpin_c0) ;
	__vr vrin_c1 = _ve_vgtu_vv(vrpin_c1) ;

	__vr vrgout0123 = _ve_vldu_vss(4, pGOut+gOutIndex) ;
	__vr vrgout4567 = _ve_vldu_vss(4, pGOut+gOutIndex+ 4*gOutHeight*gOutWidth) ;
	__vr vrgout89AB = _ve_vldu_vss(4, pGOut+gOutIndex+ 8*gOutHeight*gOutWidth) ;
	__vr vrgoutCDEF = _ve_vldu_vss(4, pGOut+gOutIndex+12*gOutHeight*gOutWidth) ;
	__vr vrgoutGHIJ = _ve_vldu_vss(4, pGOut+gOutIndex+16*gOutHeight*gOutWidth) ;
	__vr vrgoutKLMN = _ve_vldu_vss(4, pGOut+gOutIndex+20*gOutHeight*gOutWidth) ;
	__vr vrgoutOPQR = _ve_vldu_vss(4, pGOut+gOutIndex+24*gOutHeight*gOutWidth) ;
	__vr vrgoutSTUV = _ve_vldu_vss(4, pGOut+gOutIndex+28*gOutHeight*gOutWidth) ;

	__vr vrgout01234567 = _ve_vshf_vvvs(vrgout0123, vrgout4567, VE_VSHUFFLE_YUZU) ;
	__vr vrgout89ABCDEF = _ve_vshf_vvvs(vrgout89AB, vrgoutCDEF, VE_VSHUFFLE_YUZU) ;
	__vr vrgoutGHIJKLMN = _ve_vshf_vvvs(vrgoutGHIJ, vrgoutKLMN, VE_VSHUFFLE_YUZU) ;
	__vr vrgoutOPQRSTUV = _ve_vshf_vvvs(vrgoutOPQR, vrgoutSTUV, VE_VSHUFFLE_YUZU) ;

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

    _ve_lvl(VLEN) ;

    INIT_VRSUM(c0)
    INIT_VRSUM(c1)
    INIT_VRSUM(c2)
    INIT_VRSUM(c3)

    {
      const int64_t vl = gOutWidth * gOutHeight ;

      _ve_lvl(4*vl) ;

      for (int64_t n=0; n<batch; n++) {
	const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;

	const int64_t gOutIndex  = outGroupOffset + ((n * gOutChannel + k  ) * gOutHeight ) * gOutWidth ;

	__vr vrpin_c0 = _ve_vsfa_vvss(vrhw, 2, (uint64_t)pInChannel) ;
	__vr vrin_c0 = _ve_vgtu_vv(vrpin_c0) ;
	__vr vrpin_c1 = _ve_vaddul_vsv(1*4*inHeight*inWidth, vrpin_c0) ;
	__vr vrin_c1 = _ve_vgtu_vv(vrpin_c1) ;
	__vr vrpin_c2 = _ve_vaddul_vsv(2*4*inHeight*inWidth, vrpin_c0) ;
	__vr vrin_c2 = _ve_vgtu_vv(vrpin_c2) ;
	__vr vrpin_c3 = _ve_vaddul_vsv(3*4*inHeight*inWidth, vrpin_c0) ;
	__vr vrin_c3 = _ve_vgtu_vv(vrpin_c3) ;

	__vr vrgout0123 = _ve_vldu_vss(4, pGOut+gOutIndex) ;
	__vr vrgout4567 = _ve_vldu_vss(4, pGOut+gOutIndex+ 4*gOutHeight*gOutWidth) ;
	__vr vrgout89AB = _ve_vldu_vss(4, pGOut+gOutIndex+ 8*gOutHeight*gOutWidth) ;
	__vr vrgoutCDEF = _ve_vldu_vss(4, pGOut+gOutIndex+12*gOutHeight*gOutWidth) ;
	__vr vrgoutGHIJ = _ve_vldu_vss(4, pGOut+gOutIndex+16*gOutHeight*gOutWidth) ;
	__vr vrgoutKLMN = _ve_vldu_vss(4, pGOut+gOutIndex+20*gOutHeight*gOutWidth) ;
	__vr vrgoutOPQR = _ve_vldu_vss(4, pGOut+gOutIndex+24*gOutHeight*gOutWidth) ;
	__vr vrgoutSTUV = _ve_vldu_vss(4, pGOut+gOutIndex+28*gOutHeight*gOutWidth) ;

	__vr vrgout01234567 = _ve_vshf_vvvs(vrgout0123, vrgout4567, VE_VSHUFFLE_YUZU) ;
	__vr vrgout89ABCDEF = _ve_vshf_vvvs(vrgout89AB, vrgoutCDEF, VE_VSHUFFLE_YUZU) ;
	__vr vrgoutGHIJKLMN = _ve_vshf_vvvs(vrgoutGHIJ, vrgoutKLMN, VE_VSHUFFLE_YUZU) ;
	__vr vrgoutOPQRSTUV = _ve_vshf_vvvs(vrgoutOPQR, vrgoutSTUV, VE_VSHUFFLE_YUZU) ;

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

    _ve_lvl(VLEN) ;

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

      _ve_lvl(4*vl) ;

      for (int64_t n=0; n<batch; n++) {
	const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;

	const int64_t gOutIndex  = outGroupOffset + ((n * gOutChannel + k  ) * gOutHeight ) * gOutWidth ;

	__vr vrpin_c0 = _ve_vsfa_vvss(vrhw, 2, (uint64_t)pInChannel) ;
	__vr vrin_c0 = _ve_vgtu_vv(vrpin_c0) ;
	__vr vrpin_c1 = _ve_vaddul_vsv(1*4*inHeight*inWidth, vrpin_c0) ;
	__vr vrin_c1 = _ve_vgtu_vv(vrpin_c1) ;
	__vr vrpin_c2 = _ve_vaddul_vsv(2*4*inHeight*inWidth, vrpin_c0) ;
	__vr vrin_c2 = _ve_vgtu_vv(vrpin_c2) ;
	__vr vrpin_c3 = _ve_vaddul_vsv(3*4*inHeight*inWidth, vrpin_c0) ;
	__vr vrin_c3 = _ve_vgtu_vv(vrpin_c3) ;
	__vr vrpin_c4 = _ve_vaddul_vsv(4*4*inHeight*inWidth, vrpin_c0) ;
	__vr vrin_c4 = _ve_vgtu_vv(vrpin_c4) ;
	__vr vrpin_c5 = _ve_vaddul_vsv(5*4*inHeight*inWidth, vrpin_c0) ;
	__vr vrin_c5 = _ve_vgtu_vv(vrpin_c5) ;
	__vr vrpin_c6 = _ve_vaddul_vsv(6*4*inHeight*inWidth, vrpin_c0) ;
	__vr vrin_c6 = _ve_vgtu_vv(vrpin_c6) ;
	__vr vrpin_c7 = _ve_vaddul_vsv(7*4*inHeight*inWidth, vrpin_c0) ;
	__vr vrin_c7 = _ve_vgtu_vv(vrpin_c7) ;

	__vr vrgout0123 = _ve_vldu_vss(4, pGOut+gOutIndex) ;
	__vr vrgout4567 = _ve_vldu_vss(4, pGOut+gOutIndex+ 4*gOutHeight*gOutWidth) ;
	__vr vrgout89AB = _ve_vldu_vss(4, pGOut+gOutIndex+ 8*gOutHeight*gOutWidth) ;
	__vr vrgoutCDEF = _ve_vldu_vss(4, pGOut+gOutIndex+12*gOutHeight*gOutWidth) ;
	__vr vrgoutGHIJ = _ve_vldu_vss(4, pGOut+gOutIndex+16*gOutHeight*gOutWidth) ;
	__vr vrgoutKLMN = _ve_vldu_vss(4, pGOut+gOutIndex+20*gOutHeight*gOutWidth) ;
	__vr vrgoutOPQR = _ve_vldu_vss(4, pGOut+gOutIndex+24*gOutHeight*gOutWidth) ;
	__vr vrgoutSTUV = _ve_vldu_vss(4, pGOut+gOutIndex+28*gOutHeight*gOutWidth) ;

	__vr vrgout01234567 = _ve_vshf_vvvs(vrgout0123, vrgout4567, VE_VSHUFFLE_YUZU) ;
	__vr vrgout89ABCDEF = _ve_vshf_vvvs(vrgout89AB, vrgoutCDEF, VE_VSHUFFLE_YUZU) ;
	__vr vrgoutGHIJKLMN = _ve_vshf_vvvs(vrgoutGHIJ, vrgoutKLMN, VE_VSHUFFLE_YUZU) ;
	__vr vrgoutOPQRSTUV = _ve_vshf_vvvs(vrgoutOPQR, vrgoutSTUV, VE_VSHUFFLE_YUZU) ;

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

    _ve_lvl(VLEN) ;

    __vr vrseq = _ve_vseq_v() ;			// xy
    __vr vry  = _ve_vdivsl_vvs(vrseq, gOutWidth) ;
    __vr vrx  = _ve_vsubsl_vvv(vrseq, _ve_vmulul_vsv(gOutWidth,vry)) ;

    __vm256 vm_k0 = _ve_vfmkl_mcv(VECC_IL, _ve_vaddsl_vsv(-gOutPixels, vrseq)) ;
    __vm256 vm_k1 = _ve_vfmkl_mcv(VECC_IL, _ve_vaddsl_vsv(-2*gOutPixels, vrseq)) ;
    __vm256 vm_k2 = _ve_vfmkl_mcv(VECC_IL, _ve_vaddsl_vsv(-3*gOutPixels, vrseq)) ;
    __vm256 vm_k3 = _ve_negm_mm(vm_k2) ;
    vm_k2 = _ve_andm_mmm(vm_k2, _ve_negm_mm(vm_k1)) ;
    vm_k1 = _ve_andm_mmm(vm_k1, _ve_negm_mm(vm_k0)) ;

    __vr vri  = _ve_vmulsl_vsv(strideHeight, vry) ;
    __vr vrj  = _ve_vmulsl_vsv(strideWidth,  vrx) ;

    __vr vrhw = _ve_vaddul_vvv(vrj, _ve_vmulul_vsv(inWidth,vri)) ;
    vrhw = _ve_vmrg_vvvm(vrhw, _ve_vmv_vsv(-gOutPixels,vrhw), vm_k1) ;
    vrhw = _ve_vmrg_vvvm(vrhw, _ve_vmv_vsv(-gOutPixels,vrhw), vm_k2) ;
    vrhw = _ve_vmrg_vvvm(vrhw, _ve_vmv_vsv(-gOutPixels,vrhw), vm_k3) ;

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
