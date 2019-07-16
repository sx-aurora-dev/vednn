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
  const __vr vri, const __vr vrj, const int64_t nY
)
{
  int64_t c=0;
  if ( (inChannelGroup & 0x01) == 1 ) {
    const int64_t kernelIndex0 = kernGroupOffset + k     * inChannelGroup + c ;

    _ve_lvl(VLEN) ;

#define INIT_VRSUM(CTOKEN) \
    __vr vrsum_##CTOKEN = _ve_vbrdu_vs_f32(0UL) ;

    INIT_VRSUM(c0)

    for (int64_t y=0; y<gOutHeight; y+=nY) {

      const int64_t vl = gOutWidth * (gOutHeight - y < nY ? gOutHeight - y : nY) ;
      const int64_t gop = y * gOutWidth ;

      _ve_lvl(vl) ;
      __vr vrh = _ve_vaddsl_vsv(y*strideHeight, vri) ;
      __vr vrw = _ve_vaddsl_vsv(0,  vrj) ;

      __vr vrhw = _ve_vaddul_vvv(vrw, _ve_vmulul_vsv(inWidth,vrh)) ;

      for (int64_t n=0; n<batch; n++) {
	const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;

	const int64_t gOutIndex0  = outGroupOffset + ((n * gOutChannel + k  ) * gOutHeight ) * gOutWidth + gop;

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

    for (int64_t y=0; y<gOutHeight; y+=nY) {

      const int64_t vl = gOutWidth * (gOutHeight - y < nY ? gOutHeight - y : nY) ;
      const int64_t gop = y * gOutWidth ;

      _ve_lvl(vl) ;
      __vr vrh = _ve_vaddsl_vsv(y*strideHeight, vri) ;
      __vr vrw = _ve_vaddsl_vsv(0,  vrj) ;

      __vr vrhw = _ve_vaddul_vvv(vrw, _ve_vmulul_vsv(inWidth,vrh)) ;

      for (int64_t n=0; n<batch; n++) {
	const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;

	const int64_t gOutIndex0  = outGroupOffset + ((n * gOutChannel + k  ) * gOutHeight ) * gOutWidth + gop;

	__vr vrpin_c0 = _ve_vsfa_vvss(vrhw, 2, (uint64_t)pInChannel) ;
	__vr vrpin_c1 = _ve_vaddul_vsv(1*4*inHeight*inWidth, vrpin_c0) ;

	__vr vrin_c0 = _ve_vgtu_vv(vrpin_c0) ;
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

    for (int64_t y=0; y<gOutHeight; y+=nY) {

      const int64_t vl = gOutWidth * (gOutHeight - y < nY ? gOutHeight - y : nY) ;
      const int64_t gop = y * gOutWidth ;

      _ve_lvl(vl) ;
      __vr vrh = _ve_vaddsl_vsv(y*strideHeight, vri) ;
      __vr vrw = _ve_vaddsl_vsv(0,  vrj) ;

      __vr vrhw = _ve_vaddul_vvv(vrw, _ve_vmulul_vsv(inWidth,vrh)) ;

      for (int64_t n=0; n<batch; n++) {
	const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;

	const int64_t gOutIndex0  = outGroupOffset + ((n * gOutChannel + k  ) * gOutHeight ) * gOutWidth + gop;

	__vr vrpin_c0 = _ve_vsfa_vvss(vrhw, 2, (uint64_t)pInChannel) ;
	__vr vrpin_c1 = _ve_vaddul_vsv(1*4*inHeight*inWidth, vrpin_c0) ;
	__vr vrpin_c2 = _ve_vaddul_vsv(2*4*inHeight*inWidth, vrpin_c0) ;
	__vr vrpin_c3 = _ve_vaddul_vsv(3*4*inHeight*inWidth, vrpin_c0) ;

	__vr vrin_c0 = _ve_vgtu_vv(vrpin_c0) ;
	__vr vrin_c1 = _ve_vgtu_vv(vrpin_c1) ;
	__vr vrin_c2 = _ve_vgtu_vv(vrpin_c2) ;
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

    for (int64_t y=0; y<gOutHeight; y+=nY) {

      const int64_t vl = gOutWidth * (gOutHeight - y < nY ? gOutHeight - y : nY) ;
      const int64_t gop = y * gOutWidth ;

      _ve_lvl(vl) ;
      __vr vrh = _ve_vaddsl_vsv(y*strideHeight, vri) ;
      __vr vrw = _ve_vaddsl_vsv(0,  vrj) ;

      __vr vrhw = _ve_vaddul_vvv(vrw, _ve_vmulul_vsv(inWidth,vrh)) ;

      for (int64_t n=0; n<batch; n++) {
	const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;

	const int64_t gOutIndex0  = outGroupOffset + ((n * gOutChannel + k  ) * gOutHeight ) * gOutWidth + gop;

	__vr vrpin_c0 = _ve_vsfa_vvss(vrhw, 2, (uint64_t)pInChannel) ;
	__vr vrpin_c1 = _ve_vaddul_vsv(1*4*inHeight*inWidth, vrpin_c0) ;
	__vr vrpin_c2 = _ve_vaddul_vsv(2*4*inHeight*inWidth, vrpin_c0) ;
	__vr vrpin_c3 = _ve_vaddul_vsv(3*4*inHeight*inWidth, vrpin_c0) ;
	__vr vrpin_c4 = _ve_vaddul_vsv(4*4*inHeight*inWidth, vrpin_c0) ;
	__vr vrpin_c5 = _ve_vaddul_vsv(5*4*inHeight*inWidth, vrpin_c0) ;
	__vr vrpin_c6 = _ve_vaddul_vsv(6*4*inHeight*inWidth, vrpin_c0) ;
	__vr vrpin_c7 = _ve_vaddul_vsv(7*4*inHeight*inWidth, vrpin_c0) ;

	__vr vrin_c0 = _ve_vgtu_vv(vrpin_c0) ;
	__vr vrin_c1 = _ve_vgtu_vv(vrpin_c1) ;
	__vr vrin_c2 = _ve_vgtu_vv(vrpin_c2) ;
	__vr vrin_c3 = _ve_vgtu_vv(vrpin_c3) ;
	__vr vrin_c4 = _ve_vgtu_vv(vrpin_c4) ;
	__vr vrin_c5 = _ve_vgtu_vv(vrpin_c5) ;
	__vr vrin_c6 = _ve_vgtu_vv(vrpin_c6) ;
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
  const __vr vri, const __vr vrj, const int64_t nY
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

    for (int64_t y=0; y<gOutHeight; y+=nY) {

      const int64_t vl = gOutWidth * (gOutHeight - y < nY ? gOutHeight - y : nY) ;
      const int64_t gop = y * gOutWidth ;

      _ve_lvl(vl) ;
      __vr vrh = _ve_vaddsl_vsv(y*strideHeight, vri) ;
      __vr vrw = _ve_vaddsl_vsv(0,  vrj) ;

      __vr vrhw = _ve_vaddul_vvv(vrw, _ve_vmulul_vsv(inWidth,vrh)) ;

      for (int64_t n=0; n<batch; n++) {
	const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;

	const int64_t gOutIndex0  = outGroupOffset + ((n * gOutChannel + k  ) * gOutHeight ) * gOutWidth + gop;
	const int64_t gOutIndex1  = outGroupOffset + ((n * gOutChannel + k+1) * gOutHeight ) * gOutWidth + gop;

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

    for (int64_t y=0; y<gOutHeight; y+=nY) {

      const int64_t vl = gOutWidth * (gOutHeight - y < nY ? gOutHeight - y : nY) ;
      const int64_t gop = y * gOutWidth ;

      _ve_lvl(vl) ;
      __vr vrh = _ve_vaddsl_vsv(y*strideHeight, vri) ;
      __vr vrw = _ve_vaddsl_vsv(0,  vrj) ;

      __vr vrhw = _ve_vaddul_vvv(vrw, _ve_vmulul_vsv(inWidth,vrh)) ;

      for (int64_t n=0; n<batch; n++) {
	const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;

	const int64_t gOutIndex0  = outGroupOffset + ((n * gOutChannel + k  ) * gOutHeight ) * gOutWidth + gop;
	const int64_t gOutIndex1  = outGroupOffset + ((n * gOutChannel + k+1) * gOutHeight ) * gOutWidth + gop;

	__vr vrpin_c0 = _ve_vsfa_vvss(vrhw, 2, (uint64_t)pInChannel) ;
	__vr vrpin_c1 = _ve_vaddul_vsv(1*4*inHeight*inWidth, vrpin_c0) ;

	__vr vrin_c0 = _ve_vgtu_vv(vrpin_c0) ;
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

    for (int64_t y=0; y<gOutHeight; y+=nY) {

      const int64_t vl = gOutWidth * (gOutHeight - y < nY ? gOutHeight - y : nY) ;
      const int64_t gop = y * gOutWidth ;

      _ve_lvl(vl) ;
      __vr vrh = _ve_vaddsl_vsv(y*strideHeight, vri) ;
      __vr vrw = _ve_vaddsl_vsv(0,  vrj) ;

      __vr vrhw = _ve_vaddul_vvv(vrw, _ve_vmulul_vsv(inWidth,vrh)) ;

      for (int64_t n=0; n<batch; n++) {
	const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;

	const int64_t gOutIndex0  = outGroupOffset + ((n * gOutChannel + k  ) * gOutHeight ) * gOutWidth + gop;
	const int64_t gOutIndex1  = outGroupOffset + ((n * gOutChannel + k+1) * gOutHeight ) * gOutWidth + gop;

	__vr vrpin_c0 = _ve_vsfa_vvss(vrhw, 2, (uint64_t)pInChannel) ;
	__vr vrpin_c1 = _ve_vaddul_vsv(1*4*inHeight*inWidth, vrpin_c0) ;
	__vr vrpin_c2 = _ve_vaddul_vsv(2*4*inHeight*inWidth, vrpin_c0) ;
	__vr vrpin_c3 = _ve_vaddul_vsv(3*4*inHeight*inWidth, vrpin_c0) ;

	__vr vrin_c0 = _ve_vgtu_vv(vrpin_c0) ;
	__vr vrin_c1 = _ve_vgtu_vv(vrpin_c1) ;
	__vr vrin_c2 = _ve_vgtu_vv(vrpin_c2) ;
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

    for (int64_t y=0; y<gOutHeight; y+=nY) {

      const int64_t vl = gOutWidth * (gOutHeight - y < nY ? gOutHeight - y : nY) ;
      const int64_t gop = y * gOutWidth ;

      _ve_lvl(vl) ;
      __vr vrh = _ve_vaddsl_vsv(y*strideHeight, vri) ;
      __vr vrw = _ve_vaddsl_vsv(0,  vrj) ;

      __vr vrhw = _ve_vaddul_vvv(vrw, _ve_vmulul_vsv(inWidth,vrh)) ;

      for (int64_t n=0; n<batch; n++) {
	const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;

	const int64_t gOutIndex0  = outGroupOffset + ((n * gOutChannel + k  ) * gOutHeight ) * gOutWidth + gop;
	const int64_t gOutIndex1  = outGroupOffset + ((n * gOutChannel + k+1) * gOutHeight ) * gOutWidth + gop;

	__vr vrpin_c0 = _ve_vsfa_vvss(vrhw, 2, (uint64_t)pInChannel) ;
	__vr vrpin_c1 = _ve_vaddul_vsv(1*4*inHeight*inWidth, vrpin_c0) ;
	__vr vrpin_c2 = _ve_vaddul_vsv(2*4*inHeight*inWidth, vrpin_c0) ;
	__vr vrpin_c3 = _ve_vaddul_vsv(3*4*inHeight*inWidth, vrpin_c0) ;
	__vr vrpin_c4 = _ve_vaddul_vsv(4*4*inHeight*inWidth, vrpin_c0) ;
	__vr vrpin_c5 = _ve_vaddul_vsv(5*4*inHeight*inWidth, vrpin_c0) ;
	__vr vrpin_c6 = _ve_vaddul_vsv(6*4*inHeight*inWidth, vrpin_c0) ;
	__vr vrpin_c7 = _ve_vaddul_vsv(7*4*inHeight*inWidth, vrpin_c0) ;

	__vr vrin_c0 = _ve_vgtu_vv(vrpin_c0) ;
	__vr vrin_c1 = _ve_vgtu_vv(vrpin_c1) ;
	__vr vrin_c2 = _ve_vgtu_vv(vrpin_c2) ;
	__vr vrin_c3 = _ve_vgtu_vv(vrpin_c3) ;
	__vr vrin_c4 = _ve_vgtu_vv(vrpin_c4) ;
	__vr vrin_c5 = _ve_vgtu_vv(vrpin_c5) ;
	__vr vrin_c6 = _ve_vgtu_vv(vrpin_c6) ;
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
  const __vr vri, const __vr vrj, const int64_t nY
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
    __vr vrsum01_##CTOKEN = _ve_vbrd_vs_i64(0UL) ; \
    __vr vrsum23_##CTOKEN = _ve_vbrd_vs_i64(0UL) ;

    INIT_VRSUM(c0)

    for (int64_t y=0; y<gOutHeight; y+=nY) {

      const int64_t vl = gOutWidth * (gOutHeight - y < nY ? gOutHeight - y : nY) ;
      const int64_t gop = y * gOutWidth ;

      _ve_lvl(vl) ;
      __vr vrh = _ve_vaddsl_vsv(y*strideHeight, vri) ;
      __vr vrw = _ve_vaddsl_vsv(0,  vrj) ;

      __vr vrhw = _ve_vaddul_vvv(vrw, _ve_vmulul_vsv(inWidth,vrh)) ;

      for (int64_t n=0; n<batch; n++) {
	const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;

	const int64_t gOutIndex0  = outGroupOffset + ((n * gOutChannel + k  ) * gOutHeight ) * gOutWidth + gop;
	const int64_t gOutIndex1  = outGroupOffset + ((n * gOutChannel + k+1) * gOutHeight ) * gOutWidth + gop;
	const int64_t gOutIndex2  = outGroupOffset + ((n * gOutChannel + k+2) * gOutHeight ) * gOutWidth + gop;
	const int64_t gOutIndex3  = outGroupOffset + ((n * gOutChannel + k+3) * gOutHeight ) * gOutWidth + gop;

	__vr vrpin_c0 = _ve_vsfa_vvss(vrhw, 2, (uint64_t)pInChannel) ;

	__vr vrin_c0 = _ve_vgtu_vv(vrpin_c0) ;

	__vr vrgout0 = _ve_vldu_vss(4, pGOut+gOutIndex0) ;
	__vr vrgout1 = _ve_vldu_vss(4, pGOut+gOutIndex1) ;
	__vr vrgout2 = _ve_vldu_vss(4, pGOut+gOutIndex2) ;
	__vr vrgout3 = _ve_vldu_vss(4, pGOut+gOutIndex3) ;

	__vr vrgout01 = _ve_vshf_vvvs(vrgout0, vrgout1, VE_VSHUFFLE_YUZU) ;
	__vr vrgout23 = _ve_vshf_vvvs(vrgout2, vrgout3, VE_VSHUFFLE_YUZU) ;

#define VFMAD_R1S1(CTOKEN) \
          __vr vrinP_##CTOKEN = _ve_vshf_vvvs(vrin_##CTOKEN, vrin_##CTOKEN, VE_VSHUFFLE_YUZU) ; \
          vrsum01_##CTOKEN = _ve_pvfmad_vvvv(vrsum01_##CTOKEN, vrinP_##CTOKEN, vrgout01) ; \
          vrsum23_##CTOKEN = _ve_pvfmad_vvvv(vrsum23_##CTOKEN, vrinP_##CTOKEN, vrgout23) ;

	  VFMAD_R1S1(c0)

      } // batch
    } // gOutHeight

#define VFSUM_STORE_R1S1(C,CTOKEN) \
    _ve_lvl(VLEN) ; \
    __vr vrsum0_##CTOKEN = _ve_vfsums_vv(vrsum01_##CTOKEN) ; \
    __vr vrsum1_##CTOKEN = _ve_vfsums_vv(_ve_vsll_vvs(vrsum01_##CTOKEN,32)); \
    __vr vrsum2_##CTOKEN = _ve_vfsums_vv(vrsum23_##CTOKEN) ; \
    __vr vrsum3_##CTOKEN = _ve_vfsums_vv(_ve_vsll_vvs(vrsum23_##CTOKEN,32)); \
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

    for (int64_t y=0; y<gOutHeight; y+=nY) {

      const int64_t vl = gOutWidth * (gOutHeight - y < nY ? gOutHeight - y : nY) ;
      const int64_t gop = y * gOutWidth ;

      _ve_lvl(vl) ;
      __vr vrh = _ve_vaddsl_vsv(y*strideHeight, vri) ;
      __vr vrw = _ve_vaddsl_vsv(0,  vrj) ;

      __vr vrhw = _ve_vaddul_vvv(vrw, _ve_vmulul_vsv(inWidth,vrh)) ;

      for (int64_t n=0; n<batch; n++) {
	const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;

	const int64_t gOutIndex0  = outGroupOffset + ((n * gOutChannel + k  ) * gOutHeight ) * gOutWidth + gop;
	const int64_t gOutIndex1  = outGroupOffset + ((n * gOutChannel + k+1) * gOutHeight ) * gOutWidth + gop;
	const int64_t gOutIndex2  = outGroupOffset + ((n * gOutChannel + k+2) * gOutHeight ) * gOutWidth + gop;
	const int64_t gOutIndex3  = outGroupOffset + ((n * gOutChannel + k+3) * gOutHeight ) * gOutWidth + gop;

	__vr vrpin_c0 = _ve_vsfa_vvss(vrhw, 2, (uint64_t)pInChannel) ;
	__vr vrpin_c1 = _ve_vaddul_vsv(1*4*inHeight*inWidth, vrpin_c0) ;

	__vr vrin_c0 = _ve_vgtu_vv(vrpin_c0) ;
	__vr vrin_c1 = _ve_vgtu_vv(vrpin_c1) ;

	__vr vrgout0 = _ve_vldu_vss(4, pGOut+gOutIndex0) ;
	__vr vrgout1 = _ve_vldu_vss(4, pGOut+gOutIndex1) ;
	__vr vrgout2 = _ve_vldu_vss(4, pGOut+gOutIndex2) ;
	__vr vrgout3 = _ve_vldu_vss(4, pGOut+gOutIndex3) ;

	__vr vrgout01 = _ve_vshf_vvvs(vrgout0, vrgout1, VE_VSHUFFLE_YUZU) ;
	__vr vrgout23 = _ve_vshf_vvvs(vrgout2, vrgout3, VE_VSHUFFLE_YUZU) ;

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
    const int64_t kernelIndex2 = kernGroupOffset + (k+2) * inChannelGroup + c ;
    const int64_t kernelIndex3 = kernGroupOffset + (k+3) * inChannelGroup + c ;

    _ve_lvl(VLEN) ;

    INIT_VRSUM(c0)
    INIT_VRSUM(c1)
    INIT_VRSUM(c2)
    INIT_VRSUM(c3)

    for (int64_t y=0; y<gOutHeight; y+=nY) {

      const int64_t vl = gOutWidth * (gOutHeight - y < nY ? gOutHeight - y : nY) ;
      const int64_t gop = y * gOutWidth ;

      _ve_lvl(vl) ;
      __vr vrh = _ve_vaddsl_vsv(y*strideHeight, vri) ;
      __vr vrw = _ve_vaddsl_vsv(0,  vrj) ;

      __vr vrhw = _ve_vaddul_vvv(vrw, _ve_vmulul_vsv(inWidth,vrh)) ;

      for (int64_t n=0; n<batch; n++) {
	const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;

	const int64_t gOutIndex0  = outGroupOffset + ((n * gOutChannel + k  ) * gOutHeight ) * gOutWidth + gop;
	const int64_t gOutIndex1  = outGroupOffset + ((n * gOutChannel + k+1) * gOutHeight ) * gOutWidth + gop;
	const int64_t gOutIndex2  = outGroupOffset + ((n * gOutChannel + k+2) * gOutHeight ) * gOutWidth + gop;
	const int64_t gOutIndex3  = outGroupOffset + ((n * gOutChannel + k+3) * gOutHeight ) * gOutWidth + gop;

	__vr vrpin_c0 = _ve_vsfa_vvss(vrhw, 2, (uint64_t)pInChannel) ;
	__vr vrpin_c1 = _ve_vaddul_vsv(1*4*inHeight*inWidth, vrpin_c0) ;
	__vr vrpin_c2 = _ve_vaddul_vsv(2*4*inHeight*inWidth, vrpin_c0) ;
	__vr vrpin_c3 = _ve_vaddul_vsv(3*4*inHeight*inWidth, vrpin_c0) ;

	__vr vrin_c0 = _ve_vgtu_vv(vrpin_c0) ;
	__vr vrin_c1 = _ve_vgtu_vv(vrpin_c1) ;
	__vr vrin_c2 = _ve_vgtu_vv(vrpin_c2) ;
	__vr vrin_c3 = _ve_vgtu_vv(vrpin_c3) ;

	__vr vrgout0 = _ve_vldu_vss(4, pGOut+gOutIndex0) ;
	__vr vrgout1 = _ve_vldu_vss(4, pGOut+gOutIndex1) ;
	__vr vrgout2 = _ve_vldu_vss(4, pGOut+gOutIndex2) ;
	__vr vrgout3 = _ve_vldu_vss(4, pGOut+gOutIndex3) ;

	__vr vrgout01 = _ve_vshf_vvvs(vrgout0, vrgout1, VE_VSHUFFLE_YUZU) ;
	__vr vrgout23 = _ve_vshf_vvvs(vrgout2, vrgout3, VE_VSHUFFLE_YUZU) ;

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

    for (int64_t y=0; y<gOutHeight; y+=nY) {

      const int64_t vl = gOutWidth * (gOutHeight - y < nY ? gOutHeight - y : nY) ;
      const int64_t gop = y * gOutWidth ;

      _ve_lvl(vl) ;
      __vr vrh = _ve_vaddsl_vsv(y*strideHeight, vri) ;
      __vr vrw = _ve_vaddsl_vsv(0,  vrj) ;

      __vr vrhw = _ve_vaddul_vvv(vrw, _ve_vmulul_vsv(inWidth,vrh)) ;

      for (int64_t n=0; n<batch; n++) {
	const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;

	const int64_t gOutIndex0  = outGroupOffset + ((n * gOutChannel + k  ) * gOutHeight ) * gOutWidth + gop;
	const int64_t gOutIndex1  = outGroupOffset + ((n * gOutChannel + k+1) * gOutHeight ) * gOutWidth + gop;
	const int64_t gOutIndex2  = outGroupOffset + ((n * gOutChannel + k+2) * gOutHeight ) * gOutWidth + gop;
	const int64_t gOutIndex3  = outGroupOffset + ((n * gOutChannel + k+3) * gOutHeight ) * gOutWidth + gop;

	__vr vrpin_c0 = _ve_vsfa_vvss(vrhw, 2, (uint64_t)pInChannel) ;
	__vr vrpin_c1 = _ve_vaddul_vsv(1*4*inHeight*inWidth, vrpin_c0) ;
	__vr vrpin_c2 = _ve_vaddul_vsv(2*4*inHeight*inWidth, vrpin_c0) ;
	__vr vrpin_c3 = _ve_vaddul_vsv(3*4*inHeight*inWidth, vrpin_c0) ;
	__vr vrpin_c4 = _ve_vaddul_vsv(4*4*inHeight*inWidth, vrpin_c0) ;
	__vr vrpin_c5 = _ve_vaddul_vsv(5*4*inHeight*inWidth, vrpin_c0) ;
	__vr vrpin_c6 = _ve_vaddul_vsv(6*4*inHeight*inWidth, vrpin_c0) ;
	__vr vrpin_c7 = _ve_vaddul_vsv(7*4*inHeight*inWidth, vrpin_c0) ;

	__vr vrin_c0 = _ve_vgtu_vv(vrpin_c0) ;
	__vr vrin_c1 = _ve_vgtu_vv(vrpin_c1) ;
	__vr vrin_c2 = _ve_vgtu_vv(vrpin_c2) ;
	__vr vrin_c3 = _ve_vgtu_vv(vrpin_c3) ;
	__vr vrin_c4 = _ve_vgtu_vv(vrpin_c4) ;
	__vr vrin_c5 = _ve_vgtu_vv(vrpin_c5) ;
	__vr vrin_c6 = _ve_vgtu_vv(vrpin_c6) ;
	__vr vrin_c7 = _ve_vgtu_vv(vrpin_c7) ;

	__vr vrgout0 = _ve_vldu_vss(4, pGOut+gOutIndex0) ;
	__vr vrgout1 = _ve_vldu_vss(4, pGOut+gOutIndex1) ;
	__vr vrgout2 = _ve_vldu_vss(4, pGOut+gOutIndex2) ;
	__vr vrgout3 = _ve_vldu_vss(4, pGOut+gOutIndex3) ;

	__vr vrgout01 = _ve_vshf_vvvs(vrgout0, vrgout1, VE_VSHUFFLE_YUZU) ;
	__vr vrgout23 = _ve_vshf_vvvs(vrgout2, vrgout3, VE_VSHUFFLE_YUZU) ;

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
  const __vr vri, const __vr vrj, const int64_t nY
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
    __vr vrsum01_##CTOKEN = _ve_vbrd_vs_i64(0UL) ; \
    __vr vrsum23_##CTOKEN = _ve_vbrd_vs_i64(0UL) ; \
    __vr vrsum45_##CTOKEN = _ve_vbrd_vs_i64(0UL) ; \
    __vr vrsum67_##CTOKEN = _ve_vbrd_vs_i64(0UL) ;

    INIT_VRSUM(c0)

    for (int64_t y=0; y<gOutHeight; y+=nY) {

      const int64_t vl = gOutWidth * (gOutHeight - y < nY ? gOutHeight - y : nY) ;
      const int64_t gop = y * gOutWidth ;

      _ve_lvl(vl) ;
      __vr vrh = _ve_vaddsl_vsv(y*strideHeight, vri) ;
      __vr vrw = _ve_vaddsl_vsv(0,  vrj) ;

      __vr vrhw = _ve_vaddul_vvv(vrw, _ve_vmulul_vsv(inWidth,vrh)) ;

      for (int64_t n=0; n<batch; n++) {
	const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;

	const int64_t gOutIndex0  = outGroupOffset + ((n * gOutChannel + k  ) * gOutHeight ) * gOutWidth + gop;
	const int64_t gOutIndex1  = outGroupOffset + ((n * gOutChannel + k+1) * gOutHeight ) * gOutWidth + gop;
	const int64_t gOutIndex2  = outGroupOffset + ((n * gOutChannel + k+2) * gOutHeight ) * gOutWidth + gop;
	const int64_t gOutIndex3  = outGroupOffset + ((n * gOutChannel + k+3) * gOutHeight ) * gOutWidth + gop;
	const int64_t gOutIndex4  = outGroupOffset + ((n * gOutChannel + k+4) * gOutHeight ) * gOutWidth + gop;
	const int64_t gOutIndex5  = outGroupOffset + ((n * gOutChannel + k+5) * gOutHeight ) * gOutWidth + gop;
	const int64_t gOutIndex6  = outGroupOffset + ((n * gOutChannel + k+6) * gOutHeight ) * gOutWidth + gop;
	const int64_t gOutIndex7  = outGroupOffset + ((n * gOutChannel + k+7) * gOutHeight ) * gOutWidth + gop;

	__vr vrpin_c0 = _ve_vsfa_vvss(vrhw, 2, (uint64_t)pInChannel) ;

	__vr vrin_c0 = _ve_vgtu_vv(vrpin_c0) ;

	__vr vrgout0 = _ve_vldu_vss(4, pGOut+gOutIndex0) ;
	__vr vrgout1 = _ve_vldu_vss(4, pGOut+gOutIndex1) ;
	__vr vrgout2 = _ve_vldu_vss(4, pGOut+gOutIndex2) ;
	__vr vrgout3 = _ve_vldu_vss(4, pGOut+gOutIndex3) ;
	__vr vrgout4 = _ve_vldu_vss(4, pGOut+gOutIndex4) ;
	__vr vrgout5 = _ve_vldu_vss(4, pGOut+gOutIndex5) ;
	__vr vrgout6 = _ve_vldu_vss(4, pGOut+gOutIndex6) ;
	__vr vrgout7 = _ve_vldu_vss(4, pGOut+gOutIndex7) ;

	__vr vrgout01 = _ve_vshf_vvvs(vrgout0, vrgout1, VE_VSHUFFLE_YUZU) ;
	__vr vrgout23 = _ve_vshf_vvvs(vrgout2, vrgout3, VE_VSHUFFLE_YUZU) ;
	__vr vrgout45 = _ve_vshf_vvvs(vrgout4, vrgout5, VE_VSHUFFLE_YUZU) ;
	__vr vrgout67 = _ve_vshf_vvvs(vrgout6, vrgout7, VE_VSHUFFLE_YUZU) ;

#define VFMAD_R1S1(CTOKEN) \
          __vr vrinP_##CTOKEN = _ve_vshf_vvvs(vrin_##CTOKEN, vrin_##CTOKEN, VE_VSHUFFLE_YUZU) ; \
          vrsum01_##CTOKEN = _ve_pvfmad_vvvv(vrsum01_##CTOKEN, vrinP_##CTOKEN, vrgout01) ; \
          vrsum23_##CTOKEN = _ve_pvfmad_vvvv(vrsum23_##CTOKEN, vrinP_##CTOKEN, vrgout23) ; \
          vrsum45_##CTOKEN = _ve_pvfmad_vvvv(vrsum45_##CTOKEN, vrinP_##CTOKEN, vrgout45) ; \
          vrsum67_##CTOKEN = _ve_pvfmad_vvvv(vrsum67_##CTOKEN, vrinP_##CTOKEN, vrgout67) ;

	  VFMAD_R1S1(c0)

      } // batch
    } // gOutHeight

#define VFSUM_STORE_R1S1(C,CTOKEN) \
    _ve_lvl(VLEN) ; \
    __vr vrsum0_##CTOKEN = _ve_vfsums_vv(vrsum01_##CTOKEN) ; \
    __vr vrsum1_##CTOKEN = _ve_vfsums_vv(_ve_vsll_vvs(vrsum01_##CTOKEN,32)); \
    __vr vrsum2_##CTOKEN = _ve_vfsums_vv(vrsum23_##CTOKEN) ; \
    __vr vrsum3_##CTOKEN = _ve_vfsums_vv(_ve_vsll_vvs(vrsum23_##CTOKEN,32)); \
    __vr vrsum4_##CTOKEN = _ve_vfsums_vv(vrsum45_##CTOKEN) ; \
    __vr vrsum5_##CTOKEN = _ve_vfsums_vv(_ve_vsll_vvs(vrsum45_##CTOKEN,32)); \
    __vr vrsum6_##CTOKEN = _ve_vfsums_vv(vrsum67_##CTOKEN) ; \
    __vr vrsum7_##CTOKEN = _ve_vfsums_vv(_ve_vsll_vvs(vrsum67_##CTOKEN,32)); \
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

    for (int64_t y=0; y<gOutHeight; y+=nY) {

      const int64_t vl = gOutWidth * (gOutHeight - y < nY ? gOutHeight - y : nY) ;
      const int64_t gop = y * gOutWidth ;

      _ve_lvl(vl) ;
      __vr vrh = _ve_vaddsl_vsv(y*strideHeight, vri) ;
      __vr vrw = _ve_vaddsl_vsv(0,  vrj) ;

      __vr vrhw = _ve_vaddul_vvv(vrw, _ve_vmulul_vsv(inWidth,vrh)) ;

      for (int64_t n=0; n<batch; n++) {
	const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;

	const int64_t gOutIndex0  = outGroupOffset + ((n * gOutChannel + k  ) * gOutHeight ) * gOutWidth + gop;
	const int64_t gOutIndex1  = outGroupOffset + ((n * gOutChannel + k+1) * gOutHeight ) * gOutWidth + gop;
	const int64_t gOutIndex2  = outGroupOffset + ((n * gOutChannel + k+2) * gOutHeight ) * gOutWidth + gop;
	const int64_t gOutIndex3  = outGroupOffset + ((n * gOutChannel + k+3) * gOutHeight ) * gOutWidth + gop;
	const int64_t gOutIndex4  = outGroupOffset + ((n * gOutChannel + k+4) * gOutHeight ) * gOutWidth + gop;
	const int64_t gOutIndex5  = outGroupOffset + ((n * gOutChannel + k+5) * gOutHeight ) * gOutWidth + gop;
	const int64_t gOutIndex6  = outGroupOffset + ((n * gOutChannel + k+6) * gOutHeight ) * gOutWidth + gop;
	const int64_t gOutIndex7  = outGroupOffset + ((n * gOutChannel + k+7) * gOutHeight ) * gOutWidth + gop;

	__vr vrpin_c0 = _ve_vsfa_vvss(vrhw, 2, (uint64_t)pInChannel) ;
	__vr vrpin_c1 = _ve_vaddul_vsv(1*4*inHeight*inWidth, vrpin_c0) ;

	__vr vrin_c0 = _ve_vgtu_vv(vrpin_c0) ;
	__vr vrin_c1 = _ve_vgtu_vv(vrpin_c1) ;

	__vr vrgout0 = _ve_vldu_vss(4, pGOut+gOutIndex0) ;
	__vr vrgout1 = _ve_vldu_vss(4, pGOut+gOutIndex1) ;
	__vr vrgout2 = _ve_vldu_vss(4, pGOut+gOutIndex2) ;
	__vr vrgout3 = _ve_vldu_vss(4, pGOut+gOutIndex3) ;
	__vr vrgout4 = _ve_vldu_vss(4, pGOut+gOutIndex4) ;
	__vr vrgout5 = _ve_vldu_vss(4, pGOut+gOutIndex5) ;
	__vr vrgout6 = _ve_vldu_vss(4, pGOut+gOutIndex6) ;
	__vr vrgout7 = _ve_vldu_vss(4, pGOut+gOutIndex7) ;

	__vr vrgout01 = _ve_vshf_vvvs(vrgout0, vrgout1, VE_VSHUFFLE_YUZU) ;
	__vr vrgout23 = _ve_vshf_vvvs(vrgout2, vrgout3, VE_VSHUFFLE_YUZU) ;
	__vr vrgout45 = _ve_vshf_vvvs(vrgout4, vrgout5, VE_VSHUFFLE_YUZU) ;
	__vr vrgout67 = _ve_vshf_vvvs(vrgout6, vrgout7, VE_VSHUFFLE_YUZU) ;

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

    for (int64_t y=0; y<gOutHeight; y+=nY) {

      const int64_t vl = gOutWidth * (gOutHeight - y < nY ? gOutHeight - y : nY) ;
      const int64_t gop = y * gOutWidth ;

      _ve_lvl(vl) ;
      __vr vrh = _ve_vaddsl_vsv(y*strideHeight, vri) ;
      __vr vrw = _ve_vaddsl_vsv(0,  vrj) ;

      __vr vrhw = _ve_vaddul_vvv(vrw, _ve_vmulul_vsv(inWidth,vrh)) ;

      for (int64_t n=0; n<batch; n++) {
	const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;

	const int64_t gOutIndex0  = outGroupOffset + ((n * gOutChannel + k  ) * gOutHeight ) * gOutWidth + gop;
	const int64_t gOutIndex1  = outGroupOffset + ((n * gOutChannel + k+1) * gOutHeight ) * gOutWidth + gop;
	const int64_t gOutIndex2  = outGroupOffset + ((n * gOutChannel + k+2) * gOutHeight ) * gOutWidth + gop;
	const int64_t gOutIndex3  = outGroupOffset + ((n * gOutChannel + k+3) * gOutHeight ) * gOutWidth + gop;
	const int64_t gOutIndex4  = outGroupOffset + ((n * gOutChannel + k+4) * gOutHeight ) * gOutWidth + gop;
	const int64_t gOutIndex5  = outGroupOffset + ((n * gOutChannel + k+5) * gOutHeight ) * gOutWidth + gop;
	const int64_t gOutIndex6  = outGroupOffset + ((n * gOutChannel + k+6) * gOutHeight ) * gOutWidth + gop;
	const int64_t gOutIndex7  = outGroupOffset + ((n * gOutChannel + k+7) * gOutHeight ) * gOutWidth + gop;

	__vr vrpin_c0 = _ve_vsfa_vvss(vrhw, 2, (uint64_t)pInChannel) ;
	__vr vrpin_c1 = _ve_vaddul_vsv(1*4*inHeight*inWidth, vrpin_c0) ;
	__vr vrpin_c2 = _ve_vaddul_vsv(2*4*inHeight*inWidth, vrpin_c0) ;
	__vr vrpin_c3 = _ve_vaddul_vsv(3*4*inHeight*inWidth, vrpin_c0) ;

	__vr vrin_c0 = _ve_vgtu_vv(vrpin_c0) ;
	__vr vrin_c1 = _ve_vgtu_vv(vrpin_c1) ;
	__vr vrin_c2 = _ve_vgtu_vv(vrpin_c2) ;
	__vr vrin_c3 = _ve_vgtu_vv(vrpin_c3) ;

	__vr vrgout0 = _ve_vldu_vss(4, pGOut+gOutIndex0) ;
	__vr vrgout1 = _ve_vldu_vss(4, pGOut+gOutIndex1) ;
	__vr vrgout2 = _ve_vldu_vss(4, pGOut+gOutIndex2) ;
	__vr vrgout3 = _ve_vldu_vss(4, pGOut+gOutIndex3) ;
	__vr vrgout4 = _ve_vldu_vss(4, pGOut+gOutIndex4) ;
	__vr vrgout5 = _ve_vldu_vss(4, pGOut+gOutIndex5) ;
	__vr vrgout6 = _ve_vldu_vss(4, pGOut+gOutIndex6) ;
	__vr vrgout7 = _ve_vldu_vss(4, pGOut+gOutIndex7) ;

	__vr vrgout01 = _ve_vshf_vvvs(vrgout0, vrgout1, VE_VSHUFFLE_YUZU) ;
	__vr vrgout23 = _ve_vshf_vvvs(vrgout2, vrgout3, VE_VSHUFFLE_YUZU) ;
	__vr vrgout45 = _ve_vshf_vvvs(vrgout4, vrgout5, VE_VSHUFFLE_YUZU) ;
	__vr vrgout67 = _ve_vshf_vvvs(vrgout6, vrgout7, VE_VSHUFFLE_YUZU) ;

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

    for (int64_t y=0; y<gOutHeight; y+=nY) {

      const int64_t vl = gOutWidth * (gOutHeight - y < nY ? gOutHeight - y : nY) ;
      const int64_t gop = y * gOutWidth ;

      _ve_lvl(vl) ;
      __vr vrh = _ve_vaddsl_vsv(y*strideHeight, vri) ;
      __vr vrw = _ve_vaddsl_vsv(0,  vrj) ;

      __vr vrhw = _ve_vaddul_vvv(vrw, _ve_vmulul_vsv(inWidth,vrh)) ;

      for (int64_t n=0; n<batch; n++) {
	const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;

	const int64_t gOutIndex0  = outGroupOffset + ((n * gOutChannel + k  ) * gOutHeight ) * gOutWidth + gop;
	const int64_t gOutIndex1  = outGroupOffset + ((n * gOutChannel + k+1) * gOutHeight ) * gOutWidth + gop;
	const int64_t gOutIndex2  = outGroupOffset + ((n * gOutChannel + k+2) * gOutHeight ) * gOutWidth + gop;
	const int64_t gOutIndex3  = outGroupOffset + ((n * gOutChannel + k+3) * gOutHeight ) * gOutWidth + gop;
	const int64_t gOutIndex4  = outGroupOffset + ((n * gOutChannel + k+4) * gOutHeight ) * gOutWidth + gop;
	const int64_t gOutIndex5  = outGroupOffset + ((n * gOutChannel + k+5) * gOutHeight ) * gOutWidth + gop;
	const int64_t gOutIndex6  = outGroupOffset + ((n * gOutChannel + k+6) * gOutHeight ) * gOutWidth + gop;
	const int64_t gOutIndex7  = outGroupOffset + ((n * gOutChannel + k+7) * gOutHeight ) * gOutWidth + gop;

	__vr vrpin_c0 = _ve_vsfa_vvss(vrhw, 2, (uint64_t)pInChannel) ;
	__vr vrpin_c1 = _ve_vaddul_vsv(1*4*inHeight*inWidth, vrpin_c0) ;
	__vr vrpin_c2 = _ve_vaddul_vsv(2*4*inHeight*inWidth, vrpin_c0) ;
	__vr vrpin_c3 = _ve_vaddul_vsv(3*4*inHeight*inWidth, vrpin_c0) ;
	__vr vrpin_c4 = _ve_vaddul_vsv(4*4*inHeight*inWidth, vrpin_c0) ;
	__vr vrpin_c5 = _ve_vaddul_vsv(5*4*inHeight*inWidth, vrpin_c0) ;
	__vr vrpin_c6 = _ve_vaddul_vsv(6*4*inHeight*inWidth, vrpin_c0) ;
	__vr vrpin_c7 = _ve_vaddul_vsv(7*4*inHeight*inWidth, vrpin_c0) ;

	__vr vrin_c0 = _ve_vgtu_vv(vrpin_c0) ;
	__vr vrin_c1 = _ve_vgtu_vv(vrpin_c1) ;
	__vr vrin_c2 = _ve_vgtu_vv(vrpin_c2) ;
	__vr vrin_c3 = _ve_vgtu_vv(vrpin_c3) ;
	__vr vrin_c4 = _ve_vgtu_vv(vrpin_c4) ;
	__vr vrin_c5 = _ve_vgtu_vv(vrpin_c5) ;
	__vr vrin_c6 = _ve_vgtu_vv(vrpin_c6) ;
	__vr vrin_c7 = _ve_vgtu_vv(vrpin_c7) ;

	__vr vrgout0 = _ve_vldu_vss(4, pGOut+gOutIndex0) ;
	__vr vrgout1 = _ve_vldu_vss(4, pGOut+gOutIndex1) ;
	__vr vrgout2 = _ve_vldu_vss(4, pGOut+gOutIndex2) ;
	__vr vrgout3 = _ve_vldu_vss(4, pGOut+gOutIndex3) ;
	__vr vrgout4 = _ve_vldu_vss(4, pGOut+gOutIndex4) ;
	__vr vrgout5 = _ve_vldu_vss(4, pGOut+gOutIndex5) ;
	__vr vrgout6 = _ve_vldu_vss(4, pGOut+gOutIndex6) ;
	__vr vrgout7 = _ve_vldu_vss(4, pGOut+gOutIndex7) ;

	__vr vrgout01 = _ve_vshf_vvvs(vrgout0, vrgout1, VE_VSHUFFLE_YUZU) ;
	__vr vrgout23 = _ve_vshf_vvvs(vrgout2, vrgout3, VE_VSHUFFLE_YUZU) ;
	__vr vrgout45 = _ve_vshf_vvvs(vrgout4, vrgout5, VE_VSHUFFLE_YUZU) ;
	__vr vrgout67 = _ve_vshf_vvvs(vrgout6, vrgout7, VE_VSHUFFLE_YUZU) ;

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
vednnConvolutionBackwardFilter_direct_dil1_pad0_ker1_owU128(
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
    __vr vri  = _ve_vmulsl_vsv(strideHeight, vry) ;
    __vr vrj  = _ve_vmulsl_vsv(strideWidth,  vrx) ;

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
	    vri, vrj, nY
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
	    vri, vrj, nY
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
	    vri, vrj, nY
	    ) ;
	k+=4;
      }
      for ( ;k<nOChannel; k+=8) {
	f8( pIn, inWidth, inHeight,
	    pGOut, gOutWidth, gOutHeight,
	    pGKernel,
	    strideHeight, strideWidth,
	    inChannelGroup, inChannel, gOutChannel,
	    inGroupOffset, outGroupOffset, kernGroupOffset,
	    batch, k,
	    vri, vrj, nY
	    ) ;
      } // outChannel
    } // group
  }

  return VEDNN_SUCCESS;
}
