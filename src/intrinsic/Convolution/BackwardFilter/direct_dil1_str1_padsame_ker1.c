#include <stdint.h>
#include <stdlib.h>

#include "vednn.h"

#include "velintrin.h"
#define VLEN	(256)

static inline void k1(
    const float * restrict pIn,
    const float * restrict pGOut,
    float * restrict const pGKernel,
    const int64_t inChannel,
    const int64_t inWidth,
    const int64_t inHeight,
    const int64_t gOutChannel,
    const int64_t gOutWidth,
    const int64_t gOutHeight,
    const int64_t inChannelGroup,
    const int64_t inGroupOffset,
    const int64_t outGroupOffset,
    const int64_t kernGroupOffset,
    const int64_t gOutPixels,
    const int64_t batch,
    const int64_t k
)
{
  int64_t c=0 ;
  if ( (inChannelGroup & 0x01) == 1 ) {
    const int64_t kernelIndex0 = kernGroupOffset + (k     * inChannelGroup + c) ;

#define INIT_VRSUM(CTOKEN) \
    __vr vrsum_##CTOKEN = _vel_vbrds_vsl(0.f, VLEN) ;

    INIT_VRSUM(c0)

    for (int64_t n=0; n<batch; n++) {
      for (int64_t gop = 0; gop < gOutPixels; gop+=VLEN) {
	const int64_t vl = gOutPixels - gop < VLEN ? gOutPixels - gop : VLEN ;

	const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;

	const int64_t gOutIndex = outGroupOffset + ((n * gOutChannel + k  ) * gOutHeight ) * gOutWidth + gop;

	/* memory access errors mihgt be caused (vrin) */
	__vr vrin_c0    = _vel_vldu_vssl(4,&pInChannel[gop], vl) ;

	__vr vrgout0 = _vel_vldu_vssl(4, pGOut+gOutIndex+0*gOutPixels, vl) ;

#define VFMAD_R1S1(CTOKEN) \
	vrsum_##CTOKEN = _vel_vfmads_vvvvvl(vrsum_##CTOKEN, vrin_##CTOKEN, vrgout0, vrsum_##CTOKEN, vl) ;

	VFMAD_R1S1(c0)

      } // gOutPixels
    } // batch

#define VFSUM_STORE_R1S1(C,CTOKEN) \
    vrsum_##CTOKEN = _vel_vfsums_vvl(vrsum_##CTOKEN, VLEN) ; \
    _vel_vstu_vssl(vrsum_##CTOKEN, 4, pGKernel+kernelIndex0+(C), 1) ; \

    VFSUM_STORE_R1S1(0,c0)
    c+=1 ;
  }
  if ( ((inChannelGroup >> 1) & 0x01) == 1 ) {
    const int64_t kernelIndex0 = kernGroupOffset + (k     * inChannelGroup + c) ;

    INIT_VRSUM(c0)
    INIT_VRSUM(c1)

    for (int64_t n=0; n<batch; n++) {
      for (int64_t gop = 0; gop < gOutPixels; gop+=VLEN) {
	const int64_t vl = gOutPixels - gop < VLEN ? gOutPixels - gop : VLEN ;

	const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;

	const int64_t gOutIndex = outGroupOffset + ((n * gOutChannel + k  ) * gOutHeight ) * gOutWidth + gop;

	/* memory access errors mihgt be caused (vrin) */
	__vr vrin_c0    = _vel_vldu_vssl(4,&pInChannel[gop], vl) ;
	__vr vrin_c1    = _vel_vldu_vssl(4,&pInChannel[gop+1*inHeight*inWidth], vl) ;

	__vr vrgout0 = _vel_vldu_vssl(4, pGOut+gOutIndex+0*gOutPixels, vl) ;

	VFMAD_R1S1(c0)
	VFMAD_R1S1(c1)

      } // gOutPixels
    } // batch

    VFSUM_STORE_R1S1(0,c0)
    VFSUM_STORE_R1S1(1,c1)

    c+=2;
  }
  if ( ((inChannelGroup >> 2) & 0x01) == 1 ) {
    const int64_t kernelIndex0 = kernGroupOffset + (k     * inChannelGroup + c) ;

    INIT_VRSUM(c0)
    INIT_VRSUM(c1)
    INIT_VRSUM(c2)
    INIT_VRSUM(c3)

    for (int64_t n=0; n<batch; n++) {
      for (int64_t gop = 0; gop < gOutPixels; gop+=VLEN) {
	const int64_t vl = gOutPixels - gop < VLEN ? gOutPixels - gop : VLEN ;

	const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;

	const int64_t gOutIndex = outGroupOffset + ((n * gOutChannel + k  ) * gOutHeight ) * gOutWidth + gop;

	/* memory access errors mihgt be caused (vrin) */
	__vr vrin_c0    = _vel_vldu_vssl(4,&pInChannel[gop], vl) ;
	__vr vrin_c1    = _vel_vldu_vssl(4,&pInChannel[gop+1*inHeight*inWidth], vl) ;
	__vr vrin_c2    = _vel_vldu_vssl(4,&pInChannel[gop+2*inHeight*inWidth], vl) ;
	__vr vrin_c3    = _vel_vldu_vssl(4,&pInChannel[gop+3*inHeight*inWidth], vl) ;

	__vr vrgout0 = _vel_vldu_vssl(4, pGOut+gOutIndex+0*gOutPixels, vl) ;

	VFMAD_R1S1(c0)
	VFMAD_R1S1(c1)
	VFMAD_R1S1(c2)
	VFMAD_R1S1(c3)

      } // gOutPixels
    } // batch

    VFSUM_STORE_R1S1(0,c0)
    VFSUM_STORE_R1S1(1,c1)
    VFSUM_STORE_R1S1(2,c2)
    VFSUM_STORE_R1S1(3,c3)

    c+=4 ;
  }
  for (; c<inChannelGroup; c+=8) {
    const int64_t kernelIndex0 = kernGroupOffset + (k     * inChannelGroup + c) ;

    INIT_VRSUM(c0)
    INIT_VRSUM(c1)
    INIT_VRSUM(c2)
    INIT_VRSUM(c3)
    INIT_VRSUM(c4)
    INIT_VRSUM(c5)
    INIT_VRSUM(c6)
    INIT_VRSUM(c7)
#undef INIT_VRSUM

    for (int64_t n=0; n<batch; n++) {
      for (int64_t gop = 0; gop < gOutPixels; gop+=VLEN) {
	const int64_t vl = gOutPixels - gop < VLEN ? gOutPixels - gop : VLEN ;

	const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;

	const int64_t gOutIndex = outGroupOffset + ((n * gOutChannel + k  ) * gOutHeight ) * gOutWidth + gop;

	/* memory access errors mihgt be caused (vrin) */
	__vr vrin_c0    = _vel_vldu_vssl(4,&pInChannel[gop], vl) ;
	__vr vrin_c1    = _vel_vldu_vssl(4,&pInChannel[gop+1*inHeight*inWidth], vl) ;
	__vr vrin_c2    = _vel_vldu_vssl(4,&pInChannel[gop+2*inHeight*inWidth], vl) ;
	__vr vrin_c3    = _vel_vldu_vssl(4,&pInChannel[gop+3*inHeight*inWidth], vl) ;
	__vr vrin_c4    = _vel_vldu_vssl(4,&pInChannel[gop+4*inHeight*inWidth], vl) ;
	__vr vrin_c5    = _vel_vldu_vssl(4,&pInChannel[gop+5*inHeight*inWidth], vl) ;
	__vr vrin_c6    = _vel_vldu_vssl(4,&pInChannel[gop+6*inHeight*inWidth], vl) ;
	__vr vrin_c7    = _vel_vldu_vssl(4,&pInChannel[gop+7*inHeight*inWidth], vl) ;

	__vr vrgout0 = _vel_vldu_vssl(4, pGOut+gOutIndex+0*gOutPixels, vl) ;

	VFMAD_R1S1(c0)
	VFMAD_R1S1(c1)
	VFMAD_R1S1(c2)
	VFMAD_R1S1(c3)
	VFMAD_R1S1(c4)
	VFMAD_R1S1(c5)
	VFMAD_R1S1(c6)
	VFMAD_R1S1(c7)
#undef VFMAD_R1S1
      } // gOutPixels
    } // batch

    VFSUM_STORE_R1S1(0,c0)
    VFSUM_STORE_R1S1(1,c1)
    VFSUM_STORE_R1S1(2,c2)
    VFSUM_STORE_R1S1(3,c3)
    VFSUM_STORE_R1S1(4,c4)
    VFSUM_STORE_R1S1(5,c5)
    VFSUM_STORE_R1S1(6,c6)
    VFSUM_STORE_R1S1(7,c7)
#undef VFSUM_STORE_R1S1

  } // inChannel
}

static inline void k2(
    const float * restrict pIn,
    const float * restrict pGOut,
    float * restrict const pGKernel,
    const int64_t inChannel,
    const int64_t inWidth,
    const int64_t inHeight,
    const int64_t gOutChannel,
    const int64_t gOutWidth,
    const int64_t gOutHeight,
    const int64_t inChannelGroup,
    const int64_t inGroupOffset,
    const int64_t outGroupOffset,
    const int64_t kernGroupOffset,
    const int64_t gOutPixels,
    const int64_t batch,
    const int64_t k
)
{
  int64_t c=0 ;
  if ( (inChannelGroup & 0x01) == 1 ) {
    const int64_t kernelIndex0 = kernGroupOffset + (k     * inChannelGroup + c) ;
    const int64_t kernelIndex1 = kernGroupOffset + ((k+1) * inChannelGroup + c) ;

#define INIT_VRSUM(CTOKEN) \
    __vr vrsum01_##CTOKEN = _vel_vbrdl_vsl(0UL, VLEN) ;

    INIT_VRSUM(c0)

    for (int64_t n=0; n<batch; n++) {
      for (int64_t gop = 0; gop < gOutPixels; gop+=VLEN) {
	const int64_t vl = gOutPixels - gop < VLEN ? gOutPixels - gop : VLEN ;

	const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;

	const int64_t gOutIndex = outGroupOffset + ((n * gOutChannel + k  ) * gOutHeight ) * gOutWidth + gop;

	/* memory access errors mihgt be caused (vrin) */
	__vr vrin_c0    = _vel_vldu_vssl(4,&pInChannel[gop], vl) ;

	__vr vrgout0 = _vel_vldu_vssl(4, pGOut+gOutIndex+0*gOutPixels, vl) ;
	__vr vrgout1 = _vel_vldu_vssl(4, pGOut+gOutIndex+1*gOutPixels, vl) ;

	__vr vrgout01 = _vel_vshf_vvvsl(vrgout0, vrgout1, VE_VSHUFFLE_YUZU, vl) ;

#define VFMAD_R1S1(CTOKEN) \
	__vr vrinP_##CTOKEN = _vel_vshf_vvvsl(vrin_##CTOKEN, vrin_##CTOKEN, VE_VSHUFFLE_YUZU, vl) ; \
	vrsum01_##CTOKEN = _vel_pvfmad_vvvvvl(vrsum01_##CTOKEN, vrinP_##CTOKEN, vrgout01, vrsum01_##CTOKEN, vl) ;

	VFMAD_R1S1(c0)

      } // gOutPixels
    } // batch

#define VFSUM_STORE_R1S1(C,CTOKEN) \
    __vr vrsum0_##CTOKEN = _vel_vfsums_vvl(vrsum01_##CTOKEN, VLEN) ; 				\
    __vr vrsum1_##CTOKEN = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsum01_##CTOKEN,32, VLEN), VLEN);	\
    _vel_vstu_vssl(vrsum0_##CTOKEN, 4, pGKernel+kernelIndex0+(C), 1) ; \
    _vel_vstu_vssl(vrsum1_##CTOKEN, 4, pGKernel+kernelIndex1+(C), 1) ;

    VFSUM_STORE_R1S1(0,c0)
    c+=1 ;
  }
  if ( ((inChannelGroup >> 1) & 0x01) == 1 ) {
    const int64_t kernelIndex0 = kernGroupOffset + (k     * inChannelGroup + c) ;
    const int64_t kernelIndex1 = kernGroupOffset + ((k+1) * inChannelGroup + c) ;

    INIT_VRSUM(c0)
    INIT_VRSUM(c1)

    for (int64_t n=0; n<batch; n++) {
      for (int64_t gop = 0; gop < gOutPixels; gop+=VLEN) {
	const int64_t vl = gOutPixels - gop < VLEN ? gOutPixels - gop : VLEN ;

	const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;

	const int64_t gOutIndex = outGroupOffset + ((n * gOutChannel + k  ) * gOutHeight ) * gOutWidth + gop;

	/* memory access errors mihgt be caused (vrin) */
	__vr vrin_c0    = _vel_vldu_vssl(4,&pInChannel[gop], vl) ;
	__vr vrin_c1    = _vel_vldu_vssl(4,&pInChannel[gop+1*inHeight*inWidth], vl) ;

	__vr vrgout0 = _vel_vldu_vssl(4, pGOut+gOutIndex+0*gOutPixels, vl) ;
	__vr vrgout1 = _vel_vldu_vssl(4, pGOut+gOutIndex+1*gOutPixels, vl) ;

	__vr vrgout01 = _vel_vshf_vvvsl(vrgout0, vrgout1, VE_VSHUFFLE_YUZU, vl) ;

	VFMAD_R1S1(c0)
	VFMAD_R1S1(c1)

      } // gOutPixels
    } // batch

    VFSUM_STORE_R1S1(0,c0)
    VFSUM_STORE_R1S1(1,c1)

    c+=2;
  }
  if ( ((inChannelGroup >> 2) & 0x01) == 1 ) {
    const int64_t kernelIndex0 = kernGroupOffset + (k     * inChannelGroup + c) ;
    const int64_t kernelIndex1 = kernGroupOffset + ((k+1) * inChannelGroup + c) ;

    INIT_VRSUM(c0)
    INIT_VRSUM(c1)
    INIT_VRSUM(c2)
    INIT_VRSUM(c3)

    for (int64_t n=0; n<batch; n++) {
      for (int64_t gop = 0; gop < gOutPixels; gop+=VLEN) {
	const int64_t vl = gOutPixels - gop < VLEN ? gOutPixels - gop : VLEN ;

	const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;

	const int64_t gOutIndex = outGroupOffset + ((n * gOutChannel + k  ) * gOutHeight ) * gOutWidth + gop;

	/* memory access errors mihgt be caused (vrin) */
	__vr vrin_c0    = _vel_vldu_vssl(4,&pInChannel[gop], vl) ;
	__vr vrin_c1    = _vel_vldu_vssl(4,&pInChannel[gop+1*inHeight*inWidth], vl) ;
	__vr vrin_c2    = _vel_vldu_vssl(4,&pInChannel[gop+2*inHeight*inWidth], vl) ;
	__vr vrin_c3    = _vel_vldu_vssl(4,&pInChannel[gop+3*inHeight*inWidth], vl) ;

	__vr vrgout0 = _vel_vldu_vssl(4, pGOut+gOutIndex+0*gOutPixels, vl) ;
	__vr vrgout1 = _vel_vldu_vssl(4, pGOut+gOutIndex+1*gOutPixels, vl) ;

	__vr vrgout01 = _vel_vshf_vvvsl(vrgout0, vrgout1, VE_VSHUFFLE_YUZU, vl) ;

	VFMAD_R1S1(c0)
	VFMAD_R1S1(c1)
	VFMAD_R1S1(c2)
	VFMAD_R1S1(c3)

      } // gOutPixels
    } // batch

    VFSUM_STORE_R1S1(0,c0)
    VFSUM_STORE_R1S1(1,c1)
    VFSUM_STORE_R1S1(2,c2)
    VFSUM_STORE_R1S1(3,c3)

    c+=4 ;
  }
  for (; c<inChannelGroup; c+=8) {
    const int64_t kernelIndex0 = kernGroupOffset + (k     * inChannelGroup + c) ;
    const int64_t kernelIndex1 = kernGroupOffset + ((k+1) * inChannelGroup + c) ;

    INIT_VRSUM(c0)
    INIT_VRSUM(c1)
    INIT_VRSUM(c2)
    INIT_VRSUM(c3)
    INIT_VRSUM(c4)
    INIT_VRSUM(c5)
    INIT_VRSUM(c6)
    INIT_VRSUM(c7)
#undef INIT_VRSUM

    for (int64_t n=0; n<batch; n++) {
      for (int64_t gop = 0; gop < gOutPixels; gop+=VLEN) {
	const int64_t vl = gOutPixels - gop < VLEN ? gOutPixels - gop : VLEN ;

	const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;

	const int64_t gOutIndex = outGroupOffset + ((n * gOutChannel + k  ) * gOutHeight ) * gOutWidth + gop;

	/* memory access errors mihgt be caused (vrin) */
	__vr vrin_c0    = _vel_vldu_vssl(4,&pInChannel[gop], vl) ;
	__vr vrin_c1    = _vel_vldu_vssl(4,&pInChannel[gop+1*inHeight*inWidth], vl) ;
	__vr vrin_c2    = _vel_vldu_vssl(4,&pInChannel[gop+2*inHeight*inWidth], vl) ;
	__vr vrin_c3    = _vel_vldu_vssl(4,&pInChannel[gop+3*inHeight*inWidth], vl) ;
	__vr vrin_c4    = _vel_vldu_vssl(4,&pInChannel[gop+4*inHeight*inWidth], vl) ;
	__vr vrin_c5    = _vel_vldu_vssl(4,&pInChannel[gop+5*inHeight*inWidth], vl) ;
	__vr vrin_c6    = _vel_vldu_vssl(4,&pInChannel[gop+6*inHeight*inWidth], vl) ;
	__vr vrin_c7    = _vel_vldu_vssl(4,&pInChannel[gop+7*inHeight*inWidth], vl) ;

	__vr vrgout0 = _vel_vldu_vssl(4, pGOut+gOutIndex+0*gOutPixels, vl) ;
	__vr vrgout1 = _vel_vldu_vssl(4, pGOut+gOutIndex+1*gOutPixels, vl) ;

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
      } // gOutPixels
    } // batch

    VFSUM_STORE_R1S1(0,c0)
    VFSUM_STORE_R1S1(1,c1)
    VFSUM_STORE_R1S1(2,c2)
    VFSUM_STORE_R1S1(3,c3)
    VFSUM_STORE_R1S1(4,c4)
    VFSUM_STORE_R1S1(5,c5)
    VFSUM_STORE_R1S1(6,c6)
    VFSUM_STORE_R1S1(7,c7)
#undef VFSUM_STORE_R1S1

  } // inChannel
}


static inline void k4(
    const float * restrict pIn,
    const float * restrict pGOut,
    float * restrict const pGKernel,
    const int64_t inChannel,
    const int64_t inWidth,
    const int64_t inHeight,
    const int64_t gOutChannel,
    const int64_t gOutWidth,
    const int64_t gOutHeight,
    const int64_t inChannelGroup,
    const int64_t inGroupOffset,
    const int64_t outGroupOffset,
    const int64_t kernGroupOffset,
    const int64_t gOutPixels,
    const int64_t batch,
    const int64_t k
)
{
  int64_t c=0 ;
  if ( (inChannelGroup & 0x01) == 1 ) {
    const int64_t kernelIndex0 = kernGroupOffset + (k     * inChannelGroup + c) ;
    const int64_t kernelIndex1 = kernGroupOffset + ((k+1) * inChannelGroup + c) ;
    const int64_t kernelIndex2 = kernGroupOffset + ((k+2) * inChannelGroup + c) ;
    const int64_t kernelIndex3 = kernGroupOffset + ((k+3) * inChannelGroup + c) ;

#define INIT_VRSUM(CTOKEN) \
    __vr vrsum01_##CTOKEN = _vel_vbrdl_vsl(0UL, VLEN) ; \
    __vr vrsum23_##CTOKEN = _vel_vbrdl_vsl(0UL, VLEN) ;

    INIT_VRSUM(c0)

    for (int64_t n=0; n<batch; n++) {
      for (int64_t gop = 0; gop < gOutPixels; gop+=VLEN) {
	const int64_t vl = gOutPixels - gop < VLEN ? gOutPixels - gop : VLEN ;

	const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;

	const int64_t gOutIndex = outGroupOffset + ((n * gOutChannel + k  ) * gOutHeight ) * gOutWidth + gop;

	/* memory access errors mihgt be caused (vrin) */
	__vr vrin_c0    = _vel_vldu_vssl(4,&pInChannel[gop], vl) ;

	__vr vrgout0 = _vel_vldu_vssl(4, pGOut+gOutIndex+0*gOutPixels, vl) ;
	__vr vrgout1 = _vel_vldu_vssl(4, pGOut+gOutIndex+1*gOutPixels, vl) ;
	__vr vrgout2 = _vel_vldu_vssl(4, pGOut+gOutIndex+2*gOutPixels, vl) ;
	__vr vrgout3 = _vel_vldu_vssl(4, pGOut+gOutIndex+3*gOutPixels, vl) ;

	__vr vrgout01 = _vel_vshf_vvvsl(vrgout0, vrgout1, VE_VSHUFFLE_YUZU, vl) ;
	__vr vrgout23 = _vel_vshf_vvvsl(vrgout2, vrgout3, VE_VSHUFFLE_YUZU, vl) ;

#define VFMAD_R1S1(CTOKEN) \
	__vr vrinP_##CTOKEN = _vel_vshf_vvvsl(vrin_##CTOKEN, vrin_##CTOKEN, VE_VSHUFFLE_YUZU, vl) ; 			\
	vrsum01_##CTOKEN = _vel_pvfmad_vvvvvl(vrsum01_##CTOKEN, vrinP_##CTOKEN, vrgout01, vrsum01_##CTOKEN, vl) ;	\
	vrsum23_##CTOKEN = _vel_pvfmad_vvvvvl(vrsum23_##CTOKEN, vrinP_##CTOKEN, vrgout23, vrsum23_##CTOKEN, vl) ;

	VFMAD_R1S1(c0)

      } // gOutPixels
    } // batch

#define VFSUM_STORE_R1S1(C,CTOKEN) \
    __vr vrsum0_##CTOKEN = _vel_vfsums_vvl(vrsum01_##CTOKEN, VLEN) ; \
    __vr vrsum1_##CTOKEN = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsum01_##CTOKEN,32, VLEN), VLEN); \
    __vr vrsum2_##CTOKEN = _vel_vfsums_vvl(vrsum23_##CTOKEN, VLEN) ; \
    __vr vrsum3_##CTOKEN = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsum23_##CTOKEN,32, VLEN), VLEN); \
    _vel_vstu_vssl(vrsum0_##CTOKEN, 4, pGKernel+kernelIndex0+(C), 1) ; \
    _vel_vstu_vssl(vrsum1_##CTOKEN, 4, pGKernel+kernelIndex1+(C), 1) ; \
    _vel_vstu_vssl(vrsum2_##CTOKEN, 4, pGKernel+kernelIndex2+(C), 1) ; \
    _vel_vstu_vssl(vrsum3_##CTOKEN, 4, pGKernel+kernelIndex3+(C), 1) ;

    VFSUM_STORE_R1S1(0,c0)
    c+=1 ;
  }
  if ( ((inChannelGroup >> 1) & 0x01) == 1 ) {
    const int64_t kernelIndex0 = kernGroupOffset + (k     * inChannelGroup + c) ;
    const int64_t kernelIndex1 = kernGroupOffset + ((k+1) * inChannelGroup + c) ;
    const int64_t kernelIndex2 = kernGroupOffset + ((k+2) * inChannelGroup + c) ;
    const int64_t kernelIndex3 = kernGroupOffset + ((k+3) * inChannelGroup + c) ;

    INIT_VRSUM(c0)
    INIT_VRSUM(c1)

    for (int64_t n=0; n<batch; n++) {
      for (int64_t gop = 0; gop < gOutPixels; gop+=VLEN) {
	const int64_t vl = gOutPixels - gop < VLEN ? gOutPixels - gop : VLEN ;

	const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;

	const int64_t gOutIndex = outGroupOffset + ((n * gOutChannel + k  ) * gOutHeight ) * gOutWidth + gop;

	/* memory access errors mihgt be caused (vrin) */
	__vr vrin_c0    = _vel_vldu_vssl(4,&pInChannel[gop], vl) ;
	__vr vrin_c1    = _vel_vldu_vssl(4,&pInChannel[gop+1*inHeight*inWidth], vl) ;

	__vr vrgout0 = _vel_vldu_vssl(4, pGOut+gOutIndex+0*gOutPixels, vl) ;
	__vr vrgout1 = _vel_vldu_vssl(4, pGOut+gOutIndex+1*gOutPixels, vl) ;
	__vr vrgout2 = _vel_vldu_vssl(4, pGOut+gOutIndex+2*gOutPixels, vl) ;
	__vr vrgout3 = _vel_vldu_vssl(4, pGOut+gOutIndex+3*gOutPixels, vl) ;

	__vr vrgout01 = _vel_vshf_vvvsl(vrgout0, vrgout1, VE_VSHUFFLE_YUZU, vl) ;
	__vr vrgout23 = _vel_vshf_vvvsl(vrgout2, vrgout3, VE_VSHUFFLE_YUZU, vl) ;

	VFMAD_R1S1(c0)
	VFMAD_R1S1(c1)

      } // gOutPixels
    } // batch

    VFSUM_STORE_R1S1(0,c0)
    VFSUM_STORE_R1S1(1,c1)

    c+=2;
  }
  if ( ((inChannelGroup >> 2) & 0x01) == 1 ) {
    const int64_t kernelIndex0 = kernGroupOffset + (k     * inChannelGroup + c) ;
    const int64_t kernelIndex1 = kernGroupOffset + ((k+1) * inChannelGroup + c) ;
    const int64_t kernelIndex2 = kernGroupOffset + ((k+2) * inChannelGroup + c) ;
    const int64_t kernelIndex3 = kernGroupOffset + ((k+3) * inChannelGroup + c) ;

    INIT_VRSUM(c0)
    INIT_VRSUM(c1)
    INIT_VRSUM(c2)
    INIT_VRSUM(c3)

    for (int64_t n=0; n<batch; n++) {
      for (int64_t gop = 0; gop < gOutPixels; gop+=VLEN) {
	const int64_t vl = gOutPixels - gop < VLEN ? gOutPixels - gop : VLEN ;

	const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;

	const int64_t gOutIndex = outGroupOffset + ((n * gOutChannel + k  ) * gOutHeight ) * gOutWidth + gop;

	/* memory access errors mihgt be caused (vrin) */
	__vr vrin_c0    = _vel_vldu_vssl(4,&pInChannel[gop], vl) ;
	__vr vrin_c1    = _vel_vldu_vssl(4,&pInChannel[gop+1*inHeight*inWidth], vl) ;
	__vr vrin_c2    = _vel_vldu_vssl(4,&pInChannel[gop+2*inHeight*inWidth], vl) ;
	__vr vrin_c3    = _vel_vldu_vssl(4,&pInChannel[gop+3*inHeight*inWidth], vl) ;

	__vr vrgout0 = _vel_vldu_vssl(4, pGOut+gOutIndex+0*gOutPixels, vl) ;
	__vr vrgout1 = _vel_vldu_vssl(4, pGOut+gOutIndex+1*gOutPixels, vl) ;
	__vr vrgout2 = _vel_vldu_vssl(4, pGOut+gOutIndex+2*gOutPixels, vl) ;
	__vr vrgout3 = _vel_vldu_vssl(4, pGOut+gOutIndex+3*gOutPixels, vl) ;

	__vr vrgout01 = _vel_vshf_vvvsl(vrgout0, vrgout1, VE_VSHUFFLE_YUZU, vl) ;
	__vr vrgout23 = _vel_vshf_vvvsl(vrgout2, vrgout3, VE_VSHUFFLE_YUZU, vl) ;

	VFMAD_R1S1(c0)
	VFMAD_R1S1(c1)
	VFMAD_R1S1(c2)
	VFMAD_R1S1(c3)

      } // gOutPixels
    } // batch

    VFSUM_STORE_R1S1(0,c0)
    VFSUM_STORE_R1S1(1,c1)
    VFSUM_STORE_R1S1(2,c2)
    VFSUM_STORE_R1S1(3,c3)

    c+=4 ;
  }
  for (; c<inChannelGroup; c+=8) {
    const int64_t kernelIndex0 = kernGroupOffset + (k     * inChannelGroup + c) ;
    const int64_t kernelIndex1 = kernGroupOffset + ((k+1) * inChannelGroup + c) ;
    const int64_t kernelIndex2 = kernGroupOffset + ((k+2) * inChannelGroup + c) ;
    const int64_t kernelIndex3 = kernGroupOffset + ((k+3) * inChannelGroup + c) ;

    INIT_VRSUM(c0)
    INIT_VRSUM(c1)
    INIT_VRSUM(c2)
    INIT_VRSUM(c3)
    INIT_VRSUM(c4)
    INIT_VRSUM(c5)
    INIT_VRSUM(c6)
    INIT_VRSUM(c7)
#undef INIT_VRSUM

    for (int64_t n=0; n<batch; n++) {
      for (int64_t gop = 0; gop < gOutPixels; gop+=VLEN) {
	const int64_t vl = gOutPixels - gop < VLEN ? gOutPixels - gop : VLEN ;

	const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;

	const int64_t gOutIndex = outGroupOffset + ((n * gOutChannel + k  ) * gOutHeight ) * gOutWidth + gop;

	/* memory access errors mihgt be caused (vrin) */
	__vr vrin_c0    = _vel_vldu_vssl(4,&pInChannel[gop], vl) ;
	__vr vrin_c1    = _vel_vldu_vssl(4,&pInChannel[gop+1*inHeight*inWidth], vl) ;
	__vr vrin_c2    = _vel_vldu_vssl(4,&pInChannel[gop+2*inHeight*inWidth], vl) ;
	__vr vrin_c3    = _vel_vldu_vssl(4,&pInChannel[gop+3*inHeight*inWidth], vl) ;
	__vr vrin_c4    = _vel_vldu_vssl(4,&pInChannel[gop+4*inHeight*inWidth], vl) ;
	__vr vrin_c5    = _vel_vldu_vssl(4,&pInChannel[gop+5*inHeight*inWidth], vl) ;
	__vr vrin_c6    = _vel_vldu_vssl(4,&pInChannel[gop+6*inHeight*inWidth], vl) ;
	__vr vrin_c7    = _vel_vldu_vssl(4,&pInChannel[gop+7*inHeight*inWidth], vl) ;

	__vr vrgout0 = _vel_vldu_vssl(4, pGOut+gOutIndex+0*gOutPixels, vl) ;
	__vr vrgout1 = _vel_vldu_vssl(4, pGOut+gOutIndex+1*gOutPixels, vl) ;
	__vr vrgout2 = _vel_vldu_vssl(4, pGOut+gOutIndex+2*gOutPixels, vl) ;
	__vr vrgout3 = _vel_vldu_vssl(4, pGOut+gOutIndex+3*gOutPixels, vl) ;

	__vr vrgout01 = _vel_vshf_vvvsl(vrgout0, vrgout1, VE_VSHUFFLE_YUZU, vl) ;
	__vr vrgout23 = _vel_vshf_vvvsl(vrgout2, vrgout3, VE_VSHUFFLE_YUZU, vl) ;

	VFMAD_R1S1(c0)
	VFMAD_R1S1(c1)
	VFMAD_R1S1(c2)
	VFMAD_R1S1(c3)
	VFMAD_R1S1(c4)
	VFMAD_R1S1(c5)
	VFMAD_R1S1(c6)
	VFMAD_R1S1(c7)
#undef VFMAD_R1S1
      } // gOutPixels
    } // batch

    VFSUM_STORE_R1S1(0,c0)
    VFSUM_STORE_R1S1(1,c1)
    VFSUM_STORE_R1S1(2,c2)
    VFSUM_STORE_R1S1(3,c3)
    VFSUM_STORE_R1S1(4,c4)
    VFSUM_STORE_R1S1(5,c5)
    VFSUM_STORE_R1S1(6,c6)
    VFSUM_STORE_R1S1(7,c7)
#undef VFSUM_STORE_R1S1

  } // inChannel
}

static inline void k8(
    const float * restrict pIn,
    const float * restrict pGOut,
    float * restrict const pGKernel,
    const int64_t inChannel,
    const int64_t inWidth,
    const int64_t inHeight,
    const int64_t gOutChannel,
    const int64_t gOutWidth,
    const int64_t gOutHeight,
    const int64_t inChannelGroup,
    const int64_t inGroupOffset,
    const int64_t outGroupOffset,
    const int64_t kernGroupOffset,
    const int64_t gOutPixels,
    const int64_t batch,
    const int64_t k
)
{
  int64_t c=0 ;
  if ( (inChannelGroup & 0x01) == 1 ) {
    const int64_t kernelIndex0 = kernGroupOffset + (k     * inChannelGroup + c) ;
    const int64_t kernelIndex1 = kernGroupOffset + ((k+1) * inChannelGroup + c) ;
    const int64_t kernelIndex2 = kernGroupOffset + ((k+2) * inChannelGroup + c) ;
    const int64_t kernelIndex3 = kernGroupOffset + ((k+3) * inChannelGroup + c) ;
    const int64_t kernelIndex4 = kernGroupOffset + ((k+4) * inChannelGroup + c) ;
    const int64_t kernelIndex5 = kernGroupOffset + ((k+5) * inChannelGroup + c) ;
    const int64_t kernelIndex6 = kernGroupOffset + ((k+6) * inChannelGroup + c) ;
    const int64_t kernelIndex7 = kernGroupOffset + ((k+7) * inChannelGroup + c) ;

#define INIT_VRSUM(CTOKEN) \
    __vr vrsum01_##CTOKEN = _vel_vbrdl_vsl(0UL, VLEN) ; \
    __vr vrsum23_##CTOKEN = _vel_vbrdl_vsl(0UL, VLEN) ; \
    __vr vrsum45_##CTOKEN = _vel_vbrdl_vsl(0UL, VLEN) ; \
    __vr vrsum67_##CTOKEN = _vel_vbrdl_vsl(0UL, VLEN) ;

    INIT_VRSUM(c0)

    for (int64_t n=0; n<batch; n++) {
      for (int64_t gop = 0; gop < gOutPixels; gop+=VLEN) {
	const int64_t vl = gOutPixels - gop < VLEN ? gOutPixels - gop : VLEN ;

	const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;

	const int64_t gOutIndex = outGroupOffset + ((n * gOutChannel + k  ) * gOutHeight ) * gOutWidth + gop;

	/* memory access errors mihgt be caused (vrin) */
	__vr vrin_c0    = _vel_vldu_vssl(4,&pInChannel[gop], vl) ;

	__vr vrgout0 = _vel_vldu_vssl(4, pGOut+gOutIndex+0*gOutPixels, vl) ;
	__vr vrgout1 = _vel_vldu_vssl(4, pGOut+gOutIndex+1*gOutPixels, vl) ;
	__vr vrgout2 = _vel_vldu_vssl(4, pGOut+gOutIndex+2*gOutPixels, vl) ;
	__vr vrgout3 = _vel_vldu_vssl(4, pGOut+gOutIndex+3*gOutPixels, vl) ;
	__vr vrgout4 = _vel_vldu_vssl(4, pGOut+gOutIndex+4*gOutPixels, vl) ;
	__vr vrgout5 = _vel_vldu_vssl(4, pGOut+gOutIndex+5*gOutPixels, vl) ;
	__vr vrgout6 = _vel_vldu_vssl(4, pGOut+gOutIndex+6*gOutPixels, vl) ;
	__vr vrgout7 = _vel_vldu_vssl(4, pGOut+gOutIndex+7*gOutPixels, vl) ;

	__vr vrgout01 = _vel_vshf_vvvsl(vrgout0, vrgout1, VE_VSHUFFLE_YUZU, vl) ;
	__vr vrgout23 = _vel_vshf_vvvsl(vrgout2, vrgout3, VE_VSHUFFLE_YUZU, vl) ;
	__vr vrgout45 = _vel_vshf_vvvsl(vrgout4, vrgout5, VE_VSHUFFLE_YUZU, vl) ;
	__vr vrgout67 = _vel_vshf_vvvsl(vrgout6, vrgout7, VE_VSHUFFLE_YUZU, vl) ;

#define VFMAD_R1S1(CTOKEN) \
	__vr vrinP_##CTOKEN = _vel_vshf_vvvsl(vrin_##CTOKEN, vrin_##CTOKEN, VE_VSHUFFLE_YUZU, vl) ; \
	vrsum01_##CTOKEN = _vel_pvfmad_vvvvvl(vrsum01_##CTOKEN, vrinP_##CTOKEN, vrgout01, vrsum01_##CTOKEN, vl) ; \
	vrsum23_##CTOKEN = _vel_pvfmad_vvvvvl(vrsum23_##CTOKEN, vrinP_##CTOKEN, vrgout23, vrsum23_##CTOKEN, vl) ; \
	vrsum45_##CTOKEN = _vel_pvfmad_vvvvvl(vrsum45_##CTOKEN, vrinP_##CTOKEN, vrgout45, vrsum45_##CTOKEN, vl) ; \
	vrsum67_##CTOKEN = _vel_pvfmad_vvvvvl(vrsum67_##CTOKEN, vrinP_##CTOKEN, vrgout67, vrsum67_##CTOKEN, vl) ;

	VFMAD_R1S1(c0)

      } // gOutPixels
    } // batch

#define VFSUM_STORE_R1S1(C,CTOKEN) \
    __vr vrsum0_##CTOKEN = _vel_vfsums_vvl(vrsum01_##CTOKEN, VLEN) ; \
    __vr vrsum1_##CTOKEN = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsum01_##CTOKEN,32, VLEN), VLEN); \
    __vr vrsum2_##CTOKEN = _vel_vfsums_vvl(vrsum23_##CTOKEN, VLEN) ; \
    __vr vrsum3_##CTOKEN = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsum23_##CTOKEN,32, VLEN), VLEN); \
    __vr vrsum4_##CTOKEN = _vel_vfsums_vvl(vrsum45_##CTOKEN, VLEN) ; \
    __vr vrsum5_##CTOKEN = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsum45_##CTOKEN,32, VLEN), VLEN); \
    __vr vrsum6_##CTOKEN = _vel_vfsums_vvl(vrsum67_##CTOKEN, VLEN) ; \
    __vr vrsum7_##CTOKEN = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsum67_##CTOKEN,32, VLEN), VLEN); \
    _vel_vstu_vssl(vrsum0_##CTOKEN, 4, pGKernel+kernelIndex0+(C), 1) ; \
    _vel_vstu_vssl(vrsum1_##CTOKEN, 4, pGKernel+kernelIndex1+(C), 1) ; \
    _vel_vstu_vssl(vrsum2_##CTOKEN, 4, pGKernel+kernelIndex2+(C), 1) ; \
    _vel_vstu_vssl(vrsum3_##CTOKEN, 4, pGKernel+kernelIndex3+(C), 1) ; \
    _vel_vstu_vssl(vrsum4_##CTOKEN, 4, pGKernel+kernelIndex4+(C), 1) ; \
    _vel_vstu_vssl(vrsum5_##CTOKEN, 4, pGKernel+kernelIndex5+(C), 1) ; \
    _vel_vstu_vssl(vrsum6_##CTOKEN, 4, pGKernel+kernelIndex6+(C), 1) ; \
    _vel_vstu_vssl(vrsum7_##CTOKEN, 4, pGKernel+kernelIndex7+(C), 1) ;


    VFSUM_STORE_R1S1(0,c0)
    c+=1 ;
  }
  if ( ((inChannelGroup >> 1) & 0x01) == 1 ) {
    const int64_t kernelIndex0 = kernGroupOffset + (k     * inChannelGroup + c) ;
    const int64_t kernelIndex1 = kernGroupOffset + ((k+1) * inChannelGroup + c) ;
    const int64_t kernelIndex2 = kernGroupOffset + ((k+2) * inChannelGroup + c) ;
    const int64_t kernelIndex3 = kernGroupOffset + ((k+3) * inChannelGroup + c) ;
    const int64_t kernelIndex4 = kernGroupOffset + ((k+4) * inChannelGroup + c) ;
    const int64_t kernelIndex5 = kernGroupOffset + ((k+5) * inChannelGroup + c) ;
    const int64_t kernelIndex6 = kernGroupOffset + ((k+6) * inChannelGroup + c) ;
    const int64_t kernelIndex7 = kernGroupOffset + ((k+7) * inChannelGroup + c) ;

    INIT_VRSUM(c0)
    INIT_VRSUM(c1)

    for (int64_t n=0; n<batch; n++) {
      for (int64_t gop = 0; gop < gOutPixels; gop+=VLEN) {
	const int64_t vl = gOutPixels - gop < VLEN ? gOutPixels - gop : VLEN ;

	const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;

	const int64_t gOutIndex = outGroupOffset + ((n * gOutChannel + k  ) * gOutHeight ) * gOutWidth + gop;

	/* memory access errors mihgt be caused (vrin) */
	__vr vrin_c0    = _vel_vldu_vssl(4,&pInChannel[gop], vl) ;
	__vr vrin_c1    = _vel_vldu_vssl(4,&pInChannel[gop+1*inHeight*inWidth], vl) ;

	__vr vrgout0 = _vel_vldu_vssl(4, pGOut+gOutIndex+0*gOutPixels, vl) ;
	__vr vrgout1 = _vel_vldu_vssl(4, pGOut+gOutIndex+1*gOutPixels, vl) ;
	__vr vrgout2 = _vel_vldu_vssl(4, pGOut+gOutIndex+2*gOutPixels, vl) ;
	__vr vrgout3 = _vel_vldu_vssl(4, pGOut+gOutIndex+3*gOutPixels, vl) ;
	__vr vrgout4 = _vel_vldu_vssl(4, pGOut+gOutIndex+4*gOutPixels, vl) ;
	__vr vrgout5 = _vel_vldu_vssl(4, pGOut+gOutIndex+5*gOutPixels, vl) ;
	__vr vrgout6 = _vel_vldu_vssl(4, pGOut+gOutIndex+6*gOutPixels, vl) ;
	__vr vrgout7 = _vel_vldu_vssl(4, pGOut+gOutIndex+7*gOutPixels, vl) ;

	__vr vrgout01 = _vel_vshf_vvvsl(vrgout0, vrgout1, VE_VSHUFFLE_YUZU, vl) ;
	__vr vrgout23 = _vel_vshf_vvvsl(vrgout2, vrgout3, VE_VSHUFFLE_YUZU, vl) ;
	__vr vrgout45 = _vel_vshf_vvvsl(vrgout4, vrgout5, VE_VSHUFFLE_YUZU, vl) ;
	__vr vrgout67 = _vel_vshf_vvvsl(vrgout6, vrgout7, VE_VSHUFFLE_YUZU, vl) ;

	VFMAD_R1S1(c0)
	VFMAD_R1S1(c1)

      } // gOutPixels
    } // batch

    VFSUM_STORE_R1S1(0,c0)
    VFSUM_STORE_R1S1(1,c1)

    c+=2;
  }
  if ( ((inChannelGroup >> 2) & 0x01) == 1 ) {
    const int64_t kernelIndex0 = kernGroupOffset + (k     * inChannelGroup + c) ;
    const int64_t kernelIndex1 = kernGroupOffset + ((k+1) * inChannelGroup + c) ;
    const int64_t kernelIndex2 = kernGroupOffset + ((k+2) * inChannelGroup + c) ;
    const int64_t kernelIndex3 = kernGroupOffset + ((k+3) * inChannelGroup + c) ;
    const int64_t kernelIndex4 = kernGroupOffset + ((k+4) * inChannelGroup + c) ;
    const int64_t kernelIndex5 = kernGroupOffset + ((k+5) * inChannelGroup + c) ;
    const int64_t kernelIndex6 = kernGroupOffset + ((k+6) * inChannelGroup + c) ;
    const int64_t kernelIndex7 = kernGroupOffset + ((k+7) * inChannelGroup + c) ;

    INIT_VRSUM(c0)
    INIT_VRSUM(c1)
    INIT_VRSUM(c2)
    INIT_VRSUM(c3)

    for (int64_t n=0; n<batch; n++) {
      for (int64_t gop = 0; gop < gOutPixels; gop+=VLEN) {
	const int64_t vl = gOutPixels - gop < VLEN ? gOutPixels - gop : VLEN ;

	const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;

	const int64_t gOutIndex = outGroupOffset + ((n * gOutChannel + k  ) * gOutHeight ) * gOutWidth + gop;

	/* memory access errors mihgt be caused (vrin) */
	__vr vrin_c0    = _vel_vldu_vssl(4,&pInChannel[gop], vl) ;
	__vr vrin_c1    = _vel_vldu_vssl(4,&pInChannel[gop+1*inHeight*inWidth], vl) ;
	__vr vrin_c2    = _vel_vldu_vssl(4,&pInChannel[gop+2*inHeight*inWidth], vl) ;
	__vr vrin_c3    = _vel_vldu_vssl(4,&pInChannel[gop+3*inHeight*inWidth], vl) ;

	__vr vrgout0 = _vel_vldu_vssl(4, pGOut+gOutIndex+0*gOutPixels, vl) ;
	__vr vrgout1 = _vel_vldu_vssl(4, pGOut+gOutIndex+1*gOutPixels, vl) ;
	__vr vrgout2 = _vel_vldu_vssl(4, pGOut+gOutIndex+2*gOutPixels, vl) ;
	__vr vrgout3 = _vel_vldu_vssl(4, pGOut+gOutIndex+3*gOutPixels, vl) ;
	__vr vrgout4 = _vel_vldu_vssl(4, pGOut+gOutIndex+4*gOutPixels, vl) ;
	__vr vrgout5 = _vel_vldu_vssl(4, pGOut+gOutIndex+5*gOutPixels, vl) ;
	__vr vrgout6 = _vel_vldu_vssl(4, pGOut+gOutIndex+6*gOutPixels, vl) ;
	__vr vrgout7 = _vel_vldu_vssl(4, pGOut+gOutIndex+7*gOutPixels, vl) ;

	__vr vrgout01 = _vel_vshf_vvvsl(vrgout0, vrgout1, VE_VSHUFFLE_YUZU, vl) ;
	__vr vrgout23 = _vel_vshf_vvvsl(vrgout2, vrgout3, VE_VSHUFFLE_YUZU, vl) ;
	__vr vrgout45 = _vel_vshf_vvvsl(vrgout4, vrgout5, VE_VSHUFFLE_YUZU, vl) ;
	__vr vrgout67 = _vel_vshf_vvvsl(vrgout6, vrgout7, VE_VSHUFFLE_YUZU, vl) ;

	VFMAD_R1S1(c0)
	VFMAD_R1S1(c1)
	VFMAD_R1S1(c2)
	VFMAD_R1S1(c3)

      } // gOutPixels
    } // batch

    VFSUM_STORE_R1S1(0,c0)
    VFSUM_STORE_R1S1(1,c1)
    VFSUM_STORE_R1S1(2,c2)
    VFSUM_STORE_R1S1(3,c3)

    c+=4 ;
  }
  for (; c<inChannelGroup; c+=8) {
    const int64_t kernelIndex0 = kernGroupOffset + (k     * inChannelGroup + c) ;
    const int64_t kernelIndex1 = kernGroupOffset + ((k+1) * inChannelGroup + c) ;
    const int64_t kernelIndex2 = kernGroupOffset + ((k+2) * inChannelGroup + c) ;
    const int64_t kernelIndex3 = kernGroupOffset + ((k+3) * inChannelGroup + c) ;
    const int64_t kernelIndex4 = kernGroupOffset + ((k+4) * inChannelGroup + c) ;
    const int64_t kernelIndex5 = kernGroupOffset + ((k+5) * inChannelGroup + c) ;
    const int64_t kernelIndex6 = kernGroupOffset + ((k+6) * inChannelGroup + c) ;
    const int64_t kernelIndex7 = kernGroupOffset + ((k+7) * inChannelGroup + c) ;

    INIT_VRSUM(c0)
    INIT_VRSUM(c1)
    INIT_VRSUM(c2)
    INIT_VRSUM(c3)
    INIT_VRSUM(c4)
    INIT_VRSUM(c5)
    INIT_VRSUM(c6)
    INIT_VRSUM(c7)
#undef INIT_VRSUM

    for (int64_t n=0; n<batch; n++) {
      for (int64_t gop = 0; gop < gOutPixels; gop+=VLEN) {
	const int64_t vl = gOutPixels - gop < VLEN ? gOutPixels - gop : VLEN ;

	const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;

	const int64_t gOutIndex = outGroupOffset + ((n * gOutChannel + k  ) * gOutHeight ) * gOutWidth + gop;

	/* memory access errors mihgt be caused (vrin) */
	__vr vrin_c0    = _vel_vldu_vssl(4,&pInChannel[gop], vl) ;
	__vr vrin_c1    = _vel_vldu_vssl(4,&pInChannel[gop+1*inHeight*inWidth], vl) ;
	__vr vrin_c2    = _vel_vldu_vssl(4,&pInChannel[gop+2*inHeight*inWidth], vl) ;
	__vr vrin_c3    = _vel_vldu_vssl(4,&pInChannel[gop+3*inHeight*inWidth], vl) ;
	__vr vrin_c4    = _vel_vldu_vssl(4,&pInChannel[gop+4*inHeight*inWidth], vl) ;
	__vr vrin_c5    = _vel_vldu_vssl(4,&pInChannel[gop+5*inHeight*inWidth], vl) ;
	__vr vrin_c6    = _vel_vldu_vssl(4,&pInChannel[gop+6*inHeight*inWidth], vl) ;
	__vr vrin_c7    = _vel_vldu_vssl(4,&pInChannel[gop+7*inHeight*inWidth], vl) ;

	__vr vrgout0 = _vel_vldu_vssl(4, pGOut+gOutIndex+0*gOutPixels, vl) ;
	__vr vrgout1 = _vel_vldu_vssl(4, pGOut+gOutIndex+1*gOutPixels, vl) ;
	__vr vrgout2 = _vel_vldu_vssl(4, pGOut+gOutIndex+2*gOutPixels, vl) ;
	__vr vrgout3 = _vel_vldu_vssl(4, pGOut+gOutIndex+3*gOutPixels, vl) ;
	__vr vrgout4 = _vel_vldu_vssl(4, pGOut+gOutIndex+4*gOutPixels, vl) ;
	__vr vrgout5 = _vel_vldu_vssl(4, pGOut+gOutIndex+5*gOutPixels, vl) ;
	__vr vrgout6 = _vel_vldu_vssl(4, pGOut+gOutIndex+6*gOutPixels, vl) ;
	__vr vrgout7 = _vel_vldu_vssl(4, pGOut+gOutIndex+7*gOutPixels, vl) ;

	__vr vrgout01 = _vel_vshf_vvvsl(vrgout0, vrgout1, VE_VSHUFFLE_YUZU, vl) ;
	__vr vrgout23 = _vel_vshf_vvvsl(vrgout2, vrgout3, VE_VSHUFFLE_YUZU, vl) ;
	__vr vrgout45 = _vel_vshf_vvvsl(vrgout4, vrgout5, VE_VSHUFFLE_YUZU, vl) ;
	__vr vrgout67 = _vel_vshf_vvvsl(vrgout6, vrgout7, VE_VSHUFFLE_YUZU, vl) ;

	VFMAD_R1S1(c0)
	VFMAD_R1S1(c1)
	VFMAD_R1S1(c2)
	VFMAD_R1S1(c3)
	VFMAD_R1S1(c4)
	VFMAD_R1S1(c5)
	VFMAD_R1S1(c6)
	VFMAD_R1S1(c7)
#undef VFMAD_R1S1
      } // gOutPixels
    } // batch

    VFSUM_STORE_R1S1(0,c0)
    VFSUM_STORE_R1S1(1,c1)
    VFSUM_STORE_R1S1(2,c2)
    VFSUM_STORE_R1S1(3,c3)
    VFSUM_STORE_R1S1(4,c4)
    VFSUM_STORE_R1S1(5,c5)
    VFSUM_STORE_R1S1(6,c6)
    VFSUM_STORE_R1S1(7,c7)
#undef VFSUM_STORE_R1S1

  } // inChannel
}

vednnError_t
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
//  const int64_t gKernWidth  = pParamGradKernel->width;		/* must be 1 */
//  const int64_t gKernHeight = pParamGradKernel->height;		/* must be 1 */

  const int64_t group          = pParamConv->group;
//  const int64_t strideWidth    = pParamConv->strideWidth;	/* must be 1 */
//  const int64_t strideHeight   = pParamConv->strideHeight;	/* must be 1 */
//  const int64_t padWidth       = pParamConv->padWidth;	/* must be 1 */
//  const int64_t padHeight      = pParamConv->padHeight;	/* must be 1 */
//  const int64_t dilationWidth  = pParamConv->dilationWidth;	/* must be 1 */
//  const int64_t dilationHeight = pParamConv->dilationHeight;	/* must be 1 */

  const int64_t inChannelGroup   =  inChannel   / group;
  const int64_t gOutChannelGroup = gOutChannel  / group;

  const float * restrict pIn      = pDataIn;
  const float * restrict pGOut    = pDataGradOut;
  float * restrict const pGKernel = pDataGradKernel;

  const int gOutPixels= gOutHeight*gOutWidth ;

#ifndef VEDNN_USE_OPENMP
  const int64_t beginOChannel = 0 ;
  const int64_t nOChannel     = gOutChannelGroup ;
#endif
  {
    {
      for (int64_t g = 0; g < group; g++) {
	int64_t inGroupOffset   = g * inChannelGroup  * inHeight  * inWidth;
	int64_t outGroupOffset  = (g * gOutChannelGroup + beginOChannel) * gOutHeight * gOutWidth;
	int64_t kernGroupOffset = (g * gOutChannelGroup + beginOChannel) * inChannelGroup ;

	int64_t k = 0 ;
	if ( (nOChannel & 0x01) == 1 ) {
	  k1(pIn, pGOut, pGKernel,
	     inChannel, inWidth, inHeight,
	     gOutChannel, gOutWidth, gOutHeight,
	     inChannelGroup,
	     inGroupOffset, outGroupOffset, kernGroupOffset,
	     gOutPixels, batch, k ) ;
	  k+=1;
	}
	if ( ((nOChannel >> 1) & 0x01) == 1 ) {
	  k2(pIn, pGOut, pGKernel,
	     inChannel, inWidth, inHeight,
	     gOutChannel, gOutWidth, gOutHeight,
	     inChannelGroup,
	     inGroupOffset, outGroupOffset, kernGroupOffset,
	     gOutPixels, batch, k ) ;
	  k+=2;
	}
	if ( ((nOChannel >> 2) & 0x01) == 1 ) {
	  k4(pIn, pGOut, pGKernel,
	     inChannel, inWidth, inHeight,
	     gOutChannel, gOutWidth, gOutHeight,
	     inChannelGroup,
	     inGroupOffset, outGroupOffset, kernGroupOffset,
	     gOutPixels, batch, k ) ;
	  k+=4;
	}
	for ( ;k<nOChannel; k+=8) {
	  k8(pIn, pGOut, pGKernel,
	     inChannel, inWidth, inHeight,
	     gOutChannel, gOutWidth, gOutHeight,
	     inChannelGroup,
	     inGroupOffset, outGroupOffset, kernGroupOffset,
	     gOutPixels, batch, k ) ;
	} // outChannel
      } // group
    }
  }


  return VEDNN_SUCCESS;
}
