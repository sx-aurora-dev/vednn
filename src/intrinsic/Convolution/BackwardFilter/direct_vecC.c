#include <stdint.h>
#include <stdlib.h>

#include "vednn.h"

#include "velintrin.h"
#define VLEN	(256)

static inline void k1(
  const float * pIn, const int64_t inWidth, const int64_t inHeight,
  const float * pGOut, const int64_t gOutWidth, const int64_t gOutHeight,
  float * const pGKernel, const int64_t gKernWidth , const int64_t gKernHeight,
  const int64_t strideHeight, const int64_t strideWidth,
  const int64_t padHeight, const int64_t padWidth,
  const int64_t dilationHeight, const int64_t dilationWidth,
  const int64_t inChannelGroup, const int64_t inChannel, const int64_t gOutChannel,
  const int64_t inGroupOffset, const int64_t outGroupOffset, const int64_t kernGroupOffset,
  const int64_t batch,
  const int64_t k
)
{
  for (int64_t r=0; r<gKernHeight; r++) {
    for (int64_t s=0; s<gKernWidth; s++) {
      for (int64_t c=0; c<inChannelGroup; c+=VLEN) {
	int64_t kernelIndex = kernGroupOffset + ((k * inChannelGroup + c) * gKernHeight + r) * gKernWidth + s;

	const int64_t vl = inChannelGroup - c < VLEN ? inChannelGroup - c : VLEN ;

	__vr vrsum = _vel_vbrds_vsl(0.0f, vl) ;

	for (int64_t y=0; y<gOutHeight; y++) {
	  for (int64_t x=0; x<gOutWidth; x++) {
	    int64_t h = y * strideHeight - padHeight + r * dilationHeight;
	    int64_t w = x * strideWidth - padWidth + s * dilationWidth;
	    if (h < 0 || inHeight <= h) {
	      continue;
	    }
	    if (w < 0 || inWidth <= w) {
	      continue;
	    }
	    for (int64_t n=0; n<batch; n++) {
	      int64_t inputIndex  = inGroupOffset + ((n * inChannel + c) * inHeight + h) * inWidth + w;
	      int64_t outIndex  = outGroupOffset + ((n * gOutChannel + k) * gOutHeight + y) * gOutWidth + x;

	      __vr vri = _vel_vldu_vssl(4*inHeight*inWidth, &pIn[inputIndex], vl) ;
	      vrsum  = _vel_vfmads_vvsvl(vrsum, pGOut[outIndex], vri, vl);

	    } // batch
	  } // outHeight
	} // outChannel
	_vel_vstu_vssl(vrsum, 4*gKernHeight*gKernWidth, &pGKernel[kernelIndex], vl) ;
      } // inChannel
    } // outWidth
  } // kernWidth
}

static inline void k2(
  const float * pIn, const int64_t inWidth, const int64_t inHeight,
  const float * pGOut, const int64_t gOutWidth, const int64_t gOutHeight,
  float * const pGKernel, const int64_t gKernWidth , const int64_t gKernHeight,
  const int64_t strideHeight, const int64_t strideWidth,
  const int64_t padHeight, const int64_t padWidth,
  const int64_t dilationHeight, const int64_t dilationWidth,
  const int64_t inChannelGroup, const int64_t inChannel, const int64_t gOutChannel,
  const int64_t inGroupOffset, const int64_t outGroupOffset, const int64_t kernGroupOffset,
  const int64_t batch,
  const int64_t k
)
{
  for (int64_t r=0; r<gKernHeight; r++) {
    for (int64_t s=0; s<gKernWidth; s++) {
      for (int64_t c=0; c<inChannelGroup; c+=VLEN) {
	int64_t kernelIndex0 = kernGroupOffset + (((k  ) * inChannelGroup + c) * gKernHeight + r) * gKernWidth + s;
	int64_t kernelIndex1 = kernGroupOffset + (((k+1) * inChannelGroup + c) * gKernHeight + r) * gKernWidth + s;

	const int64_t vl = inChannelGroup - c < VLEN ? inChannelGroup - c : VLEN ;

	__vr vrsum01 = _vel_vbrdl_vsl(0UL, vl) ;

	for (int64_t y=0; y<gOutHeight; y++) {
	  for (int64_t x=0; x<gOutWidth; x++) {
	    int64_t h = y * strideHeight - padHeight + r * dilationHeight;
	    int64_t w = x * strideWidth - padWidth + s * dilationWidth;
	    if (h < 0 || inHeight <= h) {
	      continue;
	    }
	    if (w < 0 || inWidth <= w) {
	      continue;
	    }
	    for (int64_t n=0; n<batch; n++) {
	      int64_t inputIndex  = inGroupOffset + ((n * inChannel + c) * inHeight + h) * inWidth + w;
	      int64_t outIndex  = outGroupOffset + ((n * gOutChannel + k) * gOutHeight + y) * gOutWidth + x;

	      __vr vri  = _vel_vldu_vssl(4*inHeight*inWidth, &pIn[inputIndex], vl) ;
	      __vr vriP = _vel_vshf_vvvsl(vri, vri, VE_VSHUFFLE_YUZU, vl) ;

	      const uint64_t go01 =  _vel_pack_f32p(pGOut+outIndex+0*gOutHeight*gOutWidth, pGOut+outIndex+1*gOutHeight*gOutWidth) ;
	      vrsum01  = _vel_pvfmad_vvsvl(vrsum01, go01, vriP, vl);

	    } // batch
	  } // outHeight
	} // outChannel
	_vel_vstu_vssl(vrsum01, 4*gKernHeight*gKernWidth, &pGKernel[kernelIndex0], vl);
	_vel_vstl_vssl(vrsum01, 4*gKernHeight*gKernWidth, &pGKernel[kernelIndex1], vl);
      } // inChannel
    } // outWidth
  } // kernWidth
}

static inline void k4(
  const float * pIn, const int64_t inWidth, const int64_t inHeight,
  const float * pGOut, const int64_t gOutWidth, const int64_t gOutHeight,
  float * const pGKernel, const int64_t gKernWidth , const int64_t gKernHeight,
  const int64_t strideHeight, const int64_t strideWidth,
  const int64_t padHeight, const int64_t padWidth,
  const int64_t dilationHeight, const int64_t dilationWidth,
  const int64_t inChannelGroup, const int64_t inChannel, const int64_t gOutChannel,
  const int64_t inGroupOffset, const int64_t outGroupOffset, const int64_t kernGroupOffset,
  const int64_t batch,
  const int64_t k
)
{
  for (int64_t r=0; r<gKernHeight; r++) {
    for (int64_t s=0; s<gKernWidth; s++) {
      for (int64_t c=0; c<inChannelGroup; c+=VLEN) {
	int64_t kernelIndex0 = kernGroupOffset + (((k  ) * inChannelGroup + c) * gKernHeight + r) * gKernWidth + s;
	int64_t kernelIndex1 = kernGroupOffset + (((k+1) * inChannelGroup + c) * gKernHeight + r) * gKernWidth + s;
	int64_t kernelIndex2 = kernGroupOffset + (((k+2) * inChannelGroup + c) * gKernHeight + r) * gKernWidth + s;
	int64_t kernelIndex3 = kernGroupOffset + (((k+3) * inChannelGroup + c) * gKernHeight + r) * gKernWidth + s;

	const int64_t vl = inChannelGroup - c < VLEN ? inChannelGroup - c : VLEN ;

	__vr vrsum01 = _vel_vbrdl_vsl(0UL, vl) ;
	__vr vrsum23 = _vel_vbrdl_vsl(0UL, vl) ;

	for (int64_t y=0; y<gOutHeight; y++) {
	  for (int64_t x=0; x<gOutWidth; x++) {
	    int64_t h = y * strideHeight - padHeight + r * dilationHeight;
	    int64_t w = x * strideWidth - padWidth + s * dilationWidth;
	    if (h < 0 || inHeight <= h) {
	      continue;
	    }
	    if (w < 0 || inWidth <= w) {
	      continue;
	    }
	    for (int64_t n=0; n<batch; n++) {
	      int64_t inputIndex  = inGroupOffset + ((n * inChannel + c) * inHeight + h) * inWidth + w;
	      int64_t outIndex  = outGroupOffset + ((n * gOutChannel + k) * gOutHeight + y) * gOutWidth + x;

	      __vr vri  = _vel_vldu_vssl(4*inHeight*inWidth, &pIn[inputIndex], vl) ;
	      __vr vriP = _vel_vshf_vvvsl(vri, vri, VE_VSHUFFLE_YUZU, vl) ;

	      const uint64_t go01 =  _vel_pack_f32p(pGOut+outIndex+0*gOutHeight*gOutWidth, pGOut+outIndex+1*gOutHeight*gOutWidth) ;
	      const uint64_t go23 =  _vel_pack_f32p(pGOut+outIndex+2*gOutHeight*gOutWidth, pGOut+outIndex+3*gOutHeight*gOutWidth) ;
	      vrsum01  = _vel_pvfmad_vvsvl(vrsum01, go01, vriP, vl);
	      vrsum23  = _vel_pvfmad_vvsvl(vrsum23, go23, vriP, vl);

	    } // batch
	  } // outHeight
	} // outChannel
	_vel_vstu_vssl(vrsum01, 4*gKernHeight*gKernWidth, &pGKernel[kernelIndex0], vl);
	_vel_vstl_vssl(vrsum01, 4*gKernHeight*gKernWidth, &pGKernel[kernelIndex1], vl);
	_vel_vstu_vssl(vrsum23, 4*gKernHeight*gKernWidth, &pGKernel[kernelIndex2], vl);
	_vel_vstl_vssl(vrsum23, 4*gKernHeight*gKernWidth, &pGKernel[kernelIndex3], vl);

      } // inChannel
    } // outWidth
  } // kernWidth
}

static inline void k8(
  const float * pIn, const int64_t inWidth, const int64_t inHeight,
  const float * pGOut, const int64_t gOutWidth, const int64_t gOutHeight,
  float * const pGKernel, const int64_t gKernWidth , const int64_t gKernHeight,
  const int64_t strideHeight, const int64_t strideWidth,
  const int64_t padHeight, const int64_t padWidth,
  const int64_t dilationHeight, const int64_t dilationWidth,
  const int64_t inChannelGroup, const int64_t inChannel, const int64_t gOutChannel,
  const int64_t inGroupOffset, const int64_t outGroupOffset, const int64_t kernGroupOffset,
  const int64_t batch,
  const int64_t k
)
{
  for (int64_t r=0; r<gKernHeight; r++) {
    for (int64_t s=0; s<gKernWidth; s++) {
      for (int64_t c=0; c<inChannelGroup; c+=VLEN) {
	int64_t kernelIndex0 = kernGroupOffset + (((k  ) * inChannelGroup + c) * gKernHeight + r) * gKernWidth + s;
	int64_t kernelIndex1 = kernGroupOffset + (((k+1) * inChannelGroup + c) * gKernHeight + r) * gKernWidth + s;
	int64_t kernelIndex2 = kernGroupOffset + (((k+2) * inChannelGroup + c) * gKernHeight + r) * gKernWidth + s;
	int64_t kernelIndex3 = kernGroupOffset + (((k+3) * inChannelGroup + c) * gKernHeight + r) * gKernWidth + s;
	int64_t kernelIndex4 = kernGroupOffset + (((k+4) * inChannelGroup + c) * gKernHeight + r) * gKernWidth + s;
	int64_t kernelIndex5 = kernGroupOffset + (((k+5) * inChannelGroup + c) * gKernHeight + r) * gKernWidth + s;
	int64_t kernelIndex6 = kernGroupOffset + (((k+6) * inChannelGroup + c) * gKernHeight + r) * gKernWidth + s;
	int64_t kernelIndex7 = kernGroupOffset + (((k+7) * inChannelGroup + c) * gKernHeight + r) * gKernWidth + s;

	const int64_t vl = inChannelGroup - c < VLEN ? inChannelGroup - c : VLEN ;

	__vr vrsum01 = _vel_vbrdl_vsl(0UL, vl) ;
	__vr vrsum23 = _vel_vbrdl_vsl(0UL, vl) ;
	__vr vrsum45 = _vel_vbrdl_vsl(0UL, vl) ;
	__vr vrsum67 = _vel_vbrdl_vsl(0UL, vl) ;

	for (int64_t y=0; y<gOutHeight; y++) {
	  for (int64_t x=0; x<gOutWidth; x++) {
	    int64_t h = y * strideHeight - padHeight + r * dilationHeight;
	    int64_t w = x * strideWidth - padWidth + s * dilationWidth;
	    if (h < 0 || inHeight <= h) {
	      continue;
	    }
	    if (w < 0 || inWidth <= w) {
	      continue;
	    }
	    for (int64_t n=0; n<batch; n++) {
	      int64_t inputIndex  = inGroupOffset + ((n * inChannel + c) * inHeight + h) * inWidth + w;
	      int64_t outIndex  = outGroupOffset + ((n * gOutChannel + k) * gOutHeight + y) * gOutWidth + x;

	      __vr vri  = _vel_vldu_vssl(4*inHeight*inWidth, &pIn[inputIndex], vl) ;
	      __vr vriP = _vel_vshf_vvvsl(vri, vri, VE_VSHUFFLE_YUZU, vl) ;

	      const uint64_t go01 =  _vel_pack_f32p(pGOut+outIndex+0*gOutHeight*gOutWidth, pGOut+outIndex+1*gOutHeight*gOutWidth) ;
	      const uint64_t go23 =  _vel_pack_f32p(pGOut+outIndex+2*gOutHeight*gOutWidth, pGOut+outIndex+3*gOutHeight*gOutWidth) ;
	      const uint64_t go45 =  _vel_pack_f32p(pGOut+outIndex+4*gOutHeight*gOutWidth, pGOut+outIndex+5*gOutHeight*gOutWidth) ;
	      const uint64_t go67 =  _vel_pack_f32p(pGOut+outIndex+6*gOutHeight*gOutWidth, pGOut+outIndex+7*gOutHeight*gOutWidth) ;
	      vrsum01  = _vel_pvfmad_vvsvl(vrsum01, go01, vriP, vl);
	      vrsum23  = _vel_pvfmad_vvsvl(vrsum23, go23, vriP, vl);
	      vrsum45  = _vel_pvfmad_vvsvl(vrsum45, go45, vriP, vl);
	      vrsum67  = _vel_pvfmad_vvsvl(vrsum67, go67, vriP, vl);

	    } // batch
	  } // outHeight
	} // outChannel
	_vel_vstu_vssl(vrsum01, 4*gKernHeight*gKernWidth, &pGKernel[kernelIndex0], vl);
	_vel_vstl_vssl(vrsum01, 4*gKernHeight*gKernWidth, &pGKernel[kernelIndex1], vl);
	_vel_vstu_vssl(vrsum23, 4*gKernHeight*gKernWidth, &pGKernel[kernelIndex2], vl);
	_vel_vstl_vssl(vrsum23, 4*gKernHeight*gKernWidth, &pGKernel[kernelIndex3], vl);
	_vel_vstu_vssl(vrsum45, 4*gKernHeight*gKernWidth, &pGKernel[kernelIndex4], vl);
	_vel_vstl_vssl(vrsum45, 4*gKernHeight*gKernWidth, &pGKernel[kernelIndex5], vl);
	_vel_vstu_vssl(vrsum67, 4*gKernHeight*gKernWidth, &pGKernel[kernelIndex6], vl);
	_vel_vstl_vssl(vrsum67, 4*gKernHeight*gKernWidth, &pGKernel[kernelIndex7], vl);


      } // inChannel
    } // outWidth
  } // kernWidth
}

static inline void k16(
  const float * pIn, const int64_t inWidth, const int64_t inHeight,
  const float * pGOut, const int64_t gOutWidth, const int64_t gOutHeight,
  float * const pGKernel, const int64_t gKernWidth , const int64_t gKernHeight,
  const int64_t strideHeight, const int64_t strideWidth,
  const int64_t padHeight, const int64_t padWidth,
  const int64_t dilationHeight, const int64_t dilationWidth,
  const int64_t inChannelGroup, const int64_t inChannel, const int64_t gOutChannel,
  const int64_t inGroupOffset, const int64_t outGroupOffset, const int64_t kernGroupOffset,
  const int64_t batch,
  const int64_t k
)
{
  for (int64_t r=0; r<gKernHeight; r++) {
    for (int64_t s=0; s<gKernWidth; s++) {
      for (int64_t c=0; c<inChannelGroup; c+=VLEN) {
	int64_t kernelIndex0 = kernGroupOffset + (((k  ) * inChannelGroup + c) * gKernHeight + r) * gKernWidth + s;
	int64_t kernelIndex1 = kernGroupOffset + (((k+1) * inChannelGroup + c) * gKernHeight + r) * gKernWidth + s;
	int64_t kernelIndex2 = kernGroupOffset + (((k+2) * inChannelGroup + c) * gKernHeight + r) * gKernWidth + s;
	int64_t kernelIndex3 = kernGroupOffset + (((k+3) * inChannelGroup + c) * gKernHeight + r) * gKernWidth + s;
	int64_t kernelIndex4 = kernGroupOffset + (((k+4) * inChannelGroup + c) * gKernHeight + r) * gKernWidth + s;
	int64_t kernelIndex5 = kernGroupOffset + (((k+5) * inChannelGroup + c) * gKernHeight + r) * gKernWidth + s;
	int64_t kernelIndex6 = kernGroupOffset + (((k+6) * inChannelGroup + c) * gKernHeight + r) * gKernWidth + s;
	int64_t kernelIndex7 = kernGroupOffset + (((k+7) * inChannelGroup + c) * gKernHeight + r) * gKernWidth + s;
	int64_t kernelIndex8 = kernGroupOffset + (((k+8) * inChannelGroup + c) * gKernHeight + r) * gKernWidth + s;
	int64_t kernelIndex9 = kernGroupOffset + (((k+9) * inChannelGroup + c) * gKernHeight + r) * gKernWidth + s;
	int64_t kernelIndexA = kernGroupOffset + (((k+10) * inChannelGroup + c) * gKernHeight + r) * gKernWidth + s;
	int64_t kernelIndexB = kernGroupOffset + (((k+11) * inChannelGroup + c) * gKernHeight + r) * gKernWidth + s;
	int64_t kernelIndexC = kernGroupOffset + (((k+12) * inChannelGroup + c) * gKernHeight + r) * gKernWidth + s;
	int64_t kernelIndexD = kernGroupOffset + (((k+13) * inChannelGroup + c) * gKernHeight + r) * gKernWidth + s;
	int64_t kernelIndexE = kernGroupOffset + (((k+14) * inChannelGroup + c) * gKernHeight + r) * gKernWidth + s;
	int64_t kernelIndexF = kernGroupOffset + (((k+15) * inChannelGroup + c) * gKernHeight + r) * gKernWidth + s;

	const int64_t vl = inChannelGroup - c < VLEN ? inChannelGroup - c : VLEN ;

	__vr vrsum01 = _vel_vbrdl_vsl(0UL, vl) ;
	__vr vrsum23 = _vel_vbrdl_vsl(0UL, vl) ;
	__vr vrsum45 = _vel_vbrdl_vsl(0UL, vl) ;
	__vr vrsum67 = _vel_vbrdl_vsl(0UL, vl) ;
	__vr vrsum89 = _vel_vbrdl_vsl(0UL, vl) ;
	__vr vrsumAB = _vel_vbrdl_vsl(0UL, vl) ;
	__vr vrsumCD = _vel_vbrdl_vsl(0UL, vl) ;
	__vr vrsumEF = _vel_vbrdl_vsl(0UL, vl) ;

	for (int64_t y=0; y<gOutHeight; y++) {
	  for (int64_t x=0; x<gOutWidth; x++) {
	    int64_t h = y * strideHeight - padHeight + r * dilationHeight;
	    int64_t w = x * strideWidth - padWidth + s * dilationWidth;
	    if (h < 0 || inHeight <= h) {
	      continue;
	    }
	    if (w < 0 || inWidth <= w) {
	      continue;
	    }
	    for (int64_t n=0; n<batch; n++) {
	      int64_t inputIndex  = inGroupOffset + ((n * inChannel + c) * inHeight + h) * inWidth + w;
	      int64_t outIndex  = outGroupOffset + ((n * gOutChannel + k) * gOutHeight + y) * gOutWidth + x;

	      __vr vri  = _vel_vldu_vssl(4*inHeight*inWidth, &pIn[inputIndex], vl) ;
	      __vr vriP = _vel_vshf_vvvsl(vri, vri, VE_VSHUFFLE_YUZU, vl) ;

	      const uint64_t go01 =  _vel_pack_f32p(pGOut+outIndex+0*gOutHeight*gOutWidth, pGOut+outIndex+1*gOutHeight*gOutWidth) ;
	      const uint64_t go23 =  _vel_pack_f32p(pGOut+outIndex+2*gOutHeight*gOutWidth, pGOut+outIndex+3*gOutHeight*gOutWidth) ;
	      const uint64_t go45 =  _vel_pack_f32p(pGOut+outIndex+4*gOutHeight*gOutWidth, pGOut+outIndex+5*gOutHeight*gOutWidth) ;
	      const uint64_t go67 =  _vel_pack_f32p(pGOut+outIndex+6*gOutHeight*gOutWidth, pGOut+outIndex+7*gOutHeight*gOutWidth) ;
	      const uint64_t go89 =  _vel_pack_f32p(pGOut+outIndex+8*gOutHeight*gOutWidth, pGOut+outIndex+9*gOutHeight*gOutWidth) ;
	      const uint64_t goAB =  _vel_pack_f32p(pGOut+outIndex+10*gOutHeight*gOutWidth, pGOut+outIndex+11*gOutHeight*gOutWidth) ;
	      const uint64_t goCD =  _vel_pack_f32p(pGOut+outIndex+12*gOutHeight*gOutWidth, pGOut+outIndex+13*gOutHeight*gOutWidth) ;
	      const uint64_t goEF =  _vel_pack_f32p(pGOut+outIndex+14*gOutHeight*gOutWidth, pGOut+outIndex+15*gOutHeight*gOutWidth) ;
	      vrsum01  = _vel_pvfmad_vvsvl(vrsum01, go01, vriP, vl);
	      vrsum23  = _vel_pvfmad_vvsvl(vrsum23, go23, vriP, vl);
	      vrsum45  = _vel_pvfmad_vvsvl(vrsum45, go45, vriP, vl);
	      vrsum67  = _vel_pvfmad_vvsvl(vrsum67, go67, vriP, vl);
	      vrsum89  = _vel_pvfmad_vvsvl(vrsum89, go89, vriP, vl);
	      vrsumAB  = _vel_pvfmad_vvsvl(vrsumAB, goAB, vriP, vl);
	      vrsumCD  = _vel_pvfmad_vvsvl(vrsumCD, goCD, vriP, vl);
	      vrsumEF  = _vel_pvfmad_vvsvl(vrsumEF, goEF, vriP, vl);

	    } // batch
	  } // outHeight
	} // outChannel
	_vel_vstu_vssl(vrsum01, 4*gKernHeight*gKernWidth, &pGKernel[kernelIndex0], vl);
	_vel_vstl_vssl(vrsum01, 4*gKernHeight*gKernWidth, &pGKernel[kernelIndex1], vl);
	_vel_vstu_vssl(vrsum23, 4*gKernHeight*gKernWidth, &pGKernel[kernelIndex2], vl);
	_vel_vstl_vssl(vrsum23, 4*gKernHeight*gKernWidth, &pGKernel[kernelIndex3], vl);
	_vel_vstu_vssl(vrsum45, 4*gKernHeight*gKernWidth, &pGKernel[kernelIndex4], vl);
	_vel_vstl_vssl(vrsum45, 4*gKernHeight*gKernWidth, &pGKernel[kernelIndex5], vl);
	_vel_vstu_vssl(vrsum67, 4*gKernHeight*gKernWidth, &pGKernel[kernelIndex6], vl);
	_vel_vstl_vssl(vrsum67, 4*gKernHeight*gKernWidth, &pGKernel[kernelIndex7], vl);
	_vel_vstu_vssl(vrsum89, 4*gKernHeight*gKernWidth, &pGKernel[kernelIndex8], vl);
	_vel_vstl_vssl(vrsum89, 4*gKernHeight*gKernWidth, &pGKernel[kernelIndex9], vl);
	_vel_vstu_vssl(vrsumAB, 4*gKernHeight*gKernWidth, &pGKernel[kernelIndexA], vl);
	_vel_vstl_vssl(vrsumAB, 4*gKernHeight*gKernWidth, &pGKernel[kernelIndexB], vl);
	_vel_vstu_vssl(vrsumCD, 4*gKernHeight*gKernWidth, &pGKernel[kernelIndexC], vl);
	_vel_vstl_vssl(vrsumCD, 4*gKernHeight*gKernWidth, &pGKernel[kernelIndexD], vl);
	_vel_vstu_vssl(vrsumEF, 4*gKernHeight*gKernWidth, &pGKernel[kernelIndexE], vl);
	_vel_vstl_vssl(vrsumEF, 4*gKernHeight*gKernWidth, &pGKernel[kernelIndexF], vl);


      } // inChannel
    } // outWidth
  } // kernWidth
}

vednnError_t
vednnConvolutionBackwardFilter_direct_vecC(
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

  const int64_t group          = pParamConv->group;
  const int64_t strideWidth    = pParamConv->strideWidth;;
  const int64_t strideHeight   = pParamConv->strideHeight;
  const int64_t padWidth       = pParamConv->padWidth;
  const int64_t padHeight      = pParamConv->padHeight;
  const int64_t dilationWidth  = pParamConv->dilationWidth;
  const int64_t dilationHeight = pParamConv->dilationHeight;

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

  for (int64_t g = 0; g < group; g++) {
    int64_t inGroupOffset   = g * inChannelGroup  * inHeight  * inWidth;
    int64_t outGroupOffset  = (g * gOutChannelGroup + beginOChannel) * gOutHeight * gOutWidth;
    int64_t kernGroupOffset = (g * gOutChannelGroup + beginOChannel) * inChannelGroup * gKernHeight * gKernWidth;

    int64_t k=0;
    if ( (nOChannel & 0x01) == 1 ) {
      k1(pIn, inWidth, inHeight,
         pGOut, gOutWidth, gOutHeight,
         pGKernel, gKernWidth, gKernHeight,
         strideHeight, strideWidth,
         padHeight, padWidth,
         dilationHeight, dilationWidth,
         inChannelGroup, inChannel, gOutChannel,
         inGroupOffset, outGroupOffset, kernGroupOffset,
         batch,
         k
      ) ;
      k+=1 ;
    }
    if ( ((nOChannel >> 1) & 0x01) == 1 ) {
      k2(pIn, inWidth, inHeight,
         pGOut, gOutWidth, gOutHeight,
         pGKernel, gKernWidth, gKernHeight,
         strideHeight, strideWidth,
         padHeight, padWidth,
         dilationHeight, dilationWidth,
         inChannelGroup, inChannel, gOutChannel,
         inGroupOffset, outGroupOffset, kernGroupOffset,
         batch,
         k
      ) ;
      k+=2;
    }
    if ( ((nOChannel >> 2) & 0x01) == 1 ) {
      k4(pIn, inWidth, inHeight,
         pGOut, gOutWidth, gOutHeight,
         pGKernel, gKernWidth, gKernHeight,
         strideHeight, strideWidth,
         padHeight, padWidth,
         dilationHeight, dilationWidth,
         inChannelGroup, inChannel, gOutChannel,
         inGroupOffset, outGroupOffset, kernGroupOffset,
         batch,
         k
      ) ;
      k+=4;
    }
    if ( ((nOChannel >> 3) & 0x01) == 1 ) {
      k8(pIn, inWidth, inHeight,
         pGOut, gOutWidth, gOutHeight,
         pGKernel, gKernWidth, gKernHeight,
         strideHeight, strideWidth,
         padHeight, padWidth,
         dilationHeight, dilationWidth,
         inChannelGroup, inChannel, gOutChannel,
         inGroupOffset, outGroupOffset, kernGroupOffset,
         batch,
         k
      ) ;
      k+=8;
    }
    for ( ; k<nOChannel; ) {
      k16(pIn, inWidth, inHeight,
         pGOut, gOutWidth, gOutHeight,
         pGKernel, gKernWidth, gKernHeight,
         strideHeight, strideWidth,
         padHeight, padWidth,
         dilationHeight, dilationWidth,
         inChannelGroup, inChannel, gOutChannel,
         inGroupOffset, outGroupOffset, kernGroupOffset,
         batch,
         k
      ) ;
      k+=16;
    } // kernHeight
  } // group

  return VEDNN_SUCCESS;
}

