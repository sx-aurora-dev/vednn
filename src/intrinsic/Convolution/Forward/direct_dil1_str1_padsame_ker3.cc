#include <stdint.h>
#include <stdlib.h>

#include "vednn.h"

#include "velintrin.h"
#define VLEN	(256)

template<filterLayout_t FLAYOUT>
static inline void k1(
    const float * __restrict__ pIn,
    const float * __restrict__ pKernel,
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
    const int64_t kernGroupOffset,
    const int64_t oPixels,
    const int64_t n,
    const int64_t k
)
{
  int64_t outIndex = outGroupOffset + (n * outChannel + k  ) * oPixels ;

  for (int64_t op = 0; op < oPixels; op+=VLEN) {
    const int64_t vl = oPixels - op < VLEN ? oPixels - op : VLEN ;

    __vr vrsum = _vel_vbrds_vsl(0.f, vl) ;

    __vr vrseq = _vel_vseq_vl(vl) ;			// xy
    __vr vridx = _vel_vaddsl_vsvl(op, vrseq, vl) ;	// op + xy

    __vr vry   = _vel_vdivsl_vvsl(vridx, outWidth, vl) ;
    __vr vrx   = _vel_vsubsl_vvvl(vridx, _vel_vmulul_vsvl(outWidth,vry, vl), vl) ;

    __vr vrh_r0 = _vel_vaddsl_vsvl( -1, vry, vl) ;
    __vr vrh_r2 = _vel_vaddsl_vsvl(2-1, vry, vl) ;

    __vr vrw_s0 = _vel_vaddsl_vsvl( -1, vrx, vl) ;
    __vr vrw_s2 = _vel_vaddsl_vsvl(2-1, vrx, vl) ;

    __vm256 vm01_r0 =  _vel_vfmklge_mvl(vrh_r0, vl) ;
    __vm256 vm01_r2 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(inHeight,vrh_r2, vl), vl) ;

    __vm256 vm23_s0  =  _vel_vfmklge_mvl(vrw_s0, vl) ;
    __vm256 vm23_s2  =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(inWidth,vrw_s2, vl), vl) ;


    __vm256 vmall_r0s0 = _vel_andm_mmm(vm01_r0,vm23_s0) ;
    __vm256 vmall_r0s1 = vm01_r0 ;
    __vm256 vmall_r0s2 = _vel_andm_mmm(vm01_r0, vm23_s2) ;

    __vm256 vmall_r1s0 = vm23_s0 ;
    __vm256 vmall_r1s2 = vm23_s2 ;

    __vm256 vmall_r2s0 = _vel_andm_mmm(vm01_r2,vm23_s0) ;
    __vm256 vmall_r2s1 = vm01_r2 ;
    __vm256 vmall_r2s2 = _vel_andm_mmm(vm01_r2, vm23_s2) ;

    for (int64_t c = 0; c < inChannelGroup; c++) {
      const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;

      /* memory access errors mihgt be caused */
      __vr vrin_r0s0 = _vel_vldu_vssl(4,&pInChannel[op-inWidth-1], vl) ;
      __vr vrin_r0s1 = _vel_vldu_vssl(4,&pInChannel[op-inWidth  ], vl) ;
      __vr vrin_r0s2 = _vel_vldu_vssl(4,&pInChannel[op-inWidth+1], vl) ;
      __vr vrin_r1s0 = _vel_vldu_vssl(4,&pInChannel[op+       -1], vl) ;
      __vr vrin_r1s1 = _vel_vldu_vssl(4,&pInChannel[op          ], vl) ;
      __vr vrin_r1s2 = _vel_vldu_vssl(4,&pInChannel[op+       +1], vl) ;
      __vr vrin_r2s0 = _vel_vldu_vssl(4,&pInChannel[op+inWidth-1], vl) ;
      __vr vrin_r2s1 = _vel_vldu_vssl(4,&pInChannel[op+inWidth  ], vl) ;
      __vr vrin_r2s2 = _vel_vldu_vssl(4,&pInChannel[op+inWidth+1], vl) ;

      __vr vrzerof = _vel_vbrds_vsl(0.0f, vl) ;

      vrin_r0s0 = _vel_vmrg_vvvml(vrzerof, vrin_r0s0, vmall_r0s0, vl) ;
      vrin_r0s1 = _vel_vmrg_vvvml(vrzerof, vrin_r0s1, vmall_r0s1, vl) ;
      vrin_r0s2 = _vel_vmrg_vvvml(vrzerof, vrin_r0s2, vmall_r0s2, vl) ;
      __vr vrinP_r0s0 = _vel_vshf_vvvsl(vrin_r0s0, vrin_r0s0, VE_VSHUFFLE_YUZU, vl) ;
      __vr vrinP_r0s1 = _vel_vshf_vvvsl(vrin_r0s1, vrin_r0s1, VE_VSHUFFLE_YUZU, vl) ;
      __vr vrinP_r0s2 = _vel_vshf_vvvsl(vrin_r0s2, vrin_r0s2, VE_VSHUFFLE_YUZU, vl) ;

      vrin_r1s0 = _vel_vmrg_vvvml(vrzerof, vrin_r1s0, vmall_r1s0, vl) ;
      vrin_r1s2 = _vel_vmrg_vvvml(vrzerof, vrin_r1s2, vmall_r1s2, vl) ;
      __vr vrinP_r1s0 = _vel_vshf_vvvsl(vrin_r1s0, vrin_r1s0, VE_VSHUFFLE_YUZU, vl) ;
      __vr vrinP_r1s1 = _vel_vshf_vvvsl(vrin_r1s1, vrin_r1s1, VE_VSHUFFLE_YUZU, vl) ;
      __vr vrinP_r1s2 = _vel_vshf_vvvsl(vrin_r1s2, vrin_r1s2, VE_VSHUFFLE_YUZU, vl) ;

      vrin_r2s0 = _vel_vmrg_vvvml(vrzerof, vrin_r2s0, vmall_r2s0, vl) ;
      vrin_r2s1 = _vel_vmrg_vvvml(vrzerof, vrin_r2s1, vmall_r2s1, vl) ;
      vrin_r2s2 = _vel_vmrg_vvvml(vrzerof, vrin_r2s2, vmall_r2s2, vl) ;
      __vr vrinP_r2s0 = _vel_vshf_vvvsl(vrin_r2s0, vrin_r2s0, VE_VSHUFFLE_YUZU, vl) ;
      __vr vrinP_r2s1 = _vel_vshf_vvvsl(vrin_r2s1, vrin_r2s1, VE_VSHUFFLE_YUZU, vl) ;
      __vr vrinP_r2s2 = _vel_vshf_vvvsl(vrin_r2s2, vrin_r2s2, VE_VSHUFFLE_YUZU, vl) ;

      const float *pKerValue = ( FLAYOUT == VEDNN_FILTER_LAYOUT_NCHW ) ?
				pKernel + kernGroupOffset + ((k * inChannelGroup + c) * kernHeight) * kernWidth :
				pKernel + kernGroupOffset + c * outChannelGroup + k ;

      const int64_t kernelDistance = ( FLAYOUT == VEDNN_FILTER_LAYOUT_NCHW ) ?
				     inChannelGroup * kernHeight * kernWidth :
				     1 ;

#define FILTER_OC1(VRIN,VRSUM)							\
{										\
  VRSUM = _vel_vfmads_vvsvl(VRSUM, pKerValue[0], VRIN, vl) ;			\
}

      if ( FLAYOUT == VEDNN_FILTER_LAYOUT_NCHW ) {
	FILTER_OC1(vrinP_r0s0, vrsum) ; pKerValue++ ;
	FILTER_OC1(vrinP_r0s1, vrsum) ; pKerValue++ ;
	FILTER_OC1(vrinP_r0s2, vrsum) ; pKerValue++ ;
	FILTER_OC1(vrinP_r1s0, vrsum) ; pKerValue++ ;
	FILTER_OC1(vrinP_r1s1, vrsum) ; pKerValue++ ;
	FILTER_OC1(vrinP_r1s2, vrsum) ; pKerValue++ ;
	FILTER_OC1(vrinP_r2s0, vrsum) ; pKerValue++ ;
	FILTER_OC1(vrinP_r2s1, vrsum) ; pKerValue++ ;
	FILTER_OC1(vrinP_r2s2, vrsum) ; pKerValue++ ;
      }
      else {
	FILTER_OC1(vrinP_r0s0, vrsum) ; pKerValue+= inChannelGroup * outChannelGroup ;
	FILTER_OC1(vrinP_r0s1, vrsum) ; pKerValue+= inChannelGroup * outChannelGroup ;
	FILTER_OC1(vrinP_r0s2, vrsum) ; pKerValue+= inChannelGroup * outChannelGroup ;
	FILTER_OC1(vrinP_r1s0, vrsum) ; pKerValue+= inChannelGroup * outChannelGroup ;
	FILTER_OC1(vrinP_r1s1, vrsum) ; pKerValue+= inChannelGroup * outChannelGroup ;
	FILTER_OC1(vrinP_r1s2, vrsum) ; pKerValue+= inChannelGroup * outChannelGroup ;
	FILTER_OC1(vrinP_r2s0, vrsum) ; pKerValue+= inChannelGroup * outChannelGroup ;
	FILTER_OC1(vrinP_r2s1, vrsum) ; pKerValue+= inChannelGroup * outChannelGroup ;
	FILTER_OC1(vrinP_r2s2, vrsum) ; pKerValue+= inChannelGroup * outChannelGroup ;
      }
#undef FILTER_OC1
    } // inChannel

    _vel_vstu_vssl(vrsum, 4, pOut+outIndex, vl) ;

    outIndex += vl ;
  } // outPixels
}

template<filterLayout_t FLAYOUT>
static inline void k2(
    const float * __restrict__ pIn,
    const float * __restrict__ pKernel,
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
    const int64_t kernGroupOffset,
    const int64_t oPixels,
    const int64_t n,
    const int64_t k
)
{
  int64_t outIndex = outGroupOffset + (n * outChannel + k  ) * oPixels ;

  for (int64_t op = 0; op < oPixels; op+=VLEN) {
    const int64_t vl = oPixels - op < VLEN ? oPixels - op : VLEN ;

    __vr vrsum01 = _vel_pvbrd_vsl(0UL, vl) ;

    __vr vrseq = _vel_vseq_vl(vl) ;			// xy
    __vr vridx = _vel_vaddsl_vsvl(op, vrseq, vl) ;	// op + xy

    __vr vry   = _vel_vdivsl_vvsl(vridx, outWidth, vl) ;
    __vr vrx   = _vel_vsubsl_vvvl(vridx, _vel_vmulul_vsvl(outWidth,vry, vl), vl) ;

    __vr vrh_r0 = _vel_vaddsl_vsvl( -1, vry, vl) ;
    __vr vrh_r2 = _vel_vaddsl_vsvl(2-1, vry, vl) ;

    __vr vrw_s0 = _vel_vaddsl_vsvl( -1, vrx, vl) ;
    __vr vrw_s2 = _vel_vaddsl_vsvl(2-1, vrx, vl) ;

    __vm256 vm01_r0 =  _vel_vfmklge_mvl(vrh_r0, vl) ;
    __vm256 vm01_r2 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(inHeight,vrh_r2, vl), vl) ;

    __vm256 vm23_s0  =  _vel_vfmklge_mvl(vrw_s0, vl) ;
    __vm256 vm23_s2  =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(inWidth,vrw_s2, vl), vl) ;


    __vm256 vmall_r0s0 = _vel_andm_mmm(vm01_r0,vm23_s0) ;
    __vm256 vmall_r0s1 = vm01_r0 ;
    __vm256 vmall_r0s2 = _vel_andm_mmm(vm01_r0, vm23_s2) ;

    __vm256 vmall_r1s0 = vm23_s0 ;
    __vm256 vmall_r1s2 = vm23_s2 ;

    __vm256 vmall_r2s0 = _vel_andm_mmm(vm01_r2,vm23_s0) ;
    __vm256 vmall_r2s1 = vm01_r2 ;
    __vm256 vmall_r2s2 = _vel_andm_mmm(vm01_r2, vm23_s2) ;

    for (int64_t c = 0; c < inChannelGroup; c++) {
      const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;

      /* memory access errors mihgt be caused */
      __vr vrin_r0s0 = _vel_vldu_vssl(4,&pInChannel[op-inWidth-1], vl) ;
      __vr vrin_r0s1 = _vel_vldu_vssl(4,&pInChannel[op-inWidth  ], vl) ;
      __vr vrin_r0s2 = _vel_vldu_vssl(4,&pInChannel[op-inWidth+1], vl) ;
      __vr vrin_r1s0 = _vel_vldu_vssl(4,&pInChannel[op+       -1], vl) ;
      __vr vrin_r1s1 = _vel_vldu_vssl(4,&pInChannel[op          ], vl) ;
      __vr vrin_r1s2 = _vel_vldu_vssl(4,&pInChannel[op+       +1], vl) ;
      __vr vrin_r2s0 = _vel_vldu_vssl(4,&pInChannel[op+inWidth-1], vl) ;
      __vr vrin_r2s1 = _vel_vldu_vssl(4,&pInChannel[op+inWidth  ], vl) ;
      __vr vrin_r2s2 = _vel_vldu_vssl(4,&pInChannel[op+inWidth+1], vl) ;

      __vr vrzerof = _vel_vbrds_vsl(0.0f, vl) ;

      vrin_r0s0 = _vel_vmrg_vvvml(vrzerof, vrin_r0s0, vmall_r0s0, vl) ;
      vrin_r0s1 = _vel_vmrg_vvvml(vrzerof, vrin_r0s1, vmall_r0s1, vl) ;
      vrin_r0s2 = _vel_vmrg_vvvml(vrzerof, vrin_r0s2, vmall_r0s2, vl) ;
      __vr vrinP_r0s0 = _vel_vshf_vvvsl(vrin_r0s0, vrin_r0s0, VE_VSHUFFLE_YUZU, vl) ;
      __vr vrinP_r0s1 = _vel_vshf_vvvsl(vrin_r0s1, vrin_r0s1, VE_VSHUFFLE_YUZU, vl) ;
      __vr vrinP_r0s2 = _vel_vshf_vvvsl(vrin_r0s2, vrin_r0s2, VE_VSHUFFLE_YUZU, vl) ;

      vrin_r1s0 = _vel_vmrg_vvvml(vrzerof, vrin_r1s0, vmall_r1s0, vl) ;
      vrin_r1s2 = _vel_vmrg_vvvml(vrzerof, vrin_r1s2, vmall_r1s2, vl) ;
      __vr vrinP_r1s0 = _vel_vshf_vvvsl(vrin_r1s0, vrin_r1s0, VE_VSHUFFLE_YUZU, vl) ;
      __vr vrinP_r1s1 = _vel_vshf_vvvsl(vrin_r1s1, vrin_r1s1, VE_VSHUFFLE_YUZU, vl) ;
      __vr vrinP_r1s2 = _vel_vshf_vvvsl(vrin_r1s2, vrin_r1s2, VE_VSHUFFLE_YUZU, vl) ;

      vrin_r2s0 = _vel_vmrg_vvvml(vrzerof, vrin_r2s0, vmall_r2s0, vl) ;
      vrin_r2s1 = _vel_vmrg_vvvml(vrzerof, vrin_r2s1, vmall_r2s1, vl) ;
      vrin_r2s2 = _vel_vmrg_vvvml(vrzerof, vrin_r2s2, vmall_r2s2, vl) ;
      __vr vrinP_r2s0 = _vel_vshf_vvvsl(vrin_r2s0, vrin_r2s0, VE_VSHUFFLE_YUZU, vl) ;
      __vr vrinP_r2s1 = _vel_vshf_vvvsl(vrin_r2s1, vrin_r2s1, VE_VSHUFFLE_YUZU, vl) ;
      __vr vrinP_r2s2 = _vel_vshf_vvvsl(vrin_r2s2, vrin_r2s2, VE_VSHUFFLE_YUZU, vl) ;

      const float *pKerValue = ( FLAYOUT == VEDNN_FILTER_LAYOUT_NCHW ) ?
				pKernel + kernGroupOffset + ((k * inChannelGroup + c) * kernHeight) * kernWidth :
				pKernel + kernGroupOffset + c * outChannelGroup + k ;

      const int64_t kernelDistance = ( FLAYOUT == VEDNN_FILTER_LAYOUT_NCHW ) ?
				     inChannelGroup * kernHeight * kernWidth :
				     1 ;

#define FILTER_OC2(VRIN,VRSUM)							\
{										\
  const uint64_t kerValue = _vel_pack_f32p(pKerValue + 0 * kernelDistance,	\
					   pKerValue + 1 * kernelDistance) ;	\
  VRSUM = _vel_pvfmad_vvsvl(VRSUM, kerValue, VRIN, vl) ;			\
}

      if ( FLAYOUT == VEDNN_FILTER_LAYOUT_NCHW ) {
	FILTER_OC2(vrinP_r0s0, vrsum01) ; pKerValue++ ;
	FILTER_OC2(vrinP_r0s1, vrsum01) ; pKerValue++ ;
	FILTER_OC2(vrinP_r0s2, vrsum01) ; pKerValue++ ;
	FILTER_OC2(vrinP_r1s0, vrsum01) ; pKerValue++ ;
	FILTER_OC2(vrinP_r1s1, vrsum01) ; pKerValue++ ;
	FILTER_OC2(vrinP_r1s2, vrsum01) ; pKerValue++ ;
	FILTER_OC2(vrinP_r2s0, vrsum01) ; pKerValue++ ;
	FILTER_OC2(vrinP_r2s1, vrsum01) ; pKerValue++ ;
	FILTER_OC2(vrinP_r2s2, vrsum01) ; pKerValue++ ;
      }
      else {
	FILTER_OC2(vrinP_r0s0, vrsum01) ; pKerValue+= inChannelGroup * outChannelGroup ;
	FILTER_OC2(vrinP_r0s1, vrsum01) ; pKerValue+= inChannelGroup * outChannelGroup ;
	FILTER_OC2(vrinP_r0s2, vrsum01) ; pKerValue+= inChannelGroup * outChannelGroup ;
	FILTER_OC2(vrinP_r1s0, vrsum01) ; pKerValue+= inChannelGroup * outChannelGroup ;
	FILTER_OC2(vrinP_r1s1, vrsum01) ; pKerValue+= inChannelGroup * outChannelGroup ;
	FILTER_OC2(vrinP_r1s2, vrsum01) ; pKerValue+= inChannelGroup * outChannelGroup ;
	FILTER_OC2(vrinP_r2s0, vrsum01) ; pKerValue+= inChannelGroup * outChannelGroup ;
	FILTER_OC2(vrinP_r2s1, vrsum01) ; pKerValue+= inChannelGroup * outChannelGroup ;
	FILTER_OC2(vrinP_r2s2, vrsum01) ; pKerValue+= inChannelGroup * outChannelGroup ;
      }
#undef FILTER_OC2
    } // inChannel

    _vel_vstu_vssl(vrsum01, 4, pOut+outIndex, vl) ;
    _vel_vstl_vssl(vrsum01, 4, pOut+outIndex+   oPixels, vl) ;

    outIndex += vl ;
  } // outPixels
}


template<filterLayout_t FLAYOUT>
static inline void k4(
    const float * __restrict__ pIn,
    const float * __restrict__ pKernel,
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
    const int64_t kernGroupOffset,
    const int64_t oPixels,
    const int64_t n,
    const int64_t k
)
{
  int64_t outIndex = outGroupOffset + (n * outChannel + k  ) * oPixels ;

  for (int64_t op = 0; op < oPixels; op+=VLEN) {
    const int64_t vl = oPixels - op < VLEN ? oPixels - op : VLEN ;

    __vr vrsum01 = _vel_pvbrd_vsl(0UL, vl) ;
    __vr vrsum23 = _vel_pvbrd_vsl(0UL, vl) ;

    __vr vrseq = _vel_vseq_vl(vl) ;			// xy
    __vr vridx = _vel_vaddsl_vsvl(op, vrseq, vl) ;	// op + xy

    __vr vry   = _vel_vdivsl_vvsl(vridx, outWidth, vl) ;
    __vr vrx   = _vel_vsubsl_vvvl(vridx, _vel_vmulul_vsvl(outWidth,vry, vl), vl) ;

    __vr vrh_r0 = _vel_vaddsl_vsvl( -1, vry, vl) ;
    __vr vrh_r2 = _vel_vaddsl_vsvl(2-1, vry, vl) ;

    __vr vrw_s0 = _vel_vaddsl_vsvl( -1, vrx, vl) ;
    __vr vrw_s2 = _vel_vaddsl_vsvl(2-1, vrx, vl) ;

    __vm256 vm01_r0 =  _vel_vfmklge_mvl(vrh_r0, vl) ;
    __vm256 vm01_r2 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(inHeight,vrh_r2, vl), vl) ;

    __vm256 vm23_s0  =  _vel_vfmklge_mvl(vrw_s0, vl) ;
    __vm256 vm23_s2  =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(inWidth,vrw_s2, vl), vl) ;


    __vm256 vmall_r0s0 = _vel_andm_mmm(vm01_r0,vm23_s0) ;
    __vm256 vmall_r0s1 = vm01_r0 ;
    __vm256 vmall_r0s2 = _vel_andm_mmm(vm01_r0, vm23_s2) ;

    __vm256 vmall_r1s0 = vm23_s0 ;
    __vm256 vmall_r1s2 = vm23_s2 ;

    __vm256 vmall_r2s0 = _vel_andm_mmm(vm01_r2,vm23_s0) ;
    __vm256 vmall_r2s1 = vm01_r2 ;
    __vm256 vmall_r2s2 = _vel_andm_mmm(vm01_r2, vm23_s2) ;

    for (int64_t c = 0; c < inChannelGroup; c++) {
      const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;

      /* memory access errors mihgt be caused */
      __vr vrin_r0s0 = _vel_vldu_vssl(4,&pInChannel[op-inWidth-1], vl) ;
      __vr vrin_r0s1 = _vel_vldu_vssl(4,&pInChannel[op-inWidth  ], vl) ;
      __vr vrin_r0s2 = _vel_vldu_vssl(4,&pInChannel[op-inWidth+1], vl) ;
      __vr vrin_r1s0 = _vel_vldu_vssl(4,&pInChannel[op+       -1], vl) ;
      __vr vrin_r1s1 = _vel_vldu_vssl(4,&pInChannel[op          ], vl) ;
      __vr vrin_r1s2 = _vel_vldu_vssl(4,&pInChannel[op+       +1], vl) ;
      __vr vrin_r2s0 = _vel_vldu_vssl(4,&pInChannel[op+inWidth-1], vl) ;
      __vr vrin_r2s1 = _vel_vldu_vssl(4,&pInChannel[op+inWidth  ], vl) ;
      __vr vrin_r2s2 = _vel_vldu_vssl(4,&pInChannel[op+inWidth+1], vl) ;

      __vr vrzerof = _vel_vbrds_vsl(0.0f, vl) ;

      vrin_r0s0 = _vel_vmrg_vvvml(vrzerof, vrin_r0s0, vmall_r0s0, vl) ;
      vrin_r0s1 = _vel_vmrg_vvvml(vrzerof, vrin_r0s1, vmall_r0s1, vl) ;
      vrin_r0s2 = _vel_vmrg_vvvml(vrzerof, vrin_r0s2, vmall_r0s2, vl) ;
      __vr vrinP_r0s0 = _vel_vshf_vvvsl(vrin_r0s0, vrin_r0s0, VE_VSHUFFLE_YUZU, vl) ;
      __vr vrinP_r0s1 = _vel_vshf_vvvsl(vrin_r0s1, vrin_r0s1, VE_VSHUFFLE_YUZU, vl) ;
      __vr vrinP_r0s2 = _vel_vshf_vvvsl(vrin_r0s2, vrin_r0s2, VE_VSHUFFLE_YUZU, vl) ;

      vrin_r1s0 = _vel_vmrg_vvvml(vrzerof, vrin_r1s0, vmall_r1s0, vl) ;
      vrin_r1s2 = _vel_vmrg_vvvml(vrzerof, vrin_r1s2, vmall_r1s2, vl) ;
      __vr vrinP_r1s0 = _vel_vshf_vvvsl(vrin_r1s0, vrin_r1s0, VE_VSHUFFLE_YUZU, vl) ;
      __vr vrinP_r1s1 = _vel_vshf_vvvsl(vrin_r1s1, vrin_r1s1, VE_VSHUFFLE_YUZU, vl) ;
      __vr vrinP_r1s2 = _vel_vshf_vvvsl(vrin_r1s2, vrin_r1s2, VE_VSHUFFLE_YUZU, vl) ;

      vrin_r2s0 = _vel_vmrg_vvvml(vrzerof, vrin_r2s0, vmall_r2s0, vl) ;
      vrin_r2s1 = _vel_vmrg_vvvml(vrzerof, vrin_r2s1, vmall_r2s1, vl) ;
      vrin_r2s2 = _vel_vmrg_vvvml(vrzerof, vrin_r2s2, vmall_r2s2, vl) ;
      __vr vrinP_r2s0 = _vel_vshf_vvvsl(vrin_r2s0, vrin_r2s0, VE_VSHUFFLE_YUZU, vl) ;
      __vr vrinP_r2s1 = _vel_vshf_vvvsl(vrin_r2s1, vrin_r2s1, VE_VSHUFFLE_YUZU, vl) ;
      __vr vrinP_r2s2 = _vel_vshf_vvvsl(vrin_r2s2, vrin_r2s2, VE_VSHUFFLE_YUZU, vl) ;

      const float *pKerValue = ( FLAYOUT == VEDNN_FILTER_LAYOUT_NCHW ) ?
				pKernel + kernGroupOffset + ((k * inChannelGroup + c) * kernHeight) * kernWidth :
				pKernel + kernGroupOffset + c * outChannelGroup + k ;

      const int64_t kernelDistance = ( FLAYOUT == VEDNN_FILTER_LAYOUT_NCHW ) ?
				     inChannelGroup * kernHeight * kernWidth :
				     1 ;

#define FILTER_OC4(VRIN,VRSUM0,VRSUM1)						\
{										\
  const uint64_t kerValue0 = _vel_pack_f32p(pKerValue + 0 * kernelDistance,	\
					    pKerValue + 1 * kernelDistance) ;	\
  const uint64_t kerValue1 = _vel_pack_f32p(pKerValue + 2 * kernelDistance,	\
					    pKerValue + 3 * kernelDistance) ;	\
  VRSUM0 = _vel_pvfmad_vvsvl(VRSUM0, kerValue0, VRIN, vl) ;			\
  VRSUM1 = _vel_pvfmad_vvsvl(VRSUM1, kerValue1, VRIN, vl) ;			\
}

      if ( FLAYOUT == VEDNN_FILTER_LAYOUT_NCHW ) {
	FILTER_OC4(vrinP_r0s0, vrsum01,vrsum23) ; pKerValue++ ;
	FILTER_OC4(vrinP_r0s1, vrsum01,vrsum23) ; pKerValue++ ;
	FILTER_OC4(vrinP_r0s2, vrsum01,vrsum23) ; pKerValue++ ;
	FILTER_OC4(vrinP_r1s0, vrsum01,vrsum23) ; pKerValue++ ;
	FILTER_OC4(vrinP_r1s1, vrsum01,vrsum23) ; pKerValue++ ;
	FILTER_OC4(vrinP_r1s2, vrsum01,vrsum23) ; pKerValue++ ;
	FILTER_OC4(vrinP_r2s0, vrsum01,vrsum23) ; pKerValue++ ;
	FILTER_OC4(vrinP_r2s1, vrsum01,vrsum23) ; pKerValue++ ;
	FILTER_OC4(vrinP_r2s2, vrsum01,vrsum23) ; pKerValue++ ;
      }
      else {
	FILTER_OC4(vrinP_r0s0, vrsum01,vrsum23) ; pKerValue+= inChannelGroup * outChannelGroup ;
	FILTER_OC4(vrinP_r0s1, vrsum01,vrsum23) ; pKerValue+= inChannelGroup * outChannelGroup ;
	FILTER_OC4(vrinP_r0s2, vrsum01,vrsum23) ; pKerValue+= inChannelGroup * outChannelGroup ;
	FILTER_OC4(vrinP_r1s0, vrsum01,vrsum23) ; pKerValue+= inChannelGroup * outChannelGroup ;
	FILTER_OC4(vrinP_r1s1, vrsum01,vrsum23) ; pKerValue+= inChannelGroup * outChannelGroup ;
	FILTER_OC4(vrinP_r1s2, vrsum01,vrsum23) ; pKerValue+= inChannelGroup * outChannelGroup ;
	FILTER_OC4(vrinP_r2s0, vrsum01,vrsum23) ; pKerValue+= inChannelGroup * outChannelGroup ;
	FILTER_OC4(vrinP_r2s1, vrsum01,vrsum23) ; pKerValue+= inChannelGroup * outChannelGroup ;
	FILTER_OC4(vrinP_r2s2, vrsum01,vrsum23) ; pKerValue+= inChannelGroup * outChannelGroup ;
      }
#undef FILTER_OC4
    } // inChannel

    _vel_vstu_vssl(vrsum01, 4, pOut+outIndex, vl) ;
    _vel_vstl_vssl(vrsum01, 4, pOut+outIndex+   oPixels, vl) ;
    _vel_vstu_vssl(vrsum23, 4, pOut+outIndex+ 2*oPixels, vl) ;
    _vel_vstl_vssl(vrsum23, 4, pOut+outIndex+ 3*oPixels, vl) ;

    outIndex += vl ;
  } // outPixels
}

template<filterLayout_t FLAYOUT>
static inline void k8(
    const float * __restrict__ pIn,
    const float * __restrict__ pKernel,
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
    const int64_t kernGroupOffset,
    const int64_t oPixels,
    const int64_t n,
    const int64_t k
)
{
  int64_t outIndex = outGroupOffset + (n * outChannel + k  ) * oPixels ;

  for (int64_t op = 0; op < oPixels; op+=VLEN) {
    const int64_t vl = oPixels - op < VLEN ? oPixels - op : VLEN ;

    __vr vrsum01 = _vel_pvbrd_vsl(0UL, vl) ;
    __vr vrsum23 = _vel_pvbrd_vsl(0UL, vl) ;
    __vr vrsum45 = _vel_pvbrd_vsl(0UL, vl) ;
    __vr vrsum67 = _vel_pvbrd_vsl(0UL, vl) ;

    __vr vrseq = _vel_vseq_vl(vl) ;			// xy
    __vr vridx = _vel_vaddsl_vsvl(op, vrseq, vl) ;	// op + xy

    __vr vry   = _vel_vdivsl_vvsl(vridx, outWidth, vl) ;
    __vr vrx   = _vel_vsubsl_vvvl(vridx, _vel_vmulul_vsvl(outWidth,vry, vl), vl) ;

    __vr vrh_r0 = _vel_vaddsl_vsvl( -1, vry, vl) ;
    __vr vrh_r2 = _vel_vaddsl_vsvl(2-1, vry, vl) ;

    __vr vrw_s0 = _vel_vaddsl_vsvl( -1, vrx, vl) ;
    __vr vrw_s2 = _vel_vaddsl_vsvl(2-1, vrx, vl) ;

    __vm256 vm01_r0 =  _vel_vfmklge_mvl(vrh_r0, vl) ;
    __vm256 vm01_r2 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(inHeight,vrh_r2, vl), vl) ;

    __vm256 vm23_s0  =  _vel_vfmklge_mvl(vrw_s0, vl) ;
    __vm256 vm23_s2  =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(inWidth,vrw_s2, vl), vl) ;


    __vm256 vmall_r0s0 = _vel_andm_mmm(vm01_r0,vm23_s0) ;
    __vm256 vmall_r0s1 = vm01_r0 ;
    __vm256 vmall_r0s2 = _vel_andm_mmm(vm01_r0, vm23_s2) ;

    __vm256 vmall_r1s0 = vm23_s0 ;
    __vm256 vmall_r1s2 = vm23_s2 ;

    __vm256 vmall_r2s0 = _vel_andm_mmm(vm01_r2,vm23_s0) ;
    __vm256 vmall_r2s1 = vm01_r2 ;
    __vm256 vmall_r2s2 = _vel_andm_mmm(vm01_r2, vm23_s2) ;

    for (int64_t c = 0; c < inChannelGroup; c++) {
      const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;

      /* memory access errors mihgt be caused */
      __vr vrin_r0s0 = _vel_vldu_vssl(4,&pInChannel[op-inWidth-1], vl) ;
      __vr vrin_r0s1 = _vel_vldu_vssl(4,&pInChannel[op-inWidth  ], vl) ;
      __vr vrin_r0s2 = _vel_vldu_vssl(4,&pInChannel[op-inWidth+1], vl) ;
      __vr vrin_r1s0 = _vel_vldu_vssl(4,&pInChannel[op+       -1], vl) ;
      __vr vrin_r1s1 = _vel_vldu_vssl(4,&pInChannel[op          ], vl) ;
      __vr vrin_r1s2 = _vel_vldu_vssl(4,&pInChannel[op+       +1], vl) ;
      __vr vrin_r2s0 = _vel_vldu_vssl(4,&pInChannel[op+inWidth-1], vl) ;
      __vr vrin_r2s1 = _vel_vldu_vssl(4,&pInChannel[op+inWidth  ], vl) ;
      __vr vrin_r2s2 = _vel_vldu_vssl(4,&pInChannel[op+inWidth+1], vl) ;

      __vr vrzerof = _vel_vbrds_vsl(0.0f, vl) ;

      vrin_r0s0 = _vel_vmrg_vvvml(vrzerof, vrin_r0s0, vmall_r0s0, vl) ;
      vrin_r0s1 = _vel_vmrg_vvvml(vrzerof, vrin_r0s1, vmall_r0s1, vl) ;
      vrin_r0s2 = _vel_vmrg_vvvml(vrzerof, vrin_r0s2, vmall_r0s2, vl) ;
      __vr vrinP_r0s0 = _vel_vshf_vvvsl(vrin_r0s0, vrin_r0s0, VE_VSHUFFLE_YUZU, vl) ;
      __vr vrinP_r0s1 = _vel_vshf_vvvsl(vrin_r0s1, vrin_r0s1, VE_VSHUFFLE_YUZU, vl) ;
      __vr vrinP_r0s2 = _vel_vshf_vvvsl(vrin_r0s2, vrin_r0s2, VE_VSHUFFLE_YUZU, vl) ;

      vrin_r1s0 = _vel_vmrg_vvvml(vrzerof, vrin_r1s0, vmall_r1s0, vl) ;
      vrin_r1s2 = _vel_vmrg_vvvml(vrzerof, vrin_r1s2, vmall_r1s2, vl) ;
      __vr vrinP_r1s0 = _vel_vshf_vvvsl(vrin_r1s0, vrin_r1s0, VE_VSHUFFLE_YUZU, vl) ;
      __vr vrinP_r1s1 = _vel_vshf_vvvsl(vrin_r1s1, vrin_r1s1, VE_VSHUFFLE_YUZU, vl) ;
      __vr vrinP_r1s2 = _vel_vshf_vvvsl(vrin_r1s2, vrin_r1s2, VE_VSHUFFLE_YUZU, vl) ;

      vrin_r2s0 = _vel_vmrg_vvvml(vrzerof, vrin_r2s0, vmall_r2s0, vl) ;
      vrin_r2s1 = _vel_vmrg_vvvml(vrzerof, vrin_r2s1, vmall_r2s1, vl) ;
      vrin_r2s2 = _vel_vmrg_vvvml(vrzerof, vrin_r2s2, vmall_r2s2, vl) ;
      __vr vrinP_r2s0 = _vel_vshf_vvvsl(vrin_r2s0, vrin_r2s0, VE_VSHUFFLE_YUZU, vl) ;
      __vr vrinP_r2s1 = _vel_vshf_vvvsl(vrin_r2s1, vrin_r2s1, VE_VSHUFFLE_YUZU, vl) ;
      __vr vrinP_r2s2 = _vel_vshf_vvvsl(vrin_r2s2, vrin_r2s2, VE_VSHUFFLE_YUZU, vl) ;

      const float *pKerValue = ( FLAYOUT == VEDNN_FILTER_LAYOUT_NCHW ) ?
				pKernel + kernGroupOffset + ((k * inChannelGroup + c) * kernHeight) * kernWidth :
				pKernel + kernGroupOffset + c * outChannelGroup + k ;

      const int64_t kernelDistance = ( FLAYOUT == VEDNN_FILTER_LAYOUT_NCHW ) ?
				     inChannelGroup * kernHeight * kernWidth :
				     1 ;

#define FILTER_OC4(VRIN,VRSUM0,VRSUM1,N)									\
{											\
const uint64_t kerValue0 = _vel_pack_f32p(pKerValue +  (N)    * kernelDistance,		\
				          pKerValue + ((N)+1) * kernelDistance) ;	\
const uint64_t kerValue1 = _vel_pack_f32p(pKerValue + ((N)+2) * kernelDistance,		\
				          pKerValue + ((N)+3) * kernelDistance) ;	\
VRSUM0 = _vel_pvfmad_vvsvl(VRSUM0, kerValue0, VRIN, vl) ;								\
VRSUM1 = _vel_pvfmad_vvsvl(VRSUM1, kerValue1, VRIN, vl) ;								\
}
#define FILTER_OC8(VRIN)		\
{					\
FILTER_OC4(VRIN,vrsum01,vrsum23,0) ;	\
FILTER_OC4(VRIN,vrsum45,vrsum67,4) ;	\
}
      if ( FLAYOUT == VEDNN_FILTER_LAYOUT_NCHW ) {
	FILTER_OC8(vrinP_r0s0) ; pKerValue++ ;
	FILTER_OC8(vrinP_r0s1) ; pKerValue++ ;
	FILTER_OC8(vrinP_r0s2) ; pKerValue++ ;
	FILTER_OC8(vrinP_r1s0) ; pKerValue++ ;
	FILTER_OC8(vrinP_r1s1) ; pKerValue++ ;
	FILTER_OC8(vrinP_r1s2) ; pKerValue++ ;
	FILTER_OC8(vrinP_r2s0) ; pKerValue++ ;
	FILTER_OC8(vrinP_r2s1) ; pKerValue++ ;
	FILTER_OC8(vrinP_r2s2) ; pKerValue++ ;
      }
      else {
	FILTER_OC8(vrinP_r0s0) ; pKerValue+= inChannelGroup * outChannelGroup  ;
	FILTER_OC8(vrinP_r0s1) ; pKerValue+= inChannelGroup * outChannelGroup  ;
	FILTER_OC8(vrinP_r0s2) ; pKerValue+= inChannelGroup * outChannelGroup  ;
	FILTER_OC8(vrinP_r1s0) ; pKerValue+= inChannelGroup * outChannelGroup  ;
	FILTER_OC8(vrinP_r1s1) ; pKerValue+= inChannelGroup * outChannelGroup  ;
	FILTER_OC8(vrinP_r1s2) ; pKerValue+= inChannelGroup * outChannelGroup  ;
	FILTER_OC8(vrinP_r2s0) ; pKerValue+= inChannelGroup * outChannelGroup  ;
	FILTER_OC8(vrinP_r2s1) ; pKerValue+= inChannelGroup * outChannelGroup  ;
	FILTER_OC8(vrinP_r2s2) ; pKerValue+= inChannelGroup * outChannelGroup  ;
      }
#undef FILTER_OC4
#undef FILTER_OC8
    } // inChannel

    _vel_vstu_vssl(vrsum01, 4, pOut+outIndex, vl) ;
    _vel_vstl_vssl(vrsum01, 4, pOut+outIndex+   oPixels, vl) ;
    _vel_vstu_vssl(vrsum23, 4, pOut+outIndex+ 2*oPixels, vl) ;
    _vel_vstl_vssl(vrsum23, 4, pOut+outIndex+ 3*oPixels, vl) ;
    _vel_vstu_vssl(vrsum45, 4, pOut+outIndex+ 4*oPixels, vl) ;
    _vel_vstl_vssl(vrsum45, 4, pOut+outIndex+ 5*oPixels, vl) ;
    _vel_vstu_vssl(vrsum67, 4, pOut+outIndex+ 6*oPixels, vl) ;
    _vel_vstl_vssl(vrsum67, 4, pOut+outIndex+ 7*oPixels, vl) ;

    outIndex += vl ;
  } // outPixels
}

template<filterLayout_t FLAYOUT>
static inline void k16(
    const float * __restrict__ pIn,
    const float * __restrict__ pKernel,
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
    const int64_t kernGroupOffset,
    const int64_t oPixels,
    const int64_t n,
    const int64_t k
)
{
  int64_t outIndex = outGroupOffset + (n * outChannel + k  ) * oPixels ;

  for (int64_t op = 0; op < oPixels; op+=VLEN) {
    const int64_t vl = oPixels - op < VLEN ? oPixels - op : VLEN ;

    __vr vrsum01 = _vel_pvbrd_vsl(0UL, vl) ;
    __vr vrsum23 = _vel_pvbrd_vsl(0UL, vl) ;
    __vr vrsum45 = _vel_pvbrd_vsl(0UL, vl) ;
    __vr vrsum67 = _vel_pvbrd_vsl(0UL, vl) ;
    __vr vrsum89 = _vel_pvbrd_vsl(0UL, vl) ;
    __vr vrsumAB = _vel_pvbrd_vsl(0UL, vl) ;
    __vr vrsumCD = _vel_pvbrd_vsl(0UL, vl) ;
    __vr vrsumEF = _vel_pvbrd_vsl(0UL, vl) ;

    __vr vrseq = _vel_vseq_vl(vl) ;			// xy
    __vr vridx = _vel_vaddsl_vsvl(op, vrseq, vl) ;	// op + xy

    __vr vry   = _vel_vdivsl_vvsl(vridx, outWidth, vl) ;
    __vr vrx   = _vel_vsubsl_vvvl(vridx, _vel_vmulul_vsvl(outWidth,vry, vl), vl) ;

    __vr vrh_r0 = _vel_vaddsl_vsvl( -1, vry, vl) ;
    __vr vrh_r2 = _vel_vaddsl_vsvl(2-1, vry, vl) ;

    __vr vrw_s0 = _vel_vaddsl_vsvl( -1, vrx, vl) ;
    __vr vrw_s2 = _vel_vaddsl_vsvl(2-1, vrx, vl) ;

    __vm256 vm01_r0 =  _vel_vfmklge_mvl(vrh_r0, vl) ;
    __vm256 vm01_r2 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(inHeight,vrh_r2, vl), vl) ;

    __vm256 vm23_s0  =  _vel_vfmklge_mvl(vrw_s0, vl) ;
    __vm256 vm23_s2  =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(inWidth,vrw_s2, vl), vl) ;


    __vm256 vmall_r0s0 = _vel_andm_mmm(vm01_r0,vm23_s0) ;
    __vm256 vmall_r0s1 = vm01_r0 ;
    __vm256 vmall_r0s2 = _vel_andm_mmm(vm01_r0, vm23_s2) ;

    __vm256 vmall_r1s0 = vm23_s0 ;
    __vm256 vmall_r1s2 = vm23_s2 ;

    __vm256 vmall_r2s0 = _vel_andm_mmm(vm01_r2,vm23_s0) ;
    __vm256 vmall_r2s1 = vm01_r2 ;
    __vm256 vmall_r2s2 = _vel_andm_mmm(vm01_r2, vm23_s2) ;

    for (int64_t c = 0; c < inChannelGroup; c++) {
      const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;

      /* memory access errors mihgt be caused */
      __vr vrin_r0s0 = _vel_vldu_vssl(4,&pInChannel[op-inWidth-1], vl) ;
      __vr vrin_r0s1 = _vel_vldu_vssl(4,&pInChannel[op-inWidth  ], vl) ;
      __vr vrin_r0s2 = _vel_vldu_vssl(4,&pInChannel[op-inWidth+1], vl) ;
      __vr vrin_r1s0 = _vel_vldu_vssl(4,&pInChannel[op+       -1], vl) ;
      __vr vrin_r1s1 = _vel_vldu_vssl(4,&pInChannel[op          ], vl) ;
      __vr vrin_r1s2 = _vel_vldu_vssl(4,&pInChannel[op+       +1], vl) ;
      __vr vrin_r2s0 = _vel_vldu_vssl(4,&pInChannel[op+inWidth-1], vl) ;
      __vr vrin_r2s1 = _vel_vldu_vssl(4,&pInChannel[op+inWidth  ], vl) ;
      __vr vrin_r2s2 = _vel_vldu_vssl(4,&pInChannel[op+inWidth+1], vl) ;

      __vr vrzerof = _vel_vbrds_vsl(0.0f, vl) ;

      vrin_r0s0 = _vel_vmrg_vvvml(vrzerof, vrin_r0s0, vmall_r0s0, vl) ;
      vrin_r0s1 = _vel_vmrg_vvvml(vrzerof, vrin_r0s1, vmall_r0s1, vl) ;
      vrin_r0s2 = _vel_vmrg_vvvml(vrzerof, vrin_r0s2, vmall_r0s2, vl) ;
      __vr vrinP_r0s0 = _vel_vshf_vvvsl(vrin_r0s0, vrin_r0s0, VE_VSHUFFLE_YUZU, vl) ;
      __vr vrinP_r0s1 = _vel_vshf_vvvsl(vrin_r0s1, vrin_r0s1, VE_VSHUFFLE_YUZU, vl) ;
      __vr vrinP_r0s2 = _vel_vshf_vvvsl(vrin_r0s2, vrin_r0s2, VE_VSHUFFLE_YUZU, vl) ;

      vrin_r1s0 = _vel_vmrg_vvvml(vrzerof, vrin_r1s0, vmall_r1s0, vl) ;
      vrin_r1s2 = _vel_vmrg_vvvml(vrzerof, vrin_r1s2, vmall_r1s2, vl) ;
      __vr vrinP_r1s0 = _vel_vshf_vvvsl(vrin_r1s0, vrin_r1s0, VE_VSHUFFLE_YUZU, vl) ;
      __vr vrinP_r1s1 = _vel_vshf_vvvsl(vrin_r1s1, vrin_r1s1, VE_VSHUFFLE_YUZU, vl) ;
      __vr vrinP_r1s2 = _vel_vshf_vvvsl(vrin_r1s2, vrin_r1s2, VE_VSHUFFLE_YUZU, vl) ;

      vrin_r2s0 = _vel_vmrg_vvvml(vrzerof, vrin_r2s0, vmall_r2s0, vl) ;
      vrin_r2s1 = _vel_vmrg_vvvml(vrzerof, vrin_r2s1, vmall_r2s1, vl) ;
      vrin_r2s2 = _vel_vmrg_vvvml(vrzerof, vrin_r2s2, vmall_r2s2, vl) ;
      __vr vrinP_r2s0 = _vel_vshf_vvvsl(vrin_r2s0, vrin_r2s0, VE_VSHUFFLE_YUZU, vl) ;
      __vr vrinP_r2s1 = _vel_vshf_vvvsl(vrin_r2s1, vrin_r2s1, VE_VSHUFFLE_YUZU, vl) ;
      __vr vrinP_r2s2 = _vel_vshf_vvvsl(vrin_r2s2, vrin_r2s2, VE_VSHUFFLE_YUZU, vl) ;

      const float *pKerValue = ( FLAYOUT == VEDNN_FILTER_LAYOUT_NCHW ) ?
				pKernel + kernGroupOffset + ((k * inChannelGroup + c) * kernHeight) * kernWidth :
				pKernel + kernGroupOffset + c * outChannelGroup + k ;

      const int64_t kernelDistance = ( FLAYOUT == VEDNN_FILTER_LAYOUT_NCHW ) ?
				     inChannelGroup * kernHeight * kernWidth :
				     1 ;

#define FILTER_OC4(VRIN,VRSUM0,VRSUM1,N)									\
{											\
const uint64_t kerValue0 = _vel_pack_f32p(pKerValue +  (N)    * kernelDistance,		\
				          pKerValue + ((N)+1) * kernelDistance) ;	\
const uint64_t kerValue1 = _vel_pack_f32p(pKerValue + ((N)+2) * kernelDistance,		\
				          pKerValue + ((N)+3) * kernelDistance) ;	\
VRSUM0 = _vel_pvfmad_vvsvl(VRSUM0, kerValue0, VRIN, vl) ;								\
VRSUM1 = _vel_pvfmad_vvsvl(VRSUM1, kerValue1, VRIN, vl) ;								\
}
#define FILTER_OC16(VRIN)		\
{					\
FILTER_OC4(VRIN,vrsum01,vrsum23,0) ;	\
FILTER_OC4(VRIN,vrsum45,vrsum67,4) ;	\
FILTER_OC4(VRIN,vrsum89,vrsumAB,8) ;	\
FILTER_OC4(VRIN,vrsumCD,vrsumEF,12) ;	\
}
      if ( FLAYOUT == VEDNN_FILTER_LAYOUT_NCHW ) {
	FILTER_OC16(vrinP_r0s0) ; pKerValue++ ;
	FILTER_OC16(vrinP_r0s1) ; pKerValue++ ;
	FILTER_OC16(vrinP_r0s2) ; pKerValue++ ;
	FILTER_OC16(vrinP_r1s0) ; pKerValue++ ;
	FILTER_OC16(vrinP_r1s1) ; pKerValue++ ;
	FILTER_OC16(vrinP_r1s2) ; pKerValue++ ;
	FILTER_OC16(vrinP_r2s0) ; pKerValue++ ;
	FILTER_OC16(vrinP_r2s1) ; pKerValue++ ;
	FILTER_OC16(vrinP_r2s2) ; pKerValue++ ;
      }
      else {
	FILTER_OC16(vrinP_r0s0) ; pKerValue+= inChannelGroup * outChannelGroup  ;
	FILTER_OC16(vrinP_r0s1) ; pKerValue+= inChannelGroup * outChannelGroup  ;
	FILTER_OC16(vrinP_r0s2) ; pKerValue+= inChannelGroup * outChannelGroup  ;
	FILTER_OC16(vrinP_r1s0) ; pKerValue+= inChannelGroup * outChannelGroup  ;
	FILTER_OC16(vrinP_r1s1) ; pKerValue+= inChannelGroup * outChannelGroup  ;
	FILTER_OC16(vrinP_r1s2) ; pKerValue+= inChannelGroup * outChannelGroup  ;
	FILTER_OC16(vrinP_r2s0) ; pKerValue+= inChannelGroup * outChannelGroup  ;
	FILTER_OC16(vrinP_r2s1) ; pKerValue+= inChannelGroup * outChannelGroup  ;
	FILTER_OC16(vrinP_r2s2) ; pKerValue+= inChannelGroup * outChannelGroup  ;
      }
#undef FILTER_OC4
#undef FILTER_OC16
    } // inChannel

    _vel_vstu_vssl(vrsum01, 4, pOut+outIndex, vl) ;
    _vel_vstl_vssl(vrsum01, 4, pOut+outIndex+   oPixels, vl) ;
    _vel_vstu_vssl(vrsum23, 4, pOut+outIndex+ 2*oPixels, vl) ;
    _vel_vstl_vssl(vrsum23, 4, pOut+outIndex+ 3*oPixels, vl) ;
    _vel_vstu_vssl(vrsum45, 4, pOut+outIndex+ 4*oPixels, vl) ;
    _vel_vstl_vssl(vrsum45, 4, pOut+outIndex+ 5*oPixels, vl) ;
    _vel_vstu_vssl(vrsum67, 4, pOut+outIndex+ 6*oPixels, vl) ;
    _vel_vstl_vssl(vrsum67, 4, pOut+outIndex+ 7*oPixels, vl) ;
    _vel_vstu_vssl(vrsum89, 4, pOut+outIndex+ 8*oPixels, vl) ;
    _vel_vstl_vssl(vrsum89, 4, pOut+outIndex+ 9*oPixels, vl) ;
    _vel_vstu_vssl(vrsumAB, 4, pOut+outIndex+10*oPixels, vl) ;
    _vel_vstl_vssl(vrsumAB, 4, pOut+outIndex+11*oPixels, vl) ;
    _vel_vstu_vssl(vrsumCD, 4, pOut+outIndex+12*oPixels, vl) ;
    _vel_vstl_vssl(vrsumCD, 4, pOut+outIndex+13*oPixels, vl) ;
    _vel_vstu_vssl(vrsumEF, 4, pOut+outIndex+14*oPixels, vl) ;
    _vel_vstl_vssl(vrsumEF, 4, pOut+outIndex+15*oPixels, vl) ;

    outIndex += vl ;
  } // outPixels
}

extern "C" vednnError_t
vednnConvolutionForward_direct_dil1_str1_padsame_ker3(
    const vednnTensorParam_t *  	pParamIn,
    const void *  			pDataIn,
    const vednnFilterParam_t *  	pParamKernel,
    const void *  			pDataKernel,
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
  const int64_t kernWidth  = pParamKernel->width;		/* must be 3 */
  const int64_t kernHeight = pParamKernel->height;		/* must be 3 */

  const int64_t filter_layout = pParamKernel->layout ;

  const int64_t group          = pParamConv->group;
//  const int64_t strideWidth    = pParamConv->strideWidth;	/* must be 1 */
//  const int64_t strideHeight   = pParamConv->strideHeight;	/* must be 1 */
  const int64_t padWidth       = pParamConv->padWidth;
  const int64_t padHeight      = pParamConv->padHeight;
//  const int64_t dilationWidth  = pParamConv->dilationWidth;	/* must be 1 */
//  const int64_t dilationHeight = pParamConv->dilationHeight;	/* must be 1 */

  const int64_t inChannelGroup  = inChannel  / group;   // equal to pDataKernel->inChannel
  const int64_t outChannelGroup = outChannel / group;   // equal to pDataKernel->outChannel

  const float * pIn     = (const float *) pDataIn;
  const float * pKernel = (const float *) pDataKernel;
  float * const pOut    = (float * const) pDataOut;

  const int oPixels= outHeight*outWidth ;

  {
    for (int64_t n = 0; n < batch; n++) {
      for (int64_t g = 0; g < group; g++) {
	const int64_t inGroupOffset   = g * inChannelGroup * inHeight * inWidth;
	const int64_t outGroupOffset  = g * outChannelGroup * outHeight * outWidth;
	const int64_t kernGroupOffset = g * outChannelGroup * inChannelGroup * kernHeight * kernWidth;

	int k = 0 ;
	if ( (outChannelGroup & 0x01) == 1 ) {
	  if( filter_layout == VEDNN_FILTER_LAYOUT_NCHW) {
	    k1<VEDNN_FILTER_LAYOUT_NCHW>(pIn, pKernel, pOut,
	       inChannel, inWidth, inHeight,
	       outChannel, outWidth, outHeight,
	       kernWidth, kernHeight,
	       inChannelGroup, outChannelGroup,
	       inGroupOffset, outGroupOffset,
	       kernGroupOffset, oPixels, n, k) ;
	  }
	  else {
	    k1<VEDNN_FILTER_LAYOUT_HWCN>(pIn, pKernel, pOut,
	       inChannel, inWidth, inHeight,
	       outChannel, outWidth, outHeight,
	       kernWidth, kernHeight,
	       inChannelGroup, outChannelGroup,
	       inGroupOffset, outGroupOffset,
	       kernGroupOffset, oPixels, n, k) ;
	  }

	  k++ ;
	}
	if ( ((outChannelGroup >> 1) & 0x01) == 1 ) {
	  if( filter_layout == VEDNN_FILTER_LAYOUT_NCHW) {
	    k2<VEDNN_FILTER_LAYOUT_NCHW>(pIn, pKernel, pOut,
	       inChannel, inWidth, inHeight,
	       outChannel, outWidth, outHeight,
	       kernWidth, kernHeight,
	       inChannelGroup, outChannelGroup,
	       inGroupOffset, outGroupOffset,
	       kernGroupOffset, oPixels, n, k) ;
	  }
	  else {
	    k2<VEDNN_FILTER_LAYOUT_HWCN>(pIn, pKernel, pOut,
	       inChannel, inWidth, inHeight,
	       outChannel, outWidth, outHeight,
	       kernWidth, kernHeight,
	       inChannelGroup, outChannelGroup,
	       inGroupOffset, outGroupOffset,
	       kernGroupOffset, oPixels, n, k) ;
	  }

	  k+=2 ;
	}
	if ( ((outChannelGroup >> 2) & 0x01) == 1 ) {
	  if( filter_layout == VEDNN_FILTER_LAYOUT_NCHW) {
	    k4<VEDNN_FILTER_LAYOUT_NCHW>(pIn, pKernel, pOut,
	       inChannel, inWidth, inHeight,
	       outChannel, outWidth, outHeight,
	       kernWidth, kernHeight,
	       inChannelGroup, outChannelGroup,
	       inGroupOffset, outGroupOffset,
	       kernGroupOffset, oPixels, n, k) ;
	  }
	  else {
	    k4<VEDNN_FILTER_LAYOUT_HWCN>(pIn, pKernel, pOut,
	       inChannel, inWidth, inHeight,
	       outChannel, outWidth, outHeight,
	       kernWidth, kernHeight,
	       inChannelGroup, outChannelGroup,
	       inGroupOffset, outGroupOffset,
	       kernGroupOffset, oPixels, n, k) ;
	  }

	  k+=4 ;
	}
	if ( ((outChannelGroup >> 3) & 0x01) == 1 ) {
	  if( filter_layout == VEDNN_FILTER_LAYOUT_NCHW) {
	    k8<VEDNN_FILTER_LAYOUT_NCHW>(pIn, pKernel, pOut,
	       inChannel, inWidth, inHeight,
	       outChannel, outWidth, outHeight,
	       kernWidth, kernHeight,
	       inChannelGroup, outChannelGroup,
	       inGroupOffset, outGroupOffset,
	       kernGroupOffset, oPixels, n, k) ;
	  }
	  else {
	    k8<VEDNN_FILTER_LAYOUT_HWCN>(pIn, pKernel, pOut,
	       inChannel, inWidth, inHeight,
	       outChannel, outWidth, outHeight,
	       kernWidth, kernHeight,
	       inChannelGroup, outChannelGroup,
	       inGroupOffset, outGroupOffset,
	       kernGroupOffset, oPixels, n, k) ;
	  }
	  k+=8 ;
	}
	for ( ; k < outChannelGroup; k+=16) {
	  if( filter_layout == VEDNN_FILTER_LAYOUT_NCHW) {
	    k16<VEDNN_FILTER_LAYOUT_NCHW>(pIn, pKernel, pOut,
	       inChannel, inWidth, inHeight,
	       outChannel, outWidth, outHeight,
	       kernWidth, kernHeight,
	       inChannelGroup, outChannelGroup,
	       inGroupOffset, outGroupOffset,
	       kernGroupOffset, oPixels, n, k) ;
	  }
	  else {
	    k16<VEDNN_FILTER_LAYOUT_HWCN>(pIn, pKernel, pOut,
	       inChannel, inWidth, inHeight,
	       outChannel, outWidth, outHeight,
	       kernWidth, kernHeight,
	       inChannelGroup, outChannelGroup,
	       inGroupOffset, outGroupOffset,
	       kernGroupOffset, oPixels, n, k) ;
	  }
	} // outChannel
      } // group
    } // batch
  }

  return VEDNN_SUCCESS;
}

