#include <stdint.h>
#include <stdlib.h>

#include "vednn.h"
#include "vednn_util.h"

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
  const int64_t nY,
  const __vr    vry,
  const __vm256 vmw_s0,
  const __vm256 vmw_s1,
  const __vm256 vmw_s3,
  const __vm256 vmw_s4,
  const int64_t n,
  const int64_t k
)
{
  int64_t outIndex0 = outGroupOffset + (n * outChannel + k  ) * oPixels ;

  for (int64_t y=0; y<outHeight; y+=nY)
  {
    const int64_t vl = outWidth * (outHeight - y < nY ? outHeight - y : nY) ;
    const int64_t op = y * outWidth ;

    __vr vrsum = _vel_vbrds_vsl(0.f, vl) ;

    for (int64_t r = 0; r < kernHeight; r++) {
      __vr vrh = _vel_vaddsl_vsvl(y+r-2, vry, vl) ;

      __vm256 vmh0 = _vel_vfmklge_mvl(vrh, vl) ;				// condition(0 <= h)
      __vm256 vmh1 = _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(inHeight,vrh, vl), vl) ;	// condition(h < inHeight)
      __vm256 vmh  = _vel_andm_mmm(vmh0, vmh1) ;

      __vm256 vmall_s0 = _vel_andm_mmm(vmh,vmw_s0) ;
      __vm256 vmall_s1 = _vel_andm_mmm(vmh,vmw_s1) ;
      __vm256 vmall_s2 = vmh ;
      __vm256 vmall_s3 = _vel_andm_mmm(vmh,vmw_s3) ;
      __vm256 vmall_s4 = _vel_andm_mmm(vmh,vmw_s4) ;

      for (int64_t c = 0; c < inChannelGroup; c++) {
	const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;

	/* memory access errors mihgt be caused */
	__vr vrin_s0 = _vel_vldu_vssl(4,&pInChannel[op+(r-2)*inWidth-2], vl) ;
	__vr vrin_s1 = _vel_vldu_vssl(4,&pInChannel[op+(r-2)*inWidth-1], vl) ;
	__vr vrin_s2 = _vel_vldu_vssl(4,&pInChannel[op+(r-2)*inWidth  ], vl) ;
	__vr vrin_s3 = _vel_vldu_vssl(4,&pInChannel[op+(r-2)*inWidth+1], vl) ;
	__vr vrin_s4 = _vel_vldu_vssl(4,&pInChannel[op+(r-2)*inWidth+2], vl) ;

#define FILTER_OFFSET(k,c,r,s) ( kernGroupOffset + filter_index<FLAYOUT>(k,c,r,s, inChannelGroup, outChannelGroup, kernHeight, kernWidth) )
#define VFMAD1(VRIN, VMR, R, S) {							\
VRIN = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), VRIN, VMR, vl) ;			\
vrsum = _vel_vfmads_vvsvl(vrsum, pKernel[FILTER_OFFSET(k,c,R,S)], VRIN, vl) ;		\
}
	VFMAD1(vrin_s0, vmall_s0, r, 0) ;
	VFMAD1(vrin_s1, vmall_s1, r, 1) ;
	VFMAD1(vrin_s2, vmall_s2, r, 2) ;
	VFMAD1(vrin_s3, vmall_s3, r, 3) ;
	VFMAD1(vrin_s4, vmall_s4, r, 4) ;
#undef VFMAD1
#undef FILTER_OFFSET
      } // inChannel
    } // kernHeight

    _vel_vstu_vssl(vrsum, 4, pOut+outIndex0, vl) ;

    outIndex0 += vl ;
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
  const int64_t nY,
  const __vr    vry,
  const __vm256 vmw_s0,
  const __vm256 vmw_s1,
  const __vm256 vmw_s3,
  const __vm256 vmw_s4,
  const int64_t n,
  const int64_t k
)
{
  int64_t outIndex0 = outGroupOffset + (n * outChannel + k  ) * oPixels ;
  int64_t outIndex1 = outGroupOffset + (n * outChannel + k+1) * oPixels ;

  for (int64_t y=0; y<outHeight; y+=nY)
  {
    const int64_t vl = outWidth * (outHeight - y < nY ? outHeight - y : nY) ;
    const int64_t op = y * outWidth ;

    __vr vrsum01 = _vel_pvbrd_vsl(0UL, vl) ;

    for (int64_t r = 0; r < kernHeight; r++) {
      __vr vrh = _vel_vaddsl_vsvl(y+r-2, vry, vl) ;

      __vm256 vmh0 = _vel_vfmklge_mvl(vrh, vl) ;				// condition(0 <= h)
      __vm256 vmh1 = _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(inHeight,vrh, vl), vl) ;	// condition(h < inHeight)
      __vm256 vmh  = _vel_andm_mmm(vmh0, vmh1) ;

      __vm256 vmall_s0 = _vel_andm_mmm(vmh,vmw_s0) ;
      __vm256 vmall_s1 = _vel_andm_mmm(vmh,vmw_s1) ;
      __vm256 vmall_s2 = vmh ;
      __vm256 vmall_s3 = _vel_andm_mmm(vmh,vmw_s3) ;
      __vm256 vmall_s4 = _vel_andm_mmm(vmh,vmw_s4) ;

      for (int64_t c = 0; c < inChannelGroup; c++) {
	const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;

	/* memory access errors mihgt be caused */
	__vr vrin_s0 = _vel_vldu_vssl(4,&pInChannel[op+(r-2)*inWidth-2], vl) ;
	__vr vrin_s1 = _vel_vldu_vssl(4,&pInChannel[op+(r-2)*inWidth-1], vl) ;
	__vr vrin_s2 = _vel_vldu_vssl(4,&pInChannel[op+(r-2)*inWidth  ], vl) ;
	__vr vrin_s3 = _vel_vldu_vssl(4,&pInChannel[op+(r-2)*inWidth+1], vl) ;
	__vr vrin_s4 = _vel_vldu_vssl(4,&pInChannel[op+(r-2)*inWidth+2], vl) ;

#define FILTER_OFFSET(k,c,r,s) ( kernGroupOffset + filter_index<FLAYOUT>(k,c,r,s, inChannelGroup, outChannelGroup, kernHeight, kernWidth) )
#define PVFMAD2(VRIN, VMR, R, S) {							\
VRIN = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), VRIN, VMR, vl) ;			\
__vr vrinP = _vel_vshf_vvvsl(VRIN, VRIN, VE_VSHUFFLE_YUZU, vl) ;			\
const uint64_t kerValue01 = _vel_pack_f32p(pKernel + FILTER_OFFSET(k+0,c,R,S),		\
		                           pKernel + FILTER_OFFSET(k+1,c,R,S)) ;	\
vrsum01 = _vel_pvfmad_vvsvl(vrsum01, kerValue01, vrinP, vl) ;				\
}
	PVFMAD2(vrin_s0, vmall_s0, r, 0) ;
	PVFMAD2(vrin_s1, vmall_s1, r, 1) ;
	PVFMAD2(vrin_s2, vmall_s2, r, 2) ;
	PVFMAD2(vrin_s3, vmall_s3, r, 3) ;
	PVFMAD2(vrin_s4, vmall_s4, r, 4) ;
#undef PVFMAD2
#undef FILTER_OFFSET
      } // inChannel
    } // kernHeight

    _vel_vstu_vssl(vrsum01, 4, pOut+outIndex0, vl) ;
    _vel_vstl_vssl(vrsum01, 4, pOut+outIndex1, vl) ;

    outIndex0 += vl ;
    outIndex1 += vl ;
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
  const int64_t nY,
  const __vr    vry,
  const __vm256 vmw_s0,
  const __vm256 vmw_s1,
  const __vm256 vmw_s3,
  const __vm256 vmw_s4,
  const int64_t n,
  const int64_t k
)
{
  int64_t outIndex0 = outGroupOffset + (n * outChannel + k  ) * oPixels ;
  int64_t outIndex1 = outGroupOffset + (n * outChannel + k+1) * oPixels ;
  int64_t outIndex2 = outGroupOffset + (n * outChannel + k+2) * oPixels ;
  int64_t outIndex3 = outGroupOffset + (n * outChannel + k+3) * oPixels ;

  for (int64_t y=0; y<outHeight; y+=nY)
  {
    const int64_t vl = outWidth * (outHeight - y < nY ? outHeight - y : nY) ;
    const int64_t op = y * outWidth ;

    __vr vrsum01 = _vel_pvbrd_vsl(0UL, vl) ;
    __vr vrsum23 = _vel_pvbrd_vsl(0UL, vl) ;

    for (int64_t r = 0; r < kernHeight; r++) {
      __vr vrh = _vel_vaddsl_vsvl(y+r-2, vry, vl) ;

      __vm256 vmh0 = _vel_vfmklge_mvl(vrh, vl) ;				// condition(0 <= h)
      __vm256 vmh1 = _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(inHeight,vrh, vl), vl) ;	// condition(h < inHeight)
      __vm256 vmh  = _vel_andm_mmm(vmh0, vmh1) ;

      __vm256 vmall_s0 = _vel_andm_mmm(vmh,vmw_s0) ;
      __vm256 vmall_s1 = _vel_andm_mmm(vmh,vmw_s1) ;
      __vm256 vmall_s2 = vmh ;
      __vm256 vmall_s3 = _vel_andm_mmm(vmh,vmw_s3) ;
      __vm256 vmall_s4 = _vel_andm_mmm(vmh,vmw_s4) ;

      for (int64_t c = 0; c < inChannelGroup; c++) {
	const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;

	/* memory access errors mihgt be caused */
	__vr vrin_s0 = _vel_vldu_vssl(4,&pInChannel[op+(r-2)*inWidth-2], vl) ;
	__vr vrin_s1 = _vel_vldu_vssl(4,&pInChannel[op+(r-2)*inWidth-1], vl) ;
	__vr vrin_s2 = _vel_vldu_vssl(4,&pInChannel[op+(r-2)*inWidth  ], vl) ;
	__vr vrin_s3 = _vel_vldu_vssl(4,&pInChannel[op+(r-2)*inWidth+1], vl) ;
	__vr vrin_s4 = _vel_vldu_vssl(4,&pInChannel[op+(r-2)*inWidth+2], vl) ;

#define FILTER_OFFSET(k,c,r,s) ( kernGroupOffset + filter_index<FLAYOUT>(k,c,r,s, inChannelGroup, outChannelGroup, kernHeight, kernWidth) )
#define PVFMAD4(VRIN, VMR, R, S) {							\
VRIN = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), VRIN, VMR, vl) ;			\
__vr vrinP = _vel_vshf_vvvsl(VRIN, VRIN, VE_VSHUFFLE_YUZU, vl) ;			\
const uint64_t kerValue01 = _vel_pack_f32p(pKernel + FILTER_OFFSET(k+0,c,R,S),		\
		                           pKernel + FILTER_OFFSET(k+1,c,R,S)) ;	\
const uint64_t kerValue23 = _vel_pack_f32p(pKernel + FILTER_OFFSET(k+2,c,R,S),		\
		                           pKernel + FILTER_OFFSET(k+3,c,R,S)) ;	\
vrsum01 = _vel_pvfmad_vvsvl(vrsum01, kerValue01, vrinP, vl) ;				\
vrsum23 = _vel_pvfmad_vvsvl(vrsum23, kerValue23, vrinP, vl) ;				\
}
	PVFMAD4(vrin_s0, vmall_s0, r, 0) ;
	PVFMAD4(vrin_s1, vmall_s1, r, 1) ;
	PVFMAD4(vrin_s2, vmall_s2, r, 2) ;
	PVFMAD4(vrin_s3, vmall_s3, r, 3) ;
	PVFMAD4(vrin_s4, vmall_s4, r, 4) ;
#undef PVFMAD4
#undef FILTER_OFFSET
      } // inChannel
    } // kernHeight

    _vel_vstu_vssl(vrsum01, 4, pOut+outIndex0, vl) ;
    _vel_vstl_vssl(vrsum01, 4, pOut+outIndex1, vl) ;
    _vel_vstu_vssl(vrsum23, 4, pOut+outIndex2, vl) ;
    _vel_vstl_vssl(vrsum23, 4, pOut+outIndex3, vl) ;

    outIndex0 += vl ;
    outIndex1 += vl ;
    outIndex2 += vl ;
    outIndex3 += vl ;
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
  const int64_t nY,
  const __vr    vry,
  const __vm256 vmw_s0,
  const __vm256 vmw_s1,
  const __vm256 vmw_s3,
  const __vm256 vmw_s4,
  const int64_t n,
  const int64_t k
)
{
  int64_t outIndex0 = outGroupOffset + (n * outChannel + k  ) * oPixels ;
  int64_t outIndex1 = outGroupOffset + (n * outChannel + k+1) * oPixels ;
  int64_t outIndex2 = outGroupOffset + (n * outChannel + k+2) * oPixels ;
  int64_t outIndex3 = outGroupOffset + (n * outChannel + k+3) * oPixels ;
  int64_t outIndex4 = outGroupOffset + (n * outChannel + k+4) * oPixels ;
  int64_t outIndex5 = outGroupOffset + (n * outChannel + k+5) * oPixels ;
  int64_t outIndex6 = outGroupOffset + (n * outChannel + k+6) * oPixels ;
  int64_t outIndex7 = outGroupOffset + (n * outChannel + k+7) * oPixels ;

  for (int64_t y=0; y<outHeight; y+=nY)
  {
    const int64_t vl = outWidth * (outHeight - y < nY ? outHeight - y : nY) ;
    const int64_t op = y * outWidth ;

    __vr vrsum01 = _vel_pvbrd_vsl(0UL, vl) ;
    __vr vrsum23 = _vel_pvbrd_vsl(0UL, vl) ;
    __vr vrsum45 = _vel_pvbrd_vsl(0UL, vl) ;
    __vr vrsum67 = _vel_pvbrd_vsl(0UL, vl) ;

    for (int64_t r = 0; r < kernHeight; r++) {
      __vr vrh = _vel_vaddsl_vsvl(y+r-2, vry, vl) ;

      __vm256 vmh0 = _vel_vfmklge_mvl(vrh, vl) ;				// condition(0 <= h)
      __vm256 vmh1 = _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(inHeight,vrh, vl), vl) ;	// condition(h < inHeight)
      __vm256 vmh  = _vel_andm_mmm(vmh0, vmh1) ;

      __vm256 vmall_s0 = _vel_andm_mmm(vmh,vmw_s0) ;
      __vm256 vmall_s1 = _vel_andm_mmm(vmh,vmw_s1) ;
      __vm256 vmall_s2 = vmh ;
      __vm256 vmall_s3 = _vel_andm_mmm(vmh,vmw_s3) ;
      __vm256 vmall_s4 = _vel_andm_mmm(vmh,vmw_s4) ;

      for (int64_t c = 0; c < inChannelGroup; c++) {
	const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;

	/* memory access errors mihgt be caused */
	__vr vrin_s0 = _vel_vldu_vssl(4,&pInChannel[op+(r-2)*inWidth-2], vl) ;
	__vr vrin_s1 = _vel_vldu_vssl(4,&pInChannel[op+(r-2)*inWidth-1], vl) ;
	__vr vrin_s2 = _vel_vldu_vssl(4,&pInChannel[op+(r-2)*inWidth  ], vl) ;
	__vr vrin_s3 = _vel_vldu_vssl(4,&pInChannel[op+(r-2)*inWidth+1], vl) ;
	__vr vrin_s4 = _vel_vldu_vssl(4,&pInChannel[op+(r-2)*inWidth+2], vl) ;

#define FILTER_OFFSET(k,c,r,s) ( kernGroupOffset + filter_index<FLAYOUT>(k,c,r,s, inChannelGroup, outChannelGroup, kernHeight, kernWidth) )
#define PVFMAD8(VRIN, VMR, R, S) {							\
VRIN = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), VRIN, VMR, vl) ;			\
__vr vrinP = _vel_vshf_vvvsl(VRIN, VRIN, VE_VSHUFFLE_YUZU, vl) ;			\
const uint64_t kerValue01 = _vel_pack_f32p(pKernel + FILTER_OFFSET(k+0,c,R,S),		\
		                           pKernel + FILTER_OFFSET(k+1,c,R,S)) ;	\
const uint64_t kerValue23 = _vel_pack_f32p(pKernel + FILTER_OFFSET(k+2,c,R,S),		\
		                           pKernel + FILTER_OFFSET(k+3,c,R,S)) ;	\
const uint64_t kerValue45 = _vel_pack_f32p(pKernel + FILTER_OFFSET(k+4,c,R,S),		\
		                           pKernel + FILTER_OFFSET(k+5,c,R,S)) ;	\
const uint64_t kerValue67 = _vel_pack_f32p(pKernel + FILTER_OFFSET(k+6,c,R,S),		\
		                           pKernel + FILTER_OFFSET(k+7,c,R,S)) ;	\
vrsum01 = _vel_pvfmad_vvsvl(vrsum01, kerValue01, vrinP, vl) ;				\
vrsum23 = _vel_pvfmad_vvsvl(vrsum23, kerValue23, vrinP, vl) ;				\
vrsum45 = _vel_pvfmad_vvsvl(vrsum45, kerValue45, vrinP, vl) ;				\
vrsum67 = _vel_pvfmad_vvsvl(vrsum67, kerValue67, vrinP, vl) ;				\
}
	PVFMAD8(vrin_s0, vmall_s0, r, 0) ;
	PVFMAD8(vrin_s1, vmall_s1, r, 1) ;
	PVFMAD8(vrin_s2, vmall_s2, r, 2) ;
	PVFMAD8(vrin_s3, vmall_s3, r, 3) ;
	PVFMAD8(vrin_s4, vmall_s4, r, 4) ;
#undef PVFMAD8
#undef FILTER_OFFSET
      } // inChannel
    } // kernHeight

    _vel_vstu_vssl(vrsum01, 4, pOut+outIndex0, vl) ;
    _vel_vstl_vssl(vrsum01, 4, pOut+outIndex1, vl) ;
    _vel_vstu_vssl(vrsum23, 4, pOut+outIndex2, vl) ;
    _vel_vstl_vssl(vrsum23, 4, pOut+outIndex3, vl) ;
    _vel_vstu_vssl(vrsum45, 4, pOut+outIndex4, vl) ;
    _vel_vstl_vssl(vrsum45, 4, pOut+outIndex5, vl) ;
    _vel_vstu_vssl(vrsum67, 4, pOut+outIndex6, vl) ;
    _vel_vstl_vssl(vrsum67, 4, pOut+outIndex7, vl) ;

    outIndex0 += vl ;
    outIndex1 += vl ;
    outIndex2 += vl ;
    outIndex3 += vl ;
    outIndex4 += vl ;
    outIndex5 += vl ;
    outIndex6 += vl ;
    outIndex7 += vl ;
  } // outPixels
}

extern "C" vednnError_t
vednnConvolutionForward_direct_dil1_str1_padsame_ker5_owU128(
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
  const int64_t kernWidth  = pParamKernel->width;		/* must be 2*padWidth  + 1 */
  const int64_t kernHeight = pParamKernel->height;		/* must be 2*padHeight + 1 */

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
    const int64_t nY = VLEN / outWidth ;

    __vr vrseq = _vel_vseq_vl(VLEN) ;			// xy
    __vr vry   = _vel_vdivsl_vvsl(vrseq, outWidth, VLEN) ;
    __vr vrx   = _vel_vsubsl_vvvl(vrseq, _vel_vmulul_vsvl(outWidth,vry, VLEN), VLEN) ;

    __vm256 vmw_s0 =  _vel_vfmklge_mvl(_vel_vaddsl_vsvl(-2,  vrx, VLEN), VLEN) ;		// condition(0 <= w)
    __vm256 vmw_s1 =  _vel_vfmklge_mvl(_vel_vaddsl_vsvl(-1,  vrx, VLEN), VLEN) ;		// condition(0 <= w)

    __vm256 vmw_s3 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(inWidth-1,vrx, VLEN), VLEN) ;	// condition(w < inWidth)
    __vm256 vmw_s4 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(inWidth-2,vrx, VLEN), VLEN) ;	// condition(w < inWidth)

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
	       inGroupOffset, outGroupOffset, kernGroupOffset,
	       oPixels, nY,
	       vry, vmw_s0, vmw_s1, vmw_s3, vmw_s4,
	       n, k );
	  }
	  else {
	    k1<VEDNN_FILTER_LAYOUT_HWCN>(pIn, pKernel, pOut,
	       inChannel, inWidth, inHeight,
	       outChannel, outWidth, outHeight,
	       kernWidth, kernHeight,
	       inChannelGroup, outChannelGroup,
	       inGroupOffset, outGroupOffset, kernGroupOffset,
	       oPixels, nY,
	       vry, vmw_s0, vmw_s1, vmw_s3, vmw_s4,
	       n, k );
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
	       inGroupOffset, outGroupOffset, kernGroupOffset,
	       oPixels, nY,
	       vry, vmw_s0, vmw_s1, vmw_s3, vmw_s4,
	       n, k );
	  }
	  else {
	    k2<VEDNN_FILTER_LAYOUT_HWCN>(pIn, pKernel, pOut,
	       inChannel, inWidth, inHeight,
	       outChannel, outWidth, outHeight,
	       kernWidth, kernHeight,
	       inChannelGroup, outChannelGroup,
	       inGroupOffset, outGroupOffset, kernGroupOffset,
	       oPixels, nY,
	       vry, vmw_s0, vmw_s1, vmw_s3, vmw_s4,
	       n, k );
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
	       inGroupOffset, outGroupOffset, kernGroupOffset,
	       oPixels, nY,
	       vry, vmw_s0, vmw_s1, vmw_s3, vmw_s4,
	       n, k );
	  }
	  else {
	    k4<VEDNN_FILTER_LAYOUT_HWCN>(pIn, pKernel, pOut,
	       inChannel, inWidth, inHeight,
	       outChannel, outWidth, outHeight,
	       kernWidth, kernHeight,
	       inChannelGroup, outChannelGroup,
	       inGroupOffset, outGroupOffset, kernGroupOffset,
	       oPixels, nY,
	       vry, vmw_s0, vmw_s1, vmw_s3, vmw_s4,
	       n, k );
	  }

	  k+=4 ;
	}
	for ( ; k < outChannelGroup; k+=8) {
	  if( filter_layout == VEDNN_FILTER_LAYOUT_NCHW) {
	    k8<VEDNN_FILTER_LAYOUT_NCHW>(pIn, pKernel, pOut,
	       inChannel, inWidth, inHeight,
	       outChannel, outWidth, outHeight,
	       kernWidth, kernHeight,
	       inChannelGroup, outChannelGroup,
	       inGroupOffset, outGroupOffset, kernGroupOffset,
	       oPixels, nY,
	       vry, vmw_s0, vmw_s1, vmw_s3, vmw_s4,
	       n, k );
	  }
	  else {
	    k8<VEDNN_FILTER_LAYOUT_HWCN>(pIn, pKernel, pOut,
	       inChannel, inWidth, inHeight,
	       outChannel, outWidth, outHeight,
	       kernWidth, kernHeight,
	       inChannelGroup, outChannelGroup,
	       inGroupOffset, outGroupOffset, kernGroupOffset,
	       oPixels, nY,
	       vry, vmw_s0, vmw_s1, vmw_s3, vmw_s4,
	       n, k );
	  }
	} // outChannel
      } // group
    } // batch
  }

  return VEDNN_SUCCESS;
}

