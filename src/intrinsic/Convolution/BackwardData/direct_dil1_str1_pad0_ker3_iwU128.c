#include <stdint.h>
#include <stdlib.h>

#include "vednn.h"

#include "veintrin.h"
#define VLEN	(256)

vednnError_t
vednnConvolutionBackwardData_direct_dil1_str1_pad0_ker3_iwU128(
    const vednnTensorParam_t * restrict 	pParamGradOut,
    const void * restrict 			pDataGradOut,
    const vednnFilterParam_t * restrict 	pParamKernel,
    const void * restrict 			pDataKernel,
    const vednnConvolutionParam_t * restrict 	pParamConv,
    const vednnTensorParam_t * restrict 	pParamGradIn,
    void * restrict 				pDataGradIn
)
{
  const int64_t batch       = pParamGradOut->batch;
  const int64_t gOutChannel = pParamGradOut->channel;
  const int64_t gOutWidth   = pParamGradOut->width;
  const int64_t gOutHeight  = pParamGradOut->height;
  const int64_t gInChannel  = pParamGradIn->channel;
  const int64_t gInWidth    = pParamGradIn->width;
  const int64_t gInHeight   = pParamGradIn->height;
  const int64_t kernWidth   = pParamKernel->width;
  const int64_t kernHeight  = pParamKernel->height;

  const int64_t group          = pParamConv->group;
//  const int64_t strideWidth    = pParamConv->strideWidth;		// must be 1
//  const int64_t strideHeight   = pParamConv->strideHeight;		// must be 1
//  const int64_t padWidth       = pParamConv->padWidth;		// must be 0
//  const int64_t padHeight      = pParamConv->padHeight;		// must be 0
//  const int64_t dilationWidth  = pParamConv->dilationWidth;		// must be 1
//  const int64_t dilationHeight = pParamConv->dilationHeight;		// must be 1

  const int64_t gOutChannelGroup = gOutChannel  / group;
  const int64_t gInChannelGroup  = gInChannel / group;

  const float * restrict pGOut   = pDataGradOut;
  const float * restrict pKernel = pDataKernel;
  float * restrict const pGIn   = pDataGradIn;

  const int gInPixels= gInHeight*gInWidth ;

  /* intrinsic version 1 */
  {
    const int64_t nH = VLEN / gInWidth ;

    _ve_lvl(nH*gInWidth) ;
    __vr vrseq = _ve_vseq_v() ;
    __vr vrh  = _ve_vdivsl_vvs(vrseq, gInWidth) ;
    __vr vrw  = _ve_vsubsl_vvv(vrseq, _ve_vmulul_vsv(gInWidth,vrh)) ;
    __vr vrhw = _ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vrh), vrw) ;

    __vr vrx_s2 = _ve_vaddsl_vsv(-2, vrw) ;
    __vm256 vmx1_s2 = _ve_vfmkl_mcv(VECC_GE, vrx_s2) ;
    __vm256 vmx_s2 = vmx1_s2 ;

    __vr vrx_s1 = _ve_vaddsl_vsv(-1, vrw) ;
    __vm256 vmx1_s1 = _ve_vfmkl_mcv(VECC_GE, vrx_s1) ;
    __vm256 vmx2_s1 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(gOutWidth,vrx_s1)) ;
    __vm256 vmx_s1 = _ve_andm_mmm(vmx1_s1, vmx2_s1) ;

    __vr vrx_s0 = vrw ;
    __vm256 vmx2_s0 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(gOutWidth,vrx_s0)) ;
    __vm256 vmx_s0 = vmx2_s0 ;

    for (int64_t n=0; n<batch; n++) {
      for (int64_t g = 0; g < group; g++) {

	int64_t gInGroupOffset  = g * gInChannelGroup * gInHeight * gInWidth;
	int64_t gOutGroupOffset = g * gOutChannelGroup * gOutHeight * gOutWidth;
	int64_t kernGroupOffset = g * gOutChannelGroup * gInChannelGroup * kernHeight * kernWidth;

	int64_t k=0;
	if( (gInChannelGroup & 0x01 ) == 1 ) {

	  for (int64_t h=0; h<gInHeight; h+=nH) {
	    const int64_t vl = gInWidth * (gInHeight - h < nH ? gInHeight - h : nH) ;
	    const int64_t gip = h * gInWidth ;

	    const int64_t gInIndex = gInGroupOffset + ((n * gInChannel + k) * gInHeight ) * gInWidth + gip ;

	    _ve_lvl(vl) ;
	    __vr vrsum = _ve_vbrdu_vs_f32(0.0f) ;

	    __vr vry_r2 = _ve_vaddsl_vsv(h-2, vrh) ;
	    __vm256 vmy1_r2 = _ve_vfmkl_mcv(VECC_GE, vry_r2) ;
	    __vm256 vmy_r2 = vmy1_r2 ;

	    __vm256 vmall_r2s2 = _ve_andm_mmm(vmy_r2,vmx_s2) ;
	    __vm256 vmall_r2s1 = _ve_andm_mmm(vmy_r2,vmx_s1) ;
	    __vm256 vmall_r2s0 = _ve_andm_mmm(vmy_r2,vmx_s0) ;

	    __vr vry_r1 = _ve_vaddsl_vsv(h-1, vrh) ;
	    __vm256 vmy1_r1 = _ve_vfmkl_mcv(VECC_GE, vry_r1) ;
	    __vm256 vmy2_r1 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(gOutHeight,vry_r1)) ;
	    __vm256 vmy_r1 = _ve_andm_mmm(vmy1_r1, vmy2_r1) ;

	    __vm256 vmall_r1s2 = _ve_andm_mmm(vmy_r1,vmx_s2) ;
	    __vm256 vmall_r1s1 = _ve_andm_mmm(vmy_r1,vmx_s1) ;
	    __vm256 vmall_r1s0 = _ve_andm_mmm(vmy_r1,vmx_s0) ;

	    __vr vry_r0= _ve_vaddsl_vsv(h, vrh) ;
	    __vm256 vmy2_r0 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(gOutHeight,vry_r0)) ;
	    __vm256 vmy_r0 = vmy2_r0 ;

	    __vm256 vmall_r0s2 = _ve_andm_mmm(vmy_r0,vmx_s2) ;
	    __vm256 vmall_r0s1 = _ve_andm_mmm(vmy_r0,vmx_s1) ;
	    __vm256 vmall_r0s0 = _ve_andm_mmm(vmy_r0,vmx_s0) ;

	    for (int64_t c=0; c<gOutChannelGroup; c++) {
	      int64_t gOutIndex    = gOutGroupOffset + ((n * gOutChannel + c) * gOutHeight) * gOutWidth ;

	      const float *pKerValue = pKernel + kernGroupOffset + ((c * gInChannelGroup + k) * kernHeight + 2) * kernWidth + 2;

	      __vr vrgout_ptr_r2s2 = _ve_vsfa_vvss(vrhw, 2, (unsigned long)(pGOut+gOutIndex+(h-2)*gOutWidth-2)) ;
	      __vr vrgout_ptr_r1s2 = _ve_vsfa_vvss(vrhw, 2, (unsigned long)(pGOut+gOutIndex+(h-1)*gOutWidth-2)) ;
	      __vr vrgout_ptr_r0s2 = _ve_vsfa_vvss(vrhw, 2, (unsigned long)(pGOut+gOutIndex+(h-0)*gOutWidth-2)) ;

	      __vr vrgout_r2s2 = _ve_vgtu_vvm(vrgout_ptr_r2s2, vmall_r2s2) ;
	      __vr vrgout_r1s2 = _ve_vgtu_vvm(vrgout_ptr_r1s2, vmall_r1s2) ;
	      __vr vrgout_r0s2 = _ve_vgtu_vvm(vrgout_ptr_r0s2, vmall_r0s2) ;

	      __vr vrgout_r2s1 = _ve_vmv_vsv(1, vrgout_r2s2) ;
	      __vr vrgout_r2s0 = _ve_vmv_vsv(2, vrgout_r2s2) ;

	      __vr vrgout_r1s1 = _ve_vmv_vsv(1, vrgout_r1s2) ;
	      __vr vrgout_r1s0 = _ve_vmv_vsv(2, vrgout_r1s2) ;

	      __vr vrgout_r0s1 = _ve_vmv_vsv(1, vrgout_r0s2) ;
	      __vr vrgout_r0s0 = _ve_vmv_vsv(2, vrgout_r0s2) ;

	      vrgout_r2s2 = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), vrgout_r2s2, vmall_r2s2) ;
	      vrsum = _ve_vfmads_vvsv(vrsum, *pKerValue, vrgout_r2s2) ; pKerValue-- ;
	      vrgout_r2s1 = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), vrgout_r2s1, vmall_r2s1) ;
	      vrsum = _ve_vfmads_vvsv(vrsum, *pKerValue, vrgout_r2s1) ; pKerValue-- ;
	      vrgout_r2s0 = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), vrgout_r2s0, vmall_r2s0) ;
	      vrsum = _ve_vfmads_vvsv(vrsum, *pKerValue, vrgout_r2s0) ; pKerValue-- ;
	      vrgout_r1s2 = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), vrgout_r1s2, vmall_r1s2) ;
	      vrsum = _ve_vfmads_vvsv(vrsum, *pKerValue, vrgout_r1s2) ; pKerValue-- ;
	      vrgout_r1s1 = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), vrgout_r1s1, vmall_r1s1) ;
	      vrsum = _ve_vfmads_vvsv(vrsum, *pKerValue, vrgout_r1s1) ; pKerValue-- ;
	      vrgout_r1s0 = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), vrgout_r1s0, vmall_r1s0) ;
	      vrsum = _ve_vfmads_vvsv(vrsum, *pKerValue, vrgout_r1s0) ; pKerValue-- ;
	      vrgout_r1s2 = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), vrgout_r1s2, vmall_r1s2) ;
	      vrsum = _ve_vfmads_vvsv(vrsum, *pKerValue, vrgout_r0s2) ; pKerValue-- ;
	      vrgout_r1s1 = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), vrgout_r1s1, vmall_r1s1) ;
	      vrsum = _ve_vfmads_vvsv(vrsum, *pKerValue, vrgout_r0s1) ; pKerValue-- ;
	      vrgout_r0s0 = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), vrgout_r0s0, vmall_r0s0) ;
	      vrsum = _ve_vfmads_vvsv(vrsum, *pKerValue, vrgout_r0s0) ; pKerValue-- ;

	    } // gOutChannel

	    _ve_vstu_vss(vrsum, 4, pGIn+gInIndex) ;

	  } // gOutPixels


	  k++ ;
	}
	if( ((gInChannelGroup>>1) & 0x01 ) == 1 ) {

	  for (int64_t h=0; h<gInHeight; h+=nH) {
	    const int64_t vl = gInWidth * (gInHeight - h < nH ? gInHeight - h : nH) ;
	    const int64_t gip = h * gInWidth ;

	    const int64_t gInIndex = gInGroupOffset + ((n * gInChannel + k) * gInHeight ) * gInWidth + gip ;

	    _ve_lvl(vl) ;
	    __vr vrsum01 = _ve_pvbrd_vs_i64(0UL) ;

	    __vr vry_r2 = _ve_vaddsl_vsv(h-2, vrh) ;
	    __vm256 vmy1_r2 = _ve_vfmkl_mcv(VECC_GE, vry_r2) ;
	    __vm256 vmy_r2 = vmy1_r2 ;

	    __vm256 vmall_r2s2 = _ve_andm_mmm(vmy_r2,vmx_s2) ;
	    __vm256 vmall_r2s1 = _ve_andm_mmm(vmy_r2,vmx_s1) ;
	    __vm256 vmall_r2s0 = _ve_andm_mmm(vmy_r2,vmx_s0) ;

	    __vr vry_r1 = _ve_vaddsl_vsv(h-1, vrh) ;
	    __vm256 vmy1_r1 = _ve_vfmkl_mcv(VECC_GE, vry_r1) ;
	    __vm256 vmy2_r1 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(gOutHeight,vry_r1)) ;
	    __vm256 vmy_r1 = _ve_andm_mmm(vmy1_r1, vmy2_r1) ;

	    __vm256 vmall_r1s2 = _ve_andm_mmm(vmy_r1,vmx_s2) ;
	    __vm256 vmall_r1s1 = _ve_andm_mmm(vmy_r1,vmx_s1) ;
	    __vm256 vmall_r1s0 = _ve_andm_mmm(vmy_r1,vmx_s0) ;

	    __vr vry_r0= _ve_vaddsl_vsv(h, vrh) ;
	    __vm256 vmy2_r0 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(gOutHeight,vry_r0)) ;
	    __vm256 vmy_r0 = vmy2_r0 ;

	    __vm256 vmall_r0s2 = _ve_andm_mmm(vmy_r0,vmx_s2) ;
	    __vm256 vmall_r0s1 = _ve_andm_mmm(vmy_r0,vmx_s1) ;
	    __vm256 vmall_r0s0 = _ve_andm_mmm(vmy_r0,vmx_s0) ;

	    for (int64_t c=0; c<gOutChannelGroup; c++) {
	      int64_t gOutIndex    = gOutGroupOffset + ((n * gOutChannel + c) * gOutHeight) * gOutWidth ;

	      const float *pKerValue = pKernel + kernGroupOffset + ((c * gInChannelGroup + k) * kernHeight + 2) * kernWidth + 2;

	      __vr vrgout_ptr_r2s2 = _ve_vsfa_vvss(vrhw, 2, (unsigned long)(pGOut+gOutIndex+(h-2)*gOutWidth-2)) ;
	      __vr vrgout_ptr_r1s2 = _ve_vsfa_vvss(vrhw, 2, (unsigned long)(pGOut+gOutIndex+(h-1)*gOutWidth-2)) ;
	      __vr vrgout_ptr_r0s2 = _ve_vsfa_vvss(vrhw, 2, (unsigned long)(pGOut+gOutIndex+(h-0)*gOutWidth-2)) ;

	      __vr vrgout_r2s2 = _ve_vgtu_vvm(vrgout_ptr_r2s2, vmall_r2s2) ;
	      __vr vrgout_r1s2 = _ve_vgtu_vvm(vrgout_ptr_r1s2, vmall_r1s2) ;
	      __vr vrgout_r0s2 = _ve_vgtu_vvm(vrgout_ptr_r0s2, vmall_r0s2) ;

	      __vr vrgout_r2s1 = _ve_vmv_vsv(1, vrgout_r2s2) ;
	      __vr vrgout_r2s0 = _ve_vmv_vsv(2, vrgout_r2s2) ;

	      __vr vrgout_r1s1 = _ve_vmv_vsv(1, vrgout_r1s2) ;
	      __vr vrgout_r1s0 = _ve_vmv_vsv(2, vrgout_r1s2) ;

	      __vr vrgout_r0s1 = _ve_vmv_vsv(1, vrgout_r0s2) ;
	      __vr vrgout_r0s0 = _ve_vmv_vsv(2, vrgout_r0s2) ;

#define VFADD2(VRGOUT)									\
{											\
  __vr vrgoutP = _ve_vshf_vvvs(VRGOUT, VRGOUT, VE_VSHUFFLE_YUZU) ;			\
  const uint64_t kerValue01 = _ve_pack_f32p(pKerValue,					\
					    pKerValue + kernHeight * kernWidth ) ;	\
  vrsum01 = _ve_pvfmad_vvsv(vrsum01, kerValue01, vrgoutP) ;				\
}

	      vrgout_r2s2 = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), vrgout_r2s2, vmall_r2s2) ;
	      VFADD2(vrgout_r2s2) ; pKerValue-- ;
	      vrgout_r2s1 = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), vrgout_r2s1, vmall_r2s1) ;
	      VFADD2(vrgout_r2s1) ; pKerValue-- ;
	      vrgout_r2s0 = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), vrgout_r2s0, vmall_r2s0) ;
	      VFADD2(vrgout_r2s0) ; pKerValue-- ;

	      vrgout_r1s2 = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), vrgout_r1s2, vmall_r1s2) ;
	      VFADD2(vrgout_r1s2) ; pKerValue-- ;
	      vrgout_r1s1 = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), vrgout_r1s1, vmall_r1s1) ;
	      VFADD2(vrgout_r1s1) ; pKerValue-- ;
	      vrgout_r1s0 = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), vrgout_r1s0, vmall_r1s0) ;
	      VFADD2(vrgout_r1s0) ; pKerValue-- ;

	      vrgout_r0s2 = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), vrgout_r0s2, vmall_r0s2) ;
	      VFADD2(vrgout_r0s2) ; pKerValue-- ;
	      vrgout_r0s1 = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), vrgout_r0s1, vmall_r0s1) ;
	      VFADD2(vrgout_r0s1) ; pKerValue-- ;
	      vrgout_r0s0 = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), vrgout_r0s0, vmall_r0s0) ;
	      VFADD2(vrgout_r0s0) ; pKerValue-- ;
#undef VFADD2
	    } // gOutChannel

	    _ve_vstu_vss(vrsum01, 4, pGIn+gInIndex) ;
	    _ve_vstl_vss(vrsum01, 4, pGIn+gInIndex+  gInPixels) ;

	  } // gOutPixels


	  k+=2 ;
	}
	if( ((gInChannelGroup>>2) & 0x01 ) == 1 ) {

	  for (int64_t h=0; h<gInHeight; h+=nH) {
	    const int64_t vl = gInWidth * (gInHeight - h < nH ? gInHeight - h : nH) ;
	    const int64_t gip = h * gInWidth ;

	    const int64_t gInIndex = gInGroupOffset + ((n * gInChannel + k) * gInHeight ) * gInWidth + gip ;

	    _ve_lvl(vl) ;
	    __vr vrsum01 = _ve_pvbrd_vs_i64(0UL) ;
	    __vr vrsum23 = _ve_pvbrd_vs_i64(0UL) ;

	    __vr vry_r2 = _ve_vaddsl_vsv(h-2, vrh) ;
	    __vm256 vmy1_r2 = _ve_vfmkl_mcv(VECC_GE, vry_r2) ;
	    __vm256 vmy_r2 = vmy1_r2 ;

	    __vm256 vmall_r2s2 = _ve_andm_mmm(vmy_r2,vmx_s2) ;
	    __vm256 vmall_r2s1 = _ve_andm_mmm(vmy_r2,vmx_s1) ;
	    __vm256 vmall_r2s0 = _ve_andm_mmm(vmy_r2,vmx_s0) ;

	    __vr vry_r1 = _ve_vaddsl_vsv(h-1, vrh) ;
	    __vm256 vmy1_r1 = _ve_vfmkl_mcv(VECC_GE, vry_r1) ;
	    __vm256 vmy2_r1 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(gOutHeight,vry_r1)) ;
	    __vm256 vmy_r1 = _ve_andm_mmm(vmy1_r1, vmy2_r1) ;

	    __vm256 vmall_r1s2 = _ve_andm_mmm(vmy_r1,vmx_s2) ;
	    __vm256 vmall_r1s1 = _ve_andm_mmm(vmy_r1,vmx_s1) ;
	    __vm256 vmall_r1s0 = _ve_andm_mmm(vmy_r1,vmx_s0) ;

	    __vr vry_r0= _ve_vaddsl_vsv(h, vrh) ;
	    __vm256 vmy2_r0 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(gOutHeight,vry_r0)) ;
	    __vm256 vmy_r0 = vmy2_r0 ;

	    __vm256 vmall_r0s2 = _ve_andm_mmm(vmy_r0,vmx_s2) ;
	    __vm256 vmall_r0s1 = _ve_andm_mmm(vmy_r0,vmx_s1) ;
	    __vm256 vmall_r0s0 = _ve_andm_mmm(vmy_r0,vmx_s0) ;

	    for (int64_t c=0; c<gOutChannelGroup; c++) {
	      int64_t gOutIndex    = gOutGroupOffset + ((n * gOutChannel + c) * gOutHeight) * gOutWidth ;

	      const float *pKerValue = pKernel + kernGroupOffset + ((c * gInChannelGroup + k) * kernHeight + 2) * kernWidth + 2;

	      __vr vrgout_ptr_r2s2 = _ve_vsfa_vvss(vrhw, 2, (unsigned long)(pGOut+gOutIndex+(h-2)*gOutWidth-2)) ;
	      __vr vrgout_ptr_r1s2 = _ve_vsfa_vvss(vrhw, 2, (unsigned long)(pGOut+gOutIndex+(h-1)*gOutWidth-2)) ;
	      __vr vrgout_ptr_r0s2 = _ve_vsfa_vvss(vrhw, 2, (unsigned long)(pGOut+gOutIndex+(h-0)*gOutWidth-2)) ;

	      __vr vrgout_r2s2 = _ve_vgtu_vvm(vrgout_ptr_r2s2, vmall_r2s2) ;
	      __vr vrgout_r1s2 = _ve_vgtu_vvm(vrgout_ptr_r1s2, vmall_r1s2) ;
	      __vr vrgout_r0s2 = _ve_vgtu_vvm(vrgout_ptr_r0s2, vmall_r0s2) ;

	      __vr vrgout_r2s1 = _ve_vmv_vsv(1, vrgout_r2s2) ;
	      __vr vrgout_r2s0 = _ve_vmv_vsv(2, vrgout_r2s2) ;

	      __vr vrgout_r1s1 = _ve_vmv_vsv(1, vrgout_r1s2) ;
	      __vr vrgout_r1s0 = _ve_vmv_vsv(2, vrgout_r1s2) ;

	      __vr vrgout_r0s1 = _ve_vmv_vsv(1, vrgout_r0s2) ;
	      __vr vrgout_r0s0 = _ve_vmv_vsv(2, vrgout_r0s2) ;

#define VFADD4(VRGOUT)									\
{											\
  __vr vrgoutP = _ve_vshf_vvvs(VRGOUT, VRGOUT, VE_VSHUFFLE_YUZU) ;			\
  const uint64_t kerValue01 = _ve_pack_f32p(pKerValue,					\
					    pKerValue + kernHeight * kernWidth ) ;	\
  const uint64_t kerValue23 = _ve_pack_f32p(pKerValue + 2 * kernHeight * kernWidth,	\
					    pKerValue + 3 * kernHeight * kernWidth ) ;	\
  vrsum01 = _ve_pvfmad_vvsv(vrsum01, kerValue01, vrgoutP) ;		\
  vrsum23 = _ve_pvfmad_vvsv(vrsum23, kerValue23, vrgoutP) ;		\
}

	      vrgout_r2s2 = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), vrgout_r2s2, vmall_r2s2) ;
	      VFADD4(vrgout_r2s2) ; pKerValue-- ;
	      vrgout_r2s1 = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), vrgout_r2s1, vmall_r2s1) ;
	      VFADD4(vrgout_r2s1) ; pKerValue-- ;
	      vrgout_r2s0 = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), vrgout_r2s0, vmall_r2s0) ;
	      VFADD4(vrgout_r2s0) ; pKerValue-- ;

	      vrgout_r1s2 = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), vrgout_r1s2, vmall_r1s2) ;
	      VFADD4(vrgout_r1s2) ; pKerValue-- ;
	      vrgout_r1s1 = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), vrgout_r1s1, vmall_r1s1) ;
	      VFADD4(vrgout_r1s1) ; pKerValue-- ;
	      vrgout_r1s0 = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), vrgout_r1s0, vmall_r1s0) ;
	      VFADD4(vrgout_r1s0) ; pKerValue-- ;

	      vrgout_r0s2 = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), vrgout_r0s2, vmall_r0s2) ;
	      VFADD4(vrgout_r0s2) ; pKerValue-- ;
	      vrgout_r0s1 = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), vrgout_r0s1, vmall_r0s1) ;
	      VFADD4(vrgout_r0s1) ; pKerValue-- ;
	      vrgout_r0s0 = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), vrgout_r0s0, vmall_r0s0) ;
	      VFADD4(vrgout_r0s0) ; pKerValue-- ;
#undef VFADD4
	    } // gOutChannel

	    _ve_vstu_vss(vrsum01, 4, pGIn+gInIndex) ;
	    _ve_vstl_vss(vrsum01, 4, pGIn+gInIndex+  gInPixels) ;
	    _ve_vstu_vss(vrsum23, 4, pGIn+gInIndex+2*gInPixels) ;
	    _ve_vstl_vss(vrsum23, 4, pGIn+gInIndex+3*gInPixels) ;

	  } // gOutPixels


	  k+=4 ;
	}
	if( ((gInChannelGroup>>3) & 0x01 ) == 1 ) {

	  for (int64_t h=0; h<gInHeight; h+=nH) {
	    const int64_t vl = gInWidth * (gInHeight - h < nH ? gInHeight - h : nH) ;
	    const int64_t gip = h * gInWidth ;

	    const int64_t gInIndex = gInGroupOffset + ((n * gInChannel + k) * gInHeight ) * gInWidth + gip ;

	    _ve_lvl(vl) ;
	    __vr vrsum01 = _ve_pvbrd_vs_i64(0UL) ;
	    __vr vrsum23 = _ve_pvbrd_vs_i64(0UL) ;
	    __vr vrsum45 = _ve_pvbrd_vs_i64(0UL) ;
	    __vr vrsum67 = _ve_pvbrd_vs_i64(0UL) ;

	    __vr vry_r2 = _ve_vaddsl_vsv(h-2, vrh) ;
	    __vm256 vmy1_r2 = _ve_vfmkl_mcv(VECC_GE, vry_r2) ;
	    __vm256 vmy_r2 = vmy1_r2 ;

	    __vm256 vmall_r2s2 = _ve_andm_mmm(vmy_r2,vmx_s2) ;
	    __vm256 vmall_r2s1 = _ve_andm_mmm(vmy_r2,vmx_s1) ;
	    __vm256 vmall_r2s0 = _ve_andm_mmm(vmy_r2,vmx_s0) ;

	    __vr vry_r1 = _ve_vaddsl_vsv(h-1, vrh) ;
	    __vm256 vmy1_r1 = _ve_vfmkl_mcv(VECC_GE, vry_r1) ;
	    __vm256 vmy2_r1 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(gOutHeight,vry_r1)) ;
	    __vm256 vmy_r1 = _ve_andm_mmm(vmy1_r1, vmy2_r1) ;

	    __vm256 vmall_r1s2 = _ve_andm_mmm(vmy_r1,vmx_s2) ;
	    __vm256 vmall_r1s1 = _ve_andm_mmm(vmy_r1,vmx_s1) ;
	    __vm256 vmall_r1s0 = _ve_andm_mmm(vmy_r1,vmx_s0) ;

	    __vr vry_r0= _ve_vaddsl_vsv(h, vrh) ;
	    __vm256 vmy2_r0 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(gOutHeight,vry_r0)) ;
	    __vm256 vmy_r0 = vmy2_r0 ;

	    __vm256 vmall_r0s2 = _ve_andm_mmm(vmy_r0,vmx_s2) ;
	    __vm256 vmall_r0s1 = _ve_andm_mmm(vmy_r0,vmx_s1) ;
	    __vm256 vmall_r0s0 = _ve_andm_mmm(vmy_r0,vmx_s0) ;

	    for (int64_t c=0; c<gOutChannelGroup; c++) {
	      int64_t gOutIndex    = gOutGroupOffset + ((n * gOutChannel + c) * gOutHeight) * gOutWidth ;

	      const float *pKerValue = pKernel + kernGroupOffset + ((c * gInChannelGroup + k) * kernHeight + 2) * kernWidth + 2;

	      __vr vrgout_ptr_r2s2 = _ve_vsfa_vvss(vrhw, 2, (unsigned long)(pGOut+gOutIndex+(h-2)*gOutWidth-2)) ;
	      __vr vrgout_ptr_r1s2 = _ve_vsfa_vvss(vrhw, 2, (unsigned long)(pGOut+gOutIndex+(h-1)*gOutWidth-2)) ;
	      __vr vrgout_ptr_r0s2 = _ve_vsfa_vvss(vrhw, 2, (unsigned long)(pGOut+gOutIndex+(h-0)*gOutWidth-2)) ;

	      __vr vrgout_r2s2 = _ve_vgtu_vvm(vrgout_ptr_r2s2, vmall_r2s2) ;
	      __vr vrgout_r1s2 = _ve_vgtu_vvm(vrgout_ptr_r1s2, vmall_r1s2) ;
	      __vr vrgout_r0s2 = _ve_vgtu_vvm(vrgout_ptr_r0s2, vmall_r0s2) ;

	      __vr vrgout_r2s1 = _ve_vmv_vsv(1, vrgout_r2s2) ;
	      __vr vrgout_r2s0 = _ve_vmv_vsv(2, vrgout_r2s2) ;

	      __vr vrgout_r1s1 = _ve_vmv_vsv(1, vrgout_r1s2) ;
	      __vr vrgout_r1s0 = _ve_vmv_vsv(2, vrgout_r1s2) ;

	      __vr vrgout_r0s1 = _ve_vmv_vsv(1, vrgout_r0s2) ;
	      __vr vrgout_r0s0 = _ve_vmv_vsv(2, vrgout_r0s2) ;

#define VFADD8(VRGOUT)									\
{											\
  __vr vrgoutP = _ve_vshf_vvvs(VRGOUT, VRGOUT, VE_VSHUFFLE_YUZU) ;			\
  const uint64_t kerValue01 = _ve_pack_f32p(pKerValue,					\
					    pKerValue + kernHeight * kernWidth ) ;	\
  const uint64_t kerValue23 = _ve_pack_f32p(pKerValue + 2 * kernHeight * kernWidth,	\
					    pKerValue + 3 * kernHeight * kernWidth ) ;	\
  const uint64_t kerValue45 = _ve_pack_f32p(pKerValue + 4 * kernHeight * kernWidth,	\
					    pKerValue + 5 * kernHeight * kernWidth ) ;	\
  const uint64_t kerValue67 = _ve_pack_f32p(pKerValue + 6 * kernHeight * kernWidth,	\
					    pKerValue + 7 * kernHeight * kernWidth ) ;	\
  vrsum01 = _ve_pvfmad_vvsv(vrsum01, kerValue01, vrgoutP) ;		\
  vrsum23 = _ve_pvfmad_vvsv(vrsum23, kerValue23, vrgoutP) ;		\
  vrsum45 = _ve_pvfmad_vvsv(vrsum45, kerValue45, vrgoutP) ;		\
  vrsum67 = _ve_pvfmad_vvsv(vrsum67, kerValue67, vrgoutP) ;		\
}

	      vrgout_r2s2 = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), vrgout_r2s2, vmall_r2s2) ;
	      VFADD8(vrgout_r2s2) ; pKerValue-- ;
	      vrgout_r2s1 = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), vrgout_r2s1, vmall_r2s1) ;
	      VFADD8(vrgout_r2s1) ; pKerValue-- ;
	      vrgout_r2s0 = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), vrgout_r2s0, vmall_r2s0) ;
	      VFADD8(vrgout_r2s0) ; pKerValue-- ;

	      vrgout_r1s2 = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), vrgout_r1s2, vmall_r1s2) ;
	      VFADD8(vrgout_r1s2) ; pKerValue-- ;
	      vrgout_r1s1 = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), vrgout_r1s1, vmall_r1s1) ;
	      VFADD8(vrgout_r1s1) ; pKerValue-- ;
	      vrgout_r1s0 = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), vrgout_r1s0, vmall_r1s0) ;
	      VFADD8(vrgout_r1s0) ; pKerValue-- ;

	      vrgout_r0s2 = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), vrgout_r0s2, vmall_r0s2) ;
	      VFADD8(vrgout_r0s2) ; pKerValue-- ;
	      vrgout_r0s1 = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), vrgout_r0s1, vmall_r0s1) ;
	      VFADD8(vrgout_r0s1) ; pKerValue-- ;
	      vrgout_r0s0 = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), vrgout_r0s0, vmall_r0s0) ;
	      VFADD8(vrgout_r0s0) ; pKerValue-- ;
#undef VFADD8
	    } // gOutChannel

	    _ve_vstu_vss(vrsum01, 4, pGIn+gInIndex) ;
	    _ve_vstl_vss(vrsum01, 4, pGIn+gInIndex+  gInPixels) ;
	    _ve_vstu_vss(vrsum23, 4, pGIn+gInIndex+2*gInPixels) ;
	    _ve_vstl_vss(vrsum23, 4, pGIn+gInIndex+3*gInPixels) ;
	    _ve_vstu_vss(vrsum45, 4, pGIn+gInIndex+4*gInPixels) ;
	    _ve_vstl_vss(vrsum45, 4, pGIn+gInIndex+5*gInPixels) ;
	    _ve_vstu_vss(vrsum67, 4, pGIn+gInIndex+6*gInPixels) ;
	    _ve_vstl_vss(vrsum67, 4, pGIn+gInIndex+7*gInPixels) ;

	  } // gOutPixels

	  k+=8 ;
	}
	for (; k<gInChannelGroup; k+=16) {

	  for (int64_t h=0; h<gInHeight; h+=nH) {
	    const int64_t vl = gInWidth * (gInHeight - h < nH ? gInHeight - h : nH) ;
	    const int64_t gip = h * gInWidth ;

	    const int64_t gInIndex = gInGroupOffset + ((n * gInChannel + k) * gInHeight ) * gInWidth + gip ;

	    _ve_lvl(vl) ;
	    __vr vrsum01 = _ve_pvbrd_vs_i64(0UL) ;
	    __vr vrsum23 = _ve_pvbrd_vs_i64(0UL) ;
	    __vr vrsum45 = _ve_pvbrd_vs_i64(0UL) ;
	    __vr vrsum67 = _ve_pvbrd_vs_i64(0UL) ;
	    __vr vrsum89 = _ve_pvbrd_vs_i64(0UL) ;
	    __vr vrsumAB = _ve_pvbrd_vs_i64(0UL) ;
	    __vr vrsumCD = _ve_pvbrd_vs_i64(0UL) ;
	    __vr vrsumEF = _ve_pvbrd_vs_i64(0UL) ;

	    __vr vry_r2 = _ve_vaddsl_vsv(h-2, vrh) ;
	    __vm256 vmy1_r2 = _ve_vfmkl_mcv(VECC_GE, vry_r2) ;
	    __vm256 vmy_r2 = vmy1_r2 ;

	    __vm256 vmall_r2s2 = _ve_andm_mmm(vmy_r2,vmx_s2) ;
	    __vm256 vmall_r2s1 = _ve_andm_mmm(vmy_r2,vmx_s1) ;
	    __vm256 vmall_r2s0 = _ve_andm_mmm(vmy_r2,vmx_s0) ;

	    __vr vry_r1 = _ve_vaddsl_vsv(h-1, vrh) ;
	    __vm256 vmy1_r1 = _ve_vfmkl_mcv(VECC_GE, vry_r1) ;
	    __vm256 vmy2_r1 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(gOutHeight,vry_r1)) ;
	    __vm256 vmy_r1 = _ve_andm_mmm(vmy1_r1, vmy2_r1) ;

	    __vm256 vmall_r1s2 = _ve_andm_mmm(vmy_r1,vmx_s2) ;
	    __vm256 vmall_r1s1 = _ve_andm_mmm(vmy_r1,vmx_s1) ;
	    __vm256 vmall_r1s0 = _ve_andm_mmm(vmy_r1,vmx_s0) ;

	    __vr vry_r0= _ve_vaddsl_vsv(h, vrh) ;
	    __vm256 vmy2_r0 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(gOutHeight,vry_r0)) ;
	    __vm256 vmy_r0 = vmy2_r0 ;

	    __vm256 vmall_r0s2 = _ve_andm_mmm(vmy_r0,vmx_s2) ;
	    __vm256 vmall_r0s1 = _ve_andm_mmm(vmy_r0,vmx_s1) ;
	    __vm256 vmall_r0s0 = _ve_andm_mmm(vmy_r0,vmx_s0) ;

	    for (int64_t c=0; c<gOutChannelGroup; c++) {
	      int64_t gOutIndex    = gOutGroupOffset + ((n * gOutChannel + c) * gOutHeight) * gOutWidth ;

	      const float *pKerValue = pKernel + kernGroupOffset + ((c * gInChannelGroup + k) * kernHeight + 2) * kernWidth + 2;

	      __vr vrgout_ptr_r2s2 = _ve_vsfa_vvss(vrhw, 2, (unsigned long)(pGOut+gOutIndex+(h-2)*gOutWidth-2)) ;
	      __vr vrgout_ptr_r1s2 = _ve_vsfa_vvss(vrhw, 2, (unsigned long)(pGOut+gOutIndex+(h-1)*gOutWidth-2)) ;
	      __vr vrgout_ptr_r0s2 = _ve_vsfa_vvss(vrhw, 2, (unsigned long)(pGOut+gOutIndex+(h-0)*gOutWidth-2)) ;

	      __vr vrgout_r2s2 = _ve_vgtu_vvm(vrgout_ptr_r2s2, vmall_r2s2) ;
	      __vr vrgout_r1s2 = _ve_vgtu_vvm(vrgout_ptr_r1s2, vmall_r1s2) ;
	      __vr vrgout_r0s2 = _ve_vgtu_vvm(vrgout_ptr_r0s2, vmall_r0s2) ;

	      __vr vrgout_r2s1 = _ve_vmv_vsv(1, vrgout_r2s2) ;
	      __vr vrgout_r2s0 = _ve_vmv_vsv(2, vrgout_r2s2) ;

	      __vr vrgout_r1s1 = _ve_vmv_vsv(1, vrgout_r1s2) ;
	      __vr vrgout_r1s0 = _ve_vmv_vsv(2, vrgout_r1s2) ;

	      __vr vrgout_r0s1 = _ve_vmv_vsv(1, vrgout_r0s2) ;
	      __vr vrgout_r0s0 = _ve_vmv_vsv(2, vrgout_r0s2) ;

#define VFADD16(VRGOUT)									\
{											\
  __vr vrgoutP = _ve_vshf_vvvs(VRGOUT, VRGOUT, VE_VSHUFFLE_YUZU) ;			\
  const uint64_t kerValue01 = _ve_pack_f32p(pKerValue,					\
					    pKerValue + kernHeight * kernWidth ) ;	\
  const uint64_t kerValue23 = _ve_pack_f32p(pKerValue + 2 * kernHeight * kernWidth,	\
					    pKerValue + 3 * kernHeight * kernWidth ) ;	\
  const uint64_t kerValue45 = _ve_pack_f32p(pKerValue + 4 * kernHeight * kernWidth,	\
					    pKerValue + 5 * kernHeight * kernWidth ) ;	\
  const uint64_t kerValue67 = _ve_pack_f32p(pKerValue + 6 * kernHeight * kernWidth,	\
					    pKerValue + 7 * kernHeight * kernWidth ) ;	\
  const uint64_t kerValue89 = _ve_pack_f32p(pKerValue + 8 * kernHeight * kernWidth,	\
					    pKerValue + 9 * kernHeight * kernWidth ) ;	\
  const uint64_t kerValueAB = _ve_pack_f32p(pKerValue +10 * kernHeight * kernWidth,	\
					    pKerValue +11* kernHeight * kernWidth ) ;	\
  const uint64_t kerValueCD = _ve_pack_f32p(pKerValue +12 * kernHeight * kernWidth,	\
					    pKerValue +13 * kernHeight * kernWidth ) ;	\
  const uint64_t kerValueEF = _ve_pack_f32p(pKerValue +14 * kernHeight * kernWidth,	\
					    pKerValue +15 * kernHeight * kernWidth ) ;	\
  vrsum01 = _ve_pvfmad_vvsv(vrsum01, kerValue01, vrgoutP) ;		\
  vrsum23 = _ve_pvfmad_vvsv(vrsum23, kerValue23, vrgoutP) ;		\
  vrsum45 = _ve_pvfmad_vvsv(vrsum45, kerValue45, vrgoutP) ;		\
  vrsum67 = _ve_pvfmad_vvsv(vrsum67, kerValue67, vrgoutP) ;		\
  vrsum89 = _ve_pvfmad_vvsv(vrsum89, kerValue89, vrgoutP) ;		\
  vrsumAB = _ve_pvfmad_vvsv(vrsumAB, kerValueAB, vrgoutP) ;		\
  vrsumCD = _ve_pvfmad_vvsv(vrsumCD, kerValueCD, vrgoutP) ;		\
  vrsumEF = _ve_pvfmad_vvsv(vrsumEF, kerValueEF, vrgoutP) ;		\
}

	      vrgout_r2s2 = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), vrgout_r2s2, vmall_r2s2) ;
	      VFADD16(vrgout_r2s2) ; pKerValue-- ;
	      vrgout_r2s1 = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), vrgout_r2s1, vmall_r2s1) ;
	      VFADD16(vrgout_r2s1) ; pKerValue-- ;
	      vrgout_r2s0 = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), vrgout_r2s0, vmall_r2s0) ;
	      VFADD16(vrgout_r2s0) ; pKerValue-- ;

	      vrgout_r1s2 = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), vrgout_r1s2, vmall_r1s2) ;
	      VFADD16(vrgout_r1s2) ; pKerValue-- ;
	      vrgout_r1s1 = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), vrgout_r1s1, vmall_r1s1) ;
	      VFADD16(vrgout_r1s1) ; pKerValue-- ;
	      vrgout_r1s0 = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), vrgout_r1s0, vmall_r1s0) ;
	      VFADD16(vrgout_r1s0) ; pKerValue-- ;

	      vrgout_r0s2 = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), vrgout_r0s2, vmall_r0s2) ;
	      VFADD16(vrgout_r0s2) ; pKerValue-- ;
	      vrgout_r0s1 = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), vrgout_r0s1, vmall_r0s1) ;
	      VFADD16(vrgout_r0s1) ; pKerValue-- ;
	      vrgout_r0s0 = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), vrgout_r0s0, vmall_r0s0) ;
	      VFADD16(vrgout_r0s0) ; pKerValue-- ;
#undef VFADD16
	    } // gOutChannel

	    _ve_vstu_vss(vrsum01, 4, pGIn+gInIndex) ;
	    _ve_vstl_vss(vrsum01, 4, pGIn+gInIndex+  gInPixels) ;
	    _ve_vstu_vss(vrsum23, 4, pGIn+gInIndex+2*gInPixels) ;
	    _ve_vstl_vss(vrsum23, 4, pGIn+gInIndex+3*gInPixels) ;
	    _ve_vstu_vss(vrsum45, 4, pGIn+gInIndex+4*gInPixels) ;
	    _ve_vstl_vss(vrsum45, 4, pGIn+gInIndex+5*gInPixels) ;
	    _ve_vstu_vss(vrsum67, 4, pGIn+gInIndex+6*gInPixels) ;
	    _ve_vstl_vss(vrsum67, 4, pGIn+gInIndex+7*gInPixels) ;
	    _ve_vstu_vss(vrsum89, 4, pGIn+gInIndex+8*gInPixels) ;
	    _ve_vstl_vss(vrsum89, 4, pGIn+gInIndex+9*gInPixels) ;
	    _ve_vstu_vss(vrsumAB, 4, pGIn+gInIndex+10*gInPixels) ;
	    _ve_vstl_vss(vrsumAB, 4, pGIn+gInIndex+11*gInPixels) ;
	    _ve_vstu_vss(vrsumCD, 4, pGIn+gInIndex+12*gInPixels) ;
	    _ve_vstl_vss(vrsumCD, 4, pGIn+gInIndex+13*gInPixels) ;
	    _ve_vstu_vss(vrsumEF, 4, pGIn+gInIndex+14*gInPixels) ;
	    _ve_vstl_vss(vrsumEF, 4, pGIn+gInIndex+15*gInPixels) ;

	  } // gOutPixels
	} // gInChannel
      } // group
    } // batch
  }


  return VEDNN_SUCCESS;
}
