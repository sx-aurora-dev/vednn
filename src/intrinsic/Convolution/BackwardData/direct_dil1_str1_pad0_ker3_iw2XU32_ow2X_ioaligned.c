#include <stdint.h>
#include <stdlib.h>

#include "vednn.h"

#include "veintrin.h"
#define VLEN	(256)

vednnError_t
vednnConvolutionBackwardData_direct_dil1_str1_pad0_ker3_iw2XU32_ow2X_ioaligned(
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

  {
    const int64_t gInWidthHalf  = gInWidth >> 1 ;
    const int64_t gOutWidthHalf = gOutWidth >> 1 ;
    const int64_t nH = VLEN / gInWidthHalf ;

    _ve_lvl(VLEN) ;
    __vr vrseq = _ve_vseq_v() ;
    __vr vrh  = _ve_vdivsl_vvs(vrseq, gInWidthHalf) ;
    __vr vrw  = _ve_vmulsl_vsv(2, _ve_vsubsl_vvv(vrseq, _ve_vmulul_vsv(gInWidthHalf,vrh))) ;
    __vr vrhw = _ve_vaddsl_vvv(_ve_vmulsl_vsv(gOutWidth, vrh), vrw) ;

    __vr vrx_s2 = _ve_vaddsl_vsv(-2, vrw) ;
    __vm256 vmx1_s2 = _ve_vfmkl_mcv(VECC_GE, vrx_s2) ;
    __vm256 vmx_s2 = vmx1_s2 ;

    __vr vrx_s0 = vrw ;
    __vm256 vmx2_s0 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(gOutWidth,vrx_s0)) ;
    __vm256 vmx_s0 = vmx2_s0 ;


    // vector mask registers for vld2d ( using unroll >= 4 )
    __vm256 vm_s2, vm_s0 ;
    {
      __vr vrseq0  = _ve_vaddsl_vsv(15,vrseq) ;
      __vr vrh_s0  = _ve_vdivsl_vvs(vrseq0, 16) ;
      __vr vrw_s0  = _ve_vsubsl_vvv(vrseq0, _ve_vmulul_vsv(16,vrh_s0)) ;
      vm_s0 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(gInWidthHalf, vrw_s0)) ; // condition(x<gInWidthHalf)

      __vr vrh_s2  = _ve_vdivsl_vvs(vrseq, 16) ;
      __vr vrw_s2  = _ve_vsubsl_vvv(vrseq, _ve_vmulul_vsv(16,vrh_s2)) ;
      vm_s2 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(gInWidthHalf, vrw_s2)) ; // condition(x<gInWidthHalf)

    }

    for (int64_t n=0; n<batch; n++) {
      for (int64_t g = 0; g < group; g++) {

	int64_t gInGroupOffset  = g * gInChannelGroup * gInHeight * gInWidth;
	int64_t gOutGroupOffset = g * gOutChannelGroup * gOutHeight * gOutWidth;
	int64_t kernGroupOffset = g * gOutChannelGroup * gInChannelGroup * kernHeight * kernWidth;

	int64_t k=0;
	if( (gInChannelGroup & 0x01 ) == 1 ) {

	  for (int64_t h=0; h<gInHeight; h+=nH) {
	    const int64_t vl = gInWidthHalf * (gInHeight - h < nH ? gInHeight - h : nH) ;
	    const int64_t gip = h * gInWidth ;

	    const int64_t gInIndex = gInGroupOffset + ((n * gInChannel + k) * gInHeight ) * gInWidth + gip ;

	    _ve_lvl(vl) ;
	    __vr vrsum = _ve_vbrd_vs_i64(0UL) ;

	    __vr vry_r2 = _ve_vaddsl_vsv(h-2, vrh) ;
	    __vm256 vmy1_r2 = _ve_vfmkl_mcv(VECC_GE, vry_r2) ;
	    __vm256 vmy_r2 = vmy1_r2 ;

	    __vm256 vmall_r2s2 = _ve_andm_mmm(vmy_r2,vmx_s2) ;
	    __vm256 vmall_r2s0 = _ve_andm_mmm(vmy_r2,vmx_s0) ;

	    __vr vry_r1 = _ve_vaddsl_vsv(h-1, vrh) ;
	    __vm256 vmy1_r1 = _ve_vfmkl_mcv(VECC_GE, vry_r1) ;
	    __vm256 vmy2_r1 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(gOutHeight,vry_r1)) ;
	    __vm256 vmy_r1 = _ve_andm_mmm(vmy1_r1, vmy2_r1) ;

	    __vm256 vmall_r1s2 = _ve_andm_mmm(vmy_r1,vmx_s2) ;
	    __vm256 vmall_r1s0 = _ve_andm_mmm(vmy_r1,vmx_s0) ;

	    __vr vry_r0= _ve_vaddsl_vsv(h, vrh) ;
	    __vm256 vmy2_r0 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(gOutHeight,vry_r0)) ;
	    __vm256 vmy_r0 = vmy2_r0 ;

	    __vm256 vmall_r0s2 = _ve_andm_mmm(vmy_r0,vmx_s2) ;
	    __vm256 vmall_r0s0 = _ve_andm_mmm(vmy_r0,vmx_s0) ;

	    for (int64_t c=0; c<gOutChannelGroup; c++) {
	      int64_t gOutIndex    = gOutGroupOffset + ((n * gOutChannel + c) * gOutHeight) * gOutWidth ;

	      const float *pKerValue = pKernel + kernGroupOffset + ((c * gInChannelGroup + k) * kernHeight + 2) * kernWidth + 2;

	      __vr vrgout_ptr_r2s2 = _ve_vsfa_vvss(vrhw, 2, (unsigned long)(pGOut+gOutIndex+(h-2)*gOutWidth-2)) ;
	      __vr vrgout_ptr_r1s2 = _ve_vsfa_vvss(vrhw, 2, (unsigned long)(pGOut+gOutIndex+(h-1)*gOutWidth-2)) ;
	      __vr vrgout_ptr_r0s2 = _ve_vsfa_vvss(vrhw, 2, (unsigned long)(pGOut+gOutIndex+(h-0)*gOutWidth-2)) ;

	      __vr vrgout_r2s2 = _ve_vgt_vvm(vrgout_ptr_r2s2, vmall_r2s2) ;
	      __vr vrgout_r1s2 = _ve_vgt_vvm(vrgout_ptr_r1s2, vmall_r1s2) ;
	      __vr vrgout_r0s2 = _ve_vgt_vvm(vrgout_ptr_r0s2, vmall_r0s2) ;

	      __vr vrgout_r2s0 = _ve_vmv_vsv(1, vrgout_r2s2) ;
	      __vr vrgout_r1s0 = _ve_vmv_vsv(1, vrgout_r1s2) ;
	      __vr vrgout_r0s0 = _ve_vmv_vsv(1, vrgout_r0s2) ;

	      vrgout_r2s2 = _ve_vmrg_vvvm(_ve_vbrd_vs_i64(0UL), vrgout_r2s2, vmall_r2s2) ;
	      vrgout_r2s0 = _ve_vmrg_vvvm(_ve_vbrd_vs_i64(0UL), vrgout_r2s0, vmall_r2s0) ;
	      __vr vrgout_r2s1 = _ve_vshf_vvvs(vrgout_r2s2, vrgout_r2s0, VE_VSHUFFLE_ZLYU) ;
	      vrsum = _ve_pvfmad_vvsv(vrsum, _ve_pack_f32a(pKerValue), vrgout_r2s2) ; pKerValue-- ;
	      vrsum = _ve_pvfmad_vvsv(vrsum, _ve_pack_f32a(pKerValue), vrgout_r2s1) ; pKerValue-- ;
	      vrsum = _ve_pvfmad_vvsv(vrsum, _ve_pack_f32a(pKerValue), vrgout_r2s0) ; pKerValue-- ;

	      vrgout_r1s2 = _ve_vmrg_vvvm(_ve_vbrd_vs_i64(0UL), vrgout_r1s2, vmall_r1s2) ;
	      vrgout_r1s0 = _ve_vmrg_vvvm(_ve_vbrd_vs_i64(0UL), vrgout_r1s0, vmall_r1s0) ;
	      __vr vrgout_r1s1 = _ve_vshf_vvvs(vrgout_r1s2, vrgout_r1s0, VE_VSHUFFLE_ZLYU) ;

	      vrsum = _ve_pvfmad_vvsv(vrsum, _ve_pack_f32a(pKerValue), vrgout_r1s2) ; pKerValue-- ;
	      vrsum = _ve_pvfmad_vvsv(vrsum, _ve_pack_f32a(pKerValue), vrgout_r1s1) ; pKerValue-- ;
	      vrsum = _ve_pvfmad_vvsv(vrsum, _ve_pack_f32a(pKerValue), vrgout_r1s0) ; pKerValue-- ;

	      vrgout_r0s2 = _ve_vmrg_vvvm(_ve_vbrd_vs_i64(0UL), vrgout_r0s2, vmall_r0s2) ;
	      vrgout_r0s0 = _ve_vmrg_vvvm(_ve_vbrd_vs_i64(0UL), vrgout_r0s0, vmall_r0s0) ;
	      __vr vrgout_r0s1 = _ve_vshf_vvvs(vrgout_r0s2, vrgout_r0s0, VE_VSHUFFLE_ZLYU) ;

	      vrsum = _ve_pvfmad_vvsv(vrsum, _ve_pack_f32a(pKerValue), vrgout_r0s2) ; pKerValue-- ;
	      vrsum = _ve_pvfmad_vvsv(vrsum, _ve_pack_f32a(pKerValue), vrgout_r0s1) ; pKerValue-- ;
	      vrsum = _ve_pvfmad_vvsv(vrsum, _ve_pack_f32a(pKerValue), vrgout_r0s0) ; pKerValue-- ;

	    } // gOutChannel

	    _ve_vst_vss(vrsum, 8, pGIn+gInIndex) ;

	  } // gOutPixels


	  k++ ;
	}
	if( ((gInChannelGroup>>1) & 0x01 ) == 1 ) {
	  for (int64_t h=0; h<gInHeight; h+=nH) {
	    const int64_t vl = gInWidthHalf * (gInHeight - h < nH ? gInHeight - h : nH) ;
	    const int64_t gip = h * gInWidth ;

	    const int64_t gInIndex = gInGroupOffset + ((n * gInChannel + k) * gInHeight ) * gInWidth + gip ;

	    _ve_lvl(vl) ;
	    __vr vrsum0 = _ve_pvbrd_vs_i64(0UL) ;
	    __vr vrsum1 = _ve_pvbrd_vs_i64(0UL) ;

	    __vr vry_r2 = _ve_vaddsl_vsv(h-2, vrh) ;
	    __vm256 vmy1_r2 = _ve_vfmkl_mcv(VECC_GE, vry_r2) ;
	    __vm256 vmy_r2 = vmy1_r2 ;

	    __vm256 vmall_r2s2 = _ve_andm_mmm(vmy_r2,vmx_s2) ;
	    __vm256 vmall_r2s0 = _ve_andm_mmm(vmy_r2,vmx_s0) ;

	    __vr vry_r1 = _ve_vaddsl_vsv(h-1, vrh) ;
	    __vm256 vmy1_r1 = _ve_vfmkl_mcv(VECC_GE, vry_r1) ;
	    __vm256 vmy2_r1 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(gOutHeight,vry_r1)) ;
	    __vm256 vmy_r1 = _ve_andm_mmm(vmy1_r1, vmy2_r1) ;

	    __vm256 vmall_r1s2 = _ve_andm_mmm(vmy_r1,vmx_s2) ;
	    __vm256 vmall_r1s0 = _ve_andm_mmm(vmy_r1,vmx_s0) ;

	    __vr vry_r0= _ve_vaddsl_vsv(h, vrh) ;
	    __vm256 vmy2_r0 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(gOutHeight,vry_r0)) ;
	    __vm256 vmy_r0 = vmy2_r0 ;

	    __vm256 vmall_r0s2 = _ve_andm_mmm(vmy_r0,vmx_s2) ;
	    __vm256 vmall_r0s0 = _ve_andm_mmm(vmy_r0,vmx_s0) ;

	    for (int64_t c=0; c<gOutChannelGroup; c++) {
	      int64_t gOutIndex    = gOutGroupOffset + ((n * gOutChannel + c) * gOutHeight) * gOutWidth ;

	      const float *pKerValue = pKernel + kernGroupOffset + ((c * gInChannelGroup + k) * kernHeight + 2) * kernWidth + 2;

	      __vr vrgout_ptr_r2s2 = _ve_vsfa_vvss(vrhw, 2, (unsigned long)(pGOut+gOutIndex+(h-2)*gOutWidth-2)) ;
	      __vr vrgout_ptr_r1s2 = _ve_vsfa_vvss(vrhw, 2, (unsigned long)(pGOut+gOutIndex+(h-1)*gOutWidth-2)) ;
	      __vr vrgout_ptr_r0s2 = _ve_vsfa_vvss(vrhw, 2, (unsigned long)(pGOut+gOutIndex+(h-0)*gOutWidth-2)) ;

	      __vr vrgout_r2s2 = _ve_vgt_vvm(vrgout_ptr_r2s2, vmall_r2s2) ;
	      __vr vrgout_r1s2 = _ve_vgt_vvm(vrgout_ptr_r1s2, vmall_r1s2) ;
	      __vr vrgout_r0s2 = _ve_vgt_vvm(vrgout_ptr_r0s2, vmall_r0s2) ;

	      __vr vrgout_r2s0 = _ve_vmv_vsv(1, vrgout_r2s2) ;
	      __vr vrgout_r1s0 = _ve_vmv_vsv(1, vrgout_r1s2) ;
	      __vr vrgout_r0s0 = _ve_vmv_vsv(1, vrgout_r0s2) ;

#define VFADD2(VRGOUT)											\
{													\
  vrsum0 = _ve_pvfmad_vvsv(vrsum0, _ve_pack_f32a(pKerValue), VRGOUT) ;					\
  vrsum1 = _ve_pvfmad_vvsv(vrsum1, _ve_pack_f32a(pKerValue + kernHeight * kernWidth ), VRGOUT) ;	\
}

	      vrgout_r2s2 = _ve_vmrg_vvvm(_ve_vbrd_vs_i64(0UL), vrgout_r2s2, vmall_r2s2) ;
	      vrgout_r2s0 = _ve_vmrg_vvvm(_ve_vbrd_vs_i64(0UL), vrgout_r2s0, vmall_r2s0) ;
	      __vr vrgout_r2s1 = _ve_vshf_vvvs(vrgout_r2s2, vrgout_r2s0, VE_VSHUFFLE_ZLYU) ;
	      VFADD2(vrgout_r2s2) ; pKerValue-- ;
	      VFADD2(vrgout_r2s1) ; pKerValue-- ;
	      VFADD2(vrgout_r2s0) ; pKerValue-- ;

	      vrgout_r1s2 = _ve_vmrg_vvvm(_ve_vbrd_vs_i64(0UL), vrgout_r1s2, vmall_r1s2) ;
	      vrgout_r1s0 = _ve_vmrg_vvvm(_ve_vbrd_vs_i64(0UL), vrgout_r1s0, vmall_r1s0) ;
	      __vr vrgout_r1s1 = _ve_vshf_vvvs(vrgout_r1s2, vrgout_r1s0, VE_VSHUFFLE_ZLYU) ;
	      VFADD2(vrgout_r1s2) ; pKerValue-- ;
	      VFADD2(vrgout_r1s1) ; pKerValue-- ;
	      VFADD2(vrgout_r1s0) ; pKerValue-- ;

	      vrgout_r0s2 = _ve_vmrg_vvvm(_ve_vbrd_vs_i64(0UL), vrgout_r0s2, vmall_r0s2) ;
	      vrgout_r0s0 = _ve_vmrg_vvvm(_ve_vbrd_vs_i64(0UL), vrgout_r0s0, vmall_r0s0) ;
	      __vr vrgout_r0s1 = _ve_vshf_vvvs(vrgout_r0s2, vrgout_r0s0, VE_VSHUFFLE_ZLYU) ;
	      VFADD2(vrgout_r0s2) ; pKerValue-- ;
	      VFADD2(vrgout_r0s1) ; pKerValue-- ;
	      VFADD2(vrgout_r0s0) ; pKerValue-- ;
#undef VFADD2
	    } // gOutChannel

	    _ve_vst_vss(vrsum0, 8, pGIn+gInIndex) ;
	    _ve_vst_vss(vrsum1, 8, pGIn+gInIndex+  gInPixels) ;

	  } // gOutPixels

	  k+=2 ;
	}
	if( ((gInChannelGroup>>2) & 0x01 ) == 1 ) {
	  for (int64_t h=0; h<gInHeight; h+=16) {
	    const int64_t vl0 = 16 * (gInHeight - h < 16 ? gInHeight - h : 16) ;
	    const int64_t vl1 = gInWidthHalf * (gInHeight - h < 16 ? gInHeight - h : 16) ;

	    const int64_t gip = h * gInWidth ;

	    const int64_t gInIndex = gInGroupOffset + ((n * gInChannel + k) * gInHeight ) * gInWidth + gip ;

	    _ve_lvl(vl1) ;
	    __vr vrsum0 = _ve_pvbrd_vs_i64(0UL) ;
	    __vr vrsum1 = _ve_pvbrd_vs_i64(0UL) ;
	    __vr vrsum2 = _ve_pvbrd_vs_i64(0UL) ;
	    __vr vrsum3 = _ve_pvbrd_vs_i64(0UL) ;

	    __vr vry_r2 = _ve_vaddsl_vsv(h-2, vrh) ;
	    __vm256 vmy1_r2 = _ve_vfmkl_mcv(VECC_GE, vry_r2) ;
	    __vm256 vmy_r2 = vmy1_r2 ;

	    __vm256 vmall_r2s2 = _ve_andm_mmm(vmy_r2,vmx_s2) ;
	    __vm256 vmall_r2s0 = _ve_andm_mmm(vmy_r2,vmx_s0) ;

	    __vr vry_r1 = _ve_vaddsl_vsv(h-1, vrh) ;
	    __vm256 vmy1_r1 = _ve_vfmkl_mcv(VECC_GE, vry_r1) ;
	    __vm256 vmy2_r1 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(gOutHeight,vry_r1)) ;
	    __vm256 vmy_r1 = _ve_andm_mmm(vmy1_r1, vmy2_r1) ;

	    __vm256 vmall_r1s2 = _ve_andm_mmm(vmy_r1,vmx_s2) ;
	    __vm256 vmall_r1s0 = _ve_andm_mmm(vmy_r1,vmx_s0) ;

	    __vr vry_r0= _ve_vaddsl_vsv(h, vrh) ;
	    __vm256 vmy2_r0 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(gOutHeight,vry_r0)) ;
	    __vm256 vmy_r0 = vmy2_r0 ;

	    __vm256 vmall_r0s2 = _ve_andm_mmm(vmy_r0,vmx_s2) ;
	    __vm256 vmall_r0s0 = _ve_andm_mmm(vmy_r0,vmx_s0) ;

	    for (int64_t c=0; c<gOutChannelGroup; c++) {
	      int64_t gOutIndex    = gOutGroupOffset + ((n * gOutChannel + c) * gOutHeight) * gOutWidth ;

	      const float *pKerValue = pKernel + kernGroupOffset + ((c * gInChannelGroup + k) * kernHeight + 2) * kernWidth + 2;

	      _ve_lvl(vl0) ;
	      __vr vrgout_r2 = _ve_vld2d_vss((gOutWidthHalf<<(3+16))+8, pGOut+gOutIndex+(h-2)*gOutWidth-2) ;
	      __vr vrgout_r1 = _ve_vld2d_vss((gOutWidthHalf<<(3+16))+8, pGOut+gOutIndex+(h-1)*gOutWidth-2) ;
	      __vr vrgout_r0 = _ve_vld2d_vss((gOutWidthHalf<<(3+16))+8, pGOut+gOutIndex+(h-0)*gOutWidth-2) ;

	      __vr vrgout_r2s2 = _ve_vcp_vvmv(vrgout_r2, vm_s2, _ve_vbrd_vs_i64(0UL)) ;
	      __vr vrgout_r2s0 = _ve_vcp_vvmv(vrgout_r2, vm_s0, _ve_vbrd_vs_i64(0UL)) ;

	      __vr vrgout_r1s2 = _ve_vcp_vvmv(vrgout_r1, vm_s2, _ve_vbrd_vs_i64(0UL)) ;
	      __vr vrgout_r1s0 = _ve_vcp_vvmv(vrgout_r1, vm_s0, _ve_vbrd_vs_i64(0UL)) ;

	      __vr vrgout_r0s2 = _ve_vcp_vvmv(vrgout_r0, vm_s2, _ve_vbrd_vs_i64(0UL)) ;
	      __vr vrgout_r0s0 = _ve_vcp_vvmv(vrgout_r0, vm_s0, _ve_vbrd_vs_i64(0UL)) ;

	      _ve_lvl(vl1) ;
#define VFADD4(VRGOUT)											\
{													\
  vrsum0 = _ve_pvfmad_vvsv(vrsum0, _ve_pack_f32a(pKerValue),                               VRGOUT) ;	\
  vrsum1 = _ve_pvfmad_vvsv(vrsum1, _ve_pack_f32a(pKerValue +     kernHeight * kernWidth ), VRGOUT) ;	\
  vrsum2 = _ve_pvfmad_vvsv(vrsum2, _ve_pack_f32a(pKerValue + 2 * kernHeight * kernWidth ), VRGOUT) ;	\
  vrsum3 = _ve_pvfmad_vvsv(vrsum3, _ve_pack_f32a(pKerValue + 3 * kernHeight * kernWidth ), VRGOUT) ;	\
}

	      vrgout_r2s2 = _ve_vmrg_vvvm(_ve_vbrd_vs_i64(0UL), vrgout_r2s2, vmall_r2s2) ;
	      vrgout_r2s0 = _ve_vmrg_vvvm(_ve_vbrd_vs_i64(0UL), vrgout_r2s0, vmall_r2s0) ;
	      __vr vrgout_r2s1 = _ve_vshf_vvvs(vrgout_r2s2, vrgout_r2s0, VE_VSHUFFLE_ZLYU) ;
	      VFADD4(vrgout_r2s2) ; pKerValue-- ;
	      VFADD4(vrgout_r2s1) ; pKerValue-- ;
	      VFADD4(vrgout_r2s0) ; pKerValue-- ;

	      vrgout_r1s2 = _ve_vmrg_vvvm(_ve_vbrd_vs_i64(0UL), vrgout_r1s2, vmall_r1s2) ;
	      vrgout_r1s0 = _ve_vmrg_vvvm(_ve_vbrd_vs_i64(0UL), vrgout_r1s0, vmall_r1s0) ;
	      __vr vrgout_r1s1 = _ve_vshf_vvvs(vrgout_r1s2, vrgout_r1s0, VE_VSHUFFLE_ZLYU) ;
	      VFADD4(vrgout_r1s2) ; pKerValue-- ;
	      VFADD4(vrgout_r1s1) ; pKerValue-- ;
	      VFADD4(vrgout_r1s0) ; pKerValue-- ;

	      vrgout_r0s2 = _ve_vmrg_vvvm(_ve_vbrd_vs_i64(0UL), vrgout_r0s2, vmall_r0s2) ;
	      vrgout_r0s0 = _ve_vmrg_vvvm(_ve_vbrd_vs_i64(0UL), vrgout_r0s0, vmall_r0s0) ;
	      __vr vrgout_r0s1 = _ve_vshf_vvvs(vrgout_r0s2, vrgout_r0s0, VE_VSHUFFLE_ZLYU) ;
	      VFADD4(vrgout_r0s2) ; pKerValue-- ;
	      VFADD4(vrgout_r0s1) ; pKerValue-- ;
	      VFADD4(vrgout_r0s0) ; pKerValue-- ;
#undef VFADD4
	    } // gOutChannel

	    _ve_vst_vss(vrsum0, 8, pGIn+gInIndex) ;
	    _ve_vst_vss(vrsum1, 8, pGIn+gInIndex+  gInPixels) ;
	    _ve_vst_vss(vrsum2, 8, pGIn+gInIndex+2*gInPixels) ;
	    _ve_vst_vss(vrsum3, 8, pGIn+gInIndex+3*gInPixels) ;

	  } // gOutPixels

	  k+=4 ;
	}
	if( ((gInChannelGroup>>3) & 0x01 ) == 1 ) {
	  for (int64_t h=0; h<gInHeight; h+=16) {
	    const int64_t vl0 = 16 * (gInHeight - h < 16 ? gInHeight - h : 16) ;
	    const int64_t vl1 = gInWidthHalf * (gInHeight - h < 16 ? gInHeight - h : 16) ;

	    const int64_t gip = h * gInWidth ;

	    const int64_t gInIndex = gInGroupOffset + ((n * gInChannel + k) * gInHeight ) * gInWidth + gip ;

	    _ve_lvl(vl1) ;
	    __vr vrsum0 = _ve_pvbrd_vs_i64(0UL) ;
	    __vr vrsum1 = _ve_pvbrd_vs_i64(0UL) ;
	    __vr vrsum2 = _ve_pvbrd_vs_i64(0UL) ;
	    __vr vrsum3 = _ve_pvbrd_vs_i64(0UL) ;
	    __vr vrsum4 = _ve_pvbrd_vs_i64(0UL) ;
	    __vr vrsum5 = _ve_pvbrd_vs_i64(0UL) ;
	    __vr vrsum6 = _ve_pvbrd_vs_i64(0UL) ;
	    __vr vrsum7 = _ve_pvbrd_vs_i64(0UL) ;

	    __vr vry_r2 = _ve_vaddsl_vsv(h-2, vrh) ;
	    __vm256 vmy1_r2 = _ve_vfmkl_mcv(VECC_GE, vry_r2) ;
	    __vm256 vmy_r2 = vmy1_r2 ;

	    __vm256 vmall_r2s2 = _ve_andm_mmm(vmy_r2,vmx_s2) ;
	    __vm256 vmall_r2s0 = _ve_andm_mmm(vmy_r2,vmx_s0) ;

	    __vr vry_r1 = _ve_vaddsl_vsv(h-1, vrh) ;
	    __vm256 vmy1_r1 = _ve_vfmkl_mcv(VECC_GE, vry_r1) ;
	    __vm256 vmy2_r1 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(gOutHeight,vry_r1)) ;
	    __vm256 vmy_r1 = _ve_andm_mmm(vmy1_r1, vmy2_r1) ;

	    __vm256 vmall_r1s2 = _ve_andm_mmm(vmy_r1,vmx_s2) ;
	    __vm256 vmall_r1s0 = _ve_andm_mmm(vmy_r1,vmx_s0) ;

	    __vr vry_r0= _ve_vaddsl_vsv(h, vrh) ;
	    __vm256 vmy2_r0 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(gOutHeight,vry_r0)) ;
	    __vm256 vmy_r0 = vmy2_r0 ;

	    __vm256 vmall_r0s2 = _ve_andm_mmm(vmy_r0,vmx_s2) ;
	    __vm256 vmall_r0s0 = _ve_andm_mmm(vmy_r0,vmx_s0) ;

	    for (int64_t c=0; c<gOutChannelGroup; c++) {
	      int64_t gOutIndex    = gOutGroupOffset + ((n * gOutChannel + c) * gOutHeight) * gOutWidth ;

	      const float *pKerValue = pKernel + kernGroupOffset + ((c * gInChannelGroup + k) * kernHeight + 2) * kernWidth + 2;

	      _ve_lvl(vl0) ;
	      __vr vrgout_r2 = _ve_vld2d_vss((gOutWidthHalf<<(3+16))+8, pGOut+gOutIndex+(h-2)*gOutWidth-2) ;
	      __vr vrgout_r1 = _ve_vld2d_vss((gOutWidthHalf<<(3+16))+8, pGOut+gOutIndex+(h-1)*gOutWidth-2) ;
	      __vr vrgout_r0 = _ve_vld2d_vss((gOutWidthHalf<<(3+16))+8, pGOut+gOutIndex+(h-0)*gOutWidth-2) ;

	      __vr vrgout_r2s2 = _ve_vcp_vvmv(vrgout_r2, vm_s2, _ve_vbrd_vs_i64(0UL)) ;
	      __vr vrgout_r2s0 = _ve_vcp_vvmv(vrgout_r2, vm_s0, _ve_vbrd_vs_i64(0UL)) ;

	      __vr vrgout_r1s2 = _ve_vcp_vvmv(vrgout_r1, vm_s2, _ve_vbrd_vs_i64(0UL)) ;
	      __vr vrgout_r1s0 = _ve_vcp_vvmv(vrgout_r1, vm_s0, _ve_vbrd_vs_i64(0UL)) ;

	      __vr vrgout_r0s2 = _ve_vcp_vvmv(vrgout_r0, vm_s2, _ve_vbrd_vs_i64(0UL)) ;
	      __vr vrgout_r0s0 = _ve_vcp_vvmv(vrgout_r0, vm_s0, _ve_vbrd_vs_i64(0UL)) ;

	      _ve_lvl(vl1) ;
#define VFADD8(VRGOUT)											\
{													\
  vrsum0 = _ve_pvfmad_vvsv(vrsum0, _ve_pack_f32a(pKerValue),                               VRGOUT) ;	\
  vrsum1 = _ve_pvfmad_vvsv(vrsum1, _ve_pack_f32a(pKerValue +     kernHeight * kernWidth ), VRGOUT) ;	\
  vrsum2 = _ve_pvfmad_vvsv(vrsum2, _ve_pack_f32a(pKerValue + 2 * kernHeight * kernWidth ), VRGOUT) ;	\
  vrsum3 = _ve_pvfmad_vvsv(vrsum3, _ve_pack_f32a(pKerValue + 3 * kernHeight * kernWidth ), VRGOUT) ;	\
  vrsum4 = _ve_pvfmad_vvsv(vrsum4, _ve_pack_f32a(pKerValue + 4 * kernHeight * kernWidth ), VRGOUT) ;	\
  vrsum5 = _ve_pvfmad_vvsv(vrsum5, _ve_pack_f32a(pKerValue + 5 * kernHeight * kernWidth ), VRGOUT) ;	\
  vrsum6 = _ve_pvfmad_vvsv(vrsum6, _ve_pack_f32a(pKerValue + 6 * kernHeight * kernWidth ), VRGOUT) ;	\
  vrsum7 = _ve_pvfmad_vvsv(vrsum7, _ve_pack_f32a(pKerValue + 7 * kernHeight * kernWidth ), VRGOUT) ;	\
}

	      vrgout_r2s2 = _ve_vmrg_vvvm(_ve_vbrd_vs_i64(0UL), vrgout_r2s2, vmall_r2s2) ;
	      vrgout_r2s0 = _ve_vmrg_vvvm(_ve_vbrd_vs_i64(0UL), vrgout_r2s0, vmall_r2s0) ;
	      __vr vrgout_r2s1 = _ve_vshf_vvvs(vrgout_r2s2, vrgout_r2s0, VE_VSHUFFLE_ZLYU) ;
	      VFADD8(vrgout_r2s2) ; pKerValue-- ;
	      VFADD8(vrgout_r2s1) ; pKerValue-- ;
	      VFADD8(vrgout_r2s0) ; pKerValue-- ;

	      vrgout_r1s2 = _ve_vmrg_vvvm(_ve_vbrd_vs_i64(0UL), vrgout_r1s2, vmall_r1s2) ;
	      vrgout_r1s0 = _ve_vmrg_vvvm(_ve_vbrd_vs_i64(0UL), vrgout_r1s0, vmall_r1s0) ;
	      __vr vrgout_r1s1 = _ve_vshf_vvvs(vrgout_r1s2, vrgout_r1s0, VE_VSHUFFLE_ZLYU) ;
	      VFADD8(vrgout_r1s2) ; pKerValue-- ;
	      VFADD8(vrgout_r1s1) ; pKerValue-- ;
	      VFADD8(vrgout_r1s0) ; pKerValue-- ;

	      vrgout_r0s2 = _ve_vmrg_vvvm(_ve_vbrd_vs_i64(0UL), vrgout_r0s2, vmall_r0s2) ;
	      vrgout_r0s0 = _ve_vmrg_vvvm(_ve_vbrd_vs_i64(0UL), vrgout_r0s0, vmall_r0s0) ;
	      __vr vrgout_r0s1 = _ve_vshf_vvvs(vrgout_r0s2, vrgout_r0s0, VE_VSHUFFLE_ZLYU) ;
	      VFADD8(vrgout_r0s2) ; pKerValue-- ;
	      VFADD8(vrgout_r0s1) ; pKerValue-- ;
	      VFADD8(vrgout_r0s0) ; pKerValue-- ;
#undef VFADD8
	    } // gOutChannel

	    _ve_vst_vss(vrsum0, 8, pGIn+gInIndex) ;
	    _ve_vst_vss(vrsum1, 8, pGIn+gInIndex+  gInPixels) ;
	    _ve_vst_vss(vrsum2, 8, pGIn+gInIndex+2*gInPixels) ;
	    _ve_vst_vss(vrsum3, 8, pGIn+gInIndex+3*gInPixels) ;
	    _ve_vst_vss(vrsum4, 8, pGIn+gInIndex+4*gInPixels) ;
	    _ve_vst_vss(vrsum5, 8, pGIn+gInIndex+5*gInPixels) ;
	    _ve_vst_vss(vrsum6, 8, pGIn+gInIndex+6*gInPixels) ;
	    _ve_vst_vss(vrsum7, 8, pGIn+gInIndex+7*gInPixels) ;

	  } // gOutPixels

	  k+=8 ;
	}
	for (; k<gInChannelGroup; k+=16) {
	  for (int64_t h=0; h<gInHeight; h+=16) {
	    const int64_t vl0 = 16 * (gInHeight - h < 16 ? gInHeight - h : 16) ;
	    const int64_t vl1 = gInWidthHalf * (gInHeight - h < 16 ? gInHeight - h : 16) ;

	    const int64_t gip = h * gInWidth ;

	    const int64_t gInIndex = gInGroupOffset + ((n * gInChannel + k) * gInHeight ) * gInWidth + gip ;

	    _ve_lvl(vl1) ;
	    __vr vrsum0 = _ve_pvbrd_vs_i64(0UL) ;
	    __vr vrsum1 = _ve_pvbrd_vs_i64(0UL) ;
	    __vr vrsum2 = _ve_pvbrd_vs_i64(0UL) ;
	    __vr vrsum3 = _ve_pvbrd_vs_i64(0UL) ;
	    __vr vrsum4 = _ve_pvbrd_vs_i64(0UL) ;
	    __vr vrsum5 = _ve_pvbrd_vs_i64(0UL) ;
	    __vr vrsum6 = _ve_pvbrd_vs_i64(0UL) ;
	    __vr vrsum7 = _ve_pvbrd_vs_i64(0UL) ;
	    __vr vrsum8 = _ve_pvbrd_vs_i64(0UL) ;
	    __vr vrsum9 = _ve_pvbrd_vs_i64(0UL) ;
	    __vr vrsumA = _ve_pvbrd_vs_i64(0UL) ;
	    __vr vrsumB = _ve_pvbrd_vs_i64(0UL) ;
	    __vr vrsumC = _ve_pvbrd_vs_i64(0UL) ;
	    __vr vrsumD = _ve_pvbrd_vs_i64(0UL) ;
	    __vr vrsumE = _ve_pvbrd_vs_i64(0UL) ;
	    __vr vrsumF = _ve_pvbrd_vs_i64(0UL) ;

	    __vr vry_r2 = _ve_vaddsl_vsv(h-2, vrh) ;
	    __vm256 vmy1_r2 = _ve_vfmkl_mcv(VECC_GE, vry_r2) ;
	    __vm256 vmy_r2 = vmy1_r2 ;

	    __vm256 vmall_r2s2 = _ve_andm_mmm(vmy_r2,vmx_s2) ;
	    __vm256 vmall_r2s0 = _ve_andm_mmm(vmy_r2,vmx_s0) ;

	    __vr vry_r1 = _ve_vaddsl_vsv(h-1, vrh) ;
	    __vm256 vmy1_r1 = _ve_vfmkl_mcv(VECC_GE, vry_r1) ;
	    __vm256 vmy2_r1 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(gOutHeight,vry_r1)) ;
	    __vm256 vmy_r1 = _ve_andm_mmm(vmy1_r1, vmy2_r1) ;

	    __vm256 vmall_r1s2 = _ve_andm_mmm(vmy_r1,vmx_s2) ;
	    __vm256 vmall_r1s0 = _ve_andm_mmm(vmy_r1,vmx_s0) ;

	    __vr vry_r0= _ve_vaddsl_vsv(h, vrh) ;
	    __vm256 vmy2_r0 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(gOutHeight,vry_r0)) ;
	    __vm256 vmy_r0 = vmy2_r0 ;

	    __vm256 vmall_r0s2 = _ve_andm_mmm(vmy_r0,vmx_s2) ;
	    __vm256 vmall_r0s0 = _ve_andm_mmm(vmy_r0,vmx_s0) ;

	    for (int64_t c=0; c<gOutChannelGroup; c++) {
	      int64_t gOutIndex    = gOutGroupOffset + ((n * gOutChannel + c) * gOutHeight) * gOutWidth ;

	      const float *pKerValue = pKernel + kernGroupOffset + ((c * gInChannelGroup + k) * kernHeight + 2) * kernWidth + 2;

	      _ve_lvl(vl0) ;
	      __vr vrgout_r2 = _ve_vld2d_vss((gOutWidthHalf<<(3+16))+8, pGOut+gOutIndex+(h-2)*gOutWidth-2) ;
	      __vr vrgout_r1 = _ve_vld2d_vss((gOutWidthHalf<<(3+16))+8, pGOut+gOutIndex+(h-1)*gOutWidth-2) ;
	      __vr vrgout_r0 = _ve_vld2d_vss((gOutWidthHalf<<(3+16))+8, pGOut+gOutIndex+(h-0)*gOutWidth-2) ;

	      __vr vrgout_r2s2 = _ve_vcp_vvmv(vrgout_r2, vm_s2, _ve_vbrd_vs_i64(0UL)) ;
	      __vr vrgout_r2s0 = _ve_vcp_vvmv(vrgout_r2, vm_s0, _ve_vbrd_vs_i64(0UL)) ;

	      __vr vrgout_r1s2 = _ve_vcp_vvmv(vrgout_r1, vm_s2, _ve_vbrd_vs_i64(0UL)) ;
	      __vr vrgout_r1s0 = _ve_vcp_vvmv(vrgout_r1, vm_s0, _ve_vbrd_vs_i64(0UL)) ;

	      __vr vrgout_r0s2 = _ve_vcp_vvmv(vrgout_r0, vm_s2, _ve_vbrd_vs_i64(0UL)) ;
	      __vr vrgout_r0s0 = _ve_vcp_vvmv(vrgout_r0, vm_s0, _ve_vbrd_vs_i64(0UL)) ;

	      _ve_lvl(vl1) ;
#define VFADD16(VRGOUT)											\
{													\
  vrsum0 = _ve_pvfmad_vvsv(vrsum0, _ve_pack_f32a(pKerValue),                               VRGOUT) ;	\
  vrsum1 = _ve_pvfmad_vvsv(vrsum1, _ve_pack_f32a(pKerValue +     kernHeight * kernWidth ), VRGOUT) ;	\
  vrsum2 = _ve_pvfmad_vvsv(vrsum2, _ve_pack_f32a(pKerValue + 2 * kernHeight * kernWidth ), VRGOUT) ;	\
  vrsum3 = _ve_pvfmad_vvsv(vrsum3, _ve_pack_f32a(pKerValue + 3 * kernHeight * kernWidth ), VRGOUT) ;	\
  vrsum4 = _ve_pvfmad_vvsv(vrsum4, _ve_pack_f32a(pKerValue + 4 * kernHeight * kernWidth ), VRGOUT) ;	\
  vrsum5 = _ve_pvfmad_vvsv(vrsum5, _ve_pack_f32a(pKerValue + 5 * kernHeight * kernWidth ), VRGOUT) ;	\
  vrsum6 = _ve_pvfmad_vvsv(vrsum6, _ve_pack_f32a(pKerValue + 6 * kernHeight * kernWidth ), VRGOUT) ;	\
  vrsum7 = _ve_pvfmad_vvsv(vrsum7, _ve_pack_f32a(pKerValue + 7 * kernHeight * kernWidth ), VRGOUT) ;	\
  vrsum8 = _ve_pvfmad_vvsv(vrsum8, _ve_pack_f32a(pKerValue + 8 * kernHeight * kernWidth ), VRGOUT) ;	\
  vrsum9 = _ve_pvfmad_vvsv(vrsum9, _ve_pack_f32a(pKerValue + 9 * kernHeight * kernWidth ), VRGOUT) ;	\
  vrsumA = _ve_pvfmad_vvsv(vrsumA, _ve_pack_f32a(pKerValue +10 * kernHeight * kernWidth ), VRGOUT) ;	\
  vrsumB = _ve_pvfmad_vvsv(vrsumB, _ve_pack_f32a(pKerValue +11 * kernHeight * kernWidth ), VRGOUT) ;	\
  vrsumC = _ve_pvfmad_vvsv(vrsumC, _ve_pack_f32a(pKerValue +12 * kernHeight * kernWidth ), VRGOUT) ;	\
  vrsumD = _ve_pvfmad_vvsv(vrsumD, _ve_pack_f32a(pKerValue +13 * kernHeight * kernWidth ), VRGOUT) ;	\
  vrsumE = _ve_pvfmad_vvsv(vrsumE, _ve_pack_f32a(pKerValue +14 * kernHeight * kernWidth ), VRGOUT) ;	\
  vrsumF = _ve_pvfmad_vvsv(vrsumF, _ve_pack_f32a(pKerValue +15 * kernHeight * kernWidth ), VRGOUT) ;	\
}

	      vrgout_r2s2 = _ve_vmrg_vvvm(_ve_vbrd_vs_i64(0UL), vrgout_r2s2, vmall_r2s2) ;
	      vrgout_r2s0 = _ve_vmrg_vvvm(_ve_vbrd_vs_i64(0UL), vrgout_r2s0, vmall_r2s0) ;
	      __vr vrgout_r2s1 = _ve_vshf_vvvs(vrgout_r2s2, vrgout_r2s0, VE_VSHUFFLE_ZLYU) ;
	      VFADD16(vrgout_r2s2) ; pKerValue-- ;
	      VFADD16(vrgout_r2s1) ; pKerValue-- ;
	      VFADD16(vrgout_r2s0) ; pKerValue-- ;

	      vrgout_r1s2 = _ve_vmrg_vvvm(_ve_vbrd_vs_i64(0UL), vrgout_r1s2, vmall_r1s2) ;
	      vrgout_r1s0 = _ve_vmrg_vvvm(_ve_vbrd_vs_i64(0UL), vrgout_r1s0, vmall_r1s0) ;
	      __vr vrgout_r1s1 = _ve_vshf_vvvs(vrgout_r1s2, vrgout_r1s0, VE_VSHUFFLE_ZLYU) ;
	      VFADD16(vrgout_r1s2) ; pKerValue-- ;
	      VFADD16(vrgout_r1s1) ; pKerValue-- ;
	      VFADD16(vrgout_r1s0) ; pKerValue-- ;

	      vrgout_r0s2 = _ve_vmrg_vvvm(_ve_vbrd_vs_i64(0UL), vrgout_r0s2, vmall_r0s2) ;
	      vrgout_r0s0 = _ve_vmrg_vvvm(_ve_vbrd_vs_i64(0UL), vrgout_r0s0, vmall_r0s0) ;
	      __vr vrgout_r0s1 = _ve_vshf_vvvs(vrgout_r0s2, vrgout_r0s0, VE_VSHUFFLE_ZLYU) ;
	      VFADD16(vrgout_r0s2) ; pKerValue-- ;
	      VFADD16(vrgout_r0s1) ; pKerValue-- ;
	      VFADD16(vrgout_r0s0) ; pKerValue-- ;
#undef VFADD16
	    } // gOutChannel

	    _ve_vst_vss(vrsum0, 8, pGIn+gInIndex) ;
	    _ve_vst_vss(vrsum1, 8, pGIn+gInIndex+  gInPixels) ;
	    _ve_vst_vss(vrsum2, 8, pGIn+gInIndex+2*gInPixels) ;
	    _ve_vst_vss(vrsum3, 8, pGIn+gInIndex+3*gInPixels) ;
	    _ve_vst_vss(vrsum4, 8, pGIn+gInIndex+4*gInPixels) ;
	    _ve_vst_vss(vrsum5, 8, pGIn+gInIndex+5*gInPixels) ;
	    _ve_vst_vss(vrsum6, 8, pGIn+gInIndex+6*gInPixels) ;
	    _ve_vst_vss(vrsum7, 8, pGIn+gInIndex+7*gInPixels) ;
	    _ve_vst_vss(vrsum8, 8, pGIn+gInIndex+8*gInPixels) ;
	    _ve_vst_vss(vrsum9, 8, pGIn+gInIndex+9*gInPixels) ;
	    _ve_vst_vss(vrsumA, 8, pGIn+gInIndex+10*gInPixels) ;
	    _ve_vst_vss(vrsumB, 8, pGIn+gInIndex+11*gInPixels) ;
	    _ve_vst_vss(vrsumC, 8, pGIn+gInIndex+12*gInPixels) ;
	    _ve_vst_vss(vrsumD, 8, pGIn+gInIndex+13*gInPixels) ;
	    _ve_vst_vss(vrsumE, 8, pGIn+gInIndex+14*gInPixels) ;
	    _ve_vst_vss(vrsumF, 8, pGIn+gInIndex+15*gInPixels) ;

	  } // gOutPixels
	} // gInChannel
      } // group
    } // batch
  }

  return VEDNN_SUCCESS;
}
