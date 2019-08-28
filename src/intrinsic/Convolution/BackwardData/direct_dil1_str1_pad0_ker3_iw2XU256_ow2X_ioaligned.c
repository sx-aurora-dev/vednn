#include <stdint.h>
#include <stdlib.h>

#include "vednn.h"

#include "velintrin.h"
#define VLEN	(256)

vednnError_t
vednnConvolutionBackwardData_direct_dil1_str1_pad0_ker3_iw2XU256_ow2X_ioaligned(
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
    const int64_t gInWidthHalf = gInWidth >> 1 ;
    const int64_t nH = VLEN / gInWidthHalf ;

    __vr vrseq = _vel_vseq_vl(nH*gInWidthHalf) ;
    __vr vrh  = _vel_vdivsl_vvsl(vrseq, gInWidthHalf, nH*gInWidthHalf) ;
    __vr vrw  = _vel_vmulsl_vsvl(2, _vel_vsubsl_vvvl(vrseq, _vel_vmulul_vsvl(gInWidthHalf,vrh, nH*gInWidthHalf), nH*gInWidthHalf), nH*gInWidthHalf) ;
    __vr vrhw = _vel_vaddsl_vvvl(_vel_vmulsl_vsvl(gOutWidth, vrh, nH*gInWidthHalf), vrw, nH*gInWidthHalf) ;

    __vr vrx_s2 = _vel_vaddsl_vsvl(-2, vrw, nH*gInWidthHalf) ;
    __vm256 vmx1_s2 =  _vel_vfmklge_mvl(vrx_s2, nH*gInWidthHalf) ;
    __vm256 vmx_s2 = vmx1_s2 ;

    __vr vrx_s0 = vrw ;
    __vm256 vmx2_s0 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutWidth,vrx_s0, nH*gInWidthHalf), nH*gInWidthHalf) ;
    __vm256 vmx_s0 = vmx2_s0 ;

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

	    __vr vrsum = _vel_vbrdl_vsl(0UL, vl) ;

	    __vr vry_r2 = _vel_vaddsl_vsvl(h-2, vrh, vl) ;
	    __vm256 vmy1_r2 =  _vel_vfmklge_mvl(vry_r2, vl) ;
	    __vm256 vmy_r2 = vmy1_r2 ;

	    __vm256 vmall_r2s2 = _vel_andm_mmm(vmy_r2,vmx_s2) ;
	    __vm256 vmall_r2s0 = _vel_andm_mmm(vmy_r2,vmx_s0) ;

	    __vr vry_r1 = _vel_vaddsl_vsvl(h-1, vrh, vl) ;
	    __vm256 vmy1_r1 =  _vel_vfmklge_mvl(vry_r1, vl) ;
	    __vm256 vmy2_r1 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutHeight,vry_r1, vl), vl) ;
	    __vm256 vmy_r1 = _vel_andm_mmm(vmy1_r1, vmy2_r1) ;

	    __vm256 vmall_r1s2 = _vel_andm_mmm(vmy_r1,vmx_s2) ;
	    __vm256 vmall_r1s0 = _vel_andm_mmm(vmy_r1,vmx_s0) ;

	    __vr vry_r0= _vel_vaddsl_vsvl(h, vrh, vl) ;
	    __vm256 vmy2_r0 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutHeight,vry_r0, vl), vl) ;
	    __vm256 vmy_r0 = vmy2_r0 ;

	    __vm256 vmall_r0s2 = _vel_andm_mmm(vmy_r0,vmx_s2) ;
	    __vm256 vmall_r0s0 = _vel_andm_mmm(vmy_r0,vmx_s0) ;

	    for (int64_t c=0; c<gOutChannelGroup; c++) {
	      int64_t gOutIndex    = gOutGroupOffset + ((n * gOutChannel + c) * gOutHeight) * gOutWidth ;

	      const float *pKerValue = pKernel + kernGroupOffset + ((c * gInChannelGroup + k) * kernHeight + 2) * kernWidth + 2;

	      __vr vrgout_ptr_r2s2 = _vel_vsfa_vvssl(vrhw, 2, (unsigned long)(pGOut+gOutIndex+(h-2)*gOutHeight-2), vl) ;
	      __vr vrgout_ptr_r1s2 = _vel_vsfa_vvssl(vrhw, 2, (unsigned long)(pGOut+gOutIndex+(h-1)*gOutHeight-2), vl) ;
	      __vr vrgout_ptr_r0s2 = _vel_vsfa_vvssl(vrhw, 2, (unsigned long)(pGOut+gOutIndex+(h-0)*gOutHeight-2), vl) ;

	      __vr vrgout_r2s2 = _vel_vgt_vvssml(vrgout_ptr_r2s2, 0, 0, vmall_r2s2, vl) ;
	      __vr vrgout_r1s2 = _vel_vgt_vvssml(vrgout_ptr_r1s2, 0, 0, vmall_r1s2, vl) ;
	      __vr vrgout_r0s2 = _vel_vgt_vvssml(vrgout_ptr_r0s2, 0, 0, vmall_r0s2, vl) ;

	      __vr vrgout_r2s0 = _vel_vmv_vsvl(1, vrgout_r2s2, vl) ;
	      __vr vrgout_r1s0 = _vel_vmv_vsvl(1, vrgout_r1s2, vl) ;
	      __vr vrgout_r0s0 = _vel_vmv_vsvl(1, vrgout_r0s2, vl) ;

	      vrgout_r2s2 = _vel_vmrg_vvvml(_vel_vbrdl_vsl(0UL, vl), vrgout_r2s2, vmall_r2s2, vl) ;
	      vrgout_r2s0 = _vel_vmrg_vvvml(_vel_vbrdl_vsl(0UL, vl), vrgout_r2s0, vmall_r2s0, vl) ;
	      __vr vrgout_r2s1 = _vel_vshf_vvvsl(vrgout_r2s2, vrgout_r2s0, VE_VSHUFFLE_ZLYU, vl) ;
	      vrsum = _vel_pvfmad_vvsvl(vrsum, _vel_pack_f32a(pKerValue), vrgout_r2s2, vl) ; pKerValue-- ;
	      vrsum = _vel_pvfmad_vvsvl(vrsum, _vel_pack_f32a(pKerValue), vrgout_r2s1, vl) ; pKerValue-- ;
	      vrsum = _vel_pvfmad_vvsvl(vrsum, _vel_pack_f32a(pKerValue), vrgout_r2s0, vl) ; pKerValue-- ;

	      vrgout_r1s2 = _vel_vmrg_vvvml(_vel_vbrdl_vsl(0UL, vl), vrgout_r1s2, vmall_r1s2, vl) ;
	      vrgout_r1s0 = _vel_vmrg_vvvml(_vel_vbrdl_vsl(0UL, vl), vrgout_r1s0, vmall_r1s0, vl) ;
	      __vr vrgout_r1s1 = _vel_vshf_vvvsl(vrgout_r1s2, vrgout_r1s0, VE_VSHUFFLE_ZLYU, vl) ;

	      vrsum = _vel_pvfmad_vvsvl(vrsum, _vel_pack_f32a(pKerValue), vrgout_r1s2, vl) ; pKerValue-- ;
	      vrsum = _vel_pvfmad_vvsvl(vrsum, _vel_pack_f32a(pKerValue), vrgout_r1s1, vl) ; pKerValue-- ;
	      vrsum = _vel_pvfmad_vvsvl(vrsum, _vel_pack_f32a(pKerValue), vrgout_r1s0, vl) ; pKerValue-- ;

	      vrgout_r0s2 = _vel_vmrg_vvvml(_vel_vbrdl_vsl(0UL, vl), vrgout_r0s2, vmall_r0s2, vl) ;
	      vrgout_r0s0 = _vel_vmrg_vvvml(_vel_vbrdl_vsl(0UL, vl), vrgout_r0s0, vmall_r0s0, vl) ;
	      __vr vrgout_r0s1 = _vel_vshf_vvvsl(vrgout_r0s2, vrgout_r0s0, VE_VSHUFFLE_ZLYU, vl) ;

	      vrsum = _vel_pvfmad_vvsvl(vrsum, _vel_pack_f32a(pKerValue), vrgout_r0s2, vl) ; pKerValue-- ;
	      vrsum = _vel_pvfmad_vvsvl(vrsum, _vel_pack_f32a(pKerValue), vrgout_r0s1, vl) ; pKerValue-- ;
	      vrsum = _vel_pvfmad_vvsvl(vrsum, _vel_pack_f32a(pKerValue), vrgout_r0s0, vl) ; pKerValue-- ;

	    } // gOutChannel

	    _vel_vst_vssl(vrsum, 8, pGIn+gInIndex, vl) ;

	  } // gOutPixels


	  k++ ;
	}
	if( ((gInChannelGroup>>1) & 0x01 ) == 1 ) {

	  for (int64_t h=0; h<gInHeight; h+=nH) {
	    const int64_t vl = gInWidthHalf * (gInHeight - h < nH ? gInHeight - h : nH) ;
	    const int64_t gip = h * gInWidth ;

	    const int64_t gInIndex = gInGroupOffset + ((n * gInChannel + k) * gInHeight ) * gInWidth + gip ;

	    __vr vrsum0 = _vel_pvbrd_vsl(0UL, vl) ;
	    __vr vrsum1 = _vel_pvbrd_vsl(0UL, vl) ;

	    __vr vry_r2 = _vel_vaddsl_vsvl(h-2, vrh, vl) ;
	    __vm256 vmy1_r2 =  _vel_vfmklge_mvl(vry_r2, vl) ;
	    __vm256 vmy_r2 = vmy1_r2 ;

	    __vm256 vmall_r2s2 = _vel_andm_mmm(vmy_r2,vmx_s2) ;
	    __vm256 vmall_r2s0 = _vel_andm_mmm(vmy_r2,vmx_s0) ;

	    __vr vry_r1 = _vel_vaddsl_vsvl(h-1, vrh, vl) ;
	    __vm256 vmy1_r1 =  _vel_vfmklge_mvl(vry_r1, vl) ;
	    __vm256 vmy2_r1 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutHeight,vry_r1, vl), vl) ;
	    __vm256 vmy_r1 = _vel_andm_mmm(vmy1_r1, vmy2_r1) ;

	    __vm256 vmall_r1s2 = _vel_andm_mmm(vmy_r1,vmx_s2) ;
	    __vm256 vmall_r1s0 = _vel_andm_mmm(vmy_r1,vmx_s0) ;

	    __vr vry_r0= _vel_vaddsl_vsvl(h, vrh, vl) ;
	    __vm256 vmy2_r0 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutHeight,vry_r0, vl), vl) ;
	    __vm256 vmy_r0 = vmy2_r0 ;

	    __vm256 vmall_r0s2 = _vel_andm_mmm(vmy_r0,vmx_s2) ;
	    __vm256 vmall_r0s0 = _vel_andm_mmm(vmy_r0,vmx_s0) ;

	    for (int64_t c=0; c<gOutChannelGroup; c++) {
	      int64_t gOutIndex    = gOutGroupOffset + ((n * gOutChannel + c) * gOutHeight) * gOutWidth ;

	      const float *pKerValue = pKernel + kernGroupOffset + ((c * gInChannelGroup + k) * kernHeight + 2) * kernWidth + 2;

	      __vr vrgout_ptr_r2s2 = _vel_vsfa_vvssl(vrhw, 2, (unsigned long)(pGOut+gOutIndex+(h-2)*gOutHeight-2), vl) ;
	      __vr vrgout_ptr_r1s2 = _vel_vsfa_vvssl(vrhw, 2, (unsigned long)(pGOut+gOutIndex+(h-1)*gOutHeight-2), vl) ;
	      __vr vrgout_ptr_r0s2 = _vel_vsfa_vvssl(vrhw, 2, (unsigned long)(pGOut+gOutIndex+(h-0)*gOutHeight-2), vl) ;

	      __vr vrgout_r2s2 = _vel_vgt_vvssml(vrgout_ptr_r2s2, 0, 0, vmall_r2s2, vl) ;
	      __vr vrgout_r1s2 = _vel_vgt_vvssml(vrgout_ptr_r1s2, 0, 0, vmall_r1s2, vl) ;
	      __vr vrgout_r0s2 = _vel_vgt_vvssml(vrgout_ptr_r0s2, 0, 0, vmall_r0s2, vl) ;

	      __vr vrgout_r2s0 = _vel_vmv_vsvl(1, vrgout_r2s2, vl) ;
	      __vr vrgout_r1s0 = _vel_vmv_vsvl(1, vrgout_r1s2, vl) ;
	      __vr vrgout_r0s0 = _vel_vmv_vsvl(1, vrgout_r0s2, vl) ;

#define VFADD2(VRGOUT)											\
{													\
  vrsum0 = _vel_pvfmad_vvsvl(vrsum0, _vel_pack_f32a(pKerValue), VRGOUT, vl) ;				\
  vrsum1 = _vel_pvfmad_vvsvl(vrsum1, _vel_pack_f32a(pKerValue + kernHeight * kernWidth ), VRGOUT, vl) ;	\
}

	      vrgout_r2s2 = _vel_vmrg_vvvml(_vel_vbrdl_vsl(0UL, vl), vrgout_r2s2, vmall_r2s2, vl) ;
	      vrgout_r2s0 = _vel_vmrg_vvvml(_vel_vbrdl_vsl(0UL, vl), vrgout_r2s0, vmall_r2s0, vl) ;
	      __vr vrgout_r2s1 = _vel_vshf_vvvsl(vrgout_r2s2, vrgout_r2s0, VE_VSHUFFLE_ZLYU, vl) ;
	      VFADD2(vrgout_r2s2) ; pKerValue-- ;
	      VFADD2(vrgout_r2s1) ; pKerValue-- ;
	      VFADD2(vrgout_r2s0) ; pKerValue-- ;

	      vrgout_r1s2 = _vel_vmrg_vvvml(_vel_vbrdl_vsl(0UL, vl), vrgout_r1s2, vmall_r1s2, vl) ;
	      vrgout_r1s0 = _vel_vmrg_vvvml(_vel_vbrdl_vsl(0UL, vl), vrgout_r1s0, vmall_r1s0, vl) ;
	      __vr vrgout_r1s1 = _vel_vshf_vvvsl(vrgout_r1s2, vrgout_r1s0, VE_VSHUFFLE_ZLYU, vl) ;
	      VFADD2(vrgout_r1s2) ; pKerValue-- ;
	      VFADD2(vrgout_r1s1) ; pKerValue-- ;
	      VFADD2(vrgout_r1s0) ; pKerValue-- ;

	      vrgout_r0s2 = _vel_vmrg_vvvml(_vel_vbrdl_vsl(0UL, vl), vrgout_r0s2, vmall_r0s2, vl) ;
	      vrgout_r0s0 = _vel_vmrg_vvvml(_vel_vbrdl_vsl(0UL, vl), vrgout_r0s0, vmall_r0s0, vl) ;
	      __vr vrgout_r0s1 = _vel_vshf_vvvsl(vrgout_r0s2, vrgout_r0s0, VE_VSHUFFLE_ZLYU, vl) ;
	      VFADD2(vrgout_r0s2) ; pKerValue-- ;
	      VFADD2(vrgout_r0s1) ; pKerValue-- ;
	      VFADD2(vrgout_r0s0) ; pKerValue-- ;
#undef VFADD2
	    } // gOutChannel

	    _vel_vst_vssl(vrsum0, 8, pGIn+gInIndex, vl) ;
	    _vel_vst_vssl(vrsum1, 8, pGIn+gInIndex+  gInPixels, vl) ;

	  } // gOutPixels

	  k+=2 ;
	}
	if( ((gInChannelGroup>>2) & 0x01 ) == 1 ) {

	  for (int64_t h=0; h<gInHeight; h+=nH) {
	    const int64_t vl = gInWidthHalf * (gInHeight - h < nH ? gInHeight - h : nH) ;
	    const int64_t gip = h * gInWidth ;

	    const int64_t gInIndex = gInGroupOffset + ((n * gInChannel + k) * gInHeight ) * gInWidth + gip ;

	    __vr vrsum0 = _vel_pvbrd_vsl(0UL, vl) ;
	    __vr vrsum1 = _vel_pvbrd_vsl(0UL, vl) ;
	    __vr vrsum2 = _vel_pvbrd_vsl(0UL, vl) ;
	    __vr vrsum3 = _vel_pvbrd_vsl(0UL, vl) ;

	    __vr vry_r2 = _vel_vaddsl_vsvl(h-2, vrh, vl) ;
	    __vm256 vmy1_r2 =  _vel_vfmklge_mvl(vry_r2, vl) ;
	    __vm256 vmy_r2 = vmy1_r2 ;

	    __vm256 vmall_r2s2 = _vel_andm_mmm(vmy_r2,vmx_s2) ;
	    __vm256 vmall_r2s0 = _vel_andm_mmm(vmy_r2,vmx_s0) ;

	    __vr vry_r1 = _vel_vaddsl_vsvl(h-1, vrh, vl) ;
	    __vm256 vmy1_r1 =  _vel_vfmklge_mvl(vry_r1, vl) ;
	    __vm256 vmy2_r1 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutHeight,vry_r1, vl), vl) ;
	    __vm256 vmy_r1 = _vel_andm_mmm(vmy1_r1, vmy2_r1) ;

	    __vm256 vmall_r1s2 = _vel_andm_mmm(vmy_r1,vmx_s2) ;
	    __vm256 vmall_r1s0 = _vel_andm_mmm(vmy_r1,vmx_s0) ;

	    __vr vry_r0= _vel_vaddsl_vsvl(h, vrh, vl) ;
	    __vm256 vmy2_r0 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutHeight,vry_r0, vl), vl) ;
	    __vm256 vmy_r0 = vmy2_r0 ;

	    __vm256 vmall_r0s2 = _vel_andm_mmm(vmy_r0,vmx_s2) ;
	    __vm256 vmall_r0s0 = _vel_andm_mmm(vmy_r0,vmx_s0) ;

	    for (int64_t c=0; c<gOutChannelGroup; c++) {
	      int64_t gOutIndex    = gOutGroupOffset + ((n * gOutChannel + c) * gOutHeight) * gOutWidth ;

	      const float *pKerValue = pKernel + kernGroupOffset + ((c * gInChannelGroup + k) * kernHeight + 2) * kernWidth + 2;

	      __vr vrgout_ptr_r2s2 = _vel_vsfa_vvssl(vrhw, 2, (unsigned long)(pGOut+gOutIndex+(h-2)*gOutHeight-2), vl) ;
	      __vr vrgout_ptr_r1s2 = _vel_vsfa_vvssl(vrhw, 2, (unsigned long)(pGOut+gOutIndex+(h-1)*gOutHeight-2), vl) ;
	      __vr vrgout_ptr_r0s2 = _vel_vsfa_vvssl(vrhw, 2, (unsigned long)(pGOut+gOutIndex+(h-0)*gOutHeight-2), vl) ;

	      __vr vrgout_r2s2 = _vel_vgt_vvssml(vrgout_ptr_r2s2, 0, 0, vmall_r2s2, vl) ;
	      __vr vrgout_r1s2 = _vel_vgt_vvssml(vrgout_ptr_r1s2, 0, 0, vmall_r1s2, vl) ;
	      __vr vrgout_r0s2 = _vel_vgt_vvssml(vrgout_ptr_r0s2, 0, 0, vmall_r0s2, vl) ;

	      __vr vrgout_r2s0 = _vel_vmv_vsvl(1, vrgout_r2s2, vl) ;
	      __vr vrgout_r1s0 = _vel_vmv_vsvl(1, vrgout_r1s2, vl) ;
	      __vr vrgout_r0s0 = _vel_vmv_vsvl(1, vrgout_r0s2, vl) ;

#define VFADD4(VRGOUT)												\
{														\
  vrsum0 = _vel_pvfmad_vvsvl(vrsum0, _vel_pack_f32a(pKerValue),                               VRGOUT, vl) ;	\
  vrsum1 = _vel_pvfmad_vvsvl(vrsum1, _vel_pack_f32a(pKerValue +     kernHeight * kernWidth ), VRGOUT, vl) ;	\
  vrsum2 = _vel_pvfmad_vvsvl(vrsum2, _vel_pack_f32a(pKerValue + 2 * kernHeight * kernWidth ), VRGOUT, vl) ;	\
  vrsum3 = _vel_pvfmad_vvsvl(vrsum3, _vel_pack_f32a(pKerValue + 3 * kernHeight * kernWidth ), VRGOUT, vl) ;	\
}

	      vrgout_r2s2 = _vel_vmrg_vvvml(_vel_vbrdl_vsl(0UL, vl), vrgout_r2s2, vmall_r2s2, vl) ;
	      vrgout_r2s0 = _vel_vmrg_vvvml(_vel_vbrdl_vsl(0UL, vl), vrgout_r2s0, vmall_r2s0, vl) ;
	      __vr vrgout_r2s1 = _vel_vshf_vvvsl(vrgout_r2s2, vrgout_r2s0, VE_VSHUFFLE_ZLYU, vl) ;
	      VFADD4(vrgout_r2s2) ; pKerValue-- ;
	      VFADD4(vrgout_r2s1) ; pKerValue-- ;
	      VFADD4(vrgout_r2s0) ; pKerValue-- ;

	      vrgout_r1s2 = _vel_vmrg_vvvml(_vel_vbrdl_vsl(0UL, vl), vrgout_r1s2, vmall_r1s2, vl) ;
	      vrgout_r1s0 = _vel_vmrg_vvvml(_vel_vbrdl_vsl(0UL, vl), vrgout_r1s0, vmall_r1s0, vl) ;
	      __vr vrgout_r1s1 = _vel_vshf_vvvsl(vrgout_r1s2, vrgout_r1s0, VE_VSHUFFLE_ZLYU, vl) ;
	      VFADD4(vrgout_r1s2) ; pKerValue-- ;
	      VFADD4(vrgout_r1s1) ; pKerValue-- ;
	      VFADD4(vrgout_r1s0) ; pKerValue-- ;

	      vrgout_r0s2 = _vel_vmrg_vvvml(_vel_vbrdl_vsl(0UL, vl), vrgout_r0s2, vmall_r0s2, vl) ;
	      vrgout_r0s0 = _vel_vmrg_vvvml(_vel_vbrdl_vsl(0UL, vl), vrgout_r0s0, vmall_r0s0, vl) ;
	      __vr vrgout_r0s1 = _vel_vshf_vvvsl(vrgout_r0s2, vrgout_r0s0, VE_VSHUFFLE_ZLYU, vl) ;
	      VFADD4(vrgout_r0s2) ; pKerValue-- ;
	      VFADD4(vrgout_r0s1) ; pKerValue-- ;
	      VFADD4(vrgout_r0s0) ; pKerValue-- ;
#undef VFADD4
	    } // gOutChannel

	    _vel_vst_vssl(vrsum0, 8, pGIn+gInIndex, vl) ;
	    _vel_vst_vssl(vrsum1, 8, pGIn+gInIndex+  gInPixels, vl) ;
	    _vel_vst_vssl(vrsum2, 8, pGIn+gInIndex+2*gInPixels, vl) ;
	    _vel_vst_vssl(vrsum3, 8, pGIn+gInIndex+3*gInPixels, vl) ;

	  } // gOutPixels

	  k+=4 ;
	}
	if( ((gInChannelGroup>>3) & 0x01 ) == 1 ) {

	  for (int64_t h=0; h<gInHeight; h+=nH) {
	    const int64_t vl = gInWidthHalf * (gInHeight - h < nH ? gInHeight - h : nH) ;
	    const int64_t gip = h * gInWidth ;

	    const int64_t gInIndex = gInGroupOffset + ((n * gInChannel + k) * gInHeight ) * gInWidth + gip ;

	    __vr vrsum0 = _vel_pvbrd_vsl(0UL, vl) ;
	    __vr vrsum1 = _vel_pvbrd_vsl(0UL, vl) ;
	    __vr vrsum2 = _vel_pvbrd_vsl(0UL, vl) ;
	    __vr vrsum3 = _vel_pvbrd_vsl(0UL, vl) ;
	    __vr vrsum4 = _vel_pvbrd_vsl(0UL, vl) ;
	    __vr vrsum5 = _vel_pvbrd_vsl(0UL, vl) ;
	    __vr vrsum6 = _vel_pvbrd_vsl(0UL, vl) ;
	    __vr vrsum7 = _vel_pvbrd_vsl(0UL, vl) ;

	    __vr vry_r2 = _vel_vaddsl_vsvl(h-2, vrh, vl) ;
	    __vm256 vmy1_r2 =  _vel_vfmklge_mvl(vry_r2, vl) ;
	    __vm256 vmy_r2 = vmy1_r2 ;

	    __vm256 vmall_r2s2 = _vel_andm_mmm(vmy_r2,vmx_s2) ;
	    __vm256 vmall_r2s0 = _vel_andm_mmm(vmy_r2,vmx_s0) ;

	    __vr vry_r1 = _vel_vaddsl_vsvl(h-1, vrh, vl) ;
	    __vm256 vmy1_r1 =  _vel_vfmklge_mvl(vry_r1, vl) ;
	    __vm256 vmy2_r1 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutHeight,vry_r1, vl), vl) ;
	    __vm256 vmy_r1 = _vel_andm_mmm(vmy1_r1, vmy2_r1) ;

	    __vm256 vmall_r1s2 = _vel_andm_mmm(vmy_r1,vmx_s2) ;
	    __vm256 vmall_r1s0 = _vel_andm_mmm(vmy_r1,vmx_s0) ;

	    __vr vry_r0= _vel_vaddsl_vsvl(h, vrh, vl) ;
	    __vm256 vmy2_r0 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutHeight,vry_r0, vl), vl) ;
	    __vm256 vmy_r0 = vmy2_r0 ;

	    __vm256 vmall_r0s2 = _vel_andm_mmm(vmy_r0,vmx_s2) ;
	    __vm256 vmall_r0s0 = _vel_andm_mmm(vmy_r0,vmx_s0) ;

	    for (int64_t c=0; c<gOutChannelGroup; c++) {
	      int64_t gOutIndex    = gOutGroupOffset + ((n * gOutChannel + c) * gOutHeight) * gOutWidth ;

	      const float *pKerValue = pKernel + kernGroupOffset + ((c * gInChannelGroup + k) * kernHeight + 2) * kernWidth + 2;

	      __vr vrgout_ptr_r2s2 = _vel_vsfa_vvssl(vrhw, 2, (unsigned long)(pGOut+gOutIndex+(h-2)*gOutHeight-2), vl) ;
	      __vr vrgout_ptr_r1s2 = _vel_vsfa_vvssl(vrhw, 2, (unsigned long)(pGOut+gOutIndex+(h-1)*gOutHeight-2), vl) ;
	      __vr vrgout_ptr_r0s2 = _vel_vsfa_vvssl(vrhw, 2, (unsigned long)(pGOut+gOutIndex+(h-0)*gOutHeight-2), vl) ;

	      __vr vrgout_r2s2 = _vel_vgt_vvssml(vrgout_ptr_r2s2, 0, 0, vmall_r2s2, vl) ;
	      __vr vrgout_r1s2 = _vel_vgt_vvssml(vrgout_ptr_r1s2, 0, 0, vmall_r1s2, vl) ;
	      __vr vrgout_r0s2 = _vel_vgt_vvssml(vrgout_ptr_r0s2, 0, 0, vmall_r0s2, vl) ;

	      __vr vrgout_r2s0 = _vel_vmv_vsvl(1, vrgout_r2s2, vl) ;
	      __vr vrgout_r1s0 = _vel_vmv_vsvl(1, vrgout_r1s2, vl) ;
	      __vr vrgout_r0s0 = _vel_vmv_vsvl(1, vrgout_r0s2, vl) ;

#define VFADD8(VRGOUT)												\
{														\
  vrsum0 = _vel_pvfmad_vvsvl(vrsum0, _vel_pack_f32a(pKerValue),                               VRGOUT, vl) ;	\
  vrsum1 = _vel_pvfmad_vvsvl(vrsum1, _vel_pack_f32a(pKerValue +     kernHeight * kernWidth ), VRGOUT, vl) ;	\
  vrsum2 = _vel_pvfmad_vvsvl(vrsum2, _vel_pack_f32a(pKerValue + 2 * kernHeight * kernWidth ), VRGOUT, vl) ;	\
  vrsum3 = _vel_pvfmad_vvsvl(vrsum3, _vel_pack_f32a(pKerValue + 3 * kernHeight * kernWidth ), VRGOUT, vl) ;	\
  vrsum4 = _vel_pvfmad_vvsvl(vrsum4, _vel_pack_f32a(pKerValue + 4 * kernHeight * kernWidth ), VRGOUT, vl) ;	\
  vrsum5 = _vel_pvfmad_vvsvl(vrsum5, _vel_pack_f32a(pKerValue + 5 * kernHeight * kernWidth ), VRGOUT, vl) ;	\
  vrsum6 = _vel_pvfmad_vvsvl(vrsum6, _vel_pack_f32a(pKerValue + 6 * kernHeight * kernWidth ), VRGOUT, vl) ;	\
  vrsum7 = _vel_pvfmad_vvsvl(vrsum7, _vel_pack_f32a(pKerValue + 7 * kernHeight * kernWidth ), VRGOUT, vl) ;	\
}

	      vrgout_r2s2 = _vel_vmrg_vvvml(_vel_vbrdl_vsl(0UL, vl), vrgout_r2s2, vmall_r2s2, vl) ;
	      vrgout_r2s0 = _vel_vmrg_vvvml(_vel_vbrdl_vsl(0UL, vl), vrgout_r2s0, vmall_r2s0, vl) ;
	      __vr vrgout_r2s1 = _vel_vshf_vvvsl(vrgout_r2s2, vrgout_r2s0, VE_VSHUFFLE_ZLYU, vl) ;
	      VFADD8(vrgout_r2s2) ; pKerValue-- ;
	      VFADD8(vrgout_r2s1) ; pKerValue-- ;
	      VFADD8(vrgout_r2s0) ; pKerValue-- ;

	      vrgout_r1s2 = _vel_vmrg_vvvml(_vel_vbrdl_vsl(0UL, vl), vrgout_r1s2, vmall_r1s2, vl) ;
	      vrgout_r1s0 = _vel_vmrg_vvvml(_vel_vbrdl_vsl(0UL, vl), vrgout_r1s0, vmall_r1s0, vl) ;
	      __vr vrgout_r1s1 = _vel_vshf_vvvsl(vrgout_r1s2, vrgout_r1s0, VE_VSHUFFLE_ZLYU, vl) ;
	      VFADD8(vrgout_r1s2) ; pKerValue-- ;
	      VFADD8(vrgout_r1s1) ; pKerValue-- ;
	      VFADD8(vrgout_r1s0) ; pKerValue-- ;

	      vrgout_r0s2 = _vel_vmrg_vvvml(_vel_vbrdl_vsl(0UL, vl), vrgout_r0s2, vmall_r0s2, vl) ;
	      vrgout_r0s0 = _vel_vmrg_vvvml(_vel_vbrdl_vsl(0UL, vl), vrgout_r0s0, vmall_r0s0, vl) ;
	      __vr vrgout_r0s1 = _vel_vshf_vvvsl(vrgout_r0s2, vrgout_r0s0, VE_VSHUFFLE_ZLYU, vl) ;
	      VFADD8(vrgout_r0s2) ; pKerValue-- ;
	      VFADD8(vrgout_r0s1) ; pKerValue-- ;
	      VFADD8(vrgout_r0s0) ; pKerValue-- ;
#undef VFADD8
	    } // gOutChannel

	    _vel_vst_vssl(vrsum0, 8, pGIn+gInIndex, vl) ;
	    _vel_vst_vssl(vrsum1, 8, pGIn+gInIndex+  gInPixels, vl) ;
	    _vel_vst_vssl(vrsum2, 8, pGIn+gInIndex+2*gInPixels, vl) ;
	    _vel_vst_vssl(vrsum3, 8, pGIn+gInIndex+3*gInPixels, vl) ;
	    _vel_vst_vssl(vrsum4, 8, pGIn+gInIndex+4*gInPixels, vl) ;
	    _vel_vst_vssl(vrsum5, 8, pGIn+gInIndex+5*gInPixels, vl) ;
	    _vel_vst_vssl(vrsum6, 8, pGIn+gInIndex+6*gInPixels, vl) ;
	    _vel_vst_vssl(vrsum7, 8, pGIn+gInIndex+7*gInPixels, vl) ;

	  } // gOutPixels

	  k+=8 ;
	}
	for (; k<gInChannelGroup; k+=16) {

	  for (int64_t h=0; h<gInHeight; h+=nH) {
	    const int64_t vl = gInWidthHalf * (gInHeight - h < nH ? gInHeight - h : nH) ;
	    const int64_t gip = h * gInWidth ;

	    const int64_t gInIndex = gInGroupOffset + ((n * gInChannel + k) * gInHeight ) * gInWidth + gip ;

	    __vr vrsum0 = _vel_pvbrd_vsl(0UL, vl) ;
	    __vr vrsum1 = _vel_pvbrd_vsl(0UL, vl) ;
	    __vr vrsum2 = _vel_pvbrd_vsl(0UL, vl) ;
	    __vr vrsum3 = _vel_pvbrd_vsl(0UL, vl) ;
	    __vr vrsum4 = _vel_pvbrd_vsl(0UL, vl) ;
	    __vr vrsum5 = _vel_pvbrd_vsl(0UL, vl) ;
	    __vr vrsum6 = _vel_pvbrd_vsl(0UL, vl) ;
	    __vr vrsum7 = _vel_pvbrd_vsl(0UL, vl) ;
	    __vr vrsum8 = _vel_pvbrd_vsl(0UL, vl) ;
	    __vr vrsum9 = _vel_pvbrd_vsl(0UL, vl) ;
	    __vr vrsumA = _vel_pvbrd_vsl(0UL, vl) ;
	    __vr vrsumB = _vel_pvbrd_vsl(0UL, vl) ;
	    __vr vrsumC = _vel_pvbrd_vsl(0UL, vl) ;
	    __vr vrsumD = _vel_pvbrd_vsl(0UL, vl) ;
	    __vr vrsumE = _vel_pvbrd_vsl(0UL, vl) ;
	    __vr vrsumF = _vel_pvbrd_vsl(0UL, vl) ;

	    __vr vry_r2 = _vel_vaddsl_vsvl(h-2, vrh, vl) ;
	    __vm256 vmy1_r2 =  _vel_vfmklge_mvl(vry_r2, vl) ;
	    __vm256 vmy_r2 = vmy1_r2 ;

	    __vm256 vmall_r2s2 = _vel_andm_mmm(vmy_r2,vmx_s2) ;
	    __vm256 vmall_r2s0 = _vel_andm_mmm(vmy_r2,vmx_s0) ;

	    __vr vry_r1 = _vel_vaddsl_vsvl(h-1, vrh, vl) ;
	    __vm256 vmy1_r1 =  _vel_vfmklge_mvl(vry_r1, vl) ;
	    __vm256 vmy2_r1 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutHeight,vry_r1, vl), vl) ;
	    __vm256 vmy_r1 = _vel_andm_mmm(vmy1_r1, vmy2_r1) ;

	    __vm256 vmall_r1s2 = _vel_andm_mmm(vmy_r1,vmx_s2) ;
	    __vm256 vmall_r1s0 = _vel_andm_mmm(vmy_r1,vmx_s0) ;

	    __vr vry_r0= _vel_vaddsl_vsvl(h, vrh, vl) ;
	    __vm256 vmy2_r0 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutHeight,vry_r0, vl), vl) ;
	    __vm256 vmy_r0 = vmy2_r0 ;

	    __vm256 vmall_r0s2 = _vel_andm_mmm(vmy_r0,vmx_s2) ;
	    __vm256 vmall_r0s0 = _vel_andm_mmm(vmy_r0,vmx_s0) ;

	    for (int64_t c=0; c<gOutChannelGroup; c++) {
	      int64_t gOutIndex    = gOutGroupOffset + ((n * gOutChannel + c) * gOutHeight) * gOutWidth ;

	      const float *pKerValue = pKernel + kernGroupOffset + ((c * gInChannelGroup + k) * kernHeight + 2) * kernWidth + 2;

	      __vr vrgout_ptr_r2s2 = _vel_vsfa_vvssl(vrhw, 2, (unsigned long)(pGOut+gOutIndex+(h-2)*gOutHeight-2), vl) ;
	      __vr vrgout_ptr_r1s2 = _vel_vsfa_vvssl(vrhw, 2, (unsigned long)(pGOut+gOutIndex+(h-1)*gOutHeight-2), vl) ;
	      __vr vrgout_ptr_r0s2 = _vel_vsfa_vvssl(vrhw, 2, (unsigned long)(pGOut+gOutIndex+(h-0)*gOutHeight-2), vl) ;

	      __vr vrgout_r2s2 = _vel_vgt_vvssml(vrgout_ptr_r2s2, 0, 0, vmall_r2s2, vl) ;
	      __vr vrgout_r1s2 = _vel_vgt_vvssml(vrgout_ptr_r1s2, 0, 0, vmall_r1s2, vl) ;
	      __vr vrgout_r0s2 = _vel_vgt_vvssml(vrgout_ptr_r0s2, 0, 0, vmall_r0s2, vl) ;

	      __vr vrgout_r2s0 = _vel_vmv_vsvl(1, vrgout_r2s2, vl) ;
	      __vr vrgout_r1s0 = _vel_vmv_vsvl(1, vrgout_r1s2, vl) ;
	      __vr vrgout_r0s0 = _vel_vmv_vsvl(1, vrgout_r0s2, vl) ;

#define VFADD16(VRGOUT)												\
{														\
  vrsum0 = _vel_pvfmad_vvsvl(vrsum0, _vel_pack_f32a(pKerValue),                               VRGOUT, vl) ;	\
  vrsum1 = _vel_pvfmad_vvsvl(vrsum1, _vel_pack_f32a(pKerValue +     kernHeight * kernWidth ), VRGOUT, vl) ;	\
  vrsum2 = _vel_pvfmad_vvsvl(vrsum2, _vel_pack_f32a(pKerValue + 2 * kernHeight * kernWidth ), VRGOUT, vl) ;	\
  vrsum3 = _vel_pvfmad_vvsvl(vrsum3, _vel_pack_f32a(pKerValue + 3 * kernHeight * kernWidth ), VRGOUT, vl) ;	\
  vrsum4 = _vel_pvfmad_vvsvl(vrsum4, _vel_pack_f32a(pKerValue + 4 * kernHeight * kernWidth ), VRGOUT, vl) ;	\
  vrsum5 = _vel_pvfmad_vvsvl(vrsum5, _vel_pack_f32a(pKerValue + 5 * kernHeight * kernWidth ), VRGOUT, vl) ;	\
  vrsum6 = _vel_pvfmad_vvsvl(vrsum6, _vel_pack_f32a(pKerValue + 6 * kernHeight * kernWidth ), VRGOUT, vl) ;	\
  vrsum7 = _vel_pvfmad_vvsvl(vrsum7, _vel_pack_f32a(pKerValue + 7 * kernHeight * kernWidth ), VRGOUT, vl) ;	\
  vrsum8 = _vel_pvfmad_vvsvl(vrsum8, _vel_pack_f32a(pKerValue + 8 * kernHeight * kernWidth ), VRGOUT, vl) ;	\
  vrsum9 = _vel_pvfmad_vvsvl(vrsum9, _vel_pack_f32a(pKerValue + 9 * kernHeight * kernWidth ), VRGOUT, vl) ;	\
  vrsumA = _vel_pvfmad_vvsvl(vrsumA, _vel_pack_f32a(pKerValue +10 * kernHeight * kernWidth ), VRGOUT, vl) ;	\
  vrsumB = _vel_pvfmad_vvsvl(vrsumB, _vel_pack_f32a(pKerValue +11 * kernHeight * kernWidth ), VRGOUT, vl) ;	\
  vrsumC = _vel_pvfmad_vvsvl(vrsumC, _vel_pack_f32a(pKerValue +12 * kernHeight * kernWidth ), VRGOUT, vl) ;	\
  vrsumD = _vel_pvfmad_vvsvl(vrsumD, _vel_pack_f32a(pKerValue +13 * kernHeight * kernWidth ), VRGOUT, vl) ;	\
  vrsumE = _vel_pvfmad_vvsvl(vrsumE, _vel_pack_f32a(pKerValue +14 * kernHeight * kernWidth ), VRGOUT, vl) ;	\
  vrsumF = _vel_pvfmad_vvsvl(vrsumF, _vel_pack_f32a(pKerValue +15 * kernHeight * kernWidth ), VRGOUT, vl) ;	\
}

	      vrgout_r2s2 = _vel_vmrg_vvvml(_vel_vbrdl_vsl(0UL, vl), vrgout_r2s2, vmall_r2s2, vl) ;
	      vrgout_r2s0 = _vel_vmrg_vvvml(_vel_vbrdl_vsl(0UL, vl), vrgout_r2s0, vmall_r2s0, vl) ;
	      __vr vrgout_r2s1 = _vel_vshf_vvvsl(vrgout_r2s2, vrgout_r2s0, VE_VSHUFFLE_ZLYU, vl) ;
	      VFADD16(vrgout_r2s2) ; pKerValue-- ;
	      VFADD16(vrgout_r2s1) ; pKerValue-- ;
	      VFADD16(vrgout_r2s0) ; pKerValue-- ;

	      vrgout_r1s2 = _vel_vmrg_vvvml(_vel_vbrdl_vsl(0UL, vl), vrgout_r1s2, vmall_r1s2, vl) ;
	      vrgout_r1s0 = _vel_vmrg_vvvml(_vel_vbrdl_vsl(0UL, vl), vrgout_r1s0, vmall_r1s0, vl) ;
	      __vr vrgout_r1s1 = _vel_vshf_vvvsl(vrgout_r1s2, vrgout_r1s0, VE_VSHUFFLE_ZLYU, vl) ;
	      VFADD16(vrgout_r1s2) ; pKerValue-- ;
	      VFADD16(vrgout_r1s1) ; pKerValue-- ;
	      VFADD16(vrgout_r1s0) ; pKerValue-- ;

	      vrgout_r0s2 = _vel_vmrg_vvvml(_vel_vbrdl_vsl(0UL, vl), vrgout_r0s2, vmall_r0s2, vl) ;
	      vrgout_r0s0 = _vel_vmrg_vvvml(_vel_vbrdl_vsl(0UL, vl), vrgout_r0s0, vmall_r0s0, vl) ;
	      __vr vrgout_r0s1 = _vel_vshf_vvvsl(vrgout_r0s2, vrgout_r0s0, VE_VSHUFFLE_ZLYU, vl) ;
	      VFADD16(vrgout_r0s2) ; pKerValue-- ;
	      VFADD16(vrgout_r0s1) ; pKerValue-- ;
	      VFADD16(vrgout_r0s0) ; pKerValue-- ;
#undef VFADD16
	    } // gOutChannel

	    _vel_vst_vssl(vrsum0, 8, pGIn+gInIndex, vl) ;
	    _vel_vst_vssl(vrsum1, 8, pGIn+gInIndex+  gInPixels, vl) ;
	    _vel_vst_vssl(vrsum2, 8, pGIn+gInIndex+2*gInPixels, vl) ;
	    _vel_vst_vssl(vrsum3, 8, pGIn+gInIndex+3*gInPixels, vl) ;
	    _vel_vst_vssl(vrsum4, 8, pGIn+gInIndex+4*gInPixels, vl) ;
	    _vel_vst_vssl(vrsum5, 8, pGIn+gInIndex+5*gInPixels, vl) ;
	    _vel_vst_vssl(vrsum6, 8, pGIn+gInIndex+6*gInPixels, vl) ;
	    _vel_vst_vssl(vrsum7, 8, pGIn+gInIndex+7*gInPixels, vl) ;
	    _vel_vst_vssl(vrsum8, 8, pGIn+gInIndex+8*gInPixels, vl) ;
	    _vel_vst_vssl(vrsum9, 8, pGIn+gInIndex+9*gInPixels, vl) ;
	    _vel_vst_vssl(vrsumA, 8, pGIn+gInIndex+10*gInPixels, vl) ;
	    _vel_vst_vssl(vrsumB, 8, pGIn+gInIndex+11*gInPixels, vl) ;
	    _vel_vst_vssl(vrsumC, 8, pGIn+gInIndex+12*gInPixels, vl) ;
	    _vel_vst_vssl(vrsumD, 8, pGIn+gInIndex+13*gInPixels, vl) ;
	    _vel_vst_vssl(vrsumE, 8, pGIn+gInIndex+14*gInPixels, vl) ;
	    _vel_vst_vssl(vrsumF, 8, pGIn+gInIndex+15*gInPixels, vl) ;

	  } // gOutPixels
	} // gInChannel
      } // group
    } // batch
  }


  return VEDNN_SUCCESS;
}
