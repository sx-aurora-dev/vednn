#include <stdint.h>
#include <stdlib.h>

#include "vednn.h"

#include "velintrin.h"
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

    __vr vrseq = _vel_vseq_vl(VLEN) ;
    __vr vrh  = _vel_vdivsl_vvsl(vrseq, gInWidthHalf, VLEN) ;
    __vr vrw  = _vel_vmulsl_vsvl(2, _vel_vsubsl_vvvl(vrseq, _vel_vmulul_vsvl(gInWidthHalf,vrh, VLEN), VLEN), VLEN) ;
    __vr vrhw = _vel_vaddsl_vvvl(_vel_vmulsl_vsvl(gOutWidth, vrh, VLEN), vrw, VLEN) ;

    __vr vrx_s2 = _vel_vaddsl_vsvl(-2, vrw, VLEN) ;
    __vm256 vmx1_s2 =  _vel_vfmklge_mvl(vrx_s2, VLEN) ;
    __vm256 vmx_s2 = vmx1_s2 ;

    __vr vrx_s0 = vrw ;
    __vm256 vmx2_s0 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutWidth,vrx_s0, VLEN), VLEN) ;
    __vm256 vmx_s0 = vmx2_s0 ;


    // vector mask registers for vld2d ( using unroll >= 4 )
    __vm256 vm_s2, vm_s0 ;
    {
      __vr vrseq0  = _vel_vaddsl_vsvl(15,vrseq, VLEN) ;
      __vr vrh_s0  = _vel_vdivsl_vvsl(vrseq0, 16, VLEN) ;
      __vr vrw_s0  = _vel_vsubsl_vvvl(vrseq0, _vel_vmulul_vsvl(16,vrh_s0, VLEN), VLEN) ;
      vm_s0 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gInWidthHalf, vrw_s0, VLEN), VLEN) ; // condition(x<gInWidthHalf)

      __vr vrh_s2  = _vel_vdivsl_vvsl(vrseq, 16, VLEN) ;
      __vr vrw_s2  = _vel_vsubsl_vvvl(vrseq, _vel_vmulul_vsvl(16,vrh_s2, VLEN), VLEN) ;
      vm_s2 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gInWidthHalf, vrw_s2, VLEN), VLEN) ; // condition(x<gInWidthHalf)

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

	      __vr vrgout_ptr_r2s2 = _vel_vsfa_vvssl(vrhw, 2, (unsigned long)(pGOut+gOutIndex+(h-2)*gOutWidth-2), vl) ;
	      __vr vrgout_ptr_r1s2 = _vel_vsfa_vvssl(vrhw, 2, (unsigned long)(pGOut+gOutIndex+(h-1)*gOutWidth-2), vl) ;
	      __vr vrgout_ptr_r0s2 = _vel_vsfa_vvssl(vrhw, 2, (unsigned long)(pGOut+gOutIndex+(h-0)*gOutWidth-2), vl) ;

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

	      __vr vrgout_ptr_r2s2 = _vel_vsfa_vvssl(vrhw, 2, (unsigned long)(pGOut+gOutIndex+(h-2)*gOutWidth-2), vl) ;
	      __vr vrgout_ptr_r1s2 = _vel_vsfa_vvssl(vrhw, 2, (unsigned long)(pGOut+gOutIndex+(h-1)*gOutWidth-2), vl) ;
	      __vr vrgout_ptr_r0s2 = _vel_vsfa_vvssl(vrhw, 2, (unsigned long)(pGOut+gOutIndex+(h-0)*gOutWidth-2), vl) ;

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
	  for (int64_t h=0; h<gInHeight; h+=16) {
	    const int64_t vl0 = 16 * (gInHeight - h < 16 ? gInHeight - h : 16) ;
	    const int64_t vl1 = gInWidthHalf * (gInHeight - h < 16 ? gInHeight - h : 16) ;

	    const int64_t gip = h * gInWidth ;

	    const int64_t gInIndex = gInGroupOffset + ((n * gInChannel + k) * gInHeight ) * gInWidth + gip ;

	    __vr vrsum0 = _vel_pvbrd_vsl(0UL, vl1) ;
	    __vr vrsum1 = _vel_pvbrd_vsl(0UL, vl1) ;
	    __vr vrsum2 = _vel_pvbrd_vsl(0UL, vl1) ;
	    __vr vrsum3 = _vel_pvbrd_vsl(0UL, vl1) ;

	    __vr vry_r2 = _vel_vaddsl_vsvl(h-2, vrh, vl1) ;
	    __vm256 vmy1_r2 =  _vel_vfmklge_mvl(vry_r2, vl1) ;
	    __vm256 vmy_r2 = vmy1_r2 ;

	    __vm256 vmall_r2s2 = _vel_andm_mmm(vmy_r2,vmx_s2) ;
	    __vm256 vmall_r2s0 = _vel_andm_mmm(vmy_r2,vmx_s0) ;

	    __vr vry_r1 = _vel_vaddsl_vsvl(h-1, vrh, vl1) ;
	    __vm256 vmy1_r1 =  _vel_vfmklge_mvl(vry_r1, vl1) ;
	    __vm256 vmy2_r1 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutHeight,vry_r1, vl1), vl1) ;
	    __vm256 vmy_r1 = _vel_andm_mmm(vmy1_r1, vmy2_r1) ;

	    __vm256 vmall_r1s2 = _vel_andm_mmm(vmy_r1,vmx_s2) ;
	    __vm256 vmall_r1s0 = _vel_andm_mmm(vmy_r1,vmx_s0) ;

	    __vr vry_r0= _vel_vaddsl_vsvl(h, vrh, vl1) ;
	    __vm256 vmy2_r0 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutHeight,vry_r0, vl1), vl1) ;
	    __vm256 vmy_r0 = vmy2_r0 ;

	    __vm256 vmall_r0s2 = _vel_andm_mmm(vmy_r0,vmx_s2) ;
	    __vm256 vmall_r0s0 = _vel_andm_mmm(vmy_r0,vmx_s0) ;

	    for (int64_t c=0; c<gOutChannelGroup; c++) {
	      int64_t gOutIndex    = gOutGroupOffset + ((n * gOutChannel + c) * gOutHeight) * gOutWidth ;

	      const float *pKerValue = pKernel + kernGroupOffset + ((c * gInChannelGroup + k) * kernHeight + 2) * kernWidth + 2;

	      __vr vrgout_r2 = _vel_vld2d_vssl((gOutWidthHalf<<(3+16))+8, pGOut+gOutIndex+(h-2)*gOutWidth-2, vl0) ;
	      __vr vrgout_r1 = _vel_vld2d_vssl((gOutWidthHalf<<(3+16))+8, pGOut+gOutIndex+(h-1)*gOutWidth-2, vl0) ;
	      __vr vrgout_r0 = _vel_vld2d_vssl((gOutWidthHalf<<(3+16))+8, pGOut+gOutIndex+(h-0)*gOutWidth-2, vl0) ;

	      __vr vrgout_r2s2 = _vel_vcp_vvmvl(vrgout_r2, vm_s2, _vel_vbrdl_vsl(0UL, vl0), vl0) ;
	      __vr vrgout_r2s0 = _vel_vcp_vvmvl(vrgout_r2, vm_s0, _vel_vbrdl_vsl(0UL, vl0), vl0) ;

	      __vr vrgout_r1s2 = _vel_vcp_vvmvl(vrgout_r1, vm_s2, _vel_vbrdl_vsl(0UL, vl0), vl0) ;
	      __vr vrgout_r1s0 = _vel_vcp_vvmvl(vrgout_r1, vm_s0, _vel_vbrdl_vsl(0UL, vl0), vl0) ;

	      __vr vrgout_r0s2 = _vel_vcp_vvmvl(vrgout_r0, vm_s2, _vel_vbrdl_vsl(0UL, vl0), vl0) ;
	      __vr vrgout_r0s0 = _vel_vcp_vvmvl(vrgout_r0, vm_s0, _vel_vbrdl_vsl(0UL, vl0), vl0) ;

#define VFADD4(VRGOUT)												\
{														\
  vrsum0 = _vel_pvfmad_vvsvl(vrsum0, _vel_pack_f32a(pKerValue),                               VRGOUT, vl1) ;	\
  vrsum1 = _vel_pvfmad_vvsvl(vrsum1, _vel_pack_f32a(pKerValue +     kernHeight * kernWidth ), VRGOUT, vl1) ;	\
  vrsum2 = _vel_pvfmad_vvsvl(vrsum2, _vel_pack_f32a(pKerValue + 2 * kernHeight * kernWidth ), VRGOUT, vl1) ;	\
  vrsum3 = _vel_pvfmad_vvsvl(vrsum3, _vel_pack_f32a(pKerValue + 3 * kernHeight * kernWidth ), VRGOUT, vl1) ;	\
}

	      vrgout_r2s2 = _vel_vmrg_vvvml(_vel_vbrdl_vsl(0UL, vl1), vrgout_r2s2, vmall_r2s2, vl1) ;
	      vrgout_r2s0 = _vel_vmrg_vvvml(_vel_vbrdl_vsl(0UL, vl1), vrgout_r2s0, vmall_r2s0, vl1) ;
	      __vr vrgout_r2s1 = _vel_vshf_vvvsl(vrgout_r2s2, vrgout_r2s0, VE_VSHUFFLE_ZLYU, vl1) ;
	      VFADD4(vrgout_r2s2) ; pKerValue-- ;
	      VFADD4(vrgout_r2s1) ; pKerValue-- ;
	      VFADD4(vrgout_r2s0) ; pKerValue-- ;

	      vrgout_r1s2 = _vel_vmrg_vvvml(_vel_vbrdl_vsl(0UL, vl1), vrgout_r1s2, vmall_r1s2, vl1) ;
	      vrgout_r1s0 = _vel_vmrg_vvvml(_vel_vbrdl_vsl(0UL, vl1), vrgout_r1s0, vmall_r1s0, vl1) ;
	      __vr vrgout_r1s1 = _vel_vshf_vvvsl(vrgout_r1s2, vrgout_r1s0, VE_VSHUFFLE_ZLYU, vl1) ;
	      VFADD4(vrgout_r1s2) ; pKerValue-- ;
	      VFADD4(vrgout_r1s1) ; pKerValue-- ;
	      VFADD4(vrgout_r1s0) ; pKerValue-- ;

	      vrgout_r0s2 = _vel_vmrg_vvvml(_vel_vbrdl_vsl(0UL, vl1), vrgout_r0s2, vmall_r0s2, vl1) ;
	      vrgout_r0s0 = _vel_vmrg_vvvml(_vel_vbrdl_vsl(0UL, vl1), vrgout_r0s0, vmall_r0s0, vl1) ;
	      __vr vrgout_r0s1 = _vel_vshf_vvvsl(vrgout_r0s2, vrgout_r0s0, VE_VSHUFFLE_ZLYU, vl1) ;
	      VFADD4(vrgout_r0s2) ; pKerValue-- ;
	      VFADD4(vrgout_r0s1) ; pKerValue-- ;
	      VFADD4(vrgout_r0s0) ; pKerValue-- ;
#undef VFADD4
	    } // gOutChannel

	    _vel_vst_vssl(vrsum0, 8, pGIn+gInIndex, vl1) ;
	    _vel_vst_vssl(vrsum1, 8, pGIn+gInIndex+  gInPixels, vl1) ;
	    _vel_vst_vssl(vrsum2, 8, pGIn+gInIndex+2*gInPixels, vl1) ;
	    _vel_vst_vssl(vrsum3, 8, pGIn+gInIndex+3*gInPixels, vl1) ;

	  } // gOutPixels

	  k+=4 ;
	}
	if( ((gInChannelGroup>>3) & 0x01 ) == 1 ) {
	  for (int64_t h=0; h<gInHeight; h+=16) {
	    const int64_t vl0 = 16 * (gInHeight - h < 16 ? gInHeight - h : 16) ;
	    const int64_t vl1 = gInWidthHalf * (gInHeight - h < 16 ? gInHeight - h : 16) ;

	    const int64_t gip = h * gInWidth ;

	    const int64_t gInIndex = gInGroupOffset + ((n * gInChannel + k) * gInHeight ) * gInWidth + gip ;

	    __vr vrsum0 = _vel_pvbrd_vsl(0UL, vl1) ;
	    __vr vrsum1 = _vel_pvbrd_vsl(0UL, vl1) ;
	    __vr vrsum2 = _vel_pvbrd_vsl(0UL, vl1) ;
	    __vr vrsum3 = _vel_pvbrd_vsl(0UL, vl1) ;
	    __vr vrsum4 = _vel_pvbrd_vsl(0UL, vl1) ;
	    __vr vrsum5 = _vel_pvbrd_vsl(0UL, vl1) ;
	    __vr vrsum6 = _vel_pvbrd_vsl(0UL, vl1) ;
	    __vr vrsum7 = _vel_pvbrd_vsl(0UL, vl1) ;

	    __vr vry_r2 = _vel_vaddsl_vsvl(h-2, vrh, vl1) ;
	    __vm256 vmy1_r2 =  _vel_vfmklge_mvl(vry_r2, vl1) ;
	    __vm256 vmy_r2 = vmy1_r2 ;

	    __vm256 vmall_r2s2 = _vel_andm_mmm(vmy_r2,vmx_s2) ;
	    __vm256 vmall_r2s0 = _vel_andm_mmm(vmy_r2,vmx_s0) ;

	    __vr vry_r1 = _vel_vaddsl_vsvl(h-1, vrh, vl1) ;
	    __vm256 vmy1_r1 =  _vel_vfmklge_mvl(vry_r1, vl1) ;
	    __vm256 vmy2_r1 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutHeight,vry_r1, vl1), vl1) ;
	    __vm256 vmy_r1 = _vel_andm_mmm(vmy1_r1, vmy2_r1) ;

	    __vm256 vmall_r1s2 = _vel_andm_mmm(vmy_r1,vmx_s2) ;
	    __vm256 vmall_r1s0 = _vel_andm_mmm(vmy_r1,vmx_s0) ;

	    __vr vry_r0= _vel_vaddsl_vsvl(h, vrh, vl1) ;
	    __vm256 vmy2_r0 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutHeight,vry_r0, vl1), vl1) ;
	    __vm256 vmy_r0 = vmy2_r0 ;

	    __vm256 vmall_r0s2 = _vel_andm_mmm(vmy_r0,vmx_s2) ;
	    __vm256 vmall_r0s0 = _vel_andm_mmm(vmy_r0,vmx_s0) ;

	    for (int64_t c=0; c<gOutChannelGroup; c++) {
	      int64_t gOutIndex    = gOutGroupOffset + ((n * gOutChannel + c) * gOutHeight) * gOutWidth ;

	      const float *pKerValue = pKernel + kernGroupOffset + ((c * gInChannelGroup + k) * kernHeight + 2) * kernWidth + 2;

	      __vr vrgout_r2 = _vel_vld2d_vssl((gOutWidthHalf<<(3+16))+8, pGOut+gOutIndex+(h-2)*gOutWidth-2, vl0) ;
	      __vr vrgout_r1 = _vel_vld2d_vssl((gOutWidthHalf<<(3+16))+8, pGOut+gOutIndex+(h-1)*gOutWidth-2, vl0) ;
	      __vr vrgout_r0 = _vel_vld2d_vssl((gOutWidthHalf<<(3+16))+8, pGOut+gOutIndex+(h-0)*gOutWidth-2, vl0) ;

	      __vr vrgout_r2s2 = _vel_vcp_vvmvl(vrgout_r2, vm_s2, _vel_vbrdl_vsl(0UL, vl0), vl0) ;
	      __vr vrgout_r2s0 = _vel_vcp_vvmvl(vrgout_r2, vm_s0, _vel_vbrdl_vsl(0UL, vl0), vl0) ;

	      __vr vrgout_r1s2 = _vel_vcp_vvmvl(vrgout_r1, vm_s2, _vel_vbrdl_vsl(0UL, vl0), vl0) ;
	      __vr vrgout_r1s0 = _vel_vcp_vvmvl(vrgout_r1, vm_s0, _vel_vbrdl_vsl(0UL, vl0), vl0) ;

	      __vr vrgout_r0s2 = _vel_vcp_vvmvl(vrgout_r0, vm_s2, _vel_vbrdl_vsl(0UL, vl0), vl0) ;
	      __vr vrgout_r0s0 = _vel_vcp_vvmvl(vrgout_r0, vm_s0, _vel_vbrdl_vsl(0UL, vl0), vl0) ;

#define VFADD8(VRGOUT)												\
{														\
  vrsum0 = _vel_pvfmad_vvsvl(vrsum0, _vel_pack_f32a(pKerValue),                               VRGOUT, vl1) ;	\
  vrsum1 = _vel_pvfmad_vvsvl(vrsum1, _vel_pack_f32a(pKerValue +     kernHeight * kernWidth ), VRGOUT, vl1) ;	\
  vrsum2 = _vel_pvfmad_vvsvl(vrsum2, _vel_pack_f32a(pKerValue + 2 * kernHeight * kernWidth ), VRGOUT, vl1) ;	\
  vrsum3 = _vel_pvfmad_vvsvl(vrsum3, _vel_pack_f32a(pKerValue + 3 * kernHeight * kernWidth ), VRGOUT, vl1) ;	\
  vrsum4 = _vel_pvfmad_vvsvl(vrsum4, _vel_pack_f32a(pKerValue + 4 * kernHeight * kernWidth ), VRGOUT, vl1) ;	\
  vrsum5 = _vel_pvfmad_vvsvl(vrsum5, _vel_pack_f32a(pKerValue + 5 * kernHeight * kernWidth ), VRGOUT, vl1) ;	\
  vrsum6 = _vel_pvfmad_vvsvl(vrsum6, _vel_pack_f32a(pKerValue + 6 * kernHeight * kernWidth ), VRGOUT, vl1) ;	\
  vrsum7 = _vel_pvfmad_vvsvl(vrsum7, _vel_pack_f32a(pKerValue + 7 * kernHeight * kernWidth ), VRGOUT, vl1) ;	\
}

	      vrgout_r2s2 = _vel_vmrg_vvvml(_vel_vbrdl_vsl(0UL, vl1), vrgout_r2s2, vmall_r2s2, vl1) ;
	      vrgout_r2s0 = _vel_vmrg_vvvml(_vel_vbrdl_vsl(0UL, vl1), vrgout_r2s0, vmall_r2s0, vl1) ;
	      __vr vrgout_r2s1 = _vel_vshf_vvvsl(vrgout_r2s2, vrgout_r2s0, VE_VSHUFFLE_ZLYU, vl1) ;
	      VFADD8(vrgout_r2s2) ; pKerValue-- ;
	      VFADD8(vrgout_r2s1) ; pKerValue-- ;
	      VFADD8(vrgout_r2s0) ; pKerValue-- ;

	      vrgout_r1s2 = _vel_vmrg_vvvml(_vel_vbrdl_vsl(0UL, vl1), vrgout_r1s2, vmall_r1s2, vl1) ;
	      vrgout_r1s0 = _vel_vmrg_vvvml(_vel_vbrdl_vsl(0UL, vl1), vrgout_r1s0, vmall_r1s0, vl1) ;
	      __vr vrgout_r1s1 = _vel_vshf_vvvsl(vrgout_r1s2, vrgout_r1s0, VE_VSHUFFLE_ZLYU, vl1) ;
	      VFADD8(vrgout_r1s2) ; pKerValue-- ;
	      VFADD8(vrgout_r1s1) ; pKerValue-- ;
	      VFADD8(vrgout_r1s0) ; pKerValue-- ;

	      vrgout_r0s2 = _vel_vmrg_vvvml(_vel_vbrdl_vsl(0UL, vl1), vrgout_r0s2, vmall_r0s2, vl1) ;
	      vrgout_r0s0 = _vel_vmrg_vvvml(_vel_vbrdl_vsl(0UL, vl1), vrgout_r0s0, vmall_r0s0, vl1) ;
	      __vr vrgout_r0s1 = _vel_vshf_vvvsl(vrgout_r0s2, vrgout_r0s0, VE_VSHUFFLE_ZLYU, vl1) ;
	      VFADD8(vrgout_r0s2) ; pKerValue-- ;
	      VFADD8(vrgout_r0s1) ; pKerValue-- ;
	      VFADD8(vrgout_r0s0) ; pKerValue-- ;
#undef VFADD8
	    } // gOutChannel

	    _vel_vst_vssl(vrsum0, 8, pGIn+gInIndex, vl1) ;
	    _vel_vst_vssl(vrsum1, 8, pGIn+gInIndex+  gInPixels, vl1) ;
	    _vel_vst_vssl(vrsum2, 8, pGIn+gInIndex+2*gInPixels, vl1) ;
	    _vel_vst_vssl(vrsum3, 8, pGIn+gInIndex+3*gInPixels, vl1) ;
	    _vel_vst_vssl(vrsum4, 8, pGIn+gInIndex+4*gInPixels, vl1) ;
	    _vel_vst_vssl(vrsum5, 8, pGIn+gInIndex+5*gInPixels, vl1) ;
	    _vel_vst_vssl(vrsum6, 8, pGIn+gInIndex+6*gInPixels, vl1) ;
	    _vel_vst_vssl(vrsum7, 8, pGIn+gInIndex+7*gInPixels, vl1) ;

	  } // gOutPixels

	  k+=8 ;
	}
	for (; k<gInChannelGroup; k+=16) {
	  for (int64_t h=0; h<gInHeight; h+=16) {
	    const int64_t vl0 = 16 * (gInHeight - h < 16 ? gInHeight - h : 16) ;
	    const int64_t vl1 = gInWidthHalf * (gInHeight - h < 16 ? gInHeight - h : 16) ;

	    const int64_t gip = h * gInWidth ;

	    const int64_t gInIndex = gInGroupOffset + ((n * gInChannel + k) * gInHeight ) * gInWidth + gip ;

	    __vr vrsum0 = _vel_pvbrd_vsl(0UL, vl1) ;
	    __vr vrsum1 = _vel_pvbrd_vsl(0UL, vl1) ;
	    __vr vrsum2 = _vel_pvbrd_vsl(0UL, vl1) ;
	    __vr vrsum3 = _vel_pvbrd_vsl(0UL, vl1) ;
	    __vr vrsum4 = _vel_pvbrd_vsl(0UL, vl1) ;
	    __vr vrsum5 = _vel_pvbrd_vsl(0UL, vl1) ;
	    __vr vrsum6 = _vel_pvbrd_vsl(0UL, vl1) ;
	    __vr vrsum7 = _vel_pvbrd_vsl(0UL, vl1) ;
	    __vr vrsum8 = _vel_pvbrd_vsl(0UL, vl1) ;
	    __vr vrsum9 = _vel_pvbrd_vsl(0UL, vl1) ;
	    __vr vrsumA = _vel_pvbrd_vsl(0UL, vl1) ;
	    __vr vrsumB = _vel_pvbrd_vsl(0UL, vl1) ;
	    __vr vrsumC = _vel_pvbrd_vsl(0UL, vl1) ;
	    __vr vrsumD = _vel_pvbrd_vsl(0UL, vl1) ;
	    __vr vrsumE = _vel_pvbrd_vsl(0UL, vl1) ;
	    __vr vrsumF = _vel_pvbrd_vsl(0UL, vl1) ;

	    __vr vry_r2 = _vel_vaddsl_vsvl(h-2, vrh, vl1) ;
	    __vm256 vmy1_r2 =  _vel_vfmklge_mvl(vry_r2, vl1) ;
	    __vm256 vmy_r2 = vmy1_r2 ;

	    __vm256 vmall_r2s2 = _vel_andm_mmm(vmy_r2,vmx_s2) ;
	    __vm256 vmall_r2s0 = _vel_andm_mmm(vmy_r2,vmx_s0) ;

	    __vr vry_r1 = _vel_vaddsl_vsvl(h-1, vrh, vl1) ;
	    __vm256 vmy1_r1 =  _vel_vfmklge_mvl(vry_r1, vl1) ;
	    __vm256 vmy2_r1 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutHeight,vry_r1, vl1), vl1) ;
	    __vm256 vmy_r1 = _vel_andm_mmm(vmy1_r1, vmy2_r1) ;

	    __vm256 vmall_r1s2 = _vel_andm_mmm(vmy_r1,vmx_s2) ;
	    __vm256 vmall_r1s0 = _vel_andm_mmm(vmy_r1,vmx_s0) ;

	    __vr vry_r0= _vel_vaddsl_vsvl(h, vrh, vl1) ;
	    __vm256 vmy2_r0 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutHeight,vry_r0, vl1), vl1) ;
	    __vm256 vmy_r0 = vmy2_r0 ;

	    __vm256 vmall_r0s2 = _vel_andm_mmm(vmy_r0,vmx_s2) ;
	    __vm256 vmall_r0s0 = _vel_andm_mmm(vmy_r0,vmx_s0) ;

	    for (int64_t c=0; c<gOutChannelGroup; c++) {
	      int64_t gOutIndex    = gOutGroupOffset + ((n * gOutChannel + c) * gOutHeight) * gOutWidth ;

	      const float *pKerValue = pKernel + kernGroupOffset + ((c * gInChannelGroup + k) * kernHeight + 2) * kernWidth + 2;

	      __vr vrgout_r2 = _vel_vld2d_vssl((gOutWidthHalf<<(3+16))+8, pGOut+gOutIndex+(h-2)*gOutWidth-2, vl0) ;
	      __vr vrgout_r1 = _vel_vld2d_vssl((gOutWidthHalf<<(3+16))+8, pGOut+gOutIndex+(h-1)*gOutWidth-2, vl0) ;
	      __vr vrgout_r0 = _vel_vld2d_vssl((gOutWidthHalf<<(3+16))+8, pGOut+gOutIndex+(h-0)*gOutWidth-2, vl0) ;

	      __vr vrgout_r2s2 = _vel_vcp_vvmvl(vrgout_r2, vm_s2, _vel_vbrdl_vsl(0UL, vl0), vl0) ;
	      __vr vrgout_r2s0 = _vel_vcp_vvmvl(vrgout_r2, vm_s0, _vel_vbrdl_vsl(0UL, vl0), vl0) ;

	      __vr vrgout_r1s2 = _vel_vcp_vvmvl(vrgout_r1, vm_s2, _vel_vbrdl_vsl(0UL, vl0), vl0) ;
	      __vr vrgout_r1s0 = _vel_vcp_vvmvl(vrgout_r1, vm_s0, _vel_vbrdl_vsl(0UL, vl0), vl0) ;

	      __vr vrgout_r0s2 = _vel_vcp_vvmvl(vrgout_r0, vm_s2, _vel_vbrdl_vsl(0UL, vl0), vl0) ;
	      __vr vrgout_r0s0 = _vel_vcp_vvmvl(vrgout_r0, vm_s0, _vel_vbrdl_vsl(0UL, vl0), vl0) ;

#define VFADD16(VRGOUT)											\
{													\
  vrsum0 = _vel_pvfmad_vvsvl(vrsum0, _vel_pack_f32a(pKerValue),                               VRGOUT, vl1) ;	\
  vrsum1 = _vel_pvfmad_vvsvl(vrsum1, _vel_pack_f32a(pKerValue +     kernHeight * kernWidth ), VRGOUT, vl1) ;	\
  vrsum2 = _vel_pvfmad_vvsvl(vrsum2, _vel_pack_f32a(pKerValue + 2 * kernHeight * kernWidth ), VRGOUT, vl1) ;	\
  vrsum3 = _vel_pvfmad_vvsvl(vrsum3, _vel_pack_f32a(pKerValue + 3 * kernHeight * kernWidth ), VRGOUT, vl1) ;	\
  vrsum4 = _vel_pvfmad_vvsvl(vrsum4, _vel_pack_f32a(pKerValue + 4 * kernHeight * kernWidth ), VRGOUT, vl1) ;	\
  vrsum5 = _vel_pvfmad_vvsvl(vrsum5, _vel_pack_f32a(pKerValue + 5 * kernHeight * kernWidth ), VRGOUT, vl1) ;	\
  vrsum6 = _vel_pvfmad_vvsvl(vrsum6, _vel_pack_f32a(pKerValue + 6 * kernHeight * kernWidth ), VRGOUT, vl1) ;	\
  vrsum7 = _vel_pvfmad_vvsvl(vrsum7, _vel_pack_f32a(pKerValue + 7 * kernHeight * kernWidth ), VRGOUT, vl1) ;	\
  vrsum8 = _vel_pvfmad_vvsvl(vrsum8, _vel_pack_f32a(pKerValue + 8 * kernHeight * kernWidth ), VRGOUT, vl1) ;	\
  vrsum9 = _vel_pvfmad_vvsvl(vrsum9, _vel_pack_f32a(pKerValue + 9 * kernHeight * kernWidth ), VRGOUT, vl1) ;	\
  vrsumA = _vel_pvfmad_vvsvl(vrsumA, _vel_pack_f32a(pKerValue +10 * kernHeight * kernWidth ), VRGOUT, vl1) ;	\
  vrsumB = _vel_pvfmad_vvsvl(vrsumB, _vel_pack_f32a(pKerValue +11 * kernHeight * kernWidth ), VRGOUT, vl1) ;	\
  vrsumC = _vel_pvfmad_vvsvl(vrsumC, _vel_pack_f32a(pKerValue +12 * kernHeight * kernWidth ), VRGOUT, vl1) ;	\
  vrsumD = _vel_pvfmad_vvsvl(vrsumD, _vel_pack_f32a(pKerValue +13 * kernHeight * kernWidth ), VRGOUT, vl1) ;	\
  vrsumE = _vel_pvfmad_vvsvl(vrsumE, _vel_pack_f32a(pKerValue +14 * kernHeight * kernWidth ), VRGOUT, vl1) ;	\
  vrsumF = _vel_pvfmad_vvsvl(vrsumF, _vel_pack_f32a(pKerValue +15 * kernHeight * kernWidth ), VRGOUT, vl1) ;	\
}

	      vrgout_r2s2 = _vel_vmrg_vvvml(_vel_vbrdl_vsl(0UL, vl1), vrgout_r2s2, vmall_r2s2, vl1) ;
	      vrgout_r2s0 = _vel_vmrg_vvvml(_vel_vbrdl_vsl(0UL, vl1), vrgout_r2s0, vmall_r2s0, vl1) ;
	      __vr vrgout_r2s1 = _vel_vshf_vvvsl(vrgout_r2s2, vrgout_r2s0, VE_VSHUFFLE_ZLYU, vl1) ;
	      VFADD16(vrgout_r2s2) ; pKerValue-- ;
	      VFADD16(vrgout_r2s1) ; pKerValue-- ;
	      VFADD16(vrgout_r2s0) ; pKerValue-- ;

	      vrgout_r1s2 = _vel_vmrg_vvvml(_vel_vbrdl_vsl(0UL, vl1), vrgout_r1s2, vmall_r1s2, vl1) ;
	      vrgout_r1s0 = _vel_vmrg_vvvml(_vel_vbrdl_vsl(0UL, vl1), vrgout_r1s0, vmall_r1s0, vl1) ;
	      __vr vrgout_r1s1 = _vel_vshf_vvvsl(vrgout_r1s2, vrgout_r1s0, VE_VSHUFFLE_ZLYU, vl1) ;
	      VFADD16(vrgout_r1s2) ; pKerValue-- ;
	      VFADD16(vrgout_r1s1) ; pKerValue-- ;
	      VFADD16(vrgout_r1s0) ; pKerValue-- ;

	      vrgout_r0s2 = _vel_vmrg_vvvml(_vel_vbrdl_vsl(0UL, vl1), vrgout_r0s2, vmall_r0s2, vl1) ;
	      vrgout_r0s0 = _vel_vmrg_vvvml(_vel_vbrdl_vsl(0UL, vl1), vrgout_r0s0, vmall_r0s0, vl1) ;
	      __vr vrgout_r0s1 = _vel_vshf_vvvsl(vrgout_r0s2, vrgout_r0s0, VE_VSHUFFLE_ZLYU, vl1) ;
	      VFADD16(vrgout_r0s2) ; pKerValue-- ;
	      VFADD16(vrgout_r0s1) ; pKerValue-- ;
	      VFADD16(vrgout_r0s0) ; pKerValue-- ;
#undef VFADD16
	    } // gOutChannel

	    _vel_vst_vssl(vrsum0, 8, pGIn+gInIndex, vl1) ;
	    _vel_vst_vssl(vrsum1, 8, pGIn+gInIndex+  gInPixels, vl1) ;
	    _vel_vst_vssl(vrsum2, 8, pGIn+gInIndex+2*gInPixels, vl1) ;
	    _vel_vst_vssl(vrsum3, 8, pGIn+gInIndex+3*gInPixels, vl1) ;
	    _vel_vst_vssl(vrsum4, 8, pGIn+gInIndex+4*gInPixels, vl1) ;
	    _vel_vst_vssl(vrsum5, 8, pGIn+gInIndex+5*gInPixels, vl1) ;
	    _vel_vst_vssl(vrsum6, 8, pGIn+gInIndex+6*gInPixels, vl1) ;
	    _vel_vst_vssl(vrsum7, 8, pGIn+gInIndex+7*gInPixels, vl1) ;
	    _vel_vst_vssl(vrsum8, 8, pGIn+gInIndex+8*gInPixels, vl1) ;
	    _vel_vst_vssl(vrsum9, 8, pGIn+gInIndex+9*gInPixels, vl1) ;
	    _vel_vst_vssl(vrsumA, 8, pGIn+gInIndex+10*gInPixels, vl1) ;
	    _vel_vst_vssl(vrsumB, 8, pGIn+gInIndex+11*gInPixels, vl1) ;
	    _vel_vst_vssl(vrsumC, 8, pGIn+gInIndex+12*gInPixels, vl1) ;
	    _vel_vst_vssl(vrsumD, 8, pGIn+gInIndex+13*gInPixels, vl1) ;
	    _vel_vst_vssl(vrsumE, 8, pGIn+gInIndex+14*gInPixels, vl1) ;
	    _vel_vst_vssl(vrsumF, 8, pGIn+gInIndex+15*gInPixels, vl1) ;

	  } // gOutPixels
	} // gInChannel
      } // group
    } // batch
  }

  return VEDNN_SUCCESS;
}
