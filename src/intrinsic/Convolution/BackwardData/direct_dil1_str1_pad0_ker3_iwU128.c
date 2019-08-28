#include <stdint.h>
#include <stdlib.h>

#include "vednn.h"

#include "velintrin.h"
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

  {
    const int64_t nH = VLEN / gInWidth ;

    __vr vrseq = _vel_vseq_vl(nH*gInWidth) ;
    __vr vrh  = _vel_vdivsl_vvsl(vrseq, gInWidth, nH*gInWidth) ;
    __vr vrw  = _vel_vsubsl_vvvl(vrseq, _vel_vmulul_vsvl(gInWidth,vrh, nH*gInWidth), nH*gInWidth) ;
    __vr vrhw = _vel_vaddsl_vvvl(_vel_vmulsl_vsvl(gOutWidth, vrh, nH*gInWidth), vrw, nH*gInWidth) ;

    __vr vrx_s2 = _vel_vaddsl_vsvl(-2, vrw, nH*gInWidth) ;
    __vm256 vmx1_s2 =  _vel_vfmklge_mvl(vrx_s2, nH*gInWidth) ;
    __vm256 vmx_s2 = vmx1_s2 ;

    __vr vrx_s1 = _vel_vaddsl_vsvl(-1, vrw, nH*gInWidth) ;
    __vm256 vmx1_s1 =  _vel_vfmklge_mvl(vrx_s1, nH*gInWidth) ;
    __vm256 vmx2_s1 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutWidth,vrx_s1, nH*gInWidth), nH*gInWidth) ;
    __vm256 vmx_s1 = _vel_andm_mmm(vmx1_s1, vmx2_s1) ;

    __vr vrx_s0 = vrw ;
    __vm256 vmx2_s0 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutWidth,vrx_s0, nH*gInWidth), nH*gInWidth) ;
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

	    __vr vrsum = _vel_vbrds_vsl(0.f, vl) ;

	    __vr vry_r2 = _vel_vaddsl_vsvl(h-2, vrh, vl) ;
	    __vm256 vmy1_r2 =  _vel_vfmklge_mvl(vry_r2, vl) ;
	    __vm256 vmy_r2 = vmy1_r2 ;

	    __vm256 vmall_r2s2 = _vel_andm_mmm(vmy_r2,vmx_s2) ;
	    __vm256 vmall_r2s1 = _vel_andm_mmm(vmy_r2,vmx_s1) ;
	    __vm256 vmall_r2s0 = _vel_andm_mmm(vmy_r2,vmx_s0) ;

	    __vr vry_r1 = _vel_vaddsl_vsvl(h-1, vrh, vl) ;
	    __vm256 vmy1_r1 =  _vel_vfmklge_mvl(vry_r1, vl) ;
	    __vm256 vmy2_r1 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutHeight,vry_r1, vl), vl) ;
	    __vm256 vmy_r1 = _vel_andm_mmm(vmy1_r1, vmy2_r1) ;

	    __vm256 vmall_r1s2 = _vel_andm_mmm(vmy_r1,vmx_s2) ;
	    __vm256 vmall_r1s1 = _vel_andm_mmm(vmy_r1,vmx_s1) ;
	    __vm256 vmall_r1s0 = _vel_andm_mmm(vmy_r1,vmx_s0) ;

	    __vr vry_r0= _vel_vaddsl_vsvl(h, vrh, vl) ;
	    __vm256 vmy2_r0 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutHeight,vry_r0, vl), vl) ;
	    __vm256 vmy_r0 = vmy2_r0 ;

	    __vm256 vmall_r0s2 = _vel_andm_mmm(vmy_r0,vmx_s2) ;
	    __vm256 vmall_r0s1 = _vel_andm_mmm(vmy_r0,vmx_s1) ;
	    __vm256 vmall_r0s0 = _vel_andm_mmm(vmy_r0,vmx_s0) ;

	    for (int64_t c=0; c<gOutChannelGroup; c++) {
	      int64_t gOutIndex    = gOutGroupOffset + ((n * gOutChannel + c) * gOutHeight) * gOutWidth ;

	      const float *pKerValue = pKernel + kernGroupOffset + ((c * gInChannelGroup + k) * kernHeight + 2) * kernWidth + 2;

	      __vr vrgout_ptr_r2s2 = _vel_vsfa_vvssl(vrhw, 2, (unsigned long)(pGOut+gOutIndex+(h-2)*gOutWidth-2), vl) ;
	      __vr vrgout_ptr_r1s2 = _vel_vsfa_vvssl(vrhw, 2, (unsigned long)(pGOut+gOutIndex+(h-1)*gOutWidth-2), vl) ;
	      __vr vrgout_ptr_r0s2 = _vel_vsfa_vvssl(vrhw, 2, (unsigned long)(pGOut+gOutIndex+(h-0)*gOutWidth-2), vl) ;

	      __vr vrgout_r2s2 = _vel_vgtu_vvssml(vrgout_ptr_r2s2, 0, 0, vmall_r2s2, vl) ;
	      __vr vrgout_r1s2 = _vel_vgtu_vvssml(vrgout_ptr_r1s2, 0, 0, vmall_r1s2, vl) ;
	      __vr vrgout_r0s2 = _vel_vgtu_vvssml(vrgout_ptr_r0s2, 0, 0, vmall_r0s2, vl) ;

	      __vr vrgout_r2s1 = _vel_vmv_vsvl(1, vrgout_r2s2, vl) ;
	      __vr vrgout_r2s0 = _vel_vmv_vsvl(2, vrgout_r2s2, vl) ;

	      __vr vrgout_r1s1 = _vel_vmv_vsvl(1, vrgout_r1s2, vl) ;
	      __vr vrgout_r1s0 = _vel_vmv_vsvl(2, vrgout_r1s2, vl) ;

	      __vr vrgout_r0s1 = _vel_vmv_vsvl(1, vrgout_r0s2, vl) ;
	      __vr vrgout_r0s0 = _vel_vmv_vsvl(2, vrgout_r0s2, vl) ;

#define VFADD1(VRGOUT)						\
{								\
  const float kerValue = pKerValue[0] ;				\
  vrsum = _vel_vfmads_vvsvl(vrsum, kerValue, VRGOUT, vl) ;	\
}

	      vrgout_r2s2 = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrgout_r2s2, vmall_r2s2, vl) ;
	      VFADD1(vrgout_r2s2) ; pKerValue-- ;
	      vrgout_r2s1 = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrgout_r2s1, vmall_r2s1, vl) ;
	      VFADD1(vrgout_r2s1) ; pKerValue-- ;
	      vrgout_r2s0 = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrgout_r2s0, vmall_r2s0, vl) ;
	      VFADD1(vrgout_r2s0) ; pKerValue-- ;

	      vrgout_r1s2 = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrgout_r1s2, vmall_r1s2, vl) ;
	      VFADD1(vrgout_r1s2) ; pKerValue-- ;
	      vrgout_r1s1 = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrgout_r1s1, vmall_r1s1, vl) ;
	      VFADD1(vrgout_r1s1) ; pKerValue-- ;
	      vrgout_r1s0 = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrgout_r1s0, vmall_r1s0, vl) ;
	      VFADD1(vrgout_r1s0) ; pKerValue-- ;

	      vrgout_r0s2 = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrgout_r0s2, vmall_r0s2, vl) ;
	      VFADD1(vrgout_r0s2) ; pKerValue-- ;
	      vrgout_r0s1 = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrgout_r0s1, vmall_r0s1, vl) ;
	      VFADD1(vrgout_r0s1) ; pKerValue-- ;
	      vrgout_r0s0 = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrgout_r0s0, vmall_r0s0, vl) ;
	      VFADD1(vrgout_r0s0) ; pKerValue-- ;
#undef VFADD1
	    } // gOutChannel

	    _vel_vstu_vssl(vrsum, 4, pGIn+gInIndex, vl) ;

	  } // gOutPixels


	  k++ ;
	}
	if( ((gInChannelGroup>>1) & 0x01 ) == 1 ) {

	  for (int64_t h=0; h<gInHeight; h+=nH) {
	    const int64_t vl = gInWidth * (gInHeight - h < nH ? gInHeight - h : nH) ;
	    const int64_t gip = h * gInWidth ;

	    const int64_t gInIndex = gInGroupOffset + ((n * gInChannel + k) * gInHeight ) * gInWidth + gip ;

	    __vr vrsum01 = _vel_pvbrd_vsl(0UL, vl) ;

	    __vr vry_r2 = _vel_vaddsl_vsvl(h-2, vrh, vl) ;
	    __vm256 vmy1_r2 =  _vel_vfmklge_mvl(vry_r2, vl) ;
	    __vm256 vmy_r2 = vmy1_r2 ;

	    __vm256 vmall_r2s2 = _vel_andm_mmm(vmy_r2,vmx_s2) ;
	    __vm256 vmall_r2s1 = _vel_andm_mmm(vmy_r2,vmx_s1) ;
	    __vm256 vmall_r2s0 = _vel_andm_mmm(vmy_r2,vmx_s0) ;

	    __vr vry_r1 = _vel_vaddsl_vsvl(h-1, vrh, vl) ;
	    __vm256 vmy1_r1 =  _vel_vfmklge_mvl(vry_r1, vl) ;
	    __vm256 vmy2_r1 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutHeight,vry_r1, vl), vl) ;
	    __vm256 vmy_r1 = _vel_andm_mmm(vmy1_r1, vmy2_r1) ;

	    __vm256 vmall_r1s2 = _vel_andm_mmm(vmy_r1,vmx_s2) ;
	    __vm256 vmall_r1s1 = _vel_andm_mmm(vmy_r1,vmx_s1) ;
	    __vm256 vmall_r1s0 = _vel_andm_mmm(vmy_r1,vmx_s0) ;

	    __vr vry_r0= _vel_vaddsl_vsvl(h, vrh, vl) ;
	    __vm256 vmy2_r0 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutHeight,vry_r0, vl), vl) ;
	    __vm256 vmy_r0 = vmy2_r0 ;

	    __vm256 vmall_r0s2 = _vel_andm_mmm(vmy_r0,vmx_s2) ;
	    __vm256 vmall_r0s1 = _vel_andm_mmm(vmy_r0,vmx_s1) ;
	    __vm256 vmall_r0s0 = _vel_andm_mmm(vmy_r0,vmx_s0) ;

	    for (int64_t c=0; c<gOutChannelGroup; c++) {
	      int64_t gOutIndex    = gOutGroupOffset + ((n * gOutChannel + c) * gOutHeight) * gOutWidth ;

	      const float *pKerValue = pKernel + kernGroupOffset + ((c * gInChannelGroup + k) * kernHeight + 2) * kernWidth + 2;

	      __vr vrgout_ptr_r2s2 = _vel_vsfa_vvssl(vrhw, 2, (unsigned long)(pGOut+gOutIndex+(h-2)*gOutWidth-2), vl) ;
	      __vr vrgout_ptr_r1s2 = _vel_vsfa_vvssl(vrhw, 2, (unsigned long)(pGOut+gOutIndex+(h-1)*gOutWidth-2), vl) ;
	      __vr vrgout_ptr_r0s2 = _vel_vsfa_vvssl(vrhw, 2, (unsigned long)(pGOut+gOutIndex+(h-0)*gOutWidth-2), vl) ;

	      __vr vrgout_r2s2 = _vel_vgtu_vvssml(vrgout_ptr_r2s2, 0, 0, vmall_r2s2, vl) ;
	      __vr vrgout_r1s2 = _vel_vgtu_vvssml(vrgout_ptr_r1s2, 0, 0, vmall_r1s2, vl) ;
	      __vr vrgout_r0s2 = _vel_vgtu_vvssml(vrgout_ptr_r0s2, 0, 0, vmall_r0s2, vl) ;

	      __vr vrgout_r2s1 = _vel_vmv_vsvl(1, vrgout_r2s2, vl) ;
	      __vr vrgout_r2s0 = _vel_vmv_vsvl(2, vrgout_r2s2, vl) ;

	      __vr vrgout_r1s1 = _vel_vmv_vsvl(1, vrgout_r1s2, vl) ;
	      __vr vrgout_r1s0 = _vel_vmv_vsvl(2, vrgout_r1s2, vl) ;

	      __vr vrgout_r0s1 = _vel_vmv_vsvl(1, vrgout_r0s2, vl) ;
	      __vr vrgout_r0s0 = _vel_vmv_vsvl(2, vrgout_r0s2, vl) ;

#define VFADD2(VRGOUT)									\
{											\
  __vr vrgoutP = _vel_vshf_vvvsl(VRGOUT, VRGOUT, VE_VSHUFFLE_YUZU, vl) ;		\
  const uint64_t kerValue01 = _vel_pack_f32p(pKerValue,					\
					     pKerValue + kernHeight * kernWidth ) ;	\
  vrsum01 = _vel_pvfmad_vvsvl(vrsum01, kerValue01, vrgoutP, vl) ;			\
}

	      vrgout_r2s2 = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrgout_r2s2, vmall_r2s2, vl) ;
	      VFADD2(vrgout_r2s2) ; pKerValue-- ;
	      vrgout_r2s1 = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrgout_r2s1, vmall_r2s1, vl) ;
	      VFADD2(vrgout_r2s1) ; pKerValue-- ;
	      vrgout_r2s0 = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrgout_r2s0, vmall_r2s0, vl) ;
	      VFADD2(vrgout_r2s0) ; pKerValue-- ;

	      vrgout_r1s2 = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrgout_r1s2, vmall_r1s2, vl) ;
	      VFADD2(vrgout_r1s2) ; pKerValue-- ;
	      vrgout_r1s1 = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrgout_r1s1, vmall_r1s1, vl) ;
	      VFADD2(vrgout_r1s1) ; pKerValue-- ;
	      vrgout_r1s0 = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrgout_r1s0, vmall_r1s0, vl) ;
	      VFADD2(vrgout_r1s0) ; pKerValue-- ;

	      vrgout_r0s2 = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrgout_r0s2, vmall_r0s2, vl) ;
	      VFADD2(vrgout_r0s2) ; pKerValue-- ;
	      vrgout_r0s1 = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrgout_r0s1, vmall_r0s1, vl) ;
	      VFADD2(vrgout_r0s1) ; pKerValue-- ;
	      vrgout_r0s0 = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrgout_r0s0, vmall_r0s0, vl) ;
	      VFADD2(vrgout_r0s0) ; pKerValue-- ;
#undef VFADD2
	    } // gOutChannel

	    _vel_vstu_vssl(vrsum01, 4, pGIn+gInIndex, vl) ;
	    _vel_vstl_vssl(vrsum01, 4, pGIn+gInIndex+  gInPixels, vl) ;

	  } // gOutPixels


	  k+=2 ;
	}
	if( ((gInChannelGroup>>2) & 0x01 ) == 1 ) {

	  for (int64_t h=0; h<gInHeight; h+=nH) {
	    const int64_t vl = gInWidth * (gInHeight - h < nH ? gInHeight - h : nH) ;
	    const int64_t gip = h * gInWidth ;

	    const int64_t gInIndex = gInGroupOffset + ((n * gInChannel + k) * gInHeight ) * gInWidth + gip ;

	    __vr vrsum01 = _vel_pvbrd_vsl(0UL, vl) ;
	    __vr vrsum23 = _vel_pvbrd_vsl(0UL, vl) ;

	    __vr vry_r2 = _vel_vaddsl_vsvl(h-2, vrh, vl) ;
	    __vm256 vmy1_r2 =  _vel_vfmklge_mvl(vry_r2, vl) ;
	    __vm256 vmy_r2 = vmy1_r2 ;

	    __vm256 vmall_r2s2 = _vel_andm_mmm(vmy_r2,vmx_s2) ;
	    __vm256 vmall_r2s1 = _vel_andm_mmm(vmy_r2,vmx_s1) ;
	    __vm256 vmall_r2s0 = _vel_andm_mmm(vmy_r2,vmx_s0) ;

	    __vr vry_r1 = _vel_vaddsl_vsvl(h-1, vrh, vl) ;
	    __vm256 vmy1_r1 =  _vel_vfmklge_mvl(vry_r1, vl) ;
	    __vm256 vmy2_r1 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutHeight,vry_r1, vl), vl) ;
	    __vm256 vmy_r1 = _vel_andm_mmm(vmy1_r1, vmy2_r1) ;

	    __vm256 vmall_r1s2 = _vel_andm_mmm(vmy_r1,vmx_s2) ;
	    __vm256 vmall_r1s1 = _vel_andm_mmm(vmy_r1,vmx_s1) ;
	    __vm256 vmall_r1s0 = _vel_andm_mmm(vmy_r1,vmx_s0) ;

	    __vr vry_r0= _vel_vaddsl_vsvl(h, vrh, vl) ;
	    __vm256 vmy2_r0 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutHeight,vry_r0, vl), vl) ;
	    __vm256 vmy_r0 = vmy2_r0 ;

	    __vm256 vmall_r0s2 = _vel_andm_mmm(vmy_r0,vmx_s2) ;
	    __vm256 vmall_r0s1 = _vel_andm_mmm(vmy_r0,vmx_s1) ;
	    __vm256 vmall_r0s0 = _vel_andm_mmm(vmy_r0,vmx_s0) ;

	    for (int64_t c=0; c<gOutChannelGroup; c++) {
	      int64_t gOutIndex    = gOutGroupOffset + ((n * gOutChannel + c) * gOutHeight) * gOutWidth ;

	      const float *pKerValue = pKernel + kernGroupOffset + ((c * gInChannelGroup + k) * kernHeight + 2) * kernWidth + 2;

	      __vr vrgout_ptr_r2s2 = _vel_vsfa_vvssl(vrhw, 2, (unsigned long)(pGOut+gOutIndex+(h-2)*gOutWidth-2), vl) ;
	      __vr vrgout_ptr_r1s2 = _vel_vsfa_vvssl(vrhw, 2, (unsigned long)(pGOut+gOutIndex+(h-1)*gOutWidth-2), vl) ;
	      __vr vrgout_ptr_r0s2 = _vel_vsfa_vvssl(vrhw, 2, (unsigned long)(pGOut+gOutIndex+(h-0)*gOutWidth-2), vl) ;

	      __vr vrgout_r2s2 = _vel_vgtu_vvssml(vrgout_ptr_r2s2, 0, 0, vmall_r2s2, vl) ;
	      __vr vrgout_r1s2 = _vel_vgtu_vvssml(vrgout_ptr_r1s2, 0, 0, vmall_r1s2, vl) ;
	      __vr vrgout_r0s2 = _vel_vgtu_vvssml(vrgout_ptr_r0s2, 0, 0, vmall_r0s2, vl) ;

	      __vr vrgout_r2s1 = _vel_vmv_vsvl(1, vrgout_r2s2, vl) ;
	      __vr vrgout_r2s0 = _vel_vmv_vsvl(2, vrgout_r2s2, vl) ;

	      __vr vrgout_r1s1 = _vel_vmv_vsvl(1, vrgout_r1s2, vl) ;
	      __vr vrgout_r1s0 = _vel_vmv_vsvl(2, vrgout_r1s2, vl) ;

	      __vr vrgout_r0s1 = _vel_vmv_vsvl(1, vrgout_r0s2, vl) ;
	      __vr vrgout_r0s0 = _vel_vmv_vsvl(2, vrgout_r0s2, vl) ;

#define VFADD4(VRGOUT)									\
{											\
  __vr vrgoutP = _vel_vshf_vvvsl(VRGOUT, VRGOUT, VE_VSHUFFLE_YUZU, vl) ;		\
  const uint64_t kerValue01 = _vel_pack_f32p(pKerValue,					\
					     pKerValue + kernHeight * kernWidth ) ;	\
  const uint64_t kerValue23 = _vel_pack_f32p(pKerValue + 2 * kernHeight * kernWidth,	\
					     pKerValue + 3 * kernHeight * kernWidth ) ;	\
  vrsum01 = _vel_pvfmad_vvsvl(vrsum01, kerValue01, vrgoutP, vl) ;		\
  vrsum23 = _vel_pvfmad_vvsvl(vrsum23, kerValue23, vrgoutP, vl) ;		\
}

	      vrgout_r2s2 = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrgout_r2s2, vmall_r2s2, vl) ;
	      VFADD4(vrgout_r2s2) ; pKerValue-- ;
	      vrgout_r2s1 = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrgout_r2s1, vmall_r2s1, vl) ;
	      VFADD4(vrgout_r2s1) ; pKerValue-- ;
	      vrgout_r2s0 = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrgout_r2s0, vmall_r2s0, vl) ;
	      VFADD4(vrgout_r2s0) ; pKerValue-- ;

	      vrgout_r1s2 = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrgout_r1s2, vmall_r1s2, vl) ;
	      VFADD4(vrgout_r1s2) ; pKerValue-- ;
	      vrgout_r1s1 = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrgout_r1s1, vmall_r1s1, vl) ;
	      VFADD4(vrgout_r1s1) ; pKerValue-- ;
	      vrgout_r1s0 = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrgout_r1s0, vmall_r1s0, vl) ;
	      VFADD4(vrgout_r1s0) ; pKerValue-- ;

	      vrgout_r0s2 = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrgout_r0s2, vmall_r0s2, vl) ;
	      VFADD4(vrgout_r0s2) ; pKerValue-- ;
	      vrgout_r0s1 = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrgout_r0s1, vmall_r0s1, vl) ;
	      VFADD4(vrgout_r0s1) ; pKerValue-- ;
	      vrgout_r0s0 = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrgout_r0s0, vmall_r0s0, vl) ;
	      VFADD4(vrgout_r0s0) ; pKerValue-- ;
#undef VFADD4
	    } // gOutChannel

	    _vel_vstu_vssl(vrsum01, 4, pGIn+gInIndex, vl) ;
	    _vel_vstl_vssl(vrsum01, 4, pGIn+gInIndex+  gInPixels, vl) ;
	    _vel_vstu_vssl(vrsum23, 4, pGIn+gInIndex+2*gInPixels, vl) ;
	    _vel_vstl_vssl(vrsum23, 4, pGIn+gInIndex+3*gInPixels, vl) ;

	  } // gOutPixels


	  k+=4 ;
	}
	if( ((gInChannelGroup>>3) & 0x01 ) == 1 ) {

	  for (int64_t h=0; h<gInHeight; h+=nH) {
	    const int64_t vl = gInWidth * (gInHeight - h < nH ? gInHeight - h : nH) ;
	    const int64_t gip = h * gInWidth ;

	    const int64_t gInIndex = gInGroupOffset + ((n * gInChannel + k) * gInHeight ) * gInWidth + gip ;

	    __vr vrsum01 = _vel_pvbrd_vsl(0UL, vl) ;
	    __vr vrsum23 = _vel_pvbrd_vsl(0UL, vl) ;
	    __vr vrsum45 = _vel_pvbrd_vsl(0UL, vl) ;
	    __vr vrsum67 = _vel_pvbrd_vsl(0UL, vl) ;

	    __vr vry_r2 = _vel_vaddsl_vsvl(h-2, vrh, vl) ;
	    __vm256 vmy1_r2 =  _vel_vfmklge_mvl(vry_r2, vl) ;
	    __vm256 vmy_r2 = vmy1_r2 ;

	    __vm256 vmall_r2s2 = _vel_andm_mmm(vmy_r2,vmx_s2) ;
	    __vm256 vmall_r2s1 = _vel_andm_mmm(vmy_r2,vmx_s1) ;
	    __vm256 vmall_r2s0 = _vel_andm_mmm(vmy_r2,vmx_s0) ;

	    __vr vry_r1 = _vel_vaddsl_vsvl(h-1, vrh, vl) ;
	    __vm256 vmy1_r1 =  _vel_vfmklge_mvl(vry_r1, vl) ;
	    __vm256 vmy2_r1 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutHeight,vry_r1, vl), vl) ;
	    __vm256 vmy_r1 = _vel_andm_mmm(vmy1_r1, vmy2_r1) ;

	    __vm256 vmall_r1s2 = _vel_andm_mmm(vmy_r1,vmx_s2) ;
	    __vm256 vmall_r1s1 = _vel_andm_mmm(vmy_r1,vmx_s1) ;
	    __vm256 vmall_r1s0 = _vel_andm_mmm(vmy_r1,vmx_s0) ;

	    __vr vry_r0= _vel_vaddsl_vsvl(h, vrh, vl) ;
	    __vm256 vmy2_r0 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutHeight,vry_r0, vl), vl) ;
	    __vm256 vmy_r0 = vmy2_r0 ;

	    __vm256 vmall_r0s2 = _vel_andm_mmm(vmy_r0,vmx_s2) ;
	    __vm256 vmall_r0s1 = _vel_andm_mmm(vmy_r0,vmx_s1) ;
	    __vm256 vmall_r0s0 = _vel_andm_mmm(vmy_r0,vmx_s0) ;

	    for (int64_t c=0; c<gOutChannelGroup; c++) {
	      int64_t gOutIndex    = gOutGroupOffset + ((n * gOutChannel + c) * gOutHeight) * gOutWidth ;

	      const float *pKerValue = pKernel + kernGroupOffset + ((c * gInChannelGroup + k) * kernHeight + 2) * kernWidth + 2;

	      __vr vrgout_ptr_r2s2 = _vel_vsfa_vvssl(vrhw, 2, (unsigned long)(pGOut+gOutIndex+(h-2)*gOutWidth-2), vl) ;
	      __vr vrgout_ptr_r1s2 = _vel_vsfa_vvssl(vrhw, 2, (unsigned long)(pGOut+gOutIndex+(h-1)*gOutWidth-2), vl) ;
	      __vr vrgout_ptr_r0s2 = _vel_vsfa_vvssl(vrhw, 2, (unsigned long)(pGOut+gOutIndex+(h-0)*gOutWidth-2), vl) ;

	      __vr vrgout_r2s2 = _vel_vgtu_vvssml(vrgout_ptr_r2s2, 0, 0, vmall_r2s2, vl) ;
	      __vr vrgout_r1s2 = _vel_vgtu_vvssml(vrgout_ptr_r1s2, 0, 0, vmall_r1s2, vl) ;
	      __vr vrgout_r0s2 = _vel_vgtu_vvssml(vrgout_ptr_r0s2, 0, 0, vmall_r0s2, vl) ;

	      __vr vrgout_r2s1 = _vel_vmv_vsvl(1, vrgout_r2s2, vl) ;
	      __vr vrgout_r2s0 = _vel_vmv_vsvl(2, vrgout_r2s2, vl) ;

	      __vr vrgout_r1s1 = _vel_vmv_vsvl(1, vrgout_r1s2, vl) ;
	      __vr vrgout_r1s0 = _vel_vmv_vsvl(2, vrgout_r1s2, vl) ;

	      __vr vrgout_r0s1 = _vel_vmv_vsvl(1, vrgout_r0s2, vl) ;
	      __vr vrgout_r0s0 = _vel_vmv_vsvl(2, vrgout_r0s2, vl) ;

#define VFADD8(VRGOUT)									\
{											\
  __vr vrgoutP = _vel_vshf_vvvsl(VRGOUT, VRGOUT, VE_VSHUFFLE_YUZU, vl) ;		\
  const uint64_t kerValue01 = _vel_pack_f32p(pKerValue,					\
					     pKerValue + kernHeight * kernWidth ) ;	\
  const uint64_t kerValue23 = _vel_pack_f32p(pKerValue + 2 * kernHeight * kernWidth,	\
					     pKerValue + 3 * kernHeight * kernWidth ) ;	\
  const uint64_t kerValue45 = _vel_pack_f32p(pKerValue + 4 * kernHeight * kernWidth,	\
					     pKerValue + 5 * kernHeight * kernWidth ) ;	\
  const uint64_t kerValue67 = _vel_pack_f32p(pKerValue + 6 * kernHeight * kernWidth,	\
					     pKerValue + 7 * kernHeight * kernWidth ) ;	\
  vrsum01 = _vel_pvfmad_vvsvl(vrsum01, kerValue01, vrgoutP, vl) ;		\
  vrsum23 = _vel_pvfmad_vvsvl(vrsum23, kerValue23, vrgoutP, vl) ;		\
  vrsum45 = _vel_pvfmad_vvsvl(vrsum45, kerValue45, vrgoutP, vl) ;		\
  vrsum67 = _vel_pvfmad_vvsvl(vrsum67, kerValue67, vrgoutP, vl) ;		\
}

	      vrgout_r2s2 = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrgout_r2s2, vmall_r2s2, vl) ;
	      VFADD8(vrgout_r2s2) ; pKerValue-- ;
	      vrgout_r2s1 = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrgout_r2s1, vmall_r2s1, vl) ;
	      VFADD8(vrgout_r2s1) ; pKerValue-- ;
	      vrgout_r2s0 = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrgout_r2s0, vmall_r2s0, vl) ;
	      VFADD8(vrgout_r2s0) ; pKerValue-- ;

	      vrgout_r1s2 = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrgout_r1s2, vmall_r1s2, vl) ;
	      VFADD8(vrgout_r1s2) ; pKerValue-- ;
	      vrgout_r1s1 = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrgout_r1s1, vmall_r1s1, vl) ;
	      VFADD8(vrgout_r1s1) ; pKerValue-- ;
	      vrgout_r1s0 = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrgout_r1s0, vmall_r1s0, vl) ;
	      VFADD8(vrgout_r1s0) ; pKerValue-- ;

	      vrgout_r0s2 = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrgout_r0s2, vmall_r0s2, vl) ;
	      VFADD8(vrgout_r0s2) ; pKerValue-- ;
	      vrgout_r0s1 = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrgout_r0s1, vmall_r0s1, vl) ;
	      VFADD8(vrgout_r0s1) ; pKerValue-- ;
	      vrgout_r0s0 = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrgout_r0s0, vmall_r0s0, vl) ;
	      VFADD8(vrgout_r0s0) ; pKerValue-- ;
#undef VFADD8
	    } // gOutChannel

	    _vel_vstu_vssl(vrsum01, 4, pGIn+gInIndex, vl) ;
	    _vel_vstl_vssl(vrsum01, 4, pGIn+gInIndex+  gInPixels, vl) ;
	    _vel_vstu_vssl(vrsum23, 4, pGIn+gInIndex+2*gInPixels, vl) ;
	    _vel_vstl_vssl(vrsum23, 4, pGIn+gInIndex+3*gInPixels, vl) ;
	    _vel_vstu_vssl(vrsum45, 4, pGIn+gInIndex+4*gInPixels, vl) ;
	    _vel_vstl_vssl(vrsum45, 4, pGIn+gInIndex+5*gInPixels, vl) ;
	    _vel_vstu_vssl(vrsum67, 4, pGIn+gInIndex+6*gInPixels, vl) ;
	    _vel_vstl_vssl(vrsum67, 4, pGIn+gInIndex+7*gInPixels, vl) ;

	  } // gOutPixels

	  k+=8 ;
	}
	for (; k<gInChannelGroup; k+=16) {

	  for (int64_t h=0; h<gInHeight; h+=nH) {
	    const int64_t vl = gInWidth * (gInHeight - h < nH ? gInHeight - h : nH) ;
	    const int64_t gip = h * gInWidth ;

	    const int64_t gInIndex = gInGroupOffset + ((n * gInChannel + k) * gInHeight ) * gInWidth + gip ;

	    __vr vrsum01 = _vel_pvbrd_vsl(0UL, vl) ;
	    __vr vrsum23 = _vel_pvbrd_vsl(0UL, vl) ;
	    __vr vrsum45 = _vel_pvbrd_vsl(0UL, vl) ;
	    __vr vrsum67 = _vel_pvbrd_vsl(0UL, vl) ;
	    __vr vrsum89 = _vel_pvbrd_vsl(0UL, vl) ;
	    __vr vrsumAB = _vel_pvbrd_vsl(0UL, vl) ;
	    __vr vrsumCD = _vel_pvbrd_vsl(0UL, vl) ;
	    __vr vrsumEF = _vel_pvbrd_vsl(0UL, vl) ;

	    __vr vry_r2 = _vel_vaddsl_vsvl(h-2, vrh, vl) ;
	    __vm256 vmy1_r2 =  _vel_vfmklge_mvl(vry_r2, vl) ;
	    __vm256 vmy_r2 = vmy1_r2 ;

	    __vm256 vmall_r2s2 = _vel_andm_mmm(vmy_r2,vmx_s2) ;
	    __vm256 vmall_r2s1 = _vel_andm_mmm(vmy_r2,vmx_s1) ;
	    __vm256 vmall_r2s0 = _vel_andm_mmm(vmy_r2,vmx_s0) ;

	    __vr vry_r1 = _vel_vaddsl_vsvl(h-1, vrh, vl) ;
	    __vm256 vmy1_r1 =  _vel_vfmklge_mvl(vry_r1, vl) ;
	    __vm256 vmy2_r1 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutHeight,vry_r1, vl), vl) ;
	    __vm256 vmy_r1 = _vel_andm_mmm(vmy1_r1, vmy2_r1) ;

	    __vm256 vmall_r1s2 = _vel_andm_mmm(vmy_r1,vmx_s2) ;
	    __vm256 vmall_r1s1 = _vel_andm_mmm(vmy_r1,vmx_s1) ;
	    __vm256 vmall_r1s0 = _vel_andm_mmm(vmy_r1,vmx_s0) ;

	    __vr vry_r0= _vel_vaddsl_vsvl(h, vrh, vl) ;
	    __vm256 vmy2_r0 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(gOutHeight,vry_r0, vl), vl) ;
	    __vm256 vmy_r0 = vmy2_r0 ;

	    __vm256 vmall_r0s2 = _vel_andm_mmm(vmy_r0,vmx_s2) ;
	    __vm256 vmall_r0s1 = _vel_andm_mmm(vmy_r0,vmx_s1) ;
	    __vm256 vmall_r0s0 = _vel_andm_mmm(vmy_r0,vmx_s0) ;

	    for (int64_t c=0; c<gOutChannelGroup; c++) {
	      int64_t gOutIndex    = gOutGroupOffset + ((n * gOutChannel + c) * gOutHeight) * gOutWidth ;

	      const float *pKerValue = pKernel + kernGroupOffset + ((c * gInChannelGroup + k) * kernHeight + 2) * kernWidth + 2;

	      __vr vrgout_ptr_r2s2 = _vel_vsfa_vvssl(vrhw, 2, (unsigned long)(pGOut+gOutIndex+(h-2)*gOutWidth-2), vl) ;
	      __vr vrgout_ptr_r1s2 = _vel_vsfa_vvssl(vrhw, 2, (unsigned long)(pGOut+gOutIndex+(h-1)*gOutWidth-2), vl) ;
	      __vr vrgout_ptr_r0s2 = _vel_vsfa_vvssl(vrhw, 2, (unsigned long)(pGOut+gOutIndex+(h-0)*gOutWidth-2), vl) ;

	      __vr vrgout_r2s2 = _vel_vgtu_vvssml(vrgout_ptr_r2s2, 0, 0, vmall_r2s2, vl) ;
	      __vr vrgout_r1s2 = _vel_vgtu_vvssml(vrgout_ptr_r1s2, 0, 0, vmall_r1s2, vl) ;
	      __vr vrgout_r0s2 = _vel_vgtu_vvssml(vrgout_ptr_r0s2, 0, 0, vmall_r0s2, vl) ;

	      __vr vrgout_r2s1 = _vel_vmv_vsvl(1, vrgout_r2s2, vl) ;
	      __vr vrgout_r2s0 = _vel_vmv_vsvl(2, vrgout_r2s2, vl) ;

	      __vr vrgout_r1s1 = _vel_vmv_vsvl(1, vrgout_r1s2, vl) ;
	      __vr vrgout_r1s0 = _vel_vmv_vsvl(2, vrgout_r1s2, vl) ;

	      __vr vrgout_r0s1 = _vel_vmv_vsvl(1, vrgout_r0s2, vl) ;
	      __vr vrgout_r0s0 = _vel_vmv_vsvl(2, vrgout_r0s2, vl) ;

#define VFADD16(VRGOUT)									\
{											\
  __vr vrgoutP = _vel_vshf_vvvsl(VRGOUT, VRGOUT, VE_VSHUFFLE_YUZU, vl) ;		\
  const uint64_t kerValue01 = _vel_pack_f32p(pKerValue,					\
					     pKerValue + kernHeight * kernWidth ) ;	\
  const uint64_t kerValue23 = _vel_pack_f32p(pKerValue + 2 * kernHeight * kernWidth,	\
					     pKerValue + 3 * kernHeight * kernWidth ) ;	\
  const uint64_t kerValue45 = _vel_pack_f32p(pKerValue + 4 * kernHeight * kernWidth,	\
					     pKerValue + 5 * kernHeight * kernWidth ) ;	\
  const uint64_t kerValue67 = _vel_pack_f32p(pKerValue + 6 * kernHeight * kernWidth,	\
					     pKerValue + 7 * kernHeight * kernWidth ) ;	\
  const uint64_t kerValue89 = _vel_pack_f32p(pKerValue + 8 * kernHeight * kernWidth,	\
					     pKerValue + 9 * kernHeight * kernWidth ) ;	\
  const uint64_t kerValueAB = _vel_pack_f32p(pKerValue +10 * kernHeight * kernWidth,	\
					     pKerValue +11* kernHeight * kernWidth ) ;	\
  const uint64_t kerValueCD = _vel_pack_f32p(pKerValue +12 * kernHeight * kernWidth,	\
					     pKerValue +13 * kernHeight * kernWidth ) ;	\
  const uint64_t kerValueEF = _vel_pack_f32p(pKerValue +14 * kernHeight * kernWidth,	\
					    pKerValue +15 * kernHeight * kernWidth ) ;	\
  vrsum01 = _vel_pvfmad_vvsvl(vrsum01, kerValue01, vrgoutP, vl) ;		\
  vrsum23 = _vel_pvfmad_vvsvl(vrsum23, kerValue23, vrgoutP, vl) ;		\
  vrsum45 = _vel_pvfmad_vvsvl(vrsum45, kerValue45, vrgoutP, vl) ;		\
  vrsum67 = _vel_pvfmad_vvsvl(vrsum67, kerValue67, vrgoutP, vl) ;		\
  vrsum89 = _vel_pvfmad_vvsvl(vrsum89, kerValue89, vrgoutP, vl) ;		\
  vrsumAB = _vel_pvfmad_vvsvl(vrsumAB, kerValueAB, vrgoutP, vl) ;		\
  vrsumCD = _vel_pvfmad_vvsvl(vrsumCD, kerValueCD, vrgoutP, vl) ;		\
  vrsumEF = _vel_pvfmad_vvsvl(vrsumEF, kerValueEF, vrgoutP, vl) ;		\
}

	      vrgout_r2s2 = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrgout_r2s2, vmall_r2s2, vl) ;
	      VFADD16(vrgout_r2s2) ; pKerValue-- ;
	      vrgout_r2s1 = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrgout_r2s1, vmall_r2s1, vl) ;
	      VFADD16(vrgout_r2s1) ; pKerValue-- ;
	      vrgout_r2s0 = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrgout_r2s0, vmall_r2s0, vl) ;
	      VFADD16(vrgout_r2s0) ; pKerValue-- ;

	      vrgout_r1s2 = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrgout_r1s2, vmall_r1s2, vl) ;
	      VFADD16(vrgout_r1s2) ; pKerValue-- ;
	      vrgout_r1s1 = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrgout_r1s1, vmall_r1s1, vl) ;
	      VFADD16(vrgout_r1s1) ; pKerValue-- ;
	      vrgout_r1s0 = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrgout_r1s0, vmall_r1s0, vl) ;
	      VFADD16(vrgout_r1s0) ; pKerValue-- ;

	      vrgout_r0s2 = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrgout_r0s2, vmall_r0s2, vl) ;
	      VFADD16(vrgout_r0s2) ; pKerValue-- ;
	      vrgout_r0s1 = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrgout_r0s1, vmall_r0s1, vl) ;
	      VFADD16(vrgout_r0s1) ; pKerValue-- ;
	      vrgout_r0s0 = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrgout_r0s0, vmall_r0s0, vl) ;
	      VFADD16(vrgout_r0s0) ; pKerValue-- ;
#undef VFADD16
	    } // gOutChannel

	    _vel_vstu_vssl(vrsum01, 4, pGIn+gInIndex, vl) ;
	    _vel_vstl_vssl(vrsum01, 4, pGIn+gInIndex+  gInPixels, vl) ;
	    _vel_vstu_vssl(vrsum23, 4, pGIn+gInIndex+2*gInPixels, vl) ;
	    _vel_vstl_vssl(vrsum23, 4, pGIn+gInIndex+3*gInPixels, vl) ;
	    _vel_vstu_vssl(vrsum45, 4, pGIn+gInIndex+4*gInPixels, vl) ;
	    _vel_vstl_vssl(vrsum45, 4, pGIn+gInIndex+5*gInPixels, vl) ;
	    _vel_vstu_vssl(vrsum67, 4, pGIn+gInIndex+6*gInPixels, vl) ;
	    _vel_vstl_vssl(vrsum67, 4, pGIn+gInIndex+7*gInPixels, vl) ;
	    _vel_vstu_vssl(vrsum89, 4, pGIn+gInIndex+8*gInPixels, vl) ;
	    _vel_vstl_vssl(vrsum89, 4, pGIn+gInIndex+9*gInPixels, vl) ;
	    _vel_vstu_vssl(vrsumAB, 4, pGIn+gInIndex+10*gInPixels, vl) ;
	    _vel_vstl_vssl(vrsumAB, 4, pGIn+gInIndex+11*gInPixels, vl) ;
	    _vel_vstu_vssl(vrsumCD, 4, pGIn+gInIndex+12*gInPixels, vl) ;
	    _vel_vstl_vssl(vrsumCD, 4, pGIn+gInIndex+13*gInPixels, vl) ;
	    _vel_vstu_vssl(vrsumEF, 4, pGIn+gInIndex+14*gInPixels, vl) ;
	    _vel_vstl_vssl(vrsumEF, 4, pGIn+gInIndex+15*gInPixels, vl) ;

	  } // gOutPixels
	} // gInChannel
      } // group
    } // batch
  }


  return VEDNN_SUCCESS;
}
