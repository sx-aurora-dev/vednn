#include <stdint.h>
#include <stdlib.h>

#include "vednn.h"

#include "velintrin.h"
#define VLEN	(256)

template<filterLayout_t FLAYOUT, int NUMCHANNEL>
static inline void func(
    const float * __restrict__ pGOut,
    const float * __restrict__ pKernel,
    float * __restrict__ const pGIn,
    const int64_t gOutChannel,
    const int64_t gOutWidth,
    const int64_t gOutHeight,
    const int64_t gInChannel,
    const int64_t gInWidth,
    const int64_t gInHeight,
    const int64_t strideWidth,
    const int64_t strideHeight,
    const int64_t gInChannelGroup,
    const int64_t gOutChannelGroup,
    const int64_t gInGroupOffset,
    const int64_t gOutGroupOffset,
    const int64_t kernGroupOffset,
    const int64_t n,
    const int64_t c,
    const int64_t nY,
    const __vr    vrij
)
{
  const int64_t remain  = NUMCHANNEL & 0x1 ;
  const int64_t nPacked = NUMCHANNEL >> 1 ;

  const float *pInChannel = pGIn + gInGroupOffset + ((n * gInChannel + c) * gInHeight * gInWidth ) ;
  {
    // zero filling
    const int64_t nPixs = NUMCHANNEL * gInHeight * gInWidth ;
    __vr vrzero = _vel_vbrds_vsl(0.f, VLEN) ;
    for(int64_t ij=0; ij<nPixs ; ij+=VLEN) {
      const int64_t vl = nPixs - ij < VLEN ? nPixs - ij : VLEN ;
      _vel_vstu_vssl(vrzero, 4, (void*)(pInChannel+ij), vl) ;
    }
  }

  for (int64_t y=0; y<gOutHeight; y+=nY)
  {
    const int64_t vl = gOutWidth * (gOutHeight - y < nY ? gOutHeight - y : nY) ;
    const int64_t op = y * gOutWidth ;

    __vr vrsum0  = _vel_vbrds_vsl(0.f, vl) ;
    __vr vrsum[nPacked] ;
#pragma clang loop unroll(full)
    for(int64_t cc=0; cc<nPacked; cc++) {
	vrsum[cc] = _vel_pvbrd_vsl(0UL, vl) ;
    }

    for(int64_t k=0; k<gOutChannelGroup; k++) {
      int64_t outIndex = gOutGroupOffset + (n * gOutChannel + k  ) *  gOutHeight * gOutWidth + op ;

      __vr vrgout = _vel_vldu_vssl(4, pGOut+outIndex, vl) ;

#define FILTER_OFFSET(k,c,r,s) ( kernGroupOffset + filter_index<FLAYOUT>(k,c,r,s, gInChannelGroup, gOutChannelGroup, 1, 1) )

#define VFADD(VRGOUT,K,R,S) {									\
	__vr vrgoutP = _vel_vshf_vvvsl(VRGOUT, VRGOUT, VE_VSHUFFLE_YUZU, vl) ;			\
	if( remain ) {										\
	  const float    kerValue0  = pKernel[FILTER_OFFSET(K,c+ 0,R,S)] ;			\
	  vrsum0 = _vel_vfmads_vvsvl(vrsum0, kerValue0, VRGOUT, vl) ;				\
	}											\
	_Pragma("clang loop unroll(full)")							\
	for(int64_t cc=0; cc<nPacked; cc++) {							\
	  const uint64_t kerValue = _vel_pack_f32p(pKernel + FILTER_OFFSET(K,c+2*cc+remain,  R,S),	\
						   pKernel + FILTER_OFFSET(K,c+2*cc+remain+1,R,S)) ;	\
	  vrsum[cc] = _vel_pvfmad_vvsvl(vrsum[cc], kerValue, vrgoutP, vl) ;			\
	}											\
      }

      VFADD(vrgout, k, 0, 0) ;
#undef VFADD
#undef FILTER_OFFSET
    }

    __vr vrpgin[NUMCHANNEL] ;
    vrpgin[0] = _vel_vsfa_vvssl(vrij, 2, (uint64_t)(pInChannel+y*strideHeight*gInWidth), vl) ;
#pragma clang loop unroll(full)
    for(int64_t cc=1; cc<NUMCHANNEL; cc++) {
      vrpgin[cc] = _vel_vaddul_vsvl(cc*4*gInHeight*gInWidth,vrpgin[0], vl) ;
    }

    _vel_svob() ;

    if( remain ) {
      _vel_vscuot_vvssl(vrsum0, vrpgin[0], 0, 0, vl) ;
    }
#pragma clang loop unroll(full)
    for(int64_t cc=0; cc<nPacked; cc++) {
      _vel_vscuot_vvssl(vrsum[cc], vrpgin[2*cc+remain],   0, 0, vl) ;
      _vel_vsclot_vvssl(vrsum[cc], vrpgin[2*cc+remain+1], 0, 0, vl) ;
    }
  }

  _vel_svob() ;
}


template<int NUMCHANNEL>
static inline void func_even_filternchw_packedkernel(
    const float * __restrict__ pGOut,
    const float * __restrict__ pKernel,
    float * __restrict__ const pGIn,
    const int64_t gOutChannel,
    const int64_t gOutWidth,
    const int64_t gOutHeight,
    const int64_t gInChannel,
    const int64_t gInWidth,
    const int64_t gInHeight,
    const int64_t strideWidth,
    const int64_t strideHeight,
    const int64_t gInChannelGroup,
    const int64_t gOutChannelGroup,
    const int64_t gInGroupOffset,
    const int64_t gOutGroupOffset,
    const int64_t kernGroupOffset,
    const int64_t n,
    const int64_t c,
    const int64_t nY,
    const __vr    vrij
)
{
  const int64_t nPacked = NUMCHANNEL >> 1 ;

  const float *pInChannel = pGIn + gInGroupOffset + ((n * gInChannel + c) * gInHeight * gInWidth ) ;
  {
    // zero filling
    const int64_t nPixs = NUMCHANNEL * gInHeight * gInWidth ;
    __vr vrzero = _vel_vbrds_vsl(0.f, VLEN) ;
    for(int64_t ij=0; ij<nPixs ; ij+=VLEN) {
      const int64_t vl = nPixs - ij < VLEN ? nPixs - ij : VLEN ;
      _vel_vstu_vssl(vrzero, 4, (void*)(pInChannel+ij), vl) ;
    }
  }

  for (int64_t y=0; y<gOutHeight; y+=nY)
  {
    const int64_t vl = gOutWidth * (gOutHeight - y < nY ? gOutHeight - y : nY) ;
    const int64_t op = y * gOutWidth ;

    __vr vrsum[nPacked] ;
#pragma clang loop unroll(full)
    for(int64_t cc=0; cc<nPacked; cc++) {
	vrsum[cc] = _vel_pvbrd_vsl(0UL, vl) ;
    }

    for(int64_t k=0; k<gOutChannelGroup; k++) {
      int64_t outIndex = gOutGroupOffset + (n * gOutChannel + k  ) *  gOutHeight * gOutWidth + op ;

      __vr vrgout = _vel_vldu_vssl(4, pGOut+outIndex, vl) ;

      const float *pKerValue = pKernel + kernGroupOffset + ((k  ) * gInChannelGroup + c) ;
      const uint64_t *pKerValue_u64 = (const uint64_t*) pKerValue ;
#define VFADD(VRGOUT,K,R,S) {								\
	__vr vrgoutP = _vel_vshf_vvvsl(VRGOUT, VRGOUT, VE_VSHUFFLE_YUZU, vl) ;		\
	_Pragma("clang loop unroll(full)")						\
	for(int64_t cc=0; cc<nPacked; cc++) {						\
	  vrsum[cc] = _vel_pvfmad_vvsvl(vrsum[cc], pKerValue_u64[cc], vrgoutP, vl) ;	\
	}										\
      }

      VFADD(vrgout, k, 0, 0) ;
#undef VFADD
    }

    __vr vrpgin[NUMCHANNEL] ;
    vrpgin[0] = _vel_vsfa_vvssl(vrij, 2, (uint64_t)(pInChannel+y*strideHeight*gInWidth), vl) ;
#pragma clang loop unroll(full)
    for(int64_t cc=1; cc<NUMCHANNEL; cc++) {
      vrpgin[cc] = _vel_vaddul_vsvl(cc*4*gInHeight*gInWidth,vrpgin[0], vl) ;
    }

    _vel_svob() ;

#pragma clang loop unroll(full)
    for(int64_t cc=0; cc<nPacked; cc++) {
      _vel_vsclot_vvssl(vrsum[cc], vrpgin[2*cc],   0, 0, vl) ;
      _vel_vscuot_vvssl(vrsum[cc], vrpgin[2*cc+1], 0, 0, vl) ;
    }
  }

  _vel_svob() ;
}

template<filterLayout_t FLAYOUT>
static inline void convloop(
    const float * __restrict__ pGOut,
    const float * __restrict__ pKernel,
    float * __restrict__ const pGIn,
    const int64_t batch,
    const int64_t group,
    const int64_t gOutChannel,
    const int64_t gOutWidth,
    const int64_t gOutHeight,
    const int64_t gInChannel,
    const int64_t gInWidth,
    const int64_t gInHeight,
    const int64_t gInChannelGroup,
    const int64_t gOutChannelGroup,
    const int64_t strideWidth,
    const int64_t strideHeight
)
{

  const int64_t nY = VLEN / gOutWidth ;

  __vr vrseq = _vel_vseq_vl(nY*gOutWidth) ;
  __vr vry  = _vel_vdivsl_vvsl(vrseq, gOutWidth, nY*gOutWidth) ;
  __vr vrx  = _vel_vsubsl_vvvl(vrseq, _vel_vmulul_vsvl(gOutWidth,vry, nY*gOutWidth), nY*gOutWidth) ;

  __vr vri   = _vel_vmulsl_vsvl(strideHeight, vry, nY*gOutWidth) ;
  __vr vrj   = _vel_vmulsl_vsvl(strideWidth,  vrx, nY*gOutWidth) ;
  __vr vrij = _vel_vaddul_vvvl(vrj, _vel_vmulul_vsvl(gInWidth, vri, nY*gOutWidth), nY*gOutWidth) ;

  const int64_t usePackedKernel = (((uint64_t)pKernel) & 0x07) == 0 && (gInChannelGroup & 0x01) == 0 ?  1 : 0  ;

  for (int64_t n=0; n<batch; n++) {
    for (int64_t g = 0; g < group; g++) {

      int64_t gInGroupOffset  = g * gInChannelGroup * gInHeight * gInWidth;
      int64_t gOutGroupOffset = g * gOutChannelGroup * gOutHeight * gOutWidth;
      int64_t kernGroupOffset = g * gOutChannelGroup * gInChannelGroup ;

      const int64_t remain = gInChannelGroup & 0xf ;

      int64_t c=0;
      switch(remain) {
      case 1:
	func<FLAYOUT, 1>(pGOut, pKernel, pGIn,
	   gOutChannel, gOutWidth, gOutHeight,
	   gInChannel, gInWidth, gInHeight,
	   strideWidth, strideHeight,
	   gInChannelGroup, gOutChannelGroup,
	   gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	   n, c, nY, vrij) ;
	c+=1 ;
	break ;
      case 2:
	if ( FLAYOUT ==  VEDNN_FILTER_LAYOUT_NCHW && usePackedKernel ) {
	  func_even_filternchw_packedkernel<2>(pGOut, pKernel, pGIn,
	     gOutChannel, gOutWidth, gOutHeight,
	     gInChannel, gInWidth, gInHeight,
	     strideWidth, strideHeight,
	     gInChannelGroup, gOutChannelGroup,
	     gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	     n, c, nY, vrij) ;
	}
	else {
	  func<FLAYOUT, 2>(pGOut, pKernel, pGIn,
	     gOutChannel, gOutWidth, gOutHeight,
	     gInChannel, gInWidth, gInHeight,
	     strideWidth, strideHeight,
	     gInChannelGroup, gOutChannelGroup,
	     gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	     n, c, nY, vrij) ;
	}
	c+=2 ;
	break ;
      case 3:
	func<FLAYOUT, 3>(pGOut, pKernel, pGIn,
	   gOutChannel, gOutWidth, gOutHeight,
	   gInChannel, gInWidth, gInHeight,
	   strideWidth, strideHeight,
	   gInChannelGroup, gOutChannelGroup,
	   gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	   n, c, nY, vrij) ;
	c+=3 ;
	break ;
      case 4:
	if ( FLAYOUT ==  VEDNN_FILTER_LAYOUT_NCHW && usePackedKernel ) {
	  func_even_filternchw_packedkernel<4>(pGOut, pKernel, pGIn,
	     gOutChannel, gOutWidth, gOutHeight,
	     gInChannel, gInWidth, gInHeight,
	     strideWidth, strideHeight,
	     gInChannelGroup, gOutChannelGroup,
	     gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	     n, c, nY, vrij) ;
	}
	else {
	  func<FLAYOUT, 4>(pGOut, pKernel, pGIn,
	     gOutChannel, gOutWidth, gOutHeight,
	     gInChannel, gInWidth, gInHeight,
	     strideWidth, strideHeight,
	     gInChannelGroup, gOutChannelGroup,
	     gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	     n, c, nY, vrij) ;
	}
	c+=4 ;
	break ;
      case 5:
	func<FLAYOUT, 5>(pGOut, pKernel, pGIn,
	   gOutChannel, gOutWidth, gOutHeight,
	   gInChannel, gInWidth, gInHeight,
	   strideWidth, strideHeight,
	   gInChannelGroup, gOutChannelGroup,
	   gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	   n, c, nY, vrij) ;
	c+=5 ;
	break ;
      case 6:
	if ( FLAYOUT ==  VEDNN_FILTER_LAYOUT_NCHW && usePackedKernel ) {
	  func_even_filternchw_packedkernel<6>(pGOut, pKernel, pGIn,
	     gOutChannel, gOutWidth, gOutHeight,
	     gInChannel, gInWidth, gInHeight,
	     strideWidth, strideHeight,
	     gInChannelGroup, gOutChannelGroup,
	     gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	     n, c, nY, vrij) ;
	}
	else {
	  func<FLAYOUT, 6>(pGOut, pKernel, pGIn,
	     gOutChannel, gOutWidth, gOutHeight,
	     gInChannel, gInWidth, gInHeight,
	     strideWidth, strideHeight,
	     gInChannelGroup, gOutChannelGroup,
	     gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	     n, c, nY, vrij) ;
	}
	c+=6 ;
	break ;
      case 7:
	func<FLAYOUT, 7>(pGOut, pKernel, pGIn,
	   gOutChannel, gOutWidth, gOutHeight,
	   gInChannel, gInWidth, gInHeight,
	   strideWidth, strideHeight,
	   gInChannelGroup, gOutChannelGroup,
	   gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	   n, c, nY, vrij) ;
	c+=7 ;
	break ;
      case 8:
	if ( FLAYOUT ==  VEDNN_FILTER_LAYOUT_NCHW && usePackedKernel ) {
	  func_even_filternchw_packedkernel<8>(pGOut, pKernel, pGIn,
	     gOutChannel, gOutWidth, gOutHeight,
	     gInChannel, gInWidth, gInHeight,
	     strideWidth, strideHeight,
	     gInChannelGroup, gOutChannelGroup,
	     gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	     n, c, nY, vrij) ;
	}
	else {
	  func<FLAYOUT, 8>(pGOut, pKernel, pGIn,
	     gOutChannel, gOutWidth, gOutHeight,
	     gInChannel, gInWidth, gInHeight,
	     strideWidth, strideHeight,
	     gInChannelGroup, gOutChannelGroup,
	     gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	     n, c, nY, vrij) ;
	}
	c+=8 ;
	break ;
      case 9:
	func<FLAYOUT, 9>(pGOut, pKernel, pGIn,
	   gOutChannel, gOutWidth, gOutHeight,
	   gInChannel, gInWidth, gInHeight,
	   strideWidth, strideHeight,
	   gInChannelGroup, gOutChannelGroup,
	   gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	   n, c, nY, vrij) ;
	c+=9 ;
	break ;
      case 10:
	if ( FLAYOUT ==  VEDNN_FILTER_LAYOUT_NCHW && usePackedKernel ) {
	  func_even_filternchw_packedkernel<10>(pGOut, pKernel, pGIn,
	     gOutChannel, gOutWidth, gOutHeight,
	     gInChannel, gInWidth, gInHeight,
	     strideWidth, strideHeight,
	     gInChannelGroup, gOutChannelGroup,
	     gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	     n, c, nY, vrij) ;
	}
	else {
	  func<FLAYOUT, 10>(pGOut, pKernel, pGIn,
	     gOutChannel, gOutWidth, gOutHeight,
	     gInChannel, gInWidth, gInHeight,
	     strideWidth, strideHeight,
	     gInChannelGroup, gOutChannelGroup,
	     gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	     n, c, nY, vrij) ;
	}
	c+=10 ;
	break ;
      case 11:
	func<FLAYOUT, 11>(pGOut, pKernel, pGIn,
	   gOutChannel, gOutWidth, gOutHeight,
	   gInChannel, gInWidth, gInHeight,
	   strideWidth, strideHeight,
	   gInChannelGroup, gOutChannelGroup,
	   gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	   n, c, nY, vrij) ;
	c+=11 ;
	break ;
      case 12:
	if ( FLAYOUT ==  VEDNN_FILTER_LAYOUT_NCHW && usePackedKernel ) {
	  func_even_filternchw_packedkernel<12>(pGOut, pKernel, pGIn,
	     gOutChannel, gOutWidth, gOutHeight,
	     gInChannel, gInWidth, gInHeight,
	     strideWidth, strideHeight,
	     gInChannelGroup, gOutChannelGroup,
	     gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	     n, c, nY, vrij) ;
	}
	else {
	  func<FLAYOUT, 12>(pGOut, pKernel, pGIn,
	     gOutChannel, gOutWidth, gOutHeight,
	     gInChannel, gInWidth, gInHeight,
	     strideWidth, strideHeight,
	     gInChannelGroup, gOutChannelGroup,
	     gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	     n, c, nY, vrij) ;
	}
	c+=12 ;
	break ;
      case 13:
	func<FLAYOUT, 13>(pGOut, pKernel, pGIn,
	   gOutChannel, gOutWidth, gOutHeight,
	   gInChannel, gInWidth, gInHeight,
	   strideWidth, strideHeight,
	   gInChannelGroup, gOutChannelGroup,
	   gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	   n, c, nY, vrij) ;
	c+=13 ;
	break ;
      case 14:
	if ( FLAYOUT ==  VEDNN_FILTER_LAYOUT_NCHW && usePackedKernel ) {
	  func_even_filternchw_packedkernel<14>(pGOut, pKernel, pGIn,
	     gOutChannel, gOutWidth, gOutHeight,
	     gInChannel, gInWidth, gInHeight,
	     strideWidth, strideHeight,
	     gInChannelGroup, gOutChannelGroup,
	     gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	     n, c, nY, vrij) ;
	}
	else {
	  func<FLAYOUT, 14>(pGOut, pKernel, pGIn,
	     gOutChannel, gOutWidth, gOutHeight,
	     gInChannel, gInWidth, gInHeight,
	     strideWidth, strideHeight,
	     gInChannelGroup, gOutChannelGroup,
	     gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	     n, c, nY, vrij) ;
	}
	c+=14 ;
	break ;
      case 15:
	func<FLAYOUT, 15>(pGOut, pKernel, pGIn,
	   gOutChannel, gOutWidth, gOutHeight,
	   gInChannel, gInWidth, gInHeight,
	   strideWidth, strideHeight,
	   gInChannelGroup, gOutChannelGroup,
	   gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	   n, c, nY, vrij) ;
	c+=15 ;
	break ;
      default :
	break ;
      }
      for (; c<gInChannelGroup; ) {
	if ( FLAYOUT ==  VEDNN_FILTER_LAYOUT_NCHW && usePackedKernel ) {
	  func_even_filternchw_packedkernel<16>(pGOut, pKernel, pGIn,
	     gOutChannel, gOutWidth, gOutHeight,
	     gInChannel, gInWidth, gInHeight,
	     strideWidth, strideHeight,
	     gInChannelGroup, gOutChannelGroup,
	     gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	     n, c, nY, vrij) ;
	}
	else {
	  func<FLAYOUT, 16>(pGOut, pKernel, pGIn,
	     gOutChannel, gOutWidth, gOutHeight,
	     gInChannel, gInWidth, gInHeight,
	     strideWidth, strideHeight,
	     gInChannelGroup, gOutChannelGroup,
	     gInGroupOffset, gOutGroupOffset, kernGroupOffset,
	     n, c, nY, vrij) ;
	}
	c+= 16 ;
      } // gInChannel
    } // group
  } // batch
}

extern "C"
vednnError_t
vednnConvolutionBackwardData_direct_dil1_pad0_ker1_owU128(
    const vednnTensorParam_t * 		pParamGradOut,
    const void *			pDataGradOut,
    const vednnFilterParam_t *	 	pParamKernel,
    const void * 			pDataKernel,
    const vednnConvolutionParam_t * 	pParamConv,
    const vednnTensorParam_t * 		pParamGradIn,
    void * 				pDataGradIn
)
{
  const int64_t batch       = pParamGradOut->batch;
  const int64_t gOutChannel = pParamGradOut->channel;
  const int64_t gOutWidth   = pParamGradOut->width;
  const int64_t gOutHeight  = pParamGradOut->height;
  const int64_t gInChannel  = pParamGradIn->channel;
  const int64_t gInWidth    = pParamGradIn->width;
  const int64_t gInHeight   = pParamGradIn->height;
//  const int64_t kernWidth   = pParamKernel->width;	// 1
//  const int64_t kernHeight  = pParamKernel->height;	// 1

  const int64_t filter_layout = pParamKernel->layout ;

  const int64_t group          = pParamConv->group;
  const int64_t strideWidth    = pParamConv->strideWidth;
  const int64_t strideHeight   = pParamConv->strideHeight;
//  const int64_t padWidth       = pParamConv->padWidth;	// 0
//  const int64_t padHeight      = pParamConv->padHeight;	// 0
//  const int64_t dilationWidth  = pParamConv->dilationWidth;	// 1
//  const int64_t dilationHeight = pParamConv->dilationHeight;	// 1

  const int64_t gOutChannelGroup = gOutChannel  / group;
  const int64_t gInChannelGroup  = gInChannel / group;

  const float *  pGOut   = (const float *) pDataGradOut;
  const float *  pKernel = (const float *) pDataKernel;
  float *  const pGIn    = (float * const) pDataGradIn;


  if( filter_layout == VEDNN_FILTER_LAYOUT_NCHW) {
    convloop<VEDNN_FILTER_LAYOUT_NCHW>(pGOut, pKernel, pGIn,
	       batch, group,
	       gOutChannel, gOutWidth, gOutHeight,
	       gInChannel, gInWidth, gInHeight,
	       gInChannelGroup, gOutChannelGroup,
	       strideWidth, strideHeight ) ;
  }
  else {
    convloop<VEDNN_FILTER_LAYOUT_HWCN>(pGOut, pKernel, pGIn,
	       batch, group,
	       gOutChannel, gOutWidth, gOutHeight,
	       gInChannel, gInWidth, gInHeight,
	       gInChannelGroup, gOutChannelGroup,
	       strideWidth, strideHeight ) ;
  }

  return VEDNN_SUCCESS;
}
