#include <stdint.h>
#include <stdlib.h>

#include "vednn.h"

#include "velintrin.h"
#define VLEN	(256)


template<filterLayout_t FLAYOUT, bool ADDBIAS, int NUMKERNEL, int X>
static inline void func(
    const float * __restrict__ pIn,
    const float * __restrict__ pKernel,
    const float * __restrict__ pBias,
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
    const int64_t strideHeight,
    const int64_t strideWidth,
    const int64_t padHeight,
    const int64_t padWidth,
    const int64_t dilationHeight,
    const int64_t dilationWidth,
    const int64_t inGroupOffset,
    const int64_t outGroupOffset,
    const int64_t biasGroupOffset,
    const int64_t kernGroupOffset,
    const int64_t n,
    const int64_t k,
    const int64_t y,
    const int64_t x
)
{

  int64_t outIndex  = outGroupOffset + ((n * outChannel + k) * outHeight + y) * outWidth + x;

  const int64_t remain  = NUMKERNEL & 0x1 ;
  const int64_t nPacked = NUMKERNEL >> 1 ;

  float bias[NUMKERNEL] ;
#pragma clang loop unroll(full)
  for(int64_t kk=0; kk<NUMKERNEL; kk++) {
    bias[kk] = pBias[biasGroupOffset+k+kk] ;
  }

  __vr vrsum0[X] ;
#pragma clang loop unroll(full)
  for(int64_t xx=0; xx<X; xx++) {
    vrsum0[xx] = _vel_vbrds_vsl(0.f, VLEN) ;
  }

  __vr vrsum[nPacked*X] ;
#pragma clang loop unroll(full)
  for(int64_t kk=0; kk<nPacked; kk++) {
#pragma clang loop unroll(full)
    for(int64_t xx=0; xx<X; xx++) {
      vrsum[kk*X+xx] = _vel_vbrdl_vsl(0UL, VLEN) ;
    }
  }

  for (int64_t c=0; c<inChannelGroup; c+= VLEN) {
    const int64_t vl = inChannelGroup - c < VLEN ? inChannelGroup - c : VLEN ;

    for (int64_t r=0; r<kernHeight; r++) {
      int64_t h = y - 1 + r ;
      if( h < 0 || h >= inHeight ) continue ;

      {
	const float *pKerValue = ( FLAYOUT == VEDNN_FILTER_LAYOUT_NCHW ) ?
				  pKernel + kernGroupOffset + ((k * inChannelGroup + c) * kernHeight + r) * kernWidth :
				  pKernel + kernGroupOffset + ( ( r * kernWidth ) * inChannelGroup + c ) * outChannelGroup + k ;

	const int64_t kernelDistanceByK = ( FLAYOUT == VEDNN_FILTER_LAYOUT_NCHW ) ?
					    inChannelGroup * 3 * 3 :
					    1 ;

	const int64_t kernelDistanceByC = ( FLAYOUT == VEDNN_FILTER_LAYOUT_NCHW ) ?
					    3 * 3 :
					    outChannelGroup ;

	const int64_t kernelDistanceByS = ( FLAYOUT == VEDNN_FILTER_LAYOUT_NCHW ) ?
					    1 :
					    inChannelGroup * outChannelGroup ;


	int64_t wl = (x+0)   - 1 + 0 ;
	int64_t wr = (x+X-1) - 1 + 2 ;

	int64_t wl_valid = ( wl >= 0 && wl < inWidth ) ;
	int64_t wr_valid = ( wr >= 0 && wr < inWidth ) ;

	if( wl_valid && wr_valid ) {

	  __vr vri[X+2] ;
#pragma clang loop unroll(full)
	  for(int64_t xx=0; xx<X+2; xx++) {
	    int64_t inputIndex  = inGroupOffset + ((n * inChannel + c) * inHeight + h) * inWidth + wl + xx;
	    vri[xx] = _vel_vldu_vssl(4*inHeight*inWidth, &pIn[inputIndex], vl) ;
	  }

	  __vr vriP[X+2] ;
#pragma clang loop unroll(full)
	  for(int64_t xx=0; xx<X+2; xx++) {
	    vriP[xx] = _vel_vshf_vvvsl(vri[xx], vri[xx], VE_VSHUFFLE_YUZU, vl) ;
	  }

	  /* s0 */
	  {
	    __vr vrk_s0[NUMKERNEL] ;
#pragma clang loop unroll(full)
	    for(int64_t kk=0; kk<NUMKERNEL; kk++) {
	      vrk_s0[kk] = _vel_vldu_vssl(4*kernelDistanceByC, pKerValue+kk*kernelDistanceByK+0*kernelDistanceByS, vl) ;
	    }

	    if( remain ) {
#pragma clang loop unroll(full)
	      for(int64_t xx=0; xx<X; xx++) {
	        vrsum0[xx] = _vel_vfmads_vvvvvl(vrsum0[xx], vri[xx+0], vrk_s0[0], vrsum0[xx], vl) ;
	      }
	    }
#pragma clang loop unroll(full)
	    for(int64_t kk=0; kk<nPacked; kk++) {
	      __vr vrkp_s0 = _vel_vshf_vvvsl(vrk_s0[2*kk+remain], vrk_s0[2*kk+remain+1], VE_VSHUFFLE_YUZU, vl) ;

#pragma clang loop unroll(full)
	      for(int64_t xx=0; xx<X; xx++) {
		vrsum[kk*X+xx] = _vel_pvfmad_vvvvvl(vrsum[kk*X+xx], vriP[xx+0], vrkp_s0, vrsum[kk*X+xx], vl) ;
	      }
	    }
	  }

	  /* s1 */
	  {
	    __vr vrk_s1[NUMKERNEL] ;
#pragma clang loop unroll(full)
	    for(int64_t kk=0; kk<NUMKERNEL; kk++) {
	      vrk_s1[kk] = _vel_vldu_vssl(4*kernelDistanceByC, pKerValue+kk*kernelDistanceByK+1*kernelDistanceByS, vl) ;
	    }

	    if( remain ) {
#pragma clang loop unroll(full)
	      for(int64_t xx=0; xx<X; xx++) {
		vrsum0[xx] = _vel_vfmads_vvvvvl(vrsum0[xx], vri[xx+1], vrk_s1[0], vrsum0[xx], vl) ;
	      }
	    }

#pragma clang loop unroll(full)
	    for(int64_t kk=0; kk<nPacked; kk++) {
	      __vr vrkp_s1 = _vel_vshf_vvvsl(vrk_s1[2*kk+remain], vrk_s1[2*kk+remain+1], VE_VSHUFFLE_YUZU, vl) ;
#pragma clang loop unroll(full)
	      for(int64_t xx=0; xx<X; xx++) {
		vrsum[kk*X+xx] = _vel_pvfmad_vvvvvl(vrsum[kk*X+xx], vriP[xx+1], vrkp_s1, vrsum[kk*X+xx], vl) ;
	      }
	    }
	  }

	  /* s2 */
	  {
	    __vr vrk_s2[NUMKERNEL] ;
#pragma clang loop unroll(full)
	    for(int64_t kk=0; kk<NUMKERNEL; kk++) {
	      vrk_s2[kk] = _vel_vldu_vssl(4*kernelDistanceByC, pKerValue+kk*kernelDistanceByK+2*kernelDistanceByS, vl) ;
	    }

	    if( remain ) {
#pragma clang loop unroll(full)
	      for(int64_t xx=0; xx<X; xx++) {
		vrsum0[xx] = _vel_vfmads_vvvvvl(vrsum0[xx], vri[xx+2], vrk_s2[0], vrsum0[xx], vl) ;
	      }
	    }

#pragma clang loop unroll(full)
	    for(int64_t kk=0; kk<nPacked; kk++) {
	      __vr vrkp_s2 = _vel_vshf_vvvsl(vrk_s2[2*kk+remain], vrk_s2[2*kk+remain+1], VE_VSHUFFLE_YUZU, vl) ;
#pragma clang loop unroll(full)
	      for(int64_t xx=0; xx<X; xx++) {
		vrsum[kk*X+xx] = _vel_pvfmad_vvvvvl(vrsum[kk*X+xx], vriP[xx+2], vrkp_s2, vrsum[kk*X+xx], vl) ;
	      }
	    }
	  }
	}
	else if( wl_valid ) {

	  __vr vri[X+2] ;
#pragma clang loop unroll(full)
	  for(int64_t xx=0; xx<X+1; xx++) {
	    int64_t inputIndex  = inGroupOffset + ((n * inChannel + c) * inHeight + h) * inWidth + wl + xx;
	    vri[xx] = _vel_vldu_vssl(4*inHeight*inWidth, &pIn[inputIndex], vl) ;
	  }

	  __vr vriP[X+2] ;
#pragma clang loop unroll(full)
	  for(int64_t xx=0; xx<X+1; xx++) {
	    vriP[xx] = _vel_vshf_vvvsl(vri[xx], vri[xx], VE_VSHUFFLE_YUZU, vl) ;
	  }

	  /* s0 */
	  {
	    __vr vrk_s0[NUMKERNEL] ;
#pragma clang loop unroll(full)
	    for(int64_t kk=0; kk<NUMKERNEL; kk++) {
	      vrk_s0[kk] = _vel_vldu_vssl(4*kernelDistanceByC, pKerValue+kk*kernelDistanceByK+0*kernelDistanceByS, vl) ;
	    }

	    if( remain ) {
#pragma clang loop unroll(full)
	      for(int64_t xx=0; xx<X; xx++) {
	        vrsum0[xx] = _vel_vfmads_vvvvvl(vrsum0[xx], vri[xx+0], vrk_s0[0], vrsum0[xx], vl) ;
	      }
	    }
#pragma clang loop unroll(full)
	    for(int64_t kk=0; kk<nPacked; kk++) {
	      __vr vrkp_s0 = _vel_vshf_vvvsl(vrk_s0[2*kk+remain], vrk_s0[2*kk+remain+1], VE_VSHUFFLE_YUZU, vl) ;

#pragma clang loop unroll(full)
	      for(int64_t xx=0; xx<X; xx++) {
		vrsum[kk*X+xx] = _vel_pvfmad_vvvvvl(vrsum[kk*X+xx], vriP[xx+0], vrkp_s0, vrsum[kk*X+xx], vl) ;
	      }
	    }
	  }

	  /* s1 */
	  {
	    __vr vrk_s1[NUMKERNEL] ;
#pragma clang loop unroll(full)
	    for(int64_t kk=0; kk<NUMKERNEL; kk++) {
	      vrk_s1[kk] = _vel_vldu_vssl(4*kernelDistanceByC, pKerValue+kk*kernelDistanceByK+1*kernelDistanceByS, vl) ;
	    }

	    if( remain ) {
#pragma clang loop unroll(full)
	      for(int64_t xx=0; xx<X; xx++) {
		vrsum0[xx] = _vel_vfmads_vvvvvl(vrsum0[xx], vri[xx+1], vrk_s1[0], vrsum0[xx], vl) ;
	      }
	    }

#pragma clang loop unroll(full)
	    for(int64_t kk=0; kk<nPacked; kk++) {
	      __vr vrkp_s1 = _vel_vshf_vvvsl(vrk_s1[2*kk+remain], vrk_s1[2*kk+remain+1], VE_VSHUFFLE_YUZU, vl) ;
#pragma clang loop unroll(full)
	      for(int64_t xx=0; xx<X; xx++) {
		vrsum[kk*X+xx] = _vel_pvfmad_vvvvvl(vrsum[kk*X+xx], vriP[xx+1], vrkp_s1, vrsum[kk*X+xx], vl) ;
	      }
	    }
	  }

	  /* s2 */
	  {
	    __vr vrk_s2[NUMKERNEL] ;
#pragma clang loop unroll(full)
	    for(int64_t kk=0; kk<NUMKERNEL; kk++) {
	      vrk_s2[kk] = _vel_vldu_vssl(4*kernelDistanceByC, pKerValue+kk*kernelDistanceByK+2*kernelDistanceByS, vl) ;
	    }

	    if( remain ) {
#pragma clang loop unroll(full)
	      for(int64_t xx=0; xx<X-1; xx++) {
		vrsum0[xx] = _vel_vfmads_vvvvvl(vrsum0[xx], vri[xx+2], vrk_s2[0], vrsum0[xx], vl) ;
	      }
	    }

#pragma clang loop unroll(full)
	    for(int64_t kk=0; kk<nPacked; kk++) {
	      __vr vrkp_s2 = _vel_vshf_vvvsl(vrk_s2[2*kk+remain], vrk_s2[2*kk+remain+1], VE_VSHUFFLE_YUZU, vl) ;
#pragma clang loop unroll(full)
	      for(int64_t xx=0; xx<X-1; xx++) {
		vrsum[kk*X+xx] = _vel_pvfmad_vvvvvl(vrsum[kk*X+xx], vriP[xx+2], vrkp_s2, vrsum[kk*X+xx], vl) ;
	      }
	    }
	  }
	}
	else if( wr_valid ) {

	  __vr vri[X+2] ;
#pragma clang loop unroll(full)
	  for(int64_t xx=1; xx<X+2; xx++) {
	    int64_t inputIndex  = inGroupOffset + ((n * inChannel + c) * inHeight + h) * inWidth + wl + xx;
	    vri[xx] = _vel_vldu_vssl(4*inHeight*inWidth, &pIn[inputIndex], vl) ;
	  }

	  __vr vriP[X+2] ;
#pragma clang loop unroll(full)
	  for(int64_t xx=1; xx<X+2; xx++) {
	    vriP[xx] = _vel_vshf_vvvsl(vri[xx], vri[xx], VE_VSHUFFLE_YUZU, vl) ;
	  }

	  /* s0 */
	  {
	    __vr vrk_s0[NUMKERNEL] ;
#pragma clang loop unroll(full)
	    for(int64_t kk=0; kk<NUMKERNEL; kk++) {
	      vrk_s0[kk] = _vel_vldu_vssl(4*kernelDistanceByC, pKerValue+kk*kernelDistanceByK+0*kernelDistanceByS, vl) ;
	    }

	    if( remain ) {
#pragma clang loop unroll(full)
	      for(int64_t xx=1; xx<X; xx++) {
	        vrsum0[xx] = _vel_vfmads_vvvvvl(vrsum0[xx], vri[xx+0], vrk_s0[0], vrsum0[xx], vl) ;
	      }
	    }
#pragma clang loop unroll(full)
	    for(int64_t kk=0; kk<nPacked; kk++) {
	      __vr vrkp_s0 = _vel_vshf_vvvsl(vrk_s0[2*kk+remain], vrk_s0[2*kk+remain+1], VE_VSHUFFLE_YUZU, vl) ;

#pragma clang loop unroll(full)
	      for(int64_t xx=1; xx<X; xx++) {
		vrsum[kk*X+xx] = _vel_pvfmad_vvvvvl(vrsum[kk*X+xx], vriP[xx+0], vrkp_s0, vrsum[kk*X+xx], vl) ;
	      }
	    }
	  }

	  /* s1 */
	  {
	    __vr vrk_s1[NUMKERNEL] ;
#pragma clang loop unroll(full)
	    for(int64_t kk=0; kk<NUMKERNEL; kk++) {
	      vrk_s1[kk] = _vel_vldu_vssl(4*kernelDistanceByC, pKerValue+kk*kernelDistanceByK+1*kernelDistanceByS, vl) ;
	    }

	    if( remain ) {
#pragma clang loop unroll(full)
	      for(int64_t xx=0; xx<X; xx++) {
		vrsum0[xx] = _vel_vfmads_vvvvvl(vrsum0[xx], vri[xx+1], vrk_s1[0], vrsum0[xx], vl) ;
	      }
	    }

#pragma clang loop unroll(full)
	    for(int64_t kk=0; kk<nPacked; kk++) {
	      __vr vrkp_s1 = _vel_vshf_vvvsl(vrk_s1[2*kk+remain], vrk_s1[2*kk+remain+1], VE_VSHUFFLE_YUZU, vl) ;
#pragma clang loop unroll(full)
	      for(int64_t xx=0; xx<X; xx++) {
		vrsum[kk*X+xx] = _vel_pvfmad_vvvvvl(vrsum[kk*X+xx], vriP[xx+1], vrkp_s1, vrsum[kk*X+xx], vl) ;
	      }
	    }
	  }

	  /* s2 */
	  {
	    __vr vrk_s2[NUMKERNEL] ;
#pragma clang loop unroll(full)
	    for(int64_t kk=0; kk<NUMKERNEL; kk++) {
	      vrk_s2[kk] = _vel_vldu_vssl(4*kernelDistanceByC, pKerValue+kk*kernelDistanceByK+2*kernelDistanceByS, vl) ;
	    }

	    if( remain ) {
#pragma clang loop unroll(full)
	      for(int64_t xx=0; xx<X; xx++) {
		vrsum0[xx] = _vel_vfmads_vvvvvl(vrsum0[xx], vri[xx+2], vrk_s2[0], vrsum0[xx], vl) ;
	      }
	    }

#pragma clang loop unroll(full)
	    for(int64_t kk=0; kk<nPacked; kk++) {
	      __vr vrkp_s2 = _vel_vshf_vvvsl(vrk_s2[2*kk+remain], vrk_s2[2*kk+remain+1], VE_VSHUFFLE_YUZU, vl) ;
#pragma clang loop unroll(full)
	      for(int64_t xx=0; xx<X; xx++) {
		vrsum[kk*X+xx] = _vel_pvfmad_vvvvvl(vrsum[kk*X+xx], vriP[xx+2], vrkp_s2, vrsum[kk*X+xx], vl) ;
	      }
	    }
	  }
	}
      } // kernWidth
    } // kernHeight
  } // inChannel

  if( remain ) {
#pragma clang loop unroll(full)
    for(int64_t xx=0; xx<X; xx++) {
      vrsum0[xx] = _vel_vfsums_vvl(vrsum0[xx], VLEN) ;
      if(ADDBIAS) vrsum0[xx] = _vel_vfadds_vsvl(bias[0], vrsum0[xx], 1) ;
      _vel_vstu_vssl(vrsum0[xx], 4, &pOut[outIndex+0*outHeight*outWidth+xx], 1) ;
    }
  }

#pragma clang loop unroll(full)
  for(int64_t kk=0; kk<nPacked; kk++) {
#pragma clang loop unroll(full)
    for(int64_t xx=0; xx<X; xx++) {
      __vr vrsumU = _vel_vfsums_vvl(vrsum[kk*X+xx], VLEN) ;
      if(ADDBIAS) vrsumU = _vel_vfadds_vsvl(bias[2*kk+remain], vrsumU, 1) ;
      _vel_vstu_vssl(vrsumU, 4, &pOut[outIndex+(2*kk+remain)*outHeight*outWidth+xx], 1) ;
      __vr vrsumL = _vel_vfsums_vvl(_vel_vsll_vvsl(vrsum[kk*X+xx],32, VLEN), VLEN) ;
      if(ADDBIAS) vrsumL = _vel_vfadds_vsvl(bias[2*kk+remain+1], vrsumL, 1) ;
      _vel_vstu_vssl(vrsumL, 4, &pOut[outIndex+(2*kk+remain+1)*outHeight*outWidth+xx], 1) ;
    }
  }
}

template<filterLayout_t FLAYOUT,int NUMKERNEL, bool ADDBIAS>
static inline void convloopXY(
    const float * __restrict__ pIn,
    const float * __restrict__ pKernel,
    const float * __restrict__ pBias,
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
    const int64_t strideHeight,
    const int64_t strideWidth,
    const int64_t padHeight,
    const int64_t padWidth,
    const int64_t dilationHeight,
    const int64_t dilationWidth,
    const int64_t inGroupOffset,
    const int64_t outGroupOffset,
    const int64_t biasGroupOffset,
    const int64_t kernGroupOffset,
    const int64_t n,
    const int64_t k
)
{
  int64_t xremain = outWidth % 4 ;

  int64_t x=0;
  switch( xremain ) {
  case 1 :
    for(int64_t y=0; y<outHeight; y++) {
      func<FLAYOUT,ADDBIAS,NUMKERNEL,1>(pIn, pKernel, pBias, pOut,
	 inChannel, inWidth, inHeight,
	 outChannel, outWidth, outHeight,
	 kernWidth, kernHeight,
	 inChannelGroup, outChannelGroup,
	 strideHeight, strideWidth,
	 padHeight, padWidth,
	 dilationHeight, dilationWidth,
	 inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	 n, k, y, x) ;
    }
    x+=1 ;
    break ;
  case 2 :
    for(int64_t y=0; y<outHeight; y++) {
      func<FLAYOUT,ADDBIAS,NUMKERNEL,2>(pIn, pKernel, pBias, pOut,
	 inChannel, inWidth, inHeight,
	 outChannel, outWidth, outHeight,
	 kernWidth, kernHeight,
	 inChannelGroup, outChannelGroup,
	 strideHeight, strideWidth,
	 padHeight, padWidth,
	 dilationHeight, dilationWidth,
	 inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	 n, k, y, x) ;
    }
    x+=2 ;
    break ;
  case 3 :
    for(int64_t y=0; y<outHeight; y++) {
      func<FLAYOUT,ADDBIAS,NUMKERNEL,3>(pIn, pKernel, pBias, pOut,
	 inChannel, inWidth, inHeight,
	 outChannel, outWidth, outHeight,
	 kernWidth, kernHeight,
	 inChannelGroup, outChannelGroup,
	 strideHeight, strideWidth,
	 padHeight, padWidth,
	 dilationHeight, dilationWidth,
	 inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	 n, k, y, x) ;
    }
    x+=3 ;
    break ;
  default :
    break ;
  }
  for (; x<outWidth; ) {
    for(int64_t y=0; y<outHeight; y++) {
      func<FLAYOUT,ADDBIAS,NUMKERNEL,4>(pIn, pKernel, pBias, pOut,
	 inChannel, inWidth, inHeight,
	 outChannel, outWidth, outHeight,
	 kernWidth, kernHeight,
	 inChannelGroup, outChannelGroup,
	 strideHeight, strideWidth,
	 padHeight, padWidth,
	 dilationHeight, dilationWidth,
	 inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	 n, k, y, x) ;
    }
    x+=4 ;
  } // outWidth
}

template<filterLayout_t FLAYOUT, bool ADDBIAS>
static inline void convloop(
    const float * __restrict__ pIn,
    const float * __restrict__ pKernel,
    const float * __restrict__ pBias,
    float * __restrict__ const pOut,
    const int64_t batch,
    const int64_t group,
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
    const int64_t strideHeight,
    const int64_t strideWidth,
    const int64_t padHeight,
    const int64_t padWidth,
    const int64_t dilationHeight,
    const int64_t dilationWidth
)
{
  for (int64_t n = 0; n < batch; n++) {
    for (int64_t g = 0; g < group; g++) {
      const int64_t inGroupOffset   = g * inChannelGroup * inHeight * inWidth;
      const int64_t outGroupOffset  = g * outChannelGroup * outHeight * outWidth;
      const int64_t biasGroupOffset = g * outChannelGroup;
      const int64_t kernGroupOffset = g * outChannelGroup * inChannelGroup * kernHeight * kernWidth;

      const int64_t remain = outChannelGroup & 0xf ;

      int k = 0 ;
      switch( remain ) {
      case 1 :
	convloopXY<FLAYOUT, 1, ADDBIAS>(pIn, pKernel, pBias, pOut,
	   inChannel, inWidth, inHeight,
	   outChannel, outWidth, outHeight,
	   kernWidth, kernHeight,
	   inChannelGroup, outChannelGroup,
	   strideHeight, strideWidth,
	   padHeight, padWidth,
	   dilationHeight, dilationWidth,
	   inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	   n, k );
	k+=1 ;
	break ;
      case 2 :
	convloopXY<FLAYOUT, 2, ADDBIAS>(pIn, pKernel, pBias, pOut,
	   inChannel, inWidth, inHeight,
	   outChannel, outWidth, outHeight,
	   kernWidth, kernHeight,
	   inChannelGroup, outChannelGroup,
	   strideHeight, strideWidth,
	   padHeight, padWidth,
	   dilationHeight, dilationWidth,
	   inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	   n, k );
	k+=2 ;
	break ;
      case 3 :
	convloopXY<FLAYOUT, 3, ADDBIAS>(pIn, pKernel, pBias, pOut,
	   inChannel, inWidth, inHeight,
	   outChannel, outWidth, outHeight,
	   kernWidth, kernHeight,
	   inChannelGroup, outChannelGroup,
	   strideHeight, strideWidth,
	   padHeight, padWidth,
	   dilationHeight, dilationWidth,
	   inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	   n, k );
	k+=3 ;
	break ;
      case 4 :
	convloopXY<FLAYOUT, 4, ADDBIAS>(pIn, pKernel, pBias, pOut,
	   inChannel, inWidth, inHeight,
	   outChannel, outWidth, outHeight,
	   kernWidth, kernHeight,
	   inChannelGroup, outChannelGroup,
	   strideHeight, strideWidth,
	   padHeight, padWidth,
	   dilationHeight, dilationWidth,
	   inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	   n, k );
	k+=4 ;
	break ;
      case 5 :
	convloopXY<FLAYOUT, 5, ADDBIAS>(pIn, pKernel, pBias, pOut,
	   inChannel, inWidth, inHeight,
	   outChannel, outWidth, outHeight,
	   kernWidth, kernHeight,
	   inChannelGroup, outChannelGroup,
	   strideHeight, strideWidth,
	   padHeight, padWidth,
	   dilationHeight, dilationWidth,
	   inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	   n, k );
	k+=5 ;
	break ;
      case 6 :
	convloopXY<FLAYOUT, 6, ADDBIAS>(pIn, pKernel, pBias, pOut,
	   inChannel, inWidth, inHeight,
	   outChannel, outWidth, outHeight,
	   kernWidth, kernHeight,
	   inChannelGroup, outChannelGroup,
	   strideHeight, strideWidth,
	   padHeight, padWidth,
	   dilationHeight, dilationWidth,
	   inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	   n, k );
	k+=6 ;
	break ;
      case 7 :
	convloopXY<FLAYOUT, 7, ADDBIAS>(pIn, pKernel, pBias, pOut,
	   inChannel, inWidth, inHeight,
	   outChannel, outWidth, outHeight,
	   kernWidth, kernHeight,
	   inChannelGroup, outChannelGroup,
	   strideHeight, strideWidth,
	   padHeight, padWidth,
	   dilationHeight, dilationWidth,
	   inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	   n, k );
	k+=7 ;
	break ;
      case 8 :
	convloopXY<FLAYOUT, 8, ADDBIAS>(pIn, pKernel, pBias, pOut,
	   inChannel, inWidth, inHeight,
	   outChannel, outWidth, outHeight,
	   kernWidth, kernHeight,
	   inChannelGroup, outChannelGroup,
	   strideHeight, strideWidth,
	   padHeight, padWidth,
	   dilationHeight, dilationWidth,
	   inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	   n, k );
	k+=8 ;
	break ;
      case 9 :
	convloopXY<FLAYOUT, 9, ADDBIAS>(pIn, pKernel, pBias, pOut,
	   inChannel, inWidth, inHeight,
	   outChannel, outWidth, outHeight,
	   kernWidth, kernHeight,
	   inChannelGroup, outChannelGroup,
	   strideHeight, strideWidth,
	   padHeight, padWidth,
	   dilationHeight, dilationWidth,
	   inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	   n, k );
	k+=9 ;
	break ;
      case 10 :
	convloopXY<FLAYOUT, 10, ADDBIAS>(pIn, pKernel, pBias, pOut,
	   inChannel, inWidth, inHeight,
	   outChannel, outWidth, outHeight,
	   kernWidth, kernHeight,
	   inChannelGroup, outChannelGroup,
	   strideHeight, strideWidth,
	   padHeight, padWidth,
	   dilationHeight, dilationWidth,
	   inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	   n, k );
	k+=10 ;
	break ;
      case 11 :
	convloopXY<FLAYOUT, 11, ADDBIAS>(pIn, pKernel, pBias, pOut,
	   inChannel, inWidth, inHeight,
	   outChannel, outWidth, outHeight,
	   kernWidth, kernHeight,
	   inChannelGroup, outChannelGroup,
	   strideHeight, strideWidth,
	   padHeight, padWidth,
	   dilationHeight, dilationWidth,
	   inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	   n, k );
	k+=11 ;
	break ;
      case 12 :
	convloopXY<FLAYOUT, 12, ADDBIAS>(pIn, pKernel, pBias, pOut,
	   inChannel, inWidth, inHeight,
	   outChannel, outWidth, outHeight,
	   kernWidth, kernHeight,
	   inChannelGroup, outChannelGroup,
	   strideHeight, strideWidth,
	   padHeight, padWidth,
	   dilationHeight, dilationWidth,
	   inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	   n, k );
	k+=12 ;
	break ;
      case 13 :
	convloopXY<FLAYOUT, 13, ADDBIAS>(pIn, pKernel, pBias, pOut,
	   inChannel, inWidth, inHeight,
	   outChannel, outWidth, outHeight,
	   kernWidth, kernHeight,
	   inChannelGroup, outChannelGroup,
	   strideHeight, strideWidth,
	   padHeight, padWidth,
	   dilationHeight, dilationWidth,
	   inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	   n, k );
	k+=13 ;
	break ;
      case 14 :
	convloopXY<FLAYOUT, 14, ADDBIAS>(pIn, pKernel, pBias, pOut,
	   inChannel, inWidth, inHeight,
	   outChannel, outWidth, outHeight,
	   kernWidth, kernHeight,
	   inChannelGroup, outChannelGroup,
	   strideHeight, strideWidth,
	   padHeight, padWidth,
	   dilationHeight, dilationWidth,
	   inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	   n, k );
	k+=14 ;
	break ;
      case 15 :
	convloopXY<FLAYOUT, 15, ADDBIAS>(pIn, pKernel, pBias, pOut,
	   inChannel, inWidth, inHeight,
	   outChannel, outWidth, outHeight,
	   kernWidth, kernHeight,
	   inChannelGroup, outChannelGroup,
	   strideHeight, strideWidth,
	   padHeight, padWidth,
	   dilationHeight, dilationWidth,
	   inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	   n, k );
	k+=15 ;
	break ;
      default :
	break ;
      }
      for (; k < outChannelGroup; k+=16) {
	convloopXY<FLAYOUT, 16, ADDBIAS>(pIn, pKernel, pBias, pOut,
	   inChannel, inWidth, inHeight,
	   outChannel, outWidth, outHeight,
	   kernWidth, kernHeight,
	   inChannelGroup, outChannelGroup,
	   strideHeight, strideWidth,
	   padHeight, padWidth,
	   dilationHeight, dilationWidth,
	   inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	   n, k );
      } // outChannel
    } // group
  } // batch
}

extern "C" vednnError_t
vednnConvolutionForward_direct_vecC_dil1_str1_pad1_ker3(
    const vednnTensorParam_t *  	pParamIn,
    const void *  			pDataIn,
    const vednnFilterParam_t *  	pParamKernel,
    const void *  			pDataKernel,
    const vednnBiasParam_t * 		pParamBias,
    const void * 			pDataBias,
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
  const int64_t kernWidth  = pParamKernel->width;
  const int64_t kernHeight = pParamKernel->height;

  const int64_t filter_layout = pParamKernel->layout ;

  const int64_t group          = pParamConv->group;
  const int64_t strideWidth    = pParamConv->strideWidth;;
  const int64_t strideHeight   = pParamConv->strideHeight;
  const int64_t padWidth       = pParamConv->padWidth;
  const int64_t padHeight      = pParamConv->padHeight;
  const int64_t dilationWidth  = pParamConv->dilationWidth;
  const int64_t dilationHeight = pParamConv->dilationHeight;

  const int64_t inChannelGroup  = inChannel  / group;   // equal to pDataKernel->inChannel
  const int64_t outChannelGroup = outChannel / group;   // equal to pDataKernel->outChannel

  const float * pIn     = (const float *) pDataIn;
  const float * pKernel = (const float *) pDataKernel;
  const float * pBias   = (const float *) pDataBias;
  float * const pOut    = (float * const) pDataOut;

  if( filter_layout == VEDNN_FILTER_LAYOUT_NCHW) {
    if( pDataBias == NULL ) {
      convloop<VEDNN_FILTER_LAYOUT_NCHW, false>(pIn, pKernel, pBias, pOut,
		   batch, group,
		   inChannel, inWidth, inHeight,
		   outChannel, outWidth, outHeight,
		   kernWidth, kernHeight,
		   inChannelGroup, outChannelGroup,
		   strideHeight, strideWidth,
		   padHeight, padWidth,
		   dilationHeight, dilationWidth ) ;
    }
    else {
      convloop<VEDNN_FILTER_LAYOUT_NCHW, true>(pIn, pKernel, pBias, pOut,
		   batch, group,
		   inChannel, inWidth, inHeight,
		   outChannel, outWidth, outHeight,
		   kernWidth, kernHeight,
		   inChannelGroup, outChannelGroup,
		   strideHeight, strideWidth,
		   padHeight, padWidth,
		   dilationHeight, dilationWidth ) ;
    }
  }
  else {
    if( pDataBias == NULL ) {
      convloop<VEDNN_FILTER_LAYOUT_HWCN, false>(pIn, pKernel, pBias, pOut,
		   batch, group,
		   inChannel, inWidth, inHeight,
		   outChannel, outWidth, outHeight,
		   kernWidth, kernHeight,
		   inChannelGroup, outChannelGroup,
		   strideHeight, strideWidth,
		   padHeight, padWidth,
		   dilationHeight, dilationWidth ) ;
    }
    else {
      convloop<VEDNN_FILTER_LAYOUT_HWCN, true>(pIn, pKernel, pBias, pOut,
		   batch, group,
		   inChannel, inWidth, inHeight,
		   outChannel, outWidth, outHeight,
		   kernWidth, kernHeight,
		   inChannelGroup, outChannelGroup,
		   strideHeight, strideWidth,
		   padHeight, padWidth,
		   dilationHeight, dilationWidth ) ;
    }
  }


  return VEDNN_SUCCESS;
}
