#include <stdint.h>
#include <stdlib.h>

#include "vednn.h"

#include "velintrin.h"
#define VLEN	(256)

template<filterLayout_t FLAYOUT, int NUMKERNEL, bool ADDBIAS>
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
  const int64_t inChannelGroup,
  const int64_t outChannelGroup,
  const int64_t inGroupOffset,
  const int64_t outGroupOffset,
  const int64_t biasGroupOffset,
  const int64_t kernGroupOffset,
  const int64_t n,
  const int64_t k
)
{
  int64_t outIndex = outGroupOffset + (n * outChannel + k  ) * outHeight * outWidth ;

  const int64_t remain  = NUMKERNEL & 0x1 ;
  const int64_t nPacked = NUMKERNEL >> 1 ;

  const float   bias0  = ADDBIAS ?  pBias[biasGroupOffset+k+ 0] : 0.f ;
  int64_t bias[nPacked] ;
#pragma clang loop unroll(full)
  for(int64_t kk=0; kk<nPacked; kk++) bias[kk] = ADDBIAS ?  _vel_pack_f32p(pBias+biasGroupOffset+k+2*kk+remain, pBias+biasGroupOffset+k+2*kk+remain+1) : 0UL ;

  for (int64_t op = 0; op < outHeight * outWidth; op+=VLEN) {
    const int64_t vl = outHeight * outWidth - op < VLEN ? outHeight * outWidth - op : VLEN ;

    __vr vrsum0  = _vel_vbrds_vsl(bias0, vl) ;
    __vr vrsum[nPacked] ;
#pragma clang loop unroll(full)
    for(int64_t kk=0; kk<nPacked; kk++) vrsum[kk] = _vel_pvbrd_vsl(bias[kk], vl) ;

    int64_t c = 0 ;

    if( ( inChannelGroup & 0x01 ) == 1 ) {
      const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;


      __vr vrin_c0  = _vel_vldu_vssl(4,&pInChannel[op], vl) ;
      __vr vrin_c0P = _vel_vshf_vvvsl(vrin_c0, vrin_c0, VE_VSHUFFLE_YUZU, vl) ;

#define FILTER_OFFSET(k,c) ( kernGroupOffset + filter_index<FLAYOUT>(k,c,0,0, inChannelGroup, outChannelGroup, 1, 1) )

      if( remain ){
	const float kerValue0_c0 = pKernel[FILTER_OFFSET(k+ 0, c+0)] ;
	vrsum0 = _vel_vfmads_vvsvl(vrsum0, kerValue0_c0, vrin_c0, vl) ;
      }

#pragma clang loop unroll(full)
      for(int64_t kk=0; kk<nPacked; kk++) {												\
	const uint64_t kerValue_c0 = _vel_pack_f32p(pKernel + FILTER_OFFSET(k+2*kk+remain,   c+0),
						    pKernel + FILTER_OFFSET(k+2*kk+remain+1, c+0)) ;
	vrsum[kk] = _vel_pvfmad_vvsvl(vrsum[kk], kerValue_c0, vrin_c0P, vl) ;
      }

      c++ ;
    }
    for( ; c < inChannelGroup ; c+=2 ) {
      const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c  ) * inHeight * inWidth ) ;

      __vr vrin_c0  = _vel_vldu_vssl(4,&pInChannel[op], vl) ;
      __vr vrin_c1  = _vel_vldu_vssl(4,&pInChannel[op + inHeight * inWidth ], vl) ;
      __vr vrin_c0P = _vel_vshf_vvvsl(vrin_c0, vrin_c0, VE_VSHUFFLE_YUZU, vl) ;
      __vr vrin_c1P = _vel_vshf_vvvsl(vrin_c1, vrin_c1, VE_VSHUFFLE_YUZU, vl) ;

      if( remain ){
	const float kerValue0_c0 = pKernel[FILTER_OFFSET(k+ 0, c+0)] ;
	const float kerValue0_c1 = pKernel[FILTER_OFFSET(k+ 0, c+1)] ;
	vrsum0 = _vel_vfmads_vvsvl(vrsum0, kerValue0_c0, vrin_c0, vl) ;
	vrsum0 = _vel_vfmads_vvsvl(vrsum0, kerValue0_c1, vrin_c1, vl) ;
      }

#pragma clang loop unroll(full)
      for(int64_t kk=0; kk<nPacked; kk++) {												\
	const uint64_t kerValue_c0 = _vel_pack_f32p(pKernel + FILTER_OFFSET(k+2*kk+remain,   c+0),
						    pKernel + FILTER_OFFSET(k+2*kk+remain+1, c+0)) ;
	vrsum[kk] = _vel_pvfmad_vvsvl(vrsum[kk], kerValue_c0, vrin_c0P, vl) ;
	const uint64_t kerValue_c1 = _vel_pack_f32p(pKernel + FILTER_OFFSET(k+2*kk+remain,   c+1),
						    pKernel + FILTER_OFFSET(k+2*kk+remain+1, c+1)) ;
	vrsum[kk] = _vel_pvfmad_vvsvl(vrsum[kk], kerValue_c1, vrin_c1P, vl) ;
      }

#undef FILTER_OFFSET
    } // inChannel

    if( remain ) {
	_vel_vstu_vssl(vrsum0,  4, pOut+outIndex + 0 * outHeight*outWidth, vl) ;
    }
    for(int64_t kk=0; kk<nPacked; kk++) {
	_vel_vstu_vssl(vrsum[kk], 4, pOut+outIndex + (2*kk+remain)   * outHeight*outWidth, vl) ;
	_vel_vstl_vssl(vrsum[kk], 4, pOut+outIndex + (2*kk+remain+1) * outHeight*outWidth, vl) ;
    }

    outIndex += vl ;
  } // outPixels
}


template<int NUMKERNEL, bool ADDBIAS>
static inline void func_filternchw_avoid_l1m(
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
  const int64_t inChannelGroup,
  const int64_t outChannelGroup,
  const int64_t inGroupOffset,
  const int64_t outGroupOffset,
  const int64_t biasGroupOffset,
  const int64_t kernGroupOffset,
  const int64_t n,
  const int64_t k
)
{

  float __attribute__ ((aligned(8))) filter[NUMKERNEL*512] ;
  uint64_t* filter_u64 = (uint64_t*) filter ;

  int64_t outIndex = outGroupOffset + (n * outChannel + k  ) * outHeight * outWidth ;

  const int64_t remain  = NUMKERNEL & 0x1 ;
  const int64_t nPacked = NUMKERNEL >> 1 ;

  const float   bias0  = ADDBIAS ?  pBias[biasGroupOffset+k+ 0] : 0.f ;
  int64_t bias[nPacked] ;
#pragma clang loop unroll(full)
  for(int64_t kk=0; kk<nPacked; kk++) bias[kk] = ADDBIAS ?  _vel_pack_f32p(pBias+biasGroupOffset+k+2*kk+remain, pBias+biasGroupOffset+k+2*kk+remain+1) : 0UL ;

  for (int64_t op = 0; op < outHeight * outWidth; op+=VLEN) {
    const int64_t vl = outHeight * outWidth - op < VLEN ? outHeight * outWidth - op : VLEN ;

    __vr vrsum0  = _vel_vbrds_vsl(bias0, vl) ;
    __vr vrsum[nPacked] ;
#pragma clang loop unroll(full)
    for(int64_t kk=0; kk<nPacked; kk++) vrsum[kk] = _vel_pvbrd_vsl(bias[kk], vl) ;


    for(int64_t c0=0; c0<inChannelGroup; c0+=256) {
      const int64_t clen = inChannelGroup - c0 < 256 ? inChannelGroup - c0 : 256 ;

      const float *pKerValue = pKernel + kernGroupOffset + (k * inChannelGroup + c0) ;

      __vr vr[NUMKERNEL] ;
#pragma clang loop unroll(full)
      for(int64_t kk=0; kk<NUMKERNEL; kk++) {
	vr[kk] = _vel_vldu_vssl(4, pKerValue+ kk*inChannelGroup, clen) ;
      }

#pragma clang loop unroll(full)
      for(int64_t kk=0; kk<nPacked; kk++) {
	__vr vrp = _vel_vshf_vvvsl(vr[2*kk+remain],vr[2*kk+remain+1],VE_VSHUFFLE_YUZU, clen) ;
	_vel_vst_vssl(vrp, 8, filter_u64+kk*clen, clen) ;
      }
      if( remain ) {
	_vel_vstu_vssl(vr[0], 4, filter+(NUMKERNEL-1)*clen, clen) ;
      }

      for(int64_t c1 = 0; c1 < clen ; c1++ ) {
	const int64_t c = c0 + c1 ;

	const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;

	__vr vrin  = _vel_vldu_vssl(4,&pInChannel[op], vl) ;
	__vr vrinP = _vel_vshf_vvvsl(vrin, vrin, VE_VSHUFFLE_YUZU, vl) ;

	if( remain ) {
	  vrsum0 = _vel_vfmads_vvsvl(vrsum0, filter[(NUMKERNEL-1)*clen+c1], vrin, vl) ;
	}
#pragma clang loop unroll(full)
	for(int64_t kk=0; kk<nPacked; kk++) {
	  vrsum[kk] = _vel_pvfmad_vvsvl(vrsum[kk], filter_u64[kk*clen+c1], vrinP, vl) ;
	}
      } // inChannel
    }

    if( remain ) {
      _vel_vstu_vssl(vrsum0,  4, pOut+outIndex + 0 * outHeight*outWidth, vl) ;
    }
    for(int64_t kk=0; kk<nPacked; kk++) {
      _vel_vstu_vssl(vrsum[kk], 4, pOut+outIndex + (2*kk+remain)   * outHeight*outWidth, vl) ;
      _vel_vstl_vssl(vrsum[kk], 4, pOut+outIndex + (2*kk+remain+1) * outHeight*outWidth, vl) ;
    }

    outIndex += vl ;
  } // outPixels
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
    const int64_t inChannelGroup,
    const int64_t outChannelGroup
)
{
  for (int64_t n = 0; n < batch; n++) {
    for (int64_t g = 0; g < group; g++) {
	const int64_t inGroupOffset   = g * inChannelGroup * inHeight * inWidth;
	const int64_t outGroupOffset  = g * outChannelGroup * outHeight * outWidth;
	const int64_t biasGroupOffset = g * outChannelGroup;
	const int64_t kernGroupOffset = g * outChannelGroup * inChannelGroup * 1 * 1;

	const int64_t remain = outChannelGroup & 0xf ;

	int k = 0 ;
	switch( remain ) {
	case 1 :
	  func<FLAYOUT, 1, ADDBIAS>(pIn, pKernel, pBias, pOut,
	     inChannel, inWidth, inHeight,
	     outChannel, outWidth, outHeight,
	     inChannelGroup, outChannelGroup,
	     inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	     n, k );
	  k+=1 ;
	  break ;
	case 2 :
	  if( FLAYOUT == VEDNN_FILTER_LAYOUT_NCHW
	      && inChannelGroup >= 512
	      && ( (inChannelGroup % 8192) < 64 || (inChannelGroup % 8192) > 8192-64 ) )
	  {
	    func_filternchw_avoid_l1m<2, ADDBIAS>(pIn, pKernel, pBias, pOut,
	       inChannel, inWidth, inHeight,
	       outChannel, outWidth, outHeight,
	       inChannelGroup, outChannelGroup,
	       inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	       n, k );
	  }
	  else
	  {
	    func<FLAYOUT, 2, ADDBIAS>(pIn, pKernel, pBias, pOut,
	       inChannel, inWidth, inHeight,
	       outChannel, outWidth, outHeight,
	       inChannelGroup, outChannelGroup,
	       inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	       n, k );
	  }
	  k+=2 ;
	  break ;
	case 3 :
	  if( FLAYOUT == VEDNN_FILTER_LAYOUT_NCHW
	      && inChannelGroup >= 512
	      && ( (inChannelGroup % 4096) < 32 || (inChannelGroup % 4096) > 4096-32 ) )
	  {
	    func_filternchw_avoid_l1m<3, ADDBIAS>(pIn, pKernel, pBias, pOut,
	       inChannel, inWidth, inHeight,
	       outChannel, outWidth, outHeight,
	       inChannelGroup, outChannelGroup,
	       inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	       n, k );
	  }
	  else
	  {
	    func<FLAYOUT, 3, ADDBIAS>(pIn, pKernel, pBias, pOut,
	       inChannel, inWidth, inHeight,
	       outChannel, outWidth, outHeight,
	       inChannelGroup, outChannelGroup,
	       inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	       n, k );
	  }
	  k+=3 ;
	  break ;
	case 4 :
	  if( FLAYOUT == VEDNN_FILTER_LAYOUT_NCHW
	      && inChannelGroup >= 512
	      && ( (inChannelGroup % 4096) < 32 || (inChannelGroup % 4096) > 4096-32 ) )
	  {
	    func_filternchw_avoid_l1m<4, ADDBIAS>(pIn, pKernel, pBias, pOut,
	       inChannel, inWidth, inHeight,
	       outChannel, outWidth, outHeight,
	       inChannelGroup, outChannelGroup,
	       inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	       n, k );
	  }
	  else
	  {
	    func<FLAYOUT, 4, ADDBIAS>(pIn, pKernel, pBias, pOut,
	       inChannel, inWidth, inHeight,
	       outChannel, outWidth, outHeight,
	       inChannelGroup, outChannelGroup,
	       inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	       n, k );
	  }
	  k+=4 ;
	  break ;
	case 5 :
	  if( FLAYOUT == VEDNN_FILTER_LAYOUT_NCHW
	      && inChannelGroup >= 512
	      && ( (inChannelGroup % 2048) < 16 || (inChannelGroup % 2048) > 2048-16 ) )
	  {
	    func_filternchw_avoid_l1m<5, ADDBIAS>(pIn, pKernel, pBias, pOut,
	       inChannel, inWidth, inHeight,
	       outChannel, outWidth, outHeight,
	       inChannelGroup, outChannelGroup,
	       inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	       n, k );
	  }
	  else
	  {
	    func<FLAYOUT, 5, ADDBIAS>(pIn, pKernel, pBias, pOut,
	       inChannel, inWidth, inHeight,
	       outChannel, outWidth, outHeight,
	       inChannelGroup, outChannelGroup,
	       inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	       n, k );
	  }
	  k+=5 ;
	  break ;
	case 6 :
	  if( FLAYOUT == VEDNN_FILTER_LAYOUT_NCHW
	      && inChannelGroup >= 512
	      && ( (inChannelGroup % 2048) < 16 || (inChannelGroup % 2048) > 2048-16 ) )
	  {
	    func_filternchw_avoid_l1m<6, ADDBIAS>(pIn, pKernel, pBias, pOut,
	       inChannel, inWidth, inHeight,
	       outChannel, outWidth, outHeight,
	       inChannelGroup, outChannelGroup,
	       inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	       n, k );
	  }
	  else
	  {
	    func<FLAYOUT, 6, ADDBIAS>(pIn, pKernel, pBias, pOut,
	       inChannel, inWidth, inHeight,
	       outChannel, outWidth, outHeight,
	       inChannelGroup, outChannelGroup,
	       inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	       n, k );
	  }
	  k+=6 ;
	  break ;
	case 7 :
	  if( FLAYOUT == VEDNN_FILTER_LAYOUT_NCHW
	      && inChannelGroup >= 512
	      && ( (inChannelGroup % 2048) < 16 || (inChannelGroup % 2048) > 2048-16 ) )
	  {
	    func_filternchw_avoid_l1m<7, ADDBIAS>(pIn, pKernel, pBias, pOut,
	       inChannel, inWidth, inHeight,
	       outChannel, outWidth, outHeight,
	       inChannelGroup, outChannelGroup,
	       inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	       n, k );
	  }
	  else
	  {
	    func<FLAYOUT, 7, ADDBIAS>(pIn, pKernel, pBias, pOut,
	       inChannel, inWidth, inHeight,
	       outChannel, outWidth, outHeight,
	       inChannelGroup, outChannelGroup,
	       inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	       n, k );
	  }
	  k+=7 ;
	  break ;
	case 8 :
	  if( FLAYOUT == VEDNN_FILTER_LAYOUT_NCHW
	      && inChannelGroup >= 512
	      && ( (inChannelGroup % 2048) < 16 || (inChannelGroup % 2048) > 2048-16 ) )
	  {
	    func_filternchw_avoid_l1m<8, ADDBIAS>(pIn, pKernel, pBias, pOut,
	       inChannel, inWidth, inHeight,
	       outChannel, outWidth, outHeight,
	       inChannelGroup, outChannelGroup,
	       inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	       n, k );
	  }
	  else
	  {
	    func<FLAYOUT, 8, ADDBIAS>(pIn, pKernel, pBias, pOut,
	       inChannel, inWidth, inHeight,
	       outChannel, outWidth, outHeight,
	       inChannelGroup, outChannelGroup,
	       inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	       n, k );
	  }
	  k+=8 ;
	  break ;
	case 9 :
	  if( FLAYOUT == VEDNN_FILTER_LAYOUT_NCHW
	      && inChannelGroup >= 512
	      && ( (inChannelGroup % 1024) < 8 || (inChannelGroup % 1024) > 1024-8 ) )
	  {
	    func_filternchw_avoid_l1m<9, ADDBIAS>(pIn, pKernel, pBias, pOut,
	       inChannel, inWidth, inHeight,
	       outChannel, outWidth, outHeight,
	       inChannelGroup, outChannelGroup,
	       inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	       n, k );
	  }
	  else
	  {
	    func<FLAYOUT, 9, ADDBIAS>(pIn, pKernel, pBias, pOut,
	       inChannel, inWidth, inHeight,
	       outChannel, outWidth, outHeight,
	       inChannelGroup, outChannelGroup,
	       inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	       n, k );
	  }
	  k+=9 ;
	  break ;
	case 10 :
	  if( FLAYOUT == VEDNN_FILTER_LAYOUT_NCHW
	      && inChannelGroup >= 512
	      && ( (inChannelGroup % 1024) < 8 || (inChannelGroup % 1024) > 1024-8 ) )
	  {
	    func_filternchw_avoid_l1m<10, ADDBIAS>(pIn, pKernel, pBias, pOut,
	       inChannel, inWidth, inHeight,
	       outChannel, outWidth, outHeight,
	       inChannelGroup, outChannelGroup,
	       inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	       n, k );
	  }
	  else
	  {
	    func<FLAYOUT, 10, ADDBIAS>(pIn, pKernel, pBias, pOut,
	       inChannel, inWidth, inHeight,
	       outChannel, outWidth, outHeight,
	       inChannelGroup, outChannelGroup,
	       inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	       n, k );
	  }
	  k+=10 ;
	  break ;
	case 11 :
	  if( FLAYOUT == VEDNN_FILTER_LAYOUT_NCHW
	      && inChannelGroup >= 512
	      && ( (inChannelGroup % 1024) < 8 || (inChannelGroup % 1024) > 1024-8 ) )
	  {
	    func_filternchw_avoid_l1m<11, ADDBIAS>(pIn, pKernel, pBias, pOut,
	       inChannel, inWidth, inHeight,
	       outChannel, outWidth, outHeight,
	       inChannelGroup, outChannelGroup,
	       inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	       n, k );
	  }
	  else
	  {
	    func<FLAYOUT, 11, ADDBIAS>(pIn, pKernel, pBias, pOut,
	       inChannel, inWidth, inHeight,
	       outChannel, outWidth, outHeight,
	       inChannelGroup, outChannelGroup,
	       inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	       n, k );
	  }
	  k+=11 ;
	  break ;
	case 12 :
	  if( FLAYOUT == VEDNN_FILTER_LAYOUT_NCHW
	      && inChannelGroup >= 512
	      && ( (inChannelGroup % 1024) < 8 || (inChannelGroup % 1024) > 1024-8 ) )
	  {
	    func_filternchw_avoid_l1m<12, ADDBIAS>(pIn, pKernel, pBias, pOut,
	       inChannel, inWidth, inHeight,
	       outChannel, outWidth, outHeight,
	       inChannelGroup, outChannelGroup,
	       inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	       n, k );
	  }
	  else
	  {
	    func<FLAYOUT, 12, ADDBIAS>(pIn, pKernel, pBias, pOut,
	       inChannel, inWidth, inHeight,
	       outChannel, outWidth, outHeight,
	       inChannelGroup, outChannelGroup,
	       inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	       n, k );
	  }
	  k+=12 ;
	  break ;
	case 13 :
	  if( FLAYOUT == VEDNN_FILTER_LAYOUT_NCHW
	      && inChannelGroup >= 512
	      && ( (inChannelGroup % 1024) < 8 || (inChannelGroup % 1024) > 1024-8 ) )
	  {
	    func_filternchw_avoid_l1m<13, ADDBIAS>(pIn, pKernel, pBias, pOut,
	       inChannel, inWidth, inHeight,
	       outChannel, outWidth, outHeight,
	       inChannelGroup, outChannelGroup,
	       inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	       n, k );
	  }
	  else
	  {
	    func<FLAYOUT, 13, ADDBIAS>(pIn, pKernel, pBias, pOut,
	       inChannel, inWidth, inHeight,
	       outChannel, outWidth, outHeight,
	       inChannelGroup, outChannelGroup,
	       inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	       n, k );
	  }
	  k+=13 ;
	  break ;
	case 14 :
	  if( FLAYOUT == VEDNN_FILTER_LAYOUT_NCHW
	      && inChannelGroup >= 512
	      && ( (inChannelGroup % 1024) < 8 || (inChannelGroup % 1024) > 1024-8 ) )
	  {
	    func_filternchw_avoid_l1m<14, ADDBIAS>(pIn, pKernel, pBias, pOut,
	       inChannel, inWidth, inHeight,
	       outChannel, outWidth, outHeight,
	       inChannelGroup, outChannelGroup,
	       inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	       n, k );
	  }
	  else
	  {
	    func<FLAYOUT, 14, ADDBIAS>(pIn, pKernel, pBias, pOut,
	       inChannel, inWidth, inHeight,
	       outChannel, outWidth, outHeight,
	       inChannelGroup, outChannelGroup,
	       inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	       n, k );
	  }
	  k+=14 ;
	  break ;
	case 15 :
	  if( FLAYOUT == VEDNN_FILTER_LAYOUT_NCHW
	      && inChannelGroup >= 512
	      && ( (inChannelGroup % 1024) < 8 || (inChannelGroup % 1024) > 1024-8 ) )
	  {
	    func_filternchw_avoid_l1m<15, ADDBIAS>(pIn, pKernel, pBias, pOut,
	       inChannel, inWidth, inHeight,
	       outChannel, outWidth, outHeight,
	       inChannelGroup, outChannelGroup,
	       inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	       n, k );
	  }
	  else
	  {
	    func<FLAYOUT, 15, ADDBIAS>(pIn, pKernel, pBias, pOut,
	       inChannel, inWidth, inHeight,
	       outChannel, outWidth, outHeight,
	       inChannelGroup, outChannelGroup,
	       inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	       n, k );
	  }
	  k+=15 ;
	  break ;
	default :
	  break ;
	}
	for (; k < outChannelGroup; k+=16) {
	  if( FLAYOUT == VEDNN_FILTER_LAYOUT_NCHW
	      && inChannelGroup >= 512
	      && ( (inChannelGroup % 1024) < 8 || (inChannelGroup % 1024) > 1024-8 ) )
	  {
	    func_filternchw_avoid_l1m<16, ADDBIAS>(pIn, pKernel, pBias, pOut,
	       inChannel, inWidth, inHeight,
	       outChannel, outWidth, outHeight,
	       inChannelGroup, outChannelGroup,
	       inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	       n, k );
	  }
	  else
	  {
	    func<FLAYOUT, 16, ADDBIAS>(pIn, pKernel, pBias, pOut,
	       inChannel, inWidth, inHeight,
	       outChannel, outWidth, outHeight,
	       inChannelGroup, outChannelGroup,
	       inGroupOffset, outGroupOffset, biasGroupOffset, kernGroupOffset,
	       n, k );
	  }
	} // outChannel
    } // group
  } // batch
}

extern "C" vednnError_t
vednnConvolutionForward_direct_dil1_str1_pad0_ker1(
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
//  const int64_t kernWidth  = pParamKernel->width;		/* must be 1 */
//  const int64_t kernHeight = pParamKernel->height;		/* must be 1 */

  const int64_t filter_layout = pParamKernel->layout ;

  const int64_t group          = pParamConv->group;
//  const int64_t strideWidth    = pParamConv->strideWidth;	/* must be 1 */
//  const int64_t strideHeight   = pParamConv->strideHeight;	/* must be 1 */
//  const int64_t padWidth       = pParamConv->padWidth;	/* must be 0 */
//  const int64_t padHeight      = pParamConv->padHeight;	/* must be 0 */
//  const int64_t dilationWidth  = pParamConv->dilationWidth;	/* must be 1 */
//  const int64_t dilationHeight = pParamConv->dilationHeight;	/* must be 1 */

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
		 inChannelGroup, outChannelGroup ) ;
    }
    else {
      convloop<VEDNN_FILTER_LAYOUT_NCHW, true>(pIn, pKernel, pBias, pOut,
		 batch, group,
		 inChannel, inWidth, inHeight,
		 outChannel, outWidth, outHeight,
		 inChannelGroup, outChannelGroup ) ;
    }
  }
  else {
    if( pDataBias == NULL ) {
      convloop<VEDNN_FILTER_LAYOUT_HWCN, false>(pIn, pKernel, pBias, pOut,
		 batch, group,
		 inChannel, inWidth, inHeight,
		 outChannel, outWidth, outHeight,
		 inChannelGroup, outChannelGroup ) ;
    }
    else {
      convloop<VEDNN_FILTER_LAYOUT_HWCN, true>(pIn, pKernel, pBias, pOut,
		 batch, group,
		 inChannel, inWidth, inHeight,
		 outChannel, outWidth, outHeight,
		 inChannelGroup, outChannelGroup ) ;
    }
  }

  return VEDNN_SUCCESS;
}
