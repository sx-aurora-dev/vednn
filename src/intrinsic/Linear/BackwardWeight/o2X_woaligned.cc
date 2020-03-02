#include <stdint.h>
#include <float.h>

#include <stdio.h>

#include "vednn.h"

#include "velintrin.h"

template <int BATCH, bool UPDATE>
static inline void funcUseScratchPad(
  const uint64_t	inDim,
  const uint64_t	outDim,
  const uint64_t	nInDim,
  const float * 	pIn,
  const float * 	pGOut,
  float * 		pGWeight,
  const uint64_t 	mvl
)
{
  float __attribute__ ((aligned(8))) scratch[BATCH*256*2] ;

  for(int64_t o=0; o<outDim; o+=2*mvl) {
    const int64_t vl = (outDim-o < 2*mvl ? outDim - o : 2*mvl) >> 1 ;

    __vr vrgout[BATCH] ;
#pragma clang loop unroll(full)
    for(int64_t b=0; b<BATCH; b++) {
      vrgout[b] = _vel_vld_vssl(8, pGOut+b*outDim+o, vl) ;
    }

    for(int64_t i0=0; i0<nInDim; i0+=256) {
      const int64_t ilen = nInDim - i0 < 256 ? nInDim - i0 : 256 ;

      // copy Input to Scratch
      {
	__vr vri[BATCH] ;
#pragma clang loop unroll(full)
	for(int64_t b=0; b<BATCH; b++) {
	  vri[b] = _vel_vldu_vssl(4, pIn+b*inDim+i0, ilen) ;
	}
#pragma clang loop unroll(full)
	for(int64_t b=0; b<BATCH; b++) {
	  _vel_vstu_vssl(vri[b], 4, scratch+b*ilen, ilen) ;
	}
      }

      int64_t i1=0;
      if(ilen & 0x1) {
#define CALC_STORE(I) {							\
	  __vr vrgw ;							\
	  if(UPDATE) {							\
	    vrgw = _vel_vld_vssl(8, pGWeight+(i0+(I))*outDim+o, vl) ;	\
	    const uint64_t in = _vel_pack_f32a(scratch+0*ilen+(I)) ;	\
	    vrgw = _vel_pvfmad_vvsvl(vrgw, in, vrgout[0], vl) ;		\
	  }								\
	  else {							\
	    const uint64_t in = _vel_pack_f32a(scratch+0*ilen+(I)) ;	\
	    vrgw = _vel_pvfmul_vsvl(in, vrgout[0], vl) ;		\
	  }								\
	  _Pragma("clang loop unroll(full)")				\
	  for(int64_t b=1; b<BATCH; b++) {				\
	    const uint64_t in = _vel_pack_f32a(scratch+b*ilen+(I)) ;	\
	    vrgw = _vel_pvfmad_vvsvl(vrgw, in, vrgout[b], vl) ;		\
	  }								\
	  _vel_vst_vssl(vrgw, 8, pGWeight+(i0+(I))*outDim+o, vl) ;	\
	}

	CALC_STORE(i1+0) ;
	i1+=1;
      }
      if((ilen>>1) & 0x1) {
	CALC_STORE(i1+0) ;
	CALC_STORE(i1+1) ;
	i1+=2 ;
      }
      if((ilen>>2) & 0x1) {
	CALC_STORE(i1+0) ;
	CALC_STORE(i1+1) ;
	CALC_STORE(i1+2) ;
	CALC_STORE(i1+3) ;
	i1+=4 ;
      }
      for(; i1<ilen; i1+=8) {
	CALC_STORE(i1+0) ;
	CALC_STORE(i1+1) ;
	CALC_STORE(i1+2) ;
	CALC_STORE(i1+3) ;
	CALC_STORE(i1+4) ;
	CALC_STORE(i1+5) ;
	CALC_STORE(i1+6) ;
	CALC_STORE(i1+7) ;
#undef CALC_STORE
      }
    }
  }
}


template <int BATCH, bool UPDATE>
static inline void func(
  const uint64_t	inDim,
  const uint64_t	outDim,
  const uint64_t	nInDim,
  const float * 	pIn,
  const float * 	pGOut,
  float * 		pGWeight,
  const uint64_t 	mvl
)
{
  for(int64_t o=0; o<outDim; o+=2*mvl) {
    const int64_t vl = (outDim-o < 2*mvl ? outDim - o : 2*mvl) >> 1 ;


    __vr vrgout[BATCH] ;
#pragma clang loop unroll(full)
    for(int64_t b=0; b<BATCH; b++) {
      vrgout[b] = _vel_vld_vssl(8, pGOut+b*outDim+o, vl) ;
    }

    int64_t i=0;
    if(nInDim & 0x1) {
#define CALC_STORE(I) {						\
	__vr vrgw ;						\
	if(UPDATE) {						\
	  vrgw = _vel_vld_vssl(8, pGWeight+(I)*outDim+o, vl) ;	\
	  const uint64_t in = _vel_pack_f32a(pIn+0*inDim+(I)) ;	\
	  vrgw = _vel_pvfmad_vvsvl(vrgw, in, vrgout[0], vl) ;	\
	}							\
	else {							\
	  const uint64_t in = _vel_pack_f32a(pIn+0*inDim+(I)) ;	\
	  vrgw = _vel_pvfmul_vsvl(in, vrgout[0], vl) ;		\
	}							\
	_Pragma("clang loop unroll(full)")			\
	for(int64_t b=1; b<BATCH; b++) {			\
	  const uint64_t in = _vel_pack_f32a(pIn+b*inDim+(I)) ;	\
	  vrgw = _vel_pvfmad_vvsvl(vrgw, in, vrgout[b], vl) ;	\
	}							\
	_vel_vst_vssl(vrgw, 8, pGWeight+(I)*outDim+o, vl) ;	\
      }

      CALC_STORE(i+0) ;
      i+=1;
    }
    if((nInDim>>1) & 0x1) {
      CALC_STORE(i+0) ;
      CALC_STORE(i+1) ;
      i+=2 ;
    }
    if((nInDim>>2) & 0x1) {
      CALC_STORE(i+0) ;
      CALC_STORE(i+1) ;
      CALC_STORE(i+2) ;
      CALC_STORE(i+3) ;
      i+=4 ;
    }
    for(; i<nInDim; i+=8) {
      CALC_STORE(i+0) ;
      CALC_STORE(i+1) ;
      CALC_STORE(i+2) ;
      CALC_STORE(i+3) ;
      CALC_STORE(i+4) ;
      CALC_STORE(i+5) ;
      CALC_STORE(i+6) ;
      CALC_STORE(i+7) ;
#undef CALC_STORE
    }
  }
}


extern "C"
vednnError_t vednnLinearBackwardWeight_o2X_woaligned(
    const uint64_t			inDim,
    const uint64_t			outDim,
    const uint64_t			nBatch,
    const void * 			pDataIn,
    const void * 			pDataGradOut,
    void * 				pDataGradWeight
#ifdef VEDNN_USE_OPENMP
    ,
    const uint64_t			inDimBegin,
    const uint64_t			inDimEnd
#endif
)
{
  const float * __restrict__ pIn       = (const float * __restrict__) pDataIn;
  const float * __restrict__ pGOut     = (const float * __restrict__) pDataGradOut;
  float * __restrict__ const pGWeight  = (float * __restrict__ const) pDataGradWeight;

#ifndef VEDNN_USE_OPENMP
    const uint64_t inDimBegin = 0 ;
    const uint64_t inDimEnd   = inDim ;
#endif

  int64_t n=0;
  int64_t batchRemain = nBatch & 0xf ;

  int64_t mvl ;
  if( outDim % (256*2) == 0 )
    mvl = 256 ;
  else if ( outDim % (192*2) == 0 )
    mvl = 192 ;
  else if( outDim % (256*2) < outDim % (192*2) )
    mvl = 192 ;
  else
    mvl = 256 ;

  switch( batchRemain ) {
  case 1 :
    func<1,false>(inDim, outDim, inDimEnd-inDimBegin,
	          pIn+inDimBegin+n*inDim, pGOut+n*outDim, pGWeight+inDimBegin*outDim, mvl) ;
    n+=1 ;
    break ;
  case 2 :
   func<2,false>(inDim, outDim, inDimEnd-inDimBegin,
	         pIn+inDimBegin+n*inDim, pGOut+n*outDim, pGWeight+inDimBegin*outDim, mvl) ;
    n+=2 ;
    break ;
  case 3 :
    func<3,false>(inDim, outDim, inDimEnd-inDimBegin,
	          pIn+inDimBegin+n*inDim, pGOut+n*outDim, pGWeight+inDimBegin*outDim, mvl) ;
    n+=3 ;
    break ;
  case 4 :
    func<4,false>(inDim, outDim, inDimEnd-inDimBegin,
	          pIn+inDimBegin+n*inDim, pGOut+n*outDim, pGWeight+inDimBegin*outDim, mvl) ;
    n+=4 ;
    break ;
  case 5 :
    func<5,false>(inDim, outDim, inDimEnd-inDimBegin,
	          pIn+inDimBegin+n*inDim, pGOut+n*outDim, pGWeight+inDimBegin*outDim, mvl) ;
    n+=5 ;
    break ;
  case 6 :
    func<6,false>(inDim, outDim, inDimEnd-inDimBegin,
	          pIn+inDimBegin+n*inDim, pGOut+n*outDim, pGWeight+inDimBegin*outDim, mvl) ;
    n+=6 ;
    break ;
  case 7 :
    func<7,false>(inDim, outDim, inDimEnd-inDimBegin,
	          pIn+inDimBegin+n*inDim, pGOut+n*outDim, pGWeight+inDimBegin*outDim, mvl) ;
    n+=7 ;
    break ;
  case 8 :
    func<8,false>(inDim, outDim, inDimEnd-inDimBegin,
	          pIn+inDimBegin+n*inDim, pGOut+n*outDim, pGWeight+inDimBegin*outDim, mvl) ;
    n+=8 ;
    break ;
  case 9 :
    func<9,false>(inDim, outDim, inDimEnd-inDimBegin,
	          pIn+inDimBegin+n*inDim, pGOut+n*outDim, pGWeight+inDimBegin*outDim, mvl) ;
    n+=9 ;
    break ;
  case 10 :
    func<10,false>(inDim, outDim, inDimEnd-inDimBegin,
	          pIn+inDimBegin+n*inDim, pGOut+n*outDim, pGWeight+inDimBegin*outDim, mvl) ;
    n+=10 ;
    break ;
  case 11 :
    func<11,false>(inDim, outDim, inDimEnd-inDimBegin,
	          pIn+inDimBegin+n*inDim, pGOut+n*outDim, pGWeight+inDimBegin*outDim, mvl) ;
    n+=11 ;
    break ;
  case 12 :
    func<12,false>(inDim, outDim, inDimEnd-inDimBegin,
	          pIn+inDimBegin+n*inDim, pGOut+n*outDim, pGWeight+inDimBegin*outDim, mvl) ;
    n+=12 ;
    break ;
  case 13 :
    func<13,false>(inDim, outDim, inDimEnd-inDimBegin,
	          pIn+inDimBegin+n*inDim, pGOut+n*outDim, pGWeight+inDimBegin*outDim, mvl) ;
    n+=13 ;
    break ;
  case 14 :
    func<14,false>(inDim, outDim, inDimEnd-inDimBegin,
	          pIn+inDimBegin+n*inDim, pGOut+n*outDim, pGWeight+inDimBegin*outDim, mvl) ;
    n+=14 ;
    break ;
  case 15 :
    func<15,false>(inDim, outDim, inDimEnd-inDimBegin,
	          pIn+inDimBegin+n*inDim, pGOut+n*outDim, pGWeight+inDimBegin*outDim, mvl) ;
    n+=15 ;
    break ;
  default :
    if( nBatch >= 16 ) {
      func<16,false>(inDim, outDim, inDimEnd-inDimBegin,
		    pIn+inDimBegin+n*inDim, pGOut+n*outDim, pGWeight+inDimBegin*outDim, mvl) ;
      n+=16 ;
    }
    break ;
  }

  if( inDim % 1024 < 8 || inDim % 1024 > 1024-8) {
    // to avoid l1 cache miss, use scratch-pad
    for(; n<nBatch; n+=16) {
      funcUseScratchPad<16,true>(inDim, outDim, inDimEnd-inDimBegin,
  		 pIn+inDimBegin+n*inDim, pGOut+n*outDim, pGWeight+inDimBegin*outDim, mvl) ;
    }
  }
  else {
    for(; n<nBatch; n+=16) {
      func<16,true>(inDim, outDim, inDimEnd-inDimBegin,
  		 pIn+inDimBegin+n*inDim, pGOut+n*outDim, pGWeight+inDimBegin*outDim, mvl) ;
    }
  }
  return VEDNN_SUCCESS ;
}
