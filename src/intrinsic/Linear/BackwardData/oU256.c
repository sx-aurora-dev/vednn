#include <stdint.h>
#include <float.h>

#include <stdio.h>

#include "vednn.h"

#include "velintrin.h"
#define VLEN	(256)


static inline void b1(
  const uint64_t	inDim,
  const uint64_t	outDim,
  const float * 	pGOut,
  const float * 	pWeight,
  float * 		pGIn
)
{
  __vr vro_b0 = _vel_vldu_vssl(4, pGOut+0*outDim, outDim) ;

  int64_t i=0 ;
  switch(inDim&0x3) {
  case 1:
    {
      __vr vrw_i0 = _vel_vldu_vssl(4, pWeight+(i  )*outDim, outDim) ;

      __vr vrsum_b0_i0 = _vel_vfsums_vvl(_vel_vfmuls_vvvl(vrw_i0, vro_b0, outDim), outDim) ;

      _vel_vstu_vssl(vrsum_b0_i0, 4, pGIn+0*inDim+i, 1) ;
    }
    i+=1;
    break ;
  case 2 :
    {
      __vr vrw_i0 = _vel_vldu_vssl(4, pWeight+(i  )*outDim, outDim) ;
      __vr vrw_i1 = _vel_vldu_vssl(4, pWeight+(i+1)*outDim, outDim) ;

      __vr vrsum_b0_i0 = _vel_vfsums_vvl(_vel_vfmuls_vvvl(vrw_i0, vro_b0, outDim), outDim) ;
      __vr vrsum_b0_i1 = _vel_vfsums_vvl(_vel_vfmuls_vvvl(vrw_i1, vro_b0, outDim), outDim) ;

      _vel_vstu_vssl(vrsum_b0_i0, 4, pGIn+0*inDim+i, 1) ;
      _vel_vstu_vssl(vrsum_b0_i1, 4, pGIn+0*inDim+i+1, 1) ;
    }

    i+=2 ;
    break ;
  case 3:
    {
      __vr vrw_i0 = _vel_vldu_vssl(4, pWeight+(i  )*outDim, outDim) ;
      __vr vrw_i1 = _vel_vldu_vssl(4, pWeight+(i+1)*outDim, outDim) ;
      __vr vrw_i2 = _vel_vldu_vssl(4, pWeight+(i+2)*outDim, outDim) ;

      __vr vrsum_b0_i0 = _vel_vfsums_vvl(_vel_vfmuls_vvvl(vrw_i0, vro_b0, outDim), outDim) ;
      __vr vrsum_b0_i1 = _vel_vfsums_vvl(_vel_vfmuls_vvvl(vrw_i1, vro_b0, outDim), outDim) ;
      __vr vrsum_b0_i2 = _vel_vfsums_vvl(_vel_vfmuls_vvvl(vrw_i2, vro_b0, outDim), outDim) ;

      _vel_vstu_vssl(vrsum_b0_i0, 4, pGIn+0*inDim+i, 1) ;
      _vel_vstu_vssl(vrsum_b0_i1, 4, pGIn+0*inDim+i+1, 1) ;
      _vel_vstu_vssl(vrsum_b0_i2, 4, pGIn+0*inDim+i+2, 1) ;
    }
    i+=3 ;
    break ;
  default : break ;
  }
  for(; i<inDim; i+=4)
  {
    __vr vrw_i0 = _vel_vldu_vssl(4, pWeight+(i  )*outDim, outDim) ;
    __vr vrw_i1 = _vel_vldu_vssl(4, pWeight+(i+1)*outDim, outDim) ;
    __vr vrw_i2 = _vel_vldu_vssl(4, pWeight+(i+2)*outDim, outDim) ;
    __vr vrw_i3 = _vel_vldu_vssl(4, pWeight+(i+3)*outDim, outDim) ;

    __vr vrsum_b0_i0 = _vel_vfsums_vvl(_vel_vfmuls_vvvl(vrw_i0, vro_b0, outDim), outDim) ;
    __vr vrsum_b0_i1 = _vel_vfsums_vvl(_vel_vfmuls_vvvl(vrw_i1, vro_b0, outDim), outDim) ;
    __vr vrsum_b0_i2 = _vel_vfsums_vvl(_vel_vfmuls_vvvl(vrw_i2, vro_b0, outDim), outDim) ;
    __vr vrsum_b0_i3 = _vel_vfsums_vvl(_vel_vfmuls_vvvl(vrw_i3, vro_b0, outDim), outDim) ;

    _vel_vstu_vssl(vrsum_b0_i0, 4, pGIn+0*inDim+i, 1) ;
    _vel_vstu_vssl(vrsum_b0_i1, 4, pGIn+0*inDim+i+1, 1) ;
    _vel_vstu_vssl(vrsum_b0_i2, 4, pGIn+0*inDim+i+2, 1) ;
    _vel_vstu_vssl(vrsum_b0_i3, 4, pGIn+0*inDim+i+3, 1) ;
  }
}


static inline void b2(
  const uint64_t	inDim,
  const uint64_t	outDim,
  const float * 	pGOut,
  const float * 	pWeight,
  float * 		pGIn
)
{
  __vr vro_b0 = _vel_vldu_vssl(4, pGOut+0*outDim, outDim) ;
  __vr vro_b1 = _vel_vldu_vssl(4, pGOut+1*outDim, outDim) ;

  int64_t i=0 ;
  switch(inDim&0x3) {
  case 1:
    {
      __vr vrw_i0 = _vel_vldu_vssl(4, pWeight+(i  )*outDim, outDim) ;

      __vr vrsum_b0_i0 = _vel_vfsums_vvl(_vel_vfmuls_vvvl(vrw_i0, vro_b0, outDim), outDim) ;
      __vr vrsum_b1_i0 = _vel_vfsums_vvl(_vel_vfmuls_vvvl(vrw_i0, vro_b1, outDim), outDim) ;

      _vel_vstu_vssl(vrsum_b0_i0, 4, pGIn+0*inDim+i, 1) ;
      _vel_vstu_vssl(vrsum_b1_i0, 4, pGIn+1*inDim+i, 1) ;
    }
    i+=1;
    break ;
  case 2 :
    {
      __vr vrw_i0 = _vel_vldu_vssl(4, pWeight+(i  )*outDim, outDim) ;
      __vr vrw_i1 = _vel_vldu_vssl(4, pWeight+(i+1)*outDim, outDim) ;

      __vr vrsum_b0_i0 = _vel_vfsums_vvl(_vel_vfmuls_vvvl(vrw_i0, vro_b0, outDim), outDim) ;
      __vr vrsum_b1_i0 = _vel_vfsums_vvl(_vel_vfmuls_vvvl(vrw_i0, vro_b1, outDim), outDim) ;

      __vr vrsum_b0_i1 = _vel_vfsums_vvl(_vel_vfmuls_vvvl(vrw_i1, vro_b0, outDim), outDim) ;
      __vr vrsum_b1_i1 = _vel_vfsums_vvl(_vel_vfmuls_vvvl(vrw_i1, vro_b1, outDim), outDim) ;

      _vel_vstu_vssl(vrsum_b0_i0, 4, pGIn+0*inDim+i, 1) ;
      _vel_vstu_vssl(vrsum_b1_i0, 4, pGIn+1*inDim+i, 1) ;
      _vel_vstu_vssl(vrsum_b0_i1, 4, pGIn+0*inDim+i+1, 1) ;
      _vel_vstu_vssl(vrsum_b1_i1, 4, pGIn+1*inDim+i+1, 1) ;
    }

    i+=2 ;
    break ;
  case 3:
    {
      __vr vrw_i0 = _vel_vldu_vssl(4, pWeight+(i  )*outDim, outDim) ;
      __vr vrw_i1 = _vel_vldu_vssl(4, pWeight+(i+1)*outDim, outDim) ;
      __vr vrw_i2 = _vel_vldu_vssl(4, pWeight+(i+2)*outDim, outDim) ;

      __vr vrsum_b0_i0 = _vel_vfsums_vvl(_vel_vfmuls_vvvl(vrw_i0, vro_b0, outDim), outDim) ;
      __vr vrsum_b1_i0 = _vel_vfsums_vvl(_vel_vfmuls_vvvl(vrw_i0, vro_b1, outDim), outDim) ;

      __vr vrsum_b0_i1 = _vel_vfsums_vvl(_vel_vfmuls_vvvl(vrw_i1, vro_b0, outDim), outDim) ;
      __vr vrsum_b1_i1 = _vel_vfsums_vvl(_vel_vfmuls_vvvl(vrw_i1, vro_b1, outDim), outDim) ;

      __vr vrsum_b0_i2 = _vel_vfsums_vvl(_vel_vfmuls_vvvl(vrw_i2, vro_b0, outDim), outDim) ;
      __vr vrsum_b1_i2 = _vel_vfsums_vvl(_vel_vfmuls_vvvl(vrw_i2, vro_b1, outDim), outDim) ;

      _vel_vstu_vssl(vrsum_b0_i0, 4, pGIn+0*inDim+i, 1) ;
      _vel_vstu_vssl(vrsum_b1_i0, 4, pGIn+1*inDim+i, 1) ;
      _vel_vstu_vssl(vrsum_b0_i1, 4, pGIn+0*inDim+i+1, 1) ;
      _vel_vstu_vssl(vrsum_b1_i1, 4, pGIn+1*inDim+i+1, 1) ;
      _vel_vstu_vssl(vrsum_b0_i2, 4, pGIn+0*inDim+i+2, 1) ;
      _vel_vstu_vssl(vrsum_b1_i2, 4, pGIn+1*inDim+i+2, 1) ;
    }
    i+=3 ;
    break ;
  default : break ;
  }
  for(; i<inDim; i+=4)
  {
    __vr vrw_i0 = _vel_vldu_vssl(4, pWeight+(i  )*outDim, outDim) ;
    __vr vrw_i1 = _vel_vldu_vssl(4, pWeight+(i+1)*outDim, outDim) ;
    __vr vrw_i2 = _vel_vldu_vssl(4, pWeight+(i+2)*outDim, outDim) ;
    __vr vrw_i3 = _vel_vldu_vssl(4, pWeight+(i+3)*outDim, outDim) ;

    __vr vrsum_b0_i0 = _vel_vfsums_vvl(_vel_vfmuls_vvvl(vrw_i0, vro_b0, outDim), outDim) ;
    __vr vrsum_b1_i0 = _vel_vfsums_vvl(_vel_vfmuls_vvvl(vrw_i0, vro_b1, outDim), outDim) ;

    __vr vrsum_b0_i1 = _vel_vfsums_vvl(_vel_vfmuls_vvvl(vrw_i1, vro_b0, outDim), outDim) ;
    __vr vrsum_b1_i1 = _vel_vfsums_vvl(_vel_vfmuls_vvvl(vrw_i1, vro_b1, outDim), outDim) ;

    __vr vrsum_b0_i2 = _vel_vfsums_vvl(_vel_vfmuls_vvvl(vrw_i2, vro_b0, outDim), outDim) ;
    __vr vrsum_b1_i2 = _vel_vfsums_vvl(_vel_vfmuls_vvvl(vrw_i2, vro_b1, outDim), outDim) ;

    __vr vrsum_b0_i3 = _vel_vfsums_vvl(_vel_vfmuls_vvvl(vrw_i3, vro_b0, outDim), outDim) ;
    __vr vrsum_b1_i3 = _vel_vfsums_vvl(_vel_vfmuls_vvvl(vrw_i3, vro_b1, outDim), outDim) ;

    _vel_vstu_vssl(vrsum_b0_i0, 4, pGIn+0*inDim+i, 1) ;
    _vel_vstu_vssl(vrsum_b1_i0, 4, pGIn+1*inDim+i, 1) ;
    _vel_vstu_vssl(vrsum_b0_i1, 4, pGIn+0*inDim+i+1, 1) ;
    _vel_vstu_vssl(vrsum_b1_i1, 4, pGIn+1*inDim+i+1, 1) ;
    _vel_vstu_vssl(vrsum_b0_i2, 4, pGIn+0*inDim+i+2, 1) ;
    _vel_vstu_vssl(vrsum_b1_i2, 4, pGIn+1*inDim+i+2, 1) ;
    _vel_vstu_vssl(vrsum_b0_i3, 4, pGIn+0*inDim+i+3, 1) ;
    _vel_vstu_vssl(vrsum_b1_i3, 4, pGIn+1*inDim+i+3, 1) ;
  }
}


static inline void b3(
  const uint64_t	inDim,
  const uint64_t	outDim,
  const float * 	pGOut,
  const float * 	pWeight,
  float * 		pGIn
)
{
  __vr vro_b0 = _vel_vldu_vssl(4, pGOut+0*outDim, outDim) ;
  __vr vro_b1 = _vel_vldu_vssl(4, pGOut+1*outDim, outDim) ;
  __vr vro_b2 = _vel_vldu_vssl(4, pGOut+2*outDim, outDim) ;

  int64_t i=0 ;
  switch(inDim&0x3) {
  case 1:
    {
      __vr vrw_i0 = _vel_vldu_vssl(4, pWeight+(i  )*outDim, outDim) ;

      __vr vrsum_b0_i0 = _vel_vfsums_vvl(_vel_vfmuls_vvvl(vrw_i0, vro_b0, outDim), outDim) ;
      __vr vrsum_b1_i0 = _vel_vfsums_vvl(_vel_vfmuls_vvvl(vrw_i0, vro_b1, outDim), outDim) ;
      __vr vrsum_b2_i0 = _vel_vfsums_vvl(_vel_vfmuls_vvvl(vrw_i0, vro_b2, outDim), outDim) ;

      _vel_vstu_vssl(vrsum_b0_i0, 4, pGIn+0*inDim+i, 1) ;
      _vel_vstu_vssl(vrsum_b1_i0, 4, pGIn+1*inDim+i, 1) ;
      _vel_vstu_vssl(vrsum_b2_i0, 4, pGIn+2*inDim+i, 1) ;
    }
    i+=1;
    break ;
  case 2 :
    {
      __vr vrw_i0 = _vel_vldu_vssl(4, pWeight+(i  )*outDim, outDim) ;
      __vr vrw_i1 = _vel_vldu_vssl(4, pWeight+(i+1)*outDim, outDim) ;

      __vr vrsum_b0_i0 = _vel_vfsums_vvl(_vel_vfmuls_vvvl(vrw_i0, vro_b0, outDim), outDim) ;
      __vr vrsum_b1_i0 = _vel_vfsums_vvl(_vel_vfmuls_vvvl(vrw_i0, vro_b1, outDim), outDim) ;
      __vr vrsum_b2_i0 = _vel_vfsums_vvl(_vel_vfmuls_vvvl(vrw_i0, vro_b2, outDim), outDim) ;

      __vr vrsum_b0_i1 = _vel_vfsums_vvl(_vel_vfmuls_vvvl(vrw_i1, vro_b0, outDim), outDim) ;
      __vr vrsum_b1_i1 = _vel_vfsums_vvl(_vel_vfmuls_vvvl(vrw_i1, vro_b1, outDim), outDim) ;
      __vr vrsum_b2_i1 = _vel_vfsums_vvl(_vel_vfmuls_vvvl(vrw_i1, vro_b2, outDim), outDim) ;

      _vel_vstu_vssl(vrsum_b0_i0, 4, pGIn+0*inDim+i, 1) ;
      _vel_vstu_vssl(vrsum_b1_i0, 4, pGIn+1*inDim+i, 1) ;
      _vel_vstu_vssl(vrsum_b2_i0, 4, pGIn+2*inDim+i, 1) ;
      _vel_vstu_vssl(vrsum_b0_i1, 4, pGIn+0*inDim+i+1, 1) ;
      _vel_vstu_vssl(vrsum_b1_i1, 4, pGIn+1*inDim+i+1, 1) ;
      _vel_vstu_vssl(vrsum_b2_i1, 4, pGIn+2*inDim+i+1, 1) ;
    }

    i+=2 ;
    break ;
  case 3:
    {
      __vr vrw_i0 = _vel_vldu_vssl(4, pWeight+(i  )*outDim, outDim) ;
      __vr vrw_i1 = _vel_vldu_vssl(4, pWeight+(i+1)*outDim, outDim) ;
      __vr vrw_i2 = _vel_vldu_vssl(4, pWeight+(i+2)*outDim, outDim) ;

      __vr vrsum_b0_i0 = _vel_vfsums_vvl(_vel_vfmuls_vvvl(vrw_i0, vro_b0, outDim), outDim) ;
      __vr vrsum_b1_i0 = _vel_vfsums_vvl(_vel_vfmuls_vvvl(vrw_i0, vro_b1, outDim), outDim) ;
      __vr vrsum_b2_i0 = _vel_vfsums_vvl(_vel_vfmuls_vvvl(vrw_i0, vro_b2, outDim), outDim) ;

      __vr vrsum_b0_i1 = _vel_vfsums_vvl(_vel_vfmuls_vvvl(vrw_i1, vro_b0, outDim), outDim) ;
      __vr vrsum_b1_i1 = _vel_vfsums_vvl(_vel_vfmuls_vvvl(vrw_i1, vro_b1, outDim), outDim) ;
      __vr vrsum_b2_i1 = _vel_vfsums_vvl(_vel_vfmuls_vvvl(vrw_i1, vro_b2, outDim), outDim) ;

      __vr vrsum_b0_i2 = _vel_vfsums_vvl(_vel_vfmuls_vvvl(vrw_i2, vro_b0, outDim), outDim) ;
      __vr vrsum_b1_i2 = _vel_vfsums_vvl(_vel_vfmuls_vvvl(vrw_i2, vro_b1, outDim), outDim) ;
      __vr vrsum_b2_i2 = _vel_vfsums_vvl(_vel_vfmuls_vvvl(vrw_i2, vro_b2, outDim), outDim) ;

      _vel_vstu_vssl(vrsum_b0_i0, 4, pGIn+0*inDim+i, 1) ;
      _vel_vstu_vssl(vrsum_b1_i0, 4, pGIn+1*inDim+i, 1) ;
      _vel_vstu_vssl(vrsum_b2_i0, 4, pGIn+2*inDim+i, 1) ;
      _vel_vstu_vssl(vrsum_b0_i1, 4, pGIn+0*inDim+i+1, 1) ;
      _vel_vstu_vssl(vrsum_b1_i1, 4, pGIn+1*inDim+i+1, 1) ;
      _vel_vstu_vssl(vrsum_b2_i1, 4, pGIn+2*inDim+i+1, 1) ;
      _vel_vstu_vssl(vrsum_b0_i2, 4, pGIn+0*inDim+i+2, 1) ;
      _vel_vstu_vssl(vrsum_b1_i2, 4, pGIn+1*inDim+i+2, 1) ;
      _vel_vstu_vssl(vrsum_b2_i2, 4, pGIn+2*inDim+i+2, 1) ;
    }
    i+=3 ;
    break ;
  default : break ;
  }
  for(; i<inDim; i+=4)
  {
    __vr vrw_i0 = _vel_vldu_vssl(4, pWeight+(i  )*outDim, outDim) ;
    __vr vrw_i1 = _vel_vldu_vssl(4, pWeight+(i+1)*outDim, outDim) ;
    __vr vrw_i2 = _vel_vldu_vssl(4, pWeight+(i+2)*outDim, outDim) ;
    __vr vrw_i3 = _vel_vldu_vssl(4, pWeight+(i+3)*outDim, outDim) ;

    __vr vrsum_b0_i0 = _vel_vfsums_vvl(_vel_vfmuls_vvvl(vrw_i0, vro_b0, outDim), outDim) ;
    __vr vrsum_b1_i0 = _vel_vfsums_vvl(_vel_vfmuls_vvvl(vrw_i0, vro_b1, outDim), outDim) ;
    __vr vrsum_b2_i0 = _vel_vfsums_vvl(_vel_vfmuls_vvvl(vrw_i0, vro_b2, outDim), outDim) ;

    __vr vrsum_b0_i1 = _vel_vfsums_vvl(_vel_vfmuls_vvvl(vrw_i1, vro_b0, outDim), outDim) ;
    __vr vrsum_b1_i1 = _vel_vfsums_vvl(_vel_vfmuls_vvvl(vrw_i1, vro_b1, outDim), outDim) ;
    __vr vrsum_b2_i1 = _vel_vfsums_vvl(_vel_vfmuls_vvvl(vrw_i1, vro_b2, outDim), outDim) ;

    __vr vrsum_b0_i2 = _vel_vfsums_vvl(_vel_vfmuls_vvvl(vrw_i2, vro_b0, outDim), outDim) ;
    __vr vrsum_b1_i2 = _vel_vfsums_vvl(_vel_vfmuls_vvvl(vrw_i2, vro_b1, outDim), outDim) ;
    __vr vrsum_b2_i2 = _vel_vfsums_vvl(_vel_vfmuls_vvvl(vrw_i2, vro_b2, outDim), outDim) ;

    __vr vrsum_b0_i3 = _vel_vfsums_vvl(_vel_vfmuls_vvvl(vrw_i3, vro_b0, outDim), outDim) ;
    __vr vrsum_b1_i3 = _vel_vfsums_vvl(_vel_vfmuls_vvvl(vrw_i3, vro_b1, outDim), outDim) ;
    __vr vrsum_b2_i3 = _vel_vfsums_vvl(_vel_vfmuls_vvvl(vrw_i3, vro_b2, outDim), outDim) ;

    _vel_vstu_vssl(vrsum_b0_i0, 4, pGIn+0*inDim+i, 1) ;
    _vel_vstu_vssl(vrsum_b1_i0, 4, pGIn+1*inDim+i, 1) ;
    _vel_vstu_vssl(vrsum_b2_i0, 4, pGIn+2*inDim+i, 1) ;
    _vel_vstu_vssl(vrsum_b0_i1, 4, pGIn+0*inDim+i+1, 1) ;
    _vel_vstu_vssl(vrsum_b1_i1, 4, pGIn+1*inDim+i+1, 1) ;
    _vel_vstu_vssl(vrsum_b2_i1, 4, pGIn+2*inDim+i+1, 1) ;
    _vel_vstu_vssl(vrsum_b0_i2, 4, pGIn+0*inDim+i+2, 1) ;
    _vel_vstu_vssl(vrsum_b1_i2, 4, pGIn+1*inDim+i+2, 1) ;
    _vel_vstu_vssl(vrsum_b2_i2, 4, pGIn+2*inDim+i+2, 1) ;
    _vel_vstu_vssl(vrsum_b0_i3, 4, pGIn+0*inDim+i+3, 1) ;
    _vel_vstu_vssl(vrsum_b1_i3, 4, pGIn+1*inDim+i+3, 1) ;
    _vel_vstu_vssl(vrsum_b2_i3, 4, pGIn+2*inDim+i+3, 1) ;
  }
}


static inline void b4(
  const uint64_t	inDim,
  const uint64_t	outDim,
  const float * 	pGOut,
  const float * 	pWeight,
  float * 		pGIn
)
{
  __vr vro_b0 = _vel_vldu_vssl(4, pGOut+0*outDim, outDim) ;
  __vr vro_b1 = _vel_vldu_vssl(4, pGOut+1*outDim, outDim) ;
  __vr vro_b2 = _vel_vldu_vssl(4, pGOut+2*outDim, outDim) ;
  __vr vro_b3 = _vel_vldu_vssl(4, pGOut+3*outDim, outDim) ;

  int64_t i=0 ;
  switch(inDim&0x3) {
  case 1:
    {
      __vr vrw_i0 = _vel_vldu_vssl(4, pWeight+(i  )*outDim, outDim) ;

      __vr vrsum_b0_i0 = _vel_vfsums_vvl(_vel_vfmuls_vvvl(vrw_i0, vro_b0, outDim), outDim) ;
      __vr vrsum_b1_i0 = _vel_vfsums_vvl(_vel_vfmuls_vvvl(vrw_i0, vro_b1, outDim), outDim) ;
      __vr vrsum_b2_i0 = _vel_vfsums_vvl(_vel_vfmuls_vvvl(vrw_i0, vro_b2, outDim), outDim) ;
      __vr vrsum_b3_i0 = _vel_vfsums_vvl(_vel_vfmuls_vvvl(vrw_i0, vro_b3, outDim), outDim) ;

      _vel_vstu_vssl(vrsum_b0_i0, 4, pGIn+0*inDim+i, 1) ;
      _vel_vstu_vssl(vrsum_b1_i0, 4, pGIn+1*inDim+i, 1) ;
      _vel_vstu_vssl(vrsum_b2_i0, 4, pGIn+2*inDim+i, 1) ;
      _vel_vstu_vssl(vrsum_b3_i0, 4, pGIn+3*inDim+i, 1) ;
    }
    i+=1;
    break ;
  case 2 :
    {
      __vr vrw_i0 = _vel_vldu_vssl(4, pWeight+(i  )*outDim, outDim) ;
      __vr vrw_i1 = _vel_vldu_vssl(4, pWeight+(i+1)*outDim, outDim) ;

      __vr vrsum_b0_i0 = _vel_vfsums_vvl(_vel_vfmuls_vvvl(vrw_i0, vro_b0, outDim), outDim) ;
      __vr vrsum_b1_i0 = _vel_vfsums_vvl(_vel_vfmuls_vvvl(vrw_i0, vro_b1, outDim), outDim) ;
      __vr vrsum_b2_i0 = _vel_vfsums_vvl(_vel_vfmuls_vvvl(vrw_i0, vro_b2, outDim), outDim) ;
      __vr vrsum_b3_i0 = _vel_vfsums_vvl(_vel_vfmuls_vvvl(vrw_i0, vro_b3, outDim), outDim) ;

      __vr vrsum_b0_i1 = _vel_vfsums_vvl(_vel_vfmuls_vvvl(vrw_i1, vro_b0, outDim), outDim) ;
      __vr vrsum_b1_i1 = _vel_vfsums_vvl(_vel_vfmuls_vvvl(vrw_i1, vro_b1, outDim), outDim) ;
      __vr vrsum_b2_i1 = _vel_vfsums_vvl(_vel_vfmuls_vvvl(vrw_i1, vro_b2, outDim), outDim) ;
      __vr vrsum_b3_i1 = _vel_vfsums_vvl(_vel_vfmuls_vvvl(vrw_i1, vro_b3, outDim), outDim) ;

      _vel_vstu_vssl(vrsum_b0_i0, 4, pGIn+0*inDim+i, 1) ;
      _vel_vstu_vssl(vrsum_b1_i0, 4, pGIn+1*inDim+i, 1) ;
      _vel_vstu_vssl(vrsum_b2_i0, 4, pGIn+2*inDim+i, 1) ;
      _vel_vstu_vssl(vrsum_b3_i0, 4, pGIn+3*inDim+i, 1) ;
      _vel_vstu_vssl(vrsum_b0_i1, 4, pGIn+0*inDim+i+1, 1) ;
      _vel_vstu_vssl(vrsum_b1_i1, 4, pGIn+1*inDim+i+1, 1) ;
      _vel_vstu_vssl(vrsum_b2_i1, 4, pGIn+2*inDim+i+1, 1) ;
      _vel_vstu_vssl(vrsum_b3_i1, 4, pGIn+3*inDim+i+1, 1) ;
    }

    i+=2 ;
    break ;
  case 3:
    {
      __vr vrw_i0 = _vel_vldu_vssl(4, pWeight+(i  )*outDim, outDim) ;
      __vr vrw_i1 = _vel_vldu_vssl(4, pWeight+(i+1)*outDim, outDim) ;
      __vr vrw_i2 = _vel_vldu_vssl(4, pWeight+(i+2)*outDim, outDim) ;

      __vr vrsum_b0_i0 = _vel_vfsums_vvl(_vel_vfmuls_vvvl(vrw_i0, vro_b0, outDim), outDim) ;
      __vr vrsum_b1_i0 = _vel_vfsums_vvl(_vel_vfmuls_vvvl(vrw_i0, vro_b1, outDim), outDim) ;
      __vr vrsum_b2_i0 = _vel_vfsums_vvl(_vel_vfmuls_vvvl(vrw_i0, vro_b2, outDim), outDim) ;
      __vr vrsum_b3_i0 = _vel_vfsums_vvl(_vel_vfmuls_vvvl(vrw_i0, vro_b3, outDim), outDim) ;

      __vr vrsum_b0_i1 = _vel_vfsums_vvl(_vel_vfmuls_vvvl(vrw_i1, vro_b0, outDim), outDim) ;
      __vr vrsum_b1_i1 = _vel_vfsums_vvl(_vel_vfmuls_vvvl(vrw_i1, vro_b1, outDim), outDim) ;
      __vr vrsum_b2_i1 = _vel_vfsums_vvl(_vel_vfmuls_vvvl(vrw_i1, vro_b2, outDim), outDim) ;
      __vr vrsum_b3_i1 = _vel_vfsums_vvl(_vel_vfmuls_vvvl(vrw_i1, vro_b3, outDim), outDim) ;

      __vr vrsum_b0_i2 = _vel_vfsums_vvl(_vel_vfmuls_vvvl(vrw_i2, vro_b0, outDim), outDim) ;
      __vr vrsum_b1_i2 = _vel_vfsums_vvl(_vel_vfmuls_vvvl(vrw_i2, vro_b1, outDim), outDim) ;
      __vr vrsum_b2_i2 = _vel_vfsums_vvl(_vel_vfmuls_vvvl(vrw_i2, vro_b2, outDim), outDim) ;
      __vr vrsum_b3_i2 = _vel_vfsums_vvl(_vel_vfmuls_vvvl(vrw_i2, vro_b3, outDim), outDim) ;

      _vel_vstu_vssl(vrsum_b0_i0, 4, pGIn+0*inDim+i, 1) ;
      _vel_vstu_vssl(vrsum_b1_i0, 4, pGIn+1*inDim+i, 1) ;
      _vel_vstu_vssl(vrsum_b2_i0, 4, pGIn+2*inDim+i, 1) ;
      _vel_vstu_vssl(vrsum_b3_i0, 4, pGIn+3*inDim+i, 1) ;
      _vel_vstu_vssl(vrsum_b0_i1, 4, pGIn+0*inDim+i+1, 1) ;
      _vel_vstu_vssl(vrsum_b1_i1, 4, pGIn+1*inDim+i+1, 1) ;
      _vel_vstu_vssl(vrsum_b2_i1, 4, pGIn+2*inDim+i+1, 1) ;
      _vel_vstu_vssl(vrsum_b3_i1, 4, pGIn+3*inDim+i+1, 1) ;
      _vel_vstu_vssl(vrsum_b0_i2, 4, pGIn+0*inDim+i+2, 1) ;
      _vel_vstu_vssl(vrsum_b1_i2, 4, pGIn+1*inDim+i+2, 1) ;
      _vel_vstu_vssl(vrsum_b2_i2, 4, pGIn+2*inDim+i+2, 1) ;
      _vel_vstu_vssl(vrsum_b3_i2, 4, pGIn+3*inDim+i+2, 1) ;
    }
    i+=3 ;
    break ;
  default : break ;
  }
  for(; i<inDim; i+=4)
  {
    __vr vrw_i0 = _vel_vldu_vssl(4, pWeight+(i  )*outDim, outDim) ;
    __vr vrw_i1 = _vel_vldu_vssl(4, pWeight+(i+1)*outDim, outDim) ;
    __vr vrw_i2 = _vel_vldu_vssl(4, pWeight+(i+2)*outDim, outDim) ;
    __vr vrw_i3 = _vel_vldu_vssl(4, pWeight+(i+3)*outDim, outDim) ;

    __vr vrsum_b0_i0 = _vel_vfsums_vvl(_vel_vfmuls_vvvl(vrw_i0, vro_b0, outDim), outDim) ;
    __vr vrsum_b1_i0 = _vel_vfsums_vvl(_vel_vfmuls_vvvl(vrw_i0, vro_b1, outDim), outDim) ;
    __vr vrsum_b2_i0 = _vel_vfsums_vvl(_vel_vfmuls_vvvl(vrw_i0, vro_b2, outDim), outDim) ;
    __vr vrsum_b3_i0 = _vel_vfsums_vvl(_vel_vfmuls_vvvl(vrw_i0, vro_b3, outDim), outDim) ;

    __vr vrsum_b0_i1 = _vel_vfsums_vvl(_vel_vfmuls_vvvl(vrw_i1, vro_b0, outDim), outDim) ;
    __vr vrsum_b1_i1 = _vel_vfsums_vvl(_vel_vfmuls_vvvl(vrw_i1, vro_b1, outDim), outDim) ;
    __vr vrsum_b2_i1 = _vel_vfsums_vvl(_vel_vfmuls_vvvl(vrw_i1, vro_b2, outDim), outDim) ;
    __vr vrsum_b3_i1 = _vel_vfsums_vvl(_vel_vfmuls_vvvl(vrw_i1, vro_b3, outDim), outDim) ;

    __vr vrsum_b0_i2 = _vel_vfsums_vvl(_vel_vfmuls_vvvl(vrw_i2, vro_b0, outDim), outDim) ;
    __vr vrsum_b1_i2 = _vel_vfsums_vvl(_vel_vfmuls_vvvl(vrw_i2, vro_b1, outDim), outDim) ;
    __vr vrsum_b2_i2 = _vel_vfsums_vvl(_vel_vfmuls_vvvl(vrw_i2, vro_b2, outDim), outDim) ;
    __vr vrsum_b3_i2 = _vel_vfsums_vvl(_vel_vfmuls_vvvl(vrw_i2, vro_b3, outDim), outDim) ;

    __vr vrsum_b0_i3 = _vel_vfsums_vvl(_vel_vfmuls_vvvl(vrw_i3, vro_b0, outDim), outDim) ;
    __vr vrsum_b1_i3 = _vel_vfsums_vvl(_vel_vfmuls_vvvl(vrw_i3, vro_b1, outDim), outDim) ;
    __vr vrsum_b2_i3 = _vel_vfsums_vvl(_vel_vfmuls_vvvl(vrw_i3, vro_b2, outDim), outDim) ;
    __vr vrsum_b3_i3 = _vel_vfsums_vvl(_vel_vfmuls_vvvl(vrw_i3, vro_b3, outDim), outDim) ;

    _vel_vstu_vssl(vrsum_b0_i0, 4, pGIn+0*inDim+i, 1) ;
    _vel_vstu_vssl(vrsum_b1_i0, 4, pGIn+1*inDim+i, 1) ;
    _vel_vstu_vssl(vrsum_b2_i0, 4, pGIn+2*inDim+i, 1) ;
    _vel_vstu_vssl(vrsum_b3_i0, 4, pGIn+3*inDim+i, 1) ;
    _vel_vstu_vssl(vrsum_b0_i1, 4, pGIn+0*inDim+i+1, 1) ;
    _vel_vstu_vssl(vrsum_b1_i1, 4, pGIn+1*inDim+i+1, 1) ;
    _vel_vstu_vssl(vrsum_b2_i1, 4, pGIn+2*inDim+i+1, 1) ;
    _vel_vstu_vssl(vrsum_b3_i1, 4, pGIn+3*inDim+i+1, 1) ;
    _vel_vstu_vssl(vrsum_b0_i2, 4, pGIn+0*inDim+i+2, 1) ;
    _vel_vstu_vssl(vrsum_b1_i2, 4, pGIn+1*inDim+i+2, 1) ;
    _vel_vstu_vssl(vrsum_b2_i2, 4, pGIn+2*inDim+i+2, 1) ;
    _vel_vstu_vssl(vrsum_b3_i2, 4, pGIn+3*inDim+i+2, 1) ;
    _vel_vstu_vssl(vrsum_b0_i3, 4, pGIn+0*inDim+i+3, 1) ;
    _vel_vstu_vssl(vrsum_b1_i3, 4, pGIn+1*inDim+i+3, 1) ;
    _vel_vstu_vssl(vrsum_b2_i3, 4, pGIn+2*inDim+i+3, 1) ;
    _vel_vstu_vssl(vrsum_b3_i3, 4, pGIn+3*inDim+i+3, 1) ;
  }
}

vednnError_t vednnLinearBackwardData_oU256(
    const uint64_t			inDim,
    const uint64_t			outDim,
    const uint64_t			nBatch,
    const void * restrict		pDataGradOut,
    const void * restrict		pDataWeight,
    void * restrict			pDataGradIn
)
{
  const float * restrict pGOut   = (const float * restrict) pDataGradOut;
  const float * restrict pWeight = (const float * restrict) pDataWeight;
  float * restrict const pGIn    = (float * restrict const) pDataGradIn;

  int64_t n=0;
  int64_t batchRemain = nBatch % 4 ;

  switch( batchRemain ) {
  case 1:
    b1(inDim, outDim,
       pGOut+n*outDim, pWeight, pGIn+n*inDim) ;
    n+=1 ;
    break ;
  case 2:
    b2(inDim, outDim,
       pGOut+n*outDim, pWeight, pGIn+n*inDim) ;
    n+=2 ;
    break ;
  case 3:
    b3(inDim, outDim,
       pGOut+n*outDim, pWeight, pGIn+n*inDim) ;
    n+=3;
    break ;
  default : break ;
  }
  for(; n<nBatch; n+=4) {
    b4(inDim, outDim,
       pGOut+n*outDim, pWeight, pGIn+n*inDim) ;
  }

  return VEDNN_SUCCESS ;
}


