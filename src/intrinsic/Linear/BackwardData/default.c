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
  int64_t i=0 ;
  switch(inDim&0x3) {
  case 1:
    {
      __vr vrsum_b0 = _vel_vbrds_vsl(0.f, VLEN) ;

      for(int64_t o=0; o<outDim; o+=VLEN) {
	const int64_t vl = outDim-o < VLEN ? outDim - o : VLEN ;

	__vr vrw    = _vel_vldu_vssl(4, pWeight+i*outDim+o, vl) ;
	__vr vro_b0 = _vel_vldu_vssl(4, pGOut+0*outDim+o, vl) ;

	vrsum_b0 = _vel_vfmads_vvvvvl(vrsum_b0, vrw, vro_b0, vrsum_b0, vl) ;
      }

      vrsum_b0 = _vel_vfsums_vvl(vrsum_b0, VLEN);

      _vel_vstu_vssl(vrsum_b0, 4, pGIn+0*inDim+i, 1) ;
    }
    i+=1;
    break ;
  case 2 :
    {
      __vr vrsum_b0_i0 = _vel_vbrds_vsl(0.f, VLEN) ;
      __vr vrsum_b0_i1 = _vel_vbrds_vsl(0.f, VLEN) ;

      for(int64_t o=0; o<outDim; o+=VLEN) {
	const int64_t vl = outDim-o < VLEN ? outDim - o : VLEN ;

	__vr vrw_i0 = _vel_vldu_vssl(4, pWeight+(i  )*outDim+o, vl) ;
	__vr vrw_i1 = _vel_vldu_vssl(4, pWeight+(i+1)*outDim+o, vl) ;

	__vr vro_b0 = _vel_vldu_vssl(4, pGOut+0*outDim+o, vl) ;

	vrsum_b0_i0 = _vel_vfmads_vvvvvl(vrsum_b0_i0, vrw_i0, vro_b0, vrsum_b0_i0, vl) ;
	vrsum_b0_i1 = _vel_vfmads_vvvvvl(vrsum_b0_i1, vrw_i1, vro_b0, vrsum_b0_i1, vl) ;
      }

      vrsum_b0_i0 = _vel_vfsums_vvl(vrsum_b0_i0, VLEN);
      vrsum_b0_i1 = _vel_vfsums_vvl(vrsum_b0_i1, VLEN);

      _vel_vstu_vssl(vrsum_b0_i0, 4, pGIn+0*inDim+i,   1) ;
      _vel_vstu_vssl(vrsum_b0_i1, 4, pGIn+0*inDim+i+1, 1) ;
    }

    i+=2 ;
    break ;
  case 3:
    {
      __vr vrsum_b0_i0 = _vel_vbrds_vsl(0.f, VLEN) ;
      __vr vrsum_b0_i1 = _vel_vbrds_vsl(0.f, VLEN) ;
      __vr vrsum_b0_i2 = _vel_vbrds_vsl(0.f, VLEN) ;

      for(int64_t o=0; o<outDim; o+=VLEN) {
	const int64_t vl = outDim-o < VLEN ? outDim - o : VLEN ;

	__vr vrw_i0 = _vel_vldu_vssl(4, pWeight+(i  )*outDim+o, vl) ;
	__vr vrw_i1 = _vel_vldu_vssl(4, pWeight+(i+1)*outDim+o, vl) ;
	__vr vrw_i2 = _vel_vldu_vssl(4, pWeight+(i+2)*outDim+o, vl) ;

	__vr vro_b0 = _vel_vldu_vssl(4, pGOut+0*outDim+o, vl) ;

	vrsum_b0_i0 = _vel_vfmads_vvvvvl(vrsum_b0_i0, vrw_i0, vro_b0, vrsum_b0_i0, vl) ;
	vrsum_b0_i1 = _vel_vfmads_vvvvvl(vrsum_b0_i1, vrw_i1, vro_b0, vrsum_b0_i1, vl) ;
	vrsum_b0_i2 = _vel_vfmads_vvvvvl(vrsum_b0_i2, vrw_i2, vro_b0, vrsum_b0_i2, vl) ;
      }

      vrsum_b0_i0 = _vel_vfsums_vvl(vrsum_b0_i0, VLEN);
      vrsum_b0_i1 = _vel_vfsums_vvl(vrsum_b0_i1, VLEN);
      vrsum_b0_i2 = _vel_vfsums_vvl(vrsum_b0_i2, VLEN);

      _vel_vstu_vssl(vrsum_b0_i0, 4, pGIn+0*inDim+i,   1) ;
      _vel_vstu_vssl(vrsum_b0_i1, 4, pGIn+0*inDim+i+1, 1) ;
      _vel_vstu_vssl(vrsum_b0_i2, 4, pGIn+0*inDim+i+2, 1) ;

    }
    i+=3 ;
    break ;
  default : break ;
  }
  for(; i<inDim; i+=4)
  {
    __vr vrsum_b0_i0 = _vel_vbrds_vsl(0.f, VLEN) ;
    __vr vrsum_b0_i1 = _vel_vbrds_vsl(0.f, VLEN) ;
    __vr vrsum_b0_i2 = _vel_vbrds_vsl(0.f, VLEN) ;
    __vr vrsum_b0_i3 = _vel_vbrds_vsl(0.f, VLEN) ;

    for(int64_t o=0; o<outDim; o+=VLEN) {
      const int64_t vl = outDim-o < VLEN ? outDim - o : VLEN ;

      __vr vrw_i0 = _vel_vldu_vssl(4, pWeight+(i  )*outDim+o, vl) ;
      __vr vrw_i1 = _vel_vldu_vssl(4, pWeight+(i+1)*outDim+o, vl) ;
      __vr vrw_i2 = _vel_vldu_vssl(4, pWeight+(i+2)*outDim+o, vl) ;
      __vr vrw_i3 = _vel_vldu_vssl(4, pWeight+(i+3)*outDim+o, vl) ;

      __vr vro_b0 = _vel_vldu_vssl(4, pGOut+0*outDim+o, vl) ;

      vrsum_b0_i0 = _vel_vfmads_vvvvvl(vrsum_b0_i0, vrw_i0, vro_b0, vrsum_b0_i0, vl) ;
      vrsum_b0_i1 = _vel_vfmads_vvvvvl(vrsum_b0_i1, vrw_i1, vro_b0, vrsum_b0_i1, vl) ;
      vrsum_b0_i2 = _vel_vfmads_vvvvvl(vrsum_b0_i2, vrw_i2, vro_b0, vrsum_b0_i2, vl) ;
      vrsum_b0_i3 = _vel_vfmads_vvvvvl(vrsum_b0_i3, vrw_i3, vro_b0, vrsum_b0_i3, vl) ;
    }

    vrsum_b0_i0 = _vel_vfsums_vvl(vrsum_b0_i0, VLEN);
    vrsum_b0_i1 = _vel_vfsums_vvl(vrsum_b0_i1, VLEN);
    vrsum_b0_i2 = _vel_vfsums_vvl(vrsum_b0_i2, VLEN);
    vrsum_b0_i3 = _vel_vfsums_vvl(vrsum_b0_i3, VLEN);

    _vel_vstu_vssl(vrsum_b0_i0, 4, pGIn+0*inDim+i,   1) ;
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
  int64_t i=0 ;
  switch(inDim&0x3) {
  case 1:
    {
      __vr vrsum_b0 = _vel_vbrds_vsl(0.f, VLEN) ;
      __vr vrsum_b1 = _vel_vbrds_vsl(0.f, VLEN) ;

      for(int64_t o=0; o<outDim; o+=VLEN) {
	const int64_t vl = outDim-o < VLEN ? outDim - o : VLEN ;

	__vr vrw    = _vel_vldu_vssl(4, pWeight+i*outDim+o, vl) ;
	__vr vro_b0 = _vel_vldu_vssl(4, pGOut+0*outDim+o, vl) ;
	__vr vro_b1 = _vel_vldu_vssl(4, pGOut+1*outDim+o, vl) ;

	vrsum_b0 = _vel_vfmads_vvvvvl(vrsum_b0, vrw, vro_b0, vrsum_b0, vl) ;
	vrsum_b1 = _vel_vfmads_vvvvvl(vrsum_b1, vrw, vro_b1, vrsum_b1, vl) ;
      }

      vrsum_b0 = _vel_vfsums_vvl(vrsum_b0, VLEN);
      vrsum_b1 = _vel_vfsums_vvl(vrsum_b1, VLEN);

      _vel_vstu_vssl(vrsum_b0, 4, pGIn+0*inDim+i, 1) ;
      _vel_vstu_vssl(vrsum_b1, 4, pGIn+1*inDim+i, 1) ;
    }
    i+=1;
    break ;
  case 2 :
    {
      __vr vrsum_b0_i0 = _vel_vbrds_vsl(0.f, VLEN) ;
      __vr vrsum_b1_i0 = _vel_vbrds_vsl(0.f, VLEN) ;
      __vr vrsum_b0_i1 = _vel_vbrds_vsl(0.f, VLEN) ;
      __vr vrsum_b1_i1 = _vel_vbrds_vsl(0.f, VLEN) ;

      for(int64_t o=0; o<outDim; o+=VLEN) {
	const int64_t vl = outDim-o < VLEN ? outDim - o : VLEN ;

	__vr vrw_i0 = _vel_vldu_vssl(4, pWeight+(i  )*outDim+o, vl) ;
	__vr vrw_i1 = _vel_vldu_vssl(4, pWeight+(i+1)*outDim+o, vl) ;

	__vr vro_b0 = _vel_vldu_vssl(4, pGOut+0*outDim+o, vl) ;
	__vr vro_b1 = _vel_vldu_vssl(4, pGOut+1*outDim+o, vl) ;

	vrsum_b0_i0 = _vel_vfmads_vvvvvl(vrsum_b0_i0, vrw_i0, vro_b0, vrsum_b0_i0, vl) ;
	vrsum_b1_i0 = _vel_vfmads_vvvvvl(vrsum_b1_i0, vrw_i0, vro_b1, vrsum_b1_i0, vl) ;

	vrsum_b0_i1 = _vel_vfmads_vvvvvl(vrsum_b0_i1, vrw_i1, vro_b0, vrsum_b0_i1, vl) ;
	vrsum_b1_i1 = _vel_vfmads_vvvvvl(vrsum_b1_i1, vrw_i1, vro_b1, vrsum_b1_i1, vl) ;
      }

      vrsum_b0_i0 = _vel_vfsums_vvl(vrsum_b0_i0, VLEN);
      vrsum_b1_i0 = _vel_vfsums_vvl(vrsum_b1_i0, VLEN);
      vrsum_b0_i1 = _vel_vfsums_vvl(vrsum_b0_i1, VLEN);
      vrsum_b1_i1 = _vel_vfsums_vvl(vrsum_b1_i1, VLEN);

      _vel_vstu_vssl(vrsum_b0_i0, 4, pGIn+0*inDim+i,   1) ;
      _vel_vstu_vssl(vrsum_b1_i0, 4, pGIn+1*inDim+i,   1) ;
      _vel_vstu_vssl(vrsum_b0_i1, 4, pGIn+0*inDim+i+1, 1) ;
      _vel_vstu_vssl(vrsum_b1_i1, 4, pGIn+1*inDim+i+1, 1) ;
    }

    i+=2 ;
    break ;
  case 3:
    {
      __vr vrsum_b0_i0 = _vel_vbrds_vsl(0.f, VLEN) ;
      __vr vrsum_b1_i0 = _vel_vbrds_vsl(0.f, VLEN) ;
      __vr vrsum_b0_i1 = _vel_vbrds_vsl(0.f, VLEN) ;
      __vr vrsum_b1_i1 = _vel_vbrds_vsl(0.f, VLEN) ;
      __vr vrsum_b0_i2 = _vel_vbrds_vsl(0.f, VLEN) ;
      __vr vrsum_b1_i2 = _vel_vbrds_vsl(0.f, VLEN) ;

      for(int64_t o=0; o<outDim; o+=VLEN) {
	const int64_t vl = outDim-o < VLEN ? outDim - o : VLEN ;

	__vr vrw_i0 = _vel_vldu_vssl(4, pWeight+(i  )*outDim+o, vl) ;
	__vr vrw_i1 = _vel_vldu_vssl(4, pWeight+(i+1)*outDim+o, vl) ;
	__vr vrw_i2 = _vel_vldu_vssl(4, pWeight+(i+2)*outDim+o, vl) ;

	__vr vro_b0 = _vel_vldu_vssl(4, pGOut+0*outDim+o, vl) ;
	__vr vro_b1 = _vel_vldu_vssl(4, pGOut+1*outDim+o, vl) ;

	vrsum_b0_i0 = _vel_vfmads_vvvvvl(vrsum_b0_i0, vrw_i0, vro_b0, vrsum_b0_i0, vl) ;
	vrsum_b1_i0 = _vel_vfmads_vvvvvl(vrsum_b1_i0, vrw_i0, vro_b1, vrsum_b1_i0, vl) ;

	vrsum_b0_i1 = _vel_vfmads_vvvvvl(vrsum_b0_i1, vrw_i1, vro_b0, vrsum_b0_i1, vl) ;
	vrsum_b1_i1 = _vel_vfmads_vvvvvl(vrsum_b1_i1, vrw_i1, vro_b1, vrsum_b1_i1, vl) ;

	vrsum_b0_i2 = _vel_vfmads_vvvvvl(vrsum_b0_i2, vrw_i2, vro_b0, vrsum_b0_i2, vl) ;
	vrsum_b1_i2 = _vel_vfmads_vvvvvl(vrsum_b1_i2, vrw_i2, vro_b1, vrsum_b1_i2, vl) ;
      }

      vrsum_b0_i0 = _vel_vfsums_vvl(vrsum_b0_i0, VLEN);
      vrsum_b1_i0 = _vel_vfsums_vvl(vrsum_b1_i0, VLEN);
      vrsum_b0_i1 = _vel_vfsums_vvl(vrsum_b0_i1, VLEN);
      vrsum_b1_i1 = _vel_vfsums_vvl(vrsum_b1_i1, VLEN);
      vrsum_b0_i2 = _vel_vfsums_vvl(vrsum_b0_i2, VLEN);
      vrsum_b1_i2 = _vel_vfsums_vvl(vrsum_b1_i2, VLEN);

      _vel_vstu_vssl(vrsum_b0_i0, 4, pGIn+0*inDim+i,   1) ;
      _vel_vstu_vssl(vrsum_b1_i0, 4, pGIn+1*inDim+i,   1) ;
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
    __vr vrsum_b0_i0 = _vel_vbrds_vsl(0.f, VLEN) ;
    __vr vrsum_b1_i0 = _vel_vbrds_vsl(0.f, VLEN) ;
    __vr vrsum_b0_i1 = _vel_vbrds_vsl(0.f, VLEN) ;
    __vr vrsum_b1_i1 = _vel_vbrds_vsl(0.f, VLEN) ;
    __vr vrsum_b0_i2 = _vel_vbrds_vsl(0.f, VLEN) ;
    __vr vrsum_b1_i2 = _vel_vbrds_vsl(0.f, VLEN) ;
    __vr vrsum_b0_i3 = _vel_vbrds_vsl(0.f, VLEN) ;
    __vr vrsum_b1_i3 = _vel_vbrds_vsl(0.f, VLEN) ;

    for(int64_t o=0; o<outDim; o+=VLEN) {
      const int64_t vl = outDim-o < VLEN ? outDim - o : VLEN ;

      __vr vrw_i0 = _vel_vldu_vssl(4, pWeight+(i  )*outDim+o, vl) ;
      __vr vrw_i1 = _vel_vldu_vssl(4, pWeight+(i+1)*outDim+o, vl) ;
      __vr vrw_i2 = _vel_vldu_vssl(4, pWeight+(i+2)*outDim+o, vl) ;
      __vr vrw_i3 = _vel_vldu_vssl(4, pWeight+(i+3)*outDim+o, vl) ;

      __vr vro_b0 = _vel_vldu_vssl(4, pGOut+0*outDim+o, vl) ;
      __vr vro_b1 = _vel_vldu_vssl(4, pGOut+1*outDim+o, vl) ;

      vrsum_b0_i0 = _vel_vfmads_vvvvvl(vrsum_b0_i0, vrw_i0, vro_b0, vrsum_b0_i0, vl) ;
      vrsum_b1_i0 = _vel_vfmads_vvvvvl(vrsum_b1_i0, vrw_i0, vro_b1, vrsum_b1_i0, vl) ;

      vrsum_b0_i1 = _vel_vfmads_vvvvvl(vrsum_b0_i1, vrw_i1, vro_b0, vrsum_b0_i1, vl) ;
      vrsum_b1_i1 = _vel_vfmads_vvvvvl(vrsum_b1_i1, vrw_i1, vro_b1, vrsum_b1_i1, vl) ;

      vrsum_b0_i2 = _vel_vfmads_vvvvvl(vrsum_b0_i2, vrw_i2, vro_b0, vrsum_b0_i2, vl) ;
      vrsum_b1_i2 = _vel_vfmads_vvvvvl(vrsum_b1_i2, vrw_i2, vro_b1, vrsum_b1_i2, vl) ;

      vrsum_b0_i3 = _vel_vfmads_vvvvvl(vrsum_b0_i3, vrw_i3, vro_b0, vrsum_b0_i3, vl) ;
      vrsum_b1_i3 = _vel_vfmads_vvvvvl(vrsum_b1_i3, vrw_i3, vro_b1, vrsum_b1_i3, vl) ;
    }

    vrsum_b0_i0 = _vel_vfsums_vvl(vrsum_b0_i0, VLEN);
    vrsum_b1_i0 = _vel_vfsums_vvl(vrsum_b1_i0, VLEN);
    vrsum_b0_i1 = _vel_vfsums_vvl(vrsum_b0_i1, VLEN);
    vrsum_b1_i1 = _vel_vfsums_vvl(vrsum_b1_i1, VLEN);
    vrsum_b0_i2 = _vel_vfsums_vvl(vrsum_b0_i2, VLEN);
    vrsum_b1_i2 = _vel_vfsums_vvl(vrsum_b1_i2, VLEN);
    vrsum_b0_i3 = _vel_vfsums_vvl(vrsum_b0_i3, VLEN);
    vrsum_b1_i3 = _vel_vfsums_vvl(vrsum_b1_i3, VLEN);

    _vel_vstu_vssl(vrsum_b0_i0, 4, pGIn+0*inDim+i,   1) ;
    _vel_vstu_vssl(vrsum_b1_i0, 4, pGIn+1*inDim+i,   1) ;
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
  int64_t i=0 ;
  switch(inDim&0x3) {
  case 1:
    {
      __vr vrsum_b0 = _vel_vbrds_vsl(0.f, VLEN) ;
      __vr vrsum_b1 = _vel_vbrds_vsl(0.f, VLEN) ;
      __vr vrsum_b2 = _vel_vbrds_vsl(0.f, VLEN) ;

      for(int64_t o=0; o<outDim; o+=VLEN) {
	const int64_t vl = outDim-o < VLEN ? outDim - o : VLEN ;

	__vr vrw    = _vel_vldu_vssl(4, pWeight+i*outDim+o, vl) ;
	__vr vro_b0 = _vel_vldu_vssl(4, pGOut+0*outDim+o, vl) ;
	__vr vro_b1 = _vel_vldu_vssl(4, pGOut+1*outDim+o, vl) ;
	__vr vro_b2 = _vel_vldu_vssl(4, pGOut+2*outDim+o, vl) ;

	vrsum_b0 = _vel_vfmads_vvvvvl(vrsum_b0, vrw, vro_b0, vrsum_b0, vl) ;
	vrsum_b1 = _vel_vfmads_vvvvvl(vrsum_b1, vrw, vro_b1, vrsum_b1, vl) ;
	vrsum_b2 = _vel_vfmads_vvvvvl(vrsum_b2, vrw, vro_b2, vrsum_b2, vl) ;
      }

      vrsum_b0 = _vel_vfsums_vvl(vrsum_b0, VLEN);
      vrsum_b1 = _vel_vfsums_vvl(vrsum_b1, VLEN);
      vrsum_b2 = _vel_vfsums_vvl(vrsum_b2, VLEN);

      _vel_vstu_vssl(vrsum_b0, 4, pGIn+0*inDim+i, 1) ;
      _vel_vstu_vssl(vrsum_b1, 4, pGIn+1*inDim+i, 1) ;
      _vel_vstu_vssl(vrsum_b2, 4, pGIn+2*inDim+i, 1) ;
    }
    i+=1;
    break ;
  case 2 :
    {
      __vr vrsum_b0_i0 = _vel_vbrds_vsl(0.f, VLEN) ;
      __vr vrsum_b1_i0 = _vel_vbrds_vsl(0.f, VLEN) ;
      __vr vrsum_b2_i0 = _vel_vbrds_vsl(0.f, VLEN) ;
      __vr vrsum_b0_i1 = _vel_vbrds_vsl(0.f, VLEN) ;
      __vr vrsum_b1_i1 = _vel_vbrds_vsl(0.f, VLEN) ;
      __vr vrsum_b2_i1 = _vel_vbrds_vsl(0.f, VLEN) ;

      for(int64_t o=0; o<outDim; o+=VLEN) {
	const int64_t vl = outDim-o < VLEN ? outDim - o : VLEN ;

	__vr vrw_i0 = _vel_vldu_vssl(4, pWeight+(i  )*outDim+o, vl) ;
	__vr vrw_i1 = _vel_vldu_vssl(4, pWeight+(i+1)*outDim+o, vl) ;

	__vr vro_b0 = _vel_vldu_vssl(4, pGOut+0*outDim+o, vl) ;
	__vr vro_b1 = _vel_vldu_vssl(4, pGOut+1*outDim+o, vl) ;
	__vr vro_b2 = _vel_vldu_vssl(4, pGOut+2*outDim+o, vl) ;

	vrsum_b0_i0 = _vel_vfmads_vvvvvl(vrsum_b0_i0, vrw_i0, vro_b0, vrsum_b0_i0, vl) ;
	vrsum_b1_i0 = _vel_vfmads_vvvvvl(vrsum_b1_i0, vrw_i0, vro_b1, vrsum_b1_i0, vl) ;
	vrsum_b2_i0 = _vel_vfmads_vvvvvl(vrsum_b2_i0, vrw_i0, vro_b2, vrsum_b2_i0, vl) ;

	vrsum_b0_i1 = _vel_vfmads_vvvvvl(vrsum_b0_i1, vrw_i1, vro_b0, vrsum_b0_i1, vl) ;
	vrsum_b1_i1 = _vel_vfmads_vvvvvl(vrsum_b1_i1, vrw_i1, vro_b1, vrsum_b1_i1, vl) ;
	vrsum_b2_i1 = _vel_vfmads_vvvvvl(vrsum_b2_i1, vrw_i1, vro_b2, vrsum_b2_i1, vl) ;
      }

      vrsum_b0_i0 = _vel_vfsums_vvl(vrsum_b0_i0, VLEN);
      vrsum_b1_i0 = _vel_vfsums_vvl(vrsum_b1_i0, VLEN);
      vrsum_b2_i0 = _vel_vfsums_vvl(vrsum_b2_i0, VLEN);
      vrsum_b0_i1 = _vel_vfsums_vvl(vrsum_b0_i1, VLEN);
      vrsum_b1_i1 = _vel_vfsums_vvl(vrsum_b1_i1, VLEN);
      vrsum_b2_i1 = _vel_vfsums_vvl(vrsum_b2_i1, VLEN);

      _vel_vstu_vssl(vrsum_b0_i0, 4, pGIn+0*inDim+i,   1) ;
      _vel_vstu_vssl(vrsum_b1_i0, 4, pGIn+1*inDim+i,   1) ;
      _vel_vstu_vssl(vrsum_b2_i0, 4, pGIn+2*inDim+i,   1) ;
      _vel_vstu_vssl(vrsum_b0_i1, 4, pGIn+0*inDim+i+1, 1) ;
      _vel_vstu_vssl(vrsum_b1_i1, 4, pGIn+1*inDim+i+1, 1) ;
      _vel_vstu_vssl(vrsum_b2_i1, 4, pGIn+2*inDim+i+1, 1) ;
    }

    i+=2 ;
    break ;
  case 3:
    {
      __vr vrsum_b0_i0 = _vel_vbrds_vsl(0.f, VLEN) ;
      __vr vrsum_b1_i0 = _vel_vbrds_vsl(0.f, VLEN) ;
      __vr vrsum_b2_i0 = _vel_vbrds_vsl(0.f, VLEN) ;
      __vr vrsum_b0_i1 = _vel_vbrds_vsl(0.f, VLEN) ;
      __vr vrsum_b1_i1 = _vel_vbrds_vsl(0.f, VLEN) ;
      __vr vrsum_b2_i1 = _vel_vbrds_vsl(0.f, VLEN) ;
      __vr vrsum_b0_i2 = _vel_vbrds_vsl(0.f, VLEN) ;
      __vr vrsum_b1_i2 = _vel_vbrds_vsl(0.f, VLEN) ;
      __vr vrsum_b2_i2 = _vel_vbrds_vsl(0.f, VLEN) ;

      for(int64_t o=0; o<outDim; o+=VLEN) {
	const int64_t vl = outDim-o < VLEN ? outDim - o : VLEN ;

	__vr vrw_i0 = _vel_vldu_vssl(4, pWeight+(i  )*outDim+o, vl) ;
	__vr vrw_i1 = _vel_vldu_vssl(4, pWeight+(i+1)*outDim+o, vl) ;
	__vr vrw_i2 = _vel_vldu_vssl(4, pWeight+(i+2)*outDim+o, vl) ;

	__vr vro_b0 = _vel_vldu_vssl(4, pGOut+0*outDim+o, vl) ;
	__vr vro_b1 = _vel_vldu_vssl(4, pGOut+1*outDim+o, vl) ;
	__vr vro_b2 = _vel_vldu_vssl(4, pGOut+2*outDim+o, vl) ;

	vrsum_b0_i0 = _vel_vfmads_vvvvvl(vrsum_b0_i0, vrw_i0, vro_b0, vrsum_b0_i0, vl) ;
	vrsum_b1_i0 = _vel_vfmads_vvvvvl(vrsum_b1_i0, vrw_i0, vro_b1, vrsum_b1_i0, vl) ;
	vrsum_b2_i0 = _vel_vfmads_vvvvvl(vrsum_b2_i0, vrw_i0, vro_b2, vrsum_b2_i0, vl) ;

	vrsum_b0_i1 = _vel_vfmads_vvvvvl(vrsum_b0_i1, vrw_i1, vro_b0, vrsum_b0_i1, vl) ;
	vrsum_b1_i1 = _vel_vfmads_vvvvvl(vrsum_b1_i1, vrw_i1, vro_b1, vrsum_b1_i1, vl) ;
	vrsum_b2_i1 = _vel_vfmads_vvvvvl(vrsum_b2_i1, vrw_i1, vro_b2, vrsum_b2_i1, vl) ;

	vrsum_b0_i2 = _vel_vfmads_vvvvvl(vrsum_b0_i2, vrw_i2, vro_b0, vrsum_b0_i2, vl) ;
	vrsum_b1_i2 = _vel_vfmads_vvvvvl(vrsum_b1_i2, vrw_i2, vro_b1, vrsum_b1_i2, vl) ;
	vrsum_b2_i2 = _vel_vfmads_vvvvvl(vrsum_b2_i2, vrw_i2, vro_b2, vrsum_b2_i2, vl) ;
      }

      vrsum_b0_i0 = _vel_vfsums_vvl(vrsum_b0_i0, VLEN);
      vrsum_b1_i0 = _vel_vfsums_vvl(vrsum_b1_i0, VLEN);
      vrsum_b2_i0 = _vel_vfsums_vvl(vrsum_b2_i0, VLEN);
      vrsum_b0_i1 = _vel_vfsums_vvl(vrsum_b0_i1, VLEN);
      vrsum_b1_i1 = _vel_vfsums_vvl(vrsum_b1_i1, VLEN);
      vrsum_b2_i1 = _vel_vfsums_vvl(vrsum_b2_i1, VLEN);
      vrsum_b0_i2 = _vel_vfsums_vvl(vrsum_b0_i2, VLEN);
      vrsum_b1_i2 = _vel_vfsums_vvl(vrsum_b1_i2, VLEN);
      vrsum_b2_i2 = _vel_vfsums_vvl(vrsum_b2_i2, VLEN);

      _vel_vstu_vssl(vrsum_b0_i0, 4, pGIn+0*inDim+i,   1) ;
      _vel_vstu_vssl(vrsum_b1_i0, 4, pGIn+1*inDim+i,   1) ;
      _vel_vstu_vssl(vrsum_b2_i0, 4, pGIn+2*inDim+i,   1) ;
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
    __vr vrsum_b0_i0 = _vel_vbrds_vsl(0.f, VLEN) ;
    __vr vrsum_b1_i0 = _vel_vbrds_vsl(0.f, VLEN) ;
    __vr vrsum_b2_i0 = _vel_vbrds_vsl(0.f, VLEN) ;
    __vr vrsum_b0_i1 = _vel_vbrds_vsl(0.f, VLEN) ;
    __vr vrsum_b1_i1 = _vel_vbrds_vsl(0.f, VLEN) ;
    __vr vrsum_b2_i1 = _vel_vbrds_vsl(0.f, VLEN) ;
    __vr vrsum_b0_i2 = _vel_vbrds_vsl(0.f, VLEN) ;
    __vr vrsum_b1_i2 = _vel_vbrds_vsl(0.f, VLEN) ;
    __vr vrsum_b2_i2 = _vel_vbrds_vsl(0.f, VLEN) ;
    __vr vrsum_b0_i3 = _vel_vbrds_vsl(0.f, VLEN) ;
    __vr vrsum_b1_i3 = _vel_vbrds_vsl(0.f, VLEN) ;
    __vr vrsum_b2_i3 = _vel_vbrds_vsl(0.f, VLEN) ;

    for(int64_t o=0; o<outDim; o+=VLEN) {
      const int64_t vl = outDim-o < VLEN ? outDim - o : VLEN ;

      __vr vrw_i0 = _vel_vldu_vssl(4, pWeight+(i  )*outDim+o, vl) ;
      __vr vrw_i1 = _vel_vldu_vssl(4, pWeight+(i+1)*outDim+o, vl) ;
      __vr vrw_i2 = _vel_vldu_vssl(4, pWeight+(i+2)*outDim+o, vl) ;
      __vr vrw_i3 = _vel_vldu_vssl(4, pWeight+(i+3)*outDim+o, vl) ;

      __vr vro_b0 = _vel_vldu_vssl(4, pGOut+0*outDim+o, vl) ;
      __vr vro_b1 = _vel_vldu_vssl(4, pGOut+1*outDim+o, vl) ;
      __vr vro_b2 = _vel_vldu_vssl(4, pGOut+2*outDim+o, vl) ;

      vrsum_b0_i0 = _vel_vfmads_vvvvvl(vrsum_b0_i0, vrw_i0, vro_b0, vrsum_b0_i0, vl) ;
      vrsum_b1_i0 = _vel_vfmads_vvvvvl(vrsum_b1_i0, vrw_i0, vro_b1, vrsum_b1_i0, vl) ;
      vrsum_b2_i0 = _vel_vfmads_vvvvvl(vrsum_b2_i0, vrw_i0, vro_b2, vrsum_b2_i0, vl) ;

      vrsum_b0_i1 = _vel_vfmads_vvvvvl(vrsum_b0_i1, vrw_i1, vro_b0, vrsum_b0_i1, vl) ;
      vrsum_b1_i1 = _vel_vfmads_vvvvvl(vrsum_b1_i1, vrw_i1, vro_b1, vrsum_b1_i1, vl) ;
      vrsum_b2_i1 = _vel_vfmads_vvvvvl(vrsum_b2_i1, vrw_i1, vro_b2, vrsum_b2_i1, vl) ;

      vrsum_b0_i2 = _vel_vfmads_vvvvvl(vrsum_b0_i2, vrw_i2, vro_b0, vrsum_b0_i2, vl) ;
      vrsum_b1_i2 = _vel_vfmads_vvvvvl(vrsum_b1_i2, vrw_i2, vro_b1, vrsum_b1_i2, vl) ;
      vrsum_b2_i2 = _vel_vfmads_vvvvvl(vrsum_b2_i2, vrw_i2, vro_b2, vrsum_b2_i2, vl) ;

      vrsum_b0_i3 = _vel_vfmads_vvvvvl(vrsum_b0_i3, vrw_i3, vro_b0, vrsum_b0_i3, vl) ;
      vrsum_b1_i3 = _vel_vfmads_vvvvvl(vrsum_b1_i3, vrw_i3, vro_b1, vrsum_b1_i3, vl) ;
      vrsum_b2_i3 = _vel_vfmads_vvvvvl(vrsum_b2_i3, vrw_i3, vro_b2, vrsum_b2_i3, vl) ;
    }

    vrsum_b0_i0 = _vel_vfsums_vvl(vrsum_b0_i0, VLEN);
    vrsum_b1_i0 = _vel_vfsums_vvl(vrsum_b1_i0, VLEN);
    vrsum_b2_i0 = _vel_vfsums_vvl(vrsum_b2_i0, VLEN);
    vrsum_b0_i1 = _vel_vfsums_vvl(vrsum_b0_i1, VLEN);
    vrsum_b1_i1 = _vel_vfsums_vvl(vrsum_b1_i1, VLEN);
    vrsum_b2_i1 = _vel_vfsums_vvl(vrsum_b2_i1, VLEN);
    vrsum_b0_i2 = _vel_vfsums_vvl(vrsum_b0_i2, VLEN);
    vrsum_b1_i2 = _vel_vfsums_vvl(vrsum_b1_i2, VLEN);
    vrsum_b2_i2 = _vel_vfsums_vvl(vrsum_b2_i2, VLEN);
    vrsum_b0_i3 = _vel_vfsums_vvl(vrsum_b0_i3, VLEN);
    vrsum_b1_i3 = _vel_vfsums_vvl(vrsum_b1_i3, VLEN);
    vrsum_b2_i3 = _vel_vfsums_vvl(vrsum_b2_i3, VLEN);

    _vel_vstu_vssl(vrsum_b0_i0, 4, pGIn+0*inDim+i,   1) ;
    _vel_vstu_vssl(vrsum_b1_i0, 4, pGIn+1*inDim+i,   1) ;
    _vel_vstu_vssl(vrsum_b2_i0, 4, pGIn+2*inDim+i,   1) ;
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
  int64_t i=0 ;
  switch(inDim&0x3) {
  case 1:
    {
      __vr vrsum_b0 = _vel_vbrds_vsl(0.f, VLEN) ;
      __vr vrsum_b1 = _vel_vbrds_vsl(0.f, VLEN) ;
      __vr vrsum_b2 = _vel_vbrds_vsl(0.f, VLEN) ;
      __vr vrsum_b3 = _vel_vbrds_vsl(0.f, VLEN) ;

      for(int64_t o=0; o<outDim; o+=VLEN) {
	const int64_t vl = outDim-o < VLEN ? outDim - o : VLEN ;

	__vr vrw    = _vel_vldu_vssl(4, pWeight+i*outDim+o, vl) ;
	__vr vro_b0 = _vel_vldu_vssl(4, pGOut+0*outDim+o, vl) ;
	__vr vro_b1 = _vel_vldu_vssl(4, pGOut+1*outDim+o, vl) ;
	__vr vro_b2 = _vel_vldu_vssl(4, pGOut+2*outDim+o, vl) ;
	__vr vro_b3 = _vel_vldu_vssl(4, pGOut+3*outDim+o, vl) ;

	vrsum_b0 = _vel_vfmads_vvvvvl(vrsum_b0, vrw, vro_b0, vrsum_b0, vl) ;
	vrsum_b1 = _vel_vfmads_vvvvvl(vrsum_b1, vrw, vro_b1, vrsum_b1, vl) ;
	vrsum_b2 = _vel_vfmads_vvvvvl(vrsum_b2, vrw, vro_b2, vrsum_b2, vl) ;
	vrsum_b3 = _vel_vfmads_vvvvvl(vrsum_b3, vrw, vro_b3, vrsum_b3, vl) ;
      }

      vrsum_b0 = _vel_vfsums_vvl(vrsum_b0, VLEN);
      vrsum_b1 = _vel_vfsums_vvl(vrsum_b1, VLEN);
      vrsum_b2 = _vel_vfsums_vvl(vrsum_b2, VLEN);
      vrsum_b3 = _vel_vfsums_vvl(vrsum_b3, VLEN);

      _vel_vstu_vssl(vrsum_b0, 4, pGIn+0*inDim+i, 1) ;
      _vel_vstu_vssl(vrsum_b1, 4, pGIn+1*inDim+i, 1) ;
      _vel_vstu_vssl(vrsum_b2, 4, pGIn+2*inDim+i, 1) ;
      _vel_vstu_vssl(vrsum_b3, 4, pGIn+3*inDim+i, 1) ;
    }
    i+=1;
    break ;
  case 2 :
    {
      __vr vrsum_b0_i0 = _vel_vbrds_vsl(0.f, VLEN) ;
      __vr vrsum_b1_i0 = _vel_vbrds_vsl(0.f, VLEN) ;
      __vr vrsum_b2_i0 = _vel_vbrds_vsl(0.f, VLEN) ;
      __vr vrsum_b3_i0 = _vel_vbrds_vsl(0.f, VLEN) ;
      __vr vrsum_b0_i1 = _vel_vbrds_vsl(0.f, VLEN) ;
      __vr vrsum_b1_i1 = _vel_vbrds_vsl(0.f, VLEN) ;
      __vr vrsum_b2_i1 = _vel_vbrds_vsl(0.f, VLEN) ;
      __vr vrsum_b3_i1 = _vel_vbrds_vsl(0.f, VLEN) ;

      for(int64_t o=0; o<outDim; o+=VLEN) {
	const int64_t vl = outDim-o < VLEN ? outDim - o : VLEN ;

	__vr vrw_i0 = _vel_vldu_vssl(4, pWeight+(i  )*outDim+o, vl) ;
	__vr vrw_i1 = _vel_vldu_vssl(4, pWeight+(i+1)*outDim+o, vl) ;

	__vr vro_b0 = _vel_vldu_vssl(4, pGOut+0*outDim+o, vl) ;
	__vr vro_b1 = _vel_vldu_vssl(4, pGOut+1*outDim+o, vl) ;
	__vr vro_b2 = _vel_vldu_vssl(4, pGOut+2*outDim+o, vl) ;
	__vr vro_b3 = _vel_vldu_vssl(4, pGOut+3*outDim+o, vl) ;

	vrsum_b0_i0 = _vel_vfmads_vvvvvl(vrsum_b0_i0, vrw_i0, vro_b0, vrsum_b0_i0, vl) ;
	vrsum_b1_i0 = _vel_vfmads_vvvvvl(vrsum_b1_i0, vrw_i0, vro_b1, vrsum_b1_i0, vl) ;
	vrsum_b2_i0 = _vel_vfmads_vvvvvl(vrsum_b2_i0, vrw_i0, vro_b2, vrsum_b2_i0, vl) ;
	vrsum_b3_i0 = _vel_vfmads_vvvvvl(vrsum_b3_i0, vrw_i0, vro_b3, vrsum_b3_i0, vl) ;

	vrsum_b0_i1 = _vel_vfmads_vvvvvl(vrsum_b0_i1, vrw_i1, vro_b0, vrsum_b0_i1, vl) ;
	vrsum_b1_i1 = _vel_vfmads_vvvvvl(vrsum_b1_i1, vrw_i1, vro_b1, vrsum_b1_i1, vl) ;
	vrsum_b2_i1 = _vel_vfmads_vvvvvl(vrsum_b2_i1, vrw_i1, vro_b2, vrsum_b2_i1, vl) ;
	vrsum_b3_i1 = _vel_vfmads_vvvvvl(vrsum_b3_i1, vrw_i1, vro_b3, vrsum_b3_i1, vl) ;
      }

      vrsum_b0_i0 = _vel_vfsums_vvl(vrsum_b0_i0, VLEN);
      vrsum_b1_i0 = _vel_vfsums_vvl(vrsum_b1_i0, VLEN);
      vrsum_b2_i0 = _vel_vfsums_vvl(vrsum_b2_i0, VLEN);
      vrsum_b3_i0 = _vel_vfsums_vvl(vrsum_b3_i0, VLEN);
      vrsum_b0_i1 = _vel_vfsums_vvl(vrsum_b0_i1, VLEN);
      vrsum_b1_i1 = _vel_vfsums_vvl(vrsum_b1_i1, VLEN);
      vrsum_b2_i1 = _vel_vfsums_vvl(vrsum_b2_i1, VLEN);
      vrsum_b3_i1 = _vel_vfsums_vvl(vrsum_b3_i1, VLEN);

      _vel_vstu_vssl(vrsum_b0_i0, 4, pGIn+0*inDim+i,   1) ;
      _vel_vstu_vssl(vrsum_b1_i0, 4, pGIn+1*inDim+i,   1) ;
      _vel_vstu_vssl(vrsum_b2_i0, 4, pGIn+2*inDim+i,   1) ;
      _vel_vstu_vssl(vrsum_b3_i0, 4, pGIn+3*inDim+i,   1) ;
      _vel_vstu_vssl(vrsum_b0_i1, 4, pGIn+0*inDim+i+1, 1) ;
      _vel_vstu_vssl(vrsum_b1_i1, 4, pGIn+1*inDim+i+1, 1) ;
      _vel_vstu_vssl(vrsum_b2_i1, 4, pGIn+2*inDim+i+1, 1) ;
      _vel_vstu_vssl(vrsum_b3_i1, 4, pGIn+3*inDim+i+1, 1) ;
    }

    i+=2 ;
    break ;
  case 3:
    {
      __vr vrsum_b0_i0 = _vel_vbrds_vsl(0.f, VLEN) ;
      __vr vrsum_b1_i0 = _vel_vbrds_vsl(0.f, VLEN) ;
      __vr vrsum_b2_i0 = _vel_vbrds_vsl(0.f, VLEN) ;
      __vr vrsum_b3_i0 = _vel_vbrds_vsl(0.f, VLEN) ;
      __vr vrsum_b0_i1 = _vel_vbrds_vsl(0.f, VLEN) ;
      __vr vrsum_b1_i1 = _vel_vbrds_vsl(0.f, VLEN) ;
      __vr vrsum_b2_i1 = _vel_vbrds_vsl(0.f, VLEN) ;
      __vr vrsum_b3_i1 = _vel_vbrds_vsl(0.f, VLEN) ;
      __vr vrsum_b0_i2 = _vel_vbrds_vsl(0.f, VLEN) ;
      __vr vrsum_b1_i2 = _vel_vbrds_vsl(0.f, VLEN) ;
      __vr vrsum_b2_i2 = _vel_vbrds_vsl(0.f, VLEN) ;
      __vr vrsum_b3_i2 = _vel_vbrds_vsl(0.f, VLEN) ;

      for(int64_t o=0; o<outDim; o+=VLEN) {
	const int64_t vl = outDim-o < VLEN ? outDim - o : VLEN ;

	__vr vrw_i0 = _vel_vldu_vssl(4, pWeight+(i  )*outDim+o, vl) ;
	__vr vrw_i1 = _vel_vldu_vssl(4, pWeight+(i+1)*outDim+o, vl) ;
	__vr vrw_i2 = _vel_vldu_vssl(4, pWeight+(i+2)*outDim+o, vl) ;

	__vr vro_b0 = _vel_vldu_vssl(4, pGOut+0*outDim+o, vl) ;
	__vr vro_b1 = _vel_vldu_vssl(4, pGOut+1*outDim+o, vl) ;
	__vr vro_b2 = _vel_vldu_vssl(4, pGOut+2*outDim+o, vl) ;
	__vr vro_b3 = _vel_vldu_vssl(4, pGOut+3*outDim+o, vl) ;

	vrsum_b0_i0 = _vel_vfmads_vvvvvl(vrsum_b0_i0, vrw_i0, vro_b0, vrsum_b0_i0, vl) ;
	vrsum_b1_i0 = _vel_vfmads_vvvvvl(vrsum_b1_i0, vrw_i0, vro_b1, vrsum_b1_i0, vl) ;
	vrsum_b2_i0 = _vel_vfmads_vvvvvl(vrsum_b2_i0, vrw_i0, vro_b2, vrsum_b2_i0, vl) ;
	vrsum_b3_i0 = _vel_vfmads_vvvvvl(vrsum_b3_i0, vrw_i0, vro_b3, vrsum_b3_i0, vl) ;

	vrsum_b0_i1 = _vel_vfmads_vvvvvl(vrsum_b0_i1, vrw_i1, vro_b0, vrsum_b0_i1, vl) ;
	vrsum_b1_i1 = _vel_vfmads_vvvvvl(vrsum_b1_i1, vrw_i1, vro_b1, vrsum_b1_i1, vl) ;
	vrsum_b2_i1 = _vel_vfmads_vvvvvl(vrsum_b2_i1, vrw_i1, vro_b2, vrsum_b2_i1, vl) ;
	vrsum_b3_i1 = _vel_vfmads_vvvvvl(vrsum_b3_i1, vrw_i1, vro_b3, vrsum_b3_i1, vl) ;

	vrsum_b0_i2 = _vel_vfmads_vvvvvl(vrsum_b0_i2, vrw_i2, vro_b0, vrsum_b0_i2, vl) ;
	vrsum_b1_i2 = _vel_vfmads_vvvvvl(vrsum_b1_i2, vrw_i2, vro_b1, vrsum_b1_i2, vl) ;
	vrsum_b2_i2 = _vel_vfmads_vvvvvl(vrsum_b2_i2, vrw_i2, vro_b2, vrsum_b2_i2, vl) ;
	vrsum_b3_i2 = _vel_vfmads_vvvvvl(vrsum_b3_i2, vrw_i2, vro_b3, vrsum_b3_i2, vl) ;
      }

      vrsum_b0_i0 = _vel_vfsums_vvl(vrsum_b0_i0, VLEN);
      vrsum_b1_i0 = _vel_vfsums_vvl(vrsum_b1_i0, VLEN);
      vrsum_b2_i0 = _vel_vfsums_vvl(vrsum_b2_i0, VLEN);
      vrsum_b3_i0 = _vel_vfsums_vvl(vrsum_b3_i0, VLEN);
      vrsum_b0_i1 = _vel_vfsums_vvl(vrsum_b0_i1, VLEN);
      vrsum_b1_i1 = _vel_vfsums_vvl(vrsum_b1_i1, VLEN);
      vrsum_b2_i1 = _vel_vfsums_vvl(vrsum_b2_i1, VLEN);
      vrsum_b3_i1 = _vel_vfsums_vvl(vrsum_b3_i1, VLEN);
      vrsum_b0_i2 = _vel_vfsums_vvl(vrsum_b0_i2, VLEN);
      vrsum_b1_i2 = _vel_vfsums_vvl(vrsum_b1_i2, VLEN);
      vrsum_b2_i2 = _vel_vfsums_vvl(vrsum_b2_i2, VLEN);
      vrsum_b3_i2 = _vel_vfsums_vvl(vrsum_b3_i2, VLEN);

      _vel_vstu_vssl(vrsum_b0_i0, 4, pGIn+0*inDim+i,   1) ;
      _vel_vstu_vssl(vrsum_b1_i0, 4, pGIn+1*inDim+i,   1) ;
      _vel_vstu_vssl(vrsum_b2_i0, 4, pGIn+2*inDim+i,   1) ;
      _vel_vstu_vssl(vrsum_b3_i0, 4, pGIn+3*inDim+i,   1) ;
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
    __vr vrsum_b0_i0 = _vel_vbrds_vsl(0.f, VLEN) ;
    __vr vrsum_b1_i0 = _vel_vbrds_vsl(0.f, VLEN) ;
    __vr vrsum_b2_i0 = _vel_vbrds_vsl(0.f, VLEN) ;
    __vr vrsum_b3_i0 = _vel_vbrds_vsl(0.f, VLEN) ;
    __vr vrsum_b0_i1 = _vel_vbrds_vsl(0.f, VLEN) ;
    __vr vrsum_b1_i1 = _vel_vbrds_vsl(0.f, VLEN) ;
    __vr vrsum_b2_i1 = _vel_vbrds_vsl(0.f, VLEN) ;
    __vr vrsum_b3_i1 = _vel_vbrds_vsl(0.f, VLEN) ;
    __vr vrsum_b0_i2 = _vel_vbrds_vsl(0.f, VLEN) ;
    __vr vrsum_b1_i2 = _vel_vbrds_vsl(0.f, VLEN) ;
    __vr vrsum_b2_i2 = _vel_vbrds_vsl(0.f, VLEN) ;
    __vr vrsum_b3_i2 = _vel_vbrds_vsl(0.f, VLEN) ;
    __vr vrsum_b0_i3 = _vel_vbrds_vsl(0.f, VLEN) ;
    __vr vrsum_b1_i3 = _vel_vbrds_vsl(0.f, VLEN) ;
    __vr vrsum_b2_i3 = _vel_vbrds_vsl(0.f, VLEN) ;
    __vr vrsum_b3_i3 = _vel_vbrds_vsl(0.f, VLEN) ;

    for(int64_t o=0; o<outDim; o+=VLEN) {
      const int64_t vl = outDim-o < VLEN ? outDim - o : VLEN ;

      __vr vrw_i0 = _vel_vldu_vssl(4, pWeight+(i  )*outDim+o, vl) ;
      __vr vrw_i1 = _vel_vldu_vssl(4, pWeight+(i+1)*outDim+o, vl) ;
      __vr vrw_i2 = _vel_vldu_vssl(4, pWeight+(i+2)*outDim+o, vl) ;
      __vr vrw_i3 = _vel_vldu_vssl(4, pWeight+(i+3)*outDim+o, vl) ;

      __vr vro_b0 = _vel_vldu_vssl(4, pGOut+0*outDim+o, vl) ;
      __vr vro_b1 = _vel_vldu_vssl(4, pGOut+1*outDim+o, vl) ;
      __vr vro_b2 = _vel_vldu_vssl(4, pGOut+2*outDim+o, vl) ;
      __vr vro_b3 = _vel_vldu_vssl(4, pGOut+3*outDim+o, vl) ;

      vrsum_b0_i0 = _vel_vfmads_vvvvvl(vrsum_b0_i0, vrw_i0, vro_b0, vrsum_b0_i0, vl) ;
      vrsum_b1_i0 = _vel_vfmads_vvvvvl(vrsum_b1_i0, vrw_i0, vro_b1, vrsum_b1_i0, vl) ;
      vrsum_b2_i0 = _vel_vfmads_vvvvvl(vrsum_b2_i0, vrw_i0, vro_b2, vrsum_b2_i0, vl) ;
      vrsum_b3_i0 = _vel_vfmads_vvvvvl(vrsum_b3_i0, vrw_i0, vro_b3, vrsum_b3_i0, vl) ;

      vrsum_b0_i1 = _vel_vfmads_vvvvvl(vrsum_b0_i1, vrw_i1, vro_b0, vrsum_b0_i1, vl) ;
      vrsum_b1_i1 = _vel_vfmads_vvvvvl(vrsum_b1_i1, vrw_i1, vro_b1, vrsum_b1_i1, vl) ;
      vrsum_b2_i1 = _vel_vfmads_vvvvvl(vrsum_b2_i1, vrw_i1, vro_b2, vrsum_b2_i1, vl) ;
      vrsum_b3_i1 = _vel_vfmads_vvvvvl(vrsum_b3_i1, vrw_i1, vro_b3, vrsum_b3_i1, vl) ;

      vrsum_b0_i2 = _vel_vfmads_vvvvvl(vrsum_b0_i2, vrw_i2, vro_b0, vrsum_b0_i2, vl) ;
      vrsum_b1_i2 = _vel_vfmads_vvvvvl(vrsum_b1_i2, vrw_i2, vro_b1, vrsum_b1_i2, vl) ;
      vrsum_b2_i2 = _vel_vfmads_vvvvvl(vrsum_b2_i2, vrw_i2, vro_b2, vrsum_b2_i2, vl) ;
      vrsum_b3_i2 = _vel_vfmads_vvvvvl(vrsum_b3_i2, vrw_i2, vro_b3, vrsum_b3_i2, vl) ;

      vrsum_b0_i3 = _vel_vfmads_vvvvvl(vrsum_b0_i3, vrw_i3, vro_b0, vrsum_b0_i3, vl) ;
      vrsum_b1_i3 = _vel_vfmads_vvvvvl(vrsum_b1_i3, vrw_i3, vro_b1, vrsum_b1_i3, vl) ;
      vrsum_b2_i3 = _vel_vfmads_vvvvvl(vrsum_b2_i3, vrw_i3, vro_b2, vrsum_b2_i3, vl) ;
      vrsum_b3_i3 = _vel_vfmads_vvvvvl(vrsum_b3_i3, vrw_i3, vro_b3, vrsum_b3_i3, vl) ;
    }

    vrsum_b0_i0 = _vel_vfsums_vvl(vrsum_b0_i0, VLEN);
    vrsum_b1_i0 = _vel_vfsums_vvl(vrsum_b1_i0, VLEN);
    vrsum_b2_i0 = _vel_vfsums_vvl(vrsum_b2_i0, VLEN);
    vrsum_b3_i0 = _vel_vfsums_vvl(vrsum_b3_i0, VLEN);
    vrsum_b0_i1 = _vel_vfsums_vvl(vrsum_b0_i1, VLEN);
    vrsum_b1_i1 = _vel_vfsums_vvl(vrsum_b1_i1, VLEN);
    vrsum_b2_i1 = _vel_vfsums_vvl(vrsum_b2_i1, VLEN);
    vrsum_b3_i1 = _vel_vfsums_vvl(vrsum_b3_i1, VLEN);
    vrsum_b0_i2 = _vel_vfsums_vvl(vrsum_b0_i2, VLEN);
    vrsum_b1_i2 = _vel_vfsums_vvl(vrsum_b1_i2, VLEN);
    vrsum_b2_i2 = _vel_vfsums_vvl(vrsum_b2_i2, VLEN);
    vrsum_b3_i2 = _vel_vfsums_vvl(vrsum_b3_i2, VLEN);
    vrsum_b0_i3 = _vel_vfsums_vvl(vrsum_b0_i3, VLEN);
    vrsum_b1_i3 = _vel_vfsums_vvl(vrsum_b1_i3, VLEN);
    vrsum_b2_i3 = _vel_vfsums_vvl(vrsum_b2_i3, VLEN);
    vrsum_b3_i3 = _vel_vfsums_vvl(vrsum_b3_i3, VLEN);

    _vel_vstu_vssl(vrsum_b0_i0, 4, pGIn+0*inDim+i,   1) ;
    _vel_vstu_vssl(vrsum_b1_i0, 4, pGIn+1*inDim+i,   1) ;
    _vel_vstu_vssl(vrsum_b2_i0, 4, pGIn+2*inDim+i,   1) ;
    _vel_vstu_vssl(vrsum_b3_i0, 4, pGIn+3*inDim+i,   1) ;
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

vednnError_t vednnLinearBackwardData_default(
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
       pGOut+n*outDim, pWeight, pGIn+n*inDim ) ;
    n+=1 ;
    break ;
  case 2:
    b2(inDim, outDim,
       pGOut+n*outDim, pWeight, pGIn+n*inDim ) ;
    n+=2 ;
    break ;
  case 3:
    b3(inDim, outDim,
       pGOut+n*outDim, pWeight, pGIn+n*inDim ) ;
    n+=3;
    break ;
  default : break ;
  }
  for(; n<nBatch; n+=4) {
    b4(inDim, outDim,
       pGOut+n*outDim, pWeight, pGIn+n*inDim ) ;
  }

  return VEDNN_SUCCESS ;
}


#if 0 // reference code
vednnError_t vednnLinearBackwardData_default(
    const uint64_t			inDim,
    const uint64_t			outDim,
    const uint64_t			nBatch,
    const void * restrict		pDataGradOut,
    const void * restrict		pDataWeight,
    void * restrict			pDataGradIn
)
{
  const float * restrict pGOut   = pDataGradOut;
  const float * restrict pWeight = pDataWeight;
  float * restrict const pGIn    = pDataGradIn;

  for(int64_t n=0; n<nBatch; n++) {
    for(int64_t i=0; i<inDim; i++ ) {
      float sum = 0.f ;
      for(int64_t o=0; o<outDim; o++) {
	sum += pWeight[i*outDim+o] * pGOut[n*outDim+o] ;
      }
      pGIn[n*inDim+i] = sum ;
    }
  }

  return VEDNN_SUCCESS ;
}
#endif


