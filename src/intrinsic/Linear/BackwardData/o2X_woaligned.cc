#include <stdint.h>
#include <float.h>

#include <stdio.h>

#include "vednn.h"

#include "velintrin.h"

template<int BATCH>
static inline void func(
  const uint64_t	inDim,
  const uint64_t	outDim,
  const float * 	pGOut,
  const float * 	pWeight,
  float * 		pGIn,
  const uint64_t        mvl
)
{
  int64_t i=0 ;
#if 0
  switch(inDim&0x3) {
  case 1:
    {

      __vr vrsum_i0[BATCH] ;
#pragma clang loop unroll(full)
      for(int64_t b=0; b<BATCH; b++) {
        vrsum_i0[b] = _vel_pvbrd_vsl(0UL, VLEN) ;
      }

      int64_t o=0 ;
      for(; o+2*VLEN<outDim; o+=2*VLEN)
      {
        const int64_t vl = VLEN ;

        __vr vrw_i0 = _vel_vld_vssl(8, pWeight+(i  )*outDim+o, vl) ;

        __vr vro[BATCH] ;
#pragma clang loop unroll(full)
        for(int64_t b=0; b<BATCH; b++) {
          vro[b] = _vel_vld_vssl(8, pGOut+b*outDim+o, vl) ;
        }

#pragma clang loop unroll(full)
        for(int64_t b=0; b<BATCH; b++) {
          vrsum_i0[b] = _vel_pvfmad_vvvvvl(vrsum_i0[b], vrw_i0, vro[b], vrsum_i0[b], vl) ;
        }
      }
      {
        const int64_t vl = (outDim - o) >> 1 ;

        __vr vrw_i0 = _vel_vld_vssl(8, pWeight+(i  )*outDim+o, vl) ;

        __vr vro[BATCH] ;
#pragma clang loop unroll(full)
        for(int64_t b=0; b<BATCH; b++) {
          vro[b] = _vel_vld_vssl(8, pGOut+b*outDim+o, vl) ;
        }

#pragma clang loop unroll(full)
        for(int64_t b=0; b<BATCH; b++) {
	  vrsum_i0[b] = _vel_pvfmad_vvvvvl(vrsum_i0[b], vrw_i0, vro[b], vrsum_i0[b], vl) ;

	  vrsum_i0[b] = _vel_vfadds_vvvl(vrsum_i0[b], _vel_vsll_vvsl(vrsum_i0[b], 32, VLEN), VLEN) ;
	  vrsum_i0[b] = _vel_vfsums_vvl(vrsum_i0[b], VLEN);

	  _vel_vstu_vssl(vrsum_i0[b], 4, pGIn+3*inDim+i, 1) ;
        }
      }
    }
    i+=1;
    break ;
  case 2 :
    {
      __vr vrsum_b0_i0 = _vel_vbrdl_vsl(0UL, VLEN) ;
      __vr vrsum_b1_i0 = _vel_vbrdl_vsl(0UL, VLEN) ;
      __vr vrsum_b2_i0 = _vel_vbrdl_vsl(0UL, VLEN) ;
      __vr vrsum_b3_i0 = _vel_vbrdl_vsl(0UL, VLEN) ;
      __vr vrsum_b0_i1 = _vel_vbrdl_vsl(0UL, VLEN) ;
      __vr vrsum_b1_i1 = _vel_vbrdl_vsl(0UL, VLEN) ;
      __vr vrsum_b2_i1 = _vel_vbrdl_vsl(0UL, VLEN) ;
      __vr vrsum_b3_i1 = _vel_vbrdl_vsl(0UL, VLEN) ;

      int64_t o=0 ;
      for(; o+2*VLEN<outDim; o+=2*VLEN)
      {
        const int64_t vl = VLEN ;

        __vr vrw_i0 = _vel_vld_vssl(8, pWeight+(i  )*outDim+o, vl) ;
        __vr vrw_i1 = _vel_vld_vssl(8, pWeight+(i+1)*outDim+o, vl) ;

        __vr vro_b0 = _vel_vld_vssl(8, pGOut+0*outDim+o, vl) ;
        __vr vro_b1 = _vel_vld_vssl(8, pGOut+1*outDim+o, vl) ;
        __vr vro_b2 = _vel_vld_vssl(8, pGOut+2*outDim+o, vl) ;
        __vr vro_b3 = _vel_vld_vssl(8, pGOut+3*outDim+o, vl) ;

        if(BATCH>=1) {
	  vrsum_b0_i0 = _vel_pvfmad_vvvvvl(vrsum_b0_i0, vrw_i0, vro_b0, vrsum_b0_i0, vl) ;
	  vrsum_b0_i1 = _vel_pvfmad_vvvvvl(vrsum_b0_i1, vrw_i1, vro_b0, vrsum_b0_i1, vl) ;
        }
        if(BATCH>=2) {
	  vrsum_b1_i0 = _vel_pvfmad_vvvvvl(vrsum_b1_i0, vrw_i0, vro_b1, vrsum_b1_i0, vl) ;
	  vrsum_b1_i1 = _vel_pvfmad_vvvvvl(vrsum_b1_i1, vrw_i1, vro_b1, vrsum_b1_i1, vl) ;
        }
        if(BATCH>=3) {
	  vrsum_b2_i0 = _vel_pvfmad_vvvvvl(vrsum_b2_i0, vrw_i0, vro_b2, vrsum_b2_i0, vl) ;
	  vrsum_b2_i1 = _vel_pvfmad_vvvvvl(vrsum_b2_i1, vrw_i1, vro_b2, vrsum_b2_i1, vl) ;
        }
        if(BATCH>=4) {
	  vrsum_b3_i0 = _vel_pvfmad_vvvvvl(vrsum_b3_i0, vrw_i0, vro_b3, vrsum_b3_i0, vl) ;
	  vrsum_b3_i1 = _vel_pvfmad_vvvvvl(vrsum_b3_i1, vrw_i1, vro_b3, vrsum_b3_i1, vl) ;
        }
      }
      {
        const int64_t vl = (outDim - o) >> 1 ;

        __vr vrw_i0 = _vel_vld_vssl(8, pWeight+(i  )*outDim+o, vl) ;
        __vr vrw_i1 = _vel_vld_vssl(8, pWeight+(i+1)*outDim+o, vl) ;

        __vr vro_b0 = _vel_vld_vssl(8, pGOut+0*outDim+o, vl) ;
        __vr vro_b1 = _vel_vld_vssl(8, pGOut+1*outDim+o, vl) ;
        __vr vro_b2 = _vel_vld_vssl(8, pGOut+2*outDim+o, vl) ;
        __vr vro_b3 = _vel_vld_vssl(8, pGOut+3*outDim+o, vl) ;

        if(BATCH>=1) {
	  vrsum_b0_i0 = _vel_pvfmad_vvvvvl(vrsum_b0_i0, vrw_i0, vro_b0, vrsum_b0_i0, vl) ;
	  vrsum_b0_i1 = _vel_pvfmad_vvvvvl(vrsum_b0_i1, vrw_i1, vro_b0, vrsum_b0_i1, vl) ;

	  vrsum_b0_i0 = _vel_vfadds_vvvl(vrsum_b0_i0, _vel_vsll_vvsl(vrsum_b0_i0, 32, VLEN), VLEN) ;
	  vrsum_b0_i1 = _vel_vfadds_vvvl(vrsum_b0_i1, _vel_vsll_vvsl(vrsum_b0_i1, 32, VLEN), VLEN) ;
	  vrsum_b0_i0 = _vel_vfsums_vvl(vrsum_b0_i0, VLEN);
	  vrsum_b0_i1 = _vel_vfsums_vvl(vrsum_b0_i1, VLEN);

	  _vel_vstu_vssl(vrsum_b0_i0, 4, pGIn+0*inDim+i, 1) ;
	  _vel_vstu_vssl(vrsum_b0_i1, 4, pGIn+0*inDim+i+1, 1) ;
        }
        if(BATCH>=2) {
	  vrsum_b1_i0 = _vel_pvfmad_vvvvvl(vrsum_b1_i0, vrw_i0, vro_b1, vrsum_b1_i0, vl) ;
	  vrsum_b1_i1 = _vel_pvfmad_vvvvvl(vrsum_b1_i1, vrw_i1, vro_b1, vrsum_b1_i1, vl) ;

	  vrsum_b1_i0 = _vel_vfadds_vvvl(vrsum_b1_i0, _vel_vsll_vvsl(vrsum_b1_i0, 32, VLEN), VLEN) ;
	  vrsum_b1_i1 = _vel_vfadds_vvvl(vrsum_b1_i1, _vel_vsll_vvsl(vrsum_b1_i1, 32, VLEN), VLEN) ;
	  vrsum_b1_i0 = _vel_vfsums_vvl(vrsum_b1_i0, VLEN);
	  vrsum_b1_i1 = _vel_vfsums_vvl(vrsum_b1_i1, VLEN);

	  _vel_vstu_vssl(vrsum_b1_i0, 4, pGIn+1*inDim+i, 1) ;
	  _vel_vstu_vssl(vrsum_b1_i1, 4, pGIn+1*inDim+i+1, 1) ;
        }
        if(BATCH>=3) {
	  vrsum_b2_i0 = _vel_pvfmad_vvvvvl(vrsum_b2_i0, vrw_i0, vro_b2, vrsum_b2_i0, vl) ;
	  vrsum_b2_i1 = _vel_pvfmad_vvvvvl(vrsum_b2_i1, vrw_i1, vro_b2, vrsum_b2_i1, vl) ;

	  vrsum_b2_i0 = _vel_vfadds_vvvl(vrsum_b2_i0, _vel_vsll_vvsl(vrsum_b2_i0, 32, VLEN), VLEN) ;
	  vrsum_b2_i1 = _vel_vfadds_vvvl(vrsum_b2_i1, _vel_vsll_vvsl(vrsum_b2_i1, 32, VLEN), VLEN) ;
	  vrsum_b2_i0 = _vel_vfsums_vvl(vrsum_b2_i0, VLEN);
	  vrsum_b2_i1 = _vel_vfsums_vvl(vrsum_b2_i1, VLEN);

	  _vel_vstu_vssl(vrsum_b2_i0, 4, pGIn+2*inDim+i, 1) ;
	  _vel_vstu_vssl(vrsum_b2_i1, 4, pGIn+2*inDim+i+1, 1) ;
        }
        if(BATCH>=4) {
	  vrsum_b3_i0 = _vel_pvfmad_vvvvvl(vrsum_b3_i0, vrw_i0, vro_b3, vrsum_b3_i0, vl) ;
	  vrsum_b3_i1 = _vel_pvfmad_vvvvvl(vrsum_b3_i1, vrw_i1, vro_b3, vrsum_b3_i1, vl) ;

	  vrsum_b3_i0 = _vel_vfadds_vvvl(vrsum_b3_i0, _vel_vsll_vvsl(vrsum_b3_i0, 32, VLEN), VLEN) ;
	  vrsum_b3_i1 = _vel_vfadds_vvvl(vrsum_b3_i1, _vel_vsll_vvsl(vrsum_b3_i1, 32, VLEN), VLEN) ;
	  vrsum_b3_i0 = _vel_vfsums_vvl(vrsum_b3_i0, VLEN);
	  vrsum_b3_i1 = _vel_vfsums_vvl(vrsum_b3_i1, VLEN);

	  _vel_vstu_vssl(vrsum_b3_i0, 4, pGIn+3*inDim+i, 1) ;
	  _vel_vstu_vssl(vrsum_b3_i1, 4, pGIn+3*inDim+i+1, 1) ;
        }
      }
    }
    i+=2 ;
    break ;
  case 3:
    {
      __vr vrsum_b0_i0 = _vel_vbrdl_vsl(0UL, VLEN) ;
      __vr vrsum_b1_i0 = _vel_vbrdl_vsl(0UL, VLEN) ;
      __vr vrsum_b2_i0 = _vel_vbrdl_vsl(0UL, VLEN) ;
      __vr vrsum_b3_i0 = _vel_vbrdl_vsl(0UL, VLEN) ;
      __vr vrsum_b0_i1 = _vel_vbrdl_vsl(0UL, VLEN) ;
      __vr vrsum_b1_i1 = _vel_vbrdl_vsl(0UL, VLEN) ;
      __vr vrsum_b2_i1 = _vel_vbrdl_vsl(0UL, VLEN) ;
      __vr vrsum_b3_i1 = _vel_vbrdl_vsl(0UL, VLEN) ;
      __vr vrsum_b0_i2 = _vel_vbrdl_vsl(0UL, VLEN) ;
      __vr vrsum_b1_i2 = _vel_vbrdl_vsl(0UL, VLEN) ;
      __vr vrsum_b2_i2 = _vel_vbrdl_vsl(0UL, VLEN) ;
      __vr vrsum_b3_i2 = _vel_vbrdl_vsl(0UL, VLEN) ;

      int64_t o=0 ;
      for(; o+2*VLEN<outDim; o+=2*VLEN)
      {
        const int64_t vl = VLEN ;

        __vr vrw_i0 = _vel_vld_vssl(8, pWeight+(i  )*outDim+o, vl) ;
        __vr vrw_i1 = _vel_vld_vssl(8, pWeight+(i+1)*outDim+o, vl) ;
        __vr vrw_i2 = _vel_vld_vssl(8, pWeight+(i+2)*outDim+o, vl) ;

        __vr vro_b0 = _vel_vld_vssl(8, pGOut+0*outDim+o, vl) ;
        __vr vro_b1 = _vel_vld_vssl(8, pGOut+1*outDim+o, vl) ;
        __vr vro_b2 = _vel_vld_vssl(8, pGOut+2*outDim+o, vl) ;
        __vr vro_b3 = _vel_vld_vssl(8, pGOut+3*outDim+o, vl) ;

        if(BATCH>=1) {
	  vrsum_b0_i0 = _vel_pvfmad_vvvvvl(vrsum_b0_i0, vrw_i0, vro_b0, vrsum_b0_i0, vl) ;
	  vrsum_b0_i1 = _vel_pvfmad_vvvvvl(vrsum_b0_i1, vrw_i1, vro_b0, vrsum_b0_i1, vl) ;
	  vrsum_b0_i2 = _vel_pvfmad_vvvvvl(vrsum_b0_i2, vrw_i2, vro_b0, vrsum_b0_i2, vl) ;
        }
        if(BATCH>=2) {
	  vrsum_b1_i0 = _vel_pvfmad_vvvvvl(vrsum_b1_i0, vrw_i0, vro_b1, vrsum_b1_i0, vl) ;
	  vrsum_b1_i1 = _vel_pvfmad_vvvvvl(vrsum_b1_i1, vrw_i1, vro_b1, vrsum_b1_i1, vl) ;
	  vrsum_b1_i2 = _vel_pvfmad_vvvvvl(vrsum_b1_i2, vrw_i2, vro_b1, vrsum_b1_i2, vl) ;
        }
        if(BATCH>=3) {
	  vrsum_b2_i0 = _vel_pvfmad_vvvvvl(vrsum_b2_i0, vrw_i0, vro_b2, vrsum_b2_i0, vl) ;
	  vrsum_b2_i1 = _vel_pvfmad_vvvvvl(vrsum_b2_i1, vrw_i1, vro_b2, vrsum_b2_i1, vl) ;
	  vrsum_b2_i2 = _vel_pvfmad_vvvvvl(vrsum_b2_i2, vrw_i2, vro_b2, vrsum_b2_i2, vl) ;
        }
        if(BATCH>=4) {
	  vrsum_b3_i0 = _vel_pvfmad_vvvvvl(vrsum_b3_i0, vrw_i0, vro_b3, vrsum_b3_i0, vl) ;
	  vrsum_b3_i1 = _vel_pvfmad_vvvvvl(vrsum_b3_i1, vrw_i1, vro_b3, vrsum_b3_i1, vl) ;
	  vrsum_b3_i2 = _vel_pvfmad_vvvvvl(vrsum_b3_i2, vrw_i2, vro_b3, vrsum_b3_i2, vl) ;
        }
      }
      {
        const int64_t vl = (outDim - o) >> 1 ;

        __vr vrw_i0 = _vel_vld_vssl(8, pWeight+(i  )*outDim+o, vl) ;
        __vr vrw_i1 = _vel_vld_vssl(8, pWeight+(i+1)*outDim+o, vl) ;
        __vr vrw_i2 = _vel_vld_vssl(8, pWeight+(i+2)*outDim+o, vl) ;

        __vr vro_b0 = _vel_vld_vssl(8, pGOut+0*outDim+o, vl) ;
        __vr vro_b1 = _vel_vld_vssl(8, pGOut+1*outDim+o, vl) ;
        __vr vro_b2 = _vel_vld_vssl(8, pGOut+2*outDim+o, vl) ;
        __vr vro_b3 = _vel_vld_vssl(8, pGOut+3*outDim+o, vl) ;

        if(BATCH>=1) {
	  vrsum_b0_i0 = _vel_pvfmad_vvvvvl(vrsum_b0_i0, vrw_i0, vro_b0, vrsum_b0_i0, vl) ;
	  vrsum_b0_i1 = _vel_pvfmad_vvvvvl(vrsum_b0_i1, vrw_i1, vro_b0, vrsum_b0_i1, vl) ;
	  vrsum_b0_i2 = _vel_pvfmad_vvvvvl(vrsum_b0_i2, vrw_i2, vro_b0, vrsum_b0_i2, vl) ;

	  vrsum_b0_i0 = _vel_vfadds_vvvl(vrsum_b0_i0, _vel_vsll_vvsl(vrsum_b0_i0, 32, VLEN), VLEN) ;
	  vrsum_b0_i1 = _vel_vfadds_vvvl(vrsum_b0_i1, _vel_vsll_vvsl(vrsum_b0_i1, 32, VLEN), VLEN) ;
	  vrsum_b0_i2 = _vel_vfadds_vvvl(vrsum_b0_i2, _vel_vsll_vvsl(vrsum_b0_i2, 32, VLEN), VLEN) ;
	  vrsum_b0_i0 = _vel_vfsums_vvl(vrsum_b0_i0, VLEN);
	  vrsum_b0_i1 = _vel_vfsums_vvl(vrsum_b0_i1, VLEN);
	  vrsum_b0_i2 = _vel_vfsums_vvl(vrsum_b0_i2, VLEN);

	  _vel_vstu_vssl(vrsum_b0_i0, 4, pGIn+0*inDim+i, 1) ;
	  _vel_vstu_vssl(vrsum_b0_i1, 4, pGIn+0*inDim+i+1, 1) ;
	  _vel_vstu_vssl(vrsum_b0_i2, 4, pGIn+0*inDim+i+2, 1) ;
        }
        if(BATCH>=2) {
	  vrsum_b1_i0 = _vel_pvfmad_vvvvvl(vrsum_b1_i0, vrw_i0, vro_b1, vrsum_b1_i0, vl) ;
	  vrsum_b1_i1 = _vel_pvfmad_vvvvvl(vrsum_b1_i1, vrw_i1, vro_b1, vrsum_b1_i1, vl) ;
	  vrsum_b1_i2 = _vel_pvfmad_vvvvvl(vrsum_b1_i2, vrw_i2, vro_b1, vrsum_b1_i2, vl) ;

	  vrsum_b1_i0 = _vel_vfadds_vvvl(vrsum_b1_i0, _vel_vsll_vvsl(vrsum_b1_i0, 32, VLEN), VLEN) ;
	  vrsum_b1_i1 = _vel_vfadds_vvvl(vrsum_b1_i1, _vel_vsll_vvsl(vrsum_b1_i1, 32, VLEN), VLEN) ;
	  vrsum_b1_i2 = _vel_vfadds_vvvl(vrsum_b1_i2, _vel_vsll_vvsl(vrsum_b1_i2, 32, VLEN), VLEN) ;
	  vrsum_b1_i0 = _vel_vfsums_vvl(vrsum_b1_i0, VLEN);
	  vrsum_b1_i1 = _vel_vfsums_vvl(vrsum_b1_i1, VLEN);
	  vrsum_b1_i2 = _vel_vfsums_vvl(vrsum_b1_i2, VLEN);

	  _vel_vstu_vssl(vrsum_b1_i0, 4, pGIn+1*inDim+i, 1) ;
	  _vel_vstu_vssl(vrsum_b1_i1, 4, pGIn+1*inDim+i+1, 1) ;
	  _vel_vstu_vssl(vrsum_b1_i2, 4, pGIn+1*inDim+i+2, 1) ;
        }
        if(BATCH>=3) {
	  vrsum_b2_i0 = _vel_pvfmad_vvvvvl(vrsum_b2_i0, vrw_i0, vro_b2, vrsum_b2_i0, vl) ;
	  vrsum_b2_i1 = _vel_pvfmad_vvvvvl(vrsum_b2_i1, vrw_i1, vro_b2, vrsum_b2_i1, vl) ;
	  vrsum_b2_i2 = _vel_pvfmad_vvvvvl(vrsum_b2_i2, vrw_i2, vro_b2, vrsum_b2_i2, vl) ;

	  vrsum_b2_i0 = _vel_vfadds_vvvl(vrsum_b2_i0, _vel_vsll_vvsl(vrsum_b2_i0, 32, VLEN), VLEN) ;
	  vrsum_b2_i1 = _vel_vfadds_vvvl(vrsum_b2_i1, _vel_vsll_vvsl(vrsum_b2_i1, 32, VLEN), VLEN) ;
	  vrsum_b2_i2 = _vel_vfadds_vvvl(vrsum_b2_i2, _vel_vsll_vvsl(vrsum_b2_i2, 32, VLEN), VLEN) ;
	  vrsum_b2_i0 = _vel_vfsums_vvl(vrsum_b2_i0, VLEN);
	  vrsum_b2_i1 = _vel_vfsums_vvl(vrsum_b2_i1, VLEN);
	  vrsum_b2_i2 = _vel_vfsums_vvl(vrsum_b2_i2, VLEN);

	  _vel_vstu_vssl(vrsum_b2_i0, 4, pGIn+2*inDim+i, 1) ;
	  _vel_vstu_vssl(vrsum_b2_i1, 4, pGIn+2*inDim+i+1, 1) ;
	  _vel_vstu_vssl(vrsum_b2_i2, 4, pGIn+2*inDim+i+2, 1) ;
        }
        if(BATCH>=4) {
	  vrsum_b3_i0 = _vel_pvfmad_vvvvvl(vrsum_b3_i0, vrw_i0, vro_b3, vrsum_b3_i0, vl) ;
	  vrsum_b3_i1 = _vel_pvfmad_vvvvvl(vrsum_b3_i1, vrw_i1, vro_b3, vrsum_b3_i1, vl) ;
	  vrsum_b3_i2 = _vel_pvfmad_vvvvvl(vrsum_b3_i2, vrw_i2, vro_b3, vrsum_b3_i2, vl) ;

	  vrsum_b3_i0 = _vel_vfadds_vvvl(vrsum_b3_i0, _vel_vsll_vvsl(vrsum_b3_i0, 32, VLEN), VLEN) ;
	  vrsum_b3_i1 = _vel_vfadds_vvvl(vrsum_b3_i1, _vel_vsll_vvsl(vrsum_b3_i1, 32, VLEN), VLEN) ;
	  vrsum_b3_i2 = _vel_vfadds_vvvl(vrsum_b3_i2, _vel_vsll_vvsl(vrsum_b3_i2, 32, VLEN), VLEN) ;
	  vrsum_b3_i0 = _vel_vfsums_vvl(vrsum_b3_i0, VLEN);
	  vrsum_b3_i1 = _vel_vfsums_vvl(vrsum_b3_i1, VLEN);
	  vrsum_b3_i2 = _vel_vfsums_vvl(vrsum_b3_i2, VLEN);

	  _vel_vstu_vssl(vrsum_b3_i0, 4, pGIn+3*inDim+i, 1) ;
	  _vel_vstu_vssl(vrsum_b3_i1, 4, pGIn+3*inDim+i+1, 1) ;
	  _vel_vstu_vssl(vrsum_b3_i2, 4, pGIn+3*inDim+i+2, 1) ;
        }
      }
    }
    i+=3 ;
    break ;
  default : break ;
  }
#endif
#if 0
  for(; i<inDim; i+=4)
  {
    __vr vrsum_i0[BATCH] ;
    __vr vrsum_i1[BATCH] ;
    __vr vrsum_i2[BATCH] ;
    __vr vrsum_i3[BATCH] ;
#pragma clang loop unroll(full)
    for(int64_t b=0; b<BATCH; b++) {
      vrsum_i0[b] = _vel_pvbrd_vsl(0UL, mvl) ;
      vrsum_i1[b] = _vel_pvbrd_vsl(0UL, mvl) ;
      vrsum_i2[b] = _vel_pvbrd_vsl(0UL, mvl) ;
      vrsum_i3[b] = _vel_pvbrd_vsl(0UL, mvl) ;
    }

    int64_t o=0 ;
    for(; o+2*mvl<outDim; o+=2*mvl)
    {
      const int64_t vl = mvl ;

      __vr vrw_i0 = _vel_vld_vssl(8, pWeight+(i+0)*outDim+o, vl) ;
      __vr vrw_i1 = _vel_vld_vssl(8, pWeight+(i+1)*outDim+o, vl) ;
      __vr vrw_i2 = _vel_vld_vssl(8, pWeight+(i+2)*outDim+o, vl) ;
      __vr vrw_i3 = _vel_vld_vssl(8, pWeight+(i+3)*outDim+o, vl) ;

      __vr vro[BATCH] ;
#pragma clang loop unroll(full)
      for(int64_t b=0; b<BATCH; b++) {
        vro[b] = _vel_vld_vssl(8, pGOut+b*outDim+o, vl) ;
      }

#pragma clang loop unroll(full)
      for(int64_t b=0; b<BATCH; b++) {
	vrsum_i0[b] = _vel_pvfmad_vvvvvl(vrsum_i0[b], vrw_i0, vro[b], vrsum_i0[b], vl) ;
	vrsum_i1[b] = _vel_pvfmad_vvvvvl(vrsum_i1[b], vrw_i1, vro[b], vrsum_i1[b], vl) ;
	vrsum_i2[b] = _vel_pvfmad_vvvvvl(vrsum_i2[b], vrw_i2, vro[b], vrsum_i2[b], vl) ;
	vrsum_i3[b] = _vel_pvfmad_vvvvvl(vrsum_i3[b], vrw_i3, vro[b], vrsum_i3[b], vl) ;
      }
    }
    {
      const int64_t vl = (outDim - o) >> 1 ;

      __vr vrw_i0 = _vel_vld_vssl(8, pWeight+(i+0)*outDim+o, vl) ;
      __vr vrw_i1 = _vel_vld_vssl(8, pWeight+(i+1)*outDim+o, vl) ;
      __vr vrw_i2 = _vel_vld_vssl(8, pWeight+(i+2)*outDim+o, vl) ;
      __vr vrw_i3 = _vel_vld_vssl(8, pWeight+(i+3)*outDim+o, vl) ;

      __vr vro[BATCH] ;
#pragma clang loop unroll(full)
      for(int64_t b=0; b<BATCH; b++) {
        vro[b] = _vel_vld_vssl(8, pGOut+b*outDim+o, vl) ;
      }

#pragma clang loop unroll(full)
      for(int64_t b=0; b<BATCH; b++) {
	vrsum_i0[b] = _vel_pvfmad_vvvvvl(vrsum_i0[b], vrw_i0, vro[b], vrsum_i0[b], vl) ;
	vrsum_i1[b] = _vel_pvfmad_vvvvvl(vrsum_i1[b], vrw_i1, vro[b], vrsum_i1[b], vl) ;
	vrsum_i2[b] = _vel_pvfmad_vvvvvl(vrsum_i2[b], vrw_i2, vro[b], vrsum_i2[b], vl) ;
	vrsum_i3[b] = _vel_pvfmad_vvvvvl(vrsum_i3[b], vrw_i3, vro[b], vrsum_i3[b], vl) ;

	vrsum_i0[b] = _vel_vfadds_vvvl(vrsum_i0[b], _vel_vsll_vvsl(vrsum_i0[b], 32, mvl), mvl) ;
	vrsum_i1[b] = _vel_vfadds_vvvl(vrsum_i1[b], _vel_vsll_vvsl(vrsum_i1[b], 32, mvl), mvl) ;
	vrsum_i2[b] = _vel_vfadds_vvvl(vrsum_i2[b], _vel_vsll_vvsl(vrsum_i2[b], 32, mvl), mvl) ;
	vrsum_i3[b] = _vel_vfadds_vvvl(vrsum_i3[b], _vel_vsll_vvsl(vrsum_i3[b], 32, mvl), mvl) ;

	vrsum_i0[b] = _vel_vfsums_vvl(vrsum_i0[b], mvl);
	vrsum_i1[b] = _vel_vfsums_vvl(vrsum_i1[b], mvl);
	vrsum_i2[b] = _vel_vfsums_vvl(vrsum_i2[b], mvl);
	vrsum_i3[b] = _vel_vfsums_vvl(vrsum_i3[b], mvl);

	_vel_vstu_vssl(vrsum_i0[b], 4, pGIn+b*inDim+i+0, 1) ;
	_vel_vstu_vssl(vrsum_i1[b], 4, pGIn+b*inDim+i+1, 1) ;
	_vel_vstu_vssl(vrsum_i2[b], 4, pGIn+b*inDim+i+2, 1) ;
	_vel_vstu_vssl(vrsum_i3[b], 4, pGIn+b*inDim+i+3, 1) ;
      }
    }
  }
#endif
  switch(inDim&0x07) {
  case 1 :
    {
      __vr vrsum_i0[BATCH] ;
#pragma clang loop unroll(full)
      for(int64_t b=0; b<BATCH; b++) {
	vrsum_i0[b] = _vel_pvbrd_vsl(0UL, mvl) ;
      }

      int64_t o=0 ;
      for(; o+2*mvl<outDim; o+=2*mvl)
      {
	const int64_t vl = mvl ;

	__vr vrw_i0 = _vel_vld_vssl(8, pWeight+(i+0)*outDim+o, vl) ;

	__vr vro[BATCH] ;
#pragma clang loop unroll(full)
	for(int64_t b=0; b<BATCH; b++) {
	  vro[b] = _vel_vld_vssl(8, pGOut+b*outDim+o, vl) ;
	}

#pragma clang loop unroll(full)
	for(int64_t b=0; b<BATCH; b++) {
	  vrsum_i0[b] = _vel_pvfmad_vvvvvl(vrsum_i0[b], vrw_i0, vro[b], vrsum_i0[b], vl) ;
	}
      }
      {
	const int64_t vl = (outDim - o) >> 1 ;

	__vr vrw_i0 = _vel_vld_vssl(8, pWeight+(i+0)*outDim+o, vl) ;

	__vr vro[BATCH] ;
#pragma clang loop unroll(full)
	for(int64_t b=0; b<BATCH; b++) {
	  vro[b] = _vel_vld_vssl(8, pGOut+b*outDim+o, vl) ;
	}

#pragma clang loop unroll(full)
	for(int64_t b=0; b<BATCH; b++) {
	  vrsum_i0[b] = _vel_pvfmad_vvvvvl(vrsum_i0[b], vrw_i0, vro[b], vrsum_i0[b], vl) ;
	  vrsum_i0[b] = _vel_vfadds_vvvl(vrsum_i0[b], _vel_vsll_vvsl(vrsum_i0[b], 32, mvl), mvl) ;
	  vrsum_i0[b] = _vel_vfsums_vvl(vrsum_i0[b], mvl);
	  _vel_vstu_vssl(vrsum_i0[b], 4, pGIn+b*inDim+i+0, 1) ;
	}
      }
    }
    break ;
  case 2 :
    {
      __vr vrsum_i0[BATCH] ;
      __vr vrsum_i1[BATCH] ;
#pragma clang loop unroll(full)
      for(int64_t b=0; b<BATCH; b++) {
	vrsum_i0[b] = _vel_pvbrd_vsl(0UL, mvl) ;
	vrsum_i1[b] = _vel_pvbrd_vsl(0UL, mvl) ;
      }

      int64_t o=0 ;
      for(; o+2*mvl<outDim; o+=2*mvl)
      {
	const int64_t vl = mvl ;

	__vr vrw_i0 = _vel_vld_vssl(8, pWeight+(i+0)*outDim+o, vl) ;
	__vr vrw_i1 = _vel_vld_vssl(8, pWeight+(i+1)*outDim+o, vl) ;

	__vr vro[BATCH] ;
#pragma clang loop unroll(full)
	for(int64_t b=0; b<BATCH; b++) {
	  vro[b] = _vel_vld_vssl(8, pGOut+b*outDim+o, vl) ;
	}

#pragma clang loop unroll(full)
	for(int64_t b=0; b<BATCH; b++) {
	  vrsum_i0[b] = _vel_pvfmad_vvvvvl(vrsum_i0[b], vrw_i0, vro[b], vrsum_i0[b], vl) ;
	  vrsum_i1[b] = _vel_pvfmad_vvvvvl(vrsum_i1[b], vrw_i1, vro[b], vrsum_i1[b], vl) ;
	}
      }
      {
	const int64_t vl = (outDim - o) >> 1 ;

	__vr vrw_i0 = _vel_vld_vssl(8, pWeight+(i+0)*outDim+o, vl) ;
	__vr vrw_i1 = _vel_vld_vssl(8, pWeight+(i+1)*outDim+o, vl) ;

	__vr vro[BATCH] ;
#pragma clang loop unroll(full)
	for(int64_t b=0; b<BATCH; b++) {
	  vro[b] = _vel_vld_vssl(8, pGOut+b*outDim+o, vl) ;
	}

#pragma clang loop unroll(full)
	for(int64_t b=0; b<BATCH; b++) {
	  vrsum_i0[b] = _vel_pvfmad_vvvvvl(vrsum_i0[b], vrw_i0, vro[b], vrsum_i0[b], vl) ;
	  vrsum_i0[b] = _vel_vfadds_vvvl(vrsum_i0[b], _vel_vsll_vvsl(vrsum_i0[b], 32, mvl), mvl) ;
	  vrsum_i0[b] = _vel_vfsums_vvl(vrsum_i0[b], mvl);
	  _vel_vstu_vssl(vrsum_i0[b], 4, pGIn+b*inDim+i+0, 1) ;

	  vrsum_i1[b] = _vel_pvfmad_vvvvvl(vrsum_i1[b], vrw_i1, vro[b], vrsum_i1[b], vl) ;
	  vrsum_i1[b] = _vel_vfadds_vvvl(vrsum_i1[b], _vel_vsll_vvsl(vrsum_i1[b], 32, mvl), mvl) ;
	  vrsum_i1[b] = _vel_vfsums_vvl(vrsum_i1[b], mvl);
	  _vel_vstu_vssl(vrsum_i1[b], 4, pGIn+b*inDim+i+1, 1) ;
	}
      }
    }
    break ;
  case 3 :
    {
      __vr vrsum_i0[BATCH] ;
      __vr vrsum_i1[BATCH] ;
      __vr vrsum_i2[BATCH] ;
#pragma clang loop unroll(full)
      for(int64_t b=0; b<BATCH; b++) {
	vrsum_i0[b] = _vel_pvbrd_vsl(0UL, mvl) ;
	vrsum_i1[b] = _vel_pvbrd_vsl(0UL, mvl) ;
	vrsum_i2[b] = _vel_pvbrd_vsl(0UL, mvl) ;
      }

      int64_t o=0 ;
      for(; o+2*mvl<outDim; o+=2*mvl)
      {
	const int64_t vl = mvl ;

	__vr vrw_i0 = _vel_vld_vssl(8, pWeight+(i+0)*outDim+o, vl) ;
	__vr vrw_i1 = _vel_vld_vssl(8, pWeight+(i+1)*outDim+o, vl) ;
	__vr vrw_i2 = _vel_vld_vssl(8, pWeight+(i+2)*outDim+o, vl) ;

	__vr vro[BATCH] ;
#pragma clang loop unroll(full)
	for(int64_t b=0; b<BATCH; b++) {
	  vro[b] = _vel_vld_vssl(8, pGOut+b*outDim+o, vl) ;
	}

#pragma clang loop unroll(full)
	for(int64_t b=0; b<BATCH; b++) {
	  vrsum_i0[b] = _vel_pvfmad_vvvvvl(vrsum_i0[b], vrw_i0, vro[b], vrsum_i0[b], vl) ;
	  vrsum_i1[b] = _vel_pvfmad_vvvvvl(vrsum_i1[b], vrw_i1, vro[b], vrsum_i1[b], vl) ;
	  vrsum_i2[b] = _vel_pvfmad_vvvvvl(vrsum_i2[b], vrw_i2, vro[b], vrsum_i2[b], vl) ;
	}
      }
      {
	const int64_t vl = (outDim - o) >> 1 ;

	__vr vrw_i0 = _vel_vld_vssl(8, pWeight+(i+0)*outDim+o, vl) ;
	__vr vrw_i1 = _vel_vld_vssl(8, pWeight+(i+1)*outDim+o, vl) ;
	__vr vrw_i2 = _vel_vld_vssl(8, pWeight+(i+2)*outDim+o, vl) ;

	__vr vro[BATCH] ;
#pragma clang loop unroll(full)
	for(int64_t b=0; b<BATCH; b++) {
	  vro[b] = _vel_vld_vssl(8, pGOut+b*outDim+o, vl) ;
	}

#pragma clang loop unroll(full)
	for(int64_t b=0; b<BATCH; b++) {
	  vrsum_i0[b] = _vel_pvfmad_vvvvvl(vrsum_i0[b], vrw_i0, vro[b], vrsum_i0[b], vl) ;
	  vrsum_i0[b] = _vel_vfadds_vvvl(vrsum_i0[b], _vel_vsll_vvsl(vrsum_i0[b], 32, mvl), mvl) ;
	  vrsum_i0[b] = _vel_vfsums_vvl(vrsum_i0[b], mvl);
	  _vel_vstu_vssl(vrsum_i0[b], 4, pGIn+b*inDim+i+0, 1) ;

	  vrsum_i1[b] = _vel_pvfmad_vvvvvl(vrsum_i1[b], vrw_i1, vro[b], vrsum_i1[b], vl) ;
	  vrsum_i1[b] = _vel_vfadds_vvvl(vrsum_i1[b], _vel_vsll_vvsl(vrsum_i1[b], 32, mvl), mvl) ;
	  vrsum_i1[b] = _vel_vfsums_vvl(vrsum_i1[b], mvl);
	  _vel_vstu_vssl(vrsum_i1[b], 4, pGIn+b*inDim+i+1, 1) ;

	  vrsum_i2[b] = _vel_pvfmad_vvvvvl(vrsum_i2[b], vrw_i2, vro[b], vrsum_i2[b], vl) ;
	  vrsum_i2[b] = _vel_vfadds_vvvl(vrsum_i2[b], _vel_vsll_vvsl(vrsum_i2[b], 32, mvl), mvl) ;
	  vrsum_i2[b] = _vel_vfsums_vvl(vrsum_i2[b], mvl);
	  _vel_vstu_vssl(vrsum_i2[b], 4, pGIn+b*inDim+i+2, 1) ;

	}
      }
    }
    break ;
  case 4 :
    {
      __vr vrsum_i0[BATCH] ;
      __vr vrsum_i1[BATCH] ;
      __vr vrsum_i2[BATCH] ;
      __vr vrsum_i3[BATCH] ;
#pragma clang loop unroll(full)
      for(int64_t b=0; b<BATCH; b++) {
	vrsum_i0[b] = _vel_pvbrd_vsl(0UL, mvl) ;
	vrsum_i1[b] = _vel_pvbrd_vsl(0UL, mvl) ;
	vrsum_i2[b] = _vel_pvbrd_vsl(0UL, mvl) ;
	vrsum_i3[b] = _vel_pvbrd_vsl(0UL, mvl) ;
      }

      int64_t o=0 ;
      for(; o+2*mvl<outDim; o+=2*mvl)
      {
	const int64_t vl = mvl ;

	__vr vrw_i0 = _vel_vld_vssl(8, pWeight+(i+0)*outDim+o, vl) ;
	__vr vrw_i1 = _vel_vld_vssl(8, pWeight+(i+1)*outDim+o, vl) ;
	__vr vrw_i2 = _vel_vld_vssl(8, pWeight+(i+2)*outDim+o, vl) ;
	__vr vrw_i3 = _vel_vld_vssl(8, pWeight+(i+3)*outDim+o, vl) ;

	__vr vro[BATCH] ;
#pragma clang loop unroll(full)
	for(int64_t b=0; b<BATCH; b++) {
	  vro[b] = _vel_vld_vssl(8, pGOut+b*outDim+o, vl) ;
	}

#pragma clang loop unroll(full)
	for(int64_t b=0; b<BATCH; b++) {
	  vrsum_i0[b] = _vel_pvfmad_vvvvvl(vrsum_i0[b], vrw_i0, vro[b], vrsum_i0[b], vl) ;
	  vrsum_i1[b] = _vel_pvfmad_vvvvvl(vrsum_i1[b], vrw_i1, vro[b], vrsum_i1[b], vl) ;
	  vrsum_i2[b] = _vel_pvfmad_vvvvvl(vrsum_i2[b], vrw_i2, vro[b], vrsum_i2[b], vl) ;
	  vrsum_i3[b] = _vel_pvfmad_vvvvvl(vrsum_i3[b], vrw_i3, vro[b], vrsum_i3[b], vl) ;
	}
      }
      {
	const int64_t vl = (outDim - o) >> 1 ;

	__vr vrw_i0 = _vel_vld_vssl(8, pWeight+(i+0)*outDim+o, vl) ;
	__vr vrw_i1 = _vel_vld_vssl(8, pWeight+(i+1)*outDim+o, vl) ;
	__vr vrw_i2 = _vel_vld_vssl(8, pWeight+(i+2)*outDim+o, vl) ;
	__vr vrw_i3 = _vel_vld_vssl(8, pWeight+(i+3)*outDim+o, vl) ;

	__vr vro[BATCH] ;
#pragma clang loop unroll(full)
	for(int64_t b=0; b<BATCH; b++) {
	  vro[b] = _vel_vld_vssl(8, pGOut+b*outDim+o, vl) ;
	}

#pragma clang loop unroll(full)
	for(int64_t b=0; b<BATCH; b++) {
	  vrsum_i0[b] = _vel_pvfmad_vvvvvl(vrsum_i0[b], vrw_i0, vro[b], vrsum_i0[b], vl) ;
	  vrsum_i0[b] = _vel_vfadds_vvvl(vrsum_i0[b], _vel_vsll_vvsl(vrsum_i0[b], 32, mvl), mvl) ;
	  vrsum_i0[b] = _vel_vfsums_vvl(vrsum_i0[b], mvl);
	  _vel_vstu_vssl(vrsum_i0[b], 4, pGIn+b*inDim+i+0, 1) ;

	  vrsum_i1[b] = _vel_pvfmad_vvvvvl(vrsum_i1[b], vrw_i1, vro[b], vrsum_i1[b], vl) ;
	  vrsum_i1[b] = _vel_vfadds_vvvl(vrsum_i1[b], _vel_vsll_vvsl(vrsum_i1[b], 32, mvl), mvl) ;
	  vrsum_i1[b] = _vel_vfsums_vvl(vrsum_i1[b], mvl);
	  _vel_vstu_vssl(vrsum_i1[b], 4, pGIn+b*inDim+i+1, 1) ;

	  vrsum_i2[b] = _vel_pvfmad_vvvvvl(vrsum_i2[b], vrw_i2, vro[b], vrsum_i2[b], vl) ;
	  vrsum_i2[b] = _vel_vfadds_vvvl(vrsum_i2[b], _vel_vsll_vvsl(vrsum_i2[b], 32, mvl), mvl) ;
	  vrsum_i2[b] = _vel_vfsums_vvl(vrsum_i2[b], mvl);
	  _vel_vstu_vssl(vrsum_i2[b], 4, pGIn+b*inDim+i+2, 1) ;

	  vrsum_i3[b] = _vel_pvfmad_vvvvvl(vrsum_i3[b], vrw_i3, vro[b], vrsum_i3[b], vl) ;
	  vrsum_i3[b] = _vel_vfadds_vvvl(vrsum_i3[b], _vel_vsll_vvsl(vrsum_i3[b], 32, mvl), mvl) ;
	  vrsum_i3[b] = _vel_vfsums_vvl(vrsum_i3[b], mvl);
	  _vel_vstu_vssl(vrsum_i3[b], 4, pGIn+b*inDim+i+3, 1) ;
	}
      }
    }
    break ;
  case 5 :
    {
      __vr vrsum_i0[BATCH] ;
      __vr vrsum_i1[BATCH] ;
      __vr vrsum_i2[BATCH] ;
      __vr vrsum_i3[BATCH] ;
      __vr vrsum_i4[BATCH] ;
#pragma clang loop unroll(full)
      for(int64_t b=0; b<BATCH; b++) {
	vrsum_i0[b] = _vel_pvbrd_vsl(0UL, mvl) ;
	vrsum_i1[b] = _vel_pvbrd_vsl(0UL, mvl) ;
	vrsum_i2[b] = _vel_pvbrd_vsl(0UL, mvl) ;
	vrsum_i3[b] = _vel_pvbrd_vsl(0UL, mvl) ;
	vrsum_i4[b] = _vel_pvbrd_vsl(0UL, mvl) ;
      }

      int64_t o=0 ;
      for(; o+2*mvl<outDim; o+=2*mvl)
      {
	const int64_t vl = mvl ;

	__vr vrw_i0 = _vel_vld_vssl(8, pWeight+(i+0)*outDim+o, vl) ;
	__vr vrw_i1 = _vel_vld_vssl(8, pWeight+(i+1)*outDim+o, vl) ;
	__vr vrw_i2 = _vel_vld_vssl(8, pWeight+(i+2)*outDim+o, vl) ;
	__vr vrw_i3 = _vel_vld_vssl(8, pWeight+(i+3)*outDim+o, vl) ;
	__vr vrw_i4 = _vel_vld_vssl(8, pWeight+(i+4)*outDim+o, vl) ;

	__vr vro[BATCH] ;
#pragma clang loop unroll(full)
	for(int64_t b=0; b<BATCH; b++) {
	  vro[b] = _vel_vld_vssl(8, pGOut+b*outDim+o, vl) ;
	}

#pragma clang loop unroll(full)
	for(int64_t b=0; b<BATCH; b++) {
	  vrsum_i0[b] = _vel_pvfmad_vvvvvl(vrsum_i0[b], vrw_i0, vro[b], vrsum_i0[b], vl) ;
	  vrsum_i1[b] = _vel_pvfmad_vvvvvl(vrsum_i1[b], vrw_i1, vro[b], vrsum_i1[b], vl) ;
	  vrsum_i2[b] = _vel_pvfmad_vvvvvl(vrsum_i2[b], vrw_i2, vro[b], vrsum_i2[b], vl) ;
	  vrsum_i3[b] = _vel_pvfmad_vvvvvl(vrsum_i3[b], vrw_i3, vro[b], vrsum_i3[b], vl) ;
	  vrsum_i4[b] = _vel_pvfmad_vvvvvl(vrsum_i4[b], vrw_i4, vro[b], vrsum_i4[b], vl) ;
	}
      }
      {
	const int64_t vl = (outDim - o) >> 1 ;

	__vr vrw_i0 = _vel_vld_vssl(8, pWeight+(i+0)*outDim+o, vl) ;
	__vr vrw_i1 = _vel_vld_vssl(8, pWeight+(i+1)*outDim+o, vl) ;
	__vr vrw_i2 = _vel_vld_vssl(8, pWeight+(i+2)*outDim+o, vl) ;
	__vr vrw_i3 = _vel_vld_vssl(8, pWeight+(i+3)*outDim+o, vl) ;
	__vr vrw_i4 = _vel_vld_vssl(8, pWeight+(i+4)*outDim+o, vl) ;

	__vr vro[BATCH] ;
#pragma clang loop unroll(full)
	for(int64_t b=0; b<BATCH; b++) {
	  vro[b] = _vel_vld_vssl(8, pGOut+b*outDim+o, vl) ;
	}

#pragma clang loop unroll(full)
	for(int64_t b=0; b<BATCH; b++) {
	  vrsum_i0[b] = _vel_pvfmad_vvvvvl(vrsum_i0[b], vrw_i0, vro[b], vrsum_i0[b], vl) ;
	  vrsum_i0[b] = _vel_vfadds_vvvl(vrsum_i0[b], _vel_vsll_vvsl(vrsum_i0[b], 32, mvl), mvl) ;
	  vrsum_i0[b] = _vel_vfsums_vvl(vrsum_i0[b], mvl);
	  _vel_vstu_vssl(vrsum_i0[b], 4, pGIn+b*inDim+i+0, 1) ;

	  vrsum_i1[b] = _vel_pvfmad_vvvvvl(vrsum_i1[b], vrw_i1, vro[b], vrsum_i1[b], vl) ;
	  vrsum_i1[b] = _vel_vfadds_vvvl(vrsum_i1[b], _vel_vsll_vvsl(vrsum_i1[b], 32, mvl), mvl) ;
	  vrsum_i1[b] = _vel_vfsums_vvl(vrsum_i1[b], mvl);
	  _vel_vstu_vssl(vrsum_i1[b], 4, pGIn+b*inDim+i+1, 1) ;

	  vrsum_i2[b] = _vel_pvfmad_vvvvvl(vrsum_i2[b], vrw_i2, vro[b], vrsum_i2[b], vl) ;
	  vrsum_i2[b] = _vel_vfadds_vvvl(vrsum_i2[b], _vel_vsll_vvsl(vrsum_i2[b], 32, mvl), mvl) ;
	  vrsum_i2[b] = _vel_vfsums_vvl(vrsum_i2[b], mvl);
	  _vel_vstu_vssl(vrsum_i2[b], 4, pGIn+b*inDim+i+2, 1) ;

	  vrsum_i3[b] = _vel_pvfmad_vvvvvl(vrsum_i3[b], vrw_i3, vro[b], vrsum_i3[b], vl) ;
	  vrsum_i3[b] = _vel_vfadds_vvvl(vrsum_i3[b], _vel_vsll_vvsl(vrsum_i3[b], 32, mvl), mvl) ;
	  vrsum_i3[b] = _vel_vfsums_vvl(vrsum_i3[b], mvl);
	  _vel_vstu_vssl(vrsum_i3[b], 4, pGIn+b*inDim+i+3, 1) ;

	  vrsum_i4[b] = _vel_pvfmad_vvvvvl(vrsum_i4[b], vrw_i4, vro[b], vrsum_i4[b], vl) ;
	  vrsum_i4[b] = _vel_vfadds_vvvl(vrsum_i4[b], _vel_vsll_vvsl(vrsum_i4[b], 32, mvl), mvl) ;
	  vrsum_i4[b] = _vel_vfsums_vvl(vrsum_i4[b], mvl);
	  _vel_vstu_vssl(vrsum_i4[b], 4, pGIn+b*inDim+i+4, 1) ;
	}
      }
    }
    break ;
  case 6 :
    {
      __vr vrsum_i0[BATCH] ;
      __vr vrsum_i1[BATCH] ;
      __vr vrsum_i2[BATCH] ;
      __vr vrsum_i3[BATCH] ;
      __vr vrsum_i4[BATCH] ;
      __vr vrsum_i5[BATCH] ;
#pragma clang loop unroll(full)
      for(int64_t b=0; b<BATCH; b++) {
	vrsum_i0[b] = _vel_pvbrd_vsl(0UL, mvl) ;
	vrsum_i1[b] = _vel_pvbrd_vsl(0UL, mvl) ;
	vrsum_i2[b] = _vel_pvbrd_vsl(0UL, mvl) ;
	vrsum_i3[b] = _vel_pvbrd_vsl(0UL, mvl) ;
	vrsum_i4[b] = _vel_pvbrd_vsl(0UL, mvl) ;
	vrsum_i5[b] = _vel_pvbrd_vsl(0UL, mvl) ;
      }

      int64_t o=0 ;
      for(; o+2*mvl<outDim; o+=2*mvl)
      {
	const int64_t vl = mvl ;

	__vr vrw_i0 = _vel_vld_vssl(8, pWeight+(i+0)*outDim+o, vl) ;
	__vr vrw_i1 = _vel_vld_vssl(8, pWeight+(i+1)*outDim+o, vl) ;
	__vr vrw_i2 = _vel_vld_vssl(8, pWeight+(i+2)*outDim+o, vl) ;
	__vr vrw_i3 = _vel_vld_vssl(8, pWeight+(i+3)*outDim+o, vl) ;
	__vr vrw_i4 = _vel_vld_vssl(8, pWeight+(i+4)*outDim+o, vl) ;
	__vr vrw_i5 = _vel_vld_vssl(8, pWeight+(i+5)*outDim+o, vl) ;

	__vr vro[BATCH] ;
#pragma clang loop unroll(full)
	for(int64_t b=0; b<BATCH; b++) {
	  vro[b] = _vel_vld_vssl(8, pGOut+b*outDim+o, vl) ;
	}

#pragma clang loop unroll(full)
	for(int64_t b=0; b<BATCH; b++) {
	  vrsum_i0[b] = _vel_pvfmad_vvvvvl(vrsum_i0[b], vrw_i0, vro[b], vrsum_i0[b], vl) ;
	  vrsum_i1[b] = _vel_pvfmad_vvvvvl(vrsum_i1[b], vrw_i1, vro[b], vrsum_i1[b], vl) ;
	  vrsum_i2[b] = _vel_pvfmad_vvvvvl(vrsum_i2[b], vrw_i2, vro[b], vrsum_i2[b], vl) ;
	  vrsum_i3[b] = _vel_pvfmad_vvvvvl(vrsum_i3[b], vrw_i3, vro[b], vrsum_i3[b], vl) ;
	  vrsum_i4[b] = _vel_pvfmad_vvvvvl(vrsum_i4[b], vrw_i4, vro[b], vrsum_i4[b], vl) ;
	  vrsum_i5[b] = _vel_pvfmad_vvvvvl(vrsum_i5[b], vrw_i5, vro[b], vrsum_i5[b], vl) ;
	}
      }
      {
	const int64_t vl = (outDim - o) >> 1 ;

	__vr vrw_i0 = _vel_vld_vssl(8, pWeight+(i+0)*outDim+o, vl) ;
	__vr vrw_i1 = _vel_vld_vssl(8, pWeight+(i+1)*outDim+o, vl) ;
	__vr vrw_i2 = _vel_vld_vssl(8, pWeight+(i+2)*outDim+o, vl) ;
	__vr vrw_i3 = _vel_vld_vssl(8, pWeight+(i+3)*outDim+o, vl) ;
	__vr vrw_i4 = _vel_vld_vssl(8, pWeight+(i+4)*outDim+o, vl) ;
	__vr vrw_i5 = _vel_vld_vssl(8, pWeight+(i+5)*outDim+o, vl) ;

	__vr vro[BATCH] ;
#pragma clang loop unroll(full)
	for(int64_t b=0; b<BATCH; b++) {
	  vro[b] = _vel_vld_vssl(8, pGOut+b*outDim+o, vl) ;
	}

#pragma clang loop unroll(full)
	for(int64_t b=0; b<BATCH; b++) {
	  vrsum_i0[b] = _vel_pvfmad_vvvvvl(vrsum_i0[b], vrw_i0, vro[b], vrsum_i0[b], vl) ;
	  vrsum_i0[b] = _vel_vfadds_vvvl(vrsum_i0[b], _vel_vsll_vvsl(vrsum_i0[b], 32, mvl), mvl) ;
	  vrsum_i0[b] = _vel_vfsums_vvl(vrsum_i0[b], mvl);
	  _vel_vstu_vssl(vrsum_i0[b], 4, pGIn+b*inDim+i+0, 1) ;

	  vrsum_i1[b] = _vel_pvfmad_vvvvvl(vrsum_i1[b], vrw_i1, vro[b], vrsum_i1[b], vl) ;
	  vrsum_i1[b] = _vel_vfadds_vvvl(vrsum_i1[b], _vel_vsll_vvsl(vrsum_i1[b], 32, mvl), mvl) ;
	  vrsum_i1[b] = _vel_vfsums_vvl(vrsum_i1[b], mvl);
	  _vel_vstu_vssl(vrsum_i1[b], 4, pGIn+b*inDim+i+1, 1) ;

	  vrsum_i2[b] = _vel_pvfmad_vvvvvl(vrsum_i2[b], vrw_i2, vro[b], vrsum_i2[b], vl) ;
	  vrsum_i2[b] = _vel_vfadds_vvvl(vrsum_i2[b], _vel_vsll_vvsl(vrsum_i2[b], 32, mvl), mvl) ;
	  vrsum_i2[b] = _vel_vfsums_vvl(vrsum_i2[b], mvl);
	  _vel_vstu_vssl(vrsum_i2[b], 4, pGIn+b*inDim+i+2, 1) ;

	  vrsum_i3[b] = _vel_pvfmad_vvvvvl(vrsum_i3[b], vrw_i3, vro[b], vrsum_i3[b], vl) ;
	  vrsum_i3[b] = _vel_vfadds_vvvl(vrsum_i3[b], _vel_vsll_vvsl(vrsum_i3[b], 32, mvl), mvl) ;
	  vrsum_i3[b] = _vel_vfsums_vvl(vrsum_i3[b], mvl);
	  _vel_vstu_vssl(vrsum_i3[b], 4, pGIn+b*inDim+i+3, 1) ;

	  vrsum_i4[b] = _vel_pvfmad_vvvvvl(vrsum_i4[b], vrw_i4, vro[b], vrsum_i4[b], vl) ;
	  vrsum_i4[b] = _vel_vfadds_vvvl(vrsum_i4[b], _vel_vsll_vvsl(vrsum_i4[b], 32, mvl), mvl) ;
	  vrsum_i4[b] = _vel_vfsums_vvl(vrsum_i4[b], mvl);
	  _vel_vstu_vssl(vrsum_i4[b], 4, pGIn+b*inDim+i+4, 1) ;

	  vrsum_i5[b] = _vel_pvfmad_vvvvvl(vrsum_i5[b], vrw_i5, vro[b], vrsum_i5[b], vl) ;
	  vrsum_i5[b] = _vel_vfadds_vvvl(vrsum_i5[b], _vel_vsll_vvsl(vrsum_i5[b], 32, mvl), mvl) ;
	  vrsum_i5[b] = _vel_vfsums_vvl(vrsum_i5[b], mvl);
	  _vel_vstu_vssl(vrsum_i5[b], 4, pGIn+b*inDim+i+5, 1) ;
	}
      }
    }
    break ;
  case 7 :
    {
      __vr vrsum_i0[BATCH] ;
      __vr vrsum_i1[BATCH] ;
      __vr vrsum_i2[BATCH] ;
      __vr vrsum_i3[BATCH] ;
      __vr vrsum_i4[BATCH] ;
      __vr vrsum_i5[BATCH] ;
      __vr vrsum_i6[BATCH] ;
#pragma clang loop unroll(full)
      for(int64_t b=0; b<BATCH; b++) {
	vrsum_i0[b] = _vel_pvbrd_vsl(0UL, mvl) ;
	vrsum_i1[b] = _vel_pvbrd_vsl(0UL, mvl) ;
	vrsum_i2[b] = _vel_pvbrd_vsl(0UL, mvl) ;
	vrsum_i3[b] = _vel_pvbrd_vsl(0UL, mvl) ;
	vrsum_i4[b] = _vel_pvbrd_vsl(0UL, mvl) ;
	vrsum_i5[b] = _vel_pvbrd_vsl(0UL, mvl) ;
	vrsum_i6[b] = _vel_pvbrd_vsl(0UL, mvl) ;
      }

      int64_t o=0 ;
      for(; o+2*mvl<outDim; o+=2*mvl)
      {
	const int64_t vl = mvl ;

	__vr vrw_i0 = _vel_vld_vssl(8, pWeight+(i+0)*outDim+o, vl) ;
	__vr vrw_i1 = _vel_vld_vssl(8, pWeight+(i+1)*outDim+o, vl) ;
	__vr vrw_i2 = _vel_vld_vssl(8, pWeight+(i+2)*outDim+o, vl) ;
	__vr vrw_i3 = _vel_vld_vssl(8, pWeight+(i+3)*outDim+o, vl) ;
	__vr vrw_i4 = _vel_vld_vssl(8, pWeight+(i+4)*outDim+o, vl) ;
	__vr vrw_i5 = _vel_vld_vssl(8, pWeight+(i+5)*outDim+o, vl) ;
	__vr vrw_i6 = _vel_vld_vssl(8, pWeight+(i+6)*outDim+o, vl) ;

	__vr vro[BATCH] ;
#pragma clang loop unroll(full)
	for(int64_t b=0; b<BATCH; b++) {
	  vro[b] = _vel_vld_vssl(8, pGOut+b*outDim+o, vl) ;
	}

#pragma clang loop unroll(full)
	for(int64_t b=0; b<BATCH; b++) {
	  vrsum_i0[b] = _vel_pvfmad_vvvvvl(vrsum_i0[b], vrw_i0, vro[b], vrsum_i0[b], vl) ;
	  vrsum_i1[b] = _vel_pvfmad_vvvvvl(vrsum_i1[b], vrw_i1, vro[b], vrsum_i1[b], vl) ;
	  vrsum_i2[b] = _vel_pvfmad_vvvvvl(vrsum_i2[b], vrw_i2, vro[b], vrsum_i2[b], vl) ;
	  vrsum_i3[b] = _vel_pvfmad_vvvvvl(vrsum_i3[b], vrw_i3, vro[b], vrsum_i3[b], vl) ;
	  vrsum_i4[b] = _vel_pvfmad_vvvvvl(vrsum_i4[b], vrw_i4, vro[b], vrsum_i4[b], vl) ;
	  vrsum_i5[b] = _vel_pvfmad_vvvvvl(vrsum_i5[b], vrw_i5, vro[b], vrsum_i5[b], vl) ;
	  vrsum_i6[b] = _vel_pvfmad_vvvvvl(vrsum_i6[b], vrw_i6, vro[b], vrsum_i6[b], vl) ;
	}
      }
      {
	const int64_t vl = (outDim - o) >> 1 ;

	__vr vrw_i0 = _vel_vld_vssl(8, pWeight+(i+0)*outDim+o, vl) ;
	__vr vrw_i1 = _vel_vld_vssl(8, pWeight+(i+1)*outDim+o, vl) ;
	__vr vrw_i2 = _vel_vld_vssl(8, pWeight+(i+2)*outDim+o, vl) ;
	__vr vrw_i3 = _vel_vld_vssl(8, pWeight+(i+3)*outDim+o, vl) ;
	__vr vrw_i4 = _vel_vld_vssl(8, pWeight+(i+4)*outDim+o, vl) ;
	__vr vrw_i5 = _vel_vld_vssl(8, pWeight+(i+5)*outDim+o, vl) ;
	__vr vrw_i6 = _vel_vld_vssl(8, pWeight+(i+6)*outDim+o, vl) ;

	__vr vro[BATCH] ;
#pragma clang loop unroll(full)
	for(int64_t b=0; b<BATCH; b++) {
	  vro[b] = _vel_vld_vssl(8, pGOut+b*outDim+o, vl) ;
	}

#pragma clang loop unroll(full)
	for(int64_t b=0; b<BATCH; b++) {
	  vrsum_i0[b] = _vel_pvfmad_vvvvvl(vrsum_i0[b], vrw_i0, vro[b], vrsum_i0[b], vl) ;
	  vrsum_i0[b] = _vel_vfadds_vvvl(vrsum_i0[b], _vel_vsll_vvsl(vrsum_i0[b], 32, mvl), mvl) ;
	  vrsum_i0[b] = _vel_vfsums_vvl(vrsum_i0[b], mvl);
	  _vel_vstu_vssl(vrsum_i0[b], 4, pGIn+b*inDim+i+0, 1) ;

	  vrsum_i1[b] = _vel_pvfmad_vvvvvl(vrsum_i1[b], vrw_i1, vro[b], vrsum_i1[b], vl) ;
	  vrsum_i1[b] = _vel_vfadds_vvvl(vrsum_i1[b], _vel_vsll_vvsl(vrsum_i1[b], 32, mvl), mvl) ;
	  vrsum_i1[b] = _vel_vfsums_vvl(vrsum_i1[b], mvl);
	  _vel_vstu_vssl(vrsum_i1[b], 4, pGIn+b*inDim+i+1, 1) ;

	  vrsum_i2[b] = _vel_pvfmad_vvvvvl(vrsum_i2[b], vrw_i2, vro[b], vrsum_i2[b], vl) ;
	  vrsum_i2[b] = _vel_vfadds_vvvl(vrsum_i2[b], _vel_vsll_vvsl(vrsum_i2[b], 32, mvl), mvl) ;
	  vrsum_i2[b] = _vel_vfsums_vvl(vrsum_i2[b], mvl);
	  _vel_vstu_vssl(vrsum_i2[b], 4, pGIn+b*inDim+i+2, 1) ;

	  vrsum_i3[b] = _vel_pvfmad_vvvvvl(vrsum_i3[b], vrw_i3, vro[b], vrsum_i3[b], vl) ;
	  vrsum_i3[b] = _vel_vfadds_vvvl(vrsum_i3[b], _vel_vsll_vvsl(vrsum_i3[b], 32, mvl), mvl) ;
	  vrsum_i3[b] = _vel_vfsums_vvl(vrsum_i3[b], mvl);
	  _vel_vstu_vssl(vrsum_i3[b], 4, pGIn+b*inDim+i+3, 1) ;

	  vrsum_i4[b] = _vel_pvfmad_vvvvvl(vrsum_i4[b], vrw_i4, vro[b], vrsum_i4[b], vl) ;
	  vrsum_i4[b] = _vel_vfadds_vvvl(vrsum_i4[b], _vel_vsll_vvsl(vrsum_i4[b], 32, mvl), mvl) ;
	  vrsum_i4[b] = _vel_vfsums_vvl(vrsum_i4[b], mvl);
	  _vel_vstu_vssl(vrsum_i4[b], 4, pGIn+b*inDim+i+4, 1) ;

	  vrsum_i5[b] = _vel_pvfmad_vvvvvl(vrsum_i5[b], vrw_i5, vro[b], vrsum_i5[b], vl) ;
	  vrsum_i5[b] = _vel_vfadds_vvvl(vrsum_i5[b], _vel_vsll_vvsl(vrsum_i5[b], 32, mvl), mvl) ;
	  vrsum_i5[b] = _vel_vfsums_vvl(vrsum_i5[b], mvl);
	  _vel_vstu_vssl(vrsum_i5[b], 4, pGIn+b*inDim+i+5, 1) ;

	  vrsum_i6[b] = _vel_pvfmad_vvvvvl(vrsum_i6[b], vrw_i6, vro[b], vrsum_i6[b], vl) ;
	  vrsum_i6[b] = _vel_vfadds_vvvl(vrsum_i6[b], _vel_vsll_vvsl(vrsum_i6[b], 32, mvl), mvl) ;
	  vrsum_i6[b] = _vel_vfsums_vvl(vrsum_i6[b], mvl);
	  _vel_vstu_vssl(vrsum_i6[b], 4, pGIn+b*inDim+i+6, 1) ;
	}
      }
    }
    break ;
  default :
    break ;
  }

  for(; i<inDim; i+=8)
  {
    __vr vrsum_i0[BATCH] ;
    __vr vrsum_i1[BATCH] ;
    __vr vrsum_i2[BATCH] ;
    __vr vrsum_i3[BATCH] ;
    __vr vrsum_i4[BATCH] ;
    __vr vrsum_i5[BATCH] ;
    __vr vrsum_i6[BATCH] ;
    __vr vrsum_i7[BATCH] ;
#pragma clang loop unroll(full)
    for(int64_t b=0; b<BATCH; b++) {
      vrsum_i0[b] = _vel_pvbrd_vsl(0UL, mvl) ;
      vrsum_i1[b] = _vel_pvbrd_vsl(0UL, mvl) ;
      vrsum_i2[b] = _vel_pvbrd_vsl(0UL, mvl) ;
      vrsum_i3[b] = _vel_pvbrd_vsl(0UL, mvl) ;
      vrsum_i4[b] = _vel_pvbrd_vsl(0UL, mvl) ;
      vrsum_i5[b] = _vel_pvbrd_vsl(0UL, mvl) ;
      vrsum_i6[b] = _vel_pvbrd_vsl(0UL, mvl) ;
      vrsum_i7[b] = _vel_pvbrd_vsl(0UL, mvl) ;
    }

    int64_t o=0 ;
    for(; o+2*mvl<outDim; o+=2*mvl)
    {
      const int64_t vl = mvl ;

      __vr vrw_i0 = _vel_vld_vssl(8, pWeight+(i+0)*outDim+o, vl) ;
      __vr vrw_i1 = _vel_vld_vssl(8, pWeight+(i+1)*outDim+o, vl) ;
      __vr vrw_i2 = _vel_vld_vssl(8, pWeight+(i+2)*outDim+o, vl) ;
      __vr vrw_i3 = _vel_vld_vssl(8, pWeight+(i+3)*outDim+o, vl) ;
      __vr vrw_i4 = _vel_vld_vssl(8, pWeight+(i+4)*outDim+o, vl) ;
      __vr vrw_i5 = _vel_vld_vssl(8, pWeight+(i+5)*outDim+o, vl) ;
      __vr vrw_i6 = _vel_vld_vssl(8, pWeight+(i+6)*outDim+o, vl) ;
      __vr vrw_i7 = _vel_vld_vssl(8, pWeight+(i+7)*outDim+o, vl) ;

      __vr vro[BATCH] ;
#pragma clang loop unroll(full)
      for(int64_t b=0; b<BATCH; b++) {
        vro[b] = _vel_vld_vssl(8, pGOut+b*outDim+o, vl) ;
      }

#pragma clang loop unroll(full)
      for(int64_t b=0; b<BATCH; b++) {
	vrsum_i0[b] = _vel_pvfmad_vvvvvl(vrsum_i0[b], vrw_i0, vro[b], vrsum_i0[b], vl) ;
	vrsum_i1[b] = _vel_pvfmad_vvvvvl(vrsum_i1[b], vrw_i1, vro[b], vrsum_i1[b], vl) ;
	vrsum_i2[b] = _vel_pvfmad_vvvvvl(vrsum_i2[b], vrw_i2, vro[b], vrsum_i2[b], vl) ;
	vrsum_i3[b] = _vel_pvfmad_vvvvvl(vrsum_i3[b], vrw_i3, vro[b], vrsum_i3[b], vl) ;
	vrsum_i4[b] = _vel_pvfmad_vvvvvl(vrsum_i4[b], vrw_i4, vro[b], vrsum_i4[b], vl) ;
	vrsum_i5[b] = _vel_pvfmad_vvvvvl(vrsum_i5[b], vrw_i5, vro[b], vrsum_i5[b], vl) ;
	vrsum_i6[b] = _vel_pvfmad_vvvvvl(vrsum_i6[b], vrw_i6, vro[b], vrsum_i6[b], vl) ;
	vrsum_i7[b] = _vel_pvfmad_vvvvvl(vrsum_i7[b], vrw_i7, vro[b], vrsum_i7[b], vl) ;
      }
    }
    {
      const int64_t vl = (outDim - o) >> 1 ;

      __vr vrw_i0 = _vel_vld_vssl(8, pWeight+(i+0)*outDim+o, vl) ;
      __vr vrw_i1 = _vel_vld_vssl(8, pWeight+(i+1)*outDim+o, vl) ;
      __vr vrw_i2 = _vel_vld_vssl(8, pWeight+(i+2)*outDim+o, vl) ;
      __vr vrw_i3 = _vel_vld_vssl(8, pWeight+(i+3)*outDim+o, vl) ;
      __vr vrw_i4 = _vel_vld_vssl(8, pWeight+(i+4)*outDim+o, vl) ;
      __vr vrw_i5 = _vel_vld_vssl(8, pWeight+(i+5)*outDim+o, vl) ;
      __vr vrw_i6 = _vel_vld_vssl(8, pWeight+(i+6)*outDim+o, vl) ;
      __vr vrw_i7 = _vel_vld_vssl(8, pWeight+(i+7)*outDim+o, vl) ;

      __vr vro[BATCH] ;
#pragma clang loop unroll(full)
      for(int64_t b=0; b<BATCH; b++) {
        vro[b] = _vel_vld_vssl(8, pGOut+b*outDim+o, vl) ;
      }

#pragma clang loop unroll(full)
      for(int64_t b=0; b<BATCH; b++) {
	vrsum_i0[b] = _vel_pvfmad_vvvvvl(vrsum_i0[b], vrw_i0, vro[b], vrsum_i0[b], vl) ;
	vrsum_i0[b] = _vel_vfadds_vvvl(vrsum_i0[b], _vel_vsll_vvsl(vrsum_i0[b], 32, mvl), mvl) ;
	vrsum_i0[b] = _vel_vfsums_vvl(vrsum_i0[b], mvl);
	_vel_vstu_vssl(vrsum_i0[b], 4, pGIn+b*inDim+i+0, 1) ;

	vrsum_i1[b] = _vel_pvfmad_vvvvvl(vrsum_i1[b], vrw_i1, vro[b], vrsum_i1[b], vl) ;
	vrsum_i1[b] = _vel_vfadds_vvvl(vrsum_i1[b], _vel_vsll_vvsl(vrsum_i1[b], 32, mvl), mvl) ;
	vrsum_i1[b] = _vel_vfsums_vvl(vrsum_i1[b], mvl);
	_vel_vstu_vssl(vrsum_i1[b], 4, pGIn+b*inDim+i+1, 1) ;

	vrsum_i2[b] = _vel_pvfmad_vvvvvl(vrsum_i2[b], vrw_i2, vro[b], vrsum_i2[b], vl) ;
	vrsum_i2[b] = _vel_vfadds_vvvl(vrsum_i2[b], _vel_vsll_vvsl(vrsum_i2[b], 32, mvl), mvl) ;
	vrsum_i2[b] = _vel_vfsums_vvl(vrsum_i2[b], mvl);
	_vel_vstu_vssl(vrsum_i2[b], 4, pGIn+b*inDim+i+2, 1) ;

	vrsum_i3[b] = _vel_pvfmad_vvvvvl(vrsum_i3[b], vrw_i3, vro[b], vrsum_i3[b], vl) ;
	vrsum_i3[b] = _vel_vfadds_vvvl(vrsum_i3[b], _vel_vsll_vvsl(vrsum_i3[b], 32, mvl), mvl) ;
	vrsum_i3[b] = _vel_vfsums_vvl(vrsum_i3[b], mvl);
	_vel_vstu_vssl(vrsum_i3[b], 4, pGIn+b*inDim+i+3, 1) ;

	vrsum_i4[b] = _vel_pvfmad_vvvvvl(vrsum_i4[b], vrw_i4, vro[b], vrsum_i4[b], vl) ;
	vrsum_i4[b] = _vel_vfadds_vvvl(vrsum_i4[b], _vel_vsll_vvsl(vrsum_i4[b], 32, mvl), mvl) ;
	vrsum_i4[b] = _vel_vfsums_vvl(vrsum_i4[b], mvl);
	_vel_vstu_vssl(vrsum_i4[b], 4, pGIn+b*inDim+i+4, 1) ;

	vrsum_i5[b] = _vel_pvfmad_vvvvvl(vrsum_i5[b], vrw_i5, vro[b], vrsum_i5[b], vl) ;
	vrsum_i5[b] = _vel_vfadds_vvvl(vrsum_i5[b], _vel_vsll_vvsl(vrsum_i5[b], 32, mvl), mvl) ;
	vrsum_i5[b] = _vel_vfsums_vvl(vrsum_i5[b], mvl);
	_vel_vstu_vssl(vrsum_i5[b], 4, pGIn+b*inDim+i+5, 1) ;

	vrsum_i6[b] = _vel_pvfmad_vvvvvl(vrsum_i6[b], vrw_i6, vro[b], vrsum_i6[b], vl) ;
	vrsum_i6[b] = _vel_vfadds_vvvl(vrsum_i6[b], _vel_vsll_vvsl(vrsum_i6[b], 32, mvl), mvl) ;
	vrsum_i6[b] = _vel_vfsums_vvl(vrsum_i6[b], mvl);
	_vel_vstu_vssl(vrsum_i6[b], 4, pGIn+b*inDim+i+6, 1) ;

	vrsum_i7[b] = _vel_pvfmad_vvvvvl(vrsum_i7[b], vrw_i7, vro[b], vrsum_i7[b], vl) ;
	vrsum_i7[b] = _vel_vfadds_vvvl(vrsum_i7[b], _vel_vsll_vvsl(vrsum_i7[b], 32, mvl), mvl) ;
	vrsum_i7[b] = _vel_vfsums_vvl(vrsum_i7[b], mvl);
	_vel_vstu_vssl(vrsum_i7[b], 4, pGIn+b*inDim+i+7, 1) ;
      }
    }
  }
}

extern "C"
vednnError_t vednnLinearBackwardData_o2X_woaligned(
    const uint64_t			inDim,
    const uint64_t			outDim,
    const uint64_t			nBatch,
    const void * __restrict__		pDataGradOut,
    const void * __restrict__		pDataWeight,
    void * __restrict__			pDataGradIn
)
{
  const float * __restrict__ pGOut   = (const float * __restrict__) pDataGradOut;
  const float * __restrict__ pWeight = (const float * __restrict__) pDataWeight;
  float * __restrict__ const pGIn    = (float * __restrict__ const) pDataGradIn;

  int64_t n=0;
  int64_t batchRemain = nBatch % 4 ;

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
  case 1:
    func<1>(inDim, outDim, pGOut+n*outDim, pWeight, pGIn+n*inDim, mvl ) ;
    n+=1 ;
    break ;
  case 2:
    func<2>(inDim, outDim, pGOut+n*outDim, pWeight, pGIn+n*inDim, mvl ) ;
    n+=2 ;
    break ;
  case 3:
    func<3>(inDim, outDim, pGOut+n*outDim, pWeight, pGIn+n*inDim, mvl ) ;
    n+=3;
    break ;
  default : break ;
  }
  for(; n<nBatch; n+=4) {
    func<4>(inDim, outDim, pGOut+n*outDim, pWeight, pGIn+n*inDim, mvl ) ;
  }

  return VEDNN_SUCCESS ;
}

