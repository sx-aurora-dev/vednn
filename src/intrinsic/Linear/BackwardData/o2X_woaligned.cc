#include <stdint.h>
#include <float.h>

#include <stdio.h>

#include "vednn.h"

#include "velintrin.h"
#define VLEN	(256)

template<int BATCH>
static inline void func(
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
      __vr vrsum_b0_i0 = _vel_vbrdl_vsl(0UL, VLEN) ;
      __vr vrsum_b1_i0 = _vel_vbrdl_vsl(0UL, VLEN) ;
      __vr vrsum_b2_i0 = _vel_vbrdl_vsl(0UL, VLEN) ;
      __vr vrsum_b3_i0 = _vel_vbrdl_vsl(0UL, VLEN) ;

      int64_t o=0 ;
      for(; o+2*VLEN<outDim; o+=2*VLEN)
      {
        const int64_t vl = VLEN ;

        __vr vrw_i0 = _vel_vld_vssl(8, pWeight+(i  )*outDim+o, vl) ;

        __vr vro_b0 = _vel_vld_vssl(8, pGOut+0*outDim+o, vl) ;
        __vr vro_b1 = _vel_vld_vssl(8, pGOut+1*outDim+o, vl) ;
        __vr vro_b2 = _vel_vld_vssl(8, pGOut+2*outDim+o, vl) ;
        __vr vro_b3 = _vel_vld_vssl(8, pGOut+3*outDim+o, vl) ;

        if(BATCH>=1) {
	  vrsum_b0_i0 = _vel_pvfmad_vvvvvl(vrsum_b0_i0, vrw_i0, vro_b0, vrsum_b0_i0, vl) ;
        }
        if(BATCH>=2) {
	  vrsum_b1_i0 = _vel_pvfmad_vvvvvl(vrsum_b1_i0, vrw_i0, vro_b1, vrsum_b1_i0, vl) ;
        }
        if(BATCH>=3) {
	  vrsum_b2_i0 = _vel_pvfmad_vvvvvl(vrsum_b2_i0, vrw_i0, vro_b2, vrsum_b2_i0, vl) ;
        }
        if(BATCH>=4) {
	  vrsum_b3_i0 = _vel_pvfmad_vvvvvl(vrsum_b3_i0, vrw_i0, vro_b3, vrsum_b3_i0, vl) ;
        }
      }
      {
        const int64_t vl = (outDim - o) >> 1 ;

        __vr vrw_i0 = _vel_vld_vssl(8, pWeight+(i  )*outDim+o, vl) ;

        __vr vro_b0 = _vel_vld_vssl(8, pGOut+0*outDim+o, vl) ;
        __vr vro_b1 = _vel_vld_vssl(8, pGOut+1*outDim+o, vl) ;
        __vr vro_b2 = _vel_vld_vssl(8, pGOut+2*outDim+o, vl) ;
        __vr vro_b3 = _vel_vld_vssl(8, pGOut+3*outDim+o, vl) ;

        if(BATCH>=1) {
	  vrsum_b0_i0 = _vel_pvfmad_vvvvvl(vrsum_b0_i0, vrw_i0, vro_b0, vrsum_b0_i0, vl) ;

	  vrsum_b0_i0 = _vel_vfadds_vvvl(vrsum_b0_i0, _vel_vsll_vvsl(vrsum_b0_i0, 32, VLEN), VLEN) ;
	  vrsum_b0_i0 = _vel_vfsums_vvl(vrsum_b0_i0, VLEN);

	  _vel_vstu_vssl(vrsum_b0_i0, 4, pGIn+0*inDim+i, 1) ;
        }
        if(BATCH>=2) {
	  vrsum_b1_i0 = _vel_pvfmad_vvvvvl(vrsum_b1_i0, vrw_i0, vro_b1, vrsum_b1_i0, vl) ;

	  vrsum_b1_i0 = _vel_vfadds_vvvl(vrsum_b1_i0, _vel_vsll_vvsl(vrsum_b1_i0, 32, VLEN), VLEN) ;
	  vrsum_b1_i0 = _vel_vfsums_vvl(vrsum_b1_i0, VLEN);

	  _vel_vstu_vssl(vrsum_b1_i0, 4, pGIn+1*inDim+i, 1) ;
        }
        if(BATCH>=3) {
	  vrsum_b2_i0 = _vel_pvfmad_vvvvvl(vrsum_b2_i0, vrw_i0, vro_b2, vrsum_b2_i0, vl) ;

	  vrsum_b2_i0 = _vel_vfadds_vvvl(vrsum_b2_i0, _vel_vsll_vvsl(vrsum_b2_i0, 32, VLEN), VLEN) ;
	  vrsum_b2_i0 = _vel_vfsums_vvl(vrsum_b2_i0, VLEN);

	  _vel_vstu_vssl(vrsum_b2_i0, 4, pGIn+2*inDim+i, 1) ;
        }
        if(BATCH>=4) {
	  vrsum_b3_i0 = _vel_pvfmad_vvvvvl(vrsum_b3_i0, vrw_i0, vro_b3, vrsum_b3_i0, vl) ;

	  vrsum_b3_i0 = _vel_vfadds_vvvl(vrsum_b3_i0, _vel_vsll_vvsl(vrsum_b3_i0, 32, VLEN), VLEN) ;
	  vrsum_b3_i0 = _vel_vfsums_vvl(vrsum_b3_i0, VLEN);

	  _vel_vstu_vssl(vrsum_b3_i0, 4, pGIn+3*inDim+i, 1) ;
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
  for(; i<inDim; i+=4)
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
    __vr vrsum_b0_i3 = _vel_vbrdl_vsl(0UL, VLEN) ;
    __vr vrsum_b1_i3 = _vel_vbrdl_vsl(0UL, VLEN) ;
    __vr vrsum_b2_i3 = _vel_vbrdl_vsl(0UL, VLEN) ;
    __vr vrsum_b3_i3 = _vel_vbrdl_vsl(0UL, VLEN) ;

    int64_t o=0 ;
    for(; o+2*VLEN<outDim; o+=2*VLEN)
    {
      const int64_t vl = VLEN ;

      __vr vrw_i0 = _vel_vld_vssl(8, pWeight+(i  )*outDim+o, vl) ;
      __vr vrw_i1 = _vel_vld_vssl(8, pWeight+(i+1)*outDim+o, vl) ;
      __vr vrw_i2 = _vel_vld_vssl(8, pWeight+(i+2)*outDim+o, vl) ;
      __vr vrw_i3 = _vel_vld_vssl(8, pWeight+(i+3)*outDim+o, vl) ;

      __vr vro_b0 = _vel_vld_vssl(8, pGOut+0*outDim+o, vl) ;
      __vr vro_b1 = _vel_vld_vssl(8, pGOut+1*outDim+o, vl) ;
      __vr vro_b2 = _vel_vld_vssl(8, pGOut+2*outDim+o, vl) ;
      __vr vro_b3 = _vel_vld_vssl(8, pGOut+3*outDim+o, vl) ;

      if(BATCH>=1) {
	vrsum_b0_i0 = _vel_pvfmad_vvvvvl(vrsum_b0_i0, vrw_i0, vro_b0, vrsum_b0_i0, vl) ;
	vrsum_b0_i1 = _vel_pvfmad_vvvvvl(vrsum_b0_i1, vrw_i1, vro_b0, vrsum_b0_i1, vl) ;
	vrsum_b0_i2 = _vel_pvfmad_vvvvvl(vrsum_b0_i2, vrw_i2, vro_b0, vrsum_b0_i2, vl) ;
	vrsum_b0_i3 = _vel_pvfmad_vvvvvl(vrsum_b0_i3, vrw_i3, vro_b0, vrsum_b0_i3, vl) ;
      }
      if(BATCH>=2) {
	vrsum_b1_i0 = _vel_pvfmad_vvvvvl(vrsum_b1_i0, vrw_i0, vro_b1, vrsum_b1_i0, vl) ;
	vrsum_b1_i1 = _vel_pvfmad_vvvvvl(vrsum_b1_i1, vrw_i1, vro_b1, vrsum_b1_i1, vl) ;
	vrsum_b1_i2 = _vel_pvfmad_vvvvvl(vrsum_b1_i2, vrw_i2, vro_b1, vrsum_b1_i2, vl) ;
	vrsum_b1_i3 = _vel_pvfmad_vvvvvl(vrsum_b1_i3, vrw_i3, vro_b1, vrsum_b1_i3, vl) ;
      }
      if(BATCH>=3) {
	vrsum_b2_i0 = _vel_pvfmad_vvvvvl(vrsum_b2_i0, vrw_i0, vro_b2, vrsum_b2_i0, vl) ;
	vrsum_b2_i1 = _vel_pvfmad_vvvvvl(vrsum_b2_i1, vrw_i1, vro_b2, vrsum_b2_i1, vl) ;
	vrsum_b2_i2 = _vel_pvfmad_vvvvvl(vrsum_b2_i2, vrw_i2, vro_b2, vrsum_b2_i2, vl) ;
	vrsum_b2_i3 = _vel_pvfmad_vvvvvl(vrsum_b2_i3, vrw_i3, vro_b2, vrsum_b2_i3, vl) ;
      }
      if(BATCH>=4) {
	vrsum_b3_i0 = _vel_pvfmad_vvvvvl(vrsum_b3_i0, vrw_i0, vro_b3, vrsum_b3_i0, vl) ;
	vrsum_b3_i1 = _vel_pvfmad_vvvvvl(vrsum_b3_i1, vrw_i1, vro_b3, vrsum_b3_i1, vl) ;
	vrsum_b3_i2 = _vel_pvfmad_vvvvvl(vrsum_b3_i2, vrw_i2, vro_b3, vrsum_b3_i2, vl) ;
	vrsum_b3_i3 = _vel_pvfmad_vvvvvl(vrsum_b3_i3, vrw_i3, vro_b3, vrsum_b3_i3, vl) ;
      }
    }
    {
      const int64_t vl = (outDim - o) >> 1 ;

      __vr vrw_i0 = _vel_vld_vssl(8, pWeight+(i  )*outDim+o, vl) ;
      __vr vrw_i1 = _vel_vld_vssl(8, pWeight+(i+1)*outDim+o, vl) ;
      __vr vrw_i2 = _vel_vld_vssl(8, pWeight+(i+2)*outDim+o, vl) ;
      __vr vrw_i3 = _vel_vld_vssl(8, pWeight+(i+3)*outDim+o, vl) ;

      __vr vro_b0 = _vel_vld_vssl(8, pGOut+0*outDim+o, vl) ;
      __vr vro_b1 = _vel_vld_vssl(8, pGOut+1*outDim+o, vl) ;
      __vr vro_b2 = _vel_vld_vssl(8, pGOut+2*outDim+o, vl) ;
      __vr vro_b3 = _vel_vld_vssl(8, pGOut+3*outDim+o, vl) ;

      if(BATCH>=1) {
	vrsum_b0_i0 = _vel_pvfmad_vvvvvl(vrsum_b0_i0, vrw_i0, vro_b0, vrsum_b0_i0, vl) ;
	vrsum_b0_i1 = _vel_pvfmad_vvvvvl(vrsum_b0_i1, vrw_i1, vro_b0, vrsum_b0_i1, vl) ;
	vrsum_b0_i2 = _vel_pvfmad_vvvvvl(vrsum_b0_i2, vrw_i2, vro_b0, vrsum_b0_i2, vl) ;
	vrsum_b0_i3 = _vel_pvfmad_vvvvvl(vrsum_b0_i3, vrw_i3, vro_b0, vrsum_b0_i3, vl) ;

	vrsum_b0_i0 = _vel_vfadds_vvvl(vrsum_b0_i0, _vel_vsll_vvsl(vrsum_b0_i0, 32, VLEN), VLEN) ;
	vrsum_b0_i1 = _vel_vfadds_vvvl(vrsum_b0_i1, _vel_vsll_vvsl(vrsum_b0_i1, 32, VLEN), VLEN) ;
	vrsum_b0_i2 = _vel_vfadds_vvvl(vrsum_b0_i2, _vel_vsll_vvsl(vrsum_b0_i2, 32, VLEN), VLEN) ;
	vrsum_b0_i3 = _vel_vfadds_vvvl(vrsum_b0_i3, _vel_vsll_vvsl(vrsum_b0_i3, 32, VLEN), VLEN) ;
	vrsum_b0_i0 = _vel_vfsums_vvl(vrsum_b0_i0, VLEN);
	vrsum_b0_i1 = _vel_vfsums_vvl(vrsum_b0_i1, VLEN);
	vrsum_b0_i2 = _vel_vfsums_vvl(vrsum_b0_i2, VLEN);
	vrsum_b0_i3 = _vel_vfsums_vvl(vrsum_b0_i3, VLEN);

	_vel_vstu_vssl(vrsum_b0_i0, 4, pGIn+0*inDim+i, 1) ;
	_vel_vstu_vssl(vrsum_b0_i1, 4, pGIn+0*inDim+i+1, 1) ;
	_vel_vstu_vssl(vrsum_b0_i2, 4, pGIn+0*inDim+i+2, 1) ;
	_vel_vstu_vssl(vrsum_b0_i3, 4, pGIn+0*inDim+i+3, 1) ;
      }
      if(BATCH>=2) {
	vrsum_b1_i0 = _vel_pvfmad_vvvvvl(vrsum_b1_i0, vrw_i0, vro_b1, vrsum_b1_i0, vl) ;
	vrsum_b1_i1 = _vel_pvfmad_vvvvvl(vrsum_b1_i1, vrw_i1, vro_b1, vrsum_b1_i1, vl) ;
	vrsum_b1_i2 = _vel_pvfmad_vvvvvl(vrsum_b1_i2, vrw_i2, vro_b1, vrsum_b1_i2, vl) ;
	vrsum_b1_i3 = _vel_pvfmad_vvvvvl(vrsum_b1_i3, vrw_i3, vro_b1, vrsum_b1_i3, vl) ;

	vrsum_b1_i0 = _vel_vfadds_vvvl(vrsum_b1_i0, _vel_vsll_vvsl(vrsum_b1_i0, 32, VLEN), VLEN) ;
	vrsum_b1_i1 = _vel_vfadds_vvvl(vrsum_b1_i1, _vel_vsll_vvsl(vrsum_b1_i1, 32, VLEN), VLEN) ;
	vrsum_b1_i2 = _vel_vfadds_vvvl(vrsum_b1_i2, _vel_vsll_vvsl(vrsum_b1_i2, 32, VLEN), VLEN) ;
	vrsum_b1_i3 = _vel_vfadds_vvvl(vrsum_b1_i3, _vel_vsll_vvsl(vrsum_b1_i3, 32, VLEN), VLEN) ;
	vrsum_b1_i0 = _vel_vfsums_vvl(vrsum_b1_i0, VLEN);
	vrsum_b1_i1 = _vel_vfsums_vvl(vrsum_b1_i1, VLEN);
	vrsum_b1_i2 = _vel_vfsums_vvl(vrsum_b1_i2, VLEN);
	vrsum_b1_i3 = _vel_vfsums_vvl(vrsum_b1_i3, VLEN);

	_vel_vstu_vssl(vrsum_b1_i0, 4, pGIn+1*inDim+i, 1) ;
	_vel_vstu_vssl(vrsum_b1_i1, 4, pGIn+1*inDim+i+1, 1) ;
	_vel_vstu_vssl(vrsum_b1_i2, 4, pGIn+1*inDim+i+2, 1) ;
	_vel_vstu_vssl(vrsum_b1_i3, 4, pGIn+1*inDim+i+3, 1) ;
      }
      if(BATCH>=3) {
	vrsum_b2_i0 = _vel_pvfmad_vvvvvl(vrsum_b2_i0, vrw_i0, vro_b2, vrsum_b2_i0, vl) ;
	vrsum_b2_i1 = _vel_pvfmad_vvvvvl(vrsum_b2_i1, vrw_i1, vro_b2, vrsum_b2_i1, vl) ;
	vrsum_b2_i2 = _vel_pvfmad_vvvvvl(vrsum_b2_i2, vrw_i2, vro_b2, vrsum_b2_i2, vl) ;
	vrsum_b2_i3 = _vel_pvfmad_vvvvvl(vrsum_b2_i3, vrw_i3, vro_b2, vrsum_b2_i3, vl) ;

	vrsum_b2_i0 = _vel_vfadds_vvvl(vrsum_b2_i0, _vel_vsll_vvsl(vrsum_b2_i0, 32, VLEN), VLEN) ;
	vrsum_b2_i1 = _vel_vfadds_vvvl(vrsum_b2_i1, _vel_vsll_vvsl(vrsum_b2_i1, 32, VLEN), VLEN) ;
	vrsum_b2_i2 = _vel_vfadds_vvvl(vrsum_b2_i2, _vel_vsll_vvsl(vrsum_b2_i2, 32, VLEN), VLEN) ;
	vrsum_b2_i3 = _vel_vfadds_vvvl(vrsum_b2_i3, _vel_vsll_vvsl(vrsum_b2_i3, 32, VLEN), VLEN) ;
	vrsum_b2_i0 = _vel_vfsums_vvl(vrsum_b2_i0, VLEN);
	vrsum_b2_i1 = _vel_vfsums_vvl(vrsum_b2_i1, VLEN);
	vrsum_b2_i2 = _vel_vfsums_vvl(vrsum_b2_i2, VLEN);
	vrsum_b2_i3 = _vel_vfsums_vvl(vrsum_b2_i3, VLEN);

	_vel_vstu_vssl(vrsum_b2_i0, 4, pGIn+2*inDim+i, 1) ;
	_vel_vstu_vssl(vrsum_b2_i1, 4, pGIn+2*inDim+i+1, 1) ;
	_vel_vstu_vssl(vrsum_b2_i2, 4, pGIn+2*inDim+i+2, 1) ;
	_vel_vstu_vssl(vrsum_b2_i3, 4, pGIn+2*inDim+i+3, 1) ;
      }
      if(BATCH>=4) {
	vrsum_b3_i0 = _vel_pvfmad_vvvvvl(vrsum_b3_i0, vrw_i0, vro_b3, vrsum_b3_i0, vl) ;
	vrsum_b3_i1 = _vel_pvfmad_vvvvvl(vrsum_b3_i1, vrw_i1, vro_b3, vrsum_b3_i1, vl) ;
	vrsum_b3_i2 = _vel_pvfmad_vvvvvl(vrsum_b3_i2, vrw_i2, vro_b3, vrsum_b3_i2, vl) ;
	vrsum_b3_i3 = _vel_pvfmad_vvvvvl(vrsum_b3_i3, vrw_i3, vro_b3, vrsum_b3_i3, vl) ;

	vrsum_b3_i0 = _vel_vfadds_vvvl(vrsum_b3_i0, _vel_vsll_vvsl(vrsum_b3_i0, 32, VLEN), VLEN) ;
	vrsum_b3_i1 = _vel_vfadds_vvvl(vrsum_b3_i1, _vel_vsll_vvsl(vrsum_b3_i1, 32, VLEN), VLEN) ;
	vrsum_b3_i2 = _vel_vfadds_vvvl(vrsum_b3_i2, _vel_vsll_vvsl(vrsum_b3_i2, 32, VLEN), VLEN) ;
	vrsum_b3_i3 = _vel_vfadds_vvvl(vrsum_b3_i3, _vel_vsll_vvsl(vrsum_b3_i3, 32, VLEN), VLEN) ;
	vrsum_b3_i0 = _vel_vfsums_vvl(vrsum_b3_i0, VLEN);
	vrsum_b3_i1 = _vel_vfsums_vvl(vrsum_b3_i1, VLEN);
	vrsum_b3_i2 = _vel_vfsums_vvl(vrsum_b3_i2, VLEN);
	vrsum_b3_i3 = _vel_vfsums_vvl(vrsum_b3_i3, VLEN);

	_vel_vstu_vssl(vrsum_b3_i0, 4, pGIn+3*inDim+i, 1) ;
	_vel_vstu_vssl(vrsum_b3_i1, 4, pGIn+3*inDim+i+1, 1) ;
	_vel_vstu_vssl(vrsum_b3_i2, 4, pGIn+3*inDim+i+2, 1) ;
	_vel_vstu_vssl(vrsum_b3_i3, 4, pGIn+3*inDim+i+3, 1) ;
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

  switch( batchRemain ) {
  case 1:
    func<1>(inDim, outDim, pGOut+n*outDim, pWeight, pGIn+n*inDim ) ;
    n+=1 ;
    break ;
  case 2:
    func<2>(inDim, outDim, pGOut+n*outDim, pWeight, pGIn+n*inDim ) ;
    n+=2 ;
    break ;
  case 3:
    func<3>(inDim, outDim, pGOut+n*outDim, pWeight, pGIn+n*inDim ) ;
    n+=3;
    break ;
  default : break ;
  }
  for(; n<nBatch; n+=4) {
    func<4>(inDim, outDim, pGOut+n*outDim, pWeight, pGIn+n*inDim ) ;
  }

  return VEDNN_SUCCESS ;
}

