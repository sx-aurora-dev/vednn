#include <stdint.h>
#include <float.h>

#include <stdio.h>

#include "vednn.h"

#include "velintrin.h"
#define VLEN	(256)

// fixme : use new intrinsic
#if 0
static inline uint64_t pack_f32p(const float *pInU, const float *pInL) {
  const uint64_t InU = *((uint32_t*)(pInU)) ;
  const uint64_t InL = *((uint32_t*)(pInL)) ;
  return (InU << 32) | InL ;
}
#else
#define pack_f32p __builtin_ve_pack_f32p
#endif

static inline void b1(
  const uint64_t	inDim,
  const uint64_t	outDim,
  const float * 	pIn,
  const float * 	pWeight,
  float * 		pOut
)
{
  for(int64_t o=0; o<outDim; o+=VLEN) {
    const int64_t vl = outDim-o < VLEN ? outDim - o : VLEN ;
    __vr vrsum_b0 = _vel_vbrdl_vsl(0UL, vl) ;

    int64_t i=0;
    if((inDim & 0x01)==1) {
      __vr vrw = _vel_vldu_vssl(4, pWeight+i*outDim+o, vl) ;
      vrsum_b0 = _vel_vfmads_vvsvl(vrsum_b0, pIn[0*inDim+i], vrw, vl) ;
      i+=1 ;
    }
    if(((inDim>>1) & 0x01)==1) {
      __vr vrw_i0 = _vel_vldu_vssl(4, pWeight+(i  )*outDim+o, vl) ;
      __vr vrw_i1 = _vel_vldu_vssl(4, pWeight+(i+1)*outDim+o, vl) ;

      __vr vrw_i01 = _vel_vshf_vvvsl(vrw_i0, vrw_i1, VE_VSHUFFLE_YUZU, vl) ;

      const uint64_t in_i01_b0 = pack_f32p(pIn+0*inDim+i+0, pIn+0*inDim+i+1) ;
      vrsum_b0 = _vel_pvfmad_vvsvl(vrsum_b0, in_i01_b0, vrw_i01, vl) ;
      i+=2 ;
    }
    for(; i<inDim; i+=4 ) {
      __vr vrw_i0 = _vel_vldu_vssl(4, pWeight+(i  )*outDim+o, vl) ;
      __vr vrw_i1 = _vel_vldu_vssl(4, pWeight+(i+1)*outDim+o, vl) ;
      __vr vrw_i2 = _vel_vldu_vssl(4, pWeight+(i+2)*outDim+o, vl) ;
      __vr vrw_i3 = _vel_vldu_vssl(4, pWeight+(i+3)*outDim+o, vl) ;

      __vr vrw_i01 = _vel_vshf_vvvsl(vrw_i0, vrw_i1, VE_VSHUFFLE_YUZU, vl) ;
      __vr vrw_i23 = _vel_vshf_vvvsl(vrw_i2, vrw_i3, VE_VSHUFFLE_YUZU, vl) ;

      const uint64_t in_i01_b0 = pack_f32p(pIn+0*inDim+i+0, pIn+0*inDim+i+1) ;
      vrsum_b0 = _vel_pvfmad_vvsvl(vrsum_b0, in_i01_b0, vrw_i01, vl) ;
      const uint64_t in_i23_b0 = pack_f32p(pIn+0*inDim+i+2, pIn+0*inDim+i+3) ;
      vrsum_b0 = _vel_pvfmad_vvsvl(vrsum_b0, in_i23_b0, vrw_i23, vl) ;
    }

    vrsum_b0 = _vel_vfadds_vvvl(vrsum_b0, _vel_vsll_vvsl(vrsum_b0,32,vl),vl) ;
    _vel_vstu_vssl(vrsum_b0, 4, pOut+0*outDim+o, vl) ;
  }
}

static inline void b2(
  const uint64_t	inDim,
  const uint64_t	outDim,
  const float * 	pIn,
  const float * 	pWeight,
  float * 		pOut
)
{
  for(int64_t o=0; o<outDim; o+=VLEN) {
    const int64_t vl = outDim-o < VLEN ? outDim - o : VLEN ;
    __vr vrsum_b0 = _vel_vbrdl_vsl(0UL,vl) ;
    __vr vrsum_b1 = _vel_vbrdl_vsl(0UL,vl) ;

    int64_t i=0;
    if((inDim & 0x01)==1) {
      __vr vrw = _vel_vldu_vssl(4, pWeight+i*outDim+o, vl) ;
      vrsum_b0 = _vel_vfmads_vvsvl(vrsum_b0, pIn[0*inDim+i], vrw, vl) ;
      vrsum_b1 = _vel_vfmads_vvsvl(vrsum_b1, pIn[1*inDim+i], vrw, vl) ;

      i+=1 ;
    }
    if(((inDim>>1) & 0x01)==1) {
      __vr vrw_i0 = _vel_vldu_vssl(4, pWeight+(i  )*outDim+o, vl) ;
      __vr vrw_i1 = _vel_vldu_vssl(4, pWeight+(i+1)*outDim+o, vl) ;

      __vr vrw_i01 = _vel_vshf_vvvsl(vrw_i0, vrw_i1, VE_VSHUFFLE_YUZU, vl) ;

      const uint64_t in_i01_b0 = pack_f32p(pIn+0*inDim+i+0, pIn+0*inDim+i+1) ;
      const uint64_t in_i01_b1 = pack_f32p(pIn+1*inDim+i+0, pIn+1*inDim+i+1) ;

      vrsum_b0 = _vel_pvfmad_vvsvl(vrsum_b0, in_i01_b0, vrw_i01, vl) ;
      vrsum_b1 = _vel_pvfmad_vvsvl(vrsum_b1, in_i01_b1, vrw_i01, vl) ;

      i+=2 ;
    }
    for(; i<inDim; i+=4 ) {
      __vr vrw_i0 = _vel_vldu_vssl(4, pWeight+(i  )*outDim+o, vl) ;
      __vr vrw_i1 = _vel_vldu_vssl(4, pWeight+(i+1)*outDim+o, vl) ;
      __vr vrw_i2 = _vel_vldu_vssl(4, pWeight+(i+2)*outDim+o, vl) ;
      __vr vrw_i3 = _vel_vldu_vssl(4, pWeight+(i+3)*outDim+o, vl) ;

      __vr vrw_i01 = _vel_vshf_vvvsl(vrw_i0, vrw_i1, VE_VSHUFFLE_YUZU, vl) ;
      __vr vrw_i23 = _vel_vshf_vvvsl(vrw_i2, vrw_i3, VE_VSHUFFLE_YUZU, vl) ;

      const uint64_t in_i01_b0 = pack_f32p(pIn+0*inDim+i+0, pIn+0*inDim+i+1) ;
      const uint64_t in_i01_b1 = pack_f32p(pIn+1*inDim+i+0, pIn+1*inDim+i+1) ;

      vrsum_b0 = _vel_pvfmad_vvsvl(vrsum_b0, in_i01_b0, vrw_i01, vl) ;
      vrsum_b1 = _vel_pvfmad_vvsvl(vrsum_b1, in_i01_b1, vrw_i01, vl) ;

      const uint64_t in_i23_b0 = pack_f32p(pIn+0*inDim+i+2, pIn+0*inDim+i+3) ;
      const uint64_t in_i23_b1 = pack_f32p(pIn+1*inDim+i+2, pIn+1*inDim+i+3) ;

      vrsum_b0 = _vel_pvfmad_vvsvl(vrsum_b0, in_i23_b0, vrw_i23, vl) ;
      vrsum_b1 = _vel_pvfmad_vvsvl(vrsum_b1, in_i23_b1, vrw_i23, vl) ;
    }

    vrsum_b0 = _vel_vfadds_vvvl(vrsum_b0, _vel_vsll_vvsl(vrsum_b0,32,vl), vl) ;
    _vel_vstu_vssl(vrsum_b0, 4, pOut+0*outDim+o, vl) ;
    vrsum_b1 = _vel_vfadds_vvvl(vrsum_b1, _vel_vsll_vvsl(vrsum_b1,32,vl), vl) ;
    _vel_vstu_vssl(vrsum_b1, 4, pOut+1*outDim+o, vl) ;
  }
}

static inline void b3(
  const uint64_t	inDim,
  const uint64_t	outDim,
  const float * 	pIn,
  const float * 	pWeight,
  float * 		pOut
)
{
  for(int64_t o=0; o<outDim; o+=VLEN) {
    const int64_t vl = outDim-o < VLEN ? outDim - o : VLEN ;
    __vr vrsum_b0 = _vel_vbrdl_vsl(0UL,vl) ;
    __vr vrsum_b1 = _vel_vbrdl_vsl(0UL,vl) ;
    __vr vrsum_b2 = _vel_vbrdl_vsl(0UL,vl) ;

    int64_t i=0;
    if((inDim & 0x01)==1) {
      __vr vrw = _vel_vldu_vssl(4, pWeight+i*outDim+o, vl) ;
      vrsum_b0 = _vel_vfmads_vvsvl(vrsum_b0, pIn[0*inDim+i], vrw, vl) ;
      vrsum_b1 = _vel_vfmads_vvsvl(vrsum_b1, pIn[1*inDim+i], vrw, vl) ;
      vrsum_b2 = _vel_vfmads_vvsvl(vrsum_b2, pIn[2*inDim+i], vrw, vl) ;

      i+=1 ;
    }
    if(((inDim>>1) & 0x01)==1) {
      __vr vrw_i0 = _vel_vldu_vssl(4, pWeight+(i  )*outDim+o, vl) ;
      __vr vrw_i1 = _vel_vldu_vssl(4, pWeight+(i+1)*outDim+o, vl) ;

      __vr vrw_i01 = _vel_vshf_vvvsl(vrw_i0, vrw_i1, VE_VSHUFFLE_YUZU, vl) ;

      const uint64_t in_i01_b0 = pack_f32p(pIn+0*inDim+i+0, pIn+0*inDim+i+1) ;
      const uint64_t in_i01_b1 = pack_f32p(pIn+1*inDim+i+0, pIn+1*inDim+i+1) ;
      const uint64_t in_i01_b2 = pack_f32p(pIn+2*inDim+i+0, pIn+2*inDim+i+1) ;

      vrsum_b0 = _vel_pvfmad_vvsvl(vrsum_b0, in_i01_b0, vrw_i01, vl) ;
      vrsum_b1 = _vel_pvfmad_vvsvl(vrsum_b1, in_i01_b1, vrw_i01, vl) ;
      vrsum_b2 = _vel_pvfmad_vvsvl(vrsum_b2, in_i01_b2, vrw_i01, vl) ;

      i+=2 ;
    }
    for(; i<inDim; i+=4 ) {
      __vr vrw_i0 = _vel_vldu_vssl(4, pWeight+(i  )*outDim+o, vl) ;
      __vr vrw_i1 = _vel_vldu_vssl(4, pWeight+(i+1)*outDim+o, vl) ;
      __vr vrw_i2 = _vel_vldu_vssl(4, pWeight+(i+2)*outDim+o, vl) ;
      __vr vrw_i3 = _vel_vldu_vssl(4, pWeight+(i+3)*outDim+o, vl) ;

      __vr vrw_i01 = _vel_vshf_vvvsl(vrw_i0, vrw_i1, VE_VSHUFFLE_YUZU, vl) ;
      __vr vrw_i23 = _vel_vshf_vvvsl(vrw_i2, vrw_i3, VE_VSHUFFLE_YUZU, vl) ;

      const uint64_t in_i01_b0 = pack_f32p(pIn+0*inDim+i+0, pIn+0*inDim+i+1) ;
      const uint64_t in_i01_b1 = pack_f32p(pIn+1*inDim+i+0, pIn+1*inDim+i+1) ;
      const uint64_t in_i01_b2 = pack_f32p(pIn+2*inDim+i+0, pIn+2*inDim+i+1) ;

      vrsum_b0 = _vel_pvfmad_vvsvl(vrsum_b0, in_i01_b0, vrw_i01, vl) ;
      vrsum_b1 = _vel_pvfmad_vvsvl(vrsum_b1, in_i01_b1, vrw_i01, vl) ;
      vrsum_b2 = _vel_pvfmad_vvsvl(vrsum_b2, in_i01_b2, vrw_i01, vl) ;

      const uint64_t in_i23_b0 = pack_f32p(pIn+0*inDim+i+2, pIn+0*inDim+i+3) ;
      const uint64_t in_i23_b1 = pack_f32p(pIn+1*inDim+i+2, pIn+1*inDim+i+3) ;
      const uint64_t in_i23_b2 = pack_f32p(pIn+2*inDim+i+2, pIn+2*inDim+i+3) ;

      vrsum_b0 = _vel_pvfmad_vvsvl(vrsum_b0, in_i23_b0, vrw_i23, vl) ;
      vrsum_b1 = _vel_pvfmad_vvsvl(vrsum_b1, in_i23_b1, vrw_i23, vl) ;
      vrsum_b2 = _vel_pvfmad_vvsvl(vrsum_b2, in_i23_b2, vrw_i23, vl) ;
    }

    vrsum_b0 = _vel_vfadds_vvvl(vrsum_b0, _vel_vsll_vvsl(vrsum_b0,32,vl), vl) ;
    _vel_vstu_vssl(vrsum_b0, 4, pOut+0*outDim+o, vl) ;
    vrsum_b1 = _vel_vfadds_vvvl(vrsum_b1, _vel_vsll_vvsl(vrsum_b1,32,vl), vl) ;
    _vel_vstu_vssl(vrsum_b1, 4, pOut+1*outDim+o, vl) ;
    vrsum_b2 = _vel_vfadds_vvvl(vrsum_b2, _vel_vsll_vvsl(vrsum_b2,32,vl), vl) ;
    _vel_vstu_vssl(vrsum_b2, 4, pOut+2*outDim+o, vl) ;

  }
}

static inline void b4(
  const uint64_t	inDim,
  const uint64_t	outDim,
  const float * 	pIn,
  const float * 	pWeight,
  float * 		pOut
)
{
  for(int64_t o=0; o<outDim; o+=VLEN) {
    const int64_t vl = outDim-o < VLEN ? outDim - o : VLEN ;
    __vr vrsum_b0 = _vel_vbrdl_vsl(0UL,vl) ;
    __vr vrsum_b1 = _vel_vbrdl_vsl(0UL,vl) ;
    __vr vrsum_b2 = _vel_vbrdl_vsl(0UL,vl) ;
    __vr vrsum_b3 = _vel_vbrdl_vsl(0UL,vl) ;

    int64_t i=0;
    if((inDim & 0x01)==1) {
      __vr vrw = _vel_vldu_vssl(4, pWeight+i*outDim+o, vl) ;
      vrsum_b0 = _vel_vfmads_vvsvl(vrsum_b0, pIn[0*inDim+i], vrw, vl) ;
      vrsum_b1 = _vel_vfmads_vvsvl(vrsum_b1, pIn[1*inDim+i], vrw, vl) ;
      vrsum_b2 = _vel_vfmads_vvsvl(vrsum_b2, pIn[2*inDim+i], vrw, vl) ;
      vrsum_b3 = _vel_vfmads_vvsvl(vrsum_b3, pIn[3*inDim+i], vrw, vl) ;

      i+=1 ;
    }
    if(((inDim>>1) & 0x01)==1) {
      __vr vrw_i0 = _vel_vldu_vssl(4, pWeight+(i  )*outDim+o, vl) ;
      __vr vrw_i1 = _vel_vldu_vssl(4, pWeight+(i+1)*outDim+o, vl) ;

      __vr vrw_i01 = _vel_vshf_vvvsl(vrw_i0, vrw_i1, VE_VSHUFFLE_YUZU, vl) ;

      const uint64_t in_i01_b0 = pack_f32p(pIn+0*inDim+i+0, pIn+0*inDim+i+1) ;
      const uint64_t in_i01_b1 = pack_f32p(pIn+1*inDim+i+0, pIn+1*inDim+i+1) ;
      const uint64_t in_i01_b2 = pack_f32p(pIn+2*inDim+i+0, pIn+2*inDim+i+1) ;
      const uint64_t in_i01_b3 = pack_f32p(pIn+3*inDim+i+0, pIn+3*inDim+i+1) ;

      vrsum_b0 = _vel_pvfmad_vvsvl(vrsum_b0, in_i01_b0, vrw_i01, vl) ;
      vrsum_b1 = _vel_pvfmad_vvsvl(vrsum_b1, in_i01_b1, vrw_i01, vl) ;
      vrsum_b2 = _vel_pvfmad_vvsvl(vrsum_b2, in_i01_b2, vrw_i01, vl) ;
      vrsum_b3 = _vel_pvfmad_vvsvl(vrsum_b3, in_i01_b3, vrw_i01, vl) ;

      i+=2 ;
    }
    for(; i<inDim; i+=4 ) {
      __vr vrw_i0 = _vel_vldu_vssl(4, pWeight+(i  )*outDim+o, vl) ;
      __vr vrw_i1 = _vel_vldu_vssl(4, pWeight+(i+1)*outDim+o, vl) ;
      __vr vrw_i2 = _vel_vldu_vssl(4, pWeight+(i+2)*outDim+o, vl) ;
      __vr vrw_i3 = _vel_vldu_vssl(4, pWeight+(i+3)*outDim+o, vl) ;

      __vr vrw_i01 = _vel_vshf_vvvsl(vrw_i0, vrw_i1, VE_VSHUFFLE_YUZU, vl) ;
      __vr vrw_i23 = _vel_vshf_vvvsl(vrw_i2, vrw_i3, VE_VSHUFFLE_YUZU, vl) ;

      const uint64_t in_i01_b0 = pack_f32p(pIn+0*inDim+i+0, pIn+0*inDim+i+1) ;
      const uint64_t in_i01_b1 = pack_f32p(pIn+1*inDim+i+0, pIn+1*inDim+i+1) ;
      const uint64_t in_i01_b2 = pack_f32p(pIn+2*inDim+i+0, pIn+2*inDim+i+1) ;
      const uint64_t in_i01_b3 = pack_f32p(pIn+3*inDim+i+0, pIn+3*inDim+i+1) ;

      vrsum_b0 = _vel_pvfmad_vvsvl(vrsum_b0, in_i01_b0, vrw_i01, vl) ;
      vrsum_b1 = _vel_pvfmad_vvsvl(vrsum_b1, in_i01_b1, vrw_i01, vl) ;
      vrsum_b2 = _vel_pvfmad_vvsvl(vrsum_b2, in_i01_b2, vrw_i01, vl) ;
      vrsum_b3 = _vel_pvfmad_vvsvl(vrsum_b3, in_i01_b3, vrw_i01, vl) ;

      const uint64_t in_i23_b0 = pack_f32p(pIn+0*inDim+i+2, pIn+0*inDim+i+3) ;
      const uint64_t in_i23_b1 = pack_f32p(pIn+1*inDim+i+2, pIn+1*inDim+i+3) ;
      const uint64_t in_i23_b2 = pack_f32p(pIn+2*inDim+i+2, pIn+2*inDim+i+3) ;
      const uint64_t in_i23_b3 = pack_f32p(pIn+3*inDim+i+2, pIn+3*inDim+i+3) ;

      vrsum_b0 = _vel_pvfmad_vvsvl(vrsum_b0, in_i23_b0, vrw_i23, vl) ;
      vrsum_b1 = _vel_pvfmad_vvsvl(vrsum_b1, in_i23_b1, vrw_i23, vl) ;
      vrsum_b2 = _vel_pvfmad_vvsvl(vrsum_b2, in_i23_b2, vrw_i23, vl) ;
      vrsum_b3 = _vel_pvfmad_vvsvl(vrsum_b3, in_i23_b3, vrw_i23, vl) ;
    }

    vrsum_b0 = _vel_vfadds_vvvl(vrsum_b0, _vel_vsll_vvsl(vrsum_b0,32,vl), vl) ;
    _vel_vstu_vssl(vrsum_b0, 4, pOut+0*outDim+o, vl) ;
    vrsum_b1 = _vel_vfadds_vvvl(vrsum_b1, _vel_vsll_vvsl(vrsum_b1,32,vl), vl) ;
    _vel_vstu_vssl(vrsum_b1, 4, pOut+1*outDim+o, vl) ;
    vrsum_b2 = _vel_vfadds_vvvl(vrsum_b2, _vel_vsll_vvsl(vrsum_b2,32,vl), vl) ;
    _vel_vstu_vssl(vrsum_b2, 4, pOut+2*outDim+o, vl) ;
    vrsum_b3 = _vel_vfadds_vvvl(vrsum_b3, _vel_vsll_vvsl(vrsum_b3,32,vl), vl) ;
    _vel_vstu_vssl(vrsum_b3, 4, pOut+3*outDim+o, vl) ;

  }
}

static inline void b5(
  const uint64_t	inDim,
  const uint64_t	outDim,
  const float * 	pIn,
  const float * 	pWeight,
  float * 		pOut
)
{
  for(int64_t o=0; o<outDim; o+=VLEN) {
    const int64_t vl = outDim-o < VLEN ? outDim - o : VLEN ;
    __vr vrsum_b0 = _vel_vbrdl_vsl(0UL,vl) ;
    __vr vrsum_b1 = _vel_vbrdl_vsl(0UL,vl) ;
    __vr vrsum_b2 = _vel_vbrdl_vsl(0UL,vl) ;
    __vr vrsum_b3 = _vel_vbrdl_vsl(0UL,vl) ;
    __vr vrsum_b4 = _vel_vbrdl_vsl(0UL,vl) ;

    int64_t i=0;
    if((inDim & 0x01)==1) {
      __vr vrw = _vel_vldu_vssl(4, pWeight+i*outDim+o, vl) ;
      vrsum_b0 = _vel_vfmads_vvsvl(vrsum_b0, pIn[0*inDim+i], vrw, vl) ;
      vrsum_b1 = _vel_vfmads_vvsvl(vrsum_b1, pIn[1*inDim+i], vrw, vl) ;
      vrsum_b2 = _vel_vfmads_vvsvl(vrsum_b2, pIn[2*inDim+i], vrw, vl) ;
      vrsum_b3 = _vel_vfmads_vvsvl(vrsum_b3, pIn[3*inDim+i], vrw, vl) ;
      vrsum_b4 = _vel_vfmads_vvsvl(vrsum_b4, pIn[4*inDim+i], vrw, vl) ;

      i+=1 ;
    }
    if(((inDim>>1) & 0x01)==1) {
      __vr vrw_i0 = _vel_vldu_vssl(4, pWeight+(i  )*outDim+o, vl) ;
      __vr vrw_i1 = _vel_vldu_vssl(4, pWeight+(i+1)*outDim+o, vl) ;

      __vr vrw_i01 = _vel_vshf_vvvsl(vrw_i0, vrw_i1, VE_VSHUFFLE_YUZU, vl) ;

      const uint64_t in_i01_b0 = pack_f32p(pIn+0*inDim+i+0, pIn+0*inDim+i+1) ;
      const uint64_t in_i01_b1 = pack_f32p(pIn+1*inDim+i+0, pIn+1*inDim+i+1) ;
      const uint64_t in_i01_b2 = pack_f32p(pIn+2*inDim+i+0, pIn+2*inDim+i+1) ;
      const uint64_t in_i01_b3 = pack_f32p(pIn+3*inDim+i+0, pIn+3*inDim+i+1) ;
      const uint64_t in_i01_b4 = pack_f32p(pIn+4*inDim+i+0, pIn+4*inDim+i+1) ;

      vrsum_b0 = _vel_pvfmad_vvsvl(vrsum_b0, in_i01_b0, vrw_i01, vl) ;
      vrsum_b1 = _vel_pvfmad_vvsvl(vrsum_b1, in_i01_b1, vrw_i01, vl) ;
      vrsum_b2 = _vel_pvfmad_vvsvl(vrsum_b2, in_i01_b2, vrw_i01, vl) ;
      vrsum_b3 = _vel_pvfmad_vvsvl(vrsum_b3, in_i01_b3, vrw_i01, vl) ;
      vrsum_b4 = _vel_pvfmad_vvsvl(vrsum_b4, in_i01_b4, vrw_i01, vl) ;

      i+=2 ;
    }
    for(; i<inDim; i+=4 ) {
      __vr vrw_i0 = _vel_vldu_vssl(4, pWeight+(i  )*outDim+o, vl) ;
      __vr vrw_i1 = _vel_vldu_vssl(4, pWeight+(i+1)*outDim+o, vl) ;
      __vr vrw_i2 = _vel_vldu_vssl(4, pWeight+(i+2)*outDim+o, vl) ;
      __vr vrw_i3 = _vel_vldu_vssl(4, pWeight+(i+3)*outDim+o, vl) ;

      __vr vrw_i01 = _vel_vshf_vvvsl(vrw_i0, vrw_i1, VE_VSHUFFLE_YUZU, vl) ;
      __vr vrw_i23 = _vel_vshf_vvvsl(vrw_i2, vrw_i3, VE_VSHUFFLE_YUZU, vl) ;

      const uint64_t in_i01_b0 = pack_f32p(pIn+0*inDim+i+0, pIn+0*inDim+i+1) ;
      const uint64_t in_i01_b1 = pack_f32p(pIn+1*inDim+i+0, pIn+1*inDim+i+1) ;
      const uint64_t in_i01_b2 = pack_f32p(pIn+2*inDim+i+0, pIn+2*inDim+i+1) ;
      const uint64_t in_i01_b3 = pack_f32p(pIn+3*inDim+i+0, pIn+3*inDim+i+1) ;
      const uint64_t in_i01_b4 = pack_f32p(pIn+4*inDim+i+0, pIn+4*inDim+i+1) ;

      vrsum_b0 = _vel_pvfmad_vvsvl(vrsum_b0, in_i01_b0, vrw_i01, vl) ;
      vrsum_b1 = _vel_pvfmad_vvsvl(vrsum_b1, in_i01_b1, vrw_i01, vl) ;
      vrsum_b2 = _vel_pvfmad_vvsvl(vrsum_b2, in_i01_b2, vrw_i01, vl) ;
      vrsum_b3 = _vel_pvfmad_vvsvl(vrsum_b3, in_i01_b3, vrw_i01, vl) ;
      vrsum_b4 = _vel_pvfmad_vvsvl(vrsum_b4, in_i01_b4, vrw_i01, vl) ;

      const uint64_t in_i23_b0 = pack_f32p(pIn+0*inDim+i+2, pIn+0*inDim+i+3) ;
      const uint64_t in_i23_b1 = pack_f32p(pIn+1*inDim+i+2, pIn+1*inDim+i+3) ;
      const uint64_t in_i23_b2 = pack_f32p(pIn+2*inDim+i+2, pIn+2*inDim+i+3) ;
      const uint64_t in_i23_b3 = pack_f32p(pIn+3*inDim+i+2, pIn+3*inDim+i+3) ;
      const uint64_t in_i23_b4 = pack_f32p(pIn+4*inDim+i+2, pIn+4*inDim+i+3) ;

      vrsum_b0 = _vel_pvfmad_vvsvl(vrsum_b0, in_i23_b0, vrw_i23, vl) ;
      vrsum_b1 = _vel_pvfmad_vvsvl(vrsum_b1, in_i23_b1, vrw_i23, vl) ;
      vrsum_b2 = _vel_pvfmad_vvsvl(vrsum_b2, in_i23_b2, vrw_i23, vl) ;
      vrsum_b3 = _vel_pvfmad_vvsvl(vrsum_b3, in_i23_b3, vrw_i23, vl) ;
      vrsum_b4 = _vel_pvfmad_vvsvl(vrsum_b4, in_i23_b4, vrw_i23, vl) ;
    }

    vrsum_b0 = _vel_vfadds_vvvl(vrsum_b0, _vel_vsll_vvsl(vrsum_b0,32,vl), vl) ;
    _vel_vstu_vssl(vrsum_b0, 4, pOut+0*outDim+o, vl) ;
    vrsum_b1 = _vel_vfadds_vvvl(vrsum_b1, _vel_vsll_vvsl(vrsum_b1,32,vl), vl) ;
    _vel_vstu_vssl(vrsum_b1, 4, pOut+1*outDim+o, vl) ;
    vrsum_b2 = _vel_vfadds_vvvl(vrsum_b2, _vel_vsll_vvsl(vrsum_b2,32,vl), vl) ;
    _vel_vstu_vssl(vrsum_b2, 4, pOut+2*outDim+o, vl) ;
    vrsum_b3 = _vel_vfadds_vvvl(vrsum_b3, _vel_vsll_vvsl(vrsum_b3,32,vl), vl) ;
    _vel_vstu_vssl(vrsum_b3, 4, pOut+3*outDim+o, vl) ;
    vrsum_b4 = _vel_vfadds_vvvl(vrsum_b4, _vel_vsll_vvsl(vrsum_b4,32,vl), vl) ;
    _vel_vstu_vssl(vrsum_b4, 4, pOut+4*outDim+o, vl) ;

  }
}

static inline void b6(
  const uint64_t	inDim,
  const uint64_t	outDim,
  const float * 	pIn,
  const float * 	pWeight,
  float * 		pOut
)
{
  for(int64_t o=0; o<outDim; o+=VLEN) {
    const int64_t vl = outDim-o < VLEN ? outDim - o : VLEN ;
    __vr vrsum_b0 = _vel_vbrdl_vsl(0UL,vl) ;
    __vr vrsum_b1 = _vel_vbrdl_vsl(0UL,vl) ;
    __vr vrsum_b2 = _vel_vbrdl_vsl(0UL,vl) ;
    __vr vrsum_b3 = _vel_vbrdl_vsl(0UL,vl) ;
    __vr vrsum_b4 = _vel_vbrdl_vsl(0UL,vl) ;
    __vr vrsum_b5 = _vel_vbrdl_vsl(0UL,vl) ;

    int64_t i=0;
    if((inDim & 0x01)==1) {
      __vr vrw = _vel_vldu_vssl(4, pWeight+i*outDim+o, vl) ;
      vrsum_b0 = _vel_vfmads_vvsvl(vrsum_b0, pIn[0*inDim+i], vrw, vl) ;
      vrsum_b1 = _vel_vfmads_vvsvl(vrsum_b1, pIn[1*inDim+i], vrw, vl) ;
      vrsum_b2 = _vel_vfmads_vvsvl(vrsum_b2, pIn[2*inDim+i], vrw, vl) ;
      vrsum_b3 = _vel_vfmads_vvsvl(vrsum_b3, pIn[3*inDim+i], vrw, vl) ;
      vrsum_b4 = _vel_vfmads_vvsvl(vrsum_b4, pIn[4*inDim+i], vrw, vl) ;
      vrsum_b5 = _vel_vfmads_vvsvl(vrsum_b5, pIn[5*inDim+i], vrw, vl) ;

      i+=1 ;
    }
    if(((inDim>>1) & 0x01)==1) {
      __vr vrw_i0 = _vel_vldu_vssl(4, pWeight+(i  )*outDim+o, vl) ;
      __vr vrw_i1 = _vel_vldu_vssl(4, pWeight+(i+1)*outDim+o, vl) ;

      __vr vrw_i01 = _vel_vshf_vvvsl(vrw_i0, vrw_i1, VE_VSHUFFLE_YUZU, vl) ;

      const uint64_t in_i01_b0 = pack_f32p(pIn+0*inDim+i+0, pIn+0*inDim+i+1) ;
      const uint64_t in_i01_b1 = pack_f32p(pIn+1*inDim+i+0, pIn+1*inDim+i+1) ;
      const uint64_t in_i01_b2 = pack_f32p(pIn+2*inDim+i+0, pIn+2*inDim+i+1) ;
      const uint64_t in_i01_b3 = pack_f32p(pIn+3*inDim+i+0, pIn+3*inDim+i+1) ;
      const uint64_t in_i01_b4 = pack_f32p(pIn+4*inDim+i+0, pIn+4*inDim+i+1) ;
      const uint64_t in_i01_b5 = pack_f32p(pIn+5*inDim+i+0, pIn+5*inDim+i+1) ;

      vrsum_b0 = _vel_pvfmad_vvsvl(vrsum_b0, in_i01_b0, vrw_i01, vl) ;
      vrsum_b1 = _vel_pvfmad_vvsvl(vrsum_b1, in_i01_b1, vrw_i01, vl) ;
      vrsum_b2 = _vel_pvfmad_vvsvl(vrsum_b2, in_i01_b2, vrw_i01, vl) ;
      vrsum_b3 = _vel_pvfmad_vvsvl(vrsum_b3, in_i01_b3, vrw_i01, vl) ;
      vrsum_b4 = _vel_pvfmad_vvsvl(vrsum_b4, in_i01_b4, vrw_i01, vl) ;
      vrsum_b5 = _vel_pvfmad_vvsvl(vrsum_b5, in_i01_b5, vrw_i01, vl) ;

      i+=2 ;
    }
    for(; i<inDim; i+=4 ) {
      __vr vrw_i0 = _vel_vldu_vssl(4, pWeight+(i  )*outDim+o, vl) ;
      __vr vrw_i1 = _vel_vldu_vssl(4, pWeight+(i+1)*outDim+o, vl) ;
      __vr vrw_i2 = _vel_vldu_vssl(4, pWeight+(i+2)*outDim+o, vl) ;
      __vr vrw_i3 = _vel_vldu_vssl(4, pWeight+(i+3)*outDim+o, vl) ;

      __vr vrw_i01 = _vel_vshf_vvvsl(vrw_i0, vrw_i1, VE_VSHUFFLE_YUZU, vl) ;
      __vr vrw_i23 = _vel_vshf_vvvsl(vrw_i2, vrw_i3, VE_VSHUFFLE_YUZU, vl) ;

      const uint64_t in_i01_b0 = pack_f32p(pIn+0*inDim+i+0, pIn+0*inDim+i+1) ;
      const uint64_t in_i01_b1 = pack_f32p(pIn+1*inDim+i+0, pIn+1*inDim+i+1) ;
      const uint64_t in_i01_b2 = pack_f32p(pIn+2*inDim+i+0, pIn+2*inDim+i+1) ;
      const uint64_t in_i01_b3 = pack_f32p(pIn+3*inDim+i+0, pIn+3*inDim+i+1) ;
      const uint64_t in_i01_b4 = pack_f32p(pIn+4*inDim+i+0, pIn+4*inDim+i+1) ;
      const uint64_t in_i01_b5 = pack_f32p(pIn+5*inDim+i+0, pIn+5*inDim+i+1) ;

      vrsum_b0 = _vel_pvfmad_vvsvl(vrsum_b0, in_i01_b0, vrw_i01, vl) ;
      vrsum_b1 = _vel_pvfmad_vvsvl(vrsum_b1, in_i01_b1, vrw_i01, vl) ;
      vrsum_b2 = _vel_pvfmad_vvsvl(vrsum_b2, in_i01_b2, vrw_i01, vl) ;
      vrsum_b3 = _vel_pvfmad_vvsvl(vrsum_b3, in_i01_b3, vrw_i01, vl) ;
      vrsum_b4 = _vel_pvfmad_vvsvl(vrsum_b4, in_i01_b4, vrw_i01, vl) ;
      vrsum_b5 = _vel_pvfmad_vvsvl(vrsum_b5, in_i01_b5, vrw_i01, vl) ;

      const uint64_t in_i23_b0 = pack_f32p(pIn+0*inDim+i+2, pIn+0*inDim+i+3) ;
      const uint64_t in_i23_b1 = pack_f32p(pIn+1*inDim+i+2, pIn+1*inDim+i+3) ;
      const uint64_t in_i23_b2 = pack_f32p(pIn+2*inDim+i+2, pIn+2*inDim+i+3) ;
      const uint64_t in_i23_b3 = pack_f32p(pIn+3*inDim+i+2, pIn+3*inDim+i+3) ;
      const uint64_t in_i23_b4 = pack_f32p(pIn+4*inDim+i+2, pIn+4*inDim+i+3) ;
      const uint64_t in_i23_b5 = pack_f32p(pIn+5*inDim+i+2, pIn+5*inDim+i+3) ;

      vrsum_b0 = _vel_pvfmad_vvsvl(vrsum_b0, in_i23_b0, vrw_i23, vl) ;
      vrsum_b1 = _vel_pvfmad_vvsvl(vrsum_b1, in_i23_b1, vrw_i23, vl) ;
      vrsum_b2 = _vel_pvfmad_vvsvl(vrsum_b2, in_i23_b2, vrw_i23, vl) ;
      vrsum_b3 = _vel_pvfmad_vvsvl(vrsum_b3, in_i23_b3, vrw_i23, vl) ;
      vrsum_b4 = _vel_pvfmad_vvsvl(vrsum_b4, in_i23_b4, vrw_i23, vl) ;
      vrsum_b5 = _vel_pvfmad_vvsvl(vrsum_b5, in_i23_b5, vrw_i23, vl) ;
    }

    vrsum_b0 = _vel_vfadds_vvvl(vrsum_b0, _vel_vsll_vvsl(vrsum_b0,32,vl), vl) ;
    _vel_vstu_vssl(vrsum_b0, 4, pOut+0*outDim+o, vl) ;
    vrsum_b1 = _vel_vfadds_vvvl(vrsum_b1, _vel_vsll_vvsl(vrsum_b1,32,vl), vl) ;
    _vel_vstu_vssl(vrsum_b1, 4, pOut+1*outDim+o, vl) ;
    vrsum_b2 = _vel_vfadds_vvvl(vrsum_b2, _vel_vsll_vvsl(vrsum_b2,32,vl), vl) ;
    _vel_vstu_vssl(vrsum_b2, 4, pOut+2*outDim+o, vl) ;
    vrsum_b3 = _vel_vfadds_vvvl(vrsum_b3, _vel_vsll_vvsl(vrsum_b3,32,vl), vl) ;
    _vel_vstu_vssl(vrsum_b3, 4, pOut+3*outDim+o, vl) ;
    vrsum_b4 = _vel_vfadds_vvvl(vrsum_b4, _vel_vsll_vvsl(vrsum_b4,32,vl), vl) ;
    _vel_vstu_vssl(vrsum_b4, 4, pOut+4*outDim+o, vl) ;
    vrsum_b5 = _vel_vfadds_vvvl(vrsum_b5, _vel_vsll_vvsl(vrsum_b5,32,vl), vl) ;
    _vel_vstu_vssl(vrsum_b5, 4, pOut+5*outDim+o, vl) ;

  }
}


vednnError_t vednnLinearForward_default(
    const uint64_t			inDim,
    const uint64_t			outDim,
    const uint64_t			nBatch,
    const void * restrict		pDataIn,
    const void * restrict		pDataWeight,
    void * restrict			pDataOut
)
{

  const float * restrict pIn     = (const float * restrict) pDataIn;
  const float * restrict pWeight = (const float * restrict) pDataWeight;
  float * restrict const pOut    = (float * restrict const) pDataOut;

  int64_t n=0;
  int64_t batchRemain = nBatch % 6 ;

  switch( batchRemain ) {
  case 1 :
    b1(inDim, outDim,
       pIn+n*inDim, pWeight, pOut+n*outDim) ;
    n+=1 ;
    break ;
  case 2 :
    b2(inDim, outDim,
       pIn+n*inDim, pWeight, pOut+n*outDim) ;
    n+=2 ;
    break ;
  case 3 :
    b3(inDim, outDim,
       pIn+n*inDim, pWeight, pOut+n*outDim) ;
    n+=3 ;
    break ;
  case 4 :
    b4(inDim, outDim,
       pIn+n*inDim, pWeight, pOut+n*outDim) ;
    n+=4 ;
    break ;
  case 5 :
    b5(inDim, outDim,
       pIn+n*inDim, pWeight, pOut+n*outDim) ;
    n+=5 ;
    break ;
  default :
    break ;
  }
  for(; n<nBatch; n+=6) {
    b6(inDim, outDim,
       pIn+n*inDim, pWeight, pOut+n*outDim) ;
  }

  return VEDNN_SUCCESS ;
}

#if 0 // reference code
vednnError_t vednnLinearForward_default(
    const uint64_t			inDim,
    const uint64_t			outDim,
    const uint64_t			nBatch,
    const void * restrict		pDataIn,
    const void * restrict		pDataWeight,
    void * restrict			pDataOut
)
{
  const float * restrict pIn     = pDataIn;
  const float * restrict pWeight = pDataWeight;
  float * restrict const pOut    = pDataOut;

  for(int64_t n=0; n<nBatch; n++) {
    for(int64_t o=0; o<outDim; o++) {
      float sum = 0.f ;
      for(int64_t i=0; i<inDim; i++ ) {
	sum += pWeight[i*outDim+o] * pIn[n*inDim+i] ;
      }
      pOut[n*outDim+o] = sum ;
    }
  }

  return VEDNN_SUCCESS ;
}
#endif


