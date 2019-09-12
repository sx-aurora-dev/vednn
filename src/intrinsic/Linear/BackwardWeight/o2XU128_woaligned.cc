#include <stdint.h>
#include <float.h>

#include <stdio.h>

#include "vednn.h"

#include "velintrin.h"
#define VLEN	(256)

template <int BATCH, bool UPDATE>
static inline void func(
  const uint64_t	inDim,
  const uint64_t	outDim,
  const uint64_t	nInDim,
  const float * 	pIn,
  const float * 	pGOut,
  float * 		pGWeight
)
{
  const int64_t vl  = outDim >> 1 ;
  const int64_t vl2 = outDim ;

  __vr vrseq  = _vel_vseq_vl(vl2) ;
  __vm256 vm256f = _vel_vfmkllt_mvl(_vel_vaddsl_vsvl(-vl,vrseq, vl2), vl2) ;
  __vm256 vm256l = _vel_negm_mm(vm256f) ;

  __vm512 vm512f, vm512l ;
  vm512f = _vel_insert_vm512u(vm512f, vm256f) ;
  vm512f = _vel_insert_vm512l(vm512f, vm256f) ;
  vm512l = _vel_insert_vm512u(vm512l, vm256l) ;
  vm512l = _vel_insert_vm512l(vm512l, vm256l) ;

  __vr vrgout_b0 = _vel_vld_vssl(8, pGOut+0*outDim, vl) ;
  __vr vrgout_b1 = _vel_vld_vssl(8, pGOut+1*outDim, vl) ;
  __vr vrgout_b2 = _vel_vld_vssl(8, pGOut+2*outDim, vl) ;
  __vr vrgout_b3 = _vel_vld_vssl(8, pGOut+3*outDim, vl) ;
  __vr vrgout_b4 = _vel_vld_vssl(8, pGOut+4*outDim, vl) ;
  __vr vrgout_b5 = _vel_vld_vssl(8, pGOut+5*outDim, vl) ;
  __vr vrgout_b6 = _vel_vld_vssl(8, pGOut+6*outDim, vl) ;
  __vr vrgout_b7 = _vel_vld_vssl(8, pGOut+7*outDim, vl) ;

  __vr vrgout_b0f = vrgout_b0 ;
  __vr vrgout_b1f = vrgout_b1 ;
  __vr vrgout_b2f = vrgout_b2 ;
  __vr vrgout_b3f = vrgout_b3 ;
  __vr vrgout_b4f = vrgout_b4 ;
  __vr vrgout_b5f = vrgout_b5 ;
  __vr vrgout_b6f = vrgout_b6 ;
  __vr vrgout_b7f = vrgout_b7 ;

  __vr vrgout_b0l = _vel_vmv_vsvl(-vl, vrgout_b0, vl2) ;
  __vr vrgout_b1l = _vel_vmv_vsvl(-vl, vrgout_b1, vl2) ;
  __vr vrgout_b2l = _vel_vmv_vsvl(-vl, vrgout_b2, vl2) ;
  __vr vrgout_b3l = _vel_vmv_vsvl(-vl, vrgout_b3, vl2) ;
  __vr vrgout_b4l = _vel_vmv_vsvl(-vl, vrgout_b4, vl2) ;
  __vr vrgout_b5l = _vel_vmv_vsvl(-vl, vrgout_b5, vl2) ;
  __vr vrgout_b6l = _vel_vmv_vsvl(-vl, vrgout_b6, vl2) ;
  __vr vrgout_b7l = _vel_vmv_vsvl(-vl, vrgout_b7, vl2) ;

  int64_t i=0;
  if(nInDim & 0x1) {
    __vr vrgw ;

    if(UPDATE) {
      vrgw = _vel_vld_vssl(8, pGWeight+(i+0)*outDim, vl) ;

      const uint64_t in_b0 = _vel_pack_f32a(pIn+0*inDim+i) ;
      vrgw = _vel_pvfmad_vvsvl(vrgw, in_b0, vrgout_b0, vl) ;
    }
    else {
      const uint64_t in_b0 = _vel_pack_f32a(pIn+0*inDim+i) ;
      vrgw = _vel_pvfmul_vsvl(in_b0, vrgout_b0, vl) ;
    }

    if(BATCH>=2)  {
      const uint64_t in_b1 = _vel_pack_f32a(pIn+1*inDim+i) ;
      vrgw = _vel_pvfmad_vvsvl(vrgw, in_b1, vrgout_b1, vl) ;
    }
    if(BATCH>=3) {
      const uint64_t in_b2 = _vel_pack_f32a(pIn+2*inDim+i) ;
      vrgw = _vel_pvfmad_vvsvl(vrgw, in_b2, vrgout_b2, vl) ;
    }
    if(BATCH>=4) {
      const uint64_t in_b3 = _vel_pack_f32a(pIn+3*inDim+i) ;
      vrgw = _vel_pvfmad_vvsvl(vrgw, in_b3, vrgout_b3, vl) ;
    }
    if(BATCH>=5) {
      const uint64_t in_b4 = _vel_pack_f32a(pIn+4*inDim+i) ;
      vrgw = _vel_pvfmad_vvsvl(vrgw, in_b4, vrgout_b4, vl) ;
    }
    if(BATCH>=6) {
      const uint64_t in_b5 = _vel_pack_f32a(pIn+5*inDim+i) ;
      vrgw = _vel_pvfmad_vvsvl(vrgw, in_b5, vrgout_b5, vl) ;
    }
    if(BATCH>=7) {
      const uint64_t in_b6 = _vel_pack_f32a(pIn+6*inDim+i) ;
      vrgw = _vel_pvfmad_vvsvl(vrgw, in_b6, vrgout_b6, vl) ;
    }
    if(BATCH>=8) {
      const uint64_t in_b7 = _vel_pack_f32a(pIn+7*inDim+i) ;
      vrgw = _vel_pvfmad_vvsvl(vrgw, in_b7, vrgout_b7, vl) ;
    }

    _vel_vst_vssl(vrgw, 8, pGWeight+i*outDim, vl) ;

    i+=1;
  }
  if((nInDim>>1) & 0x1) {
    __vr vrgw_i01 ;

    if(UPDATE) {
      vrgw_i01 = _vel_vld_vssl(8, pGWeight+(i+0)*outDim, vl2) ;

      const uint64_t in_i0_b0 = _vel_pack_f32a(pIn+0*inDim+i+0) ;
      const uint64_t in_i1_b0 = _vel_pack_f32a(pIn+0*inDim+i+1) ;
      vrgw_i01 = _vel_pvfmad_vvsvMvl(vrgw_i01, in_i0_b0, vrgout_b0f, vm512f, vrgw_i01, vl2) ;
      vrgw_i01 = _vel_pvfmad_vvsvMvl(vrgw_i01, in_i1_b0, vrgout_b0l, vm512l, vrgw_i01, vl2) ;
    }
    else {
      const uint64_t in_i0_b0 = _vel_pack_f32a(pIn+0*inDim+i+0) ;
      const uint64_t in_i1_b0 = _vel_pack_f32a(pIn+0*inDim+i+1) ;
      vrgw_i01 = _vel_pvfmul_vsvMvl(in_i0_b0, vrgout_b0f, vm512f, vrgw_i01, vl2) ;
      vrgw_i01 = _vel_pvfmul_vsvMvl(in_i1_b0, vrgout_b0l, vm512l, vrgw_i01, vl2) ;
    }

    if(BATCH>=2)  {
      const uint64_t in_i0_b1 = _vel_pack_f32a(pIn+1*inDim+i+0) ;
      const uint64_t in_i1_b1 = _vel_pack_f32a(pIn+1*inDim+i+1) ;
      vrgw_i01 = _vel_pvfmad_vvsvMvl(vrgw_i01, in_i0_b1, vrgout_b1f, vm512f, vrgw_i01, vl2) ;
      vrgw_i01 = _vel_pvfmad_vvsvMvl(vrgw_i01, in_i1_b1, vrgout_b1l, vm512l, vrgw_i01, vl2) ;
    }
    if(BATCH>=3) {
      const uint64_t in_i0_b2 = _vel_pack_f32a(pIn+2*inDim+i+0) ;
      const uint64_t in_i1_b2 = _vel_pack_f32a(pIn+2*inDim+i+1) ;
      vrgw_i01 = _vel_pvfmad_vvsvMvl(vrgw_i01, in_i0_b2, vrgout_b2f, vm512f, vrgw_i01, vl2) ;
      vrgw_i01 = _vel_pvfmad_vvsvMvl(vrgw_i01, in_i1_b2, vrgout_b2l, vm512l, vrgw_i01, vl2) ;
    }
    if(BATCH>=4) {
      const uint64_t in_i0_b3 = _vel_pack_f32a(pIn+3*inDim+i+0) ;
      const uint64_t in_i1_b3 = _vel_pack_f32a(pIn+3*inDim+i+1) ;
      vrgw_i01 = _vel_pvfmad_vvsvMvl(vrgw_i01, in_i0_b3, vrgout_b3f, vm512f, vrgw_i01, vl2) ;
      vrgw_i01 = _vel_pvfmad_vvsvMvl(vrgw_i01, in_i1_b3, vrgout_b3l, vm512l, vrgw_i01, vl2) ;
    }
    if(BATCH>=5) {
      const uint64_t in_i0_b4 = _vel_pack_f32a(pIn+4*inDim+i+0) ;
      const uint64_t in_i1_b4 = _vel_pack_f32a(pIn+4*inDim+i+1) ;
      vrgw_i01 = _vel_pvfmad_vvsvMvl(vrgw_i01, in_i0_b4, vrgout_b4f, vm512f, vrgw_i01, vl2) ;
      vrgw_i01 = _vel_pvfmad_vvsvMvl(vrgw_i01, in_i1_b4, vrgout_b4l, vm512l, vrgw_i01, vl2) ;
    }
    if(BATCH>=6) {
      const uint64_t in_i0_b5 = _vel_pack_f32a(pIn+5*inDim+i+0) ;
      const uint64_t in_i1_b5 = _vel_pack_f32a(pIn+5*inDim+i+1) ;
      const uint64_t in_i2_b5 = _vel_pack_f32a(pIn+5*inDim+i+2) ;
      const uint64_t in_i3_b5 = _vel_pack_f32a(pIn+5*inDim+i+3) ;
      vrgw_i01 = _vel_pvfmad_vvsvMvl(vrgw_i01, in_i0_b5, vrgout_b5f, vm512f, vrgw_i01, vl2) ;
      vrgw_i01 = _vel_pvfmad_vvsvMvl(vrgw_i01, in_i1_b5, vrgout_b5l, vm512l, vrgw_i01, vl2) ;
    }
    if(BATCH>=7) {
      const uint64_t in_i0_b6 = _vel_pack_f32a(pIn+6*inDim+i+0) ;
      const uint64_t in_i1_b6 = _vel_pack_f32a(pIn+6*inDim+i+1) ;
      vrgw_i01 = _vel_pvfmad_vvsvMvl(vrgw_i01, in_i0_b6, vrgout_b6f, vm512f, vrgw_i01, vl2) ;
      vrgw_i01 = _vel_pvfmad_vvsvMvl(vrgw_i01, in_i1_b6, vrgout_b6l, vm512l, vrgw_i01, vl2) ;
    }
    if(BATCH>=8) {
      const uint64_t in_i0_b7 = _vel_pack_f32a(pIn+7*inDim+i+0) ;
      const uint64_t in_i1_b7 = _vel_pack_f32a(pIn+7*inDim+i+1) ;
      vrgw_i01 = _vel_pvfmad_vvsvMvl(vrgw_i01, in_i0_b7, vrgout_b7f, vm512f, vrgw_i01, vl2) ;
      vrgw_i01 = _vel_pvfmad_vvsvMvl(vrgw_i01, in_i1_b7, vrgout_b7l, vm512l, vrgw_i01, vl2) ;
    }

    _vel_vst_vssl(vrgw_i01, 8, pGWeight+(i+0)*outDim, vl2) ;

    i+=2 ;
  }
  if((nInDim>>2) & 0x1) {
    __vr vrgw_i01 ;
    __vr vrgw_i23 ;

    if(UPDATE) {
      vrgw_i01 = _vel_vld_vssl(8, pGWeight+(i+0)*outDim, vl2) ;
      vrgw_i23 = _vel_vld_vssl(8, pGWeight+(i+2)*outDim, vl2) ;

      const uint64_t in_i0_b0 = _vel_pack_f32a(pIn+0*inDim+i+0) ;
      const uint64_t in_i1_b0 = _vel_pack_f32a(pIn+0*inDim+i+1) ;
      const uint64_t in_i2_b0 = _vel_pack_f32a(pIn+0*inDim+i+2) ;
      const uint64_t in_i3_b0 = _vel_pack_f32a(pIn+0*inDim+i+3) ;
      vrgw_i01 = _vel_pvfmad_vvsvMvl(vrgw_i01, in_i0_b0, vrgout_b0f, vm512f, vrgw_i01, vl2) ;
      vrgw_i01 = _vel_pvfmad_vvsvMvl(vrgw_i01, in_i1_b0, vrgout_b0l, vm512l, vrgw_i01, vl2) ;
      vrgw_i23 = _vel_pvfmad_vvsvMvl(vrgw_i23, in_i2_b0, vrgout_b0f, vm512f, vrgw_i23, vl2) ;
      vrgw_i23 = _vel_pvfmad_vvsvMvl(vrgw_i23, in_i3_b0, vrgout_b0l, vm512l, vrgw_i23, vl2) ;
    }
    else {
      const uint64_t in_i0_b0 = _vel_pack_f32a(pIn+0*inDim+i+0) ;
      const uint64_t in_i1_b0 = _vel_pack_f32a(pIn+0*inDim+i+1) ;
      const uint64_t in_i2_b0 = _vel_pack_f32a(pIn+0*inDim+i+2) ;
      const uint64_t in_i3_b0 = _vel_pack_f32a(pIn+0*inDim+i+3) ;
      vrgw_i01 = _vel_pvfmul_vsvMvl(in_i0_b0, vrgout_b0f, vm512f, vrgw_i01, vl2) ;
      vrgw_i01 = _vel_pvfmul_vsvMvl(in_i1_b0, vrgout_b0l, vm512l, vrgw_i01, vl2) ;
      vrgw_i23 = _vel_pvfmul_vsvMvl(in_i2_b0, vrgout_b0f, vm512f, vrgw_i23, vl2) ;
      vrgw_i23 = _vel_pvfmul_vsvMvl(in_i3_b0, vrgout_b0l, vm512l, vrgw_i23, vl2) ;
    }

    if(BATCH>=2)  {
      const uint64_t in_i0_b1 = _vel_pack_f32a(pIn+1*inDim+i+0) ;
      const uint64_t in_i1_b1 = _vel_pack_f32a(pIn+1*inDim+i+1) ;
      const uint64_t in_i2_b1 = _vel_pack_f32a(pIn+1*inDim+i+2) ;
      const uint64_t in_i3_b1 = _vel_pack_f32a(pIn+1*inDim+i+3) ;
      vrgw_i01 = _vel_pvfmad_vvsvMvl(vrgw_i01, in_i0_b1, vrgout_b1f, vm512f, vrgw_i01, vl2) ;
      vrgw_i01 = _vel_pvfmad_vvsvMvl(vrgw_i01, in_i1_b1, vrgout_b1l, vm512l, vrgw_i01, vl2) ;
      vrgw_i23 = _vel_pvfmad_vvsvMvl(vrgw_i23, in_i2_b1, vrgout_b1f, vm512f, vrgw_i23, vl2) ;
      vrgw_i23 = _vel_pvfmad_vvsvMvl(vrgw_i23, in_i3_b1, vrgout_b1l, vm512l, vrgw_i23, vl2) ;
    }
    if(BATCH>=3) {
      const uint64_t in_i0_b2 = _vel_pack_f32a(pIn+2*inDim+i+0) ;
      const uint64_t in_i1_b2 = _vel_pack_f32a(pIn+2*inDim+i+1) ;
      const uint64_t in_i2_b2 = _vel_pack_f32a(pIn+2*inDim+i+2) ;
      const uint64_t in_i3_b2 = _vel_pack_f32a(pIn+2*inDim+i+3) ;
      vrgw_i01 = _vel_pvfmad_vvsvMvl(vrgw_i01, in_i0_b2, vrgout_b2f, vm512f, vrgw_i01, vl2) ;
      vrgw_i01 = _vel_pvfmad_vvsvMvl(vrgw_i01, in_i1_b2, vrgout_b2l, vm512l, vrgw_i01, vl2) ;
      vrgw_i23 = _vel_pvfmad_vvsvMvl(vrgw_i23, in_i2_b2, vrgout_b2f, vm512f, vrgw_i23, vl2) ;
      vrgw_i23 = _vel_pvfmad_vvsvMvl(vrgw_i23, in_i3_b2, vrgout_b2l, vm512l, vrgw_i23, vl2) ;
    }
    if(BATCH>=4) {
      const uint64_t in_i0_b3 = _vel_pack_f32a(pIn+3*inDim+i+0) ;
      const uint64_t in_i1_b3 = _vel_pack_f32a(pIn+3*inDim+i+1) ;
      const uint64_t in_i2_b3 = _vel_pack_f32a(pIn+3*inDim+i+2) ;
      const uint64_t in_i3_b3 = _vel_pack_f32a(pIn+3*inDim+i+3) ;
      vrgw_i01 = _vel_pvfmad_vvsvMvl(vrgw_i01, in_i0_b3, vrgout_b3f, vm512f, vrgw_i01, vl2) ;
      vrgw_i01 = _vel_pvfmad_vvsvMvl(vrgw_i01, in_i1_b3, vrgout_b3l, vm512l, vrgw_i01, vl2) ;
      vrgw_i23 = _vel_pvfmad_vvsvMvl(vrgw_i23, in_i2_b3, vrgout_b3f, vm512f, vrgw_i23, vl2) ;
      vrgw_i23 = _vel_pvfmad_vvsvMvl(vrgw_i23, in_i3_b3, vrgout_b3l, vm512l, vrgw_i23, vl2) ;
    }
    if(BATCH>=5) {
      const uint64_t in_i0_b4 = _vel_pack_f32a(pIn+4*inDim+i+0) ;
      const uint64_t in_i1_b4 = _vel_pack_f32a(pIn+4*inDim+i+1) ;
      const uint64_t in_i2_b4 = _vel_pack_f32a(pIn+4*inDim+i+2) ;
      const uint64_t in_i3_b4 = _vel_pack_f32a(pIn+4*inDim+i+3) ;
      vrgw_i01 = _vel_pvfmad_vvsvMvl(vrgw_i01, in_i0_b4, vrgout_b4f, vm512f, vrgw_i01, vl2) ;
      vrgw_i01 = _vel_pvfmad_vvsvMvl(vrgw_i01, in_i1_b4, vrgout_b4l, vm512l, vrgw_i01, vl2) ;
      vrgw_i23 = _vel_pvfmad_vvsvMvl(vrgw_i23, in_i2_b4, vrgout_b4f, vm512f, vrgw_i23, vl2) ;
      vrgw_i23 = _vel_pvfmad_vvsvMvl(vrgw_i23, in_i3_b4, vrgout_b4l, vm512l, vrgw_i23, vl2) ;
    }
    if(BATCH>=6) {
      const uint64_t in_i0_b5 = _vel_pack_f32a(pIn+5*inDim+i+0) ;
      const uint64_t in_i1_b5 = _vel_pack_f32a(pIn+5*inDim+i+1) ;
      const uint64_t in_i2_b5 = _vel_pack_f32a(pIn+5*inDim+i+2) ;
      const uint64_t in_i3_b5 = _vel_pack_f32a(pIn+5*inDim+i+3) ;
      vrgw_i01 = _vel_pvfmad_vvsvMvl(vrgw_i01, in_i0_b5, vrgout_b5f, vm512f, vrgw_i01, vl2) ;
      vrgw_i01 = _vel_pvfmad_vvsvMvl(vrgw_i01, in_i1_b5, vrgout_b5l, vm512l, vrgw_i01, vl2) ;
      vrgw_i23 = _vel_pvfmad_vvsvMvl(vrgw_i23, in_i2_b5, vrgout_b5f, vm512f, vrgw_i23, vl2) ;
      vrgw_i23 = _vel_pvfmad_vvsvMvl(vrgw_i23, in_i3_b5, vrgout_b5l, vm512l, vrgw_i23, vl2) ;
    }
    if(BATCH>=7) {
      const uint64_t in_i0_b6 = _vel_pack_f32a(pIn+6*inDim+i+0) ;
      const uint64_t in_i1_b6 = _vel_pack_f32a(pIn+6*inDim+i+1) ;
      const uint64_t in_i2_b6 = _vel_pack_f32a(pIn+6*inDim+i+2) ;
      const uint64_t in_i3_b6 = _vel_pack_f32a(pIn+6*inDim+i+3) ;
      vrgw_i01 = _vel_pvfmad_vvsvMvl(vrgw_i01, in_i0_b6, vrgout_b6f, vm512f, vrgw_i01, vl2) ;
      vrgw_i01 = _vel_pvfmad_vvsvMvl(vrgw_i01, in_i1_b6, vrgout_b6l, vm512l, vrgw_i01, vl2) ;
      vrgw_i23 = _vel_pvfmad_vvsvMvl(vrgw_i23, in_i2_b6, vrgout_b6f, vm512f, vrgw_i23, vl2) ;
      vrgw_i23 = _vel_pvfmad_vvsvMvl(vrgw_i23, in_i3_b6, vrgout_b6l, vm512l, vrgw_i23, vl2) ;
    }
    if(BATCH>=8) {
      const uint64_t in_i0_b7 = _vel_pack_f32a(pIn+7*inDim+i+0) ;
      const uint64_t in_i1_b7 = _vel_pack_f32a(pIn+7*inDim+i+1) ;
      const uint64_t in_i2_b7 = _vel_pack_f32a(pIn+7*inDim+i+2) ;
      const uint64_t in_i3_b7 = _vel_pack_f32a(pIn+7*inDim+i+3) ;
      vrgw_i01 = _vel_pvfmad_vvsvMvl(vrgw_i01, in_i0_b7, vrgout_b7f, vm512f, vrgw_i01, vl2) ;
      vrgw_i01 = _vel_pvfmad_vvsvMvl(vrgw_i01, in_i1_b7, vrgout_b7l, vm512l, vrgw_i01, vl2) ;
      vrgw_i23 = _vel_pvfmad_vvsvMvl(vrgw_i23, in_i2_b7, vrgout_b7f, vm512f, vrgw_i23, vl2) ;
      vrgw_i23 = _vel_pvfmad_vvsvMvl(vrgw_i23, in_i3_b7, vrgout_b7l, vm512l, vrgw_i23, vl2) ;
    }

    _vel_vst_vssl(vrgw_i01, 8, pGWeight+(i+0)*outDim, vl2) ;
    _vel_vst_vssl(vrgw_i23, 8, pGWeight+(i+2)*outDim, vl2) ;

    i+=4 ;
  }
  for(; i<nInDim; i+=8) {
    __vr vrgw_i01 ;
    __vr vrgw_i23 ;
    __vr vrgw_i45 ;
    __vr vrgw_i67 ;

    if(UPDATE) {
      vrgw_i01 = _vel_vld_vssl(8, pGWeight+(i+0)*outDim, vl2) ;
      vrgw_i23 = _vel_vld_vssl(8, pGWeight+(i+2)*outDim, vl2) ;
      vrgw_i45 = _vel_vld_vssl(8, pGWeight+(i+4)*outDim, vl2) ;
      vrgw_i67 = _vel_vld_vssl(8, pGWeight+(i+6)*outDim, vl2) ;

      const uint64_t in_i0_b0 = _vel_pack_f32a(pIn+0*inDim+i+0) ;
      const uint64_t in_i1_b0 = _vel_pack_f32a(pIn+0*inDim+i+1) ;
      const uint64_t in_i2_b0 = _vel_pack_f32a(pIn+0*inDim+i+2) ;
      const uint64_t in_i3_b0 = _vel_pack_f32a(pIn+0*inDim+i+3) ;
      const uint64_t in_i4_b0 = _vel_pack_f32a(pIn+0*inDim+i+4) ;
      const uint64_t in_i5_b0 = _vel_pack_f32a(pIn+0*inDim+i+5) ;
      const uint64_t in_i6_b0 = _vel_pack_f32a(pIn+0*inDim+i+6) ;
      const uint64_t in_i7_b0 = _vel_pack_f32a(pIn+0*inDim+i+7) ;
      vrgw_i01 = _vel_pvfmad_vvsvMvl(vrgw_i01, in_i0_b0, vrgout_b0f, vm512f, vrgw_i01, vl2) ;
      vrgw_i01 = _vel_pvfmad_vvsvMvl(vrgw_i01, in_i1_b0, vrgout_b0l, vm512l, vrgw_i01, vl2) ;
      vrgw_i23 = _vel_pvfmad_vvsvMvl(vrgw_i23, in_i2_b0, vrgout_b0f, vm512f, vrgw_i23, vl2) ;
      vrgw_i23 = _vel_pvfmad_vvsvMvl(vrgw_i23, in_i3_b0, vrgout_b0l, vm512l, vrgw_i23, vl2) ;
      vrgw_i45 = _vel_pvfmad_vvsvMvl(vrgw_i45, in_i4_b0, vrgout_b0f, vm512f, vrgw_i45, vl2) ;
      vrgw_i45 = _vel_pvfmad_vvsvMvl(vrgw_i45, in_i5_b0, vrgout_b0l, vm512l, vrgw_i45, vl2) ;
      vrgw_i67 = _vel_pvfmad_vvsvMvl(vrgw_i67, in_i6_b0, vrgout_b0f, vm512f, vrgw_i67, vl2) ;
      vrgw_i67 = _vel_pvfmad_vvsvMvl(vrgw_i67, in_i7_b0, vrgout_b0l, vm512l, vrgw_i67, vl2) ;
    }
    else {
      const uint64_t in_i0_b0 = _vel_pack_f32a(pIn+0*inDim+i+0) ;
      const uint64_t in_i1_b0 = _vel_pack_f32a(pIn+0*inDim+i+1) ;
      const uint64_t in_i2_b0 = _vel_pack_f32a(pIn+0*inDim+i+2) ;
      const uint64_t in_i3_b0 = _vel_pack_f32a(pIn+0*inDim+i+3) ;
      const uint64_t in_i4_b0 = _vel_pack_f32a(pIn+0*inDim+i+4) ;
      const uint64_t in_i5_b0 = _vel_pack_f32a(pIn+0*inDim+i+5) ;
      const uint64_t in_i6_b0 = _vel_pack_f32a(pIn+0*inDim+i+6) ;
      const uint64_t in_i7_b0 = _vel_pack_f32a(pIn+0*inDim+i+7) ;
      vrgw_i01 = _vel_pvfmul_vsvMvl(in_i0_b0, vrgout_b0f, vm512f, vrgw_i01, vl2) ;
      vrgw_i01 = _vel_pvfmul_vsvMvl(in_i1_b0, vrgout_b0l, vm512l, vrgw_i01, vl2) ;
      vrgw_i23 = _vel_pvfmul_vsvMvl(in_i2_b0, vrgout_b0f, vm512f, vrgw_i23, vl2) ;
      vrgw_i23 = _vel_pvfmul_vsvMvl(in_i3_b0, vrgout_b0l, vm512l, vrgw_i23, vl2) ;
      vrgw_i45 = _vel_pvfmul_vsvMvl(in_i4_b0, vrgout_b0f, vm512f, vrgw_i45, vl2) ;
      vrgw_i45 = _vel_pvfmul_vsvMvl(in_i5_b0, vrgout_b0l, vm512l, vrgw_i45, vl2) ;
      vrgw_i67 = _vel_pvfmul_vsvMvl(in_i6_b0, vrgout_b0f, vm512f, vrgw_i67, vl2) ;
      vrgw_i67 = _vel_pvfmul_vsvMvl(in_i7_b0, vrgout_b0l, vm512l, vrgw_i67, vl2) ;
    }

    if(BATCH>=2)  {
      const uint64_t in_i0_b1 = _vel_pack_f32a(pIn+1*inDim+i+0) ;
      const uint64_t in_i1_b1 = _vel_pack_f32a(pIn+1*inDim+i+1) ;
      const uint64_t in_i2_b1 = _vel_pack_f32a(pIn+1*inDim+i+2) ;
      const uint64_t in_i3_b1 = _vel_pack_f32a(pIn+1*inDim+i+3) ;
      const uint64_t in_i4_b1 = _vel_pack_f32a(pIn+1*inDim+i+4) ;
      const uint64_t in_i5_b1 = _vel_pack_f32a(pIn+1*inDim+i+5) ;
      const uint64_t in_i6_b1 = _vel_pack_f32a(pIn+1*inDim+i+6) ;
      const uint64_t in_i7_b1 = _vel_pack_f32a(pIn+1*inDim+i+7) ;
      vrgw_i01 = _vel_pvfmad_vvsvMvl(vrgw_i01, in_i0_b1, vrgout_b1f, vm512f, vrgw_i01, vl2) ;
      vrgw_i01 = _vel_pvfmad_vvsvMvl(vrgw_i01, in_i1_b1, vrgout_b1l, vm512l, vrgw_i01, vl2) ;
      vrgw_i23 = _vel_pvfmad_vvsvMvl(vrgw_i23, in_i2_b1, vrgout_b1f, vm512f, vrgw_i23, vl2) ;
      vrgw_i23 = _vel_pvfmad_vvsvMvl(vrgw_i23, in_i3_b1, vrgout_b1l, vm512l, vrgw_i23, vl2) ;
      vrgw_i45 = _vel_pvfmad_vvsvMvl(vrgw_i45, in_i4_b1, vrgout_b1f, vm512f, vrgw_i45, vl2) ;
      vrgw_i45 = _vel_pvfmad_vvsvMvl(vrgw_i45, in_i5_b1, vrgout_b1l, vm512l, vrgw_i45, vl2) ;
      vrgw_i67 = _vel_pvfmad_vvsvMvl(vrgw_i67, in_i6_b1, vrgout_b1f, vm512f, vrgw_i67, vl2) ;
      vrgw_i67 = _vel_pvfmad_vvsvMvl(vrgw_i67, in_i7_b1, vrgout_b1l, vm512l, vrgw_i67, vl2) ;
    }
    if(BATCH>=3) {
      const uint64_t in_i0_b2 = _vel_pack_f32a(pIn+2*inDim+i+0) ;
      const uint64_t in_i1_b2 = _vel_pack_f32a(pIn+2*inDim+i+1) ;
      const uint64_t in_i2_b2 = _vel_pack_f32a(pIn+2*inDim+i+2) ;
      const uint64_t in_i3_b2 = _vel_pack_f32a(pIn+2*inDim+i+3) ;
      const uint64_t in_i4_b2 = _vel_pack_f32a(pIn+2*inDim+i+4) ;
      const uint64_t in_i5_b2 = _vel_pack_f32a(pIn+2*inDim+i+5) ;
      const uint64_t in_i6_b2 = _vel_pack_f32a(pIn+2*inDim+i+6) ;
      const uint64_t in_i7_b2 = _vel_pack_f32a(pIn+2*inDim+i+7) ;
      vrgw_i01 = _vel_pvfmad_vvsvMvl(vrgw_i01, in_i0_b2, vrgout_b2f, vm512f, vrgw_i01, vl2) ;
      vrgw_i01 = _vel_pvfmad_vvsvMvl(vrgw_i01, in_i1_b2, vrgout_b2l, vm512l, vrgw_i01, vl2) ;
      vrgw_i23 = _vel_pvfmad_vvsvMvl(vrgw_i23, in_i2_b2, vrgout_b2f, vm512f, vrgw_i23, vl2) ;
      vrgw_i23 = _vel_pvfmad_vvsvMvl(vrgw_i23, in_i3_b2, vrgout_b2l, vm512l, vrgw_i23, vl2) ;
      vrgw_i45 = _vel_pvfmad_vvsvMvl(vrgw_i45, in_i4_b2, vrgout_b2f, vm512f, vrgw_i45, vl2) ;
      vrgw_i45 = _vel_pvfmad_vvsvMvl(vrgw_i45, in_i5_b2, vrgout_b2l, vm512l, vrgw_i45, vl2) ;
      vrgw_i67 = _vel_pvfmad_vvsvMvl(vrgw_i67, in_i6_b2, vrgout_b2f, vm512f, vrgw_i67, vl2) ;
      vrgw_i67 = _vel_pvfmad_vvsvMvl(vrgw_i67, in_i7_b2, vrgout_b2l, vm512l, vrgw_i67, vl2) ;
    }
    if(BATCH>=4) {
      const uint64_t in_i0_b3 = _vel_pack_f32a(pIn+3*inDim+i+0) ;
      const uint64_t in_i1_b3 = _vel_pack_f32a(pIn+3*inDim+i+1) ;
      const uint64_t in_i2_b3 = _vel_pack_f32a(pIn+3*inDim+i+2) ;
      const uint64_t in_i3_b3 = _vel_pack_f32a(pIn+3*inDim+i+3) ;
      const uint64_t in_i4_b3 = _vel_pack_f32a(pIn+3*inDim+i+4) ;
      const uint64_t in_i5_b3 = _vel_pack_f32a(pIn+3*inDim+i+5) ;
      const uint64_t in_i6_b3 = _vel_pack_f32a(pIn+3*inDim+i+6) ;
      const uint64_t in_i7_b3 = _vel_pack_f32a(pIn+3*inDim+i+7) ;
      vrgw_i01 = _vel_pvfmad_vvsvMvl(vrgw_i01, in_i0_b3, vrgout_b3f, vm512f, vrgw_i01, vl2) ;
      vrgw_i01 = _vel_pvfmad_vvsvMvl(vrgw_i01, in_i1_b3, vrgout_b3l, vm512l, vrgw_i01, vl2) ;
      vrgw_i23 = _vel_pvfmad_vvsvMvl(vrgw_i23, in_i2_b3, vrgout_b3f, vm512f, vrgw_i23, vl2) ;
      vrgw_i23 = _vel_pvfmad_vvsvMvl(vrgw_i23, in_i3_b3, vrgout_b3l, vm512l, vrgw_i23, vl2) ;
      vrgw_i45 = _vel_pvfmad_vvsvMvl(vrgw_i45, in_i4_b3, vrgout_b3f, vm512f, vrgw_i45, vl2) ;
      vrgw_i45 = _vel_pvfmad_vvsvMvl(vrgw_i45, in_i5_b3, vrgout_b3l, vm512l, vrgw_i45, vl2) ;
      vrgw_i67 = _vel_pvfmad_vvsvMvl(vrgw_i67, in_i6_b3, vrgout_b3f, vm512f, vrgw_i67, vl2) ;
      vrgw_i67 = _vel_pvfmad_vvsvMvl(vrgw_i67, in_i7_b3, vrgout_b3l, vm512l, vrgw_i67, vl2) ;
    }
    if(BATCH>=5) {
      const uint64_t in_i0_b4 = _vel_pack_f32a(pIn+4*inDim+i+0) ;
      const uint64_t in_i1_b4 = _vel_pack_f32a(pIn+4*inDim+i+1) ;
      const uint64_t in_i2_b4 = _vel_pack_f32a(pIn+4*inDim+i+2) ;
      const uint64_t in_i3_b4 = _vel_pack_f32a(pIn+4*inDim+i+3) ;
      const uint64_t in_i4_b4 = _vel_pack_f32a(pIn+4*inDim+i+4) ;
      const uint64_t in_i5_b4 = _vel_pack_f32a(pIn+4*inDim+i+5) ;
      const uint64_t in_i6_b4 = _vel_pack_f32a(pIn+4*inDim+i+6) ;
      const uint64_t in_i7_b4 = _vel_pack_f32a(pIn+4*inDim+i+7) ;
      vrgw_i01 = _vel_pvfmad_vvsvMvl(vrgw_i01, in_i0_b4, vrgout_b4f, vm512f, vrgw_i01, vl2) ;
      vrgw_i01 = _vel_pvfmad_vvsvMvl(vrgw_i01, in_i1_b4, vrgout_b4l, vm512l, vrgw_i01, vl2) ;
      vrgw_i23 = _vel_pvfmad_vvsvMvl(vrgw_i23, in_i2_b4, vrgout_b4f, vm512f, vrgw_i23, vl2) ;
      vrgw_i23 = _vel_pvfmad_vvsvMvl(vrgw_i23, in_i3_b4, vrgout_b4l, vm512l, vrgw_i23, vl2) ;
      vrgw_i45 = _vel_pvfmad_vvsvMvl(vrgw_i45, in_i4_b4, vrgout_b4f, vm512f, vrgw_i45, vl2) ;
      vrgw_i45 = _vel_pvfmad_vvsvMvl(vrgw_i45, in_i5_b4, vrgout_b4l, vm512l, vrgw_i45, vl2) ;
      vrgw_i67 = _vel_pvfmad_vvsvMvl(vrgw_i67, in_i6_b4, vrgout_b4f, vm512f, vrgw_i67, vl2) ;
      vrgw_i67 = _vel_pvfmad_vvsvMvl(vrgw_i67, in_i7_b4, vrgout_b4l, vm512l, vrgw_i67, vl2) ;
    }
    if(BATCH>=6) {
      const uint64_t in_i0_b5 = _vel_pack_f32a(pIn+5*inDim+i+0) ;
      const uint64_t in_i1_b5 = _vel_pack_f32a(pIn+5*inDim+i+1) ;
      const uint64_t in_i2_b5 = _vel_pack_f32a(pIn+5*inDim+i+2) ;
      const uint64_t in_i3_b5 = _vel_pack_f32a(pIn+5*inDim+i+3) ;
      const uint64_t in_i4_b5 = _vel_pack_f32a(pIn+5*inDim+i+4) ;
      const uint64_t in_i5_b5 = _vel_pack_f32a(pIn+5*inDim+i+5) ;
      const uint64_t in_i6_b5 = _vel_pack_f32a(pIn+5*inDim+i+6) ;
      const uint64_t in_i7_b5 = _vel_pack_f32a(pIn+5*inDim+i+7) ;
      vrgw_i01 = _vel_pvfmad_vvsvMvl(vrgw_i01, in_i0_b5, vrgout_b5f, vm512f, vrgw_i01, vl2) ;
      vrgw_i01 = _vel_pvfmad_vvsvMvl(vrgw_i01, in_i1_b5, vrgout_b5l, vm512l, vrgw_i01, vl2) ;
      vrgw_i23 = _vel_pvfmad_vvsvMvl(vrgw_i23, in_i2_b5, vrgout_b5f, vm512f, vrgw_i23, vl2) ;
      vrgw_i23 = _vel_pvfmad_vvsvMvl(vrgw_i23, in_i3_b5, vrgout_b5l, vm512l, vrgw_i23, vl2) ;
      vrgw_i45 = _vel_pvfmad_vvsvMvl(vrgw_i45, in_i4_b5, vrgout_b5f, vm512f, vrgw_i45, vl2) ;
      vrgw_i45 = _vel_pvfmad_vvsvMvl(vrgw_i45, in_i5_b5, vrgout_b5l, vm512l, vrgw_i45, vl2) ;
      vrgw_i67 = _vel_pvfmad_vvsvMvl(vrgw_i67, in_i6_b5, vrgout_b5f, vm512f, vrgw_i67, vl2) ;
      vrgw_i67 = _vel_pvfmad_vvsvMvl(vrgw_i67, in_i7_b5, vrgout_b5l, vm512l, vrgw_i67, vl2) ;
    }
    if(BATCH>=7) {
      const uint64_t in_i0_b6 = _vel_pack_f32a(pIn+6*inDim+i+0) ;
      const uint64_t in_i1_b6 = _vel_pack_f32a(pIn+6*inDim+i+1) ;
      const uint64_t in_i2_b6 = _vel_pack_f32a(pIn+6*inDim+i+2) ;
      const uint64_t in_i3_b6 = _vel_pack_f32a(pIn+6*inDim+i+3) ;
      const uint64_t in_i4_b6 = _vel_pack_f32a(pIn+6*inDim+i+4) ;
      const uint64_t in_i5_b6 = _vel_pack_f32a(pIn+6*inDim+i+5) ;
      const uint64_t in_i6_b6 = _vel_pack_f32a(pIn+6*inDim+i+6) ;
      const uint64_t in_i7_b6 = _vel_pack_f32a(pIn+6*inDim+i+7) ;
      vrgw_i01 = _vel_pvfmad_vvsvMvl(vrgw_i01, in_i0_b6, vrgout_b6f, vm512f, vrgw_i01, vl2) ;
      vrgw_i01 = _vel_pvfmad_vvsvMvl(vrgw_i01, in_i1_b6, vrgout_b6l, vm512l, vrgw_i01, vl2) ;
      vrgw_i23 = _vel_pvfmad_vvsvMvl(vrgw_i23, in_i2_b6, vrgout_b6f, vm512f, vrgw_i23, vl2) ;
      vrgw_i23 = _vel_pvfmad_vvsvMvl(vrgw_i23, in_i3_b6, vrgout_b6l, vm512l, vrgw_i23, vl2) ;
      vrgw_i45 = _vel_pvfmad_vvsvMvl(vrgw_i45, in_i4_b6, vrgout_b6f, vm512f, vrgw_i45, vl2) ;
      vrgw_i45 = _vel_pvfmad_vvsvMvl(vrgw_i45, in_i5_b6, vrgout_b6l, vm512l, vrgw_i45, vl2) ;
      vrgw_i67 = _vel_pvfmad_vvsvMvl(vrgw_i67, in_i6_b6, vrgout_b6f, vm512f, vrgw_i67, vl2) ;
      vrgw_i67 = _vel_pvfmad_vvsvMvl(vrgw_i67, in_i7_b6, vrgout_b6l, vm512l, vrgw_i67, vl2) ;
    }
    if(BATCH>=8) {
      const uint64_t in_i0_b7 = _vel_pack_f32a(pIn+7*inDim+i+0) ;
      const uint64_t in_i1_b7 = _vel_pack_f32a(pIn+7*inDim+i+1) ;
      const uint64_t in_i2_b7 = _vel_pack_f32a(pIn+7*inDim+i+2) ;
      const uint64_t in_i3_b7 = _vel_pack_f32a(pIn+7*inDim+i+3) ;
      const uint64_t in_i4_b7 = _vel_pack_f32a(pIn+7*inDim+i+4) ;
      const uint64_t in_i5_b7 = _vel_pack_f32a(pIn+7*inDim+i+5) ;
      const uint64_t in_i6_b7 = _vel_pack_f32a(pIn+7*inDim+i+6) ;
      const uint64_t in_i7_b7 = _vel_pack_f32a(pIn+7*inDim+i+7) ;
      vrgw_i01 = _vel_pvfmad_vvsvMvl(vrgw_i01, in_i0_b7, vrgout_b7f, vm512f, vrgw_i01, vl2) ;
      vrgw_i01 = _vel_pvfmad_vvsvMvl(vrgw_i01, in_i1_b7, vrgout_b7l, vm512l, vrgw_i01, vl2) ;
      vrgw_i23 = _vel_pvfmad_vvsvMvl(vrgw_i23, in_i2_b7, vrgout_b7f, vm512f, vrgw_i23, vl2) ;
      vrgw_i23 = _vel_pvfmad_vvsvMvl(vrgw_i23, in_i3_b7, vrgout_b7l, vm512l, vrgw_i23, vl2) ;
      vrgw_i45 = _vel_pvfmad_vvsvMvl(vrgw_i45, in_i4_b7, vrgout_b7f, vm512f, vrgw_i45, vl2) ;
      vrgw_i45 = _vel_pvfmad_vvsvMvl(vrgw_i45, in_i5_b7, vrgout_b7l, vm512l, vrgw_i45, vl2) ;
      vrgw_i67 = _vel_pvfmad_vvsvMvl(vrgw_i67, in_i6_b7, vrgout_b7f, vm512f, vrgw_i67, vl2) ;
      vrgw_i67 = _vel_pvfmad_vvsvMvl(vrgw_i67, in_i7_b7, vrgout_b7l, vm512l, vrgw_i67, vl2) ;
    }

    _vel_vst_vssl(vrgw_i01, 8, pGWeight+(i+0)*outDim, vl2) ;
    _vel_vst_vssl(vrgw_i23, 8, pGWeight+(i+2)*outDim, vl2) ;
    _vel_vst_vssl(vrgw_i45, 8, pGWeight+(i+4)*outDim, vl2) ;
    _vel_vst_vssl(vrgw_i67, 8, pGWeight+(i+6)*outDim, vl2) ;
  }
}


extern "C"
vednnError_t vednnLinearBackwardWeight_o2XU128_woaligned(
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
  int64_t batchRemain = nBatch % 8 ;

  switch( batchRemain ) {
  case 1 :
    func<1,false>(inDim, outDim, inDimEnd-inDimBegin,
	          pIn+inDimBegin+n*inDim, pGOut+n*outDim, pGWeight+inDimBegin*outDim) ;
    n+=1 ;
    break ;
  case 2 :
   func<2,false>(inDim, outDim, inDimEnd-inDimBegin,
	         pIn+inDimBegin+n*inDim, pGOut+n*outDim, pGWeight+inDimBegin*outDim) ;
    n+=2 ;
    break ;
  case 3 :
    func<3,false>(inDim, outDim, inDimEnd-inDimBegin,
	          pIn+inDimBegin+n*inDim, pGOut+n*outDim, pGWeight+inDimBegin*outDim) ;
    n+=3 ;
    break ;
  case 4 :
    func<4,false>(inDim, outDim, inDimEnd-inDimBegin,
	          pIn+inDimBegin+n*inDim, pGOut+n*outDim, pGWeight+inDimBegin*outDim) ;
    n+=4 ;
    break ;
  case 5 :
    func<5,false>(inDim, outDim, inDimEnd-inDimBegin,
	          pIn+inDimBegin+n*inDim, pGOut+n*outDim, pGWeight+inDimBegin*outDim) ;
    n+=5 ;
    break ;
  case 6 :
    func<6,false>(inDim, outDim, inDimEnd-inDimBegin,
	          pIn+inDimBegin+n*inDim, pGOut+n*outDim, pGWeight+inDimBegin*outDim) ;
    n+=6 ;
    break ;
  case 7 :
    func<7,false>(inDim, outDim, inDimEnd-inDimBegin,
	          pIn+inDimBegin+n*inDim, pGOut+n*outDim, pGWeight+inDimBegin*outDim) ;
    n+=7 ;
    break ;
  default :
    if( nBatch >= 8 ) {
      func<8,false>(inDim, outDim, inDimEnd-inDimBegin,
		    pIn+inDimBegin+n*inDim, pGOut+n*outDim, pGWeight+inDimBegin*outDim) ;
      n+=8 ;
    }
    break ;
  }

  for(; n<nBatch; n+=8) {
    func<8,true>(inDim, outDim, inDimEnd-inDimBegin,
		 pIn+inDimBegin+n*inDim, pGOut+n*outDim, pGWeight+inDimBegin*outDim) ;
  }

  return VEDNN_SUCCESS ;
}
