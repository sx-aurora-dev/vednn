
#include "vednn.h"
#include "veintrin.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <stdint.h>

#define VLEN (256)

#ifdef __cplusplus
extern "C" {
#endif //C++

vednnError_t cjitConvFwd00(
        const vednnTensorParam_t * restrict      pParamIn,
        const void * restrict                    pDataIn,
        const vednnFilterParam_t * restrict      pParamKernel,
        const void * restrict                    pDataKernel,
        const vednnConvolutionParam_t * restrict pParamConv,
        const vednnTensorParam_t * restrict      pParamOut,
        void * restrict                          pDataOut
        ){ // fn
#define batch 8
#define group 1
#define inChannel 3
#define inHeight 32
#define inWidth 32
#define outChannel 3
#define outHeight 32
#define outWidth 32
#define kernHeight 3
#define kernWidth 3
#define strideHeight 1
#define strideWidth 1
#define padHeight 1
#define padWidth 1
#define dilationHeight 1
#define dilationWidth 1
#define inChannelGroup 3
#define outChannelGroup 3
#define inHW 1024
#define kernHW 9
#define outHW 1024
  float const * restrict pIn  = pDataIn;
  float const * restrict pKernel = pDataKernel;
  float * restrict pOut = pDataOut;
  _ve_lvl(VLEN);
  const __vr vzeros = _ve_vbrdu_vs_f32(0.0f); // lower 32-bits are zero bits, so same as _ve_pvbrd_vs_i64(0UL)
  const __vr vrseq = _ve_vseq_v();
  const int64_t sw_x_VLEN = strideWidth * VLEN;
  int64_t const vl_x_init = outWidth /*- x0=0*/ < VLEN ? outWidth /*- x0=0*/ : VLEN ;
  int64_t vl = vl_x_init;
  _ve_lvl(vl);
  __vr const vrj_init = _ve_vaddsl_vsv(-padWidth,  _ve_vmulsl_vsv(strideWidth, vrseq));
  for(int64_t n=0; n<batch; ++n){ // loop_n
    for(int64_t g=0; g<group; ++g){ // loop_g
      const int64_t outGroupOffset  = g * outChannelGroup * outHW;
      const int64_t inGroupOffset   = g * inChannelGroup * inHW;
      const int64_t kernGroupOffset = g * outChannelGroup * inChannelGroup * kernHW;
      const float *pIn_0 = pIn + inGroupOffset + (n * inChannel + 0) * inHW;
      for(int64_t k=0 ; k<outChannelGroup; ++k){ // loop_k
        int64_t outIndex = outGroupOffset + (n * outChannel + k) * outHW;
        const float * restrict pKern_gk = pKernel + kernGroupOffset
                                        + (k * inChannelGroup + 0) * kernHW;
        //int64_t kIndex_0 = kernGroupOffset + (k * inChannelGroup + 0) * kernHW;
        for(int64_t y=0 ; y<outHeight; ++y){ // loop_y
          const int64_t i = y * strideHeight - padHeight;
          int64_t kh_end=0;
          const int64_t kh_tmp = dilationHeight-i-1;
          const int64_t kh_beg= (i>=0? 0: kh_tmp / dilationHeight);
          if (i < inHeight){
            kh_end = (inHeight + kh_tmp) / dilationHeight;
            if (kh_end >= kernHeight) kh_end = kernHeight;
          }
          int64_t vl = vl_x_init;
          _ve_lvl(vl);
          __vr vrj = vrj_init;
          for(int64_t x0=0 ; x0<outWidth; x0+=VLEN){ // loop_x0
            const int64_t vl = outWidth - x0 < VLEN ? outWidth - x0: VLEN;
            _ve_lvl(vl);
            __vr vrsum = vzeros;
            for (int64_t r = kh_beg; r < kh_end; ++r){ // loop_r
              __vr vrw = vrj;
              for (int64_t s = 0; s < kernWidth; s++){ // loop_s
                __vm256 vm2 = _ve_vfmkl_mcv(VECC_GE, vrw);        // condition(0 <= w)
                __vm256 vm3 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(inWidth,vrw));  // condition(w < inWidth)
                __vm256 vm23  = _ve_andm_mmm(vm2, vm3);
                for (int64_t c = 0; c < inChannelGroup; ++c){ // loop_c
                  const float *pIn = pIn_0 + c*inHW + (i+r*dilationHeight)*inWidth
                                   + x0*strideWidth-padWidth + s*dilationWidth;
                  const float *pKerValue = pKern_gk + c*kernHW + r*kernWidth +s;
                  __vr vrin = _ve_vldu_vss(4*strideWidth,pIn) ;
                  vrin = _ve_vmrg_vvvm(vzeros, vrin, vm23) ;
                  vrsum = _ve_vfmads_vvsv(vrsum, *pKerValue, vrin) ;
                } //loop_c
                vrw = _ve_vaddsl_vsv(dilationWidth,  vrw) ; // <--- vector induced
              } //loop_s
            } //loop_r
            _ve_vstu_vss(vrsum, 4, pOut) ;
            vrj = _ve_vaddsl_vsv(sw_x_VLEN,vrj); // induce to avoid full recalc
            pOut += vl; // visible speedup cf. outIndex+=vl
          } //loop_x0
        } //loop_y
      } //loop_k
    } //loop_g
  } //loop_n
  return VEDNN_SUCCESS;
} //fn

#ifdef __cplusplus
}//extern "C"
#endif //C++


// vim: ts=4 sw=4 et cindent cino=^=l0,\:.5s,=-.5s,N-s,g.5s,b1 cinkeys=0{,0},0),\:,0#,!^F,o,O,e,0=break
