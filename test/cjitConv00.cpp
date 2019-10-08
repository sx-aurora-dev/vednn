/** \file
 * Take cjitConv00 and add capability to read paramFile [or paramString]
 */
//#include "cjitConv00.hpp"
#include "cblock.hpp"
#include "conv_test_param.h"
#include "stringutil.hpp"
#include <string>
#include <sstream>
#include <fstream>

using namespace std;
using namespace cprog;

/** based on a very short (slow) direct_default3.c */
//std::string cjitConvolutionForward00( vednnConvolutionParam_t const* const p )
//std::string cjitConvolutionForward00( vednnConvolutionParam_t const* const p )
std::string cjitConvolutionForward00( struct param const* const p )
{
    Cunit pr("program");
    pr["includes"]<<Endl<<CSTR(#include "vednn.h")
        >>CSTR(#include "veintrin.h")
        >>"#include <stdio.h>"
        >>"#include <stdlib.h>"
        >>"#include <assert.h>"
        >>"#include <stdint.h>"
        ;
    pr["macros"]<<"\n#define VLEN (256)"
        ;
    //auto & fns = mk_extern_c(pr,"extern_C").after(pr["/macros"])["body"];
    auto & fns = mk_extern_c(pr,"extern_C")["body"];
    //auto & fns = mk_extern_c(pr,"extern_C")["body/.."];

#if 0
    // vednn.h
vednnError_t vednnConvolutionForward(
    const vednnTensorParam_t 		*pParamIn,
    const void 				*pDataIn,
    const vednnFilterParam_t		*pParamKernel,
    const void 				*pDataKernel,
    const vednnTensorParam_t 		*pParamOut,
    void 				*pDataOut,
    const vednnConvolutionParam_t	*pParamConv,
    vednnConvolutionAlgorithm_t 	algo
) ;
    // but include/C/vednnConvolutionForward.h:
typedef
vednnError_t (*vednnConvForward_t)(
    const vednnTensorParam_t * restrict 	pParamIn,
    const void * restrict 			pDataIn,
    const vednnFilterParam_t * restrict 	pParamKernel,
    const void * restrict 			pDataKernel,
    const vednnConvolutionParam_t * restrict 	pParamConv,
    const vednnTensorParam_t * restrict 	pParamOut,
    void * restrict 				pDataOut) ;
#endif

    std::string fn_declare;
    {
        std::string funcname("cjitConvFwd00");
        std::ostringstream oss;
        oss<<"vednnError_t "<<funcname<<"("
            <<"\n        const vednnTensorParam_t * restrict      pParamIn,"
            <<"\n        const void * restrict                    pDataIn,"
            <<"\n        const vednnFilterParam_t * restrict      pParamKernel,"
            <<"\n        const void * restrict                    pDataKernel,"
            <<"\n        const vednnConvolutionParam_t * restrict pParamConv,"
            <<"\n        const vednnTensorParam_t * restrict      pParamOut,"
            <<"\n        void * restrict                          pDataOut"
            <<"\n        )";
        fn_declare = oss.str();
    }
    auto& fn = mk_func(pr,"fn",fn_declare).after(fns)["body"];

    // get the vars here first.
    const int64_t batch          = p->batchNum;
    const int64_t group          = p->group;
    const int64_t inChannel      = p->inChannel;
    const int64_t inHeight       = p->inHeight;
    const int64_t inWidth        = p->inWidth;
    const int64_t outChannel     = p->outChannel;
    const int64_t outHeight      = p->outHeight;
    const int64_t outWidth       = p->outWidth;
    const int64_t kernHeight     = p->kernHeight;
    const int64_t kernWidth      = p->kernWidth;
    const int64_t strideHeight   = p->strideHeight;
    const int64_t strideWidth    = p->strideWidth;
    const int64_t padHeight      = p->padHeight;
    const int64_t padWidth       = p->padWidth;
    const int64_t dilationHeight = p->dilationHeight;
    const int64_t dilationWidth  = p->dilationWidth;
    assert( outWidth > 0 );

    const int64_t inChannelGroup  = inChannel  / group;   // equal to pDataKernel->inChannel
    const int64_t outChannelGroup = outChannel / group;   // equal to pDataKernel->outChannel

    const int64_t inHW = inHeight * inWidth;
    const int64_t kernHW = kernHeight * kernWidth;
    const int64_t outHW = outHeight * outWidth;

    // then emit them as constant cjit values (or #define them)
    // for C++, T const might be better (T const "as-good-as-macro", and properly typed)
    //#define CONST1(var) <<("\nint64_t const " #var " = "+asDec(var))
    // but for 'C', #define may hold less surprises
#define CONST1(var) >>("#define " #var " "+asDec(var))
    //auto& fn_const =
    fn["const"]
        CONST1(batch            )
        CONST1(group            )
        CONST1(inChannel        )
        CONST1(inHeight         )
        CONST1(inWidth          )
        CONST1(outChannel       )
        CONST1(outHeight        )
        CONST1(outWidth         )
        CONST1(kernHeight       )
        CONST1(kernWidth        )
        CONST1(strideHeight     )
        CONST1(strideWidth      )
        CONST1(padHeight        )
        CONST1(padWidth         )
        CONST1(dilationHeight   )
        CONST1(dilationWidth    )

        CONST1(inChannelGroup   )
        CONST1(outChannelGroup  )

        CONST1(inHW             )
        CONST1(kernHW           )
        CONST1(outHW            )
        ;
#if 0
    const float * restrict pIn     = pDataIn;
    const float * restrict pKernel = pDataKernel;
    //float * restrict const pOut    = pDataOut;
    float * restrict pOut    = pDataOut;
#endif
    auto& fn_ptrs = fn["ptrs"];
    fn_ptrs>>"float const * restrict pIn  = pDataIn;"
        >>"float const * restrict pKernel = pDataKernel;"
        >>"float * restrict pOut = pDataOut;"
        ;

    //auto& fn_vec_init =
    fn["vec_init"]
        >>"_ve_lvl(VLEN);"
        >>"const __vr vzeros = _ve_vbrdu_vs_f32(0.0f); // lower 32-bits are zero bits, so same as _ve_pvbrd_vs_i64(0UL)"
        >>"const __vr vrseq = _ve_vseq_v();"
        >>"const int64_t sw_x_VLEN = strideWidth * VLEN;"
        >>"int64_t const vl_x_init = outWidth /*- x0=0*/ < VLEN ? outWidth /*- x0=0*/ : VLEN ;"
        >>"int64_t vl = vl_x_init;"
        >>"_ve_lvl(vl);"
        >>"__vr const vrj_init = _ve_vaddsl_vsv(-padWidth,  _ve_vmulsl_vsv(strideWidth, vrseq));"
        ;

    CBLOCK_SCOPE(loop_n,"for(int64_t n=0; n<batch; ++n)",pr,fn);
    CBLOCK_SCOPE(loop_g,"for(int64_t g=0; g<group; ++g)",pr,loop_n); // OK sub-tree
    loop_g
        >>"const int64_t outGroupOffset  = g * outChannelGroup * outHW;"
        >>"const int64_t inGroupOffset   = g * inChannelGroup * inHW;"
        >>"const int64_t kernGroupOffset = g * outChannelGroup * inChannelGroup * kernHW;"
        >>"const float *pIn_0 = pIn + inGroupOffset + (n * inChannel + 0) * inHW;"
        ;
    CBLOCK_SCOPE(loop_k,"for(int64_t k=0 ; k<outChannelGroup; ++k)",pr,loop_g);
    loop_k
        >>"int64_t outIndex = outGroupOffset + (n * outChannel + k) * outHW;"
        >>"const float * restrict pKern_gk = pKernel + kernGroupOffset"
        >>"                                + (k * inChannelGroup + 0) * kernHW;"
        >>"//int64_t kIndex_0 = kernGroupOffset + (k * inChannelGroup + 0) * kernHW;"
        ;
    CBLOCK_SCOPE(loop_y,"for(int64_t y=0 ; y<outHeight; ++y)",pr,loop_k);
    loop_y
        >>"const int64_t i = y * strideHeight - padHeight;"
        >>""
        >>"int64_t kh_end=0;"
        >>"const int64_t kh_tmp = dilationHeight-i-1;"
        >>"const int64_t kh_beg= (i>=0? 0: kh_tmp / dilationHeight);"
        >>"if (i < inHeight){"
        >>"  kh_end = (inHeight + kh_tmp) / dilationHeight;"
        >>"  if (kh_end >= kernHeight) kh_end = kernHeight;"
        >>"}"
        >>""
        >>"int64_t vl = vl_x_init;"
        >>"_ve_lvl(vl);"
        >>"__vr vrj = vrj_init;"
        ;
    CBLOCK_SCOPE(loop_x0,"for(int64_t x0=0 ; x0<outWidth; x0+=VLEN)",pr,loop_y);
    loop_x0
        >>"const int64_t vl = outWidth - x0 < VLEN ? outWidth - x0: VLEN;"
        >>"_ve_lvl(vl);"
        >>"__vr vrsum = vzeros;"
        ;
    CBLOCK_SCOPE(loop_r,"for (int64_t r = kh_beg; r < kh_end; ++r)",pr,loop_x0);
    loop_r
        >>"__vr vrw = vrj;"
        ;
    CBLOCK_SCOPE(loop_s,"for (int64_t s = 0; s < kernWidth; s++)",pr,loop_r);
    loop_s
        >>"__vm256 vm2 = _ve_vfmkl_mcv(VECC_GE, vrw);        // condition(0 <= w)"
        >>"__vm256 vm3 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(inWidth,vrw));  // condition(w < inWidth)"
        >>"__vm256 vm23  = _ve_andm_mmm(vm2, vm3);"
        ;
    CBLOCK_SCOPE(loop_c,"for (int64_t c = 0; c < inChannelGroup; ++c)",pr,loop_s);
    loop_c
        >>"const float *pIn = pIn_0 + c*inHW + (i+r*dilationHeight)*inWidth"
        >>"                 + x0*strideWidth-padWidth + s*dilationWidth;"
        >>"const float *pKerValue = pKern_gk + c*kernHW + r*kernWidth +s;"
        >>"__vr vrin = _ve_vldu_vss(4*strideWidth,pIn) ;"
        >>"vrin = _ve_vmrg_vvvm(vzeros, vrin, vm23) ;"
        >>"vrsum = _ve_vfmads_vvsv(vrsum, *pKerValue, vrin) ;"
        ;
    loop_s["induce-vrw"] // BEFORE the '}' of loops_s /**/loop_s/body/induce-vrw
        >>"vrw = _ve_vaddsl_vsv(dilationWidth,  vrw) ; // <--- vector induced"
        ;
    loop_x0["induce+write"]
        >>"_ve_vstu_vss(vrsum, 4, pOut) ;"
        >>"vrj = _ve_vaddsl_vsv(sw_x_VLEN,vrj); // induce to avoid full recalc"
        >>"pOut += vl; // visible speedup cf. outIndex+=vl"
        ;
    fn["exit"]>>"return VEDNN_SUCCESS;"
        ;

    pr["end-of-file"]>>"// vim: ts=4 sw=4 et cindent cino=^=l0,\\:.5s,=-.5s,N-s,g.5s,b1 cinkeys=0{,0},0),\\:,0#,!^F,o,O,e,0=break";
    pr.v = 0;
    // Note: 'write' currently has side-effect of "emptying" the tree. Subject to change!
    //cout<<string(80,'-')<<endl;
    //pr.write(cout);
    //cout<<string(80,'-')<<endl;
    //pr.dump(cout);
    //cout<<endl;
    cout<<string(80,'-')<<pr.str() <<string(80,'-')<<endl;
    cout<<string(80,'-')<<pr.tree()<<string(80,'-')<<endl;
    return pr.str();
}

int main(int,char**){
    //vednnConvolutionParam_t pParamConv = {0,0};
    struct param p = {8,1, 3,32,32, 3,32,32, 3,3, 1,1, 1,1, 1,1, "cnvname" };
    string code = cjitConvolutionForward00( &p );
    cout<<" outputting code["<<code.size()<<"] to file cjit00.c"<<endl;
    ofstream ofs("cjit00.c");
    ofs << code;
    ofs.close();
    cout<<"\nGoodbye"<<endl; cout.flush();
}
#if 0
vednnConvolutionForward_direct_default3(
    const vednnTensorParam_t * restrict   pParamIn,
    const void * restrict       pDataIn,
    const vednnFilterParam_t * restrict   pParamKernel,
    const void * restrict       pDataKernel,
    const vednnConvolutionParam_t * restrict   pParamConv,
    const vednnTensorParam_t * restrict   pParamOut,
    void * restrict         pDataOut
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
  assert( outWidth > 0 );

  const int64_t group          = pParamConv->group;
  const int64_t strideWidth    = pParamConv->strideWidth;;
  const int64_t strideHeight   = pParamConv->strideHeight;
  const int64_t padWidth       = pParamConv->padWidth;
  const int64_t padHeight      = pParamConv->padHeight;
  const int64_t dilationWidth  = pParamConv->dilationWidth;
  const int64_t dilationHeight = pParamConv->dilationHeight;

  const int64_t inChannelGroup  = inChannel  / group;   // equal to pDataKernel->inChannel
  const int64_t outChannelGroup = outChannel / group;   // equal to pDataKernel->outChannel

  const float * restrict pIn     = pDataIn;
  const float * restrict pKernel = pDataKernel;
  //float * restrict const pOut    = pDataOut;
  float * restrict pOut    = pDataOut;

  const int64_t inHW = inHeight * inWidth;
  const int64_t kernHW = kernHeight * kernWidth;
  const int64_t outHW = outHeight * outWidth;


  _ve_lvl(VLEN) ; // <----- VERY VERY VERY IMPORTANT to remember this init !!! 1.
  const __vr vzeros = _ve_vbrdu_vs_f32(0.0f) ; // lower 32-bits are zero bits, so same as _ve_pvbrd_vs_i64(0UL)
  const __vr vrseq = _ve_vseq_v();
  const int64_t sw_x_VLEN = strideWidth * VLEN;
  int64_t const vl_x_init = outWidth /*- x0=0*/ < VLEN ? outWidth /*- x0=0*/ : VLEN ;
  int64_t vl = vl_x_init;
  _ve_lvl(vl) ;
  __vr const vrj_init = _ve_vaddsl_vsv(-padWidth,  _ve_vmulsl_vsv(strideWidth, vrseq));

  //int64_t const kByMax = 1;
  //int64_t const zero = 0;

  for (int64_t n = 0; n < batch; n++) {
    for (int64_t g = 0; g < group; g++) {
      const int64_t outGroupOffset  = g * outChannelGroup * outHW;
      const int64_t inGroupOffset   = g * inChannelGroup * inHW;
      const int64_t kernGroupOffset = g * outChannelGroup * inChannelGroup * kernHW;
      const float *pIn_0 = pIn + inGroupOffset + (n * inChannel + 0) * inHW;
      for(int64_t k=0 ; k<outChannelGroup; ++k) {

        int64_t outIndex = outGroupOffset + (n * outChannel + k) * outHW;
        const float * restrict pKern_gk = pKernel + kernGroupOffset + (k * inChannelGroup + 0) * kernHW;
        //int64_t kIndex_0 = kernGroupOffset + (k * inChannelGroup + 0) * kernHW;

        for (int64_t y=0; y<outHeight; y++) {
          const int64_t i = y * strideHeight - padHeight;

          int64_t kh_end=0;
          const int64_t kh_tmp = dilationHeight-i-1;
          const int64_t kh_beg= (i>=0? 0: kh_tmp / dilationHeight);
          if (i < inHeight){
            kh_end = (inHeight + kh_tmp) / dilationHeight;
            if (kh_end >= kernHeight) kh_end = kernHeight;
          }

          int64_t vl = vl_x_init;
          _ve_lvl(vl) ;
          __vr vrj = vrj_init;
          for ( int64_t x0=0; x0<outWidth; x0+=VLEN )
          {
            const int64_t vl = outWidth - x0 < VLEN ? outWidth - x0 : VLEN ;
            _ve_lvl(vl) ;
            __vr vrsum = vzeros;
            // slower:
            //    any use ov _ve_lvs_svs_u64/f32
            //    any type of blocking 'c' loop (many ways tried)
            //    clang prefetch will not compile
            //    precalc offset expressions (cannnot distribute scalar calc better than clang)
            for (int64_t r = kh_beg; r < kh_end; ++r) {
              //const int64_t h = i + r * dilationHeight; // kh_beg,kh_end guarantee h in [0,outHeight)
              __vr vrw = vrj;
              for (int64_t s = 0; s < kernWidth; s++) {
                __vm256 vm2 = _ve_vfmkl_mcv(VECC_GE, vrw) ;        // condition(0 <= w)
                __vm256 vm3 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(inWidth,vrw)) ;  // condition(w < inWidth)
                __vm256 vm23  = _ve_andm_mmm(vm2, vm3) ;
                for (int64_t c = 0; c < inChannelGroup; ++c)
                {
                  const float *pIn = pIn_0 + c*inHW + (i+r*dilationHeight)*inWidth + x0*strideWidth-padWidth + s*dilationWidth;

                  const float *pKerValue = pKern_gk + c*kernHW + r*kernWidth +s;
                  __vr vrin = _ve_vldu_vss(4*strideWidth,pIn) ;
                  vrin = _ve_vmrg_vvvm(vzeros, vrin, vm23) ;
                  vrsum = _ve_vfmads_vvsv(vrsum, *pKerValue, vrin) ;
                } // inChannel

                vrw = _ve_vaddsl_vsv(dilationWidth,  vrw) ; // <--- vector induced (not fully calc)
              } // s .. kernWidth
            } // r .. kernHeight
            //_ve_vstu_vss(vrsum, 4, pOut+outIndex) ;
            _ve_vstu_vss(vrsum, 4, pOut) ;
            vrj = _ve_vaddsl_vsv(sw_x_VLEN,vrj); // induce to avoid full recalc
            //outIndex += vl ; /* MUST always execute (before break) */
            pOut += vl; // visible speedup
          } // x
        } // y
      } //k..kMax..kBy (outChannelGroup)
    } // group
  } // batch

  return VEDNN_SUCCESS;
}
#endif
// vim: ts=4 sw=4 et cindent cino=^=l0,\:.5s,=-.5s,N-s,g.5s,b1 cinkeys=0{,0},0),\:,0#,!^F,o,O,e,0=break
