#include "cjitConv.hpp"
#include "dllFileAux.hpp"   // strings for declarations, paramString
#include <string>
using namespace std;
using namespace cprog;

#define CONST1(var) >>("#define " #var " "+asDec(var))
#define FREE1(var) >>("#undef " #var)

#ifndef VEL_BUG
/** 1 means use extra ve_lvl as workaround for clang bug */
#define VEL_BUG 0
#endif

#define VRW_INDUCE 1 /*def 1*/
#define OUTINDEX 0 /*def 0*/

/** based on a very short (slow) direct_default3.c
 * NEW: playing with blocking (from innermost loop side) */
//std::string cjitConvolutionForward00( vednnConvolutionParam_t const* const p )
//std::string cjitConvolutionForward00( vednnConvolutionParam_t const* const p )
DllFile cjitConvolutionForward1( struct param const* const p )
{
    string const impl = "cjitConvFwd1";
    int const verbose=0;
#if 0
    // vednn.h PUBLIC API
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
    // but include/C/vednnConvolutionForward.h: IMPL API
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
    DllFile df; // return value
    //DllFileAux dfx("Convolution","Forward");
    std::string parmstr = paramString(p);
    df.basename = impl+"_"+parmstr;
    cout<<impl<<" : df.basename = "<<df.basename<<endl;

    // we use intrinsics.  suffix matches build recipe in "bin.mk"
    df.suffix = "-vi.c";

    Cunit pr("program");
    pr.v = verbose;     // default is quite lengthy!

    auto& includes = pr["includes"]<<Endl;
    includes
        >>CSTR(#include "vednn.h")
        >>CSTR(#if __has_include("vednnx.h")) // an old clang directive
        >>CSTR(#include "vednnx.h")
        >>CSTR(#endif)
#if VEL_BUG
        >>CSTR(#include "veintrin.h")
#endif
        >>CSTR(#include "velintrin.h")
        >>"#include <stdio.h>"
        >>"#include <stdlib.h>"
        >>"#include <assert.h>"
        >>"#include <stdint.h>"
        ;
    pr["macros"]<<"\n"
        >>""
        >>"#if 0"
        >>"static void err_print(char const* file, int const line, char const* what, int const requirement){"
        >>"  if(!requirement){"
        >>"    "<<CSTR(printf(" Error %s:%d failed CHK: %s\n",file,line,what);)
        >>"  }else{"
        >>"    "<<"//"<<CSTR(printf("  OK   %s:%d\n",file,line);)
        >>"  }"
        >>"  fflush(stdout);"
        >>"}"
        >>"#define CHK(REQUIREMENT) err_print(__FILE__,__LINE__,#REQUIREMENT,(REQUIREMENT));"
        >>"#else"
        >>"#define CHK(REQUIREMENT) do {;}while(0)"
        >>"#endif"
        >>""
        >>"#if "<<asDec(VEL_BUG)
        >>"// sometimes enabling this can fix 'wrong result'"
        >>"//        Simple test case: jitconv -p mb64ih3ic1oc1_kh3ph0"
        >>"#define NO_SET_VLEN( VLEN ) _ve_lvl(VLEN)"
        >>""
        >>"#else // but pure vel intrinsics should do nothing"
        >>"#define NO_SET_VLEN( VLEN ) do{}while(0)"
        >>"#endif"
        ;

#if 0 // vednn.h **public** API and low-level impl call signature
    includes<<CSTR(#include "vednn.h");
    std::string fn_declare;
    {
        std::string funcname(df.basename);
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
#elif 0 // or vednnx.h and typedefs (publicized from vednn **low-level** API)
    includes>>CSTR(#include "vednnx.h");
    std::string fn_declare(CONVX_FWD_DECL(+df.basename+));
    cout<<fn_declare<<endl;
    prefix_lines(cout,fn_declare,"--prefixed--  ")<<"\n";
#else // more macro approach
    std::string fn_declare = "vednnError_t "+df.basename+"(\n    "+
        multiReplace(",",",\n    ", CSTR(CONVX_FWD_ORDER(
                    VEDNN_PARAMS_CONV_FORWARD,
                    VEDNN_DATARG_CONV_FORWARD))) +"\n)"
        ;
#endif
    df.syms.push_back(SymbolDecl(df.basename,
                "vednn ConvolutionForward "+paramString(p),
                fn_declare));

    //auto & fns = mk_extern_c(pr,"extern_C").after(pr["/macros"])["body"];
    auto & fns = mk_extern_c(pr,"extern_C")["body"];
    //auto & fns = mk_extern_c(pr,"extern_C")["body/.."];

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
    int64_t const vl_x_init = ve_vlen_suggest( outWidth );

#define DEF(VAR) def(#VAR, VAR)
    fn.DEF(batch).DEF(group).DEF(inChannel).DEF(inHeight).DEF(inWidth);
    fn.DEF(outChannel).DEF(outHeight).DEF(outWidth).DEF(kernHeight).DEF(kernWidth);
    fn.DEF(strideHeight).DEF(strideWidth).DEF(padHeight).DEF(padWidth).DEF(dilationHeight);
    fn.DEF(dilationWidth).DEF(inChannelGroup).DEF(outChannelGroup);
    fn.DEF(inHW).DEF(kernHW).DEF(outHW).DEF(vl_x_init);
    auto& fn_ptrs = fn["ptrs"];
    fn_ptrs>>"float const * restrict pIn  = pDataIn;"
        >>"float const * restrict pKernel = pDataKernel;"
#if OUTINDEX
        >>"float * restrict const pOut = pDataOut;"
#else
        >>"float * restrict pOut = pDataOut;"
#endif
        ;

    auto& fn_vec_init =
        fn["vec_init"]
        //>>"NO_SET_VLEN(vl_x_init);"
        >>"const __vr vzeros = _vel_vbrds_vsl(0.0f, vl_x_init );"
        >>"const __vr vrseq = _vel_vseq_vl(vl_x_init);"
        >>"int64_t vl = vl_x_init;"
        //>>"NO_SET_VLEN(vl);"
        ;
    vrj_init(fn_vec_init);

    CBLOCK_SCOPE(loop_n,"for(int64_t n=0; n<batch; ++n)",pr,fn);
    CBLOCK_SCOPE(loop_g,"for(int64_t g=0; g<group; ++g)",pr,loop_n); // OK sub-tree
    loop_g
        >>"const int64_t outGroupOffset  = g * outChannelGroup * outHW;"
        >>"const int64_t inGroupOffset   = g * inChannelGroup * inHW;"
        >>"const int64_t kernGroupOffset = g * outChannelGroup * inChannelGroup * kernHW;"
        >>"const float *pIn_0 = pIn + inGroupOffset + (n * inChannel + 0) * inHW;"
        ;
    //loop_g>>"#pragma clang unroll(8)"; still 4.50 ms for ve_cmpconv default
    CBLOCK_SCOPE(loop_k,"for(int64_t k=0 ; k<outChannelGroup; ++k)",pr,loop_g);
    // Debug:
    loop_k
#if OUTINDEX
        >>"int64_t outIndex = outGroupOffset + (n * outChannel + k) * outHeight*outWidth;"
#else
        //>>"int64_t outIndex = outGroupOffset + (n * outChannel + k) * outHeight*outWidth;"
        //>>"CHK( pOut == (float *)pDataOut + outIndex );"
        >>"CHK( pOut == (float *)pDataOut + (outGroupOffset + (n*outChannel+k) * outHW) );"
        //>>CSTR(printf("k%u pOut=%p pDataOut+outIndex=%p delta %lld\n",(unsigned)k,(void*)pOut,(void*)((float*)pDataOut+outIndex), (long long)(pOut - ((float*)pDataOut+outIndex)));)
        //>>"pOut = (float * restrict)pDataOut + outIndex; // fix??"
#endif
        >>"const float * restrict pKern_gk = pKernel + kernGroupOffset"
        >>"                                + (k * inChannelGroup + 0/*c*/) * kernHW;"
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
        ;
    CBLOCK_SCOPE(loop_x0,"for(int64_t x0=0 ; x0<outWidth; x0+=vl_x_init)",pr,loop_y);
    loop_x0[".."]
        >>"int64_t vl = vl_x_init;"
        >>"NO_SET_VLEN(vl);";
    loop_x0
        >>"vl = (outWidth - x0 < vl_x_init ? outWidth - x0: vl_x_init);"
        >>"NO_SET_VLEN(vl);"
        >>"__vr vrsum = vzeros;";
    vrj_induce(loop_x0);

    loop_x0["last"]
#if OUTINDEX
        >>"_vel_vstu_vssl(vrsum, 4, pOut+outIndex, vl);"
        >>"outIndex += vl;";
#else
        >>"_vel_vstu_vssl(vrsum, 4, pOut, vl) ;"
        >>"pOut += vl; // visible speedup cf. outIndex+=vl"
#endif
        ;
    CBLOCK_SCOPE(loop_r,"for (int64_t r = kh_beg; r < kh_end; ++r)",pr,loop_x0);
    CBLOCK_SCOPE(loop_s,"for (int64_t s = 0; s < kernWidth; ++s)",pr,loop_r);
#if VRW_INDUCE
    loop_s[".."]>>"__vr vrw = _vel_vor_vsvl(0,vrj,vl);";
    loop_s["last"]>>"vrw = _vel_vaddsl_vsvl(dilationWidth,  vrw, vl);";
#else
    loop_s>>"__vr const vrw = _vel_vaddsl_vsvl(s*dilationWidth,  vrj, vl);";
#endif
    loop_s>>"__vm256 vm23 = " VEL_VFMK_mvs_0_TO(vrw,inWidth,vl) ";";

#if 1 // no strided read for kernel values
    CBLOCK_SCOPE(loop_c,"for (int64_t c = 0; c < inChannelGroup; ++c)",pr,loop_s);
    loop_c
        >>"const float *pIn = pIn_0 + c*inHW + (i+r*dilationHeight)*inWidth"
        >>"                 + x0*strideWidth-padWidth + s*dilationWidth;"
        >>"const float *pKerValue = pKern_gk + c*kernHW + r*kernWidth +s;"
        //>>"const float *pKerValue = pKernel + kernGroupOffset + ((k * inChannelGroup + c) * kernHeight + r) * kernWidth + s;"
        >>"__vr vrin = _vel_vldu_vssl(4*strideWidth,pIn, vl);"
#if 0 // orig
        >>"vrin = _vel_vmrg_vvvml(_ve_vbrdu_vs_f32(0.0f, vl), vrin, vm23);"
        >>"vrsum = _vel_vfmads_vvsvl(vrsum, *pKerValue, vrin, vl);"
#else
        >>"vrsum = _vel_vfmads_vvsvmvl(vrsum, *pKerValue, vrin, vm23, vrsum, vl);"
#endif
        ;
#else
    int64_t c = 0;
#if 0 // here is the optional new code...  THIS IS EXTREMELY SLOW!
    int const cBy = (inChannelGroup > 256? 256: inChannelGroup);
    loop_s>>"int64_t c = 0;";
    fn["const"] CONST1(cBy);
    CBLOCK_SCOPE(loop_cB,"for ( ; c < inChannelGroup/cBy*cBy; c+=cBy)",pr,loop_s);
    loop_cB
        >>"const float *pKerValue = pKern_gk + c*kernHW + r*kernWidth +s;"
        >>"NO_SET_VLEN(cBy);"
        >>"__vr vKern_cBy =_vel_vldu_vssl(4*kernHW,pKerValue, vl); // vKern_cBy[0..cBy) are the kerValues"
        >>"NO_SET_VLEN(vl);"
        ;
    CBLOCK_SCOPE(loop_cc,"for (int64_t cc=0 ; cc < cBy; ++cc)",pr,loop_cB);
    loop_cc
        >>"const float *pIn = pIn_0 + (c+cc)*inHW + (i+r*dilationHeight)*inWidth"
        >>"                 + x0*strideWidth-padWidth + s*dilationWidth;"
        >>"__vr vrin = _vel_vldu_vssl(4*strideWidth,pIn, vl);"
        >>"float kerValue = _vel_lvs_svs_f32l( vKern_cBy, cc , vl);" 
        >>"vrsum = _vel_vfmads_vvsvmvl(vrsum, kerValue, vrin, vm23, vrsum, vl);"
        ;
    c = inChannelGroup/cBy*cBy; // where do we end up?
#elif 0 // here is the optional new code...  faster, but still 10x slower for large ic
    int const cBy = (inChannelGroup > 256? 256: inChannelGroup);
    loop_s>>"int64_t c = 0;";
    fn["const"] CONST1(cBy);
    fn["const"]>>"float * const kerMem = (void*)alloca(4*cBy);";
    CBLOCK_SCOPE(loop_cB,"for ( ; c < inChannelGroup/cBy*cBy; c+=cBy)",pr,loop_s);
    loop_cB
        >>"const float *pKerValue = pKern_gk + c*kernHW + r*kernWidth +s;"
        >>"NO_SET_VLEN(cBy);"
        >>"__vr vKern_cBy =_vel_vldu_vssl(4*kernHW,pKerValue, vl); // vKern_cBy[0..cBy) are the kerValues"
        >>"_vel_vstu_vssl(vKern_cBy, 4, kerMem, vl);"
        >>"NO_SET_VLEN(vl);"
        ;
    CBLOCK_SCOPE(loop_cc,"for (int64_t cc=0 ; cc < cBy; ++cc)",pr,loop_cB);
    loop_cc
        >>"const float *pIn = pIn_0 + (c+cc)*inHW + (i+r*dilationHeight)*inWidth"
        >>"                 + x0*strideWidth-padWidth + s*dilationWidth;"
        >>"__vr vrin = _vel_vldu_vssl(4*strideWidth,pIn, vl);"
        >>"vrsum = _vel_vfmads_vvsvmvl(vrsum, kerMem[cc], vrin, vm23, vrsum, vl);"
        ;
    c = inChannelGroup/cBy*cBy; // where do we end up?
    // conclusion: blocking only to pre-read kernel values is not good.
#elif 1 // now try simple blocking by 2 and switching to packed ops
    //
    //  this code block should be enabled -- but can be disabled for debug
    //  J              cjitConvFwd1  |    1x    18.490 ms ~34185103.0654  50.88G conv2
    //  cjitConvFwd1_mb1_ic3ih256oc96oh258kh5_ph3
    //
    int const cBy = 2;
    loop_s.def("cBy",cBy);
    loop_s>>"int64_t c = 0;";
    fn["const"] CONST1(cBy);
    if( inChannelGroup >= cBy ){
        CBLOCK_SCOPE(loop_cB,"for ( ; c < inChannelGroup/cBy*cBy; c+=cBy)",pr,loop_s);
        loop_cB
            >>"const float *pIn = pIn_0 + c*inHW + (i+r*dilationHeight)*inWidth"
            >>"                 + x0*strideWidth-padWidth + s*dilationWidth;"
            >>"__vr vrin = _vel_vldu_vssl(4*strideWidth,pIn, vl);"
            >>"__vr vrin2 = _vel_vldu_vssl(4*strideWidth,pIn +inHW , vl);"
            >>"const float *pKerValue = pKern_gk + c*kernHW + r*kernWidth +s;"
            >>"const uint64_t kerP = _vel_pack_f32p(pKerValue, pKerValue+kernHW);"
            >>"/*P*/ __vr vrinP = _vel_vshf_vvvsl(vrin, vrin2, VE_VSHUFFLE_YUZU, vl);"
            >>"vrPsum = _vel_pvfmad_vvsvMvl(vrPsum, kerP, vrinP, vmP, vrPsum, vl);"
            ;
        loop_s>>VEL_DECL_VM512(vmP, vm23,vm23, vl); // declare packed mask
        loop_x0>>"__vr vrPsum = vzeros;";           // introduce new summer
        loop_x0["induce+write"]                     // and how to fold new summer into vrsum
            >>"vrsum = _vel_vfadds_vvvl(vrsum,vrPsum, vl);"
            >>"__vr vrswap = _vel_vshf_vvvsl(vrPsum,vzeros,VE_VSHUFFLE_YLZL, vl);"
            >>"vrsum = _vel_vfadds_vvvl(vrsum,vrswap, vl);"
            ;
        c = inChannelGroup/cBy*cBy; // where do we end up?
    }
#else
    //loop_s>>"int64_t c = 0;"; // <--- which loop_c ?
#endif
    if( c < inChannelGroup ){
        //CBLOCK_SCOPE(loop_c,"for ( ; c < inChannelGroup; ++c)",pr,loop_s);
        CBLOCK_SCOPE(loop_c,"for (int64_t c=0 ; c < inChannelGroup; ++c)",pr,loop_s);
        loop_c
            >>"const float *pIn = pIn_0 + c*inHW + (i+r*dilationHeight)*inWidth"
            >>"                 + x0*strideWidth-padWidth + s*dilationWidth;"
            >>"const float *pKerValue = pKern_gk + c*kernHW + r*kernWidth +s;"
            >>"__vr vrin = _vel_vldu_vssl(4*strideWidth,pIn, vl);"
            >>"vrsum = _vel_vfmads_vvsvmvl(vrsum, *pKerValue, vrin, vm23, vrsum, vl);"
            ;
    }
#endif
    fn["exit"]>>"return VEDNN_SUCCESS;"
        ;

#if 0
    //
    //  To do iteration, we NEED vednnx iterator API.
    //  If we call existing functions, dlopen REQUIRES
    //     - shared libvednnx...so
    //     - or whole-archive libvednnx
    //  to resolve symbols
    //
    //  Currently, shared library is foobar, so we must whole-archive vednnx
    //  NEW: it works with ncc 2+ (glibc variant, with C files)
    //
    //  DO THIS LATER XXX  -- single-use approach is fine for now.
    //
    std::string fn_ok_declare = "\n\nvednnError_t "+df.basename+"_ok(\n    "
        +multiReplace(",",",\n    ", CSTR(VEDNN_PARAMS_CONV_FORWARD))
        +"\n)";
    df.syms.push_back(SymbolDecl(df.basename+"_ok",
                "vednn ConvolutionForward ok (param check) "+ paramString(p),
                fn_ok_declare));
    auto& fn_ok = mk_func(pr,"fn",fn_ok_declare).after(fns)["body"];
    fn_ok>>"return vednnConvolutionForward_direct_default_ok(\n    "
        CSTR(VEDNN_PARAMS_CONV_FORWARD_LIST) " );";
#endif


    pr["end-of-file"]>>"// vim: ts=4 sw=4 et cindent cino=^=l0,\\:.5s,=-.5s,N-s,g.5s,b1 cinkeys=0{,0},0),\\:,0#,!^F,o,O,e,0=break";
    pr.v = 0; // set Cuint (root) back to non-verbose

    if(verbose){ // dump to cout (debug)
        // Note: 'write' currently has side-effect of "emptying" the tree. Subject to change!
        //cout<<string(80,'-')<<endl;
        //pr.write(cout);
        //cout<<string(80,'-')<<endl;
        //pr.dump(cout);
        //cout<<endl;
        if(verbose>=1)
            cout<<string(80,'-')<<pr.str() <<string(80,'-')<<endl;
        if(verbose>=2)
            cout<<string(80,'-')<<pr.tree()<<string(80,'-')<<endl;
    }
    df.code = pr.str();
    return df;
}

// vim: ts=4 sw=4 et cindent cino=^=l0,\:.5s,=-.5s,N-s,g.5s,b1 cinkeys=0{,0},0),\:,0#,!^F,o,O,e,0=break
