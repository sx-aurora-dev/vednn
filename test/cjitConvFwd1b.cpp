/** \file
 * MSK_PRE did work, but is now broken.
 */
#include "cjitConv.hpp"
#include "dllFileAux.hpp"   // strings for declarations, paramString
#include <string>
using namespace std;
using namespace cprog;

#define CONST1(var) >>("#define " #var " "+asDec(var))
#define FREE1(var) >>("#undef " #var)
#define DEF(VAR) def(#VAR, VAR)

#ifndef VEL_BUG
/** 1 means use extra ve_lvl as workaround for clang bug */
#define VEL_BUG 0
#endif

#ifndef KBYMAX
/** kByMax is chosen from {1,2,4,8} */
#define KBYMAX 8
#endif

#define MSK_PRE 1 /*should be 1, but bugs?*/
#ifndef MVL
#define MVL 256
#endif

/**  cjitConvFwd1 + masks precalculated -- no huge speed difference from original Fwd1,
 * probably because loading a mask is done here with 4 scalar ops.
 * Likely need to block x,y so that unmasked kernels can omit the masking entirely.
 */
DllFile cjitConvolutionForward1b( struct param const* const p )
{
    ostringstream oss;
    int const verbose=0;
    string const impl = "cjitConvFwd1b";
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
    pr["macros"]//<<"\n#define VLEN (256)"
        ;

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

    // generate fwd declaration
    std::string fn_declare = "vednnError_t "+df.basename+"(\n    "+
        multiReplace(",",",\n    ", CSTR(CONVX_FWD_ORDER(
                    VEDNN_PARAMS_CONV_FORWARD,
                    VEDNN_DATARG_CONV_FORWARD))) +"\n)"
        ;

    df.syms.push_back(SymbolDecl(df.basename,
                "vednn ConvolutionForward "+paramString(p),
                fn_declare));

    //auto & fns = mk_extern_c(pr,"extern_C").after(pr["/macros"])["body"];
    auto & fns = mk_extern_c(pr,"extern_C")["body"];
    //auto & fns = mk_extern_c(pr,"extern_C")["body/.."];

    auto& fn = mk_func(pr,"fn",fn_declare).after(fns)["body"];
    pr["macros"]
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
    pr["last"]>>"#undef CHK";

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
    //int const vl_x_init = outWidth /*- x0=0*/ < MVL ? outWidth /*- x0=0*/ : MVL ; // mirror

    fn.DEF(batch).DEF(group).DEF(inChannel).DEF(inHeight).DEF(inWidth);
    fn.DEF(outChannel).DEF(outHeight).DEF(outWidth).DEF(kernHeight).DEF(kernWidth);
    fn.DEF(strideHeight).DEF(strideWidth).DEF(padHeight).DEF(padWidth).DEF(dilationHeight);
    fn.DEF(dilationWidth).DEF(inChannelGroup).DEF(outChannelGroup);
    fn.DEF(inHW).DEF(kernHW).DEF(outHW).DEF(vl_x_init);

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

    auto& fn_vec_init =
        fn["vec_init"]
        //>>"NO_SET_VLEN(vl_x_init);"
        >>"const __vr vzeros = _vel_vbrds_vsl(0.0f, vl_x_init );"
        >>"const __vr vrseq = _vel_vseq_vl(vl_x_init);"
        >>"int64_t vl = vl_x_init;"
        //>>"NO_SET_VLEN(vl);"
        ;
    vrj_init(fn_vec_init);

#if MSK_PRE == 1
    // runtime mask calc
    // vrj begins at vrj_init in loop_y:oh increments by sw * VLEN before loop_x0:ow closes
    // vrw begins at vrj in loop_r and increments by dw before loop_s:kw closes
    // so vm23 is a fn(x,s)
    // How many vm23? ((outWidth+vl_x_init-1)/vl_x_init) * kernWidth
    // Note: this option also applies to libvednn impls
    //
    size_t const nMask_xs = ((outWidth+vl_x_init-1)/vl_x_init) * kernWidth;
    auto& msk = fn["msk"]
        >>OSSFMT("size_t const nMask_xs = "<<nMask_xs<<"; // (outWidth+vl_x_init-1)/vl_x_init");
    if(nMask_xs > 2048){
        msk >>"uint64_t *vm_xs = (uint64_t*)malloc(4*nMask_xs*sizeof(uint64_t));"
            >>"CHK(vm_xs != NULL);";
        fn["last"]>>"free(vm_xs);";
    }else{
        msk >>"uint64_t vm_xs[4*nMask_xs];";
    }
    {
        CBLOCK_SCOPE(width,"",pr,msk);
        width>>"uint64_t *pvm_xs = &vm_xs[0];";
        // pvm_xs_x = pvm_xs
        CBLOCK_SCOPE(x,"for(int64_t x0=0 ; x0<outWidth; x0+=vl_x_init)",pr,width);
        vrj_induce(x); // vrj ~ vector of input x values

        //CBLOCK_SCOPE(r,"for (int64_t r = kh_beg; r < kh_end; ++r)",pr,loop_x0);
        //r >>"__vr vrw = vrj;" ;
        //loop_r vrw = vrj -----> pvm_xs = pvm_xs_x;
        x>>"__vr vrw = vrj;";

        CBLOCK_SCOPE(s,"for (int64_t s = 0; s < kernWidth; ++s)",pr,x);
        s 
            >>"__vm256 vm23 = " VEL_VFMK_mvs_0_TO(vrw,inWidth,vl) ";"
            >>"*(pvm_xs+0) = _vel_svm_sms(vm23,0);"
            >>"*(pvm_xs+1) = _vel_svm_sms(vm23,1);"
            >>"*(pvm_xs+2) = _vel_svm_sms(vm23,2);"
            >>"*(pvm_xs+3) = _vel_svm_sms(vm23,3);"
            >>"pvm_xs += 4; /*adv to next mask storage pos*/"
            >>"CHK( pvm_xs - vm_xs <= 4*nMask_xs ); // <-- failed when oc>256?"
            >>"vrw = _vel_vaddsl_vsvl(dilationWidth,  vrw, vl) ; // <--- vector induced"
            ;
        // r : pvm_xs_x = pvm_xs; ??
        // x : pvm_xs_x = pvm_xs; ??
    }
#elif MSK_PRE > 1
#error "TODO: simulate x,s loops and emit precalculated const data instead (perhaps in .rodata, not stack)"
    // XXX and at extreme, is full x,s unroll with optimized mask load
    // XXX or even blocking x loop so that "interior" masks can all be zero (done separately)
#endif

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
    loop_k
        >>"//int64_t outIndex = outGroupOffset + (n * outChannel + k) * outHW;"
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
        >>""
        >>"int64_t vl = vl_x_init;"
        //>>"NO_SET_VLEN(vl);"
#if MSK_PRE>0
        >>"uint64_t *pvm_xs_x = &vm_xs[0];"
        >>"uint64_t *pvm_xs; /* loop_r loops over mask set beginning at pvm_xs_x*/"
#endif
        ;
    CBLOCK_SCOPE(loop_x0,"for(int64_t x0=0 ; x0<outWidth; x0+=vl_x_init)",pr,loop_y);
//#if MSK_PRE==0
    vrj_induce(loop_x0); // vrj ~ vector of input x values
//#endif
    loop_x0
        >>"vl = outWidth - x0 < vl_x_init ? outWidth - x0: vl_x_init;"
        >>"NO_SET_VLEN(vl);"
        >>"__vr vrsum = vzeros;"
        ;
    CBLOCK_SCOPE(loop_r,"for (int64_t r = kh_beg; r < kh_end; ++r)",pr,loop_x0);
#if MSK_PRE==0
    loop_r >>"__vr vrw = vrj;" ;
#else 
    loop_r>>"pvm_xs = pvm_xs_x;";
#endif
    CBLOCK_SCOPE(loop_s,"for (int64_t s = 0; s < kernWidth; ++s)",pr,loop_r);
    // for kh11, clang can segfault (tries full unroll?)
    // XXX need to investigate how to limit this limit unroll a bit better TODO
#if 1 // heavy-handed fix
    loop_s[".."]>>"#pragma clang loop unroll(disable)"; // clang possible segfault?
#elif 0
    loop_s[".."]>>"#pragma unroll_and_jam"; // clang segfault?
    loop_s[".."]>>"#pragma clang loop unroll_count(5)"; // clang segfault?
#endif

#if MSK_PRE==0
    loop_s>>"__vm256 vm23 = " VEL_VFMK_mvs_0_TO(vrw,inWidth,vl) ";";
#elif 1
    loop_s>>"__vm256 vm23 = _vel_lvm_mmss(vm23,0,*(pvm_xs+0));";
    loop_s>>"vm23 = _vel_lvm_mmss(vm23,1,*(pvm_xs+1));";
    loop_s>>"vm23 = _vel_lvm_mmss(vm23,2,*(pvm_xs+2));";
    loop_s>>"vm23 = _vel_lvm_mmss(vm23,3,*(pvm_xs+3));";
#else // ahaa, better...
    loop_s>>"__vm256 vm23 = _vel_lvm_mmss(vm23,0,*(pvm_xs+0));";
    if(vl_x_init>64) loop_s>>"vm23 = _vel_lvm_mmss(vm23,1,*(pvm_xs+1));";
    if(vl_x_init>128) loop_s>>"vm23 = _vel_lvm_mmss(vm23,2,*(pvm_xs+2));";
    if(vl_x_init>192) loop_s>>"vm23 = _vel_lvm_mmss(vm23,3,*(pvm_xs+3));";
#endif
    int c=0; //mirror the loop index
#if 1 // now try simple blocking by 2 and switching to packed ops
    int const cBy = 2;
    loop_s.DEF(cBy) >>"int64_t c = 0;";
    if( inChannelGroup >= cBy ){
        CBLOCK_SCOPE(loop_cB,"for ( ; c < inChannelGroup/cBy*cBy; c+=cBy)",pr,loop_s);
        loop_cB[".."]>>"#pragma clang loop unroll(disable)";
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
        loop_s>>VEL_DECL_VM512( vmP, vm23,vm23, vl ); // declare packed mask
        loop_x0>>"__vr vrPsum = vzeros;";         // introduce new summer
        loop_x0["induce+write"]                   // and how to fold new summer into vrsum
            >>"vrsum = _vel_vfadds_vvvl(vrsum,vrPsum, vl);"
            >>"__vr vrswap = _vel_vshf_vvvsl(vrPsum,vzeros,VE_VSHUFFLE_YLZL, vl);"
            >>"vrsum = _vel_vfadds_vvvl(vrsum,vrswap, vl);"
            ;
        c = inChannelGroup/cBy*cBy; // where do we end up?
    }
#endif
    // conclusion: blocking only to pre-read kernel values is not good.
    if( c < inChannelGroup ){
        CBLOCK_SCOPE(loop_c,"for ( ; c < inChannelGroup; ++c)",pr,loop_s);
        loop_c
            >>"const float *pIn = pIn_0 + c*inHW + (i+r*dilationHeight)*inWidth"
            >>"                 + x0*strideWidth-padWidth + s*dilationWidth;"
            >>"const float *pKerValue = pKern_gk + c*kernHW + r*kernWidth +s;"
            >>"__vr vrin = _vel_vldu_vssl(4*strideWidth,pIn, vl);"
            >>"vrsum = _vel_vfmads_vvsvmvl(vrsum, *pKerValue, vrin, vm23, vrsum, vl);"
            ;
    }
    loop_s["induce-vrw"] // BEFORE the '}' of loops_s /**/loop_s/body/induce-vrw
#if MSK_PRE==0
        >>"vrw = _vel_vaddsl_vsvl(dilationWidth,  vrw, vl) ; // <--- vector induced"
#else
        >>"pvm_xs+=4;"
#endif
        ;
#if MSK_PRE==1
        //loop_r["reset-mask"]>>"pvm_xs_x = pvm_xs;";
#endif
    loop_x0["induce+write"]
        >>"_vel_vstu_vssl(vrsum, 4, pOut, vl) ;"
#if MSK_PRE>0
        >>"// vrj = _vel_vaddsl_vsvl(sw_x_VLEN,vrj, vl); // induce to avoid full recalc"
        >>"pvm_xs_x = pvm_xs;"
#endif
        >>"pOut += vl; // visible speedup cf. outIndex+=vl"
        ;
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
        cout<<string(80,'-')<<pr.str() <<string(80,'-')<<endl;
        cout<<string(80,'-')<<pr.tree()<<string(80,'-')<<endl;
    }
    df.code = pr.str();
    return df;
}

// vim: ts=4 sw=4 et cindent cino=^=l0,\:.5s,=-.5s,N-s,g.5s,b1 cinkeys=0{,0},0),\:,0#,!^F,o,O,e,0=break
