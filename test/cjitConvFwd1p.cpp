#include "cjitConv.hpp"
#include "dllFileAux.hpp"   // strings for declarations, paramString
#include <string>
using namespace std;
using namespace cprog;

#define CONST1(var) >>("#define " #var " "+asDec(var))
#define FREE1(var) >>("#undef " #var)

#ifndef VEL_BUG
/** 1 means enable the SHOULD_NOT code (bug workaround) */
#define VEL_BUG 0
#endif

#ifndef KBYMAX
/** kByMax is chosen from {1,2,4,8} */
#define KBYMAX 8
#endif

/// If loop_x0 update and exit test is at end of loop_x0 ...
#define LOOP_X0_UPDATE \
            "x0 += vl_x_init; /*original used VLEN, but vl_x_init is also compile-time const*/;\n" \
            "vl = outWidth - x0;\n" \
            "if( vl <= 0 ) break;\n" \
            "vl = vl < vl_x_init? vl: vl_x_init;\n" \
            "NO_SET_VLEN(vl);\n"

/** this kByMax 8 is based on implementation direct_default2p.c, POUTX==1 */
DllFile cjitConvolutionForward1p( struct param const* const p )
{
    int const verbose=0;
    string const impl = "cjitConvFwd1p";
    DllFile df; // return value
    //DllFileAux dfx("Convolution","Forward");
    std::string parmstr = paramString(p);
    df.basename = impl+"_"+parmstr;
    cout<<impl<<" : df.basename = "<<df.basename<<endl;

    // we use intrinsics.  suffix matches build recipe in "bin.mk"
    df.suffix = "-vi.c";

    Cunit pr("program");
    pr.v = verbose;     // default is quite lengthy!

    int64_t const kByMax = KBYMAX; // kByMax is chosen from {1,2,4,8}
    cout<<impl<<" KBYMAX="<<KBYMAX<<" kByMax="<<kByMax<<endl;

    auto& includes = pr["includes"]<<Endl;
    includes
        >>CSTR(#include "vednn.h")
        >>CSTR(#if __has_include("vednnx.h")) // an old clang directive
        >>CSTR(#include "vednnx.h")
        >>CSTR(#endif)
        //>>CSTR(#include "veintrin.h")
        >>CSTR(#include "velintrin.h")
        >>"#include <stdio.h>"
        >>"#include <stdlib.h>"
        >>"#include <assert.h>"
        >>"#include <stdint.h>"
        ;
        ;
    std::string fn_declare = "vednnError_t "+df.basename+"(\n    "+
        multiReplace(",",",\n    ", CSTR(CONVX_FWD_ORDER(
                    VEDNN_PARAMS_CONV_FORWARD,
                    VEDNN_DATARG_CONV_FORWARD))) +"\n)"
        ;
    df.syms.push_back(SymbolDecl(df.basename,
                "vednn ConvolutionForward "+paramString(p),
                fn_declare));

    pr["macros"]<<"\n"
        >>"#define NO_SET_VLEN(...)"
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
        ;

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
        >>"float * restrict const pOut = pDataOut;"
        ;

    auto& fn_vec_init =
        fn["vec_init"]
        >>"// TODO VLEN-->vl_x_init (also a compile-time const, but more robust to future change of vlen)"
        >>"// TODO fused-loop opt to handle outWidth much smaller than VLEN (i.e. low vl_x_init)"
        //>>"NO_SET_VLEN(vl_x_init);"
        >>"const __vr vzeros = _vel_vbrds_vsl(0.0f, vl_x_init );"
        >>"const __vr vrseq = _vel_vseq_vl(vl_x_init);"
        >>"int64_t vl = vl_x_init;"
        //>>"NO_SET_VLEN(vl);"
        >>"float * restrict pOutx = pDataOut;"
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
    // Here, we will follow direct_default2.c hand-unroll mechanics
    //CBLOCK_SCOPE(loop_k,"for(int64_t k=0 ; k<outChannelGroup; ++k)",pr,loop_g);
    // The above loop is hand-unrolled, in JIT fashion.
    // by checking end bits of outChannelGroup (a known constant)
    //   k ~ loop index [0,outChannelGroup)
    //   kBy ~ current unroll for k, in {1,2,4,8}
    //   kMax ~ k+kBy for single-time kBy, outChannelGroup for final unroll
    loop_g>>"int64_t k = 0;"; // loop_k index, outside loops for unrolling
    int64_t k=0;        // 0 .. outChannelGroup-1
    int64_t kBy = 1;    // 1,2,4,... kByMax
    int64_t kMax = k;   // min of k+kBy or outChannelGroup
    if( k<outChannelGroup ){
        if( (outChannelGroup & kBy) != 0 ) kMax = k+kBy;
        if ( kBy == kByMax || kMax>outChannelGroup ) kMax = outChannelGroup;
        loop_g[string("kBy+")+hexdec(kBy)]
            <<"#if "<<((kMax == outChannelGroup || k<kMax)? "1": "0")
            <<" // loop_g : k in [0,outChannelGroup="<<asDec(outChannelGroup)
            <<") unroll k by "<<asDec(kBy)
            ;
        if( k<kMax ){
            CBLOCK_SCOPE(scope_kMax,"",pr,loop_g);
            scope_kMax.DEF(kBy).DEF(kMax);
            CBLOCK_SCOPE(loop_k,"for(; k<kMax; k+=kBy)",pr,scope_kMax);
            loop_k
                >>"CHK(pOutx == pOut + outGroupOffset + (n * outChannel + k) * outHW);"
                >>"const float * restrict pKern_gk = pKernel + kernGroupOffset"
                >>"                                + (k * inChannelGroup + 0/*c*/) * kernHW;"
                ;
            CBLOCK_SCOPE(loop_y,"for(int64_t y=0 ; y<outHeight; ++y)",pr,loop_k);
            // this has no influence on clang unroll "wrong result" bug
            loop_y[".."]>>"#pragma nounroll";
            loop_y
                >>"const int64_t i = y * strideHeight - padHeight;"
                >>"vl = vl_x_init;"
                >>"NO_SET_VLEN(vl);"
                >>VEJ_K_BEG_END(kh_beg,kh_tmp,kh_end, kernHeight, i,inHeight, dilationHeight)
                ;
            loop_y>>"// x0+=VLEN orig., but x0+=vl_x_init also compile-time const";
            loop_y>>"// TODO: eliminate once-only loop entirely, for clarity XXX";
            CBLOCK_SCOPE(loop_x0,"for(int64_t x0=0 ; x0<outWidth; x0+=vl_x_init)",pr,loop_y);
            loop_x0
                >>"vl = outWidth - x0 < vl_x_init ? outWidth - x0: vl_x_init;"
                >>"NO_SET_VLEN(vl);"
                >>"__vr vrsum = vzeros;"
                ;
            vrj_induce(loop_x0); // vrj ~ vector of input x values
            loop_x0[".."]>>"#pragma nounroll";
            CBLOCK_SCOPE(loop_r,"for (int64_t r = kh_beg; r < kh_end; ++r)",pr,loop_x0);
            loop_r[".."]>>"#pragma nounroll";
            loop_r
                >>"__vr vrw = vrj;"
                ;
            CBLOCK_SCOPE(loop_s,"for (int64_t s = 0; s < kernWidth; ++s)",pr,loop_r);
            loop_s[".."]>>"#pragma nounroll";
            loop_s>>"__vm256 vm23 = " VEL_VFMK_mvs_0_TO(vrw,inWidth,vl) ";";
            CBLOCK_SCOPE(loop_c,"for (int64_t c = 0; c < inChannelGroup; ++c)",pr,loop_s);
            loop_c[".."]>>"#pragma nounroll";
            loop_c
                >>"const float *pIn = pIn_0 + c*inHW + (i+r*dilationHeight)*inWidth"
                >>"                 + x0*strideWidth-padWidth + s*dilationWidth;"
                >>"const float *pKerValue = pKern_gk + c*kernHW + r*kernWidth +s;"
                >>"__vr vrin = _vel_vldu_vssl(4*strideWidth,pIn, vl);"
                >>"vrsum = _vel_vfmads_vvsvmvl(vrsum, *pKerValue, vrin, vm23, vrsum, vl);"
                ;
            loop_s["induce-vrw"] // BEFORE the '}' of loops_s /**/loop_s/body/induce-vrw
                >>"vrw = _vel_vaddsl_vsvl(dilationWidth,  vrw, vl); // <--- vector induced"
                ;
            loop_x0["induce+write"]
                >>"_vel_vstu_vssl(vrsum, 4, pOutx, vl);"
                >>"pOutx += vl; // visible speedup cf. outIndex+=vl"
                >>"//"<<CSTR(printf(" k %ld vl %-3ld outIndex=%ld\n",(long)k,(long)vl,(long)(pOutx-pOut));)
                ;
            //k = kMax;
            // "simulate" effect of above JIT loop_k on 'k' (illustrative)
            for(; k<kMax; k+=kBy ) /*no-op*/;
            assert( k == kMax );
            scope_kMax["~kMax"]
                >>"CHK(pOutx == pOut + outGroupOffset + (n * outChannel + kMax) * outHW);"
                ;
        }
        loop_g[string("kBy-")+hexdec(kBy)]
            >>"#endif // unroll by "+hexdec(kBy)<<", exit w/ k="<<hexdec(k)
            <<" of outChannelGroup="<<hexdec(outChannelGroup)
            ;
        if(kByMax==1){ assert( k>=outChannelGroup ); }
    }
    if( k<outChannelGroup ){
        kBy = 2;
        if( (outChannelGroup & kBy) != 0 ) kMax = k+kBy;
        if ( kBy == kByMax || kMax>outChannelGroup ) kMax = outChannelGroup;
        loop_g[string("kBy+")+hexdec(kBy)]
            <<"#if "<<((kMax == outChannelGroup || k<kMax)? "1": "0")
            <<" // loop_k [0,outChannelGroup) unroll k by "<<hexdec(kBy)
            ;
        if( k<kMax ){
            CBLOCK_SCOPE(scope_kMax,"",pr,loop_g);
            scope_kMax.DEF(kBy).DEF(kMax)
                >>"//"<<CSTR(if(n+g==0) printf("for k = %ld, %ld, %ld...\n",(long)k,(long)kMax,(long)kBy);)
                ;
            CBLOCK_SCOPE(loop_k,"for(; k<kMax; k+=kBy)",pr,scope_kMax);
            loop_k
                >>"//CHK(pOutx == pOut + outGroupOffset + (n * outChannel + k) * outHW);"
                >>"float* pOutx1 = pOutx + outHW;"
                >>"const float * restrict pKern_gk = pKernel + kernGroupOffset"
                >>"                                + (k * inChannelGroup + 0/*c*/) * kernHW;"
                ;
            CBLOCK_SCOPE(loop_y,"for(int64_t y=0; y<outHeight; ++y)",pr,loop_k);
            loop_y[".."]>>"#pragma nounroll";
            loop_y
                >>"const int64_t i = y * strideHeight - padHeight;"
                >>"vl = vl_x_init;"
                >>"NO_SET_VLEN(vl);"
                >>VEJ_K_BEG_END(kh_beg,kh_tmp,kh_end, kernHeight, i,inHeight, dilationHeight)
                ;
            // slightly faster looping for x0... (detect vl <= 0 instead of x0 too high)
            CBLOCK_SCOPE(loop_x0,"for(int64_t x0=0; /*x0<outWidth*/; /*x0+=VLEN*/)",pr,loop_y);
            vrj_induce(loop_x0); // vrj ~ vector of input x values
            loop_x0[".."]>>"#pragma nounroll";
            loop_x0
                >>"__vr vrsum01 = vzeros;"
                ;
            CBLOCK_SCOPE(loop_r,"for (int64_t r = kh_beg; r < kh_end; ++r)",pr,loop_x0);
            loop_r[".."]>>"#pragma nounroll";
            loop_r
                >>"__vr vrw = vrj;"
                ;
            CBLOCK_SCOPE(loop_s,"for (int64_t s = 0; s < kernWidth; ++s)",pr,loop_r);
            loop_s[".."]>>"#pragma nounroll";
            loop_s>>"__vm256 vm23 = " VEL_VFMK_mvs_0_TO(vrw,inWidth,vl) ";"
                >>VEL_DECL_VM512(vmP, vm23,vm23, vl);
            CBLOCK_SCOPE(loop_c,"for (int64_t c = 0; c < inChannelGroup; ++c)",pr,loop_s);
            loop_c[".."]>>"#pragma nounroll";
            loop_c
                >>"const float *pIn = pIn_0 + c*inHW + (i+r*dilationHeight)*inWidth"
                >>"                 + x0*strideWidth-padWidth + s*dilationWidth;"
                >>"__vr vrin = _vel_vldu_vssl(4*strideWidth,pIn, vl);"
                >>"const float *pKerValue = pKern_gk + c*kernHW + r*kernWidth +s;"
                >>"const uint64_t kerValue01 = _vel_pack_f32p(pKerValue,"
                >>"                                          pKerValue + inChannelGroup*kernHW);"
                >>"__vr vrinP = _vel_vshf_vvvsl(vrin, vrin, VE_VSHUFFLE_YUZU, vl);"
                >>"vrsum01 = _vel_pvfmad_vvsvMvl(vrsum01, kerValue01, vrinP, vmP, vrsum01, vl);"
                ;
            loop_s["induce-vrw"] // BEFORE the '}' of loops_s /**/loop_s/body/induce-vrw
                >>"vrw = _vel_vaddsl_vsvl(dilationWidth, vrw, vl); // <--- vector induced"
                ;
            loop_x0["induce+write"]
                >>"_vel_vstu_vssl(vrsum01, 4, pOutx , vl);"
                >>"_vel_vstl_vssl(vrsum01, 4, pOutx1, vl);"
                >>"pOutx  += vl;"
                >>"pOutx1 += vl;"
                >>"//"<<CSTR(printf(" k %ld vl %-3ld outIndex0=%ld\n",(long)k,(long)vl,(long)(pOutx-pOut));)
                >>LOOP_X0_UPDATE
                ;
            loop_k["bump pOutx"]
                >>"pOutx += /*kBy-1*/ 1 * outHW; // inner increment is outHW, outer wants kBy*outHW"
                ;
            //k = kMax; // could be too naive?
            // "simulate" effect of above JIT loop_k on 'k' (illustrative)
            for(; k<kMax; k+=kBy ) /*no-op*/;
            assert( k == kMax );
            scope_kMax["~kMax"]
                >>"CHK(pOutx == pOut + outGroupOffset + (n * outChannel + kMax) * outHW);"
                ;
        }
        loop_g[string("kBy-")+hexdec(kBy)]
            >>"#endif // unroll by "+hexdec(kBy)<<", exit w/ k="<<hexdec(k)
            <<" of outChannelGroup="<<hexdec(outChannelGroup)
            ;
        if(kByMax==2){ assert( k>=outChannelGroup ); }
    }
    if( k<outChannelGroup ){
        kBy = 4;
        if( (outChannelGroup & kBy) != 0 ) kMax = k+kBy;
        if ( kBy == kByMax || kMax>outChannelGroup ) kMax = outChannelGroup;
        loop_g[string("kBy+")+hexdec(kBy)]
            <<"#if "<<((kMax == outChannelGroup || k<kMax)? "1": "0")
            <<" // loop_k [0,outChannelGroup) unroll k by "<<hexdec(kBy)
            ;
        if( k<kMax ){
            CBLOCK_SCOPE(scope_kMax,"",pr,loop_g);
            scope_kMax.DEF(kBy).DEF(kMax)
                >>"//"<<CSTR(if(n+g==0) printf("for k = %ld, %ld, %ld...\n",(long)k,(long)kMax,(long)kBy);)
                ;
            CBLOCK_SCOPE(loop_k,"for(; k<kMax; k+=kBy)",pr,scope_kMax);
            loop_k
                >>"float* pOutx1 = pOutx + outHW;"
                >>"float* pOutx2 = pOutx + 2*outHW;"
                >>"float* pOutx3 = pOutx + 3*outHW;"
                >>"const float * restrict pKern_gk = pKernel + kernGroupOffset"
                >>"                                + (k * inChannelGroup + 0/*c*/) * kernHW;"
                ;
            CBLOCK_SCOPE(loop_y,"for(int64_t y=0; y<outHeight; ++y)",pr,loop_k);
            loop_y[".."]>>"#pragma nounroll";
            loop_y
                >>"const int64_t i = y * strideHeight - padHeight;"
                >>"vl = vl_x_init;"
                >>"NO_SET_VLEN(vl);"
                >>VEJ_K_BEG_END(kh_beg,kh_tmp,kh_end, kernHeight, i,inHeight, dilationHeight)
                ;
            // slightly faster looping for x0... (detect vl <= 0 instead of x0 too high)
            CBLOCK_SCOPE(loop_x0,"for(int64_t x0=0; /*x0<outWidth*/; /*x0+=VLEN*/)",pr,loop_y);
            vrj_induce(loop_x0); // vrj ~ vector of input x values
            loop_x0[".."]>>"#pragma nounroll";
            loop_x0
                >>"__vr vrsum01 = vzeros;"
                >>"__vr vrsum23 = vzeros;"
                ;
            CBLOCK_SCOPE(loop_r,"for (int64_t r = kh_beg; r < kh_end; ++r)",pr,loop_x0);
            loop_r[".."]>>"#pragma nounroll";
            loop_r
                >>"__vr vrw = vrj;"
                ;
            CBLOCK_SCOPE(loop_s,"for (int64_t s = 0; s < kernWidth; ++s)",pr,loop_r);
            loop_s[".."]>>"#pragma nounroll";
            loop_s>>"__vm256 vm23 = " VEL_VFMK_mvs_0_TO(vrw,inWidth,vl) ";"
                >>VEL_DECL_VM512(vmP, vm23,vm23, vl);
            CBLOCK_SCOPE(loop_c,"for (int64_t c = 0; c < inChannelGroup; ++c)",pr,loop_s);
            loop_c[".."]>>"#pragma nounroll";
            loop_c
                >>"const float *pIn = pIn_0 + c*inHW + (i+r*dilationHeight)*inWidth"
                >>"                 + x0*strideWidth-padWidth + s*dilationWidth;"
                >>"__vr vrin = _vel_vldu_vssl(4*strideWidth,pIn, vl);"
                >>"const float *pKerValue = pKern_gk + c*kernHW + r*kernWidth +s;"
                >>"const uint64_t kerValue01 = _vel_pack_f32p("
                >>"    pKerValue,"
                >>"    pKerValue + inChannelGroup*kernHW);"
                >>"__vr vrinP = _vel_vshf_vvvsl(vrin, vrin, VE_VSHUFFLE_YUZU, vl);"
                >>"const uint64_t kerValue23 = _vel_pack_f32p("
                >>"    pKerValue + 2 * inChannelGroup * kernHW,"
                >>"    pKerValue + 3 * inChannelGroup * kernHW);"
                >>"vrsum01 = _vel_pvfmad_vvsvMvl(vrsum01, kerValue01, vrinP, vmP, vrsum01, vl);"
                >>"vrsum23 = _vel_pvfmad_vvsvMvl(vrsum23, kerValue23, vrinP, vmP, vrsum23, vl);"
                ;
            loop_s["induce-vrw"] // BEFORE the '}' of loops_s /**/loop_s/body/induce-vrw
                >>"vrw = _vel_vaddsl_vsvl(dilationWidth, vrw, vl); // <--- vector induced"
                ;
            loop_x0["induce+write"]
                >>"_vel_vstu_vssl(vrsum01, 4, pOutx , vl);"
                >>"_vel_vstl_vssl(vrsum01, 4, pOutx1, vl);"
                >>"_vel_vstu_vssl(vrsum23, 4, pOutx2, vl);"
                >>"_vel_vstl_vssl(vrsum23, 4, pOutx3, vl);"
                >>"pOutx  += vl;"
                >>"pOutx1 += vl;"
                >>"pOutx2 += vl;"
                >>"pOutx3 += vl;"
                >>LOOP_X0_UPDATE
                ;
            loop_k["bump pOutx"]
                >>"pOutx += (kBy-1) * outHW; // inner increment is outHW, outer wants kBy*outHW"
                ;
            //k = kMax; // could be too naive?
            // "simulate" effect of above JIT loop_k on 'k' (illustrative)
            for(; k<kMax; k+=kBy ) /*no-op*/;
            assert( k == kMax );
            scope_kMax["~kMax"]
                //>>"CHK(pOutx == pOut + outGroupOffset + (n * outChannel + kMax) * outHW);"
                ;
        }
        loop_g[string("kBy-")+hexdec(kBy)]
            >>"#endif // unroll by "+hexdec(kBy)<<", exit w/ k="<<hexdec(k)
            <<" of outChannelGroup="<<hexdec(outChannelGroup)
            ;
        if(kByMax==4){ assert( k>=outChannelGroup ); }
    }
    if( k<outChannelGroup ){
        kBy = 8;
        assert( kByMax == 8 );
        kMax = outChannelGroup;
        loop_g[string("kBy+")+hexdec(kBy)]
            <<"#if "<<(k<kMax? "1": "0")
            <<" // loop_k [0,outChannelGroup) unroll k by "<<asDec(kBy)
            ;
        if( k<kMax ){
            CBLOCK_SCOPE(scope_kMax,"",pr,loop_g);
            scope_kMax.DEF(kBy).DEF(kMax)
                >>"//"<<CSTR(if(n+g==0) printf("for k = %ld, %ld, %ld...\n",(long)k,(long)kMax,(long)kBy);)
                ;
            CBLOCK_SCOPE(loop_k,"for(; k<kMax; k+=kBy)",pr,scope_kMax);
            loop_k
                >>"float* pOutx1 = pOutx + outHW;"
                >>"float* pOutx2 = pOutx + 2*outHW;"
                >>"float* pOutx3 = pOutx + 3*outHW;"
                >>"float* pOutx4 = pOutx + 4*outHW;"
                >>"float* pOutx5 = pOutx + 5*outHW;"
                >>"float* pOutx6 = pOutx + 6*outHW;"
                >>"float* pOutx7 = pOutx + 7*outHW;"
                >>"const float * restrict pKern_gk = pKernel + kernGroupOffset"
                >>"                                + (k * inChannelGroup + 0/*c*/) * kernHW;"
                ;
            CBLOCK_SCOPE(loop_y,"for(int64_t y=0; y<outHeight; ++y)",pr,loop_k);
            loop_y[".."]>>"#pragma nounroll";
            loop_y
                >>"const int64_t i = y * strideHeight - padHeight;"
                >>"vl = vl_x_init;"
                >>"NO_SET_VLEN(vl);"
                >>VEJ_K_BEG_END(kh_beg,kh_tmp,kh_end, kernHeight, i,inHeight, dilationHeight)
                ;
            // slightly faster looping for x0... (detect vl <= 0 instead of x0 too high)
            CBLOCK_SCOPE(loop_x0,"for(int64_t x0=0; /*x0<outWidth*/; /*x0+=VLEN*/)",pr,loop_y);
            vrj_induce(loop_x0); // vrj ~ vector of input x values
            loop_x0[".."]>>"#pragma nounroll";
            loop_x0
                >>"__vr vrsum01 = vzeros;"
                >>"__vr vrsum23 = vzeros;"
                >>"__vr vrsum45 = vzeros;"
                >>"__vr vrsum67 = vzeros;"
                //>>"const float *pKerValue = pKernel + kernGroupOffset + k * inChannelGroup * kernHeight * kernWidth;"
                ;
            CBLOCK_SCOPE(loop_r,"for (int64_t r = kh_beg; r < kh_end; ++r)",pr,loop_x0);
            loop_r[".."]>>"#pragma nounroll";

            CBLOCK_SCOPE(loop_s,"for (int64_t s = 0; s < kernWidth; ++s)",pr,loop_r);
            loop_s[".."]>>"__vr vrw = vrj;";
            loop_s[".."]>>"#pragma nounroll";
            loop_s["last"]>>"vrw = _vel_vaddsl_vsvl(dilationWidth, vrw, vl);";
            loop_s>>"__vm256 vm23 = " VEL_VFMK_mvs_0_TO(vrw,inWidth,vl) ";"
                >>VEL_DECL_VM512(vmP, vm23,vm23, vl);

            CBLOCK_SCOPE(loop_c,"for (int64_t c = 0; c < inChannelGroup; ++c)",pr,loop_s);
            loop_c[".."]>>"#pragma nounroll";
            loop_c
                >>"const float *pIn = pIn_0 + c*inHW + (i+r*dilationHeight)*inWidth"
                >>"                 + x0*strideWidth-padWidth + s*dilationWidth;"
                >>"__vr vrin = _vel_vldu_vssl(4*strideWidth,pIn, vl);"
                >>"__vr vrinP = _vel_vshf_vvvsl(vrin, vrin, VE_VSHUFFLE_YUZU, vl);"
                >>"const float *pKerValue = pKern_gk + c*kernHW + r*kernWidth +s;"
                >>"const uint64_t kerValue01 = _vel_pack_f32p("
                >>"    pKerValue,"
                >>"    pKerValue + 1 * inChannelGroup*kernHW);"
                >>"const uint64_t kerValue23 = _vel_pack_f32p("
                >>"    pKerValue + 2 * inChannelGroup * kernHW,"
                >>"    pKerValue + 3 * inChannelGroup * kernHW);"
                >>"vrsum01 = _vel_pvfmad_vvsvMvl(vrsum01, kerValue01, vrinP, vmP, vrsum01, vl);"
                >>"vrsum23 = _vel_pvfmad_vvsvMvl(vrsum23, kerValue23, vrinP, vmP, vrsum23, vl);"
                >>"const uint64_t kerValue45 = _vel_pack_f32p("
                >>"    pKerValue + 4 * inChannelGroup * kernHW,"
                >>"    pKerValue + 5 * inChannelGroup * kernHW);"
                >>"const uint64_t kerValue67 = _vel_pack_f32p("
                >>"    pKerValue + 6 * inChannelGroup * kernHW,"
                >>"    pKerValue + 7 * inChannelGroup * kernHW);"
                >>"vrsum45 = _vel_pvfmad_vvsvMvl(vrsum45, kerValue45, vrinP, vmP, vrsum45, vl);"
                >>"vrsum67 = _vel_pvfmad_vvsvMvl(vrsum67, kerValue67, vrinP, vmP, vrsum67, vl);"
                ;
            loop_x0["induce+write"]
                >>"_vel_vstu_vssl(vrsum01, 4, pOutx , vl);"
                >>"_vel_vstl_vssl(vrsum01, 4, pOutx1, vl);"
                >>"_vel_vstu_vssl(vrsum23, 4, pOutx2, vl);"
                >>"_vel_vstl_vssl(vrsum23, 4, pOutx3, vl);"
                >>"pOutx  += vl;"
                >>"pOutx1 += vl;"
                >>"pOutx2 += vl;"
                >>"pOutx3 += vl;"
                >>"_vel_vstu_vssl(vrsum45, 4, pOutx4, vl);"
                >>"_vel_vstl_vssl(vrsum45, 4, pOutx5, vl);"
                >>"_vel_vstu_vssl(vrsum67, 4, pOutx6, vl);"
                >>"_vel_vstl_vssl(vrsum67, 4, pOutx7, vl);"
                >>"pOutx4 += vl;"
                >>"pOutx5 += vl;"
                >>"pOutx6 += vl;"
                >>"pOutx7 += vl;"
                >>LOOP_X0_UPDATE
                ;
            loop_k["bump pOutx"]
                >>"pOutx += (kBy-1) * outHW; // inner increment is outHW, outer wants kBy*outHW"
                ;
            //k = kMax; // could be too naive?
            // "simulate" effect of above JIT loop_k on 'k' (illustrative)
            for(; k<kMax; k+=kBy ) /*no-op*/;
            assert( k == kMax );
            //scope_kMax["~kMax"]>>"CHK(pOutx == pOut + outGroupOffset + (n*outChannel+kMax)*outHW);";
        }
        loop_g[string("kBy-")+hexdec(kBy)]
            >>"#endif // unroll by "+hexdec(kBy)<<", exit w/ k="<<hexdec(k)
            <<" of outChannelGroup="<<hexdec(outChannelGroup)
            ;
        if(kByMax==8){ assert( k>=outChannelGroup ); }
    }
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
