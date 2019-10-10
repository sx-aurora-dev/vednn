/** \file
 * Use dllbuild.hpp to create a 'make' subdirectory, create lib, read symbols.
 */
#include "dllFileAux.hpp"   // mostly about string productions for declarations etc.
#include "dllbuild.hpp"
#include <sstream>
#include <fstream>
#include <iomanip>

using namespace std;
using namespace cprog;

#if 0
/** the vednn \b impl call (not the public vednn API one) */
static char const* decl_ConvolutionForward =
"vednnError_t (*vednnConvolutionForward)("
"\n    const vednnTensorParam_t      * restrict pParamIn,"
"\n    const void                    * restrict  pDataIn,"
"\n    const vednnFilterParam_t      * restrict pParamKernel,"
"\n    const void                    * restrict  pDataKernel,"
"\n    const vednnConvolutionParam_t * restrict pParamConv,"
"\n    const vednnTensorParam_t      * restrict pParamOut,"
"\n    void                          * restrict  pDataOut"
"\n)";
#endif

/** based on a very short (slow) direct_default3.c */
DllFile cjitConvolutionForward1( struct param const* const p )
{
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
    DllFileAux dfx("Convolution","Forward");
    std::string parmstr = paramString(p);
    df.basename = "cjitConvFwd01_"+parmstr;
    cout<<" df.basename = "<<df.basename<<endl;

    // we use intrinsics.  suffix matches build recipe in "bin.mk"
    df.suffix = "-vi.c";

    Cunit pr("program");
    auto& includes = pr["includes"]<<Endl;
    pr["macros"]<<"\n#define VLEN (256)"
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
    includes>>cstr(#include "vednnx.h");
    std::string fn_declare(CONVX_FWD_DECL(+df.basename+));
    cout<<fn_declare<<endl;
    prefix_lines(cout,fn_declare,"--prefixed--  ")<<"\n";
#elif 1 // even more portable (XXX get rid of CONVX_FWD_DECL macro !!!!!!)
    includes>>CSTR(#include "vednnx.h");
    // NOTE: perhaps "restrict", depending on compiler
    std::string fn_declare(
            alignRight("*",
            multiReplace(",",",\n    ",
                "vednnError_t "+df.basename+"(\n    "
                +squishAfter(
                    CSTR(CONVX_FWD_ORDER(VEDNN_PARAMS_CONV_FORWARD,VEDNN_DATARG_CONV_FORWARD))
                    ,",",  " \t")
                +"\n    )"
                )
            )
            )
            ;
    cout<<fn_declare<<endl;
    prefix_lines(cout,fn_declare,"--prefixed--  ")<<"\n";
#endif
    std::string fn_ok_declare(alignRight("*",multiReplace(",",",\n    ",
                    "vednnError_t "+df.basename+"_ok(\n    "
                    +squishAfter(CSTR(VEDNN_PARAMS_CONV_FORWARD), ",",  " \t")
                    +"\n    )"
                    )));
    std::string fn_rtok_declare(alignRight("*",multiReplace(",",",\n    ",
                    "vednnError_t "+df.basename+"_rtok(\n    "
                    +squishAfter(CSTR(VEDNN_DATARG_CONV_FORWARD), ",",  " \t")
                    +"\n    )"
                    )));
    std::string layerType("vednnConvForward");
    std::string fn_Begin_declare(alignRight("*",multiReplace(",",",\n    ",
                    "vednnError_t "+df.basename+"_Begin(\n    "
                    +squishAfter(CSTR(VEDNN_PARAMS_CONV_FORWARD), ",",  " \t")
                    +"\n    )"
                    )));
    std::string fn_Next_declare(alignRight("*",multiReplace(",",",\n    ",
                    "vednnError_t "+df.basename+"_Next(\n    "
                    +squishAfter(layerType+"Impls* current,"
                        CSTR(VEDNN_PARAMS_CONV_FORWARD), ",",  " \t")
                    +"\n    )"
                    )));
    std::string fn_realNext_declare(alignRight("*",multiReplace(",",",\n    ",
                    "vednnError_t "+df.basename+"_realNext(\n    "
                    +squishAfter(layerType+"Impls* current,"
                        CSTR(VEDNN_PARAMS_CONV_FORWARD,VEDNN_DATARG_CONV_FORWARD), ",",  " \t")
                    +"\n    )"
                    )));
    std::string fn_Dump_declare(alignRight("*",multiReplace(",",",\n    ",
                    "vednnError_t "+df.basename+"_Dump(\n    "
                    +squishAfter(layerType+"Impls* current,"
                        CSTR(VEDNN_PARAMS_CONV_FORWARD,VEDNN_DATARG_CONV_FORWARD), ",",  " \t")
                    +"\n    )"
                    )));
    std::string fn_Run_declare(alignRight("*",multiReplace(",",",\n    ",
                    layerType+"_out_t "+df.basename+"_Run(\n    "
                    +squishAfter(layerType+"Impls* current,"
                        CSTR(VEDNN_PARAMS_CONV_FORWARD,VEDNN_DATARG_CONV_FORWARD), ",",  " \t")
                    +"\n    )"
                    )));
    cout<<fn_ok_declare<<endl;
    cout<<fn_rtok_declare<<endl;
    cout<<"// Iterator API declarations"<<endl;
    //   note: some of these impls will re-use existing libvednn _ok functions!
    //         (so will stress the fragile VE runtime linker).
    cout<<fn_Begin_declare<<endl;
    cout<<fn_Next_declare<<endl;
    cout<<fn_realNext_declare<<endl;
    cout<<fn_Dump_declare<<endl;
    cout<<fn_Run_declare<<endl;
    df.syms.push_back(SymbolDecl(df.basename,
                "vednn ConvolutionForward "+parmstr,
                fn_declare));

    includes>>CSTR(#include "veintrin.h")
        >>"#include <stdio.h>"
        >>"#include <stdlib.h>"
        >>"#include <assert.h>"
        >>"#include <stdint.h>"
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
    df.code = pr.str();
    return df;
}

enum {
    CONV_TEST_FORWARD = 0,
    CONV_TEST_FORWARD_ADDBIAS,
    CONV_TEST_BACKWARD_DATA,
    CONV_TEST_BACKWARD_FILTER,
    CONV_TEST_BACKWARD_BIAS
};

static struct {
    char const* pName;
    int   testtype;
} tests[] = {
    { "Forward",    CONV_TEST_FORWARD } ,
    { "ForwardAddBias",    CONV_TEST_FORWARD_ADDBIAS } ,
    { "BackwardData",    CONV_TEST_BACKWARD_DATA } ,
    { "BackwardFilter", CONV_TEST_BACKWARD_FILTER } ,
    //    { "BackwardBias", CONV_TEST_BACKWARD_BIAS } ,   // not implemented
};

char const* default_parameter_file="mb8g1_ic3ih27iw270_oc16oh14ow135_kh3kw3_ph1pw1_sw2sh2_dw1dh1";
static void help(){
    printf( "\ncjitConv01 [args]"
            "\n   -p PATH   convolution parameter file"
            "\n             -- or --  a STRING like"
            "\n   -p mb27g1_ic3ih22iw22_oc100oh22ow22_kh5kw5_ph3pw3_sw1sh1_dw1dh1"
            "\n             where '_' are ignored."
            "\n   [ -p %s ] default"
            "\n   -M like -p STRING but parameters use mkl-dnn convention (dilation=0,1,...)"
            "\n             Add 1 to dilation values to comply with libvednn"
            "\n   -m|c      m: JUST run the 'make' step, building the jit lib"
            "\n             c: JUST run the 'create' step, trying to open the jit lib"
            "\n optional:"
            "\n   -T STRING test type: [Forward],"
            "\n             ForwardAddBias, BackwardData, BackwardFilter"
            "\n", default_parameter_file);
}
int main(int argc,char**argv){
#ifdef VEDNNX_DIR
    cout<<" VEDNNX_DIR = \""<<CSTR(VEDNNX_DIR)<<"\""<<endl;
#endif
    //vednnConvolutionParam_t pParamConv = {0,0};
    //struct param p = {8,1, 3,32,32, 3,32,32, 3,3, 1,1, 1,1, 1,1, "cnvname" };
    extern int optind;
    extern char *optarg;
    int opt;
    int opt_m = 0;
    int opt_c = 0;
    char const * pParamPath = NULL ;
    int testtype     = 0 ;
    char m_for_mkldnn = 'v';

#define PARAMBUFSZ 80
    char paramBuf[PARAMBUFSZ+1];
    while ((opt = getopt(argc, argv, "p:M:T:mc")) != -1) {
        switch (opt) {
        case 'M':
            m_for_mkldnn = 'm';
            // fall-through
        case 'p':
            snprintf(paramBuf,PARAMBUFSZ,"%s",optarg);
            pParamPath = &paramBuf[0];
            break;
        case 'm':
            opt_m=1; opt_c=0;
            break;
        case 'c':
            opt_m=0; opt_c=1;
            break;
        case 'T':
                     {
                         size_t found = 0;
                         for (size_t i=0; i<sizeof(tests)/sizeof(tests[0]); i++) {
                             if (strcasecmp(optarg, tests[i].pName) == 0) {
                                 testtype = tests[i].testtype ;
                                 found = 1;
                                 break;
                             }
                         }
                         if (! found )  {
                             fprintf(stderr, "Invalid test type.\n");
                             exit(1);
                         }
                     }
                     break;
        default: /* '?' */
                     fprintf(stderr, "Unknown option.\n");
                     help();
                     exit(1);
        }
    }
    if (optind < argc) {
        fprintf(stderr, "Unexpected argument after options\n");
        exit(1);
    }
    if ( pParamPath == NULL ) {
        //pParamPath = "./params/conv/alexnet.txt";
        pParamPath = default_parameter_file;
        fprintf(stderr, "Parameter file, '-p' option, defaults to %s.\n", pParamPath);
    }

    printf("CONVOLUTION TEST TYPE    = %s\n",       tests[testtype].pName) ;
    printf("PARAMETER FILE           = %s\n",      pParamPath);
    printf(" setting params...\n"); fflush(stdout);
    struct param *pParams ;
    int nParams = readParamFile( &pParams, pParamPath );
    if( nParams <= 0 ) {
        printf(" Trying as a parameter string...\n"); fflush(stdout);
        nParams = readParamString( &pParams, pParamPath, m_for_mkldnn );
        printf(" readParamstring --> %d\n",nParams); fflush(stdout);
        if( nParams <= 0 ) {
            printf("Bad params from %s\n", pParamPath); fflush(stdout);
            exit(1);
        }
    }
    printf(" got %d sets of parameters\n", nParams); fflush(stdout);

    for(int i=0; i<nParams; ++i){
        // Convolution tests don't handle some illegal cases well
        //    (segfault or bus error)
        // Some acceptable configs report bad DIFF and also get massaged
        // to "exact-sized" output height and width.
        //
        // This allows you to use randomly-edited parameter files and
        // at least avoid segfaults/bus errors/GEMM illegal value messages.
        // (exact output height width can be tricky to guess correctly).
        //
        printf("mkConsistent..%d\n",i);
        mkConsistent( &pParams[i] );
    }

    DllBuild dllbuild;
    for(int i=0; i<nParams; ++i){
        struct param const& pConv = pParams[i];
        DllFile df = cjitConvolutionForward1(&pConv);
        dllbuild.push_back(df);
    }
#if defined(__ve)
#define TMP_CJIT_LIB_DIR "tmp_cjitConv01"
#else
#define TMP_CJIT_LIB_DIR "tmp_cjitConv01-x86"
#endif
    if(!opt_c){
        dllbuild.prep("cjitConv01",TMP_CJIT_LIB_DIR);

        std::string mkEnv;
        // ISSUE: the Makefile sets up a rather long list of CFLAGS, which work:
        // Begin with CFLAGS        =
        // -O3 -finline-functions -Wall -assembly-list -g2 
        // -I/opt/nec/ve/nlc/1.0.0/include  
        // -I/usr/work0/home/nlabhpg/kruus/vednn-ek/test/vednn/include 
        // -I/usr/work0/home/nlabhpg/kruus/vednn-ek/test/vednn/include/wrap 
        // -I/usr/work0/home/nlabhpg/kruus/vednn-ek/test/vednn/include/C 
        // -Ivejit/include
        if(1){
            std::ostringstream oss;
            oss<<"CFLAGS='"
                //" -O3 -finline-functions -Wall -assembly-list -g2"
                "  -I" CSTR(VEDNNX_DIR) "/include"
                "'"
                " CXXFLAGS='"
                //" -O3 -finline-functions -Wall -assembly-list -g2"
                "  -I" CSTR(VEDNNX_DIR) "/include"
                "'"
                " LDFLAGS='"
                // empty is fine
                //"  -shared"
                " -Wl,--enable-new-dtags"
                //" -Wl,-rpath,/opt/nec/ve/musl/lib"
                //" -Wl,-rpath,/opt/nec/ve/ncc/1.7.21/lib "
                //"  -L" CSTR(VEDNNX_DIR) "/lib" " -Wl,-rpath," CSTR(VEDNNX_DIR) "/lib" // --> BUS ERROR
                //"  -L" CSTR(VEJIT_DIR)  "/lib" " -Wl,-rpath," CSTR(VEJIT_DIR)  "/lib" // --> BUS ERROR
                //" " CSTR(VEJIT_DIR) "/lib/libjit1.a"
                //" " CSTR(VEDNNX_DIR) "/lib/libvednn_sequential.a"
                //" -ldl"
                "'"
                ;
            mkEnv=oss.str();
        }
        cout<<" mkEnv:\n\t"<<mkEnv<<endl;
#if !defined(__ve)
        cout<<" STOPPING EARLY: we are an x86 executable and should not load a VE .so"<<endl;
#else
        dllbuild.make(mkEnv);
        if(1){
            cout<<"\nDllBuild expects symbols...\n";
            cout<<"Library: "<<dllbuild.getLibName()<<endl;
            for(auto const& df: dllbuild){
                cout<<"   File: "<<df.basename<<df.suffix<<" :"<<endl;
                for(auto const& sym: df.syms){
                    cout<<"      "<<sym.symbol<<endl;
                    if(!sym.comment.empty()){prefix_lines(cout, sym.comment, "        // ");cout<<endl;}
                    if(!sym.fwddecl.empty()){prefix_lines(cout, sym.fwddecl, "        ");cout<<endl;}
                }
            }
        }
#endif
        if(!opt_m){
#if !defined(__ve)
            cout<<" STOPPING EARLY: we are an x86 executable and should not load a VE .so"<<endl;
#else
            int nerr = 0;
            std::unique_ptr<DllOpen> lib;
            cout<<" dllbuild.create()... BEGIN"<<endl;
            try{
                lib = dllbuild.create(); // it is a unique_ptr<DllOpen>
            }catch(std::exception const& e){
                cout<<" dllbuild.create error: "<<e.what()<<endl;
                ++nerr;
            }catch(...){
                cout<<" dllbuild.create unknown error"<<endl;
                ++nerr;
            }
            if(nerr){
                cerr<<" Encountered "<<nerr<<" errors during C-file and dll file creation."<<endl;
                exit(2);
            }
            cout<<" dllbuild.create()... END (no errors)"<<endl;
            if(1){
                cout<<"JIT symbols:\n"
                    <<left<<setw(80)<<"Symbol"<<" "<<"Address\n"
                    <<string(80,'-')<<" "<<string(18,'-')<<"\n";
                for(auto const& x: lib->getDlsyms()){
                    cout<<setw(80)<<x.first<<" "<<x.second<<"\n";
                }
                cout.flush();
            }
            // double-check that nonsense lookup throws
            if(1){
                bool err=false;
                try{
                    void * __attribute__((unused)) absent = (*lib)["nonexistent_symbol"];
                }catch(...){
                    err=true;
                }
                if(!err) THROW("lib lookup of absent symbol failed to throw an exception (fix libjit1!)");
                cout<<" Good: (*lib)[\"nonexistent_symbol\"] threw!"<<endl;
            }
            // double-check that all syms in dllbuild are actually findable
            if(1){
                for(auto const& df: dllbuild){
                    for(auto const& sym: df.syms){
                        void* __attribute__((unused)) funcptr = (*lib)[sym.symbol]; // throw if not there
                    }
                }
                cout<<" Good: all symbols in dllbuild have addresses now!"<<endl;
            }
#endif
        }
    }else{
        assert(opt_c);
    }

    cout<<"\nGoodbye"<<endl; cout.flush();
}
// vim: ts=4 sw=4 et cindent cino=^=l0,\:.5s,=-.5s,N-s,g.5s,b1 cinkeys=0{,0},0),\:,0#,!^F,o,O,e,0=break
