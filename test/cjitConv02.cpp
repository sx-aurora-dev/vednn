/** \file
 * demo creating full JIT convolution list entry
 *
 * Is adding the rest of libvednnx API necessary?
 * If I maintain a library of existing impls I can jit/lookup the [hashed]
 * layer parms to get the source code or library.
 */
#include "cjitConv.hpp"
#include "vednnx.h"
#include <unistd.h>  // getopt
#include <iomanip>

/** \c STR(...) eases raw text-->C-string (escaping embedded " or \\) */
#ifndef CSTR
#define CSTRING0(...) #__VA_ARGS__
#define CSTR(...) CSTRING0(__VA_ARGS__)
#endif

using namespace std;

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
    printf( "\ncjitConv02 [args]"
            "\n   -p PATH   convolution parameter file"
            "\n             -- or --  a STRING like"
            "\n   -p mb27g1_ic3ih22iw22_oc100oh22ow22_kh5kw5_ph3pw3_sw1sh1_dw1dh1"
            "\n             where '_' are ignored."
            "\n   -M like -p STRING but parameters use mkl-dnn convention (dilation=0,1,...)"
            "\n             Add 1 to dilation values to comply with libvednn"
            "\n   [ -p %s ]"
            "\n optional:"
            "\n   -T STRING test type: [Forward],"
            "\n             ForwardAddBias, BackwardData, BackwardFilter"
            "\n             *** only Forward is supported ***"
            "\n   -C test using the C api"
            "\n", default_parameter_file);
}
int main(int argc,char**argv){
    cout<<"\n\n"<<string(20,'-')<<" Running cjitConv02..."<<endl;
#ifdef VEDNNX_DIR
    cout<<" VEDNNX_DIR = \""<<CSTR(VEDNNX_DIR)<<"\""<<endl;
#endif
    //vednnConvolutionParam_t pParamConv = {0,0};
    //struct param p = {8,1, 3,32,32, 3,32,32, 3,3, 1,1, 1,1, 1,1, "cnvname" };
    extern int optind;
    extern char *optarg;
    int opt;
    char const * pParamPath = NULL ;
    int testtype     = 0 ;
    int use_c_api = 0;
    char m_for_mkldnn = 'v';

#define PARAMBUFSZ 80
    char paramBuf[PARAMBUFSZ+1];
    while ((opt = getopt(argc, argv, "p:M:T:C")) != -1) {
        switch (opt) {
          case 'M': m_for_mkldnn = 'm';
                    // fall-through
          case 'p': snprintf(paramBuf,PARAMBUFSZ,"%s",optarg);
                    pParamPath = &paramBuf[0];
                    m_for_mkldnn = 'v';
          break;
          case 'C': use_c_api=1;
          break;
          case 'T': {
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

    if(!use_c_api){ // use the C++ api (cjitConv.hpp)
        //unique_ptr<DllOpen> pLib = jitConvs(pParams, nParams,
        //        cjitConvolutionForward1);
        char const* generators[] = {"cjitConvFwd1","cjitConvFwd4",NULL};
        unique_ptr<DllOpen> pLib = jitConvs(pParams, nParams,
                generators);
        cout<<" DllOpen @ "<<pLib.get()<<endl; cout.flush();
        if(!pLib)
            THROW(" No object in unique_ptr<DllOpen> 'pLib' ?!");
        if(1){
            DllOpen& lib = *pLib;
            cout<<"JIT symbols:\n"
                <<left<<setw(80)<<"Symbol"<<" "<<"Address\n"
                <<string(80,'-')<<" "<<string(18,'-')<<"\n";
            cout.flush();
            for(auto const& x: lib.getDlsyms()){
                cout<<setw(80)<<x.first<<" "<<x.second<<"\n";
                cout.flush();
            }
            //
            // Issue?
            //   via jitConvs, I have FORGOTTEN the 'DllBuild', and
            //   esp. the detail symbol info in its 'DllFile's
            //
        }

        cout<<" Freeing pLib..."<<endl; cout.flush();
        pLib.reset();
    }else{ // use the C api (cjitConv.h)
        {
            // we need to remember the list head, cjSyms, to free memory later
            char const* generators[] = {"cjitConvFwd1","cjitConvFwd4",NULL};
            CjitOpt cjitOpt= { NULL, 0, 0 }; // { "tmp_cjitConv", full prep, full build }
            CjitSyms const* const cjitsyms = cjitSyms(pParams, nParams,
                    generators, &cjitOpt );
            CjitSym const* const cjitsym = cjitsyms->syms;
            {
                cout<<left<<setw(80)<<"C API sees JIT symbols:"
                    <<" "<<"Address\n"
                    <<string(80,'-')<<" "<<string(18,'-')<<"\n";
                cout.flush();
                for(CjitSym const* p=cjitsym; p->sym!=nullptr && p->ptr!=nullptr; ++p){
                    cout<<setw(80)<<p->sym<<" "<<p->ptr<<"\n";
                    cout.flush();
                }
            }
            cout<<" Freeing CjitSym* list..."<<endl; cout.flush();
            cjitSyms_free(cjitsyms);
        }
    }
    cout<<"\nGoodbye"<<endl; cout.flush();
    return 0;
}
// vim: ts=4 sw=4 et cindent cino=^=l0,\:.5s,=-.5s,N-s,g.5s,b1 cinkeys=0{,0},0),\:,0#,!^F,o,O,e,0=break
