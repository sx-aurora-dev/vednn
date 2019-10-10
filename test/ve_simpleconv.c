/** \file
 * program to read conv specs and compute timings for all available impls
 */
#include "conv_test_param.h"
#include "vednnx.h"

#include <assert.h>
#include <unistd.h>

struct param simpleParam[2] =
{    {1,1, 1,4,4, 1,2,2, 3,3, 1,1, 0,0, 1,1, "conv1"} // 4x4 pad 0 --> 2x2
    ,{1,1, 1,4,4, 1,4,4, 3,3, 1,1, 1,1, 1,1, "conv2"} // 4x4 pad 1 --> 4x4
    //        mb,g,icihiw,ocohow,khkw,shsw,phpw,dhdw
};
int const nSimpleParam = (sizeof(simpleParam)/sizeof(struct param));

enum Iinit { Irand=0, Iseq, Ixy }; // image init
enum Kinit { Krand=0, Kall0, Kall1, Ktl, Ktr, Kctr, Kbl, Kbr }; // kernel init

enum Iinit itp = Iseq;
enum Kinit ktp = Kall1;

char const* ktp2str( enum Kinit const ktp ){
    char const* ret = "Khuh?";
    switch(ktp){
    case(Krand): ret = "Krand"; break;
    case(Kall0): ret = "Kall0"; break;
    case(Kall1): ret = "Kall1"; break;
    case(Ktl):   ret = "Ktl"; break;
    case(Ktr):   ret = "Ktr"; break;
    case(Kctr):  ret = "Kctr"; break;
    case(Kbl):   ret = "Kbl"; break;
    case(Kbr):   ret = "Kbr"; break;
    default: ;
    }
    return ret;
}
char const* itp2str( enum Iinit const itp ){
    char const* ret = "Ihuh?";
    switch(itp){
    case(Iseq): ret = "Iseq"; break;
    case(Ixy):  ret = "Ixy"; break;
    default: ;
    }
    return ret;
}
void generateIdata(enum Iinit ii, vednnTensorParam_t const* const tpIn, void *pIn){
    assert(tpIn); assert(pIn);
    int const mb = tpIn->batch;
    int const ic = tpIn->channel;
    int const ih = tpIn->height;
    int const iw = tpIn->width;
    if(ii==Irand){
        generateRandomData(getTensorDataType(tpIn), mb*ic*ih*iw, pIn);
    }else{
        typedef float data_t;
        assert( getTensorDataType(tpIn) == DTYPE_FLOAT );
        float *in = (float*)pIn;
        // generate one image plane
        for(int y=0; y<ih; ++y){
            for(int x=0; x<iw; ++x){
                int const inOff = y * iw + x;
                data_t* const pxy = &in[inOff];
                switch(ii){
                case(Iseq): *pxy = (data_t)(inOff); break;
                case(Ixy):  *pxy = (data_t)(10*y+x); break;
                default: printf("bad Iinit type: %d",(int)ii); fflush(stdout); exit(1);
                }
            }
        }
        // copy that image plane to all other minibatch/channels
        for(int b=0; b<mb; ++b){
            for(int c=0; c<ic; ++c){
                if(b>0 || c>0){
                    for(int y=0; y<ih; ++y){
                        for(int x=0; x<iw; ++x){
                            in[((mb*ic+c)*ih+y)*iw+x] = in[y*iw+x];
                        }
                    }
                }
            }
        }
    }
}
void generateKdata(enum Kinit const ki, vednnFilterParam_t const * const tpKrn, void * const pKrn){
    assert( tpKrn );
    assert( pKrn );
    int const ic = tpKrn->inChannel;
    int const oc = tpKrn->outChannel;
    int const kh = tpKrn->height;
    int const kw = tpKrn->width;
    if(ki==Krand){
        generateRandomData(getKernelDataType(tpKrn), ic*oc*kh*kw, pKrn);
    }else{
        // generate a single input--output channel 0--0 kernel
        typedef float data_t;
        assert( getKernelDataType(tpKrn) == DTYPE_FLOAT );
        float *krn = (float*)pKrn;
        data_t const zero = (data_t)0;
        data_t const one = (data_t)1;
        data_t const init = (ki==Kall1? one: zero);
        for(int h=0; h<kh; ++h){
            for(int w=0; w<kw; ++w){
                krn[h*kw+w] = init;
            }
        }
        printf(" Kinit ki = %d\n",(int)ki);
        switch(ki){
        case(Kall0): break;
        case(Kall1): break;
        case(Ktl):   krn[0*kw+0     ] = one; break;
        case(Ktr):   krn[0*kw+kw-1  ] = one; break;
        case(Kctr):  krn[kw*kh/2    ] = one; break;
        case(Kbl):   krn[(kh-1)*kw+0] = one; break;
        case(Kbr):   krn[kh*kw-1    ] = one; break;
        default: printf("bad Kinit type: %d",(int)ki); fflush(stdout); exit(1);
        }
#if 0
        // copy the single-channel kernel data to the rest of ic x oc
        for(int i=0; i<ic; ++i){
            for(int o=0; o<oc; ++o){
                if(i>0 || o>0){
                    for(int h=0; h<kh; ++h){
                        for(int w=0; w<kw; ++w){
                            krn[((i*oc+o)*kh+h)*kw+w] = krn[h*kw+w]; 
                        }
                    }
                }
            }
        }
#endif
    }
}

void testForward(struct param *pNetwork, int nEntry, double HZ, int flagBias, int flagCSV, int reps)
{
    static int const doRef = 1; // do ref gemm calc
    static int const doStd = 1; // do libvednn default calc
    static int const doItr = 1; // do libvednnx "all" calc
    int i;
    typedef struct testconvForward conv;

    conv *pConvBuff = NULL;

    pConvBuff = (conv *) malloc(nEntry * sizeof(conv));
    if (pConvBuff == NULL) ERROR_EXIT("Memory exhausted.");

    for (i=0; i<nEntry; i++) {
        conv *pConv = &pConvBuff[i];
        struct param *pNw = &pNetwork[i];
        testconvForward_init( pConv );
        strncpy(pConv->region, pNw->pName, 128);
        snprintf(pConv->ref_region, 128, "<gemm:Fwd%s>%s", (flagBias?"B":""), pConv->region);
    }

    for (i=0; i<nEntry; i++) {
        conv *pConv = &pConvBuff[i];
        struct param *pNw = &pNetwork[i];

        testconvForward_alloc( pConv, pNw, flagBias );

        // Generate Data
        if ( pNetwork == &simpleParam[0] ) {
            if(1){ // show the parms
                testconvForward_dumpParms( pConv, flagBias );
            }
            generateIdata( itp, pConv->pParamIn, pConv->pDataIn );

            if(1){
                //int const mb = pConv->pParamIn->batch;
                //int const ic = pConv->pParamIn->channel;
                int const ih = pConv->pParamIn->height;
                int const iw = pConv->pParamIn->width;
                assert( getTensorDataType(pConv->pParamIn) == DTYPE_FLOAT );
                float *in = (float*)pConv->pDataIn;
                printf(" Simple Image data:");
                for(int y=0; y<ih; ++y){
                    printf("\n\t");
                    for(int x=0; x<iw; ++x){
                        printf(" %8.1f",(double)in[y*iw+x]);
                    }
                }
                printf("\n");
            }

            if( flagBias ) {
                assert( getBiasDataType(pConv->pParamBias) == DTYPE_FLOAT );
                for(size_t b=0; b<getBiasSize(pConv->pParamBias); ++b){
                    ((float*)pConv->pDataBias)[b] = (float)1.f;
                }
                printf("Bias is all-ones\n");
            }

            generateKdata( ktp, pConv->pParamKernel, pConv->pDataKernel);

            if(1){
                printf(" Simple Kernel data for ic=%ld oc=%ld repeats:",(long)pConv->pParamKernel->inChannel, (long)pConv->pParamKernel->outChannel);
                //int const ic = pConv->pParamKernel->inChannel;
                //int const oc = pConv->pParamKernel->outChannel;
                int const kh = pConv->pParamKernel->height;
                int const kw = pConv->pParamKernel->width;
                assert( getTensorDataType(pConv->pParamIn) == DTYPE_FLOAT );
                float *krn = (float*)pConv->pDataKernel;
                for(int y=0; y<kh; ++y){
                    printf("\n\t");
                    for(int x=0; x<kw; ++x){
                        printf(" %8.2g",(double)krn[y*kw+x]);
                    }
                }
                printf("\n");
            }
        }else{
            testconvForward_randomData( pConv, flagBias );
        }
    }

    // run Reference Convolution[s]
    if(doRef){
        testconvForward_refcalcs(pConvBuff,nEntry); // fill pBufRef fields
        if ( pNetwork == &simpleParam[0] ) {
            // print simple conv outputs entirely
            for (i=0; i<nEntry; i++) {
                conv *pConv = &pConvBuff[i];
                printf("ref output\n");
                float const* const out = (float*)pConv->pBufRef;
                int const mb = pConv->pParamOut->batch;
                int const oc = pConv->pParamOut->channel;
                int const oh = pConv->pParamOut->height;
                int const ow = pConv->pParamOut->width;
                for(int b=0; b<mb; ++b){
                    for(int c=0; c<oc; ++c){
                        for(int y=0; y<oh; ++y){
                            if(y==0) printf("\n\tb%dc%d\t",b,c);
                            else     printf("\n\t\t");
                            for(int x=0; x<ow; ++x){
                                printf(" %10.2f", out[((b*oc+c)*oh+y)*ow+x]);
                            }
                        }
                    }
                }
                printf("\n"); fflush(stdout);
            }
        }//end print simple conv outputs entirely
    }

    // run test Convolution, libvednn API
    if(doStd) for(int r=0; r < reps; ++r){
        testconvForward_vednncalcs( pConvBuff, nEntry );

        if (flagCSV) dumpParamCSV_title();

        for (i=0; i<nEntry; i++) {
            conv *pConv = &pConvBuff[i];
            struct param *pNw = &pNetwork[i];
            if (flagCSV) dumpParamCSV(pNw,"Fwd",(flagBias?"B":""));
            else         dumpParam   (pNw,"Fwd", (flagBias? "B":""));

            double diff = !doRef? -13.0
                : diffData(pConv->pParamOut, pConv->pDataOut, pConv->pBufRef);

            double f = 1.0e3 / HZ;
            double time = pConv->cycle * f / pConv->reps; // average ms
            double mintime = pConv->mincycle *  f;
            double maxtime = pConv->maxcycle *  f;
            printf((flagCSV?", %f, %f, %f, %f"
                        :" \tTIME = %8.3f msec [%8.3f,%8.3f] DIFF = %f"),
                time, mintime, maxtime, diff);
            printf("\n");
        }
        fflush(stdout);
    }

    // run convolutions, all available impls, libvednnx iterator api
    if(doItr) for(int r=0; r<reps; ++r){
        vednnError_t rv;

        for (i=0; i<nEntry; i++) {
            conv *pConv = &pConvBuff[i];

            //unsigned long long c[2];
            //c[0] = __cycle();

            char name[80];

            // Convolution
            if ( flagBias ) {
                // use parameters and args, split according to order of libvednn public api.
                // I.e. \c vednnConvolutionForwardAddBias call as defined in \ref vednn.h
#define FWDB_PARMS \
                pConv->pParamIn, pConv->pParamKernel, pConv->pParamBias, pConv->pParamOut, \
                pConv->pParamConv, VEDNN_CONV_ALGORITHM_DIRECT
#define FWDB_DATA \
                pConv->pDataIn,  pConv->pDataKernel,  pConv->pDataBias,  pConv->pDataOut
                // libvednnx "iterator over impls"
                int iter_cnt=0;
                for (vednnConvForwardAddBiasImpls * iter = vednnConvForwardAddBias_Begin(FWDB_PARMS);
                        iter->okfn != NULL;
                        iter = vednnConvForwardAddBias_Next(iter, FWDB_PARMS))
                {
                    snprintf(name,80,"%s:%d:%s",pConv->region,iter_cnt,iter->shortname);
                    //printf(" %s...",name); fflush(stdout);

                    FTRACE_BEGIN(name);
                    // NB:             CONVX_.....order for low-level call
                    rv = (*iter->impl)(CONVX_FWDB_ORDER(FWDB_PARMS, FWDB_DATA));
                    FTRACE_END(name);

                    if (r == 0){
                        if( pConv->pParamIn->batch == 1 ) {
                            printf(" batch 1 group=%d inChannel=%d outChannel=%d",
                                    (int)(pConv->pParamConv->group),
                                    (int)(pConv->pParamIn->channel),
                                    (int)(pConv->pParamOut->channel));
                        }
                        //pConv->cycle += c[1] - c[0]; // this is reserved for the libvednn time
                        //printf(" %s %llu cycles", name, c[1]-c[0]);
                        double diff = diffData(pConv->pParamOut, pConv->pDataOut, pConv->pBufRef);
                        printf(" %s : %s",(rv==VEDNN_SUCCESS?" OK":"BAD"),name); fflush(stdout);
                        printf(" DIFF = %f", diff);
                        printf("\n");
                        fflush(stdout);
                    }

                    if(++iter_cnt>=100) ERROR_EXIT("run-away iter over ConvForwardBias impls?");
                }
#undef FWDB_DATA
#undef FWDB_PARMS
            }else{
                // original args for vednn.h call:
                //pConv->pParamIn,         pConv->pDataIn,
                //pConv->pParamKernel,     pConv->pDataKernel,
                //pConv->pParamOut,        pConv->pDataOut,
                //pConv->pParamConv,
                //VEDNN_CONV_ALGORITHM_DIRECT
#define FWD_PARMS \
                /* */ pConv->pParamIn,     \
                /* */ pConv->pParamKernel, \
                /* */ pConv->pParamOut,    \
                /* */ pConv->pParamConv,   \
                /* */ VEDNN_CONV_ALGORITHM_DIRECT
#define FWD_DATA \
                /* */                      pConv->pDataIn, \
                /* */                      pConv->pDataKernel, \
                /* */                      pConv->pDataOut
                // libvednnx "iterator over impls"
                // 1 i            printf(" t%ld", (long)omp_get_num_threads()); fflush(stdout);
                // What you want: printf(" t%ld", (long)omp_get_max_threads()); fflush(stdout);
                int iter_cnt=0;
                for (vednnConvForwardImpls * iter = vednnConvForward_Begin(FWD_PARMS);
                        iter->okfn != NULL;
                        iter = vednnConvForward_Next(iter,FWD_PARMS)
                    )
                {
                    //printf("doIter "); vednnConvForward_Dump(iter); printf("\n"); fflush(stdout);
                    // so grep -k11 of ftrace will sort from "best" to "worst" [approx]
                    snprintf(name,80,"%s:%d:%s",pConv->region,iter_cnt,iter->shortname);
                    //printf(" %s... ",name); fflush(stdout);

#if 0 // original way:
                    // BADNESS:  no thread support, no rtok check at runtime for data tr alignment
                    FTRACE_BEGIN(name);
                    // libvednn calls directly into libvednn low-level routines, so use
                    // the CONVX_.. macro to get low-level arg order correct.
                    rv = (*iter->impl)(CONVX_FWD_ORDER(FWD_PARMS, FWD_DATA));
                    FTRACE_END(name);
#else
                    FTRACE_BEGIN("realNext");
                    vednnConvForwardImpls *actual = vednnConvForward_realNext(
                            iter, FWD_PARMS, FWD_DATA );
                    FTRACE_END("realNext");

                    if ( actual != NULL ) {
                        if( actual == iter ){ // almost always...
                            snprintf(name,80,"%s:%d:%s",pConv->region,iter_cnt,iter->shortname);
                        }else{
                            snprintf(name,80,"%s:%d:%s-->%s",pConv->region,iter_cnt,iter->shortname,actual->shortname);
                        }
                        FTRACE_BEGIN(name);
                        // now we also want omp support, so we call via _Run, not via (*actual->impl)
                        vednnConvForward_out_t const out = vednnConvForward_Run( actual, FWD_PARMS, FWD_DATA );
                        // ignore out.status  (hopefully VEDNN_SUCCESS)
                        FTRACE_END(name);
                        assert( out.actual == actual );
                        rv = out.status;
                    }
#endif
                    //printf(" OK\n",name); fflush(stdout);
                    if (r == 0){
                        if( pConv->pParamIn->batch == 1 ) {
                            printf(" batch 1 group=%d inChannel=%d outChannel=%d",
                                    (int)(pConv->pParamConv->group),
                                    (int)(pConv->pParamIn->channel),
                                    (int)(pConv->pParamOut->channel));
                        }
                        //pConv->cycle += c[1] - c[0]; // this is reserved for the libvednn time
                        //printf(" %s %llu cycles", name, c[1]-c[0]);
                        double diff = diffData(pConv->pParamOut, pConv->pDataOut, pConv->pBufRef);
                        printf(" %s : %s",(rv==VEDNN_SUCCESS?" OK":"BAD"),name); fflush(stdout);
                        printf(" DIFF = %f", diff);
                        printf("\n");
                        fflush(stdout);
                    }

                    if(++iter_cnt>=100) ERROR_EXIT("run-away iter over ConvForward impls?");
                    //printf("\n iter -> _Next ..."); fflush(stdout);
                    //iter = vednnConvForward_Next(iter,FWD_PARMS);
                    //printf("OK : iter@%p\n",(void*)iter); fflush(stdout);
                    //vednnConvForward_Dump(iter);
                    //printf("OK\n"); fflush(stdout);
                }
#undef FWD_DATA
#undef FWD_PARMS
            }
            if (rv != VEDNN_SUCCESS) ERROR_EXIT("convolution() failed.");

            //c[1] = __cycle();
        }

        FTRACE_END("all convolution");
    }

    // release
    fflush(stdout);
    for (i=0; i<nEntry; i++) {
        testconvForward_free(&pConvBuff[i],flagBias);
    }
    free(pConvBuff);
}

void testBackwardData(struct param *pNetwork, int nEntry, double HZ, int flagCSV, int reps)
{
    int i;
    typedef struct testconvBackwardData conv;

    conv *pConvBuff = (conv *) XMALLOC(nEntry * sizeof(conv));

    for (i=0; i<nEntry; i++) {
        conv *pConv = &pConvBuff[i];
        struct param *pNw = &pNetwork[i];
        testconvBackwardData_init(pConv);
        strncpy(pConv->region, pNw->pName, 128);
        snprintf(pConv->ref_region, 128, "<gemm:BkwD> %s",pConv->region);
    }

    for (i=0; i<nEntry; i++) {
        conv *pConv = &pConvBuff[i];
        struct param *pNw  = &pNetwork[i];
        testconvBackwardData_alloc(pConv, pNw);
        testconvBackwardData_randomData( pConv );
    }

    if(1) for(i=0; i<nEntry; ++i) {
        testconvBackwardData_dumpParms(&pConvBuff[i]);
    }

    // run Reference Convolution
    if(1) {
        testconvBackwardData_refcalcs( pConvBuff, nEntry );
    }


    // run test Convolution, libvednn API
    for(int r=0; r < reps; ++r){
        testconvBackwardData_vednncalcs( pConvBuff, nEntry );
    }

    if (flagCSV) dumpParamCSV_title();
    for (i=0; i<nEntry; i++) {
        conv *pConv = &pConvBuff[i];
        struct param *pNw = &pNetwork[i];
        if (flagCSV) dumpParamCSV(pNw,"Bkw","D");
        else         dumpParam   (pNw,"Bkw","D");

        double diff = diffData(pConv->pParamGradIn, pConv->pDataGradIn, pConv->pBufRef);

        double f = 1.0e3 / HZ;
        double time = pConv->cycle * f / pConv->reps; // average ms
        double mintime = pConv->mincycle *  f;
        double maxtime = pConv->maxcycle *  f;
        printf((flagCSV?", %f, %f, %f, %f"
                    :" \tTIME = %8.3f msec [%8.3f,%8.3f] DIFF = %f"),
                time, mintime, maxtime, diff);
        printf("\n");
    }

    // release
    for (i=0; i<nEntry; i++) {
        conv *pConv = &pConvBuff[i];
        testconvBackwardData_free(pConv);
    }
    free(pConvBuff);
}

void testBackwardFilter(struct param *pNetwork, int nEntry, double HZ, int flagCSV, int reps)
{
    int i;
    typedef struct testconvBackwardFilter conv;
    conv *pConvBuff = (conv *) XMALLOC(nEntry * sizeof(conv));

    for (i=0; i<nEntry; i++) {
        conv *pConv = &pConvBuff[i];
        struct param *pNw = &pNetwork[i];
        testconvBackwardFilter_init(pConv);
        strncpy(pConv->region, pNw->pName, 128);
        snprintf(pConv->ref_region, 128, "<gemm:BkwF> %s",pConv->region);
    }

    for (i=0; i<nEntry; i++) {
        conv *pConv = &pConvBuff[i];
        struct param *pNw  = &pNetwork[i];
        testconvBackwardFilter_alloc( pConv, pNw );
        testconvBackwardFilter_randomData( pConv );
    }

    { // run Reference Convolution[s]
        testconvBackwardFilter_refcalcs(pConvBuff, nEntry);
    }

    // run test Convolution, libvednn API
    for(int r=0; r < reps; ++r){
        testconvBackwardFilter_vednncalcs( pConvBuff, nEntry );
    }

    if (flagCSV) dumpParamCSV_title();
    for (i=0; i<nEntry; i++) {
        conv *pConv = &pConvBuff[i];
        struct param *pNw = &pNetwork[i];
        if (flagCSV) dumpParamCSV(pNw,"Bkw","F");
        else         dumpParam   (pNw,"Bkw","F");
        double f = 1.0e3 / HZ;
        double time = pConv->cycle * f / pConv->reps; // average ms
        double mintime = pConv->mincycle *  f;
        double maxtime = pConv->maxcycle *  f;
        double diff = diffData((vednnTensorParam_t*)(pConv->pParamGradKernel), pConv->pDataGradKernel, pConv->pBufRef);
        printf((flagCSV?", %f, %f, %f, %f"
                    :" \tTIME = %8.3f msec [%8.3f,%8.3f] DIFF = %f"),
                time, mintime, maxtime, diff);
        printf("\n");
    }

    // release
    for (i=0; i<nEntry; i++) {
        conv *pConv = &pConvBuff[i];
        testconvBackwardFilter_free(pConv);
    }
    free(pConvBuff);
}


enum {
    CONV_TEST_FORWARD = 0,
    CONV_TEST_FORWARD_ADDBIAS,
    CONV_TEST_BACKWARD_DATA,
    CONV_TEST_BACKWARD_FILTER,
    CONV_TEST_BACKWARD_BIAS
} ;


static struct {
    char *pName;
    int   testtype;
} tests[] = {
    { "Forward",    CONV_TEST_FORWARD } ,
    { "ForwardAddBias",    CONV_TEST_FORWARD_ADDBIAS } ,
    { "BackwardData",    CONV_TEST_BACKWARD_DATA } ,
    { "BackwardFilter", CONV_TEST_BACKWARD_FILTER } ,
    //    { "BackwardBias", CONV_TEST_BACKWARD_BIAS } ,   // not implemented
};

static void help(){
    printf( "\nve_cmpconv:"
            "\n   -p PATH   convolution parameter file"
            "\n optional:"
            "\n   -C        CSV output"
            "\n   -T STRING test type [Forward] ForwardAddBias"
            "\n                       BackwardData BackwardFilter"
            "\n   -r INT    reps [1]"
            "\n   -H FLOAT  Timer register speed (Hz) [0.8e9]"
            "\n   -t N      omp_set_num_threads(N), then run"
            "\n             N < 0 means repeat for N=0..8 (don't use ftrace output)"
            "\n");
}
int main(int argc, char **argv)
{
    extern int optind;
    extern char *optarg;
    int opt;

    char *pParamPath = NULL ;
    double HZ        = 0.8e9 ;
    int testtype     = 0 ;
    int flagCSV      = 0 ;
    int reps         = 1 ;
    int threads      = 1 ; /* set to -ve to repeat for 0..8 threads */
    printf("Test program: %s\n",__FILE__);
    //enum Iinit itp   = Iseq ; // input type: Iseq=0={0,1,2,...}, Ixy={{00,01,..},{10,11,...},...{...<isz>,<isz>}}
    //enum Kinit ktp   = Ktl ; // kernel type: Ktl=1 in top left, zeros elsewhere

    // i and k add very simple demo kernel output to forward test
    printf(" Example Arguments:\n"
            "  -i1 -k2   default: sequential input data, all-ones kernel\n"
            "  -i1 -k3   top-left 3x3 kernel\n"
            "  -i2 -k5   image[x,y]=10*x+y; identity 3x3 kernel\n"
            " Not suggested:\n"
            "  -i0 -k0   random data, parameter file required (use ve_cmpconv instead)\n"
            "\n"
            " Output uses simple kernels as filter across image.\n"
            " You should be able to visually check for correctness of ref (gemm) output.\n"
            " Then we run all available impls (fwd) and report DIFF from ref calc.\n"
          );
    while ((opt = getopt(argc, argv, "p:CH:T:t:r:i:k:")) != -1) {
        switch (opt) {
        case 'p':
            pParamPath = optarg;
            break;
        case 'r':    reps = (int)atof(optarg); break;
        case 'i':    itp = (enum Iinit)(int)atof(optarg); break;
        case 'k':    ktp = (enum Kinit)(int)atof(optarg); break;
        case 'C':    flagCSV = 1;        break;
        case 'H':    HZ = atof(optarg);    break;
        case 'T':
                     {
                         int found = 0;
                         for (int i=0; i<sizeof(tests)/sizeof(tests[0]); i++) {
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
        case 't':    threads = atof(optarg);
                     if(threads > 16) threads = 16;
                     if(threads < 0 ) threads = -1;
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
#if 0
    if( itp == Irand && ktp == Krand ){
        if ( pParamPath == NULL ) {
            fprintf(stderr, "Parameter file must be specified by '-p' option.\n");
            exit(1);
        }
    }else{
        pParamPath = NULL; // simple tests ignore param file.
    }
#endif
    if (HZ <= 0.0) {
        HZ = 8e8;
        fprintf(stderr, "Processor core frequency, '-H' options, defaulting to 8e8.\n");
    }
    if (reps < 1) {
        fprintf(stderr, "reps must be >= 1\n");
        exit(1);
    }
    printf("CONVOLUTION TEST TYPE    = %s\n",      tests[testtype].pName) ;
    printf("PROCESSOR CORE FREQUENCY = %.3e HZ\n", HZ);
    printf("PARAMETER FILE           = %s\n",      pParamPath);
    printf(" Image data              = %s\n",      itp2str(itp));
    printf(" Kernel data             = %s\n",      ktp2str(ktp));


    struct param *pParams ;
    int nParams;
    if (pParamPath) {
        nParams = readParamFile( &pParams, pParamPath ) ;
        printf(" readParamstring --> %d\n",nParams);
        if( nParams <= 0 ) {
            printf("Bad params from %s\nUsing a built-in simple default\n", pParamPath);
            pParams = &simpleParam[0];
            nParams = nSimpleParam;
        }
    }else{
        pParams = &simpleParam[0];
        nParams = nSimpleParam;
    }

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
            mkConsistent( &pParams[i] );
        }

#ifdef USE_OPENMP
    int thrlo = 0, thrhi = 8;
    if(threads >= 0){
        thrlo = threads;
        thrhi = threads;
    }
    for(int thr=thrlo; thr<=thrhi; ++thr){
        printf(" omp_set_num_threads(%d)...\n", thr);
        omp_set_num_threads(thr);
        fflush(stdout);
#endif
        switch(testtype) {
        case CONV_TEST_FORWARD :
            testForward(pParams, nParams, HZ, 0, flagCSV, reps);
            break ;
        case CONV_TEST_FORWARD_ADDBIAS :
            testForward(pParams, nParams, HZ, 1, flagCSV, reps);
            break ;
        case CONV_TEST_BACKWARD_DATA :
            testBackwardData(pParams, nParams, HZ, flagCSV, reps);
            break ;
        case CONV_TEST_BACKWARD_FILTER :
            testBackwardFilter(pParams, nParams, HZ, flagCSV, reps);
            break ;
        default :
            break ;
        }
#ifdef USE_OPENMP
    }
#endif

    //if(with_ftrace()) system("ftrace"); // oh. it must be closed during program exit
    printf("\nGoodbye%s\n",with_ftrace()?" -- [try ftrace]":""); fflush(stdout);
    return 0;
}

// vim: et ts=4 sw=4 cindent cino=^l0,\:0,N-s syntax=cpp.doxygen
