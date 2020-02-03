/** \file
 * program to read conv specs and compute timings for all available impls
 *
 * - uses vednnx \e iterator-API
 * - this does NOT iterate over any JIT impls
 * - allows flexible reading of parameter string / file
 */
#include "conv_test_param.h"
// OK in 'c" : static struct testconvForward tcfFoo;
#include "vednnx.h"
#include "cjitConv.h"
#include "math.h"

#include <assert.h>
#include <unistd.h>
#include <dlfcn.h>
#include <sys/resource.h> // getrlimit, setrlimit

#define MAX_TEST_DATA 65536
#define MAX_TEST_NAME 100

#define TESTDATA 1 /*default 1*/

#if TESTDATA
/** New stats, by {test,impl} */
struct TestData {
    unsigned long long sum_times;
    size_t test;
    size_t reps;
    double diff;
    uint64_t ops;
    int impl_idx;
    int impl_type; // 0/1/2 for libvednn-std/lbivednn-impl/JIT-impl
    char test_name[MAX_TEST_NAME];
    char impl_name[MAX_TEST_NAME]; ///< shorter
    char descr[MAX_TEST_NAME];     ///< longer (only for impl_type 2=JIT)
    // TODO: size_t dup_of; ///< if removed due to duplication, keep that info
};
struct IdxAZ{
    size_t idx, first;
};
struct ImplNameIdx {
    size_t a;
    size_t z;
    size_t nImpls; // count of unique impls
    /** array of z-a indices in 0,1,...nImpls, and 1st occurence in [a,z) */
    struct IdxAZ* idx;
};
static char testData_impl_char(int const impl_type){
    char ret;
    switch(impl_type){
    case(0): ret = '*'; break; // doStd
    case(1): ret = 'I'; break; // doItr
    case(2): ret = 'J'; break; // doJit
    default: ret = '?';
    }
    return ret;
}
static struct ImplNameIdx mk_impl_name_idx(struct TestData* test_data, size_t const a, size_t const z){
    int const v=0; // verbose
    struct ImplNameIdx ret;
    ret.a = a;
    ret.z = z;
    ret.idx = (struct IdxAZ*)malloc((z-a)*sizeof(struct IdxAZ));
    if(ret.idx == NULL) ERROR_EXIT("Memory exhausted.");
    ret.nImpls = 0U;
    if(v>1) printf("a %d z %d\n",(int)a,(int)z);
    for(size_t ntd=a; ntd<z; ++ntd){
        if(v>1) printf("%40s ntd=%d %s",test_data[ntd].impl_name, (int)ntd, test_data[ntd].descr);
        size_t dup = (size_t)(a-1U);
        for(size_t prv=a; prv<ntd; ++prv){
            if( strcmp(test_data[ntd].impl_name, test_data[prv].impl_name) == 0){
                dup = prv;
                if(v>1) printf(" dup=%d",(int)dup);
                break;
            }
        }
        if(dup==(size_t)(a-1U)){
            ret.idx[ntd-a].idx = ret.nImpls; // give it a new index
            ret.idx[ntd-a].first = ntd;
            ++ret.nImpls;
        }else{
            ret.idx[ntd-a].idx = ret.idx[dup-a].idx; // re-assign it the previous index
            ret.idx[ntd-a].first = dup;
            assert( dup == ret.idx[dup-a].first );
        }
        if(v>1) printf(" = {%d,%d}\n",(int)ret.idx[ntd-a].idx,(int)ret.idx[ntd-a].first);
    }
    for(size_t i=0; i<z-a; ++i){
        if(v>0){ 
            printf(" idx[%d] = {%d, %d} %s\n",(int)i
                    ,(int)(ret.idx[i].idx),(int)(ret.idx[i].first)
                    ,test_data[ret.idx[i].first].impl_name);
            fflush(stdout);
        }
        assert( ret.idx[i].idx < ret.nImpls );
        assert( ret.idx[i].first >= a && ret.idx[i].first < z );
    }
    return ret;
}
/** A non-rigorous points scheme to compare algorithms -- <B>not strictly correct</B>.
 * - strange test domain overlap possibilities /e not considered.
 * - see \ref dom.hpp for more general analysis work
 */
static void print_wins(struct TestData* test_data, size_t const a, size_t const z, double const HZ){
    int const v=0; // verbose
    struct ImplNameIdx idx = mk_impl_name_idx(test_data,a,z);
    size_t const ni = idx.nImpls;
    size_t* battles = (size_t*)malloc(ni*ni*sizeof(size_t));
    size_t* wins    = (size_t*)malloc(ni*ni*sizeof(size_t));
    double* speedup = (double*)malloc(ni*ni*sizeof(double));
    double* sum_t   = (double*)malloc(ni*sizeof(double));
    size_t* n_t     = (size_t*)malloc(ni*sizeof(size_t));
    if(!battles || !wins || !speedup || !sum_t || !n_t) ERROR_EXIT("Memory exhausted.");
    for(size_t ij=0U; ij<ni*ni; ++ij){ speedup[ij] = battles[ij] = wins[ij] = 0U; }
    for(size_t i =0U; i <ni   ; ++i ){ sum_t[i] = 0.0; n_t[i]=0U; }
    double const clearly_faster = 0.98; // threshold

#if 0
    // following is wrong because 'win' points should only be assigned
    // if 'A' and 'B' battled.
    // i.e. only if A and B battle for at least 1 convolution.
    // So if A and B never battle, we cannot say that either is 'better'.
    //
    // So first loop over unique test numbers, and only
    // count wins of the battling impls.
    // Besides 'wins' count how many 'battles' i vs j occured.
    // i only dominates j if it can run the same test as j
    for(size_t i=a; i<z; ++i){
        struct TestData const* tdi = &test_data[i];
        if( tdi->reps <= 0U ) continue;
        double const time_i = tdi->sum_times / tdi->reps; // average ms
        for(size_t j=a; j<i; ++j){
            struct TestData const* tdj = &test_data[j];
            if( tdj->reps <= 0U ) continue;
            double const time_j = tdi->sum_times / tdi->reps; // average ms

            if( time_i < clearly_faster * time_j ){
                wins[i-a][j-a] += 2U;
            }else if( time_j < clearly_faster * time_i ){
                wins[j-a][i-a] += 2U;
            }else{
                ++wins[i-a][j-a];
                ++wins[j-a][i-a];
            }
        }
    }
#endif
    int maxTest = 0;   //
    for(int i=a; i<z; ++i){
        if(test_data[i].test > maxTest) maxTest = test_data[i].test;
    }
    for(int t=0; t<=maxTest; ++t){
        for(size_t i=a; i<z; ++i){
            struct TestData const* tdi = &test_data[i];
            if( tdi->test != t || tdi->reps <= 0U ) continue;
            double const time_i = tdi->sum_times / tdi->reps; // average ms
            size_t const impl_i = idx.idx[i-a].idx; // impl# in [0,ni)
            assert(impl_i<ni);
            sum_t[impl_i] += time_i; ++n_t[impl_i];
            if(v) printf("\ntest %d impl %d:%d(%.6f) vs ", (int)t, (int)i,(int)impl_i, time_i);
            for(size_t j=a; j<i; ++j){
                struct TestData const* tdj = &test_data[j];
                if( tdj->test != t || tdj->reps <= 0U ) continue;
                double const time_j = tdj->sum_times / tdj->reps; // average ms
                size_t const impl_j = idx.idx[j-a].idx; // impl# in [0,ni)
                if(v)printf(" %d:%d",(int)j,(int)impl_j);
                assert(impl_j<ni);
                // i and j battled during test t !
                ++battles[impl_i*ni+impl_j];
                ++battles[impl_j*ni+impl_i]; // symmetric
                // Who won/tied?
                if( time_i < clearly_faster * time_j ){
                    if(v)printf("<");
                    wins[impl_i*ni+impl_j] += 2U;
                }else if( time_j < clearly_faster * time_i ){
                    if(v)printf(">");
                    wins[impl_j*ni+impl_i] += 2U;
                }else{
                    if(v)printf("~");
                    ++wins[impl_i*ni+impl_j];
                    ++wins[impl_j*ni+impl_i];
                }
                speedup[impl_i*ni+impl_j] += log10(time_j/time_i);
                speedup[impl_j*ni+impl_i] += log10(time_i/time_j);
            }
        }
        if(v)printf("\n");
    }
    // now wins[ii][jj] counts when impl (a+ii) clearly wins agains (a+jj)
    // print legend (impl --> name)
    printf("\n Legend : impl     -->   avg_t (ms) name\n");
    char const** impl_names = (char const**)malloc(ni*sizeof(char*));
    {
        for(size_t imp=0; imp<ni; ++imp){
            size_t test=~0U;
            for(size_t i=a; i<z; ++i){
                if( idx.idx[i-a].idx == imp ){
                    //printf("    { %d, %d }",(int)(idx.idx[i-a].idx),(int)(idx.idx[i-a].first));
                    test = idx.idx[i-a].first; // in [a,z), we can get back the name
                    break;
                }
            }
            assert( test != ~0U );
            impl_names[imp] = test_data[test].impl_name;

            double avg_t = (n_t[imp]? sum_t[imp] / n_t[imp]: 0.0);
            double const f = 1.0e3 / HZ;
            double ms = avg_t * f;
            printf("          imp %4u --> %12.3f %s\n",
                    (unsigned)imp, ms, impl_names[imp]);
        }
    }
    // a vs b battle count table:
    printf("\nBattles ");
    {
        unsigned maxBat = 0;
        for(size_t ab=0; ab<ni*ni; ++ab){
            if((unsigned)battles[ab] > maxBat) maxBat = (unsigned)battles[ab];
        }
        char const* fmt=(maxBat<=99U?" %2u": maxBat<=999U?" %3u":" %4u");
        for(size_t imp=0; imp<ni; ++imp){
            printf(fmt, (unsigned)imp);
        }
        for(size_t a=0; a<ni; ++a){
            printf("\n%6u |",(unsigned)a);
            for(size_t b=0; b<ni; ++b){
                printf(fmt,(unsigned)battles[a*ni+b]);
            }
            printf(" | %s", impl_names[a]);
        }
    }
    // a vs b dominance score
    printf("\nDominates");
    for(size_t imp=0; imp<ni; ++imp){
        printf(" %3u", (unsigned)imp);
    }
    for(size_t a=0; a<ni; ++a){
        printf("\n%7u |",(unsigned)a);
        for(size_t b=0; b<ni; ++b){
            size_t nbat = battles[a*ni+b];
            if( nbat == 0 ){
                printf("  -1"); //-1 ~ they never competed
            }else{
                double dominance = (double)wins[a*ni+b] / (2.0*nbat);
                printf(" %3d",(int)(dominance*100.0));
            }
        }
        printf(" | %u %s",(unsigned)a,impl_names[a]);
    }
    // a vs b avg speedup factor [in %, so 100 ~ "same speed on avg"]
    printf("\nRelSpeed is (avg log_10(t_i/t_j))*100 so 0 is same speed,\n"
           "           +-30 is 2x speed, and +-100 is 10x speedup\n"
           "RelSpeed%%");
    for(size_t imp=0; imp<ni; ++imp){
        printf(" %3u",(unsigned)imp);
    }
    for(size_t a=0; a<ni; ++a){
        printf("\n%7u |",(unsigned)a);
        for(size_t b=0; b<ni; ++b){
            size_t nbat = battles[a*ni+b];
            if( nbat == 0 ){
                printf(" 000"); //-1 ~ they never competed
            }else{
#if 0
                double relSpeed = log10((double)speedup[a*ni+b] / nbat);
                printf("%4d",(int)(relSpeed*100.0 + 0.5));
#else
                double relSpeed = (double)speedup[a*ni+b] / nbat;
                if( relSpeed < -0.99 ) relSpeed = -0.99;
                if( relSpeed >  0.99 ) relSpeed =  0.99;
                printf("%4d",(int)(relSpeed*100.0 + 0.5));
#endif
            }
        }
        printf(" | %u %s",(unsigned)a,impl_names[a]);
    }
    printf("\n");
    free(idx.idx);
    free(n_t);
    free(sum_t);
    free(speedup);
    free(impl_names);
    free(wins);
    free(battles);
}
/** helper for sort-by-increasing-time */
struct TestTime {
    int ntd;        ///< index into struct TestData[]
    double time;    ///< avg time of some TestData[ntd]
};
static int cmp_TestTime( void const* a, void const*b ){
    return ((struct TestTime const*)a)->time
        >=  ((struct TestTime const*)b)->time;
}
static void init_TestData( struct TestData* test_data,
        int const test, char const* test_name, //char const* descr,
        int const impl_idx, char const* impl_name, int impl_type ){
    char const* name;
    test_data->test = test;
    name = test_name;
    {
        int namelen = strlen(name);
        //char *cutname = strstr(name,"_mb");
        //if(cutname) namelen = cutname-name;
        if(namelen > MAX_TEST_NAME) namelen=MAX_TEST_NAME;
        strncpy(test_data->test_name, name, namelen);
        test_data->test_name[namelen] = '\0';
        //         ^^^^^^^^^
    }
    test_data->impl_idx = (size_t)impl_idx;
    name = impl_name;
    {
        int namelen = strlen(name);
        if(namelen > MAX_TEST_NAME) namelen=MAX_TEST_NAME;
        if(impl_type == 2/*JIT*/){
            strncpy(test_data->descr, name, namelen);
            test_data->descr[namelen] = '\0'; // full length
            //         ^^^^^
        }else{
            test_data->descr[0] = '\0';
        }
        char *cutname = strstr((char*)name,"_mb");
        if(cutname) namelen = cutname-name;
        if(namelen > MAX_TEST_NAME) namelen=MAX_TEST_NAME;
        strncpy(test_data->impl_name, name, namelen);
        test_data->impl_name[namelen] = '\0';
        //         ^^^^^^^^^
    }
    test_data->sum_times = 0ULL;
    test_data->ops = 0ULL;
    test_data->diff = -1.0;
    test_data->impl_type = impl_type;
}
/** print td[a] to td[z] sort by test (todo: then increasing avg time). */
static void print_test_data( struct TestData const* test_data, int const a, int const z, double const HZ){
    assert( a>=0 );
    assert( z>=a );
    if( a >= z ) return;
    int maxTest = 0;   //
    for(int i=a; i<z; ++i){
        if(test_data[i].test > maxTest) maxTest = test_data[i].test;
    }
    printf("*** print_test_data[%d..%d) covers tests up to %d\n",a,z,maxTest);
    struct TestTime ttime[MAX_TEST_DATA];
    for(int t=0; t<=maxTest; ++t){
        int nttime = 0;
        for(size_t ntd=a; ntd<z; ++ntd){
            struct TestData const* td = &test_data[ntd];
            if( td->test != t )
                continue;
            double const f = 1.0e3 / HZ;
            double const time = td->sum_times * f / td->reps; // average ms
            ttime[nttime].ntd = ntd;
            ttime[nttime].time = time;
            ++nttime;
        }
        // sort ttime[0..nttime)
        qsort( &ttime[0], nttime, sizeof(struct TestTime), cmp_TestTime );
        char const* fastest="**";
        for(size_t ntt=0; ntt<nttime; ++ntt){
            struct TestData const* td = &test_data[ttime[ntt].ntd];
            double const f = 1.0e3 / HZ;
            double const time = td->sum_times * f / td->reps; // average ms
            // Gop/s = Mop/ms
            double const gops = (td->ops>0? td->ops*1.0e-6 / time: 0.0);
            printf( "%c %25s %s %4ux %9.3f ms ~%.4f %6.2fG %s %s\n",
                    testData_impl_char(td->impl_type),
                    td->impl_name,
                    fastest, td->reps, time, (double)td->diff,
                    gops, td->test_name, td->descr );
            fastest = " |";
        }
    }

}
#endif

void testForward(struct param *pNetwork, int nEntry, double HZ, int flagBias, int flagCSV, int reps, filterLayout_t filter_layout)
{
    static int const doRef = 1; // do ref gemm calc
    static int const doStd = 1; // do libvednn default calc
    static int const doItr = 1; // do libvednnx "all" calc
    static int const doJit = 0;
    int t;
    typedef struct testconvForward conv;
    printf("Beginning %s\n",__PRETTY_FUNCTION__);

    conv *pConvBuff = NULL;

    pConvBuff = (conv *) malloc(nEntry * sizeof(conv));
    if (pConvBuff == NULL) ERROR_EXIT("Memory exhausted.");

    for (t=0; t<nEntry; ++t) {
        conv *pConv = &pConvBuff[t];
        struct param *pNw = &pNetwork[t];
        testconvForward_init( pConv );
        if(strlen(pNw->pName)) strncpy(pConv->region, pNw->pName, 128);
        else snprintf(pConv->region, 128, "test:Fwd%s", (flagBias?"B":""));
        snprintf(pConv->ref_region, 128, "<gemm:Fwd%s>%s", (flagBias?"B":""), pConv->region);
    }

    for (t=0; t<nEntry; ++t) {
        conv *pConv = &pConvBuff[t];
        struct param *pNw = &pNetwork[t];
        testconvForward_alloc( pConv, pNw, flagBias, filter_layout );
        testconvForward_randomData( pConv, flagBias );
    }

    if(1) for(t=0; t<nEntry; ++t) {
        testconvForward_dumpParms( &pConvBuff[t], flagBias );
    }

    // run Reference Convolution
    if(doRef){
        printf("doRef...\n");
        testconvForward_refcalcs(pConvBuff,nEntry); // fill pBufRef fields
    }

#if TESTDATA
    // Also track (test,imp) avg time. impl_type 0,1,2 ~ doStd,doItr,doJit
    struct TestData test_data[MAX_TEST_DATA];
    size_t n_test_data = 0U; //ntest_: std<vednn<jit==data
    size_t n_test_std=0U, n_test_itr = 0U, n_test_jit = 0U;
#endif

    if(doStd) printf("pConv->region is %s\n", pConvBuff[0].region);
    // run test Convolution, libvednn API
    if(doStd){
        for(int r=0; r < reps; ++r){
            testconvForward_vednncalcs( pConvBuff, nEntry );

#if TESTDATA
            size_t ntd = n_test_data; // statistics entries get appended to test_data[]
#endif
            if (flagCSV) dumpParamCSV_title();
            double sum_time = 0.0;
            double max_diff = 0.0;
            for (t=0; t<nEntry; ++t) {
                conv *pConv = &pConvBuff[t];
                struct param *pNw = &pNetwork[t];
                if (flagCSV) dumpParamCSV(pNw,"Fwd",(flagBias?"B":""));
                else         dumpParam   (pNw,"Fwd", (flagBias? "B":""));

                double diff = !doRef? -13.0
                    : diffData(pConv->pParamOut, pConv->pDataOut, pConv->pBufRef);
                max_diff = diff > max_diff? diff: max_diff;
                double f = 1.0e3 / HZ;
                double time = pConv->cycle * f / pConv->reps; // average ms
                double mintime = pConv->mincycle *  f;
                double maxtime = pConv->maxcycle *  f;
                sum_time += time;
                printf((flagCSV?", %f, %f, %f, %f"
                            :" \tTIME = %8.3f msec [%8.3f,%8.3f] DIFF = %f"),
                        time, mintime, maxtime, diff);
                printf("\n");

#if TESTDATA
                // NEW detailed stats...
                if(r==0){
                    if(n_test_data < MAX_TEST_DATA){
                        init_TestData(&test_data[ntd],
                                t, pNetwork[t].pName,
                                (size_t)0, "libvednn-std", 0/*doStd*/);
                        test_data[ntd].diff = diff;
                        test_data[ntd].ops = pConvBuff[t].ops;
                        ++n_test_data;
                    }
                }
                if(ntd < MAX_TEST_DATA){
                    test_data[ntd].sum_times += pConv->cycle;
                    test_data[ntd].reps = r+1;
                    ++ntd;
                }
#endif
            }
            printf("%4u tests. avg TIME = %9.3f msec. max DIFF = %f\n",(unsigned)nEntry, sum_time/nEntry, max_diff);
            fflush(stdout);
        }
    }
#if TESTDATA
    n_test_std = n_test_data;
    if( n_test_std > 0 ){
        printf("\n\n doStd ; libvednn default...");
        print_test_data(test_data,0,n_test_std,HZ);
    }
#endif

    // run convolutions, all available impls, libvednnx iterator api
    if(doItr){
#if TESTDATA
        size_t const n_test_dest = n_test_data;
#endif
        size_t const maxImpls = 64;
        double max_diff = 0.0;
        unsigned long long c[2];
        unsigned long long sum_times[maxImpls];
        unsigned rep_times[maxImpls];
        for(int imp=0; imp<maxImpls; ++imp) {sum_times[imp] = 0ULL; rep_times[imp] = 0U;}
        for(int r=0; r<reps; ++r){
            vednnError_t rv;
#if TESTDATA
            size_t ntd = n_test_dest; // statistics entries get appended to test_data[]
#endif
            for (t=0; t<nEntry; ++t) {
                conv *pConv = &pConvBuff[t];

                char name[80];

                // Convolution
#if 0
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

                        c[0] = __cycle();
                        FTRACE_BEGIN(name);
                        // NB:             CONVX_.....order for low-level call
                        rv = (*iter->impl)(CONVX_FWDB_ORDER(FWDB_PARMS, FWDB_DATA));
                        c[1] = __cycle();
                        FTRACE_END(name);
                        int idx = (iter - vednnConvForwardAddBiasList);
                        //printf(" idx=%d",idx);
                        if( idx>=0 && idx < maxImpls ) {sum_times[idx] += c[1] - c[0]; ++rep_times[idx];}

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
                            if(diff > max_diff) max_diff = diff;
                            printf("%s %8.3f ms DIFF = %f %s\n",(rv==VEDNN_SUCCESS?" OK":"BAD")
                                    , (1.0e3/HZ)*(c[1]-c[0]), diff ,name);
                            fflush(stdout);
#if TESTDATA
                            // NEW detailed stats...
                            if(n_test_data < MAX_TEST_DATA){
                                init_TestData(&test_data[ntd],
                                        t, pNetwork[t].pName,
                                        (size_t)idx, vednnConvForwardAddBiasList[idx].shortname,
                                        1/*doItr*/);
                                test_data[ntd].diff = diff;
                                test_data[ntd].ops = pConvBuff[t].ops;
                                ++n_test_data;
                            }
#endif
                        }
#if TESTDATA
                        if(ntd < MAX_TEST_DATA){
                            test_data[ntd].sum_times += c[1] - c[0];
                            test_data[ntd].reps = r+1;
                            ++ntd;
                        }
#endif

                        if(++iter_cnt>=100) ERROR_EXIT("run-away iter over ConvForwardBias impls?");
                    }
                }else
#endif
                {
#define FWD_PARMS \
                    /* */ pConv->pParamIn,     \
                    /* */ pConv->pParamKernel, \
                    /* */ pConv->pParamBias, \
                    /* */ pConv->pParamOut,    \
                    /* */ pConv->pParamConv,   \
                    /* */ VEDNN_CONV_ALGORITHM_DIRECT
#define FWD_DATA \
                    /* */                      pConv->pDataIn, \
                    /* */                      pConv->pDataKernel, \
                    /* */                      pConv->pDataBias, \
                    /* */                      pConv->pDataOut
                    // libvednnx "iterator over impls"
                    // 1 i            printf(" t%ld", (long)omp_get_num_threads()); fflush(stdout);
                    // What you want: printf(" t%ld", (long)omp_get_max_threads()); fflush(stdout);
                    int iter_cnt=0;
                    for (vednnConvForwardImpls * iter = vednnConvForward_Begin(FWD_PARMS);
                            iter->okfn != NULL;
                            iter = vednnConvForward_Next(iter,FWD_PARMS))
                    {
                        // so grep -k11 of ftrace will sort from "best" to "worst" [approx]
                        snprintf(name,80,"%s:%d:%s",pConv->region,iter_cnt,iter->shortname);
                        printf(" iter name %s... ",name); fflush(stdout);

                        // _realNext supports impls doing runtime check on ptr alignment
                        // also checked in _Run, but do it to get real name of impl
                        FTRACE_BEGIN("realNext");
                        vednnConvForwardImpls *actual = vednnConvForward_realNext(
                                iter, FWD_PARMS, FWD_DATA );
                        FTRACE_END("realNext");
                        assert( actual != NULL );
                        if ( actual->okfn == NULL ) {
                            printf(" not run: %s\n", name);
                            continue;
                        }
                        if( actual == iter ){ // almost always...
                            snprintf(name,80,"%s:%d:%s",pConv->region,iter_cnt,iter->shortname);
                        }else{
                            snprintf(name,80,"%s:%d:%s-->%s",pConv->region,iter_cnt,iter->shortname,actual->shortname);
                        }
                        c[0] = __cycle();
                        FTRACE_BEGIN(name);
                        // now we also want omp support, so we call via _Run, not via (*actual->impl)
                        vednnConvForward_out_t const out = vednnConvForward_Run( actual, FWD_PARMS, FWD_DATA );
                        // ignore out.status  (hopefully VEDNN_SUCCESS)
                        c[1] = __cycle();
                        FTRACE_END(name);
                        int idx = (actual - vednnConvForwardList);
                        //printf(" idx=%d",idx);
                        if( idx >= 0 && idx < maxImpls ) {
                            sum_times[idx] += c[1] - c[0]; ++rep_times[idx];
                        }
                        assert( out.actual == actual );
                        rv = out.status;
                        if(r == 0 && actual != NULL){
                            if( pConv->pParamIn->batch == 1 ) {
                                printf(" batch 1 group=%d inChannel=%d outChannel=%d",
                                        (int)(pConv->pParamConv->group),
                                        (int)(pConv->pParamIn->channel),
                                        (int)(pConv->pParamOut->channel));
                            }
                            //pConv->cycle += c[1] - c[0]; // this is reserved for the libvednn time
                            //printf(" %s %llu cycles", name, c[1]-c[0]);
                            double diff = diffData(pConv->pParamOut, pConv->pDataOut, pConv->pBufRef);
                            printf("%s DIFF = %f %s\n",(rv==VEDNN_SUCCESS?" OK":"BAD")
                                    , diff ,name);
                            fflush(stdout);
#if TESTDATA
                            // NEW detailed stats...
                            int idx = (actual - vednnConvForwardList);
                            if(n_test_data < MAX_TEST_DATA){
                                init_TestData(&test_data[ntd],
                                        t, pNetwork[t].pName,
                                        (size_t)idx, vednnConvForwardList[idx].shortname,
                                        1/*doItr*/);
                                test_data[ntd].diff = diff;
                                test_data[ntd].ops = pConvBuff[t].ops;
                                ++n_test_data;
                            }
#endif
                        }
#if TESTDATA
                        if(ntd < MAX_TEST_DATA){
                            test_data[ntd].sum_times += c[1] - c[0];
                            test_data[ntd].reps = r+1;
                            ++ntd;
                        }
#endif
                        if(++iter_cnt>=100) ERROR_EXIT("run-away iter over ConvForward impls?");
                    }
                }
                if (rv != VEDNN_SUCCESS) ERROR_EXIT("convolution() failed.");

            }
            FTRACE_END("all convolution");
        }
#if 0 // this did not work well.  ncc2+ is OK, but follows API in cjitConv.h
        // First make cjitConv01.cpp work nicely.
        // Then make a nice 'C' api for compiling multiple sources into a library
        // Doing everything via a 'bin.mk' is old and stupid, but might actually
        // be the best approach to flexibly allow flags and different compilers
        // to be customized?
        // We cannot actually do the jit call here, but IF cjit00 existed
        // AND the jit impl matched our params, we could run it, using
        // code something like:
        if(0){
            // Run cjit00.so function, if it exists (it is a 'make' target, requires clang)
            char const * dll = "cjit00.so";
            char const * symbol = "cjitConvolutionForward00_wrongname";
            char const * path = "/where/is/my/pwd";
            char const * fullpath = "/where/is/my/pwd/cjit00.so";
            // TODO get path of current directory, form fullpath and THEN dlopen
            void *jitLibHandle = dlopen(fullpath, RTLD_NOW);
            if(!jitLibHandle){
                printf(" Unable to dlopen %s [skipping]\n", dll);
            }else{
                void* cjit00_sym = dlsym(jitLibHandle, symbol);
                if(!cjit00_sym){
                    printf(" Unable to find symbol %s in %s\n",dll,symbol);
                }else{
                    printf(" Found symbol %s from %s at %p\n",dll,symbol,cjit00_sym);
                    vednnConvForward_t func = (vednnConvForward_t)cjit00_sym;
                    FTRACE_BEGIN("cjit00");
                    //func( pParamIn,pDataIn, pParamKernel,pDataKernel, pParamConv,
                    //        pParamOut,pDataOut );
                    FTRACE_END("cjit00");
                }
            }
        }
#else // see cjitConv.h
#endif

#if 1 // original stats by {impl}
        for(int imp=0; imp<maxImpls; ++imp){
            if(sum_times[imp]){
                //char const* shortname = flagBias
                //    ? vednnConvForwardAddBiasList[imp].shortname
                //    : vednnConvForwardList       [imp].shortname;
                char const* shortname = vednnConvForwardList[imp].shortname;
                double f = 1.0e3 / HZ;
                double time = sum_times[imp] * f / rep_times[imp]; // average ms
                printf(" impl %2d %30s ran %-6u times, avg %f ms\n", imp, shortname, rep_times[imp], time);
            }
        }
#endif
        printf(" max DIFF = %f\n",max_diff);
    }
#if TESTDATA // NEW: detailed stats by {test,impl}
    n_test_itr = n_test_data;
    assert(n_test_data <= MAX_TEST_DATA);
    print_test_data(test_data,n_test_std,n_test_itr,HZ);
#endif

    if(doJit){
        // hmmm I want to only JIT ONCE, use many times
        // XXX FOR NOW assume all jit symbols are forward no bias
        // added ability for C api to track which nEntry..
        char const* generators[] = {
            "cjitConvFwd1q",
            "cjitConvFwd6",
            "cjitConvFwd1p",
            "cjitConvFwd1",
            "cjitConvFwd1b", // 1 with mask precalc
            "cjitConvFwd3",
            "cjitConvFwd4",
            "cjitConvFwd2",
            "cjitConvFwd5",
            //"cjitConvFwd1b", // <-- msk (memory buffer issues for larger tests)
                NULL};
        struct CjitOpt cjitOpt= { NULL, 0, 0 }; // { "tmp_cjitConv", full prep, full build }
        CjitSyms const* allsyms = cjitSyms( pNetwork, nEntry, generators, &cjitOpt );

        if(allsyms->len==0U){
            printf("No jit2yy impls at all\n");
        }if(flagBias){
            printf("No bias impls yet\n");
        }else{
            printf("XXX Assuming all symbols are really vednnConv##Forward##_t impls\n");
            double max_diff = 0.0;
            unsigned long long c[2];
            unsigned long long* sum_times = (unsigned long long*)malloc(allsyms->len*sizeof(unsigned long long));
            unsigned* rep_times = (unsigned*)malloc(allsyms->len*sizeof(unsigned));
            if (!sum_times || !rep_times) ERROR_EXIT("Memory exhausted.");
            //if you have one JIT generator...assert( allsyms->nSrc == nEntry );
            printf(" parms: nEntry=%ld, CjitSyms: len=%ld\n",(long)nEntry,(long)allsyms->len);
            for(int imp=0; imp<allsyms->len; ++imp) {sum_times[imp] = 0ULL; rep_times[imp] = 0U;}
#if TESTDATA
            size_t const n_test_dest = n_test_data;
#endif
            for(int r=0; r<reps; ++r){
                vednnError_t rv;
#if TESTDATA
                size_t ntd = n_test_dest; // statistics entries get appended to test_data[]
#endif
                for (t=0; t<nEntry; ++t) { // test t, which of 'nEntry' convolution params...
                    conv *pConv = &pConvBuff[t];

                    CjitSym const* cjs = allsyms->syms;
                    size_t njit_t=0U;
                    for(size_t cj=0U; cj < allsyms->len; ++cj){
                        // 'tag' has been used to remember which test 't' by cjitConv.cpp
                        if( cjs[cj].tag != t || cjs[cj].ptr == NULL ) // skip symbol/functions not for test 't'
                            continue;

                        ++njit_t;
                        char const* name = cjs[cj].sym;
                        //name             = cjs[cj].sym;
                        void const* ptr  = cjs[cj].ptr;
                        //if(r==0) printf(" t%d cj%lu name=%s @ %p\n",(int)t,(long unsigned)cj,name,ptr);
                        if(r==0){
                            printf(" t%d cj%-4lu",(int)t,(long unsigned)cj);
                            fflush(stdout);
                        }
                        //if(r==0) printf(" t %d cj %lu name %s\n",(int)t,(long unsigned)cj,name);
                        assert( name != NULL );
                        assert( ptr != NULL );
                        // CjitConvFwd1 is a "default" impl, so no need to check an _ok functiion
                        // eventually will need to return the syms for a vednnConvolutionLists.h entry
                        // so that we can check ok? and rtok? functions, before invoking the
                        // actual function XXX
                        vednnConvForward_t impl = (vednnConvForward_t)ptr;
                        c[0] = __cycle();
                        FTRACE_BEGIN(name);
                        rv = (*impl)(CONVX_FWD_ORDER(FWD_PARMS, FWD_DATA));
                        c[1] = __cycle();
                        FTRACE_END(name);
                        sum_times  [cj] += c[1] - c[0]; 
                        ++rep_times[cj];
                        if (r == 0){
                            printf(" mb%dg%d_ic%doc%d",
                                    (int)(pConv->pParamIn->batch),
                                    (int)(pConv->pParamConv->group),
                                    (int)(pConv->pParamIn->channel),
                                    (int)(pConv->pParamOut->channel));
                            //pConv->cycle += c[1] - c[0]; // this is reserved for the libvednn time
                            //printf(" %s %llu cycles", name, c[1]-c[0]);
                            double diff = diffData(pConv->pParamOut, pConv->pDataOut, pConv->pBufRef);
                            if(diff > max_diff) max_diff = diff;
                            printf("%s DIFF = %f ~%f ms %s",(rv==VEDNN_SUCCESS?" OK":"BAD")
                                    , diff, (c[1]-c[0])*(1.e3/HZ), name);
                            printf("\n");
                            fflush(stdout);
#if TESTDATA
                            // NEW detailed stats...
                            if(n_test_data < MAX_TEST_DATA){
                                long const idx = cj; // invariant: t == cjs[cj].tag
                                init_TestData(&test_data[ntd],
                                        t, pNetwork[t].pName,
                                        (size_t)idx, cjs[cj].sym, 2/*JIT*/);
                                test_data[ntd].diff = diff;
                                test_data[ntd].ops = pConvBuff[t].ops;
                                ++n_test_data;
                            }
#endif
                        }
#if TESTDATA
                        if(ntd < MAX_TEST_DATA){
                            test_data[ntd].sum_times += c[1] - c[0];
                            test_data[ntd].reps = r+1;
                            ++ntd;
                        }
#endif
                    }
                    if(r==0 && njit_t==0){
                        // XXX better, record the "duplicate test" parent tag
                        printf(" t %d duplicate\n",(int)t);
                    }
                }
            }
#if TEST_DATA // NEW: detailed stats by {test,impl}
            assert(n_test_data <= MAX_TEST_DATA);
            print_test_data(test_data,n_test_itr,n_test_data,HZ);
#endif
            for (t=0; t<nEntry; ++t) {          // summary printout
                conv *pConv = &pConvBuff[t];
                CjitSym const* cjs = allsyms->syms;
                for(size_t cj=0U; cj < allsyms->len; ++cj){
                    if( cjs[cj].tag != t )
                        continue;

                    if(sum_times[cj]!=0){
                        // XXX do not have 'List' support yet XXX
                        //char const* shortname = flagBias
                        //    ? vednnConvForwardAddBiasList[imp].shortname
                        //    : vednnConvForwardList       [imp].shortname;
                        char const* shortname      = cjs[cj].sym;
                        double f = 1.0e3 / HZ;
                        double time = sum_times[cj] * f / rep_times[cj]; // average ms
                        printf(" test %2d %30s ran %-6u times, avg %f ms %s\n",
                                cj, shortname, rep_times[cj], time,
                                pConv->region);
                    }
                }
            }
        printf(" max jit DIFF = %f\n",max_diff);
        free(sum_times);
        free(rep_times);
        }
    }// end JIT impls
#if TESTDATA
    n_test_jit = n_test_data;
#endif

#if TESTDATA
    // final summary including all doStd,doItr,doJit implementations together
    if( n_test_itr > 0 && n_test_jit > n_test_itr ){
        printf("\n\n libvednn and JIT impls combined...");
        print_test_data(test_data,0,n_test_data,HZ); // test_data sorted by test, then avg speed.
        print_wins(test_data,0,n_test_data,HZ); // algorithm dominance matrix
    }
#endif
#undef FWD_DATA
#undef FWD_PARMS
#undef FWDB_DATA
#undef FWDB_PARMS
#undef MAX_TEST_DATA


    // release
    fflush(stdout);
    for (t=0; t<nEntry; ++t) {
        testconvForward_free(&pConvBuff[t],flagBias);
    }
    free(pConvBuff);
}

void testBackwardData(struct param *pNetwork, int nEntry, double HZ, int flagCSV, int reps, filterLayout_t filter_layout)
{
    int t;
    typedef struct testconvBackwardData conv;

    conv *pConvBuff = (conv *) XMALLOC(nEntry * sizeof(conv));

    for (t=0; t<nEntry; ++t) {
        conv *pConv = &pConvBuff[t];
        struct param *pNw = &pNetwork[t];
        testconvBackwardData_init(pConv);
        strcpy(pConv->region, pNw->pName);
        snprintf(pConv->ref_region, 128, "<gemm:BkwD> %s",pConv->region);
    }

    for (t=0; t<nEntry; ++t) {
        conv *pConv = &pConvBuff[t];
        struct param *pNw  = &pNetwork[t];
        testconvBackwardData_alloc(pConv, pNw, filter_layout);
        testconvBackwardData_randomData( pConv );
    }

    if(1) for(t=0; t<nEntry; ++t) {
        testconvBackwardData_dumpParms(&pConvBuff[t]);
    }

    if(1){ // run Reference Convolution
        testconvBackwardData_refcalcs( pConvBuff, nEntry );
    }

    // run test Convolution
    if(1) for(int r=0; r < reps; ++r){
        testconvBackwardData_vednncalcs( pConvBuff, nEntry );
    }

    if(1){ // output section
        if (flagCSV) dumpParamCSV_title();
        for (t=0; t<nEntry; ++t) {
            conv *pConv = &pConvBuff[t];
            struct param *pNw = &pNetwork[t];
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
    }

    // release
    for (t=0; t<nEntry; ++t) {
        conv *pConv = &pConvBuff[t];
        testconvBackwardData_free(pConv);
    }
    free(pConvBuff);
}

void testBackwardFilter(struct param *pNetwork, int nEntry, double HZ, int flagCSV, int reps, filterLayout_t filter_layout)
{
    int t;
    typedef struct testconvBackwardFilter conv;
    conv *pConvBuff = (conv *) XMALLOC(nEntry * sizeof(conv));

    for (t=0; t<nEntry; ++t) {
        conv *pConv = &pConvBuff[t];
        struct param *pNw = &pNetwork[t];
        testconvBackwardFilter_init(pConv);
        strncpy(pConv->region, pNw->pName, 128);
        snprintf(pConv->ref_region, 128, "<gemm:BkwF> %s",pConv->region);
    }

    for (t=0; t<nEntry; ++t) {
        conv *pConv = &pConvBuff[t];
        struct param *pNw  = &pNetwork[t];
        testconvBackwardFilter_alloc( pConv, pNw, filter_layout);
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
    for (t=0; t<nEntry; ++t) {
        conv *pConv = &pConvBuff[t];
        struct param *pNw = &pNetwork[t];
        if (flagCSV) dumpParamCSV(pNw,"Bkw","F");
        else         dumpParam   (pNw,"Bkw","F");

        double diff = diffData((vednnTensorParam_t*)(pConv->pParamGradKernel), pConv->pDataGradKernel, pConv->pBufRef);
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
    for (t=0; t<nEntry; ++t) {
        conv *pConv = &pConvBuff[t];
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
    char const* pName;
    int   testtype;
} tests[] = {
    { "Forward",    CONV_TEST_FORWARD } ,
    { "ForwardAddBias",    CONV_TEST_FORWARD_ADDBIAS } ,
    { "BackwardData",    CONV_TEST_BACKWARD_DATA } ,
    { "BackwardFilter", CONV_TEST_BACKWARD_FILTER } ,
    //    { "BackwardBias", CONV_TEST_BACKWARD_BIAS } ,   // not implemented
};

static struct {
    char *pName;
    filterLayout_t   layouttype;
} filterLayout[] = {
    { "filter_nchw",    VEDNN_FILTER_LAYOUT_NCHW } ,
    { "filter_hwcn",    VEDNN_FILTER_LAYOUT_HWCN }
};

char const* default_parameter_file="mb27g1_ic3ih27iw270_oc47oh14ow135_kh3kw3_ph1pw1_sw2sh2_dw1dh1";
// oc47 ~ 47=0x2f : this many outChannelGroups will check many hand-unrolled code cases
static void help(){
    printf( "\nve_cmpconv:"
            "\n   -p PATH   convolution parameter file"
            "\n             -- or --  a STRING like"
            "\n   -p mb27g1_ic3ih22iw22_oc100oh22ow22_kh5kw5_ph3pw3_sw1sh1_dw1dh1"
            "\n             where '_' are ignored."
            "\n   [ -p %s ]"
            "\n   -M like -p STRING but parameters use mkl-dnn convention (dilation=0,1,...)"
            "\n             Add 1 to mkl-dnn dilation values to comply with libvednn"
            "\n optional:"
            "\n   -C        CSV output"
            "\n   -T STRING test type: [Forward],"
            "\n             ForwardAddBias, BackwardData, BackwardFilter"
            "\n   -r INT    reps [1]"
            "\n   -H FLOAT  Timer register speed (Hz) [0.8e9]"
            "\n   -t N      omp_set_num_threads(N), then run"
            "\n             N < 0 means repeat for N=0..8 (don't use ftrace output)"
            "\n   -f STRING filter layout: [filter_nchw] filter_hwcn"
            "\n PATH file format:"
            "\n   First line: number of tests"
            "\n   Test lines: either libvednn CSV format (name,mb,g, ic,ih,iw, oc,oh,ow, kh,kw, sh,sw, ph,pw)"
            "\n         vednn dilation values begin at 1 (1x dilation)"
            "\n   OR mkl-dnn STRING"
            "\n     mkl-dnn dilations starts at 0 'none'"
            "\n     ex. mb8g1_ic3ih27iw270_oc16oh14ow135_kh3kw3_ph1pw1_sw2sh2_dw1dh1nMYNAME"
            "\n         '_' is ignored and 'nREST-OF-LINE' terminates parse"
            "\n         stride, dilation default to 0 and 0"
            "\n         i,o,k,p,s with 1 Height/Width imply Height==Width"
            "\n         missing pad is calculated (or set to zero if not calculable)"
            "\n         missing oh,ow are calculated if possible"
            "\n   TODO: #-comments, ignore whitespace lines, remove need for counting tests in file"
            "\n", default_parameter_file);
}
int main(int argc, char **argv)
{
    extern int optind;
    extern char *optarg;
    int opt;

    char const * pParamPath = NULL ;
    double HZ        = 0.8e9 ;
    int testtype     = 0 ;
    int flagCSV         = 0 ;
    int reps         = 1 ;
    int threads      = 1 ; /* set to -ve to repeat for 0..8 threads */
    char m_for_mkldnn = 'v';
    filterLayout_t filter_layout = VEDNN_FILTER_LAYOUT_NCHW;

    const rlim_t kStackSize = 16 * 1024 * 1024;   // min stack size = 16 MB
    struct rlimit rl;
    int result;

    result = getrlimit(RLIMIT_STACK, &rl);
    if (result == 0)
    {
        printf(" current stack size limit is %ld\n",(long)(rl.rlim_cur));
        if (/*rl.rlim_cur >= 0 &&*/ rl.rlim_cur < kStackSize)
        {
            rl.rlim_cur = kStackSize;
            result = setrlimit(RLIMIT_STACK, &rl);
            if (result != 0)
            {
                fprintf(stderr, "setrlimit returned result = %d\n", result);
            }else{
                printf("stack size set to %ld\n",(long)kStackSize);
            }
        }
    }else{
        printf(" getrlimit(RLIMIT_STACK) returned %ld",(long)result);
    }

    printf("Test program: %s\n",__FILE__);

#define PARAMBUFSZ 80
    char paramBuf[PARAMBUFSZ+1];
    while ((opt = getopt(argc, argv, "p:M:CH:T:t:r:f:")) != -1) {
        switch (opt) {
        case 'M': m_for_mkldnn = 'm'; // fall-through
        case 'p':
            snprintf(paramBuf,PARAMBUFSZ,"%s",optarg);
            pParamPath = &paramBuf[0];
            break;
        case 'r':    reps = (int)atof(optarg); break;
        case 'C':    flagCSV = 1;        break;
        case 'H':    HZ = atof(optarg);    break;
        case 'T':
                     {
                         int found = 0;
                         for (int t=0; t<sizeof(tests)/sizeof(tests[0]); ++t) {
                             if (strcasecmp(optarg, tests[t].pName) == 0) {
                                 testtype = tests[t].testtype ;
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
        case 'f' :
                     {
                         int found = 0;
                         for (int i=0; i<sizeof(filterLayout)/sizeof(filterLayout[0]); i++) {
                             if (strcasecmp(optarg, filterLayout[i].pName) == 0) {
                                 filter_layout= filterLayout[i].layouttype ;
                                 found = 1;
                                 break;
                             }
                         }
                         if (! found )  {
                             fprintf(stderr, "Invalid filter layout.\n");
                             exit(1);
                         }
                     }
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
        //fprintf(stderr, "Unexpected argument after options\n");
        //exit(1);
        // simpler to just pretend it is an mkl-dnn test file name, like '-p'
        snprintf(paramBuf,PARAMBUFSZ,"%s",argv[optind]);
        pParamPath = &paramBuf[0];
        ++optind;
    }
    if (optind < argc) {
        fprintf(stderr, "Unexpected arguments\n");
        exit(1);
    }
    if ( pParamPath == NULL ) {
        //pParamPath = "./params/conv/alexnet.txt";
        pParamPath = default_parameter_file;
        fprintf(stderr, "Parameter file, '-p' option, defaults to %s.\n", pParamPath);
    }
    if (HZ <= 0.0) {
        HZ = 8e8;
        fprintf(stderr, "Processor core frequency, '-H' options, defaulting to 8e8.\n");
    }
    if (reps < 1) {
        fprintf(stderr, "reps must be >= 1\n");
        exit(1);
    }
    printf("CONVOLUTION TEST TYPE    = %s\n",       tests[testtype].pName) ;
    printf("PROCESSOR CORE FREQUENCY = %.3e HZ\n", HZ);
    printf("FILTER LAYOUT            = %s\n",      filterLayout[filter_layout].pName) ;
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


    for(int t=0; t<nParams; ++t){
        // Convolution tests don't handle some illegal cases well
        //    (segfault or bus error)
        // Some acceptable configs report bad DIFF and also get massaged
        // to "exact-sized" output height and width.
        //
        // This allows you to use randomly-edited parameter files and
        // at least avoid segfaults/bus errors/GEMM illegal value messages.
        // (exact output height width can be tricky to guess correctly).
        //
        dumpParam(&pParams[t],"BEFORE","");
        printf("mkConsistent..%d\n",t);
        mkConsistent( &pParams[t] );
        dumpParam(&pParams[t],"AFTER","");
        char buf[100];
        printf("mkl-dnn format : %s\n", param_cstr_short(&pParams[t],buf,100));
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
            testForward(pParams, nParams, HZ, 0, flagCSV, reps, filter_layout);
            break ;
        case CONV_TEST_FORWARD_ADDBIAS :
            testForward(pParams, nParams, HZ, 1, flagCSV, reps, filter_layout);
            break ;
        case CONV_TEST_BACKWARD_DATA :
            testBackwardData(pParams, nParams, HZ, flagCSV, reps, filter_layout);
            break ;
        case CONV_TEST_BACKWARD_FILTER :
            testBackwardFilter(pParams, nParams, HZ, flagCSV, reps, filter_layout);
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
