/** \file
 * program to read conv specs and compute timings for all available impls
 */
// OK in 'c" : static struct testconvForward tcfFoo;
// OK in'c++ : static struct testconvForward tcfFoo;
#include "conv_test_param.hpp"
#include "vednnx.h"
#include "vednn-def.h"  // fast vednn_get/set_num_threads macros
#include "cjitConv.h"
#include "vechash.hpp"

#include "testdata.hpp"
#include "timer.h"

#include <assert.h>
#include <unistd.h>
#include <dlfcn.h>
#include <sys/resource.h> // getrlimit, setrlimit

#include <cmath>
#include <cctype>           // tooupper, ...
#include <iostream>
#include <vector>

#include <csignal>

using namespace std;

static char const* jit_dir = nullptr;
extern "C" {
static volatile sig_atomic_t control_c_flag = 0;
void set_control_c_flag(int sig){ // can be called asynchronously
    ++control_c_flag; // don't care about races.  You can't hit keyboard that fast.
}
static inline int check_control_c() {
    return control_c_flag;
}
}

struct JitconvDo {
    JitconvDo()
        : doRef(1), doStd(1), doItr(1), doJit(1), doJitunrolls(1),
        filter_layout(VEDNN_FILTER_LAYOUT_NCHW)
    {}
    int doRef; // do ref gemm calc
    int doStd; // do libvednn default calc
    int doItr; // do libvednnx "all" calc
    int doJit; // (for forward)
    int doJitunrolls; // jit unroll variant?
    //... according to "-o[R|S|I|J]" default -oRSIJ
    filterLayout_t filter_layout; // VEDNN_FILTER_LAYOUT_[NCHW|HWCN]
    // ... according to "-f filter_[nchw|hwcn]" argument
};

#if 0 // unused
static std::string getPath() {
    long const sz = pathconf(".",_PC_PATH_MAX); // assume we are interested cwd
    if(sz<=0) THROW("Invalid max path length?");
    char* const temp=(char*)malloc((size_t)sz);
    if(temp==nullptr) THROW("Out of memory");
    if ( getcwd(temp, sz) != 0) 
        return std::string(temp);
    int error = errno;
    switch ( error ) {
        // sz>0 alreay checked (no EINVAL)
        // PATH_MAX includes the terminating nul (no ERANGE)
      case EACCES: THROW("Access denied");
      case ENOMEM: THROW("Insufficient storage"); // is this possible?
      default: THROW("Unrecognised errno="<<error);
    }
}
#endif

struct CacheKiller {
    CacheKiller()
        : data(2000000)
    {
        using namespace scramble64;
        size_t const n = data.size();
        uint64_t *v = data.data();
        for(uint64_t i=0; i<n; ++i) v[i] = i*r1+13U;
    }
    /** compute no result, but read/write? a lot of memory [if enabled]. */
    void operator()(){
        if(enable){
            //cout<<" !CacheKiller"; cout.flush();
            using namespace scramble64;
            // NOTE : vector::operator[] is NOT INLINED by nc++ !!! XXX
            // NOTE : vector::size() is NOT INLINED by nc++ !!! XXX
            size_t const n = data.size();
            uint64_t *v = data.data();
            // write:
            if(slow){ // read and write all
#if 0 // 26% of time for a simple test (ouch)
#pragma omp parallel for
                for(uint64_t i=0; i<n; ++i) v[i] = v[i]*r1+13U;
#else
                uint64_t const start = v[0]%11U;
                uint64_t const step = (v[0]&1? 10U: 11U);
#pragma omp parallel for
                for(uint64_t i=start; i<n; i+=step) v[i] = v[i]*r1+13U;
#endif
            }else{ // read all (and updata 1 rand element):
                uint64_t h;
#pragma omp parallel for reduction(+:h)
                for(uint64_t i=0; i<n; ++i) h += v[i]*r1;
                size_t somewhere = (h*r1+13) % n;
                v[somewhere] = h;
            }
        }
    }
    /** destructor needs an unpredictable element,
     * so compiler can't optimize away operator(). */
    ~CacheKiller(){
        using namespace scramble64;
        if(enable){
            size_t somewhere = (data[0]*r1) % data.size();
            if(somewhere == 12345u){ // almost always quiet...
                cout<<" -CacheKiller["<<somewhere<<" of "<<data.size()
                    <<"]%13U="<<data.at(somewhere)%13U<<endl;;
            }
        }
    }
    operator bool() volatile { return enable; }
    std::vector<uint64_t> data;
    static volatile bool enable;
    static bool slow;
};
volatile bool CacheKiller::enable = false;
bool CacheKiller::slow = true;

/** return list of jit symbols (Forward convolutions->libcjitConv.so).
 * caller must cjitSyms_free(return value) */
static CjitSyms const* getForwardJitSymbols(struct param const* pNetwork, int nEntry, int const flagBias)
{
    // hmmm I want to only JIT ONCE, use many times
    // XXX FOR NOW assume all jit symbols are forward no bias
    // added ability for C api to track which nEntry..
    // If you change these, also check CjitConv.cpp dfgNames[] and Makefile
    char const* generators[] = {
        "cjitConvFwd1q", // STANDARD ITEM
        "cjitConvFwd6", // STANDARD ITEM
        // MOVED into cjitConvFwd6:    "cjitConvFwd6vel",
        //"cjitConvFwd1", // usually slower
        //"cjitConvFwd1p", // usually slower, 1 with ptrs vs offsets
        //"cjitConvFwd1b", // usually slower? 1 with mask precalc
        //"cjitConvFwd2", // usually slower
        //"cjitConvFwd3", // usually slower
        // 4 and 5 have a bug in kBy1 loop.  Should cross-check with Fwd3 and fix TODO
        //"cjitConvFwd4",
        //"cjitConvFwd5",
        NULL};
    // jit_dir is now settable as -S SUBDIR [default tmp_cjitConv]
    struct CjitOpt cjitOpt= { jit_dir, 0, 0 }; // "tmp_cjitConv", full prep, full build 
    CjitSyms const* allsyms = cjitSyms( pNetwork, nEntry, generators, &cjitOpt );

    if(allsyms->len==0U){
        printf("No jit2yy impls at all\n");
        cjitSyms_free(allsyms);
        allsyms=nullptr;
    }if(flagBias){
        printf("No bias impls yet\n");
        cjitSyms_free(allsyms);
        allsyms=nullptr;
    }else{
        printf("XXX Assuming all symbols are really vednnConv##Forward##_t impls\n");
    }
    return allsyms;
}

void testForward(struct param *pNetwork, int nEntry, double HZ, int flagBias, int flagCSV, int reps, JitconvDo const& doOpt)
{
    // extract common test run options
    int const doRef = doOpt.doRef;
    int const doStd = doOpt.doStd;
    int const doItr = doOpt.doItr;
    int const doJit = doOpt.doJit;
    int const doJitunrolls = doOpt.doJitunrolls;
    filterLayout_t const filter_layout = doOpt.filter_layout;

    int t;
    typedef struct testconvForward conv;
    printf("Beginning %s\n",__PRETTY_FUNCTION__);

    // prepare the jit library for ALL jit impls we'll need (full libcjitConv.so)
    CjitSyms const* allsyms = (!doJit? nullptr : getForwardJitSymbols(pNetwork,nEntry,flagBias));
    if(allsyms){
        printf(" allsyms : loaded forward convolution jit impls from libcjitConv.so\n");
    }

    cout<<" testForward vednn thr="<<vednn_get_num_threads()<<"="<<omp_get_num_threads()<<"/"<<omp_get_max_threads()<<endl;

    TestDataRepo tdRepo(HZ);
    CacheKiller cacheKiller;
    printf(" cacheKiller %s\n", (cacheKiller? "ENABLED": "DISABLED"));

    // NEW approach (save memory if large number of convs
    // TODO cache-killer routine between timing measurements
    for (t=0; t<nEntry; ++t) {
        cout<<"+++ entry "<<t<<" of nEntry="<<nEntry
            <<" testForward vednn thr="<<vednn_get_num_threads()<<"="<<omp_get_num_threads()<<"/"<<omp_get_max_threads()
            <<endl;;
        if(check_control_c()>0){
            char const *control_c_msg="\n\nOh, Control-C detected.  Trying to quit nicely with final summary.";
            cout<<control_c_msg<<endl;
            cerr<<control_c_msg<<endl;
            break;
        }
        struct param *pNw = &pNetwork[t];
        OneConvForward wrk( pNw, flagBias, filter_layout, 1/*verbose*/ ); // this allocates and initializes too (ease-of-use)
        cout<<" OneConvForward vednn thr="<<vednn_get_num_threads()<<"="<<omp_get_num_threads()<<"/"<<omp_get_max_threads()<<endl;
        testconvForward *pConv = &wrk.conv;
        char pNw_param_cstr[100];
        param_cstr_short(pNw,pNw_param_cstr,100);
        cout<<"layer "<<&pNw_param_cstr[0]<<"    ops="<<pConv->ops<<endl;

        if(doRef){
            cout<<" doRef vednn thr="<<vednn_get_num_threads()<<"="<<omp_get_num_threads()<<"/"<<omp_get_max_threads()<<endl;
            //wrk.doRef();
            // OK, now do some timing runs for ref (gemm) calculation
            TestData td( t, pConv->ref_region, (size_t)0/*impl_idx*/, "gemm-Ref",
                    3/*doRef*/, pNw_param_cstr/*test-wide descr*/ );
            double sum_time = 0.0;
            double max_diff = 0.0;
            unsigned long long c[2];
            if(!cacheKiller){
                printf(" wrk.doRef() warmup iterations!\n");
                wrk.doRef();
                wrk.doRef();
            }
            double mintime = 0;
            double maxtime = 0;
            cout<<" doRef-reps vednn thr="<<vednn_get_num_threads()<<"="<<omp_get_num_threads()<<"/"<<omp_get_max_threads()<<endl;
            for(int r=0; r < reps; ++r){
                if(cacheKiller){
                    printf(" cacheKiller!");
                    cacheKiller();
                }
                c[0] = __cycle();
                wrk.doRef();
                c[1] = __cycle();
                // now we have pConv->cycle, ++pConv->reps and output calculated

                if (flagCSV) dumpParamCSV_title();
                if (flagCSV) dumpParamCSV(pNw,"Fwd",(flagBias?"B":""));
                else         dumpParam   (pNw,"Fwd", (flagBias? "B":""));

                // diffData with pBufRef and pDataOut identical gives WRONG RESULT
                //double diff = diffData(pConv->pParamOut, pConv->pDataOut, pConv->pBufRef);
                double diff = 0.0;

                // doRef calc -- we do not care about CSV values much here, report individually
                max_diff = diff > max_diff? diff: max_diff;
                //double time = (c[1]-c[0]) * f / pConv->reps; // average ms
                //double mintime = pConv->mincycle *  f;
                //double maxtime = pConv->maxcycle *  f;
                //sum_time += time;
                double time = (1.0e3/HZ) * (c[1]-c[0]);
                mintime = (r==0 || time<mintime? time: mintime);
                maxtime = (r==0 || time<maxtime? time: maxtime);
                printf((flagCSV?", %f, %f, %f, %f"
                            :" \tTIME = %8.3f msec [%8.3f,%8.3f] DIFF = %f"),
                        time, mintime, maxtime, diff);
                printf("\n");

                td.diff = 0.0;
                td.ops = pConv->ops;
                td.sum_times += c[1]-c[0];
                td.reps = r+1;
                printf("%4u tests. doRef avg TIME = %9.3f msec. max DIFF = %f\n",
                        (unsigned)nEntry, sum_time/nEntry, max_diff);
                fflush(stdout);
            }
            tdRepo.append(td,1/*verbose*/);
            printf("Good, finished doRef, %d reps\n", (int)reps);
            cout<<" Good, finished doRef, "<<reps<<" reps. "
                <<" vednn thr="<<vednn_get_num_threads()<<"="<<omp_get_num_threads()<<"/"<<omp_get_max_threads() <<endl;
            printf(" max DIFF = %f\n",max_diff);
        }

        // run test Convolution, libvednn API
        if(doStd){
            cout<<"doStd pConv->region is "<<pConv->region<<"   layer "<<pNw_param_cstr
                <<"   thr="<<vednn_get_num_threads()<<"="<<omp_get_num_threads()<<"/"<<omp_get_max_threads()<<endl;
            param_cstr_short(pNw,pNw_param_cstr,100);
#if 1
            char extended_impl_name[200];
            {
                // used to always be "libvednn-std", now "libvednn-std:low-level-impl-name"
                // retrieve which impl libvednn-std executes:
                vednnCnvFwdChoice_t const choice = vednnConvolutionForwardChoice(
                        pConv->pParamIn, pConv->pDataIn,
                        pConv->pParamKernel, pConv->pDataKernel,
                        pConv->pParamBias, pConv->pDataBias,
                        pConv->pParamOut, pConv->pDataOut,
                        pConv->pParamConv, VEDNN_CONV_ALGORITHM_DIRECT );
                // typedef struct {
                //   vednnError_t rc;
                //   char const* impl;
                //   vednnConvForward_t pFunc;
                //   int mb_threads;
                // } vednnCnvFwdChoice_t;
                sprintf(extended_impl_name,"libvednn-std:%s\0",choice.impl);
            }
#else
            char const* extended_impl_name = "libvednn-std"
#endif
            TestData td( t, pConv->region, (size_t)0/*impl_idx*/,
                    extended_impl_name, // <-- used to be less informative "libvednn-std"
                    0/*doStd*/, pNw_param_cstr/*test-wide descr*/ );

            double sum_time = 0.0;
            double max_diff = 0.0;
            unsigned long long c[2];
            if(!cacheKiller){ // warmup
                testconvForward_vednncalcs( pConv, 1 ); // set up pConv for calc
                testconvForward_vednncalcs( pConv, 1 ); // set up pConv for calc
                // now we have pConv->cycle, ++pConv->reps and output calculated
                // reset doStd warmup and stats to zero
                pConv->cycle = 0;
                pConv->mincycle = 0;
                pConv->maxcycle = 0;
                pConv->reps = 0;
            }
            cout<<" doStd pre-reps thr="<<vednn_get_num_threads()<<"="<<omp_get_num_threads()<<"/"<<omp_get_max_threads()<<endl;
            for(int r=0; r < reps; ++r){
                if(cacheKiller){
                    cacheKiller();
                }
                testconvForward_oclobber(pConv); // set a few outputs "wrong"
                c[0] = pConv->cycle; // cycle count in xxxvednncalcs is cumulative
                testconvForward_vednncalcs( pConv, 1 ); // official calc
                c[1] = pConv->cycle;

                double diff = !doRef? -13.0
                    : diffData(pConv->pParamOut, pConv->pDataOut, pConv->pBufRef);
                max_diff = diff > max_diff? diff: max_diff;
                double f = 1.0e3 / HZ;
                double time = pConv->cycle * f / pConv->reps; // average ms
                double mintime = pConv->mincycle *  f;
                double maxtime = pConv->maxcycle *  f;
                sum_time += f * (c[1]-c[0]);
                // XXX oh. Probably this one should gather the CSV stats!

                td.diff = diff;
                td.ops = pConv->ops;
                td.sum_times = pConv->cycle;
                td.reps = r+1;
                if(r==r-1){
                    if (flagCSV) dumpParamCSV_title();
                    if (flagCSV) dumpParamCSV(pNw,"Fwd",(flagBias?"B":""));
                    else         dumpParam   (pNw,"Fwd", (flagBias? "B":""));
                    printf((flagCSV?", %f, %f, %f, %f"
                                :" \tTIME = %8.3f msec [%8.3f,%8.3f] DIFF = %f"),
                            time, mintime, maxtime, diff);
                    printf(" avg TIME = %9.3f msec. max DIFF = %f\n",
                        (unsigned)nEntry, sum_time/nEntry, max_diff);
                    printf("\n");
                }
                printf("doStd rep %u/%u : TIME = %9.3f msec. max DIFF = %f\n",
                        (unsigned)td.reps, (unsigned)(reps+1),
                        (1.0e3/HZ)*(c[1]-c[0]), max_diff);
                fflush(stdout);
            }
            printf("test %u doStd %d reps : avg TIME = %9.3f msec. max DIFF = %f\n",
                    (unsigned)t, td.reps, td.sum_times*(1.0e3/HZ)/td.reps, max_diff);
            cout<<" doStd post-reps thr="<<vednn_get_num_threads()<<"="<<omp_get_num_threads()<<"/"<<omp_get_max_threads()<<endl;
            tdRepo.append(td,1/*verbose*/);
        }
        if(doItr){
            param_cstr_short(pNw,pNw_param_cstr,100);
            size_t const maxImpls = 64;
            double max_diff = 0.0;
            unsigned long long c[2];
            unsigned long long sum_times[maxImpls];
            unsigned rep_times[maxImpls];
            for(int imp=0; imp<maxImpls; ++imp) {sum_times[imp] = 0ULL; rep_times[imp] = 0U;}

            std::vector<TestData> vtd;
            for(int r=0; r<reps; ++r){
                cout<<" doItr-rep-"<<r
                    <<" thr="<<vednn_get_num_threads()<<"="<<omp_get_num_threads()<<"/"<<omp_get_max_threads()<<endl;
                vednnError_t rv;
                char name[80];
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
                    int iter_cnt=0;
                    for (vednnConvForwardImpls * iter = vednnConvForward_Begin(FWD_PARMS);
                            iter->okfn != NULL;
                            iter = vednnConvForward_Next(iter,FWD_PARMS))
                    {
                        // so grep -k11 of ftrace will sort from "best" to "worst" [approx]
                        snprintf(name,80,"%s:%d:%s",pConv->region,iter_cnt,iter->shortname);
                        cout<<" doItr iter name "<<name
                            <<"   thr="<<vednn_get_num_threads()<<"="<<omp_get_num_threads()<<"/"<<omp_get_max_threads()<<endl;
                        FTRACE_BEGIN("realNext");
                        //printf(" B"); fflush(stdout);
                        vednnConvForwardImpls *actual = vednnConvForward_realNext(
                                iter, FWD_PARMS, FWD_DATA );
                        FTRACE_END("realNext");
                        assert( actual != NULL );
                        if ( actual == NULL ) {
                            printf(" not run\n");
                            continue;
                        }

                        int const long_ftrace_names = 0;
                        if(long_ftrace_names){
                            if( actual == iter ){ // almost always...
                                snprintf(name,80,"%s:%d:%s",pConv->region,iter_cnt,iter->shortname);
                            }else{
                                snprintf(name,80,"%s:%d:%s-->%s",pConv->region,iter_cnt,iter->shortname,actual->shortname);
                            }
                        }else{
                            snprintf(name,80,"vednn:%s",actual->shortname);
                        }

                        // debug : skip some impls by name ? (choose one 'if')
                        if(0) // SKIP NOTHING
                            //if( strstr(name,"cnvFwd-gemm") == nullptr) // SKIP if not gemm
                            //if( strstr(name,"cnvFwd-gemm") != nullptr) // SKIP if gemm
                        {
                            cout<<" skipping "<<name<<" based on impl name"<<endl;
                            continue;
                        }

                        // pre-timing
                        if(cacheKiller) cacheKiller();
                        testconvForward_dumpParms( pConv, flagBias );
                        printf(" C%s",name); fflush(stdout);
                        if(!cacheKiller){
                            vednnConvForward_Run( actual, FWD_PARMS, FWD_DATA );
                            printf(" c"); fflush(stdout);
                            vednnConvForward_Run( actual, FWD_PARMS, FWD_DATA );
                            printf("c"); fflush(stdout);
                        }
                        // ignore out.status  (hopefully VEDNN_SUCCESS)
                        printf("t"); fflush(stdout);
                        //testconvForward_oclobber(pConv); // set a few outputs "wrong"

                        // now we also want omp support, so we call via _Run, not via (*actual->impl)
                        c[0] = __cycle();
                        FTRACE_BEGIN(name);
                        vednnConvForward_out_t const out = vednnConvForward_Run( actual, FWD_PARMS, FWD_DATA );
                        c[1] = __cycle();
                        FTRACE_END(name);

                        int idx = (actual - vednnConvForwardList);
                        printf(" idx=%d out.status=%d time %f ms%c",(int)idx,(int)out.status,
                                (1.0e3/HZ)*(c[1]-c[0]), (r==0?' ':'\n')); fflush(stdout);
                        if( idx >= 0 && idx < maxImpls ) {
                            sum_times[idx] += c[1] - c[0]; ++rep_times[idx];
                        }
                        assert( out.actual == actual );
                        rv = out.status;
                        assert(actual != NULL);

                        if(r == 0){
                            if( pConv->pParamIn->batch == 1 ) {
                                printf(" batch 1 group=%d inChannel=%d outChannel=%d",
                                        (int)(pConv->pParamConv->group),
                                        (int)(pConv->pParamIn->channel),
                                        (int)(pConv->pParamOut->channel));
                            }
                            //pConv->cycle += c[1] - c[0]; // this is reserved for the libvednn time
                            //printf(" %s %llu cycles", name, c[1]-c[0]);
                            double diff = !doRef? -13.0
                                : (rv != VEDNN_SUCCESS)? -1.0
                                : diffData(pConv->pParamOut, pConv->pDataOut, pConv->pBufRef);
                            max_diff = diff > max_diff? diff: max_diff;
                            printf("%s %8.3f ms DIFF = %f %s\n",(rv==VEDNN_SUCCESS?"  OK":" BAD")
                                    , (1.0e3/HZ)*(c[1]-c[0]), diff ,name);
                            fflush(stdout);

                            // NEW detailed stats...
                            int idx = (actual - vednnConvForwardList);
#if 0
                            vtd.emplace_back(t, pConv->region, (size_t)idx,
                                    vednnConvForwardList[idx].shortname,
                                    1/*doItr*/, pNw_param_cstr/*test-wide descr*/ );
#else
                            TestData tmp = TestData(t, pConv->region, (size_t)idx,
                                    vednnConvForwardList[idx].shortname, // aka actual->shortname
                                    1/*doItr*/, pNw_param_cstr/*test-wide descr*/ );
                            vtd.push_back(tmp);
#endif 
                            assert(vtd.size() == iter_cnt+1U);
                            TestData& td = vtd.at(iter_cnt);
                            td.diff = diff;
                            td.ops = pConv->ops;
                        }
                        TestData& td = vtd.at(iter_cnt);
                        td.sum_times += c[1]-c[0];
                        td.reps = r+1;
                        if(++iter_cnt>=100) ERROR_EXIT("run-away iter over ConvForward impls?");
                    }
                }
                if (rv != VEDNN_SUCCESS){
                    //ERROR_EXIT("convolution() failed.");
                    printf(" conv Fwd impl failure! [continuing anyway]\n");
                }

                FTRACE_END("all convolution");
            }
            tdRepo.append(vtd,1/*verbose*/);
            printf("Good, finished doItr, %d reps\n", (int)reps); fflush(stdout);
            printf(" max DIFF = %f\n",max_diff);
        }
        if(doJit && allsyms)
        {
            cout<<" doJit thr="<<vednn_get_num_threads()<<"="<<omp_get_num_threads()<<"/"<<omp_get_max_threads()<<endl;
            param_cstr_short(pNw,pNw_param_cstr,100);
            cout<<"doJit "<<&pNw_param_cstr[0]<<"    ops="<<pConv->ops<<endl;
            printf("XXX Assuming all symbols are really vednnConv##Forward##_t impls\n");
            double max_diff = 0.0;
            unsigned long long c[2];
            //if you have one JIT generator...assert( allsyms->nSrc == nEntry );
            printf(" parms: nEntry=%ld, CjitSyms: len=%ld\n",(long)nEntry,(long)allsyms->len);
            std::vector<TestData> vtd;
            size_t njit;
            // Since we don't have JIT _ok and _rtok done yet, search for a prev dup run
            bool already_ran = 0;
            {
                char buf[100];
                char xbuf[100];
                param_cstr_short(pNw,buf,100);
                CjitSym const* cjs = allsyms->syms;
                for(size_t cj=0U; cj < allsyms->len; ++cj){
                    if( cjs[cj].ptr == NULL ) // skip symbol/functions not for test 't'
                        continue;
                    if( cjs[cj].tag == t ) // this sym was generated for test 't' (don't check for prev runs)
                        continue;
                    char const* name = cjs[cj].sym;
#ifndef NDEBUG
                    void const* ptr  = cjs[cj].ptr;
                    assert( name != NULL );
                    assert( ptr != NULL );
#endif
                    //printf(" t%d cj%lu name=%s tag %ld\n",(int)t,(long unsigned)cj,
                    //        name,/*ptr,*/ (long)cjs[cj].tag);
                    //fflush(stdout);
                    //
                    // 'tag' has been used to remember which test 't' first
                    // "caused" the JIT code file to be generated (cjitConv.cpp)
                    //
                    // try to find previous run and copy those results here
                    // HACK: if pConv string same as "everything after 1st '_' in name
                    // then we have run (can run) that function.
                    // Or just scan prev runs for impl_name==symbol name <-- easiest
                    //printf("   buf %s",buf);
                    for(auto const& x: tdRepo){
                        if( x.test < t && x.impl_type==2 && x.impl_idx==cj && x.test == cjs[cj].tag ){
                            // previous && JIT and same JIT generator impl and JIT impl generated for test t !
                            // Does the parm string match current layer?
                            auto const xNw = &pNetwork[x.test];
                            param_cstr_short(xNw,xbuf,100);
                            //printf(" xbuf %s\n",xbuf);
                            if(strncmp(buf,xbuf,100)==0){
                                printf(" t%d cj%lu name=%s tag %ld",(int)t,(long unsigned)cj,
                                        name,/*ptr,*/ (long)cjs[cj].tag);
                                printf("    *** already_ran in t%d ***\n",(int)x.test);
                                already_ran = 1;
                                TestData copee(x);
                                copee.test = t;
                                snprintf(xbuf,100," prev:t%d",(int)x.test);
                                copee.appendDescr(xbuf);
                                vtd.push_back(copee); // COPY the already_ran results
                                break;
                            }
                        }
                    }
                }
            }
            if(!already_ran) for(int r=0; r<reps; ++r){
                if(r>0) printf(" doJit-rep-%d",r); fflush(stdout);
                vednnError_t rv;
                CjitSym const* cjs = allsyms->syms;
                njit=0U;
                for(size_t cj=0U; cj < allsyms->len; ++cj){
                    if( cjs[cj].ptr == NULL ) // skip symbol/functions not for test 't'
                        continue;
                    if( cjs[cj].tag != t ){ // skip duplicate run? (caught these with 'already_ran' above
                        continue;
                    }
                    //
                    // TODO To run on "all applicable pConv", need _ok and _rtok JIT fns
                    //      (now we assume a jit impl can handle ANY input, which will soon fail)
                    //

                    char const* name = cjs[cj].sym;
                    //name             = cjs[cj].sym;
                    void const* ptr  = cjs[cj].ptr;
                    assert( name != NULL );
                    assert( ptr != NULL );
                    if(r==0){
                        printf(" t%d cj%lu name=%s @ %p tag %ld\n",(int)t,(long unsigned)cj,
                                name,ptr, (long)cjs[cj].tag);
                        //printf(" t%d cj%-4lu",(int)t,(long unsigned)cj);
                        fflush(stdout);
                    }

                    // skip unrolls? (new option, now that clang is getting VL mixed up)
                    if(!doJitunrolls
                            // accept this vel unroll, just to check new vel form
                            && strncmp(name,"unroll_cjitConvFwd6vel",22)!=0
                            && strncmp(name,"unroll",6)==0){
                        if(r==0){
                            printf(" skipping unrolls: %s\n",name);
                            fflush(stdout);
                        }
                        continue;
                    }

                    cacheKiller();
                    // CjitConvFwd1 is a "default" impl, so no need to check an _ok functiion
                    // eventually will need to return the syms for a vednnConvolutionLists.h entry
                    // so that we can check ok? and rtok? functions, before invoking the
                    // actual function XXX
                    vednnConvForward_t impl = (vednnConvForward_t)ptr;
                    // Q: Did we already call _ok and _rtok functions?
                    c[0] = __cycle();
                    FTRACE_BEGIN(name);
                    rv = (*impl)(CONVX_FWD_ORDER(FWD_PARMS, FWD_DATA));
                    c[1] = __cycle();
                    FTRACE_END(name);
                    if (r == 0){
                        printf(" mb%dg%d_ic%doc%d",
                                (int)(pConv->pParamIn->batch),
                                (int)(pConv->pParamConv->group),
                                (int)(pConv->pParamIn->channel),
                                (int)(pConv->pParamOut->channel));
                        //pConv->cycle += c[1] - c[0]; // this is reserved for the libvednn time
                        double diff = !doRef? -13.0
                            : (rv != VEDNN_SUCCESS)? -1.0
                            : diffData(pConv->pParamOut, pConv->pDataOut, pConv->pBufRef);
                        if(diff > max_diff) max_diff = diff;
                        printf("%s DIFF = %f ~%f ms %s",(rv==VEDNN_SUCCESS?"  OK":" BAD")
                                , diff, (c[1]-c[0])*(1.e3/HZ), name);
                        printf("\n");
                        fflush(stdout);
                        // NEW detailed stats...
#if 0
                        vtd.emplace_back(t, pConv->region, (size_t)cj/*idx*/, cjs[cj].sym, 2/*JIT*/);
#else
                        //TestData tmp(t, pConv->region, (size_t)cj/*idx*/, cjs[cj].sym, 2/*JIT*/ /*, default=NULL*/ );
                        TestData tmp = TestData(t, pConv->region, (size_t)cj/*idx*/,
                                cjs[cj].sym, //vednnConvForwardList[idx].shortname,
                                2/*JIT*/,
                                pNw_param_cstr/*test-wide descr*/ );
                        vtd.push_back(tmp);
#endif
                        assert(vtd.size() == njit+1U);
                        TestData& td = vtd.at(njit);
                        td.diff = diff;
                        td.ops = pConv->ops;
                    }
                    printf(" njit=%d status=%d time %d ms",njit,rv,
                            (1.0e3/HZ)*(c[1]-c[0])); fflush(stdout);
                    TestData& td = vtd.at(njit);
                    td.sum_times += c[1]-c[0];
                    td.reps = r+1;
                    ++njit;
                }
                if(r==0 && njit==0){
                    // XXX better, record the "duplicate test" parent tag
                    printf(" t %d duplicate\n",(int)t);
                }
            }
            if(reps>1) printf("\n");
            printf("Good, finished %4u doJit tests, %d reps, max DIFF = %f\n",
                    (unsigned)njit, (int)reps, max_diff );
            fflush(stdout);
            tdRepo.append(vtd,1/*verbose*/); // print and append
        }// end JIT impls

#undef FWD_DATA
#undef FWD_PARMS
#undef FWDB_DATA
#undef FWDB_PARMS
    }
    cout<<"end tdRepo.size() = "<<tdRepo.size()
        <<" testForward vednn thr="<<vednn_get_num_threads()<<"="<<omp_get_num_threads()<<"/"<<omp_get_max_threads()
        <<endl;
    printf("\n\n libvednn and JIT impls combined...\n");
    //tdRepo.print();
    // Newer API (header lines with layer structure (short string)
    print_test_data(tdRepo, pNetwork, nEntry);
    tdRepo.wins();  // algorithm dominance matrix

    cjitSyms_free(allsyms); allsyms=nullptr;
}

void testBackwardData(struct param *pNetwork, int nEntry, double HZ, int flagCSV, int reps, JitconvDo const& doOpt)
{
    // extract common test run options
    int const doRef = doOpt.doRef;
    int const doStd = doOpt.doStd;
    int const doItr = doOpt.doItr;
    //int const doJit = doOpt.doJit;
    //int const doJitunrolls = doOpt.doJitunrolls;
    filterLayout_t const filter_layout = doOpt.filter_layout;

    int t;
    typedef struct testconvBackwardData conv;
    TestDataRepo tdRepo(HZ);
    CacheKiller cacheKiller;

    // TODO cache-killer routine between timing measurements
    for (t=0; t<nEntry; ++t) {
        cout<<"+++ entry "<<t<<" of nEntry="<<nEntry;
        struct param *pNw = &pNetwork[t];
        OneConvBackwardData wrk( pNw, filter_layout, 1/*verbose*/ );
        testconvBackwardData * const pConv = &wrk.conv;
        char pNw_param_cstr[100];
        param_cstr_short(pNw,pNw_param_cstr,100);
        cout<<" "<<pNw_param_cstr<<endl;

        if(doRef){
            cout<<"+++ doRef"<<endl;
            wrk.doRef();
            TestData td( t, pConv->ref_region, (size_t)0/*impl_idx*/,
                    "gemm-Ref", 3/*doRef*/, pNw_param_cstr/*test-wide descr*/);
            double sum_time = 0.0;
            double max_diff = 0.0;
            unsigned long long c[2];
            for(int r=0; r < reps; ++r){
                cacheKiller();
                c[0] = __cycle();
                wrk.doRef();
                c[1] = __cycle();
                // now we have pConv->cycle, ++pConv->reps and output calculated

                if (flagCSV) dumpParamCSV_title();
                if (flagCSV) dumpParamCSV(pNw,"Bkw","D");
                else         dumpParam   (pNw,"Bkw","D");
                // diffData doesn't work with calc and ref pointers identical
                double diff = 0.0;
                max_diff = diff > max_diff? diff: max_diff;
                double f = 1.0e3 / HZ;
                double time = (c[1]-c[0]) * f / pConv->reps; // average ms
                double mintime = pConv->mincycle *  f;
                double maxtime = pConv->maxcycle *  f;
                sum_time += time;
                printf((flagCSV?", %f, %f, %f, %f"
                            :" \tTIME = %8.3f msec [%8.3f,%8.3f] DIFF = %f"),
                        time, mintime, maxtime, diff);
                printf("\n");
                // record TestData stats
                td.diff = 0.0;
                td.ops = pConv->ops;
                td.sum_times += c[1]-c[0];
                td.reps = r+1;
                printf("%4u tests. avg TIME = %9.3f msec. max DIFF = %f\n",
                        (unsigned)nEntry, sum_time/nEntry, max_diff);
                fflush(stdout);
            }
            tdRepo.append(td,1/*verbose*/);
            printf("Good, finished doRef, %d reps\n", (int)reps);
            printf(" max DIFF = %f\n",max_diff);
        }
        if(doStd){
            printf("doStd pConv->region is %s   layer %s\n", pConv->region, pNw_param_cstr);
            TestData td( t, pConv->region, (size_t)0/*impl_idx*/, "libvednn-std",
                    0/*doStd*/, pNw_param_cstr/*test-wide descr*/ );
            // run once and discard
            // now we have pConv->cycle, ++pConv->reps and output calculated
            testconvBackwardData_vednncalcs( pConv, 1 ); // set up pConv for calc
            pConv->cycle = 0;

            double sum_time = 0.0;
            double max_diff = 0.0;
            for(int r=0; r < reps; ++r){
                cacheKiller();
                testconvBackwardData_vednncalcs( pConv, 1 ); // set up pConv for calc

                if (flagCSV) dumpParamCSV_title();
                if (flagCSV) dumpParamCSV(pNw,"Fwd","D");
                else         dumpParam   (pNw,"Fwd","D");

                double diff = !doRef? -13.0
                    //: (rv != VEDNN_SUCCESS)? -1.0 // *_vednncalcs dos ERROR_EXIT if trouble.
                    : diffData(pConv->pParamGradIn, pConv->pDataGradIn, pConv->pBufRef);
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

                td.diff = diff;
                td.ops = pConv->ops;
                td.sum_times = pConv->cycle;
                td.reps = r+1;
                printf("%4u tests. avg TIME = %9.3f msec. max DIFF = %f\n",
                        (unsigned)nEntry, sum_time/nEntry, max_diff);
                fflush(stdout);
            }
            tdRepo.append(td,1/*verbose*/);
        }
        if(doItr){
            size_t const maxImpls = 64;
            double max_diff = 0.0;
            unsigned long long c[2];
            unsigned long long sum_times[maxImpls];
            unsigned rep_times[maxImpls];
            for(int imp=0; imp<maxImpls; ++imp) {sum_times[imp] = 0ULL; rep_times[imp] = 0U;}

#define BKWD_PARMS \
            pConv->pParamGradIn, pConv->pParamKernel, pConv->pParamGradOut, \
            pConv->pParamConv, VEDNN_CONV_ALGORITHM_DIRECT
#define BKWD_DATA \
            pConv->pDataGradIn,  pConv->pDataKernel,  pConv->pDataGradOut
            std::vector<TestData> vtd;
            for(int r=0; r<reps; ++r){
                printf(" doItr-rep-%d",r); fflush(stdout);
                vednnError_t rv;
                char name[80];
                int iter_cnt=0;
                for (vednnConvBackwardDataImpls * iter = vednnConvBackwardData_Begin(BKWD_PARMS);
                        iter->okfn != NULL;
                        iter = vednnConvBackwardData_Next(iter, BKWD_PARMS))
                {
                    snprintf(name,80,"%s:%d:%s",pConv->region,iter_cnt,iter->shortname);
                    printf(" %s...",name); fflush(stdout);
                    cacheKiller();

                    // TODO realNext (check for _rtok to run the impl, as below)
                    FTRACE_BEGIN("realNext");
                    //printf(" B"); fflush(stdout);
                    // _Next gives us parms-based possible impl
                    // _realNext if [runtime] ptr alignment precludes impl, then skip
                    vednnConvBackwardDataImpls *actual = vednnConvBackwardData_realNext(
                            iter, BKWD_PARMS, BKWD_DATA );
                    FTRACE_END("realNext");

                    int idx = 0;
                    if ( actual != NULL ) {
                        if( actual == iter ){ // almost always...
                            snprintf(name,80,"%s:%d:%s",pConv->region,iter_cnt,iter->shortname);
                        }else{
                            snprintf(name,80,"%s:%d:%s-->%s",pConv->region,iter_cnt,iter->shortname,actual->shortname);
                        }
                        //printf(" C%s",name); fflush(stdout);
                        c[0] = __cycle();
                        FTRACE_BEGIN(name);
                        // NB:             CONVX_.....order for low-level call
                        //rv = (*actual->impl)(CONVX_BKWD_ORDER(BKWD_PARMS, BKWD_DATA));
                        // now we also want omp support, so we call via _Run, not via (*actual->impl)
                        // Note: _Run *also* calls realNext to skip over inappropriate impls (_rtok)
                        //  but we want to first get the impl to set 'name' string [above]
                        vednnConvBackwardData_out_t const out = vednnConvBackwardData_Run( actual, BKWD_PARMS, BKWD_DATA );
                        c[1] = __cycle();
                        FTRACE_END(name);
                        idx = (actual - vednnConvBackwardDataList);
                        printf(" idx=%d out.status=%d time %d ms",idx,out.status,
                                (1.0e3/HZ)*(c[1]-c[0])); fflush(stdout);
                        if( idx>=0 && idx < maxImpls ) {sum_times[idx] += c[1] - c[0]; ++rep_times[idx];}
                        assert( out.actual == actual );
                        rv = out.status;
                    }else{
                        printf(" not run\n");
                        continue;
                    }

                    if (r == 0){
                        if( pConv->pParamGradIn->batch == 1 ) {
                            printf(" batch 1 group=%d gradInChannel=%d gradOutChannel=%d",
                                    (int)(pConv->pParamConv->group),
                                    (int)(pConv->pParamGradIn->channel),
                                    (int)(pConv->pParamGradOut->channel));
                        }
                        //pConv->cycle += c[1] - c[0]; // this is reserved for the libvednn time
                        //printf(" %s %llu cycles", name, c[1]-c[0]);
                        double diff = !doRef? -13.0
                            : (rv != VEDNN_SUCCESS)? -1.0
                            : diffData(pConv->pParamGradIn, pConv->pDataGradIn, pConv->pBufRef);
                        max_diff = diff > max_diff? diff: max_diff;
                        printf("%s %8.3f ms DIFF = %f %s\n",(rv==VEDNN_SUCCESS?" OK":"BAD")
                                , (1.0e3/HZ)*(c[1]-c[0]), diff ,name);
                        fflush(stdout);

                        // NEW detailed stats...
                        idx = (actual - vednnConvBackwardDataList);
#if 0
                        vtd.emplace_back(t, pConv->region, (size_t)idx,
                                vednnConvBackwardDataList[idx].shortname,
                                1/*doItr*/, pNw_param_cstr/*test-wide descr*/ );
#else
                        TestData tmp(t, pConv->region, (size_t)idx,
                                vednnConvBackwardDataList[idx].shortname,
                                1/*doItr*/, pNw_param_cstr/*test-wide descr*/ );
                        vtd.push_back(tmp);
#endif
                        assert(vtd.size() == iter_cnt+1U);
                        TestData& td = vtd.at(iter_cnt);
                        td.diff = diff;
                        td.ops = pConv->ops;
                    }
                    TestData& td = vtd.at(iter_cnt);
                    td.sum_times += c[1]-c[0];
                    td.reps = r+1;

                    if(++iter_cnt>=100) ERROR_EXIT("run-away iter over ConvBackwardData impls?");
                }
                if (rv != VEDNN_SUCCESS) ERROR_EXIT("convolution() failed.");

                FTRACE_END("all convolution");
            }
#undef BKWD_PARMS
#undef BKWD_DATA
            tdRepo.append(vtd,1/*verbose*/);
            printf("Good, finished doItr, %d reps\n", (int)reps);
            printf(" max DIFF = %f\n",max_diff);
        }
    }
#if 0
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
#endif
}

void testBackwardFilter(struct param *pNetwork, int nEntry, double HZ, int flagCSV, int reps, JitconvDo const& doOpt)
{
    // extract common test run options -- this code has not been revised!
    //int const doRef = doOpt.doRef;
    //int const doStd = doOpt.doStd;
    //int const doItr = doOpt.doItr;
    //int const doJit = doOpt.doJit;
    //int const doJitunrolls = doOpt.doJitunrolls;
    filterLayout_t const filter_layout = doOpt.filter_layout;

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
        testconvBackwardFilter_alloc( pConv, pNw , filter_layout);
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
    char const* pName;
    filterLayout_t   layouttype;
} filterLayout[] = {
    { "filter_nchw",    VEDNN_FILTER_LAYOUT_NCHW } ,
    { "filter_hwcn",    VEDNN_FILTER_LAYOUT_HWCN }
};

char const* default_parameter_file="mb27g1_ic3ih27iw270_oc47oh14ow135_kh3kw3_ph1pw1_sw2sh2_dw1dh1";
// oc47 ~ 47=0x2f : this many outChannelGroups will check many hand-unrolled code cases
static void help(){
    printf( "\njitconv [OPTIONS] [OVERRIDES]:"
            "\n   -p PATH   convolution parameter file"
            "\n             -- or --  a STRING like"
            "\n   -p mb27g1_ic3ih22iw22_oc100oh22ow22_kh5kw5_ph3pw3_sw1sh1_dw1dh1"
            "\n             where '_' are ignored."
            "\n      [ -p %s ]"
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
#if defined(_OPENMP)
            "\n             ** not compiled with OpenMP support **"
#endif
            "\n   -l LIB    **before** jit build, load LIB dll.  tests with all syms in LIB"
            "\n             won't be remade. No error if LIB is not loadable. libmegajit.so ?"
            "\n             TODO: use a newer .a to first update the .so, if possible."
            "\n   -S SUBDIR Jit build subdirectory [tmp_cjitConv]. Each run assumes exclusive"
            "\n             access to SUBDIR.  Concurrent tests should use -S"
            "\n             Jit build creates SUBDIR/libcjitConv.so"
            "\n   -w WLIB   TODO: **after** jit build, update WLIB.a with SUBDIR/tmp_cjitConv.a symbols"
            "\n   -k        enable cacheKiller before every timed call"
            "\n   -f STRING filter layout: [filter_nchw] filter_hwcn"
            "\n               Note: current JIT impls assume filter_nchw (iohw) kernel data"
            "\n   -o CHARS  doRef,doStd,doItr,doJit,doJitunroll options"
            "\n               default -o RSIJU (run all types of impls)"
            //"\n   -I        name1[,name2,...] list of allowed impl names"
            //"\n   -V        print version, standard/jit impl names ??
            "\n OVERRIDES: non-option as TAG[=]NUMBER...  Examples: mb=64 kh3kw3 ih32iw=32"
            "\n   'jitconv -M parms.txt mb32 will change every test in parms.txt to use minibatch 32"
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
/** return next int value, skipping initial [ =], returning CONV_OVERRIDE on error */
static int ovrint(char const* arg){
    int ret = CONV_OVERRIDE;
    if(arg){
        printf(" ovrint(arg=%s)\n", arg);
        // skip whitespace or '=' or '_'
        while( *arg != '\0'){
            if( isspace(*arg) || *arg=='=' || *arg=='_' )
                ++arg;
            else
                break;
        }
        // if [-+0-9] ... try atoi
        if( isdigit(*arg) || *arg=='+' || *arg=='-' ){
            ret = atoi(arg);
            // negative int gets no special treatment.
            // malformed gets value zero (no error)
        }   // else ... return value remains CONV_OVERRIDE
    }
    cout<<" ovrint-->"<<ret<<endl;
    return ret;
}
/** parse cmd line overrides like kh7 kw7 ic16.
 * <em>other side effect</em>: If possible override, set ovr->pName to "Y".
 * return number of potential overrides process, or -1 on error
 */
static int/*err?*/ check_override( struct param * ovr, char const* arg ){
    int ret = 0;
    // syntax <1 or 2 character code>[=]<int value>
    //  Codes: b t ic ih iw oc oh ow kh kw sh sw ph pw dh dw
    //  values: negative values probably make no sense, but will be accepted?
    // codes can be concatenated (embeded whitespace, =, or _ ignored)
    // but terminal 'n' will NEVER be handled correctly.
    while(arg[0]!= '\0'){
        cout<<" remaining arg = '"<<arg<<"' int(arg[0])="<<int(arg[0])<<endl;
        if(0) ;
        else if(strncmp(arg,"b" ,1)==0) {arg+=1; ovr->batchNum       = ovrint(arg); ++ret;}
        else if(strncmp(arg,"mb",2)==0) {arg+=2; ovr->batchNum       = ovrint(arg); ++ret;}
        else if(strncmp(arg,"g" ,1)==0) {arg+=1; ovr->group          = ovrint(arg); ++ret;}
        else if(strncmp(arg,"ic",2)==0) {arg+=2; ovr->inChannel      = ovrint(arg); ++ret;}
        else if(strncmp(arg,"ih",2)==0) {arg+=2; ovr->inHeight       = ovrint(arg); ++ret;}
        else if(strncmp(arg,"iw",2)==0) {arg+=2; ovr->inWidth        = ovrint(arg); ++ret;}
        else if(strncmp(arg,"oc",2)==0) {arg+=2; ovr->outChannel     = ovrint(arg); ++ret;}
        else if(strncmp(arg,"oh",2)==0) {arg+=2; ovr->outHeight      = ovrint(arg); ++ret;}
        else if(strncmp(arg,"ow",2)==0) {arg+=2; ovr->outWidth       = ovrint(arg); ++ret;}
        else if(strncmp(arg,"kh",2)==0) {arg+=2; ovr->kernHeight     = ovrint(arg); ++ret;}
        else if(strncmp(arg,"kw",2)==0) {arg+=2; ovr->kernWidth      = ovrint(arg); ++ret;}
        else if(strncmp(arg,"sh",2)==0) {arg+=2; ovr->strideHeight   = ovrint(arg); ++ret;}
        else if(strncmp(arg,"sw",2)==0) {arg+=2; ovr->strideWidth    = ovrint(arg); ++ret;}
        else if(strncmp(arg,"ph",2)==0) {arg+=2; ovr->padHeight      = ovrint(arg); ++ret;}
        else if(strncmp(arg,"pw",2)==0) {arg+=2; ovr->padWidth       = ovrint(arg); ++ret;}
        else if(strncmp(arg,"dh",2)==0) {arg+=2; ovr->dilationHeight = ovrint(arg); ++ret;}
        else if(strncmp(arg,"dw",2)==0) {arg+=2; ovr->dilationWidth  = ovrint(arg); ++ret;}
        else{ // error?
            cout<<" oops, not an override: "<<arg<<endl;
            ret = -1;
            break; // give up [includes parm string with terminal 'n' (name) field
        }
        arg += strspn(arg,"+-0123456789"); // skip 'int' chars and keep going
        //if(ret > 100) { cout<<"program error?"<<endl; break; }
    }
    if(ret){
        cout<<" found "<<ret<<" overrides"<<endl;
        char* name = (char*)&ovr->pName[0];
        name[0] = 'Y';
        name[1] = '\0';
    }
    return ret;
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
    JitconvDo doOpt;

    const rlim_t kStackSize = 16 * 1024 * 1024;   // min stack size = 16 MB
    struct rlimit rl;
    int result;

    signal(SIGINT, set_control_c_flag);

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
    struct param overrides = {
        .batchNum       = CONV_OVERRIDE,
        .group          = CONV_OVERRIDE,
        .inChannel      = CONV_OVERRIDE,
        .inHeight       = CONV_OVERRIDE,
        .inWidth        = CONV_OVERRIDE,
        .outChannel     = CONV_OVERRIDE,
        .outHeight      = CONV_OVERRIDE,
        .outWidth       = CONV_OVERRIDE,
        .kernHeight     = CONV_OVERRIDE,
        .kernWidth      = CONV_OVERRIDE,
        .strideHeight   = CONV_OVERRIDE,
        .strideWidth    = CONV_OVERRIDE,
        .padHeight      = CONV_OVERRIDE,
        .padWidth       = CONV_OVERRIDE,
        .dilationHeight = CONV_OVERRIDE,
        .dilationWidth  = CONV_OVERRIDE,
        .pName          = "N" // set to "Y" if maybe found some overrides
    };
    vector<string> ldlibs; // load libraries [if problems, report and continue]
    while((opt = getopt(argc, argv, "p:M:CH:T:t:r:l:S:kf:o:")) != -1) {
        switch (opt) {
        case 'M':
            m_for_mkldnn = 'm'; // fall-through
        case 'p':
            snprintf(paramBuf,PARAMBUFSZ,"%s",optarg);
            pParamPath = &paramBuf[0];
            break;
        case 'r':
            reps = (int)atof(optarg); break;
        case 'C':
            flagCSV = 1; break;
        case 'H':
            HZ = atof(optarg); break;
        //case 'j': doJit = 0; break;
        //case 'J': doJit = 1; break;
        //case 'I': parse csv impl names allowed for doStd or doJit
        //case 'V': print version, standard/jit impl names
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
                    fprintf(stderr, "Invalid test type.\n  -T options:");
                    for (int t=0; t<sizeof(tests)/sizeof(tests[0]); ++t) {
                        fprintf(stderr, " %s", tests[t].pName);
                    }
                    fprintf(stderr,"\n");
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
        case 'o' :
            {
                int doRef = 0;
                int doStd = 0;
                int doItr = 0;
                int doJit = 0;
                int doJitunrolls = 0;
                int o_err = 0;
                for (int i=0; optarg[i] != '\0'; ++i){
                   char const o=toupper(optarg[i]);
                   if (o=='R') doRef=1;
                   else if (o=='S') doStd=1;
                   else if (o=='I') doItr=1;
                   else if (o=='J') doJit=1;
                   else if (o=='U') doJitunrolls=1;
                   else ++o_err;
                }
                if(o_err){
                    fprintf(stderr, "Invalid -o [R|S|I|J|U]... setting");
                    exit(1);
                }
                doOpt.doRef = doRef;;
                doOpt.doStd = doStd;
                doOpt.doItr = doItr;
                doOpt.doJit = doJit;
                doOpt.doJitunrolls = doJitunrolls;
            }
        case 't':
            threads = atof(optarg);
            if(threads > 16) threads = 16;
            if(threads < 0 ) threads = -1;
            break;
        case 'l':
            ldlibs.push_back(optarg); break;
        case 'S':
            jit_dir = optarg; break;
        case 'k':
            CacheKiller::enable = true; break;
        default: /* '?' */
            fprintf(stderr, "Unknown option.\n");
            help();
            exit(1);
        }
    }
    while(optind < argc){ // non-option arguments
        // 1. special parameter overrides
        // 2. REMOVED  else pretende it is test file or single test spec ?
        //    ** now you MUST use -M or -p option **
        cout<<" non-option: "<<argv[optind]<<endl;
        int nOvr = check_override( &overrides, argv[optind] );
        if(nOvr==0){
            // just pretend it is an mkl-dnn test file name, like '-p'
            snprintf(paramBuf,PARAMBUFSZ,"%s",argv[optind]);
            pParamPath = &paramBuf[0];
        }
        ++optind;
    }
    if (optind < argc) {
        fprintf(stderr, "Unexpected arguments\n");
        exit(1);
    }

    doOpt.filter_layout = filter_layout;
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
    if(jit_dir) printf("JIT SUBDIR               = %s\n",      jit_dir);
    printf(" setting params...\n"); fflush(stdout);

    if (!ldlibs.empty()){
        for(auto libname: ldlibs){
            cout<<" -l "<<libname;
            dlerror();
            void *libHandle;
            libHandle = dlopen(libname.c_str(),RTLD_LAZY|RTLD_LOCAL|RTLD_DEEPBIND);
            if(!libHandle){
                cout<<" failed: "<<dlerror()<<" [continuing]"<<endl;
            }else{
                cout<<" OK"<<endl;
            }
        }
    }

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

    // TODO option to skip mkConsistent (for even kernel size, pad value is ?)
    //
    // Convolution tests don't handle some illegal cases well
    //    (segfault or bus error)
    // Some acceptable configs report bad DIFF and also get massaged
    // to "exact-sized" output height and width.
    //
    // This allows you to use randomly-edited parameter files and
    // at least avoid segfaults/bus errors/GEMM illegal value messages.
    // (exact output height width can be tricky to guess correctly).
    //
    for(int t=0; t<nParams; ++t){
        //dumpParam(&pParams[t],"BEFORE","");
        printf("mkConsistent..%d\n",t);
        mkConsistentOverrides( &pParams[t], &overrides );
        //dumpParam(&pParams[t],"AFTER",""); // not nice (line too long)
        char buf[100];
        printf("AFTER: %s\n", param_cstr_short(&pParams[t],buf,100));
    }
    printf(" test parameters sanitized\n");
#if 0
    cout<<" overrides name = "<<overrides.pName<<endl;
    if(overrides.pName[0]=='Y'){
        printf(" applying overrides\n");
        apply_overrides( pParams, nParams, &overrides );
        for(int t=0; t<nParams; ++t){
            printf("mkConsistent (again)..%d\n",t);
            mkConsistent( &pParams[t] );
        }
    }
#endif

#ifdef USE_OPENMP
    int thrlo = 0, thrhi = 8;
    if(threads >= 0){
        thrlo = threads;
        thrhi = threads;
    }
    for(int thr=thrlo; thr<=thrhi; ++thr){
        printf(" omp_set_num_threads(%d)...\n", thr);
        omp_set_num_threads(thr);
        // New: vednnInit also sets __vednn_omp_num_threads...
        // IMPORTANT: **we** must set this here
        // (else set vednnConvolutionLists.c to use omp_get_max_threads() or so)
#if 0 
        cout<<" Original libvednn __vednn_omp_num_threads = "<<__vednn_omp_num_threads
            <<" will be set to "<<thr<<endl;
        __vednn_omp_num_threads = thr;
#else // equivalent (api addition...)
        // BETTER: provide a public function "vednn_set_num_threads"
        //         that does both (well a macro, to avoid fn call overhead)
        cout<<" Original libvednn vednn_get_num_threads = "
            <<vednn_get_num_threads()<<" will be set to "<<thr<<endl;
        vednn_set_num_threads(thr);
#endif
#else
        cout<<" libvednn using vednn_get_num_threads() = "
            <<vednn_get_num_threads()<<" (jitconv -t "<<threads<<" ignored)"
            <<endl;
#endif
        switch(testtype) {
        case CONV_TEST_FORWARD :
            testForward(pParams, nParams, HZ, 0, flagCSV, reps, doOpt);
            break ;
        case CONV_TEST_FORWARD_ADDBIAS :
            testForward(pParams, nParams, HZ, 1, flagCSV, reps, doOpt);
            break ;
        case CONV_TEST_BACKWARD_DATA :
            testBackwardData(pParams, nParams, HZ, flagCSV, reps, doOpt);
            break ;
        case CONV_TEST_BACKWARD_FILTER :
            testBackwardFilter(pParams, nParams, HZ, flagCSV, reps, doOpt);
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

// vim: et ts=4 sw=4 cindent cino=l1,\:0,=s,N-s,g-2,h2 syntax=cpp.doxygen
