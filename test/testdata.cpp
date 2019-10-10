
#include "testdata.hpp"
#include <cstring>
#include <algorithm>		// max
#include <iostream>

using namespace std;

namespace {
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
/** helper for sort-by-increasing-time */
struct TestTime {
    int ntd;        ///< index into struct TestData[]
    double time;    ///< avg time of some TestData[ntd]
};
}//anon::

static void init_TestData( struct TestData* test_data,
        int const test, char const* test_name, /*char const* descr,*/
        int const impl_idx, char const* impl_name, int impl_type ){
    // use setDescr to fill in this-descr.  JIT impls auto-fill with long-name
    char const* name;
    test_data->test = test;
    name = test_name;
    {
        int namelen = strlen(name);
        //char *cutname = strstr(name,"_mb");
        //if(cutname) namelen = cutname-name;
        if(namelen > MAX_TEST_NAME) namelen=MAX_TEST_NAME;
        strncpy(test_data->test_name, name, namelen);
        if(namelen){
            namelen = min(namelen,MAX_TEST_NAME-1);
            test_data->test_name[namelen] = '\0';
            //         ^^^^^^^^^
        }
    }
    test_data->impl_idx = (size_t)impl_idx;
    name = impl_name;
    {
        int namelen = strlen(name);
        if(namelen > MAX_TEST_NAME) namelen=MAX_TEST_NAME;
#if 0
        // no longer useful to duplicate impl_name into descr ?
        if(impl_type == 2/*JIT*/){
            strncpy(test_data->descr, name, namelen);
            test_data->descr[namelen] = '\0'; // full length
            //         ^^^^^
        }else{
            test_data->descr[0] = '\0';
        }
#else
        test_data->descr[0] = '\0';
#endif
        char *cutname = strstr((char*)name,"_mb");
        if(cutname) namelen = cutname-name;
        if(namelen > MAX_TEST_NAME) namelen=MAX_TEST_NAME;
        strncpy(test_data->impl_name, name, namelen);
        if(namelen){
            namelen = min(namelen,MAX_TEST_NAME-1);
            test_data->impl_name[namelen] = '\0';
            //         ^^^^^^^^^
        }
    }
    test_data->sum_times = 0ULL;
    test_data->ops = 0ULL;
    test_data->diff = -1.0;
    test_data->impl_type = impl_type;
}

TestData::TestData( int const test, char const* test_name,
        int const impl_idx, char const* impl_name, int impl_type,
        char const* descr/*=nullptr*/)
{
    init_TestData( this, test, test_name, impl_idx, impl_name, impl_type );
    if(descr) this->setDescr(descr);
    else this->descr[0]='\0';
    cout<<"test_name "<<this->test_name<<endl;
    cout<<"impl_name "<<this->impl_name<<endl;
    cout<<"descr "<<this->descr<<endl;
}
void TestData::setDescr(char const* description){ ///< parm string, comment
    if(description!=NULL){
        strncpy(descr, description, MAX_TEST_NAME);
        descr[MAX_TEST_NAME-1]='\0';
    }
}
void TestData::appendDescr(char const* description){ ///< parm string, comment
    int i=0;
    while(i<MAX_TEST_NAME && descr[i]!='\0')
        ++i;
    if(description!=NULL){
        strncpy(&descr[i], description, MAX_TEST_NAME - i);
        descr[MAX_TEST_NAME-1]='\0';
    }
}

static char testData_impl_char(int const impl_type){
    char ret;
    switch(impl_type){
    case(0): ret = '*'; break; // doStd
    case(1): ret = 'I'; break; // doItr
    case(2): ret = 'J'; break; // doJit
    case(3): ret = 'R'; break; // doRef
    default: ret = '?';
    }
    return ret;
}
static struct ImplNameIdx mk_impl_name_idx(struct TestData const* test_data, size_t const a, size_t const z){
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
static void print_wins(struct TestData const* test_data, size_t const a, size_t const z, double const HZ){
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
    // a vs b avg speedup factor logarithmic with +ve = "faster"
    printf("\nRelSpeed is (avg log_10(t_i/t_j)) * 100/log_10(1.50) capped to [-99 999]\n"
            "           for printing, so 0 is same speed,"
            "           +-100 is 1.5x time diff (+/-50%), and +-10 is 4.1%% speedup\n"
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
                double relSpeed = (double)speedup[a*ni+b] / nbat; // avg log10(speed_ratio)
                // scale so +-100 is 50% speedup/slowdown.  +-10 ~ 41% speedup
                // speedup = 10 ^ ( reported_number / (100/log(1.5)) )
                //         = 10 ^ ( reported_number / 567.9 )
                // reported number  speedup
                //      10          1.041  (4.1% speedup)
                //      20          1.084
                //      30          1.129
                //      40          1.176
                //      50          1.224
                //      60          1.275
                //      70          1.328
                //      80          1.383
                //      90          1.440
                //      100         1.500 (50%)
                //      500         7.59
                //      999         57.4
                double const scale = 100.0 / log10(1.50);
                relSpeed *= scale;
                // round-to-nearest and bound to [-99,+99]
                int const nice = min(max((int)(relSpeed+0.5), -99), +999);
                printf("%4d",nice);
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


static int cmp_TestTime( void const* a, void const*b ){
    return ((struct TestTime const*)a)->time
        >=  ((struct TestTime const*)b)->time;
}
void print_test_data_single( struct TestData const* test_data, int const t,
        int const a, int const z, double const HZ,
        char const* header/*=nullptr*/ ){
    // for spreadsheet, better to print header in every line
    //if(header && header[0]!='\0'){
    //    printf("%s\n", header);
    //}
    vector<TestTime> ttime;
    ttime.reserve(256);
    for(size_t ntd=a; ntd<z; ++ntd){
        struct TestData const* td = &test_data[ntd];
        if( td->test != t )
            continue;
        double const f = 1.0e3 / HZ;
        double const time = td->sum_times * f / td->reps; // average ms
        ttime.push_back({(int)ntd,time});
    }
    // sort ttime[0..nttime)
    qsort( &ttime[0], ttime.size(), sizeof(struct TestTime), cmp_TestTime );
    char const* fastest="**";
    for(size_t ntt=0; ntt<ttime.size(); ++ntt){
        struct TestData const* td = &test_data[ttime[ntt].ntd];
        double const f = 1.0e3 / HZ;
        double const time = td->sum_times * f / td->reps; // average ms
        // Gop/s = Mop/ms
        double const gops = (td->ops>0? td->ops*1.0e-6 / time: 0.0);
        printf( "%c %25s %s %4ux %9.3f ms ~%.4f %6.2fG %s",
                testData_impl_char(td->impl_type),
                td->impl_name,
                fastest, td->reps, time, (double)td->diff,
                gops, td->test_name);
        if(header    && header[0]   !='\0') printf(" %s", header);
        // NEW: descr no holds a test-wide comment,
        //      like param_cstr_short or test duplicate info.
        //if(td->descr && td->descr[0]!='\0') printf(" %s", td->descr);
        printf("\n");
        fastest = " |";
    }
}

void print_test_data( struct TestData const* test_data, int const a, int const z,
        double const HZ, int const v/*=1*/){
    assert( a>=0 );
    assert( z>=a );
    if( a >= z ) return;
    int minTest = 1<<30;
    int maxTest = 0;   //
    for(int i=a; i<z; ++i){
        if(test_data[i].test > maxTest) maxTest = test_data[i].test;
        if(test_data[i].test < minTest) minTest = test_data[i].test;
    }
    if(v){
        printf("*** print_test_data[%d..%d) covers tests %d..%d",a,z,minTest,maxTest);
        if(minTest==maxTest && test_data[a].descr && test_data[a].descr[0] != '\0')
            printf(" %s",test_data[a].descr);
        printf("\n");
    }
    for(int t=minTest; t<=maxTest; ++t){
        print_test_data_single(test_data, t, a, z, HZ);
    }
}
/** print all (nEntry) tests (impls) in \c test_data with nice test label line */
void print_test_data( struct TestDataRepo const& tdRepo,
        struct param const *pNetwork, int nEntry ){
    char buf[100];
    //char header[150];
    for(int t=0; t<nEntry; ++t){
        struct param const *pNw = &pNetwork[t];
        param_cstr_short(pNw,buf,100);
        //snprintf(header,150,"Layer %-30s %s_n%s", pNw->pName, buf, pNw->pName);
        //print_test_data_single(&tdRepo[0],t, 0,tdRepo.size(), tdRepo.hertz,header);
        print_test_data_single(&tdRepo[0],t, 0,tdRepo.size(), tdRepo.hertz,buf);
    }
}
void TestDataRepo::append(std::vector<TestData> const& vtd, int const verbose/*=1*/){
    if(verbose){ // print before saving
        print_test_data(vtd.data(), 0, vtd.size(), hertz);
    }
    size_t const sz0 = this->size();
    for(auto const& td: vtd){
        this->emplace_back(td);
    }
    size_t const sz1 = this->size();
    cout<<" saved testDataRepo["<<sz0<<".."<<sz1<<")"<<endl;
}
void TestDataRepo::append(TestData const& td, int const verbose/*=1*/){
    if(verbose){ // print before saving
        print_test_data(&td, 0, 1, hertz);
    }
    size_t const sz0 = this->size();
    this->emplace_back(td);
    size_t const sz1 = this->size();
    cout<<" saved testDataRepo["<<sz0<<".."<<sz1<<")"<<endl;
}
void TestDataRepo::print() const {
    print_test_data(this->data(), 0, this->size(), hertz);
}
void TestDataRepo::wins() const {
    print_wins(this->data(), 0, this->size(), hertz); // algorithm dominance matrix
}
// vim: et ts=4 sw=4 cindent cino=^0,=0,l0,\:0,N-s syntax=cpp.doxygen
