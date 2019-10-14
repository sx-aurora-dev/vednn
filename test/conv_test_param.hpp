#ifndef CONV_TEST_PARAM_HPP
#define CONV_TEST_PARAM_HPP

#include "conv_test_param.h"

/// \group Easy ref calcs
//@{
struct OneConvForward;
struct OneConvBackwardData;
struct OneConvBackwardFilter;
//@}

struct OneConvForward final {
    testconvForward conv;

    OneConvForward( struct param const *pNetwork, int const flagBias, int const verbose );
    ~OneConvForward();
    // not req'd: void clobber(); ///< write garbage into output tensor [NEW]
    void doRef(); ///< run Reference Convolution
    int verbose() const {return v;}
    void verbose(int const verbose) {v=verbose;}

  private:
    int const flagBias;
    int v;
    bool haveRef;
};

struct OneConvBackwardData final {
    testconvBackwardData conv;

    OneConvBackwardData( struct param const *pNetwork, int const verbose=1 );
    ~OneConvBackwardData();
    void doRef(); ///< run Reference Convolution
    int verbose() const {return v;}
    void verbose(int const verbose) {v=verbose;}

  private:
    int v;
    bool haveRef;
};

struct OneConvBackwardFilter final {
    testconvBackwardFilter conv;

    OneConvBackwardFilter( struct param const *pNetwork, int const verbose=1 );
    ~OneConvBackwardFilter();
    void doRef(); ///< run Reference Convolution
    int verbose() const {return v;}
    void verbose(int const verbose) {v=verbose;}

    private:
    int v;
    bool haveRef;
};

//
// ------------------ impls --------------------------
//

inline OneConvForward::OneConvForward( struct param const *pNetwork, int const flagBias, int const verbose)
    : conv(), flagBias(flagBias), v(verbose), haveRef(false)
{
    assert(pNetwork);
    testconvForward_init( &conv );
    if(strlen(pNetwork->pName)) strncpy(conv.region, pNetwork->pName, 128);
    else snprintf(conv.region, 128, "test:Fwd%s", (flagBias?"B":""));
    snprintf(conv.ref_region, 128, "<gemm:Fwd%s>%s", (flagBias?"B":""), conv.region);
    testconvForward_alloc( &conv, pNetwork, flagBias );
    testconvForward_randomData( &conv, flagBias );
    if(v) testconvForward_dumpParms( &conv, flagBias );
}
inline OneConvForward::~OneConvForward(){
    testconvForward_free(&conv,flagBias);
}
inline void OneConvForward::doRef(){ // run Reference Convolution
    if(v) printf("doRef...\n");
    testconvForward_refcalcs(&conv,1);
    haveRef = true;
}
//inline void OneConvForward::clobber(){ // run Reference Convolution
//    if(v) printf("OneConvForward[doRef]::clobber...\n");
//    switch(getTensorDataType(conv.pParamIn)){
//    case DTYPE_FLOAT:
//        for(size_t i=0U; i<getTensorSize(conv.pParamOut); ++i)
//            conv.pDataOut[i] = 0.1313f;
//        break;
//    default:
//        ERROR_EXIT("Unknown dataType_t");
//        break;
//    }
//    haveRef = true;
//}

inline OneConvBackwardData::OneConvBackwardData( struct param const *pNetwork, int const verbose/*=1*/ )
    : conv(), v(verbose), haveRef(false)
{
    assert(pNetwork);
    testconvBackwardData_init( &conv );
    if(strlen(pNetwork->pName)) strncpy(conv.region, pNetwork->pName, 128);
    else snprintf(conv.region, 128, "test:BkwD");
    snprintf(conv.ref_region, 128, "<gemm:BkwD>%s", conv.region);
    testconvBackwardData_alloc( &conv, pNetwork );
    testconvBackwardData_randomData( &conv );
    if(v) testconvBackwardData_dumpParms( &conv );
}
inline OneConvBackwardData::~OneConvBackwardData(){
    testconvBackwardData_free(&conv);
}
inline void OneConvBackwardData::doRef(){ // run Reference Convolution
    if(v) printf("doRef...\n");
    testconvBackwardData_refcalcs(&conv,1);
    haveRef = true;
}

inline OneConvBackwardFilter::OneConvBackwardFilter( struct param const *pNetwork, int const verbose/*=1*/ )
    : conv(), v(verbose), haveRef(false)
{
    assert(pNetwork);
    testconvBackwardFilter_init( &conv );
    if(strlen(pNetwork->pName)) strncpy(conv.region, pNetwork->pName, 128);
    else snprintf(conv.region, 128, "test:BkwF");
    snprintf(conv.ref_region, 128, "<gemm:BkwF>%s", conv.region);
    testconvBackwardFilter_alloc( &conv, pNetwork );
    testconvBackwardFilter_randomData( &conv );
    if(v) testconvBackwardFilter_dumpParms( &conv );
}
inline OneConvBackwardFilter::~OneConvBackwardFilter(){
    testconvBackwardFilter_free(&conv);
}
inline void OneConvBackwardFilter::doRef(){ // run Reference Convolution
    if(v) printf("doRef...\n");
    testconvBackwardFilter_refcalcs(&conv,1);
    haveRef = true;
}

// vim: et ts=4 sw=4 cindent cino=^0,=0,l1,N-s,\:0,=s syntax=cpp.doxygen
#endif // CONV_TEST_PARAM_HPP
