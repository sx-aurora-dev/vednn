#include "vednnJitUtil.h"
#include <assert.h>
#include <string.h>

extern "C" { //}

char const* vednnSymConvolutionForwardSuffix( VEDNN_PARAMS_CONV_FORWARD ){
    char const* ret = nullptr;
    // 1. convert to vconv [gen-dnn] structure? or vednn struct?
    //    see conv_test_param.h, and copy useful functions to vednnJitUtil.h
    // 2. construct string [gen-dnn dilation convention=zero-based]
    return ret;
}

}//"C"
// vim: et ts=4 sw=4 cindent cino=l1,)0,u0,W2,\:0,=2s,N-s,g-2,h2 syntax=cpp.doxygen
