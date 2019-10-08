#include "conv_test_param.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>     // drand48
#include <string.h>     // memset
#include <math.h>       // sqrt
#include <stdint.h>

#ifdef __cplusplus
extern "C" { //}
#endif //C++

void
testconvForward_vednncalcs( struct testconvForward *pConvArray, int const nEntry ){
    vednnError_t rv = VEDNN_SUCCESS;

    for (int i=0; i<nEntry; i++) {
        struct testconvForward *pConv = &pConvArray[i];
        int const flagBias = (pConv->pDataBias != NULL);
#ifdef FTRACE
        FTRACE_IF(char const* all_region = (flagBias? "all FwdB convolutions": "all Fwd convolutions"));
        printf("all_region = %s\ndef_region = %s\n",all_region,pConv->region);
#endif
        FTRACE_BEGIN(all_region);

        unsigned long long c[2];
        c[0] = __cycle();

        // Convolution
        FTRACE_BEGIN(pConv->region);
        if ( flagBias ) {
            rv = vednnConvolutionForwardAddBias(pConv->pParamIn, pConv->pDataIn,
                    pConv->pParamKernel, pConv->pDataKernel,
                    pConv->pParamBias, pConv->pDataBias,
                    pConv->pParamOut, pConv->pDataOut,
                    pConv->pParamConv, VEDNN_CONV_ALGORITHM_DIRECT );
        }
        else {
            rv = vednnConvolutionForward(pConv->pParamIn, pConv->pDataIn,
                    pConv->pParamKernel, pConv->pDataKernel,
                    pConv->pParamOut, pConv->pDataOut,
                    pConv->pParamConv, VEDNN_CONV_ALGORITHM_DIRECT );
        }
        FTRACE_END(pConv->region);
        FTRACE_END(all_region);
        if (rv != VEDNN_SUCCESS) ERROR_EXIT("convolution() failed.");

        c[1] = __cycle();
        unsigned long long d = c[1] - c[0];
        if( pConv->reps == 0U || d < pConv->mincycle ) pConv->mincycle = d;
        if( pConv->reps == 0U || d > pConv->maxcycle ) pConv->maxcycle = d;
        pConv->cycle += d;
        ++pConv->reps;
    }
}
void
testconvBackwardData_vednncalcs( struct testconvBackwardData *pConvArray, int const nEntry ){
    vednnError_t rv;
    FTRACE_IF(char const* allconv = "all BkwdD convolution");
    FTRACE_BEGIN(allconv);
    for (int i=0; i<nEntry; i++) {
        struct testconvBackwardData *pConv = &pConvArray[i];
        unsigned long long c[2];
        c[0] = __cycle();
        FTRACE_BEGIN(pConv->region);
        // Convolution
        rv = vednnConvolutionBackwardData(pConv->pParamGradOut, pConv->pDataGradOut,
                pConv->pParamKernel, pConv->pDataKernel,
                pConv->pParamGradIn, pConv->pDataGradIn,
                pConv->pParamConv, VEDNN_CONV_ALGORITHM_DIRECT );
        if (rv != VEDNN_SUCCESS) ERROR_EXIT("convolution() failed.\n");
        FTRACE_END(pConv->region);
        c[1] = __cycle();
        unsigned long long d = c[1] - c[0];
        if( pConv->reps == 0U || d < pConv->mincycle ) pConv->mincycle = d;
        if( pConv->reps == 0U || d > pConv->maxcycle ) pConv->maxcycle = d;
        pConv->cycle += d;
        ++pConv->reps;
    }
    FTRACE_END(allconv);
}
void
testconvBackwardFilter_vednncalcs( struct testconvBackwardFilter *pConvArray, int const nEntry ){
    vednnError_t rv;
    FTRACE_IF(char const* all_region = "all BkwF convolution");
    FTRACE_BEGIN(all_region);
    for (int i=0; i<nEntry; i++) {
        struct testconvBackwardFilter *pConv = &pConvArray[i];
        unsigned long long c[2];
        c[0] = __cycle();
        FTRACE_BEGIN(pConv->region);
        // Convolution
        rv = vednnConvolutionBackwardFilter(pConv->pParamIn, pConv->pDataIn,
                pConv->pParamGradOut, pConv->pDataGradOut,
                pConv->pParamGradKernel, pConv->pDataGradKernel,
                pConv->pParamConv, VEDNN_CONV_ALGORITHM_DIRECT );
        if (rv != VEDNN_SUCCESS) ERROR_EXIT("convolution() failed.\n");
        FTRACE_END(pConv->region);
        c[1] = __cycle();
        unsigned long long d = c[1] - c[0];
        if( pConv->reps == 0U || d < pConv->mincycle ) pConv->mincycle = d;
        if( pConv->reps == 0U || d > pConv->maxcycle ) pConv->maxcycle = d;
        pConv->cycle += d;
        ++pConv->reps;
    }
    FTRACE_END(all_region);
}

#ifdef __cplusplus
}//extern "C"
#endif //C++
// vim: et ts=4 sw=4 cindent cino=^=l0,\:0,N-s syntax=cpp.doxygen
