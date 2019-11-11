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

static void oclobber_float(float * const out, size_t const nFloat){
    static const int oclobber_quick = 1;
    static uint64_t oclobber_rand = 13ULL;
    if(out==NULL || nFloat <= 0){
        printf(" clobber out==%p, nFloat=%lu ?? IGNORED\n", (void*)out, (long unsigned)nFloat);
    }else{
        printf(" clobber out=%p nFloat=%lu\n", (void*)out, (long unsigned)nFloat);
        if(oclobber_quick){ // just change a few entries to wrong result
            out[0] = 999.99f;
            out[oclobber_rand % nFloat] = 13.13e5f;
            oclobber_rand = oclobber_rand * (uint64_t)7664345821815920749ULL + (uint64_t)13ULL;
            out[nFloat-1U] = 999.99f;
        }else{
            //memset(pConv->pDataOut,    0, getTensorSizeInByte(pConv->pParamOut));
            for(size_t i=0U; i<nFloat; ++i) out[i] = 13.13f;
        }
    }
}

// testconvForward_FOO
    void
testconvForward_init( struct testconvForward *pConv )
{
    pConv->pParamIn     = NULL;
    pConv->pParamOut    = NULL;
    pConv->pParamBias   = NULL;
    pConv->pParamKernel = NULL;
    pConv->pParamConv   = NULL;
    pConv->pDataIn      = NULL;
    pConv->pDataOut     = NULL;
    pConv->pDataBias    = NULL;
    pConv->pDataKernel  = NULL;

    pConv->pBufRef      = NULL;
    pConv->pBufOne      = NULL;
    pConv->pBufCol      = NULL;

    pConv->cycle        = 0;
    pConv->mincycle     = 0;
    pConv->maxcycle     = 0;
    pConv->ops          = 0;
    pConv->reps         = 0;

    pConv->region[0]    = '\0';
    pConv->ref_region[0]= '\0';
}
void
testconvForward_alloc( struct testconvForward *pConv, struct param const* pNw,
        int const flagBias, filterLayout_t filter_layout ){
    pConv->ops = count_ops(pNw);
    vednnError_t rv[5];
    int inChannelGroup;
    int outChannelGroup;

    if (pNw->inChannel % pNw->group) ERROR_EXIT("Could not divide inChannel by group.");
    if (pNw->outChannel % pNw->group) ERROR_EXIT("Could not divide outChannel by group.");

    inChannelGroup    = pNw->inChannel  / pNw->group;
    outChannelGroup    = pNw->outChannel / pNw->group;

    rv[0] = createTensorParam(&pConv->pParamIn, DTYPE_FLOAT, pNw->batchNum, pNw->inChannel,  pNw->inWidth,  pNw->inHeight);
    rv[1] = createTensorParam(&pConv->pParamOut, DTYPE_FLOAT, pNw->batchNum, pNw->outChannel, pNw->outWidth, pNw->outHeight);
    if( flagBias ) {
        rv[2] = createBiasParam(&pConv->pParamBias, DTYPE_FLOAT, pNw->outChannel);
    }
    rv[3] = createKernelParam(&pConv->pParamKernel, DTYPE_FLOAT, filter_layout,
            inChannelGroup, outChannelGroup, pNw->kernWidth, pNw->kernHeight);
    rv[4] = createConvolutionParam(&pConv->pParamConv, pNw->group, pNw->strideWidth, pNw->strideHeight, pNw->padWidth, pNw->padHeight, pNw->dilationHeight, pNw->dilationWidth);
    if (rv[0] != VEDNN_SUCCESS || rv[1] != VEDNN_SUCCESS || ( flagBias && rv[2] != VEDNN_SUCCESS )
            || rv[3] != VEDNN_SUCCESS || rv[4] != VEDNN_SUCCESS )
        ERROR_EXIT("Failed to create/initialize structure.");


    pConv->pDataIn     = malloc(getTensorSizeInByte(pConv->pParamIn  ));
    pConv->pDataOut    = malloc(getTensorSizeInByte(pConv->pParamOut ));
    if( flagBias ) {
        pConv->pDataBias   = malloc(getBiasSizeInByte(pConv->pParamBias));
    }
    pConv->pDataKernel = malloc(getKernelSizeInByte(pConv->pParamKernel) * pNw->group);
    if (pConv->pDataIn == NULL || pConv->pDataOut == NULL || ( flagBias && pConv->pDataBias == NULL )
            || pConv->pDataKernel == NULL)
        ERROR_EXIT("Memory exhaust.");

    pConv->pBufRef  = malloc(getTensorSizeInByte(pConv->pParamOut )) ;
    size_t pOnesize = pConv->pParamOut->width * pConv->pParamOut->height;
    pConv->pBufOne  = (float*) malloc( pOnesize * getTensorDataSize(pConv->pParamIn));
    size_t pColrows = pConv->pParamKernel->inChannel * pConv->pParamKernel->width * pConv->pParamKernel->height;
    size_t pColcols = pConv->pParamOut->width * pConv->pParamOut->height;
    pConv->pBufCol  = (float*) malloc(pColrows*pColcols*getTensorDataSize(pConv->pParamIn));

    if (pConv->pBufRef == NULL) ERROR_EXIT("Memory exhausted.");
    if (pConv->pBufOne == NULL) ERROR_EXIT("Memory exhausted.");
    if (pConv->pBufCol == NULL) ERROR_EXIT("Memory exhausted.");

    memset(pConv->pDataIn,     0, getTensorSizeInByte(pConv->pParamIn));
    memset(pConv->pDataOut,    0, getTensorSizeInByte(pConv->pParamOut));
    if (flagBias ) {
        memset(pConv->pDataBias,   0, getBiasSizeInByte(pConv->pParamBias));
    }
    memset(pConv->pDataKernel, 0, getKernelSizeInByte(pConv->pParamKernel) * pNw->group);

    memset(pConv->pBufRef,    0, getTensorSizeInByte(pConv->pParamOut));
    for (size_t i=0; i<pOnesize; i++) {
        pConv->pBufOne[i] = 1.0f;
    }
}
void
testconvForward_dumpParms( struct testconvForward const* pConv, int const flagBias ){
    vednnTensorParam_t *tpIn = pConv->pParamIn;
    vednnFilterParam_t *tpKrn= pConv->pParamKernel;
    vednnTensorParam_t *tpOut= pConv->pParamOut;
    vednnConvolutionParam_t *parm = pConv->pParamConv;
    if(!tpIn) printf("tpIn:NULL\n");
    else printf("tpIn {f32,mb%d,%d,%d,%d}\n",tpIn->batch,tpIn->channel,tpIn->width,tpIn->height);
    if(!tpKrn) printf("tpKrn:NULL\n");
    else printf("tpKrn{f32,ic%doc%d_kw%dkh%d}\n",tpKrn->inChannel,tpKrn->outChannel,tpKrn->width,tpKrn->height);
    if( flagBias ){
        vednnBiasParam_t *tpBi = pConv->pParamBias;
        if(!tpBi) printf("tpBi:NULL\n");
        else printf("tpBi {f32,%d}\n",tpBi->channel);
    }
    if(!tpOut) printf("tpOut:NULL");
    else printf("tpOut{f32,%d,%d,%d,%d}\n",tpOut->batch,tpOut->channel,tpOut->width,tpOut->height);
    if(!parm) printf("parm:NULL");
    else printf("parm{g%d_sw%dsh%d_pw%dph%d_dw%ddh%d}\n",parm->group,
            parm->strideWidth,   parm->strideHeight,
            parm->padWidth,      parm->padHeight,
            parm->dilationWidth, parm->dilationHeight);
}
void
testconvForward_randomData( struct testconvForward const* pConv, int const flagBias ){
    generateRandomData(getTensorDataType(pConv->pParamIn),
            getTensorSize(pConv->pParamIn), pConv->pDataIn);
    if( flagBias ) {
        generateRandomData(getBiasDataType(pConv->pParamBias),
                getBiasSize(pConv->pParamBias), pConv->pDataBias);
    }
    generateRandomData(getKernelDataType(pConv->pParamKernel),
            getKernelSize(pConv->pParamKernel) * pConv->pParamConv->group,
            pConv->pDataKernel);
}
void testconvForward_oclobber( struct testconvForward const* pConv ){
    switch(getTensorDataType(pConv->pParamOut)){
    case DTYPE_FLOAT: {
                          float * const out = (float*)pConv->pDataOut;
                          size_t const nFloat = getTensorSize(pConv->pParamOut);
                          oclobber_float(out,nFloat);
                      }
                      break;
    default: ERROR_EXIT("Unknown dataType_t"); break;
    }
}
void
testconvForward_refcalcs( struct testconvForward *pConvArray, int const nEntry ){
    vednnError_t rv;
    FTRACE_IF(char const* ref_all_region = "<gemm:Fwd>all");
    FTRACE_BEGIN(ref_all_region);
    for (int i=0; i<nEntry; i++) {
        struct testconvForward *pConv = &pConvArray[i];
        // Convolution
        FTRACE_BEGIN(pConv->ref_region);
        rv = convolution_forward_gemm(pConv->pParamIn, pConv->pDataIn,
                pConv->pParamKernel, pConv->pDataKernel,
                pConv->pParamBias, pConv->pDataBias,
                pConv->pParamOut, pConv->pBufRef,
                pConv->pBufOne, pConv->pBufCol,
                pConv->pParamConv );
        if (rv != VEDNN_SUCCESS) ERROR_EXIT("convolution() failed.\n");
        FTRACE_END(pConv->ref_region);
    }
    FTRACE_END(ref_all_region);
}
void
testconvForward_vednncalcs( struct testconvForward *pConvArray, int const nEntry ){
    vednnError_t rv = VEDNN_SUCCESS;

    for (int i=0; i<nEntry; i++) {
        struct testconvForward *pConv = &pConvArray[i];
        int const flagBias = (pConv->pDataBias != NULL);
        int namedLayer = strcmp(pConv->region,"\"wip\"") != 0;
#ifdef FTRACE
        FTRACE_IF(char const* all_region = (flagBias? "vednn-def all FwdB conv": "vednn-def all Fwd conv"));
        printf("all_region = %s\ndef_region = %s\n",all_region,pConv->region);
#endif
        FTRACE_BEGIN(all_region);

        unsigned long long c[2];
        c[0] = __cycle();

        // Convolution
        if(namedLayer) FTRACE_BEGIN(pConv->region);
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
        if(namedLayer) FTRACE_END(pConv->region);
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
void testconvForward_free( struct testconvForward *pConv, int const flagBias){
    destroyTensorParam(pConv->pParamIn);
    destroyTensorParam(pConv->pParamOut);
    destroyKernelParam(pConv->pParamKernel);
    if( flagBias ){
        destroyBiasParam(pConv->pParamBias);
        free(pConv->pDataBias);
        pConv->pDataBias = NULL;
    }
    free(pConv->pDataIn);
    free(pConv->pDataOut);
    free(pConv->pDataKernel);

    free(pConv->pBufRef);
    free(pConv->pBufOne);
    free(pConv->pBufCol);
    pConv->pDataIn = pConv->pDataOut = pConv->pDataKernel
        = pConv->pBufRef = pConv->pBufOne = pConv->pBufCol = NULL;
}
void testconvBackwardData_init( struct testconvBackwardData *pConv ){
    pConv->pParamGradOut = NULL;
    pConv->pParamGradIn  = NULL;
    pConv->pParamKernel  = NULL;
    pConv->pParamConv    = NULL;
    pConv->pDataGradOut  = NULL;
    pConv->pDataGradIn   = NULL;
    pConv->pDataKernel   = NULL;

    pConv->pBufRef       = NULL;
    pConv->pBufCol       = NULL;

    pConv->cycle         = 0;
    pConv->mincycle      = 0;
    pConv->maxcycle      = 0;
    pConv->reps          = 0;
    pConv->region[0]     = '\0';
    pConv->ref_region[0] = '\0';
}
void
testconvBackwardData_alloc( struct testconvBackwardData *pConv, struct param const* pNw, filterLayout_t filter_layout ){
    pConv->ops = count_ops(pNw);
    vednnError_t rv[4];
    int inChannelGroup;
    int outChannelGroup;

    if (pNw->inChannel % pNw->group) {
        fprintf(stderr, "\nCould not divide inChannel by group.");
        exit(1);
    }
    if (pNw->outChannel % pNw->group) {
        fprintf(stderr, "\nCould not divide outChannel by group.");
        exit(1);
    }

    inChannelGroup  = pNw->inChannel  / pNw->group;
    outChannelGroup = pNw->outChannel / pNw->group;

    rv[0] = createTensorParam(&pConv->pParamGradOut, DTYPE_FLOAT, pNw->batchNum,
            pNw->outChannel, pNw->outWidth, pNw->outHeight);
    rv[1] = createTensorParam(&pConv->pParamGradIn, DTYPE_FLOAT, pNw->batchNum,
            pNw->inChannel, pNw->inWidth, pNw->inHeight);
    rv[2] = createKernelParam(&pConv->pParamKernel, DTYPE_FLOAT, filter_layout,
            inChannelGroup, outChannelGroup, pNw->kernWidth, pNw->kernHeight);
    rv[3] = createConvolutionParam(&pConv->pParamConv, pNw->group, pNw->strideWidth,
            pNw->strideHeight, pNw->padWidth, pNw->padHeight, pNw->dilationHeight, pNw->dilationWidth);
    if (rv[0] != VEDNN_SUCCESS || rv[1] != VEDNN_SUCCESS || rv[2] != VEDNN_SUCCESS || rv[3] != VEDNN_SUCCESS ) {
        fprintf(stderr, "Failed to create/initialize structure.\n");
        exit(1);
    }


    pConv->pDataGradOut = malloc(getTensorSizeInByte(pConv->pParamGradOut  ));
    pConv->pDataGradIn  = malloc(getTensorSizeInByte(pConv->pParamGradIn ));
    pConv->pDataKernel  = malloc(getKernelSizeInByte(pConv->pParamKernel) * pNw->group);

    pConv->pBufRef  = malloc(getTensorSizeInByte(pConv->pParamGradIn )) ;
    size_t pColrows = pConv->pParamKernel->inChannel * pConv->pParamKernel->width * pConv->pParamKernel->height;
    size_t pColcols = pConv->pParamGradOut->width * pConv->pParamGradOut->height;
    pConv->pBufCol  = (float*) malloc(pColrows*pColcols*getTensorDataSize(pConv->pParamGradIn));
    printf("gemm pBufCol[%ld]\n", pColrows*pColcols*getTensorDataSize(pConv->pParamGradIn));


    memset(pConv->pDataGradOut, 0, getTensorSizeInByte(pConv->pParamGradOut));
    memset(pConv->pDataGradIn,  0, getTensorSizeInByte(pConv->pParamGradIn));
    memset(pConv->pDataKernel,  0, getKernelSizeInByte(pConv->pParamKernel) * pNw->group);

    memset(pConv->pBufRef,      0, getTensorSizeInByte(pConv->pParamGradIn));
}
void
testconvBackwardData_randomData( struct testconvBackwardData const* pConv ){
    generateRandomData(getTensorDataType(pConv->pParamGradOut),
            getTensorSize(pConv->pParamGradOut), pConv->pDataGradOut);
    generateRandomData(getKernelDataType(pConv->pParamKernel),
            getKernelSize(pConv->pParamKernel) * pConv->pParamConv->group,
            pConv->pDataKernel);
}
void
testconvBackwardData_dumpParms( struct testconvBackwardData const *pConv ){
    vednnTensorParam_t      *tpGo  = pConv->pParamGradOut;
    vednnFilterParam_t      *tpKrn = pConv->pParamKernel;
    vednnTensorParam_t      *tpGin = pConv->pParamGradIn;
    vednnConvolutionParam_t *parm  = pConv->pParamConv;
    if(!tpGo) printf("tpGo:NULL\n");
    else printf("tpGo {f32,mb%d,%d,%d,%d}\n",tpGo->batch,tpGo->channel,tpGo->width,tpGo->height);
    printf("tpKrn{f32,ic%doc%d_kw%dkh%d}\n",tpKrn->inChannel,tpKrn->outChannel,tpKrn->width,tpKrn->height);
    printf("tpGin{f32,mb%d,%d,%d,%d}\n",tpGin->batch,tpGin->channel,tpGin->width,tpGin->height);
    printf("parm{g%d_sw%dsh%d_pw%dph%d_dw%ddh%d}\n",parm->group,
            parm->strideWidth,   parm->strideHeight,
            parm->padWidth,      parm->padHeight,
            parm->dilationWidth, parm->dilationHeight);
}
void
testconvBackwardData_oclobber( struct testconvBackwardData const* pConv ){
    switch(getTensorDataType(pConv->pParamGradOut)){
    case DTYPE_FLOAT: {
                          float * const out = (float*)pConv->pDataGradOut;
                          size_t const nFloat = getTensorSize(pConv->pParamGradOut);
                          oclobber_float(out,nFloat);
                      }
                      break;
    default: ERROR_EXIT("Unknown dataType_t"); break;
    }
}
void
testconvBackwardData_refcalcs( struct testconvBackwardData *pConvArray, int const nEntry ){
    // this one is nice because it uses a local ref_regions (so can remove the field)
    vednnError_t rv;
    FTRACE_IF(char const* ref_all_region = "<gemm:BkwD> all convolution");
    struct RefRegions { char name[128]; };
    struct RefRegions *ref_regions = (struct RefRegions*)XMALLOC(nEntry*sizeof(struct RefRegions));
    for (int i=0; i<nEntry; i++){
        snprintf( &ref_regions[i].name[0], 128, "<gemm:BkwD> %s%c", pConvArray[i].region,'\0');
    }
    FTRACE_BEGIN(ref_all_region);
    for (int i=0; i<nEntry; i++) {
        struct testconvBackwardData *pConv = &pConvArray[i];
        FTRACE_BEGIN(ref_regions[i].name);
        // Convolution
        rv = convolution_backward_data_gemm(pConv->pParamGradOut, pConv->pDataGradOut,
                pConv->pParamKernel, pConv->pDataKernel,
                pConv->pParamGradIn, pConv->pBufRef,
                pConv->pBufCol, pConv->pParamConv );
        FTRACE_END(ref_regions[i].name);
        if (rv != VEDNN_SUCCESS) {
            fprintf(stderr, "convolution() failed.\n");
            exit(1);
        }
    }
    FTRACE_END(ref_all_region);
    free(ref_regions);
}
void
testconvBackwardData_vednncalcs( struct testconvBackwardData *pConvArray, int const nEntry ){
    vednnError_t rv;
    FTRACE_IF(char const* allconv = "all BkwdD convolution");
    FTRACE_BEGIN(allconv);
    for (int i=0; i<nEntry; i++) {
        struct testconvBackwardData *pConv = &pConvArray[i];
        unsigned long long c[2];
        int namedLayer = strcmp(pConv->region,"\"wip\"") != 0;
        c[0] = __cycle();
        if(namedLayer) FTRACE_BEGIN(pConv->region);
        // Convolution
        rv = vednnConvolutionBackwardData(pConv->pParamGradOut, pConv->pDataGradOut,
                pConv->pParamKernel, pConv->pDataKernel,
                pConv->pParamGradIn, pConv->pDataGradIn,
                pConv->pParamConv, VEDNN_CONV_ALGORITHM_DIRECT );
        if (rv != VEDNN_SUCCESS) ERROR_EXIT("convolution() failed.\n");
        if(namedLayer) FTRACE_END(pConv->region);
        c[1] = __cycle();
        unsigned long long d = c[1] - c[0];
        if( pConv->reps == 0U || d < pConv->mincycle ) pConv->mincycle = d;
        if( pConv->reps == 0U || d > pConv->maxcycle ) pConv->maxcycle = d;
        pConv->cycle += d;
        ++pConv->reps;
    }
    FTRACE_END(allconv);
}
void testconvBackwardData_free( struct testconvBackwardData *pConv ){
    destroyTensorParam(pConv->pParamGradOut);
    destroyTensorParam(pConv->pParamGradIn);
    destroyKernelParam(pConv->pParamKernel);
    free(pConv->pDataGradOut);
    free(pConv->pDataGradIn);
    free(pConv->pDataKernel);

    free(pConv->pBufRef);
    free(pConv->pBufCol); // move into gemm-block?
    pConv->pDataGradOut = pConv->pDataGradIn = pConv->pDataKernel
        = pConv->pBufRef = pConv->pBufCol = NULL;
}

// testconvBackwardFilter_FOO
void testconvBackwardFilter_init( struct testconvBackwardFilter *pConv ){
    pConv->pParamGradOut    = NULL;
    pConv->pParamIn         = NULL;
    pConv->pParamGradKernel = NULL;
    pConv->pParamConv       = NULL;
    pConv->pDataGradOut     = NULL;
    pConv->pDataIn          = NULL;
    pConv->pDataGradKernel  = NULL;

    pConv->pBufRef          = NULL;
    pConv->pBufCol          = NULL;

    pConv->cycle            = 0;
    pConv->mincycle         = 0;
    pConv->maxcycle         = 0;
    pConv->reps             = 0;
    pConv->region[0]        = '\0';
    pConv->ref_region[0]    = '\0';
}
void testconvBackwardFilter_alloc( struct testconvBackwardFilter *pConv, struct param const* pNw, filterLayout_t filter_layout ){
    pConv->ops = count_ops(pNw);
    vednnError_t rv[4];
    int inChannelGroup;
    int outChannelGroup;

    if (pNw->inChannel % pNw->group) {
        fprintf(stderr, "\nCould not divide inChannel by group.");
        exit(1);
    }
    if (pNw->outChannel % pNw->group) {
        fprintf(stderr, "\nCould not divide outChannel by group.");
        exit(1);
    }

    inChannelGroup    = pNw->inChannel  / pNw->group;
    outChannelGroup    = pNw->outChannel / pNw->group;

    rv[0] = createTensorParam(&pConv->pParamIn, DTYPE_FLOAT, pNw->batchNum, pNw->inChannel, pNw->inWidth, pNw->inHeight);
    rv[1] = createTensorParam(&pConv->pParamGradOut, DTYPE_FLOAT, pNw->batchNum, pNw->outChannel, pNw->outWidth, pNw->outHeight);
    rv[2] = createKernelParam(&pConv->pParamGradKernel, DTYPE_FLOAT, filter_layout,
            inChannelGroup, outChannelGroup, pNw->kernWidth, pNw->kernHeight);
    rv[3] = createConvolutionParam(&pConv->pParamConv, pNw->group, pNw->strideWidth, pNw->strideHeight, pNw->padWidth, pNw->padHeight, pNw->dilationHeight, pNw->dilationWidth);
    if (rv[0] != VEDNN_SUCCESS || rv[1] != VEDNN_SUCCESS || rv[2] != VEDNN_SUCCESS || rv[3] != VEDNN_SUCCESS ) {
        fprintf(stderr, "Failed to create/initialize structure.\n");
        exit(1);
    }


    pConv->pDataIn= malloc(getTensorSizeInByte(pConv->pParamIn ));
    pConv->pDataGradOut = malloc(getTensorSizeInByte(pConv->pParamGradOut  ));
    pConv->pDataGradKernel = malloc(getKernelSizeInByte(pConv->pParamGradKernel) * pNw->group);

    pConv->pBufRef  = malloc(getKernelSizeInByte(pConv->pParamGradKernel) * pNw->group);
    size_t pColrows = pConv->pParamGradKernel->inChannel * pConv->pParamGradKernel->width * pConv->pParamGradKernel->height;
    size_t pColcols = pConv->pParamIn->width * pConv->pParamIn->height;
    pConv->pBufCol  = (float*) malloc(pColrows*pColcols*getTensorDataSize(pConv->pParamIn));

    memset(pConv->pDataIn, 0, getTensorSizeInByte(pConv->pParamIn));
    memset(pConv->pDataGradOut,  0, getTensorSizeInByte(pConv->pParamGradOut));
    memset(pConv->pDataGradKernel,  0, getKernelSizeInByte(pConv->pParamGradKernel) * pNw->group);

    memset(pConv->pBufRef,    0, getTensorSizeInByte((vednnTensorParam_t*)(pConv->pParamGradKernel)));

}
void testconvBackwardFilter_dumpParms( struct testconvBackwardFilter const *pConv ){
    vednnTensorParam_t      *tpIn  = pConv->pParamIn;
    vednnTensorParam_t      *tpGo  = pConv->pParamGradOut;
    vednnFilterParam_t      *tpGk  = pConv->pParamGradKernel;
    vednnConvolutionParam_t *parm  = pConv->pParamConv;
    if(!tpIn) printf("tpIn:NULL\n");
    else printf("tpIn {f32,mb%d,%d,%d,%d}\n",tpIn->batch,tpIn->channel,tpIn->width,tpIn->height);
    if(!tpGo) printf("tpGo:NULL\n");
    else printf("tpGo {f32,mb%d,%d,%d,%d}\n",tpGo->batch,tpGo->channel,tpGo->width,tpGo->height);
    if(!tpGk) printf("tpGk:NULL\n");
    else printf("tpGk{f32,ic%doc%d_kw%dkh%d}\n",tpGk->inChannel,tpGk->outChannel,tpGk->width,tpGk->height);
    printf("parm{g%d_sw%dsh%d_pw%dph%d_dw%ddh%d}\n",parm->group,
            parm->strideWidth,   parm->strideHeight,
            parm->padWidth,      parm->padHeight,
            parm->dilationWidth, parm->dilationHeight);
}
void testconvBackwardFilter_randomData( struct testconvBackwardFilter const* pConv ){
    generateRandomData(getTensorDataType(pConv->pParamIn),
            getTensorSize(pConv->pParamIn), pConv->pDataIn);
    generateRandomData(getTensorDataType(pConv->pParamGradOut),
            getTensorSize(pConv->pParamGradOut), pConv->pDataGradOut);
}
void
testconvBackwardFilter_refcalcs( struct testconvBackwardFilter *pConvArray, int const nEntry ){
    // this one is nice because it uses a local ref_regions (so can remove the field)
    vednnError_t rv;
#if defined(FTRACE)
    char const* ref_all_region = "<gemm:BkwF> all convolution";
    struct RefRegions { char name[128]; };
    struct RefRegions *ref_regions = (struct RefRegions*)XMALLOC(nEntry*sizeof(struct RefRegions));
    for (int i=0; i<nEntry; i++){
        snprintf( &ref_regions[i].name[0], 128, "<gemm:BkwD> %s%c", pConvArray[i].region,'\0');
    }
#endif
    FTRACE_BEGIN(ref_all_region);
    for (int i=0; i<nEntry; i++) {
        struct testconvBackwardFilter *pConv = &pConvArray[i];
        FTRACE_BEGIN(ref_regions[i].name);
        // Convolution
        rv = convolution_backward_filter_gemm(pConv->pParamIn, pConv->pDataIn,
                pConv->pParamGradOut, pConv->pDataGradOut,
                pConv->pParamGradKernel, pConv->pBufRef,
                pConv->pBufCol, pConv->pParamConv );
        FTRACE_END(ref_regions[i].name);
        if (rv != VEDNN_SUCCESS) ERROR_EXIT("convolution() failed.\n");
    }
    FTRACE_END(ref_all_region);
#if defined(FTRACE)
    free(ref_regions);
#endif
}
void
testconvBackwardFilter_vednncalcs( struct testconvBackwardFilter *pConvArray, int const nEntry ){
    vednnError_t rv;
    FTRACE_IF(char const* all_region = "all BkwF convolution");
    FTRACE_BEGIN(all_region);
    for (int i=0; i<nEntry; i++) {
        struct testconvBackwardFilter *pConv = &pConvArray[i];
        unsigned long long c[2];
        int namedLayer = strcmp(pConv->region,"\"wip\"") != 0;
        c[0] = __cycle();
        if(namedLayer) FTRACE_BEGIN(pConv->region);
        // Convolution
        rv = vednnConvolutionBackwardFilter(pConv->pParamIn, pConv->pDataIn,
                pConv->pParamGradOut, pConv->pDataGradOut,
                pConv->pParamGradKernel, pConv->pDataGradKernel,
                pConv->pParamConv, VEDNN_CONV_ALGORITHM_DIRECT );
        if (rv != VEDNN_SUCCESS) ERROR_EXIT("convolution() failed.\n");
        if(namedLayer) FTRACE_END(pConv->region);
        c[1] = __cycle();
        unsigned long long d = c[1] - c[0];
        if( pConv->reps == 0U || d < pConv->mincycle ) pConv->mincycle = d;
        if( pConv->reps == 0U || d > pConv->maxcycle ) pConv->maxcycle = d;
        pConv->cycle += d;
        ++pConv->reps;
    }
    FTRACE_END(all_region);
}
void testconvBackwardFilter_free( struct testconvBackwardFilter *pConv ){
    destroyTensorParam(pConv->pParamIn);
    destroyTensorParam(pConv->pParamGradOut);
    destroyKernelParam(pConv->pParamGradKernel);
    free(pConv->pDataIn);
    free(pConv->pDataGradOut);
    free(pConv->pDataGradKernel);

    free(pConv->pBufRef);
    free(pConv->pBufCol);
    pConv->pDataIn = pConv->pDataGradOut = pConv->pDataGradKernel
        = pConv->pBufRef = pConv->pBufCol = NULL;
}

#ifdef __cplusplus
}//extern "C"
#endif //C++
// vim: et ts=4 sw=4 cindent cino=^=l0,\:0,N-s syntax=cpp.doxygen
