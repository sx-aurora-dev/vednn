
#include "vednn_helper.h"

vednnError_t
convolution_forward_gemm(
    const vednnTensorParam_t * restrict pParamIn, const void * restrict pDataIn,
    const vednnFilterParam_t * restrict pParamKernel, const void * restrict pDataKernel,
    const vednnBiasParam_t * restrict pParamBias, const void * restrict pDataBias,
    const vednnTensorParam_t * restrict pParamOut, void * restrict pDataOut,
    const float * restrict pOne,  float * restrict pColBuff,
    const vednnConvolutionParam_t * restrict pParamConv ) ;

vednnError_t
convolution_backward_data_gemm(
    const vednnTensorParam_t * restrict pParamGradOut, const void * restrict pDataGradOut,
    const vednnFilterParam_t * restrict pParamKernel, const void * restrict pDataKernel,
    const vednnTensorParam_t * restrict pParamGradIn, void * restrict pDataGradIn,
    float * restrict pColBuff,
    const vednnConvolutionParam_t * restrict pParamConv ) ;

vednnError_t
convolution_backward_filter_gemm(
    const vednnTensorParam_t * restrict pParamIn, const void * restrict pDataIn,
    const vednnTensorParam_t * restrict pParamGradOut, const void * restrict pDataGradOut,
    const vednnFilterParam_t * restrict pParamGradKernel, void * restrict pDataGradKernel,
    float * restrict pColBuff,
    const vednnConvolutionParam_t * restrict pParamConv ) ;

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <math.h>

#ifdef FTRACE
#include <ftrace.h>
#endif

#include "timer.h"

struct param {
    const char	pName[256];
    int		batchNum;
    int		group;
    int		inChannel;
    int		inHeight;
    int		inWidth;
    int		outChannel;
    int		outHeight;
    int		outWidth;
    int		kernHeight;
    int		kernWidth;
    int		strideHeight;
    int		strideWidth;
    int		padHeight;
    int		padWidth;
};

// exact output height/width for convolution. d==0 for "no dilation".
int compute_out( int i, int k, int s, int p, int d ){
    // convolution min output size:
    return (i - ((k - 1) * (d + 1) + 1) + 2 * p) / s + 1;
    // deconv: return (i - 1) * s + (k - 1) * (d + 1) + 2 * p + 1;
}
// round i>0 upward to a multiple of a>0
int upMul( int const i, int const a ) {
    return (i+a-1)/a*a;
}
// return # of modifications needed to make the test look reasonable.
int mkConsistent( struct param* p ){
    // fix bad things so at least some test gets run.
    // (If I change padding or dilation or ... I am too lazy
    //  to calculate the new output size, for example).
    //
    // I try not to weed out valid test settings...
    //
    int g = p->group;
    // g cannot be larger then p->inChannel
    if (p->group > p->inChannel) g = p->inChannel;
    // g must divide evenly into p->inChannel
    while( p->inChannel % g != 0 ) --g;

    int ic = upMul( p->inChannel, g );
    // one pass over input channels uses ic/g kernels to make g output channels.
    int oc = upMul( p->outChannel, g );

    // "no dilation" ~ value 0 (mkl-dnn convention, does this match vednn?) XXX
    int dh = 0; // dilationHeight
    int dw = 0; // dilationWidth
    int oh = compute_out( p->inHeight, p->kernHeight, p->strideHeight, p->padHeight, dh );
    int ow = compute_out( p->inWidth , p->kernWidth , p->strideWidth , p->padWidth , dw );

#if 0 // In principle, coulld malloc any larger output region, but...
    // the 'DIFF' calc in this test program only handles exact-sized output
    // today (could be fixed, but not so important for correctness testing)
    oh = (p->outHeight >= oh? p->outHeight: oh);
    ow = (p->outWidth  >= ow? p->outWidth : ow);
#endif

    // apply changes, warn...
    int changed = 0;
#define FIXIT( ITEM, EXPR ) do { \
    int const val = (EXPR); \
    if( p->ITEM != val ){ \
        printf(" " #ITEM " %d->%d",(int)p->ITEM, val); \
        p->ITEM = val; \
        ++changed; \
    } \
}while(0)
    FIXIT( group, g );
    FIXIT( inChannel, ic );
    FIXIT( outChannel, oc ); // any number is fine
    FIXIT( outHeight, oh );
    FIXIT( outWidth,  ow );
#undef FIXIT
    if (changed){
        printf(" changed in %s\n", p->pName);
        fflush(stdout);
    }
    return changed;
}


void
generateRandomData(dataType_t type, size_t size, void *pData)
{
    int i;

    switch(type) {
    case DTYPE_FLOAT:
        {
            float *p = (float *)pData;
            for (i=0; i<size; i++) {
                p[i] = drand48();
            }
        }
        break;
    default:
        assert(0);              /* BUG */
        break;
    }
}

double
diffData(const vednnTensorParam_t *pParam, const void *pData, const void *pExpectedResult)
{
    int i;

    size_t size = getTensorSize(pParam);
    double sum = 0.0;

    switch(getTensorDataType(pParam)) {
    case DTYPE_FLOAT:
        {
            const float *pData1 = (float *)pExpectedResult;
            const float *pData2 = (float *)pData;
            for (i=0; i<size; i++) {
                printf("%lf %lf\n", pData1[i], pData2[i]) ; 
                double diff = pData1[i] - pData2[i] ;
                sum += (diff/(fabs(pData1[i])+1e-7)) * (diff/(fabs(pData2[i])+1e-7)) ;
            }
        }
        break;
    default:
        assert(0);              /* BUG */
        break;
    }

    return sqrt(sum);
}

double
diffFilter(const vednnFilterParam_t *pParam, const void *pData, const void *pExpectedResult)
{
    int i;

    size_t size = pParam->inChannel * pParam->outChannel * pParam->width * pParam->height ;
    double sum = 0.0;

    switch(pParam->dtype) {
    case DTYPE_FLOAT:
        {
            const float *pData1 = (float *)pExpectedResult;
            const float *pData2 = (float *)pData;
            for (i=0; i<size; i++) {
                double diff = pData1[i] - pData2[i] ;
                sum += (diff/(fabs(pData1[i])+1e-7)) * (diff/(fabs(pData2[i])+1e-7)) ;
            }
        }
        break;
    default:
        assert(0);              /* BUG */
        break;
    }

    return sqrt(sum);
}


void testForward(struct param *pNetwork, int nEntry, double HZ, int flagBias, int flagCSV, filterLayout_t filter_layout)
{
    struct conv {
        vednnTensorParam_t *pParamIn;
        vednnTensorParam_t *pParamOut;
        vednnBiasParam_t   *pParamBias;
        vednnFilterParam_t *pParamKernel;
        vednnConvolutionParam_t *pParamConv;
        void *pDataIn;
        void *pDataOut;
        void *pDataBias;
        void *pDataKernel;

        void *pBufRef;
        float *pBufOne;
        float *pBufCol;

        unsigned long long cycle;
        char region[128];
    };

    int i;

    struct conv *pConvBuff = NULL;

    pConvBuff = (struct conv *) malloc(nEntry * sizeof(struct conv));
    if (pConvBuff == NULL) {
        fprintf(stderr, "Memory exhaust.\n");
        exit(1);
    }

    for (i=0; i<nEntry; i++) {
        struct conv *pConv = &pConvBuff[i];
        struct param *pNw = &pNetwork[i];

        pConv->pParamIn		= NULL;
        pConv->pParamOut	= NULL;
        pConv->pParamBias	= NULL;
        pConv->pParamKernel	= NULL;
        pConv->pParamConv	= NULL;
        pConv->pDataIn		= NULL;
        pConv->pDataOut		= NULL;
        pConv->pDataBias	= NULL;
        pConv->pDataKernel	= NULL;
        pConv->pDataKernel	= NULL;

        pConv->pBufRef		= NULL;
        pConv->pBufOne		= NULL;
        pConv->pBufCol		= NULL;

        pConv->cycle		= 0;

        strcpy(pConv->region, pNw->pName);
    }

    for (i=0; i<nEntry; i++) {
        struct conv *pConv = &pConvBuff[i];
        struct param *pNw = &pNetwork[i];

        vednnError_t rv[5];
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

        inChannelGroup	= pNw->inChannel  / pNw->group;
        outChannelGroup	= pNw->outChannel / pNw->group;

        rv[0] = createTensorParam(&pConv->pParamIn, DTYPE_FLOAT, pNw->batchNum, pNw->inChannel,  pNw->inWidth,  pNw->inHeight);
        rv[1] = createTensorParam(&pConv->pParamOut, DTYPE_FLOAT, pNw->batchNum, pNw->outChannel, pNw->outWidth, pNw->outHeight);
        if( flagBias ) {
          rv[2] = createBiasParam(&pConv->pParamBias, DTYPE_FLOAT, pNw->outChannel);
        }
        rv[3] = createKernelParam(&pConv->pParamKernel, DTYPE_FLOAT, filter_layout, inChannelGroup, outChannelGroup, pNw->kernWidth, pNw->kernHeight);
        rv[4] = createConvolutionParam(&pConv->pParamConv, pNw->group, pNw->strideWidth, pNw->strideHeight, pNw->padWidth, pNw->padHeight, 1, 1);
        if (rv[0] != VEDNN_SUCCESS || rv[1] != VEDNN_SUCCESS || ( flagBias && rv[2] != VEDNN_SUCCESS ) || rv[3] != VEDNN_SUCCESS || rv[4] != VEDNN_SUCCESS ) {
            fprintf(stderr, "Failed to create/initialize structure.\n");
            exit(1);
        }


	pConv->pDataIn     = malloc(getTensorSizeInByte(pConv->pParamIn  ));
	pConv->pDataOut    = malloc(getTensorSizeInByte(pConv->pParamOut ));
        if( flagBias ) {
          pConv->pDataBias   = malloc(getBiasSizeInByte(pConv->pParamBias));
        }
	pConv->pDataKernel = malloc(getKernelSizeInByte(pConv->pParamKernel) * pNw->group);
	if (pConv->pDataIn == NULL || pConv->pDataOut == NULL || ( flagBias && pConv->pDataBias == NULL ) || pConv->pDataKernel == NULL) {
	    fprintf(stderr, "Memory exhaust.\n");
	    exit(1);
	}

        pConv->pBufRef  = malloc(getTensorSizeInByte(pConv->pParamOut )) ;
        size_t pOnesize = pConv->pParamOut->width * pConv->pParamOut->height;
        pConv->pBufOne  = (float*) malloc( pOnesize * getTensorDataSize(pConv->pParamIn));
        size_t pColrows = pConv->pParamKernel->inChannel * pConv->pParamKernel->width * pConv->pParamKernel->height;
        size_t pColcols = pConv->pParamOut->width * pConv->pParamOut->height;
        pConv->pBufCol  = (float*) malloc(pColrows*pColcols*getTensorDataSize(pConv->pParamIn));


        memset(pConv->pDataIn,     0, getTensorSizeInByte(pConv->pParamIn));
        memset(pConv->pDataOut,    0, getTensorSizeInByte(pConv->pParamOut));
        if (flagBias ) {
          memset(pConv->pDataBias,   0, getBiasSizeInByte(pConv->pParamBias));
        }
        memset(pConv->pDataKernel, 0, getKernelSizeInByte(pConv->pParamKernel) * pNw->group);

        memset(pConv->pBufRef,    0, getTensorSizeInByte(pConv->pParamOut));
        for (int i=0; i<pOnesize; i++) {
          pConv->pBufOne[i] = 1.0f;
        }

        // Generate Data
        generateRandomData(getTensorDataType(pConv->pParamIn), getTensorSize(pConv->pParamIn), pConv->pDataIn);
        if( flagBias ) {
          generateRandomData(getBiasDataType(pConv->pParamBias), getBiasSize(pConv->pParamBias), pConv->pDataBias);
        }
        generateRandomData(getKernelDataType(pConv->pParamKernel), getKernelSize(pConv->pParamKernel) * pNw->group, pConv->pDataKernel);
    }

    // run test Convolution
    {
        vednnError_t rv;


#ifdef FTRACE
        ftrace_region_begin("all convolution");
#endif


	for (i=0; i<nEntry; i++) {
	    struct conv *pConv = &pConvBuff[i];

	    unsigned long long c[2];
	    c[0] = __cycle();

#ifdef FTRACE
	    ftrace_region_begin(pConv->region);
#endif
	      // Convolution
	    if ( flagBias ) {
	      rv = vednnConvolutionForwardAddBias(pConv->pParamIn, pConv->pDataIn, pConv->pParamKernel, pConv->pDataKernel, pConv->pParamBias, pConv->pDataBias,  pConv->pParamOut, pConv->pDataOut, pConv->pParamConv, VEDNN_CONV_ALGORITHM_DIRECT );
	    }
	    else {
	      rv = vednnConvolutionForward(pConv->pParamIn, pConv->pDataIn, pConv->pParamKernel, pConv->pDataKernel, pConv->pParamOut, pConv->pDataOut, pConv->pParamConv, VEDNN_CONV_ALGORITHM_DIRECT );
	    }
	    if (rv != VEDNN_SUCCESS) {
		fprintf(stderr, "convolution() failed.\n");
		exit(1);
	    }


#ifdef FTRACE
	    ftrace_region_end(pConv->region);
#endif
	    c[1] = __cycle();
	    pConv->cycle += c[1] - c[0];
	}


#ifdef FTRACE
        ftrace_region_end("all convolution");
#endif

    }

    // run Reference Convolution
    {
        vednnError_t rv;
	for (i=0; i<nEntry; i++) {
	    struct conv *pConv = &pConvBuff[i];
	    // Convolution
	    rv = convolution_forward_gemm(pConv->pParamIn, pConv->pDataIn, pConv->pParamKernel, pConv->pDataKernel, pConv->pParamBias, pConv->pDataBias,  pConv->pParamOut, pConv->pBufRef, pConv->pBufOne, pConv->pBufCol, pConv->pParamConv );
	    if (rv != VEDNN_SUCCESS) {
		fprintf(stderr, "convolution() failed.\n");
		exit(1);
	    }
	}
    }

    if (flagCSV) {
	printf ("# convolution name, batch, group, bottom channel, bottom height, bottom width, top channel, top width, top height, kernel height, kernel width, stride height, stride width, pad height, pad width, time(msec), DIFF\n");
    }

    for (i=0; i<nEntry; i++) {
        struct conv *pConv = &pConvBuff[i];
        struct param *pNw = &pNetwork[i];

	if (flagCSV) {
	    printf("%s, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d",
		   pNw->pName, pNw->batchNum, pNw->group,
		   pNw->inChannel, pNw->inHeight, pNw->inWidth,
		   pNw->outChannel, pNw->outHeight, pNw->outWidth,
		   pNw->kernHeight, pNw->kernWidth,
		   pNw->strideHeight, pNw->strideWidth,
		   pNw->padHeight, pNw->padWidth );
	} else {
	    printf("%-30s : batch %d group %d \tbottom %4d %3d %4d \ttop %4d %3d %4d \tkernel %2d %2d stride %d %d pad %d %d \t",
		   pNw->pName,
		   pNw->batchNum,
		   pNw->group,
		   pNw->inChannel, pNw->inHeight, pNw->inWidth,
		   pNw->outChannel, pNw->outHeight, pNw->outWidth,
		   pNw->kernHeight, pNw->kernWidth,
		   pNw->strideHeight, pNw->strideWidth,
		   pNw->padHeight, pNw->padWidth );
	}

	double diff = diffData(pConv->pParamOut, pConv->pDataOut, pConv->pBufRef);

	double time = pConv->cycle * 1.0e3 / HZ;
	if (flagCSV) {
	    printf(", %f", time);
	} else {
	    printf(" \tTIME = %8.3f msec", time);
	}

	if (flagCSV) {
	    printf(", %f", diff);
	} else {
	    printf(" DIFF = %f", diff);
	}
	printf("\n");
    }

    // release
    for (i=0; i<nEntry; i++) {
        struct conv *pConv = &pConvBuff[i];

        destroyTensorParam(pConv->pParamIn);
        destroyTensorParam(pConv->pParamOut);
        if( flagBias ) {
          destroyBiasParam(pConv->pParamBias);
        }
        destroyKernelParam(pConv->pParamKernel);
        free(pConv->pDataIn);
        free(pConv->pDataOut);
        if( flagBias ) {
          free(pConv->pDataBias);
        }
        free(pConv->pDataKernel);

        free(pConv->pBufRef);
        free(pConv->pBufOne);
        free(pConv->pBufCol);
    }

    free(pConvBuff);
}

void testBackwardData(struct param *pNetwork, int nEntry, double HZ, int flagCSV, filterLayout_t filter_layout)
{
    struct conv {
        vednnTensorParam_t *pParamGradOut;
        vednnTensorParam_t *pParamGradIn;
        vednnFilterParam_t *pParamKernel;
        vednnConvolutionParam_t *pParamConv;
        void *pDataGradOut;
        void *pDataGradIn;
        void *pDataKernel;

        void  *pBufRef;
        float *pBufCol;

        unsigned long long cycle;
        char region[128];
    };

    int i;

    struct conv *pConvBuff = NULL;

    pConvBuff = (struct conv *) malloc(nEntry * sizeof(struct conv));
    if (pConvBuff == NULL) {
        fprintf(stderr, "Memory exhaust.\n");
        exit(1);
    }

    for (i=0; i<nEntry; i++) {
        struct conv *pConv = &pConvBuff[i];
        struct param *pNw = &pNetwork[i];

        pConv->pParamGradOut	= NULL;
        pConv->pParamGradIn	= NULL;
        pConv->pParamKernel	= NULL;
        pConv->pParamConv	= NULL;
        pConv->pDataGradOut	= NULL;
        pConv->pDataGradIn	= NULL;
        pConv->pDataKernel	= NULL;

        pConv->pBufRef		= NULL;
        pConv->pBufCol		= NULL;

        pConv->cycle		= 0;

        strcpy(pConv->region, pNw->pName);
    }

    for (i=0; i<nEntry; i++) {
        struct conv *pConv = &pConvBuff[i];
        struct param *pNw  = &pNetwork[i];

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

        inChannelGroup	= pNw->inChannel  / pNw->group;
        outChannelGroup	= pNw->outChannel / pNw->group;

        rv[0] = createTensorParam(&pConv->pParamGradOut, DTYPE_FLOAT, pNw->batchNum, pNw->outChannel, pNw->outWidth, pNw->outHeight);
        rv[1] = createTensorParam(&pConv->pParamGradIn, DTYPE_FLOAT, pNw->batchNum, pNw->inChannel, pNw->inWidth, pNw->inHeight);
        rv[2] = createKernelParam(&pConv->pParamKernel, DTYPE_FLOAT, filter_layout, inChannelGroup, outChannelGroup, pNw->kernWidth, pNw->kernHeight);
        rv[3] = createConvolutionParam(&pConv->pParamConv, pNw->group, pNw->strideWidth, pNw->strideHeight, pNw->padWidth, pNw->padHeight, 1, 1);
        if (rv[0] != VEDNN_SUCCESS || rv[1] != VEDNN_SUCCESS || rv[2] != VEDNN_SUCCESS || rv[3] != VEDNN_SUCCESS ) {
            fprintf(stderr, "Failed to create/initialize structure.\n");
            exit(1);
        }


	pConv->pDataGradOut = malloc(getTensorSizeInByte(pConv->pParamGradOut  ));
	pConv->pDataGradIn= malloc(getTensorSizeInByte(pConv->pParamGradIn ));
	pConv->pDataKernel = malloc(getKernelSizeInByte(pConv->pParamKernel) * pNw->group);

        pConv->pBufRef  = malloc(getTensorSizeInByte(pConv->pParamGradIn )) ;
        size_t pColrows = pConv->pParamKernel->inChannel * pConv->pParamKernel->width * pConv->pParamKernel->height;
        size_t pColcols = pConv->pParamGradOut->width * pConv->pParamGradOut->height;
        pConv->pBufCol  = (float*) malloc(pColrows*pColcols*getTensorDataSize(pConv->pParamGradIn));


        memset(pConv->pDataGradOut,  0, getTensorSizeInByte(pConv->pParamGradOut));
        memset(pConv->pDataGradIn, 0, getTensorSizeInByte(pConv->pParamGradIn));
        memset(pConv->pDataKernel,  0, getKernelSizeInByte(pConv->pParamKernel) * pNw->group);

        memset(pConv->pBufRef,    0, getTensorSizeInByte(pConv->pParamGradIn));

        // Generate Data
        generateRandomData(getTensorDataType(pConv->pParamGradOut), getTensorSize(pConv->pParamGradOut), pConv->pDataGradOut);
        generateRandomData(getKernelDataType(pConv->pParamKernel), getKernelSize(pConv->pParamKernel) * pNw->group, pConv->pDataKernel);
        generateRandomData(getTensorDataType(pConv->pParamGradIn), getTensorSize(pConv->pParamGradIn), pConv->pDataGradIn);
    }

    // run test Convolution
    {
        vednnError_t rv;


#ifdef FTRACE
        ftrace_region_begin("all convolution");
#endif


	for (i=0; i<nEntry; i++) {
	    struct conv *pConv = &pConvBuff[i];

	    unsigned long long c[2];
	    c[0] = __cycle();

#ifdef FTRACE
	    ftrace_region_begin(pConv->region);
#endif
	    // Convolution
	    rv = vednnConvolutionBackwardData(pConv->pParamGradOut, pConv->pDataGradOut, pConv->pParamKernel, pConv->pDataKernel, pConv->pParamGradIn, pConv->pDataGradIn, pConv->pParamConv, VEDNN_CONV_ALGORITHM_DIRECT );
	    if (rv != VEDNN_SUCCESS) {
		fprintf(stderr, "convolution() failed.\n");
		exit(1);
	    }


#ifdef FTRACE
	    ftrace_region_end(pConv->region);
#endif
	    c[1] = __cycle();
	    pConv->cycle += c[1] - c[0];
	}


#ifdef FTRACE
        ftrace_region_end("all convolution");
#endif

    }

    // run Reference Convolution
    {
        vednnError_t rv;
	for (i=0; i<nEntry; i++) {
	    struct conv *pConv = &pConvBuff[i];
	    // Convolution
	    rv = convolution_backward_data_gemm(pConv->pParamGradOut, pConv->pDataGradOut, pConv->pParamKernel, pConv->pDataKernel, pConv->pParamGradIn, pConv->pBufRef, pConv->pBufCol, pConv->pParamConv );
	    if (rv != VEDNN_SUCCESS) {
		fprintf(stderr, "convolution() failed.\n");
		exit(1);
	    }
	}
    }

    if (flagCSV) {
	printf ("# convolution name, batch, group, bottom channel, bottom height, bottom width, top channel, top width, top height, kernel height, kernel width, stride height, stride width, pad height, pad width, time(msec), DIFF\n");
    }

    for (i=0; i<nEntry; i++) {
        struct conv *pConv = &pConvBuff[i];
        struct param *pNw = &pNetwork[i];

	if (flagCSV) {
	    printf("%s, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d",
		   pNw->pName, pNw->batchNum, pNw->group,
		   pNw->inChannel, pNw->inHeight, pNw->inWidth,
		   pNw->outChannel, pNw->outHeight, pNw->outWidth,
		   pNw->kernHeight, pNw->kernWidth,
		   pNw->strideHeight, pNw->strideWidth,
		   pNw->padHeight, pNw->padWidth );
	} else {
	    printf("%-30s : batch %d group %d \tbottom %4d %3d %4d \ttop %4d %3d %4d \tkernel %2d %2d stride %d %d pad %d %d \t",
		   pNw->pName,
		   pNw->batchNum,
		   pNw->group,
		   pNw->inChannel, pNw->inHeight, pNw->inWidth,
		   pNw->outChannel, pNw->outHeight, pNw->outWidth,
		   pNw->kernHeight, pNw->kernWidth,
		   pNw->strideHeight, pNw->strideWidth,
		   pNw->padHeight, pNw->padWidth );
	}

	double diff = diffData(pConv->pParamGradIn, pConv->pDataGradIn, pConv->pBufRef);

	double time = pConv->cycle * 1.0e3 / HZ;
	if (flagCSV) {
	    printf(", %f", time);
	} else {
	    printf(" \tTIME = %8.3f msec", time);
	}

	if (flagCSV) {
	    printf(", %f", diff);
	} else {
	    printf(" DIFF = %f", diff);
	}
	printf("\n");
    }

    // release
    for (i=0; i<nEntry; i++) {
        struct conv *pConv = &pConvBuff[i];

        destroyTensorParam(pConv->pParamGradOut);
        destroyTensorParam(pConv->pParamGradIn);
        destroyKernelParam(pConv->pParamKernel);
        free(pConv->pDataGradOut);
        free(pConv->pDataGradIn);
        free(pConv->pDataKernel);

        free(pConv->pBufRef);
        free(pConv->pBufCol);
    }

    free(pConvBuff);
}

void testBackwardFilter(struct param *pNetwork, int nEntry, double HZ, int flagCSV, \
        filterLayout_t filter_layout)
{
    struct conv {
        vednnTensorParam_t *pParamIn;
        vednnTensorParam_t *pParamGradOut;
        vednnFilterParam_t *pParamGradKernel;
        vednnConvolutionParam_t *pParamConv;
        void *pDataIn;
        void *pDataGradOut;
        void *pDataGradKernel;

        void  *pBufRef;
        float *pBufCol;

        unsigned long long cycle;
        char region[128];
    };

    int i;

    struct conv *pConvBuff = NULL;

    pConvBuff = (struct conv *) malloc(nEntry * sizeof(struct conv));
    if (pConvBuff == NULL) {
        fprintf(stderr, "Memory exhaust.\n");
        exit(1);
    }

    for (i=0; i<nEntry; i++) {
        struct conv *pConv = &pConvBuff[i];
        struct param *pNw = &pNetwork[i];

        pConv->pParamGradOut	= NULL;
        pConv->pParamIn	= NULL;
        pConv->pParamGradKernel	= NULL;
        pConv->pParamConv	= NULL;
        pConv->pDataGradOut	= NULL;
        pConv->pDataIn	= NULL;
        pConv->pDataGradKernel	= NULL;

        pConv->pBufRef		= NULL;
        pConv->pBufCol		= NULL;

        pConv->cycle		= 0;

        strcpy(pConv->region, pNw->pName);
    }

    for (i=0; i<nEntry; i++) {
        struct conv *pConv = &pConvBuff[i];
        struct param *pNw  = &pNetwork[i];

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

        inChannelGroup	= pNw->inChannel  / pNw->group;
        outChannelGroup	= pNw->outChannel / pNw->group;

        rv[0] = createTensorParam(&pConv->pParamIn, DTYPE_FLOAT, pNw->batchNum, pNw->inChannel, pNw->inWidth, pNw->inHeight);
        rv[1] = createTensorParam(&pConv->pParamGradOut, DTYPE_FLOAT, pNw->batchNum, pNw->outChannel, pNw->outWidth, pNw->outHeight);
        rv[2] = createKernelParam(&pConv->pParamGradKernel, DTYPE_FLOAT, filter_layout, inChannelGroup, outChannelGroup, pNw->kernWidth, pNw->kernHeight);
        rv[3] = createConvolutionParam(&pConv->pParamConv, pNw->group, pNw->strideWidth, pNw->strideHeight, pNw->padWidth, pNw->padHeight, 1, 1);
        if (rv[0] != VEDNN_SUCCESS || rv[1] != VEDNN_SUCCESS || rv[2] != VEDNN_SUCCESS || rv[3] != VEDNN_SUCCESS ) {
            fprintf(stderr, "Failed to create/initialize structure.\n");
            exit(1);
        }


	pConv->pDataIn= malloc(getTensorSizeInByte(pConv->pParamIn ));
	pConv->pDataGradOut = malloc(getTensorSizeInByte(pConv->pParamGradOut  ));
	pConv->pDataGradKernel = malloc(getKernelSizeInByte(pConv->pParamGradKernel) * pNw->group);

        pConv->pBufRef  = malloc(getKernelSizeInByte(pConv->pParamGradKernel) * pNw->group);
        size_t pColrows = pConv->pParamGradKernel->inChannel * pConv->pParamGradKernel->width * pConv->pParamGradKernel->height;
        size_t pColcols = pConv->pParamGradOut->width * pConv->pParamGradOut->height;
        pConv->pBufCol  = (float*) malloc(pColrows*pColcols*getTensorDataSize(pConv->pParamIn));

        memset(pConv->pDataIn, 0, getTensorSizeInByte(pConv->pParamIn));
        memset(pConv->pDataGradOut,  0, getTensorSizeInByte(pConv->pParamGradOut));
        memset(pConv->pDataGradKernel,  0, getKernelSizeInByte(pConv->pParamGradKernel) * pNw->group);

        memset(pConv->pBufRef,  0, getKernelSizeInByte(pConv->pParamGradKernel) * pNw->group);

        // Generate Data
        generateRandomData(getTensorDataType(pConv->pParamIn), getTensorSize(pConv->pParamIn), pConv->pDataIn);
        generateRandomData(getTensorDataType(pConv->pParamGradOut), getTensorSize(pConv->pParamGradOut), pConv->pDataGradOut);
    }

    // run test Convolution
    {
        vednnError_t rv;

#ifdef FTRACE
        ftrace_region_begin("all convolution");
#endif


	for (i=0; i<nEntry; i++) {
	    struct conv *pConv = &pConvBuff[i];

	    unsigned long long c[2];
	    c[0] = __cycle();

#ifdef FTRACE
	    ftrace_region_begin(pConv->region);
#endif
	    // Convolution
	    rv = vednnConvolutionBackwardFilter(pConv->pParamIn, pConv->pDataIn, pConv->pParamGradOut, pConv->pDataGradOut, pConv->pParamGradKernel, pConv->pDataGradKernel, pConv->pParamConv, VEDNN_CONV_ALGORITHM_DIRECT );
	    if (rv != VEDNN_SUCCESS) {
		fprintf(stderr, "convolution() failed.\n");
		exit(1);
	    }


#ifdef FTRACE
	    ftrace_region_end(pConv->region);
#endif
	    c[1] = __cycle();
	    pConv->cycle += c[1] - c[0];
	}


#ifdef FTRACE
        ftrace_region_end("all convolution");
#endif

    }

    // run Reference Convolution
    {
        vednnError_t rv;
	for (i=0; i<nEntry; i++) {
	    struct conv *pConv = &pConvBuff[i];
	    // Convolution
	    rv = convolution_backward_filter_gemm(pConv->pParamIn, pConv->pDataIn, pConv->pParamGradOut, pConv->pDataGradOut, pConv->pParamGradKernel, pConv->pBufRef, pConv->pBufCol, pConv->pParamConv );
	    if (rv != VEDNN_SUCCESS) {
		fprintf(stderr, "convolution() failed.\n");
		exit(1);
	    }
	}
    }


    if (flagCSV) {
	printf ("# convolution name, batch, group, bottom channel, bottom height, bottom width, top channel, top width, top height, kernel height, kernel width, stride height, stride width, pad height, pad width, time(msec), DIFF\n");
    }

    for (i=0; i<nEntry; i++) {
        struct conv *pConv = &pConvBuff[i];
        struct param *pNw = &pNetwork[i];

	if (flagCSV) {
	    printf("%s, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d",
		   pNw->pName, pNw->batchNum, pNw->group,
		   pNw->inChannel, pNw->inHeight, pNw->inWidth,
		   pNw->outChannel, pNw->outHeight, pNw->outWidth,
		   pNw->kernHeight, pNw->kernWidth,
		   pNw->strideHeight, pNw->strideWidth,
		   pNw->padHeight, pNw->padWidth );
	} else {
	    printf("%-30s : batch %d group %d \tbottom %4d %3d %4d \ttop %4d %3d %4d \tkernel %2d %2d stride %d %d pad %d %d \t",
		   pNw->pName,
		   pNw->batchNum,
		   pNw->group,
		   pNw->inChannel, pNw->inHeight, pNw->inWidth,
		   pNw->outChannel, pNw->outHeight, pNw->outWidth,
		   pNw->kernHeight, pNw->kernWidth,
		   pNw->strideHeight, pNw->strideWidth,
		   pNw->padHeight, pNw->padWidth );
	}

	double diff = diffFilter(pConv->pParamGradKernel, pConv->pDataGradKernel, pConv->pBufRef);

	double time = pConv->cycle * 1.0e3 / HZ;
	if (flagCSV) {
	    printf(", %f", time);
	} else {
	    printf(" \tTIME = %8.3f msec", time);
	}

	if (flagCSV) {
	    printf(", %f", diff);
	} else {
	    printf(" DIFF = %f", diff);
	}
	printf("\n");

    }

    // release
    for (i=0; i<nEntry; i++) {
        struct conv *pConv = &pConvBuff[i];

        destroyTensorParam(pConv->pParamIn);
        destroyTensorParam(pConv->pParamGradOut);
        destroyKernelParam(pConv->pParamGradKernel);
        free(pConv->pDataIn);
        free(pConv->pDataGradOut);
        free(pConv->pDataGradKernel);

        free(pConv->pBufRef);
        free(pConv->pBufCol);
    }

    free(pConvBuff);
}


int readParamFile(struct param **ppParams, const char *pParamPath )
{
  struct param *pParams  = NULL ;
  int nParams            = 0 ;

  FILE *fp = fopen(pParamPath, "r") ;
  if( fp == NULL ) {
    fprintf(stderr, "Cannot open parameter file : %s .\n", pParamPath );
    exit(1);
  }

  fscanf(fp, "%d\n", &nParams) ;
  if( nParams <= 0 ) {
    fprintf(stderr, "Parameter file read error.\n");
    fclose(fp) ;
    exit(1);
  }
  pParams = (struct param *) malloc(sizeof(struct param) * nParams) ;

  for(int i=0; i<nParams; i++) {

    fscanf(fp, "%[^,],%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d\n",
	pParams[i].pName,
	&(pParams[i].batchNum),
	&(pParams[i].group),
	&(pParams[i].inChannel),
	&(pParams[i].inHeight),
	&(pParams[i].inWidth),
	&(pParams[i].outChannel),
	&(pParams[i].outHeight),
	&(pParams[i].outWidth),
	&(pParams[i].kernHeight),
	&(pParams[i].kernWidth),
	&(pParams[i].strideHeight),
	&(pParams[i].strideWidth),
	&(pParams[i].padHeight),
	&(pParams[i].padWidth) ) ;
  }
  fclose(fp) ;

  *ppParams = pParams ;
  return nParams ;
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
    { "ConvForward",	    CONV_TEST_FORWARD } ,
    { "ConvForwardAddBias",	CONV_TEST_FORWARD_ADDBIAS } ,
    { "ConvBackwardData",	CONV_TEST_BACKWARD_DATA } ,
    { "ConvBackwardFilter", CONV_TEST_BACKWARD_FILTER } ,
//    { "ConvBackwardBias", CONV_TEST_BACKWARD_BIAS } ,   // not implemented
};

static struct {
    char *pName;
    filterLayout_t   layouttype;
} filterLayout[] = {
    { "filter_nchw",	VEDNN_FILTER_LAYOUT_NCHW } ,
    { "filter_hwcn",	VEDNN_FILTER_LAYOUT_HWCN }
};

int main(int argc, char **argv)
{
    extern int optind;
    extern char *optarg;
    int opt;

    char *pParamPath = NULL ;
    double HZ        = 0.0 ;
    int testtype     = 0 ;
    filterLayout_t filter_layout= 0 ;
    int flagCSV	     = 0 ;

    int flagNoMkConsistent = 0 ;

    while ((opt = getopt(argc, argv, "p:CH:T:nf:")) != -1) {
        switch (opt) {
        case 'p':
          pParamPath = optarg;
          break;
        case 'C':	flagCSV = 1;		break;
        case 'H':	HZ = atof(optarg);	break;
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
        case 'n':	flagNoMkConsistent = 1;		break;
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
	  break ;
        default: /* '?' */
            fprintf(stderr, "Unknown option.\n");
            exit(1);
        }
    }
    if (optind < argc) {
        fprintf(stderr, "Expected argument after options\n");
        exit(1);
    }
    if ( pParamPath == NULL ) {
      fprintf(stderr, "Parameter file must be specified by '-p' option.\n");
      exit(1);
    }
    if (HZ <= 0.0) {
        fprintf(stderr, "Processor core frequency must be set by '-H' option.\n");
        exit(1);
    }
    printf("TEST TYPE        = %s\n",	   tests[testtype].pName) ;
    printf("FILTER LAYOUT    = %s\n",	   filterLayout[filter_layout].pName) ;
    printf("STM FREQUENCY    = %.3e HZ\n", HZ);
    printf("PARAMETER FILE   = %s\n",      pParamPath);


    struct param *pParams ;
    int nParams = readParamFile( &pParams, pParamPath ) ;
    if( nParams <= 0 ) {
      exit(1);
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
        if(!flagNoMkConsistent) mkConsistent( &pParams[i] );
    }

    switch(testtype) {
    case CONV_TEST_FORWARD :
      testForward(pParams, nParams, HZ, 0, flagCSV, filter_layout);
      break ;
    case CONV_TEST_FORWARD_ADDBIAS :
      testForward(pParams, nParams, HZ, 1, flagCSV, filter_layout);
      break ;
    case CONV_TEST_BACKWARD_DATA :
      testBackwardData(pParams, nParams, HZ, flagCSV, filter_layout);
      break ;
    case CONV_TEST_BACKWARD_FILTER :
      testBackwardFilter(pParams, nParams, HZ, flagCSV, filter_layout);
      break ;
    default :
      break ;
    }


    return 0;
}


// vim: et ts=4 sw=4 cindent cino=^=l0,\:0,N-s syntax=cpp.doxygen
