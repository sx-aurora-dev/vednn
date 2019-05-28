
#include <vednn.h>
#include "vednn_helper.h"

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

vednnError_t maxpooling_forward(
    const vednnTensorParam_t 		*pParamIn,
    const void 				*pDataIn,
    const vednnTensorParam_t 		*pParamOut,
    void 				*pDataOut,
    const vednnPoolingParam_t		*pParamPool
) ;

vednnError_t maxpooling_backward(
    const vednnTensorParam_t 		*pParamGradOut,
    const void 				*pDataGradOut,
    const vednnTensorParam_t 		*pParamOut,
    const void 				*pDataOut,
    const vednnTensorParam_t 		*pParamIn,
    const void 				*pDataIn,
    const vednnTensorParam_t 		*pParamGradIn,
    void 				*pDataGradIn,
    const vednnPoolingParam_t		*pParamPool
) ;

struct param {
    const char	pName[256];
    int		batchNum;
    int		inChannel;
    int		inHeight;
    int		inWidth;
    int		outChannel;
    int		outHeight;
    int		outWidth;
    int		windowHeight;
    int		windowWidth;
    int		strideHeight;
    int		strideWidth;
    int		padHeight;
    int		padWidth;
};

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
                double diff = pData1[i] - pData2[i] ;
                sum += fabs(diff) ;
            }
        }
        break;
    default:
        assert(0);              /* BUG */
        break;
    }

    return sum ;
}


void testForward(struct param *pNetwork, int nEntry, double HZ, int flagCSV)
{
    struct pool {
        vednnTensorParam_t *pParamIn;
        vednnTensorParam_t *pParamOut;
        vednnPoolingParam_t *pParamPool;
        void *pDataIn;
        void *pDataOut;

        void *pBufRef;

        unsigned long long cycle;
        char region[128];
    };

    int i;

    struct pool *pPoolBuff = NULL;

    pPoolBuff = (struct pool *) malloc(nEntry * sizeof(struct pool));
    if (pPoolBuff == NULL) {
        fprintf(stderr, "Memory exhaust.\n");
        exit(1);
    }

    for (i=0; i<nEntry; i++) {
        struct pool *pPool = &pPoolBuff[i];
        struct param *pNw  = &pNetwork[i];

        pPool->pParamIn		= NULL;
        pPool->pParamOut	= NULL;
        pPool->pParamPool	= NULL;
        pPool->pDataIn		= NULL;
        pPool->pDataOut		= NULL;

        pPool->pBufRef		= NULL;

        pPool->cycle		= 0;

        strcpy(pPool->region, pNw->pName);
    }

    for (i=0; i<nEntry; i++) {
        struct pool  *pPool = &pPoolBuff[i];
        struct param *pNw   = &pNetwork[i];

        vednnError_t rv[5];

        rv[0] = createTensorParam(&pPool->pParamIn, DTYPE_FLOAT, pNw->batchNum, pNw->inChannel,  pNw->inWidth,  pNw->inHeight);
        rv[1] = createTensorParam(&pPool->pParamOut, DTYPE_FLOAT, pNw->batchNum, pNw->outChannel, pNw->outWidth, pNw->outHeight);
        rv[2] = createPoolingParam(&pPool->pParamPool, pNw->windowWidth, pNw->windowHeight, pNw->strideWidth, pNw->strideHeight, pNw->padWidth, pNw->padHeight);
        if (rv[0] != VEDNN_SUCCESS || rv[1] != VEDNN_SUCCESS || rv[2] != VEDNN_SUCCESS ) {
            fprintf(stderr, "Failed to create/initialize structure.\n");
            exit(1);
        }

	pPool->pDataIn    = malloc(getTensorSizeInByte(pPool->pParamIn  ));
	pPool->pDataOut   = malloc(getTensorSizeInByte(pPool->pParamOut ));
        pPool->pBufRef    = malloc(getTensorSizeInByte(pPool->pParamOut )) ;
	if (pPool->pDataIn == NULL || pPool->pDataOut == NULL || pPool->pBufRef == NULL) {
	    fprintf(stderr, "Memory exhaust.\n");
	    exit(1);
	}

        memset(pPool->pDataIn,     0, getTensorSizeInByte(pPool->pParamIn));
        memset(pPool->pDataOut,    0, getTensorSizeInByte(pPool->pParamOut));
        memset(pPool->pBufRef,     0, getTensorSizeInByte(pPool->pParamOut));

        // Generate Data
        generateRandomData(getTensorDataType(pPool->pParamIn), getTensorSize(pPool->pParamIn), pPool->pDataIn);
   }

    // run test pooling
    {
        vednnError_t rv;


#ifdef FTRACE
        ftrace_region_begin("all pooling");
#endif


	for (i=0; i<nEntry; i++) {
	    struct pool *pPool = &pPoolBuff[i];

	    unsigned long long c[2];
	    c[0] = __cycle();

#ifdef FTRACE
	    ftrace_region_begin(pPool->region);
#endif
	    // Pooling Forward
	    rv = vednnMaxPoolingForward(pPool->pParamIn, pPool->pDataIn,  pPool->pParamOut, pPool->pDataOut, pPool->pParamPool) ;
	    if (rv != VEDNN_SUCCESS) {
		fprintf(stderr, "pooling() failed.\n");
		exit(1);
	    }


#ifdef FTRACE
	    ftrace_region_end(pPool->region);
#endif
	    c[1] = __cycle();
	    pPool->cycle += c[1] - c[0];
	}


#ifdef FTRACE
        ftrace_region_end("all pooling");
#endif

    }

    // run Reference Pooling
    {
        vednnError_t rv;
	for (i=0; i<nEntry; i++) {
	    struct pool *pPool = &pPoolBuff[i];
	    // Forward
	    rv = maxpooling_forward(pPool->pParamIn, pPool->pDataIn,  pPool->pParamOut, pPool->pBufRef, pPool->pParamPool) ;
	    if (rv != VEDNN_SUCCESS) {
		fprintf(stderr, "pooling() failed.\n");
		exit(1);
	    }
	}
    }

    if (flagCSV) {
	printf ("# name, batch, bottom channel, bottom height, bottom width, top channel, top width, top height, window height, window width, stride height, stride width, pad height, pad width, time(msec), DIFF\n");
    }

    for (i=0; i<nEntry; i++) {
        struct pool *pPool = &pPoolBuff[i];
        struct param *pNw  = &pNetwork[i];

	if (flagCSV) {
	    printf("%s, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d",
		   pNw->pName, pNw->batchNum,
		   pNw->inChannel, pNw->inHeight, pNw->inWidth,
		   pNw->outChannel, pNw->outHeight, pNw->outWidth,
		   pNw->windowHeight, pNw->windowWidth,
		   pNw->strideHeight, pNw->strideWidth,
		   pNw->padHeight, pNw->padWidth );
	} else {
	    printf("%-30s : batch %d \tbottom %4d %3d %4d \ttop %4d %3d %4d \twindow %2d %2d stride %d %d pad %d %d \t",
		   pNw->pName,
		   pNw->batchNum,
		   pNw->inChannel, pNw->inHeight, pNw->inWidth,
		   pNw->outChannel, pNw->outHeight, pNw->outWidth,
		   pNw->windowHeight, pNw->windowWidth,
		   pNw->strideHeight, pNw->strideWidth,
		   pNw->padHeight, pNw->padWidth );
	}

	double diff = diffData(pPool->pParamOut, pPool->pDataOut, pPool->pBufRef);

	double time = pPool->cycle * 1.0e3 / HZ;
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
        struct pool *pPool = &pPoolBuff[i];

        destroyTensorParam(pPool->pParamIn);
        destroyTensorParam(pPool->pParamOut);
        destroyPoolingParam(pPool->pParamPool) ;

        free(pPool->pDataIn);
        free(pPool->pDataOut);
        free(pPool->pBufRef);
    }

    free(pPoolBuff);
}


void testBackward(struct param *pNetwork, int nEntry, double HZ, int flagCSV)
{
    struct pool {
        vednnTensorParam_t *pParamGradOut;
        vednnTensorParam_t *pParamOut;
        vednnTensorParam_t *pParamIn;
        vednnTensorParam_t *pParamGradIn;
        vednnPoolingParam_t *pParamPool;
        void *pDataGradOut;
        void *pDataOut;
        void *pDataIn;
        void *pDataGradIn;

        void *pBufRef;

        unsigned long long cycle;
        char region[128];
    };

    int i;

    struct pool *pPoolBuff = NULL;

    pPoolBuff = (struct pool *) malloc(nEntry * sizeof(struct pool));
    if (pPoolBuff == NULL) {
        fprintf(stderr, "Memory exhaust.\n");
        exit(1);
    }

    for (i=0; i<nEntry; i++) {
        struct pool *pPool = &pPoolBuff[i];
        struct param *pNw  = &pNetwork[i];

        pPool->pParamGradOut	= NULL;
        pPool->pParamOut	= NULL;
        pPool->pParamIn		= NULL;
        pPool->pParamGradIn	= NULL;
        pPool->pParamPool	= NULL;
        pPool->pDataGradOut	= NULL;
        pPool->pDataOut		= NULL;
        pPool->pDataIn		= NULL;
        pPool->pDataGradIn	= NULL;

        pPool->pBufRef		= NULL;

        pPool->cycle		= 0;

        strcpy(pPool->region, pNw->pName);
    }

    for (i=0; i<nEntry; i++) {
        struct pool  *pPool = &pPoolBuff[i];
        struct param *pNw   = &pNetwork[i];

        vednnError_t rv[5];

        rv[0] = createTensorParam(&pPool->pParamGradOut, DTYPE_FLOAT, pNw->batchNum, pNw->outChannel, pNw->outWidth, pNw->outHeight);
        rv[1] = createTensorParam(&pPool->pParamOut, DTYPE_FLOAT, pNw->batchNum, pNw->outChannel, pNw->outWidth, pNw->outHeight);
        rv[2] = createTensorParam(&pPool->pParamIn, DTYPE_FLOAT, pNw->batchNum, pNw->inChannel,  pNw->inWidth,  pNw->inHeight);
        rv[3] = createTensorParam(&pPool->pParamGradIn, DTYPE_FLOAT, pNw->batchNum, pNw->inChannel,  pNw->inWidth,  pNw->inHeight);
        rv[4] = createPoolingParam(&pPool->pParamPool, pNw->windowWidth, pNw->windowHeight, pNw->strideWidth, pNw->strideHeight, pNw->padWidth, pNw->padHeight);
        if (rv[0] != VEDNN_SUCCESS || rv[1] != VEDNN_SUCCESS || rv[2] != VEDNN_SUCCESS || rv[3] != VEDNN_SUCCESS || rv[4] != VEDNN_SUCCESS ) {
            fprintf(stderr, "Failed to create/initialize structure.\n");
            exit(1);
        }

	pPool->pDataGradOut = malloc(getTensorSizeInByte(pPool->pParamGradOut ));
	pPool->pDataOut     = malloc(getTensorSizeInByte(pPool->pParamOut ));
	pPool->pDataIn      = malloc(getTensorSizeInByte(pPool->pParamIn  ));
	pPool->pDataGradIn  = malloc(getTensorSizeInByte(pPool->pParamGradIn ));
        pPool->pBufRef      = malloc(getTensorSizeInByte(pPool->pParamGradIn )) ;
	if (pPool->pDataOut == NULL || pPool->pDataGradOut == NULL || pPool->pDataIn == NULL || pPool->pDataGradIn == NULL || pPool->pBufRef == NULL) {
	    fprintf(stderr, "Memory exhaust.\n");
	    exit(1);
	}

        memset(pPool->pDataGradIn, 0, getTensorSizeInByte(pPool->pParamGradIn));
        memset(pPool->pBufRef,     0, getTensorSizeInByte(pPool->pParamGradIn));

        // Generate Data
        generateRandomData(getTensorDataType(pPool->pParamGradOut), getTensorSize(pPool->pParamGradOut), pPool->pDataGradOut);
        generateRandomData(getTensorDataType(pPool->pParamIn), getTensorSize(pPool->pParamIn), pPool->pDataIn) ;

        maxpooling_forward(pPool->pParamIn, pPool->pDataIn,  pPool->pParamOut, pPool->pDataOut, pPool->pParamPool) ;
   }

    // run test Pooling
    {
        vednnError_t rv;


#ifdef FTRACE
        ftrace_region_begin("all pooling");
#endif


	for (i=0; i<nEntry; i++) {
	    struct pool *pPool = &pPoolBuff[i];

	    unsigned long long c[2];
	    c[0] = __cycle();

#ifdef FTRACE
	    ftrace_region_begin(pPool->region);
#endif
	    // Pooling Backward
	    rv = vednnMaxPoolingBackward(pPool->pParamGradOut, pPool->pDataGradOut,
					 pPool->pParamOut,     pPool->pDataOut,
			                 pPool->pParamIn,      pPool->pDataIn,
					 pPool->pParamGradIn,  pPool->pDataGradIn,
					 pPool->pParamPool) ;
	    if (rv != VEDNN_SUCCESS) {
		fprintf(stderr, "pooling() failed.\n");
		exit(1);
	    }


#ifdef FTRACE
	    ftrace_region_end(pPool->region);
#endif
	    c[1] = __cycle();
	    pPool->cycle += c[1] - c[0];
	}


#ifdef FTRACE
        ftrace_region_end("all pooling");
#endif

    }

    // run Reference Pooling
    {
        vednnError_t rv;
	for (i=0; i<nEntry; i++) {
	    struct pool *pPool = &pPoolBuff[i];
	    // Backward
	    rv = maxpooling_backward(pPool->pParamGradOut, pPool->pDataGradOut,
			             pPool->pParamOut,     pPool->pDataOut,
				     pPool->pParamIn,      pPool->pDataIn,
				     pPool->pParamGradIn,  pPool->pBufRef,
				     pPool->pParamPool) ;
	    if (rv != VEDNN_SUCCESS) {
		fprintf(stderr, "pooling() failed.\n");
		exit(1);
	    }
	}
    }

    if (flagCSV) {
	printf ("# name, batch, bottom channel, bottom height, bottom width, top channel, top width, top height, window height, window width, stride height, stride width, pad height, pad width, time(msec), DIFF\n");
    }

    for (i=0; i<nEntry; i++) {
        struct pool *pPool = &pPoolBuff[i];
        struct param *pNw  = &pNetwork[i];

	if (flagCSV) {
	    printf("%s, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d",
		   pNw->pName, pNw->batchNum,
		   pNw->inChannel, pNw->inHeight, pNw->inWidth,
		   pNw->outChannel, pNw->outHeight, pNw->outWidth,
		   pNw->windowHeight, pNw->windowWidth,
		   pNw->strideHeight, pNw->strideWidth,
		   pNw->padHeight, pNw->padWidth );
	} else {
	    printf("%-30s : batch %d \tbottom %4d %3d %4d \ttop %4d %3d %4d \twindow %2d %2d stride %d %d pad %d %d \t",
		   pNw->pName,
		   pNw->batchNum,
		   pNw->inChannel, pNw->inHeight, pNw->inWidth,
		   pNw->outChannel, pNw->outHeight, pNw->outWidth,
		   pNw->windowHeight, pNw->windowWidth,
		   pNw->strideHeight, pNw->strideWidth,
		   pNw->padHeight, pNw->padWidth );
	}

	double diff = diffData(pPool->pParamGradIn, pPool->pDataGradIn, pPool->pBufRef);

	double time = pPool->cycle * 1.0e3 / HZ;
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
        struct pool *pPool = &pPoolBuff[i];

        destroyTensorParam(pPool->pParamGradOut);
        destroyTensorParam(pPool->pParamOut);
        destroyTensorParam(pPool->pParamIn);
        destroyTensorParam(pPool->pParamGradIn);

        destroyPoolingParam(pPool->pParamPool) ;

        free(pPool->pDataGradOut);
        free(pPool->pDataOut);
        free(pPool->pDataIn);
        free(pPool->pDataGradIn);
        free(pPool->pBufRef);
    }

    free(pPoolBuff);
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

    fscanf(fp, "%[^,],%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d\n",
	pParams[i].pName,
	&(pParams[i].batchNum),
	&(pParams[i].inChannel),
	&(pParams[i].inHeight),
	&(pParams[i].inWidth),
	&(pParams[i].outChannel),
	&(pParams[i].outHeight),
	&(pParams[i].outWidth),
	&(pParams[i].windowHeight),
	&(pParams[i].windowWidth),
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
  MAX_POOL_FORWARD = 0,
  MAX_POOL_BACKWARD
} ;


static struct {
    char *pName;
    int   testtype;
} tests[] = {
    { "MaxPoolForward",		MAX_POOL_FORWARD } ,
    { "MaxPoolBackward",	MAX_POOL_BACKWARD } ,
 };

int main(int argc, char **argv)
{
    extern int optind;
    extern char *optarg;
    int opt;

    char *pParamPath = NULL ;
    double HZ        = 0.0 ;
    int testtype     = 0 ;
    int flagCSV	     = 0 ;

    while ((opt = getopt(argc, argv, "p:CH:T:")) != -1) {
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
    printf("TEST TYPE        = %s\n",	 tests[testtype].pName) ;
    printf("STM FREQUENCY    = %.3e HZ\n", HZ);
    printf("PARAMETER FILE   = %s\n",      pParamPath);


    struct param *pParams ;
    int nParams = readParamFile( &pParams, pParamPath ) ;
    if( nParams <= 0 ) {
      exit(1);
    }

    switch(testtype) {
    case MAX_POOL_FORWARD :
      testForward(pParams, nParams, HZ, flagCSV);
      break ;
    case MAX_POOL_BACKWARD :
      testBackward(pParams, nParams, HZ, flagCSV);
      break ;
    default :
      break ;
    }


    return 0;
}



