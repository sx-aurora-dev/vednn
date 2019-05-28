
#include <vednn.h>
#include "vednn_helper.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>
#include <math.h>

#ifdef FTRACE
#include <ftrace.h>
#endif

#include "timer.h"

vednnError_t linear_forward_gemm(
    const int64_t			inDim,
    const int64_t			outDim,
    const int64_t			nBatch,
    const void * restrict		*pDataIn,
    const void * restrict		*pDataWeight,
    void * restrict			*pDataOut
) ;

vednnError_t linear_backward_data_gemm(
    const int64_t			inDim,
    const int64_t			outDim,
    const int64_t			nBatch,
    const void * restrict		*pDataGradOut,
    const void * restrict		*pDataWeight,
    void * restrict			*pDataGradIn
) ;

vednnError_t linear_backward_weight_gemm(
    const int64_t			inDim,
    const int64_t			outDim,
    const int64_t			nBatch,
    const void * restrict		*pDataIn,
    const void * restrict		*pDataGradOut,
    void * restrict			*pDataGradWeight
) ;

struct param {
    const char	pName[256];
    int		inDim;
    int		outDim;
    int		nBatch;
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
diffData(size_t size, const void *pData, const void *pExpectedResult)
{
    int i;
    double sum = 0.0;

    const float *pData1 = (float *)pExpectedResult;
    const float *pData2 = (float *)pData;
    for (i=0; i<size; i++) {
       double diff = pData1[i] - pData2[i] ;
       sum += (diff/(fabs(pData1[i])+1e-7)) * (diff/(fabs(pData2[i])+1e-7)) ;
    }

    return sqrt(sum);
}


void testForward(struct param *pNetwork, int nEntry, double HZ, int flagBias, int flagCSV)
{
    struct linear {
        int   inDim;
        int   outDim;
        int   nBatch;

        void *pDataIn;
        void *pDataOut;
        // void *pDataBias;
        void *pDataWeight;

        void *pBufRef;

        unsigned long long cycle;
        char region[128];
    };

    int i;

    struct linear *pLinearBuff = NULL;

    pLinearBuff = (struct linear *) malloc(nEntry * sizeof(struct linear));
    if (pLinearBuff == NULL) {
        fprintf(stderr, "Memory exhaust.\n");
        exit(1);
    }

    for (i=0; i<nEntry; i++) {
        struct linear *pLinear = &pLinearBuff[i];
        struct param  *pNw     = &pNetwork[i];
        
        pLinear->inDim		= pNw->inDim;
        pLinear->outDim		= pNw->outDim;
        pLinear->nBatch		= pNw->nBatch;
        
	pLinear->pDataIn	= NULL;
        pLinear->pDataOut	= NULL;
        //pLinear->pDataBias	= NULL;
        pLinear->pDataWeight	= NULL;

        pLinear->pBufRef	= NULL;

        pLinear->cycle		= 0;

        strcpy(pLinear->region, pNw->pName);
    }

    for (i=0; i<nEntry; i++) {
        struct linear *pLinear = &pLinearBuff[i];

        pLinear->pDataIn     = malloc(sizeof(float)*pLinear->inDim*pLinear->nBatch);
        pLinear->pDataOut    = malloc(sizeof(float)*pLinear->outDim*pLinear->nBatch);
//        if( flagBias ) {
//          pLinear->pDataBias   = malloc(sizeof(float)*pLinear->outDim);
//        }
        pLinear->pDataWeight = malloc(sizeof(float)*pLinear->inDim*pLinear->outDim);
	if (pLinear->pDataIn == NULL || pLinear->pDataOut == NULL || /*( flagBias && pLinear->pDataBias == NULL ) ||*/  pLinear->pDataWeight == NULL) {
	    fprintf(stderr, "Memory exhaust.\n");
	    exit(1);
	}

	pLinear->pBufRef  = malloc(sizeof(float)*pLinear->outDim*pLinear->nBatch);

        memset(pLinear->pDataIn,     0, sizeof(float)*pLinear->inDim*pLinear->nBatch);
        memset(pLinear->pDataOut,    0, sizeof(float)*pLinear->outDim*pLinear->nBatch);
//        if (flagBias ) {
//          memset(pLinear->pDataBias,   0, sizeof(float)*pLinear->outDim);
//        }
        memset(pLinear->pDataWeight, 0, sizeof(float)*pLinear->inDim*pLinear->outDim );

        memset(pLinear->pBufRef,    0, sizeof(float)*pLinear->outDim*pLinear->nBatch);


        // Generate Data
        generateRandomData(DTYPE_FLOAT, pLinear->inDim*pLinear->nBatch, pLinear->pDataIn);
//        if( flagBias ) {
//          generateRandomData(DTYPE_FLOAT, pLinear->outDim, pLinear->pDataBias);
//        }
        generateRandomData(DTYPE_FLOAT, pLinear->inDim*pLinear->outDim, pLinear->pDataWeight);
    }

    // run test
    {
        vednnError_t rv;


#ifdef FTRACE
        ftrace_region_begin("all");
#endif


	for (i=0; i<nEntry; i++) {
	    struct linear *pLinear = &pLinearBuff[i];

	    unsigned long long c[2];
	    c[0] = __cycle();

#ifdef FTRACE
	    ftrace_region_begin(pLinear->region);
#endif
	      // Convolution
//	    if ( flagBias ) {
//	    }
//	    else
	    {
	      rv = vednnLinearForward(pLinear->inDim, pLinear->outDim, pLinear->nBatch,
				      pLinear->pDataIn, pLinear->pDataWeight, pLinear->pDataOut) ;
	    }
	    if (rv != VEDNN_SUCCESS) {
		fprintf(stderr, "vednnLinearForward() failed.\n");
		exit(1);
	    }


#ifdef FTRACE
	    ftrace_region_end(pLinear->region);
#endif
	    c[1] = __cycle();
	    pLinear->cycle += c[1] - c[0];
	}


#ifdef FTRACE
        ftrace_region_end("all");
#endif

    }

    // run Reference
    {
        vednnError_t rv;
	for (i=0; i<nEntry; i++) {
	    struct linear *pLinear = &pLinearBuff[i];
	    // Convolution
	    rv = linear_forward_gemm(pLinear->inDim, pLinear->outDim, pLinear->nBatch,
				      pLinear->pDataIn, pLinear->pDataWeight, pLinear->pBufRef) ;
	    if (rv != VEDNN_SUCCESS) {
		fprintf(stderr, "LinearForwardGemm() failed.\n");
		exit(1);
	    }
	}
    }

    if (flagCSV) {
	printf ("# name, batch, inDim, outDim, time(msec), DIFF\n");
    }

    for (i=0; i<nEntry; i++) {
	    struct linear *pLinear = &pLinearBuff[i];
        struct param *pNw = &pNetwork[i];

	if (flagCSV) {
	    printf("%s, %d, %d, %d",
		   pNw->pName, pNw->nBatch,
		   pNw->inDim, pNw->outDim );
	} else {
	    printf("%-30s : batch %d in %d out %d\t",
		   pNw->pName, pNw->nBatch,
		   pNw->inDim, pNw->outDim ) ;
	}

	double diff = diffData(pLinear->nBatch*pLinear->outDim, pLinear->pDataOut, pLinear->pBufRef);

	double time = pLinear->cycle * 1.0e3 / HZ;
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
        struct linear *pLinear = &pLinearBuff[i];


        free(pLinear->pDataIn);
        free(pLinear->pDataOut);
//        if( flagBias ) {
//          free(pLinear->pDataBias);
//        }
        free(pLinear->pDataWeight);

        free(pLinear->pBufRef);
    }

    free(pLinearBuff);
}


void testBackwardData(struct param *pNetwork, int nEntry, double HZ, int flagBias, int flagCSV)
{
    struct linear {
        int   inDim;
        int   outDim;
        int   nBatch;

        void *pDataGradOut;
        void *pDataWeight;
        void *pDataGradIn;

        void *pBufRef;

        unsigned long long cycle;
        char region[128];
    };

    int i;

    struct linear *pLinearBuff = NULL;

    pLinearBuff = (struct linear *) malloc(nEntry * sizeof(struct linear));
    if (pLinearBuff == NULL) {
        fprintf(stderr, "Memory exhaust.\n");
        exit(1);
    }

    for (i=0; i<nEntry; i++) {
        struct linear *pLinear = &pLinearBuff[i];
        struct param  *pNw     = &pNetwork[i];

        pLinear->inDim		= pNw->inDim;
        pLinear->outDim		= pNw->outDim;
        pLinear->nBatch		= pNw->nBatch;

        pLinear->pDataGradOut	= NULL;
        pLinear->pDataWeight	= NULL;
	pLinear->pDataGradIn	= NULL;

        pLinear->pBufRef	= NULL;

        pLinear->cycle		= 0;

        strcpy(pLinear->region, pNw->pName);
    }

    for (i=0; i<nEntry; i++) {
        struct linear *pLinear = &pLinearBuff[i];
        pLinear->pDataGradOut = malloc(sizeof(float)*pLinear->outDim*pLinear->nBatch);
        pLinear->pDataWeight  = malloc(sizeof(float)*pLinear->inDim*pLinear->outDim);
        pLinear->pDataGradIn  = malloc(sizeof(float)*pLinear->inDim*pLinear->nBatch);
	if (pLinear->pDataGradOut == NULL || pLinear->pDataGradIn == NULL || pLinear->pDataWeight == NULL) {
	    fprintf(stderr, "Memory exhaust.\n");
	    exit(1);
	}

	pLinear->pBufRef  = malloc(sizeof(float)*pLinear->inDim*pLinear->nBatch);

        memset(pLinear->pDataGradOut, 0, sizeof(float)*pLinear->outDim*pLinear->nBatch);
        memset(pLinear->pDataWeight,  0, sizeof(float)*pLinear->inDim*pLinear->outDim );
        memset(pLinear->pDataGradIn,  0, sizeof(float)*pLinear->inDim*pLinear->nBatch);

        memset(pLinear->pBufRef,    0, sizeof(float)*pLinear->inDim*pLinear->nBatch);


        // Generate Data
        generateRandomData(DTYPE_FLOAT, pLinear->outDim*pLinear->nBatch, pLinear->pDataGradOut);
        generateRandomData(DTYPE_FLOAT, pLinear->inDim*pLinear->outDim, pLinear->pDataWeight);
    }

    // run test
    {
        vednnError_t rv;

#ifdef FTRACE
        ftrace_region_begin("all");
#endif


	for (i=0; i<nEntry; i++) {
	    struct linear *pLinear = &pLinearBuff[i];

	    unsigned long long c[2];
	    c[0] = __cycle();

#ifdef FTRACE
	    ftrace_region_begin(pLinear->region);
#endif

	    rv = vednnLinearBackwardData(pLinear->inDim, pLinear->outDim, pLinear->nBatch,
					 pLinear->pDataGradOut, pLinear->pDataWeight, pLinear->pDataGradIn) ;

	    if (rv != VEDNN_SUCCESS) {
		fprintf(stderr, "vednnLinearBackwardData() failed.\n");
		exit(1);
	    }


#ifdef FTRACE
	    ftrace_region_end(pLinear->region);
#endif
	    c[1] = __cycle();
	    pLinear->cycle += c[1] - c[0];
	}


#ifdef FTRACE
        ftrace_region_end("all");
#endif

    }

    // run Reference
    {
        vednnError_t rv;
	for (i=0; i<nEntry; i++) {
	    struct linear *pLinear = &pLinearBuff[i];
	    // Convolution
	    rv = linear_backward_data_gemm(pLinear->inDim, pLinear->outDim, pLinear->nBatch,
				           pLinear->pDataGradOut, pLinear->pDataWeight, pLinear->pBufRef) ;
	    if (rv != VEDNN_SUCCESS) {
		fprintf(stderr, "LinearBackwardDataGemm() failed.\n");
		exit(1);
	    }
	}
    }

    if (flagCSV) {
	printf ("# name, batch, inDim, outDim, time(msec), DIFF\n");
    }

    for (i=0; i<nEntry; i++) {
	    struct linear *pLinear = &pLinearBuff[i];
        struct param *pNw = &pNetwork[i];

	if (flagCSV) {
	    printf("%s, %d, %d, %d",
		   pNw->pName, pNw->nBatch,
		   pNw->inDim, pNw->outDim );
	} else {
	    printf("%-30s : batch %d in %d out %d\t",
		   pNw->pName, pNw->nBatch,
		   pNw->inDim, pNw->outDim ) ;
	}

	double diff = diffData(pLinear->nBatch*pLinear->inDim, pLinear->pDataGradIn, pLinear->pBufRef);

	double time = pLinear->cycle * 1.0e3 / HZ;
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
        struct linear *pLinear = &pLinearBuff[i];


        free(pLinear->pDataGradOut);
        free(pLinear->pDataWeight);
        free(pLinear->pDataGradIn);

        free(pLinear->pBufRef);
    }

    free(pLinearBuff);
}

void testBackwardWeight(struct param *pNetwork, int nEntry, double HZ, int flagBias, int flagCSV)
{
    struct linear {
        int   inDim;
        int   outDim;
        int   nBatch;

        void *pDataGradOut;
        void *pDataIn;
        void *pDataGradWeight;

        void *pBufRef;

        unsigned long long cycle;
        char region[128];
    };

    int i;

    struct linear *pLinearBuff = NULL;

    pLinearBuff = (struct linear *) malloc(nEntry * sizeof(struct linear));
    if (pLinearBuff == NULL) {
        fprintf(stderr, "Memory exhaust.\n");
        exit(1);
    }

    for (i=0; i<nEntry; i++) {
        struct linear *pLinear = &pLinearBuff[i];
        struct param  *pNw     = &pNetwork[i];

        pLinear->inDim		= pNw->inDim;
        pLinear->outDim		= pNw->outDim;
        pLinear->nBatch		= pNw->nBatch;

        pLinear->pDataGradOut	= NULL;
	pLinear->pDataIn	= NULL;
        pLinear->pDataGradWeight= NULL;

        pLinear->pBufRef	= NULL;

        pLinear->cycle		= 0;

        strcpy(pLinear->region, pNw->pName);
    }

    for (i=0; i<nEntry; i++) {
        struct linear *pLinear = &pLinearBuff[i];
        pLinear->pDataGradOut     = malloc(sizeof(float)*pLinear->outDim*pLinear->nBatch);
        pLinear->pDataIn          = malloc(sizeof(float)*pLinear->inDim*pLinear->nBatch);
        pLinear->pDataGradWeight  = malloc(sizeof(float)*pLinear->inDim*pLinear->outDim);
	if (pLinear->pDataGradOut == NULL || pLinear->pDataIn == NULL || pLinear->pDataGradWeight == NULL) {
	    fprintf(stderr, "Memory exhaust.\n");
	    exit(1);
	}

	pLinear->pBufRef  = malloc(sizeof(float)*pLinear->inDim*pLinear->outDim);

        memset(pLinear->pDataGradOut,     0, sizeof(float)*pLinear->outDim*pLinear->nBatch);
        memset(pLinear->pDataIn,          0, sizeof(float)*pLinear->inDim*pLinear->nBatch);
        memset(pLinear->pDataGradWeight,  0, sizeof(float)*pLinear->inDim*pLinear->outDim);

        memset(pLinear->pBufRef,          0, sizeof(float)*pLinear->inDim*pLinear->outDim);

        // Generate Data
        generateRandomData(DTYPE_FLOAT, pLinear->outDim*pLinear->nBatch, pLinear->pDataGradOut);
        generateRandomData(DTYPE_FLOAT, pLinear->inDim*pLinear->nBatch,  pLinear->pDataIn);
    }

    // run test
    {
        vednnError_t rv;

#ifdef FTRACE
        ftrace_region_begin("all");
#endif


	for (i=0; i<nEntry; i++) {
	    struct linear *pLinear = &pLinearBuff[i];

	    unsigned long long c[2];
	    c[0] = __cycle();

#ifdef FTRACE
	    ftrace_region_begin(pLinear->region);
#endif

	    rv = vednnLinearBackwardWeight(pLinear->inDim, pLinear->outDim, pLinear->nBatch,
		                           pLinear->pDataIn, pLinear->pDataGradOut, pLinear->pDataGradWeight) ;

	    if (rv != VEDNN_SUCCESS) {
		fprintf(stderr, "vednnLinearBackwardWeight() failed.\n");
		exit(1);
	    }


#ifdef FTRACE
	    ftrace_region_end(pLinear->region);
#endif
	    c[1] = __cycle();
	    pLinear->cycle += c[1] - c[0];
	}


#ifdef FTRACE
        ftrace_region_end("all");
#endif

    }

    // run Reference
    {
        vednnError_t rv;
	for (i=0; i<nEntry; i++) {
	    struct linear *pLinear = &pLinearBuff[i];
	    // Convolution
	    rv = linear_backward_weight_gemm(pLinear->inDim, pLinear->outDim, pLinear->nBatch,
              		                   pLinear->pDataIn, pLinear->pDataGradOut, pLinear->pBufRef ) ;
	    if (rv != VEDNN_SUCCESS) {
		fprintf(stderr, "LinearBackwardWeightGemm() failed.\n");
		exit(1);
	    }
	}
    }

    if (flagCSV) {
	printf ("# name, batch, inDim, outDim, time(msec), DIFF\n");
    }

    for (i=0; i<nEntry; i++) {
	    struct linear *pLinear = &pLinearBuff[i];
        struct param *pNw = &pNetwork[i];

	if (flagCSV) {
	    printf("%s, %d, %d, %d",
		   pNw->pName, pNw->nBatch,
		   pNw->inDim, pNw->outDim );
	} else {
	    printf("%-30s : batch %d in %d out %d\t",
		   pNw->pName, pNw->nBatch,
		   pNw->inDim, pNw->outDim ) ;
	}

	double diff = diffData(pLinear->outDim*pLinear->inDim, pLinear->pDataGradWeight, pLinear->pBufRef);

	double time = pLinear->cycle * 1.0e3 / HZ;
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
        struct linear *pLinear = &pLinearBuff[i];


        free(pLinear->pDataGradOut);
        free(pLinear->pDataIn);
        free(pLinear->pDataGradWeight);

        free(pLinear->pBufRef);
    }

    free(pLinearBuff);
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

    fscanf(fp, "%[^,],%d,%d,%d\n",
	pParams[i].pName,
	&(pParams[i].nBatch),
	&(pParams[i].inDim),
	&(pParams[i].outDim) ) ;
  }
  fclose(fp) ;

  *ppParams = pParams ;
  return nParams ;
}

enum {
  LINEAR_FORWARD = 0,
  LINEAR_BACKWARD_DATA,
  LINEAR_BACKWARD_WEIGHT,
} ;


static struct {
    char *pName;
    int   testtype;
} tests[] = {
    { "LinearForward",		LINEAR_FORWARD } ,
    { "LinearBackwardData",	LINEAR_BACKWARD_DATA } ,
    { "LinearBackwardWeight",	LINEAR_BACKWARD_WEIGHT } ,
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
    case LINEAR_FORWARD :
      testForward(pParams, nParams, HZ, 0, flagCSV);
      break ;
    case LINEAR_BACKWARD_DATA :
      testBackwardData(pParams, nParams, HZ, 0, flagCSV);
      break ;
    case LINEAR_BACKWARD_WEIGHT :
      testBackwardWeight(pParams, nParams, HZ, 0, flagCSV);
      break ;
    default :
      break ;
    }


    return 0;
}



