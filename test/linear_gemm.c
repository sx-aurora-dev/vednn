/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#include <vednn.h>
#include "vednn_helper.h"

#define SGEMM	sgemm_
void sgemm_(char *TRANSA, char *TRANSB, int *M, int *N, int *K,
            float *ALPHA, float *A,  int *LDA, float *B, int *LDB,
            float *BETA, float *C, int *LDC ) ;

static char  TRANS   = 'T';
static char  NOTRANS = 'N';
static float FONE    = 1.0f;
static float FZERO   = 0.0f;

/* ----------------------------------------------------------------------- */

vednnError_t linear_forward_gemm(
    const int64_t			inDim,
    const int64_t			outDim,
    const int64_t			nBatch,
    const void * restrict		*pDataIn,
    const void * restrict		*pDataWeight,
    void * restrict			*pDataOut
)
{

    int M = outDim ;
    int N = nBatch ;
    int K = inDim ;
    int LDA = M ;
    int LDB = K ;
    int LDC = M ;

    SGEMM(&NOTRANS, &NOTRANS, &M, &N, &K,
	  &FONE, (void*)pDataWeight, &LDA, (void*)pDataIn, &LDB,
	  &FZERO, (void*)pDataOut, &LDC ) ;

    return VEDNN_SUCCESS;
}


vednnError_t linear_backward_data_gemm(
    const int64_t			inDim,
    const int64_t			outDim,
    const int64_t			nBatch,
    const void * restrict		*pDataGradOut,
    const void * restrict		*pDataWeight,
    void * restrict			*pDataGradIn
)
{

    int M = inDim ;
    int N = nBatch ;
    int K = outDim ;
    int LDA = K ;
    int LDB = K ;
    int LDC = M ;

    SGEMM(&TRANS, &NOTRANS, &M, &N, &K,
	  &FONE, (void*)pDataWeight, &LDA, (void*)pDataGradOut, &LDB,
	  &FZERO, (void*)pDataGradIn, &LDC ) ;

    return VEDNN_SUCCESS;
}

vednnError_t linear_backward_weight_gemm(
    const int64_t			inDim,
    const int64_t			outDim,
    const int64_t			nBatch,
    const void * restrict		*pDataIn,
    const void * restrict		*pDataGradOut,
    void * restrict			*pDataGradWeight
)
{

    int M = outDim ;
    int N = inDim ;
    int K = nBatch ;
    int LDA = M ;
    int LDB = N ;
    int LDC = M ;

    SGEMM(&NOTRANS, &TRANS, &M, &N, &K,
	  &FONE, (void*)pDataGradOut, &LDA, (void*)pDataIn, &LDB,
	  &FZERO, (void*)pDataGradWeight, &LDC ) ;

    return VEDNN_SUCCESS;
}

