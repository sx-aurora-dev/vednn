
#include "vednnConvolutionLists.h"

#include "vednnx.h"

#ifdef VEDNN_USE_OPENMP
#include <omp.h>
#endif

#include <stddef.h> // NULL
#include <stdio.h> // fflush
#include <stdint.h> // int64_t

#ifdef __cplusplus
extern "C" {
#endif

#define IMPL_FNS(BASENAME,STR)    IMPL_FNS_(BASENAME,STR)
#define IMPL_RTFNS(BASENAME,STR)  IMPL_RTFNS_(BASENAME,STR)
#define JIT_FNS(BASENAME,STR)     JIT_FNS_(BASENAME,STR)

#define IMPL_FNS_(BASENAME,STR)   { BASENAME, #BASENAME, STR, BASENAME##_ok, NULL, NULL, NULL }
#define IMPL_RTFNS_(BASENAME,STR) { BASENAME, #BASENAME, STR, BASENAME##_ok, BASENAME##_rtok, NULL, NULL }
#define JIT_FNS_(BASENAME,STR)    { #BASENAME, STR, BASENAME##_ok, NULL, BASENAME##_pd, BASENAME##_jit }

// try to put most specialized first, because they are likely fastest
static vednnConvForwardImpls vednnConvForwardList_[] = {
    // k1 NOTE: dil1 is irrelevant for ker1 (should remove from name/files)
    //IMPL_FNS(vednnConvolutionForward_direct_dil1_str1_pad0_ker1_c1024x,"cnvFwd-d1s1p0k1c1024x"),
    // d1s1pS
    IMPL_FNS(vednnConvolutionForward_direct_dil1_str1_padsame_ker3_c1_owU128,"cnvFwd-d1s1pSk3c1owU128"),
    //IMPL_FNS(vednnConvolutionForward_direct_dil1_str1_padsame_ker3_c1_owU128A,"cnvFwd-d1s1pSk3c1owU128A"),
    IMPL_FNS(vednnConvolutionForward_direct_dil1_str1_padsame_ker3_c1,"cnvFwd-d1s1pSk3_c1"),
    //IMPL_FNS(vednnConvolutionForward_direct_dil1_str1_padsame_ker3_c1A,"cnvFwd-d1s1pSk3_c1A"),
    IMPL_FNS(vednnConvolutionForward_direct_dil1_str1_padsame_ker3_c1024x,"cnvFwd-d1s1pSk3c1024x"),
    //IMPL_FNS(vednnConvolutionForward_direct_dil1_str1_padsame_ker3_c1024xA,"cnvFwd-d1s1pSk3c1024xA"),
    IMPL_FNS(vednnConvolutionForward_direct_dil1_str1_padsame_ker3,"cnvFwd-d1s1pSk3"),
    //IMPL_FNS(vednnConvolutionForward_direct_dil1_str1_padsame_ker3A,"cnvFwd-d1s1pSk3A"),
    IMPL_FNS(vednnConvolutionForward_direct_dil1_str1_padsame_ker5_owU128,"cnvFwd-d1s1pSk5owU128"),
    //IMPL_FNS(vednnConvolutionForward_direct_dil1_str1_padsame_ker5_owU128A,"cnvFwd-d1s1pSk5owU128A"),
    IMPL_FNS(vednnConvolutionForward_direct_dil1_str1_padsame_ker5,"cnvFwd-d1s1pSk5"),
    //IMPL_FNS(vednnConvolutionForward_direct_dil1_str1_padsame_ker5A,"cnvFwd-d1s1pSk5A"),
    IMPL_FNS(vednnConvolutionForward_direct_dil1_str1_padsame_ker2,"cnvFwd-d1s1pSk2"),
    //IMPL_FNS(vednnConvolutionForward_direct_dil1_str1_padsame_ker2A,"cnvFwd-d1s1pSk2A"),
    IMPL_FNS(vednnConvolutionForward_direct_dil1_str1_padsame,"cnvFwd-d1s1pS"),
    // d1s1p0
    IMPL_RTFNS(vednnConvolutionForward_direct_dil1_str1_pad0_ker3_iw2XU256_ow2X_ioaligned,"cnvFwd-d1s1p0k3iw2XU256_ow2X_ioaligned"),
    IMPL_FNS(vednnConvolutionForward_direct_dil1_str1_pad0_owU128,"cnvFwd-d1s1p0_owU128"),
    //IMPL_FNS(vednnConvolutionForward_direct_dil1_str1_pad0_owU128A,"cnvFwd-d1s1p0_owU128A"),
    IMPL_FNS(vednnConvolutionForward_direct_dil1_str1_pad0,"cnvFwd-d1s1p0"),
    //IMPL_FNS(vednnConvolutionForward_direct_dil1_str1_pad0A,"cnvFwd-d1s1p0A"),
    // d1p0
    IMPL_FNS(vednnConvolutionForward_direct_dil1_pad0_owU128,"cnvFwd-d1p0_owU128"),
    //IMPL_FNS(vednnConvolutionForward_direct_dil1_pad0_owU128A,"cnvFwd-d1p0_owU128A"),
    IMPL_FNS(vednnConvolutionForward_direct_dil1_pad0,"cnvFwd-d1p0"),
    //IMPL_FNS(vednnConvolutionForward_direct_dil1_pad0A,"cnvFwd-d1p0A"),
	// generic libvednn
    IMPL_FNS(vednnConvolutionForward_direct_owU128,"cnvFwd-owU128"),
    //IMPL_FNS(vednnConvolutionForward_direct_owU128A,"cnvFwd-owU128A"),
    IMPL_FNS(vednnConvolutionForward_direct_vecC,"cnvFwd-vecC"),
    IMPL_FNS(vednnConvolutionForward_direct_default,"cnvFwd-def"),
    //IMPL_FNS(vednnConvolutionForward_direct_defaultA,"cnvFwd-defA"),
    // customizations (stable, working, but win in isolated circumstances)
    //IMPL_FNS(vednnConvolutionForward_direct_default2,"cnvFwd-def2"),
        //IMPL_FNS(vednnConvolutionForward_direct_default2p,"cnvFwd-def2p"),
    //IMPL_FNS(vednnConvolutionForward_direct_default3,"cnvFwd-def3"),
        //IMPL_FNS(vednnConvolutionForward_direct_default3b,"cnvFwd-df3b"),
        //IMPL_FNS(vednnConvolutionForward_direct_alt,"cnvFwd-alt"),
    //IMPL_FNS(vednnConvolutionForward_direct_dil1_str1_padsameA,"cnvFwd-d1s1pSA"), // XXX testing
    //IMPL_FNS(vednnConvolutionForward_direct_dil1_str1_padsameB,"cnvFwd-d1s1pSB"), // XXX testing
    //IMPL_FNS(vednnConvolutionForward_direct_dil1_str1_padsameAB,"cnvFwd-d1s1pSA+B"), // XXX testing
    // ker1
    IMPL_FNS(vednnConvolutionForward_direct_dil1_str1_pad0_ker1,"cnvFwd-s1p0k1"),
    //IMPL_FNS(vednnConvolutionForward_direct_dil1_str1_pad0_ker1A,"cnvFwd-s1p0k1A"),
    IMPL_FNS(vednnConvolutionForward_direct_dil1_pad0_owU128_ker1,"cnvFwd-p0k1_owU128"),
    //IMPL_FNS(vednnConvolutionForward_direct_dil1_pad0_owU128_ker1A,"cnvFwd-p0k1_owU128A"),
    // Problem: ker1 + nonequal height width buggy?  or else [d1] s1p0k1A maybe clobber output?
    // WRONG OUTPUT for mb1ih640iw360__ic128oc4__kh1___n"RNxt101-conv2a-ungrouped"
    //IMPL_FNS(vednnConvolutionForward_direct_dil1_pad0_ker1,"cnvFwd-p0k1"),
    //IMPL_FNS(vednnConvolutionForward_direct_dil1_pad0_ker1A,"cnvFwd-p0k1A"),
    {NULL,"NULL","null",NULL, NULL, NULL, NULL}
};

#if 0
static vednnConvForwardAddBiasImpls vednnConvForwardAddBiasList_[] = {
    IMPL_FNS(VEDNN_FUNC_CNVFWB(dil1_str1_pad0_ker1_c1024x),         "cnvFwB-d1s1p0k1c1024x"),
    IMPL_FNS(VEDNN_FUNC_CNVFWB(dil1_str1_pad0_ker1),                "cnvFwB-d1s1p0k1"),
    IMPL_FNS(VEDNN_FUNC_CNVFWB(dil1_str1_padsame_ker3_c1_owU128),   "cnvFwB-d1s1pSk3c1owU128"),
    IMPL_FNS(VEDNN_FUNC_CNVFWB(dil1_str1_padsame_ker3_c1024x),      "cnvFwB-d1s1pSk3_c1024x"),
    IMPL_FNS(VEDNN_FUNC_CNVFWB(dil1_str1_padsame_ker3_c1),          "cnvFwB-d1s1pSk3c1"),
    IMPL_FNS(VEDNN_FUNC_CNVFWB(dil1_str1_padsame_ker3),             "cnvFwB-d1s1pSk3"),
    IMPL_FNS(VEDNN_FUNC_CNVFWB(dil1_str1_padsame),                  "cnvFwB-d1s1pS"),
    IMPL_FNS(VEDNN_FUNC_CNVFWB(default),                            "cnvFwB-def"),
    IMPL_FNS(VEDNN_FUNC_CNVFWB(default2),                           "cnvFwB-def2"),
    {NULL}
};
#endif

static vednnConvBackwardDataImpls vednnConvBackwardDataList_[] = {
    IMPL_FNS(vednnConvolutionBackwardData_direct_dil1_str1_padsame,"cnvBkD-d1s1pS"), //1st?
    IMPL_RTFNS(vednnConvolutionBackwardData_direct_dil1_str1_pad0_ker3_iw2XU32_ow2X_ioaligned,
            "cnvBkD-d1s1p0k3iw2XU32_ow2X_ioaligned"),
    IMPL_RTFNS(vednnConvolutionBackwardData_direct_dil1_str1_pad0_ker3_iw2XU256_ow2X_ioaligned,
            "cnvBkD-d1s1p0k3iw2XU256_ow2X_ioaligned"),
    IMPL_FNS(vednnConvolutionBackwardData_direct_dil1_str1_pad0_ker3_iwU128,"cnvBkD-d1s1p0k3_iwU128"),
    IMPL_FNS(vednnConvolutionBackwardData_direct_dil1_str1_iwU128,"cnvBkD-d1s1_iwU128"),
    IMPL_FNS(vednnConvolutionBackwardData_direct_dil1_str1,"cnvBkD-d1s1"),
    IMPL_FNS(vednnConvolutionBackwardData_direct_iwU128,"cnvBkD-iwU128"),
    IMPL_FNS(vednnConvolutionBackwardData_direct_default,"cnvBkD-def"),
    //IMPL_FNS(vednnConvolutionBackwardData_direct_default2,"cnvBkD-def2"),
    {NULL}
};

static vednnConvBackwardFilterImpls vednnConvBackwardFilterList_[] = {
    IMPL_FNS(vednnConvolutionBackwardFilter_direct_dil1_str1_padsame_ker1,"cnvBkF-d1s1pSk1"),
    IMPL_FNS(vednnConvolutionBackwardFilter_direct_dil1_str1_padsame_ker2_owU128,"cnvBkF-d1s1pSk2_owU128"),
    IMPL_FNS(vednnConvolutionBackwardFilter_direct_dil1_str1_padsame_ker2,"cnvBkF-d1s1pSk2"),
    IMPL_FNS(vednnConvolutionBackwardFilter_direct_dil1_str1_padsame_ker3_ohwU256,"cnvBkF-d1s1pSk3_ohwU256"),
    IMPL_FNS(vednnConvolutionBackwardFilter_direct_dil1_str1_padsame_ker3_owU128,"cnvBkF-d1s1pSk3_owU128"),
    IMPL_FNS(vednnConvolutionBackwardFilter_direct_dil1_str1_padsame_ker3,"cnvBkF-d1s1pSk3"),
    IMPL_FNS(vednnConvolutionBackwardFilter_direct_dil1_str1_padsame_ker5_owU128,"cnvBkF-d1s1pSk5_owU128"),
    IMPL_FNS(vednnConvolutionBackwardFilter_direct_dil1_str1_padsame_ker5,"cnvBkF-d1s1pSk5"),
    IMPL_FNS(vednnConvolutionBackwardFilter_direct_dil1_str1_padsame,"cnvBkF-d1s1pS"),
    IMPL_RTFNS(vednnConvolutionBackwardFilter_direct_dil1_str1_pad0_ker3_ow2X_iw2XU256_igoaligned,
            "cnvBkFd1s1p0k3_ow2XU128_iw2X_igoaligned"),
    IMPL_FNS(vednnConvolutionBackwardFilter_direct_dil1_pad0_ker1_ohwU128,"cnvBkF-d1p0k1_ohwU128"),
    IMPL_FNS(vednnConvolutionBackwardFilter_direct_dil1_pad0_ker1_ohwU64,"cnvBkF-d1p0k1_ohwU64"),
    IMPL_FNS(vednnConvolutionBackwardFilter_direct_dil1_pad0_ker1_owU32,"cnvBkF-d1p0k1_owU32"),
    IMPL_FNS(vednnConvolutionBackwardFilter_direct_dil1_pad0_ker1,"cnvBkF-d1p0k1"),
    IMPL_FNS(vednnConvolutionBackwardFilter_direct_dil1_pad0_ker3_owU128,"cnvBkF-d1p0k3_owU128"),
    IMPL_FNS(vednnConvolutionBackwardFilter_direct_dil1_pad0_owU32,"cnvBkF-d1p0_owU32"),
    //IMPL_FNS(vednnConvolutionBackwardFilter_direct_dil1_pad0_owU128,"cnvBkF-d1p0_owU128"),
    IMPL_FNS(vednnConvolutionBackwardFilter_direct_dil1_pad0,"cnvBkF-d1p0"),
    IMPL_FNS(vednnConvolutionBackwardFilter_direct_owU128,"cnvBkF-owU128"),
    IMPL_FNS(vednnConvolutionBackwardFilter_direct_default,"cnvBkF-def"),
    {NULL}
};

vednnConvForwardImpls *        vednnConvForwardList        = &vednnConvForwardList_[0];
//vednnConvForwardAddBiasImpls * vednnConvForwardAddBiasList = &vednnConvForwardAddBiasList_[0];
vednnConvBackwardDataImpls *   vednnConvBackwardDataList   = &vednnConvBackwardDataList_[0];
vednnConvBackwardFilterImpls * vednnConvBackwardFilterList = &vednnConvBackwardFilterList_[0];

#define ITERATOR_BEGIN(Forward,FORWARD) \
vednnConv##Forward##Impls * vednnConv##Forward##_Begin( \
        VEDNN_PARAMS_CONV_##FORWARD ) \
{ \
    vednnConv##Forward##Impls *i = &vednnConv##Forward##List[0]; \
    while( i->okfn != NULL \
            && (i->okfn)(VEDNN_PARAMS_CONV_##FORWARD##_LIST) != VEDNN_SUCCESS){ \
        ++i; \
    } \
    return i; \
}
ITERATOR_BEGIN(Forward,        FORWARD);
//ITERATOR_BEGIN(ForwardAddBias, FORWARDADDBIAS);
ITERATOR_BEGIN(BackwardData,   BACKWARD_DATA);
ITERATOR_BEGIN(BackwardFilter, BACKWARD_FILTER);
#undef ITERATOR_BEGIN

#define ITERATOR_DUMP(Forward,FORWARD) \
void vednnConv##Forward##_Dump( vednnConv##Forward##Impls const* it) \
{ \
    printf(" impl@%-8p name=\"%s\" shortname=\"%s\" okfn@%-8p" \
            " rtokfn@%-8p getPd@%-8p getImpl@%-8p\n", \
            (void*)(it->impl), (it->name? it->name:"NULL"), (it->shortname? it->shortname:"NULL"), (void*)(it->okfn), \
            (void*)(it->rtokfn), (void*)(it->getPd), (void*)(it->getImpl) ); \
}
ITERATOR_DUMP(Forward,        FORWARD);
//ITERATOR_DUMP(ForwardAddBias, FORWARDADDBIAS);
ITERATOR_DUMP(BackwardData,   BACKWARD_DATA);
ITERATOR_DUMP(BackwardFilter, BACKWARD_FILTER);
#undef ITERATOR_DUMP

#define ITERATOR_NEXT_CAREFUL(current,Forward,FORWARD) do \
{ \
    vednnConv##Forward##Impls *i = &vednnConv##Forward##List[0]; \
    while( i != current && i->okfn != NULL ){ ++i; } /* MUST find current inside List */ \
    if( i->okfn != NULL ) \
        for( ++i; i->okfn != NULL \
            && (i->okfn)(VEDNN_PARAMS_CONV_##FORWARD##_LIST) != VEDNN_SUCCESS; \
            ) ++i; \
    current = i; \
}while(0)
#define ITERATOR_NEXT(Forward,FORWARD) \
vednnConv##Forward##Impls * vednnConv##Forward##_Next( \
        vednnConv##Forward##Impls* current, \
        VEDNN_PARAMS_CONV_##FORWARD ) \
{ \
    ITERATOR_NEXT_CAREFUL(current,Forward,FORWARD); \
    return current; \
}
ITERATOR_NEXT(Forward,        FORWARD);
//ITERATOR_NEXT(ForwardAddBias, FORWARDADDBIAS);
ITERATOR_NEXT(BackwardData,   BACKWARD_DATA);
ITERATOR_NEXT(BackwardFilter, BACKWARD_FILTER);

/** if we have _rtok fn and it says "no" advance to next impl compatible
 * with both _ok and _rtok fns */
#define ADVANCE_RTOK(current,Forward,FORWARD) do \
{ \
    if(current != NULL && current->rtokfn != NULL && \
            current->rtokfn(VEDNN_DATARG_CONV_##FORWARD##_LIST) \
            != VEDNN_SUCCESS) { \
        while((current = vednnConv##Forward##_Next(current, \
                        VEDNN_PARAMS_CONV_##FORWARD##_LIST)) != NULL){ \
            if(current->rtokfn == NULL \
                    || current->rtokfn(VEDNN_DATARG_CONV_##FORWARD##_LIST) \
                    == VEDNN_SUCCESS){ \
                break; \
            } \
        } \
    } \
}while(0)

#define REALNEXT(Forward,FORWARD) \
vednnConv##Forward##Impls * vednnConv##Forward##_realNext( \
        vednnConv##Forward##Impls* current, \
        VEDNN_PARAMS_CONV_##FORWARD, VEDNN_DATARG_CONV_##FORWARD ) \
{ \
    ADVANCE_RTOK(current,Forward,FORWARD); \
    return current; \
}
REALNEXT(Forward,        FORWARD);
//REALNEXT(ForwardAddBias, FORWARDADDBIAS);
REALNEXT(BackwardData,   BACKWARD_DATA);
REALNEXT(BackwardFilter, BACKWARD_FILTER);
#undef REALNEXT

#ifdef VEDNN_USE_OPENMP
#define GET_NTHREADS(nthreads) int64_t const nthreads = omp_get_max_threads() ;
    // NOTE: outside parallel region, use omp_get_max_threads
    //       inside                   use omp_get_num_threads (current #)
    //printf(" openmp threads %ld batch %ld",nthreads,allBatch); fflush(stdout);
#else
//#warning "no openmp?"
#define GET_NTHREADS(nthreads) int64_t const nthreads = 1;
    //printf(" threads %ld batch %ld",nthreads,allBatch); fflush(stdout);
#endif

/** based on \ref C/vednnConvolutionForward.c version */
    static inline vednnError_t
vednnConvolutionForward_wrapper(
        vednnConvForward_t pFunc,
        CONVX_FWD_ORDER( VEDNN_PARAMS_CONV_FORWARD, VEDNN_DATARG_CONV_FORWARD )
        )
{
    int64_t const allBatch = pParamIn->batch ;
    GET_NTHREADS(nthreads);
    if(nthreads==1 || allBatch==1){
        return pFunc( CONVX_FWD_ORDER(
                    VEDNN_PARAMS_CONV_FORWARD_LIST,
                    VEDNN_DATARG_CONV_FORWARD_LIST ));
    }
#ifdef VEDNN_USE_OPENMP
    vednnError_t rc = VEDNN_SUCCESS ;
#pragma omp parallel reduction(|:rc)
    {
        int64_t const threadid = omp_get_thread_num() ;

        int64_t const nBatch = allBatch / nthreads ;
        int64_t const remain = allBatch % nthreads ;

        int64_t const batchBegin = nBatch * threadid + ( threadid < remain ? threadid : remain ) ;
        int64_t const myBatch = nBatch + ( threadid < remain ? 1 : 0 ) ;

        if( myBatch == 0 ) {
            rc |= VEDNN_SUCCESS ;
        }
        else {
            vednnTensorParam_t _pParamIn  = *pParamIn  ; _pParamIn.batch = myBatch ;
            vednnTensorParam_t _pParamOut = *pParamOut ; _pParamOut.batch = myBatch ;
            float* _pDataIn  = ((float *)pDataIn) + batchBegin * pParamIn->channel * pParamIn->height * pParamIn->width ;
            float* _pDataOut = ((float *)pDataOut) + batchBegin * pParamOut->channel * pParamOut->height * pParamOut->width ;

            rc |= pFunc(&_pParamIn, (void*)_pDataIn, pParamKernel, pDataKernel,
                    pParamBias, pDataBias,
                    pParamConv, &_pParamOut, (void*) _pDataOut) ;
        }
    }
    return rc ;
#endif
}
#if 0
static inline vednnError_t
vednnConvolutionForwardAddBias_wrapper(
    vednnConvForwardAddBias_t		pFunc,
    CONVX_FWDB_ORDER( VEDNN_PARAMS_CONV_FORWARDADDBIAS, VEDNN_DATARG_CONV_FORWARDADDBIAS )
    )
{
    int64_t const allBatch = pParamIn->batch ;
    GET_NTHREADS(nthreads);
    if(nthreads==1 || allBatch==1){
        return pFunc( CONVX_FWDB_ORDER(
                    VEDNN_PARAMS_CONV_FORWARDADDBIAS_LIST,
                    VEDNN_DATARG_CONV_FORWARDADDBIAS_LIST ));
    }
#ifdef VEDNN_USE_OPENMP
  vednnError_t rc = VEDNN_SUCCESS ;
#pragma omp parallel reduction(|:rc)
  {
    int64_t nthreads = omp_get_num_threads() ;
    int64_t threadid = omp_get_thread_num() ;

    int64_t allBatch = pParamIn->batch ;

    int64_t nBatch = allBatch / nthreads ;
    int64_t remain = allBatch % nthreads ;

    int64_t batchBegin = nBatch * threadid + ( threadid < remain ? threadid : remain ) ;
    int64_t myBatch = nBatch + ( threadid < remain ? 1 : 0 ) ;

    if( myBatch == 0 ) {
      rc |= VEDNN_SUCCESS ;
    }
    else  {
      vednnTensorParam_t _pParamIn  = *pParamIn  ; _pParamIn.batch = myBatch ;
      vednnTensorParam_t _pParamOut = *pParamOut ; _pParamOut.batch = myBatch ;
      float* _pDataIn  = ((float *)pDataIn) + batchBegin * pParamIn->channel * pParamIn->height * pParamIn->width ;
      float* _pDataOut = ((float *)pDataOut) + batchBegin * pParamOut->channel * pParamOut->height * pParamOut->width ;

      rc |= pFunc(&_pParamIn, (void*)_pDataIn, pParamKernel, pDataKernel,
		  pParamBias, pDataBias, pParamConv, &_pParamOut, (void*) _pDataOut) ;
    }
  }
  return rc ;
#endif
}
#endif

/** \ref C/vednnConvolutionBackwardData.c (static inline openmp handling) */
static inline vednnError_t
vednnConvolutionBackwardData_wrapper(
    vednnConvBackwardData_t		pFunc,
    CONVX_BKWD_ORDER( VEDNN_PARAMS_CONV_BACKWARD_DATA, VEDNN_DATARG_CONV_BACKWARD_DATA )
)
{
    int64_t const allBatch = pParamGradOut->batch ;
    GET_NTHREADS(nthreads);
    if(nthreads==1 || allBatch==1){
        return pFunc( CONVX_BKWD_ORDER(
                    VEDNN_PARAMS_CONV_BACKWARD_DATA_LIST,
                    VEDNN_DATARG_CONV_BACKWARD_DATA_LIST ));
    }
#ifdef VEDNN_USE_OPENMP
  vednnError_t rc = VEDNN_SUCCESS ;
#pragma omp parallel reduction(|:rc)
  {
    int64_t nthreads = omp_get_num_threads() ;
    int64_t threadid = omp_get_thread_num() ;

    int64_t allBatch = pParamGradOut->batch ;

    int64_t nBatch = allBatch / nthreads ;
    int64_t remain = allBatch % nthreads ;

    int64_t batchBegin = nBatch * threadid + ( threadid < remain ? threadid : remain ) ;
    int64_t myBatch = nBatch + ( threadid < remain ? 1 : 0 ) ;

    if( myBatch == 0 ) {
      rc |= VEDNN_SUCCESS ;
    }
    else  {
      vednnTensorParam_t _pParamGradOut = *pParamGradOut; _pParamGradOut.batch = myBatch ;
      vednnTensorParam_t _pParamGradIn  = *pParamGradIn ; _pParamGradIn.batch = myBatch ;
      float* _pDataGradOut = ((float *)pDataGradOut) + batchBegin * pParamGradOut->channel * pParamGradOut->height * pParamGradOut->width ;
      float* _pDataGradIn  = ((float *)pDataGradIn) + batchBegin * pParamGradIn->channel * pParamGradIn->height * pParamGradIn->width ;

      rc |= pFunc(&_pParamGradOut, (void*)_pDataGradOut, pParamKernel, pDataKernel,
		  pParamConv, &_pParamGradIn, (void*) _pDataGradIn) ;
    }
  }
  return rc ;
#endif
}
/** \ref C/vednnConvolutionBackwardData.c (static inline openmp handling) */
static inline vednnError_t
vednnConvolutionBackwardFilter_wrapper(
    vednnConvBackwardFilter_t		pFunc,
    CONVX_BKWF_ORDER( VEDNN_PARAMS_CONV_BACKWARD_FILTER, VEDNN_DATARG_CONV_BACKWARD_FILTER )
)
{
    int64_t gOutChannelGroup;
    {
        int64_t gOutChannel = pParamGradOut->channel;
        int64_t group       = pParamConv->group;
        gOutChannelGroup    = gOutChannel  / group;
    }
    GET_NTHREADS(nthreads);
    if(nthreads==1 || gOutChannelGroup==1){
        int64_t beginOChannel = 0;
        int64_t nOChannel = gOutChannelGroup;
        return pFunc( CONVX_BKWF_ORDER(
                    VEDNN_PARAMS_CONV_BACKWARD_FILTER_LIST,
                    VEDNN_DATARG_CONV_BACKWARD_FILTER_LIST )
                VEDNN_DATARG_CONV_BACKWARD_FILTER_OPENMP_LIST
                );
    }
#ifdef VEDNN_USE_OPENMP
  vednnError_t rc = VEDNN_SUCCESS ;
#pragma omp parallel reduction(|:rc)
  {
    int64_t nthreads = omp_get_num_threads() ;
    int64_t threadid = omp_get_thread_num() ;

    int64_t gOutChannel = pParamGradOut->channel;
    int64_t group       = pParamConv->group;
    int64_t gOutChannelGroup = gOutChannel  / group;

    int64_t nOChannlel = gOutChannelGroup / nthreads ;
    int64_t remain     = gOutChannelGroup % nthreads ;

    int64_t beginOChannel = nOChannlel * threadid + ( threadid < remain ? threadid : remain ) ;
    int64_t myOChannel    = nOChannlel + ( threadid < remain ? 1 : 0 ) ;

    if( myOChannel == 0 ) {
      rc |= VEDNN_SUCCESS ;
    }
    else  {

      rc |= pFunc(pParamIn, pDataIn, pParamGradOut, pDataGradOut,
		  pParamConv, pParamGradKernel, pDataGradKernel,
		  beginOChannel, myOChannel );
    }
  }
  return rc ;
#endif
}

#define INVOKE_CONV_OPENMP_WRAPPER(ret,current,Forward,FWD,FORWARD) \
    vednnConv##Forward##_out_t ret = {current, VEDNN_ERROR_INVALID_PARAM}; \
{ \
    if(current != NULL) \
    /* invoke via _wrapper to do parallelization */ \
    ret.status = vednnConvolution##Forward##_wrapper( \
            current->impl, \
            CONVX_##FWD##_ORDER( \
                VEDNN_PARAMS_CONV_##FORWARD##_LIST, \
                VEDNN_DATARG_CONV_##FORWARD##_LIST )); \
}

#define CONV_RUN(Forward,FWD,FORWARD) \
vednnConv##Forward##_out_t vednnConv##Forward##_Run( \
        vednnConv##Forward##Impls* current, \
        VEDNN_PARAMS_CONV_##FORWARD, \
        VEDNN_DATARG_CONV_##FORWARD ) \
{ \
    ADVANCE_RTOK(current,Forward,FORWARD); \
    /* current PARAMS ok **and** DATA rtok, so can run it, if still non-NULL */ \
    INVOKE_CONV_OPENMP_WRAPPER(ret,current,Forward,FWD,FORWARD) \
    return ret; \
}
CONV_RUN(Forward,        FWD,  FORWARD)
//CONV_RUN(ForwardAddBias, FWDB, FORWARDADDBIAS)
CONV_RUN(BackwardData,   BKWD, BACKWARD_DATA)
CONV_RUN(BackwardFilter, BKWF, BACKWARD_FILTER)

#if 0
/// \group add _Begin and _Next functions to traverse lists of libvednn impls and helpers
//@{
//
/** find \b first [best? fastest?] {okfn, impl} in list such that \c okfn yields VEDNN_SUCCESS.
 * \return list terminator (all-NULLs) if none. */
vednnConvForwardImpls * vednnConvForward_Begin(
        VEDNN_PARAMS_CONV_FORWARD );
/** return next available implementation, after \c current list item.
 * return list terminator (all-NULLs) if none.
 * \pre \c current points within vednnConvForwardList[]
 */
vednnConvForwardImpls * vednnConvForward_Next(
        vednnConvForwardImpls* current,
        VEDNN_PARAMS_CONV_FORWARD );

vednnConvForwardAddBiasImpls * vednnConvForwardAddBias_Begin(
        VEDNN_PARAMS_CONV_FORWARDADDBIAS );
vednnConvForwardAddBiasImpls * vednnConvForwardAddBias_Next(
        vednnConvForwardAddBiasImpls* current,
        VEDNN_PARAMS_CONV_FORWARDADDBIAS );

vednnConvBackwardDataImpls * vednnConvBackwardData_Begin(
        VEDNN_PARAMS_CONV_BACKWARD_DATA );
vednnConvBackwardDataImpls * vednnConvBackwardData_Next(
        vednnConvBackwardDataImpls* current,
        VEDNN_PARAMS_CONV_BACKWARD_DATA );

vednnConvBackwardFilterImpls * vednnConvBackwardFilter_Begin(
        VEDNN_PARAMS_CONV_BACKWARD_FILTER );
vednnConvBackwardFilterImpls * vednnConvBackwardFilter_Next(
        vednnConvBackwardFilterImpls* current,
        VEDNN_PARAMS_CONV_BACKWARD_FILTER );
//@}
#endif
#if 0
/** \group add dynamic redirection of impls at runtime
 * Most runtime-checks of data parameters simply return the same
 * list element.  If data alignment is no good, then at runtime
 * we dispatch to an alternate impl */
//@{
//
/** \fn vednnConvForwardImpls * vednnConvForward_Begin(
        VEDNN_PARAMS_CONV_FORWARD );
 * find \b first [best? fastest?] {okfn, impl} in list
 * such that \c okfn yields VEDNN_SUCCESS.
 * \return list terminator (all-NULLs) if none. */

/** \fn vednnConvForwardImpls * vednnConvForward_Next(
 *          vednnConvForwardImpls* current,
 *          VEDNN_PARAMS_CONV_FORWARD )
 * Return next available implementation, after \c current list item.
 * return list terminator (all-NULLs) if none.
 * This impl is based ONLY on convolution parameters.
 * \pre \c current points within vednnConvForwardList[]
 */

/** \fn vednnConvForwardImpls * vednnConvForward_realNext(
        vednnConvForwardImpls* current,
        VEDNN_PARAMS_CONV_FORWARD,
        VEDNN_DATARG_CONV_FORWARD );
 * Runtime check of current + data for ptr alignment.
 * - Note: we assume Params have been checked
 *   - i.e. \c current was obtained from a _Begin or _Next
 *
 * - if data ptr alignment of \c current is ok, return \c current
 * - o/w iterate and return next impl which is ok with both params + data
 * - return list terminator (all-NULLs) if none.
 *
 * Typically we just return \c current, because there is no runtime
 * check on pointer alignment.
 *
 * Use this, instead of _Run when you need to grab the name of the
 * actual impl that was used, \e before actually executing a timing loop.
 *
 * \pre \c current points within vednnConvForwardList[]
 * \post it is really OK to run the returned list impl
 */

/** \fn vednnConvForward_out_t vednnConvForward_Run(
        vednnConvForwardImpls* current,
        VEDNN_PARAMS_CONV_FORWARD,
        VEDNN_DATARG_CONV_FORWARD );
 * runs current [if possible], else the next fully-compatible impl.
 * Assumes \c current is compatible with conv params,
 * updates \c current if there are data ptr alignment issues,
 * and then runs the operation.
 *
 * Always run \c current via a _Run call.
 *
 * You should never invoke \c current->impl directly, because this skips
 * some wrapperization that might do OpenMP parallelization.
 *
 * \return {actual,status}, where actual is usually current, except
 *         when a data ptr alignment check failed and a [next] actual
 *         impl had to run instead.
 */        

/** run-time dispatch, based on additional data-pointer checks.
 * Default behavior is simply <tt>return current</tt>.
 * o.w. we continue forward from \c current and return the
 * first impl that passes its runtime checks (if any).
 *
 * So instead of invoking (cfi=vednnConvForward_Next)->impl directly,
 * you do 2 steps:
 * ```
 * nxt = vednnConvForward_Begin(...); // or _Next(...)
 * nxt = vednnConvForward_rtNext(nxt,FWD_DATA); // usually rt_nxt == nxt
 * FTRACE_BEGIN(rt_nxt->name)
 * rt_nxt->impl( CONVX_FWDB_ORDER(FWD_PARMS,FWD_DATA) ); // or so (dep on bias)
 * FTRACE_END(rt_nxt->name)
 * ```
 * We require getting \c rt_nxt so that you can (say) FTRACE with a string of the
 * \e actual rt impl that ran.
 *
 * \c nxt is always precalculated based just on FWD_PARMS
 * \c rt_nxt now includes runtime ok-ness with FWD_DATA too.
 *
 * This is so a \e single layer might sometimes be called with correctly
 * aligned data, sometimes without, and still the 'best' impl gets run.
 */
vednnConvForwardImpls * vednnConvForward_rtok(
        vednnConvForwardImpls* current,
        VEDNN_DATARG_CONV_FORWARD );

//@}
#endif
#ifdef __cplusplus
}
#endif
// vim: et ts=4 sw=4 cindent cino=^=l0,\:0,N-s
