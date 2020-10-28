#include "vednnConvolutionForward.h"
#include "vednn-def.h"
#include <stdint.h>
#include <assert.h>
#include <stdio.h>

  static inline vednnError_t
vednnConvolutionForward_wrapper(
    vednnConvForward_t                  pFunc,
    VEDNN_CONVFWD_ARGS )
{
#ifndef VEDNN_USE_OPENMP
  return pFunc(VEDNN_CONVFWD_ARGS_LIST);
#else
  int64_t allBatch = pParamIn->batch; // check as in vednnx
  if (allBatch == 1 || __vednn_omp_num_threads == 1) {
    return pFunc(VEDNN_CONVFWD_ARGS_LIST);
  }else{
    vednnError_t rc = VEDNN_SUCCESS ;
#pragma omp parallel reduction(|:rc)
    {
      int64_t nthreads = omp_get_num_threads() ;
      int64_t threadid = omp_get_thread_num() ;

      int64_t nBatch = allBatch / nthreads ;
      int64_t remain = allBatch % nthreads ;

      int64_t batchBegin = nBatch * threadid + ( threadid < remain ? threadid : remain ) ;
      int64_t myBatch = nBatch + ( threadid < remain ? 1 : 0 ) ;

      if( myBatch == 0 ) {
        rc |= VEDNN_SUCCESS ;
      }else{
        vednnTensorParam_t _pParamIn  = *pParamIn  ; _pParamIn.batch = myBatch ;
        vednnTensorParam_t _pParamOut = *pParamOut ; _pParamOut.batch = myBatch ;
        float* _pDataIn  = ((float *)pDataIn) + batchBegin * pParamIn->channel * pParamIn->height * pParamIn->width ;
        float* _pDataOut = ((float *)pDataOut) + batchBegin * pParamOut->channel * pParamOut->height * pParamOut->width ;

        rc |= pFunc(&_pParamIn, (void*)_pDataIn, pParamKernel, pDataKernel,
            pParamBias, pDataBias, pParamConv, &_pParamOut, (void*) _pDataOut) ;
      }
    }
    return rc ;
  }
#endif
}

/* ----------------------------------------------------------------------- */
  static inline
vednnError_t vednnConvolutionForwardBody(
    const vednnTensorParam_t            *pParamIn,
    const void                          *pDataIn,
    const vednnFilterParam_t            *pParamKernel,
    const void                          *pDataKernel,
    const vednnBiasParam_t              *pParamBias,
    const void                          *pDataBias,
    const vednnTensorParam_t            *pParamOut,
    void                                *pDataOut,
    const vednnConvolutionParam_t       *pParamConv,
    vednnConvolutionAlgorithm_t         algo
    )
{
  switch( pParamKernel->layout ) {
  case VEDNN_FILTER_LAYOUT_NCHW :
  break ;
  case VEDNN_FILTER_LAYOUT_HWCN :
  if( pParamConv->group > 1 ) {
    fprintf(stderr, "[VEDNN ERROR] VEDNN does not support grouped convolution with filter_hwcn\n") ;
    return VEDNN_ERROR_INVALID_PARAM ;
  }
  break ;
  default :
  fprintf(stderr, "[VEDNN ERROR] Unknown Filter Layout %d\n", pParamKernel->layout) ;
  return VEDNN_ERROR_INVALID_PARAM ;
  }

  // normal ||ism over minibatch and group
#define OMPWRAP( IMPL ) WRAP_RET(vednnConvolutionForward_direct_##IMPL, \
    vednnConvolutionForward_wrapper, \
    VEDNN_CONVFWD_ARGS_LIST)

#ifdef VEDNN_ALT_PARALLEL
  // alternate ||ism handled internally via some other means
  // XXX public or private API?
  // These can use the private API !!!
#define ALT_RET( IMPL ) return vednnConvolutionForward_direct_##IMPL( \
    VEDNN_CONVFWD_ARGS_LIST )
#endif

  if (algo == VEDNN_CONV_ALGORITHM_DIRECT)
  {
#define DIL(N) (pParamConv->dilationHeight == (N) && pParamConv->dilationWidth == (N))
#define PAD(N) (pParamConv->padHeight == (N) && pParamConv->padWidth == (N))
#define STR(N) (pParamConv->strideHeight == (N) && pParamConv->strideWidth == (N))
#define KER(N) (pParamKernel->width == (N) && pParamKernel->height == (N))
#define IWU(N) (pParamIn->width <= (N))
#define OWU(N) (pParamOut->width <= (N))
#define ICoGU(N) (pParamIn->channel / pParamConv->group <= (N))
#define OCoGU(N) (pParamOut->channel / pParamConv->group <= (N))
    if ((pParamOut->height * pParamOut->width <= 16) ||
        ((pParamOut->height * pParamOut->width < 64)
         && (pParamOut->height * pParamOut->width <  pParamIn->channel)
         // ... !(DIL(1) && STR(1) && KER(1)) ???
         && ( pParamConv->dilationHeight | pParamConv->dilationWidth
           | pParamConv->strideHeight | pParamConv->strideWidth
           | pParamKernel->height | pParamKernel->width) != 1 )
         )
    {
      // small images may have a fast vecC
      if (KER(3) && DIL(1) && STR(1) && PAD(1)) OMPWRAP(vecC_dil1_str1_pad1_ker3) ;
      else if (KER(1) && DIL(1) && PAD(0)
          && pParamOut->height == (pParamIn->height - pParamKernel->height) / pParamConv->strideHeight + 1
          && pParamOut->width == (pParamIn->width - pParamKernel->width) / pParamConv->strideWidth + 1)
      {
        if (ICoGU(1024)) OMPWRAP(vecC_dil1_pad0_ker1_cU1024) ;
        else             OMPWRAP(vecC_dil1_pad0_ker1) ;
      }
      OMPWRAP(vecC);
#ifdef VEDNN_ALT_PARALLEL // resnext branch : AGGRESSIVE use of gemm for all stride > 1 ?
    }else if (!STR(1)) {
      // try using gemm in most cases with stride > 1
      if(OCoGU(256) && OWU(128)) ALT_RET(owU128_T);
      else ALT_RET(gemm);
#endif
    }else if (STR(1) && DIL(1)
        && pParamIn->height == pParamOut->height
        && pParamIn->width == pParamOut->width )
    { // d1s1pS ...
      if (KER(1)) {
#ifdef VEDNN_ALT_PARALLEL // new: CHECKME
        if(OWU(128)) ALT_RET(dil1_str1_pad0_ker1_T);
        //else         OMPWRAP(dil1_str1_pad0_ker1);
        else ALT_RET(gemm); // always faster?
#else
        OMPWRAP(dil1_str1_pad0_ker1);
#endif
      }else if (KER(3)){ // d1s1pSk3
        if (pParamIn->channel == pParamConv->group){ // aka inputChannelGroup==1
          if (OWU(128)) OMPWRAP(dil1_str1_padsame_ker3_c1_owU128);
          else          OMPWRAP(dil1_str1_padsame_ker3_c1);
        }else{
#ifdef VEDNN_ALT_PARALLEL
          if (pParamKernel->inChannel % 1024 == 0)
            ALT_RET(dil1_str1_padsame_ker3_c1024x_T);
          else
            ALT_RET(dil1_str1_padsame_ker3_T);
#else
          OMPWRAP(dil1_str1_padsame_ker3); // is this ever faster?
#endif
        }
      }else if (KER(5)) {  // d1s1pSk5
        if (OWU(128)) OMPWRAP(dil1_str1_padsame_ker5_owU128);
        else          OMPWRAP(dil1_str1_padsame_ker5);
      }else if (KER(2)) OMPWRAP(dil1_str1_padsame_ker2);
      OMPWRAP(dil1_str1_padsame);
      // end d1s1pS
    }else if (DIL(1) && PAD(0)
        && pParamOut->height == (pParamIn->height - pParamKernel->height) / pParamConv->strideHeight + 1
        && pParamOut->width == (pParamIn->width - pParamKernel->width) / pParamConv->strideWidth + 1 )
    { // d1p0 and oh expected value
      if (STR(1))
      { // d1s1p0
        if (KER(3) && IWU(256)
            && (pParamIn->width & 0x1) == 0  && (((uint64_t)pDataIn) & 0x7) == 0
            && (pParamOut->width & 0x1) == 0 && (((uint64_t)pDataOut) & 0x7) == 0 )
          OMPWRAP(dil1_str1_pad0_ker3_iw2XU256_ow2X_ioaligned);
        else if (KER(4) && IWU(256)) OMPWRAP(dil1_str1_pad0_ker4_iwU256);
        else if (OWU(128))           OMPWRAP(dil1_str1_pad0_owU128);
        else                         OMPWRAP(dil1_str1_pad0);
      } else if (KER(1)) { // d1s>1p0k1
        if (OWU(128)) OMPWRAP(dil1_pad0_owU128_ker1);
        else          OMPWRAP(dil1_pad0_ker1);
      }else{ // d1s>1p0k>1
        if (OWU(128)){
#ifdef VEDNN_ALT_PARALLEL
          // XXX 3 possibilities:
          //  OMPWRAP(dil1_pad0_owU128);
          //  ALT_RET(owU128_T);
          //  ALT_RET(gemm)
          if(OCoGU(256)) ALT_RET(owU128_T); // NEW
          else           ALT_RET(gemm); // NEW: is this case always faster than dil1_pad0_owU128?
#else
          OMPWRAP(dil1_pad0_owU128);
#endif
        }else{
#ifdef VEDNN_ALT_PARALLEL
          ALT_RET(gemm); // always faster than dil1_pad0 ?
#else
          OMPWRAP(dil1_pad0);
#endif
        }
      }
    }
    else if (STR(2) && KER(3) && DIL(1) && PAD(1) && OWU(128))
      OMPWRAP(dil1_str2_pad1_ker3_owU128); // N/A
    else if (STR(2) && KER(4) && DIL(1) && PAD(1) && OWU(128))
      OMPWRAP(dil1_str2_pad1_ker4_owU128);
    else if (OWU(128))
      OMPWRAP(owU128);
    OMPWRAP(default);
  }
  else{
    return VEDNN_ERROR_INVALID_PARAM ;
  }
}

#undef OCoGU
#undef ICoGU
#undef OWU
#undef IWU
#undef KER
#undef STR
#undef PAD
#undef DIL
#undef OMPWRAP

#ifdef VEDNN_ALT_PARALLEL
#undef ALT_RET
#endif

/* ----------------------------------------------------------------------- */
vednnError_t vednnConvolutionForwardAddBias(
    const vednnTensorParam_t            *pParamIn,
    const void                          *pDataIn,
    const vednnFilterParam_t            *pParamKernel,
    const void                          *pDataKernel,
    const vednnBiasParam_t              *pParamBias,
    const void                          *pDataBias,
    const vednnTensorParam_t            *pParamOut,
    void                                *pDataOut,
    const vednnConvolutionParam_t       *pParamConv,
    vednnConvolutionAlgorithm_t         algo
    )
{
  return vednnConvolutionForwardBody(pParamIn, pDataIn,
      pParamKernel, pDataKernel, pParamBias, pDataBias,
      pParamOut, pDataOut, pParamConv, algo );
}

vednnError_t vednnConvolutionForward(
    const vednnTensorParam_t            *pParamIn,
    const void                          *pDataIn,
    const vednnFilterParam_t            *pParamKernel,
    const void                          *pDataKernel,
    const vednnTensorParam_t            *pParamOut,
    void                                *pDataOut,
    const vednnConvolutionParam_t       *pParamConv,
    vednnConvolutionAlgorithm_t         algo
    )
{
  return vednnConvolutionForwardBody(pParamIn, pDataIn,
      pParamKernel, pDataKernel, NULL, NULL,
      pParamOut, pDataOut, pParamConv, algo );
}
// vim: et ts=2 sw=2 cindent cino=+4s,^=l0,\:0,N-s syntax=cpp.doxygen
