#include "vednnConvolutionForward.h"
#include "vednn-def.h"
#include <stdint.h>
#include <assert.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" { //}
#endif

    inline vednnError_t
vednnConvolutionForward_mb_threads( vednnConvForward_t pFunc,
        VEDNN_CONVFWD_ARGS )
{
#ifndef VEDNN_USE_OPENMP
  return pFunc(VEDNN_CONVFWD_ARGS_LIST);
#else
  int64_t allBatch = pParamIn->batch; // check as in vednnx
  if (allBatch == 1 || __vednn_omp_num_threads == 1) {
    return pFunc(VEDNN_CONVFWD_ARGS_LIST);
  }else{
    //vednnError_t rc = VEDNN_SUCCESS ;
    int rc = VEDNN_SUCCESS ; // for C++... now explicitly return valid enum
//#pragma omp parallel reduction(|:rc)
// Above is permitted if pFunc has no omp barriers,
// but DEADLY (hang) if pFunc has an omp barrier!
// Why? myBatch==0 test would NOT allow all threads to reach synchronization
//      points in pFunc
// Ex. pFunc allocates a shared read-only scratchpad that expects
//     **all** threads to synchronize on the size needed
// So instead set num_threads so myBatch>0 always.
    int par = omp_get_max_threads();
    if (allBatch < par) par = allBatch; // avoiding myBatch==0 is important.
#pragma omp parallel num_threads(par)
    {
      int64_t nthreads = omp_get_num_threads() ;
      int64_t threadid = omp_get_thread_num() ;

      int64_t nBatch = allBatch / nthreads ;
      int64_t remain = allBatch % nthreads ;

      int64_t batchBegin = nBatch * threadid + ( threadid < remain ? threadid : remain ) ;
      int64_t myBatch = nBatch + ( threadid < remain ? 1 : 0 ) ;

      assert(myBatch > 0); // NEW (bugfix)
      //if( myBatch == 0 ) {
      //  rc |= VEDNN_SUCCESS ;
      //}else{
        vednnTensorParam_t _pParamIn  = *pParamIn  ; _pParamIn.batch = myBatch ;
        vednnTensorParam_t _pParamOut = *pParamOut ; _pParamOut.batch = myBatch ;
        float* _pDataIn  = ((float *)pDataIn) + batchBegin * pParamIn->channel * pParamIn->height * pParamIn->width ;
        float* _pDataOut = ((float *)pDataOut) + batchBegin * pParamOut->channel * pParamOut->height * pParamOut->width ;

        rc |= pFunc(&_pParamIn, (void*)_pDataIn, pParamKernel, pDataKernel,
            pParamBias, pDataBias, pParamConv, &_pParamOut, (void*) _pDataOut) ;
      //}
    }
    return (vednnError_t)(rc<3? rc: 1); // 3 is not an allowed value
  }
#endif
}

/* ----------------------------------------------------------------------- */

/** Weak Library symbol: override to test improved strategies.
 *
 * \return rc==VEDNN_SUCCESS and pFunc non-null,
 *      or rc==VEDNN_ERROR_INVALID_PARM
 */
vednnCnvFwdChoice_t
vednnConvolutionForwardChoice( VEDNN_CONVFWD_API_ARGS )
{
  // decision tree will set rc, pFunc, impl and wrapper type
  vednnError_t rc = VEDNN_SUCCESS;
  vednnConvForward_t pFunc = NULL; // internal function pointer
  char const* impl = "unset"; // internal impl name (for messages or ftrace)
  int mb_threads = 1; // threads-wrapper type
  // TODO: harmonize impl name with libvednnx (maybe via vednn.h API mods)

  // A quick initial INVALID_PARM check...
  switch( pParamKernel->layout ) {
    case VEDNN_FILTER_LAYOUT_NCHW :
      break ;
    case VEDNN_FILTER_LAYOUT_HWCN :
      if( pParamConv->group > 1 ) {
        fprintf(stderr, "[VEDNN ERROR] VEDNN does not support grouped convolution with filter_hwcn\n") ;
        rc = VEDNN_ERROR_INVALID_PARAM ;
      }
      break ;
    default :
      fprintf(stderr, "[VEDNN ERROR] Unknown Filter Layout %d\n", pParamKernel->layout) ;
      rc = VEDNN_ERROR_INVALID_PARAM ;
  }

  // NOTE: OMPWRAP and NOWRAP are CODE-BLOCK macros, not statements
  // Set normal ||ism over minibatch, and exit decision tree
#define OMPWRAP( IMPL ) \
  { \
    impl = "mb-" #IMPL; \
    pFunc = vednnConvolutionForward_direct_##IMPL; \
    /*mb_threads = 1; default*/ \
    break; \
  }

  // pFunc handles ||ism internally, and exit decision tree
#define NOWRAP( IMPL ) \
  { \
    impl = #IMPL; \
    pFunc = vednnConvolutionForward_direct_##IMPL; \
    mb_threads = 0; \
    break; \
  }

  if (rc == VEDNN_SUCCESS) do { // allow 'break' to easily exit after any pFunc or rc is set.

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
        if (KER(3) && DIL(1) && STR(1) && PAD(1))
          OMPWRAP(vecC_dil1_str1_pad1_ker3)//;
        else if (KER(1) && DIL(1) && PAD(0)
            && pParamOut->height == (pParamIn->height - pParamKernel->height) / pParamConv->strideHeight + 1
            && pParamOut->width == (pParamIn->width - pParamKernel->width) / pParamConv->strideWidth + 1)
        {
          if (ICoGU(1024)) OMPWRAP(vecC_dil1_pad0_ker1_cU1024)//;
          else             OMPWRAP(vecC_dil1_pad0_ker1)//;
        }
        OMPWRAP(vecC)//;
      }
#ifdef VEDNN_ALT_PARALLEL // resnext branch : AGGRESSIVE use of gemm for all stride > 1 ?
      if (!STR(1)) {
        //  if (STR(2) && DIL(1) && PAD(1) && OWU(128)) {
        //      if (KER(3)) OMPWRAP(dil1_str2_pad1_ker3_owU128)//;
        //      if (KER(4)) OMPWRAP(dil1_str2_pad1_ker4_owU128)//;
        //  }
        // try using gemm in most cases with stride > 1
        if(OCoGU(256) && OWU(128)) NOWRAP(owU128_T)//;
        else NOWRAP(gemm)//;
      }
#endif
      if (STR(1) && DIL(1)
          && pParamIn->height == pParamOut->height
          && pParamIn->width == pParamOut->width )
      { // d1s1pS ...
        if (KER(1)) {
#ifdef VEDNN_ALT_PARALLEL // new: CHECKME
          if(OWU(128)) NOWRAP(dil1_str1_pad0_ker1_T)//;
          //else OMPWRAP(dil1_str1_pad0_ker1)//;
          NOWRAP(gemm) // always faster?;
#else
          OMPWRAP(dil1_str1_pad0_ker1)//;
#endif
        }
        if (KER(3)){ // d1s1pSk3
          if (pParamIn->channel == pParamConv->group){ // aka inputChannelGroup==1
            if (OWU(128)) OMPWRAP(dil1_str1_padsame_ker3_c1_owU128)//;
            OMPWRAP(dil1_str1_padsame_ker3_c1)//;
          }
#ifdef VEDNN_ALT_PARALLEL
          if (pParamIn->batch < 8) { // checkme!
            if (pParamKernel->inChannel % 1024 == 0) // really!?
              NOWRAP(dil1_str1_padsame_ker3_c1024x_T)//;
            NOWRAP(dil1_str1_padsame_ker3_T)//;
          }
#else
          OMPWRAP(dil1_str1_padsame_ker3) // is this ever faster?//;
#endif
        }
        if (KER(5)) {  // d1s1pSk5
          if (OWU(128)) OMPWRAP(dil1_str1_padsame_ker5_owU128)//;
          //
          // XXX the following change 01-29-2021 "mem error fix"
          //     produces wrong output and even sometimes memory corruption.
          //     Removed (perhaps revert to the memory error version?
          //
          //else if(pParamIn->height >= 5) OMPWRAP(dil1_str1_padsame_ker5)//;
          //
          //  The following is a much slower substitute, gemm seems faster.
          //OMPWRAP(dil1_str1_padsame)//;
          //
          // uninvestigated (sometimes slightly faster): if (pParamIn->batch >= 4) OMPWRAP(gemm);
          NOWRAP(gemm); // this seems to do very well (often 25% faster)
          //
        }
        if (KER(2)) OMPWRAP(dil1_str1_padsame_ker2)//;
        OMPWRAP(dil1_str1_padsame)//;
      } // end d1s1pS
      if (DIL(1) && PAD(0)
          && pParamOut->height == (pParamIn->height - pParamKernel->height) / pParamConv->strideHeight + 1
          && pParamOut->width == (pParamIn->width - pParamKernel->width) / pParamConv->strideWidth + 1 )
      { // d1p0 and oh expected value
        if (STR(1))
        { // d1s1p0
          if (KER(3) // && IWU(256) // XXX original concords with impl name
              // XXX but actually it seems the "correctly able to run" condition is
              //     (though often the ioaligned may not be fastest, even though code
              //      looks good.  If many channels, often 2x slower).
              && OWU(256)
              && (pParamIn->width & 0x1) == 0  && (((uint64_t)pDataIn) & 0x7) == 0
              && (pParamOut->width & 0x1) == 0 && (((uint64_t)pDataOut) & 0x7) == 0 )
            OMPWRAP(dil1_str1_pad0_ker3_iw2XU256_ow2X_ioaligned)//;
          if (KER(4) && IWU(256)) OMPWRAP(dil1_str1_pad0_ker4_iwU256)//;
          if (OWU(128))           OMPWRAP(dil1_str1_pad0_owU128)//;
          OMPWRAP(dil1_str1_pad0)//;
        } else if (KER(1)) { // d1s>1p0k1
          if (OWU(128)) OMPWRAP(dil1_pad0_owU128_ker1)//;
          OMPWRAP(dil1_pad0_ker1)//;
        }
        { // d1s>1p0k>1
          // todo: this part of tree seems to target d1p0owU128, mostly
          if (OWU(128)){
#ifdef VEDNN_ALT_PARALLEL
            // XXX 3 possibilities:
            //  OMPWRAP(dil1_pad0_owU128)//;
            //  NOWRAP(owU128_T)//;
            //  NOWRAP(gemm)//;
            if(OCoGU(256)) NOWRAP(owU128_T) // NEW mb+g threading//;
            NOWRAP(gemm) // NEW: is this case always faster than dil1_pad0_owU128?//;
#else
            OMPWRAP(dil1_pad0_owU128)//;
#endif
          }else{
#ifdef VEDNN_ALT_PARALLEL
            NOWRAP(gemm) // always faster than dil1_pad0 ?//;
#else
            OMPWRAP(dil1_pad0)//;
#endif
          }
        }
      } // end d1p0 and oh expected value
      if (STR(2) && DIL(1) && PAD(1) && OWU(128)) {
        if (KER(3)) OMPWRAP(dil1_str2_pad1_ker3_owU128)//;
        if (KER(4)) OMPWRAP(dil1_str2_pad1_ker4_owU128)//;
      }
      if (OWU(128)) OMPWRAP(owU128)//;
      OMPWRAP(default)//;
    }

  }while(0);
  // Decision tree has set impl, pFunc and mb_threads [hopefully]

  if (pFunc == NULL) rc = VEDNN_ERROR_INVALID_PARAM;

  vednnCnvFwdChoice_t ret = { rc, impl, pFunc, mb_threads };
  return ret;
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
//#undef ALT_RET
#undef NOWRAP
#endif
/* ----------------------------------------------------------------------- */

/** \b with the bias arguments.
 *
 * This implementation is non-gemm, no-intrinsics ncc code.
 * Surprisingly, it is the fastest impl in some cases!
 */
vednnError_t vednnConvolutionForwardAddBias( VEDNN_CONVFWD_API_ARGS )
{
  // run the decision tree
  vednnCnvFwdChoice_t const c = vednnConvolutionForwardChoice(VEDNN_CONVFWD_API_ARGS_LIST);

  vednnError_t rc = c.rc; // initial value only
  if (rc == VEDNN_SUCCESS) { // 
      // debug...
      fprintf(stderr, " cnvFwd-def=%s\n", c.impl); fflush(stderr);
      assert( c.pFunc != NULL );

      // ftrace according to compile flags
      // Consider changing impl to reflect bias XXX
      FTRACE_BEGIN(c.impl); // impl likely differs from vednnConvolutionLists.c name
      // run with or without threading over minibatch
      if (c.mb_threads) { // call with default conv fwd ||ism wrapper
          rc = vednnConvolutionForward_mb_threads(c.pFunc, VEDNN_CONVFWD_ARGS_LIST);
      }else{            // call without any threading wrapper
          rc = c.pFunc(VEDNN_CONVFWD_ARGS_LIST);
      }
      FTRACE_END(c.impl); // note different from src/wrap vednnx extensions :(
  }
  return rc;
}

/* ----------------------------------------------------------------------- */
/** \b without the bias arguments (auto-supplying NULL for bias args). */
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
  return vednnConvolutionForwardAddBias(pParamIn, pDataIn,
      pParamKernel, pDataKernel, NULL, NULL,
      pParamOut, pDataOut, pParamConv, algo );
}
#ifdef __cplusplus
}//extern "C"
#endif
// vim: et ts=2 sw=2 cindent cino=+4s,^l0,\:s syntax=cpp.doxygen
