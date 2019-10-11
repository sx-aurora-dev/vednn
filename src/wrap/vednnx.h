#ifndef VEDNNX_H_
#define VEDNNX_H_


#if defined(__cplusplus) && !defined(restrict)
//#if !define(__ve)
#define restrict __restrict__
//#endif
#endif

/** \file
 * libvednn public and internal api have slightly different argument
 * ordering.  The iterator api of libvednnx must follow the libvednn
 * low-level convention!  So provide macros to combine parms and arguments
 * in the correct order for either call.
 *
 * ### Guiding principle
 *
 * Order of parameter and data arguments follows libvednn public call API
 * as in \ref vednn.h.  We split parameters from data. Then we give
 * macros to combine parameters and data to be able to call the libvednn
 * monolithic wrapper, OR low-level routines of libvednn exposed by the
 * libvednnx iterator interface.
 *
 * User should arrange arguments always in \ref vednn.h order, shunting
 * through re-order macros.
 *
 * - Two reordering macros are useful:
 *   - If you want to write out all parameters, use the CONV_FOO_ORDER_(lots of args) macro
 *   - If [suggested] you #define'd two parameter packs, use CONV_FOO_ORDER(PARMS,DATA) macro
 */

/** expand macro args, then reorder them for \c libvednn \b public interface.
 * These re-orderings can be written based on \ref vednn.h alone. */
#define CONV_FWD_ORDER(...) CONV_FWD_ORDER_(__VA_ARGS__)
#define CONV_FWD_ORDER_(PI,PK,PO,PC,ALGO,   DI,DK,DO) /* order: libvednn public */ \
    PI,DI, PK,DK, PO,DO, PC,ALGO

#define CONV_FWDB_ORDER(...) CONV_FWDB_ORDER_(__VA_ARGS__)
#define CONV_FWDB_ORDER_(PI,PK,PB,PO,PC,ALGO,   DI,DK,DB,DO) /* order: libvednn public */ \
    PI,DI, PK,DK, PB,DB, PO,DO, PC,ALGO

/** \ref vednn.h */
#define CONV_BKWD_ORDER(...) CONV_BKWD_ORDER_(__VA_ARGS__)
#define CONV_BKWD_ORDER_(PGI,PK,PGO,PC,ALGO,   DGI,DK,DGO) /* order: libvednn public */ \
    PGI,DGI, PK,DK, PGO,DGO, PC,ALGO

/** \ref vednn.h */
#define CONV_BKWF_ORDER(...) CONV_BKWF_ORDER_(__VA_ARGS__)
#define CONV_BKWF_ORDER_(PI,PGO,PGK,PC,ALGO,   DI,DGO,DGK) /* order: libvednn public */ \
    PI,DI, PGO,DGO, PGK,DGK, PC,ALGO
/** expand macro args, then reorder them for \c libvednn \b internal interface,
 * or \c libvednnx \b iter->impl calls.
 *
 * While the libvednnx _Begin().._End() api accepts \ref vednn.h ordering
 * of the \em parameter arguments, when you \b call (*Iter_impl) you need an
 * arrangement of args that does not quite match \ref vednn.h (\em annoying!)
 *
 * While CONV_* orderings were from \ref vednn.h, CONVX_* macros mimic
 * low-level conventions like ine \ref src/C/vednnConvolutionForward.h
 *
 * (Order differs, and the vednnConvolutionAlgorithm_t \c ALGO arg is absent)
 */
#define CONVX_FWD_ORDER(...) CONVX_FWD_ORDER_(__VA_ARGS__)
#define CONVX_FWD_ORDER_(PI,PK,PO,PC,ALGO,   DI,DK,DO) /* order: libvednn low-level */ \
    PI,DI, PK,DK, PC, PO,DO
// JIT mirror of low-level function type in src/C/vednnConvolutionForward.h
#define CONVX_FWD_DECL(FUNCNAME) \
"\nvednnError_t " FUNCNAME "(" \
"\n    const vednnTensorParam_t     * restrict  pParamIn," \
"\n    const void                   * restrict   pDataIn," \
"\n    const vednnFilterParam_t     * restrict  pParamKernel," \
"\n    const void                   * restrict   pDataKernel," \
"\n    const vednnConvolutionParam_t* restrict pParamConv," \
"\n    const vednnTensorParam_t     * restrict  pParamOut," \
"\n          void                   * restrict   pDataOut   )"

#define CONVX_FWDB_ORDER(...) CONVX_FWDB_ORDER_(__VA_ARGS__)
#define CONVX_FWDB_ORDER_(PI,PK,PB,PO,PC,ALGO,   DI,DK,DB,DO) /* order: libvednn low-level */ \
    PI,DI, PK,DK, PB,DB, PC, PO,DO
#define CONVX_FWDB_DECL(FUNCNAME) \
"\nvednnError_t " FUNCNAME "(" \
"\n    const vednnTensorParam_t     * restrict  pParamIn," \
"\n    const void                   * restrict   pDataIn," \
"\n    const vednnFilterParam_t     * restrict  pParamKernel," \
"\n    const void                   * restrict   pDataKernel," \
"\n    const vednnBiasParam_t       * restrict  pParamBias," \
"\n    const void                   * restrict   pDataBias," \
"\n    const vednnConvolutionParam_t* restrict pParamConv," \
"\n    const vednnTensorParam_t     * restrict  pParamOut," \
"\n          void                   * restrict   pDataOut   )"

/** \ref vednnConvolutionBackwardData.h */
#define CONVX_BKWD_ORDER(...) CONVX_BKWD_ORDER_(__VA_ARGS__)
#define CONVX_BKWD_ORDER_(PGI,PK,PGO,PC,ALGO,   DGI,DK,DGO) /* order: libvednn low-level */ \
    PGO,DGO, PK,DK, PC, PGI,DGI
#define CONVX_BKWD_DECL(FUNCNAME) \
"\nvednnError_t " FUNCNAME "(" \
"\n    const vednnTensorParam_t     * restrict  pParamGradOut," \
"\n    const void                   * restrict   pDataGradOut," \
"\n    const vednnFilterParam_t     * restrict  pParamKernel," \
"\n    const void                   * restrict   pDataKernel," \
"\n    const vednnConvolutionParam_t* restrict pParamConv," \
"\n    const vednnTensorParam_t     * restrict  pParamGradIn," \
"\n          void                   * restrict   pDataGradIn )"

/** \ref vednnConvolutionBackwardFilter.h (we don't deal with optional openmp args here) */
#define CONVX_BKWF_ORDER(...) CONVX_BKWF_ORDER_(__VA_ARGS__)
#define CONVX_BKWF_ORDER_(PI,PGO,PGK,PC,ALGO,   DI,DGO,DGK) /* order: libvednn low-level */ \
    PI,DI, PGO,DGO, PC, PGK,DGK
#define CONVX_BKWF_NOWRAP_DECL(FUNCNAME) \
"\nvednnError_t " FUNCNAME "(" \
"\n    const vednnTensorParam_t     * restrict  pParamIn," \
"\n    const void                   * restrict   pDataIn," \
"\n    const vednnTensorParam_t     * restrict  pParamGradOut," \
"\n    const void                   * restrict   pDataGradOut," \
"\n    const vednnConvolutionParam_t* restrict pParamConv," \
"\n    const vednnFilterParam_t     * restrict  pParamGradKernel," \
"\n          void                   * restrict   pDataGradKernel)"
#ifdef VEDNN_USE_OPENMP
#define CONVX_BKWF_DECL(FUNCNAME) CONVX_BKWF_NOWRAP_DECL(FUNCNAME)
#else
#define CONVX_BKWF_DECL(FUNCNAME) \
"\nvednnError_t " FUNCNAME "(" \
"\n    const vednnTensorParam_t     * restrict  pParamIn," \
"\n    const void                   * restrict   pDataIn," \
"\n    const vednnTensorParam_t     * restrict  pParamGradOut," \
"\n    const void                   * restrict   pDataGradOut," \
"\n    const vednnConvolutionParam_t* restrict pParamConv," \
"\n    const vednnFilterParam_t     * restrict  pParamGradKernel," \
"\n          void                   * restrict   pDataGradKernel," \
"\n    const int64_t                beginOChannel," \
"\n    const int64_t                nOChannel )"
#endif

#include "wrap/vednnConvolutionLists.h" // well parts should be public, like the Begin,Next api

// vim: et ts=4 sw=4 cindent cino=^=l0,\:0,N-s
#endif // VEDNNX_H_
