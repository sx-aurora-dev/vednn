#ifndef VEDNNCONVOLUTIONLISTS_H
#define VEDNNCONVOLUTIONLISTS_H
#include "wrap/vednnConvolution_ok.h"

#include "C/vednnConvolutionForward.h"
//#include "C/vednnConvolutionForwardAddBias.h"
#include "C/vednnConvolutionBackwardData.h"
#include "C/vednnConvolutionBackwardFilter.h"

#include <stddef.h> // NULL

#ifdef __cplusplus
extern "C" {
#endif //}
/** \file
 * Iteration-over-implementations API.
 *
 * You can iterate over lists of layer implementations.
 *
 * - \b _ok functions from \b _Begin/_Next iterate through compatible impls
 *   according to PARAMS, much like in the mkl-dnn API
 * - \b _rtok functions during \b _realNext adjust the _Next iterator
 *   as req'd to the next impl compatible with both PARAMS and DATARGS
 * - impls within each list are ordered so that the first "ok" impl
 *   should be the fastest impl.
 * - \b _Run on an iterator does \b _realNext and invokes an openmp wrapper
 *   that calls the libvednn low-level implementation.  The wrapper does the
 *   same as the unexposed wrapper inside of libvednn itself.
 *   - run arguments follow the libvednn.h public order, and use macros
 *     to reorder to the args used by libvednn internal impls.
 *
 * Usage overview:
 *
 *     - client see XXX_Begin/Next and gets opaque ptr with impl.
 *     - Client calls 'XXX_Run' (or whatever) with the opaque ptr
 *       followed by the normal \ref vednn.h parameters.
 *     - XXX_Run just calls the appropriate low-level impl, re-ordering
 *       parameters as required (and doing openmp parallelization)
 *
 * Note: for use in external projects like mkl-dnn [gen-dnn, for Aurora],
 *       the _ok and _rtok functions and the raw impl lists must be
 *       exported from libvednnx.
 */

/** \group Impl Begin/Next API */
/// - _realNext and _Run are new, and need review:
///   - _realNext implements the runtime check (rtok)
///   - and _Run allows the impls to use openmp.
/// - \p BASE ~ vednnConvForward
/// - \p BASE_PARAMS ~ VEDNN_PARAMS_CONV_FORWARD (layer parameters)
/// - \p BASE_DATARG ~ VEDNN_DATARG_CONV_FORWARD (tensor pointer args)
//@{
#define ITERATOR_FUNC_API_( BASE, BASE_PARAMS, BASE_DATARG ) \
    \
struct BASE##Impls_s; /*fwd decl*/ \
typedef struct BASE##Impls_s BASE##Impls; \
extern BASE##Impls * BASE##List; \
\
BASE##Impls * BASE##_Begin( BASE_PARAMS ); \
\
BASE##Impls * BASE##_Next( BASE##Impls* current, BASE_PARAMS ); \
\
BASE##Impls * BASE##_realNext( BASE##Impls* current, BASE_PARAMS, BASE_DATARG ); \
\
void          BASE##_Dump( BASE##Impls const* current ); \
\
typedef struct { \
    BASE##Impls* actual; \
    vednnError_t status; \
} BASE##_out_t; \
\
BASE##_out_t BASE##_Run( BASE##Impls* current, BASE_PARAMS, BASE_DATARG );

#define ITERATOR_FUNC_API(Conv,Forward,CONV,FORWARD) ITERATOR_FUNC_API_( \
        vednn##Conv##Forward, \
        VEDNN_PARAMS_##CONV##_##FORWARD, \
        VEDNN_DATARG_##CONV##_##FORWARD)

//                 func names           macro names
ITERATOR_FUNC_API( Conv,Forward,        CONV,FORWARD )
//ITERATOR_FUNC_API( Conv,ForwardAddBias, CONV,FORWARDADDBIAS )
ITERATOR_FUNC_API( Conv,BackwardData,   CONV,BACKWARD_DATA )
ITERATOR_FUNC_API( Conv,BackwardFilter, CONV,BACKWARD_FILTER )
// following are TBD:
ITERATOR_FUNC_API( Lin,Forward,         LIN,FORWARD) // no backward (reuse fwd)
ITERATOR_FUNC_API( MaxPool,Forward,     MAXPOOL,FORWARD)
ITERATOR_FUNC_API( MaxPool,Backward,    MAXPOOL,BACKWARD)
ITERATOR_FUNC_API( Act,Forward,         ACT,FORWARD)
ITERATOR_FUNC_API( Act,Backward,        ACT,BACKWARD)

#undef ITERATOR_FUNC_API
//@}

/** \addtogroup Impl list elements
 *  These flesh out low-level vednn implementations with some handy info,
 * and some handy auxiliary functions.
 */
//@{    

#define CONVLIST_ENTRY(Forward,FORWARD) \
struct vednnConv##Forward##Impls_s { \
    vednnConv##Forward##_t          impl; /**< vednn library function */ \
    char const*                     name; /**< full name of impl */ \
    char const*                shortname; /**< shorter label for impl */ \
    vednnConv##Forward##_okfn_t     okfn; /**< check if PARAMS ok */ \
    vednnConv##Forward##_rtokfn_t rtokfn; /**< usually NULL, (else check DATA alignment ok) */ \
    void* (*getPd)(VEDNN_PARAMS_CONV_##FORWARD); /**< WIP: precalc data */ \
    vednnConv##Forward##Pd_t (*getImpl)(void* pd, VEDNN_PARAMS_CONV_##FORWARD); /**< WIP: JIT */ \
}
CONVLIST_ENTRY(Forward,        FORWARD);
//CONVLIST_ENTRY(ForwardAddBias, FORWARDADDBIAS);
CONVLIST_ENTRY(BackwardData,   BACKWARD_DATA);
CONVLIST_ENTRY(BackwardFilter, BACKWARD_FILTER);

/** \struct vednnConvForwardImpls_s
 *
 *   \param okfn, returning VEDNN_SUCCESS if parameters seem compatible
 *   \param rtokfn, returning VEDNN_SUCCESS if data ptrs seem compatible
 *   \param impl, a hardwired pointer to the libvednn function
 *   \param name, full name of function
 *   \param shortname, shorter string for function
 *   \param getPd, reserved for future use
 *   \param getImpl, reserved for future use
 *
 * - \c impl arg order does not match public API! There are arg-reordering
 *   macros to handle this nicely.
 *   - \ref C/vednnConvolutionForward.h vs public libvednn api in \ref vednn.h
 *
 * - TODO: newer style convolution functions may wish to precalculate
 *   some private data, \c getPd, which is a function of the parameters
 *   (and will typically also store a verbatim copy of all parameters)
 *   - Then we can return a new style library function that accepts as
 *     arguments the private data, and only the data inputs (no \e pParamFoo
 *     args.
 *   - This means \c impl is null, because a new function signature is used
 *   - The new <em>private data</em> implementation accepts just a \e void*
 *     in lieu of all pParamFoo args, followed by actual tensor data args
 *   - Actually, the new impl is returned via a function call, which allows
 *     the function pointer to be jit-ed at runtime, if desired.
 *
 * - Each newer [or jit] \c impl would NULL, instead using \b two helpers:
 *   - \c getPd, a function taking Params and returning
 *     a single <TT>void*</TT> private data
 *     - private data would typically maintain a copy of all pParamFoo parms
 *     - private data memory could also contain precalculated constant data,
 *       in whatever form best can speed up the implementation.
 *     - Ex. \c vednnConForwardPd_t is a modification of libvednn arguments as
 *       in \ref vednnConvolutionForward.h
 *   - \c getImpl, returning a function that accepts private data, followed by
 *     the usual convolution arguments (just the data arguments, params could be
 *     cached within private data)
 *
 * \todo make okfn, rtokfn and impl void const ptrs, to force iterator API.
 *       Rationale:
 *       User might not do full checks, or might avoid a wrapper that adds
 *       thread support, if he tries to use these ptrs directly. also some
 *       of the fn ptrs can be NULL.
 *
 * \todo working example of getPd+getImpl
 */
//@}

#ifdef __cplusplus //{
}
#endif
// vim: et ts=4 sw=4 cindent cino=^=l0,\:0,N-s syntax=cpp.doxygen
#endif // VEDNNCONVOLUTIONLISTS_H
