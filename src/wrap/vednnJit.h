#ifndef VEDNNJIT_H
#define VEDNNJIT_H
/** \file
 * preliminary api, unimplemented.
 */
#include "vednnJitDetail.h"

#ifdef __cplusplus
extern "C" { //}
#endif

// fwd decl
struct vednnSymDetail_s;
struct vednnSym_s;
typedef struct vednnSym_s vednnSym_t;

/** Initialize jit library context \c jitctx for this process.
 * \c jitctx supports symbol resolution, and is passed to
 * all \c vednnSym_t constructor functions.
 *
 * Whenever a jit generator needs to generate new symbols
 * the \c process_subdir gets [over]written with build files,
 * \e make is called and a small jit library gets created.
 *
 * The if \c process_libpath is writable, we update a growing
 * library of symbols that this process newly generated, setting
 * flags so that we can detect symbols that may have new addressess
 * as a result.
 *
 * You should  not use same subdir or libpath in different processes
 * running at the same time.
 *
 * If you never call this function, you can still use a NULL jit context
 * in symbol constructors, and we will still fast-track the calls to
 * libvednn's default impls, avoiding the JIT process entirely.
 */
vednnError_t vednnInitJit(void** jitctx,
        char const* process_subdir,
        char const* process_lib);

/** Add \c libpath dll to \c jitctx symbol resolution procedures.
 * \pre non-NULL \c jitctx set via \c vednnInitJit.
 *
 * This can be called zero times, in which case git generators will
 * still look up their symbols in process namespace (in case process
 * is pre-linked with a previously generated static JIT archive)
 *
 * Otherwise generators will first look in the \c libpath dlls,
 * in order added, and finally in global namespace for symbols.
 * If generators don't find their symbol name, they will regenerate
 * JIT code, compile to \c process_subdir/process_lib and then
 * attempt to add their functions to 
 */
vednnError_t vednnJitLib(void* jitctx, char const* libpath);

/** A jit symbol lookup result for a low-level vednn (or jit)
 * implementation.
 *
 * - There are 2 main classes of functions using \c vednnSym_t
 *   - layer-specific constructors
 *     - jit constructors (generate new symbol if unknown)
 *     - vednn default impl lookup (jit or vednn default flavors)
 *   - layer-specific invokers
 *
 * If the constructor sets \c addr non-NULL, then there is a fast-path
 * call available -- such symbols guarantee never needing to check for
 * stale dlls, and never need to check \e data pointer alignments.
 */
struct vednnSym_s {
    char const* const symbol; ///< a low-level vednn jit impl (NULL means vednn public impl)
    void * addr;              ///< for fastest calling (not always exposed)
    vednnSymDetail_t opaque;  ///< opaque details
};

/** release scratchpad memory or other privately held data.
 * At least releases \c jitsym->symbol string memory. */
void vednnSym_free( vednnSym_t * jitsym );

/** @group vednnSymCall invoke a LAYER via \c vednnSym_t.
 * The first argument is always a \c vednnSym_t*, followed by
 * all the \e data pointers in \ref vednn.h (public API) order,
 * or \e param+data pointers, again in vednn.h order.
 */
//@{
/** Recommended generic call w/ jit support (fast case inlined).
 * Resupply full \e param and \e data info (in case we are a non-JIT libvednn impl).
 * Arguments mirror <B>low-level</B> vednn call, with appended \c jitsym arg.
 *
 * The fast case, \c (*jitsym)->addr known, is inlined, otherwise we call \c vednnLAYERJitCheck
 *
 * Note that fast case does NOT include jit generators that this process has
 * already run -- so you are encouraged to manage process-specific jit libraries
 * and add them the fast way, via \c vednnJitLib instead of recompiling them
 * every time.
 *
 * Least code change to support JIT, because arguments almost exactly match libvednn PUBLIC API.
 */
inline vednnError_t vednnConvolutionForwardJit(
        // PUBLIC API args (incl the ALGO)
        CONV_FWD_ORDER( VEDNN_PARAMS_CONV_FORWARD, VEDNN_DATARG_CONV_FORWARD ),
        // shorthand for:
        //const vednnTensorParam_t      * pParamIn,
        //const void * restrict         pDataIn,
        //const vednnFilterParam_t      * pParamKernel,
        //const void * restrict         pDataKernel,
        //const vednnConvolutionParam_t * pParamConv,
        //const vednnTensorParam_t      * pParamOut,
        //void * restrict               pDataOut
        //vednnConvolutionAlgorithm_t   algo, /*VEDNN_CONV_ALGORITHM_DIRECT*/
        vednnSym_t* jitsym ///< additional arg for call via jitsym [may be NULL]
        );

/** ConvolutionForward call helper function [usable only for a JIT \c jitsym].
 * Checks data args and invokes \c jitsym->symbol or some backup low-level impl.
 * \pre jitsym has \e params and \e data pointers exactly conform.
 * \pre \e data pointers exactly correspond to params given to \c jitsym constructor,
 */
vednnError_t vednnConvolutionForwardJitCheck(
        vednnSym_t *jitsym,       ///< ptr to some low-level impl
        VEDNN_DATARG_CONV_FORWARD ///< \ref vednnImpl_args.h
        // shorthand for:
        //void const *pDataIn,
        //void const *pDataKernel,
        //void const *pDataBias, // or NULL
        //void const *pDataOut
        );

//@}

/** @group vednnSymOperation jitsym constructors.
 * \c jitsym is assigned a string function name, and possibly a fully-resolved
 * address to a vednn low-level impl function (possibly jitted).
 * The function name is typically the operation (layer) name with a suffix
 * that is some canonical string mangling of the layer parameters, or the
 * actual name of a non-jit libvednn implementation.
 */
//@{
/** \c jitsym that punts to the non-jit vednn low-level impl.
 * This runs the decision tree logic of the public libvednn function
 * and returns the low-level impl pointer (without invoking it).
 * For now, maintain a copy the params, to provide maximum flexibility of
 * ways to invoke the Convolution Forward (and error checking in debug compile).
 */
vednnError_t vednnSymConvolutionForwardDefault(
        vednnSym_t ** jitsym,
        VEDNN_PARAMS_CONV_FORWARD
        //vednnTensorParam_t      pParamIn,
        //vednnBiasParam_t        pParamBias,
        //vednnFilterParam_t      pParamFilter,
        //vednnTensorParam_t      pParamOut,
        //vednnConvolutionParam_t pParamConv
        //vednnConvolutionAlgorithm_t 	algo
        );

/** construct jit wrapper for a Convolution Forward layer. This does copy the
 * layer params, and if possible provides a public \c (*jitsym)->addr. */
vednnError_t vednnSymConvolutionForward(
        vednnSym_t** jitsym,
        VEDNN_PARAMS_CONV_FORWARD
        //vednnTensorParam_t      pParamIn,
        //vednnBiasParam_t        pParamBias,
        //vednnFilterParam_t      pParamFilter,
        //vednnTensorParam_t      pParamOut,
        //vednnConvolutionParam_t pParamConv
        );
//@}
//
// --------------- inlines -----------------
//
inline vednnError_t vednnConvolutionForwardJit(
        CONV_FWD_ORDER(VEDNN_PARAMS_CONV_FORWARD, VEDNN_DATARG_CONV_FORWARD),
        vednnSym_t* jitsym )
{
    vednnError_t ret = VEDNN_ERROR_INVALID_PARAM;
    if(jitsym && jitsym->addr){ // fast path : cast and invoke an
        // impl with no ptr alignment checks.  Impls with data ptr
        // checks needed *BEFORE* call will set jitsym->addr=NULL
        ret = ((vednnConvForward_t)(jitsym->addr))( // Cast to low-level
                               // function signature and invoke, with
              CONVX_FWD_ORDER(  // low-level (CONVX) argument ordering.
              VEDNN_PARAMS_CONV_FORWARD_LIST,
              VEDNN_DATARG_CONV_FORWARD_LIST )
            );
    }
    else if(jitsym==NULL){ // NULL jitsym --> vednn public API call.
        // not CONVX_FWD_ORDER, so public API ...
        ret = vednnConvolutionForward( CONV_FWD_ORDER(
                VEDNN_PARAMS_CONV_FORWARD_LIST,
                VEDNN_DATARG_CONV_FORWARD_LIST ));
        // equiv:   pParamIn, pDataIn, pParamKernel, pDataKernel,
        //          pParamBias, pDataBias, pParamOut, pDataOut,
        //          pParamConv, VEDNN_CONV_ALGORITHM_DIRECT );
    }else { // jitsym with private symbol addr : run alignment checks and invoke/reroute
        ret = vednnConvolutionForwardJitCheck(
                jitsym, VEDNN_DATARG_CONV_FORWARD_LIST );
        // equiv: jitsym, pDataIn, pDataKernel, pDataBias, pDataOut
    }
    return ret;
}
// is there a boilerplate macro version of the above inlines?

#ifdef __cplusplus
}//"C"
#endif
/** \struct vednnSym_s
 * \c vednnSymLAYER(ParamsOnly,..) construct these objects, which are freed
 * by \c vednnSym_free.
 *
 * \p symbol is the name of a low-level libvednn impl, jit or not-jit.
 *
 * \p addr is a fast-path low-level call address [optional]
 *
 * \p opaque details contain optional runtime checks, and provisions to call a backup
 * implementation.  Jit constructors will also maintain a full copy of all layer Params.
 *
 * Recommended invocation of the jitsym is via \c vednnLAYERJit(paramsAndDataArgs...,vednnSym_t)
 * - this will invoke libvednn built-in impl if nec.
 * - or a fast-call via \c addr
 * - check rtok if nec. and invoke, or determine a backup impl and invoke.
 *
 * - Multithreading?
 *   - No issues.
 *   - addresses from dlls are valid for all threads and processes.
 *   - statically linked libraries in executable also OK, although different
 *     processes will get different function addresses.
 */
/** \fn vednnConvolutionForwardJitCheck
 * \pre \c jitsym was obtained from vednnSymConvolutionForward
 * \pre \c pData pointers describe memory exactly match the
 *      parameters supplied to \c vednnSymConvolutionForward.
 *
 * Recommended method is to call \c vednnConvolutionForwardJit,
 * which will circumvent this call if possible.
 *
 * But if you use \c vednnSymConvolutionForwardDefault to construct, you
 * can call this directly.  Any time \c jitsym->addr is available, it
 * is safe to invoke via \c vednnConvolutionForwardJitCheck.  It is a pain
 * to typecast \c vednnSym_t::addr to the correct function signature and
 * directly call the low-level impl.
 */
/** \fn vednnSymConvolutionForwardDefault
 *
 * For speed, we do \b not copy the parameters, so you must use this jitsym
 * via \c vednnConvolutionForwardJit (with low-level arg list + jitsyms).
 *
 * \pre jitsym non-NULL
 * \post *jitsym non-NULL
 * \post (*jitsym)->addr points to default libvednn low-level impl
 * \post jitsym does \b not remember the parameters.
 *
 * \return VEDNN_SUCCESS and set *jitsym, or an error and *jitsym=NULL
 */
/** \fn vednnSymConvolutionForward
 * The suffix is a version of an \b mkl-dnn parameter string.
 * Note that mkl-dnn dilation convention begins "no dilation" at zero,
 * whereas the "no dilation" value in vednn begins at one.
 *
 * Bias and Output settings are not really needed, but are checked for full consistency
 * with input, kernel and convolution paremters.
 *
 * \return VEDNN_ERROR_INVALID_PARAM if any parameters seem inconsistent,
 * and set \c *symbol to NULL.
 */
// xim: et ts=4 sw=4 cindent cino=^=l0,\:0,N-s
// vim: et ts=4 sw=4 cindent cino=l1,)0,u0,W2,\:0,=2s,N-s,g-2,h2 syntax=cpp.doxygen
#endif //VEDNNJIT_H
