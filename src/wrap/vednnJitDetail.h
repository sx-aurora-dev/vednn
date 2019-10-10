#ifndef VEDNNJITDETAIL_H
#define VEDNNJITDETAIL_H
/** \file
 * preliminary api, unimplemented.
 */
#include "vednnx.h"

#ifdef __cplusplus
extern "C" { //}
#endif

/** opaque \c vednnSym_t details.
 *
 * - Main functionality:
 *   - copy of Params [used for JIT, but not for default impl]
 *   - \b _rtok consistency check function [optional]
 *   - forward pointer to "backup" impls [optional, lazy]
 *
 * - Most pointers need to typecast to correct function signatures,
 *   which is hidden by invoking via \c vednnSymLAYERJit [preferred]
 *   or \c vecnnSymLayerJitCheck [helper] functions
 *
 * Other Details:
 *
 * The symbol \e SYMBOL actually represents a set of functions, 2 of which have
 * suffixes \c _ok and \c _rtok. The \b _ok function checks that parameters are
 * consistent with a jit function, while the \b _rtok does additional check
 * given actual pointer values (ex. pointer alignment validity).
 *
 * - \e SYMBOL      : jit name of a vednn low-level function
 * - \e SYMBOL_ok   : jit name "compile-time" applicability check
 *   - This receives the 'Param' arguments of the low-level call, in vednn order.
 * - \e SYMBOL_rtok : jit name of "run-time" applicability check
 *   - This receives all arguments just like the low-level vednn call.
 *
 * \b _ok and \b _rtok symbols are \e optional.  If absent we assume it
 * is OK to call \e SYMBOL, with only usual consistency checks (in, out
 * dimensions make sense, etc.)
 *
 * The \b _ok functions are called during \c vednnSym_t construction, to
 * choose a good prospective implementation.  If available, \b _rtok is called
 * to check data-pointer validity, at call time.
 *
 * Item one in the \c jit_ctx name resolution context is a process library
 * that may get rereshed.  \c lib_seq is used to detect that \c addr is
 * stale and needs to be rereshed via \c dlsym called again.  This only
 * happens for symbols in entry one of the \c jit_ctx (the modifiable
 * process library).
 */
typedef struct vednnSymDetail_s {
    // The [optional] ok function will be called when we are constructed, and is not
    // needed afterward. Typically this function is some libvednn/jit
    // name with "_ok" appended.
    //void *ok;

    /** this is a private copy of the address, always available. */
    void *addr;

    /** a \c vednnJitCtx*, \ref vednnJitDetail.hpp. */
    void * const jit_ctx;

    /** sequence number to detect stale symbols.
     * This is assigned and managed by the \c VednnJitCtx.
     * It detects when jit generators signal that the process-library
     * has been grown by newly-added symbols.
     */
    uint64_t lib_seq;

    /** pointer to some Layer_rtok function [usually NULL].  Some
     * layers might need to check pointer alignment at call time. */
    void *rtok;

    /** ptr to an alternate fallback impl [usually NULL].
     * - Lazy init occurs if rtok fails.
     * - When \c haveParams, we can determine a fallback in case \c !rtok.
     * - We can even fall back to the usual libvednn default impl.
     */
    void *addrNext;

    /** zero if ParamsUnion not initialized [libvednn default low-level impl]
     * always one for JIT constructors. */
    int haveParams;

    union Params {
        struct{
            vednnTensorParam_t pParamIn;
            vednnFilterParam_t pParamKernel;
            vednnBiasParam_t   pParamBias;
            vednnTensorParam_t pParamOut;
            vednnConvolutionParam_t pParamConv;
            vednnConvolutionAlgorithm_t alg;
        } convFwdParam;
    } params;

} vednnSymDetail_t;

#ifdef __cplusplus
}//"C"
#endif
// vim: et ts=4 sw=4 cindent cino=^0,=0,l1,\:0,=s,N-s,g-2,h2 syntax=cpp.doxygen
#endif // VEDNNJITDETAIL_H
