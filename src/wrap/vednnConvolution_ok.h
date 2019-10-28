#ifndef VEDNNCONVOLUTION_OK_H
#define VEDNNCONVOLUTION_OK_H
/** \file
 * This file declares "ok" precondition checkers on convolution descriptors
 * Checking can proceed in two phases:
 *
 * 1. Compile-time conditions based on PARAMS that describe the operation,
 *    like array sizes, padding, or other optional layer features.
 *
 * 2. Run-time conditions that check DATARG [and PARAMS] for things
 *    like alignment restrictions on pointer[s] into data tensors.
 *    Failed alignment checks then may choose an alternate impl with
 *    relaxed tenosr alignment requirements.
 *
 * - \ref vednnConvolutionLists.h or \b _Begin/_Next/_ok and \b _realNext/_rtok
 *        iteration functions.
 */
#include "vednn.h"
#include "wrap/vednnImpl_args.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif /*}*/

/** parameter order follows \ref vednn.h (just remove the pDataXXX args).
 * There are slight differences for \e restrict keyword. */
#define DECLARE_OK_FNS(Forward,FORWARD) \
typedef vednnError_t   vednnConv##Forward##_okfn_decl(  VEDNN_PARAMS_CONV_##FORWARD); \
typedef                vednnConv##Forward##_okfn_decl * vednnConv##Forward##_okfn_t; \
typedef vednnError_t   vednnConv##Forward##_rtokfn_decl(VEDNN_DATARG_CONV_##FORWARD); \
typedef vednnError_t (*vednnConv##Forward##_rtokfn_t)(  VEDNN_DATARG_CONV_##FORWARD);
DECLARE_OK_FNS(Forward,         FORWARD)
DECLARE_OK_FNS(ForwardAddBias,  FORWARDADDBIAS)
DECLARE_OK_FNS(BackwardData,    BACKWARD_DATA)
DECLARE_OK_FNS(BackwardFilter,  BACKWARD_FILTER)
#undef DECLARE_OK_FNS

// declare both the _ok and _rtok functioncs
#define FWD_FN_OK(BASENAME) FWD_FN_OK_(BASENAME)
#define FWD_RT_OK(BASENAME) FWD_RT_OK_(BASENAME)
#define FWD_FN_OK_(BASENAME) vednnConvForward_okfn_decl BASENAME##_ok;
#define FWD_RT_OK_(BASENAME) vednnConvForward_rtokfn_decl BASENAME##_rtok;
//FWD_FN_OK(vednnConvolutionForward_direct_dil1_str1_pad0_ker1_c1024x);
FWD_FN_OK(vednnConvolutionForward_direct_dil1_str1_pad0_ker1);
FWD_FN_OK(vednnConvolutionForward_direct_dil1_str1_pad0_ker1A);
FWD_FN_OK(vednnConvolutionForward_direct_dil1_str1_pad0_ker1_T);
FWD_FN_OK(vednnConvolutionForward_direct_dil1_pad0_owU128_ker1);
FWD_FN_OK(vednnConvolutionForward_direct_dil1_pad0_owU128_ker1A);
FWD_FN_OK(vednnConvolutionForward_direct_dil1_pad0_ker1);
FWD_FN_OK(vednnConvolutionForward_direct_dil1_pad0_ker1A);
FWD_FN_OK(vednnConvolutionForward_direct_dil1_str1_padsame_ker3_c1_owU128);
FWD_FN_OK(vednnConvolutionForward_direct_dil1_str1_padsame_ker3_c1_owU128A);
FWD_FN_OK(vednnConvolutionForward_direct_dil1_str1_padsame_ker3_c1024x);
FWD_FN_OK(vednnConvolutionForward_direct_dil1_str1_padsame_ker3_c1024xA);
FWD_FN_OK(vednnConvolutionForward_direct_dil1_str1_padsame_ker3_c1024x_T);
FWD_FN_OK(vednnConvolutionForward_direct_dil1_str1_padsame_ker3);
FWD_FN_OK(vednnConvolutionForward_direct_dil1_str1_padsame_ker3_T);
FWD_FN_OK(vednnConvolutionForward_direct_dil1_str1_padsame_ker3A);
FWD_FN_OK(vednnConvolutionForward_direct_dil1_str1_padsame);
FWD_FN_OK(vednnConvolutionForward_direct_dil1_str1_padsameA); // XXX testing
FWD_FN_OK(vednnConvolutionForward_direct_dil1_str1_padsameB); // XXX testing
FWD_FN_OK(vednnConvolutionForward_direct_dil1_str1_padsameAB); // XXX testing
FWD_FN_OK(vednnConvolutionForward_direct_dil1_str1_padsame_ker3_c1);
FWD_FN_OK(vednnConvolutionForward_direct_dil1_str1_padsame_ker3_c1A);
FWD_FN_OK(vednnConvolutionForward_direct_dil1_str1_padsame_ker5_owU128);
FWD_FN_OK(vednnConvolutionForward_direct_dil1_str1_padsame_ker5_owU128A);
FWD_FN_OK(vednnConvolutionForward_direct_dil1_str1_padsame_ker5);
FWD_FN_OK(vednnConvolutionForward_direct_dil1_str1_padsame_ker5A);
FWD_FN_OK(vednnConvolutionForward_direct_dil1_str1_padsame_ker2);
FWD_FN_OK(vednnConvolutionForward_direct_dil1_str1_padsame_ker2A);
FWD_FN_OK(vednnConvolutionForward_direct_dil1_str1_pad0_ker3_iw2XU256_ow2X_ioaligned);
// also has runtime "_rtok" check function:
FWD_RT_OK(vednnConvolutionForward_direct_dil1_str1_pad0_ker3_iw2XU256_ow2X_ioaligned);
FWD_FN_OK(vednnConvolutionForward_direct_dil1_str1_pad0_owU128);
FWD_FN_OK(vednnConvolutionForward_direct_dil1_str1_pad0_owU128A);
FWD_FN_OK(vednnConvolutionForward_direct_dil1_str1_pad0);
FWD_FN_OK(vednnConvolutionForward_direct_dil1_str1_pad0A);
FWD_FN_OK(vednnConvolutionForward_direct_dil1_pad0_owU128);
FWD_FN_OK(vednnConvolutionForward_direct_dil1_pad0_owU128A);
FWD_FN_OK(vednnConvolutionForward_direct_dil1_pad0);
FWD_FN_OK(vednnConvolutionForward_direct_dil1_pad0A);
FWD_FN_OK(vednnConvolutionForward_direct_dil1_ker1);
FWD_FN_OK(vednnConvolutionForward_direct_dil1_ker1A);
FWD_FN_OK(vednnConvolutionForward_direct_owU128);
FWD_FN_OK(vednnConvolutionForward_direct_owU128A);
FWD_FN_OK(vednnConvolutionForward_direct_owU128_T);
FWD_FN_OK(vednnConvolutionForward_direct_vecC);
FWD_FN_OK(vednnConvolutionForward_direct_default2);
FWD_FN_OK(vednnConvolutionForward_direct_default2p);
FWD_FN_OK(vednnConvolutionForward_direct_default3);
FWD_FN_OK(vednnConvolutionForward_direct_default3b);
FWD_FN_OK(vednnConvolutionForward_direct_default);
FWD_FN_OK(vednnConvolutionForward_direct_defaultA);
FWD_FN_OK(vednnConvolutionForward_direct_alt);
FWD_FN_OK(vednnConvolutionForward_direct_gemm);
FWD_FN_OK(vednnConvolutionForward_direct_gemmA);
#undef FWD_FN_OK
#undef FWD_RT_OK
#undef FWD_FN_OK_
#undef FWD_RT_OK_
#define FWDBIAS_FN_OK(BASENAME) FWDBIAS_FN_OK_(BASENAME)
#define FWDBIAS_RT_OK(BASENAME) FWDBIAS_RT_OK_(BASENAME)
#define FWDBIAS_FN_OK_(BASENAME) vednnConvForwardAddBias_okfn_decl BASENAME##_ok;
#define FWDBIAS_RT_OK_(BASENAME) vednnConvForwardAddBias_rtokfn_decl BASENAME##_rtok;
FWDBIAS_FN_OK(vednnConvolutionForwardAddBias_direct_dil1_str1_pad0_ker1_c1024x);
FWDBIAS_FN_OK(vednnConvolutionForwardAddBias_direct_dil1_str1_pad0_ker1);
FWDBIAS_FN_OK(vednnConvolutionForwardAddBias_direct_dil1_str1_padsame_ker3_c1_owU128);
FWDBIAS_FN_OK(vednnConvolutionForwardAddBias_direct_dil1_str1_padsame_ker3_c1024x);
FWDBIAS_FN_OK(vednnConvolutionForwardAddBias_direct_dil1_str1_padsame_ker3_c1);
FWDBIAS_FN_OK(vednnConvolutionForwardAddBias_direct_dil1_str1_padsame_ker3);
FWDBIAS_FN_OK(vednnConvolutionForwardAddBias_direct_dil1_str1_padsame);
FWDBIAS_FN_OK(vednnConvolutionForwardAddBias_direct_default);
FWDBIAS_FN_OK(vednnConvolutionForwardAddBias_direct_default2);
#undef FWDBIAS_RT_OK
#undef FWDBIAS_FN_OK
#undef FWDBIAS_RT_OK_
#undef FWDBIAS_FN_OK_
#define BKWD_FN_OK(BASENAME) BKWD_FN_OK_(BASENAME)
#define BKWD_RT_OK(BASENAME) BKWD_RT_OK_(BASENAME)
#define BKWD_FN_OK_(BASENAME) vednnConvBackwardData_okfn_decl BASENAME##_ok;
#define BKWD_RT_OK_(BASENAME) vednnConvBackwardData_rtokfn_decl BASENAME##_rtok;
BKWD_FN_OK(vednnConvolutionBackwardData_direct_dil1_str1_padsame_ker5); // new
BKWD_FN_OK(vednnConvolutionBackwardData_direct_dil1_str1_padsame_ker3); // new
BKWD_FN_OK(vednnConvolutionBackwardData_direct_dil1_str1_padsame_ker2); // new
BKWD_FN_OK(vednnConvolutionBackwardData_direct_dil1_str1_padsame_ker1); // new (XXX remove 'dil1_'
BKWD_FN_OK(vednnConvolutionBackwardData_direct_dil1_str1_padsame); // really up top?
BKWD_FN_OK(vednnConvolutionBackwardData_direct_dil1_str1_pad0_ker3_iw2XU32_ow2X_ioaligned); // new
BKWD_RT_OK(vednnConvolutionBackwardData_direct_dil1_str1_pad0_ker3_iw2XU32_ow2X_ioaligned); // rt align check
BKWD_FN_OK(vednnConvolutionBackwardData_direct_dil1_str1_pad0_ker3_iw2XU256_ow2X_ioaligned);
BKWD_RT_OK(vednnConvolutionBackwardData_direct_dil1_str1_pad0_ker3_iw2XU256_ow2X_ioaligned);
BKWD_FN_OK(vednnConvolutionBackwardData_direct_dil1_str1_pad0_ker3_iwU128);
BKWD_FN_OK(vednnConvolutionBackwardData_direct_dil1_str1_iwU128);
BKWD_FN_OK(vednnConvolutionBackwardData_direct_dil1_str1);
BKWD_FN_OK(vednnConvolutionBackwardData_direct_iwU128);
BKWD_FN_OK(vednnConvolutionBackwardData_direct_default);
BKWD_FN_OK(vednnConvolutionBackwardData_direct_default2);
BKWD_FN_OK(vednnConvolutionBackwardData_direct_gemm);
BKWD_FN_OK(vednnConvolutionBackwardData_direct_gemmA);
#undef BKWD_RT_OK
#undef BKWD_FN_OK
#undef BKWD_RT_OK_
#undef BKWD_FN_OK_
#define BKWF_FN_OK(BASENAME) BKWF_FN_OK_(BASENAME)
#define BKWF_RT_OK(BASENAME) BKWF_RT_OK_(BASENAME)
#define BKWF_FN_OK_(BASENAME) vednnConvBackwardFilter_okfn_decl BASENAME##_ok;
#define BKWF_RT_OK_(BASENAME) vednnConvBackwardFilter_rtokfn_decl BASENAME##_rtok;
BKWF_FN_OK(vednnConvolutionBackwardFilter_direct_dil1_str1_padsame_ker1);
BKWF_FN_OK(vednnConvolutionBackwardFilter_direct_dil1_str1_padsame_ker2_owU128);
BKWF_FN_OK(vednnConvolutionBackwardFilter_direct_dil1_str1_padsame_ker2);
BKWF_FN_OK(vednnConvolutionBackwardFilter_direct_dil1_str1_padsame_ker3_ohwU256);
BKWF_FN_OK(vednnConvolutionBackwardFilter_direct_dil1_str1_padsame_ker3_owU128);
BKWF_FN_OK(vednnConvolutionBackwardFilter_direct_dil1_str1_padsame_ker3);
BKWF_FN_OK(vednnConvolutionBackwardFilter_direct_dil1_str1_padsame_ker5_owU128);
BKWF_FN_OK(vednnConvolutionBackwardFilter_direct_dil1_str1_padsame_ker5);
BKWF_FN_OK(vednnConvolutionBackwardFilter_direct_dil1_str1_padsame);
BKWF_FN_OK(vednnConvolutionBackwardFilter_direct_dil1_str1_pad0_ker3_ow2X_iw2XU256_igoaligned);
BKWF_RT_OK(vednnConvolutionBackwardFilter_direct_dil1_str1_pad0_ker3_ow2X_iw2XU256_igoaligned);
BKWF_FN_OK(vednnConvolutionBackwardFilter_direct_dil1_pad0_ker1_ohwU128);
BKWF_FN_OK(vednnConvolutionBackwardFilter_direct_dil1_pad0_ker1_ohwU64);
BKWF_FN_OK(vednnConvolutionBackwardFilter_direct_dil1_pad0_ker1_owU32);
BKWF_FN_OK(vednnConvolutionBackwardFilter_direct_dil1_pad0_ker1);
BKWF_FN_OK(vednnConvolutionBackwardFilter_direct_dil1_pad0_ker3_owU128);
BKWF_FN_OK(vednnConvolutionBackwardFilter_direct_dil1_pad0_owU32);
BKWF_FN_OK(vednnConvolutionBackwardFilter_direct_dil1_pad0_owU128);
BKWF_FN_OK(vednnConvolutionBackwardFilter_direct_dil1_pad0);
BKWF_FN_OK(vednnConvolutionBackwardFilter_direct_owU128);
BKWF_FN_OK(vednnConvolutionBackwardFilter_direct_default);
BKWF_FN_OK(vednnConvolutionBackwardFilter_direct_default2);
BKWF_FN_OK(vednnConvolutionBackwardFilter_direct_gemm);
BKWF_FN_OK(vednnConvolutionBackwardFilter_direct_gemmA);
#undef BKWF_RT_OK
#undef BKWF_FN_OK
#undef BKWF_RT_OK_
#undef BKWF_FN_OK_
#ifdef __cplusplus /*{*/
}
#endif
// vim: et ts=4 sw=4 cindent cino=^=l0,\:0,N-s
#endif // VEDNNCONVOLUTION_OK_H
