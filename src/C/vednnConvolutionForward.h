#ifndef SRC_VEDNNCONVOLUTIONFORWARD_H_
#define SRC_VEDNNCONVOLUTIONFORWARD_H_

#include "vednn.h"

#ifdef __cplusplus
extern "C" { //}
#endif

/** low-level impl std args signature, \e always with optional bias parameters.
 * Use NULL for \c pDataBias [and \p pParamBias] if layer does not need bias. */
#define VEDNN_CONVFWD_ARGS \
    const vednnTensorParam_t *restrict      pParamIn, \
    const void *restrict                    pDataIn, \
    const vednnFilterParam_t *restrict      pParamKernel, \
    const void *restrict                    pDataKernel, \
    const vednnBiasParam_t * restrict       pParamBias, \
    const void * restrict                   pDataBias, \
    const vednnConvolutionParam_t *restrict pParamConv, \
    const vednnTensorParam_t *restrict      pParamOut, \
    void *restrict                          pDataOut
// low-level impl std args list
#define VEDNN_CONVFWD_ARGS_LIST pParamIn, pDataIn, pParamKernel, pDataKernel, \
    pParamBias, pDataBias, pParamConv, pParamOut, pDataOut

typedef
vednnError_t (*vednnConvForward_t)( VEDNN_CONVFWD_ARGS );

/** this is the signature of \c pFunc arg to the wrapper, which we use
 * directly for low-level impls marked as VEDNN_WRAP_NONE in libvednnx */
typedef vednnConvForward_t vednnConvForward_nowrap_t;

#define VEDNN_FUNC_CONVFWD( SUFFIX ) vednnConvolutionForward_direct_##SUFFIX
/** low-level implementations. */
#define VEDNN_DECL_CONVFWD( SUFFIX ) vednnError_t \
    vednnConvolutionForward_direct_##SUFFIX ( VEDNN_CONVFWD_ARGS );

VEDNN_DECL_CONVFWD(default);
VEDNN_DECL_CONVFWD(gemm); ///< same parms, but do \b not call via omp wrapper
VEDNN_DECL_CONVFWD(vecC);
VEDNN_DECL_CONVFWD(vecC_dil1_str1_pad1_ker3);
VEDNN_DECL_CONVFWD(vecC_dil1_pad0_ker1);
VEDNN_DECL_CONVFWD(owU128);
VEDNN_DECL_CONVFWD(dil1_pad0);
VEDNN_DECL_CONVFWD(dil1_pad0_ker1);
VEDNN_DECL_CONVFWD(dil1_pad0_owU128);
VEDNN_DECL_CONVFWD(dil1_pad0_owU128_ker1);
VEDNN_DECL_CONVFWD(dil1_str1_pad0);
VEDNN_DECL_CONVFWD(dil1_str1_pad0_ker1);
VEDNN_DECL_CONVFWD(dil1_str1_pad0_ker3_iw2XU256_ow2X_ioaligned);
VEDNN_DECL_CONVFWD(dil1_str1_pad0_ker4_iwU256);
VEDNN_DECL_CONVFWD(dil1_str1_pad0_owU128);
VEDNN_DECL_CONVFWD(dil1_str1_padsame);
VEDNN_DECL_CONVFWD(dil1_str1_padsame_ker2);
VEDNN_DECL_CONVFWD(dil1_str1_padsame_ker3);
VEDNN_DECL_CONVFWD(dil1_str1_padsame_ker3_c1);
VEDNN_DECL_CONVFWD(dil1_str1_padsame_ker3_c1_owU128);
VEDNN_DECL_CONVFWD(dil1_str1_padsame_ker5);
VEDNN_DECL_CONVFWD(dil1_str1_padsame_ker5_owU128);
VEDNN_DECL_CONVFWD(dil1_str2_pad1_ker3_owU128);
VEDNN_DECL_CONVFWD(dil1_str2_pad1_ker4_owU128);
//VEDNN_DECL_CONVFWD(dil1_str1_padsame_ker3_c1024x);
//VEDNN_DECL_CONVFWD(gendnn);
//VEDNN_DECL_CONVFWD(alt);
//VEDNN_DECL_CONVFWD(defaultA);
//VEDNN_DECL_CONVFWD(default2);
//VEDNN_DECL_CONVFWD(default2p);
//VEDNN_DECL_CONVFWD(default3);
//VEDNN_DECL_CONVFWD(default3b);
//VEDNN_DECL_CONVFWD(owU128A);
//VEDNN_DECL_CONVFWD(dil1_pad0A);
//VEDNN_DECL_CONVFWD(dil1_pad0_owU128A);
//VEDNN_DECL_CONVFWD(dil1_pad0_ker1A);
//VEDNN_DECL_CONVFWD(dil1_pad0_owU128_ker1A);
//VEDNN_DECL_CONVFWD(dil1_str1_pad0_ker1A);
//VEDNN_DECL_CONVFWD(dil1_str1_pad0A);
//VEDNN_DECL_CONVFWD(dil1_str1_pad0_owU128A);
//VEDNN_DECL_CONVFWD(dil1_str1_padsameA); // try fastdiv
//VEDNN_DECL_CONVFWD(dil1_str1_padsameB); // try masked FMA
//VEDNN_DECL_CONVFWD(dil1_str1_padsameAB); // both above mods
//VEDNN_DECL_CONVFWD(dil1_str1_padsame_ker3A);
//VEDNN_DECL_CONVFWD(dil1_str1_padsame_ker3_c1A);
//VEDNN_DECL_CONVFWD(dil1_str1_padsame_ker3_c1_owU128A);
//VEDNN_DECL_CONVFWD(dil1_str1_padsame_ker3_c1024xA);
//VEDNN_DECL_CONVFWD(dil1_str1_padsame_ker5A);
//VEDNN_DECL_CONVFWD(dil1_str1_padsame_ker5_owU128A);
//VEDNN_DECL_CONVFWD(dil1_str1_padsame_ker2A);

#ifdef __cplusplus
}//extern "C"
#endif
// vim: et ts=4 sw=4 cindent cino=^=l0,\:0,N-s syntax=cpp.doxygen
#endif /* SRC_VEDNNCONVOLUTION_H_ */
