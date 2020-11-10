#ifndef SRC_VEDNNCONVOLUTIONBACKWARDDATA_H_
#define SRC_VEDNNCONVOLUTIONBACKWARDDATA_H_

#include "vednn.h"

#ifdef __cplusplus
extern "C" { //}
#endif

#define VEDNN_CONVBKD_ARGS \
  const vednnTensorParam_t * restrict        pParamGradOut, \
const void * restrict                      pDataGradOut, \
const vednnFilterParam_t * restrict        pParamKernel, \
const void * restrict                      pDataKernel, \
const vednnConvolutionParam_t * restrict   pParamConv, \
const vednnTensorParam_t * restrict        pParamGradIn, \
void * restrict                            pDataGradIn

#define VEDNN_CONVBKD_ARGS_LIST pParamGradOut, pDataGradOut, \
  pParamKernel, pDataKernel, pParamConv, pParamGradIn, pDataGradIn

typedef
vednnError_t (*vednnConvBackwardData_t) ( VEDNN_CONVBKD_ARGS );

/** this is the signature of \c pFunc arg to the wrapper, which we use
 * directly for low-level impls marked as VEDNN_WRAP_NONE in libvednnx */
typedef vednnConvBackwardData_t vednnConvBackwardData_nowrap_t;

#define VEDNN_CONVBKD_DECL(IMPL) vednnError_t \
	vednnConvolutionBackwardData_direct_##IMPL( VEDNN_CONVBKD_ARGS );

VEDNN_CONVBKD_DECL(default);
VEDNN_CONVBKD_DECL(gemm); ///< Originally plain, now can also call via wrapper threads
VEDNN_CONVBKD_DECL(vecC);
VEDNN_CONVBKD_DECL(iwU128);
VEDNN_CONVBKD_DECL(dil1_str1);
VEDNN_CONVBKD_DECL(dil1_str1_iwU128);
VEDNN_CONVBKD_DECL(dil1_str1_pad0_ker3_iwU128);
VEDNN_CONVBKD_DECL(dil1_str1_pad0_ker4_iwU128);
VEDNN_CONVBKD_DECL(dil1_str1_pad0_ker3_iw2XU256_ow2X_ioaligned);
VEDNN_CONVBKD_DECL(dil1_str1_pad0_ker3_iw2XU32_ow2X_ioaligned);
VEDNN_CONVBKD_DECL(dil1_str1_padsame);
VEDNN_CONVBKD_DECL(dil1_str1_padsame_ker1);
VEDNN_CONVBKD_DECL(dil1_pad0_ker1_owU128);
VEDNN_CONVBKD_DECL(dil1_str2_pad2_ker5_iwU128);
VEDNN_CONVBKD_DECL(dil1_str2_pad2_ker5);
VEDNN_CONVBKD_DECL(dil1_str2_pad1_ker4_iw2xU256);
VEDNN_CONVBKD_DECL(dil1_str2_pad1_ker4_iwU256);
VEDNN_CONVBKD_DECL(dil1_str1_padsame_ker2);
VEDNN_CONVBKD_DECL(dil1_str1_padsame_ker3);
VEDNN_CONVBKD_DECL(dil1_str1_padsame_ker5);
VEDNN_CONVBKD_DECL(dil1_str2_ker3_iwU256);
VEDNN_CONVBKD_DECL(dil1_str2_pad1_ker3_iwU256);
VEDNN_CONVBKD_DECL(ker3_iwU128);
VEDNN_CONVBKD_DECL(ker5_iwU128);
VEDNN_CONVBKD_DECL(iwU128);
VEDNN_CONVBKD_DECL(ker5);
// extra
//VEDNN_CONVBKD_DECL(default2);
#ifdef __cplusplus
}//extern "C"
#endif
// vim: sw=2 ts=2 et ai syntax=cpp.doxygen
#endif /* SRC_VEDNNCONVOLUTIONBACKWARDDATA_H_ */
