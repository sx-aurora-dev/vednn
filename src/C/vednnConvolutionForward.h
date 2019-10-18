
#ifndef SRC_VEDNNCONVOLUTION_FORWARD_H_
#define SRC_VEDNNCONVOLUTION_FORWARD_H_

#include "vednn.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef
vednnError_t (*vednnConvForward_t)(
    const vednnTensorParam_t * restrict         pParamIn,
    const void * restrict                       pDataIn,
    const vednnFilterParam_t * restrict         pParamKernel,
    const void * restrict                       pDataKernel,
    const vednnConvolutionParam_t * restrict    pParamConv,
    const vednnTensorParam_t * restrict         pParamOut,
    void * restrict                             pDataOut) ;
/** this is the signature of \c pFunc arg to the wrapper, which we use
 * directly for low-level impls marked as VEDNN_WRAP_NONE in libvednnx */
typedef vednnConvForward_t vednnConvForward_nowrap_t;

vednnError_t
vednnConvolutionForward_direct_default(
    const vednnTensorParam_t * restrict         pParamIn,
    const void * restrict                       pDataIn,
    const vednnFilterParam_t * restrict         pParamKernel,
    const void * restrict                       pDataKernel,
    const vednnConvolutionParam_t * restrict    pParamConv,
    const vednnTensorParam_t * restrict         pParamOut,
    void * restrict                             pDataOut
) ;

vednnError_t
vednnConvolutionForward_direct_gemm(
    const vednnTensorParam_t * restrict         pParamIn,
    const void * restrict                       pDataIn,
    const vednnFilterParam_t * restrict         pParamKernel,
    const void * restrict                       pDataKernel,
    const vednnConvolutionParam_t * restrict    pParamConv,
    const vednnTensorParam_t * restrict         pParamOut,
    void * restrict                             pDataOut
) ;

vednnError_t
vednnConvolutionForward_direct_vecC(
    const vednnTensorParam_t * restrict         pParamIn,
    const void * restrict                       pDataIn,
    const vednnFilterParam_t * restrict         pParamKernel,
    const void * restrict                       pDataKernel,
    const vednnConvolutionParam_t * restrict    pParamConv,
    const vednnTensorParam_t * restrict         pParamOut,
    void * restrict                             pDataOut
) ;

vednnError_t
vednnConvolutionForward_direct_owU128(
    const vednnTensorParam_t * restrict         pParamIn,
    const void * restrict                       pDataIn,
    const vednnFilterParam_t * restrict         pParamKernel,
    const void * restrict                       pDataKernel,
    const vednnConvolutionParam_t * restrict    pParamConv,
    const vednnTensorParam_t * restrict         pParamOut,
    void * restrict                             pDataOut
) ;

vednnError_t
vednnConvolutionForward_direct_owU128_T(
    const vednnTensorParam_t * restrict         pParamIn,
    const void * restrict                       pDataIn,
    const vednnFilterParam_t * restrict         pParamKernel,
    const void * restrict                       pDataKernel,
    const vednnConvolutionParam_t * restrict    pParamConv,
    const vednnTensorParam_t * restrict         pParamOut,
    void * restrict                             pDataOut
) ;

vednnError_t
vednnConvolutionForward_direct_owU128_T_subkernel(
    const vednnTensorParam_t * restrict         pParamIn,
    const void * restrict                       pDataIn,
    const vednnFilterParam_t * restrict         pParamKernel,
    const void * restrict                       pDataKernel,
    const vednnConvolutionParam_t * restrict    pParamConv,
    const vednnTensorParam_t * restrict         pParamOut,
    void * restrict                             pDataOut,
    int                                         n,
    int                                         group,
    int                                         curOutChannelGroupPrime,
    int                                         curOutYPrime
) ;

vednnError_t
vednnConvolutionForward_direct_dil1_pad0(
    const vednnTensorParam_t * restrict         pParamIn,
    const void * restrict                       pDataIn,
    const vednnFilterParam_t * restrict         pParamKernel,
    const void * restrict                       pDataKernel,
    const vednnConvolutionParam_t * restrict    pParamConv,
    const vednnTensorParam_t * restrict         pParamOut,
    void * restrict                             pDataOut
) ;

vednnError_t
vednnConvolutionForward_direct_dil1_pad0_owU128(
    const vednnTensorParam_t * restrict         pParamIn,
    const void * restrict                       pDataIn,
    const vednnFilterParam_t * restrict         pParamKernel,
    const void * restrict                       pDataKernel,
    const vednnConvolutionParam_t * restrict    pParamConv,
    const vednnTensorParam_t * restrict         pParamOut,
    void * restrict                             pDataOut
) ;

vednnError_t
vednnConvolutionForward_direct_dil1_pad0_ker1(
    const vednnTensorParam_t * restrict         pParamIn,
    const void * restrict                       pDataIn,
    const vednnFilterParam_t * restrict         pParamKernel,
    const void * restrict                       pDataKernel,
    const vednnConvolutionParam_t * restrict    pParamConv,
    const vednnTensorParam_t * restrict         pParamOut,
    void * restrict                             pDataOut
) ;

vednnError_t
vednnConvolutionForward_direct_dil1_pad0_owU128_ker1(
    const vednnTensorParam_t * restrict         pParamIn,
    const void * restrict                       pDataIn,
    const vednnFilterParam_t * restrict         pParamKernel,
    const void * restrict                       pDataKernel,
    const vednnConvolutionParam_t * restrict    pParamConv,
    const vednnTensorParam_t * restrict         pParamOut,
    void * restrict                             pDataOut
) ;

vednnError_t
vednnConvolutionForward_direct_dil1_str1_pad0(
    const vednnTensorParam_t * restrict         pParamIn,
    const void * restrict                       pDataIn,
    const vednnFilterParam_t * restrict         pParamKernel,
    const void * restrict                       pDataKernel,
    const vednnConvolutionParam_t * restrict    pParamConv,
    const vednnTensorParam_t * restrict         pParamOut,
    void * restrict                             pDataOut
) ;

vednnError_t
vednnConvolutionForward_direct_dil1_str1_pad0_owU128(
    const vednnTensorParam_t * restrict         pParamIn,
    const void * restrict                       pDataIn,
    const vednnFilterParam_t * restrict         pParamKernel,
    const void * restrict                       pDataKernel,
    const vednnConvolutionParam_t * restrict    pParamConv,
    const vednnTensorParam_t * restrict         pParamOut,
    void * restrict                             pDataOut
) ;

vednnError_t
vednnConvolutionForward_direct_dil1_str1_pad0_ker3_iw2XU256_ow2X_ioaligned(
    const vednnTensorParam_t * restrict         pParamIn,
    const void * restrict                       pDataIn,
    const vednnFilterParam_t * restrict         pParamKernel,
    const void * restrict                       pDataKernel,
    const vednnConvolutionParam_t * restrict    pParamConv,
    const vednnTensorParam_t * restrict         pParamOut,
    void * restrict                             pDataOut
) ;

vednnError_t
vednnConvolutionForward_direct_dil1_str1_pad0_ker1(
    const vednnTensorParam_t * restrict         pParamIn,
    const void * restrict                       pDataIn,
    const vednnFilterParam_t * restrict         pParamKernel,
    const void * restrict                       pDataKernel,
    const vednnConvolutionParam_t * restrict    pParamConv,
    const vednnTensorParam_t * restrict         pParamOut,
    void * restrict                             pDataOut
) ;

vednnError_t
vednnConvolutionForward_direct_dil1_str1_pad0_ker1_T(
    const vednnTensorParam_t * restrict         pParamIn,
    const void * restrict                       pDataIn,
    const vednnFilterParam_t * restrict         pParamKernel,
    const void * restrict                       pDataKernel,
    const vednnConvolutionParam_t * restrict    pParamConv,
    const vednnTensorParam_t * restrict         pParamOut,
    void * restrict                             pDataOut
) ;

vednnError_t
vednnConvolutionForward_direct_dil1_str1_pad0_ker1_T_subkernel(
    const vednnTensorParam_t * restrict         pParamIn,
    const void * restrict                       pDataIn,
    const vednnFilterParam_t * restrict         pParamKernel,
    const void * restrict                       pDataKernel,
    const vednnConvolutionParam_t * restrict    pParamConv,
    const vednnTensorParam_t * restrict         pParamOut,
    void * restrict                             pDataOut,
    int                                         n,
    int                                         group,
    int                                         curOutChannelGroupPrime,
    int                                         curOutPixelPrime
) ;

vednnError_t
vednnConvolutionForward_direct_dil1_str1_padsame(
    const vednnTensorParam_t * restrict         pParamIn,
    const void * restrict                       pDataIn,
    const vednnFilterParam_t * restrict         pParamKernel,
    const void * restrict                       pDataKernel,
    const vednnConvolutionParam_t * restrict    pParamConv,
    const vednnTensorParam_t * restrict         pParamOut,
    void * restrict                             pDataOut
) ;

vednnError_t
vednnConvolutionForward_direct_dil1_str1_padsame_ker3(
    const vednnTensorParam_t * restrict         pParamIn,
    const void * restrict                       pDataIn,
    const vednnFilterParam_t * restrict         pParamKernel,
    const void * restrict                       pDataKernel,
    const vednnConvolutionParam_t * restrict    pParamConv,
    const vednnTensorParam_t * restrict         pParamOut,
    void * restrict                             pDataOut
) ;

vednnError_t
vednnConvolutionForward_direct_dil1_str1_padsame_ker3_T(
    const vednnTensorParam_t * restrict         pParamIn,
    const void * restrict                       pDataIn,
    const vednnFilterParam_t * restrict         pParamKernel,
    const void * restrict                       pDataKernel,
    const vednnConvolutionParam_t * restrict    pParamConv,
    const vednnTensorParam_t * restrict         pParamOut,
    void * restrict                             pDataOut
) ;

vednnError_t
vednnConvolutionForward_direct_dil1_str1_padsame_ker3_T_remainder(
    const vednnTensorParam_t * restrict         pParamIn,
    const void * restrict                       pDataIn,
    const vednnFilterParam_t * restrict         pParamKernel,
    const void * restrict                       pDataKernel,
    const vednnConvolutionParam_t * restrict    pParamConv,
    const vednnTensorParam_t * restrict         pParamOut,
    void * restrict                             pDataOut,
    int                                         n,
    int                                         group,
    int                                         op
) ;

vednnError_t
vednnConvolutionForward_direct_dil1_str1_padsame_ker3_T_subkernel(
    const vednnTensorParam_t * restrict         pParamIn,
    const void * restrict                       pDataIn,
    const vednnFilterParam_t * restrict         pParamKernel,
    const void * restrict                       pDataKernel,
    const vednnConvolutionParam_t * restrict    pParamConv,
    const vednnTensorParam_t * restrict         pParamOut,
    void * restrict                             pDataOut,
    int                                         n,
    int                                         group,
    int                                         curOutChannelGroupPrime,
    int                                         curOutPixelPrime
) ;

vednnError_t
vednnConvolutionForward_direct_dil1_str1_padsame_ker3_c1(
    const vednnTensorParam_t * restrict         pParamIn,
    const void * restrict                       pDataIn,
    const vednnFilterParam_t * restrict         pParamKernel,
    const void * restrict                       pDataKernel,
    const vednnConvolutionParam_t * restrict    pParamConv,
    const vednnTensorParam_t * restrict         pParamOut,
    void * restrict                             pDataOut
) ;

vednnError_t
vednnConvolutionForward_direct_dil1_str1_padsame_ker3_c1_owU128(
    const vednnTensorParam_t * restrict         pParamIn,
    const void * restrict                       pDataIn,
    const vednnFilterParam_t * restrict         pParamKernel,
    const void * restrict                       pDataKernel,
    const vednnConvolutionParam_t * restrict    pParamConv,
    const vednnTensorParam_t * restrict         pParamOut,
    void * restrict                             pDataOut
) ;

vednnError_t
vednnConvolutionForward_direct_dil1_str1_padsame_ker3_c1024x(
    const vednnTensorParam_t * restrict         pParamIn,
    const void * restrict                       pDataIn,
    const vednnFilterParam_t * restrict         pParamKernel,
    const void * restrict                       pDataKernel,
    const vednnConvolutionParam_t * restrict    pParamConv,
    const vednnTensorParam_t * restrict         pParamOut,
    void * restrict                             pDataOut
) ;

vednnError_t
vednnConvolutionForward_direct_dil1_str1_padsame_ker3_c1024x_T(
    const vednnTensorParam_t * restrict         pParamIn,
    const void * restrict                       pDataIn,
    const vednnFilterParam_t * restrict         pParamKernel,
    const void * restrict                       pDataKernel,
    const vednnConvolutionParam_t * restrict    pParamConv,
    const vednnTensorParam_t * restrict         pParamOut,
    void * restrict                             pDataOut
) ;

vednnError_t
vednnConvolutionForward_direct_dil1_str1_padsame_ker3_c1024x_T_subkernel(
    const vednnTensorParam_t * restrict         pParamIn,
    const void * restrict                       pDataIn,
    const vednnFilterParam_t * restrict         pParamKernel,
    const void * restrict                       pDataKernel,
    const vednnConvolutionParam_t * restrict    pParamConv,
    const vednnTensorParam_t * restrict         pParamOut,
    void * restrict                             pDataOut,
    int                                         n,
    int                                         group,
    int                                         curOutChannelGroupPrime,
    int                                         curOutPixelPrime
) ;

vednnError_t
vednnConvolutionForward_direct_dil1_str1_padsame_ker5(
    const vednnTensorParam_t * restrict         pParamIn,
    const void * restrict                       pDataIn,
    const vednnFilterParam_t * restrict         pParamKernel,
    const void * restrict                       pDataKernel,
    const vednnConvolutionParam_t * restrict    pParamConv,
    const vednnTensorParam_t * restrict         pParamOut,
    void * restrict                             pDataOut
) ;

vednnError_t
vednnConvolutionForward_direct_dil1_str1_padsame_ker5_owU128(
    const vednnTensorParam_t * restrict         pParamIn,
    const void * restrict                       pDataIn,
    const vednnFilterParam_t * restrict         pParamKernel,
    const void * restrict                       pDataKernel,
    const vednnConvolutionParam_t * restrict    pParamConv,
    const vednnTensorParam_t * restrict         pParamOut,
    void * restrict                             pDataOut
) ;

vednnError_t
vednnConvolutionForward_direct_dil1_str1_padsame_ker2(
    const vednnTensorParam_t * restrict         pParamIn,
    const void * restrict                       pDataIn,
    const vednnFilterParam_t * restrict         pParamKernel,
    const void * restrict                       pDataKernel,
    const vednnConvolutionParam_t * restrict    pParamConv,
    const vednnTensorParam_t * restrict         pParamOut,
    void * restrict                             pDataOut
) ;

#ifdef __cplusplus
}  /* extern "C" */
#endif
// vim: ts=4 sw=4 et
#endif /* SRC_VEDNNCONVOLUTION_FORWARD_H_ */
