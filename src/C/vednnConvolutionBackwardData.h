
#ifndef SRC_VEDNNCONVOLUTION_BACKWARD_DATA_H_
#define SRC_VEDNNCONVOLUTION_BACKWARD_DATA_H_

#include "vednn.h"

typedef
vednnError_t (*vednnConvBackwardData_t) (
    const vednnTensorParam_t * restrict         pParamGradOut,
    const void * restrict                       pDataGradOut,
    const vednnFilterParam_t * restrict         pParamKernel,
    const void * restrict                       pDataKernel,
    const vednnConvolutionParam_t * restrict    pParamConv,
    const vednnTensorParam_t * restrict         pParamGradIn,
    void * restrict                             pDataGradIn
) ;
/** this is the signature of \c pFunc arg to the wrapper, which we use
 * directly for low-level impls marked as VEDNN_WRAP_NONE in libvednnx */
typedef vednnConvBackwardData_t vednnConvBackwardData_nowrap_t;

vednnError_t
vednnConvolutionBackwardData_direct_default(
    const vednnTensorParam_t * restrict         pParamGradOut,
    const void * restrict                       pDataGradOut,
    const vednnFilterParam_t * restrict         pParamKernel,
    const void * restrict                       pDataKernel,
    const vednnConvolutionParam_t * restrict    pParamConv,
    const vednnTensorParam_t * restrict         pParamGradIn,
    void * restrict                             pDataGradIn
) ;

vednnError_t
vednnConvolutionBackwardData_direct_gemm(
    const vednnTensorParam_t * restrict         pParamGradOut,
    const void * restrict                       pDataGradOut,
    const vednnFilterParam_t * restrict         pParamKernel,
    const void * restrict                       pDataKernel,
    const vednnConvolutionParam_t * restrict    pParamConv,
    const vednnTensorParam_t * restrict         pParamGradIn,
    void * restrict                             pDataGradIn
) ;

vednnError_t
vednnConvolutionBackwardData_direct_gemmA(
    const vednnTensorParam_t * restrict         pParamGradOut,
    const void * restrict                       pDataGradOut,
    const vednnFilterParam_t * restrict         pParamKernel,
    const void * restrict                       pDataKernel,
    const vednnConvolutionParam_t * restrict    pParamConv,
    const vednnTensorParam_t * restrict         pParamGradIn,
    void * restrict                             pDataGradIn
) ;

vednnError_t
vednnConvolutionBackwardData_direct_vecC(
    const vednnTensorParam_t * restrict         pParamGradOut,
    const void * restrict                       pDataGradOut,
    const vednnFilterParam_t * restrict         pParamKernel,
    const void * restrict                       pDataKernel,
    const vednnConvolutionParam_t * restrict    pParamConv,
    const vednnTensorParam_t * restrict         pParamGradIn,
    void * restrict                             pDataGradIn
) ;

vednnError_t
vednnConvolutionBackwardData_direct_dil1_str2_pad2_ker5(
    const vednnTensorParam_t * restrict         pParamGradOut,
    const void * restrict                       pDataGradOut,
    const vednnFilterParam_t * restrict         pParamKernel,
    const void * restrict                       pDataKernel,
    const vednnConvolutionParam_t * restrict    pParamConv,
    const vednnTensorParam_t * restrict         pParamGradIn,
    void * restrict                             pDataGradIn
) ;
vednnError_t
vednnConvolutionBackwardData_direct_dil1_str2_pad2_ker5_iwU128(
    const vednnTensorParam_t * restrict         pParamGradOut,
    const void * restrict                       pDataGradOut,
    const vednnFilterParam_t * restrict         pParamKernel,
    const void * restrict                       pDataKernel,
    const vednnConvolutionParam_t * restrict    pParamConv,
    const vednnTensorParam_t * restrict         pParamGradIn,
    void * restrict                             pDataGradIn
) ;

vednnError_t
vednnConvolutionBackwardData_direct_ker5(
    const vednnTensorParam_t * restrict         pParamGradOut,
    const void * restrict                       pDataGradOut,
    const vednnFilterParam_t * restrict         pParamKernel,
    const void * restrict                       pDataKernel,
    const vednnConvolutionParam_t * restrict    pParamConv,
    const vednnTensorParam_t * restrict         pParamGradIn,
    void * restrict                             pDataGradIn
) ;


vednnError_t
vednnConvolutionBackwardData_direct_iwU128(
    const vednnTensorParam_t * restrict         pParamGradOut,
    const void * restrict                       pDataGradOut,
    const vednnFilterParam_t * restrict         pParamKernel,
    const void * restrict                       pDataKernel,
    const vednnConvolutionParam_t * restrict    pParamConv,
    const vednnTensorParam_t * restrict         pParamGradIn,
    void * restrict                             pDataGradIn
) ;

vednnError_t
vednnConvolutionBackwardData_direct_ker3_iwU128(
    const vednnTensorParam_t * restrict         pParamGradOut,
    const void * restrict                       pDataGradOut,
    const vednnFilterParam_t * restrict         pParamKernel,
    const void * restrict                       pDataKernel,
    const vednnConvolutionParam_t * restrict    pParamConv,
    const vednnTensorParam_t * restrict         pParamGradIn,
    void * restrict                             pDataGradIn
) ;

vednnError_t
vednnConvolutionBackwardData_direct_ker5_iwU128(
    const vednnTensorParam_t * restrict         pParamGradOut,
    const void * restrict                       pDataGradOut,
    const vednnFilterParam_t * restrict         pParamKernel,
    const void * restrict                       pDataKernel,
    const vednnConvolutionParam_t * restrict    pParamConv,
    const vednnTensorParam_t * restrict         pParamGradIn,
    void * restrict                             pDataGradIn
) ;

vednnError_t
vednnConvolutionBackwardData_direct_dil1_pad0_ker1_owU128(
    const vednnTensorParam_t * restrict         pParamGradOut,
    const void * restrict                       pDataGradOut,
    const vednnFilterParam_t * restrict         pParamKernel,
    const void * restrict                       pDataKernel,
    const vednnConvolutionParam_t * restrict    pParamConv,
    const vednnTensorParam_t * restrict         pParamGradIn,
    void * restrict                             pDataGradIn
) ;


vednnError_t
vednnConvolutionBackwardData_direct_dil1_str1(
    const vednnTensorParam_t * restrict         pParamGradOut,
    const void * restrict                       pDataGradOut,
    const vednnFilterParam_t * restrict         pParamKernel,
    const void * restrict                       pDataKernel,
    const vednnConvolutionParam_t * restrict    pParamConv,
    const vednnTensorParam_t * restrict         pParamGradIn,
    void * restrict                             pDataGradIn
) ;

vednnError_t
vednnConvolutionBackwardData_direct_dil1_str1_iwU128(
    const vednnTensorParam_t * restrict         pParamGradOut,
    const void * restrict                       pDataGradOut,
    const vednnFilterParam_t * restrict         pParamKernel,
    const void * restrict                       pDataKernel,
    const vednnConvolutionParam_t * restrict    pParamConv,
    const vednnTensorParam_t * restrict         pParamGradIn,
    void * restrict                             pDataGradIn
) ;

vednnError_t
vednnConvolutionBackwardData_direct_dil1_str1_pad0_ker3_iwU128(
    const vednnTensorParam_t * restrict         pParamGradOut,
    const void * restrict                       pDataGradOut,
    const vednnFilterParam_t * restrict         pParamKernel,
    const void * restrict                       pDataKernel,
    const vednnConvolutionParam_t * restrict    pParamConv,
    const vednnTensorParam_t * restrict         pParamGradIn,
    void * restrict                             pDataGradIn
) ;

vednnError_t
vednnConvolutionBackwardData_direct_dil1_str1_pad0_ker3_iw2XU256_ow2X_ioaligned(
    const vednnTensorParam_t * restrict         pParamGradOut,
    const void * restrict                       pDataGradOut,
    const vednnFilterParam_t * restrict         pParamKernel,
    const void * restrict                       pDataKernel,
    const vednnConvolutionParam_t * restrict    pParamConv,
    const vednnTensorParam_t * restrict         pParamGradIn,
    void * restrict                             pDataGradIn
) ;

vednnError_t
vednnConvolutionBackwardData_direct_dil1_str1_pad0_ker3_iw2XU32_ow2X_ioaligned(
    const vednnTensorParam_t * restrict         pParamGradOut,
    const void * restrict                       pDataGradOut,
    const vednnFilterParam_t * restrict         pParamKernel,
    const void * restrict                       pDataKernel,
    const vednnConvolutionParam_t * restrict    pParamConv,
    const vednnTensorParam_t * restrict         pParamGradIn,
    void * restrict                             pDataGradIn
) ;


vednnError_t
vednnConvolutionBackwardData_direct_dil1_str1_padsame(
    const vednnTensorParam_t * restrict         pParamGradOut,
    const void * restrict                       pDataGradOut,
    const vednnFilterParam_t * restrict         pParamKernel,
    const void * restrict                       pDataKernel,
    const vednnConvolutionParam_t * restrict    pParamConv,
    const vednnTensorParam_t * restrict         pParamGradIn,
    void * restrict                             pDataGradIn
) ;

vednnError_t
vednnConvolutionBackwardData_direct_dil1_str1_padsame_ker3(
    const vednnTensorParam_t * restrict         pParamGradOut,
    const void * restrict                       pDataGradOut,
    const vednnFilterParam_t * restrict         pParamKernel,
    const void * restrict                       pDataKernel,
    const vednnConvolutionParam_t * restrict    pParamConv,
    const vednnTensorParam_t * restrict         pParamGradIn,
    void * restrict                             pDataGradIn
) ;

vednnError_t
vednnConvolutionBackwardData_direct_dil1_str1_padsame_ker5(
    const vednnTensorParam_t * restrict         pParamGradOut,
    const void * restrict                       pDataGradOut,
    const vednnFilterParam_t * restrict         pParamKernel,
    const void * restrict                       pDataKernel,
    const vednnConvolutionParam_t * restrict    pParamConv,
    const vednnTensorParam_t * restrict         pParamGradIn,
    void * restrict                             pDataGradIn
) ;

vednnError_t
vednnConvolutionBackwardData_direct_dil1_str1_padsame_ker2(
    const vednnTensorParam_t * restrict         pParamGradOut,
    const void * restrict                       pDataGradOut,
    const vednnFilterParam_t * restrict         pParamKernel,
    const void * restrict                       pDataKernel,
    const vednnConvolutionParam_t * restrict    pParamConv,
    const vednnTensorParam_t * restrict         pParamGradIn,
    void * restrict                             pDataGradIn
) ;

vednnError_t
vednnConvolutionBackwardData_direct_dil1_str1_padsame_ker1(
    const vednnTensorParam_t * restrict         pParamGradOut,
    const void * restrict                       pDataGradOut,
    const vednnFilterParam_t * restrict         pParamKernel,
    const void * restrict                       pDataKernel,
    const vednnConvolutionParam_t * restrict    pParamConv,
    const vednnTensorParam_t * restrict         pParamGradIn,
    void * restrict                             pDataGradIn
) ;

// vim: ts=4 sw=4 et
#endif /* SRC_VEDNNCONVOLUTION_BACKWARD_DATA_H_ */
