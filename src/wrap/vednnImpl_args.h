#ifndef VEDNNCONVOLUTION_ARGS_H
#define VEDNNCONVOLUTION_ARGS_H
/** \file
 */
#include "vednn.h"

#ifdef __cplusplus
extern "C" {
#endif /*}*/

/// \group standard parameter packs (reduce copy-paste errors)
/** Split the original args of src/C/ libvednn headers into parameters
 * and data arguments. <B>Splitting follows argument order established
 * by \e libveddn public API</B> as in \ref libvednn.h
 */
//@{
#define VEDNN_PARAMS_CONV_FORWARD \
    const vednnTensorParam_t *      restrict pParamIn, \
    const vednnFilterParam_t *      restrict pParamKernel, \
    const vednnTensorParam_t *      restrict pParamOut, \
    const vednnConvolutionParam_t * restrict pParamConv, \
    vednnConvolutionAlgorithm_t     algo
#define VEDNN_PARAMS_CONV_FORWARD_LIST pParamIn, pParamKernel, pParamOut, pParamConv, algo

#define VEDNN_DATARG_CONV_FORWARD_OPENMP /* tbd? */
#define VEDNN_DATARG_CONV_FORWARD \
    /*void const * const restrict pd,*/ \
    void const *       restrict pDataIn, \
    void const *       restrict pDataKernel, \
    void *             restrict pDataOut \
    VEDNN_DATARG_CONV_FORWARD_OPENMP
#define VEDNN_DATARG_CONV_FORWARD_LIST pDataIn, pDataKernel, pDataOut

#define VEDNN_PARAMS_CONV_FORWARDADDBIAS \
    const vednnTensorParam_t *      restrict pParamIn, \
    const vednnFilterParam_t *      restrict pParamKernel, \
    const vednnBiasParam_t *        restrict pParamBias, \
    const vednnTensorParam_t *      restrict pParamOut, \
    const vednnConvolutionParam_t * restrict pParamConv, \
    vednnConvolutionAlgorithm_t     algo
#define VEDNN_PARAMS_CONV_FORWARDADDBIAS_LIST pParamIn, pParamKernel, pParamBias, pParamOut, pParamConv, algo

#define VEDNN_DATARG_CONV_FORWARDBIAS_OPENMP /* tbd? */
#define VEDNN_DATARG_CONV_FORWARDADDBIAS \
    /*void const * const restrict pd,*/ \
    void const * restrict pDataIn, \
    void const * restrict pDataKernel, \
    void const * restrict pDataBias, \
    void *       restrict pDataOut \
    VEDNN_DATARG_CONV_FORWARDBIAS_OPENMP
#define VEDNN_DATARG_CONV_FORWARDADDBIAS_LIST pDataIn, pDataKernel, pDataBias, pDataOut

#define VEDNN_PARAMS_CONV_BACKWARD_DATA \
    const vednnTensorParam_t      * pParamGradIn, \
    const vednnFilterParam_t      * pParamKernel, \
    const vednnTensorParam_t      * pParamGradOut, \
    const vednnConvolutionParam_t * pParamConv, \
    vednnConvolutionAlgorithm_t     algo
#define VEDNN_PARAMS_CONV_BACKWARD_DATA_LIST pParamGradIn, pParamKernel, pParamGradOut, pParamConv, algo

#define VEDNN_DATARG_CONV_BACKWARD_DATA_OPENMP /* tbd? */
#define VEDNN_DATARG_CONV_BACKWARD_DATA \
    /*void const * const restrict pd,*/ \
    void *       restrict pDataGradIn, \
    const void * restrict pDataKernel, \
    const void * restrict pDataGradOut \
    VEDNN_DATARG_CONV_BACKWARD_DATA_OPENMP
#define VEDNN_DATARG_CONV_BACKWARD_DATA_LIST pDataGradIn, pDataKernel, pDataGradOut
    /*void const * const restrict pd,*/

#define VEDNN_PARAMS_CONV_BACKWARD_FILTER \
    const vednnTensorParam_t      * pParamIn, \
    const vednnTensorParam_t      * pParamGradOut, \
    const vednnFilterParam_t      * pParamGradKernel, \
    const vednnConvolutionParam_t * pParamConv, \
    vednnConvolutionAlgorithm_t     algo
#define VEDNN_PARAMS_CONV_BACKWARD_FILTER_LIST pParamIn, pParamGradOut, pParamGradKernel, pParamConv, algo

#ifdef VEDNN_USE_OPENMP
#define VEDNN_DATARG_CONV_BACKWARD_FILTER_OPENMP \
    , const int64_t beginOChannel \
    , const int64_t   nOChannel
#define VEDNN_DATARG_CONV_BACKWARD_FILTER_OPENMP_LIST , beginOChannel, nOChannel
#else
#define VEDNN_DATARG_CONV_BACKWARD_FILTER_OPENMP
#define VEDNN_DATARG_CONV_BACKWARD_FILTER_OPENMP_LIST
#endif

#define VEDNN_DATARG_CONV_BACKWARD_FILTER \
    /*void const * const restrict pd,*/ \
    const void * restrict pDataIn, \
    const void * restrict pDataGradOut, \
    void *       restrict pDataGradKernel
#define VEDNN_DATARG_CONV_BACKWARD_FILTER_LIST pDataIn, pDataGradOut, pDataGradKernel
//@}

/// \group typedefs for convolution Impls that accept private-data
//@{
typedef vednnError_t (*vednnConvForwardPd_t)( void const* const pd,
                VEDNN_DATARG_CONV_FORWARD );
typedef vednnError_t (*vednnConvForwardAddBiasPd_t)( void const* const pd,
                VEDNN_DATARG_CONV_FORWARDADDBIAS );
typedef vednnError_t (*vednnConvBackwardDataPd_t)( void const* const pd,
                VEDNN_DATARG_CONV_BACKWARD_DATA );
typedef vednnError_t (*vednnConvBackwardFilterPd_t)( void const* const pd,
                VEDNN_DATARG_CONV_BACKWARD_DATA );
//@}

#define VEDNN_PARAMS_LIN_FORWARD \
    const unsigned long   inDim, \
    const unsigned long   outDim, \
    const unsigned long   nBatch
#define VEDNN_DATARG_LIN_FORWARD \
    const void * restrict pDataIn, \
    void *       restrict pDataWeight, \
    const void * restrict pDataOut
#define VEDNN_PARAMS_LIN_FORWARD_LIST inDim, outDim, nBatch
#define VEDNN_DATARG_LIN_FORWARD_LIST pDataIn, pDataWeight, pDataOut

#define VEDNN_PARAMS_MAXPOOL_FORWARD \
    const vednnTensorParam_t  *pParamIn, \
    const vednnTensorParam_t  *pParamOut, \
    const vednnPoolingParam_t *pParamPool
#define VEDNN_DATARG_MAXPOOL_FORWARD \
    const void                *pDataIn, \
    void                      *pDataOut
#define VEDNN_PARAMS_MAXPOOL_FORWARD_LIST pParamIn, pParamOut, pParamPool 
#define VEDNN_DATARG_MAXPOOL_FORWARD_LIST pDataIn,  pDataOut

#define VEDNN_PARAMS_MAXPOOL_BACKWARD \
    const vednnTensorParam_t  *pParamGradOut, \
    const vednnTensorParam_t  *pParamOut, \
    const vednnTensorParam_t  *pParamIn, \
    const vednnTensorParam_t  *pParamGradIn, \
    const vednnPoolingParam_t *pParamPool
#define VEDNN_DATARG_MAXPOOL_BACKWARD \
    const void                *pDataGradOut, \
    const void                *pDataOut, \
    const void                *pDataIn, \
    void                      *pDataGradIn
#define VEDNN_PARAMS_MAXPOOL_BACKWARD_LIST pParamGradOut, pParamOut, pParamIn, pParamGradIn, pParamPool
#define VEDNN_DATARG_MAXPOOL_BACKWARD_LIST pDataGradOut, pDataOut, pDataIn, pDataGradIn

#define VEDNN_PARAMS_ACT_FORWARD \
    const vednnActivationMode_t mode, \
    const unsigned long         nElements
#define VEDNN_DATARG_ACT_FORWARD \
    const void                  *pDataIn, \
    void                        *pDataOut
#define VEDNN_PARAMS_ACT_FORWARD_LIST mode, nElements
#define VEDNN_DATARG_ACT_FORWARD_LIST pDataIn, pDataOut
#define VEDNN_PARAMS_ACT_BACKWARD \
    const vednnActivationMode_t mode, \
    const unsigned long         nElements
#define VEDNN_DATARG_ACT_BACKWARD \
    const void                  *pDataGradOut, \
    const void                  *pDataIn, \
    void                        *pDataGradIn
#define VEDNN_PARAMS_ACT_BACKWARD_LIST mode, nElements
#define VEDNN_DATARG_ACT_BACKWARD_LIST pDataGradOut, pDataIn, pDataGradIn

#ifdef __cplusplus /*{*/
}
#endif
// vim: et ts=4 sw=4 cindent cino=^=l0,\:0,N-s
#endif // VEDNNCONVOLUTION_ARGS_H
