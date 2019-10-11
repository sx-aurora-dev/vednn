#ifndef CONVOLUTION_GEMM_H
#define CONVOLUTION_GEMM_H
#include "vednn.h"

#ifdef __GNUC__
#define restrict __restrict
#endif
#ifdef __cplusplus
extern "C" {
#endif /*}*/


// The following are original 'C' api from test/ (now "detail")

/** pColBuff is malloced as:
 * `(float*) malloc(pColrows*pColcols*getTensorDataSize(pConv->pParamIn));`
 * where
 * `size_t pColrows = pConv->pParamKernel->inChannel * pConv->pParamKernel->width * pConv->pParamKernel->height;`
 * and
 * `size_t pColcols = pConv->pParamOut->width * pConv->pParamOut->height;`
 *  This can be quite big, and usual impls may do "one col at a time" (re-using the buffer)
 *
 *  This can be a libvednn scratchpad, I think (thread-local should be ok, shared among omp threads
 *  during gemm call)
 *
 *  note getTensorDataSize for pParam->dtype==DTYPE_FLOAT is sizeof(float). (vednn_helper.c)
 *
 *  pOnesize is float[ow*oh] and is used to add in the bias via a second gemm_ call.
 *
 *  For libvednn, these scratchpads are now auto-supplied (re-usable between multiple
 *  layers, layer types).
 */
vednnError_t
convolution_forward_gemm(
    const vednnTensorParam_t * restrict pParamIn, const void * restrict pDataIn,
    const vednnFilterParam_t * restrict pParamKernel, const void * restrict pDataKernel,
    const vednnBiasParam_t * restrict pParamBias, const void * restrict pDataBias,
    const vednnTensorParam_t * restrict pParamOut, void * restrict pDataOut,
    const float * restrict pOne,  float * restrict pColBuff,
    const vednnConvolutionParam_t * restrict pParamConv ) ;

vednnError_t
convolution_backward_data_gemm(
    const vednnTensorParam_t * restrict pParamGradOut, const void * restrict pDataGradOut,
    const vednnFilterParam_t * restrict pParamKernel, const void * restrict pDataKernel,
    const vednnTensorParam_t * restrict pParamGradIn, void * restrict pDataGradIn,
    float * restrict pColBuff,
    const vednnConvolutionParam_t * restrict pParamConv ) ;

vednnError_t
convolution_backward_filter_gemm(
    const vednnTensorParam_t * restrict pParamIn, const void * restrict pDataIn,
    const vednnTensorParam_t * restrict pParamGradOut, const void * restrict pDataGradOut,
    const vednnFilterParam_t * restrict pParamGradKernel, void * restrict pDataGradKernel,
    float * restrict pColBuff,
    const vednnConvolutionParam_t * restrict pParamConv ) ;

#ifdef __cplusplus /*{*/
}
#endif
#endif // CONVOLUTION_GEMM_H
