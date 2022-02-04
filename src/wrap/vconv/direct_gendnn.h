
#include "vednnConvolutionForward.h"
#include "vednnConvolutionBackwardData.h"
#include "vednnConvolutionBackwardFilter.h"

//#include "gemm_convolution.hpp"

#ifdef __cplusplus
extern "C" { //}
#endif

vednnError_t
vednnConvolutionForward_direct_gendnn(
        VEDNN_CONVFWD_ARGS);

#if 0 // this is a NOT-IMPLEMENTED STUB
vednnError_t vednnConvolutionBackwardData_direct_gendnn(
    const vednnTensorParam_t         *pParamGradIn,
    const void                 *pDataGradIn,
    const vednnFilterParam_t        *pParamKernel,
    const void                 *pDataKernel,
    const vednnTensorParam_t         *pParamGradOut,
    void                 *pDataGradOut,
    const vednnConvolutionParam_t    *pParamConv,
    vednnConvolutionAlgorithm_t     algo
);
#endif

#if 0 // this is a NOT-IMPLEMENTED STUB
vednnError_t vednnConvolutionBackwardFilter_direct_gendnn(
    const vednnTensorParam_t         *pParamIn,
    const void                 *pDataIn,
    const vednnTensorParam_t         *pParamGradOut,
    const void                 *pDataGradOut,
    const vednnFilterParam_t        *pParamGradKernel,
    void                 *pDataGradKernel,
    const vednnConvolutionParam_t    *pParamConv,
    vednnConvolutionAlgorithm_t     algo
);
#endif

#ifdef __cplusplus
} //extern "C"
#endif
