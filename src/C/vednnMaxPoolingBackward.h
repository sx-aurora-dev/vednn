
#ifndef _VEDNNMAXPOOLINGBACKWARD_H_
#define _VEDNNMAXPOOLINGBACKWARD_H_

#include "vednn.h"

typedef
vednnError_t (*vednnMaxPoolBackward_t)(
    const vednnTensorParam_t 		*pParamGradOut,
    const void 				*pDataGradOut,
    const vednnTensorParam_t 		*pParamOut,
    const void 				*pDataOut,
    const vednnTensorParam_t 		*pParamIn,
    const void 				*pDataIn,
    const vednnTensorParam_t 		*pParamGradIn,
    void 				*pDataGradIn,
    const vednnPoolingParam_t		*pParamPool
) ;

vednnError_t vednnMaxPoolingBackward_default(
    const vednnTensorParam_t 		*pParamGradOut,
    const void 				*pDataGradOut,
    const vednnTensorParam_t 		*pParamOut,
    const void 				*pDataOut,
    const vednnTensorParam_t 		*pParamIn,
    const void 				*pDataIn,
    const vednnTensorParam_t 		*pParamGradIn,
    void 				*pDataGradIn,
    const vednnPoolingParam_t		*pParamPool
) ;

vednnError_t vednnMaxPoolingBackward_regular(
    const vednnTensorParam_t 		*pParamGradOut,
    const void 				*pDataGradOut,
    const vednnTensorParam_t 		*pParamOut,
    const void 				*pDataOut,
    const vednnTensorParam_t 		*pParamIn,
    const void 				*pDataIn,
    const vednnTensorParam_t 		*pParamGradIn,
    void 				*pDataGradIn,
    const vednnPoolingParam_t		*pParamPool
) ;

vednnError_t vednnMaxPoolingBackward_regular_owU128(
    const vednnTensorParam_t 		*pParamGradOut,
    const void 				*pDataGradOut,
    const vednnTensorParam_t 		*pParamOut,
    const void 				*pDataOut,
    const vednnTensorParam_t 		*pParamIn,
    const void 				*pDataIn,
    const vednnTensorParam_t 		*pParamGradIn,
    void 				*pDataGradIn,
    const vednnPoolingParam_t		*pParamPool
) ;

vednnError_t vednnMaxPoolingBackward_regular_ww2X_owU128_ialigned(
    const vednnTensorParam_t 		*pParamGradOut,
    const void 				*pDataGradOut,
    const vednnTensorParam_t 		*pParamOut,
    const void 				*pDataOut,
    const vednnTensorParam_t 		*pParamIn,
    const void 				*pDataIn,
    const vednnTensorParam_t 		*pParamGradIn,
    void 				*pDataGradIn,
    const vednnPoolingParam_t		*pParamPool
) ;

#endif /* _VEDNNMAXPOOLINGFORWARD_H_ */
