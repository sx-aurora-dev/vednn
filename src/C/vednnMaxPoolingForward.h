
#ifndef _VEDNNMAXPOOLINGFORWARD_H_
#define _VEDNNMAXPOOLINGFORWARD_H_

#include "vednn.h"

typedef
vednnError_t (*vednnMaxPoolForward_t) (
    const vednnTensorParam_t 		*pParamIn,
    const void 				*pDataIn,
    const vednnTensorParam_t 		*pParamOut,
    void 				*pDataOut,
    const vednnPoolingParam_t		*pParamPool
) ;

vednnError_t vednnMaxPoolingForward_default(
    const vednnTensorParam_t 		*pParamIn,
    const void 				*pDataIn,
    const vednnTensorParam_t 		*pParamOut,
    void 				*pDataOut,
    const vednnPoolingParam_t		*pParamPool
) ;


vednnError_t vednnMaxPoolingForward_regular(
    const vednnTensorParam_t 		*pParamIn,
    const void 				*pDataIn,
    const vednnTensorParam_t 		*pParamOut,
    void 				*pDataOut,
    const vednnPoolingParam_t		*pParamPool
) ;

vednnError_t vednnMaxPoolingForward_regular_owU128(
    const vednnTensorParam_t 		*pParamIn,
    const void 				*pDataIn,
    const vednnTensorParam_t 		*pParamOut,
    void 				*pDataOut,
    const vednnPoolingParam_t		*pParamPool
) ;

vednnError_t vednnMaxPoolingForward_regular_ww2X_owU128_ialigned(
    const vednnTensorParam_t 		*pParamIn,
    const void 				*pDataIn,
    const vednnTensorParam_t 		*pParamOut,
    void 				*pDataOut,
    const vednnPoolingParam_t		*pParamPool
) ;

#endif /* _VEDNNMAXPOOLINGFORWARD_H_ */
