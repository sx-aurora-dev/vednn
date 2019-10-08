#ifndef _VEDNNMAXPOOLINGBACKWARD_H_
#define _VEDNNMAXPOOLINGBACKWARD_H_

#include "vednn.h"

#ifdef __cplusplus
extern "C" {
#endif

#define VEDNN_MAXPOOLINGBKW_ARGS \
    const vednnTensorParam_t  *pParamGradOut, \
    const void                *pDataGradOut, \
    const vednnTensorParam_t  *pParamOut, \
    const void                *pDataOut, \
    const vednnTensorParam_t  *pParamIn, \
    const void                *pDataIn, \
    const vednnTensorParam_t  *pParamGradIn, \
    void                      *pDataGradIn, \
    const vednnPoolingParam_t *pParamPool
#define VEDNN_MAXPOOLINGBKW_ARGS_LIST pParamGradOut, pDataGradOut, \
    pParamOut, pDataOut, pParamIn, pDataIn, \
    pParamGradIn, pDataGradIn, pParamPool

typedef
vednnError_t (*vednnMaxPoolBackward_t)( VEDNN_MAXPOOLINGBKW_ARGS );

#define VEDNN_MAXPOOLINGBKW_DECL(IMPL) vednnError_t \
vednnMaxPoolingBackward_##IMPL( VEDNN_MAXPOOLINGBKW_ARGS );

VEDNN_MAXPOOLINGBKW_DECL(default);
VEDNN_MAXPOOLINGBKW_DECL(regular);
VEDNN_MAXPOOLINGBKW_DECL(regular_owU128);
VEDNN_MAXPOOLINGBKW_DECL(regular_ww2X_owU128_ialigned);

#ifdef __cplusplus
}//extern "C"
#endif
// vim: ts=4 sw=4 et ai
#endif /* _VEDNNMAXPOOLINGFORWARD_H_ */
