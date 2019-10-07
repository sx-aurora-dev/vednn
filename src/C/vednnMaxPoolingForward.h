#ifndef _VEDNNMAXPOOLINGFORWARD_H_
#define _VEDNNMAXPOOLINGFORWARD_H_

#include "vednn.h"

#ifdef __cplusplus
extern "C" {
#endif

#define VEDNN_MAXPOOLINGFWD_ARGS \
    const vednnTensorParam_t  *pParamIn, \
    const void                *pDataIn, \
    const vednnTensorParam_t  *pParamOut, \
    void                      *pDataOut, \
    const vednnPoolingParam_t *pParamPool

#define VEDNN_MAXPOOLINGFWD_ARGS_LIST pParamIn, pDataIn, \
    pParamOut, pDataOut, pParamPool

typedef
vednnError_t (*vednnMaxPoolForward_t) ( VEDNN_MAXPOOLINGFWD_ARGS );

#define VEDNN_MAXPOOLINGFWD_DECL(IMPL) vednnError_t \
vednnMaxPoolingForward_##IMPL( VEDNN_MAXPOOLINGFWD_ARGS )

VEDNN_MAXPOOLINGFWD_DECL(default);
VEDNN_MAXPOOLINGFWD_DECL(regular);
VEDNN_MAXPOOLINGFWD_DECL(regular_owU128);
VEDNN_MAXPOOLINGFWD_DECL(regular_ww2X_owU128_ialigned);

#ifdef __cplusplus
}//extern "C"
#endif
// vim: sw=4 ts=4 et
#endif /* _VEDNNMAXPOOLINGFORWARD_H_ */
