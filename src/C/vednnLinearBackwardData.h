#ifndef SRC_VEDNNLINEARBACKWARDDATA_H_
#define SRC_VEDNNLINEARBACKWARDDATA_H_

#include "vednn.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define VEDNN_LINEARBKD_ARGS \
    const uint64_t inDim, \
    const uint64_t outDim, \
    const uint64_t nBatch, \
    const void *   pDataGradOut, \
    const void *   pDataWeight, \
    void *         pDataGradIn
#define VEDNN_LINEARBKD_ARGS_LIST inDim, outDim, nBatch, \
    pDataGradOut, pDataWeight, pDataGradIn

typedef
vednnError_t (*vednnLinearBackwardData_t)( VEDNN_LINEARBKD_ARGS );

#define VEDNN_LINEARBKD_DECL(IMPL) vednnError_t \
vednnLinearBackwardData_##IMPL( VEDNN_LINEARBKD_ARGS )

VEDNN_LINEARBKD_DECL(default);
VEDNN_LINEARBKD_DECL(o2X_woaligned);
VEDNN_LINEARBKD_DECL(oU128);
VEDNN_LINEARBKD_DECL(o2XU128_waligned);
VEDNN_LINEARBKD_DECL(oU256);

#ifdef __cplusplus
}//extern "C"
#endif
// vim: sw=4 ts=4 et ai
#endif /* SRC_VEDNNLINEARBACKWARDDATA_H_ */
