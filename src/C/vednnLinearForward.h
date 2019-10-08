#ifndef SRC_VEDNNLINEARFORWARD_H_
#define SRC_VEDNNLINEARFORWARD_H_

#include "vednn.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define VEDNN_LINEARFWD_ARGS \
    const uint64_t inDim, \
    const uint64_t outDim, \
    const uint64_t nBatch, \
    const void *   pDataIn, \
    const void *   pDataWeight, \
    void *         pDataOut
#define VEDNN_LINEARFWD_ARGS_LIST inDim, outDim, nBatch, \
pDataIn, pDataWeight, pDataOut

typedef
vednnError_t (*vednnLinearForward_t)( VEDNN_LINEARFWD_ARGS );

#define VEDNN_LINEARFWD_DECL(IMPL) vednnError_t \
vednnLinearForward_##IMPL( VEDNN_LINEARFWD_ARGS )

VEDNN_LINEARFWD_DECL(default);
VEDNN_LINEARFWD_DECL(oU32);
VEDNN_LINEARFWD_DECL(o2X_woaligned);

#ifdef __cplusplus
}//extern "C"
#endif
// vim: ts=4 sw=4 et ai
#endif /* SRC_VEDNNLINEARFORWARD_H_ */
