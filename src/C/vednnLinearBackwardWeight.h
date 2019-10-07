#ifndef SRC_VEDNNLINEARBACKWARDWEIGHT_H_
#define SRC_VEDNNLINEARBACKWARDWEIGHT_H_

#include "vednn.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define VEDNN_LINEARBKW_ARGS \
    const uint64_t inDim, \
    const uint64_t outDim, \
    const uint64_t nBatch, \
    const void *   pDataIn, \
    const void *   pDataGradOut, \
    void *         pDataGradWeight
#define VEDNN_LINEARBKW_ARGS_LIST inDim, outDim, nBatch, \
    pDataIn, pDataGradOut, pDataGradWeight

#ifndef VEDNN_USE_OPENMP
#define VEDNN_LINEARBKW_OMPARGS VEDNN_LINEARBKW_ARGS

#else
#define VEDNN_LINEARBKW_OMPARGS VEDNN_LINEARBKW_ARGS, \
    const uint64_t inDimBegin, \
    const uint64_t inDimEnd
#endif

typedef
vednnError_t (*vednnLinearBackwardWeight_t)( VEDNN_LINEARBKW_OMPARGS );

#define VEDNN_LINEARBKW_DECL(IMPL) vednnError_t \
vednnLinearBackwardWeight_##IMPL( VEDNN_LINEARBKW_OMPARGS );

VEDNN_LINEARBKW_DECL(o2X_woaligned);
VEDNN_LINEARBKW_DECL(o2XU128_woaligned);
VEDNN_LINEARBKW_DECL(default);

#ifdef __cplusplus
}//extern "C"
#endif
// vim: sw=4 ts=4 et ai
#endif /* SRC_VEDNNLINEARBACKWARDWEIGHT_H_ */
