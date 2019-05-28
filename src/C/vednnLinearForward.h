
#ifndef SRC_VEDNNLINEARFORWARD_H_
#define SRC_VEDNNLINEARFORWARD_H_

#include "vednn.h"

typedef
vednnError_t (*vednnLinearForward_t)(
    const uint64_t			inDim,
    const uint64_t			outDim,
    const uint64_t			nBatch,
    const void * 			pDataIn,
    const void * 			pDataWeight,
    void * 				pDataOut
) ;

vednnError_t vednnLinearForward_default(
    const uint64_t			inDim,
    const uint64_t			outDim,
    const uint64_t			nBatch,
    const void * 			pDataIn,
    const void * 			pDataWeight,
    void * 				pDataOut
) ;

vednnError_t vednnLinearForward_oU32(
    const uint64_t			inDim,
    const uint64_t			outDim,
    const uint64_t			nBatch,
    const void * 			pDataIn,
    const void * 			pDataWeight,
    void * 				pDataOut
) ;

vednnError_t vednnLinearForward_o2X_woaligned(
    const uint64_t			inDim,
    const uint64_t			outDim,
    const uint64_t			nBatch,
    const void * 			pDataIn,
    const void * 			pDataWeight,
    void * 				pDataOut
) ;

#endif /* SRC_VEDNNLINEARFORWARD_H_ */
