
#ifndef SRC_VEDNNLINEARBACKWARDDATA_H_
#define SRC_VEDNNLINEARBACKWARDDATA_H_

#include "vednn.h"

typedef
vednnError_t (*vednnLinearBackwardData_t)(
    const uint64_t			inDim,
    const uint64_t			outDim,
    const uint64_t			nBatch,
    const void * 			pDataGradOut,
    const void * 			pDataWeight,
    void * 				pDataGradIn
) ;

vednnError_t vednnLinearBackwardData_default(
    const uint64_t			inDim,
    const uint64_t			outDim,
    const uint64_t			nBatch,
    const void * 			pDataGradOut,
    const void * 			pDataWeight,
    void * 				pDataGradIn
) ;

vednnError_t vednnLinearBackwardData_o2X_woaligned(
    const uint64_t			inDim,
    const uint64_t			outDim,
    const uint64_t			nBatch,
    const void * 			pDataGradOut,
    const void * 			pDataWeight,
    void * 				pDataGradIn
) ;

vednnError_t vednnLinearBackwardData_oU128(
    const uint64_t			inDim,
    const uint64_t			outDim,
    const uint64_t			nBatch,
    const void * 			pDataGradOut,
    const void * 			pDataWeight,
    void * 				pDataGradIn
) ;

vednnError_t vednnLinearBackwardData_o2XU128_waligned(
    const uint64_t			inDim,
    const uint64_t			outDim,
    const uint64_t			nBatch,
    const void * 			pDataGradOut,
    const void * 			pDataWeight,
    void * 				pDataGradIn
) ;

vednnError_t vednnLinearBackwardData_oU256(
    const uint64_t			inDim,
    const uint64_t			outDim,
    const uint64_t			nBatch,
    const void * 			pDataGradOut,
    const void * 			pDataWeight,
    void * 				pDataGradIn
) ;

#endif /* SRC_VEDNNLINEARBACKWARDDATA_H_ */
