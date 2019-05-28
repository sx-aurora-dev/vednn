
#ifndef SRC_VEDNNLINEARBACKWARDWEIGHT_H_
#define SRC_VEDNNLINEARBACKWARDWEIGHT_H_

#include "vednn.h"

typedef
vednnError_t (*vednnLinearBackwardWeight_t)(
    const uint64_t			inDim,
    const uint64_t			outDim,
    const uint64_t			nBatch,
    const void * 			pDataIn,
    const void * 			pDataGradOut,
    void * 				pDataGradWeight
#ifdef VEDNN_USE_OPENMP
    ,
    const uint64_t			inDimBegin,
    const uint64_t			inDimEnd
#endif
) ;

vednnError_t vednnLinearBackwardWeight_default(
    const uint64_t			inDim,
    const uint64_t			outDim,
    const uint64_t			nBatch,
    const void * 			pDataIn,
    const void * 			pDataGradOut,
    void * 				pDataGradWeight
#ifdef VEDNN_USE_OPENMP
    ,
    const uint64_t			inDimBegin,
    const uint64_t			inDimEnd
#endif

) ;

#endif /* SRC_VEDNNLINEARBACKWARDWEIGHT_H_ */
