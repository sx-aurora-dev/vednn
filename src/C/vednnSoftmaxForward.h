
#ifndef _VEDNNSOFTMAXFORWARD_H_
#define _VEDNNSOFTMAXFORWARD_H_

#include "vednn.h"

typedef
vednnError_t (*vednnSoftmaxForward_t) (
    const void 			*pDataIn,
    void 			*pDataOut,
    const uint64_t		nBatch,
    const uint64_t		nClass
) ;

vednnError_t vednnSoftmaxForward_Fast (
    const void 			*pDataIn,
    void 			*pDataOut,
    const uint64_t		nBatch,
    const uint64_t		nClass
) ;

vednnError_t vednnSoftmaxForward_Accurate (
    const void 			*pDataIn,
    void 			*pDataOut,
    const uint64_t		nBatch,
    const uint64_t		nClass
) ;

vednnError_t vednnSoftmaxForward_Log (
    const void 			*pDataIn,
    void 			*pDataOut,
    const uint64_t		nBatch,
    const uint64_t		nClass
) ;

#endif /* _VEDNNSOFTMAXFORWARD_H_ */
