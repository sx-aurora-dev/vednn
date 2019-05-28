#ifndef _VEDNNACTIVATIONFORWARD_H_
#define _VEDNNACTIVATIONFORWARD_H_

#include "vednn.h"

typedef
vednnError_t (*vednnActivationForward_t) (
    const void 				*pDataIn,
    void 				*pDataOut,
    const uint64_t			nElements
) ;

vednnError_t vednnActivationForward_Relu(
    const void 				*pDataIn,
    void 				*pDataOut,
    const uint64_t			nElements
) ;

#endif /* _VEDNNACTIVATIONFORWARD_H_ */

