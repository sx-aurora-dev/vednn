
#ifndef _VEDNNACTIVATIONBACWARD_H_
#define _VEDNNACTIVATIONBACWARD_H_

#include "vednn.h"

typedef
vednnError_t (*vednnActivationBackward_t) (
    const void 				*pDataGradOut,
    const void 				*pDataIn,
    void 				*pDataGradIn,
    const uint64_t			nElements
) ;

vednnError_t vednnActivationBackward_Relu(
    const void 				*pDataGradOut,
    const void 				*pDataIn,
    void 				*pDataGradIn,
    const uint64_t			nElements
) ;

#endif /* _VEDNNACTIVATIONBACWARD_H_ */
