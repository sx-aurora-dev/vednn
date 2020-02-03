#ifndef _VEDNNACTIVATIONBACWARD_H_
#define _VEDNNACTIVATIONBACWARD_H_

#include "vednn.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" { //}
#endif

#define VEDNN_ACTIVATIONBKW_ARGS \
        const void      *pDataGradOut, \
        const void      *pDataIn, \
        void            *pDataGradIn, \
        const uint64_t  nElements
#define VEDNN_ACTIVATIONBKW_ARGS_LIST pDataGradOut, pDataIn, pDataGradIn, nElements


typedef
vednnError_t (*vednnActivationBackward_t) ( VEDNN_ACTIVATIONBKW_ARGS );

vednnError_t vednnActivationBackward_Relu( VEDNN_ACTIVATIONBKW_ARGS );

#ifdef __cplusplus
}//extern "C"
#endif
// vim: set et sw=4 ts=4 ai
#endif /* _VEDNNACTIVATIONBACWARD_H_ */
