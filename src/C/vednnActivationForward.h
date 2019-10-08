#ifndef _VEDNNACTIVATIONFORWARD_H_
#define _VEDNNACTIVATIONFORWARD_H_

#include "vednn.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define VEDNN_ACTIVATIONFWD_ARGS \
  const void     *pDataIn, \
  void           *pDataOut, \
  const uint64_t nElements
#define VEDNN_ACTIVATIONFWD_ARGS_LIST pDataIn, pDataOut, nElements

typedef
vednnError_t (*vednnActivationForward_t) ( VEDNN_ACTIVATIONFWD_ARGS );

vednnError_t vednnActivationForward_Relu( VEDNN_ACTIVATIONFWD_ARGS );

#ifdef __cplusplus
}//extern "C"
#endif
// vim: sw=2 ts=2 et ai
#endif /* _VEDNNACTIVATIONFORWARD_H_ */

