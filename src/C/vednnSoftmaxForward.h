#ifndef _VEDNNSOFTMAXFORWARD_H_
#define _VEDNNSOFTMAXFORWARD_H_

#include "vednn.h"
#include <stdint.h>

#ifdef __cplusplus
}//extern "C"
#endif

#define VEDNN_SOFTMAXFWD_ARGS \
    const void     *pDataIn, \
    void           *pDataOut, \
    const uint64_t  nBatch, \
    const uint64_t  nClass
#define VEDNN_SOFTMAXFWD_ARGS_LIST pDataIn, pDataOut, nBatch, nClass

typedef
vednnError_t (*vednnSoftmaxForward_t) ( VEDNN_SOFTMAXFWD_ARGS );

#define VEDNN_SOFTMAXFWD_DECL(IMPL) vednnError_t \
    vednnSoftmaxForward_##IMPL( VEDNN_SOFTMAXFWD_ARGS )

#if 0 // these symbols are not exposed, static in .c file
VEDNN_SOFTMAXFWD_DECL(Fast);
VEDNN_SOFTMAXFWD_DECL(Accurate);
VEDNN_SOFTMAXFWD_DECL(Log);
#endif

#ifdef __cplusplus
}//extern "C"
#endif
// vim: sw=4 ts=4 et
#endif /* _VEDNNSOFTMAXFORWARD_H_ */
