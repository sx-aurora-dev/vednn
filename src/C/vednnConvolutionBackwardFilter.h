#ifndef SRC_VEDNNCONVOLUTIONBACKWARDFILTER_H_
#define SRC_VEDNNCONVOLUTIONBACKWARDFILTER_H_

#include "vednn.h"

#ifdef VEDNN_USE_OPENMP
#include <stdint.h>
#endif

#ifdef __cplusplus
extern "C" { //}
#endif

// basic arg list
#define VEDNN_CONVBKF_ARGS \
const vednnTensorParam_t *       pParamIn, \
const void *                     pDataIn, \
const vednnTensorParam_t *       pParamGradOut, \
const void *                     pDataGradOut, \
const vednnConvolutionParam_t *  pParamConv, \
const vednnFilterParam_t *       pParamGradKernel, \
void *                           pDataGradKernel
#define VEDNN_CONVBKF_ARGS_LIST pParamIn, pDataIn, pParamGradOut, pDataGradOut, \
    pParamConv, pParamGradKernel, pDataGradKernel /*, beginOChannel, nOChannel*/

// omp additonal args...
#ifndef VEDNN_USE_OPENMP
#define VEDNN_CONVBKF_OMPARGS      VEDNN_CONVBKF_ARGS
#define VEDNN_CONVBKF_OMPARGS_LIST VEDNN_CONVBKF_ARGS
#else // VEDNN_USE_OPENMP
#ifndef VEDNN_OMP_GROUP_PARALLEL
#define VEDNN_CONVBKF_OMPARGS VEDNN_CONVBKF_ARGS, \
    const int64_t beginOChannel, /* openmp only */ \
const int64_t nOChannel      /* openmp only */
#define VEDNN_CONVBKF_OMPARGS_LIST pParamIn, pDataIn, pParamGradOut, pDataGradOut, \
    pParamConv, pParamGradKernel, pDataGradKernel, beginOChannel, nOChannel
#else // VEDNN_OMP_GROUP_PARALLEL
#define VEDNN_CONVBKF_OMPARGS VEDNN_CONVBKF_ARGS, \
    const int64_t beginOChannel, /* openmp only */ \
    const int64_t nOChannel,     /* openmp only */ \
    const int64_t beginGroup,    /* openmp-group-parallel only */ \
    const int64_t nGroup         /* openmp-group-parallel only */
#define VEDNN_CONVBKF_OMPARGS_LIST pParamIn, pDataIn, pParamGradOut, pDataGradOut, \
    pParamConv, pParamGradKernel, pDataGradKernel, beginOChannel, nOChannel, beginGroup, nGroup
#endif // VEDNN_OMP_GROUP_PARALLEL
#endif // VEDNN_USE_OPENMP

typedef
vednnError_t (*vednnConvBackwardFilter_t) ( VEDNN_CONVBKF_OMPARGS );
typedef
vednnError_t (*vednnConvBackwardFilter_nowrap_t) ( VEDNN_CONVBKF_ARGS );

/** Note that gemm convolutions <B>never</B> use the default \e wrapper function
 * signature, so do not get extra thread-related channel arguments. */
vednnError_t vednnConvolutionBackwardFilter_direct_gemm( VEDNN_CONVBKF_ARGS );

/** other CnvBkF impls add in OMP threading args */
#define VEDNN_CONVBKF_DECL(IMPL) vednnError_t \
    vednnConvolutionBackwardFilter_direct_##IMPL( VEDNN_CONVBKF_OMPARGS );

VEDNN_CONVBKF_DECL(default);
VEDNN_CONVBKF_DECL(vecC);
VEDNN_CONVBKF_DECL(dil1_pad0);
VEDNN_CONVBKF_DECL(dil1_pad0_ker1);
VEDNN_CONVBKF_DECL(dil1_pad0_ker1_owU32);
VEDNN_CONVBKF_DECL(dil1_pad0_ker1_ohwU64);
VEDNN_CONVBKF_DECL(dil1_pad0_ker1_ohwU128);
VEDNN_CONVBKF_DECL(dil1_pad0_owU32);
VEDNN_CONVBKF_DECL(owU128);
VEDNN_CONVBKF_DECL(dil1_pad0_ker3_owU128);
VEDNN_CONVBKF_DECL(dil1_str1_pad0_ker3_owU128);
VEDNN_CONVBKF_DECL(dil1_str1_pad0_ker3_ow2X_iw2XU256_igoaligned);
VEDNN_CONVBKF_DECL(dil1_str1_pad0_ker4_owU128);
VEDNN_CONVBKF_DECL(dil1_str1_padsame);
VEDNN_CONVBKF_DECL(dil1_str1_padsame_ker1);
VEDNN_CONVBKF_DECL(dil1_str1_padsame_ker3);
VEDNN_CONVBKF_DECL(dil1_str1_padsame_ker3_ohwU256);
VEDNN_CONVBKF_DECL(dil1_str1_padsame_ker3_owU128);
VEDNN_CONVBKF_DECL(dil1_str1_padsame_ker5);
VEDNN_CONVBKF_DECL(dil1_str1_padsame_ker5_owU128);
VEDNN_CONVBKF_DECL(dil1_str1_padsame_ker2);
VEDNN_CONVBKF_DECL(dil1_str1_padsame_ker2_owU128);
VEDNN_CONVBKF_DECL(dil1_str2_pad1_ker3_owU128);
VEDNN_CONVBKF_DECL(ker3_owU128) ;
VEDNN_CONVBKF_DECL(dil1_str2_pad1_ker4_owU128);

// extra
//VEDNN_CONVBKF_DECL(default2);
#ifdef __cplusplus
}//extern "C"
#endif
// vim: et ts=4 sw=4 cindent cino=^=l0,\:0,N-s
#endif /* SRC_VEDNNCONVOLUTIONBACKWARDFILTER_H_ */
