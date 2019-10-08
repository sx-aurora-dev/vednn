#ifndef SRC_VEDNNCONVOLUTIONBACKWARDFILTER_H_
#define SRC_VEDNNCONVOLUTIONBACKWARDFILTER_H_

#include "vednn.h"

#ifdef VEDNN_USE_OPENMP
#include <stdint.h>
#endif

<<<<<<< HEAD
typedef
vednnError_t (*vednnConvBackwardFilter_t) (
    const vednnTensorParam_t * restrict 	pParamIn,
    const void * restrict 			pDataIn,
    const vednnTensorParam_t * restrict 	pParamGradOut,
    const void * restrict 			pDataGradOut,
    const vednnConvolutionParam_t * restrict 	pParamConv,
    const vednnFilterParam_t * restrict 	pParamGradKernel,
    void * restrict 				pDataGradKernel
#ifdef VEDNN_USE_OPENMP
    ,
    const int64_t				beginOChannel,
    const int64_t				nOChannel
#endif
) ;

vednnError_t
vednnConvolutionBackwardFilter_direct_default(
    const vednnTensorParam_t * restrict 	pParamIn,
    const void * restrict 			pDataIn,
    const vednnTensorParam_t * restrict 	pParamGradOut,
    const void * restrict 			pDataGradOut,
    const vednnConvolutionParam_t * restrict 	pParamConv,
    const vednnFilterParam_t * restrict 	pParamGradKernel,
    void * restrict 				pDataGradKernel
#ifdef VEDNN_USE_OPENMP
    ,
    const int64_t				beginOChannel,
    const int64_t				nOChannel
#endif
) ;

vednnError_t
vednnConvolutionBackwardFilter_direct_vecC(
    const vednnTensorParam_t * restrict 	pParamIn,
    const void * restrict 			pDataIn,
    const vednnTensorParam_t * restrict 	pParamGradOut,
    const void * restrict 			pDataGradOut,
    const vednnConvolutionParam_t * restrict 	pParamConv,
    const vednnFilterParam_t * restrict 	pParamGradKernel,
    void * restrict 				pDataGradKernel
#ifdef VEDNN_USE_OPENMP
    ,
    const int64_t				beginOChannel,
    const int64_t				nOChannel
#endif
) ;


vednnError_t
vednnConvolutionBackwardFilter_direct_dil1_pad0(
    const vednnTensorParam_t * restrict 	pParamIn,
    const void * restrict 			pDataIn,
    const vednnTensorParam_t * restrict 	pParamGradOut,
    const void * restrict 			pDataGradOut,
    const vednnConvolutionParam_t * restrict 	pParamConv,
    const vednnFilterParam_t * restrict 	pParamGradKernel,
    void * restrict 				pDataGradKernel
#ifdef VEDNN_USE_OPENMP
    ,
    const int64_t				beginOChannel,
    const int64_t				nOChannel
#endif
) ;

vednnError_t
vednnConvolutionBackwardFilter_direct_dil1_pad0_ker1(
    const vednnTensorParam_t * restrict 	pParamIn,
    const void * restrict 			pDataIn,
    const vednnTensorParam_t * restrict 	pParamGradOut,
    const void * restrict 			pDataGradOut,
    const vednnConvolutionParam_t * restrict 	pParamConv,
    const vednnFilterParam_t * restrict 	pParamGradKernel,
    void * restrict 				pDataGradKernel
#ifdef VEDNN_USE_OPENMP
    ,
    const int64_t				beginOChannel,
    const int64_t				nOChannel
#endif
) ;

vednnError_t
vednnConvolutionBackwardFilter_direct_dil1_pad0_ker1_owU32(
    const vednnTensorParam_t * restrict 	pParamIn,
    const void * restrict 			pDataIn,
    const vednnTensorParam_t * restrict 	pParamGradOut,
    const void * restrict 			pDataGradOut,
    const vednnConvolutionParam_t * restrict 	pParamConv,
    const vednnFilterParam_t * restrict 	pParamGradKernel,
    void * restrict 				pDataGradKernel
#ifdef VEDNN_USE_OPENMP
    ,
    const int64_t				beginOChannel,
    const int64_t				nOChannel
#endif
) ;

vednnError_t
vednnConvolutionBackwardFilter_direct_dil1_pad0_ker1_ohwU64(
    const vednnTensorParam_t * restrict 	pParamIn,
    const void * restrict 			pDataIn,
    const vednnTensorParam_t * restrict 	pParamGradOut,
    const void * restrict 			pDataGradOut,
    const vednnConvolutionParam_t * restrict 	pParamConv,
    const vednnFilterParam_t * restrict 	pParamGradKernel,
    void * restrict 				pDataGradKernel
#ifdef VEDNN_USE_OPENMP
    ,
    const int64_t				beginOChannel,
    const int64_t				nOChannel
#endif
) ;

vednnError_t
vednnConvolutionBackwardFilter_direct_dil1_pad0_ker1_ohwU128(
    const vednnTensorParam_t * restrict 	pParamIn,
    const void * restrict 			pDataIn,
    const vednnTensorParam_t * restrict 	pParamGradOut,
    const void * restrict 			pDataGradOut,
    const vednnConvolutionParam_t * restrict 	pParamConv,
    const vednnFilterParam_t * restrict 	pParamGradKernel,
    void * restrict 				pDataGradKernel
#ifdef VEDNN_USE_OPENMP
    ,
    const int64_t				beginOChannel,
    const int64_t				nOChannel
#endif
) ;

vednnError_t
vednnConvolutionBackwardFilter_direct_dil1_pad0_owU32(
    const vednnTensorParam_t * restrict 	pParamIn,
    const void * restrict 			pDataIn,
    const vednnTensorParam_t * restrict 	pParamGradOut,
    const void * restrict 			pDataGradOut,
    const vednnConvolutionParam_t * restrict 	pParamConv,
    const vednnFilterParam_t * restrict 	pParamGradKernel,
    void * restrict 				pDataGradKernel
#ifdef VEDNN_USE_OPENMP
    ,
    const int64_t				beginOChannel,
    const int64_t				nOChannel
#endif
) ;

vednnError_t
vednnConvolutionBackwardFilter_direct_ker3_owU128(
    const vednnTensorParam_t * restrict 	pParamIn,
    const void * restrict 			pDataIn,
    const vednnTensorParam_t * restrict 	pParamGradOut,
    const void * restrict 			pDataGradOut,
    const vednnConvolutionParam_t * restrict 	pParamConv,
    const vednnFilterParam_t * restrict 	pParamGradKernel,
    void * restrict 				pDataGradKernel
#ifdef VEDNN_USE_OPENMP
    ,
    const int64_t				beginOChannel,
    const int64_t				nOChannel
#endif
) ;

vednnError_t
vednnConvolutionBackwardFilter_direct_owU128(
    const vednnTensorParam_t * restrict 	pParamIn,
    const void * restrict 			pDataIn,
    const vednnTensorParam_t * restrict 	pParamGradOut,
    const void * restrict 			pDataGradOut,
    const vednnConvolutionParam_t * restrict 	pParamConv,
    const vednnFilterParam_t * restrict 	pParamGradKernel,
    void * restrict 				pDataGradKernel
#ifdef VEDNN_USE_OPENMP
    ,
    const int64_t				beginOChannel,
    const int64_t				nOChannel
#endif
) ;

vednnError_t
vednnConvolutionBackwardFilter_direct_dil1_pad0_ker3_owU128(
    const vednnTensorParam_t * restrict 	pParamIn,
    const void * restrict 			pDataIn,
    const vednnTensorParam_t * restrict 	pParamGradOut,
    const void * restrict 			pDataGradOut,
    const vednnConvolutionParam_t * restrict 	pParamConv,
    const vednnFilterParam_t * restrict 	pParamGradKernel,
    void * restrict 				pDataGradKernel
#ifdef VEDNN_USE_OPENMP
    ,
    const int64_t				beginOChannel,
    const int64_t				nOChannel
#endif
) ;
=======
#ifdef __cplusplus
extern "C" { //}
#endif

// basic arg list
#define VEDNN_CONVBKF_ARGS \
const vednnTensorParam_t * restrict      pParamIn, \
const void * restrict                    pDataIn, \
const vednnTensorParam_t * restrict      pParamGradOut, \
const void * restrict                    pDataGradOut, \
const vednnConvolutionParam_t * restrict pParamConv, \
const vednnFilterParam_t * restrict      pParamGradKernel, \
void * restrict                          pDataGradKernel
#define VEDNN_CONVBKF_ARGS_LIST pParamIn, pDataIn, pParamGradOut, pDataGradOut, \
    pParamConv, pParamGradKernel, pDataGradKernel /*, beginOChannel, nOChannel*/

// omp additonal args...
#ifndef VEDNN_USE_OPENMP
#define VEDNN_CONVBKF_OMPARGS      VEDNN_CONVBKF_ARGS
#define VEDNN_CONVBKF_OMPARGS_LIST VEDNN_CONVBKF_ARGS
#else
#define VEDNN_CONVBKF_OMPARGS VEDNN_CONVBKF_ARGS, \
    const int64_t beginOChannel, /* openmp only */ \
const int64_t nOChannel      /* openmp only */
#define VEDNN_CONVBKF_OMPARGS_LIST pParamIn, pDataIn, pParamGradOut, pDataGradOut, \
    pParamConv, pParamGradKernel, pDataGradKernel, beginOChannel, nOChannel
#endif // VEDNN_USE_OPENMP
>>>>>>> 5547392796048fc953c746fa87cae7cce77508e3

typedef
vednnError_t (*vednnConvBackwardFilter_t) ( VEDNN_CONVBKF_OMPARGS );

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
VEDNN_CONVBKF_DECL(dil1_str1_padsame);
VEDNN_CONVBKF_DECL(dil1_str1_padsame_ker1);
VEDNN_CONVBKF_DECL(dil1_str1_padsame_ker3);
VEDNN_CONVBKF_DECL(dil1_str1_padsame_ker3_ohwU256);
VEDNN_CONVBKF_DECL(dil1_str1_padsame_ker3_owU128);
VEDNN_CONVBKF_DECL(dil1_str1_padsame_ker5);
VEDNN_CONVBKF_DECL(dil1_str1_padsame_ker5_owU128);
VEDNN_CONVBKF_DECL(dil1_str1_padsame_ker2);
VEDNN_CONVBKF_DECL(dil1_str1_padsame_ker2_owU128);
// extra
//VEDNN_CONVBKF_DECL(default2);
//VEDNN_CONVBKF_DECL(dil1_pad0_owU128);
//VEDNN_CONVBKF_DECL(dil1_pad0_ker3_owU32);
#ifdef __cplusplus
}//extern "C"
#endif
// vim: et ts=4 sw=4 cindent cino=^=l0,\:0,N-s
#endif /* SRC_VEDNNCONVOLUTIONBACKWARDFILTER_H_ */
