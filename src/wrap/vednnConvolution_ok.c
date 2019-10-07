#include "vednnConvolution_ok.h"
#include "vednnConvolutionLists.h" // circular? YES!
#include <stdint.h>
#include <stddef.h> // NULL

#ifdef __cplusplus
extern "C" {
#endif


int (*pfun) (void) = NULL;

/** Planning-time Const-parameter checks */
#define FWD_FN_OK( BASENAME ) vednnError_t BASENAME##_ok(  VEDNN_PARAMS_CONV_FORWARD)
/** Run-time Pointer alignment checks */
#define FWD_RT_OK( BASENAME ) vednnError_t BASENAME##_rtok(VEDNN_DATARG_CONV_FORWARD)

#define FWD_LIKE(IMPL) (VEDNN_SUCCESS == \
        vednnConvolutionForward_direct_##IMPL##_ok( VEDNN_PARAMS_CONV_FORWARD_LIST))

#define FWD_FN_OK_LIKE( IMPL, PREV_IMPL, EXTRA_COND ) \
FWD_FN_OK(vednnConvolutionForward_direct_##IMPL) { \
    int ok = FWD_LIKE(PREV_IMPL) && (EXTRA_COND); \
    return ok? VEDNN_SUCCESS: VEDNN_ERROR_INVALID_PARAM; \
}
// better than macros would be aliasing identcal functions to a 
// single implementation, such as:
//static vednnError_t vednnFwd_pDataInOut_align8( VEDNN_DATARG_CONV_FORWARD )
//{
//    int ok = ( (((uint64_t)pDataIn)  & 0x7) == 0
//            && (((uint64_t)pDataOut) & 0x7) == 0);
//    return ok? VEDNN_SUCCESS: VEDNN_ERROR_INVALID_PARAM;
//}
// -----> CHK2_ALIGN8(pDataIn,pDataOut)
#define CHK1_ALIGN8(PTR) \
{ \
    int ok = ( (((uint64_t)PTR) & 0x07) == 0 ); \
    return ok? VEDNN_SUCCESS: VEDNN_ERROR_INVALID_PARAM; \
}
#define CHK2_ALIGN8(P1,P2) \
{ \
    int ok = ( (((uint64_t)P1) & 0x07) == 0 \
           &&  (((uint64_t)P2) & 0x07) == 0  ); \
    return ok? VEDNN_SUCCESS: VEDNN_ERROR_INVALID_PARAM; \
}
FWD_FN_OK(vednnConvolutionForward_direct_default)
{
    int ok = algo == VEDNN_CONV_ALGORITHM_DIRECT
        && pParamConv->group > 0
        && pParamIn->channel % pParamConv->group == 0
        && pParamOut->channel % pParamConv->group == 0
        //
        //&& pParamKernel->inChannel == pParamIn->channel / pParamConv->group
        //&& pParamKernel->outChannel == pParamOut->channel / pParamConv->group
        //
        && pParamConv->padWidth >= 0
        && pParamConv->padHeight >= 0
        && pParamConv->strideWidth > 0
        && pParamConv->strideHeight > 0
        && pParamConv->dilationWidth > 0
        && pParamConv->dilationHeight > 0
        && pParamKernel->width >= 1
        && pParamKernel->height >= 1
        ;
    return ok? VEDNN_SUCCESS: VEDNN_ERROR_INVALID_PARAM;
}

//FWD_FN_OK(vednnConvolutionForward_direct_default2)
//{
//    int ok = FWD_LIKE(default);
//    return ok? VEDNN_SUCCESS: VEDNN_ERROR_INVALID_PARAM;
//}
FWD_FN_OK_LIKE(default2, default, 1);
FWD_FN_OK_LIKE(default2p, default, 1);
FWD_FN_OK_LIKE(default3, default, 1);
FWD_FN_OK_LIKE(default3b, default, 1);
FWD_FN_OK_LIKE(alt, default, 1);
FWD_FN_OK_LIKE(defaultA, default, 1);
FWD_FN_OK_LIKE(owU128, default, pParamOut->width <= 128);
FWD_FN_OK_LIKE(owU128A, owU128, 1);

FWD_FN_OK_LIKE(vecC, default,
        pParamOut->height * pParamOut->width <= 16
        );

FWD_FN_OK_LIKE(dil1_pad0, default,
        pParamConv->dilationHeight == 1 && pParamConv->dilationWidth == 1
        && pParamConv->padHeight == 0  && pParamConv->padWidth == 0
        && pParamOut->height == (pParamIn->height - pParamKernel->height) / pParamConv->strideHeight + 1
        && pParamOut->width == (pParamIn->width - pParamKernel->width) / pParamConv->strideWidth + 1
        );
FWD_FN_OK_LIKE(dil1_pad0A, dil1_pad0, 1);

FWD_FN_OK_LIKE(dil1_str1_padsame, default,
        pParamConv->strideWidth==1
        && pParamConv->strideHeight==1
        && pParamConv->dilationWidth==1
        && pParamConv->dilationHeight==1
        && pParamKernel->width == 2*pParamConv->padWidth + 1
        && pParamKernel->height == 2*pParamConv->padHeight + 1 );
FWD_FN_OK_LIKE(dil1_str1_padsameA, dil1_str1_padsame, 1);
FWD_FN_OK_LIKE(dil1_str1_padsameB, dil1_str1_padsame, 1);
FWD_FN_OK_LIKE(dil1_str1_padsameAB, dil1_str1_padsame, 1);

FWD_FN_OK_LIKE(dil1_str1_pad0, dil1_pad0,
        pParamConv->strideWidth==1 && pParamConv->strideHeight==1);
FWD_FN_OK_LIKE(dil1_str1_pad0A, dil1_str1_pad0, 1);

FWD_FN_OK_LIKE(dil1_str1_pad0_ker1, dil1_str1_pad0,
        pParamKernel->width == 1 && pParamKernel->height == 1);
FWD_FN_OK_LIKE(dil1_str1_pad0_ker1A, dil1_str1_pad0_ker1, 1);

FWD_FN_OK_LIKE(dil1_pad0_ker1, dil1_pad0,
        pParamKernel->width == 1 && pParamKernel->height == 1);
FWD_FN_OK_LIKE(dil1_pad0_ker1A, dil1_pad0_ker1, 1);

FWD_FN_OK_LIKE(dil1_pad0_owU128_ker1, dil1_pad0_ker1, pParamOut->width <= 128);
FWD_FN_OK_LIKE(dil1_pad0_owU128_ker1A, dil1_pad0_owU128_ker1, 1);

FWD_FN_OK_LIKE(dil1_str1_padsame_ker3, dil1_str1_padsame,
        pParamKernel->width == 3 && pParamKernel->height == 3);
FWD_FN_OK_LIKE(dil1_str1_padsame_ker3A, dil1_str1_padsame_ker3, 1);

FWD_FN_OK_LIKE(dil1_str1_padsame_ker3_c1024x, dil1_str1_padsame_ker3,
        1 /* CAN validly run in other cases! ParamKernel->inChannel % 1024 == 0 */
        );
FWD_FN_OK_LIKE(dil1_str1_padsame_ker3_c1024xA, dil1_str1_padsame_ker3_c1024x, 1);

FWD_FN_OK_LIKE(dil1_str1_padsame_ker3_c1, dil1_str1_padsame_ker3,
        pParamIn->channel == pParamConv->group);
FWD_FN_OK_LIKE(dil1_str1_padsame_ker3_c1A, dil1_str1_padsame_ker3_c1, 1);

FWD_FN_OK_LIKE(dil1_str1_padsame_ker3_c1_owU128, dil1_str1_padsame_ker3_c1,
        pParamOut->width <= 128);
FWD_FN_OK_LIKE(dil1_str1_padsame_ker3_c1_owU128A, dil1_str1_padsame_ker3_c1_owU128, 1);

FWD_FN_OK_LIKE(dil1_str1_padsame_ker5, dil1_str1_padsame,
        pParamKernel->width == 5 && pParamKernel->height == 5);
FWD_FN_OK_LIKE(dil1_str1_padsame_ker5A, dil1_str1_padsame_ker5, 1);

FWD_FN_OK_LIKE(dil1_str1_padsame_ker5_owU128, dil1_str1_padsame_ker5,
        pParamOut->width <= 128);
FWD_FN_OK_LIKE(dil1_str1_padsame_ker5_owU128A, dil1_str1_padsame_ker5_owU128, 1);

FWD_FN_OK_LIKE(dil1_str1_padsame_ker2, dil1_str1_padsame,
        pParamKernel->width == 2 && pParamKernel->height == 2);
FWD_FN_OK_LIKE(dil1_str1_padsame_ker2A, dil1_str1_padsame_ker2, 1);

FWD_FN_OK_LIKE(dil1_str1_padsame_ker2_owU128, dil1_str1_padsame_ker2,
        pParamOut->width <= 128);
FWD_FN_OK_LIKE(dil1_str1_padsame_ker2_owU128A, dil1_str1_padsame_ker2_owU128, 1);

FWD_FN_OK_LIKE(dil1_str1_pad0_ker3_iw2XU256_ow2X_ioaligned, dil1_str1_pad0,
        pParamKernel->width == 3 && pParamKernel->height == 3
        && (pParamOut->width <= 256)
        && (pParamIn->width  & 0x1) == 0 /*&& (((uint64_t)pDataIn)  & 0x7) == 0*/
        && (pParamOut->width & 0x1) == 0 /*&& (((uint64_t)pDataOut) & 0x7) == 0*/ );
FWD_RT_OK(vednnConvolutionForward_direct_dil1_str1_pad0_ker3_iw2XU256_ow2X_ioaligned)
    CHK2_ALIGN8(pDataIn,pDataOut);

FWD_FN_OK_LIKE(dil1_str1_pad0_owU128, dil1_str1_pad0, pParamOut->width <= 128);
FWD_FN_OK_LIKE(dil1_str1_pad0_owU128A, dil1_str1_pad0_owU128, 1);

FWD_FN_OK_LIKE(dil1_pad0_owU128, dil1_pad0, pParamOut->width <= 128);
FWD_FN_OK_LIKE(dil1_pad0_owU128A, dil1_pad0_owU128, 1);
//---------------------
#undef FWD_FN_OK_LIKE
#undef FWD_LIKE
#undef FWD_FN_OK
#undef FWD_RT_OK

#if 0
#define FWDBIAS_FN_OK( BASENAME ) vednnError_t BASENAME##_ok(  VEDNN_PARAMS_CONV_FORWARDADDBIAS)
#define FWDBIAS_RT_OK( BASENAME ) vednnError_t BASENAME##_rtok(VEDNN_DATARG_CONV_FORWARDADDBIAS)
#define FWDBIAS_LIKE(BASENAME) (VEDNN_SUCCESS == BASENAME##_ok( \
            pParamIn, pParamKernel, pParamBias, pParamOut, pParamConv, algo))
// pParamBias is *never* checked within src/C/vednnConvolutionForwardAddBias.c
#define FWDBIAS_LIKE_NOBIAS(BASENAME) (VEDNN_SUCCESS == BASENAME##_ok( \
            pParamIn, pParamKernel, pParamOut, pParamConv, algo))
#define FWDBIAS_JUST_LIKE( FBIAS, FNOBIAS ) \
    FWDBIAS_FN_OK(FBIAS) \
{ \
    int ok = FWDBIAS_LIKE_NOBIAS(FNOBIAS) \
    /* test validity of pParamBias ? */ \
    ; \
    return ok? VEDNN_SUCCESS: VEDNN_ERROR_INVALID_PARAM; \
}

FWDBIAS_JUST_LIKE(vednnConvolutionForwardAddBias_direct_default,
        vednnConvolutionForward_direct_default)

FWDBIAS_JUST_LIKE(vednnConvolutionForwardAddBias_direct_default2,
        vednnConvolutionForward_direct_default)

FWDBIAS_JUST_LIKE(vednnConvolutionForwardAddBias_direct_dil1_str1_padsame,
        vednnConvolutionForward_direct_dil1_str1_padsame)

FWDBIAS_JUST_LIKE(vednnConvolutionForwardAddBias_direct_dil1_str1_padsame_ker3,
        vednnConvolutionForward_direct_dil1_str1_padsame_ker3)

FWDBIAS_JUST_LIKE(vednnConvolutionForwardAddBias_direct_dil1_str1_padsame_ker3_c1,
        vednnConvolutionForward_direct_dil1_str1_padsame_ker3_c1)

FWDBIAS_JUST_LIKE(vednnConvolutionForwardAddBias_direct_dil1_str1_padsame_ker3_c1024x,
        vednnConvolutionForward_direct_dil1_str1_padsame_ker3_c1024x)

FWDBIAS_JUST_LIKE(vednnConvolutionForwardAddBias_direct_dil1_str1_padsame_ker3_c1_owU128,
        vednnConvolutionForward_direct_dil1_str1_padsame_ker3_c1_owU128)

FWDBIAS_JUST_LIKE(vednnConvolutionForwardAddBias_direct_dil1_str1_pad0_ker1,
        vednnConvolutionForward_direct_dil1_str1_pad0_ker1)

    // note Forward...c1024x was bundled into a more general impl (todo here too?)
FWDBIAS_FN_OK(vednnConvolutionForwardAddBias_direct_dil1_str1_pad0_ker1_c1024x)
{
    //int ok = FWDBIAS_LIKE(vednnConvolutionForwardAddBias_direct_dil1_str1_pad0_ker1);
    int ok = FWDBIAS_LIKE_NOBIAS(vednnConvolutionForward_direct_dil1_str1_pad0_ker1);
    ok = ok && pParamKernel->inChannel % 1024 == 0
        ;
    return ok? VEDNN_SUCCESS: VEDNN_ERROR_INVALID_PARAM;
}
#undef FWDBIAS_JUST_LIKE
#undef FWDBIAS_LIKE_NOBIAS
#undef FWDBIAS_LIKE
#undef FWDBIAS_RT_OK
#undef FWDBIAS_FN_OK
#endif

#define BKWD_FN_OK( BASENAME ) vednnError_t BASENAME##_ok(   VEDNN_PARAMS_CONV_BACKWARD_DATA )
#define BKWD_RT_OK( BASENAME ) vednnError_t BASENAME##_rtok( VEDNN_DATARG_CONV_BACKWARD_DATA )
#define BKWD_LIKE0(BASENAME) (VEDNN_SUCCESS == BASENAME##_ok( \
            pParamGradIn, pParamKernel, pParamGradOut, pParamConv, algo))
#define BKWD_LIKE(IMPL) BKWD_LIKE0( vednnConvolutionBackwardData_direct_##IMPL)
#define BKWD_FN_OK_LIKE( IMPL, PREV_IMPL, EXTRA_COND ) \
    BKWD_FN_OK(vednnConvolutionBackwardData_direct_##IMPL) { \
        int ok = BKWD_LIKE(PREV_IMPL) && (EXTRA_COND); \
        return ok? VEDNN_SUCCESS: VEDNN_ERROR_INVALID_PARAM; \
    }

BKWD_FN_OK(vednnConvolutionBackwardData_direct_default)
{
    // generic, same as FWD_FN_OK(vednnConvolutionForward_direct_default)
    int ok = algo == VEDNN_CONV_ALGORITHM_DIRECT
        && pParamConv->group > 0
        && pParamGradIn->channel % pParamConv->group == 0
        && pParamGradOut->channel % pParamConv->group == 0
        //
        //&& pDataKernel->inChannel == pParamGradIn->channel / pParamConv->group
        //&& pDataKernel->outChannel == pParamGradOut->channel / pParamConv->group
        //
        && pParamConv->padWidth >= 0
        && pParamConv->padHeight >= 0
        && pParamConv->strideWidth > 0
        && pParamConv->strideHeight > 0
        && pParamConv->dilationWidth > 0
        && pParamConv->dilationHeight > 0
        ;
    return ok? VEDNN_SUCCESS: VEDNN_ERROR_INVALID_PARAM;
}
BKWD_FN_OK_LIKE(default2, default, 1);
BKWD_FN_OK_LIKE(iwU128, default, pParamGradIn->width <= 128);

BKWD_FN_OK_LIKE(dil1_str1, default,
        pParamConv->strideHeight == 1 && pParamConv->strideWidth == 1
        && pParamConv->dilationHeight == 1 && pParamConv->dilationWidth == 1);

BKWD_FN_OK_LIKE(dil1_str1_iwU128, dil1_str1, pParamGradIn->width <= 128);

BKWD_FN_OK_LIKE(dil1_str1_pad0_ker3_iwU128, dil1_str1_iwU128,
        pParamConv->padHeight == 0 && pParamConv->padWidth == 0
        /* TODO check output size correct */
        && pParamKernel->height == 3 && pParamKernel->width == 3);

BKWD_RT_OK(vednnConvolutionBackwardData_direct_dil1_str1_pad0_ker3_iw2XU256_ow2X_ioaligned)
    CHK2_ALIGN8(pDataGradIn,pDataGradOut);
BKWD_FN_OK_LIKE(dil1_str1_pad0_ker3_iw2XU256_ow2X_ioaligned, dil1_str1,
        pParamConv->padHeight == 0 && pParamConv->padWidth == 0
        /* TODO check output size correct */
        && pParamKernel->height == 3 && pParamKernel->width == 3
        && pParamGradIn->width <=256
        && (pParamGradIn->width & 0x01) == 0
        && (pParamGradOut->width & 0x01) == 0
        /*&& (((uint64_t)pDataGradIn) & 0x07) == 0*/
        /*&& (((uint64_t)pDataGradOut) & 0x07) == 0*/
        );

BKWD_RT_OK(vednnConvolutionBackwardData_direct_dil1_str1_pad0_ker3_iw2XU32_ow2X_ioaligned)
    CHK2_ALIGN8(pDataGradIn,pDataGradOut);
BKWD_FN_OK_LIKE(dil1_str1_pad0_ker3_iw2XU32_ow2X_ioaligned, dil1_str1_pad0_ker3_iw2XU256_ow2X_ioaligned,
        pParamGradIn->width <=256);

BKWD_FN_OK_LIKE(dil1_str1_padsame, dil1_str1,
        pParamKernel->width == 2*pParamConv->padWidth + 1
        && pParamKernel->height == 2*pParamConv->padHeight + 1);

BKWD_FN_OK_LIKE(dil1_str1_padsame_ker5, dil1_str1_padsame,
        pParamKernel->width == 5 && pParamKernel->height == 5);

#undef BKWD_LIKE
#undef BKWD_LIKE0
#undef BKWD_FN_OK_LIKE
#undef BKWD_FN_OK
#undef BKWD_RT_OK

#define BKWF_FN_OK( BASENAME ) vednnError_t BASENAME##_ok(  VEDNN_PARAMS_CONV_BACKWARD_FILTER)
#define BKWF_RT_OK( BASENAME ) vednnError_t BASENAME##_rtok(VEDNN_DATARG_CONV_BACKWARD_FILTER)
#define BKWF_LIKE_(BASENAME) (VEDNN_SUCCESS == BASENAME##_ok( \
            pParamIn, pParamGradOut, pParamGradKernel, pParamConv, algo))
#define BKWF_LIKE(IMPL) BKWF_LIKE_(vednnConvolutionBackwardFilter_direct_##IMPL)
#define BKWF_FN_OK_LIKE( IMPL, PREV_IMPL, EXTRA_COND ) \
    BKWF_FN_OK(vednnConvolutionBackwardFilter_direct_##IMPL) { \
        int ok = BKWF_LIKE(PREV_IMPL) && (EXTRA_COND); \
        return ok? VEDNN_SUCCESS: VEDNN_ERROR_INVALID_PARAM; \
    }
#define KER(N) pParamGradKernel->width == (N) && pParamGradKernel->height == (N)
#define OW(N) pParamGradOut->width <= (N)

// see src/C/vednnConvolutionBackwardFilter.c and mirror those cases.
BKWF_FN_OK(vednnConvolutionBackwardFilter_direct_default)
{
    int ok = algo == VEDNN_CONV_ALGORITHM_DIRECT
        && pParamConv->group > 0
        //&& pParamIn->channel % pParamConv->group == 0
        //&& pParamGradOut->channel % pParamConv->group == 0
        //
        //&& pDataKernel->inChannel == pParamGradIn->channel / pParamConv->group
        //&& pDataKernel->outChannel == pParamGradOut->channel / pParamConv->group
        //
        && pParamConv->padWidth >= 0
        && pParamConv->padHeight >= 0
        && pParamConv->strideWidth > 0
        && pParamConv->strideHeight > 0
        && pParamConv->dilationWidth > 0
        && pParamConv->dilationHeight > 0
        ;
    return ok? VEDNN_SUCCESS: VEDNN_ERROR_INVALID_PARAM;
}
BKWF_FN_OK_LIKE(dil1_str1_padsame, default,
        pParamConv->strideWidth==1 && pParamConv->strideHeight==1
        && pParamConv->dilationWidth==1 && pParamConv->dilationHeight==1
        && pParamIn->height == pParamGradOut->height
        && pParamIn->width == pParamGradOut->width);
BKWF_FN_OK_LIKE(dil1_str1_padsame_ker1, dil1_str1_padsame, KER(1));
BKWF_FN_OK_LIKE(dil1_str1_padsame_ker2, dil1_str1_padsame, KER(2));
BKWF_FN_OK_LIKE(dil1_str1_padsame_ker3, dil1_str1_padsame, KER(3));
BKWF_FN_OK_LIKE(dil1_str1_padsame_ker5, dil1_str1_padsame, KER(5));

BKWF_FN_OK_LIKE(dil1_str1_padsame_ker2_owU128, dil1_str1_padsame_ker2, OW(128));
BKWF_FN_OK_LIKE(dil1_str1_padsame_ker3_ohwU256, dil1_str1_padsame_ker3,
        pParamGradOut->width * pParamGradOut->height <= 256);
BKWF_FN_OK_LIKE(dil1_str1_padsame_ker3_owU128, dil1_str1_padsame_ker3, OW(128));
BKWF_FN_OK_LIKE(dil1_str1_padsame_ker5_owU128, dil1_str1_padsame_ker5, OW(128));

BKWF_FN_OK_LIKE(dil1_pad0, default,
        pParamConv->dilationHeight == 1 && pParamConv->dilationWidth == 1
        && pParamConv->padHeight == 0 && pParamConv->padWidth == 0
        && pParamGradOut->height == (pParamIn->height - pParamGradKernel->height) / pParamConv->strideHeight + 1
        && pParamGradOut->width == (pParamIn->width - pParamGradKernel->width) / pParamConv->strideWidth + 1);
BKWF_FN_OK_LIKE(dil1_pad0_owU32, dil1_pad0, OW(32));
BKWF_FN_OK_LIKE(dil1_pad0_owU128, dil1_pad0, OW(128));

BKWF_FN_OK_LIKE(dil1_str1_pad0_ker3_ow2X_iw2XU256_igoaligned, dil1_pad0,
        KER(3)
        && pParamConv->strideHeight == 1 && pParamConv->strideWidth == 1
        && pParamIn->width <= 256
        && (pParamIn->width & 0x01) == 0      /*RT: && (((uint64_t)pDataIn) & 0x07) == 0*/
        && (pParamGradOut->width & 0x01) == 0 /*RT: && (((uint64_t)pDataGradOut) & 0x07) == 0*/
        );
BKWF_RT_OK(vednnConvolutionBackwardFilter_direct_dil1_str1_pad0_ker3_ow2X_iw2XU256_igoaligned) {
    CHK2_ALIGN8(pDataIn,pDataGradOut);
}

BKWF_FN_OK_LIKE(dil1_pad0_ker1, dil1_pad0, KER(1));
BKWF_FN_OK_LIKE(dil1_pad0_ker1_owU32, dil1_pad0_ker1, OW(32));
BKWF_FN_OK_LIKE(dil1_pad0_ker1_ohwU64, dil1_pad0_ker1,
        pParamGradOut->height * pParamGradOut->width <= 64);
BKWF_FN_OK_LIKE(dil1_pad0_ker1_ohwU128, dil1_pad0_ker1,
        pParamGradOut->height * pParamGradOut->width <= 128);
BKWF_FN_OK_LIKE(dil1_pad0_ker3_owU128, dil1_pad0, KER(3) && OW(128));
BKWF_FN_OK_LIKE(owU128, default, OW(128));

#undef OW
#undef KER
#undef BKWF_FN_OK_LIKE
#undef BKWF_LIKE_
#undef BKWF_LIKE
#undef BKWF_FN_OK
#undef BKWF_RT_OK

#ifdef __cplusplus
}
#endif
// vim: et ts=4 sw=4 cindent cino=^=l0,\:0,N-s syntax=cpp.doxygen
