
#include "vednn.h"
#include "vednn2gendnn.h"

#include "gemm_convolution.hpp"
#include <assert.h>

using namespace mkldnn::impl;
using namespace mkldnn::impl::cpu;

extern "C" { //}

vednnError_t
vednnConvolutionForward_direct_gendnn(
        VEDNN_CONVFWD_ARGS)
{
    //const vednnBiasParam_t * pParamBias = nullptr;
    //const void* pDataBias = nullptr;

    convolution_desc_t cd;
    mk_mkldnn_convolution_desc(
            pParamIn, pParamKernel, pParamBias, pParamOut,
            pParamConv, &cd);

    mkldnn_memory_desc_t src, weights, bias, dst;
    mk_mkldnn_nchw_memory_desc( pParamIn,     &src );
    mk_mkldnn_oihw_memory_desc( pParamKernel, &weights);
    mk_mkldnn_x_memory_desc( pParamBias,      &bias);
    mk_mkldnn_nchw_memory_desc( pParamOut,    &dst);
    int const max_threads = omp_get_max_threads();

    jit_gemm_conv_conf_t jcp;
    jit_gemm_convolution_utils::init_conf(
            jcp, cd,
            memory_desc_wrapper(src),
            memory_desc_wrapper(weights),
            memory_desc_wrapper(dst),
            max_threads /*, with_relu=false, relu_negative_slope=-1.0*/
            );

    scratchpad_t *scratchpad = nullptr;
    jit_gemm_convolution_utils::prepare_scratchpad( jcp, &scratchpad,
            jcp.im2col_sz * sizeof(float/*data_t*/),
            jcp.nthr );

    vconv_gemm_fwd( jcp,
            (data_t*)pDataIn, (data_t*)pDataKernel,
            (data_t*)pDataBias, (data_t*)pDataOut,
            *scratchpad /*, post_ops_=nullptr*/ );

    delete scratchpad;
}
#if 0
vednnError_t vednnConvolutionForwardAddBias_direct_gendnn(
    const vednnTensorParam_t         *pParamIn,
    const void                 *pDataIn,
    const vednnFilterParam_t        *pParamKernel,
    const void                 *pDataKernel,
    const vednnBiasParam_t         *pParamBias,
    const void                 *pDataBias,
    const vednnTensorParam_t         *pParamOut,
    void                 *pDataOut,
    const vednnConvolutionParam_t    *pParamConv,
    vednnConvolutionAlgorithm_t     algo
)
{
}
#endif

vednnError_t vednnConvolutionBackwardData_direct_gendnn(
    const vednnTensorParam_t         *pParamGradIn,
    const void                 *pDataGradIn,
    const vednnFilterParam_t        *pParamKernel,
    const void                 *pDataKernel,
    const vednnTensorParam_t         *pParamGradOut,
    void                 *pDataGradOut,
    const vednnConvolutionParam_t    *pParamConv,
    vednnConvolutionAlgorithm_t     algo
)
{
    assert(NULL=="TBD");
}

vednnError_t vednnConvolutionBackwardFilter_direct_gendnn(
    const vednnTensorParam_t         *pParamIn,
    const void                 *pDataIn,
    const vednnTensorParam_t         *pParamGradOut,
    const void                 *pDataGradOut,
    const vednnFilterParam_t        *pParamGradKernel,
    void                 *pDataGradKernel,
    const vednnConvolutionParam_t    *pParamConv,
    vednnConvolutionAlgorithm_t     algo
)
{
    assert(NULL=="TBD");
}
}//extern "C"
/* vim: set et ts=4 sw=4 cino=^=l0,\:0,N-s: */
