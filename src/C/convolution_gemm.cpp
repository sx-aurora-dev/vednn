/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */

#include "vednn-def.hpp" // vednn.h + C++ scratchpad API (C++ can inline one call)
//#include "vednn_helper.h"
#include "convolution_gemm.h"
#include "vednnConvolutionForward.h"
#include "vednnConvolutionBackwardData.h"
#include "vednnConvolutionBackwardFilter.h"

#include <stdio.h>
#include <assert.h>
#include <string.h> // memset
#include <stdint.h>

/** 0=malloc+free, 1=vednn_scratchpad_shared */
#ifndef SCRATCHPAD
#define SCRATCHPAD 0
#endif

/** cmake Debug build will set VERBOSE=1 */
#ifndef VERBOSE
#define VERBOSE 0
#endif
#if VERBOSE
#define DBG(...) do{ printf(__VA_ARGS__); fflush(stdout); }while(0)
#else
#define DBG(...) do{}while(0)
#endif

// borrow from test/conv_test_param.h
#ifndef HAVE_XMALLOC
#define HAVE_XMALLOC
#define XMALLOC(BYTES) \
    xmalloc(BYTES,__FILE__,__LINE__);
static inline void * xmalloc(size_t const bytes, char const* file, size_t const line){
    typedef long unsigned lu;
    assert( file != nullptr );
    if(VERBOSE) printf("%s:%lu xmalloc %lu bytes\n", file, (lu)line, (lu)bytes);
    void *ret = malloc(bytes);
    if(!ret) {printf("Memory exhausted: %s:%lu", file, (lu)line); exit(1);}
    return ret;
}
#endif

/** return sizeof(float) if pParam is DTYPE_FLOAT */
static inline size_t getTensorDataSize(const vednnTensorParam_t * restrict pParam)
{
    size_t dataSize = 0;
    assert( pParam!=nullptr );
    switch (pParam->dtype) {
    case DTYPE_FLOAT: dataSize = sizeof(float);
                      break;
    }
    assert(dataSize>0); /* BUG */
    return dataSize;
}


#define LOCAL_FTRACE 1
#if LOCAL_FTRACE
#define LFTRACE_BEGIN(...) FTRACE_BEGIN(__VA_ARGS__)
#define LFTRACE_END(...) FTRACE_END(__VA_ARGS__)
#define LFTRACE_IF(...) FTRACE_IF(__VA_ARGS__)
#else
#define LFTRACE_BEGIN(...) do{}while(0)
#define LFTRACE_END(...) do{}while(0)
#define LFTRACE_IF(...) do{}while(0)
#endif // LOCAL_FTRACE

#define SGEMM   sgemm_

extern "C" {
  void sgemm_(char *TRANSA, char *TRANSB, int *M, int *N, int *K,
          float *ALPHA, float *A,  int *LDA, float *B, int *LDB,
          float *BETA, float *C, int *LDC ) ;
}

static char  TRANS   = 'T';
static char  NOTRANS = 'N';
static float FONE    = 1.0f;
static float FZERO   = 0.0f;
static int   IONE    = 1;

/* ----------------------------------------------------------------------- */
static inline int is_a_ge_zero_and_a_lt_b(int a, int b) {
    //return (unsigned)a < (unsigned)b;
    return a>=0  && a<b; // for ncc auto vectorization, this is better
}
static inline int64_t is_a_ge_zero_and_a_lt_b(int64_t a, int64_t b) {
    return a>=0  && a<b; // for ncc auto vectorization, this is better
}

/** data_col size to hold float[ic*kw*kh*ow*oh]. */
    static void
im2col_cpu(const float * restrict data_im, const int64_t channels,
        const int64_t height, const int64_t width, const int64_t kernel_h, const int64_t kernel_w,
        const int64_t pad_h, const int64_t pad_w,
        const int64_t stride_h, const int64_t stride_w,
        const int64_t dilation_h, const int64_t dilation_w,
        float * restrict data_col)
{
    DBG("im2col_cpu...");
    LFTRACE_BEGIN("im2col_cpu");
    const int64_t output_h = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
    const int64_t output_w = (width + 2 * pad_w -  (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
    const int64_t channel_size = height * width;

    int64_t channel;

    // nc++ does not support 'if' clause : omp if(channel>=3)
    OMP(parallel for if(channels>=3))//;
    for (channel = 0 ; channel < channels; channel++) {             // inChannel
        int64_t kernel_row, kernel_col, output_rows, output_cols, output_col;

        int64_t inOffset = channel * channel_size;
        int64_t outOffset = channel * output_h * output_w * kernel_h * kernel_w;

        for (kernel_row = 0; kernel_row < kernel_h; kernel_row++) {     // kernHeight
            for (kernel_col = 0; kernel_col < kernel_w; kernel_col++) {     // kernWidth
                int64_t input_row = -pad_h + kernel_row * dilation_h;
                for (output_rows = output_h; output_rows; output_rows--) {  // outHeight
                    if (!is_a_ge_zero_and_a_lt_b(input_row, height)) {
                        for (output_cols = output_w; output_cols; output_cols--) { // outWidth
                            data_col[outOffset++] = 0; //*(data_col++) = 0;
                        }
                    } else {
                        int64_t input_col = -pad_w + kernel_col * dilation_w;
                        for (output_col = output_w; output_col; output_col--) { // outWidth
                            data_col[outOffset++] //*(data_col++)
                                = (is_a_ge_zero_and_a_lt_b(input_col, width)
                                        ? data_im[inOffset + input_row * width + input_col]
                                        : 0.f);
                            input_col += stride_w;
                        }
                    }
                    input_row += stride_h;
                }
            }
        }
    }
    LFTRACE_END("im2col_cpu");
    DBG("DONE\n");
}

    static void
col2im_cpu(
        const float* data_col, const int channels,
        const int height, const int width, const int kernel_h, const int kernel_w,
        const int pad_h, const int pad_w,
        const int stride_h, const int stride_w,
        const int dilation_h, const int dilation_w,
        float* data_im)
{
    DBG("col2im_cpu...");
    LFTRACE_BEGIN("col2im_cpu");
    memset(data_im, 0, sizeof(float)*height*width*channels) ;

    const int output_h = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
    const int output_w = (width + 2 * pad_w -  (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
    const int channel_size = height * width;

    int channel;

    OMP(parallel for if(channels>3))//;
    for (channel = 0 ; channel < channels; channel++) {             // inChannel
        int kernel_row, kernel_col, output_rows, output_cols, output_col;

        int inOffset = channel * channel_size;
        int outOffset = channel * output_h * output_w * kernel_h * kernel_w;

        for (kernel_row = 0; kernel_row < kernel_h; kernel_row++) {     // kernHeight
            for (kernel_col = 0; kernel_col < kernel_w; kernel_col++) {     // kernWidth
                int input_row = -pad_h + kernel_row * dilation_h;
                for (output_rows = output_h; output_rows; output_rows--) {  // outHeight
                    if (!is_a_ge_zero_and_a_lt_b(input_row, height)) {
                        for (output_cols = output_w; output_cols; output_cols--) {
                            // *(data_col++) = 0;
                            //data_col[outOffset++] ;
                            ++outOffset;
                        }
                    } else {
                        int input_col = -pad_w + kernel_col * dilation_w;
                        for (output_col = output_w; output_col; output_col--) { // outWidth
                            if (is_a_ge_zero_and_a_lt_b(input_col, width)) {
                                // *(data_col++) = data_im[input_row * width + input_col];
                                data_im[inOffset + input_row * width + input_col] += data_col[outOffset++] ;
                            } else {
                                // *(data_col++) = 0;
                                //data_col[outOffset++] ;
                                ++outOffset;
                            }
                            input_col += stride_w;
                        }
                    }
                    input_row += stride_h;
                }
            }
        }
    }
    LFTRACE_END("col2im_cpu");
    DBG("DONE\n");
}

extern "C" { //}
    static vednnError_t
convolution_forward_gemm(
        const vednnTensorParam_t * restrict pParamIn, const void * restrict pDataIn,
        const vednnFilterParam_t * restrict pParamKernel, const void * restrict pDataKernel,
        const vednnBiasParam_t * restrict pParamBias, const void * restrict pDataBias,
        const vednnTensorParam_t * restrict pParamOut, void * restrict pDataOut,
        const float * restrict pOne,  float * restrict pColBuff,
        const vednnConvolutionParam_t * restrict pParamConv )
{
    DBG(" vednn-ConvFwd-gemm");
    LFTRACE_BEGIN("vednn-ConvFwd-gemm");
    int n, g;

    int batch       = pParamIn->batch;
    int inChannel   = pParamIn->channel;
    int inWidth     = pParamIn->width;
    int inHeight    = pParamIn->height;
    int outChannel  = pParamOut->channel;
    int outWidth    = pParamOut->width;
    int outHeight   = pParamOut->height;
    int kernWidth   = pParamKernel->width;
    int kernHeight  = pParamKernel->height;

    int group       = pParamConv->group;
    int strideWidth = pParamConv->strideWidth;;
    int strideHeight    = pParamConv->strideHeight;
    int padWidth    = pParamConv->padWidth;
    int padHeight   = pParamConv->padHeight;
    int dilationWidth   = pParamConv->dilationWidth;
    int dilationHeight  = pParamConv->dilationHeight;

    int inChannelGroup  = inChannel  / group;   // pParamKernel->inChannel と同じ
    int outChannelGroup = outChannel / group;   // pParamKernel->outChannel と同じ

    const float * restrict pIn     = (const float*)pDataIn;
    const float * restrict pBias   = (const float*)pDataBias;
    const float * restrict pKernel = (const float*)pDataKernel;
    float * restrict pOut    = (float *)pDataOut;

    int no_im2col = (kernWidth == 1 && kernHeight == 1 && strideWidth == 1 && strideHeight == 1 && padWidth == 0 && padHeight == 0);

    for (n = 0; n < batch; n++) { // this->num_
        int inBatchOffset  = n * inChannel  * inWidth  * inHeight;
        int outBatchOffset = n * outChannel * outWidth * outHeight;

        for (g = 0; g < group; g++) {
            int inGroupOffset   = g * inChannelGroup                   * inHeight   * inWidth;
            int outGroupOffset  = g * outChannelGroup                  * outHeight  * outWidth;
            int kernGroupOffset = g * outChannelGroup * inChannelGroup * kernHeight * kernWidth;
            int biasGroupOffset = g * outChannelGroup;

            int inOffset  = inBatchOffset  + inGroupOffset;
            int outOffset = outBatchOffset + outGroupOffset;

            if (no_im2col) {
                DBG(" no_im2col GEMM");
                int M = outChannelGroup;
                int N = outWidth * outHeight;
                int K = inChannelGroup;
                int LDB = inWidth * inHeight;

                SGEMM(&NOTRANS, &NOTRANS, &N, &M, &K,
                        &FONE, (float *) &pIn[inOffset], &LDB,
                        (float *) &pKernel[kernGroupOffset], &K,
                        &FZERO, &pOut[outOffset], &N);

                if (pBias) {
                    SGEMM(&NOTRANS, &NOTRANS, &N, &M, &IONE,
                            &FONE, (float *) pOne, &N,
                            (float *) &pBias[biasGroupOffset], &IONE,
                            &FONE, &pOut[outOffset], &N);
                }

            } else {

                int M = outChannelGroup;
                int N = outWidth * outHeight;
                int K = inChannelGroup * kernWidth * kernHeight;

                im2col_cpu(&pIn[inOffset],
                        inChannelGroup, inHeight, inWidth, kernHeight, kernWidth,
                        padHeight, padWidth, strideHeight, strideWidth, dilationHeight, dilationWidth,
                        pColBuff);

                DBG(" GEMM");
                SGEMM(&NOTRANS, &NOTRANS, &N, &M, &K,
                        &FONE, pColBuff, &N,
                        (float *)&pKernel[kernGroupOffset], &K,
                        &FZERO, &pOut[outOffset], &N);

                if (pBias) {
                    DBG(" bias...");
                    SGEMM(&NOTRANS, &NOTRANS, &N, &M, &IONE,
                            &FONE, (float *)pOne, &N,
                            (float *) &pBias[biasGroupOffset], &IONE,
                            &FONE, &pOut[outOffset], &N);
                }
            }
        } // group
    } // batch

    LFTRACE_END("vednn-ConvFwd-gemm");
    DBG(" vednn-ConvFwd-gemm DONE");
    return VEDNN_SUCCESS;
}

    static vednnError_t
convolution_backward_data_gemm(
        const vednnTensorParam_t * restrict pParamGradOut, const void * restrict pDataGradOut,
        const vednnFilterParam_t * restrict pParamKernel, const void * restrict pDataKernel,
        const vednnTensorParam_t * restrict pParamGradIn, void * restrict pDataGradIn,
        float * restrict pColBuff,
        const vednnConvolutionParam_t * restrict pParamConv )
{
    LFTRACE_BEGIN("vednn-ConvBkD-gemm");
    int n, g;

    int batch       = pParamGradOut->batch;
    int gOutChannel = pParamGradOut->channel;
    int gOutWidth   = pParamGradOut->width;
    int gOutHeight  = pParamGradOut->height;
    int gInChannel  = pParamGradIn->channel;
    int gInWidth    = pParamGradIn->width;
    int gInHeight   = pParamGradIn->height;
    int kernWidth   = pParamKernel->width;
    int kernHeight  = pParamKernel->height;

    int group       = pParamConv->group;
    int strideWidth = pParamConv->strideWidth;;
    int strideHeight    = pParamConv->strideHeight;
    int padWidth    = pParamConv->padWidth;
    int padHeight   = pParamConv->padHeight;
    int dilationWidth   = pParamConv->dilationWidth;
    int dilationHeight  = pParamConv->dilationHeight;

    int gOutChannelGroup = gOutChannel  / group;
    int gInChannelGroup  = gInChannel / group;

    const float * restrict pGradOut = (float const*)pDataGradOut;
    const float * restrict pKernel  = (float const*)pDataKernel;
    float * restrict pGradIn  = (float*)pDataGradIn;

    int no_im2col = (kernWidth == 1 && kernHeight == 1 && strideWidth == 1 && strideHeight == 1 && padWidth == 0 && padHeight == 0);

    for (n = 0; n < batch; n++) { // this->num_
        int gOutBatchOffset  = n * gOutChannel  * gOutWidth  * gOutHeight;
        int gInBatchOffset = n * gInChannel * gInWidth * gInHeight;

        for (g = 0; g < group; g++) {
            int gOutGroupOffset = g *                   gOutChannelGroup * gOutHeight * gOutWidth;
            int gInGroupOffset  = g * gInChannelGroup                    * gInHeight  * gInWidth;
            int kernGroupOffset = g * gInChannelGroup * gOutChannelGroup * kernHeight * kernWidth;

            int gOutOffset = gOutBatchOffset + gOutGroupOffset;
            int gInOffset  = gInBatchOffset  + gInGroupOffset;

            int M = gInChannelGroup * kernWidth * kernHeight;
            int N = gOutWidth * gOutHeight;
            int K = gOutChannelGroup;

            if( no_im2col ) {
                SGEMM(&NOTRANS, &TRANS, &N, &M, &K,
                        &FONE, (float *) &pGradOut[gOutOffset], &N,
                        (float *) &pKernel[kernGroupOffset], &M,
                        &FZERO, &pGradIn[gInOffset], &N);
            }
            else {
                SGEMM(&NOTRANS, &TRANS, &N, &M, &K,
                        &FONE, (float *) &pGradOut[gOutOffset], &N,
                        (float *) &pKernel[kernGroupOffset], &M,
                        &FZERO, pColBuff, &N);

                col2im_cpu(pColBuff,
                        gInChannelGroup, gInHeight, gInWidth, kernHeight, kernWidth,
                        padHeight, padWidth, strideHeight, strideWidth, dilationHeight, dilationWidth,
                        &pGradIn[gInOffset]);
            }
        } // group
    } // batch

    LFTRACE_END("vednn-ConvBkD-gemm");
    return VEDNN_SUCCESS;
}

    static vednnError_t
convolution_backward_filter_gemm(
        const vednnTensorParam_t * restrict pParamIn, const void * restrict pDataIn,
        const vednnTensorParam_t * restrict pParamGradOut, const void * restrict pDataGradOut,
        const vednnFilterParam_t * restrict pParamGradKernel, void * restrict pDataGradKernel,
        float * restrict pColBuff,
        const vednnConvolutionParam_t * restrict pParamConv )
{
    LFTRACE_BEGIN("vednn-ConvBkF-gemm");
    int n, g;

    int batch       = pParamIn->batch;
    int inChannel   = pParamIn->channel;
    int inWidth     = pParamIn->width;
    int inHeight    = pParamIn->height;
    int outChannel  = pParamGradOut->channel;
    int outWidth    = pParamGradOut->width;
    int outHeight   = pParamGradOut->height;
    int kernWidth   = pParamGradKernel->width;
    int kernHeight  = pParamGradKernel->height;

    int group       = pParamConv->group;
    int strideWidth = pParamConv->strideWidth;;
    int strideHeight    = pParamConv->strideHeight;
    int padWidth    = pParamConv->padWidth;
    int padHeight   = pParamConv->padHeight;
    int dilationWidth   = pParamConv->dilationWidth;
    int dilationHeight  = pParamConv->dilationHeight;

    int inChannelGroup  = inChannel  / group;   // pParamKernel->inChannel と同じ
    int outChannelGroup = outChannel / group;   // pParamKernel->outChannel と同じ

    const float * restrict pIn     = (float const*)pDataIn;
    const float * restrict pOut    = (float const*)pDataGradOut;
    float * restrict pKernel       = (float*)pDataGradKernel ;

    int no_im2col = (kernWidth == 1 && kernHeight == 1 && strideWidth == 1 && strideHeight == 1 && padWidth == 0 && padHeight == 0);

    for (n = 0; n < batch; n++) { // this->num_
        int inBatchOffset  = n * inChannel  * inWidth  * inHeight;
        int outBatchOffset = n * outChannel * outWidth * outHeight;

        for (g = 0; g < group; g++) {
            int inGroupOffset   = g * inChannelGroup                   * inHeight   * inWidth;
            int outGroupOffset  = g * outChannelGroup                  * outHeight  * outWidth;
            int kernGroupOffset = g * outChannelGroup * inChannelGroup * kernHeight * kernWidth;

            int inOffset  = inBatchOffset  + inGroupOffset;
            int outOffset = outBatchOffset + outGroupOffset;

            if( no_im2col ) {
                int M = outChannelGroup;
                int N = inChannelGroup * kernWidth * kernHeight;
                int K = outWidth * outHeight;

                SGEMM(&TRANS, &NOTRANS, &N, &M, &K,
                        &FONE,  (float*)&pIn[inOffset], &K,
                        (float*)&pOut[outOffset], &K,
                        &FONE, &pKernel[kernGroupOffset], &N);
            }
            else {
                im2col_cpu(&pIn[inOffset],
                        inChannelGroup, inHeight, inWidth, kernHeight, kernWidth,
                        padHeight, padWidth, strideHeight, strideWidth, dilationHeight, dilationWidth,
                        pColBuff);

                int M = outChannelGroup;
                int N = inChannelGroup * kernWidth * kernHeight;
                int K = outWidth * outHeight;

                SGEMM(&TRANS, &NOTRANS, &N, &M, &K,
                        &FONE,  pColBuff, &K,
                        (float*)&pOut[outOffset], &K,
                        &FONE, &pKernel[kernGroupOffset], &N);
            }
        } // group
    } // batch

    LFTRACE_END("vednn-ConvBkF-gemm");
    return VEDNN_SUCCESS;
}

#if 0 // new : low-level impls ALWAYS take bias (can be nullptrs)
vednnError_t
vednnConvolutionForward_direct_gemm(
    const vednnTensorParam_t * restrict         pParamIn,
    const void * restrict                       pDataIn,
    const vednnFilterParam_t * restrict         pParamKernel,
    const void * restrict                       pDataKernel,
    const vednnConvolutionParam_t * restrict    pParamConv,
    const vednnTensorParam_t * restrict         pParamOut,
    void * restrict                             pDataOut
){
    size_t pColrows = pParamKernel->inChannel * pParamKernel->width * pParamKernel->height;
    size_t pColcols = pParamOut->width * pParamOut->height;
    // This buffer is only used for bias term
    float * restrict pOnes = nullptr;
    //float const* restrict pOnes = (float const*)
    //    (void*)vednn_scratchpad_float_ones(pColcols); // ow * oh float 1.0f

    // This buffer is used for a monolithic im2col
    // (could use a smaller buffer if im2col were done "as-needed")
    size_t const nBytes = pColrows * pColcols * getTensorDataSize(pParamIn);
#if SCRATCHPAD==0
    float * restrict pColBuff = (float*) XMALLOC(nBytes);
#else
    using vednn::scratchpad::vednn_scratchpad_shared;
    float * restrict pColBuff = (float*)(void*)
        vednn_scratchpad_shared(nBytes);
#endif

    vednnError_t ret = convolution_forward_gemm(
            pParamIn, pDataIn,
            pParamKernel, pDataKernel,
            nullptr, nullptr/*avoids bias gemm call*/ ,
            pParamOut, pDataOut/*output*/,
            pOnes, pColBuff, pParamConv );
#if SCRATCHPAD==0
    free(pColBuff);
#endif
    return ret;
} 
#endif
vednnError_t
vednnConvolutionForward_direct_gemm(
        VEDNN_CONVFWD_ARGS /* low-level impl std params */
)
{
    size_t pColrows = pParamKernel->inChannel * pParamKernel->width * pParamKernel->height;
    size_t pColcols = pParamOut->width * pParamOut->height;

    // This buffer is only used for bias term
#if SCRATCHPAD==0
    float * restrict pOnes = (float *) XMALLOC(pColcols * sizeof(float)/*bytes*/);
    for(int i = 0; i < pColcols; ++i)
      pOnes[i] = 1.0f;
#else
    using vednn::scratchpad::vednn_scratchpad_float_ones;
    float const* restrict pOnes = (float const*)
        vednn_scratchpad_float_ones(pColcols/*floats*/); // ow * oh float 1.0f
#endif

    // This buffer is used for a monolithic im2col
    // (could use a smaller buffer if im2col were done "as-needed")
    size_t const nBytes = pColrows * pColcols * getTensorDataSize(pParamIn);
#if SCRATCHPAD==0
    float * restrict pColBuff = (float *) XMALLOC(nBytes);
#else
    using vednn::scratchpad::vednn_scratchpad_shared;
    float * restrict pColBuff = (float*)(void*)
        vednn_scratchpad_shared(nBytes);
#endif
    vednnError_t ret = convolution_forward_gemm(
            pParamIn, pDataIn,
            pParamKernel, pDataKernel,
            pParamBias, pDataBias,
            pParamOut, pDataOut/*output*/,
            pOnes, pColBuff, pParamConv );
#if SCRATCHPAD==0
    free(pOnes);
    free(pColBuff);
#endif
    return ret;
}
vednnError_t
vednnConvolutionBackwardFilter_direct_gemm(
        VEDNN_CONVBKF_ARGS /* "NOWRAP" impl --> not VEDNN_CONV_BKF_OMPARGS */
){
    size_t pColrows = pParamGradKernel->inChannel * pParamGradKernel->width * pParamGradKernel->height;
    size_t pColcols = pParamGradOut->width * pParamGradOut->height;
    size_t const nBytes = pColrows * pColcols * getTensorDataSize(pParamIn);
#if SCRATCHPAD==0
    float * restrict pColBuff  = (float*) XMALLOC(nBytes);
#else
    using vednn::scratchpad::vednn_scratchpad_shared;
    float * restrict pColBuff = (float*)(void*)
        vednn_scratchpad_shared(nBytes);
#endif
    vednnError_t ret = convolution_backward_filter_gemm(
            pParamIn, pDataIn,
            pParamGradOut, pDataGradOut,
            pParamGradKernel, pDataGradKernel/*output*/,
            pColBuff, pParamConv );
#if SCRATCHPAD==0
    free(pColBuff);
#endif
    return ret;
}

vednnError_t
vednnConvolutionBackwardData_direct_gemm(
        VEDNN_CONVBKD_ARGS /* "NOWRAP" impl --> not VEDNN_CONV_BKF_OMPARGS */
)
{
    size_t pColrows = pParamKernel->inChannel * pParamKernel->width * pParamKernel->height;
    size_t pColcols = pParamGradOut->width * pParamGradOut->height;
    size_t const nBytes = pColrows * pColcols * getTensorDataSize(pParamGradIn);
#if SCRATCHPAD==0
    float * restrict pColBuff = (float*) XMALLOC(nBytes);
#else
    using vednn::scratchpad::vednn_scratchpad_shared;
    float * restrict pColBuff = (float*)(void*)
        vednn_scratchpad_shared(nBytes);
#endif
    vednnError_t ret =  convolution_backward_data_gemm(
            pParamGradOut, pDataGradOut,
            pParamKernel, pDataKernel,
            pParamGradIn, pDataGradIn/*output*/,
            pColBuff, pParamConv );
#if SCRATCHPAD==0
    free(pColBuff);
#endif
    return ret;
}
}//extern "C"
// vim: et ts=4 sw=4 cindent cino=^0,=0,l0,g2,\:0,N-s syntax=cpp.doxygen
