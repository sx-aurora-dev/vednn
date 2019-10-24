/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "vednn.h"
#include "C/vednn-def.h"  // __vednn_omp_num_threads
#include "vednn_helper.h"
#include "convolution_gemm.h"

#define LOCAL_FTRACE 1
#if LOCAL_FTRACE
#include "conv_test_param.h" // just for FTRACE macros
#define LFTRACE_BEGIN(...) FTRACE_BEGIN(__VA_ARGS__)
#define LFTRACE_END(...) FTRACE_END(__VA_ARGS__)
#define LFTRACE_IF(...) FTRACE_IF(__VA_ARGS__)
#else
#define LFTRACE_BEGIN(...) do{}while(0)
#define LFTRACE_END(...) do{}while(0)
#define LFTRACE_IF(...) do{}while(0)
#endif // LOCAL_FTRACE

#define SGEMM sgemm_
void sgemm_(char *TRANSA, char *TRANSB, int *M, int *N, int *K,
    float *ALPHA, float *A,  int *LDA, float *B, int *LDB,
    float *BETA, float *C, int *LDC ) ;

static char  TRANS   = 'T';
static char  NOTRANS = 'N';
static float FONE    = 1.0f;
static float FZERO   = 0.0f;
static int   IONE    = 1;

/* ----------------------------------------------------------------------- */
#if 0
static inline int is_a_ge_zero_and_a_lt_b(int a, int b) {
  //return (unsigned)a < (unsigned)b;
  return a>=0  && a<b; // for ncc auto vectorization, this is better
}
#else
static inline int64_t zero_le_a_lt_b(int64_t a, int64_t b) {
  return a>=0  && a<b; // for ncc auto vectorization, this is better
}
#endif

#if 0 // fast.  Unfortunately __vednn_omp_num_threads may often be zero (why?)
#ifdef USE_OPENMP
static int nThreads = 0;
#else
static int const nThreads = 1U;
#endif
#else // slightly less fast
#ifdef USE_OPENMP
//static int nThreads = 0;
static inline void chkThreads() {
  if(__vednn_omp_num_threads<=0) __vednn_omp_num_threads = omp_get_max_threads();
}
static inline int getThreads() {
  //return __vednn_omp_num_threads? __vednn_omp_num_threads: omp_get_max_threads();
  return __vednn_omp_num_threads;
}
#else
static inline void chkThreads() {}
static inline int getThreads() { return 1; }
#endif
#endif


/** data_col size to hold float[ic*kw*kh*ow*oh]. */
  static void
im2col_cpu(const float * restrict data_im, const int64_t channels,
    const int64_t height, const int64_t width, const int64_t kernel_h, const int64_t kernel_w,
    const int64_t pad_h, const int64_t pad_w,
    const int64_t stride_h, const int64_t stride_w,
    const int64_t dilation_h, const int64_t dilation_w,
    float * restrict data_col, int const threads)
{
  LFTRACE_BEGIN("im2col_cpu");
  const int64_t output_h = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
  const int64_t output_w = (width + 2 * pad_w -  (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
  const int64_t channel_size = height * width;

  int64_t channel, kernel_row;
  typedef long unsigned lu;

  int64_t const workPerChannel = kernel_h*kernel_w * output_h*output_w;
  //printf(" im2col channels %lu threads %lu vednn-threads %lu omp_threads %lu\n",(lu)channels,
  //    (lu)threads, (lu)__vednn_omp_num_threads, (lu)omp_get_max_threads());
  if( channels%threads && channels < 2*threads )
  { // collapse 2 loops [channels,kernel_h] to get more work per thread.
    //printf(" collapse(2)\n");
#if 0 // actually no diff for stride_w==1 "optimization"
    if( stride_w == 1 )
#pragma omp parallel
      //#pragma omp parallel if(channels>1 && workPerChannel > 65536)
    {
#pragma omp for private(channel,kernel_row) collapse(2)
      //#pragma omp parallel for private(channel) if(workPerChannel > 111)
      //#pragma omp parallel for private(channel,kernel_row)
      for (channel = 0 ; channel < channels; ++channel) {       // inChannel
        for (kernel_row = 0; kernel_row < kernel_h; ++kernel_row) {   // kernHeight
          int64_t kernel_col, output_row, output_col;

          int64_t const inOffset = channel * channel_size;
          int64_t outOffset = channel * workPerChannel + kernel_row * kernel_w*output_h*output_w;
          //if(channel==0) printf("inOffset=%-8lu outOffset=%-8lu\n",(lu)inOffset,(lu)outOffset);

          for (kernel_col = 0; kernel_col < kernel_w; kernel_col++) {   // kernWidth
#if 1 // orig code (faster)
            int64_t input_row = -pad_h + kernel_row * dilation_h;
            for (output_row = output_h; output_row; output_row--) { // outHeight
              //int64_t const input_row = -pad_h + kernel_row * dilation_h + output_row*stride_h;
              if (input_row < 0 || input_row >= height)//(!zero_le_a_lt_b(input_row,height))
              {
                for (output_col = 0; output_col<output_w; ++output_col) { // outWidth
                  data_col[outOffset++] = FZERO; //*(data_col++) = 0;
                }
              } else {
                int64_t const input_col = -pad_w + kernel_col * dilation_w;
                for (output_col = input_col; output_col<input_col+output_w; ++output_col) { // outWidth
                  data_col[outOffset++] //*(data_col++)
                    = ( output_col >= 0 && output_col < width
                        ? data_im[inOffset + input_row * width + output_col]
                        : FZERO);
                }
              }
              input_row += stride_h;
            }
#else // slower w/ combined loops
            int64_t const input_col = -pad_w + kernel_col * dilation_w;
            for (output_row= 0; output_row<output_h; ++output_row) {  // outHeight
              int64_t const input_row = -pad_h + kernel_row * dilation_h + output_row*stride_h;
              for (output_col = input_col; output_col<input_col+output_w; ++output_col) { // outWidth
                data_col[outOffset++] //*(data_col++)
                  = ( input_row >= 0 && input_row < height &&
                      output_col >= 0 && output_col < width
                      ? data_im[inOffset + input_row * width + output_col]
                      : FZERO);
              }
            }
#endif
          }
        }
      }
    }
    else // stride_w != 1
#endif
#pragma omp parallel
      //#pragma omp parallel if(channels>1 && workPerChannel > 65536)
    { // any stride_w > 0 ...
#pragma omp for private(channel,kernel_row) collapse(2)
      //#pragma omp parallel for private(channel) if(workPerChannel > 111)
      //#pragma omp parallel for private(channel,kernel_row)
      for (channel = 0 ; channel < channels; ++channel) {       // inChannel
        for (kernel_row = 0; kernel_row < kernel_h; ++kernel_row) {   // kernHeight
          int64_t kernel_col, output_rows, output_col;

          int64_t const inOffset = channel * channel_size;
          int64_t outOffset = channel * workPerChannel + kernel_row * kernel_w*output_h*output_w;
          //if(channel==0) printf("inOffset=%-8lu outOffset=%-8lu\n",(lu)inOffset,(lu)outOffset);

          for (kernel_col = 0; kernel_col < kernel_w; kernel_col++) {   // kernWidth
            int64_t input_row = -pad_h + kernel_row * dilation_h;
            for (output_rows = output_h; output_rows; output_rows--) {  // outHeight
              if (input_row < 0 || input_row >= height)//(!zero_le_a_lt_b(input_row,height))
              {
                for (output_col = output_w; output_col; output_col--) { // outWidth
                  data_col[outOffset++] = FZERO; //*(data_col++) = 0;
                }
              } else {
                int64_t input_col = -pad_w + kernel_col * dilation_w;
                for (output_col = output_w; output_col; output_col--) { // outWidth
                  data_col[outOffset++] //*(data_col++)
                    = //(zero_le_a_lt_b(input_col, width)
                    (0 <= input_col && input_col < width
                     ? data_im[inOffset + input_row * width + input_col]
                     : FZERO);
                  input_col += stride_w;
                }
              }
              input_row += stride_h;
            }
          }
        }
      }
    }
  }else{ // collapse just one outer [channels] loop
#pragma omp parallel
//#pragma omp parallel if(channels>1 && workPerChannel > 65536)
    {
#pragma omp for private(channel,kernel_row)
      //#pragma omp parallel for private(channel) if(workPerChannel > 111)
      //#pragma omp parallel for private(channel)
      for (channel = 0 ; channel < channels; ++channel) {       // inChannel
        int64_t kernel_row, kernel_col, output_rows, output_cols, output_col;

        int64_t inOffset = channel * channel_size;
        int64_t outOffset = channel * workPerChannel;

        for (kernel_row = 0; kernel_row < kernel_h; kernel_row++) {   // kernHeight
          //if(channel==0) printf("inOffset=%-8lu outOffset=%-8lu\n",(lu)inOffset,(lu)outOffset);
          for (kernel_col = 0; kernel_col < kernel_w; kernel_col++) {   // kernWidth
            int64_t input_row = -pad_h + kernel_row * dilation_h;
            for (output_rows = output_h; output_rows; output_rows--) {  // outHeight
              if (input_row < 0 || input_row >= height)//(!zero_le_a_lt_b(input_row,height))
              {
                for (output_cols = output_w; output_cols; output_cols--) { // outWidth
                  data_col[outOffset++] = 0; //*(data_col++) = 0;
                }
              } else {
                int64_t input_col = -pad_w + kernel_col * dilation_w;
                for (output_col = output_w; output_col; output_col--) { // outWidth
                  data_col[outOffset++] //*(data_col++)
                    = //(zero_le_a_lt_b(input_col, width)
                    (0 <= input_col && input_col < width
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
    }
  }
  LFTRACE_END("im2col_cpu");
}

  static void
col2im_cpu(
    const float* data_col, const int64_t channels,
    const int64_t height, const int64_t width, const int64_t kernel_h, const int64_t kernel_w,
    const int64_t pad_h, const int64_t pad_w,
    const int64_t stride_h, const int64_t stride_w,
    const int64_t dilation_h, const int64_t dilation_w,
    float* data_im, int const threads)
{
  LFTRACE_BEGIN("col2im_cpu");
  memset(data_im, 0, sizeof(float)*height*width*channels) ;

  const int64_t output_h = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
  const int64_t output_w = (width + 2 * pad_w -  (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
  const int64_t channel_size = height * width;

  int64_t channel;

  int64_t const work = channels * kernel_h*kernel_w * output_h*output_w;
#pragma omp parallel if(channels>=3 && work>1024)
  {
#pragma omp for
    for (channel = 0 ; channel < channels; channel++) {       // inChannel
      int64_t kernel_row, kernel_col, output_rows, output_cols, output_col;

      int64_t inOffset = channel * channel_size;
      int64_t outOffset = channel * output_h * output_w * kernel_h * kernel_w;

      for (kernel_row = 0; kernel_row < kernel_h; kernel_row++) {   // kernHeight
        for (kernel_col = 0; kernel_col < kernel_w; kernel_col++) {   // kernWidth
          int64_t input_row = -pad_h + kernel_row * dilation_h;
          for (output_rows = output_h; output_rows; output_rows--) {  // outHeight
            if (!zero_le_a_lt_b(input_row, height)) {
              for (output_cols = output_w; output_cols; output_cols--) {
                // *(data_col++) = 0;
                //data_col[outOffset++] ;
                ++outOffset;
              }
            } else {
              int64_t input_col = -pad_w + kernel_col * dilation_w;
              for (output_col = output_w; output_col; output_col--) { // outWidth
                if (zero_le_a_lt_b(input_col, width)) {
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
  }
  LFTRACE_END("col2im_cpu");
}

  vednnError_t
convolution_forward_gemm(
    const vednnTensorParam_t * restrict pParamIn, const void * restrict pDataIn,
    const vednnFilterParam_t * restrict pParamKernel, const void * restrict pDataKernel,
    const vednnBiasParam_t * restrict pParamBias, const void * restrict pDataBias,
    const vednnTensorParam_t * restrict pParamOut, void * restrict pDataOut,
    const float * restrict pOne,  float * restrict pColBuff,
    const vednnConvolutionParam_t * restrict pParamConv )
{
  LFTRACE_BEGIN("convolution_forward_gemm");
  int n, g;

  int batch   = pParamIn->batch;
  int inChannel = pParamIn->channel;
  int inWidth   = pParamIn->width;
  int inHeight  = pParamIn->height;
  int outChannel  = pParamOut->channel;
  int outWidth  = pParamOut->width;
  int outHeight = pParamOut->height;
  int kernWidth = pParamKernel->width;
  int kernHeight  = pParamKernel->height;

  int group   = pParamConv->group;
  int strideWidth = pParamConv->strideWidth;;
  int strideHeight  = pParamConv->strideHeight;
  int padWidth  = pParamConv->padWidth;
  int padHeight = pParamConv->padHeight;
  int dilationWidth = pParamConv->dilationWidth;
  int dilationHeight  = pParamConv->dilationHeight;

  int inChannelGroup  = inChannel  / group; // pParamKernel->inChannel と同じ
  int outChannelGroup = outChannel / group; // pParamKernel->outChannel と同じ

  const float * restrict pIn     = pDataIn;
  const float * restrict pBias   = pDataBias;
  const float * restrict pKernel = pDataKernel;
  float * restrict pOut    = pDataOut;

  int no_im2col = (kernWidth == 1 && kernHeight == 1 && strideWidth == 1 && strideHeight == 1 && padWidth == 0 && padHeight == 0);
  chkThreads();

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
            pColBuff, getThreads());


        SGEMM(&NOTRANS, &NOTRANS, &N, &M, &K,
            &FONE, pColBuff, &N,
            (float *)&pKernel[kernGroupOffset], &K,
            &FZERO, &pOut[outOffset], &N);

        if (pBias) {
          SGEMM(&NOTRANS, &NOTRANS, &N, &M, &IONE,
              &FONE, (float *)pOne, &N,
              (float *) &pBias[biasGroupOffset], &IONE,
              &FONE, &pOut[outOffset], &N);
        }
      }
    } // group
  } // batch

  LFTRACE_END("convolution_forward_gemm");
  return VEDNN_SUCCESS;
}

  vednnError_t
convolution_backward_data_gemm(
    const vednnTensorParam_t * restrict pParamGradOut, const void * restrict pDataGradOut,
    const vednnFilterParam_t * restrict pParamKernel, const void * restrict pDataKernel,
    const vednnTensorParam_t * restrict pParamGradIn, void * restrict pDataGradIn,
    float * restrict pColBuff,
    const vednnConvolutionParam_t * restrict pParamConv )
{
  LFTRACE_BEGIN("convolution_backward_data_gemm");
  int n, g;

  int batch   = pParamGradOut->batch;
  int gOutChannel = pParamGradOut->channel;
  int gOutWidth = pParamGradOut->width;
  int gOutHeight  = pParamGradOut->height;
  int gInChannel  = pParamGradIn->channel;
  int gInWidth  = pParamGradIn->width;
  int gInHeight = pParamGradIn->height;
  int kernWidth = pParamKernel->width;
  int kernHeight  = pParamKernel->height;

  int group   = pParamConv->group;
  int strideWidth = pParamConv->strideWidth;;
  int strideHeight  = pParamConv->strideHeight;
  int padWidth  = pParamConv->padWidth;
  int padHeight = pParamConv->padHeight;
  int dilationWidth = pParamConv->dilationWidth;
  int dilationHeight  = pParamConv->dilationHeight;

  int gOutChannelGroup = gOutChannel  / group;
  int gInChannelGroup  = gInChannel / group;

  const float * restrict pGradOut = pDataGradOut;
  const float * restrict pKernel  = pDataKernel;
  float * restrict pGradIn  = pDataGradIn;

  int no_im2col = (kernWidth == 1 && kernHeight == 1 && strideWidth == 1 && strideHeight == 1 && padWidth == 0 && padHeight == 0);
  chkThreads();

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
            &pGradIn[gInOffset], getThreads());
      }
    } // group
  } // batch

  LFTRACE_END("convolution_backward_data_gemm");
  return VEDNN_SUCCESS;
}

  vednnError_t
convolution_backward_filter_gemm(
    const vednnTensorParam_t * restrict pParamIn, const void * restrict pDataIn,
    const vednnTensorParam_t * restrict pParamGradOut, const void * restrict pDataGradOut,
    const vednnFilterParam_t * restrict pParamGradKernel, void * restrict pDataGradKernel,
    float * restrict pColBuff,
    const vednnConvolutionParam_t * restrict pParamConv )
{
  LFTRACE_BEGIN("convolution_backward_filter_gemm");
  int n, g;

  int batch   = pParamIn->batch;
  int inChannel = pParamIn->channel;
  int inWidth   = pParamIn->width;
  int inHeight  = pParamIn->height;
  int outChannel  = pParamGradOut->channel;
  int outWidth  = pParamGradOut->width;
  int outHeight = pParamGradOut->height;
  int kernWidth = pParamGradKernel->width;
  int kernHeight  = pParamGradKernel->height;

  int group   = pParamConv->group;
  int strideWidth = pParamConv->strideWidth;;
  int strideHeight  = pParamConv->strideHeight;
  int padWidth  = pParamConv->padWidth;
  int padHeight = pParamConv->padHeight;
  int dilationWidth = pParamConv->dilationWidth;
  int dilationHeight  = pParamConv->dilationHeight;

  int inChannelGroup  = inChannel  / group; // pParamKernel->inChannel と同じ
  int outChannelGroup = outChannel / group; // pParamKernel->outChannel と同じ

  const float * restrict pIn     = pDataIn;
  const float * restrict pOut    = pDataGradOut;
  float * restrict pKernel = pDataGradKernel ;

  int no_im2col = (kernWidth == 1 && kernHeight == 1 && strideWidth == 1 && strideHeight == 1 && padWidth == 0 && padHeight == 0);
  chkThreads();


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
            pColBuff, getThreads());

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

  LFTRACE_END("convolution_backward_filter_gemm");
  return VEDNN_SUCCESS;
}
// vim: et ts=2 sw=2 cindent cino=^0,=0,l0,\:0,N-s syntax=cpp.doxygen
