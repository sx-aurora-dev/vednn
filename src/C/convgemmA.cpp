/* -*- Mode: C; c-basic-offset:2 ; indent-tabs-mode:nil ; -*- */

#include "vednn.h"
#include "vednn-def.hpp" // vednn.h + C++ scratchpad API (C++ can inline one call)

#include <stdio.h>
#include <assert.h>
#include <string.h> // memset
#include <stdint.h>
//#include <mcheck.h> // mprobe(malloc-ptr)

/** 0=malloc+free, 1=vednn_scratchpad_shared, 2=vednn_scratchpadTLS.
 * - malloc+free is safe, but maybe slow.
 * - shared scratch is fine but will be shared between all threads (take care)
 * - TLS is thread-local scratchpad, may work in more situations.
 */
#ifndef SCRATCHPAD
#define SCRATCHPAD 2
#endif

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

/** local scratchpad alloc via \c SCRATCHPAD method */
static inline float* restrict getScratch( size_t nBytes ){
#if SCRATCHPAD==0
  float * restrict ret = (float*) malloc(nBytes);
#elif SCRATCHPAD==1
  //printf(" dirGemmA: scratchpad_shared!"); fflush(stdout);
  using vednn::scratchpad::vednn_scratchpad_shared;
  float * restrict ret = (float*)(void*)
    vednn_scratchpad_shared(nBytes);
#else // 2... TLS
  using vednn::scratchpad::vednn_scratchpadTLS;
  float * restrict ret = (float*)(void*)
    vednn_scratchpadTLS(nBytes);
#endif
  return ret;
}

/** return sizeof(float) if pParam is DTYPE_FLOAT */
static inline size_t getTensorDataSize(const vednnTensorParam_t * restrict pParam)
{
  size_t dataSize = 0;
  switch (pParam->dtype) {
  case DTYPE_FLOAT: dataSize = sizeof(float);
                    break;
  }
  assert(dataSize>0); /* BUG */
  return dataSize;
}

typedef long unsigned lu;

#ifndef OPENMP
// put omp workarounds in header XXX
static inline int omp_get_thread_num() {return 0;}
static inline int omp_get_num_threads() {return 1;}
#endif

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
//static inline int is_a_ge_zero_and_a_lt_b(int a, int b) {
//  //return (unsigned)a < (unsigned)b;
//  return a>=0  && a<b; // for ncc auto vectorization, this is better
//}

/** data_col size to hold float[ic*kw*kh*ow*oh]. */
  static void
vednn_im2colA(const float * restrict data_im, const int64_t channels,
    const int64_t height, const int64_t width, const int64_t kernel_h, const int64_t kernel_w,
    const int64_t pad_h, const int64_t pad_w,
    const int64_t stride_h, const int64_t stride_w,
    const int64_t dilation_h, const int64_t dilation_w,
    float * restrict const data_col)
{
  LFTRACE_BEGIN("vednn_im2colA");
  const int64_t output_h = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
  const int64_t output_w = (width + 2 * pad_w -  (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
  const int64_t channel_size = height * width;
  //printf(" vednn_im2colA"); fflush(stdout); mcheck_check_all(); printf( "Begins..."); fflush(stdout);

  if( stride_w != 1 ){
    // nc++ may now support if clause?
    // dnnl tests and special-cases: stride==1 (and some other thing)
    //OMP(parallel for if(channels>=3))//;
    OMP(parallel for if(channels>1))//;
    for (int64_t channel = 0 ; channel < channels; channel++) {         // inChannel
      int64_t const inOffset = channel * channel_size;
      int64_t outOffset = channel * output_h * output_w * kernel_h * kernel_w;
      for (int64_t kernel_row = 0; kernel_row < kernel_h; kernel_row++) {     // kernHeight
        for (int64_t kernel_col = 0; kernel_col < kernel_w; kernel_col++) {   // kernWidth
          int64_t input_row = -pad_h + kernel_row * dilation_h;
          for (int64_t output_rows = output_h; output_rows; output_rows--) {  // outHeight
            if (input_row < 0 || input_row >= height ) {
              for (int64_t output_col = output_w; output_col; --output_col) { // outWidth
                data_col[outOffset++] = FZERO; //*(data_col++) = 0;
              }
            } else {
              int64_t input_col = -pad_w + kernel_col * dilation_w;
              for (int64_t output_col = 0; output_col<output_w; ++output_col) {	// outWidth
                data_col[outOffset++] //*(data_col++)
                  = ( input_col >= 0 && input_col < width
                    ? data_im[inOffset + input_row * width + input_col]
                    : FZERO);
                input_col += stride_w;
              }
            }
            input_row += stride_h;
          }
        }
      }
      //if(channel+1 >= channels){ printf(" thr%d exit outOffset=%lu\n",
      //    omp_get_thread_num(), (lu)outOffset); fflush(stdout); }
    } // inChannel
  }else{ //stride_w == 1
    OMP(parallel for if(channels>=3))//;
    for (int64_t channel = 0 ; channel < channels; channel++) {             // inChannel
      int64_t kernel_row, kernel_col, output_rows, output_cols, output_col;

      int64_t inOffset = channel * channel_size;
      int64_t outOffset = channel * output_h * output_w * kernel_h * kernel_w;
      //mcheck_check_all();
      //if(channel==0) {printf(" im2col stride_w=%lu channel 0: outOffset=%lu"
      //    "omp thr %d of %d . pColBuff[%lu] @ %p\n",
      //    (lu)stride_w, (lu)outOffset, omp_get_thread_num(), omp_get_num_threads(),
      //    (void*)data_col); fflush(stdout); }

      for (kernel_row = 0; kernel_row < kernel_h; kernel_row++) {     // kernHeight
        for (kernel_col = 0; kernel_col < kernel_w; kernel_col++) {     // kernWidth
          int64_t input_row = -pad_h + kernel_row * dilation_h;
          for (output_rows = output_h; output_rows; output_rows--) {  // outHeight
            if (input_row < 0 || input_row >= height ) {
              for (output_col = 0; output_col<output_w; ++output_col) {	// outWidth
                data_col[outOffset++] = FZERO;
              }
            } else {
              int64_t const input_col = -pad_w + kernel_col * dilation_w;
              for (output_col = 0; output_col<output_w; ++output_col) {	// outWidth
                data_col[outOffset++]
                  = ( input_col+output_col >= 0 && input_col+output_col < width
                      ? data_im[inOffset + input_row * width + input_col + output_col]
                      : FZERO);
              }
            }
            input_row += stride_h;
          }
        }
      }
    }
  }// end stride_w cases
  //printf(" vednn_im2colA exit..."); fflush(stdout); mcheck_check_all(); printf(" OK\n"); fflush(stdout);
  LFTRACE_END("vednn_im2colA");
  //printf(" vednn_im2colA exitB..."); fflush(stdout); mcheck_check_all(); printf(" OK\n"); fflush(stdout);
}

/** maskless specialization: kh1, a stride != 1.
 * kh1sh1 is a "no in2col" situation. Hmmm. very little speedup? */
  static inline void
vednn_im2col_k1p0_strA(const float * restrict data_im, const int64_t channels,
    const int64_t height, const int64_t width,
    /* kernel 1x1  : const int64_t kernel_h, const int64_t kernel_w,*/
    /* pad    == 0 : const int64_t pad_h, const int64_t pad_w,*/
    const int64_t stride_h, const int64_t stride_w,
    /* dilation irrelevant for 1x1 kernel : const int64_t dilation_h, const int64_t dilation_w, */
    float * restrict data_col)
{
  LFTRACE_BEGIN("vednn_im2col_k1p0_strA");
  const int64_t output_h = (height - 1) / stride_h + 1;
  const int64_t output_w = (width  - 1) / stride_w + 1;
  const int64_t channel_size = height * width;
  int64_t const output_hw = output_h*output_w; // workPerChannel

  //if( channels < 2*threads && channels%threads && channels*output_h > 1 )
  if( channels < 8 && output_h > 1 )
  {
#pragma omp for collapse(2) // no effect of collapse(2) ?
    for (int64_t channel = 0u ; channel < channels; ++channel) {       // inChannel
      for(int64_t output_row = 0; output_row < output_h; ++output_row){
        for(int64_t output_col = 0; output_col < output_w; output_col += stride_w){
          // input_row = output_row * stride_h; input_col = output_col * stride_w;
          data_col[channel*output_hw  + output_row          * output_w + output_col]
            = data_im[channel*channel_size + output_row*stride_h * width    + output_col*stride_w];
        } } }
  }else{
#pragma omp parallel for
  for (int64_t channel = 0u ; channel < channels; ++channel) {       // inChannel
    int64_t inOffset = channel * channel_size;
    // __builtin_vprefetch( &data_im[inOffset], output_hw*sizeof(float) );
    int64_t outOffset = channel * output_hw;
    for(int64_t input_row = 0; input_row < output_h*stride_h; input_row += stride_h){
      for(int64_t input_col = 0; input_col < output_w*stride_w; input_col += stride_w){
        data_col[outOffset++] = data_im[inOffset + input_row * width + input_col];
      } } }
  }
  LFTRACE_END("vednn_im2col_k1p0_strA");
}

  static void
vednn_col2imA(
    const float* data_col, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    float* data_im)
{
  LFTRACE_BEGIN("vednn_col2imA");
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
          if (input_row < 0 || input_row >= height){
            //for (output_cols = output_w; output_cols; output_cols--) {
            //  // *(data_col++) = 0;
            //  //data_col[outOffset++] ;
            //  ++outOffset;
            outOffset += output_cols;
          } else {
            int input_col = -pad_w + kernel_col * dilation_w;
            for (output_col = output_w; output_col; output_col--) { // outWidth
              if (input_col >= 0 && input_col < width) {
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
  LFTRACE_END("vednn_col2imA");
}

extern "C" { //}
  static vednnError_t
vednnConvFwd_gemmA(
    const vednnTensorParam_t * restrict pParamIn, const void * restrict pDataIn,
    const vednnFilterParam_t * restrict pParamKernel, const void * restrict pDataKernel,
    const vednnBiasParam_t * restrict pParamBias, const void * restrict pDataBias,
    const vednnTensorParam_t * restrict pParamOut, void * restrict pDataOut,
    const float * restrict pOne,  float * restrict pColBuff,
    const vednnConvolutionParam_t * restrict pParamConv )
{
  LFTRACE_BEGIN("vednnConvFwd_gemmA");
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

        if( kernHeight+kernWidth==2 && padHeight+padWidth==0 && strideHeight+strideWidth>2 ){
          LFTRACE_BEGIN("vednn_im2col_k1p0_inlA");
          const int64_t output_h = (inHeight - 1) / strideHeight + 1;
          const int64_t output_w = (inWidth  - 1) / strideWidth + 1;
          const int64_t channel_size = inHeight * inWidth;
          int64_t const workPerChannel = output_h*output_w;
#pragma omp parallel for
          for (int64_t channel = 0 ; channel < inChannelGroup; ++channel) {       // inChannel
            int64_t outOffset = channel * workPerChannel;
            for(int64_t input_row = 0; input_row < output_h*strideHeight; input_row += strideHeight){
              for(int64_t input_col = 0; input_col < output_w*strideWidth; input_col += strideWidth){
                pColBuff[outOffset++] = pIn[inOffset + channel*channel_size + input_row*inWidth + input_col];
              } } }
          LFTRACE_END("vednn_im2col_k1p0_inlA");

        }else{

          vednn_im2colA(&pIn[inOffset],
              inChannelGroup, inHeight, inWidth, kernHeight, kernWidth,
              padHeight, padWidth, strideHeight, strideWidth, dilationHeight, dilationWidth,
              pColBuff);

        }

        int M = outChannelGroup;
        int N = outWidth * outHeight;
        int K = inChannelGroup * kernWidth * kernHeight;

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

  LFTRACE_END("vednnConvFwd_gemmA");
  return VEDNN_SUCCESS;
}

  static vednnError_t
vednnConvBkD_gemmA(
    const vednnTensorParam_t * restrict pParamGradOut, const void * restrict pDataGradOut,
    const vednnFilterParam_t * restrict pParamKernel, const void * restrict pDataKernel,
    const vednnTensorParam_t * restrict pParamGradIn, void * restrict pDataGradIn,
    float * restrict pColBuff,
    const vednnConvolutionParam_t * restrict pParamConv )
{
  LFTRACE_BEGIN("vednnConvBkD_gemmA");
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

        vednn_col2imA(pColBuff,
            gInChannelGroup, gInHeight, gInWidth, kernHeight, kernWidth,
            padHeight, padWidth, strideHeight, strideWidth, dilationHeight, dilationWidth,
            &pGradIn[gInOffset]);
      }
    } // group
  } // batch

  LFTRACE_END("vednnConvBkD_gemmA");
  return VEDNN_SUCCESS;
}

  vednnError_t
vednnConvBkF_gemmA(
    const vednnTensorParam_t * restrict pParamIn, const void * restrict pDataIn,
    const vednnTensorParam_t * restrict pParamGradOut, const void * restrict pDataGradOut,
    const vednnFilterParam_t * restrict pParamGradKernel, void * restrict pDataGradKernel,
    float * restrict pColBuff,
    const vednnConvolutionParam_t * restrict pParamConv )
{
  LFTRACE_BEGIN("vednnConvBkF_gemmA");
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
        vednn_im2colA(&pIn[inOffset],
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

  LFTRACE_END("vednnConvBkF_gemmA");
  return VEDNN_SUCCESS;
}

vednnError_t
vednnConvolutionForward_direct_gemmA(
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
  float * restrict pColBuf = getScratch(nBytes);
  if(!pColBuf) return VEDNN_ERROR_MEMORY_EXHAUST;
  //printf(" nBytes %lu = ic*kw*kh %lu * ow*oh %lu *%lu\n",
  //    (lu)nBytes,(lu)pColrows,(lu)pColcols,(lu)getTensorDataSize(pParamIn));
  //fflush(stdout);
  //printf(" omp thr %d of %d . pColBuff[%lu] @ %p\n",omp_get_thread_num(),
  //    omp_get_num_threads(), (lu)nBytes,(void*)pColBuff); fflush(stdout);
  //mprobe(pColBuff);
  //mcheck_check_all();

  vednnError_t ret = vednnConvFwd_gemmA(
      pParamIn, pDataIn,
      pParamKernel, pDataKernel,
      nullptr, nullptr/*avoids bias gemm call*/ ,
      pParamOut, pDataOut/*output*/,
      pOnes, pColBuf, pParamConv );
#if SCRATCHPAD==0
  free(pColBuff);
#endif
  return ret;
} 
  vednnError_t
vednnConvolutionForwardAddBias_direct_gemmA(
    const vednnTensorParam_t * restrict         pParamIn,
    const void * restrict                       pDataIn,
    const vednnFilterParam_t * restrict         pParamKernel,
    const void * restrict                       pDataKernel,
    const vednnBiasParam_t * restrict           pParamBias,
    const void * restrict                       pDataBias,
    const vednnConvolutionParam_t * restrict    pParamConv,
    const vednnTensorParam_t * restrict         pParamOut,
    void * restrict                             pDataOut
    )
{
  size_t pColrows = pParamKernel->inChannel * pParamKernel->width * pParamKernel->height;
  size_t pColcols = pParamOut->width * pParamOut->height;

  // This buffer is only used for bias term
#if SCRATCHPAD==0
  float * restrict pOnes = (float *) malloc(pColcols * sizeof(float)/*bytes*/);
  if(!pOnes) return VEDNN_ERROR_MEMORY_EXHAUST;
  for(int i = 0; i < pColcols; ++i)
    pOnes[i] = 1.0f;
#else
  using vednn::scratchpad::vednn_scratchpad_float_ones;
  float const* restrict pOnes = (float const*)
    vednn_scratchpad_float_ones(pColcols/*floats*/); // ow * oh float 1.0f
  if(!pOnes) return VEDNN_ERROR_MEMORY_EXHAUST;
#endif

  // This buffer is used for a monolithic im2col
  // (could use a smaller buffer if im2col were done "as-needed")
  size_t const nBytes = pColrows * pColcols * getTensorDataSize(pParamIn);
  float * restrict const pColBuf = getScratch(nBytes);
  if(!pColBuf) return VEDNN_ERROR_MEMORY_EXHAUST;
  //printf(" nBytes %lu = ic*kw*kh %lu * ow*oh %lu *%lu\n",
  //    (lu)nBytes,(lu)pColrows,(lu)pColcols,(lu)getTensorDataSize(pParamIn));
  //fflush(stdout);

  vednnError_t ret = vednnConvFwd_gemmA(
      pParamIn, pDataIn,
      pParamKernel, pDataKernel,
      pParamBias, pDataBias,
      pParamOut, pDataOut/*output*/,
      pOnes, pColBuf, pParamConv );
#if SCRATCHPAD==0
  free(pOnes);
  free(pColBuf);
#endif
  return ret;
}
vednnError_t
vednnConvolutionBackwardFilter_direct_gemmA(
    const vednnTensorParam_t * restrict         pParamIn,
    const void * restrict                       pDataIn,
    const vednnTensorParam_t * restrict         pParamGradOut,
    const void * restrict                       pDataGradOut,
    const vednnConvolutionParam_t * restrict    pParamConv,
    const vednnFilterParam_t * restrict         pParamGradKernel,
    void * restrict                             pDataGradKernel
    // *** NOTE: direct_gemmA does not go via standard openmp wrappers!
    //#ifdef VEDNN_USE_OPENMP
    //    ,
    //    const int64_t                               beginOChannel,
    //    const int64_t                               nOChannel
    //#endif
    ){
  size_t const pColrows = pParamGradKernel->inChannel * pParamGradKernel->width * pParamGradKernel->height;
  size_t const pColcols = pParamGradOut->width * pParamGradOut->height;
  size_t const nBytes = pColrows * pColcols * getTensorDataSize(pParamIn);
  float * restrict const pColBuf = getScratch(nBytes);
  if(!pColBuf) return VEDNN_ERROR_MEMORY_EXHAUST;
  vednnError_t ret = vednnConvBkF_gemmA(
      pParamIn, pDataIn,
      pParamGradOut, pDataGradOut,
      pParamGradKernel, pDataGradKernel/*output*/,
      pColBuf, pParamConv );
#if SCRATCHPAD==0
  free(pColBuf);
#endif
  return ret;
}

  vednnError_t
vednnConvolutionBackwardData_direct_gemmA(
    const vednnTensorParam_t * restrict         pParamGradOut,
    const void * restrict                       pDataGradOut,
    const vednnFilterParam_t * restrict         pParamKernel,
    const void * restrict                       pDataKernel,
    const vednnConvolutionParam_t * restrict    pParamConv,
    const vednnTensorParam_t * restrict         pParamGradIn,
    void * restrict                             pDataGradIn
    )
{
  size_t const pColrows = pParamKernel->inChannel * pParamKernel->width * pParamKernel->height;
  size_t const pColcols = pParamGradOut->width * pParamGradOut->height;
  size_t const nBytes = pColrows * pColcols * getTensorDataSize(pParamGradIn);
  float * restrict const pColBuf = getScratch(nBytes);
  if(!pColBuf) return VEDNN_ERROR_MEMORY_EXHAUST;
  vednnError_t ret =  vednnConvBkD_gemmA(
      pParamGradOut, pDataGradOut,
      pParamKernel, pDataKernel,
      pParamGradIn, pDataGradIn/*output*/,
      pColBuf, pParamConv );
#if SCRATCHPAD==0
  free(pColBuff);
#endif
  return ret;
}



}//extern "C"
// vim: et ts=2 sw=2 cindent cino=^0,=0,l0,g2,\:0,N-s syntax=cpp.doxygen
