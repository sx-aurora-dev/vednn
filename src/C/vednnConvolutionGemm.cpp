/* -*- Mode: C; c-basic-offset:2 ; indent-tabs-mode:nil ; -*- */

#include "vednnConvolutionForward.h" // VEDNN_CONVFWD_ARGS_LIST
#include "vednn.h"
#include "vednn-def.hpp"             // vednn.h + C++ scratchpad API (C++ can inline one call)
#include "gen-dnn/mkldnn_os.h"       // OMP pragmas (part of mkldnn_thread now?)
#include "gen-dnn/mkldnn_thread.hpp" // omp_get_xxx stubs (if nec), parallel_nd
#include "gen-dnn/utils.hpp"

#include <stdio.h>
#include <assert.h>
#include <string.h> // memset
#include <stdint.h>

/** 0=malloc+free, 1=vednn_scratchpad_shared, 2=vednn_scratchpadTLS.
 * - '0' is dumb and safe
 * - '1' works, but needs care when client code is itself multi-threaded,
 *              (and needs care if trying 'outer threading' around im2col)
 * - '2' was ok for mb=1, but may SOMETIMES have malloc issues when mb>=8 ? XXX CONFIRMED, Dec2020
 *   - perhaps related to thread_local init issues ??
 *
 * - really want `omp threadprivate`, but not available for Aurora yet
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

#if 0
#define DBG(...) do{printf(__VA_ARGS__);fflush(stdout);}while(0)
#else
#define DBG(...)
#endif

/** local scratchpad alloc via \c SCRATCHPAD method */
#if SCRATCHPAD==0 // gemm prefers TLS, but ncc may have issues with thread_local?
static inline float* restrict getScratch( size_t nBytes ){
  float * restrict ret = (float*) malloc(nBytes);
  return ret;
}
static inline void freeScratch(void* scratch){
  free(scratch);
}
#elif SCRATCHPAD==1 // shared: memory ptr accessible by all threads
static inline float* restrict getScratch( size_t nBytes ){
  //printf(" dirGemmA: scratchpad_shared(%lu)!",(long unsigned)nBytes)); fflush(stdout);
  using vednn::scratchpad::vednn_scratchpad_shared;
  float * restrict ret = (float*)(void*)
    vednn_scratchpad_shared(nBytes);
  return ret;
}
static inline void freeScratch(void* scratch){
  (void)scratch;
  //printf(" dirGemmA: free scratchpad_shared!"); fflush(stdout);
}
#else // 2... TLS
static inline float* restrict getScratch( size_t nBytes ){
  //printf(" dirGemmA: scratchpadTLS(%lu)!",(long unsigned)nBytes)); fflush(stdout);
  using vednn::scratchpad::vednn_scratchpadTLS;
  float * restrict ret = (float*)(void*)
    vednn_scratchpadTLS(nBytes);
  return ret;
}
static inline void freeScratch(void* scratch){
  (void)scratch;
  //printf(" dirGemmA: free scratchpad_shared!"); fflush(stdout);
}
#endif

typedef long unsigned lu;

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

// Hmm replace using vednn.h macros vednn_get/set_num_threads (API addition)
#ifdef VEDNN_USE_OPENMP
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

static inline
void krn_i2c_sw1( int64_t const kernel_row, int64_t kernel_col,
    int64_t const pad_h, int64_t const dilation_h, int64_t const stride_h,
    int64_t const pad_w, int64_t const dilation_w, /* int64_t const stride_w = 1, */
    int64_t const output_h, int64_t output_w, //uint64_t output_wB,
    int64_t const height, int64_t const width,
    float const * restrict data_im_channel,
    float       * restrict data_col)
{
  typedef uint64_t u64;
  int64_t outOffset=0;
  int64_t const ir0 = -pad_h + kernel_row * dilation_h;
  for (int64_t output_row = 0; output_row<output_h; ++output_row) {	// outHeight
    int64_t input_row = ir0 + output_row*stride_h;
    if ( (u64)input_row >= (u64)height ) {
      for (int64_t output_col = output_w; output_col; output_col--) { // outWidth
        data_col[outOffset++] = 0.f; //*(data_col++) = 0;
      }
    } else {
      int64_t const ic0 = -pad_w + kernel_col * dilation_w;
      float const* restrict data_im_row = &data_im_channel[input_row * width];
      int64_t ic_sw[256];
#pragma _NEC vreg(ic_sw)
      for(int64_t i=0; i<256; ++i) ic_sw[i] = ic0 + i*1;
      float x[256];
#pragma _NEC vreg(x)
      // unblock loop, generate register explicitly, force single store
      for( int64_t ocB = 0; ocB<output_w; ocB += 256U ){
        int64_t const ocBlen = (output_w-ocB < 256? output_w-ocB: 256);
#pragma _NEC shortloop
        for (int64_t output_col = ocB; output_col<ocB+ocBlen; ++output_col) {	// outWidth
          x[output_col-ocB] = data_im_row[ ic0 + output_col*1];
          if ( (u64)(ic0 + output_col )>= (u64)width )
            x[output_col-ocB] = 0.f; //fzeros[output_col-ocB];
          data_col[outOffset++] = x[output_col-ocB]; // single store
        }
      }//outWidth
#pragma _NEC shortloop
      for(int i=0; i<256; ++i) ic_sw[i] += 256*1;
    }//outWidth Blocking
  }
  DBG(" outOffset=%lu",(u64)outOffset); // output_h * output_w
}

/** data_col size to hold float[ic*kw*kh*ow*oh]. */
  static void
vednn_im2col(const float * restrict data_im, const int64_t channels,
    const int64_t height, const int64_t width, const int64_t kernel_h, const int64_t kernel_w,
    const int64_t pad_h, const int64_t pad_w,
    const int64_t stride_h, const int64_t stride_w,
    const int64_t dilation_h, const int64_t dilation_w,
    float * restrict data_col, int const threads)
{
  typedef uint64_t u64;
  LFTRACE_BEGIN("vednn_im2col");
  const int64_t output_h = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
  const int64_t output_w = (width + 2 * pad_w -  (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
  const int64_t kernel_hw = kernel_h*kernel_w;
  const int64_t channel_size = height * width;
  const int64_t output_hw = output_h * output_w;

  int64_t const workPerChannel = kernel_hw * output_hw;
  //int const collapse_2_loops = kernel_h > 1 && channels%threads && channels < 2*threads;
  //if( collapse_2_loops ) // [channels,kernel_h]
  if( channels < 2*threads && channels%threads && channels*kernel_h > 1 ){
    if(stride_w==1){
      DBG(" omp2str1 ");
      OMP(parallel for collapse(2))//;
      for (int64_t channel = 0 ; channel < channels; ++channel) {       // inChannel
        for (int64_t kernel_row = 0; kernel_row < kernel_h; ++kernel_row) {   // kernHeight
          int64_t const inOffset = channel * channel_size;
          int64_t outOffset = channel * workPerChannel + kernel_row * kernel_w*output_hw;
          for (int64_t kernel_col = 0; kernel_col < kernel_w; kernel_col++) {   // kernWidth
            krn_i2c_sw1( kernel_row, kernel_col,
                pad_h, dilation_h, stride_h, pad_w, dilation_w, /*stride_w,*/
                output_h, output_w, //256U,
                height, width,
                &data_im[inOffset], &data_col[outOffset] );
            outOffset += output_hw;
          } } }
    }else{ //stride_w>1
      DBG(" omp2str>1 ");
      OMP(parallel for collapse(2))//;
      for (int64_t channel = 0 ; channel < channels; ++channel) {       // inChannel
        for (int64_t kernel_row = 0; kernel_row < kernel_h; ++kernel_row) {   // kernHeight
          int64_t i_x_stride_w[256]; // const
#pragma _NEC vreg(i_x_stride_w)
          float fzeros[256];
#pragma _NEC vreg(fzeros)
          for(int i=0; i<256; ++i) i_x_stride_w[i] = i*stride_w;
          for(int i=0; i<256; ++i) fzeros[i] = 0.f;
          int64_t const inOffset = channel * channel_size;
          int64_t outOffset = channel * workPerChannel + kernel_row * kernel_w*output_hw;
          for (int64_t kernel_col = 0; kernel_col < kernel_w; ++kernel_col) {   // kernWidth

            int64_t const ir0 = -pad_h + kernel_row * dilation_h;
            for (int64_t output_row = 0; output_row < output_h; ++output_row) {  // outHeight
              int64_t input_row = ir0 + output_row*stride_h;
              if ( (u64)input_row >= (u64)height ) {
                for (int64_t oc = output_w; oc; --oc) { // outWidth
                  data_col[outOffset++] = FZERO;
                }
              } else {
                float const* restrict data_im_row = &data_im[inOffset + input_row * width];
                int64_t const ic0 = -pad_w + kernel_col * dilation_w;
                int64_t ic_sw[256];
#pragma _NEC vreg(ic_sw)
                for(int i=0; i<256; ++i) ic_sw[i] = ic0 + i_x_stride_w[i];
                for(int64_t ocB = 0; ocB<output_w; ocB += 256U ){
                  int64_t const ocBlen = (output_w-ocB < 256? output_w-ocB: 256);
#pragma _NEC shortloop
                  for (int64_t output_col = ocB; output_col<ocB+ocBlen; ++output_col) {	// outWidth
                    float x[256];
#pragma _NEC vreg(x)
                    x[output_col-ocB] = data_im_row[ ic0 + output_col * stride_w ];
                    if ( (u64)ic_sw[output_col-ocB] >= (u64)width )
                      x[output_col-ocB] = fzeros[output_col-ocB];
                    data_col[outOffset++] = x[output_col-ocB];
                  }//outWidth, vector
#pragma _NEC shortloop
                  for(int i=0; i<256; ++i) ic_sw[i] += 256*stride_w;
                }//outWidth, blocking
              }
            }//outHeight
          }
        }
      }
    }
  }else{ // collapse just one outer [channels] loop
    if(stride_w==1){
      DBG(" omp1str1 ");
      OMP(parallel for)//;
      for (int64_t channel = 0 ; channel < channels; ++channel) {       // inChannel
        for (int64_t kernel_row = 0; kernel_row < kernel_h; ++kernel_row) {   // kernHeight
          int64_t const inOffset = channel * channel_size;
          int64_t outOffset = channel * workPerChannel + kernel_row * kernel_w*output_hw;
          for (int64_t kernel_col = 0; kernel_col < kernel_w; kernel_col++) {   // kernWidth
            krn_i2c_sw1( kernel_row, kernel_col,
                pad_h, dilation_h, stride_h, pad_w, dilation_w, /*stride_w,*/
                output_h, output_w, //256U,
                height, width,
                &data_im[inOffset], &data_col[outOffset] );
            outOffset += output_hw;
          } } }
    }else{ //stride_w>1
      DBG(" omp1str>1 ");
      OMP(parallel for)//;
      for (int64_t channel = 0 ; channel < channels; ++channel) {       // inChannel
        int64_t i_x_stride_w[256]; // const
#pragma _NEC vreg(i_x_stride_w)
        float fzeros[256];
#pragma _NEC vreg(fzeros)
        for(int i=0; i<256; ++i) i_x_stride_w[i] = i*stride_w;
        for(int i=0; i<256; ++i) fzeros[i] = 0.f;
        for (int64_t kernel_row = 0; kernel_row < kernel_h; kernel_row++) {   // kernHeight
          int64_t const inOffset = channel * channel_size;
          int64_t outOffset = channel * workPerChannel + kernel_row * kernel_w*output_hw;
          int64_t const ir0 = -pad_h + kernel_row * dilation_h;
          for (int64_t kernel_col = 0; kernel_col < kernel_w; ++kernel_col) {   // kernWidth

            for (int64_t output_row = 0; output_row < output_h; ++output_row) {  // outHeight
              int64_t input_row = ir0 + output_row*stride_h;
              if ( (u64)input_row >= (u64)height ) {
                for (int64_t oc = output_w; oc; --oc) { // outWidth
                  data_col[outOffset++] = FZERO;
                }
              } else {
                float const* restrict data_im_row = &data_im[inOffset + input_row * width];
                int64_t const ic0 = -pad_w + kernel_col * dilation_w;
                //float const* restrict data_im_row = &data_im[inOffset + ic0 + input_row * width];
                int64_t ic_sw[256];
#pragma _NEC vreg(ic_sw)
                for(int i=0; i<256; ++i) ic_sw[i] = ic0 + i_x_stride_w[i];
                for(int64_t ocB = 0; ocB<output_w; ocB += 256U ){
                  int64_t const ocBlen = (output_w-ocB < 256? output_w-ocB: 256);
#pragma _NEC shortloop
                  for (int64_t output_col = ocB; output_col<ocB+ocBlen; ++output_col) {	// outWidth
                    float x[256];
#pragma _NEC vreg(x)
                    x[output_col-ocB] = data_im_row[ ic0 + output_col * stride_w ];
                    //x[output_col-ocB] = data_im[ inOffset + input_row*width + ic0 + output_col * stride_w ];
                    if ( (u64)ic_sw[output_col-ocB] >= (u64)width )
                      x[output_col-ocB] = fzeros[output_col-ocB];
                    data_col[outOffset++] = x[output_col-ocB];
                  }//outWidth, vector
#pragma _NEC shortloop
                  for(int i=0; i<ocBlen; ++i) ic_sw[i] += 256*stride_w;
                }//outWidth, blocking
              }
            }//outHeight
          }
        }
      }
#if 0
      OMP(parallel for if(channels>1))//;
      for (int64_t channel = 0 ; channel < channels; ++channel) {       // inChannel
        int64_t inOffset = channel * channel_size;
        int64_t outOffset = channel * workPerChannel;
        for (int64_t kernel_row = 0; kernel_row < kernel_h; kernel_row++) {   // kernHeight
          //if(channel==0) DBG("inOffset=%-8lu outOffset=%-8lu\n",(lu)inOffset,(lu)outOffset);
          for (int64_t kernel_col = 0; kernel_col < kernel_w; kernel_col++) {   // kernWidth
            int64_t input_row = -pad_h + kernel_row * dilation_h;
            for (int64_t const input_row_max = input_row + output_h*stride_h;
                input_row < input_row_max; input_row += stride_h){ // outHeight-->inRow
              int64_t       input_col     = -pad_w + kernel_col * dilation_w;
              int64_t const input_col_max = input_col + output_w * stride_w;
              for( ; input_col < input_col_max; input_col+=stride_w) { // outWidth
                data_col[outOffset++]
                  = ( (0 <= input_row && input_row < height) // (ncc reduces 4 conds to 2 vfmk, good)
                      && (0 <= input_col && input_col < width)
                      ? data_im[inOffset + input_row * width + input_col]
                      : FZERO);
              } } } } }
#endif
    }
  }// end collapse one
  LFTRACE_END("vednn_im2col");
}

/** maskless specialization: kh1, a stride != 1.
 * kh1sh1 is a "no in2col" situation. Hmmm. very little speedup? */
  static inline void
vednn_im2col_k1p0_str(const float * restrict data_im, const int64_t channels,
    const int64_t height, const int64_t width,
    /* kernel 1x1  : const int64_t kernel_h, const int64_t kernel_w,*/
    /* pad    == 0 : const int64_t pad_h, const int64_t pad_w,*/
    const int64_t stride_h, const int64_t stride_w,
    /* dilation irrelevant for 1x1 kernel : const int64_t dilation_h, const int64_t dilation_w, */
    float * restrict data_col, int const nThreads)
{
  DBG(" vednn_im2col_k1p0_str ");
  const int64_t output_h = (height - 1) / stride_h + 1;
  const int64_t output_w = (width  - 1) / stride_w + 1;
  const int64_t channel_size = height * width;
  int64_t const output_hw = output_h*output_w; // workPerChannel

  if( channels < 2*nThreads && channels%nThreads && channels*output_h > 1 )
  {
    LFTRACE_BEGIN("vednn_im2col_k1p0_str-collapse(1)");
#pragma omp for collapse(2) // no effect of collapse(2) ?
    for (int64_t channel = 0u ; channel < channels; ++channel) {       // inChannel
      for(int64_t output_row = 0; output_row < output_h; ++output_row){
        for(int64_t output_col = 0; output_col < output_w; output_col += stride_w){
          // input_row = output_row * stride_h; input_col = output_col * stride_w;
          data_col[channel*output_hw  + output_row          * output_w + output_col]
            = data_im[channel*channel_size + output_row*stride_h * width    + output_col*stride_w];
        } } }
    LFTRACE_END("vednn_im2col_k1p0_str-collapse(1)");
  }else{
    LFTRACE_BEGIN("vednn_im2col_k1p0_str-collapse(2)");
#pragma omp parallel for
    for (int64_t channel = 0u ; channel < channels; ++channel) {       // inChannel
      int64_t inOffset = channel * channel_size;
      // __builtin_vprefetch( &data_im[inOffset], output_hw*sizeof(float) );
      int64_t outOffset = channel * output_hw;
      for(int64_t input_row = 0; input_row < output_h*stride_h; input_row += stride_h){
        for(int64_t input_col = 0; input_col < output_w*stride_w; input_col += stride_w){
          data_col[outOffset++] = data_im[inOffset + input_row * width + input_col];
        } } }
    LFTRACE_END("vednn_im2col_k1p0_str-collapse(2)");
  }
}

  static void
vednn_col2im(
    const float* data_col, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    float* data_im, int const threads)
{
  LFTRACE_BEGIN("vednn_col2im");
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
            int input_col = -pad_w + kernel_col * dilation_w; // prevents opt?
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
  LFTRACE_END("vednn_col2im");
}

extern "C" { //}
  static vednnError_t
vednnConvFwd_gemm(
    const vednnTensorParam_t * restrict pParamIn, const void * restrict pDataIn,
    const vednnFilterParam_t * restrict pParamKernel, const void * restrict pDataKernel,
    const vednnBiasParam_t * restrict pParamBias, const void * restrict pDataBias,
    const vednnTensorParam_t * restrict pParamOut, void * restrict pDataOut,
    const float * restrict pOne,  float * restrict pColBuf,
    const vednnConvolutionParam_t * restrict pParamConv )
{
  char const* ftrace_impl;
#if VEDNN_USE_OPENMP
  // if not in nested omp, sgemm internally threaded --> "_T" suffix,
  //                       else assume wrapper has usual minibatch threading
  // assume omp_set_dynamic(1) [default] so nested omp ||ism does not happen
  ftrace_impl = (omp_in_parallel()? "vednnConvFwd_gemm_mb": "vednnConvFwd_gemm");
#else
  ftrace_impl = "vednnConvFwd_gemm"; // sgemm may use threads
#endif
  DBG(" %s pDataIn@%p pDataKernel@%p pDataBias@%p pDataOut@%p pOne@%p pColBuf@%p\n",
      ftrace_impl, (void*)pDataIn, (void*)pDataKernel, (void*)pDataBias,
      (void*)pDataOut, (void*)pOne, (void*)pColBuf);
  LFTRACE_BEGIN(ftrace_impl);
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

  int inChannelGroup  = inChannel  / group;   // pParamKernel->inChannel
  int outChannelGroup = outChannel / group;   // pParamKernel->outChannel

  const float * restrict pIn     = (const float*)pDataIn;
  const float * restrict pBias   = (const float*)pDataBias;
  const float * restrict pKernel = (const float*)pDataKernel;
  float * restrict pOut    = (float *)pDataOut;

  int no_im2col = (kernWidth == 1 && kernHeight == 1 && strideWidth == 1 && strideHeight == 1 && padWidth == 0 && padHeight == 0);
  //chkThreads();

  if (no_im2col) {
    DBG(" noim2col ");
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
        // no im2col ...
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
      } // group
    } // batch
  }else if( kernHeight+kernWidth==2 && padHeight+padWidth==0 && strideHeight+strideWidth>2 ){
    DBG(" k1p0_str ");
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

        int M = outChannelGroup;
        int N = outWidth * outHeight;
        int K = inChannelGroup * kernWidth * kernHeight;

        // k1, padding irrelevant, assume output sz matches input sz,str,dil settings
        //assert( kernHeight == 1 && kernHeight == 1 );
        //assert(  padHeight == 1 &&  padHeight == 1 );
        vednn_im2col_k1p0_str(&pIn[inOffset],
            inChannelGroup, inHeight, inWidth, /*kernHeight, kernWidth,*/
            /*padHeight, padWidth,*/ strideHeight, strideWidth, /*dilationHeight, dilationWidth,*/
            pColBuf, getThreads());

        SGEMM(&NOTRANS, &NOTRANS, &N, &M, &K,
            &FONE, pColBuf, &N,
            (float *)&pKernel[kernGroupOffset], &K,
            &FZERO, &pOut[outOffset], &N);

        if (pBias) {
          SGEMM(&NOTRANS, &NOTRANS, &N, &M, &IONE,
              &FONE, (float *)pOne, &N,
              (float *) &pBias[biasGroupOffset], &IONE,
              &FONE, &pOut[outOffset], &N);
        }
      } // group
    } // batch

    // Hmmm. perhaps separate the collapse(2) case (first layer, ic=3)
    // or even channels*kernel_h too small ==> sequential (or outer threading)
    //}else if( channels < 2*threads && channels%threads && channels*kernel_h > 1 ){
    // ... existing code, above, good for ic=3 input layers

  }else{ // generic im2col
    DBG(" im2col ");
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

        int M = outChannelGroup;
        int N = outWidth * outHeight;
        int K = inChannelGroup * kernWidth * kernHeight;

        vednn_im2col(&pIn[inOffset],
            inChannelGroup, inHeight, inWidth, kernHeight, kernWidth,
            padHeight, padWidth, strideHeight, strideWidth, dilationHeight, dilationWidth,
            pColBuf, getThreads());

        SGEMM(&NOTRANS, &NOTRANS, &N, &M, &K,
            &FONE, pColBuf, &N,
            (float *)&pKernel[kernGroupOffset], &K,
            &FZERO, &pOut[outOffset], &N);

        if (pBias) {
          SGEMM(&NOTRANS, &NOTRANS, &N, &M, &IONE,
              &FONE, (float *)pOne, &N,
              (float *) &pBias[biasGroupOffset], &IONE,
              &FONE, &pOut[outOffset], &N);
        }
      } // group
    } // batch
  }
  LFTRACE_END(ftrace_impl);
  return VEDNN_SUCCESS;
}

  static vednnError_t
vednnConvBkD_Gemm(
    const vednnTensorParam_t * restrict pParamGradOut, const void * restrict pDataGradOut,
    const vednnFilterParam_t * restrict pParamKernel, const void * restrict pDataKernel,
    const vednnTensorParam_t * restrict pParamGradIn, void * restrict pDataGradIn,
    float * restrict pColBuf,
    const vednnConvolutionParam_t * restrict pParamConv )
{
  char const* ftrace_impl;
#if VEDNN_USE_OPENMP
  ftrace_impl = (omp_in_parallel()? "vednnConvBkD_gemm_mb": "vednnConvBkD_gemm");
#else
  ftrace_impl = "vednnConvBkD_gemm"; // sgemm may use threads
#endif
  LFTRACE_BEGIN(ftrace_impl);
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
  //chkThreads();

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
            &FZERO, pColBuf, &N);

        vednn_col2im(pColBuf,
            gInChannelGroup, gInHeight, gInWidth, kernHeight, kernWidth,
            padHeight, padWidth, strideHeight, strideWidth, dilationHeight, dilationWidth,
            &pGradIn[gInOffset], getThreads());
      }
    } // group
  } // batch

  LFTRACE_END(ftrace_impl);
  return VEDNN_SUCCESS;
}

  vednnError_t
vednnConvBkF_gemm(
    const vednnTensorParam_t * restrict pParamIn, const void * restrict pDataIn,
    const vednnTensorParam_t * restrict pParamGradOut, const void * restrict pDataGradOut,
    const vednnFilterParam_t * restrict pParamGradKernel, void * restrict pDataGradKernel,
    float * restrict pColBuf,
    const vednnConvolutionParam_t * restrict pParamConv )
{
  char const* ftrace_impl;
#if VEDNN_USE_OPENMP
  ftrace_impl = (omp_in_parallel()? "vednnConvBkF_gemm_mb": "vednnConvBkF_gemm");
#else
  ftrace_impl = "vednnConvBkF_gemm"; // sgemm may use threads
#endif
  LFTRACE_BEGIN(ftrace_impl);
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

  int inChannelGroup  = inChannel  / group;   // pParamKernel->inChannel
  int outChannelGroup = outChannel / group;   // pParamKernel->outChannel

  const float * restrict pIn     = (float const*)pDataIn;
  const float * restrict pOut    = (float const*)pDataGradOut;
  float * restrict pKernel       = (float*)pDataGradKernel ;

  int no_im2col = (kernWidth == 1 && kernHeight == 1 && strideWidth == 1 && strideHeight == 1 && padWidth == 0 && padHeight == 0);
  //chkThreads();

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
        vednn_im2col(&pIn[inOffset],
            inChannelGroup, inHeight, inWidth, kernHeight, kernWidth,
            padHeight, padWidth, strideHeight, strideWidth, dilationHeight, dilationWidth,
            pColBuf, getThreads());

        int M = outChannelGroup;
        int N = inChannelGroup * kernWidth * kernHeight;
        int K = outWidth * outHeight;

        SGEMM(&TRANS, &NOTRANS, &N, &M, &K,
            &FONE,  pColBuf, &K,
            (float*)&pOut[outOffset], &K,
            &FONE, &pKernel[kernGroupOffset], &N);
      }
    } // group
  } // batch

  LFTRACE_END(ftrace_impl);
  return VEDNN_SUCCESS;
}

// Now ALL forward convolutions include bias, which can be nullptr
// 2 versions:
//              direct_gemm with internal blas/sgemm threading
//              direct_gemm_mb with minibatch-threading
vednnError_t vednnConvolutionForward_direct_gemm(VEDNN_CONVFWD_ARGS)
    //const vednnTensorParam_t * restrict         pParamIn,
    //const void * restrict                       pDataIn,
    //const vednnFilterParam_t * restrict         pParamKernel,
    //const void * restrict                       pDataKernel,
    //const vednnBiasParam_t * restrict           pParamBias,
    //const void * restrict                       pDataBias,
    //const vednnConvolutionParam_t * restrict    pParamConv,
    //const vednnTensorParam_t * restrict         pParamOut,
    //void * restrict                             pDataOut
{
  DBG(" vednnConvolutionForward_direct_gemm ");
  DBG(" pDataIn@%p pDataKernel@%p pParamConv@%p pParamOut@%p pDataOut@%p\n",
      (void*)pDataIn, (void*)pDataKernel, (void*)pParamConv, (void*)pParamOut, (void*)pDataOut);
  size_t pColrows = pParamKernel->inChannel * pParamKernel->width * pParamKernel->height;
  size_t pColcols = pParamOut->width * pParamOut->height; // same for all omp threads

  // XXX scratch size set as if NO caller minibatch-threading (thread inside sgemm)
  // This buffer is only used for bias term
  // This is a const buffer, so a global pointer is OK.
  // BUT if called via wrapper, only 'master' thread should set its size
  //      ( ?? size lower per thread ? )
  float * pOnes_;
  {
#if SCRATCHPAD==0
    pOnes_ = (float *) malloc(pColcols * sizeof(float)/*bytes*/);
    for(int i = 0; i < pColcols; ++i)
      pOnes_[i] = 1.0f;
#else
    using vednn::scratchpad::vednn_scratchpad_float_ones;
    pOnes_ = vednn_scratchpad_float_ones(pColcols/*floats*/); // ow * oh of 1.0f
    DBG(" pOnes_=%p\n",(void*)pOnes_);
#endif
  }
  float const* restrict const pOnes = (float const*)pOnes_;

  // This buffer is used for a monolithic im2col
  // (could use a smaller buffer if im2col were done "as-needed")
  // (also, internal-thread version might use smaller buffer?)
  size_t const nBytes = pColrows * pColcols * getTensorDataSize(pParamIn);
  float * restrict pColBuf = getScratch(nBytes);
  if(!pColBuf) return VEDNN_ERROR_MEMORY_EXHAUST;

  chkThreads();
  vednnError_t ret = vednnConvFwd_gemm(
      pParamIn, pDataIn,
      pParamKernel, pDataKernel,
      pParamBias, pDataBias,
      pParamOut, pDataOut/*output*/,
      pOnes, pColBuf, pParamConv );
  freeScratch((void*)pOnes);
  freeScratch(pColBuf);
  return ret;
}

/** Supply gemm pFunc to standard minibatch wrapper.
 * \ref vednnConvolutionForward.h minibatch threads twiddle pData/pParam
 * per minibatch thread, invoking the direct_gemm version.
 *
 * - \b with minibatch threads, sgemm/blas calls are single-threaded.
 * - \b without, directly calling direct_gemm will do a single large gemm,
 *   allowing sgemm/blas to use internal threads.
 *
 * \note a third option has "_T" kernels that thread via both \c mb and \c g.
 */
vednnError_t vednnConvolutionForward_direct_gemm_mb(VEDNN_CONVFWD_ARGS)
    //const vednnTensorParam_t * restrict         pParamIn,
    //const void * restrict                       pDataIn,
    //const vednnFilterParam_t * restrict         pParamKernel,
    //const void * restrict                       pDataKernel,
    //const vednnBiasParam_t * restrict           pParamBias,
    //const void * restrict                       pDataBias,
    //const vednnConvolutionParam_t * restrict    pParamConv,
    //const vednnTensorParam_t * restrict         pParamOut,
    //void * restrict                             pDataOut
{
    // standard libvednn minibatch threading, 
    WRAP_RET(vednnConvolutionForward_direct_gemm,
            vednnConvolutionForward_mb_threads,
            VEDNN_CONVFWD_ARGS_LIST);
}

vednnError_t
vednnConvolutionBackwardFilter_direct_gemm(
    const vednnTensorParam_t * restrict         pParamIn,
    const void * restrict                       pDataIn,
    const vednnTensorParam_t * restrict         pParamGradOut,
    const void * restrict                       pDataGradOut,
    const vednnConvolutionParam_t * restrict    pParamConv,
    const vednnFilterParam_t * restrict         pParamGradKernel,
    void * restrict                             pDataGradKernel
    // *** NOTE: direct_gemm does not go via standard openmp wrappers!
    //#ifdef VEDNN_USE_OPENMP
    //    ,
    //    const int64_t                               beginOChannel,
    //    const int64_t                               nOChannel,
    //    const int64_t                               beginGroup,
    //    const int64_t                               nGroup
    //#endif
    ){
  size_t const pColrows = pParamGradKernel->inChannel * pParamGradKernel->width * pParamGradKernel->height;
  size_t const pColcols = pParamGradOut->width * pParamGradOut->height;
  size_t const nBytes = pColrows * pColcols * getTensorDataSize(pParamIn);
  float * restrict pColBuf = getScratch(nBytes);
  if(!pColBuf) return VEDNN_ERROR_MEMORY_EXHAUST;

  vednnError_t ret = vednnConvBkF_gemm(
      pParamIn, pDataIn,
      pParamGradOut, pDataGradOut,
      pParamGradKernel, pDataGradKernel/*output*/,
      pColBuf, pParamConv );
  freeScratch(pColBuf);
  return ret;
}

  vednnError_t
vednnConvolutionBackwardData_direct_gemm(
    const vednnTensorParam_t * restrict         pParamGradOut,
    const void * restrict                       pDataGradOut,
    const vednnFilterParam_t * restrict         pParamKernel,
    const void * restrict                       pDataKernel,
    const vednnConvolutionParam_t * restrict    pParamConv,
    const vednnTensorParam_t * restrict         pParamGradIn,
    void * restrict                             pDataGradIn
    )
{
  size_t pColrows = pParamKernel->inChannel * pParamKernel->width * pParamKernel->height;
  size_t pColcols = pParamGradOut->width * pParamGradOut->height;
  size_t const nBytes = pColrows * pColcols * getTensorDataSize(pParamGradIn);
  float * restrict pColBuf = getScratch(nBytes);
  if(!pColBuf) return VEDNN_ERROR_MEMORY_EXHAUST;

  vednnError_t ret =  vednnConvBkD_Gemm(
      pParamGradOut, pDataGradOut,
      pParamKernel, pDataKernel,
      pParamGradIn, pDataGradIn/*output*/,
      pColBuf, pParamConv );
  freeScratch(pColBuf);
  return ret;
}



}//extern "C"
// vim: et ts=2 sw=2 cindent cino=^0,=0,l0,g2,\:0,N-s syntax=cpp.doxygen
