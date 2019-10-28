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

  typedef long unsigned lu;

  //int const vlen=256;
  //float zeros[256];
  //for(int i=0; i<vlen; ++i) zeros[i] = 0.f;

  //printf(" im2col channels %lu threads %lu vednn-threads %lu omp_threads %lu\n",(lu)channels,
  //    (lu)threads, (lu)__vednn_omp_num_threads, (lu)omp_get_max_threads());
  int64_t const workPerChannel = kernel_h*kernel_w * output_h*output_w;
  int collapse_2_loops = kernel_h > 1 && channels%threads && channels < 2*threads;
  // XXX check if low workPerChannel make us want to avoid collapse(2) or even omp

  if( collapse_2_loops ) // [channels,kernel_h]
  {
    int64_t channel, kernel_row;
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
//#define PFCH_MAX (1UL<<14)
      //size_t const prefetch_bytes = (height*width*sizeof(float)<PFCH_MAX? height*width*sizeof(float): PFCH_MAX);
      //size_t const prefetch_row   = (width*sizeof(float)<PFCH_MAX? width*sizeof(float): PFCH_MAX);
#pragma omp for private(channel,kernel_row) collapse(2)
      for (channel = 0 ; channel < channels; ++channel) {       // inChannel
        for (kernel_row = 0; kernel_row < kernel_h; ++kernel_row) {   // kernHeight
          int64_t const inOffset = channel * channel_size;
          int64_t outOffset = channel * workPerChannel + kernel_row * kernel_w*output_h*output_w;

          int64_t kernel_col, output_row, output_col;
          for (kernel_col = 0; kernel_col < kernel_w; kernel_col++) {   // kernWidth
            int64_t input_row = -pad_h + kernel_row * dilation_h;

            for (output_row = output_h; output_row; output_row--) {  // outHeight
              if (input_row < 0 || input_row >= height)
              {
                for (output_col = output_w; output_col; output_col--) { // outWidth
                  data_col[outOffset++] = FZERO;
                }
              } else {
                int64_t input_col = -pad_w + kernel_col * dilation_w;
                for (output_col = output_w; output_col; output_col--) { // outWidth
                  data_col[outOffset++]
                    = (0 <= input_col && input_col < width
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
  }else{ // ! collapse_2_loops, enough channels to spread the work
    int64_t channel;
    // input_col = -pad_w + kernel_col*dilation_w  + [0..output_w-1]*stride_w
    //   (*) iw(ow) = -PW + KW*DW + ow*SW
    // with all-caps constants positive, ow=0..output_w-1
    // 1) at what ow_iwGE0 does iw(ow) >= 0?
    //   iw(0) = -PW + KW*DW + 0*SW
    //   If iw(0) >= 0, ow_iwGE0 = 0
    //   else [iw(0)<0] ow_iwGE0 = divup(-iw(0)/SW) = (-iw(0)+SW-1)/SW
    //   [ combine: ow_iwGE0 = (-max(0,-PW+KW*DW) + SW - 1) / SW ] (normal division)
    //   [ alt: ow_iwGE0 = idiv(0+PW-KH*DW, SW) which may be -ve
    // 2) at what ow_iwGEIW does iw(ow) >= IW
    //   Solving (*) for ow:
    //      ow(iw) = idiv(iw+PW-KH*DW, SW),  where idiv is round-neg-inf division
    //   So ow(IW) = idiv(IW+PW-KH*DW, SW) = ow_iwGEIW
    // 3) so masking as ow increases has inclusive regions:
    //   a) masking from 0 to ow_GE0-1
    //   b) no mask from ow_GE0-1 to ow_iwGEIW-1
    //   c) masking from ow_iwLTIW to output_w,
    //   where regions with upper limit < lower limit are empty.
#pragma omp parallel
    {
#pragma omp for private(channel)
#if 0 // original
      for (channel = 0 ; channel < channels; ++channel) {       // inChannel
        int64_t inOffset = channel * channel_size;
        int64_t outOffset = channel * workPerChannel;

        int64_t kernel_row, kernel_col, output_rows, output_cols, output_col;
        for (kernel_row = 0; kernel_row < kernel_h; kernel_row++) {   // kernHeight
          //if(channel==0) printf("inOffset=%-8lu outOffset=%-8lu\n",(lu)inOffset,(lu)outOffset);
          for (kernel_col = 0; kernel_col < kernel_w; kernel_col++) {   // kernWidth
            int64_t input_row = -pad_h + kernel_row * dilation_h;
            for (output_rows = output_h; output_rows; output_rows--)
            {  // outHeight
              if (input_row < 0 || input_row >= height)//(!zero_le_a_lt_b(input_row,height))
              {
                for (output_cols = output_w; output_cols; output_cols--) { // outWidth
                  data_col[outOffset++] = FZERO; //*(data_col++) = 0;
                }
              } else {
                int64_t input_col = -pad_w + kernel_col * dilation_w;
                for (output_col = output_w; output_col; output_col--) { // outWidth
                  // would be better with vmrg and SINGLE vstu, but could not convince compiler
                  data_col[outOffset++] //*(data_col++)
                    = (0 <= input_col && input_col < width
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
#elif 1 // shorter, same asm
      for (channel = 0 ; channel < channels; ++channel) {       // inChannel
        int64_t inOffset = channel * channel_size;
        //__builtin_vprefetch(&data_im[inOffset], prefetch_bytes);
        int64_t outOffset = channel * workPerChannel;

        int64_t kernel_row, kernel_col, output_rows, output_col;
        for (kernel_row = 0; kernel_row < kernel_h; kernel_row++) {   // kernHeight
          //if(channel==0) printf("inOffset=%-8lu outOffset=%-8lu\n",(lu)inOffset,(lu)outOffset);
          for (kernel_col = 0; kernel_col < kernel_w; kernel_col++) {   // kernWidth
            int64_t input_row = -pad_h + kernel_row * dilation_h;
            for (output_rows = output_h; output_rows; output_rows--) {  // outHeight
              int64_t input_col = -pad_w + kernel_col * dilation_w;
              for (output_col = output_w; output_col; output_col--) { // outWidth
                data_col[outOffset++]
                  = ( (0 <= input_row && input_row < height) // (ncc reduces 4 conds to 2 vfmk, good)
                      && (0 <= input_col && input_col < width)
                      ? data_im[inOffset + input_row * width + input_col]
                      : FZERO);
                input_col += stride_w;
              }
              input_row += stride_h;
            }
          }
        }
      }
#elif 0
      for (channel = 0 ; channel < channels; ++channel) {       // inChannel
        int64_t kernel_row, kernel_col, output_rows, output_cols, output_col;

        int64_t inOffset = channel * channel_size;
        //__builtin_vprefetch(&data_im[inOffset], prefetch_bytes);
        int64_t outOffset = channel * workPerChannel;

        for (kernel_row = 0; kernel_row < kernel_h; kernel_row++) {   // kernHeight
          //if(channel==0) printf("inOffset=%-8lu outOffset=%-8lu\n",(lu)inOffset,(lu)outOffset);
          for (kernel_col = 0; kernel_col < kernel_w; kernel_col++) {   // kernWidth
            int64_t input_row = -pad_h + kernel_row * dilation_h;
            for (output_rows = output_h; output_rows; output_rows--)
#if 0
            {  // outHeight
              if (input_row < 0 || input_row >= height)//(!zero_le_a_lt_b(input_row,height))
              {
                for (output_cols = output_w; output_cols; output_cols--) { // outWidth
                  data_col[outOffset++] = FZERO; //*(data_col++) = 0;
                }
              } else {
                int64_t input_col = -pad_w + kernel_col * dilation_w;
                for (output_col = output_w; output_col; output_col--) { // outWidth
                  data_col[outOffset++] //*(data_col++)
                    = (0 <= input_col && input_col < width
                        ? data_im[inOffset + input_row * width + input_col]
                        : FZERO);
                  input_col += stride_w;
                }
              }
              input_row += stride_h;
            }
#else // ncc now does well with:
            {  // outHeight
#if 0
              int64_t input_col = -pad_w + kernel_col * dilation_w;
              for (output_col = output_w; output_col; output_col--) { // outWidth
                data_col[outOffset++] //*(data_col++)
                  = //(zero_le_a_lt_b(input_col, width)
                  ( (0 <= input_row && input_row < width)
                    && (0 <= input_col && input_col < width)
                    ? data_im[inOffset + input_row * width + input_col]
                    : FZERO);
                input_col += stride_w;
              }
#elif 0
              int64_t input_col = -pad_w + kernel_col * dilation_w;
              for (output_col = 0; output_col<output_w; ++output_col) { // outWidth
                data_col[outOffset++] //*(data_col++)
                  = //(zero_le_a_lt_b(input_col, width)
                  ( (0 <= input_row && input_row < width)
                    && (0 <= input_col+output_col*stride_w && input_col+output_col*stride_w < width)
                    ? data_im[inOffset + input_row * width + input_col+output_col*stride_w]
                    : FZERO);
              }
#elif 0 // also same speed
              int64_t input_col = -pad_w + kernel_col * dilation_w;
              int64_t const input_col_max = input_col + output_w * stride_w;
              for (output_col = input_col; output_col<input_col_max; output_col+=stride_w) { // outWidth
                data_col[outOffset++] //*(data_col++)
                  = ( (0 <= input_row && input_row < width)
                      && (0 <= output_col && output_col < width)
                      ? data_im[inOffset + input_row * width + output_col]
                      : FZERO);
              }
#else
              int64_t input_col = -pad_w + kernel_col * dilation_w;
              int64_t const input_col_max = input_col + output_w * stride_w;
              for (output_col = input_col; output_col<input_col_max; output_col+=stride_w) { // outWidth
                data_col[outOffset++] //*(data_col++)
                  = ( (0 <= input_row && input_row < width)
                      && (0 <= output_col && output_col < width)
                      ? data_im[inOffset + input_row * width + output_col]
                      : FZERO);
              }
#endif
              input_row += stride_h;
            }
#endif
          }
        }
      }
#elif 0 //---------------- cleaned up ---------------------
      for (channel = 0 ; channel < channels; ++channel) {       // inChannel
        int64_t kernel_row, kernel_col, output_rows, output_col;
        int64_t inOffset = channel * channel_size;
        int64_t outOffset = channel * workPerChannel;
        for (kernel_row = 0; kernel_row < kernel_h; kernel_row++) {   // kernHeight
#if 0
          for (kernel_col = 0; kernel_col < kernel_w; kernel_col++) {   // kernWidth
            int64_t input_row = -pad_h + kernel_row * dilation_h;
            for (output_rows = output_h; output_rows; output_rows--) {  // outHeight
              int64_t input_col = -pad_w + kernel_col * dilation_w;
              int64_t const input_col_max = input_col + output_w * stride_w;
              for (output_col = input_col; output_col<input_col_max; output_col+=stride_w) { // outWidth
                data_col[outOffset++] //*(data_col++)
                  = ( (0 <= input_row && input_row < width)
                      && (0 <= output_col && output_col < width)
                      ? data_im[inOffset + input_row * width + output_col]
                      : FZERO);
              }
              input_row += stride_h;
            }
          }
#elif 0
          for (kernel_col = 0; kernel_col < kernel_w; kernel_col++) {   // kernWidth
            int64_t input_row = -pad_h + kernel_row * dilation_h;
            for (output_rows = 0; output_rows<output_h; output_rows++) {  // outHeight
              int64_t input_col = -pad_w + kernel_col * dilation_w;
              int64_t const input_col_max = input_col + output_w * stride_w;
              for (output_col = input_col; output_col<input_col_max; output_col+=stride_w) { // outWidth
                data_col[outOffset++] //*(data_col++)
                  = ( (0 <= input_row+output_rows*stride_h && input_row+output_rows*stride_h < width)
                      && (0 <= output_col && output_col < width)
                      ? data_im[inOffset + (input_row+output_rows*stride_h) * width + output_col]
                      : FZERO);
              }
              //input_row += stride_h;
            }
          }
#else
          int64_t input_row, input_col;
          for (kernel_col = 0; kernel_col < kernel_w; kernel_col++) {   // kernWidth
            int64_t const input_row_beg = -pad_h + kernel_row * dilation_h;
            int64_t const input_row_max = input_row_beg + output_h * stride_h;
            for(input_row = input_row_beg; input_row < input_row_max; input_row+=stride_h) {  // outHeight-->input_row
              int64_t const input_col_beg = -pad_w + kernel_col * dilation_w;
              int64_t const input_col_max = input_col_beg + output_w * stride_w;
              for (input_col = input_col_beg; input_col<input_col_max; input_col+=stride_w) { // outWidth-->input_col
                data_col[outOffset++] //*(data_col++)
                  = ( (0 <= input_row && input_row < width)
                      && (0 <= input_col && input_col < width)
                      ? data_im[inOffset + input_row * width + input_col]
                      : FZERO);
              }
            }
          }
#endif
        }
      }
#elif 0 //---------------- cleaned up --------------------- equiv asm (same speed)
      for (channel = 0 ; channel < channels; ++channel) {       // inChannel
        int64_t kernel_row, kernel_col, output_rows, output_col;
        int64_t input_col_beg, input_row_beg;
        int64_t outOffset = channel * workPerChannel;
        int64_t const inOffset = channel * channel_size;
        int64_t const input_row_beg_kr0 = -pad_h;
        int64_t const input_row_beg_max = kernel_h * dilation_h - pad_h;
        //for (kernel_row = 0; kernel_row < kernel_h; ++kernel_row) {   // kernHeight
        for (input_row_beg = input_row_beg_kr0; input_row_beg < input_row_beg_max; input_row_beg+=dilation_h) {   // kernHeight-->kernel_row-->input_row_beg
          int64_t input_row, input_col;
          //int64_t const input_row_beg = -pad_h + kernel_row * dilation_h;
          int64_t const input_row_max = input_row_beg + output_h * stride_h;
          int64_t const input_col_beg_kc0 = -pad_w;
          int64_t const input_col_beg_max = kernel_w * dilation_w - pad_w;
          for (input_col_beg = input_col_beg_kc0; input_col_beg < input_col_beg_max; input_col_beg += dilation_w) {   // kernWidth-->kernel_col-->input_col_beg
            int64_t const input_col_max = input_col_beg + output_w * stride_w;
            for(input_row = input_row_beg; input_row < input_row_max; input_row+=stride_h) {  // outHeight-->input_row
              for (input_col = input_col_beg; input_col<input_col_max; input_col+=stride_w) { // outWidth-->input_col
                data_col[outOffset++] //*(data_col++)
                  = ( (0 <= input_row && input_row < width)
                      && (0 <= input_col && input_col < width)
                      ? data_im[inOffset + input_row * width + input_col]
                      : FZERO);
      } } } } }
#elif 0 //---------------- cleaned up --------------------- equiv asm (same speed)
      for (channel = 0 ; channel < channels; ++channel) {       // inChannel
        int64_t input_row_beg, input_row, input_col_beg, input_col;
        int64_t outOffset = channel * workPerChannel;
        int64_t const inOffset = channel * channel_size;
        int64_t const input_row_beg_max = kernel_h * dilation_h - pad_h;
        for (input_row_beg = -pad_h; input_row_beg < input_row_beg_max; input_row_beg+=dilation_h) {   // kernHeight-->kernel_row-->input_row_beg
          int64_t const input_row_max = input_row_beg + output_h * stride_h;
          int64_t const input_col_beg_max = kernel_w * dilation_w - pad_w;
          for (input_col_beg = -pad_w; input_col_beg < input_col_beg_max; input_col_beg += dilation_w) {   // kernWidth-->kernel_col-->input_col_beg
            int64_t const input_col_max = input_col_beg + output_w * stride_w;
            for(input_row = input_row_beg; input_row < input_row_max; input_row+=stride_h) {  // outHeight-->input_row
              for (input_col = input_col_beg; input_col<input_col_max; input_col+=stride_w) { // outWidth-->input_col
                // ncc tries 2 vstu rather than vmrg and a single write !!!
                data_col[outOffset++] //*(data_col++)
                  = ( (0 <= input_row && input_row < width)
                      && (0 <= input_col && input_col < width)
                      ? data_im[inOffset + input_row * width + input_col]
                      : FZERO);
      } } } } }
#else //---------------- cleaned up --------------------- equiv asm (same speed)
      for (channel = 0 ; channel < channels; ++channel) {       // inChannel
        int64_t input_row_beg, input_row, input_col_beg, input_col;
        int64_t outOffset = channel * workPerChannel;
        int64_t const inOffset = channel * channel_size;
        int64_t const input_row_beg_max = kernel_h * dilation_h - pad_h;
        for (input_row_beg = -pad_h; input_row_beg < input_row_beg_max; input_row_beg+=dilation_h) {   // kernHeight-->kernel_row-->input_row_beg
          int64_t const input_row_max = input_row_beg + output_h * stride_h;
          int64_t const input_col_beg_max = kernel_w * dilation_w - pad_w;
          for (input_col_beg = -pad_w; input_col_beg < input_col_beg_max; input_col_beg += dilation_w) {   // kernWidth-->kernel_col-->input_col_beg
            int64_t const input_col_max = input_col_beg + output_w * stride_w;
            for(input_row = input_row_beg; input_row < input_row_max; input_row+=stride_h) {  // outHeight-->input_row
#if 0 // still cannot produce a VMRG op
              if(output_w < vlen){
                for (input_col = input_col_beg; input_col<input_col_max; input_col+=stride_w) { // outWidth-->input_col
                  float *pzero = &zeros[0];
                  data_col[outOffset++] //*(data_col++)
                    = ( (0 <= input_row && input_row < width)
                        && (0 <= input_col && input_col < width)
                        ? data_im[inOffset + input_row * width + input_col]
                        : *pzero++);
                }
              }else
#endif
              {
                for (input_col = input_col_beg; input_col<input_col_max; input_col+=stride_w) { // outWidth-->input_col
                  // ncc tries 2 vstu rather than vmrg and a single write !!!
                  data_col[outOffset++] //*(data_col++)
                    = ( (0 <= input_row && input_row < width)
                        && (0 <= input_col && input_col < width)
                        ? data_im[inOffset + input_row * width + input_col]
                        : FZERO);
                }
              }
      } } } }
#endif
    }
  }
  LFTRACE_END("im2col_cpu");
}
  static void
im2col_seq(const float * restrict data_im, const int64_t channels,
    const int64_t height, const int64_t width, const int64_t kernel_h, const int64_t kernel_w,
    const int64_t pad_h, const int64_t pad_w,
    const int64_t stride_h, const int64_t stride_w,
    const int64_t dilation_h, const int64_t dilation_w,
    float * restrict data_col)
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
  if(0) // channels%threads && channels < 2*threads )
  { // collapse 2 loops [channels,kernel_h] to get more work per thread.
  }else{ // collapse just one outer [channels] loop
//#pragma omp parallel
    {
//#pragma omp for private(channel)
      for (channel = 0 ; channel < channels; ++channel) {       // inChannel
        int64_t kernel_col, output_rows, output_cols, output_col;

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

  const float * restrict const pIn     = pDataIn;
  const float * restrict const pBias   = pDataBias;
  const float * restrict const pKernel = pDataKernel;
  float * restrict const pOut    = pDataOut;

  int no_im2col = (kernWidth == 1 && kernHeight == 1 && strideWidth == 1 && strideHeight == 1 && padWidth == 0 && padHeight == 0);
  chkThreads();
#if 1
  //int const outer_threading = 0;
  int const inner_threads = getThreads();
#else // XXX outer_threading requires thread-local scratchpad memories XXX
  // pColBuff should be *** thread_local *** in this case !!!
  int outer_tasks = batch * group;
  int const threads = getThreads();
  //int const outer_threading = (outer_tasks%threads==0 || outer_tasks > 3*threads);
  int const outer_threading = 1;
  int const inner_threads = (outer_threading? 1: threads);
#endif
//#pragma omp parallel if(outer_threading)
  {
//#pragma omp for private(n,g) collapse(2)
    for (n = 0; n < batch; n++) {
      for (g = 0; g < group; g++) {
        int const inBatchOffset  = n * inChannel  * inWidth  * inHeight;
        int const outBatchOffset = n * outChannel * outWidth * outHeight;

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

          if(inner_threads > 1){
            im2col_cpu(&pIn[inOffset],
                inChannelGroup, inHeight, inWidth, kernHeight, kernWidth,
                padHeight, padWidth, strideHeight, strideWidth, dilationHeight, dilationWidth,
                pColBuff, inner_threads);
          }else{
            im2col_seq(&pIn[inOffset],
                inChannelGroup, inHeight, inWidth, kernHeight, kernWidth,
                padHeight, padWidth, strideHeight, strideWidth, dilationHeight, dilationWidth,
                pColBuff);
          }


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
  }//omp||

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
    int const inBatchOffset  = n * inChannel  * inWidth  * inHeight;
    int const outBatchOffset = n * outChannel * outWidth * outHeight;

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
