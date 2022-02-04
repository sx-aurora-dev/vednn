/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/** \file
 * jitconv "doRef" calculation - a simplified gemm impl.
 *
 * This should be correct, and may even be somewhat fast "in general".
 * Do not expect "fastest" performance.
 *
 * libvednn has a more sophisticated GEMM convolution.
 */
#include "vednn_helper.h"
#include "convolution_gemm.h"
#include <string.h>
#include <stdlib.h>
#include <cblas.h>

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

// Scratchpad: none -- hardwired to malloc/free im2col buffers
//                     slower, but safer.

#define GEMM_PARA_THRESH 32768

// Around Jan. 2022, ncc-3.4.20, SGEMM began to segfault for M==1,
// so a workaround writes out the trivial matrix multiply.
/// 0 : SGEMM works great
/// 1 : safe macros
/// 2: dev code (generic)
#define SGEMM_M1_SEGFAULTS 1
/// 0 : old code, bias via sgemm (w/ problems)
/// 1 : just use cblas_saxpy alternate (simpler and avoids sgemm bug!)
#define BIAS_SAXPY 0

#if SGEMM_M1_SEGFAULTS==0 // issues ncc-3.4.20 and M==1 ???
#define SGEMM sgemm_
#define SGEMM_A1B0 sgemm_
#define SGEMM_A1B1K1 sgemm_
#define SGEMM_A1B0t sgemm_
#define SGEMM_A1tB1 sgemm_
#else
#define SGEMM sgemm_ // dangerous now - require bug workaround for M=1
#define SGEMM_A1B0   SGEMM_SAFE_A1B0
#define SGEMM_A1B1K1 SGEMM_SAFE_A1B1K1
#define SGEMM_A1B0t SGEMM_SAFE_A1B0t
#define SGEMM_A1tB1 SGEMM_SAFE_A1tB1
#endif

/// A workaround for SGEMM M==1 segfaults. (circa ncc-3.4.20, Jan 2022)
/// For alpha=1, beta=0 (main gemm calculation)
#define SGEMM_SAFE_A1B0(TRANSA,TRANSB, N,M,K, ALPHA,A,LDA, B,LDB, BETA,C,LDC) \
  do { \
    if(*(M) > 1){ \
      sgemm_(TRANSA,TRANSB, N,M,K, ALPHA,A,LDA, B,LDB, BETA,C,LDC); \
    }else{ \
      int const NN = *(N); \
      /*int const MM = *(M);*/ \
      int const KK = *(K); \
      if(*(M) == 1 && *(K) > 1){ /* using just M==1 */ \
        _Pragma("omp parallel if(NN * KK > 32768)") /* C99 */ \
        for (int n=0; n < (NN); ++n) { \
          float acc = 0.0f; \
          for (int k=0; k < (KK); ++k) { \
            acc += (A)[k * (NN) + n] * (B)[k]; \
          } \
          (C)[n] = acc; /* M==1 && beta==0.0 : no accumulation into C */ \
        } \
      }else{ /* M=1, K=1 */ \
        _Pragma("omp parallel if((NN) > 32768)") \
        for (int n=0; n < *(N); ++n) { \
          (C)[n] = (A)[n] * (B)[0]; \
        } \
      } \
    } \
  }while(0)
/// Backward Data also has alpha=1, beta=0, but B[] is transposed
// XXX test with jitconv -T BackwardData
#define SGEMM_SAFE_A1B0t(TRANSA,TRANSB, N,M,K, ALPHA,A,LDA, B,LDB, BETA,C,LDC) \
  do { \
    if(*(M) > 1){ \
      sgemm_(TRANSA,TRANSB, N,M,K, ALPHA,A,LDA, B,LDB, BETA,C,LDC); \
    }else{ \
      /* for M=1, B is K x 1, so vector ignoring the transpose is OK */ \
      int const NN = *(N); \
      /*int const MM = *(M);*/ \
      int const KK = *(K); \
      if(*(M) == 1 && *(K) > 1){ /* using just M==1 */ \
        _Pragma("omp parallel if(NN * KK > 32768)") /* C99 */ \
        for (int n=0; n < (NN); ++n) { \
          float acc = 0.0f; \
          for (int k=0; k < (KK); ++k) { \
            acc += (A)[k * (NN) + n] * (B)[k]; \
          } \
          (C)[n] = acc; /* M==1 && beta==0.0 : no accumulation into C */ \
        } \
      }else{ /* M=1, K=1 */ \
        _Pragma("omp parallel if((NN) > 32768)") \
        for (int n=0; n < *(N); ++n) { \
          (C)[n] = (A)[n] * (B)[0]; \
        } \
      } \
    } \
  }while(0)
/// for BackwardFilter, alpha=1, beta=1 and A is transposed
// XXX test with jitconv -T BackwardFilter
// try only a single omp || ?
// needs testing of ALL impls (update jitconv testBackwardFilter!!!)
#define SGEMM_SAFE_A1tB1_0(TRANSA,TRANSB, N,M,K, ALPHA,A,LDA, B,LDB, BETA,C,LDC) \
  do { \
    /*printf(" A1tB1 N=%d M=%d K=%d\n",*(N),*(M),*(K)); fflush(stdout);*/ \
    if(*(M) > 1){ \
      sgemm_(TRANSA,TRANSB, N,M,K, ALPHA,A,LDA, B,LDB, BETA,C,LDC); \
    }else{ \
      /* for M=1, A is N x K, so need A transpose wrt. Forward impl */ \
      int const NN = *(N); \
      int const KK = *(K); \
      int const NNKK = NN*KK; \
      /* M==1, any K */ \
      _Pragma("omp parallel if(NNKK > 32768)") /* C99 */ \
      for (int n=0; n < (NN); ++n) { \
        float acc = 0.0f; \
        for (int k=0; k < (KK); ++k) { \
          acc += (A)[n * (KK) + k] * (B)[k]; \
        } \
        (C)[n] += acc; /* beta=1 accumulation into C */ \
      } \
    } \
  }while(0)
#define SGEMM_SAFE_A1tB1(TRANSA,TRANSB, N,M,K, ALPHA,A,LDA, B,LDB, BETA,C,LDC) \
  do { \
    /*printf(" A1tB1 N=%d M=%d K=%d\n",*(N),*(M),*(K)); fflush(stdout);*/ \
    if(*(M) > 1){ \
      sgemm_(TRANSA,TRANSB, N,M,K, ALPHA,A,LDA, B,LDB, BETA,C,LDC); \
    }else{ \
      /* for M=1, A is N x K, so need A transpose wrt. Forward impl */ \
      int const NN = *(N); \
      int const KK = *(K); \
      if(*(M) == 1 && *(K) > 1){ /* using just M==1 */ \
        /*_Pragma("omp parallel if(NN * KK > 32768)")*/ /* C99 */ \
        for (int n=0; n < (NN); ++n) { \
          float acc = 0.0f; \
          for (int k=0; k < (KK); ++k) { \
            acc += (A)[n * (KK) + k] * (B)[k]; \
          } \
          (C)[n] += acc; /* beta=1 accumulation into C */ \
        } \
      }else{ /* M=1, K=1 */ \
        /*_Pragma("omp parallel if((NN) > 32768)")*/ \
        for (int n=0; n < (NN); ++n) { \
          (C)[n] += (A)[n] * (B)[0]; /* beta=1 accum */ \
        } \
      } \
    } \
  }while(0)
// using M==1 and K==1
// here A[N] is 1.0
/// workaround for M=1 bias segfault.
/// Here K=1, alpha=1, beta=1, \b and A[] is all-1.0  (bias accumulation)
#define SGEMM_SAFE_A1B1K1(TRANSA,TRANSB, N,M,K, ALPHA,A,LDA, B,LDB, BETA,C,LDC) \
  do { \
    if(1 && *(M) > 1) /* always elide? */ \
    { \
      sgemm_(TRANSA,TRANSB, N,M,K, ALPHA,A,LDA, B,LDB, BETA,C,LDC); \
    }else{ \
      int const MN = *(M) * *(N); \
      float const B_0 = *(B);/* (B)[0] */ \
      /* wrong output if try to parallelize? */ \
      /* _Pragma("omp parallel if(MN > 32768)") */ \
      for (int mn=0; mn < MN; ++mn) { \
        (C)[mn] += B_0; \
      } \
    } \
  }while(0)

#if 0
#define DBG(...) do{printf(__VA_ARGS__);fflush(stdout);}while(0)
#else
#define DBG(...)
#endif

#if 1
void sgemm_(char *TRANSA, char *TRANSB, int *M, int *N, int *K,
    float *ALPHA, float *A,  int *LDA, float *B, int *LDB,
    float *BETA, float *C, int *LDC ) ;
//void cblas_saxpy(const int N, const float alpha, const float *X,
//                 const int incX, float *Y, const int incY);
#endif

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

  static void
#if 0
im2col_cpu(const float * restrict data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int output_h, const int output_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    float * restrict data_col)
#else
im2col_cpu(const float * data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int output_h, const int output_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    float * data_col)
#endif
{
  LFTRACE_BEGIN("im2col_cpu");
#if 0
  const int output_h = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
  const int output_w = (width + 2 * pad_w -  (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
#endif
  const int channel_size = height * width;

  int channel;
  if (0){
    // Note: alloc is for ic * ih * kw, but probably only need (ic/g) * kh * kw here?
    DBG("im2col_cpu _alloc needs %llu floats\n",(long long)channels*channel_size);
    for(size_t i=0; i<(size_t)channels*(size_t)channel_size; ++i){
      data_col[i] = 0.0f;
    }
    DBG("data_col / pColBuf accessible");
  }

//#pragma omp parallel for if(channels>=3)
  for (channel = 0 ; channel < channels; channel++) {       // inChannel
    //printf(" i2c c%d/%d",channel,channels); fflush(stdout);
    int kernel_row, kernel_col, output_rows, output_cols, output_col;

    int inOffset = channel * channel_size;
    int outOffset = channel * output_h * output_w * kernel_h * kernel_w;

    for (kernel_row = 0; kernel_row < kernel_h; kernel_row++) {   // kernHeight
      for (kernel_col = 0; kernel_col < kernel_w; kernel_col++) {   // kernWidth
        int input_row = -pad_h + kernel_row * dilation_h;
        for (output_rows = output_h; output_rows; output_rows--) {  // outHeight
          if (!is_a_ge_zero_and_a_lt_b(input_row, height)) {
            for (output_cols = output_w; output_cols; output_cols--) {
              // *(data_col++) = 0;
              data_col[outOffset++] = 0.f;
            }
          } else {
            int input_col = -pad_w + kernel_col * dilation_w;
            // following still bombed
            //#pragma _NEC novector
            for (output_col = output_w; output_col; output_col--) { // outWidth
#if 1 // newer
              data_col[outOffset++] //*(data_col++)
                = (is_a_ge_zero_and_a_lt_b(input_col, width)
                    ? data_im[inOffset + input_row * width + input_col]
                    : 0.f);
#else // older
              if (outOffset < 0 || outOffset >= channels*kernel_h*kernel_w*output_h*output_w){
                printf("ERROR: outOffset"); fflush(stdout); exit(-1);
              }
              if (is_a_ge_zero_and_a_lt_b(input_col, width)) {
                // *(data_col++) = data_im[input_row * width + input_col];
                data_col[outOffset] = data_im[inOffset + input_row * width + input_col];
              } else {
                // *(data_col++) = 0;
                data_col[outOffset] = 0;
              }
              ++outOffset;
#endif
              input_col += stride_w;
            }
          }
          input_row += stride_h;
        }
      }
    }
  }
  LFTRACE_END("im2col_cpu");
}

static void
col2im_cpu(
    const float* data_col, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int output_h, const int output_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    float* data_im) {

  LFTRACE_BEGIN("col2im_cpu");
  memset(data_im, 0, sizeof(float)*height*width*channels) ;

#if 0
  const int output_h = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
  const int output_w = (width + 2 * pad_w -  (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
#endif

  const int channel_size = height * width;

  int channel;

#pragma omp parallel for if(channels>=3)
  for (channel = 0 ; channel < channels; channel++) {       // inChannel
    int kernel_row, kernel_col, output_rows, output_cols, output_col;

    int inOffset = channel * channel_size;
    int outOffset = channel * output_h * output_w * kernel_h * kernel_w;

    for (kernel_row = 0; kernel_row < kernel_h; kernel_row++) {   // kernHeight
      for (kernel_col = 0; kernel_col < kernel_w; kernel_col++) {   // kernWidth
        int input_row = -pad_h + kernel_row * dilation_h;
        for (output_rows = output_h; output_rows; output_rows--) {  // outHeight
          if (!is_a_ge_zero_and_a_lt_b(input_row, height)) {
            for (output_cols = output_w; output_cols; output_cols--) {
              data_col[outOffset++] ;
            }
          } else {
            int input_col = -pad_w + kernel_col * dilation_w;
            for (output_col = output_w; output_col; output_col--) { // outWidth
              if (is_a_ge_zero_and_a_lt_b(input_col, width)) {
                data_im[inOffset + input_row * width + input_col] += data_col[outOffset++] ;
              } else {
                outOffset++ ;
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
}

// pOne is oh*ow of 1.0f
// pColBuff is scratch of ic*kw*kh * ow*oh * iw*ih (huge, in this version) see conv_test_param.c
  vednnError_t
convolution_forward_gemm(
    const vednnTensorParam_t * restrict pParamIn, const void * restrict pDataIn,
    const vednnFilterParam_t * restrict pParamKernel, const void * restrict pDataKernel,
    const vednnBiasParam_t * restrict pParamBias, const void * restrict pDataBias,
    const vednnTensorParam_t * restrict pParamOut, void * restrict pDataOut,
    //const float * restrict pOne,  float * restrict pColBuff,
    const float * pOne,  float * pColBuff,
    const vednnConvolutionParam_t * restrict pParamConv )
{
  LFTRACE_BEGIN("convolution_forward_gemm");

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

  int inChannelGroup  = inChannel  / group; // pParamKernel->inChannel
  int outChannelGroup = outChannel / group; // pParamKernel->outChannel

  int no_im2col = (kernWidth == 1 && kernHeight == 1 && strideWidth == 1 && strideHeight == 1 && padWidth == 0 && padHeight == 0);
  if (0){
    printf("mb=%d g=%d ic=%d ic/g=%d noi2c=%d", batch,group,inChannel,inChannelGroup, no_im2col);
    fflush(stdout);
    float const* pk = (float const*)(pDataKernel); // assume float (debug) XXX
    int const ksz = group * outChannelGroup * inChannelGroup * kernHeight * kernWidth;
    size_t const ksz2 = getKernelSize(pParamKernel) * pParamConv->group;
    if (ksz != ksz2) {
      printf("ksz %d != ksz2 %d\n");
    }
    for(int i=0; i<ksz; ++i){
      if (isnan(pk[i])) {
        printf("generateRandomData --> nans!\n");
        printf("ksz=%d i=%d\n");
        exit(-1);
      }
      if (pk[i] < -5.0 || pk[i] > +5.0){
        printf("generateRandomData --> outside [-5.0,5.0]\n");
        printf("ksz=%d i=%d\n");
        exit(-1);
      }
    }
    printf("input pDataKernel[0..%d - 1] looks good\n",ksz);
    printf("input pDataKernel[0..%lu - 1] looks good\n",(long unsigned)ksz2);
  }

  float * transformed_filter = NULL ;
  if( pParamKernel->layout == VEDNN_FILTER_LAYOUT_HWCN ) { // only support group=1
    if (group!=1){
      printf("Unsupported ref calc: HWCN wants group==1");
      exit(-1);
      //return VEDNN_ERROR_INVALID_PARAM;
    }

    const int N = outChannel ;
    const int C = inChannel ;
    const int H = kernHeight ;
    const int W = kernWidth ;

    float * filter = (float *) pDataKernel ;
    transformed_filter = (float *) malloc(sizeof(float)*N*C*H*W) ;
#pragma omp parallel for
    for(int n=0; n<N ; n++) {
      for(int c=0; c<C ; c++) {
        for(int hw=0; hw<H*W ; hw++) {
          transformed_filter[((n*C+c)*H)*W+hw] = filter[((hw)*C+c)*N+n] ;
        }
      }
    }
  }

  const float * restrict pIn     = pDataIn;
  const float * restrict pBias   = pDataBias;
  const float * restrict pKernel = transformed_filter == NULL ? pDataKernel : transformed_filter ;
  float * restrict pOut    = pDataOut;

  for (int n = 0; n < batch; n++) { // this->num_
    int inBatchOffset  = n * inChannel  * inWidth  * inHeight;
    int outBatchOffset = n * outChannel * outWidth * outHeight;

    for (int g = 0; g < group; g++) {
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
        int LDA = inWidth * inHeight;
        //printf(" M=%d N=%d K=%d ",M,N,K);

#if SGEMM_M1_SEGFAULTS==0 // issues ncc-3.4.20 and M==1 ???
        SGEMM(&NOTRANS, &NOTRANS, &N, &M, &K,
            &FONE, (float *) &pIn[inOffset], &LDA,
            (float *) &pKernel[kernGroupOffset], &K,
            &FZERO, &pOut[outOffset], &N);
#elif SGEMM_M1_SEGFAULTS==1 // issues ncc-3.4.20 and M==1 ???
        SGEMM_A1B0(&NOTRANS, &NOTRANS, &N, &M, &K,
            &FONE, (float *) &pIn[inOffset], &LDA,
            (float *) &pKernel[kernGroupOffset], &K,
            &FZERO, &pOut[outOffset], &N);
#else // ncc-3.4.20 fixup...  Here I show general equivalent for the SGEMM...
        if (M>1) { // || K>1) {
          SGEMM(&NOTRANS, &NOTRANS, &N, &M, &K,
              &FONE, (float *) &pIn[inOffset], &LDA, // LDA=N for no_im2col
              (float *) &pKernel[kernGroupOffset], &K, // LDB=K for no_im2col
              &FZERO, &pOut[outOffset], &N);
        }else{
          // M==1 has some BLAS segv !!! just write it out for now...
          // At -O3 and -O4, ncc should have -fassociative-math -fmatrix-multiply
          // and should emit matrix-multiply code for these
          // consulting sgemm docs
          //  3rd dim is the summation index
          float const* A = (float const*)&pIn[inOffset];            // size N x K
          float const* B = (float const*)&pKernel[kernGroupOffset]; // size K x M
          float * C = &pOut[outOffset];                             // size N x M
#if 0 // generic, long-hand matrix multiply
          if (M>1) { // actually this is NOT quite right yet :(
            for (int n=0; n<N; n++) {
              for (int m=0; m<M; m++) {
                C[m*N + n] = 0.0f;
              }
            }
            for (int n=0; n<N; n++) {
              for (int m=0; m<M; m++) {
                float acc = 0.0f;
                for (int k=0; k<K; k++) {
                  acc += A[k*N + n] * B[m*K + k];
                }
                C[m*N + n] += acc; // beta=0
              }
            }
          }else
#endif
#if 1
            if(M==1 && K>1){ // using just M==1
#pragma omp parallel if(N*K>GEMM_PARA_THRESH)
              for (int n=0; n<N; n++) {
                float acc = 0.0f;
                for (int k=0; k<K; k++) {
                  acc += A[k*N+n] * B[k];
                }
                C[n] = acc; // beta=0, no accumulation into C
              }
            }else
#endif
          { // using M==1 and K==1
            float b0 = B[0];
#pragma omp parallel if(N*K>GEMM_PARA_THRESH)
            for (int n=0; n<N; n++) {
              C[n] = A[n] * b0;
            }
          }
        }
#endif

        if (pBias) {
          //printf("noi2c pBias M=%d N=%d K=%d ", M,N,K);
#if SGEMM_M1_SEGFAULTS==0 // issues ncc-3.4.20 and M==1 ???
          SGEMM(&NOTRANS, &NOTRANS, &N, &M, &IONE,
              &FONE, (float *) pOne, &N,
              (float *) &pBias[biasGroupOffset], &IONE,
              &FONE, &pOut[outOffset], &N);
#elif SGEMM_M1_SEGFAULTS==1 // issues ncc-3.4.20 and M==1 ???
#if BIAS_SAXPY==0
          SGEMM_A1B1K1(&NOTRANS, &NOTRANS, &N, &M, &IONE,
              &FONE, (float *) pOne, &N,                // N x 1
              (float *) &pBias[biasGroupOffset], &IONE, // 1 x M
              &FONE, &pOut[outOffset], &N);             // N x M
#else
          // note that this might be formulated as a
          // SAXPY( N=MN,
          //        ALPHA=1.0,
          //        X=&pBias[biasGroupOffset],
          //        INCX=0, /* <-- "add constant" */
          //        Y=&pOut[outOffset], /* add to  this */
          //        INCY=1
          //      )
          // Unfortunately, this only handles for M=1
          if (M>1) {
            SGEMM(&NOTRANS, &NOTRANS, &N, &M, &IONE,
                &FONE, (float *) pOne, &N,
                (float *) &pBias[biasGroupOffset], &IONE,
                &FONE, &pOut[outOffset], &N);
          }else{
            //printf(" saxpy M=%d N=%d\n", M,N);
            cblas_saxpy( N, 1.0, &pBias[biasGroupOffset], 0,
                &pOut[outOffset], 1);
          }
#endif
#elif 0 // debug 
          {
            // M==1 has some BLAS segv !!! just write it out for now...
            // K=1 summation index is a huge simplification
            //   We might want to never fully call the SGEMM!
            //float const* A = (float const*)pOne;            // size N x K
            float const* B = (float const*)&pBias[biasGroupOffset]; // size K x M
            float * C = &pOut[outOffset];                   // size N x M
            if (1) { // further simplificcation only elides 1 scalar multiply
              //for (int n=0; n<N; n++) for (int m=0; m<M; m++) C[m*N + n] = 0.0f;
#if 0 // version 0, all loops before simplification
              printf("x7");
              int const K = 1;
              for (int n=0; n<N; n++) {
                for (int m=0; m<M; m++) {
                  float acc = 0.0f;
                  for (int k=0; k<K; k++) {
                    acc += 1.0/* A[k*N + n] */ * B[m*K + k];
                  }
                  C[m*N + n] += acc; // beta=0
                }
              }
#elif 0 // K=1 is a drastic simplification
              printf("x8");
              for (int m=0; m<M; ++m) {
                for (int n=0; n<N; ++n) {
                  C[m*N + n] += B[0]; // beta=0
                }
              }
#else // not working with omp parllel
              DBG("x9");
              int const MN = M * N;
//#pragma omp parallel if(MN > GEMM_PARA_THRESH) /* this cause wrong output */
              for (int mn=0; mn < MN; ++mn) {
                C[mn] += B[0]; // beta=0
              }
            }
          }
#endif

#else // dev code, summarized
          // maybe it's faster to always elide the SGEMM?
          if (M>1) {
            SGEMM(&NOTRANS, &NOTRANS, &N, &M, &IONE,
                &FONE, (float *) pOne, &N,
                (float *) &pBias[biasGroupOffset], &IONE,
                &FONE, &pOut[outOffset], &N);
          }else{
            // workaround for M=1 segfault circa ncc 3.4.20
            // using M==1 and K==1
            // here A[N] is 1.0
            float b0 = pBias[biasGroupOffset+0]; // B[0]
            int const MN = M * N;
            /* _Pragma("omp parallel if(MN > 32768)") //wrong output? */
            for (int mn=0; mn < MN; ++mn) {
              pOut[outOffset+mn] += b0; // C[n]
            }
          }
#endif
        }// if pBias

      } else {

        int M = outChannelGroup;
        int N = outWidth * outHeight;
        int K = inChannelGroup * kernWidth * kernHeight;

        im2col_cpu(&pIn[inOffset],
            inChannelGroup, inHeight, inWidth, kernHeight, kernWidth, outHeight, outWidth,
            padHeight, padWidth, strideHeight, strideWidth, dilationHeight, dilationWidth,
            pColBuff);

#if SGEMM_M1_SEGFAULTS==0 // issues ncc-3.4.20 and M==1 ???
        SGEMM(&NOTRANS, &NOTRANS, &N, &M, &K,
            &FONE, pColBuff, &N,
            (float *)&pKernel[kernGroupOffset], &K,
            &FZERO, &pOut[outOffset], &N);
        // segfault if M==1, at least w/ ncc 3.4.20 etc.
#elif SGEMM_M1_SEGFAULTS==1 // issues ncc-3.4.20 and M==1 ???
        SGEMM_A1B0(&NOTRANS, &NOTRANS, &N, &M, &K,
            &FONE, pColBuff, &N,
            (float *)&pKernel[kernGroupOffset], &K,
            &FZERO, &pOut[outOffset], &N);
#else
        if (M>1) {
          SGEMM(&NOTRANS, &NOTRANS, &N, &M, &K,
              &FONE, pColBuff, &N,                    // N x K
              (float *)&pKernel[kernGroupOffset], &K, // K x M
              &FZERO, &pOut[outOffset], &N);          // N x M
        }else{
          // M==1 has some BLAS segv !!! just write it out for now...
          // At -O3 and -O4, ncc should have -fassociative-math -fmatrix-multiply
          // and should emit matrix-multiply code for these
          float const* A = pColBuff;                                  // osz x icg*ksz (?)
          float const* B = (float const*)&pKernel[kernGroupOffset];   // icg*ksz x ocg
          float      * C = &pOut[outOffset];                          // osz x ocg (?)
#if 0
          if (M>1) {
            for (int n=0; n<N; n++) {
              for (int m=0; m<M; m++) {
                C[m*N + n] = 0.0f;
              }
            }
            for (int n=0; n<N; n++) {
              for (int m=0; m<M; m++) {
                float acc = 0.0f;
                for (int k=0; k<K; k++) {
                  acc += A[k*N + n] * B[m*K + k];
                }
                C[m*N + n] += acc; // beta=0, no accumulation into C
              }
            }
          }else
#endif
          { // M==1
            for (int n=0; n<N; n++) {
              float acc = 0.0f;
              for (int k=0; k<K; k++) {
                acc += A[k*N+n] * B[k];
              }
              C[n] = acc;
            }
          }
        }
#endif
        //printf(" back from SGEMM..."); fflush(stdout);

        if (pBias) {
          //printf("i2c bias...\n"); fflush(stdout);
#if SGEMM_M1_SEGFAULTS==0 // issues ncc-3.4.20 and M==1 ???
          SGEMM(&NOTRANS, &NOTRANS, &N, &M, &IONE,
              &FONE, (float *)pOne, &N,
              (float *) &pBias[biasGroupOffset], &IONE,
              &FONE, &pOut[outOffset], &N);
#elif SGEMM_M1_SEGFAULTS==1 // issues ncc-3.4.20 and M==1 ???
#if 1 //BIAS_SAXPY==0
          SGEMM_A1B1K1(&NOTRANS, &NOTRANS, &N, &M, &IONE,
              &FONE, (float *)pOne, &N,                 // size N x 1
              (float *) &pBias[biasGroupOffset], &IONE, // size 1 x M
              &FONE, &pOut[outOffset], &N);             // size N x M
#else
          cblas_saxpy( M*N, 1.0, &pBias[biasGroupOffset], 0,
              &pOut[outOffset], 1);
#endif
#else
          if (M>1) {
            SGEMM(&NOTRANS, &NOTRANS, &N, &M, &IONE,
                &FONE, (float *) pOne, &N,
                (float *) &pBias[biasGroupOffset], &IONE,
                &FONE, &pOut[outOffset], &N);
          }else{
            // workaround for M=1 segfault circa ncc 3.4.20
            // using M==1 and K==1
            // here A[N] is 1.0
            float b0 = pBias[biasGroupOffset+0]; // B[0]
            for (int n=0; n<N; n++) {
              pOut[outOffset+n] += b0; // C[n]
            }
          }
#endif
        }
      } // no_im2col?
    } // group
  } // batch

  if( transformed_filter != NULL ) free(transformed_filter) ;

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


  int no_im2col = (kernWidth == 1 && kernHeight == 1 && strideWidth == 1 && strideHeight == 1 && padWidth == 0 && padHeight == 0);

  float * transformed_filter = NULL ;
  if( pParamKernel->layout == VEDNN_FILTER_LAYOUT_HWCN ) { // only support group=1

    const int N = gOutChannel ;
    const int C = gInChannel ;
    const int H = kernHeight ;
    const int W = kernWidth ;

    float * filter = (float *) pDataKernel ;
    transformed_filter = (float *) malloc(sizeof(float)*N*C*H*W) ;
#pragma omp parallel for
    for(int n=0; n<N ; n++) {
      for(int c=0; c<C ; c++) {
        for(int hw=0; hw<H*W ; hw++) {
          transformed_filter[((n*C+c)*H)*W+hw] = filter[((hw)*C+c)*N+n] ;
        }
      }
    }
  }

  const float * restrict pGradOut = pDataGradOut;
  const float * restrict pKernel  = transformed_filter == NULL ? pDataKernel : transformed_filter ;
  float * restrict pGradIn  = pDataGradIn;

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
        SGEMM_A1B0t(&NOTRANS, &TRANS, &N, &M, &K,
            &FONE, (float *) &pGradOut[gOutOffset], &N, // N x K
            (float *) &pKernel[kernGroupOffset], &M,    // M x K (trans!)
            &FZERO, &pGradIn[gInOffset], &N);           // N x M
      }
      else {
        SGEMM_A1B0t(&NOTRANS, &TRANS, &N, &M, &K,
            &FONE, (float *) &pGradOut[gOutOffset], &N,
            (float *) &pKernel[kernGroupOffset], &M,
            &FZERO, pColBuff, &N);

        col2im_cpu(pColBuff,
            gInChannelGroup, gInHeight, gInWidth, kernHeight, kernWidth, gOutHeight, gOutWidth,
            padHeight, padWidth, strideHeight, strideWidth, dilationHeight, dilationWidth,
            &pGradIn[gInOffset]);
      }
    } // group
  } // batch

  if( transformed_filter != NULL ) free(transformed_filter) ;

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

  int inChannelGroup  = inChannel  / group; // pParamKernel->inChannel
  int outChannelGroup = outChannel / group; // pParamKernel->outChannel

  int no_im2col = (kernWidth == 1 && kernHeight == 1 && strideWidth == 1 && strideHeight == 1 && padWidth == 0 && padHeight == 0);

  float * transformed_filter = NULL ;
  if( pParamGradKernel->layout == VEDNN_FILTER_LAYOUT_HWCN ) { // only support group=1
    const int N = outChannel ;
    const int C = inChannel ;
    const int H = kernHeight ;
    const int W = kernWidth ;

    transformed_filter = (float *) malloc(sizeof(float)*N*C*H*W) ;
#pragma omp parallel for
    for(int i=0; i<N*C*H*W; i++) transformed_filter[i] = 0.f ;
  }

  const float * restrict pIn     = pDataIn;
  const float * restrict pOut    = pDataGradOut;
  float * restrict pKernel = transformed_filter == NULL ? pDataGradKernel : transformed_filter ;


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

        SGEMM_A1tB1(&TRANS, &NOTRANS, &N, &M, &K,
            &FONE,  (float*)&pIn[inOffset], &K,
            (float*)&pOut[outOffset], &K,
            &FONE, &pKernel[kernGroupOffset], &N);
      }
      else {
        im2col_cpu(&pIn[inOffset],
            inChannelGroup, inHeight, inWidth, kernHeight, kernWidth, outHeight, outWidth,
            padHeight, padWidth, strideHeight, strideWidth, dilationHeight, dilationWidth,
            pColBuff);

        int M = outChannelGroup;
        int N = inChannelGroup * kernWidth * kernHeight;
        int K = outWidth * outHeight;

        SGEMM_A1tB1(&TRANS, &NOTRANS, &N, &M, &K,
            &FONE,  pColBuff, &K,
            (float*)&pOut[outOffset], &K,
            &FONE, &pKernel[kernGroupOffset], &N);
      }
    } // group
  } // batch

  if( transformed_filter != NULL ) {
    const int N = outChannel ;
    const int C = inChannel ;
    const int H = kernHeight ;
    const int W = kernWidth ;

    float * filter = (float *) pDataGradKernel ;
#pragma omp parallel for
    for(int n=0; n<N ; n++) {
      for(int c=0; c<C ; c++) {
        for(int hw=0; hw<H*W ; hw++) {
          filter[((hw)*C+c)*N+n] += transformed_filter[((n*C+c)*H)*W+hw] ;
        }
      }
    }

    free(transformed_filter) ;
  }

  LFTRACE_END("convolution_backward_filter_gemm");
  return VEDNN_SUCCESS;
}
// vim: et ts=2 sw=2 cindent cino=^0,=0,l0,\:0,N-s syntax=cpp.doxygen
