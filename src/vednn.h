#ifndef __VEDNN__
#define __VEDNN__

#define VEDNN_MAJOR        1
#define VEDNN_MINOR        0
#define VEDNN_PATCHLEVEL   5
#define VEDNN_VERSION    (VEDNN_MAJOR * 1000 + VEDNN_MINOR * 100 + VEDNN_PATCHLEVEL)

#if defined(__cplusplus) && !defined(restrict)
#define restrict __restrict__
#endif

#ifdef __cplusplus
extern "C" { /*}*/
#endif

/** vednn always initializes this to env OMP_NUM_THREADS, but you may wish to
 * control it better.
 *
 * - This routine does 1 thing:
 *   - set \c __vednn_omp_num_threads used by libvednn internals
 *
 * - It does NOT call \c omp_set_num_threads(threads)
 *
 * Note that if you do not have OMP_NUM_THREADS in env, libvednn init code would
 * use zero for \c __vednn_omp_num_threads in \ref vednnInit.c
 *
 * This was changed so default will be \c omp_get_max_threads() for -fopenmp
 * compilation.   This way if env vars are unset, gemm convolution will still
 * allow blas to use the desired number of threads.
 *
 * Often the following snippet will do something reasonable, and can be inlined
 * if you include \ref vednn-def.h :
 * ```
 * omp_set_num_threads(threads);
 * __vednn_omp_num_threads = threads;
 * ```
 */
#if 0
void vednn_set_num_threads(int const threads);
int vednn_get_num_threads();
#else // simpler, no problems w/ C vs C++ linkage, but need vednn-def.h
#define vednn_set_num_threads(THREADS) (__vednn_omp_num_threads = THREADS)
#define vednn_get_num_threads(...) (__vednn_omp_num_threads)
#endif

typedef enum {
    VEDNN_SUCCESS = 0,
    VEDNN_ERROR_INVALID_PARAM   = 1,
    VEDNN_ERROR_MEMORY_EXHAUST  = 2,
} vednnError_t;

typedef enum {
    DTYPE_FLOAT,
    // DTYPE_DOUBLE,
} dataType_t;

typedef enum {
    VEDNN_ACTIVATION_RELU = 0,
} vednnActivationMode_t;

typedef enum {
    VEDNN_SOFTMAX_FAST = 0,
    VEDNN_SOFTMAX_ACCURATE,
    VEDNN_SOFTMAX_LOG
} vednnSoftmaxMode_t;

/* Convolution Parametes */

typedef enum {
    VEDNN_CONV_ALGORITHM_DIRECT,
//    VEDNN_CONV_ALGORITHM_GEMM,
} vednnConvolutionAlgorithm_t;

typedef struct {
    dataType_t dtype;
    int        batch;
    int        channel;
    int        width;
    int        height;
} vednnTensorParam_t;

typedef struct {
    dataType_t dtype;
    int        channel;
} vednnBiasParam_t;

typedef enum {
  VEDNN_FILTER_LAYOUT_NCHW = 0,
  VEDNN_FILTER_LAYOUT_HWCN = 1    // support group=1 only
} filterLayout_t;

typedef struct {
    dataType_t     dtype;
    filterLayout_t layout;
    int            inChannel;    // inChannel / group
    int            outChannel;    // outChannel / group
    int            width;
    int            height;
} vednnFilterParam_t;

typedef struct {
    int        group;
    int        strideWidth;
    int        strideHeight;
    int        padWidth;
    int        padHeight;
    int        dilationWidth;
    int        dilationHeight;
} vednnConvolutionParam_t ;

typedef struct {
    int        windowWidth;
    int        windowHeight;
    int        strideWidth;
    int        strideHeight;
    int        padWidth;
    int        padHeight;
} vednnPoolingParam_t ;

vednnError_t vednnConvolutionForward(
    const vednnTensorParam_t      *pParamIn,
    const void                    *pDataIn,
    const vednnFilterParam_t      *pParamKernel,
    const void                    *pDataKernel,
    const vednnTensorParam_t      *pParamOut,
    void                          *pDataOut,
    const vednnConvolutionParam_t *pParamConv,
    vednnConvolutionAlgorithm_t   algo
) ;

vednnError_t vednnConvolutionForwardAddBias(
    const vednnTensorParam_t      *pParamIn,
    const void                    *pDataIn,
    const vednnFilterParam_t      *pParamKernel,
    const void                    *pDataKernel,
    const vednnBiasParam_t        *pParamBias,
    const void                    *pDataBias,
    const vednnTensorParam_t      *pParamOut,
    void                          *pDataOut,
    const vednnConvolutionParam_t *pParamConv,
    vednnConvolutionAlgorithm_t   algo
) ;

vednnError_t vednnConvolutionBackwardData(
    const vednnTensorParam_t      *pParamGradOut,
    const void                    *pDataGradOut,
    const vednnFilterParam_t      *pParamKernel,
    const void                    *pDataKernel,
    const vednnTensorParam_t      *pParamGradIn,
    void                          *pDataGradIn,
    const vednnConvolutionParam_t *pParamConv,
    vednnConvolutionAlgorithm_t   algo
) ;

vednnError_t vednnConvolutionBackwardFilter(
    const vednnTensorParam_t      *pParamIn,
    const void                    *pDataIn,
    const vednnTensorParam_t      *pParamGradOut,
    const void                    *pDataGradOut,
    const vednnFilterParam_t      *pParamGradKernel,
    void                          *pDataGradKernel,
    const vednnConvolutionParam_t *pParamConv,
    vednnConvolutionAlgorithm_t   algo
) ;

vednnError_t vednnLinearForward(
    const unsigned long inDim,
    const unsigned long outDim,
    const unsigned long nBatch,
    const void          *pDataIn,
    const void          *pDataWeight,
    void                *pDataOut
) ;

vednnError_t vednnLinearBackwardData(
    const unsigned long inDim,
    const unsigned long outDim,
    const unsigned long nBatch,
    const void          *pDataGradOut,
    const void          *pDataWeight,
    void                *pData
) ;

vednnError_t vednnLinearBackwardWeight(
    const unsigned long inDim,
    const unsigned long outDim,
    const unsigned long nBatch,
    const void *        pDataIn,
    const void *        pDataGradOut,
    void *              pDataGradWeight
) ;

vednnError_t vednnMaxPoolingForward(
    const vednnTensorParam_t  *pParamIn,
    const void                *pDataIn,
    const vednnTensorParam_t  *pParamOut,
    void                      *pDataOut,
    const vednnPoolingParam_t *pParamPool
) ;

vednnError_t vednnMaxPoolingBackward(
    const vednnTensorParam_t  *pParamGradOut,
    const void                *pDataGradOut,
    const vednnTensorParam_t  *pParamOut,
    const void                *pDataOut,
    const vednnTensorParam_t  *pParamIn,
    const void                *pDataIn,
    const vednnTensorParam_t  *pParamGradIn,
    void                      *pDataGradIn,
    const vednnPoolingParam_t *pParamPool
) ;

vednnError_t vednnActivationForward(
    const vednnActivationMode_t mode,
    const void                  *pDataIn,
    void                        *pDataOut,
    const unsigned long         nElements
) ;

vednnError_t vednnActivationBackward(
    const vednnActivationMode_t mode,
    const void                  *pDataGradOut,
    const void                  *pDataIn,
    void                        *pDataGradIn,
    const unsigned long         nElements
) ;

vednnError_t vednnSoftmaxForward(
    const vednnSoftmaxMode_t mode,
    const void               *pDataIn,
    void                     *pDataOut,
    const unsigned long      nBatch,
    const unsigned long      nClass
) ;

#ifdef __cplusplus
}

#include "vednn_util.hpp"
#endif
// vim: ts=4 sw=4 et ai
#endif
