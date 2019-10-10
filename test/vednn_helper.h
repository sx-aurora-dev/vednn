#ifndef TEST_VEDNN_HELPER_H_
#define TEST_VEDNN_HELPER_H_

#include "vednn.h"

#include <stddef.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
// not sure about next condition (has it changed to be defined in necvals.h?)
#if !defined(restrict)
#define restrict __restrict__
#endif
#endif

vednnError_t
createTensorParam(vednnTensorParam_t **ppParam, dataType_t dtype,
                  int batch, int channel, int width, int height);

void
destroyTensorParam(vednnTensorParam_t *pParam);

dataType_t
getTensorDataType(const vednnTensorParam_t * restrict pParam);

size_t
getTensorDataSize(const vednnTensorParam_t * restrict pParam);

size_t
getTensorSize(const vednnTensorParam_t * restrict pParam);

size_t
getTensorSizeInByte(const vednnTensorParam_t * restrict pParam);

void
dumpTensorData(FILE *pFile, const char * restrict pName,
               const vednnTensorParam_t * restrict pParam, const void * restrict pData);



vednnError_t
createBiasParam(vednnBiasParam_t **ppParam, dataType_t dtype, int channel);

void
destroyBiasParam(vednnBiasParam_t *pParam);

dataType_t
getBiasDataType(const vednnBiasParam_t * restrict pParam);

size_t
getBiasDataSize(const vednnBiasParam_t * restrict pParam);

size_t
getBiasSize(const vednnBiasParam_t * restrict pParam);

size_t
getBiasSizeInByte(const vednnBiasParam_t * restrict pParam);

void
dumpBiasData(FILE * restrict pFile, const char * restrict pName,
             const vednnBiasParam_t * restrict pParam, const void * restrict pData);



vednnError_t
createKernelParam(vednnFilterParam_t **ppParam, dataType_t dtype,
                  int inChannel, int outChannel, int width, int height);

void
destroyKernelParam(vednnFilterParam_t *pParam);

dataType_t
getKernelDataType(const vednnFilterParam_t * restrict pParam);

size_t
getKernelDataSize(const vednnFilterParam_t * restrict pParam);

size_t
getKernelSize(const vednnFilterParam_t * restrict pParam);

size_t
getKernelSizeInByte(const vednnFilterParam_t * restrict pParam);

void
dumpKernelData(FILE * restrict pFile, const char * restrict pName,
               const vednnFilterParam_t * restrict pParam, const void * restrict pData);

vednnError_t
createConvolutionParam(vednnConvolutionParam_t **ppParam,
                       int group,
                       int strideWidth, int strideHeight,
                       int padWidth, int padHeight,
                       int dilationWidth, int dilationHeight);

void
destroyConvolutionParam(vednnConvolutionParam_t *pParam);


vednnError_t
createPoolingParam(vednnPoolingParam_t **ppParam,
                   int windowWidth, int windowHeight,
		   int strideWidth, int strideHeight,
		   int padWidth, int padHeight) ;

void destroyPoolingParam(vednnPoolingParam_t *pParam) ;

#ifdef __cplusplus
}// extern "C"
#endif
#endif /* TEST_VEDNN_HELPER_H_ */
