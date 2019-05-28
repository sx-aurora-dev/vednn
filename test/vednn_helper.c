

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>

#include "vednn.h"


/* ----------------------------------------------------------------------- */
vednnError_t
createTensorParam(vednnTensorParam_t **ppParam, dataType_t dtype, int batch, int channel, int width, int height)
{
    vednnTensorParam_t *pParam = NULL;

    *ppParam = NULL;

    if (dtype != DTYPE_FLOAT) {
        return VEDNN_ERROR_INVALID_PARAM;
    }
    if (batch <= 0) {
        return VEDNN_ERROR_INVALID_PARAM;
    }
    if (channel <= 0) {
        return VEDNN_ERROR_INVALID_PARAM;
    }
    if (width <= 0) {
        return VEDNN_ERROR_INVALID_PARAM;
    }
    if (height <= 0) {
        return VEDNN_ERROR_INVALID_PARAM;
    }

    pParam = malloc(sizeof(vednnTensorParam_t));
    if (pParam == NULL) {
        return VEDNN_ERROR_MEMORY_EXHAUST;
    }
    memset(pParam, 0, sizeof(vednnTensorParam_t));

    pParam->dtype = dtype;
    pParam->batch = batch;
    pParam->channel = channel;
    pParam->width = width;
    pParam->height = height;

    *ppParam = pParam;

    return VEDNN_SUCCESS;
}



void
destroyTensorParam(vednnTensorParam_t *pParam)
{
    if (pParam) {
        free(pParam);
    }
}



dataType_t
getTensorDataType(const vednnTensorParam_t * restrict pParam)
{
    return pParam->dtype;
}



size_t
getTensorDataSize(const vednnTensorParam_t * restrict pParam)
{
    size_t dataSize = 0;

    switch (pParam->dtype) {
    case DTYPE_FLOAT:
        dataSize = sizeof(float);
        break;
    default:
        assert(0);              /* BUG */
        break;
    }

    return dataSize;
}



size_t
getTensorSize(const vednnTensorParam_t * restrict pParam)
{
    size_t size = pParam->batch * pParam->channel * pParam->width * pParam->height;

    return size;
}



size_t
getTensorSizeInByte(const vednnTensorParam_t * restrict pParam)
{
    size_t size = getTensorSize(pParam) * getTensorDataSize(pParam);

    return size;
}



void
dumpTensorData(FILE *pFile, const char * restrict pName, const vednnTensorParam_t * restrict pParam, const void * restrict pData)
{
    int b, c, h, w;

    fprintf(pFile, "%s[%d][%d][%d][%d] = {\n", pName, pParam->batch, pParam->channel, pParam->height, pParam->width);
    for (b = 0; b < pParam->batch; b++) {
        fprintf(pFile, "  { // batch %d\n",  b);
        for (c = 0; c < pParam->channel; c++) {
            fprintf(pFile, "    { // channel %d\n",  c);
            for (h = 0; h < pParam->height; h++) {
                fprintf(pFile, "      {");
                for (w = 0; w < pParam->width; w++) {
                    double data = 0.0;
                    switch (pParam->dtype) {
                    case DTYPE_FLOAT:
                        data = ((float *)pData)[((b * pParam->channel + c) * pParam->height + h) * pParam->width + w];
                        break;
                    default:
                        assert(0);              /* BUG */
                        break;
                    }
                    fprintf(pFile, "%s%f", (w == 0 ? " " : ", "), data);
                }
                fprintf(pFile, " },\n");
            }
            fprintf(pFile, "    },\n");
        }
        fprintf(pFile, "  },\n");
    }
    fprintf(pFile, "};\n\n");
}



/* ----------------------------------------------------------------------- */
vednnError_t
createBiasParam(vednnBiasParam_t **ppParam, dataType_t dtype, int channel)
{
    vednnBiasParam_t *pParam = NULL;

    *ppParam = NULL;

    if (dtype != DTYPE_FLOAT) {
        return VEDNN_ERROR_INVALID_PARAM;
    }
    if (channel <= 0) {
        return VEDNN_ERROR_INVALID_PARAM;
    }

    pParam = malloc(sizeof(vednnBiasParam_t));
    if (pParam == NULL) {
        return VEDNN_ERROR_MEMORY_EXHAUST;
    }
    memset(pParam, 0, sizeof(vednnBiasParam_t));

    pParam->dtype = dtype;
    pParam->channel = channel;

    *ppParam = pParam;

    return VEDNN_SUCCESS;
}



void
destroyBiasParam(vednnBiasParam_t *pParam)
{
    if (pParam) {
        free(pParam);
    }
}



dataType_t
getBiasDataType(const vednnBiasParam_t * restrict pParam)
{
    return pParam->dtype;
}



size_t
getBiasDataSize(const vednnBiasParam_t * restrict pParam)
{
    size_t dataSize = 0;

    switch (pParam->dtype) {
    case DTYPE_FLOAT:
        dataSize = sizeof(float);
        break;
    default:
        assert(0);              /* BUG */
        break;
    }

    return dataSize;
}



size_t
getBiasSize(const vednnBiasParam_t * restrict pParam)
{
    size_t size = pParam->channel;

    return size;
}



size_t
getBiasSizeInByte(const vednnBiasParam_t * restrict pParam)
{
    size_t size = getBiasSize(pParam) * getBiasDataSize(pParam);

    return size;
}



void
dumpBiasData(FILE *pFile, const char * restrict pName, const vednnBiasParam_t * restrict pParam, const void * restrict pData)
{
    int c;

    fprintf(pFile, "%s[%d] = {\n", pName, pParam->channel);
    for (c = 0; c < pParam->channel; c++) {
        double data = 0.0;
        switch (pParam->dtype) {
        case DTYPE_FLOAT:
            data = ((float *)pData)[c];
            break;
        default:
            assert(0);              /* BUG */
            break;
        }
        fprintf(pFile, "%s%f", (c == 0 ? " " : ", "), data);
    }
    fprintf(pFile, "};\n\n");
}



/* ----------------------------------------------------------------------- */
vednnError_t
createKernelParam(vednnFilterParam_t **ppParam, dataType_t dtype, int inChannel, int outChannel, int width, int height)
{
    vednnFilterParam_t *pParam = NULL;

    *ppParam = NULL;

    if (dtype != DTYPE_FLOAT) {
        return VEDNN_ERROR_INVALID_PARAM;
    }
    if (inChannel <= 0) {
        return VEDNN_ERROR_INVALID_PARAM;
    }
    if (outChannel <= 0) {
        return VEDNN_ERROR_INVALID_PARAM;
    }
    if (width <= 0) {
        return VEDNN_ERROR_INVALID_PARAM;
    }
    if (height <= 0) {
        return VEDNN_ERROR_INVALID_PARAM;
    }

    pParam = malloc(sizeof(vednnFilterParam_t));
    if (pParam == NULL) {
        return VEDNN_ERROR_MEMORY_EXHAUST;
    }
    memset(pParam, 0, sizeof(vednnFilterParam_t));

    pParam->dtype	= dtype;
    pParam->inChannel	= inChannel;
    pParam->outChannel	= outChannel;
    pParam->width	= width;
    pParam->height	= height;

    *ppParam = pParam;

    return VEDNN_SUCCESS;
}



void
destroyKernelParam(vednnFilterParam_t *pParam)
{
    if (pParam) {
        free(pParam);
    }
}



dataType_t
getKernelDataType(const vednnFilterParam_t * restrict pParam)
{
    return pParam->dtype;
}



size_t
getKernelDataSize(const vednnFilterParam_t * restrict pParam)
{
    size_t dataSize = 0;

    switch (pParam->dtype) {
    case DTYPE_FLOAT:
        dataSize = sizeof(float);
        break;
    default:
        assert(0);              /* BUG */
        break;
    }

    return dataSize;
}



size_t
getKernelSize(const vednnFilterParam_t * restrict pParam)
{
    size_t size = pParam->outChannel * pParam->inChannel * pParam->width * pParam->height;

    return size;
}



size_t
getKernelSizeInByte(const vednnFilterParam_t * restrict pParam)
{
    size_t size = getKernelSize(pParam) * getKernelDataSize(pParam);

    return size;
}



void
dumpKernelData(FILE *pFile, const char * restrict pName, const vednnFilterParam_t * restrict pParam, const void * restrict pData)
{
    int out, in, h, w;

    fprintf(pFile, "%s[%d][%d][%d][%d] = {\n", pName, pParam->outChannel, pParam->inChannel, pParam->height, pParam->width);
    for (out = 0; out < pParam->outChannel; out++) {
        fprintf(pFile, "  { // out channel %d\n",  out);
        for (in = 0; in < pParam->inChannel; in++) {
            fprintf(pFile, "    { // channel %d\n",  in);
            for (h = 0; h < pParam->height; h++) {
                fprintf(pFile, "      {");
                for (w = 0; w < pParam->width; w++) {
                    double data = 0.0;
                    switch (pParam->dtype) {
                    case DTYPE_FLOAT:
                        data = ((float *)pData)[((out * pParam->inChannel + in) * pParam->height + h) * pParam->width + w];
                        break;
                    default:
                        assert(0);              /* BUG */
                        break;
                    }
                    fprintf(pFile, "%s%f", (w == 0 ? " " : ", "), data);
                }
                fprintf(pFile, " },\n");
            }
            fprintf(pFile, "    },\n");
        }
        fprintf(pFile, "  },\n");
    }
    fprintf(pFile, "};\n\n");
}



/* ----------------------------------------------------------------------- */
vednnError_t
createConvolutionParam(vednnConvolutionParam_t **ppParam, int group, int strideWidth, int strideHeight, int padWidth, int padHeight, int dilationWidth, int dilationHeight)
{
    vednnConvolutionParam_t *pParam = NULL;

    *ppParam = NULL;

    if (group <= 0) {
        return VEDNN_ERROR_INVALID_PARAM;
    }
    if (strideWidth <= 0) {
        return VEDNN_ERROR_INVALID_PARAM;
    }
    if (strideHeight <= 0) {
        return VEDNN_ERROR_INVALID_PARAM;
    }
    if (padWidth < 0) {
        return VEDNN_ERROR_INVALID_PARAM;
    }
    if (padHeight < 0) {
        return VEDNN_ERROR_INVALID_PARAM;
    }
    if (dilationWidth < 0) {
        return VEDNN_ERROR_INVALID_PARAM;
    }
    if (dilationHeight < 0) {
        return VEDNN_ERROR_INVALID_PARAM;
    }

    pParam = malloc(sizeof(vednnConvolutionParam_t));
    if (pParam == NULL) {
        return VEDNN_ERROR_MEMORY_EXHAUST;
    }
    memset(pParam, 0, sizeof(vednnConvolutionParam_t));

    pParam->group		= group;
    pParam->strideWidth		= strideWidth;
    pParam->strideHeight	= strideHeight;
    pParam->padWidth		= padWidth;
    pParam->padHeight		= padHeight;
    pParam->dilationWidth	= dilationWidth;
    pParam->dilationHeight	= dilationHeight;

    *ppParam = pParam;

    return VEDNN_SUCCESS;
}



void
destroyConvolutionParam(vednnConvolutionParam_t *pParam)
{
    if (pParam) {
        free(pParam);
    }
}


/* ----------------------------------------------------------------------- */
vednnError_t
createPoolingParam(vednnPoolingParam_t **ppParam, int windowWidth, int windowHeight, int strideWidth, int strideHeight, int padWidth, int padHeight)
{
    vednnPoolingParam_t *pParam = NULL;

    *ppParam = NULL;

    if (windowWidth < 0) {
        return VEDNN_ERROR_INVALID_PARAM;
    }
    if (windowHeight < 0) {
        return VEDNN_ERROR_INVALID_PARAM;
    }
    if (strideWidth <= 0) {
        return VEDNN_ERROR_INVALID_PARAM;
    }
    if (strideHeight <= 0) {
        return VEDNN_ERROR_INVALID_PARAM;
    }
    if (padWidth < 0) {
        return VEDNN_ERROR_INVALID_PARAM;
    }
    if (padHeight < 0) {
        return VEDNN_ERROR_INVALID_PARAM;
    }


    pParam = malloc(sizeof(vednnPoolingParam_t));
    if (pParam == NULL) {
        return VEDNN_ERROR_MEMORY_EXHAUST;
    }
    memset(pParam, 0, sizeof(vednnPoolingParam_t));

    pParam->windowWidth		= windowWidth;
    pParam->windowHeight	= windowHeight;
    pParam->strideWidth		= strideWidth;
    pParam->strideHeight	= strideHeight;
    pParam->padWidth		= padWidth;
    pParam->padHeight		= padHeight;

    *ppParam = pParam;

    return VEDNN_SUCCESS;
}



void
destroyPoolingParam(vednnPoolingParam_t *pParam)
{
    if (pParam) {
        free(pParam);
    }
}


