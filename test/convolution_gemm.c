/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <vednn.h>
#include "vednn_helper.h"

#define SGEMM	sgemm_
void sgemm_(char *TRANSA, char *TRANSB, int *M, int *N, int *K,
            float *ALPHA, float *A,  int *LDA, float *B, int *LDB,
            float *BETA, float *C, int *LDC ) ;

static char  TRANS   = 'T';
static char  NOTRANS = 'N';
static float FONE    = 1.0f;
static float FZERO   = 0.0f;
static int   IONE    = 1;

/* ----------------------------------------------------------------------- */
static inline int is_a_ge_zero_and_a_lt_b(int a, int b) {
    return (unsigned)a < (unsigned)b;
}

static void
im2col_cpu(const float * restrict data_im, const int channels,
           const int height, const int width, const int kernel_h, const int kernel_w,
           const int pad_h, const int pad_w,
           const int stride_h, const int stride_w,
           const int dilation_h, const int dilation_w,
           float * restrict data_col)
{
    const int output_h = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
    const int output_w = (width + 2 * pad_w -  (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
    const int channel_size = height * width;

    int channel;

    for (channel = 0 ; channel < channels; channel++) {				// inChannel
        int kernel_row, kernel_col, output_rows, output_cols, output_col;

        int inOffset = channel * channel_size;
        int outOffset = channel * output_h * output_w * kernel_h * kernel_w;

        for (kernel_row = 0; kernel_row < kernel_h; kernel_row++) {		// kernHeight
            for (kernel_col = 0; kernel_col < kernel_w; kernel_col++) {		// kernWidth
                int input_row = -pad_h + kernel_row * dilation_h;
                for (output_rows = output_h; output_rows; output_rows--) {	// outHeight
                    if (!is_a_ge_zero_and_a_lt_b(input_row, height)) {
                        for (output_cols = output_w; output_cols; output_cols--) {
                            // *(data_col++) = 0;
                            data_col[outOffset++] = 0;
                        }
                    } else {
                        int input_col = -pad_w + kernel_col * dilation_w;
                        for (output_col = output_w; output_col; output_col--) {	// outWidth
                            if (is_a_ge_zero_and_a_lt_b(input_col, width)) {
                                // *(data_col++) = data_im[input_row * width + input_col];
                                data_col[outOffset++] = data_im[inOffset + input_row * width + input_col];
                            } else {
                                // *(data_col++) = 0;
                                data_col[outOffset++] = 0;
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

static void
col2im_cpu(
    const float* data_col, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    float* data_im) {

  memset(data_im, 0, sizeof(float)*height*width*channels) ;

  const int output_h = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
  const int output_w = (width + 2 * pad_w -  (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
  const int channel_size = height * width;

  int channel;

  for (channel = 0 ; channel < channels; channel++) {				// inChannel
      int kernel_row, kernel_col, output_rows, output_cols, output_col;

      int inOffset = channel * channel_size;
      int outOffset = channel * output_h * output_w * kernel_h * kernel_w;

      for (kernel_row = 0; kernel_row < kernel_h; kernel_row++) {		// kernHeight
          for (kernel_col = 0; kernel_col < kernel_w; kernel_col++) {		// kernWidth
              int input_row = -pad_h + kernel_row * dilation_h;
              for (output_rows = output_h; output_rows; output_rows--) {	// outHeight
                  if (!is_a_ge_zero_and_a_lt_b(input_row, height)) {
                      for (output_cols = output_w; output_cols; output_cols--) {
                          // *(data_col++) = 0;
                          data_col[outOffset++] ;
                      }
                  } else {
                      int input_col = -pad_w + kernel_col * dilation_w;
                      for (output_col = output_w; output_col; output_col--) {	// outWidth
                          if (is_a_ge_zero_and_a_lt_b(input_col, width)) {
                              // *(data_col++) = data_im[input_row * width + input_col];
                            data_im[inOffset + input_row * width + input_col] += data_col[outOffset++] ;
                          } else {
                              // *(data_col++) = 0;
                              data_col[outOffset++] ;
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

vednnError_t
convolution_forward_gemm(
    const vednnTensorParam_t * restrict pParamIn, const void * restrict pDataIn,
    const vednnFilterParam_t * restrict pParamKernel, const void * restrict pDataKernel,
    const vednnBiasParam_t * restrict pParamBias, const void * restrict pDataBias,
    const vednnTensorParam_t * restrict pParamOut, void * restrict pDataOut,
    const float * restrict pOne,  float * restrict pColBuff,
    const vednnConvolutionParam_t * restrict pParamConv )
{
    int n, g;

    int batch		= pParamIn->batch;
    int inChannel	= pParamIn->channel;
    int inWidth		= pParamIn->width;
    int inHeight	= pParamIn->height;
    int outChannel	= pParamOut->channel;
    int outWidth	= pParamOut->width;
    int outHeight	= pParamOut->height;
    int kernWidth	= pParamKernel->width;
    int kernHeight	= pParamKernel->height;

    int group		= pParamConv->group;
    int strideWidth	= pParamConv->strideWidth;;
    int strideHeight	= pParamConv->strideHeight;
    int padWidth	= pParamConv->padWidth;
    int padHeight	= pParamConv->padHeight;
    int dilationWidth	= pParamConv->dilationWidth;
    int dilationHeight	= pParamConv->dilationHeight;

    int inChannelGroup	= inChannel  / group;	// pParamKernel->inChannel と同じ
    int outChannelGroup	= outChannel / group;	// pParamKernel->outChannel と同じ

    const float * restrict pIn     = pDataIn;
    const float * restrict pBias   = pDataBias;
    const float * restrict pKernel = pDataKernel;
          float * restrict pOut    = pDataOut;

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

                int M = outChannelGroup;
                int N = outWidth * outHeight;
                int K = inChannelGroup * kernWidth * kernHeight;

                im2col_cpu(&pIn[inOffset],
                           inChannelGroup, inHeight, inWidth, kernHeight, kernWidth,
                           padHeight, padWidth, strideHeight, strideWidth, dilationHeight, dilationWidth,
                           pColBuff);


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
    int n, g;

    int batch		= pParamGradOut->batch;
    int gOutChannel	= pParamGradOut->channel;
    int gOutWidth	= pParamGradOut->width;
    int gOutHeight	= pParamGradOut->height;
    int gInChannel	= pParamGradIn->channel;
    int gInWidth	= pParamGradIn->width;
    int gInHeight	= pParamGradIn->height;
    int kernWidth	= pParamKernel->width;
    int kernHeight	= pParamKernel->height;

    int group		= pParamConv->group;
    int strideWidth	= pParamConv->strideWidth;;
    int strideHeight	= pParamConv->strideHeight;
    int padWidth	= pParamConv->padWidth;
    int padHeight	= pParamConv->padHeight;
    int dilationWidth	= pParamConv->dilationWidth;
    int dilationHeight	= pParamConv->dilationHeight;

    int gOutChannelGroup = gOutChannel  / group;
    int gInChannelGroup	 = gInChannel / group;

    const float * restrict pGradOut = pDataGradOut;
    const float * restrict pKernel  = pDataKernel;
          float * restrict pGradIn  = pDataGradIn;

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
    int n, g;

    int batch		= pParamIn->batch;
    int inChannel	= pParamIn->channel;
    int inWidth		= pParamIn->width;
    int inHeight	= pParamIn->height;
    int outChannel	= pParamGradOut->channel;
    int outWidth	= pParamGradOut->width;
    int outHeight	= pParamGradOut->height;
    int kernWidth	= pParamGradKernel->width;
    int kernHeight	= pParamGradKernel->height;

    int group		= pParamConv->group;
    int strideWidth	= pParamConv->strideWidth;;
    int strideHeight	= pParamConv->strideHeight;
    int padWidth	= pParamConv->padWidth;
    int padHeight	= pParamConv->padHeight;
    int dilationWidth	= pParamConv->dilationWidth;
    int dilationHeight	= pParamConv->dilationHeight;

    int inChannelGroup	= inChannel  / group;	// pParamKernel->inChannel と同じ
    int outChannelGroup	= outChannel / group;	// pParamKernel->outChannel と同じ

    const float * restrict pIn     = pDataIn;
    const float * restrict pOut    = pDataGradOut;
          float * restrict pKernel = pDataGradKernel ;

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

    return VEDNN_SUCCESS;
}

