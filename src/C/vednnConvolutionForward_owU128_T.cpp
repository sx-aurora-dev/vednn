#include <stdint.h>
#include <stdlib.h>

#include "vednn.h"

#define VLEN	(256)

#include "mkldnn_thread.hpp"
#include "vednnConvolutionForward.h"

#ifndef restrict
#define restrict __restrict__
#endif

extern "C" {

vednnError_t
vednnConvolutionForward_direct_owU128_T(
    const vednnTensorParam_t * restrict 	pParamIn,
    const void * restrict 			pDataIn,
    const vednnFilterParam_t * restrict 	pParamKernel,
    const void * restrict 			pDataKernel,
    const vednnConvolutionParam_t * restrict 	pParamConv,
    const vednnTensorParam_t * restrict 	pParamOut,
    void * restrict 				pDataOut
)
{

  const int64_t batch      = pParamIn->batch;
  const int64_t group      = pParamConv->group;
  const int64_t outChannel = pParamOut->channel;
  const int64_t outWidth   = pParamOut->width;
  const int64_t outHeight  = pParamOut->height;
  const int64_t outChannelGroup = outChannel / group;   // equal to pDataKernel->outChannel
  const int oPixels= outHeight*outWidth ;

  int64_t outChannelGroupPrime = (outChannelGroup + 15) / 16;

  const int64_t nY = VLEN / outWidth ;
  const int64_t oYPrime = (outHeight + (nY-1)) / nY;

  mkldnn::impl::parallel_nd(batch, group, outChannelGroupPrime, oYPrime,
              [&](int n, int g, int kPrime, int oyPrime) {

    vednnConvolutionForward_direct_owU128_T_subkernel(
      pParamIn, pDataIn, pParamKernel, pDataKernel,
      pParamConv, pParamOut, pDataOut, n, g, kPrime, oyPrime);

  });  /* kernel of parallel_nd() */

  return VEDNN_SUCCESS;
}

}

