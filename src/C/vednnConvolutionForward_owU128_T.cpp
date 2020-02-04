/** \file
 * internal impl with threading done inside mb,g loops
 */
#include "vednnConvolutionForward.h"
#include "mkldnn_thread.hpp"

#define VLEN    (256)


extern "C" {//}

// Note: internal impl API here.
vednnError_t
vednnConvolutionForward_direct_owU128_T(
        VEDNN_CONVFWD_ARGS
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
                pParamBias, pDataBias,
                pParamConv, pParamOut, pDataOut, n, g, kPrime, oyPrime);

            });  /* kernel of parallel_nd() */

    return VEDNN_SUCCESS;
}

} // "C"
// vim: et ts=4 sw=4 cindent cino=+2s,^=lt0,\:0,N-s
