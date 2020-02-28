/** \file
 * internal impl with threading done inside mb,g loops
 */
#include "vednnConvolutionForward.h"
#include "mkldnn_thread.hpp"

#define VLEN    (256)


extern "C" { //}

// Note: internal impl API here.
vednnError_t
vednnConvolutionForward_direct_dil1_str1_pad0_ker1_T(
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
    int64_t oPixelsPrime = (oPixels + VLEN-1) / VLEN;
    mkldnn::impl::parallel_nd(batch, group, outChannelGroupPrime, oPixelsPrime,
            [&](int n, int g, int kPrime, int opPrime) {

            vednnConvolutionForward_direct_dil1_str1_pad0_ker1_T_subkernel(
                pParamIn, pDataIn, pParamKernel, pDataKernel,
                //nullptr, nullptr,
                pParamBias, pDataBias,
                pParamConv, pParamOut, pDataOut, n, g, kPrime, opPrime);

            });  /* kernel of parallel_nd() */

    return VEDNN_SUCCESS;
}

}
// vim: et ts=4 sw=4 cindent cino=+2s,^=l0,\:0,N-s
