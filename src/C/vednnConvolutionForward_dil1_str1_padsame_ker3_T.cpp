/* extern "C" { */
#include <stdint.h>
#include <stdlib.h>

#include "vednn.h"
/* } */

#define VLEN	(256)

#include "mkldnn_thread.hpp"
#include "vednnConvolutionForward.h"

#define DEBUG_THREADS 0
#if DEBUG_THREADS
# define RESERVE_THREADS 9
# include <iostream>
#endif

extern "C" {

vednnError_t
vednnConvolutionForward_direct_dil1_str1_padsame_ker3_T(
    const vednnTensorParam_t * restrict 	pParamIn,
    const void * restrict 			pDataIn,
    const vednnFilterParam_t * restrict 	pParamKernel,
    const void * restrict 			pDataKernel,
    const vednnBiasParam_t * restrict           pParamBias,
    const void * restrict                       pDataBias,
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

#if DEBUG_THREADS
  int ithrUsed[RESERVE_THREADS];
  int nthrUsed[RESERVE_THREADS];
  for (int i = 0; i < RESERVE_THREADS; ++i) {
    ithrUsed[i] =  0;
    nthrUsed[i] = 0;
   }
#endif
  
  int64_t outChannelGroupPrime = (outChannelGroup + 15) / 16;
  int64_t oPixelsPrime = (oPixels + VLEN-1) / VLEN;
  mkldnn::impl::parallel_nd(batch, group, outChannelGroupPrime, oPixelsPrime,
              [&](int n, int g, int kPrime, int opPrime) {
    // int k = 16 * kPrime + kDone;
    // int64_t op = VLEN * opPrime;

    vednnConvolutionForward_direct_dil1_str1_padsame_ker3_T_subkernel(
      pParamIn, pDataIn, pParamKernel, pDataKernel, pParamBias, pDataBias,
      pParamConv, pParamOut, pDataOut, n, g, kPrime, opPrime);

#if DEBUG_THREADS
          int ithr = omp_get_thread_num();
          int nthr = omp_get_num_threads();
          if(ithr < RESERVE_THREADS) {
            ithrUsed[ithr] = 1;
            nthrUsed[ithr] = nthr;
          }
#endif

  });  /* kernel of parallel_nd() */

#if DEBUG_THREADS
#ifdef VEDNN_USE_OPENMP
  std::cout << "VEDNN_USE_OPENMP is defined." << std::endl;
#else
  std::cout << "VEDNN_USE_OPENMP is not defined." << std::endl;
#endif
  char *e = getenv("OMP_NUM_THREADS");
  if(e)
    std::cout << "OMP_NUM_THREADS is defined and is " << e << std::endl;
  else
    std::cout << "OMP_NUM_THREADS is not defined." << std::endl;
#ifdef SXAURORA
  std::cout << "SXAURORA is defined." << std::endl;
#else
  std::cout << "SXAURORA is not defined." << std::endl;
#endif
#ifdef OMP
  std::cout << "OMP is defined." << std::endl;
#else
  std::cout << "OMP is not defined." << std::endl;
#endif

  for(int i = 0; i < RESERVE_THREADS; ++i) {
    std::cout << "Thread " << i << ": used: " << ithrUsed[i] << " total ";
    std::cout << nthrUsed[i] << std::endl;
  }
#endif /* DEBUG_THREADS */

  return VEDNN_SUCCESS;
}

}
