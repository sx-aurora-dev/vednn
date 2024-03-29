/*******************************************************************************
* Copyright 2016-2018 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#ifndef CPU_JIT_GEMM_CONVOLUTION_UTILS_HPP
#define CPU_JIT_GEMM_CONVOLUTION_UTILS_HPP

#include "c_types_map.hpp"
//#include "cpu_convolution_pd.hpp"
//#include "cpu_engine.hpp"
// This jit_*.hpp file is also used for vanilla, with heavily pruned conf_t structures
//#include "jit_primitive_conf.hpp"
#include "memory_desc_wrapper.hpp"

#include "../vgemm/mkldnn_thread.hpp"
#include "scratchpad.hpp"
#include "conv_primitive_conf.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

namespace jit_gemm_convolution_utils {

    void im2col_3d(jit_gemm_conv_conf_t const& jcp, const float *im, float *col,
        int od);
    void im2col(jit_gemm_conv_conf_t const& jcp, const float *im, float *col);
    void im2col_u8(jit_gemm_conv_conf_t const& jcp, const uint8_t *im, uint8_t *col);
    void col2im_s32(jit_gemm_conv_conf_t const& jcp, const int32_t *col, int32_t *im);
    void col2im_3d(jit_gemm_conv_conf_t const& jcp, const float *col, float *im,
        int od);
    void col2im(jit_gemm_conv_conf_t const& jcp, const float *col, float *im);

    // note: gemm convolution utils don't care about bias
    void init_conf(jit_gemm_conv_conf_t &jcp,
        const convolution_desc_t &cd, const memory_desc_wrapper &src_d,
        const memory_desc_wrapper &weights_d, const memory_desc_wrapper &dst_d,
        int max_threads, bool with_relu = false, float relu_negative_slope = -1.0);

    // XXX jcp is unused, and size is not same as jcp.im2col_sz ? CHECKME
    status_t prepare_scratchpad(jit_gemm_conv_conf_t const& jcp,
                scratchpad_t **col_scratchpad_, size_t size, const int nthr);

    void bwd_weights_balance(int ithr, int nthr,
        int ngroups, int mb, int &ithr_g, int &nthr_g, int &ithr_mb,
            int &nthr_mb);
    void bwd_weights_reduction_par(int ithr, int nthr,
        jit_gemm_conv_conf_t const& jcp,
	const float *weights_reduce_ws, float *weights);
};

}
}
}

#endif
