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

#include "c_types_map.hpp"
#include "gemm_convolution.hpp"
#include "utils.hpp"
#include "type_helpers.hpp"
#include "mkldnn_thread.hpp"

#ifndef MKLDNN_ENABLE_CONCURRENT_EXEC
#define MKLDNN_ENABLE_CONCURRENT_EXEC
#endif
#include "scratchpad.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

#if ! USE_MKL && ! USE_CBLAS // provide empty stubs (init always will say "NO")
//#pragma warning "gemm_convolution stubs only -- (no MKL or CBLAS)"

template <bool with_relu>
void _gemm_convolution_fwd_t<with_relu>::execute_forward() {}

#if PARTIAL >= 1
void gemm_convolution_bwd_data_t::execute_backward_data() {}
#endif

#if PARTIAL >= 2
void gemm_convolution_bwd_weights_t::execute_backward_weights() {}
#endif

#else // some sort of gemm (jit? cblas?) is available

using namespace mkldnn::impl::status;
using namespace mkldnn::impl::memory_format;
using namespace mkldnn::impl::utils;

#if 1
template <bool with_relu>
void _gemm_convolution_fwd_t<with_relu>::execute_forward() {
    //auto src = reinterpret_cast<const data_t *>(this->input_memory(0));
    //auto weights = reinterpret_cast<const data_t *>(this->input_memory(1));
    //auto bias = reinterpret_cast<const data_t *>(this->input_memory(2));
    //auto dst = reinterpret_cast<data_t*>(this->memory());
    auto src    = pDataIn;
    auto weights= pDataKernel;
    auto bias   = pDataBias;
    auto dst    = pDataOut;

    jit_gemm_conv_conf_t const& jcp = this->conf_.jcp_;

    const int M = jcp.os * jcp.od;
    const size_t src_step = jcp.ic * jcp.ih * jcp.iw * jcp.id;
    const size_t dst_step = jcp.oc * M;
    const size_t weights_g_size = jcp.ic * jcp.oc * jcp.ks;

    const int K = jcp.ic * jcp.ks;
    const int N = jcp.oc;
    const int m = jcp.os;
    const int LDA = jcp.im2col_sz ? m : M;

    const auto &post_ops = conf_.attr()->post_ops_;

    float nslope = jcp.with_relu ? jcp.relu_negative_slope : 0.f;
    int entry_idx = -1;
    for (int idx = 0; idx < post_ops.len_; ++idx) {
        const auto &e = post_ops.entry_[idx];
        if (e.is_relu(true, false)) {
            entry_idx = idx;
            nslope = post_ops.entry_[entry_idx].eltwise.alpha;
            break;
        }
    }
    const bool do_relu = jcp.with_relu || entry_idx >= 0;

    const data_t one = 1.0;

    data_t *col = (jcp.im2col_sz)
        ? (data_t *)this->scratchpad_->get()
        : nullptr;

    const size_t work_amount = jcp.ngroups * jcp.mb * jcp.od;
    OMP(parallel num_threads(jcp.nthr))//;
    {
        const int ithr = omp_get_thread_num();
        const int nthr = omp_get_num_threads();

        data_t *_col = col + (ptrdiff_t)ithr * jcp.im2col_sz;

        OMP(parallel for if(jcp.nthr == 1))
        for (ptrdiff_t i = 0; i < jcp.im2col_sz; ++i) _col[i] = (data_t)0;

        int g{0}, n{0}, od{0};
        size_t start = 0, end = 0;

        balance211(work_amount, nthr, ithr, start, end); // sets start and end
        nd_iterator_init(start, g, jcp.ngroups, n, jcp.mb, od, jcp.od);

        for (size_t iwork = start; iwork < end; ++iwork) {
            const data_t *_src = src + (n * jcp.ngroups + g) * src_step;
            const data_t *_weights = weights + g * weights_g_size;
            data_t *_dst = dst + (n * jcp.ngroups + g) * dst_step;

            if (jcp.im2col_sz) {
                if (jcp.id == 1)
                    jit_gemm_convolution_utils::im2col(jcp, _src, _col);
                else
                    jit_gemm_convolution_utils::im2col_3d(jcp, _src, _col, od);
            }

            extended_sgemm("N", "N", &m, &N, &K, &one,
                    jcp.im2col_sz ? _col : _src + od * m, &LDA, _weights, &K,
                    &this->beta_, _dst + od * m, &M);

            if (jcp.with_bias || do_relu) {
                data_t *d = _dst + od * m, b = 0.0;
                for (int oc = 0; oc < jcp.oc; ++oc) {
                    if(jcp.with_bias) b = bias[g * jcp.oc + oc];
                    for (int oS = 0; oS < m; ++oS) {
                        if (jcp.with_bias) d[oS] += b;
                        if (do_relu && d[oS] < 0)
                            d[oS] *= nslope;
                    }
                    d += M;
                }
            }
            nd_iterator_step(g, jcp.ngroups, n, jcp.mb, od, jcp.od);
        }
    }
}
template struct _gemm_convolution_fwd_t<true>;
template struct _gemm_convolution_fwd_t<false>;
#endif

#if PARTIAL >= 1
void gemm_convolution_bwd_data_t::execute_backward_data() {
    //auto diff_dst = reinterpret_cast<const data_t *>(this->input_memory(0));
    //auto weights = reinterpret_cast<const data_t *>(this->input_memory(1));
    //auto diff_src = reinterpret_cast<data_t*>(this->memory());
    auto diff_dst = pDataGradOut;
    auto weights  = pDataKernel;
    auto diff_src = pDataGradIn;

    jit_gemm_conv_conf_t &jcp = this->conf_.jcp_;

    const int M = jcp.os * jcp.od;
    const size_t src_step = jcp.ic * jcp.ih * jcp.iw * jcp.id;
    const size_t dst_step = jcp.oc * M;
    const size_t weights_g_size = jcp.ic * jcp.oc * jcp.ks;

    const int m = jcp.os;
    const int K = jcp.oc;
    const int N = jcp.ic * jcp.ks;
    const int LDC = jcp.im2col_sz ? m : M;
    const data_t zero = 0.0, one = 1.0;

    data_t *col = (jcp.im2col_sz)
        ? (data_t *)this->scratchpad_->get()
        : nullptr;

    const size_t work_amount = (size_t)jcp.ngroups * jcp.mb;
    OMP(parallel num_threads(jcp.nthr))
    {
        const int ithr = omp_get_thread_num();
        const int nthr = omp_get_num_threads();

        data_t *_col = col + (ptrdiff_t)ithr * jcp.im2col_sz;

        OMP(parallel for if(jcp.nthr == 1))//;
        for (ptrdiff_t i = 0; i < jcp.im2col_sz; ++i) _col[i] = (data_t)0;

        if (jcp.id > 1) {
            ptrdiff_t diff_src_sz = (ptrdiff_t)(work_amount * src_step);
            OMP(for)//;
            for (ptrdiff_t i = 0; i < diff_src_sz; ++i)
                diff_src[i] = 0.;
        }

        int g{0}, n{0};
        size_t start = 0, end = 0;
        balance211(work_amount, nthr, ithr, start, end);
        nd_iterator_init(start, g, jcp.ngroups, n, jcp.mb);
        for (size_t iwork = start; iwork < end; ++iwork) {

            data_t *_diff_src = diff_src + (n * jcp.ngroups + g)*src_step;
            const data_t *_weights = weights + g * weights_g_size;
            for (int od = 0; od < jcp.od; ++od) {
                const data_t *_diff_dst = diff_dst + (n * jcp.ngroups + g)
                    *dst_step + od * m;

                extended_sgemm("N", "T", &m, &N, &K, &one, _diff_dst, &M,
                    _weights, &N, &zero,
                    jcp.im2col_sz ? _col:_diff_src + od * m, &LDC);

                if (jcp.im2col_sz) {
                    if (jcp.id == 1)
                        jit_gemm_convolution_utils::col2im(jcp, _col,
                            _diff_src);
                    else
                        jit_gemm_convolution_utils::col2im_3d(jcp, _col,
                            _diff_src, od);
                }
            }
            nd_iterator_step(g, jcp.ngroups, n, jcp.mb);
        }
    }
}
#endif
#if PARTIAL >= 2
// ncc issue with multiple OMP clauses in same function :(
void gemm_convolution_bwd_weights_t::execute_backward_weights() {
    //auto src = reinterpret_cast<const data_t *>(this->input_memory(0));
    //auto diff_dst = reinterpret_cast<const data_t *>(this->input_memory(1));
    //auto diff_weights = reinterpret_cast<data_t*>(this->memory(0));
    //auto diff_bias = reinterpret_cast<data_t *>(this->memory(1));
    auto src            = pDataIn;
    auto diff_dst       = pDataGradOut;
    auto diff_weights   = pDataGradKernel;
#if !VE_OPENMP_BUG
    auto diff_bias      = pDataGradBias; // can be null (libvednn API does not support)
#endif

    jit_gemm_conv_conf_t &jcp = this->conf_.jcp_;
    const int K = jcp.os * jcp.od;
    const size_t src_step = jcp.ic * jcp.ih * jcp.iw * jcp.id;
    const size_t dst_step = jcp.oc * K;
    const size_t weights_g_size = jcp.ic * jcp.oc * jcp.ks;

    const int k = jcp.os;
    const int N = jcp.oc;
    const int M = jcp.ic * jcp.ks;
    const int LDA = jcp.im2col_sz ? k : K;
    const data_t zero = 0.0, one = 1.0;

    data_t *col = nullptr, *wei_reduction = nullptr;
    ptrdiff_t wei_offset = 0;
    if (jcp.im2col_sz) {
        col = (data_t *)this->scratchpad_->get();
        wei_offset = jcp.im2col_sz * jcp.nthr;
    }
    if (jcp.need_wei_reduction)
        wei_reduction = (data_t *)this->scratchpad_->get() + wei_offset;

    OMP(parallel num_threads(jcp.nthr))
    {
        const int ithr = omp_get_thread_num();
        const int nthr = omp_get_num_threads();

        int ithr_g, nthr_g, ithr_mb, nthr_mb;
        size_t g_start{0}, g_end{0}, mb_start{0}, mb_end{0};

        jit_gemm_convolution_utils::bwd_weights_balance(ithr, nthr,
                jcp.ngroups, jcp.mb, ithr_g, nthr_g, ithr_mb, nthr_mb);

        const int need_reduction = nthr_mb != 1;

        if (ithr_g != -1 && ithr_mb != -1) {
            balance211((size_t)jcp.ngroups, nthr_g, ithr_g, g_start, g_end);
            balance211((size_t)jcp.mb, nthr_mb, ithr_mb, mb_start, mb_end);

            assert(implication((g_end - g_start) > 1, need_reduction == 0));

            data_t *_col = col + (ptrdiff_t)ithr * jcp.im2col_sz;
            data_t *weights_reduce_base = wei_reduction
                    + ithr_g * nthr_mb * weights_g_size;
            data_t *weights_reduce = weights_reduce_base
                    + ithr_mb * weights_g_size;

            OMP(parallel for if(jcp.nthr == 1))//;
            for (ptrdiff_t i = 0; i < jcp.im2col_sz; ++i) _col[i] = (data_t)0;

            for (size_t g = g_start; g < g_end; ++g) {
                data_t *_diff_weights = need_reduction
                        ? weights_reduce : (diff_weights + g * weights_g_size);
                for (size_t mb = mb_start; mb < mb_end; ++mb) {
                    const data_t *_src = src + (mb*jcp.ngroups+g)*src_step;
                    for (int od = 0; od < jcp.od; ++od) {
                    const data_t *_diff_dst = diff_dst
                            + (mb*jcp.ngroups+g)*dst_step + od * k;

                    if (jcp.im2col_sz) {
                        if (jcp.id == 1)
                            jit_gemm_convolution_utils::im2col(jcp, _src, _col);
                        else
                            jit_gemm_convolution_utils::im2col_3d(jcp, _src,
                                _col, od);
                    }

                    extended_sgemm(
                        "T", "N", &M, &N, &k, &one,
                        jcp.im2col_sz ? _col : _src + od * k,
                        &LDA, _diff_dst, &K,
                        mb == mb_start && od == 0 ? &zero : &one,
                        _diff_weights, &M);
                    }
                }
            }
            if (need_reduction) {
                OMP(barrier)//;
                data_t *weights_base = diff_weights + g_start * weights_g_size;
                jit_gemm_convolution_utils::bwd_weights_reduction_par(
                    ithr_mb, nthr_mb, jcp, weights_reduce_base, weights_base);
            }
        } else
            if (need_reduction) {
                OMP(barrier)//;
            }
    } 
#if !VE_OPENMP_BUG
    // XXX libvednn compat allows pDataGradBias to be nullptr :(
    //     so you may need to manually do this update.
    if (jcp.with_bias && pDataGradBias!=nullptr) {
        // sum the output gradients over the image pixels
        // in each output channel.
        //   The gen-dnn vednnx code may have a faster VE way to do this.
        const size_t work_amount = jcp.ngroups * jcp.oc;
        OMP(parallel)//;
        {
            const int ithr = omp_get_thread_num();
            const int nthr = omp_get_num_threads();
            int g{0}, oc{0};
            size_t start = 0, end = 0;
            balance211(work_amount, nthr, ithr, start, end);
            nd_iterator_init(start, g, jcp.ngroups, oc, jcp.oc);
            for (size_t iwork = start; iwork < end; ++iwork) {
                data_t db = 0;
                size_t offset_ = (size_t)g*dst_step + (size_t)oc * K;
                for (int mb = 0; mb < jcp.mb; ++mb)
                {
                    size_t offset = offset_ + (size_t)mb*jcp.ngroups*dst_step;
                    // this is DENSE, so may not be fastest way to write this.
                    for (int od = 0; od < jcp.od; ++od)
                    for (int oh = 0; oh < jcp.oh; ++oh)
#if defined(__ve)
                    ShortLoopTest()// not this is compelely orthogonal to omp reduce;
#else
                    // actually this is a thread local reduction, so why is omp reduction even used?
                    PRAGMA_OMP_SIMD(reduction(+:db))
#endif
                    for (int ow = 0; ow < jcp.ow; ++ow)
                    {
                        db += diff_dst[offset];
                        offset ++;
                    }
                }
                //diff_bias[diff_bias_d.off(g*jcp.oc+oc)] = db;
                diff_bias[g*jcp.oc+oc] = db;
                nd_iterator_step(g, jcp.ngroups, oc, jcp.oc);
            }
        }
    }
#else
    if (jcp.with_bias && pDataGradBias != nullptr){
        execute_backward_weights_bias();
    }
#endif
}
#if VE_OPENMP_BUG
void gemm_convolution_bwd_weights_t::execute_backward_weights_bias() {
    ////auto src = reinterpret_cast<const data_t *>(this->input_memory(0));
    //auto diff_dst = reinterpret_cast<const data_t *>(this->input_memory(1));
    ////auto diff_weights = reinterpret_cast<data_t*>(this->memory(0));
    //auto diff_bias = reinterpret_cast<data_t *>(this->memory(1));
    auto diff_dst  = pDataGradOut;
    auto diff_bias = pDataGradBias; // can be null (libvednn API does not support)

    jit_gemm_conv_conf_t &jcp = this->conf_.jcp_;
    const int K = jcp.os * jcp.od;
    //const size_t src_step = jcp.ic * jcp.ih * jcp.iw * jcp.id;
    const size_t dst_step = jcp.oc * K;
    //const size_t weights_g_size = jcp.ic * jcp.oc * jcp.ks;

    //const int k = jcp.os;
    //const int N = jcp.oc;
    //const int M = jcp.ic * jcp.ks;
    //const int LDA = jcp.im2col_sz ? k : K;
    //const data_t zero = 0.0, one = 1.0;

    //data_t *col = nullptr, *wei_reduction = nullptr;
    //ptrdiff_t wei_offset = 0;
    //if (jcp.im2col_sz) {
    //    col = (data_t *)this->scratchpad_->get();
    //    wei_offset = jcp.im2col_sz * jcp.nthr;
    //}
    //if (jcp.need_wei_reduction)
    //    wei_reduction = (data_t *)this->scratchpad_->get() + wei_offset;
#if 0
    // weights update, omp loop -- see 
#endif
#if 1
    if (jcp.with_bias) {
        const size_t work_amount = jcp.ngroups * jcp.oc;
        OMP(parallel)//;
        {
            const int ithr = omp_get_thread_num();
            const int nthr = omp_get_num_threads();
            int g{0}, oc{0};
            size_t start = 0, end = 0;
            balance211(work_amount, nthr, ithr, start, end);
            nd_iterator_init(start, g, jcp.ngroups, oc, jcp.oc);
            for (size_t iwork = start; iwork < end; ++iwork) {
                data_t db = 0;
                size_t offset_ = (size_t)g*dst_step + (size_t)oc * K;
                for (int mb = 0; mb < jcp.mb; ++mb)
                {
                    size_t offset = offset_ + (size_t)mb*jcp.ngroups*dst_step;
                    for (int od = 0; od < jcp.od; ++od)
                    for (int oh = 0; oh < jcp.oh; ++oh)
#if defined(__ve)
                    _Pragma("_NEC shortloop_reduction")// not this is compelely orthogonal to omp reduce;
#else
                    // actually this is a thread local reduction, so why is omp reduction even used?
                    PRAGMA_OMP_SIMD(reduction(+:db))
#endif
                    for (int ow = 0; ow < jcp.ow; ++ow)
                    {
                        db += diff_dst[offset];
                        offset ++;
                    }
                }
                //diff_bias[diff_bias_d.off(g*jcp.oc+oc)] = db;
                diff_bias[g*jcp.oc+oc] = db;
                nd_iterator_step(g, jcp.ngroups, oc, jcp.oc);
            }
        }
    }
#endif
}
#endif
#endif
#endif

}}}//mkldnn::impl::cpu
#if VCONV_STANDALONE
void vconv_gemm_fwd(
        mkldnn::impl::cpu::jit_gemm_conv_conf_t const& jcp,
        data_t* src,     //pDataIn
        data_t* weights, //pDataKernel
        data_t* bias,    //pDataBias
        data_t* dst,     //pDataOut
        mkldnn::impl::scratchpad_t& scratchpad, // from create_scratchpad(any_size,true/*per_thread*/)
        mkldnn::impl::post_ops_t const* const post_ops_ /*= nullptr*/
        )
{
    using namespace mkldnn::impl::status;
    using namespace mkldnn::impl::memory_format;
    using namespace mkldnn::impl::utils;
    using namespace mkldnn::impl;
    using namespace mkldnn::impl::cpu;

    //auto src    = pDataIn;
    //auto weights= pDataKernel;
    //auto bias   = pDataBias;
    //auto dst    = pDataOut;

    //jit_gemm_conv_conf_t const& jcp = this->conf_.jcp_;

    const int M = jcp.os * jcp.od;
    const size_t src_step = jcp.ic * jcp.ih * jcp.iw * jcp.id;
    const size_t dst_step = jcp.oc * M;
    const size_t weights_g_size = jcp.ic * jcp.oc * jcp.ks;

    const int K = jcp.ic * jcp.ks;
    const int N = jcp.oc;
    const int m = jcp.os;
    const int LDA = jcp.im2col_sz ? m : M;

    float nslope = jcp.with_relu ? jcp.relu_negative_slope : 0.f;
    const data_t one = 1.0, zero = 0.0;
    data_t beta_ = zero;

    int entry_idx = -1;
    if(post_ops_){
        //const auto &post_ops = conf_.attr()->post_ops_;
        const auto& post_ops = *post_ops_;
        for (int idx = 0; idx < post_ops.len_; ++idx) {
            const auto &e = post_ops.entry_[idx];
            if (e.is_relu(true, false)) {
                entry_idx = idx;
                nslope = post_ops.entry_[entry_idx].eltwise.alpha;
                break;
            }
        }
        if(post_ops.find(primitive_kind::sum) >= 0) // combine w/ prev loop?
            beta_ = one;
    }
    const bool do_relu = jcp.with_relu || entry_idx >= 0;

    //data_t *col = (jcp.im2col_sz)
    //    ? (data_t *)this->scratchpad_->get()
    //    : nullptr;
    data_t *col = (jcp.im2col_sz
            ? reinterpret_cast<data_t*>(scratchpad.get())
            : nullptr);

    const size_t work_amount = jcp.ngroups * jcp.mb * jcp.od;
    OMP(parallel num_threads(jcp.nthr))//;
    {
        const int ithr = omp_get_thread_num();
        const int nthr = omp_get_num_threads();

        data_t *_col = col + (ptrdiff_t)ithr * jcp.im2col_sz;

        OMP(parallel for if(jcp.nthr == 1))
        for (ptrdiff_t i = 0; i < jcp.im2col_sz; ++i) _col[i] = (data_t)0;

        int g{0}, n{0}, od{0};
        size_t start = 0, end = 0;

        balance211(work_amount, nthr, ithr, start, end); // sets start and end
        nd_iterator_init(start, g, jcp.ngroups, n, jcp.mb, od, jcp.od);

        for (size_t iwork = start; iwork < end; ++iwork) {
            const data_t *_src = src + (n * jcp.ngroups + g) * src_step;
            const data_t *_weights = weights + g * weights_g_size;
            data_t *_dst = dst + (n * jcp.ngroups + g) * dst_step;

            if (jcp.im2col_sz) {
                if (jcp.id == 1)
                    jit_gemm_convolution_utils::im2col(jcp, _src, _col);
                else
                    jit_gemm_convolution_utils::im2col_3d(jcp, _src, _col, od);
            }

            extended_sgemm("N", "N", &m, &N, &K, &one,
                    jcp.im2col_sz ? _col : _src + od * m, &LDA, _weights, &K,
                    &beta_, //&this->beta_,
                    _dst + od * m, &M);

            if (jcp.with_bias || do_relu) {
                data_t *d = _dst + od * m, b = 0.0;
                for (int oc = 0; oc < jcp.oc; ++oc) {
                    if(jcp.with_bias) b = bias[g * jcp.oc + oc];
                    for (int oS = 0; oS < m; ++oS) {
                        if (jcp.with_bias) d[oS] += b;
                        if (do_relu && d[oS] < 0)
                            d[oS] *= nslope;
                    }
                    d += M;
                }
            }
            nd_iterator_step(g, jcp.ngroups, n, jcp.mb, od, jcp.od);
        }
    }
}
#endif // VCONV_STANDALONE

// vim: et ts=4 sw=4 cindent nopaste ai cino=^=l0,\:0,N-s
