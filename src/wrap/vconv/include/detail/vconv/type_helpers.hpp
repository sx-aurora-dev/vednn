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

#ifndef TYPE_HELPERS_HPP
#define TYPE_HELPERS_HPP

#include <assert.h>
#include <math.h>
#include <string.h> // memset

//#include "mkldnn.h"
//#include "c_types_map.hpp"
#include "mkldnn_traits.hpp"    // prec_traits etc.
#include "../vgemm/nstl.hpp"
#include "../vgemm/utils.hpp"
//#include "math_utils.hpp"

#include <type_traits>

namespace mkldnn {
namespace impl {

template <typename T>
status_t safe_ptr_assign(T * &lhs, T* rhs) {
    if (rhs == nullptr) return status::out_of_memory;
    lhs = rhs;
    return status::success;
}

template <typename T, typename U> struct is_subset
{ static constexpr bool value = false; };
template <typename T> struct is_subset<T, T>
{ static constexpr bool value = true; };
template <typename T> struct is_subset<T,
         typename utils::enable_if<nstl::is_integral<T>::value, float>::type>
{ static constexpr bool value = true; };
#define ISSPEC(t1, t2) template <> \
    struct is_subset<t1, t2> { static constexpr bool value = true; }
ISSPEC(int16_t, int32_t);
ISSPEC(int8_t, int32_t);
ISSPEC(uint8_t, int32_t);
ISSPEC(int8_t, int16_t);
ISSPEC(uint8_t, int16_t);
#undef ISSPEC

namespace types {

inline size_t data_type_size(data_type_t data_type) {
    using namespace data_type;
    switch (data_type) {
    case f32: return sizeof(prec_traits<f32>::type);
    case s32: return sizeof(prec_traits<s32>::type);
    case s16: return sizeof(prec_traits<s16>::type);
    case s8: return sizeof(prec_traits<s8>::type);
    case u8: return sizeof(prec_traits<u8>::type);
    case data_type::undef:
    default: assert(!"unknown data_type");
    }
    return 0; /* not supposed to be reachable */
}

inline memory_format_t flat_memory_format(int ndims) {
    switch (ndims) {
    case 1: return memory_format::x;
    case 2: return memory_format::nc;
    case 4: return memory_format::nchw;
    case 5: return memory_format::ncdhw;
    default: return memory_format::undef;
    }
    return memory_format::undef;
}

inline memory_format_t format_normalize(const memory_format_t fmt) {
    using namespace memory_format;
    /* FIXME: double blocked formats are special cases -- the blocking
     *        structure doesn't correctly describe memory layout (wrt
     *        the strides within blocks). Though as long as the code
     *        uses memory_desc_wrapper::off() or explicit offset
     *        calculations everything should be fine. */
    // [ejk] internal option -We,--pending_instantions=100 for __ve compiler
    //       will increase the allowed recursion depth for functions.
    const bool is_blocked = utils::one_of(fmt, blocked,
            x,
            nc,
            nchw,
            nhwc,
            chwn,
            nChw8c,
            nChw16c,
            ncdhw,
            ndhwc,
            nCdhw8c,
            nCdhw16c,
            oi,
            io,
            oihw,
            ihwo,
            hwio,
            dhwio,
            oidhw,
            OIdhw8i8o,
            OIdhw8o8i,
            Odhwi8o,
            OIdhw16i16o,
            OIdhw16o16i,
            Oidhw16o,
            Odhwi16o,
            oIhw8i,
            oIhw16i,
            oIdhw8i,
            oIdhw16i,
            OIhw8i8o,
            OIhw16i16o,
            OIhw4i16o4i,
            OIhw8i16o2i,
            OIdhw8i16o2i,
            OIhw8o16i2o,
            OIhw8o8i,
            OIhw16o16i,
            IOhw16o16i,
            Oihw16o,
            Ohwi8o,
            Ohwi16o,
            goihw,
            hwigo,
            gOIhw8i8o,
            gOIhw16i16o,
            gOIhw4i16o4i,
            gOIhw8i16o2i,
            gOIdhw8i16o2i,
            gOIhw8o16i2o,
            gOIhw8o8i,
            gOIhw16o16i,
            gIOhw16o16i,
            gOihw16o,
            gOhwi8o,
            gOhwi16o,
            Goihw8g,
            Goihw16g,
            goidhw,
            gOIdhw8i8o,
            gOIdhw8o8i,
            gOdhwi8o,
            gOIdhw16i16o,
            gOIdhw16o16i,
            gOidhw16o,
            gOdhwi16o,
            ntc,
            tnc,
            ldsnc,
            ldigo,
            ldgoi,
            ldgo);
    return is_blocked ? blocked : fmt;
}

inline bool is_format_double_blocked(memory_format_t fmt) {
    using namespace memory_format;
    return utils::one_of(OIhw8i16o2i, OIdhw8i16o2i, OIhw8o16i2o, OIhw4i16o4i,
            gOIhw8i16o2i, gOIdhw8i16o2i, gOIhw8o16i2o,gOIhw4i16o4i);
}

inline bool blocking_desc_is_equal(const blocking_desc_t &lhs,
        const blocking_desc_t &rhs, int ndims = TENSOR_MAX_DIMS) {
    using mkldnn::impl::utils::array_cmp;
    return lhs.offset_padding == rhs.offset_padding
        && array_cmp(lhs.block_dims, rhs.block_dims, ndims)
        && array_cmp(lhs.strides[0], rhs.strides[0], ndims)
        && array_cmp(lhs.strides[1], rhs.strides[1], ndims)
        && array_cmp(lhs.padding_dims, rhs.padding_dims, ndims)
        && array_cmp(lhs.offset_padding_to_data, rhs.offset_padding_to_data,
                ndims);
}

inline bool wino_desc_is_equal(const wino_data_t &lhs,
    const wino_data_t &rhs) {
    return lhs.wino_format == rhs.wino_format
        && lhs.alpha == rhs.alpha
        && lhs.ic == rhs.ic
        && lhs.oc == rhs.oc
        && lhs.ic_block == rhs.ic_block
        && lhs.oc_block == rhs.oc_block
        && lhs.ic2_block == rhs.ic2_block
        && lhs.oc2_block == rhs.oc2_block
        && lhs.r == rhs.r;
}

inline bool operator==(const memory_desc_t &lhs, const memory_desc_t &rhs) {
    // XXX assert(lhs.primitive_kind == mkldnn::impl::primitive_kind::memory);
    // XXX assert(rhs.primitive_kind == mkldnn::impl::primitive_kind::memory);
    bool base_equal = true
        && lhs.ndims == rhs.ndims
        && mkldnn::impl::utils::array_cmp(lhs.dims, rhs.dims, lhs.ndims)
        && lhs.data_type == rhs.data_type
        && lhs.format == rhs.format; /* FIXME: normalize format? */
    if (!base_equal) return false;
    if (lhs.format == memory_format::blocked)
        return blocking_desc_is_equal(lhs.layout_desc.blocking,
                rhs.layout_desc.blocking, lhs.ndims);
    else if (lhs.format == memory_format::wino_fmt)
        return wino_desc_is_equal(lhs.layout_desc.wino_desc,
            rhs.layout_desc.wino_desc);
    return true;
}

inline bool operator!=(const memory_desc_t &lhs, const memory_desc_t &rhs) {
    return !operator==(lhs, rhs);
}

inline memory_desc_t zero_md() {
    static_assert(std::is_pod<mkldnn_memory_desc_t>::value,"Failed: is_pod C++ default constructor for mkldnn_memory_desc_t");
    // no static_assert(std::is_trivially_constructible<mkldnn_memory_desc_t,void>::value,"Failed: is_trivially_constructible mkldnn_memory_desc_t");
#if defined(__ve) // zero-initialization is garbage-initialization, inside non-main() function!
    auto zero = mkldnn_memory_desc_t{}; // NB {} forces value-initialization (to zero)
    //mkldnn_memory_desc_t zero = {mkldnn_memory,0};
    // also OK, but fails -Werror because mkl-dnn mandates ALL values specified
    // (should fail to compile with -Werror)
#else
    auto zero = mkldnn_memory_desc_t();
#endif
    //static int garbage = -1;
    //memset(&zero,--garbage,sizeof(memory_desc_t));
    //memset(&zero,0,sizeof(memory_desc_t));
    // XXX zero.primitive_kind = primitive_kind::memory;
    return zero;
}

inline bool is_zero_md(const memory_desc_t *md) {
    return md == nullptr || *md == zero_md();
}

#if 0
inline status_t set_default_format(memory_desc_t &md, memory_format_t fmt) {
    return mkldnn_memory_desc_init(&md, md.ndims, md.dims, md.data_type, fmt);
}
#endif

inline data_type_t default_accum_data_type(data_type_t src_dt,
        data_type_t dst_dt) {
    using namespace utils;
    using namespace data_type;

    if (one_of(f32, src_dt, dst_dt)) return f32;
    if (one_of(s32, src_dt, dst_dt)) return s32;
    if (one_of(s16, src_dt, dst_dt)) return s32;

    if (one_of(s8, src_dt, dst_dt) || one_of(u8, src_dt, dst_dt)) return s32;

    assert(!"unimplemented use-case: no default parameters available");
    return dst_dt;
}

inline data_type_t default_accum_data_type(data_type_t src_dt,
        data_type_t wei_dt, data_type_t dst_dt, prop_kind_t prop_kind) {
    using namespace utils;
    using namespace data_type;
    using namespace prop_kind;

    /* prop_kind doesn't matter */
    if (everyone_is(f32, src_dt, wei_dt, dst_dt)) return f32;

    if (one_of(prop_kind, forward_training, forward_inference)) {
        if (src_dt == s16 && wei_dt == s16 && dst_dt == s32)
            return s32;
        if (src_dt == u8 && wei_dt == s8 && one_of(dst_dt, f32, s32, s8, u8))
            return s32;
    } else if (prop_kind == backward_data) {
        if (src_dt == s32 && wei_dt == s16 && dst_dt == s16)
            return s32;
        if (one_of(src_dt, f32, s32, s8, u8) && wei_dt == s8 && dst_dt == u8)
            return s32;
    } else if (prop_kind == backward_weights) {
        if (src_dt == s16 && wei_dt == s32 && dst_dt == s16)
            return s32;
    }

    assert(!"unimplemented use-case: no default parameters available");
    return dst_dt;
}

}
}
}

#include "memory_desc_wrapper.hpp"

#endif

// vim: et ts=4 sw=4 cindent cino=^=l0,\:0,N-s
