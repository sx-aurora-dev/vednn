#ifndef VEDNN_DEF_HPP
#define VEDNN_DEF_HPP
#include "vednn-def.h"
#include <cassert>

// from gen-dnn sources:
//#include "mkldnn_thread.hpp"
#include "utils.hpp"            // THREAD_LOCAL mkldnn::impl::malloc/free (with alignment)

namespace vednn {
namespace scratchpad {

/** vednn C++ layers should use these to obtain the scratchpad memory addresses.
 * The extern "C" functions of same name are not inlined. */
char* vednn_scratchpad(size_t bytes);
float* vednn_scratchpad_float_ones(size_t const floats);

/** borrowed from mkldnn scratchpad, Apache 2.0 license.
 * Overkill, for vednn, because we only have the 'global' type of scratchpad?
 */
struct scratchpad_t {
    virtual ~scratchpad_t() {}
    virtual char *get() const = 0;
};

/** Implementation of the scratchpad_t interface that uses a global
 *  scratchpad. */
struct global_scratchpad_t : public scratchpad_t {
    global_scratchpad_t(size_t bytes) {
        using mkldnn::impl::malloc;
        using mkldnn::impl::free;
        bytes = (bytes+15U)/16U*16U;
        if (bytes > size_) {
            if (scratchpad_ != nullptr) free(scratchpad_);
            size_ = bytes;
            /* Allocating memory buffers on a page boundary to reduce TLB/page misses */
            const size_t page_size = 2097152;
            scratchpad_ = (char *) malloc(bytes, page_size);
            printf(" vednn global_scratchpad[ %lu bytes ] @ %p\n",
                    (long unsigned)bytes, (void*)scratchpad_);
            assert(scratchpad_ != nullptr);
        }
        reference_count_++;
    }

    ~global_scratchpad_t() {
        using mkldnn::impl::free;
        reference_count_--;
        if (reference_count_ == 0) {
            free(scratchpad_);
            scratchpad_ = nullptr;
            size_ = 0;
        }
    }

    virtual char *get() const {
        return scratchpad_;
    }

private:
    THREAD_LOCAL static char *scratchpad_;
    THREAD_LOCAL static size_t size_;
    THREAD_LOCAL static unsigned int reference_count_;
    //OMP(threadprivate(scratchpad_, size_, reference_count_))//;
};

/* Implementation of the scratchpad_t interface that uses a global
 * scratchpad that grows by filling with 1.0f values. The constructor
 * size is number-of-floats (not bytes). */
struct global_scratchpad_float_ones_t : public scratchpad_t {
    global_scratchpad_float_ones_t(size_t floats) {
        using mkldnn::impl::malloc;
        using mkldnn::impl::free;
        floats = (floats+3U)/4U*4U;
        if (floats > size_) {
            if (scratchpad_ != nullptr) free(scratchpad_);
            size_ = floats;
            /* Allocating memory buffers on a page boundary to reduce TLB/page misses */
            const size_t page_size = 2097152;
            void* sp = malloc(floats*sizeof(float), page_size);
            printf(" vednn global_scratchpad_float_ones_t[ %lu bytes ] @ %p\n",
                    (long unsigned)(floats*sizeof(float)), sp);
            scratchpad_ = (char *) sp;
            assert(scratchpad_ != nullptr);
            // fill with 1.0 whenever resized
            float *spf = (float*) sp;
            for(size_t i=0U; i<floats; ++i){
                spf[i] = 1.0f;
            }
        }
        ++reference_count_;
    }

    ~global_scratchpad_float_ones_t() {
        using mkldnn::impl::free;
        --reference_count_;
        if (reference_count_ == 0) {
            free(scratchpad_);
            scratchpad_ = nullptr;
            size_ = 0;
        }
    }

    virtual char *get() const {
        return scratchpad_;
    }

private:
    THREAD_LOCAL static char *scratchpad_;
    THREAD_LOCAL static size_t size_;
    THREAD_LOCAL static unsigned int reference_count_;
    //OMP(threadprivate(scratchpad_, size_, reference_count_))//;
};

extern global_scratchpad_t             *global_scratchpad;
extern global_scratchpad_float_ones_t  *global_scratchpad_float_ones;

inline char* vednn_scratchpad(size_t bytes){
    return global_scratchpad->get();
}
inline float* vednn_scratchpad_float_ones(size_t const floats){
    return reinterpret_cast<float*>(global_scratchpad_float_ones->get());
}


}}//vednn::scratchpad::
// vim: et ts=4 sw=4 cindent cino=^=lg0,\:0,N-s
#endif // VEDNN_DEF_HPP
