#ifndef VEDNN_DEF_HPP
#define VEDNN_DEF_HPP
#include "vednn-def.h"

// from gen-dnn sources:
//#include "mkldnn_thread.hpp"
#include "utils.hpp"            // THREAD_LOCAL mkldnn::impl::malloc/free (with alignment)

#include <cassert>
#include <cstddef>  // size_t
#include <cstdio>

#ifndef VERBOSE
#define VERBOSE 0
#endif
#ifndef VEDNN_SCRATCH_DBG
#if VERBOSE
#define VEDNN_SCRATCH_DBG(...) do{ printf(__VA_ARGS__); fflush(stdout); }while(0)
#else
#define VEDNN_SCRATCH_DBG(...) do{}while(0)
#endif
#endif

namespace vednn {
namespace scratchpad {

/** vednn C++ layers should use these to obtain the scratchpad memory addresses.
 * The extern "C" functions of same name are not inlined. */
//char* vednn_scratchpad(size_t bytes);
char* vednn_scratchpad_shared(size_t bytes);
char* vednn_scratchpad_tls(size_t bytes);
float* vednn_scratchpad_float_ones(size_t const floats);

/** borrowed from mkldnn scratchpad, Apache 2.0 license.
 * Overkill, for vednn, and introduces extra function calls if used as public API.
 */
struct ScratchpadBase {
    virtual ~ScratchpadBase() {}
    virtual char *get() const = 0;
    virtual float *pfloat() const = 0;
};
// fwd decl
struct ScratchpadMallocFree;
struct ScratchpadShared;
struct ScratchpadTLS;
struct ScratchpadFloatOnes;
// library provides some re-usable scratchpads (use locally, within single layer funcs)
extern ScratchpadShared     *scratchpadShared;
extern ScratchpadTLS        *scratchpadTLS;
extern ScratchpadFloatOnes  *scratchpadFloatOnes;

/** Implementation of the Scratchpad interface that uses malloc/free
 * every time (single-use, non-resizing scratchpad).  This is \b not
 * a library-wide shared scratchpad!  Not exposed via C api (just use
 * malloc/free as usual). */
struct ScratchpadMallocFree : public ScratchpadBase {
    ScratchpadMallocFree(size_t size) {
        using mkldnn::impl::malloc;
        size_ = size;
        const size_t page_size = 2097152;
        scratchpad_ = (char *) malloc(size, page_size);
        assert(scratchpad_ != nullptr);
    }
    ~ScratchpadMallocFree() { free(scratchpad_); }
    virtual char *get() const { return scratchpad_; }
    virtual float *pfloat() const { return (float*)(void*)scratchpad_; }
private:
    char *scratchpad_;
    size_t size_;
};

/** growable local-use Scratchpad implemented as a shared pointer. Not thread-safe,
 * but can pass pointer to threads from OpenMP master. */
struct ScratchpadShared : public ScratchpadBase {
    ScratchpadShared(size_t bytes) {
        using mkldnn::impl::malloc;
        using mkldnn::impl::free;
        bytes = (bytes+15U)/16U*16U;
        if (bytes > size_) {
            if (scratchpad_ != nullptr) free(scratchpad_);
            size_ = bytes;
            /* Allocating on a page boundary to reduce TLB/page misses */
            const size_t page_size = 2097152;
            scratchpad_ = (char *) malloc(bytes, page_size);
            VEDNN_SCRATCH_DBG(" vednn ScratchpadShared[ %lu bytes ] @ %p\n",
                    (long unsigned)bytes, (void*)scratchpad_);
            assert(scratchpad_ != nullptr);
        }
        ++reference_count_;
    }
    ~ScratchpadShared() {
        using mkldnn::impl::free;
        reference_count_--;
        if (reference_count_ == 0) {
            VEDNN_SCRATCH_DBG(" ~ScratchpadShared[ %lu bytes ] @ %p\n",
                    (long unsigned)size_, (void*)scratchpad_);
            free(scratchpad_);
            scratchpad_ = nullptr;
            size_ = 0;
        }
    }
    virtual char *get() const { return scratchpad_; }
    virtual float *pfloat() const { return (float*)(void*)scratchpad_; }
private:
    static char *scratchpad_;
    static size_t size_;
    static unsigned int reference_count_;
};

/** Implementation of the ScratchpadBase interface that uses a
 *  thread-local scratchpad. */
struct ScratchpadTLS : public ScratchpadBase {
    ScratchpadTLS(size_t bytes) {
        using mkldnn::impl::malloc;
        using mkldnn::impl::free;
        bytes = (bytes+15U)/16U*16U;
        if (bytes > size_) {
            if (scratchpad_ != nullptr) free(scratchpad_);
            size_ = bytes;
            /* Allocating on a page boundary to reduce TLB/page misses */
            const size_t page_size = 2097152;
            scratchpad_ = (char *) malloc(bytes, page_size);
            VEDNN_SCRATCH_DBG(" vednn ScratchpadTLS[ %lu bytes ] @ %p\n",
                    (long unsigned)bytes, (void*)scratchpad_);
            assert(scratchpad_ != nullptr);
        }
        ++reference_count_;
    }
    ~ScratchpadTLS() {
        using mkldnn::impl::free;
        reference_count_--;
        if (reference_count_ == 0) {
            VEDNN_SCRATCH_DBG(" ~ScratchpadTLS[ %lu bytes ] @ %p\n",
                    (long unsigned)size_, (void*)scratchpad_);
            free(scratchpad_);
            scratchpad_ = nullptr;
            size_ = 0;
        }
    }
    virtual char *get() const { return scratchpad_; }
    virtual float *pfloat() const { return (float*)(void*)scratchpad_; }
private:
    THREAD_LOCAL static char *scratchpad_;
    THREAD_LOCAL static size_t size_;
    THREAD_LOCAL static unsigned int reference_count_;
    // nc++ 2.4 : "unsuitable threadprivate variable"
    //OMP(threadprivate(scratchpad_))//;
    //OMP(threadprivate(size_))//;
    //OMP(threadprivate(reference_count_))//;
};

/* Implementation of the ScratchpadBase interface that uses a global
 * scratchpad (simple shared pointer) that grows by filling with 1.0f values.
 * The constructor size is number-of-floats (not bytes). */
struct ScratchpadFloatOnes : public ScratchpadBase {
    ScratchpadFloatOnes(size_t floats) {
        using mkldnn::impl::malloc;
        using mkldnn::impl::free;
        floats = (floats+3U)/4U*4U;
        if (floats > size_) {
            if (scratchpad_ != nullptr) free(scratchpad_);
            size_ = floats;
            /* Allocating memory buffers on a page boundary to reduce TLB/page misses */
            const size_t page_size = 2097152;
            void* sp = malloc(floats*sizeof(float), page_size);
            VEDNN_SCRATCH_DBG(" vednn ScratchpadFloatOnes[ %lu bytes ] @ %p\n",
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
    ~ScratchpadFloatOnes() {
        using mkldnn::impl::free;
        --reference_count_;
        if (reference_count_ == 0) {
            VEDNN_SCRATCH_DBG(" ~ScratchpadFloatOnes[ %lu bytes ] @ %p\n",
                    (long unsigned)size_, scratchpad_);
            free(scratchpad_);
            scratchpad_ = nullptr;
            size_ = 0;
        }
    }
    virtual char *get() const { return scratchpad_; }
    virtual float *pfloat() const { return (float*)(void*)scratchpad_; }
private:
    static char *scratchpad_;
    static size_t size_;
    static unsigned int reference_count_;
};

inline char* vednn_scratchpad_shared(size_t bytes){
    return scratchpadShared->get();
}
inline char* vednn_scratchpadTLS(size_t bytes){
    return scratchpadTLS->get();
}
inline float* vednn_scratchpad_float_ones(size_t const floats){
    return reinterpret_cast<float*>(scratchpadFloatOnes->get());
}


}}//vednn::scratchpad::
// vim: et ts=4 sw=4 cindent cino=^=lg0,\:0,N-s
#endif // VEDNN_DEF_HPP
