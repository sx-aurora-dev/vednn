/** based on (mkldnn) scratchpad, originally from Intel with Apache 2.0 license. */
#include "vednn-def.hpp"
#include <stdio.h>

namespace vednn {
namespace scratchpad {

THREAD_LOCAL char *global_scratchpad_t::scratchpad_ = nullptr;
THREAD_LOCAL size_t global_scratchpad_t::size_ = 0;
THREAD_LOCAL unsigned int global_scratchpad_t::reference_count_ = 0;

THREAD_LOCAL char *global_scratchpad_float_ones_t::scratchpad_ = nullptr;
THREAD_LOCAL size_t global_scratchpad_float_ones_t::size_ = 0;
THREAD_LOCAL unsigned int global_scratchpad_float_ones_t::reference_count_ = 0;

//static global_scratchpad_t             *global_scratchpad = nullptr;
//static global_stratchpad_float_ones_t  *global_scratchpad_float_ones = nullptr;
global_scratchpad_t             *global_scratchpad = nullptr;
global_scratchpad_float_ones_t  *global_scratchpad_float_ones = nullptr;

/** process scratchpad init.  This bumps ref count so the memory is only
 * release when library unloads (or process ends) and \c __vednn_free()
 * is [automatically] called. */
void vednn_init_scratchpad(size_t size){
    global_scratchpad = /*static_cast<scratchpad_t*>*/ (
            new global_scratchpad_t(size));
    // clients use it by calling vednn_scratchpad(nBytes) and getting a pointer
    printf(" vednn scratchpad            @ %p\n",vednn_scratchpad(1));
}
void vednn_init_scratchpad_float_ones(size_t size){
    global_scratchpad_float_ones = /*static_cast<scratchpad_t*>*/(
            new global_scratchpad_float_ones_t(size));
    // clients use it by calling vednn_scratchpad_float_ones(nFloats) and getting a pointer
    printf(" vednn scratchpad float ones @ %p\n",vednn_scratchpad_float_ones(1));
}


}}//vednn::scratchpad::

//
// simple C API, for use during static lib __vednn_init/free
//
extern "C"{ //}
void vednn_init_global_scratchpads(){
    printf("vednn_init_scratchpad()!\n");
    using namespace vednn::scratchpad;
    // create a GLOBAL SCRATCHPAD. To use, call create_scratchpad at
    // will, and it will resize the scratchpad area as needed.
    // Scratchpad ptrs should only be used within a single layer.
    // They should be created in the "wrapper" thread,
    // and then shared with omp threads.  (I think).
    // DO NOT call create_scratchpad from omp threads.
    vednn_init_scratchpad((1UL<<18)/*bytes*/);
    // We create sp, which bumps the (thread-local) ref count,
    // so we re-use create_scratchpad allocations until
    // end-of-process.
    //
    // We don't care about maintaining global scratchpad normally,
    // but expose a void* version "for future use"".
    vednn_init_scratchpad_float_ones(2048);
}

/** process scratchpad destroy. */
void vednn_free_global_scratchpads(){
    delete vednn::scratchpad::global_scratchpad;
    delete vednn::scratchpad::global_scratchpad_float_ones;
}

char* vednn_scratchpad(size_t bytes){
    return vednn::scratchpad::global_scratchpad->get();
}
float* vednn_scratchpad_float_ones(size_t const floats){
    return (float*)vednn::scratchpad::global_scratchpad_float_ones->get();
}

}//extern "C"

#if 0 // not used in vednn
/*
   Implementation of the scratchpad_t interface that is compatible with
   a concurrent execution.
   */
template<int tag>
struct concurrent_scratchpad_t : public mkldnn::impl::scratchpad_t {
    concurrent_scratchpad_t(size_t size) {
        size_ = size;
        scratchpad_ = (char *) malloc(size, page_size);
        assert(scratchpad_ != nullptr);
    }

    ~concurrent_scratchpad_t() {
        free(scratchpad_);
    }

    virtual char *get() const {
        return scratchpad_;
    }

private:
    char *scratchpad_;
    size_t size_;
};
#endif


#if 0
/*
   Scratchpad creation routine
   */
scratchpad_t *create_scratchpad(size_t size, bool per_thread) {
    //#ifndef MKLDNN_ENABLE_CONCURRENT_EXEC
    //    return new global_scratchpad_t(size);
    //#else
    //    return new concurrent_scratchpad_t(size);
    //#endif
    return per_thread
        ? static_cast<scratchpad_t*>(new concurrent_scratchpad_t(size))
        : static_cast<scratchpad_t*>(new global_scratchpad_t(size));
}
#endif
// vim: et ts=4 sw=4 cindent cino=^=lg0,\:0,N-s syntax=cpp.doxygen
