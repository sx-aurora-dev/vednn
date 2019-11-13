/** based on (mkldnn) scratchpad, originally from Intel with Apache 2.0 license. */
#include "vednn-def.hpp"
#include <stdio.h>

namespace vednn {
namespace scratchpad {

/// \group scratchpad member variables
//@{
void *ScratchpadShared::scratchpad_ = nullptr;
size_t ScratchpadShared::size_ = 0;
unsigned int ScratchpadShared::reference_count_ = 0;

thread_local void *ScratchpadTLS::scratchpad_ = nullptr;
thread_local size_t ScratchpadTLS::size_ = 0;
thread_local volatile unsigned int ScratchpadTLS::reference_count_ = 0;

void *ScratchpadFloatOnes::scratchpad_ = nullptr;
size_t ScratchpadFloatOnes::size_ = 0;
unsigned int ScratchpadFloatOnes::reference_count_ = 0;
//@}

/// \group library-wide stratchpads
//@{
ScratchpadShared            *scratchpadShared = nullptr;
thread_local ScratchpadTLS  *scratchpadTLS = nullptr; // thread-specific scratchpads not yet used by libvednn!
ScratchpadFloatOnes         *scratchpadFloatOnes = nullptr;
//@}

/// \group process scratchpad init
/** process scratchpad init.  This bumps ref count so the memory is only
 * release when library unloads (or process ends) and \c __vednn_free()
 * is [automatically] called. */
//@{
void ScratchpadTLS::freeme(){
    // vednn TLS scratchpad is almost never freed (usual to leak)
#pragma omp critical
    {
        //vednn::mcheck();
        if (scratchpad_ != nullptr){
            VEDNN_SCRATCH_DBG(" free...");
            vednn::free( (char*)scratchpad_ - pad() );
            scratchpad_ = nullptr;
        }
        size_ = 0U;
        VEDNN_SCRATCH_DBG(" check...");
        //vednn::mcheck();
        VEDNN_SCRATCH_DBG(" vednn-ompthr %d ~ScratchpadTLS DESTROYED\n\n", get_thread_num());
        reference_count_ = 0;
    }
}
void ScratchpadTLS::free_malloc(size_t const bytes){
#if 1
        if (scratchpad_ != nullptr) vednn::free(scratchpad_);
        size_ = bytes;
        VEDNN_SCRATCH_DBG(" malloc... ");
        scratchpad_ = vednn::malloc(bytes, page_size());
        if(size_) ScratchpadBase::checkNonNULL(scratchpad_,__FILE__,__LINE__);
#else
#pragma omp critical
    {
        VEDNN_SCRATCH_DBG("\n\n vednn-ompthr %d ScratchpadTLS resize [ %lu bytes--> %lu ]\n",
                get_thread_num(), (long unsigned)size_, (long unsigned)bytes);
        //vednn::mcheck();
        VEDNN_SCRATCH_DBG(" free... ");
        if (scratchpad_ != nullptr) vednn::free(scratchpad_);
        size_ = bytes;
        /* Allocating on a page boundary to reduce TLB/page misses */
        const size_t page_size = 4096; //2097152;
        VEDNN_SCRATCH_DBG(" malloc... ");
        scratchpad_ = vednn::malloc(bytes, page_size);
        VEDNN_SCRATCH_DBG(" check..." );
        //vednn::mcheck();
        if(size_) ScratchpadBase::checkNonNULL(scratchpad_,__FILE__,__LINE__);
    }
#endif
}
void vednn_init_scratchpad_shared(size_t const bytes){
    scratchpadShared = /*static_cast<ScratchpadBase*>*/ (
            new ScratchpadShared(bytes));
    // clients use it by calling vednn_scratchpad(nBytes) and getting a pointer
    //printf(" vednn scratchpad_shared     @ %p REF %u\n",vednn_scratchpad_shared(bytes), scratchpadShared->ref());
    printf(" vednn INIT scratchpad_shared     @ %p REF %u\n",scratchpadShared->get(), scratchpadShared->ref());
}
// This must be done per thread, but the threads may not exist yet (except for 'master')
// This is internal to libvednnn.
void vednn_init_scratchpadTLS(size_t const bytes){
    scratchpadTLS = /*static_cast<ScratchpadBase*>*/ (
            new ScratchpadTLS(bytes));
    // C code calls vednn_scratchpad(nBytes) and getting a pointer
    printf(" vednn INIT scratchpadTLS         @ %p REF %u\n",scratchpadTLS->get(), scratchpadTLS->ref());
}
void vednn_init_scratchpad_float_ones(size_t const floats){
    scratchpadFloatOnes = /*static_cast<ScratchpadBase*>*/(
            new ScratchpadFloatOnes(floats));
    // clients use it by calling vednn_scratchpad_float_ones(nFloats) and getting a pointer
    printf(" vednn INIT scratchpad float ones @ %p REF %u\n",scratchpadFloatOnes->get(), scratchpadFloatOnes->ref());
}
//@}


}}//vednn::scratchpad::

//
// simple C API, for use during static lib __vednn_init/free
//
extern "C"{ //}
void vednn_init_global_scratchpads(){
    printf("vednn_init_scratchpad()!\n");
    using namespace vednn::scratchpad;
    /**create a GLOBAL SCRATCHPAD. To use, call create_scratchpad at
     * will, and it will resize the scratchpad area as needed.
     * Scratchpad ptrs should only be used within a single layer.
     * They should be created in the "wrapper" thread,
     * and then shared with omp threads.  (I think).
     * DO NOT call create_scratchpad from omp threads.
     *
     * At library init, we bumps the (thread-local) ref count,
     * so we re-use shared scratchpad allocations until
     * end-of-process. */
    vednn_init_scratchpad_shared((0)/*bytes*/); // 1U<<18 ?
    vednn_init_scratchpadTLS(0/*bytes*/);
    /** re-usable \e const buffer of 1.0f values. */
    vednn_init_scratchpad_float_ones(0/*floats*/); // 4096U ?
}

/** process scratchpad destroy, called when library unloaded from process. */
void vednn_free_global_scratchpads(){
    using vednn::scratchpad::scratchpadShared;
    using vednn::scratchpad::scratchpadTLS;
    using vednn::scratchpad::scratchpadFloatOnes;
    printf("vednn_free_global_scratchpads()...\n"); fflush(stdout);
    delete scratchpadShared;
    scratchpadShared = nullptr;
    // todo: can we delete omp scratchpads for thr>0 ?
    if( scratchpadTLS ){
        // this only hits master, perhaps won't hit omp threads anyway.
        delete scratchpadTLS;
        scratchpadTLS = nullptr;
    }
    // Trouble if try to clean up openmp threads
#if 0
#ifdef _OPENMP
#pragma omp parallel for schedule(static,1)
    for(int i=0; i<omp_get_max_threads(); ++i){
        if( scratchpadTLS ){
            delete scratchpadTLS;
            scratchpadTLS = nullptr;
        }
    }
#endif
#endif
    delete scratchpadFloatOnes;
    scratchpadFloatOnes = nullptr;
}

void* vednn_scratchpad_shared(size_t bytes){
    using vednn::scratchpad::ScratchpadShared;
    using vednn::scratchpad::scratchpadShared;
    assert(scratchpadShared);
    return (scratchpadShared->size() >= bytes
            ? scratchpadShared->get() // fast path:
            // else resizing path (also general API)
            : ScratchpadShared(bytes).get() );
}
void* vednn_scratchpadTLS(size_t bytes){
    // hack: check if null and do a 'new' during constructor
    // (until 'threadprivate' is supported)
    using vednn::scratchpad::ScratchpadTLS;
    using vednn::scratchpad::scratchpadTLS;
    return ( scratchpadTLS!=nullptr && scratchpadTLS->size() >= bytes
            ? scratchpadTLS->get() // fast path:
            : ScratchpadTLS(bytes).get() );
}
float* vednn_scratchpad_float_ones(size_t const floats){
    using vednn::scratchpad::ScratchpadFloatOnes;
    using vednn::scratchpad::scratchpadFloatOnes;
    assert(scratchpadFloatOnes);
    //return (float*)vednn::scratchpad::ScratchpadFloatOnes(floats).get();
    return ( scratchpadFloatOnes!=nullptr && scratchpadFloatOnes->size() >= floats
            ? (float*)scratchpadFloatOnes->get() // fast path:
            : (float*)ScratchpadFloatOnes(floats).get() );
}

}//extern "C"

// vim: et ts=4 sw=4 cindent cino=^=lg0,\:0,N-s syntax=cpp.doxygen
