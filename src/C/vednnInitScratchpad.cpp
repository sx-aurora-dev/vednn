/** based on (mkldnn) scratchpad, originally from Intel with Apache 2.0 license. */
#include "vednn-def.hpp"
#include <stdio.h>
#include <cstring> // memset

namespace vednn {
namespace scratchpad {

// now defined in vednn-def.hpp
//#define VEDNN_SCRATCH_VERBOSE 0

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
thread_local ScratchpadTLS  *threadScratch = nullptr;
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
        if (scratchpad_ != nullptr){
            VEDNN_SCRATCH_DBG(" free...");
            vednn::free( (char*)scratchpad_ /*- pad()*/ );
            scratchpad_ = nullptr;
        }
        size_ = 0U;
        VEDNN_SCRATCH_DBG(" vednn-ompthr %d ~ScratchpadTLS DESTROYED\n\n", get_thread_num());
        reference_count_ = 0;
    }
}
void ScratchpadTLS::free_malloc(size_t const bytes){
#if 1
#pragma omp critical /* should NOT be required */
    {
        if (scratchpad_ != nullptr) vednn::free(scratchpad_);
        VEDNN_SCRATCH_DBG(" free_malloc(%lu-->%lu bytes)... ",(long unsigned)size_,(long unsigned)bytes);
        size_ = bytes;
        scratchpad_ = vednn::malloc(bytes, page_size());
        if(size_) ScratchpadBase::checkNonNULL(scratchpad_,__FILE__,__LINE__);
    }
#else // debug attempt...
#pragma omp critical
    {
        VEDNN_SCRATCH_DBG("\n\n vednn-ompthr %d ScratchpadTLS resize [ %lu bytes--> %lu ]\n",
                get_thread_num(), (long unsigned)size_, (long unsigned)bytes);
        VEDNN_SCRATCH_DBG(" free... ");
        if (scratchpad_ != nullptr) vednn::free(scratchpad_);
        size_ = bytes;
        /* Allocating on a page boundary to reduce TLB/page misses */
        const size_t page_size = 4096; //2097152;
        VEDNN_SCRATCH_DBG(" malloc... ");
        scratchpad_ = vednn::malloc(bytes, page_size);
        VEDNN_SCRATCH_DBG(" check..." );
        if(size_) ScratchpadBase::checkNonNULL(scratchpad_,__FILE__,__LINE__);
    }
#endif
}
static void vednn_init_scratchpad_shared(size_t const bytes){
    scratchpadShared = /*static_cast<ScratchpadBase*>*/ (
            new ScratchpadShared(bytes));
    // clients use it by calling vednn_scratchpad(nBytes) and getting a pointer
    //printf(" vednn scratchpad_shared     @ %p REF %u\n",vednn_scratchpad_shared(bytes), scratchpadShared->ref());
    if (VEDNN_SCRATCH_VERBOSE)
        printf(" vednn INIT scratchpad_shared     @ %p REF %u\n",
                scratchpadShared->get(), scratchpadShared->ref());
}
// This must be done per thread, but the threads may not exist yet (except for 'master')
// This is internal to libvednnn.
void vednn_init_scratchpadTLS(){
#pragma omp parallel
    {
        threadScratch = new ScratchpadTLS();
        // C code calls vednn_scratchpad(nBytes) and getting a pointer
        if (VEDNN_SCRATCH_VERBOSE)
            printf(" vednn INIT threadScratch() %lu bytes  @ %p REF %u\n",
                    (unsigned long)threadScratch->size(),
                    threadScratch->get(), threadScratch->ref());
    }
    if (threadScratch == nullptr) threadScratch = new ScratchpadTLS();
}
/** This is called only by program thread (master) */
static void vednn_init_scratchpad_float_ones(size_t const floats){
    scratchpadFloatOnes = /*static_cast<ScratchpadBase*>*/(
            new ScratchpadFloatOnes(floats));
    // clients use it by calling vednn_scratchpad_float_ones(nFloats) and getting a pointer
    VEDNN_SCRATCH_DBG(" vednn INIT scratchpad float ones REF %u\n",
                scratchpadFloatOnes->ref());
}
//@}


}}//vednn::scratchpad::

//
// simple C API, for use during static lib __vednn_init/free
//
extern "C"{ //}
/** Initialize SCRATCHPAD memories. Generic idea: call create_scratchpad at
 * will, and it grows/reallocates the scratchpad area as needed.
 * Scratchpad ptrs should only be used within a single layer [local use only]
 * \b Global scratchpads should be created in the "wrapper" thread
 * and then shared with omp threads for read-only usage.
 * \b TLS scratchpad pointers should be allocated and used within single omp code block.
 *
 * These re-usable scratchpads can avoid reallocs, operating in "grow-only" mode.
 * ... until \c vednn_free_global_scratchpads
 * (note: tls scratchpads behave a bit differently)
 *
 * At library init, we bumps the (thread-local) ref count,
 * so we re-use shared scratchpad allocations until
 * end-of-process. */
void vednn_init_global_scratchpads(){
    printf("vednn_init_global_scratchpads()!\n");
    using namespace vednn::scratchpad;
    vednn_init_scratchpad_shared((0)/*bytes*/); // 1U<<18 ?
    vednn_init_scratchpadTLS();
    /** re-usable \e const buffer of 1.0f values. */
    vednn_init_scratchpad_float_ones(0/*floats*/); // 4096U ?
}

/** process scratchpad destroy, called when library unloaded from process. */
void vednn_free_global_scratchpads(){
    using vednn::scratchpad::scratchpadShared;
    using vednn::scratchpad::threadScratch;
    using vednn::scratchpad::scratchpadFloatOnes;
    printf("vednn_free_global_scratchpads()...\n"); fflush(stdout);
    delete scratchpadShared;
    scratchpadShared = nullptr;

    // todo: can we delete omp scratchpads for thr>0 ?
    VEDNN_SCRATCH_DBG("free thread_local scratchpads\n");
    if( threadScratch )
    {
#ifdef VEDNN_USE_OPENMP
#pragma omp critical
        {
            printf("free_global_scratchpads thr %d\n",
                    vednn::get_thread_num());
        }
#endif
#pragma omp barrier
#pragma omp parallel
        {
            // this only hits master, perhaps won't hit omp threads anyway.
            delete threadScratch;
            threadScratch = nullptr;
        }
    }
    // Trouble if try to clean up openmp threads
#if 0
#ifdef _OPENMP
#pragma omp parallel for schedule(static,1)
    for(int i=0; i<omp_get_max_threads(); ++i){
        if( threadScratch ){
            delete threadScratch;
            threadScratch = nullptr;
        }
    }
#endif
#endif

    VEDNN_SCRATCH_DBG("free global scratchpadFloatOnes\n");
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
    using vednn::scratchpad::threadScratch;
    void* ret = nullptr;
#pragma omp critical /* PARANOIA */
    if (threadScratch==nullptr) {
        threadScratch = new ScratchpadTLS(bytes);
        printf("\n%s:%u ERROR: library init should ideally avoid this paranoia\n");
    }

    if (threadScratch!=nullptr) {
        if (threadScratch->get()!=nullptr &&
            threadScratch->size() >= bytes+threadScratch->pad()) {
            ret = threadScratch->get();
        }else{
            ret = ScratchpadTLS(bytes).get();
        }
        memset(ret,0,threadScratch->size()); // PARANOIA
    }
    return ret;
}
float* vednn_scratchpad_float_ones(size_t const floats){
    using vednn::scratchpad::ScratchpadFloatOnes;
    using vednn::scratchpad::scratchpadFloatOnes;
    assert(scratchpadFloatOnes);
    //return (float*)vednn::scratchpad::ScratchpadFloatOnes(floats).get();
    // If in omp section following is DANGEROUS
    //   (differing omp executaion path --> possible hang at omp synch points
    //return ( scratchpadFloatOnes!=nullptr && scratchpadFloatOnes->size() >= floats
    //        ? (float*)scratchpadFloatOnes->get() // fast path:
    //        : (float*)ScratchpadFloatOnes(floats).get() );
    return vednn::scratchpad::vednn_scratchpad_float_ones(floats);
}

}//extern "C"

// vim: et ts=4 sw=4 cindent cino=^=lg0,\:0,N-s syntax=cpp.doxygen
