#ifndef VEDNN_DEF_HPP
#define VEDNN_DEF_HPP
#include "vednn-def.h"

// from gen-dnn sources: (removed dependencies)
//#include "gen-dnn/mkldnn_thread.hpp"
//#include "gen-dnn/utils.hpp" // mkldnn::impl::malloc/free (with alignment)

#ifdef _OPENMP
#include <omp.h>
#endif

#include <cassert>
#include <cstddef>  // size_t
#include <cstdio>
#if __cplusplus > 201700L
#include <cstdlib>  // aligned_alloc
#else
#include <stdlib.h> // posix_memalign
#endif

// I find no easy way to get an integer thread id (without operator <<)
//#include <thread>   // C++ get_id() [vs threads.h thrd_current()]
//#include <threads.h>  // thrd_current() -- ncc will not compile this (issue with _Thread_local static member)
//
//#include <sys/types.h>
//#include <sys/syscall.h> // nc++ : "syscall" is undefined
//static inline pid_t gettid() { return syscall(__NR_gettid); }

#ifndef VEDNN_SCRATCH_VERBOSE
#define VEDNN_SCRATCH_VERBOSE 0
#endif
#ifndef VEDNN_SCRATCH_DBG
#if VEDNN_SCRATCH_VERBOSE
#define VEDNN_SCRATCH_DBG(...) do{ printf(__VA_ARGS__); fflush(stdout); }while(0)
#else
#define VEDNN_SCRATCH_DBG(...) do{}while(0)
#endif
#endif

namespace vednn {

#if 0 && defined(VEDNN_SCRATCH_VERBOSE)
/** scratchpads are rarely reallocated, so full walk should not happen often.
 * Beware: mcheck is NOT threadsafe, so we might skip the check if dangerous.
 */
inline void mcheck() {
#ifdef _OPENMP
#if 0 // first version, really try to run it.
#pragma omp single
    {
        mcheck_check_all();
    }
#else // run it only avoid running it within an openmp parallelized region
    if( !omp_in_parallel() ){
        mcheck_check_all()
    }
#endif
#else // no openmp
    mcheck_check_all();
#endif
}
#else
inline void mcheck() {}
#endif

#ifdef _OPENMP
inline int get_thread_num() {return omp_get_thread_num();}
#else
inline int get_thread_num() {return 0;}
#endif

/** To reduce dependence on gen-dnn/utils.hpp, here's a VE-specific aligned malloc.
 * - keep this thread safe (no mcheck!). */
inline void *malloc(size_t size, int alignment) {
    void * ptr;
#if 1
#if __cplusplus > 201700L
        ptr = std::aligned_alloc(alignment, size);
#else // older C++ (or C)
        int rc = ::posix_memalign(&ptr, alignment, size);
        ptr = (rc==0? ptr: nullptr);
#endif
#else
#pragma omp critical // serialize debug messages (slow)
    {
        VEDNN_SCRATCH_DBG(" vednn::malloc BEGIN(thr %d ", get_thread_num());
#if __cplusplus > 201700L
        ptr = std::aligned_alloc(alignment, size);
#else // older C++ (or C)
        int rc = ::posix_memalign(&ptr, alignment, size);
        ptr = (rc==0? ptr: nullptr);
#endif
        VEDNN_SCRATCH_DBG(" size %lu align %lu @ %p",(long unsigned)size, (long unsigned)alignment, ptr);
        VEDNN_SCRATCH_DBG(")END\n");
    }
#endif
    return ptr;
}
/** free (for scratchpads, possibly verbose. */
inline void free(void *p) {
#if 1
    ::free(p);
#else
#pragma omp critical
    {
        VEDNN_SCRATCH_DBG(" vednn::free BEGIN(thr %d ptr %p", get_thread_num(), p);
        VEDNN_SCRATCH_DBG(" ::free...");
        ::free(p);
        VEDNN_SCRATCH_DBG(")END\n");
    }
#endif
}

namespace scratchpad {

/** vednn C++ layers should use these to obtain the scratchpad memory addresses.
 * The extern "C" functions of same name are not inlined. */
//void* vednn_scratchpad(size_t bytes);
void* vednn_scratchpad_shared(size_t bytes);
void* vednn_scratchpad_tls(size_t bytes);
float* vednn_scratchpad_float_ones(size_t const floats);

// Yet another exposed internal detail from vednnInitScratchpad.cpp
//void vednn_init_scratchpadTLS(size_t const bytes);

/** borrowed from mkldnn scratchpad, Apache 2.0 license.
 * Overkill, for vednn, and introduces extra function calls if used as public API.
 */
struct ScratchpadBase {
    virtual ~ScratchpadBase() {}
    virtual void * get()  const = 0;
    virtual size_t size() const = 0;
    virtual unsigned ref() const = 0;

    // not really saving much typeing: 
    //template<typename T> inline operator T() { return static_cast<T>(this->get()); }
    // debug? optional?
    static void checkNonNULL(void* ptr, char const* file, size_t const line){
        if(ptr == nullptr) {
            fprintf(stderr,"error: failed alloc! %s:%lu",
                    file, (long unsigned)line);
            //throw "failed alloc!"; // CANNOT throw if in omp section
            //  (with work, can catch and save, then check and rethrow outside omp block)
        }
    }
};
// fwd decl
struct ScratchpadMallocFree;
struct ScratchpadShared;
struct ScratchpadTLS;
struct ScratchpadFloatOnes;
// library provides some re-usable scratchpads (use locally, within single layer funcs)
// These provide a ref count to keep regions allocated
// for duration that library is loaded into executing process.
extern ScratchpadShared             *scratchpadShared;
extern thread_local ScratchpadTLS   *threadScratch;
extern ScratchpadFloatOnes          *scratchpadFloatOnes;

/** Implementation of the Scratchpad interface that uses malloc/free
 * every time (single-use, non-resizing scratchpad).  This is \b not
 * a library-wide shared scratchpad!  Not exposed via C api (just use
 * malloc/free as usual). <em>safest option for debugging.</em> */
struct ScratchpadMallocFree final : public ScratchpadBase {
    ScratchpadMallocFree(size_t size) {
        size_ = size;
        const size_t page_size = 4096U; //2097152;
        scratchpad_ = vednn::malloc(size, page_size);
        if(size_) ScratchpadBase::checkNonNULL(scratchpad_,__FILE__,__LINE__);
    }
    ~ScratchpadMallocFree() { vednn::free(scratchpad_); }
    void *get() const { return scratchpad_; }
    size_t size() const { return size_; }
    unsigned ref() const { return 1U; }
private:
    void *scratchpad_;
    size_t size_;
};

/** growable local-use Scratchpad implemented as a shared pointer. Not thread-safe,
 * but can pass pointer to threads from OpenMP master.
 *
 * Scenario: several objects disjointly need temporary workspace.
 * Init phase: Every object initializes 1 ScratchpadShared
 * Exec phase: Noncurrent executing objects get() workspace ptr.
 *             Each disjointly executing object uses get() for a mem ptr
 */
struct ScratchpadShared : public ScratchpadBase {
    ScratchpadShared(size_t bytes) {
#ifdef _OPENMP
        if( omp_in_parallel() ){
            printf("program error: vednn ScrachpadShared constructed during parallel execution\n");
        }
#endif
        bytes = (bytes+15U)/16U*16U;
        ++reference_count_;
        if (bytes > size_) {
            // If threads *might* be executing in background, mcheck is NOT safe!
            //vednn::mcheck();
            if (scratchpad_ != nullptr) vednn::free(scratchpad_);
            size_ = bytes;
            /* Allocating on a page boundary to reduce TLB/page misses */
            const size_t page_size = 4096U; //2097152;
            scratchpad_ = vednn::malloc(bytes, page_size);
            VEDNN_SCRATCH_DBG(" vednn resize ScratchpadShared[ %lu bytes ] @ %p\n",
                    (long unsigned)bytes, (void*)scratchpad_);
            if(size_) ScratchpadBase::checkNonNULL(scratchpad_,__FILE__,__LINE__);
            //vednn::mcheck();
        }
    }
    ~ScratchpadShared() {
        reference_count_--;
        if (reference_count_ == 1) {
            VEDNN_SCRATCH_DBG(" ~ScratchpadShared[ %lu bytes ] @ %p\n",
                    (long unsigned)size_, (void*)scratchpad_);
            if (scratchpad_ != nullptr){
                //vednn::mcheck();
                vednn::free(scratchpad_);
            }
            scratchpad_ = nullptr;
            size_ = 0;
            //vednn::mcheck();
        }
    }
    virtual void *get() const { return scratchpad_; }
    virtual size_t size() const { return size_; }
    virtual unsigned ref() const { return reference_count_; }
private:
    static void *scratchpad_;
    static size_t size_;
    static unsigned int reference_count_;
};

/** Implementation of the ScratchpadBase interface that uses a
 *  thread-local scratchpad <B>ncc has issues here</B>.
 *
 * - must be constructed AND destructed from same thread.
 * - libvednn constructs via new/delete to keep ref count positive
 *   a long time (causing growing resize behavior).
 */
struct ScratchpadTLS : public ScratchpadBase {
private:
    friend void vednn_init_scratchpadTLS();
    ScratchpadTLS() {
        scratchpad_ = nullptr;
        size_ = 0U;
        reference_count_ = 1U;
    }
public:
    ScratchpadTLS(size_t bytes) {
        // the following "keeps" a single TLS scratchpad open.
        // auto-init for omp was problematic -- no threadprivate
        // yet for VE ! :(
        bytes = ((bytes+pad()*2U)+15U)/16U*16U;
        unsigned tmp = ++reference_count_;
        if(tmp == 1){
            VEDNN_SCRATCH_DBG(" vednn: ScratchpadTLS INIT!\n");
            //vednn_init_scratchpadTLS(bytes);
        }
        //assert( reference_count_ >= 2U );
        if (bytes > size_)
            free_malloc(bytes);
        VEDNN_SCRATCH_DBG(" vednn-ompthr %d ScratchpadTLS[ %lu bytes ] @ %p REF %u\n",
                get_thread_num(), (long unsigned)bytes, (void*)scratchpad_, reference_count_+1U);
    }
    ~ScratchpadTLS() {
        unsigned tmp = --reference_count_;
        VEDNN_SCRATCH_DBG(" vednn-ompthr %d ~ScratchpadTLS[ %lu bytes ] @ %p REF %u\n",
                get_thread_num(), (long unsigned)size_, (void*)scratchpad_, tmp);
        if (tmp== 0)
            freeme();
    }
    virtual void *get() const { 
        //VEDNN_SCRATCH_DBG(" ScratchPadTLS@%p get()--> %p\n", (void*)this, (void*)scratchpad_);
        //return scratchpad_;
        return (void*)( (char*)scratchpad_+pad() ); }
    virtual size_t size() const { return size_; }
    virtual unsigned ref() const { return reference_count_; }
    static unsigned pad() { return 4096; }
private:
    static size_t page_size() { return 4096U; } // 2097152U ideally want page boundary (2M?)
    static void freeme();
    static void free_malloc(size_t const bytes);
    // ncc historically had definite issues running thread_local object constructors.
    // Here I ASSUME ncc will properly initialize thread_local fundamental types.
    static thread_local void *scratchpad_;
    static thread_local size_t size_;
    static thread_local volatile unsigned int reference_count_;
    // nc++ 2.4 : "unsuitable threadprivate variable"
    //OMP(threadprivate(scratchpad_))//;
    //OMP(threadprivate(size_))//;
};
// OK thread_local int xxx;
// OK void *yyy;
// OK #pragma omp threadprivate(xxx)
// OK #pragma omp threadprivate(yyy)
// no #pragma omp threadprivate(ScratchpadTLS::size_)

/* Implementation of the ScratchpadBase interface that uses a global
 * scratchpad (simple shared pointer) that grows by filling with 1.0f values.
 * The constructor size is number-of-floats (not bytes).
 * If previous 1s buffer was big enough, we re-use, else we get a new buffer.
 * Only one such object should be active at any one time.
 *
 * Note: refresh ptr with get(), because memory can be reallocated by any new
 * 'ScratchpadFloatOnes' constructors.
 *
 * \pre all concurrent threads agree on the value of `floats`. This restriction
 * can be removed once a trustworthy max reduction for VE is written.
 */
struct ScratchpadFloatOnes : public ScratchpadBase {
    /** Threadsafe.
     * All threads of omp team call constructor, possibly with thread-specific
     * 'size_t floats' values (we'll determine max of these).
     *
     * note: omp settings tricky -- perhaps there is an easier way. But remember
     * historical issues with shared pointer object constructors in VE libraries!
     */
    ScratchpadFloatOnes(size_t floats) {
        floats = (floats+3U)/4U*4U;
#ifdef VEDNN_USE_OPENMP
#if VEDNN_SCRATCH_VERBOSE
#pragma omp critical
        {
            VEDNN_SCRATCH_DBG(" %s%sthr %d/%d floatOnes=%lu",
                    (omp_in_parallel()? "||":""), (omp_get_nested()? "nested-":""),
                    omp_get_thread_num(), omp_get_num_threads(),
                    (long unsigned)floats);
        }
#endif

#if 0 // This code block allows threads to agree on a max size of read-only shared data
        // However, omp barrier should NOT appear within omp section (hang)
        // OTOH, without barrier have a race condition.
        // SOLUTION: allocate resources OUTSIDE omp section and pass in
        //           as arg (do not use a file-scope statics in libraries)
        // BUT this changes API (or just copy the (short) direct_gemm impl
        // into a helper routine that is file static and has the nec. added
        // resource args.

        // ERROR -- ok only with mb=8 or 1; otherwise can hang
        // BUG was in vednnConvolutionForward.c default "mb" wrapper:
        //   You CANNOT "if(myBatch==0) {rc|=VEDNN_SUCCESS}", because then some
        //   threads may never get here.  Hence program would hang.
        // FIX: if mb size < omp_get_max_threads(), use few threads,
        //      so myBatch is never 0, and all threads get here.
        //
        // paranoia : adjust floats to determine max size amongst all threads
        //  (not needed if all threads have same 'floats' value)
        //  does VE support max reduction?
        size_t maxfloats = 0U;
#pragma omp critical
        {
            if (floats > maxfloats) maxfloats = floats;
            VEDNN_SCRATCH_DBG(" %s%sthr %d/%d ",
                    (omp_in_parallel()? "||":""), (omp_get_nested()? "nested-":""),
                    omp_get_thread_num(), omp_get_num_threads());
            VEDNN_SCRATCH_DBG(" maxfloats=%lu\n", (long unsigned)maxfloats);
        }
#pragma omp barrier
        // try an alt "implicit barrier"? No still can hang 
//#pragma omp parallel
//        {
//            ;
//        }
        floats = maxfloats;

#elif 0 // simpler: assume all threads agree on 'floats' size, 1st thread here can allocate
        // Still need to know all threads will never access a previous buffer:
#pragma omp barrier
#else
        // DO NOTHING, use floats as is
        // NOTE: this is also a bug because jitconv on multiple tests might
        // have a thread get here "early" and invalidate a pointer that is
        // still in use by another thread.  Probably some shared ptr hack
        // that could circumvent.
        //
        // This is partly why I changed test scripts to run a single test and quit
        // (i.e. each line of tests file starts up a new jitconv test)
#endif
#endif

#ifdef VEDNN_USE_OPENMP
        // now any one thread can do the allocation ...
//#pragma omp barrier
//#pragma omp single /* one thread allocates, omp barrier at end, threads have now agreed on max size. */
        // issues when executed inside || region?
#endif
#pragma omp critical // I don't favor this way  (I wish barrer+single could work in all cases)
        {
            VEDNN_SCRATCH_DBG(" flobts=%lu ", (long unsigned)floats);
            if (floats > size_) {
                //vednn::mcheck();
                if (scratchpad_ != nullptr) free(scratchpad_); size_ = floats;
                /* Allocating memory buffers on a page boundary to reduce TLB/page misses */
                const size_t page_size = 4096U;
                void* sp = vednn::malloc(floats*sizeof(float), page_size);
                VEDNN_SCRATCH_DBG(" vednn ScratchpadFloatOnes[ %lu bytes ] @ %p ",
                        (long unsigned)(floats*sizeof(float)), sp);
                //vednn::mcheck();
                if(size_) ScratchpadBase::checkNonNULL(sp,__FILE__,__LINE__);
                scratchpad_ = sp;
                // fill with 1.0 whenever resized
                float *spf = (float*) sp;
                for(size_t i=0U; i<floats; ++i){
                    spf[i] = 1.0f;
                }
            }
            ++reference_count_; // thread group contributes 1 to ref count
            VEDNN_SCRATCH_DBG(" rff=%lu\n", (long unsigned)reference_count_);
        } // single => implicit barrier
    }
    /** also threadsafe, with thread-barrier at both \e entry and \e exit.
     * \post pointer value from previous \c get() should no longer be used. */
    ~ScratchpadFloatOnes() {
#ifdef VEDNN_USE_OPENMP
        // for dealloc, need barrier at beginning, not end, of omp block
//#pragma omp barrier
//#pragma omp single
        // but may have issues inside omp region
#pragma omp critical
#endif
        {
            if (reference_count_ == 0) { // (cannot throw if in omp)
                fprintf(stderr,"\n FloatOnes dealloc error! %s:%lu",
                        __FILE__, (long unsigned)__LINE__);
            }
            --reference_count_;
            if (reference_count_ == 0) {
                VEDNN_SCRATCH_DBG(" ~ScratchpadFloatOnes[ %lu bytes ] @ %p\n",
                        (long unsigned)size_, scratchpad_);
                //vednn::mcheck();
                if (scratchpad_ != nullptr){
                    free(scratchpad_);
                }
                scratchpad_ = nullptr;
                size_ = 0;
                //vednn::mcheck();
            }
        } // implicit omp barrier so next constructor can safely run
    }
    virtual size_t size() const { return size_; }
    virtual unsigned ref() const { return reference_count_; }
private:
    static void *scratchpad_;
    static size_t size_;
    static unsigned int reference_count_;
    friend float* vednn_scratchpad_float_ones(size_t const floats);
    /** dangerous function */
    virtual void *get() const {
        // paranoia:
        if(scratchpad_ == nullptr && reference_count_ != 0) {
            // (cannot throw if in omp)
            fprintf(stderr,"\nwarning: FloatOnes get() null?! %s:%lu",
                    __FILE__, (long unsigned)__LINE__);
        }
        return scratchpad_;
    }
};

inline void* vednn_scratchpad_shared(size_t bytes){
    return  ScratchpadShared{bytes}.get();
}
inline void* vednn_scratchpadTLS(size_t bytes){
    return ScratchpadTLS{bytes}.get();
}
inline float* vednn_scratchpad_float_ones(size_t const floats){
    return reinterpret_cast<float*>(ScratchpadFloatOnes{floats}.get());
}


}}//vednn::scratchpad::
// vim: et ts=4 sw=4 cindent cino=^=lg0,\:0,N-s syntax=cpp.doxygen
#endif // VEDNN_DEF_HPP
