/** based on (mkldnn) scratchpad, originally from Intel with Apache 2.0 license. */
#include "vednn-def.hpp"
#include <stdio.h>

#define DBG(...) do{ printf(__VA_ARGS__); fflush(stdout); }while(0)
//#define DBG(...) do{}while(0)

namespace vednn {
namespace scratchpad {

/// \group scratchpad member variables
//@{
char *ScratchpadShared::scratchpad_ = nullptr;
size_t ScratchpadShared::size_ = 0;
unsigned int ScratchpadShared::reference_count_ = 0;

THREAD_LOCAL char *ScratchpadTLS::scratchpad_ = nullptr;
THREAD_LOCAL size_t ScratchpadTLS::size_ = 0;
THREAD_LOCAL unsigned int ScratchpadTLS::reference_count_ = 0;

char *ScratchpadFloatOnes::scratchpad_ = nullptr;
size_t ScratchpadFloatOnes::size_ = 0;
unsigned int ScratchpadFloatOnes::reference_count_ = 0;
//@}

/// \group library-wide stratchpads
//@{
ScratchpadShared     *scratchpadShared = nullptr;
ScratchpadTLS        *scratchpadTLS = nullptr; // thread-specific scratchpads not yet used by libvednn!
ScratchpadFloatOnes  *scratchpadFloatOnes = nullptr;
//@}

/// \group process scratchpad init
/** process scratchpad init.  This bumps ref count so the memory is only
 * release when library unloads (or process ends) and \c __vednn_free()
 * is [automatically] called. */
//@{
void vednn_init_scratchpad_shared(size_t size){
    scratchpadShared = /*static_cast<ScratchpadBase*>*/ (
            new ScratchpadShared(size));
    // clients use it by calling vednn_scratchpad(nBytes) and getting a pointer
    DBG(" vednn scratchpad_shared     @ %p\n",vednn_scratchpad_shared(1));
}
void vednn_init_scratchpadTLS(size_t size){
    scratchpadTLS = /*static_cast<ScratchpadBase*>*/ (
            new ScratchpadTLS(size));
    // clients use it by calling vednn_scratchpad(nBytes) and getting a pointer
    DBG(" vednn scratchpadTLS         @ %p\n",vednn_scratchpadTLS(1));
}
void vednn_init_scratchpad_float_ones(size_t size){
    scratchpadFloatOnes = /*static_cast<ScratchpadBase*>*/(
            new ScratchpadFloatOnes(size));
    // clients use it by calling vednn_scratchpad_float_ones(nFloats) and getting a pointer
    DBG(" vednn scratchpad float ones @ %p\n",vednn_scratchpad_float_ones(1));
}
//@}


}}//vednn::scratchpad::

//
// simple C API, for use during static lib __vednn_init/free
//
extern "C"{ //}
void vednn_init_global_scratchpads(){
    DBG("vednn_init_scratchpad()!\n");
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
    vednn_init_scratchpad_shared((1UL<<18)/*bytes*/);
    vednn_init_scratchpadTLS(0/*bytes*/);
    /** re-usable \e const buffer of 1.0f values. */
    vednn_init_scratchpad_float_ones(4096/*floats*/);
}

/** process scratchpad destroy. */
void vednn_free_global_scratchpads(){
    delete vednn::scratchpad::scratchpadShared;
    delete vednn::scratchpad::scratchpadTLS;
    delete vednn::scratchpad::scratchpadFloatOnes;
}

#if 0 // not really compatible with 'C' interface
char* vednn_scratchpad(size_t bytes){
    assert(vednn::scratchpad::scratchpad);
    return vednn::scratchpad::scratchpad->get();
}
#endif
char* vednn_scratchpad_shared(size_t bytes){
    assert(vednn::scratchpad::scratchpadShared);
    return vednn::scratchpad::scratchpadShared->get();
    //return (char*) vednn::scratchpad::ScratchpadShared::scratchpad_;
}
char* vednn_scratchpadTLS(size_t bytes){
    assert(vednn::scratchpad::scratchpadTLS);
    return vednn::scratchpad::scratchpadTLS->get();
}
float* vednn_scratchpad_float_ones(size_t const floats){
    assert(vednn::scratchpad::scratchpadFloatOnes);
    return (float*)vednn::scratchpad::scratchpadFloatOnes->get();
}

}//extern "C"

// vim: et ts=4 sw=4 cindent cino=^=lg0,\:0,N-s syntax=cpp.doxygen