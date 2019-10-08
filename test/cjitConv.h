#ifndef CJITCONV_H
#define CJITCONV_H
/** \file
 * To use cjitConv.hpp from 'C' code, we provide a 'C' API.
 * We lose a lot of the C++ information, though, so you really
 * have to know what the function signatures are.
 * (The C++ backend can print them, to help out).
 */

#include "conv_test_param.h"	// struct param for vednn convolutions

#ifdef __cplusplus
extern "C" { //}
#endif

/** {\c sym,\c ptr} info */
typedef struct {
    char const* sym;
    void const* ptr;
    size_t tag; ///< tag (test number, not source file number)
} CjitSym;

typedef struct {
    void *   opaque;  ///< remember the dlopen handle, so we dlcose during free
    CjitSym* syms;    ///< array of {sym,ptr} (and src number)
    size_t   len;     ///< \c syms[len] will also be {NULL,NULL} end marker (with src=0)
#if 0
    /** src files (useless?).
     * \c syms[0<=i<len].src is client-specified DllFile::tag */
    size_t   nSrc;
#endif
} CjitSyms;

struct CjitOpt {
    char const* jit_dir; ///< NULL --> "tmp_cjitConv"
    int skip_prep; ///< try to skip Makefile 'prep' stage {0/1}
    int skip_make; ///< try to skip jit library 'build' stage {0|1}
    //int openmp;    ///< DllBuild with openmp flags (default if compiled with -fopenmp)
};

/** Hmmm, easiest is C API is to return just array of CjitSym, with
 * all info just copied from C++.  Delete the C++ objects after use.
 * Sure you lose a lot of info, but it's convenient.
 *
 * - \c pParams describes vednn test convolution
 * - \c nParams <B>should be 1 for C API</B>, otherwise you might
 *   have difficulty figuring out which symbol corresponds
 *   to which pParam[i]. (could be corrected, I suppose).
 * - \c dllGeneratorName ARRAY-of-names a C++ code generator routine
 *   - Example: "cjitConvFwd1" means use C++ function cjitConvolutionForward1
 *   - and "cjitConvFwd1" likely appears in the \c sym names of the C functions.
 *   - terminate with NULL
 *
 * \return list of loaded symbols \c sym and their memory addresses \c ptr
 * - returned array terminated by {NULL,NULL} entry.
 * - returned array sorted alphabetically by \c sym
 * - actually, also need to remember some \c opaque info so that
 *   \c cjitSyms_free can dlcose the jit shared library.
 *
 * \c opt NULL (or \c *opt={NULL,0,0}) ~ default jit_dir, full prep, full make
 *
 * \note actual \c DllBuild via `libjit1` is in \ref jitConvs , \ref cjitConv.cpp
 */
CjitSyms * cjitSyms( struct param const* const pParams,
        int const nParams,
        char const** const dllGeneratorNames,
        struct CjitOpt const *opt );

#if 0
/** fold \c other CjitSyms into \c cjs */
void cjitsyms_add( CjitSyms *cjs, CjitSyms const* other );
#endif

/** Free all memory associated with a CjitSym array \c beg .
 * \pre \c cjitsyms must have been obtained from cjitSyms.
 * \post list memory and CjitSym::sym string memory freed.
 */
void cjitSyms_free( CjitSyms const* const cjitsyms );

#if 0 // do not really need iterator api, unless we want to C to retrieve more details

struct CjitHandle_s { void const* ptr; }; ///< opaque handle to C++ JIT machinery
struct CjitIter_s { void const* ptr; };   ///< opaque iterator over symbols

typedef CjitHandle_s CjitHandle;
typedef CjitIter_s CjitIter;

/** C API to jitConv C++ JIT routines is simplified.
 * - \c pParams describes vednn test convolution
 * - \c nParams <B>should be 1 for C API</B>, otherwise you might
 *   have difficulty figuring out which symbol corresponds
 *   to which pParam[i]. (could be corrected, I suppose).
 * - \c dllGeneratorName names a C++ code generator routine
 *   - Example: "cjitConvFwd1" means use C++ function cjitConvolutionForward1
 *   - and "cjitConvFwd1" likely appears in the \c sym names of the C functions.
 * - output is a void* CjitHandle.ptr (or NULL)
 * - you iterate over {\c sym, \c ptr} \c CjitSym's
 *   - using the returned handle and \c cjit_begin/next/free functions
 * - you cast \c CjitSym.ptr to the correct function type and call it.
 *
 * Before returning, this function creates JIT .c files, compiles
 * them into a static library, \c dlopen's the library, and
 * finds addresses of the expected symbols.
 *
 * CjitHandle.ptr [opaque] describes a summary of the known information
 * about the JIT dll.
 *
 * \return CjitHandle.ptr==NULL if there are problems.
 */
CjitHandle cjitConvs( struct param const* const pParams,
        int const nParams,
        char const* dllGeneratorName );

/** free memory used by cjitConvs handle.
 * Call this after all \c cjit_begin have been \c cjit_free'd.
 * \post cjithandle->ptr is NULL and memory freed
 * Invalidates all instances of \c cjit_iter(cjithandle).
 *
 * However, since the symbols have been loaded, you should be able
 * to keep your own version of \c cjit_at pointers-to-symbols.
 */
void cjitConvs_free( CjitHandle* cjithandle );

/** begin iterating over symbols.
 * The symbols will be returned in alphabetical order. */
CjitIter cjit_iter( CjitHandle* cjitHandle ); 

/** is iterator usable?
 * \return true iff cjit_at(cjitIter) should be a valid {sym,ptr}. */
int cjit_iter_empty( CjitIter const cjitIter );

/** retrieve current {symbol,ptr} pair
 * \pre CjitIter obtained from \c cjit_begin
 * \return {NULL,NULL} at end of iteration.
 * Expect client to cast the CjitSym.ptr to correct function type.
 */
CjitSym cjit_at( CjitIter cjitIter );

/** Advance a \c CjitIter [if possible].
 * \return bool "ok" (\e not at end) status. */
int cjit_next( CjitIter cjitIter );

/** return an end marker for \c cjit_at(CjitIter). */
inline CjitSym cjit_end() {
    CjitSym ret = {NULL,NULL};
    return ret;
}
/** To properly free memory, cjit_begin and cjit_end should
 * be paired. After this call, cjitIter should not be used. */
void cjit_free( CjitIter* cjitIter );
#endif

#ifdef __cplusplus
/* { */
} //extern "C"
#endif
// vim: ts=4 sw=4 et cindent cino=^=l0,\:.5s,=-.5s,N-s,g.5s,b1 cinkeys=0{,0},0),\:,0#,!^F,o,O,e,0=break
#endif // CJITCONV_H
