#ifndef VEDNNJITDETAIL_HPP
#define VEDNNJITDETAIL_HPP
/** \file
 * preliminary api, unimplemented.
 * void* jitctx in C API is really a C++ structure
 */
#include "vednnx.h"
#include "nstl.hpp"

/** Jit context is mostly a table of handles that \c dlsym can
 * use to \e resolve symbols, usually used to mapping symbols
 * to function addresses.
 *
 * These will usually be JIT functions, but might also resolve
 * to functions built into libvednn or libvednnx (in global
 * process namespace).
 *
 * It also contains lists of jit generators, where the \b _ok match
 * provides a preferred jit generator.  Not sure yet how lazy the
 * jit generation process should be (original test/ code generated
 * "all" jit impls).
 */
struct VednnJitCtx {
    /** Initialize a context for symbol resolution.
     * \c throw if subdir is not writable [perhaps also if
     * libpath is not creatable or writable]
     *
     * You can supply nullptr for either to forcibly ignore
     * JIT \c generators.  (This still allows access to any jit
     * functions linked into the process itself)
     *
     * The list of jit generator functions is built-in (you cannot
     * yet add your own jit generators).
     */
    VednnJitCtx( char const* process_subdir,
            char const* process_libpath );
    ~VednnJitCtx();

    struct Libs {
        char const* libPath; ///< copy of \c vednnJitLib \c libpath argument
        void * dlHandle;     ///< handle of successful dlopen
        bool stale;          ///< true if libPath might have changed
        struct Libs* next;   ///< forward list
    }
    /** root always has libPath==nullptr and dlHandle to access
     * symbols built into the executable itself (static libs
     * with --whole-archive).
     *
     * \c root forms a forward linked list.
     *
     * Entry one, \c root->next is reserved for newly generated jit functions
     * for this process.  This library is grown by jit generators when we
     * cannot resolve a symbol (or need to go further along the alternate
     * implementation list).  We use \c stale to lazily dlclose and dlopen
     * the library again.  All old symbols who used to be in this library
     * get re-resolved automatically via their \c vednnSymDetail_s::lib_seq.
     *
     * Subsequent calls to \c \c vednnJitLib may add other unchanging dlls to
     * \c Libs lookup path.  Such libraries are searched \b before
     * \c root->next and \c root, in the order they were added.  So symbols
     * bind more strongly to dlls added with \c vednnJitLib.
     *
     *
     * Generators finding their symbols never regenerate sources and run make.
     * To force recreation, run a process without any jit libraries added, and
     * then all required libs will end up in \c process_lib, which can replace
     * any old versions that you had.
     *
     * optional TODO: perhaps add a flag that forces \b all JIT symbols to be
     * recompiled into \c process_lib. (jit generator version changed?)
     */
    Libs *root;

    /** add a new dll to tail of \c Libs
     * no-op on dlopen error, printing message. */
    void addlib(char const *libPath);

};

// vim: et ts=4 sw=4 cindent cino=^0,=0,l1,\:0,=s,N-s,g-2,h2 syntax=cpp.doxygen
#endif // VEDNNJITDETAIL_HPP
