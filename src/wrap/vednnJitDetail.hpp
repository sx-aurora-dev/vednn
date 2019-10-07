#ifndef VEDNNJITDETAIL_HPP
#define VEDNNJITDETAIL_HPP
/** \file
 * preliminary api, unimplemented.
 * void* jitctx in C API is really a C++ structure
 */
#include "vednnx.h"
#include "vednnJit.h"   // vednnSym_t
#include <assert.h>

/** max # of open dll handles */
#define VEDNNJIT_MAXLIBS 64

namespace vednn {
namespace wrap {

#if 0
/** a tiny bloom filter to hash a small set of 64-bit values. */
struct TinyBF64 {
    constexpr uint64_t r1() {return uint64_t{7664345821815920749ULL}; }
    constexpr uint64_t h1(uint64_t const x) {return x*r1()+0x59595ULL; }
    constexpr uint64_t h2(uint64_t const x) {return x*r2() }
    TinyBF64() : bf(0U);
    uint64_t bf;
    constexpr uint64_t scramble(uint64_t u) {
        return h2(h1(u));
    }
}
#endif
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
 *
 * \c VednnJitCtx is \b not thread-safe.
 */
struct VednnJitCtx final {
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
    VednnJitCtx( char const* const process_subdir,
            char const* const process_libpath );
    ~VednnJitCtx();

    struct Libs {
        char const* libPath; ///< copy of \c vednnJitLib \c libpath argument
        void * dlHandle;     ///< handle of successful dlopen
        uint64_t seq;        ///< stale iff symbol seq not match this
        struct Libs* next;   ///< forward list
    };

    /** add a new dll to tail of \c Libs
     * - no-op on dlopen error, printing message.
     * - can reopen existing Libs entry.
     * - \b no list compaction of stale entries yet.
     *   (for more efficient \c mkseqs() and \c contains(sq))
     */
    void addlib(char const *libPath);

    /** fast check for symbol validity. */
    bool contains( uint64_t const sq ) const {
        bool found = false;
        for(size_t i=0U; i<nseqs; ++i ){
            if( seqs[i] == sq ){
                found = true;
            }
        }
        return found;
    }

private:
    /** construct raw Libs entry, without yet attaching it anywhere.
     *  (a 'new Libs' constructor would lack access to this->seq). */
    Libs * newLibs(char const* libPath);
    /** close and reopen, marking as stale */
    void reopen( Libs* stale );
    /** delete a Libs* completely (not just dlcose and mark stale). */
    static void del( Libs* old );
    /** new char[] copy of input c_str */
    static char const* dup(char const* src);


    /** Sequence counter for \c Libs::dlHandle pointers */
    uint64_t seq;
#if 0
    /** hash of all usable Libs::seq values, for \c contains(seq).
     * \c seqhash() to update this after Libs content changes.
     * We expect only a small number of libraries, so a tiny Bloom
     * Filter suffices. */
    TinyBF64 seqbf;
#else
    /** set seqs[nseqs] to list of all open dlHandle (those
     * Libs whose \c seq is nonzero). \return false iff too many
     * open dlls. */
    bool mkseqs();
    uint64_t nseqs;
    /** array of seq values for all valid dlHandles, same as
     * all traversing from root and getting all nonzero \c seq. */
    uint64_t seqs[VEDNNJIT_MAXLIBS];
#endif
    /** jit build subdir */
    char const* process_subdir;
    /** jit process dll (jit symbol accumulator) */
    char const* process_libpath;

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
    Libs root;

};

/** Temporary wrapper to ease use of \c vednnSym_t and its opaque content.
 * - Typical pointer-wrapper lifetime considerations apply.
 * - public members only a new[] \symbol and a \c void*\ addr
 * \ref vednnJitDetail.h for \c opaque data.
 */
struct Sym {
private:
    vednnSym_t& sy;         ///< ref to symbol
    //vednnSymDetail_t& op;   ///< ref to symbol.opaque [properly typed]
public:
    vednnSymDetail_t      & op()        { return sy.opaque; }
    vednnSymDetail_t const& op()  const { return sy.opaque; }
    vednnSymDetail_t const& cop() const { return sy.opaque; }
    VednnJitCtx           & ctx() {
        return *reinterpret_cast<VednnJitCtx*>(op().jit_ctx);
    }
    VednnJitCtx      const& ctx() const {
        return *reinterpret_cast<VednnJitCtx*>(op().jit_ctx);
    }
    VednnJitCtx      const& cctx() const {
        return *reinterpret_cast<VednnJitCtx*>(op().jit_ctx);
    }
    //Sym(vednnSym_t *s) : sy(*s) {}
    Sym(vednnSym_t &s) : sy( s) { assert(&s); }
    ~Sym() {}

    /** zero the symbol, free memory. */
    void erase(){
        delete[] sy.symbol;
        sy.symbol = nullptr;
        sy.addr = nullptr;
        op().addr = nullptr;
        //op().jit_ctx = nullptr; // const member
        op().seq = 0;
        op().rtok = nullptr;
        op().addrNext = nullptr;
        op().haveParams = 0;
    }

    // basic fastcall is that sy.addr is non-NULL.
    // When libraries get reloaded, however, the op().seq is
    // no longer recognizable with the jit_ctx and this also
    // means that addr is not usable.
    bool fastcall() const {
        return sy.addr && cctx().contains(cop().seq);
    }
};
}}//vednn::wrap::
// vim: et ts=4 sw=4 cindent cino=^0,=0,l1,\:0,=s,N-s,g-s,hs syntax=cpp.doxygen
#endif // VEDNNJITDETAIL_HPP
