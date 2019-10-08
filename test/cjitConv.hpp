#ifndef CJITCONV_HPP
#define CJITCONV_HPP
#include "cjitConv.h"       // C frontend [opaque CjitHandle and CjitIter]
#include "dllbuild.hpp"		// C JIT file/dll/symbol objects
#include <cassert>

// If defined, API exposes deprecated _ve_ functionality.
//#define ALLOW_VE_INTRINSICS
#ifndef ALLOW_VE_INTRINSICS
#define ALLOW_VE_INTRINSICS 0
#endif

namespace cprog{
class Cblock; // fwd decl
}

/** 0=full calc, 1=fast calc[w/ unroll bug], 2=induce[UNKNOWN CLANG BUG, even w/o unroll] */
#define VRJ_INDUCE 1
void vrj_init(cprog::Cblock &vec_init);
void vrj_induce(cprog::Cblock &loop_x0);

/** \group DllFile Generators
 * These functions return code strings for JIT impls and symbol info
 */
//@{
/** based on a very short (slow) direct_default3.c, name \e "cjitConvFwd1" */
DllFile cjitConvolutionForward1( struct param const* const p );
/** 1 with ptrs instead of indices */
DllFile cjitConvolutionForward1p( struct param const* const p );
/** 1+ mask precalc (NOT speedup - full mask load is 4 lvm ops) */
DllFile cjitConvolutionForward1b( struct param const* const p );
/** 1p + kernel re-order */
DllFile cjitConvolutionForward1q( struct param const* const p );
/** NEW: mask precalc demo.
 * OLD: hand-unrolled version of direct_default_2p, name \e "cjitConvFwd2" */
DllFile cjitConvolutionForward2( struct param const* const p );
/** sBy [kw] loop blocking speedup */
DllFile cjitConvolutionForward3( struct param const* const p );
/** try pvfmad for sBy again... */
DllFile cjitConvolutionForward4( struct param const* const p );
DllFile cjitConvolutionForward5( struct param const* const p );
/** 1q code cleanup (remove optional flags + readability) */
DllFile cjitConvolutionForward6( struct param const* const p ); // XXX missing unroll calculations from 1q XXX
//DllFile cjitConvolutionForward6vel( struct param const* const p );
//@}

/** C++ counterpart of 'C' CjitOpt.  Hmmm. no \c cjitSyms counterpart? */
struct JitConvsOpt {
    JitConvsOpt();
    JitConvsOpt(std::string jit_dir_,
            bool const skip_prep_=false, bool const skip_make_=false);
    JitConvsOpt(CjitOpt const* cjitOpt); ///< \c cjitOpt NULL also OK
    char const* jit_dir;
    bool skip_prep;
    bool skip_make;
  private:
    static char const* const default_jit_dir; ///< tmp_cjitConv[-x86]
};

inline JitConvsOpt::JitConvsOpt()
    : jit_dir(default_jit_dir)
    , skip_prep(0)
    , skip_make(0)
{}
inline JitConvsOpt::JitConvsOpt(std::string jit_dir_,
        bool const skip_prep_/*=false*/, bool const skip_make_/*=false*/)
    : jit_dir(std::strcpy(new char[jit_dir_.length()+1], jit_dir_.c_str()))
      //strcpy returns first arg
    , skip_prep(skip_prep_)
    , skip_make(skip_make_)
{}
inline JitConvsOpt::JitConvsOpt( CjitOpt const* cjitOpt ) : JitConvsOpt()
{
    if(cjitOpt){
        assert( (cjitOpt->skip_prep==0 || cjitOpt->skip_prep==1)
                && (cjitOpt->skip_make==0 || cjitOpt->skip_make==1)
                && "check that CjitOpt POD struct is fully initialized"!=nullptr );
        if(cjitOpt->jit_dir) jit_dir = const_cast<char const*>(cjitOpt->jit_dir);
        skip_prep = cjitOpt->skip_prep;
        skip_make = cjitOpt->skip_make;
    }
}

/** Generate a DllOpen map of JIT symbols to their addresses.
 * \c pParams are the convolution parameters (possibly several?)
 * \c dllFileGenerator is a function like cjitConvolutionForward1,
 * that generates JIT C + VE intrinsics code.
 */
std::unique_ptr<DllOpen> jitConvs(struct param const* const pParams,
        int const nParams,
        DllFile (*dllFileGenerator)(struct param const* const p),
        JitConvsOpt const opt=JitConvsOpt()
        );
/** Resembles 'C' interface where JIT generator name are looked up in \c dfgNames[].
 * \c generators is null-terminated list of C-string JIT generator names.
 * Warn for unknown generator names. \throw if no valid generators
 */
std::unique_ptr<DllOpen> jitConvs(struct param const* const pParams,
        int const nParams,
        char const** generators, // null terminated array of C-string tags for dfgNames[] lookup
        JitConvsOpt const opt=JitConvsOpt()
        );
/** Sometimes you want to pack JIT functions from several JIT generators
 * into one library. vector of generator function ptrs --> dll. */
std::unique_ptr<DllOpen> jitConvs(struct param const* const pParams,
        int const nParams,
        std::vector<DllFile (*)(struct param const* const p)> const& dllFileGenerators,
        JitConvsOpt const opt=JitConvsOpt()
        );

//
// -------- temporarily here --------------
//

/** often we need to form a mask when an int64_t \c vInt is in range [0,\c sEnd). */
inline std::string vfmk_mvs_0to(std::string vInt, std::string sEnd){
    return "_ve_vfmkl_mcvm(VECC_GE,"+vInt+",      /* >= 0 && < "+sEnd+" */"
        "\n        _ve_vfmkl_mcv (VECC_IG,_ve_vcmpsl_vsv("+sEnd+","+vInt+")))";
}
#if ALLOW_VE_INTRINSICS
/** alt isto use a macro... */
#define VEJ_VFMK_mvs_0_TO( V_INT, S_END ) \
    "_ve_vfmkl_mcvm(VECC_GE, " #V_INT ",    /* >= 0 && < " #S_END " */" \
"\n        _ve_vfmkl_mcv (VECC_IG,_ve_vcmpsl_vsv( " #S_END ", " #V_INT " )))"
#endif
#define VEL_VFMK_mvs_0_TO( V_INT, S_END, S_VL ) \
    "_vel_vfmklge_mvml(" #V_INT ",    /* >= 0 && < " #S_END " */" \
"\n        _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl( " #S_END ", " #V_INT ", " #S_VL")," \
#S_VL ")," #S_VL ")"

#if ALLOW_VE_INTRINSICS
/** declare a new /e __vm512 mask register pair /c VM512,
 * and set its value from VM[i] and VM[i+1] mask registers \c VM256_I and \c VM256_INEXT */
#define VEJ_DECL_VM512( VM512, VM256_I, VM256_INEXT ) \
    "__vm512 " #VM512 ";" \
"\n" #VM512 " = _ve_insert_vm512l(" #VM512 ", " #VM256_I "); /* l ~ VM[i] */" \
"\n" #VM512 " = _ve_insert_vm512u(" #VM512 ", " #VM256_INEXT "); /* u ~ VM[i+1] */"
#endif
#define VEL_DECL_VM512( VM512, VM256_I, VM256_INEXT, S_VL ) \
    "__vm512 " #VM512 ";" \
"\n" #VM512 " = _vel_insert_vm512l(" #VM512 ", " #VM256_I "); /* l ~ VM[i] */" \
"\n" #VM512 " = _vel_insert_vm512u(" #VM512 ", " #VM256_INEXT "); /* u ~ VM[i+1] */"

/** string of code for kernel loop limits.
 *
 * \pre IY is \e first input image coord, such as i=y*stridHeight-padHeight,
 *      where y is an output pixel.
 *
 * Ex: first set i=y*stridHeight-padHeight and then output string
 *     K_BEG_END(kh_beg,kh_tmp,kh_end, kernHeight,i,inHeight,dilationHeight)
 *
 * On down-side, if we had JIT constants for KDIM,IY,IDIM,DIL (and stride and pad)
 * we could generate shorter code, less difficult for compiler to optimize
 */
#define VEJ_K_BEG_END(K_BEG,K_TMP,K_END,  KDIM,IY,IDIM,DIL) \
    "\nint64_t " #K_END "=0;" \
"\nconst int64_t " #K_TMP " = " #DIL "-" #IY "-1;" \
"\nconst int64_t " #K_BEG " = (" #IY ">=0? 0: " #K_TMP " / " #DIL ");" \
"\nif (" #IY " < " #IDIM "){" \
"\n  " #K_END " = (" #IDIM " + " #K_TMP ") / " #DIL ";" \
"\n  if (" #K_END " >= " #KDIM " ) " #K_END " = " #KDIM " ;" \
"\n}"

/** suggest a big vector length that sometimes can redistribute latency of
 * vector operations more equitably. When important convolution speedups
 * of 20% have been observed simply by not using \c MVL.
 *
 * Usage:
 * ```
 * int64_t const vl_init = ve_vlen_suggest(nitems);
 * bool const vl_chklast = nitems>vl_init && nitems%vl_init != 0;
 * ```
 * followed by a loop that might resemble:
 * ```
 * _ve_lvl(vl_init);
 * for(int i=0; i<nitems; i+=vl_init){
 *   if(vl_chklast)
 *     _ve_lvl( i+vl_init > nitems? nitems-vl_init: vl_init );
 * }
 * ```
 * For latency, we round up the equitable vector length to a multiple
 * of 32.  But this can be modified.  If \c nitems can be a perfect
 * multiple of vector length without increasing the loop count, we'll
 * use that value instead.
 *
 * A perfect multiple means \e last-time-through-loop checks
 * and instructions get simplified, or even disappear. When multiples
 * are exact divisions, or power-of-two divisions, easier methods fo
 * fused-loop vectorizations become available.
 *
 * \sa DEFINE_UNROLL for a related task of dividing a loop upper
 * limit equitably for purposes of unrolling, where the best value
 * is the lowest equitable split [fewer constraints].
 */
int64_t ve_vlen_suggest(int64_t const nitems);
/** ve_vlen_suggest without the 'roundup up to mult of 32' step. */
int64_t ve_vlen_suggest_equ(int64_t const nitems);

#if defined(ALLOW_VE_INTRINSICS)
/** \deprecated Please add an extra 'vl' string or int arg for _vel_ intrinsics. */
cprog::Cblock& ve_pvfadd(cprog::Cblock& cb, std::vector<std::string> const& regs);
#endif
/** pvfadd all \c regs into each other, final sum into \c regs[0].
 * \p cb        Cblock code node
 * \p regs      list of packed-float register names to sum
 * \p vl        string or int vector length
 *
 * \pre No checks for non-duplication of registers.
 * \post Many register values will change to hold partial sums.
 * \return cb
 */
cprog::Cblock& ve_pvfadd(cprog::Cblock& cb, std::vector<std::string> const& regs, std::string vl);
cprog::Cblock& ve_pvfadd(cprog::Cblock& cb, std::vector<std::string> const& regs, int const vl);

/** \group unroll support */
//@{
/** example:
 * ```
 * int64_t const max_unroll_outer = (KH_BE<=1 || maskW ? 12: 16);
 * int64_t sofar = 1;
 * DEFINE_UNROLL(un_s , max_unroll_outer, sofar, kernWidth);
 * DEFINE_UNROLL(un_r , max_unroll_outer, sofar, kernHeight);
 * // etc for more outer loops
 * ```
 * Note: approximately, this is min(max( MAX_UNROLL/SOFAR, 1), LOOPLIM),
 * but modified to give lower unrolls where it doesn't affect loop count
 *
 * Ex. MAX_UNROLL/SOFAR = 8 and LOOPLIM = 11 should use unroll (11+1)/2 = 6
 *     which is a bit smaller than the naive unroll(8))
 *
 * We wish to find the \e smallest \c u s.t. the same number of loops are
 * executed as when we choose the largest permitted value.
 *
 * Mathematically, we want
 * \f$\operatorname*{arg\,min}_u \lceil N/u \rceil = \lceil N/M \rceil\f$
 *
 * - You can think of this \c u as a good \e equi-dividing value for a loop.
 *
 * - While similar reasoning \e might apply to selecting a vector length, there
 *   are differences for loop fusion, where exact or "nice" [power-of-two]
 *   divisibility issues might also become important.
 *
 * \todo formula that takes an exact-division value if possible, because in
 * some cases you can add an additional pass through the loop to avoid a
 * \e break-out check in the loop [beneficial probably only for very fast
 * loop content?].
 */
#define CALC_UNROLL(MAXUN,LOOPLIM) \
            ( ((LOOPLIM) == 0 || (MAXUN) <= 0) \
                    ? 0 /* allow some corner cases? */ \
                    : (  (LOOPLIM) + (((LOOPLIM)-1)/(MAXUN)) ) \
                    / ( ((LOOPLIM) + (MAXUN) - 1)  /(MAXUN)  ) \
                    )
#define CALC_UNROLL_ASSERT(CONSTVAR,MAXUN,LOOPLIM) \
            assert(   (( ((LOOPLIM) == 0 || (MAXUN) <= 0) && CONSTVAR==0)) \
                    || (!((LOOPLIM) == 0 || (MAXUN) <= 0) && CONSTVAR>=1 && CONSTVAR<=(MAXUN)) )
#define DEFINE_UNROLL(CONSTVAR,MAXUNROLL,SOFAR,LOOPLIM) \
            int64_t const CONSTVAR = CALC_UNROLL( (MAXUNROLL)/(SOFAR), LOOPLIM); \
            CALC_UNROLL_ASSERT(         CONSTVAR, (MAXUNROLL)/(SOFAR), LOOPLIM); \
            SOFAR *= CONSTVAR;
#define DEFINE_NOUNROLL(CONSTVAR,...) int64_t const CONSTVAR = 0 /*explicit nounroll*/
#define DEFINE_DEFUNROLL(CONSTVAR,...) int64_t const CONSTVAR = -1 /*unspecified unroll*/
//#define DEFINE_UNROLL_ORIG(CONSTVAR,SOFAR,LOOPLIM) \
//            int64_t const CONSTVAR = min<long>(max<long>(max_unroll_outer/(SOFAR), 1), LOOPLIM); \
//            SOFAR *= CONSTVAR;
#if 0 // moved to cblock.hpp
/** Usage \c OSSFMT(UNROLL(u)"for(...)"). */
#define UNROLL(N) ve_pragma_unroll(N)<<
#define NO_UNROLL(...) "#pragma nounroll\n"
/** \return empty if N<0, #pragma nounroll if N==0, or #pragma unroll(N). */
std::string ve_pragma_unroll(int64_t const N);
#endif

#define CBLOCK_FOR(CBVAR,UN_ROLL,INTRO,CBPARENT) \
  CBLOCK_SCOPE(CBVAR,OSSFMT(UNROLL(UN_ROLL) INTRO),CBPARENT.getRoot(),CBPARENT)

//@} 
// vim: ts=4 sw=4 et cindent cino=^=l0,\:.5s,=-.5s,N-s,g.5s,b1 cinkeys=0{,0},0),\:,0#,!^F,o,O,e,0=break
#endif // CJITCONV_HPP
