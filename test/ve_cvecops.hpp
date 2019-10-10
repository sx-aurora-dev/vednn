#ifndef VE_CVECOPS_HPP
#define VE_CVECOPS_HPP
#include "stringutil.hpp"

/** \file
 * common VE+intrinsics code string productions
 *
 * Most routines are not thread safe (they may share string formatting buffer)
 */
#include <sstream>

#ifndef MVL
#define MVL 256
#endif

#ifndef ALLOW_VE_INTRINSICS
#define ALLOW_VE_INTRINSICS 0
#endif
/** for length \c n > 0 return a nice [1,256] vector length.
 * - For now:
 *   - if n%ivl==0 and ivl does not increase the loop count, return ivl
 *   - o/w return MVL
 * - Why?
 *   - zero remainder means we can avoid loop exit fixup code for "remainder"
 *
 * Other strategy to check is to favor remainder lengths that have the
 * same number of iterations, but where the remainder has some target size
 * (e.g. there might be latency effects when vector length hits a next
 * multiple of 32 (need to check)).
 */
#if 1
int64_t nice_vector_length( int64_t const n );
/// \group simple JIT string productions
/// - \c stride and \c offset args are in 4- or 8-byte units [not bytes]
///   according to the load/store element.
/// - When \c vl, \c offset or stride are supplied as compile-time integers
///   instead of strings, we might generate better code.
/// - I think there is still some optimization to be done here.
//@{

/// vrPacked = _ve_vshf_vvvs(vruA, vruB, VE_VSHUFFLE_YUZU);
#if ALLOW_VE_INTRINSICS
std::string vyuzu(std::string vrPacked, std::string vruA, std::string vruB);
#endif
/// vel form
std::string vyuzu(std::string vrPacked, std::string vruA, std::string vruB, std::string vl);
std::string vyuzu(std::string vrPacked, std::string vruA, std::string vruB, int64_t const vl);

/// vr = _ve_vldu_vss(4*stride, (float*)(void*)(ptr32));
#if ALLOW_VE_INTRINSICS
std::string vloadu32(std::string vr, std::string ptr32, std::string stride, int64_t const offset=0);
std::string vloadu32(std::string vr, std::string ptr32, std::string stride, std::string offset);
#endif
/// vel forms
std::string vloadu32(std::string vr, std::string ptr32, std::string stride, std::string vl, int64_t const offset);
std::string vloadu32(std::string vr, std::string ptr32, std::string stride, std::string vl, std::string offset);
std::string vloadu32(std::string vr, std::string ptr32, std::string stride, int64_t const vl, int64_t const offset);
std::string vloadu32(std::string vr, std::string ptr32, std::string stride, int64_t const vl, std::string offset);

/// vr = _ve_vldl_vss(4*stride, (float*)(void*)(ptr32));
#if ALLOW_VE_INTRINSICS
std::string vloadl32(std::string vr, std::string ptr32, std::string stride, int64_t const offset=0);
std::string vloadl32(std::string vr, std::string ptr32, std::string stride, std::string offset);
#endif
std::string vloadl32(std::string vr, std::string ptr32, std::string stride, std::string vl, int64_t const offset);
std::string vloadl32(std::string vr, std::string ptr32, std::string stride, std::string vl, std::string offset);
std::string vloadl32(std::string vr, std::string ptr32, std::string stride, int64_t const vl, int64_t const offset);
std::string vloadl32(std::string vr, std::string ptr32, std::string stride, int64_t const vl, std::string offset);

/// vr = _ve_vst_vss(8*stride, (uint64_t*)(void*)(ptr64));
#if ALLOW_VE_INTRINSICS
std::string vstore64(std::string vr, std::string ptr64, std::string stride, int64_t const offset=0);
std::string vstore64(std::string vr, std::string ptr64, std::string stride, std::string offset);
#endif
/// vel form
std::string vstore64(std::string vr, std::string ptr64, std::string stride, std::string vl, int64_t const offset);
std::string vstore64(std::string vr, std::string ptr64, std::string stride, std::string vl, std::string offset);
std::string vstore64(std::string vr, std::string ptr64, std::string stride, int64_t const vl, int64_t const offset);
std::string vstore64(std::string vr, std::string ptr64, std::string stride, int64_t const vl, std::string offset);
//@}
#endif
/* [JIT] Merge two strided 4-byte vectors \c a32 and \c b32
 *       into an 8-byte-aligned \c dst.
 *
 * \b Function: [produce code to] merge two strided 4-byte memory vectors
 * \f$\vec{a}\f$, \f$\vec{b}\f$ by interleaving them and writing a vector
 * of strided 8-byte values \f$\overrightarrow{a_ib_i}\f$.
 *
 * \b Operation:
 *   \c vej_vmerge32( a32,aBy, b32,bBy, n, dst,dBy, nullptr, "")
 *
 *   for(i=0..n-1) dst[i*dBy] = {a32[i*aBy], b32[i*bBy]},
 *
 *   Where a32 is written into the MSBs and b32 into the LSBs.
 *
 *   API:
 * ```
 * vej_vmerge32( void const* a32, int64_t aBy,
 *               void const* b32, int64_t const bBy,
 *               int const n,
 *               uint64_t* const dst, dBy=1
 *               );
 * ```
 * For JIT, \c a32, \c b32 and \c dst are \e string quantities, while the integer
 * constants are known at compile time.  We return a code string
 *
 * \param a32       name of source, 4-byte alignment, 4-byte units
 * \param aBy       integer source stride, in 4-byte-units (1 is packed)
 * \param b32       name of source, 4-byte alignment, 4-byte units
 * \param bBy       integer source stride, in 4-byte-units (1 is packed)
 * \param n         how many {a[i],b[i]} units to merge [interleave] into destination
 * \param dst       name of destination, \b 8-byte alignment, 8-byte units
 * \param dBy       integer destination stride, in \b 8-byte-units (1 is packed)
 * \param vlen      if NULL, modify vector length at will; on entry; otherwise,
 *                  one entry, nonzero guarantees a known vector length; and
 *                  on exit, <em>if we need to set the vector length</em>, then
 *                  \c *vlen returns the final vector length we used.
 *                  [default = nullptr]
 * \param s         register name disambiguation suffix [default = ""]
 *
 * Implementation: approx. vldu(a,aBy), vldlzx(b,bBy), vmrg, vst(dst,dBy)
 *
 * - stride a/bBy is in 4-byte units, while dBy stride is in 8-byte units
 * - a32 is written to MSBs of the 8-byte outputs, so memory order may
 *   be machine dependent FIXME
 * - memory order can be twiddled by swapping a32 and b32
 *   to simulate _ve_pack_f32p ---
 *
 * Example:
 *
 * ```
 * vej_vmerge32( pKerValue+inChannelGroup*kernHW, pKerValue, kernHW, inChannelGroup,
 *               &k2out.pair, 1 )
 * k2Out += inChannelGroup
 * ```
 *
 * \todo both vcopy32 and vmerge32 can be generalized to an mkl-dnn format conversion
 *       of style 'gchw' --> 'ghwC2'
 *
 */
#if ALLOW_VE_INTRINSICS
std::string vej_vmerge32(
        std::string a32, int64_t const aBy,
        std::string b32, int64_t const bBy,
        int64_t const n,
        std::string const dst,
        int64_t const dBy,
        int64_t *vlen=nullptr,  // nonzero if known, to perhaps avoid uselessly setting vector length
        std::string s="" // register name disambiguation suffix
        );
#endif

/** vel version of vej_vmerge32. \sa vej_vmerge32,
 * except no vlen arg (in general, may clobber VL). */
std::string vel_vmerge32(
        std::string a32, int64_t const aBy,
        std::string b32, int64_t const bBy,
        int64_t const n,
        std::string const dst,
        int64_t const dBy,
        std::string s="" // register name disambiguation suffix
        );

#if 0 // multiple sources with shared interleave is common case and could be generalized
template<typename... Targs>
std::string vej_vmerge32(
        std::string const dst, int64_t const dBy,
        int64_t const n,        // length of each input vector
        std::string s,          // register name disambiguation suffix
        Targs... Fargs          // even number of ptr32 arguments
        )
#endif

/** copy n 32-bit values from \c src+i*sstride to \c dst with dstride
 * \pre pointers 4-byte aligned and strides measured in u32 (not asm-friendly bytes)
 * - warn if strides same ?
 * - how to return if VL register was clobbered?
 * - **assume** src,dst nonoverlapping (restrict ptrs in C code)
 *
 * If dstride==1, then is it faster to do packed shuffle and single output op?
 *
 * Exit with src and dst pointers unchanged?
 *
 * Equiv `for(i=0..n-1) dst[dstride/4] = src[sstride/4]`
 *
 * \todo vej_vcopy32 is non-optimal if 'n' is small and inside an enclosing loop.  This
 * can sometimes be done nicer with VGT/VSC,  or reordering the loop.  Ultimately we really
 * want a MATRIX re-ordering dst[i][j] <-- src[j][i].    ... or even vld2d?
 */
#if ALLOW_VE_INTRINSICS
std::string vej_vcopy32(std::string src, int64_t const sstride, int64_t const n,
                        std::string dst, int64_t const dstride, int64_t& vlen);
#endif
/** vel version of vej_vcopy32 (jit vector copy). vlen may change. I think
 * it is now up to compiler to recognize if vl changed.  I see no way to
 * avoid VL change/not optimization. */
std::string vel_vcopy32(std::string src, int64_t const sstride, int64_t const n,
                        std::string dst, int64_t const dstride);

#if 0 // old code ... arg pack ???
template<typename... Targs>
static std::string vej_vmerge32(
        std::string const dst, int64_t const dBy,
        int64_t const n,        // length of each input vector
        std::string s,          // register name disambiguation suffix
        Targs... Fargs          // even number of ptr32 arguments
        )
{
    cout<<" "<<__PRETTY_FUNCTION__<<endl; cout.flush();
    int const v = 1;
    Cunit tmp("tmp");
    tmp.v = 0; // not verbose
    std::ostringstream oss;
    //CBLOCK_SCOPE(vmerge32, "", tmp, tmp.root);
    auto& vmerge32 = tmp.root;
    oss<<" // merge (interleave) "<<n<<" u32"
        <<"\n // from "<<a32<<" stride "<<aBy
        <<"\n // and  "<<b32<<" stride "<<bBy
        <<"\n // to u64 "<<dst<<", stride "<<dBy<<"\n";
    if(v>=1) cout<<oss.str();
    vmerge32<<oss.str(); oss.str("");

    vmerge32 CONST1(aBy) CONST1(bBy) CONST1(dBy);
    CBLK(vmerge32,"__vr u"<<s<<", l"<<s<<", ul"<<s<<";");

    auto vopOnce [[maybe_unused]] = [&](Cblock& blk,int const i0){
        if(i0==0){
            CBLK(blk,"u"<<s<<" = _ve_vldu_vss(4*aBy, (float*)(void*)("<<a32<<"));");
            CBLK(blk,"l"<<s<<" = _ve_vldu_vss(4*bBy, (float*)(void*)("<<b32<<"));");
            CBLK(blk,"ul"<<s<<" = _ve_vshf_vvvs(u"<<s<<", l"<<s<<", VE_VSHUFFLE_YUZU);");
            CBLK(blk,"_ve_vst_vss(ul"<<s<<", 8*dBy, (double*)(void*)("<<dst<<"));");
        }else{
            CBLK(blk,"u"<<s<<" = _ve_vldu_vss(4*aBy, (float*)(void*)("<<a32<<") + "<<i0<<"*aBy );");
            CBLK(blk,"l"<<s<<" = _ve_vldu_vss(4*bBy, (float*)(void*)("<<b32<<") + "<<i0<<"*bBy );");
            CBLK(blk,"ul"<<s<<" = _ve_vshf_vvvs(u"<<s<<", l"<<s<<", VE_VSHUFFLE_YUZU);");
            CBLK(blk,"_ve_vst_vss(ul"<<s<<", 8*dBy, (double*)(void*)("<<dst<<") + "<<i0<<"*dBy );");
        }
    };
    auto vopOnceStr [[maybe_unused]] = [&](Cblock& blk, string i0){
        CBLK(blk,"u"<<s<<" = _ve_vldu_vss(4*aBy, (float*)(void*)("<<a32<<") + ("<<i0<<")*aBy );");
        CBLK(blk,"l"<<s<<" = _ve_vldu_vss(4*bBy, (float*)(void*)("<<b32<<") + ("<<i0<<")*bBy );");
        CBLK(blk,"ul"<<s<<" = _ve_vshf_vvvs(u"<<s<<", l"<<s<<", VE_VSHUFFLE_YUZU);");
        CBLK(blk,"_ve_vst_vss(ul"<<s<<", 8*dBy, (double*)(void*)("<<dst<<") + ("<<i0<<")*dBy );");
    };
    // vopMany assumes 'vlen' and 'reps' refer to meaningful constants/#defines
    auto vopMany [[maybe_unused]] = [&](Cblock& blk, std::string vlen, int64_t const reps){
        blk>>"NO_SET_VLEN(vlen);";
        CBLOCK_SCOPE(vl,"for(int64_t i=0; i<"<<asDec(reps); ++i)",tmp,blk);
        CBLK(vl,"u"<<s<<" = _ve_vldu_vss(4*aBy, (float*)(void*)("<<a32<<") + i*vlen*aBy);");
        CBLK(vl,"l"<<s<<" = _ve_vldu_vss(4*bBy, (float*)(void*)("<<b32<<") + i*vlen*bBy);");
        CBLK(vl,"ul"<<s<<" = _ve_vshf_vvvs(u"<<s<<", l"<<s<<", VE_VSHUFFLE_YUZU);");
        CBLK(vl,"_ve_vst_vss(ul"<<s<<", 8*dBy, (double*)(void*)("<<dst<<") + i*vlen*dBy);");
    };
    // NB for more "generic-looking" JIT output could just as well produce the entire loop
    auto vopScalarUnroll = [&](Cblock& blk, int64_t const i0, int64_t const i){
        CBLK(blk,"((uint64_t*)(void*)("<<dst<<"))["<<i0<<"+"<<i<<"*dBy]\n"
                "   = (((uint64_t) ( ((uint32_t*)(void*)("<<a32<<")) ["<<i0<<"+"<<i<<"*aBy] )) << 32)\n"
                "   | (((uint64_t) ( ((uint32_t*)(void*)("<<b32<<")) ["<<i0<<"+"<<i<<"*bBy] ))      )"
                ";");
    };
    //int const max_unroll = 4; // XXX TODO
    bool ok=false;
    bool vl_clobbered = 0;
    if(n<=2/*?*/){
        for(int64_t i=0; i<n; ++i)
            vopScalarUnroll(vmerge32,0,i);
        ok = true;
    }else if(n <= MVL){
        CBLK(vmerge32,"NO_SET_VLEN("<<n<<"); /* single load-store is enough */");
        vopOnce(vmerge32,0);
        vl_clobbered = true;
        ok = true;
    }else{
        // LATER: XXX int64_t const vlen = exactFulls( MVL, n );
        // to what vlen can we go down to and still have same # ops?
        // includes easy case of n%MVL==0
        // Note: it is possible that certain values of vlen (256, 256-32 ?) are much better
        //       than others.  and even a "remainder" impl might be prefered if the final
        //       remainder is small (because this might decrease to vector-op pipeling length
        //       as we return?)   Would need measurements!
        // i.e. is vlen 64 "just as fast" for load/store as vlen 256 ?
        // How can we control these to be "overtaking"?
        int64_t const nFull = (n+MVL-1)/MVL;
        int64_t const minvlen = n/nFull;
        if(v>=1) cout<<" nFull="<<nFull<<" minvlen="<<minvlen;
        int64_t vlen;
        for(vlen=MVL; !ok && vlen>=minvlen; --vlen){ // largest vlen exactly subdividing n?
            if(n%vlen==0){
                if(v>=1) cout<<" n%vlen = "<<n<<"%"<<vlen;
                ok=true;
                break;
            }
        }
        if(ok){
            if(v>=1) cout<<endl;
            int64_t fulls = n/vlen;
            assert( fulls > 0 );
            assert( fulls == nFull ); // logic error?
            CBLK(vmerge32,"int64_t const vlen = "<<vlen<<"; /* n = vlen*reps = "<<vlen<<" * "<<fulls<<" */");
            CBLK(vmerge32,"int64_t const reps = "<<fulls<<";");
            vopMany(vmerge32, "vlen", fulls);
            vl_clobbered = true;
        }
    }
    if(!ok) { // fall back to vlen 256 with remainder
        int64_t const fulls = n/MVL;
        assert( fulls >= 1 );
        int64_t const rem = n%MVL;
        // if rem too small is there benefit to extended exact-divisibity search? XXX
        assert( rem > 0 );
        CBLK(vmerge32,"NO_SET_VLEN("<<MVL<<"); /* n="<<fulls<<"*MVL+"<<rem<<" */");
        vl_clobbered = true;
        if( fulls == 1 ){
            vopOnce(vmerge32["one_MVL"],0);
        }else{
            CBLK(vmerge32,"int64_t const vlen = "<<setw(4)<<MVL <<"; /* n = vlen*reps ("<<MVL<<" * "<<fulls<<") */");
            CBLK(vmerge32,"int64_t const reps = "<<setw(4)<<fulls<<"; /*   + remainder ("<<rem<<") */");
            vopMany(vmerge32, "vlen", fulls);
        }
        auto& vend = vmerge32["end"]; // after all of the above... (introduces new subnode)
        // could recurse for the remainder...
        int64_t cnt = fulls*MVL;    // current 4-byte output position ('rem' more to finish all 'n')
        CBLK(vend,"// process last "<<rem<<" items, beginning at item# "<<cnt);
        if(rem<2/*?*/){
            for(int64_t i=0; i<n; ++i)
                vopScalarUnroll(vmerge32,fulls*MVL,i);
        }else{
            CBLK(vend,"NO_SET_VLEN("<<rem<<"); /* single load-store is enough */");
            vopOnce(vend,fulls*MVL);
        }
        vl_clobbered = true;
        ok = true;
    }
    vmerge32["end"] FREE1(dBy) FREE1(bBy) FREE1(aBy);
    //return vl_clobbered;
    return tmp.str();
}
#endif // vmerge32 arg pack

// vim: ts=4 sw=4 et cindent cino=^=l0,\:.5s,=-.5s,N-s,g.5s,b1 cinkeys=0{,0},0),\:,0#,!^F,o,O,e,0=break
#endif // VE_CVECOPS_HPP
