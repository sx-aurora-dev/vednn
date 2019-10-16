
#include "ve_cvecops.hpp"
#include "cblock.hpp"
#include "stringutil.hpp"

using namespace std;
using namespace cprog;

//#define LOCAL_SYMBOL static/*debugging!*/
#define LOCAL_SYMBOL /*in library!*/

#define CVT ""
//#define CVT "(float*)(void*)"
//#define CVT "(float*)(uintptr_t)"

LOCAL_SYMBOL int64_t nice_vector_length( int64_t const n ){
    int64_t ivl;
    bool ok=false;
    // LATER: XXX int64_t const vlen = exactFulls( MVL, n );
    // to what vlen can we go down to and still have same # ops?
    // includes easy case of n%MVL==0
    //
    // Note: it is possible that certain values of vlen, like multiples of 32,
    //       are much better than others.
    //
    //       So even a "remainder" impl might be preferred if the final remainder
    //       is small (because this might decrease to vector-op pipeling length
    //       as we return?)   Would need measurements!
    // i.e. is vlen 64 "just as fast" for load/store as vlen 256 ?
    // How can we control these to be "overtaking"?
    int64_t const nFull = (n+MVL-1)/MVL;
    int64_t const minvlen = n/nFull;
    for(ivl=MVL; !ok && ivl>=minvlen; --ivl){ // largest vlen exactly subdividing n?
        if(n%ivl==0){
            ok=true;
            break;
        }
    }
    if(!ok) ivl = MVL;
    return ivl;
}

#if ALLOW_VE_INTRINSICS
LOCAL_SYMBOL std::string vyuzu(std::string vrPacked, std::string vruA, std::string vruB){
    return vrPacked+" = _ve_vshf_vvvs("+vruA+", "+vruB+", VE_VSHUFFLE_YUZU);";
}
#endif
LOCAL_SYMBOL std::string vyuzu(std::string vrPacked, std::string vruA, std::string vruB, std::string vl){
    return vrPacked+" = _vel_vshf_vvvsl("+vruA+", "+vruB+", VE_VSHUFFLE_YUZU, "+vl+");";
}
LOCAL_SYMBOL std::string vyuzu(std::string vrPacked, std::string vruA, std::string vruB, int64_t const vl){
    return vyuzu(vrPacked,vruA,vruB,asDec(vl));
}

/// vr = _ve_vldu_vss(4*stride, (float*)(void*)(ptr32));
#if ALLOW_VE_INTRINSICS
LOCAL_SYMBOL std::string vloadu32(std::string vr, std::string ptr32, std::string stride, int64_t const offset/*=0*/){
    std::string ret;
    ret.reserve(128);
    ret.append(vr).append(" = _ve_vldu_vss(").append("4*(").append(stride).append("), ");
    if(offset)
        ret.append("(float*)((uintptr_t)(").append(ptr32).append(")")
            .append(" + 4*").append(asDec(offset))
            .append("*(").append(stride).append(")");
    else
        ret.append("(float*)(").append(ptr32).append(")");
    ret.append(");");
    return ret;
}
LOCAL_SYMBOL std::string vloadu32(std::string vr, std::string ptr32, std::string stride, std::string offset){
    if(offset.empty())
        return vloadu32(vr,ptr32,stride,0);
    std::string ret;
    ret.reserve(128);
    ret.append(vr).append(" = _ve_vldu_vss(").append("4*(").append(stride).append("), ");
    std::string byteStride;
    ret.append("(float*)((uintptr_t)(").append(ptr32).append(")")
        .append(" + 4*(").append(offset).append(")*(").append(stride).append("))");
    ret.append(");");
    return ret;
}
#endif
LOCAL_SYMBOL std::string vloadu32(std::string vr, std::string ptr32, std::string stride, std::string vl, int64_t const offset){
    std::string ret;
    ret.reserve(128);
    ret.append(vr).append(" = _vel_vldu_vssl(").append("4*(").append(stride).append("),  ");
    if(offset)
        ret.append("(").append(ptr32).append(")")
            .append(" + ").append(asDec(offset))
            .append("*(").append(stride).append(")");
    else
        ret.append("(").append(ptr32).append(")");
    ret.append(", ").append(vl).append(");");
    return ret;
}
LOCAL_SYMBOL std::string vloadu32(std::string vr, std::string ptr32, std::string stride, std::string vl, std::string offset){
    if(offset.empty())
        return vloadu32(vr,ptr32,stride,vl,0);
    std::string ret;
    ret.reserve(128);
    ret.append(vr).append(" = _vel_vldu_vssl(").append("4*(").append(stride).append("), ");
    ret.append("(").append(ptr32).append(")")
        .append(" + (").append(offset).append(")*(").append(stride)
        .append("), ").append(vl).append(");");
    return ret;
}
LOCAL_SYMBOL std::string vloadu32(std::string vr, std::string ptr32, std::string stride, int64_t const vl, int64_t const offset){
    return vloadu32(vr,ptr32,stride,asDec(vl), offset);
}
LOCAL_SYMBOL std::string vloadu32(std::string vr, std::string ptr32, std::string stride, int64_t const vl, std::string offset){
    return vloadu32(vr,ptr32,stride,asDec(vl),offset);
}
/// vr = _ve_vldl_vss(4*stride, (float*)(void*)(ptr32));
#if 0
LOCAL_SYMBOL std::string vloadl32(std::string vr, std::string ptr32, std::string stride, int64_t const offset/*=0*/){
    std::string ret = vr+" = _ve_vldl_vss(4*"+stride+", (float*)(void*)("+ptr32+")";
    if(offset) ret.append(" + ").append(asDec(offset)).append("*").append(stride);
    ret.append(");");
    return ret;
}
LOCAL_SYMBOL std::string vloadl32(std::string vr, std::string ptr32, std::string stride, std::string offset){
    std::string ret = vr+" = _ve_vldl_vss(4*"+stride+", (float*)(void*)("+ptr32+")";
    if(!offset.empty()) ret.append(" + ").append(offset).append("*").append(stride);
    ret.append(");");
    return ret;
}
#else // stride in 4-byte units
#if ALLOW_VE_INTRINSICS
LOCAL_SYMBOL std::string vloadl32(std::string vr, std::string ptr32, std::string stride, int64_t const offset/*=0*/){
    std::string ret;
    ret.reserve(128);
    ret.append(vr).append(" = _ve_vldl_vss(").append("4*(").append(stride).append("), ");
    if(offset)
        ret.append("(float*)((intptr_t)(").append(ptr32).append(")")
            .append(" + 4*").append(asDec(offset))
            .append("*(").append(stride).append("))");
    else
        ret.append("(float*)(").append(ptr32).append(")");
    ret.append(");");
    return ret;
}
LOCAL_SYMBOL std::string vloadl32(std::string vr, std::string ptr32, std::string stride, std::string offset){
    if(offset.empty())
        return vloadl32(vr,ptr32,stride,0);
    std::string ret;
    ret.reserve(128);
    ret.append(vr).append(" = _ve_vldl_vss(").append("4*(").append(stride).append("), ");
    std::string byteStride;
    ret.append("(float*)((intptr_t)(").append(ptr32).append(")")
        .append(" + (").append(offset).append(")*4*(").append(stride).append(")");
    ret.append(");");
    return ret;
}
#endif
LOCAL_SYMBOL std::string vloadl32(std::string vr, std::string ptr32, std::string stride, std::string vl, int64_t const offset){
    std::string ret;
    ret.reserve(128);
    ret.append(vr).append(" = _vel_vldl_vssl(").append("4*(").append(stride).append("), ");
    if(offset)
        ret.append("(float*)((uintptr_t)(").append(ptr32).append(")")
            .append(" + 4*").append(asDec(offset))
            .append("*(").append(stride).append("))");
    else
        ret.append("(float*)(").append(ptr32).append(")");
    ret.append(", ").append(vl).append(");");
    return ret;
}
LOCAL_SYMBOL std::string vloadl32(std::string vr, std::string ptr32, std::string stride, std::string vl, std::string offset){
    if(offset.empty())
        return vloadl32(vr,ptr32,stride,vl,0);
    std::string ret;
    ret.reserve(128);
    ret.append(vr).append(" = _ve_vldl_vssl(").append("4*(").append(stride).append("), ");
    std::string byteStride;
    ret.append("(float*)((uint64_t)(").append(ptr32).append(")")
        .append(" + 4*(").append(offset).append(")*(").append(stride)
        .append(")), ").append(vl).append(");");
    return ret;
}
std::string vloadl32(std::string vr, std::string ptr32, std::string stride, int64_t const vl, int64_t const offset){
    return vloadl32(vr,ptr32,stride,asDec(vl),offset);
}
std::string vloadl32(std::string vr, std::string ptr32, std::string stride, int64_t const vl, std::string offset){
    return vloadl32(vr,ptr32,stride,asDec(vl),offset);
}
#endif

/// vr = _ve_vst_vss(8*stride, (uint64_t*)(void*)(ptr64));
#if ALLOW_VE_INTRINSICS
LOCAL_SYMBOL std::string vstore64(std::string vr, std::string ptr64, std::string stride, int64_t const offset/*=0*/){
    std::string ret = "_ve_vst_vss("+vr+", 8*("+stride+"), (uint64_t*)((uintptr_t)("+ptr64+")";
    if(offset) ret.append(" + 4*(").append(asDec(offset)).append(")*(").append(stride).append(")");
    ret.append("));");
    return ret;
}
LOCAL_SYMBOL std::string vstore64(std::string vr, std::string ptr64, std::string stride, std::string offset){
    std::string ret = "_ve_vst_vss("+vr+", 8*("+stride+"), (uint64_t*)((uintptr_t)("+ptr64+")";
    if(!offset.empty()) ret.append(" + 4*(").append(offset).append(")*(").append(stride).append(")");
    ret.append("));");
    return ret;
}
#endif
LOCAL_SYMBOL std::string vstore64(std::string vr, std::string ptr64, std::string stride, std::string vl, int64_t const offset/*=0*/){
    std::string ret = "_vel_vst_vssl("+vr+", 8*("+stride+"), "+ptr64;
    if(offset) ret.append(" + (").append(asDec(offset)).append(")*(").append(stride).append(")");
    ret.append(", ").append(vl).append(");");
    return ret;
}
LOCAL_SYMBOL std::string vstore64(std::string vr, std::string ptr64, std::string stride, std::string vl, std::string offset){
    std::string ret = "_vel_vst_vssl("+vr+", 8*("+stride+"), "+ptr64;
    if(!offset.empty()) ret.append(" + (").append(offset).append(")*(").append(stride).append(")");
    ret.append(", ").append(vl).append(");");
    return ret;
}
LOCAL_SYMBOL std::string vstore64(std::string vr, std::string ptr64, std::string stride, int64_t const vl, int64_t const offset/*=0*/){
    return vstore64(vr,ptr64,stride,asDec(vl),offset);
}
LOCAL_SYMBOL std::string vstore64(std::string vr, std::string ptr64, std::string stride, int64_t const vl, std::string offset){
    return vstore64(vr,ptr64,stride,asDec(vl),offset);
}

#if ALLOW_VE_INTRINSICS
/** \TODO [tricky] aBy or bBy 4 load optimizations. 8-byte loads followed by VEX might
 * be faster (but uses a mask register, so not a high priority). for 8-byte loads should be
 * selectively enabled if vector length is sufficiently long to absorb
 * the address alignment check on src/dst addresses. */
std::string vej_vmerge32(
        std::string a32, int64_t const aBy,
        std::string b32, int64_t const bBy,
        int64_t const n,
        std::string const dst,
        int64_t const dBy,
        int64_t *pvlen,  // nonzero if known, to perhaps avoid uselessly setting vector length
        std::string s/*=""*/ // register name disambiguation suffix
        )
{
    static ostringstream oss; // a common (reset-to-empty) string formatting buffer.
    int const v = 0; // verbosity
    // Optionally use a client-side vector length for guarantees and return.
    int64_t unknown_vlen = 0U;  // 0 : always set vlen if we need to.
    int64_t & vlen = (pvlen? *pvlen: unknown_vlen);

    if(v>=2) cout<<" "<<__PRETTY_FUNCTION__<<endl; cout.flush();

    Cunit tmp("tmp");
    tmp.v = 0; // not verbose
    //CBLOCK_SCOPE(vmerge32, "", tmp, tmp.root);
    auto& vmerge32 = tmp.root;
    oss<<" // merge (interleave) "<<n<<" u32"
        <<"\n // from "<<a32<<" stride "<<aBy
        <<"\n // and  "<<b32<<" stride "<<bBy
        <<"\n // to u64 "<<dst<<", stride "<<dBy<<"\n";
    if(v>=1) cout<<oss.str();
    vmerge32<<oss.str(); oss.str("");

    //vmerge32 CONST1(aBy) CONST1(bBy) CONST1(dBy);
    CBLK(vmerge32,"__vr u"<<s<<", l"<<s<<", ul"<<s<<";");
    auto vopOnce [[maybe_unused]] = [&](Cblock& blk, int const _vlen, std::string i0=""){
        if(_vlen != vlen) blk>>"_ve_lvl("+asDec(vlen=_vlen)+");";
        // NOTE: if aBy==bBy and b32==a32+1 then ul can be loaded with a single 64-bit load
        //       (and then this becomes a '64-bit copy', perhaps avoidable)
        blk >>vloadu32("u"+s, a32, asDec(aBy), i0)
            >>vloadu32("l"+s, b32, asDec(bBy), i0)
            >>vyuzu("ul"+s, "u"+s, "l"+s)
            >>vstore64("ul"+s, dst, asDec(dBy), i0);
    };
    // vopMany assumes 'vlen' and 'reps' refer to meaningful constants/#defines
    auto vopMany [[maybe_unused]] = [&](Cblock& blk, int const _vlen, int const reps){
        if( reps == 1 ){
            vopOnce(blk,_vlen);
        }else{
            if(_vlen != vlen) blk>>"_ve_lvl("+asDec(vlen=_vlen)+");";
            CBLOCK_SCOPE(vl,"for(int64_t i=0; i<"+asDec(reps)+"; ++i)",tmp,blk);
            vl  >>vloadu32("u"+s, a32, asDec(aBy), "i*"+asDec(vlen))
                >>vloadu32("l"+s, b32, asDec(bBy), "i*"+asDec(vlen))
                >>vyuzu("ul"+s, "u"+s, "l"+s)
                >>vstore64("ul"+s, dst, asDec(dBy), "i*"+asDec(vlen))
                ;
        }
    };
//#define ADD_OFF_BY_BYTES(PTR,BASE,OFF,BY) \
//    *(uint32_t*)( (uintptr_t)(PTR) + (BASE+OFF)*4*ABY )
    // NB for more "generic-looking" JIT output could just as well produce the entire loop
    auto vopScalarUnroll = [&](Cblock& blk, int64_t const i0, int64_t const i){
        CBLK(blk,"*(uint64_t*)( (uintptr_t)("<<dst<<") "
                " + ("<<i0<<"+"<<i<<")*8*"<<dBy<<")\n"
                "   = ((uint64_t) ( *(uint32_t*)( (uintptr_t)("<<a32<<") "
                " + ("<<i0<<"+"<<i<<")*4*"<<aBy<<") ) << 32)\n"
                "   | ((uint64_t) ( *(uint32_t*)( (uintptr_t)("<<b32<<") "
                " + ("<<i0<<"+"<<i<<")*4*"<<aBy<<") ) << 32)\n"
                ";");
    };
    //int const max_unroll = 4; // XXX TODO
    bool ok=false;
    if(n<=2/*?*/){
        for(int64_t i=0; i<n; ++i) vopScalarUnroll(vmerge32,0,i);
        ok = true;
    }else if(n <= MVL){
        vopOnce(vmerge32,n);
        ok = true;
    }else{
        int64_t ivl = nice_vector_length(n); // find a nice vector length (default = MVL = 256)
        // fall back to vlen 'ivl' , maybe with remainder
        int64_t const fulls = n/ivl;        assert( fulls >= 1 );
        int64_t const rem = n%ivl;          assert( rem >= 0 && rem < MVL );
        CBLK(vmerge32,"// n = "<<fulls<<" * "<<ivl<<" + "<<rem);
        vopMany(vmerge32, ivl, fulls); // this creates a sub-block, so need 'vend'...
        auto& vend = vmerge32["end"];  // (introduces new subnode)
        if(rem){
            if(rem<2/*?*/){
                for(int64_t i=0; i<rem; ++i) vopScalarUnroll(vend,fulls*ivl,i);
            }else{
                s.append("_rem");          // let's allow a second register set
                CBLK(vend,"__vr u"<<s<<", l"<<s<<", ul"<<s<<";");
                vopOnce(vend,rem,asDec(fulls)+"*"+asDec(ivl));
            }
        }
    }
    return tmp.str();
}
#endif

std::string vel_vmerge32(
        std::string a32, int64_t const aBy,
        std::string b32, int64_t const bBy,
        int64_t const n,
        std::string const dst,
        int64_t const dBy,
        std::string s/*=""*/ // register name disambiguation suffix
        )
{
    // NOTE: if aBy==bBy and b32==a32+1 then ul can be loaded with a single 64-bit load
    //       (and then this becomes a '64-bit copy', perhaps avoidable)
    static ostringstream oss; // a common (reset-to-empty) string formatting buffer.
    int const v = 0; // verbosity

    if(v>=2) cout<<" "<<__PRETTY_FUNCTION__<<endl; cout.flush();

    Cunit tmp("tmp");
    tmp.v = 0; // not verbose
    //CBLOCK_SCOPE(vmerge32, "", tmp, tmp.root);
    auto& vmerge32 = tmp.root;
    oss<<" // vel_vmerge32 (interleave) "<<n<<" u32"
        <<" from "<<a32<<" stride "<<aBy
        <<" and  "<<b32<<" stride "<<bBy
        <<" to u64 "<<dst<<", stride "<<dBy<<"\n";
    if(v>=1) cout<<oss.str();
    vmerge32<<oss.str(); oss.str("");

    //vmerge32 CONST1(aBy) CONST1(bBy) CONST1(dBy);
    CBLK(vmerge32,"__vr u"<<s<<", l"<<s<<", ul"<<s<<";");
    auto vopOnce [[maybe_unused]] = [&](Cblock& blk, int const _vlen, std::string i0=""){
#if defined(VEL_BUG)
#if VEL_BUG
        blk>>"_ve_lvl("<<asDec(_vlen)<<");";
#endif
#endif
        blk >>vloadu32("u"+s, a32, asDec(aBy), _vlen, i0)
            >>vloadu32("l"+s, b32, asDec(bBy), _vlen, i0)
            >>vyuzu("ul"+s, "u"+s, "l"+s, _vlen)
            >>vstore64("ul"+s, dst, asDec(dBy), _vlen, i0);
    };
    // vopMany assumes 'vlen' and 'reps' refer to meaningful constants/#defines
    auto vopMany [[maybe_unused]] = [&](Cblock& blk, int const _vlen, int const reps){
        if( reps == 1 ){
            vopOnce(blk,_vlen);
        }else{
            //if(_vlen != vlen) blk>>"_ve_lvl("+asDec(vlen=_vlen)+");";
#if defined(VEL_BUG)
#if VEL_BUG
            blk>>"_ve_lvl("<<asDec(_vlen)<<");";
#endif
#endif
            CBLOCK_SCOPE(vl,"for(int64_t i=0; i<"+asDec(reps)+"; ++i)",tmp,blk);
            vl  >>vloadu32("u"+s, a32, asDec(aBy), _vlen, "i*"+asDec(_vlen))<<" // vopMany"
                >>vloadu32("l"+s, b32, asDec(bBy), _vlen, "i*"+asDec(_vlen))
                >>vyuzu("ul"+s, "u"+s, "l"+s, _vlen)
                >>vstore64("ul"+s, dst, asDec(dBy), _vlen, "i*"+asDec(_vlen))
                ;
        }
    };
    // NB for more "generic-looking" JIT output could just as well produce the entire loop
    auto vopScalarUnroll = [&](Cblock& blk, int64_t const i0, int64_t const i){
        CBLK(blk,"((uint64_t*)(void*)("<<dst<<"))[("<<i0<<"+"<<i<<")*"<<dBy<<"]\n"
                "   = (((uint64_t) ( ((uint32_t*)(void*)("<<a32<<")) [("<<i0<<"+"<<i<<")*"<<aBy<<"] )) << 32)\n"
                "   | (((uint64_t) ( ((uint32_t*)(void*)("<<b32<<")) [("<<i0<<"+"<<i<<")*"<<bBy<<"] ))      )"
                ";");
    };
    //int const max_unroll = 4; // XXX TODO
    bool ok=false;
    if(n<=2/*?*/){
        for(int64_t i=0; i<n; ++i) vopScalarUnroll(vmerge32,0,i);
        ok = true;
    }else if(n <= MVL){
        vopOnce(vmerge32,n);
        ok = true;
    }else{
        int64_t ivl = nice_vector_length(n); // find a nice vector length (default = MVL = 256)
        // fall back to vlen 'ivl' , maybe with remainder
        int64_t const fulls = n/ivl;        assert( fulls >= 1 );
        int64_t const rem = n%ivl;          assert( rem >= 0 && rem < MVL );
        CBLK(vmerge32,"// n = "<<fulls<<" * "<<ivl<<" + "<<rem);
        vopMany(vmerge32, ivl, fulls); // this creates a sub-block, so need 'vend'...
        auto& vend = vmerge32["end"];  // (introduces new subnode)
        if(rem){
            if(rem<2/*?*/){
                for(int64_t i=0; i<rem; ++i) vopScalarUnroll(vend,fulls*ivl,i);
            }else{
                s.append("_rem");          // let's allow a second register set
                CBLK(vend,"__vr u"<<s<<", l"<<s<<", ul"<<s<<";");
                vopOnce(vend,rem,asDec(fulls)+"*"+asDec(ivl));
            }
        }
    }
    return tmp.str();
}


#if ALLOW_VE_INTRINSICS
/** \TODO sstride or dstride 4 optimizations for 8-byte loads should be
 * selectively enabled if vector length is sufficiently long to absorb
 * the address alignment check on src/dst addresses. */
std::string vej_vcopy32(std::string src, int64_t const sstride, int64_t const n,
                        std::string dst, int64_t const dstride, int64_t& vlen)
{
    static ostringstream oss; // a common (reset-to-empty) string formatting buffer.
    cout<<" "<<__PRETTY_FUNCTION__<<endl; cout.flush();
    int const v = 0;
    bool vl_clobbered = 0;
    Cunit tmp("tmp");
    tmp.v = 0; // not verbose
    int const s4 = 4*sstride; // stride in bytes (for asm)
    int const d4 = 4*dstride;
    CBLOCK_SCOPE(vcopy32, "", tmp, tmp.root);
    oss<<"// copy "<<n<<" elements from "<<src<<" stride "<<sstride<<" to "<<dst<<", stride "<<dstride<<"\n";
    if(v>=1) cout<<oss.str();
    vcopy32<<oss.str(); oss.str("");
    //int const max_unroll = 4; // XXX TODO
    bool ok=false;
    if(n<4/*?*/){
        for(int64_t i=0; i<n; ++i){
            CBLK(vcopy32,"((float*)(void*)"<<dst<<")["<<i*dstride<<"] = ((float*)(void*)"<<src<<")["<<i*sstride<<"];");
        }
        ok = true;
    }else if(n <= MVL){
        CBLK(vcopy32,"_ve_lvl("<<n<<"); /* single load-store is enough */");
        CBLK(vcopy32,"__vr tmp = _ve_vldu_vss("<<s4<<", (float*)(void*)"<<src<<");");
        CBLK(vcopy32,"_ve_vstu_vss(tmp, "<<d4<<", (float*)(void*)"<<dst<<");");
        vl_clobbered = true;
        ok = true;
    }else{
        // to what vlen can we go down to and still have same # ops?
        // includes easy case of n%MVL==0
        // Note: it is possible that certain values of vlen (256, 256-32 ?) are much better
        //       than others.  and even a "remainder" impl might be prefered if the final
        //       remainder is small (because this might decrease to vector-op pipeling length
        //       as we return?)   Would need measurements!
        int64_t const nFull = (n+MVL-1)/MVL;
        int64_t const minvlen = n/nFull;
        if(v>=1) cout<<" nFull="<<nFull<<" minvlen="<<minvlen;
        int64_t ivl;
        for(ivl=MVL; !ok && ivl>=minvlen; --ivl){ // largest vlen exactly subdividing n?
            if(n%ivl==0){
                if(v>=1) cout<<" n%ivl = "<<n<<"%"<<ivl;
                ok=true;
                break;
            }
        }
        if(ok){
            if(v>=1) cout<<endl;
            int64_t fulls = n/ivl;
            assert( fulls > 0 );
            assert( fulls == nFull ); // logic error?
            // XXX Hmmm, can unrolling be done with string transform?
            CBLK(vcopy32,"_ve_lvl("<<ivl<<"); /* multiple longish load stores, no remainder */");
            CBLOCK_SCOPE(vl,OSSFMT("for(int64_t i=0; i<"<<fulls<<"; ++i)"),tmp,vcopy32);
            CBLK(vl,"__vr tmp = _ve_vldu_vss("<<s4<<", ((float*)(void*)"<<src<<") + i*"<<ivl<<"*"<<sstride<<");");
            CBLK(vl,"_ve_vstu_vss(tmp, "<<d4<<", ((float*)(void*)"<<dst<<") + i*"<<ivl<<"*"<<dstride<<");");
            vl_clobbered = true;
        }
    }
    if(!ok) { // fall back to vlen 256 with remainder
        int64_t const fulls = n/MVL;
        assert( fulls >= 1 );
        int64_t const rem = n%MVL;
        assert( rem > 0 );
        CBLK(vcopy32,"_ve_lvl("<<MVL<<"); /* n="<<fulls<<"*MVL+"<<rem<<" */");
        if( fulls == 1 ){
            CBLK(vcopy32,"__vr tmp = _ve_vldu_vss("<<s4<<", (float*)(void*)"<<src<<");");
            CBLK(vcopy32,"_ve_vstu_vss(tmp, "<<d4<<", (float*)(void*)"<<dst<<");");
        }else{
            CBLOCK_SCOPE(vl,OSSFMT("for(int64_t i=0; i<"<<fulls<<"; ++i)"),tmp,vcopy32);
            CBLK(vl,"__vr tmp = _ve_vldu_vss("<<s4<<", ((float*)(void*)"<<src<<") + i*"<<MVL<<"*"<<sstride<<");");
            CBLK(vl,"_ve_vstu_vss(tmp, "<<d4<<", ((float*)(void*)"<<dst<<") + i*"<<MVL<<"*"<<dstride<<");");
        }
        auto& vend = vcopy32["end"]; // after all of the above... (introduces new subnode)
        // could recurse for the remainder...
        int64_t cnt = fulls*MVL;    // current 4-byte output position ('rem' more to finish all 'n')
        if(rem<4/*?*/){
            for(int64_t i=0; i<rem; ++i){
                CBLK(vend,"((float*)(void*)"<<dst<<")["<<(cnt+i)*dstride<<"] = ((float*)(void*)"<<src<<")["<<(cnt+i)*sstride<<"];");
            }
            ok = true;
        }else{
            CBLK(vend,"_ve_lvl("<<rem<<"); /* single load-store is enough */");
            CBLK(vend,"__vr tmp = _ve_vldu_vss("<<s4<<", (float*)(void*)"<<src<<"+"<<cnt<<"*"<<sstride<<");");
            CBLK(vend,"_ve_vstu_vss(tmp, "<<d4<<", (float*)(void*)"<<dst<<"+"<<cnt<<"*"<<dstride<<");");
            ok = true;
        }
        vl_clobbered = true;
    }
    //return vl_clobbered;
    return tmp.str();
}
#endif
std::string vel_vcopy32(std::string src, int64_t const sstride, int64_t const n,
                        std::string dst, int64_t const dstride)
{
    static ostringstream oss; // a common (reset-to-empty) string formatting buffer.
    cout<<" "<<__PRETTY_FUNCTION__<<endl; cout.flush();
    int const v = 0;
    bool vl_clobbered = 0;
    Cunit tmp("tmp");
    tmp.v = 0; // not verbose
    int const s4 = 4*sstride; // stride in bytes (for asm)
    int const d4 = 4*dstride;
    CBLOCK_SCOPE(vcopy32, "", tmp, tmp.root);
    oss<<"// copy "<<n<<" elements from "<<src<<" stride "<<sstride<<" to "<<dst<<", stride "<<dstride<<"\n";
    if(v>=1) cout<<oss.str();
    vcopy32<<oss.str(); oss.str("");
    //int const max_unroll = 4; // XXX TODO
    bool ok=false;
    if(n<4/*?*/){
        for(int64_t i=0; i<n; ++i){ // xxx
            CBLK(vcopy32,"("<<CVT<<dst<<")["<<i*dstride<<"] = (" CVT<<src<<")["<<i*sstride<<"];");
        }
        ok = true;
    }else if(n <= MVL){
        //CBLK(vcopy32,"_ve_lvl("<<n<<"); /* single load-store is enough */");
        CBLK(vcopy32,"__vr tmp = _vel_vldu_vssl("<<s4<<", " CVT<<src<<","<<asDec(n)<<");");
        CBLK(vcopy32,"_vel_vstu_vssl(tmp, "<<d4<<", " CVT<<dst<<","<<asDec(n)<<");");
        vl_clobbered = true;
        ok = true;
    }else{
        // to what vlen can we go down to and still have same # ops?
        // includes easy case of n%MVL==0
        // Note: it is possible that certain values of vlen (256, 256-32 ?) are much better
        //       than others.  and even a "remainder" impl might be prefered if the final
        //       remainder is small (because this might decrease to vector-op pipeling length
        //       as we return?)   Would need measurements!
        int64_t const nFull = (n+MVL-1)/MVL;
        int64_t const minvlen = n/nFull;
        if(v>=1) cout<<" nFull="<<nFull<<" minvlen="<<minvlen;
        int64_t ivl;
        for(ivl=MVL; !ok && ivl>=minvlen; --ivl){ // largest vlen exactly subdividing n?
            if(n%ivl==0){
                if(v>=1) cout<<" n%ivl = "<<n<<"%"<<ivl;
                ok=true;
                break;
            }
        }
        if(ok){
            if(v>=1) cout<<endl;
            int64_t fulls = n/ivl;
            assert( fulls > 0 );
            assert( fulls == nFull ); // logic error?
            // XXX Hmmm, can unrolling be done with string transform?
            //CBLK(vcopy32,"_ve_lvl("<<ivl<<"); /* multiple longish load stores, no remainder */");
            CBLOCK_SCOPE(vl,OSSFMT("for(int64_t i=0; i<"<<fulls<<"; ++i)"),tmp,vcopy32);
            CBLK(vl,"__vr tmp = _vel_vldu_vssl("<<s4<<", (" CVT<<src<<") + i*"<<ivl<<"*"<<sstride<<","<<ivl<<");");
            CBLK(vl,"_vel_vstu_vssl(tmp, "<<d4<<", (" CVT<<dst<<") + i*"<<ivl<<"*"<<dstride<<","<<ivl<<");");
            vl_clobbered = true;
        }
    }
    if(!ok) { // fall back to vlen 256 with remainder
        int64_t const fulls = n/MVL;
        assert( fulls >= 1 );
        int64_t const rem = n%MVL;
        assert( rem > 0 );
        //CBLK(vcopy32,"_ve_lvl("<<MVL<<"); /* n="<<fulls<<"*MVL+"<<rem<<" */");
        //if( fulls == 1 ){
        //    CBLK(vcopy32,"__vr tmp = _vel_vldu_vssl("<<s4<<", " CVT<<src<<","<<MVL<<");");
        //    CBLK(vcopy32,"_vel_vstu_vssl(tmp, "<<d4<<", " CVT<<dst<<","<<MVL<<");");
        //}else{
            CBLOCK_SCOPE(vl,OSSFMT("for(int64_t i=0; i<"<<fulls<<"; ++i)"),tmp,vcopy32);
            CBLK(vl,"__vr tmp = _vel_vldu_vssl("<<s4<<", (" CVT<<src<<") + i*"<<MVL<<"*"<<sstride<<","<<MVL<<");");
            CBLK(vl,"_vel_vstu_vssl(tmp, "<<d4<<", (" CVT<<dst<<") + i*"<<MVL<<"*"<<dstride<<","<<MVL<<");");
        //}
        auto& vend = vcopy32["end"]; // after all of the above... (introduces new subnode)
        // could recurse for the remainder...
        int64_t cnt = fulls*MVL;    // current 4-byte output position ('rem' more to finish all 'n')
        if(rem<4/*?*/){
            for(int64_t i=0; i<rem; ++i){
                CBLK(vend,"(" CVT<<dst<<")["<<(cnt+i)*dstride<<"] = (" CVT<<src<<")["<<(cnt+i)*sstride<<"];");
            }
            ok = true;
        }else{
            //CBLK(vend,"_ve_lvl("<<rem<<"); /* single load-store is enough */");
            CBLK(vend,"__vr tmp = _vel_vldu_vssl("<<s4<<", " CVT<<src<<"+"<<cnt<<"*"<<sstride<<","<<rem<<");");
            CBLK(vend,"_vel_vstu_vssl(tmp, "<<d4<<", " CVT<<dst<<"+"<<cnt<<"*"<<dstride<<","<<rem<<");");
            ok = true;
        }
        vl_clobbered = true;
    }
    //return vl_clobbered;
    return tmp.str();
}
// vim: ts=4 sw=4 et cindent cino=^=l0,\:.5s,=-.5s,N-s,g.5s,b1 cinkeys=0{,0},0),\:,0#,!^F,o,O,e,0=break
