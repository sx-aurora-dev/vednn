// TODO:
//    bug in cjitConvFwd1q and cjitConvFwd6 for d1s1pS_ih1oh1 case
//    (perhaps special-case the ih1iw1 and similar cases?)
// NOT DONE:
//    [VL bug?] in cjitConvFwd4 and 5 (removed them)
#include "cjitConv.hpp"
#include "dllFileAux.hpp"   // strings for declarations, paramString
#include <sstream>
#include <iomanip>
#include <unordered_set>

/** \c STR(...) eases raw text-->C-string (escaping embedded " or \\) */
#ifndef CSTR
#define CSTRING0(...) #__VA_ARGS__
#define CSTR(...) CSTRING0(__VA_ARGS__)
#endif

#ifndef MVL
#define MVL 256
#endif

using namespace std;
using namespace cprog;

int64_t ve_vlen_suggest_equ(int64_t const nitems){
    int64_t ret=MVL;
    //bool x0_check_vl = true;
    if( nitems <= MVL ){
        ret = nitems; // trivial: only once through the loop
    }else{
        int64_t const nFull = nitems/MVL;
        int64_t const nLoops = (nitems+MVL-1)/MVL;
        int64_t const rem   = nitems%MVL;
        cout<<" nFull="<<nFull<<" rem="<<rem<<" nLoops="<<nLoops<<endl;
        //if(rem+32 >= MVL){ // rem also large, latency already roughly equal.
        //    ret = MVL; } else
        if( nitems%nLoops == 0 ){
            // this is "good enough" for vector latency, but more importantly
            // avoids some special handling for last-time-through-loop.
            ret = nitems/nLoops;
            cout<<"loop 0.."<<nitems<<" vlen perfect division as "<<ret<<"*"<<nitems/ret<<endl;
            //x0_check_vl = false;
        }else{ // redistribute...
            // we need some remainder, set up a "nice" main vector length
            // DO NOT use rem as loop entry value, only as last-time-through value.
            // [ vector inductions REQUIRE last-loop vlen <= main vlen ]
            int64_t vleq = (nitems+nLoops-1) / nLoops;
            //
            // Note: above ensures that remainder (vector length for last pass)
            // is always less than vleq.
            // For example, N=257, MVL=256 should not use 128+129.
            // Instead, we propose 160+97
            //
            // Why? you will usually set up vector loop induction for the initial
            // vector length, which may then be inalid if vector length ever
            // increases (but stays OK if you shorten vector length).
            //
            // I.e. you can choose between
            // A. ignoring correct unused values, and
            // B. using incorrect [uncalculated?] extra values
            // during the last loop pass.  (A.) is much better.
            //
            cout<<" vleq="<<vleq<<" = "<<nitems<<"/"<<nLoops<<" = nitems/nLoops"<<endl;
            assert( vleq <= MVL );
            if( nitems%vleq != 0 ){
                //vleq = (vleq+31)/32*32;
                //cout<<"vleq rounded up to "<<vleq<<endl;

                ret = vleq;
                cout<<"loop 0.."<<nitems<<" vlen redistributed from "<<MVL<<"*"<<nFull<<"+"<<rem
                    <<" to "<<vleq<<"*"<<nitems/vleq<<"+"<<nitems%vleq<<endl;
                // Paranoia: if we somehow increased loop count, this logic has a bug
                assert( (nitems+vleq-1)/vleq == nLoops );
            }
        }
    }
    return ret;
}

int64_t ve_vlen_suggest(int64_t const nitems){
    int64_t ret=MVL;
    //bool x0_check_vl = true;
    if( nitems <= MVL ){
        ret = nitems; // trivial: only once through the loop
    }else{
        int64_t const nFull = nitems/MVL;
        int64_t const nLoops = (nitems+MVL-1)/MVL;
        int64_t const rem   = nitems%MVL;
        cout<<" nFull="<<nFull<<" rem="<<rem<<" nLoops="<<nLoops<<endl;
        if(rem+32 >= MVL){ // rem also large, latency already roughly equal.
            ret = MVL;
        }else if( nitems%nLoops == 0 ){
            // this is "good enough" for vector latency, but more importantly
            // avoids some special handling for last-time-through-loop.
            ret = nitems/nLoops;
            cout<<"loop 0.."<<nitems<<" vlen perfect division as "<<ret<<"*"<<nitems/ret<<endl;
            //x0_check_vl = false;
        }else{ // redistribute...
            // we need some remainder, set up a "nice" main vector length
            // DO NOT use rem as loop entry value, only as last-time-through value.
            // [ vector inductions REQUIRE last-loop vlen <= main vlen ]
            int64_t vleq = (nitems+nLoops-1) / nLoops;
            //
            // Note: above ensures that remainder (vector length for last pass)
            // is always less than vleq.
            // For example, N=257, MVL=256 should not use 128+129.
            // Instead, we propose 160+97
            //
            // Why? you will usually set up vector loop induction for the initial
            // vector length, which may then be inalid if vector length ever
            // increases (but stays OK if you shorten vector length).
            //
            // I.e. you can choose between
            // A. ignoring correct unused values, and
            // B. using incorrect [uncalculated?] extra values
            // during the last loop pass.  (A.) is much better.
            //
            cout<<" vleq="<<vleq<<" = "<<nitems<<"/"<<nLoops<<" = nitems/nLoops"<<endl;
            assert( vleq <= MVL );
            if( nitems%vleq == 0 ){
            }else{
                // guess latency increments for VE as vector length passes multiples of 32
                // so round "main" vector up to a multiple of 32 (take whatever remains as remainder)
                vleq = (vleq+31)/32*32;
                cout<<"vleq rounded up to "<<vleq<<endl;

                ret = vleq;
                cout<<"loop 0.."<<nitems<<" vlen redistributed from "<<MVL<<"*"<<nFull<<"+"<<rem
                    <<" to "<<vleq<<"*"<<nitems/vleq<<"+"<<nitems%vleq<<endl;
                // Paranoia: if we somehow increased loop count, this logic has a bug
                assert( (nitems+vleq-1)/vleq == nLoops );
            }
        }
    }
    return ret;
}

#if defined(ALLOW_VE_INTRINSICS)
Cblock& ve_pvfadd(Cblock& cb, std::vector<std::string> const& regs){
    // deprecated
    if(!regs.empty()){
        std::ostringstream oss;
        unsigned d=1, d2;
        cb>>"// ve_pvfadd(cb,regNames["<<asDec(regs.size())<<"]) --> "<<regs[0];
        // binary-tree like additions.  Final result in regs.front()
        for( ; d<regs.size(); d=d2 ){
            d2 = d<<1;
            for(unsigned i=0; i<regs.size(); i+=d2 ){
                if(i+d<regs.size()){
                    oss<<regs[i]<<" = _ve_pvfadd_vvv("<<regs[i]<<", "<<regs[i+d]<<");\n";
                }
            }
        }
        cb>>oss.str();
    }
    return cb;
};// end multiple pvfadd
#endif
Cblock& ve_pvfadd(Cblock& cb, std::vector<std::string> const& regs, std::string vl){
    if(!regs.empty()){
        std::ostringstream oss;
        unsigned d=1, d2;
        cb>>"// ve_pvfadd(cb,regNames["<<asDec(regs.size())<<"],"<<vl<<") --> "<<regs[0];
        // binary-tree like additions.  Final result in regs.front()
        for( ; d<regs.size(); d=d2 ){
            d2 = d<<1;
            for(unsigned i=0; i<regs.size(); i+=d2 ){
                if(i+d<regs.size()){
                    oss<<regs[i]<<" = _vel_pvfadd_vvvl("<<regs[i]<<", "<<regs[i+d]<<", "<<vl<<");\n";
                }
            }
        }
        cb>>oss.str();
    }
    return cb;
};// end multiple pvfadd
Cblock& ve_pvfadd(Cblock& cb, std::vector<std::string> const& regs, int const vl){
    return ve_pvfadd(cb,regs,asDec(vl));
}

void vrj_init(Cblock &vec_init){
    vec_init
#if VRJ_INDUCE>1
        //>>"const int64_t sw_x_VLEN = strideWidth * VLEN;"
        >>"const int64_t sw_x_VLEN = strideWidth * vl_x_init;"
#endif
#if VRJ_INDUCE>0
        //>>"__vr const vrj_init = _ve_vaddsl_vsv(-padWidth,  _ve_vmulsl_vsv(strideWidth, vrseq));"
        >>"__vr const vrj_init = _vel_vaddsl_vsvl(-padWidth,  "
        " _vel_vmulsl_vsvl(strideWidth, vrseq, vl_x_init), vl_x_init);"
#endif
        ;
}

std::string ve_pragma_unroll(int64_t const N){
    std::string ret(""); // return empty if N<0
    if(N==0) return "#pragma nounroll\n";
    else if(N>0){
        std::ostringstream oss;
        ret = OSSFMT("#pragma unroll("<<N<<")\n");
    }
    return ret;
}

void vrj_induce(Cblock &loop_x0){
#if VRJ_INDUCE==0
    // vrj[i] = (x0+vrseq[i])*strideWidth - padWidth
    //        = x0*strideWidth + vrseq[i]*strideWidth - padWidth
    //                           -------------------------------
    //        = x0*strideWidth +            vrj_init        // VRJ_INDUCE==1
    //
    // but x0 for N'th loop_x0 begins at N*VLEN (or vl_x_init)
    // vrj_N = N*VLEN*strideWidth     + vrj_init
    //       = vrj_{N-1} + VLEN*strideWidth                 // VRJ_INDUCE==2
    //                   ^^^^^^^^^^^^^^^^^^
    //        i.e. vrj = _ve_vaddsl_vsv(vl*strideWidth,vrj)
    //
    // VRJ_INDUCE==2 should be fastest (adding a constant, save a multiply
    //
    //loop_x0>>"__vr const vrj = _ve_vaddsl_vsv(-padWidth, _ve_vmulsl_vsv(strideWidth, _ve_vaddsl_vsv(x0, vrseq)));";
    loop_x0>>"__vr const vrj = _vel_vaddsl_vsvl(-padWidth, _vel_vmulsl_vsv(strideWidth, _vel_vaddsl_vsvl(x0, vrseq, vl), vl), vl);";

#elif VRJ_INDUCE==1 // simplified full calc (using vrj_init) (WORKING)
    //loop_x0>>"__vr const vrj = _ve_vaddsl_vsv(x0*strideWidth, vrj_init);";
    loop_x0>>"__vr const vrj = _vel_vaddsl_vsvl(x0*strideWidth, vrj_init, vl);";

#else // VRJ_INDUCE==2 What is the problem with this induction?
    // IN PRINCIPLE:
    //    loop_x0["last"]>>"vrj = _ve_vaddsl_vsv(sw_x_VLEN,vrj);";
    //        should work!
    // sometimes clang compiles it wrong, sometimes clang unrolls it wrong
    // (and also causes issues with all subsequent runs!)
    //loop_x0[".."]>>"__vr vrj = vrj_init;";
    loop_x0[".."]>>"__vr vrj = _vel_vor_vsvl(0,vrj_init,vl);";
    //loop_x0["last"]>>"vrj = _ve_vaddsl_vsv(sw_x_VLEN,vrj);";
    loop_x0["last"]>>"vrj = _vel_vaddsl_vsvl(sw_x_VLEN,vrj,vl);";
    //loop_x0["last"]>>"vrj = _ve_vaddsl_vsv(vl*strideWidth,vrj);";
#if 0
    loop_x0>>"__vr const vrj = vrjNext;";
    loop_x0["last"]
        >>"asm(\"### VRJ_INDUCE==2\");"
        >>"vrjNext = _ve_vaddsl_vsv(sw_x_VLEN,vrj); // induce to avoid full recalc"
        >>"asm(\"### VRJ_INDUCE==2\");"
        ;
#endif
#endif
}

/** \group C API */
//@{
#ifdef __cplusplus
extern "C" { //}
#endif

typedef DllFile (*DllGenerator)(struct param const* const p);

char const* const JitConvsOpt::default_jit_dir
#if defined(__ve)
= "tmp_cjitConv"
#else
= "tmp_cjitConv-x86" // x86 build TBD
#endif
;

/** known names for DllFileGenerator functions */
struct DfgNames {
    char const* str;
    DllGenerator func;
};

static DfgNames const dfgNames[] = {
    {"cjitConvFwd1q", cjitConvolutionForward1q} // STANDARD ITEM
    ,{"cjitConvFwd6", cjitConvolutionForward6} // STANDARD ITEM
    // optional ones: see also 'generators[]' in jitconv.cpp, and Makefile
    ,{"cjitConvFwd1", cjitConvolutionForward1}
    ,{"cjitConvFwd1b", cjitConvolutionForward1b}
    ,{"cjitConvFwd1p", cjitConvolutionForward1p}
    ,{"cjitConvFwd2", cjitConvolutionForward2}
    ,{"cjitConvFwd3", cjitConvolutionForward3}
    ,{"cjitConvFwd4", cjitConvolutionForward4}
    ,{"cjitConvFwd5", cjitConvolutionForward5}
};
static size_t const nDfgNames = sizeof(dfgNames)/sizeof(DfgNames);

static DllGenerator getDllFileGenerator(char const* const cstr)
{
    DllGenerator func = nullptr;
    for(size_t i=0; i<nDfgNames; ++i){
        DfgNames const& dn = dfgNames[i];
        int const safelen = strlen(dn.str);
        // oh, to do exact match, just use safelen+1
        if(strncmp(cstr, dn.str, safelen+1) == 0){
            func = dn.func;
            break;
        }
    }
    if( func==nullptr ) cout<<"\nWarning: unrecognized JIT impl "<<cstr<<endl;
    return func;
}

struct CjitOpaque {
    unique_ptr<DllOpen> uPtr;
    void* symMem;
};

#if 0
void cjitsyms_add( CjitSyms *cjs, CjitSyms const* other )
{
}
#endif
CjitSyms  * cjitSyms( struct param const* const pParams,
        int const nParams,
        char const** const dllGeneratorNames,
        struct CjitOpt const* opt_a )
{
    int const v=0; // verbose: 0,1,2
    assert( dllGeneratorNames != nullptr );
    assert( dllGeneratorNames[0] != nullptr );
    assert( pParams != nullptr );
    assert( nParams > 0 );
    vector<DllGenerator> dllGenerators;
    for( char const** gn=dllGeneratorNames; *gn!=NULL; ++gn ){
        printf(" cjitSyms generator \"%s\"", *gn); fflush(stdout);
        DllGenerator dllgen = getDllFileGenerator(*gn);
        if( dllgen == nullptr )
            THROW("No generator named \""<<*gn<<"\"?");
        //assert( dllgen != nullptr );
        dllGenerators.push_back( dllgen );
        printf(" @ %p\n", (void*)dllgen); fflush(stdout);
    }
    CjitSym * cjs = nullptr;
    CjitSyms* ret = new CjitSyms{new CjitOpaque, cjs, 0U};
    CjitOpaque* opaque = reinterpret_cast<CjitOpaque*>(ret->opaque);
    opaque->uPtr = jitConvs(pParams,nParams,
            dllGenerators,
            JitConvsOpt(opt_a));
    if(!(opaque->uPtr)){
        THROW(" No object in unique_ptr<DllOpen> for jitConvs");
    }
    {
        // copy from uPtr into a plain 'C' array of C Jit {sym,ptr}
        if(v>0){
            cout<<"C JIT symbols:\n"
                <<left<<setw(80)<<"Symbol"<<" "<<"Address\n"
                <<string(80,'-')<<" "<<string(18,'-')<<"\n";
            cout.flush();
        }
        // allocate list memory: string memory points into opaque const C++ DllOpen
        auto const symMap = opaque->uPtr->getDlsyms();
        size_t const nSyms = symMap.size();
        cjs = (CjitSym*)malloc((nSyms+1U)*sizeof(CjitSym));
        if(v>1){cout<<" allocated cjs["<<nSyms<<"+1] list @ "<<(void*)cjs<<endl; cout.flush();}
        size_t nSymBytes = 0U;
        for(auto const& x: symMap) nSymBytes += x.first.size() + 1U;
        char* symMem = new char[nSymBytes];
        if(v>1){cout<<" allocated "<<nSymBytes<<" of string memory"<<endl; cout.flush();}
        // iterate , setting symbol name pointer and symbol address
        char *psym = symMem;
        CjitSym * pcjs = &cjs[0];
        //for(auto const& x: symMap) // NEW: retain original "test" (source file) number...
        size_t const nSrc = opaque->uPtr->nSrc();       // src files
        auto const symSrc = opaque->uPtr->getDlsrcs();  // syms per file
        for(size_t f=0U; f<nSrc; ++f)                   // src file#
        {
            int tag = opaque->uPtr->srcTag(f);          // src file tag = which parameter set (test number)
            // keeping own copy of psym info perhaps not necessary
            for(auto const& str: symSrc[f]){    // for each symbol in source code file f
                memcpy( psym, str.c_str(), str.size()+1U );
                assert( symMap.find(psym) != symMap.end() );
                // Set all fields of the CjitSym:
                pcjs->sym = psym;
                pcjs->ptr = (*opaque->uPtr)[psym];
                pcjs->tag = tag;
                if(v>0){cout<<setw(80)<<pcjs->sym<<" "<<pcjs->ptr<<" tag "<<pcjs->tag<<"\n"; cout.flush();}
                if(v>1){cout<<" pcjs="<<(void*)pcjs<<" symbol name @ "<<(void*)pcjs->sym<<endl; cout.flush();}
                //
                ++pcjs;
                psym += str.size() + 1U;
            }
        }
        assert( psym - symMem == nSymBytes );
        // array terminator (deprecate this? CjitSyms::len is there now)
        pcjs->sym = nullptr;
        pcjs->ptr = nullptr;
        pcjs->tag = 0U; // (this value is likely valid)
        // remember some useful things
        opaque->symMem = symMem;
        ret->len = nSyms;
        //ret->nSrc = nSrc;

        // don't sort the last item.
        std::sort( cjs, cjs+nSyms, [](CjitSym const& a, CjitSym const& b){
                return strcmp(a.sym, b.sym) < 0; });
    }
    // DO NOT free the C++ object memory before exit - it will dlclose the jit dll
    ret->syms   = cjs;
    cout<<" returning new Cjitsyms @ "<<(void*)ret<<endl;
    cout.flush();
    return ret;
}
void cjitSyms_free( CjitSyms const* const cjitsyms ){
    if(cjitsyms==nullptr){
        return;
    }
    int const v=0;
    if(v){cout<<" cjitSyms_free( cjitsyms @ "<<(void*)cjitsyms<<" )"<<endl; cout.flush();}
    if(cjitsyms->opaque){ // help gaurd against double-free
        // now free the C++ opaque info
        // side-effect: destroying the DllOpen will call dlclose.
        // This might invalidate pointers, because the dll might be released from memory (?)
        CjitOpaque* opaque = reinterpret_cast<CjitOpaque*>(cjitsyms->opaque);
#ifndef NDEBUG
        const_cast<CjitSyms*>(cjitsyms)->opaque = nullptr;
#endif
        if(v){cout<<" resetting unique Ptr "<<(void*)opaque->uPtr.get()<<endl; cout.flush();}
        opaque->uPtr.reset();
        if(v){cout<<" probably dlclose was called"<<endl; cout.flush();}

        if(v){cout<<" free string memory @ "<<(void*)opaque->symMem<<endl; cout.flush();}
        free(opaque->symMem);

        { // free the list memory
            CjitSym* cjs = cjitsyms->syms;
#ifndef NDEBUG
            const_cast<CjitSyms*>(cjitsyms)->syms = nullptr;
            const_cast<CjitSyms*>(cjitsyms)->len = 0U;
#endif
            if(v){cout<<" free(cjitsyms->syms cjs @ "<<(void*)cjs<<endl; cout.flush();}
            free((void*)cjs);
        }
        if(v){cout<<" finally delete CjitSyms @ "<<(void*)cjitsyms<<endl; cout.flush();}
        delete cjitsyms;
    }
}





#if 0 // I realize I don't really need a fancier iterator API since I can discard the C++ objects immediately
struct CxxjitHandle {
    unique_ptr<DllOpen> uPtr;
    CxxjitHandle( unique_ptr<DllOpen> uPtr ) : uPtr(uPtr) {}
    CxxjitHandle( CxxjitHandle const& src ) = delete;
    CxxjitHandle& operator=( CxxjitHandle const& src ) = delete;
}
/** casting the opaque CjitHandle */
static inline unique_ptr<DllOpen>* obj( CjitHandle const h ){
    return reinterpret_cast<unique_ptr<DllOpen>*>(
            const_cast<void*>( h.ptr ));
}
/** casting the opaque CjitHandle */
static inline unique_ptr<DllOpen>* const cobj( CjitHandle const h ){
    return reinterpret_cast<unique_ptr<DllOpen> const*>(
            const_cast<void const*>( h.ptr ));
}

CjitHandle cjitConvs( struct param const* const pParams,
        int const nParams,
        char const* dllGeneratorName )
{
    assert( dllGeneratorName != nullptr );
    assert( pParams != nullptr );
    CjitHandle ret;
    ret.uPtr = jitConvs(pParams,nParams,dllGeneratorName);
    return ret;
}

void cjitConvs_free( CjitHandle* cjithandle ){
    obj(cjithandle).reset();  // free the unique_ptr.  All is forgotten
    cjitHandle->ptr = nullptr;
}

struct CjitIter_t {
    CxxjitHandle

/** casting the opaque CjitIter */
static inline CxxjitIter* obj( CjitIter const h ){
    return reinterpret_cast<unique_ptr<DllOpen>>(
            const_cast<void*>( h.ptr ));
}
/** casting the opaque CjitIter */
static inline unique_ptr<DllOpen>& const cobj( CjitIter const h ){
    return reinterpret_cast<unique_ptr<DllOpen> const>(
            const_cast<void const*>( h.ptr ));
}
CjitIter cjit_iter( CjitHandle* cjitHandle ){ 
}
void cjit_free( CjitIter* cjitIter ){
}


int cjit_iter_empty( CjitIter const cjitIter ){
}

CjitSym cjit_at( CjitIter cjitIter ){
}

int cjit_next( CjitIter cjitIter ){
}

inline CjitSym cjit_end() {
    CjitSym ret = {NULL,NULL};
    return ret;
}
#endif

#ifdef __cplusplus
/* { */
} //extern "C"
#endif
//@}

#if 0
/** the vednn \b impl call (not the public vednn API one) */
static char const* decl_ConvolutionForward =                    
"vednnError_t (*vednnConvolutionForward)("
"\n    const vednnTensorParam_t      * restrict pParamIn,"
"\n    const void                    * restrict  pDataIn,"
"\n    const vednnFilterParam_t      * restrict pParamKernel,"
"\n    const void                    * restrict  pDataKernel,"
"\n    const vednnConvolutionParam_t * restrict pParamConv,"
"\n    const vednnTensorParam_t      * restrict pParamOut,"
"\n    void                          * restrict  pDataOut"
"\n)";
#endif

// then emit them as constant cjit values (or #define them)
// for C++, T const might be better (T const "as-good-as-macro", and properly typed)
//#define CONST1(var) <<("\nint64_t const " #var " = "+asDec(var))
// but for 'C', #define may hold less surprises
#define CONST1(var) >>("#define " #var " "+asDec(var))
#define FREE1(var) >>("#undef " #var)

/** ease-of-use -- look into how to make it non-verbose, now that
 * things are finally working with the glibc ncc2+ */
unique_ptr<DllOpen> jitConvs(struct param const* const pParams,
        int const nParams,
        DllFile (*dllFileGenerator)(struct param const* const p),
        JitConvsOpt const opt /*=JitConvsOpt()*/
        )
{
    std::vector<DllFile (*)(struct param const* const p)> dllFileGenerators
        = {dllFileGenerator};
    assert( dllFileGenerators.size() == 1 );
    return jitConvs(pParams, nParams, dllFileGenerators);
}
std::unique_ptr<DllOpen> jitConvs(struct param const* const pParams,
        int const nParams,
        char const** generators, // null terminated array of C-string tags for dfgNames[] lookup
        JitConvsOpt const opt /*=JitConvsOpt()*/
        )
{
    std::vector<DllFile (*)(struct param const* const p)> dllFileGenerators;
    for(char const** g = &generators[0]; *g!=nullptr; ++g){
        DllGenerator dg = getDllFileGenerator( *g );
        if( dg != nullptr ) dllFileGenerators.push_back(dg);
    }
    if( dllFileGenerators.empty() ) THROW("No recognized JIT generators");
    return jitConvs(pParams, nParams, dllFileGenerators);
}
std::unique_ptr<DllOpen> jitConvs(struct param const* const pParams,
        int const nParams,
        std::vector<DllFile (*)(struct param const* const p)> const& dllFileGenerators,
        JitConvsOpt const opt /*=JitConvsOpt()*/
        )
{
    cout<<" jitConvs from "<<dllFileGenerators.size()<<" Jit generators..."<<endl; cout.flush();
    { // check arguments
        assert( nParams > 0 );
        assert( !dllFileGenerators.empty() );
        // check for no dups
        unordered_set<void*> ptrs;
        for(auto g: dllFileGenerators){
            void* v = (void*)g;
            if(v == nullptr) THROW(" bad generator function (NULL)!");
            ptrs.insert((void*)v);
        }
        if( ptrs.size() != dllFileGenerators.size() ){
            for(size_t i=0U; i<dllFileGenerators.size(); ++i){
                cout<<" dllFileGenerators["<<i<<"] = "<<(void*)dllFileGenerators[i]<<endl;
            }
            cout.flush();
            cout<<"Error: duplicate pointer!"<<std::endl; std::cout.flush();
            THROW("duplicate JIT generators might cause duplicate symbols");
        }
    }
    int nerr = 0;
    DllBuild dllbuild;
    for(auto dllFileGenerator: dllFileGenerators){
        assert( dllFileGenerator != nullptr );
        for(int tag=0; tag<nParams; ++tag){
            struct param const& pConv = pParams[tag];
            DllFile df = dllFileGenerator(&pConv);
            df.tag = tag;
            dllbuild.push_back(df);
        }
    }

    if(opt.skip_prep){
        dllbuild.skip_prep("cjitConv",opt.jit_dir);
    }else{
        dllbuild.prep("cjitConv",opt.jit_dir);
    }

    std::string mkEnv;
    if(1){
        // mucking about too much here can actually harm things.
        std::ostringstream oss;
        oss<<"CFLAGS='"
            "  -I\"" CSTR(VEDNNX_DIR) "/include\""
#if defined(_OPENMP)
            " -fopenmp" // also some link libraries may need _omp or so suffix
#endif
            "'"
            " CXXFLAGS='"
            "  -I\"" CSTR(VEDNNX_DIR) "/include\""
#if defined(_OPENMP)
            " -fopenmp" // also some link libraries may need _omp or so suffix
#endif
            "'"
            " LDFLAGS='"
            "  -L\"" CSTR(VEDNNX_DIR) "/lib\""
            " -Wl,-rpath-link,\"" CSTR(VEDNNX_DIR) "/lib\""
            " -Wl,-rpath-link,\"" CSTR(VEJIT_DIR) "/lib\""
#if defined(_OPENMP)
            //" -lvednnx_openmp"
#else
            //" -lvednnx_sequential"
#endif
            //" -ldl"
            //"  -L\"" CSTR(VEJIT_DIR)  "/lib\"" " -Wl,-rpath,\"" CSTR(VEJIT_DIR)  "/lib\""
            //" " CSTR(VEJIT_DIR) "/lib/libjit1.a"
            //" " CSTR(VEDNNX_DIR) "/lib/libvednn_sequential.a"
            //" -ldl"
            "'"
            " LIBFLAGS=''" // empty, to trun off default very-verbose linking debug
            ;
        mkEnv=oss.str();
    }
    cout<<" mkEnv:\n\t"<<mkEnv<<endl;
    cout.flush();

    if(opt.skip_make){
        dllbuild.skip_make(mkEnv);
    }else{
        dllbuild.make(mkEnv);
    }

    if(1){ // possibly useful debug.
        cout<<"\nDllBuild expects symbols...\n";
        cout<<"Library: "<<dllbuild.getLibName()<<endl;
        for(auto const& df: dllbuild){
            cout<<"   File: "<<df.basename<<df.suffix<<" :"<<endl;
            for(auto const& sym: df.syms){
                cout<<"      "<<sym.symbol<<endl;
                if(!sym.comment.empty()){prefix_lines(cout, sym.comment, "        // ");cout<<endl;}
                if(!sym.fwddecl.empty()){prefix_lines(cout, sym.fwddecl, "        ");cout<<endl;}
            }
        }
    }
    unique_ptr<DllOpen> pLib;
    {
#if !defined(__ve)
        cout<<" STOPPING EARLY: we are an x86 executable and should not load a VE .so"<<endl;
#else
        try{
            pLib = dllbuild.create(); // it is a unique_ptr<DllOpen>
        }catch(std::exception const& e){
            cout<<" dllbuild.create error: "<<e.what()<<endl;
            ++nerr;
        }catch(...){
            cout<<" dllbuild.create unknown error"<<endl;
            ++nerr;
        }
#endif
    }
    if(nerr){
        cerr<<" Encountered "<<nerr<<" errors during C-file and dll file creation."<<endl;
        THROW(" "<<nerr<<" errors in jitConvs");
    }
    if(pLib){
        cout<<" pLib->nSrc() = "<<pLib->nSrc()<<endl;
        auto const& dlsrcs = pLib->getDlsrcs();
        for(size_t i=0U; i<dlsrcs.size(); ++i){
            cout<<" test #"<<pLib->srcTag(i)<<" file "<<pLib->srcFile(i)<<endl;
            for(auto const& sym: dlsrcs[i]){
                cout<<"    sym "<<sym<<endl;
            }
        }
    }
    return pLib;
}
// vim: ts=4 sw=4 et cindent cino=^=l0,N-s,\:.5s,=-.5s,g.5s,b1 cinkeys=0{,0},0),\:,0#,!^F,o,O,e,0=break
