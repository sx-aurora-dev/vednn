/** \file
 * precalc section + fast alg + compile-time knobs.
 * Cleaned up in \ref cjitConvFwd6.cpp
 */
#include "cjitConv.hpp"
#include "ve_cvecops.hpp"
#include "dllFileAux.hpp"   // strings for declarations, paramString
#include <string>
#include <iostream>
#include <iomanip>
using namespace std;
using namespace cprog;

// icg 64 * ocg 8 = 512 *3*3 ~ 5k *flt4 = 20k
// Setting icg,ocg:
//   - icg=256, ocg=8 --> if < LIM, accept
//   - diminish icg halving towrds 8 until < LIM
//   - else diminish ocg..  then icg...
// OR LIM/4/kernHW --> sqrt --> icg=ocg= power of 2 less than this
// (quck and dirty)
//
#define JIT_MALLOC_THRESHOLD_K 16

#ifndef MVL
#define MVL 256
#endif

#define CONST1(var) >>("#define " #var " "+asDec(var))
#define FREE1(var) >>("#undef " #var)
#define DEF(VAR) def(#VAR, VAR)

#ifndef VEL_BUG
/** 1 means use extra ve_lvl as workaround for clang bug */
#define VEL_BUG 0
#endif

#ifndef KBYMAX
/** kByMax is chosen from {1,2,4,8} */
#define KBYMAX 8
#endif

#define PRE_KH_BEG_END 1
#define PRE_KRN_MAX 8

#if PRE_KH_BEG_END
/** 1: do we use a static array kh_be? 2: do we elide some things? 3: elide more? */
#define KH_BE 2
#if KH_BE // static array
//#define DEFINE_KH_BEG_END "uint8_t const * restrict kh_beg_end = &kh_be[0];"
#define DEFINE_KH_BEG_END ""
#else     // cpre calc
#define DEFINE_KH_BEG_END "uint8_t const * restrict kh_beg_end = (uint8_t const*)( buffer + kh_beg_end_off );"
#endif

#else // NO precalc, kh_{beg,end} explicit calc
#define KH_BE 0
#define DEFINE_KH_BEG_END ""
#endif

typedef long long unsigned llu;

#if KH_BE
/** create a stack-based static arrays of kh_beg[[], kh_end[] data.
 * This is a compile-time array, shared by all threads. */
static string str_kh_be_static_array( struct param const* const p )
{
    cout<<" "<<__PRETTY_FUNCTION__<<endl; cout.flush();
    const int64_t outHeight      = p->outHeight;
    const int64_t strideHeight   = p->strideHeight;
    const int64_t padHeight      = p->padHeight;
    const int64_t dilationHeight = p->dilationHeight;
    const int64_t inHeight       = p->inHeight;
    const int64_t kernHeight     = p->kernHeight;
    std::ostringstream ossb, osse;
    // XXX Paranoia: int_type_for(kernHeight)  (uint8_t probably OK always, though)
    ossb<<"static uint8_t const kh_btab[outHeight*sizeof(uint8_t)] = {";
    osse<<"static uint8_t const kh_etab[outHeight*sizeof(uint8_t)] = {";
    for(int64_t y=0; y<outHeight; ++y){ // loop_y
        const int64_t i = y * strideHeight - padHeight;
        int64_t kh_end=0;
        const int64_t kh_tmp = dilationHeight-i-1;
        const int64_t kh_beg = (i>=0? 0: kh_tmp / dilationHeight);
        if (i < inHeight){
            kh_end = (inHeight + kh_tmp) / dilationHeight;
            if (kh_end >= kernHeight ) kh_end = kernHeight ;
        }
        if(y%8==0) {ossb<<"\n        "; osse<<"\n        ";}
        ossb<<(y==0?"  ":", ")<<setw(3)<<kh_beg;
        osse<<(y==0?"  ":", ")<<setw(3)<<kh_end;
    }
    ossb<<"\n        };\n";
    osse<<"\n        };";
    return ossb.str()+osse.str();
}
#endif
static void kernLims( const int64_t outSz, const int64_t strideSz, const int64_t padSz,
        const int64_t dilationSz, const int64_t inSz, const int64_t kernSz,
        std::vector<int64_t>& kbeg, std::vector<int64_t>& kend)
{
    int const verbose=1;
    kbeg.clear();
    kend.clear();
    kbeg.reserve(outSz);
    kend.reserve(outSz);
    for(int64_t out=0; out<outSz; ++out){ // loop_y
        const int64_t in = out * strideSz - padSz;
        int64_t k_end=0;
        const int64_t k_tmp = dilationSz-in-1;
        const int64_t k_beg = (in>=0? 0: k_tmp / dilationSz);
        if (in < inSz){
            k_end = (inSz + k_tmp) / dilationSz;
            if (k_end >= kernSz ) k_end = kernSz ;
        }
        kbeg.push_back( k_beg );
        kend.push_back( k_end );
    }
    if(verbose){
        cout<<" kernLims:"<<endl;
        cout<<"     kbeg["<<kbeg.size()<<"]: ";for(auto i:kbeg)cout<<" "<<i; cout<<endl;
        cout<<"     kend["<<kend.size()<<"]: ";for(auto i:kend)cout<<" "<<i; cout<<endl;
    }
}
struct KernLims { std::vector<int64_t> kh_b, kh_e, kw_b, kw_e; };

static KernLims kernLims( struct param const* const p )
{
    KernLims ret;
    kernLims( p->outHeight, p->strideHeight, p->padHeight,
            p->dilationHeight, p->inHeight, p->kernHeight,
            ret.kh_b, ret.kh_e );
    kernLims( p->outWidth, p->strideWidth, p->padWidth,
            p->dilationWidth, p->inWidth, p->kernWidth,
            ret.kw_b, ret.kw_e );
    return ret;
}
/** Find y range [yBeg,yEnd) where all kernel height values are ok (input height within range).
 * If no such range, return yBeg >= yEnd.
 *  Note: simpler loops possible, but ncc vectorization errors for
 *  condition inside loop. */
static void nomask_Height( KernLims const& kl, int64_t kernHeight, int64_t &yBeg, int64_t &yEnd ){
    assert( kl.kh_b.size() == kl.kh_e.size() ); // == outHeight
    assert( kl.kh_b.size() > 0 );
    size_t const odim = kl.kh_b.size();
    yBeg = -1;
    for(size_t y=0; y<odim; ++y){
        if( kl.kh_b[y]      ==          0 ){
            yBeg = y; // first normal value
            cout<<" yBeg="<<yBeg;
            break;
        }
    }
    if(yBeg < 0) yBeg = yEnd = kl.kh_b.size();
    else{
        yEnd = -1;
        for(size_t y=yBeg; y<odim; ++y){
            if( kl.kh_e[y] != kernHeight ){
                yEnd = y; // at yEnd, khEnd value begins dropping
                cout<<" yEnd="<<yEnd;
                break;
            }
        }
        if(yEnd < 0) yEnd = odim;
    }
}
static void nomask_Width( KernLims const& kl, int64_t kernWidth, int64_t &xBeg, int64_t &xEnd ){
    assert( kl.kw_b.size() == kl.kw_e.size() ); // == outHeight
    assert( kl.kw_b.size() > 0 );
    size_t const odim = kl.kw_b.size();
    xBeg = -1;
    for(size_t x=0; x<odim; ++x){
        if( kl.kw_b[x]      ==          0 ){
            xBeg = x; // first normal value
            cout<<" xBeg="<<xBeg;
            break;
        }
    }
    if(xBeg < 0) xBeg = xEnd = kl.kw_b.size();
    else{
        xEnd = -1;
        for(size_t x=xBeg; x<odim; ++x){
            if( kl.kw_e[x] != kernWidth ){
                xEnd = x; // at xEnd, kwEnd value begins dropping
                cout<<" xEnd="<<xEnd;
                break;
            }
        }
        if(xEnd < 0) xEnd = odim;
    }
}

union PairedFloat {
    float f[2];                     // packed-pair floats
    uint64_t pair;                  // forces alignment 8
};

struct PreConvFwd1q {
    std::vector<PairedFloat> krn_gkrsc;     ///< krn contiguous access for kBy = 2,4,8
    size_t kstarts[3];                      ///< offsets in krn_gkrsc for kBy = 2,4,8 data
    std::vector<float> krn_gkrsc_k1;        ///< krn access pattern for kBy = 1
    std::vector<int8_t> kh_beg_end;         ///< precalculated values as fn of y in [0,outHeight)
};

std::ostream& operator<<(std::ostream& os, PreConvFwd1q const& pre){
    cout<<pre.krn_gkrsc.size()<<" PairedFloat kernel values."
        <<"  kstarts={"<<pre.kstarts[0]<<","<<pre.kstarts[1]<<","<<pre.kstarts[2]<<"} for kBy=2,4,8.\n"
        <<"  kBy=1 has krn_gkrsc_k1 size "<<pre.krn_gkrsc_k1.size()<<" floats.\n"
        <<"  kh_beg_end size "<<pre.kh_beg_end.size()
        <<endl;
    return os;
}

struct PreConvFwd1q mkPreConvFwd1q( struct param const* const p,
        float const* const pKernel = nullptr ){
    cout<<" "<<__PRETTY_FUNCTION__<<endl; cout.flush();
    assert( p != nullptr );
    int const kByMax = 8;

    //const int64_t batch          = p->batchNum;
    const int64_t group          = p->group;
    const int64_t inChannel      = p->inChannel;
    const int64_t inHeight       = p->inHeight;
    //const int64_t inWidth        = p->inWidth;
    const int64_t outChannel     = p->outChannel;
    const int64_t outHeight      = p->outHeight;
    //const int64_t outWidth       = p->outWidth;
    const int64_t kernHeight     = p->kernHeight;
    const int64_t kernWidth      = p->kernWidth;
    const int64_t strideHeight   = p->strideHeight;
    //const int64_t strideWidth    = p->strideWidth;
    const int64_t padHeight      = p->padHeight;
    //const int64_t padWidth       = p->padWidth;
    const int64_t dilationHeight = p->dilationHeight;
    //const int64_t dilationWidth  = p->dilationWidth;

    const int64_t inChannelGroup  = inChannel  / group;   // equal to pDataKernel->inChannel
    const int64_t outChannelGroup = outChannel / group;   // equal to pDataKernel->outChannel

    //const int64_t inHW = inHeight * inWidth;
    const int64_t kernHW = kernHeight * kernWidth;
    //const int64_t outHW = outHeight * outWidth;

    PreConvFwd1q pre;
    pre.kh_beg_end.reserve(outHeight*2);
    for(int64_t y=0; y<outHeight; ++y){ // loop_y
        const int64_t i = y * strideHeight - padHeight;
        int64_t kh_end=0;
        const int64_t kh_tmp = dilationHeight-i-1;
        const int64_t kh_beg = (i>=0? 0: kh_tmp / dilationHeight);
        if (i < inHeight){
            kh_end = (inHeight + kh_tmp) / dilationHeight;
            if (kh_end >= kernHeight ) kh_end = kernHeight ;
        }
        pre.kh_beg_end[2*y+0] = kh_beg;
        pre.kh_beg_end[2*y+1] = kh_end;
    }
    // Not same for x0, since x is the vectorization dimn (many values of x)
    // BUT, so from x[]  could generate kw_beg[] and kw_end[] fairly efficiently
    // using vector gather, VGT op.   HOWEVER, still end up calculating a MASK,
    // so unclear whether this is a win.

    // krn_yxrsc size is an overestimate XXX
#if 0
    pre.krn_yxrsc.reserve(group*outChannelGroup*outHeight*outWidth*kernHeight*kernWidth*inChannelGroup);
    for(int64_t g=0; g<group; ++g){ // loop_g
        const int64_t kernGroupOffset = g * outChannelGroup * inChannelGroup * kernHW;
        //int64_t k = 0;
        for(; k<kMax; k+=kBy){ // loop_k
            const float * restrict pKern_gk = pKernel + kernGroupOffset
                /*                       */ + (k * inChannelGroup + 0/*c*/) * kernHW;
            pre.kstarts.push_back(pKern_gk - pKernel); // aid multithreaded breakup of loop_k
            for(int64_t y=0; y<outHeight; ++y){ // loop_y
                const int64_t kh_beg = pre.kh_beg[y];
                const int64_t kh_end = pre.kh_end[y];
                for(int64_t x0=0; x0<outWidth; x0+=vl){ // loop_x0
                    for (int64_t r = kh_beg; r < kh_end; ++r){ // loop_r
                        for (int64_t s = 0; s < kernWidth; ++s){ // loop_s
                            for (int64_t c = 0; c < inChannelGroup; ++c){ // loop_c
                                const float *pKerValue = pKern_gk + c*kernHW + r*kernWidth +s;
                                for(unsigned cc=0; cc<kBy; cc+=2){
                                    pre.krn_yxrsc.push_back( pKerValue + (cc+0)*inChannelGroup*kernHW );
                                    pre.krn_yxrsc.push_back( pKerValue + (cc+1)*inChannelGroup*kernHW );
                                }
                            }
                        }
                    }
                }
            }
        }
    }
#else // make krn indt of both x,y ...
    //
    // Reorder kernel weights so fastest index now loops over outChannel in groups of kBy.
    // (Kinda')
    // kern(g,k,r,s,c) re-ordered as kern(g,K,r,s,c,k') where
    //    K "tries" to be 0..outChannelGroup by 8
    //    (with initial corrections for non-divisibility by 8),
    // and  k' is a block of 8 (or 4,2,1) outChannelGroup values made contiguous
    // so they 2 floats can be loaded together.
    //
    // You could instead work with a more standard reorder of kern(g,K,r,s,c,k={0,1})
    // with perhaps a small index calc overhead for different access patterns for K
    // and less contiguity in loading the kernel values.
    //
    // In the more custom direction, you could calc both kh_beg_end and kw_beg_end
    // and get the minimal-size linearly-accessed kernel values (FASTEST)
    //
    pre.krn_gkrsc_k1.clear();
    pre.krn_gkrsc.clear();
    if( (outChannelGroup & 0x01) != 0 ){
        pre.krn_gkrsc_k1.reserve(group*1*kernHW*inChannelGroup);
    }
    if( outChannelGroup > 1 ){
        pre.krn_gkrsc.reserve((group*(outChannelGroup/2*2)*kernHW*inChannelGroup) / 2);
    }

    const float * restrict pKern_gk;

    for(int64_t g=0; g<group; ++g){ // loop_g
        const int64_t kernGroupOffset = g * outChannelGroup * inChannelGroup * kernHW;
        int64_t k=0;        // 0 .. outChannelGroup-1
        int64_t kBy = 1;    // 1,2,4,... kByMax
        int64_t kMax = k;   // min of k+kBy or outChannelGroup
        if( k<outChannelGroup ){
            if( (outChannelGroup & kBy) != 0 ) kMax = k+kBy;
            if ( kBy == kByMax || kMax>outChannelGroup ) kMax = outChannelGroup;
            int nk1=0;
            for( ; k<kMax; k+=kBy){ // loop_k
                ++nk1;
                pKern_gk = pKernel + kernGroupOffset + (k * inChannelGroup + 0/*c*/) * kernHW;
                // y=0..outHeight
                // x0=0..outWidth by vl
                for (int64_t rs = 0; rs < kernHW; ++rs){ // loop_r, loop_s
                    for (int64_t c = 0; c < inChannelGroup; ++c){ // loop_c
                        const float *pKerValue = pKern_gk + c*kernHW + rs;
                        pre.krn_gkrsc_k1.push_back( pKernel? *pKerValue: 0.0f );
                    }
                }
            }
            assert( pre.krn_gkrsc_k1.size() == (g+1) * nk1 *kernHW*inChannelGroup );
        }else{
            assert( pre.krn_gkrsc_k1.size() == 0 );
        }

        auto krn_rsc_kBy_gt_1 = [&](){
            assert( kBy/2*2 == kBy );
            assert( kBy==2 || kBy==4 || kBy==8 );
            pre.kstarts[(kBy==2? 0: kBy==4? 1: 2)] = pre.krn_gkrsc.size();
            // same krn access pattern for
            // loop_y : y=0..outHeight
            // loop_x : x0=0..outWidth by vl
            // loop_r|s : all r, all s, re-usable FOR ALL y
            for (int64_t rs = 0; rs < kernHW; ++rs) // loop_r, loop_s (krn data packed in rs)
            {
                for (int64_t c = 0; c < inChannelGroup; ++c){ // loop_c
                    const float *pKerValue = pKern_gk + c*kernHW + rs;
                    for(unsigned cc=0; cc<kBy; cc+=2){
                        pre.krn_gkrsc.push_back( PairedFloat{
                                (pKernel? *(pKerValue + (cc+0)*inChannelGroup*kernHW): 0.f),
                                (pKernel? *(pKerValue + (cc+1)*inChannelGroup*kernHW): 0.f)});
                    }
                }
            }
        };
        // kernHW * inChannelGroup pairs pushed back

        if( k<outChannelGroup ){
            kBy = 2;
            if( (outChannelGroup & kBy) != 0 ) kMax = k+kBy;
            if ( kBy == kByMax || kMax>outChannelGroup ) kMax = outChannelGroup;
            auto const k2_0 = pre.krn_gkrsc.size();
#ifndef NDEBUG
            auto const k0 = k;
#endif
            cout<<" g"<<g<<" k"<<k<<" k2@"<<k2_0<<endl;
            assert( k2_0 == g * ((kMax-k)/kBy) * (kernHW*inChannelGroup) );
            for( ; k<kMax; k+=kBy){ // loop_k
                pKern_gk = pKernel + kernGroupOffset + (k * inChannelGroup + 0/*c*/) * kernHW;
                krn_rsc_kBy_gt_1(); // kernHW * inChannelGroup pairs pushed back
                assert( pre.krn_gkrsc.size() - k2_0 == kernHW*inChannelGroup );
            }
            cout<<" kBy=2 : pre.krn_gkrsc.size()="<<pre.krn_gkrsc.size()<<endl;
            assert( pre.krn_gkrsc.size() == k2_0 + ((kMax-k0)/kBy) * kernHW*inChannelGroup );
        }
        if( k<outChannelGroup ){
            kBy = 4;
            if( (outChannelGroup & kBy) != 0 ) kMax = k+kBy;
            if ( kBy == kByMax || kMax>outChannelGroup ) kMax = outChannelGroup;
#ifndef NDEBUG
            auto const k4_0 = pre.krn_gkrsc.size();
            auto const k0 = k;
            assert( k4_0 == g * ((kMax-k)/kBy) * (kernHW*inChannelGroup) );
#endif
            for( ; k<kMax; k+=kBy){ // loop_k
                pKern_gk = pKernel + kernGroupOffset + (k * inChannelGroup + 0/*c*/) * kernHW;
                krn_rsc_kBy_gt_1();
            }
            cout<<" kBy=4 : pre.krn_gkrsc.size()="<<pre.krn_gkrsc.size()<<endl;
            assert( pre.krn_gkrsc.size() == k4_0 + ((kMax-k0)/kBy) * kernHW*inChannelGroup );
        }
        if( k<outChannelGroup ){
            kBy = 8;
            assert( kByMax == 8 );
            kMax = outChannelGroup;
#ifndef NDEBUG
            auto const k8_0 = pre.krn_gkrsc.size();
            auto const k0 = k;
            assert( k8_0 == g * ((kMax-k)/kBy) * (kernHW*inChannelGroup) );
#endif
            for( ; k<kMax; k+=kBy){ // loop_k
                pKern_gk = pKernel + kernGroupOffset + (k * inChannelGroup + 0/*c*/) * kernHW;
                krn_rsc_kBy_gt_1();
            }
            cout<<" kBy=8 : pre.krn_gkrsc.size()="<<pre.krn_gkrsc.size()<<endl;
            assert( pre.krn_gkrsc.size() == k8_0 + ((kMax-k0)/kBy) * kernHW*inChannelGroup );
        }
    }
    assert( pre.krn_gkrsc.size() == (group*(outChannelGroup/2U*2U)*kernHW*inChannelGroup) / 2U );
#endif
    return pre;
}

// C version ...
struct CPreConvFwd1q {
    /** krn_gkrsc.size()*sizeof(PairedFloat) + 3*sizeof(size_t)
     * + krn_gkrsc_k1.size()*sizeof(float) + 2*outHeight*sizeof(int8_t).
     * Minimum size of \c buffer in bytes. buffer must have alignment 8. */
    size_t bufsz;

    /** layout like \e packed struct version of PreConvFwd1q */
    int8_t * buffer;

    /** 0 (\b byte offset of \c krn_gkrsc_k1 within \c buffer) */
    size_t krn_gkrsc_off;

    /** krn_gkrsc.size()*sizeof(PairedFloat) + 3*sizeof(size_t) */
    size_t krn_krsc_k1_off;

    /** krn_gkrsc.size()*sizeof(PairedFloat) + 3*sizeof(size_t) + krn_gkrsc_k1.size()*sizeof(float) */
    size_t kh_beg_end_off;
};

struct CPreConvFwd1q mkCPreConvFwd1q( struct param const* const p,
        float const* const pKernel = nullptr ){
    cout<<" "<<__PRETTY_FUNCTION__<<endl; cout.flush();
    assert( p != nullptr );
    int const kByMax = 8;

    //const int64_t batch          = p->batchNum;
    const int64_t group          = p->group;
    const int64_t inChannel      = p->inChannel;
    const int64_t inHeight       = p->inHeight;
    //const int64_t inWidth        = p->inWidth;
    const int64_t outChannel     = p->outChannel;
    const int64_t outHeight      = p->outHeight;
    //const int64_t outWidth       = p->outWidth;
    const int64_t kernHeight     = p->kernHeight;
    const int64_t kernWidth      = p->kernWidth;
    const int64_t strideHeight   = p->strideHeight;
    //const int64_t strideWidth    = p->strideWidth;
    const int64_t padHeight      = p->padHeight;
    //const int64_t padWidth       = p->padWidth;
    const int64_t dilationHeight = p->dilationHeight;
    //const int64_t dilationWidth  = p->dilationWidth;

    const int64_t inChannelGroup  = inChannel  / group;   // equal to pDataKernel->inChannel
    const int64_t outChannelGroup = outChannel / group;   // equal to pDataKernel->outChannel

    //const int64_t inHW = inHeight * inWidth;
    const int64_t kernHW = kernHeight * kernWidth;
    //const int64_t outHW = outHeight * outWidth;

    PreConvFwd1q pre = mkPreConvFwd1q(p,pKernel);
    CPreConvFwd1q cpre;
    cpre.bufsz =  pre.krn_gkrsc.size()*sizeof(PairedFloat)
        + 3*sizeof(size_t)
        + pre.krn_gkrsc_k1.size()*sizeof(float)
        + outHeight*2*sizeof(int8_t) // assert(kernel size < 128) additional restriction
        ;
    assert( cpre.bufsz == ((group*(outChannelGroup/2U*2U)*kernHW*inChannelGroup) / 2U) * sizeof(PairedFloat)
            + 3U*sizeof(size_t)
            + ((outChannelGroup&0x01)!=0? group*1*kernHW*inChannelGroup: 0) * sizeof(float)
            + outHeight*2U*sizeof(int8_t) );

    //cpre.krn_gkrsc_off = 0U;
    cpre.krn_krsc_k1_off = cpre.krn_gkrsc_off
        + ((group*(outChannelGroup/2U*2U)*kernHW*inChannelGroup) / 2U) * sizeof(PairedFloat);
    cpre.kh_beg_end_off = cpre.krn_krsc_k1_off
        + ((outChannelGroup&0x01)!=0? group*1*kernHW*inChannelGroup: 0) * sizeof(float);

    assert( cpre.bufsz = cpre.kh_beg_end_off + outHeight*2U*sizeof(uint8_t) );
    assert(sizeof(PairedFloat) == 2*sizeof(float));
    assert(sizeof(PairedFloat) == sizeof(uint64_t));

    int8_t* buffer = (int8_t*)malloc((cpre.bufsz+sizeof(uint64_t)-1U)/sizeof(uint64_t)*sizeof(uint64_t));
    cpre.buffer = buffer;
    uint8_t     * kh_beg_end   = (uint8_t *)    ( buffer + cpre.kh_beg_end_off );
    float       * krn_gkrsc_k1 = (float *)      ( buffer + cpre.krn_krsc_k1_off );
    PairedFloat * krn_gkrsc    = (PairedFloat *)( buffer + cpre.krn_gkrsc_off );
    //PairedFloat * krn_gkrsc_0  = krn_gkrsc;

    for(int64_t y=0; y<outHeight; ++y){ // loop_y
        const int64_t i = y * strideHeight - padHeight;
        int64_t kh_end=0;
        const int64_t kh_tmp = dilationHeight-i-1;
        const int64_t kh_beg = (i>=0? 0: kh_tmp / dilationHeight);
        if (i < inHeight){
            kh_end = (inHeight + kh_tmp) / dilationHeight;
            if (kh_end >= kernHeight ) kh_end = kernHeight ;
        }
        assert(kh_beg >= 0);
        assert(kh_end <= 255);
        *kh_beg_end++ = (uint8_t)kh_beg;
        *kh_beg_end++ = (uint8_t)kh_end;
    }
    assert( kh_beg_end - (uint8_t*)buffer == cpre.bufsz );

    const float * restrict pKern_gk;
    for(int64_t g=0; g<group; ++g){ // loop_g
        const int64_t kernGroupOffset = g * outChannelGroup * inChannelGroup * kernHW;
        int64_t k=0;        // 0 .. outChannelGroup-1
        int64_t kBy = 1;    // 1,2,4,... kByMax
        int64_t kMax = k;   // min of k+kBy or outChannelGroup
        if( k<outChannelGroup ){
            if( (outChannelGroup & kBy) != 0 ) kMax = k+kBy;
            if ( kBy == kByMax || kMax>outChannelGroup ) kMax = outChannelGroup;
            for( ; k<kMax; k+=kBy){ // loop_k
                pKern_gk = pKernel + kernGroupOffset + (k * inChannelGroup + 0/*c*/) * kernHW;
                for (int64_t rs = 0; rs < kernHW; ++rs){ // loop_r, loop_s
                    for (int64_t c = 0; c < inChannelGroup; ++c){ // loop_c
                        const float *pKerValue = pKern_gk + c*kernHW + rs;
                        *krn_gkrsc_k1++ = ( pKernel? *pKerValue: 0.0f );
                    }
                }
            }
        }

        if( k<outChannelGroup ){
            kBy = 2;
            if( (outChannelGroup & kBy) != 0 ) kMax = k+kBy;
            if ( kBy == kByMax || kMax>outChannelGroup ) kMax = outChannelGroup;
            for( ; k<kMax; k+=kBy){ // loop_k
                pKern_gk = pKernel + kernGroupOffset + (k * inChannelGroup + 0/*c*/) * kernHW;
                //printf("g%lluk%llu pKern_gk-pKernel=%llu",(llu)g,(llu)k,(llu)(pKern_gk-pKernel));
                for (int64_t rs = 0; rs < kernHW; ++rs){ // loop_r, loop_s (krn data packed in rs)
                    for (int64_t c = 0; c < inChannelGroup; ++c){ // loop_c
                        const float *pKerValue = pKern_gk + c*kernHW + rs;
                        for(unsigned cc=0; cc<kBy; cc+=2){
                            krn_gkrsc->f[0] = (pKernel? *(pKerValue + (cc+0)*inChannelGroup*kernHW): 0.f);
                            krn_gkrsc->f[1] = (pKernel? *(pKerValue + (cc+1)*inChannelGroup*kernHW): 0.f);
                            krn_gkrsc++;
                        }
                    }
                }
            }
        }
        if( k<outChannelGroup ){
            kBy = 4;
            if( (outChannelGroup & kBy) != 0 ) kMax = k+kBy;
            if ( kBy == kByMax || kMax>outChannelGroup ) kMax = outChannelGroup;
            for( ; k<kMax; k+=kBy){ // loop_k
                pKern_gk = pKernel + kernGroupOffset + (k * inChannelGroup + 0/*c*/) * kernHW;
                for (int64_t rs = 0; rs < kernHW; ++rs){ // loop_r, loop_s (krn data packed in rs)
                    for (int64_t c = 0; c < inChannelGroup; ++c){ // loop_c
                        const float *pKerValue = pKern_gk + c*kernHW + rs;
                        for(unsigned cc=0; cc<kBy; cc+=2){
                            krn_gkrsc->f[0] = (pKernel? *(pKerValue + (cc+0)*inChannelGroup*kernHW): 0.f);
                            krn_gkrsc->f[1] = (pKernel? *(pKerValue + (cc+1)*inChannelGroup*kernHW): 0.f);
                            krn_gkrsc++;
                        }
                    }
                }
            }
        }
        if( k<outChannelGroup ){
            kBy = 8;
            assert( kByMax == 8 );
            kMax = outChannelGroup;
            for( ; k<kMax; k+=kBy){ // loop_k
                pKern_gk = pKernel + kernGroupOffset + (k * inChannelGroup + 0/*c*/) * kernHW;
                for (int64_t rs = 0; rs < kernHW; ++rs){ // loop_r, loop_s (krn data packed in rs)
                    for (int64_t c = 0; c < inChannelGroup; ++c){ // loop_c
                        const float *pKerValue = pKern_gk + c*kernHW + rs;
                        for(unsigned cc=0; cc<kBy; cc+=2){
                            krn_gkrsc->f[0] = (pKernel? *(pKerValue + (cc+0)*inChannelGroup*kernHW): 0.f);
                            krn_gkrsc->f[1] = (pKernel? *(pKerValue + (cc+1)*inChannelGroup*kernHW): 0.f);
                            krn_gkrsc++;
                        }
                    }
                }
            }
        }
    }
    assert( (int8_t*)krn_gkrsc    - (int8_t*)buffer == cpre.krn_krsc_k1_off );
    assert( (int8_t*)krn_gkrsc_k1 - (int8_t*)buffer == cpre.kh_beg_end_off );
    //assert( (int8_t*)kh_beg_end   - (int8_t*)buffer == cpre.bufsz );
    return cpre;
}
std::pair<string,string> strCPreConvFwd1q( struct param const* const p, int const kByMax, int64_t& vlen ){
    cout<<" "<<__PRETTY_FUNCTION__<<endl; cout.flush();
    //const int64_t batch          = p->batchNum;
    const int64_t group          = p->group;
    const int64_t inChannel      = p->inChannel;
    //const int64_t inWidth        = p->inWidth;
    const int64_t outChannel     = p->outChannel;
    //const int64_t outWidth       = p->outWidth;
    const int64_t kernWidth      = p->kernWidth;
    const int64_t kernHeight     = p->kernHeight;
    //const int64_t strideWidth    = p->strideWidth;
    //const int64_t padWidth       = p->padWidth;
    //const int64_t dilationWidth  = p->dilationWidth;
    const int64_t inChannelGroup  = inChannel  / group;   // equal to pDataKernel->inChannel
    const int64_t outChannelGroup = outChannel / group;   // equal to pDataKernel->outChannel

    //const int64_t inHW = inHeight * inWidth;
    const int64_t kernHW = kernHeight * kernWidth;
    //const int64_t outHW = outHeight * outWidth;

    // mirror the loop structure and count how many things are needed
    // XXX remove the loop over 'g' and then multiply by 'group' for nk1,2,4,8
    size_t nk1=0U;
    size_t nk2=0U;
    size_t nk4=0U;
    size_t nk8=0U;
    size_t kMax1=0U;
    size_t kMax2=0U;
    size_t kMax4=0U;
    size_t kMax8=0U;
    for(int64_t g=0; g<group; ++g){ // loop_g
        int64_t k=0;        // 0 .. outChannelGroup-1
        int64_t kBy = 1;    // 1,2,4,... kByMax
        int64_t kMax = k;   // min of k+kBy or outChannelGroup
        if( k<outChannelGroup ){
            if( (outChannelGroup & kBy) != 0 ) kMax = k+kBy;
            if ( kBy == kByMax || kMax>outChannelGroup ) kMax = outChannelGroup;
            for( ; k<kMax; k+=kBy){ // loop_k
                for (int64_t rs = 0; rs < kernHW; ++rs){ // loop_r, loop_s
                    for (int64_t c = 0; c < inChannelGroup; ++c){ // loop_c
                        //if(rs==0 && c==0) printf("kBy=%d g=%d k=%d nk=%d\n",(int)kBy,(int)g,(int)k,(int)nk1);
                        nk1+=1; // single kernel value
                    }
                }
            }
            kMax1 = k;
        }
        if( k<outChannelGroup ){
            kBy = 2;
            if( (outChannelGroup & kBy) != 0 ) kMax = k+kBy;
            if ( kBy == kByMax || kMax>outChannelGroup ) kMax = outChannelGroup;
#ifndef NDEBUG
            auto const k0 = k;
#endif
            for( ; k<kMax; k+=kBy){ // loop_k
                for (int64_t rs = 0; rs < kernHW; ++rs){ // loop_r, loop_s (krn data packed in rs)
                    for (int64_t c = 0; c < inChannelGroup; ++c){ // loop_c
                        nk2+=(kBy/2); // these are actually pairs of kernel values
                    }
                }
            }
            assert( nk2 == (g+1) * ((k-k0)/kBy) * (kBy/2)*kernHW*inChannelGroup );
            kMax2 = k;
        }
        if( k<outChannelGroup ){
            kBy = 4;
            if( (outChannelGroup & kBy) != 0 ) kMax = k+kBy;
            if ( kBy == kByMax || kMax>outChannelGroup ) kMax = outChannelGroup;
            kMax4 = kMax;
            for( ; k<kMax; k+=kBy){ // loop_k
                for (int64_t rs = 0; rs < kernHW; ++rs){ // loop_r, loop_s (krn data packed in rs)
                    for (int64_t c = 0; c < inChannelGroup; ++c){ // loop_c
                        nk4+=(kBy/2);   // uses 2 pairs of floats
                    }
                }
            }
            kMax4 = k;
        }
        if( k<outChannelGroup ){
            kBy = 8;
            assert( kByMax == 8 );
            kMax = outChannelGroup;
            for( ; k<kMax; k+=kBy){ // loop_k
                for (int64_t rs = 0; rs < kernHW; ++rs){ // loop_r, loop_s (krn data packed in rs)
                    for (int64_t c = 0; c < inChannelGroup; ++c){ // loop_c
                        nk8+=(kBy/2); // 4 pairs of floats
                    }
                }
            }
            kMax8 = k;
        }
    }
    pair<string,string> ret("","");
    if(    !((PRE_KRN_MAX>=1) && nk1>0)
        && !((PRE_KRN_MAX>=2) && nk2>0)
        && !((PRE_KRN_MAX>=4) && nk4>0)
        && !((PRE_KRN_MAX>=8) && nk8>0)
        && (PRE_KH_BEG_END==0 || (PRE_KH_BEG_END && KH_BE))
      ){ // then we have no work to do
        cout<<" (CpreConvFwd1q not needed)";
    }else{
        ostringstream oss;
        Cunit tmp("tmp");
        tmp.v=0;
        tmp.root
            >>OSSFMT("// orig: nk1="<<nk1<<" nk2="<<nk2<<" nk4="<<nk4<<" nk8="<<nk8)
            ;
        if(PRE_KRN_MAX < 1) nk1 = 0;
        if(PRE_KRN_MAX < 2) nk2 = 0;
        if(PRE_KRN_MAX < 4) nk4 = 0;
        if(PRE_KRN_MAX < 8) nk8 = 0;
        tmp.root
            >>OSSFMT("// PRE_KRN_MAX="<<PRE_KRN_MAX)
            >>OSSFMT("//       nk1="<<nk1<<" nk2="<<nk2<<" nk4="<<nk4<<" nk8="<<nk8)
            ;
            //>>"//const int64_t batch          = p->batchNum;"
            //>>"const int64_t group          = p->group;"
            //>>"const int64_t inChannel      = p->inChannel;"
            //>>"const int64_t inHeight       = p->inHeight;"
            //>>"//const int64_t inWidth        = p->inWidth;"
            //>>"const int64_t outChannel     = p->outChannel;"
            //>>"const int64_t outHeight      = p->outHeight;"
            //>>"//const int64_t outWidth       = p->outWidth;"
            //>>"const int64_t kernHeight     = p->kernHeight;"
            //>>"const int64_t kernWidth      = p->kernWidth;"
            //>>"const int64_t strideHeight   = p->strideHeight;"
            //>>"//const int64_t strideWidth    = p->strideWidth;"
            //>>"const int64_t padHeight      = p->padHeight;"
            //>>"//const int64_t padWidth       = p->padWidth;"
            //>>"const int64_t dilationHeight = p->dilationHeight;"
            //>>"//const int64_t dilationWidth  = p->dilationWidth;"

            //>>"const int64_t inChannelGroup  = inChannel  / group;   // equal to pDataKernel->inChannel"
            //>>"const int64_t outChannelGroup = outChannel / group;   // equal to pDataKernel->outChannel"

            //>>"//const int64_t inHW = inHeight * inWidth;"
            //>>"const int64_t kernHW = kernHeight * kernWidth;"
            //>>"//const int64_t outHW = outHeight * outWidth;"
#if KH_BE==1
        const int64_t outHeight      = p->outHeight;
#endif
        int64_t bufsz_bytes = 
            (nk2+nk4+nk8) * sizeof(union PairedFloat)
            + nk1 * sizeof(float)
#if KH_BE==1
            + outHeight*2U*sizeof(int8_t)
#endif
            ;
        int64_t krn_krsc_k2_off = 0U;
        int64_t krn_krsc_k4_off = krn_krsc_k2_off + nk2 * sizeof(union PairedFloat);
        int64_t krn_krsc_k8_off = krn_krsc_k4_off + nk4 * sizeof(union PairedFloat);
        int64_t krn_krsc_k1_off = krn_krsc_k2_off  + (nk2+nk4+nk8) * sizeof(union PairedFloat);
        auto& tmp_beg = tmp["beg"];
        auto& tmp_last = tmp["last"];
        tmp_beg CONST1(PRE_KH_BEG_END) CONST1(KH_BE);
        tmp_last FREE1(PRE_KH_BEG_END) FREE1(KH_BE);
        // XXX if bufsz_bytes is "too large", instead use malloc/free TODO
        //rough estimate of kernel parms:
        // group*outChannelGroup*inChannelGroup*kernHW *sizeof(float)
        //Best: set outChannelGroup and inChannelGroup BLOCKING (outer loops!)
        //      so that inner loops re-use a limited-size kernel re-ordering buffer

        if(bufsz_bytes > JIT_MALLOC_THRESHOLD_K*1024L ){
            cout<<"malloc buf64[(bufsz_bytes + 7) / 8] = buf64["<<(bufsz_bytes + 7) / 8<<"]"<<endl;
            cout<<"should BLOCK outer loops better!"<<endl;
            tmp_beg>>"uint64_t* buf64 = (uint64_t*)malloc("<<asDec((bufsz_bytes+7)/8*8)<<");"
                >>"assert(buf64!=NULL);";
            tmp_last>>"free(buf64);";
        }else{
            tmp_beg
                //>>"struct CPreConvFwd1q cpre;"
                >>"uint64_t buf64["
#if 0
                >>"        ( (nk2+nk4+nk8) * sizeof(union PairedFloat)"
                >>"        + nk1 * sizeof(float)"
#if KH_BE==1
                >>"        + outHeight*2U*sizeof(int8_t)"
#endif
                >>"        + 7) /*bytes*/   /    8   /*round up to force 8-byte aligment*/"
#else
                >>"        "+asDec( (bufsz_bytes + 7) / 8 )+" /* >= "+asDec(bufsz_bytes)+" bytes */"
#endif
                >>"        ];"
                ;
        }
        tmp_beg
            >>"int8_t *buffer = (int8_t*)&buf64[0];"
            >>"// exported precalc constants:"
            CONST1(nk1) CONST1(nk2) CONST1(nk4) CONST1(nk8)
            CONST1(krn_krsc_k1_off)
            CONST1(krn_krsc_k2_off)
            CONST1(krn_krsc_k4_off)
            CONST1(krn_krsc_k8_off)
            ;
        tmp_last FREE1(nk1) FREE1(nk2) FREE1(nk4) FREE1(nk8)
            FREE1(krn_krsc_k1_off)
            FREE1(krn_krsc_k2_off)
            FREE1(krn_krsc_k4_off)
            FREE1(krn_krsc_k8_off)
            ;
#if KH_BE==1
        int64_t kh_beg_end_off  = krn_krsc_k1_off + nk1 * sizeof(float);
        tmp_beg CONST1(kh_beg_end_off);
        tmp_last FREE1(kh_beg_end_off);
#endif

#if KH_BE==1
        //const int64_t inHeight       = p->inHeight;
        //const int64_t outHeight      = p->outHeight;
        //const int64_t strideHeight   = p->strideHeight;
        //const int64_t padHeight      = p->padHeight;
        //const int64_t dilationHeight = p->dilationHeight;

        // NOTE: these are const, and should not be calculated at all (i.e. not even in 'buffer')
        //tmp_beg>>"cpre.kh_beg_end_off = kh_beg_end_off;";
        CBLOCK_SCOPE(kh_beg_end,"",tmp,tmp_beg);
        kh_beg_end
            >>"uint8_t * kh_beg_end = (uint8_t *)( buffer + kh_beg_end_off );"
            >>"for(int64_t y=0; y<outHeight; ++y){ // loop_y"
            >>"    const int64_t i = y * strideHeight - padHeight;"
            >>"    int64_t kh_end=0;"
            >>"    const int64_t kh_tmp = dilationHeight-i-1;"
            >>"    const int64_t kh_beg = (i>=0? 0: kh_tmp / dilationHeight);"
            >>"    if (i < inHeight){"
            >>"        kh_end = (inHeight + kh_tmp) / dilationHeight;"
            >>"        if (kh_end >= kernHeight ) kh_end = kernHeight ;"
            >>"    }"
            //>>"    assert(kh_beg >= 0);"
            //>>"    assert(kh_end <= 255);"
            //>>"    assert(kh_end <= kernHeight);"
            //>>"    printf(\" kh_beg_end @ %p ={%d, %d}\\n\", (void*)kh_beg_end, (int)kh_beg, (int)kh_end);"
            >>"    *kh_beg_end++ = (uint8_t)kh_beg;"
            >>"    *kh_beg_end++ = (uint8_t)kh_end;"
            >>"}"
            //>>"assert( kh_beg_end - (uint8_t*)buffer == cpre.bufsz );"
            ;
#endif

        if(        ((PRE_KRN_MAX>=1) && nk1>0)
                || ((PRE_KRN_MAX>=2) && nk2>0)
                || ((PRE_KRN_MAX>=4) && nk4>0)
                || ((PRE_KRN_MAX>=8) && nk8>0) )
        {
#if PRE_KRN_MAX >= 1
            CBLOCK_SCOPE(precalc,"",tmp,tmp_beg);
            auto& setup = precalc["setup"];
            setup>>"const float * restrict pKern_gk;";

            CBLOCK_SCOPE(loop_g,"for(int64_t g=0; g<group; ++g)",tmp,setup); // OK sub-tree
            //auto& setup_end = precalc["last"];

            loop_g
                >>"const int64_t kernGroupOffset = g * outChannelGroup * inChannelGroup * kernHW;"
                >>"int64_t k=0;        // 0 .. outChannelGroup-1"
                ;
            if(nk1 > 0){
                int64_t kBy = 1, kMax=kMax1;
                setup>>"float* restrict k1Out = (float *)(buffer + krn_krsc_k1_off);";

                auto& prekby = loop_g["pre-kBy1"];
                prekby CONST1(kBy) CONST1(kMax);
                CBLOCK_SCOPE(for_k_kBy,"for( ; k<kMax; k+=kBy)",tmp,prekby);
                prekby["last"] FREE1(kMax) FREE1(kBy);

                for_k_kBy>>"pKern_gk = pKernel + kernGroupOffset + (k * inChannelGroup + 0/*c*/) * kernHW;";
                CBLOCK_SCOPE(for_rs,"for (int64_t rs = 0; rs < kernHW; ++rs)",tmp,for_k_kBy);
#if 0
                for_rs
                    >>"for (int64_t c = 0; c < inChannelGroup; ++c){ // loop_c"
                    >>"    const float *pKerValue = pKern_gk + c*kernHW + rs;"
                    >>"    *k1Out++ = *pKerValue;"
                    >>"}"
                    ;
#else
                cout<<"krn Weight copy using vel_vcopy32 (not always optimal?)"<<endl;
                for_rs
                    >>"float const* pKerValue = pKern_gk + rs;"
                    // TODO: MOVE a Cblock (subtree) from some tmp Cunit after a given CBlock (in another Cunit)
                    //       or maybe move a full Cunit under one of our known nodes ?
                    // TODO: optimize enclosing loop by providing a submatrix copy (2 arb strides-->[unit] const stride)
                    >>vel_vcopy32("pKerValue",kernHW,inChannelGroup,"k1Out",1)
                    //            source     ,stride,     N        , dest  ,stride
                    >>OSSFMT("k1Out += inChannelGroup; /* +="<<inChannelGroup<<" */");
                    ;
#endif
            }else{
                setup>>"/* no kBy1 loop */";
            }
#if PRE_KRN_MAX >= 2
            //precalc>>"cpre.krn_krsc_k2_off = 0U;";
            if(nk2>0U){
                int64_t kBy = 2, kMax=kMax2;
                setup>>"union PairedFloat* restrict k2Out0 = (union PairedFloat *)(buffer + krn_krsc_k2_off);";
                setup>>"union PairedFloat* restrict k2Out = k2Out0;";

                auto& prekby = loop_g["pre-kBy2"];
                prekby CONST1(kBy) CONST1(kMax);
                CBLOCK_SCOPE(for_k_kBy,"for( ; k<kMax; k+=kBy)",tmp,prekby);
                prekby["last"] FREE1(kMax) FREE1(kBy);

                for_k_kBy>>"pKern_gk = pKernel + kernGroupOffset + (k*inChannelGroup+0/*c*/) * kernHW;";
                CBLOCK_SCOPE(for_rs,"for (int64_t rs = 0; rs < kernHW; ++rs)",tmp,for_k_kBy);
                //for_rs>>"printf(\" precalc 1q-kBy2 gk=%d %d rs=%d pKern %d -> k2 %d\\n\",(int)g,(int)k,(int)rs,(int)(pKern_gk+rs-pKernel),(int)(k2Out - k2Out0));";

#if 0 // original
                for_rs
                    >>"for (int64_t c = 0; c < inChannelGroup; ++c){ // loop_c"
                    >>"    const float *pKerValue = pKern_gk + c*kernHW + rs;"
#if 1 // original uses an intrinsic...
                    >>"    const uint64_t kerValue01 = _vel_pack_f32p("
                    >>"                   pKerValue,"
                    >>"                   pKerValue + inChannelGroup*kernHW);"
                    >>"    k2Out->pair = kerValue01;"
#else // NOTE THE INVERSION ........
                    >>"    k2Out->f[1] = *(pKerValue+0*inChannelGroup*kernHW);"
                    >>"    k2Out->f[0] = *(pKerValue+1*inChannelGroup*kernHW);"
#endif
                    >>"    ++k2Out;"
                    >>"}"
                    ;
#elif 0 // scalar loop version
                for_rs
                    >>"const uint32_t* pKerValue = (uint32_t const*)(float const*)(pKern_gk) /* + c*kernHW */ + rs;"
                    >>"for (int64_t c = 0; c < inChannelGroup; ++c){ // loop_c"
                    >>"    k2Out->pair"
                    //>>"    ((uint64_t*)(void*)(k2Out))[c*1]"
                    >>"      = (((uint64_t) ((pKerValue+0*inChannelGroup*kernHW)[c*kernHW])) << 32)"
                    >>"      | (((uint64_t) ((pKerValue+1*inChannelGroup*kernHW)[c*kernHW]))      );"
                    >>"    ++k2Out;"
                    >>"}"
                    ;
#elif 0 // scalar loop version #2
                for_rs
                    >>"uint32_t const* pKerValue = (uint32_t const*)(float const*)(pKern_gk) /* + c*kernHW */ + rs;"
                    >>"for (int64_t c = 0; c < inChannelGroup; ++c){ // loop_c"
                    >>"    ((uint64_t*)(void*)(k2Out))[c*1]"
                    >>"      = (((uint64_t) ((pKerValue+0*inChannelGroup*kernHW)[c*kernHW])) << 32)"
                    >>"      | (((uint64_t) ((pKerValue+1*inChannelGroup*kernHW)[c*kernHW]))      );"
                    >>"}"
                    >>"k2Out += inChannelGroup;"
                    ;
#else
                // roughly, want
                // &k2Out->f[1] stride 8 <-- 'ic' values stride kernHW beginning at pKerValue
                // &k2Out->f[0] stride 8 <-- 'ic' values stride kernHW beginning at pKerValue+inChannelGroup*kernHW
                //    perhaps as vldu, vldlzx, vshuf, packed store
                // k2Out+=inChannelGroup
                //assert( ((uintptr_t)(pKerValue+0*inChannelGroup*kernHW) & 0x3) == 0)
                //assert( ((uintptr_t)(k2Out) & 0x7) == 0)
                for_rs
                    >>"float const* pKerValue = pKern_gk + rs;"
                    >>vel_vmerge32( // memory merge
                            "pKerValue+0*inChannelGroup*kernHW", kernHW,
                            "pKerValue+1*inChannelGroup*kernHW", kernHW,
                            inChannelGroup,
                            "k2Out",1)
                    ;
                for_rs>>"k2Out += inChannelGroup;"; // position?
#endif
            }else{
                setup>>"/* no kBy2 loop */";
            }
#if PRE_KRN_MAX >= 4
            //precalc>>"cpre.krn_krsc_k4_off = (nk2) * sizeof(union PairedFloat);";
            if(nk4>0){
                int64_t kBy = 4, kMax=kMax4;
                setup>>"union PairedFloat* restrict k4Out = (union PairedFloat *)(buffer + krn_krsc_k4_off);";
                auto& prekby = loop_g["pre-kBy4"];
                prekby CONST1(kBy) CONST1(kMax);

                CBLOCK_SCOPE(for_k_kBy,"for( ; k<kMax; k+=kBy)",tmp,prekby);
                prekby["last"] FREE1(kMax) FREE1(kBy);

                for_k_kBy>>"pKern_gk = pKernel + kernGroupOffset + (k * inChannelGroup + 0/*c*/) * kernHW;";
                CBLOCK_SCOPE(for_rs,"for (int64_t rs = 0; rs < kernHW; ++rs)",tmp,for_k_kBy);
#if 0 // original
                for_rs
                    >>"for (int64_t c = 0; c < inChannelGroup; ++c){ // loop_c"
                    >>"    const float *pKerValue = pKern_gk + c*kernHW + rs;"
                    >>"    k4Out->f[1] = *(pKerValue+0*inChannelGroup*kernHW);"
                    >>"    k4Out->f[0] = *(pKerValue+1*inChannelGroup*kernHW);"
                    >>"    ++k4Out;"
                    >>"    k4Out->f[1] = *(pKerValue+2*inChannelGroup*kernHW);"
                    >>"    k4Out->f[0] = *(pKerValue+3*inChannelGroup*kernHW);"
                    >>"    ++k4Out;"
                    >>"}"
                    ;
#elif 0
                for_rs
                    >>"const float *pKerValue = pKern_gk /*+ c*kernHW*/ + rs;"
                    >>"for (int64_t c = 0; c < inChannelGroup; ++c){ // loop_c"
                    >>"    (k4Out+0)[2*c].f[1] = *(pKerValue+0*inChannelGroup*kernHW + c*kernHW);"
                    >>"    (k4Out+0)[2*c].f[0] = *(pKerValue+1*inChannelGroup*kernHW + c*kernHW);"
                    >>"    (k4Out+1)[2*c].f[1] = *(pKerValue+2*inChannelGroup*kernHW + c*kernHW);"
                    >>"    (k4Out+1)[2*c].f[0] = *(pKerValue+3*inChannelGroup*kernHW + c*kernHW);"
                    >>"}"
                    >>"k4Out += 2*inChannelGroup;"
                    ;
#else // vectorized ...
                for_rs
                    >>"float const* pKerValue = pKern_gk + rs;"
                    >>vel_vmerge32( // memory merge
                            "pKerValue+0*inChannelGroup*kernHW", kernHW,
                            "pKerValue+1*inChannelGroup*kernHW", kernHW,
                            inChannelGroup,
                            "k4Out+0",2,"") // 2 is kBy/2
                    >>vel_vmerge32( // memory merge
                            "pKerValue+2*inChannelGroup*kernHW", kernHW,
                            "pKerValue+3*inChannelGroup*kernHW", kernHW,
                            inChannelGroup,
                            "k4Out+1",2,"1"/*reg-name-disambiguation-suffix*/)
                    >>"k4Out += 2 * inChannelGroup;"
                    ;
#endif
            }else{
                setup>>"/* no kBy4 loop */";
            }
#if PRE_KRN_MAX >= 8
            //precalc>>"cpre.krn_krsc_k8_off = (nk2+nk4) * sizeof(union PairedFloat);";
            if(nk8>0){
                int64_t kBy = 8, kMax=kMax8;
                setup>>"union PairedFloat* restrict k8Out = (union PairedFloat *)(buffer + krn_krsc_k8_off);";
                auto& prekby = loop_g["pre-kBy8"];
                prekby CONST1(kBy) CONST1(kMax);

                CBLOCK_SCOPE(for_k_kBy,"for( ; k<kMax; k+=kBy)",tmp,prekby);
                prekby["last"] FREE1(kMax) FREE1(kBy);

                for_k_kBy>>"pKern_gk = pKernel + kernGroupOffset + (k * inChannelGroup + 0/*c*/) * kernHW;";
                CBLOCK_SCOPE(for_rs,"for (int64_t rs = 0; rs < kernHW; ++rs)",tmp,for_k_kBy);
#if 0
                for_rs
                    >>"for (int64_t c = 0; c < inChannelGroup; ++c){ // loop_c"
                    >>"    const float *pKerValue = pKern_gk + c*kernHW + rs;"
                    >>"    for(unsigned cc=0; cc<kBy; cc+=2){"
                    >>"        k8Out->f[1] = *(pKerValue+(cc  )*inChannelGroup*kernHW);"
                    >>"        k8Out->f[0] = *(pKerValue+(cc+1)*inChannelGroup*kernHW);"
                    >>"        ++k8Out;"
                    >>"    }"
                    >>"}"
                    ;
#else // vectorized
                for_rs
                    >>"float const* pKerValue = pKern_gk + rs;"
                    >>vel_vmerge32( // memory merge
                            "pKerValue+0*inChannelGroup*kernHW", kernHW,
                            "pKerValue+1*inChannelGroup*kernHW", kernHW,
                            inChannelGroup,
                            "k8Out+0",4,"01") // 4 is kBy/4
                    >>vel_vmerge32( // memory merge
                            "pKerValue+2*inChannelGroup*kernHW", kernHW,
                            "pKerValue+3*inChannelGroup*kernHW", kernHW,
                            inChannelGroup,
                            "k8Out+1",4,"23")
                    >>vel_vmerge32( // memory merge
                            "pKerValue+4*inChannelGroup*kernHW", kernHW,
                            "pKerValue+5*inChannelGroup*kernHW", kernHW,
                            inChannelGroup,
                            "k8Out+2",4,"45")
                    >>vel_vmerge32( // memory merge
                            "pKerValue+6*inChannelGroup*kernHW", kernHW,
                            "pKerValue+7*inChannelGroup*kernHW", kernHW,
                            inChannelGroup,
                            "k8Out+3",4,"67")
                    >>"k8Out += 4 * inChannelGroup;"
                ;
#endif
            }else{
                setup>>"/* no kBy8 loop */";
            }
#endif
#endif
#endif
#endif // PRE_KRN_MAX
        } // Some sort or kernel weights rewrite has been enabled & is required
        ret.first = tmp_beg.str();
        ret.second = tmp_last.str();
    }
    return ret;
}


/** this kByMax 8 is based on implementation direct_default2p.c, POUTX==1.
 *
 * \todo if kh_beg is always 0 and kh_end is always max, then elide kh_btab and kh_etab tables
 * (or whatever KH_BEG_END macro does) with "trivial case" implementations.
 *
 */
DllFile cjitConvolutionForward1q( struct param const* const p )
{
    std::ostringstream oss;
    int const verbose=0;
    string const impl = "cjitConvFwd1q";
    DllFile df; // return value
    //DllFileAux dfx("Convolution","Forward");
    std::string parmstr = paramString(p);
    df.basename = impl+"_"+parmstr;
    cout<<impl<<" : df.basename = "<<df.basename<<endl;

    // we use intrinsics.  suffix matches build recipe in "bin.mk"
    df.suffix = "-vi.c";

    Cunit pr("program");
    pr.v = verbose;     // default is quite lengthy!

    int64_t const kByMax = KBYMAX; // kByMax is chosen from {1,2,4,8}
    cout<<impl<<" KBYMAX="<<KBYMAX
        <<" PRE_KH_BEG_END="<<PRE_KH_BEG_END
        <<" PRE_KRN_MAX="<<PRE_KRN_MAX
        <<" KH_BE="<<KH_BE<<" kByMax="<<kByMax<<endl;

#if 0
    PreConvFwd1q pre = mkPreConvFwd1q( p, nullptr ); // nullptr : check assertions
    cout<<"pre: "<<pre<<endl;
#endif

    auto& includes = pr["includes"]<<Endl;
    includes
        >>CSTR(#include "vednn.h")
        >>CSTR(#if __has_include("vednnx.h")) // an old clang directive
        >>CSTR(#include "vednnx.h")
        >>CSTR(#endif)
#if VEL_BUG
        >>CSTR(#include "veintrin.h")
#endif
        >>CSTR(#include "velintrin.h")
        >>"#include <stdio.h>"
        >>"#include <stdlib.h>"
        >>"#include <assert.h>"
        >>"#include <stdint.h>"
        ;
    pr["macros"]
        >>"#if 0"
        >>"static void err_print(char const* file, int const line, char const* what, int const requirement){"
        >>"  if(!requirement){"
        >>"    "<<CSTR(printf(" Error %s:%d failed CHK: %s\n",file,line,what);)
        >>"  }else{"
        >>"    "<<"//"<<CSTR(printf("  OK   %s:%d\n",file,line);)
        >>"  }"
        >>"  fflush(stdout);"
        >>"}"
        >>"#define CHK(REQUIREMENT) err_print(__FILE__,__LINE__,#REQUIREMENT,(REQUIREMENT));"
        >>"#else"
        >>"#define CHK(REQUIREMENT) do {;}while(0)"
        >>"#endif"
        >>""
        >>"#if "<<asDec(VEL_BUG)
        >>"// sometimes enabling this can fix 'wrong result'"
        >>"//        Simple test case: jitconv -p mb64ih3ic1oc1_kh3ph0"
        >>"#define NO_SET_VLEN( VLEN ) _ve_lvl(VLEN)"
        >>""
        >>"#else // but pure vel intrinsics should do nothing"
        >>"#define NO_SET_VLEN( VLEN ) do{}while(0)"
        >>"#endif"
        // XXX missing #undefs for above
        //.def("MVL",256)
        ;
    std::string fn_declare = "vednnError_t "+df.basename+"(\n    "+
        multiReplace(",",",\n    ", CSTR(CONVX_FWD_ORDER(
                    VEDNN_PARAMS_CONV_FORWARD,
                    VEDNN_DATARG_CONV_FORWARD))) +"\n)"
        ;
    df.syms.push_back(SymbolDecl(df.basename,
                "vednn ConvolutionForward "+paramString(p),
                fn_declare));

    pr["structs"]
        >>"union PairedFloat {"
        >>"    float f[2];                     // packed-pair floats"
        >>"    uint64_t pair;                  // forces alignment 8"
        >>"};"
        >>"struct CPreConvFwd1q {"
        >>"    size_t bufsz;"
        >>"    int8_t * buffer;"
        >>"    size_t krn_gkrsc_off;"
        >>"    size_t krn_krsc_k1_off;"
        >>"    size_t krn_krsc_k2_off;"
        >>"    size_t krn_krsc_k4_off;"
        >>"    size_t krn_krsc_k8_off;"
#if KH_BEG==1
        >>"    size_t kh_beg_end_off;"
#endif
        >>"};"
        >>"typedef long long unsigned llu;"
        ;



    //auto & fns = mk_extern_c(pr,"extern_C").after(pr["/macros"])["body"];
    auto & fns = mk_extern_c(pr,"extern_C")["body"];
    //auto & fns = mk_extern_c(pr,"extern_C")["body/.."];

    auto& fn = mk_func(pr,"fn",fn_declare).after(fns)["body"];

    // get the vars here first.
    const int64_t batch          = p->batchNum;
    const int64_t group          = p->group;
    const int64_t inChannel      = p->inChannel;
    const int64_t inHeight       = p->inHeight;
    const int64_t inWidth        = p->inWidth;
    const int64_t outChannel     = p->outChannel;
    const int64_t outHeight      = p->outHeight;
    const int64_t outWidth       = p->outWidth;
    const int64_t kernHeight     = p->kernHeight;
    const int64_t kernWidth      = p->kernWidth;
    const int64_t strideHeight   = p->strideHeight;
    const int64_t strideWidth    = p->strideWidth;
    const int64_t padHeight      = p->padHeight;
    const int64_t padWidth       = p->padWidth;
    const int64_t dilationHeight = p->dilationHeight;
    const int64_t dilationWidth  = p->dilationWidth;
    assert( outWidth > 0 );

    const int64_t inChannelGroup  = inChannel  / group;   // equal to pDataKernel->inChannel
    const int64_t outChannelGroup = outChannel / group;   // equal to pDataKernel->outChannel

    const int64_t inHW = inHeight * inWidth;
    const int64_t kernHW = kernHeight * kernWidth;
    const int64_t outHW = outHeight * outWidth;

    fn.DEF(batch).DEF(group).DEF(inChannel).DEF(inHeight).DEF(inWidth);
    fn.DEF(outChannel).DEF(outHeight).DEF(outWidth).DEF(kernHeight).DEF(kernWidth);
    fn.DEF(strideHeight).DEF(strideWidth).DEF(padHeight).DEF(padWidth).DEF(dilationHeight);
    fn.DEF(dilationWidth).DEF(inChannelGroup).DEF(outChannelGroup);
    fn.DEF(inHW).DEF(kernHW).DEF(outHW).DEF(kByMax);

    auto& fn_ptrs = fn["ptrs"];
    fn_ptrs>>"float const * restrict pIn  = pDataIn;"
        >>"float const * restrict pKernel = pDataKernel;"
        >>"float * restrict const pOut = pDataOut;"
        ;

    int64_t vlen=0; // we have not yet set vlen.
    pair<string,string> precalc_code = strCPreConvFwd1q(p,kByMax,vlen);
    // \post vlen changes if vlen register was clobbered
    auto& precalc = fn["precalc"];

    // just for reference (and KH_BE>1)
    // determine x and y output coords that don't need masking (all inputs good)
    KernLims kl = kernLims(p);
    int64_t yok_beg, yok_end;
    nomask_Height( kl, kernHeight, yok_beg, yok_end );
    cout<<" nomask_Height(kl,"<<kernHeight<<",...):"<<yok_beg<<","<<yok_end;
    int64_t xok_beg, xok_end;
    nomask_Width( kl, kernWidth, xok_beg, xok_end );
    cout<<" nomask_Width(kl,"<<kernWidth<<",...):"<<xok_beg<<","<<xok_end;

    bool maskH = true, maskW = true;
    if( yok_beg==0 && yok_end==outHeight ) maskH = false; // circuitous but
    if( xok_beg==0 && xok_end==outWidth  ) maskW = false; // safe way.
    cout<<" yok_beg,end="<<yok_beg<<","<<yok_end;
    cout<<" xok_beg,end="<<xok_beg<<","<<xok_end;
    cout<<" kh,kw="<<kernHeight<<","<<kernWidth;
    cout<<" ph,pw="<<padHeight<<"<"<<padWidth;
    cout<<" maskH,W="<<maskH<<","<<maskW;
    cout<<endl;
    // actual condition, as a formula, is much more complex
    // (it must also consider inputHeight wrt stride and dilation)
    //cout<<" k"<<kernHeight<<" p,s,d "<<padHeight<<", "<<strideHeight<<", "<<dilationHeight<<endl;
    //assert( maskH == (padHeight > 0 || (dilationHeight>1 && kernHeight>1)) );

    precalc CONST1(maskH) CONST1(maskW) CONST1(yok_beg) CONST1(yok_end) CONST1(xok_beg) CONST1(xok_end); //debug

#if PRE_KH_BEG_END      // precalc
#if KH_BE==0
    precalc
        >>"#define KH_BEG_END \\"
        >>"                  int64_t const kh_beg = kh_beg_end[0]; \\"
        >>"                  int64_t const kh_end = kh_beg_end[1]; \\"
        >>"                  kh_beg_end += 2"
        ;
#elif KH_BE==1
    precalc
        >>str_kh_be_static_array(p)
        >>"#define KH_BEG_END \\"
        >>"                  int64_t const kh_beg = kh_btab[y]; \\"
        >>"                  int64_t const kh_end = kh_etab[y]"
        ;
#else
    if( maskH ) precalc>>str_kh_be_static_array(p);
    precalc>>"#define KH_BEG_END \\";
    if( yok_beg==0 ) precalc>>"                  int64_t const kh_beg = 0; \\";
    else             precalc>>"                  int64_t const kh_beg = kh_btab[y]; \\";
    if( yok_end==outHeight ) precalc>>"                  int64_t const kh_end = kernHeight";
    else                     precalc>>"                  int64_t const kh_end = kh_etab[y]";
#endif

#else                   // run-time calc
    precalc
        // oops, removes terminating backslashes
        // >>"#define KH_BEG_END " VEJ_K_BEG_END(kh_beg,kh_tmp,kh_end, kernHeight, i,inHeight, dilationHeight)
        //
        >>"#define KH_BEG_END  \\\n"
        "    int64_t kh_end=0; \\\n"
        "    const int64_t kh_tmp = dilationHeight-i-1; \\\n"
        "    const int64_t kh_beg = (i>=0? 0: kh_tmp / dilationHeight); \\\n"
        "    if (i < inHeight){ \\\n"
        "        kh_end = (inHeight + kh_tmp) / dilationHeight; \\\n"
        "        if (kh_end >= kernHeight ) kh_end = kernHeight ; \\\n"
        "    }"
        ;
#endif
    precalc
        >>precalc_code.first
        ;

    int64_t const vl_x_init = outWidth /*- x0=0*/ < MVL ? outWidth /*- x0=0*/ : MVL;
    //int64_t const vl_x_init = ve_vlen_suggest( outWidth );
    //bool const x0_check_vl = outWidth%vl_x_init != 0;
    //std::string str_vl(x0_check_vl? "vl":"vl_x_init"); // vl_x_init is a #define
    auto& fn_vec_init =
    fn["vec_init"]
        >>"NO_SET_VLEN(vl_x_init);"
        >>"const __vr vzeros = _vel_vbrds_vsl(0.0f,vl_x_init);"
        // lower 32-bits are zero bits, so same as _ve_pvbrd_vs_i64(0UL)"
        >>"const __vr vrseq = _vel_vseq_vl(vl_x_init);"
        >>OSSFMT("int64_t vl = vl_x_init; // "<<vl_x_init)
        >>"NO_SET_VLEN(vl);"
        >>"float * restrict pOutx = pDataOut;"
        ;
    fn_vec_init.DEF(vl_x_init);

    vlen = vl_x_init;
    if(KH_BE<=1 || maskW)
        vrj_init(fn_vec_init);

    CBLOCK_SCOPE(loop_n,"for(int64_t n=0; n<batch; ++n)",pr,fn);
    CBLOCK_SCOPE(loop_g,"for(int64_t g=0; g<group; ++g)",pr,loop_n); // OK sub-tree
    loop_g
        >>"const int64_t outGroupOffset  = g * outChannelGroup * outHW;"
        >>"const int64_t inGroupOffset   = g * inChannelGroup * inHW;"
#if 1 || PRE_KRN_MAX < 1
        >>"const int64_t kernGroupOffset = g * outChannelGroup * inChannelGroup * kernHW;"
#endif
        >>"const float *pIn_0 = pIn + inGroupOffset + (n * inChannel + 0) * inHW;"
        ;
    // Here, we will follow direct_default2.c hand-unroll mechanics
    //CBLOCK_SCOPE(loop_k,"for(int64_t k=0 ; k<outChannelGroup; ++k)",pr,loop_g);
    // The above loop is hand-unrolled, in JIT fashion.
    // by checking end bits of outChannelGroup (a known constant)
    //   k ~ loop index [0,outChannelGroup)
    //   kBy ~ current unroll for k, in {1,2,4,8}
    //   kMax ~ k+kBy for single-time kBy, outChannelGroup for final unroll
    loop_g>>"int64_t k = 0;"; // loop_k index, outside loops for unrolling
    int64_t k=0;        // 0 .. outChannelGroup-1
    int64_t kBy=1;    // 1,2,4,... kByMax
    int64_t kMax = k;   // min of k+kBy or outChannelGroup
    if( k<outChannelGroup ){
        if( (outChannelGroup & kBy) != 0 ) kMax = k+kBy;
        if ( kBy == kByMax || kMax>outChannelGroup ) kMax = outChannelGroup;
        loop_g[string("kBy+")+hexdec(kBy)]
            <<"#if "<<((kMax == outChannelGroup || k<kMax)? "1": "0")
            <<" // loop_g : k in [0,outChannelGroup="<<asDec(outChannelGroup)
            <<") unroll k by "<<asDec(kBy)
            ;
        if( k<kMax ){
            CBLOCK_SCOPE(scope_kMax,"",pr,loop_g);
            scope_kMax.DEF(kBy).DEF(kMax);

            int64_t const max_unroll_outer = (KH_BE<=1 || maskW ? 12: 16);
            int64_t sofar = 1;
            //DEFINE_UNROLL(un_c , max_unroll_outer, sofar, inChannelGroup);
            int64_t un_c = min<int64_t>(16,inChannelGroup);
            int64_t un_s;
            if(KH_BE<=1 || maskW){
                // XXX no warn and INCORRECT RESULT result for mb1_ic1ih32oc1oh30kh3 XXX
                un_s = 0; // explicit 'nounroll'
            }else{
                DEFINE_UNROLL(un_ss , max_unroll_outer, sofar, kernWidth);
                un_s = un_ss;
            }

            DEFINE_UNROLL(un_r , max_unroll_outer, sofar, kernHeight);
            DEFINE_UNROLL(un_x0, max_unroll_outer, sofar, (outWidth+vl_x_init-1)/vl_x_init);
            DEFINE_UNROLL(un_y , -1, sofar, outHeight);
            DEFINE_UNROLL(un_k , -1, sofar, (kMax-k+kBy-1)/kBy);
            CBLOCK_FOR(loop_k,un_k,"for(; k<kMax; k+=kBy)",scope_kMax);
            loop_k
                >>"CHK(pOutx == pOut + outGroupOffset + (n*outChannel+k) * outHW);"
#if PRE_KRN_MAX < 1
                >>"const float * restrict pKern_gk = pKernel + kernGroupOffset"
                >>"                                + (k * inChannelGroup + 0/*c*/) * kernHW;"
#endif
                >>DEFINE_KH_BEG_END
                ;
            CBLOCK_FOR(loop_y,un_y,"for(int64_t y=0 ; y<outHeight; ++y)",loop_k);
            loop_y
                >>"const int64_t i = y * strideHeight - padHeight;"
                >>"KH_BEG_END;"
                >>OSSFMT("vl = vl_x_init; /*"<<vl_x_init<<"*/\n")
                >>"NO_SET_VLEN(vl);"
                ;
            CBLOCK_FOR(loop_x0,un_x0,"for(int64_t x0=0 ; x0<outWidth; x0+=vl_x_init)",loop_y);
            loop_x0
                >>"vl = outWidth - x0 < vl_x_init ? outWidth - x0: vl_x_init;"
                >>"NO_SET_VLEN(vl);"
                >>"__vr vrsum = vzeros;"
                ;
            if(KH_BE<=1 || maskW)
                vrj_induce(loop_x0); // vrj ~ vector of input x values

#if PRE_KRN_MAX >= 1
            precalc>>"/*pre*/ float const* restrict const krn_krsc_k1 = (float const*)(buffer + krn_krsc_k1_off);";
            loop_k[".."]>>"/*pre*/ float const* restrict krn_gk = krn_krsc_k1 + (nk1/group)*g;";
            loop_k["last"]>>"/*pre*/ krn_gk += kBy*inChannelGroup*kernHW;";
            //loop_k["ind"]>>"printf(" CSTR("krn_gk --> off %d\n") ", (int)(krn_gk - krn_krsc_k1));";
            loop_x0>>"/*pre*/ float const* restrict pKerValue = krn_gk + kh_beg*inChannelGroup*kernWidth; // skip some";
#endif
            CBLOCK_FOR(loop_r,un_r,"for (int64_t r = kh_beg; r < kh_end; ++r)",loop_x0);
            CBLOCK_FOR(loop_s,un_s,"for (int64_t s = 0; s < kernWidth; ++s)",loop_r);
            if(KH_BE<=1 || maskW){
                loop_s[".."]>>"__vr vrw = vrj; // input row pixels";
                loop_s>>"__vm256 vm23 = " VEL_VFMK_mvs_0_TO(vrw,inWidth,vl) ";";
                loop_s["last"]>>"vrw = _vel_vaddsl_vsvl(dilationWidth, vrw,vl);";
            }
            CBLOCK_FOR(loop_c,un_c,"for (int64_t c = 0; c < inChannelGroup; ++c)",loop_s);
            loop_c
                >>"const float *pIn = pIn_0 + c*inHW + (i+r*dilationHeight)*inWidth"
                >>"                 + x0*strideWidth-padWidth + s*dilationWidth;"
                >>"__vr vrin = _vel_vldu_vssl(4*strideWidth,pIn,vl);"
#if PRE_KRN_MAX < 1
                >>"const float *pKerValue = pKern_gk + c*kernHW + r*kernWidth +s;"
                >>"vrsum = _vel_vfmads_vvsvmvl(vrsum, *pKerValue, vrin, vm23, vrsum,vl);"
#else
                ;
            loop_c.define("DOSUM(VRSUM,PAIR)",OSSFMT("VRSUM = "<<( (KH_BE<=1 || maskW)
                        ? "_vel_vfmads_vvsvmvl(vrsum, *pKerValue, vrin, vm23, vrsum,vl)"
                        : "_vel_vfmads_vvsvl  (vrsum, *pKerValue, vrin,vl)")))
                >>"DOSUM(vrsum, *pKerValue);"
                >>"#undef DOSUM"
                >>"++pKerValue;"
#endif
                ;
            loop_x0["induce+write"]
                >>"_vel_vstu_vssl(vrsum, 4, pOutx,vl);"
                >>"pOutx += vl; // visible speedup cf. outIndex+=vl"
                >>"//"<<CSTR(printf(" k %ld vl %-3ld outIndex=%ld\n",(long)k,(long)vl,(long)(pOutx-pOut));)
                ;
            //k = kMax;
            // "simulate" effect of above JIT loop_k on 'k' (illustrative)
            for(; k<kMax; k+=kBy ) /*no-op*/;
            assert( k == kMax );
            scope_kMax["~kMax"]
                >>"CHK(pOutx == pOut + outGroupOffset + (n * outChannel + kMax) * outHW);"
                FREE1(kBy)
                FREE1(kMax)
                ;
        }
        loop_g[string("kBy-")+hexdec(kBy)]
            >>"#endif // unroll by "+hexdec(kBy)<<", exit w/ k="<<hexdec(k)
            <<" of outChannelGroup="<<hexdec(outChannelGroup)
#if PRE_KRN_MAX >= 2
            //>>"int const krnPairStart = "<<asDec(k)<<";"
            // krn_gk_0 goes up by inChannelGroup*kernHW for every increment in k.
            //>>"krn_gk_0 = krn_gkrsc_0;" + "<<asDec(k)<<" * inChannelGroup * kernHW;"

#endif
            ;
        if(kByMax==1){ assert( k>=outChannelGroup ); }
    }
    if( k<outChannelGroup ){
        kBy=2;
        if( (outChannelGroup & kBy) != 0 ) kMax = k+kBy;
        if ( kBy == kByMax || kMax>outChannelGroup ) kMax = outChannelGroup;
        loop_g[string("kBy+")+hexdec(kBy)]
            <<"#if "<<((kMax == outChannelGroup || k<kMax)? "1": "0")
            <<" // loop_k [0,outChannelGroup) unroll k by "<<hexdec(kBy)
            ;
        if( k<kMax ){
            CBLOCK_SCOPE(scope_kMax,"",pr,loop_g);
            scope_kMax.DEF(kBy).DEF(kMax)
                >>"//"<<CSTR(if(n+g==0) printf("for k = %ld, %ld, %ld...\n",(long)k,(long)kMax,(long)kBy);)
                ;
            CBLOCK_SCOPE(loop_k,"for(; k<kMax; k+=kBy)",pr,scope_kMax);
            loop_k
                >>"CHK(pOutx == pOut + outGroupOffset + (n*outChannel+k) * outHW);"
                >>"float* pOutx1 = pOutx + outHW;"
#if PRE_KRN_MAX < 2
                >>"const float * restrict pKern_gk = pKernel + kernGroupOffset"
                >>"                                + (k * inChannelGroup + 0/*c*/) * kernHW;"
                //>>CSTR(printf("g%lluk%llu pKern_gk-pKernel=%llu kGO=%lld",(llu)g,(llu)k,(llu)(pKern_gk-pKernel),(llu)kernGroupOffset);)
#if PRE_KRN_MAX >= 2
                //>>"printf("<<CSTR(" krn_gk-krn_krsc_k2=%llu",(llu)(krn_gk-krn_krsc_k2))<<");"
#endif
                //>>CSTR(printf("\n");)
#endif
                >>DEFINE_KH_BEG_END
                ;
            CBLOCK_SCOPE(loop_y,"for(int64_t y=0; y<outHeight; ++y)",pr,loop_k);
            loop_y
                >>"const int64_t i = y * strideHeight - padHeight;"
                >>"KH_BEG_END;"
                >>"vl = vl_x_init;"
                >>"NO_SET_VLEN(vl);"
                ;
            // slightly faster looping for x0... (detect vl <= 0 instead of x0 too high)
            CBLOCK_SCOPE(loop_x0,"for(int64_t x0=0; /*x0<outWidth*/; /*x0+=VLEN*/)",pr,loop_y);
            if(KH_BE<=1 || maskW) vrj_induce(loop_x0); // vrj ~ vector of input x values
            loop_x0
                >>"__vr vrsum01 = vzeros;"
                ;
#if PRE_KRN_MAX >= 2
            precalc>>"union PairedFloat const* restrict const krn_krsc_k2 = (union PairedFloat const*)"
                >>"       (buffer + krn_krsc_k2_off);";
            loop_k[".."]>>"union PairedFloat const* restrict krn_gk = krn_krsc_k2 + g*(nk2/group);";
            // these are PairedFloat, so use kBy/2
            loop_k["ind"]>>"krn_gk += (kBy/2U)*inChannelGroup*kernHW;";
            loop_x0>>"union PairedFloat const* restrict pKer2 = krn_gk"
               >>"        + kh_beg*kernWidth*inChannelGroup*(kBy/2); // skip some";
#endif
            // using 1 vmP, any unroll r*s > 7 will probably fail, I think
            int const max_unroll_outer = 7; int64_t sofar = 1;
            DEFINE_UNROLL(un_s , max_unroll_outer, sofar, kernWidth); // usually ok, but VM-VM copy for mb1g2_ic62ih32oc62oh32kh3
            DEFINE_UNROLL(un_r , max_unroll_outer, sofar, kernHeight);
            DEFINE_UNROLL(un_x0, max_unroll_outer, sofar, (outWidth+vl_x_init-1)/vl_x_init);
            DEFINE_UNROLL(un_y , max_unroll_outer, sofar, outHeight);
            DEFINE_UNROLL(un_k , max_unroll_outer, sofar, (kMax-k+kBy-1)/kBy);
            CBLOCK_FOR(loop_r,un_r,"for (int64_t r = kh_beg; r < kh_end; ++r)",loop_x0);
            CBLOCK_FOR(loop_s,un_s,"for (int64_t s = 0; s < kernWidth; ++s)",loop_r);
            if(KH_BE<=1 || maskW){
                loop_s[".."]>>"__vr vrw = vrj; // input row pixels";
                loop_s>>"__vm256 vm23 = " VEL_VFMK_mvs_0_TO(vrw,inWidth,vl) ";"
                    >>VEL_DECL_VM512( vmP, vm23,vm23 ,vl);
                loop_s["last"]>>"vrw = _vel_vaddsl_vsvl(dilationWidth, vrw,vl);";
            }
            //loop_s>>""<<"if(1){"
            //    >>"     int64_t const rs = r*kernWidth+s;"
            //    >>"    "<<CSTR(printf(" 1q-kBy2 yx0=%ld %ld gk=%ld %ld vl %-3ld rs=%ld outIndex0=%ld",(long)y,(long)x0,(long)g,(long)k,(long)vl,(long)rs,(long)(pOutx-pOut));)
            //    >>"    "<<CSTR(printf(" pKern %ld",(long)(pKern_gk+rs-pKernel));)
#if PRE_KRN_//MAX >= 2
            //    >>"    "<<CSTR(printf(" pKer2 %ld",(long)(pKer2-krn_krsc_k2));)
#endif
            //    >>"    "<<CSTR(printf("\n");)
            //    >>"}";
            CBLOCK_SCOPE(loop_c,"for (int64_t c = 0; c < inChannelGroup; ++c)",pr,loop_s);
            loop_c
                >>"const float *pIn = pIn_0 + c*inHW + (i+r*dilationHeight)*inWidth"
                >>"                 + x0*strideWidth-padWidth + s*dilationWidth;"
                >>"__vr vrin = _vel_vldu_vssl(4*strideWidth,pIn,vl);"
#if PRE_KRN_MAX < 2
                >>"const float *pKerValue = pKern_gk + c*kernHW + r*kernWidth +s;"
                >>"const uint64_t kerValue01 = _vel_pack_f32p(pKerValue,"
                >>"                                          pKerValue + inChannelGroup*kernHW);"
                >>"__vr vrinP = _vel_vshf_vvvsl(vrin, vrin, VE_VSHUFFLE_YUZU,vl);"
                >>"vrsum01 = _vel_pvfmad_vvsvMvl(vrsum01, kerValue01, vrinP, vmP, vrsum01,vl);"
#else
                ;
            loop_c.define("DOSUM(VRSUM,PAIR)",OSSFMT("VRSUM = "<<( (KH_BE<=1 || maskW)
                        ? "_vel_pvfmad_vvsvMvl(VRSUM,PAIR,vrinP,vmP,VRSUM,vl)"
                        : "_vel_pvfmad_vvsvl(VRSUM,PAIR,vrinP,vl)")))
                >>"__vr vrinP = _vel_vshf_vvvsl(vrin, vrin, VE_VSHUFFLE_YUZU,vl);"
                >>"DOSUM(vrsum01, pKer2[0].pair);"
                >>"pKer2+=kBy/2U;"
#endif
                ;
            loop_x0["induce+write"]
                >>"_vel_vstu_vssl(vrsum01, 4, pOutx ,vl);"
                >>"_vel_vstl_vssl(vrsum01, 4, pOutx1,vl);"
                >>"pOutx  += vl;"
                >>"pOutx1 += vl;"
                //for_rs>>"printf(\" precalc k2Out gk=%d %d rs=%d %d, cc=%d, offset = %d\\n\",(int)g,(int)k,(int)rs,(int)0,(int)(k2Out - krn_gkrsc_0));";
                >>""
                >>"x0 += vl_x_init;"
                >>"vl = outWidth - x0;"
                >>"if( vl <= 0 ) break;"
                >>"vl = vl < vl_x_init? vl: vl_x_init;"
                >>"NO_SET_VLEN(vl);"
                ;
            loop_k["bump pOutx"]
                >>"pOutx += /*kBy-1*/ 1 * outHW; // inner increment is outHW, outer wants kBy*outHW"
                ;
            //k = kMax; // could be too naive?
            // "simulate" effect of above JIT loop_k on 'k' (illustrative)
            for(; k<kMax; k+=kBy ) /*no-op*/;
            assert( k == kMax );
            //scope_kMax["~kMax"]>>"CHK(pOutx == pOut + outGroupOffset + (n*outChannel+kMax)*outHW);";
        }
        loop_g[string("kBy-")+hexdec(kBy)]
            >>"#endif // unroll by "+hexdec(kBy)<<", exit w/ k="<<hexdec(k)
            <<" of outChannelGroup="<<hexdec(outChannelGroup)
            ;
        if(kByMax==2){ assert( k>=outChannelGroup ); }
    }
    if( k<outChannelGroup ){
        kBy=4;
        if( (outChannelGroup & kBy) != 0 ) kMax = k+kBy;
        if ( kBy == kByMax || kMax>outChannelGroup ) kMax = outChannelGroup;
        loop_g[string("kBy+")+hexdec(kBy)]
            <<"#if "<<((kMax == outChannelGroup || k<kMax)? "1": "0")
            <<" // loop_k [0,outChannelGroup) unroll k by "<<hexdec(kBy)
            ;
        if( k<kMax ){
            CBLOCK_SCOPE(scope_kMax,"",pr,loop_g);
            scope_kMax.DEF(kBy).DEF(kMax)
                >>"//"<<CSTR(if(n+g==0) printf("for k = %ld, %ld, %ld...\n",(long)k,(long)kMax,(long)kBy);)
                ;
            CBLOCK_SCOPE(loop_k,"for(; k<kMax; k+=kBy)",pr,scope_kMax);
            loop_k
                >>"float* pOutx1 = pOutx + outHW;"
                >>"float* pOutx2 = pOutx + 2*outHW;"
                >>"float* pOutx3 = pOutx + 3*outHW;"
                >>DEFINE_KH_BEG_END
                ;
            CBLOCK_SCOPE(loop_y,"for(int64_t y=0; y<outHeight; ++y)",pr,loop_k);
            loop_y
                >>"const int64_t i = y * strideHeight - padHeight;"
                >>"KH_BEG_END;"
                >>"vl = vl_x_init;"
                >>"NO_SET_VLEN(vl);"
                ;
            // slightly faster looping for x0... (detect vl <= 0 instead of x0 too high)
            CBLOCK_SCOPE(loop_x0,"for(int64_t x0=0; /*x0<outWidth*/; /*x0+=VLEN*/)",pr,loop_y);
            if(KH_BE<=1 || maskW) vrj_induce(loop_x0); // vrj ~ vector of input x values
            loop_x0
                >>"__vr vrsum01 = vzeros;"
                >>"__vr vrsum23 = vzeros;"
                ;
#if PRE_KRN_MAX < 4
            loop_k
                >>"const float * restrict pKern_gk = pKernel + kernGroupOffset"
                >>"                                + (k * inChannelGroup + 0/*c*/) * kernHW;"
                ;
#else
            precalc>>"union PairedFloat const* restrict const krn_krsc_k4 ="
                >>"         (union PairedFloat const*)(buffer + krn_krsc_k4_off);";
            loop_k[".."]>>"union PairedFloat const* restrict krn_gk = krn_krsc_k4 + g*(nk4/group);";
            loop_k["last"]>>"krn_gk += (kBy/2U)*inChannelGroup*kernHW;";
            loop_x0>>"union PairedFloat const* restrict pKer2 = krn_gk"
                >>"        + kh_beg*kernWidth*inChannelGroup*(kBy/2); // skip some";
#endif
            //CBLOCK_SCOPE(loop_r,"for (int64_t r = kh_beg; r < kh_end; ++r)",pr,loop_x0);
            //CBLOCK_SCOPE(loop_s,"for (int64_t s = 0; s < kernWidth; ++s)",pr,loop_r);
            // Here we still have 1 vmP, so once again we have unroll limits:
            int const max_unroll_outer = 7;
            int const un_s = (kernWidth>max_unroll_outer? max_unroll_outer: kernWidth);
            int const un_r = max_unroll_outer/un_s;
            CBLOCK_SCOPE(loop_r,OSSFMT("#pragma unroll("<<un_r<<")\n"
                        "for (int64_t r = kh_beg; r < kh_end; ++r)"), pr,loop_x0);
            CBLOCK_SCOPE(loop_s,OSSFMT("#pragma unroll("<<un_s<<")\n"
                        "for (int64_t s = 0; s < kernWidth; ++s)"), pr,loop_r);
            if(KH_BE<=1 || maskW){
                loop_s[".."]>>"__vr vrw = vrj; // input row pixels";
                loop_s>>"__vm256 vm23 = " VEL_VFMK_mvs_0_TO(vrw,inWidth,vl) ";"
                    >>VEL_DECL_VM512( vmP, vm23,vm23 ,vl);
                loop_s["last"]>>"vrw = _vel_vaddsl_vsvl(dilationWidth, vrw,vl);";
            }
            CBLOCK_SCOPE(loop_c,"for (int64_t c = 0; c < inChannelGroup; ++c)",pr,loop_s);
            loop_c
                >>"const float *pIn = pIn_0 + c*inHW + (i+r*dilationHeight)*inWidth"
                >>"                 + x0*strideWidth-padWidth + s*dilationWidth;"
                >>"__vr vrin = _vel_vldu_vssl(4*strideWidth,pIn,vl);"
#if PRE_KRN_MAX < 4
                >>"const float *pKerValue = pKern_gk + c*kernHW + r*kernWidth +s;"
                >>"const uint64_t kerValue01 = _vel_pack_f32p("
                >>"    pKerValue,"
                >>"    pKerValue + inChannelGroup*kernHW);"
                >>"__vr vrinP = _vel_vshf_vvvsl(vrin, vrin, VE_VSHUFFLE_YUZU,vl);"
                >>"const uint64_t kerValue23 = _vel_pack_f32p("
                >>"    pKerValue + 2 * inChannelGroup * kernHW,"
                >>"    pKerValue + 3 * inChannelGroup * kernHW);"
                >>"vrsum01 = _vel_pvfmad_vvsvMvl(vrsum01, kerValue01, vrinP, vmP, vrsum01,vl);"
                >>"vrsum23 = _vel_pvfmad_vvsvMvl(vrsum23, kerValue23, vrinP, vmP, vrsum23,vl);"
#else
                ;
            loop_c.define("DOSUM(VRSUM,PAIR)",OSSFMT("VRSUM = "<<( (KH_BE<=1 || maskW)
                        ? "_vel_pvfmad_vvsvMvl(VRSUM,PAIR,vrinP,vmP,VRSUM,vl)"
                        : "_vel_pvfmad_vvsvl(VRSUM,PAIR,vrinP,vl)")))
                >>"__vr vrinP = _vel_vshf_vvvsl(vrin, vrin, VE_VSHUFFLE_YUZU,vl);"
                >>"DOSUM(vrsum01, pKer2[0].pair);"
                >>"DOSUM(vrsum23, pKer2[1].pair);"
                >>"pKer2+=kBy/2U;"
#endif
                ;
            loop_x0["induce+write"]
                >>"_vel_vstu_vssl(vrsum01, 4, pOutx ,vl);"
                >>"_vel_vstl_vssl(vrsum01, 4, pOutx1,vl);"
                >>"_vel_vstu_vssl(vrsum23, 4, pOutx2,vl);"
                >>"_vel_vstl_vssl(vrsum23, 4, pOutx3,vl);"
                >>"pOutx  += vl;"
                >>"pOutx1 += vl;"
                >>"pOutx2 += vl;"
                >>"pOutx3 += vl;"
                >>""
                >>"x0 += vl_x_init;"
                >>"vl = outWidth - x0;"
                >>"if( vl <= 0 ) break;"
                >>"vl = vl < vl_x_init? vl: vl_x_init;"
                >>"NO_SET_VLEN(vl);"
                ;
            loop_k["bump pOutx"]
                >>"pOutx += (kBy-1) * outHW; // inner increment is outHW, outer wants kBy*outHW"
                ;
            //k = kMax; // could be too naive?
            // "simulate" effect of above JIT loop_k on 'k' (illustrative)
            for(; k<kMax; k+=kBy ) /*no-op*/;
            assert( k == kMax );
            //scope_kMax["~kMax"]
            //    >>"CHK(pOutx == pOut + outGroupOffset + (n * outChannel + kMax) * outHW);";
        }
        loop_g[string("kBy-")+hexdec(kBy)]
            >>"#endif // unroll by "+hexdec(kBy)<<", exit w/ k="<<hexdec(k)
            <<" of outChannelGroup="<<hexdec(outChannelGroup)
            ;
        if(kByMax==4){ assert( k>=outChannelGroup ); }
    }
    if( k<outChannelGroup ){
        kBy=8;
        assert( kByMax == 8 );
        kMax = outChannelGroup;
        loop_g[string("kBy+")+hexdec(kBy)]
            <<"#if "<<(k<kMax? "1": "0")
            <<" // loop_k [0,outChannelGroup) unroll k by "<<asDec(kBy)
            ;
        if( k<kMax ){
            CBLOCK_SCOPE(scope_kMax,"",pr,loop_g);
            scope_kMax.DEF(kBy).DEF(kMax)
                >>"//"<<CSTR(if(n+g==0) printf("for k = %ld, %ld, %ld...\n",(long)k,(long)kMax,(long)kBy);)
                ;
            CBLOCK_SCOPE(loop_k,"for(; k<kMax; k+=kBy)",pr,scope_kMax);
            loop_k
                >>"float* pOutx1 = pOutx + outHW;"
                >>"float* pOutx2 = pOutx + 2*outHW;"
                >>"float* pOutx3 = pOutx + 3*outHW;"
                >>"float* pOutx4 = pOutx + 4*outHW;"
                >>"float* pOutx5 = pOutx + 5*outHW;"
                >>"float* pOutx6 = pOutx + 6*outHW;"
                >>"float* pOutx7 = pOutx + 7*outHW;"
                >>DEFINE_KH_BEG_END
                ;
            CBLOCK_SCOPE(loop_y,"for(int64_t y=0; y<outHeight; ++y)",pr,loop_k);
            loop_y
                >>"const int64_t i = y * strideHeight - padHeight;"
                >>"KH_BEG_END;"
                >>"vl = vl_x_init;"
                >>"NO_SET_VLEN(vl);"
                ;
            // slightly faster looping for x0... (detect vl <= 0 instead of x0 too high)
            CBLOCK_SCOPE(loop_x0,"for(int64_t x0=0; /*x0<outWidth*/; /*x0+=VLEN*/)",pr,loop_y);
            if(KH_BE<=1 || maskW)
                vrj_induce(loop_x0); // vrj ~ vector of input x values
            loop_x0
                >>"__vr vrsum01 = vzeros;"
                >>"__vr vrsum23 = vzeros;"
                >>"__vr vrsum45 = vzeros;"
                >>"__vr vrsum67 = vzeros;"
                ;
#if PRE_KRN_MAX < 8
            loop_k
                >>"const float * restrict pKern_gk = pKernel + kernGroupOffset"
                >>"                                + (k * inChannelGroup + 0/*c*/) * kernHW;"
                ;
#else
            precalc>>"union PairedFloat const* restrict const krn_krsc_k8 ="
                >>"        (union PairedFloat const*)(buffer + krn_krsc_k8_off);";
            loop_k[".."]>>"union PairedFloat const* restrict krn_gk = krn_krsc_k8 + g*(nk8/group);";
            loop_k["last"]>>"krn_gk += (kBy/2U)*inChannelGroup*kernHW;";
            loop_x0>>"union PairedFloat const* restrict pKer2 = krn_gk"
                >>"        + kh_beg*kernWidth*inChannelGroup*(kBy/2); // skip some";
#endif
            //CBLOCK_SCOPE(loop_r,"for (int64_t r = kh_beg; r < kh_end; ++r)",pr,loop_x0);
            //CBLOCK_SCOPE(loop_s,"for (int64_t s = 0; s < kernWidth; ++s)",pr,loop_r);
            // Here we MIGHT have 1 vmP, so once again we have unroll limits:
            int const max_unroll_outer = (KH_BE<=1 || maskW ? 7: 8); //maybe change the '8'?
            int const un_s = (kernWidth>max_unroll_outer? max_unroll_outer: kernWidth);
            int const un_r = max_unroll_outer/un_s;
            CBLOCK_SCOPE(loop_r,OSSFMT("#pragma unroll("<<un_r<<")\n"
                        "for (int64_t r = kh_beg; r < kh_end; ++r)"), pr,loop_x0);
            CBLOCK_SCOPE(loop_s,OSSFMT("#pragma unroll("<<un_s<<")\n"
                        "for (int64_t s = 0; s < kernWidth; ++s)"), pr,loop_r);
            if(KH_BE<=1 || maskW){
                loop_s[".."]>>"__vr vrw = vrj; // input row pixels";
                loop_s>>"__vm256 vm23 = " VEL_VFMK_mvs_0_TO(vrw,inWidth,vl) ";"
                    >>VEL_DECL_VM512( vmP, vm23,vm23 ,vl);
                loop_s["last"]>>"vrw = _vel_vaddsl_vsvl(dilationWidth, vrw,vl);";
            }
            CBLOCK_SCOPE(loop_c,"for (int64_t c = 0; c < inChannelGroup; ++c)",pr,loop_s);
            loop_c
                >>"const float *pIn = pIn_0 + c*inHW + (i+r*dilationHeight)*inWidth"
                >>"                 + x0*strideWidth-padWidth + s*dilationWidth;"
                >>"__vr vrin = _vel_vldu_vssl(4*strideWidth,pIn,vl);"
                >>"__vr vrinP = _vel_vshf_vvvsl(vrin, vrin, VE_VSHUFFLE_YUZU,vl);"
#if PRE_KRN_MAX<8
                >>"const float *pKerValue = pKern_gk + c*kernHW + r*kernWidth +s;"
                >>"const uint64_t kerValue01 = _vel_pack_f32p("
                >>"    pKerValue,"
                >>"    pKerValue + 1 * inChannelGroup*kernHW);"
                >>"const uint64_t kerValue23 = _vel_pack_f32p("
                >>"    pKerValue + 2 * inChannelGroup * kernHW,"
                >>"    pKerValue + 3 * inChannelGroup * kernHW);"
                >>"vrsum01 = _vel_pvfmad_vvsvMvl(vrsum01, kerValue01, vrinP, vmP, vrsum01,vl);"
                >>"vrsum23 = _vel_pvfmad_vvsvMvl(vrsum23, kerValue23, vrinP, vmP, vrsum23,vl);"
                >>"const uint64_t kerValue45 = _vel_pack_f32p("
                >>"    pKerValue + 4 * inChannelGroup * kernHW,"
                >>"    pKerValue + 5 * inChannelGroup * kernHW);"
                >>"const uint64_t kerValue67 = _vel_pack_f32p("
                >>"    pKerValue + 6 * inChannelGroup * kernHW,"
                >>"    pKerValue + 7 * inChannelGroup * kernHW);"
                >>"vrsum45 = _vel_pvfmad_vvsvMvl(vrsum45, kerValue45, vrinP, vmP, vrsum45,vl);"
                >>"vrsum67 = _vel_pvfmad_vvsvMvl(vrsum67, kerValue67, vrinP, vmP, vrsum67,vl);"
#else
                ;
            loop_c.define("DOSUM(VRSUM,PAIR)",OSSFMT("VRSUM = "<<( (KH_BE<=1 || maskW)
                        ? "_vel_pvfmad_vvsvMvl(VRSUM,PAIR,vrinP,vmP,VRSUM,vl)"
                        : "_vel_pvfmad_vvsvl(VRSUM,PAIR,vrinP,vl)")))
                >>"DOSUM(vrsum01, pKer2[0].pair);"
                >>"DOSUM(vrsum23, pKer2[1].pair);"
                >>"DOSUM(vrsum45, pKer2[2].pair);"
                >>"DOSUM(vrsum67, pKer2[3].pair);"
                >>"pKer2+=kBy/2;"
#endif
                ;
            loop_x0["induce+write"]
                >>"_vel_vstu_vssl(vrsum01, 4, pOutx ,vl);"
                >>"_vel_vstl_vssl(vrsum01, 4, pOutx1,vl);"
                >>"_vel_vstu_vssl(vrsum23, 4, pOutx2,vl);"
                >>"_vel_vstl_vssl(vrsum23, 4, pOutx3,vl);"
                >>"pOutx  += vl;"
                >>"pOutx1 += vl;"
                >>"pOutx2 += vl;"
                >>"pOutx3 += vl;"
                >>"_vel_vstu_vssl(vrsum45, 4, pOutx4,vl);"
                >>"_vel_vstl_vssl(vrsum45, 4, pOutx5,vl);"
                >>"_vel_vstu_vssl(vrsum67, 4, pOutx6,vl);"
                >>"_vel_vstl_vssl(vrsum67, 4, pOutx7,vl);"
                >>"pOutx4 += vl;"
                >>"pOutx5 += vl;"
                >>"pOutx6 += vl;"
                >>"pOutx7 += vl;"
                >>" "
                >>"x0 += vl_x_init;"
                >>"vl = outWidth - x0;"
                >>"if( vl <= 0 ) break;"
                >>"vl = vl < vl_x_init? vl: vl_x_init;"
                >>"NO_SET_VLEN(vl);"
                ;
            loop_k["bump pOutx"]
                >>"pOutx += (kBy-1) * outHW; // inner increment is outHW, outer wants kBy*outHW"
                ;
            //k = kMax; // could be too naive?
            // "simulate" effect of above JIT loop_k on 'k' (illustrative)
            for(; k<kMax; k+=kBy ) /*no-op*/;
            assert( k == kMax );
            //scope_kMax["~kMax"]>>"CHK(pOutx == pOut + outGroupOffset + (n*outChannel+kMax)*outHW);";
        }
        loop_g[string("kBy-")+hexdec(kBy)]
            >>"#endif // unroll by "+hexdec(kBy)<<", exit w/ k="<<hexdec(k)
            <<" of outChannelGroup="<<hexdec(outChannelGroup)
            ;
        if(kByMax==8){ assert( k>=outChannelGroup ); }
    }
    fn["undef"]
        >>precalc_code.second
        ;
    fn["exit"]>>"return VEDNN_SUCCESS;"
        ;

#if 0
    //
    //  To do iteration, we NEED vednnx iterator API.
    //  If we call existing functions, dlopen REQUIRES
    //     - shared libvednnx...so
    //     - or whole-archive libvednnx
    //  to resolve symbols
    //
    //  Currently, shared library is foobar, so we must whole-archive vednnx
    //  NEW: it works with ncc 2+ (glibc variant, with C files)
    //
    //  DO THIS LATER XXX  -- single-use approach is fine for now.
    //
    std::string fn_ok_declare = "\n\nvednnError_t "+df.basename+"_ok(\n    "
        +multiReplace(",",",\n    ", CSTR(VEDNN_PARAMS_CONV_FORWARD))
        +"\n)";
    df.syms.push_back(SymbolDecl(df.basename+"_ok",
                "vednn ConvolutionForward ok (param check) "+ paramString(p),
                fn_ok_declare));
    auto& fn_ok = mk_func(pr,"fn",fn_ok_declare).after(fns)["body"];
    fn_ok>>"return vednnConvolutionForward_direct_default_ok(\n    "
        CSTR(VEDNN_PARAMS_CONV_FORWARD_LIST) " );";
#endif


    pr["end-of-file"]>>"// vim: ts=4 sw=4 et cindent cino=^=l0,\\:.5s,=-.5s,N-s,g.5s,b1 cinkeys=0{,0},0),\\:,0#,!^F,o,O,e,0=break";
    pr.v = 0; // set Cuint (root) back to non-verbose

    if(verbose){ // dump to cout (debug)
        // Note: 'write' currently has side-effect of "emptying" the tree. Subject to change!
        //cout<<string(80,'-')<<endl;
        //pr.write(cout);
        //cout<<string(80,'-')<<endl;
        //pr.dump(cout);
        //cout<<endl;
        if(verbose>=1)
            cout<<string(80,'-')<<pr.str() <<string(80,'-')<<endl;
        if(verbose>=2)
            cout<<string(80,'-')<<pr.tree()<<string(80,'-')<<endl;
    }
    df.code = pr.str();
    return df;
}
// vim: ts=4 sw=4 et cindent cino=^=l0,\:.5s,=-.5s,N-s,g.5s,b1 cinkeys=0{,0},0),\:,0#,!^F,o,O,e,0=break
