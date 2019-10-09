/** \file
 * - first vel conversion
 * - cleanup of dev code in \ref cjitConvFwd1q.cpp */
#ifndef VEL_BUG
/** 1 means use extra ve_lvl as workaround for clang bug */
#define VEL_BUG 0
#endif

#include "cjitConv.hpp"
#include "ve_cvecops.hpp"
#include "dllFileAux.hpp"   // strings for declarations, paramString
#include <string>
#include <iostream>
#include <iomanip>
using namespace std;
using namespace cprog;

#define CONST1(var) >>("#define " #var " "+asDec(var))
#define FREE1(var) >>("#undef " #var)
#define DEF(VAR) def(#VAR, VAR)

#define JIT_MALLOC_THRESHOLD_K 16

#ifndef KBYMAX
/** kByMax is chosen from {1,2,4,8} */
#define KBYMAX 8
// KBYMAX   1   2   4   8
// Gflops
#endif

#define PRE_KRN_MAX KBYMAX

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
static void kernLims( const int64_t outSz, const int64_t strideSz, const int64_t padSz,
    const int64_t dilationSz, const int64_t inSz, const int64_t kernSz,
    std::vector<int64_t>& kbeg, std::vector<int64_t>& kend)
{
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
#if 0 // ncc bug (cond in loop? fixable with #pragma?)
/** Find y range [yBeg,yEnd) where all kernel height values are ok (input height within range).
 * If no such range, return yBeg >= yEnd.
 * (somewhere I have the equation implementation for this). */
static void nomask_Height( KernLims const& kl, int64_t kernHeight, int64_t &yBeg, int64_t &yEnd ){
  yBeg = -1;
  yEnd = -1;
  assert( kl.kh_b.size() == kl.kh_e.size() ); // == outHeight
  assert( kl.kh_b.size() > 0 );
  size_t const odim = kl.kh_b.size();
  for(size_t y=0; y<odim; ++y){
    if( yBeg < 0 && kl.kh_b[y]      ==          0 ) yBeg = y; // first normal value
    if( yEnd < 0 && kl.kh_e[odim-y] == kernHeight ) yEnd = odim-y; // last normal value
  }
  if(yBeg < 0) yBeg = kl.kh_b.size();
  if(yEnd < 0) yEnd = yBeg;
  else ++yEnd;
}
static void nomask_Width( KernLims const& kl, int64_t kernWidth, int64_t &xBeg, int64_t &xEnd ){
  xBeg = -1;
  xEnd = -1;
  assert( kl.kw_b.size() == kl.kw_e.size() ); // == outWidth
  assert( kl.kw_b.size() > 0 );
  size_t const odim = kl.kw_b.size();
  for(size_t x=0; x<odim; ++x){
    if( xBeg < 0 && kl.kw_b[x]      == 0         ) xBeg = x;
    if( xEnd < 0 && kl.kw_e[odim-x] == kernWidth ) xEnd = odim-x;
  }
  if(xBeg < 0) xBeg = kl.kw_b.size();
  if(xEnd < 0) xEnd = xBeg;
  else ++xEnd;
}
#else
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
#endif

union PairedFloat {
  float f[2];                     // packed-pair floats
  uint64_t pair;                  // forces alignment 8
};

#if 0
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
#endif

static std::pair<string,string> strCPreConvFwd6( struct param const* const p, int const kByMax, int64_t& vlen ){
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
            if(rs==0 && c==0) printf("kBy=%d g=%d k=%d nk=%d\n",(int)kBy,(int)g,(int)k,(int)nk1);
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
      for( ; k<kMax; k+=kBy){ // loop_k
        for (int64_t rs = 0; rs < kernHW; ++rs){ // loop_r, loop_s (krn data packed in rs)
          for (int64_t c = 0; c < inChannelGroup; ++c){ // loop_c
            nk2+=(kBy/2); // these are actually pairs of kernel values
          }
        }
      }
      kMax2 = k;
    }
    if( k<outChannelGroup ){
      kBy=4;
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
    ){ // then we have no work to do
    cout<<" (CpreConvFwd6 not needed)";
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
    int64_t bufsz_bytes = 
      (nk2+nk4+nk8) * sizeof(union PairedFloat)
      + nk1 * sizeof(float)
      ;
    int64_t krn_krsc_k2_off = 0U;
    int64_t krn_krsc_k4_off = krn_krsc_k2_off + nk2 * sizeof(union PairedFloat);
    int64_t krn_krsc_k8_off = krn_krsc_k4_off + nk4 * sizeof(union PairedFloat);
    int64_t krn_krsc_k1_off = krn_krsc_k2_off  + (nk2+nk4+nk8) * sizeof(union PairedFloat);
    auto& tmp_beg = tmp["beg"];
    auto& tmp_last = tmp["last"];

    if(bufsz_bytes > JIT_MALLOC_THRESHOLD_K*1024L ){
      cout<<"malloc buf64[(bufsz_bytes + 7) / 8] = buf64["<<(bufsz_bytes + 7) / 8<<"]"<<endl;
      cout<<"should BLOCK outer loops better!"<<endl;

      tmp_beg>>"uint64_t* buf64 = (uint64_t*)malloc("<<asDec((bufsz_bytes+7)/8*8)<<");"
        >>"assert(buf64!=NULL);";
      tmp_last>>"free(buf64);";
    }else{
      tmp_beg
        //>>"struct CPreConvFwd6 cpre;"
        >>"uint64_t buf64["
#if 0
        >>"        ( (nk2+nk4+nk8) * sizeof(union PairedFloat)"
        >>"        + nk1 * sizeof(float)"
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
#if 1
    tmp_last FREE1(nk1) FREE1(nk2) FREE1(nk4) FREE1(nk8)
      FREE1(krn_krsc_k1_off)
      FREE1(krn_krsc_k2_off)
      FREE1(krn_krsc_k4_off)
      FREE1(krn_krsc_k8_off)
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
      setup>>"const float * RESTRICT pKern_gk;";

      CBLOCK_SCOPE(loop_g,"for(int64_t g=0; g<group; ++g)",tmp,setup); // OK sub-tree
      //auto& setup_end = precalc["last"];

      loop_g
        >>"const int64_t kernGroupOffset = g * outChannelGroup * inChannelGroup * kernHW;"
        >>"int64_t k=0;        // 0 .. outChannelGroup-1"
        ;
      if(nk1 > 0){
        int64_t kBy = 1, kMax=kMax1;
        setup>>"float* RESTRICT k1Out = (float *)(buffer + krn_krsc_k1_off);";

        CBLOCK_SCOPE(for_k_kBy,"for( ; k<kMax; k+=kBy)",tmp,loop_g);
        for_k_kBy.DEF(kBy).DEF(kMax)
          >>"pKern_gk = pKernel + kernGroupOffset + (k * inChannelGroup + 0/*c*/) * kernHW;";
        CBLOCK_SCOPE(for_rs,"for (int64_t rs = 0; rs < kernHW; ++rs)",tmp,for_k_kBy);
#if 0 
        for_rs
          >>"for (int64_t c = 0; c < inChannelGroup; ++c){ // loop_c"
          >>"    const float *pKerValue = pKern_gk + c*kernHW + rs;"
          >>"    *k1Out++ = *pKerValue;"
          >>"}"
          ;
#else
        cout<<"krn Weight copy using ve_vcopy32 (not always optimal)"<<endl;
#endif

        for_rs
          >>"float const* pKerValue = pKern_gk + rs;"
          // TODO: MOVE a Cblock (subtree) from some tmp Cunit after a given CBlock (in another Cunit)
          //       or maybe move a full Cunit under one of our known nodes ?
          // TODO: optimize enclosing loop by providing a submatrix copy (2 arb strides-->[unit] const stride)
          >>vel_vcopy32("pKerValue",kernHW,inChannelGroup,"k1Out",1)
          //            source     ,stride,     N        , dest  ,stride
          >>OSSFMT("k1Out += inChannelGroup; /* +="<<inChannelGroup<<" */");
        ;
      }else{
        setup>>"/* no kBy1 loop */";
      }
#if PRE_KRN_MAX >= 2
      //precalc>>"cpre.krn_krsc_k2_off = 0U;";
      if(nk2>0U){
        int64_t kBy = 2, kMax=kMax2;
        setup>>"union PairedFloat* RESTRICT k2Out = (union PairedFloat *)(buffer + krn_krsc_k2_off);";

        CBLOCK_SCOPE(for_k_kBy,"for( ; k<kMax; k+=kBy)",tmp,loop_g);
        for_k_kBy.DEF(kBy).DEF(kMax);
        for_k_kBy>>"pKern_gk = pKernel + kernGroupOffset + (k * inChannelGroup + 0/*c*/) * kernHW;";
        CBLOCK_SCOPE(for_rs,"for (int64_t rs = 0; rs < kernHW; ++rs)",tmp,for_k_kBy);
#if 0 // original
        for_rs
          >>"for (int64_t c = 0; c < inChannelGroup; ++c){ // loop_c"
          >>"    const float *pKerValue = pKern_gk + c*kernHW + rs;"
#if 0 // original uses an intrinsic...
          >>"    const uint64_t kerValue01 = _ve_pack_f32p("
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
        //>>"printf(\" precalc k2Out gk=%d %d rs=%d %d, cc=%d, offset = %d\\n\",(int)g,(int)k,(int)r,(int)s,(int)0,(int)(k2Out - krn_gkrsc_0));"
      }else{
        setup>>"/* no kBy2 loop */";
      }
#if PRE_KRN_MAX >= 4
      //precalc>>"cpre.krn_krsc_k4_off = (nk2) * sizeof(union PairedFloat);";
      if(nk4>0){
        int64_t kBy = 4, kMax=kMax4;
        setup>>"union PairedFloat* RESTRICT k4Out = (union PairedFloat *)(buffer + krn_krsc_k4_off);";
        CBLOCK_SCOPE(for_k_kBy,"for( ; k<kMax; k+=kBy)",tmp,loop_g);
        for_k_kBy.DEF(kBy).DEF(kMax);
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
              "k4Out+1",2,"1")
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
        setup>>"union PairedFloat* RESTRICT k8Out = (union PairedFloat *)(buffer + krn_krsc_k8_off);";
        CBLOCK_SCOPE(for_k_kBy,"for( ; k<kMax; k+=kBy)",tmp,loop_g);
        for_k_kBy.DEF(kBy).DEF(kMax);
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
              "k8Out+0",4,"") // 4 is kBy/4
          >>vel_vmerge32( // memory merge
              "pKerValue+2*inChannelGroup*kernHW", kernHW,
              "pKerValue+3*inChannelGroup*kernHW", kernHW,
              inChannelGroup,
              "k8Out+1",4,"1")
          >>vel_vmerge32( // memory merge
              "pKerValue+4*inChannelGroup*kernHW", kernHW,
              "pKerValue+5*inChannelGroup*kernHW", kernHW,
              inChannelGroup,
              "k8Out+2",4,"2")
          >>vel_vmerge32( // memory merge
              "pKerValue+6*inChannelGroup*kernHW", kernHW,
              "pKerValue+7*inChannelGroup*kernHW", kernHW,
              inChannelGroup,
              "k8Out+3",4,"3")
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

static void unroll_c_kBy1(Cblock &loop_x0, Cblock &loop_s,
    int64_t const inChannelGroup, int64_t const maskW)
{
  // inner loop already has no unroll and 0 or 1 VM 
#if 1
  int64_t const ncBy8 = inChannelGroup/8;
  int64_t const ncBy4 = (inChannelGroup & (2*4-1)) / 4;
  int64_t const ncBy2 = (inChannelGroup & (2*2-1)) / 2;
  int64_t const ncBy1 = (inChannelGroup & (2*1-1)) / 1;
  //                ^                        ^
#elif 1
  int64_t const ncBy8 = 0;
  int64_t const ncBy4 = inChannelGroup/4;
  int64_t const ncBy2 = (inChannelGroup & (2*2-1)) / 2;
  int64_t const ncBy1 = (inChannelGroup & (2*1-1)) / 1;
#elif 1
  int64_t const ncBy8 = 0;
  int64_t const ncBy4 = 0;
  int64_t const ncBy2 = inChannelGroup/2;
  int64_t const ncBy1 = (inChannelGroup & (2*1-1)) / 1;
#elif 1
  int64_t const ncBy8 = 0;
  int64_t const ncBy4 = 0;
  int64_t const ncBy2 = 0;
  int64_t const ncBy1 = inChannelGroup;
#endif
  Cunit& pr = loop_x0.getRoot();
  ostringstream oss;
  CBLOCK_SCOPE(loop_c,"",pr,loop_s);

  assert( ncBy8*8 + ncBy4*4 + ncBy2*2 + ncBy1 == inChannelGroup );
  std::string sumv_u="";
  std::vector<std::string> sumv_P;
  if(ncBy8){ sumv_P = {"vrPsum0","vrPsum2","vrPsum4","vrPsum6"}; }
  else if(ncBy4){ sumv_P = {"vrPsum0","vrPsum2"}; }
  else if(ncBy2){ sumv_P = {"vrPsum0"}; }
  if(ncBy1){
    sumv_u = "vrsum0";
    loop_c.def(sumv_u, "vrsum"); // alias our upper-sum register
    for(auto const& vr: sumv_P)
      loop_x0["first"]>>OSSFMT("__vr "<<vr<<" = vzeros;");
  }else{
    assert( !sumv_P.empty() );
    loop_c.def(sumv_P[0], "vrsum"); // alias our upper-sum register
    for(size_t i=1U; i<sumv_P.size(); ++i)
      loop_x0["first"]>>OSSFMT("__vr "<<sumv_P[i]<<" = vzeros;");
  }
  //if(!sumv_u.empty()) // just a single register name
  //  loop_x0["first"]>>OSSFMT("__vr "<<sumv_u<<" = vzeros;");
  // final summation code (both packed halves co-added into an Upper __vr)
  if(!sumv_P.empty()){
    if(!sumv_u.empty()){
      sumv_u = "vrsum"; // our 'outside' name
      ve_pvfadd(loop_x0["last"]["sumP"], sumv_P, "vl"); // packed sum into sumv_P[0]
      loop_x0["last"]["sum"]
        >>OSSFMT(sumv_u<<" = _vel_vfadds_vvvl("<<sumv_u<<","<<sumv_P[0]<<",vl);")
        >>OSSFMT("__vr vrswap = _vel_vshf_vvvsl("<<sumv_P[0]<<",vzeros,VE_VSHUFFLE_YLZL,vl);")
        >>OSSFMT(sumv_u<<" = _vel_vfadds_vvvl("<<sumv_u<<",vrswap,vl);");
    }else{
      //sumv_u = "vrsum0";
      sumv_u = sumv_P[0] = "vrsum";
      ve_pvfadd(loop_x0["last"]["sumP"], sumv_P, "vl"); // packed sum into sumv_P[0]
      // re-use the packed register for the both-halves-to-upper sum
      loop_x0["last"]["sum"]
        //>>OSSFMT("__vr vrsum0 = "<<sumv_P[0]<<";")
        >>OSSFMT("__vr vrswap = _vel_vshf_vvvsl("<<sumv_P[0]<<",vzeros,VE_VSHUFFLE_YLZL,vl);")
        >>OSSFMT(sumv_u<<" = _vel_vfadds_vvvl("<<sumv_u<<",vrswap,vl);");
    }
  }

  // To retain alignment-guarantee of re-ordered kernel data,
  // do high cBy loops **first**.
  loop_c[".."]>>OSSFMT("// inChannelGroup="<<inChannelGroup<<" : ncBy8="<<ncBy8<<" ncBy4="<<ncBy4<<" ncBy2="<<ncBy2<<" ncBy1="<<ncBy1);
  auto pvfmad = [&](int64_t const N){
    return (maskW
        ? OSSFMT("vrPsum"<<N<<" = _vel_pvfmad_vvsvMvl(vrPsum"<<N<<",kerP"<<N
          <<",vrin"<<N<<",vmP,vrPsum"<<N<<",vl);")
        : OSSFMT("vrPsum"<<N<<" = _vel_pvfmad_vvsvl  (vrPsum"<<N<<",kerP"<<N
          <<",vrin"<<N<<",vl);"));
  };
  auto load2 = [&](int64_t const N){
    return OSSFMT(
        "__vr vrin"<<N  <<" = _vel_vldu_vssl(4*strideWidth,pIn + "<<N  <<"*inHW,vl);\n"
        "__vr vrin"<<N+1<<" = _vel_vldu_vssl(4*strideWidth,pIn + "<<N+1<<"*inHW,vl);\n"
        // XXX I actualy did not see any big speed effect of removing pack_f32p (check assembler!)
        //"uint64_t const kerP"<<N<<" = _ve_pack_f32p(pKerValue+"<<N<<", pKerValue+"<<N+1<<");\n"
        //"// Opposite endian-ness from _ve_pack_f32p :(\n"
        "uint64_t const kerP"<<N<<" = *(uint64_t const*)(void const*)(pKerValue+"<<N<<");\n");
  };
  auto calc2 = [&](int64_t const N){
    return OSSFMT(
        //"__vr vrinP"<<N<<" = _ve_vshf_vvvs(vrin"<<N<<",vrin"<<N+1<<", VE_VSHUFFLE_YUZU);\n"
        "vrin"<<N<<" = _vel_vshf_vvvsl(vrin"<<N+1<<",vrin"<<N<<", VE_VSHUFFLE_YUZU, vl);\n"
        << pvfmad(N));
  };
  auto kern2 = [&](int64_t const N){
    return load2(N)+"\n"+calc2(N);
  };
  auto bump = [&](int64_t const N){ // pIn c-stride is inHW, but pKerValue [re-ordered] c-stride is 1
    return OSSFMT("pIn += "<<N<<"*inHW;\n"
        "pKerValue += "<<N<<";");
  };
  auto cByComment = [&](int64_t const ca, int64_t const cz, int64_t const ncBy){
    return OSSFMT("// loop_c in ["<<ca<<","<<cz<<") by "<<ncBy<<", of inChannelGroup="<<inChannelGroup);
  };
  //loop_c[".."]>>(ncBy8? "__vr vrin0, vrin1, vrin2, vrin3, vrin4, vrin5, vrin6, vrin7;"
  //        : ncBy4? "__vr vrin0, vrin1, vrin2, vrin3;"
  //        : ncBy2? "__vr vrin0, vrin1;"
  //        : /*ncBy1?*/ "__vr vrin0;");
  loop_c>>"CHK( (intptr_t)pKerValue % 8 == 0); // uint64_t load swaps U|L wrt _vel_pack_f32p(ptr,ptr+1)\n"
    >>"float const* RESTRICT pIn;"
    >>"float const* RESTRICT pKerValue_end;";
  loop_c>>OSSFMT("pIn = pIn_0 + 0/*c*/ *inHW + (i+r*dilationHeight)*inWidth\n"
      "    + x0*strideWidth-padWidth + s*dilationWidth;");
  if(ncBy8){
    int64_t const ca=0;
    int64_t const cz=ncBy8*8;
    CBLOCK_FOR(loop_c8,0,"for ( ; pKerValue<pKerValue_end; )",loop_c);
    loop_c8[".."]>>cByComment(ca,cz,8)
      >>OSSFMT("pKerValue_end = pKerValue + "<<cz-ca<<";"); // = ncBy8
    loop_c8>>kern2(0)>>kern2(2)>>kern2(4)>>kern2(6)>>bump(8);
  }
  if(ncBy4){
    int64_t const ca=ncBy8*8;
    int64_t const cz=ca + ncBy4*4;
    CBLOCK_FOR(loop_c4,0,"for ( ; pKerValue<pKerValue_end; )",loop_c);
    loop_c4[".."]>>cByComment(ca,cz,4)
      >>OSSFMT("pKerValue_end = pKerValue + "<<cz-ca<<";");
    loop_c4>>kern2(0)>>kern2(2)>>bump(4);
  }
  if(ncBy2){
    int64_t const ca=ncBy8*8+ncBy4*4;
    int64_t const cz=ca + ncBy2*2;
    if(ncBy2==1){
      loop_c["By2"]>>cByComment(ca,cz,ncBy2)
        >>kern2(0)
        >>"pKerValue+=2;";
      if(cz < inChannelGroup) loop_c["By2"]>>"pIn += 2*inHW;";
    }else{
      //CBLOCK_FOR(loop_c2,0,OSSFMT("for(int64_t c="<<ca<<"; c<"<<cz<<"; c+=2)"),loop_c);
      CBLOCK_FOR(loop_c2,0,"for( ; pKerValue<pKerValue_end; )",loop_c);
      loop_c2[".."]>>cByComment(ca,cz,2)
        >>OSSFMT("pKerValue_end = pKerValue + "<<cz-ca<<";");
      loop_c2>>kern2(0)<<bump(2);
    }
  }
  if(ncBy1){
    int64_t const ca=ncBy8*8+ncBy4*4+ncBy2*2;
    int64_t const cz=inChannelGroup;
    assert( cz == ca + ncBy1 );
    // cBy1 loop:
    auto vfmads = [&](int64_t const N){
      return (maskW
          ? OSSFMT("vrsum"<<N<<" = _vel_vfmads_vvsvmvl(vrsum"<<N
            <<", *(pKerValue+"<<N<<"),vrinBy1_"<<N<<",vm23,vrsum"<<N<<",vl);")
          : OSSFMT("vrsum"<<N<<" = _vel_vfmads_vvsvl  (vrsum"<<N
            <<", *(pKerValue+"<<N<<"),vrinBy1_"<<N<<",vl);"));
    };
    auto kern = [&](int64_t const N){
      return
        OSSFMT("__vr vrinBy1_"<<N<<" = _vel_vldu_vssl(4*strideWidth,pIn + "<<N<<"*inHW,vl);\n"
          <<vfmads(N));
    };
    if(ncBy1==1){
      loop_c["By1"]>>cByComment(ca,cz,1)
        >>kern(0)
        >>"++pKerValue;";
      if(cz < inChannelGroup) loop_c["By1"]>>"pIn += 1*inHW;";
    }else{
      CBLOCK_FOR(loop_c1,0,"for ( ; pKerValue<pKerValue_end; )",loop_c);
      loop_c1[".."]>>cByComment(ca,cz,1)
        >>OSSFMT("pKerValue_end = pKerValue + "<<cz-ca<<";"); // = ncBy1
      loop_c1>>kern(0)>>bump(1);
    }
  }
}

/** this kByMax 8 is based on implementation direct_default2p.c, POUTX==1.
 *
 * \todo if kh_beg is always 0 and kh_end is always max, then elide kh_btab and kh_etab tables
 * (or whatever KH_BEG_END macro does) with "trivial case" implementations.
 *
 */
DllFile cjitConvolutionForward6( struct param const* const p )
{
  ostringstream oss; // scratchpad for OSSFMT etc.
  int const verbose=1;
  string const impl = "cjitConvFwd6";
  DllFile df; // return value
  //DllFileAux dfx("Convolution","Forward");
  std::string parmstr = paramString(p);
  df.basename = impl+"_"+parmstr;
  cout<<impl<<" : df.basename = "<<df.basename<<endl;

  // we use intrinsics.  suffix matches build recipe in "bin.mk"
  df.suffix = "-vi.c";

  Cunit pr("program");
  pr.v = verbose;     // default is quite lengthy!

  int64_t const kByMax = KBYMAX; // kByMax is chosen from 1,2,4,8
  cout<<impl<<" KBYMAX="<<KBYMAX
    <<" PRE_KRN_MAX="<<PRE_KRN_MAX
    <<" kByMax="<<kByMax<<endl;

  auto& includes = pr["includes"]<<Endl;
  includes
    >>CSTR(#include "vednn.h")
    >>CSTR(#if __has_include("vednnx.h")) // an old clang directive
    >>CSTR(#include "vednnx.h")
    >>CSTR(#endif)
    //>>CSTR(#include "veintrin.h")
    >>CSTR(#include "velintrin.h")
    >>"#include <stdio.h>"
    >>"#include <stdlib.h>"
    >>"#include <assert.h>"
    >>"#include <stdint.h>"
    ;
  pr["macros"]<<"\n#define MVL (256)"
    >>""
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
    ;
  pr["structs"]
    >>"union PairedFloat {"
    >>"    float f[2];                     // packed-pair floats"
    >>"    uint64_t pair;                  // forces alignment 8"
    >>"};"
    >>"struct CPreConvFwd6 {"
    >>"    size_t bufsz;"
    >>"    int8_t * buffer;"
    >>"    size_t krn_gkrsc_off;"
    >>"    size_t krn_krsc_k1_off;"
    >>"    size_t krn_krsc_k2_off;"
    >>"    size_t krn_krsc_k4_off;"
    >>"    size_t krn_krsc_k8_off;"
    >>"};"
    ;

  std::string fn_declare = "vednnError_t "+df.basename+"(\n    "+
    multiReplace(",",",\n    ", CSTR(CONVX_FWD_ORDER(
            VEDNN_PARAMS_CONV_FORWARD,
            VEDNN_DATARG_CONV_FORWARD))) +"\n)"
    ;
  df.syms.push_back(SymbolDecl(df.basename,
        "vednn ConvolutionForward "+paramString(p),
        fn_declare));



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
  fn_ptrs
    //.def("RESTRICT","")
    .def("RESTRICT","restrict")
    >>"float const * RESTRICT pIn  = pDataIn;"
    >>"float const * RESTRICT pKernel = pDataKernel;"
    >>"float * RESTRICT const pOut = pDataOut;"
    ;

  int64_t vlen=0; // we have not yet set vlen (so precalc_cocde need not save/restore it)
  pair<string,string> precalc_code = strCPreConvFwd6(p,kByMax,vlen);
  // \post vlen changes if vlen register was clobbered
  auto& precalc = fn["precalc"];
  precalc>>"// [opt] packing kernel values for sequential access and single-u64 loading";

  // determine x and y output coords that don't need masking (all inputs good)
  KernLims kl = kernLims(p);
  int64_t yok_beg, yok_end;
  nomask_Height( kl, kernHeight, yok_beg, yok_end );
  int64_t xok_beg, xok_end;
  nomask_Width( kl, kernWidth, xok_beg, xok_end );

  bool maskH = true, maskW = true;
  if( yok_beg==0 && yok_end==outHeight ) maskH = false;
  if( xok_beg==0 && xok_end==outWidth  ) maskW = false;

  precalc CONST1(maskH) CONST1(maskW) CONST1(yok_beg) CONST1(yok_end) CONST1(xok_beg) CONST1(xok_end); //debug
  if( maskH ) precalc>>str_kh_be_static_array(p);
  precalc
    >>"#define KH_BEG_END \\"
    >>(yok_beg==0
        ? "                  int64_t const kh_beg = 0; \\"
        : "                  int64_t const kh_beg = kh_btab[y]; \\")
    >>(yok_end==outHeight
        ? "                  int64_t const kh_end = kernHeight"
        : "                  int64_t const kh_end = kh_etab[y]")
    >>precalc_code.first
    ;

#if 1 // so far have not seen any diff between ot stores and normal
      auto ve_vstu = [&oss](std::string ptr, std::string vr, std::string vl){
        return OSSFMT("_vel_vstu_vssl("<<vr<<", 4,"<<ptr<<","<<vl<<");");
      };
      auto ve_vstl = [&oss](std::string ptr, std::string vr, std::string vl){
        return OSSFMT("_vel_vstl_vssl("<<vr<<", 4,"<<ptr<<","<<vl<<");");
      };
      auto ve_svob = [](){ return "//_vel_svob();"; };
#else // ot stores
      auto ve_vstu = [&oss](std::string ptr, std::string vr, std::string vl){
        return OSSFMT("_vel_vstuot_vssl("<<vr<<", 4,"<<ptr<<","<<vl<<");");
      };
      auto ve_vstl = [&oss](std::string ptr, std::string vr, std::string vl){
        return OSSFMT("_vel_vstlot_vssl("<<vr<<", 4,"<<ptr<<","<<vl<<");");
      };
      auto ve_svob = [](){ return "_vel_svob();"; };
#endif

#if 0 // ORIGINAL
  int64_t const vl_x_init = min<long>(outWidth, MVL);
  bool const x0_check_vl = vl_x_init < outWidth && outWidth%vl_x_init != 0;
  // Compare usual with "equalized" vector length:
  // J              cjitConvFwd6  |    1x     2.010 ms ~0.0001  57.23G Bug6
  //    cjitConvFwd6_mb1g1_ic32ih26iw260_oc32oh24ow264_kh3ph0sh1dh0_kw3pw3sw1dw0
  // versus equalized:
  // J              cjitConvFwd6  |    1x     1.624 ms ~0.0001  70.84G Bug6
  //
  // This simple modification gave 20% speedup.
  //
#endif
  //
  // if "remainder" vlen really small, maybe can get smaller latency by ~equalizing vlen
  // 20% speedup observed when in most important cases for this optimization
  //
  int64_t const vl_x_init = ve_vlen_suggest( outWidth );
  bool const x0_check_vl = outWidth%vl_x_init != 0;
  std::string str_vl(x0_check_vl? "vl":"vl_x_init"); // vl_x_init is a #define
#define NO_SET_VLEN(...) OSSFMT("NO_SET_VLEN("<<__VA_ARGS__<<");")

  auto& fn_vec_init =
    fn["vec_init"]
    // perhaps output const if possible
    >>OSSFMT("int64_t vl = vl_x_init; // "<<vl_x_init)
    >>NO_SET_VLEN(str_vl)
    >>"const __vr vzeros = _vel_vbrds_vsl(0.0f, vl_x_init );"
    >>(maskW? "":"//")<<"const __vr vrseq = _vel_vseq_vl( vl_x_init );"
    >>"float * RESTRICT pOutx = pDataOut;"
    ;

  fn_vec_init.DEF(vl_x_init);
  vlen = vl_x_init;
  if(maskW) vrj_init(fn_vec_init);

  CBLOCK_SCOPE(loop_n,"for(int64_t n=0; n<batch; ++n)",pr,fn);
  CBLOCK_SCOPE(loop_g,"for(int64_t g=0; g<group; ++g)",pr,loop_n); // OK sub-tree
  loop_g
    >>"const int64_t outGroupOffset  = g * outChannelGroup * outHW;"
    >>"const int64_t inGroupOffset   = g * inChannelGroup * inHW;"
    >>"const int64_t kernGroupOffset = g * outChannelGroup * inChannelGroup * kernHW;"
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
  int64_t kBy=1;      // 1,2,4,... kByMax
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
      vlen = vl_x_init;
      bool const do_unroll_c = inChannelGroup >= 8;
      //bool const do_unroll_c = false;

      int64_t max_unroll_outer = (maskW ? 12: 16);
      int64_t sofar = 1;
      if(do_unroll_c && maskW) max_unroll_outer /= 2; // will use VMP registers

      //DEFINE_UNROLL(un_c , 1, sofar, inChannelGroup);
      int64_t un_c;
      if(do_unroll_c){
        un_c = -1;
      }else{
        un_c = min<int64_t>(16,inChannelGroup); // 16 or 14?
      }

      //DEFINE_UNROLL(un_s , 1, sofar, kernWidth); // usually ok, but VM-VM copy for mb1g2_ic62ih32oc62oh32kh3
      int64_t un_s;
      if(maskW){
        // XXX no warn and INCORRECT RESULT result for mb1_ic1ih32oc1oh30kh3 XXX or mb1g2_ic62ih32oc10oh32kh3_ph1
        //DEFINE_UNROLL(un_ss , max_unroll_outer, sofar, kernWidth);

        DEFINE_NOUNROLL(un_ss , max_unroll_outer, sofar, kernWidth);
        un_s = un_ss;
      }else{
        DEFINE_UNROLL(un_ss , max_unroll_outer, sofar, kernWidth);
        un_s = un_ss; // wrong result, no warning
        //un_s = 0; // explicit 'nounroll'
      }
      DEFINE_UNROLL(un_r , max_unroll_outer, sofar, kernHeight);
      DEFINE_UNROLL(un_x0, max_unroll_outer, sofar, (outWidth+vl_x_init-1)/vl_x_init);
      DEFINE_NOUNROLL(un_y , 1, sofar, outHeight);
      DEFINE_NOUNROLL(un_k , max_unroll_outer, sofar, (kMax-k+kBy-1)/kBy);
      CBLOCK_SCOPE(scope_kMax,"",pr,loop_g);
      scope_kMax.DEF(kBy).DEF(kMax);
      scope_kMax["last"]>>"CHK(pOutx == pOut + outGroupOffset + (n*outChannel+kMax) * outHW);";
      //scope_kMax["last"]>>ve_svob();
      CBLOCK_FOR(loop_k,un_k,"for(; k<kMax; k+=kBy)",scope_kMax);
      precalc>>"/*pre*/ float const* RESTRICT const krn_krsc_k1 = (float const*)(buffer + krn_krsc_k1_off);";
      // every minibatch restarts the kernel pointer
      loop_k[".."]>>"/*pre*/ float const* RESTRICT krn_gk = krn_krsc_k1 + (nk1/group)*g;";
      loop_k["last"]>>"/*pre*/ krn_gk += kBy*inChannelGroup*kernHW;";
      loop_k>>"CHK(pOutx == pOut + outGroupOffset + (n*outChannel+k) * outHW);";

      CBLOCK_FOR(loop_y,un_y,"for(int64_t y=0 ; y<outHeight; ++y)",loop_k);
      loop_y>>"const int64_t i = y * strideHeight - padHeight;"
        >>"KH_BEG_END;";
      //
      // problem case: mb1_ic1ih32oc1oh30kh3
      // With _ve_lvl ABSENT, get INCORRECT result for -fno-unroll XXX (correct for unroll!!!)
      // With both _ve_lvl there (even if they do nothing) CORRECT results
      //
      //loop_y
      //  //>>""
      //  >>(x0_check_vl? "": "//")
      //  <<OSSFMT("_ve_lvl(vl = vl_x_init /*"<<vl_x_init<<"*/);");

      CBLOCK_FOR(loop_x0,un_x0,"for(int64_t x0=0 ; x0<outWidth; x0+=vl_x_init)",loop_y);

      if(x0_check_vl){
        //loop_x0<<"_ve_lvl(vl = outWidth-x0 < vl_x_init ? outWidth-x0: vl_x_init);";
        loop_x0<<"vl = outWidth-x0 < vl_x_init ? outWidth-x0: vl_x_init;";
      }
      //  >>(x0_check_vl? "": "//") // it was done before, IN PRINCIPLE can elide
      //... BUT eliding --> INCORRECT for no-unroll, (correct for unroll)
      //    asm code ALMOST IDENTICAL !!! (test case mb100_ic1ih32oc1oh30kh3)
      //  <<"_ve_lvl(vl = outWidth-x0 < vl_x_init ? outWidth-x0: vl_x_init);";
      // is this enough to get correct compile? *** YES *** XXX (maybe)
      loop_x0>>NO_SET_VLEN(str_vl);

      loop_x0["first"]>>"__vr vrsum = vzeros;";
      //loop_x0["first"]>>"__vr vrsum = _vel_vor_vsvl(0,vzeros,"<<str_vl<<");";
      if(maskW) vrj_induce(loop_x0); // vrj ~ input row pixels
      // store output, bump output pointer
      loop_x0["last"]["last"]
        //>>OSSFMT("_vel_vstu_vssl(vrsum, 4, pOutx, "<<str_vl<<");")
        >>ve_vstu("pOutx","vrsum",str_vl)
        >>"pOutx += "<<str_vl<<";";

      // loop_x0 kBy=1 kernel...
      CBLOCK_FOR(loop_r,un_r,"for (int64_t r = kh_beg; r < kh_end; ++r)",loop_x0);
      loop_r[".."]>>"/*pre*/ float const* RESTRICT pKerValue = krn_gk\n"
        /*      */ "               + kh_beg*inChannelGroup*kernWidth; // skip some";

      CBLOCK_FOR(loop_s,un_s,"for (int64_t s = 0; s < kernWidth; ++s)",loop_r);
      if(maskW){
        loop_s[".."]>>"__vr vrw = _vel_vor_vsvl(0,vrj,"<<str_vl<<"); // input row pixels";
        if(x0_check_vl){
          loop_s>>"__vm256 vm23 = " VEL_VFMK_mvs_0_TO(vrw,inWidth,vl) ";";
        }else{
          loop_s>>"__vm256 vm23 = " VEL_VFMK_mvs_0_TO(vrw,inWidth,vl_x_init) ";";
        }
        if(do_unroll_c)
          if(x0_check_vl){
            loop_s>>VEL_DECL_VM512(vmP, vm23,vm23, vl);
          }else{
            loop_s>>VEL_DECL_VM512(vmP, vm23,vm23, vl_x_init);
          }
        loop_s["last"]>>"vrw = _vel_vaddsl_vsvl(dilationWidth, vrw, "<<str_vl<<");";
      }
      if(do_unroll_c){
        unroll_c_kBy1(loop_x0, loop_s, inChannelGroup, maskW);
      }else{
        CBLOCK_FOR(loop_c,un_c,"for (int64_t c = 0; c < inChannelGroup; ++c)",loop_s);
        loop_c
          >>"const float *pIn = pIn_0 + c*inHW + (i+r*dilationHeight)*inWidth"
          >>"                 + x0*strideWidth-padWidth + s*dilationWidth;"
          >>"__vr vrin = _vel_vldu_vssl(4*strideWidth,pIn,"<<str_vl<<");"
          //>>"_vel_vpfchv(4*strideWidth,pIn+inHW,"<<str_vl<<");"
          >>"vrsum = "<<(maskW
              ? OSSFMT("_vel_vfmads_vvsvmvl(vrsum, *pKerValue, vrin, vm23, vrsum,"<<str_vl<<");")
              : OSSFMT("_vel_vfmads_vvsvl(vrsum, *pKerValue, vrin, "<<str_vl<<");"))
          >>"++pKerValue;"
          ;
      }// end do_unroll_c
      //k = kMax;
      // "simulate" effect of above JIT loop_k on 'k' (illustrative)
      for(; k<kMax; k+=kBy ) /*no-op*/;
      assert( k == kMax );
    }
    loop_g[string("kBy-")+hexdec(kBy)]
      >>"#endif // unroll by "+hexdec(kBy)<<", exit w/ k="<<hexdec(k)
      <<" of outChannelGroup="<<hexdec(outChannelGroup)
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
      // maybe 1 VMP
      int64_t const max_unroll_outer = (maskW ? 6: 12);
      int64_t sofar = 1;
      int64_t un_c = 0;
      DEFINE_UNROLL(un_s , max_unroll_outer, sofar, kernWidth); // VMP-copy mb1g2_ic62ih32oc62oh32kh3_ph1
      DEFINE_UNROLL(un_r , max_unroll_outer, sofar, kernHeight);
      DEFINE_UNROLL(un_x0, max_unroll_outer, sofar, (outWidth+vl_x_init-1)/vl_x_init);
      DEFINE_UNROLL(un_y , max_unroll_outer, sofar, outHeight);
      DEFINE_UNROLL(un_k , max_unroll_outer, sofar, (kMax-k+kBy-1)/kBy);
      CBLOCK_SCOPE(scope_kMax,"",pr,loop_g);
      scope_kMax.DEF(kBy).DEF(kMax)
        >>OSSFMT("//"<<max_unroll_outer<<" un_s/r/x0 "<<un_s<<" "<<un_r<<" "<<un_x0)
        >>"//"<<CSTR(if(n+g==0) printf("for k = %ld, %ld, %ld...\n",(long)k,(long)kMax,(long)kBy););
      scope_kMax["last"]>>"CHK(pOutx == pOut + outGroupOffset + (n*outChannel+kMax) * outHW);";
      //scope_kMax["last"]>>ve_svob();
      CBLOCK_FOR(loop_k,un_k,"for(; k<kMax; k+=kBy)",scope_kMax);
      loop_k
        >>"CHK(pOutx == pOut + outGroupOffset + (n*outChannel+k) * outHW);"
        >>"float* pOutx1 = pOutx + outHW;"
        ;
      CBLOCK_FOR(loop_y,un_y,"for(int64_t y=0; y<outHeight; ++y)",loop_k);
      loop_y
        >>"const int64_t i = y * strideHeight - padHeight;"
        >>"KH_BEG_END;"
        >>"vl = vl_x_init;"
        >>NO_SET_VLEN(str_vl)
        ;
      // slightly faster looping for x0... (detect vl <= 0 instead of x0 too high)
      CBLOCK_FOR(loop_x0,un_x0,"for(int64_t x0=0; /*x0<outWidth*/; /*x0+=vl_x_init*/)",loop_y);
      loop_x0>>NO_SET_VLEN(str_vl);

      if(maskW) vrj_induce(loop_x0); // vrj ~ vector of input x values
      loop_x0>>"__vr vrsum01 = vzeros;";

      precalc>>"union PairedFloat const* RESTRICT const krn_krsc_k2 ="
        >>"        (union PairedFloat const*)(buffer + krn_krsc_k2_off);";
      loop_k[".."]>>"union PairedFloat const* RESTRICT krn_gk = krn_krsc_k2 + g*(nk2/group);";
      loop_k["ind"]>>"krn_gk += (kBy/2U)*inChannelGroup*kernHW;";

      CBLOCK_FOR(loop_r,un_r,"for (int64_t r = kh_beg; r < kh_end; ++r)",loop_x0);
      loop_r[".."]>>"union PairedFloat const* RESTRICT pKer2 = krn_gk"
        >>"        + kh_beg*kernWidth*inChannelGroup*(kBy/2); // skip some";
      CBLOCK_FOR(loop_s,un_s,"for (int64_t s = 0; s < kernWidth; ++s)",loop_r);
      if(maskW){
        loop_s[".."]>>"__vr vrw = _vel_vor_vsvl(0,vrj,vl); // input row pixels";
        loop_s
          >>"__vm256 vm23 = " VEL_VFMK_mvs_0_TO(vrw,inWidth,vl) ";"
          >>VEL_DECL_VM512( vmP, vm23,vm23, vl);
        loop_s["last"]>>"vrw = _vel_vaddsl_vsvl(dilationWidth, vrw, vl);";
      }
      CBLOCK_FOR(loop_c,un_c,"for (int64_t c = 0; c < inChannelGroup; ++c)",loop_s);
      loop_c.define("DOSUM(VRSUM,PAIR)",OSSFMT("VRSUM = "<<(maskW
              ? "_vel_pvfmad_vvsvMvl(VRSUM,PAIR,vrinP,vmP,VRSUM,vl)"
              : "_vel_pvfmad_vvsvl  (VRSUM,PAIR,vrinP,vl)")))
        >>"const float *pIn = pIn_0 + c*inHW + (i+r*dilationHeight)*inWidth"
        >>"                 + x0*strideWidth-padWidth + s*dilationWidth;"
        >>"__vr vrin = _vel_vldu_vssl(4*strideWidth,pIn,vl);"
        >>"__vr vrinP = _vel_vshf_vvvsl(vrin, vrin, VE_VSHUFFLE_YUZU, vl);"
        >>"DOSUM(vrsum01, pKer2[0].pair);"
        >>"pKer2+=kBy/2U;"
        ;
      loop_x0["induce+write"]
        //>>"_vel_vstu_vssl(vrsum01, 4, pOutx , vl);"
        //>>"_vel_vstl_vssl(vrsum01, 4, pOutx1, vl);"
        >>ve_vstu("pOutx", "vrsum01",str_vl)
        >>ve_vstl("pOutx1","vrsum01",str_vl)
        //>>"pOutx  += vl;"
        //>>"pOutx1 += vl;"
        >>OSSFMT("pOutx  += "<<str_vl<<";")
        >>OSSFMT("pOutx1 += "<<str_vl<<";")
        >>"//"<<CSTR(printf(" k %ld vl %-3ld outIndex0=%ld\n",(long)k,(long)vl,(long)(pOutx-pOut));)
        ;
      //bool const x0_check_vl = true;
      if(x0_check_vl){ // tiny bit faster than normal-looking loop exit check
        loop_x0["check_vl"]
          >>"x0 += vl_x_init;"
          >>"vl = outWidth - x0;"
          >>"if( vl <= 0 ) break;"
          >>"vl = vl < vl_x_init? vl: vl_x_init;"
          >>NO_SET_VLEN(str_vl)
          ;
      }else{ // equiv to normal-looking loop
        loop_x0["check_vl"]
          >>"x0 += vl_x_init;"
          >>"if( x0 >= outWidth ) break; // VL const for all loop passes"
          ;
      }
      loop_k["bump pOutx"]
        >>"pOutx += /*kBy-1*/ 1 * outHW; // inner increment is outHW, outer wants kBy*outHW"
        ;
      // "simulate" effect of above JIT loop_k on 'k' (illustrative)
      for(; k<kMax; k+=kBy ) /*no-op*/;
      // [ Note: might also sometimes need to mirror what has happened to vl ]
      assert( k == kMax );
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
      // maybe 1 VMP
      int64_t const max_unroll_outer = (maskW ? 7: 18);
      int64_t sofar = 1;
      int64_t un_c = 0;
      DEFINE_UNROLL(un_s , max_unroll_outer, sofar, kernWidth);
      DEFINE_UNROLL(un_r , max_unroll_outer, sofar, kernHeight);
      DEFINE_UNROLL(un_x0, max_unroll_outer, sofar, (outWidth+vl_x_init-1)/vl_x_init);
      DEFINE_UNROLL(un_y , max_unroll_outer, sofar, outHeight);
      DEFINE_DEFUNROLL(un_k , max_unroll_outer, sofar, (kMax-k+kBy-1)/kBy);
      CBLOCK_SCOPE(scope_kMax,"",pr,loop_g);
      scope_kMax.DEF(kBy).DEF(kMax)
        >>"//"<<CSTR(if(n+g==0) printf("for k = %ld, %ld, %ld...\n",(long)k,(long)kMax,(long)kBy););
      //scope_kMax["last"]>>ve_svob();
      CBLOCK_FOR(loop_k,un_k,"for(; k<kMax; k+=kBy)",scope_kMax);
      loop_k
        >>"float* pOutx1 = pOutx + outHW;"
        >>"float* pOutx2 = pOutx + 2*outHW;"
        >>"float* pOutx3 = pOutx + 3*outHW;"
        ;
      CBLOCK_FOR(loop_y,un_y,"for(int64_t y=0; y<outHeight; ++y)",loop_k);
      loop_y
        >>"const int64_t i = y * strideHeight - padHeight;"
        >>"KH_BEG_END;"
#define VLC 1
#if VLC==1
        >>"vl = vl_x_init;"
        >>NO_SET_VLEN(str_vl)
#endif
        ;
#if VLC==1
      // slightly faster looping for x0... (detect vl <= 0 instead of x0 too high)
      CBLOCK_FOR(loop_x0,un_x0,"for(int64_t x0=0; /*x0<outWidth*/; /*x0+=vl_x_init*/)",loop_y);

      loop_x0>>NO_SET_VLEN(str_vl);

#else
      CBLOCK_FOR(loop_x0,un_x0,"for(int64_t x0=0; x0<outWidth; x0+=vl_x_init)",loop_y);
      //loop_x0<<"_ve_lvl( vl = (outWidth-x0 < vl_x_init? outWidth-x0: vl_x_init));";
      loop_x0<<"vl = (outWidth-x0 < vl_x_init? outWidth-x0: vl_x_init));";
      loop_x0>>NO_SET_VLEN(str_vl);
#endif
      if(maskW) vrj_induce(loop_x0); // vrj ~ vector of input x values
      loop_x0>>"__vr vrsum01 = vzeros;"
        >>"__vr vrsum23 = vzeros;";

      precalc>>"union PairedFloat const* RESTRICT const krn_krsc_k4 ="
        >>"         (union PairedFloat const*)(buffer + krn_krsc_k4_off);";
      loop_k[".."]>>"union PairedFloat const* RESTRICT krn_gk = krn_krsc_k4 + g*(nk4/group);";
      loop_k["last"]>>"krn_gk += (kBy/2U)*inChannelGroup*kernHW;";

      CBLOCK_FOR(loop_r,un_r,"for (int64_t r = kh_beg; r < kh_end; ++r)",loop_x0);
      loop_r[".."]>>"union PairedFloat const* RESTRICT pKer2 = krn_gk"
        >>"        + kh_beg*kernWidth*inChannelGroup*(kBy/2); // skip some";
      CBLOCK_FOR(loop_s,un_s,"for (int64_t s = 0; s < kernWidth; ++s)",loop_r);
      if(maskW){
        loop_s[".."]>>"__vr vrw = _vel_vor_vsvl(0,vrj,vl); // input row pixels";
        loop_s
          >>"__vm256 vm23 = " VEL_VFMK_mvs_0_TO(vrw,inWidth,vl) ";"
          >>VEL_DECL_VM512( vmP, vm23,vm23, vl)
          ;
        loop_s["last"]>>"vrw = _vel_vaddsl_vsvl(dilationWidth, vrw, vl);";
      }
      CBLOCK_FOR(loop_c,un_c,"for (int64_t c = 0; c < inChannelGroup; ++c)",loop_s);
      loop_c.define("DOSUM(VRSUM,PAIR)", OSSFMT("VRSUM = "<<(maskW
              ? "_vel_pvfmad_vvsvMvl(VRSUM,PAIR,vrinP,vmP,VRSUM,vl)"
              : "_vel_pvfmad_vvsvl(VRSUM,PAIR,vrinP,vl)")))
        >>"const float *pIn = pIn_0 + c*inHW + (i+r*dilationHeight)*inWidth"
        >>"                 + x0*strideWidth-padWidth + s*dilationWidth;"
        >>"__vr vrinP= _vel_vldu_vssl(4*strideWidth,pIn,vl);"
        >>"vrinP = _vel_vshf_vvvsl(vrinP, vrinP, VE_VSHUFFLE_YUZU, vl);"
        >>"DOSUM(vrsum01, pKer2[0].pair);"
        >>"DOSUM(vrsum23, pKer2[1].pair);"
        >>"pKer2+=kBy/2U;"
        ;
      loop_x0["induce+write"]
        //>>"_vel_vstu_vssl(vrsum01, 4, pOutx , vl);"
        //>>"_vel_vstl_vssl(vrsum01, 4, pOutx1, vl);"
        //>>"_vel_vstu_vssl(vrsum23, 4, pOutx2, vl);"
        //>>"_vel_vstl_vssl(vrsum23, 4, pOutx3, vl);"
        >>ve_vstu("pOutx ","vrsum01",str_vl)
        >>ve_vstl("pOutx1","vrsum01",str_vl)
        >>ve_vstu("pOutx2","vrsum23",str_vl)
        >>ve_vstl("pOutx3","vrsum23",str_vl)
        >>OSSFMT("pOutx  += "<<str_vl<<";")
        >>OSSFMT("pOutx1 += "<<str_vl<<";")
        >>OSSFMT("pOutx2 += "<<str_vl<<";")
        >>OSSFMT("pOutx3 += "<<str_vl<<";")
#if VLC==1
        >>"x0 += vl_x_init;"
#endif
        ;
#if VLC==1
      //bool const x0_check_vl = true;
      if(x0_check_vl){
        loop_x0["check_vl"]
          >>"vl = outWidth - x0;"
          >>"if( vl <= 0 ) break;"
          >>"vl = vl < vl_x_init? vl: vl_x_init;"
        >>"NO_SET_VLEN(vl);"
          ;
      }else{
        loop_x0["check_vl"]
          >>"if( x0 >= outWidth ) break;"
          >>"NO_SET_VLEN(vl_x_init);"
          ;
      }
#endif
      loop_k["bump pOutx"]
        >>"pOutx += (kBy-1) * outHW; // inner increment is outHW, outer wants kBy*outHW"
        ;
      //k = kMax; // could be too naive?
      // "simulate" effect of above JIT loop_k on 'k' (illustrative)
      for(; k<kMax; k+=kBy ) /*no-op*/;
      assert( k == kMax );
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
        >>"//"<<CSTR(if(n+g==0) printf("for k = %ld, %ld, %ld...\n",(long)k,(long)kMax,(long)kBy););
      //scope_kMax["last"]>>ve_svob();
      vlen = vl_x_init;
      //int64_t const max_unroll_outer = (maskW ? 7: 18);
      int64_t const max_unroll_outer = (maskW ? 9: 9);
      int64_t sofar = 1;
      // provide biggest unrolls to innermost loops first
      //int64_t un_c = -1;
      DEFINE_UNROLL(un_s , max_unroll_outer, sofar, kernWidth);
      DEFINE_UNROLL(un_r , max_unroll_outer, sofar, kernHeight);
      DEFINE_UNROLL(un_x0, max_unroll_outer, sofar, (outWidth+vl_x_init-1)/vl_x_init);
      DEFINE_UNROLL(un_y , max_unroll_outer, sofar, outHeight);
      DEFINE_UNROLL(un_k , max_unroll_outer, sofar, (kMax-k+kBy-1)/kBy);
      cout<<" Fwd6: unrolls-kBy8: "<<un_s<<" "<<un_r<<" "<<un_x0<<" "<<un_y<<" "<<un_k<<endl;
      if(!x0_check_vl)
        scope_kMax
          >>"vl = vl_x_init;"
          >>NO_SET_VLEN(str_vl)
          ;
      CBLOCK_FOR(loop_k,un_k,"for(; k<kMax; k+=kBy)",scope_kMax);
      loop_k
        >>"float* pOutx1 = pOutx + outHW;"
        >>"float* pOutx2 = pOutx + 2*outHW;"
        >>"float* pOutx3 = pOutx + 3*outHW;"
        >>"float* pOutx4 = pOutx + 4*outHW;"
        >>"float* pOutx5 = pOutx + 5*outHW;"
        >>"float* pOutx6 = pOutx + 6*outHW;"
        >>"float* pOutx7 = pOutx + 7*outHW;"
        ;
      CBLOCK_FOR(loop_y,un_y,"for(int64_t y=0; y<outHeight; ++y)",loop_k);
      loop_y>>"const int64_t i = y * strideHeight - padHeight;">>"KH_BEG_END;";
      if(x0_check_vl)
        loop_y
          >>"vl = vl_x_init;"
          >>NO_SET_VLEN(str_vl)
          ;

      // slightly faster looping for x0... (detect vl <= 0 instead of x0 too high)
      CBLOCK_FOR(loop_x0,un_x0,"for(int64_t x0=0; /*x0<outWidth*/; /*x0+=vl_x_init*/)",loop_y);

      loop_x0>>NO_SET_VLEN(str_vl);

      if(maskW) vrj_induce(loop_x0); // vrj ~ vector of input x values
      loop_x0
        >>"__vr vrsum01 = vzeros;"
        >>"__vr vrsum23 = vzeros;"
        >>"__vr vrsum45 = vzeros;"
        >>"__vr vrsum67 = vzeros;"
        ;
      precalc>>"union PairedFloat const* RESTRICT const krn_krsc_k8 ="
        >>"        (union PairedFloat const*)(buffer + krn_krsc_k8_off);";
      loop_k[".."]>>"union PairedFloat const* RESTRICT krn_gk = krn_krsc_k8 + g*(nk8/group);";
      loop_k["last"]>>"krn_gk += (kBy/2U)*inChannelGroup*kernHW;";

      CBLOCK_FOR(loop_r,un_r,"for (int64_t r = kh_beg; r < kh_end; ++r)",loop_x0);
      loop_r[".."]>>"union PairedFloat const* RESTRICT pKer2 = krn_gk"
        >>"        + kh_beg*kernWidth*inChannelGroup*(kBy/2); // skip some";
      CBLOCK_FOR(loop_s,un_s,"for (int64_t s = 0; s < kernWidth; ++s)",loop_r);
      if(maskW){
        loop_s[".."]>>"__vr vrw = _vel_vor_vsvl(0,vrj,vl); // input row pixels";
        loop_s
          >>"__vm256 vm23 = " VEL_VFMK_mvs_0_TO(vrw,inWidth,vl) ";"
          >>VEL_DECL_VM512( vmP, vm23,vm23, vl)
          ;
        loop_s["last"]>>"vrw = _vel_vaddsl_vsvl(dilationWidth, vrw, vl);";
      }
      CBLOCK_FOR(loop_c,-1,"for (int64_t c = 0; c < inChannelGroup; ++c)",loop_s);
      loop_c.define("DOSUM(VRSUM,VRINP,PAIR)", OSSFMT("VRSUM = "<<(maskW
              ? "_vel_pvfmad_vvsvMvl(VRSUM,PAIR,VRINP,vmP,VRSUM,vl)"
              : "_vel_pvfmad_vvsvl  (VRSUM,PAIR,VRINP,vl)")))
        >>"const float *pIn = pIn_0 + c*inHW + (i+r*dilationHeight)*inWidth"
        >>"                 + x0*strideWidth-padWidth + s*dilationWidth;"
        >>"__vr vrin = _vel_vldu_vssl(4*strideWidth,pIn,vl);"
        >>"__vr vrinP = _vel_vshf_vvvsl(vrin, vrin, VE_VSHUFFLE_YUZU,vl);"
        >>"DOSUM(vrsum01, vrinP, pKer2[0].pair);"
        >>"DOSUM(vrsum23, vrinP, pKer2[1].pair);"
        >>"DOSUM(vrsum45, vrinP, pKer2[2].pair);"
        >>"DOSUM(vrsum67, vrinP, pKer2[3].pair);"
        >>"pKer2+=kBy/2;"
        ;
      loop_x0["induce+write"]
        //>>"_vel_vstu_vssl(vrsum01, 4, pOutx , vl);"
        //>>"_vel_vstl_vssl(vrsum01, 4, pOutx1, vl);"
        //>>"_vel_vstu_vssl(vrsum23, 4, pOutx2, vl);"
        //>>"_vel_vstl_vssl(vrsum23, 4, pOutx3, vl);"
        //>>"pOutx  += vl;"
        //>>"pOutx1 += vl;"
        //>>"pOutx2 += vl;"
        //>>"pOutx3 += vl;"
        >>ve_vstu("pOutx ","vrsum01",str_vl)
        >>ve_vstl("pOutx1","vrsum01",str_vl)
        >>ve_vstu("pOutx2","vrsum23",str_vl)
        >>ve_vstl("pOutx3","vrsum23",str_vl)
        >>OSSFMT("pOutx  += "<<str_vl<<";")
        >>OSSFMT("pOutx1 += "<<str_vl<<";")
        >>OSSFMT("pOutx2 += "<<str_vl<<";")
        >>OSSFMT("pOutx3 += "<<str_vl<<";")
        //>>"_vel_vstu_vssl(vrsum45, 4, pOutx4, vl);"
        //>>"_vel_vstl_vssl(vrsum45, 4, pOutx5, vl);"
        //>>"_vel_vstu_vssl(vrsum67, 4, pOutx6, vl);"
        //>>"_vel_vstl_vssl(vrsum67, 4, pOutx7, vl);"
        //>>"pOutx4 += vl;"
        //>>"pOutx5 += vl;"
        //>>"pOutx6 += vl;"
        //>>"pOutx7 += vl;"
        >>ve_vstu("pOutx4","vrsum45",str_vl)
        >>ve_vstl("pOutx5","vrsum45",str_vl)
        >>ve_vstu("pOutx6","vrsum67",str_vl)
        >>ve_vstl("pOutx7","vrsum67",str_vl)
        >>OSSFMT("pOutx4 += "<<str_vl<<";")
        >>OSSFMT("pOutx5 += "<<str_vl<<";")
        >>OSSFMT("pOutx6 += "<<str_vl<<";")
        >>OSSFMT("pOutx7 += "<<str_vl<<";")
        >>"\n"
        >>"x0 += vl_x_init;"
        ;
      if(x0_check_vl){
        loop_x0["check_vl"]
          >>"vl = outWidth - x0;"
          >>"if( vl <= 0 ) break;"
          >>"vl = vl < vl_x_init? vl: vl_x_init;"
          >>NO_SET_VLEN(str_vl);
          ;
      }else{
        loop_x0["check_vl"]
          >>"if( x0 >= outWidth ) break;"
          ;
      }
      loop_k["bump pOutx"]
        >>"pOutx += (kBy-1) * outHW; // inner increment is outHW, outer wants kBy*outHW"
        ;
      //k = kMax; // could be too naive?
      // "simulate" effect of above JIT loop_k on 'k' (illustrative)
      for(; k<kMax; k+=kBy ) /*no-op*/;
      assert( k == kMax );
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
  fn["exit"]
    >>ve_svob()
    >>"return VEDNN_SUCCESS;"
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
// vim: ts=2 sw=2 et cindent cino=^=l0,\:.5s,=-.5s,N-s,g.5s,b1 cinkeys=0{,0},0),\:,0#,!^F,o,O,e,0=break
