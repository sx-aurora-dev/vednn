#include "cjitConv.hpp"
#include "dllFileAux.hpp"   // strings for declarations, paramString
#include <utility>          // pair
using namespace std;
using namespace cprog;

#define CONST1(var) >>("#define " #var " "+asDec(var))
#define FREE1(var) >>("#undef " #var)

#ifndef VEL_BUG
/** 1 means use extra ve_lvl as workaround for clang bug */
#define VEL_BUG 0
#endif

#if 0
/** for oo in [0,o) output image coord (oh or ow), return the valid
 * [first,second) range for a kernel loop, such that an input pixel
 * in range [0,i), given stride,pad,dilation convolution values
 * and in,out,kernel dimension.
 *
 * \pre in,out dims and convolution parameters are self-consistent.
 * \pre \c d dilation follows libvednn convention (1 for no dilation, 1x input footprint exansion)
 *
 * Ex. (kh_beg,kh_end) = k_beg_end_for_out(y,outHeight, inHeight,
 *     kernHeight,strideHeight,padHeight,dilationHeight).
 */
static pair<int64_t,int64_t> k_beg_end_for_out( int64_t const oo, int64_t const o, int64_t const i,
    int64_t const k, int64_t const s, int64_t const p, int64_t const d ){
  const int64_t ii = oo * s - p;
  const int64_t kTmp = d - ii - 1;
  pair<int64_t,int64_t> ret;
  ret.second = 0;
  ret.first = ( ii>=0? 0: kTmp/d );
  if (ii < i){
    ret.second = (i + kTmp) / d;
    if (ret.second >= k) ret.second = k;
  }
  return ret;
  assert( ret.first >=0 && ret.second <= k );
}
#endif

/** Here we block loop_s (kernWidth), by 2 (can be good for large kernels).
 * XXX add loop_c blocking from Fwd4 here (and to Fwd2?) */
DllFile cjitConvolutionForward5( struct param const* const p )
{
  ostringstream oss; // scratchpad for OSSFMT etc
  string const impl = "cjitConvFwd5";
  int const verbose=0;
  DllFile df; // return value
  std::string parmstr = paramString(p);
  df.basename = impl+"_"+parmstr;
  cout<<impl<<" : df.basename = "<<df.basename<<endl;

  // we use intrinsics.  suffix matches build recipe in "bin.mk"
  df.suffix = "-vi.c";

  Cunit pr("program");
  pr.v = verbose;     // default is quite lengthy!

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
    >>"#if "<<asDec(VEL_BUG)
    >>"// sometimes enabling this can fix 'wrong result'"
    >>"//        Simple test case: jitconv -p mb64ih3ic1oc1_kh3ph0"
    >>"#define NO_SET_VLEN( VLEN ) _ve_lvl(VLEN)"
    >>""
    >>"#else // but pure vel intrinsics should do nothing"
    >>"#define NO_SET_VLEN( VLEN ) do{}while(0)"
    >>"#endif"
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
  int64_t const vl_x_init = ve_vlen_suggest( outWidth );
  int64_t const kByMax = 1; // kByMax=1,2 have been OK

#define DEF(VAR) def(#VAR, VAR)
  fn.DEF(batch).DEF(group).DEF(inChannel).DEF(inHeight).DEF(inWidth);
  fn.DEF(outChannel).DEF(outHeight).DEF(outWidth).DEF(kernHeight).DEF(kernWidth);
  fn.DEF(strideHeight).DEF(strideWidth).DEF(padHeight).DEF(padWidth).DEF(dilationHeight);
  fn.DEF(dilationWidth).DEF(inChannelGroup).DEF(outChannelGroup);
  fn.DEF(inHW).DEF(kernHW).DEF(outHW).DEF(kByMax).DEF(vl_x_init);

  auto& fn_ptrs = fn["ptrs"];
  fn_ptrs>>"float const * restrict pIn  = pDataIn;"
    >>"float const * restrict pKernel = pDataKernel;"
    >>"float * restrict const pOut = pDataOut;"
    ;

  auto& fn_vec_init =
    fn["vec_init"]
    >>"// TODO VLEN-->vl_x_init (also a compile-time const, but more robust to future change of vlen)"
    >>"// TODO fused-loop opt to handle outWidth much smaller than VLEN (i.e. low vl_x_init)"
    >>"NO_SET_VLEN(vl_x_init);"
    >>"const __vr vzeros = _vel_vbrds_vsl(0.0f, vl_x_init );"
    >>"const __vr vrseq = _vel_vseq_vl(vl_x_init);"
    >>"int64_t vl = vl_x_init;"
    >>"NO_SET_VLEN(vl);"
    >>"float * restrict pOutx = pDataOut;"
    ;
  vrj_init(fn_vec_init);

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
  int64_t kBy = 1;    // 1,2,4,... kByMax
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
      CBLOCK_SCOPE(loop_k,"for(; k<kMax; k+=kBy)",pr,scope_kMax);
      loop_k
        >>"CHK(pOutx == pOut + outGroupOffset + (n * outChannel + k) * outHW);"
        >>"const float * restrict pKern_gk = pKernel + kernGroupOffset"
        >>"                                + (k * inChannelGroup + 0/*c*/) * kernHW;"
        ;
      CBLOCK_SCOPE(loop_y,"for(int64_t y=0 ; y<outHeight; ++y)",pr,loop_k);
      loop_y
        >>"const int64_t i = y * strideHeight - padHeight;"
        ;
#if 0
      const int64_t imin = (0          )*strideHeight - padHeight; // range of any value i
      const int64_t imax = (outHeight-1)*strideHeight - padHeight; // cf. jmin,max for Width
      int64_t const jmin = (0         )*strideWidth - padWidth; // range of any value in vrj
      int64_t const jmax = (outWidth-1)*strideWidth - padWidth; // cf. imin,imax for Height
      auto kbeMin = k_beg_end_for_out(0,          outHeight, inHeight,
          kernHeight,strideHeight,padHeight,dilationHeight);
      auto kbeMax = k_beg_end_for_out(outHeight-1,outHeight, inHeight,
          kernHeight,strideHeight,padHeight,dilationHeight);
      // Following may be counterintuitive...
      assert( kbeMin.first  >= kbeMax.first  );
      assert( kbeMin.second >= kbeMax.second );
      int kh_always_full_range =
        (kbeMin.first == 0 && kbeMax.second == kernHeight);
#endif
      loop_y
        >>"int64_t kh_end=0;"
        >>"const int64_t kh_tmp = dilationHeight-1 -i;"
        >>"const int64_t kh_beg= (i>=0? 0: kh_tmp / dilationHeight);"
        >>"if (i < inHeight){"
        >>"  kh_end = (inHeight + kh_tmp) / dilationHeight;"
        >>"  if (kh_end >= kernHeight) kh_end = kernHeight;"
        >>"}"

        >>"vl = vl_x_init;"
        >>"NO_SET_VLEN(vl);"
        ;
      loop_y>>"// x0+=VLEN orig., but x0+=vl_x_init also compile-time const";
      loop_y>>"// TODO: eliminate once-only loop entirely, for clarity XXX";
      CBLOCK_SCOPE(loop_x0,"for(int64_t x0=0 ; x0<outWidth; x0+=vl_x_init)",pr,loop_y);
      loop_x0
        >>"vl = outWidth - x0 < vl_x_init ? outWidth - x0: vl_x_init;"
        >>"NO_SET_VLEN(vl);"
        ;
      vrj_induce(loop_x0); // vrj ~ vector of input x values
      CBLOCK_SCOPE(loop_r,"for (int64_t r = kh_beg; r < kh_end; ++r)",pr,loop_x0);
      loop_r[".."]<<"#pragma clang loop unroll(disable)";
      auto& good_h=loop_r["good_h"];
      good_h//loop_r
        >>"__vr vrw = vrj;"
        ;
      int64_t s=0, have_sBy=0, have_vrsum=0; // XXX kernel of size == 1 is separate case.
      if((kernWidth & 0x01) != 0){
        loop_x0>>"__vr vrsum = vzeros;";
        have_vrsum = 1;
        CBLOCK_SCOPE(loop_s,"",pr,good_h/*loop_r*/);
        loop_s
          >>"int const s=0;"
          >>"__vm256 vm23 = " VEL_VFMK_mvs_0_TO(vrw,inWidth,vl) ";";
          ;
        // TODO: block next loop
        CBLOCK_SCOPE(loop_c,"for (int64_t c = 0; c < inChannelGroup; ++c)",pr,loop_s);
        loop_c[".."]>>"#pragma clang loop unroll(disable)";
        loop_c
          >>"const float *pIn = pIn_0 + c*inHW + (i+r*dilationHeight)*inWidth"
          >>"                 + x0*strideWidth-padWidth + s*dilationWidth;"
          >>"const float *pKerValue = pKern_gk + c*kernHW + r*kernWidth +s;"
          >>"__vr vrin = _vel_vldu_vssl(4*strideWidth,pIn, vl);"
          >>"vrsum = _vel_vfmads_vvsvmvl(vrsum, *pKerValue, vrin, vm23, vrsum, vl);"
          ;
        loop_s["induce-vrw"] // BEFORE the '}' of loop_s /**/loop_s/body/induce-vrw
          >>"vrw = _vel_vaddsl_vsvl(dilationWidth,  vrw, vl) ; // <--- vector induced"
          ;
        s=1;  // mirror loop_s progress
      }
#if 0 // why do the following compares fail XXX ? signed-ness of dilationWidth has no effect?
      >>"// no effect: _vel_pfchvl(pIn_0 + 0*inHW + (i+r*dilationHeight, vl)*inWidth + x0*strideWidth-padWidth + s*dilationWidth, 4*inHW;"
        // vrw2 = vrw + dilationWidth, so SHOULD be able to "just" adjust comparison thresholds.
        // so vrw2>=0 means vrw+dw>=0, means vrw>=(0-dw)
        >>"__vm256 vm2 = _vel_vfmkl_mcvl(VECC_GE, vrw, vl);"
        >>"__vm256 vm4 = _vel_vfmkl_mcvl(VECC_GE, _ve_vcmpsl_vsv(0-dilationWidth,vrw, vl));"
        >>"__vm256 vm23 = _vel_vfmkl_mcvml(VECC_IG, _ve_vcmpsl_vsv(inWidth,vrw, vl),vm2);"
        >>"__vm256 vm45 = _vel_vfmkl_mcvml(VECC_IG, _ve_vcmpsl_vsv(inWidth-dilationWidth,vrw, vl),vm4);"
        >>"// no effect: #pragma clang loop unroll(8)"
#endif
      auto& x0_end = loop_x0["induce+write"];
      if(s < kernWidth){
        int const sBy = 2;
        have_sBy = sBy;
        assert( sBy==2 );
        good_h["+sBy2"] CONST1(sBy)
          //>>"vrsum = _vel_vshf_vvvsl(vrsum,vzeros,VE_VSHUFFLE_YUZL, vl); /*nilpotent*/"
          ;
        CBLOCK_SCOPE(loop_sBy,"for (int64_t s="+asDec(s)+"; s<kernWidth; s+=sBy)",pr,/*loop_r*/good_h);
        loop_sBy[".."]<<"#pragma clang loop unroll(disable)";
        loop_sBy // for s: 0<=w and w<=inWidth; s+1 adds dilationWidth to w thresholds
          >>"__vr vrw2 = _vel_vaddsl_vsvl(dilationWidth,  vrw, vl);"
          >>"__vm256 vm23 = " VEL_VFMK_mvs_0_TO(vrw,inWidth,vl) ";"
          >>"__vm256 vm45 = " VEL_VFMK_mvs_0_TO(vrw2,inWidth,vl) ";"
          >>VEL_DECL_VM512( vmP, vm23,vm45, vl )
          >>"vrw = _vel_vaddsl_vsvl(2*dilationWidth,  vrw, vl);"
          ;
        loop_x0>>"__vr vrPsum_1 = vzeros;";
#if 1
        // actually a slight slowdown for one test:
        // k11ic100pad          implJ    9       unroll_cjitConvFwd5 *      1-x   313.050 ms DIFF 0.001161
        // k11ic100pad          implJ    4              cjitConvFwd5 |      1-x   313.678 ms DIFF 0.001161
        // k11ic100pad          impl*    0              libvednn-std |      1-x   345.482 ms DIFF 0.001380
        // k11ic100pad          implJ    0              cjitConvFwd1 |      1-x   410.135 ms DIFF 0.001380
        // k11ic100pad          implJ    3              cjitConvFwd4 |      1-x   615.810 ms DIFF 0.001169
        int c=0;
        if( (inChannelGroup&0x01) != 0 ){
          loop_sBy
            >>"const float *pIn = pIn_0 + 0*inHW + (i+r*dilationHeight)*inWidth"
            >>"                 + x0*strideWidth-padWidth + s*dilationWidth;"
            >>"const float *pKerValue = pKern_gk + 0*kernHW + r*kernWidth +s;"
            >>"__vr vrin  = _vel_vldu_vssl(4*strideWidth,pIn, vl);"
            >>"__vr vrin2 = _vel_vldu_vssl(4*strideWidth,pIn+1*dilationWidth, vl);"
            >>"/*P*/ __vr vrinP = _vel_vshf_vvvsl(vrin, vrin2, VE_VSHUFFLE_YUZU, vl);"
            >>"/*P*/ const uint64_t kerValue01 = _vel_pack_f32p(pKerValue,pKerValue+1);" // !!!
            >>"/*P*/ vrPsum_1 = _vel_pvfmad_vvsvMvl(vrPsum_1, kerValue01, vrinP, vmP, vrPsum_1, vl);"
            ;
          c=1;
        }
        if( c<inChannelGroup ){
          CBLOCK_SCOPE(loop_c_sBy_2,"for (int64_t c ="+asDec(c)+"; c < inChannelGroup; c+=2)",pr,loop_sBy);
          loop_c_sBy_2[".."]<<"#pragma clang loop unroll(disable)";
          loop_x0>>"__vr vrPsum_2 = vzeros;";
          loop_c_sBy_2
            >>"const float *pIn = pIn_0 + c*inHW + (i+r*dilationHeight)*inWidth"
            >>"                 + x0*strideWidth-padWidth + s*dilationWidth;"
            >>"const float *pKerValue = pKern_gk + c*kernHW + r*kernWidth +s;"
            >>"__vr vrin1u  = _vel_vldu_vssl(4*strideWidth,pIn, vl);"
            >>"__vr vrin1l = _vel_vldlzx_vssl(4*strideWidth,pIn+1*dilationWidth, vl);"
            >>"__vr vrin2u = _vel_vldu_vssl(4*strideWidth,pIn+inHW, vl);"
            >>"__vr vrin2l = _vel_vldlzx_vssl(4*strideWidth,pIn+inHW+1*dilationWidth, vl);"
            >>"/*P*/ __vr vrinP_1 = _vel_vshf_vvvsl(vrin1u, vrin1l, VE_VSHUFFLE_YUZL, vl);"
            >>"/*P*/ const uint64_t ker_1 = _vel_pack_f32p(pKerValue,pKerValue+1);" // but maybe align 4!
            >>"/*P*/ vrPsum_1 = _vel_pvfmad_vvsvMvl(vrPsum_1, ker_1, vrinP_1, vmP, vrPsum_1, vl);"
            >>"pKerValue+=kernHW;"
            >>"/*P*/ __vr vrinP_2 = _vel_vshf_vvvsl(vrin2u, vrin2l, VE_VSHUFFLE_YUZL, vl);"
            >>"/*P*/ const uint64_t ker_2 = _vel_pack_f32p(pKerValue,pKerValue+1);"
            >>"/*P*/ vrPsum_2 = _vel_pvfmad_vvsvMvl(vrPsum_2, ker_2, vrinP_2, vmP, vrPsum_2, vl);"
            ;
          x0_end>>"vrPsum_1 = _vel_pvfadd_vvvl(vrPsum_1,vrPsum_2, vl);"; // final fold of cBy2
        }
#else
        // seems good for large kernels
        // k11ic100pad          implJ    9       unroll_cjitConvFwd5 *      1-x   308.484 ms DIFF 0.001218
        // k11ic100pad          implJ    4              cjitConvFwd5 |      1-x   310.945 ms DIFF 0.001218
        // k11ic100pad          impl*    0              libvednn-std |      1-x   345.562 ms DIFF 0.001380
        // k11ic100pad          implJ    0              cjitConvFwd1 |      1-x   410.134 ms DIFF 0.001380
        // k11ic100pad          implJ    3              cjitConvFwd4 |      1-x   615.861 ms DIFF 0.001169
        CBLOCK_SCOPE(loop_c_sBy,"for (int64_t c = 0; c < inChannelGroup; ++c)",pr,loop_sBy);
        loop_c_sBy
          >>"const float *pIn = pIn_0 + c*inHW + (i+r*dilationHeight)*inWidth"
          >>"                 + x0*strideWidth-padWidth + s*dilationWidth;"
          >>"const float *pKerValue = pKern_gk + c*kernHW + r*kernWidth +s;"
          >>"__vr vrin  = _vel_vldu_vssl(4*strideWidth,pIn, vl);"
          >>"__vr vrin2 = _vel_vldu_vssl(4*strideWidth,pIn+1*dilationWidth, vl);"
          >>"/*P*/ __vr vrinP = _vel_vshf_vvvsl(vrin, vrin2, VE_VSHUFFLE_YUZU, vl);"
          >>"/*P*/ const uint64_t kerValue01 = _vel_pack_f32p(pKerValue,pKerValue+1);" // !!!
          >>"/*P*/ vrPsum_1 = _vel_pvfmad_vvsvMvl(vrPsum_1, kerValue01, vrinP, vmP, vrPsum_1, vl);"
          ;
#endif
        good_h["-sBy2"] FREE1(sBy);
      }
      //  SEPARATE pvfmad and vfmads registers are OK. vfmads ZEROES!
      //  IN THEORY, vfmads FIRST and then pvfmad OUGHT to work, but ??
      if(have_sBy){ // add both halves of vrPsum_1 to vrsum/upper
        if(have_vrsum) x0_end>>"vrsum = _vel_vfadds_vvvl(vrsum,vrPsum_1, vl);";
        else           x0_end>>"__vr vrsum = vrPsum_1;";
        x0_end
          >>"__vr vrswap = _vel_vshf_vvvsl(vrPsum_1,vzeros,VE_VSHUFFLE_YLZL, vl);"
          >>"vrsum = _vel_vfadds_vvvl(vrsum,vrswap, vl);"
          ;
      }
      x0_end
        >>"_vel_vstu_vssl(vrsum, 4, pOutx, vl);"
        >>"pOutx += vl; // visible speedup cf. outIndex+=vl"
        >>"//"<<CSTR(printf(" k %ld vl %-3ld outIndex=%ld\n",(long)k,(long)vl,(long)(pOutx-pOut));)
        ;
      //k = kMax;
      // "simulate" effect of above JIT loop_k on 'k' (illustrative)
      for(; k<kMax; k+=kBy ) /*no-op*/;
      assert( k == kMax );
      scope_kMax["~kMax"]>>"CHK(pOutx == pOut + outGroupOffset + (n*outChannel+kMax)*outHW);";
    }
    loop_g[string("kBy-")+hexdec(kBy)]
      >>"#endif // unroll by "+hexdec(kBy)<<", exit w/ k="<<hexdec(k)
      <<" of outChannelGroup="<<hexdec(outChannelGroup)
      ;
    if(kByMax==1){ assert( k>=outChannelGroup ); }
  }
  if( k<outChannelGroup ){
    kBy = 2;
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
        >>"CHK(pOutx == pOut + outGroupOffset + (n * outChannel + k) * outHW);"
        >>"float* pOutx1 = pOutx + outHW;"
        >>"const float * restrict pKern_gk = pKernel + kernGroupOffset"
        >>"                                + (k * inChannelGroup + 0/*c*/) * kernHW;"
        ;
      CBLOCK_SCOPE(loop_y,"for(int64_t y=0; y<outHeight; ++y)",pr,loop_k);
      loop_y
        >>"const int64_t i = y * strideHeight - padHeight;"
        >>""
        >>"int64_t kh_end=0;"
        >>"const int64_t kh_tmp = dilationHeight-i-1;"
        >>"const int64_t kh_beg= (i>=0? 0: kh_tmp / dilationHeight);"
        >>"if (i < inHeight){"
        >>"  kh_end = (inHeight + kh_tmp) / dilationHeight;"
        >>"  if (kh_end >= kernHeight) kh_end = kernHeight;"
        >>"}"
        >>""
        >>"vl = vl_x_init;"
        >>"NO_SET_VLEN(vl);"
        ;
      // slightly faster looping for x0... (detect vl <= 0 instead of x0 too high)
      CBLOCK_SCOPE(loop_x0,"for(int64_t x0=0; /*x0<outWidth*/; /*x0+=vl_x_init*/)",pr,loop_y);
      vrj_induce(loop_x0); // vrj ~ vector of input x values
      loop_x0
        >>"__vr vrsum_1 = vzeros; /* final summation register */"
        ;
      // Here we MIGHT? have 1 vmP, so once again we have unroll limits:
      int const max_unroll_outer = 7; //(noMaskW ? 8: 6); //maybe change the '8'?
      int const un_s = (kernWidth>max_unroll_outer? max_unroll_outer: kernWidth);
      int const un_r = max_unroll_outer/un_s;
      CBLOCK_SCOPE(loop_r,OSSFMT("#pragma unroll("<<un_r<<")\n"
            "for (int64_t r = kh_beg; r < kh_end; ++r)"), pr,loop_x0);
      CBLOCK_SCOPE(loop_s,OSSFMT("#pragma unroll("<<un_s<<")\n"
            "for (int64_t s = 0; s < kernWidth; ++s)"), pr,loop_r);
      //CBLOCK_SCOPE(loop_r,"for (int64_t r = kh_beg; r < kh_end; ++r)",pr,loop_x0);
      //CBLOCK_SCOPE(loop_s,"for (int64_t s = 0; s < kernWidth; ++s)",pr,loop_r);
      loop_s[".."]>>"__vr vrw = vrj;";
      loop_s>>"__vm256 vm23 = " VEL_VFMK_mvs_0_TO(vrw,inWidth,vl) ";"
        >>VEL_DECL_VM512( vmP, vm23,vm23, vl );
      // c=0..inChannelGroup blocking
      auto& x0_end = loop_x0["induce+write"];
      int c=0;
      int have_cBy=0;
      for(int ccBy=4; ccBy>1; --ccBy){
        if( inChannelGroup%ccBy == 0 ){
          have_cBy = ccBy;
          break;
        }
      }
      if(!have_cBy) have_cBy = (inChannelGroup<4? inChannelGroup: 4);
      if(have_cBy)
      {
        int const cBy=have_cBy;
        loop_s>>"int const cBy="+asDec(cBy)+";";
        CBLOCK_SCOPE(loop_c,"for (int64_t c="+asDec(c)+"; c<"+asDec(inChannelGroup/cBy*cBy)+"; c+="+asDec(cBy)+")",pr,loop_s);
        for ( ; c<inChannelGroup/cBy*cBy; c+=cBy) /*NOP*/ ; // at what 'c' does loop_c exit?
        assert( c == inChannelGroup/cBy*cBy );
        loop_c
          >>"const float *pIn = pIn_0 + c*inHW + (i+r*dilationHeight)*inWidth"
          >>"                 + x0*strideWidth-padWidth + s*dilationWidth;"
          >>"const float *pKerValue = pKern_gk + c*kernHW + r*kernWidth +s;"
          //>>"const uint64_t ker01 = _vel_pack_f32p(pKerValue,"
          //>>"                                     pKerValue + inChannelGroup*kernHW);"
          //>>"__vr vrin = _vel_vldu_vssl(4*strideWidth,pIn, vl);"
          //>>"__vr vrinP = _vel_vshf_vvvsl(vrin, vrin, VE_VSHUFFLE_YUZU, vl);"
          //>>"vrsum01 = _vel_pvfmad_vvsvMvl(vrsum01, ker01, vrinP, vmP, vrsum01, vl);"
          ;
        string zero_vrsum_CBY =
          "__vr vrsum_CBY = _vel_vor_vsvl(0,vzeros,vl);";
        string update_vrsum_CBY =
          "\n/*cByCBY*/ __vr vrin_CBY = _vel_vldu_vssl(4*strideWidth,pIn+(CBY-1)*inHW, vl);"
          "\nconst uint64_t ker_CBY = _vel_pack_f32p("
          "\n                                pKerValue+(CBY-1)*kernHW,"
          "\n                                pKerValue+(CBY-1)*kernHW + inChannelGroup*kernHW);"
          "\nvrin_CBY = _vel_vshf_vvvsl(vrin_CBY, vrin_CBY, VE_VSHUFFLE_YUZU, vl);"
          "\nvrsum_CBY = _vel_pvfmad_vvsvMvl(vrsum_CBY, ker_CBY, vrin_CBY, vmP, vrsum_CBY, vl);"
          ;
        string fold_vrsum_CBY =
          "vrsum_1 = _vel_pvfadd_vvvl(vrsum_1,vrsum_CBY, vl);"; // XXX ovlp sums sometimes!
#if 0
        if(cBy>=2){
          loop_x0 >>multiReplace("CBY","2",zero_vrsum_CBY);
          loop_c >>multiReplace("CBY","2",update_vrsum_CBY);
          x0_end >>multiReplace("CBY","2",fold_vrsum_CBY);
        }
        if(cBy>=3){
          loop_x0 >>multiReplace("CBY","3",zero_vrsum_CBY);
          loop_c >>multiReplace("CBY","3",update_vrsum_CBY);
          x0_end >>multiReplace("CBY","3",fold_vrsum_CBY);
        }
#else
        for(int ccBy=1; ccBy<=cBy; ++ccBy){
          string cBy_numeric=asDec(ccBy);
          if(ccBy>1) loop_x0>>multiReplace("CBY",cBy_numeric, zero_vrsum_CBY);
          loop_c >>multiReplace("CBY",cBy_numeric, update_vrsum_CBY);
          // simple folding back into vrsum_1
          //if(ccBy>1) x0_end >>multiReplace("CBY",cBy_numeric, fold_vrsum_CBY);
        }
        // optimize the multiple terminal _vel_pvfadd_vvv summations l(binary-tree-like add, vl)
        // (there is now a jitconv.hpp subroutine to binary-tree add a vector of registers)
        {
          unsigned d=1, d2;
          for( ; d<cBy; d=d2 ){
            d2 = d<<1;
            unsigned i=0;
            for( ; i<cBy; i+=d2 ){
              if(i+d<cBy){
                string vrA = "vrsum_"+asDec(i+1);
                string vrB = "vrsum_"+asDec(i+d+1);
                x0_end>>vrA+" = _vel_pvfadd_vvvl("+vrA+","+vrB+", vl);";
              }
            }
          }
        }// end multiple pvfadd
#endif
      }
      // leftovers (here for now)
      if(c<inChannelGroup){
        // original : when clang unrolled this, it reused same __vr, creating dependencies,
        // so it is better to hand-unroll, using separate summation registers, as above.
        CBLOCK_SCOPE(loop_c,"for (int64_t c="+asDec(c)+"; c<inChannelGroup; ++c)",pr,loop_s);
        c = inChannelGroup; // where do we end up after above loop?
        loop_c
          >>"const float *pIn = pIn_0 + c*inHW + (i+r*dilationHeight)*inWidth"
          >>"                 + x0*strideWidth-padWidth + s*dilationWidth;"
          >>"__vr vrin = _vel_vldu_vssl(4*strideWidth,pIn, vl);"
          >>"const float *pKerValue = pKern_gk + c*kernHW + r*kernWidth +s;"
          >>"const uint64_t ker01 = _vel_pack_f32p(pKerValue,"
          >>"                                     pKerValue + inChannelGroup*kernHW);"
          >>"__vr vrinP = _vel_vshf_vvvsl(vrin, vrin, VE_VSHUFFLE_YUZU, vl);"
          >>"vrsum_1 = _vel_pvfmad_vvsvMvl(vrsum_1, ker01, vrinP, vmP, vrsum_1, vl);"
          ;
      }
      assert( c==inChannelGroup );
      loop_s["induce-vrw"] // BEFORE the '}' of loops_s /**/loop_s/body/induce-vrw
        >>"vrw = _vel_vaddsl_vsvl(dilationWidth, vrw, vl); // <--- vector induced"
        ;
      x0_end
        >>"_vel_vstu_vssl(vrsum_1, 4, pOutx , vl);"
        >>"_vel_vstl_vssl(vrsum_1, 4, pOutx1, vl);"
        >>"pOutx  += vl;"
        >>"pOutx1 += vl;"
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
      scope_kMax["~kMax"]>>"CHK(pOutx == pOut + outGroupOffset + (n*outChannel+kMax)*outHW);";
    }
    loop_g[string("kBy-")+hexdec(kBy)]
      >>"#endif // unroll by "+hexdec(kBy)<<", exit w/ k="<<hexdec(k)
      <<" of outChannelGroup="<<hexdec(outChannelGroup)
      ;
    if(kByMax==2){ assert( k>=outChannelGroup ); }
  }
  if( k<outChannelGroup ){
    kBy = 4;
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
        >>"const float * restrict pKern_gk = pKernel + kernGroupOffset"
        >>"                                + (k * inChannelGroup + 0/*c*/) * kernHW;"
        ;
      CBLOCK_SCOPE(loop_y,"for(int64_t y=0; y<outHeight; ++y)",pr,loop_k);
      loop_y
        >>"const int64_t i = y * strideHeight - padHeight;"
        >>"int64_t kh_end=0;"
        >>"const int64_t kh_tmp = dilationHeight-i-1;"
        >>"const int64_t kh_beg= (i>=0? 0: kh_tmp / dilationHeight);"
        >>"if (i < inHeight){"
        >>"  kh_end = (inHeight + kh_tmp) / dilationHeight;"
        >>"  if (kh_end >= kernHeight) kh_end = kernHeight;"
        >>"}"
        >>"vl = vl_x_init;"
        >>"NO_SET_VLEN(vl);"
        ;
      // slightly faster looping for x0... (detect vl <= 0 instead of x0 too high)
      CBLOCK_SCOPE(loop_x0,"for(int64_t x0=0; /*x0<outWidth*/; /*x0+=vl_x_init*/)",pr,loop_y);
      vrj_induce(loop_x0); // vrj ~ vector of input x values
      loop_x0
        >>"__vr vrsum01 = vzeros;"
        >>"__vr vrsum23 = vzeros;"
        ;
      CBLOCK_SCOPE(loop_r,"for (int64_t r = kh_beg; r < kh_end; ++r)",pr,loop_x0);
      loop_r
        >>"__vr vrw = vrj;"
        ;
      CBLOCK_SCOPE(loop_s,"for (int64_t s = 0; s < kernWidth; ++s)",pr,loop_r);
      loop_s>>"__vm256 vm23 = " VEL_VFMK_mvs_0_TO(vrw,inWidth,vl) ";";
      CBLOCK_SCOPE(loop_c,"for (int64_t c = 0; c < inChannelGroup; ++c)",pr,loop_s);
      loop_c
        >>"const float *pIn = pIn_0 + c*inHW + (i+r*dilationHeight)*inWidth"
        >>"                 + x0*strideWidth-padWidth + s*dilationWidth;"
        >>"__vr vrin = _vel_vldu_vssl(4*strideWidth,pIn, vl);"
        >>"const float *pKerValue = pKern_gk + c*kernHW + r*kernWidth +s;"
        >>"vrin = _vel_vmrg_vvvml(vzeros, vrin, vm23, vl);"
        >>"const uint64_t kerValue01 = _vel_pack_f32p(pKerValue,"
        >>"    pKerValue + inChannelGroup*kernHW);"
        >>"__vr vrinP = _vel_vshf_vvvsl(vrin, vrin, VE_VSHUFFLE_YUZU, vl);"
        >>"const uint64_t kerValue23 = _vel_pack_f32p(pKerValue + 2 * inChannelGroup * kernHW,"
        >>"    pKerValue + 3 * inChannelGroup * kernHW);"
        >>"vrsum01 = _vel_pvfmad_vvsvl(vrsum01, kerValue01, vrinP, vl);"
        >>"vrsum23 = _vel_pvfmad_vvsvl(vrsum23, kerValue23, vrinP, vl);"
        ;
      loop_s["induce-vrw"] // BEFORE the '}' of loops_s /**/loop_s/body/induce-vrw
        >>"vrw = _vel_vaddsl_vsvl(dilationWidth, vrw, vl); // <--- vector induced"
        ;
      loop_x0["induce+write"]
        >>"_vel_vstu_vssl(vrsum01, 4, pOutx , vl);"
        >>"_vel_vstl_vssl(vrsum01, 4, pOutx1, vl);"
        >>"_vel_vstu_vssl(vrsum23, 4, pOutx2, vl);"
        >>"_vel_vstl_vssl(vrsum23, 4, pOutx3, vl);"
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
      scope_kMax["~kMax"]
        //>>"CHK(pOutx == pOut + outGroupOffset + (n * outChannel + kMax) * outHW);"
        FREE1(kBy)
        FREE1(kMax)
        ;
    }
    loop_g[string("kBy-")+hexdec(kBy)]
      >>"#endif // unroll by "+hexdec(kBy)<<", exit w/ k="<<hexdec(k)
      <<" of outChannelGroup="<<hexdec(outChannelGroup)
      ;
    if(kByMax==4){ assert( k>=outChannelGroup ); }
  }
  if( k<outChannelGroup ){
    kBy = 8;
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
        >>"const float * restrict pKern_gk = pKernel + kernGroupOffset"
        >>"                                + (k * inChannelGroup + 0/*c*/) * kernHW;"
        ;
      CBLOCK_SCOPE(loop_y,"for(int64_t y=0; y<outHeight; ++y)",pr,loop_k);
      loop_y
        >>"const int64_t i = y * strideHeight - padHeight;"
        >>" "
        >>"int64_t kh_end=0;"
        >>"const int64_t kh_tmp = dilationHeight-i-1;"
        >>"const int64_t kh_beg= (i>=0? 0: kh_tmp / dilationHeight);"
        >>"if (i < inHeight){"
        >>"  kh_end = (inHeight + kh_tmp) / dilationHeight;"
        >>"  if (kh_end >= kernHeight) kh_end = kernHeight;"
        >>"}"
        >>" "
        >>"vl = vl_x_init;"
        >>"NO_SET_VLEN(vl);"
        ;
      // slightly faster looping for x0... (detect vl <= 0 instead of x0 too high)
      CBLOCK_SCOPE(loop_x0,"for(int64_t x0=0; /*x0<outWidth*/; /*x0+=vl_x_init*/)",pr,loop_y);
      vrj_induce(loop_x0); // vrj ~ vector of input x values
      loop_x0
        >>"__vr vrsum01 = vzeros;"
        >>"__vr vrsum23 = vzeros;"
        >>"__vr vrsum45 = vzeros;"
        >>"__vr vrsum67 = vzeros;"
        >>" "
        >>"const float *pKerValue = pKernel + kernGroupOffset + k * inChannelGroup * kernHeight * kernWidth;"
        ;
      CBLOCK_SCOPE(loop_r,"for (int64_t r = kh_beg; r < kh_end; ++r)",pr,loop_x0);
      loop_r
        >>"__vr vrw = vrj;"
        ;
      CBLOCK_SCOPE(loop_s,"for (int64_t s = 0; s < kernWidth; ++s)",pr,loop_r);
      loop_s>>"__vm256 vm23 = " VEL_VFMK_mvs_0_TO(vrw,inWidth,vl) ";";
      CBLOCK_SCOPE(loop_c,"for (int64_t c = 0; c < inChannelGroup; ++c)",pr,loop_s);
      loop_c
        >>"const float *pIn = pIn_0 + c*inHW + (i+r*dilationHeight)*inWidth"
        >>"                 + x0*strideWidth-padWidth + s*dilationWidth;"
        >>"__vr vrin = _vel_vldu_vssl(4*strideWidth,pIn, vl);"
        >>"vrin = _vel_vmrg_vvvml(vzeros, vrin, vm23, vl);"
        >>"__vr vrinP = _vel_vshf_vvvsl(vrin, vrin, VE_VSHUFFLE_YUZU, vl);"
        >>"const float *pKerValue = pKern_gk + c*kernHW + r*kernWidth +s;"
        >>"const uint64_t kerValue01 = _vel_pack_f32p("
        >>"    pKerValue,"
        >>"    pKerValue + 1 * inChannelGroup*kernHW);"
        >>"const uint64_t kerValue23 = _vel_pack_f32p("
        >>"    pKerValue + 2 * inChannelGroup * kernHW,"
        >>"    pKerValue + 3 * inChannelGroup * kernHW);"
        >>"const uint64_t kerValue45 = _vel_pack_f32p("
        >>"    pKerValue + 4 * inChannelGroup * kernHW,"
        >>"    pKerValue + 5 * inChannelGroup * kernHW);"
        >>"const uint64_t kerValue67 = _vel_pack_f32p("
        >>"    pKerValue + 6 * inChannelGroup * kernHW,"
        >>"    pKerValue + 7 * inChannelGroup * kernHW);"
        >>"vrsum01 = _vel_pvfmad_vvsvl(vrsum01, kerValue01, vrinP, vl);"
        >>"vrsum23 = _vel_pvfmad_vvsvl(vrsum23, kerValue23, vrinP, vl);"
        >>"vrsum45 = _vel_pvfmad_vvsvl(vrsum45, kerValue45, vrinP, vl);"
        >>"vrsum67 = _vel_pvfmad_vvsvl(vrsum67, kerValue67, vrinP, vl);"
        ;
      loop_s["induce-vrw"] // BEFORE the '}' of loops_s /**/loop_s/body/induce-vrw
        >>"vrw = _vel_vaddsl_vsvl(dilationWidth, vrw, vl); // <--- vector induced"
        ;
      loop_x0["induce+write"]
        >>"_vel_vstu_vssl(vrsum01, 4, pOutx , vl);"
        >>"_vel_vstl_vssl(vrsum01, 4, pOutx1, vl);"
        >>"_vel_vstu_vssl(vrsum23, 4, pOutx2, vl);"
        >>"_vel_vstl_vssl(vrsum23, 4, pOutx3, vl);"
        >>"pOutx  += vl;"
        >>"pOutx1 += vl;"
        >>"pOutx2 += vl;"
        >>"pOutx3 += vl;"
        >>"_vel_vstu_vssl(vrsum45, 4, pOutx4, vl);"
        >>"_vel_vstl_vssl(vrsum45, 4, pOutx5, vl);"
        >>"_vel_vstu_vssl(vrsum67, 4, pOutx6, vl);"
        >>"_vel_vstl_vssl(vrsum67, 4, pOutx7, vl);"
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
    if(kByMax==4){ assert( k>=outChannelGroup ); }
  }
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
    cout<<string(80,'-')<<pr.str() <<string(80,'-')<<endl;
    cout<<string(80,'-')<<pr.tree()<<string(80,'-')<<endl;
  }
  df.code = pr.str();
  return df;
}
// vim: ts=2 sw=2 et cindent cino=^=l0,\:.5s,=-.5s,N-s,g.5s,b1 cinkeys=0{,0},0),\:,0#,!^F,o,O,e,0=break
