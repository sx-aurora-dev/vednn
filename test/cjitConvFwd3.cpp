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

// KH_LIMITS 0 is Forward2 version (precalc valid khBeg,khEnd)
// KH_LIMITS 1 allows full unroll, but has in-loop 'h' range test
#define KH_LIMITS 0

// Can we forego khBeg/khEnd (or good_h test) entirely?
#define H_TEST_OPT 0

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
DllFile cjitConvolutionForward3( struct param const* const p )
{
  string const impl = "cjitConvFwd3";
  // this is based on implementation direct_default2p.c, POUTX==1
  int const verbose=0;
  DllFile df; // return value
  //DllFileAux dfx("Convolution","Forward");
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
  pr["macros"]//<<"\n#define VLEN (256)"
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
#define NO_SET_VLEN(...) OSSFMT("NO_SET_VLEN("<<__VA_ARGS__<<");");

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

  int64_t const kByMax = 8; // kByMax=1,2 OK s
  int64_t const vl_x_init = ve_vlen_suggest( outWidth );

#define DEF(VAR) def(#VAR, VAR)
  fn.DEF(batch).DEF(group).DEF(inChannel).DEF(inHeight).DEF(inWidth);
  fn.DEF(outChannel).DEF(outHeight).DEF(outWidth).DEF(kernHeight).DEF(kernWidth);
  fn.DEF(strideHeight).DEF(strideWidth).DEF(padHeight).DEF(padWidth).DEF(dilationHeight);
  fn.DEF(dilationWidth).DEF(inChannelGroup).DEF(outChannelGroup);
  fn.DEF(inHW).DEF(kernHW).DEF(outHW).DEF(vl_x_init);

  auto& fn_ptrs = fn["ptrs"];
  fn_ptrs>>"float const * restrict pIn  = pDataIn;"
    >>"float const * restrict pKernel = pDataKernel;"
    >>"float * restrict const pOut = pDataOut;"
    ;

  auto& fn_vec_init =
    fn["vec_init"]
    >>"// TODO fused-loop opt to handle outWidth much smaller than VLEN (i.e. low vl_x_init)"
    //>>"NO_SET_VLEN(vl_x_init);"
    >>"const __vr vzeros = _vel_vbrds_vsl(0.0f, vl_x_init );"
    >>"const __vr vrseq = _vel_vseq_vl(vl_x_init);"
    >>"int64_t vl = vl_x_init;"
    //>>"NO_SET_VLEN(vl);"
    >>"float * restrict pOutx = pDataOut;"
    ;
  vrj_init(fn_vec_init);

  auto x0_vl_update_check_done = [&outWidth,&vl_x_init](Cblock& loop_x0){
    loop_x0>>"";
    if(0&& vl_x_init>=outWidth){
      loop_x0>>"break;";
    }else{
      loop_x0>>"x0 += vl_x_init;";
      if(1|| outWidth % vl_x_init){
        loop_x0
          >>"{ int64_t vl_rem = outWidth - x0;"
          >>"  if( vl_rem <= 0 ) break;"
          >>"  vl = vl_rem < vl_x_init? vl_rem: vl_x_init;"
          >>"}"
          ;
      }else{
        loop_x0>>"if( x0 >= outWidth ) break; // hits outWidth exactly";
      }
    }
  };

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
      //const int64_t imin = (0          )*strideHeight - padHeight; // range of any value i
      //const int64_t imax = (outHeight-1)*strideHeight - padHeight; // cf. jmin,max for Width
      //int64_t const jmin = (0         )*strideWidth - padWidth; // range of any value in vrj
      //int64_t const jmax = (outWidth-1)*strideWidth - padWidth; // cf. imin,imax for Height
      auto kbeMin = k_beg_end_for_out(0,          outHeight, inHeight,
          kernHeight,strideHeight,padHeight,dilationHeight);
      auto kbeMax = k_beg_end_for_out(outHeight-1,outHeight, inHeight,
          kernHeight,strideHeight,padHeight,dilationHeight);
      // Following may be counterintuitive...
      assert( kbeMin.first  >= kbeMax.first  );
      assert( kbeMin.second >= kbeMax.second );
      int kh_always_full_range =
        (kbeMin.first == 0 && kbeMax.second == kernHeight);

#if KH_LIMITS == 0
      if( H_TEST_OPT==0 || (H_TEST_OPT==1 && !kh_always_full_range) ){
        loop_y
          >>"int64_t kh_end=0;"
          >>"const int64_t kh_tmp = dilationHeight-i-1;"
          >>"const int64_t kh_beg= (i>=0? 0: kh_tmp / dilationHeight);"
          >>"if (i < inHeight){"
          >>"  kh_end = (inHeight + kh_tmp) / dilationHeight;"
          >>"  if (kh_end >= kernHeight) kh_end = kernHeight;"
          >>"}"
          ;
      }
#endif
      loop_y
        >>"vl = vl_x_init;"
        //>>"NO_SET_VLEN(vl);"
        ;
      loop_y>>"// TODO: eliminate once-only loop entirely, for clarity XXX";
      CBLOCK_SCOPE(loop_x0,"for(int64_t x0=0 ; /*x0<outWidth*/; /*x0+=vl_x_init*/)",pr,loop_y);
      //CBLOCK_SCOPE(loop_x0,"for(int64_t x0=0 ; x0<outWidth; x0+=vl_x_init)",pr,loop_y);
      loop_x0
        //>>"vl = outWidth - x0 < vl_x_init ? outWidth - x0: vl_x_init;"
        //>>"NO_SET_VLEN(vl);"
        >>"__vr vrsum = vzeros;"
        ;
      vrj_induce(loop_x0); // vrj ~ vector of input x values
#if KH_LIMITS == 0
      Cblock* ploop_r=nullptr;
      {
        if( H_TEST_OPT==0 ){
          ploop_r = CBLOCK_SCOPE_PTR(loop_r,"#pragma nounroll\n"
              "for (int64_t r = kh_beg; r < kh_end; ++r)",pr,loop_x0);
        }else if(kh_always_full_range){
          printf("HAPPENED000");
          ploop_r = CBLOCK_SCOPE_PTR(loop_r,"#pragma nounroll\n"
              "for (int64_t r = 0; r < kernHeight; ++r)",pr,loop_x0);
        }else{ // other subcases (less common) (actually include prev case)
          printf("HAPPENED001");
          char const * skhBeg = (kbeMin.first==0? "0": "kh_beg");
          char const * skhEnd = (kbeMax.second==kernHeight? "kernHeight": "kh_end");
          ploop_r = CBLOCK_SCOPE_PTR(loop_r,string("#pragma nounroll\n"
                "for (int64_t r = ")+skhBeg+"; r < "+skhEnd+"; ++r)",pr,loop_x0);
        }
      }
      assert( ploop_r != nullptr );
      auto& loop_r = *ploop_r;
      auto& good_h=loop_r["good_h"];
#else
      //int64_t const hmin = imin + (0           )*dilationHeight; // h ~ where in input do we get output y?
      //int64_t const hmax = imax + (kernHeight-1)*dilationHeight;
      //int64_t const wmin = jmin + (0          )*dilationWidth; // w ~ where in input do we get output x0?
      //int64_t const wmax = jmax + (kernWidth-1)*dilationWidth;
      //loop_x0 // /* bad */ >> "#pragma clang unroll(full)" ;
      CBLOCK_SCOPE(loop_r,"#pragma nounroll\n"
          "for (int64_t r = 0; r < kernHeight; ++r)",pr,loop_x0);
      loop_r
        >>"int64_t const h = i + r * dilationHeight; // must be in [0,inHeight)"
        ;
      char const *good_h_test = "!(h < 0 || inHeight <= h)";
      CBLOCK_SCOPE(good_h,string("if ( ")+good_h_test+" )",pr,loop_r);
#endif
      int const sBy = 3; // for kh11kw11 no diff 2 vs 3
      good_h.DEF(sBy) //loop_r
        >>"__vr vrw = vrj;"
        >>"int64_t s=0;"
        ;
      auto& x0_end = loop_x0["induce+write"];
      if(kernWidth/sBy*sBy > 0){
        CBLOCK_SCOPE(loop_sBy,"#pragma nounroll\n"
            "for (s = 0; s < kernWidth/sBy*sBy; s+=sBy)",pr,/*loop_r*/good_h);
        loop_sBy>>"__vm256 vm23 = " VEL_VFMK_mvs_0_TO(vrw,inWidth,vl) ";";
        char const* latest_vrw = "vrw";
        if(sBy >= 2){
          loop_x0>>"__vr vrsum2 = vzeros;// introducing separate vrsums for sBy is a SMALL win";
          loop_sBy
            >>"__vr vrw2 = _vel_vaddsl_vsvl(dilationWidth,  vrw, vl);"
            >>"__vm256 vm45 = " VEL_VFMK_mvs_0_TO(vrw2,inWidth,vl) ";";
          latest_vrw = "vrw2";
        }
        if(sBy >= 3){
          loop_x0>>"__vr vrsum3 = vzeros;// introducing separate vrsums for sBy is a SMALL win";
          loop_sBy
            >>"__vr vrw3 = _vel_vaddsl_vsvl(dilationWidth,  vrw2, vl);"
            >>"__vm256 vm67 = " VEL_VFMK_mvs_0_TO(vrw3,inWidth,vl) ";";
          latest_vrw = "vrw3";
        }
        loop_sBy>>"vrw = _vel_vaddsl_vsvl(dilationWidth, "+string(latest_vrw)+", vl);";

        CBLOCK_SCOPE(loop_c_sBy,"for (int64_t c = 0; c < inChannelGroup; ++c)",pr,loop_sBy);
        loop_c_sBy
#if KH_LIMITS==0
          >>"const float *pIn = pIn_0 + c*inHW + (i+r*dilationHeight)*inWidth"
          >>"                 + x0*strideWidth-padWidth + s*dilationWidth;"
#else
          >>"const float *pIn = pIn_0 + c*inHW + (h)*inWidth"
          >>"                 + x0*strideWidth-padWidth + s*dilationWidth;"
#endif
          >>"const float *pKerValue = pKern_gk + c*kernHW + r*kernWidth +s;"
          >>"__vr vrin = _vel_vldu_vssl(4*strideWidth,pIn,vl);"
          >>"vrsum = _vel_vfmads_vvsvmvl(vrsum, *pKerValue, vrin, vm23, vrsum, vl);"
          ;
        if(sBy>=2){
          loop_c_sBy
            >>"__vr vrin2 = _vel_vldu_vssl(4*strideWidth,pIn+1*dilationWidth,vl);"
            >>"vrsum2 = _vel_vfmads_vvsvmvl(vrsum2, *(pKerValue+1), vrin2, vm45, vrsum2, vl);"
            ;
        }
        if(sBy>=3){
          loop_c_sBy
            >>"__vr vrin3 = _vel_vldu_vssl(4*strideWidth,pIn+2*dilationWidth, vl);"
            >>"vrsum3 = _vel_vfmads_vvsvmvl(vrsum3, *(pKerValue+2), vrin3, vm67, vrsum3, vl);"
            ;
        }
        // (for sBy>=4 could find better summation alg)
        if(sBy>=3) x0_end>>"vrsum2 = _vel_vfadds_vvvl(vrsum2,vrsum3,vl);";
        if(sBy>=2) x0_end>>"vrsum = _vel_vfadds_vvvl(vrsum,vrsum2,vl);";
      }
      if(kernWidth/sBy*sBy != kernWidth){
        CBLOCK_SCOPE(loop_s,"#pragma nounroll\n"
            "for (; s < kernWidth; ++s)",pr,good_h/*loop_r*/);
        loop_s>>"__vm256 vm23 = " VEL_VFMK_mvs_0_TO(vrw,inWidth,vl) ";";
        loop_s>>"vrw = _vel_vaddsl_vsvl(dilationWidth, vrw, vl);";
        CBLOCK_SCOPE(loop_c,"for (int64_t c = 0; c < inChannelGroup; ++c)",pr,loop_s);
        loop_c
#if KH_LIMITS==0
          >>"const float *pIn = pIn_0 + c*inHW + (i+r*dilationHeight)*inWidth"
          >>"                 + x0*strideWidth-padWidth + s*dilationWidth;"
#else
          >>"const float *pIn = pIn_0 + c*inHW + (h)*inWidth"
          >>"                 + x0*strideWidth-padWidth + s*dilationWidth;"
#endif
          >>"const float *pKerValue = pKern_gk + c*kernHW + r*kernWidth +s;"
          >>"__vr vrin = _vel_vldu_vssl(4*strideWidth,pIn, vl);"
          >>"vrsum = _vel_vfmads_vvsvmvl(vrsum, *pKerValue, vrin, vm23,vrsum, vl);"
          ;
      }


      x0_end
        >>"_vel_vstu_vssl(vrsum, 4, pOutx, vl);"
        >>"pOutx += vl; // visible speedup cf. outIndex+=vl"
        >>"//"<<CSTR(printf(" k %ld vl %-3ld outIndex=%ld\n",(long)k,(long)vl,(long)(pOutx-pOut));)
        ;
      x0_vl_update_check_done(x0_end);
      //k = kMax;
      // "simulate" effect of above JIT loop_k on 'k' (illustrative)
      for(; k<kMax; k+=kBy ) /*no-op*/;
      assert( k == kMax );
      scope_kMax["~kMax"]
        >>"CHK(pOutx == pOut + outGroupOffset + (n * outChannel + kMax) * outHW);"
        ;
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
        >>"//CHK(pOutx == pOut + outGroupOffset + (n * outChannel + k) * outHW);"
        >>"float* pOutx1 = pOutx + outHW;"
        >>"const float * restrict pKern_gk = pKernel + kernGroupOffset"
        >>"                                + (k * inChannelGroup + 0/*c*/) * kernHW;"
        ;
      CBLOCK_SCOPE(loop_y,"for(int64_t y=0; y<outHeight; ++y)",pr,loop_k);
      loop_y
        >>"const int64_t i = y * strideHeight - padHeight;"
        >>VEJ_K_BEG_END(kh_beg,kh_tmp,kh_end, kernHeight, i,inHeight, dilationHeight)
        >>"vl = vl_x_init;"
        //>>"NO_SET_VLEN(vl);"
        ;
      // slightly faster looping for x0... (detect vl <= 0 instead of x0 too high)
      CBLOCK_SCOPE(loop_x0,"for(int64_t x0=0; /*x0<outWidth*/; /*x0+=vl_x_init*/)",pr,loop_y);
      vrj_induce(loop_x0); // vrj ~ vector of input x values
      loop_x0>>"__vr vrsum01 = vzeros;";
      CBLOCK_SCOPE(loop_r,"#pragma nounroll\n"
          "for (int64_t r = kh_beg; r < kh_end; ++r)",pr,loop_x0);
      CBLOCK_SCOPE(loop_s,"#pragma nounroll\n"
          "for (int64_t s = 0; s < kernWidth; ++s)",pr,loop_r);
      loop_s[".."]>>"__vr vrw = vrj;";
      loop_s>>"__vm256 vm23 = " VEL_VFMK_mvs_0_TO(vrw,inWidth,vl) ";"
        >>VEL_DECL_VM512(vmP, vm23,vm23, vl);
      loop_s["last"] // BEFORE the '}' of loops_s /**/loop_s/body/induce-vrw
        >>"vrw = _vel_vaddsl_vsvl(dilationWidth, vrw, vl); // <--- vector induced";
      CBLOCK_SCOPE(loop_c,"for (int64_t c = 0; c < inChannelGroup; ++c)",pr,loop_s);
      loop_c
        >>"const float *pIn = pIn_0 + c*inHW + (i+r*dilationHeight)*inWidth"
        >>"                 + x0*strideWidth-padWidth + s*dilationWidth;"
        >>"__vr vrin = _vel_vldu_vssl(4*strideWidth,pIn, vl);"
        >>"const float *pKerValue = pKern_gk + c*kernHW + r*kernWidth +s;"
        >>"const uint64_t kerValue01 = _vel_pack_f32p(pKerValue,"
        >>"                                          pKerValue + inChannelGroup*kernHW);"
        >>"__vr vrinP = _vel_vshf_vvvsl(vrin, vrin, VE_VSHUFFLE_YUZU, vl);"
        >>"vrsum01 = _vel_pvfmad_vvsvMvl(vrsum01, kerValue01, vrinP, vmP, vrsum01, vl);"
        ;
      loop_x0["induce+write"]
        >>"_vel_vstu_vssl(vrsum01, 4, pOutx , vl);"
        >>"_vel_vstl_vssl(vrsum01, 4, pOutx1, vl);"
        >>"pOutx  += vl;"
        >>"pOutx1 += vl;"
        >>"//"<<CSTR(printf(" k %ld vl %-3ld outIndex0=%ld\n",(long)k,(long)vl,(long)(pOutx-pOut));)
        ;
      x0_vl_update_check_done(loop_x0["induce+write"]);
      loop_k["bump pOutx"]
        >>"pOutx += /*kBy-1*/ 1 * outHW; // inner increment is outHW, outer wants kBy*outHW"
        ;
      //k = kMax; // could be too naive?
      // "simulate" effect of above JIT loop_k on 'k' (illustrative)
      for(; k<kMax; k+=kBy ) /*no-op*/;
      assert( k == kMax );
      scope_kMax["~kMax"]
        >>"CHK(pOutx == pOut + outGroupOffset + (n * outChannel + kMax) * outHW);"
        ;
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
        >>VEJ_K_BEG_END(kh_beg,kh_tmp,kh_end, kernHeight, i,inHeight, dilationHeight)
        >>"vl = vl_x_init;"
        //>>"NO_SET_VLEN(vl);"
        ;
      // slightly faster looping for x0... (detect vl <= 0 instead of x0 too high)
      CBLOCK_SCOPE(loop_x0,"for(int64_t x0=0; /*x0<outWidth*/; /*x0+=vl_x_init*/)",pr,loop_y);
      vrj_induce(loop_x0); // vrj ~ vector of input x values
      loop_x0
        >>"__vr vrsum01 = vzeros;"
        >>"__vr vrsum23 = vzeros;"
        ;
      CBLOCK_SCOPE(loop_r,"#pragma nounroll\n"
          "for (int64_t r = kh_beg; r < kh_end; ++r)",pr,loop_x0);
      loop_r
        >>"__vr vrw = vrj;"
        ;
      CBLOCK_SCOPE(loop_s,"#pragma nounroll\n"
          "for (int64_t s = 0; s < kernWidth; ++s)",pr,loop_r);
      loop_s>>"__vm256 vm23 = " VEL_VFMK_mvs_0_TO(vrw,inWidth,vl) ";"
        >>VEL_DECL_VM512(vmP, vm23,vm23, vl);
      CBLOCK_SCOPE(loop_c,"for (int64_t c = 0; c < inChannelGroup; ++c)",pr,loop_s);
      loop_c
        >>"const float *pIn = pIn_0 + c*inHW + (i+r*dilationHeight)*inWidth"
        >>"                 + x0*strideWidth-padWidth + s*dilationWidth;"
        >>"__vr vrin = _vel_vldu_vssl(4*strideWidth,pIn, vl);"
        >>"const float *pKerValue = pKern_gk + c*kernHW + r*kernWidth +s;"
        >>"const uint64_t kerValue01 = _vel_pack_f32p("
        >>"    pKerValue,"
        >>"    pKerValue + inChannelGroup*kernHW);"
        >>"__vr vrinP = _vel_vshf_vvvsl(vrin, vrin, VE_VSHUFFLE_YUZU, vl);"
        >>"const uint64_t kerValue23 = _vel_pack_f32p("
        >>"    pKerValue + 2 * inChannelGroup * kernHW,"
        >>"    pKerValue + 3 * inChannelGroup * kernHW);"
        >>"vrsum01 = _vel_pvfmad_vvsvMvl(vrsum01, kerValue01, vrinP, vmP, vrsum01, vl);"
        >>"vrsum23 = _vel_pvfmad_vvsvMvl(vrsum23, kerValue23, vrinP, vmP, vrsum23, vl);"
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
        ;
      x0_vl_update_check_done(loop_x0["induce+write"]);
      loop_k["bump pOutx"]
        >>"pOutx += (kBy-1) * outHW; // inner increment is outHW, outer wants kBy*outHW"
        ;
      //k = kMax; // could be too naive?
      // "simulate" effect of above JIT loop_k on 'k' (illustrative)
      for(; k<kMax; k+=kBy ) /*no-op*/;
      assert( k == kMax );
      scope_kMax["~kMax"]
        //>>"CHK(pOutx == pOut + outGroupOffset + (n * outChannel + kMax) * outHW);"
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
        >>VEJ_K_BEG_END(kh_beg,kh_tmp,kh_end, kernHeight, i,inHeight, dilationHeight)
        >>"vl = vl_x_init;"
        //>>"NO_SET_VLEN(vl);"
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
      CBLOCK_SCOPE(loop_r,"#pragma nounroll\n"
          "for (int64_t r = kh_beg; r < kh_end; ++r)",pr,loop_x0);
      loop_r
        >>"__vr vrw = vrj;"
        ;
      CBLOCK_SCOPE(loop_s,"#pragma nounroll\n"
          "for (int64_t s = 0; s < kernWidth; ++s)",pr,loop_r);
      loop_s>>"__vm256 vm23 = " VEL_VFMK_mvs_0_TO(vrw,inWidth,vl) ";"
        >>VEL_DECL_VM512(vmP, vm23,vm23, vl);
      CBLOCK_SCOPE(loop_c,"for (int64_t c = 0; c < inChannelGroup; ++c)",pr,loop_s);
      loop_c
        >>"const float *pIn = pIn_0 + c*inHW + (i+r*dilationHeight)*inWidth"
        >>"                 + x0*strideWidth-padWidth + s*dilationWidth;"
        >>"__vr vrin = _vel_vldu_vssl(4*strideWidth,pIn, vl);"
        >>"__vr vrinP = _vel_vshf_vvvsl(vrin, vrin, VE_VSHUFFLE_YUZU, vl);"
        >>"const float *pKerValue = pKern_gk + c*kernHW + r*kernWidth +s;"
        >>"const uint64_t kerValue01 = _vel_pack_f32p("
        >>"    pKerValue,"
        >>"    pKerValue + 1 * inChannelGroup*kernHW);"
        >>"const uint64_t kerValue23 = _vel_pack_f32p("
        >>"    pKerValue + 2 * inChannelGroup * kernHW,"
        >>"    pKerValue + 3 * inChannelGroup * kernHW);"
        >>"vrsum01 = _vel_pvfmad_vvsvMvl(vrsum01, kerValue01, vrinP, vmP, vrsum01, vl);"
        >>"vrsum23 = _vel_pvfmad_vvsvMvl(vrsum23, kerValue23, vrinP, vmP, vrsum23, vl);"
        >>"const uint64_t kerValue45 = _vel_pack_f32p("
        >>"    pKerValue + 4 * inChannelGroup * kernHW,"
        >>"    pKerValue + 5 * inChannelGroup * kernHW);"
        >>"const uint64_t kerValue67 = _vel_pack_f32p("
        >>"    pKerValue + 6 * inChannelGroup * kernHW,"
        >>"    pKerValue + 7 * inChannelGroup * kernHW);"
        >>"vrsum45 = _vel_pvfmad_vvsvMvl(vrsum45, kerValue45, vrinP, vmP, vrsum45, vl);"
        >>"vrsum67 = _vel_pvfmad_vvsvMvl(vrsum67, kerValue67, vrinP, vmP, vrsum67, vl);"
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
        ;
      x0_vl_update_check_done(loop_x0["induce+write"]);
      loop_k["bump pOutx"]
        >>"pOutx += (kBy-1) * outHW; // inner increment is outHW, outer wants kBy*outHW"
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
    if(kByMax==8){ assert( k>=outChannelGroup ); }
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
    if(verbose>=1)
      cout<<string(80,'-')<<pr.str() <<string(80,'-')<<endl;
    if(verbose>=2)
      cout<<string(80,'-')<<pr.tree()<<string(80,'-')<<endl;
  }
  df.code = pr.str();
  return df;
}
// vim: ts=2 sw=2 et cindent cino=^=l0,\:.5s,=-.5s,N-s,g.5s,b1 cinkeys=0{,0},0),\:,0#,!^F,o,O,e,0=break
