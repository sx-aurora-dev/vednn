#ifndef CONV_TEST_PARAM_H
#define CONV_TEST_PARAM_H

#ifdef __cplusplus
extern "C" {
#endif //C++

#include "vednn_helper.h"

#ifdef __cplusplus
}   /* extern "C" */
#endif //C++

#include "convolution_gemm.h"
#include "timer.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>     // drand48
#include <string.h>     // memset
#include <math.h>       // sqrt
#include <stdint.h>

#if defined(FTRACE) && !defined(__ve) // FTRACE **only** for VE
#warning "ignoring attempt to use FTRACE with non-VE compiler (ftrace header may be missing)"
#undef FTRACE
#endif

#ifdef FTRACE
#include <ftrace.h>
#define FTRACE_BEGIN(...) ftrace_region_begin(__VA_ARGS__)
#define FTRACE_END(...) ftrace_region_end(__VA_ARGS__)
#define FTRACE_IF(...) __VA_ARGS__
#else
#define FTRACE_BEGIN(...) do{}while(0)
#define FTRACE_END(...) do{}while(0)
#define FTRACE_IF(...) do{}while(0)
#endif

#ifdef USE_OPENMP
#include <omp.h>
#endif

#define ERROR_EXIT_(F,L,STR) do{ fprintf(stderr,"\nError: %s:%d %s\n",F,L,STR); exit(1); }while(0)
#define ERROR_EXIT(STR) ERROR_EXIT_(__FILE__,__LINE__,STR)

#define CONV_OVERRIDE -13 /* value meaning NO override for this struct param item */

#ifdef __cplusplus
extern "C" {
#endif //C++

#ifndef HAVE_XMALLOC
#define HAVE_XMALLOC
#define XMALLOC(BYTES) \
    xmalloc(BYTES,__FILE__,__LINE__);
inline void * xmalloc(size_t const bytes, char const* file, size_t const line){
    void *ret = malloc(bytes);
    if(!ret) ERROR_EXIT("Memory exhausted");
    return ret;
}
#define DECL_MALLOC(TYPE,var,BYTES) \
    TYPE *var = (TYPE *)malloc(BYTES); \
if(!var) ERROR_EXIT("Memory exhausted")
#endif

struct param;
struct testconvForward;
struct testconvBackwardData;
struct testconvBackwardFilter;

#ifdef FTRACE
inline int with_ftrace() {return 1;}
#else
inline int with_ftrace() {return 0;}
#endif
// which are convolution-specific?
inline int compute_out( int i, int k, int s, int p, int d );
inline int compute_pad( int o, int i, int k, int s, int d );
inline int upMul( int const i, int const a );
inline int mkConsistent( struct param* p );
inline int apply_overrides( struct param* const pParams, int const nParams, struct param* const ovr );
inline int mkConsistentOverrides( struct param* p, struct param* const ovr );
inline void generateRandomData(dataType_t type, size_t size, void *pData);
inline double diffData(const vednnTensorParam_t *pParam, const void *pData, const void *pExpectedResult);
inline int readParamFile(struct param **ppParams, const char *pParamPath );
inline int readParamString(struct param **ppParams, const char *pParamString, char const m_for_mkldnn );
inline void dumpParam(struct param* p, char const* dirn, char const* other);
inline void dumpParamCSV_title();
inline void dumpParamCSV(struct param* p, char const* dirn, char const* other);
inline char const* param_cstr(struct param const* const p, char * const buf, size_t const n); ///< mb#g#_icihiw_ocohow_khshphdh_kwswpwdw
inline char const* param_cstr_short(struct param const* const p, char * const buf, size_t const n);

void testconvForward_init( struct testconvForward *pConv );
void testconvForward_alloc( struct testconvForward *pConv, struct param const* pNw, int const flagBias );
void testconvForward_dumpParms( struct testconvForward const *pConv, int const flagBias );
void testconvForward_randomData( struct testconvForward const* pConv, int const flagBias );
void testconvForward_oclobber( struct testconvForward const* pConv );
void testconvForward_refcalcs( struct testconvForward *pConvArray, int const nEntry ); ///< ->pBufRef
void testconvForward_vednncalcs( struct testconvForward *pConvArray, int const nEntry );
void testconvForward_free( struct testconvForward *pConv, int const flagBias);

void testconvBackwardData_init( struct testconvBackwardData *pConv );
void testconvBackwardData_alloc( struct testconvBackwardData *pConv, struct param const* pNw );
void testconvBackwardData_randomData( struct testconvBackwardData const* pConv );
void testconvBackwardData_dumpParms( struct testconvBackwardData const *pConv );
void testconvBackwardData_oclobber( struct testconvBackwardData const* pConv );
void testconvBackwardData_refcalcs( struct testconvBackwardData *pConvArray, int const nEntry );
void testconvBackwardData_vednncalcs( struct testconvBackwardData *pConvArray, int const nEntry );
void testconvBackwardData_free( struct testconvBackwardData *pConv );

void testconvBackwardFilter_init( struct testconvBackwardFilter *pConv );
void testconvBackwardFilter_alloc( struct testconvBackwardFilter *pConv, struct param const* pNw );
void testconvBackwardFilter_randomData( struct testconvBackwardFilter const* pConv );
void testconvBackwardFilter_dumpParms( struct testconvBackwardFilter const *pConv );
void testconvBackwardFilter_oclobber( struct testconvBackwardFilter const* pConv );
void testconvBackwardFilter_refcalcs( struct testconvBackwardFilter *pConvArray, int const nEntry );
void testconvBackwardFilter_vednncalcs( struct testconvBackwardFilter *pConvArray, int const nEntry );
void testconvBackwardFilter_free( struct testconvBackwardFilter *pConv );

inline unsigned long long count_ops(struct param const* p); // as in bench-dnn

struct param {
    int        batchNum;
    int        group;
    int        inChannel;
    int        inHeight;
    int        inWidth;
    int        outChannel;
    int        outHeight;
    int        outWidth;
    int        kernHeight;
    int        kernWidth;
    int        strideHeight;
    int        strideWidth;
    int        padHeight;
    int        padWidth;
    int        dilationHeight; /* "no dilation" is 1 (mkl-dnn value + 1) */
    int        dilationWidth;
    const char pName[256];
};
//struct vednnTensorParam_t;
struct vednnBiasParm_t;
struct vednFilterParam_t;
//struct vednnConvolutionParam_t;
struct testconvForward {
    vednnTensorParam_t      *pParamIn;
    vednnTensorParam_t      *pParamOut;
    vednnBiasParam_t        *pParamBias;
    vednnFilterParam_t      *pParamKernel;
    vednnConvolutionParam_t *pParamConv;
    void *pDataIn;
    void *pDataOut; // normal "output" buffer
    void *pDataBias;
    void *pDataKernel;

    void *pBufRef;  // "reference output" buffer (pDataOut)
    float *pBufOne;
    float *pBufCol;

    unsigned long long cycle;
    unsigned long long mincycle;
    unsigned long long maxcycle;
    unsigned long long ops;
    unsigned reps;
    char region[128];
    char ref_region[128];
};

struct testconvBackwardData {
    vednnTensorParam_t *pParamGradOut;
    vednnTensorParam_t *pParamGradIn;
    vednnFilterParam_t *pParamKernel;
    vednnConvolutionParam_t *pParamConv;
    void *pDataGradOut;
    void *pDataGradIn;
    void *pDataKernel;

    void  *pBufRef;
    float *pBufCol;

    unsigned long long cycle;
    unsigned long long mincycle;
    unsigned long long maxcycle;
    unsigned long long ops;
    unsigned reps;
    char region[128];
    char ref_region[128];
};
struct testconvBackwardFilter {
    vednnTensorParam_t *pParamIn;
    vednnTensorParam_t *pParamGradOut;
    vednnFilterParam_t *pParamGradKernel;
    vednnConvolutionParam_t *pParamConv;
    void *pDataIn;
    void *pDataGradOut;
    void *pDataGradKernel;

    void  *pBufRef;
    float *pBufCol;

    unsigned long long cycle;
    unsigned long long mincycle;
    unsigned long long maxcycle;
    unsigned long long ops;
    unsigned reps;
    char region[128];
    char ref_region[128];
};


/** exact output height/width for convolution. d==0 for "no dilation" (cf. vednn 1 for none) */
// exact output height/width for convolution.
// d==0 for "no dilation". NO NO. libvednn uses d==1.
// this function does NOT agree with mkl-dnn calc, as a result.
// *--p--*-------i-------*--p--*
// | (k-1)*s+1 | ...    |
//
// mb1ic16ih5oc16oh3kh7ph3, 1,1, 16,5,5, 16,3,3, 7,7, 1,1, 1,1
// pppiiiiippp   i+2p = 11, kern reach is 7 = (7-1)*1+1 = (k-1)*s+1
// (11 - 7) / d +1 
// ppppp iiiiiiiii ppppppp
// 1 2 3 4 5 6 7 8 9 10 11
// 1^^^^^^^^^^^^ i+2p
//   2^^^^^^^^^^^^ ...
//     3 4 5^^^^^^^^^^^^^^^  5 outputs
inline int
compute_out( int i, int k, int s, int p, int d ){
    // convolution min output size:
    //    mkl-dnn dilation convention ~ d==0 for "no dilation"
    //return (i - ((k - 1) * (d + 1) + 1) + 2 * p) / s + 1;
    //    libvednn convention
    //
    // Better: if out size should be zero detect -ve
    // (round-to-negative-infinity signed division would be better here)
    //
    int numer = (i - ((k - 1) * (d + 0) + 1) + 2 * p);
    return numer < 0
        ? 0
        : numer / s + 1;
    // mkl-dnn deconv: return (i - 1) * s + (k - 1) * (d + 1) + 2 * p + 1;
}
inline int
compute_pad( int o, int i, int k, int s, int d ){
    int numer = ((o-1) * s - i + ((k - 1) * (d + 0) + 1)) / 2;
    return numer < 0
        ? 0
        : numer / 2;
}
/** round i>0 upward to a multiple of a>0 */
inline int
upMul( int const i, int const a ) {
    return (i+a-1)/a*a;
}
/** return # of modifications needed to make the test look reasonable. */
inline int
mkConsistent( struct param* p ){
    // fix bad things so at least some test gets run.
    // (If I change padding or dilation or ... I am too lazy
    //  to calculate the new output size, for example).
    //
    // I try not to weed out valid test settings...
    //
    // apply changes, warn...
    int changed = 0;
#define FIXIT( ITEM, EXPR ) do \
    { \
        int const val = (EXPR); \
        if( p->ITEM != val ){ \
            printf(" " #ITEM " %d->%d",(int)p->ITEM, val); \
            p->ITEM = val; \
            ++changed; \
        } \
    }while(0)

    int g = p->group;
    // g cannot be larger then p->inChannel
    if (p->group > p->inChannel) g = p->inChannel;
    // g must divide evenly into p->inChannel
    while( p->inChannel % g != 0 ) --g;
    FIXIT( group, g );

    int ic = upMul( p->inChannel, g );
    // one pass over input channels uses ic/g kernels to make g output channels.
    int oc = upMul( p->outChannel, g );
    FIXIT( inChannel, ic );
    FIXIT( outChannel, oc ); // any number is fine

    // "no dilation" is 1 (mkl-dnn value + 1)
    int dh = p->dilationHeight;
    int dw = p->dilationWidth;
    if(dh<1) dh=1;
    if(dw<1) dw=1;
    FIXIT( dilationHeight, dh );
    FIXIT( dilationWidth,  dw );

    if(p->padHeight < 0) FIXIT( padHeight, 0 );
    if(p->padWidth  < 0) FIXIT( padWidth,  0 );
    if(p->kernHeight < 1) FIXIT( kernHeight, 3 );
    if(p->kernWidth  < 1) FIXIT( kernWidth,  3 );
    if(p->strideHeight < 1) FIXIT( strideHeight, 1 );
    if(p->strideWidth  < 1) FIXIT( strideWidth,  1 );

    //int min_inHeight = p->kernHeight + 2*p->padHeight;
    int min_inHeight = 1;
    if(p->inHeight < min_inHeight) FIXIT(inHeight, min_inHeight);
    //int min_inWidth = p->kernWidth + 2*p->padWidth;
    int min_inWidth = 1;
    if(p->inWidth < min_inWidth) FIXIT(inWidth, min_inWidth);

    int oh = compute_out( p->inHeight, p->kernHeight, p->strideHeight, p->padHeight, dh );
    int ow = compute_out( p->inWidth , p->kernWidth , p->strideWidth , p->padWidth , dw );
#if 0 // In principle, coulld malloc any larger output region, but...
    // the 'DIFF' calc in test programs might only handle exact-sized output
    // today (could be fixed, but not so important for correctness testing)
    oh = (p->outHeight >= oh? p->outHeight: oh);
    ow = (p->outWidth  >= ow? p->outWidth : ow);
#endif
    FIXIT( outHeight, oh );
    FIXIT( outWidth,  ow );
#undef FIXIT

    if (changed){
        printf(" changed in %s\n", p->pName);
        fflush(stdout);
    }
    return changed;
}
inline int
apply_overrides( struct param* const pParams, int const nParams, struct param* const ovr ){
    unsigned n=0U;
    printf(" applying overrides... ");
#define OVRCHK(VAR) do \
    { \
        if(ovr->VAR != CONV_OVERRIDE){ \
            ++n; \
            printf(" " #VAR "-->%d",ovr->VAR); \
            for(int i=0; i<nParams; ++i){ \
                pParams[i].VAR = ovr->VAR; \
            }}}while(0)
    OVRCHK(batchNum       );
    OVRCHK(group          );
    OVRCHK(inChannel      );
    OVRCHK(inHeight       );
    OVRCHK(inWidth        );
    OVRCHK(outChannel     );
    OVRCHK(outHeight      );
    OVRCHK(outWidth       );
    OVRCHK(kernHeight     );
    OVRCHK(kernWidth      );
    OVRCHK(strideHeight   );
    OVRCHK(strideWidth    );
    OVRCHK(padHeight      );
    OVRCHK(padWidth       );
    OVRCHK(dilationHeight );
    OVRCHK(dilationWidth  );
#undef OVRCHK
    if(n) printf("\n");
    return n;
}
/** constrained to not change change non-CONV_OVERRIDE values in \c ovr,
 * return # of modifications needed to make the test \c p look reasonable. */
inline int
mkConsistentOverrides( struct param* p, struct param* const ovr ){
    // fix bad things so at least some test gets run.
    // (If I change padding or dilation or ... I am too lazy
    //  to calculate the new output size, for example).
    //
    // I try not to weed out valid test settings...
    // But never change non-CONV_OVERRIDE settings in 'ovr'
    //
    // apply changes, warn...
    int changed = apply_overrides(p, 1/*nEntry*/, ovr);
    printf(" consistency ... ");
#define FIXIT( ITEM, EXPR ) do \
    { \
        int const val = (EXPR); \
        if( p->ITEM != val ){ \
            printf(" " #ITEM " %d->%d",(int)p->ITEM, val); \
            p->ITEM = val; \
            ++changed; \
        } \
    }while(0)
#define MODIFIABLE(PARAM) (ovr->PARAM != CONV_OVERRIDE)

    int g = p->group;
    if( g < 1 ) FIXIT( group, g = 1 );
    int ic = p->inChannel;
    if( ic < 1 ) FIXIT( inChannel, ic = 1 );
    int oc = p->outChannel;
    if( oc < 1 ) FIXIT( outChannel, oc = 1);
    if( p->strideHeight < 1 ) FIXIT(strideHeight, 1);
    if( p->strideWidth  < 1 ) FIXIT(strideWidth,  1);
    if( p->dilationHeight < 1 ) FIXIT(dilationHeight, 1);
    if( p->dilationWidth  < 1 ) FIXIT(dilationWidth,  1);

    // ic oc must be divisible by g
    if(MODIFIABLE(group) || !MODIFIABLE(inChannel)){
        // g cannot be larger then p->inChannel
        if(p->group > p->inChannel) g = p->inChannel;
        // g must divide evenly into p->inChannel
        while( p->inChannel % g != 0 ) --g;
    }else if(MODIFIABLE(inChannel)){
        if(p->group > p->inChannel) ic = p->group;
    }
    ic = upMul( p->inChannel, g );
    FIXIT( inChannel, ic );
    // one pass over input channels uses ic/g kernels to make g output channels.
    if(MODIFIABLE(outChannel)){
        oc = upMul( p->outChannel, g );
    }else if(MODIFIABLE(group) && oc%g ){
        while( oc % g != 0 ) --g;
        ic = upMul( p->inChannel, g );
    }
    FIXIT( group, g );
    FIXIT( inChannel, ic );
    FIXIT( outChannel, oc );

    // "no dilation" is 1 (mkl-dnn value + 1)
    int dh = p->dilationHeight;
    int dw = p->dilationWidth;
    int kh = p->kernHeight;
    int kw = p->kernWidth;
    if(dh<1 || kh==1/*dh irrelevant*/) dh=1;
    if(dw<1 || kw==1/*dw irrelvant */) dw=1;
    FIXIT( dilationHeight, dh );
    FIXIT( dilationWidth,  dw );

    if(kh < 1) FIXIT( kernHeight, kh=3 );
    if(kw < 1) FIXIT( kernWidth,  kw=3 );

    // TODO extra padding not handled well by vednn tests (SHOULD be allowed):
    // TODO check even kernels
    if(p->padHeight < 0)
        FIXIT( padHeight, 0 );
    else if( p->padHeight ==0 )
        ; // always acceptable
    else if( p->kernHeight != 2*p->padHeight + 1 )
        FIXIT( padHeight, (p->kernHeight - 1)/2 );

    if(p->padWidth  < 0)
        FIXIT( padWidth,  0 );
    else if( p->padWidth ==0 )
        ; // always acceptable
    else if( p->kernWidth  != 2*p->padWidth  + 1 )
        FIXIT( padWidth , (p->kernWidth  - 1)/2 );

    printf(" (kh=%d~%d)",kh,p->kernHeight);

    //if(p->inHeight <= 0) FIXIT( inHeight, 1 );
    //if(p->inWidth  <= 0) FIXIT( inWidth,  1 );
    int min_inHeight = p->kernHeight + 2*p->padHeight;
    if(p->inHeight < min_inHeight) FIXIT(inHeight, min_inHeight);
    int min_inWidth = p->kernWidth + 2*p->padWidth;
    if(p->inWidth < min_inWidth) FIXIT(inWidth, min_inWidth);

    // a common case of pad,kern,stride,dil begin changed...
    int oh = p->outHeight, ow = p->outWidth;
    if(MODIFIABLE( outHeight )){
        oh = compute_out( p->inHeight, kh,  p->strideHeight, p->padHeight, dh );
        printf(" oh:compute_out(%d,%d,%d,%d,%d)->%d", p->inHeight , kh,
                p->strideHeight , p->padHeight , dh, oh);
        FIXIT( outHeight, oh );
    }
    if(MODIFIABLE(outWidth)){
        ow = compute_out( p->inWidth , kw, p->strideWidth , p->padWidth , dw );
        printf(" ow:compute_out(%d,%d,%d,%d,%d)->%d", p->inWidth , kw,
                p->strideWidth , p->padWidth , dw, ow);
        FIXIT( outWidth,  ow );
    }
    // rationale: often may have tests spanning input image size
    // NOTE check is this is OK for even kernel sizes too, because 'pad' might be
    // zero or one depending on input dim ... messy.
    // NOTE readParamString does a more careful job (bugfix for oh, ow Aug 2019)
    if( oh <= 0 || oh != compute_out( p->inHeight, p->kernHeight, p->strideHeight, p->padHeight, dh)){
        oh = compute_out( p->inHeight, p->kernHeight, p->strideHeight, p->padHeight, dh );
        printf(" oh:fix2(%d,%d,%d,%d,%d)->%d", p->inHeight , p->kernHeight,
                p->strideHeight , p->padHeight , dh, oh);
        if(oh<=0){
            FIXIT( kernHeight, kh=1 );
            FIXIT( kernWidth,  kw=1 );
            oh = compute_out( p->inHeight, p->kernHeight, p->strideHeight, p->padHeight, dh );
        }
    }
    if( ow <= 0 || ow != compute_out( p->inWidth, p->kernWidth, p->strideWidth, p->padWidth, dw)){
        ow = compute_out( p->inWidth, p->kernWidth, p->strideWidth, p->padWidth, dw );
        if(ow<=0){
            FIXIT( kernWidth, kh=1 );
            FIXIT( kernWidth,  kw=1 );
            ow = compute_out( p->inWidth, p->kernWidth, p->strideWidth, p->padWidth, dw );
        }
        printf(" ow:fix2(%d,%d,%d,%d,%d)->%d", p->inWidth , p->kernWidth,
                p->strideWidth , p->padWidth , dw, ow);
    }
    FIXIT( outHeight, oh);
    FIXIT( outWidth,  ow);

#if 0 // In principle, coulld malloc any larger output region, but...
    // the 'DIFF' calc in test programs might only handle exact-sized output
    // today (could be fixed, but not so important for correctness testing)
    oh = (p->outHeight >= oh? p->outHeight: oh);
    ow = (p->outWidth  >= ow? p->outWidth : ow);
#endif
#undef FIXIT
#undef MODIFIABLE

    if (changed){
        printf(" changed in %s\n", p->pName);
        fflush(stdout);
    }
    return changed;
}
#if 0 // very slow
    inline void
generateRandomData(dataType_t type, size_t size, void *pData)
{
    size_t i;
    switch(type) {
    case DTYPE_FLOAT:
    {
        float *p = (float *)pData;
        for (i=0; i<size; i++) {
            p[i] = drand48();
        }
    }
    break;
    default:
    ERROR_EXIT("Unknown dataType_t");
    break;
    }
}
#elif 0 // still not vectorized, but pretty fast
    inline void
generateRandomData(dataType_t type, size_t size, void *pData)
{
    size_t i;
    static uint64_t r=1234567UL;
    union uf64 {
        uint64_t u;
        double d;
    };
    union uf32 {
        uint32_t u;
        float f;
    };
    switch(type) {
    case DTYPE_FLOAT:
    {
        float *p = (float *)pData;
        uf64 uf;
        uint64_t rr=r;
        for (i=0; i<size; i++) {
            p[i] = drand48();
            // no func call, fast scalar ops
            rr = rr * 2862933555777941757U /*LCGMUL4*/ + 13U;
            // convert to valid double:
            // float rand_float_co()
            // {
            //     return as_float(0x3F800000U | (rand32() >> 9)) - 1.0f;
            // }
            //      
            // double rand_double_co()
            // {
            //     return as_double(0x3FF0000000000000ULL | (rand64() >> 12)) - 1.0;
            // }
            uf.u = 0x3FF0000000000000ULL | (rr >> 12);
            p[i] = uf.d - 1.0;
        }
    }
    break;
    default:
    ERROR_EXIT("Unknown dataType_t");
    break;
    }
    r = r * 7664345821815920749U /*scramble64::r1*/ + 21U;
}
#else // this one vectorizes [fastest], but could be faster
/* - could use u32 rand
 * - or u64 with 512-blocking (packed ops) */
    inline void
generateRandomData(dataType_t type, size_t size, void *pData)
{
    size_t i;
    uint64_t const r=rand();
    union uf64 {
        uint64_t u;
        double d;
    };
    switch(type) {
    case DTYPE_FLOAT:
    {
        float *p = (float *)pData;
        uint64_t rr[256];
        for(i=0; i<256; ++i) rr[i] = r + i;

        union uf64 uf[256];
        for (i=0; i<size/256U*256U; i+=256U) {
            for(int j=0;j<256;++j){
                uf[j].u = 0x3FF0000000000000ULL | (rr[j] >> 12);
                p[i] = uf[j].d - 1.0;
                rr[j]=rr[j]*2862933555777941757ULL+13ULL;
            }
        }
        for (size_t i0=i ; i<size; ++i) {
            uf[i-i0].u = 0x3FF0000000000000ULL | (rr[i-i0] >> 12);
            p[i] = uf[i-i0].d - 1.0;
            rr[i-i0]=rr[i-i0]*2862933555777941757ULL+13ULL;
        }
    }
    break;
    default:
    ERROR_EXIT("Unknown dataType_t");
    break;
    }
    //r = r * 7664345821815920749U /*scramble64::r1*/ + 21U;
}
#endif

    inline double
diffData(const vednnTensorParam_t *pParam, const void *pData, const void *pExpectedResult)
{
    size_t i;
    size_t size = getTensorSize(pParam);
    double sum = 0.0;

    switch(getTensorDataType(pParam)) {
    case DTYPE_FLOAT:
    {
        const float *pData1 = (float *)pExpectedResult;
        const float *pData2 = (float *)pData;
        for (i=0; i<size; i++) {
            double diff = pData1[i] - pData2[i] ;
            sum += (diff/(fabs(pData1[i])+1e-7)) * (diff/(fabs(pData2[i])+1e-7)) ;
        }
    }
    break;
    default:
    ERROR_EXIT("Unknown dataType_t");
    break;
    }

    return sqrt(sum);
}

/** Reads multiple tests from file.
 * - historical: do not read dilation parameters.
 * - NEW: fall back to interpretation as readParamString of mkl-dnn format
 *        if there is any trouble with libvednn CSV format.
 * \return 0 on error, else number of test params read.
 *
 * \todo readParamFile if 1st line is not a single number should read "all" lines,
 *       and keep the ones that look OK; and ignore '#'-lines and blank lines;
 *       and ignore bad input lines (with warning)
 */
    inline int
readParamFile(struct param **ppParams, const char *pParamPath )
{
    int const v=2; //verbose
    struct param *pParams  = NULL ;
    int nParams            = 0 ;

    FILE *fp = fopen(pParamPath, "r") ;
    if( fp == NULL ) {
        fprintf(stderr, "Cannot open parameter file : %s .\n", pParamPath );
        return 0;
    }

    int nscanned = fscanf(fp, "%d\n", &nParams) ;
    if( nscanned != 1 || nParams <= 0 ) {
        fprintf(stderr, "Parameter file read error.\n");
        fclose(fp) ;
        return 0;
    }
    pParams = (struct param *) malloc(sizeof(struct param) * nParams) ;

    for(int i=0; i<nParams; i++) {
        memset(&pParams[i], 0, sizeof(struct param));
        fpos_t pos;
        int pos_err = fgetpos(fp, &pos);
        if( pos_err!=0 ) printf("oh? fgetpos returned %d ?\n",pos_err);
        // First attempt to read the line in libvednn format
        int nscanned =
            fscanf(fp, "%[^,],%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d\n",
                (char*)&pParams[i].pName[0],
                &(pParams[i].batchNum),
                &(pParams[i].group),
                &(pParams[i].inChannel),
                &(pParams[i].inHeight),
                &(pParams[i].inWidth),
                &(pParams[i].outChannel),
                &(pParams[i].outHeight),
                &(pParams[i].outWidth),
                &(pParams[i].kernHeight),
                &(pParams[i].kernWidth),
                &(pParams[i].strideHeight),
                &(pParams[i].strideWidth),
                &(pParams[i].padHeight),
                &(pParams[i].padWidth) );
        int scanned = 1;
        if( nscanned < 5 ){
            if(v) printf(" We must at least have 5 parms per line\n"
                    " bs,grp, ic,ih,iw [,oc,oh,ow, kh,kw, sh,sw, ph,pw] [TDOD: ,dh,dw]\n");
            if( pos_err==0 ){
                // Else try to parse an mkl-dnn string spec
                if(v) printf("Will try as parameter string in mkl-dnn input format\n");
                pos_err = fsetpos(fp,&pos); // go back to beginning of the line
                if( pos_err ){
                    printf(" ? Could not go back to beginning of line (continuing anyway)\n");
                    continue;
                }
                // retry this line as if it were an mkl-dnn test string
                char line[300];
                int linesz;
                fscanf(fp, "%299[^\n]%n\n", &line[0], &linesz);
                // change the newline to NUL and parse that string
                line[linesz]='\0';
                if(line[0]!='#'){
                    if(v) printf("re-read line: %s\n",line);
                    struct param* tmp = NULL;
                    if(readParamString( &tmp, &line[0], 'm' )){
                        // maybe got a valid mkl-dnn test string
                        memcpy( &pParams[i], tmp, sizeof(struct param) );
                        if(v) printf("re-parsed as mkl-dnn test spec\n");
                        scanned = 0; // do NOT set dilation to default value
                    }
                    if(tmp) free(tmp);
                }else{
                    --i; // ignore this line
                }
            }
        }
        if(scanned){
            // historical: parm files did not support dilation
            pParams[i].dilationHeight = 1; /* vednn "no dilation" = mkldnn value + 1 */
            pParams[i].dilationWidth = 1;
        }
    }
    fclose(fp) ;

    *ppParams = pParams ;
    return nParams ;
}
/** interpret an mkl-dnn-like string for test parameters.
 * - ignore embedded '_'
 * \return 1 and a freshly allocated *pParams [or 0 and *pParams=NULL].
 */
inline int
readParamString(struct param **ppParams, const char *pParamString, char const m_for_mkldnn ){
    int const v=2; // verbose
    int const nParams = 1;
    struct param* pParams;      // pParams  : decoded values
    struct param* have;         // have     : 0/1 value was given (we handle missing values)
    pParams = (struct param *) malloc(sizeof(struct param) * nParams) ;
    memset(pParams, 0, nParams*sizeof(struct param));
    // opt: could allocate just a single 'have' buffer
    have = (struct param *) malloc(sizeof(struct param) * nParams) ;
    memset(have, 0, nParams*sizeof(struct param));
    int err = 0;
    char * end = (char*)pParamString;
    //printf(" &end = %p\n", &end);
    for(int ii=0; ii<nParams; ++ii)
    {
        int const i=ii;
        if(v>0){printf("pParamstring %s\n",pParamString); fflush(stdout);}
        for( char * s=(char*)pParamString;
                s!=NULL && *s!='\0'; )
        {
            char *s0 = s; // remember where we started
            if(v>1){printf(" enter: s@%p\n",(void*)s); fflush(stdout);}
            if(v>1){printf(" enter: s %s\n",s); fflush(stdout);}
            // Special: "#"  means ignore rest of line (comment)
            if(*s == '#'){
                break;
            }
            if(*s == '_'){
               ++s;
               continue;
            }
            if(s[1] == '\0' || s[2] == '\0'){++err; break;}
#define CONV_TEST_PARAM_CHK(STR,LEN,batchNum) \
            do { \
                if(strncmp(s,STR,LEN)==0){ \
                    /*printf("  s @  %p\n",(void*)s);*/ \
                    if(v>1){printf(" matched %s in %s",STR,s);} \
                    int val = (int)strtol(&s[LEN],&end,10); \
                    if(end==&s[LEN]){++err;break;} \
                    if(v>1){printf(" = %d\n",val); fflush(stdout);} \
                    if(v==1){printf(" %s=%d",STR,val);} \
                    pParams[i].batchNum = val; \
                    have[i].batchNum = 1; \
                    s=end; \
                    while(*s=='_') ++s; \
                    /*printf(" remain %s\n",s); fflush(stdout);*/ \
                    continue; \
                }else{ \
                    /*printf(" no %s in %s\n",STR,s); fflush(stdout);*/ \
                } \
            } while(0)

            CONV_TEST_PARAM_CHK("mb",2,batchNum);
            CONV_TEST_PARAM_CHK("g",1,group);
            CONV_TEST_PARAM_CHK("ic",2,inChannel);
            CONV_TEST_PARAM_CHK("ih",2,inHeight);
            CONV_TEST_PARAM_CHK("iw",2,inWidth);
            CONV_TEST_PARAM_CHK("oc",2,outChannel);
            CONV_TEST_PARAM_CHK("oh",2,outHeight);
            CONV_TEST_PARAM_CHK("ow",2,outWidth);
            CONV_TEST_PARAM_CHK("kh",2,kernHeight);
            CONV_TEST_PARAM_CHK("kw",2,kernWidth);
            CONV_TEST_PARAM_CHK("sh",2,strideHeight);
            CONV_TEST_PARAM_CHK("sw",2,strideWidth);
            CONV_TEST_PARAM_CHK("ph",2,padHeight);
            CONV_TEST_PARAM_CHK("pw",2,padWidth);
            CONV_TEST_PARAM_CHK("dh",2,dilationHeight);
            CONV_TEST_PARAM_CHK("dw",2,dilationWidth);
            // Special: "n"  name is a terminal option, default = \"wip\", # terminates
            if( s[0] == 'n' ){
                if(v>0){printf(" matched n in %s, name terminates options",s);}
                char *__restrict__ pName = (char *)&pParams[i].pName[0];
                //strncpy(pName, s+1, 256); // but also use '#' as terminator
                for(unsigned j=0; j<256; ++j){
                    if(j==255 || s[j+1]=='\0' || s[j+1]=='#'){
                        pName[j]='\0';
                        break;
                    }
                    pName[j] = s[j+1];
                }
                *(char*)&have[i].pName[0]=1;
                if(v>0){printf(" name is <%s>\n",pParams[i].pName);}
                break;
            }
            // if still where we started, we did not recognize any option (error)
            if(s==s0){
                printf(" unrecognized characters at: %s\n",s);
                ++err; break;
            }
#undef CONV_TEST_PARAM_CHK
        }
        if(err) break;
        // 
        // In 'm'kl-dnn convention, dilation parameters begin at 0
        // so we add one to comply with libvednn convention
        //
        if(m_for_mkldnn == 'm'){
            if(have[i].dilationHeight) pParams[i].dilationHeight += 1;
            if(have[i].dilationWidth) pParams[i].dilationWidth += 1;
        }

        /* canonical form:
         * dYgXmbXicXihXiwXocXohXowXkhXkwXshXswXphXpwXdhXdwXnS
         *
         * where: Y = {fb, fd, bd, bw, bb}, X is number, S - string
         * note: symbol `_` is ignored
         *
         * implicit rules:
         *  - default values:
         *      mb = 2, g = 1, d = fd, sh = sw = 1, dh = dw = 0, S="wip"
         *  - if H is undefined => H = W
         *  - if W is undefined => W = H
         *  - if padding is undefined => compute trivial padding (else pad=0)
         *  - if `output` is undefined => compute output
         */
        if( !have[i].pName[0] ){
            strncpy( (char*)&pParams[i].pName[0], "\"wip\"", 6);
            *(char*)&have[i].pName[0] = 1;
        }
        if( !have[i].batchNum ){
            pParams[i].batchNum = 2; // this is mkl-dnn convention (perhaps surprising)
            have[i].batchNum = 1;
        }
        if( !have[i].group ) {
            pParams[i].group = 1;
            have[i].group = 1;
        }
        /* if no Height/Width info, dilation, stride, kern get default values */
        if( !have[i].dilationHeight && !have[i].dilationWidth ){
            // NB: libvednn "no dilation" value is 1, which is mkl-dnn value plus 1
            pParams[i].dilationHeight = pParams[i].dilationWidth = 1;
            have[i].dilationHeight    = have[i].dilationWidth = 1;
        }
        if( !have[i].strideHeight && !have[i].strideWidth ){
            pParams[i].strideHeight = pParams[i].strideWidth = 1;
            have[i].strideHeight    = have[i].strideWidth = 1;
        }
        if( !have[i].kernHeight && !have[i].kernWidth ){
            pParams[i].kernHeight = pParams[i].kernWidth = 3;
            have[i].kernHeight    = have[i].kernWidth = 1;
        }
        /* many missing fields default to "square" Height = Width */
#define MISSING_SAME(FIELD1,FIELD2) \
        do { \
            if( have[i].FIELD1 && !have[i].FIELD2 ){ \
                have[i].FIELD2 = 1; \
                pParams[i].FIELD2 = pParams[i].FIELD1; \
            }else if( have[i].FIELD2 && !have[i].FIELD1 ){ \
                have[i].FIELD1 = 1; \
                pParams[i].FIELD1 = pParams[i].FIELD2; \
            } \
        }while(0)
        MISSING_SAME(inHeight,inWidth);
        MISSING_SAME(outHeight,outWidth);
        MISSING_SAME(kernHeight,kernWidth);
        MISSING_SAME(strideHeight,strideWidth);
        MISSING_SAME(padHeight,padWidth);
        MISSING_SAME(dilationHeight,dilationWidth);
        assert( have[i].kernHeight && have[i].kernWidth );
        assert( have[i].dilationHeight && have[i].dilationWidth );
        assert( have[i].strideHeight && have[i].strideWidth );
        // infer padding from in/out + kernel size, if possible, else pad=0
        // padHeight?  If pad < 0, assume outH/W was incorrect and zero the pad
        if( !have[i].padHeight ){
            if( have[i].outHeight && have[i].inHeight && have[i].kernHeight ){
                pParams[i].padHeight = compute_pad(pParams[i].outHeight,
                        pParams[i].inHeight,       pParams[i].kernHeight,
                        pParams[i].strideHeight,   pParams[i].dilationHeight);
                if(pParams[i].padHeight < 0){
                    pParams[i].padHeight = 0;
                    have[i].outHeight = 0; // assume mistake here
                }
            }else{
                pParams[i].padHeight = 0;
            }
            have[i].padHeight = 1;
        }
        // padWidth?
        if( !have[i].padWidth ){
            if( have[i].outWidth && have[i].inWidth && have[i].kernWidth ){
                pParams[i].padWidth = compute_pad(pParams[i].outWidth,
                        pParams[i].inWidth,       pParams[i].kernWidth,
                        pParams[i].strideWidth,   pParams[i].dilationWidth);
                if(pParams[i].padWidth < 0){
                    pParams[i].padWidth = 0;
                    have[i].outWidth = 0; // assume mistake here
                }
            }else{
                pParams[i].padWidth = 0;
            }
            have[i].padWidth = 1;
        }
        assert( have[i].padHeight && have[i].padWidth );

        // Can we calculate a missing out size?
        if( !have[i].outHeight && have[i].inHeight && have[i].kernHeight && have[i].strideHeight && have[i].padHeight && have[i].dilationHeight ){
            pParams[i].outHeight = compute_out( pParams->inHeight,
                    pParams->kernHeight,        pParams->strideHeight,
                    pParams->padHeight,         pParams->dilationHeight);
            have[i].outHeight = 1;
        }
        if( !have[i].outWidth && have[i].inWidth && have[i].kernWidth && have[i].strideWidth && have[i].padWidth && have[i].dilationWidth ){
            pParams[i].outWidth = compute_out( pParams->inWidth,
                    pParams->kernWidth,        pParams->strideWidth,
                    pParams->padWidth,         pParams->dilationWidth);
            have[i].outWidth = 1;
        }
        // We do not try to infer input size.
        // Any undefined fields at this point mean there is some error.
        if( have[i].batchNum == 0
                || have[i].group == 0
                || have[i].inChannel == 0
                || have[i].inHeight == 0
                || have[i].inWidth == 0
                || have[i].outChannel == 0
                || have[i].outHeight == 0
                || have[i].outWidth == 0
                || have[i].kernHeight == 0
                || have[i].kernWidth == 0
                || have[i].strideHeight == 0
                || have[i].strideWidth == 0
                || have[i].padHeight == 0
                || have[i].padWidth == 0
                || have[i].dilationHeight == 0
                || have[i].dilationWidth == 0 ){
            ++err;
        }
        // NB: you still MUST call mkConsistent because libvednn tests have
        //     undefined behavior if you specify inconsistent in/out/pad values!
    }
    free(have);
    //printf(" out of loop\n"); fflush(stdout);
    if(err){
        //printf(" error\n"); fflush(stdout);
        free(pParams);
        *ppParams = NULL;
        return 0;
    }
    *ppParams = pParams;
    return nParams;
}
inline void param_cstr_error_exit(){
    perror("param_cstr convolution-->cstr error");
    printf("param_cstr error -- exiting now");
    exit(1);
}
inline char const*
param_cstr(struct param const* const p, char * const buf, size_t const n){
    if(snprintf(buf, n,
            "mb%dg%d_ic%dih%diw%d_oc%doh%dow%d_kh%dph%dsh%ddh%d_kw%dpw%dsw%ddw%d",
            p->batchNum, p->group,
            p->inChannel, p->inHeight, p->inWidth,
            p->outChannel, p->outHeight, p->outWidth,
            p->kernHeight, p->padHeight, p->strideHeight, p->dilationHeight-1, // dil-1 for mkl-dnn convention
            p->kernWidth, p->padWidth, p->strideWidth, p->dilationWidth-1 )
            < 0) /* error? */ {
        param_cstr_error_exit();
    }
    return buf;
}
/** NOTE: this output mkl-dnn strings, where dilation 0 means "no dilation".
 * vednn uses dilation 1 for no dilation (which is more reasonable, actually).
 *
 * - \b SO... you may use the output in a \c -M PARAMETERFILE
 *   - <B>\c -M PARAMETERFILE</B> for mkl-dnn convention, dilation starts at 0
 *   - \c -p PARAMETERFILE for vednn convention, dilation starts at 1.
 *   - If dilations absent, then it doesn't matter (both use "no dilation" default).
 *
 * - \b never include the layer name (output appears in dll symbol names, so
 *   "same" convolutions should all get same symbol name)
 */
inline char const*
param_cstr_short(struct param const* const p, char * const buf, size_t const n){
    int const half_form = ( 1
            && p->inHeight       == p->inWidth
            && p->kernHeight     == p->kernWidth
            && p->outHeight      == p->outWidth
            && p->strideHeight   == p->strideWidth
            && p->padHeight      == p->padWidth
            && p->dilationHeight == p->dilationWidth
            //&& p->inDepth==1
            );
    // TODO: allow half-form for some but not all?
    // ex. ihXiwY_kh3 with no kw assumes kw=3, etc.
    // (to match what readParamString can handle)
#define DPRINT(...) do \
    { \
        int l = snprintf(buffer, rem_len, __VA_ARGS__); \
        if(l<0) param_cstr_error_exit(); \
        buffer += l; rem_len -= l; \
    }while(0)
    if(half_form){ // as per mkl
        int rem_len = n;
        char *buffer = buf;
        // *** NOTE *** mkl-dnn default mb is 2 !
        int f=0; // field count
        if(p->batchNum!=2){ DPRINT("mb%d",p->batchNum); ++f; }
        if(p->group!=1)   { DPRINT("g%d",p->group);     ++f; }
        if(f) DPRINT("_");
        DPRINT( "ic%dih%doc%doh%dkh%d",
                p->inChannel, p->inHeight,
                p->outChannel, p->outHeight,
                p->kernHeight );
        if(p->strideHeight  !=1
                || p->padHeight     !=0
                || p->dilationHeight!=1 ){
            DPRINT("_");
            if(p->strideHeight  !=1) DPRINT("sh%d",p->strideHeight);
            if(p->padHeight     !=0) DPRINT("ph%d",p->padHeight);
            if(p->dilationHeight!=1) DPRINT("dh%d",p->dilationHeight-1); // dil-1 for mkl-dnn string
        }
    }else if(1){ // more aggressive default-or-equal eliminations
        // --> shorter jit symbol names, nicer screen output
        int rem_len = n;
        char *buffer = buf;
        // *** NOTE *** mkl-dnn default mb is 2 !
        int f=0; // field count
        if(1 || p->batchNum!=2){ DPRINT("mb%d",p->batchNum); ++f; } // now always start with "mb"
        if(p->group!=1)   { DPRINT("g%d",p->group);     ++f; }
        //if(f) DPRINT("_");
        DPRINT("_ic%d",p->inChannel);
        DPRINT("ih%d",p->inHeight);
        if(p->inWidth!=p->inHeight) DPRINT("iw%d",p->inWidth);

        DPRINT("_oc%d",p->outChannel);
        DPRINT("oh%d",p->outHeight);
        if(p->outWidth!=p->outHeight) DPRINT("ow%d",p->outWidth);

        DPRINT("_kh%d",p->kernHeight );
        if(p->kernWidth!=p->kernHeight) DPRINT("kw%d",p->kernWidth);
        if(p->strideHeight !=1 || p->strideWidth != 1){
            if(p->strideWidth == p->strideHeight) DPRINT("sh%d",p->strideHeight);
            else DPRINT("sh%dsw%d",p->strideHeight,p->strideWidth);
        }
        if(p->padHeight !=0 || p->padWidth != 0){
            if(p->padWidth == p->padHeight) DPRINT("ph%d",p->padHeight);
            else DPRINT("ph%dpw%d",p->padHeight,p->padWidth);
        }
        if(p->dilationHeight !=1 || p->dilationWidth != 1){ // 1 is libvednn "no dilation"
            if(p->dilationWidth == p->dilationHeight) DPRINT("dh%d",p->dilationHeight);
            else DPRINT("dh%ddw%d",p->dilationHeight,p->dilationWidth);
        }
    }else{ // very long string, every field explicit
        param_cstr(p,buf,n);
    }
#undef DPRINT
    return buf;
}

inline void
dumpParam(struct param* p, char const* dirn, char const* other){
    fprintf(stdout,
            "%30s %11s %-4d %11s %-4d %11s %-5d %4s %-4d x %-4d %11s %-5d %5s %-4d x %-4d"
            " %6s %-2d x %-2d %6s %-2d x %-2d %6s %-2d x %-2d %6s %-2d x %-2d %s%s",
            p->pName,
            "batchNum",p->batchNum, "group",p->group,
            "inChannel",p->inChannel, "inHW", p->inHeight, p->inWidth,
            "outChannel",p->outChannel, "outHW", p->outHeight, p->outWidth,
            "kern",  p->kernHeight,     p->kernWidth,
            "stride",p->strideHeight,   p->strideWidth,
            "pad",   p->padHeight,      p->padWidth,
            "dilate",p->dilationHeight, p->dilationWidth,
            dirn, other);
    fflush(stdout);
}
inline void
dumpParamCSV_title(){
    printf ("# convolution name, batch, group, inChannel|Height|Width,"
            " outChannel|Height|Width, kernHeight|Width, strideHeight|Width,"
            " padHeight|Width, dilationHeight|Width, dirn, time(msec)[min,max], diff\n");
}
inline void
dumpParamCSV(struct param* p, char const* dirn, char const* other){
    printf("%s, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %s%s",
            p->pName, p->batchNum, p->group,
            p->inChannel, p->inHeight, p->inWidth,
            p->outChannel, p->outHeight, p->outWidth,
            p->kernHeight, p->kernWidth,
            p->strideHeight, p->strideWidth,
            p->padHeight, p->padWidth,
            p->dilationHeight, p->dilationWidth,
            dirn, other);
    fflush(stdout);
}

/** fwd ops brute-force count, as in bench-dnn */
inline unsigned long long count_ops(struct param const* p){
    unsigned long long sp_ops = 0;
    //for (int od = 0; od < this->od; ++od) {
    for (int oh = 0; oh < p->outHeight; ++oh) {
    for (int ow = 0; ow < p->outWidth; ++ow) {
        //for (int kd = 0; kd < this->kd; ++kd) {
        //    const int id = od * this->sd - this->pd + kd * (this->dd + 1);
        //    if (id < 0 || id >= this->id) continue;
            for (int kh = 0; kh < p->kernHeight; ++kh) {
                //const int ih = oh * this->sh - this->ph + kh * (this->dh + 1);
                const int ih = oh * p->strideHeight - p->padHeight + kh * (p->dilationHeight);
                if (ih < 0 || ih >= p->inHeight) continue;
                for (int kw = 0; kw < p->kernWidth; ++kw) {
                    //const int iw = ow * this->sw - this->pw + kw * (this->dw + 1);
                    const int iw = ow * p->strideWidth - p->padWidth + kw * (p->dilationWidth);
                    if (iw < 0 || iw >= p->inWidth) continue;
                    sp_ops += 1;
                }
            }
        //}
    }
    }
    //}

    //ops = 2 * this->mb * this->oc * this->ic / this->g * sp_ops;
    return 2 * sp_ops * p->batchNum * p->outChannel * p->inChannel / p->group;
}

#ifdef __cplusplus
}//extern "C"
#endif //C++

// vim: et ts=4 sw=4 cindent cino=^=l0,\:0,N-s syntax=cpp.doxygen
#endif // CONV_TEST_PARAM_H
