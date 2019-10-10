/** Copyright 2019 NEC Labs America
 * see LICENSE file */
/** \file
 * \c mkldnn_memory_desc_init demo. */
#include "vednn2gendnn.h"
#include "md_util.h"
#include "mkldnn_desc_init.h"
#include "mkldnn_debug.h"
#include <stdio.h>
#include <assert.h>

static inline void md_0xFF( mkldnn_memory_desc_t *md ){
    for(int i=0; i<sizeof(mkldnn_memory_desc_t); ++i)
        ((char*)md)[i] = 0xFF;
}

mkldnn_status_t mkmd(int const ndims, mkldnn_memory_format_t const fmt,
        char const* msg)
{    
    int const verbose=1;
    mkldnn_data_type_t dt=mkldnn_f32;
#if 0
    int const dims[12]={1,2,3,4,5,6,7,8,9,10,11,12};
#else
    int dims[(ndims<0? 0: ndims)];
    for(int i=0; i<ndims; ++i) dims[i]=i+1;
#endif
    if(verbose>0){
        printf("\n=== md_0xFF; %d-dim %s",ndims,mkldnn_fmt2str(fmt));
        if(msg) printf(" : %s",msg);
        printf("\n");
    }
    mkldnn_memory_desc_t md;
    md_0xFF(&md);
    if(verbose>1)
        printf(" about to call memory_desc_init\n"); fflush(stdout);
    mkldnn_status_t s = mkldnn_memory_desc_init(&md,
            ndims, dims, dt, fmt);
    if(s != mkldnn_success) printf("ERROR: status = %d\n",s);
    if(verbose>1)
        printf(" back from memory_desc_init\n"); fflush(stdout);
    print_md_full(&md); printf("\n");
    print_md_nice(&md); printf("\n");
    return s;
}
int main(int argc, char** argv){
    if(1){
        printf("\n=== md_0xFF\n");
        mkldnn_memory_desc_t md;
        md_0xFF(&md);
        print_md_full(&md);
    }
    mkmd(666/*don't care*/, mkldnn_format_undef,""); // zero_md(), all zeros (no garbage)
    mkmd(1, mkldnn_x,"");
    mkmd(4, mkldnn_nchw,"");
    {
        int dims[4] = {1,2,3,4};
        mkldnn_memory_desc_t md1;
        mkldnn_memory_desc_init(&md1, 4, dims, mkldnn_f32, mkldnn_nchw);
        printf("\nmd1: "); print_md_nice(&md1); printf("\n");

        // NB vednn often swaps nchw height and width order in structs !!!
        vednnTensorParam_t tp = {DTYPE_FLOAT,1,2,4,3};
        mkldnn_memory_desc_t md2;
        mk_mkldnn_nchw_memory_desc(&tp, &md2);
        printf("\nmd2: "); print_md_nice(&md2); printf("\n");

        assert(md_equal( &md1, &md2 ));
    }

    mkmd(4, mkldnn_nchw,"");
    //mkmd(0, mkldnn_x,""); // acceptable! returns zero_md() [all-zeroes]
    mkmd(0, mkldnn_nchw,""); // zero_md() [all-zeros]
    printf("\nGoodbye test_md.cpp\n");
    return 0;
}
/* vim: set et ts=4 sw=4 cino=^=l0,\:0,N-s: */
