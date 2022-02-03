#include "detail/vconv/memory_desc_wrapper.hpp"
#include "md_util.h"
#include "detail/vconv/mkldnn_debug.h"
#include <stdio.h>

namespace vconv {

void print_dims( mkldnn_dims_t dims ){ // int[TENSOR_MAX_DIMS]
    for(int i=0; i<TENSOR_MAX_DIMS; ++i){
        printf("%c%d",(i==0?'{':','), dims[i]);
    }
    printf("}");
}

void print_strides( mkldnn_strides_t dims ){ // identical to mkldnn_dims_t
    for(int i=0; i<TENSOR_MAX_DIMS; ++i){
        printf("%c%lld",(i==0?'{':','), (long long int)dims[i]);
    }
    printf("}");
}
void print_data_type( mkldnn_data_type_t data_type ){
    printf("%s",mkldnn_dt2str(data_type));
    //printf( data_type       == mkldnn_f32 ? "f32"
    //        : data_type == mkldnn_s32 ? "s32"
    //        : data_type == mkldnn_s16 ? "s16"
    //        : data_type == mkldnn_s8  ? "s8"
    //        : data_type == mkldnn_u8  ? "u8"
    //        : /*mkldnn_data_type_undef*/ "dt_undef" );

}
void print_blocking( mkldnn_blocking_desc_t *blk ){
    printf("{");
    if(!blk){
        printf("NULL");
    }else{
        printf("block_dims="); print_dims(blk->block_dims);
        printf(",strides[0]="); print_strides(blk->strides[0]);
        printf(",strides[1]="); print_strides(blk->strides[1]);
        printf(",padding_dims="); print_dims(blk->padding_dims);
        printf(",offset_padding_to_data="); print_dims(blk->offset_padding_to_data);
        printf(",offset_padding=%lld",(long long int)(blk->offset_padding));
    }
    printf("}");
}

void print_dims_nice( char const* name, int const ndims, mkldnn_dims_t dims ){
    printf("%s[%d]",(name?name:"dims"),ndims); // name[ndims]{a,b,...}
if(ndims){
    for(int i=0; i<ndims; ++i)
        printf("%c%d",(i==0?'{':','), dims[i]);
    printf("}");
}
}
void print_strides_nice( char const* name, int const ndims, mkldnn_strides_t dims ){
    printf("%s[%d]",(name?name:"dims"),ndims); // name[ndims]{a,b,...}
if(ndims){
    for(int i=0; i<ndims; ++i)
        printf("%c%lld",(i==0?'{':','), (long long int)dims[i]);
    printf("}");
}
}
void print_blocking_nice( int const ndims, mkldnn_blocking_desc_t *blk ){
    printf("{");
    if(!blk){
        printf("NULL");
    }else{
        print_dims_nice("block_dims",ndims,blk->block_dims);
        print_strides_nice(",strides[0]",ndims,blk->strides[0]);
        print_strides_nice(",strides[1]",ndims,blk->strides[1]);
        print_dims_nice(",padding_dims",ndims,blk->padding_dims);
        print_dims_nice(",offset_padding_to_data",ndims,blk->offset_padding_to_data);
        printf(",offset_padding=%lld",(long long int)(blk->offset_padding));
    }
    printf("}");
}
}//vconv::

using namespace vconv;
using namespace mkldnn::impl;

extern "C" {
int md_equal( mkldnn_memory_desc_t *a, mkldnn_memory_desc_t *b ){
    // via mkldnn::impl we correctly ignoring irrelevant comparisons
    return memory_desc_wrapper(a) == memory_desc_wrapper(b);
}
void print_md_full( mkldnn_memory_desc_t *md ){
    printf(" mkldnn_memory_desc_t{");
    if(!md){
        printf("NULL");
    }else{
        printf(" ndims=%d", md->ndims);
        print_dims( md->dims );
        print_data_type( md->data_type );
        print_blocking( &md->layout_desc.blocking );
    }
    printf("}");
    fflush(stdout);
}

void print_md_nice( mkldnn_memory_desc_t *md ){
    printf(" mkldnn_memory_desc_t{");
    if(!md){
        printf("NULL");
    }else{
        print_dims_nice( "dims", md->ndims, md->dims );
        print_data_type( md->data_type );
        print_blocking_nice( md->ndims, &md->layout_desc.blocking );
    }
    printf("}");
    fflush(stdout);
}
}//"C"
/* vim: set et ts=4 sw=4 cino=^=l0,\:0,N-s: */
