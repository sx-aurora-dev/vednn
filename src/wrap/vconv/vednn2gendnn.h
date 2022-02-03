#ifndef VEDNN2GENDNN_H
#define VEDNN2GENDNN_H
#include "vednn.h"
#include "detail/vconv/mkldnn_types.h"

#if defined(__cplusplus)
extern "C" {
#endif

// mk_mkldnn_FMT_memory_desc are simplified versions.
//   see common/memory.cpp      for mkldnn_memory_desc_init
//       common/convolution.cpp for conv_desc_init
//   ...

inline mkldnn_data_type_t mk_mkldnn_data_type( dataType_t const dt )
{
    // DTYPE_FLOAT --> mkldnn_f32 ... vednn has only a single data type
    return mkldnn_f32;
}

/** trivial case of \ref memory_desc_wrapper.cpp \c fill_nonblocked.
 * Reproduce here so we can be inline (avoid a C++ func call). */
inline void mk_mkldnn_nchw_memory_desc( vednnTensorParam_t const* tp, mkldnn_memory_desc_t *md )
{
    if(tp){
        md->ndims = 4; // batch, channel, width, height  in NCHW format [only a single format]
        md->dims[0] = tp->batch;
        md->dims[1] = tp->channel;
        md->dims[2] = tp->height;
        md->dims[3] = tp->width;
        //md->dims[4] = 0;
        md->data_type = mk_mkldnn_data_type( tp->dtype );
        md->format = mkldnn_nchw;

        // see gen-dnn/src/common/memory.cpp, mkldnn_memory_desc_init(...)
        //mkldnn_blocking_desc_t foo = {0,0,0,0,0};
        //md->layout_desc.blocking = foo;
        // nchw executes fill_nonblocking(md,{0,1,2,3}) (memory_desc_wrapper.cpp)
        for(int i=0; i<md->ndims; ++i){
            md->layout_desc.blocking.block_dims[i] = 1;
            md->layout_desc.blocking.strides[1][i] = 1;
            md->layout_desc.blocking.padding_dims[i] = md->dims[i];
            md->layout_desc.blocking.offset_padding_to_data[i] = 0;
        }
        md->layout_desc.blocking.offset_padding = 0; 

        // strides
        //   no support for zero-length dimensions
        md->layout_desc.blocking.strides[0][3] = 1;
        md->layout_desc.blocking.strides[0][2] = md->dims[3];                        //tp->height;
        md->layout_desc.blocking.strides[0][1] = md->dims[2]*md->dims[3];             //tp->width * tp->height;
        md->layout_desc.blocking.strides[0][0] = md->dims[1]*md->dims[2]*md->dims[3];  //tp->channel * tp->width * tp->height;
    }else{
        md->ndims=0;
        md->format = mkldnn_format_undef;
    }
}
inline void mk_mkldnn_oihw_memory_desc( vednnFilterParam_t const* fp, mkldnn_memory_desc_t *md )
{
    if(fp){
        md->ndims = 4; // outChannel, inChannel, , width, height  in NCHW format [only a single format]
        md->dims[0] = fp->outChannel;
        md->dims[1] = fp->inChannel;
        md->dims[2] = fp->height;
        md->dims[3] = fp->width;
        md->dims[4] = 0;
        md->data_type = mk_mkldnn_data_type( fp->dtype );
        md->format = mkldnn_oihw;

        // see gen-dnn/src/common/memory.cpp, mkldnn_memory_desc_init(...)
        //mkldnn_blocking_desc_t foo = {0,0,0,0,0};
        //md->layout_desc.blocking = foo;
        // nchw executes fill_nonblocking(md,{0,1,2,3}) (memory_desc_wrapper.cpp)
        for(int i=0; i<md->ndims; ++i){
            md->layout_desc.blocking.block_dims[i] = 0;
            md->layout_desc.blocking.strides[1][i] = 1;
            md->layout_desc.blocking.padding_dims[i] = md->dims[i];
            md->layout_desc.blocking.offset_padding_to_data[i] = 0;
        }
        md->layout_desc.blocking.offset_padding = 0; 

        // strides
        //   no support for zero-length dimensions
        md->layout_desc.blocking.strides[0][3] = 1;
        md->layout_desc.blocking.strides[0][2] = md->dims[3];                        //fp->height;
        md->layout_desc.blocking.strides[0][1] = md->dims[2]*md->dims[3];             //fp->width * fp->height;
        md->layout_desc.blocking.strides[0][0] = md->dims[1]*md->dims[2]*md->dims[3];  //fp->inChannel * fp->width * fp->height;
    }else{
        md->ndims=0;
        md->format = mkldnn_format_undef;
    }
}
inline void mk_mkldnn_x_memory_desc( vednnBiasParam_t const* bp, mkldnn_memory_desc_t *md )
{
    if(bp){
        md->ndims = 1; // outChannel, inChannel, , width, height  in NCHW format [only a single format]
        md->dims[0] = bp->channel;
        md->dims[1] = 0;
        md->data_type = mk_mkldnn_data_type( bp->dtype );
        md->format = mkldnn_x;

        // see gen-dnn/src/common/memory.cpp, mkldnn_memory_desc_init(...)
        //mkldnn_blocking_desc_t foo = {0,0,0,0,0};
        //md->layout_desc.blocking = foo;
        // nchw executes fill_nonblocking(md,{0,1,2,3}) (memory_desc_wrapper.cpp)
        for(int i=0; i<md->ndims; ++i){
            md->layout_desc.blocking.block_dims[i] = 0;
            md->layout_desc.blocking.strides[1][i] = 1;
            md->layout_desc.blocking.padding_dims[i] = md->dims[i];
            md->layout_desc.blocking.offset_padding_to_data[i] = 0;
        }
        md->layout_desc.blocking.offset_padding = 0; 

        // strides
        //   no support for zero-length dimensions
        md->layout_desc.blocking.strides[0][0] = 1;
    }else{
        md->ndims=0;
        md->format = mkldnn_format_undef;
    }
}

inline void mk_mkldnn_convolution_desc(
        vednnTensorParam_t      const* pParamIn,
        vednnFilterParam_t      const* pParamKernel,
        vednnBiasParam_t        const* pParamBias, // may be NULL
        vednnTensorParam_t      const* pParamOut,
        vednnConvolutionParam_t const* cp, // pParamConv
        mkldnn_convolution_desc_t *cd )
{
    cd->primitive_kind = mkldnn_convolution;
    cd->alg_kind = mkldnn_convolution_direct;

    // section changes according to 'prop_kind'
    cd->prop_kind = mkldnn_forward_inference; // you will adjust this
    mk_mkldnn_nchw_memory_desc(pParamIn,    &cd->src_desc);
    mk_mkldnn_oihw_memory_desc(pParamKernel,&cd->weights_desc);
    mk_mkldnn_nchw_memory_desc(pParamOut,   &cd->dst_desc);
    mk_mkldnn_x_memory_desc   (pParamBias,  &cd->bias_desc);
    //cd->diff_src_desc       = cd->src_desc;
    //cd->diff_weights_desc   = cd->weights_desc;
    //cd->diff_dst_desc       = cd->dst_desc;
    cd->diff_src_desc    .format = mkldnn_format_undef; // maybe?
    cd->diff_weights_desc.format = mkldnn_format_undef;
    cd->diff_bias_desc   .format = mkldnn_format_undef;
    cd->diff_dst_desc    .format = mkldnn_format_undef;
    // end of changing section

    cd->strides[0] = 1;
    cd->strides[1] = 1;
    cd->strides[2] = cp->strideHeight;
    cd->strides[3] = cp->strideWidth;
    cd->dilates[0] = 0;
    cd->dilates[1] = 0;
    cd->dilates[2] = cp->dilationHeight-1;
    cd->dilates[3] = cp->dilationWidth-1;
    /* Padding in each spatial dimension. padding[0] is a padding in the
     * beginning (@p padding_l), padding[1] is a padding in the end (@p
     * padding_r). */
    //mkldnn_dims_t padding[2];
    cd->padding[0][0] = 0;
    cd->padding[0][1] = 0;
    cd->padding[0][2] = cp->padHeight;
    cd->padding[0][3] = cp->padWidth;
    cd->padding[1][0] = 0;
    cd->padding[1][1] = 0;
    cd->padding[1][2] = cp->padHeight;   // what about even kernel size (is is still same?)
    cd->padding[1][3] = cp->padWidth;
    cd->padding_kind    = mkldnn_padding_zero;
    cd->accum_data_type = mkldnn_f32;
}


#if defined(__cplusplus)
}//"C"
#endif
/* vim: set et ts=4 sw=4 cino=^=l0,\:0,N-s: */
#endif // VEDNN2GENDNN_H
