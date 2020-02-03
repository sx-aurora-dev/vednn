#ifndef PRINT_H
#define PRINT_H
#include "mkldnn_types.h"

#ifdef __cplusplus
extern "C" {
#endif

void print_md_nice( mkldnn_memory_desc_t *md );
void print_md_full( mkldnn_memory_desc_t *md );

int/*bool*/ md_equal( mkldnn_memory_desc_t *a, mkldnn_memory_desc_t *b );
// note memory_desc_wrapper also gives similar_to and consistent_with

#ifdef __cplusplus
}//"C"
#endif
/* vim: set et ts=4 sw=4 cino=^=l0,\:0,N-s: */
#endif // PRINT_H
