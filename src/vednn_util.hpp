#ifndef VEDNN_UTIL_HPP
#define VEDNN_UTIL_HPP
// Note: including only vednn.h will always pull in this header if compiling C++ code.
#include "vednn.h"
#include <stdint.h>

#ifndef __cplusplus
#error "rename this into a .hpp header?"
#endif

template <filterLayout_t FLAYOUT>
inline int64_t filter_index(
  int64_t k,
  int64_t c,
  int64_t r,
  int64_t s,
  int64_t inChannelGroup,
  int64_t outChannelGroup,
  int64_t kernHeight,
  int64_t kernWidth
)
{
  return ( FLAYOUT == VEDNN_FILTER_LAYOUT_NCHW ) ?
      ((k * inChannelGroup + c) * kernHeight + r) * kernWidth + s :
      (( r * kernWidth + s ) * inChannelGroup + c ) * outChannelGroup + k ;
}
// vim: sw=2 ts=2 et ai
#endif // VEDNN_UTIL_HPP
