
#ifndef __VEDNN_UTIL__
#define __VEDNN_UTIL__

template <filterLayout_t FLAYOUT>
static inline int64_t filter_index(
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

#endif
