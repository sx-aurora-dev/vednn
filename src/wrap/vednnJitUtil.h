#ifndef VEDNNJITDETAIL_H
#define VEDNNJITDETAIL_H
/** \file
 * preliminary api, unimplemented.
 */
#include "vednnx.h"

#ifdef __cplusplus
extern "C" { //}
#endif

/** Convert \e param info to a canonical suffix string.
 * \return NULL on any error (inconsistent/impossible \e param settings)
 *
 * - Symbol names are made up of three parts:
 *   - 1. short layer type, eg convFwd
 *   - 2. jit generator version, eg 1q or 6 or d1s1k3p1, ...
 *   - 3. symbol suffix
 *
 * Constructors will iterate through jit generators searching for a "best" one,
 * and create the vednnSym_t::symbol name using this returned suffix.
 *
 * This string is the equivalent of 'param_cstr_short' in old test codes.
 * This string follows mkl-dnn conventions as far as possible.
 */
char const* vednnSymConvolutionForwardSuffix( VEDNN_PARAMS_CONV_FORWARD );
// etc. for other layers

#ifdef __cplusplus
}//"C"
#endif
// vim: et ts=4 sw=4 cindent cino=^0,=0,l1,\:0,=s,N-s,g-2,h2 syntax=cpp.doxygen
#endif // VEDNNJITDETAIL_H
