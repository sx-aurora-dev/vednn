#ifndef VEDNN_DEF_H
#define VEDNN_DEF_H

#ifdef __cplusplus
extern "C" {
#endif

#ifdef VEDNN_USE_OPENMP
#include <omp.h>
extern int __vednn_omp_num_threads ;
#endif

#ifdef FTRACE
#include <ftrace.h>
#define FTRACE_BEGIN(...) ftrace_region_begin(__VA_ARGS__)
#define FTRACE_END(...) ftrace_region_end(__VA_ARGS__)
#else
#define FTRACE_BEGIN(...)
#define FTRACE_END(...)
#endif

#define WRAP_RET(FUNC, OMP_WRAPPER, ...) do{ \
	FTRACE_BEGIN(#FUNC); \
	vednnError_t ret = OMP_WRAPPER(FUNC, __VA_ARGS__); \
	FTRACE_END(#FUNC); \
	return ret; \
} while(0)

/** For __vr \c VR, as int64_t, calculate { VM[i]=1 iff 0<=VR[i]<END } 256-bit mask reg.
 * Ex. __vm256 vm23 = MASK_0TO(vrw,inWidth) to check that input pixels vrw[i] lie
 *     in range [0,inWidth). */
#define MASK_0TO(VREG,END)  _ve_andm_mmm( \
    _ve_vfmkl_mcv(VECC_GE, VREG), /* 0<=VREG[i] */ \
    _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv((END),VREG)) )

#define VEL_MASK_0TO(V_INT,END,VL) _vel_vfmklge_mvml(V_INT /* >= 0 */, \
			_vel_vfmklgt_mvl( /* && > END */ \
				_vel_vcmpsl_vsvl( END, V_INT, VL),VL),VL)

/** Declare a `__vm512 M512` variable that sets VM[i} and
 * VM[i+1] to existing \c __vm256 register \c A256 and \c B256.
 *
 * \note Following intrinsics swapped in very old VE clang.
 * But often used as VE_DECL_VM512(vmFoo,vmBar,vmBar).
 * Add compiler version check if important. */
#define VE_DECL_VM512( M512, A256, B256 ) \
  __vm512 M512; \
M512 = _ve_insert_vm512l(M512, A256); /* l ~ VM[i] */ \
M512 = _ve_insert_vm512u(M512, B256)  /* u ~ VM[i+1] */

#define VEL_DECL_VM512( VM512, VM256_I, VM256_INEXT ) \
    __vm512 VM512; \
VM512 = _vel_insert_vm512l(VM512, VM256_I    ); /* l ~ VM[i]   */ \
VM512 = _vel_insert_vm512u(VM512, VM256_INEXT); /* u ~ VM[i+1] */

#ifdef __cplusplus
}//extern "C"
#endif
#endif /* VEDNN_DEF_H */
