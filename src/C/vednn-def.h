#ifndef VEDNN_DEF_H
#define VEDNN_DEF_H
#include <stddef.h> //size_t

#ifdef VEDNN_USE_OPENMP
#include <omp.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

/** \c  __vednn_init() always sets this to env OMP_NUM_THREADS. which is not nec. same as omp max
 * threads! */
extern int __vednn_omp_num_threads ;

#if 0
static inline void vednn_set_num_threads(int const threads){
#ifdef VEDNN_USE_OPENMP
    __vednn_omp_num_threads = threads;
#else
    __vednn_omp_num_threads = 0;
#endif
}
static inline int vednn_get_num_threads(){
    return __vednn_omp_num_threads;
}
#endif

/// \group vednn scratchpads
/** Scratchpads are local to the current process (or libvednn load/unload).
 * Scratchpads are nicely-sized persistent regions, reducing malloc calls.
 * They have ALIGNMENT at least 16, and actual length is the desired length
 * rounded up to at least a multiple of 16 bytes. Note: x86 also page-aligns
 * these at 2MB pages. What governs 2M vs 64M pages on Aurora? */
//@{
/** Resize to \c size bytes and access vednn global scratchpad.
 * Initialized during \c __vednn_init, via \c vednn_init_global_scratchpads().
 * Modeled after mkldnn::impl::scratchpad_t.
 * This is a general-purpose read-write scratchpad, usable in omp wrapper functions.
 */
void* vednn_scratchpad_shared(size_t bytes);

/** thread-local re-usable scratchpads (e.g. for omp threads).
 * Such scratchpads will be 'local' to each OpenMP thread.
 *
 * These thread-locals will leak, except for master thread, because
 * the client does not 'free' them, and OpenMP has no thread
 * cancellation hooks.
 */
void* vednn_scratchpadTLS(size_t bytes);

/** Resize to \c floats and, if resized, initialize all values to 1.0f.
 * Client is expected to treat this scratchpad as const memory. */
float* vednn_scratchpad_float_ones(size_t floats);

/** Bump ref counts so global scratchpads are grow-only
 * to reduce malloc calls.
 *
 * Each thread will create its own global scratchpad area, so
 * use this from layer wrapper (in process thread) before omp calls,
 * and pass the pointer to omp threads (I think).
 *
 * Scratchpads have large alignment, and actual malloced size is always
 * rounded upward to a multiple of 16.
 *
 * These functions are automatically called via __vednn_init / _vednn_free
 * when your process runs (or library gets loaded/unloaded).
 */
void vednn_init_global_scratchpads(); // called during __vednn_init
void vednn_free_global_scratchpads(); // called during __vednn_init


//@}

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

#define VEDNN_INVALID_PRINTF_ret(F,L,...) do{ \
    fprintf(stderr,"\n%s:%lu INVALID PARAM ",__FILE__,(long unsigned)__LINE__); \
    fprintf(stderr,__VA_ARGS__); \
    return VEDNN_INVALID_PARAM; \
}while(0)
#define VEDNN_INVALID_PRINTF_RET(...) VEDNN_INVALID_PRINTF_ret(__FILE__,__LINE__,__VA_ARGS__)

/** For __vr \c VR, as int64_t, calculate { VM[i]=1 iff 0<=VR[i]<END } 256-bit mask reg.
 * Ex. __vm256 vm23 = MASK_0TO(vrw,inWidth) to check that input pixels vrw[i] lie
 *     in range [0,inWidth). */
#define MASK_0TO(VREG,END)  _ve_andm_mmm( \
    _ve_vfmkl_mcv(VECC_GE, VREG), /* 0<=VREG[i] */ \
    _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv((END),VREG)) )

#define VEL_MASK_0TO(V_INT,END,VL) _vel_vfmklge_mvml(V_INT /* >= 0 */, \
			_vel_vfmklgt_mvl( /* && > END */ \
				_vel_vcmpsl_vsvl( END, V_INT, VL),VL),VL)

/** Declare a `__vm512 M512` variable that sets VM[i] and
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
// vim: ts=4 sw=4 et ai
#endif /* VEDNN_DEF_H */
