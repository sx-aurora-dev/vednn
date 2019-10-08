
Adding a new **intrinsics** impl:
1. IMPL in `src/intrinsic/Convolution/TYPE/IMPL.c`
1. add IMPL.c to `CMakLists.txt` (same directory)
1. add fn declarations to `src/C/vednnConvolutionTYPE.h`
1. add `libvednnx` support by modifying these files:
---------------------------|------------
   vednnConvolutionLists.c |   easy edit
   vednnConvolution_ok.h   |   easy edit
   vednnConvolution_ok.c   |   harmonize IMPL.c
---------------------------|------------
1. [opt] When IMPL.c is *correct* and *fastest* under some conditions,
make it official by adding to `src/C/vednnConvolutionTYPE.h`

**JIT** impls are orthogonal to *libvednn* [src/C/, src/intrinsic/]
and *libvednnx* [src/wrap/] code.  JIT speed comparisons are done
by compiling JIT generators into the *test/ve\_cmpconv* utility.

---------------------------------------------------
direct_default3b shows an asm-like distribution of the linear functions needed
to calculate innermost-loop addresses.  clang probably does a decent job for
such scalar optimizations, but sometimes still see a tiny speed gain for
the by-hand version.

It is kind of difficult, so this tedious pain should some how be automated
when generating the assembly jit.  Producing clang jit does not need to do
this tedious program transform because clang is very good at hoisting constants
out of loops already.
(i.e. clang jit could be "the same" except for various variables being #define'd
to constant values, at least for initial tests).
--------------------------------------------------

