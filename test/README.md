### jitconv
- heavy duty convolution tester
- run with .sh scripts for long-running tests (days)
- for long tests, edit Makefile.big and keep only the jit conv fwd 1q and 6 versions
  - otherwise slow jit impls just take too long (and I think 3 and 4 are sometimes buggy)
- jit code production is WIP.

### libvednn default conv fwd
- things are just too complicated with now a few gemms, a couple of parallelizations,
  and jit being always available for generic convolution.
- often selects a too-slow impl
- a plain decision tree with 1400 nodes gets about 93% accuracy for
  selecting the fastest impl, but really want a more elegant solution.

### TODO
- limit size of gemms to avoid too-large malloc during im2col?
- ncc-3.0.27 --> ncc-3.1.23 full retest is probably a good idea.
  - can use same test set of ~10000 'latin hypercube sampling' cases
