
## Install llvm

llvm is used to build vednn.
Install llvm for VE from [[https://github.com/sx-aurora-dev/llvm-project]].

## Build
```
$ mkdir build
$ cd build
$ cmake -DLLVM_DIR=<llvm_install_prefix>/lib/cmake/llvm [OPTIONS] ..
$ make && make install

<CMake Options>
-DNCC=<Path of ncc>           [Default : /opt/nec/ve/bin/ncc]
-DNCXX=<Path of nc++>         [Default : /opt/nec/ve/bin/nc++]
-DUSE_OPENMP=ON/OFF           [Default : ON ]
-DBUILD_SHARED_LIB=ON/OFF     [Default : OFF]

-DCMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT	[Default : OFF ( Install to /usr/local ) ]
-DCMAKE_INSTALL_PREFIX=<PATH> [Default : /usr/local]
```

## Run some tests

Bring **ncc** into the PATH, with all that's needed. For example by doing:
```
export NLC_PREFIX=/opt/nec/ve/nlc/2.2.0
. ${NLC_PREFIX}/bin/nlcvars.sh
export PATH=/opt/nec/ve/bin:$PATH
```

Build and run some tests:
```
$ cd ../test
$ CC=ncc CXX=nC++ make vednn_conv_test vednn_linear_test vednn_pool_test
$ ./vednn_conv_test   -H 0.8e9 -p params/conv/alexnet.txt   -T ConvForward
$ ftrace
$ ./vednn_linear_test -H 0.8e9 -p params/linear/alexnet.txt -T LinearForward
$ ftrace
$ ./vednn_pool_test   -H 0.8e9 -p params/pool/alexnet.txt   -T MaxPoolForward
$ ftrace
```

## additions:
- src/wrap/ has:
  - iteration over low-level impls (support comparing speeds of all applicable impls)
  - jit API idea
  - these extensions go into libvednxx ('x' for extensions)
  - src/test/ codes that use libvednnx still have to be copied over

- src/wrap/vconv has:
  - a subset of gen-dnn (v0.16) that supports im2col convolution (+3d),
  - as well as some useful C++ objects like parallel_nd_iterator.
  - (not fully hooked up to src/C/ yet, mostly for speed comparison)

- a simple im2col-gemm, good for low minibatch until parallel_nd intrinsics
  get merged into this branch. im2col should be done with intrinsics, though.
- probably should decide whether libvednn should use nlc libcblas,
  as it is likely faster than the C++ gemm grabbed from mkl-dnn
- Method:
  - extended_sgemm can be compiled with USE_CBLAS flag,
    and linked with nlc version of libcblas

### jit (testing only)
- in test/ directory, extra targets use Makefile.big to build jitconv (main test program)
- When you update vejit.tar.gz (a product of the jit support project),
```
# make sure ncc, nc++ are in the $PATH
export CC=ncc CXX=nc++
cd test
make -f Makefile.big vejit-unpack      # will install it under test/vejit/
make -f Makefile.big VEDNN_DIR=../build
```
