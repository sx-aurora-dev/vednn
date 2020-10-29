### ncc/nc++ code

- front-end public APIs hiding details of src/intrinsics/... impls
- gen-dnn contains a standalone gemm and a standalone convolution
  originating from mkl-dnn v0.16.
  - this is mostly for speed comparison purposes
  - **CHECKME** there may be a better im2col+gemm that uses cblas these days

