#### gen-dnn convolutions

1. in gen-dnn, `cd src; ./mk-vconv.sh` and copy vconv-ve.tar.gz to vednn project directory
2. `make`

- compile local copy of gen-dnn/src/vconv/test\_md.c linking with -lvconv-ve -lgemm-ve .
  - demo mkldnn\_memory\_desc\_init

