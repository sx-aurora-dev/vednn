FTRACE=YES
OPENMP=YES
VEDNN_DIR=../build/

# If no tarballs, just use ../build/ as usual
# Otherwise unpack an appropriate tarball into a local VEDNN_DIR
ifeq ($(FTRACE),YES)
TGZ:=$(wildcard ../vednn-ftrace1.tar.gz)
ifneq (,$(TGZ))
VEDNN_DIR:=./vednn-ftrace1
endif
else
TGZ:=$(wildcard ../vednn.tar.gz)
ifneq (,$(TGZ))
VEDNN_DIR:=./vednn
endif
endif

TARGETS	= vednn_conv_test vednn_linear_test vednn_pool_test
.PHONY: Makefile all vars
all: vars $(VEDNN_DIR) $(TARGETS)
vars:
	echo 'TGZ = $(TGZ)'
	echo 'VEDNN_DIR = $(VEDNN_DIR)'
#$(TARGETS):| $(VEDNN_DIR) # another way to say we need VEDNN_DIR

./vednn-ftrace1: ../vednn-ftrace1.tar.gz
	rm -rf tmp-vft1; mkdir tmp-vft1
	cd tmp-vft1 && tar xzmf ../$^
	rm -rf $@; mv tmp-vft1/vednn $@; rm -rf tmp-vft1
	ls -l $@/lib $^
./vednn: ../vednn.tar.gz
	rm -rf tmp-v; mkdir tmp-v
	cd tmp-v && tar xzmf ../$^
	rm -rf $@; mv tmp-v/vednn $@; rm -rf tmp-v
	ls -l $@/lib $^

SRCS	= vednn_conv_test.c vednn_linear_test.c vednn_pool_test.c vednn_helper.c convolution_gemm.c linear_gemm.c maxpooling.c
OBJS	= $(SRCS:%.c=%.o)

VE_BIN_DIR	= /opt/nec/ve/bin

CC	= $(VE_BIN_DIR)/ncc
COPTS	= -O4 -finline-functions -traceback -report-all -Wall -assembly-list
CFLAGS	= $(COPTS)
LDLIBS	= $(COPTS) -lm
AR	= $(AURORA_BIN_DIR)/nar

BLAS_DIR	= /opt/nec/ve/nlc/1.0.0
BLAS_INC_DIR	= $(BLAS_DIR)/include
BLAS_LIB_DIR	= $(BLAS_DIR)/lib
CFLAGS		+= -I$(BLAS_INC_DIR) 

CFLAGS	+= -I${VEDNN_DIR}/include

ifeq ($(OPENMP),YES)
LDLIBS	+= -L${VEDNN_DIR}/lib -lvednn_openmp
LDLIBS  += -L$(BLAS_LIB_DIR) -lblas_openmp
LDLIBS  += -fopenmp
else
LDLIBS	+= -L${VEDNN_DIR}/lib -lvednn_sequential
LDLIBS	+= -L$(BLAS_LIB_DIR) -lblas_sequential
endif

ifeq ($(FTRACE),YES)
CFLAGS		+= -DFTRACE
LDLIBS		+= -ftrace
endif



vednn_conv_test : vednn_conv_test.o vednn_helper.o convolution_gemm.o
vednn_linear_test : vednn_linear_test.o vednn_helper.o linear_gemm.o
vednn_pool_test : vednn_pool_test.o vednn_helper.o maxpooling.o

clean:
	rm -f $(OBJS) $(TARGETS) *.L *.O *.s ftrace.out *~
realclean: clean	
	rm -rf ./vednn-ftrace1 ./vednn
#
