FTRACE:=YES
OPENMP:=YES
VE_BIN_DIR=/opt/nec/ve/bin
CC=$(VE_BIN_DIR)/ncc
COMPILE_TYPE:=$(word 1,$(shell $(CC) --version 2>&1))
$(if $(COMPILE_TYPE),,$(error could not determine C compiler type?)) # VE(ncc) vs x86(gcc or ...)

# NEW: set Makefile default to use C++ linker
LINK.o = $(LINK.cc)

# Supply a vednn-unpack target and add it to 'all' target
# Set variables:
# 	VEDNN_DIR vednn install directory, ex. ../install
# 	LIBVEDNN  libvednn library name, ex. vednn_openmp
# More info in the .mk file
include vednn-unpack.mk


TARGETS:=vednn_conv_test vednn_linear_test vednn_pool_test
.PHONY: Makefile all vars
all: vars $(TARGETS)
vars:
	echo 'COMPILE_TYPE      $(COMPILE_TYPE)'
	echo 'LIBVEDNN          $(LIBVEDNN)'
	echo 'MAKECMDGOALS      $(MAKECMDGOALS)'
	echo 'first             $(word 1,$(MAKECMDGOALS))'

SRCS	= vednn_conv_test.c vednn_linear_test.c vednn_pool_test.c vednn_helper.c convolution_gemm.c linear_gemm.c maxpooling.c
OBJS	= $(SRCS:%.c=%.o)

COPTS	= -O4 -finline-functions -traceback -report-all -Wall -assembly-list
CFLAGS	= $(COPTS)
LDLIBS	= $(COPTS) -lm
AR	= $(VE_BIN_DIR)/nar

BLAS_DIR	= /opt/nec/ve/nlc/1.0.0
BLAS_INC_DIR	= $(BLAS_DIR)/include
BLAS_LIB_DIR	= $(BLAS_DIR)/lib
CFLAGS		+= -I$(BLAS_INC_DIR) 
CFLAGS	+= -I${VEDNN_DIR}/include
LDLIBS	+= -L${VEDNN_DIR}/lib -l${LIBVEDNN}

ifeq ($(OPENMP),YES)
CFLAGS	+= -fopenmp
LDLIBS  += -L$(BLAS_LIB_DIR) -lblas_openmp
LDLIBS  += -fopenmp
else
LDLIBS	+= -L$(BLAS_LIB_DIR) -lblas_sequential
endif

ifeq ($(FTRACE),YES)
CFLAGS		+= -ftrace -pthread -DFTRACE
LDLIBS		+= -ftrace
ifeq ($(OPENMP),YES)
LDFLAGS		+= -pthread # ncc-2.x.x recommends
endif
endif

vednn_conv_test : vednn_conv_test.o vednn_helper.o convolution_gemm.o
vednn_linear_test : vednn_linear_test.o vednn_helper.o linear_gemm.o
vednn_pool_test : vednn_pool_test.o vednn_helper.o maxpooling.o
test.log : vednn_conv_test vednn_linear_test vednn_pool_test Makefile
	{ ./vednn_conv_test -H 0.8e9 -p params/conv/alexnet.txt   -T ConvForward    \
		|| echo "Error: vednn_conv_test"; ftrace || true; \
	./vednn_linear_test -H 0.8e9 -p params/linear/alexnet.txt -T LinearForward  \
		|| echo "Error: vednn_linear_test"; ftrace || true; \
	./vednn_pool_test   -H 0.8e9 -p params/pool/alexnet.txt   -T MaxPoolForward \
		|| echo "Error: vednn_pool_test"; ftrace || true; \
	} >& $@
#convolution_gemm-clang.s: convolution_gemm.c
#	$(CLANG) $(CLANG_FLAGS) -o $@ -S $<
convolution_gemm-ncc.s: convolution_gemm.c
	$(CC) $(CFLAGS) -std=gnu11 -report-all -Wall \
		-O3 -fcse-after-vectorization -msched-block \
		-o $@ -S $<
convolution_gemm.o: convolution_gemm-ncc.s
	$(CC) $(CFLAGS) -o $@ -c $<
.PHONY: force
jitconv resnext-t8.log: force
	$(MAKE) -fMakefile.big $@
clean:
	rm -f $(OBJS) $(TARGETS) *.L *.O *.s ftrace.out *~
	$(MAKE) -fMakefile.big clean
clean_vejit:
	@for d in vejit; do \
		if [ -d "$${d}" ]; then \
		echo "Removing subdir $${d}"; \
		chmod -R ugo+rw "$${d}"; rm -rf "$${d}"; \
		fi; \
		done;
clean_vednn:		
	@for d in vednn vednn-ftrace1; do \
		if [ -d "$${d}" ]; then \
		echo "Removing subdir $${d}"; \
		chmod -R ugo+rw "$${d}"; rm -rf "$${d}"; \
		fi; \
		done;
clean_subdirs: clean_vejit clean_vednn # get rid of unpacked tarballs (if any)
	@for d in mnist tmp_cjitConv tmp_cjitConv01 z.dir; do \
		if [ -d "$${d}" ]; then \
		echo "Removing subdir $${d}"; \
		chmod -R ugo+rw "$${d}"; rm -rf "$${d}"; \
		fi; \
		done;
realclean: clean clean_subdirs
	$(MAKE) -fMakefile.big realclean
#
