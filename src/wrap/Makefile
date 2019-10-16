#
# For quick'n'dirty testing... official way is cmake
#
# This can test compile, but won't link to cmake build path.
#
FTRACE=YES
OPENMP=YES

SRCS	=
TARGETS	= 
LIBSRCS	= vednnConvolution_ok.c vednnConvolutionLists.c

OBJS	= $(SRCS:%.c=%.o)
#USE_FPIC?=-fPIC
USE_FPIC=-fPIC

ifeq ($(USE_FPIC),-fPIC)
LIBTARGETS = ../libvednnx.so
else
LIBTARGETS = ../libvednnx.a
endif

VE_BIN_DIR	= /opt/nec/ve/bin

CC	= $(VE_BIN_DIR)/ncc
COPTS	= -O4 -finline-functions -Wall
X86_CFLAGS = $(COPTS)
CFLAGS	= $(X86_CFLAGS) -traceback -report-all -assembly-list
LDLIBS	= $(COPTS) -lm
AR	= $(AURORA_BIN_DIR)/nar

LLVM_DIR	= /proj/share/local/x86_64/llvm
CLANG		= ${LLVM_DIR}/bin/clang
CLANG_FLAGS	= -O3 ${USE_FPIC} -fno-vectorize -fno-slp-vectorize -fno-crash-diagnostics -target ve -I../

BLAS_DIR	= /opt/nec/ve/nlc/1.0.0
BLAS_INC_DIR	= $(BLAS_DIR)/include
BLAS_LIB_DIR	= $(BLAS_DIR)/lib
COPTS		+= -I$(BLAS_INC_DIR) 

VEDNN_DIR	= ..
COPTS  += -I${VEDNN_DIR} -I${VEDNN_DIR}/C

ifeq ($(OPENMP),YES)
CFLAGS      += -fopenmp -DVEDNN_USE_OPENMP
CLANG_FLAGS += -fopenmp -DVEDNN_USE_OPENMP
LDLIBS	+= -L${VEDNN_DIR} -Wl,-rpath,$(realpath $(VEDNN_DIR)) -lvednn_openmp
LDLIBS  += -L$(BLAS_LIB_DIR) -lblas_openmp
LDLIBS  += -fopenmp
else
LDLIBS	+= -L${VEDNN_DIR} -Wl,-rpath,$(realpath $(VEDNN_DIR)) -lvednn
LDLIBS	+= -L$(BLAS_LIB_DIR) -lblas_sequential
endif

ifeq ($(FTRACE),YES)
VEPERF_DIR	= /usr/uhome/aurora/mpc/pub/veperf/180218-ELF
VEPERF_INC_DIR	= $(VEPERF_DIR)/include
VEPERF_LIB_DIR	= $(VEPERF_DIR)/lib
CFLAGS		+= -I$(VEPERF_INC_DIR) -DFTRACE
LDLIBS		+= -L$(VEPERF_LIB_DIR) -Wl,-rpath,$(realpath $(VEPERF_LIB_DIR)) -ftrace
endif


all: $(LIBTARGETS) $(TARGETS)

LIBHDRS:=../vednn.h ../C/vednnConvolutionForward.h ../C/vednnConvolutionForwardAddBias.h \
	../C/vednnConvolutionBackwardData.h ../C/vednnConvolutionBackwardFilter.h
vednnConvolution_ok.s: vednnConvolution_ok.c $(LIBHDRS) vednnConvolution_ok.h
vednnConvolutionLists.s: vednnConvolutionLists.c $(LIBHDRS) vednnConvolutionLists.h vednnConvolution_ok.h
%.lo: %.s
	$(CC) $(CFLAGS) -o $@ -c $<

%.s: %.c
	#$(CLANG) $(CLANG_FLAGS) -o $@ -S $<
	$(CC) $(CFLAGS) -o $@ -S $<

$(LIBTARGETS): $(patsubst %.c,%.lo,$(LIBSRCS))
	@# stick RPATHS into library
	$(CC) -shared -fPIC $(LDLIBS) -o $@ $^

X86_CC:=gcc -std=c11 $(X86_CFLAGS)
X86_CXX:=g++ -std=c++11 $(X86_CFLAGS)
.PHONY: vednnjit-chk vednnjit-chdrs vednnjit-cxxhdrs vednnjit-src
vednnjit-chk: vednnjit-chdrs vednnjit-cxxhdrs vednnjit-src \
	vednnJit.cpp.x86.o
vednnjit-chdrs : vednnJitDetail.h vednnJitUtil.h vednnJit.h
	@for f in $^; do \
		echo "Header check: $$f"; \
		$(X86_CC) -dD -E "$${f}" > "$${f}.i"; \
		$(X86_CC) -c "$${f}"; \
	done
	@echo "Good, vednnjit C headers pass basic sanity checks"
	@rm -f *.gch
vednnjit-cxxhdrs : vednnJitDetail.hpp
	@for f in $^; do \
		echo "Header check: $$f"; \
		$(X86_CXX) -dD -E "$${f}" > "$${f}.i"; \
		$(X86_CXX) -c "$${f}"; \
	done
	@echo "Good, vednnjit C++ headers pass basic sanity checks"
	@rm -f *.gch
vednnjit-src : vednnJit.cpp
	@for f in $^; do \
		echo "Compile check: $$f"; set -x; \
		$(X86_CXX) -dD -E "$${f}" > "$${f}.i"; \
		$(X86_CXX) -c "$${f}" -o "$${f}-x86.o"; \
	done
	@echo "Good, vednnjit C++ headers pass basic sanity checks"
	@rm -f *.gch
%.cpp.x86.o: %.cpp
	$(X86_CXX) -dD -E $< > $*.cpp.x86.i
	$(X86_CXX) -c $< -o $@
clean:
	rm -f $(OBJS) $(TARGETS) *.L *.O *.o *.lo *.s ftrace.out *~ *.i *.gch
	$(MAKE) -C vconv $@
realclean:
	rm -f $(LIBTARGETS)
	$(MAKE) -C vconv $@
