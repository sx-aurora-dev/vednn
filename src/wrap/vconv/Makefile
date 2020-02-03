OBJECTS:= # FOO.cpp.o
CHEADERS:=
CXXHEADERS:=
# Optionally, try to force -ve or -x86 library
ifeq ($(TARGET),-x86)
$(warning Not all libvdenn components make sense for -x86 targets)
CC=gcc
CXX=g++
NM=nm
endif
ifeq ($(TARGET),-ve)
CC=ncc
CXX=nc++
NM=nnm
endif

SHELL:=bash
INC:=
OPT:=
AS_OPT:=
CFLAGS:=-fopenmp -std=c11 -Wall
CXXFLAGS:=-fopenmp -Wall
CFLAGS+=-g
CXXFLAGS+=-g
INC:=-I. -I./include/gen-dnn/vconv -I./include/gen-dnn/vgemm

INC+=-I../..
INC+=-I../../wrap
OPT+=-O3 -finline -finline-functions
OPT+=-fopenmp
#OPT+=-DNDEBUG

TARGET:=-ve
ifeq ($(patsubst ncc%,ncc,$(CC)),ncc) # if $(CC) begins with 'ncc'
$(info ncc compiler detected)
AS_OPT+=-report-all
CXXFLAGS+=-std=gnu++14
INC+=-I/opt/nec/ve/nlc/2.0.0/include
#CC_OPT+=-report-all
else
TARGET:=-x86
CXXFLAGS+=-std=c++11
AS_OPT+=-Wa,-adhln# -g
LIBS:=-lcblas -lblas # system libs
endif
INSTALLDIR:=./install${TARGET}
OBJECTS:=${patsubst %.o,%${TARGET}.o,${OBJECTS}}

ifeq ($(TARGET),-x86) # override gcc compiler version (testing...)
CC=gcc
CXX=g++
endif

LIBS:=-L./lib -Wl,-rpath,./lib -lvconv${TARGET} -lvgemm${TARGET}
ifeq ($(patsubst ncc%,ncc,$(CC)),ncc) # if $(CC) begins with 'ncc'
LIBS+=-L/opt/nec/ve/nlc/2.0.0/lib -lcblas -lblas_openmp -Wl,-rpath,/opt/nec/ve/nlc/2.0.0/lib
else
LIBS+=-lblas
endif

#
##################################### all #####################################
#
all: header_check test_mdC test_md
	echo "DONE"
TARGETS:=test_mdC test_md
header_check: include
	gcc -std=c11 $(INC) vednn2gendnn.h
	gcc -std=c11 $(INC) md_util.h
include: ../../../vconv-ve.tar.gz
	chmod -R u+rw include lib src; rm -rf include lib src
	tar xzf ../../../vconv-ve.tar.gz
	chmod -R ugo-w include lib src
	chmod ugo+rw include lib src
	@#ls -lR include lib src
lib: |include
src: |include
.PHONY: unpack Makefile
unpack: include lib src # Forces regenerating include/ lib/ and src/ from tarball
%.c.o: %.c |include
	$(CC) $(CFLAGS) $(INC) -o $@ -c $^
%.cpp.o: %.cpp |include
	$(CXX) $(CXXFLAGS) $(INC) -o $@ -c $^
test_mdC: test_md.c.o
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LIBS)
test_md: test_md.cpp.o md_util.cpp.o |include # expanded functionality
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LIBS)
clean:
	rm -f *.o *.lo *.a *.so a.out *.gch $(TARGETS)
realclean: clean
	for f in include lib src; do \
		if [ -d "$$f" ]; then chmod -R u+rw $$f; rm -rf $$f; fi; \
	done
	rm -f *.log
#last line
