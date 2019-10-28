.PHONY: all hdr-collect hdr-check compile-check clean force
all: hdr-check compile-check
force:
INC:=-I.. -I. -Ivconv/include/gen-dnn/vconv/ -Ivconv/include/gen-dnn/vgemm
DEF:=
# openmp?
DEF+=-DVEDNN_USE_OPENMP=1 -fopenmp
# ftrace?
INC+=-I/opt/nec/ve/include
DEF+=-DFTRACE

vconv/include: ../../vconv-ve.tar.gz
	$(MAKE) -C vconv unpack
hdr-collect: vconv/include
	@if [ ! -d gen-dnn ]; then mkdir -v gen-dnn; fi
	@for f in vconv/include/gen-dnn/vconv/* vconv/include/gen-dnn/vgemm/*; do \
		cp -uav $$f gen-dnn; done
hdr-check: hdr-collect
	gcc -std=c11 $(INC) $(DEF) vednn-def.h
	g++ -std=c++11 $(INC) $(DEF) vednn-def.hpp
	@#gcc -std=c11 $(INC) $(DEF) convolution_gemm.h
	rm -f *.gch
	@echo -e "hdr-check PASSED\n\n"
%.check.o: %.cpp force hdr-collect
	g++ -std=c++11 $(INC) $(DEF) -Wall -c $< -o $@
	@echo -e "\n\n"
%.check.o: %.c force hdr-collect
	gcc -std=c11 $(INC) $(DEF) -Wall -c $< -o $@
	@echo -e "\n\n"
%.s: %.cpp force hdr-collect
	ncc -O3 -g2 $(INC) $(DEF) -S $*.cpp -o $*.s
	ncc -O1 -g2 $(INC) $(DEF) -S $*.cpp -o $*-O1.s
compile-check: vednnInitScratchpad.check.o \
	vednnInit.check.o \
	convolution_gemm.check.o
	rm -f *.check.o
	@echo -e "compile-check PASSED\n"
clean:
	rm -f *.gch *.check.o
	rm -rf gen-dnn

