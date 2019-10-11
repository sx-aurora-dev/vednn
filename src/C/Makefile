.PHONY: all hdr-check compile-check clean force
all: hdr-check compile-check
force:
INC:=-I.. -I. -Ivconv/include/gen-dnn/vconv/ -Ivconv/include/gen-dnn/vgemm
DEF:=
# openmp?
DEF+=-DVEDNN_USE_OPENMP=1 -fopenmp
# ftrace?
INC+=-I/opt/nec/ve/include
DEF+=-DFTRACE

hdr-check:
	gcc -std=c11 $(INC) $(DEF) vednn-def.h
	g++ -std=c++11 $(INC) $(DEF) vednn-def.hpp
	gcc -std=c11 $(INC) $(DEF) convolution_gemm.h
	rm -f *.gch
	@echo -e "hdr-check PASSED\n\n"
%.check.o: %.cpp force
	g++ -std=c++11 $(INC) $(DEF) -Wall -c $< -o $@
	@echo -e "\n\n"
%.check.o: %.c force
	gcc -std=c11 $(INC) $(DEF) -Wall -c $< -o $@
	@echo -e "\n\n"
compile-check: vednnInitScratchpad.check.o \
	convolution_gemm.check.o
	rm -f *.check.o
	@echo -e "compile-check PASSED\n"
clean:
	rm -f *.gch *.check.o

