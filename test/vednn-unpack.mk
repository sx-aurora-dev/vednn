#
# Input:
# 	caller sets up	CC	(usually ncc*, w/ gcc get header-only functionality)
# 			FTRACE	[YES/NO]
# 		and	OPENMP	[YES/NO]
# Main make target is 'vednn-unpack', which is added to the 'all' target
# Also 'vednn-rebuild' which can update the ../install quickly (or tarballs, slowly)
#
# Ouput variables:
#       VEDNN_SUBDIR  vednn install directory (short local path)
# 	LIBVEDNN      set to either vednn_openmp  or vednn_sequential
# 	TGZ           non-blank if VEDNN_DIR installed from tarball
# 	VEDNN_DIR     vednn install directory (possibly an absolute path)
# 	VEDNN_SUFFIX  set to either _openmp or _sequential
# 	LIBVEDNNX     set to either vednnx_openmp or vednnx_sequential
#
# If no tarballs, just use ../build/ and ../install/ as usual
# Otherwise unpack an appropriate tarball into a local VEDNN_DIR
# Tarball install allows linking to libvednn compilation matching
# current FTRACE and OPENMP settings, which might not be desired
# Tarball generation can be quite slow.
#
COMPILE_TYPE:=$(word 1,$(shell $(CC) --version 2>&1))

.PHONY: all vednn-unpack vednn-unpack_vars
all: vednn-unpack vednn-unpack_vars

myfile_path:=$(abspath $(lastword $(MAKEFILE_LIST)))
myfile_dir:=$(dir $(myfile_path))
MAKE_UNPACK:=$(MAKE) -C $(myfile_dir) -f $(myfile_path)

VEDNN_SUBDIR:=../install
# Default to "left-over" build directories (historical default)
# If you REMOVE the tarballs, we'll try an existing build into ../install :
# realpath would require dir to exist; so use abspath
# Both tarballs supply libvednn[x].a, but differ in -ftrace compile flag
ifeq ($(FTRACE),YES)
TGZ:=$(wildcard ../vednn-ftrace1.tar.gz)
ifneq (,$(TGZ))
VEDNN_SUBDIR:=./vednn-ftrace1
endif
else
TGZ:=$(wildcard ../vednn.tar.gz)
ifneq (,$(TGZ))
VEDNN_SUBDIR:=./vednn
endif
endif

VEDNN_DIR:=$(abspath $(myfile_dir)$(VEDNN_SUBDIR))
VEDNNX_DIR:=$(VEDNN_DIR)

ifeq ($(COMPILE_TYPE),ncc)
ifneq ($(TGZ),)           #111111111
$(warning ncc compile -- libvednn from tarball $(TGZ))
ifeq ($(OPENMP),YES)
VEDNN_SUFFIX:=_openmp
else
VEDNN_SUFFIX:=_sequential
endif
LIBVEDNN:=vednn$(VEDNN_SUFFIX)
LIBVEDNNX:=vednnx$(VEDNN_SUFFIX)
./vednn: ../vednnx.tar.gz
	rm -rf tmp-v; mkdir tmp-v
	cd tmp-v && tar xzmf ../$^
	rm -rf $@; mv tmp-v/vednn $@; rm -rf tmp-v
	touch $@; chmod -R ugo-w $@;
	ls -l $@/lib $^
./vednn-ftrace1: ../vednn-ftrace1.tar.gz
	rm -rf tmp-vft1; mkdir tmp-vft1
	cd tmp-vft1 && tar xzmf ../$^
	rm -rf $@; mv tmp-vft1/vednn $@; rm -rf tmp-vft1
	touch $@; chmod -R ugo-w $@;
	ls -l $@/lib $^
vednn-unpack:
	if [ -d '$(VEDNN_SUBDIR)' ]; then chmod -R ugo+rw '$(VEDNN_SUBDIR)'; rm -rf '$(VEDNN_SUBDIR)'; fi
	if [ ! -z "$(VEDNN_SUBDIR)" ]; then \
		echo 'VEDNN_SUBDIR "$(VEDNN_SUBDIR)" being remade'; \
		$(MAKE_UNPACK) ./$(VEDNN_SUBDIR); \
	fi
../vednn.tar.gz: ../Makefile	
	@echo 'Regenerating vednn tarballs (SLOW)'
	-$(MAKE) -C $(myfile_dir)/.. libvednn.tar.gz libvednn.tar.gz
vednn-rebuild: ../vednn.tar.gz
	@# this is a less nice target, that regenerates tarballs -- QUITE SLOW
	$(MAKE_UNPACK) vednn-unpack
else
$(warning ncc compile -- using ../build and ../install default libvednn [openmp no ftrace])
	# non-tarball install : default ../build is [openmp, no ftrace]
VEDNN_SUFFIX:=_openmp
LIBVEDNN:=vednn$(VEDNN_SUFFIX)
LIBVEDNNX:=vednnx$(VEDNN_SUFFIX)
.PRECIOUS: ../build ../install
../build:
	@echo '../build does not exist. Running cmake build of libvednn.'
	(cd .. && mkdir build && cd build && cmake ..)
../install: | ../build
	@echo 'Local ../install out-of-date. Running local install from ../build dir.'
	-$(MAKE) -C $(myfile_dir)/../build -j6 install PREFIX=../install
	-ls -l $@
../install/lib/lib$(LIBVEDNN).a: | ../install
	if [ -f "$@" ]; then $(MAKE_UNPACK) --touch $@; \
		else echo "Ohoh. Did not create expected lib $@"; fi
vednn-unpack: ../install/lib/lib$(LIBVEDNN).a
vednn-rebuild: ../build
	@# this is a less nice target, that forces the 'make install' to run
	@echo 'Updating local ../install from ../build dir.'
	-$(MAKE) -C $(myfile_dir)/../build -j6 install PREFIX=../install
endif	
else #------------ x86?
$(warning CC = $(CC))
$(warning x86 compile -- no VE targets can be built, but headers still available for compile checks)
# for x86 compile checks, grab headers into ../install
./vednn/include/vednn.h:
	echo "------- x86 version of vednn/ ---------------- "
	rm -rf tmp-v; mkdir tmp-v; mkdir tmp-v/include
	cp -uav ../src/vednn.h tmp-v/include/ \
		&& cp -uav ../src/wrap/vednnx.h tmp-v/include/ \
		&& mkdir tmp-v/include/wrap && cp -uav ../src/wrap/vednn[^x]*.h tmp-v/include/wrap/ \
		&& mkdir tmp-v/include/C && cp -uav ../src/C/*.h tmp-v/include/C/ \
		&& ls -lrRst tmp-v
	rm -rf $@; mv -v tmp-v $@; rm -rf tmp-v
vednn-unpack: ./vednn/include/vednn.h
endif #------------ x86?

vednn-unpack_vars:
	@echo 'LIBVEDNN            = ${LIBVEDNN}'
	@echo 'LIBVEDNNX           = ${LIBVEDNNX}'
	@echo 'TGZ                 = ${TGZ}'
	@echo 'VEDNN_SUBDIR        = ${VEDNN_SUBDIR}'
	@echo 'VEDNN_DIR           = ${VEDNN_DIR}'
	@echo 'VEDNNX_DIR          = ${VEDNNX_DIR}'
	-@ls -l '${VEDNN_DIR}/lib'
#
