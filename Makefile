PRJ:=vednn
all: force-build lib${PRJ}.tar.gz lib${PRJ}-ftrace1.tar.gz test
# unpack one of the distro tarballs only in external projects.
# -ft1 tarball will need to be linked with veperf library
.PHONY: test build force-build clean realclean
lib${PRJ}.tar.gz:
	rm -rf ${PRJ}
	# now for sequential
	rm -rf build-${PRJ} ${PRJ}
	mkdir build-${PRJ} ${PRJ}
	cd build-${PRJ} && cmake -DCMAKE_INSTALL_PREFIX=../${PRJ} -DUSE_OPENMP=OFF ${CMAKE_ARGS} .. 2>&1 | tee ../mk-${PRJ}.log
	cd build-${PRJ} && make VERBOSE=1 >> ../mk-${PRJ}.log 2>&1
	cd build-${PRJ} && make VERBOSE=1 install 2>&1 | tee -a ../mk-${PRJ}.log
	ls -l ${PRJ}
	# now for omp
	echo "--- see mk-${PRJ}_omp.log for the omp build log ---" >> mk-${PRJ}.log
	rm -rf build-${PRJ}_omp
	mkdir build-${PRJ}_omp
	cd build-${PRJ}_omp && cmake -DCMAKE_INSTALL_PREFIX=../${PRJ} -DUSE_OPENMP=ON ${CMAKE_ARGS} .. 2>&1 | tee ../mk-${PRJ}_omp.log
	cd build-${PRJ}_omp && make VERBOSE=1 >> ../mk-${PRJ}_omp.log 2>&1
	cd build-${PRJ}_omp && make VERBOSE=1 install 2>&1 | tee -a ../mk-${PRJ}_omp.log
	# tarball
	rm -f ${PRJ}.tar.gz
	tar cvzf ${PRJ}.tar.gz ${PRJ} 2>&1 | tee -a mk-${PRJ}.log
lib${PRJ}-ftrace1.tar.gz:
	rm -rf ${PRJ}
	# now for sequential + ftrace 1
	rm -rf build-ft1-${PRJ} ${PRJ}
	mkdir build-ft1-${PRJ}
	cd build-ft1-${PRJ} && cmake -DCMAKE_INSTALL_PREFIX=../${PRJ} -DUSE_OPENMP=OFF -DUSE_FTRACE=1 ${CMAKE_ARGS} .. 2>&1 | tee ../mk-ft1-${PRJ}.log
	cd build-ft1-${PRJ} && make VERBOSE=1 >> ../mk-ft1-${PRJ}.log 2>&1
	cd build-ft1-${PRJ} && make VERBOSE=1 install >> ../mk-ft1-${PRJ}.log 2>&1
	# now for omp + ftrace 1
	echo "--- see mk-ft1-${PRJ}_omp.log for the USE_FTRACE=1 omp build log ---" >> mk-${PRJ}.log
	rm -rf build-ft1-${PRJ}_omp
	mkdir build-ft1-${PRJ}_omp
	cd build-ft1-${PRJ}_omp && cmake -DCMAKE_INSTALL_PREFIX=../${PRJ} -DUSE_OPENMP=ON -DUSE_FTRACE=1 ${CMAKE_ARGS} .. >& ../mk-ft1-${PRJ}_omp.log
	cd build-ft1-${PRJ}_omp && make VERBOSE=1 >> ../mk-ft1-${PRJ}_omp.log 2>&1
	cd build-ft1-${PRJ}_omp && make VERBOSE=1 install >> ../mk-ft1-${PRJ}_omp.log 2>&1
	# tarball
	rm -f ${PRJ}-ftrace1.tar.gz
	tar cvzf ${PRJ}-ftrace1.tar.gz ${PRJ} 2>&1 | tee -a mk-ft1-${PRJ}.log
test: build # default build dir might be an assumed install location for tests/Makefile
	ls -l build/lib
	cd test && make realclean
	cd test && make VERBOSE=1 2>&1 | tee mk-test.log
force-build:
	rm -rf build; mkdir build;
	$(MAKE) build	
build: # original tests/Makefile always links against this build directory
	rm -rf build; mkdir build;
	cd build && cmake .. 2>&1 | tee ../mk-build.log
	{ cd build && make VERBOSE=1 install 2>&1 | tee -a ../mk-build.log; } && echo BUILD OK || echo BUILD FAILED, see build/mk-build.log
clean:
	rm -rf build-${PRJ}* build-ft1* ${PRJ} mk-build.log mk-${PRJ}.log mk-${PRJ}_omp.log mk-ft1-${PRJ}.log mkft1-${PRJ}_omp.log
realclean: clean
	rm -rf build ${PRJ}.tar.gz ${PRJ}-ftrace1.tar.gz
