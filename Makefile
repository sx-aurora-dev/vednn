PRJ:=vednn
#PRJ:=vednnx
CMAKE_SHARED:='-DBUILD_SHARED_LIB=ON'
CMAKE_ARGS:="-DCMAKE_BUILD_TYPE=Release ${CMAKE_SHARED}"
all: force-build lib${PRJ}.tar.gz lib${PRJ}-ftrace1.tar.gz test
#all: force-build test
# To skip the tarball generation (4 different builds)
# 	make force-build test
#
# unpack one of the distro tarballs only in external projects.
# -ft1 tarball will need to be linked with veperf library
.PHONY: test build empty-build force-build clean realclean
MKJOB:=VERBOSE=1 -j1
lib${PRJ}.tar.gz:
	rm -rf ${PRJ}
	# now for sequential
	rm -rf build-${PRJ} ${PRJ}
	mkdir build-${PRJ} ${PRJ}
	cd build-${PRJ} && cmake -DCMAKE_INSTALL_PREFIX=../${PRJ} -DUSE_OPENMP=OFF ${CMAKE_ARGS} .. 2>&1 | tee ../mk-${PRJ}.log
	cd build-${PRJ} && make ${MKJOB} VERBOSE=1 >> ../mk-${PRJ}.log 2>&1
	cd build-${PRJ} && make VERBOSE=1 install 2>&1 | tee -a ../mk-${PRJ}.log
	ls -l ${PRJ}
	# now for omp
	echo "--- see mk-${PRJ}_omp.log for the omp build log ---" >> mk-${PRJ}.log
	rm -rf build-${PRJ}_omp
	mkdir build-${PRJ}_omp
	cd build-${PRJ}_omp && cmake -DCMAKE_INSTALL_PREFIX=../${PRJ} -DUSE_OPENMP=ON ${CMAKE_ARGS} .. 2>&1 | tee ../mk-${PRJ}_omp.log
	cd build-${PRJ}_omp && make ${MKJOB} VERBOSE=1 >> ../mk-${PRJ}_omp.log 2>&1
	cd build-${PRJ}_omp && make VERBOSE=1 install 2>&1 | tee -a ../mk-${PRJ}_omp.log
	# tarball
	rm -f ${PRJ}.tar.gz
	tar cvzf ${PRJ}.tar.gz ${PRJ} 2>&1 | tee -a mk-${PRJ}.log
lib${PRJ}-ftrace1.tar.gz: # NOTE check USE_FTRACE value (sometimes I switch it to 2)
	rm -rf ${PRJ}
	# now for sequential + ftrace 1
	rm -rf build-ft1-${PRJ} ${PRJ}
	mkdir build-ft1-${PRJ}
	cd build-ft1-${PRJ} && cmake -DCMAKE_INSTALL_PREFIX=../${PRJ} -DUSE_OPENMP=OFF -DUSE_FTRACE=2 ${CMAKE_ARGS} .. 2>&1 | tee ../mk-ft1-${PRJ}.log
	cd build-ft1-${PRJ} && make ${MKJOB} VERBOSE=1 >> ../mk-ft1-${PRJ}.log 2>&1
	cd build-ft1-${PRJ} && make VERBOSE=1 install >> ../mk-ft1-${PRJ}.log 2>&1
	# now for omp + ftrace 1
	echo "--- see mk-ft1-${PRJ}_omp.log for the USE_FTRACE=1 omp build log ---" >> mk-${PRJ}.log
	rm -rf build-ft1-${PRJ}_omp
	mkdir build-ft1-${PRJ}_omp
	cd build-ft1-${PRJ}_omp && cmake -DCMAKE_INSTALL_PREFIX=../${PRJ} -DUSE_OPENMP=ON -DUSE_FTRACE=2 ${CMAKE_ARGS} .. >& ../mk-ft1-${PRJ}_omp.log
	cd build-ft1-${PRJ}_omp && make ${MKJOB} VERBOSE=1 >> ../mk-ft1-${PRJ}_omp.log 2>&1
	cd build-ft1-${PRJ}_omp && make VERBOSE=1 install >> ../mk-ft1-${PRJ}_omp.log 2>&1
	# tarball
	rm -f ${PRJ}-ftrace1.tar.gz
	tar cvzf ${PRJ}-ftrace1.tar.gz ${PRJ} 2>&1 | tee -a mk-ft1-${PRJ}.log
quicktest: # remove tarball build, use build/ and install/ only
	@for f in "lib${PRJ}.tar.gz" "lib${PRJ}-ftrace1.tar.gz"; do \
		if [ -f "$$f" ]; then mv -v "$${f}" "$${f}.old"; fi; done
	${MAKE} -C test clean_vednn
	${MAKE} test0 test1 test2 && \
		{ echo make $@ OK; true; } || { echo make $@ FAILED; false; }
.PHONY: test0 test1 test2 test3 test4 test-resnext
test0:
	cd test && make VERBOSE=1 -f Makefile all ve_cmpconv jitconv && { \
		./vednn_conv_test -H 8e8 -p params/conv/alexnet.txt -T ConvForward; ftrace; \
		./vednn_linear_test -H 0.8e9 -p params/linear/alexnet.txt -T LinearForward; ftrace; \
		./vednn_pool_test   -H 0.8e9 -p params/pool/alexnet.txt   -T MaxPoolForward; ftrace; \
		} \
		&& { echo make $@ OK; true; } || { echo make $@ FAILED; false; }
test1:
	cd test && time make VERBOSE=1 -f Makefile.big jitconv \
		&& echo "test/t8mb8.log ..." \
		&& ./jitconv -t 8 mb8 >& t8mb8.log \
		&& echo "GOOD: test/t8mb8.log!" \
		|| { echo "OHOH: test/t8mb8.log!" && false; }
test2:
	cd test && time make VERBOSE=1 -f Makefile.big jitconv \
		&& echo "test/t8mb9.log ..." \
		&& ./jitconv -t 8 mb9 -p mb1ic31ih1oc31_kh1ph0 2>&1 | tee t8mb9.log \
		&& echo "GOOD: test/t8mb9.log!" \
		|| { echo "OHOH: test/t8mb9.log!" && false; }
test3:
	cd test && time make VERBOSE=1 -f Makefile.big jitconv \
		&& echo "test/alexmb9.log ..." \
		&& ./jitconv -t 8 mb9 -p params/conv/alexnet.txt 2>&1 | tee t8mb9.log \
		&& echo "GOOD: test/t8mb9.log!" \
		|| { echo "OHOH: test/t8mb9.log!" && false; }
test-resnext: # this takes quite some time
	cd test && \
		{ echo "resnext via Makefile.big..." \
		&& { time make VERBOSE=1 -f Makefile.big jitconv resnext-t8.log \
		&& echo "GOOD: test/resnext-t8.log!" \
		|| { echo "OHOH: test/resnext-t8.log!" && false; } \
	       	} && \
		{ echo "resnext-t8-mb8.log..." \
		&& { time make VERBOSE=1 -f Makefile.big jitconv resnext-t8-mb8.log \
		&& echo "GOOD: test/resnext-t8-mb8.log!" \
		|| { echo "OHOH: test/resnext-t8-mb8.log!"; false; } \
		}
test: build
	@for f in "lib${PRJ}.tar.gz" "lib${PRJ}-ftrace1.tar.gz"; do \
		if [ -f "$$f" ]; then mv -v "$${f}" "$${f}.old"; fi; done
	@# default build dir might be an assumed install location for tests/Makefile
	-ls -l build/src
	-cd test && make -f Makefile.tiny realclean
	@#{ cd test && make VERBOSE=1 all ve_cmpconv && BIN_MK_VERBOSE=0 ./ve_cmpconv -r 10; } 2>&1 | tee mk-test.log
	@#} && echo "fails2 tests via dtree.sh..."
	@#&& { time VERBOSE=1 ./dtree.sh fails2- ./fails2.txt -k -r 9 -t 8 -M 2>&1 | tee fails2.log; ! grep FAILED fails2.log; }
	@#&& echo "DONE: test/fails2.log!"
	@#|| { echo "OHOH: test/fails2.log!" && false; }
	{ ${MAKE} test0 \
		&& ${MAKE} test1 \
		&& ${MAKE} test2 \
		&& ${MAKE} test3 \
		&& echo SKIPPING LONG TEST ${MAKE} test-resnext \
		&& echo "vednn make $@ passed"; \
	} 2>&1 | tee mk-test.log; ps=($${PIPESTATUS[@]}); \
	echo "make test ---> mk-test.log, test/{t8mb8,resnext-t8,resnext-t8-mb8}.log : PIPESTATUS $${ps[@]}"; \
	[ "$${ps[0]}" == "0" ];
	@echo "make test OK"
empty-build:
	rm -rf build; mkdir build;
force-build: empty-build build
# close to default build...
build: # if no tarballs, test/original tests/Makefile always links against this build directory
	if [ ! -d build ]; then mkdir build; echo "Fresh build/"; else echo "Remake in build/"; fi
	cd build && cmake --trace -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON .. -DCMAKE_INSTALL_PREFIX=../install \
		-DCMAKE_BUILD_TYPE=RelWithDebInfo \
	       	2>&1 | tee ../mk-build.log
	{ { cd build && make VERBOSE=1 ${MKJOB} install; }; status=$$?; \
		if [ "$$status" == "0" ]; then echo BUILD OK; else echo BUILD FAILED; fi; \
		test "$$status" == "0" ; \
		} 2>&1 | tee -a mk-build.log; ps=($${PIPESTATUS[@]}); \
		echo "make build ---> mk-build.log PIPESTATUS 0:$${ps[0]} 1:$${ps[1]}"; \
		[ "$${ps[0]}" == "0" ];
	@echo "make build OK"
clean:
	rm -rf build-${PRJ}* build-ft1* ${PRJ} mk-build.log mk-${PRJ}.log mk-${PRJ}_omp.log mk-ft1-${PRJ}.log mk-ft1-${PRJ}_omp.log
	$(MAKE) -C test clean
realclean: clean
	-rm -rf build build-vednn build-vednn_omp install vednnx ${PRJ}.tar.gz ${PRJ}-ftrace1.tar.gz
	$(MAKE) -C src/wrap realclean
	$(MAKE) -C test realclean
#
