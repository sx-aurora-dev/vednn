#!/bin/bash
# To avoid out-of-memory issues running a large number of tests, this script reads
# tests from testfile one at a time.  This means algorithm dominance output is no
# longer useful, but at least the tests run.  Instead, we create some summary
# files to compare most important timings.
# Mods:
#   awk, instead of head -n
#   PARAMETER_FILE line missing end-quote on named tests?
#   adjust ve_cmpconv.c log output when doJit=0 so this script behaves
#   put single-log files into a subdirectory
make -f Makefile.big ve_cmpconv
if [ $# -eq 2 ]; then 
	pfx="$1"
	testfile="$2"
else
	pfx="mini-"
	testfile="params/txt/mini.txt"
fi
nlines=`wc ${testfile} | awk '{print $1}'`
make ve_cmpconv &&
	{ for ii in `seq 2 ${nlines}`; do
		echo -n "$pfx$ii.log ... ";
		./ve_cmpconv -r 3  -M `cat ${testfile} | awk "NR==$ii"` >& $pfx$ii.log;
		tail -n1 $pfx$ii.log;
	done;
}
grep '^ max jit DIFF' ${pfx}*.log
# file list sorted by line number suffix == line number in ${testfile}
pfxfiles=`ls -1 ${pfx}*.log | sort -t '-' -k 2,2n`
# print the impls in speed order
{ for f in ${pfxfiles}; do
	#params=`head -n20 "${f}" | grep '^PARAMETER' | awk '{print $4;}'`;
	#params=`awk 'BEGIN{FS="=";}/^PARAMETER/{print $2;exit;}' ${f}` ;
	#echo "$f  $params";
	# XXX log file misses terminating '"' quote character?
	#paramd=`sed -n -e '/^mkl-dnn format/p' -e '/^ max DIFF/p' ${f} | tr '\\n' ' ' | cut -d' ' -f4-`;
	paramd=`awk 's==0&&/^mkl-dnn format/{s=1; printf("%s    ", $4);} s==1&&/^ max DIFF/{print; exit}' "${f}"`
	echo "$f    $paramd";
	sed -n -e '/JIT\ impls\ combined/,/Legend/ {/^[IJ*]/ p}' ${f} | nl -s' ';
done; } 2>&1 | tee ${pfx}cmp.summary

# print all fastest impls, until the first JIT one
#{ for f in ${pfxfiles}; do params=`head "${f}" | grep '^PARAMETER' | awk '{print $4;}'`; echo "$f  $params"; sed -n -e '/JIT\ impls\ combined/,/Legend/ {/^[IJ*]/ p}' ${f} | nl -s' ' | sed -n -e '/^ *1 /,/ J / s/^ *[0-9]\+ \+// p'; done; } 2>&1 | tee ${pfx}fastest.summary
{ for f in ${pfxfiles}; do
	#params=`head -n20 "${f}" | grep '^PARAMETER' | awk '{print $4;}'`;
	params=`awk 'BEGIN{FS="=";}/^PARAMETER/{print $2;exit;}' ${f}` ;
       	echo "$f  $params";
       	sed -n -e '/JIT\ impls\ combined/,/Legend/ {/^[IJ*]/ p}' ${f} | nl -s' ' \
		| sed -n -e '/^ *1 /,/ J / p';
done; } 2>&1 | tee ${pfx}fastest.summary

# Hone in in any specific interests:
# print fastest two and 'd1q' timings	 (if doJit=1 in ve_cmpconv.c, may use jit impl Forward1q)
{ for f in ${pfxfiles}; do
	#params=`head -n20 "${f}" | grep '^PARAMETER' | awk '{print $4;}'`;
	params=`awk 'BEGIN{FS="=";}/^PARAMETER/{print $2;exit;}' ${f}` ;
	echo "$f  $params";
	sed -n -e '/JIT\ impls\ combined/,/Legend/ {/^[IJ*]/ p}' ${f} | nl -s' ' \
		| awk '/^ *[12] /{print} /-std/{print} /d1q/{print}'
	# less easy to read : sed -n -e '/\(^ *[12] \)\|\(-std \)\|\(d1q\)/ p';
done; } 2>&1 |tee ${pfx}d1q.summary

# unclutter main dir, leaving the .summary files
if [ -d "$pfx" ]; then rmdir "${pfx}"; fi
mkdir "${pfx}";
for f in ${pfxfiles}; do mv "${f}" "${pfx}"/; done
