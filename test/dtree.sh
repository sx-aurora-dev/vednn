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
JITCONV=jitconv
make -f Makefile.big $JITCONV
if [ $# -eq 2 ]; then 
	OPTS="-k -r 9 -t 8 -p"  # -p means dilation 1 convention (libvednn)
	pfx="$1"
	testfile="$2"
	shift; shift; OPTS="${OPTS} $*"
elif [ $# -gt 2 ]; then 
	pfx="$1"
	testfile="$2"
	shift; shift; OPTS="$*"
	# Ex. isz.txt (all with mb1) with mb7 override:
	#  ./dtree.sh iszmb7- -S jit-iszmb7 isz.txt -k -r 9 -t 8 mb7 -p
	# -p MUST be last option (it will be followed by isz.txt)
	# -S was given so runs could go on bothe VE_NODE_NUMBER 0 and 2 concurrently
else
	OPTS="-r 9 -t 8 -M"  # -M means dilation 0 convention (OneDNN)
	pfx="mini-"
	testfile="params/txt/mini.txt"
fi
echo "$0 run parameters:"
echo "pfx     : ${pfx}"
echo "testfile: ${testfile}"
echo "OPTS    : ${OPTS}"
echo ""
nlines=`wc ${testfile} | awk '{print $1}'`
echo "testfile nlines = ${nlines}"
#
# Note: 1st line is assumed to be a number, and skipped
#
make ${JITCONV} &&
	{ for ii in `seq 2 ${nlines}`; do
		# Do not maintain jit impls (huge number of files)
		if [ -d "tmp_cjitConv" ]; then rm -rf tmp_cjitConv; fi
		echo -n "$pfx$ii.log ... ";
		parms=`cat ${testfile} | awk "NR==$ii"`
		cmd="./${JITCONV} ${OPTS} ${parms}"
	        echo -n "${cmd} ... ";
		{ VE_OMP_NUM_THREADS=8 LIBC_FATAL_STDERR_=1 \
			time ${cmd}; \
		       	status=$?; ftrace || true; } >& $pfx$ii.log;
		if [ "${status}" != "0" ]; then echo "FAILED";
		else echo OK; #tail -n1 $pfx$ii.log;
		fi
	done;
}
grep '^ max jit DIFF' ${pfx}*.log
# file list sorted by line number suffix == line number in ${testfile}
#pfxfiles=`ls -1 ${pfx}*.log | sort -t '-' -k 2,2n`
pfxfiles=`ls -1v ${pfx}*.log`
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
# print fastest two and 'd1q' timings	 (if doJit=1 in ${JITCONV}, may use jit impl Forward1q)
{ for f in ${pfxfiles}; do
	#params=`head -n20 "${f}" | grep '^PARAMETER' | awk '{print $4;}'`;
	params=`awk 'BEGIN{FS="=";}/^PARAMETER/{print $2;exit;}' ${f}` ;
	echo "$f  $params";
	sed -n -e '/JIT\ impls\ combined/,/Legend/ {/^[IJ*]/ p}' ${f} | nl -s' ' \
		| awk '/^ *[12] /{print} /-std/{print} /d1q/{print}'
	# less easy to read : sed -n -e '/\(^ *[12] \)\|\(-std \)\|\(d1q\)/ p';
done; } 2>&1 |tee ${pfx}d1q.summary

# unclutter main dir, leaving the .summary files
if [ -d "${pfx}" ]; then rm -r "${pfx}.bak"; mv "${pfx}" "${pfx}.bak"; fi
mkdir "${pfx}";
for f in ${pfxfiles}; do mv "${f}" "${pfx}"/; done

# create a summary .csv file (ordered by numeric suffix, to agree with run order, not alphabetically)
{  for f in `ls -v ${pfx}/${pfx}*log`;
	sed '/.*combined/,/.*Legend/!d;//d;/^$/d' "${f}" | gawk -f 2r.awk;
done; } | gawk 'p==1 && /^param/{next} /^param/{p=p+1} //{print}' \
	> ${pfx}.csv
