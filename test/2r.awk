# This awk script searches each line of a "combined" summary from jitconv
# Preprocess like:
#    sed '/.*combined/,/.*Legend/!d;//d;/^$/d' "${infile}"
# and pipe the output into 
#    gawk -f 2r.awk -
# Combined multiple single-run files (output from dtree.sh):
#   { for i in `seq 2 30`; do cat resnext-t8-mb1-/resnext-t8-mb1-${i}.log
#     | sed '/.*combined/,/.*Legend/!d;//d;/^$/d'
#     | gawk -f 2r.awk; done;
#   } | gawk 'p==1 && /^param/{next} /^param/{p=p+1} //{print}'
#   > foo.csv
#   # after the sed, we have each run separated by a '^param' line
#   # and all-but-first of those get removed in the last gawk step
# or easier, get all the files (in numeric order with ls -v)
#   pfx='resnext-t8-'
#   { for f in `ls -v ${pfx}/${pfx}*log`; do
#       sed '/.*combined/,/.*Legend/!d;//d;/^$/d' "${f}" | gawk -f 2r.awk;
#     done; } | gawk 'p==1 && /^param/{next} /^param/{p=p+1} //{print}' > ${pfx}r.csv

function print_legend() {
	printf("%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n",
	       "param", "name", "layertype", "layer",
	       "best", "reps", "threads",
	       "ms", "Gflops", "err")
}
BEGIN{ print_legend();
	fmt="%s,%s,%s,%s,%d,%d,%d,%g,%g,%g\n";
	threads=8;
	#run=0
}
!threads && /omp_set_num_threads\(/{ patsplit($1,tmp,/[()]/); threads=tmp[2];
	print "threads ",threads;
}
!threads && /Original libvednn vednn_get_num_threads/{ threads = $10;
	printf "threads", threads;
}
/^[IJR*]/{
	# example
	# I            cnvFwd-gemm_mb  |    9x    21.527 ms ~0.0000 776.73G "Rx101:1:conv4b" mb8g32_ic2048ih45iw80_oc2048oh23ow40_kh3sh2ph1
	impltype=$1
	impl=$2
	#isbest=($3=="**"? 1: 0) # also means "next RUN" starts with this entry
	isbest = $3
	if (isbest == "**") {isbest=1} else {isbest=0}
	#if (isbest) {run = run + 1;}
	reps=$4; sub(/x/,"",reps)
	ms=$5
	if(substr($6,1,1)=="t") {threads = substr($6,2);}
	err=$7; sub(/~/,"",err)
	gflops=$8;
	if(index(gflops,"G")) {sub(/G/,"",gflops); }
	else if(index(gflops,"M")) {sub(/M/,"",gflops); gflops*=1e-3;}
	else if(index(gflops,"k")) {sub(/k/,"",gflops); gflops*=1e-6;}
	paramname=$9
	sub(/<gemm:Fwd-Ref>/,"",paramname)
	sub(/^_*n/,"",paramname)
	param = $10
	printf(fmt, param, paramname, impltype, impl,
	       isbest, reps, threads,
	       ms, gflops, err);
}
