# This awk script searches each line of a "combined" summary from jitconv
# Preprocess like:
#    sed '/.*combined/,/.*Legend/!d;//d;/^$/d' "${infile}"
# and pipe the output into 
#    gawk -f 2csv.awk -
function print_legend() {
	printf("%s,%s,%s,%s,%s,%s,%s,%s\n",
	       "param", "layertype", "layer", "reps", "threads", "ms", "Gflops", "err")
}
BEGIN{ print_legend();
	fmt="%s,%s,%s,%d,%d,%g,%g,%g\n";
	threads=8;
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
	isbest=($3=="**"? 1: 0)
	reps=$4; sub(/x/,"",reps)
	ms=$5
	if(substr($6,1,1)=="t") {threads = substr($6,2);}
	err=$7; sub(/~/,"",err)
	gflops=$8;
	if(index(gflops,"G")) {sub(/G/,"",gflops); }
	else if(index(gflops,"M")) {sub(/M/,"",gflops); gflops*=1e-3;}
	else if(index(gflops,"k")) {sub(/k/,"",gflops); gflops*=1e-6;}
	layername=$9
	sub(/^_*n/,"",layername)
	sub(/<gemm:Fwd-Ref>/,"",layername)
	param = $10
	printf(fmt, param, impltype, impl, reps, threads, ms, gflops, err);
}
