# This awk script searches each line of a "combined" summary from jitconv
# Preprocess like:
#    sed '/.*combined/,/.*Legend/!d;//d;/^$/d' "${infile}"
# and pipe the output into 
#    gawk -f 2csv.awk -
function print_legend() {
	printf("%30s, %8s, %8s, %8s, %8s, %8s, %8s, %20s, %15s, %20s, %s\n",
	       "layer", "best", "vednn", "std", "gemm", "nongemm", "jit",
	       "\"Best Impl\"", "\"vednn Impl\"", "\"non-gemm impl\"", "\"Jit impl\"", "Params");
}
/\*\*/{
	if(layer){
		printf("%30s, %8.3f, %8.3f, %8.3f, %8.3f, %8.3f, %8.3f, %20s, %15s, %15s, %20s, %s\n",
		       layer,bestG,vednnG,stdG,gemmG,nongemmG,jitG,
		       bestImpl, vednnImpl, nongemmImpl, jitImpl, param);
	}else { print_legend(); }
}
/\*\*/{
	layer=$9; param=$10; bestImpl=$2; bestG=$8;
	sub(/^_*n/,"",layer)
	sub(/<gemm:Fwd-Ref>/,"",layer)
	vednnImpl="NONE"; vednnG=0;
	stdG=0; gemmG=0;
	jitImpl="NONE"; jitG=0;
	nongemmImpl="NONE"; nongemmG=0;
}
/^J/ && jitG==0{ jitImpl=$2; jitG=$8; }
/^I/ && vednnG==0{
	vednnImpl=$2; vednnG=$8; sub(/<gemm:Fwd-Ref>/,"",vednnImpl)
	#print "vednnG", vednnG, vednnImpl;
}
/^I/ && !/gemm/ && !/-std / && nongemmG==0{
	nongemmImpl=$2; nongemmG=$8;
	#print "nongemmG", nongemmG, nongemmImpl;
}
/libvednn-std/{ stdG = $8 }
/cnvFwd-gemm/{ gemmG = $8 }
//{next;}
