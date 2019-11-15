#!/bin/bash
if [ $# -ne 2 -o ! -f "$1" ]; then
	echo '2csv.sh INFILE OUTFILE'
	echo '  INFILE is a jitconv logfile'
	echo '  OUTFILE is a summary file for spreadsheet'
else
	infile="$1"
	outfile="$2"
	sed '/.*combined/,/.*Legend/!d;//d;/^$/d' "${infile}" | \
		gawk -f 2csv.awk - \
		>& "${outfile}"
fi
