#!/bin/bash
run=""
tests=0
txt=""
function iszmk {
#for ih in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 50 60 70 80 90 100 110 120 128 130 140 150 160 170 180 190 200 210 230 240 250 256 260 270 280 290 300 320 340 360 380 400
#for ih in 1 2 3 4 5 6 7 8 15 16 17 20 25 32 40 50 60 80 100 120 128 130 150 200 240 256 299 300
for ih in 1 2 3 4 5 6 7 8 16 32 64 128 192 256
#for ih in 31 32
do
    a="mb${1}ic${2}ih${ih}oc${3}_kh${4}ph${5}_"
    if [ ! -z "${6}" -a "${6}" != "1" ]; then a="${a}sh${6}"; fi
    # mkl-dnn convention -- "every pixel" is dilation zero
    if [ ! -z "${7}" -a "${7}" != "0" ]; then a="${a}dh${7}"; fi
    tests=$((tests + 1))
    a="${a}_n\"iszmk${tests}\""
    #echo "${a}"
    txt="${txt}${a}\n"
done
}
if [ 1 -eq 1 ]; then for kp in '1 0' '3 0' '5 0' '3 2' '5 2' '4 0' '4 1'; do
    iszmk 1 31 31 ${kp} 1 0
    iszmk 1 31 31 ${kp} 2 0
    iszmk 1 31 31 ${kp} 1 1
    iszmk 1 31 31 ${kp} 2 1
done; fi
if [ 1 -eq 1 ]; then for kp in '1 0' '3 0' '5 0' '3 2' '5 2' '4 0' '4 1'; do
    for ic in 1 2 3 4 8 16 64; do
        iszmk 1 ${ic} 31 ${kp} 1 0
    done
    for ic in 64; do
        iszmk 1 ${ic} 512 ${kp} 1 0
    done
done; fi
if [ 1 -eq 1 ]; then for kp in '1 0' '3 0' '5 0' '3 2' '5 2' '4 0' '4 1'; do
    for ic in 16; do
        iszmk 1 ${ic}  64 ${kp} 1 0
        iszmk 1 ${ic} 256 ${kp} 1 0
        iszmk 1 ${ic} 512 ${kp} 1 0
    done
done; fi
if [ 1 -eq 1 ]; then for kp in '1 0' '3 0' '5 0' '7 0' '3 2' '5 2' '7 3' '4 0' '4 1'; do
    for ic in 16; do
        iszmk 1 ${ic} 256 ${kp} 2 0
    done
done; fi
echo "${tests}"
echo -e "${txt}"
# k1
# k5
# p0
# ic31, for oc in 1 16 64 256; do
