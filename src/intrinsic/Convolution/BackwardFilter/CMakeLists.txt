cmake_minimum_required(VERSION 3.0)

add_library(vednn_intrinsic_convolution_backward_filter OBJECT
	direct_default.c
	direct_dil1_pad0.cc
    direct_dil1_pad0_owU32.cc
	direct_dil1_pad0_ker1.c
	direct_dil1_pad0_ker1_owU32.c
    direct_dil1_pad0_ker1_ohwU64.c
	direct_dil1_pad0_ker1_ohwU128.c
	direct_dil1_pad0_ker3_owU128.c
	direct_dil1_str1_pad0_ker3_owU128.c
	direct_dil1_str1_pad0_ker3_ow2X_iw2XU256_igoaligned.c
	direct_dil1_str1_padsame.c
	direct_dil1_str1_padsame_ker1.c
	direct_dil1_str1_padsame_ker3.c
	direct_dil1_str1_padsame_ker3_owU128.c
	direct_dil1_str1_padsame_ker3_ohwU256.c
	direct_dil1_str1_padsame_ker5.c
	direct_dil1_str1_padsame_ker5_owU128.c
	direct_dil1_str1_padsame_ker2.c
    direct_dil1_str1_padsame_ker2_owU128.c
	direct_owU128.c
	direct_vecC.c
)
