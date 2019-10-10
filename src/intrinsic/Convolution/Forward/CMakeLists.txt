cmake_minimum_required(VERSION 3.0)

add_library(vednn_intrinsic_convolution_forward OBJECT
	direct_default.c
	direct_vecC.c
	direct_dil1_pad0.c
    direct_dil1_pad0_owU128.c
    direct_dil1_pad0_ker1.c
    direct_dil1_pad0_owU128_ker1.c
	direct_dil1_str1_pad0.c
	direct_dil1_str1_pad0_ker1.c
	direct_dil1_str1_pad0_ker3_iw2XU256_ow2X_ioaligned.c
	direct_dil1_str1_pad0_owU128.c
	direct_dil1_str1_padsame.c
	direct_dil1_str1_padsame_ker3.c
	direct_dil1_str1_padsame_ker3_T_subkernel.c
	direct_dil1_str1_padsame_ker3_c1.c
	direct_dil1_str1_padsame_ker3_c1024x.c
	direct_dil1_str1_padsame_ker3_c1_owU128.c
	direct_dil1_str1_padsame_ker5.c
	direct_dil1_str1_padsame_ker5_owU128.c
	direct_dil1_str1_padsame_ker2.c
	direct_owU128.c
)
