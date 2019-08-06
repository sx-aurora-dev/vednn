cmake_minimum_required(VERSION 3.0)

add_library(vednn_intrinsic_convolution_backward_data OBJECT
	direct_default.c
	direct_vecC.c
	direct_dil1_str1.c
	direct_dil1_str1_iwU128.c
	direct_dil1_str1_pad0_ker3_iw2XU256_ow2X_ioaligned.c
	direct_dil1_str1_pad0_ker3_iw2XU32_ow2X_ioaligned.c
	direct_dil1_str1_pad0_ker3_iwU128.c
	direct_dil1_str1_padsame.c
	direct_dil1_str1_padsame_ker3.c
	direct_dil1_str1_padsame_ker5.c
	direct_dil1_str1_padsame_ker2.c
	direct_dil1_str1_padsame_ker1.c
	direct_dil1_pad0_ker1_owU128.c
	direct_ker3_iwU128.c
	direct_ker5_iwU128.c
	direct_iwU128.c
	direct_ker5.c
)
