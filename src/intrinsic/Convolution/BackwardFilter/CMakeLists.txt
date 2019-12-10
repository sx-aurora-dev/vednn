cmake_minimum_required(VERSION 3.0)

add_library(vednn_intrinsic_convolution_backward_filter OBJECT
    direct_default.cc
    direct_dil1_pad0.cc
    direct_dil1_pad0_owU32.cc
    direct_dil1_pad0_ker1.cc
    direct_dil1_pad0_ker1_owU32.cc
    direct_dil1_pad0_ker1_ohwU64.cc
    direct_dil1_pad0_ker1_ohwU128.cc
    direct_dil1_pad0_ker3_owU128.cc
    direct_dil1_str1_pad0_ker3_owU128.cc
    direct_dil1_str1_pad0_ker3_ow2X_iw2XU256_igoaligned.cc
    direct_dil1_str1_pad0_ker4_owU128.cc
    direct_dil1_str1_padsame.cc
    direct_dil1_str1_padsame_ker1.cc
    direct_dil1_str1_padsame_ker3.cc
    direct_dil1_str1_padsame_ker3_owU128.cc
    direct_dil1_str1_padsame_ker3_ohwU256.cc
    direct_dil1_str1_padsame_ker5.cc
    direct_dil1_str1_padsame_ker5_owU128.cc
    direct_dil1_str1_padsame_ker2.cc
    direct_dil1_str1_padsame_ker2_owU128.cc
    direct_dil1_str2_pad1_ker3_owU128.cc
	direct_ker3_owU128.cc
	direct_owU128.cc
	direct_vecC.cc
)
# vim: sw=4 ts=4 et ai
