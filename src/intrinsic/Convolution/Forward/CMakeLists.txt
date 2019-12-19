cmake_minimum_required(VERSION 3.0)

add_library(vednn_intrinsic_convolution_forward OBJECT
    direct_default.cc
    direct_vecC.cc
    direct_vecC_dil1_str1_pad1_ker3.cc
    direct_dil1_pad0.cc
    direct_dil1_pad0_owU128.cc
    direct_dil1_pad0_ker1.cc
    direct_dil1_pad0_owU128_ker1.cc
    direct_dil1_str1_pad0.cc
    direct_dil1_str1_pad0_ker1.cc
    direct_dil1_str1_pad0_ker3_iw2XU256_ow2X_ioaligned.cc
    direct_dil1_str1_pad0_ker4_iwU256.cc
    direct_dil1_str1_pad0_owU128.cc
    direct_dil1_str1_padsame.cc
    direct_dil1_str1_padsame_ker3.cc
    direct_dil1_str1_padsame_ker3_c1.cc
    direct_dil1_str1_padsame_ker3_c1_owU128.cc
    direct_dil1_str1_padsame_ker5.cc
    direct_dil1_str1_padsame_ker5_owU128.cc
    direct_dil1_str1_padsame_ker2.cc
    direct_dil1_str2_pad1_ker3_owU128.cc
    direct_dil1_str2_pad1_ker4_owU128.cc
    direct_owU128.cc
)
target_compile_options(vednn_intrinsic_convolution_forward PRIVATE "-DTMP20")
# vim: sw=4 ts=4 et ai
