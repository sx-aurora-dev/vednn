cmake_minimum_required(VERSION 3.0)

add_library(vednn_intrinsic_convolution_backward_data OBJECT
    direct_default.cc
    direct_vecC.cc
    direct_dil1_str1.cc
    direct_dil1_str1_iwU128.cc
    direct_dil1_str1_pad0_ker3_iw2XU256_ow2X_ioaligned.cc
    direct_dil1_str1_pad0_ker3_iw2XU32_ow2X_ioaligned.cc
    direct_dil1_str1_pad0_ker3_iwU128.cc
    direct_dil1_str1_pad0_ker4_iwU128.cc
    direct_dil1_str1_padsame.cc
    direct_dil1_str1_padsame_ker3.cc
    direct_dil1_str1_padsame_ker5.cc
    direct_dil1_str1_padsame_ker2.cc
    direct_dil1_str1_padsame_ker1.cc
    direct_dil1_pad0_ker1_owU128.cc
    direct_ker3_iwU128.cc
    direct_ker5_iwU128.cc
    direct_iwU128.cc
    direct_ker5.cc
    direct_dil1_str2_ker3_iwU256.cc
    direct_dil1_str2_pad1_ker3_iwU256.cc
    direct_dil1_str2_pad1_ker4_iwU256.cc
    direct_dil1_str2_pad1_ker4_iw2xU256.cc
    direct_dil1_str2_pad2_ker5.cc
    direct_dil1_str2_pad2_ker5_iwU128.cc
)
# vim: sw=4 ts=4 et ai
