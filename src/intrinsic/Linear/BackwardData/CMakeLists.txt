cmake_minimum_required(VERSION 3.0)

message(STATUS "intrinsic/Linear/BackwardData CMAKE_C_COMPILER = ${CMAKE_C_COMPILER}")

add_library(vednn_intrinsic_linear_backward_data OBJECT
    default.cc
    o2X_woaligned.cc
    oU256.cc
    oU128.cc
    o2XU128_waligned.cc
)
# vim: sw=4 ts=4 et ai
