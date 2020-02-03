cmake_minimum_required(VERSION 3.0)

add_library(vednn_intrinsic_linear_backward_weight OBJECT
    default.cc
    o2X_woaligned.cc
    o2XU128_woaligned.cc
)
# vim: sw=4 ts=4 et ai
