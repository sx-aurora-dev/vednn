cmake_minimum_required(VERSION 3.0)

add_library(vednn_intrinsic_activation OBJECT
    Forward_Relu.c
    Backward_Relu.c
)
# vim: sw=4 ts=4 et ai
