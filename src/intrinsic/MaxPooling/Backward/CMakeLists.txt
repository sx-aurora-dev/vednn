cmake_minimum_required(VERSION 3.0)

add_library(vednn_intrinsic_maxpooling_backward OBJECT
    regular.c
    regular_owU128.c
    regular_ww2X_owU128_ialigned.c
)
# vim: sw=4 ts=4 et ai
