cmake_minimum_required(VERSION 3.0)

add_library(vednn_intrinsic_linear_backward_data OBJECT
	default.c
	o2X_woaligned.c
    oU256.c
	oU128.c
	o2XU128_waligned.c
)
