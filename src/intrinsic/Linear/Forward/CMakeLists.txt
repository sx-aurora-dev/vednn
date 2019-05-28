cmake_minimum_required(VERSION 3.0)

add_library(vednn_intrinsic_linear_forward OBJECT
	default.c
	o2X_woaligned.c
	oU32.c
)
