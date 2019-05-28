cmake_minimum_required(VERSION 3.0)

set(CMAKE_C_COMPILER ${NCC})
set(CMAKE_C_FLAGS "-O4")

if(USE_OPENMP)
	add_definitions(-DVEDNN_USE_OPENMP)
	set(CMAKE_C_FLAGS "${CMAKE_CFLAGS} -fopenmp")
endif()

if(BUILD_SHARED_LIB)
	set(CMAKE_C_FLAGS "${CMAKE_CFLAGS} -fPIC")
endif()

include_directories("../")

add_library(vednn_c_code OBJECT
	vednnConvolutionForward.c
	vednnConvolutionForwardAddBias.c
	vednnConvolutionBackwardData.c
	vednnConvolutionBackwardFilter.c
	vednnLinearForward.c
	vednnLinearBackwardData.c
	vednnLinearBackwardWeight.c
	vednnActivationForward.c
	vednnActivationBackward.c
	vednnMaxPoolingForward.c
	vednnMaxPoolingForward_default.c
	vednnMaxPoolingBackward.c
	vednnMaxPoolingBackward_default.c
	vednnSoftmaxForward.c
	vednnInit.c
)
