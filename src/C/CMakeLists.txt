cmake_minimum_required(VERSION 3.0)

set(CMAKE_C_COMPILER /opt/nec/ve/bin/ncc)
set(CMAKE_CXX_COMPILER /opt/nec/ve/bin/nc++)
if(DEFINED ENV{NCC})
    set(CMAKE_C_COMPILER ENV{NCC})
endif()
if(DEFINED ENV{NCXX})
    set(CMAKE_C_COMPILER ENV{NCXX})
endif()
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O4 -DNDEBUG -pthread")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O4 -finline -finline-functions -DNDEBUG -pthread")

if(USE_FTRACE AND NOT USE_FTRACE STREQUAL 1)
    message(STATUS " ??? adding ftrace compilation flag ??? ")
    set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS}   -ftrace")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ftrace")
endif()

if(USE_OPENMP)
    add_definitions(-DVEDNN_USE_OPENMP)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
endif()

if(BUILD_SHARED_LIB)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
endif()

message(STATUS "src/C/ CMAKE_C_FLAGS=${CMAKE_C_FLAGS}")
message(STATUS "src/C/ CMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}")

include_directories("../")
include_directories("vconv/include/gen-dnn/vgemm")
include_directories("vconv/include/gen-dnn/vconv")

add_library(vednn_c_code OBJECT
	vednnConvolutionForward.c
	vednnConvolutionForwardAddBias.c
    vednnConvolutionForward_dil1_str1_padsame_ker3_T.cpp
    vednnConvolutionForward_dil1_str1_pad0_ker1_T.cpp
    vednnConvolutionForward_owU128_T.cpp
	vednnConvolutionBackwardData.c
	vednnConvolutionBackwardFilter.c
    convolution_gemm.cpp
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
    vednnInitScratchpad.cpp
    ve_fastdiv.c
)

add_subdirectory(vconv)

# libvednnx iterator exposes low-level calls (with different parameter ordering)
# Some external projects may want to directly access the low-level libvednn
# impls, with their slightly different parameter ordering... for now.
# Normally the user will never include these (even indirectly).
# As libvednnx.h matures it may be possible to remove these headers.
install(FILES vednnConvolutionForward.h
    vednnConvolutionForwardAddBias.h
    vednnConvolutionBackwardData.h
    vednnConvolutionBackwardFilter.h
    ve_fastdiv.h
    vednn-def.h
    DESTINATION include/C)
# vim: et ts=4 sw=4 ai
