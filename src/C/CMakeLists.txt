cmake_minimum_required(VERSION 3.0)

#set(CMAKE_C_COMPILER ${NCC})
#set(CMAKE_CXX_COMPILER ${NCXX})
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O4 -DNDEBUG")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O4 -finline -finline-functions -DNDEBUG")

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

include_directories("../" "gen-dnn")

add_library(vednn_c_code OBJECT
	vednnConvolutionForward.c
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
    ve_fastdiv.c
)
add_library(vednn_gemm_convolution OBJECT
    vednnInitScratchpad.cpp
    convolution_gemm.cpp
    )
#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O4 -DNDEBUG")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O4 -finline -finline-functions -DNDEBUG")
target_compile_options(vednn_c_code PRIVATE
    $<$<CONFIG:Release>:-O4 -DNDEBUG>
    $<$<CONFIG:ReleaseWithDebInfo>:-g -O4 -DNDEBUG>
    $<$<CONFIG:Debug>:-g2 -O3>
    )
target_compile_options(vednn_gemm_convolution PRIVATE
    -DVERBOSE=1
    $<$<CONFIG:Release>:-O4 -DNDEBUG>
    $<$<CONFIG:ReleaseWithDebInfo>:-g -O4 -DNDEBUG>
    $<$<CONFIG:Debug>:-g2 -O0>
    )


# libvednnx iterator exposes low-level calls (with different parameter ordering)
# Some external projects may want to directly access the low-level libvednn
# impls, with their slightly different parameter ordering... for now.
# Normally the user will never include these (even indirectly).
# As libvednnx.h matures it may be possible to remove these headers.
install(FILES
    vednn-def.h
    ve_fastdiv.h
    vednnConvolutionForward.h
    #vednnConvolutionForwardAddBias.h
    vednnConvolutionBackwardData.h
    vednnConvolutionBackwardFilter.h
    DESTINATION include/C)
# vim: et ts=4 sw=4 ai
