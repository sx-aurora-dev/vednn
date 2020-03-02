cmake_minimum_required(VERSION 3.0)

# Isn't this propagated to us?
set(CMAKE_C_COMPILER ${NCC})
set(CMAKE_CXX_COMPILER ${NCXX})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DXX_BEG")
# Need some targets with different compile flags
#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O4 -DNDEBUG")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O4 -finline -finline-functions -DNDEBUG")

if(USE_FTRACE AND USE_FTRACE STREQUAL 2)
    #message(STATUS " ??? adding ftrace compilation flag ??? check if -pthread is for link/compile")
    set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS} -ftrace -pthread")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ftrace -pthread")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -ftrace -pthread -DSRC_C_LF1")
endif()

if(USE_OPENMP)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp -pthread")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -pthread")
    add_definitions(-DVEDNN_USE_OPENMP)
    add_definitions(-DVEDNN_OMP_GROUP_PARALLEL)
endif()

option(USE_ALTERNATE_PARALLELISM   "Use Alternate Parallelism" ON)
if(USE_ALTERNATE_PARALLELISM)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DVEDNN_ALT_PARALLEL")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DVEDNN_ALT_PARALLEL")
endif()
    

if(BUILD_SHARED_LIB)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DXX_END")
message(STATUS "src/C/ CMAKE_C_FLAGS=${CMAKE_C_FLAGS}")
message(STATUS "src/C/ CMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}")

set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS}   -DCMK_C")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DCMK_C")
message(STATUS "src/C/ CMAKE_C_FLAGS=${CMAKE_C_FLAGS}")
message(STATUS "src/C/ CMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}")

include_directories("../" "gen-dnn")

add_library(vednn_c_code OBJECT
	vednnConvolutionForward.c
    vednnConvolutionForward_dil1_str1_padsame_ker3_T.cpp
    vednnConvolutionForward_dil1_str1_pad0_ker1_T.cpp
    vednnConvolutionForward_owU128_T.cpp
    vednnConvolutionForward_dil1_str1_padsame_ker3_c1024x_T.cpp
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
    vednnConvolutionGemm.cpp
    )
set(SRC_C_OPT4 -O4) # original default setting
set(SRC_C_OPT3 -O3 -finline -finline-functions -fcse-after-vectorization -msched-block)
target_compile_options(vednn_c_code PRIVATE
    $<$<CONFIG:Release>:${SRC_C_OPT4} -DTMP00>
    $<$<CONFIG:RelWithDebInfo>:${SRC_C_OPT4} -DTMP01>
    $<$<CONFIG:Debug>:-O3 -DTMP02>
    )
# having some trouble with SCRATCHPAD=1 (reduce malloc calls)
# -fcse-after-vectorization -msched-block important for convolution_gemm speed:
target_compile_options(vednn_gemm_convolution PRIVATE
    $<$<CONFIG:Release>:${SRC_C_OPT3} -DVERBOSE=0 -DTMP03>
    $<$<CONFIG:RelWithDebInfo>:${SRC_C_OPT3} -DVERBOSE=0 -DTMP04>
    $<$<CONFIG:Debug>:-O0 -DVERBOSE=1 -DTMP05>
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
    ve_fastdiv.h
    vednn-def.h
    vednn-def.hpp
    DESTINATION include/C)
# vim: et ts=4 sw=4 ai
