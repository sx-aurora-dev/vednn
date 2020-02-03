cmake_minimum_required(VERSION 3.0)

message(STATUS "src/intrinsic CMAKE_C_FLAGS     ${CMAKE_C_FLAGS}")
set(CMAKE_C_COMPILER ${LLVM_INSTALL_PREFIX}/bin/clang)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -target ve -O3 -mllvm -show-spill-message-vec -fno-vectorize -fno-unroll-loops -fno-slp-vectorize -fno-crash-diagnostics")

set(CMAKE_CXX_COMPILER ${LLVM_INSTALL_PREFIX}/bin/clang++)
set(CMAKE_CXX_FLAGS ${CMAKE_C_FLAGS})

if(USE_OPENMP)
    add_definitions(-DVEDNN_USE_OPENMP)
    add_definitions(-DVEDNN_OMP_GROUP_PARALLEL)
endif()

# clang does not accept -ftrace as compiler flag

if(BUILD_SHARED_LIB)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
endif()

message(STATUS "src/intrinsic CMAKE_C_FLAGS --> ${CMAKE_C_FLAGS}")

include_directories("../")
# include_directories("../C/vconv/include/gen-dnn/vgemm")

add_subdirectory(Linear/Forward)
add_subdirectory(Linear/BackwardData)
add_subdirectory(Linear/BackwardWeight)
add_subdirectory(Convolution/Forward)
add_subdirectory(Convolution/BackwardData)
add_subdirectory(Convolution/BackwardFilter)
add_subdirectory(MaxPooling/Backward)
add_subdirectory(MaxPooling/Forward)
add_subdirectory(Activation)
# vim: et ts=4 sw=4 ai
