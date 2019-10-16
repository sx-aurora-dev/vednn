cmake_minimum_required(VERSION 3.0)

#set(CMAKE_C_COMPILER /opt/nec/ve/bin/ncc)
#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O4")
if(USE_FTRACE AND NOT USE_FTRACE STREQUAL 1)
    message(STATUS " ??? adding ftrace compilation flag ??? ")
    set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS}   -ftrace")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ftrace")
endif()

if(USE_OPENMP)
    add_definitions(-DVEDNN_USE_OPENMP)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp")
endif()

if(BUILD_SHARED_LIB)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC")
endif()

message(STATUS "src/wrap/ CMAKE_C_FLAGS=${CMAKE_C_FLAGS}")

include_directories(".." "../C")

set(CMAKE_CFLAGS     "${CMAKE_CFLAGS} -I${NLCDIR}/include")
set(CMAKE_CXXFLAGS "${CMAKE_CXXFLAGS} -I${NLCDIR}/include")
#set(CMAKE_BUILD_RPATH "${CMAKE_BUILD_RPATH} ${NLCDIR}/lib")
#set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_RPATH} ${NLCDIR}/lib")
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,-rpath,${NLCDIR}/lib")

add_library(vednn_wrap OBJECT
    vednnConvolutionLists.c
    vednnConvolution_ok.c
    )
target_compile_options(vednn_wrap PRIVATE
    $<$<CONFIG:Release>:-O4 -DNDEBUG>
    $<$<CONFIG:RelWithDebInfo>:-g2 -O4 -DNDEBUG>
    $<$<CONFIG:Debug>:-g2 -O0 -DVERBOSE>
    )


add_subdirectory(vconv)

install(FILES vednnx.h DESTINATION include)
# TODO modify vednnx.h public API to pull in the wrap/ headers so all
#      external projects may need to call 'ok' functions and access
#      the invokers and re-order params correctly, so we still need
#      to provide [at least some] of the wrap/ headers
#      (Goal:  expose JUST the iterator API,
#              like vednnConvForward_Begin/Next etc,
#              and add wrappers for parm-reordering that take an opaque
#              impl-info ptr as a first arg)
install(FILES vednnConvolutionLists.h
    vednnConvolution_ok.h
    vednnImpl_args.h
    DESTINATION include/wrap)
# vim: et ts=4 sw=4 ai
