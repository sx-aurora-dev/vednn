cmake_minimum_required(VERSION 3.0)

set(CMAKE_C_COMPILER ${NCC})
set(CMAKE_CXX_COMPILER ${NCXX})

if(USE_OPENMP)
	set(LIBNAME vednn_openmp)
	set(LIBWRAP vednnx_openmp)
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -fopenmp")
else()
	set(LIBNAME vednn_sequential)
	set(LIBWRAP vednnx_sequential)
endif()

if(BUILD_SHARED)
	set(LIB_MODE SHARED)
else()
	set(LIB_MODE STATIC)
endif()

# for vconv or convFwd-gemm
set(CMAKE_CFLAGS     "${CMAKE_CFLAGS} -I${NLCDIR}/include")
set(CMAKE_CXXFLAGS "${CMAKE_CXXFLAGS} -I${NLCDIR}/include")
#set(CMAKE_BUILD_RPATH "${CMAKE_BUILD_RPATH} ${NLCDIR}/lib")
#set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_RPATH} ${NLCDIR}/lib")
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,-rpath,${NLCDIR}/lib")

message(STATUS "src/ CMAKE_C_FLAGS=${CMAKE_C_FLAGS}")

add_library(${LIBNAME} ${LIB_MODE}
    $<TARGET_OBJECTS:vednn_c_code>
    $<TARGET_OBJECTS:vednn_gemm_convolution>
    $<TARGET_OBJECTS:vednn_intrinsic_convolution_forward>
    $<TARGET_OBJECTS:vednn_intrinsic_convolution_backward_data>
    $<TARGET_OBJECTS:vednn_intrinsic_convolution_backward_filter>
    $<TARGET_OBJECTS:vednn_intrinsic_linear_forward>
    $<TARGET_OBJECTS:vednn_intrinsic_linear_backward_data>
    $<TARGET_OBJECTS:vednn_intrinsic_linear_backward_weight>        
    $<TARGET_OBJECTS:vednn_intrinsic_maxpooling_backward>
    $<TARGET_OBJECTS:vednn_intrinsic_maxpooling_forward>
    $<TARGET_OBJECTS:vednn_intrinsic_activation>
    )
if(BUILD_SHARED)
    set_target_properties(${LIBNAME} PROPERTIES
        INSTALL_RPATH "\$ORIGIN/../lib"
        )
endif()

# vednnx extensions: iterator and jit api, gen-dnn subset
add_library(${LIBWRAP} ${LIB_MODE}
    $<TARGET_OBJECTS:vednn_wrap>
    $<TARGET_OBJECTS:vconv_code>
    )

if(USE_OPENMP)
    #target_link_libraries(${LIBNAME} ${NLCDIR}/libcblas ${NLCDIR}/libblas_openmp.so)
    target_link_libraries(${LIBNAME} -L${NLCDIR}/lib -lcblas -lblas_openmp)
    target_link_libraries(${LIBWRAP} -L${NLCDIR}/lib -lcblas -lblas_openmp)
else()
    target_link_libraries(${LIBNAME} ${NLCDIR}/lib/libcblas.so ${NLCDIR}/lib/libblas_sequential.so)
    target_link_libraries(${LIBWRAP} ${NLCDIR}/lib/libcblas.so ${NLCDIR}/lib/libblas_sequential.so)
endif()

#if(USE_FTRACE)
#    # ftrace/veperf location
#    find_file(found_VEPERF_H NAME veperf.h
#        NO_DEFAULT_PATH
#        # /opt/nec/ve default path ?
#        PATHS /usr/uhome/aurora/mpc/pub/veperf/latest
#        PATH_SUFFIXES include
#        )
#    message(STATUS "veperf.h --> ${found_VEPERF_H}")
#    if(NOT found_VEPERF_H)
#        message(WARNING "ve.cmake: veperf.h not found (looking for veperf.h [ftrace.h might be there too])")
#    else()
#        get_filename_component(VE_VEPERF_INCLUDE "${found_VEPERF_H}" DIRECTORY)
#        get_filename_component(VE_VEPERF_DIR "${VE_VEPERF_INCLUDE}" DIRECTORY)
#        set(VE_VEPERF_DIR "${VE_VEPERF_DIR}" CACHE PATH "Aurora ftrace/veperf root directory" FORCE)
#        set(VE_VEPERF_INCLUDE "${VE_VEPERF_INCLUDE}" CACHE PATH "Aurora ftrace/veperf include path" FORCE)
#        set(VE_VEPERF_INCFLAGS "-I${VE_VEPERF_INCLUDE}" CACHE STRING "Aurora ftrace/veperf include path" FORCE)
#        set(VE_VEPERF_LDFLAGS "-L${VE_VEPERF_DIR}/lib" CACHE STRING "Aurora ftrace/veperf C/CXX compile/link options" FORCE)
#        set(VE_VEPERF_LIBS "-lveperf") # NOTE: there are static and shared libs
#        mark_as_advanced(VE_VEPERF_DIR VE_VEPERF_INCLUDE VE_VEPERF_LDFLAGS VE_VEPERF_LIBS)
#        # Q: What is difference between   libveperf and libveperf_sp
#        # TODO: add library existence and compilation checks
#        message(STATUS "veperf.h                         : ${found_VEPERF_H}")
#        message(STATUS "VE_VEPERF_INCLUDE                : ${VE_VEPERF_INCLUDE}")
#        message(STATUS "VE_VEPERF_DIR [ftrace|veperf]    : ${VE_VEPERF_DIR}")
#        message(STATUS "VE_VEPERF_INCFLAGS               : ${VE_VEPERF_INCFLAGS}")
#        message(STATUS "VE_VEPERF_LDFLAGS                : ${VE_VEPERF_LDFLAGS}")
#        message(STATUS "VE_VEPERF_LIBS                   : ${VE_VEPERF_LIBS}")
#        # Expected libraries: libveperf.{a|so} headers: ftrace.h veperf.h
#        # add LDFLAGS="-lveperf" (nothing for ftrace)
#        # VE_VEPERF_DIR for ftrace/veperf
#        set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS} -DFTRACE=${USE_FTRACE} -I${VE_VEPERF_INCLUDE}")
#        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DFTRACE=${USE_FTRACE} -I${VE_VEPERF_INCLUDE}")
#        set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,-rpath,${VEPERF_LIB_DIR}")
#    endif()
#    find_library(veperf NAMES libveperf.a PATHS ${VE_VEPERF_DIR}/lib NO_DEFAULT_PATH)
#endif()

#if(USE_FTRACE)
#    # does not produce ftrace output of libvednn internals:
#    #target_link_libraries(${LIBNAME} -lveftrace)
#    #target_link_libraries(${LIBWRAP} ${LBNAME} -lveftrace)
#    target_link_libraries(${LIBNAME} ${veperf})
#endif()

install(TARGETS ${LIBNAME} ${LIBWRAP} ARCHIVE DESTINATION lib LIBRARY DESTINATION lib)
install(FILES
    vednn.h vednn_util.hpp
    DESTINATION include)
# vim: sw=4 ts=4 et ai
